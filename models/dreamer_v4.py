"""Paper-faithful Dreamer 4 implementation, adapted to vector-state APC.

Reference: Hafner, Yan, Lillicrap (2025), "Training Agents Inside of Scalable
World Models" — arXiv:2509.24527.

Architecture
------------
Two components, both built on the same efficient transformer trunk:

1. **Causal Tokenizer** (``Tokenizer``): encoder + low-D linear+tanh
   bottleneck + decoder. For the original paper's video setting it
   processes image patches; for APC we feed the per-step observation
   vector directly. Trained with MAE-style channel dropout + MSE
   reconstruction (LPIPS dropped — does not apply to scalar features).

2. **Interactive Dynamics** (``DynamicsTransformer``): block-causal-in-time
   transformer that operates on the interleaved sequence

       [ register_1, …, register_S_r,
         action_token,
         (τ, d)_token,
         observation_token z̃ ]   per timestep

   It is trained with the **shortcut forcing** objective
   (paper §3.2, eq. 7) using x-prediction. At inference time the model
   denoises observations via K=4 forward passes per timestep.

Heads
-----
- ``policy``  : per-action-dim categorical over uniform bins in [-1, 1].
- ``reward``  : symexp-twohot (255 bins on [-20, 20]).
- ``value``   : symexp-twohot (255 bins on [-20, 20]).
- ``target_value`` : EMA copy of ``value`` for TD-λ bootstrap stability.

The per-action-dim categorical and symexp-twohot heads carry over from
Dreamer 3 (paper still uses these in V4, see §3.3 "Behavior cloning and
reward model"). The crucial change vs. V3 is that the *world model* is
no longer an RSSM — it is the transformer + shortcut-forcing pair.

Three-phase training (paper Algorithm 1)
----------------------------------------
- Phase 1 (pretrain world model)  : tokenizer recon (eq. 5) + dynamics
                                    shortcut forcing (eq. 7)
- Phase 2 (agent finetune)        : add policy + reward MTP heads (eq. 9),
                                    keep eq. 5 + eq. 7 live
- Phase 3 (imagination training)  : freeze tokenizer + dynamics + reward,
                                    train policy via PMPO (eq. 11) and
                                    value via TD-λ (eq. 10) on imagined
                                    rollouts (one rollout per dataset
                                    context). Transformer frozen.

This module supplies the building blocks; the trainer in
``training/train.py`` orchestrates the three phases.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Symlog / symexp / twohot (carried over from V3 — paper §B)
# ---------------------------------------------------------------------------

def symlog(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.log1p(torch.abs(x))


def symexp(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1.0)


def twohot_encode(values: torch.Tensor, bin_edges: torch.Tensor) -> torch.Tensor:
    """Encode scalar targets into a two-hot distribution over ``bin_edges``."""
    values = values.unsqueeze(-1)
    n_bins = bin_edges.shape[0]
    bins = bin_edges.view(*([1] * (values.dim() - 1)), n_bins)
    diff = values - bins
    right = (diff <= 0).float().cumsum(-1)
    right = (right == 1).float().argmax(-1).clamp_(1, n_bins - 1)
    left = (right - 1).clamp_min_(0)
    bl = bin_edges[left]
    br = bin_edges[right]
    span = (br - bl).clamp_min_(1e-8)
    w_right = ((values.squeeze(-1) - bl) / span).clamp_(0.0, 1.0)
    w_left = 1.0 - w_right
    out = torch.zeros(*values.shape[:-1], n_bins, device=values.device,
                      dtype=values.dtype)
    out.scatter_(-1, left.unsqueeze(-1), w_left.unsqueeze(-1))
    out.scatter_add_(-1, right.unsqueeze(-1), w_right.unsqueeze(-1))
    return out


# ---------------------------------------------------------------------------
# Building blocks: RMSNorm, SwiGLU, RoPE, attention with QKNorm + soft-cap
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * rms).to(x.dtype) * self.weight


class SwiGLU(nn.Module):
    """Standard SwiGLU MLP block (paper §3.4)."""

    def __init__(self, dim: int, ff_mult: int = 4):
        super().__init__()
        ff = ff_mult * dim
        self.w1 = nn.Linear(dim, ff, bias=False)
        self.w2 = nn.Linear(dim, ff, bias=False)
        self.w3 = nn.Linear(ff, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


def _rope_cache(seq_len: int, dim: int, device: torch.device,
                base: float = 10_000.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute (cos, sin) for RoPE over ``seq_len`` time positions.

    ``dim`` must be even (per-head dim). Returns shape ``(seq_len, dim)``.
    """
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=device).float() / dim))
    t = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.einsum('i,j->ij', t, inv_freq)        # (seq_len, dim/2)
    emb = torch.cat([freqs, freqs], dim=-1)             # (seq_len, dim)
    return emb.cos(), emb.sin()


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
                ) -> torch.Tensor:
    """Apply RoPE to ``x`` of shape ``(B, n_heads, L, head_dim)``.

    ``cos`` and ``sin`` are of shape ``(L, head_dim)``.
    """
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    x1, x2 = x.chunk(2, dim=-1)
    rot = torch.cat([-x2, x1], dim=-1)
    return (x * cos) + (rot * sin)


class CausalAttention(nn.Module):
    """Causal multi-head attention with QKNorm + attention logit soft-cap.

    Paper §3.4: pre-RMSNorm transformer, RoPE on **time positions**, QKNorm
    and attention soft-cap for stability. We use a precomputed block-causal
    mask supplied by the caller via the ``attn_mask`` argument so that the
    same module can serve both intra-step full attention and inter-step
    causal attention in a single 1-D sequence.

    ``attn_impl`` selects the backend:
      * ``'manual'`` (default if ``soft_cap > 0``) — explicit matmul + softmax
        with logit soft-cap. Paper-faithful but ~2× slower.
      * ``'sdpa'``  — ``F.scaled_dot_product_attention`` (auto-dispatches to
        FlashAttention-2 / cuDNN / mem-efficient). Drops soft-cap; QKNorm
        provides the main numerical safety net. Set via env
        ``DREAMER_FAST_ATTN=1`` or constructor arg.
    """

    def __init__(self, dim: int, n_heads: int, soft_cap: float = 50.0,
                 attn_impl: str = 'auto'):
        super().__init__()
        assert dim % n_heads == 0, f'dim {dim} must be divisible by n_heads {n_heads}'
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        # QKNorm: separate RMSNorm on Q and K (paper §3.4).
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
        self.soft_cap = soft_cap
        # Resolve backend.
        # Default policy (2026-05-12): SDPA whenever a CUDA device is
        # available (FlashAttention-2 / cuDNN — ~3-9× faster than the
        # manual soft-cap path with QKNorm providing the numerical
        # safety net). CPU paths stay on 'manual' (SDPA gains are tiny
        # without GPU kernels and ONNX export still passes
        # attn_impl='manual' explicitly). Override via env
        # DREAMER_FAST_ATTN: '0'/'manual' forces manual, '1'/'sdpa'
        # forces sdpa.
        if attn_impl == 'auto':
            env_fast = os.environ.get('DREAMER_FAST_ATTN', '').strip().lower()
            if env_fast in ('0', 'false', 'manual', 'off'):
                attn_impl = 'manual'
            elif env_fast in ('1', 'true', 'sdpa', 'on'):
                attn_impl = 'sdpa'
            elif torch.cuda.is_available():
                attn_impl = 'sdpa'
            else:
                attn_impl = 'manual'
        assert attn_impl in ('manual', 'sdpa'), f'unknown attn_impl={attn_impl}'
        self.attn_impl = attn_impl

    def forward(self, x: torch.Tensor,
                cos: torch.Tensor, sin: torch.Tensor,
                attn_mask: torch.Tensor) -> torch.Tensor:
        """``x`` (B, L, D); ``cos/sin`` (L, head_dim); ``attn_mask`` (L, L) bool.

        ``attn_mask[i, j] = True`` means token i is allowed to attend to j.
        """
        B, L, D = x.shape
        qkv = self.qkv(x).view(B, L, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)                # each (B, L, H, head_dim)
        q = self.q_norm(q)
        k = self.k_norm(k)
        # Reshape to (B, H, L, head_dim) for attention.
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)

        if self.attn_impl == 'sdpa':
            # Fast path: PyTorch SDPA dispatches to FlashAttention-2 / cuDNN
            # / mem-efficient. Soft-cap is not representable in this kernel
            # — QKNorm carries the numerical-stability load. The boolean
            # ``attn_mask`` semantics here: True == allowed (same as our
            # convention).
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask, is_causal=False,
                dropout_p=0.0,
            )
        else:
            # Manual path with paper soft-cap (kept for fidelity).
            scale = 1.0 / math.sqrt(self.head_dim)
            logits = torch.matmul(q, k.transpose(-2, -1)) * scale
            if self.soft_cap and self.soft_cap > 0:
                logits = self.soft_cap * torch.tanh(logits / self.soft_cap)
            mask = (~attn_mask).to(logits.dtype) * torch.finfo(logits.dtype).min
            logits = logits + mask
            attn = F.softmax(logits.float(), dim=-1).to(v.dtype)
            out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        return self.proj(out)

    # ------------------------------------------------------------------
    # KV-cache fast path (used by imagine_next_z incremental rollout).
    # ------------------------------------------------------------------
    def project_kv(self, x: torch.Tensor, cos: torch.Tensor,
                    sin: torch.Tensor
                    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Project ``x`` to (K, V), apply QKNorm + RoPE, return cache-shape.

        Used to materialize the past-positions K/V once per
        ``imagine_next_z`` call so the K-step inner loop can reuse it.

        ``x``       : (B, L_past, D)
        ``cos/sin`` : (L_past, head_dim) — RoPE for the same positions
        Returns (k, v), each (B, n_heads, L_past, head_dim).
        """
        B, L, D = x.shape
        qkv = self.qkv(x).view(B, L, 3, self.n_heads, self.head_dim)
        _q, k, v = qkv.unbind(dim=2)
        k = self.k_norm(k)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        k = _apply_rope(k, cos, sin)
        return k, v

    def forward_last_only(self, x_last: torch.Tensor,
                            cos_last: torch.Tensor, sin_last: torch.Tensor,
                            k_past: torch.Tensor, v_past: torch.Tensor
                            ) -> torch.Tensor:
        """Attention for the last step's tokens with cached past K/V.

        ``x_last``         : (B, L_last, D) — only the last step's input tokens
        ``cos_last/sin_last``: (L_last, head_dim) — RoPE for last positions
        ``k_past, v_past`` : (B, n_heads, L_past, head_dim) — cached past K/V
                            (already QKNormed + RoPE'd by ``project_kv``)
        Returns (B, L_last, D) — only the last step's attention output.

        Block-causal mask: last-step tokens attend to ALL past tokens
        (always allowed) AND to ALL last-step tokens (intra-step
        bidirectional).  No mask is needed because every (i, j) pair
        is allowed.
        """
        B, L_last, D = x_last.shape
        qkv = self.qkv(x_last).view(B, L_last, 3, self.n_heads, self.head_dim)
        q, k_last, v_last = qkv.unbind(dim=2)
        q = self.q_norm(q)
        k_last = self.k_norm(k_last)
        q = q.transpose(1, 2)
        k_last = k_last.transpose(1, 2)
        v_last = v_last.transpose(1, 2)
        q = _apply_rope(q, cos_last, sin_last)
        k_last = _apply_rope(k_last, cos_last, sin_last)

        # Concatenate past + last along the sequence axis.
        k = torch.cat([k_past, k_last], dim=2)        # (B, H, L_past+L_last, hd)
        v = torch.cat([v_past, v_last], dim=2)

        if self.attn_impl == 'sdpa':
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=None, is_causal=False, dropout_p=0.0,
            )
        else:
            scale = 1.0 / math.sqrt(self.head_dim)
            logits = torch.matmul(q, k.transpose(-2, -1)) * scale
            if self.soft_cap and self.soft_cap > 0:
                logits = self.soft_cap * torch.tanh(logits / self.soft_cap)
            attn = F.softmax(logits.float(), dim=-1).to(v.dtype)
            out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, L_last, D)
        return self.proj(out)


class TransformerBlock(nn.Module):
    """Pre-RMSNorm + causal attention + SwiGLU (paper §3.4)."""

    def __init__(self, dim: int, n_heads: int, ff_mult: int = 4,
                 soft_cap: float = 50.0, attn_impl: str = 'auto'):
        super().__init__()
        self.norm_attn = RMSNorm(dim)
        self.attn = CausalAttention(dim, n_heads, soft_cap=soft_cap,
                                     attn_impl=attn_impl)
        self.norm_ff = RMSNorm(dim)
        self.ff = SwiGLU(dim, ff_mult=ff_mult)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                attn_mask: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm_attn(x), cos, sin, attn_mask)
        x = x + self.ff(self.norm_ff(x))
        return x

    def project_past_kv(self, x_past: torch.Tensor,
                          cos_past: torch.Tensor, sin_past: torch.Tensor
                          ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Cache builder: returns (K, V) for past positions at this layer.

        Note: input is the PAST positions' residual stream BEFORE this
        block.  Caller must norm via this block's ``norm_attn`` first?
        No — we apply norm here to keep the API symmetric with
        ``forward``.
        """
        return self.attn.project_kv(self.norm_attn(x_past), cos_past, sin_past)

    def forward_last_only(self, x_last: torch.Tensor,
                            cos_last: torch.Tensor, sin_last: torch.Tensor,
                            k_past: torch.Tensor, v_past: torch.Tensor
                            ) -> torch.Tensor:
        """Forward only the last step's tokens with cached past K/V.

        ``x_last`` is the LAST step's residual stream entering this block.
        Returns the LAST step's residual stream after this block.
        """
        x_last = x_last + self.attn.forward_last_only(
            self.norm_attn(x_last), cos_last, sin_last, k_past, v_past)
        x_last = x_last + self.ff(self.norm_ff(x_last))
        return x_last


# ---------------------------------------------------------------------------
# Small MLP used by tokenizer encoder/decoder and the heads
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int,
                 n_layers: int = 2, zero_init_last: bool = False):
        super().__init__()
        layers: List[nn.Module] = []
        d = in_dim
        for _ in range(n_layers):
            layers.append(nn.Linear(d, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.SiLU())
            d = hidden_dim
        head = nn.Linear(d, out_dim)
        if zero_init_last:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        layers.append(head)
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Causal Tokenizer (paper §3.1, adapted to vector observations)
# ---------------------------------------------------------------------------

class Tokenizer(nn.Module):
    """Thin vector-observation tokenizer (no compression bottleneck).

    Architectural note (deviation from paper §3.1): the V4 paper's
    tokenizer is designed for image observations, where compressing
    2-D patches to discrete latents is essential and a tanh / VQ
    bottleneck plus MAE reconstruction prevents shortcut learning.
    For our low-D vector obs (n ≈ 10), forcing the encoder through an
    MLP+tanh+recon-MAE pipeline empirically collapses the encoder to a
    near-constant function (the recon MAE is trivially solved by
    memorizing the marginal mean when most channels are constant
    within an episode).  Diagnosed 2026-05-03: pre-tanh per-dim std
    < 4e-3 for all 24 dims while obs varied with std up to 10.

    We therefore use a thin learned **linear projection + LayerNorm**
    as encode (no compression — actually a learned lift from obs_dim
    to z_dim ≥ obs_dim — which preserves all state information by
    construction) and a symmetric linear decode.  Recon loss is kept
    for compat but ``recon_scale=0`` is the default; the dynamics's
    shortcut-forcing loss carries the world-model training.
    """

    def __init__(self, obs_dim: int, hidden_dim: int, z_dim: int,
                 mae_p_max: float = 0.0):
        super().__init__()
        self.obs_dim = obs_dim
        self.z_dim = z_dim
        self.mae_p_max = float(mae_p_max)
        # Thin linear projection + LayerNorm — no MLP, no tanh bottleneck.
        self.encode_proj = nn.Linear(obs_dim, z_dim)
        self.encode_norm = nn.LayerNorm(z_dim)
        self.decode_proj = nn.Linear(z_dim, obs_dim)
        # Learned per-channel mask embedding (broadcast across batch).
        # Kept for forward_with_mae compat; only used when mae_p_max > 0.
        self.mask_embed = nn.Parameter(torch.zeros(obs_dim))

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        """``obs`` of shape ``(..., obs_dim)`` → ``z`` of shape ``(..., z_dim)``.

        Linear lift + LayerNorm.  No compression, no saturating
        nonlinearity.  Empirically this preserves state-dependence
        through the dynamics transformer.
        """
        return self.encode_norm(self.encode_proj(obs))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decode_proj(z)

    def forward_with_mae(self, obs: torch.Tensor
                          ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply MAE channel dropout, encode, decode. Returns ``(z, recon)``.

        Mask probability is sampled per-example.
        """
        if not self.training or self.mae_p_max <= 0.0:
            z = self.encode(obs)
            recon = self.decode(z)
            return z, recon
        # Sample per-example mask probability in [0, p_max].
        shape = obs.shape[:-1]
        p = torch.empty(shape + (1,), device=obs.device, dtype=obs.dtype
                        ).uniform_(0.0, self.mae_p_max)
        mask = (torch.rand_like(obs) < p).to(obs.dtype)
        obs_masked = obs * (1.0 - mask) + self.mask_embed * mask
        z = self.encode(obs_masked)
        recon = self.decode(z)
        return z, recon

    def recon_loss(self, obs: torch.Tensor, recon: torch.Tensor
                    ) -> torch.Tensor:
        """MSE on symlog-encoded observations (paper §B-style robustness)."""
        return F.mse_loss(symlog(recon), symlog(obs))


# ---------------------------------------------------------------------------
# Shortcut forcing utilities (paper §2 + §3.2)
# ---------------------------------------------------------------------------

def sample_tau_d(shape: Tuple[int, ...], k_max: int,
                 device: torch.device, dtype: torch.dtype = torch.float32
                 ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample (τ, d) per the paper's grid (eq. 4).

    ``d ~ 1/U({1, 2, 4, …, k_max})``  → smallest is ``d_min = 1/k_max``.
    ``τ ~ U({0, 1/d, …, 1 − 1/d})`` (noise level, 0 = full noise, 1 = clean).
    Returns float tensors of shape ``shape``.
    """
    # Number of available step sizes: log2(k_max) + 1 (e.g. k_max=4 → {1,2,4}).
    n = int(math.log2(k_max)) + 1
    k_choices = torch.tensor([2 ** i for i in range(n)], device=device,
                             dtype=dtype)
    idx = torch.randint(0, n, shape, device=device)
    k = k_choices[idx]
    d = 1.0 / k                                       # in {1/k_max, …, 1}
    # τ uniform on {0, 1/k, …, (k-1)/k}.
    j = torch.randint(0, 2 ** 30, shape, device=device).to(dtype)
    j = (j % k).floor()
    tau = j / k
    return tau, d


def shortcut_corrupt(z1: torch.Tensor, tau: torch.Tensor,
                     z0: Optional[torch.Tensor] = None
                     ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build the corrupted ``z̃ = (1−τ) z₀ + τ z₁`` and return ``(z̃, z₀)``.

    ``z1`` shape ``(..., z_dim)``; ``tau`` shape ``(...)`` (broadcast).
    """
    if z0 is None:
        z0 = torch.randn_like(z1)
    tau_b = tau.unsqueeze(-1)
    z_tilde = (1.0 - tau_b) * z0 + tau_b * z1
    return z_tilde, z0


def ramp_weight(tau: torch.Tensor) -> torch.Tensor:
    """Eq. 8: ``w(τ) = 0.9 τ + 0.1`` (linear ramp giving more weight to clean)."""
    return 0.9 * tau + 0.1


# ---------------------------------------------------------------------------
# Interactive Dynamics Transformer (paper §3.2)
# ---------------------------------------------------------------------------

@dataclass
class DynamicsConfig:
    z_dim: int                      # tokenizer bottleneck dim
    action_dim: int                 # continuous action dim (pre-discretization)
    n_action_bins: int              # for action embedding lookup (categorical)
    d_model: int = 256
    n_layers: int = 6
    n_heads: int = 8
    ff_mult: int = 4
    n_register: int = 4             # learned register tokens per timestep
    k_max: int = 4                  # finest step size = 1/k_max
    tau_n_bins: int = 32            # discrete embedding lookup for τ
    soft_cap: float = 50.0
    rope_base: float = 10_000.0
    attn_impl: str = 'auto'         # 'auto' | 'manual' | 'sdpa'
    sf_bootstrap: bool = True       # shortcut self-consistency term (paper eq. 7)
    # n_tokens_per_step is computed internally:
    #   1 (z̃) + 1 (action) + 1 (τ,d) + n_register


class DynamicsTransformer(nn.Module):
    """Block-causal-in-time 1-D transformer that denoises z via shortcut forcing.

    Per timestep we feed ``n_tokens_per_step = n_register + 3`` tokens in this
    fixed order:

        [ register_1, …, register_S_r,
          action_token,
          (τ, d)_token,
          z̃_token ]

    Attention is causal in time (token at step t can attend to all tokens at
    steps ≤ t) and full within a step. The clean-z prediction ẑ₁ is read out
    from the z̃-token's hidden state. The agent / reward heads (added in
    Phase 2 by the trainer) read out from a *separate* register slot — see
    ``hidden_for_agent``.
    """

    def __init__(self, cfg: DynamicsConfig):
        super().__init__()
        self.cfg = cfg
        D = cfg.d_model
        self.n_per_step = cfg.n_register + 3
        self.AGENT_REGISTER_INDEX = 0   # first register reserved for agent head

        # Per-modality input projections (paper §3.2).
        self.z_proj = nn.Linear(cfg.z_dim, D)
        self.act_cont_proj = nn.Linear(cfg.action_dim, D)
        # Discrete action embedding (per bin per dim) — supplements continuous
        # projection so the network can leverage the categorical structure.
        self.act_disc_embed = nn.Embedding(cfg.n_action_bins * cfg.action_dim, D)
        # τ and d are discrete grid points; embed each then add channels.
        self.tau_embed = nn.Embedding(cfg.tau_n_bins, D // 2)
        self.d_embed = nn.Embedding(int(math.log2(cfg.k_max)) + 1, D // 2)
        # Learned register tokens (shared across timesteps).
        self.register_tokens = nn.Parameter(torch.zeros(cfg.n_register, D))
        nn.init.normal_(self.register_tokens, std=0.02)

        # Stack of transformer blocks.
        self.blocks = nn.ModuleList([
            TransformerBlock(D, cfg.n_heads, ff_mult=cfg.ff_mult,
                             soft_cap=cfg.soft_cap,
                             attn_impl=cfg.attn_impl)
            for _ in range(cfg.n_layers)
        ])
        self.norm_out = RMSNorm(D)
        # x-prediction head: ẑ₁ from z̃-token's hidden state.
        self.z1_head = nn.Linear(D, cfg.z_dim)

        # Cached attention mask & rope — built lazily per (T_ctx, device).
        self._mask_cache: Dict[Tuple[int, torch.device], torch.Tensor] = {}
        self._rope_cache: Dict[Tuple[int, torch.device], Tuple[torch.Tensor,
                                                                torch.Tensor]] = {}

    # ----------------------------------------------------- attention scaffolding
    def _block_causal_mask(self, T: int, device: torch.device) -> torch.Tensor:
        """``(L, L)`` boolean mask — True means *allowed*.

        Block-causal: token at (t, k) attends to all (t', k') with t' < t,
        plus all k' at t' = t.
        """
        key = (T, device)
        if key in self._mask_cache:
            return self._mask_cache[key]
        L = T * self.n_per_step
        # Time index for each position.
        t_idx = torch.arange(L, device=device) // self.n_per_step
        mask = t_idx.unsqueeze(0) <= t_idx.unsqueeze(1)
        self._mask_cache[key] = mask
        return mask

    def _rope(self, T: int, device: torch.device
              ) -> Tuple[torch.Tensor, torch.Tensor]:
        """RoPE applied to **time** positions, repeated per intra-step token.

        Length L = T * n_per_step.
        """
        key = (T, device)
        if key in self._rope_cache:
            return self._rope_cache[key]
        head_dim = self.cfg.d_model // self.cfg.n_heads
        cos_t, sin_t = _rope_cache(T, head_dim, device, base=self.cfg.rope_base)
        cos = cos_t.repeat_interleave(self.n_per_step, dim=0)
        sin = sin_t.repeat_interleave(self.n_per_step, dim=0)
        self._rope_cache[key] = (cos, sin)
        return cos, sin

    # --------------------------------------------------- per-step input assembly
    def _action_token(self, action: torch.Tensor) -> torch.Tensor:
        """Continuous action projection (B, T, A) -> (B, T, D).

        2026-05-10 simplification: removed the discrete-bin embedding path.
        Paper uses a discrete+continuous fusion because video games have
        discrete action spaces; for continuous APC actions the binning is
        a quantization bottleneck (21 bins on a [-1, 1] continuous knob
        loses ~5 bits of precision and adds noise to gradient).  The
        continuous projection alone is the cleaner interface.

        ``act_disc_embed`` is still constructed in ``__init__`` for
        checkpoint back-compat but is no longer in the forward path.
        """
        return self.act_cont_proj(action)

    def _tau_d_token(self, tau: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        """``tau`` and ``d`` shape ``(B, T)`` floats → ``(B, T, D)``."""
        # Discretize τ to ``tau_n_bins`` grid (paper uses discrete embeddings).
        tau_idx = (tau.clamp(0.0, 1.0) * (self.cfg.tau_n_bins - 1)
                   ).long().clamp_(0, self.cfg.tau_n_bins - 1)
        # d ∈ {1, 1/2, …, 1/k_max}; map to integer log2(1/d) ∈ {0, …, log2(k_max)}.
        d_idx = (-torch.log2(d.clamp_min(1e-6))).round().long()
        d_idx = d_idx.clamp_(0, int(math.log2(self.cfg.k_max)))
        return torch.cat([self.tau_embed(tau_idx), self.d_embed(d_idx)], dim=-1)

    def assemble_tokens(self, z_tilde: torch.Tensor, tau: torch.Tensor,
                        d: torch.Tensor, action: torch.Tensor
                        ) -> torch.Tensor:
        """Build the per-step token sequence.

        Inputs (all batched ``(B, T, …)``):
          ``z_tilde`` (B, T, z_dim) — corrupted observation latents
          ``tau``      (B, T)       — per-step signal level
          ``d``        (B, T)       — per-step step size
          ``action``   (B, T, A)    — continuous actions

        Returns ``(B, T * n_per_step, D)``.
        """
        B, T = z_tilde.shape[:2]
        regs = self.register_tokens.view(1, 1, self.cfg.n_register, -1
                                          ).expand(B, T, -1, -1)
        a_tok = self._action_token(action).unsqueeze(2)              # (B,T,1,D)
        td_tok = self._tau_d_token(tau, d).unsqueeze(2)              # (B,T,1,D)
        z_tok = self.z_proj(z_tilde).unsqueeze(2)                    # (B,T,1,D)
        # Order: registers, action, (τ,d), z̃.
        seq = torch.cat([regs, a_tok, td_tok, z_tok], dim=2)         # (B,T,K,D)
        return seq.reshape(B, T * self.n_per_step, -1)

    # ------------------------------------------------------------------ forward
    def forward(self, z_tilde: torch.Tensor, tau: torch.Tensor,
                d: torch.Tensor, action: torch.Tensor
                ) -> Dict[str, torch.Tensor]:
        """Run the trunk; return predicted ẑ₁ + per-step agent hidden state.

        Returns dict with:
          ``z1_hat``     : ``(B, T, z_dim)``   — clean-z prediction (x-prediction)
          ``agent_hid``  : ``(B, T, D)``       — agent register hidden state
                                                 (used by Phase-2 BC heads
                                                 and Phase-3 RL heads)
        """
        B, T = z_tilde.shape[:2]
        x = self.assemble_tokens(z_tilde, tau, d, action)
        cos, sin = self._rope(T, x.device)
        mask = self._block_causal_mask(T, x.device)
        for blk in self.blocks:
            x = blk(x, cos, sin, mask)
        x = self.norm_out(x)
        x = x.view(B, T, self.n_per_step, -1)
        # z̃ token is at the last intra-step slot.
        z1_hat = self.z1_head(x[:, :, -1, :])
        agent_hid = x[:, :, self.AGENT_REGISTER_INDEX, :]
        return {'z1_hat': z1_hat, 'agent_hid': agent_hid, 'all_hidden': x}

    # ------------------------------------------------------------------
    # KV-cache fast path for autoregressive imagination.
    # ------------------------------------------------------------------
    def build_past_kv_cache(self, z_past: torch.Tensor, tau_past: torch.Tensor,
                              d_past: torch.Tensor, action_past: torch.Tensor
                              ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Run trunk over PAST positions only and cache per-layer (K, V).

        The cached K/V live for the lifetime of one imagination call.
        Past positions never change between K-shortcut iterations because:
          * z_past, tau_past, d_past, action_past are fixed
          * block-causal mask: past tokens never attend to the last step
        so layer-l outputs for past positions also never change.

        Returns list ``[(K_past_l, V_past_l)]`` for each transformer block,
        each shape ``(B, n_heads, T_past * n_per_step, head_dim)``.

        Also returns the FINAL-LAYER past activations as part of the cache
        only if needed (we don't read past outputs in imagine_next_z).
        """
        B, T_past = z_past.shape[:2]
        L_past = T_past * self.n_per_step
        x_past = self.assemble_tokens(z_past, tau_past, d_past, action_past)
        # RoPE for past positions = positions 0..L_past-1.  We can reuse
        # the global rope cache and slice.
        cos_full, sin_full = self._rope(T_past + 1, x_past.device)
        cos_past = cos_full[:L_past]
        sin_past = sin_full[:L_past]
        mask_past = self._block_causal_mask(T_past, x_past.device)
        cache: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for blk in self.blocks:
            # Cache (K, V) projections for THIS block's input (x_past).
            k_p, v_p = blk.project_past_kv(x_past, cos_past, sin_past)
            cache.append((k_p, v_p))
            # Advance x_past through the block (full forward — no cache).
            # Past tokens only attend to past tokens, so this is correct.
            x_past = blk(x_past, cos_past, sin_past, mask_past)
        # We discard the post-trunk x_past — imagine_next_z only reads the
        # last-step output.
        return cache

    def forward_last_step_with_cache(
        self, z_past: torch.Tensor, tau_past: torch.Tensor,
        d_past: torch.Tensor, action_past: torch.Tensor,
        z_last: torch.Tensor, tau_last: torch.Tensor,
        d_last: torch.Tensor, action_last: torch.Tensor,
        cache: List[Tuple[torch.Tensor, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        """Compute trunk output for the LAST step only, given cached past K/V.

        Inputs:
          ``z_past, tau_past, d_past, action_past`` : (B, T_past, …) — same
            tensors used to build ``cache`` (used here only to assemble
            the dummy past tokens for shape checks; we never re-run the
            past through the trunk).  Pass ``None`` to skip the assembly
            and rely on cache shapes only.
          ``z_last``       : (B, 1, z_dim)
          ``tau_last``     : (B, 1)
          ``d_last``       : (B, 1)
          ``action_last``  : (B, 1, A)
          ``cache``        : output of ``build_past_kv_cache``

        Returns dict with:
          ``z1_hat_last``  : (B, z_dim)
          ``agent_hid_last``: (B, D)
        """
        B = z_last.shape[0]
        T_last = 1
        L_last = T_last * self.n_per_step
        # Number of past positions inferred from cache.
        L_past = cache[0][0].shape[2]
        T_past = L_past // self.n_per_step
        T_total = T_past + T_last
        x_last = self.assemble_tokens(z_last, tau_last, d_last, action_last)
        # RoPE for last positions = positions L_past..L_past+L_last-1.
        cos_full, sin_full = self._rope(T_total, x_last.device)
        cos_last = cos_full[L_past : L_past + L_last]
        sin_last = sin_full[L_past : L_past + L_last]
        for blk, (k_p, v_p) in zip(self.blocks, cache):
            x_last = blk.forward_last_only(x_last, cos_last, sin_last, k_p, v_p)
        x_last = self.norm_out(x_last)
        x_last = x_last.view(B, T_last, self.n_per_step, -1)
        z1_hat_last = self.z1_head(x_last[:, 0, -1, :])               # (B, z)
        agent_hid_last = x_last[:, 0, self.AGENT_REGISTER_INDEX, :]   # (B, D)
        return {'z1_hat_last': z1_hat_last, 'agent_hid_last': agent_hid_last}


# ---------------------------------------------------------------------------
# Heads: TwohotHead (reward + value), PolicyHead (per-dim categorical)
# ---------------------------------------------------------------------------

class TwohotHead(nn.Module):
    """MLP head outputting logits over a fixed twohot symlog support.

    With ``mtp_length > 1`` the head produces ``L`` parallel logit vectors
    (paper §3.2 multi-token-prediction). ``forward`` returns offset-0 logits
    so existing single-step call sites keep working; ``forward_mtp`` returns
    all ``L`` offsets.
    """

    def __init__(self, in_dim: int, hidden_dim: int, n_layers: int = 2,
                 n_bins: int = 255, low: float = -20.0, high: float = 20.0,
                 mtp_length: int = 1):
        super().__init__()
        self.n_bins = n_bins
        self.mtp_length = max(1, int(mtp_length))
        self.register_buffer('bin_edges', torch.linspace(low, high, n_bins))
        self.head = MLP(in_dim, hidden_dim, self.mtp_length * n_bins,
                        n_layers=n_layers, zero_init_last=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # offset-0 logits (used for value / current-step reward).
        return self.head(x)[..., : self.n_bins]

    def forward_mtp(self, x: torch.Tensor) -> torch.Tensor:
        """All-offset logits, shape ``(..., L, n_bins)``."""
        out = self.head(x)
        return out.view(*x.shape[:-1], self.mtp_length, self.n_bins)

    @torch.no_grad()
    def expectation(self, logits: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=-1)
        sym = (probs * self.bin_edges).sum(dim=-1)
        return symexp(sym)

    def loss(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        sym = symlog(target)
        twohot = twohot_encode(sym, self.bin_edges)
        log_probs = F.log_softmax(logits, dim=-1)
        return -(twohot * log_probs).sum(dim=-1)

    def loss_mtp(self, logits_all: torch.Tensor,
                  targets_all: torch.Tensor) -> torch.Tensor:
        """L-step twohot CE loss.

        ``logits_all`` shape ``(..., L, n_bins)``; ``targets_all`` ``(..., L)``.
        Returns per-element loss summed over L (caller can mean / weight).
        """
        sym = symlog(targets_all)
        twohot = twohot_encode(sym, self.bin_edges)              # (...,L,K)
        log_probs = F.log_softmax(logits_all, dim=-1)
        return -(twohot * log_probs).sum(dim=-1)                  # (...,L)


class PolicyHead(nn.Module):
    """Per-action-dim categorical over uniform bins in [-1, 1].

    Used by the actor in all three phases. Phase 2 trains via
    cross-entropy (BC) on dataset actions; Phase 3 trains via PMPO.
    """

    def __init__(self, in_dim: int, hidden_dim: int, action_dim: int,
                 n_action_bins: int = 21, n_layers: int = 2,
                 mtp_length: int = 1):
        super().__init__()
        self.action_dim = action_dim
        self.n_bins = n_action_bins
        self.mtp_length = max(1, int(mtp_length))
        self.head = MLP(in_dim, hidden_dim,
                        self.mtp_length * action_dim * n_action_bins,
                        n_layers=n_layers, zero_init_last=True)
        self.register_buffer('bin_centres',
                             torch.linspace(-1.0, 1.0, n_action_bins))

    def logits(self, latent: torch.Tensor) -> torch.Tensor:
        B = latent.shape[0]
        out = self.head(latent)[..., : self.action_dim * self.n_bins]
        return out.view(B, self.action_dim, self.n_bins)

    def logits_mtp(self, latent: torch.Tensor) -> torch.Tensor:
        """All-offset action logits, shape ``(B, L, action_dim, n_bins)``."""
        B = latent.shape[0]
        return self.head(latent).view(B, self.mtp_length,
                                       self.action_dim, self.n_bins)

    def forward(self, latent: torch.Tensor, *,
                deterministic: bool = False
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        action, log_prob, entropy, _ = self.sample_with_raw(
            latent, deterministic=deterministic)
        return action, log_prob, entropy

    def sample_with_raw(self, latent: torch.Tensor, *,
                         deterministic: bool = False
                         ) -> Tuple[torch.Tensor, torch.Tensor,
                                     torch.Tensor, torch.Tensor]:
        """Sample and return ``(action, log_prob, entropy, raw)``.

        ``raw`` is the **canonical sample representation** that should
        be stored when the caller needs to recompute log_prob later
        (e.g. RL actor losses).  For the discrete head this is the bin
        index.  Using ``raw`` instead of ``action`` for log_prob
        recomputation avoids any quantization-or-precision round-trip.
        """
        logits = self.logits(latent)
        if deterministic:
            idx = logits.argmax(dim=-1)
        else:
            probs = F.softmax(logits, dim=-1)
            idx = torch.distributions.Categorical(probs=probs).sample()
        action = self.bin_centres[idx]
        log_probs = F.log_softmax(logits, dim=-1)
        log_prob = log_probs.gather(-1, idx.unsqueeze(-1)).squeeze(-1).sum(-1)
        entropy = -(F.softmax(logits, dim=-1) * log_probs).sum(-1).sum(-1)
        return action, log_prob, entropy, idx

    def log_prob_of_raw(self, latent: torch.Tensor,
                         raw: torch.Tensor) -> torch.Tensor:
        """Log-prob of a stored bin index ``raw`` shape ``(B, action_dim)``."""
        logits = self.logits(latent)
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs.gather(-1, raw.long().unsqueeze(-1)
                                  ).squeeze(-1).sum(-1)

    def log_prob_of(self, latent: torch.Tensor, action: torch.Tensor
                     ) -> torch.Tensor:
        """Log-prob of a discretized continuous action (for BC + PMPO)."""
        logits = self.logits(latent)
        # Map continuous action ∈ [-1, 1] back to nearest bin index.
        idx = ((action + 1.0) * 0.5 * (self.n_bins - 1)
               ).round().long().clamp_(0, self.n_bins - 1)   # (B, action_dim)
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs.gather(-1, idx.unsqueeze(-1)).squeeze(-1).sum(-1)

    def log_prob_of_mtp(self, latent: torch.Tensor,
                         actions: torch.Tensor) -> torch.Tensor:
        """Log-prob of ``L`` future discretized actions.

        ``actions`` shape: ``(B, L, action_dim)`` in [-1, 1].
        Returns ``(B, L)`` summed over action dims (per offset).
        """
        logits = self.logits_mtp(latent)                          # (B,L,A,K)
        idx = ((actions + 1.0) * 0.5 * (self.n_bins - 1)
               ).round().long().clamp_(0, self.n_bins - 1)         # (B,L,A)
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs.gather(-1, idx.unsqueeze(-1)).squeeze(-1).sum(-1)

    # -- shared interface (used by pmpo_loss + early-stop entropy threshold) --

    def kl_to(self, other: 'PolicyHead', latent: torch.Tensor) -> torch.Tensor:
        """KL(self || other) at ``latent``.  Analytic for categoricals."""
        cur_logp = F.log_softmax(self.logits(latent), dim=-1)     # (B,A,K)
        with torch.no_grad():
            prior_logp = F.log_softmax(other.logits(latent), dim=-1)
        cur_p = cur_logp.exp()
        return (cur_p * (cur_logp - prior_logp)).sum(-1).sum(-1)  # (B,)

    def entropy(self, latent: torch.Tensor) -> torch.Tensor:
        """Per-state entropy H[π(·|latent)] summed over action dims.

        Used by the PMPO entropy bonus (DreamerV3 §3, η = 3e-4).
        """
        logits = self.logits(latent)                              # (B,A,K)
        log_p = F.log_softmax(logits, dim=-1)
        p = log_p.exp()
        return -(p * log_p).sum(-1).sum(-1)                       # (B,)

    @staticmethod
    def reference_entropy(action_dim: int, n_action_bins: int) -> float:
        """Max-entropy reference: uniform over all bins per dim."""
        return float(action_dim) * math.log(max(2, int(n_action_bins)))


class ContinuousPolicyHead(nn.Module):
    """Tanh-squashed-Gaussian (TanhNormal) actor for continuous APC actions.

    Outputs per-action-dim ``(mu, log_std)``; samples via the reparam trick
    ``a = tanh(mu + sigma * eps)`` with ``eps ~ N(0, I)``.  This is the
    standard continuous-control distribution (SAC, DreamerV3-continuous):
    bounded to ``[-1, 1]`` by construction with no boundary singularity, and
    the underlying Gaussian gives well-behaved gradients & analytic KL for
    PMPO.

    Replaces the discrete-bin ``PolicyHead`` for chemistry / process control
    where 6%-of-range bin steps (n_bins=21 over [-1,1] → 0.1) are too coarse
    to track tight setpoints — a major contributor to actor collapse on
    test_sim diagnosed 2026-05-05.

    Interface mirrors ``PolicyHead`` (forward / log_prob_of /
    log_prob_of_mtp / kl_to / reference_entropy) so the trainer + PMPO loss
    code is policy-type agnostic.
    """

    # Default log‐std bounds follow DreamerV3 §3 (σ ∈ [0.1, 1.0]).
    # The bounds were a key stability fix that allowed V3's single
    # hyperparameter set to work across 150+ tasks. They can be widened
    # per-simulator via the constructor / TrainConfig if a particular
    # plant needs broader exploration.
    LOG_STD_MIN: float = -2.3          # σ ≥ 0.10
    LOG_STD_MAX: float = 0.0           # σ ≤ 1.00

    def __init__(self, in_dim: int, hidden_dim: int, action_dim: int,
                 n_layers: int = 2, mtp_length: int = 1,
                 init_log_std: float = -0.5,
                 log_std_min: Optional[float] = None,
                 log_std_max: Optional[float] = None):
        super().__init__()
        self.action_dim = action_dim
        self.mtp_length = max(1, int(mtp_length))
        # Per-instance overrides (fall back to V3 defaults if None).
        self.log_std_min = (float(log_std_min) if log_std_min is not None
                             else self.LOG_STD_MIN)
        self.log_std_max = (float(log_std_max) if log_std_max is not None
                             else self.LOG_STD_MAX)
        # Clamp init within bounds so the starting σ is always realisable.
        init_log_std = float(min(max(init_log_std, self.log_std_min),
                                   self.log_std_max))
        # Output 2 * action_dim per offset (mu, log_std).
        self.head = MLP(in_dim, hidden_dim,
                        self.mtp_length * action_dim * 2,
                        n_layers=n_layers, zero_init_last=False)
        # Bias the log_std output toward ``init_log_std`` so the policy
        # starts with a usable exploration spread (σ≈0.6) rather than
        # whatever zero-init gives.  We add a learned per-dim log_std bias
        # since zero_init_last=False already gave the head normal-init
        # weights; the offset just shifts the starting point.
        self.register_buffer('log_std_init',
                              torch.full((action_dim,), float(init_log_std)))

    # ---- raw (mu, log_std) extraction --------------------------------

    def _params_offset(self, latent: torch.Tensor, L: int
                        ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(mu, log_std)`` for offsets ``[0, L)`` shape (B, L, A)."""
        B = latent.shape[0]
        out = self.head(latent).view(B, self.mtp_length, self.action_dim, 2)
        out = out[:, :L]
        # ``mu`` is the pre-tanh mean (unbounded); the action is bounded to
        # ``[-1, 1]`` by the tanh squash applied at sample time.  Squashing
        # ``mu`` here would clip the deterministic action range.  We do
        # however soft-cap ``mu`` to keep gradients well-behaved when the
        # head is overconfident — same idea as the dynamics ``soft_cap``.
        cap = 8.0
        mu = cap * torch.tanh(out[..., 0] / cap)                  # (B,L,A)
        log_std = out[..., 1] + self.log_std_init.view(1, 1, -1)
        log_std = log_std.clamp(self.log_std_min, self.log_std_max)
        return mu, log_std

    def dist_params(self, latent: torch.Tensor
                     ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Distribution params at offset 0.  Returns ``(mu, log_std)``
        each shape ``(B, action_dim)``."""
        mu, log_std = self._params_offset(latent, L=1)
        return mu[:, 0], log_std[:, 0]

    # ---- sampling + log-prob ------------------------------------------

    @staticmethod
    def _tanh_log_prob(mu: torch.Tensor, log_std: torch.Tensor,
                        u: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """log p(a) under TanhNormal where ``u`` is the pre-tanh sample.

        Returns shape matching ``mu`` (per-dim, before sum)."""
        std = log_std.exp()
        # Underlying Gaussian log-prob.
        log_prob_u = -0.5 * (((u - mu) / std) ** 2
                              + 2.0 * log_std
                              + math.log(2.0 * math.pi))
        # Tanh squash Jacobian: log|da/du| = log(1 - tanh(u)^2).
        # Numerically stable form: 2 * (log(2) - u - softplus(-2u)).
        log_det = 2.0 * (math.log(2.0) - u - F.softplus(-2.0 * u))
        return log_prob_u - log_det

    def forward(self, latent: torch.Tensor, *,
                deterministic: bool = False
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        action, log_prob, entropy, _ = self.sample_with_raw(
            latent, deterministic=deterministic)
        return action, log_prob, entropy

    def sample_with_raw(self, latent: torch.Tensor, *,
                         deterministic: bool = False
                         ) -> Tuple[torch.Tensor, torch.Tensor,
                                     torch.Tensor, torch.Tensor]:
        """Sample and return ``(action, log_prob, entropy, raw)``.

        ``raw`` is the **pre-tanh** sample ``u``.  Using ``u`` directly
        when later recomputing log_prob (instead of going through
        ``action = tanh(u) → atanh(action)``) avoids a serious bfloat16
        precision pitfall: at moderate ``|u| ≳ 3``, ``tanh(u)`` rounds
        to exactly ±1.0 in bfloat16, after which ``atanh`` returns a
        completely wrong ``u`` (clamped at ±7.6).  In a typical training
        run this corrupts the actor gradient by orders of magnitude and
        manifests as ``log_prob`` values around −1500 with std ≋ 2750.
        """
        mu, log_std = self.dist_params(latent)
        std = log_std.exp()
        if deterministic:
            u = mu
            action = torch.tanh(u)
            log_prob = torch.zeros(latent.shape[0], device=latent.device,
                                    dtype=latent.dtype)
        else:
            eps = torch.randn_like(mu)
            u = mu + std * eps
            action = torch.tanh(u)
            log_prob = self._tanh_log_prob(mu, log_std, u, action).sum(-1)
        # Differential-entropy of the underlying Gaussian.  See class
        # docstring for why we ignore the (intractable) tanh correction.
        entropy = (0.5 * (math.log(2.0 * math.pi * math.e)
                           + 2.0 * log_std)).sum(-1)
        return action, log_prob, entropy, u

    def log_prob_of_raw(self, latent: torch.Tensor,
                         raw: torch.Tensor) -> torch.Tensor:
        """Log-prob given the **pre-tanh** sample ``u`` (shape (B, A)).

        This is the numerically-correct path for RL actor losses: the
        caller stores ``u`` returned by :meth:`sample_with_raw` rather
        than the post-tanh action, so we can skip the lossy ``atanh``
        round-trip and evaluate the Gaussian density directly on ``u``.
        """
        mu, log_std = self.dist_params(latent)
        action = torch.tanh(raw)
        return self._tanh_log_prob(mu, log_std, raw, action).sum(-1)

    def log_prob_of(self, latent: torch.Tensor, action: torch.Tensor
                     ) -> torch.Tensor:
        """Log-prob of a continuous action ``(B, action_dim)`` ∈ [-1, 1]."""
        mu, log_std = self.dist_params(latent)
        # Invert tanh: u = atanh(action), clamped for numerical stability
        # near ±1 (atanh(±1) is ±inf).
        a_clamped = action.clamp(-1.0 + 1e-6, 1.0 - 1e-6)
        u = 0.5 * torch.log1p(2.0 * a_clamped / (1.0 - a_clamped))
        return self._tanh_log_prob(mu, log_std, u, a_clamped).sum(-1)

    def log_prob_of_mtp(self, latent: torch.Tensor,
                         actions: torch.Tensor) -> torch.Tensor:
        """``actions`` shape ``(B, L, A)``; returns ``(B, L)``."""
        L = actions.shape[1]
        mu, log_std = self._params_offset(latent, L=L)            # (B,L,A)
        a_clamped = actions.clamp(-1.0 + 1e-6, 1.0 - 1e-6)
        u = 0.5 * torch.log1p(2.0 * a_clamped / (1.0 - a_clamped))
        return self._tanh_log_prob(mu, log_std, u, a_clamped).sum(-1)

    # ---- shared interface (matches PolicyHead) ------------------------

    def kl_to(self, other: 'ContinuousPolicyHead',
               latent: torch.Tensor) -> torch.Tensor:
        """Analytic KL(self || other) using the underlying Gaussians.

        The tanh squash is identical for both, so the KL of the
        squashed distributions equals the KL of the underlying Gaussians.
        """
        mu1, log_std1 = self.dist_params(latent)
        with torch.no_grad():
            mu2, log_std2 = other.dist_params(latent)
        var1 = (2.0 * log_std1).exp()
        var2 = (2.0 * log_std2).exp()
        kl_per_dim = (log_std2 - log_std1
                       + (var1 + (mu1 - mu2) ** 2) / (2.0 * var2)
                       - 0.5)
        return kl_per_dim.sum(-1)                                # (B,)

    def entropy(self, latent: torch.Tensor) -> torch.Tensor:
        """Per-state Gaussian entropy summed over action dims.

        We report the entropy of the underlying Gaussian (pre-tanh)
        rather than the squashed distribution — the tanh correction
        ``E[log(1 - tanh(u)²)]`` is intractable in closed form and
        roughly state-independent, so it does not affect gradient
        directions for the entropy bonus / collapse trip.  Matches the
        SAC convention.
        """
        _, log_std = self.dist_params(latent)
        return (0.5 * (math.log(2.0 * math.pi * math.e)
                        + 2.0 * log_std)).sum(-1)

    @staticmethod
    def reference_entropy(action_dim: int, n_action_bins: int = 0) -> float:
        """Reference entropy for the early-stop trip threshold.

        Uses a Gaussian at σ=1 per dim (≈ 1.4189 nats) — interpreted as
        "the policy retains a unit-std spread around its mean".  When the
        actual entropy drops to ``frac * reference_entropy``, σ has
        collapsed by roughly ``exp(frac - 1)``.
        """
        unit_gaussian_entropy = 0.5 * math.log(2.0 * math.pi * math.e)
        return float(action_dim) * unit_gaussian_entropy


# ---------------------------------------------------------------------------
# Top-level Dreamer 4 container
# ---------------------------------------------------------------------------

@dataclass
class DreamerV4Config:
    obs_dim: int
    action_dim: int
    lookback: int                          # transformer context length T_ctx
    # Tokenizer
    tok_hidden: int = 256
    z_dim: int = 24
    mae_p_max: float = 0.5
    # Dynamics
    d_model: int = 256
    n_layers: int = 6
    n_heads: int = 8
    ff_mult: int = 4
    n_register: int = 4
    k_max: int = 4
    tau_n_bins: int = 32
    soft_cap: float = 50.0
    attn_impl: str = 'auto'                # 'auto'|'manual'|'sdpa' (DREAMER_FAST_ATTN=1)
    # Heads
    n_action_bins: int = 21
    head_hidden: int = 256
    head_n_layers: int = 2
    mtp_length: int = 1                    # paper L=8 (Phase-2 MTP)
    # Policy distribution.  ``'continuous'`` (default) uses TanhNormal;
    # ``'discrete'`` uses the legacy categorical-bin head from the
    # paper.  See ``ContinuousPolicyHead`` docstring for rationale.
    policy_type: str = 'continuous'
    policy_init_log_std: float = -0.5      # σ ≈ 0.6 at init
    # DreamerV3 §3 prescribes σ ∈ [0.1, 1.0] (⇔ log_std ∈ [-2.3, 0])
    # for the continuous actor.  Bounds can be widened per-simulator
    # via ``TrainConfig`` if a particular plant needs broader
    # exploration; defaults are chosen for stable adaptive operation.
    policy_log_std_min: float = -2.3
    policy_log_std_max: float = 0.0
    # ===== World-model backbone selection =====
    # ``'rssm'`` (new default 2026-05-30) uses the DreamerV3 recurrent
    # state-space model (GRU + categorical latent) whose deterministic
    # core can learn a held-action fixed point — the property the
    # SF-transformer lacked (0% steady-state convergence drove the
    # bootstrap-cascade across P64/P66/P67).  ``'sf_transformer'`` keeps
    # the original V4 shortcut-forcing transformer dynamics.
    world_model_type: str = 'rssm'
    rssm_deter_dim: int = 512
    rssm_n_categoricals: int = 32
    rssm_n_classes: int = 32
    rssm_embed_dim: int = 256
    rssm_hidden_dim: int = 256
    rssm_unimix: float = 0.01
    # ===== TSSM (transformer-SSM) backbone (neural-apc-mbrl) =====
    # ``'tssm'`` swaps the GRU recurrent core for a causal transformer that does
    # in-context system-ID over the lookback (sharp per-domain fixed point for
    # wide-DR generalization).  Implements the SAME RSSMDynamics interface, so it
    # reuses the entire RSSM pipeline (heads, disturbance estimator, losses,
    # imagination).  Reuses the rssm_* categorical-latent dims (n_categoricals/
    # n_classes/embed_dim/unimix); these are the transformer-specific dims.
    tssm_d_model: int = 512            # transformer width == deter_dim (h_t)
    tssm_n_layers: int = 4
    tssm_n_heads: int = 8
    tssm_max_seq_len: int = 256        # context window cap (>= lookback + horizon)
    # ===== WM disturbance-estimator head (P87, 2026-06-05) =====
    # Auxiliary supervised head: predicts the hidden/unmeasured disturbance
    # (per CV channel) from the RSSM posterior feature ``[h, z]``.  ``0`` ⇒
    # head not built (pre-P87 model).  Sized at runtime to ``len(cv_indices)``.
    disturbance_head_dim: int = 0
    disturbance_head_hidden: int = 0
    disturbance_head_layers: int = 2


class DreamerV4(nn.Module):
    def __init__(self, cfg: DreamerV4Config):
        super().__init__()
        self.cfg = cfg
        self.world_model_type = str(
            getattr(cfg, 'world_model_type', 'sf_transformer')).lower()
        if self.world_model_type == 'rssm':
            # DreamerV3 RSSM backbone.  No tokenizer (the RSSM has an
            # integrated MLP encoder/decoder); heads read from the
            # posterior feature ``[h, z_flat]`` of width ``feat_dim``.
            from models.dreamer_v4_rssm import RSSMDynamics, RSSMConfig
            self.tokenizer = None
            rssm_cfg = RSSMConfig(
                obs_dim=cfg.obs_dim, action_dim=cfg.action_dim,
                deter_dim=int(getattr(cfg, 'rssm_deter_dim', 512)),
                n_categoricals=int(getattr(cfg, 'rssm_n_categoricals', 32)),
                n_classes=int(getattr(cfg, 'rssm_n_classes', 32)),
                embed_dim=int(getattr(cfg, 'rssm_embed_dim', 256)),
                hidden_dim=int(getattr(cfg, 'rssm_hidden_dim', 256)),
                unimix=float(getattr(cfg, 'rssm_unimix', 0.01)),
            )
            self.dynamics = RSSMDynamics(rssm_cfg)
            D = self.dynamics.feat_dim
        elif self.world_model_type == 'tssm':
            # neural-apc-mbrl: transformer-SSM core implementing the SAME
            # RSSMDynamics interface (feat=[h, z_flat]).  No tokenizer; heads +
            # disturbance estimator read the feature of width ``feat_dim`` just
            # like the RSSM path.  Reuses rssm_* categorical-latent dims.
            from models.transformer_ssm import (TransformerSSMDynamics,
                                                TransformerSSMConfig)
            self.tokenizer = None
            tssm_cfg = TransformerSSMConfig(
                obs_dim=cfg.obs_dim, action_dim=cfg.action_dim,
                deter_dim=int(getattr(cfg, 'tssm_d_model', 512)),
                n_categoricals=int(getattr(cfg, 'rssm_n_categoricals', 32)),
                n_classes=int(getattr(cfg, 'rssm_n_classes', 32)),
                embed_dim=int(getattr(cfg, 'rssm_embed_dim', 256)),
                n_layers=int(getattr(cfg, 'tssm_n_layers', 4)),
                n_heads=int(getattr(cfg, 'tssm_n_heads', 8)),
                unimix=float(getattr(cfg, 'rssm_unimix', 0.01)),
                max_seq_len=int(getattr(cfg, 'tssm_max_seq_len', 256)),
            )
            self.dynamics = TransformerSSMDynamics(tssm_cfg)
            D = self.dynamics.feat_dim
        else:
            self.tokenizer = Tokenizer(cfg.obs_dim, cfg.tok_hidden, cfg.z_dim,
                                        mae_p_max=cfg.mae_p_max)
            dyn_cfg = DynamicsConfig(
                z_dim=cfg.z_dim, action_dim=cfg.action_dim,
                n_action_bins=cfg.n_action_bins,
                d_model=cfg.d_model, n_layers=cfg.n_layers, n_heads=cfg.n_heads,
                ff_mult=cfg.ff_mult, n_register=cfg.n_register,
                k_max=cfg.k_max, tau_n_bins=cfg.tau_n_bins,
                soft_cap=cfg.soft_cap,
                attn_impl=cfg.attn_impl,
            )
            self.dynamics = DynamicsTransformer(dyn_cfg)
            # Heads read from the agent-register hidden state (dim = d_model).
            D = cfg.d_model
        self.policy_type = str(getattr(cfg, 'policy_type', 'continuous')).lower()
        if self.policy_type == 'continuous':
            self.policy = ContinuousPolicyHead(
                D, cfg.head_hidden, cfg.action_dim,
                n_layers=cfg.head_n_layers, mtp_length=cfg.mtp_length,
                init_log_std=getattr(cfg, 'policy_init_log_std', -0.5),
                log_std_min=getattr(cfg, 'policy_log_std_min', None),
                log_std_max=getattr(cfg, 'policy_log_std_max', None))
        else:
            self.policy = PolicyHead(D, cfg.head_hidden, cfg.action_dim,
                                      n_action_bins=cfg.n_action_bins,
                                      n_layers=cfg.head_n_layers,
                                      mtp_length=cfg.mtp_length)
        self.reward = TwohotHead(D, cfg.head_hidden,
                                  n_layers=cfg.head_n_layers,
                                  mtp_length=cfg.mtp_length)
        self.value = TwohotHead(D, cfg.head_hidden,
                                 n_layers=cfg.head_n_layers)
        # EMA target for value (TD-λ stability, paper §3.3).
        self.target_value = TwohotHead(D, cfg.head_hidden,
                                         n_layers=cfg.head_n_layers)
        self.target_value.load_state_dict(self.value.state_dict())
        for p in self.target_value.parameters():
            p.requires_grad_(False)
        # Frozen prior policy snapshot (PMPO behavioural prior, paper eq. 11).
        if self.policy_type == 'continuous':
            self.prior_policy = ContinuousPolicyHead(
                D, cfg.head_hidden, cfg.action_dim,
                n_layers=cfg.head_n_layers, mtp_length=cfg.mtp_length,
                init_log_std=getattr(cfg, 'policy_init_log_std', -0.5),
                log_std_min=getattr(cfg, 'policy_log_std_min', None),
                log_std_max=getattr(cfg, 'policy_log_std_max', None))
        else:
            self.prior_policy = PolicyHead(D, cfg.head_hidden, cfg.action_dim,
                                            n_action_bins=cfg.n_action_bins,
                                            n_layers=cfg.head_n_layers,
                                            mtp_length=cfg.mtp_length)
        self.prior_policy.load_state_dict(self.policy.state_dict())
        for p in self.prior_policy.parameters():
            p.requires_grad_(False)
        # Return-scale EMA (used for diagnostic logging; PMPO does not need it).
        self.register_buffer('ret_scale', torch.ones(1))
        # P87: auxiliary WM disturbance-estimator head.  Reads the same
        # latent feature ``D`` the policy/value heads consume and is supervised
        # to regress the hidden unmeasured disturbance (per CV channel).  Last
        # layer zero-init so it contributes no gradient at step 0 (the latent
        # is shaped gradually, never shocked).  ``None`` ⇒ disabled.
        dist_dim = int(getattr(cfg, 'disturbance_head_dim', 0) or 0)
        if dist_dim > 0:
            dist_hidden = int(getattr(cfg, 'disturbance_head_hidden', 0) or 0) \
                or cfg.head_hidden
            dist_layers = int(getattr(cfg, 'disturbance_head_layers', 2) or 2)
            self.disturbance = MLP(D, dist_hidden, dist_dim,
                                   n_layers=dist_layers, zero_init_last=True)
        else:
            self.disturbance = None

    # ---------------------------------------------------------- compile
    def maybe_compile(self, mode: str = 'default') -> None:
        """Compile the dynamics transformer + tokenizer with ``torch.compile``.

        Big P3 speedup (typically 2-3×) by fusing kernels and removing Python
        overhead in the K=4 imagination inner loop. Idempotent — calling
        twice is a no-op. Triggered by ``DREAMER_COMPILE=1`` (or ``=mode``)
        env var, or by calling this directly. Errors are downgraded to a
        warning so the trainer still runs unoptimised on unsupported setups.

        ``mode='default'`` is the safe pick: ``'reduce-overhead'`` enables
        cudagraphs which conflicts with our cached RoPE tensors, and
        ``'max-autotune'`` triggers very long warmups for marginal gain on
        a transformer at this scale.
        """
        if getattr(self, '_compiled', False):
            return
        try:
            # P3 imagination calls dynamics with context lengths T=seq_len..
            # seq_len+H (≈43 distinct shapes). The default recompile_limit=8
            # would bail to eager after 8 shapes despite dynamic=True. Bump
            # to a comfortable margin so every shape stays compiled.
            import torch._dynamo as _dynamo
            try:
                _dynamo.config.recompile_limit = max(
                    int(getattr(_dynamo.config, 'recompile_limit', 8)), 128)
            except Exception:
                pass
            try:
                _dynamo.config.cache_size_limit = max(
                    int(getattr(_dynamo.config, 'cache_size_limit', 8)), 128)
            except Exception:
                pass
            # Production safety net: if ANY graph fails to compile at runtime
            # (e.g. an un-warmed branch hit only deep into a long run), fall
            # back to eager for that graph instead of crashing the whole run.
            try:
                _dynamo.config.suppress_errors = True
            except Exception:
                pass
            if self.world_model_type == 'rssm':
                # RSSM hot path is ``rollout_observed`` (always called with a
                # static (B, seq_len) batch in every phase — the SF-transformer's
                # varying imagination context that forced ``dynamic=True`` does
                # not exist here).  Compiling the *module* is a no-op for RSSM
                # because ``rollout_observed`` is invoked as a method, not
                # ``forward``; so compile the bound method directly with
                # ``dynamic=False`` to let ``reduce-overhead`` capture the full
                # ~128-step GRU rollout as a single CUDA graph — this is what
                # actually removes the per-step kernel-launch overhead that
                # keeps the WM launch-bound at ~16% GPU util.
                self.dynamics.rollout_observed = torch.compile(
                    self.dynamics.rollout_observed, mode=mode, dynamic=False)
                # img_step is the per-step hot path in BOTH P3 imagination and
                # the #2 latent-overshoot loop (max_starts x len calls/iter) —
                # the dominant compute cost the imagination/overshoot loops
                # incur OUTSIDE rollout_observed (which they bypass).  Static
                # (B, ...) shape so dynamic=False captures it cleanly.  Wrapped
                # separately so a failure here does not lose the proven
                # rollout_observed compile.
                try:
                    self.dynamics.img_step = torch.compile(
                        self.dynamics.img_step, mode=mode, dynamic=False)
                    _imgs = '+ img_step'
                except Exception as _e2:
                    _imgs = '(img_step skipped)'
                    print(f'[dreamer_v4] img_step compile skipped ({_e2!r})',
                          flush=True)
                self._compiled = True
                print(f'[dreamer_v4] torch.compile(mode={mode}, dynamic=False) '
                      f'enabled on RSSM rollout_observed {_imgs}', flush=True)
            elif self.world_model_type == 'tssm':
                # TSSM: compile the teacher-forced ``rollout_observed`` (T is
                # fixed = seq_len in WM training, so a static graph captures it).
                # ``img_step`` is NOT compiled: its causal window grows by one
                # token per imagination step, so a static graph would recompile
                # every step — leave it eager until the KV-cache lands.
                # ``dynamic=True`` tolerates the seq_len differences between WM
                # training and the diagnostics probes.
                try:
                    self.dynamics.rollout_observed = torch.compile(
                        self.dynamics.rollout_observed, mode=mode, dynamic=True)
                    self._compiled = True
                    print('[dreamer_v4] torch.compile(mode='
                          f'{mode}, dynamic=True) enabled on TSSM '
                          'rollout_observed (img_step eager: variable window)',
                          flush=True)
                except Exception as _et:
                    print(f'[dreamer_v4] TSSM compile skipped ({_et!r})',
                          flush=True)
                    self._compiled = False
            else:
                self.dynamics = torch.compile(self.dynamics, mode=mode,
                                                dynamic=True)
                if self.tokenizer is not None:
                    self.tokenizer = torch.compile(self.tokenizer, mode=mode,
                                                     dynamic=True)
                self._compiled = True
                _tok = '+ tokenizer' if self.tokenizer is not None else '(rssm)'
                print(f'[dreamer_v4] torch.compile(mode={mode}) enabled '
                      f'on dynamics {_tok}', flush=True)
        except Exception as e:
            print(f'[dreamer_v4] torch.compile failed ({e!r}); '
                  f'falling back to eager', flush=True)
            self._compiled = False

    # ---------------------------------------------------------- target / prior
    def update_target(self, tau: float = 0.02) -> None:
        with torch.no_grad():
            for p, t in zip(self.value.parameters(),
                            self.target_value.parameters()):
                t.data.mul_(1.0 - tau).add_(tau * p.data)

    def snapshot_prior_policy(self) -> None:
        """Capture the current policy as the frozen PMPO prior (start of Phase 3)."""
        self.prior_policy.load_state_dict(self.policy.state_dict())
        for p in self.prior_policy.parameters():
            p.requires_grad_(False)

    def update_return_scale(self, returns: torch.Tensor,
                             ema: float = 0.99,
                             abs_cap: float = 500.0,
                             ) -> torch.Tensor:
        """EMA of the (p95-p05) return spread, with an absolute upper cap.

        On the critic-pessimism cascade the EMA tracks a monotonically
        growing spread (critic targets keep growing → spread keeps
        growing → critic keeps chasing), which normalises the actor
        advantage to death (P77: 4→109×, frozen actor).  The earlier
        per-update GROWTH-RATE clamp (P63) regressed because it also
        throttled legitimate early growth.  Following the working Cursor
        APC-Dreamer reference, we instead apply an ABSOLUTE cap
        (``abs_cap``, default 500): normal growth is untouched and only
        the implausible runaway is arrested.  ``abs_cap=0`` recovers the
        original DreamerV3-faithful uncapped behaviour.
        """
        with torch.no_grad():
            r = returns.detach().reshape(-1).float()
            if r.numel() < 2:
                spread = torch.tensor(1.0, device=r.device)
            else:
                p05 = torch.quantile(r, 0.05)
                p95 = torch.quantile(r, 0.95)
                spread = torch.clamp(p95 - p05, min=1.0)
            self.ret_scale.mul_(ema).add_((1.0 - ema) * spread)
            if abs_cap and abs_cap > 0.0:
                self.ret_scale.clamp_max_(float(abs_cap))
        return self.ret_scale.clamp_min(1.0)

    # ------------------------------------------------------- parameter groups
    def parameters_world(self):
        """World-model + reward head — trained in Phases 1 & 2.

        SF-transformer: tokenizer + dynamics + reward head.
        RSSM: dynamics (integrated encoder/decoder/GRU/prior/post) +
        reward head (no separate tokenizer).
        """
        if getattr(self, 'world_model_type', 'sf_transformer') == 'rssm':
            return (list(self.dynamics.parameters())
                    + list(self.reward.parameters()))
        return (list(self.tokenizer.parameters())
                + list(self.dynamics.parameters())
                + list(self.reward.parameters()))

    def parameters_actor(self):
        """Policy head — trained in Phases 2 (BC) & 3 (PMPO)."""
        return list(self.policy.parameters())

    def parameters_critic(self):
        """Value head — trained in Phase 3 only."""
        return list(self.value.parameters())

    # --------------------------------------------------- inference: latent step
    @torch.no_grad()
    def imagine_next_z(self, z_history: torch.Tensor, action: torch.Tensor,
                       k_steps: int = None, tau_ctx: float = None,
                       action_history: torch.Tensor = None
                       ) -> torch.Tensor:
        """Sample the next z given a history of clean z's and an action.

        ``z_history``     : ``(B, T_ctx, z_dim)``  — clean past latents.
        ``action``        : ``(B, action_dim)``    — action taken at the next step.
        ``action_history``: ``(B, T_ctx, action_dim)`` — REAL past actions
            that produced ``z_history``.  REQUIRED for correctness: during
            WM training the dynamics always sees real actions at every
            position; passing zeros for the past creates a train/inference
            distribution mismatch and the dynamics produces garbage.
            (Defaults to zeros for back-compat with old callers; warn-loud
            via ``DREAMER_ACT_HIST_REQUIRED=1`` to catch missing hookups.)
        ``tau_ctx``       : context noise level.  ``None`` (default) → auto
            ``1.0 / k_max`` so the past τ lands at ``(k_max-1)/k_max`` which
            is the MAX value in the training τ-grid (sample_tau_d only ever
            samples τ ∈ {0, 1/k, …, (k-1)/k} for k ≤ k_max).  Using
            tau_ctx=0.1 with k_max=4 puts past τ=0.9 which the model has
            literally never seen during training.

        Returns ``z_next`` of shape ``(B, z_dim)``.

        Uses paper-faithful K=k_max shortcut sampling at d=1/k_max.
        """
        cfg = self.cfg
        K = int(k_steps if k_steps is not None else cfg.k_max)
        if tau_ctx is None:
            tau_ctx = 1.0 / float(cfg.k_max)
        B, T_ctx, _ = z_history.shape
        device = z_history.device
        # Pad with one dummy step at the end to denoise.
        z0 = torch.randn(B, 1, cfg.z_dim, device=device, dtype=z_history.dtype)
        # τ for past = 1 - tau_ctx (slight corruption); for current = 0 (full noise).
        z_past_corr = (1.0 - tau_ctx) * z_history + tau_ctx * torch.randn_like(z_history)
        z_seq = torch.cat([z_past_corr, z0], dim=1)                       # (B, T_ctx+1, z)
        # Action: real past actions + supplied current action.
        if action_history is None:
            import os as _os
            if _os.environ.get('DREAMER_ACT_HIST_REQUIRED', '').strip() in ('1','true','True'):
                raise ValueError(
                    'imagine_next_z called without action_history; this is '
                    'a train/inference distribution bug.  Pass the real '
                    'past actions that produced z_history.')
            act_past = torch.zeros(B, T_ctx, cfg.action_dim, device=device,
                                    dtype=action.dtype)
        else:
            assert action_history.shape == (B, T_ctx, cfg.action_dim), (
                f'action_history shape {tuple(action_history.shape)} != '
                f'expected ({B}, {T_ctx}, {cfg.action_dim})')
            act_past = action_history.to(device=device, dtype=action.dtype)
        act_seq = torch.cat([act_past, action.unsqueeze(1)], dim=1)       # (B, T_ctx+1, A)
        # τ / d sequences: past = (1 - tau_ctx, d_min), current = (0 → 1, d_min).
        d_min = 1.0 / cfg.k_max
        tau_seq = torch.full((B, T_ctx + 1), 1.0 - tau_ctx, device=device,
                             dtype=z_history.dtype)
        d_seq = torch.full((B, T_ctx + 1), d_min, device=device,
                           dtype=z_history.dtype)
        # ---- KV-cache fast path ----------------------------------------
        # Past T_ctx positions never change across the K shortcut steps
        # (z_past, tau_past, d_past, action_past are all fixed and the
        # transformer is block-causal so past outputs are also fixed).
        # Build the per-layer (K, V) cache once and reuse it K times.
        cache = self.dynamics.build_past_kv_cache(
            z_seq[:, :-1], tau_seq[:, :-1], d_seq[:, :-1], act_seq[:, :-1])
        z_cur = z_seq[:, -1]                                              # (B, z)
        for k in range(K):
            tau_now = float(k) / K
            tau_last = torch.full((B, 1), tau_now, device=device,
                                    dtype=z_history.dtype)
            d_last = torch.full((B, 1), d_min, device=device,
                                  dtype=z_history.dtype)
            out = self.dynamics.forward_last_step_with_cache(
                None, None, None, None,
                z_last=z_cur.unsqueeze(1), tau_last=tau_last, d_last=d_last,
                action_last=action.unsqueeze(1), cache=cache)
            z1_hat_last = out['z1_hat_last']                              # (B, z)
            # Advance via x-prediction: take a step of size d_min toward ẑ₁.
            v_hat = (z1_hat_last - z_cur) / max(1e-6, 1.0 - tau_now)
            z_cur = z_cur + v_hat * d_min
        return z_cur

    @torch.no_grad()
    def policy_action(self, agent_hid: torch.Tensor, *,
                       deterministic: bool = True) -> torch.Tensor:
        """Sample an action from ``policy`` given an agent-register hidden state."""
        action, _, _ = self.policy(agent_hid, deterministic=deterministic)
        return action


# ---------------------------------------------------------------------------
# PMPO loss (paper eq. 11)
# ---------------------------------------------------------------------------

def pmpo_loss(policy, prior_policy,
              latent: torch.Tensor, raw_action: torch.Tensor,
              advantage: torch.Tensor,
              alpha: float = 0.5, beta: float = 0.1,
              entropy_coef: float = 0.0
              ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Policy Maximum Likelihood Optimization loss (Dreamer 4 eq. 11).

    Splits states by sign of advantage:
      - D⁺ = {s | A ≥ 0} → upweight chosen action
      - D⁻ = {s | A < 0} → downweight chosen action
    Plus a KL term to a frozen prior policy and (optionally) an entropy
    bonus following DreamerV3 §3 (``η = 3e-4`` by default in V3).  The
    entropy bonus acts as a soft σ-floor for the continuous TanhNormal
    head and a uniform-prior pull for the discrete categorical head;
    both are essential for stability when the advantage signal is
    heavy-tailed (e.g. process-control violation penalties).

    All tensors are flat ``(N, …)``.  ``raw_action`` is the canonical
    sample representation produced by ``policy.sample_with_raw`` (bin
    index for discrete, pre-tanh ``u`` for continuous) — see those
    methods for why this is required for numerical correctness in
    bfloat16 training.  Polymorphic in policy class — both
    ``PolicyHead`` and ``ContinuousPolicyHead`` expose
    ``log_prob_of_raw`` / ``kl_to`` / ``entropy``.
    """
    log_prob = policy.log_prob_of_raw(latent, raw_action)        # (N,)
    kl = policy.kl_to(prior_policy, latent)                      # (N,)

    pos_mask = (advantage >= 0).float()
    neg_mask = 1.0 - pos_mask
    n_pos = pos_mask.sum().clamp_min(1.0)
    n_neg = neg_mask.sum().clamp_min(1.0)
    loss_pos = -(alpha * (log_prob * pos_mask).sum() / n_pos)
    loss_neg = -((1.0 - alpha) * (-(log_prob) * neg_mask).sum() / n_neg)
    loss_kl = beta * kl.mean()
    if entropy_coef and entropy_coef > 0.0:
        ent = policy.entropy(latent)                            # (N,)
        loss_ent = -float(entropy_coef) * ent.mean()
        ent_mean_diag = ent.mean().detach()
    else:
        loss_ent = torch.zeros((), device=log_prob.device,
                                dtype=log_prob.dtype)
        ent_mean_diag = torch.zeros((), device=log_prob.device,
                                     dtype=log_prob.dtype)
    total = loss_pos + loss_neg + loss_kl + loss_ent
    diag = {
        'pmpo_loss': total.detach(),
        'pmpo_pos_frac': (n_pos / (n_pos + n_neg)).detach(),
        'pmpo_kl': kl.mean().detach(),
        'pmpo_logp_mean': log_prob.mean().detach(),
        'pmpo_entropy_bonus': ent_mean_diag,
    }
    return total, diag


# ---------------------------------------------------------------------------
# DreamerV3 REINFORCE actor loss (V3 §3, eq. 3) — robust alternative
# ---------------------------------------------------------------------------

def reinforce_actor_loss(policy, prior_policy,
                          latent: torch.Tensor, raw_action: torch.Tensor,
                          advantage: torch.Tensor,
                          entropy_coef: float = 3e-4,
                          kl_coef: float = 0.0,
                          ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """DreamerV3-style REINFORCE actor loss.

    Replaces V4's PMPO advantage-sign-split (eq. 11) with the V3 surrogate:

        L = -E[A · log π(a|s)] - η · H[π] + β · KL(π || π_prior)

    ``raw_action`` is the canonical sample produced by
    ``policy.sample_with_raw`` (pre-tanh ``u`` for continuous, bin
    index for discrete).  Using ``raw_action`` instead of the
    post-squash action is **required** for numerically-correct log_prob
    recomputation under bfloat16 autocast — at moderate ``|u|``,
    ``tanh(u)`` rounds to ±1 in bfloat16 and ``atanh`` returns the wrong
    pre-image (manifests as ``log_prob`` ~ -1500 with std ~ 2750).
    """
    log_prob = policy.log_prob_of_raw(latent, raw_action)       # (N,)
    entropy = policy.entropy(latent)                            # (N,)

    # Reinforce surrogate. ``advantage`` is detached so the actor only
    # backprops through the policy, not the critic baseline.
    pg_loss = -(advantage.detach() * log_prob).mean()
    ent_loss = -float(entropy_coef) * entropy.mean()
    if kl_coef and kl_coef > 0.0:
        kl = policy.kl_to(prior_policy, latent)                 # (N,)
        kl_loss = float(kl_coef) * kl.mean()
        kl_diag = kl.mean().detach()
    else:
        kl_loss = torch.zeros((), device=log_prob.device,
                                dtype=log_prob.dtype)
        kl_diag = torch.zeros((), device=log_prob.device,
                                dtype=log_prob.dtype)
    total = pg_loss + ent_loss + kl_loss
    diag = {
        'actor_pg_loss': pg_loss.detach(),
        'actor_entropy_bonus': entropy.mean().detach(),
        'actor_kl_pen': kl_diag,
        'actor_logp_mean': log_prob.mean().detach(),
        'actor_logp_std': log_prob.std().detach(),
        # Mirror PMPO diag keys so existing logging / plotting works.
        'pmpo_loss': total.detach(),
        'pmpo_pos_frac': (advantage >= 0).float().mean().detach(),
        'pmpo_kl': kl_diag,
        'pmpo_logp_mean': log_prob.mean().detach(),
        'pmpo_entropy_bonus': entropy.mean().detach(),
    }
    return total, diag


# ---------------------------------------------------------------------------
# Shortcut forcing world-model loss (paper eq. 7)
# ---------------------------------------------------------------------------

def shortcut_forcing_loss(dynamics: DynamicsTransformer,
                           z_clean: torch.Tensor, action: torch.Tensor,
                           ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Shortcut-forcing (flow-matching + self-consistency) world-model loss.

    ``z_clean`` (B, T, z_dim)  — frozen tokenizer outputs (target z₁'s).
    ``action``  (B, T, A)      — actions taken before each step.

    Returns (loss, diag).

    Paper-faithful DreamerV4 shortcut-forcing objective (eq. 7), restored
    2026-05-31.  Two terms, sampled jointly per the (τ, d) grid:

      * **Flow-matching** (where ``d == d_min``): the model's x-prediction
        ``ẑ₁`` is regressed onto the clean target ``z₁``.
      * **Self-consistency / bootstrap** (where ``d > d_min``): a single
        step of size ``d`` must equal two consecutive steps of size
        ``d/2`` (stop-grad target).  This is the mechanism that makes
        *few-step* Euler integration accurate — without it the K-step
        inference rollout drifts off-manifold and the WM fails to settle
        under constant action (observed as
        ``wm_pred_converges_under_constant_action == 0``).

    Earlier this term was disabled under the assumption that APC always
    runs K = k_max steps so the bootstrap gradient is "pure noise".  That
    reasoning was wrong: the self-consistency constraint governs the
    *accuracy* of every multi-step Euler rollout (not just K' < K
    generation), so disabling it is what caused the convergence failure.

    Set ``DynamicsConfig.sf_bootstrap = False`` to fall back to the
    flow-matching-only behaviour.
    """
    cfg = dynamics.cfg
    B, T, Z = z_clean.shape
    device = z_clean.device
    dtype = z_clean.dtype
    d_min = 1.0 / cfg.k_max
    use_bootstrap = bool(getattr(cfg, 'sf_bootstrap', True))

    # Per-(B, T) sample of (τ, d) from the paper grid.  When bootstrap is
    # disabled we force d == d_min (pure flow-matching, the old behaviour).
    tau, d = sample_tau_d((B, T), cfg.k_max, device, dtype)
    if not use_bootstrap:
        d = torch.full_like(tau, d_min)

    z0 = torch.randn_like(z_clean)
    tau_b = tau.unsqueeze(-1)
    z_tilde = (1.0 - tau_b) * z0 + tau_b * z_clean

    # ---- Build the per-position x-prediction target -------------------
    # Flow positions (d == d_min) → clean z₁.
    # Bootstrap positions (d > d_min) → endpoint of two stop-grad d/2 steps,
    # converted back into the equivalent single-step x-prediction target.
    target = z_clean
    if use_bootstrap:
        is_flow = (d <= d_min + 1e-6)                               # (B, T)
        with torch.no_grad():
            # Half-step size; clamp keeps the d==d_min rows on-grid (their
            # bootstrap target is discarded by the mask below).
            d_half = (d * 0.5).clamp_min(d_min)
            inv_omt = (1.0 - tau_b).clamp_min(1e-6)                 # 1 / (1−τ)
            # First d/2 step from z_tilde at τ.
            z1_a = dynamics(z_tilde, tau, d_half, action)['z1_hat']
            v_a = (z1_a - z_tilde) / inv_omt
            z_mid = z_tilde + v_a * d_half.unsqueeze(-1)
            tau_mid = tau + d_half
            # Second d/2 step from z_mid at τ + d/2.
            inv_omt_mid = (1.0 - tau_mid.unsqueeze(-1)).clamp_min(1e-6)
            z1_b = dynamics(z_mid, tau_mid, d_half, action)['z1_hat']
            v_b = (z1_b - z_mid) / inv_omt_mid
            z_two = z_mid + v_b * d_half.unsqueeze(-1)
            # Single d-step x-prediction target that lands at z_two:
            #   z_tilde + (ẑ₁ − z_tilde) · d/(1−τ) = z_two
            target_boot = z_tilde + (z_two - z_tilde) * inv_omt / d.unsqueeze(-1).clamp_min(1e-6)
            target = torch.where(is_flow.unsqueeze(-1), z_clean, target_boot)

    out = dynamics(z_tilde, tau, d, action)
    z1_hat = out['z1_hat']                                          # (B, T, Z)
    loss_mse = (z1_hat - target).pow(2).sum(-1)                     # (B, T)

    # Per-step ramp weight (eq. 8) — paper-faithful.
    w = ramp_weight(tau)
    loss = (w * loss_mse).mean()

    # NOTE (2026-05-23, P41 sf_loss diag bug fix): do NOT include
    # 'sf_loss' in diag.  The caller (``world_model_loss``) does
    # ``losses.update(sf_diag)`` which would overwrite the live
    # ``losses['sf_loss']`` (the returned ``loss`` tensor with grad)
    # with this detached copy, breaking autograd.grad in the diag-A
    # block (observed as ``diag_grad_sf=-1.0`` + RuntimeError "element 0
    # of tensors does not require grad" through P39–P41).  Keep diag
    # empty for now; if per-loss diagnostics are needed later, use a
    # different key (e.g. ``sf_loss_val``).
    diag: Dict[str, torch.Tensor] = {}
    return loss, diag


__all__ = [
    'symlog', 'symexp', 'twohot_encode',
    'MLP', 'RMSNorm', 'SwiGLU', 'CausalAttention', 'TransformerBlock',
    'Tokenizer', 'DynamicsConfig', 'DynamicsTransformer',
    'TwohotHead', 'PolicyHead', 'ContinuousPolicyHead',
    'DreamerV4Config', 'DreamerV4',
    'sample_tau_d', 'shortcut_corrupt', 'ramp_weight',
    'shortcut_forcing_loss', 'pmpo_loss', 'reinforce_actor_loss',
]
