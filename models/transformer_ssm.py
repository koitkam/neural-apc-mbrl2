"""Transformer state-space world-model backbone (TSSM) — neural-apc-mbrl.

The "RSSM-training-structure + transformer dynamics-core" backbone: keep the
*entire* proven phased/joint training pipeline (clean steady-state seeds, noise +
DR curriculum, realistic hidden disturbances, overshoot + held-rollout losses,
critic warmup, joint mode) and swap ONLY the recurrent dynamics core (GRU + 32x32
categorical RSSM) for a **causal transformer sequence model** that performs
IN-CONTEXT SYSTEM IDENTIFICATION over the lookback window.

STATUS (2026-06-06): FUNCTIONAL + wired (build_model 'tssm' branch, dispatch,
diagnostics, collection, gpu-calib all route it as an rssm-interface backbone).
Transitions implemented with a **per-layer KV-CACHE** (``_step`` advances one
token in O(window) vs O(window^2) recompute); the cached path is validated EQUAL
to the full-sequence forward by tools/_smoke_tssm.py (max_err ~5e-7).  Custom
causal transformer (``_CausalSelfAttention`` + ``_Block``, pre-LN) supports both
a full forward (``forward_full`` — training / reference) and a cached single-step
(``forward_step`` — imagination) on the SAME weights.  REMAINING: a GPU A/B run
vs RSSM (ideally under DR), and consumer-compat for the overshoot/held-rollout
losses (currently no-op for TSSM — feat-only Markovian reconstruction loses the
transformer context; windowed attention already supervises multi-step natively).
NOTE the KV-cache assumes a single imagination rollout stays within
``max_seq_len`` (true for H<=horizon from a lookback-sized context); it does not
slide the cache, so absolute positional encoding stays exact.  ``NotImplemented``
no longer applies.  See "Wiring plan" + "Open design decisions"
below.

WHY a transformer core (vs the current SF/flow transformer):
  The existing ``world_model_type='sf_transformer'`` is a shortcut-forcing/flow
  model with its OWN training machinery (MAE tokenizer, shortcut-forcing loss) —
  it does NOT reuse the RSSM training structure, and it has NO recurrent fixed
  point (steady-state convergence 0% by construction).  This TSSM instead
  implements the SAME interface as ``RSSMDynamics`` so it is a drop-in core for
  the RSSM pipeline.  Motivation (P90 RCA): under narrow domain randomization the
  RSSM's single fixed recurrent state averages domains into a FUZZY fixed point
  (gain 0.354, 3x too small).  A transformer attends over the full lookback and
  can INFER the plant's gain/tau/dead-time from the recent (obs, action) history,
  conditioning its prediction on the identified domain -> a SHARP per-domain
  fixed point.  This is the principled route to TRUE wide-DR generalization that
  a fixed-state RSSM cannot reach (the agent then trains in imagination that
  contains the right per-domain dynamics).

INTERFACE CONTRACT (must match models.dreamer_v4_rssm.RSSMDynamics exactly so the
existing dispatch in train.py / world_model_loss / _imagination_step_rssm / the
overshoot + held-rollout losses / the WM probes all work unchanged):
  attributes : deter_dim, n_categoricals, n_classes, obs_dim, prior_net,
               post_net, pre_gru-equivalent, encoder, decoder
  state      : object with .h (..., deter_dim), .z_logits (..., K, C),
               .z (..., K, C one-hot ST), .feat ([h, z_flat]), .stoch_flat
  methods    : embed(obs)->(...,embed_dim); initial_state(B,device)->State;
               img_step(prev, prev_action, sample)->State;
               obs_step(prev, prev_action, embed, sample)->(post, prior);
               rollout_observed(obs, act, sample)->(feats, post_logits,
                                                     prior_logits, last_state);
               decode(feat)->obs
  conventions: act[:, t] drives the transition INTO obs[:, t] (contemporaneous-
               action: feat[t] has seen a_t).  ``feat = [h, z_flat]`` with
               feat_dim = deter_dim + n_categoricals*n_classes so the V4
               reward/value/policy heads (built on feat_dim) are reused unchanged.

CORE ARCHITECTURE (the transition methods to implement):
  Token at step t = proj([ z_{t-1}_flat ; a_t ]) + (optional obs-embed for the
  posterior) + positional/time encoding.  A causal Transformer encoder over the
  running token sequence produces a per-step hidden ``g_t``.  Define the
  RSSM-compatible deterministic state ``h_t := g_t`` (so deter_dim = d_model).
  Prior head: ``prior_net(h_t) -> (K, C) categorical logits`` (reuse the RSSM
  ``_CategoricalLatent`` head verbatim).  Posterior head: ``post_net([h_t,
  embed_t]) -> (K, C)``.  ``z_t`` = straight-through one-hot sample (sample=True
  for training prior grad — same ST requirement the overshoot/held-rollout losses
  rely on).  decode(feat) reconstructs obs.

  img_step (imagination, the PERF-CRITICAL path): advance ONE step under a held/
  given action with NO obs.  Naive = re-run the transformer over the whole token
  history each step (O(T^2) over H imagination steps).  CORRECT + FAST = maintain
  a KV-CACHE in the State object: each img_step appends one token, attends it
  against cached keys/values -> O(T) per step.  *** This KV-cache is the main
  reason this is a scaffold, not a finished impl — it must be implemented +
  numerically validated (img_step result == teacher-forced rollout on the same
  actions) before any training run. ***

WIRING PLAN (when implemented):
  1. models/dreamer_v4.py DreamerV4.__init__: add ``elif world_model_type ==
     'tssm': self.dynamics = TransformerSSMDynamics(tssm_cfg)`` alongside the
     RSSM branch; ensure parameters_world() includes it (it already globs
     self.dynamics.parameters()).
  2. training/train.py world_model_loss / imagination_step dispatch: the RSSM
     branch checks ``world_model_type == 'rssm'``; widen to
     ``in ('rssm', 'tssm')`` since the interface is identical (feat=[h,z_flat],
     rollout_observed/img_step/decode).  Verify _wm_latent_overshoot_loss and
     _wm_held_rollout_stationarity_loss (which read rssm.deter_dim /
     n_categoricals / img_step) work unchanged — they will, by contract.
  3. ENV_OVERRIDES: DREAMER_WORLD_MODEL_TYPE already exists; add TSSM dims
     (DREAMER_TSSM_{D_MODEL,N_LAYERS,N_HEADS}).
  4. compile + the inference/export_onnx path (RSSM ONNX is not implemented;
     TSSM ONNX is a separate task).

OPEN DESIGN DECISIONS (resolve before implementing):
  - deter_dim == d_model couples the feature size to the transformer width; the
    V4 heads are built on feat_dim = deter_dim + K*C, so picking d_model sets the
    head input size (fine, but note it for model-size BO).
  - KL free-bits: reuse rssm_kl_loss verbatim (post vs prior logits) — no change.
  - Positional encoding over the lookback: absolute vs rotary; rotary preferred
    for length generalization (imagination H may exceed training seq_len).
  - Imagination determinism: RSSM uses sample=False (mode) under
    DREAMER_RSSM_IMAG_LATENT_MODE; keep the same flag semantics.
  - Numerical-equivalence test (MUST pass): img_step rolled K steps under the
    SAME actions as a teacher-forced rollout must match within tol (proves the
    KV-cache path == the full-attention path).

Until the transition methods' KV-cache + dispatch wiring are done, ``build_model``
must NOT dispatch to this class.  STATUS: transitions IMPLEMENTED (naive windowed
recompute, CPU-tested via tools/_smoke_tssm.py); KV-cache + consumer-compat +
dispatch remain (see top-of-file STATUS).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Reuse the proven RSSM building blocks so the categorical latent + KL are
# bit-for-bit identical to the default backbone (only the dynamics core changes).
from models.dreamer_v4_rssm import _CategoricalLatent


@dataclass
class TransformerSSMConfig:
    """Config for the transformer dynamics core.  ``deter_dim`` == d_model."""
    obs_dim: int
    action_dim: int
    deter_dim: int = 512          # = transformer d_model (h_t := g_t)
    n_categoricals: int = 32      # match RSSM paper default
    n_classes: int = 32
    embed_dim: int = 256
    n_layers: int = 4
    n_heads: int = 8
    ffn_mult: int = 4
    dropout: float = 0.0
    unimix: float = 0.01          # match RSSM categorical mixing
    max_seq_len: int = 256        # context window cap (>= lookback + horizon)
    # DV-as-input (Option B, 2026-06-07) — mirror of RSSMConfig: measured DV
    # channels (at ``dv_indices`` in the obs vector) become an exogenous token
    # input instead of being predicted forward.  ``dv_dim = 0`` = paper default.
    dv_dim: int = 0
    dv_indices: Tuple[int, ...] = ()
    # DV→decoder+heads FEEDFORWARD (2026-06-19, p129 RCA) — mirror of RSSMConfig.
    # When True (and dv_dim>0) the measured DV is appended to ``feat`` AND fed
    # directly into the decoder so the CV reconstruction ``g(h, z, dv)`` has a
    # DIRECT exogenous-DV path that skips the categorical bottleneck (the DV gain
    # dies in the autoencoder, p129), and the heads see the disturbance.
    dv_feedforward: bool = True
    # Neural Kalman filter / disturbance observer (DOB) — mirror of RSSMConfig.
    # See models/dreamer_v4_rssm.RSSMConfig + docs/architecture.md §3.  Shared
    # feat->decode interface, so the observer math is identical to the RSSM.
    dob_enabled: bool = False
    cv_indices: Tuple[int, ...] = ()
    dob_decay_init: float = 3.0
    dob_gain_init: float = -2.2
    # Continuous gain+disturbance latent (2026-06-22) — mirror of RSSMConfig.
    # A Gaussian latent alongside the categorical for the precision-critical
    # gain (supervised, in-context DR) + unmeasured disturbance (amortized
    # Kalman).  cont_gain_dim==cont_dist_dim==0 ⇒ pre-continuous-latent model.
    cont_gain_dim: int = 0
    cont_dist_dim: int = 0
    cont_min_std: float = 0.1
    cont_max_std: float = 2.0


@dataclass
class TSSMState:
    """Duck-compatible with RSSMState (.h, .z_logits, .z, .feat, .stoch_flat)
    PLUS the transformer continuation context: a per-layer KV-CACHE and the
    absolute position ``pos``, so ``img_step`` advances in O(window) (attend the
    new token against the cached K/V) instead of O(window²) recompute.
    ``kv_cache=None`` => no history (feat-only reconstruction by a Markovian
    consumer; the next step starts a fresh single-token context).
    """
    h: torch.Tensor             # (..., deter_dim) transformer output at step t
    z_logits: torch.Tensor      # (..., n_categoricals, n_classes)
    z: torch.Tensor             # (..., n_categoricals, n_classes) one-hot (ST)
    # per-layer (k, v) each (B, n_heads, pos, head_dim); None = empty context.
    kv_cache: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None
    pos: int = 0                # number of tokens already in the cache
    d: Optional[torch.Tensor] = None  # (..., n_cv) DOB disturbance state (None=off)
    dv: Optional[torch.Tensor] = None  # (..., dv_dim) exogenous DV feedforward (None=off)

    @property
    def stoch_flat(self) -> torch.Tensor:
        return self.z.flatten(start_dim=-2)

    @property
    def feat(self) -> torch.Tensor:
        # DV feedforward (2026-06-19) + Scope 2 DOB (2026-06-11) — mirror of
        # RSSMState.feat: ``[h, z_flat, (dv), (d.detach())]``.  ``dv`` sits right
        # after the latent core (the decoder reads ``[h, z, dv]`` as a
        # contiguous front slice — the direct DV path skipping the categorical
        # bottleneck — and the heads condition on the measured DV); the DOB
        # ``d`` is appended last and DETACHED (heads see it; the decoder slices
        # it off).  Both None ⇒ feat = [h, z_flat] (byte-identical paper TSSM).
        parts = [self.h, self.stoch_flat]
        if self.dv is not None:
            parts.append(self.dv)
        if self.d is not None:
            parts.append(self.d.detach())
        return torch.cat(parts, dim=-1)


class _CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with an optional KV-cache.

    ``forward_full`` (training / reference): standard causal attention over a
    full (B, S, d) sequence.  ``forward_step`` (imagination): attend ONE new
    token against the cached past K/V (+ its own), returning the updated cache.
    Both share the SAME weights so the cached path is provably equal to the
    full recompute (validated by tools/_smoke_tssm.py equivalence test).
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = float(dropout)

    def _split(self, t: torch.Tensor) -> torch.Tensor:
        B, S, _ = t.shape
        return t.reshape(B, S, self.n_heads, self.head_dim).transpose(1, 2)

    def forward_full(self, x: torch.Tensor) -> torch.Tensor:
        B, S, d = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = self._split(q), self._split(k), self._split(v)
        p = self.dropout if self.training else 0.0
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                             dropout_p=p)
        out = out.transpose(1, 2).reshape(B, S, d)
        return self.proj(out)

    def forward_step(self, x_t: torch.Tensor,
                     cache: Optional[Tuple[torch.Tensor, torch.Tensor]]
                     ) -> Tuple[torch.Tensor,
                                Tuple[torch.Tensor, torch.Tensor]]:
        B, _, d = x_t.shape                              # x_t: (B, 1, d)
        q, k, v = self.qkv(x_t).chunk(3, dim=-1)
        q, k, v = self._split(q), self._split(k), self._split(v)  # (B,H,1,hd)
        if cache is not None:
            pk, pv = cache
            k = torch.cat([pk, k], dim=2)
            v = torch.cat([pv, v], dim=2)
        new_cache = (k, v)
        # q (the single new token) attends to ALL of k (past + self) => exactly
        # the causal pattern for the last position.  No mask needed.
        out = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        out = out.transpose(1, 2).reshape(B, 1, d)
        return self.proj(out), new_cache


class _Block(nn.Module):
    """Pre-LayerNorm transformer block (matches norm_first=True semantics)."""

    def __init__(self, d_model: int, n_heads: int, ffn_mult: int,
                 dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = _CausalSelfAttention(d_model, n_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * ffn_mult), nn.GELU(),
            nn.Linear(d_model * ffn_mult, d_model))

    def forward_full(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn.forward_full(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x

    def forward_step(self, x_t: torch.Tensor,
                     cache: Optional[Tuple[torch.Tensor, torch.Tensor]]
                     ) -> Tuple[torch.Tensor,
                                Tuple[torch.Tensor, torch.Tensor]]:
        a, new_cache = self.attn.forward_step(self.norm1(x_t), cache)
        x_t = x_t + a
        x_t = x_t + self.ff(self.norm2(x_t))
        return x_t, new_cache


def _sinusoidal_pos(n: int, d: int, device, dtype) -> torch.Tensor:
    """(n, d) sinusoidal positional encoding (Vaswani et al.)."""
    pos = torch.arange(n, device=device, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(torch.arange(0, d, 2, device=device, dtype=torch.float32)
                    * (-math.log(10000.0) / d))
    pe = torch.zeros(n, d, device=device, dtype=torch.float32)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe.to(dtype)


class TransformerSSMDynamics(nn.Module):
    """Causal-transformer dynamics core implementing the RSSMDynamics interface.

    Naive (recompute) transitions — correct + CPU-tested.  KV-cache is a future
    pure-speed optimization gated by the equivalence test (see module docstring).
    """

    def __init__(self, cfg: TransformerSSMConfig):
        super().__init__()
        self.cfg = cfg
        self.obs_dim = int(cfg.obs_dim)
        self.action_dim = int(cfg.action_dim)
        self.deter_dim = int(cfg.deter_dim)
        self.n_categoricals = int(cfg.n_categoricals)
        self.n_classes = int(cfg.n_classes)
        self.embed_dim = int(cfg.embed_dim)
        self.unimix = float(cfg.unimix)
        self.max_seq_len = int(cfg.max_seq_len)
        self.stoch_flat_dim = self.n_categoricals * self.n_classes
        # DV-as-input (Option B): exogenous measured-DV channels appended to the
        # token input; ``dv_index_t`` selects them out of the obs vector.
        self.dv_dim = int(getattr(cfg, 'dv_dim', 0) or 0)
        self.register_buffer(
            'dv_index_t',
            torch.tensor(list(getattr(cfg, 'dv_indices', ()) or ()),
                         dtype=torch.long),
            persistent=False)
        # DV→decoder+heads feedforward (2026-06-19) — mirror of RSSMDynamics.
        self.dv_feedforward = bool(getattr(cfg, 'dv_feedforward', True)) \
            and self.dv_dim > 0
        self._dv_feed_dim = self.dv_dim if self.dv_feedforward else 0

        # ----- shared, low-risk pieces (real implementations) -----
        self.encoder = nn.Sequential(
            nn.Linear(self.obs_dim, cfg.embed_dim), nn.SiLU(),
            nn.Linear(cfg.embed_dim, cfg.embed_dim),
        )
        self.decoder = nn.Sequential(
            # DV feedforward + Scope 2: decode ``[h, z, (dv)]`` (the DV gives the
            # CV reconstruction a direct ∂CV/∂dv path skipping the categorical
            # bottleneck); the DOB d-tail is sliced off in ``decode`` and
            # re-added via ``apply_dob``.
            nn.Linear(self.deter_dim + self.stoch_flat_dim + self._dv_feed_dim,
                      cfg.embed_dim),
            nn.SiLU(),
            nn.Linear(cfg.embed_dim, self.obs_dim),
        )
        # Direct DV→obs FEEDFORWARD SKIP (2026-06-20, p132 RCA; mirror of
        # RSSMDynamics).  The single measured-DV channel is diluted ~1/(d_model)
        # inside the decoder MLP, so the DV→CV gain still dies in the autoencoder
        # (p132 DV real→post < MV).  This zero-init linear skip gives the DV a
        # clean direct path to the reconstructed obs (decode + W·dv), bypassing
        # the dilution + the categorical bottleneck.  Zero-init ⇒ exact no-op at
        # start; learns ∂CV/∂dv from the residual.
        self.dv_skip = (nn.Linear(self._dv_feed_dim, self.obs_dim)
                        if self._dv_feed_dim > 0 else None)
        if self.dv_skip is not None:
            nn.init.zeros_(self.dv_skip.weight)
            nn.init.zeros_(self.dv_skip.bias)
        # Token projection: [z_{t-1}_flat ; a_t ; (dv_t)] -> d_model.
        self.token_proj = nn.Linear(
            self.stoch_flat_dim + self.action_dim + self.dv_dim,
            self.deter_dim)
        # Causal transformer (custom blocks: support full + KV-cached step).
        self.n_heads = int(cfg.n_heads)
        self.blocks = nn.ModuleList([
            _Block(self.deter_dim, cfg.n_heads, cfg.ffn_mult, cfg.dropout)
            for _ in range(cfg.n_layers)])
        # Categorical latent heads (reuse RSSM block: prior from h, post from
        # [h, embed]).  prior_net is read by the smoke tests + the overshoot /
        # held-rollout losses, so the attribute name MUST match the RSSM.
        self.prior_net = _CategoricalLatent(
            self.deter_dim, self.n_categoricals, self.n_classes,
            unimix=cfg.unimix)
        self.post_net = _CategoricalLatent(
            self.deter_dim + cfg.embed_dim, self.n_categoricals,
            self.n_classes, unimix=cfg.unimix)

        # ----- Neural Kalman filter / disturbance observer (DOB) -----
        # Identical to RSSMDynamics: a first-order learned observer (per-CV A,K
        # in (0,1)) on the one-step prediction residual, added to the decoded CV.
        self.register_buffer(
            'cv_index_t',
            torch.tensor(list(getattr(cfg, 'cv_indices', ()) or ()),
                         dtype=torch.long),
            persistent=False)
        self.n_cv = int(self.cv_index_t.numel())
        self.dob_enabled = bool(getattr(cfg, 'dob_enabled', False)) and self.n_cv > 0
        # Curriculum Stage-1 suppression (2026-06-12, mirror of RSSMDynamics):
        # ``dob_active=False`` forces d_t==0 (clean-plant identification stage)
        # while keeping a zero d-tail in feat so head dims stay constant.
        self.dob_active = True
        if self.dob_enabled:
            self.dob_log_decay = nn.Parameter(torch.full(
                (self.n_cv,), float(getattr(cfg, 'dob_decay_init', 3.0))))
            self.dob_log_gain = nn.Parameter(torch.full(
                (self.n_cv,), float(getattr(cfg, 'dob_gain_init', -2.2))))

    @property
    def feat_dim(self) -> int:
        # Mirror of RSSMDynamics.feat_dim: head-facing feature = latent core +
        # DV feedforward (when on) + DOB ``d`` (one scalar per CV).  The decoder
        # reads ``[h, z, (dv)]`` (see ``_decode_in_dim`` / ``decode``).
        core = self.deter_dim + self.stoch_flat_dim
        return (core + self._dv_feed_dim
                + (self.n_cv if getattr(self, 'dob_enabled', False) else 0))

    @property
    def _decode_in_dim(self) -> int:
        # Width of the decoder input slice = latent core + DV feedforward.
        return self.deter_dim + self.stoch_flat_dim + self._dv_feed_dim

    # ----- DOB helpers (mirror RSSMDynamics) -----
    def dob_decay(self) -> torch.Tensor:
        return torch.sigmoid(self.dob_log_decay)

    def dob_gain(self) -> torch.Tensor:
        return torch.sigmoid(self.dob_log_gain)

    def apply_dob(self, decoded: torch.Tensor,
                  d: Optional[torch.Tensor]) -> torch.Tensor:
        if not self.dob_enabled or d is None:
            return decoded
        out = decoded.clone()
        out.index_add_(-1, self.cv_index_t, d.to(out.dtype))
        return out

    # ----- shared pieces (real) -----
    def embed(self, obs: torch.Tensor) -> torch.Tensor:
        return self.encoder(obs)

    def decode(self, feat: torch.Tensor) -> torch.Tensor:
        # DV feedforward + Scope 2 (mirror of RSSMDynamics.decode): decode
        # ``[h, z, (dv)]`` (the contiguous front slice); any DOB d-tail beyond
        # it is sliced off (re-added by ``apply_dob``).  No-op slice when both
        # DV-feedforward and the DOB are off.
        x = feat[..., :self._decode_in_dim]
        out = self.decoder(x)
        if self.dv_skip is not None:
            core = self.deter_dim + self.stoch_flat_dim
            dv = feat[..., core:core + self._dv_feed_dim]
            out = out + self.dv_skip(dv)
        return out

    def initial_state(self, batch_size: int,
                      device: torch.device) -> TSSMState:
        h = torch.zeros(batch_size, self.deter_dim, device=device)
        z_logits = torch.zeros(batch_size, self.n_categoricals,
                               self.n_classes, device=device)
        z = torch.zeros_like(z_logits)
        z[..., 0] = 1.0
        d = (torch.zeros(batch_size, self.n_cv, device=device)
             if self.dob_enabled else None)
        dv = (torch.zeros(batch_size, self.dv_dim, device=device)
              if self.dv_feedforward else None)
        return TSSMState(h=h, z_logits=z_logits, z=z, kv_cache=None, pos=0, d=d,
                         dv=dv)

    # ----- internal: token build + causal encode -----
    def _build_token(self, z: torch.Tensor,
                     action: torch.Tensor,
                     dv: Optional[torch.Tensor] = None) -> torch.Tensor:
        """token = proj([z_flat ; action ; (dv)]) -> (B, d_model)."""
        z_flat = z.flatten(start_dim=-2)
        if self.dv_dim > 0:
            if dv is None:
                dv = torch.zeros(action.shape[0], self.dv_dim,
                                 device=action.device, dtype=action.dtype)
            return self.token_proj(torch.cat([z_flat, action, dv], dim=-1))
        return self.token_proj(torch.cat([z_flat, action], dim=-1))

    def _encode_window(self, window: torch.Tensor) -> torch.Tensor:
        """Full-sequence causal forward over (B, S, d_model) -> (B, S, d_model).
        The reference path (training-free) the KV-cached step is validated
        against; also reused by callers that have the whole token window."""
        S = window.shape[1]
        pe = _sinusoidal_pos(S, self.deter_dim, window.device, window.dtype)
        x = window + pe.unsqueeze(0)
        for blk in self.blocks:
            x = blk.forward_full(x)
        return x

    def _step(self, token: torch.Tensor,
              kv_cache: Optional[List[Tuple[torch.Tensor, torch.Tensor]]],
              pos: int
              ) -> Tuple[torch.Tensor,
                         List[Tuple[torch.Tensor, torch.Tensor]]]:
        """KV-cached single-token advance.  ``token`` (B, d_model) at absolute
        position ``pos``; returns ``(h (B, d_model), new_kv_cache)``.  O(window)
        instead of the O(window²) full recompute."""
        pe = _sinusoidal_pos(pos + 1, self.deter_dim, token.device,
                             token.dtype)[pos]            # (d_model,)
        x = (token + pe.unsqueeze(0)).unsqueeze(1)        # (B, 1, d)
        new_cache: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for li, blk in enumerate(self.blocks):
            layer_cache = None if kv_cache is None else kv_cache[li]
            x, lc = blk.forward_step(x, layer_cache)
            new_cache.append(lc)
        return x[:, 0], new_cache

    # ----- transitions -----
    def img_step(self, prev: TSSMState, prev_action: torch.Tensor,
                 dv: Optional[torch.Tensor] = None,
                 sample: bool = True) -> TSSMState:
        """Imagined (prior-only) step: build the token from (prev.z, action,
        dv), advance the KV-cached transformer ONE step, read the prior off the
        new position.  ``dv`` (B, dv_dim) is the exogenous measured-DV input
        when DV-as-input is on; ``None`` -> zeros.  ``kv_cache=None`` (feat-only
        reconstruction by a Markovian consumer) starts a fresh context."""
        # DV feedforward: materialise the (possibly zero-filled) DV so it can be
        # both fed to the token AND stored in the state (decoder + heads read it).
        if self.dv_feedforward and dv is None:
            dv = torch.zeros(prev_action.shape[0], self.dv_dim,
                             device=prev_action.device, dtype=prev_action.dtype)
        token = self._build_token(prev.z, prev_action, dv)
        cache = getattr(prev, 'kv_cache', None)
        pos = int(getattr(prev, 'pos', 0) or 0)
        h, new_cache = self._step(token, cache, pos)
        z_logits, z = self.prior_net(h, sample=sample)
        d_new = (self.dob_decay() * prev.d
                 if (self.dob_enabled and prev.d is not None) else prev.d)
        dv_new = dv if self.dv_feedforward else None
        return TSSMState(h=h, z_logits=z_logits, z=z,
                         kv_cache=new_cache, pos=pos + 1, d=d_new, dv=dv_new)

    def obs_step(self, prev: TSSMState, prev_action: torch.Tensor,
                 embed: torch.Tensor, dv: Optional[torch.Tensor] = None,
                 sample: bool = True, obs: Optional[torch.Tensor] = None
                 ) -> Tuple[TSSMState, TSSMState]:
        """Observation step -> (posterior, prior).  Prior is needed for KL; both
        share ``h``; the posterior conditions on the obs embedding and is the z
        carried forward (with the prior's KV-cache + position).  DOB: when
        ``obs`` is supplied the posterior carries the corrected disturbance
        state ``d_t = A*d_{t-1} + K*nu`` (innovation on the prior forecast),
        identical to RSSMDynamics."""
        prior = self.img_step(prev, prev_action, dv=dv, sample=sample)
        post_in = torch.cat([prior.h, embed], dim=-1)
        post_logits, post_z = self.post_net(post_in, sample=sample)
        d_post = prior.d
        if self.dob_enabled and obs is not None and prior.d is not None:
            cv_pred = (self.decode(prior.feat).index_select(-1, self.cv_index_t)
                       + prior.d)
            cv_obs = obs.index_select(-1, self.cv_index_t)
            nu = cv_obs - cv_pred
            d_post = prior.d + self.dob_gain() * nu
        # Posterior inherits the prior's exogenous DV feedforward (same measured
        # DV drove both) so ``post.feat`` / ``decode(post.feat)`` expose it.
        post = TSSMState(h=prior.h, z_logits=post_logits, z=post_z,
                         kv_cache=prior.kv_cache, pos=prior.pos, d=d_post,
                         dv=prior.dv)
        return post, prior

    def rollout_observed(self, obs: torch.Tensor, act: torch.Tensor,
                         sample: bool = True
                         ) -> Tuple[torch.Tensor, torch.Tensor,
                                    torch.Tensor, TSSMState]:
        """Teacher-forced posterior rollout over (B, T, *).  ``act[:, t]`` drives
        the transition INTO ``obs[:, t]`` (contemporaneous-action convention, as
        in RSSMDynamics).  Returns (feats, post_logits, prior_logits,
        last_state, ds) with shapes (B, T, F), (B, T, K, C), (B, T, K, C), the
        last state, and ds (B, T, n_cv) = per-step DOB estimate (None=off)."""
        B, T = obs.shape[:2]
        device = obs.device
        embeds = self.embed(obs)                         # (B, T, embed_dim)
        dvs = (obs.index_select(-1, self.dv_index_t)
               if self.dv_dim > 0 else None)             # (B, T, dv_dim) | None
        state = self.initial_state(B, device)
        core = self.deter_dim + self.stoch_flat_dim
        dec_in = self._decode_in_dim                     # core (+ dv feedforward)
        feats_l, post_l, prior_l, prior_core_l = [], [], [], []
        for t in range(T):
            dv_t = dvs[:, t] if dvs is not None else None
            # COMPILE-EFFICIENT DOB (2026-06-12, mirror of RSSMDynamics): run the
            # (h, z) recurrence WITHOUT the per-step DOB correction (``obs=None``)
            # — ``d`` does not affect h/z (the token is built from z+action+dv,
            # post_net from h+embed) — so the per-step prior decode for the
            # innovation is hoisted OUT of the loop and done ONCE, batched, below.
            # Keeps the compiled graph free of T× decoder MLPs.  Math identical
            # (first-order linear filter); validated in tools/_smoke_dob.py.
            post, prior = self.obs_step(state, act[:, t], embeds[:, t],
                                        dv=dv_t, sample=sample, obs=None)
            state = post
            feats_l.append(post.feat[..., :dec_in])      # decoder feat [h, z, (dv)]
            post_l.append(post.z_logits)
            prior_l.append(prior.z_logits)
            if self.dob_enabled:
                prior_core_l.append(prior.feat[..., :dec_in])
        post_core = torch.stack(feats_l, dim=1)          # (B, T, dec_in) = [h,z,(dv)]
        post_logits = torch.stack(post_l, dim=1)
        prior_logits = torch.stack(prior_l, dim=1)
        ds = None
        if self.dob_enabled:
            if self.dob_active:
                # ONE batched prior decode → CV forecast base, then the scalar per-CV
                # Kalman filter: d_t = (1−K)·A·d_{t-1} + K·(CV_obs − base).
                prior_core = torch.stack(prior_core_l, dim=1)         # (B, T, dec_in)
                base = self.decode(prior_core).index_select(-1, self.cv_index_t)
                cv_obs = obs.index_select(-1, self.cv_index_t)        # (B, T, n_cv)
                A = self.dob_decay(); K = self.dob_gain()             # (n_cv,)
                u = K * (cv_obs - base)                               # drive (B,T,n_cv)
                coef = (1.0 - K) * A                                  # (n_cv,)
                d_prev = torch.zeros(B, self.n_cv, device=device, dtype=post_core.dtype)
                ds_l = []
                for t in range(T):
                    d_prev = coef * d_prev + u[:, t]
                    ds_l.append(d_prev)
                ds = torch.stack(ds_l, dim=1)                         # (B, T, n_cv)
            else:
                # Stage-1 suppression: d_t ≡ 0 (force g to explain all CV motion).
                ds = torch.zeros(B, T, self.n_cv, device=device,
                                 dtype=post_core.dtype)
            feats = torch.cat([post_core, ds.detach()], dim=-1)
            state = TSSMState(h=state.h, z_logits=state.z_logits, z=state.z,
                              kv_cache=state.kv_cache, pos=state.pos,
                              d=ds[:, -1], dv=state.dv)
        else:
            feats = post_core
        return feats, post_logits, prior_logits, state, ds
