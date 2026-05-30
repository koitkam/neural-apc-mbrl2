"""Recurrent State-Space Model (RSSM) world-model for DreamerV4.

Drop-in alternative to the SF-transformer dynamics, selected via
``TrainConfig.world_model_type == 'rssm'`` (the new default as of
2026-05-30).  Motivation: across P64/P66/P67 the SF-transformer's
``wm_pred_converges_under_constant_action`` was pinned at **0.0** — it
has no recurrent state able to hold an equilibrium (sliding-window
attention + a freshly resampled ``z0 ~ N(0, I)`` each call → a
non-contractive map with no attractor).  Every downstream critic /
reward-side fix (return-scale clamp, reward-tail clip, potential-based
shaping, replay-grounded critic anchor, clean-τ) therefore failed,
because the bootstrap-cascade is a *symptom* of WM imagination
divergence, not a critic pathology.

The RSSM's deterministic GRU core ``h_t = f(h_{t-1}, z_{t-1}, a_{t-1})``
*can* learn a contractive fixed point ``h* = f(h*, z*, a)`` under a held
action — the structural property the SF-transformer lacks.

Integration philosophy (keep V4's proven machinery, swap only the WM):
  * Heads (reward / value / target_value / policy / prior_policy) stay as
    V4's ``TwohotHead`` / ``ContinuousPolicyHead``, built with
    ``in_dim = feat_dim`` (= deter_dim + n_categoricals*n_classes).  This
    preserves V4's twohot+symlog reward/value, return-scale EMA, PMPO /
    REINFORCE actor losses, MTP heads, and the whole phase/auto-tune/
    validation pipeline unchanged.
  * The RSSM provides ONLY: encoder (obs → embed), decoder (feat → obs
    recon), GRU + pre-GRU projection, prior network p(z'|h'), posterior
    network q(z'|h',x').
  * No reward/continue head inside the RSSM.  Reward is V4's TwohotHead
    (trained in P1/P2 via reward-MTP exactly as the SF path).  There is
    NO continue head: the APC control task is non-terminating
    (``cont ≡ 1``), so a Bernoulli continue predictor would be a constant
    — dropping it removes a degenerate loss term and matches the
    workflow's continuing-control objective.

Obs space (documented deviation from the paper's symlog-on-raw):
  V4 already z-scores observations upstream (``APCEnv._normalize_obs``),
  so the buffer stores well-scaled obs.  The RSSM encoder/decoder operate
  directly in that normalized space — applying the paper's symlog on top
  of an already-z-scored signal would be a redundant second non-linearity.
  This is consistent with V4's existing tokenizer (which also consumes
  normalized obs without symlog) and keeps the imagination diagnostic
  (which compares decoded vs real normalized obs) on a single scale.

Paper-faithful details retained from DreamerV3 (Hafner et al. 2023, §3-4):
  * 32 categoricals × 32 classes straight-through one-hot latent.
  * 1% unimix on the categorical (avoids vanishing gradients on
    saturated logits).
  * KL-balanced free-bits loss: dyn term (post detached) weight 0.5,
    repr term (prior detached) weight 0.1, free-bits floor 1.0 nat on the
    batch-and-time mean of the K-summed KL.
  * Zero-init is applied to the V4 value/policy heads only (handled in
    those modules), NOT to the decoder / prior / posterior — zero-init on
    those creates the init gradient-deadlock documented in the source
    repo (decoder predicts 0 → dL/dfeat=0 freezes encoder/RSSM, and
    prior=post=uniform → KL=0 → free-bits floor kills the KL gradient).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import OneHotCategorical


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class RSSMConfig:
    obs_dim: int
    action_dim: int
    deter_dim: int = 512            # paper Medium GRU hidden size
    n_categoricals: int = 32        # paper
    n_classes: int = 32             # paper
    embed_dim: int = 256
    hidden_dim: int = 256
    unimix: float = 0.01            # paper 1% uniform mixture


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class _MLP(nn.Module):
    """LayerNorm + SiLU MLP (DreamerV3 reference block)."""

    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 256,
                 num_layers: int = 3, layernorm: bool = True):
        super().__init__()
        layers = []
        d = in_dim
        for _ in range(max(1, num_layers)):
            layers.append(nn.Linear(d, hidden_dim))
            if layernorm:
                layers.append(nn.LayerNorm(hidden_dim, eps=1e-6))
            layers.append(nn.SiLU())
            d = hidden_dim
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class RSSMState:
    h: torch.Tensor             # (..., deter_dim) deterministic recurrent state
    z_logits: torch.Tensor      # (..., n_categoricals, n_classes)
    z: torch.Tensor             # (..., n_categoricals, n_classes) one-hot (ST grad)

    @property
    def stoch_flat(self) -> torch.Tensor:
        return self.z.flatten(start_dim=-2)

    @property
    def feat(self) -> torch.Tensor:
        return torch.cat([self.h, self.stoch_flat], dim=-1)


class _CategoricalLatent(nn.Module):
    """Feature → (n_categoricals × n_classes) categorical logits.

    Straight-through one-hot sample with a ``unimix`` uniform mixture.
    """

    def __init__(self, in_dim: int, n_categoricals: int, n_classes: int,
                 hidden_dim: int = 256, unimix: float = 0.01):
        super().__init__()
        self.n_categoricals = int(n_categoricals)
        self.n_classes = int(n_classes)
        self.unimix = float(unimix)
        self.net = _MLP(in_dim, n_categoricals * n_classes, hidden_dim,
                        num_layers=2)

    def forward(self, x: torch.Tensor, sample: bool = True
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.net(x).view(*x.shape[:-1], self.n_categoricals,
                                   self.n_classes)
        # Unimix: (1-u)·softmax + u·uniform, re-expressed as logits.
        probs = F.softmax(logits, dim=-1)
        if self.unimix > 0.0:
            probs = (1.0 - self.unimix) * probs + self.unimix / self.n_classes
        logits = torch.log(probs.clamp(min=1e-8))
        if sample:
            dist = OneHotCategorical(logits=logits)
            sample_oh = dist.sample()
            # Straight-through: forward = hard sample, gradient = probs.
            sample_st = sample_oh + probs - probs.detach()
        else:
            idx = logits.argmax(dim=-1)
            sample_st = F.one_hot(idx, num_classes=self.n_classes).to(
                logits.dtype)
        return logits, sample_st


# ---------------------------------------------------------------------------
# RSSM dynamics
# ---------------------------------------------------------------------------

class RSSMDynamics(nn.Module):
    """DreamerV3 RSSM core adapted as a DreamerV4 world-model backbone."""

    def __init__(self, cfg: RSSMConfig):
        super().__init__()
        self.cfg = cfg
        self.obs_dim = int(cfg.obs_dim)
        self.action_dim = int(cfg.action_dim)
        self.deter_dim = int(cfg.deter_dim)
        self.n_categoricals = int(cfg.n_categoricals)
        self.n_classes = int(cfg.n_classes)
        self.embed_dim = int(cfg.embed_dim)
        self.hidden_dim = int(cfg.hidden_dim)
        self.stoch_flat_dim = self.n_categoricals * self.n_classes

        # Encoder: obs → per-frame embedding.
        self.encoder = _MLP(self.obs_dim, self.embed_dim,
                            hidden_dim=self.hidden_dim, num_layers=3)
        # Decoder: [h, z] → reconstructed (normalized) obs.
        self.decoder = _MLP(self.feat_dim, self.obs_dim,
                            hidden_dim=self.hidden_dim, num_layers=3)
        # Recurrent dynamics: pre-GRU projection then GRUCell.
        self.pre_gru = _MLP(self.stoch_flat_dim + self.action_dim,
                            self.stoch_flat_dim + self.action_dim,
                            hidden_dim=self.hidden_dim, num_layers=1)
        self.gru = nn.GRUCell(self.stoch_flat_dim + self.action_dim,
                              self.deter_dim)
        # Prior p(z'|h') and posterior q(z'|h', embed).
        self.prior_net = _CategoricalLatent(
            self.deter_dim, self.n_categoricals, self.n_classes,
            hidden_dim=self.hidden_dim, unimix=cfg.unimix)
        self.post_net = _CategoricalLatent(
            self.deter_dim + self.embed_dim, self.n_categoricals,
            self.n_classes, hidden_dim=self.hidden_dim, unimix=cfg.unimix)

    @property
    def feat_dim(self) -> int:
        return self.deter_dim + self.stoch_flat_dim

    # ----- embedding ----------------------------------------------------
    def embed(self, obs: torch.Tensor) -> torch.Tensor:
        """obs: (..., obs_dim) → (..., embed_dim).  Normalized-space input."""
        return self.encoder(obs)

    def initial_state(self, batch_size: int,
                      device: torch.device) -> RSSMState:
        h = torch.zeros(batch_size, self.deter_dim, device=device)
        z_logits = torch.zeros(batch_size, self.n_categoricals,
                               self.n_classes, device=device)
        z = torch.zeros_like(z_logits)
        z[..., 0] = 1.0  # arbitrary valid one-hot
        return RSSMState(h=h, z_logits=z_logits, z=z)

    # ----- transitions --------------------------------------------------
    def img_step(self, prev: RSSMState, prev_action: torch.Tensor,
                 sample: bool = True) -> RSSMState:
        """Imagined (prior-only) step: advance the state with no obs."""
        x = torch.cat([prev.stoch_flat, prev_action], dim=-1)
        x = self.pre_gru(x)
        h = self.gru(x, prev.h)
        z_logits, z = self.prior_net(h, sample=sample)
        return RSSMState(h=h, z_logits=z_logits, z=z)

    def obs_step(self, prev: RSSMState, prev_action: torch.Tensor,
                 embed: torch.Tensor, sample: bool = True
                 ) -> Tuple[RSSMState, RSSMState]:
        """Observation step → (posterior, prior).  Prior is needed for KL."""
        prior = self.img_step(prev, prev_action, sample=sample)
        post_in = torch.cat([prior.h, embed], dim=-1)
        post_logits, post_z = self.post_net(post_in, sample=sample)
        post = RSSMState(h=prior.h, z_logits=post_logits, z=post_z)
        return post, prior

    # ----- sequence rollout ---------------------------------------------
    def rollout_observed(self, obs: torch.Tensor, act: torch.Tensor,
                         sample: bool = True
                         ) -> Tuple[torch.Tensor, torch.Tensor,
                                    torch.Tensor, RSSMState]:
        """Teacher-forced posterior rollout over a (B, T, *) batch.

        ``act[:, t]`` is the action that drives the transition INTO
        ``obs[:, t]`` (matches the V4 contemporaneous-action convention:
        ``feat[t]`` has seen ``a_t`` so ``reward(feat[t])`` predicts the
        reward of the action taken at step ``t``).

        Returns ``(feats, post_logits, prior_logits, last_state)`` with
        shapes ``(B, T, F)``, ``(B, T, K, C)``, ``(B, T, K, C)`` and the
        final ``RSSMState`` (for imagination warm-start).
        """
        B, T = obs.shape[:2]
        device = obs.device
        embeds = self.embed(obs)                       # (B, T, embed_dim)
        state = self.initial_state(B, device)
        feats_l, post_l, prior_l = [], [], []
        for t in range(T):
            post, prior = self.obs_step(state, act[:, t], embeds[:, t],
                                        sample=sample)
            state = post
            feats_l.append(post.feat)
            post_l.append(post.z_logits)
            prior_l.append(prior.z_logits)
        feats = torch.stack(feats_l, dim=1)
        post_logits = torch.stack(post_l, dim=1)
        prior_logits = torch.stack(prior_l, dim=1)
        return feats, post_logits, prior_logits, state

    def decode(self, feat: torch.Tensor) -> torch.Tensor:
        return self.decoder(feat)


# ---------------------------------------------------------------------------
# KL-balanced free-bits loss
# ---------------------------------------------------------------------------

def rssm_kl_loss(post_logits: torch.Tensor, prior_logits: torch.Tensor,
                 free_bits: float = 1.0, dyn_w: float = 0.5,
                 repr_w: float = 0.1
                 ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """DreamerV3 KL-balanced loss with a single batch-and-time free-bits
    floor on the K-summed categorical KL.

    ``dyn`` term (stop-grad on posterior) trains the prior toward the
    posterior; ``repr`` term (stop-grad on prior) trains the posterior
    toward the prior.  The free-bits floor is applied to the *mean*
    K-summed KL (NOT per-categorical) — a per-group floor silently
    multiplies the floor by K=32 and pins every categorical at the floor,
    trapping the latent at ``post == prior == uniform``.
    """
    def _kl_cat_summed(p_logits, q_logits):
        # KL(p||q) over the last two dims (K × C), summed over the K groups.
        p = F.softmax(p_logits, dim=-1)
        log_p = F.log_softmax(p_logits, dim=-1)
        log_q = F.log_softmax(q_logits, dim=-1)
        return (p * (log_p - log_q)).sum(dim=-1).sum(dim=-1)   # (B, T)

    kl_dyn_raw = _kl_cat_summed(post_logits.detach(), prior_logits)
    kl_repr_raw = _kl_cat_summed(post_logits, prior_logits.detach())
    fb = torch.tensor(float(free_bits), device=post_logits.device,
                      dtype=kl_dyn_raw.dtype)
    kl_dyn = torch.maximum(kl_dyn_raw.mean(), fb)
    kl_repr = torch.maximum(kl_repr_raw.mean(), fb)
    kl_loss = dyn_w * kl_dyn + repr_w * kl_repr
    diag = {
        'kl_dyn': kl_dyn.detach(),
        'kl_repr': kl_repr.detach(),
        'kl_dyn_raw': kl_dyn_raw.mean().detach(),
        'kl_repr_raw': kl_repr_raw.mean().detach(),
    }
    return kl_loss, diag
