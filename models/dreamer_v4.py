"""Paper-faithful DreamerV4 implementation for APC.

Reference: Hafner et al. (2024), "Mastering Diverse Domains through World
Models" (DreamerV4), arXiv:2407.04693.

This module provides:

- ``RSSM``: encoder + GRU + categorical posterior/prior + decoder + reward
  head + continuation head.
- ``Actor``: per-action-dim categorical over ``n_action_bins`` uniform bins
  in [-1, 1] (V4 paper default for continuous control).
- ``Critic``: twohot symlog distribution head (paper §B).
- ``DreamerV4``: top-level container that holds all submodules and exposes
  ``.parameters_world()``, ``.parameters_actor()``, ``.parameters_critic()``.

All shapes use the convention ``(B, T, ...)`` for batched sequences.

Naming follows the paper:

- ``h``  : deterministic GRU state                  shape (B, deter_dim)
- ``z``  : stochastic categorical state             shape (B, n_categoricals * n_classes)
- ``e``  : encoder embedding of an observation      shape (B, embed_dim)
- ``a``  : action vector (continuous, in [-1, 1])   shape (B, action_dim)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Symlog / twohot utilities (paper §B)
# ---------------------------------------------------------------------------

def symlog(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.log1p(torch.abs(x))


def symexp(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1.0)


def twohot_encode(values: torch.Tensor, bin_edges: torch.Tensor) -> torch.Tensor:
    """Encode scalar targets into a two-hot distribution over ``bin_edges``.

    ``values`` shape ``(...)``; ``bin_edges`` shape ``(n_bins,)`` (monotonic).
    Returns ``(..., n_bins)`` with mass concentrated on the two bins that
    bracket each value (linear interpolation).
    """
    values = values.unsqueeze(-1)
    n_bins = bin_edges.shape[0]
    # Find right bin index per value
    bins = bin_edges.view(*([1] * (values.dim() - 1)), n_bins)
    diff = values - bins                    # (..., n_bins)
    # right index = first bin >= value (clamped)
    right = (diff <= 0).float().cumsum(-1)  # 0..1 transition at right edge
    right = (right == 1).float().argmax(-1).clamp_(1, n_bins - 1)
    left = (right - 1).clamp_min_(0)
    # Linear weights
    bl = bin_edges[left]
    br = bin_edges[right]
    span = (br - bl).clamp_min_(1e-8)
    w_right = ((values.squeeze(-1) - bl) / span).clamp_(0.0, 1.0)
    w_left = 1.0 - w_right
    out = torch.zeros(*values.shape[:-1], n_bins, device=values.device, dtype=values.dtype)
    out.scatter_(-1, left.unsqueeze(-1), w_left.unsqueeze(-1))
    out.scatter_add_(-1, right.unsqueeze(-1), w_right.unsqueeze(-1))
    return out


# ---------------------------------------------------------------------------
# MLP block (paper uses LayerNorm + SiLU)
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int,
                 n_layers: int = 2, zero_init_last: bool = False):
        super().__init__()
        layers = []
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
# RSSM — recurrent state space model
# ---------------------------------------------------------------------------

@dataclass
class RSSMConfig:
    obs_dim: int                 # per-step observation dimension (state + aug)
    action_dim: int              # continuous action dim (pre-discretization)
    lookback: int                # observation window length fed to encoder
    deter_dim: int = 512         # GRU hidden size
    embed_dim: int = 512         # encoder output size
    hidden_dim: int = 512        # MLP width
    n_categoricals: int = 32     # paper default
    n_classes: int = 32          # paper default
    n_layers: int = 2
    free_nats: float = 1.0       # summed-K free bits target (V4 KL clamp)


class RSSM(nn.Module):
    """Encoder + GRU + categorical posterior/prior + decoder + heads.

    Forward conventions:

        observe(obs_window, prev_action, prev_h, prev_z) -> (h, z, post_logits, prior_logits)
            for online step ingestion (used in env loop and training).

        imagine_step(prev_h, prev_z, action) -> (h, z, prior_logits)
            for actor-critic rollout in latent space (no observation).
    """

    def __init__(self, cfg: RSSMConfig):
        super().__init__()
        self.cfg = cfg
        self.stoch_dim = cfg.n_categoricals * cfg.n_classes

        # Encoder: flattens (lookback * obs_dim) -> embed.
        self.encoder = MLP(cfg.lookback * cfg.obs_dim, cfg.hidden_dim,
                           cfg.embed_dim, n_layers=cfg.n_layers)

        # GRU cell: input = (z + action), state = h
        self.gru = nn.GRUCell(self.stoch_dim + cfg.action_dim, cfg.deter_dim)
        # Initial state parameters (learned).
        self.init_h = nn.Parameter(torch.zeros(cfg.deter_dim))
        self.init_z = nn.Parameter(torch.zeros(self.stoch_dim))

        # Posterior head: q(z | h, e)
        self.posterior_head = MLP(cfg.deter_dim + cfg.embed_dim, cfg.hidden_dim,
                                  self.stoch_dim, n_layers=cfg.n_layers)
        # Prior head: p(z | h)
        self.prior_head = MLP(cfg.deter_dim, cfg.hidden_dim, self.stoch_dim,
                              n_layers=cfg.n_layers)

        # Reconstruction decoder: predicts the next observation (latest frame
        # of the lookback window — i.e. the current step's obs).
        self.decoder = MLP(cfg.deter_dim + self.stoch_dim, cfg.hidden_dim,
                           cfg.obs_dim, n_layers=cfg.n_layers)

        # Reward head — twohot symlog logits handled by Critic-style head.
        self.reward_head = TwohotHead(cfg.deter_dim + self.stoch_dim,
                                      cfg.hidden_dim, cfg.n_layers)
        # Continuation (1 - done) head — bernoulli logit.
        self.cont_head = MLP(cfg.deter_dim + self.stoch_dim, cfg.hidden_dim,
                             1, n_layers=cfg.n_layers)

    # ------------------------------------------------------------------ utils
    def initial_state(self, batch_size: int, device: torch.device
                      ) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.init_h.unsqueeze(0).expand(batch_size, -1).contiguous()
        z = self.init_z.unsqueeze(0).expand(batch_size, -1).contiguous()
        return h, z

    def _sample_z(self, logits: torch.Tensor) -> torch.Tensor:
        """Sample categorical z with straight-through gradient.

        ``logits`` shape (B, n_categoricals * n_classes).  Returns one-hot
        tensor flattened back to (B, n_categoricals * n_classes).
        """
        cfg = self.cfg
        B = logits.shape[0]
        logits = logits.view(B, cfg.n_categoricals, cfg.n_classes)
        # Unimix (paper §C.2): mix 1% uniform into the categorical.
        probs = F.softmax(logits, dim=-1)
        unimix = 0.01
        probs = (1.0 - unimix) * probs + unimix / cfg.n_classes
        logits = torch.log(probs + 1e-8)
        # Straight-through one-hot sample.
        sample = F.gumbel_softmax(logits, tau=1.0, hard=True, dim=-1)
        return sample.view(B, self.stoch_dim)

    # ----------------------------------------------------- single online step
    def observe_step(self,
                     obs_window: torch.Tensor,    # (B, lookback, obs_dim)
                     prev_action: torch.Tensor,   # (B, action_dim)
                     prev_h: torch.Tensor,        # (B, deter_dim)
                     prev_z: torch.Tensor,        # (B, stoch_dim)
                     ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B = obs_window.shape[0]
        flat = obs_window.reshape(B, -1)
        e = self.encoder(flat)
        # GRU advances h with previous (z, a).
        gru_in = torch.cat([prev_z, prev_action], dim=-1)
        h = self.gru(gru_in, prev_h)
        # Posterior and prior over z_t.
        post_logits = self.posterior_head(torch.cat([h, e], dim=-1))
        prior_logits = self.prior_head(h)
        z = self._sample_z(post_logits)
        return h, z, post_logits, prior_logits

    # ---------------------------------------------- imagined rollout (latent)
    def imagine_step(self,
                     prev_h: torch.Tensor,
                     prev_z: torch.Tensor,
                     action: torch.Tensor,
                     ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        gru_in = torch.cat([prev_z, action], dim=-1)
        h = self.gru(gru_in, prev_h)
        prior_logits = self.prior_head(h)
        z = self._sample_z(prior_logits)
        return h, z, prior_logits

    # ---------------------------------------------------------- decoder/heads
    def decode(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(torch.cat([h, z], dim=-1))

    def predict_reward(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return self.reward_head(torch.cat([h, z], dim=-1))

    def predict_cont(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return self.cont_head(torch.cat([h, z], dim=-1))


# ---------------------------------------------------------------------------
# Twohot symlog head (paper §B)
# ---------------------------------------------------------------------------

class TwohotHead(nn.Module):
    """MLP head that outputs logits over a fixed twohot symlog support.

    Support: ``n_bins`` evenly spaced bin centres in symlog space across
    [-low, +high].  Defaults from paper: 255 bins on [-20, 20].
    """

    def __init__(self, in_dim: int, hidden_dim: int, n_layers: int = 2,
                 n_bins: int = 255, low: float = -20.0, high: float = 20.0):
        super().__init__()
        self.n_bins = n_bins
        self.register_buffer('bin_edges',
                             torch.linspace(low, high, n_bins))
        self.head = MLP(in_dim, hidden_dim, n_bins, n_layers=n_layers,
                        zero_init_last=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)            # logits over symlog bins

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


# ---------------------------------------------------------------------------
# Actor — per-action-dim categorical over uniform bins in [-1, 1]
# ---------------------------------------------------------------------------

class Actor(nn.Module):
    """V4-paper categorical actor for continuous control.

    Each action dimension is discretized into ``n_action_bins`` uniform bins
    in [-1, 1].  At training time we sample a bin per dim and receive the
    bin centre as the continuous action; at inference we take argmax for a
    deterministic policy.
    """

    def __init__(self, in_dim: int, hidden_dim: int, action_dim: int,
                 n_action_bins: int = 21, n_layers: int = 2):
        super().__init__()
        self.action_dim = action_dim
        self.n_bins = n_action_bins
        self.head = MLP(in_dim, hidden_dim, action_dim * n_action_bins,
                        n_layers=n_layers, zero_init_last=True)
        self.register_buffer('bin_centres',
                             torch.linspace(-1.0, 1.0, n_action_bins))

    def _logits(self, latent: torch.Tensor) -> torch.Tensor:
        B = latent.shape[0]
        return self.head(latent).view(B, self.action_dim, self.n_bins)

    def forward(self, latent: torch.Tensor, *, deterministic: bool = False
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (action, log_prob, entropy).

        ``action`` shape (B, action_dim) in [-1, 1].
        ``log_prob`` shape (B,)  — sum across action dims.
        ``entropy``  shape (B,)  — sum across action dims.
        """
        logits = self._logits(latent)
        if deterministic:
            idx = logits.argmax(dim=-1)
        else:
            probs = F.softmax(logits, dim=-1)
            idx = torch.distributions.Categorical(probs=probs).sample()
        action = self.bin_centres[idx]  # (B, action_dim)
        log_probs = F.log_softmax(logits, dim=-1)
        log_prob = log_probs.gather(-1, idx.unsqueeze(-1)).squeeze(-1).sum(dim=-1)
        entropy = -(F.softmax(logits, dim=-1) * log_probs).sum(dim=-1).sum(dim=-1)
        return action, log_prob, entropy


# ---------------------------------------------------------------------------
# Critic — twohot symlog over lambda returns
# ---------------------------------------------------------------------------

class Critic(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, n_layers: int = 2):
        super().__init__()
        self.head = TwohotHead(in_dim, hidden_dim, n_layers=n_layers)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.head(latent)             # logits

    def value(self, latent: torch.Tensor) -> torch.Tensor:
        return self.head.expectation(self.head(latent))

    def loss(self, latent: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.head.loss(self.head(latent), target)


# ---------------------------------------------------------------------------
# Top-level container
# ---------------------------------------------------------------------------

@dataclass
class DreamerV4Config:
    rssm: RSSMConfig
    n_action_bins: int = 21
    actor_hidden: int = 512
    critic_hidden: int = 512


class DreamerV4(nn.Module):
    def __init__(self, cfg: DreamerV4Config):
        super().__init__()
        self.cfg = cfg
        self.rssm = RSSM(cfg.rssm)
        latent_dim = cfg.rssm.deter_dim + cfg.rssm.n_categoricals * cfg.rssm.n_classes
        self.actor = Actor(latent_dim, cfg.actor_hidden,
                           cfg.rssm.action_dim, n_action_bins=cfg.n_action_bins)
        self.critic = Critic(latent_dim, cfg.critic_hidden)
        # EMA target critic for stability (paper §C).
        self.target_critic = Critic(latent_dim, cfg.critic_hidden)
        self.target_critic.load_state_dict(self.critic.state_dict())
        for p in self.target_critic.parameters():
            p.requires_grad_(False)

    def update_target(self, tau: float = 0.02) -> None:
        with torch.no_grad():
            for p, t in zip(self.critic.parameters(), self.target_critic.parameters()):
                t.data.mul_(1.0 - tau).add_(tau * p.data)

    # Convenience parameter groups
    def parameters_world(self):
        return list(self.rssm.parameters())

    def parameters_actor(self):
        return list(self.actor.parameters())

    def parameters_critic(self):
        return list(self.critic.parameters())


# ---------------------------------------------------------------------------
# KL helpers (free-bits, summed across categoricals — paper recommendation)
# ---------------------------------------------------------------------------

def categorical_kl(post_logits: torch.Tensor, prior_logits: torch.Tensor,
                   n_categoricals: int, n_classes: int) -> torch.Tensor:
    """KL[ post || prior ] summed across the K categoricals.

    Shapes: ``(B, n_categoricals * n_classes)`` -> ``(B,)``.
    """
    B = post_logits.shape[0]
    p = post_logits.view(B, n_categoricals, n_classes)
    q = prior_logits.view(B, n_categoricals, n_classes)
    p_log = F.log_softmax(p, dim=-1)
    q_log = F.log_softmax(q, dim=-1)
    p_prob = p_log.exp()
    kl_per_cat = (p_prob * (p_log - q_log)).sum(dim=-1)   # (B, K)
    return kl_per_cat.sum(dim=-1)                         # (B,)


def free_bits_kl(post_logits: torch.Tensor, prior_logits: torch.Tensor,
                 n_categoricals: int, n_classes: int,
                 free_nats: float) -> torch.Tensor:
    """Summed-K KL clamped at ``free_nats`` (V4 spec)."""
    kl = categorical_kl(post_logits, prior_logits, n_categoricals, n_classes)
    return torch.clamp(kl, min=free_nats).mean()
