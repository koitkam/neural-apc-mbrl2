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
    # DV-as-input (Option B, 2026-06-07).  When ``dv_dim > 0`` the measured
    # disturbance-variable channels (at ``dv_indices`` within the obs vector)
    # are fed as an EXOGENOUS input to the transition (concatenated with the
    # action) instead of being PREDICTED forward by the latent.  In imagination
    # the DV is held at its last measured value (MPC feedforward persistence);
    # in teacher-forced training the real per-step DV is supplied so the WM
    # learns dCV/dDV directly.  ``dv_dim = 0`` (no-DV sims / opt-out) is
    # bit-identical to the paper behaviour.
    dv_dim: int = 0
    dv_indices: Tuple[int, ...] = ()
    # ---- Neural Kalman filter / disturbance observer (DOB), 2026-06-11 ----
    # When ``dob_enabled`` the WM carries an explicit additive output-disturbance
    # state ``d_t`` (one scalar per CV channel) that INTEGRATES the one-step
    # prediction residual (innovation) and is ADDED to the decoded CV at recon
    # time: ``CV = g(feat) + d_t``.  This gives the decoder/dynamics ``g`` a
    # dedicated channel to absorb the unmeasured-load movement so it is no longer
    # forced to soak it up — de-confounding the omitted-variable gain attenuation
    # (p112: gain 0.36 with the disturbance ON vs 0.18 with it OFF, Exp A p113).
    # Predict (img_step, no obs): d_t = A·d_{t-1}.  Correct (obs_step, real obs):
    # d_t = A·d_{t-1} + K·(CV_obs − [g(prior.feat)+A·d_{t-1}]).  A,K are learned
    # per-CV scalars in (0,1) (sigmoid) — a first-order learned Kalman gain.
    # ``cv_indices`` = the CV obs-vector positions (== env.cv_indices); 0 CVs or
    # ``dob_enabled=False`` ⇒ byte-identical to the pre-DOB model.
    dob_enabled: bool = False
    cv_indices: Tuple[int, ...] = ()
    dob_decay_init: float = 3.0     # sigmoid(3.0)=0.953 — slow persistence
    dob_gain_init: float = -2.2     # sigmoid(-2.2)=0.10 — small correction


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
    d: Optional[torch.Tensor] = None  # (..., n_cv) DOB disturbance state (None=off)

    @property
    def stoch_flat(self) -> torch.Tensor:
        return self.z.flatten(start_dim=-2)

    @property
    def feat(self) -> torch.Tensor:
        # Scope 2 (DOB feed-forward, 2026-06-11): when the DOB carries a
        # disturbance estimate ``d`` we APPEND it (detached) so the actor /
        # critic / reward heads CONDITION on the disturbance — explicit
        # feed-forward.  p114 RCA: without this the imagined world is
        # disturbance-free and the actor learns to be passive (mv_tv≈0 while
        # CV drifts).  ``d`` is detached because it is a sensor-like estimate
        # driven by the recon innovation (the DOB is trained via ``apply_dob``
        # on the recon, NOT by the heads), so head gradients must not reshape
        # it.  The DECODER still reads only ``[h, z_flat]`` (``decode`` slices
        # the d-tail off) — the clean-gain ``g`` + additive ``d`` factorisation
        # is preserved.  ``d is None`` (DOB off) ⇒ feat = [h, z_flat] exactly.
        if self.d is not None:
            return torch.cat([self.h, self.stoch_flat, self.d.detach()], dim=-1)
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
        # DV-as-input (Option B): exogenous measured-DV channels fed into the
        # transition.  ``dv_index_t`` selects them out of the obs vector.
        self.dv_dim = int(getattr(cfg, 'dv_dim', 0) or 0)
        self.register_buffer(
            'dv_index_t',
            torch.tensor(list(getattr(cfg, 'dv_indices', ()) or ()),
                         dtype=torch.long),
            persistent=False)
        # Transition input = [z_flat ; action ; (dv)].
        trans_in = self.stoch_flat_dim + self.action_dim + self.dv_dim

        # Encoder: obs → per-frame embedding.
        self.encoder = _MLP(self.obs_dim, self.embed_dim,
                            hidden_dim=self.hidden_dim, num_layers=3)
        # Decoder: [h, z] → reconstructed (normalized) obs.  Reads the CORE
        # feature only (deter + stoch); the DOB d-tail (Scope 2) is sliced off
        # in ``decode`` and re-added via ``apply_dob`` (the g + d factorisation).
        self.decoder = _MLP(self.deter_dim + self.stoch_flat_dim, self.obs_dim,
                            hidden_dim=self.hidden_dim, num_layers=3)
        # Recurrent dynamics: pre-GRU projection then GRUCell.
        self.pre_gru = _MLP(trans_in, trans_in,
                            hidden_dim=self.hidden_dim, num_layers=1)
        self.gru = nn.GRUCell(trans_in, self.deter_dim)
        # Prior p(z'|h') and posterior q(z'|h', embed).
        self.prior_net = _CategoricalLatent(
            self.deter_dim, self.n_categoricals, self.n_classes,
            hidden_dim=self.hidden_dim, unimix=cfg.unimix)
        self.post_net = _CategoricalLatent(
            self.deter_dim + self.embed_dim, self.n_categoricals,
            self.n_classes, hidden_dim=self.hidden_dim, unimix=cfg.unimix)

        # ----- Neural Kalman filter / disturbance observer (DOB) -----
        # ``d_t`` (per-CV) is a first-order learned observer on the one-step
        # prediction residual.  A (decay) and K (innovation gain) are learned
        # per-CV scalars in (0,1) via sigmoid.  Requires CV obs-vector indices;
        # with 0 CVs the DOB is force-disabled (no channel to observe).
        self.register_buffer(
            'cv_index_t',
            torch.tensor(list(getattr(cfg, 'cv_indices', ()) or ()),
                         dtype=torch.long),
            persistent=False)
        self.n_cv = int(self.cv_index_t.numel())
        self.dob_enabled = bool(getattr(cfg, 'dob_enabled', False)) and self.n_cv > 0
        if self.dob_enabled:
            self.dob_log_decay = nn.Parameter(torch.full(
                (self.n_cv,), float(getattr(cfg, 'dob_decay_init', 3.0))))
            self.dob_log_gain = nn.Parameter(torch.full(
                (self.n_cv,), float(getattr(cfg, 'dob_gain_init', -2.2))))

    @property
    def feat_dim(self) -> int:
        # Scope 2: the head-facing feature includes the DOB disturbance estimate
        # ``d`` (one scalar per CV) so the actor/critic/reward heads condition
        # on it.  The decoder reads only the core (deter+stoch) — see ``decode``.
        core = self.deter_dim + self.stoch_flat_dim
        return core + (self.n_cv if getattr(self, 'dob_enabled', False) else 0)

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
        d = (torch.zeros(batch_size, self.n_cv, device=device)
             if self.dob_enabled else None)
        return RSSMState(h=h, z_logits=z_logits, z=z, d=d)

    # ----- DOB helpers --------------------------------------------------
    def dob_decay(self) -> torch.Tensor:
        return torch.sigmoid(self.dob_log_decay)

    def dob_gain(self) -> torch.Tensor:
        return torch.sigmoid(self.dob_log_gain)

    def apply_dob(self, decoded: torch.Tensor,
                  d: Optional[torch.Tensor]) -> torch.Tensor:
        """Add the DOB disturbance state ``d`` (..., n_cv) into the CV channels
        of a decoded obs tensor (..., obs_dim).  Identity when DOB is off."""
        if not self.dob_enabled or d is None:
            return decoded
        out = decoded.clone()
        out.index_add_(-1, self.cv_index_t, d.to(out.dtype))
        return out

    # ----- transitions --------------------------------------------------
    def img_step(self, prev: RSSMState, prev_action: torch.Tensor,
                 dv: Optional[torch.Tensor] = None,
                 sample: bool = True) -> RSSMState:
        """Imagined (prior-only) step: advance the state with no obs.

        ``dv`` (B, dv_dim) is the exogenous measured-DV input when DV-as-input
        is enabled (``dv_dim > 0``); ``None`` is filled with zeros.  Ignored
        entirely when ``dv_dim == 0`` (paper behaviour)."""
        if self.dv_dim > 0:
            if dv is None:
                dv = torch.zeros(prev_action.shape[0], self.dv_dim,
                                 device=prev_action.device,
                                 dtype=prev_action.dtype)
            x = torch.cat([prev.stoch_flat, prev_action, dv], dim=-1)
        else:
            x = torch.cat([prev.stoch_flat, prev_action], dim=-1)
        x = self.pre_gru(x)
        h = self.gru(x, prev.h)
        z_logits, z = self.prior_net(h, sample=sample)
        # DOB predict step: decay the disturbance estimate (no obs to correct).
        d_new = (self.dob_decay() * prev.d
                 if (self.dob_enabled and prev.d is not None) else prev.d)
        return RSSMState(h=h, z_logits=z_logits, z=z, d=d_new)

    def obs_step(self, prev: RSSMState, prev_action: torch.Tensor,
                 embed: torch.Tensor, dv: Optional[torch.Tensor] = None,
                 sample: bool = True, obs: Optional[torch.Tensor] = None
                 ) -> Tuple[RSSMState, RSSMState]:
        """Observation step → (posterior, prior).  Prior is needed for KL.

        When the DOB is active and ``obs`` (the raw obs vector, for the CV
        channels) is supplied, the posterior carries the CORRECTED disturbance
        state ``d_t = A·d_{t-1} + K·ν`` where ``ν`` is the one-step prediction
        residual on the PRIOR forecast (a genuine innovation; the prior has not
        seen the current obs).  ``obs=None`` (probes / diagnostics) ⇒ the
        posterior just carries the decayed prior ``d`` (pure process model)."""
        prior = self.img_step(prev, prev_action, dv=dv, sample=sample)
        post_in = torch.cat([prior.h, embed], dim=-1)
        post_logits, post_z = self.post_net(post_in, sample=sample)
        d_post = prior.d
        if self.dob_enabled and obs is not None and prior.d is not None:
            cv_pred = (self.decode(prior.feat).index_select(-1, self.cv_index_t)
                       + prior.d)                       # one-step CV forecast
            cv_obs = obs.index_select(-1, self.cv_index_t)
            nu = cv_obs - cv_pred                        # innovation
            d_post = prior.d + self.dob_gain() * nu      # = A·d_{t-1} + K·ν
        post = RSSMState(h=prior.h, z_logits=post_logits, z=post_z, d=d_post)
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

        Returns ``(feats, post_logits, prior_logits, last_state, ds)`` with
        shapes ``(B, T, F)``, ``(B, T, K, C)``, ``(B, T, K, C)``, the final
        ``RSSMState`` (for imagination warm-start), and ``ds`` ``(B, T, n_cv)``
        = the per-step DOB disturbance estimate (``None`` when DOB is off).
        """
        B, T = obs.shape[:2]
        device = obs.device
        embeds = self.embed(obs)                       # (B, T, embed_dim)
        # Exogenous DV input per step (teacher-forced from the real obs).
        dvs = (obs.index_select(-1, self.dv_index_t)
               if self.dv_dim > 0 else None)           # (B, T, dv_dim) | None
        state = self.initial_state(B, device)
        feats_l, post_l, prior_l, ds_l = [], [], [], []
        for t in range(T):
            dv_t = dvs[:, t] if dvs is not None else None
            post, prior = self.obs_step(state, act[:, t], embeds[:, t],
                                        dv=dv_t, sample=sample, obs=obs[:, t])
            state = post
            feats_l.append(post.feat)
            post_l.append(post.z_logits)
            prior_l.append(prior.z_logits)
            if self.dob_enabled and post.d is not None:
                ds_l.append(post.d)
        feats = torch.stack(feats_l, dim=1)
        post_logits = torch.stack(post_l, dim=1)
        prior_logits = torch.stack(prior_l, dim=1)
        ds = torch.stack(ds_l, dim=1) if ds_l else None
        return feats, post_logits, prior_logits, state, ds

    def decode(self, feat: torch.Tensor) -> torch.Tensor:
        # Scope 2: the decoder learns the CLEAN g([h, z]); slice off any DOB
        # d-tail that ``feat`` may carry (it is re-added by ``apply_dob``).  When
        # the DOB is off, ``feat`` is already core-width so this is a no-op.
        core = self.deter_dim + self.stoch_flat_dim
        return self.decoder(feat[..., :core])


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
