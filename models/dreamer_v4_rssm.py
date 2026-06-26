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
    # DV→decoder+heads FEEDFORWARD (2026-06-19, p129 RCA).  When True (and
    # dv_dim>0) the measured DV is appended to the head-facing ``feat`` AND fed
    # directly into the decoder, so the reconstructed CV is ``g(h, z, dv)`` — a
    # DIRECT exogenous-DV path that SKIPS the lossy categorical bottleneck.  The
    # p129 DV posterior-prior decomp proved the DV→CV gain dies ENTIRELY in the
    # autoencoder (real→post ×0.77, post→1step ×1.00); routing the DV around the
    # latent lets the decoder represent ∂CV/∂DV directly, and the value/policy/
    # reward heads finally SEE the disturbance (fixes the passive actor).
    # ``dv_dim = 0`` ⇒ no-op (byte-identical to the pre-feedforward model).
    dv_feedforward: bool = True
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
    # ---- Continuous gain + disturbance latent (2026-06-22) ----
    # A small Gaussian latent ALONGSIDE the categorical, giving precision-
    # critical CONTINUOUS quantities (the subdominant input GAIN and the
    # unmeasured DISTURBANCE) an UN-quantized home that the categorical
    # small-signal attenuation cannot reach.  Split into a GAIN block
    # (supervised toward the identified steady-state gain — fixes the DV
    # categorical-attenuation bias AND carries the per-episode gain in-context
    # so the WM ADAPTS to DR) and a DISTURBANCE block (= n_cv, an amortized
    # Kalman state inferred from the innovation + rolled forward by the prior —
    # the "inherent" replacement for the bolt-on DOB).  Both feed the GRU (so
    # ``h`` carries them forward) and the decoder (so the recon forces them to
    # mean what we want).  ``cont_gain_dim == cont_dist_dim == 0`` ⇒
    # byte-identical to the pre-continuous-latent model.
    cont_gain_dim: int = 0
    cont_dist_dim: int = 0
    cont_min_std: float = 0.1       # σ floor (numerical + KL well-posedness)
    cont_max_std: float = 2.0       # σ ceiling


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
    dv: Optional[torch.Tensor] = None  # (..., dv_dim) exogenous DV feedforward (None=off)
    c: Optional[torch.Tensor] = None       # (..., cont_dim) continuous latent sample (None=off)
    c_mean: Optional[torch.Tensor] = None  # (..., cont_dim) post/prior mean (for KL)
    c_std: Optional[torch.Tensor] = None   # (..., cont_dim) post/prior std (for KL)

    @property
    def stoch_flat(self) -> torch.Tensor:
        return self.z.flatten(start_dim=-2)

    @property
    def feat(self) -> torch.Tensor:
        # Scope 2 (DOB feed-forward, 2026-06-11) + DV feed-forward (2026-06-19)
        # + continuous gain/disturbance latent (2026-06-22): the head-facing
        # feature is ``[h, z_flat, (c), (dv), (d.detach())]``.
        #  * ``c`` (continuous gain+disturbance latent) is appended RIGHT AFTER
        #    the categorical core so the DECODER reads ``[h, z, c, (dv)]`` (a
        #    contiguous front slice) — the un-quantized path for the gain and
        #    the unmeasured disturbance.  NOT detached: the decoder learns to
        #    use the gain/disturbance through it.
        #  * ``dv`` (DV feedforward) follows ``c``.  Not detached.
        #  * ``d`` (DOB) is appended LAST and DETACHED (sliced off by decode).
        # ``c is None`` AND ``dv is None`` AND ``d is None`` ⇒ feat = [h, z_flat]
        # (byte-identical to the paper RSSM).
        parts = [self.h, self.stoch_flat]
        if self.c is not None:
            parts.append(self.c)
        if self.dv is not None:
            parts.append(self.dv)
        if self.d is not None:
            parts.append(self.d.detach())
        return torch.cat(parts, dim=-1)


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


class _ContinuousLatent(nn.Module):
    """Feature → diagonal-Gaussian (mean, std) continuous latent.

    A small reparameterised Gaussian channel that lives ALONGSIDE the
    categorical ``_CategoricalLatent``.  Its purpose is to hold precision-
    critical CONTINUOUS quantities (the per-episode input gain and the
    unmeasured disturbance) that the discrete categorical attenuates by
    quantization.  ``std`` is a softplus output clamped to
    ``[min_std, max_std]`` for KL well-posedness; the sample uses the
    reparameterisation trick so gradients flow into both the inference net
    and (through the decoder/recon) the value of the latent.
    """

    def __init__(self, in_dim: int, cont_dim: int, hidden_dim: int = 256,
                 min_std: float = 0.1, max_std: float = 2.0):
        super().__init__()
        self.cont_dim = int(cont_dim)
        self.min_std = float(min_std)
        self.max_std = float(max_std)
        self.net = _MLP(in_dim, 2 * cont_dim, hidden_dim, num_layers=2)

    def forward(self, x: torch.Tensor, sample: bool = True
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, std_raw = self.net(x).chunk(2, dim=-1)
        # Bounded std: min_std + (max_std-min_std)·sigmoid(std_raw) keeps σ
        # strictly inside (min_std, max_std) — no exp() blow-up, well-posed KL.
        std = self.min_std + (self.max_std - self.min_std) * torch.sigmoid(std_raw)
        if sample:
            c = mean + std * torch.randn_like(std)
        else:
            c = mean
        return c, mean, std


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
        # Continuous gain+disturbance latent (2026-06-22).  cont_dim splits into
        # a supervised GAIN block (first ``cont_gain_dim`` dims) and a
        # DISTURBANCE block (last ``cont_dist_dim`` dims).  Both feed the GRU
        # (so ``h`` carries them forward → the prior can roll them) AND the
        # decoder (so the recon forces them to mean what we want).
        self.cont_gain_dim = int(getattr(cfg, 'cont_gain_dim', 0) or 0)
        self.cont_dist_dim = int(getattr(cfg, 'cont_dist_dim', 0) or 0)
        self.cont_dim = self.cont_gain_dim + self.cont_dist_dim
        self.cont_min_std = float(getattr(cfg, 'cont_min_std', 0.1))
        self.cont_max_std = float(getattr(cfg, 'cont_max_std', 2.0))
        # DV-as-input (Option B): exogenous measured-DV channels fed into the
        # transition.  ``dv_index_t`` selects them out of the obs vector.
        self.dv_dim = int(getattr(cfg, 'dv_dim', 0) or 0)
        self.register_buffer(
            'dv_index_t',
            torch.tensor(list(getattr(cfg, 'dv_indices', ()) or ()),
                         dtype=torch.long),
            persistent=False)
        # DV→decoder+heads feedforward (2026-06-19): only meaningful with DVs.
        self.dv_feedforward = bool(getattr(cfg, 'dv_feedforward', True)) \
            and self.dv_dim > 0
        self._dv_feed_dim = self.dv_dim if self.dv_feedforward else 0
        # Transition input = [z_flat ; (c) ; action ; (dv)].  The continuous
        # latent feeds the GRU so ``h`` carries the gain/disturbance forward.
        trans_in = (self.stoch_flat_dim + self.cont_dim
                    + self.action_dim + self.dv_dim)

        # Encoder: obs → per-frame embedding.
        self.encoder = _MLP(self.obs_dim, self.embed_dim,
                            hidden_dim=self.hidden_dim, num_layers=3)
        # Decoder: [h, z, (dv)] → reconstructed (normalized) obs.  Reads the
        # latent core (deter + stoch) PLUS the exogenous DV when DV-feedforward
        # is on, so the CV reconstruction ``g(h, z, dv)`` has a DIRECT ∂CV/∂dv
        # path that skips the categorical bottleneck (p129 RCA).  The DOB d-tail
        # (Scope 2) is sliced off in ``decode`` and re-added via ``apply_dob``
        # (the g + d factorisation).
        self.decoder = _MLP(self.deter_dim + self.stoch_flat_dim + self.cont_dim
                            + self._dv_feed_dim, self.obs_dim,
                            hidden_dim=self.hidden_dim, num_layers=3)
        # Direct DV→obs FEEDFORWARD SKIP (2026-06-20, p132 RCA).  The measured
        # DV is a single channel concatenated into the ~1500-d decoder MLP input
        # — its gradient is diluted ~1/1500, so the decoder under-uses it and the
        # DV→CV gain still dies in the autoencoder (p132 DV real→post 0.67 < MV
        # 0.84, i.e. the "direct" dv-feedforward is NOT actually direct).  This
        # linear skip gives the exogenous DV a CLEAN, high-gradient path straight
        # to the reconstructed obs (g(h,z,dv) + W·dv), bypassing BOTH the dilution
        # AND the categorical bottleneck — exactly the role dv-feedforward was
        # meant to play.  ZERO-INIT ⇒ starts as an exact no-op (byte-identical to
        # the pre-skip decode) and learns the clean ∂CV/∂dv from the residual.
        self.dv_skip = (nn.Linear(self._dv_feed_dim, self.obs_dim)
                        if self._dv_feed_dim > 0 else None)
        if self.dv_skip is not None:
            nn.init.zeros_(self.dv_skip.weight)
            nn.init.zeros_(self.dv_skip.bias)
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
        # Continuous-latent prior p(c'|h') and posterior q(c'|h', embed).
        if self.cont_dim > 0:
            self.cont_prior_net = _ContinuousLatent(
                self.deter_dim, self.cont_dim, hidden_dim=self.hidden_dim,
                min_std=self.cont_min_std, max_std=self.cont_max_std)
            # Innovation-driven posterior (2026-06-26, p139 RCA / Option B).  The
            # DISTURBANCE block of the cont posterior infers the unmeasured load
            # from the one-step CV INNOVATION ν = CV_obs − prior CV forecast (the
            # DOB residual that IS the load) — NOT from [h, embed] alone, which
            # could not (p139: the load is observable, det_r(ν)=0.32, but a
            # non-innovation posterior learned an excited-CV shortcut that died
            # under closed-loop control, det_r 0.03).  Appending ν makes c_dist a
            # LEARNED amortized Kalman that transfers to deployment.  ν is n_cv =
            # cont_dist_dim wide; only added when the disturbance block exists.
            # Width = n_cv (the actual CV count = cont_dist_dim in a resolved
            # run); gated on n_cv>0 so a CV-less config is a clean no-op.
            _n_cv = len(getattr(cfg, 'cv_indices', ()) or ())
            self._cont_post_uses_innov = self.cont_dist_dim > 0 and _n_cv > 0
            _innov_dim = _n_cv if self._cont_post_uses_innov else 0
            self.cont_post_net = _ContinuousLatent(
                self.deter_dim + self.embed_dim + _innov_dim, self.cont_dim,
                hidden_dim=self.hidden_dim, min_std=self.cont_min_std,
                max_std=self.cont_max_std)
        else:
            self._cont_post_uses_innov = False

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
        # Curriculum Stage-1 suppression (2026-06-12): when ``dob_active`` is
        # False (Stage 1 = clean-plant identification) the disturbance estimate
        # d_t is forced to ZERO so the dynamics ``g`` must explain ALL CV
        # movement (no observer escape-hatch) -> unbiased input->CV gain.  The
        # feature still carries a ZERO d-tail so head dims stay constant across
        # stages (no checkpoint head-dim mismatch).  Set via
        # ``DreamerV4.set_dob_active``.  Default True = observer runs.
        self.dob_active = True
        if self.dob_enabled:
            self.dob_log_decay = nn.Parameter(torch.full(
                (self.n_cv,), float(getattr(cfg, 'dob_decay_init', 3.0))))
            self.dob_log_gain = nn.Parameter(torch.full(
                (self.n_cv,), float(getattr(cfg, 'dob_gain_init', -2.2))))

    @property
    def feat_dim(self) -> int:
        # Scope 2: the head-facing feature includes the DV feedforward (dv_dim
        # when on) so the heads condition on the measured DV, plus the DOB
        # disturbance estimate ``d`` (one scalar per CV).  The decoder reads
        # ``[h, z, (dv)]`` (see ``_decode_in_dim`` / ``decode``).
        core = self.deter_dim + self.stoch_flat_dim + self.cont_dim
        return (core + self._dv_feed_dim
                + (self.n_cv if getattr(self, 'dob_enabled', False) else 0))

    @property
    def _decode_in_dim(self) -> int:
        # Width of the decoder input slice = latent core + cont latent + DV ff.
        return (self.deter_dim + self.stoch_flat_dim + self.cont_dim
                + self._dv_feed_dim)

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
        dv = (torch.zeros(batch_size, self.dv_dim, device=device)
              if self.dv_feedforward else None)
        c = (torch.zeros(batch_size, self.cont_dim, device=device)
             if self.cont_dim > 0 else None)
        return RSSMState(h=h, z_logits=z_logits, z=z, d=d, dv=dv, c=c)

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
        # GRU transition input = [z_flat ; (c) ; action ; (dv)].  The continuous
        # latent feeds the recurrence so ``h`` carries the gain/disturbance
        # forward and the prior ``cont_prior_net(h')`` can roll them.
        parts = [prev.stoch_flat]
        if self.cont_dim > 0:
            c_prev = prev.c
            if c_prev is None:
                c_prev = torch.zeros(prev_action.shape[0], self.cont_dim,
                                     device=prev_action.device,
                                     dtype=prev_action.dtype)
            parts.append(c_prev)
        parts.append(prev_action)
        if self.dv_dim > 0:
            if dv is None:
                dv = torch.zeros(prev_action.shape[0], self.dv_dim,
                                 device=prev_action.device,
                                 dtype=prev_action.dtype)
            parts.append(dv)
        x = torch.cat(parts, dim=-1)
        x = self.pre_gru(x)
        h = self.gru(x, prev.h)
        z_logits, z = self.prior_net(h, sample=sample)
        # Continuous-latent prior p(c'|h'): gain persists (carried via h) and
        # the disturbance OU-rolls; both inferred from the recurrent state.
        c_new = c_mean = c_std = None
        if self.cont_dim > 0:
            c_new, c_mean, c_std = self.cont_prior_net(h, sample=sample)
        # DOB predict step: decay the disturbance estimate (no obs to correct).
        d_new = (self.dob_decay() * prev.d
                 if (self.dob_enabled and prev.d is not None) else prev.d)
        # DV feedforward: carry the (real / held / zero-filled) DV into the
        # state so ``feat`` + ``decode`` expose it to the decoder and heads.
        dv_new = dv if self.dv_feedforward else None
        return RSSMState(h=h, z_logits=z_logits, z=z, d=d_new, dv=dv_new,
                         c=c_new, c_mean=c_mean, c_std=c_std)

    def obs_step(self, prev: RSSMState, prev_action: torch.Tensor,
                 embed: torch.Tensor, dv: Optional[torch.Tensor] = None,
                 sample: bool = True, obs: Optional[torch.Tensor] = None,
                 cont_innov: Optional[torch.Tensor] = None
                 ) -> Tuple[RSSMState, RSSMState]:
        """Observation step → (posterior, prior).  Prior is needed for KL.

        When the DOB is active and ``obs`` (the raw obs vector, for the CV
        channels) is supplied, the posterior carries the CORRECTED disturbance
        state ``d_t = A·d_{t-1} + K·ν`` where ``ν`` is the one-step prediction
        residual on the PRIOR forecast (a genuine innovation; the prior has not
        seen the current obs).  ``obs=None`` (probes / diagnostics) ⇒ the
        posterior just carries the decayed prior ``d`` (pure process model).

        ``cont_innov`` (B, cont_dist_dim) is the same CV innovation, precomputed
        BATCHED by ``rollout_observed`` and fed to the innovation-driven cont
        DISTURBANCE posterior (Option B).  When omitted but ``obs`` is given
        (standalone calls) it is computed inline; with neither it is zeros."""
        prior = self.img_step(prev, prev_action, dv=dv, sample=sample)
        post_in = torch.cat([prior.h, embed], dim=-1)
        post_logits, post_z = self.post_net(post_in, sample=sample)
        # Continuous-latent posterior q(c'|h', embed[, ν]): the GAIN block infers
        # from the history in h; the DISTURBANCE block infers from the CV
        # innovation ν (the amortized Kalman update — Option B).
        c_post = c_post_mean = c_post_std = None
        if self.cont_dim > 0:
            cont_in = post_in
            if self._cont_post_uses_innov:
                if cont_innov is None:
                    if obs is not None and self.n_cv > 0:
                        cv_fore = self.decode(prior.feat).index_select(
                            -1, self.cv_index_t)
                        if prior.d is not None:
                            cv_fore = cv_fore + prior.d
                        cont_innov = (obs.index_select(-1, self.cv_index_t)
                                      - cv_fore)
                    else:
                        cont_innov = torch.zeros(
                            post_in.shape[0], self.n_cv,
                            device=post_in.device, dtype=post_in.dtype)
                cont_in = torch.cat([post_in, cont_innov], dim=-1)
            c_post, c_post_mean, c_post_std = self.cont_post_net(
                cont_in, sample=sample)
        d_post = prior.d
        if self.dob_enabled and obs is not None and prior.d is not None:
            cv_pred = (self.decode(prior.feat).index_select(-1, self.cv_index_t)
                       + prior.d)                       # one-step CV forecast
            cv_obs = obs.index_select(-1, self.cv_index_t)
            nu = cv_obs - cv_pred                        # innovation
            d_post = prior.d + self.dob_gain() * nu      # = A·d_{t-1} + K·ν
        # Posterior inherits the prior's exogenous DV feedforward (same measured
        # DV drove both) so ``post.feat`` / ``decode(post.feat)`` expose it.
        post = RSSMState(h=prior.h, z_logits=post_logits, z=post_z, d=d_post,
                         dv=prior.dv, c=c_post, c_mean=c_post_mean,
                         c_std=c_post_std)
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
        core = self.deter_dim + self.stoch_flat_dim
        dec_in = self._decode_in_dim                   # core (+ cont + dv ff)
        # Option B (2026-06-26, p139 RCA): the innovation-driven cont DISTURBANCE
        # posterior needs the one-step CV innovation ν, which needs a PRIOR
        # DECODE — too expensive PER STEP inside the compiled loop (the same
        # reason the DOB below batches it).  So when the cont disturbance block
        # is on we run TWO compile-friendly passes: pass 1 rolls a ZERO-
        # innovation cont posterior to harvest the prior feats; ONE batched
        # decode of those gives ν; pass 2 re-rolls feeding ν[:, t] so the c that
        # feeds h is innovation-driven (→ the prior rolls the load forward in
        # imagination).  pass-1 ν ≈ the full load (its c_dist is ~uninformative),
        # exactly the signal the posterior should map.  Single pass when off.
        two_pass = bool(getattr(self, '_cont_post_uses_innov', False))
        _need_prior_core = self.dob_enabled or two_pass
        feats_l, post_l, prior_l, prior_core_l = [], [], [], []
        c_qm_l, c_qs_l, c_pm_l, c_ps_l = [], [], [], []
        for t in range(T):
            dv_t = dvs[:, t] if dvs is not None else None
            # COMPILE-EFFICIENT recurrence (2026-06-12): run the (h, z) recurrence
            # with ``obs=None`` so neither the DOB d-update NOR the per-step prior
            # decode (used for both the DOB and the cont innovation) enters the
            # compiled loop — the EXPENSIVE decode is hoisted OUT and done ONCE,
            # batched, below (the T× decoder-MLP copies otherwise made the
            # rollout ~15 min to compile / run launch-bound).  ``d`` does NOT
            # affect h/z, and the cont innovation is fed in pass 2.
            post, prior = self.obs_step(state, act[:, t], embeds[:, t],
                                        dv=dv_t, sample=sample, obs=None)
            state = post
            feats_l.append(post.feat[..., :dec_in])    # decoder feat [h,z,(c),(dv)]
            post_l.append(post.z_logits)
            prior_l.append(prior.z_logits)
            if self.cont_dim > 0:
                c_qm_l.append(post.c_mean); c_qs_l.append(post.c_std)
                c_pm_l.append(prior.c_mean); c_ps_l.append(prior.c_std)
            if _need_prior_core:
                prior_core_l.append(prior.feat[..., :dec_in])
        if two_pass:
            # ONE batched prior decode → CV forecast → innovation ν, then
            # re-roll with the innovation-driven cont posterior.
            prior_core1 = torch.stack(prior_core_l, dim=1)         # (B, T, dec_in)
            base = self.decode(prior_core1).index_select(-1, self.cv_index_t)
            nu_seq = obs.index_select(-1, self.cv_index_t) - base  # (B, T, n_cv)
            state = self.initial_state(B, device)
            feats_l, post_l, prior_l, prior_core_l = [], [], [], []
            c_qm_l, c_qs_l, c_pm_l, c_ps_l = [], [], [], []
            for t in range(T):
                dv_t = dvs[:, t] if dvs is not None else None
                post, prior = self.obs_step(state, act[:, t], embeds[:, t],
                                            dv=dv_t, sample=sample, obs=None,
                                            cont_innov=nu_seq[:, t])
                state = post
                feats_l.append(post.feat[..., :dec_in])
                post_l.append(post.z_logits)
                prior_l.append(prior.z_logits)
                c_qm_l.append(post.c_mean); c_qs_l.append(post.c_std)
                c_pm_l.append(prior.c_mean); c_ps_l.append(prior.c_std)
                if self.dob_enabled:
                    prior_core_l.append(prior.feat[..., :dec_in])
        post_core = torch.stack(feats_l, dim=1)        # (B, T, dec_in)=[h,z,(c),(dv)]
        post_logits = torch.stack(post_l, dim=1)
        prior_logits = torch.stack(prior_l, dim=1)
        ds = None
        if self.dob_enabled:
            if self.dob_active:
                # ONE batched prior decode → CV forecast base (d-free), then the
                # scalar per-CV Kalman filter.  d_t = A·d_{t-1} + K·ν with
                # ν = CV_obs − (base + A·d_{t-1}) ⇒ d_t = (1−K)·A·d_{t-1} + K·(CV_obs − base).
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
            state = RSSMState(h=state.h, z_logits=state.z_logits, z=state.z,
                              d=ds[:, -1], dv=state.dv, c=state.c,
                              c_mean=state.c_mean, c_std=state.c_std)
        else:
            feats = post_core
        # Continuous-latent KL stats + posterior sample (for the cont KL +
        # gain-matching aux loss + disturbance readout).  ``None`` when off.
        cont = None
        if self.cont_dim > 0:
            cont = {
                'post_mean': torch.stack(c_qm_l, dim=1),   # (B,T,cont_dim)
                'post_std': torch.stack(c_qs_l, dim=1),
                'prior_mean': torch.stack(c_pm_l, dim=1),
                'prior_std': torch.stack(c_ps_l, dim=1),
                'sample': post_core[..., core:core + self.cont_dim],
            }
        return feats, post_logits, prior_logits, state, ds, cont

    def decode(self, feat: torch.Tensor) -> torch.Tensor:
        # Scope 2 + DV feedforward: the decoder learns ``g([h, z, (dv)])``; the
        # DV (when fed forward) sits right after the latent core so it is part
        # of the contiguous front slice, while any DOB d-tail beyond it is
        # sliced OFF (re-added by ``apply_dob``).  When DV-feedforward and DOB
        # are both off, ``feat`` is already core-width so this is a no-op slice.
        x = feat[..., :self._decode_in_dim]
        out = self.decoder(x)
        if self.dv_skip is not None:
            core = self.deter_dim + self.stoch_flat_dim + self.cont_dim
            dv = feat[..., core:core + self._dv_feed_dim]
            out = out + self.dv_skip(dv)
        return out


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


def rssm_cont_kl_loss(post_mean: torch.Tensor, post_std: torch.Tensor,
                      prior_mean: torch.Tensor, prior_std: torch.Tensor,
                      free_bits: float = 1.0, dyn_w: float = 0.5,
                      repr_w: float = 0.1
                      ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """KL-balanced free-bits loss for the continuous (diagonal-Gaussian)
    latent — the Gaussian analogue of ``rssm_kl_loss``.

    ``dyn`` term (stop-grad on posterior) trains the prior toward the
    posterior (so the prior learns to ROLL the gain/disturbance forward);
    ``repr`` term (stop-grad on prior) trains the posterior toward the prior.
    The free-bits floor is applied to the mean dim-summed KL.
    """
    def _kl_gauss(mq, sq, mp, sp):
        # KL(N(mq,sq²) || N(mp,sp²)) summed over the last (cont) dim → (B, T).
        var_q = sq * sq
        var_p = sp * sp
        return (torch.log(sp) - torch.log(sq)
                + (var_q + (mq - mp) ** 2) / (2.0 * var_p) - 0.5).sum(dim=-1)

    kl_dyn_raw = _kl_gauss(post_mean.detach(), post_std.detach(),
                           prior_mean, prior_std)
    kl_repr_raw = _kl_gauss(post_mean, post_std,
                            prior_mean.detach(), prior_std.detach())
    fb = torch.tensor(float(free_bits), device=post_mean.device,
                      dtype=kl_dyn_raw.dtype)
    kl_dyn = torch.maximum(kl_dyn_raw.mean(), fb)
    kl_repr = torch.maximum(kl_repr_raw.mean(), fb)
    kl_loss = dyn_w * kl_dyn + repr_w * kl_repr
    diag = {
        'cont_kl_dyn': kl_dyn.detach(),
        'cont_kl_repr': kl_repr.detach(),
        'cont_kl_dyn_raw': kl_dyn_raw.mean().detach(),
    }
    return kl_loss, diag
