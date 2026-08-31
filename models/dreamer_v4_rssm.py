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


_DOB_SCAN_MIX_BUDGET_BYTES: Optional[int] = None


def _dob_scan_mix_budget_bytes() -> int:
    """Cap the T×T Kalman mix by device RAM (identity on a 24 GB A10).

    Closed-form mix is ``T×T×C``; ~16 MiB on this A10 (24 GiB / 1500).
    Smaller GPUs drop the cap so a huge ``seq_len`` cannot OOM; larger
    hosts may mix a longer window before falling back to the sequential
    recurrence.  Floor 4 MiB / ceiling 64 MiB.  CPU smokes keep 16 MiB.
    Cached after the first call (one device per process).
    """
    global _DOB_SCAN_MIX_BUDGET_BYTES
    if _DOB_SCAN_MIX_BUDGET_BYTES is not None:
        return _DOB_SCAN_MIX_BUDGET_BYTES
    try:
        if torch.cuda.is_available():
            idx = torch.cuda.current_device()
            total = int(torch.cuda.get_device_properties(idx).total_memory)
            _DOB_SCAN_MIX_BUDGET_BYTES = int(
                min(64 * 1024 * 1024,
                    max(4 * 1024 * 1024, total // 1500)))
            return _DOB_SCAN_MIX_BUDGET_BYTES
    except Exception:
        pass
    _DOB_SCAN_MIX_BUDGET_BYTES = 16 * 1024 * 1024
    return _DOB_SCAN_MIX_BUDGET_BYTES


def dob_kalman_scan(u: torch.Tensor, coef: torch.Tensor) -> torch.Tensor:
    """Vectorized ``d_t = coef * d_{t-1} + u_t`` with ``d_{-1}=0``.

    ``u`` is ``(B, T, n_cv)``, ``coef`` is ``(n_cv,)``.  Same recurrence as
    the old Python T-loop; that loop was T sequential GPU launches per WM
    step in P2 (``dob_active``).  Closed form
    ``d_t = sum_{s<=t} coef^{t-s} u_s`` via a lower-triangular mix.
    Host-adaptive: if the T×T mix would exceed the device budget (≈16 MiB
    on a 24 GB A10), fall back to the sequential recurrence (huge
    ``seq_len`` / many CVs).  Differentiable in ``u`` and ``coef`` (P2
    trains Kalman A,K through this).
    """
    B, T, C = u.shape
    if T == 0:
        return u
    nbytes = T * T * C * int(u.element_size())
    coef_f = coef.to(dtype=u.dtype)
    if T == 1 or nbytes > _dob_scan_mix_budget_bytes():
        d_prev = torch.zeros(B, C, device=u.device, dtype=u.dtype)
        out = u.new_empty(B, T, C)
        for t in range(T):
            d_prev = coef_f * d_prev + u[:, t]
            out[:, t] = d_prev
        return out
    t_idx = torch.arange(T, device=u.device)
    lags = t_idx.view(T, 1) - t_idx.view(1, T)
    mask = lags >= 0
    lags_f = lags.clamp_min(0).to(dtype=u.dtype)
    pows = coef_f.view(1, 1, C).pow(lags_f.unsqueeze(-1))
    pows = pows.masked_fill(~mask.unsqueeze(-1), 0)
    return torch.einsum('tsc,bsc->btc', pows, u)


def _time_unbind(x: Optional[torch.Tensor]):
    """Unbind time dim=1 once. ``None`` stays ``None``.

    Identity vs ``x[:, t]`` inside a Python T/K loop (one tuple of views
    instead of T advanced-index views per WM step).  Host-adaptive.
    """
    if x is None:
        return None
    return x.unbind(1)


def _append_decode_core(h_l, z_l, c_l, dv_l, st) -> None:
    """Collect ``[h, z_flat, (c), (dv)]`` views for ``_stack_decode_core``.

    Same parts as ``state.feat[..., :dec_in]`` (d-tail is sliced off
    ``feat`` / appended later as ``ds``).  Identity vs per-step ``cat``.
    """
    h_l.append(st.h)
    z_l.append(st.stoch_flat)
    if st.c is not None:
        c_l.append(st.c)
    if st.dv is not None:
        dv_l.append(st.dv)


def _stack_decode_core(h_l, z_l, c_l, dv_l) -> torch.Tensor:
    """``(B, T, dec_in)`` ≡ ``stack(state.feat[..., :dec_in], dim=1)``.

    Main WM encode used to ``cat`` 2–4 views every t then stack T
    cores (100 inner × T=128 on test_sim).  One cat after T stacks.
    Host-adaptive (no extra threads).
    """
    parts = [torch.stack(h_l, 1), torch.stack(z_l, 1)]
    if c_l:
        parts.append(torch.stack(c_l, 1))
    if dv_l:
        parts.append(torch.stack(dv_l, 1))
    return torch.cat(parts, dim=-1)


def cached_zeros_btd(mod, B: int, T: int, D: int, dtype, device,
                     attr: str = '_zeros_btd_cache') -> torch.Tensor:
    """Reuse a ``(B,T,D)`` zero buffer. Identity vs ``torch.zeros`` each call
    as long as callers do not write the buffer in-place (``cat`` / ``detach``
    do not).  Stage-1 DOB ``d_t≡0`` tail is this class.
    """
    key = (int(B), int(T), int(D), str(dtype), str(device))
    cached = getattr(mod, attr, None)
    if cached is None or cached[0] != key:
        z = torch.zeros(B, T, D, device=device, dtype=dtype)
        setattr(mod, attr, (key, z))
        return z
    return cached[1]


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
    # Latent type (2026-08-12).  ``'deterministic'`` (default, P26/P29 RCA) = a
    # continuous tanh latent with NO variational KL (prior/posterior consistency
    # via joint-embedding) — no quantization, so the continuous CV/DV gain is
    # not attenuated (bias-free observer).  ``'categorical'`` = paper DreamerV3
    # (opt-in).  ``latent_noise`` > 0 adds reparameterization noise to the
    # deterministic sample (information-bottleneck regularizer; no quantization).
    latent_type: str = 'deterministic'
    latent_noise: float = 0.0
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
    # DV→DECODER half of the feedforward (2026-08-06, p146 RCA).  The measured
    # dv_t appended to the DECODER input is a MEMORYLESS dv_t→CV_t path: it
    # SKIPS the GRU dead-time (the WM DV response LEADS the plant) and fits
    # dv_step to the TRANSIENT CV (biases the DC gain LOW + starves the dynamic
    # path).  Net-harmful vs the MV, which has NO feedforward yet settles ×0.85
    # with correct dynamics while the DV (with ff) was ×0.76 AND led.  When
    # False the decoder reconstructs CV from the latent core alone (the DV
    # drives CV ONLY through the transition, like the MV), while the DV STAYS in
    # the head-facing ``feat`` (``dv_feedforward``) so the actor still sees the
    # load.  No-op when ``dv_feedforward`` is off or the plant has no DV.
    dv_decoder_feedforward: bool = False
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
    # Roll the DISTURBANCE block of the cont latent DETERMINISTICALLY (prior
    # MEAN, not a sample) in imagination (2026-06-29, p140 RCA).  The cont
    # disturbance is a FEEDFORWARD signal: the actor needs the PREDICTED load,
    # not a per-rollout sampled realization that injects uncontrollable noise
    # into the imagined reward (p140: imag_reward_dv_corr 0.44 buried the action
    # signal → imag_adv_action_corr 0.095 → actor thrash + return_scale cap
    # cascade).  Mirrors the DOB persistence roll (d_t = A·d, no sampling).
    # No-op when sample=False (gain-match / probes already use the mean) or no
    # dist block.
    cont_dist_deterministic_roll: bool = True
    # Roll the GAIN block of the cont latent DETERMINISTICALLY (prior MEAN) in
    # imagination too (2026-08-14, p20 observer-bias RCA).  The GAIN was left
    # SAMPLED here, but the open-loop gain enters NONLINEARLY through the
    # multi-step GRU+decoder, so the strong sample=True gain supervisor
    # (wm_overshoot_loss) optimizes E[f(c_sampled)] (~×0.79) while the ACTOR +
    # transfer matrix use sample=False = f(mean) (~×0.61) — the strong
    # supervisor never trains the actor's path and the sampled gain injects the
    # run-to-run gain variance (×0.61↔×1.09).  Rolling the gain at its MEAN
    # redirects the supervisor onto f(mean) (the actor's belief) AND removes the
    # sampling variance → a stable, unbiased observer gain.  No-op when
    # sample=False or no gain block.
    cont_gain_deterministic_roll: bool = True


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

    def detach(self, keep_c: bool = False) -> 'RSSMState':
        """Truncated-BPTT cut that bounds the recurrent gradient path.

        P25 RCA: detaching the CONTINUOUS gain channel ``c`` on a K-step
        gain-match / isolation roll severs the DC-gain supervisor (forward
        loss still looks healthy; transfer-matrix gain does not).  ``keep_c``
        leaves ``c`` / ``c_mean`` / ``c_std`` attached so the un-quantized
        gain path survives a GRU ``h`` cut.  Forward values are unchanged.
        """
        def _d(t):
            return t.detach() if t is not None else None
        return RSSMState(
            h=_d(self.h), z_logits=_d(self.z_logits), z=_d(self.z),
            d=_d(self.d), dv=_d(self.dv),
            c=(self.c if keep_c else _d(self.c)),
            c_mean=(self.c_mean if keep_c else _d(self.c_mean)),
            c_std=(self.c_std if keep_c else _d(self.c_std)))

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
    """Feature → (n_categoricals × n_classes) latent head.

    ``latent_type='categorical'`` (default, DreamerV3): straight-through
    one-hot sample with a ``unimix`` uniform mixture; prior/posterior
    consistency is the categorical KL (``rssm_kl_loss``, reads ``logits``).

    ``latent_type='deterministic'``: a DETERMINISTIC CONTINUOUS latent
    (``tanh`` of the head output) — NO quantization and NO variational KL, so
    the continuous CV/DV gain is not attenuated (the categorical straight-
    through quantizes it → biased gain).  The forward value IS the latent (no
    sampling); prior/posterior consistency is the joint-embedding predict-
    next-latent MSE (``rssm_joint_embed_loss``).  The returned ``logits`` slot
    carries the same continuous latent (fed to that MSE).
    """

    def __init__(self, in_dim: int, n_categoricals: int, n_classes: int,
                 hidden_dim: int = 256, unimix: float = 0.01,
                 latent_type: str = 'categorical', latent_noise: float = 0.0):
        super().__init__()
        self.n_categoricals = int(n_categoricals)
        self.n_classes = int(n_classes)
        self.unimix = float(unimix)
        self.latent_type = str(latent_type).lower()
        self.latent_noise = float(latent_noise)
        self.net = _MLP(in_dim, n_categoricals * n_classes, hidden_dim,
                        num_layers=2)

    def forward(self, x: torch.Tensor, sample: bool = True
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.net(x).view(*x.shape[:-1], self.n_categoricals,
                                   self.n_classes)
        if self.latent_type == 'deterministic':
            # Continuous bounded latent: no softmax quantization, no KL.  With
            # ``latent_noise`` > 0 the FORWARD sample gets reparameterization
            # noise — an information-bottleneck regularizer that replaces the
            # categorical's stochastic-sampling regularization WITHOUT
            # quantizing (curbs the overfit-to-wrong-gain the pure-deterministic
            # latent showed).  The clean ``mean`` is the joint-embedding target.
            mean = torch.tanh(logits)
            if sample and self.latent_noise > 0.0:
                z = mean + self.latent_noise * torch.randn_like(mean)
            else:
                z = mean
            return mean, z
        # ---- categorical (DreamerV3) ----
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
        # Deterministic cont-disturbance roll in imagination (p140 RCA).
        self.cont_dist_deterministic_roll = bool(
            getattr(cfg, 'cont_dist_deterministic_roll', True))
        # Deterministic cont-GAIN roll in imagination (p20 observer-bias RCA):
        # roll the gain block at its prior MEAN so the strong sample=True gain
        # supervisor trains the actor's sample=False (mean) belief, not the
        # nonlinearly-inflated sampled gain.
        self.cont_gain_deterministic_roll = bool(
            getattr(cfg, 'cont_gain_deterministic_roll', True))
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
        # DV→DECODER half (p146): decoupled from the head feat.  When off the
        # decoder reconstructs CV from the latent core alone (dynamic path);
        # ``_dv_feed_dim`` still keeps the DV in ``feat`` for the heads.
        self.dv_decoder_feedforward = (
            bool(getattr(cfg, 'dv_decoder_feedforward', False))
            and self.dv_feedforward)
        self._dv_decode_dim = self.dv_dim if self.dv_decoder_feedforward else 0
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
                            + self._dv_decode_dim, self.obs_dim,
                            hidden_dim=self.hidden_dim, num_layers=3)
        # Recurrent dynamics: pre-GRU projection then GRUCell.
        self.pre_gru = _MLP(trans_in, trans_in,
                            hidden_dim=self.hidden_dim, num_layers=1)
        self.gru = nn.GRUCell(trans_in, self.deter_dim)
        # Prior p(z'|h') and posterior q(z'|h', embed).
        _lt = str(getattr(cfg, 'latent_type', 'deterministic'))
        _ln = float(getattr(cfg, 'latent_noise', 0.0) or 0.0)
        self.prior_net = _CategoricalLatent(
            self.deter_dim, self.n_categoricals, self.n_classes,
            hidden_dim=self.hidden_dim, unimix=cfg.unimix, latent_type=_lt,
            latent_noise=_ln)
        self.post_net = _CategoricalLatent(
            self.deter_dim + self.embed_dim, self.n_categoricals,
            self.n_classes, hidden_dim=self.hidden_dim, unimix=cfg.unimix,
            latent_type=_lt, latent_noise=_ln)
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
        # Width of the decoder input slice = latent core + cont latent + DV
        # DECODER feedforward (0 unless dv_decoder_feedforward is on).
        return (self.deter_dim + self.stoch_flat_dim + self.cont_dim
                + self._dv_decode_dim)

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
        of a decoded obs tensor (..., obs_dim).  Identity when DOB is off
        or Stage-1-suppressed (``dob_active=False`` → ``d_t≡0``, skip the
        clone+index_add)."""
        if (not self.dob_enabled or d is None
                or not bool(getattr(self, 'dob_active', True))):
            return decoded
        out = decoded.clone()
        out.index_add_(-1, self.cv_index_t, d.to(out.dtype))
        return out

    # ----- transitions --------------------------------------------------
    def _gru_transition(self, prev: RSSMState, prev_action: torch.Tensor,
                        dv: Optional[torch.Tensor] = None
                        ) -> Tuple[torch.Tensor, Optional[torch.Tensor],
                                   Optional[torch.Tensor]]:
        """pre_gru + GRUCell + DOB predict + DV carry.

        Shared by ``img_step`` (prior heads on ``h``) and rest-IC
        ``_posterior_step`` (posterior heads on ``h``; prior_net unused).
        Returns ``(h, d_new, dv_new)``.
        """
        # GRU input = [z_flat ; (c) ; action ; (dv)].  Continuous latent
        # feeds the recurrence so ``h`` carries gain/disturbance forward.
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
        h = self.gru(self.pre_gru(torch.cat(parts, dim=-1)), prev.h)
        # Stage-1 (``dob_active=False``) forces ``d_t≡0`` after the loop
        # and ``d`` is not a GRU input — skip the unused sigmoid·d
        # (P1 rest-IC + main WM T-loop, 100 inner steps).  P2 Kalman
        # still needs the prior predict ``A·d``.
        d_new = prev.d
        if (self.dob_enabled and prev.d is not None
                and bool(getattr(self, 'dob_active', True))):
            d_new = self.dob_decay() * prev.d
        dv_new = dv if self.dv_feedforward else None
        return h, d_new, dv_new

    def img_step(self, prev: RSSMState, prev_action: torch.Tensor,
                 dv: Optional[torch.Tensor] = None,
                 sample: bool = True) -> RSSMState:
        """Imagined (prior-only) step: advance the state with no obs.

        ``dv`` (B, dv_dim) is the exogenous measured-DV input when DV-as-input
        is enabled (``dv_dim > 0``); ``None`` is filled with zeros.  Ignored
        entirely when ``dv_dim == 0`` (paper behaviour)."""
        h, d_new, dv_new = self._gru_transition(prev, prev_action, dv)
        z_logits, z = self.prior_net(h, sample=sample)
        # Continuous-latent prior p(c'|h'): gain persists (carried via h) and
        # the disturbance OU-rolls; both inferred from the recurrent state.
        c_new = c_mean = c_std = None
        if self.cont_dim > 0:
            c_new, c_mean, c_std = self.cont_prior_net(h, sample=sample)
            # Roll selected blocks DETERMINISTICALLY (prior MEAN) in imagination.
            # DISTURBANCE block (p140 RCA): the actor needs the PREDICTED load as
            # a clean feedforward, not a per-rollout sample that buries the action
            # signal in the imagined reward.  GAIN block (p20 observer-bias RCA):
            # the open-loop gain enters nonlinearly through the multi-step
            # rollout, so E[f(c_sampled)] (what the sample=True supervisor trains)
            # ≠ f(mean) (what the sample=False actor/transfer-matrix use) — roll
            # the gain at its mean so the strong supervisor trains the actor's
            # belief and the gain stops varying run-to-run.  No-op when
            # sample=False (c_new == c_mean already) or the block is absent.
            if sample and self.cont_dim > 0 and (
                    self.cont_gain_deterministic_roll
                    or self.cont_dist_deterministic_roll):
                gain_part = (
                    c_mean[..., :self.cont_gain_dim]
                    if self.cont_gain_deterministic_roll
                    else c_new[..., :self.cont_gain_dim])
                dist_part = (
                    c_mean[..., self.cont_gain_dim:]
                    if self.cont_dist_deterministic_roll
                    else c_new[..., self.cont_gain_dim:])
                c_new = torch.cat([gain_part, dist_part], dim=-1)
        return RSSMState(h=h, z_logits=z_logits, z=z, d=d_new, dv=dv_new,
                         c=c_new, c_mean=c_mean, c_std=c_std)

    def _posterior_step(self, prev: RSSMState, prev_action: torch.Tensor,
                        embed: torch.Tensor, dv: Optional[torch.Tensor] = None,
                        sample: bool = True) -> RSSMState:
        """Teacher-forced posterior step without unused prior heads.

        Rest-IC ``last_only`` encode: the next GRU input is this posterior
        ``(z, c)``, never the prior.  ``prior_net`` / ``cont_prior_net`` do
        not feed ``h/z/c_mean``.  Same GRU + posterior nets as
        ``obs_step(..., obs=None)[0]`` when Kalman / two-pass are off
        (Stage-1 P1 rest-IC: ``dob_active=False``, ``cont_dist_dim=0``).
        """
        h, d_new, dv_new = self._gru_transition(prev, prev_action, dv)
        post_in = torch.cat([h, embed], dim=-1)
        post_logits, post_z = self.post_net(post_in, sample=sample)
        c_post = c_post_mean = c_post_std = None
        if self.cont_dim > 0:
            c_post, c_post_mean, c_post_std = self.cont_post_net(
                post_in, sample=sample)
        return RSSMState(h=h, z_logits=post_logits, z=post_z, d=d_new,
                         dv=dv_new, c=c_post, c_mean=c_post_mean,
                         c_std=c_post_std)

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
                         sample: bool = True, store_aux: bool = True,
                         last_only: bool = False,
                         return_feats: bool = True
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

        ``store_aux=False`` skips stacking logits / cont-KL stats (same
        ``feats``).  Isolation's no-grad encode discards those tensors;
        keeping them alive for T steps was a dead alloc on the P1 hot path
        (100 WM steps/iter).  Default ``True`` is the training path.

        ``last_only=True`` returns feats/ds with T=1 (the last step) and
        skips the unused T-stack.  GRU recurrence is identical; last
        ``RSSMState`` ≡ full-roll last state (P45 rest-IC encode only
        needs that IC).  Aux logit/cont stacks stay ``None`` (T-stacks
        would not match).  Isolation / main WM still need the full T.
        ``return_feats=False`` (with ``last_only``) skips even the last
        feat / Stage-1 zero-``d`` tail — rest-IC only reads ``h/z/c_mean``.
        Ignored when ``last_only`` is False.

        When ``last_only`` and Kalman / two-pass are off, the loop uses
        ``_posterior_step`` (skip unused ``prior_net`` / ``cont_prior_net``;
        next GRU input is the posterior).  Identity vs ``obs_step`` last
        ``h/z/c_mean``.  P2 ``dob_active`` still needs the prior core.
        """
        B, T = obs.shape[:2]
        device = obs.device
        embeds = self.embed(obs)                       # (B, T, embed_dim)
        # Exogenous DV input per step (teacher-forced from the real obs).
        dvs = (obs.index_select(-1, self.dv_index_t)
               if self.dv_dim > 0 else None)           # (B, T, dv_dim) | None
        state = self.initial_state(B, device)
        core = self.deter_dim + self.stoch_flat_dim
        # Head-facing feat width = [h, z, (c), (dv)] (the DOB d-tail is appended
        # separately below).  p146: KEEP the DV in the head feat for the actor/
        # critic even when it is dropped from the DECODER input — ``decode()``
        # slices its own (narrower) ``_decode_in_dim`` internally, so the recon
        # path stays DV-free while the heads still see the measured load.
        dec_in = (self.deter_dim + self.stoch_flat_dim + self.cont_dim
                  + self._dv_feed_dim)
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
        # Prior-core is only consumed by the batched DOB decode (P2
        # ``dob_active``) or the cont-dist two-pass.  Stage-1 P1 forces
        # ``d_t≡0`` and discarded the T-list — skip the append.
        _need_prior_core = two_pass or (self.dob_enabled and self.dob_active)
        post_l, prior_l = [], []
        h_l, z_l, c_l, dv_l = [], [], [], []
        ph_l, pz_l, pc_l, pdv_l = [], [], [], []
        c_qm_l, c_qs_l, c_pm_l, c_ps_l = [], [], [], []
        keep_aux = bool(store_aux) and not last_only
        # last_only rest-IC only needs the last RSSMState.  Building
        # post.feat every t was T concatenations of [h,z,c,dv,d] then
        # discarding all but the last.  Kalman / two-pass still need
        # per-step prior core.  Materialize the last post.feat once.
        # Full-T encode stacks h/z/(c)/(dv) then one cat (not T cats).
        _stack_post = not last_only
        # Stage-1 rest-IC: posterior-only (prior heads unused).  P2 DOB /
        # cont-dist two-pass still need obs_step + prior core.
        use_post_only = bool(last_only) and not two_pass and not _need_prior_core
        act_t = _time_unbind(act)
        emb_t = _time_unbind(embeds)
        dv_seq = _time_unbind(dvs)
        if use_post_only:
            pstep = self._posterior_step
            for t in range(T):
                state = pstep(
                    state, act_t[t], emb_t[t],
                    dv=None if dv_seq is None else dv_seq[t],
                    sample=sample)
        else:
            for t in range(T):
                dv_t = None if dv_seq is None else dv_seq[t]
                # COMPILE-EFFICIENT recurrence (2026-06-12): run the (h, z)
                # recurrence with ``obs=None`` so neither the DOB d-update
                # NOR the per-step prior decode enters the compiled loop —
                # the EXPENSIVE decode is hoisted OUT and done ONCE, batched,
                # below.  ``d`` does NOT affect h/z; cont innovation is
                # fed in pass 2.
                post, prior = self.obs_step(state, act_t[t], emb_t[t],
                                            dv=dv_t, sample=sample, obs=None)
                state = post
                if _stack_post:
                    _append_decode_core(h_l, z_l, c_l, dv_l, post)
                if keep_aux:
                    post_l.append(post.z_logits)
                    prior_l.append(prior.z_logits)
                    if self.cont_dim > 0:
                        c_qm_l.append(post.c_mean); c_qs_l.append(post.c_std)
                        c_pm_l.append(prior.c_mean); c_ps_l.append(prior.c_std)
                if _need_prior_core:
                    _append_decode_core(ph_l, pz_l, pc_l, pdv_l, prior)
        if two_pass:
            # ONE batched prior decode → CV forecast → innovation ν, then
            # re-roll with the innovation-driven cont posterior.
            prior_core1 = _stack_decode_core(ph_l, pz_l, pc_l, pdv_l)
            base = self.decode(prior_core1).index_select(-1, self.cv_index_t)
            nu_seq = obs.index_select(-1, self.cv_index_t) - base  # (B, T, n_cv)
            nu_t = _time_unbind(nu_seq)
            state = self.initial_state(B, device)
            post_l, prior_l = [], []
            h_l, z_l, c_l, dv_l = [], [], [], []
            ph_l, pz_l, pc_l, pdv_l = [], [], [], []
            c_qm_l, c_qs_l, c_pm_l, c_ps_l = [], [], [], []
            for t in range(T):
                dv_t = None if dv_seq is None else dv_seq[t]
                post, prior = self.obs_step(state, act_t[t], emb_t[t],
                                            dv=dv_t, sample=sample, obs=None,
                                            cont_innov=nu_t[t])
                state = post
                if _stack_post:
                    _append_decode_core(h_l, z_l, c_l, dv_l, post)
                if keep_aux:
                    post_l.append(post.z_logits)
                    prior_l.append(prior.z_logits)
                    c_qm_l.append(post.c_mean); c_qs_l.append(post.c_std)
                    c_pm_l.append(prior.c_mean); c_ps_l.append(prior.c_std)
                if self.dob_enabled and self.dob_active:
                    _append_decode_core(ph_l, pz_l, pc_l, pdv_l, prior)
        if last_only and not return_feats:
            return None, None, None, state, None, None
        if last_only:
            post_core = state.feat[..., :dec_in].unsqueeze(1)  # (B, 1, dec_in)
        else:
            post_core = _stack_decode_core(h_l, z_l, c_l, dv_l)
        post_logits = (torch.stack(post_l, dim=1) if keep_aux else None)
        prior_logits = (torch.stack(prior_l, dim=1) if keep_aux else None)
        ds = None
        if self.dob_enabled:
            if self.dob_active:
                # ONE batched prior decode → CV forecast base (d-free), then the
                # scalar per-CV Kalman filter.  d_t = A·d_{t-1} + K·ν with
                # ν = CV_obs − (base + A·d_{t-1}) ⇒ d_t = (1−K)·A·d_{t-1} + K·(CV_obs − base).
                prior_core = _stack_decode_core(ph_l, pz_l, pc_l, pdv_l)
                base = self.decode(prior_core).index_select(-1, self.cv_index_t)
                cv_obs = obs.index_select(-1, self.cv_index_t)        # (B, T, n_cv)
                A = self.dob_decay(); K = self.dob_gain()             # (n_cv,)
                u = K * (cv_obs - base)                               # drive (B,T,n_cv)
                coef = (1.0 - K) * A                                  # (n_cv,)
                ds = dob_kalman_scan(u, coef)                         # (B, T, n_cv)
                if last_only:
                    ds = ds[:, -1:]
            else:
                # Stage-1 suppression: d_t ≡ 0 (force g to explain all CV motion).
                # Reuse a zeros buffer (identity; ``cat`` does not write it).
                ds = cached_zeros_btd(
                    self, B, int(post_core.shape[1]), self.n_cv,
                    post_core.dtype, device)
            feats = torch.cat([post_core, ds.detach()], dim=-1)
            state = RSSMState(h=state.h, z_logits=state.z_logits, z=state.z,
                              d=ds[:, -1], dv=state.dv, c=state.c,
                              c_mean=state.c_mean, c_std=state.c_std)
        else:
            feats = post_core
        # Continuous-latent KL stats + posterior sample (for the cont KL +
        # gain-matching aux loss + disturbance readout).  ``None`` when off.
        cont = None
        if self.cont_dim > 0 and keep_aux:
            cont = {
                'post_mean': torch.stack(c_qm_l, dim=1),   # (B,T,cont_dim)
                'post_std': torch.stack(c_qs_l, dim=1),
                'prior_mean': torch.stack(c_pm_l, dim=1),
                'prior_std': torch.stack(c_ps_l, dim=1),
                'sample': post_core[..., core:core + self.cont_dim],
            }
        return feats, post_logits, prior_logits, state, ds, cont

    def img_rollout(self, h0: torch.Tensor, z0: torch.Tensor,
                    actions: torch.Tensor,
                    dvs: Optional[torch.Tensor] = None,
                    sample: bool = True,
                    c0: Optional[torch.Tensor] = None,
                    last_only: bool = False,
                    out: str = 'feat') -> torch.Tensor:
        """Prior-only (imagined) rollout of K steps from ``(h0, z0[, c0])``.

        ``h0`` (Bm, deter_dim), ``z0`` (Bm, n_categoricals, n_classes),
        ``actions`` (Bm, K, A), ``dvs`` (Bm, K, dv_dim) | None,
        ``c0`` (Bm, cont_dim) | None = posterior continuous latent MEAN at
        the start (gain + optional cont-dist).  Prefer ``cont['post_mean']``
        (P28 follow-up 14 / p20: isolation, actor, transfer-matrix are
        ``sample=False`` = f(mean); slicing the sample from feat trains
        ``E[f(c_sampled)]`` on the first GRU step).  ``c0=None`` with
        ``cont_dim>0`` zero-fills — same as ``img_step`` when ``prev.c is None``.
        Returns stacked ``feat`` ``(Bm, K, F)`` = ``[h, z_flat, (c), (dv), (d)]``.
        ``last_only=True`` returns only the K-step value ``(Bm, *)`` — same
        recurrence / last-step as ``stack[:, -1]``, without keeping the
        unused K-stack (gain-match FD Huber is last-step feat).
        ``out`` selects what is stacked (GRU recurrence is identical):
          * ``'feat'`` (default) — full ``state.feat``
          * ``'h'`` — ``state.h`` only (held-rollout drift; no F-stack)
          * ``'obs'`` — ``decode(feat)`` per step.  Pointwise MLP ⇒
            ``stack(decode(feat_k))`` ≡ ``decode(stack(core))`` (overshoot
            does not need the unused F-stack; one decode after K).
        ``last_only`` materializes ``out`` once after the K-loop (no
        intermediate decode / feat copies).

        P28 follow-up 12: overshoot / held-rollout used to omit ``c0``, so
        the open-loop gain supervisor trained a ``c=0`` GRU path while
        isolation / gain-match / the actor / transfer-matrix start from
        posterior ``c`` (p20 family: supervisor ≠ metric path).
        Follow-up 14: pass the posterior MEAN, not the reparameterized sample.

        Compiled the SAME way as ``rollout_observed`` (see ``maybe_compile``):
        capturing the whole K-step ``img_step`` loop as ONE graph removes the
        per-step Python / kernel-launch overhead that otherwise makes the
        multi-step WM aux losses (latent-overshoot + held-rollout +
        gain-match FD + isolation TBPTT chunks) launch-bound.  Eager default
        still wins by stacking independent rolls on the batch dim
        (gain-match: baseline + one step per MV/DV).  Isolation TBPTT is
        applied *between* ``img_rollout`` chunks in train.py (``h``-only
        ``keep_c``) so compile-on always sees a TBPTT-free graph.
        """
        Bm = h0.shape[0]
        K = actions.shape[1]
        c = None
        if self.cont_dim > 0:
            c = (c0 if c0 is not None else torch.zeros(
                Bm, self.cont_dim, device=h0.device, dtype=h0.dtype))
        if out not in ('feat', 'h', 'obs'):
            raise ValueError(f'img_rollout out={out!r}')
        state = RSSMState(
            h=h0,
            z_logits=torch.zeros(Bm, self.n_categoricals, self.n_classes,
                                 device=h0.device, dtype=h0.dtype),
            z=z0, c=c)
        feats = None if last_only else []
        h_l = z_l = c_l = dv_l = None
        if not last_only and out != 'h':
            h_l, z_l, c_l, dv_l = [], [], [], []
        img_step = self.img_step
        out_h = out == 'h'
        out_obs = out == 'obs'
        act_k = _time_unbind(actions)
        dv_seq = _time_unbind(dvs)
        for k in range(K):
            dv_k = None if dv_seq is None else dv_seq[k]
            state = img_step(state, act_k[k], dv=dv_k, sample=sample)
            if last_only:
                continue
            if out_h:
                feats.append(state.h)
            else:
                _append_decode_core(h_l, z_l, c_l, dv_l, state)
        if last_only:
            if out_h:
                return state.h
            if out_obs:
                return self.decode(state.feat)
            return state.feat                                 # (Bm, *)
        if out_h:
            return torch.stack(feats, dim=1)                  # (Bm, K, H)
        core = _stack_decode_core(h_l, z_l, c_l, dv_l)
        if out_obs:
            return self.decode(core)                          # (Bm, K, obs)
        return core                                           # (Bm, K, F)

    def decode(self, feat: torch.Tensor) -> torch.Tensor:
        # Scope 2 + DV feedforward: the decoder learns ``g([h, z, (dv)])``; the
        # DV (when fed forward) sits right after the latent core so it is part
        # of the contiguous front slice, while any DOB d-tail beyond it is
        # sliced OFF (re-added by ``apply_dob``).  When DV-feedforward and DOB
        # are both off, ``feat`` is already core-width so this is a no-op slice.
        x = feat[..., :self._decode_in_dim]
        out = self.decoder(x)
        return out


def stream_serve_step(dyn, state, prev_action: torch.Tensor,
                      obs_t: torch.Tensor, sample: bool = False):
    """One on-policy / val posterior step matching ``rollout_observed``.

    Training ``_realsim_actor_critic_step`` re-encodes with measured DV
    sliced from obs and batched Kalman when ``dob_active``. Collect/val
    used to call ``_posterior_step`` / ``obs_step`` with ``dv=None`` and
    ``obs=None`` (GRU zero-fills DV; ``d_t`` decays from 0). The actor
    then trained on a different ``feat=[h,z,(c),(dv),d_t]`` than it
    acted on — the P3 train/serve hole (P45–P47 bang-bang / limit-ride).

    ``obs_t`` is ``(B, obs_dim)`` or ``(obs_dim,)``. Duck-typed for
    RSSM and TSSM. ``d`` is not a GRU input, so Kalman here matches
    the batched ``dob_kalman_scan`` on feats; Stage-1
    (``dob_active=False``) still skips Kalman like the teacher.
    """
    if obs_t.dim() == 1:
        obs_t = obs_t.unsqueeze(0)
    emb = dyn.embed(obs_t)
    dv = None
    if int(getattr(dyn, 'dv_dim', 0) or 0) > 0:
        dv = obs_t.index_select(-1, dyn.dv_index_t)
    dob_live = (bool(getattr(dyn, 'dob_enabled', False))
                and bool(getattr(dyn, 'dob_active', True)))
    if dob_live:
        post, _ = dyn.obs_step(
            state, prev_action, emb, dv=dv, sample=sample, obs=obs_t)
        return post
    pstep = getattr(dyn, '_posterior_step', None)
    if callable(pstep):
        return pstep(state, prev_action, emb, dv=dv, sample=sample)
    post, _ = dyn.obs_step(
        state, prev_action, emb, dv=dv, sample=sample, obs=None)
    return post


def alloc_pinned_obs_host(device, obs_dim: int):
    """Pinned 1-D host row for H2D obs copies.  ``None`` off CUDA."""
    if getattr(device, 'type', '') != 'cuda':
        return None
    try:
        return torch.empty(int(obs_dim), dtype=torch.float32, pin_memory=True)
    except Exception:
        return None


def copy_obs_row(dst: torch.Tensor, row, host: Optional[torch.Tensor] = None
                 ) -> None:
    """Copy a 1-D obs vector into ``dst[0]``.

    Collect/val used to ``from_numpy`` a new CPU tensor every step.
    Optional pinned ``host`` (from ``alloc_pinned_obs_host``) stages the
    row so the H2D can be non-blocking.  Identity values.
    """
    src = row if isinstance(row, torch.Tensor) else torch.as_tensor(
        row, dtype=torch.float32)
    src = src.detach().to(dtype=torch.float32).reshape(-1)
    if host is not None:
        host.copy_(src)
        dst[0].copy_(host, non_blocking=True)
    else:
        dst[0].copy_(src)


# ---------------------------------------------------------------------------
# P3 collect: CUDA-graph one frozen-RSSM stream_serve_step (B=1)
# ---------------------------------------------------------------------------

_RSSM_STATE_TENSORS = (
    'h', 'z_logits', 'z', 'd', 'dv', 'c', 'c_mean', 'c_std')


def _rssm_dob_live(dyn) -> bool:
    return (bool(getattr(dyn, 'dob_enabled', False))
            and bool(getattr(dyn, 'dob_active', True)))


def _clone_rssm_state(state) -> RSSMState:
    kw = {}
    for name in _RSSM_STATE_TENSORS:
        t = getattr(state, name, None)
        kw[name] = None if t is None else t.detach().clone()
    return RSSMState(**kw)


def _copy_rssm_state(dst: RSSMState, src: RSSMState) -> None:
    for name in _RSSM_STATE_TENSORS:
        a = getattr(dst, name)
        b = getattr(src, name)
        if a is None and b is None:
            continue
        if a is None or b is None:
            raise RuntimeError(
                f'collect-serve CUDA graph state {name!r} None mismatch')
        if a.dtype != b.dtype or a.device != b.device:
            a.copy_(b.to(device=a.device, dtype=a.dtype))
        else:
            a.copy_(b)


class CollectServeCudaGraph:
    """Replay one frozen-RSSM ``stream_serve_step`` (B=1 collect/val).

    Policy stays eager (σ sampling + in-place Adam writes). Observer
    weights are frozen in P3; replay reads those parameter addresses.
    Host-adaptive: CPU / TSSM / capture-fail stay on eager
    ``stream_serve_step``. Transient VRAM skip does not pin fail.
    """

    __slots__ = (
        'graph', 'in_state', 'out_state', 'feat', 'obs', 'prev_a',
        'dob_live')

    def __init__(self, graph, in_state, out_state, feat, obs, prev_a,
                 dob_live: bool):
        self.graph = graph
        self.in_state = in_state
        self.out_state = out_state
        self.feat = feat
        self.obs = obs
        self.prev_a = prev_a
        self.dob_live = bool(dob_live)

    def reset(self, state) -> None:
        _copy_rssm_state(self.in_state, state)
        self.prev_a.zero_()

    def replay(self) -> torch.Tensor:
        self.graph.replay()
        _copy_rssm_state(self.in_state, self.out_state)
        return self.feat


def _capture_collect_serve_cuda_graph(
        dyn, example_state, device, obs_dim: int, act_dim: int,
        dob_live: bool):
    """Capture or ``(None, pin_fail)``. VRAM skip is ``(None, False)``."""
    if not hasattr(torch.cuda, 'CUDAGraph'):
        return None, True
    try:
        free, _total = torch.cuda.mem_get_info(device)
        if int(free) < 128 * 1024 * 1024:
            print('[collect] serve CUDA graph skipped '
                  f'(free VRAM {free / 1024 ** 2:.0f} MiB < 128); eager',
                  flush=True)
            return None, False
    except Exception:
        pass
    try:
        static_obs = torch.empty(
            1, int(obs_dim), device=device, dtype=torch.float32)
        static_prev_a = torch.zeros(
            1, int(act_dim), device=device, dtype=torch.float32)
        in_state = _clone_rssm_state(example_state)
        side = torch.cuda.Stream(device=device)
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                tmp = stream_serve_step(
                    dyn, in_state, static_prev_a, static_obs, sample=False)
                _copy_rssm_state(in_state, tmp)
        torch.cuda.current_stream().wait_stream(side)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            out_state = stream_serve_step(
                dyn, in_state, static_prev_a, static_obs, sample=False)
            feat = out_state.feat
        print('[collect] serve CUDA graph captured '
              '(B=1 stream_serve_step)', flush=True)
        return CollectServeCudaGraph(
            g, in_state, out_state, feat, static_obs, static_prev_a,
            dob_live), False
    except Exception as exc:
        print(f'[collect] serve CUDA graph skipped ({exc!r}); eager '
              'stream_serve_step', flush=True)
        return None, True


def get_collect_serve_cuda_graph(dyn, example_state, device, obs_dim: int,
                                  act_dim: int):
    """Lazy B=1 collect graph. ``None`` → eager ``stream_serve_step``."""
    if getattr(device, 'type', '') != 'cuda':
        return None
    if not torch.cuda.is_available():
        return None
    if type(example_state).__name__ != 'RSSMState':
        return None
    if getattr(dyn, '_collect_serve_cg_fail', False):
        return None
    dob_live = _rssm_dob_live(dyn)
    existing = getattr(dyn, '_collect_serve_cg', None)
    if existing is not None and existing.dob_live == dob_live:
        return existing
    captured, pin_fail = _capture_collect_serve_cuda_graph(
        dyn, example_state, device, obs_dim, act_dim, dob_live)
    if pin_fail:
        dyn._collect_serve_cg_fail = True  # type: ignore[attr-defined]
    if captured is None:
        return None
    dyn._collect_serve_cg = captured  # type: ignore[attr-defined]
    return captured


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


def rssm_joint_embed_loss(post_latent: torch.Tensor,
                          prior_latent: torch.Tensor) -> torch.Tensor:
    """Joint-embedding (predict-next-latent) consistency for the deterministic
    continuous latent.

    Trains the prior (dynamics) to predict the posterior's latent:
    ``mean‖ prior − sg(posterior) ‖²`` (mean over the latent dims and (B, T)).
    REPLACES the variational KL for imagination consistency (the deterministic
    latent has no distribution to KL) and shapes the latent to be dynamics-
    predictable (TD-MPC2 / objective mismatch, Lambert 2020).  Mean (not sum)
    over dims keeps it scale-stable across latent sizes.  Stop-grad on the
    posterior (the encoder is the anchor).
    """
    return (prior_latent - post_latent.detach()).pow(2).mean()
