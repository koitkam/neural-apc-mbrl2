"""Three-phase Dreamer 4 trainer for APC.

Reference: Hafner, Yan, Lillicrap (2025) — "Training Agents Inside of
Scalable World Models", arXiv:2509.24527.

Algorithm 1 (paper-faithful, adapted to single-task online APC sim):

    Phase 1 — pretrain world model
        Collect random-action episodes; train tokenizer recon (eq. 5)
        + dynamics shortcut forcing (eq. 7).

    Phase 2 — agent finetune
        Continue collecting random-action episodes (paper-strict for the
        offline-data setting); keep eq. 5 + eq. 7 live and add policy +
        reward multi-token-prediction heads (eq. 9).

    Phase 3 — imagination training
        Freeze tokenizer + dynamics + reward head. Snapshot the current
        policy as the PMPO prior. Sample dataset contexts from the
        replay buffer; imagine H steps using K=4 shortcut sampling per
        step; train the value head with TD-λ (eq. 10) and the policy
        head with PMPO (eq. 11). Run periodic evaluation episodes for
        the return-window score (no buffer writes).

Phase budget: ``cfg.phaseN_frac`` of ``cfg.total_steps`` for N ∈ {1, 2, 3}.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import time
from collections import deque
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from models.dreamer_v4 import (  # noqa: F401
    DreamerV4, DreamerV4Config,
    shortcut_forcing_loss, pmpo_loss, reinforce_actor_loss,
)
from utils.sim_factory import create_sim, resolve_sim_metadata
from utils.objective_runtime import compute_objective_components
from utils.runtime_setpoints import RuntimeSetpointManager, RuntimeSetpointConfig
from utils.training_disturbance import (
    build_training_disturbance_schedule,
    apply_disturbance_schedule,
)
from utils.hidden_disturbance import (
    get_phase_disturbance_prob,
    maybe_build_hidden_disturbance,
)
from utils.noise_config import noise_curriculum_scale
from utils.derived_observations import (
    DerivedFeatures,
    derived_observables_enabled,
    derived_observables_window,
)
from utils.agent_utils import (
    load_objective_weights, load_objective_bounds, load_full_objective_spec,
    action_to_control, control_to_action,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    # ----- Tokenizer -----
    tok_hidden: int = 256
    z_dim: int = 24
    mae_p_max: float = 0.0   # disabled — see Tokenizer docstring (recon collapse)

    # ----- Dynamics transformer -----
    d_model: int = 256
    n_layers: int = 6
    n_heads: int = 8
    ff_mult: int = 4
    n_register: int = 4
    k_max: int = 4
    tau_n_bins: int = 32
    soft_cap: float = 50.0

    # ----- Heads -----
    n_action_bins: int = 21
    head_hidden: int = 256
    head_n_layers: int = 2
    # Policy distribution type.  ``'continuous'`` (default) uses a
    # tanh-squashed Gaussian (TanhNormal) actor — appropriate for chemical
    # / process control where the action is a continuous valve / setpoint
    # in [-1, 1] and discrete-bin actors collapse to corner-bins for
    # fine-grained tracking tasks.  ``'discrete'`` retains the paper's
    # categorical-bin head for back-compat / Atari-style sims.
    policy_type: str = 'continuous'
    # ``policy_init_log_std=-2.0`` (σ ≈ 0.135 in pre-tanh space) gives
    # initial actions near mid-MV → first episodes stay in-bounds → the
    # buffer contains some non-catastrophic transitions before the
    # actor has learned anything.  Old default (-0.5 / σ ≈ 0.6)
    # produced uniform-violation rollouts that left REINFORCE with no
    # positive advantage to amplify (see 2026-05-05 root-cause).
    policy_init_log_std: float = -2.0
    # DreamerV3 §3 stable defaults: σ ∈ [0.1, 1.0] (log_std ∈ [-2.3, 0]).
    # Both bounds are auto-derived in ``auto_tune_seed_buffer`` from the
    # plant-aware ``baseline_seed_action_std`` (which itself flows from
    # MV authority).  The dataclass values below are sentinels that
    # mark "use the auto-derived value" — user / env-var overrides win
    # via the ``_AUTO_TUNE_FIELD_DEFAULTS`` mechanism.
    policy_log_std_min: float = -2.3
    policy_log_std_max: float = 0.0
    # ----- σ_max / σ_min auto-tune formula inputs (P59, 2026-05-27) -----
    # Previously read directly via ``os.environ.get`` inside
    # ``auto_tune_seed_buffer`` (lines ~2490).  Promoted to TrainConfig
    # fields so they (a) appear in ``run_plan.json → config``, (b) are
    # whitelisted in ``ENV_OVERRIDES`` per the documented launch
    # contract, and (c) are picked up by the paper-defaults-audit skill.
    # Defaults preserve the legacy code constants — no behaviour change
    # without an explicit override.  See `dreamer-hyperparameter-
    # rationale` skill for paper-grounded recommendations (post-p24:
    # mult=0.7, cap=0.20 are the recommended values for cascade-prone
    # plants; defaults here are the conservative legacy values that
    # match historical run reproducibility).
    sigma_max_mult: float = 1.0       # σ_max = sigma_max_mult × σ_seed
    sigma_max_floor: float = 0.10     # hard floor on σ_max
    sigma_max_cap: float = 0.30       # hard ceiling on σ_max
    sigma_min_ratio: float = 2.5      # σ_min = σ_max / sigma_min_ratio
    # PMPO entropy bonus. Auto-derived from the auto-tuned σ_max in
    # ``auto_initialize_hyperparams`` as ``η = η_v3 × σ_max / σ_v3_ref``
    # (V3 paper default 3e-4 anchored at σ=1.0). Recovers V3 exactly
    # for paper-scale plants and shrinks linearly for tighter
    # exploration bands typical of process control.
    # The dataclass default below (1e-4) is the sentinel value used
    # only when auto-tune is disabled or fails; it matches what the
    # formula produces for σ_max ≈ 0.33. Set explicitly via env var
    # PMPO_ENTROPY_COEF_BASELINE / PMPO_ENTROPY_SIGMA_REF or override
    # the cfg field directly to bypass the auto-derivation.
    pmpo_entropy_coef: float = 1e-4
    # Actor loss type. ``'reinforce'`` (DreamerV3 §3) is robust across
    # simulators — V3 used this single recipe across 150+ tasks. The
    # ``'pmpo'`` option uses V4's eq. 11 advantage-sign-split loss; this
    # is the paper-faithful V4 actor for **discrete** actions, but it is
    # numerically unstable for continuous TanhNormal actors because its
    # negative-advantage branch is unbounded below.  We default to V3
    # REINFORCE which has bounded gradients and proven cross-task
    # robustness; switch to PMPO only for discrete-action sims if
    # desired (env: DREAMER_ACTOR_LOSS=pmpo).
    actor_loss_type: str = 'reinforce'
    # ----- Actor KL trust region (p136, 2026-06-21) -----
    # The REINFORCE surrogate above already NORMALISES the advantage
    # (adv/return_scale + disturbance-baseline + clip) — that IS the paper's
    # "normalized_advantage", and on its own it lets the policy HUNT (p135:
    # actor_logp_mean swung 0.35↔10.7; the actor underperformed the static
    # expert).  ``reinforce_actor_loss`` carries a built-in KL-to-prior term
    # (a trust region) that was hard-disabled (kl_coef=0).  ``actor_kl_coef``>0
    # enables it: a GENTLE penalty on KL(π ‖ π_prior) toward a periodically
    # refreshed snapshot of the recent policy, damping the per-update policy
    # swing WITHOUT PMPO's continuous-action-unstable advantage-sign split.
    # Sim-agnostic (unitless KL nats).  0.0 = legacy (no trust region); PMPO
    # remains the stronger fallback.  ``DREAMER_ACTOR_KL_COEF``.
    # REVERTED 0.3→0.0 (p136 verdict 2026-06-22): the KL trust region did NOT
    # prevent the entropy collapse (actor_kl_pen faded to 0.001 — a refreshed-
    # prior TR follows a SLOW σ-collapse, the wrong mechanism) and only added a
    # confound.  The p136 collapse was DOWNSTREAM of the missing disturbance in
    # imagination (DOB removed); the continuous-latent disturbance restores the
    # objective.  Re-enable only if the policy HUNTS once the objective is fixed.
    actor_kl_coef: float = 0.0
    # Phased-P3 trust-region prior refresh cadence (iters).  The phased run
    # snapshots π_prior ONCE at P3 start (near-uniform) — a static KL anchor is
    # an entropy pull, NOT a trust region.  Refreshing every N P3 iters makes
    # KL(π‖π_prior) a MOVING trust region (penalises rapid change from the
    # recent policy → anti-hunting).  0 = keep the once-at-start snapshot.
    # Sim-agnostic.  ``DREAMER_P3_PRIOR_REFRESH_ITERS``.
    p3_prior_refresh_iters: int = 5

    # ----- Plant / windowing -----
    lookback: int = 32        # transformer context length T_ctx
    sample_rate: int = 5
    episode_length: int = 600

    # ----- Training overall -----
    total_steps: int = 100_000
    train_steps_per_iter: int = 100
    ep_per_iter: int = 5
    seq_len: int = 64
    batch_size: int = 16
    horizon: int = 15

    # ----- Phase budget fractions (paper Algorithm 1) -----
    phase1_frac: Optional[float] = None
    phase2_frac: Optional[float] = None
    phase3_frac: Optional[float] = None

    # ----- Training mode (neural-apc-mbrl fork, 2026-06-06) -----
    # ``phased`` (default, legacy Dreamer-4 curriculum): P1 WM -> P2 reward+BC
    # -> P3 actor+critic, with phase boundaries.  ``joint`` (DreamerV1/V2/V3
    # style): after the seed-buffer PREFILL, co-train WM + actor + critic EVERY
    # step from step 1 (no phase boundaries) by running the P3 update path for
    # the entire run.  Joint mode eliminates the phase-boundary pathologies
    # (recon spike at P1->P2 grad-bleed, cold-critic cascade at P2->P3,
    # checkpoint-discard) since WM/actor/critic co-adapt from the start.  The
    # critic warmup (p3_critic_warmup_iters) still runs at the very start so
    # the value head calibrates before actor coupling.  DREAMER_TRAIN_MODE.
    train_mode: str = 'phased'
    # ----- Actor training data source (mbrl2 fork) -----
    # ``realsim`` (default; the only supported mode — imagination was removed):
    # the WM(RSSM)+DOB are a FROZEN OBSERVER and the actor-critic trains on
    # λ-returns from REAL rollouts of the true simulator with domain
    # randomisation (``_realsim_actor_critic_step``).  Exact policy gradient
    # w.r.t. the true dynamics, real-return-grounded critic (no cascade),
    # DreamerV3 scale-invariant normalisation.  DREAMER_ACTOR_SOURCE.
    actor_train_source: str = 'realsim'
    # joint mode: re-snapshot the PMPO prior policy every N iters (0 = once at
    # start, like phased P3).  A slowly-refreshed prior keeps the KL anchor
    # from going stale over a long single-phase run.
    joint_prior_refresh_iters: int = 0

    # ----- Optimizers -----
    lr_world: float = 1e-4
    lr_actor: float = 3e-5    # DreamerV3 §3; auto-bumped to 1e-4 for
                              # discrete heads (back-compat).  See
                              # build_optimizers in this module.
    # Bumped 2026-05-14 (run_p22 RCA: critic_r=-0.50 with wider σ_max,
    # critic could not keep up with the broader experience distribution
    # induced by SEED_TARGET_CV_FRAC=0.20 + SIGMA_MAX_CAP=0.30).
    # Bumped further 2026-05-18 (run_p24 RCA: critic_r=+0.19, critic
    # informativeness still too weak — value head failed to converge
    # even after σ_max was bounded.  8e-5 is the DreamerV3 paper-
    # validated upper bound (used for Atari tasks); plant-agnostic
    # since the formula is set absolutely, not as a ratio).  Override
    # via ``DREAMER_LR_CRITIC`` env var (now wired in single_run.py)
    # or TrainConfig field.
    lr_critic: float = 8e-5
    grad_clip: float = 100.0  # DreamerV3 default; was 1000 (too loose,
                              # let the actor explode at the BC→PMPO
                              # transition).

    # ----- Loss weights -----
    # recon_scale: Phase-1 tokenizer reconstruction weight on **z-scored**
    # observations (running stats from ``_build_obs_vec``).
    #
    # History (2026-05-06 RCA): with recon_scale=0 the tokenizer received
    # no direct supervision on the observation channels and shortcut-
    # forcing alone failed to anchor the encoder to plant state — WM
    # next-state correlation r≈0, reward head r≈-0.4, critic r≈-0.05
    # (validate diag against ckpt 140 of run_p0adapt).  The 2026-05-03
    # "encoder collapse on raw obs" risk that motivated recon_scale=0
    # does not apply once obs is z-scored (marginal mean = 0, std = 1
    # per channel ⇒ no shortcut to "predict the mean").  We re-enable
    # recon at a small weight (0.1) so SF + recon jointly anchor the
    # latent.  This matches DreamerV4 §3.1 "tokenizer trained jointly
    # with reconstruction loss" in spirit (we deviate only in using a
    # thin linear projection without VQ for low-D vector obs).
    recon_scale: float = 0.1
    # ---- (WM autoencoder lever, 2026-06-09) per-channel CV recon weight ----
    # The posterior-prior probe identified the WM's residual gain bias as
    # dominated by the AUTOENCODER (real→posterior ~0.85), NOT the prior/
    # free-bits.  The uniform recon MSE drowns the small CV step-gain under the
    # high-variance MV/DV channels, so the decoder under-fits the very channel
    # whose gain the controller needs.  When ``!= 1.0`` the CV obs channels'
    # recon error is scaled by this factor (others stay at 1.0), forcing the
    # autoencoder to reproduce the CV response faithfully.  Applied in BOTH
    # backbones (RSSM decode-MSE + SF tokenizer recon) so the fix transfers to
    # TSSM.  ``1.0`` = uniform = paper / p106-baseline behaviour (identity).
    # Resolved CV obs indices live in ``cv_obs_indices`` (set at runtime).
    # Default 4.0 = p117 curriculum-recipe (promoted 2026-06-14; was 1.0).
    wm_recon_cv_weight: float = 4.0
    cv_obs_indices: Tuple[int, ...] = ()
    sf_scale: float = 1.0
    # P2+P3 reward-MTP loss weight (Dreamer-V4 paper Eq. 9).  Lowered
    # 1.0 → 0.3 on 2026-05-23 (P43 RCA, run_20260523_p43_data_fixed):
    # diag_grad_rmtp/(diag_grad_recon+diag_grad_sf+diag_grad_rmtp)
    # climbed from 0.000 (P1 end) → 0.139 (P2 iter 55) → 0.313 (P2 iter
    # 65), and the WM fidelity probe collapsed from best_h=15/15 (iter
    # 60) to best_h=0/15 (iter 70) the SAME iter the ratio peaked.
    # Same encoder-conflict mechanism that broke P1 (fixed in P40); the
    # reward head's gradient backflow into the shared encoder/dynamics
    # was pulling the representation away from forward dynamics.  0.3
    # keeps the ratio under ~0.10 (linear scaling from measured 0.31)
    # while remaining a paper-compatible weight knob.  History: 4.0
    # (P5) → 1.0 (P40, 2026-05-22) → 0.3 (P43 root-cause).  Override
    # with DREAMER_REWARD_SCALE_LOSS.
    reward_scale_loss: float = 0.3
    # (2026-06-07) Exclude EXPERT-injected transitions from the reward-head
    # (reward-MTP) supervision.  The expert seeds + P3 re-injection ride the
    # economic constraint edge in a NARROW, low-variance reward region; with
    # them in the reward-MTP target the head fits that shifted distribution and
    # ANTI-correlates on the broader on-policy/validation distribution
    # (reward_head_r dropped 0.96->0.20 in the original repo's P83, -0.30 in the
    # mbrl p95 joint run) -> miscalibrated imagined reward -> the critic's
    # flat-pessimistic value (img_ret pinned at -H) -> no actor advantage
    # gradient.  With this ON the reward head trains ONLY on non-expert steps
    # (on-policy + exploration + PRBS + baseline) so it stays calibrated across
    # the policy's TRUE distribution.  BC still clones the expert (separate
    # mask) — only the REWARD supervision drops them.  DREAMER_REWARD_HEAD_EXCLUDE_EXPERT.
    # PROMOTED to default ON (2026-06-07): p98/p99 confirmed it lifts the
    # validation reward_head_r from -0.30 (anti-correlated) to +0.30 with no
    # downside (BC still clones the expert via a separate mask).  Orthogonal to
    # the return-cap cascade fix.  Set DREAMER_REWARD_HEAD_EXCLUDE_EXPERT=0 to revert.
    reward_head_exclude_expert: bool = True
    # P1 (WM pretrain) reward-MTP loss weight.  Paper default 0.0:
    # Dreamer-V4 §3 trains the WM (tokenizer + dynamics) without the
    # reward head in pretrain.  The reward head joins in agent
    # fine-tune (Phase 2, Eq. 9).  We kept this at the same weight as
    # P2 from 2026-05-06 through P39 — that wired reward-head gradient
    # into the dynamics head via z_clean.detach(), producing a strong
    # asymmetric gradient that destabilised the dynamics transformer
    # once recon+sf had nearly converged.  Symptom: H=15 WM fidelity
    # peaked around iter 20 then collapsed to ~0 (P38 + P39 cliff).
    # Restored to paper-default 0.0 on 2026-05-22 after P40 diag-B
    # ablation (DREAMER_DIAG_DISABLE_REWARD_MTP_IN_P1=1) confirmed the
    # cliff disappears.  Override with DREAMER_REWARD_MTP_WEIGHT_P1
    # if you want to re-enable a small P1 weight.
    reward_scale_loss_p1: float = 0.0
    # ---- Adaptive negative-tail clip (P62, 2026-05-28) ----------------
    # When the operating-region reward distribution is heavily left-
    # skewed (rare large negative spikes from cliff penalties), the
    # symlog-twohot critic bins collapse positive reward into the same
    # bin as zero — the critic loses signal for "this is good", actor
    # has no gradient direction, and a critic-pessimism cascade follows
    # (P56 RCA + P58b/P59/P60/P61 falsified knob A/Bs).  The fix is to
    # clip the raw-reward negative tail to a multiple of the operating-
    # region p95(|raw|) so the distribution becomes roughly symmetric
    # in symlog space.  Both knobs below are dimensionless ratios →
    # sim-agnostic.  Triggered automatically right after
    # ``calibrate_reward_scale`` from the calibration sample; user's
    # explicit ``DREAMER_REWARD_RAW_CLIP_MIN`` always wins.
    #
    # ``reward_clip_asymmetry_threshold``: ratio |raw_min|/max(|raw_max|,
    # raw_abs_p95_full) above which the adaptive clip activates.  20.0
    # is the P56 RCA threshold (memo preferences.md item 32).  Set to a
    # very large number (e.g. 1e9) to disable the adaptive clip
    # globally.
    reward_clip_asymmetry_threshold: float = 20.0
    # ``reward_clip_tail_k``: when the clip activates, the negative
    # tail is clipped at ``-k × raw_abs_p95_full``.  k=3.0 keeps the
    # clip well outside the operating-region typical magnitude (so
    # ~95% of operating-region rewards are untouched) while bringing
    # the asymmetry ratio to roughly 3 — symlog support becomes
    # near-symmetric and the bin distribution recovers.
    reward_clip_tail_k: float = 3.0
    # ---- Return-scale absolute cap (P79, 2026-06-02) ---------------
    # The (p95-p05)-spread EMA normaliser at ``DreamerV4.update_return_scale``
    # tracks a monotonically growing spread on the critic-pessimism
    # cascade — critic targets grow → spread grows → next critic step
    # has even larger targets, runaway feedback (P58b: 12.2× over 55
    # iters; P77: 109× → bootstrap_cascade early-stop, frozen actor).
    # The per-step GROWTH-RATE clamp tried earlier (P63) REGRESSED
    # (-134997 vs -795) because it also throttled legitimate early-iter
    # spread growth.  The working Cursor APC-Dreamer reference instead
    # uses an ABSOLUTE cap (``max_scale=500`` in its percentile_scale):
    # it never touches normal growth, only arrests the runaway once the
    # spread is implausibly large.  Dimensionless (return units) and
    # sim-agnostic because returns are themselves bounded by the
    # bounded-reward remap (≈ B·H).  Set 0.0 to disable (uncapped EMA).
    return_scale_abs_cap: float = 500.0

    # ---- A' : dense potential-based reward shaping (P66-RCA, 2026-05-29) ----
    # The economic objective is a one-sided cliff (raw_abs_p95≈38 on the
    # negative/violation side, positive tail p95≈0.15 → tail_asymmetry≈142):
    # ~0 reward when in-band, a large penalty on CV violation, almost no
    # POSITIVE gradient toward the band interior.  Under a drifting WM the
    # imagined rollouts keep hitting the negative cliff → growing-negative
    # λ-returns that FUEL the critic-pessimism cascade (critic_target_v_r→0.95).
    # Clipping the negative tail is unsafe here (it would delete the CV-
    # violation safety signal — basis k×positive_tail_p95≈-0.46 truncates ~67%
    # of the penalty mass).  Instead we add a dense, *policy-invariant*
    # potential-based shaping term (Ng et al. 1999):
    #     F_t = coef · (γ·Φ(s_{t+1}) − Φ(s_t)),   Φ(s) ∈ [0,1]
    # where Φ is the normalised margin to the nearest CV band edge (1 at band
    # centre / at an enabled CV target, 0 at the edge, clamped to 0 outside).
    # Because it is potential-based with the SAME γ, it does NOT change the
    # optimal economic policy — it only densifies the learning signal so the
    # policy/critic have a gradient toward the band interior instead of a flat
    # zero.  ``coef`` is in SCALED-reward units (added to ``raw*reward_scale``).
    # TRAINING ONLY: the trainer sets ``env._shaping_enabled=True``; validation
    # scores on the unshaped economic ``info['raw_reward']`` so the audited
    # objective is unchanged.  Set ``reward_shaping_coef=0.0`` to disable.
    reward_shaping_coef: float = 1.0
    # Flat-top safety-margin width (2026-06-16, p125 RCA) for the RANGE/limit
    # case of the shaping potential (no enabled CV target).  The legacy
    # band-keeping potential was a TENT peaked at the band CENTRE, which (a)
    # center-biases the actor away from an economic limit-riding operating point
    # and (b) spreads the safety gradient thinly across the whole band.  Instead
    # Φ is FLAT (=1) across the interior and ramps 0→1 only within a margin band
    # of width ``shaping_safe_margin_frac · half_band`` at each edge — zero
    # center-pull (economics free) with a CONCENTRATED, ~1/frac steeper pull-back
    # exactly in the near-constraint zone where disturbance-driven overshoot
    # happens.  Still potential-based (policy-invariant) and sim-adaptive (a
    # fraction of the plant's own half-band).  ``1.0`` recovers the legacy tent.
    shaping_safe_margin_frac: float = 0.25
    # Fix 2a economic-shaping weight (2026-06-19, p129 RC-A).  Φ = Φ_safe +
    # ``shaping_econ_coef``·gate·Φ_econ, where Φ_econ ∈ [0,1] is a STATE-BASED
    # economic-direction potential (per-channel linear band ramp oriented by the
    # sign of each economic weight — the direction that LOWERS the objective's
    # economic penalty) and ``gate`` ∈ [0,1] is a CV-SAFETY MARGIN gate that
    # SUPPRESSES the economic pull near a constraint (gate→0 at the limit) and
    # enables it only with safe headroom (gate→1 in the interior).  This is the
    # margin-gated successor to the reverted R2a (which, ungated, pushed the CV
    # INTO the high limit under a disturbance-blind WM).  Still potential-based
    # (telescoping, same γ) ⇒ POLICY-INVARIANT (Ng 1999) — gating by a state
    # function keeps it a valid potential — so it densifies the near-invisible
    # economic gradient WITHOUT changing the constrained optimum and is safe on
    # nonlinear plants.  ``0.0`` disables Φ_econ (pure safety shaping).
    # 2026-06-19 (p130 RCA): tempered 0.5→0.25.  Fix 2 (p130) ACTIVATED the
    # actor (imag_adv_action_corr 0→0.74) but on a still-biased WM (gain 0.81)
    # the 0.5 econ push over-drove it → oscillation + imagined_return runaway.
    # A gentler 0.25 lets the now-active actor track the WM while it heals
    # (D1 removal recovers the gain) instead of chasing economics into the WM's
    # bias.  Restore 0.5 once the WM gain is back >0.9.
    shaping_econ_coef: float = 0.25
    # Width (as a fraction of the CV half-band) of the inner SAFE zone over which
    # the economic gate ramps 0→1: gate = clip(margin_to_edge / (frac·half), 0,
    # 1).  ``0.5`` ⇒ economics is fully active only in the inner 50 % and fully
    # OFF at the limit; smaller = econ allowed closer to the edge.  Sim-adaptive
    # (a fraction of the plant's own band).
    shaping_econ_margin_frac: float = 0.5

    # ---- P73 : bounded training reward (cascade root-cause fix) ------
    # The bootstrap-cascade root cause (P69-P71) is UNBOUNDED reward scale:
    # raw economic reward (±50 after clip) × adaptive ``reward_scale`` (≤200)
    # → per-step training reward ±50-200 → imagined H-step λ-returns reach
    # -2000..-6000 → return_scale percentile spread runs away (5756×) → critic
    # pessimism cascade (flat MV).  The working Cursor APC-Dreamer reference
    # bounds the per-step training reward into [-1, 1] via a symlog squash:
    #     r_train = clip( sign(r_raw)·log1p(|r_raw|), -B, B )
    # computed on the UNSCALED economic reward (``reward_scale`` is bypassed
    # when bounding is on — it is meaningless once rewards saturate).  Bounded
    # rewards ⇒ bounded returns ⇒ return_scale CANNOT run away.  Applied to the
    # TRAINING reward only; ``info['raw_reward']`` stays the unshaped economic
    # reward so validation scoring is unaffected.  P73 (2026-05-31) PROMOTED
    # the default to True after the bounded-reward run unfroze the actor
    # (MV moves, mv_violation 0→1.75) and tamed return_scale runaway 5756×→27×.
    # Set DREAMER_BOUND_TRAINING_REWARD=0 to restore legacy (scaled) reward.
    bound_training_reward: bool = True
    # P77: B for the scale-invariant linear remap
    # ``reward = clip(raw * B/reward_clip_ref, -B, B)``.  Raised 1.0→6.0 so
    # the per-step reward spans ~12 twohot bins (head resolution) while
    # imagined returns stay bounded ~B·H (cascade-safe).  See env.step.
    # Default 3.0 = p117 curriculum-recipe (promoted 2026-06-14; was 6.0).
    bound_training_reward_max: float = 3.0
    # Fallback ``reward_clip_ref`` when objective_runtime does not expose
    # one (older comps / degenerate weights); matches the adaptive-clip floor.
    bound_training_reward_ref: float = 50.0

    # ---- P74 : advantage clipping (Cursor stabilizer #3) ------------
    # Clamp the normalised advantage ``adv/scale`` into [-clip, +clip] before
    # the REINFORCE/PMPO actor loss.  Without it a single large imagined
    # return produces an outsized policy-gradient step → jerky, chattering
    # MV (P73 deterministic-eval reversal_rate up to 0.70).  The working
    # Cursor reference clamps at ±4 (single-MV) / ±8 (multi-MV); we default
    # to 8.0.  Set ``advantage_clip=0.0`` to disable (legacy unclamped).
    advantage_clip: float = 8.0

    # ---- C : replay-grounded critic anchor (P66-RCA, 2026-05-29) ----
    # The cascade's root cause is that the P3 critic regresses purely on
    # IMAGINED λ-returns from a frozen, non-convergent WM: targets become
    # ~95% self-bootstrap (critic_target_v_r→0.95, reward <1% of target var),
    # so the critic drifts into a self-consistent growing-negative fixed point
    # of its own making.  This anchor adds a critic loss term on REAL replayed
    # transitions — a TD-λ target built from the buffer's REAL rewards and the
    # slow target-value bootstrap on the REAL latents — so the critic is pinned
    # to genuine reward variance and cannot float free.  ``coef`` weights the
    # anchor vs the imagined critic loss.  Set ``critic_replay_anchor_coef=0.0``
    # to disable (legacy pure-imagination critic).
    # Default 0.0 = p117 recipe: grounding is done by the MC term below, not
    # this anchor (promoted 2026-06-14; was 0.5).
    critic_replay_anchor_coef: float = 0.0

    # ---- B : long-horizon critic-anchor grounding (P85, 2026-06-04) ----
    # The replay anchor above computes its TD-λ target over the FULL real
    # context (``Treal = seq_len`` steps), which spans MANY plant time
    # constants — but it reuses the myopic imagination ``gae_lambda`` (0.90)
    # in its backward recursion, giving it an effective credit horizon of
    # only ~1/(1-γλ) ≈ 10 steps.  A constraint-riding limit cycle whose
    # period (~40 steps) EXCEEDS that horizon is therefore invisible to the
    # critic target even though the real buffer data contains several full
    # cycles: the delayed overshoot a too-aggressive action causes is
    # down-weighted to ~1 % and never co-occurs with its cause in the
    # value target.  ``critic_anchor_lambda`` decouples the ANCHOR's λ from
    # the imagination λ so the anchor can use a near-Monte-Carlo
    # return-to-go (λ→1) over the real sequence — injecting the FULL
    # realised multi-cycle cost into the critic target, GROUNDED IN REAL
    # DATA (not a long WM rollout).  The actor still trains purely on H-step
    # imagination; only the critic's real-grounding horizon changes, so the
    # value bootstrap V(s_H) the actor reads becomes calibrated to the
    # long-horizon cost.  ``None`` (default) ⇒ fall back to ``gae_lambda``
    # (exact legacy behaviour).  Set ~0.97–1.0 to engage.  Does NOT touch
    # the cascade-sensitive imagination λ.
    critic_anchor_lambda: Optional[float] = None
    # Raise the anchor's pull when its long-horizon target must overcome the
    # myopic imagined critic loss that keeps dragging V back to a ~10-step
    # estimate.  ``None`` ⇒ use ``critic_replay_anchor_coef`` unchanged.
    critic_anchor_coef_long: Optional[float] = None
    # ---- #1 (P88, 2026-06-05): critic real-grounding rebalance ----
    # Weight on the IMAGINED critic CE.  The cascade through-line is that the
    # critic regresses almost entirely on its own imagined bootstrap
    # (critic_target_v_r->0.97, critic_rew_to_tgt_var->0.001 = reward <0.1% of
    # target variance) -> a self-referential pessimistic fixed point that
    # freezes the actor.  Down-weighting the imagined CE (<1.0) lets the
    # REAL-return replay anchor (``critic_replay_anchor_coef`` /
    # ``critic_anchor_coef_long`` with ``critic_anchor_lambda``->1.0 = near-MC
    # return-to-go over the real buffer) DOMINATE the critic target, so the
    # value is grounded in realised economics instead of model fiction.  Pairs
    # with #2 (latent overshooting): once the WM is accurate at long H the
    # critic trained on REAL states also values IMAGINED states correctly
    # (value-equivalence).  ``1.0`` = legacy (imagined-primary).  Env
    # ``DREAMER_CRITIC_IMAG_LOSS_COEF``.
    # Default 0.3 = p117 recipe: let the MC grounding term dominate the critic
    # target so it can't self-inflate (promoted 2026-06-14; was 1.0).
    critic_imag_loss_coef: float = 0.3

    # mbrl2 real-sim (p04, 2026-07-09): Monte-Carlo GROUNDING weight for the
    # real-sim actor-critic critic.  The critic target is the λ-return CE plus
    # ``critic_mc_grounding_coef`` × the PURE discounted reward-to-go (λ=1, no
    # bootstrap) CE.  p03 RCA: the bootstrap-only λ-return let the value head
    # drift/INVERT (val critic_r -0.23 → MV railed high); a full-real-episode MC
    # anchor pins V to realised economics so the advantage sign stays correct.
    # 0.0 = OFF (legacy).  Env ``DREAMER_CRITIC_MC_GROUNDING_COEF``.
    critic_mc_grounding_coef: float = 1.0

    # When True, cap the MC return with a single discounted tail bootstrap
    # ``γ^N·V_target(s_N)`` to remove the truncated-horizon bias; when False
    # (default) the return is PURE MC (truncated at the segment end).  At
    # ``seq_len`` = 128 / γ = 0.97 the truncation bias γ^128 ≈ 0.02 is
    # negligible, so pure MC is the cleaner, fully-grounded default.  Env
    # ``DREAMER_CRITIC_MC_TAIL_BOOTSTRAP``.
    critic_mc_tail_bootstrap: bool = False

    # ---- MV / limit consistency (P86, 2026-06-05) ----
    # (1) Action→MV mapping basis.  The normalised actor action is mapped to an
    # engineering-unit MV over a FIXED reference band.  Historically (pre-P60)
    # that band was the moving ``current_mv_bounds`` — which warped the mapping
    # on every operator-limit step (spurious MV jump).  P60 fixed the jump by
    # mapping against the STATIC base bounds instead, but that introduced two
    # defects: (a) the agent can never command an MV outside the base box, so
    # when an active limit steps ABOVE ``base_hi`` the agent physically cannot
    # track it; (b) action=0 always lands on ``base_lo``, baking a fixed
    # "rest on the base floor" point into the actuator that the agent then
    # rides even when the active floor has stepped up → MV violations.
    #
    # The fix: map the action over the FIXED PHYSICAL ENVELOPE (the simulator's
    # MV normalisation range), which never moves with the operator limits.
    # This keeps the P60 jump fix (fixed band ⇒ no warp), gives the agent full
    # reach over every attainable active limit, and removes the baked-in base
    # floor so the agent must LEARN the (moving) active limits from the bounds
    # observation channels + the violation penalty — there is now a single
    # consistent operating-limit set (the active limits) that the agent learns,
    # rather than a base-vs-active mapping/penalty split.  Env
    # ``DREAMER_MV_ACTION_FULL_RANGE=0`` restores the P60 base-bounds mapping.
    mv_action_map_full_range: bool = True
    # (2) Hard-clamp the engineering-unit MV to the CURRENT (active) operator
    # limits before it reaches the plant.  This is a DCS-style RUNTIME SAFETY
    # limiter only — it is OFF during training/validation so the agent must
    # genuinely LEARN to respect the limits (a clamp would mask whether it
    # learned).  Enable at deployment via ``DREAMER_MV_HARD_CLAMP=1``.
    mv_hard_clamp: bool = False
    # (3) Runtime operator-limit variation (the RuntimeSetpointManager that
    # steps the MV/CV limits mid-episode to teach operator-limit tracking).
    # Kept ON: the agent is SUPPOSED to learn to track changing operator
    # limits.  The ACTIVE limits are the single operating-limit set; BASE is
    # only their nominal starting value.  Set False (``DREAMER_RUNTIME_SETPOINT
    # _VARIATION=0``) only to freeze active≡base for a no-limit-step ablation.
    runtime_setpoint_variation: bool = True

    # ---- WM disturbance-estimator head (P87, 2026-06-05) ----
    # ML analogue of an APC disturbance observer / "prediction-error
    # feedforward".  A small auxiliary head reads the RSSM posterior feature
    # ``[h, z]`` and is supervised to predict the hidden, unmeasured OU
    # disturbance (ground truth from ``utils/hidden_disturbance.py``, which is
    # invisible to the agent + dynamics).  Two effects: (a) it forces the
    # latent to EXPLICITLY encode the unmeasured-load state, so the policy —
    # which already reads the same feature — becomes disturbance-aware and can
    # react/feed-forward; (b) the smooth slow-OU target regularises the latent
    # toward a stable held-action representation, improving WM steady-state
    # prediction.  Imagination-safe: the head reads only the latent (no
    # measured-vs-predicted innovation that would be undefined inside a dream).
    # Default ON; sized per-simulator (output dim = number of CV channels).
    # Disable with ``DREAMER_DISTURBANCE_HEAD=0``.
    disturbance_head: bool = True
    # Output width — resolved at runtime to ``len(env.cv_indices)``; 0 ⇒ no
    # head is built (byte-identical to the pre-P87 model).  Do not set by hand.
    disturbance_head_dim: int = 0
    # Head MLP width / depth.  ``disturbance_head_hidden=0`` ⇒ reuse
    # ``head_hidden``.
    disturbance_head_hidden: int = 0
    disturbance_head_layers: int = 2
    # Supervised-loss weight added to the world-model total.  Under stop-grad
    # (the default) this is the head-only (read-out) weight; it CANNOT affect
    # the WM trunk, so its magnitude is harmless.  ``DREAMER_DISTURBANCE_LOSS_SCALE``.
    disturbance_loss_scale: float = 1.0
    # Stop-gradient the latent feeding the disturbance head (2026-06-10).
    # When True (DEFAULT), the head is a pure READ-OUT probe: it trains on the
    # WM feature but its gradient does NOT flow back into the encoder/dynamics
    # trunk, so it can never degrade the WM gain/dynamics — yet the policy still
    # sees the disturbance in the shared latent (feed-forward rejection still
    # works; the validation disturbance-prediction diagnostic measures how well).
    # RCA (p109, 2026-06-10): the head's optimizer-coverage bug was fixed
    # (commit 5a31041) so the head finally trained — but at the untuned
    # ``disturbance_loss_scale=1.0`` its loss term DOMINATED wm_total (10-27x
    # the recon term) and, while FAILING to predict (loss rising), dragged the
    # latent toward encoding the disturbance: WM gain rel_err 0.186->0.365,
    # real->post 0.844->0.783, KL 0.30->0.72, and the actor never sharpened.
    # Defaulting to stop-grad restores the p106 WM-trunk gradient exactly while
    # keeping the head trainable.  Set False to OPT IN to latent-SHAPING (the
    # original P87 feed-forward intent) — and when you do, the shaping gradient
    # is bounded ADAPTIVELY by ``disturbance_loss_rel_weight`` below (NOT the raw
    # ``disturbance_loss_scale``), so it stays sim-agnostic and trunk-safe.
    # ``DREAMER_DISTURBANCE_HEAD_STOP_GRAD``.
    disturbance_head_stop_grad: bool = True
    # Adaptive SIM-AGNOSTIC shaping weight — used ONLY when stop_grad is False
    # (latent-shaping / active feed-forward).  Each step the disturbance term is
    # set to ``rel`` x the RECON term (via a detached loss-magnitude ratio), so
    # the feed-forward gradient is a fixed fraction of the gain-carrying recon
    # gradient REGARDLESS of the simulator's disturbance scale or obs variance
    # (the absolute magnitudes cancel).  This is the p109 RCA fix: it makes
    # shaping structurally unable to swamp recon (the failure mode was a 27x
    # ratio).  ``< 1.0`` keeps recon dominant; default 0.3 = a gentle nudge.
    # ``0`` falls back to the legacy absolute (NON-sim-agnostic) path.  The
    # absolute ``disturbance_loss_scale`` acts as a hard ceiling on the adaptive
    # coefficient.  ``DREAMER_DISTURBANCE_LOSS_REL_WEIGHT``.
    disturbance_loss_rel_weight: float = 0.3
    # Soft WM-fidelity gate (legacy absolute path only): the disturbance term is
    # scaled by ``min(1, gate_recon / recon_loss)`` so a not-yet-converged world
    # model is not destabilised by the auxiliary target early in P1.  ``<=0``
    # disables the gate.  Ignored under stop-grad and under the adaptive path.
    disturbance_loss_gate_recon: float = 1.0

    # ---- Neural Kalman filter / disturbance observer (DOB), 2026-06-11 ----
    # The unmeasured load is an OMITTED VARIABLE that attenuates the WM gain
    # (Exp A / p113: gain rel_err 0.36 with the disturbance ON vs 0.18 with it
    # OFF; the autoencoder real->post recovered 0.766->0.94 disturbance-off).
    # The DOB augments the RSSM/TSSM with an explicit additive output-disturbance
    # state ``d_t`` (per CV) that integrates the one-step prediction residual and
    # is added to the decoded CV at recon (``CV = g(feat) + d_t``), so ``g``
    # learns the CLEAN input->CV gain while ``d_t`` absorbs the load AND becomes
    # the unmeasured-disturbance estimate (feeds the disturbance diagnostic).
    # See docs/architecture.md §3.  ``dob_enabled=False`` = byte-identical to the
    # pre-DOB model.  Backbone-agnostic (RSSM + TSSM share feat->decode).  ENV:
    # DREAMER_DOB_ENABLED / _REG_COEF / _DECAY_INIT / _GAIN_INIT.
    # Default False (p136, 2026-06-21): the DOB was added to DE-CONFOUND the gain
    # (omitted-variable attenuation), but the REAL confound was domain
    # randomization — fixed in Stage A (DR-off ID → gain 0.84→0.97).  With the
    # gain de-confounded, the DOB's job is subsumed: the frozen-g GRU latent
    # already tracks the unmeasured disturbance (p109 read-out r=0.95) and the
    # always-trainable disturbance_head reads it out.  So we identify clean
    # (DR off, g frozen in Stage 2) WITHOUT the DOB and let the head be the
    # disturbance estimator.  The DOB code is retained + gated (verify-then-strip):
    # set True to restore the additive d_t observer.  ``DREAMER_DOB_ENABLED=1``.
    dob_enabled: bool = False
    dob_reg_coef: float = 0.01      # L2 "process-noise-is-small" prior on d_t
    dob_decay_init: float = 3.0     # sigmoid(3.0)=0.953 — slow persistence (A)
    # sigmoid(-2.0)=0.119 — modest Kalman correction (K).  History: -2.2 (p121)
    # under-predicted the disturbance amplitude (pred_std 1.16 vs true 1.93,
    # ratio 0.60); -1.8 (p122) OVER-shot (pred_std 2.27, ratio 1.18) which both
    # mis-calibrated the disturbance head (R² -0.26→-1.77) AND over-amplified the
    # validation-time d_t feeding the actor's feat.  -2.0 (p123) lands the
    # amplitude ratio ~0.9 — the sweet spot between the two — and reverts the
    # actor's d_t toward the p117 active-actor regime.  Pairs with the DV-PRBS
    # fix (a correct DV gain cleans the innovation feeding K).
    dob_gain_init: float = -2.0

    # ---- Staged clean->disturbance curriculum (2026-06-12) ----
    # The textbook system-ID / Kalman recipe applied to the DOB: identify the
    # plant on CLEAN data FIRST, THEN identify the observer on the FIXED plant,
    # THEN design the controller.  Fixes the gain<->disturbance identifiability
    # confound (p114/p115: when g and d_t co-train on confounded closed-loop
    # disturbance data, d_t "steals" gain from g).  Runs ON TOP of phased mode
    # (the 3 stages = phases P1/P2/P3, budgeted by phase{1,2,3}_frac):
    #   Stage 1 (P1): CLEAN (hidden disturbance OFF) + DOB d_t SUPPRESSED ->
    #     g learns the UNBIASED input->CV gain (no omitted-variable confound).
    #   Stage 2 (P2): FREEZE g (NOT the DOB) + disturbance ON + d_t ACTIVE ->
    #     the recon innovation trains the Kalman observer (A,K) on the fixed g
    #     (all CV movement g can't explain is attributed to d_t = identifiable).
    #     (reuses the P2 loss path; BC also warms the actor as a free bonus.)
    #   Stage 3 (P3): FREEZE g AND the DOB + disturbance + domain-randomization
    #     ON -> actor/critic train on the static unbiased WM + working observer
    #     and learn to REJECT disturbances (d_t feed-forward) for runtime
    #     robustness.  (reuses the P3 _wm_frozen_now path; reward head adapts.)
    # REQUIRES ``dob_enabled=True`` and ``train_mode != 'joint'`` (it IS the
    # phased curriculum).  Default OFF = the phased schedule is byte-identical.
    # ENV: DREAMER_CURRICULUM_ENABLED / _STAGE2_DISTURBANCE_PROB / _STAGE3_*.
    # Default True = p117 curriculum-recipe (promoted 2026-06-14; was False).
    # Self-validates at runtime: needs dob_enabled + phased + n_cv>0, else it
    # hard-disables with a warning, so defaulting ON is safe on any plant.
    curriculum_enabled: bool = True
    # Per-stage hidden-disturbance per-episode probability (overrides the
    # adaptive ``get_phase_disturbance_prob`` ramp).  Stage 1 is always 0.0
    # (clean by construction).  Stage 2 = max density so the observer sees lots
    # of disturbance residual to identify A,K.  Stage 3 < 1.0 so the actor also
    # sees some clean episodes (covers the no-disturbance operating point).
    curriculum_stage2_disturbance_prob: float = 1.0
    curriculum_stage3_disturbance_prob: float = 0.85
    # Domain-randomization (±frac output-gain/bias/actuator jitter) is an
    # ACTOR-robustness mechanism, but it was active during the Stage-1/2 WORLD-
    # MODEL identification too (initialised globally in SimNoiseWrapper, never
    # stage-gated) — so the WM gain was fit to a RANDOMISED gain and scored
    # against the NOMINAL plant (eval disables DR), forcing the categorical
    # latent to model a gain DISTRIBUTION ⇒ a systematically ATTENUATED
    # identified gain (the cross-run ~0.85 'ceiling').  When True the curriculum
    # turns DR OFF for the seed collection + P1 (clean WM id) + P2 (DOB id).  As
    # of Stage A (p135) DR STAYS OFF for P3 too — the actor's loop-gain
    # robustness now comes from IMAGINATION-time gain randomization
    # (actor_imag_gain_random_frac) rather than REAL-data DR, which created a
    # train/imagination mismatch (p134 actor regression: the real loop gain
    # varied ±frac but the actor imagined on the nominal frozen WM).  Textbook
    # system-ID: identify the plant clean, then design a robust controller
    # (robustness injected in imagination).  Sim-agnostic (toggles the existing
    # randomizer).  ``DREAMER_CURRICULUM_WM_ID_DR_OFF``.
    curriculum_wm_id_dr_off: bool = True


    # ---- World-model backbone (P68, 2026-05-30) ----
    # ``'rssm'`` (DreamerV3 recurrent state-space model) is the new
    # default: its deterministic GRU core can learn a held-action fixed
    # point ``h* = f(h*, z*, a)`` — the structural property the
    # SF-transformer lacked (``wm_pred_converges_under_constant_action``
    # pinned at 0.0 across P64/P66/P67, the upstream cause of the
    # bootstrap-cascade that every critic/reward-side fix failed to break).
    # ``'sf_transformer'`` selects the original V4 shortcut-forcing WM.
    world_model_type: str = 'rssm'
    rssm_deter_dim: int = 512          # GRU hidden (paper Medium)
    rssm_n_categoricals: int = 32      # paper
    rssm_n_classes: int = 32           # paper
    rssm_embed_dim: int = 256
    rssm_hidden_dim: int = 256
    rssm_unimix: float = 0.01          # paper 1% uniform mixture
    rssm_free_bits: float = 0.5        # p117 recipe (promoted 2026-06-14; paper=1.0)
    rssm_kl_dyn_w: float = 0.5         # paper KL-balance dyn weight
    rssm_kl_repr_w: float = 0.1        # paper KL-balance repr weight
    # ---- Continuous gain + disturbance latent (2026-06-22) ----
    # A Gaussian latent ALONGSIDE the categorical, giving precision-critical
    # CONTINUOUS quantities an un-quantized home the categorical attenuates:
    #   * cont_gain_dim = the in-context GAIN block (supervised by C(1) toward
    #     the identified steady-state gain → fixes the DV categorical-attenuation
    #     bias + carries the per-episode gain so the WM ADAPTS to DR), and
    #   * cont_dist_dim = the unmeasured-DISTURBANCE block (= n_cv; an amortized
    #     Kalman state inferred from the innovation + rolled by the prior — the
    #     INHERENT, DOB-free disturbance estimator).
    # ``cont_latent_enabled=False`` (or both dims 0) ⇒ pre-continuous-latent
    # model.  When enabled the dims are auto-resolved from the plant
    # (cont_dist_dim=n_cv, cont_gain_dim=n_cv·(n_mv+n_dv)).  Backbone-agnostic
    # (RSSM + TSSM); every field is a TrainConfig knob so BO inherits it.
    cont_latent_enabled: bool = False
    cont_gain_dim: int = 0
    cont_dist_dim: int = 0
    cont_min_std: float = 0.1
    cont_max_std: float = 2.0
    cont_free_bits: float = 0.5
    cont_kl_scale: float = 1.0
    # C(1) gain-matching: supervise the WM's finite-difference step-response
    # ASYMPTOTE toward the identified steady-state gain (the un-cheatable DC
    # supervisor that pins the subdominant DV gain).  Resolved >0 only when the
    # continuous gain channel is on AND the identified gains are available.
    gain_match_coef: float = 0.0
    gain_match_len: int = 0            # K step-response rollout steps (= horizon)
    gain_match_max_starts: int = 6
    gain_match_step: float = 1.0       # Δinput (normalized) for the FD probe
    # Resolved gain targets (WM/normalized units): per-input rows of n_cv gains.
    # ``*_kinds``/``*_idx`` map each row to an action col (mv) or dv-vector col.
    gain_match_mv_target: Tuple[Tuple[float, ...], ...] = ()
    gain_match_dv_target: Tuple[Tuple[float, ...], ...] = ()
    # Gain-channel persistence: a light L2 on the step-to-step change of the
    # gain block (the gain is a per-episode CONSTANT, so it should not wander).
    cont_gain_persist_coef: float = 0.0
    # C(2) disturbance-matching (p138 RCA): supervise the cont DISTURBANCE
    # channel's posterior mean toward the recorded true hidden load so it
    # actually ENCODES the unmeasured disturbance (the inherent amortized-Kalman
    # estimate) instead of staying a free OU the decoder uses to inject drift.
    # Symmetric with C(1) gain-match.  Resolved >0 only when the cont
    # disturbance channel is on (auto 0.3); the MSE is normalised by the load
    # variance so the coef is sim-agnostic.  ``DREAMER_DIST_MATCH_COEF``.
    dist_match_coef: float = 0.0
    # Roll the cont DISTURBANCE block DETERMINISTICALLY (prior MEAN) in
    # imagination (2026-06-29, p140 RCA).  The cont disturbance is a FEEDFORWARD
    # signal — the actor needs the PREDICTED load, not a per-rollout sampled
    # realization that buries the action signal in the imagined reward (p140:
    # imag_reward_dv_corr 0.44 → imag_adv_action_corr 0.095 → actor thrash +
    # return_scale cap cascade).  GAIN block stays sampled.  ``DREAMER_CONT_DIST_DET_ROLL``.
    cont_dist_deterministic_roll: bool = True
    # Static DV→obs feedthrough skip (p132).  DEFAULT OFF (2026-06-29, p140 RCA):
    # the memoryless ``W·dv_t`` is a physically-wrong instant feedthrough (DV→CV
    # has dead-time) AND a gain_match crutch (lets the dynamic DV path stay weak
    # → slow rise); the cont GAIN block + gain_match supersede it.  Ablation
    # lever only — ``DREAMER_DV_STATIC_SKIP=1`` restores the p132 skip.
    dv_static_skip: bool = False
    # DV-as-input (Option B, 2026-06-07): feed the measured disturbance-variable
    # channels as an EXOGENOUS transition input (teacher-forced from the real
    # obs in WM training; HELD CONSTANT over the imagination horizon = the MPC
    # feedforward persistence assumption) instead of letting the WM PREDICT
    # (hallucinate) them forward.  The agent reacts to a real DV change by
    # closed-loop feedback (observes it next step, re-plans).  Frees WM capacity
    # + removes DV-misprediction rollout error (also tightens held-action
    # steady-state, since DV no longer drifts in the held rollout).  DEFAULT ON;
    # opt out with DREAMER_DV_AS_INPUT=0.  ``dv_dim``/``dv_indices`` are resolved
    # at runtime from ``env.meta['dv_indices']`` (sims with 0 DVs -> dv_dim=0 =
    # paper behaviour).  Backbone-agnostic (RSSM + TSSM both thread ``dv``).
    dv_as_input: bool = True
    dv_dim: int = 0
    dv_indices: Tuple[int, ...] = ()
    # DV→decoder+heads FEEDFORWARD (2026-06-19, p129 RCA).  Route the measured
    # DV directly into the WM decoder AND the reward/value/policy heads (in
    # addition to the transition), so the CV reconstruction ``g(h, z, dv)`` has
    # a DIRECT ∂CV/∂dv path that SKIPS the categorical bottleneck where the
    # DV→CV gain was dying (p129 DV posterior-prior decomp: real→post ×0.77,
    # post→1step ×1.00 — the loss is the autoencoder, not data/excitation), and
    # the actor finally SEES the disturbance in imagination (fixes the passive
    # actor).  Sim-adaptive: no-op when the plant has no measured DV (dv_dim=0).
    # Backbone-agnostic (RSSM + TSSM).  Opt out with DREAMER_DV_FEEDFORWARD=0.
    dv_feedforward: bool = True
    # De-contaminate the disturbance head from the MEASURED dv (2026-06-19, p130
    # RCA).  ``dv_feedforward`` routes the measured DV into ``feat`` (decoder +
    # heads).  The disturbance head predicts the UNMEASURED load — but it reads
    # the SAME ``feat``, so the measured dv leaks in and the head conflates the
    # measured-DV-driven CV move with the unmeasured disturbance (p130: head r
    # 0.78→0.54, R² −1.6→−2.4, std flipped under→over-predict).  When True, the
    # measured-dv columns of ``feat`` are ZEROED before the disturbance head
    # only (the decoder + actor/critic heads still get the dv feed-forward).
    # The head can still infer the load indirectly via the (h, z) latent — this
    # removes only the DIRECT measured-dv shortcut.  No-op when dv_feedforward is
    # off or the plant has no DV.  ``DREAMER_DISTURBANCE_HEAD_EXCLUDE_DV``.
    disturbance_head_exclude_dv: bool = True
    # Disturbance-prediction DETREND window for the CONTROL-RELEVANT Kalman score
    # (2026-06-20).  The DOB d_t feeds forward; a slow drift in the estimate
    # (timescale ≫ closed-loop settling) is rejected by the feedback integral
    # action, so only the DYNAMIC tracking error matters for feed-forward.  The
    # validation disturbance metric is ALSO reported high-pass-detrended with a
    # window = ``this × settling`` (settling = the auto-tuned ``horizon``), so it
    # is SIM-ADAPTIVE.  4× keeps the settling-band dynamics, removes slower drift.
    # ``DREAMER_DISTURBANCE_DETREND_SETTLE_MULT``.
    disturbance_detrend_settle_mult: float = 4.0
    # ===== TSSM (transformer-SSM) backbone dims (neural-apc-mbrl) =====
    # Used only when world_model_type='tssm'.  Reuses the rssm_* categorical-
    # latent dims (n_categoricals/n_classes/embed_dim/unimix/free_bits/kl_*).
    tssm_d_model: int = 512
    tssm_n_layers: int = 4
    tssm_n_heads: int = 8
    tssm_max_seq_len: int = 256
    # P70 (2026-05-30) imagination steady-state fixes.  The trained RSSM
    # prior map CONTRACTS to a fixed point under a held action (offline
    # probe: deterministic-mode tail_std ~0.01), but per-step categorical
    # RE-sampling in imagination re-injects ~0.84 nats/group of latent
    # noise every step → the reward head (≈quadratic CV penalty) sees a
    # curvature·Var positive penalty bias on EVERY imagined step → too-
    # negative imagined returns → critic pessimism → return_scale runaway
    # → bootstrap cascade (flat MV).  Opt-in mitigation:
    #   ``rssm_imag_latent_mode``: roll the imagined PRIOR with the
    #       categorical MODE (argmax, sample=False) instead of a sample —
    #       removes the per-step jitter so the reward head sees the
    #       settled mean latent.  Actor exploration still comes from the
    #       policy's own action sampling (TD-MPC2-style deterministic
    #       latent + stochastic action).  Default True = p117 recipe: removes the
    #       per-step imagined-latent jitter that biases imagined returns negative
    #       and feeds the cascade (promoted 2026-06-14; paper-faithful = False).
    rssm_imag_latent_mode: bool = True
    # Buffer seeding (P0 cold-start fix, 2026-05-05; expanded 2026-05-06).
    # Replace the two random-action seed episodes with ``baseline_seed_episodes``
    # of small-noise actions around mid-MV.  Stays in-bounds on cliff-shaped
    # reward landscapes so the buffer carries some positive-advantage
    # transitions before the actor has trained.  Defaults bumped from
    # 8/2 → 16/6 (Option 3 in 2026-05-06 RCA): more state-space coverage
    # for the WM pretrain phase, addressing the under-trained dynamics
    # diagnosed in run_p0adapt.
    baseline_seed_episodes: int = 16
    baseline_seed_action_std: float = 0.05
    random_seed_episodes: int = 6
    # Structured PRBS-style exploration episodes for WM coverage breadth
    # (2026-05-08, run_p7 RCA).  Drives MV across the operating band
    # ``[-prbs_seed_op_band, +prbs_seed_op_band]`` in segments ~2τ long
    # so the WM sees clean step responses across the full MV range.
    # Auto-derived in ``auto_tune_seed_buffer`` to ~⅓ × baseline_seed
    # episodes.  Set to 0 to disable.
    exploration_seed_episodes: int = 6
    # PRBS target operating band as fraction of normalised action range.
    # 0.95 covers MV ∈ [−0.95, +0.95] (i.e. 97.5 % of each side of the
    # normalised range), so the WM sees plant behaviour all the way up
    # to ~5 % from the hard bounds.  Critical for boundary accuracy:
    # the controller will most often need to operate near MV bounds
    # exactly when CV violations are imminent, and a WM trained only
    # on the safe middle 70 % extrapolates badly there.
    prbs_seed_op_band: float = 0.95
    # Stratified-sampling guarantee: divide ``[-op_band, +op_band]``
    # into N strata and ensure at least one PRBS center per stratum
    # per episode (per action dim).  Prevents the boundary bins from
    # being statistically empty on plants where #segments_per_episode
    # is small (e.g. fast plants where T/(θ+4τ)/sr can be < 10).
    # Set to 0 to disable stratification (pure uniform sampling).
    prbs_seed_n_strata: int = 8
    # PRBS segment length (agent steps).  Auto-derived in
    # ``auto_tune_seed_buffer`` to ``(θ + 4τ) / sample_rate``, which
    # spans ~98 % of a first-order step response so the WM sees both
    # the transient and a clear settled steady-state at each operating
    # point — necessary to learn the full transfer function (gain +
    # time constant) rather than just the early transient slope.
    # Floor 8, cap T/4 so each episode still contains ≥ 4 segments.
    # Sentinel 0 means "use auto-derived" — any positive value
    # overrides.
    prbs_seed_segment_steps: int = 0
    # PRBS segment-length MIN (agent steps).  When > 1 and
    # < prbs_seed_segment_steps, each PRBS segment's hold time is
    # sampled log-uniformly in [seg_min, seg_max] so the WM sees a
    # MIX of fast (~τ/3) and slow (~4τ) excitation in the same
    # episode.  Single fixed long-hold PRBS gives mostly steady-state
    # data — WM learns gain but compounds error on transients
    # (run_p12 P1 plateau: r(H=1)=0.65 but r(H=7)=0.19, never
    # improved).  Multi-timescale GBN-style PRBS is the standard
    # system-ID remedy.  Auto-derived in ``auto_tune_seed_buffer``
    # to ``max(2, round(τ / (3 sr)))``.  Sentinel 0 = use auto.
    prbs_seed_segment_steps_min: int = 0
    # Constant-action seed episodes (2026-05-21, p31 RCA).  Holds the
    # action perfectly constant for the entire episode at a sampled
    # level in ``[-constant_action_seed_op_band, +constant_action_seed_op_band]``.
    # Why: the PRBS seed only ever holds an action for ``(θ + 4τ)/sr``
    # ≈ 80–150 agent steps, so the WM never sees a long-horizon
    # constant-action steady state during pretrain.  When the controller
    # later sits near a setpoint and holds the MV, the WM extrapolates
    # outside its training distribution and drifts (run_p31 noise-free
    # diagnostic: real plant converges 75–88%, WM 0% under constant
    # action).  10 full-length constant-action episodes give the WM
    # explicit steady-state coverage at a stratified spread of operating
    # points.  Set to 0 to disable.
    constant_action_seed_episodes: int = 40
    # Operating-band fraction for the constant-action seed.  Narrower
    # than ``prbs_seed_op_band`` (0.95) on purpose: at the very edges of
    # the MV range the plant typically saturates and the long hold is
    # uninformative.  0.6 covers the central 60% × 2 = 120% (yes, both
    # signs) of the realistic operating envelope.
    constant_action_seed_op_band: float = 0.6
    # Step-and-settle seed fraction (2026-05-23, P42 design).  Fraction
    # of ``constant_action_seed_episodes`` that should be allocated to
    # **step-and-settle** episodes instead of pure constant-action:
    # hold u₀ for a short prefix → step to u₁ → hold u₁ to episode end.
    # Why: a pure constant-action episode is *information-poor* — if
    # the env init lands near SS(u₀) the WM sees ~1220 nearly-identical
    # samples and gets near-zero gradient.  A step-and-settle episode
    # is a strict superset: contains the late steady-state (same as
    # const-action) plus a clean step response with known timing →
    # direct gain + time-constant supervision.  Also breaks the
    # degenerate "given constant input → predict no-change" shortcut
    # that const-action-only seeds may enable (run_p40 RCA: WM
    # steady-state probe 0% conv at H=200 despite L=32 + 40 const
    # seeds).  Step-and-settle forces the WM to predict CHANGE under
    # constant input during settling, which is the actual physics.
    # 0.5 = half const, half step-settle.  0.0 = pure const (legacy).
    step_settle_seed_fraction: float = 0.5
    # Range of the step in normalized action units, |u₁ − u₀|.  Min
    # ensures the step dominates plant noise and gives a clear gain
    # readout; max keeps the step in a regime where the plant doesn't
    # saturate at one end.  ``u₀`` is stratified over op_band as before;
    # ``u₁ = clip(u₀ + Δ, -1, 1)`` with Δ uniformly sampled from
    # ``±[step_seed_delta_min, step_seed_delta_max]``.
    step_seed_delta_min: float = 0.20
    step_seed_delta_max: float = 0.60
    # Fraction of the episode used for the pre-step hold (settling
    # from random IC to SS(u₀)).  Default 0.10 = 122 steps at
    # ep_len=1220 ≈ 7 settling times at τ=53/sr=4 (τ_steps≈13) — well
    # past the SS for u₀.  Leaves 1098 steps post-step for the new
    # settling, ~84 settling times → ample long-horizon SS coverage.
    step_seed_prefix_frac_min: float = 0.05
    step_seed_prefix_frac_max: float = 0.20
    # ---- APC step-test seed (P51 design, 2026-05-25) ----
    # Mixed MV + DV step events in a single held-baseline episode,
    # mirroring industrial APC step-testing.  Each episode:
    #   1. holds an MV operating point for an initial settle phase
    #   2. fires a sequence of MV-step and DV-step events with mixed
    #      spacing — most ≥ 4·(τ+θ) apart so each gain is identifiable
    #      in isolation; a configurable fraction overlap (~0.5–1·(τ+θ))
    #      so the WM also sees coupled multi-channel transients
    #   3. preserves the held-baseline regime (hidden OU OFF) so the
    #      pre- and inter-event SS targets remain unambiguous.
    # Strict superset of ``step_settle_seed_episodes`` (which had 0 DV
    # activity and only 1 MV step) and of disturbance-active PRBS
    # (which never holds an MV).  Fixes the P49/P50 DV-coverage gap:
    # the WM never saw a clean ∂CV/∂DV with action held, and the
    # actor never saw "DV stepped while I held" in-distribution —
    # exactly the disturbance-rejection validation scenario.
    # Sim-adaptive via ``dyn_horizon`` (from identifier) for spacing
    # and via ``dv_gain_to_cv`` (from identifier) for DV magnitudes.
    step_test_seed_episodes: int = 20
    # Minimum step-test episodes PER input channel (n_mv + n_dv).  The
    # seed loop emits ``max(step_test_seed_episodes,
    # step_test_episodes_per_channel * (n_mv + n_dv))`` episodes so the
    # WM gets ≥ k clean isolated step responses for every input axis,
    # even on plants with many MVs/DVs.  Sim-agnostic (unitless count
    # per channel); sim-adaptive scaling is applied at runtime once
    # ``env.sim`` is built.
    step_test_episodes_per_channel: int = 4
    # Fraction of inter-event gaps that are in the OVERLAP regime
    # (~0.5–1·dyn_horizon) vs the SETTLED regime (~4–5·dyn_horizon).
    # Default 0.25 = 75% settled + 25% overlap.
    step_test_overlap_frac: float = 0.25
    # Fraction of step-test events that are DV events (rest are MV).
    # 0.5 = balanced MV/DV coverage.
    step_test_dv_share: float = 0.5
    # Fraction of DV events in a single episode that target the
    # episode's *primary* DV channel (round-robined across episodes so
    # every DV gets balanced isolated-step coverage).  The remainder
    # are picked uniformly at random across all DV channels — covers
    # cross-DV cases.  0.7 = strong stratification while still seeing
    # the off-primary DVs.
    step_test_primary_dv_bias: float = 0.7
    # ---- DV-PRBS seed episodes (2026-06-14, p121 DV-gain RCA) ----
    # The DV analogue of the MV's ``collect_prbs_episode``: hold the MV and
    # sweep every measured-DV channel with a full-range, multi-timescale,
    # stratified PRBS (hidden disturbance OFF).  Fixes the ~30× MV-vs-DV
    # excitation asymmetry that left the DV→CV gain attenuated (~0.75) while
    # the MV gain reached 0.93.  ``collect_dv_prbs_episode`` is a no-op
    # fallback (MV-hold) on plants with 0 DV channels, so defaulting >0 is
    # safe.  Re-injected through Stage 1 via DREAMER_DV_PRBS_INJECT_EVERY/_N.
    # Raised 16→24 on 2026-06-14 (p122 RCA): the P1→P2 wm_best warm-restore
    # keeps an EARLY (~iter 30) correlation-peak checkpoint and the probe is
    # gain-BLIND, so it discards the later DV-PRBS re-injections.  More SEED
    # DV-PRBS (trained from iter 1) lands the DV excitation inside the kept
    # early checkpoint instead of relying on re-injects that get rolled back.
    dv_prbs_seed_episodes: int = 24
    # Fraction of HALF the DV engineering span used as the stratified-level
    # sweep amplitude (|offset| ≤ op_frac·span/2).  0.6 = sweep ~60% of the
    # range each side of baseline: large enough that Var(DV) ≫ Var(noise)
    # (kills regression dilution) yet safely inside the channel bounds.
    dv_prbs_op_frac: float = 0.6
    # P2 BC bootstrap weight.  Default 0 because we have no offline expert
    # data — random-action episodes from P1 collection are uniform, so a
    # non-zero bc_scale clones uniform → uniform prior_policy → PMPO KL
    # term in P3 pins the policy near uniform → policy collapse.  Set
    # this >0 only when expert demonstrations populate the buffer.
    bc_scale: float = 0.0            # Phase-2 policy BC weight

    # ----- APC expert (BC anchor for the policy mean) -----
    # P81 (2026-06-03): the deterministic policy mean converges to a
    # worse-than-do-nothing basin because imagination exploitation under
    # bootstrap dominance (critic_rew_to_tgt_var→~0.001) lands μ in the wrong
    # place and the reward's deceptive geometry gives nothing to climb back
    # out with (gamma/σ/entropy all ruled out as levers, P79/P80).  An
    # objective-ALIGNED steady-state expert (utils/apc_expert.py) demonstrates
    # constraint-riding economic moves; BC toward it (Cursor stabilizer #6)
    # grounds μ WITHOUT changing the optimum (RL still owns the true optimum
    # via the real reward + WM).
    #   expert_type ∈ {'none', 'static', 'nn'}:
    #     'static' = robust gain-scheduled mover (GainScheduleExpert),
    #     'nn'     = MLP steady-state surrogate + projected-grad reward optimiser,
    #     'none'   = disabled (legacy behaviour, bc_scale stays 0).
    # When the expert is usable and expert_type != 'none', ``bc_scale`` is
    # auto-set to ``expert_bc_scale`` (cloning is MASKED to expert steps only).
    expert_type: str = 'static'
    expert_bc_scale: float = 0.15     # bc_scale applied when expert populates buffer
    expert_seed_episodes: int = 24    # expert-driven seed episodes added in P1
    expert_action_jitter: float = 0.03  # Gaussian σ around expert action (norm units)
    expert_keep_schedule: bool = True   # demonstrate under curriculum disturbances
    expert_use_ss_samples: bool = True  # fit/train from a fresh steady-state sweep

    # ----- P83: decaying P3 expert-BC anchor (anti-reversion) -----
    # P79/P80/P81b all show the SAME late-P3 reversion: the deterministic
    # policy mean drifts BACK toward the worse-than-do-nothing basin once
    # imagination exploitation under bootstrap dominance takes over
    # (critic_rew_to_tgt_var→~4e-4, return_scale 3→101).  The P2 BC anchor
    # is dropped at the P2→P3 boundary, so nothing holds μ near the expert
    # during P3 — the anchor evaporates exactly when it's needed most.
    # P83 keeps a MASKED expert-BC term alive THROUGH P3, applied to the
    # actor params only (WM isolation preserved), with a Kickstarting-style
    # decay 1→floor over the phase so RL still owns the true optimum late.
    # DEFAULT ON: it makes no sense to leave the anchor off when the run
    # provably reverts without it.  Disable for ablation via
    # DREAMER_EXPERT_BC_P3=0.
    expert_bc_p3: bool = True
    # Decay floor: the P3 BC weight decays expert_bc_scale*(1→floor) across
    # P3 (phase-length adaptive — stretches with the run).  A non-zero floor
    # is a PERMANENT anti-reversion anchor.
    # SET 0.05→0.0 (Fix B, 2026-06-22): the 0.05 floor is a permanent leash on
    # the policy — the likely reason the actor never beats the static expert
    # (p135 8%).  Decay the anchor fully to 0 so it warm-starts the policy in the
    # expert basin then RELEASES it to find the economic optimum.  Re-add a small
    # floor only if the actor reverts/diverges without the anchor late in P3.
    expert_bc_p3_floor: float = 0.0
    # TD3+BC return-scale normalisation (Fujimoto 2021).  OPT-IN (default
    # off): REINFORCE already divides the advantage by return_scale, so the
    # PG gradient on μ is already O(1) regardless of scale and a fixed-weight
    # MSE-on-μ BC is already proportionate.  Enable only if the logged
    # bc_p3/pg grad ratio shows the anchor being drowned as return_scale
    # grows (then bc_weight_eff = w * advantage_clip / max(return_scale,1)).
    expert_bc_p3_adaptive_scale: bool = False
    # ---- (BC learning-vs-crutch tracking, 2026-06-09) ----
    # Diagnostic: is the actor LEARNING beyond the expert, or just being held
    # at the expert by the permanent BC floor (expert_bc_p3_floor)?  When
    # ``> 0`` and an expert is active, every ``bc_track_expert_every`` det-eval
    # cadences roll ONE deterministic expert episode under the SAME eval
    # protocol and log ``expert_det_return`` + ``agent_minus_expert_return``
    # to train_log.jsonl.  A persistently ~0 gap = crutch (actor cloning the
    # expert); a positive, growing gap = the actor is genuinely surpassing the
    # expert via the real-reward/MC-critic objective.  ``0`` = OFF (no extra
    # eval cost = p106-baseline behaviour).  To remove the floor confound for a
    # clean learning test, ALSO set DREAMER_EXPERT_BC_P3_FLOOR=0 (full release).
    # Default 1 = p117 curriculum-recipe (promoted 2026-06-14; was 0).
    bc_track_expert_every: int = 1

    # ----- (a) adaptive bounded-return envelope (WM-drift mitigation) -----
    # With per-step reward bounded to [-B, B] (P77 linear remap), the λ-return
    # over the effective GAE credit horizon 1/(1-γλ) cannot legitimately
    # exceed ~B/(1-γλ).  But the BOOTSTRAP critic value V is unbounded, so a
    # biased H-step bootstrap (from a drifting world model) can still drive a
    # self-consistent return_scale runaway (3→101 observed; critic_rew_to_
    # tgt_var collapse).  Clamping the bootstrap target-values AND the λ-return
    # targets (imagined + the replay anchor) to ±k·B/(1-γλ) makes "bounded
    # reward ⇒ bounded return" actually bind, decoupling the cascade from WM
    # steady-state fidelity (the Cursor #1 mechanism, made adaptive).
    # Sim-agnostic: B, γ, λ are all config; backbone-independent (shared tail).
    # DEFAULT ON; disable via DREAMER_RETURN_VALUE_ADAPTIVE_CAP=0.  k=2 spares
    # a healthy return spread (~45) and clips the runaway (~102).
    return_value_adaptive_cap: bool = True
    return_value_cap_k: float = 2.0
    # (2026-06-07) STRUCTURAL CASCADE-ROOT FIX.  The cap clamps the bootstrap
    # target-VALUES + λ-return TARGETS, which live on the γ VALUE-horizon
    # (|V| ≤ B/(1-γ)).  The legacy denominator (1-γλ) is the λ CREDIT-horizon
    # (~9 steps at γλ=0.89), making the cap ~10× too tight: at γ=0.99,B=3,k=2
    # the cap is 55 but a -1/step policy's true value is -100 -> the cap
    # FLATTENS the whole legitimate value range [-50..-300] into a wall at -55
    # (the exact ``img_ret`` pin seen in p94-p99) -> zero advantage spread ->
    # the critic/actor cascade.  With this ON the cap uses 1/(1-γ) (the value
    # horizon): at γ=0.99 -> cap=600 = 2× the theoretical max |V|, so it ONLY
    # catches a true runaway and never flattens a legitimate value.  PROMOTED
    # to default True (2026-06-09): validated across p100-p105 (the legacy
    # λ-horizon cap was the structural cascade root; the γ-horizon cap removed
    # the img_ret pin without ever flattening a legitimate value, and the
    # observed cascades in that series came from B=6 / γ choices, NOT the cap).
    # DREAMER_RETURN_VALUE_CAP_GAMMA_HORIZON=0 restores the legacy λ-horizon cap.
    return_value_cap_gamma_horizon: bool = True

    # ----- (b) world-model held-action steady-state consistency loss -----
    # The RSSM has no explicit held-action fixed point (wm_pred_converges_
    # under_constant_action≈0): under a constant action at a settled state the
    # one-step prior keeps drifting.  This term penalises that drift at
    # SETTLED + HELD steps detected adaptively from the batch (P64 failed with
    # ABSOLUTE thresholds defeated by OU+meas noise → here the settled mask is
    # a RELATIVE fraction of each channel's batch std).  Added to wm_total in
    # BOTH backbones (RSSM: img_step fixed point; transformer: explicit held-
    # action anchor on top of its native sf_bootstrap self-consistency).
    # DEFAULT ON (modest coef); disable via DREAMER_WM_STEADY_CONSISTENCY_COEF=0.
    wm_steady_consistency_coef: float = 0.5
    wm_steady_settle_frac: float = 0.15   # settled if |Δobs| < frac·channel_std
    wm_steady_held_eps: float = 0.02      # action held if |Δact| ≤ eps (norm)

    # ---- (P89, 2026-06-06) noise curriculum + clean steady-state seeds ----
    # P89 RCA: the WM never learned a held-action fixed point because its
    # training data never contained a clean settled trajectory — process OU
    # (~1.3 % span, ~133-step correlation) + measurement noise are on in 100 %
    # of episodes (incl. the const-action steady-state seeds).  Two fixes:
    #  (A) ``clean_steady_seeds`` — fully disable process OU + measurement
    #      noise for the held-action steady-state seed episodes so the WM gets
    #      pure fixed-point supervision (DREAMER_CLEAN_STEADY_SEEDS=0 to off).
    #  (B) ``process_noise_curriculum`` — ramp process+measurement noise from
    #      ~0 to full over P1 (DREAMER_PROCESS_NOISE_AMP_RAMP="start:reach",
    #      default 0.0:0.4) so the WM learns base dynamics + the attractor
    #      first.  P3 always full noise (robust rejection).  Set False / ramp
    #      "1.0:1e-6" for legacy full-noise-from-step-0.
    clean_steady_seeds: bool = True
    process_noise_curriculum: bool = True

    # ---- (P90, 2026-06-06) freeze WM after P1 (critic/WM coherence) ----
    # The WM's held-action fixed point is UNSTABLE — it converges mid-P1 then
    # RE-DRIFTS during continued P2 training (P90 probe: conv 1.0->0.0).  With
    # this ON, the WM CORE (dynamics + encoder/decoder; tokenizer for SF) is
    # frozen (requires_grad=False) at the P1->P2 boundary AFTER the wm_best
    # warm-restore, and stays frozen through P2 and P3.  The REWARD head keeps
    # training (it is in parameters_world but must warm up in P2), the policy
    # (P2 BC) and value (P3) train normally.  Result: the critic warms up on
    # EXACTLY the WM that P3 + the post-training diagnostics use, and the
    # fixed point cannot re-drift.  Default OFF (paper co-trains the WM
    # throughout); opt-in via DREAMER_WM_FREEZE_AFTER_P1=1.
    wm_freeze_after_p1: bool = False
    # ---- (WM freeze-after-PRETRAIN, 2026-06-09) joint-mode WM freeze ----
    # ``wm_freeze_after_p1`` only fires at the phased P1→P2 boundary, which
    # NEVER happens in JOINT mode (``_phase_for`` always returns 3).  This
    # knob freezes the WM CORE (dynamics + tokenizer for SF) ONCE the WM has
    # been pretrained for ``wm_freeze_after_iters`` joint iters — the held-
    # action fixed point + gain converge mid-pretrain then re-drift under
    # continued co-training (P90), and a controller trained on a moving WM
    # never sees the WM its diagnostics use.  After the freeze the reward
    # head, actor (BC) and critic keep training on a STATIC, converged WM.
    # ``0`` = OFF (co-train WM the whole run = paper / p106-baseline).  Pick a
    # value past the WM-gain plateau (watch wm_gain_rel_err in validation).
    # Backbone-agnostic (handles the SF tokenizer); transfers to TSSM.
    wm_freeze_after_iters: int = 0

    # ---- #2 (P88, 2026-06-05): multi-step latent overshooting ----
    # Dreamer-v3 trains the prior ONLY one step ahead (the KL term), so the
    # open-loop imagination rollout the actor depends on compounds error every
    # step (per-offset fidelity r 0.74@H13 -> 0.52@H55) — extending H then just
    # leans on the WM's weakest capability (the P86/P87 H=55 regression).  This
    # is the PlaNet/Dreamer-v1 latent-overshooting objective (dropped in v2/v3
    # because 1-step sufficed for Atari): roll the PRIOR forward ``len`` steps
    # under REAL actions with NO obs and penalise decode-vs-real-obs, directly
    # training accurate MULTI-step prediction so a long H becomes legitimately
    # usable (and steady-state behaviour is learnable in imagination).  RSSM
    # only (the SF backbone's shortcut-forcing is its native multi-step term).
    # ``coef=0`` = OFF (legacy / paper-faithful).  Cost ~ O(B·max_starts·len)
    # GRU steps.  Env DREAMER_WM_OVERSHOOT_{COEF,LEN,MAX_STARTS}.
    # Default 0.3 = p117 recipe: the open-loop compounding lever (promoted
    # 2026-06-14; was 0.0).  ``wm_overshoot_len`` is set = horizon in single_run.
    # (p143 TESTED coef 0.5 + tail_power 1.0 to fix the categorical MV@H slow-
    # rise; the calibrated transfer matrix on wm_best showed it made MV@H WORSE
    # 0.86->0.68 (de-emphasising the settled tail where the MV gain compounds)
    # while only marginally helping DV -> REVERTED.  The proven MV@H fix is the
    # cont-gain latent, p140 MV@H 0.94, not the categorical overshoot.)
    wm_overshoot_coef: float = 0.3
    wm_overshoot_len: int = 15            # K open-loop prior steps to supervise
    wm_overshoot_max_starts: int = 24     # cap start positions (stride) for cost
    # Steady-state TAIL emphasis (2026-06-20, p131 RCA): per-step weight
    # ``(k/K)^tail_power`` over the K-step rollout (then Σw-normalised).  The
    # open-loop gain dies in COMPOUNDING — a DC/steady-state phenomenon — but a
    # uniform ``/K`` mean dilutes the settled tail to ~1/K weight.  A power>0
    # concentrates the gain gradient on the DC-gain region (the settled tail)
    # WITHOUT inflating the term (bounded weighted mean) so it can't destabilise
    # the WM; the noisy early transient (already covered by 1-step recon/KL) is
    # de-emphasised.  ``2.0`` ≈ last step gets ~3× its uniform weight.  ``0.0``
    # recovers the exact uniform mean.  Env DREAMER_WM_OVERSHOOT_TAIL_POWER.
    # (p143 tested 1.0 to weight the mid-rise; it HURT MV@H 0.86->0.68 -> the
    # p131 tail emphasis was right, reverted to 2.0.)
    wm_overshoot_tail_power: float = 2.0
    # Soft recon-fidelity gate (mirrors the disturbance head): the overshoot
    # term is scaled by ``min(1, gate_recon / recon_loss)`` so it RAMPS IN only
    # as 1-step reconstruction converges — early in P1 the WM cannot predict
    # multi-step at all, and an ungated overshoot loss would dominate and stall
    # encoder/decoder convergence.  ``<=0`` disables the gate (always full).
    wm_overshoot_gate_recon: float = 0.1

    # ---- (b2, P89, 2026-06-06): multi-step HELD-ACTION rollout stationarity --
    # The existing ``wm_steady_consistency`` term only fires on NATURALLY held+
    # settled replay segments (p88c: held_frac mean 0.84%, max 6.3%) and is a
    # single-step ``sample=False`` fixed point — far too starved/shallow to stop
    # the MULTI-step compounding drift the steady-state diagnostic shows (0% WM
    # convergence under a held action at H_train).  This term ACTIVELY CREATES
    # the held condition: from strided starts it rolls the PRIOR forward ``len``
    # steps under the action HELD CONSTANT (``sample=True`` straight-through so
    # the stochastic prior is trained), then penalises the NET DRIFT of the
    # deterministic state ``h`` between an early post-transient window and the
    # final window — "once you stop changing the action, you must stop moving".
    # GAIN-NEUTRAL by construction (penalises tail DISPLACEMENT, not magnitude;
    # the transient ``[0, settle)`` is unconstrained so overshoot/recon still set
    # the gain) and RSSM-only (the SF backbone uses its native sf_bootstrap plus
    # the ``_sf_steady_consistency`` anchor).  ``coef=0`` = OFF.  Env
    # DREAMER_WM_HELD_ROLLOUT_{COEF,LEN,SETTLE_FRAC,WIN,MAX_STARTS,GATE_RECON}.
    # Default coef 0.5 / max_starts 8 = p117 recipe: the held-action steady-state
    # lever that kills multi-step compounding drift (promoted 2026-06-14; were
    # 0.0 / 12).  ``wm_held_rollout_len`` is set = horizon in single_run.
    wm_held_rollout_coef: float = 0.5
    wm_held_rollout_len: int = 64          # K prior steps under the HELD action
    wm_held_rollout_settle_frac: float = 0.5   # early window starts at frac·K
    wm_held_rollout_win: int = 8           # window length for drift averaging
    wm_held_rollout_max_starts: int = 8    # cap start positions (stride) for cost
    wm_held_rollout_gate_recon: float = 0.1    # soft recon-fidelity ramp gate

    # ----- MTP -----
    # mtp_length: bumped 8 → 32 on 2026-05-21 (p31 RCA).  The MTP head
    # provides the only training pressure that asks the WM to predict
    # state ``L`` steps ahead with high fidelity.  At L=8 the WM has no
    # gradient signal beyond ~8 steps and learns a representation that
    # is locally smooth but globally drifts — confirmed by the
    # noise-free steady-state diagnostic on p31, where WM imagined
    # trajectories converge in 0% of cases at horizon=200 even though
    # the real plant converges in 75–88%.  L=32 spans roughly 4× the
    # planning horizon (15) and matches the seq_len=64 segment so the
    # WM is supervised over the full BPTT window.  Paper precedent:
    # DreamerV3 uses L≥8 only on Atari; control-task ablations in the
    # appendix favour larger L when episodes carry long settled
    # transients.
    # 2026-05-23 (P41 RCA): reverted 32 → 8 (paper default).  The p31
    # rationale for L=32 ("WM doesn't converge under const action at
    # H=200") was falsified by P40 (0% steady-state convergence with L=32
    # + 40 const-action seeds) and P41 (cascade reappears identically at
    # L=8).  The cascade root cause is bootstrap-dominated AC-loop
    # targets, NOT WM-pretrain quality, so L has no leverage.  Save 4×
    # MTP compute.  Override via DREAMER_MTP_LENGTH.
    mtp_length: int = 8               # paper default (P41 RCA)

    # ----- PMPO (Phase 3) -----
    # alpha=0.7 (paper default).  alpha=0.5 caused near-perfect
    # cancellation between positive and negative-advantage gradient
    # branches whenever the actor sampled the same action repeatedly
    # within a trajectory (which is the failure mode we observed).
    pmpo_alpha: float = 0.7
    # PMPO KL-to-prior weight.  Lowered from the paper's 0.1 because the
    # prior_policy snapshot at start of P3 is taken from a near-uniform
    # policy (no expert BC), so a stronger KL pull would freeze the
    # policy at uniform.  0.01 lets the advantage signal dominate.
    pmpo_beta: float = 0.01

    # ----- Returns -----
    # 2026-05-24 (P48 RCA): structural γ/H mismatch caused the recurring
    # critic-pessimism cascade across p41/p43/p48.  Paper γ=0.997 gives
    # an effective value horizon 1/(1-γ)=333 steps; with imagination
    # H=15 the critic must bootstrap ~318 steps through target_value.
    # For dense signed process-control rewards (mean ~-24/step, V_ss
    # ~-8000) the bootstrap is asked to do too much → critic self-
    # references (tv_r→0.95, rew/v→0.005) regardless of WM quality.
    # γ is now auto-tuned in ``auto_tune_seed_buffer`` to
    # ``clip(1 - 1/(4·H), 0.97, 0.99)`` so the effective value horizon
    # is ≤4× the imagination horizon.  The dataclass default below is
    # retained at the paper value as a fallback only.
    gamma: float = 0.997
    # 2026-05-24 (P48 RCA): default lowered 0.95→0.90.  Bootstrap weight
    # at (γλ)^H drops from 44% to 20% at H=15 — see
    # dreamer-training-diagnosis skill, "critic-training cascade" entry.
    gae_lambda: float = 0.90
    # 2026-05-24 (P48 RCA): default raised 0.02→0.05.  Faster target
    # tracking reduces the stale-target bias that feeds the cascade.
    target_critic_tau: float = 0.05
    tau_ctx: float = 0.1             # context-noise corruption at inference

    # ----- Buffer -----
    # 2026-05-18 raised 200k→400k after reward-head saturation diagnosis
    # on test_sim p25: with 200k cap and ep_len=1220 the buffer holds
    # only 163 episodes (≈32 P3 on-policy + 131 Phase-1 random), so the
    # WM/reward head over-weights stale Phase-1 calibration data even
    # though Phase-3 returns are 30–1000× larger in magnitude than the
    # calibration baseline. 400k holds ~327 episodes and lets fresh
    # on-policy data displace the calibration set faster.
    buffer_capacity_steps: int = 400_000

    # ----- Phase-3 evaluation cadence -----
    phase3_eval_every_iters: int = 5
    # Collect a fresh on-policy episode every K P3 iters. DreamerV3
    # paper Algorithm 1 line 22 calls for every iter; we previously
    # defaulted to K=4 to save wall-clock on CPU plants, but 2026-05-18
    # diagnosis on test_sim p25 showed the reward head saturates and
    # the actor drifts catastrophically when the buffer’s on-policy
    # fraction is too small to retrain the head between P3 iters. K=1
    # restores paper behaviour at the cost of ~30% throughput.
    phase3_collect_every_iters: int = 1
    # (REMOVED 2026-06-12) ``wm_excitation_buffer_frac`` — the WM-only open-loop
    # excitation PARTITION.  It existed to keep gain-rich open-loop data flowing
    # to the WM-update during JOINT closed-loop co-training, but: (1) it was only
    # ever drawn in the P3/joint branch (inert in the phased curriculum's P1/P2),
    # (2) it never demonstrably helped (p109/p110/p115 regressed or were
    # confounded; the known-good p106 never used it), and (3) the p117 clean
    # Stage-1 gain probe proved the WM DYNAMICS are already perfectly identified
    # (posterior→1-step gain x0.998) from the settle-aware seed buffer + random
    # collection — the residual gain under-read is the AUTOENCODER (real→post
    # x0.847), which more open-loop data cannot fix.  Removed as dead weight; the
    # gain-rich seed excitation (PRBS + const/step-settle + step-test, all in the
    # shared buffer) already gives open-loop coverage.
    # Reduce inner train steps in P3 so more iters happen per fixed
    # env-step budget — Optuna's pruner gets more samples and we get
    # finer-grained logs of actor / entropy progression.
    phase3_train_steps_per_iter: int = 25
    # mbrl2 real-sim (2026-07-08): the P3 actor-critic REINFORCE update is
    # ON-POLICY, so it samples a SEPARATE rolling buffer holding ONLY the last
    # ``phase3_onpolicy_buffer_eps`` current-policy episodes — NOT the shared
    # replay buffer, whose Phase-1/2 PRBS / random / step-test / expert seed
    # actions would corrupt the on-policy policy gradient (the p01 MV-chatter
    # root cause: vanilla REINFORCE on off-policy actions pulls the policy toward
    # imitating the full-range excitation).  ``phase3_onpolicy_prefill_eps`` warms
    # it at P3 entry so the critic warmup + first actor steps see current-policy
    # data.  Staleness is bounded to ≈ buffer_eps P3 iters (lr_actor is small).
    phase3_onpolicy_buffer_eps: int = 16
    phase3_onpolicy_prefill_eps: int = 8
    # P3 warmup before reporting EMA to pruner (avoid pruning trials on
    # the first few P3 returns which are dominated by the snapshot
    # actor that hasn't been updated by imagination yet).
    phase3_pruner_warmup_iters: int = 8

    # ---- (P93, 2026-06-06) critic warmup at P3 start (re-added) ----
    # The value head trains ONLY in P3 (P2 does WM + BC + reward-MTP, no value).
    # So the critic enters P3 COLD and its first imagined-return targets are
    # noisy → a documented cascade trigger (rew_to_tgt_var collapse).  This
    # trains the critic ALONE (actor frozen at the snapshot prior) for the
    # first ``p3_critic_warmup_iters`` P3 iters so it is CALIBRATED before it
    # is coupled to the actor — i.e. "critic converges before actor coupling"
    # (functionally = warming it at the end of P2).  An earlier short-budget
    # version was removed 2026-05-20; re-added P93 as an env-gated, properly
    # placed warmup.  Default 10 = p117 recipe: calibrate the critic before it
    # couples to the actor, so P3 doesn't open with a self-inflating critic
    # (promoted 2026-06-14; paper co-trains from P3 start = 0).
    p3_critic_warmup_iters: int = 10
    # ---- (P93, 2026-06-06) protect the WM trunk from agent grads in P2 ----
    # At P1→P2 the BC + reward-MTP losses (agent_finetune_loss, read off
    # ``agent_hid`` = the dynamics' own feature) are added; their gradient
    # flows back into the shared WM dynamics trunk and destabilises it (P90:
    # recon spikes 0.13→0.57 exactly at the P1→P2 boundary, then slowly
    # recovers).  With this ON, ``agent_hid`` is detached before the P2 agent
    # losses so the reward/BC heads still train but the WM dynamics keeps
    # converging on its OWN losses (recon/kl/overshoot/held) — the WM then
    # cleanly converges + plateaus by end of P2 without needing the wm_best
    # restore crutch.  Default True = p117 curriculum-recipe (promoted
    # 2026-06-14; was False).  DREAMER_WM_TRUNK_STOPGRAD_IN_P2.
    wm_trunk_stopgrad_in_p2: bool = True

    # NOTE: ``p3_critic_warmup_iters`` and the ``p3_critic_stability_*``
    # gate (introduced 2026-05-06 and 2026-05-08) were removed on
    # 2026-05-20 along with the entropy-decay belt.  They were
    # short-budget symptom fixes: at 600k env steps a freshly-init
    # critic produced noisy advantages for a few iters before settling
    # and we papered over it by freezing the actor.  With the budget
    # bumped to 1M (paper's control-task minimum) the critic settles
    # naturally during normal training; the freeze just wasted P3
    # iters that the actor needs.


    # NOTE: the adaptive σ-saturation entropy-decay belt (2026-05-08
    # → 2026-05-20) was removed.  Paper (DreamerV3/V4) uses a constant
    # η; the belt actively caused the failure mode it was meant to
    # prevent (run_p30: halving η collapsed entropy to floor, narrowed
    # the replay buffer, degraded WM step-1 MAE 0.10→0.29).  σ-saturation
    # is now handled by the ``policy_log_std_max`` clamp alone.

    # ----- Early stopping (within a single trial) -----
    # Master switch.  All sub-criteria are gated on this.
    early_stop_enable: bool = True
    # P3 plateau: stop when no new best ``return_window_mean`` for this
    # many P3 iters AND we are past ``phase3_pruner_warmup_iters``.
    # (P49 RCA, 2026-05-25: switched from stochastic ``ema_return`` to
    # the deterministic-eval ``return_window_mean`` deque.  ema_return
    # tracks the stochastic on-policy collection return — for narrow-σ
    # process-control policies near a nonlinear plant boundary the
    # stochastic and deterministic returns decouple, and ema_return
    # froze ~iter 110 in P49 while raw deterministic return improved
    # 100× to iter 310.  The wrong ckpt was promoted as best.pt.)
    # The trainer writes ``best.pt`` whenever a new best is reached and
    # copies it to ``final.pt`` on plateau-stop so validation auto-picks
    # the best state without needing runner / validate.py changes.
    early_stop_p3_patience_iters: int = 200
    # Minimum relative improvement (vs current best) that counts as
    # "new best" (avoids ratcheting on noise).
    early_stop_p3_min_improvement: float = 0.01
    # Minimum number of deterministic-eval episodes in the
    # ``return_window`` deque (maxlen=10) before a best can be locked
    # in.  Prevents the first single eval from anchoring best at a
    # noisy early value.  At ``phase3_eval_every_iters=5`` this is
    # ~25 P3 iters of warmup beyond ``phase3_pruner_warmup_iters``.
    early_stop_p3_min_window_n: int = 5
    # Entropy-collapse trip: detect a *sustained* low-entropy regime, not
    # just a single dip.  We maintain a sliding window of the last
    # ``window_iters`` P3 entropy values and trip when at least
    # ``min_frac_below * window_iters`` of them are below
    # ``frac * log(n_action_bins)``.  Empirically (run_20260505_095652)
    # the policy entropy bounces 0.1–1.5 around the threshold during
    # collapse and never accumulates a long enough consecutive streak,
    # so the legacy patience-based detector never trips.
    early_stop_entropy_collapse_frac: float = 0.20
    early_stop_entropy_collapse_window_iters: int = 30
    early_stop_entropy_collapse_min_frac_below: float = 0.70
    # Performance gate (Fix B, 2026-06-22): an entropy-near-floor window is only
    # a COLLAPSE if the policy is also degenerate.  Trip only when the recent
    # median |imag_adv_action_corr| is below this — a committed-and-learning
    # low-σ policy (high corr) is NOT killed.  0.0 ⇒ entropy-only (legacy).
    early_stop_entropy_collapse_min_adv_corr: float = 0.05
    # Legacy: kept for backward compat with env-var overrides.  Set to
    # the same value as ``window_iters`` so single-streak users still get
    # a sensible default; the sliding-window check usually trips first.
    early_stop_entropy_collapse_patience_iters: int = 30
    # Critic-divergence trip: critic_loss above ``factor`` ×
    # rolling-median (window 200 P3 iters) for ``patience`` consecutive
    # logs → stop.
    early_stop_critic_divergence_factor: float = 5.0
    early_stop_critic_divergence_patience_iters: int = 20
    # 2026-05-23 (P41 RCA): bootstrap-cascade canary.  Trip when the
    # critic's target is dominated by its own bootstrap V_slow (reward
    # contributes <``min_rew_var_frac`` of the target variance) AND the
    # return_scale has drifted >``min_return_scale_growth``× since the
    # P3 start, sustained for ``patience`` consecutive P3 log iters.
    # This is the in-training signature of the critic-pessimism /
    # bootstrap-runaway cascade documented in P40/P41 RCAs; aborting
    # on this signal avoids burning compute on the plateau detector.
    # Set ``factor=0.0`` to disable.
    # P74 (2026-05-31): default growth threshold raised 3.0 → 100.0 for the
    # bounded-reward regime.  With reward squashed to [-1,1] a benign P3
    # settles around 25-30× return_scale growth (NOT the 5756× unbounded
    # runaway), and ``rew_to_tgt_var`` is structurally tiny (small rewards),
    # so the old 3.0× threshold was a FALSE POSITIVE that cut P3 to ~19 iters
    # (undertrained actor → noisy MV).  100× still catches a genuine runaway.
    early_stop_cascade_min_rew_var_frac: float = 0.015
    early_stop_cascade_min_return_scale_growth: float = 100.0
    early_stop_cascade_patience_iters: int = 10
    # Grad-skip storm: more than ``max_skips`` skipped optimizer steps
    # within ``window_iters`` consecutive iters → stop (NaN/Inf in
    # actor or critic gradient is unrecoverable in this run).
    early_stop_grad_skip_window_iters: int = 100
    early_stop_grad_skip_max: int = 5
    # P1 mid-check: at the P1→P2 transition, require ``sf_loss`` to have
    # dropped at least ``min_drop_frac`` from its initial value.  If not,
    # WM never learned dynamics; flag the trial so the BO score reflects
    # this (we still let it run — P3 plateau will catch it).  Set to 0.0
    # to disable.
    early_stop_p1_min_sf_drop_frac: float = 0.10
    # P2 mid-check: at the P2→P3 transition, require ``reward_mtp_loss``
    # to be below this absolute value (random-baseline ≈ log(255) ≈ 5.5
    # for default twohot 255-bin head; 4.5 is "started learning").
    early_stop_p2_max_reward_mtp_loss: float = 4.5

    # ----- Phase-transition quality gates (P52 RCA, 2026-05-26) -----
    # P51 entered P2 with an underfit WM (fidelity probe peaked at iter
    # 20 and never improved through iter 70) and P3 with a critic still
    # bootstrap-leaning (rew_to_tgt_var=0.054 at P3 entry → cascade).
    # The fix is to make ``phase{1,2}_env_steps`` lower bounds + quality
    # gates that can extend each phase up to ``max_extension`` × budget.
    #
    # P1 gate: at the P1 env-step budget, only transition to P2 if
    #   (a) ``wm_score_ema >= p1_gate_wm_ema_min``  AND
    #   (b) ``wm_score_ema`` has been within ±``plateau_frac`` of
    #       ``wm_score_ema_best`` for ≥ ``plateau_probes`` probes
    #       (i.e. it's plateaued at a healthy level, not just stalled
    #       at a low one).
    # Else extend P1 by 10 % of budget and re-check at next probe, up
    # to a hard cap of ``(1+max_extension)`` × ``phase1_env_steps``.
    # Set ``p1_gate_wm_ema_min=0.0`` to disable.
    p1_gate_wm_ema_min: float = 1.5
    p1_gate_plateau_frac: float = 0.05
    p1_gate_plateau_probes: int = 3
    p1_gate_max_extension: float = 1.0

    # P2 gate: same idea, on ``reward_mtp_loss``.  In this codebase the
    # critic head only trains in P3 — P2 is WM + reward-MTP head only
    # — so ``critic_rew_to_tgt_var`` is unavailable here.  Use the
    # reward-MTP loss as the P2 lock-in signal: paper-baseline log(255)
    # ≈ 5.5 (random twohot), existing P2 mid-check uses 4.5 as the
    # "started learning" line.  Tighter floor for the gate: P52 reached
    # 2.78 after ~40 P2 iters (still falling); 3.0 is "locked in".
    # The gate compares the median of the last ``recent_iters`` log
    # rows; we require it to be <= ``p2_gate_reward_mtp_max``.
    # Set ``p2_gate_reward_mtp_max=0.0`` (≤0 disables: nothing can be
    # ≤ 0) to disable.
    p2_gate_reward_mtp_max: float = 3.0
    p2_gate_recent_iters: int = 5
    p2_gate_max_extension: float = 0.5

    # 2026-05-27 (P57 RCA): P3 budget floor.  P57 burned 1.14 M of 1.61 M
    # env_steps on P1 extensions, then P2 + the wm-fidelity ES consumed
    # the rest before P3 (actor-critic) ever started — validation ran
    # with an untrained critic.  ``phase3_min_frac`` reserves a fraction
    # of ``total_steps`` for P3 *before* P1/P2 extensions can borrow from
    # it: the runtime cap on ``p1_gate_max_ext_steps + p2_gate_max_ext_steps``
    # is clamped to ``(total_steps - p1 - p2 - phase3_min_frac × total_steps)``,
    # split proportionally to the configured extension ratios.  Set to
    # 0.0 to disable (legacy behaviour: extensions can consume P3).
    phase3_min_frac: float = 0.20

    # ----- I/O -----
    out_dir: str = ''
    log_every: int = 1
    save_every_iters: int = 20
    # Path to a checkpoint (e.g. best.pt from a previous run) to load
    # model weights from at startup.  Optimizers, env-step counter,
    # phase tracking start fresh — only the model state_dict is
    # restored.  Use to warm-start a new run when WM/critic/actor
    # have already learned but you want to change params (σ clamp,
    # warmup, etc.) without throwing away weights.  Empty = cold start.
    init_from_ckpt: str = ''

    # ----- Speedups (DREAMER_FAST_ATTN=1, DREAMER_COMPILE=1) -----
    attn_impl: str = 'auto'          # 'auto'|'manual'|'sdpa'
    compile_mode: str = ''           # '' (off) | 'default' | 'reduce-overhead' | 'max-autotune'

    # ----- Resolved at build-time -----
    obs_dim: int = 0
    action_dim: int = 0
    aug_obs_dim: int = 0
    state_dim: int = 0


# ---------------------------------------------------------------------------
# Env wrapper — stacks state + aug-obs, builds lookback window, computes reward
# ---------------------------------------------------------------------------

class APCEnv:
    """Slim env wrapper around the carryover simulator.

    ``obs_window`` shape ``(lookback, obs_dim)``; ``obs_dim = state_dim + aug_obs_dim``.
    The lookback window is maintained for streaming inference (the V4
    dynamics transformer attends over the last ``lookback`` frames of
    encoded latents).
    """

    def __init__(self, cfg: TrainConfig, rng: np.random.Generator):
        self.cfg = cfg
        self.rng = rng
        self.sim = create_sim(episode_length=cfg.episode_length,
                              sample_rate=cfg.sample_rate)
        self.meta = resolve_sim_metadata(self.sim)
        self.action_dim = int(self.meta['action_dim'])
        self.state_dim = int(self.meta['state_dim'] or 0)
        self.cv_indices = list(self.meta['cv_indices'])
        self.mv_norm_ranges = list(self.meta['mv_normalization_ranges'])
        self.cv_norm_ranges = list(self.meta['cv_normalization_ranges'])

        self.bounds = load_objective_bounds() or {}
        self.obj_w = load_objective_weights() or {}
        self.obj_spec = load_full_objective_spec() or {}

        # P86: action→MV maps over the FIXED physical envelope (the MV
        # normalisation range) rather than the static base bounds, so the
        # agent can reach & learn every attainable active operator limit and
        # no fixed "base floor" is baked into the actuator (see
        # TrainConfig.mv_action_map_full_range).  ``action_to_control`` /
        # ``control_to_action`` read this key from the bounds dict; when it is
        # absent they fall back to the P60 static-base-bounds mapping.
        if (bool(getattr(self.cfg, 'mv_action_map_full_range', True))
                and self.mv_norm_ranges):
            self.bounds['mv_action_range'] = [list(b) for b in self.mv_norm_ranges]

        # Setpoint manager — packed aug-obs.
        mvs_raw = self.bounds.get('mvs')
        if isinstance(mvs_raw, dict) and mvs_raw:
            mvb = np.asarray([mvs_raw[k] for k in sorted(mvs_raw)],
                             dtype='float32').reshape(-1, 2)
        elif isinstance(mvs_raw, (list, tuple)) and len(mvs_raw) > 0:
            mvb = np.asarray(mvs_raw, dtype='float32').reshape(-1, 2)
        else:
            mv_bounds_list = list(self.bounds.get('mv_bounds', []))
            if mv_bounds_list:
                mvb = np.asarray(mv_bounds_list, dtype='float32').reshape(-1, 2)
            else:
                mvb = np.tile(np.array([[0.0, 100.0]], dtype='float32'),
                              (self.action_dim, 1))
        cv_bounds_raw = self.bounds.get('outputs') or self.bounds.get('cvs') or {}
        if isinstance(cv_bounds_raw, dict) and cv_bounds_raw:
            cvb = np.asarray([cv_bounds_raw[k] for k in sorted(cv_bounds_raw)],
                             dtype='float32').reshape(-1, 2)
        elif isinstance(cv_bounds_raw, (list, tuple)) and len(cv_bounds_raw) > 0:
            cvb = np.asarray(cv_bounds_raw, dtype='float32').reshape(-1, 2)
        else:
            cv_bounds_list = list(self.bounds.get('cv_bounds', []))
            if cv_bounds_list:
                cvb = np.asarray(cv_bounds_list, dtype='float32').reshape(-1, 2)
            else:
                cvb = np.tile(np.array([[0.0, 100.0]], dtype='float32'),
                              (max(1, len(self.cv_indices)), 1))
        n_cv = cvb.shape[0]

        # Engineering-unit base bounds, exposed for the APC expert (BC anchor).
        # The expert reasons in engineering units and respects these base
        # operating limits; its demonstration MV is converted to the actor's
        # BC target via ``control_to_action`` over the env's action-mapping
        # band (P86), so the round-trip through ``env.step`` is exact whatever
        # the mapping basis (physical envelope or base bounds).
        self.mv_bounds_eu = np.asarray(mvb, dtype='float64').reshape(-1, 2)
        self.cv_bounds_eu = np.asarray(cvb, dtype='float64').reshape(-1, 2)

        rt_spec = (self.obj_spec or {}).get('runtime_setpoints', {}) or {}
        targets_enabled_spec = rt_spec.get('targets_enabled', False)
        if isinstance(targets_enabled_spec, bool):
            cv_target_enabled = np.array([targets_enabled_spec] * n_cv, dtype=bool)
        else:
            te = list(targets_enabled_spec) + [False] * n_cv
            cv_target_enabled = np.array(te[:n_cv], dtype=bool)
        cv_targets = np.array(
            [0.5 * (cvb[i, 0] + cvb[i, 1]) for i in range(n_cv)], dtype='float32')

        self.setpoint_mgr = RuntimeSetpointManager(
            n_mv=self.action_dim, n_cv=n_cv,
            base_mv_bounds=mvb,
            base_cv_bounds=cvb,
            base_cv_targets=cv_targets,
            cv_target_enabled=cv_target_enabled,
            cfg=RuntimeSetpointConfig(
                # P86: allow training/validation on a single consistent limit
                # set (active ≡ base) by disabling mid-episode limit-step
                # variation.  Default True preserves the legacy operator-limit
                # tracking curriculum.
                bounds_enabled=bool(getattr(
                    self.cfg, 'runtime_setpoint_variation', True)),
            ),
            mv_norm_bounds=np.asarray(self.mv_norm_ranges, dtype='float32')
                if self.mv_norm_ranges else np.zeros((0, 2), dtype='float32'),
            cv_norm_bounds=np.asarray(self.cv_norm_ranges, dtype='float32')
                if self.cv_norm_ranges else np.zeros((0, 2), dtype='float32'),
        )
        self.setpoint_mgr._rng = rng

        self.aug_obs_dim = int(self.setpoint_mgr.aug_obs_dim)
        # ---- Derived observable block (Stage B, 2026-05-21) ---------------
        # Belief-state augmentation: cheap per-CV rolling statistics
        # (mean tracking error, Δcv, variance) appended to the
        # augmented-obs block.  OFF by default; enable via
        # ``DREAMER_DERIVED_OBSERVABLES=1``.  When ON, ``aug_obs_dim``
        # grows by ``3 * n_cv`` and the running obs normalizer auto-
        # adapts (features are bounded by construction).
        self._derived_features: Optional[DerivedFeatures] = None
        if derived_observables_enabled():
            n_cv = int(len(self.cv_indices))
            self._derived_features = DerivedFeatures(
                n_cv=n_cv,
                window=derived_observables_window(
                    tau=(self._resolve_plant_timing()[0] or None),
                    sample_rate=float(getattr(self.cfg, 'sample_rate', 1.0)
                                       or 1.0),
                ),
            )
            self.aug_obs_dim += int(self._derived_features.feat_dim)
        # ---- Integral (accumulated-violation) observation channel -------
        # When the integral reward term is enabled (OBJECTIVE_INTEGRAL_COEF
        # > 0 or spec ``integral_coef``), expose the per-CV anti-windup
        # accumulator to the agent + WM so the otherwise non-Markov
        # integral penalty stays predictable (state augmentation).  The
        # accumulator is normalised by the windup cap to [0, 1].  CV-only.
        from utils.objective_runtime import resolve_integral_config
        self._integral_enabled, _intg_coef, self._integral_windup = \
            resolve_integral_config(self.obj_spec)
        n_cv_intg = int(len(self.cv_indices))
        self._integral_cv = np.zeros(n_cv_intg, dtype='float32')
        if self._integral_enabled:
            self.aug_obs_dim += n_cv_intg
        self.obs_dim = self.state_dim + self.aug_obs_dim

        self._window: Optional[np.ndarray] = None
        self._t = 0
        self._prev_control = np.zeros(self.action_dim, dtype='float32')
        # Previous-step raw per-channel violation depth (lo_viol+hi_viol),
        # consumed by ``compute_objective_components`` for the optional
        # derivative (violation-rate) term.  ``None`` on the first step of
        # an episode (rate term skipped that step).
        self._prev_mv_violation_per_channel: Optional[list] = None
        self._prev_cv_violation_per_channel: Optional[list] = None
        self._schedule: List[Dict] = []
        self.reward_scale: float = 1.0
        self._last_cv_violation_sum: float = 0.0
        self._last_mv_violation_sum: float = 0.0
        # ---- Hidden (truly unmeasured) CV disturbance process -------------
        # Per-episode OU process injecting a smoothly evolving bias into
        # the CV state channels.  The state is NOT exposed to the agent
        # or the WM — it is the "unmeasured upstream upset" a deployed
        # APC controller must reject.  Replaces the legacy unmeasured-CV
        # step events (now removed) which were Dirac spikes the WM had
        # no way to predict.  See ``utils/hidden_disturbance.py``.
        self._hidden_disturbance = None  # type: Optional[object]
        # Phase-aware per-episode disturbance probability override.
        # Set by the trainer at phase transitions (P1 default 0.10,
        # P2 ramp 0.10->0.20, P3 ramp 0.20->0.50).  ``None`` = read
        # from env var directly.
        self._disturbance_prob_override: Optional[float] = None
        # Force flag: when True, every reset() always builds the hidden
        # process (validation path).  False = Bernoulli toggle.
        self._hidden_disturbance_force: bool = False
        # Training progress in [0, 1] (env_steps / total_steps).  Used by
        # the hidden-OU amplitude curriculum at reset() time.  Updated
        # by the trainer at every episode boundary via ``set_training_progress``.
        self._training_progress: float = 0.0
        # Current phase (1=WM, 2=critic, 3=actor+critic).  Used by the
        # phase-aware amplitude cap in ``curriculum_amp_scale``.
        self._current_phase: int = 1

        # ---- Raw-reward clipping (P37 onward, 2026-05-22) ---------------
        # The objective's quadratic violation tail can produce
        # ``raw_reward`` magnitudes 1000× above the operating-region
        # median (p36: raw_min=-185, raw_abs_p95=0.20).  That dynamic
        # range pushes the symlog-twohot reward predictor's mass into
        # ~14 of 255 bins, capping ``reward_mtp_loss`` at the
        # operating-region entropy floor regardless of training time.
        # Clipping the raw tail at ``DREAMER_REWARD_RAW_CLIP_MIN``
        # P43 (2026-05-23): default loosened from -30 to -1e6.  Symlog
        # in the V4 two-hot reward head already compresses heavy tails
        # (symlog(-30)≈-3.4); the explicit -30 clip was censoring 5 %
        # of the operating-region reward distribution and contributing
        # to critic underestimation at the operating-band edges (see
        # audit output/test_sim/_data_audit_v2_*).  The remaining
        # -1e6 / +1e18 clip is a runaway-bug safety net, not a routine
        # censor; ``_reward_clip_warned`` emits a one-shot warning if
        # it ever triggers so we notice silent saturation.
        try:
            self._reward_clip_min: float = float(
                os.environ.get('DREAMER_REWARD_RAW_CLIP_MIN', '-1e6'))
        except Exception:
            self._reward_clip_min = -1e6
        try:
            self._reward_clip_max: float = float(
                os.environ.get('DREAMER_REWARD_RAW_CLIP_MAX', '1e18'))
        except Exception:
            self._reward_clip_max = 1e18
        self._reward_clip_warned: bool = False
        # P62 (2026-05-28): track whether the negative-tail clip was set
        # explicitly by the user (DREAMER_REWARD_RAW_CLIP_MIN in env)
        # vs left at the default -1e6.  The post-calibration adaptive
        # setter (see ``train`` after ``calibrate_reward_scale``) only
        # overrides this when the user did NOT set it explicitly.
        self._reward_clip_min_user_set: bool = (
            'DREAMER_REWARD_RAW_CLIP_MIN' in os.environ)

        # ---- A' : potential-based reward shaping state ------------------
        # Dense band-keeping shaping (see TrainConfig.reward_shaping_coef).
        # Disabled by default; the trainer sets ``_shaping_enabled=True`` on
        # the training env only.  ``_prev_potential`` caches Φ(s_t) across
        # steps for the F = γΦ(s') − Φ(s) telescoping form.
        self._shaping_enabled: bool = False
        self._shaping_coef: float = float(
            getattr(cfg, 'reward_shaping_coef', 0.0) or 0.0)
        self._shaping_gamma: float = float(getattr(cfg, 'gamma', 0.997) or 0.997)
        self._prev_potential: Optional[float] = None
        # Fix 2a (p129): margin-gated economic-shaping weight (Φ_econ pull,
        # suppressed near a constraint by the CV-safety gate).  0 ⇒ off.
        self._shaping_econ_coef: float = float(
            getattr(cfg, 'shaping_econ_coef', 0.0) or 0.0)

        # ---- P73/P77 : bounded training reward --------------------------
        # When enabled, the per-step TRAINING reward is mapped into [-B, B]
        # (``reward_scale`` bypassed) so imagined returns stay bounded and
        # the return_scale percentile cannot run away (cascade root-cause
        # fix).  ``info['raw_reward']`` stays unshaped for validation scoring.
        #
        # P77: the mapping is now a SCALE-INVARIANT LINEAR REMAP
        #   reward = clip(raw * (B / reward_clip_ref), -B, B)
        # where ``reward_clip_ref`` is the econ-derived adaptive reward clip
        # exposed by objective_runtime (== max-violation-weight * d_diff²).
        # Because objective_runtime already tanh-saturates ``raw`` at
        # reward_clip_ref, |raw| <= reward_clip_ref, so the remap lands in
        # [-B, B] and the absolute magnitude of the user's economic weights
        # CANCELS (double the econ weights ⇒ reward_clip_ref doubles ⇒ the
        # ratio is unchanged).  This replaces the old symlog squash, which
        # (a) double-symlogged against the twohot head's own symlog and
        # (b) saturated the actor reward to a near-binary signal whenever
        # reward_clip_ref was large.  Single symlog now happens only inside
        # the reward head, restoring head resolution + actor gradient.
        self._bound_reward: bool = bool(
            getattr(cfg, 'bound_training_reward', False))
        self._bound_reward_max: float = float(
            getattr(cfg, 'bound_training_reward_max', 6.0) or 6.0)
        # Fallback reward_clip_ref when a comps dict lacks the field (older
        # objective_runtime / degenerate weights).  Matches the historical
        # adaptive-clip floor so small-weight sims behave sensibly.
        self._bound_reward_ref_fallback: float = float(
            getattr(cfg, 'bound_training_reward_ref', 50.0) or 50.0)
        self._bound_reward_ref: float = self._bound_reward_ref_fallback

        # ---- Per-dim observation standardizer ---------------------------
        # Running mean/std updated from every raw obs vector seen by the
        # env.  Applied in ``_build_obs_vec`` before the obs is exposed to
        # the world model so the tokenizer sees zero-mean unit-std inputs
        # regardless of physical units (CV ~50, setpoint ~50, MV ~25).
        # Diagnosed 2026-05-03: without this, channels of std=10 saturate
        # the encoder while constant channels (std=0) carry no info,
        # collapsing the latent.  ``learn`` flag controls whether new
        # samples update the stats — set False at validation/eval time so
        # the tokenizer sees the same statistics it was trained against.
        self._obs_mean: np.ndarray = np.zeros(self.obs_dim, dtype='float64')
        self._obs_var: np.ndarray = np.ones(self.obs_dim, dtype='float64')
        self._obs_count: float = 1e-4
        self._obs_norm_learn: bool = True

    # ---------- observation normalizer (load/save/apply) -----------------
    def set_training_progress(self, progress: float) -> None:
        """Set ``progress = env_steps / total_steps`` in ``[0, 1]``.

        Consumed by the hidden-OU amplitude curriculum at every ``reset()``.
        Trainer pushes this at each outer-loop iteration; defaults to 0.0
        (curriculum start) until updated.
        """
        try:
            self._training_progress = float(max(0.0, min(1.0, progress)))
        except Exception:
            self._training_progress = 0.0

    def set_sim_noise_scale(self, scale: float) -> None:
        """Set the global process+measurement noise multiplier on the sim
        noise wrapper (P89).  ``0.0`` = fully clean episode (steady-state
        seeds); ``1.0`` = full configured noise.  No-op when the underlying
        sim has no ``SimNoiseWrapper`` (e.g. a noise-free test double).
        """
        s = self.sim
        try:
            if hasattr(s, 'set_noise_scale'):
                s.set_noise_scale(float(scale))
        except Exception:
            pass

    def set_obs_norm_stats(self, mean: np.ndarray, var: np.ndarray,
                            count: float = 1.0, *, learn: bool = False) -> None:
        m = np.asarray(mean, dtype='float64').reshape(-1)
        v = np.asarray(var, dtype='float64').reshape(-1)
        if m.shape[0] != self.obs_dim or v.shape[0] != self.obs_dim:
            raise ValueError(f'obs_norm shape mismatch: got {m.shape}/{v.shape}, '
                             f'expected ({self.obs_dim},)')
        self._obs_mean = m.copy()
        self._obs_var = np.maximum(v.copy(), 1e-6)
        self._obs_count = float(count)
        self._obs_norm_learn = bool(learn)

    def get_obs_norm_stats(self) -> Dict[str, np.ndarray]:
        return {
            'mean': self._obs_mean.astype('float64'),
            'var':  self._obs_var.astype('float64'),
            'count': float(self._obs_count),
        }

    def _update_obs_norm(self, raw_obs: np.ndarray) -> None:
        # Welford-style running mean/var update on a single sample.
        x = np.asarray(raw_obs, dtype='float64').reshape(-1)
        self._obs_count += 1.0
        delta = x - self._obs_mean
        self._obs_mean += delta / self._obs_count
        delta2 = x - self._obs_mean
        # Running variance (uncorrected, fine for scale stats).
        self._obs_var = self._obs_var + (delta * delta2 - self._obs_var) / self._obs_count

    def _normalize_obs(self, raw_obs: np.ndarray) -> np.ndarray:
        std = np.sqrt(np.maximum(self._obs_var, 1e-6))
        # Clamp std lower bound: dims that are constant within an episode
        # would otherwise divide by ~0 and amplify noise; treat them as
        # "zero info, identity scale".
        std = np.clip(std, 1e-3, None)
        out = (np.asarray(raw_obs, dtype='float32') -
               self._obs_mean.astype('float32')) / std.astype('float32')
        # Guard against pathological values (numerical edge cases).
        return np.clip(out.astype('float32'), -10.0, 10.0)

    def _build_obs_vec(self, state: np.ndarray) -> np.ndarray:
        aug = self.setpoint_mgr.get_augmented_obs_channels()
        parts = [np.asarray(state, dtype='float32').reshape(-1),
                 np.asarray(aug, dtype='float32').reshape(-1)]
        if self._derived_features is not None:
            # Update with the current CV slice + current CV setpoints,
            # then append the feature block.  CV setpoints may be NaN
            # for slots without a target — DerivedFeatures handles that.
            try:
                cv_now = np.asarray(state, dtype='float64')[self.cv_indices]
            except Exception:
                cv_now = np.zeros(self._derived_features.n_cv, dtype='float64')
            sp_now = np.asarray(
                getattr(self.setpoint_mgr, 'current_cv_targets',
                        np.full(self._derived_features.n_cv, np.nan)),
                dtype='float64',
            ).reshape(-1)
            if sp_now.shape[0] != self._derived_features.n_cv:
                sp_now = np.full(self._derived_features.n_cv, np.nan,
                                 dtype='float64')
            self._derived_features.update(cv_now, sp_now)
            parts.append(self._derived_features.features())
        if self._integral_enabled:
            # Anti-windup accumulator normalised to [0, 1] by the windup cap
            # so it sits on the same scale as the other augmented channels.
            cap = max(1e-6, float(self._integral_windup))
            parts.append(np.clip(
                np.asarray(self._integral_cv, dtype='float32') / cap,
                0.0, 1.0).reshape(-1))
        raw = np.concatenate(parts, axis=0).astype('float32')
        if self._obs_norm_learn:
            self._update_obs_norm(raw)
        return self._normalize_obs(raw)

    def _resolve_plant_timing(self) -> "tuple[float, float]":
        """Resolve ``(tau_dominant, dead_time)`` in plant time units.

        ``TrainConfig`` carries NO plant-timing fields, so reading
        ``cfg.tau`` / ``cfg.dead_time`` always yielded 0.0 — which collapsed
        the hidden-disturbance schedule to a 1-sample timescale (settle≈4):
        ``ou_drift`` became α=1 white noise and step/ramp events became
        sub-settling spikes crammed into the first ~50 steps (front-loaded,
        high-frequency, never reaching steady state).  Both ``single_run.py``
        and ``evaluation.validate`` export the identified plant timing as
        ``IDENTIFIED_TAU_DOMINANT`` / ``IDENTIFIED_DEAD_TIME`` env vars, so
        source from there (with the legacy ``SIM_`` prefix and the sim's own
        attributes as fallbacks).  Fixed 2026-06-08.
        """
        def _envf(*names: str) -> float:
            for n in names:
                v = str(os.environ.get(n, '')).strip()
                if not v:
                    continue
                try:
                    x = float(v)
                except Exception:
                    continue
                if x > 0:
                    return x
            return 0.0
        sim = getattr(self, 'sim', None)
        tau = float(getattr(self.cfg, 'tau', 0.0) or 0.0)
        if tau <= 0:
            tau = _envf('IDENTIFIED_TAU_DOMINANT', 'SIM_IDENTIFIED_TAU_DOMINANT')
        if tau <= 0 and sim is not None:
            tau = float(getattr(sim, 'tau_dominant', 0.0)
                        or getattr(sim, 'tau', 0.0) or 0.0)
        dead = float(getattr(self.cfg, 'dead_time', 0.0) or 0.0)
        if dead <= 0:
            dead = _envf('IDENTIFIED_DEAD_TIME', 'SIM_IDENTIFIED_DEAD_TIME')
        if dead <= 0 and sim is not None:
            dead = float(getattr(sim, 'dead_time', 0.0) or 0.0)
        return float(tau), float(dead)

    def set_domain_randomization(self, enabled: bool) -> bool:
        """Toggle the sim's per-episode domain randomizer (output gain/bias/
        actuator-lag jitter).  Returns True if a randomizer was found.

        Used by the staged curriculum (2026-06-20, p132 DR RCA): DR is an
        ACTOR-robustness mechanism (Stage 3), but it was leaking into the
        Stage-1/2 WORLD-MODEL identification — the WM was being fit to a
        ±frac-randomised OUTPUT GAIN and then scored against the NOMINAL plant,
        so the categorical latent had to model a gain DISTRIBUTION and the
        identified gain came out attenuated (the same ~0.85 'ceiling' seen
        across runs; the eval already disables DR via _quiet_env, so this is a
        TRAIN-time confound).  Identify the plant CLEAN (DR off in P1/P2), then
        randomise for the actor (DR on in P3).  ``frac`` is preserved so
        re-enabling restores the configured magnitude.
        """
        sim = getattr(self, 'sim', None)
        rd = getattr(sim, '_randomizer', None)
        if rd is None and sim is not None:
            rd = getattr(getattr(sim, '_sim', None), '_randomizer', None)
        if rd is None or not hasattr(rd, 'enabled'):
            return False
        rd.enabled = bool(enabled)
        return True

    def reset(self, *, exploration: bool = False) -> np.ndarray:
        state = self.sim.reset()
        if isinstance(state, tuple):
            state = state[0]
        state = np.asarray(state, dtype='float32').reshape(-1)
        self._t = 0
        self._prev_control = np.zeros(self.action_dim, dtype='float32')
        self._prev_mv_violation_per_channel = None
        self._prev_cv_violation_per_channel = None
        self._integral_cv = np.zeros_like(self._integral_cv)
        self._last_cv_violation_sum = 0.0
        self._last_mv_violation_sum = 0.0
        self._prev_potential = None
        if self._derived_features is not None:
            try:
                cv0 = np.asarray(state, dtype='float64')[self.cv_indices]
            except Exception:
                cv0 = None
            self._derived_features.reset(cv0)
        self.setpoint_mgr.reset(episode_length=self.cfg.episode_length,
                                curriculum_fraction=1.0)
        intensity = 1.0 if not exploration else 1.2
        self._schedule = build_training_disturbance_schedule(
            episode_length=self.cfg.episode_length,
            rng=self.rng,
            intensity=intensity,
            sim=self.sim,
        )
        # Per-episode hidden OU disturbance.  Bernoulli toggle gated
        # by phase-aware prob (P1/P2: 0.3, P3: 0.5).  Set force=True
        # in validation to always build.
        prob = (self._disturbance_prob_override
                if self._disturbance_prob_override is not None
                else get_phase_disturbance_prob(phase=1))
        tau_dom, dead_time = self._resolve_plant_timing()
        sample_rate = float(getattr(self.cfg, 'sample_rate', 1.0) or 1.0)
        self._hidden_disturbance = maybe_build_hidden_disturbance(
            rng=self.rng,
            sim=self.sim,
            tau_dom=tau_dom,
            sample_rate=sample_rate,
            prob=float(prob),
            force=bool(self._hidden_disturbance_force),
            progress=float(self._training_progress),
            phase=int(self._current_phase),
            dead_time=dead_time,
            episode_length=int(self.cfg.episode_length),
        )
        # P89 noise-amplitude curriculum: ramp process+measurement noise from
        # ~0 at progress=0 to full by ~40 % (P1), full in P3.  Applied every
        # reset so the live training_progress/phase is reflected.  Clean
        # steady-state seeds override this to 0.0 after their own reset().
        if bool(getattr(self.cfg, 'process_noise_curriculum', True)):
            self.set_sim_noise_scale(noise_curriculum_scale(
                float(self._training_progress), int(self._current_phase)))
        else:
            self.set_sim_noise_scale(1.0)
        # Fresh per-episode trace for the WM disturbance-estimator head target.
        self._hidden_disturbance_trace = []
        obs_vec = self._build_obs_vec(state)
        self._window = np.tile(obs_vec, (self.cfg.lookback, 1)).astype('float32')
        return self._window.copy()

    def pop_episode_disturbance(self, T: int) -> np.ndarray:
        """Return the recorded hidden-disturbance trace for the just-finished
        episode as a ``(T, n_cv)`` float32 array (zero-padded / truncated to
        ``T`` steps), then clear it.  Consumed by ``TrajectoryBuffer`` to build
        the WM disturbance-estimator head's supervised target.  When no trace
        was recorded (head disabled) an all-zero array is returned."""
        n_cv = len(self.cv_indices)
        out = np.zeros((int(T), n_cv), dtype='float32')
        trace = getattr(self, '_hidden_disturbance_trace', None)
        if trace:
            arr = np.asarray(trace, dtype='float32')
            if arr.ndim == 2 and arr.shape[1] == n_cv:
                m = min(int(T), arr.shape[0])
                out[:m] = arr[:m]
        self._hidden_disturbance_trace = []
        return out

    def _shaping_potential(self, state: np.ndarray) -> float:
        """Dense shaping potential Φ(s) for reward shaping.

        Φ(s) = Φ_safe(s) + ``_shaping_econ_coef``·gate(s)·Φ_econ(s).

        * Φ_safe ∈ [0, 1]: mean over CVs of the normalised margin to the nearest
          band edge (1.0 at the band centre / at an enabled CV target, 0.0 at
          the edge, clamped to 0 outside) — the positive-gradient signal the
          one-sided economic objective lacks inside the safe band.
        * Φ_econ ∈ [0, 1] (Fix 2a, added only when ``_shaping_econ_coef`` > 0):
          a state-based economic-direction potential (``_economic_potential``),
          MARGIN-GATED by ``gate`` — the mean CV safety headroom ramped over the
          inner ``shaping_econ_margin_frac`` of the half-band — so the economic
          pull is SUPPRESSED near a constraint (gate→0) and active only with safe
          headroom (gate→1).  Densifies the near-invisible economic gradient
          without driving the CV into the limit (the reverted-R2a failure).

        Potential-based (telescoping, same γ; gate is a state function) ⇒
        policy-invariant (Ng 1999).  Read-only; no side effects.
        """
        sp = self.setpoint_mgr
        try:
            bounds = np.asarray(sp.current_cv_bounds, dtype='float64').reshape(-1, 2)
            targets = np.asarray(sp.current_cv_targets, dtype='float64').reshape(-1)
            tgt_en = np.asarray(sp.cv_target_enabled, dtype=bool).reshape(-1)
        except Exception:
            return 0.0
        vals: List[float] = []
        gate_vals: List[float] = []
        mf_econ = float(np.clip(
            getattr(self.cfg, 'shaping_econ_margin_frac', 0.5), 1e-3, 1.0))
        for i, ci in enumerate(self.cv_indices):
            if ci is None or i >= bounds.shape[0] or int(ci) >= state.shape[0]:
                continue
            lo = float(bounds[i, 0]); hi = float(bounds[i, 1])
            half = 0.5 * (hi - lo)
            if half <= 1e-9:
                continue
            cv = float(state[int(ci)])
            # CV safety headroom (0 at either edge, 1 at the band centre) — used
            # both for the (no-target) flat-top safety potential and the
            # economic gate.
            m = min(cv - lo, hi - cv) / half
            gate_vals.append(float(np.clip(m / mf_econ, 0.0, 1.0)))
            if i < tgt_en.shape[0] and bool(tgt_en[i]) and np.isfinite(targets[i]):
                d = abs(cv - float(targets[i])) / half
                vals.append(max(0.0, 1.0 - d))
            else:
                # Flat-top SAFETY-MARGIN potential (p125 RCA, 2026-06-16): for a
                # range/limit objective (no enabled CV target) Φ is FLAT (=1) in
                # the interior and ramps 0→1 only within a margin band of width
                # ``margin_frac·half`` at each edge — no centre-pull (economics
                # free) with a concentrated, ~1/frac steeper pull-back in the
                # near-constraint zone.  ``margin_frac=1.0`` recovers the legacy
                # centre-peaked tent.
                mf = float(getattr(self.cfg, 'shaping_safe_margin_frac', 0.25))
                mf = float(np.clip(mf, 1e-3, 1.0))
                vals.append(float(np.clip(m / mf, 0.0, 1.0)))
        phi_safe = float(np.mean(vals)) if vals else 0.0
        econ_coef = float(getattr(self, '_shaping_econ_coef', 0.0) or 0.0)
        if econ_coef > 0.0:
            gate = float(np.mean(gate_vals)) if gate_vals else 0.0
            return phi_safe + econ_coef * gate * self._economic_potential(state)
        return phi_safe

    def _economic_potential(self, state: np.ndarray) -> float:
        """State-based economic-direction potential Φ_econ ∈ [0, 1] (Fix 2a).

        A linear ramp across each economically-weighted MV / CV channel's
        engineering band, oriented by the SIGN of its economic weight — the
        direction that lowers the objective's economic penalty
        (``reward −= Σ (x−typical)·w``, so ∂reward/∂x = −w):

            w > 0  ⇒  lower value is economically better  ⇒  g = 1 − x_norm
            w < 0  ⇒  higher value is economically better  ⇒  g = x_norm

        aggregated as the |weight|-weighted mean over all economic channels.
        ``x_norm`` is clamped to the band so Φ_econ adds ZERO gradient outside
        the limits (matches the objective's clipped-to-bounds economic term).
        Returns 0.5 (a constant, policy-inert) when no economic weights are
        configured.  Read-only.  Policy-invariant as a potential for ANY plant
        (Ng 1999) — safe on nonlinear simulators (densifies the economic
        gradient, never moves the optimum).
        """
        num = 0.0
        den = 0.0
        try:
            mv_idx = [int(x) for x in self.meta.get('mv_indices', [])]
        except Exception:
            mv_idx = []
        mv_w = list(self.obj_w.get('mv_economic_weights', []) or [])
        for i, si in enumerate(mv_idx):
            if i >= self.mv_bounds_eu.shape[0] or si < 0 or si >= state.shape[0]:
                continue
            w = float(mv_w[i]) if i < len(mv_w) else 0.0
            if abs(w) < 1e-12:
                continue
            lo = float(self.mv_bounds_eu[i, 0]); hi = float(self.mv_bounds_eu[i, 1])
            if hi - lo <= 1e-9:
                continue
            x = float(np.clip((float(state[si]) - lo) / (hi - lo), 0.0, 1.0))
            g = (1.0 - x) if w > 0.0 else x
            num += abs(w) * g
            den += abs(w)
        cv_w = list(self.obj_w.get('cv_economic_weights', []) or [])
        for j, ci in enumerate(self.cv_indices):
            if ci is None or j >= self.cv_bounds_eu.shape[0] \
                    or int(ci) >= state.shape[0]:
                continue
            w = float(cv_w[j]) if j < len(cv_w) else 0.0
            if abs(w) < 1e-12:
                continue
            lo = float(self.cv_bounds_eu[j, 0]); hi = float(self.cv_bounds_eu[j, 1])
            if hi - lo <= 1e-9:
                continue
            y = float(np.clip((float(state[int(ci)]) - lo) / (hi - lo), 0.0, 1.0))
            g = (1.0 - y) if w > 0.0 else y
            num += abs(w) * g
            den += abs(w)
        if den <= 0.0:
            return 0.5
        return float(np.clip(num / den, 0.0, 1.0))

    def step(self, action_norm: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        action_norm = np.asarray(action_norm, dtype='float32').reshape(self.action_dim)
        action_01 = 0.5 * (np.clip(action_norm, -1.0, 1.0) + 1.0)
        control = action_to_control(action_01, self.bounds, self.setpoint_mgr)
        self.setpoint_mgr.step(self._t)
        # P86 MV hard-clamp: DCS-style RUNTIME SAFETY limiter only.  Default
        # OFF during training/validation so the agent must LEARN to respect the
        # (moving) active operator limits — a clamp here would mask whether it
        # actually learned.  Enable at deployment via ``DREAMER_MV_HARD_CLAMP=1``.
        if bool(getattr(self.cfg, 'mv_hard_clamp', False)):
            try:
                cmb = np.asarray(self.setpoint_mgr.current_mv_bounds,
                                 dtype='float32').reshape(-1, 2)
                control = np.asarray(control, dtype='float32').reshape(-1)
                k = min(cmb.shape[0], control.shape[0])
                if k > 0:
                    control[:k] = np.clip(control[:k], cmb[:k, 0], cmb[:k, 1])
            except Exception:
                pass
        next_state = self.sim.step(control)
        if isinstance(next_state, tuple):
            next_state = next_state[0]
        next_state = np.asarray(next_state, dtype='float32').reshape(-1)
        try:
            apply_disturbance_schedule(next_state, self.sim, self._schedule)
        except Exception as _de:
            # Don't crash training on schedule errors, but no longer swallow
            # them silently — the previous ``except: pass`` masked a numpy
            # array-truth bug that caused validation's scripted-disturbance
            # episode to skip without explanation.  Log once per env.
            if not getattr(self, '_disturbance_err_logged', False):
                import traceback
                print(f'[env.step] apply_disturbance_schedule error '
                      f'(further occurrences silenced): {_de!r}', flush=True)
                traceback.print_exc()
                self._disturbance_err_logged = True
        # Hidden (truly unmeasured) LOAD disturbance: advance the load schedule,
        # filter it through the disturbance transfer function Gd (dead-time +
        # first-order lag) and add the resulting smooth CV effect.  Only active
        # on episodes where the per-episode Bernoulli toggle (or force flag)
        # fired in reset().  ``last_applied`` = the per-CV d_cv just injected.
        hidden_applied = np.zeros(len(self.cv_indices), dtype='float32')
        if self._hidden_disturbance is not None:
            try:
                self._hidden_disturbance.step(next_state)
                # Record the per-CV offset just injected (aligned to
                # cv_indices) so evaluation can surface the otherwise-
                # invisible unmeasured disturbance on its plots/npz.
                la = np.asarray(getattr(self._hidden_disturbance,
                                        'last_applied', []),
                                dtype='float32').reshape(-1)
                proc_idx = list(getattr(self._hidden_disturbance,
                                        'cv_indices', []))
                for p, idx in enumerate(proc_idx):
                    if p < la.shape[0] and idx in self.cv_indices:
                        hidden_applied[self.cv_indices.index(idx)] = la[p]
            except Exception as _hde:
                if not getattr(self, '_hidden_dist_err_logged', False):
                    import traceback
                    print(f'[env.step] hidden_disturbance error '
                          f'(further occurrences silenced): {_hde!r}', flush=True)
                    traceback.print_exc()
                    self._hidden_dist_err_logged = True
        # Accumulate the per-step hidden-disturbance offset for the WM
        # disturbance-estimator head's supervised target (env-bound capture;
        # the TrajectoryBuffer pulls the episode trace in add_episode()).
        trace = getattr(self, '_hidden_disturbance_trace', None)
        if trace is not None:
            trace.append(hidden_applied.copy())
        comps = compute_objective_components(
            state=next_state, sim=self.sim,
            control=control, prev_control=self._prev_control,
            obj_w=self.obj_w, bounds=self.bounds,
            setpoint_manager=self.setpoint_mgr,
            objective_spec=self.obj_spec,
            prev_mv_violation_per_channel=self._prev_mv_violation_per_channel,
            prev_cv_violation_per_channel=self._prev_cv_violation_per_channel,
            prev_integral_cv_per_channel=self._integral_cv,
        )
        raw_reward = float(comps['reward'])
        # Apply raw clip BEFORE scaling so calibration (which percentile-
        # fits ``raw_reward``) and the agent both see the same clipped
        # distribution.  See ``self._reward_clip_min/max`` rationale.
        if (self._reward_clip_min > -1e17) or (self._reward_clip_max < 1e17):
            clipped = float(np.clip(raw_reward,
                                     self._reward_clip_min,
                                     self._reward_clip_max))
            if (clipped != raw_reward) and (not self._reward_clip_warned):
                print(f'[env.step] WARNING reward clip triggered: raw={raw_reward:.4g} '
                      f'-> {clipped:.4g} (range [{self._reward_clip_min:.4g},'
                      f'{self._reward_clip_max:.4g}]); further occurrences '
                      f'silenced.', flush=True)
                self._reward_clip_warned = True
            raw_reward = clipped
        if self._bound_reward:
            # P77: scale-invariant linear remap of the (already tanh-bounded)
            # economic reward into [-B, B].  ``reward_clip_ref`` is the
            # econ-derived adaptive clip from objective_runtime; dividing by
            # it cancels the absolute magnitude of the user's economic
            # weights so the actor sees the SAME reward shape across every
            # simulator and economic configuration.  Single symlog happens
            # only at the twohot reward head (no double-symlog squash).
            b = self._bound_reward_max
            ref = float(comps.get('reward_clip', self._bound_reward_ref))
            if not np.isfinite(ref) or ref <= 1e-9:
                ref = self._bound_reward_ref_fallback
            self._bound_reward_ref = ref
            reward = float(np.clip(raw_reward * (b / ref), -b, b))
        else:
            reward = raw_reward * float(self.reward_scale)
        # A' : dense potential-based reward shaping (training env only;
        # ``info['raw_reward']`` below stays the unshaped economic reward so
        # validation scoring is unaffected).  F = coef·(γΦ(s') − Φ(s)) is
        # policy-invariant (Ng et al. 1999) — it densifies the learning
        # signal toward the band interior without changing the optimal
        # economic policy.  Added in scaled-reward units.
        if self._shaping_enabled and self._shaping_coef > 0.0:
            phi_next = self._shaping_potential(next_state)
            phi_prev = (self._prev_potential
                        if self._prev_potential is not None else phi_next)
            # Read γ live from cfg so the shaping discount tracks the
            # auto-tuned training γ (set AFTER __init__).  Strict
            # policy-invariance (Ng et al. 1999) requires shaping γ ==
            # RL γ; caching the __init__ default would break it.
            shaping_gamma = float(getattr(self.cfg, 'gamma',
                                          self._shaping_gamma)
                                  or self._shaping_gamma)
            shaping = self._shaping_coef * (
                shaping_gamma * phi_next - phi_prev)
            self._prev_potential = phi_next
            reward = reward + float(shaping)
            if self._bound_reward:
                # Keep the shaped reward inside the bounded envelope so the
                # densifier cannot reintroduce an unbounded scale.
                b = self._bound_reward_max
                reward = float(np.clip(reward, -b, b))
        self._prev_control = np.asarray(control, dtype='float32')
        # Stash raw per-channel violation depths for next step's
        # derivative (violation-rate) term (off unless its coef > 0).
        self._prev_mv_violation_per_channel = list(
            comps.get('mv_violation_per_channel_raw', []) or [])
        self._prev_cv_violation_per_channel = list(
            comps.get('cv_violation_per_channel_raw', []) or [])
        # Update the integral accumulator from the reward engine's
        # anti-windup-clamped value BEFORE building the obs so the agent
        # observes I_t (state augmentation keeps the integral Markov).
        if self._integral_enabled:
            intg = comps.get('integral_cv_per_channel', None)
            if intg is not None:
                self._integral_cv = np.asarray(intg, dtype='float32').reshape(-1)
        self._t += 1
        done = self._t >= self.cfg.episode_length
        obs_vec = self._build_obs_vec(next_state)
        self._window = np.concatenate([self._window[1:], obs_vec[None, :]], axis=0)
        self._last_cv_violation_sum += float(comps.get('cv_violation_penalty', 0.0))
        self._last_mv_violation_sum += float(comps.get('mv_violation_penalty', 0.0))
        info = {'reward_components': comps, 't': self._t,
                'raw_reward': raw_reward,
                'hidden_disturbance': hidden_applied,
                'raw_state': np.asarray(next_state, dtype='float32').copy()}
        return self._window.copy(), reward, done, info


# ---------------------------------------------------------------------------
# Replay buffer — episode-major
# ---------------------------------------------------------------------------

class TrajectoryBuffer:
    def __init__(self, capacity_eps: int, episode_length: int,
                 obs_dim: int, action_dim: int,
                 lookback: int = 0, n_dist: int = 0):
        # ``lookback`` is accepted for backward-compat with callers that
        # still pass it but is no longer used — the replay buffer stores
        # per-step observations ``(N, T, D)`` rather than per-step
        # windows ``(N, T, L, D)``.  The dropped L dimension was dead
        # weight: the world-model loss already read only ``obs[..., -1,
        # :]`` (the current frame) and the inference rollout maintains
        # its own sliding window from live env returns.  Paper alignment:
        # DreamerV3/V4 Algorithm 1 stores trajectories as sequences of
        # single-frame transitions, not stacked windows.  See discussion
        # 2026-05-24 (Phase 2 “unify lookback := seq_len”).
        del lookback
        self.capacity_eps = int(capacity_eps)
        self.T = int(episode_length)
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.obs = np.zeros((capacity_eps, self.T, self.obs_dim),
                            dtype='float32')
        self.act = np.zeros((capacity_eps, self.T, self.action_dim),
                            dtype='float32')
        self.rew = np.zeros((capacity_eps, self.T), dtype='float32')
        self.cont = np.ones((capacity_eps, self.T), dtype='float32')
        # Per-step flag marking transitions produced by the APC expert.  BC is
        # applied ONLY where this is 1.0 (avoids the documented "clone the
        # whole replay buffer → policy collapse" trap).
        self.expert = np.zeros((capacity_eps, self.T), dtype='float32')
        # Per-step hidden-disturbance target for the WM disturbance-estimator
        # head (P87).  Allocated only when ``n_dist > 0``; otherwise the
        # buffer is byte-identical to the pre-P87 layout.  ``_dist_source``
        # (bound to the single APCEnv) supplies each episode's recorded trace
        # so the ~10 add_episode call sites need no edits.
        self.n_dist = int(n_dist)
        self.dist = (np.zeros((capacity_eps, self.T, self.n_dist), dtype='float32')
                     if self.n_dist > 0 else None)
        self._dist_source = None
        self.filled = 0
        self.write = 0

    def add_episode(self, obs: np.ndarray, act: np.ndarray, rew: np.ndarray,
                    cont: np.ndarray, expert: Optional[np.ndarray] = None,
                    dist: Optional[np.ndarray] = None) -> None:
        T = obs.shape[0]
        assert T == self.T, f"episode length mismatch: {T} vs {self.T}"
        assert obs.ndim == 2 and obs.shape[1] == self.obs_dim, (
            f"expected obs shape (T, D)=(T, {self.obs_dim}); got {obs.shape}")
        i = self.write
        self.obs[i] = obs
        self.act[i] = act
        self.rew[i] = rew
        self.cont[i] = cont
        self.expert[i] = (expert if expert is not None
                          else np.zeros(self.T, dtype='float32'))
        if self.dist is not None:
            if dist is None and self._dist_source is not None:
                try:
                    dist = self._dist_source.pop_episode_disturbance(self.T)
                except Exception:
                    dist = None
            if dist is not None:
                d = np.asarray(dist, dtype='float32')
                if d.shape == self.dist.shape[1:]:
                    self.dist[i] = d
                else:
                    self.dist[i] = 0.0
            else:
                self.dist[i] = 0.0
        self.write = (self.write + 1) % self.capacity_eps
        self.filled = min(self.filled + 1, self.capacity_eps)

    def sample(self, batch_size: int, seq_len: int, rng: np.random.Generator
               ) -> Dict[str, np.ndarray]:
        if self.filled == 0:
            raise ValueError("empty buffer")
        ep_idx = rng.integers(0, self.filled, size=batch_size)
        max_start = self.T - seq_len
        if max_start <= 0:
            starts = np.zeros(batch_size, dtype=np.int64)
        else:
            starts = rng.integers(0, max_start + 1, size=batch_size)
        out_obs = np.zeros((batch_size, seq_len, self.obs_dim),
                           dtype='float32')
        out_act = np.zeros((batch_size, seq_len, self.action_dim),
                           dtype='float32')
        out_rew = np.zeros((batch_size, seq_len), dtype='float32')
        out_cont = np.zeros((batch_size, seq_len), dtype='float32')
        out_expert = np.zeros((batch_size, seq_len), dtype='float32')
        out_dist = (np.zeros((batch_size, seq_len, self.n_dist), dtype='float32')
                    if self.dist is not None else None)
        for b in range(batch_size):
            s = starts[b]
            out_obs[b] = self.obs[ep_idx[b], s:s + seq_len]
            out_act[b] = self.act[ep_idx[b], s:s + seq_len]
            out_rew[b] = self.rew[ep_idx[b], s:s + seq_len]
            out_cont[b] = self.cont[ep_idx[b], s:s + seq_len]
            out_expert[b] = self.expert[ep_idx[b], s:s + seq_len]
            if out_dist is not None:
                out_dist[b] = self.dist[ep_idx[b], s:s + seq_len]
        out = {'obs': out_obs, 'act': out_act, 'rew': out_rew,
               'cont': out_cont, 'expert': out_expert}
        if out_dist is not None:
            out['dist'] = out_dist
        return out


# ---------------------------------------------------------------------------
# Episode collection (V4 streaming inference)
# ---------------------------------------------------------------------------

@torch.no_grad()
def collect_episode(env: APCEnv, model: DreamerV4, device: torch.device,
                    cfg: TrainConfig, *, random_action: bool = False,
                    deterministic: bool = False) -> Dict[str, np.ndarray]:
    """Collect one episode using V4 model for inference.

    Streaming inference path (per timestep):
      1. Encode the last ``lookback`` observations through the tokenizer
         → ``z_ctx`` (clean past latents).
      2. Run the dynamics transformer over ``(z_ctx, a_ctx)`` with τ=1−τ_ctx
         (slight context noise) and d=d_min; read the agent-register
         hidden state at the latest time slot.
      3. Sample (or argmax) the action from the policy head.
    """
    obs_window = env.reset(exploration=random_action)
    T, L, D = cfg.episode_length, cfg.lookback, env.obs_dim
    # Phase 2 (2026-05-24): replay storage is per-step ``(T, D)`` only;
    # the L-length sliding window persists in ``obs_window`` for the
    # encoder/dynamics call but is no longer copied into the episode.
    obs_buf = np.zeros((T, D), dtype='float32')
    act_buf = np.zeros((T, env.action_dim), dtype='float32')
    rew_buf = np.zeros(T, dtype='float32')
    cont_buf = np.ones(T, dtype='float32')

    a_history = np.zeros((L, env.action_dim), dtype='float32')
    d_min = 1.0 / cfg.k_max
    # Past τ must land in the trained grid.  sample_tau_d emits
    # τ ∈ {0, 1/k, …, (k-1)/k} for k ≤ k_max, so the maximum trained
    # τ value is (k_max-1)/k_max.  Using cfg.tau_ctx=0.1 (τ=0.9) with
    # k_max=4 (max trained τ=0.75) is OOD → dynamics output garbage.
    tau_ctx_val = 1.0 - max(float(cfg.tau_ctx), 1.0 / float(cfg.k_max))

    _is_rssm = getattr(model, 'world_model_type', 'sf_transformer') in ('rssm', 'tssm')
    # RSSM streaming inference: carry a running recurrent state across the
    # episode.  Each step advances the posterior with (prev_action, obs)
    # and the posterior feature drives the policy — the GRU holds the
    # plant context, so the lookback-window encode is not needed.
    # ('tssm' uses the same interface; its state carries the token window.)
    #
    # Collection is pure inference, so the whole rollout runs under
    # ``torch.inference_mode()``: no autograd graph is built (lower memory
    # and lower per-step overhead than ``no_grad`` on the bs=1 streaming
    # path, which is the collection bottleneck).  The RSSM recurrent state
    # is kept ON-DEVICE across steps and the previous action is reused
    # directly from the policy's on-device output instead of round-tripping
    # host→device every step — that removes one transfer per step.  The
    # single device→host ``.cpu()`` of the chosen action is unavoidable
    # because the simulator runs on CPU/numpy.
    with torch.inference_mode():
        _rssm_state = (model.dynamics.initial_state(1, device)
                       if _is_rssm else None)
        _rssm_prev_a = (torch.zeros(1, env.action_dim, device=device)
                        if _is_rssm else None)

        for t in range(T):
            obs_buf[t] = obs_window[-1]
            if random_action:
                a_np = env.rng.uniform(-1.0, 1.0,
                                        size=(env.action_dim,)).astype('float32')
                if _is_rssm:
                    # Advance the posterior even on random actions so the
                    # recurrent state stays consistent if a later step is
                    # policy-driven (mixed-mode collection).
                    with torch.amp.autocast(device_type=device.type,
                                             dtype=torch.bfloat16,
                                             enabled=(device.type == 'cuda')):
                        _o = torch.from_numpy(
                            obs_window[-1]).to(device).unsqueeze(0)
                        _emb = model.dynamics.embed(_o)
                        _post, _ = model.dynamics.obs_step(
                            _rssm_state, _rssm_prev_a, _emb, sample=True)
                    _rssm_state = _post
                    # Random action originates on host; the (1, A) host→device
                    # copy is negligible and necessary (no device tensor yet).
                    _rssm_prev_a = torch.from_numpy(
                        a_np).to(device).unsqueeze(0)
            elif _is_rssm:
                with torch.amp.autocast(device_type=device.type,
                                         dtype=torch.bfloat16,
                                         enabled=(device.type == 'cuda')):
                    _o = torch.from_numpy(obs_window[-1]).to(device).unsqueeze(0)
                    _emb = model.dynamics.embed(_o)
                    # mbrl2 real-sim: advance the posterior with the MODE
                    # (sample=False) when the POLICY drives.  The actor is TRAINED
                    # on the mode belief (``_realsim_actor_critic_step`` re-encodes
                    # with sample=False) and LQG separation puts control on the
                    # OPTIMAL STATE ESTIMATE — so acting on a SAMPLED belief here is
                    # a train/inference mismatch that injects latent-sampling noise
                    # into the MV (part of the p01 chatter, esp. at deterministic
                    # eval).  Random-exploration collection above keeps sample=True
                    # (diverse latents for WM training).
                    _post, _ = model.dynamics.obs_step(
                        _rssm_state, _rssm_prev_a, _emb, sample=False)
                    action_t, _, _ = model.policy(_post.feat,
                                                   deterministic=deterministic)
                a_np = action_t.float().squeeze(0).cpu().numpy().astype('float32')
                _rssm_state = _post
                # Reuse the on-device action for the next GRU step — no
                # host→device round-trip (the .cpu() above already paid the
                # only unavoidable sync for env.step).
                _rssm_prev_a = action_t.detach().float().reshape(1, -1)
            else:
                ow = torch.from_numpy(obs_window).to(device)            # (L, D)
                a_ctx = torch.from_numpy(a_history).to(device)           # (L, A)
                with torch.amp.autocast(device_type=device.type,
                                         dtype=torch.bfloat16,
                                         enabled=(device.type == 'cuda')):
                    z_ctx = model.tokenizer.encode(ow).unsqueeze(0)     # (1, L, z)
                    tau = torch.full((1, L), tau_ctx_val, device=device,
                                      dtype=z_ctx.dtype)
                    d = torch.full((1, L), d_min, device=device,
                                    dtype=z_ctx.dtype)
                    out = model.dynamics(z_ctx, tau, d, a_ctx.unsqueeze(0))
                    agent_hid = out['agent_hid'][:, -1]                  # (1, D)
                    action_t, _, _ = model.policy(agent_hid,
                                                    deterministic=deterministic)
                a_np = action_t.float().squeeze(0).cpu().numpy().astype('float32')
            next_window, reward, done, _ = env.step(a_np)
            act_buf[t] = a_np
            rew_buf[t] = reward
            cont_buf[t] = 0.0 if done and t == T - 1 else 1.0
            a_history = np.concatenate([a_history[1:], a_np[None, :]], axis=0)
            obs_window = next_window
            if done:
                break
    return {'obs': obs_buf, 'act': act_buf, 'rew': rew_buf, 'cont': cont_buf}


def collect_baseline_episode(env: APCEnv, cfg: TrainConfig, *,
                              action_std: float = 0.05,
                              center: float = 0.0,
                              ) -> Dict[str, np.ndarray]:
    """Collect one episode driven by small-noise actions around ``center``.

    Used to **seed** the replay buffer with non-catastrophic transitions
    before the actor has learned anything (P0 cold-start fix,
    2026-05-05; P43 stratified-center upgrade, 2026-05-23).  Actions
    are drawn from ``N(center, action_std)`` clipped to ``[-1, 1]`` in
    the env's normalized action space.  ``center=0.0`` reproduces the
    legacy mid-MV-hold behaviour; ``center`` stratified across
    ``[-op_band, +op_band]`` per episode (driven by the P1 loop in
    ``train_dreamer_v4``) gives the WM operating-point coverage
    matching PRBS, but with smooth held trajectories that let the
    steady-state reach equilibrium at each centre.

    Centre is sampled per episode upstream, not within the episode, so
    each episode is a clean small-noise hold at one stratum.  This is
    sim-agnostic: no plant-specific code lives here.

    Returns the same dict shape as ``collect_episode`` so the result can
    be passed straight to ``buf.add_episode``.
    """
    obs_window = env.reset(exploration=True)
    T, L, D = cfg.episode_length, cfg.lookback, env.obs_dim
    obs_buf = np.zeros((T, D), dtype='float32')
    act_buf = np.zeros((T, env.action_dim), dtype='float32')
    rew_buf = np.zeros(T, dtype='float32')
    cont_buf = np.ones(T, dtype='float32')
    for t in range(T):
        obs_buf[t] = obs_window[-1]
        a_np = env.rng.normal(float(center), float(action_std),
                                size=(env.action_dim,)).astype('float32')
        np.clip(a_np, -1.0, 1.0, out=a_np)
        next_window, reward, done, _ = env.step(a_np)
        act_buf[t] = a_np
        rew_buf[t] = reward
        cont_buf[t] = 0.0 if done and t == T - 1 else 1.0
        obs_window = next_window
        if done:
            break
    return {'obs': obs_buf, 'act': act_buf, 'rew': rew_buf, 'cont': cont_buf}


def collect_prbs_episode(env: APCEnv, cfg: TrainConfig, *,
                          action_std: float = 0.05,
                          n_segments: Optional[int] = None,
                          op_band: float = 0.95,
                          n_strata: Optional[int] = None,
                          ) -> Dict[str, np.ndarray]:
    """Collect one episode that PRBS-toggles MV across the operating band.

    Why this exists (2026-05-08, run_p7 RCA): the WM was systematically
    under-trained because every seed episode held MV near mid-bound (the
    ``collect_baseline_episode`` operating point).  Result: WM next-state
    correlation r=0.638 (marginal), critic r=0.443 (marginal), and a
    deterministic actor that only knows how to hold MV near 50.

    PRBS sweep produces a buffer whose state distribution covers the
    full ``[-op_band, +op_band]`` MV operating range.  Within a segment,
    the action is held at a constant target (drawn from a stratified
    sample over ``[-op_band, +op_band]``) plus ``N(0, action_std)``
    noise.  Segment length is auto-derived in ``auto_tune_seed_buffer``
    to ``(θ + 4τ)/sr`` so the plant settles at each operating point
    before toggling — gives the WM clean step-response transitions
    across the full range, including near boundaries (which uniform
    sampling under-covers when #segments per episode is small).

    Returns the same dict shape as ``collect_episode``.
    """
    obs_window = env.reset(exploration=True)
    T, L, D = cfg.episode_length, cfg.lookback, env.obs_dim
    obs_buf = np.zeros((T, D), dtype='float32')
    act_buf = np.zeros((T, env.action_dim), dtype='float32')
    rew_buf = np.zeros(T, dtype='float32')
    cont_buf = np.ones(T, dtype='float32')
    # Segment length: prefer cfg-supplied (auto-derived from plant
    # timing in auto_tune_seed_buffer ⇒ (θ + 4τ)/sr ≈ 98% settling
    # time).  Fall back to env-var SIM_IDENTIFIED_TAU_DOMINANT for
    # back-compat (old runs) and finally to a generous T/12 default
    # (only triggers when neither plant timing nor cfg is available).
    seg_cfg = int(getattr(cfg, 'prbs_seed_segment_steps', 0) or 0)
    if seg_cfg > 0:
        seg_max = max(8, min(seg_cfg, T // 4))
    else:
        sr = max(1, int(getattr(cfg, 'sample_rate', 1)))
        tau_dom_env = float(os.environ.get(
            'SIM_IDENTIFIED_TAU_DOMINANT', '0') or 0)
        if tau_dom_env > 0:
            seg_max = max(8, int(round(4.0 * tau_dom_env / sr)))
            seg_max = min(seg_max, T // 4)
        else:
            seg_max = max(8, T // 12)
    # Multi-timescale GBN (run_p12 RCA): when a fast-hold floor is
    # configured, sample each segment's hold log-uniformly in
    # [seg_min, seg_max] so the WM gets BOTH transient (~τ/3) and
    # steady-state (~4τ) excitation.  Single-timescale long PRBS
    # leaves the WM compounding-error blind.
    seg_min_cfg = int(getattr(cfg, 'prbs_seed_segment_steps_min', 0) or 0)
    seg_min = max(2, min(seg_min_cfg, seg_max - 1)) if seg_min_cfg > 1 else seg_max
    multi_timescale = (seg_min < seg_max)
    if multi_timescale:
        # Pre-roll segment lengths log-uniformly; expand episode in
        # ``draw_seg_lens`` so total covered steps >= T (last truncated).
        log_lo = float(np.log(max(1, seg_min)))
        log_hi = float(np.log(max(seg_min + 1, seg_max)))
        seg_lens = []
        covered = 0
        # Generous upper bound on segment count.
        max_segs_guess = int(np.ceil(T / max(1, seg_min))) + 4
        draws = env.rng.uniform(log_lo, log_hi, size=max_segs_guess)
        for u in draws:
            sl = int(round(float(np.exp(u))))
            sl = max(seg_min, min(seg_max, sl))
            seg_lens.append(sl)
            covered += sl
            if covered >= T:
                break
        seg_lens = np.asarray(seg_lens, dtype='int32')
        seg_starts = np.concatenate(([0], np.cumsum(seg_lens)[:-1])).astype('int32')
        n_seg_int = int(len(seg_lens))
        if n_segments is not None:
            # Caller-supplied count overrides only the count semantics;
            # we still keep the multi-timescale lengths.
            pass
    else:
        seg_len_uniform = seg_max
        if n_segments is None:
            n_segments = max(2, int(np.ceil(T / seg_len_uniform)))
        n_seg_int = int(n_segments)
        seg_lens = np.full(n_seg_int, seg_len_uniform, dtype='int32')
        seg_starts = (np.arange(n_seg_int) * seg_len_uniform).astype('int32')
    # Pre-roll PRBS targets per segment, per action dim.
    # Use stratified sampling to guarantee boundary coverage even when
    # #segments per episode is small: divide [-op, +op] into N strata
    # and place at least one center per stratum, then fill the
    # remaining segments with random draws.  Without stratification,
    # the outer 10 % of the MV range is statistically under-covered
    # — exactly the region where the controller most needs accurate
    # WM predictions when defending against disturbances.
    op = float(np.clip(op_band, 0.05, 0.95))
    A = int(env.action_dim)
    strata_n = (int(n_strata) if n_strata is not None
                  else int(getattr(cfg, 'prbs_seed_n_strata', 8)))
    strata_n = max(0, min(strata_n, n_seg_int))
    targets = np.empty((n_seg_int, A), dtype='float32')
    if strata_n > 0:
        # Stratum centers: midpoints of equal-width bins over [-op, +op].
        edges = np.linspace(-op, +op, strata_n + 1)
        centers = 0.5 * (edges[:-1] + edges[1:])
        for a in range(A):
            order = env.rng.permutation(n_seg_int)
            assigned = np.empty(n_seg_int, dtype='float32')
            # First strata_n positions: one per stratum, jittered within
            # the stratum to avoid identical center repeats.
            half_w = (op / strata_n)
            jitter = env.rng.uniform(-half_w, +half_w,
                                        size=strata_n).astype('float32')
            assigned[:strata_n] = (centers + jitter).astype('float32')
            # Remaining positions: uniform random draws.
            if n_seg_int > strata_n:
                assigned[strata_n:] = env.rng.uniform(
                    -op, +op, size=n_seg_int - strata_n
                    ).astype('float32')
            # Shuffle so stratum-anchored segments are not all up front
            # (we want operating-point variety across episode time too).
            targets[:, a] = assigned[order]
    else:
        targets = env.rng.uniform(-op, +op,
                                    size=(n_seg_int, A)
                                    ).astype('float32')
    # Build per-step segment-index map from variable seg_lens.
    seg_index_for_t = np.zeros(T, dtype='int32')
    for k in range(n_seg_int):
        s = int(seg_starts[k])
        e = min(T, s + int(seg_lens[k]))
        if s >= T:
            break
        seg_index_for_t[s:e] = k
    for t in range(T):
        obs_buf[t] = obs_window[-1]
        seg_idx = int(seg_index_for_t[t])
        center = targets[seg_idx]
        noise = env.rng.normal(0.0, float(action_std),
                                 size=(env.action_dim,)).astype('float32')
        a_np = (center + noise).astype('float32')
        np.clip(a_np, -1.0, 1.0, out=a_np)
        next_window, reward, done, _ = env.step(a_np)
        act_buf[t] = a_np
        rew_buf[t] = reward
        cont_buf[t] = 0.0 if done and t == T - 1 else 1.0
        obs_window = next_window
        if done:
            break
    return {'obs': obs_buf, 'act': act_buf, 'rew': rew_buf, 'cont': cont_buf}


def collect_constant_action_episode(env: APCEnv, cfg: TrainConfig, *,
                                      action_level: float,
                                      ) -> Dict[str, np.ndarray]:
    """Collect one episode driven by a single, perfectly constant action.

    Used to seed the buffer with **long-horizon constant-action steady
    states** (2026-05-21, p31 RCA).  PRBS seed episodes only hold an
    action for one segment (~80\u2013150 agent steps); the WM then never sees
    a settled response over the full ``episode_length`` horizon and
    extrapolates badly when the trained controller later sits near a
    setpoint and holds the MV.  Holding the action constant for the full
    episode forces the WM to learn the long-tail steady-state of the
    plant at this operating point.

    The curriculum disturbance schedule is **suppressed** for these
    episodes (``env._schedule = []``) so the WM sees a clean settled
    response — a CV/DV step partway through would defeat the purpose.
    Domain randomization, OU process noise, and measurement noise remain
    active (DR fires in ``sim.reset()`` upstream of the schedule build,
    so it is untouched; OU/measurement live in the ``SimNoiseWrapper``
    and are independent of ``_schedule``).  This is sim-agnostic:
    ``APCEnv._schedule`` is the single hook used by every simulator.

    ``action_level`` is in the env's normalized action space and is
    clipped to ``[-1, 1]``.  Returns the same dict shape as
    ``collect_episode``.
    """
    obs_window = env.reset(exploration=True)
    # Clear curriculum disturbance schedule so this seed is a clean
    # held-action steady-state probe.  DR + OU + measurement noise stay
    # active EXCEPT the hidden (truly-unmeasured) OU disturbance, which
    # would corrupt the steady-state target the WM is meant to learn
    # from this episode (P43, 2026-05-23 audit finding: const_action
    # was firing hidden OU in 12.5 % of episodes, contaminating the SS
    # signal).  Sim-agnostic: ``_hidden_disturbance`` lives on APCEnv.
    env._schedule = []
    env._hidden_disturbance = None
    # P89 Fix A: make this held-action steady-state seed fully NOISE-FREE
    # (disable process OU + measurement noise) so the WM gets pure
    # fixed-point supervision.  The P89 RCA showed the WM never learned a
    # held-action fixed point because EVERY training trajectory (incl. these
    # seeds) carried persistent OU + measurement noise so the plant never
    # truly settled.  Off-switch: DREAMER_CLEAN_STEADY_SEEDS=0.
    if bool(getattr(cfg, 'clean_steady_seeds', True)):
        env.set_sim_noise_scale(0.0)
    T, L, D = cfg.episode_length, cfg.lookback, env.obs_dim
    obs_buf = np.zeros((T, D), dtype='float32')
    act_buf = np.zeros((T, env.action_dim), dtype='float32')
    rew_buf = np.zeros(T, dtype='float32')
    cont_buf = np.ones(T, dtype='float32')
    a_const = np.full((env.action_dim,),
                       float(np.clip(action_level, -1.0, 1.0)),
                       dtype='float32')
    for t in range(T):
        obs_buf[t] = obs_window[-1]
        next_window, reward, done, _ = env.step(a_const)
        act_buf[t] = a_const
        rew_buf[t] = reward
        cont_buf[t] = 0.0 if done and t == T - 1 else 1.0
        obs_window = next_window
        if done:
            break
    return {'obs': obs_buf, 'act': act_buf, 'rew': rew_buf, 'cont': cont_buf}


def collect_expert_episode(env: APCEnv, cfg: TrainConfig, *,
                            expert,
                            keep_schedule: bool = True,
                            action_jitter: float = 0.0,
                            rng: Optional[np.random.Generator] = None,
                            ) -> Dict[str, np.ndarray]:
    """Collect one episode driven by the APC steady-state expert.

    The expert (``utils.apc_expert.SteadyStateExpert``) emits an engineering
    MV target each step; we map it to the actor's normalised ``[-1, 1]`` space
    via ``control_to_action`` (the exact inverse of ``env.step``'s action
    mapping over the same fixed band — physical envelope or base bounds — so
    the round-trip is exact) and feed that to ``env.step``.  CV/DV feedback is
    read in engineering units from
    ``info['raw_state']`` via the simulator metadata denormaliser.

    Every transition is flagged ``expert=1`` so the BC loss anchors the policy
    mean ONLY on these steps.  The curriculum disturbance schedule is kept on by
    default (``keep_schedule=True``) so the expert demonstrates good economic +
    constraint behaviour under realistic disturbances; ``action_jitter`` adds a
    small Gaussian exploration around the expert action so BC sees a tube, not a
    razor-thin trajectory.  Returns the same dict shape as ``collect_episode``
    plus an ``expert`` mask channel.
    """
    from utils.dynamics_identifier import _state_value_to_engineering

    obs_window = env.reset(exploration=True)
    if not keep_schedule:
        env._schedule = []
        env._hidden_disturbance = None
    if rng is None:
        rng = getattr(env, 'rng', np.random.default_rng())

    meta = env.meta
    cv_idx = list(env.cv_indices)
    dv_idx = [int(x) for x in meta.get('dv_indices', []) if x is not None]
    cv_bounds_eu = np.asarray(env.cv_bounds_eu, dtype='float64')

    T, D = cfg.episode_length, env.obs_dim
    obs_buf = np.zeros((T, D), dtype='float32')
    act_buf = np.zeros((T, env.action_dim), dtype='float32')
    rew_buf = np.zeros(T, dtype='float32')
    cont_buf = np.ones(T, dtype='float32')
    exp_buf = np.zeros(T, dtype='float32')

    expert.reset()
    # Causal init: first action uses the band-midpoint CV (no feedback yet).
    last_cv = cv_bounds_eu.mean(axis=1) if cv_bounds_eu.size else np.zeros(len(cv_idx))
    last_dv = None

    for t in range(T):
        obs_buf[t] = obs_window[-1]
        u_eu = expert.step_eu(last_cv, cv_bounds=cv_bounds_eu, dv_eu=last_dv)
        # Invert through the SAME mapping basis ``env.step`` uses (physical
        # envelope when active, else base bounds), so the BC target round-trips
        # exactly regardless of mapping basis (P86).
        a_norm = control_to_action(u_eu, env.bounds).astype('float32').reshape(-1)
        if action_jitter > 0.0:
            a_norm = np.clip(
                a_norm + rng.normal(0.0, action_jitter, size=a_norm.shape),
                -1.0, 1.0).astype('float32')
        next_window, reward, done, info = env.step(a_norm)
        rs = info.get('raw_state')
        if rs is not None:
            rs = np.asarray(rs, dtype='float64')
            last_cv = np.asarray(
                [_state_value_to_engineering(rs, ci, meta) for ci in cv_idx],
                dtype='float64')
            if dv_idx:
                last_dv = np.asarray(
                    [_state_value_to_engineering(rs, di, meta) for di in dv_idx],
                    dtype='float64')
        act_buf[t] = a_norm
        rew_buf[t] = reward
        cont_buf[t] = 0.0 if done and t == T - 1 else 1.0
        exp_buf[t] = 1.0
        obs_window = next_window
        if done:
            break
    return {'obs': obs_buf, 'act': act_buf, 'rew': rew_buf,
            'cont': cont_buf, 'expert': exp_buf}


def collect_step_settle_episode(env: APCEnv, cfg: TrainConfig, *,
                                  action_start: float,
                                  action_end: float,
                                  switch_step: int,
                                  ) -> Dict[str, np.ndarray]:
    """Collect a step-and-settle episode for WM seeding (P42 design).

    Hold ``action_start`` for the first ``switch_step`` agent steps,
    then step to ``action_end`` and hold to episode end.  Strict
    superset of ``collect_constant_action_episode``:

      [0, switch_step)        — pre-step settling from random IC to SS(u₀).
      [switch_step]           — clean step response with known timing.
      [switch_step, T)        — long-horizon SS at the new operating point.

    Compared to a constant-action episode, this provides:
      - explicit transient with known excitation timing → direct gain +
        time-constant supervision for the WM,
      - same late-episode SS supervision as constant-action (over a
        shorter but still >>τ tail),
      - breaks the "predict no-change under constant input" shortcut
        because between ``switch_step`` and ~``switch_step + 5τ`` the
        plant changes despite the action being held.

    Curriculum disturbance schedule is suppressed (``env._schedule = []``)
    so the step's effect is unambiguous; DR + OU + measurement noise
    remain active.  Sim-agnostic via ``APCEnv._schedule``.

    Both ``action_start`` and ``action_end`` are clipped to ``[-1, 1]``.
    Returns the same dict shape as ``collect_episode``.
    """
    obs_window = env.reset(exploration=True)
    env._schedule = []
    # P43 (2026-05-23): also suppress hidden OU disturbance so the step
    # transient is unambiguous (see ``collect_constant_action_episode``
    # rationale).  Sim-agnostic.
    env._hidden_disturbance = None
    # P89 Fix A: noise-free so the step transient + settled tail give the WM
    # clean gain + fixed-point supervision (DREAMER_CLEAN_STEADY_SEEDS=0 off).
    if bool(getattr(cfg, 'clean_steady_seeds', True)):
        env.set_sim_noise_scale(0.0)
    T, L, D = cfg.episode_length, cfg.lookback, env.obs_dim
    obs_buf = np.zeros((T, D), dtype='float32')
    act_buf = np.zeros((T, env.action_dim), dtype='float32')
    rew_buf = np.zeros(T, dtype='float32')
    cont_buf = np.ones(T, dtype='float32')
    a_start = np.full((env.action_dim,),
                       float(np.clip(action_start, -1.0, 1.0)),
                       dtype='float32')
    a_end = np.full((env.action_dim,),
                     float(np.clip(action_end, -1.0, 1.0)),
                     dtype='float32')
    sw = int(np.clip(switch_step, 1, T - 1))
    for t in range(T):
        a_t = a_start if t < sw else a_end
        obs_buf[t] = obs_window[-1]
        next_window, reward, done, _ = env.step(a_t)
        act_buf[t] = a_t
        rew_buf[t] = reward
        cont_buf[t] = 0.0 if done and t == T - 1 else 1.0
        obs_window = next_window
        if done:
            break
    return {'obs': obs_buf, 'act': act_buf, 'rew': rew_buf, 'cont': cont_buf}


def _sample_step_settle_params(rng: np.random.Generator, cfg: TrainConfig,
                                  u0: float) -> Tuple[float, int]:
    """Sample ``(u1, switch_step)`` for a step-and-settle episode.

    ``u1`` is ``u0 + Δ`` with ``|Δ| ∈ [step_seed_delta_min, max]`` and
    a random sign; then clipped to ``[-1, 1]``.  If the clip would
    shrink the step below ``step_seed_delta_min``, the sign is
    reversed before re-clipping (keeps the step magnitude meaningful
    even when ``u0`` is near the operating-band edge).

    ``switch_step`` is uniformly sampled from
    ``int(prefix_frac * episode_length)`` with
    ``prefix_frac ∈ [step_seed_prefix_frac_min, max]``.
    """
    d_min = float(cfg.step_seed_delta_min)
    d_max = float(cfg.step_seed_delta_max)
    if d_max < d_min:
        d_min, d_max = d_max, d_min
    mag = float(rng.uniform(d_min, d_max))
    sign = 1.0 if rng.uniform() < 0.5 else -1.0
    u1 = float(np.clip(u0 + sign * mag, -1.0, 1.0))
    if abs(u1 - u0) < d_min:
        u1 = float(np.clip(u0 - sign * mag, -1.0, 1.0))
    pf_min = float(cfg.step_seed_prefix_frac_min)
    pf_max = float(cfg.step_seed_prefix_frac_max)
    if pf_max < pf_min:
        pf_min, pf_max = pf_max, pf_min
    prefix_frac = float(rng.uniform(pf_min, pf_max))
    T = int(cfg.episode_length)
    switch_step = int(np.clip(round(prefix_frac * T), 1, T - 1))
    return u1, switch_step


def _seed_one_const_or_step(env: APCEnv, cfg: TrainConfig, *,
                              level: float, do_step: bool,
                              ) -> Dict[str, np.ndarray]:
    """Dispatch one constant-action OR step-and-settle seed episode.

    Used by the initial seed loop and the P1 periodic injection block
    so the const-vs-step allocation is identical in both code paths.
    """
    if do_step:
        u1, sw = _sample_step_settle_params(env.rng, cfg, float(level))
        return collect_step_settle_episode(env, cfg,
                                             action_start=float(level),
                                             action_end=u1,
                                             switch_step=sw)
    return collect_constant_action_episode(env, cfg,
                                             action_level=float(level))


def collect_step_test_episode(env: APCEnv, cfg: TrainConfig, *,
                                initial_level: float,
                                primary_dv_pos: int = -1,
                                ) -> Dict[str, np.ndarray]:
    """APC-style step-test seed episode (P51 design, 2026-05-25).

    Mixes MV step events (held action sequence) with DV step events
    (explicit ``_schedule``) in a SINGLE held-baseline episode.  The
    layout mirrors industrial APC step-testing:

      [0, t_settle)              — hold u₀ to clear IC transients.
      [t_settle, T)              — alternating MV/DV step events with
                                    mixed spacing:
        * ~(1 - overlap_frac)    — SETTLED: spacing ≥ 4·(τ+θ),
                                    so each gain is identifiable from
                                    a clean isolated step (the WM
                                    learns ∂CV/∂u_i and ∂CV/∂d_j in
                                    isolation).
        * ~overlap_frac           — OVERLAP: spacing ~0.5–1·(τ+θ),
                                    so the WM also sees coupled
                                    multi-channel transients (closer
                                    to real plant operation where
                                    DVs perturb during MV moves).

    Why this episode type (P49/P50 RCA, 2026-05-24/25):
      * The 60 cleanest seed episodes (40 constant-action + 20
        step-settle) had ZERO DV events because both forced
        ``env._schedule = []``.  Result: the WM never saw a clean
        ∂CV/∂DV under held action, and the actor never saw
        "DV stepped while I held" in-distribution — exactly the
        disturbance-rejection scenario the policy fails at.
      * Disturbance-active PRBS seeds DO contain DV events but the MV
        never holds, so DV gain is confounded with MV-induced motion
        (the WM cannot factor out which input caused which CV move).
      * Step-test is the strict superset: held baselines + isolated
        events of both types + a controllable coupling regime.

    Sim-adaptive (unitless thresholds, engineering values derived from
    the identifier):
      * Event spacing = multiple of ``dyn_horizon = τ_dom + dead_time``
        from the identifier context.
      * MV deltas reuse ``cfg.step_seed_delta_min/max`` (normalized
        action units).
      * DV deltas = ``uniform(0.10, 0.30) * (dv_hi - dv_lo)`` of the
        identifier-reported channel range (engineering units).
      * Hidden OU disturbance OFF — preserves a clean baseline so
        each event's response is unambiguous (P43 SS-fidelity logic).

    ``initial_level`` is in normalized action space; clipped to [-1, 1].
    Returns the same dict shape as ``collect_episode``.
    """
    from utils.training_disturbance import (_load_identifier_context,
                                              _channel_catalog)
    rng = env.rng
    T = int(cfg.episode_length)
    D = env.obs_dim

    # ----- Sim-adaptive timing from identifier --------------------------
    # ``tau_dominant_identified`` and ``dead_time_identified`` are stored
    # in AGENT steps by convention (see utils/training_disturbance.py
    # which uses ``dyn_horizon = tau_dom + dead_time`` directly as a
    # step count when building schedules).
    id_ctx = _load_identifier_context()
    dyn = id_ctx.get('dynamics', {}) if isinstance(id_ctx, dict) else {}
    tau_dom = float(dyn.get('tau_dominant_identified', 0.0) or 0.0)
    dead_time = float(dyn.get('dead_time_identified', 0.0) or 0.0)
    if tau_dom <= 0.0:
        # Fallback: spread events evenly across the episode.
        dyn_horizon_steps = max(8, T // 12)
    else:
        dyn_horizon_steps = max(8, int(round(tau_dom + dead_time)))
    # SETTLED gap = 4·(τ+θ) (one full settle + margin).
    # OVERLAP gap = uniform[0.5, 1.0] · (τ+θ) (events still overlap).
    settled_gap = max(2 * dyn_horizon_steps, 4 * dyn_horizon_steps)
    overlap_gap_lo = max(1, int(round(0.5 * dyn_horizon_steps)))
    overlap_gap_hi = max(overlap_gap_lo + 1, dyn_horizon_steps)

    # Initial settle = 3·dyn_horizon (or 1/8 of the episode, whichever
    # is smaller — keeps room for events on short episodes).
    t_settle = min(max(3 * dyn_horizon_steps, 8), T // 4)

    # ----- DV channel catalog -------------------------------------------
    obs_window = env.reset(exploration=True)
    env._schedule = []
    env._hidden_disturbance = None
    channels = _channel_catalog(env.sim)
    dv_chs = list(channels.get('dv', []))
    has_dv = len(dv_chs) > 0
    dv_share = float(np.clip(cfg.step_test_dv_share, 0.0, 1.0))
    overlap_frac = float(np.clip(cfg.step_test_overlap_frac, 0.0, 0.9))
    if not has_dv:
        dv_share = 0.0  # no DV channels → all events are MV

    # ----- Build event timeline -----------------------------------------
    # Walk forward from t_settle picking inter-event gaps from the
    # mixed {settled, overlap} distribution until we run out of room.
    event_times: List[Tuple[int, str]] = []   # (start_step, 'mv'|'dv')
    t = int(t_settle)
    while t < T - max(8, dyn_horizon_steps):
        if rng.random() < overlap_frac:
            gap = int(rng.integers(overlap_gap_lo, overlap_gap_hi + 1))
        else:
            gap = int(rng.integers(settled_gap,
                                     int(round(1.25 * settled_gap)) + 1))
        # Pick channel type.
        is_dv = (rng.random() < dv_share) and has_dv
        event_times.append((int(t), 'dv' if is_dv else 'mv'))
        t += gap

    # Guarantee at least one MV and one DV (when DVs exist) so the
    # episode is always informative.
    types = [k for _, k in event_times]
    if has_dv and 'dv' not in types and event_times:
        i = int(rng.integers(0, len(event_times)))
        event_times[i] = (event_times[i][0], 'dv')
    if 'mv' not in types and event_times:
        i = int(rng.integers(0, len(event_times)))
        event_times[i] = (event_times[i][0], 'mv')

    # ----- Build MV action timeline + DV schedule -----------------------
    u_min = float(cfg.step_seed_delta_min)
    u_max = float(cfg.step_seed_delta_max)
    if u_max < u_min:
        u_min, u_max = u_max, u_min
    u_band = float(getattr(cfg, 'constant_action_seed_op_band', 0.6))

    act_buf = np.zeros((T, env.action_dim), dtype='float32')
    cur_u = float(np.clip(initial_level, -1.0, 1.0))
    last_t = 0
    dv_schedule: List[Dict] = []
    mv_event_count = 0
    dv_event_count = 0
    for start, kind in event_times:
        # Fill the held segment up to this event.
        if start > last_t:
            act_buf[last_t:start, :] = cur_u
        if kind == 'mv':
            # Step the action: pick magnitude and a sign that keeps u in
            # the operating band when possible.
            mag = float(rng.uniform(u_min, u_max))
            sign = +1.0 if rng.random() < 0.5 else -1.0
            cand = cur_u + sign * mag
            if abs(cand) > u_band:
                sign = -sign   # flip toward the centre
                cand = cur_u + sign * mag
            cur_u = float(np.clip(cand, -1.0, 1.0))
            mv_event_count += 1
        else:
            # DV step: pick a channel and a magnitude in the channel's
            # engineering range (10–30 % of the span, random sign).
            # Stratified: a configurable fraction of DV events target
            # this episode's ``primary_dv_pos`` so each channel gets
            # balanced isolated-step coverage across the seed batch.
            use_primary = (
                0 <= int(primary_dv_pos) < len(dv_chs)
                and rng.random()
                    < float(cfg.step_test_primary_dv_bias))
            if use_primary:
                ch = dv_chs[int(primary_dv_pos)]
            else:
                ch = dv_chs[int(rng.integers(0, len(dv_chs)))]
            b = ch.get('bounds')
            if isinstance(b, list) and len(b) >= 2:
                span = float(b[1]) - float(b[0])
            else:
                span = 1.0
            mag = float(rng.uniform(0.10, 0.30)) * abs(span)
            sign = +1.0 if rng.random() < 0.5 else -1.0
            dv_schedule.append({
                'name': f"step_test_dv_{ch.get('name', ch.get('pos', '?'))}_"
                          f"t{int(start)}",
                'target_group': 'dv',
                'target_pos': int(ch.get('pos', 0)),
                'start': int(start),
                'duration': 1,
                'shape': 'step',
                'delta': float(sign * mag),
                'source': 'step_test_seed',
                '_applied': False,
            })
            dv_event_count += 1
        last_t = start
    # Fill the tail after the last event.
    if last_t < T:
        act_buf[last_t:T, :] = cur_u

    # Install the DV schedule (env.step → apply_disturbance_schedule).
    env._schedule = dv_schedule

    # ----- Run the episode ----------------------------------------------
    obs_buf = np.zeros((T, D), dtype='float32')
    rew_buf = np.zeros(T, dtype='float32')
    cont_buf = np.ones(T, dtype='float32')
    for t in range(T):
        obs_buf[t] = obs_window[-1]
        next_window, reward, done, _ = env.step(act_buf[t])
        rew_buf[t] = reward
        cont_buf[t] = 0.0 if done and t == T - 1 else 1.0
        obs_window = next_window
        if done:
            break
    return {'obs': obs_buf, 'act': act_buf, 'rew': rew_buf, 'cont': cont_buf}


def _build_dv_prbs_schedule(env: 'APCEnv', cfg: TrainConfig) -> List[Dict]:
    """Full-range, multi-timescale, stratified DV-PRBS disturbance schedule.

    Schedule-construction core shared by the DV-PRBS SEED episodes
    (``collect_dv_prbs_episode``) and the Stage-1 ON-POLICY DV excitation
    (R1a, p128) so both use identical excitation statistics.  Sweeps EVERY
    measured-DV channel independently across ±``dv_prbs_op_frac``·(span/2) at
    mixed timescales (seg_min … seg_max) via telescoping step deltas
    (``delta_k = L_k − L_{k−1}``) so the accumulated offset tracks the
    stratified level sequence.  Returns an empty list when the sim has no
    measured-DV channels (caller leaves the existing schedule untouched).

    Why (RC-W1, p119–p127 DV-gain RCA): the WM's DV→CV gain is identified on
    the SLOW on-policy FEED motion (≈0.29 std OU) plus only ~1-5 sparse step
    events/episode from ``build_training_disturbance_schedule`` — a ~30×
    deficit vs the MV's continuous full-range PRBS.  With ``dv_as_input`` +
    held-const-in-imagination the FEED→CV weight then gets almost no gradient
    (DV gain stuck ~0.76 across 7 runs).  Persistent, large-amplitude DV
    excitation kills both the under-excitation and the errors-in-variables
    regression dilution.
    """
    from utils.training_disturbance import _channel_catalog
    T = int(cfg.episode_length)
    dv_chs = list(_channel_catalog(env.sim).get('dv', []))
    if not dv_chs:
        return []
    # ----- Multi-timescale segment lengths (mirror collect_prbs_episode) -
    # seg_max ≈ (θ+4τ)/sr (settling time, from auto_tune_seed_buffer) so the
    # DV settles at each level (steady-state gain identifiable); seg_min ≈
    # τ/(3·sr) (fast transient) so the WM also learns the DV dynamics.
    seg_max = int(getattr(cfg, 'prbs_seed_segment_steps', 0) or 0)
    seg_max = max(8, min(seg_max if seg_max > 0 else T // 12, T // 4))
    seg_min_cfg = int(getattr(cfg, 'prbs_seed_segment_steps_min', 0) or 0)
    seg_min = max(2, min(seg_min_cfg, seg_max - 1)) if seg_min_cfg > 1 else seg_max
    # Pre-roll segment start times (log-uniform multi-timescale).
    seg_starts: List[int] = []
    t = 0
    while t < T - max(8, seg_min):
        seg_starts.append(int(t))
        if seg_min < seg_max:
            u = env.rng.uniform(np.log(seg_min), np.log(max(seg_min + 1, seg_max)))
            sl = int(round(float(np.exp(u))))
        else:
            sl = seg_max
        t += max(seg_min, min(seg_max, sl))
    n_seg = len(seg_starts)
    if n_seg == 0:
        return []
    # ----- Build an independent stratified-level PRBS per DV channel -----
    # Levels are OFFSETS from the channel baseline, stratified over
    # ±op_frac·(span/2) so the sweep covers the operating range (boundary
    # strata included).  The disturbance schedule accumulates achieved
    # deltas as a persistent offset (telescoping), so delta_k = L_k − L_{k−1}
    # makes the held DV level track L_k.
    op_frac = float(np.clip(getattr(cfg, 'dv_prbs_op_frac', 0.6), 0.05, 0.95))
    dv_schedule: List[Dict] = []
    for ch in dv_chs:
        b = ch.get('bounds')
        span = (float(b[1]) - float(b[0])) if (isinstance(b, list)
                                                and len(b) >= 2) else 1.0
        amp = op_frac * 0.5 * abs(span)            # max |offset| from baseline
        # Stratified target levels across [-amp, +amp] (guarantees the
        # extremes are visited even when n_seg is small).
        strata_n = max(1, min(int(getattr(cfg, 'prbs_seed_n_strata', 8)), n_seg))
        edges = np.linspace(-amp, +amp, strata_n + 1)
        centers = 0.5 * (edges[:-1] + edges[1:])
        levels = np.empty(n_seg, dtype='float64')
        order = env.rng.permutation(n_seg)
        half_w = amp / strata_n
        jit = env.rng.uniform(-half_w, +half_w, size=strata_n)
        levels[:strata_n] = centers + jit
        if n_seg > strata_n:
            levels[strata_n:] = env.rng.uniform(-amp, +amp, size=n_seg - strata_n)
        levels = levels[order]
        prev = 0.0
        for k, start in enumerate(seg_starts):
            delta = float(levels[k] - prev)
            prev = float(levels[k])
            if abs(delta) < 1e-9:
                continue
            dv_schedule.append({
                'name': f"dv_prbs_{ch.get('name', ch.get('pos', '?'))}_t{int(start)}",
                'target_group': 'dv',
                'target_pos': int(ch.get('pos', 0)),
                'start': int(start),
                'duration': 1,
                'shape': 'step',
                'delta': float(delta),
                'source': 'dv_prbs_seed',
                '_applied': False,
            })
    return dv_schedule


def collect_dv_prbs_episode(env: APCEnv, cfg: TrainConfig, *,
                             mv_level: float = 0.0,
                             ) -> Dict[str, np.ndarray]:
    """DV-PRBS seed episode (2026-06-14): the DV analogue of
    ``collect_prbs_episode``.  Holds the MV at a (stratified) operating
    point and sweeps EVERY measured-DV channel with an independent,
    full-range, multi-timescale, stratified PRBS, with the hidden OU
    disturbance OFF.

    Why this exists (p119–p121 RCA): in the staged curriculum the WM's
    input→CV gain is identified on CLEAN Stage-1 data.  ``collect_prbs_
    episode`` gives the MV full-range stratified PRBS in (nearly) every
    seed episode, and the WM conditions on the NOISE-FREE commanded MV
    action — so the MV→CV gain is identified unbiasedly (p121 ratio 0.93).
    The DV, by contrast, is never PRBS-swept: it is only nudged by sparse
    10–30 %-span steps in the ~20 step-test episodes (``dv_share`` 0.5),
    and during clean Stage 1 (hidden disturbance OFF) that is the ONLY DV
    motion.  The result is a ~30× MV-vs-DV excitation asymmetry → the
    DV→CV gain is systematically ATTENUATED (p119–p121 DV ratio stuck
    ~0.75) by two signal-theory mechanisms:
      * insufficient / non-persistent excitation (the WM rarely sees the
        DV held long enough to reach steady state), and
      * errors-in-variables / regression dilution — the WM's DV regressor
        is the *measured* (noisy) DV obs, not a clean command, so a low
        DV-signal-to-noise ratio biases the learned gain toward zero.

    This episode removes BOTH: it drives the DV with the SAME persistent,
    full-range, multi-timescale excitation the MV gets, at LARGE amplitude
    (so Var(DV) ≫ Var(meas-noise) and the dilution factor → 1), with the
    MV HELD (so ∂CV/∂DV is identifiable in isolation, no MV–DV confound).
    Backbone-agnostic: with ``dv_as_input`` the WM consumes the measured
    DV as an exogenous input, so it sees clean ``(DV_prbs, resulting CV)``
    pairs and learns the DV gain + dynamics directly.  A correct DV gain
    also cleans the DOB innovation (DV-driven CV no longer leaks into the
    disturbance estimate), so it improves the unmeasured-disturbance head
    as a coupled bonus.

    Falls back to an MV-hold episode when the sim has no DV channels.
    ``mv_level`` (normalized action space) sets the held MV operating
    point; vary it across the seed batch for MV-level coverage.  Returns
    the same dict shape as ``collect_episode``.
    """
    from utils.training_disturbance import _channel_catalog
    T = int(cfg.episode_length)
    D = env.obs_dim
    obs_window = env.reset(exploration=True)
    env._schedule = []
    env._hidden_disturbance = None

    channels = _channel_catalog(env.sim)
    dv_chs = list(channels.get('dv', []))
    has_dv = len(dv_chs) > 0

    # Held MV operating point (clip to the seed op-band, then [-1, 1]).
    u_band = float(getattr(cfg, 'constant_action_seed_op_band', 0.6))
    cur_u = float(np.clip(mv_level, -u_band, u_band))
    cur_u = float(np.clip(cur_u, -1.0, 1.0))
    act_buf = np.full((T, env.action_dim), cur_u, dtype='float32')

    if not has_dv:
        # No DV channels → just run the held-MV episode (still useful as a
        # steady-state MV anchor; matches collect_constant semantics).
        obs_buf = np.zeros((T, D), dtype='float32')
        rew_buf = np.zeros(T, dtype='float32')
        cont_buf = np.ones(T, dtype='float32')
        for t in range(T):
            obs_buf[t] = obs_window[-1]
            obs_window, reward, done, _ = env.step(act_buf[t])
            rew_buf[t] = reward
            cont_buf[t] = 0.0 if done and t == T - 1 else 1.0
            if done:
                break
        return {'obs': obs_buf, 'act': act_buf, 'rew': rew_buf, 'cont': cont_buf}

    # Full-range, multi-timescale, stratified DV-PRBS schedule (shared
    # builder; byte-identical excitation statistics to the R1a Stage-1
    # on-policy DV excitation in ``APCEnv.reset``).
    env._schedule = _build_dv_prbs_schedule(env, cfg)

    # ----- Run the episode (MV held, DV swept by the schedule) ----------
    obs_buf = np.zeros((T, D), dtype='float32')
    rew_buf = np.zeros(T, dtype='float32')
    cont_buf = np.ones(T, dtype='float32')
    for t in range(T):
        obs_buf[t] = obs_window[-1]
        obs_window, reward, done, _ = env.step(act_buf[t])
        rew_buf[t] = reward
        cont_buf[t] = 0.0 if done and t == T - 1 else 1.0
        if done:
            break
    return {'obs': obs_buf, 'act': act_buf, 'rew': rew_buf, 'cont': cont_buf}


# ---------------------------------------------------------------------------
# Phase 1 / 2 — World model loss (tokenizer recon + shortcut forcing)
# ---------------------------------------------------------------------------

def _adaptive_return_cap(cfg: TrainConfig) -> Optional[float]:
    """Option (a): adaptive bounded-return envelope ``k·B/(1-γλ)``.

    Returns the symmetric clamp magnitude for bootstrap target-values and
    λ-return targets, or ``None`` when disabled / the reward is unbounded
    (no ``B`` envelope to derive a bound from).  Sim-agnostic: derived purely
    from ``cfg.bound_training_reward_max``, ``gamma`` and ``gae_lambda``;
    backbone-independent (applied in the shared λ-return tail of both paths).
    """
    if not bool(getattr(cfg, 'return_value_adaptive_cap', True)):
        return None
    if not bool(getattr(cfg, 'bound_training_reward', False)):
        return None
    B = float(getattr(cfg, 'bound_training_reward_max', 0.0) or 0.0)
    k = float(getattr(cfg, 'return_value_cap_k', 2.0) or 0.0)
    if B <= 0.0 or k <= 0.0:
        return None
    gamma = float(getattr(cfg, 'gamma', 0.997))
    lam = float(getattr(cfg, 'gae_lambda', 0.95))
    if bool(getattr(cfg, 'return_value_cap_gamma_horizon', False)):
        # Cap the VALUE-horizon (1/(1-γ)); see TrainConfig note — the legacy
        # λ-horizon below made the cap ~10× too tight and flattened the value.
        denom = max(1e-6, 1.0 - gamma)
    else:
        gl = max(0.0, min(0.999999, gamma * lam))
        denom = 1.0 - gl
    return k * B / denom


def _critic_anchor_lambda(cfg: TrainConfig) -> float:
    """Option (B): the λ used by the REPLAY critic anchor's TD-λ recursion.

    Defaults to the imagination ``gae_lambda`` (exact legacy behaviour) when
    ``critic_anchor_lambda`` is unset.  Setting it ~0.97–1.0 turns the anchor
    into a near-Monte-Carlo return-to-go over the real context so the critic
    target sees the FULL long-horizon (multi-cycle) cost contained in the real
    buffer data.  Decoupled from the cascade-sensitive imagination λ.
    Clamped to [0, 1].
    """
    v = getattr(cfg, 'critic_anchor_lambda', None)
    if v is None:
        return float(getattr(cfg, 'gae_lambda', 0.95))
    return float(max(0.0, min(1.0, float(v))))


def _critic_anchor_coef(cfg: TrainConfig) -> float:
    """Option (B): the anchor weight, optionally raised above the base
    ``critic_replay_anchor_coef`` so the long-horizon real target can overcome
    the myopic imagined critic loss.  ``critic_anchor_coef_long=None`` ⇒ use
    the base coef unchanged (legacy)."""
    base = float(getattr(cfg, 'critic_replay_anchor_coef', 0.0) or 0.0)
    long = getattr(cfg, 'critic_anchor_coef_long', None)
    if long is None:
        return base
    return float(max(0.0, float(long)))


def _steady_held_mask(obs: torch.Tensor, act: torch.Tensor,
                       cfg: TrainConfig) -> Optional[torch.Tensor]:
    """Option (b): adaptive SETTLED + HELD step mask, backbone-independent.

    ``obs`` (B, T, D), ``act`` (B, T, A).  Returns a (B, T-1) float mask that
    is 1 at step ``t`` when the action is HELD over ``[t, t+1]`` (|Δa| ≤ eps)
    AND the observation has SETTLED (|Δo| < settle_frac · per-channel batch
    std — RELATIVE, the fix for the P64 absolute-threshold failure).  Returns
    ``None`` when the window is too short.  Never NaN.
    """
    if obs.dim() != 3 or obs.shape[1] < 2:
        return None
    held_eps = float(getattr(cfg, 'wm_steady_held_eps', 0.02))
    settle_frac = float(getattr(cfg, 'wm_steady_settle_frac', 0.15))
    da = (act[:, 1:] - act[:, :-1]).abs().amax(dim=-1)            # (B, T-1)
    held = (da <= held_eps).float()
    do = (obs[:, 1:] - obs[:, :-1]).abs()                        # (B, T-1, D)
    ch_std = obs.reshape(-1, obs.shape[-1]).float().std(dim=0).clamp_min(1e-6)
    settled = (do <= (settle_frac * ch_std)).all(dim=-1).float()  # (B, T-1)
    return held * settled


def _rssm_steady_consistency(model: DreamerV4, feats: torch.Tensor,
                              obs: torch.Tensor, act: torch.Tensor,
                              cfg: TrainConfig,
                              ) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Option (b), RSSM: held-action one-step fixed-point penalty.

    Reconstructs the posterior RSSMState at each step ``t`` from ``feats``,
    advances ONE prior step under the action driving ``t+1`` (no sampling),
    decodes, and penalises ``MSE(pred_obs_{t+1}, obs_{t+1})`` at SETTLED+HELD
    steps.  Trains the gru+prior+decoder toward a held-action fixed point at
    observed settled states.  Graceful zero fallback (never NaN).
    """
    device = obs.device
    zero = torch.zeros((), device=device, dtype=feats.dtype)
    mask = _steady_held_mask(obs, act, cfg)
    if mask is None or float(mask.sum()) <= 0.0:
        return zero, {'wm_steady_held_frac': 0.0}
    from models.dreamer_v4_rssm import RSSMState
    rssm = model.dynamics
    B, T = obs.shape[:2]
    f = feats[:, :-1]                                  # (B, T-1, F)
    Bm = B * (T - 1)
    h = f[..., :rssm.deter_dim].reshape(Bm, -1)
    # Scope 2: slice EXACTLY the stochastic block; ``feat`` may carry a DOB
    # d-tail (deter+stoch+n_cv) that must not bleed into the z reshape.
    _ze = rssm.deter_dim + rssm.stoch_flat_dim
    z_flat = f[..., rssm.deter_dim:_ze]
    z = z_flat.reshape(Bm, rssm.n_categoricals, rssm.n_classes)
    state = RSSMState(
        h=h,
        z_logits=torch.zeros(Bm, rssm.n_categoricals, rssm.n_classes,
                             device=device, dtype=f.dtype),
        z=z)
    a_next = act[:, 1:].reshape(Bm, -1)                # action driving t+1
    nxt = rssm.img_step(state, a_next, sample=False)
    pred_obs = rssm.decode(nxt.feat).reshape(B, T - 1, -1)
    tgt = obs[:, 1:].detach()
    se = (pred_obs - tgt).pow(2).mean(dim=-1)          # (B, T-1)
    denom = mask.sum().clamp_min(1.0)
    loss = (se * mask).sum() / denom
    return loss, {'wm_steady_held_frac': float(mask.mean())}


def _sf_steady_consistency(model: DreamerV4, z_clean: torch.Tensor,
                            obs: torch.Tensor, act: torch.Tensor,
                            cfg: TrainConfig,
                            ) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Option (b), transformer: explicit held-action anchor at the rollout
    boundary, complementing the dynamics' native ``sf_bootstrap`` self-
    consistency (the transformer's intrinsic steady-state mechanism).

    At the final context position, if the action is HELD and the obs has
    SETTLED, the K-step imagined next-z must decode back to the same settled
    obs.  One extra (differentiable) ``imagine_next_z`` forward; cheap.
    Graceful zero fallback (never NaN).
    """
    device = obs.device
    zero = torch.zeros((), device=device, dtype=z_clean.dtype)
    B, T = obs.shape[:2]
    if T < 2:
        return zero, {'wm_steady_held_frac': 0.0}
    held_eps = float(getattr(cfg, 'wm_steady_held_eps', 0.02))
    settle_frac = float(getattr(cfg, 'wm_steady_settle_frac', 0.15))
    da = (act[:, -1] - act[:, -2]).abs().amax(dim=-1)            # (B,)
    held = (da <= held_eps).float()
    do = (obs[:, -1] - obs[:, -2]).abs()                        # (B, D)
    ch_std = obs.reshape(-1, obs.shape[-1]).float().std(dim=0).clamp_min(1e-6)
    settled = (do <= (settle_frac * ch_std)).all(dim=-1).float()  # (B,)
    mask = held * settled                                       # (B,)
    if float(mask.sum()) <= 0.0:
        return zero, {'wm_steady_held_frac': 0.0}
    z_next = model.imagine_next_z(z_clean, act[:, -1], k_steps=int(cfg.k_max),
                                  action_history=act)           # (B, z)
    pred_obs = model.tokenizer.decode(z_next)                   # (B, D)
    tgt = obs[:, -1].detach()
    se = (pred_obs - tgt).pow(2).mean(dim=-1)                   # (B,)
    denom = mask.sum().clamp_min(1.0)
    loss = (se * mask).sum() / denom
    return loss, {'wm_steady_held_frac': float(mask.mean())}


def _mask_measured_dv_from_feat(model: DreamerV4,
                                 feat: torch.Tensor) -> torch.Tensor:
    """Zero the MEASURED-dv columns of a WM ``feat`` (p130 RCA).

    ``dv_feedforward`` appends the measured DV after the latent core
    ``[h, z]`` (see ``RSSMState.feat`` / ``TSSMState.feat``), so the disturbance
    head — which must predict the UNMEASURED load — would otherwise read the
    measured DV directly and conflate the two.  Returns ``feat`` unchanged when
    no dv is fed forward; else a clone with ``feat[..., core:core+dv_feed]``
    zeroed (the (h, z) latent and any DOB ``d`` tail are untouched, so the head
    can still infer the load indirectly).  Backbone-agnostic (RSSM + TSSM share
    ``deter_dim`` / ``stoch_flat_dim`` / ``_dv_feed_dim``).
    """
    dyn = getattr(model, 'dynamics', None)
    dv_feed = int(getattr(dyn, '_dv_feed_dim', 0) or 0)
    if dyn is None or dv_feed <= 0:
        return feat
    # feat = [h, z, (c), (dv), (d)] — the continuous latent ``c`` (2026-06-22)
    # sits BETWEEN the categorical core and the dv feedforward, so the dv block
    # starts at deter+stoch+cont_dim (NOT deter+stoch).  Missing the cont_dim
    # offset zeroed a cont/gain channel and let the measured DV LEAK into the
    # disturbance head (the exact p130 conflation this guard prevents).
    cont_dim = int(getattr(dyn, 'cont_dim', 0) or 0)
    core = int(dyn.deter_dim) + int(dyn.stoch_flat_dim) + cont_dim
    if feat.shape[-1] < core + dv_feed:
        return feat
    out = feat.clone()
    out[..., core:core + dv_feed] = 0.0
    return out


def _disturbance_head_loss(model: DreamerV4, feat: torch.Tensor,
                            dist_target: Optional[torch.Tensor],
                            recon_loss: torch.Tensor, cfg: TrainConfig,
                            ) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """Auxiliary WM disturbance-estimator loss (P87), backbone-agnostic.

    Regresses the hidden/unmeasured disturbance from the world-model feature
    ``feat`` (RSSM posterior ``[h, z]`` or the transformer agent-register
    hidden state).  Returns ``(weighted_term, loss_detached, rmse)`` where
    ``weighted_term`` is what to add to ``wm_total`` (zero when the head is
    disabled / no target).

    Two regimes (the p109 RCA design):
      * ``stop_grad`` (DEFAULT): the head reads a DETACHED latent — a pure
        read-out probe.  It cannot touch the WM trunk, so we train it at the
        full ``disturbance_loss_scale`` (its magnitude is harmless).  The
        policy still sees the disturbance in the latent (feed-forward works);
        the validation diagnostic measures how well.
      * shaping (``stop_grad=False``): the head's gradient reaches the trunk
        (the original P87 feed-forward-shaping intent).  The weight is then set
        ADAPTIVELY to ``disturbance_loss_rel_weight`` x the RECON term via a
        detached loss-ratio, so it is SIM-AGNOSTIC and can never swamp the
        gain-carrying recon gradient (p109: an absolute 1.0 made it 27x recon).
    """
    zero = torch.zeros((), device=feat.device, dtype=feat.dtype)
    dist_coef = float(getattr(cfg, 'disturbance_loss_scale', 0.0) or 0.0)
    dist_head = getattr(model, 'disturbance', None)
    if dist_head is None or dist_target is None or dist_coef <= 0.0:
        return zero, zero, 0.0
    stop_grad = bool(getattr(cfg, 'disturbance_head_stop_grad', True))
    feat_in = feat.detach() if stop_grad else feat
    if bool(getattr(cfg, 'disturbance_head_exclude_dv', True)):
        feat_in = _mask_measured_dv_from_feat(model, feat_in)
    dpred = dist_head(feat_in)
    if dpred.shape != dist_target.shape:
        return zero, zero, 0.0
    dt = dist_target.to(dpred.dtype)
    dist_loss = F.mse_loss(dpred.float(), dt.float())
    if stop_grad:
        # Read-out probe: gradient-isolated from the trunk, so the loss
        # magnitude is harmless — train the head at full weight so it learns a
        # clean estimate fast.
        term = dist_coef * dist_loss
    else:
        # Latent-SHAPING (active feed-forward): bound the trunk gradient.
        rel = float(getattr(cfg, 'disturbance_loss_rel_weight', 0.0) or 0.0)
        if rel > 0.0:
            # Adaptive sim-agnostic: term magnitude == rel x recon-term.  The
            # detached ratio cancels both the sim-specific disturbance scale and
            # the recon scale, leaving exactly ``rel`` x the recon gradient.
            recon_scale = float(getattr(cfg, 'recon_scale', 0.1) or 0.0)
            recon_term = (recon_scale * recon_loss).detach().clamp_min(0.0)
            eff = rel * recon_term / dist_loss.detach().clamp_min(1e-8)
            eff = torch.clamp(eff, max=dist_coef)      # absolute ceiling
            term = eff * dist_loss
        else:
            # Legacy absolute path (NOT sim-agnostic — discouraged); gated.
            thr = float(getattr(cfg, 'disturbance_loss_gate_recon', 0.0) or 0.0)
            gate = (torch.clamp(thr / recon_loss.detach().clamp_min(1e-6),
                                max=1.0)
                    if thr > 0.0 else torch.ones((), device=feat.device))
            term = dist_coef * gate * dist_loss
    with torch.no_grad():
        rmse = float(dist_loss.clamp_min(0.0).sqrt())
    return term, dist_loss.detach(), rmse


def _wm_latent_overshoot_loss(model: DreamerV4, feats: torch.Tensor,
                               obs: torch.Tensor, act: torch.Tensor,
                               cfg: TrainConfig,
                               recon_loss: Optional[torch.Tensor] = None,
                               ) -> Tuple[torch.Tensor, float]:
    """Option #2 (P88): multi-step LATENT OVERSHOOTING — open-loop prior
    rollout accuracy.

    Dreamer-v3 trains the prior only ONE step ahead (the KL term), so the
    open-loop imagination rollout the actor depends on accumulates error every
    step and per-offset WM fidelity decays fast (r 0.74@H13 -> 0.52@H55).  This
    is the PlaNet/Dreamer-v1 "latent overshooting" objective (v2/v3 dropped it
    because 1-step sufficed for Atari): from a strided set of start positions
    ``t`` we reconstruct the posterior state, roll the PRIOR forward ``K`` steps
    under the REAL actions ``a_{t+1..t+K}`` WITH NO OBSERVATIONS, decode each
    predicted feature and penalise ``MSE(decode, obs_{t+1..t+K})``.  This
    directly trains the gru+prior+decoder for accurate MULTI-step prediction —
    what makes a long imagination horizon H legitimately usable instead of
    leaning on the WM's weakest capability.

    RSSM-only; returns ``(0, 0.0)`` for the SF-transformer backbone (its
    shortcut-forcing loss is the native multi-step-prediction mechanism).
    Cost ~ O(B · n_starts · K) GRU steps; ``n_starts`` capped via a stride by
    ``wm_overshoot_max_starts`` so the term is bounded.  ``sample=True`` so the
    straight-through categorical grad trains the PRIOR (sample=False would give
    the prior logits no gradient).
    """
    coef = float(getattr(cfg, 'wm_overshoot_coef', 0.0) or 0.0)
    K = int(getattr(cfg, 'wm_overshoot_len', 0) or 0)
    device = obs.device
    zero = torch.zeros((), device=device, dtype=obs.dtype)
    if coef <= 0.0 or K < 1:
        return zero, 0.0
    if getattr(model, 'world_model_type', 'sf_transformer') != 'rssm':
        return zero, 0.0
    from models.dreamer_v4_rssm import RSSMState
    rssm = model.dynamics
    B, T = obs.shape[:2]
    K = min(K, T - 1)
    n_valid = T - K
    if n_valid < 1:
        return zero, 0.0
    max_starts = max(1, int(getattr(cfg, 'wm_overshoot_max_starts', 24) or 24))
    stride = max(1, n_valid // max_starts)
    starts = torch.arange(0, n_valid, stride, device=device)      # (S,)
    S = int(starts.numel())
    f0 = feats[:, starts]                                         # (B, S, F)
    Bm = B * S
    h = f0[..., :rssm.deter_dim].reshape(Bm, -1)
    # Scope 2: slice EXACTLY the stochastic block (exclude any DOB d-tail).
    _ze = rssm.deter_dim + rssm.stoch_flat_dim
    z = f0[..., rssm.deter_dim:_ze].reshape(
        Bm, rssm.n_categoricals, rssm.n_classes)
    # Per-step REAL action + DV sequences for k=1..K (gathered ONCE).
    k_off = torch.arange(1, K + 1, device=device)                 # (K,)
    idx = starts.view(S, 1) + k_off.view(1, K)                    # (S, K) time idx
    a_all = act[:, idx].reshape(Bm, K, -1)                        # (Bm, K, A)
    dv_all = (obs[:, idx].index_select(-1, rssm.dv_index_t).reshape(Bm, K, -1)
              if getattr(rssm, 'dv_dim', 0) > 0 else None)         # (Bm,K,dv)|None
    # COMPILED prior rollout (whole K-step img_step loop = ONE graph) + ONE
    # batched decode (the per-step decode was the launch-bound killer, exactly
    # what rollout_observed hoists out).
    roll_feats = rssm.img_rollout(h, z, a_all, dvs=dv_all, sample=True)  # (Bm,K,F)
    preds = rssm.decode(roll_feats).reshape(B, S, K, -1)          # (B, S, K, D)
    total = zero
    # Steady-state TAIL weighting (2026-06-20, p131 RCA).  The open-loop gain
    # contraction (decomp 1step→openloop ×0.876; probe: sampled open-loop gain
    # 0.79 vs real, and sample=False is WORSE 0.32 → the gain lives in the
    # learned SAMPLED prior, the loss is weak supervision NOT a sampling EIV) is
    # a STEADY-STATE / DC-gain phenomenon: the 1-step prior is faithful (×1.001)
    # but the gain compounds away over the rollout.  A UNIFORM ``/K`` mean
    # dilutes the settled tail to ~1/K weight, so the DC gain (where the
    # contraction lives) is under-supervised.  Weight step ``k`` by
    # ``(k/K)^tail_power`` (a smooth low-frequency / DC emphasis) and normalise
    # by Σw — bounded magnitude (still a weighted mean, no term inflation) so it
    # cannot destabilise the WM, but it concentrates the gain gradient on the
    # steady-state (p=2 → the last step gets ~3× its uniform weight, the noisy
    # early transient less — which the 1-step recon/KL already cover).
    # ``tail_power=0`` recovers the exact uniform mean.  Sim-agnostic (unitless
    # step fraction), backbone-agnostic.  ``DREAMER_WM_OVERSHOOT_TAIL_POWER``.
    tail_power = float(getattr(cfg, 'wm_overshoot_tail_power', 0.0) or 0.0)
    wsum = 0.0
    # Per-step CV-weighted MSE on the PRE-DECODED rollout (no per-step decode /
    # img_step launches now — those are batched above).  The CV-weight (p124)
    # keeps the small-variance CV step-response from being drowned by the
    # high-variance MV/DV channels so the open-loop gain stays supervised.
    for ki in range(K):
        pred = preds[:, :, ki]                                    # (B, S, D)
        tgt = obs[:, idx[:, ki]].detach()                        # (B, S, D)
        w_k = (float(ki + 1) / float(K)) ** tail_power if tail_power > 0.0 else 1.0
        total = total + w_k * _weighted_recon_mse(pred, tgt, cfg)
        wsum += w_k
    loss = total / max(wsum, 1e-8)
    # Soft recon-fidelity gate: ramp the term in only as 1-step recon converges
    # (early P1 the WM can't predict multi-step; an ungated term would swamp the
    # encoder/decoder).  ``gate_recon<=0`` disables.
    thr = float(getattr(cfg, 'wm_overshoot_gate_recon', 0.0) or 0.0)
    if thr > 0.0 and recon_loss is not None:
        gate = torch.clamp(thr / recon_loss.detach().clamp_min(1e-6), max=1.0)
        loss = gate * loss
    return loss, float(S)


def _wm_held_rollout_stationarity_loss(model: DreamerV4, feats: torch.Tensor,
                                        obs: torch.Tensor, act: torch.Tensor,
                                        cfg: TrainConfig,
                                        recon_loss: Optional[torch.Tensor] = None,
                                        ) -> Tuple[torch.Tensor, float]:
    """Option (b2, P89): multi-step HELD-ACTION rollout stationarity.

    Complements the 1-step ``_rssm_steady_consistency`` (which only fires on the
    rare naturally held+settled replay steps — p88c held_frac≈0.84%) by ACTIVELY
    creating the held condition: from a strided set of start positions ``t`` we
    reconstruct the posterior RSSMState, hold the action at ``a_t`` CONSTANT and
    roll the PRIOR forward ``K`` steps (``sample=True`` straight-through so the
    stochastic prior receives gradient), then penalise the NET DRIFT of the
    deterministic state ``h`` between an early post-transient window
    ``[s, s+win)`` (``s = settle_frac·K``) and the final window ``[K-win, K)``.

    Rationale: the steady-state diagnostic shows the WM imagination DRIFTS under
    a held action (0% convergence) instead of reaching a fixed point.  Measuring
    the net displacement of ``h`` between two late windows (a) averages out the
    categorical sampling noise, (b) is GAIN-NEUTRAL — it constrains only the
    tail DISPLACEMENT, never the response magnitude, and leaves the transient
    ``[0, s)`` free so the overshoot/recon terms still set the gain — and (c) is
    scale-robust (normalised by the rollout's own ``h`` std).  RSSM-only;
    returns ``(0, 0.0)`` for the SF backbone.  Cost ~ O(B·max_starts·K) GRU
    steps, bounded by ``wm_held_rollout_max_starts``.  ``sample=True`` so the
    straight-through categorical grad trains the PRIOR (the drift source).
    """
    coef = float(getattr(cfg, 'wm_held_rollout_coef', 0.0) or 0.0)
    K = int(getattr(cfg, 'wm_held_rollout_len', 0) or 0)
    device = obs.device
    zero = torch.zeros((), device=device, dtype=obs.dtype)
    if coef <= 0.0 or K < 4:
        return zero, 0.0
    if getattr(model, 'world_model_type', 'sf_transformer') != 'rssm':
        return zero, 0.0
    from models.dreamer_v4_rssm import RSSMState
    rssm = model.dynamics
    B, T = obs.shape[:2]
    win = max(1, int(getattr(cfg, 'wm_held_rollout_win', 8) or 8))
    s = int(float(getattr(cfg, 'wm_held_rollout_settle_frac', 0.5) or 0.5) * K)
    # keep the two averaging windows non-overlapping inside [0, K)
    s = max(win, min(s, K - 2 * win))
    if s < win or K - win <= s + win:
        return zero, 0.0
    max_starts = max(1, int(getattr(cfg, 'wm_held_rollout_max_starts', 12) or 12))
    stride = max(1, T // max_starts)
    starts = torch.arange(0, T, stride, device=device)            # (S,)
    S = int(starts.numel())
    f0 = feats[:, starts]                                         # (B, S, F)
    Bm = B * S
    h = f0[..., :rssm.deter_dim].reshape(Bm, -1)
    # Scope 2: slice EXACTLY the stochastic block (exclude any DOB d-tail).
    _ze = rssm.deter_dim + rssm.stoch_flat_dim
    z = f0[..., rssm.deter_dim:_ze].reshape(
        Bm, rssm.n_categoricals, rssm.n_classes)
    a_hold = act[:, starts].reshape(Bm, -1).detach()              # HELD action a_t
    # DV-as-input: hold the measured DV CONSTANT at its start value across the
    # rollout too — so this probes true held-(action+DV) stationarity and the
    # WM no longer needs to hallucinate a drifting DV (the steady-state win).
    dv_hold = (obs[:, starts].index_select(-1, rssm.dv_index_t).reshape(Bm, -1)
               .detach()
               if getattr(rssm, 'dv_dim', 0) > 0 else None)
    # HELD action + DV constant across the rollout: broadcast to K so the whole
    # loop runs through the COMPILED img_rollout (one graph, no per-step launch).
    a_hold_seq = a_hold.unsqueeze(1).expand(Bm, K, a_hold.shape[-1])   # (Bm,K,A)
    dv_hold_seq = (dv_hold.unsqueeze(1).expand(Bm, K, dv_hold.shape[-1])
                   if dv_hold is not None else None)
    roll_feats = rssm.img_rollout(h, z, a_hold_seq, dvs=dv_hold_seq,
                                  sample=True)                    # (Bm, K, F)
    Hroll = roll_feats[..., :rssm.deter_dim]                      # (Bm, K, deter)
    h_scale = Hroll.detach().std().clamp_min(1e-3)
    early = Hroll[:, s:s + win].mean(dim=1)                      # (B*S, deter)
    late = Hroll[:, K - win:].mean(dim=1)                        # (B*S, deter)
    loss = ((late - early) / h_scale).pow(2).mean()
    # Soft recon-fidelity gate (mirror overshoot): ramp the term in only as
    # 1-step recon converges so an untrained decoder/prior is not destabilised
    # early in P1.  ``gate_recon<=0`` disables.
    thr = float(getattr(cfg, 'wm_held_rollout_gate_recon', 0.0) or 0.0)
    if thr > 0.0 and recon_loss is not None:
        gate = torch.clamp(thr / recon_loss.detach().clamp_min(1e-6), max=1.0)
        loss = gate * loss
    return loss, float(S)


def _weighted_recon_mse(recon: torch.Tensor, target: torch.Tensor,
                        cfg: TrainConfig) -> torch.Tensor:
    """Reconstruction MSE with optional CV-channel up-weighting.

    WM autoencoder lever (2026-06-09): the uniform per-channel MSE lets the
    small CV step-gain be drowned by the high-variance MV/DV channels, so the
    decoder under-fits the very channel whose gain the controller needs (the
    posterior-prior probe's dominant residual-bias lever, real→posterior~0.85).
    When ``cfg.wm_recon_cv_weight != 1.0`` the CV obs channels' squared error
    is scaled by that factor (others stay 1.0), then the weight vector is
    renormalised to mean 1.0 so the OVERALL recon magnitude is preserved
    (``recon_scale`` need not be retuned) — only the WITHIN-recon emphasis
    shifts toward the CV.  ``cv_weight == 1.0`` or no CV indices ⇒ byte-for-byte
    ``F.mse_loss`` (identity = p106-baseline).  Backbone-agnostic.
    """
    cv_w = float(getattr(cfg, 'wm_recon_cv_weight', 1.0) or 1.0)
    cv_idx = tuple(getattr(cfg, 'cv_obs_indices', ()) or ())
    if cv_w == 1.0 or not cv_idx:
        return F.mse_loss(recon, target)
    D = int(target.shape[-1])
    valid = [int(i) for i in cv_idx if 0 <= int(i) < D]
    if not valid:
        return F.mse_loss(recon, target)
    w = torch.ones(D, device=target.device, dtype=torch.float32)
    w[torch.tensor(valid, device=target.device, dtype=torch.long)] = cv_w
    w = w * (float(w.numel()) / w.sum().clamp_min(1e-8))  # renorm: mean→1.0
    se = (recon.float() - target.float()).pow(2)           # (..., D)
    return (se * w).mean()


def _wm_gain_match_loss(model: DreamerV4, feats: torch.Tensor,
                        obs: torch.Tensor, act: torch.Tensor, cfg: TrainConfig
                        ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """C(1): finite-difference step-response asymptote gain-matching (RSSM).

    From a strided set of posterior start states, roll the PRIOR forward ``K``
    steps under (a) a HELD baseline action/DV and (b) a unit STEP in each input
    channel.  The DIFFERENCE of the decoded CV at step ``K`` cancels the common
    transient and isolates the WM's realized STEADY-STATE gain ``ΔCV/Δinput``;
    we match it to the identified gain (in WM/normalized units).  The continuous
    gain channel gives the WM the un-quantized capacity this loss grabs onto, so
    together they pin the subdominant DV gain the categorical attenuates.

    ``sample=False`` freezes the categorical at its argmax so the gain gradient
    flows into the CONTINUOUS gain channel + decoder + GRU (not the categorical
    we are trying to bypass).  RSSM + TSSM (the TSSM rolls from a fresh
    KV-cache); ``(0, {})`` for other backbones / when off.
    """
    zero = torch.zeros((), device=obs.device, dtype=obs.dtype)
    diag: Dict[str, torch.Tensor] = {}
    if float(getattr(cfg, 'gain_match_coef', 0.0) or 0.0) <= 0.0:
        return zero, diag
    _wmt = getattr(model, 'world_model_type', 'sf_transformer')
    if _wmt not in ('rssm', 'tssm'):
        return zero, diag
    rssm = model.dynamics
    if int(getattr(rssm, 'cont_gain_dim', 0) or 0) <= 0 or rssm.n_cv <= 0:
        return zero, diag
    mv_target = list(getattr(cfg, 'gain_match_mv_target', ()) or ())
    dv_target = list(getattr(cfg, 'gain_match_dv_target', ()) or ())
    if not mv_target and not dv_target:
        return zero, diag
    if _wmt == 'tssm':
        from models.transformer_ssm import TSSMState as _State
    else:
        from models.dreamer_v4_rssm import RSSMState as _State
    B, T = obs.shape[:2]
    K = int(getattr(cfg, 'gain_match_len', 0) or 0)
    K = min(K, T - 1) if K > 0 else (T - 1)
    if K < 2:
        return zero, diag
    step = float(getattr(cfg, 'gain_match_step', 1.0) or 1.0)
    max_starts = max(1, int(getattr(cfg, 'gain_match_max_starts', 6) or 6))
    n_valid = T - K
    if n_valid < 1:
        return zero, diag
    stride = max(1, n_valid // max_starts)
    starts = torch.arange(0, n_valid, stride, device=obs.device)
    S = int(starts.numel())
    f0 = feats[:, starts]                                   # (B, S, F)
    Bm = B * S
    h0 = f0[..., :rssm.deter_dim].reshape(Bm, -1)
    _ze = rssm.deter_dim + rssm.stoch_flat_dim
    z0 = f0[..., rssm.deter_dim:_ze].reshape(
        Bm, rssm.n_categoricals, rssm.n_classes)
    c0 = (f0[..., _ze:_ze + rssm.cont_dim].reshape(Bm, -1)
          if rssm.cont_dim > 0 else None)
    cv_idx = rssm.cv_index_t

    def _state():
        kw = dict(
            h=h0.clone(),
            z_logits=torch.zeros(Bm, rssm.n_categoricals, rssm.n_classes,
                                 device=obs.device, dtype=f0.dtype),
            z=z0.clone(), c=(c0.clone() if c0 is not None else None))
        if _wmt == 'tssm':
            kw.update(kv_cache=None, pos=0)   # roll from a fresh transformer ctx
        return _State(**kw)

    a_base = act[:, starts].reshape(Bm, -1)                 # (Bm, A)
    dv0 = (obs[:, starts].index_select(-1, rssm.dv_index_t).reshape(Bm, -1)
           if getattr(rssm, 'dv_dim', 0) > 0 else None)

    def _roll(a_held, dv_held):
        st = _state()
        for _ in range(K):
            st = rssm.img_step(st, a_held, dv=dv_held, sample=False)
        return rssm.decode(st.feat).index_select(-1, cv_idx)   # (Bm, n_cv)

    cv_base = _roll(a_base, dv0)
    total = zero
    nterm = 0
    for j, tgt_row in enumerate(mv_target):                # MV: step the action
        if j >= a_base.shape[-1]:
            break
        a_step = a_base.clone()
        a_step[:, j] = a_step[:, j] + step
        g_wm = (_roll(a_step, dv0) - cv_base) / step        # (Bm, n_cv)
        tgt = torch.tensor(list(tgt_row), device=obs.device, dtype=g_wm.dtype)
        total = total + (g_wm - tgt).pow(2).mean()
        nterm += 1
    if dv0 is not None:
        for j, tgt_row in enumerate(dv_target):            # DV: step the DV input
            if j >= dv0.shape[-1]:
                break
            dv_step = dv0.clone()
            dv_step[:, j] = dv_step[:, j] + step
            g_wm = (_roll(a_base, dv_step) - cv_base) / step
            tgt = torch.tensor(list(tgt_row), device=obs.device,
                               dtype=g_wm.dtype)
            total = total + (g_wm - tgt).pow(2).mean()
            nterm += 1
    if nterm == 0:
        return zero, diag
    loss = total / float(nterm)
    diag['gain_match_n'] = torch.tensor(float(nterm), device=obs.device)
    return loss, diag


def _resolve_gain_match_targets(env: 'APCEnv', cfg: TrainConfig) -> None:
    """C(1): convert the identified steady-state gains (engineering units) into
    the WM's NORMALIZED units and store them on ``cfg`` for the gain-match loss.

    Normalized-gain identities (the WM operates in obs-normalized space; the MV
    enters as the raw action ∈[-1,1], the DV as the normalized obs channel):
      * MV col:  ∂CV_norm/∂action = g_eng · (ΔMV_eng/Δaction) / cv_std
                 where ΔMV_eng/Δaction = (mv_hi − mv_lo)/2 (the action map).
      * DV col:  ∂CV_norm/∂dv_norm = g_eng · dv_std / cv_std.
    ``g_eng = amplitude/delta`` (signed) averaged over the valid identified
    step trials for each (input, CV) pair.  Raises on missing data → the caller
    disables the loss (graceful no-op; the cont gain channel still trains via
    recon).
    """
    out_dir = Path(getattr(cfg, 'out_dir', '.') or '.')
    roots = [out_dir]
    cur = out_dir
    for _ in range(4):
        if cur.parent == cur:
            break
        cur = cur.parent
        roots.append(cur)
    raw = None
    for root in roots:
        for cand in (root / 'plant_id' / 'dynamics_identification.json',
                     root / 'dynamics_identification.json'):
            if cand.exists():
                with open(cand) as _f:
                    raw = json.load(_f) or {}
                break
        if raw:
            break
    if not raw:
        raise FileNotFoundError('dynamics_identification.json not found')
    acc: Dict[tuple, List[float]] = {}
    for e in raw.get('per_pair_estimates', []) or []:
        if not e.get('valid'):
            continue
        try:
            delta = float(e.get('delta', 0.0))
            amp = float(e.get('amplitude', 0.0))
        except (TypeError, ValueError):
            continue
        if abs(delta) < 1e-9 or not np.isfinite(amp):
            continue
        it = str(e.get('input_type') or '')
        inp = str(e.get('input') or e.get('mv') or e.get('dv') or '')
        cvn = str(e.get('cv') or '')
        if it and inp and cvn:
            acc.setdefault((it, inp, cvn), []).append(amp / delta)
    g_eng = {k: float(np.mean(v)) for k, v in acc.items()}
    if not g_eng:
        raise ValueError('no valid identified gain estimates')
    sv = list(env.meta.get('state_variables', []) or [])

    def _nm(idxs):
        return [sv[i] if 0 <= i < len(sv) else f'idx{i}' for i in idxs]

    mv_idx = [int(x) for x in (env.meta.get('mv_indices') or [])]
    cv_idx = [int(x) for x in env.cv_indices]
    dv_idx = [int(x) for x in (env.meta.get('dv_indices') or [])
              if x is not None]
    mv_names, cv_names, dv_names = _nm(mv_idx), _nm(cv_idx), _nm(dv_idx)
    var = np.asarray(env.get_obs_norm_stats().get('var'), dtype='float64')
    std = np.sqrt(np.clip(var, 1e-8, None))
    cv_std = [float(std[i]) if i < len(std) else 1.0 for i in cv_idx]
    dv_std = [float(std[i]) if i < len(std) else 1.0 for i in dv_idx]
    mv_scale = [(float(hi) - float(lo)) / 2.0
                for lo, hi in list(env.mv_norm_ranges)]
    mv_target = []
    for i, mvn in enumerate(mv_names):
        sc = mv_scale[i] if i < len(mv_scale) else 1.0
        mv_target.append(tuple(
            float(g_eng.get(('mv', mvn, cvn), 0.0) * sc / max(cv_std[j], 1e-6))
            for j, cvn in enumerate(cv_names)))
    dv_target = []
    for i, dvn in enumerate(dv_names):
        dv_target.append(tuple(
            float(g_eng.get(('dv', dvn, cvn), 0.0) * dv_std[i]
                  / max(cv_std[j], 1e-6))
            for j, cvn in enumerate(cv_names)))
    cfg.gain_match_mv_target = tuple(mv_target)
    cfg.gain_match_dv_target = tuple(dv_target)
    if float(getattr(cfg, 'gain_match_coef', 0.0) or 0.0) <= 0.0:
        cfg.gain_match_coef = 1.0
    if int(getattr(cfg, 'gain_match_len', 0) or 0) <= 0:
        cfg.gain_match_len = int(getattr(cfg, 'horizon', 15) or 15)
    print(f'[gain-match] targets (WM-norm) mv={cfg.gain_match_mv_target} '
          f'dv={cfg.gain_match_dv_target} coef={cfg.gain_match_coef} '
          f'len={cfg.gain_match_len}', flush=True)


def _rssm_world_model_loss(model: DreamerV4, obs_cur: torch.Tensor,
                            act: torch.Tensor, cfg: TrainConfig,
                            dist_target: Optional[torch.Tensor] = None,
                            ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor,
                                        torch.Tensor]:
    """RSSM world-model loss: reconstruction + KL-balanced free-bits.

    Mirrors the SF return signature ``(losses, z_clean, agent_hid)``:
      * ``z_clean`` → posterior stochastic features (B, T, stoch_flat) so
        any legacy consumer that expects a latent still gets one (the
        Phase-2 BC/reward heads use ``agent_hid`` instead).
      * ``agent_hid`` → posterior feature ``[h, z_flat]`` (B, T, feat_dim),
        fed to the V4 BC + reward-MTP heads unchanged.
    """
    from models.dreamer_v4_rssm import rssm_kl_loss
    rssm = model.dynamics
    feats, post_logits, prior_logits, _last, ds, cont = rssm.rollout_observed(
        obs_cur, act, sample=True)               # feats (B,T,F); ds (B,T,n_cv)|None
    recon = rssm.decode(feats)                    # (B, T, obs_dim) = g(feat)
    # DOB (neural Kalman filter): add the disturbance estimate d_t into the CV
    # channels so the recon target the decoder/dynamics ``g`` must fit becomes
    # ``obs - d_t`` — i.e. g learns the CLEAN input->CV response and d_t absorbs
    # the unmeasured load (de-confounds the omitted-variable gain attenuation).
    dob_on = bool(getattr(rssm, 'dob_enabled', False)) and ds is not None
    if dob_on:
        recon = rssm.apply_dob(recon, ds)
    recon_loss = _weighted_recon_mse(recon, obs_cur, cfg)
    kl_loss, kl_diag = rssm_kl_loss(
        post_logits, prior_logits,
        free_bits=float(getattr(cfg, 'rssm_free_bits', 1.0)),
        dyn_w=float(getattr(cfg, 'rssm_kl_dyn_w', 0.5)),
        repr_w=float(getattr(cfg, 'rssm_kl_repr_w', 0.1)))
    wm_total = cfg.recon_scale * recon_loss + kl_loss
    # ----- continuous-latent KL (gain + disturbance channels) -----
    # The Gaussian analogue of the categorical KL: trains the prior to ROLL the
    # gain (persist) + disturbance (OU) forward so imagination carries them.
    cont_kl = torch.zeros((), device=feats.device)
    cont_gain_persist = torch.zeros((), device=feats.device)
    dist_match_loss = torch.zeros((), device=feats.device)
    if cont is not None:
        from models.dreamer_v4_rssm import rssm_cont_kl_loss
        cont_kl, _cont_kl_diag = rssm_cont_kl_loss(
            cont['post_mean'], cont['post_std'],
            cont['prior_mean'], cont['prior_std'],
            free_bits=float(getattr(cfg, 'cont_free_bits', 0.5)),
            dyn_w=float(getattr(cfg, 'rssm_kl_dyn_w', 0.5)),
            repr_w=float(getattr(cfg, 'rssm_kl_repr_w', 0.1)))
        wm_total = wm_total + float(getattr(cfg, 'cont_kl_scale', 1.0)) * cont_kl
        # Gain-channel persistence: the gain block (first cont_gain_dim dims) is
        # a per-episode CONSTANT → penalise its step-to-step drift so the channel
        # holds a stable gain (separates it from the time-varying disturbance).
        gp_coef = float(getattr(cfg, 'cont_gain_persist_coef', 0.0) or 0.0)
        n_gain = int(getattr(model.dynamics, 'cont_gain_dim', 0) or 0)
        if gp_coef > 0.0 and n_gain > 0:
            g_seq = cont['sample'][..., :n_gain]           # (B, T, n_gain)
            cont_gain_persist = (g_seq[:, 1:] - g_seq[:, :-1]).pow(2).mean()
            wm_total = wm_total + gp_coef * cont_gain_persist
        # ----- C(2) disturbance-matching: supervise c_dist = true OU load -----
        # The SYMMETRIC analogue of C(1) gain-matching for the DISTURBANCE
        # block.  p138 RCA (DECISIVE probe tools/_probe_disturbance_localize):
        # the unmeasured load is OBSERVABLE (the prior-CV INNOVATION tracks the
        # true OU at det_r~0.37 = the DOB residual) but the WM posterior NEVER
        # WRITES it into the latent (held-out probe on the full [h,z,c]=0.027,
        # c_dist=0.06) because nothing supervises it: under CLOSED-LOOP control
        # the controlled CV hides the load so recon is satisfied by [h,z] alone
        # and the stop-grad read-out head is passive.  So c_dist stays a FREE OU
        # (std~2.3) the decoder uses to inject open-loop DRIFT (the over-gain).
        # FIX: pin the posterior disturbance mean to the recorded true load
        # (known in training, in the same normalized-CV units the decoder
        # consumes) so the posterior LEARNS the innovation->load inference (the
        # PRBS-confounded buffer forces the MV-residual, not the raw CV =>
        # transfers to closed loop) and the prior ROLLS the bounded OU forward
        # (feed-forward).  NON-stop-grad (shaping the latent IS the point).
        # Sim-agnostic: the MSE is normalised by the load variance => O(1) for
        # any plant.  ``dist_match_coef=0`` (default) => byte-clean no-op.
        dm_coef = float(getattr(cfg, 'dist_match_coef', 0.0) or 0.0)
        n_dist = int(getattr(model.dynamics, 'cont_dist_dim', 0) or 0)
        if (dm_coef > 0.0 and n_dist > 0 and dist_target is not None
                and cont.get('post_mean') is not None):
            n_g = int(getattr(model.dynamics, 'cont_gain_dim', 0) or 0)
            c_dist = cont['post_mean'][..., n_g:n_g + n_dist]   # (B, T, n_cv)
            dt = dist_target.to(c_dist.dtype)
            if (dt.dim() == c_dist.dim() and dt.shape[:2] == c_dist.shape[:2]
                    and dt.shape[-1] >= n_dist):
                dt = dt[..., :n_dist]
                dvar = dt.float().var().clamp_min(1e-4)
                # supervise only when the load is actually present (var>0)
                if float(dvar) > 1e-3:
                    dist_match_loss = (F.mse_loss(c_dist.float(), dt.float())
                                       / dvar)
                    wm_total = wm_total + dm_coef * dist_match_loss
    # DOB regulariser: a small L2 prior that the disturbance estimate is small
    # (the Kalman "process noise is small" assumption).  Keeps d_t from absorbing
    # MORE than the genuine unexplained residual — the model prefers to explain
    # CV movement with g (the inputs) whenever it can, using d_t only for the
    # slow unmeasured load.  ``dob_reg_coef=0`` disables the prior.
    dob_reg = torch.zeros((), device=feats.device)
    if dob_on:
        dob_reg = ds.float().pow(2).mean()
        wm_total = wm_total + float(getattr(cfg, 'dob_reg_coef', 0.0) or 0.0) * dob_reg

    # ----- (b) held-action steady-state consistency (RSSM) -----
    # P89 consolidation: the multi-step held-action ROLLOUT stationarity loss
    # (wm_held_rollout_*) supersedes this starved 1-step fixed-point term for
    # RSSM — the 1-step term only fires on NATURALLY held+settled replay steps
    # (p88c held_frac mean ~0.84%).  When held-rollout is active don't double
    # up; when it's off (legacy/paper) the 1-step term still runs unchanged.
    # The SF backbone keeps its own _sf_steady_consistency (held-rollout is a
    # no-op there), so SF is unaffected by this gate.
    steady_coef = float(getattr(cfg, 'wm_steady_consistency_coef', 0.0) or 0.0)
    _held_active = float(getattr(cfg, 'wm_held_rollout_coef', 0.0) or 0.0) > 0.0
    steady_loss = torch.zeros((), device=feats.device)
    steady_held_frac = 0.0
    if steady_coef > 0.0 and not _held_active:
        steady_loss, steady_diag = _rssm_steady_consistency(
            model, feats, obs_cur, act, cfg)
        steady_held_frac = float(steady_diag.get('wm_steady_held_frac', 0.0))
        wm_total = wm_total + steady_coef * steady_loss

    # ----- (P87) auxiliary disturbance-estimator head -----
    # Supervised regression of the hidden/unmeasured disturbance from the
    # posterior feature, shaping the latent to encode the load state (the
    # policy reads the same feature) and regularising it toward the smooth
    # slow-OU target.  Scaled by a soft WM-fidelity gate so a not-yet-
    # converged decoder is not destabilised early in P1.
    dist_term, dist_loss, dist_rmse = _disturbance_head_loss(
        model, feats, dist_target, recon_loss, cfg)
    wm_total = wm_total + dist_term

    # ----- (#2, P88) multi-step latent overshooting (RSSM) -----
    # mbrl2 real-sim perf (2026-07-08): the two multi-step aux rollouts
    # (overshoot + held) are ~73% of the WM step but supervise SLOW DC-gain /
    # drift properties.  p03 RCA (2026-07-09): the OVERSHOOT loss is the
    # OPEN-LOOP GAIN supervisor — gating it to every-other step slipped the MV
    # gain (5.7%→12.3% rel-err) and let the DV compound open-loop, so run
    # OVERSHOOT EVERY step now (the compiled ``img_rollout`` keeps it cheap).
    # The HELD-rollout (drift/stationarity, less gain-critical) stays
    # every-other for the residual speedup.
    _wm_aux_n = int(getattr(model, '_wm_aux_step', 0)) + 1
    model._wm_aux_step = _wm_aux_n
    _run_held = (_wm_aux_n % 2 == 0)
    overshoot_coef = float(getattr(cfg, 'wm_overshoot_coef', 0.0) or 0.0)
    if overshoot_coef > 0.0:
        overshoot_loss, overshoot_starts = _wm_latent_overshoot_loss(
            model, feats, obs_cur, act, cfg, recon_loss=recon_loss)
    else:
        overshoot_loss = torch.zeros((), device=feats.device)
        overshoot_starts = 0.0
    wm_total = wm_total + overshoot_coef * overshoot_loss

    # ----- (b2, P89) multi-step held-action rollout stationarity (RSSM) -----
    held_coef = float(getattr(cfg, 'wm_held_rollout_coef', 0.0) or 0.0)
    if _run_held and held_coef > 0.0:
        held_loss, _held_starts = _wm_held_rollout_stationarity_loss(
            model, feats, obs_cur, act, cfg, recon_loss=recon_loss)
    else:
        held_loss = torch.zeros((), device=feats.device)
    wm_total = wm_total + held_coef * held_loss

    # ----- C(1) gain-matching step-response asymptote (RSSM) -----
    # Supervise the WM's finite-difference step-response asymptote toward the
    # identified steady-state gain — the un-cheatable DC supervisor that pins
    # the subdominant DV gain the categorical attenuates (the continuous gain
    # channel gives the WM the un-quantized CAPACITY this loss grabs onto).
    gm_coef = float(getattr(cfg, 'gain_match_coef', 0.0) or 0.0)
    gain_match_loss, gain_match_diag = _wm_gain_match_loss(
        model, feats, obs_cur, act, cfg)
    wm_total = wm_total + gm_coef * gain_match_loss

    losses: Dict[str, torch.Tensor] = {
        'recon_loss': recon_loss,
        'sf_loss': torch.zeros((), device=feats.device),  # N/A for RSSM
        'kl_loss': kl_loss,
        'wm_total': wm_total,
        'wm_steady_loss': steady_loss.detach(),
        'wm_steady_held_frac': torch.tensor(steady_held_frac,
                                            device=feats.device),
        'disturbance_loss': dist_loss,
        'disturbance_rmse': torch.tensor(dist_rmse, device=feats.device),
        'wm_overshoot_loss': overshoot_loss.detach(),
        'wm_overshoot_starts': torch.tensor(float(overshoot_starts),
                                            device=feats.device),
        'wm_held_rollout_loss': held_loss.detach(),
        'cont_kl': cont_kl.detach(),
        'cont_gain_persist': cont_gain_persist.detach(),
        'gain_match_loss': gain_match_loss.detach(),
        'dist_match_loss': dist_match_loss.detach(),
        'dob_reg': dob_reg.detach(),
        'dob_d_absmean': (ds.abs().mean().detach() if dob_on
                          else torch.zeros((), device=feats.device)),
    }
    losses.update(kl_diag)
    losses.update(gain_match_diag)
    # Encoder-quality diagnostics on the posterior stochastic features.
    with torch.no_grad():
        # Scope 2: stochastic block only (exclude any DOB d-tail).
        _ze = rssm.deter_dim + rssm.stoch_flat_dim
        z_flat = feats[..., rssm.deter_dim:_ze]
        obs_var = obs_cur.float().var(dim=(0, 1)).mean().clamp_min(1e-8)
        z_var_per_dim = z_flat.float().var(dim=(0, 1))
        losses['encoder_var_ratio'] = (z_var_per_dim.mean() / obs_var).detach()
        s_var = z_var_per_dim.sum().clamp_min(1e-12)
        s_var2 = (z_var_per_dim.pow(2)).sum().clamp_min(1e-12)
        losses['z_eff_rank'] = (s_var * s_var / s_var2).detach()
        losses['z_dim'] = torch.tensor(float(z_flat.shape[-1]),
                                        device=feats.device)
        v_max = z_var_per_dim.max().clamp_min(1e-12)
        losses['z_alive_dims'] = (
            (z_var_per_dim > 0.01 * v_max).float().sum().detach())
    # z_clean here = posterior stochastic features (detached).
    # Scope 2: stochastic block only (exclude any DOB d-tail).
    return losses, feats[..., rssm.deter_dim:rssm.deter_dim
                         + rssm.stoch_flat_dim].detach(), feats


def world_model_loss(model: DreamerV4, batch: Dict[str, torch.Tensor],
                      cfg: TrainConfig,
                      ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor,
                                  torch.Tensor]:
    """Eq. 5 + Eq. 7. Returns (losses, z_clean, agent_hid).

    ``z_clean``  : (B, T, z_dim)  — frozen-tokenizer outputs (no MAE).
    ``agent_hid``: (B, T, d_model) — agent register hidden state from a
                    *clean* dynamics pass (used by Phase 2 BC heads).
    """
    obs = batch['obs']                     # (B, T, D)
    act = batch['act']                     # (B, T, A)
    # Phase 2 (2026-05-24): replay buffer now stores per-step obs;
    # the legacy ``obs[:, :, -1, :]`` slice is no longer needed because
    # the L (lookback) axis has been removed from the storage path.
    obs_cur = obs                          # (B, T, D)

    # ===== RSSM world-model branch =====
    # neural-apc-mbrl: 'tssm' (transformer-SSM) shares the RSSM-interface path
    # (feat=[h,z_flat], rollout_observed/img_step/decode) so it routes here too.
    if getattr(model, 'world_model_type', 'sf_transformer') in ('rssm', 'tssm'):
        return _rssm_world_model_loss(model, obs_cur, act, cfg,
                                      dist_target=batch.get('dist'))

    # Tokenizer with MAE: recon the masked obs.
    z_mae, recon = model.tokenizer.forward_with_mae(obs_cur)
    recon_loss = model.tokenizer.recon_loss(obs_cur, recon)
    # WM autoencoder lever (2026-06-09): up-weight the CV channels' recon in
    # symlog space (the SF tokenizer recon_loss is symlog-MSE).  Identity when
    # wm_recon_cv_weight == 1.0 (default) — keeps the p106 SF path unchanged.
    if float(getattr(cfg, 'wm_recon_cv_weight', 1.0) or 1.0) != 1.0:
        from models.dreamer_v4 import symlog as _symlog
        recon_loss = _weighted_recon_mse(_symlog(recon), _symlog(obs_cur), cfg)

    # Shortcut forcing target = clean tokenizer output (no MAE).
    z_clean = model.tokenizer.encode(obs_cur)            # (B, T, z_dim)
    sf_loss, sf_diag = shortcut_forcing_loss(model.dynamics,
                                                z_clean.detach(), act)

    # Compute agent_hid from a near-clean dynamics pass (τ=tau_max, d=d_min).
    # Used by the Phase 2 BC + reward MTP heads.
    # CRITICAL: τ=1.0 is OOD — sample_tau_d only emits τ ∈ {0, 1/k, ...,
    # (k-1)/k} so the max trained τ is (k_max-1)/k_max.  At τ=1.0 the
    # τ-embedding bucket is untrained and the transformer's agent_hid
    # output is effectively random (transformer trained for τ ≤ τ_max).
    # Use the highest in-grid value so the auxiliary heads see the
    # cleanest TRAINED features.
    B, T = z_clean.shape[:2]
    device = z_clean.device
    # Conditioning τ for the agent-head pass: use the cleanest TRAINED
    # grid value (k_max-1)/k_max (τ=1.0 is OOD).
    tau_max = (float(cfg.k_max) - 1.0) / float(cfg.k_max)
    tau_clean = torch.full((B, T), tau_max, device=device, dtype=z_clean.dtype)
    d_min = torch.full((B, T), 1.0 / cfg.k_max, device=device,
                        dtype=z_clean.dtype)
    out_clean = model.dynamics(z_clean, tau_clean, d_min, act)
    agent_hid = out_clean['agent_hid']

    # ----- (b) held-action steady-state consistency (transformer) -----
    steady_coef = float(getattr(cfg, 'wm_steady_consistency_coef', 0.0) or 0.0)
    steady_loss = torch.zeros((), device=device)
    steady_held_frac = 0.0
    if steady_coef > 0.0:
        steady_loss, steady_diag = _sf_steady_consistency(
            model, z_clean, obs_cur, act, cfg)
        steady_held_frac = float(steady_diag.get('wm_steady_held_frac', 0.0))

    # ----- (P87) auxiliary disturbance-estimator head (transformer) -----
    # Same auxiliary supervised head as the RSSM path, reading the agent-
    # register hidden state ``agent_hid`` (the feature the BC/reward heads
    # use).  Recon proxy for the fidelity gate is the tokenizer recon loss.
    dist_term, dist_loss, dist_rmse = _disturbance_head_loss(
        model, agent_hid, batch.get('dist'), recon_loss, cfg)

    # ----- (#2, P88) latent overshooting — RSSM-ONLY BY DESIGN ---------------
    # Parity decision (not a TODO): the SF-transformer ALREADY trains multi-step
    # prediction via its shortcut-forcing loss (the flow/x-prediction objective
    # IS a multi-step term), so an extra open-loop overshoot here would be
    # redundant.  The helper therefore no-ops for SF (returns 0).  The RSSM path
    # needs overshoot because DreamerV3 trains the prior ONE step ahead only.
    # If a future SF run shows a broken gain in ``wm_transfer_matrix`` (the
    # backbone-agnostic gain metric), add an ``imagine_next_z``-based overshoot
    # at that point.
    overshoot_coef = float(getattr(cfg, 'wm_overshoot_coef', 0.0) or 0.0)
    overshoot_loss, overshoot_starts = _wm_latent_overshoot_loss(
        model, z_clean, obs_cur, act, cfg, recon_loss=recon_loss)

    # ----- (b2, P89) held-action rollout stationarity — RSSM-ONLY (SF no-op) --
    held_coef = float(getattr(cfg, 'wm_held_rollout_coef', 0.0) or 0.0)
    held_loss, _held_starts = _wm_held_rollout_stationarity_loss(
        model, z_clean, obs_cur, act, cfg, recon_loss=recon_loss)

    losses: Dict[str, torch.Tensor] = {
        'recon_loss': recon_loss,
        'sf_loss': sf_loss,
        'wm_total': (cfg.recon_scale * recon_loss + cfg.sf_scale * sf_loss
                     + steady_coef * steady_loss + dist_term
                     + overshoot_coef * overshoot_loss
                     + held_coef * held_loss),
        'wm_steady_loss': steady_loss.detach(),
        'wm_steady_held_frac': torch.tensor(steady_held_frac, device=device),
        'disturbance_loss': dist_loss,
        'disturbance_rmse': torch.tensor(dist_rmse, device=device),
        'wm_overshoot_loss': overshoot_loss.detach(),
        'wm_overshoot_starts': torch.tensor(float(overshoot_starts),
                                            device=device),
        'wm_held_rollout_loss': held_loss.detach(),
    }
    # Encoder-quality diagnostic (2026-05-06): ratio of latent variance
    # to observation variance.  An encoder that "throws away
    # information" by averaging-out the noise will show var(z) <<
    # var(obs); a healthy encoder preserves at least the meaningful
    # signal variance.  Computed per-batch (cheap), logged every
    # ``log_every`` iters.  Heuristic interpretation:
    #   - var_ratio ≪ 0.1 → encoder is collapsing (low-rank latent)
    #   - var_ratio ≈ 0.5–2.0 → healthy bottleneck
    #   - var_ratio ≫ 5     → encoder is over-amplifying (rare)
    with torch.no_grad():
        obs_var = obs_cur.float().var(dim=(0, 1)).mean().clamp_min(1e-8)
        z_var_per_dim = z_clean.float().var(dim=(0, 1))           # (Z,)
        z_var = z_var_per_dim.mean()
        losses['encoder_var_ratio'] = (z_var / obs_var).detach()
        # Latent participation ratio (effective rank) — codebook health.
        # If a handful of z-dims carry all the variance, the latent has
        # collapsed and the dynamics module cannot express plant modes.
        # PR = (sum var)^2 / sum(var^2);  PR == Z means all dims equal;
        # PR == 1 means a single dim dominates.
        s_var = z_var_per_dim.sum().clamp_min(1e-12)
        s_var2 = (z_var_per_dim.pow(2)).sum().clamp_min(1e-12)
        losses['z_eff_rank'] = (s_var * s_var / s_var2).detach()
        losses['z_dim'] = torch.tensor(float(z_clean.shape[-1]),
                                          device=z_clean.device)
        # Count "alive" dims: variance > 1% of max-dim variance.
        v_max = z_var_per_dim.max().clamp_min(1e-12)
        losses['z_alive_dims'] = (
            (z_var_per_dim > 0.01 * v_max).float().sum().detach())
    losses.update({k: v for k, v in sf_diag.items()})
    return losses, z_clean.detach(), agent_hid


def agent_finetune_loss(model: DreamerV4, batch: Dict[str, torch.Tensor],
                         agent_hid: torch.Tensor, cfg: TrainConfig
                         ) -> Dict[str, torch.Tensor]:
    """Eq. 9: policy MTP (BC) + reward MTP, length L = ``cfg.mtp_length``.

    For each context position ``t`` we read the agent-register hidden state
    ``agent_hid[t]`` and predict the next ``L`` actions and rewards
    ``(a_{t+1..t+L}, r_{t+1..t+L})``. Per the paper this is implemented as
    one head with ``L`` parallel output projections (shared trunk).
    """
    act = batch['act']                       # (B, T, A)
    rew = batch['rew']                       # (B, T)
    B, T, A = act.shape
    L_mtp = max(1, int(cfg.mtp_length))
    # Number of context positions for which all L future targets exist.
    T_ctx = T - L_mtp
    if T_ctx <= 0:
        # Sequence too short — fall back to L=1 BC on whatever we have.
        T_ctx = max(1, T - 1)
        L_mtp_eff = min(L_mtp, T - T_ctx)
    else:
        L_mtp_eff = L_mtp

    feat = agent_hid[:, :T_ctx].reshape(-1, agent_hid.shape[-1])  # (B*T_ctx,D)

    # Build (B, T_ctx, L, A) future-action and (B, T_ctx, L) future-reward
    # tensors via a strided slice — no Python loop, fully vectorised.
    # BC predicts a_{t+1..t+L} (FUTURE actions; agent_hid[t] doesn't see them).
    fut_act = torch.stack(
        [act[:, 1 + l : 1 + l + T_ctx] for l in range(L_mtp_eff)], dim=2
    )                                                       # (B, T_ctx, L, A)
    # Reward prediction at offset l predicts r_{t+l} (CURRENT-step reward at
    # offset 0).  agent_hid[t] sees both s_t and a_t (within-step block-causal
    # attention), so r_t is conditional on the action that produced it — this
    # is what makes the predicted reward action-sensitive in P3 imagination.
    # Note shift vs. fut_act: BC predicts future actions, reward predicts the
    # reward of the just-taken action (offset 0 = current step).
    fut_rew = torch.stack(
        [rew[:, l : l + T_ctx] for l in range(L_mtp_eff)], dim=2
    )                                                       # (B, T_ctx, L)
    fut_act = fut_act.reshape(B * T_ctx, L_mtp_eff, A)
    fut_rew = fut_rew.reshape(B * T_ctx, L_mtp_eff)

    # Policy MTP (BC over L future actions).  BC is masked to EXPERT context
    # positions only: cloning the whole replay buffer (random/PRBS/baseline
    # actions) drags the policy mean toward a uniform/no-op clone and collapses
    # it.  ``batch['expert'][t]`` flags transitions produced by the APC expert;
    # only those anchor the policy mean.  When no expert steps are present in a
    # batch the BC term is exactly zero (no gradient).
    if L_mtp_eff < model.policy.mtp_length:
        # Pad target with zeros so we can use logits_mtp; mask out the pad in loss.
        pad_act = torch.zeros(B * T_ctx,
                               model.policy.mtp_length - L_mtp_eff, A,
                               device=fut_act.device, dtype=fut_act.dtype)
        fut_act_full = torch.cat([fut_act, pad_act], dim=1)
        bc_lp_full = model.policy.log_prob_of_mtp(feat, fut_act_full)
        bc_lp_pos = bc_lp_full[:, :L_mtp_eff].mean(dim=1)          # (BT,)
    else:
        bc_lp = model.policy.log_prob_of_mtp(feat, fut_act)        # (BT, L)
        bc_lp_pos = bc_lp.mean(dim=1)                              # (BT,)

    # ``batch['expert']`` is always present (TrajectoryBuffer.sample always
    # emits the channel — zeros for non-expert episodes), so the mask is the
    # single BC code path.  Align it with context positions t (predicting
    # a_{t+1..t+L}) and clone ONLY expert-flagged steps; denom>=1 keeps the
    # term well-defined (and exactly zero) when a batch has no expert steps.
    em = batch['expert'][:, :T_ctx].reshape(-1).to(bc_lp_pos.dtype)  # (BT,)
    denom = em.sum().clamp_min(1.0)
    bc_loss = -(bc_lp_pos * em).sum() / denom

    # Reward MTP (twohot CE over L future rewards)
    rew_logits_all = model.reward.forward_mtp(feat)           # (BT, L, K)
    if L_mtp_eff < model.reward.mtp_length:
        rew_logits_all = rew_logits_all[:, :L_mtp_eff]
    rew_loss_per = model.reward.loss_mtp(rew_logits_all, fut_rew)  # (BT, L)
    if bool(getattr(cfg, 'reward_head_exclude_expert', False)):
        # Train the reward head ONLY on non-expert steps so it stays
        # calibrated on the policy's true distribution (see TrainConfig).
        # ``em`` flags expert context positions; (1-em) selects the rest.
        # denom>=1 keeps it well-defined if a batch were all-expert.
        nem = (1.0 - em)                                       # (BT,)
        rew_denom = nem.sum().clamp_min(1.0)
        reward_mtp_loss = (rew_loss_per.mean(dim=1) * nem).sum() / rew_denom
    else:
        reward_mtp_loss = rew_loss_per.mean()

    total = cfg.bc_scale * bc_loss + cfg.reward_scale_loss * reward_mtp_loss
    return {
        'bc_loss': bc_loss.detach(),
        'reward_mtp_loss': reward_mtp_loss.detach(),
        # Non-detached reward MTP for callers that compose their own
        # total (e.g. P1 uses ``reward_mtp_total`` to add reward-head
        # gradient WITHOUT the BC term).  Bug fix 2026-05-07: P1 was
        # using the detached ``reward_mtp_loss`` key so the reward
        # head got ZERO gradient in P1 — visible as
        # reward_mtp_loss == constant 5.541 across all P1 iters.
        'reward_mtp_total': reward_mtp_loss,
        'agent_total': total,
    }


def expert_bc_p3_loss(model: DreamerV4, batch: Dict[str, torch.Tensor],
                       agent_hid: torch.Tensor
                       ) -> Tuple[torch.Tensor, torch.Tensor]:
    """P83: masked MSE-on-μ expert BC anchor for Phase 3.

    Regresses the policy's DETERMINISTIC mean action ``μ = tanh(mu)`` onto
    the stored expert action on expert-flagged real transitions only.  The
    feature ``agent_hid`` is the same agent-register state P2 BC consumed;
    it is DETACHED here so ONLY the policy params receive gradient — the WM
    update (``opt_world``) is fully isolated.

    MSE-on-μ (vs. masked NLL) is preferred because it (a) matches the
    deterministic-mean evaluation used to score the controller and (b) does
    not fight the policy σ — the anchor pulls the mean without collapsing
    exploration.  Returns ``(bc_loss, n_expert_steps)``; the loss is exactly
    zero (no gradient) when a batch carries no expert steps.
    """
    act = batch['act']                                   # (B, T, A)
    B, T, A = act.shape
    if T < 2:
        z = act.new_zeros(())
        return z, z
    T_ctx = T - 1
    feat = agent_hid[:, :T_ctx].reshape(-1, agent_hid.shape[-1]).detach()
    # Deterministic policy mean (offset 0) at each context state.
    mu, _log_std = model.policy.dist_params(feat)        # (B*T_ctx, A)
    det_action = torch.tanh(mu)                          # μ ∈ [-1, 1]
    # Mirror P2 BC alignment: agent_hid[t] predicts a_{t+1}; mask on the
    # expert flag at the context position t.
    tgt = act[:, 1:1 + T_ctx].reshape(-1, A).to(det_action.dtype)
    em = batch['expert'][:, :T_ctx].reshape(-1).to(det_action.dtype)  # (BT,)
    denom = em.sum().clamp_min(1.0)
    se = ((det_action - tgt) ** 2).mean(dim=-1)          # (BT,)
    bc_loss = (se * em).sum() / denom
    return bc_loss, em.sum()


# ---------------------------------------------------------------------------
# Phase 3 — Imagination training (PMPO + TD-λ)
# ---------------------------------------------------------------------------

def _realsim_actor_critic_step(model: DreamerV4, batch: Dict[str, torch.Tensor],
                                cfg: TrainConfig,
                                critic_batch: Optional[Dict[str, torch.Tensor]] = None,
                                ) -> Dict[str, torch.Tensor]:
    """Phase 3 (real-sim mode): actor-critic on REAL-environment λ-returns.

    The mbrl2 pivot: the actor is NOT trained in the world model's imagination
    (which imports the WM's gain/dynamics bias into the policy gradient — the
    p106→p143 objective-mismatch failures).  Instead the WM (RSSM) + DOB act as
    a FROZEN OBSERVER: we re-encode the on-policy real trajectory
    ``batch = {obs, act, rew}`` (collected by ``collect_episode`` on the true
    simulator, with domain randomisation) through the posterior to get the
    per-step belief ``feat = [h, z, (dv), (d_t)]``, then compute λ-returns from
    the **REAL** rewards + a bootstrapped slow critic and apply the SAME
    DreamerV3 actor/critic loss + percentile return normalisation used in
    imagination.  Only the data source changes (real vs imagined):

      * the policy gradient is now exact w.r.t. the TRUE dynamics — no model
        exploitation, no WM-@H-gain-induced under-actuation;
      * real returns are bounded by the reward function ⇒ ``return_scale``
        cannot run away (the cascade was an imagined-return artefact);
      * the scale-invariant normalisation (symlog/twohot/percentile) is what
        gives the fixed-hyperparameters-across-sims property — retained intact.

    The observer is frozen (``no_grad`` + ``detach``) so only the actor + critic
    receive gradients; the caller therefore skips the P3 world-model update.
    RSSM/TSSM only.  Returns the same diag keys the P3 logger consumes.
    """
    rssm = model.dynamics
    obs = batch['obs']                                   # (B, T, D)
    act = batch['act']                                   # (B, T, A)
    rew = batch['rew'].float()                           # (B, T)  REAL reward
    B, T = obs.shape[:2]
    device = obs.device

    # ----- FROZEN OBSERVER: real trajectory -> per-step belief feat -----
    # sample=False = the posterior MODE (certainty-equivalent belief), which is
    # deterministic + reproducible so the critic value and the actor log-prob
    # are evaluated on the SAME belief the control acts on.
    with torch.no_grad():
        feats, _pl, _prl, _last, _ds, _cont = rssm.rollout_observed(
            obs, act, sample=False)                      # (B, T, F)
    feats = feats.detach()
    feat_flat = feats.reshape(B * T, -1)

    # ----- critic value (grad) + slow-target bootstrap value (frozen) -----
    value_logits = model.value(feat_flat)                # (B*T, n_bins)
    with torch.no_grad():
        v_slow = model.target_value.expectation(
            model.target_value(feat_flat)).reshape(B, T)

    # ----- λ-returns from REAL rewards (same TD-λ recursion as imagination) --
    gamma = float(cfg.gamma)
    lam = float(cfg.gae_lambda)
    _ret_cap = _adaptive_return_cap(cfg)
    if _ret_cap is not None:
        v_slow = v_slow.clamp(-_ret_cap, _ret_cap)
    returns = torch.zeros_like(v_slow)
    returns[:, -1] = v_slow[:, -1]
    for t in reversed(range(T - 1)):
        bootstrap = (1.0 - lam) * v_slow[:, t + 1] + lam * returns[:, t + 1]
        returns[:, t] = rew[:, t] + gamma * bootstrap
    target_returns = returns.detach()
    if _ret_cap is not None:
        target_returns = target_returns.clamp(-_ret_cap, _ret_cap)

    # ----- CRITIC loss (twohot CE): DIVERSE replay states + MC-grounding -----
    # p03 RCA (2026-07-09): training the critic ONLY on the narrow on-policy
    # buffer STARVED it of state diversity — once the actor drifted to a corner
    # the buffer held only corner states, so the value head INVERTED (val
    # critic_r -0.23) → wrong-signed advantage → MV railed high.  Train the
    # CRITIC on the DIVERSE shared replay ``critic_batch`` (an action-independent
    # value baseline is UNBIASED for REINFORCE) while the ACTOR stays on-policy
    # (the advantage/log-prob below are on ``batch``).  ``critic_batch`` None =
    # legacy (critic shares the on-policy states).
    if critic_batch is not None:
        with torch.no_grad():
            _fc, _cpl, _cprl, _clast, _cds, _ccont = rssm.rollout_observed(
                critic_batch['obs'], critic_batch['act'], sample=False)
        Bc, Tc = critic_batch['obs'].shape[:2]
        feat_c = _fc.detach().reshape(Bc * Tc, -1)
        rew_c = critic_batch['rew'].float()
        with torch.no_grad():
            v_slow_c = model.target_value.expectation(
                model.target_value(feat_c)).reshape(Bc, Tc)
            if _ret_cap is not None:
                v_slow_c = v_slow_c.clamp(-_ret_cap, _ret_cap)
        value_logits_c = model.value(feat_c)
    else:
        Bc, Tc = B, T
        rew_c, v_slow_c, value_logits_c = rew, v_slow, value_logits
    # λ-return target on the critic's (replay) states
    ret_c = torch.zeros_like(v_slow_c)
    ret_c[:, -1] = v_slow_c[:, -1]
    for _tc in reversed(range(Tc - 1)):
        _bc = (1.0 - lam) * v_slow_c[:, _tc + 1] + lam * ret_c[:, _tc + 1]
        ret_c[:, _tc] = rew_c[:, _tc] + gamma * _bc
    ret_c = ret_c.detach()
    if _ret_cap is not None:
        ret_c = ret_c.clamp(-_ret_cap, _ret_cap)
    critic_loss = model.value.loss(
        value_logits_c, ret_c.reshape(-1)).mean()
    # ----- Fix 1: Monte-Carlo GROUNDING (anchor V to REAL returns) -----
    # Add a PURE discounted reward-to-go (λ=1, no bootstrap) CE so the critic is
    # pinned to realised economics and cannot drift/invert (replaces the deleted
    # imagination MC-grounding — now cleaner because real-sim gives full real
    # episodes).  ``critic_mc_grounding_coef`` (re-added).
    _mc_coef = float(getattr(cfg, 'critic_mc_grounding_coef', 0.0) or 0.0)
    if _mc_coef > 0.0:
        ret_mc = torch.zeros_like(v_slow_c)
        ret_mc[:, -1] = v_slow_c[:, -1]
        for _tm in reversed(range(Tc - 1)):
            ret_mc[:, _tm] = rew_c[:, _tm] + gamma * ret_mc[:, _tm + 1]
        ret_mc = ret_mc.detach()
        if _ret_cap is not None:
            ret_mc = ret_mc.clamp(-_ret_cap, _ret_cap)
        critic_loss = critic_loss + _mc_coef * model.value.loss(
            value_logits_c, ret_mc.reshape(-1)).mean()

    # ----- advantage + percentile return-scale normalisation — REUSED -----
    with torch.no_grad():
        v_pred = model.value.expectation(value_logits).reshape(B, T)
        adv_raw = target_returns - v_pred
        scale = model.update_return_scale(
            target_returns,
            abs_cap=float(getattr(cfg, 'return_scale_abs_cap', 500.0)),
        ).clamp_min(1.0)
    adv_flat = (adv_raw / scale).reshape(-1)
    _adv_clip = float(getattr(cfg, 'advantage_clip', 0.0) or 0.0)
    if _adv_clip > 0.0:
        adv_flat = adv_flat.clamp(-_adv_clip, _adv_clip)
    adv_flat = adv_flat.detach()

    # ----- actor loss: REINFORCE on the TAKEN real action + entropy bonus -----
    act_flat = act.reshape(B * T, -1)
    logp = model.policy.log_prob_of(feat_flat, act_flat)     # (B*T,)
    entropy = model.policy.entropy(feat_flat)                # (B*T,)
    ent_coef = float(getattr(cfg, 'pmpo_entropy_coef', 3e-4))
    actor_loss = -(adv_flat * logp).mean() - ent_coef * entropy.mean()

    # ----- diagnostics (mirror the imagination keys the P3 logger reads) -----
    with torch.no_grad():
        rew_var = rew.var().clamp_min(1e-8)
        tgt_var = target_returns.float().var().clamp_min(1e-8)
        _adv_c = (adv_raw.float() - adv_raw.float().mean()).reshape(-1)
        _corr = []
        for _ai in range(act_flat.shape[-1]):
            _a = act_flat[:, _ai].float()
            _a_c = _a - _a.mean()
            _den = (_adv_c.norm() * _a_c.norm()).clamp_min(1e-8)
            _corr.append(((_adv_c * _a_c).sum() / _den).abs())
        adv_action_corr = (torch.stack(_corr).mean()
                           if _corr else torch.zeros((), device=device))
    return {
        'actor_loss': actor_loss,
        'critic_loss': critic_loss,
        'entropy_mean': entropy.mean().detach(),
        'realsim_return_mean': target_returns.mean().detach(),
        'imagined_return_mean': target_returns.mean().detach(),
        'realsim_reward_mean': rew.mean().detach(),
        'adv_std_mean': adv_raw.std(dim=1).mean().detach(),
        'adv_global_std': adv_raw.std().detach(),
        'return_scale': scale.detach().squeeze(),
        'critic_rew_to_tgt_var': (rew_var / tgt_var).clamp_max(10.0).detach(),
        'imag_adv_action_corr': adv_action_corr.detach(),
        'actor_logp_mean': logp.mean().detach(),
        'actor_logp_std': logp.std().detach(),
        'pmpo_pos_frac': (adv_flat >= 0).float().mean().detach(),
    }


# ---------------------------------------------------------------------------
# Model + reward calibration
# ---------------------------------------------------------------------------

def build_model(cfg: TrainConfig) -> DreamerV4:
    model_cfg = DreamerV4Config(
        obs_dim=cfg.obs_dim, action_dim=cfg.action_dim, lookback=cfg.lookback,
        tok_hidden=cfg.tok_hidden, z_dim=cfg.z_dim, mae_p_max=cfg.mae_p_max,
        d_model=cfg.d_model, n_layers=cfg.n_layers, n_heads=cfg.n_heads,
        ff_mult=cfg.ff_mult, n_register=cfg.n_register,
        k_max=cfg.k_max, tau_n_bins=cfg.tau_n_bins, soft_cap=cfg.soft_cap,
        attn_impl=cfg.attn_impl,
        n_action_bins=cfg.n_action_bins,
        head_hidden=cfg.head_hidden, head_n_layers=cfg.head_n_layers,
        mtp_length=max(1, int(cfg.mtp_length)),
        policy_type=str(getattr(cfg, 'policy_type', 'continuous')),
        policy_init_log_std=float(getattr(cfg, 'policy_init_log_std', -0.5)),
        policy_log_std_min=float(getattr(cfg, 'policy_log_std_min', -2.3)),
        policy_log_std_max=float(getattr(cfg, 'policy_log_std_max', 0.0)),
        world_model_type=str(getattr(cfg, 'world_model_type', 'rssm')),
        rssm_deter_dim=int(getattr(cfg, 'rssm_deter_dim', 512)),
        rssm_n_categoricals=int(getattr(cfg, 'rssm_n_categoricals', 32)),
        rssm_n_classes=int(getattr(cfg, 'rssm_n_classes', 32)),
        rssm_embed_dim=int(getattr(cfg, 'rssm_embed_dim', 256)),
        rssm_hidden_dim=int(getattr(cfg, 'rssm_hidden_dim', 256)),
        rssm_unimix=float(getattr(cfg, 'rssm_unimix', 0.01)),
        tssm_d_model=int(getattr(cfg, 'tssm_d_model', 512)),
        tssm_n_layers=int(getattr(cfg, 'tssm_n_layers', 4)),
        tssm_n_heads=int(getattr(cfg, 'tssm_n_heads', 8)),
        tssm_max_seq_len=int(getattr(cfg, 'tssm_max_seq_len', 256)),
        disturbance_head_dim=int(getattr(cfg, 'disturbance_head_dim', 0) or 0),
        disturbance_head_hidden=int(getattr(cfg, 'disturbance_head_hidden', 0) or 0),
        disturbance_head_layers=int(getattr(cfg, 'disturbance_head_layers', 2) or 2),
        dv_dim=int(getattr(cfg, 'dv_dim', 0) or 0),
        dv_indices=tuple(getattr(cfg, 'dv_indices', ()) or ()),
        dv_feedforward=bool(getattr(cfg, 'dv_feedforward', True)),
        dob_enabled=bool(getattr(cfg, 'dob_enabled', False)),
        cv_obs_indices=tuple(getattr(cfg, 'cv_obs_indices', ()) or ()),
        dob_decay_init=float(getattr(cfg, 'dob_decay_init', 3.0)),
        dob_gain_init=float(getattr(cfg, 'dob_gain_init', -2.2)),
        cont_gain_dim=int(getattr(cfg, 'cont_gain_dim', 0) or 0),
        cont_dist_dim=int(getattr(cfg, 'cont_dist_dim', 0) or 0),
        cont_min_std=float(getattr(cfg, 'cont_min_std', 0.1)),
        cont_max_std=float(getattr(cfg, 'cont_max_std', 2.0)),
        cont_dist_deterministic_roll=bool(getattr(
            cfg, 'cont_dist_deterministic_roll', True)),
        dv_static_skip=bool(getattr(cfg, 'dv_static_skip', False)),
    )
    model = DreamerV4(model_cfg)
    # torch.compile — DEFAULT ON (2026-06-05).  Compiles the WM hot paths
    # (RSSM rollout_observed + img_step; transformer dynamics + tokenizer);
    # ``maybe_compile`` falls back to eager on any failure.  Precedence:
    # ``cfg.compile_mode`` (``DREAMER_COMPILE_MODE``) > ``DREAMER_COMPILE`` env
    # > default-on.  Disable with ``DREAMER_COMPILE=0`` / ``off`` / ``false``.
    cm = (cfg.compile_mode or '').strip()
    if cm.lower() in ('0', 'off', 'false', 'none', 'no'):
        cm = ''                                   # explicit cfg/env-mode disable
    elif not cm:
        env_cm = os.environ.get('DREAMER_COMPILE', '').strip().lower()
        if env_cm in ('0', 'off', 'false', 'no'):
            cm = ''                               # explicitly disabled
        elif env_cm and env_cm not in ('1', 'true', 'yes'):
            cm = env_cm                           # explicit mode string
        else:
            cm = 'default'                        # DEFAULT ON (unset / 1 / true)
    if cm:
        model.maybe_compile(mode=cm)
    return model


def _probe_wm_held_convergence(model, env, device, cfg: 'TrainConfig'):
    """Probe WM held-action CONVERGENCE (anti-drift) — companion to the
    correlation probe.

    The ``wm_best.pt`` fidelity score is otherwise PURELY correlation (sum of
    per-offset Pearson r + depth bonus), which is SCALE-INVARIANT: it cannot
    distinguish a WM whose imagination settles to a fixed point under a held
    action from one that DRIFTS (the exact failure the steady-state diagnostic
    reports as 0% convergence and the held-rollout loss targets).  Selecting /
    restoring ``wm_best`` on correlation alone can therefore DISCARD a drift-
    fixed WM.  This measures held-action convergence cheaply at probe cadence
    so the best-ckpt selection + the P1->P2 restore are not blind to it.

    Protocol: collect a short real segment, then from a few starts warm the
    posterior over the lookback and roll the PRIOR forward H steps under the
    last action HELD CONSTANT; a trajectory "converges" if its tail std is
    small (reuses the steady-state diagnostic's ``_convergence_stats``).
    Returns ``{wm_converge_frac, tail_drift_mean, n_starts}`` or ``None``.
    RSSM-interface only (rssm + tssm; SF -> None, the held-rollout loss is a
    no-op there anyway).  Never fatal — any failure returns ``None`` and the
    score falls back to correlation-only.
    """
    if getattr(model, 'world_model_type', 'sf_transformer') not in ('rssm', 'tssm'):
        return None
    H = int(getattr(cfg, 'horizon', 15))
    if H < 8:
        return None
    try:
        from tools.wm_steady_state_diagnostic import (
            _imagine_open_loop_rssm, _convergence_stats)
    except Exception as e:
        print(f'[wm-converge-probe] import failed: {e!r}', flush=True)
        return None
    L = int(getattr(cfg, 'lookback', 8))
    obs_dim = env.obs_dim
    action_dim = env.action_dim
    n_starts = 4
    stride = 2
    seg = L + H + n_starts * stride + 2
    rng = np.random.default_rng(20260606)
    try:
        env.reset(exploration=False)
        real_obs = np.zeros((seg, obs_dim), dtype='float32')
        real_act = np.zeros((seg, action_dim), dtype='float32')
        got = 0
        for t in range(seg):
            a = rng.uniform(-1.0, 1.0, size=(action_dim,)).astype('float32')
            ow_next, _, done, _ = env.step(a)
            real_obs[t] = np.asarray(ow_next)[-1]
            real_act[t] = a
            got = t + 1
            if done:
                break
    except Exception as e:
        print(f'[wm-converge-probe] collection failed: {e!r}', flush=True)
        return None
    if got < L + H + 2:
        return None
    eps_std = float(os.environ.get('DREAMER_WM_CONVERGE_EPS_STD', '0.05'))
    conv_flags: List[float] = []
    drifts: List[float] = []
    for i in range(n_starts):
        s = L + i * stride
        if s + H > got:
            break
        lookback_obs = real_obs[s - L:s]
        lookback_act = real_act[s - L:s]
        a_hold = real_act[s - 1]
        action_seq = np.tile(a_hold, (H, 1)).astype('float32')
        try:
            with torch.no_grad():
                traj = _imagine_open_loop_rssm(
                    model, lookback_obs, lookback_act, action_seq, H, device)
        except Exception as e:
            print(f'[wm-converge-probe] rollout failed: {e!r}', flush=True)
            return None
        cs = _convergence_stats(traj, tail_frac=0.3, eps_std=eps_std)
        conv_flags.append(1.0 if cs.get('converged') else 0.0)
        drifts.append(float(cs.get('tail_drift', float('nan'))))
    if not conv_flags:
        return None
    return {
        'wm_converge_frac': float(np.mean(conv_flags)),
        'tail_drift_mean': float(np.nanmean(drifts)),
        'n_starts': int(len(conv_flags)),
    }


def _probe_wm_fidelity(model, env, device, cfg: 'TrainConfig'):
    """Probe WM k-step fidelity.  Returns dict or ``None`` on failure.

    Output dict keys::
        H            target imagination horizon
        r_floor      threshold from DREAMER_HORIZON_R_FLOOR
        per_offset   {offset: r_mean} (sorted ascending)
        best_h       deepest offset with r_mean >= r_floor (0 if none)
        summary      human-readable string for logs
        passes_full  bool: best_h >= H (WM is reliable at full horizon)
    """
    H = int(getattr(cfg, 'horizon', 15))
    if H <= 1:
        return None
    try:
        from evaluation.diagnostics import _wm_kstep_rollout
    except Exception as e:
        print(f'[wm-fidelity-probe] import failed: {e!r}', flush=True)
        return None
    r_floor = float(os.environ.get('DREAMER_HORIZON_R_FLOOR', '0.40'))
    r_floor = float(np.clip(r_floor, 0.0, 0.95))
    try:
        wm = _wm_kstep_rollout(model, env, device,
                                k_max=H, n_starts=16, seed=20260510)
    except Exception as e:
        print(f'[wm-fidelity-probe] rollout failed: {e!r}', flush=True)
        return None
    per = wm.get('per_offset') if isinstance(wm, dict) else None
    if not per:
        return None
    candidates = []
    for k, v in per.items():
        try:
            off = int(k)
            r = float(v.get('r_mean', float('nan')))
            if np.isfinite(r):
                candidates.append((off, r))
        except Exception:
            continue
    if not candidates:
        return None
    candidates.sort()
    best_h = 0
    for off, r in candidates:
        if r >= r_floor:
            best_h = max(best_h, off)
    summary = ' '.join(f'H={o}:r={r:+.3f}' for o, r in candidates)
    result = {
        'H': H,
        'r_floor': r_floor,
        'per_offset': candidates,
        'best_h': best_h,
        'summary': summary,
        'passes_full': best_h >= H,
    }
    # GAIN-FIDELITY term (p126 RCA, 2026-06-17): the correlation + convergence +
    # recon terms are all SHAPE/scale-invariant or aggregate — none directly
    # measure whether the WM reproduces the CV's open-loop GAIN, the control-
    # relevant property the transfer matrix scores.  The wm_best pick therefore
    # rode correlation noise and the frozen WM gain bounced 0.85–0.95 run-to-run
    # (p121–p126), under-reading the DV→CV gain (~0.78) so the actor under-reacts
    # to disturbances → catastrophic CV overshoot.  Measure the CV-channel
    # std-ratio over the k-step open-loop rollout (under REAL actions + DV teacher-
    # forced): when the WM under-reads the gain, the predicted CV varies LESS than
    # the real CV → ratio < 1.  ``min(ratio, 1)`` credits a faithful/over-reading
    # gain fully and penalises only under-prediction (the actual bias).  Averaged
    # over CV channels; NaN-safe (returns None on degenerate std).  cv_obs_indices
    # is set at runtime from env.cv_indices.
    try:
        cv_idx = [int(i) for i in (getattr(cfg, 'cv_obs_indices', ()) or ())]
        po = np.asarray(wm.get('pred_obs'), dtype='float64')   # (S, K, D)
        ro = np.asarray(wm.get('real_obs'), dtype='float64')
        if cv_idx and po.ndim == 3 and po.shape == ro.shape:
            ratios = []
            for c in cv_idx:
                if 0 <= c < po.shape[-1]:
                    rs = float(ro[..., c].std())
                    ps = float(po[..., c].std())
                    if rs > 1e-6:
                        ratios.append(min(ps / rs, 1.0))
            if ratios:
                result['wm_gain_fidelity'] = float(np.mean(ratios))
    except Exception as e:
        print(f'[wm-gain-fidelity] skipped: {e!r}', flush=True)
    # P89: held-action convergence companion (anti-drift) so the wm_best score
    # is not blind to imagination drift (the r-terms above are scale-invariant).
    # Gated (DREAMER_WM_FIDELITY_CONV_PROBE), RSSM-only, never fatal.
    if os.environ.get('DREAMER_WM_FIDELITY_CONV_PROBE',
                       '1').lower() not in ('0', 'off', 'false', 'no'):
        try:
            _conv = _probe_wm_held_convergence(model, env, device, cfg)
        except Exception as e:
            print(f'[wm-converge-probe] failed: {e!r}', flush=True)
            _conv = None
        if _conv is not None:
            result['wm_converge_frac'] = _conv['wm_converge_frac']
            result['tail_drift_mean'] = _conv['tail_drift_mean']
            result['summary'] = (
                f"{summary} conv={_conv['wm_converge_frac']:.2f} "
                f"drift={_conv['tail_drift_mean']:.3f}")
    return result


def auto_tune_seed_buffer(env: 'APCEnv', cfg: TrainConfig
                            ) -> Dict[str, Dict[str, object]]:
    """Derive plant-adaptive defaults for the cold-start seed buffer.

    Three knobs are computed; each is returned with a ``source`` so the
    caller can tell which were truly auto-derived vs. fallback constants:

    * ``baseline_seed_action_std`` — std of N(0, σ) actions used to drive
      the env during seed-buffer pre-fill and reward calibration.
      Solved so one MV step at ±σ produces an *expected steady-state CV
      swing* below ``target_cv_frac`` of the average CV bound width:

          σ = clip(target_cv_frac * mean(cv_bound_width)
                    / mv_authority_to_cv,  0.01, 0.10)

      Falls back to ``0.05`` when the identifier produced no usable
      MV→CV gain (or when running on a sim that hasn't been identified).

    * ``baseline_seed_episodes`` — number of small-noise seed episodes.
      Targets ~5 %% of the replay buffer, scaled by complexity:

          n = clip(buffer_capacity_steps * 0.05 / episode_length
                    * max(1, complexity_factor),  4, 24)

      Falls back to ``max(4, buffer_capacity / (20 * episode_length))``
      when complexity isn't computable.

    * ``policy_init_log_std`` — universal generic default of ``-2.0``
      (σ ≈ 0.135 in pre-tanh space).  Not made plant-adaptive: this
      already produces small-amplitude initial actions across all
      simulators we've tested, and tying it to the action space adds
      coupling without measurable benefit.
    """
    out: Dict[str, Dict[str, object]] = {}

    # ---- baseline_seed_action_std --------------------------------------
    sigma = 0.05
    sigma_source = 'default'
    try:
        from utils.training_disturbance import compute_mv_authority_to_cv
        mv_auth = float(compute_mv_authority_to_cv(env.sim))
    except Exception:
        mv_auth = 0.0
    # Fallback: V4 plant-id writes dynamics_identification.json (no
    # underscore-suffix) which the legacy identifier glob misses.  Read
    # the file directly from the run's plant_id dir so the auto-tune is
    # plant-aware in both V3 and V4 layouts.  V4's file lacks the
    # ``mv_gain_to_cv`` summary; synthesize it from ``per_pair_estimates``
    # by averaging |amplitude / delta| across valid MV-step trials per MV
    # name.  Falls back to legacy loader output if the V4 file is absent.
    if mv_auth <= 1e-6:
        try:
            from utils.training_disturbance import compute_mv_authority_to_cv as _auth_fn
            out_dir = Path(getattr(cfg, 'out_dir', '.') or '.')
            # Walk up a few levels so BO trials (cfg.out_dir =
            # <study>/trial_NNNN/) find the shared <study>/plant_id/.
            search_roots: List[Path] = [out_dir]
            cur = out_dir
            for _ in range(4):
                if cur.parent == cur:
                    break
                cur = cur.parent
                search_roots.append(cur)
            cands: List[Path] = []
            for root in search_roots:
                cands.append(root / 'plant_id' / 'dynamics_identification.json')
                cands.append(root / 'dynamics_identification.json')
            for cand in cands:
                if not cand.exists():
                    continue
                with open(cand) as _f:
                    raw = json.load(_f) or {}
                ctx = dict(raw)
                if 'mv_gain_to_cv' not in ctx or not ctx.get('mv_gain_to_cv'):
                    gain_acc: Dict[str, List[float]] = {}
                    for est in raw.get('per_pair_estimates', []) or []:
                        if (not est.get('valid')) or est.get('input_type') != 'mv':
                            continue
                        try:
                            delta = float(est.get('delta', 0.0))
                            amp = float(est.get('amplitude', 0.0))
                        except (TypeError, ValueError):
                            continue
                        if abs(delta) < 1e-9 or not np.isfinite(amp):
                            continue
                        name = str(est.get('mv') or est.get('input') or '')
                        if not name:
                            continue
                        gain_acc.setdefault(name, []).append(abs(amp / delta))
                    if gain_acc:
                        ctx['mv_gain_to_cv'] = {
                            k: float(np.mean(v)) for k, v in gain_acc.items()
                        }
                mv_auth = float(_auth_fn(env.sim, identifier_ctx=ctx))
                if mv_auth > 1e-6:
                    break
        except Exception:
            pass
    cv_widths = []
    try:
        for lo, hi in (env.cv_norm_ranges or []):
            w = float(hi) - float(lo)
            if np.isfinite(w) and w > 0:
                cv_widths.append(w)
    except Exception:
        pass
    if mv_auth > 1e-6 and cv_widths:
        # ``target_cv_frac`` sets the seed-PRBS amplitude as a fraction of
        # the average CV-bound width.  At 0.10 (legacy) the seed buffer
        # only covered ~11 % of the MV operating band on test_sim
        # (mv_auth=25.6, cv_w=28 ⇒ σ ≈ 0.11) — too narrow to teach the
        # WM about big-swing rejection.  Bumped 2026-05-12 (run_p21
        # RCA: validation showed mv_bound_usage=0.24, agent never used
        # MV range) to 0.20: simulation-agnostic (still scales with the
        # plant's identified gain & CV width) but covers 2× more of the
        # MV range so the WM sees clean step-response transitions
        # across most of the operating band.  Override via env
        # ``SEED_TARGET_CV_FRAC``.
        target_frac = float(os.environ.get('SEED_TARGET_CV_FRAC', '0.20'))
        cv_w = float(np.mean(cv_widths))
        # Bumped 2026-05-08 (run_p7 RCA): cap raised 0.10 → 0.30 to give
        # low-MV-authority plants enough seed-buffer coverage breadth.
        # σ_max for the policy is now derived with its own independent
        # cap (see ``policy_log_std_max`` below) so the policy clamp
        # does not widen with this knob.
        sigma_seed_cap = float(os.environ.get('SEED_SIGMA_CAP', '0.30'))
        sigma = float(np.clip(target_frac * cv_w / mv_auth,
                                0.01, sigma_seed_cap))
        sigma_source = (f'mv_authority(target_cv_frac={target_frac:.2f}, '
                          f'cv_w={cv_w:.3f}, mv_auth={mv_auth:.3f}, '
                          f'cap={sigma_seed_cap:.2f})')
    out['baseline_seed_action_std'] = {
        'value': float(sigma), 'source': sigma_source,
    }

    # ---- baseline_seed_episodes ----------------------------------------
    # Scale with buffer capacity; floor 8, cap 32 (was 24 pre-2026-05-06;
    # raised to give the WM more diverse pretrain transitions).
    buf_cap = int(getattr(cfg, 'buffer_capacity_steps', 200_000) or 0)
    ep_len = max(1, int(getattr(cfg, 'episode_length', 1)))
    base = max(8, int(round(buf_cap * 0.08 / ep_len))) if buf_cap > 0 else 16
    n_eps = int(np.clip(base, 8, 32))
    n_source = (f'buffer_8pct({buf_cap}/{ep_len}={base})'
                  if buf_cap > 0 else 'default')
    out['baseline_seed_episodes'] = {
        'value': int(n_eps), 'source': n_source,
    }

    # ---- random_seed_episodes ------------------------------------------
    # Random-action episodes drive state-space coverage breadth (off-
    # policy / extreme-action transitions the WM needs to generalise).
    # Scale at ~1/3 of baseline_seed_episodes; floor 4, cap 12.
    n_rand = int(np.clip(round(n_eps / 3.0), 4, 12))
    out['random_seed_episodes'] = {
        'value': int(n_rand),
        'source': f'baseline_seed_episodes/3 (={n_eps}/3)',
    }

    # ---- exploration_seed_episodes (PRBS sweep across MV band) ---------
    # WM coverage breadth (run_p7 RCA): random-action episodes only
    # accumulate net drift slowly because Δ-style action tasks integrate
    # to near-zero.  PRBS holds an MV target for ~2τ then toggles to a
    # new random target inside ``[-prbs_seed_op_band, +prbs_seed_op_band]``,
    # forcing the WM to see clean step-response transitions across the
    # full operating range.  Scale at ~1/3 of baseline_seed (matches the
    # random_seed budget) so total seed = baseline + random + PRBS ≈
    # baseline × 1.66.
    n_prbs = int(np.clip(round(n_eps / 3.0), 4, 12))
    out['exploration_seed_episodes'] = {
        'value': int(n_prbs),
        'source': f'baseline_seed_episodes/3 (={n_eps}/3)',
    }

    # ---- policy_init_log_std (universal default) -----------------------
    # p03 RCA (2026-07-09): the actor's entropy collapsed early (locked below the
    # early-stop floor) before the critic calibrated.  Start the policy with MORE
    # exploration (σ≈0.22 vs 0.135) so the early on-policy data is diverse enough
    # to keep the (now MC-grounded, replay-trained) critic well-conditioned; the
    # plant-adaptive ``policy_log_std_max`` clamp still bounds it from above.
    out['policy_init_log_std'] = {
        'value': -1.5,
        'source': 'realsim_exploration_default(σ≈0.22)',
    }

    # ---- policy_log_std_max (plant-adaptive σ ceiling) -----------------
    # DreamerV3 §3 prescribes σ ∈ [0.1, 1.0] (⇒ log_std ∈ [-2.3, 0]).
    # The dataclass default (log_std_max = 0.0) sits at the upper end of
    # that band, which lets REINFORCE+critic-over-optimism push the
    # actor's σ all the way to ≈ 1.0 → policy ≈ uniform random across
    # the MV bound range.  Observed in run_p0adapt and run_p1: entropy
    # locked at the clamp ceiling for 100+ P3 iters.
    #
    # Narrow the ceiling to ``log(2 × baseline_seed_action_std)`` so σ
    # cannot exceed roughly 2× the operating-region scale.  This stays
    # *inside* the V3 band [-2.3, 0] (we never widen, only tighten the
    # upper bound) while preventing the saturation trap on plants
    # whose useful σ is much smaller than 1.0.
    sigma_seed = float(out.get('baseline_seed_action_std',
                                {}).get('value', 0.05))
    # Floor at σ_max = 0.30: tight enough that the policy must commit
    # to a directional mean to make progress, while still preserving
    # 3× headroom over the V3 paper σ_min = 0.1.  Cap at 1.0 (V3 paper
    # upper).  For test_sim (σ_seed=0.10): σ_max=0.30 ⇒ log_std_max
    # ≈ -1.20.
    #
    # Tuned 2026-05-08 (run_p7 RCA): σ_max=0.30 was still too wide for
    # rate-style (Δ-MV) action spaces — σ-noise (0.30) dwarfed the
    # mean signal the actor outputs (μ ≈ 0.03), drowning PG and
    # leaving the actor pinned at μ ≈ const, σ = ceiling.  Tighten
    # the formula to σ_max = 1.0 × baseline_seed_action_std (was 2.0×)
    # with floor 0.10 (was 0.30).  For test_sim (σ_seed=0.10) this
    # gives σ_max = 0.10 (was 0.30), matching the noise scale the
    # collector actually used during seeding so the actor's
    # exploration band aligns with the WM's training distribution.
    # Tightened 2026-05-18 (run_p24 RCA: σ pinned at the σ_max=0.219
    # ceiling for all 442 P3 iters, actor never committed, critic_r
    # stuck at 0.19).  Lower the multiplier 1.0 → 0.7 so the policy
    # clamp is *tighter* than the seed-buffer exploration std,
    # forcing μ-commitment.
    # Reverted 2026-05-19 (run_p26 RCA: with the reward-head fix in
    # commit bb81cb6, the audit showed the head is now real-reward-
    # correlated, so σ at the ceiling no longer reflects a critic
    # pessimism trap — it reflects the actor genuinely needing more
    # exploration to escape the −500 plateau.  Restore multiplier
    # 0.7 → 1.0 and cap 0.20 → 0.30 so σ_max ≈ σ_seed instead of
    # 0.7×σ_seed.  This puts the policy clamp at the same scale the
    # seed buffer was collected with, aligning the actor's explore
    # band with the WM's training distribution.
    # 2026-05-27 (P59 refactor): these formula inputs are now
    # TrainConfig fields (``sigma_max_mult``, ``sigma_max_floor``,
    # ``sigma_max_cap``) wired through ``ENV_OVERRIDES``.  Legacy
    # ``DREAMER_SIGMA_MAX_*`` / ``SIGMA_MAX_*`` env-vars still honoured
    # for back-compat but the canonical path is ``cfg.sigma_max_*``.
    _legacy_mult_env = os.environ.get(
        'DREAMER_SIGMA_MAX_OVER_SEED',
        os.environ.get('SIGMA_MAX_OVER_SEED', None))
    sigma_max_mult = float(_legacy_mult_env) if _legacy_mult_env is not None \
        else float(getattr(cfg, 'sigma_max_mult', 1.0))
    _legacy_floor_env = os.environ.get('SIGMA_MAX_FLOOR', None)
    sigma_max_floor = float(_legacy_floor_env) if _legacy_floor_env is not None \
        else float(getattr(cfg, 'sigma_max_floor', 0.10))
    # Cap σ_max independently of the seed-σ cap so a wide seed-buffer
    # exploration band does not propagate into a wide policy clamp.
    # History: 0.20 → 0.30 on 2026-05-12 (p21 RCA: too tight for high-
    # disturbance plants).  Lowered back 0.30 → 0.20 on 2026-05-18
    # (p24 RCA: σ-saturation trap at 0.219 prevented critic learning).
    # Restored 0.20 → 0.30 on 2026-05-19 (p26 RCA: reward-head fix
    # removed the saturation-trap mechanism; σ-saturation is now
    # benign).
    _legacy_cap_env = os.environ.get(
        'DREAMER_SIGMA_MAX_CAP',
        os.environ.get('SIGMA_MAX_CAP', None))
    sigma_max_cap = float(_legacy_cap_env) if _legacy_cap_env is not None \
        else float(getattr(cfg, 'sigma_max_cap', 0.30))
    target_sigma_max = float(np.clip(sigma_max_mult * sigma_seed,
                                       sigma_max_floor, sigma_max_cap))
    log_std_max_val = float(np.log(target_sigma_max))
    out['policy_log_std_max'] = {
        'value': log_std_max_val,
        'source': f'clip({sigma_max_mult:.1f}*baseline_seed_action_std,'
                  f'{sigma_max_floor:.2f},{sigma_max_cap:.2f})='
                  f'clip({sigma_max_mult:.1f}*{sigma_seed:.3f},'
                  f'{sigma_max_floor:.2f},{sigma_max_cap:.2f})='
                  f'log({target_sigma_max:.3f})={log_std_max_val:+.3f}',
    }

    # ---- policy_log_std_min (auto-derived from σ_max) ------------------
    # When σ_max is tightened (e.g. 0.10 for rate-style controls), the
    # dataclass default log_std_min = -2.3 (σ_min = 0.10) collides with
    # σ_max — there's no room left for the actor to express a
    # *confident* action.  Auto-derive log_std_min as
    # ``log(σ_max / sigma_min_ratio)`` so the actor always has at
    # least one decade of headroom to commit to a near-deterministic
    # action when it has learned a good μ.
    #
    # 2026-05-10 (run_p11 RCA): σ_min_ratio=5 → σ_min = σ_max / 5 was
    # too aggressive on tight σ_max regimes: under noisy critic
    # advantage the policy collapsed all the way to σ_min, killing
    # exploration and producing a near-constant deterministic actor
    # (validation policy_dist std = 0.015 across 1220 steps).  Tighten
    # to ratio=2.5 → σ_min = σ_max / 2.5 ≈ 40 % of σ_max.  This keeps
    # the actor confident-enough (still > 1 decade below the V3
    # paper's σ_max=1.0 reference) while preventing total exploration
    # collapse when the critic has not stabilised.
    # 2026-05-27 (P59 refactor): ``sigma_min_ratio`` is now a TrainConfig
    # field (default 2.5).  Legacy ``SIGMA_MIN_RATIO_OF_MAX`` env-var
    # still honoured for back-compat.
    _legacy_ratio_env = os.environ.get('SIGMA_MIN_RATIO_OF_MAX', None)
    sigma_min_ratio = max(2.0,
        float(_legacy_ratio_env) if _legacy_ratio_env is not None
        else float(getattr(cfg, 'sigma_min_ratio', 2.5)))
    target_sigma_min = target_sigma_max / sigma_min_ratio
    log_std_min_val = float(np.log(target_sigma_min))
    out['policy_log_std_min'] = {
        'value': log_std_min_val,
        'source': f'log(sigma_max/{sigma_min_ratio:.1f})='
                  f'log({target_sigma_max:.3f}/{sigma_min_ratio:.1f})='
                  f'{log_std_min_val:+.3f}',
    }

    # ---- pmpo_entropy_coef (action-scale-adaptive, V3-anchored) --------
    # DreamerV3/V4 use η = 3e-4 across all 150+ benchmark tasks where
    # the useful action σ is on the order of 1.0 (random-explore-and-
    # find-the-target tasks). For process control the useful σ is
    # 0.05-0.30, so a fixed η=3e-4 over-weights the entropy bonus and
    # pegs the actor at the σ ceiling (root-cause of run_p6 failure).
    #
    # Re-anchor V3's default at its native action scale and scale
    # linearly with the auto-tuned σ_max:
    #
    #     η = η_v3_baseline × σ_max / σ_v3_reference
    #       = 3e-4         × σ_max / 1.0
    #
    # Recovers V3 paper exactly when σ_max=1.0 (Atari/DMC scale) and
    # auto-shrinks for plants whose σ_max is tighter. For test_sim
    # (σ_max=0.30) → η = 9e-5, matching the value found by manual
    # tuning in run_p7.
    eta_v3_baseline = float(os.environ.get('PMPO_ENTROPY_COEF_BASELINE', '3e-4'))
    sigma_v3_ref = max(1e-3,
        float(os.environ.get('PMPO_ENTROPY_SIGMA_REF', '1.0')))
    eta_adaptive = eta_v3_baseline * (target_sigma_max / sigma_v3_ref)
    out['pmpo_entropy_coef'] = {
        'value': float(eta_adaptive),
        'source': f'V3_eta * sigma_max / sigma_v3_ref = '
                  f'{eta_v3_baseline:.1e} * {target_sigma_max:.3f} / '
                  f'{sigma_v3_ref:.2f} = {eta_adaptive:.2e}',
    }

    # ---- Plant timing (used by PRBS-segment derivation below) ---------
    # ``horizon`` is no longer auto-tuned here; the paper default H=15
    # (DreamerV3/V4) is used unless the caller passed an explicit value
    # via ``cfg.horizon`` or env override ``DREAMER_HORIZON``.  Removed
    # 2026-05-20 with the short-budget knob cleanup: the plant-derived
    # H=(θ+3τ)/sr formula compounded WM error over much deeper
    # imagined trajectories than the paper validates against.
    #
    # ``p3_critic_warmup_iters`` auto-tune block was removed in the
    # same cleanup.  With ``--steps`` defaulted to 1M (paper minimum
    # for control) the critic settles naturally.
    #
    # 2026-05-24 (P48 RCA): adaptive γ — bring the effective value
    # horizon 1/(1-γ) close to the imagination horizon H so the critic
    # bootstrap is not asked to estimate hundreds of unimagined steps.
    # Formula: γ = clip(1 - 1/(4·H), 0.97, 0.99).  At H=15 → γ=0.9833
    # (effective horizon 60 = 4·H).  At H=30 → γ=0.9917 (eff 120).  At
    # H=50+ approaches the paper γ=0.99 ceiling.  Sim-adaptive via H
    # (which is itself sim-adaptive via DREAMER_HORIZON / paper H=15).
    # Skipped only when the caller set ``cfg.gamma`` explicitly
    # (DREAMER_GAMMA env or constructor kwarg).
    H_for_gamma = int(getattr(cfg, 'horizon', 15) or 15)
    gamma_adaptive = float(np.clip(1.0 - 1.0 / (4.0 * H_for_gamma),
                                    0.97, 0.99))
    if 'gamma' not in getattr(cfg, '_explicit_fields', set()):
        # Do NOT set cfg.gamma here; the apply loop in ``train`` checks
        # ``cur == _AUTO_TUNE_FIELD_DEFAULTS['gamma']`` and will set it
        # iff still at dataclass default, then mark ``applied: True``.
        out['gamma'] = {
            'value': gamma_adaptive,
            'source': f'clip(1-1/(4*H), 0.97, 0.99)=clip(1-1/(4*{H_for_gamma}), '
                      f'0.97, 0.99)={gamma_adaptive:.4f} '
                      f'(eff_horizon=1/(1-γ)={1.0/(1-gamma_adaptive):.0f} steps, '
                      f'4·H target)',
        }

    # ---- WM multi-step supervision windows + return-scale cap (H-derived) ----
    # All three are functions of the imagination horizon H (= the identified
    # plant settling time, itself sim-adaptive via derive_horizon = round(
    # (θ+4τ)/sr)).  They live HERE in the SHARED auto-tune layer rather than in
    # single_run.py so that BOTH single_run.py AND workflow/bo_runner.py inherit
    # them per-plant (p124 config-audit RCA: bo_runner built TrainConfig()
    # directly and never derived these, leaving them stuck at the test-sim-
    # inappropriate dataclass constants 15 / 64 / 500).  The apply loop in
    # ``train`` sets each only when the field is still at its dataclass default,
    # so an explicit env-override / constructor value still wins.
    H_wm = int(getattr(cfg, 'horizon', 15) or 15)
    # (a) overshoot + held-rollout supervision span = one settling response so
    # the WM learns the asymptotic gain, not a truncated step.
    # NOTE (p139 RCA): extending the held-rollout to 2xH (p138 Fix B) did NOT
    # reduce the WM gain variance (its loss stayed tiny ~0.0014 — it constrains
    # only the deterministic h-state, so it misses the categorical-z + cont-c
    # open-loop drift that actually moves the decoded gain) and its premise was
    # self-contradictory (it penalised the WM's OWN legitimate slow settling
    # over [H,2H]).  Reverted to H.
    out['wm_overshoot_len'] = {
        'value': int(H_wm),
        'source': f'horizon (={H_wm}) — one settling response',
    }
    out['wm_held_rollout_len'] = {
        'value': int(H_wm),
        'source': f'horizon (={H_wm}) — one settling response',
    }
    # (b) return-scale runaway ceiling.  ``return_scale`` is the p95-p5 spread of
    # the bounded-reward λ-returns; it NORMALISES the ACTOR advantage ONLY
    # (adv_flat = adv_raw/return_scale — the critic targets are RAW symlog, they
    # do NOT see return_scale), bounded by the envelope B·H.  The cap arrests the
    # critic-pessimism RUNAWAY (return_scale → ∞ → adv → 0 → passive actor).
    # p137 RCA (2026-06-23) — the 0.12/20 ceiling was TOO AGGRESSIVE and bound
    # PREMATURELY, TRIGGERING the opposite failure (a hunting actor):
    #   · The actor turns on at the critic-warmup end (P3 iter 125).  return_scale
    #     was climbing freely (5→17, critic HEALTHY + STABLE: rew_to_tgt_var
    #     ~0.015, NOT decaying) and the raw advantage spread is large
    #     (adv_std 5-11 ≫ the DreamerV3 O(1) target).
    #   · At iter 129 return_scale SATURATED the cap (20) while adv_raw kept
    #     growing → the advantage was UNDER-normalised (frozen denominator) right
    #     as the actor began PMPO → the policy BLEW UP (actor_logp_std 0.78→7.3).
    #   · The wild actor drove the latent OFF-DISTRIBUTION → the WM imagination
    #     diverged PESSIMISTICALLY (imagined_return −44→−206) while the REAL
    #     economics IMPROVED to best-ever → the critic chased the imagined
    #     cascade (pred_target_r→0.99, rew_to_tgt_var→4e-4).  i.e. the cascade is
    #     an under-normalised-ACTOR artefact, NOT a critic-grounding failure
    #     (bumping critic_mc_grounding_coef is contraindicated — mc_rew_to_tgt_var
    #     is inherently ~1e-3 near steady state, so it weights a low-variance
    #     target; 1.0→2.0 already failed in p132/p137).
    # FIX = raise the sim-adaptive fraction 0.12→0.30 so the scale can track the
    # NATURAL return spread and normalise the advantage toward O(1) (DreamerV3-
    # faithful) AS the actor turns on → a calm, on-distribution policy → no
    # imagination cascade.  test_sim: cap 20→~50.  Even if return_scale pegs at
    # the new cap the advantage stays ≥1 (adv_raw_std/cap ≈ 1-2 — calm, NOT
    # passive).  Runaway is still bounded (higher backstop) + the cascade early-
    # stop (early_stop_cascade_min_return_scale_growth).  Sim-adaptive via H;
    # sim-agnostic via B (a dimensionless post-calibration reward bound) + the
    # 0.30 / 20 consts.  Short-horizon plants (0.30·B·H < 20) keep the floor 20
    # unchanged (no regression where the old cap was not the binding constraint).
    B_rs = float(getattr(cfg, 'bound_training_reward_max', 3.0) or 3.0)
    rs_cap = round(max(20.0, 0.30 * B_rs * float(H_wm)), 1)
    out['return_scale_abs_cap'] = {
        'value': float(rs_cap),
        'source': f'max(20, 0.30·B·H)=max(20, 0.30·{B_rs:g}·{H_wm})={rs_cap}',
    }

    sr = max(1, int(getattr(cfg, 'sample_rate', 1)))
    tau_plant = 0.0
    theta_plant = 0.0
    try:
        out_dir_h = Path(getattr(cfg, 'out_dir', '.') or '.')
        roots: List[Path] = [out_dir_h]
        cur_h = out_dir_h
        for _ in range(4):
            if cur_h.parent == cur_h:
                break
            cur_h = cur_h.parent
            roots.append(cur_h)
        cands_h: List[Path] = []
        for root in roots:
            cands_h.append(root / 'run_plan.json')
            cands_h.append(root / 'plant_id' / 'dynamics_identification.json')
            cands_h.append(root / 'dynamics_identification.json')
        for cand in cands_h:
            if not cand.exists():
                continue
            with open(cand) as _f:
                raw = json.load(_f) or {}
            payload = raw.get('plan', raw) if isinstance(raw, dict) else {}
            t = payload.get('tau') or payload.get('tau_dom') \
                or payload.get('tau_dominant') or 0.0
            d = payload.get('dead_time') or payload.get('dead_dom') \
                or payload.get('theta') or 0.0
            try:
                tau_plant = float(t or 0.0)
                theta_plant = float(d or 0.0)
            except (TypeError, ValueError):
                tau_plant = 0.0
                theta_plant = 0.0
            if tau_plant > 0.0:
                break
    except Exception:
        pass

    # ---- prbs_seed_segment_steps (full transfer-function coverage) ----
    # PRBS segment must be long enough that, after the dead-time delay
    # θ, the plant reaches an *observable steady state* before the next
    # MV step toggles.  For a first-order plant: 1τ = 63 %, 2τ = 86 %,
    # 3τ = 95 %, 4τ = 98 %, 5τ = 99 %.  We use 4τ + θ so the WM sees
    # both the transient and a clear settled value at every operating
    # point — critical to learn the *gain* of the transfer function
    # rather than just the early transient slope (which is dominated
    # by the actuator τ_fast).
    #
    # Episode-length cap (T/4) ensures ≥ 4 segments per episode so
    # the PRBS still toggles often enough to populate the buffer
    # with diverse operating points within the episode budget.
    if tau_plant > 0.0:
        ep_len = max(1, int(getattr(cfg, 'episode_length', 1)))
        seg_target = (theta_plant + 4.0 * tau_plant) / float(sr)
        seg_min_pgate = int(os.environ.get('PRBS_SEG_MIN', '8'))
        seg_cap = max(seg_min_pgate + 1, ep_len // 4)
        seg_auto = int(np.clip(round(seg_target), seg_min_pgate, seg_cap))
        # Multi-timescale PRBS: fast hold ~ τ / 3 / sr.  This excites
        # the WM at the dominant pole's natural frequency so it learns
        # the *transient* dynamics (not just steady-state gain).
        # Floor 2 (need at least 2 steps for a settled action).
        seg_min_floor = int(os.environ.get('PRBS_SEG_MIN_FLOOR', '2'))
        seg_min_target = (tau_plant / 3.0) / float(sr)
        seg_min_auto = int(np.clip(
            round(seg_min_target), seg_min_floor,
            max(seg_min_floor + 1, seg_auto - 1)))
        # Estimated # segments under log-uniform mix:
        #   E[seg_len] = (seg_max - seg_min) / log(seg_max/seg_min)
        if seg_min_auto < seg_auto:
            mean_seg = max(1.0, (seg_auto - seg_min_auto) /
                           max(1e-6, float(np.log(seg_auto / seg_min_auto))))
        else:
            mean_seg = float(seg_auto)
        n_seg_per_ep = max(1, int(round(ep_len / mean_seg)))
        out['prbs_seed_segment_steps'] = {
            'value': int(seg_auto),
            'source': (f'clip(round((theta+4*tau)/sr),{seg_min_pgate},'
                       f'episode_length/4)='
                       f'clip(round(({theta_plant:.1f}+4*{tau_plant:.1f})/'
                       f'{sr}),{seg_min_pgate},{seg_cap})={seg_auto} '
                       f'(slow hold; multi-timescale mix '
                       f'[{seg_min_auto}..{seg_auto}] gives '
                       f'~{n_seg_per_ep} segments/episode)'),
        }
        out['prbs_seed_segment_steps_min'] = {
            'value': int(seg_min_auto),
            'source': (f'clip(round(tau/(3*sr)),{seg_min_floor},'
                       f'seg_max-1)='
                       f'clip(round({tau_plant:.1f}/(3*{sr})),'
                       f'{seg_min_floor},{seg_auto - 1})={seg_min_auto} '
                       f'(fast-hold for transient excitation)'),
        }
    return out


# Dataclass defaults captured for sentinel detection in ``train``.  When
# a cfg field still equals its dataclass default after env-var injection,
# auto-tune is allowed to overwrite it; user/env overrides survive.
_AUTO_TUNE_FIELD_DEFAULTS: Dict[str, object] = {
    'baseline_seed_action_std': TrainConfig().baseline_seed_action_std,
    'baseline_seed_episodes':   TrainConfig().baseline_seed_episodes,
    'random_seed_episodes':     TrainConfig().random_seed_episodes,
    'exploration_seed_episodes': TrainConfig().exploration_seed_episodes,
    'policy_init_log_std':      TrainConfig().policy_init_log_std,
    'policy_log_std_max':       TrainConfig().policy_log_std_max,
    'policy_log_std_min':       TrainConfig().policy_log_std_min,
    'pmpo_entropy_coef':        TrainConfig().pmpo_entropy_coef,
    'prbs_seed_segment_steps':  TrainConfig().prbs_seed_segment_steps,
    'prbs_seed_segment_steps_min': TrainConfig().prbs_seed_segment_steps_min,
    # P48 (2026-05-24) structural γ/H mismatch fix: γ adaptive to horizon.
    'gamma':                    TrainConfig().gamma,
    # p124 config-audit (2026-06-15): H-derived, moved from single_run.py into
    # the shared auto-tune so bo_runner inherits them per-plant too.
    'wm_overshoot_len':         TrainConfig().wm_overshoot_len,
    'wm_held_rollout_len':      TrainConfig().wm_held_rollout_len,
    'return_scale_abs_cap':     TrainConfig().return_scale_abs_cap,
}


# --- Calibration robustness constants (plant-agnostic) ---
# Cap the calibration exploration noise *inside* the calibrator regardless
# of the caller's ``baseline_action_std`` (often = policy σ ceiling, which
# is large enough on tight-bound plants to drive 100% violation during
# calibration).  Values are in normalized action space [-1, +1].
_REWARD_CAL_SIGMA_LADDER: Tuple[float, ...] = (0.05, 0.02, 0.005, 0.0)
# A calibration sample is "violation-dominated" when the mean of raw
# rewards is closer to the cliff (-p95(|raw|)) than to zero.  Equivalent
# to: the typical sample is below the midpoint between 0 and the cliff.
# Ratio is dimensionless ⇒ generalises across any reward magnitude.
_REWARD_CAL_VIOLATION_RATIO: float = 0.5


def _collect_calibration_rewards(env: 'APCEnv', rng: np.random.Generator,
                                  n_steps: int, mode: str,
                                  sigma: float
                                  ) -> Tuple[List[float], List[np.ndarray]]:
    """Roll out ``n_steps`` of ``env`` under small-σ baseline (or random)
    exploration and return ``(raw_rewards, obs_trace)``.  Pure-collect
    helper — does not mutate ``env.reward_scale`` (caller pins it to 1.0
    first).  Used by ``calibrate_reward_scale`` for σ-ladder retries.

    2026-05-27 (P57 RCA): when ``sigma`` is small (≤ 0.01), the env
    needs ``settle_steps`` zero-action steps after each reset before
    the plant reaches its steady-state operating band.  Recording
    rewards during settling pollutes the calibration sample with
    transient violation rewards and hides the in-band-bonus mass that
    only fires once CV converges to its setpoint band.  P57 saw
    ``raw_max = +0.333`` despite a ``+1.0`` in-band bonus because the
    bonus rarely fired during settling.  Skipping settling makes the
    sample reflect the *true* operating distribution.  ``settle_steps``
    is sim-agnostic — capped by ``n_steps // 10`` and 200 so it
    scales with the calibration budget.
    """
    raw_rewards: List[float] = []
    obs_trace: List[np.ndarray] = []
    is_small_sigma = (mode != 'random' and sigma <= 0.01)
    settle_steps = min(200, max(0, int(n_steps) // 10)) if is_small_sigma else 0

    def _settle():
        if settle_steps <= 0:
            return
        a0 = np.zeros((env.action_dim,), dtype='float32')
        for _ in range(settle_steps):
            _, _, _done, _ = env.step(a0)
            if _done:
                env.reset(exploration=True)

    env.reset(exploration=True)
    _settle()
    for _ in range(int(n_steps)):
        if mode == 'random':
            a = rng.uniform(-1.0, 1.0,
                              size=(env.action_dim,)).astype('float32')
        else:  # 'baseline' (σ may be 0.0 ⇒ pure zero-action probe)
            if sigma <= 0.0:
                a = np.zeros((env.action_dim,), dtype='float32')
            else:
                a = rng.normal(0.0, float(sigma),
                                size=(env.action_dim,)).astype('float32')
                np.clip(a, -1.0, 1.0, out=a)
        obs, _, done, info = env.step(a)
        raw_rewards.append(float(info.get('raw_reward', 0.0)))
        try:
            obs_trace.append(np.asarray(obs, dtype='float32').copy())
        except Exception:
            pass
        if done:
            env.reset(exploration=True)
            _settle()
    return raw_rewards, obs_trace


def calibrate_reward_scale(env: 'APCEnv', rng: np.random.Generator,
                            n_steps: int = 3000,
                            target_std: float = 1.0,
                            min_scale: float = 1e-4,
                            max_scale: float = 1000.0,
                            mode: str = 'baseline',
                            baseline_action_std: float = 0.05,
                            target_mode: str = 'percentile',
                            target_percentile: float = 50.0,
                            target_percentile_value: float = 0.5,
                            ) -> Dict[str, float]:
    """Empirically choose a per-step reward scale to match V4's twohot range.

    ``mode='baseline'`` (default, P0 fix): drive the env with small-noise
    actions around mid-MV (``a ~ N(0, σ)`` clipped to ``[-1, 1]``) to
    sample the *operating-region* reward distribution rather than the
    violation-saturated distribution produced by aggressive exploration.

    ``mode='random'``: legacy behaviour (uniform on ``[-1, 1]``); kept
    for back-compat / debugging.

    σ-ladder (root-cause fix 2026-05-19, plant-agnostic):  the caller's
    ``baseline_action_std`` is treated as an *upper bound*; the
    calibrator iterates a fixed ladder ``(0.05, 0.02, 0.005, 0.0)``
    capped by the caller's value, and stops at the first σ whose
    rollout is *not* "violation-dominated".  A rollout is
    violation-dominated when ``|raw_mean| > 0.5 * p95(|raw|)`` — i.e.
    the typical step is closer to the cliff than to the safe region.
    This generalises to any simulator because the criterion is
    dimensionless and the σ-ladder lives in normalised action space.

    Tripwire (plant-agnostic): if *even* σ=0 (pure zero-action) yields
    a violation-dominated distribution, the simulator's ``reset()`` is
    returning out-of-spec states.  Raises ``RuntimeError`` with a
    diagnostic dict so the user fixes the sim/bounds rather than
    silently training on a degenerate reward scale.

    ``target_mode='percentile'`` (default): pick scale so that the
    ``target_percentile``-th percentile of ``|raw_reward|`` maps to
    ``target_percentile_value``.  Default p50 → 0.5 (median |reward|
    becomes symlog magnitude ≈0.5, occupying ~3 twohot bins from zero
    ⇒ operating-region span ≈6 bins, enough discrimination for the
    critic).  The historical p95→1.0 was robust against the violation
    cliff *only* when calibration sampled the operating region; the
    σ-ladder above now guarantees that, so p50 is safe and gives
    better operating-region resolution.

    ``target_mode='std'``: legacy behaviour, ``scale = target_std/std``.
    """
    if env.reward_scale != 1.0:
        env.reward_scale = 1.0
    mode = str(mode).lower()
    target_mode = str(target_mode).lower()

    # ---- σ-ladder: try progressively smaller exploration noise until
    # the calibration rollout is no longer violation-dominated.  The
    # caller's ``baseline_action_std`` caps the ladder so we never
    # explore *more* aggressively than the caller requested, only less.
    caller_sigma = max(0.0, float(baseline_action_std))
    if mode == 'random':
        # Random mode bypasses the ladder (legacy debug path).
        sigma_ladder: Tuple[float, ...] = (caller_sigma,)
    else:
        sigma_ladder = tuple(
            s for s in _REWARD_CAL_SIGMA_LADDER if s <= caller_sigma + 1e-9
        )
        if not sigma_ladder:
            sigma_ladder = (caller_sigma,)

    raw_rewards: List[float] = []
    obs_trace: List[np.ndarray] = []
    ladder_attempts: List[Dict[str, float]] = []
    sigma_used: float = float(sigma_ladder[0])
    violation_dominated = True
    for sigma in sigma_ladder:
        raw_rewards, obs_trace = _collect_calibration_rewards(
            env, rng, n_steps, mode, float(sigma))
        arr_probe = np.asarray(raw_rewards, dtype='float64')
        if arr_probe.size == 0:
            ladder_attempts.append({
                'sigma': float(sigma), 'raw_mean': 0.0,
                'raw_abs_p95': 0.0, 'violation_dominated': True,
                'reason': 'no_samples',
            })
            continue
        probe_p95 = float(np.percentile(np.abs(arr_probe), 95.0))
        probe_mean = float(arr_probe.mean())
        # Violation-dominated iff typical step is closer to cliff than zero.
        # Equivalent test: |mean| > ratio * p95(|raw|).  Dimensionless.
        vd = (probe_p95 > 1e-8) and (
            abs(probe_mean) > _REWARD_CAL_VIOLATION_RATIO * probe_p95)
        ladder_attempts.append({
            'sigma': float(sigma), 'raw_mean': probe_mean,
            'raw_abs_p95': probe_p95,
            'violation_dominated': bool(vd),
        })
        sigma_used = float(sigma)
        violation_dominated = bool(vd)
        if not vd:
            break  # success: operating-region sample obtained

    if violation_dominated and mode != 'random':
        # Tripwire: every σ on the ladder (including σ=0) put the env in
        # sustained violation ⇒ plant resets out-of-spec.  This is a
        # simulator/bounds bug, not a calibration knob.  Fail loudly so
        # the user fixes the sim or relaxes the bounds rather than
        # training on a degenerate reward scale.  Scoped to baseline
        # mode; ``mode='random'`` is a legacy debug path that may
        # legitimately produce a violation-dominated sample.
        raise RuntimeError(
            "[reward-scale] Calibration cannot find an operating-region "
            "sample: every σ on the ladder yielded a violation-dominated "
            "reward distribution (|raw_mean| > "
            f"{_REWARD_CAL_VIOLATION_RATIO} * p95(|raw|)). The plant's "
            "reset() is producing out-of-spec initial states or the "
            "control bounds are infeasible. Inspect control_objective.json "
            "(MV/CV bounds), simulator.reset(), and any setpoint "
            f"manager. Ladder diagnostics: {ladder_attempts}"
        )

    arr = np.asarray(raw_rewards, dtype='float64')
    std = float(arr.std())
    mean = float(arr.mean())
    abs_arr = np.abs(arr)
    pct_q = float(np.clip(target_percentile, 1.0, 99.9))
    p_target_abs = float(np.percentile(abs_arr, pct_q)) if abs_arr.size else 0.0
    # P62 (2026-05-28): true p95 of |raw| regardless of target_percentile,
    # plus positive-tail p95 — the positive-tail p95 is the right scale
    # for "asymmetric reward distribution" detection, since it represents
    # the magnitude of the *signal we want the critic to preserve*.
    # ``raw_abs_p95_full`` is dominated by the negative bulk on
    # mostly-negative-reward plants (e.g. test_sim) and is NOT a useful
    # asymmetry denominator on its own (P62 RCA, 2026-05-28).
    raw_abs_p95_full = float(np.percentile(abs_arr, 95.0)) if abs_arr.size else 0.0
    pos_arr = arr[arr > 0.0]
    positive_tail_p95 = (
        float(np.percentile(pos_arr, 95.0)) if pos_arr.size else 0.0
    )
    positive_fraction = (
        float(pos_arr.size) / float(arr.size) if arr.size else 0.0
    )
    if target_mode == 'std':
        if std < 1e-8:
            scale = 1.0
        else:
            scale = float(target_std / std)
    else:  # 'percentile'
        if p_target_abs < 1e-8:
            # Degenerate: fall back to std-based, then to identity.
            scale = float(target_std / std) if std >= 1e-8 else 1.0
        else:
            scale = float(target_percentile_value / p_target_abs)

    # Opportunistic bin-fill bump (root-cause fix 2026-05-22, P36 RCA):
    # the percentile target sets a floor that guarantees operating-region
    # resolution, but with a bounded raw range (e.g. when
    # ``DREAMER_REWARD_RAW_CLIP_MIN`` clips the violation tail), the
    # symlog support is under-used and most active bins cluster near zero.
    # Raise ``scale`` up to the point where the largest |raw_reward|
    # maps to symlog magnitude ``DREAMER_REWARD_CAL_TARGET_SYM_MAG``
    # (default 6.0 — well inside the [-20,+20] twohot support, with
    # ~115 bins per side at 0.157 sym-units/bin).  Never *reduces*
    # the percentile-floor scale, so operating-region resolution
    # cannot regress.
    raw_abs_max = float(abs_arr.max()) if abs_arr.size else 0.0
    target_sym_mag = float(
        os.environ.get('DREAMER_REWARD_CAL_TARGET_SYM_MAG', '6.0'))
    if raw_abs_max > 1e-8 and target_sym_mag > 0.0:
        scale_fill = (math.exp(target_sym_mag) - 1.0) / raw_abs_max
        scale = max(scale, min(scale_fill, max_scale))

    scale_unclamped = scale
    scale = float(np.clip(scale, min_scale, max_scale))
    env.reward_scale = scale
    # Saturation diagnostic: the V4 twohot support is symlog([-20,+20]).
    # If even a single per-step scaled reward exceeds symlog's mid-band,
    # the head will struggle.  symlog(x)≈18 when |x|≈6.6e7; symlog(x)≈10
    # when |x|≈2.2e4.  WARN threshold tracks the bin-fill target
    # (``target_sym_mag``) since the autoscaler now intentionally aims
    # for that magnitude; warn only when we overshoot it by 1 unit.
    raw_min = float(arr.min())
    raw_max = float(arr.max())
    scaled_min = raw_min * scale
    scaled_max = raw_max * scale
    def _symlog(x: float) -> float:
        return float(np.sign(x) * np.log1p(abs(x)))
    sym_min = _symlog(scaled_min)
    sym_max = _symlog(scaled_max)
    sym_mag = max(abs(sym_min), abs(sym_max))
    twohot_warn = sym_mag > (target_sym_mag + 1.0)
    twohot_critical = sym_mag > 15.0
    # Bin-coverage diagnostic (root-cause fix 2026-05-19): how many
    # twohot bins receive non-trivial mass under the chosen scale.
    # If top-1 bin holds >80% of mass, the head cannot discriminate
    # operating-region states and critic learning will collapse.
    bin_centers = np.linspace(-20.0, 20.0, 255)
    sym_scaled = np.sign(arr * scale) * np.log1p(np.abs(arr * scale))
    idx = np.clip(np.searchsorted(bin_centers, sym_scaled, side='left'), 1, 254)
    left = idx - 1
    right = idx
    wr = np.clip((sym_scaled - bin_centers[left]) /
                  np.maximum(bin_centers[right] - bin_centers[left], 1e-8),
                  0.0, 1.0)
    wl = 1.0 - wr
    mass = np.zeros(255, dtype='float64')
    np.add.at(mass, left, wl); np.add.at(mass, right, wr)
    mass_frac = mass / max(mass.sum(), 1e-12)
    active_bins = int((mass_frac > 1e-3).sum())
    top1_mass = float(mass_frac.max())
    bin_coverage_critical = top1_mass > 0.80
    return {
        'reward_scale': scale, 'reward_scale_unclamped': scale_unclamped,
        'raw_std': std, 'raw_mean': mean,
        'raw_min': raw_min, 'raw_max': raw_max,
        'raw_abs_p95': p_target_abs,
        'raw_abs_p95_full': raw_abs_p95_full,
        'positive_tail_p95': positive_tail_p95,
        'positive_fraction': positive_fraction,
        # P62 RCA (2026-05-28): asymmetry uses positive-tail scale (the
        # signal-to-preserve), NOT raw_abs_p95_full (which on mostly-
        # negative plants trivially yields asymmetry≈1 and disables the
        # safety net).  Falls back to |raw_max| if no positive rewards.
        'tail_asymmetry': float(
            abs(raw_min) /
            max(abs(raw_max), positive_tail_p95, 1e-12)
        ),
        'target_std': float(target_std),
        'target_mode': target_mode,
        'target_percentile': pct_q,
        'target_percentile_value': float(target_percentile_value),
        'scaled_min': scaled_min, 'scaled_max': scaled_max,
        'scaled_symlog_min': sym_min, 'scaled_symlog_max': sym_max,
        'scaled_symlog_mag': sym_mag,
        'twohot_support_warn': bool(twohot_warn),
        'twohot_support_critical': bool(twohot_critical),
        'twohot_support_warn_threshold': float(target_sym_mag + 1.0),
        'reward_cal_target_sym_mag': float(target_sym_mag),
        'twohot_active_bins': active_bins,
        'twohot_top1_mass': top1_mass,
        'twohot_bin_coverage_critical': bool(bin_coverage_critical),
        'min_scale': float(min_scale), 'max_scale': float(max_scale),
        'n_steps': int(n_steps),
        'mode': mode,
        'baseline_action_std': float(sigma_used),
        'baseline_action_std_requested': float(caller_sigma),
        'sigma_ladder_attempts': ladder_attempts,
        '_obs_trace': obs_trace,
    }


# ---------------------------------------------------------------------------
# Diagnostics plot
# ---------------------------------------------------------------------------

def _save_training_diagnostics_plot(log_path: Path, out_path: Path) -> None:
    """Plot a 2x3 diagnostic grid from train_log.jsonl.

    Panels: (1) ema_return + return_window_mean, (2) WM losses (recon/sf),
    (3) phase 2/3 losses (bc/reward_mtp/actor/critic), (4) entropy & adv_std,
    (5) grad norms (w/a/c) + skip count, (6) violations.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    rows = []
    with open(log_path, 'r') as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    if not rows:
        return
    steps = [r.get('env_steps', 0) for r in rows]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)

    ax = axes[0, 0]
    ax.plot(steps, [r.get('ema_return') for r in rows], label='ema_return', lw=1.0)
    rw = [r.get('return_window_mean') for r in rows]
    if any(v is not None for v in rw):
        ax.plot(steps, rw, label='return_window_mean', lw=1.2, color='C3')
    # Phase boundaries (vertical lines).
    phases = [r.get('phase') for r in rows]
    for i in range(1, len(phases)):
        if phases[i] is not None and phases[i] != phases[i - 1]:
            ax.axvline(steps[i], color='gray', lw=0.5, ls=':',
                        alpha=0.5)
    ax.axhline(0, color='gray', lw=0.5)
    ax.set_ylabel('return'); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax.set_title('Returns')

    ax = axes[0, 1]
    for k, c in [('recon_loss', 'C0'), ('sf_loss', 'C1')]:
        vals = [r.get(k) for r in rows]
        if any(v is not None for v in vals):
            ax.plot(steps, vals, label=k, lw=1.0, color=c)
    ax.set_yscale('symlog', linthresh=1e-3)
    ax.set_ylabel('WM losses'); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax.set_title('World model losses')

    ax = axes[0, 2]
    for k, c in [('bc_loss', 'C0'), ('reward_mtp_loss', 'C1'),
                  ('actor_loss', 'C2'), ('critic_loss', 'C3')]:
        vals = [r.get(k) for r in rows]
        if any(v is not None for v in vals):
            ax.plot(steps, vals, label=k, lw=1.0, color=c)
    ax.set_ylabel('Phase 2/3 losses'); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax.set_title('Agent / RL losses')

    ax = axes[1, 0]
    ax.plot(steps, [r.get('entropy_mean') for r in rows], label='entropy', lw=1.0)
    ax2 = ax.twinx()
    ax2.plot(steps, [r.get('adv_std_mean') for r in rows], color='C3',
              label='adv_std', lw=1.0)
    ax.set_ylabel('entropy'); ax2.set_ylabel('adv_std', color='C3')
    ax.legend(loc='upper left', fontsize=8); ax.grid(alpha=0.3)
    ax.set_title('Policy entropy / advantage std')

    ax = axes[1, 1]
    for k, c in [('wm_grad_norm', 'C0'), ('actor_grad_norm', 'C1'),
                  ('critic_grad_norm', 'C2')]:
        ax.plot(steps, [r.get(k) for r in rows], label=k, lw=1.0, color=c)
    ax.set_yscale('symlog', linthresh=1e-2)
    ax.set_ylabel('grad norm'); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    skips = [r.get('n_grad_skip', 0) for r in rows]
    ax.set_title(f'Grad norms (skip total={skips[-1] if skips else 0})')

    ax = axes[1, 2]
    cv = [r.get('iter_cv_violation_mean') for r in rows]
    mv = [r.get('iter_mv_violation_mean') for r in rows]
    _has_cv = any(v is not None for v in cv)
    _has_mv = any(v is not None for v in mv)
    if _has_cv:
        ax.plot(steps, cv, label='cv_v', lw=1.0, color='C3')
    if _has_mv:
        ax.plot(steps, mv, label='mv_v', lw=1.0, color='C1')
    ax.set_ylabel('mean violation')
    if _has_cv or _has_mv:
        ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_title('Violations (per-iter mean)')

    for ax in axes[1, :]:
        ax.set_xlabel('env_steps')
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)

    # Companion CSV with the same series (one row per train_log row).
    # Lets downstream analysis (and humans without image vision) recover the
    # panel data without re-parsing train_log.jsonl.  Columns mirror the six
    # panels above + iter/phase for context.
    try:
        import csv
        csv_path = out_path.with_suffix('.csv')
        cols = [
            'iter', 'phase', 'env_steps',
            'ema_return', 'return_window_mean',                    # panel 1
            'recon_loss', 'sf_loss',                                # panel 2
            'bc_loss', 'reward_mtp_loss', 'actor_loss', 'critic_loss',  # panel 3
            'entropy_mean', 'adv_std_mean',                         # panel 4
            'wm_grad_norm', 'actor_grad_norm', 'critic_grad_norm',  # panel 5
            'iter_cv_violation_mean', 'iter_mv_violation_mean',     # panel 6
            # Critic-cascade canary (not plotted; essential for triage).
            'critic_rew_to_tgt_var', 'critic_pred_target_r',
            'critic_target_v_r', 'imagined_return_mean', 'return_scale',
        ]
        with open(csv_path, 'w', newline='') as fh:
            w = csv.writer(fh)
            w.writerow(cols)
            for r in rows:
                w.writerow([r.get(c) for c in cols])
    except Exception as e:
        print(f'[train] training_diagnostics.csv skipped: {e!r}', flush=True)


# ---------------------------------------------------------------------------
# Main trainer (three explicit phases)
# ---------------------------------------------------------------------------

def train(cfg: TrainConfig, on_iter_end=None) -> Dict:
    rng = np.random.default_rng(int(os.environ.get('SEED', '0')))
    torch.manual_seed(int(os.environ.get('SEED', '0')))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if device.type == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision('high')
        except Exception:
            pass

    env = APCEnv(cfg, rng)
    cfg.action_dim = env.action_dim
    cfg.state_dim = env.state_dim
    cfg.aug_obs_dim = env.aug_obs_dim
    cfg.obs_dim = env.obs_dim
    # P87: size the WM disturbance-estimator head per-simulator (one output
    # per CV channel).  0 ⇒ no head (pre-P87 model).  Resolved here so every
    # downstream builder (build_model) and the replay buffer agree.
    # Scope 2 / DOB (2026-06-11): when the neural-Kalman DOB is on it REPLACES
    # the P87 read-out head — the DOB's d_t IS the disturbance estimate, now fed
    # into ``feat`` (so the actor/critic condition on it) and read directly by
    # the validation diagnostic.  Retire the redundant head so it cannot
    # trivially echo the ``d`` that is now present in ``feat`` (a feat-cheat).
    _dob_on = (bool(getattr(cfg, 'dob_enabled', False))
               and len(env.cv_indices) > 0)
    cfg.disturbance_head_dim = (int(len(env.cv_indices))
                                if (bool(getattr(cfg, 'disturbance_head', True))
                                    and not _dob_on)
                                else 0)
    # WM autoencoder lever (2026-06-09): resolve the CV obs indices per-sim so
    # the per-channel recon weight (cfg.wm_recon_cv_weight) can up-weight the
    # CV channels' reconstruction.  State channels lead the obs vector, so the
    # state-space cv_indices are valid OBS-vector indices (same as dv_indices).
    cfg.cv_obs_indices = tuple(int(i) for i in env.cv_indices)
    # DV-as-input (Option B): resolve the measured-DV obs indices per-simulator.
    # State channels lead the obs vector (obs = [state, aug, derived, integral]),
    # so the state-space ``dv_indices`` are valid OBS-vector indices.  Default ON
    # (cfg.dv_as_input); 0-DV sims or opt-out (DREAMER_DV_AS_INPUT=0) -> dv_dim=0
    # = paper behaviour (WM predicts the DV).  Resolved here so build_model + the
    # WM transition agree.
    _dv_idx_obs = [int(x) for x in (env.meta.get('dv_indices') or [])
                   if x is not None]
    if bool(getattr(cfg, 'dv_as_input', True)) and _dv_idx_obs:
        cfg.dv_indices = tuple(_dv_idx_obs)
        cfg.dv_dim = len(_dv_idx_obs)
        print(f'[dv-as-input] ENABLED: feeding measured DV channels '
              f'{cfg.dv_indices} as exogenous WM input (held constant in '
              f'imagination).', flush=True)
    else:
        cfg.dv_indices = ()
        cfg.dv_dim = 0

    # ---- Continuous gain+disturbance latent dims (2026-06-22) ----
    # Resolve from the plant when enabled: one DISTURBANCE channel per CV
    # (the amortized-Kalman, DOB-free estimator) and one GAIN channel per
    # (CV × input) where inputs = MVs + measured DVs (the un-quantized,
    # in-context, C(1)-supervised gain).  Disabled (both 0) ⇒ pre-cont model.
    if bool(getattr(cfg, 'cont_latent_enabled', False)) and len(env.cv_indices) > 0:
        _n_cv = int(len(env.cv_indices))
        _n_mv = int(len([x for x in (env.meta.get('mv_indices') or [])
                         if x is not None]))
        _n_dv = int(cfg.dv_dim)
        cfg.cont_gain_dim = _n_cv * (_n_mv + _n_dv)
        if _dob_on:
            # DOB owns the unmeasured disturbance (the classical neural-Kalman
            # observer — proven det_r 0.354, the OPTIMAL linear estimator for a
            # Gauss-Markov load).  The cont latent keeps ONLY the GAIN block (the
            # C(1) gain-match de-confounder that fixed the DV bias).  DROP the
            # cont DISTURBANCE channel + dist_match: a learned cont-disturbance
            # block would COMPETE with the DOB d_t for the SAME CV innovation
            # (the gain↔disturbance identifiability confound), and it FAILED 5
            # runs (p137-141: held-out det_r −0.05, dist_match diverged at 0.6).
            # gain_match still pins g, so d_t cleanly gets the load residual.
            cfg.cont_dist_dim = 0
            cfg.dist_match_coef = 0.0
            print(f'[cont-latent] GAIN-ONLY (DOB owns the disturbance): '
                  f'gain_dim={cfg.cont_gain_dim} '
                  f'(n_cv={_n_cv}×(n_mv={_n_mv}+n_dv={_n_dv})); cont disturbance '
                  f'channel + dist_match DISABLED (the DOB d_t is the estimator).',
                  flush=True)
        else:
            cfg.cont_dist_dim = _n_cv
            # C(2) disturbance-match auto-enable (p138 RCA): the cont disturbance
            # channel is USELESS unless supervised toward the true load (it stays a
            # free OU otherwise).  Mirror the C(1) gain-match auto-enable: turn it on
            # by default when the channel exists; user/BO override via
            # DREAMER_DIST_MATCH_COEF (sim-agnostic, variance-normalised coef).
            # NOTE (p141): 0.6 BACKFIRED (dist_match diverged, c_dist lost phase);
            # the cont disturbance direction was superseded by the DOB revert
            # (p142) — this branch is the DOB-OFF fallback only.
            if float(getattr(cfg, 'dist_match_coef', 0.0) or 0.0) <= 0.0:
                cfg.dist_match_coef = 0.6
            print(f'[cont-latent] ENABLED: gain_dim={cfg.cont_gain_dim} '
                  f'(n_cv={_n_cv}×(n_mv={_n_mv}+n_dv={_n_dv})), '
                  f'dist_dim={cfg.cont_dist_dim} (DOB-free disturbance estimator); '
                  f'dist_match_coef={cfg.dist_match_coef}.',
                  flush=True)
    else:
        cfg.cont_gain_dim = 0
        cfg.cont_dist_dim = 0

    # A' : enable potential-based reward shaping on the TRAINING env only.
    # Validation builds its own APCEnv instances (evaluation/validate.py)
    # which leave shaping OFF, so the audited economic score is unshaped.
    if float(getattr(cfg, 'reward_shaping_coef', 0.0) or 0.0) > 0.0:
        env._shaping_enabled = True
        print(f"[reward-shaping] potential-based shaping ENABLED on training "
              f"env (coef={env._shaping_coef:.3g}, γ={env._shaping_gamma:.4g}); "
              f"validation scores on unshaped economic reward.", flush=True)

    # ---- Adaptive cold-start seed-buffer knobs (P0, 2026-05-05) ----
    # Derive plant-aware defaults for the seed buffer so a fresh sim
    # works without operator intervention.  User / env-var overrides
    # take precedence (detected via dataclass-default sentinel).
    auto = auto_tune_seed_buffer(env, cfg)
    auto_summary: Dict[str, Dict[str, object]] = {}
    # 2026-05-19 fix: prefer explicit-set tracking over value-equality.
    # The legacy sentinel ``cur == default`` cannot distinguish
    # "user explicitly set the paper-default" (e.g. log_std_max=0.0)
    # from "user did not set".  ``single_run.py`` and ``_cfg_from_env``
    # now record explicitly-injected field names in ``cfg._explicit_fields``;
    # we honour that first, falling back to the value-equality check
    # for legacy entry-points that have not been migrated.
    explicit: set = set(getattr(cfg, '_explicit_fields', set()) or set())
    for field, info in auto.items():
        cur = getattr(cfg, field)
        default = _AUTO_TUNE_FIELD_DEFAULTS.get(field)
        was_explicit = field in explicit
        if was_explicit:
            auto_summary[field] = {
                'value': cur, 'source': 'user_override_explicit',
                'applied': False,
            }
            continue
        if cur == default:
            setattr(cfg, field, info['value'])
            auto_summary[field] = {**info, 'applied': True}
            print(f'[auto-tune] {field}={info["value"]} '
                  f'(source: {info["source"]})', flush=True)
        else:
            auto_summary[field] = {
                'value': cur, 'source': 'user_override', 'applied': False,
            }
    try:
        out_dir_pre = Path(cfg.out_dir or '.')
        out_dir_pre.mkdir(parents=True, exist_ok=True)
        with open(out_dir_pre / 'auto_tune_seed_buffer.json', 'w') as f:
            json.dump(auto_summary, f, indent=2)
        plan_path = out_dir_pre / 'run_plan.json'
        if plan_path.exists():
            with open(plan_path) as f:
                plan = json.load(f)
            plan['auto_tune_seed_buffer'] = auto_summary
            with open(plan_path, 'w') as f:
                json.dump(plan, f, indent=2)
    except Exception:
        pass

    # ---- Reward calibration (V4 reward head expects O(1) per-step rewards) ----
    obj_scale_env = os.environ.get('OBJ_REWARD_SCALE', 'auto').strip().lower()
    if obj_scale_env in ('', 'auto', '1', 'on', 'true'):
        cal_mode = os.environ.get('DREAMER_REWARD_CAL_MODE',
                                    'baseline').strip().lower() or 'baseline'
        cal_target_mode = os.environ.get(
            'DREAMER_REWARD_CAL_TARGET',
            'percentile').strip().lower() or 'percentile'
        try:
            cal_target_pct = float(os.environ.get(
                'DREAMER_REWARD_CAL_PCT', '95') or 95.0)
        except Exception:
            cal_target_pct = 95.0
        try:
            cal_target_pct_value = float(os.environ.get(
                'DREAMER_REWARD_CAL_PCT_VAL', '0.5') or 0.5)
        except Exception:
            cal_target_pct_value = 0.5
        # Cover at least 2 episodes so the cohort is representative
        # (paper-aligned: V3 calibrates on rolling buffer of full episodes).
        ep_len = max(1, int(getattr(cfg, 'episode_length', 1500) or 1500))
        cal_n_steps = max(3000, 2 * ep_len)
        cal = calibrate_reward_scale(
            env, rng, mode=cal_mode,
            n_steps=cal_n_steps,
            baseline_action_std=float(cfg.baseline_seed_action_std),
            target_mode=cal_target_mode,
            target_percentile=cal_target_pct,
            target_percentile_value=cal_target_pct_value,
        )
        # Pop the obs trace before serialising — kept on the dict only
        # to feed the SNR diagnostic below; np.ndarray is not JSON-safe.
        cal_obs = cal.pop('_obs_trace', [])
        # ---- P62 (2026-05-28): adaptive negative-tail reward clip ----
        # P56-P61 hypothesised that an asymmetric raw reward distribution
        # collapses the symlog-twohot critic bins and causes a bootstrap
        # cascade.  P62 RCA on test_sim FALSIFIED the bin-collapse half
        # of that hypothesis on its own plant: same raw distribution
        # (raw_min=-38, raw_max=+0.23, asymmetry|raw_max=166) but the
        # twohot calibrator reports 45 active bins, top1=17.8%,
        # scaled_symlog_mag=6.0 — the head can discriminate just fine.
        #
        # Therefore the adaptive clip is gated on THREE conditions, all
        # of which must hold to fire:
        #   (1) asymmetry > threshold                     — distribution is skewed
        #   (2) twohot is actually unhealthy              — symptom present
        #   (3) positive signal exists and clip is safe   — no information loss
        # If any condition fails, the clip is skipped with a clear reason
        # in ``adaptive_clip_skipped_reason`` so the audit trail is
        # complete.  User's explicit ``DREAMER_REWARD_RAW_CLIP_MIN``
        # always wins (highest precedence skip).
        cal['reward_clip_asymmetry_threshold'] = float(
            getattr(cfg, 'reward_clip_asymmetry_threshold', 20.0))
        cal['reward_clip_tail_k'] = float(
            getattr(cfg, 'reward_clip_tail_k', 3.0))
        cal['reward_clip_min_before'] = float(env._reward_clip_min)
        cal['reward_clip_min_user_set'] = bool(
            getattr(env, '_reward_clip_min_user_set', False))
        asym = float(cal.get('tail_asymmetry', 0.0))
        pos_p95 = float(cal.get('positive_tail_p95', 0.0))
        twohot_unhealthy = bool(
            cal.get('twohot_bin_coverage_critical', False)
            or (
                float(cal.get('twohot_top1_mass', 0.0)) > 0.50
                and int(cal.get('twohot_active_bins', 0)) < 10
            )
        )
        cal['twohot_unhealthy'] = twohot_unhealthy
        cal['adaptive_clip_triggered'] = False
        cal['adaptive_clip_skipped_reason'] = None
        if cal['reward_clip_min_user_set']:
            cal['adaptive_clip_skipped_reason'] = 'user_override_env'
        elif asym <= cal['reward_clip_asymmetry_threshold']:
            cal['adaptive_clip_skipped_reason'] = (
                f'asymmetry_{asym:.2f}_<=_threshold_'
                f'{cal["reward_clip_asymmetry_threshold"]:.2f}'
            )
        elif not twohot_unhealthy:
            cal['adaptive_clip_skipped_reason'] = (
                f'twohot_healthy(bins={cal.get("twohot_active_bins", 0)},'
                f'top1={cal.get("twohot_top1_mass", 0.0):.2f},'
                f'critical={cal.get("twohot_bin_coverage_critical", False)})'
                f'_no_symptom_to_fix'
            )
        elif pos_p95 < 1e-3:
            cal['adaptive_clip_skipped_reason'] = (
                f'no_positive_signal(positive_fraction='
                f'{cal.get("positive_fraction", 0.0):.3f},'
                f'positive_tail_p95={pos_p95:.4g})_clip_would_destroy_signal'
            )
        else:
            proposed_clip = -cal['reward_clip_tail_k'] * pos_p95
            try:
                _arr_dbg = np.asarray(
                    [float(x) for x in [
                        cal['raw_min'], cal['raw_max'], cal['raw_mean']
                    ]]
                )
                # mass test: re-derive from sigma-ladder by approximating
                # via raw_min/mean/std rather than re-sampling
                _approx_mass = float(np.clip(
                    (proposed_clip - cal['raw_mean']) /
                    max(cal['raw_std'], 1e-9),
                    -5.0, 5.0))
                # one-sided Gaussian tail approximation: phi(z)
                from math import erf, sqrt as _sqrt
                clipped_mass = 0.5 * (1.0 + erf(_approx_mass / _sqrt(2.0)))
            except Exception:
                clipped_mass = 0.0
            cal['clipped_mass_estimate'] = clipped_mass
            if clipped_mass > 0.25:
                cal['adaptive_clip_skipped_reason'] = (
                    f'clip_would_truncate_{clipped_mass*100:.1f}%_of_mass'
                    f'_(>25%_safety_limit)_proposed_clip={proposed_clip:.4g}'
                )
            else:
                env._reward_clip_min = float(proposed_clip)
                cal['adaptive_clip_triggered'] = True
                cal['reward_clip_min_after'] = float(proposed_clip)
                print(
                    f"[reward-clip] ADAPTIVE clip activated: "
                    f"asymmetry={asym:.1f} > threshold="
                    f"{cal['reward_clip_asymmetry_threshold']:.1f}; "
                    f"twohot unhealthy (bins="
                    f"{cal.get('twohot_active_bins', 0)}, top1="
                    f"{cal.get('twohot_top1_mass', 0.0):.2f}); "
                    f"positive_tail_p95={pos_p95:.4g}, k="
                    f"{cal['reward_clip_tail_k']:.2f} → "
                    f"raw_clip_min {cal['reward_clip_min_before']:.4g} → "
                    f"{proposed_clip:.4g} (est. clipped mass "
                    f"{clipped_mass*100:.1f}%).  Recalibrating "
                    f"reward_scale with clipped tail.", flush=True)
                cal_post = calibrate_reward_scale(
                    env, rng, mode=cal_mode,
                    n_steps=cal_n_steps,
                    baseline_action_std=float(cfg.baseline_seed_action_std),
                    target_mode=cal_target_mode,
                    target_percentile=cal_target_pct,
                    target_percentile_value=cal_target_pct_value,
                )
                cal_post.pop('_obs_trace', None)
                cal_post['reward_clip_asymmetry_threshold'] = cal[
                    'reward_clip_asymmetry_threshold']
                cal_post['reward_clip_tail_k'] = cal['reward_clip_tail_k']
                cal_post['reward_clip_min_before'] = cal['reward_clip_min_before']
                cal_post['reward_clip_min_after'] = float(proposed_clip)
                cal_post['reward_clip_min_user_set'] = cal['reward_clip_min_user_set']
                cal_post['adaptive_clip_triggered'] = True
                cal_post['adaptive_clip_skipped_reason'] = None
                cal_post['pre_clip_calibration'] = {
                    k: v for k, v in cal.items()
                    if k != 'pre_clip_calibration'
                }
                cal = cal_post
        if not cal['adaptive_clip_triggered']:
            print(
                f"[reward-clip] adaptive clip not applied "
                f"(reason: {cal['adaptive_clip_skipped_reason']}); "
                f"raw_clip_min={env._reward_clip_min:.4g}.",
                flush=True)
        print(f"[reward-scale] auto-calibrated ({cal['mode']}, "
              f"target={cal['target_mode']}): "
              f"scale={cal['reward_scale']:.4g}  "
              f"raw_std={cal['raw_std']:.4g} "
              f"raw_p{cal['target_percentile']:.0f}_abs={cal['raw_abs_p95']:.4g} "
              f"raw_range=[{cal['raw_min']:.4g},{cal['raw_max']:.4g}]  "
              f"scaled_symlog_mag={cal['scaled_symlog_mag']:.3f}",
              flush=True)
        if cal.get('reward_scale_unclamped') and \
                abs(cal['reward_scale_unclamped'] - cal['reward_scale']) > \
                1e-9 * max(1.0, abs(cal['reward_scale_unclamped'])):
            print(f"[reward-scale] WARNING: scale was clamped "
                  f"({cal['reward_scale_unclamped']:.4g} -> "
                  f"{cal['reward_scale']:.4g}) by "
                  f"[min_scale={cal['min_scale']:.4g}, "
                  f"max_scale={cal['max_scale']:.4g}]; "
                  f"twohot calibration may be sub-optimal.",
                  flush=True)
        if cal.get('twohot_support_critical'):
            print(f"[reward-scale] CRITICAL: scaled symlog magnitude "
                  f"{cal['scaled_symlog_mag']:.2f} > 15 — reward/critic "
                  f"twohot support [-20, +20] is being saturated. "
                  f"Reduce DREAMER_REWARD_CAL_PCT_VAL or smooth the "
                  f"reward cliff.", flush=True)
        elif cal.get('twohot_support_warn'):
            print(f"[reward-scale] WARN: scaled symlog magnitude "
                  f"{cal['scaled_symlog_mag']:.2f} > "
                  f"{cal['twohot_support_warn_threshold']:.2f} "
                  f"(target_sym_mag={cal['reward_cal_target_sym_mag']:.2f} + 1) "
                  f"— twohot head is in the high-curvature region; "
                  f"reduce DREAMER_REWARD_CAL_TARGET_SYM_MAG.", flush=True)
        # Bin-coverage tripwire (2026-05-19): if the chosen scale
        # collapses operating-region rewards into ≤1 twohot bin, the
        # critic cannot learn a useful gradient and training will
        # diverge.  Plant-agnostic check: top-1 bin mass > 80%.
        if cal.get('twohot_bin_coverage_critical'):
            print(f"[reward-scale] CRITICAL: twohot bin coverage is "
                  f"degenerate — top-1 bin holds "
                  f"{cal['twohot_top1_mass']*100:.1f}% of operating-region "
                  f"reward mass across only {cal['twohot_active_bins']} "
                  f"active bins of 255.  The reward/critic head cannot "
                  f"discriminate operating-region states.  Increase "
                  f"reward_scale (e.g. raise DREAMER_REWARD_CAL_PCT_VAL "
                  f"toward 1.0) or shrink DREAMER_REWARD_CAL_PCT.",
                  flush=True)
        else:
            print(f"[reward-scale] bin coverage: "
                  f"{cal['twohot_active_bins']} active bins, "
                  f"top-1 mass {cal['twohot_top1_mass']*100:.1f}% "
                  f"(σ_used={cal['baseline_action_std']:.4f}, "
                  f"σ_requested={cal['baseline_action_std_requested']:.4f})",
                  flush=True)
        out_dir_pre = Path(cfg.out_dir or '.')
        out_dir_pre.mkdir(parents=True, exist_ok=True)
        try:
            with open(out_dir_pre / 'reward_calibration.json', 'w') as f:
                json.dump(cal, f, indent=2)
        except Exception:
            pass
        try:
            plan_path = out_dir_pre / 'run_plan.json'
            if plan_path.exists():
                with open(plan_path) as f:
                    plan = json.load(f)
                plan['reward_calibration'] = cal
                with open(plan_path, 'w') as f:
                    json.dump(plan, f, indent=2)
        except Exception:
            pass
    elif obj_scale_env in ('off', '0', 'none', 'false'):
        env.reward_scale = 1.0
    else:
        try:
            env.reward_scale = float(obj_scale_env)
        except Exception:
            env.reward_scale = 1.0

    # ---- Channel-level SNR diagnostic (2026-05-06) ----
    # Use the obs_trace from baseline reward calibration (no extra compute).
    # SNR_per_channel ≈ var(low-freq trend = signal) / var(high-freq detail
    # = noise).  Low-freq trend = moving average over τ_dom/sample_rate
    # steps (≈ plant time-constant in agent steps).  Anything > 10× span
    # is noise-dominated; < 0.1× span is signal-buried.  Logs to
    # ``snr_diagnostic.json`` alongside reward_calibration.json.
    if 'cal_obs' not in dir():
        cal_obs = []
    if cal_obs:
        try:
            arr = np.asarray(cal_obs, dtype='float64')
            # ``obs`` returned by env.step is a (lookback, obs_dim) window;
            # collapse to per-step (T, obs_dim) by taking the latest frame.
            if arr.ndim == 3:
                arr = arr[:, -1, :]
            elif arr.ndim != 2:
                raise ValueError(f'unexpected obs_trace ndim={arr.ndim}')
            sr = max(1, int(getattr(cfg, 'sample_rate', 1)))
            tau_dom = float(os.environ.get(
                'SIM_IDENTIFIED_TAU_DOMINANT', '50') or 50)
            window = max(3, int(round(tau_dom / sr)))
            if arr.shape[0] > window + 2:
                trend = np.array([
                    np.convolve(arr[:, c],
                                  np.ones(window) / window, mode='valid')
                    for c in range(arr.shape[1])
                ]).T  # (T-window+1, obs_dim)
                # Align lengths so detail is per-step (signal_var vs
                # high-freq residual).
                aligned = arr[window - 1:, :]
                detail = aligned - trend
                signal_var = np.var(trend, axis=0)
                noise_var = np.var(detail, axis=0)
                snr_per_ch = signal_var / np.maximum(noise_var, 1e-12)
                snr_db = 10.0 * np.log10(np.maximum(snr_per_ch, 1e-12))
                # Channel names if available; otherwise just indices.
                ch_names = list(getattr(env, 'obs_channel_names', []) or [])
                snr_report = {
                    'window_steps': int(window),
                    'tau_dom': float(tau_dom),
                    'sample_rate': int(sr),
                    'per_channel': [
                        {
                            'index': int(i),
                            'name': (ch_names[i] if i < len(ch_names)
                                       else f'obs[{i}]'),
                            'signal_std': float(np.sqrt(signal_var[i])),
                            'noise_std': float(np.sqrt(noise_var[i])),
                            'snr': float(snr_per_ch[i]),
                            'snr_db': float(snr_db[i]),
                        }
                        for i in range(arr.shape[1])
                    ],
                    'snr_db_min': float(snr_db.min()),
                    'snr_db_median': float(np.median(snr_db)),
                    'snr_db_max': float(snr_db.max()),
                }
                low_ch = [
                    p for p in snr_report['per_channel']
                    if p['snr_db'] < 10.0
                ]
                summary = (f"SNR median={snr_report['snr_db_median']:+.1f}dB"
                           f" min={snr_report['snr_db_min']:+.1f}dB"
                           f" max={snr_report['snr_db_max']:+.1f}dB"
                           f" window={window}step")
                if low_ch:
                    names = ','.join(p['name'] for p in low_ch[:3])
                    summary += (f"  WARN: {len(low_ch)} ch <10dB "
                                  f"({names})")
                print(f"[snr] {summary}", flush=True)
                out_dir_pre = Path(cfg.out_dir or '.')
                with open(out_dir_pre / 'snr_diagnostic.json', 'w') as f:
                    json.dump(snr_report, f, indent=2)
        except Exception as exc:  # pragma: no cover — diagnostic only
            print(f"[snr] SKIPPED ({exc!r})", flush=True)

    # ---- C(1) gain-match target resolution (2026-06-22) ----
    # Now that obs-norm is populated (calibration ran episodes), convert the
    # identified steady-state gains into WM-normalized units for the gain-match
    # loss.  Only when the continuous gain channel is on; graceful no-op (coef
    # stays 0, the channel still trains via recon) if the gains are unavailable.
    if int(getattr(cfg, 'cont_gain_dim', 0) or 0) > 0:
        try:
            _resolve_gain_match_targets(env, cfg)
        except Exception as _gm_exc:
            print(f'[gain-match] target resolution SKIPPED ({_gm_exc!r}); '
                  f'gain_match_coef=0 (cont gain channel still trains via '
                  f'recon).', flush=True)
            cfg.gain_match_coef = 0.0

    model = build_model(cfg).to(device)

    # ---- Optional warm-start from a previous run's checkpoint ----------
    init_path = str(getattr(cfg, 'init_from_ckpt', '') or '').strip()
    if init_path:
        if not os.path.exists(init_path):
            raise FileNotFoundError(
                f'init_from_ckpt={init_path!r} does not exist')
        print(f'[init] loading model weights from {init_path}', flush=True)
        ckpt = torch.load(init_path, map_location=device, weights_only=False)
        sd = ckpt.get('model', ckpt)
        # strict=False because we may have removed parameters since the
        # checkpoint was saved (e.g. act_disc_embed in commit 1215bb3).
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing:
            print(f'[init] {len(missing)} missing keys (e.g. {missing[:3]})',
                   flush=True)
        if unexpected:
            print(f'[init] {len(unexpected)} unexpected keys '
                   f'(e.g. {unexpected[:3]})', flush=True)
        prev_iter = ckpt.get('best_iter')
        # P49: best.pt now stores ``best_det_return``; fall back to the
        # legacy ``best_ema_return`` key for ckpts produced before the
        # 2026-05-25 switch.
        prev_best = (ckpt.get('best_det_return')
                     if 'best_det_return' in ckpt
                     else ckpt.get('best_ema_return'))
        if prev_iter is not None:
            print(f'[init] resumed from iter={prev_iter} '
                   f'best_det_return={prev_best}', flush=True)

    # Square-root LR scaling for adaptive batch (kept from V3 trainer).
    bs_ref = 16
    lr_scale = math.sqrt(max(1, cfg.batch_size) / float(bs_ref)) \
                if cfg.batch_size > bs_ref else 1.0
    # DreamerV3 §3 prescribes lr_actor=3e-5 for the continuous actor.
    # The discrete categorical head historically used 1e-4 (faster
    # convergence on Atari-style sims).  Auto-pick the discrete default
    # only when ``lr_actor`` is at the V3 default — user / BO overrides
    # are respected.
    cont_default_actor_lr = 3e-5
    disc_default_actor_lr = 1e-4
    if (str(getattr(cfg, 'policy_type', 'continuous')).lower() == 'discrete'
            and abs(cfg.lr_actor - cont_default_actor_lr) < 1e-12):
        eff_lr_actor_base = disc_default_actor_lr
        print(f'[lr-actor] policy_type=discrete → auto lr_actor=' 
              f'{disc_default_actor_lr:.0e} (V3 default for categorical)',
              flush=True)
    else:
        eff_lr_actor_base = cfg.lr_actor
    eff_lr_world = cfg.lr_world * lr_scale
    eff_lr_actor = eff_lr_actor_base * lr_scale
    eff_lr_critic = cfg.lr_critic * lr_scale
    if abs(lr_scale - 1.0) > 1e-9:
        print(f'[lr-scale] batch_size={cfg.batch_size} ref={bs_ref} '
              f'-> sqrt lr_scale={lr_scale:.3f}', flush=True)

    opt_world = torch.optim.AdamW(model.parameters_world(), lr=eff_lr_world,
                                  eps=1e-8, weight_decay=0.0)
    opt_actor = torch.optim.AdamW(model.parameters_actor(), lr=eff_lr_actor,
                                  eps=1e-8, weight_decay=0.0)
    opt_critic = torch.optim.AdamW(model.parameters_critic(), lr=eff_lr_critic,
                                   eps=1e-8, weight_decay=0.0)

    capacity_eps = max(1, cfg.buffer_capacity_steps // cfg.episode_length)
    buf = TrajectoryBuffer(capacity_eps, cfg.episode_length,
                            cfg.obs_dim, cfg.action_dim,
                            n_dist=int(getattr(cfg, 'disturbance_head_dim', 0) or 0))
    # P87: bind the single training env so add_episode() can pull each
    # episode's recorded hidden-disturbance trace (zero call-site edits).
    if buf.dist is not None:
        buf._dist_source = env

    # mbrl2 real-sim (2026-07-08): a SEPARATE rolling ON-POLICY buffer for the P3
    # actor-critic update.  Vanilla REINFORCE (``_realsim_actor_critic_step``) is
    # on-policy-only, so it MUST NOT sample the shared replay buffer above (which
    # holds Phase-1/2 PRBS / random / step-test / expert seed actions — off-policy
    # actions pull the policy toward imitating the full-range excitation, the p01
    # MV-chatter RCA).  This ring buffer keeps ONLY the last
    # ``phase3_onpolicy_buffer_eps`` current-policy episodes (n_dist=0: the frozen
    # observer needs no disturbance target at P3).
    onpol_buf = None
    if str(getattr(cfg, 'actor_train_source', 'realsim')) == 'realsim':
        _onpol_eps = max(2, int(getattr(cfg, 'phase3_onpolicy_buffer_eps', 16) or 16))
        onpol_buf = TrajectoryBuffer(_onpol_eps, cfg.episode_length,
                                     cfg.obs_dim, cfg.action_dim, n_dist=0)

    out_dir = Path(cfg.out_dir or '.')
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / 'train_log.jsonl'
    log_f = open(log_path, 'a')

    # Phase budgets: use provided fracs or fall back to paper defaults.
    p1_frac = cfg.phase1_frac if cfg.phase1_frac is not None else 0.4
    p2_frac = cfg.phase2_frac if cfg.phase2_frac is not None else 0.2
    p3_frac = cfg.phase3_frac if cfg.phase3_frac is not None else 0.4
    p1 = int(p1_frac * cfg.total_steps)
    p2 = int(p2_frac * cfg.total_steps)
    p3 = cfg.total_steps - p1 - p2     # absorb rounding

    print(f"# train start: {time.strftime('%Y-%m-%d %H:%M:%S')} "
          f"device={device.type} bs={cfg.batch_size} "
          f"seq_len={cfg.seq_len} horizon={cfg.horizon} "
          f"d_model={cfg.d_model} layers={cfg.n_layers} heads={cfg.n_heads} "
          f"z_dim={cfg.z_dim} lookback={cfg.lookback} "
          f"phases={p1}/{p2}/{p3}",
          flush=True)

    total_env_steps = 0
    total_iters = 0
    p3_iters = 0
    start_time = time.time()
    last_log_time = start_time
    last_log_steps = 0
    ema_return: Optional[float] = None
    return_window: 'deque[float]' = deque(maxlen=10)
    iter_returns: List[float] = []
    iter_cv_violations: List[float] = []
    iter_mv_violations: List[float] = []
    iter_raw_returns: List[float] = []
    n_grad_skip = 0
    current_phase = 1
    t_collect_acc = 0.0
    t_sample_acc = 0.0
    t_wm_acc = 0.0
    t_ac_acc = 0.0
    # BC learning-vs-crutch tracking (2026-06-09): rolled-expert det-return
    # baseline + agent-minus-expert gap, refreshed every bc_track_expert_every
    # det-evals (0 = off).  Stashed here, logged into each train_log row.
    bc_track_expert_every = int(getattr(cfg, 'bc_track_expert_every', 0) or 0)
    _expert_eval_count = 0
    last_expert_det_return: Optional[float] = None
    last_agent_minus_expert: Optional[float] = None

    # ----- Early-stop bookkeeping -----
    es_enable = bool(getattr(cfg, 'early_stop_enable', True))
    best_p3_ema: Optional[float] = None
    best_p3_iter: int = -1
    best_ckpt_path: Optional[Path] = None
    iters_since_best: int = 0
    ent_collapse_streak: int = 0
    ent_window: List[float] = []
    adv_corr_window: List[float] = []
    critic_div_streak: int = 0
    critic_loss_window: 'deque[float]' = deque(maxlen=200)
    # 2026-05-23 (P41 RCA): bootstrap-cascade canary state.  Captured
    # at P2→P3 transition; tripped from the actor-critic log block.
    p3_start_return_scale: Optional[float] = None
    p3_start_steps: int = 0
    cascade_streak: int = 0
    grad_skip_history: 'deque[Tuple[int, int]]' = deque(maxlen=512)  # (iter, skip_count)
    grad_skip_prev_total: int = 0
    early_stop_reason: Optional[str] = None
    p1_initial_sf: Optional[float] = None
    p2_final_reward_mtp: Optional[float] = None
    mid_check_flags: List[str] = []
    # ----- WM fidelity tracking (2026-05-22, P37 RCA) -----
    # The wm-fidelity probe is the single most reliable signal of whether
    # the WM is critic-ready (sf_loss can plateau at a value that no
    # longer correlates with imagination skill).  Save a separate
    # ``wm_best.pt`` whenever the probe score improves, and trigger an
    # early-stop when the probe has not improved for ``wm_fidelity_patience_iters``
    # past a warmup of ``wm_fidelity_warmup_iters``.  P37 peaked at iter
    # 70 and degraded through iter 150; this catches that wastefully
    # long tail.
    wm_best_score: float = -1e18
    wm_best_iter: int = -1
    wm_best_ckpt_path: Optional[Path] = None
    # 2026-05-24 (P47 RCA): the raw fidelity score has empirical
    # std = 0.10–0.14 r-points across consecutive probes (measured
    # P43/P45/P47), peak-to-trough oscillation up to 0.40.  The ES
    # logic compares a single instantaneous probe against an
    # instantaneous best, so it routinely fires on within-noise
    # oscillation.  Track an EMA of the score in addition to the raw
    # value and use the EMA for ES decisions; checkpoint promotion
    # still uses raw score (so wm_best.pt captures the actual best).
    wm_score_ema: float = -1e18
    wm_score_ema_best: float = -1e18
    wm_score_ema_best_iter: int = -1
    wm_score_ema_alpha = float(
        os.environ.get('DREAMER_WM_FIDELITY_EMA_ALPHA', '0.5'))
    # P2-relative ES: reset the "best" tracker on P1→P2 entry so the
    # P2 critic head gets a fair patience window from its own best,
    # not from an unreachable P1 best (P47 RCA: iter 50 in P2 was 30
    # iters past wm_best_iter=20 in P1 → instant trip on P2 entry).
    wm_es_p2_baseline_iter: int = -1
    wm_fidelity_warmup_iters = int(
        os.environ.get('DREAMER_WM_FIDELITY_WARMUP_ITERS', '40'))
    # P47 RCA: 20 → 40 iters (4 probes) so the EMA can stabilise across
    # the natural ±0.12 noise band.  Combined with EMA smoothing,
    # genuine multi-probe degradation still trips within ~50 iters.
    wm_fidelity_patience_iters = int(
        os.environ.get('DREAMER_WM_FIDELITY_PATIENCE_ITERS', '40'))
    # ----- Phase-transition quality gates (P52 RCA, 2026-05-26) -----
    # ``phase{1,2}_env_steps`` become *lower bounds*; quality gates can
    # extend each phase up to ``(1+max_extension)`` × budget.  Gate
    # outcomes recorded in ``phase_gate_decisions`` for run_summary.
    p1_gate_wm_ema_min_v = float(getattr(cfg, 'p1_gate_wm_ema_min', 0.0))
    p1_gate_plateau_frac_v = float(getattr(cfg, 'p1_gate_plateau_frac', 0.05))
    p1_gate_plateau_probes_v = int(getattr(cfg, 'p1_gate_plateau_probes', 3))
    p1_gate_max_ext_steps = int(
        float(getattr(cfg, 'p1_gate_max_extension', 0.0)) * p1)
    p2_gate_reward_mtp_max_v = float(
        getattr(cfg, 'p2_gate_reward_mtp_max', 0.0))
    p2_gate_recent_iters_v = int(getattr(cfg, 'p2_gate_recent_iters', 5))
    p2_gate_max_ext_steps = int(
        float(getattr(cfg, 'p2_gate_max_extension', 0.0)) * p2)
    # P57 RCA: clamp p1+p2 extension caps so they cannot consume the
    # P3 budget floor.  ``phase3_min_frac`` (default 0.20) reserves
    # ≥20 % of total_steps for actor-critic training; the remaining
    # extension budget is split between p1 and p2 proportionally to
    # their configured ``max_extension`` ratios.  Sim-agnostic — no
    # plant-specific constants.
    _p3_min_frac = float(getattr(cfg, 'phase3_min_frac', 0.20))
    if _p3_min_frac > 0.0 and (p1_gate_max_ext_steps + p2_gate_max_ext_steps) > 0:
        _p3_floor = int(_p3_min_frac * cfg.total_steps)
        _ext_budget = max(0, cfg.total_steps - p1 - p2 - _p3_floor)
        _ext_requested = p1_gate_max_ext_steps + p2_gate_max_ext_steps
        if _ext_requested > _ext_budget:
            _scale = _ext_budget / max(1, _ext_requested)
            _p1_cap_new = int(p1_gate_max_ext_steps * _scale)
            _p2_cap_new = int(p2_gate_max_ext_steps * _scale)
            print(f"[gate-budget] P3-floor={_p3_floor} reserves "
                  f"{_p3_min_frac:.0%} of total_steps; clamping "
                  f"p1_ext_cap {p1_gate_max_ext_steps}→{_p1_cap_new}, "
                  f"p2_ext_cap {p2_gate_max_ext_steps}→{_p2_cap_new} "
                  f"(ext_budget={_ext_budget})", flush=True)
            p1_gate_max_ext_steps = _p1_cap_new
            p2_gate_max_ext_steps = _p2_cap_new
    p1_ext_steps = 0
    p2_ext_steps = 0
    p2_reward_mtp_recent: 'deque[float]' = deque(
        maxlen=max(1, p2_gate_recent_iters_v))
    p1_score_ema_history: List[Tuple[int, float]] = []  # (iter, ema)
    phase_gate_decisions: List[Dict] = []
    p1_gate_check_step = max(1, int(0.10 * max(1, p1)))  # 10 % steps
    # ----- P1 const-action re-injection (P39, 2026-05-22) -----
    # P38 RCA: supervised losses stay flat from iter ~25 onward but the
    # H=15 imagination-fidelity probe collapses past iter 50, and the
    # wm_steady_state_diagnostic shows 0% convergence under held actions.
    # Root cause: the front-loaded const-action seeds (40 episodes ≈
    # 49k steps) are swamped as the buffer fills with 5×+ that volume of
    # P1 random-action episodes, so the WM forgets long-horizon
    # steady-state behaviour even while its short-horizon next-state
    # loss continues to improve.  Periodically inject fresh const-action
    # episodes during P1 to keep the steady-state regime represented in
    # the buffer.  Sim-agnostic: counts are env-tunable, action levels
    # stratified within ``constant_action_seed_op_band``.
    const_inject_every = int(
        os.environ.get('DREAMER_CONST_ACTION_INJECT_EVERY', '20'))
    const_inject_n = int(
        os.environ.get('DREAMER_CONST_ACTION_INJECT_N', '5'))
    # P49 RCA (2026-05-25): WM steady-state probe still shows 0%
    # convergence under zero/constant action even after P39's periodic
    # P1 injection — because the const-action episodes are evicted
    # from the replay buffer (capacity ~65 iters worth) once P2+P3
    # collect new on-policy data.  The fix attempted in P50 was to
    # continue injecting through P2 and P3 — this was FALSIFIED:
    # persistent injection of low-reward-variance episodes during
    # critic training collapses Var(r)/Var(target_v), triggers the
    # bootstrap_cascade early-stop, and does NOT restore WM
    # steady-state convergence (still 0% in P50).  Defaults reverted
    # to OFF (P1-only injection).  Opt-in for experimentation via
    # DREAMER_CONST_ACTION_INJECT_IN_{P2,P3}=1.  See P50 RCA.
    const_inject_in_p2 = int(
        os.environ.get('DREAMER_CONST_ACTION_INJECT_IN_P2', '0'))
    const_inject_in_p3 = int(
        os.environ.get('DREAMER_CONST_ACTION_INJECT_IN_P3', '0'))
    # ----- Periodic STEP-TEST (DV-exciting) re-injection (2026-06-13) -----
    # The const-inject above replenishes only the MV exciters (const +
    # step-settle); the DV-exciting STEP-TEST episodes are seed-only.  Once the
    # buffer saturates (~iter 40 on test_sim) the isolated DV->CV step responses
    # are FIFO-evicted, so the WM's DV->CV steady-state gain DE-TRAINS over P1
    # and (in the curriculum) is frozen biased — p117: DV gain ratio 0.62 vs MV
    # 0.78, while the aggregate wm_gain_rel_err looked healthy.  This re-injects
    # step-test episodes on the same cadence so the DV gain stays supervised
    # right up to the WM freeze.  Default ON in P1 (matches const-inject), P2/P3
    # opt-in (P50 cascade caution), NO-OP when the sim has no DV channel.
    step_test_inject_every = int(
        os.environ.get('DREAMER_STEP_TEST_INJECT_EVERY', '20'))
    step_test_inject_n = int(
        os.environ.get('DREAMER_STEP_TEST_INJECT_N', '2'))
    step_test_inject_in_p2 = int(
        os.environ.get('DREAMER_STEP_TEST_INJECT_IN_P2', '0'))
    step_test_inject_in_p3 = int(
        os.environ.get('DREAMER_STEP_TEST_INJECT_IN_P3', '0'))
    # DV-PRBS re-injection (2026-06-14): keep the full-range DV sweep fresh
    # in the ring buffer through Stage 1 so the DV→CV gain stays supervised
    # right up to the WM freeze (the seed-time dv-prbs episodes are FIFO-
    # evicted by ~iter 40 otherwise, exactly the eviction that left the DV
    # gain under-identified).  Default ON in P1, P2/P3 opt-in, NO-OP when
    # the sim has no DV channel.  Cadence 20→10 on 2026-06-14 (p122 RCA): the
    # P1→P2 wm_best warm-restore keeps an early (~iter 30) checkpoint, so the
    # re-injects must land BEFORE it — every-10 fires at iter 10/20/30 (all
    # inside the kept window) instead of 20/40/60 (40/60 rolled back).
    dv_prbs_inject_every = int(
        os.environ.get('DREAMER_DV_PRBS_INJECT_EVERY', '10'))
    dv_prbs_inject_n = int(
        os.environ.get('DREAMER_DV_PRBS_INJECT_N', '2'))
    dv_prbs_inject_in_p2 = int(
        os.environ.get('DREAMER_DV_PRBS_INJECT_IN_P2', '0'))
    dv_prbs_inject_in_p3 = int(
        os.environ.get('DREAMER_DV_PRBS_INJECT_IN_P3', '0'))
    # ----- Periodic EXPERT re-injection (P81 RCA, 2026-06-03) -----
    # Same eviction failure mode as the const-action seeds above, but for
    # the objective-aligned expert demonstrations: the expert episodes are
    # added ONCE during seeding (before P1), and the 327-episode ring
    # buffer is fully lapped ~1.4x during P1's 70 iters, so ZERO expert
    # steps survive to P2 — the masked BC term reads an empty mask and
    # ``bc_loss`` is exactly 0.0 for the entire P2 phase (expert is a
    # no-op; P81 degenerates to a P80 re-run).  Fix: periodically
    # re-inject fresh expert episodes during P1+P2 so the masked BC
    # always has demonstrations to clone.  Unlike the P50 const-inject
    # cascade risk, expert episodes ride the constraint edge with healthy
    # reward variance, and BC runs only in P2 (critic trains in P3), so
    # P2 injection does not threaten Var(r)/Var(target_v).  P83 keeps a
    # decaying masked-BC anchor alive THROUGH P3, so the expert episodes
    # must also survive in the buffer during P3 — hence injection in P3 is
    # now ON by default.  Volume is kept low (3 eps every 20 iters) and the
    # episodes ride the constraint edge with healthy reward variance, so the
    # P50 const-inject cascade risk (flat steady-state collapsing Var(r))
    # does not apply.  Disable via DREAMER_EXPERT_INJECT_IN_P3=0 for ablation.
    expert_inject_every = int(
        os.environ.get('DREAMER_EXPERT_INJECT_EVERY', '20'))
    expert_inject_n = int(
        os.environ.get('DREAMER_EXPERT_INJECT_N', '3'))
    expert_inject_in_p3 = int(
        os.environ.get('DREAMER_EXPERT_INJECT_IN_P3', '1'))
    # ----- wm_best.pt warm-restore at P1→P2 (P39, 2026-05-22) -----
    # When the WM's fidelity peak is reached well before P1 ends and the
    # subsequent iters drift to a lower-quality basin (P38: peak iter 50,
    # collapse by iter 70), starting critic training from the final P1
    # weights hands P2 an already-degraded WM.  Restoring wm_best.pt at
    # the P1→P2 boundary gives critic training the cleanest available
    # latent dynamics.  Skipped when wm_best.pt is essentially the
    # current state (gap < ``min_gap``) to avoid wasted I/O.  Disable
    # with DREAMER_WM_BEST_RESTORE_AT_P2=0.
    wm_best_restore_at_p2 = bool(int(
        os.environ.get('DREAMER_WM_BEST_RESTORE_AT_P2', '1') or 0))
    # ----- wm_best.pt warm-restore at P2→P3 (P90, 2026-06-06) -----
    # The WM keeps training through P2 (paper Algorithm 1 co-trains it during
    # critic warmup), but its held-action fixed point is UNSTABLE — the probe
    # shows it converges mid-P1 (conv=1.0) then RE-DRIFTS during P2 (conv→0).
    # The WM is then FROZEN at the end-of-P2 (drifted) state for all of P3 and
    # for the post-training steady-state/transfer diagnostics, so a genuinely
    # converged WM that existed earlier is silently discarded (P90 RCA).
    # CAVEAT (default OFF): this is a FULL-model reload, so it ALSO resets the
    # reward head + policy to the wm_best (early) checkpoint — wiping P2's
    # reward-MTP + policy-BC warmup.  Prefer ``wm_freeze_after_p1`` (surgical:
    # freezes only the WM CORE via requires_grad, preserving reward/policy
    # training).  Opt in with DREAMER_WM_BEST_RESTORE_AT_P3=1 only if not using
    # the freeze.
    wm_best_restore_at_p3 = bool(int(
        os.environ.get('DREAMER_WM_BEST_RESTORE_AT_P3', '0') or 0))
    # P90: freeze the WM core after P1 (restore best at P1→P2, then no WM-core
    # training in P2/P3) for critic/WM coherence + drift immunity.  cfg flag,
    # env-overridable via DREAMER_WM_FREEZE_AFTER_P1.
    wm_freeze_after_p1 = bool(getattr(cfg, 'wm_freeze_after_p1', False))
    # WM freeze-after-PRETRAIN (2026-06-09): joint-mode WM-core freeze once the
    # WM has been pretrained ``wm_freeze_after_iters`` joint iters (the phased
    # wm_freeze_after_p1 never fires in joint mode).  ``_wm_frozen_now`` is the
    # one-shot latch so the freeze + banner happen exactly once.
    wm_freeze_after_iters = int(getattr(cfg, 'wm_freeze_after_iters', 0) or 0)
    _wm_frozen_now = False
    p3_critic_warmup_iters = int(getattr(cfg, 'p3_critic_warmup_iters', 0) or 0)
    # One-shot guard so the critic-warmup banner logs EXACTLY once.  The print
    # sits inside the inner train-steps loop, which would otherwise repeat it
    # once per step for the whole first warmup iter (~25x).
    _critic_warmup_logged = False
    wm_trunk_stopgrad_in_p2 = bool(getattr(cfg, 'wm_trunk_stopgrad_in_p2', False))
    # neural-apc-mbrl JOINT training mode (DreamerV1/V2/V3 style).
    joint_mode = str(getattr(cfg, 'train_mode', 'phased')).lower() == 'joint'
    joint_prior_refresh_iters = int(
        getattr(cfg, 'joint_prior_refresh_iters', 0) or 0)
    p3_prior_refresh_iters = int(
        getattr(cfg, 'p3_prior_refresh_iters', 0) or 0)
    # ----- Staged clean->disturbance curriculum (2026-06-12) -----
    # Precondition: needs phased mode (it IS the phased curriculum) + the DOB.
    # If misconfigured, hard-disable with a loud warning rather than running a
    # half-applied curriculum.  ``_cur_stage`` latches the applied stage so the
    # freeze/dob_active/banner fire exactly once per transition (the per-iter
    # disturbance-prob override is cheap and re-applied every iter).
    curriculum = bool(getattr(cfg, 'curriculum_enabled', False))
    if curriculum and joint_mode:
        print('[curriculum] DISABLED: curriculum_enabled requires phased mode '
              '(train_mode != joint) — it IS the phased curriculum.',
              flush=True)
        curriculum = False
    # p136: the curriculum no longer REQUIRES the DOB — Stage 2 identifies the
    # disturbance with the always-trainable disturbance_head (on the frozen g)
    # when the DOB is off.  It needs a CV to control + SOME disturbance estimator
    # (the DOB or the head).  This DECOUPLES the DR-off clean-WM-id gating (which
    # lives inside this curriculum block) from dob_enabled, so turning the DOB
    # off does NOT silently disable the curriculum and re-confound the gain.
    _has_dist_est = (bool(getattr(cfg, 'dob_enabled', False))
                     or bool(getattr(cfg, 'disturbance_head', True)))
    if curriculum and not (len(env.cv_indices) > 0 and _has_dist_est):
        print('[curriculum] DISABLED: curriculum needs >=1 CV channel + a '
              'disturbance estimator (dob_enabled OR disturbance_head).',
              flush=True)
        curriculum = False
    # mbrl2 real-sim: actor loop-gain robustness now comes from REAL domain
    # randomisation on the true plant in P3 (``set_domain_randomization(True)``),
    # not an imagined gain perturbation — the imagination gain-rand resolver was
    # removed with the rest of the imagination stack.
    _cur_stage = 0
    # Continuous-latent curriculum (2026-06-22): with the cont gain channel +
    # C(1) gain-match de-confounding the gain INHERENTLY, there is NO clean-P1 /
    # frozen-g-P2 staging — WM-id trains g WITH the disturbance present (so the
    # cont disturbance channel learns the amortized-Kalman estimate), then the
    # actor trains on the frozen WM.  (DOB off; the cont disturbance channel is
    # the estimator.)
    _cont_curric = (curriculum
                    and bool(getattr(cfg, 'cont_latent_enabled', False))
                    and not bool(getattr(cfg, 'dob_enabled', False))
                    and len(env.cv_indices) > 0)
    if curriculum:
        # Stage-1 state applied BEFORE the seed fill so the seed buffer is
        # collected CLEAN (no hidden disturbance) and the DOB is suppressed
        # (d_t==0 -> g must explain all CV movement).  wm_freeze_after_p1 is
        # forced off so the Stage-2 (P2) loss keeps wm_total (its recon
        # innovation is what trains the observer on the frozen g).
        wm_freeze_after_p1 = False
        _fz = model.set_world_model_trainable(g=True, dob=False, reward=True)
        model.set_dob_active(False)
        # Continuous-latent path: disturbance ON from the seed buffer onward
        # (the cont disturbance channel needs it); the gain is protected by the
        # cont gain channel + gain-match, not by clean data.  Legacy DOB path:
        # clean seed (disturbance prob 0).
        env._disturbance_prob_override = (
            float(getattr(cfg, 'curriculum_stage2_disturbance_prob', 1.0))
            if _cont_curric else 0.0)
        # DR RCA (2026-06-20): turn OFF domain randomization for the clean WM/DOB
        # identification in P1/P2 (curriculum_wm_id_dr_off).  mbrl2 real-sim: DR is
        # then turned ON at the P2->P3 transition (set_domain_randomization(True)),
        # so the real-sim actor gets loop-gain robustness on the RANDOMISED true
        # plant — the imagination gain-rand path was removed with the imagination stack.
        _dr_gated = bool(getattr(cfg, 'curriculum_wm_id_dr_off', True))
        _dr_found = env.set_domain_randomization(False) if _dr_gated else False
        if _cont_curric:
            print('[curriculum] ENABLED — continuous-latent curriculum '
                  '(DOB-free; cont gain channel + C(1) gain-match de-confound '
                  'the gain INHERENTLY). WM-id (P1+P2): g TRAINS WITH the '
                  f'disturbance present (prob={env._disturbance_prob_override:.2f}) '
                  'so the cont disturbance channel learns the amortized-Kalman '
                  f'estimate (g={_fz["g"]} dob={_fz["dob"]} reward={_fz["reward"]} '
                  f'tensors; domain_randomization='
                  f'{"OFF in P1/P2 (clean WM/DOB id) -> ON in P3 (real-plant DR)" if (_dr_gated and _dr_found) else "on"}).',
                  flush=True)
        else:
            print('[curriculum] ENABLED — staged clean->disturbance curriculum '
                  f'(disturbance estimator: {"DOB (neural-Kalman d_t)" if bool(getattr(cfg, "dob_enabled", False)) else "disturbance_head readout"}). '
                  'Stage 1 (P1): CLEAN data -> WM '
                  f'learns the unbiased gain (g={_fz["g"]} dob={_fz["dob"]} '
                  f'reward={_fz["reward"]} tensors; disturbance prob=0; '
                  f'domain_randomization={"OFF in P1/P2 (clean WM/DOB id) -> ON in P3 (real-plant DR)" if (_dr_gated and _dr_found) else "on"}).',
                  flush=True)
        # NOTE (2026-06-16, p124 RCA): the p123 experiment that DISABLED the
        # P1->P2 wm_best warm-restore in curriculum mode was REVERTED — it
        # regressed the MV gain (warm-restore OFF p124 MV 0.849 vs ON p121-123
        # avg 0.926).  P1 recon is non-monotonic (p124: bottoms iter40 0.085,
        # rises to iter70 0.108), so freezing the END-of-P1 WM froze a WORSE
        # checkpoint than wm_best.  The warm-restore (P39 default) stays ON; the
        # real gain fix is the CV-weighted overshoot loss (supervises the
        # open-loop CV gain directly), not the checkpoint-selection policy.
    wm_best_restore_min_gap = int(
        os.environ.get('DREAMER_WM_BEST_RESTORE_MIN_GAP', '10'))
    # ----- Diagnostics for reward-MTP/WM coupling RCA (P39, 2026-05-22) -----
    # All four are cheap and gated by env vars.  A + D are standing
    # observability (default ON, run at probe cadence → <2% overhead).
    # B + C are controlled-experiment switches (default OFF) used to
    # causally isolate whether reward-MTP gradients distort the latent.
    diag_perhead_grads_every = int(
        os.environ.get('DREAMER_DIAG_PERHEAD_GRADS_EVERY', '10') or 0)
    diag_latent_stability_every = int(
        os.environ.get('DREAMER_DIAG_LATENT_STABILITY_EVERY', '10') or 0)
    diag_disable_reward_mtp_in_p1 = bool(int(
        os.environ.get('DREAMER_DIAG_DISABLE_REWARD_MTP_IN_P1', '0') or 0))
    diag_reward_mtp_stop_grad_in_p1 = bool(int(
        os.environ.get('DREAMER_DIAG_REWARD_MTP_STOP_GRAD_IN_P1', '0') or 0))
    diag_latent_ref: Optional[Dict[str, torch.Tensor]] = None
    diag_perhead_last: Dict[str, float] = {}
    if diag_disable_reward_mtp_in_p1 and diag_reward_mtp_stop_grad_in_p1:
        print('[diag] WARNING: both DISABLE_REWARD_MTP_IN_P1 and '
              'REWARD_MTP_STOP_GRAD_IN_P1 set; DISABLE takes precedence.',
              flush=True)
    if diag_disable_reward_mtp_in_p1:
        print('[diag] reward MTP disabled in P1 (experiment B)', flush=True)
    elif diag_reward_mtp_stop_grad_in_p1:
        print('[diag] reward MTP stop-grad in P1 (experiment C)', flush=True)
    # Reference entropy used for the entropy-collapse early-stop trip.
    # For the discrete categorical actor this is the max-entropy uniform
    # baseline (``log K``) per action dim.  For the continuous TanhNormal
    # actor we use a unit-Gaussian baseline (``0.5 * log(2*pi*e)`` per
    # dim ≈ 1.4189 nats); when the policy's reported Gaussian entropy
    # falls below ``frac * reference``, σ has collapsed and exploration
    # has effectively stopped — same trip semantics as for the discrete
    # head, just calibrated for the continuous distribution.
    if str(getattr(cfg, 'policy_type', 'continuous')).lower() == 'continuous':
        from models.dreamer_v4 import ContinuousPolicyHead as _ContPH
        n_action_bins_log = _ContPH.reference_entropy(int(cfg.action_dim))
    else:
        # Per-dim reference (legacy behaviour for the discrete head).
        n_action_bins_log = math.log(
            max(2, int(getattr(cfg, 'n_action_bins', 21))))
    # Pre-initialize loss dicts so the phase-transition mid-checks can
    # read them safely even when the prior phase produced no log iter.
    wm_losses: Dict = {}
    ag_losses: Dict = {}
    ac_losses: Dict = {}

    def _phase_for(env_steps: int) -> int:
        # neural-apc-mbrl JOINT mode: no phase boundaries — co-train WM +
        # actor + critic every step via the P3 update path for the whole run.
        if joint_mode:
            return 3
        # 2026-05-26 (P52 RCA): boundaries are dynamic — the P1/P2
        # quality gates can extend the phase by up to
        # ``p{1,2}_gate_max_ext_steps`` if convergence criteria aren't
        # met at the nominal env-step budget.
        p1_eff = p1 + p1_ext_steps
        p2_eff = p2 + p2_ext_steps
        if env_steps < p1_eff:
            return 1
        if env_steps < p1_eff + p2_eff:
            return 2
        return 3

    # Seed buffer.  P0 (2026-05-05): instead of two uniform-random episodes
    # — which on cliff-shaped reward landscapes (this plant: raw_min=-250,
    # raw_max=+0.1) produce nothing but violation transitions and leave
    # REINFORCE with no positive-advantage seed — prepend a few
    # ``baseline_seed_episodes`` driven by small-noise actions around
    # mid-MV.  Operator analogue: "don't move the valves while the
    # buffer is empty".  Falls back to all-random if the user explicitly
    # opts out.
    n_baseline_seed = int(getattr(cfg, 'baseline_seed_episodes', 8))
    baseline_seed_std = float(getattr(cfg, 'baseline_seed_action_std', 0.05))
    n_random_seed = int(getattr(cfg, 'random_seed_episodes', 2))
    n_prbs_seed = int(getattr(cfg, 'exploration_seed_episodes', 0))
    prbs_op_band = float(getattr(cfg, 'prbs_seed_op_band', 0.95))
    # P43 (2026-05-23): baseline_seed centres stratified over
    # ``[-baseline_op_band, +baseline_op_band]`` so the WM sees
    # held-action steady-state at multiple operating points instead of
    # only mid-MV.  Audit (output/test_sim/_data_audit_v2_*) showed the
    # legacy zero-centred baseline_seed only excited 20 % of the MV
    # band; this stratification raises that to ~100 % with the same
    # episode budget.  Op-band defaults to PRBS-1-sigma (0.6) and is
    # overridable via ``DREAMER_BASELINE_SEED_OP_BAND``.
    baseline_op_band = float(os.environ.get(
        'DREAMER_BASELINE_SEED_OP_BAND',
        str(min(0.6, float(prbs_op_band)))))
    if n_baseline_seed > 0:
        # Stratified centres: split the operating band into
        # ``n_baseline_seed`` equal strata, draw one centre per stratum
        # (shuffled).  Guarantees each band-fraction is visited at
        # least once even for small N.
        edges = np.linspace(-baseline_op_band, +baseline_op_band,
                             n_baseline_seed + 1)
        centres = env.rng.uniform(edges[:-1], edges[1:]).astype('float32')
        env.rng.shuffle(centres)
        for i in range(n_baseline_seed):
            ep = collect_baseline_episode(env, cfg,
                                            action_std=baseline_seed_std,
                                            center=float(centres[i]))
            buf.add_episode(ep['obs'], ep['act'], ep['rew'], ep['cont'])
            total_env_steps += cfg.episode_length
    for _ in range(max(0, n_random_seed)):
        ep = collect_episode(env, model, device, cfg, random_action=True)
        buf.add_episode(ep['obs'], ep['act'], ep['rew'], ep['cont'])
        total_env_steps += cfg.episode_length
    # PRBS-style operating-band sweep (run_p7 RCA): forces WM to see
    # step-response transitions across the full MV operating range so
    # next-state predictions don't degrade outside the mid-bound region
    # the actor will eventually need to leave.  Stratified sampling
    # (cfg.prbs_seed_n_strata) guarantees boundary-bin coverage even
    # when #segments per episode is small.
    for _ in range(max(0, n_prbs_seed)):
        ep = collect_prbs_episode(env, cfg,
                                    action_std=baseline_seed_std,
                                    op_band=prbs_op_band)
        buf.add_episode(ep['obs'], ep['act'], ep['rew'], ep['cont'])
        total_env_steps += cfg.episode_length

    # Constant-action / step-and-settle seed (run_p31 RCA 2026-05-21
    # + P42 design 2026-05-23): the PRBS sweep only ever holds an
    # action for ~80–150 agent steps, so the WM has no long-horizon
    # constant-action data and extrapolates badly when the trained
    # controller later sits near a setpoint.
    #
    # Two episode flavours are used (mix controlled by
    # ``step_settle_seed_fraction``):
    #   1. pure constant-action — long-horizon SS at one operating point,
    #   2. step-and-settle — hold u₀ briefly, step to u₁, hold u₁ to
    #      episode end.  Strict superset: late-episode SS PLUS a clean
    #      step response with known timing (gain + time-constant signal)
    #      PLUS supervision for "the plant CHANGES under constant input
    #      during settling" (breaks the degenerate const-in→const-out
    #      shortcut the WM may learn from pure const-action seeds —
    #      run_p40 RCA: WM steady-state probe 0% conv at H=200 despite
    #      L=32 + 40 const seeds).
    #
    # Operating points u₀ are stratified over
    # ``[-constant_action_seed_op_band, +constant_action_seed_op_band]``.
    n_const_seed = int(getattr(cfg, 'constant_action_seed_episodes', 0))
    const_op_band = float(getattr(cfg, 'constant_action_seed_op_band', 0.6))
    step_frac = float(getattr(cfg, 'step_settle_seed_fraction', 0.0))
    step_frac = float(np.clip(step_frac, 0.0, 1.0))
    if n_const_seed > 0:
        # Stratified u₀ levels: evenly-spaced over [-op_band, +op_band]
        # with a small jitter so re-running the workflow does not hit
        # exactly the same operating points.
        levels = np.linspace(-const_op_band, const_op_band, n_const_seed,
                              dtype='float32')
        jitter = env.rng.uniform(-0.05, 0.05, size=levels.shape).astype('float32')
        levels = np.clip(levels + jitter * const_op_band, -1.0, 1.0)
        # Alternate step / const so the const-vs-step split is evenly
        # spread across the operating-point sweep instead of clustered
        # at one end.  ``do_step_mask[i]`` is True for step-and-settle.
        n_step = int(round(step_frac * n_const_seed))
        do_step_mask = np.zeros(n_const_seed, dtype=bool)
        if n_step > 0:
            # Interleave: pick the n_step indices most evenly spread.
            step_idx = np.linspace(0, n_const_seed - 1, n_step,
                                     dtype=int)
            do_step_mask[step_idx] = True
        n_step_emitted = 0
        n_const_emitted = 0
        for i, lvl in enumerate(levels):
            ep = _seed_one_const_or_step(env, cfg,
                                          level=float(lvl),
                                          do_step=bool(do_step_mask[i]))
            buf.add_episode(ep['obs'], ep['act'], ep['rew'], ep['cont'])
            total_env_steps += cfg.episode_length
            if do_step_mask[i]:
                n_step_emitted += 1
            else:
                n_const_emitted += 1
        print(f"[seed] const-action={n_const_emitted} "
              f"step-settle={n_step_emitted} "
              f"(step_fraction={step_frac:.2f}, op_band={const_op_band:.2f})",
              flush=True)

    # APC step-test seed episodes (P51 design, 2026-05-25).  Held-MV
    # baseline episodes with interleaved MV and DV step events at
    # mixed spacing.  Adds the missing DV-coverage-with-held-action
    # regime that constant_action / step_settle episodes deliberately
    # blank out (both set ``env._schedule = []``).  Sim-adaptive via:
    #   * dyn_horizon (event spacing)
    #   * n_channels = n_mv + n_dv (episode count: ≥ k per channel so
    #     each input axis gets balanced isolated-step coverage)
    #   * primary_dv_pos round-robin (within-episode DV stratification)
    n_step_test_floor = int(getattr(cfg, 'step_test_seed_episodes', 0))
    n_per_ch = int(getattr(cfg, 'step_test_episodes_per_channel', 0))
    n_mv = int(len(getattr(env.sim, 'mv_indices', []) or []))
    n_dv = int(len(getattr(env.sim, 'dv_indices', []) or []))
    n_channels = max(1, n_mv + n_dv)
    n_step_test_seed = max(n_step_test_floor, n_per_ch * n_channels)
    if n_step_test_seed > 0:
        st_levels = np.linspace(-const_op_band, const_op_band,
                                  n_step_test_seed, dtype='float32')
        st_jit = env.rng.uniform(-0.05, 0.05,
                                   size=st_levels.shape).astype('float32')
        st_levels = np.clip(st_levels + st_jit * const_op_band, -1.0, 1.0)
        for ep_idx, lvl in enumerate(st_levels):
            # Round-robin primary DV so each DV channel gets balanced
            # coverage across the seed batch.  -1 disables when n_dv=0.
            primary = (ep_idx % n_dv) if n_dv > 0 else -1
            ep = collect_step_test_episode(env, cfg,
                                             initial_level=float(lvl),
                                             primary_dv_pos=int(primary))
            buf.add_episode(ep['obs'], ep['act'], ep['rew'], ep['cont'])
            total_env_steps += cfg.episode_length
        print(f"[seed] step-test={n_step_test_seed} "
              f"(n_mv={n_mv}, n_dv={n_dv}, per_ch={n_per_ch}, "
              f"floor={n_step_test_floor}, "
              f"dv_share={cfg.step_test_dv_share:.2f}, "
              f"overlap_frac={cfg.step_test_overlap_frac:.2f}, "
              f"primary_dv_bias={cfg.step_test_primary_dv_bias:.2f})",
              flush=True)

    # ---------- DV-PRBS seed episodes (2026-06-14, p121 DV-gain RCA) ----------
    # Sweep every measured-DV channel with a full-range, multi-timescale,
    # stratified PRBS (MV held), the DV analogue of the MV PRBS seeding above.
    # Closes the ~30× MV-vs-DV excitation asymmetry that left the DV→CV gain
    # attenuated (~0.75).  No-op fallback (MV-hold) when n_dv=0.  MV operating
    # point is stratified across the batch so the DV gain is identified at
    # several MV levels.
    n_dv_prbs_seed = int(getattr(cfg, 'dv_prbs_seed_episodes', 0))
    if n_dv_prbs_seed > 0 and n_dv > 0:
        dvp_levels = np.linspace(-const_op_band, const_op_band,
                                  n_dv_prbs_seed, dtype='float32')
        dvp_jit = env.rng.uniform(-0.05, 0.05,
                                   size=dvp_levels.shape).astype('float32')
        dvp_levels = np.clip(dvp_levels + dvp_jit * const_op_band, -1.0, 1.0)
        for lvl in dvp_levels:
            ep = collect_dv_prbs_episode(env, cfg, mv_level=float(lvl))
            buf.add_episode(ep['obs'], ep['act'], ep['rew'], ep['cont'])
            total_env_steps += cfg.episode_length
        print(f"[seed] dv-prbs={n_dv_prbs_seed} "
              f"(n_dv={n_dv}, op_frac={cfg.dv_prbs_op_frac:.2f}, "
              f"seg=[{int(getattr(cfg, 'prbs_seed_segment_steps_min', 0))}.."
              f"{int(getattr(cfg, 'prbs_seed_segment_steps', 0))}])",
              flush=True)

    # ---------- APC expert seed episodes (P81 design, 2026-06-03) ----------
    # Build the steady-state expert (static gain-schedule or NN surrogate),
    # collect ``expert_seed_episodes`` expert-driven demonstrations (flagged
    # expert=1 so BC is masked to them), and auto-enable ``bc_scale`` when the
    # expert is usable.  The expert is objective-ALIGNED: it grounds the policy
    # mean without changing the optimum (RL still owns the true optimum via the
    # real reward + WM).  Disabled cleanly when expert_type='none' or the
    # identified gains / steady-state sweep are unavailable.
    cfg._expert_active = False
    expert_type = str(getattr(cfg, 'expert_type', 'none') or 'none').lower()
    n_expert_seed = int(getattr(cfg, 'expert_seed_episodes', 0))
    if expert_type not in ('', 'none') and n_expert_seed > 0:
        try:
            from utils import apc_expert as _apc_expert
            sv = list(env.meta.get('state_variables', []) or [])

            def _names(idxs):
                return [sv[i] if 0 <= i < len(sv) else f'idx{i}' for i in idxs]

            mv_names = _names([int(x) for x in env.meta.get('mv_indices', [])])
            cv_names = _names([int(x) for x in env.cv_indices])
            dv_names = _names([int(x) for x in env.meta.get('dv_indices', [])
                               if x is not None])
            expert, expert_info = _apc_expert.build_expert(
                expert_type=expert_type,
                out_dir=getattr(cfg, 'out_dir', '') or '.',
                obj_spec=env.obj_spec, obj_w=env.obj_w,
                mv_bounds=env.mv_bounds_eu, cv_bounds=env.cv_bounds_eu,
                mv_names=mv_names, cv_names=cv_names, dv_names=dv_names,
                use_ss_samples=bool(getattr(cfg, 'expert_use_ss_samples', True)),
                seed=int(getattr(cfg, 'seed', 0)),
            )
        except Exception as _exc:
            expert, expert_info = None, {'expert_type': expert_type,
                                         'usable': False,
                                         'build_error': repr(_exc)}

        # Persist the build report next to the run artefacts for diagnosis.
        try:
            _ed = Path(cfg.out_dir or '.')
            _ed.mkdir(parents=True, exist_ok=True)
            with open(_ed / 'expert_seed_info.json', 'w', encoding='utf-8') as _f:
                json.dump(expert_info, _f, indent=2, default=str)
        except Exception:
            pass

        if expert is not None and expert_info.get('usable', False):
            env._apc_expert = expert
            jitter = float(getattr(cfg, 'expert_action_jitter', 0.0))
            keep_sched = bool(getattr(cfg, 'expert_keep_schedule', True))
            for _ in range(n_expert_seed):
                ep = collect_expert_episode(
                    env, cfg, expert=expert,
                    keep_schedule=keep_sched, action_jitter=jitter,
                    rng=env.rng)
                buf.add_episode(ep['obs'], ep['act'], ep['rew'],
                                ep['cont'], ep.get('expert'))
                total_env_steps += cfg.episode_length
            # Auto-enable masked BC toward the expert demonstrations.
            cfg.bc_scale = float(getattr(cfg, 'expert_bc_scale', 0.15))
            cfg._expert_active = True
            print(f"[seed] apc-expert={n_expert_seed} type={expert_type} "
                  f"bc_scale->{cfg.bc_scale:.3f} jitter={jitter:.3f} "
                  f"src={expert_info.get('gain_source', expert_info.get('reason', '?'))}",
                  flush=True)
        else:
            print(f"[seed] apc-expert SKIPPED type={expert_type} "
                  f"reason={expert_info.get('reason', expert_info.get('build_error', 'not_usable'))}",
                  flush=True)

    # Cached optimizer set per phase.
    # Initialize hidden-disturbance probability for the starting phase
    # (hidden CV disturbance is the default unmeasured-disturbance model).
    env._disturbance_prob_override = get_phase_disturbance_prob(phase=1)
    # neural-apc-mbrl JOINT mode: skip the P1/P2 curriculum entirely.  The
    # seed-buffer fill above is the random PREFILL (DreamerV3 prefill);
    # from here co-train WM + actor + critic every step via the P3 path.
    # Run the P3-entry setup ONCE (snapshot the PMPO prior, anchor the BC
    # decay + cascade canary) since we never hit the phase-transition block.
    if joint_mode:
        current_phase = 3
        model.snapshot_prior_policy()
        p3_start_steps = total_env_steps
        try:
            _rs0 = float(model.ret_scale.detach().item())
            if _rs0 > 0.0 and np.isfinite(_rs0):
                p3_start_return_scale = _rs0
        except Exception:
            pass
        env._current_phase = 3
        env._disturbance_prob_override = get_phase_disturbance_prob(phase=3)
        # mbrl2 real-sim: joint mode goes straight to P3 -> enable DR now.
        if env.set_domain_randomization(True):
            print('[realsim] domain randomization ENABLED (joint P3)', flush=True)
        print(f"[joint] DreamerV3-style joint training: WM+actor+critic "
              f"co-trained every step from prefill "
              f"(critic_warmup={p3_critic_warmup_iters} iters, "
              f"prior_refresh={joint_prior_refresh_iters})", flush=True)
    while total_env_steps < cfg.total_steps:
        # Push training progress into the env so the hidden-OU amplitude
        # curriculum (DREAMER_HIDDEN_OU_AMP_RAMP) sees the latest value
        # at every episode reset.  No-op when curriculum env var unset.
        env.set_training_progress(total_env_steps / max(1, int(cfg.total_steps)))
        # Compute per-phase progress for phase-aware OU trigger ramps.
        if joint_mode:
            # No phases in joint mode: drive the disturbance ramp off GLOBAL
            # progress (env_steps / total_steps) so it ramps over the whole
            # run instead of being pinned at 0 until env_steps passes the
            # dead p1+p2 budget (the phased _phase_start would mis-scale it).
            _phase_progress = float(np.clip(
                total_env_steps / max(1, int(cfg.total_steps)), 0.0, 1.0))
        elif current_phase == 1:
            _phase_start = 0
            _phase_len = max(1, int(p1))
            _phase_progress = float(np.clip(
                (total_env_steps - _phase_start) / _phase_len, 0.0, 1.0))
        elif current_phase == 2:
            _phase_start = int(p1)
            _phase_len = max(1, int(p2))
            _phase_progress = float(np.clip(
                (total_env_steps - _phase_start) / _phase_len, 0.0, 1.0))
        else:
            _phase_start = int(p1 + p2)
            _phase_len = max(1, int(p3))
            _phase_progress = float(np.clip(
                (total_env_steps - _phase_start) / _phase_len, 0.0, 1.0))
        env._current_phase = int(current_phase)
        # Refresh adaptive hidden-OU per-episode probability (P38):
        # P1: interpolated on wm_best_score.
        # P2/P3: ramped on per-phase progress.
        try:
            env._disturbance_prob_override = get_phase_disturbance_prob(
                phase=int(current_phase),
                wm_best_score=(float(wm_best_score)
                                if wm_best_score > -1e17 else None),
                phase_progress=_phase_progress,
            )
        except Exception:
            pass

        # ---------- Staged curriculum stage control (2026-06-12) ----------
        # Stage == current_phase.  Override the disturbance prob per stage
        # (cheap, every iter) and apply the freeze/dob_active partition ONCE
        # per stage via the ``_cur_stage`` latch (idempotent; survives the
        # quality-gate phase extensions since it keys off the actual phase).
        if curriculum:
            if _cont_curric:
                # Continuous-latent curriculum: WM-id (g trains + disturbance
                # present) for stages 1-2, then actor on the frozen WM (stage 3).
                # Disturbance ON throughout; the cont gain channel + gain-match
                # hold the gain unbiased (no clean-data / frozen-g protection).
                env._disturbance_prob_override = (
                    float(getattr(cfg, 'curriculum_stage2_disturbance_prob', 1.0))
                    if int(current_phase) < 3
                    else float(getattr(cfg, 'curriculum_stage3_disturbance_prob',
                                       0.85)))
                if _cur_stage != int(current_phase):
                    _cur_stage = int(current_phase)
                    model.set_dob_active(False)
                    if _cur_stage < 3:
                        _fz = model.set_world_model_trainable(
                            g=True, dob=False, reward=True)
                        _desc = (f'WM-id (cont gain+disturbance latent; g TRAINS, '
                                 f'disturbance {env._disturbance_prob_override:.2f}, '
                                 f'gain-match supervises the gain)')
                    else:
                        _fz = model.set_world_model_trainable(
                            g=False, dob=False, reward=True)
                        _wm_frozen_now = True
                        _igf = float(getattr(cfg,
                                'actor_imag_gain_random_frac', 0.0) or 0.0)
                        _desc = (f'actor/critic on FROZEN cont-WM (disturbance '
                                 f'{env._disturbance_prob_override:.2f}; real-plant '
                                 f'DR OFF; imagination loop-gain rand '
                                 f'±{_igf:.3f}; DOB-free disturbance via cont '
                                 f'channel)')
                    print(f"[curriculum] >>> STAGE {_cur_stage} @iter{total_iters} "
                          f"steps{total_env_steps}: {_desc} "
                          f"[g={_fz['g']} dob={_fz['dob']} reward={_fz['reward']}]",
                          flush=True)
            elif int(current_phase) == 1:
                env._disturbance_prob_override = 0.0
            elif int(current_phase) == 2:
                env._disturbance_prob_override = float(
                    getattr(cfg, 'curriculum_stage2_disturbance_prob', 1.0))
            else:
                env._disturbance_prob_override = float(
                    getattr(cfg, 'curriculum_stage3_disturbance_prob', 0.85))
            if _cur_stage != int(current_phase):
                _cur_stage = int(current_phase)
                if _cur_stage == 1:
                    model.set_dob_active(False)
                    _fz = model.set_world_model_trainable(
                        g=True, dob=False, reward=True)
                    _desc = ('CLEAN-WM id (g trains, '
                             + ('DOB suppressed, ' if bool(getattr(cfg, 'dob_enabled', False)) else '')
                             + 'disturbance 0)')
                elif _cur_stage == 2:
                    # Freeze g (protect the clean Stage-1 gain), turn the
                    # disturbance estimator ON, keep wm_total so its innovation
                    # trains the estimator on the FROZEN g.  With the DOB on this
                    # is the neural-Kalman A,K; with the DOB off (p136 default)
                    # it is the always-trainable disturbance_head reading the
                    # frozen GRU latent's disturbance tracking.
                    _dob_on = bool(getattr(cfg, 'dob_enabled', False))
                    model.set_dob_active(_dob_on)
                    _fz = model.set_world_model_trainable(
                        g=False, dob=_dob_on, reward=True)
                    _est = ('DOB id (observer A,K)' if _dob_on
                            else 'disturbance-head id (frozen-g readout)')
                    _desc = (f'{_est} (g FROZEN + reward train via recon '
                             f'innovation, disturbance '
                             f'{env._disturbance_prob_override:.2f})')
                else:
                    # Freeze g AND the observer; the real-sim actor/critic train
                    # on the FROZEN WM(+DOB) observer.  _wm_frozen_now drops
                    # wm_total in the P3 path so only the actor/critic optimise.
                    model.set_dob_active(bool(getattr(cfg, 'dob_enabled', False)))
                    _fz = model.set_world_model_trainable(
                        g=False, dob=False, reward=True)
                    _wm_frozen_now = True
                    # mbrl2 real-sim: the actor/critic train on REAL rollouts of
                    # the true plant with domain randomisation ENABLED at this
                    # transition (set_domain_randomization(True)); the observer
                    # was identified CLEAN in P1/P2.  The unmeasured OU
                    # disturbance rides the real P3 data (a feed-forward target
                    # the actor rejects via the DOB estimate in feat).
                    _wmtag = 'WM+DOB' if bool(getattr(cfg, 'dob_enabled', False)) else 'WM'
                    _desc = (f'real-sim actor/critic on FROZEN {_wmtag} observer '
                             f'(disturbance {env._disturbance_prob_override:.2f}; '
                             f'real-plant DR ON for actor robustness)')
                print(f"[curriculum] >>> STAGE {_cur_stage} @iter{total_iters} "
                      f"steps{total_env_steps}: {_desc} "
                      f"[g={_fz['g']} dob={_fz['dob']} reward={_fz['reward']} "
                      f"trainable-flags set]", flush=True)

        # ---------- P52 RCA: phase-transition quality gates ----------
        # When the env-step budget says "leave the current phase", check
        # the corresponding quality gate.  If it fails AND extension
        # headroom remains, bump the boundary by ``p1_gate_check_step``
        # (10 % of P1 budget) and stay in the current phase.  If it
        # fails at the cap, log a flag and proceed anyway.
        _candidate_phase = _phase_for(total_env_steps)
        if (current_phase == 1 and _candidate_phase == 2
                and p1_gate_wm_ema_min_v > 0.0):
            # Healthy?  EMA above floor AND plateaued (≥ K probes
            # within ±plateau_frac of EMA-best).
            _ema_ok = (wm_score_ema_best > -1e17
                       and wm_score_ema_best >= p1_gate_wm_ema_min_v)
            _plateau_ok = False
            if _ema_ok and len(p1_score_ema_history) >= p1_gate_plateau_probes_v:
                # 2026-05-27 (P57 RCA): adaptive plateau band.  The
                # static ``plateau_frac × ema_best`` band (default 5 %)
                # is too tight for plants where the WM-fidelity probe
                # has high per-probe variance (P57: σ(probe)≈0.4 →
                # σ(EMA)≈0.3 → 5 % of ema_best=2.7 = 0.135 ≪ noise).
                # The gate then never sees "plateaued" even after the
                # EMA has flattened in the noisy sense.  Widen the
                # band to ``max(plateau_frac × ema_best, k × σ(recent EMAs))``
                # — sim-agnostic, derived from the actual EMA noise
                # observed in the last ``2 × plateau_probes`` probes.
                recent = p1_score_ema_history[-p1_gate_plateau_probes_v:]
                _noise_window = p1_score_ema_history[
                    -max(p1_gate_plateau_probes_v * 2, 4):]
                _noise_std = (float(np.std([e for _, e in _noise_window]))
                              if len(_noise_window) >= 2 else 0.0)
                _band_static = p1_gate_plateau_frac_v * wm_score_ema_best
                _band_noise = 2.0 * _noise_std
                _band = max(1e-6, _band_static, _band_noise)
                _plateau_ok = all(
                    abs(e - wm_score_ema_best) <= _band for _, e in recent)
            if _ema_ok and _plateau_ok:
                phase_gate_decisions.append({
                    'gate': 'p1->p2', 'iter': int(total_iters),
                    'env_steps': int(total_env_steps),
                    'pass': True, 'ext_steps': int(p1_ext_steps),
                    'wm_ema_best': float(wm_score_ema_best),
                    'wm_ema_best_iter': int(wm_score_ema_best_iter),
                })
                print(f'[gate p1->p2] PASS at iter {total_iters}: '
                      f'wm_ema_best={wm_score_ema_best:.3f} '
                      f'(iter {wm_score_ema_best_iter}, '
                      f'min={p1_gate_wm_ema_min_v:.2f}), '
                      f'plateaued over last {p1_gate_plateau_probes_v} probes; '
                      f'p1_extension={p1_ext_steps}/{p1_gate_max_ext_steps} steps',
                      flush=True)
            elif p1_ext_steps + p1_gate_check_step <= p1_gate_max_ext_steps:
                p1_ext_steps += p1_gate_check_step
                reason = ('ema_below_floor' if not _ema_ok
                          else 'not_plateaued')
                phase_gate_decisions.append({
                    'gate': 'p1->p2', 'iter': int(total_iters),
                    'env_steps': int(total_env_steps),
                    'pass': False, 'extended_to': int(p1 + p1_ext_steps),
                    'reason': reason,
                    'wm_ema': float(wm_score_ema)
                        if wm_score_ema > -1e17 else None,
                    'wm_ema_best': float(wm_score_ema_best)
                        if wm_score_ema_best > -1e17 else None,
                })
                print(f'[gate p1->p2] FAIL ({reason}) at iter {total_iters}: '
                      f'wm_ema_best={wm_score_ema_best:.3f} < '
                      f'{p1_gate_wm_ema_min_v:.2f} — extending P1 by '
                      f'{p1_gate_check_step} steps to '
                      f'{p1 + p1_ext_steps} (cap {p1 + p1_gate_max_ext_steps})',
                      flush=True)
            else:
                phase_gate_decisions.append({
                    'gate': 'p1->p2', 'iter': int(total_iters),
                    'env_steps': int(total_env_steps),
                    'pass': False, 'capped': True,
                    'wm_ema_best': float(wm_score_ema_best)
                        if wm_score_ema_best > -1e17 else None,
                })
                print(f'[gate p1->p2] CAPPED — proceeding to P2 despite '
                      f'wm_ema_best={wm_score_ema_best:.3f} < '
                      f'{p1_gate_wm_ema_min_v:.2f} (cap '
                      f'{p1_gate_max_ext_steps} steps reached)',
                      flush=True)
                mid_check_flags.append(
                    f'p1_gate_capped: wm_ema_best={wm_score_ema_best:.3f}')
        elif (current_phase == 2 and _candidate_phase == 3
                and p2_gate_reward_mtp_max_v > 0.0):
            _rml_med = (float(np.median(p2_reward_mtp_recent))
                        if len(p2_reward_mtp_recent) > 0 else float('inf'))
            _rml_ok = (len(p2_reward_mtp_recent)
                        >= max(1, p2_gate_recent_iters_v)
                       and _rml_med <= p2_gate_reward_mtp_max_v)
            _p2_step = max(1, int(0.10 * max(1, p2)))
            if _rml_ok:
                phase_gate_decisions.append({
                    'gate': 'p2->p3', 'iter': int(total_iters),
                    'env_steps': int(total_env_steps),
                    'pass': True, 'ext_steps': int(p2_ext_steps),
                    'reward_mtp_median': _rml_med,
                })
                print(f'[gate p2->p3] PASS at iter {total_iters}: '
                      f'reward_mtp_loss median={_rml_med:.3f} '
                      f'(max={p2_gate_reward_mtp_max_v:.2f}, '
                      f'n={len(p2_reward_mtp_recent)}); '
                      f'p2_extension={p2_ext_steps}/{p2_gate_max_ext_steps} steps',
                      flush=True)
            elif p2_ext_steps + _p2_step <= p2_gate_max_ext_steps:
                p2_ext_steps += _p2_step
                phase_gate_decisions.append({
                    'gate': 'p2->p3', 'iter': int(total_iters),
                    'env_steps': int(total_env_steps),
                    'pass': False,
                    'extended_to': int(p2 + p2_ext_steps),
                    'reward_mtp_median': _rml_med,
                    'reason': 'reward_mtp_above_floor',
                })
                print(f'[gate p2->p3] FAIL at iter {total_iters}: '
                      f'reward_mtp_loss median={_rml_med:.3f} > '
                      f'{p2_gate_reward_mtp_max_v:.2f} — extending P2 by '
                      f'{_p2_step} steps to {p2 + p2_ext_steps} '
                      f'(cap {p2 + p2_gate_max_ext_steps})', flush=True)
            else:
                phase_gate_decisions.append({
                    'gate': 'p2->p3', 'iter': int(total_iters),
                    'env_steps': int(total_env_steps),
                    'pass': False, 'capped': True,
                    'reward_mtp_median': _rml_med,
                })
                print(f'[gate p2->p3] CAPPED — proceeding to P3 despite '
                      f'reward_mtp_loss={_rml_med:.3f} > '
                      f'{p2_gate_reward_mtp_max_v:.2f} (cap '
                      f'{p2_gate_max_ext_steps} steps reached)',
                      flush=True)
                mid_check_flags.append(
                    f'p2_gate_capped: reward_mtp_loss={_rml_med:.3f}')

        new_phase = _phase_for(total_env_steps)
        if new_phase != current_phase:
            print(f'[phase] transition {current_phase} -> {new_phase} '
                  f'at env_steps={total_env_steps}', flush=True)
            # mbrl2 real-sim: enable domain randomisation at P3 entry so the
            # actor trains on a RANDOMISED true plant (robustness), while the
            # observer was identified CLEAN in P1/P2 (DR off).  No-op if the
            # sim has no randomizer.
            if new_phase == 3 and env.set_domain_randomization(True):
                print('[realsim] domain randomization ENABLED for P3 '
                      '(actor trains on the randomised true plant)', flush=True)
            # Phase mid-checks: emit a flag in the trial summary if the
            # *previous* phase did not meet its convergence floor.  We
            # do not abort here — the trial still runs — but the flag
            # surfaces in summary and is also re-detected in
            # ``evaluation/diagnostics.py``.
            if es_enable and current_phase == 1 and new_phase == 2:
                # P1→P2 transition: WM training done, hand off to P2
                # (paper Algorithm 1).  The runtime WM-fidelity probe +
                # P1-extension + horizon-clip mechanism was removed
                # 2026-05-20; the 1M-step default budget gives P1
                # enough time at the paper-default H=15.
                if p1_initial_sf is not None and 'sf_loss' in wm_losses:
                    _sf_val = wm_losses.get('sf_loss', p1_initial_sf)
                    last_sf = float(_sf_val.detach().item()
                                    if torch.is_tensor(_sf_val) else _sf_val)
                    print(f"[p1→p2] sf_loss {p1_initial_sf:.4f} → "
                          f"{last_sf:.4f}", flush=True)
                # P39: warm-restore WM to its fidelity peak before P2.
                if (wm_best_restore_at_p2
                        and wm_best_ckpt_path is not None
                        and wm_best_iter > 0
                        and (total_iters - wm_best_iter)
                                >= wm_best_restore_min_gap
                        and wm_best_ckpt_path.exists()):
                    try:
                        _blob = torch.load(wm_best_ckpt_path,
                                            map_location=device,
                                            weights_only=False)
                        model.load_state_dict(_blob['model'])
                        print(f"[p1→p2] WM warm-restore: loaded "
                              f"wm_best.pt (iter {wm_best_iter}, "
                              f"score {wm_best_score:.3f}) — discarded "
                              f"{total_iters - wm_best_iter} iters of "
                              f"post-peak drift", flush=True)
                    except Exception as _e:
                        print(f"[p1→p2] WM warm-restore failed: {_e} "
                              f"— continuing with current weights",
                              flush=True)
                # P90: freeze the WM CORE for P2+P3 (after the restore) so the
                # critic warms up on the exact WM P3 uses and the fixed point
                # can't re-drift.  Freeze dynamics (+ tokenizer for SF); KEEP
                # the reward head trainable (it is in parameters_world but must
                # warm up in P2).
                if wm_freeze_after_p1:
                    _nfz = 0
                    for _p in model.dynamics.parameters():
                        _p.requires_grad_(False)
                        _nfz += 1
                    if (getattr(model, 'world_model_type', 'sf_transformer')
                            != 'rssm'
                            and getattr(model, 'tokenizer', None) is not None):
                        for _p in model.tokenizer.parameters():
                            _p.requires_grad_(False)
                            _nfz += 1
                    print(f"[p1→p2] WM CORE FROZEN ({_nfz} param tensors) "
                          f"for P2+P3 — reward/policy/value still train; "
                          f"critic warms up on the P3 WM", flush=True)
            if es_enable and current_phase == 2 and new_phase == 3:
                # End of P2: is reward MTP head learning?
                max_rmtp = float(getattr(cfg, 'early_stop_p2_max_reward_mtp_loss',
                                            float('inf')))
                last_rmtp = float(ag_losses.get('reward_mtp_loss', float('inf')))
                p2_final_reward_mtp = last_rmtp
                if last_rmtp > max_rmtp:
                    msg = (f'P2: reward_mtp_loss={last_rmtp:.3f} > '
                            f'{max_rmtp:.3f} — reward head not learning')
                    print(f'[early-stop-flag] {msg}', flush=True)
                    mid_check_flags.append(msg)
                # P90: warm-restore WM to its conv-aware fidelity peak before
                # the P2→P3 FREEZE so P3 + the post-training diagnostics use the
                # best-converged WM, not the drifted end-of-P2 one.
                if (wm_best_restore_at_p3
                        and wm_best_ckpt_path is not None
                        and wm_best_iter > 0
                        and (total_iters - wm_best_iter)
                                >= wm_best_restore_min_gap
                        and wm_best_ckpt_path.exists()):
                    try:
                        _blob = torch.load(wm_best_ckpt_path,
                                            map_location=device,
                                            weights_only=False)
                        model.load_state_dict(_blob['model'])
                        _cf = None
                        try:
                            _cf = _blob.get('wm_fidelity_probe', {}).get(
                                'wm_converge_frac')
                        except Exception:
                            _cf = None
                        print(f"[p2→p3] WM freeze warm-restore: loaded "
                              f"wm_best.pt (iter {wm_best_iter}, "
                              f"score {wm_best_score:.3f}, "
                              f"conv={_cf}) — discarded "
                              f"{total_iters - wm_best_iter} iters of "
                              f"post-peak P2 drift", flush=True)
                    except Exception as _e:
                        print(f"[p2→p3] WM freeze warm-restore failed: {_e} "
                              f"— continuing with current weights",
                              flush=True)
            current_phase = new_phase
            if current_phase == 3:
                # Snapshot the prior policy (PMPO behavioural prior, eq. 11).
                model.snapshot_prior_policy()
                # 2026-05-23 (P41 RCA): snapshot return_scale at P3 start
                # so the bootstrap-cascade canary can detect runaway growth.
                # ``model.ret_scale`` is the EMA buffer used by the critic
                # target normalisation; it's already initialised by P2.
                try:
                    _rs0 = float(model.ret_scale.detach().item())
                    if _rs0 > 0.0 and np.isfinite(_rs0):
                        p3_start_return_scale = _rs0
                        print(f'[p3-start] return_scale={_rs0:.2f} '
                              '(cascade canary baseline)', flush=True)
                except Exception:
                    pass
                # P83: anchor the P3 BC decay schedule to the step count at
                # phase start (the schedule is phase-length adaptive).
                p3_start_steps = total_env_steps
            # Refresh hidden-disturbance per-episode probability for the
            # new phase (default: 0.3 in P1/P2, 0.5 in P3).
            env._disturbance_prob_override = get_phase_disturbance_prob(
                phase=int(current_phase))

        # ----- Periodic const/step injection (all phases) -----
        # Keep the steady-state + step-response regimes represented in
        # the buffer.  Cheap: 5 episodes every 20 iters ≈ 6% buffer-add
        # overhead.  Mix controlled by ``step_settle_seed_fraction``
        # (P42 design).  P49 RCA: continue into P2/P3 so const-action
        # coverage doesn't get evicted as on-policy episodes fill the
        # buffer (was the root cause of persistent 0% WM steady-state
        # convergence under zero/constant action protocols).
        # JOINT mode (neural-apc-mbrl): current_phase is always 3, so this is
        # gated by ``const_inject_in_p3`` (default OFF) — deliberately, because
        # the critic trains from step 1 in joint mode and the P50 RCA showed
        # injecting low-reward-variance const-action episodes during critic
        # training collapses Var(r) and triggers the very bootstrap cascade
        # joint mode exists to avoid.  Opt in with DREAMER_CONST_ACTION_INJECT_
        # IN_P3=1 only if a run needs the WM steady-state coverage.
        _inject_active = (
            (current_phase == 1)
            or (current_phase == 2 and const_inject_in_p2)
            or (current_phase == 3 and const_inject_in_p3))
        if (_inject_active
                and const_inject_every > 0
                and const_inject_n > 0
                and total_iters > 0
                and (total_iters % const_inject_every) == 0):
            _op_band = float(getattr(cfg,
                                       'constant_action_seed_op_band', 0.6))
            _step_frac = float(np.clip(
                getattr(cfg, 'step_settle_seed_fraction', 0.0), 0.0, 1.0))
            _levels = np.linspace(-_op_band, _op_band, const_inject_n,
                                   dtype='float32')
            _jit = env.rng.uniform(-0.05, 0.05,
                                     size=_levels.shape).astype('float32')
            _levels = np.clip(_levels + _jit * _op_band, -1.0, 1.0)
            _n_step = int(round(_step_frac * const_inject_n))
            _do_step_mask = np.zeros(const_inject_n, dtype=bool)
            if _n_step > 0:
                _do_step_mask[np.linspace(
                    0, const_inject_n - 1, _n_step, dtype=int)] = True
            _n_c = 0
            _n_s = 0
            for _i, _lvl in enumerate(_levels):
                _ep = _seed_one_const_or_step(
                    env, cfg, level=float(_lvl),
                    do_step=bool(_do_step_mask[_i]))
                buf.add_episode(_ep['obs'], _ep['act'],
                                 _ep['rew'], _ep['cont'])
                total_env_steps += cfg.episode_length
                if _do_step_mask[_i]:
                    _n_s += 1
                else:
                    _n_c += 1
            print(f"[const-inject p{current_phase}] iter {total_iters}: added "
                  f"const={_n_c} step={_n_s} episodes "
                  f"(buf_fill={buf.filled}/{buf.capacity_eps})",
                  flush=True)

        # ----- Periodic STEP-TEST (DV-exciting) re-injection (2026-06-13) -----
        # Keep the isolated DV->CV step-response data fresh in the buffer so the
        # WM's DV gain doesn't de-train via FIFO eviction (the const-inject above
        # only replenishes the MV exciters).  Same phase gating as const-inject;
        # no-op when the sim has no DV channel.  collect_step_test_episode holds
        # the MV baseline and drives isolated MV+DV step EVENTS (independent of
        # the curriculum's hidden-disturbance setting), so it excites dCV/dDV
        # directly even in a CLEAN Stage 1.
        _st_inject_active = (
            (current_phase == 1)
            or (current_phase == 2 and step_test_inject_in_p2)
            or (current_phase == 3 and step_test_inject_in_p3))
        _n_dv_inj = int(len(getattr(env.sim, 'dv_indices', []) or []))
        if (_st_inject_active
                and step_test_inject_every > 0
                and step_test_inject_n > 0
                and _n_dv_inj > 0
                and total_iters > 0
                and (total_iters % step_test_inject_every) == 0):
            _st_op = float(getattr(cfg, 'constant_action_seed_op_band', 0.6))
            _st_levels = np.linspace(-_st_op, _st_op, step_test_inject_n,
                                      dtype='float32')
            _st_jit = env.rng.uniform(-0.05, 0.05,
                                       size=_st_levels.shape).astype('float32')
            _st_levels = np.clip(_st_levels + _st_jit * _st_op, -1.0, 1.0)
            for _si, _slvl in enumerate(_st_levels):
                _primary = _si % _n_dv_inj
                _stp = collect_step_test_episode(
                    env, cfg, initial_level=float(_slvl),
                    primary_dv_pos=int(_primary))
                buf.add_episode(_stp['obs'], _stp['act'], _stp['rew'],
                                 _stp['cont'])
                total_env_steps += cfg.episode_length
            print(f"[step-test-inject p{current_phase}] iter {total_iters}: "
                  f"added step-test={step_test_inject_n} episodes "
                  f"(n_dv={_n_dv_inj}, buf_fill={buf.filled}/"
                  f"{buf.capacity_eps})", flush=True)

        # ----- Periodic DV-PRBS re-injection (2026-06-14, p121 DV-gain RCA) ---
        # Keep the FULL-RANGE DV sweep fresh in the buffer so the DV->CV gain
        # stays supervised right up to the WM freeze.  Complements step-test-
        # inject (isolated DV steps): dv-prbs gives the DV the same persistent,
        # stratified, multi-timescale, large-amplitude excitation the MV gets
        # from PRBS — the lever that actually closes the DV-gain attenuation.
        # Same phase gating as the others; no-op when the sim has no DV channel.
        _dvp_inject_active = (
            (current_phase == 1)
            or (current_phase == 2 and dv_prbs_inject_in_p2)
            or (current_phase == 3 and dv_prbs_inject_in_p3))
        if (_dvp_inject_active
                and dv_prbs_inject_every > 0
                and dv_prbs_inject_n > 0
                and _n_dv_inj > 0
                and total_iters > 0
                and (total_iters % dv_prbs_inject_every) == 0):
            _dvp_op = float(getattr(cfg, 'constant_action_seed_op_band', 0.6))
            _dvp_levels = np.linspace(-_dvp_op, _dvp_op, dv_prbs_inject_n,
                                       dtype='float32')
            _dvp_jit = env.rng.uniform(-0.05, 0.05,
                                        size=_dvp_levels.shape).astype('float32')
            _dvp_levels = np.clip(_dvp_levels + _dvp_jit * _dvp_op, -1.0, 1.0)
            for _dvl in _dvp_levels:
                _dvp = collect_dv_prbs_episode(env, cfg, mv_level=float(_dvl))
                buf.add_episode(_dvp['obs'], _dvp['act'], _dvp['rew'],
                                 _dvp['cont'])
                total_env_steps += cfg.episode_length
            print(f"[dv-prbs-inject p{current_phase}] iter {total_iters}: "
                  f"added dv-prbs={dv_prbs_inject_n} episodes "
                  f"(n_dv={_n_dv_inj}, buf_fill={buf.filled}/"
                  f"{buf.capacity_eps})", flush=True)

        # ----- Periodic EXPERT re-injection (P81 RCA) -----
        # Keep objective-aligned expert demonstrations alive in the ring
        # buffer so the masked BC term in P2 always has steps to clone.
        # Without this the seed-time expert episodes are evicted before
        # P2 begins (capacity ~327 eps, fully lapped during P1) and
        # bc_loss collapses to exactly 0.0 (expert no-op).  Active in
        # P1+P2 by default; P3 opt-in for the decaying-P3-BC contingency.
        _expert_inject_active = (
            bool(getattr(cfg, '_expert_active', False))
            and getattr(env, '_apc_expert', None) is not None
            and ((current_phase == 1)
                 or (current_phase == 2)
                 or (current_phase == 3 and expert_inject_in_p3)))
        if (_expert_inject_active
                and expert_inject_every > 0
                and expert_inject_n > 0
                and total_iters > 0
                and (total_iters % expert_inject_every) == 0):
            _ej_jit = float(getattr(cfg, 'expert_action_jitter', 0.0))
            _ej_keep = bool(getattr(cfg, 'expert_keep_schedule', True))
            for _ in range(expert_inject_n):
                _eep = collect_expert_episode(
                    env, cfg, expert=env._apc_expert,
                    keep_schedule=_ej_keep, action_jitter=_ej_jit,
                    rng=env.rng)
                buf.add_episode(_eep['obs'], _eep['act'], _eep['rew'],
                                 _eep['cont'], _eep.get('expert'))
                total_env_steps += cfg.episode_length
            print(f"[expert-inject p{current_phase}] iter {total_iters}: "
                  f"added expert={expert_inject_n} episodes "
                  f"(buf_fill={buf.filled}/{buf.capacity_eps})",
                  flush=True)

        # ----- Collection -----
        # Phases 1 & 2: random-action episodes append to the buffer.
        # Phase 3: collect on-policy episodes (stochastic actor) and
        # append to the buffer so the world model + reward head keep
        # adapting to the actor's actual trajectory distribution
        # (paper Algorithm 1 line 22 — "collect").  Periodic
        # deterministic eval episodes are still produced separately
        # to score the policy (no buffer write, no extra env-step
        # accounting beyond budget pacing).
        _t = time.time()
        if current_phase in (1, 2):
            for _ in range(cfg.ep_per_iter):
                ep = collect_episode(env, model, device, cfg, random_action=True)
                buf.add_episode(ep['obs'], ep['act'], ep['rew'], ep['cont'])
                total_env_steps += cfg.episode_length
                ret = float(ep['rew'].sum())
                # Don't include phase-1/2 random-action returns in the
                # return_window — they are not policy returns.
                ema_return = (ret if ema_return is None
                                else 0.95 * ema_return + 0.05 * ret)
        else:
            # Phase 3: on-policy collection (stochastic actor) every
            # ``phase3_collect_every_iters`` P3 iters (default 4).  The
            # trainer also keeps the world-model heads (tokenizer +
            # dynamics + reward) alive on these batches via the P3 WM
            # update below, so the WM remains aligned with the actor's
            # current state-visit distribution.
            collect_every = max(1, int(cfg.phase3_collect_every_iters))
            # mbrl2 real-sim: warm the dedicated ON-POLICY buffer at P3 entry so
            # the critic warmup + first actor steps train on current-policy data
            # (never the seed/replay buffer's off-policy PRBS/random/expert eps).
            if onpol_buf is not None and onpol_buf.filled == 0:
                _n_pref = max(1, int(getattr(cfg, 'phase3_onpolicy_prefill_eps', 8) or 8))
                for _ in range(_n_pref):
                    ep0 = collect_episode(env, model, device, cfg,
                                          random_action=False,
                                          deterministic=False)
                    buf.add_episode(ep0['obs'], ep0['act'], ep0['rew'], ep0['cont'])
                    onpol_buf.add_episode(
                        ep0['obs'], ep0['act'], ep0['rew'], ep0['cont'])
                    total_env_steps += cfg.episode_length
                print(f"[realsim] on-policy buffer pre-filled with {_n_pref} "
                      f"episodes (P3 actor-critic trains on current-policy data "
                      f"only)", flush=True)
            if (total_iters % collect_every) == 0:
                ep = collect_episode(env, model, device, cfg,
                                       random_action=False,
                                       deterministic=False)
                buf.add_episode(ep['obs'], ep['act'], ep['rew'], ep['cont'])
                if onpol_buf is not None:
                    onpol_buf.add_episode(
                        ep['obs'], ep['act'], ep['rew'], ep['cont'])
                total_env_steps += cfg.episode_length
                ret = float(ep['rew'].sum())
                ema_return = (ret if ema_return is None
                                else 0.95 * ema_return + 0.05 * ret)
            # (Open-loop excitation re-injection into the buffer was removed
            # 2026-06-12: the p105 shared-buffer reinject fed the actor/critic
            # imagination off-distribution step-test start-states, and the
            # separate WM-only partition that replaced it never helped — the
            # settle-aware seed buffer already covers open-loop gain.)
            # Periodic deterministic eval (for the BO score window).
            if (total_iters % max(1, cfg.phase3_eval_every_iters)) == 0:
                ep_eval = collect_episode(env, model, device, cfg,
                                            random_action=False,
                                            deterministic=True)
                ret_eval = float(ep_eval['rew'].sum())
                return_window.append(ret_eval)
                iter_returns.append(ret_eval)
                rs = float(env.reward_scale) if env.reward_scale else 1.0
                iter_raw_returns.append(ret_eval / rs if rs else ret_eval)
                iter_cv_violations.append(float(getattr(env,
                                            '_last_cv_violation_sum', 0.0)))
                iter_mv_violations.append(float(getattr(env,
                                            '_last_mv_violation_sum', 0.0)))
                # BC learning-vs-crutch (2026-06-09): periodically roll the
                # expert under the SAME deterministic protocol (regime-matched,
                # jitter=0) and log its return + the agent-minus-expert gap.
                # A persistently ~0 gap = the actor is a crutch (cloning the
                # expert via the permanent BC floor); a positive, growing gap =
                # the actor is genuinely surpassing the expert.  Set
                # DREAMER_EXPERT_BC_P3_FLOOR=0 to remove the floor confound.
                if (bc_track_expert_every > 0
                        and bool(getattr(cfg, '_expert_active', False))
                        and getattr(env, '_apc_expert', None) is not None):
                    _expert_eval_count += 1
                    if (_expert_eval_count % bc_track_expert_every) == 0:
                        try:
                            _xep = collect_expert_episode(
                                env, cfg, expert=env._apc_expert,
                                keep_schedule=True, action_jitter=0.0,
                                rng=env.rng)
                            last_expert_det_return = float(_xep['rew'].sum())
                            last_agent_minus_expert = (
                                ret_eval - last_expert_det_return)
                        except Exception as _bce:
                            print(f"[bc-track] expert eval skipped @iter"
                                  f"{total_iters}: {_bce!r}", flush=True)
        t_collect_acc += time.time() - _t

        # ----- WM freeze-after-PRETRAIN (joint mode, 2026-06-09) -----
        # Once the WM has been pretrained ``wm_freeze_after_iters`` joint iters,
        # freeze its CORE (dynamics + tokenizer for SF) ONCE so the actor/critic
        # finish training on a STATIC, gain-converged WM (the phased
        # wm_freeze_after_p1 never fires in joint mode).  Reward head stays
        # trainable (it is in parameters_world).  Drives the ``_wm_frozen_now``
        # latch read by the P3 update to drop wm_total from the optimised loss.
        if (joint_mode and wm_freeze_after_iters > 0 and not _wm_frozen_now
                and total_iters >= wm_freeze_after_iters):
            _nfz = 0
            for _p in model.dynamics.parameters():
                _p.requires_grad_(False)
                _nfz += 1
            if (getattr(model, 'world_model_type', 'sf_transformer') != 'rssm'
                    and getattr(model, 'tokenizer', None) is not None):
                for _p in model.tokenizer.parameters():
                    _p.requires_grad_(False)
                    _nfz += 1
            _wm_frozen_now = True
            print(f"[joint] WM CORE FROZEN at iter {total_iters} "
                  f"({_nfz} param tensors) after pretrain — reward/actor/"
                  f"critic keep training on the static WM "
                  f"(wm_freeze_after_iters={wm_freeze_after_iters})", flush=True)

        # ----- Train -----
        wm_grad_norm = torch.tensor(0.0)
        actor_grad_norm = torch.tensor(0.0)
        critic_grad_norm = torch.tensor(0.0)
        wm_losses: Dict[str, torch.Tensor] = {}
        ag_losses: Dict[str, torch.Tensor] = {}
        ac_losses: Dict[str, torch.Tensor] = {}

        for _ in range(cfg.train_steps_per_iter
                          if current_phase != 3
                          else max(1, int(cfg.phase3_train_steps_per_iter))):
            _t = time.time()
            # mbrl2 real-sim: P3 actor-critic REINFORCE is ON-POLICY — sample the
            # dedicated on-policy buffer (recent current-policy episodes only), NOT
            # the shared replay buffer (whose Phase-1/2 PRBS/random/expert seed
            # actions corrupt the policy gradient → the p01 MV-chatter RCA).  P1/P2
            # WM training still samples the full replay buffer.
            _src_buf = (onpol_buf
                        if (current_phase == 3 and onpol_buf is not None
                            and onpol_buf.filled > 0)
                        else buf)
            batch_np = _src_buf.sample(cfg.batch_size, cfg.seq_len, rng)
            batch: Dict[str, torch.Tensor] = {}
            for k, v in batch_np.items():
                t = torch.from_numpy(v)
                if device.type == 'cuda':
                    t = t.pin_memory().to(device, non_blocking=True)
                else:
                    t = t.to(device)
                batch[k] = t
            t_sample_acc += time.time() - _t

            if current_phase in (1, 2):
                # World-model losses (always live in P1 + P2).
                _t = time.time()
                with torch.amp.autocast(device_type=device.type,
                                          dtype=torch.bfloat16,
                                          enabled=(device.type == 'cuda')):
                    wm_losses, _, agent_hid = world_model_loss(
                        model, batch, cfg)
                    # P93: optionally detach the agent feature in P2 so the BC +
                    # reward-MTP gradients train the heads but DON'T flow back
                    # into the WM dynamics trunk (which destabilises recon at
                    # the P1→P2 boundary).  The WM then keeps converging on its
                    # own losses.  P1 is unaffected (reward-MTP weight ~0 there).
                    if (wm_trunk_stopgrad_in_p2 and current_phase == 2
                            and isinstance(agent_hid, torch.Tensor)):
                        agent_hid = agent_hid.detach()
                    # P1 + P2 both train the reward MTP head (paper
                    # Algorithm 1: tokenizer + dynamics + reward + value
                    # are co-trained throughout WM pretraining).
                    # Previously P1 used only ``recon + sf``, leaving
                    # the reward head untrained until P2 (~10 iters of
                    # reward gradient at our default budget).
                    # validate-iter80 RCA (2026-05-06): reward head
                    # Pearson r = 0.16 with pred_std=2.8 vs real_std=80
                    # — under-trained.  Adding reward MTP to P1 gives
                    # ~3× more reward-head gradient updates over the
                    # full schedule.  BC loss is *not* added in P1
                    # because random-action episodes carry no expert
                    # signal; cloning them collapses the actor prior
                    # to uniform (preserves the existing P2-only BC
                    # rationale documented at TrainConfig.bc_scale).
                    if current_phase == 1:
                        # P39 diag C: optional stop-gradient on agent_hid
                        # before reward-MTP head, to isolate whether
                        # reward-MTP gradients distort the encoder/dynamics
                        # latent (without disabling the head's own
                        # learning, unlike diag B).
                        _hid_for_agent = agent_hid
                        if (diag_reward_mtp_stop_grad_in_p1
                                and not diag_disable_reward_mtp_in_p1
                                and isinstance(agent_hid, torch.Tensor)):
                            _hid_for_agent = agent_hid.detach()
                        ag_losses = agent_finetune_loss(model, batch,
                                                          _hid_for_agent, cfg)
                # Phase 2: also update reward + policy via MTP (eq. 9).
                if current_phase == 2:
                    with torch.amp.autocast(device_type=device.type,
                                              dtype=torch.bfloat16,
                                              enabled=(device.type == 'cuda')):
                        ag_losses = agent_finetune_loss(model, batch,
                                                          agent_hid, cfg)
                    if wm_freeze_after_p1:
                        # P90: WM core frozen in P2 — drop wm_total (its grads
                        # would hit frozen params).  The reward head still
                        # trains via agent_total→reward-MTP (it is NOT frozen).
                        total_loss = ag_losses['agent_total']
                    else:
                        total_loss = wm_losses['wm_total'] + ag_losses['agent_total']
                elif current_phase == 1:
                    # P1: WM losses + (optional) reward-head MTP.
                    # Paper default: reward_scale_loss_p1 = 0 (reward
                    # head trains only in P2+).  Diag B env var still
                    # honoured as a hard override to force-zero the P1
                    # weight regardless of cfg, for backward-compat with
                    # ad-hoc ablation runs.
                    p1_rmtp_weight = (0.0 if diag_disable_reward_mtp_in_p1
                                       else float(cfg.reward_scale_loss_p1))
                    if p1_rmtp_weight == 0.0:
                        total_loss = wm_losses['wm_total']
                    else:
                        # Use the *non-detached* ``reward_mtp_total`` key —
                        # using ``reward_mtp_loss`` (which is .detach()'d
                        # for diagnostics) silently zeros the reward-head
                        # gradient, leaving it untrained.
                        total_loss = (wm_losses['wm_total']
                                       + p1_rmtp_weight
                                         * ag_losses['reward_mtp_total'])
                else:
                    total_loss = wm_losses['wm_total']

                opt_world.zero_grad(set_to_none=True)
                if current_phase == 2:
                    opt_actor.zero_grad(set_to_none=True)
                # P39 diag A: per-head gradient norms (encoder/dynamics
                # subgraph) — answers "which loss term dominates the WM
                # gradient at iter N?".  Probe-cadence-gated so cost is
                # ~3 extra backwards every N iters; negligible at default
                # cadence 10 (<2% wall-clock).  Uses
                # ``torch.autograd.grad`` to compute partial grads w.r.t.
                # representative parameters of each subgraph.  Per-loss
                # ref parameter choice (P40 fix):
                #   - recon depends on tokenizer encoder + decoder.
                #   - sf depends on dynamics (z_clean is .detach()'d).
                #   - rmtp depends on dynamics + reward head (agent_hid is
                #     built from dynamics output on detached z_clean).
                # So a single tokenizer ref param only sees recon's
                # gradient — sf and rmtp return None (allow_unused=True).
                # Use a *dynamics* ref param so sf and rmtp both register;
                # for recon we additionally use a tokenizer ref param.
                if (diag_perhead_grads_every > 0
                        and total_iters > 0
                        and (total_iters % diag_perhead_grads_every) == 0
                        and current_phase in (1, 2)
                        and model.tokenizer is not None):
                    try:
                        # P41 fix: pass the *full* param list of the
                        # relevant submodule (not a single ``next()``
                        # param) so we sum grad norms over all in-graph
                        # params.  This eliminates the silent
                        # "first-param-not-in-graph → grad=0.0" failure
                        # mode that produced ``diag_grad_recon=0.0`` in
                        # P39/P40 even when recon was actually training.
                        _tok_params = [p for p in model.tokenizer.parameters()
                                          if p.requires_grad]
                        _dyn_params = [p for p in model.dynamics.parameters()
                                          if p.requires_grad]
                        _diag_terms = {
                            # (loss_term, [ref_params])
                            'recon': (wm_losses.get('recon_loss'), _tok_params),
                            'sf':    (wm_losses.get('sf_loss'),    _dyn_params),
                        }
                        if 'reward_mtp_total' in ag_losses:
                            # Match the actual rmtp weight applied to
                            # total_loss for this phase: paper-default
                            # is 0 in P1 (untouched dynamics gradient)
                            # and reward_scale_loss in P2.  Skip the
                            # diag entirely when the effective weight
                            # is 0 — gradient would be exactly 0 and
                            # the logged value would be meaningless.
                            if current_phase == 1:
                                _rmtp_w = (0.0 if diag_disable_reward_mtp_in_p1
                                            else float(cfg.reward_scale_loss_p1))
                            else:
                                _rmtp_w = float(cfg.reward_scale_loss)
                            if _rmtp_w != 0.0:
                                _diag_terms['rmtp'] = (
                                    _rmtp_w * ag_losses['reward_mtp_total'],
                                    _dyn_params)
                        diag_perhead_last = {}
                        for _name, (_lt, _refs) in _diag_terms.items():
                            if _lt is None or not torch.is_tensor(_lt):
                                continue
                            # P41 fix: try autograd.grad regardless of
                            # the requires_grad / grad_fn quick-check.
                            # Under bf16 autocast the loss tensor's
                            # ``grad_fn`` attribute can occasionally
                            # read as None on the autocast boundary
                            # while autograd still has the graph
                            # (observed as ``diag_grad_sf=-1.0`` in
                            # P39/P40).  Cast to fp32 first to be
                            # robust to autocast dtype mismatch with
                            # fp32 model params.
                            try:
                                _lt_f = _lt.float()
                                _gs = torch.autograd.grad(
                                    _lt_f, _refs,
                                    retain_graph=True,
                                    allow_unused=True)
                                _sq = 0.0
                                _hit = 0
                                for _g in _gs:
                                    if _g is None:
                                        continue
                                    _sq += float(_g.detach().float()
                                                   .pow(2).sum().item())
                                    _hit += 1
                                diag_perhead_last[f'diag_grad_{_name}'] = (
                                    _sq ** 0.5)
                                # Count of params in-graph — diagnostic
                                # cross-check for "did we actually reach
                                # any param of this submodule?".  0 →
                                # loss term is detached or constant.
                                diag_perhead_last[
                                    f'diag_grad_{_name}_nparams'] = _hit
                            except Exception as _e_inner:
                                diag_perhead_last[f'diag_grad_{_name}'] = -1.0
                                diag_perhead_last[
                                    f'diag_grad_{_name}_err'] = (
                                        type(_e_inner).__name__
                                        + ': '
                                        + str(_e_inner)[:120])
                    except Exception as _e:
                        # Non-fatal — diagnostic only.  Capture the message
                        # so silent autograd failures (graph freed, param not
                        # in graph, etc.) are visible in train_log.jsonl.
                        diag_perhead_last = {
                            'diag_grad_error': 1.0,
                            'diag_grad_error_msg': f'{type(_e).__name__}: {str(_e)[:120]}',
                        }
                total_loss.backward()
                wm_grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters_world(), cfg.grad_clip)
                if current_phase == 2:
                    actor_grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters_actor(), cfg.grad_clip)
                if not torch.isfinite(wm_grad_norm):
                    n_grad_skip += 1
                    opt_world.zero_grad(set_to_none=True)
                    if current_phase == 2:
                        opt_actor.zero_grad(set_to_none=True)
                else:
                    opt_world.step()
                    if current_phase == 2 and torch.isfinite(actor_grad_norm):
                        opt_actor.step()
                t_wm_acc += time.time() - _t

            else:  # Phase 3 — imagination RL + continuous WM update
                # Paper Algorithm 1: tokenizer + dynamics + reward head
                # keep training in P3 alongside the actor / critic so the
                # world model tracks the on-policy state-visit
                # distribution.  We split the gradient steps: each P3
                # iter does (a) one WM step on a fresh buffer batch
                # (recon/sf + reward MTP), (b) one actor/critic step
                # via imagination from a possibly different batch.
                _t = time.time()
                # mbrl2 real-sim controller: the WM(RSSM)+DOB is a FROZEN
                # OBSERVER — there is NO P3 world-model update.  Train ONLY the
                # actor + critic on λ-returns from the REAL simulator (the
                # on-policy ``batch`` collected by ``collect_episode``, with
                # domain randomisation), via ``_realsim_actor_critic_step``.
                # ``wm_losses``/``ag_losses`` stay empty so the log-row merge
                # ``{**wm_losses, **ag_losses, **ac_losses}`` is a clean no-op.
                wm_grad_norm = 0.0
                wm_losses, ag_losses = {}, {}
                # Fix 2 (p03 RCA): the CRITIC trains on the DIVERSE shared replay
                # buffer (``buf``) — not the narrow on-policy ``batch`` — to keep
                # the value head from inverting once the actor drifts to a corner.
                # The ACTOR stays on-policy (``batch``).  When there is no separate
                # on-policy buffer (fallback), ``batch`` already IS the replay, so
                # leave ``critic_batch`` None (critic shares the actor states).
                _critic_batch = None
                if (onpol_buf is not None and onpol_buf.filled > 0
                        and buf.filled > 0):
                    _cb_np = buf.sample(cfg.batch_size, cfg.seq_len, rng)
                    _critic_batch = {}
                    for _k, _v in _cb_np.items():
                        _t = torch.from_numpy(_v)
                        if device.type == 'cuda':
                            _t = _t.pin_memory().to(device, non_blocking=True)
                        else:
                            _t = _t.to(device)
                        _critic_batch[_k] = _t
                with torch.amp.autocast(device_type=device.type,
                                          dtype=torch.bfloat16,
                                          enabled=(device.type == 'cuda')):
                    ac_losses = _realsim_actor_critic_step(
                        model, batch, cfg, critic_batch=_critic_batch)
                _actor_loss = ac_losses['actor_loss']
                opt_actor.zero_grad(set_to_none=True)
                opt_critic.zero_grad(set_to_none=True)
                (_actor_loss
                 + ac_losses['critic_loss']).backward()
                actor_grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters_actor(), cfg.grad_clip)
                critic_grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters_critic(), cfg.grad_clip)
                if (not torch.isfinite(actor_grad_norm)
                        or not torch.isfinite(critic_grad_norm)):
                    n_grad_skip += 1
                    opt_actor.zero_grad(set_to_none=True)
                    opt_critic.zero_grad(set_to_none=True)
                else:
                    # P93 critic warmup: for the first
                    # ``p3_critic_warmup_iters`` P3 iters, train the CRITIC only
                    # (actor frozen at the snapshot prior) so the value head is
                    # CALIBRATED before it is coupled to the actor — stops the
                    # cold-critic cascade.
                    if (p3_critic_warmup_iters > 0
                            and p3_iters < p3_critic_warmup_iters):
                        opt_critic.step()
                        if not _critic_warmup_logged:
                            print(f"[critic-warmup] training critic only for "
                                  f"{p3_critic_warmup_iters} iters "
                                  f"(actor frozen)", flush=True)
                            _critic_warmup_logged = True
                    else:
                        opt_actor.step()
                        opt_critic.step()
                model.update_target(cfg.target_critic_tau)
                t_ac_acc += time.time() - _t

        # Periodically refresh the KL trust-region prior so KL(π‖π_prior) tracks
        # a slowly-moving target (a MOVING trust region, anti-hunting) instead of
        # a static once-at-start snapshot.  JOINT: joint_prior_refresh_iters.
        # PHASED P3 (p136): p3_prior_refresh_iters (the actor_kl_coef trust
        # region needs a recent prior to BE a trust region, not an entropy pull).
        _refresh_n = (joint_prior_refresh_iters if joint_mode
                      else p3_prior_refresh_iters)
        if (current_phase == 3 and p3_iters > 0 and _refresh_n > 0
                and (p3_iters % _refresh_n) == 0):
            model.snapshot_prior_policy()

        total_iters += 1
        if current_phase == 3:
            p3_iters += 1

        if total_iters % cfg.log_every == 0:
            now = time.time()
            iter_dt = now - last_log_time
            steps_dt = total_env_steps - last_log_steps
            sps = (steps_dt / iter_dt) if iter_dt > 0 else 0.0
            gpu_mem_mb = None
            gpu_mem_peak_mb = None
            if device.type == 'cuda':
                try:
                    gpu_mem_mb = torch.cuda.memory_allocated(device) / (1024 ** 2)
                    gpu_mem_peak_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
                    torch.cuda.reset_peak_memory_stats(device)
                except Exception:
                    pass
            row = {
                'iter': total_iters,
                'phase': current_phase,
                'env_steps': total_env_steps,
                'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
                'wallclock_s': now - start_time,
                'iter_time_s': iter_dt,
                'env_steps_per_s': sps,
                'ema_return': float(ema_return) if ema_return is not None else None,
                'buf_eps': buf.filled,
                'wm_grad_norm': float(wm_grad_norm.detach().item()
                                        if torch.is_tensor(wm_grad_norm)
                                        else wm_grad_norm),
                'actor_grad_norm': float(actor_grad_norm.detach().item()
                                           if torch.is_tensor(actor_grad_norm)
                                           else actor_grad_norm),
                'critic_grad_norm': float(critic_grad_norm.detach().item()
                                            if torch.is_tensor(critic_grad_norm)
                                            else critic_grad_norm),
                'gpu_mem_mb': gpu_mem_mb,
                'gpu_mem_peak_mb': gpu_mem_peak_mb,
                't_collect_s': t_collect_acc,
                't_sample_s': t_sample_acc,
                't_wm_s': t_wm_acc,
                't_ac_s': t_ac_acc,
                'return_window_mean': (float(np.mean(return_window))
                                        if return_window else None),
                'return_window_std': (float(np.std(return_window))
                                        if len(return_window) > 1 else None),
                'return_window_n': int(len(return_window)),
                'iter_return_mean': (float(np.mean(iter_returns))
                                      if iter_returns else None),
                'iter_raw_return_mean': (float(np.mean(iter_raw_returns))
                                          if iter_raw_returns else None),
                'iter_cv_violation_mean': (float(np.mean(iter_cv_violations))
                                            if iter_cv_violations else None),
                'iter_mv_violation_mean': (float(np.mean(iter_mv_violations))
                                            if iter_mv_violations else None),
                'n_grad_skip': int(n_grad_skip),
                'buf_fill_pct': float(buf.filled) / max(1, buf.capacity_eps),
                'wm_frozen': bool(_wm_frozen_now),
                'expert_det_return': last_expert_det_return,
                'agent_minus_expert_return': last_agent_minus_expert,
            }
            for k, v in {**wm_losses, **ag_losses, **ac_losses}.items():
                row[k] = float(v.detach().item() if torch.is_tensor(v) else v)
            # P39 diag A: emit last computed per-head grad norms (if any).
            # Values may be float (grad norms) or str (error messages); pass
            # strings through unchanged so jsonl serialisation works.
            for _k, _v in diag_perhead_last.items():
                if isinstance(_v, str):
                    row[_k] = _v
                else:
                    row[_k] = float(_v)
            # P39 diag D: latent stability probe.  Cheap (one tokenizer
            # forward on N fixed transitions; <0.5% overhead at cadence
            # 10).  Tracks cosine similarity of encoder outputs against
            # a reference set seeded at first call.  A sharp drop ==
            # the encoder is re-organising its representation; correlate
            # with iter index of any fidelity-probe cliff.
            if (diag_latent_stability_every > 0
                    and total_iters > 0
                    and (total_iters % diag_latent_stability_every) == 0
                    and current_phase in (1, 2)
                    and model.tokenizer is not None):
                try:
                    _N = 64
                    if diag_latent_ref is None:
                        _b = buf.sample(min(_N, max(1, buf.filled)),
                                          cfg.seq_len, rng)
                        _obs0 = torch.from_numpy(_b['obs']).to(device)
                        with torch.no_grad():
                            _z0 = model.tokenizer.encode(_obs0).float()
                        diag_latent_ref = {'obs': _obs0, 'z': _z0,
                                            'iter': int(total_iters)}
                        row['diag_latent_ref_iter'] = int(total_iters)
                        row['diag_latent_cos_mean'] = 1.0
                    else:
                        with torch.no_grad():
                            _z_now = model.tokenizer.encode(
                                diag_latent_ref['obs']).float()
                        # Per-token cosine sim, then mean.
                        _a = _z_now.reshape(-1, _z_now.shape[-1])
                        _b = diag_latent_ref['z'].reshape(-1, _z_now.shape[-1])
                        _cos = torch.nn.functional.cosine_similarity(
                            _a, _b, dim=-1)
                        row['diag_latent_ref_iter'] = int(
                            diag_latent_ref['iter'])
                        row['diag_latent_cos_mean'] = float(_cos.mean().item())
                        row['diag_latent_cos_min'] = float(_cos.min().item())
                except Exception as _e:
                    row['diag_latent_error'] = str(_e)[:80]
            # P52 RCA: feed the P2→P3 gate from the freshly-logged row.
            # ``reward_mtp_loss`` is the P2 reward-head loss — the
            # critic head doesn't train until P3 in this codebase.
            if current_phase == 2:
                _rml = row.get('reward_mtp_loss')
                try:
                    if _rml is not None and np.isfinite(float(_rml)):
                        p2_reward_mtp_recent.append(float(_rml))
                except (TypeError, ValueError):
                    pass
            log_f.write(json.dumps(row) + '\n')
            log_f.flush()
            rwm = row.get('return_window_mean')
            rwm_str = f"{rwm:+.2f}" if rwm is not None else 'n/a'
            ema_str = (f"{row['ema_return']:.2f}"
                        if row['ema_return'] is not None else 'n/a')
            # JOINT mode has no phases -> tag rows 'joint' instead of 'P3';
            # phased mode keeps the informative P1/P2/P3 label.
            phase_tag = 'joint' if joint_mode else f'P{current_phase}'
            print(f"[{row['timestamp']}] {phase_tag} iter {total_iters:4d} "
                  f"steps {total_env_steps:6d} sps {sps:5.1f} "
                  f"ret_ema {ema_str} ret_w {rwm_str} "
                  f"recon {row.get('recon_loss', 0.0):.4f} "
                  f"sf {row.get('sf_loss', 0.0):.4f} "
                  f"encvar {row.get('encoder_var_ratio', 0.0):.2f} "
                  f"zrank {row.get('z_eff_rank', 0.0):.1f}/{int(row.get('z_dim', 0))} "
                  f"alive {int(row.get('z_alive_dims', 0))} "
                  f"bc {row.get('bc_loss', 0.0):.3f} "
                  f"actor {row.get('actor_loss', 0.0):+.3f} "
                  f"critic {row.get('critic_loss', 0.0):.3f} "
                  f"ent {row.get('entropy_mean', 0.0):.3f} "
                  f"img_ret {row.get('imagined_return_mean', 0.0):+.3f} "
                  f"skip {row.get('n_grad_skip', 0)}",
                  flush=True)
            last_log_time = now
            last_log_steps = total_env_steps
            t_collect_acc = t_sample_acc = t_wm_acc = t_ac_acc = 0.0
            iter_returns = []
            iter_cv_violations = []
            iter_mv_violations = []
            iter_raw_returns = []

            # ----- Periodic WM-fidelity probe (P1 / P2 only) -----
            # 2026-05-10 (run_p13): the sf_loss trace alone hides
            # whether the WM is actually learning predictive dynamics.
            # Periodically run the same probe used at the P1->P2 gate
            # so we can watch r(H=1..H) trend during training rather
            # than waiting for the phase transition.  Default every
            # 10 log-iters; disable via DREAMER_WM_PROBE_EVERY_ITERS=0.
            try:
                _probe_every = int(os.environ.get(
                    'DREAMER_WM_PROBE_EVERY_ITERS', '10') or 0)
            except ValueError:
                _probe_every = 0
            if (_probe_every > 0
                    and current_phase in (1, 2)
                    and total_iters > 0
                    and (total_iters % _probe_every) == 0):
                try:
                    _pbe = _probe_wm_fidelity(model, env, device, cfg)
                    if _pbe is not None:
                        print(f"[wm-fidelity-probe-iter{total_iters}] "
                              f"{_pbe['summary']} "
                              f"floor={_pbe['r_floor']:.2f} "
                              f"best_h={_pbe['best_h']}/{_pbe['H']}"
                              + (f" gain_fid={_pbe['wm_gain_fidelity']:.3f}"
                                 if _pbe.get('wm_gain_fidelity') is not None
                                 else ''),
                              flush=True)
                        # ---- Fidelity-based best-ckpt + early stop ----
                        # Score = sum of positive Pearson r across
                        # probed horizons + small bonus for depth.
                        # Robust to single-horizon noise.
                        _per = _pbe.get('per_offset') or []
                        _r_vals = [float(r) for (_, r) in _per]
                        # Score: sum-of-positive-r PLUS a depth bonus
                        # weighted heavily enough to actually win ties.
                        # 2026-05-24 (P44 RCA): bonus weight 0.05 → 0.5.
                        # P44 iter 30 had best_h=7/15 (one horizon
                        # crossed floor, real improvement) but lost the
                        # "best" race against iter 10's best_h=0/15 by
                        # 0.08 of sum-of-r noise because the old bonus
                        # (0.05*7/15=0.023) was 3 orders of magnitude
                        # too small. With bonus=0.5*best_h/H, a single
                        # horizon crossing the floor adds 0.033 — enough
                        # to overcome typical r-sum noise (<0.10) while
                        # not dominating a genuinely better probe.
                        _score = (sum(max(0.0, r) for r in _r_vals)
                                   + 0.5 * float(_pbe.get('best_h', 0))
                                            / max(1, int(_pbe.get('H', 1))))
                        # P89: convergence-aware term so best-ckpt selection +
                        # P1->P2 restore are not blind to held-action drift (the
                        # r-terms are scale-invariant — a drifting WM scores as
                        # high as a converged one).  Gated on best_h>0 so a
                        # degenerate flat-but-"converged" WM earns no credit.
                        # Weight via DREAMER_WM_FIDELITY_CONV_WEIGHT (0=off).
                        _conv_frac = _pbe.get('wm_converge_frac')
                        _conv_w = float(os.environ.get(
                            'DREAMER_WM_FIDELITY_CONV_WEIGHT', '1.0'))
                        if (_conv_w > 0.0 and _conv_frac is not None
                                and int(_pbe.get('best_h', 0)) > 0):
                            _score = _score + _conv_w * float(_conv_frac)
                        # P92 RCA (2026-06-06): the correlation + conv terms
                        # above are SHAPE/scale-invariant and PEAK EARLY — p92's
                        # score peaked at iter 30 (recon 0.247) and never
                        # improved, so the freeze/restore grabbed an UNDER-
                        # TRAINED WM even though recon kept falling to 0.145 by
                        # iter 100 (a strictly better-reconstructing, higher-
                        # gain WM that scored LOWER on shape alone).  Add an
                        # ABSOLUTE one-step-fidelity term: reward lower recon so
                        # the "best" tracks a genuinely better-trained WM (lower
                        # recon ⇒ better gain, the control-relevant property).
                        # Weight via DREAMER_WM_FIDELITY_RECON_WEIGHT (0=off,
                        # legacy).  Uses the most recent training recon_loss.
                        _recon_w = float(os.environ.get(
                            'DREAMER_WM_FIDELITY_RECON_WEIGHT', '3.0'))
                        if _recon_w > 0.0:
                            try:
                                _rl = wm_losses.get('recon_loss')
                                _rlv = float(_rl.detach().item()
                                              if torch.is_tensor(_rl)
                                              else _rl)
                                if np.isfinite(_rlv):
                                    _score = _score - _recon_w * _rlv
                            except Exception:
                                pass
                        # GAIN-FIDELITY term (p126 RCA, 2026-06-17): reward the
                        # checkpoint whose open-loop CV std-ratio is closest to
                        # the real plant (= accurate CV gain), so the wm_best
                        # pick + P1->P2 restore stop riding correlation noise and
                        # the frozen WM gain is both HIGHER and STABLE run-to-run
                        # (the DV-gain under-read → actor under-reaction → cv
                        # blow-up chain).  Gated by a recon ramp so an untrained,
                        # high-variance early checkpoint cannot win on spurious
                        # CV variance (mirrors the overshoot/held recon gates).
                        # Weight via DREAMER_WM_FIDELITY_GAIN_WEIGHT (0=off).
                        _gain_w = float(os.environ.get(
                            'DREAMER_WM_FIDELITY_GAIN_WEIGHT', '3.0'))
                        _gain_fid = _pbe.get('wm_gain_fidelity')
                        if _gain_w > 0.0 and _gain_fid is not None:
                            try:
                                _rl = wm_losses.get('recon_loss')
                                _rlv = float(_rl.detach().item()
                                              if torch.is_tensor(_rl)
                                              else _rl)
                                _gate_thr = float(os.environ.get(
                                    'DREAMER_WM_FIDELITY_GAIN_GATE_RECON', '0.15'))
                                _gate = (min(1.0, _gate_thr / max(_rlv, 1e-6))
                                          if np.isfinite(_rlv) else 0.0)
                                _score = _score + _gain_w * float(_gain_fid) * _gate
                            except Exception:
                                pass
                        # Update EMA of the score (used for ES only).
                        if wm_score_ema <= -1e17:
                            wm_score_ema = float(_score)
                        else:
                            wm_score_ema = (
                                wm_score_ema_alpha * float(_score) +
                                (1.0 - wm_score_ema_alpha) * wm_score_ema)
                        # P52 RCA: history feeds the P1→P2 gate.
                        p1_score_ema_history.append(
                            (int(total_iters), float(wm_score_ema)))
                        # Track raw-best (checkpoint promotion) and
                        # EMA-best (ES decisions) separately.
                        if _score > wm_best_score:
                            wm_best_score = _score
                            wm_best_iter = int(total_iters)
                            wm_best_ckpt_path = out_dir / 'wm_best.pt'
                            torch.save({
                                'model': model.state_dict(),
                                'cfg': asdict(cfg),
                                'obs_norm': env.get_obs_norm_stats(),
                                'wm_fidelity_score': float(_score),
                                'wm_fidelity_probe': {
                                    'iter': int(total_iters),
                                    'env_steps': int(total_env_steps),
                                    'per_offset': [(int(o), float(r))
                                                    for (o, r) in _per],
                                    'best_h': int(_pbe.get('best_h', 0)),
                                    'H': int(_pbe.get('H', 0)),
                                    'wm_converge_frac': (
                                        float(_pbe['wm_converge_frac'])
                                        if _pbe.get('wm_converge_frac')
                                        is not None else None),
                                    'tail_drift_mean': (
                                        float(_pbe['tail_drift_mean'])
                                        if _pbe.get('tail_drift_mean')
                                        is not None else None),
                                },
                            }, wm_best_ckpt_path)
                            print(f"[wm-best] new best fidelity score "
                                  f"{_score:.3f} at iter {total_iters} "
                                  f"-> saved {wm_best_ckpt_path.name}",
                                  flush=True)
                        # EMA-best tracking for the phase gates.
                        # 2026-05-26 (P53 RCA): previously scoped to
                        # P2 only, which left ``wm_score_ema_best`` at
                        # the -1e18 sentinel during P1.  The P1→P2
                        # gate reads this field, so it could never
                        # pass on the EMA criterion and burned through
                        # the full extension cap.  Update continuously
                        # in P1 as well; the P2-entry baseline reset
                        # still gives the P2 patience window a fresh
                        # clock.
                        if wm_score_ema > wm_score_ema_best:
                            wm_score_ema_best = wm_score_ema
                            wm_score_ema_best_iter = int(total_iters)
                        if current_phase == 2 and wm_es_p2_baseline_iter < 0:
                            wm_es_p2_baseline_iter = int(total_iters)
                            # Reset EMA-best on P2 entry so the P2
                            # patience window starts fresh.
                            wm_score_ema_best = wm_score_ema
                            wm_score_ema_best_iter = int(total_iters)
                        if (es_enable
                              and current_phase == 2
                              and wm_score_ema_best_iter > 0
                              and total_iters >= wm_fidelity_warmup_iters
                              and (total_iters - wm_score_ema_best_iter)
                                    >= wm_fidelity_patience_iters):
                            # 2026-05-27 (P57 RCA): suppress wm-fidelity
                            # ES when P2 is close to its nominal end —
                            # the WM has done its job in P1, and P2's
                            # remaining iters are reward-MTP training,
                            # not WM fidelity.  Killing P2 within
                            # ``p1_gate_check_step`` env_steps of the
                            # P2→P3 transition starves P3 of any
                            # budget at all (P57 outcome).  Threshold
                            # uses the existing p1 10 %-budget step
                            # for consistency.
                            _p2_end_env = p1 + p1_ext_steps + p2 + p2_ext_steps
                            _p2_remaining = _p2_end_env - total_env_steps
                            if _p2_remaining < p1_gate_check_step:
                                print(f"[wm-fidelity-ES] suppressed at "
                                      f"iter {total_iters}: P2 remaining "
                                      f"{_p2_remaining} env_steps < "
                                      f"{p1_gate_check_step}; deferring "
                                      f"to natural P2→P3 transition",
                                      flush=True)
                            else:
                                early_stop_reason = (
                                    f'wm_fidelity_degradation: EMA score '
                                    f'no improvement over '
                                    f'ema_best={wm_score_ema_best:.3f} '
                                    f'(iter {wm_score_ema_best_iter}) for '
                                    f'{total_iters - wm_score_ema_best_iter} iters '
                                    f'(patience={wm_fidelity_patience_iters}, '
                                    f'ema_alpha={wm_score_ema_alpha:.2f}); '
                                    f'raw_best={wm_best_score:.3f} '
                                    f'(iter {wm_best_iter})')
                except Exception as _e:
                    print(f"[wm-fidelity-probe-iter{total_iters}] "
                          f"error: {_e!r}", flush=True)

            # ----- Early-stop detection (per log iter) -----
            if es_enable:
                # Capture P1 initial sf_loss on the first P1 log iter that has
                # sf_loss available (the very first iter often has 0.0 dummy).
                if (current_phase == 1 and p1_initial_sf is None
                        and 'sf_loss' in row and row['sf_loss'] > 1e-6):
                    p1_initial_sf = float(row['sf_loss'])

                # --- Hard fails ---
                # Grad-skip storm.
                cur_total_skip = int(row.get('n_grad_skip', 0))
                d_skip = cur_total_skip - grad_skip_prev_total
                grad_skip_prev_total = cur_total_skip
                grad_skip_history.append((total_iters, d_skip))
                window_iters = max(1, int(getattr(cfg,
                            'early_stop_grad_skip_window_iters', 100)))
                window_max = int(getattr(cfg, 'early_stop_grad_skip_max', 5))
                # Count skips within the last ``window_iters`` iters.
                window_skips = sum(s for (it_, s) in grad_skip_history
                                    if it_ > total_iters - window_iters)
                if window_skips > window_max:
                    early_stop_reason = (
                        f'grad_skip_storm: {window_skips} skips in last '
                        f'{window_iters} iters (>{window_max})')

                # P3-only hard fails.
                if early_stop_reason is None and current_phase == 3:
                    ent = row.get('entropy_mean')
                    if ent is not None:
                        thr = (float(getattr(cfg,
                                'early_stop_entropy_collapse_frac', 0.20))
                               * n_action_bins_log)
                        # Floor-relative collapse threshold
                        # (2026-05-14, run_p22 RCA): the legacy
                        # ceiling-relative formula
                        #   thr = H_max(σ_max) − 0.10
                        # tripped on every healthy run with a tight
                        # σ_max because *any* shrink below the ceiling
                        # (even down to σ_max/2) was flagged as
                        # collapse.  σ legitimately moves below the
                        # ceiling as the actor learns to commit;
                        # collapse only matters when σ is approaching
                        # σ_min (the floor) and exploration is dying.
                        # Recompute thr from σ_min:
                        #   H_floor(σ_min) = log_std_min + 0.5·log(2πe)
                        #   thr = H_floor + margin
                        # Trip only when entropy stays within ``margin``
                        # nats of the floor (default 0.20 ≈ σ within
                        # 22% of σ_min).  Keep the legacy ``thr`` as
                        # the *upper* bound (still trip on truly
                        # silent policies that fall below
                        # 0.20·log(n_bins)), so the floor-relative
                        # check only loosens the trip, never tightens
                        # it past the legacy default.
                        if str(getattr(cfg, 'policy_type', 'continuous')
                                ).lower() == 'continuous':
                            log_std_min = float(getattr(cfg,
                                    'policy_log_std_min', -2.3))
                            unit_g = 0.5 * math.log(2.0 * math.pi * math.e)
                            h_floor = (
                                float(cfg.action_dim)
                                * (log_std_min + unit_g))
                            margin = float(getattr(cfg,
                                    'early_stop_entropy_collapse_floor_margin',
                                    0.20))
                            floor_aware_thr = h_floor + margin
                            # Use the *lower* of the two: the heuristic
                            # is "collapse when entropy is essentially
                            # at the floor".  Floor-relative thr is
                            # always ≤ legacy thr for any σ_min ≤ 1,
                            # so this only loosens the trip on tight
                            # σ_max regimes — never overrides a true
                            # discrete-policy entropy crash.
                            if floor_aware_thr < thr:
                                thr = floor_aware_thr
                        # Maintain sliding window of P3 entropy values.
                        ent_window.append(float(ent))
                        win_n = max(2, int(getattr(cfg,
                                'early_stop_entropy_collapse_window_iters',
                                30)))
                        if len(ent_window) > win_n:
                            ent_window.pop(0)
                        # Sliding-window detector: trip when a sufficient
                        # fraction of the last ``win_n`` entropies is
                        # below ``thr``.
                        # PERFORMANCE-AWARE GATE (Fix B, 2026-06-22): a low σ is
                        # only a COLLAPSE if the policy is also DEGENERATE — a
                        # healthy actor that has legitimately COMMITTED (low σ,
                        # GOOD returns) reads the same entropy.  The p136
                        # collapse had imag_adv_action_corr crash 0.77→0.014;
                        # a committed-and-learning policy keeps it high.  So
                        # only trip when the recent advantage-action correlation
                        # is also low (genuine degeneracy), never on a
                        # performing low-σ policy.  ``adv_corr`` None (pre-corr
                        # logging) ⇒ fall back to the entropy-only trip.
                        _ac = row.get('imag_adv_action_corr')
                        if _ac is not None:
                            adv_corr_window.append(abs(float(_ac)))
                            if len(adv_corr_window) > win_n:
                                adv_corr_window.pop(0)
                        if len(ent_window) >= win_n:
                            n_below = sum(1 for e in ent_window if e < thr)
                            min_below = (float(getattr(cfg,
                                    'early_stop_entropy_collapse_min_frac_below',
                                    0.70))
                                          * win_n)
                            _corr_gate = float(getattr(cfg,
                                    'early_stop_entropy_collapse_min_adv_corr',
                                    0.05))
                            _recent_corr = (float(np.median(adv_corr_window))
                                            if adv_corr_window else 0.0)
                            _degenerate = (_recent_corr < _corr_gate
                                           or not adv_corr_window)
                            if n_below >= min_below and _degenerate:
                                early_stop_reason = (
                                    f'entropy_collapse_window: '
                                    f'{n_below}/{win_n} iters below '
                                    f'thr={thr:.3f} '
                                    f'(latest={ent:.3f}, '
                                    f'adv_corr={_recent_corr:.3f})')
                        # Legacy consecutive-streak detector (kept as a
                        # fallback for very long sustained collapse).  Same
                        # performance gate: only a DEGENERATE low-σ streak trips.
                        if ent < thr:
                            ent_collapse_streak += 1
                        else:
                            ent_collapse_streak = 0
                        _streak_corr = (float(np.median(adv_corr_window))
                                        if adv_corr_window else 0.0)
                        _streak_degen = (
                            _streak_corr < float(getattr(cfg,
                                'early_stop_entropy_collapse_min_adv_corr', 0.05))
                            or not adv_corr_window)
                        if (early_stop_reason is None
                                and _streak_degen
                                and ent_collapse_streak >= int(getattr(cfg,
                                'early_stop_entropy_collapse_patience_iters',
                                30))):
                            early_stop_reason = (
                                f'entropy_collapse: ent={ent:.3f} < '
                                f'{thr:.3f} for {ent_collapse_streak} iters '
                                f'(adv_corr={_streak_corr:.3f})')

                    cl = row.get('critic_loss')
                    if cl is not None and cl > 0:
                        if len(critic_loss_window) >= 20:
                            med = float(np.median(critic_loss_window))
                            factor = float(getattr(cfg,
                                    'early_stop_critic_divergence_factor', 5.0))
                            if med > 1e-8 and cl > factor * med:
                                critic_div_streak += 1
                            else:
                                critic_div_streak = 0
                            if critic_div_streak >= int(getattr(cfg,
                                    'early_stop_critic_divergence_patience_iters',
                                    20)):
                                early_stop_reason = (
                                    f'critic_divergence: critic_loss={cl:.3f} > '
                                    f'{factor:.1f}× median={med:.3f} for '
                                    f'{critic_div_streak} iters')
                        critic_loss_window.append(float(cl))

                    # --- Bootstrap-cascade canary (P41 RCA, 2026-05-23) ---
                    # Trip when the critic target is bootstrap-dominated
                    # (rew/tgt variance fraction < threshold) AND
                    # return_scale has grown >factor× since P3 start,
                    # sustained for patience consecutive P3 iters.
                    # See critic-diag interpretation table in
                    # dreamer-training-diagnosis skill.
                    if early_stop_reason is None:
                        _rv = row.get('critic_rew_to_tgt_var')
                        _rs = row.get('return_scale')
                        _min_rv = float(getattr(cfg,
                                'early_stop_cascade_min_rew_var_frac', 0.0))
                        _min_gr = float(getattr(cfg,
                                'early_stop_cascade_min_return_scale_growth',
                                0.0))
                        _patience = int(getattr(cfg,
                                'early_stop_cascade_patience_iters', 0))
                        if (_min_rv > 0.0 and _patience > 0
                                and _rv is not None and _rs is not None
                                and p3_start_return_scale is not None
                                and np.isfinite(float(_rv))
                                and np.isfinite(float(_rs))):
                            _rv_f = float(_rv)
                            _growth = float(_rs) / max(
                                p3_start_return_scale, 1e-8)
                            if _rv_f < _min_rv and _growth > _min_gr:
                                cascade_streak += 1
                            else:
                                cascade_streak = 0
                            if cascade_streak >= _patience:
                                early_stop_reason = (
                                    f'bootstrap_cascade: '
                                    f'rew_to_tgt_var={_rv_f:.4f} < {_min_rv:.4f} '
                                    f'AND return_scale_growth={_growth:.2f}× > '
                                    f'{_min_gr:.1f}× for '
                                    f'{cascade_streak} iters')

                # --- Soft fail: P3 plateau on best deterministic-eval ---
                # P49 RCA (2026-05-25): track best on ``return_window``
                # (deterministic eval, deque maxlen=10) instead of
                # ``ema_return`` (stochastic on-policy EMA).  These
                # decouple for narrow-σ process-control policies, and
                # the deterministic signal is what ``validate.py`` and
                # deployment actually run.
                det_ret = (float(np.mean(return_window))
                           if return_window else None)
                det_n = len(return_window)
                min_window_n = int(getattr(cfg,
                        'early_stop_p3_min_window_n', 5))
                if (early_stop_reason is None and current_phase == 3
                        and det_ret is not None and np.isfinite(det_ret)
                        and det_n >= min_window_n):
                    min_imp = float(getattr(cfg,
                            'early_stop_p3_min_improvement', 0.01))
                    if best_p3_ema is None:
                        best_p3_ema = float(det_ret)
                        best_p3_iter = total_iters
                        iters_since_best = 0
                        # Save first qualifying best so we have a ckpt
                        # to promote even if no further improvement.
                        try:
                            best_ckpt_path = out_dir / 'best.pt'
                            torch.save({'model': model.state_dict(),
                                        'cfg': asdict(cfg),
                                        'obs_norm': env.get_obs_norm_stats(),
                                        'best_det_return': best_p3_ema,
                                        'best_window_n': det_n,
                                        'best_iter': best_p3_iter},
                                       best_ckpt_path)
                        except Exception as _e:
                            print(f'[early-stop] best-ckpt save failed: '
                                   f'{_e!r}', flush=True)
                    else:
                        # Relative improvement against |best| with floor 1.0
                        # so trials hovering near 0 don't ratchet on noise.
                        denom = max(1.0, abs(best_p3_ema))
                        improvement = (det_ret - best_p3_ema) / denom
                        if improvement >= min_imp:
                            best_p3_ema = float(det_ret)
                            best_p3_iter = total_iters
                            iters_since_best = 0
                            # Persist best ckpt for plateau-stop recovery.
                            try:
                                best_ckpt_path = out_dir / 'best.pt'
                                torch.save({'model': model.state_dict(),
                                            'cfg': asdict(cfg),
                                            'obs_norm': env.get_obs_norm_stats(),
                                            'best_det_return': best_p3_ema,
                                            'best_window_n': det_n,
                                            'best_iter': best_p3_iter},
                                           best_ckpt_path)
                            except Exception as _e:
                                print(f'[early-stop] best-ckpt save failed: '
                                       f'{_e!r}', flush=True)
                        else:
                            iters_since_best += 1
                    p3_patience = int(getattr(cfg,
                            'early_stop_p3_patience_iters', 200))
                    warmup = int(getattr(cfg,
                            'phase3_pruner_warmup_iters', 8))
                    if (iters_since_best >= p3_patience
                            and p3_iters > warmup):
                        early_stop_reason = (
                            f'p3_plateau: no >+{min_imp*100:.1f}% improvement '
                            f'over best_det_return={best_p3_ema:.3f} for '
                            f'{iters_since_best} iters')

                if early_stop_reason is not None:
                    print(f'[early-stop] tripped: {early_stop_reason}',
                           flush=True)
                    break

        if on_iter_end is not None:
            # Gate the BO pruner to *post-warmup P3 only*.  Reporting
            # P1/P2 random-action signals (or early P3 dominated by
            # the snapshot actor) makes the pruner kill trials on
            # pre-learning noise, not on actual learning quality.
            # P49 RCA (2026-05-25): report deterministic-eval window
            # mean instead of stochastic ema_return so the BO pruner
            # ranks trials by the same signal used to pick best.pt.
            warmup = max(0, int(getattr(cfg, 'phase3_pruner_warmup_iters', 0)))
            if current_phase == 3 and p3_iters > warmup:
                _det = (float(np.mean(return_window))
                        if return_window else None)
                try:
                    stop = bool(on_iter_end(int(total_iters),
                                              int(total_env_steps),
                                              float(_det)
                                              if _det is not None else 0.0))
                except Exception:
                    stop = False
                if stop:
                    print(f'[train] early-stop requested at iter {total_iters}',
                          flush=True)
                    break
        if total_iters % cfg.save_every_iters == 0:
            torch.save({'model': model.state_dict(), 'cfg': asdict(cfg),
                        'obs_norm': env.get_obs_norm_stats()},
                       out_dir / f'ckpt_iter_{total_iters:05d}.pt')

    log_f.close()
    final_path = out_dir / 'final.pt'
    torch.save({'model': model.state_dict(), 'cfg': asdict(cfg),
                'obs_norm': env.get_obs_norm_stats()}, final_path)

    # Best-checkpoint promotion: regardless of early-stop status, the
    # snapshot at ``best_p3_iter`` is a more trustworthy controller than
    # the most recent state (which may have drifted past the best EMA
    # return). Promote it to final.pt so validation + ONNX export +
    # downstream BO trial comparison all pick the best policy
    # automatically. Falls back to the most-recent-iter ``final.pt`` we
    # just wrote if there is no best checkpoint (e.g. P3 produced no
    # improvement at all).
    if (best_ckpt_path is not None and Path(best_ckpt_path).exists()):
        try:
            shutil.copy2(str(best_ckpt_path), str(final_path))
            tag = ('early-stop' if early_stop_reason is not None
                    else 'completion')
            print(f'[{tag}] promoted best.pt (iter={best_p3_iter}, '
                  f'det_return={best_p3_ema:.3f}) -> final.pt', flush=True)
        except Exception as _e:
            print(f'[best->final] promotion failed: {_e!r}',
                   flush=True)

    try:
        _save_training_diagnostics_plot(log_path, out_dir / 'training_diagnostics.png')
    except Exception as e:
        print(f'[train] training_diagnostics.png skipped: {e!r}', flush=True)

    # ---------------------------------------------------------------
    # End-of-training WM steady-state diagnostic.
    # Probes whether the trained world model can represent the
    # steady-state (not just transients) the actor's imagined
    # rollouts depend on for terminal-value bootstrapping.  Cheap on
    # GPU (~10s); falls back to CPU if GPU is busy.  Disable with
    # DREAMER_RUN_WM_DIAGNOSTIC=0.
    # ---------------------------------------------------------------
    if int(os.environ.get('DREAMER_RUN_WM_DIAGNOSTIC', '1') or 0) == 1:
        try:
            # Pre-free GPU before the diagnostic so its auto-device
            # picker doesn't fall back to CPU just because OUR training
            # tensors still occupy >50% of GPU memory.  Added 2026-05-23
            # after P43's diagnostic ran on CPU (5× slower) despite the
            # GPU being idle (util=0%) — the mem_frac heuristic counted
            # our own model+optimizer+buffer as "busy".
            try:
                import gc as _gc
                # Drop large training-only references; the diagnostic
                # loads its own model from final.pt.
                for _name in ('opt_wm', 'opt_actor', 'opt_critic',
                               'opt', 'buf', 'replay', 'model'):
                    if _name in locals():
                        try:
                            del locals()[_name]
                        except Exception:
                            pass
                _gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
            except Exception as _e:
                print(f'[wm-ss-diag] pre-free skipped: {_e!r}',
                       flush=True)
            from tools.wm_steady_state_diagnostic import (
                run_wm_steady_state_diagnostic)
            n_starts = int(os.environ.get('DREAMER_WM_DIAG_N_STARTS', '8'))
            # (c) value-equivalence: probe horizon adaptive to the training
            # horizon H (default max(200, 8·H)) so large-H sims aren't under-
            # probed and small-H sims still get the long structural drift
            # signal; the diagnostic separately reports convergence AT H.
            _H_train = int(getattr(cfg, 'horizon', 15))
            _diag_h_default = max(200, 8 * _H_train)
            horizon = int(os.environ.get('DREAMER_WM_DIAG_HORIZON',
                                          str(_diag_h_default)))
            # Force CUDA for inline diagnostics: the auto-picker reads
            # nvidia-smi util, which sees *our own* training process as
            # "GPU busy" and falls back to CPU — making the WM rollout
            # very slow.  A manual override still wins (set
            # DREAMER_WM_DIAG_DEVICE=cpu/cuda explicitly).
            if (torch.cuda.is_available() and
                    not os.environ.get('DREAMER_WM_DIAG_DEVICE')):
                os.environ['DREAMER_WM_DIAG_DEVICE'] = 'cuda'
            run_wm_steady_state_diagnostic(
                out_dir, ckpt_name='final.pt',
                n_starts=n_starts, horizon=horizon,
                output_dir=(out_dir / 'validation'))
        except Exception as e:
            print(f'[train] wm_steady_state_diagnostic skipped: {e!r}',
                   flush=True)

    return {
        'final_ckpt': str(final_path), 'iters': total_iters,
        'env_steps': total_env_steps,
        'final_ema_return': (float(ema_return)
                              if ema_return is not None else None),
        'final_return_window_mean': (float(np.mean(return_window))
                                       if return_window else None),
        'final_return_window_std': (float(np.std(return_window))
                                      if len(return_window) > 1 else None),
        'final_return_window_n': int(len(return_window)),
        'n_grad_skip': int(n_grad_skip),
        'early_stop_reason': early_stop_reason,
        # P49 RCA: ``best_p3_det_return`` tracks the deterministic-eval
        # window mean used for best.pt selection.  ``best_p3_ema_return``
        # alias retained for backward-compat with any downstream tools
        # that parse the older key name (both hold the same value).
        'best_p3_det_return': (float(best_p3_ema)
                                if best_p3_ema is not None else None),
        'best_p3_ema_return': (float(best_p3_ema)
                                if best_p3_ema is not None else None),
        'best_p3_iter': int(best_p3_iter) if best_p3_iter >= 0 else None,
        'best_ckpt': (str(best_ckpt_path)
                       if best_ckpt_path is not None else None),
        'mid_check_flags': list(mid_check_flags),
        # P52 RCA: record phase-gate decisions for post-hoc analysis.
        'phase_gate_decisions': list(phase_gate_decisions),
        'p1_ext_steps': int(p1_ext_steps),
        'p2_ext_steps': int(p2_ext_steps),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cfg_from_env() -> TrainConfig:
    cfg = TrainConfig()
    explicit: set = set()
    for name, attr, cast in [
        ('DREAMER_D_MODEL', 'd_model', int),
        ('DREAMER_N_LAYERS', 'n_layers', int),
        ('DREAMER_N_HEADS', 'n_heads', int),
        ('DREAMER_FF_MULT', 'ff_mult', int),
        ('DREAMER_N_REGISTER', 'n_register', int),
        ('DREAMER_Z_DIM', 'z_dim', int),
        ('DREAMER_TOK_HIDDEN', 'tok_hidden', int),
        ('DREAMER_HEAD_HIDDEN', 'head_hidden', int),
        ('DREAMER_K_MAX', 'k_max', int),
        ('DREAMER_LOOKBACK', 'lookback', int),
        ('DREAMER_HORIZON', 'horizon', int),
        ('DREAMER_SEQ_LEN', 'seq_len', int),
        ('DREAMER_BATCH_SIZE', 'batch_size', int),
        ('DREAMER_PHASE1_FRAC', 'phase1_frac', float),
        ('DREAMER_PHASE2_FRAC', 'phase2_frac', float),
        ('DREAMER_PHASE3_FRAC', 'phase3_frac', float),
        ('DREAMER_PMPO_BETA', 'pmpo_beta', float),
        ('DREAMER_PMPO_ALPHA', 'pmpo_alpha', float),
        ('DREAMER_BC_SCALE', 'bc_scale', float),
        ('DREAMER_MAE_PMAX', 'mae_p_max', float),
        ('DREAMER_MTP_LENGTH', 'mtp_length', int),
        ('DREAMER_POLICY_TYPE', 'policy_type', str),
        ('DREAMER_POLICY_INIT_LOG_STD', 'policy_init_log_std', float),
        ('DREAMER_POLICY_LOG_STD_MIN', 'policy_log_std_min', float),
        ('DREAMER_POLICY_LOG_STD_MAX', 'policy_log_std_max', float),
        ('DREAMER_PMPO_ENTROPY_COEF', 'pmpo_entropy_coef', float),
        ('DREAMER_ACTOR_LOSS', 'actor_loss_type', str),
        ('DREAMER_GRAD_CLIP', 'grad_clip', float),
        ('DREAMER_LR_ACTOR', 'lr_actor', float),
        ('DREAMER_LR_CRITIC', 'lr_critic', float),
        ('DREAMER_LR_WORLD', 'lr_world', float),
        ('DREAMER_GAE_LAMBDA', 'gae_lambda', float),
        ('DREAMER_BASELINE_SEED_EPS', 'baseline_seed_episodes', int),
        ('DREAMER_BASELINE_SEED_STD', 'baseline_seed_action_std', float),
        ('DREAMER_RANDOM_SEED_EPS', 'random_seed_episodes', int),
        ('DREAMER_P3_COLLECT_EVERY', 'phase3_collect_every_iters', int),
        ('DREAMER_P3_ONPOLICY_BUFFER_EPS', 'phase3_onpolicy_buffer_eps', int),
        ('DREAMER_P3_ONPOLICY_PREFILL_EPS', 'phase3_onpolicy_prefill_eps', int),
        ('DREAMER_BUFFER_CAP_STEPS', 'buffer_capacity_steps', int),
        ('DREAMER_ATTN_IMPL', 'attn_impl', str),
        ('DREAMER_COMPILE_MODE', 'compile_mode', str),
        ('AGENT_TOTAL_STEPS', 'total_steps', int),
        ('SIM_EPISODE_LENGTH', 'episode_length', int),
        ('SIM_SAMPLE_RATE', 'sample_rate', int),
        ('CONTROLLER_OUT_DIR', 'out_dir', str),
        # ----- Early-stop overrides -----
        ('DREAMER_EARLY_STOP', 'early_stop_enable',
            lambda v: bool(int(v))),
        ('DREAMER_ES_P3_PATIENCE', 'early_stop_p3_patience_iters', int),
        ('DREAMER_ES_P3_MIN_IMPROVEMENT',
            'early_stop_p3_min_improvement', float),
        ('DREAMER_ES_ENT_FRAC', 'early_stop_entropy_collapse_frac', float),
        ('DREAMER_ES_ENT_PATIENCE',
            'early_stop_entropy_collapse_patience_iters', int),
        ('DREAMER_ES_ENT_WINDOW',
            'early_stop_entropy_collapse_window_iters', int),
        ('DREAMER_ES_ENT_MIN_BELOW',
            'early_stop_entropy_collapse_min_frac_below', float),
        ('DREAMER_ES_CRITIC_FACTOR',
            'early_stop_critic_divergence_factor', float),
        ('DREAMER_ES_CRITIC_PATIENCE',
            'early_stop_critic_divergence_patience_iters', int),
        # 2026-05-23 (P41 RCA): bootstrap-cascade canary thresholds.
        ('DREAMER_ES_CASCADE_REWVAR',
            'early_stop_cascade_min_rew_var_frac', float),
        ('DREAMER_ES_CASCADE_GROWTH',
            'early_stop_cascade_min_return_scale_growth', float),
        ('DREAMER_ES_CASCADE_PATIENCE',
            'early_stop_cascade_patience_iters', int),
        ('DREAMER_ES_GRADSKIP_WINDOW',
            'early_stop_grad_skip_window_iters', int),
        ('DREAMER_ES_GRADSKIP_MAX', 'early_stop_grad_skip_max', int),
        ('DREAMER_ES_P1_MIN_SF_DROP',
            'early_stop_p1_min_sf_drop_frac', float),
        ('DREAMER_ES_P2_MAX_RMTP',
            'early_stop_p2_max_reward_mtp_loss', float),
        # 2026-05-26 (P52 RCA): phase-transition quality gates.
        ('DREAMER_P1_GATE_WM_EMA_MIN', 'p1_gate_wm_ema_min', float),
        ('DREAMER_P1_GATE_PLATEAU_FRAC', 'p1_gate_plateau_frac', float),
        ('DREAMER_P1_GATE_PLATEAU_PROBES', 'p1_gate_plateau_probes', int),
        ('DREAMER_P1_GATE_MAX_EXTENSION', 'p1_gate_max_extension', float),
        ('DREAMER_P2_GATE_REWARD_MTP_MAX', 'p2_gate_reward_mtp_max', float),
        ('DREAMER_P2_GATE_RECENT_ITERS', 'p2_gate_recent_iters', int),
        ('DREAMER_P2_GATE_MAX_EXTENSION', 'p2_gate_max_extension', float),
        # 2026-05-30 (P70): RSSM imagination steady-state fix.
        ('DREAMER_RSSM_IMAG_LATENT_MODE', 'rssm_imag_latent_mode',
            lambda v: str(v).strip().lower() in ('1', 'true', 'yes', 'on', 't', 'y')),
        # 2026-05-31 (P73): bounded training reward (cascade root-cause fix).
        ('DREAMER_BOUND_TRAINING_REWARD', 'bound_training_reward',
            lambda v: str(v).strip().lower() in ('1', 'true', 'yes', 'on', 't', 'y')),
        ('DREAMER_BOUND_TRAINING_REWARD_MAX', 'bound_training_reward_max', float),
        ('DREAMER_BOUND_TRAINING_REWARD_REF', 'bound_training_reward_ref', float),
        # 2026-05-31 (P74): advantage clipping (Cursor stabilizer #3).
        ('DREAMER_ADVANTAGE_CLIP', 'advantage_clip', float),
    ]:
        v = os.environ.get(name)
        if v is not None and v != '':
            setattr(cfg, attr, cast(v))
            explicit.add(attr)
    # Stash explicitly-set field names so the auto-tune apply loop can
    # skip them even when the env-injected value equals the dataclass
    # default (e.g. paper-faithful log_std_max=0.0).
    try:
        cfg._explicit_fields = explicit  # type: ignore[attr-defined]
    except Exception:
        pass
    return cfg


if __name__ == '__main__':
    cfg = _cfg_from_env()
    summary = train(cfg)
    print(json.dumps(summary, indent=2))
