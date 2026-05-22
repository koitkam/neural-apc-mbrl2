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
from utils.derived_observations import (
    DerivedFeatures,
    derived_observables_enabled,
    derived_observables_window,
)
from utils.agent_utils import (
    load_objective_weights, load_objective_bounds, load_full_objective_spec,
    action_to_control,
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
    phase1_frac: float = 0.4
    phase2_frac: float = 0.2
    phase3_frac: float = 0.4

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
    sf_scale: float = 1.0
    reward_scale_loss: float = 4.0   # P1+P2+P3 reward MTP weight (was 2.0;
    # bumped 2026-05-07 after run_p5 RCA: reward gate passed (r=0.62) but
    # critic gate failed (r=-0.25) due to reward-cliff bimodality.  After
    # switching the reward saturator from hard-clip to tanh (smooth
    # gradient everywhere), the head can finally fit the full violation
    # tail; doubling the loss weight again accelerates that convergence
    # so P3 critic learning sees stable targets sooner.
    # bumped 2026-05-06 after run_p2 RCA showed reward head pred_std=5.2 vs
    # real_std=44.9 — head learned ranking r=0.62 but not scale.  2× weight
    # on the symlog/2-hot reward loss drives the head toward fitting the
    # full distribution width rather than collapsing onto the mean.
    # Stays well within stable WM-loss balance (recon+sf already O(1)).
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
    # P2 BC bootstrap weight.  Default 0 because we have no offline expert
    # data — random-action episodes from P1 collection are uniform, so a
    # non-zero bc_scale clones uniform → uniform prior_policy → PMPO KL
    # term in P3 pins the policy near uniform → policy collapse.  Set
    # this >0 only when expert demonstrations populate the buffer.
    bc_scale: float = 0.0            # Phase-2 policy BC weight

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
    mtp_length: int = 32              # bumped from paper L=8 (p31 RCA)

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
    gamma: float = 0.997
    gae_lambda: float = 0.95
    target_critic_tau: float = 0.02
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
    # Reduce inner train steps in P3 so more iters happen per fixed
    # env-step budget — Optuna's pruner gets more samples and we get
    # finer-grained logs of actor / entropy progression.
    phase3_train_steps_per_iter: int = 25
    # P3 warmup before reporting EMA to pruner (avoid pruning trials on
    # the first few P3 returns which are dominated by the snapshot
    # actor that hasn't been updated by imagination yet).
    phase3_pruner_warmup_iters: int = 8

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
    # P3 plateau: stop when no new best ``ema_return`` for this many P3
    # iters AND we are past ``phase3_pruner_warmup_iters``.  The trainer
    # writes ``best.pt`` whenever a new best is reached and copies it to
    # ``final.pt`` on plateau-stop so validation auto-picks the best
    # state without needing runner / validate.py changes.
    early_stop_p3_patience_iters: int = 200
    # Minimum relative improvement (vs current best) that counts as
    # "new best" (avoids ratcheting on noise).
    early_stop_p3_min_improvement: float = 0.01
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
    # Legacy: kept for backward compat with env-var overrides.  Set to
    # the same value as ``window_iters`` so single-streak users still get
    # a sensible default; the sliding-window check usually trips first.
    early_stop_entropy_collapse_patience_iters: int = 30
    # Critic-divergence trip: critic_loss above ``factor`` ×
    # rolling-median (window 200 P3 iters) for ``patience`` consecutive
    # logs → stop.
    early_stop_critic_divergence_factor: float = 5.0
    early_stop_critic_divergence_patience_iters: int = 20
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
                    tau=float(getattr(self.cfg, 'tau', 0.0) or 0.0) or None,
                    sample_rate=float(getattr(self.cfg, 'sample_rate', 1.0)
                                       or 1.0),
                ),
            )
            self.aug_obs_dim += int(self._derived_features.feat_dim)
        self.obs_dim = self.state_dim + self.aug_obs_dim

        self._window: Optional[np.ndarray] = None
        self._t = 0
        self._prev_control = np.zeros(self.action_dim, dtype='float32')
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
        # Set by the trainer at phase transitions (P1/P2 default 0.3,
        # P3 default 0.5).  ``None`` = read from env var directly.
        self._disturbance_prob_override: Optional[float] = None
        # Force flag: when True, every reset() always builds the hidden
        # process (validation path).  False = Bernoulli toggle.
        self._hidden_disturbance_force: bool = False
        # Training progress in [0, 1] (env_steps / total_steps).  Used by
        # the hidden-OU amplitude curriculum at reset() time.  Updated
        # by the trainer at every episode boundary via ``set_training_progress``.
        self._training_progress: float = 0.0

        # ---- Raw-reward clipping (P37 onward, 2026-05-22) ---------------
        # The objective's quadratic violation tail can produce
        # ``raw_reward`` magnitudes 1000× above the operating-region
        # median (p36: raw_min=-185, raw_abs_p95=0.20).  That dynamic
        # range pushes the symlog-twohot reward predictor's mass into
        # ~14 of 255 bins, capping ``reward_mtp_loss`` at the
        # operating-region entropy floor regardless of training time.
        # Clipping the raw tail at ``DREAMER_REWARD_RAW_CLIP_MIN``
        # (default -30) lets the auto-scaler pack the operating-region
        # into ~60 bins without altering the agent's incentive
        # direction (catastrophe still strongly avoided).  Set
        # ``DREAMER_REWARD_RAW_CLIP_MIN=-1e9`` to disable.
        try:
            self._reward_clip_min: float = float(
                os.environ.get('DREAMER_REWARD_RAW_CLIP_MIN', '-30.0'))
        except Exception:
            self._reward_clip_min = -30.0
        try:
            self._reward_clip_max: float = float(
                os.environ.get('DREAMER_REWARD_RAW_CLIP_MAX', '1e18'))
        except Exception:
            self._reward_clip_max = 1e18

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
        raw = np.concatenate(parts, axis=0).astype('float32')
        if self._obs_norm_learn:
            self._update_obs_norm(raw)
        return self._normalize_obs(raw)

    def reset(self, *, exploration: bool = False) -> np.ndarray:
        state = self.sim.reset()
        if isinstance(state, tuple):
            state = state[0]
        state = np.asarray(state, dtype='float32').reshape(-1)
        self._t = 0
        self._prev_control = np.zeros(self.action_dim, dtype='float32')
        self._last_cv_violation_sum = 0.0
        self._last_mv_violation_sum = 0.0
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
        tau_dom = float(getattr(self.cfg, 'tau', 0.0) or 0.0)
        sample_rate = float(getattr(self.cfg, 'sample_rate', 1.0) or 1.0)
        self._hidden_disturbance = maybe_build_hidden_disturbance(
            rng=self.rng,
            sim=self.sim,
            tau_dom=tau_dom,
            sample_rate=sample_rate,
            prob=float(prob),
            force=bool(self._hidden_disturbance_force),
            progress=float(self._training_progress),
        )
        obs_vec = self._build_obs_vec(state)
        self._window = np.tile(obs_vec, (self.cfg.lookback, 1)).astype('float32')
        return self._window.copy()

    def step(self, action_norm: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        action_norm = np.asarray(action_norm, dtype='float32').reshape(self.action_dim)
        action_01 = 0.5 * (np.clip(action_norm, -1.0, 1.0) + 1.0)
        control = action_to_control(action_01, self.bounds, self.setpoint_mgr)
        self.setpoint_mgr.step(self._t)
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
        # Hidden (truly unmeasured) CV disturbance: advance OU and add to
        # CV state channels.  Only active on episodes where the per-episode
        # Bernoulli toggle (or force flag) fired in reset().
        if self._hidden_disturbance is not None:
            try:
                self._hidden_disturbance.step(next_state)
            except Exception as _hde:
                if not getattr(self, '_hidden_dist_err_logged', False):
                    import traceback
                    print(f'[env.step] hidden_disturbance error '
                          f'(further occurrences silenced): {_hde!r}', flush=True)
                    traceback.print_exc()
                    self._hidden_dist_err_logged = True
        comps = compute_objective_components(
            state=next_state, sim=self.sim,
            control=control, prev_control=self._prev_control,
            obj_w=self.obj_w, bounds=self.bounds,
            setpoint_manager=self.setpoint_mgr,
            objective_spec=self.obj_spec,
        )
        raw_reward = float(comps['reward'])
        # Apply raw clip BEFORE scaling so calibration (which percentile-
        # fits ``raw_reward``) and the agent both see the same clipped
        # distribution.  See ``self._reward_clip_min/max`` rationale.
        if (self._reward_clip_min > -1e17) or (self._reward_clip_max < 1e17):
            raw_reward = float(np.clip(raw_reward,
                                       self._reward_clip_min,
                                       self._reward_clip_max))
        reward = raw_reward * float(self.reward_scale)
        self._prev_control = np.asarray(control, dtype='float32')
        self._t += 1
        done = self._t >= self.cfg.episode_length
        obs_vec = self._build_obs_vec(next_state)
        self._window = np.concatenate([self._window[1:], obs_vec[None, :]], axis=0)
        self._last_cv_violation_sum += float(comps.get('cv_violation_penalty', 0.0))
        self._last_mv_violation_sum += float(comps.get('mv_violation_penalty', 0.0))
        info = {'reward_components': comps, 't': self._t,
                'raw_reward': raw_reward,
                'raw_state': np.asarray(next_state, dtype='float32').copy()}
        return self._window.copy(), reward, done, info


# ---------------------------------------------------------------------------
# Replay buffer — episode-major
# ---------------------------------------------------------------------------

class TrajectoryBuffer:
    def __init__(self, capacity_eps: int, episode_length: int,
                 lookback: int, obs_dim: int, action_dim: int):
        self.capacity_eps = int(capacity_eps)
        self.T = int(episode_length)
        self.L = int(lookback)
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.obs = np.zeros((capacity_eps, self.T, self.L, self.obs_dim),
                            dtype='float32')
        self.act = np.zeros((capacity_eps, self.T, self.action_dim),
                            dtype='float32')
        self.rew = np.zeros((capacity_eps, self.T), dtype='float32')
        self.cont = np.ones((capacity_eps, self.T), dtype='float32')
        self.filled = 0
        self.write = 0

    def add_episode(self, obs: np.ndarray, act: np.ndarray, rew: np.ndarray,
                    cont: np.ndarray) -> None:
        T = obs.shape[0]
        assert T == self.T, f"episode length mismatch: {T} vs {self.T}"
        i = self.write
        self.obs[i] = obs
        self.act[i] = act
        self.rew[i] = rew
        self.cont[i] = cont
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
        out_obs = np.zeros((batch_size, seq_len, self.L, self.obs_dim),
                           dtype='float32')
        out_act = np.zeros((batch_size, seq_len, self.action_dim),
                           dtype='float32')
        out_rew = np.zeros((batch_size, seq_len), dtype='float32')
        out_cont = np.zeros((batch_size, seq_len), dtype='float32')
        for b in range(batch_size):
            s = starts[b]
            out_obs[b] = self.obs[ep_idx[b], s:s + seq_len]
            out_act[b] = self.act[ep_idx[b], s:s + seq_len]
            out_rew[b] = self.rew[ep_idx[b], s:s + seq_len]
            out_cont[b] = self.cont[ep_idx[b], s:s + seq_len]
        return {'obs': out_obs, 'act': out_act, 'rew': out_rew, 'cont': out_cont}


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
    obs_buf = np.zeros((T, L, D), dtype='float32')
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

    for t in range(T):
        obs_buf[t] = obs_window
        if random_action:
            a_np = env.rng.uniform(-1.0, 1.0,
                                    size=(env.action_dim,)).astype('float32')
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
                              ) -> Dict[str, np.ndarray]:
    """Collect one episode driven by small-noise actions around mid-MV.

    Used to **seed** the replay buffer with non-catastrophic transitions
    before the actor has learned anything (P0 cold-start fix,
    2026-05-05).  Actions are drawn from ``N(0, action_std)`` clipped to
    ``[-1, 1]`` in the env's normalized action space (``0.0`` is
    mid-bound for an MV channel).  No model inference is needed, so this
    runs on CPU regardless of device.

    Returns the same dict shape as ``collect_episode`` so the result can
    be passed straight to ``buf.add_episode``.
    """
    obs_window = env.reset(exploration=True)
    T, L, D = cfg.episode_length, cfg.lookback, env.obs_dim
    obs_buf = np.zeros((T, L, D), dtype='float32')
    act_buf = np.zeros((T, env.action_dim), dtype='float32')
    rew_buf = np.zeros(T, dtype='float32')
    cont_buf = np.ones(T, dtype='float32')
    for t in range(T):
        obs_buf[t] = obs_window
        a_np = env.rng.normal(0.0, float(action_std),
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
    obs_buf = np.zeros((T, L, D), dtype='float32')
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
        obs_buf[t] = obs_window
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
    # active.
    env._schedule = []
    T, L, D = cfg.episode_length, cfg.lookback, env.obs_dim
    obs_buf = np.zeros((T, L, D), dtype='float32')
    act_buf = np.zeros((T, env.action_dim), dtype='float32')
    rew_buf = np.zeros(T, dtype='float32')
    cont_buf = np.ones(T, dtype='float32')
    a_const = np.full((env.action_dim,),
                       float(np.clip(action_level, -1.0, 1.0)),
                       dtype='float32')
    for t in range(T):
        obs_buf[t] = obs_window
        next_window, reward, done, _ = env.step(a_const)
        act_buf[t] = a_const
        rew_buf[t] = reward
        cont_buf[t] = 0.0 if done and t == T - 1 else 1.0
        obs_window = next_window
        if done:
            break
    return {'obs': obs_buf, 'act': act_buf, 'rew': rew_buf, 'cont': cont_buf}


# ---------------------------------------------------------------------------
# Phase 1 / 2 — World model loss (tokenizer recon + shortcut forcing)
# ---------------------------------------------------------------------------

def world_model_loss(model: DreamerV4, batch: Dict[str, torch.Tensor],
                      cfg: TrainConfig,
                      ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor,
                                  torch.Tensor]:
    """Eq. 5 + Eq. 7. Returns (losses, z_clean, agent_hid).

    ``z_clean``  : (B, T, z_dim)  — frozen-tokenizer outputs (no MAE).
    ``agent_hid``: (B, T, d_model) — agent register hidden state from a
                    *clean* dynamics pass (used by Phase 2 BC heads).
    """
    obs = batch['obs']                     # (B, T, L, D)
    act = batch['act']                     # (B, T, A)
    obs_cur = obs[:, :, -1, :]             # (B, T, D)

    # Tokenizer with MAE: recon the masked obs.
    z_mae, recon = model.tokenizer.forward_with_mae(obs_cur)
    recon_loss = model.tokenizer.recon_loss(obs_cur, recon)

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
    tau_max = (float(cfg.k_max) - 1.0) / float(cfg.k_max)
    tau_clean = torch.full((B, T), tau_max, device=device, dtype=z_clean.dtype)
    d_min = torch.full((B, T), 1.0 / cfg.k_max, device=device,
                        dtype=z_clean.dtype)
    out_clean = model.dynamics(z_clean, tau_clean, d_min, act)
    agent_hid = out_clean['agent_hid']

    losses: Dict[str, torch.Tensor] = {
        'recon_loss': recon_loss,
        'sf_loss': sf_loss,
        'wm_total': cfg.recon_scale * recon_loss + cfg.sf_scale * sf_loss,
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

    # Policy MTP (BC over L future actions)
    if L_mtp_eff < model.policy.mtp_length:
        # Pad target with zeros so we can use logits_mtp; mask out the pad in loss.
        pad_act = torch.zeros(B * T_ctx,
                               model.policy.mtp_length - L_mtp_eff, A,
                               device=fut_act.device, dtype=fut_act.dtype)
        fut_act_full = torch.cat([fut_act, pad_act], dim=1)
        bc_lp_full = model.policy.log_prob_of_mtp(feat, fut_act_full)
        bc_loss = -bc_lp_full[:, :L_mtp_eff].mean()
    else:
        bc_lp = model.policy.log_prob_of_mtp(feat, fut_act)        # (BT, L)
        bc_loss = -bc_lp.mean()

    # Reward MTP (twohot CE over L future rewards)
    rew_logits_all = model.reward.forward_mtp(feat)           # (BT, L, K)
    if L_mtp_eff < model.reward.mtp_length:
        rew_logits_all = rew_logits_all[:, :L_mtp_eff]
    rew_loss_per = model.reward.loss_mtp(rew_logits_all, fut_rew)  # (BT, L)
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


# ---------------------------------------------------------------------------
# Phase 3 — Imagination training (PMPO + TD-λ)
# ---------------------------------------------------------------------------

def imagination_step(model: DreamerV4, batch: Dict[str, torch.Tensor],
                      cfg: TrainConfig) -> Dict[str, torch.Tensor]:
    """Phase 3: roll out H imagined steps, compute PMPO + TD-λ.

    Transformer + tokenizer + reward head are kept frozen here (callers
    only pass ``parameters_actor()`` and ``parameters_critic()`` to the
    optimizer in Phase 3). Imagination uses the V4 K=4 shortcut sampler.

    **Action-conditioned reward read** (fix 2026-05-03): the original
    loop sampled ``a_t = π(agent_hid_t)`` and then read
    ``r_t = reward(agent_hid_t)``, where ``agent_hid_t`` was computed
    *before* ``a_t`` was placed into the action history.  This made the
    predicted reward bit-identical across all 21 candidate action bins
    (verified by sweep) — the actor had no signal on which action was
    good in a given state.  We now sample ``a_t`` from the post-action
    agent_hid of the previous step (which already saw ``a_{t-1}``),
    then re-run dynamics with the new ``(z_t, a_t)`` appended, and read
    ``r_t``, ``v_t``, ``v_target_t`` from *that* agent_hid — which
    therefore depends on the chosen action.
    """
    obs = batch['obs']                       # (B, T, L, D)
    act = batch['act']                       # (B, T, A)
    B, T = obs.shape[:2]
    H = int(cfg.horizon)
    device = obs.device

    # Encode the buffered context (frozen tokenizer).
    with torch.no_grad():
        obs_cur = obs[:, :, -1, :]
        z_ctx = model.tokenizer.encode(obs_cur)              # (B, T, z)
    a_ctx = act

    z_history = z_ctx
    a_history = a_ctx
    d_min_v = 1.0 / cfg.k_max

    # Initial agent_hid at the buffer's last position — already sees the
    # last buffered (z_{T-1}, a_{T-1}) thanks to within-step bidirectional
    # block-causal attention.  This is the "post-action" hidden state we
    # feed to the policy to sample the first imagined action.
    # Use τ_max = (k_max-1)/k_max (the cleanest TRAINED τ) instead of
    # 1.0 — τ=1.0 hits an untrained τ-embedding bin and agent_hid is
    # garbage.  Same fix as in world_model_loss.
    tau_max_v = (float(cfg.k_max) - 1.0) / float(cfg.k_max)
    with torch.no_grad():
        tau_clean = torch.full((B, T), tau_max_v, device=device,
                                 dtype=z_history.dtype)
        d_min_t = torch.full((B, T), d_min_v, device=device,
                              dtype=z_history.dtype)
        out0 = model.dynamics(z_history, tau_clean, d_min_t, a_history)
        agent_hid_post = out0['agent_hid'][:, -1]            # (B, D)

    imagined_rewards: List[torch.Tensor] = []
    imagined_logp: List[torch.Tensor] = []
    imagined_entropy: List[torch.Tensor] = []
    imagined_actions: List[torch.Tensor] = []
    # Canonical sample representation for log_prob recomputation in the
    # actor loss — pre-tanh ``u`` for the continuous head, bin index
    # for the discrete head.  See ContinuousPolicyHead.sample_with_raw
    # docstring for the bfloat16 precision rationale.
    imagined_raws: List[torch.Tensor] = []
    imagined_values: List[torch.Tensor] = []
    imagined_target_v: List[torch.Tensor] = []
    imagined_agent_hid: List[torch.Tensor] = []
    # Pre-rollout latent (the one the policy was actually conditioned
    # on when it produced ``raw_t``).  Required for the actor loss to
    # recompute log_prob on the SAME (μ, σ) that produced the sample;
    # using the post-rollout latent here is a silent bug because the
    # dynamics roll changes ``agent_hid_post`` between sample-time and
    # loss-time, so log_prob_of_raw(s_{h+1}, u_h) evaluates the Gaussian
    # at a completely different (μ, σ) and ``((u-μ)/σ)²`` blows up.
    imagined_agent_hid_pre: List[torch.Tensor] = []

    for h in range(H):
        # 1. Sample action conditioned on post-action agent_hid of the
        #    previous step (or the buffer's last step at h=0).
        agent_hid_for_policy = agent_hid_post
        action_t, logp_t, ent_t, raw_t = model.policy.sample_with_raw(
            agent_hid_for_policy)

        # 2. Imagine the next z under that action (K=4 shortcut sampler).
        #    Pass the REAL action history that produced z_history; without
        #    it imagine_next_z falls back to zeros and the dynamics is
        #    queried in a distribution it was never trained on.
        #    tau_ctx=None lets imagine_next_z auto-pick 1/k_max so the
        #    past τ lands at (k_max-1)/k_max (in-distribution).
        with torch.no_grad():
            z_next = model.imagine_next_z(z_history, action_t,
                                           k_steps=cfg.k_max,
                                           tau_ctx=None,
                                           action_history=a_history)      # (B, z)

        # 3. Slide histories: append (z_t, a_t).
        z_history = torch.cat([z_history[:, -(T - 1):],
                                 z_next.unsqueeze(1)], dim=1)
        a_history = torch.cat([a_history[:, -(T - 1):],
                                 action_t.unsqueeze(1)], dim=1)

        # 4. Re-run dynamics so agent_hid at the new last-position sees
        #    the freshly-sampled a_t.  This is the key change vs. before.
        #    τ = τ_max (in-grid) per the same fix as above.
        T_now = z_history.shape[1]
        with torch.no_grad():
            tau_clean = torch.full((B, T_now), tau_max_v, device=device,
                                     dtype=z_history.dtype)
            d_min_t = torch.full((B, T_now), d_min_v, device=device,
                                  dtype=z_history.dtype)
            out = model.dynamics(z_history, tau_clean, d_min_t, a_history)
            agent_hid_post = out['agent_hid'][:, -1]          # (B, D)

        # 5. Frozen reward + target-value heads — now action-aware.
        with torch.no_grad():
            r_logits = model.reward(agent_hid_post)
            reward_pred = model.reward.expectation(r_logits)
            tv_logits = model.target_value(agent_hid_post)
            target_v_pred = model.target_value.expectation(tv_logits)
        # 6. Current value head (with grad).
        value_logits = model.value(agent_hid_post)

        imagined_rewards.append(reward_pred)
        imagined_logp.append(logp_t)
        imagined_entropy.append(ent_t)
        imagined_actions.append(action_t)
        imagined_raws.append(raw_t)
        imagined_values.append(value_logits)
        imagined_target_v.append(target_v_pred)
        imagined_agent_hid.append(agent_hid_post)
        imagined_agent_hid_pre.append(agent_hid_for_policy)

    rewards = torch.stack(imagined_rewards, dim=1)            # (B, H)
    target_values = torch.stack(imagined_target_v, dim=1)     # (B, H)
    actions = torch.stack(imagined_actions, dim=1)            # (B, H, A)
    raws = torch.stack(imagined_raws, dim=1)                  # (B, H, ...)
    entropies = torch.stack(imagined_entropy, dim=1)          # (B, H)
    agent_hids = torch.stack(imagined_agent_hid, dim=1)       # (B, H, D)
    # Pre-rollout latents: the (B, H, D) tensor whose [b, h] entry is
    # exactly the latent that the policy was conditioned on when it
    # produced ``raws[b, h]``.  This is what the actor loss must use
    # to recompute log_prob — see imagined_agent_hid_pre comment.
    agent_hids_pre = torch.stack(imagined_agent_hid_pre, dim=1)  # (B, H, D)
    value_logits_seq = torch.stack(imagined_values, dim=1)    # (B, H, n_bins)

    # λ-returns (eq. 10).
    gamma = cfg.gamma
    lam = cfg.gae_lambda
    returns = torch.zeros_like(target_values)
    returns[:, -1] = target_values[:, -1]
    for t in reversed(range(H - 1)):
        bootstrap = (1.0 - lam) * target_values[:, t + 1] + lam * returns[:, t + 1]
        returns[:, t] = rewards[:, t] + gamma * bootstrap
    target_returns = returns.detach()

    # Critic loss (twohot CE).
    val_logits_flat = value_logits_seq.reshape(-1, value_logits_seq.shape[-1])
    val_target_flat = target_returns.reshape(-1)
    critic_loss = model.value.loss(val_logits_flat, val_target_flat).mean()

    # Advantage (current critic baseline) → PMPO (eq. 11).
    # Paper-faithful: A_t = R_t - V(s_t), then divide by EMA return scale
    # (model.update_return_scale).  Per-trajectory mean centering was
    # used previously as a workaround when the upstream tokenizer
    # collapse dominated within-trajectory advantage variance with
    # value-function noise.  With the May-2026 architecture fix the
    # within-trajectory advantage signal is real, so global centering
    # is correct again — it preserves the "this trajectory was overall
    # better than that one" signal that PMPO needs to upweight good
    # trajectories.
    with torch.no_grad():
        baseline = model.value.expectation(
            model.value(agent_hids.reshape(-1, agent_hids.shape[-1]))
        ).view(B, H)
        adv_raw = target_returns - baseline
        scale = model.update_return_scale(target_returns).clamp_min(1.0)

    feat_flat = agent_hids.reshape(-1, agent_hids.shape[-1])
    # Actor recomputes log_prob_of_raw on the SAME latent the policy
    # was conditioned on at sample time (= pre-rollout latent).  Mixing
    # in the post-rollout latent here is the silent bug that caused
    # the May-2026 actor_logp = -2400 explosion: under bf16 the dynamics
    # roll changes ``agent_hid_post`` enough that ``μ(s_{h+1})`` differs
    # from ``μ(s_h)`` by many σ-units, and the resulting Gaussian
    # density on ``u_h`` is astronomically small.
    feat_flat_for_policy = agent_hids_pre.reshape(-1, agent_hids_pre.shape[-1])
    actions_flat = actions.reshape(-1, actions.shape[-1])
    # ``raws`` may be either pre-tanh ``u`` (continuous, shape (B,H,A))
    # or bin index (discrete, shape (B,H,A) of int64).  Flatten while
    # preserving last-dim layout.
    raws_flat = raws.reshape(-1, raws.shape[-1]) if raws.dim() > 2 \
                  else raws.reshape(-1)
    adv_flat = (adv_raw / scale).reshape(-1).detach()

    actor_loss_type = str(getattr(cfg, 'actor_loss_type', 'reinforce')).lower()
    if actor_loss_type == 'pmpo':
        actor_loss, pmpo_diag = pmpo_loss(
            model.policy, model.prior_policy,
            feat_flat_for_policy, raws_flat, adv_flat,
            alpha=cfg.pmpo_alpha, beta=cfg.pmpo_beta,
            entropy_coef=float(getattr(cfg, 'pmpo_entropy_coef', 0.0)))
    else:
        # DreamerV3 REINFORCE — robust default, no per-sim retuning.
        # The V4 KL-to-prior is left off (kl_coef=0); the bounded
        # REINFORCE surrogate plus entropy bonus is sufficient anchor.
        actor_loss, pmpo_diag = reinforce_actor_loss(
            model.policy, model.prior_policy,
            feat_flat_for_policy, raws_flat, adv_flat,
            entropy_coef=float(getattr(cfg, 'pmpo_entropy_coef', 3e-4)),
            kl_coef=0.0)

    diag = {
        'actor_loss': actor_loss,
        'critic_loss': critic_loss,
        'imagined_return_mean': target_returns.mean().detach(),
        'imagined_reward_mean': rewards.mean().detach(),
        'imagined_reward_std': rewards.std().detach(),
        'entropy_mean': entropies.mean().detach(),
        'adv_std_mean': adv_raw.std(dim=1).mean().detach(),
        'adv_global_std': adv_raw.std().detach(),
        'return_scale': scale.detach().squeeze(),
    }
    diag.update(pmpo_diag)
    return diag


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
    )
    model = DreamerV4(model_cfg)
    # Optional torch.compile (set via TrainConfig.compile_mode or env var).
    cm = (cfg.compile_mode or '').strip()
    if not cm:
        env_cm = os.environ.get('DREAMER_COMPILE', '').strip()
        if env_cm in ('1', 'true', 'True'):
            cm = 'default'
        elif env_cm:
            cm = env_cm
    if cm:
        model.maybe_compile(mode=cm)
    return model


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
    return {
        'H': H,
        'r_floor': r_floor,
        'per_offset': candidates,
        'best_h': best_h,
        'summary': summary,
        'passes_full': best_h >= H,
    }


def _maybe_clip_horizon_to_wm_fidelity(model, env, device,
                                         cfg: 'TrainConfig') -> None:
    """Deprecated no-op.

    The runtime WM-fidelity horizon clip + P1-extension mechanism was
    removed 2026-05-20 alongside the short-budget knob cleanup.  With
    the 1M-step default budget the P1 schedule has enough time to
    train the WM at the paper-default H=15; runtime horizon shrinkage
    is no longer needed.  This stub is retained so out-of-tree callers
    do not break.
    """
    return


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
    out['policy_init_log_std'] = {
        'value': -2.0,
        'source': 'universal_default(σ≈0.135)',
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
    sigma_max_mult = float(os.environ.get(
        'DREAMER_SIGMA_MAX_OVER_SEED',
        os.environ.get('SIGMA_MAX_OVER_SEED', '1.0')))
    sigma_max_floor = float(os.environ.get('SIGMA_MAX_FLOOR', '0.10'))
    # Cap σ_max independently of the seed-σ cap so a wide seed-buffer
    # exploration band does not propagate into a wide policy clamp.
    # History: 0.20 → 0.30 on 2026-05-12 (p21 RCA: too tight for high-
    # disturbance plants).  Lowered back 0.30 → 0.20 on 2026-05-18
    # (p24 RCA: σ-saturation trap at 0.219 prevented critic learning).
    # Restored 0.20 → 0.30 on 2026-05-19 (p26 RCA: reward-head fix
    # removed the saturation-trap mechanism; σ-saturation is now
    # benign).
    sigma_max_cap = float(os.environ.get(
        'DREAMER_SIGMA_MAX_CAP',
        os.environ.get('SIGMA_MAX_CAP', '0.30')))
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
    sigma_min_ratio = max(2.0,
        float(os.environ.get('SIGMA_MIN_RATIO_OF_MAX', '2.5')))
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
    # ``p3_critic_warmup_iters`` and ``gae_lambda`` auto-tune blocks
    # were removed in the same cleanup.  With ``--steps`` defaulted to
    # 1M (paper minimum for control), the critic settles naturally and
    # paper-default ``gae_lambda=0.95`` is appropriate for any horizon.
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
    """
    raw_rewards: List[float] = []
    obs_trace: List[np.ndarray] = []
    env.reset(exploration=True)
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
    if any(v is not None for v in cv):
        ax.plot(steps, cv, label='cv_v', lw=1.0, color='C3')
    if any(v is not None for v in mv):
        ax.plot(steps, mv, label='mv_v', lw=1.0, color='C1')
    ax.set_ylabel('mean violation'); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax.set_title('Violations (per-iter mean)')

    for ax in axes[1, :]:
        ax.set_xlabel('env_steps')
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


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
                'DREAMER_REWARD_CAL_PCT', '50') or 50.0)
        except Exception:
            cal_target_pct = 50.0
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
        prev_ema = ckpt.get('best_ema_return')
        if prev_iter is not None:
            print(f'[init] resumed from iter={prev_iter} '
                   f'best_ema_return={prev_ema}', flush=True)

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
                            cfg.lookback, cfg.obs_dim, cfg.action_dim)

    out_dir = Path(cfg.out_dir or '.')
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / 'train_log.jsonl'
    log_f = open(log_path, 'a')

    # Phase budgets.
    p1 = int(cfg.phase1_frac * cfg.total_steps)
    p2 = int(cfg.phase2_frac * cfg.total_steps)
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

    # ----- Early-stop bookkeeping -----
    es_enable = bool(getattr(cfg, 'early_stop_enable', True))
    best_p3_ema: Optional[float] = None
    best_p3_iter: int = -1
    best_ckpt_path: Optional[Path] = None
    iters_since_best: int = 0
    ent_collapse_streak: int = 0
    ent_window: List[float] = []
    critic_div_streak: int = 0
    critic_loss_window: 'deque[float]' = deque(maxlen=200)
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
    wm_fidelity_warmup_iters = int(
        os.environ.get('DREAMER_WM_FIDELITY_WARMUP_ITERS', '40'))
    wm_fidelity_patience_iters = int(
        os.environ.get('DREAMER_WM_FIDELITY_PATIENCE_ITERS', '50'))
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
        if env_steps < p1:
            return 1
        if env_steps < p1 + p2:
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
    if n_baseline_seed > 0:
        for _ in range(n_baseline_seed):
            ep = collect_baseline_episode(env, cfg,
                                            action_std=baseline_seed_std)
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

    # Constant-action seed (run_p31 RCA, 2026-05-21): the PRBS sweep
    # only ever holds an action for ~80–150 agent steps, so the WM has
    # no long-horizon constant-action data and extrapolates badly when
    # the trained controller later sits near a setpoint.  These full-
    # episode constant-action seeds give the WM explicit steady-state
    # coverage at a stratified spread of operating points within
    # ``[-constant_action_seed_op_band, +constant_action_seed_op_band]``.
    n_const_seed = int(getattr(cfg, 'constant_action_seed_episodes', 0))
    const_op_band = float(getattr(cfg, 'constant_action_seed_op_band', 0.6))
    if n_const_seed > 0:
        # Stratified levels: evenly-spaced over [-op_band, +op_band]
        # with a small jitter so re-running the workflow does not hit
        # exactly the same operating points.
        levels = np.linspace(-const_op_band, const_op_band, n_const_seed,
                              dtype='float32')
        jitter = env.rng.uniform(-0.05, 0.05, size=levels.shape).astype('float32')
        levels = np.clip(levels + jitter * const_op_band, -1.0, 1.0)
        for lvl in levels:
            ep = collect_constant_action_episode(env, cfg,
                                                  action_level=float(lvl))
            buf.add_episode(ep['obs'], ep['act'], ep['rew'], ep['cont'])
            total_env_steps += cfg.episode_length

    # Cached optimizer set per phase.
    # Initialize hidden-disturbance probability for the starting phase
    # (hidden CV disturbance is the default unmeasured-disturbance model).
    env._disturbance_prob_override = get_phase_disturbance_prob(phase=1)
    while total_env_steps < cfg.total_steps:
        # Push training progress into the env so the hidden-OU amplitude
        # curriculum (DREAMER_HIDDEN_OU_AMP_RAMP) sees the latest value
        # at every episode reset.  No-op when curriculum env var unset.
        env.set_training_progress(total_env_steps / max(1, int(cfg.total_steps)))
        new_phase = _phase_for(total_env_steps)
        if new_phase != current_phase:
            print(f'[phase] transition {current_phase} -> {new_phase} '
                  f'at env_steps={total_env_steps}', flush=True)
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
                    last_sf = float(wm_losses.get('sf_loss', p1_initial_sf))
                    print(f"[p1→p2] sf_loss {p1_initial_sf:.4f} → "
                          f"{last_sf:.4f}", flush=True)
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
            current_phase = new_phase
            if current_phase == 3:
                # Snapshot the prior policy (PMPO behavioural prior, eq. 11).
                model.snapshot_prior_policy()
            # Refresh hidden-disturbance per-episode probability for the
            # new phase (default: 0.3 in P1/P2, 0.5 in P3).
            env._disturbance_prob_override = get_phase_disturbance_prob(
                phase=int(current_phase))

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
            if (total_iters % collect_every) == 0:
                ep = collect_episode(env, model, device, cfg,
                                       random_action=False,
                                       deterministic=False)
                buf.add_episode(ep['obs'], ep['act'], ep['rew'], ep['cont'])
                total_env_steps += cfg.episode_length
                ret = float(ep['rew'].sum())
                ema_return = (ret if ema_return is None
                                else 0.95 * ema_return + 0.05 * ret)
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
        t_collect_acc += time.time() - _t

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
            batch_np = buf.sample(cfg.batch_size, cfg.seq_len, rng)
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
                        ag_losses = agent_finetune_loss(model, batch,
                                                          agent_hid, cfg)
                # Phase 2: also update reward + policy via MTP (eq. 9).
                if current_phase == 2:
                    with torch.amp.autocast(device_type=device.type,
                                              dtype=torch.bfloat16,
                                              enabled=(device.type == 'cuda')):
                        ag_losses = agent_finetune_loss(model, batch,
                                                          agent_hid, cfg)
                    total_loss = wm_losses['wm_total'] + ag_losses['agent_total']
                elif current_phase == 1:
                    # P1: WM losses + reward-head MTP only (no BC).
                    # Use the *non-detached* ``reward_mtp_total`` key —
                    # using ``reward_mtp_loss`` (which is .detach()'d for
                    # diagnostics) silently zeros the reward-head gradient
                    # in P1, leaving it untrained until P2 starts.
                    total_loss = (wm_losses['wm_total']
                                   + cfg.reward_scale_loss
                                     * ag_losses['reward_mtp_total'])
                else:
                    total_loss = wm_losses['wm_total']

                opt_world.zero_grad(set_to_none=True)
                if current_phase == 2:
                    opt_actor.zero_grad(set_to_none=True)
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
                # ---- (a) WM + reward-head update -----------------
                with torch.amp.autocast(device_type=device.type,
                                          dtype=torch.bfloat16,
                                          enabled=(device.type == 'cuda')):
                    wm_losses, _, agent_hid = world_model_loss(model, batch,
                                                                  cfg)
                    ag_losses = agent_finetune_loss(model, batch,
                                                      agent_hid, cfg)
                    # Drop BC term in P3 (the actor is now driven by
                    # imagination/PMPO; BC against random-action data
                    # would just pull it back toward uniform).  Keep
                    # reward MTP because that head is what the actor's
                    # value target depends on.
                    # Use non-detached ``reward_mtp_total`` (see the
                    # P1 comment above for the detach-bug rationale).
                    p3_total_world = (wm_losses['wm_total']
                                       + cfg.reward_scale_loss
                                         * ag_losses['reward_mtp_total'])
                opt_world.zero_grad(set_to_none=True)
                p3_total_world.backward()
                wm_grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters_world(), cfg.grad_clip)
                if torch.isfinite(wm_grad_norm):
                    opt_world.step()
                else:
                    n_grad_skip += 1

                # ---- (b) Actor + Critic via imagination ----------
                with torch.amp.autocast(device_type=device.type,
                                          dtype=torch.bfloat16,
                                          enabled=(device.type == 'cuda')):
                    ac_losses = imagination_step(model, batch, cfg)
                opt_actor.zero_grad(set_to_none=True)
                opt_critic.zero_grad(set_to_none=True)
                (ac_losses['actor_loss']
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
                    opt_actor.step()
                    opt_critic.step()
                model.update_target(cfg.target_critic_tau)
                t_ac_acc += time.time() - _t

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
            }
            for k, v in {**wm_losses, **ag_losses, **ac_losses}.items():
                row[k] = float(v.detach().item() if torch.is_tensor(v) else v)
            log_f.write(json.dumps(row) + '\n')
            log_f.flush()
            rwm = row.get('return_window_mean')
            rwm_str = f"{rwm:+.2f}" if rwm is not None else 'n/a'
            ema_str = (f"{row['ema_return']:.2f}"
                        if row['ema_return'] is not None else 'n/a')
            print(f"[{row['timestamp']}] P{current_phase} iter {total_iters:4d} "
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
                              f"best_h={_pbe['best_h']}/{_pbe['H']}",
                              flush=True)
                        # ---- Fidelity-based best-ckpt + early stop ----
                        # Score = sum of positive Pearson r across
                        # probed horizons + small bonus for depth.
                        # Robust to single-horizon noise.
                        _per = _pbe.get('per_offset') or []
                        _r_vals = [float(r) for (_, r) in _per]
                        _score = (sum(max(0.0, r) for r in _r_vals)
                                   + 0.05 * float(_pbe.get('best_h', 0))
                                            / max(1, int(_pbe.get('H', 1))))
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
                                },
                            }, wm_best_ckpt_path)
                            print(f"[wm-best] new best fidelity score "
                                  f"{_score:.3f} at iter {total_iters} "
                                  f"-> saved {wm_best_ckpt_path.name}",
                                  flush=True)
                        elif (es_enable
                              and current_phase in (1, 2)
                              and wm_best_iter > 0
                              and total_iters >= wm_fidelity_warmup_iters
                              and (total_iters - wm_best_iter)
                                    >= wm_fidelity_patience_iters):
                            early_stop_reason = (
                                f'wm_fidelity_degradation: no improvement '
                                f'over best={wm_best_score:.3f} '
                                f'(iter {wm_best_iter}) for '
                                f'{total_iters - wm_best_iter} iters '
                                f'(patience={wm_fidelity_patience_iters})')
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
                        if len(ent_window) >= win_n:
                            n_below = sum(1 for e in ent_window if e < thr)
                            min_below = (float(getattr(cfg,
                                    'early_stop_entropy_collapse_min_frac_below',
                                    0.70))
                                          * win_n)
                            if n_below >= min_below:
                                early_stop_reason = (
                                    f'entropy_collapse_window: '
                                    f'{n_below}/{win_n} iters below '
                                    f'thr={thr:.3f} '
                                    f'(latest={ent:.3f})')
                        # Legacy consecutive-streak detector (kept as a
                        # fallback for very long sustained collapse).
                        if ent < thr:
                            ent_collapse_streak += 1
                        else:
                            ent_collapse_streak = 0
                        if (early_stop_reason is None
                                and ent_collapse_streak >= int(getattr(cfg,
                                'early_stop_entropy_collapse_patience_iters',
                                30))):
                            early_stop_reason = (
                                f'entropy_collapse: ent={ent:.3f} < '
                                f'{thr:.3f} for {ent_collapse_streak} iters')

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

                # --- Soft fail: P3 plateau on best ema_return ---
                if (early_stop_reason is None and current_phase == 3
                        and ema_return is not None and np.isfinite(ema_return)):
                    min_imp = float(getattr(cfg,
                            'early_stop_p3_min_improvement', 0.01))
                    if best_p3_ema is None:
                        best_p3_ema = float(ema_return)
                        best_p3_iter = total_iters
                        iters_since_best = 0
                    else:
                        # Relative improvement against |best| with floor 1.0
                        # so trials hovering near 0 don't ratchet on noise.
                        denom = max(1.0, abs(best_p3_ema))
                        improvement = (ema_return - best_p3_ema) / denom
                        if improvement >= min_imp:
                            best_p3_ema = float(ema_return)
                            best_p3_iter = total_iters
                            iters_since_best = 0
                            # Persist best ckpt for plateau-stop recovery.
                            try:
                                best_ckpt_path = out_dir / 'best.pt'
                                torch.save({'model': model.state_dict(),
                                            'cfg': asdict(cfg),
                                            'obs_norm': env.get_obs_norm_stats(),
                                            'best_ema_return': best_p3_ema,
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
                            f'over best={best_p3_ema:.3f} for '
                            f'{iters_since_best} iters')

                if early_stop_reason is not None:
                    print(f'[early-stop] tripped: {early_stop_reason}',
                           flush=True)
                    break

        if on_iter_end is not None:
            # Gate the BO pruner to *post-warmup P3 only*.  Reporting
            # P1/P2 random-action EMAs (or early P3 EMAs dominated by
            # the snapshot actor) makes the pruner kill trials on
            # pre-learning noise, not on actual learning quality.
            warmup = max(0, int(getattr(cfg, 'phase3_pruner_warmup_iters', 0)))
            if current_phase == 3 and p3_iters > warmup:
                try:
                    stop = bool(on_iter_end(int(total_iters),
                                              int(total_env_steps),
                                              float(ema_return)
                                              if ema_return is not None else 0.0))
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
                  f'ema={best_p3_ema:.3f}) -> final.pt', flush=True)
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
            from tools.wm_steady_state_diagnostic import (
                run_wm_steady_state_diagnostic)
            n_starts = int(os.environ.get('DREAMER_WM_DIAG_N_STARTS', '8'))
            horizon = int(os.environ.get('DREAMER_WM_DIAG_HORIZON', '200'))
            run_wm_steady_state_diagnostic(
                out_dir, ckpt_name='final.pt',
                n_starts=n_starts, horizon=horizon)
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
        'best_p3_ema_return': (float(best_p3_ema)
                                if best_p3_ema is not None else None),
        'best_p3_iter': int(best_p3_iter) if best_p3_iter >= 0 else None,
        'best_ckpt': (str(best_ckpt_path)
                       if best_ckpt_path is not None else None),
        'mid_check_flags': list(mid_check_flags),
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
        ('DREAMER_ES_GRADSKIP_WINDOW',
            'early_stop_grad_skip_window_iters', int),
        ('DREAMER_ES_GRADSKIP_MAX', 'early_stop_grad_skip_max', int),
        ('DREAMER_ES_P1_MIN_SF_DROP',
            'early_stop_p1_min_sf_drop_frac', float),
        ('DREAMER_ES_P2_MAX_RMTP',
            'early_stop_p2_max_reward_mtp_loss', float),
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
