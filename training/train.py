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
    # Override per-simulator only if a plant genuinely needs broader
    # exploration; the V3 prescription works generically across 150+
    # tasks and is the right starting point for any adaptive APC.
    policy_log_std_min: float = -2.3
    policy_log_std_max: float = 0.0
    # PMPO entropy bonus (DreamerV3 § actor loss, η = 3e-4).  Acts as a
    # soft σ-floor for the continuous actor / uniform-prior pull for the
    # discrete actor; essential for stability when the advantage signal
    # is heavy-tailed (process-control violation penalties).  Set to 0
    # to disable.
    pmpo_entropy_coef: float = 3e-4
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
    lr_critic: float = 3e-5
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
    reward_scale_loss: float = 2.0   # P1+P2+P3 reward MTP weight (was 1.0;
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
    # P2 BC bootstrap weight.  Default 0 because we have no offline expert
    # data — random-action episodes from P1 collection are uniform, so a
    # non-zero bc_scale clones uniform → uniform prior_policy → PMPO KL
    # term in P3 pins the policy near uniform → policy collapse.  Set
    # this >0 only when expert demonstrations populate the buffer.
    bc_scale: float = 0.0            # Phase-2 policy BC weight

    # ----- MTP -----
    mtp_length: int = 8              # paper L=8 (Phase-2 multi-token prediction)

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
    buffer_capacity_steps: int = 200_000

    # ----- Phase-3 evaluation cadence -----
    phase3_eval_every_iters: int = 5
    # Collect a fresh on-policy episode every K P3 iters (paper
    # Algorithm 1 line 22 calls for every iter, but every-iter
    # collection on a CPU plant simulator burns ~50% of wall-clock
    # without changing the on-policy distribution meaningfully on the
    # short BO horizons we use; K=4 keeps the actor data within ≤4
    # gradient steps of the policy snapshot used to collect it).
    phase3_collect_every_iters: int = 4
    # Reduce inner train steps in P3 so more iters happen per fixed
    # env-step budget — Optuna's pruner gets more samples and we get
    # finer-grained logs of actor / entropy progression.
    phase3_train_steps_per_iter: int = 25
    # P3 warmup before reporting EMA to pruner (avoid pruning trials on
    # the first few P3 returns which are dominated by the snapshot
    # actor that hasn't been updated by imagination yet).
    phase3_pruner_warmup_iters: int = 8
    # Critic warm-up: freeze the actor for the first N P3 iters so the
    # value head sees real imagined returns before REINFORCE reacts to
    # its baseline.  Prevents the entropy-saturation trap where a
    # freshly-initialised critic produces noisy advantages → REINFORCE
    # pushes σ to the clamp ceiling within 1–2 batches (validate-iter80
    # RCA, 2026-05-06).  V3-aligned in spirit (§3.3 EMA-target).  Set
    # to 0 to disable.
    #
    # Bumped 8 → 16 (2026-05-06): run_p2 RCA showed critic_loss only
    # dropped 5.2 → 4.0 in 8 iters (= 24% drop); actor was unfrozen
    # while critic was still 4× its eventual saturation level (~1.5).
    # 16 iters lets critic reach ~2.5 (~50% of P3 saturation) before
    # actor reacts, which materially reduces the over-optimism that
    # produced pmpo_pos_frac = 6% on run_p2.
    p3_critic_warmup_iters: int = 16

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
        self.obs_dim = self.state_dim + self.aug_obs_dim

        self._window: Optional[np.ndarray] = None
        self._t = 0
        self._prev_control = np.zeros(self.action_dim, dtype='float32')
        self._schedule: List[Dict] = []
        self.reward_scale: float = 1.0
        self._last_cv_violation_sum: float = 0.0
        self._last_mv_violation_sum: float = 0.0

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
        raw = np.concatenate([np.asarray(state, dtype='float32').reshape(-1),
                                aug], axis=0).astype('float32')
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
        self.setpoint_mgr.reset(episode_length=self.cfg.episode_length,
                                curriculum_fraction=1.0)
        intensity = 1.0 if not exploration else 1.2
        self._schedule = build_training_disturbance_schedule(
            episode_length=self.cfg.episode_length,
            rng=self.rng,
            intensity=intensity,
            sim=self.sim,
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
        comps = compute_objective_components(
            state=next_state, sim=self.sim,
            control=control, prev_control=self._prev_control,
            obj_w=self.obj_w, bounds=self.bounds,
            setpoint_manager=self.setpoint_mgr,
            objective_spec=self.obj_spec,
        )
        raw_reward = float(comps['reward'])
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
    tau_ctx_val = 1.0 - cfg.tau_ctx

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


# ---------------------------------------------------------------------------
# Phase 1 / 2 — World model loss (tokenizer recon + shortcut forcing)
# ---------------------------------------------------------------------------

def world_model_loss(model: DreamerV4, batch: Dict[str, torch.Tensor],
                      cfg: TrainConfig
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

    # Compute agent_hid from a clean dynamics pass (τ=1, d=d_min). Used by
    # the Phase 2 BC + reward MTP heads.
    B, T = z_clean.shape[:2]
    device = z_clean.device
    tau_clean = torch.full((B, T), 1.0, device=device, dtype=z_clean.dtype)
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
        z_var = z_clean.float().var(dim=(0, 1)).mean()
        losses['encoder_var_ratio'] = (z_var / obs_var).detach()
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
    with torch.no_grad():
        tau_clean = torch.full((B, T), 1.0, device=device,
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
        with torch.no_grad():
            z_next = model.imagine_next_z(z_history, action_t,
                                           k_steps=cfg.k_max,
                                           tau_ctx=cfg.tau_ctx)      # (B, z)

        # 3. Slide histories: append (z_t, a_t).
        z_history = torch.cat([z_history[:, -(T - 1):],
                                 z_next.unsqueeze(1)], dim=1)
        a_history = torch.cat([a_history[:, -(T - 1):],
                                 action_t.unsqueeze(1)], dim=1)

        # 4. Re-run dynamics so agent_hid at the new last-position sees
        #    the freshly-sampled a_t.  This is the key change vs. before.
        T_now = z_history.shape[1]
        with torch.no_grad():
            tau_clean = torch.full((B, T_now), 1.0, device=device,
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
        target_frac = 0.10
        cv_w = float(np.mean(cv_widths))
        sigma = float(np.clip(target_frac * cv_w / mv_auth, 0.01, 0.10))
        sigma_source = (f'mv_authority(target_cv_frac={target_frac:.2f}, '
                          f'cv_w={cv_w:.3f}, mv_auth={mv_auth:.3f})')
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
    # Floor at σ_max = 0.50 so we always preserve ≥ 5× headroom over
    # the V3 paper σ_min = 0.1.  Cap at 1.0 (V3 paper upper).  For
    # test_sim (σ_seed=0.10): σ_max=0.50 ⇒ log_std_max ≈ -0.69.
    #
    # Tuned 2026-05-06: previous floor of 0.30 left only 0.219 nats of
    # entropy headroom (= max H of σ=0.30 policy), which collided with
    # the entropy-collapse early-stop threshold (0.284 nats).  σ_max
    # = 0.50 gives H_max = 0.708 nats, well above the trip floor and
    # plenty of room for REINFORCE to explore action-dependent μ
    # without saturating against the ceiling on the first noisy
    # negative-advantage batch.
    target_sigma_max = float(np.clip(2.0 * sigma_seed, 0.50, 1.0))
    log_std_max_val = float(np.log(target_sigma_max))
    out['policy_log_std_max'] = {
        'value': log_std_max_val,
        'source': f'log(2*baseline_seed_action_std)='
                  f'log(2*{sigma_seed:.3f})={log_std_max_val:+.3f}',
    }
    return out


# Dataclass defaults captured for sentinel detection in ``train``.  When
# a cfg field still equals its dataclass default after env-var injection,
# auto-tune is allowed to overwrite it; user/env overrides survive.
_AUTO_TUNE_FIELD_DEFAULTS: Dict[str, object] = {
    'baseline_seed_action_std': TrainConfig().baseline_seed_action_std,
    'baseline_seed_episodes':   TrainConfig().baseline_seed_episodes,
    'random_seed_episodes':     TrainConfig().random_seed_episodes,
    'policy_init_log_std':      TrainConfig().policy_init_log_std,
    'policy_log_std_max':       TrainConfig().policy_log_std_max,
}


def calibrate_reward_scale(env: 'APCEnv', rng: np.random.Generator,
                            n_steps: int = 3000,
                            target_std: float = 1.0,
                            min_scale: float = 1e-4,
                            max_scale: float = 1000.0,
                            mode: str = 'baseline',
                            baseline_action_std: float = 0.05,
                            target_mode: str = 'percentile',
                            target_percentile: float = 95.0,
                            target_percentile_value: float = 1.0,
                            ) -> Dict[str, float]:
    """Empirically choose a per-step reward scale to match V4's twohot range.

    ``mode='baseline'`` (default, P0 fix): drive the env with small-noise
    actions around mid-MV (``a ~ N(0, baseline_action_std)`` clipped to
    ``[-1, 1]``).  This produces the *operating-region* reward distribution
    that the agent will actually see once it has learned to stay near
    safe set-points, instead of the violation-saturated distribution
    produced by uniform-random actions.  Avoids baking the ``raw_min ~
    -250`` cliff into ``reward_scale``.

    ``mode='random'``: legacy behaviour (uniform on ``[-1, 1]``); kept
    for back-compat / debugging.

    ``target_mode='percentile'`` (default, root-cause fix 2026-05-07):
    pick scale so that the ``target_percentile``-th percentile of
    ``|raw_reward|`` maps to ``target_percentile_value`` (default p95
    → 1.0).  Robust to bimodal/long-tailed APC reward distributions
    where ``std``-based calibration is dominated by the violation cliff
    and yields a degenerate scale.  After scaling, the 'normal'
    operating mass lands in ``[-1, +1]`` and the cliff lands at
    ``raw_min/p95`` which symlog spreads safely inside the
    ``[-20, +20]`` twohot support.

    ``target_mode='std'``: legacy behaviour, ``scale = target_std/std``.

    The historical ``min_scale=1.0`` clamp was a transposition bug that
    *prevented scaling rewards down* — it has been fixed to ``1e-4``
    so the calibrator can actually move into the V3/V4 design range
    (rewards O(1) before symlog).
    """
    if env.reward_scale != 1.0:
        env.reward_scale = 1.0
    raw_rewards: List[float] = []
    obs_trace: List[np.ndarray] = []
    env.reset(exploration=True)
    mode = str(mode).lower()
    target_mode = str(target_mode).lower()
    for _ in range(int(n_steps)):
        if mode == 'random':
            a = rng.uniform(-1.0, 1.0,
                              size=(env.action_dim,)).astype('float32')
        else:  # 'baseline'
            a = rng.normal(0.0, float(baseline_action_std),
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
    arr = np.asarray(raw_rewards, dtype='float64')
    std = float(arr.std())
    mean = float(arr.mean())
    abs_arr = np.abs(arr)
    pct_q = float(np.clip(target_percentile, 50.0, 99.9))
    p95_abs = float(np.percentile(abs_arr, pct_q)) if abs_arr.size else 0.0
    if target_mode == 'std':
        if std < 1e-8:
            scale = 1.0
        else:
            scale = float(target_std / std)
    else:  # 'percentile'
        if p95_abs < 1e-8:
            # Degenerate: fall back to std-based, then to identity.
            scale = float(target_std / std) if std >= 1e-8 else 1.0
        else:
            scale = float(target_percentile_value / p95_abs)
    scale_unclamped = scale
    scale = float(np.clip(scale, min_scale, max_scale))
    env.reward_scale = scale
    # Saturation diagnostic: the V4 twohot support is symlog([-20,+20]).
    # If even a single per-step scaled reward exceeds symlog's mid-band,
    # the head will struggle.  symlog(x)≈18 when |x|≈6.6e7; symlog(x)≈10
    # when |x|≈2.2e4.  Flag if scaled-cliff symlog magnitude > 5
    # (already encroaching on the head's high-curvature region).
    raw_min = float(arr.min())
    raw_max = float(arr.max())
    scaled_min = raw_min * scale
    scaled_max = raw_max * scale
    def _symlog(x: float) -> float:
        return float(np.sign(x) * np.log1p(abs(x)))
    sym_min = _symlog(scaled_min)
    sym_max = _symlog(scaled_max)
    sym_mag = max(abs(sym_min), abs(sym_max))
    twohot_warn = sym_mag > 5.0
    twohot_critical = sym_mag > 15.0
    return {
        'reward_scale': scale, 'reward_scale_unclamped': scale_unclamped,
        'raw_std': std, 'raw_mean': mean,
        'raw_min': raw_min, 'raw_max': raw_max,
        'raw_abs_p95': p95_abs,
        'target_std': float(target_std),
        'target_mode': target_mode,
        'target_percentile': pct_q,
        'target_percentile_value': float(target_percentile_value),
        'scaled_min': scaled_min, 'scaled_max': scaled_max,
        'scaled_symlog_min': sym_min, 'scaled_symlog_max': sym_max,
        'scaled_symlog_mag': sym_mag,
        'twohot_support_warn': bool(twohot_warn),
        'twohot_support_critical': bool(twohot_critical),
        'min_scale': float(min_scale), 'max_scale': float(max_scale),
        'n_steps': int(n_steps),
        'mode': mode, 'baseline_action_std': float(baseline_action_std),
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
    for k, c in [('recon_loss', 'C0'), ('sf_loss', 'C1'),
                  ('sf_loss_flow', 'C2'), ('sf_loss_boot', 'C3')]:
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
    for field, info in auto.items():
        cur = getattr(cfg, field)
        default = _AUTO_TUNE_FIELD_DEFAULTS.get(field)
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
                'DREAMER_REWARD_CAL_PCT_VAL', '1.0') or 1.0)
        except Exception:
            cal_target_pct_value = 1.0
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
                  f"{cal['scaled_symlog_mag']:.2f} > 5 — twohot head is "
                  f"in the high-curvature region; consider reducing "
                  f"DREAMER_REWARD_CAL_PCT_VAL.", flush=True)
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
            arr = np.asarray(cal_obs, dtype='float64')   # (T, obs_dim)
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

    # Cached optimizer set per phase.
    while total_env_steps < cfg.total_steps:
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
                # End of P1: did sf_loss drop?
                min_drop = float(getattr(cfg, 'early_stop_p1_min_sf_drop_frac', 0.0))
                if (min_drop > 0 and p1_initial_sf is not None):
                    last_sf = float(wm_losses.get('sf_loss', p1_initial_sf))
                    drop = (p1_initial_sf - last_sf) / max(1e-8, abs(p1_initial_sf))
                    if drop < min_drop:
                        msg = (f'P1: sf_loss drop {drop*100:.1f}% < '
                                f'{min_drop*100:.1f}% (initial={p1_initial_sf:.3f} '
                                f'final={last_sf:.3f}) — WM may not have learned')
                        print(f'[early-stop-flag] {msg}', flush=True)
                        mid_check_flags.append(msg)
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
                    wm_losses, _, agent_hid = world_model_loss(model, batch, cfg)
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
                    total_loss = (wm_losses['wm_total']
                                   + cfg.reward_scale_loss
                                     * ag_losses['reward_mtp_loss'])
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
                    wm_losses, _, agent_hid = world_model_loss(model, batch, cfg)
                    ag_losses = agent_finetune_loss(model, batch,
                                                      agent_hid, cfg)
                    # Drop BC term in P3 (the actor is now driven by
                    # imagination/PMPO; BC against random-action data
                    # would just pull it back toward uniform).  Keep
                    # reward MTP because that head is what the actor's
                    # value target depends on.
                    p3_total_world = (wm_losses['wm_total']
                                       + cfg.reward_scale_loss
                                         * ag_losses['reward_mtp_loss'])
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
                # Critic warm-up (2026-05-06): for the first
                # ``p3_critic_warmup_iters`` P3 iters, only step the
                # critic; freeze the actor.  This prevents a freshly
                # initialised value head from feeding noisy advantages
                # into REINFORCE — under-trained-critic + negative-
                # advantage noise reliably saturates ``log_std`` against
                # the σ clamp ceiling within 1–2 batches.  Letting the
                # critic see real imagined returns first (TD-λ
                # bootstrap) before the actor reacts to its baseline
                # mirrors the actor-critic warm-up convention from
                # DreamerV3 community implementations.  Paper-aligned
                # in spirit (V3 §3.3 EMA-target stabilisation has the
                # same goal); the explicit iteration cap is a stronger
                # version that helps when the P3 budget is short
                # relative to the WM/critic settling time.
                p3_warmup = int(getattr(cfg, 'p3_critic_warmup_iters', 0))
                actor_frozen = (p3_warmup > 0 and p3_iters < p3_warmup)
                if actor_frozen:
                    # Backprop only the critic loss to keep the actor
                    # graph frozen (actor params receive no gradient).
                    ac_losses['critic_loss'].backward()
                else:
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
                    if not actor_frozen:
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
                  f"bc {row.get('bc_loss', 0.0):.4f} "
                  f"actor {row.get('actor_loss', 0.0):+.4f} "
                  f"critic {row.get('critic_loss', 0.0):.4f} "
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
                        # Clamp-aware floor (2026-05-06): the continuous
                        # policy's entropy is upper-bounded by
                        #   H_max(σ_max) = log_std_max + 0.5·log(2πe)
                        # When auto-tune sets a tight σ_max (e.g. 0.30 →
                        # H_max ≈ 0.22), a fixed ``thr`` calibrated to
                        # σ=1 (≈ 0.28) is *above* the maximum attainable
                        # entropy → false-positive "collapse" trip on
                        # every healthy run that hits the clamp.  Move
                        # the trip floor below H_max by a fixed margin
                        # (default 0.10 nats ≈ 10% σ shrink from clamp).
                        if str(getattr(cfg, 'policy_type', 'continuous')
                                ).lower() == 'continuous':
                            log_std_max = float(getattr(cfg,
                                    'policy_log_std_max', 0.0))
                            unit_g = 0.5 * math.log(2.0 * math.pi * math.e)
                            h_max_at_clamp = (
                                float(cfg.action_dim)
                                * (log_std_max + unit_g))
                            margin = float(getattr(cfg,
                                    'early_stop_entropy_collapse_clamp_margin',
                                    0.10))
                            clamp_aware_thr = h_max_at_clamp - margin
                            if clamp_aware_thr < thr:
                                thr = clamp_aware_thr
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

    # Early-stop checkpoint promotion: when training stopped on a P3
    # plateau, the *current* model state has been the same (or worse)
    # as the snapshot at ``best_p3_iter``.  Hard-fail trips
    # (entropy_collapse, critic_divergence, grad_skip_storm) usually
    # corrupt the current state, so the saved best.pt is a more
    # trustworthy controller.  Promote it to final.pt so validation +
    # ONNX export pick it up automatically.
    if (early_stop_reason is not None and best_ckpt_path is not None
            and Path(best_ckpt_path).exists()):
        try:
            shutil.copy2(str(best_ckpt_path), str(final_path))
            print(f'[early-stop] promoted best.pt (iter={best_p3_iter}, '
                  f'ema={best_p3_ema:.3f}) -> final.pt', flush=True)
        except Exception as _e:
            print(f'[early-stop] best->final promotion failed: {_e!r}',
                   flush=True)

    try:
        _save_training_diagnostics_plot(log_path, out_dir / 'training_diagnostics.png')
    except Exception as e:
        print(f'[train] training_diagnostics.png skipped: {e!r}', flush=True)

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
        ('DREAMER_BASELINE_SEED_EPS', 'baseline_seed_episodes', int),
        ('DREAMER_BASELINE_SEED_STD', 'baseline_seed_action_std', float),
        ('DREAMER_RANDOM_SEED_EPS', 'random_seed_episodes', int),
        ('DREAMER_P3_CRITIC_WARMUP_ITERS', 'p3_critic_warmup_iters', int),
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
    return cfg


if __name__ == '__main__':
    cfg = _cfg_from_env()
    summary = train(cfg)
    print(json.dumps(summary, indent=2))
