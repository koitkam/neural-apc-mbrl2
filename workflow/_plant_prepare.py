"""Shared plant-preparation helpers for ``workflow/single_run.py`` and
``workflow/bo_runner.py``.

Both entry-points run the same boilerplate before launching training:

  1. Identify plant dynamics (τ, dead-time).
  2. Build a plant-aware noise config (OU + measurement noise).
  3. Identify a lookback window from the derived sample-rate.
  4. Apply the ``DREAMER_*`` env-var whitelist onto the ``TrainConfig``.

Keeping these in one module avoids the drift class of bug where a fix
applied to one workflow silently misses the other (e.g. the 2026-05-21
``max_lb`` cap fix had to be made in two places, and the
``_env_overrides`` whitelist existed only in ``single_run.py`` for
several commits — every ``DREAMER_*`` override silently lost in BO mode).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple


# ---------------------------------------------------------------------------
# 1. Dynamics identification
# ---------------------------------------------------------------------------

def identify_dynamics(out_dir: Path) -> Dict:
    """Run plant dynamics identification and persist the report.

    Returns a dict with ``tau``, ``dead_time``, ``tau_fast``,
    ``dead_time_fast``, ``dynamics_report`` (path), ``dynamics_raw``
    (full report payload).  Also exports ``DYNAMICS_IDENTIFICATION_JSON``
    and ``IDENTIFIED_TAU_DOMINANT`` / ``IDENTIFIED_DEAD_TIME`` env vars
    for downstream consumers.
    """
    from utils.dynamics_identifier import identify_and_save_dynamics

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dyn_path = out_dir / 'dynamics_identification.json'
    dyn = identify_and_save_dynamics(output_path=str(dyn_path))
    os.environ['DYNAMICS_IDENTIFICATION_JSON'] = str(dyn_path)

    # Missing keys → 0, not a fake 50 s / 5 s plant. Downstream
    # formulas already treat τ≤0 as "use unitless floors"
    # (``derive_sample_rate`` default, ``derive_episode_length`` 1000).
    tau = float(dyn.get('tau_dominant_identified',
                         dyn.get('tau_dominant', 0.0)) or 0.0)
    dead = float(dyn.get('dead_time_identified',
                          dyn.get('dead_time', 0.0)) or 0.0)
    tau_fast = dyn.get('tau_fastest_identified', tau)
    dt_fast = dyn.get('dead_time_fastest_identified', dead)

    os.environ['IDENTIFIED_TAU_DOMINANT'] = f'{tau:g}'
    os.environ['IDENTIFIED_DEAD_TIME'] = f'{dead:g}'

    return {
        'tau': tau,
        'dead_time': dead,
        'tau_fast': float(tau_fast) if tau_fast else float(tau),
        'dead_time_fast': float(dt_fast) if dt_fast else float(dead),
        'dynamics_report': str(dyn_path),
        'dynamics_raw': dyn,
    }


# ---------------------------------------------------------------------------
# 2. Plant-aware noise config
# ---------------------------------------------------------------------------

def build_noise_config(out_dir: Path, *, dynamics_raw: Dict,
                        sample_rate: int,
                        log_prefix: str = '[run]') -> Optional[Path]:
    """Build dynamics-derived OU + measurement noise and persist it as
    ``<out_dir>/noise_config.json``.

    Side-effects: exports ``SIM_NOISE_CONFIG_JSON`` via ``save_noise_config``
    so every downstream subprocess (training, validation) picks up the same
    noise profile through ``SimNoiseWrapper``.

    Returns the written path, or ``None`` if construction fails.  Failures
    are non-fatal (run continues with no process / measurement noise).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        from utils.sim_factory import create_sim
        from utils.noise_config import (
            build_noise_config_from_sim, save_noise_config,
        )
        probe = create_sim(episode_length=10,
                            sample_rate=max(1, int(sample_rate)))
        bare = probe
        for _ in range(4):
            inner = getattr(bare, '_sim', None)
            if inner is None:
                break
            bare = inner
        noise_cfg = build_noise_config_from_sim(
            bare,
            dynamics_json=dynamics_raw or {},
            lookback_json={'identified_lookback': 0},
        )
        noise_cfg_path = out_dir / 'noise_config.json'
        save_noise_config(noise_cfg, str(noise_cfg_path))
        print(f"{log_prefix} noise_config: {noise_cfg_path} "
              f"(OU={len(noise_cfg.get('ou_noise', []))} "
              f"meas={len(noise_cfg.get('measurement_noise', []))})",
              flush=True)
        return noise_cfg_path
    except Exception as exc:
        print(f"{log_prefix} noise_config: SKIPPED ({exc!r}) — running with no "
              "process / measurement noise", flush=True)
        return None


# ---------------------------------------------------------------------------
# 3. Lookback identification
# ---------------------------------------------------------------------------

def identify_lookback(out_dir: Path, *, tau: float, dead_time: float,
                       sample_rate: int, dynamics_raw: Dict,
                       tau_fast: Optional[float] = None,
                       dead_time_fast: Optional[float] = None) -> Dict:
    """Lookback identification using the *derived* ``sample_rate``.

    Must be called after dynamics identification + sample-rate derivation
    so ``min_lb`` / ``max_lb`` reflect the actual scan rate the agent will
    see.

    ``max_lb`` is expressed in raw samples (same units as
    ``identified_lookback``).  The previous formula divided by
    ``sample_rate``, which collapsed the cap to ~τ for any
    ``sample_rate >= 4`` and clamped the inferred seed ``dead + 2τ``
    back down to τ.  P34 (τ=53) showed the WM needs ~3τ worth of context
    to infer hidden OU disturbance state; ``dead + 3τ`` in raw samples
    gives the natural seed ``dead + 2τ`` room to win without artificial
    truncation.
    """
    from utils.lookback_identifier import identify_and_save_lookback

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    lb_path = out_dir / 'lookback_identification.json'
    seed = int(os.environ.get('SEED', '0'))
    sr = int(max(1, sample_rate))
    min_lb = max(8, int(round(tau / sr)))
    max_lb = max(min_lb + 8, int(round(float(dead_time) + 3.0 * float(tau))))
    pair_est = (dynamics_raw.get('per_pair_estimates')
                or dynamics_raw.get('pair_estimates') or [])

    lb = identify_and_save_lookback(
        seed=seed, min_lb=min_lb, max_lb=max_lb,
        output_path=str(lb_path),
        tau_dominant=tau, dead_time=dead_time,
        tau_fastest=tau_fast if tau_fast is not None else tau,
        dead_time_fastest=(dead_time_fast if dead_time_fast is not None
                            else dead_time),
        per_pair_estimates=pair_est,
    )
    lookback = int(lb.get('identified_lookback',
                           lb.get('lookback', max(min_lb, 32))))
    os.environ['IDENTIFIED_LOOKBACK_SEED'] = str(lookback)
    return {'lookback': lookback, 'lookback_report': str(lb_path)}


# ---------------------------------------------------------------------------
# 4. DREAMER_* env-var whitelist
# ---------------------------------------------------------------------------

# Single source of truth for the env-var overrides that map onto
# ``TrainConfig`` fields.  Both ``workflow/single_run.py`` and
# ``workflow/bo_runner.py`` (per-trial) call ``apply_dreamer_env_overrides``
# so a paper-faithful baseline can be launched purely via env vars in
# either workflow with no code changes.
#
# Setting any of these pre-empts the corresponding auto-tune branch via
# the dataclass-default sentinel (``cfg._explicit_fields``).


def _as_bool(s: str) -> bool:
    """Parse an env-var string into a bool (1/true/yes/on -> True)."""
    return str(s).strip().lower() in ('1', 'true', 'yes', 'on', 't', 'y')


def _as_compile_mode(s: str) -> str:
    """Map ``DREAMER_COMPILE`` / ``DREAMER_COMPILE_MODE`` onto TrainConfig
    ``compile_mode``.  ``''`` = eager.  ``1``/true → ``default``."""
    v = str(s).strip().lower()
    if v in ('', '0', 'off', 'false', 'none', 'no'):
        return ''
    if v in ('1', 'true', 'yes', 'on'):
        return 'default'
    return v


def _as_attn_impl(s: str) -> str:
    """Map ``DREAMER_ATTN_IMPL`` / leftover ``DREAMER_FAST_ATTN`` onto
    TrainConfig ``attn_impl``.  ``auto`` = SDPA on CUDA."""
    v = str(s).strip().lower()
    if v in ('0', 'false', 'off', 'manual'):
        return 'manual'
    if v in ('1', 'true', 'yes', 'on', 'sdpa'):
        return 'sdpa'
    if v in ('auto', 'manual', 'sdpa'):
        return v
    raise ValueError(f'unknown attn_impl={s!r}')


def explicit_batch_size() -> Optional[int]:
    """Skip the GPU-calib probe when a batch size is already pinned.

    Canonical ``DREAMER_BATCH_SIZE`` wins.  Leftover ``OBJ_BATCH_SIZE``
    still wins when the DREAMER field is unset (same leftover-env class
    as ``OBJ_REWARD_SCALE``).  ``None`` → empirical probe.
    """
    for key in ('DREAMER_BATCH_SIZE', 'OBJ_BATCH_SIZE'):
        raw = os.environ.get(key, '').strip()
        if not raw:
            continue
        try:
            return max(1, int(raw))
        except ValueError:
            continue
    return None


def gpu_probe_knobs() -> Tuple[float, float, int]:
    """WM-overhead / target util / max_bs for the GPU-calib probe.

    The empirical probe runs before the plant-filled ``TrainConfig`` exists,
    so it used to hard-code ``1.30 / 0.80 / 512`` (or, for BO, silently
    fall through to ``wm_overhead_factor=1.0``).  Identity for env-free
    ``single_run``: those numbers **are** the TrainConfig defaults.
    Changing the dataclass now actually sizes B.  Leftover
    ``DREAMER_WM_OVERHEAD`` / ``DREAMER_TARGET_UTIL`` / ``DREAMER_MAX_BS``
    still win when set (same keys as ``ENV_OVERRIDES``).  Does **not**
    call ``apply_dreamer_env_overrides`` (that would reprint
    ``[env-override]`` at probe time and again at train start).
    """
    from training.train import TrainConfig
    cfg = TrainConfig()
    oh = float(getattr(cfg, 'wm_overhead', 1.30) or 1.30)
    util = float(getattr(cfg, 'gpu_target_util', 0.80) or 0.80)
    max_bs = int(getattr(cfg, 'gpu_max_bs', 512) or 512)
    raw = os.environ.get('DREAMER_WM_OVERHEAD', '').strip()
    if raw:
        try:
            oh = float(raw)
        except ValueError:
            pass
    raw = os.environ.get('DREAMER_TARGET_UTIL', '').strip()
    if raw:
        try:
            util = float(raw)
        except ValueError:
            pass
    raw = os.environ.get('DREAMER_MAX_BS', '').strip()
    if raw:
        try:
            max_bs = int(raw)
        except ValueError:
            pass
    return max(1.0, oh), max(0.1, min(0.95, util)), max(16, max_bs)


ENV_OVERRIDES: Dict[str, tuple] = {
    'DREAMER_GAE_LAMBDA':         ('gae_lambda',                 float),
    'DREAMER_PHASE1_FRAC':        ('phase1_frac',                float),
    'DREAMER_PHASE2_FRAC':        ('phase2_frac',                float),
    'DREAMER_PHASE3_FRAC':        ('phase3_frac',                float),
    'DREAMER_LR_CRITIC':          ('lr_critic',                  float),
    'DREAMER_LR_ACTOR':           ('lr_actor',                   float),
    'DREAMER_LR_WORLD':           ('lr_world',                   float),
    'DREAMER_BC_SCALE':           ('bc_scale',                   float),
    # 2026-05-24 (P48 RCA): structural γ/H mismatch fix.  See
    # auto_tune_seed_buffer in training/train.py for adaptive formula.
    'DREAMER_GAMMA':              ('gamma',                      float),
    'DREAMER_TARGET_CRITIC_TAU':  ('target_critic_tau',          float),
    'DREAMER_P3_COLLECT_EVERY':   ('phase3_collect_every_iters', int),
    # Inner WM / P3 steps per outer iter (already TrainConfig; were not
    # in the whitelist so a DREAMER_* A/B silently lost).
    'DREAMER_TRAIN_STEPS_PER_ITER':    ('train_steps_per_iter',          int),
    'DREAMER_P3_TRAIN_STEPS_PER_ITER': ('phase3_train_steps_per_iter',   int),
    # mbrl2 real-sim (2026-07-08): the dedicated ON-POLICY buffer for the P3
    # actor-critic REINFORCE update (recent current-policy episodes only — the
    # shared replay buffer's off-policy seed actions corrupt the policy gradient,
    # the p01 MV-chatter RCA).  buffer_eps = rolling capacity, prefill_eps = P3-entry warmup.
    'DREAMER_P3_ONPOLICY_BUFFER_EPS':  ('phase3_onpolicy_buffer_eps',  int),
    'DREAMER_P3_ONPOLICY_PREFILL_EPS': ('phase3_onpolicy_prefill_eps', int),
    'DREAMER_BUFFER_CAP_STEPS':   ('buffer_capacity_steps',      int),
    # (DREAMER_EXCITATION_REINJECT_EVERY removed 2026-06-12 — the shared-buffer
    #  re-injection was the p105 anti-pattern; use DREAMER_WM_EXCITATION_BUFFER_
    #  FRAC, the WM-only partition the actor/critic never sample.)
    # 2026-05-19 paper-strip-back knobs (p28 A/B): expose the remaining
    # auto-tuned cfg fields so a fully paper-faithful baseline can be
    # launched purely via env vars.
    'DREAMER_POLICY_LOG_STD_MAX': ('policy_log_std_max',         float),
    'DREAMER_POLICY_LOG_STD_MIN': ('policy_log_std_min',         float),
    'DREAMER_PMPO_ENTROPY_COEF':  ('pmpo_entropy_coef',          float),
    # Auto-tune formula inputs (were ``os.environ.get`` inside
    # ``auto_tune_seed_buffer``; identity defaults).  Leftover
    # ``PMPO_ENTROPY_COEF_BASELINE`` / ``PMPO_ENTROPY_SIGMA_REF`` /
    # ``SEED_TARGET_CV_FRAC`` / ``SEED_SIGMA_CAP`` / ``PRBS_SEG_MIN`` /
    # ``PRBS_SEG_MIN_FLOOR`` still win when the DREAMER_* field is not
    # explicit.  Do not collide with ``DREAMER_PMPO_ENTROPY_COEF`` (the
    # resolved η).
    'DREAMER_PMPO_ENTROPY_ETA_V3':     ('pmpo_entropy_eta_v3',     float),
    'DREAMER_PMPO_ENTROPY_SIGMA_REF':  ('pmpo_entropy_sigma_ref',  float),
    'DREAMER_SEED_TARGET_CV_FRAC':     ('seed_target_cv_frac',     float),
    'DREAMER_SEED_SIGMA_CAP':          ('seed_sigma_cap',          float),
    'DREAMER_PRBS_SEG_MIN':            ('prbs_seg_min',            int),
    'DREAMER_PRBS_SEG_MIN_FLOOR':      ('prbs_seg_min_floor',      int),
    # Resolved PRBS hold (sentinel 0 = auto (θ+4τ)/sr).  Were
    # TrainConfig only; auto-tune already honours ``_explicit_fields``.
    'DREAMER_PRBS_SEED_SEGMENT_STEPS':     ('prbs_seed_segment_steps',     int),
    'DREAMER_PRBS_SEED_SEGMENT_STEPS_MIN': ('prbs_seed_segment_steps_min', int),
    'DREAMER_HORIZON':            ('horizon',                    int),
    # Formula inputs for ``derive_horizon``.  Identity 4.0 / 120.
    # ``horizon_formula_knobs()`` reads TrainConfig then leftover env
    # (called before a plant-filled cfg exists).
    'DREAMER_HORIZON_SETTLE_NTAU': ('horizon_settle_n_tau',       float),
    'DREAMER_HORIZON_MAX':         ('horizon_max',                int),
    # Formula inputs for ``derive_episode_length``.  Identity 20 / 500 / 4000.
    # ``episode_formula_knobs()`` reads TrainConfig then leftover env
    # (called before a plant-filled cfg exists).  Explicit
    # ``SIM_EPISODE_LENGTH`` still hard-overrides the derived length.
    'DREAMER_EPISODE_SETTLE_MULTIPLE': ('episode_settle_multiple', float),
    'DREAMER_EPISODE_MIN_LENGTH':       ('episode_min_length',     int),
    'DREAMER_EPISODE_MAX_LENGTH':      ('episode_max_length',      int),
    # IC domain-randomization + GPU-calib overhead.  ``ic_randomization_knobs()``
    # / ``gpu_probe_knobs()`` read TrainConfig then leftover env (no plant-filled
    # cfg at those call sites).
    'DREAMER_INIT_RANDOMIZATION':      ('init_randomization',      _as_bool),
    'DREAMER_INIT_RANDOMIZATION_FRAC': ('init_randomization_frac', float),
    'DREAMER_WM_OVERHEAD':             ('wm_overhead',             float),
    # GPU-calib budget + hard batch pin.  Were leftover ``os.environ.get``
    # in ``tools/gpu_calibrate.py`` / ``OBJ_BATCH_SIZE`` in single_run/BO
    # (worked, missing from ``run_plan``).  Identity 0.80 / 512.
    # Probe still dual-reads env (runs before TrainConfig exists).
    'DREAMER_TARGET_UTIL':             ('gpu_target_util',         float),
    'DREAMER_MAX_BS':                  ('gpu_max_bs',              int),
    'DREAMER_BATCH_SIZE':              ('batch_size',              int),
    # Per-CV rolling int_err/Δcv/var in aug-obs (P37 ON via leftover env).
    # Sentinel window 0 = auto 2τ/sr.  Identity: default ON / auto.
    'DREAMER_DERIVED_OBSERVABLES':     ('derived_observables',        _as_bool),
    'DREAMER_DERIVED_OBS_WINDOW':      ('derived_observables_window', int),
    # 2026-05-21 (P37 robustness sweep): allow overriding the
    # plant-derived seq_len so longer-context WM training can be
    # launched without code changes.  Useful when the hidden OU
    # autocorrelation requires a chunk longer than the auto-derived
    # settling-time-based default.
    'DREAMER_SEQ_LEN':            ('seq_len',                    int),
    # 2026-05-22 (P37 entropy-floor RCA): reward MTP loss weight.
    # Lowered 4.0 → 1.0 in TrainConfig on 2026-05-22 (P40 RCA); env
    # override remains for tuning experiments.
    'DREAMER_REWARD_MTP_WEIGHT':  ('reward_scale_loss',          float),
    # 2026-05-22 (P40 RCA): P1 reward-MTP weight.  Paper default 0.0
    # (reward head trains only from P2).  Override >0 to re-enable a
    # small P1 weight for experiments.
    'DREAMER_REWARD_MTP_WEIGHT_P1': ('reward_scale_loss_p1',     float),
    # 2026-06-07: exclude expert-injected steps from the reward-head (reward-MTP)
    # supervision so it stays calibrated on the policy's true distribution.
    'DREAMER_REWARD_HEAD_EXCLUDE_EXPERT': ('reward_head_exclude_expert',  _as_bool),
    # 2026-06-07 (Option B): feed measured DV channels as an exogenous WM
    # transition input (held constant in imagination = MPC feedforward) instead
    # of predicting them.  Default ON; DREAMER_DV_AS_INPUT=0 reverts to paper.
    'DREAMER_DV_AS_INPUT':                ('dv_as_input',                _as_bool),
    # DV in the *head* feat (actor sees the load).  Decoder half is a
    # separate knob (``DREAMER_DV_DECODER_FEEDFORWARD``, default off —
    # p146 memoryless dv_t→CV_t skipped GRU dead-time and biased DV
    # DC-gain).  ``DREAMER_DV_FEEDFORWARD=0`` = transition-only DV.
    'DREAMER_DV_FEEDFORWARD':             ('dv_feedforward',             _as_bool),
    'DREAMER_DV_DECODER_FEEDFORWARD':     ('dv_decoder_feedforward',     _as_bool),
    # De-contaminate the disturbance head from the MEASURED dv (2026-06-19,
    # p130): zero the dv-feedforward columns of feat before the disturbance
    # head so it predicts the UNMEASURED load, not the measured DV.  Default ON.
    'DREAMER_DISTURBANCE_HEAD_EXCLUDE_DV': ('disturbance_head_exclude_dv', _as_bool),
    # Detrend window (× settling) for the control-relevant dynamic Kalman score
    # (2026-06-20): the slow drift is feedback-rejectable, so the disturbance
    # metric is also reported high-pass-detrended over this × the settling time.
    'DREAMER_DISTURBANCE_DETREND_SETTLE_MULT': ('disturbance_detrend_settle_mult', float),
    # 2026-05-22: number of constant-action seed episodes (steady-state
    # coverage for the WM before random/imagination data dominates).
    # Default 40 in TrainConfig (P42 SS coverage).  Count is the
    # test_sim sentinel; MIMO covers OP combinations via per-MV
    # permutations, not extra episodes.
    'DREAMER_CONST_ACTION_SEEDS': ('constant_action_seed_episodes', int),
    # Unitless operating-band fractions (sim-agnostic).  Baseline used to
    # read ``DREAMER_BASELINE_SEED_OP_BAND`` via ``os.environ.get`` inside
    # ``train()`` (worked, but missing from ``run_plan``).  Const-action /
    # PRBS bands were TrainConfig-only (``single_run`` could not A/B them).
    'DREAMER_BASELINE_SEED_OP_BAND': ('baseline_seed_op_band',        float),
    'DREAMER_CONST_ACTION_OP_BAND':  ('constant_action_seed_op_band', float),
    'DREAMER_PRBS_SEED_OP_BAND':     ('prbs_seed_op_band',            float),
    # P1 re-inject cadence (P28 follow-up 5/6): EVERY auto-scales from
    # buffer lap; N auto-scales from n_mv/n_dv.  Explicit env wins; 0 disables.
    'DREAMER_CONST_ACTION_INJECT_EVERY':  ('const_action_inject_every',      int),
    'DREAMER_CONST_ACTION_INJECT_N':      ('const_action_inject_n',          int),
    'DREAMER_CONST_ACTION_INJECT_IN_P2':  ('const_action_inject_in_p2',      _as_bool),
    'DREAMER_CONST_ACTION_INJECT_IN_P3':  ('const_action_inject_in_p3',      _as_bool),
    'DREAMER_STEP_TEST_INJECT_EVERY':     ('step_test_inject_every',         int),
    'DREAMER_STEP_TEST_INJECT_N':         ('step_test_inject_n',             int),
    'DREAMER_STEP_TEST_INJECT_IN_P2':     ('step_test_inject_in_p2',         _as_bool),
    'DREAMER_STEP_TEST_INJECT_IN_P3':     ('step_test_inject_in_p3',         _as_bool),
    'DREAMER_DV_PRBS_INJECT_EVERY':       ('dv_prbs_inject_every',           int),
    'DREAMER_DV_PRBS_INJECT_N':           ('dv_prbs_inject_n',               int),
    'DREAMER_DV_PRBS_INJECT_IN_P2':       ('dv_prbs_inject_in_p2',           _as_bool),
    'DREAMER_DV_PRBS_INJECT_IN_P3':       ('dv_prbs_inject_in_p3',           _as_bool),
    'DREAMER_EXPERT_INJECT_EVERY':        ('expert_inject_every',            int),
    'DREAMER_EXPERT_INJECT_N':            ('expert_inject_n',                int),
    'DREAMER_EXPERT_INJECT_IN_P3':        ('expert_inject_in_p3',            _as_bool),
    # 2026-05-22 (P41): MTP head sequence length.  Paper default 8.
    # Bumped to 32 in TrainConfig on 2026-05-21 (p31 RCA) but P40
    # falsified that rationale (0% steady-state WM convergence at
    # H=200 even with L=32 + const-action seeds).  Override to 8 to
    # test paper-faithful setting.
    'DREAMER_MTP_LENGTH':         ('mtp_length',                 int),
    # 2026-05-23 (P42): step-and-settle seed fraction.  0.0 = legacy
    # pure const-action seeds.  0.5 = half const, half step-settle.
    # 1.0 = all step-settle.  Strict superset of const-action
    # supervision; recommended ≥0.5 for plants with long settling
    # times where pure const-action seeds are info-poor.
    'DREAMER_STEP_SETTLE_FRAC':   ('step_settle_seed_fraction',  float),
    # Step-settle |Δu| band + pre-step hold fraction.  Were TrainConfig
    # only (``single_run`` silently dropped A/B).  Identity 0.20 / 0.60
    # / 0.05 / 0.20 (unitless).
    'DREAMER_STEP_SEED_DELTA_MIN': ('step_seed_delta_min',       float),
    'DREAMER_STEP_SEED_DELTA_MAX': ('step_seed_delta_max',       float),
    'DREAMER_STEP_SEED_PREFIX_FRAC_MIN': ('step_seed_prefix_frac_min', float),
    'DREAMER_STEP_SEED_PREFIX_FRAC_MAX': ('step_seed_prefix_frac_max', float),
    # 2026-05-25 (P51): APC step-test seed episodes — mixed MV+DV step
    # events with held baselines.  Strict superset of step_settle
    # (adds DV coverage with held action).  Default 20 in TrainConfig.
    'DREAMER_STEP_TEST_SEEDS':    ('step_test_seed_episodes',    int),
    # Minimum step-test episodes per input channel (n_mv + n_dv).
    'DREAMER_STEP_TEST_PER_CHANNEL': ('step_test_episodes_per_channel', int),
    # Fraction of step-test events that fire in the OVERLAP regime
    # (0.5–1·dyn_horizon apart) vs SETTLED (≥4·dyn_horizon apart).
    'DREAMER_STEP_TEST_OVERLAP_FRAC': ('step_test_overlap_frac', float),
    # Fraction of step-test events that are DV (rest are MV).
    'DREAMER_STEP_TEST_DV_SHARE': ('step_test_dv_share',         float),
    # Fraction of DV events that target the episode's primary DV
    # channel (round-robined across episodes for balanced coverage).
    'DREAMER_STEP_TEST_PRIMARY_DV_BIAS': ('step_test_primary_dv_bias', float),
    # 2026-05-26 (P52 RCA): phase-transition quality gates.  P51
    # entered P2 with an underfit WM and P3 with a critic still
    # bootstrap-leaning → cascade.  Gates make ``phase{1,2}_env_steps``
    # lower bounds + adaptive extensions up to the respective
    # ``max_extension`` × budget cap.
    'DREAMER_P1_GATE_WM_EMA_MIN':  ('p1_gate_wm_ema_min',         float),
    'DREAMER_P1_GATE_PLATEAU_FRAC': ('p1_gate_plateau_frac',      float),
    'DREAMER_P1_GATE_PLATEAU_PROBES': ('p1_gate_plateau_probes',  int),
    'DREAMER_P1_GATE_MAX_EXTENSION': ('p1_gate_max_extension',    float),  # default 1.0
    'DREAMER_P1_GAIN_GATE':        ('p1_gain_gate',               _as_bool),
    'DREAMER_GAIN_READY_LO':       ('gain_ready_lo',              float),
    'DREAMER_GAIN_READY_HI':       ('gain_ready_hi',              float),
    'DREAMER_GAIN_READY_LEVELS':   ('gain_ready_levels',          int),
    'DREAMER_GAIN_READY_NOISE_MAX': ('gain_ready_noise_max',      float),
    'DREAMER_GAIN_READY_FLIP_MAX': ('gain_ready_flip_max',        int),
    'DREAMER_WM_BEST_GAIN_GATE':   ('wm_best_gain_gate',          _as_bool),
    'DREAMER_WM_FIDELITY_EMA_ALPHA': ('wm_fidelity_ema_alpha',    float),
    'DREAMER_WM_FIDELITY_WARMUP_ITERS': ('wm_fidelity_warmup_iters', int),
    'DREAMER_WM_FIDELITY_PATIENCE_ITERS': ('wm_fidelity_patience_iters', int),
    'DREAMER_WM_FIDELITY_CONV_PROBE': ('wm_fidelity_conv_probe',  _as_bool),
    'DREAMER_WM_FIDELITY_CONV_WEIGHT': ('wm_fidelity_conv_weight', float),
    'DREAMER_WM_FIDELITY_RECON_WEIGHT': ('wm_fidelity_recon_weight', float),
    'DREAMER_WM_FIDELITY_GAIN_WEIGHT': ('wm_fidelity_gain_weight', float),
    'DREAMER_WM_FIDELITY_GAIN_GATE_RECON': ('wm_fidelity_gain_gate_recon', float),
    'DREAMER_WM_PROBE_EVERY_ITERS': ('wm_probe_every_iters',      int),
    'DREAMER_HORIZON_R_FLOOR':     ('horizon_r_floor',            float),
    'DREAMER_WM_CONVERGE_EPS_STD': ('wm_converge_eps_std',        float),
    'DREAMER_P2_GATE_REWARD_MTP_MAX': ('p2_gate_reward_mtp_max',  float),
    'DREAMER_P2_GATE_RECENT_ITERS': ('p2_gate_recent_iters',      int),
    'DREAMER_P2_GATE_MAX_EXTENSION': ('p2_gate_max_extension',    float),  # default 0.5
    # 2026-05-27 (P57 RCA): minimum fraction of total_steps reserved
    # for P3 (actor-critic) regardless of P1/P2 extensions.  Default
    # 0.20 in TrainConfig.  Set to 0.0 to disable (legacy behaviour).
    'DREAMER_PHASE3_MIN_FRAC':    ('phase3_min_frac',            float),
    # 2026-05-27 (P59 refactor): σ_max / σ_min auto-tune formula inputs.
    # Canonical DREAMER_* path.  Leftover ``SIGMA_MAX_OVER_SEED`` /
    # ``SIGMA_MAX_FLOOR`` / ``SIGMA_MAX_CAP`` / ``SIGMA_MIN_RATIO_OF_MAX``
    # still win inside ``_resolve_policy_sigma_bounds`` when the field
    # is not explicit.  Auto-tune no longer re-reads DREAMER_* (would
    # beat leftover-vs-explicit order) and no longer floors
    # ``sigma_min_ratio`` at 1.3 (that undid TrainConfig 1.2).
    'DREAMER_SIGMA_MAX_OVER_SEED': ('sigma_max_mult',            float),
    'DREAMER_SIGMA_MAX_FLOOR':     ('sigma_max_floor',           float),
    'DREAMER_SIGMA_MAX_CAP':       ('sigma_max_cap',             float),
    'DREAMER_SIGMA_MIN_RATIO':     ('sigma_min_ratio',           float),
    # P62 (2026-05-28): adaptive negative-tail reward clip — both knobs
    # are dimensionless ratios (sim-agnostic per design principle).
    # See TrainConfig.reward_clip_asymmetry_threshold /
    # reward_clip_tail_k docstrings for rationale.
    'DREAMER_REWARD_CLIP_ASYM_THRESHOLD': ('reward_clip_asymmetry_threshold', float),
    'DREAMER_REWARD_CLIP_TAIL_K':         ('reward_clip_tail_k',              float),
    # P43/P57/P62: raw-reward safety clip + calibration.  Were
    # ``os.environ.get`` in APCEnv / ``train()`` (worked, missing from
    # ``run_plan``).  Defaults unchanged (−1e6 / 1e18 / baseline /
    # percentile / p95 / 0.5 / sym_mag 6.0).
    'DREAMER_REWARD_RAW_CLIP_MIN':        ('reward_raw_clip_min',           float),
    'DREAMER_REWARD_RAW_CLIP_MAX':        ('reward_raw_clip_max',           float),
    'DREAMER_REWARD_CAL_MODE':             ('reward_cal_mode',              str),
    'DREAMER_REWARD_CAL_TARGET':           ('reward_cal_target',            str),
    'DREAMER_REWARD_CAL_PCT':              ('reward_cal_pct',               float),
    'DREAMER_REWARD_CAL_PCT_VAL':          ('reward_cal_pct_val',           float),
    'DREAMER_REWARD_CAL_TARGET_SYM_MAG':   ('reward_cal_target_sym_mag',    float),
    # Gate for percentile→twohot scale.  Was leftover ``OBJ_REWARD_SCALE``
    # (worked, missing from ``run_plan``).  Identity default ``auto``.
    'DREAMER_OBJ_REWARD_SCALE':            ('obj_reward_scale',             str),
    # Reward-engine leftovers (``utils/objective_runtime.py``).  Identity.
    # Dual-read leftover ``OBJECTIVE_*`` / ``OBJ_AUTO_*``.  Clip ``<0``
    # = adaptive.  ``obj_auto_cv_over_econ_ratio=0`` follows margin.
    'DREAMER_OBJECTIVE_INTEGRAL_COEF':     ('objective_integral_coef',      float),
    'DREAMER_OBJECTIVE_INTEGRAL_WINDUP':   ('objective_integral_windup',    float),
    'DREAMER_OBJECTIVE_INTEGRAL_LEAK':     ('objective_integral_leak',      float),
    'DREAMER_OBJ_AUTO_INTEGRAL_SOFT_COMPENSATE': (
        'obj_auto_integral_soft_compensate', _as_bool),
    'DREAMER_OBJ_AUTO_INTEGRAL_SOFT_COMPENSATE_MAX': (
        'obj_auto_integral_soft_compensate_max', float),
    'DREAMER_OBJ_AUTO_INTEGRAL_DEADTIME_K': (
        'obj_auto_integral_deadtime_k', float),
    'DREAMER_OBJ_AUTO_VIOLATION_MARGIN':   ('obj_auto_violation_margin',    float),
    'DREAMER_OBJ_AUTO_CV_OVER_ECON_RATIO': ('obj_auto_cv_over_econ_ratio',  float),
    'DREAMER_OBJ_AUTO_VIOLATION_TOLERANCE': (
        'obj_auto_violation_tolerance', float),
    # Auto-weights leftover formula knobs.  Identity.  Dual-read leftover
    # ``OBJ_AUTO_*`` inside ``derive_auto_weights``.
    'DREAMER_OBJ_AUTO_MV_VIOLATION_BASE': (
        'obj_auto_mv_violation_base', float),
    'DREAMER_OBJ_AUTO_CV_VIOLATION_BASE': (
        'obj_auto_cv_violation_base', float),
    'DREAMER_OBJ_AUTO_CV_RANK_DECAY':      ('obj_auto_cv_rank_decay',       float),
    'DREAMER_OBJ_AUTO_MV_OVER_CV_RATIO':   ('obj_auto_mv_over_cv_ratio',    float),
    'DREAMER_OBJ_AUTO_ECON_OVER_TARGET_RATIO': (
        'obj_auto_econ_over_target_ratio', float),
    'DREAMER_OBJ_AUTO_TARGET_BASE':        ('obj_auto_target_base',         float),
    'DREAMER_OBJ_AUTO_CV_PENALTY_CAP_FRAC': (
        'obj_auto_cv_penalty_cap_frac', float),
    'DREAMER_OBJ_AUTO_TYPICAL_CV_VIOLATION': (
        'obj_auto_typical_cv_violation', float),
    'DREAMER_OBJ_AUTO_MOVE_OVER_CV_K':     ('obj_auto_move_over_cv_k',      float),
    'DREAMER_OBJ_AUTO_MOVE_BASE':          ('obj_auto_move_base',           float),
    'DREAMER_OBJ_AUTO_MOVE_TARGET_COST_FRAC': (
        'obj_auto_move_target_cost_frac', float),
    'DREAMER_OBJ_AUTO_MOVE_SIGMA_REF':     ('obj_auto_move_sigma_ref',      float),
    'DREAMER_OBJ_AUTO_ECON_OVER_MOVE_RATIO': (
        'obj_auto_econ_over_move_ratio', float),
    'DREAMER_OBJ_AUTO_REVERSAL_GAIN':      ('obj_auto_reversal_gain',       float),
    'DREAMER_OBJ_AUTO_VIOLATION_RATE_COEF_DIVISOR': (
        'obj_auto_violation_rate_coef_divisor', float),
    'DREAMER_OBJ_AUTO_VIOLATION_RATE_COEF_MIN': (
        'obj_auto_violation_rate_coef_min', float),
    'DREAMER_OBJ_AUTO_VIOLATION_RATE_COEF_MAX': (
        'obj_auto_violation_rate_coef_max', float),
    'DREAMER_OBJ_AUTO_DIFFERENTIABLE_DEPTH': (
        'obj_auto_differentiable_depth', float),
    'DREAMER_OBJ_AUTO_REWARD_CLIP_FLOOR':  ('obj_auto_reward_clip_floor',   float),
    'DREAMER_OBJ_USE_NORMALIZED':          ('objective_use_normalized',     _as_bool),
    'DREAMER_OBJECTIVE_VIOLATION_RATE_COEF': (
        'objective_violation_rate_coef', str),
    'DREAMER_OBJECTIVE_PENALTY_SAT_MODE':  ('objective_penalty_sat_mode',   str),
    'DREAMER_OBJECTIVE_PENALTY_CLIP':      ('objective_penalty_clip',       float),
    'DREAMER_OBJECTIVE_REWARD_CLIP':       ('objective_reward_clip',        float),
    'DREAMER_OBJECTIVE_FEASIBILITY_CAP':   ('objective_feasibility_cap',    float),
    'DREAMER_OBJECTIVE_FEASIBILITY_SCALE': ('objective_feasibility_scale',  float),
    # May-2026 P39 probes.  Default 0 / off.  Extra retain_graph
    # backward — not env-free.  Opt in ``=10``.
    'DREAMER_DIAG_PERHEAD_GRADS_EVERY':    ('diag_perhead_grads_every',      int),
    'DREAMER_DIAG_LATENT_STABILITY_EVERY': ('diag_latent_stability_every',   int),
    'DREAMER_DIAG_DISABLE_REWARD_MTP_IN_P1': (
        'diag_disable_reward_mtp_in_p1', _as_bool),
    'DREAMER_DIAG_REWARD_MTP_STOP_GRAD_IN_P1': (
        'diag_reward_mtp_stop_grad_in_p1', _as_bool),
    # End-of-training WM diagnostic (default ON).  Horizon 0 = auto.
    'DREAMER_RUN_WM_DIAGNOSTIC':           ('run_wm_diagnostic',            _as_bool),
    'DREAMER_WM_DIAG_N_STARTS':          ('wm_diag_n_starts',             int),
    'DREAMER_WM_DIAG_HORIZON':           ('wm_diag_horizon',              int),
    # Eval TM protocol + val-suite gates.  Were ``os.environ.get`` in
    # ``evaluation/`` (worked, missing from ``run_plan``).  Identity
    # defaults (levels=5, span=0.6, step_frac=0.4, horizon/settle 0=auto,
    # all three val gates ON).  ``DREAMER_WM_DIAG_DEVICE`` stays env-only.
    'DREAMER_WM_TF_LEVELS':              ('wm_tf_levels',                 int),
    'DREAMER_WM_TF_SPAN':                ('wm_tf_span',                   float),
    'DREAMER_WM_TF_STEP_FRAC':           ('wm_tf_step_frac',              float),
    'DREAMER_WM_TF_HORIZON':             ('wm_tf_horizon',                int),
    'DREAMER_WM_TF_SETTLE':              ('wm_tf_settle',                 int),
    'DREAMER_VAL_WM_TRANSFER':           ('val_wm_transfer',              _as_bool),
    'DREAMER_VAL_WM_POSTPRIOR':          ('val_wm_postprior',             _as_bool),
    'DREAMER_VAL_WM_DISTPRED':           ('val_wm_distpred',              _as_bool),
    # P79 (2026-06-02): return-scale ABSOLUTE cap — dimensionless (return
    # units).  Arrests the critic-pessimism cascade runaway once the spread
    # is implausibly large WITHOUT throttling legitimate early growth (the
    # P63 growth-rate clamp regressed for that reason).  Set 0.0 to recover
    # the paper-faithful uncapped EMA.
    'DREAMER_RETURN_SCALE_ABS_CAP':       ('return_scale_abs_cap',           float),
    # P26 RCA / P27: freeze the p95-p05 EMA after critic warmup (default ON).
    'DREAMER_RETURN_SCALE_FREEZE_AFTER_WARMUP': (
        'return_scale_freeze_after_warmup', _as_bool),
    # P26 RCA / P27: TD3 min-of-N twohot critics (default 2).
    'DREAMER_N_CRITICS':                  ('n_critics',                      int),
    # P45 RCA / P46: restore Gaussian σ at P3 entry (default off).
    'DREAMER_P3_RESET_LOG_STD':           ('p3_reset_log_std',               _as_bool),
    # P51: P3 REINFORCE stop-grad log_std (default ON). Opt out ``=0``.
    'DREAMER_P3_STOP_GRAD_LOG_STD':       ('p3_stop_grad_log_std',           _as_bool),
    # P52: clamp P3 REINFORCE logp (default 8). Opt out ``=0``.
    'DREAMER_P3_LOGP_CLIP':               ('p3_logp_clip',                   float),
    # P53: PPO ratio clip vs frozen unfreeze-μ (default 0.2). Opt out ``=0``.
    'DREAMER_P3_MU_RATIO_CLIP':           ('p3_mu_ratio_clip',               float),
    # P55: recopy μ-ratio snapshot every N P3 iters (default 1 = PPO
    # epoch per collect). 0 = P53 freeze-forever. Opt out ``=0``.
    'DREAMER_P3_MU_RATIO_REFRESH':       ('p3_mu_ratio_refresh_iters',    int),
    'DREAMER_WM_GRAD_SKIP_NORM':          ('wm_grad_skip_norm',              float),
    # P27 RCA / P28: restore wm_best and continue to P2 on a P1 skip-storm
    # instead of aborting the run (default ON).
    'DREAMER_SKIP_STORM_RECOVER_P1':      ('skip_storm_recover_p1',          _as_bool),
    'DREAMER_SKIP_STORM_LAST_OK_RECON_RATIO': (
        'skip_storm_last_ok_recon_ratio', float),
    # P40: lock last-ok after a silent recon spike (default 20× best).
    'DREAMER_SKIP_STORM_LAST_OK_LOCK_RATIO': (
        'skip_storm_last_ok_lock_ratio', float),
    # P31: first skip-storm continues P1; Nth caps (default 2).
    'DREAMER_SKIP_STORM_P1_CAP_AFTER':     ('skip_storm_p1_cap_after',     int),
    # P28 GPU RCA: P1→P2 fidelity-peak restore is gain-blind. Default OFF.
    'DREAMER_WM_BEST_RESTORE_AT_P2':      ('wm_best_restore_at_p2',       _as_bool),
    'DREAMER_WM_BEST_RESTORE_AT_P3':      ('wm_best_restore_at_p3',       _as_bool),
    'DREAMER_WM_BEST_RESTORE_MIN_GAP':    ('wm_best_restore_min_gap',     int),
    # P29 bookkeeping: skip P3 when the freeze is GAIN_NOT_READY / wm_best
    # skip-storm fallback (default ON). Observer validation still runs.
    'DREAMER_SKIP_INVALID_P3':            ('skip_invalid_p3',             _as_bool),
    # Cascade RCA (2026-05-29): the two corrected anti-cascade fixes.
    # A' — potential-based reward shaping (dense, policy-invariant, same
    # γ; training-only, validation scores on unshaped raw_reward).
    # C — replay-grounded critic anchor: pins the critic to a TD-λ
    # target from REAL buffered rewards + slow-target bootstrap on the
    # REAL latents, breaking the self-referential growing-negative
    # fixed point (critic_target_v_r→0.95) that drives the cascade.
    # Both sim-agnostic dimensionless coefficients.
    'DREAMER_REWARD_SHAPING_COEF':        ('reward_shaping_coef',            float),
    # Fix 2a (2026-06-19, p129): margin-gated economic shaping weight + the
    # CV-safety-margin gate width.  Φ = Φ_safe + coef·gate·Φ_econ; econ pull is
    # suppressed near a constraint.  Policy-invariant.  0.0 disables Φ_econ.
    'DREAMER_SHAPING_ECON_COEF':          ('shaping_econ_coef',              float),
    'DREAMER_SHAPING_ECON_MARGIN_FRAC':   ('shaping_econ_margin_frac',       float),
    # RANGE/limit Φ flat-top width (p125).  Sibling of econ-margin;
    # was TrainConfig-only so ``single_run`` dropped A/B.  Identity 0.25.
    'DREAMER_SHAPING_SAFE_MARGIN_FRAC':   ('shaping_safe_margin_frac',       float),
    # Imagination-era critic_replay_anchor / critic_anchor_* / imag CE
    # REMOVED (never read by ``_realsim_actor_critic_step``).  Real-sim
    # grounding is ``DREAMER_CRITIC_MC_GROUNDING_COEF`` (default 2.0).
    'DREAMER_CRITIC_MC_GROUNDING_COEF':   ('critic_mc_grounding_coef',       float),
    'DREAMER_MV_HARD_CLAMP':              ('mv_hard_clamp',                  _as_bool),
    'DREAMER_MV_ACTION_FULL_RANGE':       ('mv_action_map_full_range',       _as_bool),
    'DREAMER_RUNTIME_SETPOINT_VARIATION': ('runtime_setpoint_variation',     _as_bool),
    # APCEnv schedule (dataclass 0.15 / 0.20 / 1–2 / 0.10 / 3 / 0.05).
    # Dual-read leftover ``RUNTIME_SETPOINT_*_JITTER_FRACTION``.  Do **not**
    # switch APCEnv to ``auto_derive`` (τ-derived change-count / ramp).
    'DREAMER_RUNTIME_SETPOINT_BOUNDS_JITTER_FRAC': (
        'runtime_setpoint_bounds_jitter_frac', float),
    'DREAMER_RUNTIME_SETPOINT_TARGET_JITTER_FRAC': (
        'runtime_setpoint_target_jitter_frac', float),
    'DREAMER_RUNTIME_SETPOINT_BOUNDS_CHANGES_MIN': (
        'runtime_setpoint_bounds_changes_min', int),
    'DREAMER_RUNTIME_SETPOINT_BOUNDS_CHANGES_MAX': (
        'runtime_setpoint_bounds_changes_max', int),
    'DREAMER_RUNTIME_SETPOINT_TARGET_CHANGES_MIN': (
        'runtime_setpoint_target_changes_min', int),
    'DREAMER_RUNTIME_SETPOINT_TARGET_CHANGES_MAX': (
        'runtime_setpoint_target_changes_max', int),
    'DREAMER_RUNTIME_SETPOINT_RAMP_DURATION_FRAC': (
        'runtime_setpoint_ramp_duration_frac', float),
    'DREAMER_RUNTIME_SETPOINT_CURRICULUM_WARMUP_FRAC': (
        'runtime_setpoint_curriculum_warmup_frac', float),
    'DREAMER_RUNTIME_SETPOINT_N_MAGNITUDE_STRATA': (
        'runtime_setpoint_n_magnitude_strata', int),
    'DREAMER_RUNTIME_SETPOINT_TARGET_INSIDE_MARGIN_FRAC': (
        'runtime_setpoint_target_inside_margin_frac', float),
    # ---- World-model backbone (P68, 2026-05-30) ----
    # ``rssm`` (default) vs ``sf_transformer``; RSSM latent sizes and
    # KL-balance knobs.  See TrainConfig for paper rationale.
    'DREAMER_WORLD_MODEL_TYPE':           ('world_model_type',               str),
    'DREAMER_RSSM_DETER_DIM':             ('rssm_deter_dim',                 int),
    'DREAMER_RSSM_N_CATEGORICALS':        ('rssm_n_categoricals',            int),
    'DREAMER_RSSM_N_CLASSES':             ('rssm_n_classes',                 int),
    'DREAMER_RSSM_EMBED_DIM':             ('rssm_embed_dim',                 int),
    'DREAMER_RSSM_HIDDEN_DIM':            ('rssm_hidden_dim',                int),
    'DREAMER_RSSM_UNIMIX':                ('rssm_unimix',                    float),
    'DREAMER_RSSM_FREE_BITS':             ('rssm_free_bits',                 float),
    'DREAMER_RSSM_KL_DYN_W':              ('rssm_kl_dyn_w',                  float),
    'DREAMER_RSSM_KL_REPR_W':             ('rssm_kl_repr_w',                 float),
    # Latent type (categorical|deterministic) + joint-embedding consistency.
    'DREAMER_RSSM_LATENT_TYPE':           ('rssm_latent_type',               str),
    'DREAMER_RSSM_LATENT_NOISE':          ('rssm_latent_noise',              float),
    'DREAMER_RSSM_JOINT_EMBED_COEF':      ('rssm_joint_embed_coef',          float),
    # torch.compile (P29 leftover): env-free is eager.  COMPILE before
    # COMPILE_MODE so an explicit mode wins if both are set.  ``1`` →
    # ``default``.  Also read in ``_resolve_compile_mode`` so tests that
    # skip ``apply_dreamer_env_overrides`` still work.
    'DREAMER_COMPILE':                    ('compile_mode',                   _as_compile_mode),
    'DREAMER_COMPILE_MODE':               ('compile_mode',                   _as_compile_mode),
    # Attention backend.  Env-free ``auto`` = SDPA on CUDA.  FAST_ATTN
    # first so ATTN_IMPL wins if both are set.  Was CLI-only
    # (``single_run`` silently dropped ATTN_IMPL) + constructor env.
    'DREAMER_FAST_ATTN':                  ('attn_impl',                      _as_attn_impl),
    'DREAMER_ATTN_IMPL':                  ('attn_impl',                      _as_attn_impl),
    # TSSM (transformer-SSM) backbone dims (world_model_type='tssm').
    'DREAMER_TSSM_D_MODEL':               ('tssm_d_model',                   int),
    'DREAMER_TSSM_N_LAYERS':              ('tssm_n_layers',                  int),
    'DREAMER_TSSM_N_HEADS':               ('tssm_n_heads',                   int),
    'DREAMER_TSSM_MAX_SEQ_LEN':           ('tssm_max_seq_len',               int),
    # P73 (2026-05-31): bounded training reward (cascade root-cause fix).
    # symlog-squash per-step training reward into [-B,B] so imagined returns
    # stay bounded and return_scale cannot run away.  Sim-agnostic.
    # P77: bounded path is now a scale-invariant linear remap
    # reward = clip(raw * B/reward_clip_ref, -B, B); _REF is the fallback
    # reward_clip_ref when objective_runtime does not expose one.
    'DREAMER_BOUND_TRAINING_REWARD':      ('bound_training_reward',          _as_bool),
    'DREAMER_BOUND_TRAINING_REWARD_MAX':  ('bound_training_reward_max',      float),
    'DREAMER_BOUND_TRAINING_REWARD_REF':  ('bound_training_reward_ref',      float),
    # P74 (2026-05-31): advantage clip (smooths actor grad -> less MV chatter).
    'DREAMER_ADVANTAGE_CLIP':             ('advantage_clip',                 float),
    # P81 (2026-06-03): APC steady-state expert (BC anchor for the policy mean).
    # expert_type ∈ {none, static, nn}; bc_scale auto-set to expert_bc_scale when
    # the expert is usable (cloning MASKED to expert steps).  See
    # utils/apc_expert.py + TrainConfig for rationale.  Dimensionless / sim-
    # adaptive.  Move-law knobs are TrainConfig + whitelist below.
    'DREAMER_EXPERT_TYPE':                ('expert_type',                    str),
    'DREAMER_EXPERT_BC_SCALE':            ('expert_bc_scale',                float),
    # P50: P1/P2 BC clones μ only (default ON). Legacy Gaussian NLL
    # pinned σ_min (P45–P49). Opt out ``=0``.
    'DREAMER_BC_MEAN_ONLY':               ('bc_mean_only',                   _as_bool),
    'DREAMER_EXPERT_SEED_EPISODES':       ('expert_seed_episodes',           int),
    'DREAMER_EXPERT_ACTION_JITTER':       ('expert_action_jitter',           float),
    'DREAMER_EXPERT_KEEP_SCHEDULE':       ('expert_keep_schedule',           _as_bool),
    'DREAMER_EXPERT_USE_SS_SAMPLES':      ('expert_use_ss_samples',          _as_bool),
    # Move-law (were leftover ``os.environ.get`` in ``apc_expert``).
    # Identity 0.30 / 0.12 / 0.02 / 0.6 / 0.05 / 0.02 / 1.0 / 40 / 0.1.
    # Dual-read leftover when the field is not explicit.
    'DREAMER_EXPERT_MOVE_FRAC':           ('expert_move_frac',               float),
    'DREAMER_EXPERT_BACKOFF_FRAC':        ('expert_backoff_frac',            float),
    'DREAMER_EXPERT_ECON_FRAC':           ('expert_econ_frac',               float),
    'DREAMER_EXPERT_LOOP_GAIN':           ('expert_loop_gain',               float),
    'DREAMER_EXPERT_RIDGE_FRAC':          ('expert_ridge_frac',              float),
    'DREAMER_EXPERT_FEAS_SCALE':          ('expert_feas_scale',              float),
    'DREAMER_EXPERT_ECON_SCALE':          ('expert_econ_scale',              float),
    'DREAMER_EXPERT_OPT_ITERS':           ('expert_opt_iters',               int),
    'DREAMER_EXPERT_OPT_LR':              ('expert_opt_lr',                  float),
    # P83: decaying P3 expert-BC anchor (default ON via TrainConfig; expose
    # for ablation).  expert_bc_p3 toggles the anchor, _floor sets the decay
    # floor, _adaptive_scale enables the TD3+BC return-scale normalisation.
    'DREAMER_EXPERT_BC_P3':               ('expert_bc_p3',                   _as_bool),
    'DREAMER_EXPERT_BC_P3_FLOOR':         ('expert_bc_p3_floor',             float),
    'DREAMER_EXPERT_BC_P3_ADAPTIVE_SCALE': ('expert_bc_p3_adaptive_scale',   _as_bool),
    # (a) adaptive bounded-return envelope (default ON; both backbones).
    'DREAMER_RETURN_VALUE_ADAPTIVE_CAP':  ('return_value_adaptive_cap',      _as_bool),
    'DREAMER_RETURN_VALUE_CAP_K':         ('return_value_cap_k',             float),
    'DREAMER_RETURN_VALUE_CAP_GAMMA_HORIZON': ('return_value_cap_gamma_horizon', _as_bool),
    # (b) WM held-action steady-state consistency loss (default ON; both backbones).
    'DREAMER_WM_STEADY_CONSISTENCY_COEF': ('wm_steady_consistency_coef',     float),
    'DREAMER_WM_STEADY_SETTLE_FRAC':      ('wm_steady_settle_frac',          float),
    'DREAMER_WM_STEADY_HELD_EPS':         ('wm_steady_held_eps',             float),
    'DREAMER_WM_HELD_ROLLOUT_COEF':       ('wm_held_rollout_coef',           float),
    'DREAMER_WM_HELD_ROLLOUT_LEN':        ('wm_held_rollout_len',            int),
    'DREAMER_WM_HELD_ROLLOUT_SETTLE_FRAC':('wm_held_rollout_settle_frac',    float),
    'DREAMER_WM_HELD_ROLLOUT_WIN':        ('wm_held_rollout_win',            int),
    'DREAMER_WM_HELD_ROLLOUT_MAX_STARTS': ('wm_held_rollout_max_starts',     int),
    'DREAMER_WM_HELD_ROLLOUT_GATE_RECON': ('wm_held_rollout_gate_recon',     float),
    # (P89) noise curriculum + clean steady-state seeds (default ON).
    # Ramp / hidden-load schedule knobs were leftover ``os.environ.get``
    # (worked, missing from ``run_plan``).  Identity defaults.  Dual-read
    # at noise_config / hidden_disturbance (APCEnv.reset has cfg).
    'DREAMER_CLEAN_STEADY_SEEDS':         ('clean_steady_seeds',             _as_bool),
    'DREAMER_PROCESS_NOISE_CURRICULUM':   ('process_noise_curriculum',       _as_bool),
    'DREAMER_PROCESS_NOISE_AMP_RAMP':     ('process_noise_amp_ramp',         str),
    # Plant SNR (process OU + measurement).  Were leftover ``SIM_*``
    # ``os.environ.get`` in ``utils/noise_config.py`` (worked, missing
    # from ``run_plan``).  Identity.  Dual-read at bake (phase 1a,
    # before TrainConfig).  ``SIM_NOISE_CONFIG_JSON`` stays env-only.
    'DREAMER_SIM_NOISE_ADAPTIVE':         ('sim_noise_adaptive',            _as_bool),
    'DREAMER_SIM_OU_SIGMA_FRAC':          ('sim_ou_sigma_frac',             float),
    'DREAMER_SIM_OU_GAIN_CV':             ('sim_ou_gain_cv',                float),
    'DREAMER_SIM_OU_GAIN_DV':             ('sim_ou_gain_dv',                float),
    'DREAMER_SIM_MEAS_NOISE_CV_FRAC':     ('sim_meas_noise_cv_frac',        float),
    'DREAMER_SIM_MEAS_NOISE_DV_FRAC':     ('sim_meas_noise_dv_frac',        float),
    # Runtime wrapper.  Were leftover ``SIM_NOISE_SEED`` /
    # ``SIM_NOISE_JITTER_PCT`` / ``SIM_DOMAIN_RANDOMIZATION``.
    # Identity.  Dual-read at wrap / DomainRandomizer (no cfg).
    'DREAMER_SIM_NOISE_ENABLED':          ('sim_noise_enabled',            _as_bool),
    'DREAMER_SIM_NOISE_SEED':             ('sim_noise_seed',               str),
    'DREAMER_SIM_NOISE_JITTER_PCT':       ('sim_noise_jitter_pct',         float),
    'DREAMER_SIM_DOMAIN_RANDOMIZATION':   ('sim_domain_randomization',     _as_bool),
    'DREAMER_SIM_DOMAIN_RANDOMIZATION_SEED': ('sim_domain_randomization_seed', str),
    # Sentinel <0 = identifier-derived bake.  Leftover SIM_PARAM was
    # dead after wrap overwrite.  Identity env-free (auto).
    'DREAMER_SIM_PARAM_RANDOMIZATION_PCT': ('sim_param_randomization_pct', float),
    # Operator-event schedule.  Were leftover ``AGENT_DISTURBANCE_*``
    # (worked, missing from ``run_plan``).  Identity.  Dual-read at
    # ``build_training_disturbance_schedule``.  Settle 0 = auto.
    'DREAMER_DISTURBANCE_AUTHORITY_FRAC': ('disturbance_authority_frac',    float),
    'DREAMER_DISTURBANCE_RECOVERY_FRAC':  ('disturbance_recovery_frac',     float),
    'DREAMER_DISTURBANCE_SETTLE_STEPS':   ('disturbance_settle_steps',      int),
    'DREAMER_DISTURBANCE_QUIET_FRAC':     ('disturbance_quiet_frac',        float),
    'DREAMER_HIDDEN_DISTURBANCE':         ('hidden_disturbance',             _as_bool),
    'DREAMER_HIDDEN_OU_AMP_RAMP':         ('hidden_ou_amp_ramp',             str),
    'DREAMER_HIDDEN_OU_AMP_MAX_SCALE':    ('hidden_ou_amp_max_scale',        float),
    'DREAMER_HIDDEN_OU_AMP_MAX_SCALE_P3': ('hidden_ou_amp_max_scale_p3',     float),
    'DREAMER_HIDDEN_OU_AMP_JITTER':       ('hidden_ou_amp_jitter',           str),
    'DREAMER_HIDDEN_OU_DRIFT_FRAC':       ('hidden_ou_drift_frac',           float),
    'DREAMER_DISTURBANCE_PROB_AGENT':     ('disturbance_prob_agent',         float),
    'DREAMER_DISTURBANCE_PROB_P2':        ('disturbance_prob_p2',            float),
    'DREAMER_DISTURBANCE_PROB_WM':        ('disturbance_prob_wm',            float),
    'DREAMER_HIDDEN_OU_PROB_P3_RAMP_REACH': ('hidden_ou_prob_p3_ramp_reach', float),
    'DREAMER_HIDDEN_OU_PROB_P2_RAMP_REACH': ('hidden_ou_prob_p2_ramp_reach', float),
    'DREAMER_HIDDEN_OU_PROB_MIN':         ('hidden_ou_prob_min',             float),
    'DREAMER_HIDDEN_OU_PROB_MAX':         ('hidden_ou_prob_max',             float),
    'DREAMER_HIDDEN_OU_PROB_TARGET_SCORE': ('hidden_ou_prob_target_score',   float),
    'DREAMER_HIDDEN_DIST_SETTLE_NTAU':    ('hidden_dist_settle_n_tau',       float),
    'DREAMER_HIDDEN_DIST_MAX_EVENTS':     ('hidden_dist_max_events',         int),
    'DREAMER_HIDDEN_DIST_P_ISOLATED':     ('hidden_dist_p_isolated',         float),
    'DREAMER_HIDDEN_DIST_P_REVERT':       ('hidden_dist_p_revert',           float),
    'DREAMER_HIDDEN_DIST_SHAPE_WEIGHTS':  ('hidden_dist_shape_weights',      str),
    'DREAMER_HIDDEN_DIST_SPREAD':         ('hidden_dist_spread',             _as_bool),
    'DREAMER_HIDDEN_DIST_TAU_FRAC':       ('hidden_dist_tau_frac',           str),
    'DREAMER_HIDDEN_DIST_DEADTIME_FRAC':  ('hidden_dist_deadtime_frac',      str),
    'DREAMER_WM_FREEZE_AFTER_P1':         ('wm_freeze_after_p1',             _as_bool),
    # WM-fix workstream (2026-06-09): all default-OFF (identity to p106).
    'DREAMER_WM_FREEZE_AFTER_ITERS':      ('wm_freeze_after_iters',          int),
    'DREAMER_WM_RECON_CV_WEIGHT':         ('wm_recon_cv_weight',             float),
    'DREAMER_BC_TRACK_EXPERT_EVERY':      ('bc_track_expert_every',          int),
    'DREAMER_P3_CRITIC_WARMUP_ITERS':     ('p3_critic_warmup_iters',         int),
    'DREAMER_WM_TRUNK_STOPGRAD_IN_P2':    ('wm_trunk_stopgrad_in_p2',        _as_bool),
    'DREAMER_TRAIN_MODE':                 ('train_mode',                     str),
    # Imagination actor is deleted.  Override is fail-loud in train()
    # (``_require_realsim_actor``) so a leftover env is not a false A/B.
    'DREAMER_ACTOR_SOURCE':               ('actor_train_source',             str),
    'DREAMER_JOINT_PRIOR_REFRESH_ITERS':  ('joint_prior_refresh_iters',      int),
    # Early-stop knobs (mirror train.py _cfg_from_env names) so single_run/bo
    # runs can relax/disable the stops for diagnostic runs (e.g. let a run
    # continue PAST the entropy-collapse stop to observe the cascade trajectory).
    'DREAMER_EARLY_STOP':                 ('early_stop_enable',              _as_bool),
    'DREAMER_ES_P3_PATIENCE':             ('early_stop_p3_patience_iters',   int),
    'DREAMER_ES_P3_MIN_IMPROVEMENT':      ('early_stop_p3_min_improvement',  float),
    'DREAMER_ES_ENT_FRAC':                ('early_stop_entropy_collapse_frac',          float),
    'DREAMER_ES_ENT_FLOOR_FRAC':          ('early_stop_entropy_collapse_floor_frac',    float),
    'DREAMER_ES_ENT_PATIENCE':            ('early_stop_entropy_collapse_patience_iters', int),
    'DREAMER_ES_ENT_WINDOW':              ('early_stop_entropy_collapse_window_iters',   int),
    'DREAMER_ES_ENT_MIN_BELOW':           ('early_stop_entropy_collapse_min_frac_below', float),
    # Were only in train.py ``_cfg_from_env`` — single_run/bo silently ignored them.
    'DREAMER_ES_CRITIC_FACTOR':           ('early_stop_critic_divergence_factor', float),
    'DREAMER_ES_CRITIC_PATIENCE':         ('early_stop_critic_divergence_patience_iters', int),
    'DREAMER_ES_CASCADE_REWVAR':          ('early_stop_cascade_min_rew_var_frac', float),
    'DREAMER_ES_CASCADE_GROWTH':          ('early_stop_cascade_min_return_scale_growth', float),
    'DREAMER_ES_CASCADE_PATIENCE':        ('early_stop_cascade_patience_iters', int),
    'DREAMER_ES_GRADSKIP_WINDOW':         ('early_stop_grad_skip_window_iters', int),
    'DREAMER_ES_GRADSKIP_MAX':            ('early_stop_grad_skip_max', int),
    'DREAMER_ES_P1_MIN_SF_DROP':          ('early_stop_p1_min_sf_drop_frac', float),
    'DREAMER_ES_P2_MAX_RMTP':             ('early_stop_p2_max_reward_mtp_loss', float),
    # (c) WM disturbance-estimator head (P87, default ON; RSSM backbone).
    'DREAMER_DISTURBANCE_HEAD':           ('disturbance_head',               _as_bool),
    'DREAMER_DISTURBANCE_LOSS_SCALE':     ('disturbance_loss_scale',         float),
    'DREAMER_DISTURBANCE_HEAD_STOP_GRAD': ('disturbance_head_stop_grad',     _as_bool),
    'DREAMER_DISTURBANCE_LOSS_REL_WEIGHT':('disturbance_loss_rel_weight',    float),
    'DREAMER_DISTURBANCE_LOSS_GATE_RECON':('disturbance_loss_gate_recon',    float),
    'DREAMER_DISTURBANCE_HEAD_HIDDEN':    ('disturbance_head_hidden',        int),
    'DREAMER_DISTURBANCE_HEAD_LAYERS':    ('disturbance_head_layers',        int),
    # Neural Kalman filter / disturbance observer (DOB, 2026-06-11; default off).
    'DREAMER_DOB_ENABLED':                ('dob_enabled',                    _as_bool),
    'DREAMER_DOB_REG_COEF':               ('dob_reg_coef',                   float),
    'DREAMER_DOB_DECAY_INIT':             ('dob_decay_init',                 float),
    'DREAMER_DOB_GAIN_INIT':              ('dob_gain_init',                  float),
    'DREAMER_DOB_GROUND_COEF':            ('dob_ground_coef',                float),
    # Staged clean->disturbance curriculum (2026-06-12; default off).  Requires
    # dob_enabled + phased mode.  See TrainConfig.curriculum_enabled.
    'DREAMER_CURRICULUM_ENABLED':         ('curriculum_enabled',             _as_bool),
    'DREAMER_CURRICULUM_STAGE2_DISTURBANCE_PROB': ('curriculum_stage2_disturbance_prob', float),
    'DREAMER_CURRICULUM_STAGE3_DISTURBANCE_PROB': ('curriculum_stage3_disturbance_prob', float),
    # DR RCA (2026-06-20): gate domain randomization OFF during the Stage-1/2 WM
    # + DOB identification (clean nominal-plant gain), back ON for the Stage-3
    # actor.  =0 to keep DR on throughout (the old, gain-biasing behaviour).
    'DREAMER_CURRICULUM_WM_ID_DR_OFF':    ('curriculum_wm_id_dr_off',        _as_bool),
    # PMPO prior-refresh cadence (unused for env-free real-sim REINFORCE).
    # p136 ``DREAMER_ACTOR_KL_COEF`` REMOVED (FALSIFIED; never wired in
    # ``_realsim_actor_critic_step``).
    'DREAMER_P3_PRIOR_REFRESH_ITERS':     ('p3_prior_refresh_iters',         int),
    # #2 (P88): multi-step latent overshooting (open-loop prior rollout
    # accuracy; RSSM backbone).  coef=0 = OFF (paper-faithful default).
    'DREAMER_WM_OVERSHOOT_COEF':          ('wm_overshoot_coef',              float),
    'DREAMER_WM_OVERSHOOT_LEN':           ('wm_overshoot_len',               int),
    'DREAMER_WM_OVERSHOOT_MAX_STARTS':    ('wm_overshoot_max_starts',        int),
    'DREAMER_WM_OVERSHOOT_TAIL_POWER':    ('wm_overshoot_tail_power',        float),
    'DREAMER_WM_OVERSHOOT_GATE_RECON':    ('wm_overshoot_gate_recon',        float),
    # Continuous gain+disturbance latent (2026-06-22).  cont_latent_enabled on +
    # dob_enabled off ⇒ the cont gain channel (C(1) gain-match) fixes the DV
    # bias + the cont disturbance channel is the DOB-free amortized-Kalman
    # estimator.  Dims auto-resolve from the plant; the rest are tuning knobs.
    'DREAMER_CONT_LATENT_ENABLED':        ('cont_latent_enabled',            _as_bool),
    'DREAMER_CONT_MIN_STD':               ('cont_min_std',                   float),
    'DREAMER_CONT_MAX_STD':               ('cont_max_std',                   float),
    'DREAMER_CONT_FREE_BITS':             ('cont_free_bits',                 float),
    'DREAMER_CONT_KL_SCALE':              ('cont_kl_scale',                  float),
    'DREAMER_CONT_GAIN_PERSIST_COEF':     ('cont_gain_persist_coef',         float),
    # Deterministic cont-disturbance roll in imagination (p140 RCA, default on).
    'DREAMER_CONT_DIST_DET_ROLL':         ('cont_dist_deterministic_roll',   _as_bool),
    # Deterministic cont-GAIN roll in imagination (p20 observer-bias RCA, default on).
    'DREAMER_CONT_GAIN_DET_ROLL':         ('cont_gain_deterministic_roll',   _as_bool),
    # C(2) disturbance-matching: supervise the cont disturbance channel toward
    # the true hidden load (auto-on when the cont disturbance channel is on).
    'DREAMER_DIST_MATCH_COEF':            ('dist_match_coef',                float),
    # C(1) gain-matching (the step-response asymptote DC supervisor).  coef/len
    # auto-resolve when the cont gain channel is on; these override.
    'DREAMER_GAIN_MATCH_COEF':            ('gain_match_coef',                float),
    'DREAMER_GAIN_MATCH_LEN':             ('gain_match_len',                 int),
    'DREAMER_GAIN_MATCH_MAX_STARTS':      ('gain_match_max_starts',          int),
    'DREAMER_GAIN_MATCH_STEP':            ('gain_match_step',                float),
    'DREAMER_GAIN_MATCH_HUBER_BETA':      ('gain_match_huber_beta',          float),
    # P43: per-element Huber β = |tgt_ij| (L1 sat ±1; not relative Huber).
    'DREAMER_GAIN_MATCH_HUBER_PER_INPUT': ('gain_match_huber_per_input',     _as_bool),
    # P44: held settle before gain-match FD (default -1 = off / P43;
    # 0 = auto control horizon; TM probe settle is 4×horizon).
    # P44 storm 2/2 REVERT of env-free auto=H.
    'DREAMER_GAIN_MATCH_SETTLE_LEN':      ('gain_match_settle_len',          int),
    # P45 EXIT PROMOTE: TM-protocol rest-IC teacher (env-free True).
    # Isolation loss stays 0.  ``=0`` reverts to PRBS-posterior FD.
    'DREAMER_GAIN_MATCH_REST_IC':         ('gain_match_rest_ic',             _as_bool),
    # Identity speed: CUDA-graph rest-IC T-loop (RSSM, sample=False, full
    # BPTT).  ``=0`` keeps the eager Python loop.
    'DREAMER_GAIN_MATCH_REST_IC_CUDA_GRAPH': ('gain_match_rest_ic_cuda_graph', _as_bool),
    'DREAMER_AUX_TBPTT_STEPS':            ('aux_tbptt_steps',                int),
    # Self-supervised WM gain supervisors (auto-on with the cont gain channel):
    # per-input isolation trajectory match + the steady-state DC-gain match (the
    # nonlinear / black-box unbiased-gain path; no identified value needed).
    'DREAMER_WM_INPUT_ISOLATION_COEF':    ('wm_input_isolation_coef',        float),
    'DREAMER_WM_INPUT_ISOLATION_LEN':     ('wm_input_isolation_len',         int),
    'DREAMER_WM_SS_MATCH_COEF':           ('wm_ss_match_coef',               float),
    'DREAMER_WM_SS_MATCH_SETTLE_VAR':     ('wm_ss_match_settle_var',         float),
    # Terminal fraction of K used as the SS window.  Isolation/ss-match
    # off env-free (P40); identity 0.34 when the teacher is on.
    'DREAMER_WM_SS_MATCH_WINDOW_FRAC':    ('wm_ss_match_window_frac',        float),
    'DREAMER_WM_ISOLATION_SETTLE_EPISODES': ('wm_isolation_settle_episodes', int),
    # Per isolated input (auto 24).  Seed loop emits n × n_mv + n × n_dv.
    # Equalize isolation |ΔCV| via Δu ∝ 1/|G| with scale floor 1.0
    # (default ON; P33 drowning, P38 match-at-g_min starved MV).
    'DREAMER_WM_ISOLATION_DCV_MATCH':     ('wm_isolation_dcv_match',     _as_bool),
    # Unitless floor on dcv isolation scale (default 1.0 = never shrink
    # strong-|G| below op-band).  0 = P38 match-at-g_min (FALSIFIED).
    'DREAMER_WM_ISOLATION_DCV_MIN_SCALE': ('wm_isolation_dcv_min_scale', float),
    # p10 RCA: treat the DV like a clean input — low-pass the noisy measured DV
    # into the WM (errors-in-variables gain fix) + make it input-only (no recon).
    'DREAMER_DV_LOWPASS_TAU':             ('dv_lowpass_tau',                 float),
    'DREAMER_WM_RECON_DV_WEIGHT':         ('wm_recon_dv_weight',             float),
    # p11 RCA: DCS-style output VELOCITY (rate) limit on the applied MV command —
    # a physical actuator/DCS CONSTRAINT (not a reward move-penalty) that kills
    # the degenerate full-range bang-bang the objective otherwise rewards.
    'DREAMER_MV_RATE_LIMIT':              ('mv_rate_limit',                  float),
    # p09 RCA: tight SEPARATE actor grad clip (the tanh-squashed REINFORCE actor
    # explodes in P3; the shared grad_clip=100 is too loose for it).
    'DREAMER_ACTOR_GRAD_CLIP':            ('actor_grad_clip',                float),
    # CLI-only extras that ``single_run`` previously dropped (only
    # ``python -m training.train`` honored them via ``_CLI_ONLY_ENV``).
    # Identity defaults.  Architecture / lookback are plant-derived in
    # ``single_run``; an explicit env now wins after that construction.
    'DREAMER_PMPO_ALPHA':                 ('pmpo_alpha',                     float),
    'DREAMER_PMPO_BETA':                  ('pmpo_beta',                      float),
    'DREAMER_MAE_PMAX':                   ('mae_p_max',                      float),
    'DREAMER_POLICY_TYPE':                ('policy_type',                    str),
    'DREAMER_POLICY_INIT_LOG_STD':        ('policy_init_log_std',            float),
    'DREAMER_ACTOR_LOSS':                 ('actor_loss_type',                str),  # train() refuses ≠reinforce (false A/B)
    'DREAMER_GRAD_CLIP':                  ('grad_clip',                      float),
    'DREAMER_BASELINE_SEED_EPS':          ('baseline_seed_episodes',         int),
    'DREAMER_BASELINE_SEED_STD':          ('baseline_seed_action_std',       float),
    'DREAMER_RANDOM_SEED_EPS':            ('random_seed_episodes',           int),
    'DREAMER_EXPLORATION_SEED_EPS':       ('exploration_seed_episodes',      int),
    'DREAMER_DV_PRBS_SEEDS':              ('dv_prbs_seed_episodes',          int),
    'DREAMER_DV_PRBS_OP_FRAC':            ('dv_prbs_op_frac',                float),
    'DREAMER_PRBS_SEED_N_STRATA':         ('prbs_seed_n_strata',             int),
    'DREAMER_D_MODEL':                    ('d_model',                        int),
    'DREAMER_N_LAYERS':                   ('n_layers',                       int),
    'DREAMER_N_HEADS':                    ('n_heads',                        int),
    'DREAMER_FF_MULT':                    ('ff_mult',                        int),
    'DREAMER_N_REGISTER':                 ('n_register',                     int),
    'DREAMER_Z_DIM':                      ('z_dim',                          int),
    'DREAMER_TOK_HIDDEN':                 ('tok_hidden',                     int),
    'DREAMER_HEAD_HIDDEN':                ('head_hidden',                    int),
    'DREAMER_K_MAX':                      ('k_max',                          int),
    'DREAMER_LOOKBACK':                   ('lookback',                       int),
    # Fix B: performance-aware entropy-collapse early-stop gate (only trip when
    # the policy is also degenerate: low imag_adv_action_corr).
    'DREAMER_EARLY_STOP_ENT_COLLAPSE_MIN_ADV_CORR': ('early_stop_entropy_collapse_min_adv_corr', float),
}


def apply_dreamer_env_overrides(cfg) -> Iterable[str]:
    """Apply the ``DREAMER_*`` env-var overrides onto ``cfg`` in-place.

    Each successful override:
      - sets the dataclass field via ``setattr``,
      - records the field in ``cfg._explicit_fields`` so ``training/train.py``'s
        auto-tune apply loop skips it (even when the injected value equals
        the dataclass default, e.g. paper σ_max=1.0 → log_std_max=0.0),
      - logs a single ``[env-override]`` line.

    Returns the iterable of field names that were overridden.

    NOTE: ``training/train.py``'s ``_cfg_from_env()`` (CLI) now also
    calls this function after architecture extras, so both entry-points
    share one ``DREAMER_*`` contract.
    """
    overridden = []
    for env_k, (field, cast) in ENV_OVERRIDES.items():
        val = os.environ.get(env_k, '').strip()
        if not val:
            continue
        try:
            setattr(cfg, field, cast(val))
            try:
                if not hasattr(cfg, '_explicit_fields'):
                    cfg._explicit_fields = set()  # type: ignore[attr-defined]
                cfg._explicit_fields.add(field)  # type: ignore[attr-defined]
            except Exception:
                pass
            overridden.append(field)
            print(f"[env-override] {field}={cast(val)} (from {env_k})",
                  flush=True)
        except Exception as e:
            print(f"[env-override] {env_k}={val!r} ignored: {e}", flush=True)
    return overridden


# ---------------------------------------------------------------------------
# 5. Pin eval modules at launch (P47/P48 late-import race)
# ---------------------------------------------------------------------------

_PINNED_EVAL_MODULES = False


def pin_eval_modules_at_launch() -> None:
    """Import validation/TM/diag modules once at train start.

    P47 ``ImportError: resolve_wm_tf_knobs``, P48
    ``ImportError: alloc_pinned_obs_host``, P49
    ``TypeError: get_authority_target_frac() got an unexpected
    keyword argument 'cfg'``: late ``from evaluation.validate import
    run_validation`` loaded HEAD ``validate.py`` against already-imported
    launch-time ``wm_transfer_matrix`` / ``dreamer_v4_rssm`` /
    ``training_disturbance``.  Pinning at launch binds the process to
    the launch-time eval stack so a GPU-occupied leftover commit cannot
    race the val pass.
    """
    global _PINNED_EVAL_MODULES
    if _PINNED_EVAL_MODULES:
        return
    import utils.training_disturbance  # noqa: F401
    import evaluation.diagnostics  # noqa: F401
    import evaluation.validate  # noqa: F401
    import evaluation.wm_disturbance_prediction  # noqa: F401
    import evaluation.wm_transfer_matrix  # noqa: F401
    import tools.wm_posterior_prior_probe  # noqa: F401
    import tools.wm_steady_state_diagnostic  # noqa: F401
    _PINNED_EVAL_MODULES = True
    print('[run] pinned eval modules at launch '
          '(P47/P48 late-import race)', flush=True)
