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
from typing import Dict, Iterable, Optional


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

    tau = float(dyn.get('tau_dominant_identified',
                         dyn.get('tau_dominant', 50.0)) or 50.0)
    dead = float(dyn.get('dead_time_identified',
                          dyn.get('dead_time', 5.0)) or 5.0)
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


def _as_opt_float(s: str):
    """Parse an env-var string into Optional[float].  Empty / ``none`` /
    ``null`` -> None (use the TrainConfig default); otherwise float."""
    v = str(s).strip().lower()
    if v in ('', 'none', 'null', 'na', 'default'):
        return None
    return float(s)


ENV_OVERRIDES: Dict[str, tuple] = {
    'DREAMER_GAE_LAMBDA':         ('gae_lambda',                 float),
    'DREAMER_PHASE1_FRAC':        ('phase1_frac',                float),
    'DREAMER_PHASE2_FRAC':        ('phase2_frac',                float),
    'DREAMER_PHASE3_FRAC':        ('phase3_frac',                float),
    'DREAMER_LR_CRITIC':          ('lr_critic',                  float),
    'DREAMER_LR_ACTOR':           ('lr_actor',                   float),
    # 2026-05-24 (P48 RCA): structural γ/H mismatch fix.  See
    # auto_tune_seed_buffer in training/train.py for adaptive formula.
    'DREAMER_GAMMA':              ('gamma',                      float),
    'DREAMER_TARGET_CRITIC_TAU':  ('target_critic_tau',          float),
    'DREAMER_P3_COLLECT_EVERY':   ('phase3_collect_every_iters', int),
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
    'DREAMER_HORIZON':            ('horizon',                    int),
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
    # DV→decoder+heads feedforward (2026-06-19, p129): route the measured DV
    # around the categorical bottleneck (where the DV→CV gain dies) directly
    # into the decoder + heads.  Default ON; =0 reverts to transition-only DV.
    'DREAMER_DV_FEEDFORWARD':             ('dv_feedforward',             _as_bool),
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
    # Default 24 in TrainConfig.
    'DREAMER_CONST_ACTION_SEEDS': ('constant_action_seed_episodes', int),
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
    'DREAMER_P2_GATE_REWARD_MTP_MAX': ('p2_gate_reward_mtp_max',  float),
    'DREAMER_P2_GATE_RECENT_ITERS': ('p2_gate_recent_iters',      int),
        'DREAMER_P2_GATE_MAX_EXTENSION': ('p2_gate_max_extension',    float),  # default 0.5
    # 2026-05-27 (P57 RCA): minimum fraction of total_steps reserved
    # for P3 (actor-critic) regardless of P1/P2 extensions.  Default
    # 0.20 in TrainConfig.  Set to 0.0 to disable (legacy behaviour).
    'DREAMER_PHASE3_MIN_FRAC':    ('phase3_min_frac',            float),
    # 2026-05-27 (P59 refactor): σ_max / σ_min auto-tune formula inputs
    # — previously read directly via os.environ.get inside the auto-tune
    # body, now promoted to TrainConfig fields with whitelist entries
    # so they appear in run_plan.json → config.  Legacy
    # ``DREAMER_SIGMA_MAX_OVER_SEED`` / ``DREAMER_SIGMA_MAX_CAP`` /
    # ``SIGMA_MAX_FLOOR`` / ``SIGMA_MIN_RATIO_OF_MAX`` env-vars still
    # honoured for back-compat inside auto_tune_seed_buffer; the
    # canonical path is the cfg field bound here.
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
    # P79 (2026-06-02): return-scale ABSOLUTE cap — dimensionless (return
    # units).  Arrests the critic-pessimism cascade runaway once the spread
    # is implausibly large WITHOUT throttling legitimate early growth (the
    # P63 growth-rate clamp regressed for that reason).  Set 0.0 to recover
    # the paper-faithful uncapped EMA.
    'DREAMER_RETURN_SCALE_ABS_CAP':       ('return_scale_abs_cap',           float),
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
    # Fix 2b (2026-06-19, p129): disturbance-aware advantage baseline (subtract
    # the per-horizon batch-mean advantage = the uncontrollable common-mode).
    'DREAMER_ACTOR_DISTURBANCE_BASELINE': ('actor_disturbance_baseline',     _as_bool),
    'DREAMER_CRITIC_REPLAY_ANCHOR_COEF':  ('critic_replay_anchor_coef',      float),
    # (B) P85 (2026-06-04): long-horizon critic-anchor grounding.  The
    # replay anchor's own λ (decoupled from the cascade-sensitive
    # imagination ``gae_lambda``) — ~0.97–1.0 turns it into a near-MC
    # return-to-go over the real context so a constraint-riding limit cycle
    # whose period exceeds the myopic ~10-step credit horizon becomes
    # visible in the critic target.  ``_LONG`` optionally raises the anchor
    # weight so the long target overcomes the myopic imagined critic loss.
    # Both None (unset) ⇒ exact legacy behaviour.  Sim-agnostic.
    'DREAMER_CRITIC_ANCHOR_LAMBDA':       ('critic_anchor_lambda',           _as_opt_float),
    'DREAMER_CRITIC_ANCHOR_COEF_LONG':    ('critic_anchor_coef_long',        _as_opt_float),
    # Real-return (Monte-Carlo) critic grounding (Option #1 / TD-MPC, 2026-06-09):
    # a PURE discounted return-to-go over the real buffer (no value bootstrap)
    # added to the critic target so it is pinned to realised economics.  Pair
    # the coef with a reduced DREAMER_CRITIC_IMAG_LOSS_COEF so the MC target
    # dominates.  Coef 0 (default) = off; _TAIL_BOOTSTRAP adds a single γ^N tail.
    'DREAMER_CRITIC_MC_GROUNDING_COEF':   ('critic_mc_grounding_coef',       float),
    'DREAMER_CRITIC_MC_TAIL_BOOTSTRAP':   ('critic_mc_tail_bootstrap',       _as_bool),
    'DREAMER_MV_HARD_CLAMP':              ('mv_hard_clamp',                  _as_bool),
    'DREAMER_MV_ACTION_FULL_RANGE':       ('mv_action_map_full_range',       _as_bool),
    'DREAMER_RUNTIME_SETPOINT_VARIATION': ('runtime_setpoint_variation',     _as_bool),
    # ---- World-model backbone (P68, 2026-05-30) ----
    # ``rssm`` (default) vs ``sf_transformer``; RSSM categorical-latent
    # sizes and KL-balance knobs.  See TrainConfig for paper rationale.
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
    # TSSM (transformer-SSM) backbone dims (world_model_type='tssm').
    'DREAMER_TSSM_D_MODEL':               ('tssm_d_model',                   int),
    'DREAMER_TSSM_N_LAYERS':              ('tssm_n_layers',                  int),
    'DREAMER_TSSM_N_HEADS':               ('tssm_n_heads',                   int),
    'DREAMER_TSSM_MAX_SEQ_LEN':           ('tssm_max_seq_len',               int),
    # P70 (2026-05-30): RSSM imagination steady-state fix (opt-in).
    # latent-mode = roll imagined prior with categorical MODE (kills the
    # per-step jitter that biases the reward head).  Sim-agnostic.
    'DREAMER_RSSM_IMAG_LATENT_MODE':      ('rssm_imag_latent_mode',          _as_bool),
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
    # adaptive; the DREAMER_EXPERT_* move-law knobs are read inside apc_expert.
    'DREAMER_EXPERT_TYPE':                ('expert_type',                    str),
    'DREAMER_EXPERT_BC_SCALE':            ('expert_bc_scale',                float),
    'DREAMER_EXPERT_SEED_EPISODES':       ('expert_seed_episodes',           int),
    'DREAMER_EXPERT_ACTION_JITTER':       ('expert_action_jitter',           float),
    'DREAMER_EXPERT_KEEP_SCHEDULE':       ('expert_keep_schedule',           _as_bool),
    'DREAMER_EXPERT_USE_SS_SAMPLES':      ('expert_use_ss_samples',          _as_bool),
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
    # (P89) noise curriculum + clean steady-state seeds (default ON).  The
    # process-noise ramp + per-event hidden-disturbance schedule knobs are read
    # straight from os.environ (not cfg fields): DREAMER_PROCESS_NOISE_AMP_RAMP,
    # DREAMER_HIDDEN_DIST_{MODE,SETTLE_NTAU,MAX_EVENTS,P_ISOLATED,P_REVERT,
    # SHAPE_WEIGHTS}.  These two booleans gate the cfg-level behaviour.
    'DREAMER_CLEAN_STEADY_SEEDS':         ('clean_steady_seeds',             _as_bool),
    'DREAMER_PROCESS_NOISE_CURRICULUM':   ('process_noise_curriculum',       _as_bool),
    'DREAMER_WM_FREEZE_AFTER_P1':         ('wm_freeze_after_p1',             _as_bool),
    # WM-fix workstream (2026-06-09): all default-OFF (identity to p106).
    'DREAMER_WM_FREEZE_AFTER_ITERS':      ('wm_freeze_after_iters',          int),
    'DREAMER_WM_RECON_CV_WEIGHT':         ('wm_recon_cv_weight',             float),
    'DREAMER_BC_TRACK_EXPERT_EVERY':      ('bc_track_expert_every',          int),
    'DREAMER_EXPERT_BC_P3_FLOOR':         ('expert_bc_p3_floor',             float),
    'DREAMER_P3_CRITIC_WARMUP_ITERS':     ('p3_critic_warmup_iters',         int),
    'DREAMER_WM_TRUNK_STOPGRAD_IN_P2':    ('wm_trunk_stopgrad_in_p2',        _as_bool),
    'DREAMER_TRAIN_MODE':                 ('train_mode',                     str),
    'DREAMER_ACTOR_SOURCE':               ('actor_train_source',             str),
    'DREAMER_JOINT_PRIOR_REFRESH_ITERS':  ('joint_prior_refresh_iters',      int),
    # Early-stop knobs (mirror train.py _cfg_from_env names) so single_run/bo
    # runs can relax/disable the stops for diagnostic runs (e.g. let a run
    # continue PAST the entropy-collapse stop to observe the cascade trajectory).
    'DREAMER_EARLY_STOP':                 ('early_stop_enable',              _as_bool),
    'DREAMER_ES_P3_PATIENCE':             ('early_stop_p3_patience_iters',   int),
    'DREAMER_ES_P3_MIN_IMPROVEMENT':      ('early_stop_p3_min_improvement',  float),
    'DREAMER_ES_ENT_FRAC':                ('early_stop_entropy_collapse_frac',          float),
    'DREAMER_ES_ENT_PATIENCE':            ('early_stop_entropy_collapse_patience_iters', int),
    'DREAMER_ES_ENT_WINDOW':              ('early_stop_entropy_collapse_window_iters',   int),
    'DREAMER_ES_ENT_MIN_BELOW':           ('early_stop_entropy_collapse_min_frac_below', float),
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
    # Staged clean->disturbance curriculum (2026-06-12; default off).  Requires
    # dob_enabled + phased mode.  See TrainConfig.curriculum_enabled.
    'DREAMER_CURRICULUM_ENABLED':         ('curriculum_enabled',             _as_bool),
    'DREAMER_CURRICULUM_STAGE2_DISTURBANCE_PROB': ('curriculum_stage2_disturbance_prob', float),
    'DREAMER_CURRICULUM_STAGE3_DISTURBANCE_PROB': ('curriculum_stage3_disturbance_prob', float),
    # DR RCA (2026-06-20): gate domain randomization OFF during the Stage-1/2 WM
    # + DOB identification (clean nominal-plant gain), back ON for the Stage-3
    # actor.  =0 to keep DR on throughout (the old, gain-biasing behaviour).
    'DREAMER_CURRICULUM_WM_ID_DR_OFF':    ('curriculum_wm_id_dr_off',        _as_bool),
    # Stage A (p135): actor-imagination loop-gain randomization spread (replaces
    # the Stage-3 real-data DR).  Float; 0 disables; unset -> auto (inherit the
    # sim's DR output-gain frac).  See TrainConfig.actor_imag_gain_random_frac.
    'DREAMER_ACTOR_IMAG_GAIN_RANDOM_FRAC':('actor_imag_gain_random_frac',    float),
    # p136: actor KL trust region (damps policy hunting) + phased-P3 prior
    # refresh cadence.  actor_kl_coef=0 disables (legacy); see TrainConfig.
    'DREAMER_ACTOR_KL_COEF':              ('actor_kl_coef',                  float),
    'DREAMER_P3_PRIOR_REFRESH_ITERS':     ('p3_prior_refresh_iters',         int),
    # #1 (P88): critic real-grounding rebalance (down-weight imagined critic CE
    # so the real-return anchor dominates -> breaks bootstrap self-dominance).
    'DREAMER_CRITIC_IMAG_LOSS_COEF':      ('critic_imag_loss_coef',          float),
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
    # Deterministic cont-disturbance roll in imagination (p140 RCA, default on)
    # + the static DV feedthrough skip (p132, default OFF — superseded).
    'DREAMER_CONT_DIST_DET_ROLL':         ('cont_dist_deterministic_roll',   _as_bool),
    'DREAMER_DV_STATIC_SKIP':             ('dv_static_skip',                 _as_bool),
    # C(2) disturbance-matching: supervise the cont disturbance channel toward
    # the true hidden load (auto-on when the cont disturbance channel is on).
    'DREAMER_DIST_MATCH_COEF':            ('dist_match_coef',                float),
    # C(1) gain-matching (the step-response asymptote DC supervisor).  coef/len
    # auto-resolve when the cont gain channel is on; these override.
    'DREAMER_GAIN_MATCH_COEF':            ('gain_match_coef',                float),
    'DREAMER_GAIN_MATCH_LEN':             ('gain_match_len',                 int),
    'DREAMER_GAIN_MATCH_MAX_STARTS':      ('gain_match_max_starts',          int),
    'DREAMER_GAIN_MATCH_STEP':            ('gain_match_step',                float),
    # Fix B: performance-aware entropy-collapse early-stop gate (only trip when
    # the policy is also degenerate: low imag_adv_action_corr).
    'DREAMER_EARLY_STOP_ENT_COLLAPSE_MIN_ADV_CORR': ('early_stop_entropy_collapse_min_adv_corr', float),
    'DREAMER_EXPERT_BC_P3_FLOOR':         ('expert_bc_p3_floor',             float),
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

    NOTE: ``training/train.py``'s ``_cfg_from_env()`` only runs when
    ``train.py`` is invoked as a CLI; when ``single_run.py`` or
    ``bo_runner.py`` is the entry-point we must perform the binding
    ourselves.
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
