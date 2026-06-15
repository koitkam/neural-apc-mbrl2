"""Adaptive single-run workflow for DreamerV4.

Usage::

    python -m workflow.single_run --simulation-dir simulation/test_sim

Everything else is auto-derived (paper-faithful defaults + plant
identification).  The user only needs to pick the simulation.

What is auto-derived (in order):
  1. ``CONTROL_SETUP_JSON``  = ``<sim_dir>/control_setup.json``
  2. ``CONTROL_OBJECTIVE_JSON`` = ``<sim_dir>/control_objective.json``
  3. ``sample_rate`` from setup file (key ``sample_rate``) or env / default 5.
  4. ``tau``, ``dead_time``         <- ``dynamics_identifier``.
  5. ``lookback``                   <- ``lookback_identifier`` (centred on tau).
  6. ``episode_length``             <- ``auto_episode_length.derive_episode_length``.
  7. ``horizon``                    <- ``auto_episode_length.derive_horizon``
     (identified time-to-steady-state = dead_time + 4*tau, in agent steps;
     floored at the paper default 15, capped by DREAMER_HORIZON_MAX).
  8. ``model_size``                 <- paper default ``M`` (512/512/512, 32x32).
  9. ``seq_len = 64``, ``batch_size = 16``    (paper §C).
 10. ``total_steps``                <- ``--steps`` (default 1 000 000).

CLI flags allow targeted overrides for advanced usage; everything has a
sensible default so the typical command is:

    python -m workflow.single_run --simulation-dir simulation/distillation
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_setup_sample_rate(setup_path: Path, default: int = 5) -> int:
    try:
        with open(setup_path, 'r', encoding='utf-8') as f:
            d = json.load(f)
        sr = d.get('sample_rate')
        if sr is None:
            sr = (d.get('simulator') or {}).get('sample_rate')
        if sr is None:
            sr = (d.get('simulator', {}).get('kwargs') or {}).get('sample_rate')
        if sr is None:
            return int(default)
        return int(sr)
    except Exception:
        return int(default)


def _resolve_sim_dir(arg: str) -> Path:
    """Accept either an absolute path, a path relative to repo root, or just
    a sim name (``test_sim`` -> ``simulation/test_sim``)."""
    repo = Path(__file__).resolve().parent.parent
    p = Path(arg)
    if p.is_absolute() and p.exists():
        return p
    cand = repo / arg
    if cand.exists():
        return cand
    cand2 = repo / 'simulation' / arg
    if cand2.exists():
        return cand2
    raise FileNotFoundError(
        f"Cannot find simulation directory for '{arg}'. Tried: "
        f"{p}, {cand}, {cand2}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description='Run a full DreamerV4 training on a simulation. '
                    'Only --simulation-dir is required; everything else is '
                    'auto-derived from the plant.')
    parser.add_argument('--simulation-dir', '-s', required=True,
                        help='Path to a simulation directory (e.g. '
                             'simulation/test_sim, distillation, or absolute path).')
    parser.add_argument('--out-dir', '-o', default=None,
                        help='Output directory. Default: '
                             '<repo>/output/<sim>/run_<timestamp>/')
    parser.add_argument('--steps', type=int, default=0,
                        help='Total environment steps. Default 0 = plant-tied '
                             'auto-derivation (utils.phase_budget). Pass a '
                             'positive integer to override.')
    parser.add_argument('--model-size', choices=['S', 'M', 'L'], default=None,
                        help='Architecture preset. Default: auto-derived from '
                             'plant complexity (channels + multiscale + state_dim).')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--no-noise', action='store_true',
                        help='Skip plant-aware noise config (debug mode).')
    parser.add_argument('--no-validate', action='store_true',
                        help='Skip post-training validation.')
    parser.add_argument('--val-episodes', type=int, default=3)
    parser.add_argument('--val-seeds', type=int, default=3)
    parser.add_argument('--init-from-ckpt', type=str, default='',
                        help='Path to a previous run\'s checkpoint (e.g. '
                             'best.pt) to warm-start model weights from. '
                             'Optimizers/counters/phase tracking start fresh.')
    args = parser.parse_args()

    sim_dir = _resolve_sim_dir(args.simulation_dir)
    setup_path = sim_dir / 'control_setup.json'
    obj_path = sim_dir / 'control_objective.json'
    if not setup_path.exists():
        raise FileNotFoundError(f'Missing {setup_path}')
    if not obj_path.exists():
        raise FileNotFoundError(f'Missing {obj_path}')

    sim_name = sim_dir.name
    repo = Path(__file__).resolve().parent.parent
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        ts = time.strftime('%Y%m%d_%H%M%S')
        out_dir = repo / 'output' / sim_name / f'run_{ts}'
    out_dir.mkdir(parents=True, exist_ok=True)

    # Mirror stdout/stderr to <out_dir>/workflow.log so the run is self-contained.
    from workflow.bo_runner import _install_workflow_log
    _install_workflow_log(out_dir)
    print(f'[run] workflow log: {out_dir}/workflow.log', flush=True)

    # Set the env vars our utilities expect — these MUST be set before
    # importing training.train (it reads them at config-build time).
    os.environ['CONTROL_SETUP_JSON'] = str(setup_path)
    os.environ['CONTROL_OBJECTIVE_JSON'] = str(obj_path)
    os.environ['SIMULATION_DIR'] = str(sim_dir)
    os.environ.setdefault('SEED', str(args.seed))

    sys.path.insert(0, str(repo))

    # ── Phase 1a: Dynamics identification (no lookback yet — needs sr) ────
    print(f'[run] simulation: {sim_dir}', flush=True)
    print('[run] phase 1a: dynamics identification', flush=True)
    from workflow._plant_prepare import (
        identify_dynamics, build_noise_config, identify_lookback,
        apply_dreamer_env_overrides,
    )
    plant_dir = out_dir / 'plant_id'
    plant_info = identify_dynamics(plant_dir)
    dyn = plant_info['dynamics_raw']
    dyn_path = Path(plant_info['dynamics_report'])
    tau = plant_info['tau']
    dead = plant_info['dead_time']
    tau_fast = plant_info['tau_fast']
    dead_fast = plant_info['dead_time_fast']

    # ── Phase 1b: Plant-tied derivations (sample rate, model size, seq_len) ─
    # Sample rate from the *fastest* identified channel.  An explicit
    # SIM_SAMPLE_RATE in env or in the setup file overrides the derivation.
    from utils.sim_factory import create_sim, resolve_sim_metadata
    from utils.plant_init import derive_all
    sr_env = os.environ.get('SIM_SAMPLE_RATE', '').strip()
    sr_setup = _read_setup_sample_rate(setup_path, default=0)
    sr_override = int(sr_env) if sr_env else int(sr_setup)
    # Build a temporary sim instance so we can read mv/cv/dv counts + state_dim.
    tmp_sim = create_sim(episode_length=10, sample_rate=max(1, sr_override or 5))
    sim_meta = resolve_sim_metadata(tmp_sim)
    derived = derive_all(dyn, sim_meta, sample_rate_override=sr_override)
    sample_rate = derived['sample_rate']
    if args.model_size:
        model_size = args.model_size
        model_size_source = 'cli'
    else:
        model_size = derived['model_size']
        model_size_source = 'auto:complexity'
    os.environ['SIM_SAMPLE_RATE'] = str(sample_rate)
    os.environ['IDENTIFIED_TAU_DOMINANT'] = f'{tau:g}'
    os.environ['IDENTIFIED_DEAD_TIME'] = f'{dead:g}'

    # ── Phase 1b½: Plant-aware noise config (parity with workflow.bo_runner) ──
    # Builds OU process + measurement noise from identified plant dynamics
    # and exports SIM_NOISE_CONFIG_JSON so SimNoiseWrapper picks it up in
    # both training and validation subprocesses.  Skip with --no-noise for
    # debugging without stochastic confounders.
    if not args.no_noise:
        build_noise_config(out_dir, dynamics_raw=dyn,
                            sample_rate=sample_rate, log_prefix='[run]')
    else:
        print('[run] noise_config: DISABLED (--no-noise)', flush=True)

    # ── Phase 1c: Lookback identification (DIAGNOSTIC ONLY) ──────────
    # Historically ``identify_lookback`` set the runtime history-window
    # length independently from ``seq_len``.  As of 2026-05-24 the two
    # are unified: ``lookback := seq_len`` so the world-model's training
    # context length matches its deployment context length exactly.  The
    # identifier is still run for the report (and to flag plants where
    # the identified memory horizon exceeds ``seq_len`` — in which case
    # ``derive_seq_len`` was too aggressive a downsizer).
    print('[run] phase 1c: lookback identification (diagnostic)', flush=True)
    lb_info = identify_lookback(plant_dir, tau=tau, dead_time=dead,
                                 sample_rate=sample_rate, dynamics_raw=dyn,
                                 tau_fast=tau_fast, dead_time_fast=dead_fast)
    identified_lookback = int(lb_info['lookback'])
    lb_path = Path(lb_info['lookback_report'])
    # Unified history-window length: train and deployment use the same
    # transformer context length.
    lookback = int(derived['seq_len'])
    if identified_lookback > lookback:
        print(f'[run] WARNING identified_lookback={identified_lookback} > '
              f'seq_len={lookback}; ``derive_seq_len`` may be undersizing '
              f'the WM context (consider raising the multiplier).',
              flush=True)

    plant = {
        'tau': tau, 'dead_time': dead,
        'tau_fast': tau_fast, 'dead_time_fast': dead_fast,
        'lookback': lookback,
        'identified_lookback': identified_lookback,
        'dynamics_report': str(dyn_path),
        'lookback_report': str(lb_path),
    }
    with open(out_dir / 'plant_id.json', 'w') as f:
        json.dump(plant, f, indent=2)

    # Episode length & horizon from plant.
    from utils.auto_episode_length import derive_episode_length, derive_horizon
    from workflow.bo_runner import MODEL_SIZE_PRESETS
    episode_length, ep_source = derive_episode_length()
    os.environ['SIM_EPISODE_LENGTH'] = str(episode_length)
    # Imagination horizon: sized to the identified time-to-steady-state
    # (2% settling time = dead_time + 4*tau, in agent steps) so the
    # actor/critic credit the full settling response of the slowest loop —
    # including the consequence of riding vs not-riding a moved operator
    # limit over the whole transient.  Floored at the paper default 15 and
    # capped (DREAMER_HORIZON_MAX) to bound WM-rollout error; tune the settle
    # multiple via DREAMER_HORIZON_SETTLE_NTAU.  An explicit DREAMER_HORIZON
    # still hard-overrides downstream via the env-override layer.
    horizon, horizon_source = derive_horizon(
        tau=tau, dead_time=dead, sample_rate=sample_rate)
    print(f'[run] horizon={horizon} ({horizon_source}; '
          f'tau={tau:g} dead_time={dead:g} sr={sample_rate})', flush=True)
    seq_len = derived['seq_len']
    k_max = derived['k_max']
    arch = MODEL_SIZE_PRESETS[model_size]

    # Plant-tied step budget (parity with workflow.bo_runner).  CLI > 0 wins;
    # otherwise derive trial_steps from episode_length + complexity.

    # --- Iter-based, model-size-aware phase budget derivation ---
    from utils.phase_budget import derive_phase_budgets
    if int(args.steps) > 0:
        total_steps = int(args.steps)
        phase1_frac = None
        phase2_frac = None
        phase3_frac = None
        steps_source = 'cli'
    else:
        phase_budgets = derive_phase_budgets(
            episode_length=episode_length,
            complexity_score=derived['complexity_score'],
            model_size=model_size,
        )
        total_steps = int(phase_budgets['total_steps'])
        phase1_frac = phase_budgets['phase1_frac']
        phase2_frac = phase_budgets['phase2_frac']
        phase3_frac = phase_budgets['phase3_frac']
        steps_source = f"iter-based:{phase_budgets['source']}"
    print(f'[run] total_steps={total_steps} ({steps_source})', flush=True)
    if phase1_frac is not None:
        print(f"[run] phase_fracs: {phase1_frac:.2f} / {phase2_frac:.2f} / {phase3_frac:.2f}", flush=True)

    # Build TrainConfig — every value either plant-tied or paper-faithful default.
    from training.train import TrainConfig, train as run_training
    from tools.gpu_calibrate import pick_batch_size_for_plant
    # Empirical batch sizing: spend ~10 s on a real fwd+bwd probe of
    # world_model_loss on synthetic data with the actual derived
    # (model_size, seq_len, lookback, horizon, obs_dim, action_dim),
    # measure cuda.max_memory_allocated(), and snap bs to power-of-two
    # under target_util*gpu_total budget.  Replaces the prior formulaic
    # sizing which under-predicted by 1.27–2.74× across plants
    # (cross-sim measurements 2026-05-24).  On CPU or probe failure
    # falls back to paper_default=16.
    #
    # ``wm_overhead_factor`` reserves headroom the WM-only probe does NOT
    # measure: the actor/critic/optimizer state and the Phase-3 imagination
    # rollout (horizon-step latent unroll).  Tunable via DREAMER_WM_OVERHEAD;
    # default 1.30 (≈30% reserve) keeps the RSSM run inside the card in P3.
    try:
        _wm_overhead = float(os.environ.get('DREAMER_WM_OVERHEAD', '1.30'))
    except ValueError:
        _wm_overhead = 1.30
    _wm_overhead = max(1.0, _wm_overhead)
    bs_env = os.environ.get('OBJ_BATCH_SIZE', '').strip()
    if bs_env:
        try:
            batch_size = max(1, int(bs_env))
            bs_info = {'batch_size': batch_size, 'source': 'env_override'}
        except Exception:
            bs_info = pick_batch_size_for_plant(
                model_size=model_size, seq_len=seq_len, lookback=lookback,
                horizon=horizon, k_max=k_max, sample_rate=sample_rate,
                episode_length=int(episode_length),
                wm_overhead_factor=_wm_overhead)
            batch_size = int(bs_info['batch_size'])
    else:
        bs_info = pick_batch_size_for_plant(
            model_size=model_size, seq_len=seq_len, lookback=lookback,
            horizon=horizon, k_max=k_max, sample_rate=sample_rate,
            episode_length=int(episode_length),
            wm_overhead_factor=_wm_overhead)
        batch_size = int(bs_info['batch_size'])
    if bs_info.get('source', '').startswith('empirical'):
        print(f"[gpu-calib] empirical probe: "
              f"per_sample={bs_info.get('per_sample_mb_measured', 0):.1f} MB, "
              f"bs={batch_size}, "
              f"predicted_peak={bs_info.get('predicted_peak_gib', 0):.1f} GiB",
              flush=True)
    print(f"[run] batch_size={batch_size} ({bs_info['source']}; "
          f"per_batch≈{bs_info.get('per_batch_mb','?')}MB, "
          f"gpu={bs_info.get('gpu_total_gb',0):.1f}GB)", flush=True)
    cfg = TrainConfig(
        d_model=arch['d_model'],
        n_layers=arch['n_layers'],
        n_heads=arch['n_heads'],
        z_dim=arch['z_dim'],
        n_register=arch['n_register'],
        tok_hidden=arch['tok_hidden'],
        head_hidden=arch['head_hidden'],
        lookback=lookback,
        sample_rate=sample_rate,
        episode_length=episode_length,
        total_steps=total_steps,
        horizon=horizon,
        seq_len=seq_len,
        k_max=k_max,
        batch_size=batch_size,
        out_dir=str(out_dir),
        init_from_ckpt=str(args.init_from_ckpt or ''),
        # Plant-tie the WM multi-step supervision windows to the imagination
        # horizon (= 2% settling time).  Both the open-loop overshoot term and
        # the held-action steady-state term should span ~one full settling
        # response so the WM learns the asymptotic gain, not a truncated step.
        # p117 set these = horizon via env-override; promoted here 2026-06-14 so
        # they derive per-plant and can't be silently dropped from a launch.
        wm_overshoot_len=horizon,
        wm_held_rollout_len=horizon,
        **(dict(phase1_frac=phase1_frac, phase2_frac=phase2_frac, phase3_frac=phase3_frac) 
           if phase1_frac is not None else {}),
    )
    # Optional env-var overrides for A/B experiments.  These apply
    # *after* dataclass construction so auto-tune (which compares
    # against the dataclass default to decide whether to overwrite)
    # treats env-injected values as user overrides and skips them.
    # Note: ``training/train.py``'s ``_cfg_from_env()`` only runs when
    # train.py is invoked as a CLI; when ``single_run.py`` is the
    # entry-point we must perform the binding ourselves.  The whitelist
    # lives in ``workflow/_plant_prepare.ENV_OVERRIDES`` and is shared
    # with ``workflow/bo_runner.py`` so future knobs only need to be
    # added in one place.
    apply_dreamer_env_overrides(cfg)

    plan = {
        'simulation_dir': str(sim_dir),
        'simulation_name': sim_name,
        'out_dir': str(out_dir),
        'sample_rate': sample_rate,
        'sample_rate_source': derived['sample_rate_source'],
        # Deployment manifest (canonical names for the runtime API):
        # ``sample_rate_seconds`` is the seconds between successive
        # control decisions / history-window samples;
        # ``history_window_samples`` is the length of the history window
        # that must be supplied to the exported model.  These are aliased
        # to ``sample_rate`` and ``lookback`` (= ``seq_len``) which are
        # the internal training-config names.  Phase 1 unification
        # 2026-05-24: ``lookback == seq_len`` so train and deploy contexts
        # match exactly.
        'sample_rate_seconds': sample_rate,
        'history_window_samples': lookback,
        'tau': tau,
        'dead_time': dead,
        'tau_fast': tau_fast,
        'dead_time_fast': dead_fast,
        'lookback': lookback,
        'identified_lookback': identified_lookback,
        'horizon': horizon,
        'seq_len': seq_len,
        'k_max': k_max,
        'episode_length': episode_length,
        'episode_length_source': ep_source,
        'model_size': model_size,
        'model_size_source': model_size_source,
        'complexity_score': derived['complexity_score'],
        'complexity_inputs': derived['inputs'],
        'batch_size': batch_size,
        'batch_size_source': bs_info['source'],
        'seed': int(args.seed),
        'total_steps': total_steps,
        'total_steps_source': steps_source,
        'config': asdict(cfg),
    }
    with open(out_dir / 'run_plan.json', 'w') as f:
        json.dump(plan, f, indent=2)

    print('[run] auto-derived plan:', flush=True)
    print(json.dumps({k: v for k, v in plan.items() if k != 'config'},
                     indent=2), flush=True)

    print('[run] phase 2: training', flush=True)
    summary = run_training(cfg)
    with open(out_dir / 'run_summary.json', 'w') as f:
        json.dump({'plan': plan, 'summary': summary}, f, indent=2)

    # ── Phase 3: validation (parity with workflow.bo_runner final retrain) ──
    val_summary: Dict = {}
    if not args.no_validate:
        print('[run] phase 3: validation', flush=True)
        try:
            from evaluation.validate import run_validation
            # Validate best.pt (deterministic-best P3 ckpt) rather than final.pt
            # (post-cascade-degraded) — fair cross-run comparison. See P60 RCA.
            # Fall back to final.pt when no best.pt was ever written (P3
            # collapsed before any deterministic-eval improvement, e.g. the
            # bootstrap_cascade early-stop). Validating the degraded final
            # controller still produces the disturbance-rejection plots and a
            # comparable (if poor) score, which is far more useful than
            # crashing with FileNotFoundError and emitting no plots at all.
            val_ckpt = 'best.pt'
            if not (out_dir / 'best.pt').exists():
                val_ckpt = 'final.pt'
                print('[run] validation: best.pt not found '
                      '(no improving P3 ckpt — likely P3 collapse/early-stop); '
                      'falling back to final.pt (degraded controller)',
                      flush=True)
            val_summary = run_validation(controller_dir=out_dir,
                                          episodes=int(args.val_episodes),
                                          seeds=int(args.val_seeds),
                                          ckpt=val_ckpt)
            val_summary['validated_ckpt'] = val_ckpt
            with open(out_dir / 'validation_summary.json', 'w') as f:
                json.dump(val_summary, f, indent=2, default=str)
            print(f"[run] validation cum_raw_reward "
                  f"mean={val_summary.get('cum_raw_reward_mean', float('nan')):.2f} "
                  f"std={val_summary.get('cum_raw_reward_std', float('nan')):.2f} "
                  f"-> {out_dir}/validation/", flush=True)
        except Exception as e:
            import traceback
            print(f'[run] validation FAILED: {e!r}', flush=True)
            traceback.print_exc()
    else:
        print('[run] validation: DISABLED (--no-validate)', flush=True)

    print('[run] done.', flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
