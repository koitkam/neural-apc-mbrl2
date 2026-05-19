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
  7. ``horizon = ceil((θ + 3τ)/sr)`` (Dreamer-V4 paper §C plant initialization).
  8. ``model_size``                 <- paper default ``M`` (512/512/512, 32x32).
  9. ``seq_len = 64``, ``batch_size = 16``    (paper §C).
 10. ``total_steps``                <- ``--steps`` (default 200 000).

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
                        help='Total environment steps. 0 = plant-tied auto '
                             '(derive_step_budgets.trial_steps).')
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
    from utils.dynamics_identifier import identify_and_save_dynamics
    plant_dir = out_dir / 'plant_id'
    plant_dir.mkdir(parents=True, exist_ok=True)
    dyn_path = plant_dir / 'dynamics_identification.json'
    dyn = identify_and_save_dynamics(output_path=str(dyn_path))
    os.environ['DYNAMICS_IDENTIFICATION_JSON'] = str(dyn_path)
    tau = float(dyn.get('tau_dominant_identified', dyn.get('tau_dominant', 50.0)) or 50.0)
    dead = float(dyn.get('dead_time_identified', dyn.get('dead_time', 5.0)) or 5.0)
    tau_fast = float(dyn.get('tau_fastest_identified', tau) or tau)
    dead_fast = float(dyn.get('dead_time_fastest_identified', dead) or dead)

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
        try:
            from utils.sim_factory import create_sim as _create_sim
            from utils.noise_config import (
                build_noise_config_from_sim, save_noise_config,
            )
            _probe_sim = _create_sim(episode_length=10,
                                      sample_rate=max(1, sample_rate))
            bare = _probe_sim
            for _ in range(4):
                inner = getattr(bare, '_sim', None)
                if inner is None:
                    break
                bare = inner
            # lookback not yet identified at this point; pass 0 (only used
            # for record-keeping inside noise_config).
            noise_cfg = build_noise_config_from_sim(
                bare, dynamics_json=dyn,
                lookback_json={'identified_lookback': 0},
            )
            noise_cfg_path = out_dir / 'noise_config.json'
            save_noise_config(noise_cfg, str(noise_cfg_path))
            print(f"[run] noise_config: {noise_cfg_path} "
                  f"(OU={len(noise_cfg.get('ou_noise', []))} "
                  f"meas={len(noise_cfg.get('measurement_noise', []))})",
                  flush=True)
        except Exception as exc:
            print(f"[run] noise_config: SKIPPED ({exc!r}) — running with no "
                  "process / measurement noise", flush=True)
    else:
        print('[run] noise_config: DISABLED (--no-noise)', flush=True)

    # ── Phase 1c: Lookback identification (uses derived sample_rate) ──────
    print('[run] phase 1c: lookback identification', flush=True)
    from utils.lookback_identifier import identify_and_save_lookback
    lb_path = plant_dir / 'lookback_identification.json'
    seed = int(os.environ.get('SEED', '0'))
    min_lb = max(8, int(round(tau / max(1, sample_rate))))
    max_lb = max(min_lb + 8, int(round(4.0 * tau / max(1, sample_rate))))
    lb = identify_and_save_lookback(
        seed=seed, min_lb=min_lb, max_lb=max_lb,
        output_path=str(lb_path),
        tau_dominant=tau, dead_time=dead,
        tau_fastest=tau_fast, dead_time_fastest=dead_fast,
        per_pair_estimates=dyn.get('per_pair_estimates')
                           or dyn.get('pair_estimates') or [],
    )
    lookback = int(lb.get('identified_lookback', lb.get('lookback', max(min_lb, 32))))

    plant = {
        'tau': tau, 'dead_time': dead,
        'tau_fast': tau_fast, 'dead_time_fast': dead_fast,
        'lookback': lookback,
        'dynamics_report': str(dyn_path),
        'lookback_report': str(lb_path),
    }
    with open(out_dir / 'plant_id.json', 'w') as f:
        json.dump(plant, f, indent=2)

    # Episode length & horizon from plant.
    from utils.auto_episode_length import derive_episode_length
    from workflow.bo_runner import horizon_init, MODEL_SIZE_PRESETS
    episode_length, ep_source = derive_episode_length()
    os.environ['SIM_EPISODE_LENGTH'] = str(episode_length)
    horizon = horizon_init(tau, dead, sample_rate)
    seq_len = derived['seq_len']
    k_max = derived['k_max']
    arch = MODEL_SIZE_PRESETS[model_size]

    # Plant-tied step budget (parity with workflow.bo_runner).  CLI > 0 wins;
    # otherwise derive trial_steps from episode_length + complexity.
    from utils.plant_init import derive_step_budgets
    if int(args.steps) > 0:
        total_steps = int(args.steps)
        steps_source = 'cli'
    else:
        budgets = derive_step_budgets(
            episode_length=episode_length,
            complexity_score=derived['complexity_score'],
        )
        total_steps = int(budgets['trial_steps'])
        steps_source = f"plant-tied:{budgets['source']}"
    print(f'[run] total_steps={total_steps} ({steps_source})', flush=True)

    # Build TrainConfig — every value either plant-tied or paper-faithful default.
    from training.train import TrainConfig, train as run_training
    from utils.plant_init import derive_batch_size
    # Horizon-adaptive batch size (parity with workflow.bo_runner.run_trial).
    bs_env = os.environ.get('OBJ_BATCH_SIZE', '').strip()
    if bs_env:
        try:
            batch_size = max(1, int(bs_env))
            bs_info = {'batch_size': batch_size, 'source': 'env_override'}
        except Exception:
            bs_info = derive_batch_size(model_size, horizon=horizon,
                                         horizon_ref=horizon)
            batch_size = int(bs_info['batch_size'])
    else:
        bs_info = derive_batch_size(model_size, horizon=horizon,
                                     horizon_ref=horizon)
        batch_size = int(bs_info['batch_size'])
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
    )
    # Optional env-var overrides for A/B experiments.  These apply
    # *after* dataclass construction so auto-tune (which compares
    # against the dataclass default to decide whether to overwrite)
    # treats env-injected values as user overrides and skips them.
    # Note: ``training/train.py``'s ``_cfg_from_env()`` only runs when
    # train.py is invoked as a CLI; when ``single_run.py`` is the
    # entry-point we must perform the binding ourselves.
    _env_overrides = {
        'DREAMER_GAE_LAMBDA':       ('gae_lambda',                 float),
        'DREAMER_PHASE1_FRAC':      ('phase1_frac',                float),
        'DREAMER_PHASE2_FRAC':      ('phase2_frac',                float),
        'DREAMER_PHASE3_FRAC':      ('phase3_frac',                float),
        'DREAMER_LR_CRITIC':        ('lr_critic',                  float),
        'DREAMER_LR_ACTOR':         ('lr_actor',                   float),
        'DREAMER_P3_CRITIC_CV_MAX': ('p3_critic_stability_max_cv', float),
        'DREAMER_P3_COLLECT_EVERY': ('phase3_collect_every_iters', int),
        'DREAMER_BUFFER_CAP_STEPS': ('buffer_capacity_steps',      int),
        # 2026-05-19 paper-strip-back knobs (p28 A/B): expose the
        # remaining auto-tuned cfg fields so a fully paper-faithful
        # baseline can be launched purely via env vars, with no code
        # changes. Setting any of these pre-empts the corresponding
        # auto-tune branch via the dataclass-default sentinel.
        'DREAMER_POLICY_LOG_STD_MAX':       ('policy_log_std_max',           float),
        'DREAMER_POLICY_LOG_STD_MIN':       ('policy_log_std_min',           float),
        'DREAMER_PMPO_ENTROPY_COEF':        ('pmpo_entropy_coef',            float),
        'DREAMER_P3_CRITIC_WARMUP_ITERS':   ('p3_critic_warmup_iters',       int),
        'DREAMER_ENTROPY_DECAY_ON_SAT':     ('pmpo_entropy_decay_on_saturation',
                                             lambda v: bool(int(v))),
    }
    for _env_k, (_field, _cast) in _env_overrides.items():
        _val = os.environ.get(_env_k, '').strip()
        if _val:
            try:
                setattr(cfg, _field, _cast(_val))
                print(f"[env-override] {_field}={_cast(_val)} "
                      f"(from {_env_k})", flush=True)
            except Exception as _e:
                print(f"[env-override] {_env_k}={_val!r} ignored: {_e}",
                      flush=True)

    plan = {
        'simulation_dir': str(sim_dir),
        'simulation_name': sim_name,
        'out_dir': str(out_dir),
        'sample_rate': sample_rate,
        'sample_rate_source': derived['sample_rate_source'],
        'tau': tau,
        'dead_time': dead,
        'tau_fast': tau_fast,
        'dead_time_fast': dead_fast,
        'lookback': lookback,
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
            val_summary = run_validation(controller_dir=out_dir,
                                          episodes=int(args.val_episodes),
                                          seeds=int(args.val_seeds))
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
