"""Adaptive single-run workflow for DreamerV4.

Usage::

    python -m workflow.run --simulation-dir simulation/test_sim

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

    python -m workflow.run --simulation-dir simulation/distillation
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
                             '<repo>/_runs/<sim>_<timestamp>/')
    parser.add_argument('--steps', type=int, default=200_000,
                        help='Total environment steps (default 200 000).')
    parser.add_argument('--model-size', choices=['S', 'M', 'L'], default='M',
                        help='Architecture preset from BO; paper default M.')
    parser.add_argument('--seed', type=int, default=0)
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
        out_dir = repo / '_runs' / f'{sim_name}_{ts}'
    out_dir.mkdir(parents=True, exist_ok=True)

    # Set the env vars our utilities expect — these MUST be set before
    # importing training.train (it reads them at config-build time).
    os.environ['CONTROL_SETUP_JSON'] = str(setup_path)
    os.environ['CONTROL_OBJECTIVE_JSON'] = str(obj_path)
    os.environ['SIMULATION_DIR'] = str(sim_dir)
    os.environ.setdefault('SEED', str(args.seed))

    sample_rate = int(os.environ.get('SIM_SAMPLE_RATE') or
                      _read_setup_sample_rate(setup_path, default=5))
    os.environ['SIM_SAMPLE_RATE'] = str(sample_rate)

    sys.path.insert(0, str(repo))

    # Plant identification (tau, dead_time, lookback).
    print(f'[run] simulation: {sim_dir}', flush=True)
    print('[run] phase 1: plant identification', flush=True)
    from workflow.bo_runner import initialize_from_plant, horizon_init
    plant = initialize_from_plant(out_dir / 'plant_id')
    with open(out_dir / 'plant_id.json', 'w') as f:
        json.dump(plant, f, indent=2)

    tau = float(plant['tau'])
    dead = float(plant['dead_time'])
    lookback = int(plant['lookback'])
    horizon = horizon_init(tau, dead, sample_rate)

    os.environ['IDENTIFIED_TAU_DOMINANT'] = f'{tau:g}'
    os.environ['IDENTIFIED_DEAD_TIME'] = f'{dead:g}'

    # Episode length: auto-derived (≈ 20 × (τ+θ), clamped 500–4000).
    from utils.auto_episode_length import derive_episode_length
    episode_length, ep_source = derive_episode_length()
    os.environ['SIM_EPISODE_LENGTH'] = str(episode_length)

    # Architecture preset (paper §C).
    from workflow.bo_runner import MODEL_SIZE_PRESETS
    arch = MODEL_SIZE_PRESETS[args.model_size]

    # Build TrainConfig with paper-faithful defaults.
    from training.train import TrainConfig, train as run_training
    cfg = TrainConfig(
        deter_dim=arch['deter_dim'],
        embed_dim=arch['embed_dim'],
        hidden_dim=arch['hidden_dim'],
        n_categoricals=arch['n_categoricals'],
        n_classes=arch['n_classes'],
        lookback=lookback,
        sample_rate=sample_rate,
        episode_length=episode_length,
        total_steps=int(args.steps),
        horizon=horizon,
        out_dir=str(out_dir),
    )

    plan = {
        'simulation_dir': str(sim_dir),
        'simulation_name': sim_name,
        'out_dir': str(out_dir),
        'sample_rate': sample_rate,
        'tau': tau,
        'dead_time': dead,
        'lookback': lookback,
        'horizon': horizon,
        'episode_length': episode_length,
        'episode_length_source': ep_source,
        'model_size': args.model_size,
        'seed': int(args.seed),
        'total_steps': int(args.steps),
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

    print('[run] done.', flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
