"""Bayesian Optimization driver for DreamerV4.

Three axes (the only knobs we tune):

  - lookback : grid centred on plant-identified lookback.
  - model_size_preset : {S, M, L} — coordinated triples
                       ``(deter_dim, hidden_dim, embed_dim, n_classes)``.
  - horizon : 5-point band ``{0.5, 0.75, 1.0, 1.5, 2.0} * H_init`` with
              ``H_init = ceil((dead_time + 3 * tau) / sample_rate)``.

All three axes are seeded from plant identification (`dynamics_identifier` +
`lookback_identifier`).  No per-knob NN dimension search; no auto-tuning
band-aids.

For each trial we:
  1. set the 3 parameters,
  2. train ``trial_total_steps`` with `training/train.train()`,
  3. score by the final EMA return on the policy buffer.

After the study, we re-train the best config for the full
``final_total_steps`` and export a single integrated ONNX artifact.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import optuna

from training.train import TrainConfig, train as run_training
from inference.export_onnx import export_dreamer_v4_onnx
from models.dreamer_v4 import DreamerV4, DreamerV4Config, RSSMConfig
import torch


# ---------------------------------------------------------------------------
# Tee stdout/stderr to <out_dir>/workflow.log
# ---------------------------------------------------------------------------

class _Tee:
    """Duplicate writes to multiple streams.  Used to mirror stdout/stderr
    into ``workflow.log`` so the run directory is self-contained."""
    def __init__(self, *streams):
        self._streams = streams
    def write(self, s):
        for st in self._streams:
            try:
                st.write(s)
            except Exception:
                pass
    def flush(self):
        for st in self._streams:
            try:
                st.flush()
            except Exception:
                pass
    def isatty(self):
        return False


def _install_workflow_log(out_dir: Path) -> None:
    """Mirror stdout + stderr into ``<out_dir>/workflow.log``.

    Idempotent: subsequent calls overwrite the previous Tee target so the
    log always tracks the current run directory.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / 'workflow.log'
    f = open(log_path, 'a', buffering=1)  # line-buffered
    sys.stdout = _Tee(sys.__stdout__, f)
    sys.stderr = _Tee(sys.__stderr__, f)


# ---------------------------------------------------------------------------
# Model size presets (paper §C scales)
# ---------------------------------------------------------------------------

MODEL_SIZE_PRESETS: Dict[str, Dict[str, int]] = {
    'S': {'deter_dim': 256, 'embed_dim': 256, 'hidden_dim': 256,
          'n_categoricals': 16, 'n_classes': 16},
    'M': {'deter_dim': 512, 'embed_dim': 512, 'hidden_dim': 512,
          'n_categoricals': 32, 'n_classes': 32},
    'L': {'deter_dim': 1024, 'embed_dim': 1024, 'hidden_dim': 1024,
          'n_categoricals': 32, 'n_classes': 64},
}

HORIZON_BAND = (0.5, 0.75, 1.0, 1.5, 2.0)


# ---------------------------------------------------------------------------
# Plant initialization
# ---------------------------------------------------------------------------

def initialize_from_plant(out_dir: Path) -> Dict:
    """Run dynamics + lookback identification, save reports.

    Returns a dict with at least ``tau``, ``dead_time``, ``lookback``.
    """
    from utils.dynamics_identifier import identify_and_save_dynamics
    from utils.lookback_identifier import identify_and_save_lookback

    dyn_path = out_dir / 'dynamics_identification.json'
    lb_path = out_dir / 'lookback_identification.json'
    out_dir.mkdir(parents=True, exist_ok=True)

    dyn = identify_and_save_dynamics(output_path=str(dyn_path))
    tau = float(dyn.get('tau_dominant_identified', dyn.get('tau_dominant', 50.0)) or 50.0)
    dead = float(dyn.get('dead_time_identified', dyn.get('dead_time', 5.0)) or 5.0)
    tau_fast = dyn.get('tau_fastest_identified', tau)
    dt_fast = dyn.get('dead_time_fastest_identified', dead)
    pair_est = dyn.get('per_pair_estimates') or dyn.get('pair_estimates') or []

    seed = int(os.environ.get('SEED', '0'))
    # Default scan range: ~lookback expected to be within [round(tau/sr), round(4*tau/sr)].
    sr = int(os.environ.get('SIM_SAMPLE_RATE', '5'))
    min_lb = max(8, int(round(tau / max(1, sr))))
    max_lb = max(min_lb + 8, int(round(4.0 * tau / max(1, sr))))

    lb = identify_and_save_lookback(
        seed=seed, min_lb=min_lb, max_lb=max_lb,
        output_path=str(lb_path),
        tau_dominant=tau, dead_time=dead,
        tau_fastest=tau_fast, dead_time_fastest=dt_fast,
        per_pair_estimates=pair_est,
    )
    lookback = int(lb.get('identified_lookback', lb.get('lookback', max(min_lb, 32))))

    # Export identified dynamics so downstream helpers (auto_episode_length,
    # objective_runtime, etc.) see the same plant numbers as workflow/run.py.
    os.environ['IDENTIFIED_TAU_DOMINANT'] = f'{tau:g}'
    os.environ['IDENTIFIED_DEAD_TIME'] = f'{dead:g}'
    os.environ['IDENTIFIED_LOOKBACK_SEED'] = str(lookback)

    return {
        'tau': tau,
        'dead_time': dead,
        'tau_fast': float(tau_fast) if tau_fast else float(tau),
        'dead_time_fast': float(dt_fast) if dt_fast else float(dead),
        'lookback': lookback,
        'dynamics_report': str(dyn_path),
        'lookback_report': str(lb_path),
        'dynamics_raw': dyn,
    }


def horizon_init(tau: float, dead_time: float, sample_rate: int) -> int:
    """Plant-derived horizon initial value: ceil((θ + 3τ) / sr)."""
    return max(3, int(math.ceil((dead_time + 3.0 * tau) / max(1, sample_rate))))


def lookback_grid(plant_lookback: int) -> List[int]:
    """3-point grid around the plant-identified lookback."""
    a = max(8, int(round(0.75 * plant_lookback)))
    b = max(a + 1, int(plant_lookback))
    c = max(b + 1, int(round(1.5 * plant_lookback)))
    return sorted({a, b, c})


# ---------------------------------------------------------------------------
# Trial runner
# ---------------------------------------------------------------------------

def make_trial_config(base: TrainConfig, *, lookback: int,
                      model_size: str, horizon: int,
                      total_steps: int, out_dir: Path,
                      batch_size: int | None = None) -> TrainConfig:
    preset = MODEL_SIZE_PRESETS[model_size]
    overrides: Dict[str, object] = {
        'lookback': int(lookback),
        'horizon': int(horizon),
        'total_steps': int(total_steps),
        'out_dir': str(out_dir),
        **preset,
    }
    if batch_size is not None:
        overrides['batch_size'] = int(batch_size)
    cfg = TrainConfig(**{**asdict(base), **overrides})
    return cfg


def run_trial(trial: optuna.Trial, base: TrainConfig, plant: Dict,
              study_dir: Path, trial_steps: int) -> float:
    from utils.plant_init import derive_batch_size
    lookback = trial.suggest_categorical('lookback', lookback_grid(plant['lookback']))
    model_size = trial.suggest_categorical('model_size', list(MODEL_SIZE_PRESETS))
    horizon_mult = trial.suggest_categorical('horizon_mult', HORIZON_BAND)
    H_init = horizon_init(plant['tau'], plant['dead_time'], base.sample_rate)
    horizon = max(3, int(round(horizon_mult * H_init)))

    # Per-trial batch size: re-derive from the trial's actual model_size +
    # horizon so OOM-prone (L, large horizon) trials use a smaller batch.
    bs_env = os.environ.get('OBJ_BATCH_SIZE', '').strip()
    if bs_env:
        try:
            bs = max(1, int(bs_env))
            bs_info = {'batch_size': bs, 'source': 'env_override'}
        except Exception:
            bs_info = derive_batch_size(model_size, horizon=horizon, horizon_ref=H_init)
            bs = int(bs_info['batch_size'])
    else:
        bs_info = derive_batch_size(model_size, horizon=horizon, horizon_ref=H_init)
        bs = int(bs_info['batch_size'])

    trial_dir = study_dir / f'trial_{trial.number:04d}'
    trial_dir.mkdir(parents=True, exist_ok=True)
    print(f"[trial {trial.number}] lookback={lookback} model={model_size} "
          f"horizon={horizon} batch={bs} ({bs_info['source']}; "
          f"per_batch≈{bs_info.get('per_batch_mb',0):.0f}MB)", flush=True)
    cfg = make_trial_config(base, lookback=lookback, model_size=model_size,
                            horizon=horizon, total_steps=trial_steps,
                            out_dir=trial_dir, batch_size=bs)

    # Pruning hook: report the running EMA return after each log iter so the
    # MedianPruner can stop visibly-bad trials early.
    def _on_iter(it: int, steps: int, ema: float) -> bool:
        try:
            trial.report(float(ema), step=int(steps))
        except Exception:
            pass
        return bool(trial.should_prune())

    try:
        summary = run_training(cfg, on_iter_end=_on_iter)
    except optuna.TrialPruned:
        with open(trial_dir / 'trial_summary.json', 'w') as f:
            json.dump({'params': trial.params, 'horizon_concrete': horizon,
                       'H_init': H_init, 'pruned': True}, f, indent=2)
        raise

    score = summary.get('final_ema_return')
    if score is None or not np.isfinite(score):
        score = float('-inf')
    with open(trial_dir / 'trial_summary.json', 'w') as f:
        json.dump({'params': trial.params, 'horizon_concrete': horizon,
                   'H_init': H_init, 'summary': summary,
                   'score': float(score)}, f, indent=2)
    return float(score)


# ---------------------------------------------------------------------------
# Final retrain + ONNX export
# ---------------------------------------------------------------------------

def train_final_and_export(base: TrainConfig, plant: Dict, best_params: Dict,
                           out_dir: Path, total_steps: int) -> Dict:
    from utils.plant_init import derive_batch_size
    lookback = int(best_params['lookback'])
    model_size = str(best_params['model_size'])
    H_init = horizon_init(plant['tau'], plant['dead_time'], base.sample_rate)
    horizon = max(3, int(round(float(best_params['horizon_mult']) * H_init)))

    bs_env = os.environ.get('OBJ_BATCH_SIZE', '').strip()
    if bs_env:
        try:
            bs = max(1, int(bs_env))
        except Exception:
            bs = int(derive_batch_size(model_size, horizon=horizon,
                                        horizon_ref=H_init)['batch_size'])
    else:
        bs = int(derive_batch_size(model_size, horizon=horizon,
                                    horizon_ref=H_init)['batch_size'])
    print(f'[final] model={model_size} lookback={lookback} horizon={horizon} '
          f'batch={bs}', flush=True)

    final_dir = out_dir / 'final'
    final_dir.mkdir(parents=True, exist_ok=True)
    cfg = make_trial_config(base, lookback=lookback, model_size=model_size,
                            horizon=horizon, total_steps=total_steps,
                            out_dir=final_dir, batch_size=bs)
    summary = run_training(cfg)

    # Reload model from final.pt and export ONNX.
    ckpt = torch.load(final_dir / 'final.pt', map_location='cpu', weights_only=False)
    cfg_loaded = TrainConfig(**{k: v for k, v in ckpt['cfg'].items()
                                 if k in {f for f in TrainConfig.__dataclass_fields__}})
    rssm_cfg = RSSMConfig(
        obs_dim=cfg_loaded.obs_dim, action_dim=cfg_loaded.action_dim,
        lookback=cfg_loaded.lookback,
        deter_dim=cfg_loaded.deter_dim, embed_dim=cfg_loaded.embed_dim,
        hidden_dim=cfg_loaded.hidden_dim,
        n_categoricals=cfg_loaded.n_categoricals,
        n_classes=cfg_loaded.n_classes,
        free_nats=cfg_loaded.free_nats,
    )
    model_cfg = DreamerV4Config(rssm=rssm_cfg,
                                n_action_bins=cfg_loaded.n_action_bins,
                                actor_hidden=cfg_loaded.hidden_dim,
                                critic_hidden=cfg_loaded.hidden_dim)
    model = DreamerV4(model_cfg)
    model.load_state_dict(ckpt['model'])
    onnx_path = out_dir / 'dreamer_v4.onnx'
    export_dreamer_v4_onnx(model, onnx_path)

    # Auto-validation on held-out seeds (paper-faithful "test set" gate).
    val_summary: Dict = {}
    try:
        from evaluation.validate import run_validation
        print('[final] phase 4: validation on held-out seeds', flush=True)
        val_summary = run_validation(controller_dir=final_dir,
                                     episodes=3, seeds=3)
        print(f"[final] validation cum_raw_reward "
              f"mean={val_summary.get('cum_raw_reward_mean', float('nan')):.2f} "
              f"std={val_summary.get('cum_raw_reward_std', float('nan')):.2f} "
              f"-> {final_dir}/validation/", flush=True)
    except Exception as e:
        print(f'[final] validation skipped: {e!r}', flush=True)

    return {
        'final_summary': summary,
        'best_params': best_params,
        'horizon_concrete': horizon,
        'H_init': H_init,
        'onnx_path': str(onnx_path),
        'final_ckpt': str(final_dir / 'final.pt'),
        'validation': val_summary,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_bo(out_dir: str | Path, n_trials: int = 8,
           trial_steps: int = 0, final_steps: int = 0,
           study_name: str = 'dreamer_v4_bo') -> Dict:
    """Run BO.  ``trial_steps`` / ``final_steps`` ≤ 0 → plant-tied auto."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _install_workflow_log(out_dir)
    print(f'[runner] workflow log: {out_dir}/workflow.log', flush=True)

    base = TrainConfig()
    # Sample-rate env override is supported here for sims that hard-code their
    # scan rate.  Episode length is auto-derived from identification (or from
    # SIM_EPISODE_LENGTH env via derive_episode_length).
    sr_env = os.environ.get('SIM_SAMPLE_RATE', '').strip()

    print('[BO] Phase 1: plant identification', flush=True)
    plant = initialize_from_plant(out_dir / 'plant_id')

    # Plant-tied derivations (sample_rate from fastest dynamics, model_size
    # from complexity, seq_len ≥ settling time).  Env-supplied values take
    # precedence so this stays simulator-agnostic.
    from utils.sim_factory import create_sim, resolve_sim_metadata
    from utils.plant_init import derive_all, derive_step_budgets, derive_batch_size
    sr_override = int(sr_env) if sr_env else 0
    tmp_sim = create_sim(episode_length=10,
                         sample_rate=max(1, sr_override or base.sample_rate))
    sim_meta = resolve_sim_metadata(tmp_sim)
    derived = derive_all(plant.get('dynamics_raw') or {}, sim_meta,
                         sample_rate_override=sr_override)
    sr = derived['sample_rate']
    base.sample_rate = sr
    base.seq_len = derived['seq_len']

    # Episode length: env override > auto-derived from settling time > paper.
    from utils.auto_episode_length import derive_episode_length
    ep_len, ep_source = derive_episode_length()
    base.episode_length = int(ep_len)
    os.environ['SIM_EPISODE_LENGTH'] = str(base.episode_length)
    print(f"[BO] episode_length={base.episode_length} ({ep_source})", flush=True)

    derived_model_size = derived['model_size']
    print(f"[BO] derived: sample_rate={sr} ({derived['sample_rate_source']}) "
          f"seq_len={base.seq_len} model_size_seed={derived_model_size} "
          f"complexity={derived['complexity_score']:.2f}", flush=True)

    # Adaptive batch size SEED (per-trial batch is re-derived in run_trial
    # from the trial's actual model_size + horizon).  This block only
    # records the seed-config batch in run_plan.json.
    bs_env = os.environ.get('OBJ_BATCH_SIZE', '').strip()
    if bs_env:
        try:
            base.batch_size = max(1, int(bs_env))
            bs_info = {'batch_size': base.batch_size, 'source': 'env_override'}
        except Exception:
            bs_info = derive_batch_size(derived_model_size)
            base.batch_size = int(bs_info['batch_size'])
    else:
        bs_info = derive_batch_size(derived_model_size)
        base.batch_size = int(bs_info['batch_size'])
    print(f"[BO] batch_size_seed={base.batch_size} ({bs_info['source']}; "
          f"per_batch≈{bs_info.get('per_batch_mb',0):.0f}MB, "
          f"gpu={bs_info.get('gpu_total_gb',0):.1f}GB; "
          f"per-trial batch re-derived from trial config)", flush=True)

    # Plant-tied step budgets (trial_steps / final_steps).  CLI / caller
    # values > 0 take precedence so power users can still override.
    budgets = derive_step_budgets(
        episode_length=base.episode_length,
        complexity_score=derived['complexity_score'],
    )
    if int(trial_steps) > 0:
        budgets['trial_steps'] = int(trial_steps)
        budgets['source'] = 'override'
    if int(final_steps) > 0:
        budgets['final_steps'] = int(final_steps)
        budgets['source'] = 'override'
    trial_steps = int(budgets['trial_steps'])
    final_steps = int(budgets['final_steps'])
    print(f"[BO] step budgets: trial={trial_steps} final={final_steps} "
          f"({budgets['source']}; trial_eps_target={budgets['trial_episodes_target']}, "
          f"final_eps_target={budgets['final_episodes_target']})", flush=True)

    with open(out_dir / 'plant_id.json', 'w') as f:
        json.dump({**plant, 'derived': derived, 'step_budgets': budgets,
                   'batch_size': bs_info},
                  f, indent=2, default=str)

    # Workflow-level run_plan.json — single source of truth for everything
    # downstream (trials, validation, ONNX export) reads from.
    plan = {
        'mode': 'bo',
        'simulation_dir': os.environ.get('SIMULATION_DIR'),
        'simulation_name': Path(os.environ.get('SIMULATION_DIR', 'unknown')).name,
        'out_dir': str(out_dir),
        'sample_rate': sr,
        'sample_rate_source': derived['sample_rate_source'],
        'tau': plant['tau'], 'dead_time': plant['dead_time'],
        'tau_fast': plant['tau_fast'], 'dead_time_fast': plant['dead_time_fast'],
        'lookback': plant['lookback'],
        'episode_length': base.episode_length,
        'episode_length_source': ep_source,
        'seq_len': base.seq_len,
        'model_size_seed': derived_model_size,
        'model_size_source': 'auto:complexity',
        'complexity_score': derived['complexity_score'],
        'complexity_inputs': derived['inputs'],
        'batch_size': base.batch_size,
        'batch_size_source': bs_info['source'],
        'trial_steps': trial_steps, 'final_steps': final_steps,
        'step_budget_source': budgets['source'],
        'n_trials': int(n_trials),
        'seed': int(os.environ.get('SEED', '0')),
    }
    with open(out_dir / 'run_plan.json', 'w') as f:
        json.dump(plan, f, indent=2, default=str)
    print(f"[BO] plant: tau={plant['tau']:.2f}  dead_time={plant['dead_time']:.2f}  "
          f"lookback={plant['lookback']}  H_init="
          f"{horizon_init(plant['tau'], plant['dead_time'], sr)}", flush=True)

    print('[BO] Phase 2: Optuna search', flush=True)
    study = optuna.create_study(
        study_name=study_name, direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=int(os.environ.get('SEED', '0'))),
        # Prune trials whose intermediate EMA return is below the median of
        # completed trials at the same step, after the first 3 trials have
        # finished and ≥ 5 reports have come in.
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=3, n_warmup_steps=5, interval_steps=1),
    )
    study_dir = out_dir / 'trials'
    study_dir.mkdir(parents=True, exist_ok=True)
    # Seed the first trial at the plant-derived configuration (lookback from
    # lookback_identifier, model_size from complexity, horizon at the centre
    # of the plant band).  Optuna explores the rest from there.
    seed_lookback = min(lookback_grid(plant['lookback']),
                        key=lambda v: abs(v - plant['lookback']))
    study.enqueue_trial({
        'lookback': seed_lookback,
        'model_size': derived_model_size,
        'horizon_mult': 1.0,
    })
    study.optimize(lambda t: run_trial(t, base, plant, study_dir, trial_steps),
                   n_trials=n_trials, show_progress_bar=False)

    best = study.best_trial
    print(f"[BO] best trial #{best.number}: score={best.value:.2f}  params={best.params}",
          flush=True)
    with open(out_dir / 'study_summary.json', 'w') as f:
        json.dump({'best_trial': best.number, 'best_score': float(best.value),
                   'best_params': best.params,
                   'all_trials': [{'number': t.number, 'value': t.value,
                                   'params': t.params, 'state': str(t.state)}
                                  for t in study.trials]}, f, indent=2)

    print('[BO] Phase 3: retrain best config and export ONNX', flush=True)
    final = train_final_and_export(base, plant, best.params, out_dir, final_steps)
    with open(out_dir / 'workflow_summary.json', 'w') as f:
        json.dump({'plant': plant, 'study_best': best.params,
                   'study_best_score': float(best.value),
                   'final': final}, f, indent=2)
    print(f"[BO] done. ONNX -> {final['onnx_path']}", flush=True)
    return final


if __name__ == '__main__':
    import argparse
    import time as _time
    p = argparse.ArgumentParser(
        description='Full DreamerV4 workflow: plant identification → '
                    'Optuna BO over (lookback, model_size, horizon) → '
                    'final retrain → ONNX export.')
    p.add_argument('--simulation-dir', '-s', default=None,
                   help='Simulation directory (e.g. simulation/test_sim). '
                        'Required unless --out is given (legacy mode where '
                        'CONTROL_SETUP_JSON / SIMULATION_DIR are set externally).')
    p.add_argument('--out', '-o', default=None,
                   help='Workflow output directory. Default: '
                        '<repo>/output/<sim>/bo_<timestamp>/')
    p.add_argument('--n_trials', type=int, default=8)
    p.add_argument('--trial_steps', type=int, default=0,
                   help='per-trial training steps (0 = plant-tied auto)')
    p.add_argument('--final_steps', type=int, default=0,
                   help='final retrain steps (0 = plant-tied auto)')
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()

    repo = Path(__file__).resolve().parent.parent
    if args.simulation_dir is not None:
        from workflow.run import _resolve_sim_dir
        sim_dir = _resolve_sim_dir(args.simulation_dir)
        setup_path = sim_dir / 'control_setup.json'
        obj_path = sim_dir / 'control_objective.json'
        if not setup_path.exists():
            raise FileNotFoundError(f'Missing {setup_path}')
        if not obj_path.exists():
            raise FileNotFoundError(f'Missing {obj_path}')
        os.environ['CONTROL_SETUP_JSON'] = str(setup_path)
        os.environ['CONTROL_OBJECTIVE_JSON'] = str(obj_path)
        os.environ['SIMULATION_DIR'] = str(sim_dir)
        os.environ.setdefault('SEED', str(args.seed))
        sim_name = sim_dir.name
    else:
        sim_name = Path(os.environ.get('SIMULATION_DIR', 'unknown')).name

    if args.out:
        out_dir = args.out
    else:
        ts = _time.strftime('%Y%m%d_%H%M%S')
        out_dir = str(repo / 'output' / sim_name / f'bo_{ts}')

    print(f'[runner] sim={sim_name}  out_dir={out_dir}', flush=True)
    run_bo(out_dir, n_trials=args.n_trials,
           trial_steps=args.trial_steps, final_steps=args.final_steps)
