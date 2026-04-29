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
                      total_steps: int, out_dir: Path) -> TrainConfig:
    preset = MODEL_SIZE_PRESETS[model_size]
    cfg = TrainConfig(**{**asdict(base),
                          'lookback': int(lookback),
                          'horizon': int(horizon),
                          'total_steps': int(total_steps),
                          'out_dir': str(out_dir),
                          **preset})
    return cfg


def run_trial(trial: optuna.Trial, base: TrainConfig, plant: Dict,
              study_dir: Path, trial_steps: int) -> float:
    lookback = trial.suggest_categorical('lookback', lookback_grid(plant['lookback']))
    model_size = trial.suggest_categorical('model_size', list(MODEL_SIZE_PRESETS))
    horizon_mult = trial.suggest_categorical('horizon_mult', HORIZON_BAND)
    H_init = horizon_init(plant['tau'], plant['dead_time'], base.sample_rate)
    horizon = max(3, int(round(horizon_mult * H_init)))

    trial_dir = study_dir / f'trial_{trial.number:04d}'
    trial_dir.mkdir(parents=True, exist_ok=True)
    cfg = make_trial_config(base, lookback=lookback, model_size=model_size,
                            horizon=horizon, total_steps=trial_steps,
                            out_dir=trial_dir)
    summary = run_training(cfg)
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
    lookback = int(best_params['lookback'])
    model_size = str(best_params['model_size'])
    H_init = horizon_init(plant['tau'], plant['dead_time'], base.sample_rate)
    horizon = max(3, int(round(float(best_params['horizon_mult']) * H_init)))

    final_dir = out_dir / 'final'
    final_dir.mkdir(parents=True, exist_ok=True)
    cfg = make_trial_config(base, lookback=lookback, model_size=model_size,
                            horizon=horizon, total_steps=total_steps,
                            out_dir=final_dir)
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

    return {
        'final_summary': summary,
        'best_params': best_params,
        'horizon_concrete': horizon,
        'H_init': H_init,
        'onnx_path': str(onnx_path),
        'final_ckpt': str(final_dir / 'final.pt'),
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_bo(out_dir: str | Path, n_trials: int = 8,
           trial_steps: int = 50_000, final_steps: int = 200_000,
           study_name: str = 'dreamer_v4_bo') -> Dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base = TrainConfig()
    # Allow env-driven overrides for sample_rate / episode_length so the BO
    # itself stays simulator-agnostic.
    sr_env = os.environ.get('SIM_SAMPLE_RATE', '').strip()
    ep_env = os.environ.get('SIM_EPISODE_LENGTH', '').strip()

    print('[BO] Phase 1: plant identification', flush=True)
    plant = initialize_from_plant(out_dir / 'plant_id')

    # Plant-tied derivations (sample_rate from fastest dynamics, model_size
    # from complexity, seq_len ≥ settling time).  Env-supplied values take
    # precedence so this stays simulator-agnostic.
    from utils.sim_factory import create_sim, resolve_sim_metadata
    from utils.plant_init import derive_all
    sr_override = int(sr_env) if sr_env else 0
    tmp_sim = create_sim(episode_length=10,
                         sample_rate=max(1, sr_override or base.sample_rate))
    sim_meta = resolve_sim_metadata(tmp_sim)
    derived = derive_all(plant.get('dynamics_raw') or {}, sim_meta,
                         sample_rate_override=sr_override)
    sr = derived['sample_rate']
    base.sample_rate = sr
    base.seq_len = derived['seq_len']
    if ep_env:
        base.episode_length = int(ep_env)
    derived_model_size = derived['model_size']
    print(f"[BO] derived: sample_rate={sr} ({derived['sample_rate_source']}) "
          f"seq_len={base.seq_len} model_size_seed={derived_model_size} "
          f"complexity={derived['complexity_score']:.2f}", flush=True)

    with open(out_dir / 'plant_id.json', 'w') as f:
        json.dump({**plant, 'derived': derived}, f, indent=2, default=str)
    print(f"[BO] plant: tau={plant['tau']:.2f}  dead_time={plant['dead_time']:.2f}  "
          f"lookback={plant['lookback']}  H_init="
          f"{horizon_init(plant['tau'], plant['dead_time'], sr)}", flush=True)

    print('[BO] Phase 2: Optuna search', flush=True)
    study = optuna.create_study(
        study_name=study_name, direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=int(os.environ.get('SEED', '0'))),
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
    p = argparse.ArgumentParser()
    p.add_argument('--out', required=True, help='workflow output directory')
    p.add_argument('--n_trials', type=int, default=8)
    p.add_argument('--trial_steps', type=int, default=50_000)
    p.add_argument('--final_steps', type=int, default=200_000)
    args = p.parse_args()
    run_bo(args.out, n_trials=args.n_trials,
           trial_steps=args.trial_steps, final_steps=args.final_steps)
