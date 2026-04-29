# neural-apc-dreamerV4

Paper-faithful DreamerV4 controller for Advanced Process Control (APC).

Reference: Hafner et al. (2024), "Mastering Diverse Domains through World
Models" (DreamerV4), [arXiv:2407.04693](https://arxiv.org/abs/2407.04693).

## Goals

- Single algorithm (DreamerV4) — no TD3/PPO/SAC/Transformer scaffolding.
- Stay close to the paper. Do not add tuning knobs to compensate for issues
  we have not fully understood.
- Simulator-agnostic via small, focused Bayesian Optimization on three axes
  only: `lookback`, `model_size`, `horizon` (initialized from plant ID).
- One ONNX artifact per training run: a single integrated graph
  `(prev_h, prev_z, prev_action, obs_window) → (next_h, next_z, action)`.
  No separate observer model.

## Layout

```
models/dreamer_v4.py        # RSSM (encoder+GRU+categorical posterior) + actor + critic
training/train.py           # single trainer
utils/                      # plant-side modules (carried from neural-apc-pytorch)
  runtime_setpoints.py      # rewritten: packed per-CV (lo,hi,target,active) aug-obs
  sim_factory.py            # generic simulator loader
  objective_config.py       # objective spec parsing
  objective_runtime.py      # 5-term reward computation
  agent_utils.py            # action_to_control + reward wrappers
  dynamics_identifier.py    # τ/θ identification
  lookback_identifier.py    # window length identification
  auto_episode_length.py    # episode length from dynamics
  sim_noise.py              # measurement / actuator noise
  training_disturbance.py   # disturbance scheduler
  noise_config.py           # noise spec parsing
  resource_gate.py          # CPU/GPU resource gating
  state_normalization.py    # MV/CV normalization helpers
  time_sampling.py          # sample-rate utilities
simulation/                 # plant simulators
  test_sim/                 # toy tau/dead-time plant for smoke testing
inference/export_onnx.py    # single-graph deterministic ONNX export
workflow/bo_runner.py       # Optuna 3-axis BO
evaluation/                 # validation harness
tools/                      # diagnostics
docs/                       # design notes
```

## Augmented observation layout

`runtime_setpoints.py` produces per-step augmentation channels:

- per MV channel: `[lo, hi]`                                          → `2 * n_mv`
- per CV channel: `[lo, hi, target, target_active_flag]`              → `4 * n_cv`

Total `aug_obs_dim = 2 * n_mv + 4 * n_cv`. The `target_active_flag ∈ {0, 1}`
explicitly tells the policy whether to track a target on that CV; this avoids
overloading `0.0` as both "centred target" and "disabled".

## BO axes

Only three axes; each initialized from plant identification:

- `lookback`: small grid around the value from `lookback_identifier`.
- `model_size`: discrete `{S, M, L}` presets for coordinated
  `(deter_dim, hidden_dim, embed_dim, n_classes)` triples (paper §C).
- `horizon`: small band around `H_init = ⌈(θ + 3τ) / sample_rate⌉`.

No per-knob dimension search; no auto-tuning band-aids.

## Setup

```bash
python3 -m venv ../neural-apc-dreamerV4-env
source ../neural-apc-dreamerV4-env/bin/activate
pip install -r requirements.txt
```

## Status

Scaffold in progress. See `docs/` for design notes and todo.
