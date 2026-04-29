# neural-apc-dreamerV4

Paper-faithful DreamerV4 controller for Advanced Process Control (APC).

Reference: Hafner et al. (2024), "Mastering Diverse Domains through World
Models" (DreamerV4), [arXiv:2407.04693](https://arxiv.org/abs/2407.04693).

## Goals

- Single algorithm (DreamerV4) — no TD3/PPO/SAC/Transformer scaffolding.
- Stay close to the paper.  Add adaptive knobs only when they remain a strict
  superset of the paper recipe (paper defaults as floors / minimums).
- Simulator-agnostic via small, focused Bayesian Optimization on three axes
  only: `lookback`, `model_size`, `horizon` (initialized from plant ID).
- One ONNX artifact per workflow: a single integrated graph
  `(prev_h, prev_z, prev_action, obs_window) → (next_h, next_z, action)`.
  No separate observer model — the RSSM *is* the observer.

## Layout

```
models/dreamer_v4.py        # RSSM (encoder + GRU + categorical posterior) + actor + critic
training/train.py           # single trainer (RSSM + actor-critic with paper §C global return scale)
utils/
  plant_init.py             # sample_rate / model_size / seq_len / batch_size / step-budget derivations
  runtime_setpoints.py      # packed per-CV (lo, hi, target, active) augmentation
  sim_factory.py            # generic simulator loader
  objective_config.py       # objective spec parsing
  objective_runtime.py      # 5-term reward computation
  agent_utils.py            # action_to_control + reward wrappers
  dynamics_identifier.py    # τ / θ identification
  lookback_identifier.py    # window length identification
  auto_episode_length.py    # episode length from identified dynamics
  sim_noise.py              # measurement / actuator noise
  training_disturbance.py   # disturbance scheduler
  noise_config.py           # noise spec parsing
  state_normalization.py    # MV / CV normalization helpers
  time_sampling.py          # sample-rate utilities
simulation/                 # plant simulators (one folder per simulator)
inference/export_onnx.py    # single-graph deterministic ONNX export
workflow/
  runner.py                 # FULL PIPELINE: plant ID → BO → final retrain → ONNX
  run.py                    # single training run (no BO; paper-faithful seed config)
evaluation/validate.py      # held-out deterministic validation + timeseries plots
output/<sim>/<run_id>/      # all artefacts for one workflow run (default location)
```

## Two entry points (consistent plant-derivation pipeline)

Both share the same plant-derivation chain:

1. `dynamics_identifier`  → τ_dom, τ_fast, θ_dom, θ_fast.
2. `lookback_identifier`  → lookback (centred on plant value).
3. `plant_init.derive_all`  → sample_rate (from τ_fast / θ_fast),
   model_size_seed (from complexity score), seq_len (≥ settling time).
4. `auto_episode_length.derive_episode_length`  → episode length
   (env override > 20 × (τ + θ) > paper default).
5. `plant_init.derive_batch_size`  → batch_size from GPU mem headroom
   (target ~50 % util, paper batch=16 as floor).
6. `plant_init.derive_step_budgets` (BO only)  → trial / final step budgets
   from episode length × complexity factor.

| | `workflow/run.py` | `workflow/runner.py` |
|---|---|---|
| Purpose | one fixed-config training run | full BO + final retrain + ONNX |
| Plant ID | yes (full chain above) | yes (full chain above, identical) |
| Adaptive batch size | yes | yes |
| Adaptive episode length | yes | yes |
| Reward calibration | yes (in `train()`) | yes (in `train()`) |
| Output | `output/<sim>/run_<ts>/` | `output/<sim>/bo_<ts>/` |
| `workflow.log` (Tee) | yes | yes |
| Optuna BO | no | yes (3 axes, MedianPruner) |
| ONNX export | no | yes |

## Quick start — full workflow (recommended)

```bash
source ../neural-apc-dreamerV4-env/bin/activate
python -m workflow.runner --simulation-dir simulation/test_sim
```

That's the whole command.  Everything is auto-derived:

```
output/test_sim/bo_<timestamp>/
├── workflow.log                 # full stdout/stderr (Tee)
├── plant_id.json                # plant + derived block + step budgets + batch size
├── plant_id/
│   ├── dynamics_identification.json
│   └── lookback_identification.json
├── run_plan.json                # canonical config snapshot
├── trials/
│   ├── trial_0000/
│   │   ├── train_log.jsonl      # per-iter metrics (incl. per-phase timing)
│   │   ├── reward_calibration.json
│   │   └── final.pt             # checkpoint at end of trial budget
│   └── trial_0001/ …
├── study_summary.json           # Optuna study results
├── final/
│   ├── train_log.jsonl
│   └── final.pt                 # best-config retrained checkpoint
├── dreamer_v4.onnx              # exported ONNX
└── workflow_summary.json
```

### Common flags

```bash
python -m workflow.runner --simulation-dir simulation/test_sim \
  --n_trials 8 \
  --trial_steps 0 \    # 0 = plant-tied auto
  --final_steps 0 \    # 0 = plant-tied auto
  --seed 0 \
  --out output/my_custom_dir   # optional
```

### Environment overrides (all optional)

| Var | Effect |
|---|---|
| `OBJ_REWARD_SCALE` | `auto` (default) / `off` / `<float>` — disable or force reward scale |
| `OBJ_BATCH_SIZE` | force a batch size (else auto from GPU mem) |
| `SIM_EPISODE_LENGTH` | force episode length (else auto from settling time) |
| `SIM_SAMPLE_RATE` | force sample rate (else auto from `τ_fast / 10`) |
| `SEED` | RNG seed (default 0) |

## Single training run (no BO)

```bash
python -m workflow.run --simulation-dir simulation/test_sim --steps 500000
```

Same plant-derivation chain; output goes to `output/<sim>/run_<ts>/`.
Useful for quick smoke tests or debugging without the BO overhead.

## Validation

```bash
python -m evaluation.validate \
  --controller-dir output/test_sim/bo_<ts>/final \
  --episodes 3 --seeds 3
```

- Held-out RNG seeds (`10_000 + s`) so the disturbance schedule is genuinely
  out-of-distribution from training (which uses seeds `0..N`).
- Reuses the training `APCEnv` + the calibrated `reward_scale` from
  `reward_calibration.json` if present.
- Outputs:
  - `<dir>/validation/seed_<s>/ep_<i>.png` — CV / MV / reward / cumulative
    reward timeseries with bound bands and disturbance markers.
  - `<dir>/validation/summary.png` — cross-seed boxplots of cum reward,
    CV violation, MV violation.
  - `<dir>/validation/validation_summary.json` — aggregate metrics.

## Augmented observation layout

`runtime_setpoints.py` produces per-step augmentation channels:

- per MV channel: `[lo, hi]`                                  → `2 * n_mv`
- per CV channel: `[lo, hi, target, target_active_flag]`      → `4 * n_cv`

Total `aug_obs_dim = 2 * n_mv + 4 * n_cv`.  `target_active_flag ∈ {0, 1}`
explicitly tells the policy whether to track a target on that CV; this avoids
overloading `0.0` as both "centred target" and "disabled".

## BO axes

Only three axes; each initialized from plant identification:

- `lookback` — 3-point grid around the `lookback_identifier` value.
- `model_size` — `{S, M, L}` presets for coordinated
  `(deter_dim, hidden_dim, embed_dim, n_classes)` triples (paper §C).
- `horizon` — 5-point band `{0.5, 0.75, 1.0, 1.5, 2.0} × H_init`,
  where `H_init = ⌈(θ + 3τ) / sample_rate⌉`.

The first trial is enqueued at the plant-derived seed config.  An Optuna
`MedianPruner` (3 startup trials, 5 warmup steps) cuts visibly bad trials
early via the `on_iter_end` callback in the trainer.

## Adaptive knobs (paper-superset)

| Knob | Floor (paper) | Auto rule | Override |
|---|---|---|---|
| `batch_size` | 16 | nearest power of two filling ~50 % of GPU memory | `OBJ_BATCH_SIZE` |
| `reward_scale` | 1.0 | `target_std=1.0 / measured_raw_std`, clamped ≥ 1.0 | `OBJ_REWARD_SCALE` |
| `episode_length` | 600 | `20 × (τ + θ)` clamped to `[500, 4000]` | `SIM_EPISODE_LENGTH` |
| `sample_rate` | 5 | `min(τ_fast / 10, θ_fast / 2)` | `SIM_SAMPLE_RATE` |
| `seq_len` | 64 | `max(64, ⌈(3τ + θ) / sr⌉)` | — |
| `model_size` | M | `S/M/L` from complexity score | — |
| `trial_steps` | 50 000 | `40 eps × max(1, complexity / 4) × ep_len`, clamped | `--trial_steps` |
| `final_steps` | 200 000 | `10 × trial_steps`, clamped | `--final_steps` |

## Setup

```bash
python3 -m venv ../neural-apc-dreamerV4-env
source ../neural-apc-dreamerV4-env/bin/activate
pip install -r requirements.txt
```

## Status

- DreamerV4 paper-faithful trainer + RSSM + categorical actor + twohot critic.
- Adaptive batch / reward / episode-length / step-budget knobs (paper-superset).
- Single-arg workflow entry (`workflow/runner.py`).
- Validation harness with timeseries plots.
- ONNX export of integrated `(rssm + actor)` graph.

See `docs/` for design notes.
