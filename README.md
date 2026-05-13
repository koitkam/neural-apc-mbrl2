# neural-apc-dreamerV4

Paper-faithful **Dreamer 4** controller for Advanced Process Control (APC).

Reference: Hafner, Yan, Lillicrap (2025), "Training Agents Inside of Scalable
World Models" (Dreamer 4), [arXiv:2509.24527](https://arxiv.org/abs/2509.24527).

## Goals

- Single algorithm (Dreamer 4) — no TD3/PPO/SAC scaffolding.
- Stay close to the paper. Add adaptive knobs only when they remain a strict
  superset of the paper recipe (paper defaults as floors / minimums).
- Simulator-agnostic via small, focused Bayesian Optimization on two axes
  only: `model_size`, `horizon` (initialized from plant ID; lookback is
  pinned to the identified value).
- One ONNX artifact per workflow: a single integrated graph
  `(obs_window, prev_actions) → action`. No separate observer model — the
  causal tokenizer + dynamics transformer *is* the observer.

## Architecture (paper-faithful, adapted to vector APC observations)

- **Causal Tokenizer** (`models/dreamer_v4.py:Tokenizer`): MLP encoder +
  linear+tanh bottleneck + MLP decoder. MAE-style channel dropout during
  training. MSE reconstruction loss (paper eq. 5; LPIPS dropped — not
  applicable to scalar observations).
- **Interactive Dynamics** (`models/dreamer_v4.py:DynamicsTransformer`):
  block-causal-in-time transformer with pre-RMSNorm, RoPE, SwiGLU, QKNorm
  and attention soft-cap (paper §3.4). Per timestep we feed `n_register`
  register tokens + 1 action token + 1 (τ, d) token + 1 z̃ token. Trained
  with **shortcut forcing** (paper eq. 7) using x-prediction; bootstrap
  distillation handles d > d_min. K = 4 sampling steps per inference frame.
- **Three explicit phases** (paper Algorithm 1, adapted to single-task online APC):
  - Phase 1 — pretrain world model: tokenizer recon + dynamics shortcut forcing.
  - Phase 2 — agent finetune: keep WM losses live; add policy + reward MTP heads (eq. 9).
  - Phase 3 — imagination training: freeze WM transformer, train policy via PMPO (eq. 11) and value via TD-λ (eq. 10) on K=4 imagined rollouts.
- **Heads** — carried over from Dreamer 3 (paper still uses these in V4):
  - Policy: per-action-dim categorical over 21 uniform bins in [−1, 1].
  - Reward + Value: symexp twohot (255 bins on [−20, 20]).

## Layout

```
models/dreamer_v4.py        # Causal tokenizer + dynamics transformer + heads (V4 paper-faithful)
training/train.py           # Three-phase trainer (Phase 1 WM pretrain / Phase 2 agent finetune / Phase 3 imagination RL)
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

| | `workflow/single_run.py` | `workflow/bo_runner.py` |
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
python -m workflow.bo_runner --simulation-dir simulation/test_sim
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
├── study.db                     # Optuna SQLite study (resume via load_if_exists)
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
python -m workflow.bo_runner --simulation-dir simulation/test_sim \
  --n_trials 8 \
  --trial_steps 0 \    # 0 = plant-tied auto
  --final_steps 0 \    # 0 = plant-tied auto
  --seed 0 \
  --init-from-ckpt path/to/best.pt \   # optional warm-start (see below)
  --out output/my_custom_dir   # optional
```

### Warm-starting from a previous checkpoint (`--init-from-ckpt`)

Both `workflow/single_run.py` and `workflow/bo_runner.py` accept
`--init-from-ckpt PATH` to load **model weights only** from a previous
run's `best.pt` / `final.pt` (`strict=False`, so removed/renamed
parameters are tolerated). Optimisers, replay buffer, phase counters,
and auto-tuned hyperparameters all start fresh.

For BO, the warm-start is applied to **every trial** (including the
final retrain) via `TrainConfig.init_from_ckpt`, so each trial starts
from the same pretrained weights and only the BO-tuned axes
(`model_size`, `horizon`) vary.

**Optuna study resume *is* supported.** The study is persisted to
`<out_dir>/study.db` (SQLite) with `load_if_exists=True`, so re-running
`workflow/bo_runner.py` against an existing `--out` directory reattaches
to the same study with the full trial history (TPE prior +
MedianPruner statistics intact). Behaviour on resume:

- Completed + pruned trials are kept and counted toward `--n_trials`.
  Only the *remaining* budget is run (`remaining = n_trials - prior_done`).
  Pass a larger `--n_trials` to extend an existing study.
- Stale `RUNNING` trials left over from a crashed/killed process are
  auto-marked `FAIL` so Optuna's bookkeeping is consistent.
- The plant-derived first trial (`model_size=auto, horizon_mult=1.0`)
  is enqueued only on a fresh study; resumed studies skip it (it was
  already run trial #0).
- Per-trial directories on disk (`trials/trial_XXXX/`) survive but are
  not re-validated. **Caveat:** if the BO process died mid-trial, that
  trial's folder may be partially written; inspect / delete it before
  restart if you care about the artefacts.

To start a clean study, delete `<out_dir>/study.db` or pass a fresh
`--out` directory. Combine with `--init-from-ckpt` to both resume the
search *and* keep warm-starting model weights.

### Environment overrides (all optional)

| Var | Effect |
|---|---|
| `OBJ_REWARD_SCALE` | `auto` (default) / `off` / `<float>` — disable or force reward scale |
| `OBJ_BATCH_SIZE` | force a batch size (else auto from GPU mem) |
| `SIM_EPISODE_LENGTH` | force episode length (else auto from settling time) |
| `SIM_SAMPLE_RATE` | force sample rate (else auto from `τ_fast / 10`) |
| `SEED` | RNG seed (default 0) |
| `DREAMER_FAST_ATTN` | `1`/`sdpa` force SDPA, `0`/`manual` force the paper soft-cap path. **Default: SDPA whenever a CUDA device is available** (~6–9× faster than manual; QKNorm provides numerical safety). ONNX export always uses `manual`. |
| `SEED_TARGET_CV_FRAC` | seed-PRBS amplitude as a fraction of avg CV-bound width (default 0.20). Lower → narrower exploration; raise for plants where the actor needs to learn large MV moves. |
| `SIGMA_MAX_CAP` | upper bound on the auto-derived policy `σ_max` (default 0.30). Raise to allow wider directional MV swings; lower to keep exploration tight. |
| `SIGMA_MAX_FLOOR` | lower bound on the auto-derived `σ_max` (default 0.10). |
| `SIGMA_MAX_OVER_SEED` | multiplier of `baseline_seed_action_std` used to set `σ_max` (default 1.0). |
| `SIGMA_MIN_RATIO_OF_MAX` | `σ_min = σ_max / ratio` (default 2.5, min 2.0). |
| `OBJ_AUTO_ECON_OVER_MOVE_RATIO` | minimum ratio of `econ_budget` to per-step MV move penalty at typical actor jitter (default 2.0). Caps `move_base` so the user's economics term always strictly dominates the move term. Set to 1.0 to disable the cap; set higher (e.g. 5.0) for plants where you want the actor to ignore move pressure entirely while economics is small. |

## Single training run (no BO)

```bash
python -m workflow.single_run --simulation-dir simulation/test_sim --steps 500000
```

Same plant-derivation chain; output goes to `output/<sim>/run_<ts>/`.
Useful for quick smoke tests or debugging without the BO overhead.
Accepts `--init-from-ckpt PATH` for warm-starts (see *BO* section above
for semantics; identical for single-run).

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
  `(d_model, n_layers, n_heads, z_dim, n_register)` tuples adapted from
  paper §3.4 to vector-observation APC scale.
- `horizon` — 5-point band `{0.5, 0.75, 1.0, 1.5, 2.0} × H_init`,
  where `H_init = ⌈(θ + 3τ) / sample_rate⌉`.

The first trial is enqueued at the plant-derived seed config.  An Optuna
`MedianPruner` (3 startup trials, 5 warmup steps) cuts visibly bad trials
early via the `on_iter_end` callback in the trainer.

## Adaptive knobs (paper-superset)

| Knob | Floor (paper) | Auto rule | Override |
|---|---|---|---|
| `batch_size` | 16 | nearest power of two filling ~50 % of GPU memory; per-batch cost = `{S:220, M:330, L:640} MB × horizon/42` and is scaled by ~0.55 when SDPA is on; re-derived per BO trial | `OBJ_BATCH_SIZE` |
| `reward_scale` | 1.0 | `target_std=1.0 / measured_raw_std`, clamped ≥ 1.0 | `OBJ_REWARD_SCALE` |
| `episode_length` | 600 | `20 × (τ + θ)` clamped to `[500, 4000]` | `SIM_EPISODE_LENGTH` |
| `sample_rate` | 5 | `min(τ_fast / 10, θ_fast / 2)` | `SIM_SAMPLE_RATE` |
| `seq_len` | 64 | `max(64, ⌈(3τ + θ) / sr⌉)` | — |
| `model_size` | M | `S/M/L` from complexity score | — |
| `trial_steps` | 50 000 | `40 eps × max(1, complexity / 4) × ep_len`, clamped | `--trial_steps` |
| `final_steps` | 200 000 | `10 × trial_steps`, clamped | `--final_steps` |
| `attn_impl` | manual (paper soft-cap) | `sdpa` whenever CUDA is available | `DREAMER_FAST_ATTN` |
| `baseline_seed_action_std` | n/a | `clip(target_cv_frac × cv_w / mv_auth, 0.01, SEED_SIGMA_CAP)` with `target_cv_frac=0.20` | `SEED_TARGET_CV_FRAC`, `SEED_SIGMA_CAP` |
| `policy_log_std_max` | log(1.0) | `log(clip(SIGMA_MAX_OVER_SEED × σ_seed, FLOOR=0.10, CAP=0.30))` — plant-adaptive | `SIGMA_MAX_CAP`, `SIGMA_MAX_FLOOR`, `SIGMA_MAX_OVER_SEED` |
| `policy_log_std_min` | log(0.1) | `log(σ_max / SIGMA_MIN_RATIO_OF_MAX)` (default ratio 2.5) | `SIGMA_MIN_RATIO_OF_MAX` |

## Performance notes

- **SDPA attention** (FlashAttention-2 / cuDNN, auto-dispatched) is the
  default on CUDA. Bench on test_sim (model L, B=16, T_ctx=128, K=8):
  manual no-cache 1567 ms/call → manual + KV-cache 256 ms (6.1×) →
  SDPA + KV-cache 180 ms (8.7×).
- **KV-cache for `imagine_next_z`**: per-layer past-step keys/values are
  built once at the start of each imagination rollout and only the
  current step's tokens are re-projected through each block per K
  iteration. Numerical equivalence vs the uncached path: max abs err
  1.16e-6.
- ONNX export still uses the manual soft-cap attention path for
  exporter compatibility; only training inference uses SDPA.

## Setup

```bash
python3 -m venv ../neural-apc-dreamerV4-env
source ../neural-apc-dreamerV4-env/bin/activate
pip install -r requirements.txt
```

## Status

- DreamerV4 paper-faithful trainer + RSSM + categorical actor + twohot critic.
- Adaptive batch / reward / episode-length / step-budget knobs (paper-superset).
- Single-arg workflow entry (`workflow/bo_runner.py`).
- Validation harness with timeseries plots.
- ONNX export of integrated `(rssm + actor)` graph.
- SDPA attention + KV-cached imagination rollouts (default on CUDA;
  6–9× faster Phase-3 iter).
- Warm-start from previous checkpoint via `--init-from-ckpt`
  (single-run *and* every BO trial).
- Optuna BO study resume via persistent `<out_dir>/study.db` (SQLite,
  `load_if_exists=True`); stale `RUNNING` trials auto-marked `FAIL`,
  remaining budget = `n_trials − prior_done`.

See `docs/` for design notes.
