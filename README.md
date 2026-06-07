# neural-apc-mbrl

**Model-based reinforcement-learning controller for Advanced Process Control
(APC).** Forked from `neural-apc-dreamerV4` (2026-06-06) to pursue a
paper-faithful **joint** training loop (DreamerV1/V2/V3 style) and a
multi-backbone world model (RSSM default; transformer opt-in), beyond the
staged Dreamer-4 curriculum. Sibling of `neural-apc-pytorch` (model-FREE
SAC/PPO/TD3); this repo is the model-BASED line.

Reference: Hafner, Yan, Lillicrap (2025), "Training Agents Inside of Scalable
World Models" (Dreamer 4), [arXiv:2509.24527](https://arxiv.org/abs/2509.24527),
plus the DreamerV1–V3 lineage (joint WM+actor+critic training).

## Goals

- Model-based control with a learned world model (RSSM default, transformer opt-in).
- Stay close to the Dreamer papers. Add adaptive knobs only when they remain a strict
  superset of the paper recipe (paper defaults as floors / minimums).
- Simulator-agnostic via small, focused Bayesian Optimization on two axes
  only: `model_size`, `horizon` (initialized from plant ID; lookback is
  pinned to the identified value).
- One ONNX artifact per workflow: a single integrated graph
  `(obs_window, prev_actions) → action`. No separate observer model — the
  causal tokenizer + dynamics transformer *is* the observer.

## Training modes (`DREAMER_TRAIN_MODE`)

- **`phased`** (default): the staged Dreamer-4 curriculum — P1 world-model
  pretraining → P2 reward-head + policy-BC warmup → P3 actor+critic via
  imagination, with phase boundaries. Best when the world model is expensive
  to train (transformer backbone) and benefits from amortized pretraining.
- **`joint`** (DreamerV1/V2/V3 style): after the seed-buffer **prefill**,
  co-train the world model, actor, and critic **every step from step 1** — no
  phase boundaries. This eliminates the phase-boundary failure modes (recon
  destabilization at P1→P2 from gradient bleed, cold-critic cascade at P2→P3,
  checkpoint-discard) because all three components co-adapt. Recommended for
  the cheap RSSM backbone. The critic warmup (`DREAMER_P3_CRITIC_WARMUP_ITERS`)
  still runs at the very start so the value head calibrates before actor
  coupling; `DREAMER_JOINT_PRIOR_REFRESH_ITERS` periodically refreshes the
  PMPO prior.

## Architecture (paper-faithful, adapted to vector APC observations)

The **world model is selectable** via `world_model_type` (TrainConfig /
`DREAMER_WORLD_MODEL_TYPE`). Two backbones share the same reward / value /
policy heads and the same three-phase trainer:

### World model — `rssm` (DreamerV3 RSSM, **default** since P68)

- **`models/dreamer_v4_rssm.py:RSSMDynamics`**: DreamerV3 recurrent
  state-space model — a deterministic GRU core
  `h_t = f(h_{t-1}, z_{t-1}, a_{t-1})` plus a 32×32 categorical stochastic
  latent `z` (straight-through one-hot, 1% uniform mixture). MLP
  encoder/decoder (LayerNorm+SiLU) operate in **already-normalized** obs
  space (no extra symlog). Trained with reconstruction (MSE) + KL-balanced
  free-bits loss (dyn weight 0.5, repr weight 0.1, free-bits 1.0 nat).
  The agent feature is `feat = [h, z_flat]`.
- **Why it is the default**: the SF-transformer below has no recurrent
  fixed point (`wm_pred_converges_under_constant_action = 0.0` by
  construction), which drove a Phase-3 bootstrap cascade that no
  critic/reward-side fix could break (P64/P66/P67). The RSSM's GRU can
  learn a held-action fixed point `h* = f(h*, z*, a)`.

### World model — `sf_transformer` (Dreamer 4 shortcut-forcing, legacy opt-in)

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
  Select via `DREAMER_WORLD_MODEL_TYPE=sf_transformer`.

### Shared trainer + heads

- **Three explicit phases** (paper Algorithm 1, adapted to single-task online APC):
  - Phase 1 — pretrain world model: RSSM recon + KL (or tokenizer recon + dynamics shortcut forcing for `sf_transformer`).
  - Phase 2 — agent finetune: keep WM losses live; add policy + reward MTP heads (eq. 9).
  - Phase 3 — imagination training: freeze WM, train policy via PMPO (eq. 11) and value via TD-λ (eq. 10) on imagined rollouts.
- **Heads** — carried over from Dreamer 3 (paper still uses these in V4),
  shared by both backbones (built on `feat`):
  - Policy: continuous Tanh-Normal (per-action-dim) — `ContinuousPolicyHead`.
  - Reward + Value: symexp twohot.

## Layout

```
models/dreamer_v4.py        # Heads (reward/value/policy) + SF-transformer WM (tokenizer + dynamics) + world_model_type dispatch
models/dreamer_v4_rssm.py   # DreamerV3 RSSM world model (GRU + 32×32 categorical latent) — DEFAULT backbone (P68)
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
source ../neural-apc-mbrl-env/bin/activate
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

#### Hidden unmeasured-disturbance model + noise curriculum (P90)

Unmeasured (truly hidden) CV upsets the controller must reject are modeled by a
**realistic, simulator-adaptive event schedule** (`HiddenDisturbanceSchedule`)
— not a constant wiggle. Each episode draws a sequence of discrete events with
varied **shape** (`step` instant load change, `ramp` gradual drift, `pulse`
temporary excursion, `ou_drift` noisy patch), **timing** (sometimes isolated
with ≥ settling-time gaps so each upset reaches steady state, sometimes
overlapping/serial), and **persistence** (some revert to baseline, some hold
permanently). All timescales derive from the identified dead time + dominant
time constant (`settle = (dead + N·τ)/sample_rate` agent steps); all magnitudes
are capped by the agent's MV→CV authority. The schedule is never exposed to the
agent or world model.

**P89 noise RCA + fix:** the world model never learned a held-action steady
state because *every* training trajectory — including the const-action
steady-state seeds — carried persistent process-OU (~1.3% span) + measurement
noise, so the plant never truly settled. Two coupled fixes give the WM clean
fixed-point supervision while keeping realistic noise for disturbance rejection:
the **clean steady-state seeds** (`clean_steady_seeds`, default ON) make the
held-action seed episodes fully noise-free, and the **process-noise curriculum**
(`process_noise_curriculum`, default ON) ramps process + measurement noise from
~0 up to full over P1 (and always full in P3).

| Var | Effect |
|---|---|
| `DREAMER_HIDDEN_DISTURBANCE` | `0` to disable the hidden disturbance entirely (default ON). |
| `DREAMER_HIDDEN_DIST_MODE` | `schedule` (default, realistic event schedule) or `ou` (legacy single always-on OU drift). |
| `DREAMER_HIDDEN_DIST_SETTLE_NTAU` | `N` in `settle = (dead + N·τ)/sr` (default 4 ≈ 98% settling). Controls event spacing + held-to-steady durations. |
| `DREAMER_HIDDEN_DIST_MAX_EVENTS` | Cap on events per episode (default 6). |
| `DREAMER_HIDDEN_DIST_P_ISOLATED` | P(gap ≥ settle, i.e. event fully settles before the next) vs overlap/serial (default 0.5). |
| `DREAMER_HIDDEN_DIST_P_REVERT` | P(an event reverts to baseline) vs holds permanently (default 0.5; `pulse` always reverts). |
| `DREAMER_HIDDEN_DIST_SHAPE_WEIGHTS` | `"step,ramp,pulse,ou"` sampling weights (default `0.3,0.3,0.2,0.2`). |
| `DREAMER_CLEAN_STEADY_SEEDS` | `0` to keep noise on the const-action / step-settle steady-state seeds (default ON = noise-free seeds). |
| `DREAMER_PROCESS_NOISE_CURRICULUM` | `0` to disable the P1 process+measurement noise ramp (default ON). |
| `DREAMER_PROCESS_NOISE_AMP_RAMP` | `"<start>:<reach>"` (default `0.0:0.4`): noise scale ramps from `start` at progress=0 to full by `progress=reach` in P1/P2; P3 always full. |
| `DREAMER_DISTURBANCE_PROB_WM` | Per-episode probability cap in P1/P2 (default 0.10). In P1 acts as the upper bound of the adaptive ramp; in P2 acts as the floor (P2 starts at this value). Observable schedule events (SP/DV) fire on 100% of episodes, so 0.10 gives the WM ~10× more clean episodes than disturbed ones during early learning. |
| `DREAMER_DISTURBANCE_PROB_P2` | Per-episode probability cap in P2 (default 0.20). P2 linearly ramps from `DREAMER_DISTURBANCE_PROB_WM` (0.10) up to this cap as critic training progresses. Rationale: critic learns value of imagined rollouts starting from buffered real states; broadening buffer coverage with more disturbed episodes lets the critic estimate value across the disturbed manifold. |
| `DREAMER_DISTURBANCE_PROB_AGENT` | Per-episode probability cap in P3 (**default 0.30**, P89: was 0.50 — a realistic plant sees occasional upsets ~20–30% of the time, and 50% corrupted too much of the actor's gradient + never let the CV settle). P3 linearly ramps from `DREAMER_DISTURBANCE_PROB_P2` (0.20) up to this cap. |
| `DREAMER_HIDDEN_OU_PROB_MIN` | Floor of the **P1 adaptive** trigger probability (default 0.05). The OU fires at least this often even before the WM has learned anything. |
| `DREAMER_HIDDEN_OU_PROB_MAX` | Cap of the **P1 adaptive** trigger probability (default = `DREAMER_DISTURBANCE_PROB_WM`, i.e. 0.10). |
| `DREAMER_HIDDEN_OU_PROB_TARGET_SCORE` | WM fidelity score at which the **P1** trigger probability reaches `PROB_MAX` (default 2.0 ≈ all four probe horizons pass the 0.40 Pearson-r floor). Score = `sum(max(0, r_h)) + 0.05·best_h/H`. P1 only. |
| `DREAMER_HIDDEN_OU_PROB_P2_RAMP_REACH` | Fraction of P2 budget at which the P2 trigger probability reaches `DREAMER_DISTURBANCE_PROB_P2` (default 0.5 = midpoint of P2). |
| `DREAMER_HIDDEN_OU_PROB_P3_RAMP_REACH` | Fraction of P3 budget at which the P3 trigger probability reaches `DREAMER_DISTURBANCE_PROB_AGENT` (default 0.5 = midpoint of P3). |
| `DREAMER_HIDDEN_OU_AMP_RAMP` | `"<start>:<reach>"` (default `0.1:0.4`). Linear amplitude ramp from `start` at progress=0 to 1.0 at `progress=reach`, then capped by the phase-aware amplitude cap. |
| `DREAMER_HIDDEN_OU_AMP_MAX_SCALE` | Hard cap on `curriculum_amp_scale()` in **P1/P2** (default 0.2). With base `amp_frac=0.10`, peak disturbance ≈ 2% of MV authority during WM/critic learning. |
| `DREAMER_HIDDEN_OU_AMP_MAX_SCALE_P3` | Hard cap on `curriculum_amp_scale()` in **P3** (default 1.0). The WM is frozen in P3 and the actor must learn realistic-magnitude rejection, so amplitude jumps to full nominal. |
| `DREAMER_HIDDEN_OU_AMP_JITTER` | `"<lo>:<hi>"` per-episode amplitude DR multiplier (default `0.6:1.6`). |
| `DREAMER_HIDDEN_OU_DRIFT_FRAC` | Max constant per-episode mean offset as fraction of amp (default 0.4). |

**Per-phase OU curriculum (P38, 2026-05-22):** The hidden disturbance
ramps along *two* orthogonal axes, each scoped to where it does the
least harm:

| Phase | Trigger probability | Amplitude cap | Driving signal |
|---|---|---|---|
| P1 (WM) | 0.05 → 0.10 | 0.2 | `wm_best_score / TARGET_SCORE` |
| P2 (critic) | 0.10 → 0.20 | 0.2 | `phase_progress / P2_RAMP_REACH` |
| P3 (actor) | 0.20 → 0.30 | 1.0 | `phase_progress / P3_RAMP_REACH` |

This gives a continuous monotonic ramp across phases: trigger
probability and amplitude both start small (clean signal for WM
learning), broaden gradually as the critic needs broader state-space
coverage, and reach full operational magnitude only in P3 when the
actor is the one learning to reject.

#### WM fidelity early-stop and `wm_best.pt`

| Var | Effect |
|---|---|
| `DREAMER_WM_PROBE_EVERY_ITERS` | Run the fidelity probe every N P1/P2 iters (default 10; set 0 to disable). |
| `DREAMER_WM_FIDELITY_WARMUP_ITERS` | Iters before degradation early-stop can trigger (default 40). |
| `DREAMER_WM_FIDELITY_PATIENCE_ITERS` | Iters without a new best score before P1/P2 early-stop trips (default 50). |

When the score improves, a `wm_best.pt` checkpoint is written to the run
output directory alongside the periodic `ckpt_iter_*.pt` saves, with
the probe metadata (per-horizon r, best_h, iter) embedded. At the
P1→P2 transition the trainer auto-restores `wm_best.pt` (P39, see
below) so critic training starts from the highest-fidelity WM rather
than from post-peak drift weights.

#### P1 const-action re-injection and WM warm-restore (P39)

P38 RCA showed the WM's H=15 imagination fidelity peaks early in P1
(iter ≈ 50 of 100) then collapses to noise even though supervised
losses stay flat. Root cause: the front-loaded const-action seeds
(`constant_action_seed_episodes`) are diluted as the buffer fills with
several×their volume of random-action episodes, so the WM forgets
steady-state behaviour. Mitigations:

| Var | Effect |
|---|---|
| `DREAMER_CONST_ACTION_INJECT_EVERY` | Inject N fresh const-action episodes every K P1 iters (default K=20; set 0 to disable). |
| `DREAMER_CONST_ACTION_INJECT_N` | Episodes per injection (default 5, stratified across `constant_action_seed_op_band`). |
| `DREAMER_WM_BEST_RESTORE_AT_P2` | Reload `wm_best.pt` at the P1→P2 boundary (default 1; set 0 to disable). |
| `DREAMER_WM_BEST_RESTORE_MIN_GAP` | Skip restore when `total_iters - wm_best_iter` is below this (default 10). |

Both knobs are sim-agnostic and adaptive: injection uses
`cfg.episode_length` and the env's existing action range; warm-restore
no-ops when `wm_best.pt` is essentially the current state.

#### Reward-MTP / WM-coupling diagnostics (P39)

Four lightweight knobs to localise *why* the H=15 fidelity collapses
even when supervised losses look healthy. A and D are standing
observability (default ON, log-cadence-gated, <2% total overhead).
B and C are controlled-experiment switches (default OFF) for one-off
causal runs.

| Var | Effect |
|---|---|
| `DREAMER_DIAG_PERHEAD_GRADS_EVERY` | (A) Log `diag_grad_{recon,sf,rmtp}` — per-loss gradient norms at the tokenizer's first parameter. Default 10 iters; 0 = off. |
| `DREAMER_DIAG_LATENT_STABILITY_EVERY` | (D) Log `diag_latent_cos_{mean,min}` — cosine similarity of the encoder output on a fixed 64-transition reference set, vs. its values when first sampled. A sharp drop = the encoder is re-organising. Default 10 iters; 0 = off. |
| `DREAMER_DIAG_DISABLE_REWARD_MTP_IN_P1` | (B) Ablate the reward MTP loss term from P1's `total_loss`. Reward head receives no gradient in P1 (will be untrained entering P2). If the fidelity cliff disappears, reward gradients are the cause. Default 0. |
| `DREAMER_DIAG_REWARD_MTP_STOP_GRAD_IN_P1` | (C) Keep training the reward head but detach `agent_hid` before it. Head still learns; encoder/dynamics latent no longer receives reward-head gradient. Complements B: if C alone fixes the cliff, the problem is gradient distortion of shared params; if only B fixes it, it's something deeper. Default 0. Ignored when B is also set. |

Recommended next experiment (P39-B): run with
`DREAMER_DIAG_DISABLE_REWARD_MTP_IN_P1=1` and compare the
`diag_latent_cos_mean` + fidelity-probe trajectory to the standing
P39 baseline.

#### WM gain fidelity, latent overshooting & critic grounding (P88, 2026-06-05)

The correlation-based fidelity probe (`per_offset` Pearson r, `best_h`,
`wm_next_state_r`) is **scale-invariant**: it confirms the WM moves the
right *direction* but says nothing about the **gain** (ΔCV per ΔMV) or
settling dynamics — the property that actually matters for control. A run
can score `wm_r ≈ 0.75` while its steady-state gain is 3–4× too small.
Validation now also emits a **DMC-style transfer-function matrix**
(`validation/wm_transfer_matrix.{png,json}`, `evaluation/wm_transfer_matrix.py`):
per MV→CV pair it steps the MV from settled operating points across the
region and overlays the **world-model vs real-sim** engineering-gain curves
(mean + min/max band), with a `wm_gain_rel_err` gate in `fidelity_gates`
(healthy < 0.35, pass < 1.0). The critic calibration diagnostic now reports
`slope_g_on_v` (OLS slope; ≈1 = calibrated, ≈0 = compressed, <0 = sign-flipped)
and `nmae` alongside the (scale-free-blind) `r_pearson`.

Two training levers target the two failure modes those metrics expose — both
default-OFF (paper-faithful) and apply to both entry points (single run + BO):

| Var | Effect |
|---|---|
| `DREAMER_WM_OVERSHOOT_COEF` | **(#2) Multi-step latent overshooting (RSSM).** Roll the prior open-loop `LEN` steps under real actions with no obs and penalise decode-vs-real-obs, directly training multi-step **gain/dynamics** accuracy (DreamerV3 trains the prior 1-step only). `0` = off. The SF-transformer no-ops (its shortcut-forcing loss is the native multi-step term). |
| `DREAMER_WM_OVERSHOOT_LEN` | Open-loop horizon K to supervise (default 15; set to the imagination horizon for long-H runs). |
| `DREAMER_WM_OVERSHOOT_MAX_STARTS` | Cap on strided start positions per batch (default 24) — bounds the added GRU cost. |
| `DREAMER_WM_OVERSHOOT_GATE_RECON` | Soft recon-fidelity gate (default 0.1): scales the term by `min(1, gate/recon_loss)` so it ramps in only as 1-step reconstruction converges (no early-P1 destabilisation). |
| `DREAMER_CRITIC_IMAG_LOSS_COEF` | **(#1) Critic real-grounding rebalance.** Weight on the *imagined* critic CE (default 1.0 = legacy). `<1.0` lets the real-return replay anchor (`critic_replay_anchor_coef`, `critic_anchor_lambda`) dominate the value target, breaking the bootstrap self-dominance (`critic_rew_to_tgt_var → ~0.001`) that freezes the actor. Both backbones. |

## Single training run (no BO)

```bash
python -m workflow.single_run --simulation-dir simulation/test_sim
```

Same plant-derivation chain; output goes to `output/<sim>/run_<ts>/`.
All knobs auto-derive from the plant (including `--steps`, which now
defaults to `0` = plant-tied auto-derivation via `utils.phase_budget`).
Pass `--steps N` only to override; useful for quick smoke tests.
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

## Cascade stabilizers — fixed defaults

These are the **non-adaptive** anti-cascade / anti-runaway defaults. They are
universal (sim-agnostic) constants, not auto-tuned. Both world-model backbones
(`rssm` and `sf_transformer`) carry their own copy in each imagination path, and
`workflow/bo_runner.py` inherits all of them via `TrainConfig()` defaults. Keep
this table in sync whenever a default changes.

| Knob | Default | Role | Override | Origin |
|---|---|---|---|---|
| `bound_training_reward` | `True` | clip per-step training reward to `[-B, B]` after scale-invariant remap (`raw·B/ref`); `info['raw_reward']` stays unshaped for validation | `DREAMER_BOUND_TRAINING_REWARD` | Cursor stabilizer #1 (P73/P77) |
| `bound_training_reward_max` (`B`) | `6.0` | the bound `B`; dimensionless because `ref` cancels econ magnitude | `DREAMER_BOUND_TRAINING_REWARD_MAX` | P77 (under test at `B=3`, P79) |
| `bound_training_reward_ref` (`ref`) | `50.0` | econ-derived adaptive clip used to normalize raw reward before bounding | — | P77 |
| `return_scale_abs_cap` | `500.0` | absolute hard cap on the percentile return_scale EMA (`0` disables); prevents the return-normalization runaway that froze the actor | `DREAMER_RETURN_SCALE_ABS_CAP` | Cursor stabilizer #2 (P79) |
| `advantage_clip` | `8.0` | clamp normalized advantage to `±clip` before the actor loss | `DREAMER_ADVANTAGE_CLIP` | Cursor stabilizer #3 (P74) |
| `critic_replay_anchor_coef` | `0.5` | anchors the critic on replayed real returns to resist the pessimistic self-consistent fixed point | — | anti-cascade |

## Disabled / dormant knobs (intentionally off)

Off by design — **not** failed code, but levers we keep wired for a specific
future use or paper parity. Do **not** delete these without re-checking the
rationale here.

| Knob | Default | Why off now | When to enable |
|---|---|---|---|
| `bc_scale` | `0.0` | Phase-2 policy behaviour-cloning weight. Clones the **logged buffer actions** over the MTP horizon. Our buffer is filled with *uniform-random* P1 exploration, so `bc_scale>0` would clone uniform → uniform `prior_policy` → the P3 PMPO-KL term pins the policy near uniform → collapse. | The day the seed buffer is populated with **expert demonstrations**. This is the exact mechanism for Cursor stabilizer #6 (expert-BC, `actor_bc_coef 0.15–0.18`): seed the trivial `pi_trim` expert (`0.75·mid + 0.25·trim`, computable from the CV/setpoint spec with **no real controller**) into the buffer, then set `bc_scale>0`. **Keep — it is the hook for an identified future port.** |
| `mae_p_max` | `0.0` | Tokenizer MAE-reconstruction masking prob. MAE is genuine V4 paper §3.1 but is defined for **image** observations; on low-D APC vector obs it collapses the encoder (diagnosed 2026-05-03). | Image / high-D observation plants. **Keep — it is a paper knob.** |

> **Removed knobs** (do not re-add — failed experiments, superseded):
> `return_scale_max_step_growth` (P63 growth-rate clamp — regressed −134997 vs
> −795; replaced by `return_scale_abs_cap`) and the `wm_steady_*` regulariser
> cluster (P64 — superseded structurally by the P68 RSSM held-action fixed
> point). The `wm_steady_state_diagnostic.py` *tool* is unrelated and retained.

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
python3 -m venv ../neural-apc-mbrl-env
source ../neural-apc-mbrl-env/bin/activate
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
