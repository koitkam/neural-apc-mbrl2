# KNOWN-GOOD BASELINE — p106 (MC-critic), 2026-06-09

This is the **best run of the series** and the safe revert point before the large
WM-fix workstream. If a later change regresses, revert code to git tag
`baseline-p106-known-good` and relaunch with the recipe below.

## Why this is the baseline
- **agentEcon −30.6 vs do-nothing baseline −98.8 = +69%** (the actor genuinely
  optimizes economics — the core goal).
- cv_violation 10.7, mv_violation 0.52 (limits respected), mv_tv 979 (smooth, not
  oscillating), overshoot 0.82.
- wm_gain_rel_err 0.186 (healthy), critic_r 0.738.
- Stable: 546 iters, NO cascade, best_det −127 @ iter 346 (late, kept improving).

## The fix that made it work
The **real-return (Monte-Carlo) grounded critic** (`critic_mc_grounding_coef`):
grounds the critic target in actual observed economic returns (no value bootstrap),
which lifted the actor from "conservative/noisy" to "decisive economic". See the
critic_mc_rew_to_tgt_var canary.

## Exact launch recipe (test_sim)
```
DREAMER_GAMMA=0.97
DREAMER_CRITIC_MC_GROUNDING_COEF=1.0
DREAMER_CRITIC_IMAG_LOSS_COEF=0.3
DREAMER_CRITIC_REPLAY_ANCHOR_COEF=0.0
DREAMER_RSSM_FREE_BITS=0.5
DREAMER_RSSM_IMAG_LATENT_MODE=1
DREAMER_BOUND_TRAINING_REWARD_MAX=3.0
DREAMER_EXPERT_TYPE=static
DREAMER_WM_HELD_ROLLOUT_COEF=0.5
DREAMER_WM_HELD_ROLLOUT_LEN=55
DREAMER_WM_HELD_ROLLOUT_SETTLE_FRAC=0.5
DREAMER_WM_HELD_ROLLOUT_MAX_STARTS=8
DREAMER_P3_CRITIC_WARMUP_ITERS=10
DREAMER_WM_TRUNK_STOPGRAD_IN_P2=1
DREAMER_TRAIN_MODE=joint
DREAMER_WM_OVERSHOOT_COEF=0.3
DREAMER_WM_OVERSHOOT_LEN=55
python -m workflow.single_run --simulation-dir simulation/test_sim --out-dir <OUT>
```
NOTE: `return_value_cap_gamma_horizon` is now a code DEFAULT (was an explicit env
in p106) — no behavior change. The objective ran at the DEFAULT (no
`OBJ_AUTO_CV_OVER_ECON_RATIO`) — i.e. the standard CV-dominates-economics ladder.

## Files here
- `best.pt` — the p106 best checkpoint (gitignored; on disk for revert/eval).
- `run_plan.json` — frozen resolved config.
- `validation_summary.json` — full validation metrics.
- `env_overrides.txt` — the resolved env-overrides from the run log.

## What came after (and FAILED — do not re-use without the fixes)
- p107: econ-led objective (`OBJ_AUTO_CV_OVER_ECON_RATIO=1.0`) + integral boost →
  **limit cycle** (dead-time plant, mv_tv 4019, mv_viol 34.8).
- p108: econ-led + integral boost OFF → cycle gone but parks OUTSIDE CV limit
  (cv_v 72, agentEcon −131, worse than baseline). The econ-led objective at
  ratio 1.0 needs more work; the default (this baseline) is best for now.
