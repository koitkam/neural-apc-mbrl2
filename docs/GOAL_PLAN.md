# GOAL_PLAN — neural-APC-mbrl2

Living plan. Update every visit (live analysis and EXIT). Champions live in `docs/RUN_HISTORY.md`.

**Product:** simulator-agnostic neural APC — smooth CV on the economic limit without violating; faithful observer; unmeasured-load rejection. Envelope: learned observer + neural Kalman/DOB + neural actor-critic. No gray-box plant, no PID/LQR/MPC as the product, no DV-only FF.

Plant this cycle: `nonlinear_sim` HeatExchangerTower (job 2 of 2 after P117, then return-to-`test_sim`).

## Residual board vs champions (test_sim unless noted)

| Axis | Score we trust | Champion | Last VALID (P116) | This plant P117 | Status |
|---|---|---|---|---|---|
| R1 TM MV ss/@H / curve | val TM, not jsonl ×1 | P26 ×0.973 / @H ×0.880 | ×0.706 / ×0.711 | freeze CAPPED 0.71@DV — **do not score** | OPEN |
| R1 TM DV ss/@H / curve | same | P64 ×0.893 / @H ×0.962 | ×0.692 / ×0.712 | same | OPEN |
| R1 compounding | 1step→OL | P64 ×0.85 | ×0.911 | n/a (INVALID freeze) | OPEN |
| R2 CV smoothness | worst-seed `cv_d2_rms_normed`≤0.05 AND `cv_reversal_rate`≤0.25 | — | P116 `smooth_pass` | n/a | OPEN |
| R2 headroom / viol | `cv_opt_headroom`, `cv_viol_frac` (not a P3-skip gate) | — | hugging 0.35; mv_viol 1.67 | n/a | OPEN |
| R3 Kalman | `det_r` + pred_std vs true | P26 det_r 0.68 | 0.676 / 1.479 vs 1.93 | n/a | OPEN |
| Actor econ | paired vs baseline **and** P64; only if `actor_experiment_valid` | P64 −4.54 vs −94 | −8.24 vs −109.83 VALID | **do not score** | OPEN |

## Metric audit (trusted vs on trial)

**Trusted:** val TM ss/@H/curve_iae (MV and DV); gain decomp real→post / post→1step / 1step→OL; `det_r` + pred_std vs true; paired econ vs baseline **and** champion when freeze GAIN-READY and P3 ran; CV d2/reversal (smooth_pass); `cv_opt_headroom` / `cv_viol_frac`.

**On trial / do not GPU-optimize:** jsonl teacher ×1 vs SysID median (tautology on OP-varying plants — P117); ss-ratio alone; mv_reversal as smoothness gate; CV total variation as smoothness gate; critic_r without `critic_rew_to_tgt_var`; raw dist R²; VALID 9/9 / GAIN-READY / all_pass / family-closed as “residual closed”; beating P64 to KEEP a smoothness/headroom/DR win.

P117 evidence: jsonl Huber sat ~×1 while 5-level TM 0.71@DV. SysID same-Δu OP-gain MV **2.23×** / DV **2.34×**. Linear RSSM + one identified G cannot be the teacher.

## Closed families (do not N+1)

extra-P1 as freeze (P41; **P117 on this plant**); compounding-teacher Huber (P111 traj unpromoted, P115 k1 REVERT, P116 ol1 REVERT); isolation-off KEEP as default; 2TS-α as champ; kfeat as freeze; quiet_env as freeze; decoder FF; TSSM; GRU keep-h.

## Next job (this visit)

**P118 `opscale`** — one mechanism, two attributable pieces:

1. LPV `op_scale_net` on GRU/token inputs (identity init; group `g`).
2. Rest-IC Huber `G_tgt` = plant FD at each rest OP (not SysID median).

Predicted signature: GAIN-READY vs P117 CAPPED 0.71@DV; jsonl ratio vs **local** G; `wm_op_scale_dev` leaves 0 after P1.

Falsifier: still CAPPED ~0.71@DV **and/or** test_sim smoke/teacher detonates.

After P118 EXIT: if GAIN-READY, score actor if P3; then **return-to-test_sim** on the worst residual. If still GAIN_NOT_READY: change observer piece or take the residual back to test_sim — do **not** extra-P1 N+1.

## RCA this visit

- SysID: OP-varying DC (equal-% + 1/feed). Median G is the wrong pin.
- Signal: P117 jsonl NaN/Inf 0; skip 0; first P2 dobg 0.0049.
- Control: freeze never GAIN-READY — do not score actor.
- ML: extra-P1 lottery FALSIFIED as freeze.
- Plant: HeatExchangerTower is the test of env-free identity, not a knob dump.
- Metric: jsonl ×1 vs identified G is not val TM.

P117 process died mid-P2 (SIGKILL / session teardown). Not a training crash. Do not identity-relaunch.
