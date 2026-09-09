# GOAL_PLAN — neural-APC-mbrl2

Living plan. Update every visit (live analysis and EXIT). Champions live in `docs/RUN_HISTORY.md`.

**Product:** simulator-agnostic neural APC — smooth CV on the economic limit without violating; faithful observer; unmeasured-load rejection. Envelope: learned observer + neural Kalman/DOB + neural actor-critic. No gray-box plant, no PID/LQR/MPC as the product, no DV-only FF.

Plant this cycle: `nonlinear_sim` HeatExchangerTower (P118 = job 2 of 2, **LIVE**; then return-to-`test_sim`).

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

**On trial / do not GPU-optimize:** jsonl teacher ×1 vs SysID median (tautology on OP-varying plants — P117); **P118 jsonl `gain_match_mv_ratio` vs local G** (teacher Δu was engineering `_prev_control` − WM-norm action → local MV G **0.015** vs identified **2.63**; ratio ±10–70 is not TM); ss-ratio alone; mv_reversal as smoothness gate; CV total variation as smoothness gate; critic_r without `critic_rew_to_tgt_var`; raw dist R²; VALID 9/9 / GAIN-READY / all_pass / family-closed as “residual closed”; beating P64 to KEEP a smoothness/headroom/DR win.

P118 live: DV local G **−0.345** vs ident **−0.426** (same space, OK). MV local G is the unit bug, not equal-% OP variation (span **0.015** would be ~2× if the 2.23× SysID story were in WM-norm).

## Closed families (do not N+1)

extra-P1 as freeze (P41; **P117 on this plant**); compounding-teacher Huber (P111 traj unpromoted, P115 k1 REVERT, P116 ol1 REVERT); isolation-off KEEP as default; 2TS-α as champ; kfeat as freeze; quiet_env as freeze; decoder FF; TSSM; GRU keep-h.

## Live this visit — P118 `opscale` (do not kill / do not relaunch / do not second GPU)

tmux `mbrl2_p118` pid **551549** sha **`ff5f84c`** `device=cuda` bs=128 compile=eager nvidia **~14745 MiB**. `[resolved-cfg] opscale=True` no ol1. STAGE 1 `g=84 dob=8`. Rest-IC graph captured N=6 T=128. skip **0**. jsonl NaN/Inf **0**.

**Teacher print:** local G mean MV **0.0153** / DV **−0.345** vs identified **2.627 / −0.426**. jsonl@1–5: `gain_match_mv_ratio` **−74 → −3.6** (sign-flipping); `gain_match_dv_ratio` **~0.7–1.05**; `wm_op_scale_dev` **0.52→0.93** (tanh leaving identity, as designed, but chasing a unit-wrong MV pin). recon **0.44→0.25**. **Do not score actor.** **Do not treat opscale as FALSIFIED** — MV teacher was not WM-norm.

RCA: `_plant_fd_rest_local_g` used `env._prev_control` (valve %) minus rest action ∈[-1,1] as Δu. |du|~50 ⇒ G_MV≈dcv/50≈0.015. DV path already used obs-norm Δu.

## Next job (after P118 EXIT or hard-fail — not while LIVE)

**P119 `locgfix`** — same mechanism as P118 (LPV + local rest-IC G), **one bugfix**: MV plant-FD Δu = `_prev_cmd_norm` (WM-norm; P61 realized/rate-limit), never engineering `_prev_control`. Snapshot/restore `_prev_cmd_norm`. Print WARNING if |G_MV| << 5% of identified. Env-free. No new TrainConfig / no `DREAMER_*`.

Predicted signature: rest-ic local G MV same order as SysID (~2, OP span ~2× not 0.015); jsonl `gain_match_mv_ratio` ~O(1) not ±70; then GAIN-READY vs P117 CAPPED **0.71@DV**.

Falsifier: still CAPPED ~0.71@DV **after** local MV G is WM-norm.

Do **not** extra-P1 N+1 / identity-relaunch P118 / rewrite the live recipe.

After a fair P119 VALID or a fair GAIN_NOT_READY with a **correct** local teacher: return-to-`test_sim` on the worst residual (cadence: 2 jobs on this plant then return).

## Ranked follow-ups (not this GPU)

1. **P119 locgfix** (above) — causal for R1 on this plant.
2. If P119 GAIN-READY but val TM still short: keep LPV, audit rest-IC K vs 4τ settle, not extra-P1.
3. Return-to-`test_sim` for R2 headroom / critic_rew_to_tgt_var (P3 collapse is standing).
4. Observer successor (LPV inside RSSM beyond input scale) only if local-G teacher is WM-norm and still cannot represent equal-% DC.

## RCA this visit

- SysID: OP-varying DC is real (equal-% + 1/feed). Median G is still the wrong pin **once local G is in WM-norm**.
- Signal: P118 jsonl NaN/Inf 0; skip 0; GPU 14.7 GB; sps ~52 after iter 1.
- Control: do not score actor until freeze GAIN-READY and P3.
- ML: LPV `wm_op_scale_dev` left 0 (good) but Huber MV target is ~0.
- Plant: HeatExchangerTower `step` is engineering; APCEnv denorms. Teacher must use `_prev_cmd_norm`.
- Metric: jsonl MV ratio vs 0.015 is not val TM.

P117 process died mid-P2 (SIGKILL). Not a training crash. Do not identity-relaunch.
