# GOAL_PLAN — neural-APC-mbrl2

Living plan. Update every visit (live analysis and EXIT). Champions live in `docs/RUN_HISTORY.md`.

**Product:** simulator-agnostic neural APC — smooth CV on the economic limit without violating; faithful observer; unmeasured-load rejection. Envelope: learned observer + neural Kalman/DOB + neural actor-critic. No gray-box plant, no PID/LQR/MPC as the product, no DV-only FF.

Plant this cycle: `nonlinear_sim` HeatExchangerTower (P118 = job 2 of 2, **LIVE P1**; P119 is the fair repeat of the same mechanism after the MV-Δu unit bug; then return-to-`test_sim`).

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

**Trusted:** val TM ss/@H/curve_iae (MV and DV); gain decomp real→post / post→1step / 1step→OL; `det_r` + pred_std vs true; paired econ vs baseline **and** champion when freeze GAIN-READY and P3 ran; CV d2/reversal (smooth_pass); `cv_opt_headroom` / `cv_viol_frac`. Skip-storm 5-level TM (`DCgain_ratio` / `@H` / `compound_ok`) is a freeze/RCA diagnostic, not a champ score. Launch `|du_mv|` vs teacher `step` (P119 print): O(step) = WM-norm; ≫5×step = engineering-unit bug.

**On trial / do not GPU-optimize:** jsonl teacher ×1 vs SysID median (tautology on OP-varying plants — P117); **P118 jsonl `gain_match_mv_ratio` vs local G** (teacher Δu was engineering `_prev_control` − WM-norm action → local MV G **0.015** vs identified **2.63**; late-P1 ratio ~0.58 is still vs G≈0.015, not TM); ss-ratio alone; mv_reversal as smoothness gate; CV total variation as smoothness gate; critic_r without `critic_rew_to_tgt_var`; raw dist R²; VALID 9/9 / GAIN-READY / all_pass / family-closed as “residual closed”; beating P64 to KEEP a smoothness/headroom/DR win; **`wm_grad_norm=inf` at a skip-storm restore** (P118@13) as a kill; **recon-spike ~3 iters after P1 inject** as a kill (13/23/33/45 cadence); **persist_rel spikes to 31** as a kill (gain-c fighting G_tgt≈0); **H=1 fidelity r** as a freeze score (`wm_best` @10 is gain-blind).

P118 live: DV local G **−0.345** vs ident **−0.426** (same space, OK). MV local G is the unit bug, not equal-% OP variation (span **0.015** would be ~2× if the 2.23× SysID story were in WM-norm). LPV `wm_op_scale_dev` **0.52→0.99** then pinned **~0.87** while Huber MV target ≈0 — do not score LPV from this pid.

## Closed families (do not N+1)

extra-P1 as freeze (P41; **P117 on this plant**); compounding-teacher Huber (P111 traj unpromoted, P115 k1 REVERT, P116 ol1 REVERT); isolation-off KEEP as default; 2TS-α as champ; kfeat as freeze; quiet_env as freeze; decoder FF; TSSM; GRU keep-h.

## Live this visit — P118 `opscale` (do not kill / do not relaunch / do not second GPU)

tmux `mbrl2_p118` pid **551549** sha **`ff5f84c`** `device=cuda` bs=128 compile=eager nvidia **~14751 MiB**. `[resolved-cfg] opscale=True` no ol1. STAGE 1 `g=84 dob=8`. Rest-IC graph captured N=6 T=128. sps **~52**. jsonl **55** P1. Heartbeat ~2 min/iter. orig-P1 budget **796080** steps (~iter 82–87 class). Process 100% CPU, `wm_last_ok` walked, **not** locked.

**Teacher print (launch):** local G mean MV **0.0153** / DV **−0.345** vs identified **2.627 / −0.426**. jsonl MV ratio still wild early (**−74…+16**) then late **~0.25–0.58** (still vs G≈0.015). DV ratio **~0.7–1.1** (storm@13 DV **2.11**; spike@45 DV **0.27**). `wm_op_scale_dev` **0.52→0.99@18–22** then **~0.87**. recon recovered **0.050@50** (best **0.030@44**). skip **0** except storms. `wm_best` still **iter 10** (gain-blind). Probe@50 H=1 r=**+0.096** H=56 r=**+0.339** gain_fid=0.904 `best_h=0/56`.

**Skip-storm 1 @iter 13 (RCA, not kill):** recon **0.676** `wm_grad_norm=inf` skip **56/56**. Probe median 3/3 worsts **[0.60, 0.33, 0.54]** DC **[0.55, 0.86]** `@H[0.40, 0.95]` worst **0.55@DV**. Restored `wm_last_ok` iter **12**. **cap-deferred** (`ready_n=0`). Inf is only `wm_grad_norm@13`. jsonl NaN/Inf otherwise **0**.

**Post-storm inject cadence (RCA, not kill):** recon spikes **@23 / 33 / 45** (~3 iters after dv-prbs/const/step/expert injects @20/30/38–40). skip 1 @33 and @46 (not a second storm). persist_rel **31@36**, **10@43** then recovered. zrank dipped ~527@48 then alive recovered. Same unit-bug story: chasing G_MV≈0.015 detonates after a buffer refresh.

**Do not score actor. Do not treat opscale as FALSIFIED** — MV teacher was not WM-norm. Chasing G_MV≈0.015 detonates grads; LPV saturating is the same story.

HEAD (not this pid): `_wm_norm_realized_du` + snapshot/restore `_prev_cmd_norm` including **None**; launch prints `|du_mv|` vs teacher `step` and WARNINGs if G_MV≪ident **or** `|du_mv|`≫5×step. CVD="" locgfix smoke **PASSED**.

## Next job (after P118 EXIT or hard-fail — not while LIVE)

**P119 `locgfix`** — same mechanism as P118 (LPV + local rest-IC G), **one bugfix**: MV plant-FD Δu = `_prev_cmd_norm` (WM-norm; P61 realized/rate-limit), never engineering `_prev_control`. Snapshot/restore `_prev_cmd_norm` **including None**. Print WARNING if |G_MV| ≪ 5% of identified **or** `|du_mv|` ≫ 5× teacher step. Env-free. No new TrainConfig / no `DREAMER_*`. Tag `locgfix`. Session `mbrl2_p119`. Out-dir `output/nonlinear_sim/run_p119_locgfix`.

Predicted signature: rest-ic local G MV same order as SysID (~2, OP span ~2× not 0.015); `|du_mv|` ~ teacher step **0.4** not ~50; **no** WARNING; jsonl `gain_match_mv_ratio` ~O(1) not ±70; `wm_op_scale_dev` not pinned at tanh sat; skip-storm Inf@13-class and inject-cadence recon spikes absent or rare; persist_rel not 31; then GAIN-READY vs P117 CAPPED **0.71@DV**.

Falsifier: still CAPPED ~0.71@DV **after** local MV G is WM-norm (WARNING absent; `|du_mv|`~step; jsonl MV ratio O(1)).

Do **not** extra-P1 N+1 / identity-relaunch P118 / rewrite the live recipe.

After a fair P119 VALID or a fair GAIN_NOT_READY with a **correct** local teacher: return-to-`test_sim` on the worst residual (cadence: 2 jobs on this plant then return). P118 is job 2 of 2 **but** the unit bug means P119 is the fair repeat of the same mechanism, not a third plant-hop.

## Ranked follow-ups (not this GPU)

1. **P119 locgfix** (above) — causal for R1 on this plant. HEAD is launch-ready.
2. If P119 GAIN-READY but val TM still short: keep LPV, audit rest-IC K vs 4τ settle (this plant K=H=56 ≈ 4τ/sr = 54; settle L=128 is the encode, not the hole). Not extra-P1.
3. If P119 still CAPPED ~0.71@DV **with** WM-norm local G: input-scale LPV is too thin for equal-% DC. **P120 `gop`** (still neural, one mechanism): OP-conditioned **gain-c**, not another `1+tanh` on GRU inputs.
   - **Mechanism:** after `obs_step`/`img_step` produce `c`, `c[..., :cont_gain_dim] = c[..., :cont_gain_dim] * (1 + tanh(MLP(stop-grad OP)))` with last Linear zero-init (step-0 ≡ P119). Scale **gain-c only**. Decoder sees scaled gain-c so DC G(op) lives in the observer. `op_scale_net` LPV KEEP (no N+1 of that net). Group `g`. No new TrainConfig / no `DREAMER_*`. Banner `gop=True`. jsonl `wm_gain_op_dev`.
   - **Why not more input-scale:** equal-% DC is G=G(u) on the **output gain**. P118 `wm_op_scale_dev` sat at ~0.99 chasing G_tgt≈0 — that is not a test of gain-c G(op). Decoder FF closed (P74). Gray-box `G=k·valve%` is out of envelope.
   - **Predicted signature:** local G span ~2× (SysID); freeze GAIN-READY vs P117 0.71@DV; val TM ss closer across OP; `wm_gain_op_dev` moves off 0.
   - **Falsifier:** still CAPPED ~0.71@DV with locgfix teacher **and** a G(op) net that actually moved (`|(scale−1)|` not ~0).
   - **Files:** `models/dreamer_v4_rssm.py` (apply after c update), TSSM parity, `training/train.py` jsonl, `tools/_smoke_rssm.py`.
4. Return-to-`test_sim` for R2 headroom / critic_rew_to_tgt_var (P3 collapse is standing) after a fair plant verdict.

## RCA this visit

- SysID: OP-varying DC is real (equal-% + 1/feed; MV amp **2.23×** / DV **2.34×**). Median G is still the wrong pin **once local G is in WM-norm**.
- Signal: P118 jsonl NaN/Inf = **one** skip-storm `wm_grad_norm=inf@13`; recon recovered after inject-cadence spikes; GPU 14.7 GB; sps ~52; heartbeat 2 min/iter.
- Control: do not score actor until freeze GAIN-READY and P3.
- ML: LPV `wm_op_scale_dev` left identity then sat (~0.99) chasing a ~0 MV Huber target. Inf grad @13 and persist_rel 31 are teacher-unit, not RSSM collapse. `wm_best` @10 is gain-blind (H=1 r=+0.096@50) — do not restore at P2.
- Plant: HeatExchangerTower `step` is engineering; APCEnv denorms. Teacher must use `_prev_cmd_norm`. Restore must not leave a post-FD leftover when the snapshot was `None`.
- Metric: jsonl MV ratio vs 0.015 is not val TM. Skip-storm 0.55@DV is a live TM probe on a wrong-teacher WM — RCA, not a P119 falsifier.

P117 process died mid-P2 (SIGKILL). Not a training crash. Do not identity-relaunch.
