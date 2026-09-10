# GOAL_PLAN — neural-APC-mbrl2

Living plan. Update every visit (live analysis and EXIT). Champions live in `docs/RUN_HISTORY.md`. **`RUN_HISTORY` appends at P1-gate / P2→P3 / EXIT only** — not LIVE@iter snapshots.

**Product:** simulator-agnostic neural APC — smooth CV on the economic limit without violating; faithful observer; unmeasured-load rejection. Envelope: learned observer + neural Kalman/DOB + neural actor-critic. No gray-box plant, no PID/LQR/MPC as the product, no DV-only FF.

Plant this cycle: **return-to-`test_sim`**. **P121 `srmed` LIVE P1** pid **588400** sha `4422be5` @iter **23**. P120 EXIT `[p3-skip]` INVALID (wrap-locked last_ok **85**). **P120 `gop` PARKED.** Do **not** score P120 as opscale or as an actor result. Extra-P1 lottery **closed**. Snapshot only until P1-gate.

## Residual board vs champions (from `validation/residual_board.json`)

| Axis | Score we trust | Champion | P116 test_sim (`best.pt`, VALID) | P120 test_sim (`final.pt`, INVALID) | P119 nonlinear_sim (`best.pt`, VALID) | Status |
|---|---|---|---|---|---|---|
| R1 TM MV ss/@H / curve | val TM + `curve_iae_normed` | P26 ×0.973 / @H ×0.880 | ss **×0.706** / @H **×0.711** curve **0.262**. lever=**autoencoder** | ss **×0.816** / @H **×0.852** curve **0.163**. H=74. lever=**free_bits**. Do **not** KEEP opscale | ss **×0.758** / @H **×0.922** curve **0.143**. vs P118 ×0.609 / 0.35 | OPEN vs P26; P120 mixed (sr+wrap) |
| R1 TM DV ss/@H / curve | same | P64 ×0.893 / @H ×0.962 | ×0.692 / ×0.712 | ×1.011 / ×1.060 curve **0.083** | ×0.942 / ×0.950 curve **0.080** | P116 OPEN; P120 DC-ok on wrap freeze ≠ READY |
| R1 compounding | 1step→OL | P64 ×0.85 | **×0.911** (better than P64) | **×1.095** OL **×0.779** post→1step **×0.584** | **×0.846** OL **×0.783** lever=compounding | test_sim family **closed** as champ; P120 prior lag |
| R2 CV smoothness | worst-seed d2≤0.05 AND rev≤0.25 | — | d2 **0.034** PASS; rev **0.421** FAIL. `smooth_pass=False` | d2 **0.034** PASS; rev **0.353** FAIL. expert-BC — **do not score** | d2 **0.053** FAIL; rev **0.395** FAIL. `smooth_pass=False` | OPEN (CV oscillation) |
| R2 headroom / viol | `cv_opt_headroom`, `cv_viol_frac` | — | econ=**hi** (G<0). head **0.183** / viol_frac **0.112**. Not mid-band | econ=**hi**. head **0.234** / viol **0.263**. expert-BC | econ=**lo** (G>0). head **0.116** / viol_frac **0.445** | P116 headroom OK; P119 viol OPEN |
| R3 Kalman | `det_r` + pred_std vs true | P26 det_r 0.68 | **0.676** / **1.48 vs 1.93** | **0.237** / **1.49 vs 1.33**. wrap freeze | **0.346** / **7.10 vs 3.07** | P116 near champ; P120 freeze miss |
| R3 return-to-limit | `cv_return_headroom` / `cv_return_time_frac` | — | ret_h **0.135** ret_t **0.022** (returns to hi) | ret_h **0.185** ret_t **0.112**. expert-BC | ret_h **0.146** ret_t **0.030** | score only on VALID actor |
| Actor econ | paired vs baseline **and** P64; VALID only | P64 −4.54 vs −94 | **−8.24 vs −109.83** 9/9 VALID | **−31.63 vs −159.86** expert-BC. **do not score** | **−32.11 vs −230.94** 9/9 VALID. `critic_r` FAIL | Do not swap P64 |

P119/P116 boards: R2/R3 return-to-limit from npz (JSON lacked those keys); R3 event IAE from JSON seed-medians (P119 **20.3**, not the npz overwrite 64.36). P120 board written live at val; TM json agrees with residual_board (MV ×0.816 / curve 0.163). Do **not** treat P120 lineage `wm_gain_healthy=True` as GAIN-READY.

## Metric audit (trusted vs on trial)

**Trusted:** val TM ss/@H/`curve_iae_normed` (MV and DV); gain decomp; `det_r` + pred_std vs true; paired econ vs baseline **and** champion when freeze GAIN-READY and P3 ran; **CV d2/reversal** (`cv_d2_rms_normed` + `cv_reversal_rate` — this **is** `smooth_pass` as of P120); `cv_opt_headroom` / `cv_viol_frac` with econ side = **sign(SysID MV→CV gain)** (G<0 hug hi / G>0 hug lo — not `cv_side_scale`); DR `cv_return_headroom` / `cv_return_time_frac` on that same bound. Skip-storm 5-level TM is freeze/RCA, not a champ score.

**On trial / do not GPU-optimize:** jsonl teacher ×1 vs SysID median; jsonl `gain_match_mv_ratio` vs local G (teacher pin ≠ GAIN-READY); ss-ratio alone; lineage `wm_gain_pass` / `wm_gain_healthy` (P120 HEALTHY rel_err **0.18** on a GAIN_NOT_READY freeze); **mv_reversal as a smoothness gate** (now diagnostic only; `mv_reversal_rate_observed` still logged); **train `mv_reversal_penalty` (`obj_auto_reversal_gain=0.3` default)** — val `smooth_pass` is CV d2/reversal, but the reward still quiets MV chatter the product allows. P116 FAIL was CV rev **0.421** with MV rev **0.235**; quieting MV did not sit a smooth CV on the limit. After a VALID observer, retarget this term to CV (#4) rather than N+1 the MV gain; CV total variation as a smoothness gate; critic_r without `critic_rew_to_tgt_var`; raw dist R²; event IAE without return-to-limit; VALID 9/9 / GAIN-READY / all_pass / family-closed as “residual closed”; beating P64 to KEEP a smoothness/headroom/DR win; **`cv_side_scale` as the economic riding bound** (it is violation urgency; P116 “headroom 0.82 / never returns” was hugging **hi** scored against **lo**); **`wm_op_scale_dev~0` as “linear plant stays identity”** (metric is mean `|1+tanh−1|`; last Linear is zero only at step-0; P120 P2 **~0.90**; P121 P1 **~0.70**); **P120 val TM ×0.816 / DV ×1.01 / curve 0.163 as an opscale KEEP or FALSIFY** (`sr=3 H=74` vs P116 `sr=4 H=54`; wrap-locked last_ok **85**; expert-BC); extra-P1@98 LIVE TM **0.43@MV** (`not_noisy=False`) ≠ freeze TM (last_ok **85**); residual_board event IAE from npz when JSON already has seed-median IAE (P119 board was **64.36** vs summary **20.3** — **fixed**; JSON IAE is the score).

## Closed families (do not N+1)

extra-P1 as freeze (P41; **P117**; **P118**); compounding-teacher Huber (P111 traj unpromoted, P115 k1 REVERT, P116 ol1 REVERT); isolation-off KEEP as default; 2TS-α as champ; kfeat as freeze; quiet_env as freeze; decoder FF; TSSM; GRU keep-h.

## P120 `tshome` — EXIT `[p3-skip]` INVALID (do not score actor / do not FALSIFY opscale)

tmux `mbrl2_p120` **gone**. pid **583595 DEAD**. sha **`03a93f1`**. **EXIT=0** 165 jsonl iters ES `p3_skipped_invalid_observer`. nvidia **0 MiB**. No `best.pt`. Val `final.pt` expert-BC. `actor_experiment_valid=false`. jsonl NaN/Inf **0**. Env-free. `[resolved-cfg] opscale=True` no ol1. Train start **2026-09-09 19:10:17**. `run_plan` **sr=3 H=74** `dead_dom=8` `dead_fast=6` `tau_fast=47`. locgfix `|du_mv|=step`. **Do not identity-relaunch.**

**P1 gates (lottery closed; freeze = last_ok 85 not READY):**
- **Orig-P1 @86 FAIL** last_ok **85** **0.70@MV** `@H 0.71` DV **1.00/@H 1.03** `not_noisy=True`. Wrap #3 live recon **0.0818 = 27×**.
- **Extra-P1 @98 FAIL** LIVE **0.43@MV** (`not_noisy=False`; recon 3.58×). last_ok **still 85** stay-locked (`gain_ready_locked=False`).
- **Cap @109 CAPPED** last_ok **85** **0.75@MV**; restore 85 re-probe **0.80@MV** `@H 0.84` DV **1.01/@H 1.04** still not READY. Warm-restore SKIPPED (`wm_best` @140 gain-blind). `[phase] 1→2` **943410**.

**P2 110–165 then `[p3-skip]`:** first dobg **0.0384** skip **0** KEEP vs P95. `dob_A` **0.95257** held. `dob_K` **0.018@110→0.056@165** SS **≈1.19**. leftover **0.77@110→0.55@165**. `std_ratio` **1.69e5@110 → 0.67@165**. α **0.00391**. `opdev` **~0.90**. P2→P3@**165 PASS** MTP **0.912** then **`[p3-skip]`**. last_ok **85** locked not-READY through EXIT.

**Val (do not score actor):** MV TM ss/@H **×0.816 / ×0.852** curve **0.163** (wm −0.261 vs real −0.320; H=**74**). DV **×1.011 / ×1.060** curve **0.083**. Lineage `wm_gain_pass` HEALTHY (rel_err **0.184 / 0.011**) / observer-gain HEALTHY ≠ GAIN-READY. Decomp MV real→post **×0.958** post→1step **×0.584** 1step→OL **×1.095** OL-vs-real **×0.779** lever=**free_bits**. DV real→post **×1.125** post→1step **×0.441** lever=prior. det_r **0.237** pred_std **1.49 vs 1.33**. Event IAE **23.2**. R2 d2 **0.034** PASS rev **0.353** FAIL `smooth_pass=False` head **0.234** viol_frac **0.263** econ=**hi**. `critic_r=nan`. Paired **−31.63 vs −159.86** 9/9 expert-BC. TM json agrees with residual_board.

**Judge:** `[p3-skip]` on wrap-locked last_ok **85** is **P40 stay-lock + orig-P1 miss + sr=3**, not an opscale verdict. Val TM did **not** collapse vs P116 ×0.706 (ss **better**, different H) — still **not** an opscale KEEP. extra-P1 lottery closed. Next GPU **P121 `srmed`**. Stay-lock-only-if-READY is **#2** if orig-P1 still wrap-locks after sr=4. Champion **P64**. Dist champ **P26**.

## P118 EXIT — `opscale` INVALID observer (do not score / do not FALSIFY opscale)

tmux `mbrl2_p118` **gone**. pid **551549 DEAD**. sha **`ff5f84c`**. **EXIT=0** 166 jsonl iters ES `p3_skipped_invalid_observer`. nvidia **0 MiB**. No `best.pt`. Val `final.pt` expert-BC. `actor_experiment_valid=false`. jsonl NaN/Inf = **one** `wm_grad_norm=inf@13`. Env-free. `[resolved-cfg] opscale=True` no ol1. Local G MV **0.0153** vs ident **2.627** (engineering `_prev_control` Δu). **Do not identity-relaunch.**

**P1 gates:** orig-P1@**87 FAIL** last_ok **85** **0.34@MV**; extra-P1@**98 FAIL** live **0.11@MV**; cap@**110** last_ok **108** **GAIN_NOT_READY 0.68@MV / 0.85@DV** `@H[0.77,0.79]` `not_noisy=True`. Detonated-freeze restored 108. Warm-restore SKIPPED (`wm_best` @100 gain-blind). Graph released.

**P2 111–166:** first dobg **0.0209** skip **0** KEEP vs P95. `dob_A` **0.95257** held. `dob_K` **0.119→0.017@111→0.0018@117** min **0.0016@119** then recovered **0.047@166** SS **≈1.00**. leftover `|d_slow|/|d|` **0.80@111→1.53@117→0.416@166**. `std_ratio` **144389@111 → 0.77@166**. α **0.00391**. `wm_op_scale_dev` **~0.93**. P2→P3@**166 PASS** MTP **1.216** then **`[p3-skip]`**.

**Val (do not score actor):** MV TM ss/@H **×0.609 / ×0.705** curve IAE/\|real\| **0.35**. DV **×0.799 / ×0.703** curve **0.24**. Lineage `wm_gain_pass` PASS (rel_err 0.39 / 0.20) / `wm_gain_healthy=False` / observer-gain PASS ≠ GAIN-READY. Decomp MV real→post **×0.882** post→1step **×0.846** 1step→OL **×1.045** OL-vs-real **×0.607** lever=autoencoder. DV real→post **×0.868** post→1step **×0.827** lever=autoencoder. det_r **0.343** pred_std **3.00 vs 3.07**. Event IAE median-of-medians **29.2**. `critic_r=nan`. `all_pass=False` (`critic_pass`). Paired **−38.39 vs −212.28** 9/9 expert-BC.

**Judge:** locgfix **already on HEAD** (`414eca4`). **Not** an opscale FALSIFY. extra-P1 lottery closed. Next was **P119 `locgfix`** (now EXIT VALID). Champion **P64**. Dist champ **P26**.

## P119 EXIT — `locgfix` VALID (do not identity-relaunch / do not launch `gop`)

tmux `mbrl2_p119` **gone**. pid **557257 DEAD**. sha **`ea7def9`**. **EXIT=0** 406 jsonl iters ES `p3_plateau` (no >+1.0% over `best_det_return=−131.981` for 200 iters) best **206**. env_steps **1645000**. nvidia **0 MiB**. `best.pt` + `final.pt`. Val ckpt **`best.pt`**. `actor_experiment_valid=true`. jsonl NaN/Inf **0**. Env-free (`CUDA_VISIBLE_DEVICES=0`, no `DREAMER_*`). `[resolved-cfg] opscale=True` no ol1. Train start **2026-09-09 14:10:33**. **`[p3-skip]` did not fire.** P3 **159–406** (248 iters) skip_iter **0** (cum skip **11** is P1 storm-1). Critic-warmup 159–168; `[return-scale] FREEZE` **2.128** @169; unfreeze @169. @406 actor **+0.116** critic **9.51** ent **−0.467** logp **0.54** clip **0.10** rscale **2.13** rtgt **0.00067** `critic_pred_target_r` **0.991**. **Do not identity-relaunch.** **Do not launch P120 `gop`** (val TM **×0.758**, not ~P118 ×0.61). Next was **P120 `tshome`** (now EXIT `[p3-skip]`).

**Teacher print (locgfix CONFIRMED):** local G `plant_fd=6/6` mean MV **2.890** / DV **−0.341** vs identified **2.709 / −0.435**. span MV **2.01** / DV **0.46**. `|du_mv|=0.4000` **=** step **0.4000**. **No** WARNING. jsonl MV O(1) since ~@5.

**Orig-P1 @91 FAIL (fair locgfix freeze miss):** live g **0.63@MV** `@H 0.61` DV **1.03/@H 0.98**. jsonl MV **×0.991** ≠ TM. extra-P1 lottery **closed**.

**Extra-P1 @101 PASS (not a freeze KEEP):** last_ok **100** GAIN-READY **0.83@MV** `@H 0.98` DV **0.92/@H 0.95** `compound_ok` 1step→OL **0.86**. Detonated-freeze restore **100**; re-probe **0.85@MV** 1step→OL **0.90**. Warm-restore SKIPPED. `[phase] 1→2` env_steps **908750**. P100-class confound.

**P2→P3 @158 PASS:** MTP median **1.180** (max=3.00 n=5) `p2_extension=0/42897`. env_steps **1276250**. P2 skip **0**. End-P2 recon **0.0527** dobg **0.0319** `dob_A` **0.95257** `dob_K` **0.167** SS **≈3.52** leftover **0.159** std **0.885**. First dobg **0.0034@102** skip **0** KEEP vs P95. vs P118 leftover **0.416** K **0.047/1.00** MTP **1.216** then `[p3-skip]`.

**Val (score actor — VALID):** MV TM ss/@H **×0.758 / ×0.922** curve IAE/|real| **0.153** (wm +0.693 vs real +0.914). DV **×0.942 / ×0.950** curve **0.080** (wm −0.390 vs real −0.414). Lineage MV rel_err **0.24** HEALTHY; DV **0.06** HEALTHY; `wm_observer_gain` HEALTHY. Decomp MV real→post **×0.960** post→1step **×0.962** 1step→OL **×0.846** OL-vs-real **×0.783** lever=**compounding**. DV real→post **×0.895** post→1step **×1.096** lever=autoencoder. det_r **0.346** R²_det **−4.27** pred_std **7.10 vs true 3.07**. Event IAE median-of-medians **20.3** (P118 **29.2**). Paired **−32.11 vs −230.94** **9/9 WIN**. cum_raw **−55585** (36). `critic_r` **+0.253** FAIL (<0.3) → `all_pass=False`. `smooth_pass=True` (mv_reversal **0.171** all-eps / **0.196** DR). hugging **0.384** / **0.316** DR; cv_viol **16.7**; mv_viol **2.79**. No `cv_d2_rms_normed` / `cv_opt_headroom` / `cv_viol_frac`. Val “[val] P3: 11 grad-clip skips” is **cumulative** P1 storm, not P3 this-iter.

**Judge:** ✅ KEEP locgfix as **teacher-unit fix** vs P118 0.015 / val ×0.609. ✅ KEEP as **P3-entry** vs P118 `[p3-skip]`. ❌ extra-P1 as freeze KEEP (orig-P1 **0.63@MV** is the fair miss; freeze last_ok **100** 0.83@MV transferred to val ×0.758, not P109-class collapse). ❌ **FALSIFIED as TM-to-P26** (×0.758 vs ×0.973). Remaining R1 hole is **compounding**. ❌ **FALSIFIED as Kalman amp** (pred_std 7.10 vs 3.07; det_r ~P118 0.343). Actor **VALID** vs baseline; **do not swap P64** (different plant). **P120 `gop` PARKED** — trigger was val TM still ~×0.61; it is ×0.758. Cadence: fair plant verdict done. GPU **free**. Do **not** extra-P1 N+1 / identity-relaunch P117/P118/P119 / rewrite locgfix / in-place `c*=gop`.

## Ranked follow-ups

**#1 P121 `srmed` — LIVE P1 @iter 23** pid **588400** sha `4422be5` tmux `mbrl2_p121` out-dir `output/test_sim/run_p121_srmed`. Env-free `CUDA_VISIBLE_DEVICES=0` (proc env: CVD=0 + `PYTORCH_CUDA_ALLOC_CONF` only; no `DREAMER_*`). Train start **2026-09-10 01:06:11** `device=cuda` bs=128 compile=eager. **`run_plan` sr=4 H=55** `sample_rate_source=auto:tau_fast/10_or_dead_dom/2`. This ID draw `dead_fast=7` `tau_fast=45` (old min-θ formula would also have been sr=4 this draw; source string confirms the new path; P120-class θ=6 is what the formula closes). `[resolved-cfg] opscale=True` gmatch_len=55. locgfix CONFIRMED `|du_mv|=0.4000=step`. Teacher local G MV **−2.53** vs ident **−2.57**; DV rest-IC **0.297** vs ident **0.581** (span 0.13 — not locgfix). Snapshot **2026-09-10 01:53**: jsonl **23** NaN/Inf **0** skip **0** recon **0.081** after wrap. jsonl MV teacher **×0.999** (tautology). `wm_op_scale_dev` **0.69–0.75** (on trial). Fidelity @20 H=55 r=+0.54 gain_fid=0.842. **Orig-P1 wrap lock@21** recon **0.326** (49× best **0.0066**) then **unlock@23** recon **0.081** — P116-class (`extra_p1=False`); P116 wrap@39 unlock@~42. Not extra-P1 stay-lock. Not a kill. ~2 min/iter; orig-P1 still ~@87. Do **not** score actor / stay-lock on this pid / wait for jsonl.
- **Mechanism:** `derive_all` now calls `derive_sample_rate(tau_fast, dead_dom)` with `sr_source='auto:tau_fast/10_or_dead_dom/2'`. APC delay sampling is ~2 samples in the *channel* θ, not the noisiest FOPDT repeat. P120 binding constraint was `dead_fast=6` from REFLUX r4; `dead_dom=8` already matches `per_mv_dynamics`. Keep `dead_time_fastest=min` for lookback / diagnostics. Do **not** also median-`tau_fast` (τ lottery is not binding: `round(47/10)=round(53.5/10)=5`). No `DREAMER_SAMPLE_RATE=4` pin.
- **Predicted signature:** test_sim `sr=4` `H~54` `gmatch_len~54`; `[resolved-cfg]` opscale still ON; orig-P1 timing P116-class (~87 **unlocked**, not wrap-coincident).
- **Falsifier:** `run_plan` sr still 3; **or** a real fast channel is undersampled (lookback/TM @H worse than P116 with no other change). Orig-P1 still wrap-locked FAIL after sr=4 → then **#2**, not opscale REVERT.
- **Files:** `utils/plant_init.py`; smoke `_test_sample_rate_uses_dead_dom_not_min`. Lookback `derive_sample_rate_for_lookback` KEEP.
- **Why this before opscale KEEP/REVERT and before stay-lock:** P120 mixes opscale + sr + wrap-at-gate. sr=4 may restore P116-class orig-P1 **unlocked PASS** without touching P40.

**#2 Extra-P1 stay-lock only if last_ok is GAIN-READY (after P121, only if orig-P1 still wrap-locks / `[p3-skip]` on a not-READY last_ok).** Align wrap stay-lock with skip-storm P92. **P120 GPU-confirmed:** extra-P1@98 LIVE recon healthy but last_ok **stuck at 85**; cap restored that miss. **P121 live:** orig-P1 wrap@21 **unlocked@23** — current `extra_p1=False` recovery KEEP. Do not land #2 to “fix” that wrap.
- **Mechanism:** `_should_lock_last_ok` `if already_locked: if extra_p1 or gain_ready_locked: return True`. Extra-P1 recovered basin stay-locks **even when** `gain_ready_locked=False` (P120). Change: `if already_locked and extra_p1: return bool(gain_ready_locked)` then fall through to recon-vs-`lock_ratio` (same as orig-P1). Orig-P1 wrap-recovery unlock KEEP when not READY. P40 overwrite of a **READY** last_ok still stay-locks via `gain_ready_locked`.
- **Predicted signature:** wrap@gate FAIL then extra-P1 last_ok walks (P119-class).
- **Falsifier:** P40 overwrite of a **READY** last_ok — must still stay-lock when the locked snapshot **was** GAIN-READY. Smoke (add next to P40 extra-P1 lock tests in `tools/_smoke_rssm.py`): `extra_p1 + already_locked + gain_ready_locked=False + recon<20×` **unlocks**; same with `gain_ready_locked=True` **stays** locked. Existing P40 test (`extra_p1=True`, no `gain_ready_locked`) currently asserts lock — rewrite it to pass `gain_ready_locked=True` (READY) vs False (P120 not-READY).
- **Files:** `training/train.py` `_should_lock_last_ok`; smoke in `tools/_smoke_rssm.py`. No new TrainConfig / no `DREAMER_*`. **Do not** land this on the P121 pid (one attributed change). **Do not** treat as extra-P1-as-freeze N+1.

**#3 After `srmed` VALID:** score opscale vs P116 R1 TM (autoencoder ×0.706). If TM << P116 with sr matched → gate `op_scale_net` when rest-IC `|G|` span ≲ 1.3× (linear identity; SysID-derived). If TM ≥ P116 → KEEP opscale as default even on linear. **Do not** identity-regularizer λ lottery.

**#4 R2 CV reversal (after a VALID observer on this plant — P121 EXIT GAIN-READY, not expert-BC).** One mechanism: retarget the existing reversal term from MV to CV. Today `obj_auto_reversal_gain=0.3` builds `mv_reversal_weights` and `objective_runtime` penalises `relu(-du_t·du_prev)` (MV chatter). Val `smooth_pass` already ignores MV. P116 VALID FAIL is CV rev **0.421** (d2 **0.034** PASS) while hugging hi (`econ_side=hi`, head **0.183**, viol **0.112**) — the MV term is the wrong loss for the open residual.
- **Mechanism:** disable MV reversal in the reward (gain 0 or drop `mv_reversal_weights`); add a CV analogue `relu(-dCV_t·dCV_{t-1})` (and/or a unitless d2) scaled as a fraction of `cv_base` and hard-capped at `cv_base` so econ+viol still dominate. Same `_knob_float` path; leftover `OBJ_AUTO_REVERSAL_GAIN` dual-read **delete** when touching. No new TrainConfig name if the existing gain is retargeted.
- **Predicted signature:** worst-seed `cv_reversal_rate` ≤ 0.25 with d2 still ≤ 0.05; `cv_opt_headroom` not worse (still on the economic bound); `cv_return_headroom` / `cv_return_time_frac` not slower. MV reversal may rise (allowed).
- **Falsifier:** mid-band sit (`cv_opt_headroom` up) or slow DR return; or CV still oscillating because the term is tautological with viol. Do **not** stack critic knobs / rtgt / `mv_reversal` N+1.
- **Files:** `utils/auto_weights.py`, `utils/objective_runtime.py`, `tools/_test_mv_reversal.py` (CV cases). Env-free. **Do not** land on the P121 pid.

**#5 P120 `gop` (PARKED)** — **do not launch**. P119 val TM **×0.758** is not ~×0.61. P73 persist forbids in-place `c*=gop`.

**#6 After `srmed` if a later plant has a real fast DV:** min of *per-channel* median θ, not global median of every repeat. Do **not** launch this while test_sim channel medians are all 8.

**Do not:** extra-P1 N+1, ol1 N+1, identity-relaunch P117/P118/P119/P120, launch `gop`, stack critic knobs, second GPU, sr pin as engineering default, stay-lock on this pid.

## RCA this visit

- SysID: test_sim **θ_dom=8**. P121 `dead_fast=7` `tau_fast=45` → sr=4 H=55 (source `dead_dom`). locgfix `|du_mv|=step`. Rest-IC DV local G **0.30 vs ident 0.58** is rest-IC vs SysID, not srmed.
- Signal: P121 @23 skip **0**, no Inf, ~119 s/iter after compile. Wrap lock@21 (49×) **unlocked@23** (`extra_p1=False`). P116 wrap@39 also unlocked — expected first-fill jitter, not P120 stay-lock. Disk `/home` 64% / 61G. Keep P121+P120+P119+P118+P117+P116+P64+P53.
- Control: do not score actor. R2 still open on P116 (CV rev 0.421). Train still penalises MV reversal.
- ML: jsonl MV ×1 @23 is teacher pin. `opdev` **~0.70** is not identity (on trial). `wm_best` @20 gain-blind. Orig-P1 wrap-unlock KEEP this pid so far.
- Plant: env-free HEAD. opscale ON vs P116 still unscored until EXIT val at matched sr.
- Metric: do not GPU-optimize jsonl ×1 / `opdev` / MV reversal gate. Trusted R2 is CV d2/reversal + headroom on `econ_side=sign(G)`.
- Literature (docs/papers PDFs not on host; HTTPS blocked): delay-dominant sampling is 2+ samples in the **channel** dead time. Economic APC allows MV move; penalising MV reversal in the reward fights limit-hugging. Stay-lock-if-READY is P92 applied to extra-P1, not orig-P1 wrap.

Config audit Step 4 (P121 live @23): env-free HEAD CONFIRMED (proc: CVD=0 only). `run_plan` sr=**4** H=**55** source `dead_dom`. `[resolved-cfg] opscale=True`. locgfix `|du_mv|=step`. Wrap@21 unlocked@23. #❌ **0**. #🆕 **0**. Do not score TM/actor until P1-gate / EXIT. Step 5 (P120): opscale **unscored**; extra-P1 **closed**; `[p3-skip]` KEEP as validity gate.

P117 process died mid-P2 (SIGKILL). P118 completed P2 then `[p3-skip]`. P119 completed P2 then **P3 then EXIT VALID**. P120 completed P2 then **`[p3-skip]`**. Do not identity-relaunch P117–P120.
