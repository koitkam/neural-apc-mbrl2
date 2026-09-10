# GOAL_PLAN — neural-APC-mbrl2

Living plan. Update every visit (live analysis and EXIT). Champions live in `docs/RUN_HISTORY.md`. **`RUN_HISTORY` appends at P1-gate / P2→P3 / EXIT only** — not LIVE@iter snapshots.

**Product:** simulator-agnostic neural APC — smooth CV on the economic limit without violating; faithful observer; unmeasured-load rejection. Envelope: learned observer + neural Kalman/DOB + neural actor-critic. No gray-box plant, no PID/LQR/MPC as the product, no DV-only FF.

Plant this cycle: **return-to-`test_sim`** (**P120 `tshome` LIVE**). Cadence: P117+P118 were the two `nonlinear_sim` jobs; P119 was the fair locgfix repeat. **P120 `gop` stays PARKED.** Do **not** score P120 as a pure opscale A/B (sr/H drifted).

## Residual board vs champions (from `validation/residual_board.json`)

| Axis | Score we trust | Champion | P116 test_sim (`best.pt`, VALID) | P119 nonlinear_sim (`best.pt`, VALID) | Status |
|---|---|---|---|---|---|
| R1 TM MV ss/@H / curve | val TM + `curve_iae_normed` | P26 ×0.973 / @H ×0.880 | ss **×0.706** / @H **×0.711** curve **0.262**. lever=**autoencoder** | ss **×0.758** / @H **×0.922** curve **0.143**. vs P118 ×0.609 / 0.35 | OPEN vs P26; P119 KEEP vs P118 |
| R1 TM DV ss/@H / curve | same | P64 ×0.893 / @H ×0.962 | ×0.692 / ×0.712 | ×0.942 / ×0.950 curve **0.080** | P116 OPEN; P119 DV HEALTHY vs P64 |
| R1 compounding | 1step→OL | P64 ×0.85 | **×0.911** (better than P64) | **×0.846** OL **×0.783** lever=compounding | test_sim family **closed**; nonlinear OPEN |
| R2 CV smoothness | worst-seed d2≤0.05 AND rev≤0.25 | — | d2 **0.034** PASS; rev **0.421** FAIL. `smooth_pass=False` | d2 **0.053** FAIL; rev **0.395** FAIL. `smooth_pass=False` | OPEN (CV oscillation). Suite `smooth_pass` is now this gate |
| R2 headroom / viol | `cv_opt_headroom`, `cv_viol_frac` | — | econ=**hi** (G<0). head **0.183** / viol_frac **0.112**. Not mid-band | econ=**lo** (G>0). head **0.116** / viol_frac **0.445** | P116 headroom OK; P119 viol OPEN |
| R3 Kalman | `det_r` + pred_std vs true | P26 det_r 0.68 | **0.676** / **1.48 vs 1.93** | **0.346** / **7.10 vs 3.07** | P116 near champ; P119 amp FALSIFIED |
| R3 return-to-limit | `cv_return_headroom` / `cv_return_time_frac` | — | ret_h **0.135** ret_t **0.022** (returns to hi) | ret_h **0.146** ret_t **0.030** | P116 IAE-without-return was a **wrong-bound** artefact |
| Actor econ | paired vs baseline **and** P64; VALID only | P64 −4.54 vs −94 | **−8.24 vs −109.83** 9/9 VALID | **−32.11 vs −230.94** 9/9 VALID. `critic_r` FAIL | Do not swap P64 |

P119/P116 boards: R2/R3 return-to-limit from npz (JSON lacked those keys); R3 event IAE from JSON seed-medians (P119 **20.3**, not the npz overwrite 64.36). P120 val writes the board live. TM files agree with `[val]` plot lines.

## Metric audit (trusted vs on trial)

**Trusted:** val TM ss/@H/`curve_iae_normed` (MV and DV); gain decomp; `det_r` + pred_std vs true; paired econ vs baseline **and** champion when freeze GAIN-READY and P3 ran; **CV d2/reversal** (`cv_d2_rms_normed` + `cv_reversal_rate` — this **is** `smooth_pass` as of P120); `cv_opt_headroom` / `cv_viol_frac` with econ side = **sign(SysID MV→CV gain)** (G<0 hug hi / G>0 hug lo — not `cv_side_scale`); DR `cv_return_headroom` / `cv_return_time_frac` on that same bound. Skip-storm 5-level TM is freeze/RCA, not a champ score.

**On trial / do not GPU-optimize:** jsonl teacher ×1 vs SysID median; jsonl `gain_match_mv_ratio` vs local G (teacher pin ≠ GAIN-READY); ss-ratio alone; lineage `wm_gain_pass`; **mv_reversal as a smoothness gate** (now diagnostic only; `mv_reversal_rate_observed` still logged); CV total variation as a smoothness gate; critic_r without `critic_rew_to_tgt_var`; raw dist R²; event IAE without return-to-limit; VALID 9/9 / GAIN-READY / all_pass / family-closed as “residual closed”; beating P64 to KEEP a smoothness/headroom/DR win; **`cv_side_scale` as the economic riding bound** (it is violation urgency; P116 “headroom 0.82 / never returns” was hugging **hi** scored against **lo**); **`wm_op_scale_dev~0` as “linear plant stays identity”** (metric is mean `|1+tanh−1|`; last Linear is zero only at step-0; P120 P1 **0.68–0.93** on test_sim); **P120 val TM / wrap recon / wrap-duration / live probe (locked *or* post-unlock @60) as an opscale FALSIFY** until sample-rate matches P116 (this pid `sr=3 H=74` vs P116 `sr=4 H=54`; first-fill wrap unlocked@56, last_ok walking@67); residual_board event IAE from npz when JSON already has seed-median IAE (P119 board was **64.36** vs summary **20.3** — **fixed** prior visit; JSON IAE is the score).

## Closed families (do not N+1)

extra-P1 as freeze (P41; **P117**; **P118**); compounding-teacher Huber (P111 traj unpromoted, P115 k1 REVERT, P116 ol1 REVERT); isolation-off KEEP as default; 2TS-α as champ; kfeat as freeze; quiet_env as freeze; decoder FF; TSSM; GRU keep-h.

## P120 `tshome` — LIVE (mode 2 snapshot; do not score actor)

**Live:** tmux `mbrl2_p120`. pid **583595** (CPU 100%, nvidia **19255 MiB**, A10). sha **`03a93f1`**. out-dir `output/test_sim/run_p120_tshome`. Env-free (no `DREAMER_*` in pid environ). Train start **2026-09-09 19:10:17**. Heartbeat jsonl **@69** `2026-09-09T21:54` (~2.3 min/iter, `t_wm` **136–138 s**, sps **~44**). STAGE 1. skip **0**. jsonl NaN/Inf **0**. Periodic `ckpt_iter_{40,60}` (HEAD keep=2). `wm_best.pt` still @**30** (fidelity EMA **4.718**; P116 moved best @60/70). Graph `N=6 T=128`. **Not a hard failure — do not kill / second GPU / rewrite recipe.**

**Teacher / resolved:** `[resolved-cfg] opscale=True` **no** ol1. locgfix `|du_mv|=0.4000`**=** step; rest-IC local G MV **−2.54** vs ident **−2.55** span **0.13**. P116 `t_wm` **~99 s** / sps **~59**. P1 env budget **782280** (~**81** iters at 9610 steps/iter) vs P116 orig-P1@**87** (phases **776820**). Gate timing is another sr confound.

**Launch confound (not opscale):** `run_plan` **sr=3 H=74** `gmatch_len=74` vs P116 **sr=4 H=54**. Identifier 16 pairs: **θ median=8** (12×8, 2×9, 1×7, **1×6**). θ=6 **and** τ=47 (`tau_fastest`) are the **same MV REFLUX r4** pair, not a fast DV (all DV θ=8). `derive_sample_rate(tau_fast=47, dead_fast=6)` → `min(round(47/10)=5, round(6/2)=3)` = **3**. Median θ → `round(8/2)=4` (τ still 5, not binding). P116 min θ=7 → sr=4. Channel medians in `per_mv_dynamics` / `per_cv_dynamics` already **θ=8**.

**P1 wrap (P42/P49; unlocked this visit):** buf fill **1.00 @32**. last_ok **locked 36** recon **0.0030→0.5417** (20× of best **0.00305**). Peak **0.772 @38** alive **886**. Secondary spike **0.587 @49** while *still the same lock* (not a second lock event). **Unlocked @56** recon **0.0588 < 0.061**. P50 5× overwrite held last_ok **36** through @66 (recon 0.016 still ~5×); **walking 67→69** once recon **0.0127**. Disk `wm_last_ok.pt` mtime is the **lock write @36** (`_persist_last_ok_ckpt` is lock-time; **RAM last_ok is the freeze source**). P116 first wrap: lock **39** recon **0.220** unlock **@42** recon **0.067** (~3 iters) then later locks **75** / **78** near the gate. P120 first wrap is deeper/slower (20 iters vs 3) — do **not** FALSIFY opscale on wrap depth while sr/H drifted. Live probes while locked **ignore** (@40 gain_fid **0.653**, @50 **0.640** H=1 r=**−0.017**). Probe@30 pre-wrap gain_fid **0.867**. Probe@60 post-unlock H=1 **+0.289** H=74 **+0.289** conv **0.25** gain_fid **0.836** (not a freeze/val TM). `opdev` **0.68@36 → 0.93@69**. jsonl MV @69 **×1.001** is teacher.

**P1 pre-wrap vs P116 (not a freeze score):** @19 recon **0.0092** skip 0. Probe@10 H=1 **+0.395** H=74 **+0.268** conv **1.00** gain_fid **0.638** (P116@10 H=54 **+0.293** conv **0.25** gain_fid **0.666**). @69 recon **0.0132** vs P116 @69 **0.0044**.

**Mechanism (intended):** return-to-`test_sim` on HEAD = opscale ON + ol1 gone + locgfix identity. **Cannot attribute TM to opscale on this pid.**

**Largest residual this plant:** R1 TM autoencoder (P116 MV ×0.706 / curve 0.262 vs P26 ×0.973) and R2 CV reversal **0.421**. Compounding ×0.911 is **not** the hole.

**Predicted signature:** opscale trains immediately (`opdev` O(0.7–0.9) even on linear); sr=3 from one noisy θ; first-fill wrap longer than P116; orig-P1 ~**81** not 87; freeze = RAM last_ok (now walking, unless a P116-class late wrap re-locks); val TM is mixed (opscale+sr+ol1-gone).

**Falsifier (at EXIT, still mixed):** orig-P1 GAIN_NOT_READY with healthy recon → freeze×(opscale|sr). Val TM collapse vs P116 ×0.706 → **do not** solely REVERT opscale; next job is sr-median first. Val TM ≥ P116 with opdev~0.9 → opscale did not obviously hurt despite sr=3 (KEEP pending `srmed`). A new 20× lock at orig-P1 is P42 restore, not a kill.

**Do not:** extra-P1 N+1; ol1/k1/traj N+1; identity-relaunch P116/P119; launch `gop`; stack critic knobs; second GPU; pin `DREAMER_SAMPLE_RATE=4` as a test_sim magic default; kill on wrap recon; score probe@60 / jsonl ×1 as val TM.

Config audit Step 4 (launch): env-free HEAD. vs P116: drop ol1; add opscale; **unplanned** sr 4→3. #❌ sr lottery **open**. #🆕 **0**.

## P118 EXIT — `opscale` INVALID observer (do not score / do not FALSIFY opscale)

tmux `mbrl2_p118` **gone**. pid **551549 DEAD**. sha **`ff5f84c`**. **EXIT=0** 166 jsonl iters ES `p3_skipped_invalid_observer`. nvidia **0 MiB**. No `best.pt`. Val `final.pt` expert-BC. `actor_experiment_valid=false`. jsonl NaN/Inf = **one** `wm_grad_norm=inf@13`. Env-free. `[resolved-cfg] opscale=True` no ol1. Local G MV **0.0153** vs ident **2.627** (engineering `_prev_control` Δu). **Do not identity-relaunch.**

**P1 gates:** orig-P1@**87 FAIL** last_ok **85** **0.34@MV**; extra-P1@**98 FAIL** live **0.11@MV**; cap@**110** last_ok **108** **GAIN_NOT_READY 0.68@MV / 0.85@DV** `@H[0.77,0.79]` `not_noisy=True`. Detonated-freeze restored 108. Warm-restore SKIPPED (`wm_best` @100 gain-blind). Graph released.

**P2 111–166:** first dobg **0.0209** skip **0** KEEP vs P95. `dob_A` **0.95257** held. `dob_K` **0.119→0.017@111→0.0018@117** min **0.0016@119** then recovered **0.047@166** SS **≈1.00**. leftover `|d_slow|/|d|` **0.80@111→1.53@117→0.416@166**. `std_ratio` **144389@111 → 0.77@166**. α **0.00391**. `wm_op_scale_dev` **~0.93**. P2→P3@**166 PASS** MTP **1.216** then **`[p3-skip]`**.

**Val (do not score actor):** MV TM ss/@H **×0.609 / ×0.705** curve IAE/\|real\| **0.35**. DV **×0.799 / ×0.703** curve **0.24**. Lineage `wm_gain_pass` PASS (rel_err 0.39 / 0.20) / `wm_gain_healthy=False` / observer-gain PASS ≠ GAIN-READY. Decomp MV real→post **×0.882** post→1step **×0.846** 1step→OL **×1.045** OL-vs-real **×0.607** lever=autoencoder. DV real→post **×0.868** post→1step **×0.827** lever=autoencoder. det_r **0.343** pred_std **3.00 vs 3.07**. Event IAE median-of-medians **29.2**. `critic_r=nan`. `all_pass=False` (`critic_pass`). Paired **−38.39 vs −212.28** 9/9 expert-BC.

**Judge:** locgfix **already on HEAD** (`414eca4`). **Not** an opscale FALSIFY. extra-P1 lottery closed. Next was **P119 `locgfix`** (now EXIT VALID). Champion **P64**. Dist champ **P26**.

## P119 EXIT — `locgfix` VALID (do not identity-relaunch / do not launch `gop`)

tmux `mbrl2_p119` **gone**. pid **557257 DEAD**. sha **`ea7def9`**. **EXIT=0** 406 jsonl iters ES `p3_plateau` (no >+1.0% over `best_det_return=−131.981` for 200 iters) best **206**. env_steps **1645000**. nvidia **0 MiB**. `best.pt` + `final.pt`. Val ckpt **`best.pt`**. `actor_experiment_valid=true`. jsonl NaN/Inf **0**. Env-free (`CUDA_VISIBLE_DEVICES=0`, no `DREAMER_*`). `[resolved-cfg] opscale=True` no ol1. Train start **2026-09-09 14:10:33**. **`[p3-skip]` did not fire.** P3 **159–406** (248 iters) skip_iter **0** (cum skip **11** is P1 storm-1). Critic-warmup 159–168; `[return-scale] FREEZE` **2.128** @169; unfreeze @169. @406 actor **+0.116** critic **9.51** ent **−0.467** logp **0.54** clip **0.10** rscale **2.13** rtgt **0.00067** `critic_pred_target_r` **0.991**. **Do not identity-relaunch.** **Do not launch P120 `gop`** (val TM **×0.758**, not ~P118 ×0.61). Next GPU this visit: **P120 `tshome`** return-to-`test_sim`.

**Teacher print (locgfix CONFIRMED):** local G `plant_fd=6/6` mean MV **2.890** / DV **−0.341** vs identified **2.709 / −0.435**. span MV **2.01** / DV **0.46**. `|du_mv|=0.4000` **=** step **0.4000**. **No** WARNING. jsonl MV O(1) since ~@5.

**Orig-P1 @91 FAIL (fair locgfix freeze miss):** live g **0.63@MV** `@H 0.61` DV **1.03/@H 0.98**. jsonl MV **×0.991** ≠ TM. extra-P1 lottery **closed**.

**Extra-P1 @101 PASS (not a freeze KEEP):** last_ok **100** GAIN-READY **0.83@MV** `@H 0.98` DV **0.92/@H 0.95** `compound_ok` 1step→OL **0.86**. Detonated-freeze restore **100**; re-probe **0.85@MV** 1step→OL **0.90**. Warm-restore SKIPPED. `[phase] 1→2` env_steps **908750**. P100-class confound.

**P2→P3 @158 PASS:** MTP median **1.180** (max=3.00 n=5) `p2_extension=0/42897`. env_steps **1276250**. P2 skip **0**. End-P2 recon **0.0527** dobg **0.0319** `dob_A` **0.95257** `dob_K` **0.167** SS **≈3.52** leftover **0.159** std **0.885**. First dobg **0.0034@102** skip **0** KEEP vs P95. vs P118 leftover **0.416** K **0.047/1.00** MTP **1.216** then `[p3-skip]`.

**Val (score actor — VALID):** MV TM ss/@H **×0.758 / ×0.922** curve IAE/|real| **0.153** (wm +0.693 vs real +0.914). DV **×0.942 / ×0.950** curve **0.080** (wm −0.390 vs real −0.414). Lineage MV rel_err **0.24** HEALTHY; DV **0.06** HEALTHY; `wm_observer_gain` HEALTHY. Decomp MV real→post **×0.960** post→1step **×0.962** 1step→OL **×0.846** OL-vs-real **×0.783** lever=**compounding**. DV real→post **×0.895** post→1step **×1.096** lever=autoencoder. det_r **0.346** R²_det **−4.27** pred_std **7.10 vs true 3.07**. Event IAE median-of-medians **20.3** (P118 **29.2**). Paired **−32.11 vs −230.94** **9/9 WIN**. cum_raw **−55585** (36). `critic_r` **+0.253** FAIL (<0.3) → `all_pass=False`. `smooth_pass=True` (mv_reversal **0.171** all-eps / **0.196** DR). hugging **0.384** / **0.316** DR; cv_viol **16.7**; mv_viol **2.79**. No `cv_d2_rms_normed` / `cv_opt_headroom` / `cv_viol_frac`. Val “[val] P3: 11 grad-clip skips” is **cumulative** P1 storm, not P3 this-iter.

**Judge:** ✅ KEEP locgfix as **teacher-unit fix** vs P118 0.015 / val ×0.609. ✅ KEEP as **P3-entry** vs P118 `[p3-skip]`. ❌ extra-P1 as freeze KEEP (orig-P1 **0.63@MV** is the fair miss; freeze last_ok **100** 0.83@MV transferred to val ×0.758, not P109-class collapse). ❌ **FALSIFIED as TM-to-P26** (×0.758 vs ×0.973). Remaining R1 hole is **compounding**. ❌ **FALSIFIED as Kalman amp** (pred_std 7.10 vs 3.07; det_r ~P118 0.343). Actor **VALID** vs baseline; **do not swap P64** (different plant). **P120 `gop` PARKED** — trigger was val TM still ~×0.61; it is ×0.758. Cadence: fair plant verdict done. GPU **free**. Do **not** extra-P1 N+1 / identity-relaunch P117/P118/P119 / rewrite locgfix / in-place `c*=gop`.

## Ranked follow-ups

**#1 P120 `tshome` — LIVE.** Snapshot only until EXIT. Do **not** score actor / launch `gop` / extra-P1 / second GPU.

**#2 P121 `srmed` (next GPU after P120 EXIT)** — one attributed change: sample-rate Nyquist from **median/dominant θ**, not `min(all_dead)`.
- **Mechanism:** `derive_all` currently calls `derive_sample_rate(tau_fast, dead_fast)` (`plant_init.py` ~257) with `sr_source='auto:tau_fast/10_or_dead_fast/2'`. APC delay sampling is ~2 samples in the *channel* θ (already `round(θ/2)`), not the noisiest FOPDT repeat. P120 binding constraint is `dead_fast=6` from REFLUX r4 (same pair as `tau_fast=47`); `dead_dom=8` already matches `per_mv_dynamics`/`per_cv_dynamics`. Pass **`dead_dom` into `derive_sample_rate`**; keep `dead_time_fastest=min` for lookback / diagnostics. Do **not** rename min→median on the fastest field (lookback ceiling would silently retune). Do **not** also median-`tau_fast` this job (τ lottery is not binding: `round(47/10)=round(53.5/10)=5`).
- **Predicted signature:** test_sim `sr=4` `H~54` `gmatch_len~54`; `[resolved-cfg]` opscale still ON; `t_wm` closer to P116 ~99 s if opscale is the remaining extra; orig-P1 timing P116-class (~87); first-fill wrap closer to P116 0.22 / unlock +2 iters (P120 was 0.77 / 20 iters).
- **Falsifier:** `run_plan` sr still 3; **or** a real fast channel is undersampled (lookback/TM @H worse than P116 with no other change). A MIMO plant with a genuine fast DV whose *channel-median* θ is short would still need min-of-per-channel-medians — that is **#6**, not this A/B.
- **Files:** `utils/plant_init.py` (`derive_all` pass `dead_dom`; docstring of `derive_sample_rate`; `sr_source` → `auto:tau_fast/10_or_dead_dom/2`). Smoke in `tools/_smoke_rssm.py` next to the existing `derive_all` pin test (~5522): dyn `dead_time_identified=8` / `dead_time_fastest_identified=6` / `tau_fastest_identified=47` must auto-sr **4**. Identifier comment already notes P121. No `DREAMER_SAMPLE_RATE=4` pin. Do not change `lookback_identifier.derive_sample_rate_for_lookback` (fastest ceiling KEEP).
- **Why this before opscale KEEP/REVERT:** P120 mixes opscale + sr + ol1-gone. Family still open; N+1 opscale would be lottery. Wrap-deeper-than-P116 is the same confound until sr matches.

**#3 After `srmed` VALID:** score opscale vs P116 R1 TM (autoencoder ×0.706). If TM << P116 with sr matched → gate `op_scale_net` when rest-IC `|G|` span ≲ 1.3× (linear identity; SysID-derived, not a test_sim magic). If TM ≥ P116 → KEEP opscale as default even on linear. **Do not** identity-regularizer λ lottery.

**#4 R2 CV reversal (after a VALID observer on this plant):** real-sim penalty on `cv_reversal_rate` / `cv_d2_rms_normed` only (not `mv_reversal`). Predicted: `smooth_pass` without headroom/return_time_frac regression. Falsifier: mid-band sit or slow DR return. Files: `_realsim_actor_critic_step` / objective. Do **not** stack critic knobs / rtgt.

**#5 P120 `gop` (PARKED)** — **do not launch**. P119 val TM **×0.758** is not ~×0.61. P73 persist forbids in-place `c*=gop`.

**#6 After `srmed` if a later plant has a real fast DV:** min of *per-channel* median θ (already in `per_mv_dynamics` / `per_dv` pairs), not global median of every repeat. Do **not** launch this while test_sim channel medians are all 8.

**Do not:** extra-P1 N+1, ol1 N+1, identity-relaunch P117/P118/P119, launch `gop`, stack critic knobs, second GPU, sr pin as engineering default.

## RCA this visit

- SysID: test_sim **θ_dom=8** (12/16 pairs; channel medians 8). **θ_fast=min** is one noisy MV REFLUX r4 (θ=6, τ=47) that flips `sr` 3/4 and `H` 74/54. Not a fast DV. τ lottery is **not** binding (`sr_tau=5` either way). Nonlinear OP-span (2.23×) is a different plant — do not mix.
- Signal: P120 skip 0, no Inf, heartbeat ~2.3 min. First-fill wrap **unlocked @56** (20-iter lock, peak 0.77, second spike 0.59@49). last_ok RAM walking **67–69**; disk file still the lock-36 blob (expected). `wm_best` still **30**. Probe@60 gain_fid **0.836** is post-unlock live, not val TM. `opdev` **0.93**. Disk `/home` 64% / 62G. Live keep-2 `ckpt_iter`. Keep P120+P119+P118+P117+P116+P64+P53.
- Control: do not score actor. R2 still open on P116 (rev 0.421). Do not kill / second GPU / extra-P1 / `gop`.
- ML: identity-init opscale is **not** an identity prior after step 0 (hidden MLP + last Linear in group `g`). Linear `opdev` matching P119 ~0.8–0.9 is expected. P42 20× lock + P49 unlock + P50 5× overwrite **worked** (unlock@56, walk@67).
- Plant: locgfix identity on test_sim CONFIRMED (`|du_mv|=step`, local G ≈ ident).
- Metric: wrap recon / live TM-probe while `last_ok_locked` are **not** opscale scores. jsonl ×1 is teacher. Probe@60 still not val TM. Trust JSON IAE + npz return-to-limit. `wm_op_scale_dev~0` on trial as linear-identity test. Wrap-*duration* vs P116 is on trial as an sr score, not an opscale FALSIFY.
- Literature (docs/papers PDFs not on host; HTTPS blocked): delay-dominant sampling is 2+ samples in the **channel** dead time (the `round(θ/2)` rule already in `derive_sample_rate`), not `min` over noisy FOPDT repeats. DreamerV3/V4 papers do not choose plant Δt. No new lever from the papers README this visit.

P117 process died mid-P2 (SIGKILL). P118 completed P2 then `[p3-skip]`. P119 completed P2 then **P3 then EXIT VALID**. Do not identity-relaunch P117, P118, or P119.
