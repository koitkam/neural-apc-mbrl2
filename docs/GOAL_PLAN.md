# GOAL_PLAN — neural-APC-mbrl2

Living plan. Update every visit (live analysis and EXIT). Champions live in `docs/RUN_HISTORY.md`. **`RUN_HISTORY` appends at P1-gate / P2→P3 / EXIT only** — not LIVE@iter snapshots.

**Product:** simulator-agnostic neural APC — smooth CV on the economic limit without violating; faithful observer; unmeasured-load rejection. Envelope: learned observer + neural Kalman/DOB + neural actor-critic. No gray-box plant, no PID/LQR/MPC as the product, no DV-only FF.

Plant this cycle: `nonlinear_sim` HeatExchangerTower (P118 EXIT INVALID CAPPED **0.68@MV**; **P119 locgfix LIVE P3 @262** — P2→P3@**158 PASS** MTP **1.180**; `[p3-skip]` did **not** fire; orig-P1@**91 FAIL 0.63@MV**; extra-P1@**101 PASS** last_ok **100** GAIN-READY **0.83@MV**; extra-P1 lottery **closed**; then return-to-`test_sim`).

## Residual board vs champions (test_sim unless noted)

| Axis | Score we trust | Champion | Last VALID (P116) | This plant P118 val (`final.pt`, do **not** score) | Status |
|---|---|---|---|---|---|
| R1 TM MV ss/@H / curve | val TM, not jsonl ×1 | P26 ×0.973 / @H ×0.880 | ×0.706 / ×0.711 | ss **×0.609** / @H **×0.705** curve IAE/\|real\| **0.35** (wm +0.556 vs real +0.914). Freeze last_ok **108** **0.68@MV** — wrong-unit teacher | OPEN |
| R1 TM DV ss/@H / curve | same | P64 ×0.893 / @H ×0.962 | ×0.692 / ×0.712 | ss **×0.799** / @H **×0.703** curve IAE/\|real\| **0.24** (wm −0.331 vs real −0.414). Lineage HEALTHY 0.80; not GAIN-READY | OPEN |
| R1 compounding | 1step→OL | P64 ×0.85 | ×0.911 | MV **×1.045** (OL-vs-real **×0.607**) lever=autoencoder | OPEN |
| R2 CV smoothness | worst-seed `cv_d2_rms_normed`≤0.05 AND `cv_reversal_rate`≤0.25 | — | P116 `smooth_pass` | suite does not emit `cv_d2_rms_normed`. `smooth_pass=True` (mv_reversal **0.141** all-eps / **0.176** DR). **Do not score** (expert-BC) | OPEN |
| R2 headroom / viol | `cv_opt_headroom`, `cv_viol_frac` (not a P3-skip gate) | — | hugging 0.35; mv_viol 1.67 | suite does not emit those names. hugging **0.389** all-eps / **0.330** DR; cv_viol **35.4**; mv_viol **1.90**. **Do not score** | OPEN |
| R3 Kalman | `det_r` + pred_std vs true | P26 det_r 0.68 | 0.676 / 1.479 vs 1.93 | det_r **0.343** R²_det **−0.961** pred_std **3.00 vs true 3.07** (this plant; not test_sim 1.93). Do not credit std-match | OPEN |
| Actor econ | paired vs baseline **and** P64; only if `actor_experiment_valid` | P64 −4.54 vs −94 | −8.24 vs −109.83 VALID | expert-BC **−38.39 vs −212.28** 9/9; cum_raw **−76887** (36). `actor_experiment_valid=false`. **Do not score** | OPEN |

No standalone `residual_board.json` — axes above are `validation_summary.json` + TM/decomp/dist JSON. Files agree with `[val]` plot lines (MV real=+0.914 wm=+0.556; DV ss 0.80; decomp 0.882/0.846/1.045; det_r 0.343).

## Metric audit (trusted vs on trial)

**Trusted:** val TM ss/@H/curve_iae (MV and DV); gain decomp real→post / post→1step / 1step→OL; `det_r` + pred_std vs true; paired econ vs baseline **and** champion when freeze GAIN-READY and P3 ran; CV d2/reversal (`cv_d2_rms_normed` + `cv_reversal_rate` — **not** the suite's current `smooth_pass`); `cv_opt_headroom` / `cv_viol_frac`; DR `cv_return_headroom` / `cv_return_time_frac`. Skip-storm 5-level TM (`DCgain_ratio` / `@H` / `compound_ok`) is a freeze/RCA diagnostic, not a champ score. Launch `|du_mv|` vs teacher `step` (P119 print): O(step) = WM-norm; ≫5×step = engineering-unit bug. jsonl `wm_op_scale_dev` is observability that LPV moved, not a TM score. Post-restore recon + skip **this-iter** (P119 P2-end @158 recon **0.0527** skip **0**) is the skip-storm liveness check, not a TM score.

**On trial / do not GPU-optimize:** jsonl teacher ×1 vs SysID median (tautology on OP-varying plants — P117); **P119 jsonl `gain_match_mv_ratio` vs local rest-IC G** (honest after locgfix: orig-P1 jsonl MV **×0.991** while 5-level TM **0.63@MV** vs identified **2.71** — teacher pin ≠ GAIN-READY); **P118 jsonl `gain_match_mv_ratio` vs local G** (teacher Δu was engineering `_prev_control` − WM-norm action → local MV G **0.015** vs identified **2.63**; extra-P1@98 jsonl MV **1.20** and cap last_ok@108 jsonl MV **0.71** are still vs G≈0.015, not TM); ss-ratio alone; lineage `wm_gain_pass` (P118 rel_err=0.39 PASS while GAIN-READY failed); mv_reversal as a smoothness gate (**suite `smooth_pass` still keys `mv_reversal_rate≤0.5`** in `evaluation/validate.py:control_quality_gates` — gamed vs the standing residual; `compute_episode_kpis` still emits only `mv_reversal_rate` / `mv_bound_hugging_score`, not CV d2); CV total variation as a smoothness gate; critic_r without `critic_rew_to_tgt_var`; raw dist R²; pred_std matching true_std when R²_det is −0.96; event IAE without return-to-limit; VALID 9/9 / GAIN-READY / all_pass / family-closed as “residual closed”; beating P64 to KEEP a smoothness/headroom/DR win; **`wm_grad_norm=inf` at a skip-storm restore** (P118@13) as a kill; **finite `wm_grad_norm` 5e17 on a skip-storm iter** (P119@77) as a kill; **20× lock + inject@76** as a kill; **recon-spike ~3 iters after P1 inject** as a kill; **persist_rel spikes to 31** as a kill; **H=1 / H=56 fidelity r** as a freeze score (`wm_best` @80 EMA **6.299** is gain-blind; Probe@100 H=56 **+0.590** / conv **0.25**); **skip-storm 1 last_ok TM 0.56@MV** as a freeze score (READY-only cap deferred; orig-P1@91 is the fair locgfix miss); **P119 extra-P1 last_ok 100 GAIN-READY 0.83@MV** as the champ freeze (lottery closed — orig-P1 **0.63@MV** is the fair miss; val TM is the score); **skip-storm 2 0.02@MV** as a last_ok freeze score; **extra-P1@98 live TM 0.11@MV** as the freeze score (cap used last_ok **108** **0.68@MV**); **P118 expert-BC paired −38 vs −212** as actor-econ; **mutating stored gain-c in-place as the P120 `gop` mechanism** (would fight P73 persist 0.1).

P118 EXIT: DV local G **−0.345** vs ident **−0.426** (same space, OK). MV local G is the unit bug, not equal-% OP variation. LPV `wm_op_scale_dev` **0.52→0.99** then **~0.86–0.93** while Huber MV target ≈0 — do not score LPV from this pid. Freeze TM **0.68@MV / 0.85@DV** is last_ok **108** (recon-best **0.0073**), not live detonated g. Val TM **×0.609/@H ×0.705** is the same under-gained observer, not a new LPV result.

## Closed families (do not N+1)

extra-P1 as freeze (P41; **P117 on this plant**; **P118 extra-P1 0.11@MV worse than orig-P1 0.34@MV**); compounding-teacher Huber (P111 traj unpromoted, P115 k1 REVERT, P116 ol1 REVERT); isolation-off KEEP as default; 2TS-α as champ; kfeat as freeze; quiet_env as freeze; decoder FF; TSSM; GRU keep-h.

## P118 EXIT — `opscale` INVALID observer (do not score / do not FALSIFY opscale)

tmux `mbrl2_p118` **gone**. pid **551549 DEAD**. sha **`ff5f84c`**. **EXIT=0** 166 jsonl iters ES `p3_skipped_invalid_observer`. nvidia **0 MiB**. No `best.pt`. Val `final.pt` expert-BC. `actor_experiment_valid=false`. jsonl NaN/Inf = **one** `wm_grad_norm=inf@13`. Env-free. `[resolved-cfg] opscale=True` no ol1. Local G MV **0.0153** vs ident **2.627** (engineering `_prev_control` Δu). **Do not identity-relaunch.**

**P1 gates:** orig-P1@**87 FAIL** last_ok **85** **0.34@MV**; extra-P1@**98 FAIL** live **0.11@MV**; cap@**110** last_ok **108** **GAIN_NOT_READY 0.68@MV / 0.85@DV** `@H[0.77,0.79]` `not_noisy=True`. Detonated-freeze restored 108. Warm-restore SKIPPED (`wm_best` @100 gain-blind). Graph released.

**P2 111–166:** first dobg **0.0209** skip **0** KEEP vs P95. `dob_A` **0.95257** held. `dob_K` **0.119→0.017@111→0.0018@117** min **0.0016@119** then recovered **0.047@166** SS **≈1.00**. leftover `|d_slow|/|d|` **0.80@111→1.53@117→0.416@166**. `std_ratio` **144389@111 → 0.77@166**. α **0.00391**. `wm_op_scale_dev` **~0.93**. P2→P3@**166 PASS** MTP **1.216** then **`[p3-skip]`**.

**Val (do not score actor):** MV TM ss/@H **×0.609 / ×0.705** curve IAE/\|real\| **0.35**. DV **×0.799 / ×0.703** curve **0.24**. Lineage `wm_gain_pass` PASS (rel_err 0.39 / 0.20) / `wm_gain_healthy=False` / observer-gain PASS ≠ GAIN-READY. Decomp MV real→post **×0.882** post→1step **×0.846** 1step→OL **×1.045** OL-vs-real **×0.607** lever=autoencoder. DV real→post **×0.868** post→1step **×0.827** lever=autoencoder. det_r **0.343** pred_std **3.00 vs 3.07**. Event IAE median-of-medians **29.2**. `critic_r=nan`. `all_pass=False` (`critic_pass`). Paired **−38.39 vs −212.28** 9/9 expert-BC.

**Judge:** locgfix **already on HEAD** (`414eca4`). **Not** an opscale FALSIFY. extra-P1 lottery closed. Next **P119 `locgfix`**. If P119 still CAPPED with WM-norm teacher: **P120 `gop`**. Champion **P64**. Dist champ **P26**.

## Live this visit — P119 `locgfix` (do not kill / do not second GPU)

tmux `mbrl2_p119` pid **557257** sha **`ea7def9`** `device=cuda` bs=128 compile=eager nvidia **~2859 MiB**. Env-free (`CUDA_VISIBLE_DEVICES=0`, no `DREAMER_*`). `[resolved-cfg] opscale=True` no ol1. STAGE 3 `wm_frozen` FROZEN WM+DOB. Train start **2026-09-09 14:10:33**. **One GPU job.** jsonl NaN/Inf **0**.

**Teacher print (locgfix CONFIRMED):** local G `plant_fd=6/6` mean MV **2.890** / DV **−0.341** vs identified **2.709 / −0.435**. span MV **2.01** / DV **0.46**. `|du_mv|=0.4000` **=** step **0.4000**. **No** WARNING.

**P2→P3 @158 PASS (2026-09-09 18:10:26):** `[gate p2->p3] PASS` reward_mtp median **1.180** (max=3.00, n=5); `p2_extension=0/42897`. `[phase] 2→3` env_steps **1276250**. **`[p3-skip]` did not fire** (`skip_invalid_p3=True`; freeze last_ok **100** GAIN-READY). P2 **102–158** skip **0** (cum skip **11** is P1 storm-1). End-P2 recon **0.0527** dobg **0.0319**. `dob_A` **0.95257** held. `dob_K` **0.119→0.017@102** min **0.006@104** then **0.167@158** SS **≈3.52**. leftover `|d_slow|/|d|` **0.85@102→0.159@158**. `std_ratio` **38822@102 → 0.885@158**. α **0.00391**. vs P118 end-P2 K **0.047/1.00** leftover **0.416** std **0.77** MTP **1.216** then `[p3-skip]`. First dobg **0.0034@102** skip **0** KEEP vs P95.

**LIVE P3 @262 (2026-09-09 18:19:04, ~249 min):** skip_iter **0** (entire P3) last_ok **100 locked**. env_steps **1435000**. ret_ema **−209.56** ret_w **−276.13**. actor **+0.174** critic **9.98** ent **−0.486** logp **0.50** clip **0.13** rscale **2.13** **rtgt 0.0014** (collapsed from **0.087@159** — standing P3 critic collapse, do not stack knobs). `critic_pred_target_r` **0.986**. sps **~305**. nvidia **~2859 MiB**. jsonl NaN/Inf **0**. `wm_op_scale_dev` **0** / dob_K **0** (frozen). Critic-warmup 159–168; `[return-scale] FREEZE` **2.128**; unfreeze **@169**. No `p3_plateau` yet.

**Orig-P1 @91 FAIL:** live g **0.63@MV** `@H 0.61` DV **1.03/@H 0.98**. Fair locgfix miss (jsonl ×0.99 ≠ TM). extra-P1 lottery **closed**.

**Extra-P1 @101 PASS (log, not a freeze KEEP):** wrap@101 after inject@100 recon **0.0872 = 31×** best **0.0028** skip **0** gnorm **5.23** → last_ok **100 locked**. Gate measured last_ok **100** (live >5×). `[gain-ready-probe] median 3/3 worsts=[0.82,0.84,0.83]` `DCgain_ratio[0.83,0.92] @H[0.95,0.98]` worst **0.83@MV** `@H 0.98` DV **0.92/@H 0.95** `unbiased=True` `not_noisy=True` `compound_ok=True` 1step→OL **0.86**. `[gate p1->p2] PASS` recent_max=**6.299**. Detonated-freeze restore last_ok **100**; re-probe **0.85@MV** 1step→OL **0.90**. Warm-restore SKIPPED (`wm_best` gain-blind). `[phase] 1→2` env_steps **908750**. extra-P1 **not** a KEEP of extra-P1 as freeze (P41/P117/P118). Freeze g **is** last_ok **100** GAIN-READY. Score actor only after P3 if `actor_experiment_valid`. Val TM vs orig-P1 **0.63@MV** is the locgfix score.

**Last P2 Probe@150:** `H=1:r=+0.526 H=14:r=+0.438 H=28:r=+0.503 H=56:r=+0.480 conv=0.75 drift=0.129 floor=0.40 best_h=56/56 gain_fid=0.840`. vs @140 H=56 **+0.352** (below floor, recovered). `wm_best` EMA **5.958@130** — gain-blind.

Predicted remaining: P3 budget **592500** from **1276250** → ~**1.869e6**. Remaining **~434k** steps ≈ **~24 min** @~305 sps, or earlier `p3_plateau`. Score actor only if freeze GAIN-READY **and** P3 **and** `actor_experiment_valid`. If val TM still ~P118 **×0.61**: encode-L then **P120 `gop`**. Do **not** extra-P1 N+1.

Falsifier for locgfix as freeze: orig-P1 **0.63@MV** (fair). extra-P1 last_ok **100** **0.83@MV** is P100-class confound. Do **not** use P118’s 0.68@MV as the falsifier.

Do **not** extra-P1 N+1 / identity-relaunch P118 / rewrite the live recipe / second GPU.

After a fair P119 VALID or a fair GAIN_NOT_READY with a **correct** local teacher: return-to-`test_sim` on the worst residual (cadence: 2 jobs on this plant then return). P118 is job 2 of 2 **but** the unit bug means P119 is the fair repeat of the same mechanism, not a third plant-hop.

## Ranked follow-ups (not this GPU until P119 verdict)

1. **P119 locgfix** (LIVE P3 @262) — P2→P3@**158 PASS** MTP **1.180**; `[p3-skip]` did **not** fire. Freeze last_ok **100** GAIN-READY **0.83@MV** (extra-P1; lottery closed). Orig-P1 **0.63@MV** is the fair miss. Do not rewrite the live recipe. Do **not** extra-P1 N+1. Score actor only if `actor_experiment_valid`.
2. If val TM still short after this GAIN-READY freeze: keep LPV, do **not** extra-P1. First check encode vs ID: `identified_lookback=131` vs `seq_len=128` (WARNING at launch; τ=54.5 sr=4 → 4τ/sr=54.5, K=H=56 is the settle, not the hole). Optionally pin encode L=`identified_lookback` (one attributed change) **only after** EXIT val TM is still short.
3. If val TM still ~P118 **×0.61** (extra-P1 freeze confound / orig-P1 0.63) **with** WM-norm local G: input-scale LPV is too thin for equal-% DC. **P120 `gop`** (still neural, one mechanism): OP-conditioned **gain-c readout**, not another `1+tanh` on GRU inputs and **not** an in-place write into stored `c`.
   - **Mechanism:** new `gain_op_net` (same `_MLP` + zero-init last Linear as `op_scale_net`; **separate weights**). OP=`concat(action,dv)` stop-grad. `scale = 1+tanh(net(OP))`. Stored `RSSMState.c` stays the reference gain (P73 persist KEEP: `cont_gain_persist` is step-to-step MSE on `cont['sample'][..., :n_gain]` — mutating stored c would fight persist whenever OP walks inside T=128). Apply `c_eff[..., :G] = c[..., :G] * scale` **at use**:
     - `_gru_transition` / TSSM token: scale `_recurrence_c` before concat (P71: G must enter GRU).
     - `feat` / `decode`: scale the c-slice so DC G(op) is what the observer emits (P74 additive skip stays closed).
   - **Do not** `c[..., :G] *= scale` after `obs_step`/`img_step` (old spec — persist N+1). `op_scale_net` KEEP (P119 `wm_op_scale_dev` **0.80** proves input-LPV moved; still the wrong place for G=G(u)). Group `g`. No new TrainConfig / no `DREAMER_*`. Banner `gop=True`. jsonl `wm_gain_op_dev` = mean `|scale−1|` (mirror `wm_op_scale_dev`).
   - **Why not more input-scale:** equal-% DC is G=G(u) on the **output gain**. P118 `wm_op_scale_dev` sat at ~0.99 chasing G_tgt≈0 — not a test of gain-c G(op). P119 opscale **is** a test (dev 0.80 + WM-norm teacher) and still may CAPPED. Decoder FF closed (P74). Gray-box `G=k·valve%` is out of envelope.
   - **Predicted signature:** step-0 scale≡1 (zero-init); `wm_gain_op_dev` moves off 0 while persist_rel stays P119-class (not a new persist war); freeze GAIN-READY vs P117 0.71@DV / P118 0.68@MV; val TM ss closer across OP.
   - **Falsifier:** still CAPPED ~0.68@MV with locgfix teacher **and** `wm_gain_op_dev` not ~0; **or** persist detonates / GAIN_NOT_READY because scale was written into stored c.
   - **Files:** `models/dreamer_v4_rssm.py` (`init_gain_op_net` next to `init_op_scale_net`; helper used from `_gru_transition` + feat/decode), `models/transformer_ssm.py` parity, `training/train.py` jsonl, `tools/_smoke_rssm.py` (identity at init; persist graph unchanged; scale moves after noise).
4. **CPU (mode 3 / next EXIT visit, not this GPU):** emit `validation/residual_board.json` at val with R1 (ss/@H/curve_iae MV+DV, 1step→OL), R2 (`cv_d2_rms_normed`, `cv_reversal_rate`, `cv_opt_headroom`, `cv_viol_frac` — **stop using `mv_reversal` as `smooth_pass`**), R3 (`det_r`, pred_std vs true, event IAE, `cv_return_headroom`, `cv_return_time_frac`). `control_quality_gates.smooth_pass` currently keys `mv_reversal_rate≤0.5` (`evaluation/validate.py`) — that gate is obsolete. Implement only when GPU-free or at P119 EXIT val-pin (P47 modules pinned at launch, so editing validate.py now does **not** change this pid's EXIT suite).
5. Return-to-`test_sim` for R2 headroom / critic_rew_to_tgt_var (P3 collapse is standing) after a fair plant verdict.

## RCA this visit

- SysID: OP-varying DC is real (equal-% + 1/feed; MV amp **2.23×** / DV **2.34×**). Median G is still the wrong pin **once local G is in WM-norm**. P118 local MV 0.015 is units, not that 2.23× span.
- Signal: P119 locgfix teacher CONFIRMED. **Orig-P1@91 FAIL 0.63@MV**. **Extra-P1@101 PASS** last_ok **100** GAIN-READY **0.83@MV** 1step→OL **0.86** (lottery closed — not a KEEP). **P2→P3@158 PASS** MTP **1.180**; `[p3-skip]` did **not** fire. End-P2 K **0.167/3.52** leftover **0.159** std **0.885** skip **0**. LIVE P3 @262 skip **0** rscale **2.13** **rtgt 0.0014** (collapsed). GPU **~2.9 GB**. Disk `/home` 64% / 62G. Keep P119+P118+P117+P116+P64+P53.
- Control: freeze last_ok **100** GAIN-READY → P3 **is running** (`skip_invalid_p3` did not skip). Score P119 only if `actor_experiment_valid`. Do not kill P119 / second GPU / extra-P1 N+1.
- ML: Orig-P1 **0.63@MV** is the fair locgfix miss. extra-P1 last_ok **100** **0.83@MV** is P100-class confound. End-P2 leftover **0.159** / K **0.167** hotter than P118 **0.416 / 0.047**. LPV `wm_op_scale_dev` **0.86** at P2-end. Detonated-freeze restored 100 (wrap@101 31×). Warm-restore SKIPPED. extra-P1 lottery closed. If val TM still short: encode-L then **P120 `gop`**. P73 persist 0.1 forbids in-place `c*=gop`.
- Plant: HeatExchangerTower `step` is engineering; APCEnv denorms. Teacher must use `_prev_cmd_norm`. Restore must not leave a post-FD leftover when the snapshot was `None`. `identified_lookback=131 > seq_len=128` is a 3-sample encode shortfall, not this freeze.
- Metric: jsonl MV ratio vs 0.015 is not val TM. Cap 0.68@MV is last_ok TM vs identified, still GAIN_NOT_READY because the teacher never trained identified-scale MV G. Lineage gain PASS (rel_err 0.39) is looser than band [0.8, 1.3]. Suite `smooth_pass` on mv_reversal is on trial. No `residual_board.json` producer exists — plan #4, not a GPU job.

P117 process died mid-P2 (SIGKILL). P118 completed P2 then `[p3-skip]`. P119 completed P2 then **P3 started**. Do not identity-relaunch P117 or P118. P119 is LIVE P3.
