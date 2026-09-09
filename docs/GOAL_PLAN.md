# GOAL_PLAN — neural-APC-mbrl2

Living plan. Update every visit (live analysis and EXIT). Champions live in `docs/RUN_HISTORY.md`.

**Product:** simulator-agnostic neural APC — smooth CV on the economic limit without violating; faithful observer; unmeasured-load rejection. Envelope: learned observer + neural Kalman/DOB + neural actor-critic. No gray-box plant, no PID/LQR/MPC as the product, no DV-only FF.

Plant this cycle: `nonlinear_sim` HeatExchangerTower (P118 EXIT INVALID CAPPED **0.68@MV**; **P119 locgfix LIVE P1 @37** — probe@30 H=56 **+0.455** above floor / conv **0** / **not** GAIN-READY; then return-to-`test_sim`).

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

**Trusted:** val TM ss/@H/curve_iae (MV and DV); gain decomp real→post / post→1step / 1step→OL; `det_r` + pred_std vs true; paired econ vs baseline **and** champion when freeze GAIN-READY and P3 ran; CV d2/reversal (smooth_pass); `cv_opt_headroom` / `cv_viol_frac`. Skip-storm 5-level TM (`DCgain_ratio` / `@H` / `compound_ok`) is a freeze/RCA diagnostic, not a champ score. Launch `|du_mv|` vs teacher `step` (P119 print): O(step) = WM-norm; ≫5×step = engineering-unit bug.

**On trial / do not GPU-optimize:** jsonl teacher ×1 vs SysID median (tautology on OP-varying plants — P117); **P118 jsonl `gain_match_mv_ratio` vs local G** (teacher Δu was engineering `_prev_control` − WM-norm action → local MV G **0.015** vs identified **2.63**; extra-P1@98 jsonl MV **1.20** and cap last_ok@108 jsonl MV **0.71** are still vs G≈0.015, not TM); ss-ratio alone; lineage `wm_gain_pass` (P118 rel_err=0.39 PASS while GAIN-READY failed); mv_reversal as a smoothness gate; CV total variation as a smoothness gate; critic_r without `critic_rew_to_tgt_var`; raw dist R²; pred_std matching true_std when R²_det is −0.96; VALID 9/9 / GAIN-READY / all_pass / family-closed as “residual closed”; beating P64 to KEEP a smoothness/headroom/DR win; **`wm_grad_norm=inf` at a skip-storm restore** (P118@13) as a kill; **recon-spike ~3 iters after P1 inject** as a kill; **persist_rel spikes to 31** as a kill; **H=1 fidelity r** as a freeze score (`wm_best` @100 is gain-blind; warm-restore SKIPPED); **skip-storm 2 0.02@MV** as a last_ok freeze score; **extra-P1@98 live TM 0.11@MV** as the freeze score (cap used last_ok **108** **0.68@MV**); **P118 expert-BC paired −38 vs −212** as actor-econ.

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

tmux `mbrl2_p119` pid **557257** sha **`ea7def9`** `device=cuda` bs=128 compile=eager nvidia **~14751 MiB**. Env-free (`CUDA_VISIBLE_DEVICES=0`, no `DREAMER_*`). `[resolved-cfg] opscale=True` no ol1. STAGE 1 `g=84 dob=8`. Rest-IC CUDA graph **captured** `N=6 T=128`. Train start **2026-09-09 14:10:33**. **One GPU job.** ~2.0 min/iter after first. jsonl NaN/Inf **0**.

**Teacher print (locgfix CONFIRMED):** local G `plant_fd=6/6` mean MV **2.890** / DV **−0.341** vs identified **2.709 / −0.435**. span MV **2.01** / DV **0.46**. `|du_mv|=0.4000` **=** step **0.4000**. **No** WARNING.

**LIVE P1 @37 (2026-09-09 15:28:10, ~78 min):** skip **0** last_ok **37 unlocked**. recon **0.0089** (best **0.0056@34**; @36 **0.0060**). jsonl MV/DV **×0.998 / ×0.996**. gnorm **0.69 finite** (peak **40.2@28** wrap). persist_rel **1.71@37** (peak **14.41@34**, not 31). `wm_op_scale_dev` **0.80**. jsonl NaN/Inf **0**. **Inf@13 ABSENT**. P117 storm@23 **ABSENT**. Wrap@28 **16.2×** recovered; inject@30 no lock.

**Wrap@28:** recon **0.1466** = **16.2×** best **0.0091** (above restore 5×, **below lock 20×**). last_ok stayed **27** unlocked. jsonl **×1.312 / ×−0.930**. gnorm **40.2** finite skip **0**. Recovered @29 last_ok **29** recon **0.0198** jsonl **×1.001 / ×0.951**. Not a 20×-lock. Inject@30 dv-prbs: recon **0.0116→0.0086@31** — no lock.

**Probe@30:** `H=1:r=+0.372 H=14:r=+0.458 H=28:r=+0.474 H=56:r=+0.455 conv=0.00 drift=0.212 floor=0.40 best_h=56/56 gain_fid=0.859`. vs probe@20 H=56 **+0.514** conv **0.25**; vs P118@30 H=56 **+0.292** best_h=0. H=56 still above floor (slipped). H=1 still below. conv **0.25→0**. **Not** GAIN-READY. `wm_best` EMA **4.512@30** — gain-blind.

Predicted remaining: probe@40 (~6 min) / 20× lock@60-class / orig-P1@~87 (~1.7 h); then GAIN-READY vs P118 CAPPED **0.68@MV** / P117 **0.71@DV**. Score actor only if freeze GAIN-READY **and** P3.

Falsifier: still CAPPED ~0.68@MV **after** this WM-norm teacher (WARNING absent; `|du_mv|`~step; jsonl MV ratio O(1); Inf@13 absent). Do **not** use P118’s 0.68@MV as the falsifier.

Do **not** extra-P1 N+1 / identity-relaunch P118 / rewrite the live recipe / second GPU.

After a fair P119 VALID or a fair GAIN_NOT_READY with a **correct** local teacher: return-to-`test_sim` on the worst residual (cadence: 2 jobs on this plant then return). P118 is job 2 of 2 **but** the unit bug means P119 is the fair repeat of the same mechanism, not a third plant-hop.

## Ranked follow-ups (not this GPU until P119 verdict)

1. **P119 locgfix** (LIVE above) — causal for R1 on this plant. Locgfix teacher CONFIRMED.
2. If P119 GAIN-READY but val TM still short: keep LPV, audit rest-IC K vs 4τ settle (this plant K=H=56 ≈ 4τ/sr = 54; settle L=128 is the encode, not the hole). Not extra-P1.
3. If P119 still CAPPED ~0.68@MV / 0.71@DV **with** WM-norm local G: input-scale LPV is too thin for equal-% DC. **P120 `gop`** (still neural, one mechanism): OP-conditioned **gain-c**, not another `1+tanh` on GRU inputs.
   - **Mechanism:** after `obs_step`/`img_step` produce `c`, `c[..., :cont_gain_dim] = c[..., :cont_gain_dim] * (1 + tanh(MLP(stop-grad OP)))` with last Linear zero-init (step-0 ≡ P119). Scale **gain-c only**. Decoder sees scaled gain-c so DC G(op) lives in the observer. `op_scale_net` LPV KEEP (no N+1 of that net). Group `g`. No new TrainConfig / no `DREAMER_*`. Banner `gop=True`. jsonl `wm_gain_op_dev`.
   - **Why not more input-scale:** equal-% DC is G=G(u) on the **output gain**. P118 `wm_op_scale_dev` sat at ~0.99 chasing G_tgt≈0 — that is not a test of gain-c G(op). Decoder FF closed (P74). Gray-box `G=k·valve%` is out of envelope.
   - **Predicted signature:** local G span ~2× (SysID); freeze GAIN-READY vs P117 0.71@DV; val TM ss closer across OP; `wm_gain_op_dev` moves off 0.
   - **Falsifier:** still CAPPED ~0.68@MV with locgfix teacher **and** a G(op) net that actually moved (`|(scale−1)|` not ~0).
   - **Files:** `models/dreamer_v4_rssm.py` (apply after c update), TSSM parity, `training/train.py` jsonl, `tools/_smoke_rssm.py`.
4. Return-to-`test_sim` for R2 headroom / critic_rew_to_tgt_var (P3 collapse is standing) after a fair plant verdict.

## RCA this visit

- SysID: OP-varying DC is real (equal-% + 1/feed; MV amp **2.23×** / DV **2.34×**). Median G is still the wrong pin **once local G is in WM-norm**. P118 local MV 0.015 is units, not that 2.23× span.
- Signal: P119 locgfix teacher CONFIRMED (jsonl @37 MV **1.00** skip **0** recon **0.009**). **Inf@13 ABSENT**. Wrap@28 **16.2×** recovered no lock. Probe@30 H=56 **+0.455** above floor / conv **0** — **not** GAIN-READY. GPU P119 P1 **~14.8 GB**. Disk `/home` 64% / 62G. Keep P119+P118+P117+P116+P64+P53.
- Control: do not score P118 actor (`skip_invalid_p3`; freeze GAIN_NOT_READY). Score P119 only if freeze GAIN-READY **and** P3. Do not kill P119 / second GPU.
- ML: LPV `wm_op_scale_dev` **0.81** (not sat). Wrap@28 **16.2×** (below 20× lock) recovered @29 — wrap hygiene, not freeze. Probe@30 H=56 still above floor but conv **0** / H=1 below — **not** GAIN-READY. persist **14@34** recovered — not persist-31. 20× lock@60-class still ahead. `wm_best` @30 EMA **4.512** is gain-blind. P118 freeze **0.68@MV** is not a P119 falsifier. extra-P1 lottery closed.
- Plant: HeatExchangerTower `step` is engineering; APCEnv denorms. Teacher must use `_prev_cmd_norm`. Restore must not leave a post-FD leftover when the snapshot was `None`.
- Metric: jsonl MV ratio vs 0.015 is not val TM. Cap 0.68@MV is last_ok TM vs identified, still GAIN_NOT_READY because the teacher never trained identified-scale MV G. Lineage gain PASS (rel_err 0.39) is looser than band [0.8, 1.3].

P117 process died mid-P2 (SIGKILL). P118 completed P2 then `[p3-skip]`. Do not identity-relaunch P117 or P118. P119 is LIVE.
