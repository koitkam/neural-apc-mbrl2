# GOAL_PLAN — neural-APC-mbrl2

Living plan. Update every visit (live analysis and EXIT). Champions live in `docs/RUN_HISTORY.md`. **`RUN_HISTORY` appends at P1-gate / P2→P3 / EXIT only** — not LIVE@iter snapshots.

**Product:** simulator-agnostic neural APC — smooth CV on the economic limit without violating; faithful observer; unmeasured-load rejection. Envelope: learned observer + neural Kalman/DOB + neural actor-critic. No gray-box plant, no PID/LQR/MPC as the product, no DV-only FF.

Plant this cycle: `nonlinear_sim` HeatExchangerTower (P118 EXIT INVALID CAPPED **0.68@MV**; **P119 locgfix LIVE P1 @78** — wrap@77 20× lock then skip-storm **1** GAIN_NOT_READY **0.56@MV** cap-deferred / recovered; probe@70 conv **0.75** / **not** GAIN-READY; then return-to-`test_sim`).

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

**Trusted:** val TM ss/@H/curve_iae (MV and DV); gain decomp real→post / post→1step / 1step→OL; `det_r` + pred_std vs true; paired econ vs baseline **and** champion when freeze GAIN-READY and P3 ran; CV d2/reversal (`cv_d2_rms_normed` + `cv_reversal_rate` — **not** the suite's current `smooth_pass`); `cv_opt_headroom` / `cv_viol_frac`; DR `cv_return_headroom` / `cv_return_time_frac`. Skip-storm 5-level TM (`DCgain_ratio` / `@H` / `compound_ok`) is a freeze/RCA diagnostic, not a champ score. Launch `|du_mv|` vs teacher `step` (P119 print): O(step) = WM-norm; ≫5×step = engineering-unit bug. jsonl `wm_op_scale_dev` is observability that LPV moved, not a TM score.

**On trial / do not GPU-optimize:** jsonl teacher ×1 vs SysID median (tautology on OP-varying plants — P117); **P118 jsonl `gain_match_mv_ratio` vs local G** (teacher Δu was engineering `_prev_control` − WM-norm action → local MV G **0.015** vs identified **2.63**; extra-P1@98 jsonl MV **1.20** and cap last_ok@108 jsonl MV **0.71** are still vs G≈0.015, not TM); ss-ratio alone; lineage `wm_gain_pass` (P118 rel_err=0.39 PASS while GAIN-READY failed); mv_reversal as a smoothness gate (**suite `smooth_pass` still keys `mv_reversal_rate≤0.5`** in `evaluation/validate.py:control_quality_gates` — gamed vs the standing residual); CV total variation as a smoothness gate; critic_r without `critic_rew_to_tgt_var`; raw dist R²; pred_std matching true_std when R²_det is −0.96; event IAE without return-to-limit; VALID 9/9 / GAIN-READY / all_pass / family-closed as “residual closed”; beating P64 to KEEP a smoothness/headroom/DR win; **`wm_grad_norm=inf` at a skip-storm restore** (P118@13) as a kill; **recon-spike ~3 iters after P1 inject** as a kill; **persist_rel spikes to 31** as a kill; **H=1 fidelity r** as a freeze score (`wm_best` @100 is gain-blind; warm-restore SKIPPED); **skip-storm 2 0.02@MV** as a last_ok freeze score; **extra-P1@98 live TM 0.11@MV** as the freeze score (cap used last_ok **108** **0.68@MV**); **P118 expert-BC paired −38 vs −212** as actor-econ; **mutating stored gain-c in-place as the P120 `gop` mechanism** (would fight P73 persist 0.1).

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

**LIVE P1 @78 (2026-09-09 16:53:07, ~163 min):** skip_iter **0** last_ok **78 unlocked**. recon **0.0054** (best **0.0028@72**). jsonl MV/DV **×1.002 / ×0.996**. gnorm **0.32 finite**. persist_rel **1.07**. `wm_op_scale_dev` **0.79**. jsonl NaN/Inf **0**. STAGE 1.

**20× lock@60 ABSENT** (P118 recon **0.935**). **Wrap@77 after inject@76** (const/step/expert): recon **0.5081 = 182×** best **0.0028** → last_ok **76 locked**; skip **11**; gnorm **5.05e17 finite** (not Inf); jsonl MV/DV **×0.368 / ×−1.58**. Skip-storm **1 @77** 5-level TM `ready=False` median 3/3 worsts **[0.53, 0.56, 0.58]** `DCgain_ratio[0.56,1.01] @H[0.63,1.07]` worst **0.56@MV** DV **1.01/@H 1.07** `not_noisy=True` `unbiased=False`. **cap-deferred** (`ready_n=0/2`, last_ok GAIN_NOT_READY). Restored `wm_last_ok.pt` iter **76**; unlocked; **@78 recovered**. This skip-storm TM is **not** orig-P1.

**Probe@70:** `H=1:r=+0.587 H=14:r=+0.557 H=28:r=+0.434 H=56:r=+0.444 conv=0.75 drift=0.142 floor=0.40 best_h=56/56 gain_fid=0.974`. vs @60 H=56 **+0.678** / H=28 **+0.688** — long-H slipped, still above floor. **Not** GAIN-READY. `wm_best` EMA **5.992@70** — gain-blind. Next probe ~80.

Predicted remaining: orig-P1@~87 (~18 min); then GAIN-READY vs P118 CAPPED **0.68@MV** / P117 **0.71@DV**. Score actor only if freeze GAIN-READY **and** P3.

Falsifier: still CAPPED ~0.68@MV **after** this WM-norm teacher (WARNING absent; `|du_mv|`~step; jsonl MV ratio O(1); Inf@13 absent). Do **not** use P118’s 0.68@MV as the falsifier.

Do **not** extra-P1 N+1 / identity-relaunch P118 / rewrite the live recipe / second GPU.

After a fair P119 VALID or a fair GAIN_NOT_READY with a **correct** local teacher: return-to-`test_sim` on the worst residual (cadence: 2 jobs on this plant then return). P118 is job 2 of 2 **but** the unit bug means P119 is the fair repeat of the same mechanism, not a third plant-hop.

## Ranked follow-ups (not this GPU until P119 verdict)

1. **P119 locgfix** (LIVE above) — causal for R1 on this plant. Locgfix teacher CONFIRMED. Do not rewrite the live recipe.
2. If P119 GAIN-READY but val TM still short: keep LPV, do **not** extra-P1. First check encode vs ID: `identified_lookback=131` vs `seq_len=128` (WARNING at launch; τ=54.5 sr=4 → 4τ/sr=54.5, K=H=56 is the settle, not the hole). Optionally pin encode L=`identified_lookback` (one attributed change) **only after** a fair GAIN-READY freeze whose val TM is still short.
3. If P119 still CAPPED ~0.68@MV / 0.71@DV **with** WM-norm local G: input-scale LPV is too thin for equal-% DC. **P120 `gop`** (still neural, one mechanism): OP-conditioned **gain-c readout**, not another `1+tanh` on GRU inputs and **not** an in-place write into stored `c`.
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
- Signal: P119 locgfix teacher CONFIRMED (jsonl @78 MV **1.00** skip_iter **0** recon **0.0054** best **0.0028@72**). **20× lock@60 ABSENT**. Wrap@77 182× + skip-storm **1** GAIN_NOT_READY **0.56@MV** cap-deferred / recovered @78. Probe@70 all-H above floor (H=1 **+0.587** H=56 **+0.444**) / conv **0.75** — **not** GAIN-READY. GPU P119 P1 **~14.8 GB**. Disk `/home` 64% / 61G. Keep P119+P118+P117+P116+P64+P53.
- Control: do not score P118 actor (`skip_invalid_p3`; freeze GAIN_NOT_READY). Score P119 only if freeze GAIN-READY **and** P3. Do not kill P119 / second GPU.
- ML: LPV `wm_op_scale_dev` **0.79** (not sat). **20× lock@60 ABSENT** (P118 recon **0.935**). Wrap@77 after inject@76 **did** 20×-lock then skip-storm restore — same class as Wrap@28, **not** a kill / **not** orig-P1. Skip-storm TM **0.56@MV** is last_ok **76** RCA, cap-deferred `ready_n=0`. Probe@70 conv **0.75** / long-H slipped vs @60 — mid-P1 fidelity, **not** freeze. `wm_best` @70 EMA **5.992** is gain-blind. gnorm **5e17@77** recovered finite (P118 Inf@13 was teacher-unit). P118 freeze **0.68@MV** is not a P119 falsifier. extra-P1 lottery closed. P73 persist 0.1 forbids in-place `c*=gop`.
- Plant: HeatExchangerTower `step` is engineering; APCEnv denorms. Teacher must use `_prev_cmd_norm`. Restore must not leave a post-FD leftover when the snapshot was `None`. `identified_lookback=131 > seq_len=128` is a 3-sample encode shortfall, not this freeze.
- Metric: jsonl MV ratio vs 0.015 is not val TM. Cap 0.68@MV is last_ok TM vs identified, still GAIN_NOT_READY because the teacher never trained identified-scale MV G. Lineage gain PASS (rel_err 0.39) is looser than band [0.8, 1.3]. Suite `smooth_pass` on mv_reversal is on trial. No `residual_board.json` producer exists — plan #4, not a GPU job.

P117 process died mid-P2 (SIGKILL). P118 completed P2 then `[p3-skip]`. Do not identity-relaunch P117 or P118. P119 is LIVE.
