# Training Run History — neural-apc-mbrl

Lab-notebook ledger of every training run: the **change/hypothesis**, the
**key results**, and the **conclusion/next action**. Maintained by the
`dreamer-training-diagnosis` skill — a new row is appended (or the run's row
updated) at the end of **every** run diagnosis/verdict. Newest at the bottom.

> **2026-07-07 — neural-apc-mbrl2 fork (real-sim controller).** From here the
> actor no longer trains in WM imagination: imagination is **deleted**, the
> WM(RSSM)+DOB is a **frozen observer**, and the actor-critic trains on λ-returns
> from **real rollouts of the true simulator** with domain randomisation
> (`_realsim_actor_critic_step`, `actor_train_source='realsim'` default). The
> p95→p143 rows below are **imagination-era history** (inherited from the source
> `neural-apc-mbrl` repo), kept for context. New mbrl2 runs
> (`run_YYYYMMDD_realsimN`) are model-free on the real plant — judge them on the
> same validation suite (econ vs baseline, CV/MV violation, disturbance
> rejection); the imagination-only metrics (`imag_adv_action_corr`, the
> `return_scale` cascade, the WM-@H-gain-as-seen-by-the-actor) no longer apply.

- Plant unless noted: `test_sim` (SISO, gain≈−0.28, τ=53, θ=8, sample_rate=4,
  ep_len=1220, H=55; 1 MV REFLUX / 1 CV CONTROL_TEMP 78.5–85.5 / 1 DV FEED).
- Metric glossary: **gain** = `wm_gain_rel_err` (0=perfect; ≤0.186 = p106-good);
  **reward_r** = validation `reward_head_r` (reward-MTP corr, ≥0.3 healthy);
  **mv_tv** = MV total variation (≈979 good, ≫ = oscillation); **cv_viol** =
  mean CV violation; **econ** = agent economic score (less negative = better;
  p106 −30.6 = +69% over baseline); **decomp** = posterior-prior gain decomp
  (real→post / post→1step / 1step→openloop); **dist r/R²** = disturbance
  prediction (d_t or P87 head) correlation / R².
- Deep narrative + RCA detail lives in `/memories/repo/mbrl_open_items.md`
  (agent memory). This file is the scannable cross-run history for humans.

## BEST-RUN BASELINES (per subsystem) — UPDATE THIS EACH VERDICT

The current champion per subsystem — the baselines a new run must **beat** (or
not regress below). Update the row whenever a run sets a new best. `test_sim`.
Targets: gain ratios →1.0, disturbance **detrended** r→1 / R²→+1, critic
`critic_rew_to_tgt_var` >0.015 (P3 mean), actor `cum_raw`→0 + `cv_viol`→0.

| Subsystem | Metric (target) | Champion | Value | Notes |
|---|---|---|---|---|
| **MV WM gain** | `wm_transfer` ss-ratio (→1.0) | **P26** (mbrl2) | **×0.973 / @H ×0.880** | full-BPTT gain-match + live `n_dist`. Imagination-era p128 0.967 is not the current fork. |
| **DV WM gain** | `wm_dv_transfer` ss-ratio (→1.0) | **P26** (mbrl2) | **×0.87** | residual under-weight vs MV; abs Huber. Imagination p131 0.868 used DV-FF (wiped). |
| **Disturbance (Kalman, dynamic)** | `r_detrended` (→1) | **P26** (mbrl2) | **det_r 0.68** | P25 was 0.24. Imagination p129 0.593/0.778 is a different stack. |
| **Critic** | `critic_rew_to_tgt_var` (>0.015) | none healthy | — | P45 first valid P3: warmup rtgt **0.08–0.09** then **0.0004** after actor unfreeze. Freeze rscale KEEP (1.15→**2.18**, not 49.5). min-of-2 **FALSIFIED as sufficient**. |
| **Actor** | econ vs baseline + vs expert BC | **P53** (mbrl2) | paired **−13 vs −98** | PPO μ-ratio clip; 9/9 pairs. Beats P50 **−56 vs −104**. Train ES still entropy-thr vs H(σ_init). |

> **Note on the disturbance metric (2026-06-20, control-theory re-frame)**: the
> DOB `d_t` feeds **forward**, and a slow drift in the estimate (timescale ≫
> closed-loop settling) is rejected by the feedback **integral action** (`S(jω)→0`
> as ω→0) — so it is *benign*. The RAW R² is dominated by that drift and is the
> WRONG score for feed-forward quality. The champion is now ranked on the
> **high-pass-detrended** `r2_detrended` / `r_detrended` (window = 4× the settling
> horizon), which isolates the DYNAMIC tracking that actually reaches the CV.
> Re-scored on this metric, **p129 (0.593/0.778) and p128 (0.558/0.747) are the
> best** — exactly the runs that *looked* good — while p130/p131 show a **real
> dynamic** collapse (det R² <0, dyn-error ≈2×), not merely drift. The dynamic
> error is downstream of the MV WM gain (#1) + actor activity — fix the gain first.

## Lineage at a glance

> **Backfill caveat (p95–p105)**: these predate the MC-critic (landed p106) and
> likely the 4× WM-gain-horizon fix (commit 830cdc8), so their `gain` is on the
> old anchor-critic regime and NOT directly comparable to p106+. Rows are a
> lightweight JSON backfill (no plot re-inspection). Treat as historical context.

| Run | Date | Change / hypothesis vs prev | Headline result | Verdict |
|---|---|---|---|---|
| p95 | 2026-06-07 | joint-mode isolation baseline (anchor-critic, pre-MC) | gain 0.402, reward_r −0.30, econ −33.6, cv 20.5, mv_tv 2002 | 🔬 baseline; reward head anti-correlated |
| p96 | 2026-06-07 | γ0.985 + anchor_coef_long=2.0 + critic_imag=0.1 (critic bundle) | gain 0.377, reward_r 0.13, econ −72.6, cv 56.8 | ❌ econ/cv worse |
| p97 | 2026-06-07 | flat BC (`expert_bc_p3_floor=1.0`) | no validation (aborted/failed) | ⏹️ no result |
| p98 | 2026-06-07 | `reward_head_exclude_expert=True` (reward-head fix) | reward_r 0.003, econ −37.6, cv 22.2 | 🔬 reward-head decoupled from expert (kept) |
| p99 | 2026-06-07 | + DV-as-input (measured DV as exogenous WM input) | gain 0.257, reward_r 0.30, econ −39.1, cv 23.9 | ✅ DV-input helps reward_r+gain (kept) |
| p100 | 2026-06-07 | + `return_value_cap_gamma_horizon` (cap fix) | gain 0.164, reward_r 0.12, econ −71.9, cv 37.4 | 🔬 gain good, econ noisy (cap kept) |
| p101 | 2026-06-08 | γ0.985→**0.97** | gain 0.708, reward_r 0.29, econ −107.6, cv 92.1, mv_tv 3621 | ❌ regressed hard at γ0.97 alone (needed MC-critic) |
| p102 | 2026-06-08 | disturbance τ-fix + low-freq spread | gain 0.331, reward_r 0.07, econ **−25.5**, cv 13.7, mv_tv 1018 | ✅ best econ pre-MC; disturbance fix kept |
| p103 | 2026-06-08 | B=6 (`bound_training_reward_max=6.0`, aggressive) | gain 0.328, econ −114.1, cv 97.5 | ❌ B=6 cascaded → keep B=3 |
| p104 | 2026-06-08 | `rssm_free_bits=0.25` | no validation (aborted/failed) | ⏹️ no result |
| p105 | 2026-06-08 | excitation reinject into SHARED buffer (every 5 iters) | gain 0.288, reward_r −0.08, econ −74.7, cv 50.7, mv_tv 4157 | ❌ shared-buffer reinject HURT actor (→ later WM-only partition) |
| p106 | 2026-06-09 | MC-critic + γ0.97 + B3 + DV-input + WM recipe (the proven stack) | gain 0.186, reward_r 0.177, mv_tv 979, cv_viol ~11, econ −30.6 (+69%) | ✅ **BEST baseline (KNOWN-GOOD)** |
| p107 | 2026-06-09 | econ-led `OBJ_AUTO_CV_OVER_ECON_RATIO=1.0` + early-stop 120 | gain 0.094 (best WM) BUT mv_tv 4019, cv_viol 46, econ −108.5 | ❌ FAILED — constraint limit cycle |
| p108 | 2026-06-09 | econ-led 1.0 + integral-boost OFF (single-var ablation of p107) | cycle gone (mv_tv 1777, mv_viol 0) but parks outside limit: cv_viol 72, econ −131, gain 0.183 | ❌ econ-led not ready; p106 stays best |
| p109 | 2026-06-09 | WM-fix levers: recon_cv=6 + excitation=0.4 + P87 dist-head trains (no stop-grad) | gain 0.365 (worse), reward_r −0.086, cv_viol 54, decomp 0.783/–/0.660 | ❌ dist-head flooded the WM, regressed |
| p110 | 2026-06-10 | recon_cv=3 + excitation=0.4 + dist-head **stop-grad** | gain 0.311, reward_r 0.534, cv_viol 54, decomp 0.815/–/0.930 | ❌ still regressed vs p106 (recon_cv backfired) |
| p111 | 2026-06-10 | clean p106 replica (control) | — (killed before completion) | ⏹️ aborted |
| p112 | 2026-06-10 | Gd hidden-disturbance ON (realistic FOPDT load) | gain 0.357, actor best-but-oscillates, mv_tv 1855 | 🔬 omitted-variable confound visible |
| p113 | 2026-06-10 | **Exp A**: hidden-disturbance OFF (ablation) | gain 0.176, real→post 0.940, mv_tv 813 | 🎯 **DECISIVE** — omitted-variable attenuation confirmed |
| p114 | 2026-06-11 | **DOB Scope 1** (neural Kalman filter; d_t output-additive only) | gain 0.365 (NOT recovered), reward_r 0.024, mv_tv 1007, cv_viol 31.5, econ −48.7; decomp 0.798/**1.000**/0.850; dist r **+0.70** R² −0.55 | ⚠️ prior dynamics perfect + dist-corr positive + no oscillation, BUT gain not recovered (autoencoder) + **actor PASSIVE** (mv_viol 0.13 vs p106 35.9) |
| p115 | 2026-06-11 | **DOB + Scope 2** (d_t fed into feat) + excitation 0.6 + recon_cv 4 + P87 head retired | gain **0.298** (healthy✓, ↓ from p114 0.365), reward_r **0.160** (recovered from 0.024), real→post **0.886**, dist r 0.64 R² **+0.30** (flipped +); econ −49.2, cv_viol 28.1 | ⚠️ **WM #1+#2 advanced** (gain healthy, dist R² positive) but not yet p106's 0.186; actor still mv_viol≈0 (econ #4 deferred); residual = autoencoder+compounding |
| p116 | 2026-06-12 | **Stage 1 of staged plan**: clean data (`HIDDEN_DISTURBANCE=0`) + Kalman/DOB OFF + excitation 0.6 + recon_cv 4 + **compile ON** (default) | killed @iter270 (~20%, joint, redundant) — confirmed compile default-on works end-to-end; recon converged ~0.02–0.10 | ⏹️ superseded by p117 (its Stage 1 = the clean-WM probe, in phased mode) |
| p117 | 2026-06-12 | **Staged curriculum** (phased): S1 clean+DOB-suppressed → S2 freeze-g+observer-id → S3 frozen-WM actor + DR; DOB + curriculum_enabled, phases 0.45/0.25/0.30, recon_cv 4, compile on | gain **0.217** (healthy), **all_pass=1 (FIRST in series)**, reward_r **0.436 (best)**, real→post **0.926 (best)**, lever→**compounding**; actor **ACTIVE** (mv_viol 0.295, mv_tv 799 smooth, no cascade), econ −39.0; **dist R² −0.626 (REGRESSED)** | ✅ **curriculum WORKS** — #1 WM all-pass, #3 critic healthy, #4 actor active+smooth; ONE regression: #2 dist amplitude (d_t over-shrunk by dob_reg on the better clean g) |
| p118 | 2026-06-13 | `DOB_REG_COEF 0.01→0.002` on the p117 recipe (dob_reg #2 fix) | killed @iter19 (~7%) — superseded by p119 (old code = no DV-gain fix → confounded #2 signal) | ⏹️ superseded by p119 |
| p119 | 2026-06-13 | **p117 + TWO independent fixes**: (1) **step-test re-injection in P1** (`STEP_TEST_INJECT`) → fixes the DV→CV gain bias (was 0.62 — DV step-test seeds were evicted before the WM froze); (2) **`DOB_REG_COEF 0.01→0.002`** → fixes the #2 disturbance amplitude (R² was −0.626). Independently measured (DV gain ratio vs disturbance R²) | see later p119 verdict | imagination-era (not mbrl2) |
| P26 | 2026-08-25 | full-BPTT gain-match + live `_replay_n_dist` | MV ss/@H ×0.973/×0.880, DV ×0.87, det_r 0.68 | ✅ **mbrl2 observer champion** |
| P27 | 2026-08-25 | relative Huber | P1 skip-storm abort @50 | ❌ DISCARDED (then REMOVED) |
| P28 | 2026-08-26 | abs Huber; restore gain-blind `wm_best` | val MV ×0.52; actor INVALID | ❌ restore default OFF |
| P29 | 2026-08-26 | skip-restore OFF; env-free dropped latent+compile | categorical leftover; actor INVALID | ❌ det+eager defaults |
| P30 | 2026-08-26 | env-free det+eager; cap P1 on first skip-storm | froze iter 18; val MV mean ×1.88; P3 skipped | ❌ cap-on-first REVERTED |
| P31 | 2026-08-26 | `storm_cap=2`; no skip-storm | healthy P1 to 94; CAPPED exploded g; DV ×0.11 | ❌ freeze last-ok (P32); storm_cap KEEP |
| P32 | 2026-08-27 | last-ok at detonated freeze; continue-after-storm | storm 1/2 @53 KEEP; val MV ×1.09 DV ×0.68 det_r 0.11; `[p3-skip]` | ❌ CAPPED 0.71@DV — extension closed; keep-ext = P33 |
| P33 | 2026-08-27 | keep P1 extension on skip-storm continue | val MV ×1.08 DV ×0.66 det_r 0.40; `[p3-skip]` | ❌ keep-ext KEEP as mechanism; FALSIFIED as DV lever |
| P34 | 2026-08-27 | AM/HM inv-var isolation | iso 7088 skip 99 ES iter 1 | ❌ formula REVERTED |
| P35 | 2026-08-27 | mean-1 per-seq inv-var isolation | storm 2/2 @16 CAPPED GAIN_NOT_READY 0.01@DV; val DV ×0.013 | ❌ per-seq SUPERSEDED |
| P36 | 2026-08-27 | per-input \|G\|² inv-var isolation | storm 2/2 @7 CAPPED 0.00@DV; val DV ×0.004 | ❌ inv-var DISCARDED then **REMOVED** |
| P37 | 2026-08-27 | abs isolation (P33 recipe) | val MV ×0.981/×1.005 DV ×0.690/×0.783 det_r 0.370; `[p3-skip]` | ❌ KEEP as P1 form; FALSIFIED as DV pin |
| P38 | 2026-08-27 | dcv_match match-at-g_min (no floor) | val MV ×1.25 DV ×0.007 det_r 0.43; storm 2/2 @48; `[p3-skip]` | ❌ FALSIFIED |
| P39 | 2026-08-27 | dcv_match scale floor 1.0 | val MV ×0.954/×0.954 DV ×0.679/×0.785 det_r 0.326; CAPPED 0.70@DV; `[p3-skip]` | ❌ KEEP as P1 form; FALSIFIED as DV pin |
| P40 | 2026-08-28 | isolation auto-enable OFF (gain-match-only) | val MV ×0.995/×0.964 DV ×0.723/×0.743 det_r 0.490; CAPPED 0.75@DV; `[p3-skip]` | ❌ KEEP as env-free default; FALSIFIED as DV pin |
| P41 | 2026-08-28 | P1 gate recent-floor (not warmup ema_best) | val MV ×0.986/×0.973 DV ×0.700/×0.775 det_r 0.079; CAPPED 0.74@DV; `[p3-skip]` | ❌ KEEP as mechanism; FALSIFIED as DV pin / extra-P1 lever |
| P42 | 2026-08-28 | last-ok lock after silent recon spike (20×) | val MV ×1.179/×1.161 DV ×0.737/×0.804 det_r 0.124; freeze 0.75@DV; `[p3-skip]` | ❌ KEEP as 20× fire; FALSIFIED as DV pin / det_r fix; <5× overwrite untested |
| P43 | 2026-08-28 | per-input Huber β = \|tgt_ij\| (L1 sat ±1) | val MV ×0.985/×0.993 DV ×0.740/×0.849 det_r −0.215; freeze 0.76@DV; `[p3-skip]` | ❌ KEEP as P1 form (not P27); **FALSIFIED as DV pin** |
| P44 | 2026-08-28 | held settle S=H before gain-match FD | storm 2/2 @66 G_pred≈0; val MV ×0.926/×0.943 DV ×0.751/×0.842 det_r 0.099; freeze last_ok 57; `[p3-skip]` | ❌ **REVERT** env-free settle `-1`; do not retry S=H |
| P45 | 2026-08-28 | TM-protocol rest-IC (`DREAMER_GAIN_MATCH_REST_IC=1`) | first GAIN-READY since P40–P44; val MV ×0.877/×0.887 DV ×0.815/×0.875 det_r 0.148; P3 entropy-collapse; econ −216 vs −92 | ✅ rest-IC **PROMOTE**; min-of-2 **FALSIFIED as sufficient** |
| P46 | 2026-08-29 | P3 log_std reset (`DREAMER_P3_RESET_LOG_STD=1`, weights-only) | GAIN-READY 0.88@DV; first P3 ent **−0.101**; warmup still railed; ES entropy-collapse @220; econ **−256 vs −129** | ❌ KEEP-AS-OVERRIDE (do not promote); next P47 Adam-complete |
| P47 | 2026-08-29 | P3 σ-reset + Adam log_std-row zero | GAIN-READY 0.90@DV; first P3 ent **−0.101**; unfreeze yank 145 **−0.336** (Adam FALSIFIED); ES entropy-collapse @253; econ **−221 vs −121**; reversal **0.53** | ❌ Adam-complete **FALSIFIED as yank lever**; do not promote `p3_reset_log_std`; next P48 collect DV+Kalman |
| P48 | 2026-08-29 | P3 collect/val stream measured DV + Kalman | ES entropy_collapse @236; best.pt **161**. GAIN-READY live **0.89@DV** @82; freeze last_ok **24** → **0.81@DV**. First P3 ent **−0.283**; rscale **2.51 KEEP**; rtgt 0.059→0.0004; mvv 17k/223k. Val TM skipped ImportError | ❌ KEEP train/serve; **FALSIFIED as cascade lever**. Freeze-24 confound. Next P49 wrap-unlock |
| P49 | 2026-08-29 | original-P1 wrap last-ok unlock | EXIT=0, 292 rows, ES entropy_collapse. Gate **GAIN-READY** @82 last_ok **82**. Val MV **×0.816/×0.830** DV **×0.867/×0.924** det_r **0.632** (amp-dead 0.213 vs 1.93). First P3 ent **−0.283**; rscale **2.10 KEEP**; unfreeze 147 mvv 667k; rtgt 0.048→0.0001; seed kpi **−714**. Scripted dist skipped TypeError; 0-vs-0 false all_pass | ❌ wrap-unlock KEEP hygiene; **FALSIFIED as cascade**. Next P50 `bc_mean_only` |
| P50 | 2026-08-29 | P1/P2 BC MSE-on-μ (`bc_mean_only=True`) | EXIT=0, 286 iters, ES entropy_collapse @286. Freeze last_ok **88** GAIN-READY **0.90@DV**. Val MV **×0.935/×0.877** DV **×0.836/×0.868** det_r **0.290** (amp-dead 0.220 vs 1.93). First P3 ent **−0.107**; unfreeze 169 yank; rscale **2.33 KEEP**. Val paired **−56 vs −104 BEATS** (9 pairs); seed kpi **−74**. best.pt **191** | ✅ μ-only KEEP as σ-open + first val econ beat; **FALSIFIED as yank lever**. Next P51 stop-grad log_std |
| P51 | 2026-08-30 | P3 REINFORCE stop-grad log_std (`p3_stop_grad_log_std=True`) | EXIT=0, 361 iters, ES **p3_plateau** @361 (not entropy_collapse). best.pt **161** det **−86.6**. Gate **GAIN-READY** 0.86@MV / 0.88@DV @82; last_ok **82 unlocked**. First P3 ent **−0.125**; unfreeze 147 ent **HELD**. Dip −0.282 @315 recovered −0.106. Val MV **×0.812/×0.782** DV **×0.819/×0.855** det_r **0.626**. Paired **−72 vs −111 BEATS** (worse than P50 **−56 vs −104**); seed kpi **−82**. reversal **0.143**. rscale **2.17 KEEP**. | ✅ KEEP as unfreeze-yank + non-sticky-floor; **FALSIFIED as cascade lever**. Next P52 `p3_logp_clip=8` |
| P52 | 2026-08-30 | P3 REINFORCE logp clip (`p3_logp_clip=8` per action dim) | EXIT=0, 209 iters, ES **entropy_collapse** @209 (thr −0.083; ent HELD −0.111). best.pt **176** det **−634**. Gate **GAIN-READY** @82 last-ok **78** MV **0.95** DV **0.84**. Unfreeze 147 delayed bound then @165 std **80.8** / actor **−5.80**. Val MV **×0.925/×0.927** DV **×0.815/×0.860** det_r **0.127**. Paired **−377 vs −87 FAIL** (0/9; worse than P50 **−56/−104** and P51 **−72/−111**); seed kpi **−306**. rscale **2.83 KEEP**. | ✅ KEEP as delayed first-unfreeze bound; **FALSIFIED as cascade lever**. Next: in-support μ-walk (clip cannot stop \|logp\|<8 SGD) |
| P53 | 2026-08-30 | P3 PPO ratio clip vs frozen unfreeze-μ (`p3_mu_ratio_clip=0.2`) | EXIT=0, 262 iters, ES **entropy_collapse** @262 (thr −0.083; ent HELD −0.101; **not** μ-rail). best.pt **166** det **−46.2**. Gate **GAIN-READY** @82 0.88@DV last_ok **82**. Unfreeze 147: logp_std spiked 6.71@149 then **held ~0.47**. Val MV **×0.900/×0.898** DV **×0.848/×0.903** det_r **0.279**. Paired **−13 vs −98 BEATS** (9/9; beats P50 **−56/−104**, P51 **−72/−111**, P52 **−377/−87**); seed kpi **−13**. rscale **2.30 KEEP**. | ✅ KEEP as μ-walk limiter **and** cascade. Actor champion. Next P54 ES σ-band frac |
| P54 | 2026-08-30 | ES entropy thr = σ-band frac (`early_stop_entropy_collapse_floor_frac=0.25`) | EXIT=0, 446 iters, ES **p3_plateau** @446 (not entropy_collapse). best.pt **246** det **−36.2**. Gate **GAIN-READY** @82 0.91@DV last_ok **82**. Open-σ **−0.101**; **0/310** below new trip −0.238. Val MV **×0.821/×0.815** DV **×0.835/×0.874** det_r **0.280**. Paired **−26 vs −101 BEATS 9/9** (worse than P53 **−13 vs −98**); seed kpi **−26**. rscale **2.06 KEEP**. clip_frac 0.42→0.11 | ✅ KEEP as ES false-trip fix; **FALSIFIED as actor-econ lever**. Champion stays P53. Next P55 μ-ratio refresh |
| P55 | 2026-08-30 | PPO μ-ratio recopy every P3 iter (`p3_mu_ratio_refresh_iters=1`) | LIVE env-free `run_p55_murefresh` pid **103504** sha `b321d89`. Watch clip_frac not die to ~0.11; val vs P53 −13/−98 | LIVE |

## Run details

### p114 — DOB Scope 1 (neural Kalman filter, d_t output-additive)
- **Change**: `DREAMER_DOB_ENABLED=1` on the p106 stack. d_t integrates the
  one-step CV innovation and is added to the decoded CV (`CV = g(feat)+d_t`) to
  de-confound the omitted-variable gain attenuation proven in p113.
- **Result**: training fine (best.pt det −73.99 @iter201 = best of the series).
  Validation: gain **0.365** (≈p112's 0.357 — NOT recovered); reward_r collapsed
  0.177→**0.024**; mv_tv 1007 (oscillation gone); cv_viol 31.5 (one outlier seed
  max 124); econ −48.7 (regressed from −30.6). Decomp real→post 0.798
  (autoencoder lever) / post→1step **1.000** (DOB cleaned the prior!) /
  1step→openloop 0.850. Disturbance d_t r **+0.70** (vs p112 head −0.29) but
  R² −0.55 (mis-scaled).
- **RCA (structural)**: the transfer probe rolls open-loop where d→0, so it
  measures g WITHOUT the DOB → gain unchanged; the residual loss MOVED to the
  autoencoder + compounding, which the DOB can't fix. **Actor passivity**: d_t
  lives only in OUTPUT space (not in `feat`), so the actor/reward heads are
  BLIND to it → imagined world is disturbance-free → actor minimises MV
  (passive: mv_viol 0.13 vs p106 35.9) and reward_r collapses.
- **Verdict / next**: keep the DOB (prior fixed, dist-corr positive, no
  oscillation) but (a) implement **Scope 2** = feed d_t into feat (cure
  passivity); (b) attack the now-dominant autoencoder with recon_cv +
  open-loop excitation; (c) retire the redundant P87 head. → **p115**.

### p115 — DOB + Scope 2 + open-loop excitation
- **Change vs p114**: Scope 2 (RSSMState/TSSMState `feat = [h, z_flat,
  d.detach()]` so the actor/critic/reward heads condition on d_t — explicit
  feed-forward; decoder still reads the clean core) + `WM_EXCITATION_BUFFER_FRAC=0.6`
  (open-loop step-tests de-confound gain↔d_t identifiability — p109/p110 used
  only 0.4 AND had the confounding head) + `WM_RECON_CV_WEIGHT=4.0` (autoencoder)
  + P87 head retired (`DISTURBANCE_LOSS_SCALE=0.0` → `disturbance_head_dim=0`;
  the DOB d_t replaces it).
- **Result** (326 iters, p3_plateau early-stop, best det −98.1 @iter126): gain
  **0.298** (HEALTHY, down from p114's 0.365 — open-loop excitation + head
  removal recovered part of it, but NOT to p106's 0.186); reward_r **0.160**
  (recovered from p114's 0.024 — Scope 2 gave the head the disturbance signal);
  decomp real→post **0.886** (autoencoder, improved from 0.798) / post→1step
  1.000 / 1step→openloop 0.854 (compounding, ≈unchanged); disturbance d_t r 0.64
  **R² +0.30** (flipped positive from p114's −0.55 — d_t now correctly scaled).
  Economics still regressed: econ −49.2, cv_viol 28.1, mv_viol≈0 (actor still
  not actuating much in steady state — but this is the #4 priority, deferred).
- **Read**: Scope 2 + excitation advanced the TWO top priorities — **#1 WM gain**
  (0.365→0.298, now in the healthy band) and **#2 disturbance** (R² −0.55→+0.30,
  reward_r recovered). The gain is NOT yet at p106's 0.186; the decomp localises
  the residual to the **autoencoder (0.886)** + **compounding (0.854)** — neither
  is a disturbance-confounding problem the DOB can fix, and co-training the WM
  with the actor keeps re-contaminating it.
- **Verdict / next**: keep DOB+Scope2 (clear progress on #1+#2). The remaining
  gain gap is the autoencoder/compounding under closed-loop co-training → the
  next lever is the **pure open-loop WM+DOB pretrain-then-FREEZE** (train the WM
  on open-loop excitation until the gain converges, freeze the WM core incl. the
  DOB A/K, THEN train actor/critic on the static unbiased WM). Compile refactor
  (below) lands first so the pretrain phase is fast.

### p116 — Stage 1: clean-data WM (compile default-on)
- **Context**: first stage of the user's staged clean→disturbance curriculum
  (the proper Kalman/DOB design: identify the plant on CLEAN data, THEN fit the
  observer on the fixed plant). Launched standalone (no new code) while the full
  3-stage curriculum is built.
- **Change vs p115**: `DREAMER_HIDDEN_DISTURBANCE=0` (CLEAN — no unmeasured
  disturbance; measured DV + noise + DR stay) + DOB **OFF** + `DISTURBANCE_LOSS_SCALE=0`
  (P87 head retired, `disturbance_head_dim=0`); keeps excitation 0.6 + recon_cv 4;
  **`torch.compile` ON** (the default — stopped passing `DREAMER_COMPILE=0`; the
  refactor f0faa3b made the DOB graph compile, and this DOB-off run is the proven
  p106 compile path = live end-to-end compile validation).
- **Purpose / judge by**: the unbiased-WM **gain ceiling** — with zero
  omitted-variable confound + the recon_cv/excitation levers, how low can
  `wm_gain_rel_err` go (expect ≤ p113's 0.176)? + decomp `real→post` → ~1.0
  (isolates how much of p115's gap was confound vs autoencoder). Do **NOT** judge
  by actor econ / disturbance-rejection — a clean-trained actor will not reject
  disturbances **by design**; the disturbance-capable actor is **Stage 3** of the
  curriculum (with disturbances + domain randomization for runtime robustness).
- **Next**: build the integrated 3-stage curriculum (clean-WM → freeze-g-not-DOB
  + disturbance+DOB on → actor) as ONE run; this p116 clean WM is the reference
  for the achievable gain ceiling.

### p117 — Staged curriculum (the payoff run)
- **What**: the full 3-stage curriculum executed flawlessly. `dob_d_absmean` by
  stage: P1=0.0 (suppressed ✓) → P2=0.088 (observer learning ✓) → P3=0.139
  (active feedforward ✓). Warm-restore loaded the best clean WM (iter 70) at
  P1→P2 before freezing.
- **Result — the best run of the series**:
  - **WM #1**: gain `rel_err 0.217` (healthy); **all_pass=1 — the FIRST run in the
    series to pass every internal fidelity gate** (wm_r 0.537, reward_r 0.436,
    critic_r 0.810). Decomp: real→post **0.926 (best ever)** / post→1step 0.994 /
    1step→openloop 0.836 → **dominant lever is now COMPOUNDING, not autoencoder**
    (the clean staged ID fixed the autoencoder).
  - **#3 critic**: healthy, calibrated (critic_pred_target_r 0.994, critic_r 0.810).
  - **#4 actor**: **ACTIVE again** (val mv_viol 0.295 vs p115's passive 0.000),
    **smooth** (mv_tv 799, below p106's 979), no cascade (return_scale stable 9.9);
    econ −39.0 (better than p115 −49.2, short of p106 −26.0 — but p117's actor
    faces disturbances p106's never did).
  - **#2 disturbance — REGRESSED**: d_t R² **−0.626** (p115 was +0.30). r=0.606
    (direction right) but NRMSE 1.275 (amplitude too small). Cause: the better
    clean `g` (0.926) explains more CV movement → smaller residual → `dob_reg`
    (0.01) over-shrinks d_t → amplitude under-predicted.
- **Verdict**: the curriculum is a **keeper** — it cured the p115 actor passivity,
  produced the first all-pass WM, and lifted the autoencoder. The single
  regression (#2 amplitude) has a clean, **safe** fix: lower `dob_reg` — and
  because `g` is FROZEN in Stages 2/3, a larger d_t **cannot steal gain** (the
  p114 failure mode is structurally impossible now).

### p118 — DEVIATION: dob_reg fix (not the planned recon_cv)
- **Why deviate**: the planned p118 (recon_cv 4→6-8 to attack the autoencoder) is
  **mis-targeted** — p117's decomp proves the autoencoder is fixed (real→post
  0.926) and the bottleneck moved to **compounding**; recon_cv also has a backfire
  history (p109/p110 made the gain worse). The pressing issue is the **#2
  disturbance regression** (user priority #2), not the autoencoder.
- **Change vs p117 (single variable)**: `DREAMER_DOB_REG_COEF 0.01 → 0.002` so the
  Kalman d_t isn't over-shrunk → amplitude matches → R² recovers. Everything else
  = p117 (also drops the now-removed `wm_excitation_buffer_frac` knob).
- **Judge by**: dist R² back > 0 (amplitude, NRMSE → ~1) **while** WM gain (0.217),
  all_pass, reward_r (0.436) and the active/smooth actor hold p117 levels.
- **If it works**: promote `curriculum_enabled` + `dob_enabled` to default-on. The
  remaining WM-gain refinement (compounding 0.836) is a separate, lower-priority
  lever (raise overshoot/held-rollout coefs), not recon_cv.

### p119 — step-test re-injection (DV-gain fix) + dob_reg 0.002 — MIXED
- **Recipe**: p117 curriculum + **step-test-inject** (`EVERY=20 N=2`, re-injects
  isolated MV+DV step events into P1 so the DV→CV gain stays supervised to the
  WM freeze) + `DREAMER_DOB_REG_COEF 0.002` (the p118 disturbance fix). Phases
  P1=1‑86, P2=87‑150, P3=151‑492.
- **#1 WM gain — IMPROVED (step-test-inject WORKED)**: aggregate gain rel_err
  **0.217 → 0.164** (best yet), `all_pass` HELD. **Per-input** (the open user
  question): MV ratio **0.783 → 0.836**, DV ratio **0.625 → 0.761** — DV improved
  most, exactly what step-test-inject targets. Decomp real→post 0.926→0.933,
  1step→openloop 0.836→**0.884** (compounding improved). Residual DV bias is a
  *genuine ~24% under-read*, NOT a horizon artifact (DV WM curve settled by H:
  0.753@¾H → 0.760@end).
- **dob_reg 0.002 — BACKFIRED on #2 + actor**: lowering the L2 prior let `d_t`
  grow (P3 dob_d 0.139→**0.246**) but it became **mis-scaled + sign-flipped**
  noise — disturbance R² **−0.626 → −2.48**, r **+0.606 → −0.058** (lost
  direction), NRMSE 1.275→1.865 (vision: blue d_t often opposite-sign to true).
  The actor conditions on this corrupted d_t → **passive again** (mv_viol
  0.295→**0.000**) and parks outside limits (cv_viol 24→**86**) → econ −39→**−105**.
- **Lesson**: `dob_reg` is NOT the #2 lever to loosen on a clean curriculum WM —
  a smaller residual (clean `g` explains more CV) leaves d_t *less* signal, so
  loosening reg amplifies noise/sign error rather than recovering amplitude.
  **Keep `dob_reg=0.01` (p117).** step-test-inject is a clear **KEEP**.

### p120 — revert dob_reg + STRONGER step-test (reduce DV bias further)
- **Two changes vs p119** (well-isolated): (A) **REVERT** `DREAMER_DOB_REG_COEF
  0.002 → 0.01` (fixes the p119 actor passivity + disturbance sign-flip; back to
  p117 known-good). (B) **STRENGTHEN** step-test-inject `EVERY 20→10`, `N 2→4`.
- **Why (B) — decisive checkpoint timing**: dynamics `g` (which holds the DV→CV
  gain) trains ONLY in P1 then FREEZES at P1→P2. `wm_best` peaked **iter 60**;
  the P1→P2 warm-restore loaded iter-60 and **discarded iters 61‑86**. So the DV
  gain is set entirely by step-test data present **before iter 60**. p119 fired
  at iters 20/40/60/80 but iter-80 was discarded → only **6 episodes** (3 cycles
  ×2) shaped the gain, and the buffer saturated@iter40 (FIFO-evicting the iter-20
  batch). p120 `EVERY=10 N=4` injects at 10/20/30/40/50/60 = **24 episodes**
  (6 cycles ×4) concentrated in the iters 1‑60 gain-learning window — **4× the
  effective DV freshness at the peak**.
- **Judge by**: DV ratio **0.76 → >0.85** (MV holds ~0.84+), `all_pass` held,
  actor **ACTIVE again** (mv_viol > 0.2, cv_viol < 25, econ beats −39),
  disturbance r back **positive** (≳ +0.5 like p117).
- **If DV still < 0.85**: escalate `EVERY=8 / N=6`, or make `wm_best` selection
  **gain-aware** (it is recon-fidelity only today, so it can discard a
  better-gain late-P1 checkpoint — the deeper lever).
- **Deferred (separate run)**: structural disturbance-R² fix — even p117's 0.01
  gave R² −0.626; DOB on a near-perfect clean `g` has tiny innovation → needs the
  `disturbance_loss_rel_weight`/`stop_grad=0` active-shaping path, not a reg tweak.
- **If p120 confirms**: promote `curriculum` + `dob` + step-test-inject
  (`EVERY=10 N=4`) to default-on.

### p120 — VERDICT: not a real result, a CONFIG ACCIDENT (critic cascade)
- **What happened**: the p120 launch carried only **3** env-overrides
  (`dob_enabled`, `dob_reg_coef`, `curriculum_enabled`) but p117/p119 used **~25**.
  It silently **dropped ~22 overrides → 20 knobs reverted to TrainConfig
  defaults**. p120 is therefore *not* a clean step-test test.
- **Critic cascade (the headline)**: `critic_mc_grounding_coef 1.0→0.0`,
  `critic_imag_loss_coef 0.3→1.0`, `p3_critic_warmup_iters 10→0`,
  `rssm_imag_latent_mode T→F`, `rssm_free_bits 0.5→1.0` all reverted. Within ~20
  P3 iters `critic_rew_to_tgt_var` collapsed **0.0187 → 0.001**, `return_scale`
  ran **2.6 → 139** (53×), `critic_pred_target_r` pinned 0.99 = textbook
  bootstrap runaway. Actor thrashed downstream: val mv_viol **5.26**, cv_viol
  **78.9**, cum_raw **−128.7k** (min −285k / max −16k).
- **WM still OK despite the mess**: `wm_overshoot/held=0.0`, `recon_cv=1.0` (levers
  OFF) yet gain came out **MV 0.805 / DV 0.783** (step-test `EVERY=10 N=4`: DV
  0.761→0.783, MV 0.836→0.805, aggregate 0.164→**0.188** ≈ flat — 10/4 traded MV
  for DV, no net gain over 20/2). **Curriculum + step-test are robust.**
  Disturbance r=**+0.713** (best yet), R² −0.900.
- **Training-data question (noise/disturbances)**: **not the cause.** CV output
  SNR **18 dB** (clean, meas-noise σ 0.14), DV 12 dB; the −9 dB obs[2]/obs[11]
  are the **MV being PRBS-dithered** (the WM conditions on the *commanded action*,
  not the noisy MV obs → no gain attenuation), and `g` freezes on **clean**
  Stage-1 data so Stage-2/3 disturbances can't steal gain. The residual ~0.8 gain
  is decomp-localized to **real→post 0.931** (recon, lever `wm_recon_cv_weight`)
  + **1step→openloop 0.89** (compounding, levers `wm_overshoot`/`wm_held`) — both
  of which p120 had **turned off**.

### p121 — FIX: promote the proven recipe to DEFAULTS, env-free restoration
- **Root-cause fix (the user's "update training defaults")**: promoted the full
  p117 winning recipe from fragile env-overrides into **`TrainConfig` defaults**
  so a thin launch can never silently regress them again. 14 knobs in
  `training/train.py` (`critic_mc_grounding_coef 1.0`, `critic_imag_loss_coef 0.3`,
  `critic_replay_anchor_coef 0.0`, `p3_critic_warmup_iters 10`, `rssm_free_bits 0.5`,
  `rssm_imag_latent_mode True`, `bound_training_reward_max 3.0`,
  `wm_recon_cv_weight 4.0`, `bc_track_expert_every 1`, `wm_trunk_stopgrad_in_p2 True`,
  `curriculum_enabled True`, `dob_enabled True`, `wm_overshoot_coef 0.3`,
  `wm_held_rollout_coef 0.5`, `wm_held_rollout_max_starts 8`) + 2 plant-tied
  lengths in `single_run.py` (`wm_overshoot_len = wm_held_rollout_len = horizon`).
  Left alone: `gamma` (auto-tunes to 0.99 at H=55), `disturbance_loss_scale=1.0`
  (harmless under stop-grad), phase fracs (auto-derive). Curriculum smoke green
  (both backbones).
- **Launch**: **env-free** — `python -m workflow.single_run --simulation-dir
  simulation/test_sim --out-dir …`. Resolved cfg verified: mc=1.0 / imag=0.3 /
  anchor=0.0 / warmup=10 / free_bits=0.5 / imag_latent=T / overshoot=0.3 len=55 /
  held=0.5 len=55 / recon_cv=4.0 / curriculum+dob=T / gamma=0.99(auto) /
  bound_max=3.0. Step-test 20/2 (default), dob_reg 0.01 (default).
- **This isolates the critic fix AND turns the WM-bias levers (overshoot / held /
  recon_cv) back ON** (p120 had them off) — that is the "reduce WM bias further"
  the user asked for, on a known-good base.
- **Judge by**: (1) critic `return_scale < 15` (p117=9.7), `rew_to_tgt_var`
  recovers, **no cascade**; (2) actor active+economic (mv_viol ~0.3, cv_viol < 25,
  cum_raw beats −47k); (3) WM gain MV ≥ 0.80 / DV ≥ 0.78, ideally toward 1.0;
  (4) disturbance r > +0.5.
- **If confirmed**: commit + push the defaults promotion; run the
  paper-defaults-audit to log the new baseline. Residual WM bias → p122 (longer
  step-test holds for steady-state dwell, or gain-aware `wm_best`), not stacked
  onto p121.

### p121 — VERDICT: critic-fix worked for MV, DV under-excited, actor still poor
- **MV gain FIXED**: ratio **0.805 → 0.932** (best ever) — the default-restore
  (critic grounding + WM levers back on) did it. Decomp shows compounding is
  essentially solved (1step→openloop **0.981**, post→1step 1.001), so any
  residual is identification, not rollout.
- **DV gain STUCK ~0.75** (0.761/0.783/0.753 across p119/p120/p121 — unchanged by
  anything tried). It is **settled** by the horizon (not a measurement artifact),
  so it is a genuine **gain-identification** failure.
- **Disturbance prediction still lacking**: r **0.557**, R² −0.258, pred_std 1.16
  vs true 1.93 → **under-amplitude ~1.7×** with local sign flips. Same
  under-prediction signature as the DV gain (they're coupled).
- **Critic better but not healthy**: MC grounding engaged (mc_loss = 93% of
  critic loss) so **no p120-style cascade**, but `return_scale` creeps 15→35 and
  `ema_return` collapses in the back half (−337 → −2326 after iter ~428).
- **Actor still poor**: validation `best.pt` is iter **341** (captured *before*
  the collapse) yet still cum_raw −110k, cv_viol 64.8 — never rides the limit.
  Entropy swings −0.10 ↔ −1.0 = the "oscillate ↔ passive" the user sees.

### Root cause of the DV gap — ~30× MV-vs-DV excitation asymmetry
- `collect_prbs_episode` gives the **MV** full-range, stratified, multi-timescale
  PRBS in (nearly) every seed episode, and the WM conditions on the **noise-free
  commanded** MV → MV gain identified unbiasedly (0.93).
- The **DV is never PRBS-swept**: it only gets sparse 10–30 %-span steps in ~20
  step-test episodes (`dv_share` 0.5), and during clean Stage 1 that is the ONLY
  DV motion. Two signal-theory failures follow: **(a) insufficient/non-persistent
  excitation** (DV rarely held to steady state) and **(b) errors-in-variables /
  regression dilution** — the WM's DV regressor is the *measured* (noisy) DV, so a
  low DV SNR biases the learned gain toward zero. A wrong DV gain also leaks
  DV-driven CV into the DOB innovation → the disturbance under-prediction. So
  **fixing DV excitation fixes both** the DV gain and the disturbance head.

### p122 — fixes: DV-PRBS excitation + observer gain + phase rebalance
- **Fix 1 (DV gain, the headline)** — new `collect_dv_prbs_episode`: the DV
  analogue of the MV PRBS. Holds the MV and sweeps **every** measured-DV channel
  with a full-range (`dv_prbs_op_frac=0.6`), multi-timescale, stratified PRBS via
  the persistent-offset disturbance schedule (Δ_k = L_k − L_{k−1}), hidden
  disturbance off. Seeded (`dv_prbs_seed_episodes=16`) **and** re-injected through
  Stage 1 (`DREAMER_DV_PRBS_INJECT_EVERY=20 N=2`, default-on in P1) so the DV gain
  stays supervised to the WM freeze. Removes both excitation deficits: persistent
  large-amplitude excitation (Var(DV) ≫ Var(noise) → dilution → 1) with the MV
  held (∂CV/∂DV identifiable in isolation). Smoke-verified: DV span 7.15 vs
  step-test 1.77, MV std 0.0. No-op fallback when n_dv=0.
- **Fix 2 (disturbance/critic-observer)** — `dob_gain_init −2.2 → −1.8` (Kalman
  K 0.10 → 0.14) so the observer tracks the disturbance amplitude better (was
  under-predicting 1.7×); pairs with Fix 1, which cleans the innovation feeding K.
- **Fix 3 (actor/critic + WM budget)** — rebalanced `derive_phase_budgets`
  P3_ITERS (S/M/L 50/70/90 → **35/45/55**) so P3 ≤ P1. Restores the proven p117
  **0.45/0.25/0.30** split (was 0.37/0.21/0.42): more Stage-1/2 WM-identification
  budget, and P3 ends before the late actor-critic drift regime that the
  over-long p121 P3 exposed.
- **Held at proven (no confound)**: critic grounding mc=1.0 / imag=0.3, warmup=10,
  all WM levers, curriculum+DOB — all from defaults (env-free launch). Verified
  resolved cfg + `[seed] dv-prbs=16` + phase split 0.45/0.25/0.30 in p122.
- **Judge by**: DV ratio **0.75 → >0.85** (MV holds 0.93); disturbance r **>0.6**
  and pred_std/true_std **>0.75**; **no late `ema_return` collapse** (return_scale
  stays <15); actor rides the limit (cv_viol <25, cum_raw beats −47k). Attribution
  is clean — DV gain, disturbance, critic, actor each have separate metrics.

### p122 — VERDICT: small WM progress, disturbance miscalibrated, actor passive
- **WM gain**: MV 0.932 → **0.947**, DV 0.753 → **0.792**. DV-PRBS helped but the
  improvement is **capped** (see root cause B).
- **Disturbance**: r 0.557 → **0.654** (DV-PRBS cleaned the observer innovation —
  the DV-gain↔disturbance coupling is **confirmed**), BUT R² −0.258 → **−1.775**
  and pred_std 1.16 → **2.27** vs true 1.93 = now **over-predicts** (ratio 1.18).
- **Critic**: the phase rebalance **worked** — P3 is 249 iters (vs p121's 391); the
  mid-P3 cascade peaked at return_scale 55 then **recovered to 13** (vs p121's
  runaway to 35+ and ema −2326). But `rew2tgt` stays **<0.015** throughout
  (bootstrap dominance) despite MC grounding at 93% of the critic loss.
- **Actor**: cum_raw −110k → **−138k**, cv_viol 64.8 → **94.9**, mv_viol 6.5 →
  **0.0** (fully passive; vision: "MV flat, CV violates high, passive not
  active-economic"). Entropy collapsed to −0.10 (σ floor) early in P3.

**Three distinct root causes:**
- **(A) #2 disturbance over-predict** — `dob_gain_init −1.8` **overshot** (Kalman
  K 0.142 too reactive at validation: −2.2 under-predicted 0.60×, −1.8 over 1.18×).
- **(B) #1 DV gain capped** — the P1→P2 `wm_best` **warm-restore** loaded the
  iter-30 correlation peak, and the fidelity probe (`_probe_wm_fidelity` = Pearson
  r + held-convergence) is **scale-invariant / gain-blind**, so it **discarded the
  DV-PRBS re-injections** at iter 40/60. Only the 16 **seed** dv-prbs episodes
  survived into the frozen WM.
- **(C) #4 actor passive** — downstream of the #3 cascade (NOT `d_t`: in-training
  `dob_d` 0.275 is *lower* than p121's 0.378, and anti-correlates with
  return_scale). The over-amplified **validation-time** `d_t` (pred_std 2.27)
  corrupts the actor's `feat` at validation.

### p123 — fixes: dob_gain revert + DV front-loading (clean per-metric attribution)
- **(1) #2 disturbance** — `dob_gain_init −1.8 → −2.0` (K 0.142 → 0.119; amplitude
  ratio 1.18 → ~0.9). Also reverts the validation-time `d_t` toward p117's
  active-actor regime → helps #4.
- **(2) #1 DV gain** — `dv_prbs_seed_episodes 16 → 24`: more DV excitation in the
  **early** checkpoint the warm-restore keeps, bypassing the gain-blind probe.
- **(3) #1 DV gain** — DV-PRBS re-inject `every 20 → 10`: fires at iter 10/20/30
  (all inside the ≤30 kept window) instead of 20/40/60 (40/60 rolled back).
- **Each fix targets a distinct metric via a distinct mechanism** (DV gain ← seed
  + inject cadence; disturbance amplitude ← dob_gain; actor ← the dob_gain regime
  revert) so attribution stays clean. Held at proven: critic grounding mc=1.0 /
  imag=0.3 / warmup=10, phase 0.45/0.25/0.30, all WM levers. Curriculum smoke green.
- **Deliberately NOT changed**: entropy floor / critic warmup / cascade early-stop
  — speculative and risk backfire (more warmup on a passive BC-warmed actor can
  *reinforce* passivity) or confound the WM attribution.
- **Judge by**: DV ratio 0.79 → **>0.85** (MV holds 0.95); disturbance r **>0.6**
  with pred/true **0.85–1.1** (not over); actor **less passive** (mv_viol >0.2,
  cv_viol <40, cum_raw beats −110k); critic no worse.
- **If the actor stays passive** after the dob_gain revert: the next run needs the
  **structural #1 lever** (make `_probe_wm_fidelity` gain-aware via predicted-vs-
  real slope, OR disable the P1→P2 warm-restore in curriculum mode — the
  overshoot+held losses already prevent drift), which *also* helps #3/#4 (an
  accurate WM → less erratic imagined returns → smaller cascade), plus a dedicated
  critic intervention (return_scale clamp, or tighten the cascade early-stop
  growth 100× → 30×).

### p123 — VERDICT: fixes applied but didn't work; gain-blind wm_best is the root cause
- **Fixes confirmed applied**: `dob_gain −2.0`, `dv_prbs_seed 24`, DV-PRBS inject
  every 10 (fired iter 10/20/30/40/50/60). But results barely moved vs p122:
  - **WM gain**: MV 0.947 → **0.898** (worse), DV 0.792 → **0.772** (~flat).
  - **Disturbance**: amplitude **fixed** by `dob_gain` (pred/true 1.18 → **0.96**,
    the target) BUT correlation **collapsed** r 0.654 → **0.092**, R² −3.60.
  - **Actor**: cum_raw −150k, cv_viol 101, `return_scale` stuck **~27** all P3
    (p117-healthy 9.7), rew2tgt <0.001, entropy pinned. Vision: partially
    economic (tracks limit changes) but violates + mild oscillation.
- **Decisive RCA (P1 probe trace)**: the `wm_best` fidelity score is **dominated
  by correlation noise**. The per-offset Pearson r bounces ±0.15 with no trend;
  iter 30 won "best" only on a **noise spike** crossing the r-floor (best_h 27 at
  iter 30 vs **0** at iter 40–70). The **stable, gain-relevant** metrics improve
  monotonically to P1 end — recon **0.102 → 0.087**, convergence **0.25 → 1.00**.
  So the P1→P2 warm-restore froze the **under-trained iter-30 `g`** and discarded
  the late-P1 DV-PRBS gain data (injects at 40/50/60).
- **One cause, three symptoms**: (1) WM gain capped/randomized; (2) the DOB
  observer built on the frozen-random `g` → disturbance r swings **0.557 / 0.654
  / 0.092** across p121/122/123 (near-identical configs) = observer uncontrolled;
  (3) noisy observer → noisy imagined returns → `return_scale` runaway (27) →
  shrunk actor advantage (`adv = adv_raw / return_scale`).

### p124 — fixes: disable curriculum warm-restore (root) + adaptive return-scale cap
- **(A, root cause)** In **curriculum mode**, disable the P1→P2 `wm_best`
  warm-restore — freeze the **full-P1-trained `g`** (all clean + DV-PRBS gain
  data; lower recon = better gain; conv = 1.00) instead of rolling back to the
  noisy correlation-peak checkpoint. Justified because the "post-peak drift" is
  correlation **noise** (recon + convergence prove iter 70 > iter 30), and the
  anti-drift `overshoot`(0.3) + `held_rollout`(0.5) losses protect the gain.
  Gated on `curriculum_enabled` and honours an explicit
  `DREAMER_WM_BEST_RESTORE_AT_P2`.
- **(B, safety net)** `return_scale_abs_cap` 500 → **sim-adaptive
  `max(20, 0.12·B·H)`** (test_sim = 20). Sits above p117's healthy max (17.5),
  below the 27–55 runaway → never distorts a healthy run but arrests the
  return-norm runaway that shrinks the actor's economic advantage. Sim-adaptive
  via the plant's own `B` and `horizon`.
- **Clean attribution**: A = WM gain + disturbance-r consistency; B =
  `return_scale` + actor economics. Both in defaults (env-free launch); verified
  `return_scale_abs_cap=20.0` + `warm-restore DISABLED` banners. Curriculum smoke
  green (both backbones).
- **Judge by**: DV ratio 0.77 → **>0.85** with MV ≥ 0.93 (gain no longer capped);
  disturbance r **stable >0.5** (no more 0.09 collapse = observer controlled);
  `return_scale` settles **<20**; actor active + economic (cv_viol <40, cum_raw
  beats −110k, rides the limit).
- **If A works**: the gain-blind-checkpoint saga is closed; an optional general
  follow-up is to make the `wm_best` score gain-aware (CV std-ratio from the
  k-step rollout `pred_obs`/`real_obs`) for non-curriculum runs.

### p124 — VERDICT: warm-restore-disable was a regression; found the WM-gain root
- **The p123 warm-restore-disable HURT**: MV gain 0.947 → **0.849** (worse). Warm-restore
  ON (p121/122/123) averaged **0.926**; OFF (p124) gave 0.849. The p123 hypothesis
  ("full-P1 g is better") was wrong — P1 recon is **non-monotonic** (bottoms iter 40
  = 0.085, rises to iter 70 = 0.108), so freezing end-of-P1 froze a *worse* WM.
- **`return_scale_abs_cap=20` WORKED (clean keep)**: return_scale pinned at 20.00 all
  P3 (vs p123's 27 runaway). The return-norm runaway is arrested — but the **actor is
  still passive**, proving passivity is NOT the return-norm runaway.
- **Actor-passivity root = reward asymmetry 659:1**: raw_min −488 vs raw_max +0.74,
  positive_fraction only **8.9%**. MC grounding is 90% engaged yet `critic_target_v_r`
  = 0.97 — the economic upside (+0.74 for riding the limit) is a sliver below the noise
  floor against the −488 violation cliff. Deferred to a dedicated run (objective design).
- **WM-gain structural root (vision-confirmed)**: the decomp splits the bias into
  `real→post` **0.855** (1-step autoencoder, already CV-weighted since 2026-06-09) +
  `1step→openloop` **0.906** (multi-step open-loop = the gain over the horizon). The
  `_wm_latent_overshoot_loss` is THE open-loop gain supervisor but used **uniform MSE**
  — so the small-variance CV step-response is drowned by the high-variance PRBS'd MV/DV
  channels. Vision: the WM "rises fast then plateaus early **below**" the real gain =
  premature saturation = undersupervised asymptote. (The held-rollout loss is
  gain-neutral by construction — not the lever.)

### p125 — fix: CV-weight the multi-step overshoot loss + revert warm-restore
- **(structural)** CV-weight `_wm_latent_overshoot_loss`: replace
  `(pred−tgt).pow(2).mean()` with `_weighted_recon_mse(pred, tgt, cfg)` so the
  multi-step open-loop **CV** response (the gain over the settling horizon) is directly
  supervised instead of drowned. Reuses `cv_obs_indices` + `wm_recon_cv_weight=4.0`
  (sim-agnostic, within-loss emphasis, renorm mean-1 preserves magnitude; identity at
  weight 1). This is the same CV-weighting the 1-step recon got in 2026-06-09, finally
  applied to the multi-step term that actually sets the open-loop gain.
- **(revert)** Removed the p123 curriculum warm-restore-disable — back to the P39
  default (ON), since p124 proved OFF regressed the gain.
- **Kept**: `return_scale_abs_cap=20` (now via shared auto-tune), all proven defaults.
- **Judge by**: open-loop gain ratio 0.775 → **>0.90** (WM reaches steady state, no
  premature plateau), MV ≥ 0.92 / DV ≥ 0.85, disturbance r **stable >0.5** (observer on
  a converged g). Single coherent WM-gain change for clean attribution.
- **If WM gain fixed but actor still passive**: next run attacks the reward asymmetry
  (659:1) directly — that's the binding actor constraint now, separate from the WM.

### p125 — VERDICT: CV-weighted overshoot WORKED (best WM); critic healthy; actor economically right but imprecise
- **WM (best in series)**: MV gain 0.849 → **0.950**, DV 0.761 → **0.859**, `real→post`
  0.855 → **0.959** (autoencoder ~fixed), `1step→openloop` **0.926**, disturbance r
  **0.738** (best ever) with pred/true **1.03** (well-calibrated). The CV-weighted
  multi-step overshoot loss fixed the open-loop gain undersupervision — **keep**.
- **Critic — healthy** (the user's question): fits its target (`critic_pred_target_r`
  0.983), MC grounding 91% engaged, `return_scale` **cleanly capped at 20.00** all P3
  (the `abs_cap` fix works — no cascade, no runaway). `rew_to_tgt` 0.0009 is **expected**
  for a long correlated horizon (H=55, persistent violations → return variance ~20× iid),
  not pathological; `adv_std` 0.54 shows the critic does distinguish states. **The critic
  is not the bottleneck.**
- **Actor (vision overturned the "passive" read)**: it **is** economically optimizing —
  low reflux, riding the **upper** temperature limit, MV actively moving (`mv_viol 0.000`
  = reflux stays inside its own actuator bounds, not passivity). The real problem is
  **imprecise constraint handling**: cv_viol 76, it overshoots the limit it rides, with
  violations mostly **disturbance / band-step driven**.
- **Root of imprecision**: the operating-region reward is ~30× below the band-keeping
  shaping + imagined-reward noise (0.19). The bounded-reward slope `B/ref = 3/100 = 0.03`
  compresses economics (+0.73) and mild violations (to ~−10 raw) into `[−0.3, +0.02]`, so
  the actor gets a usable gradient only from catastrophic violations (raw < −100 → −3).
  The 770:1 reward asymmetry is fundamental — a symmetric scale keeps economics tiny, and
  amplifying it asymmetrically risks flipping the optimum toward violating.

### p126 — fix: flat-top safety-margin shaping (actor precision)
- The band-keeping shaping potential (`_shaping_potential`, no-target/range case) was a
  **tent peaked at the band centre** — it center-biases the actor (diluting economic
  limit-riding) and spreads the safety gradient thinly. Replace with a **flat-top**:
  Φ = 1 across the interior (no center-pull, economics free) ramping 0→1 only within a
  margin band of width `shaping_safe_margin_frac · half_band` (default **0.25**) at each
  edge — a concentrated, steeper pull-back exactly in the near-constraint zone where the
  disturbance-driven overshoot happens. Still **potential-based (policy-invariant** —
  cannot change the economic optimum) and **sim-adaptive** (margin = fraction of the
  plant's own half-band). The target-tracking path is unchanged. Verified: zero interior
  center-pull, steeper near-limit gradient; curriculum smoke green.
- **WM + critic kept as-is** (WM is good at 0.95/0.86; critic healthy) for clean
  attribution of the shaping change.
- **Judge by**: cv_viol 76 → materially lower (**< 40**) **while** the actor stays
  economic (still rides the upper limit, cum_raw no worse than −128k) and MV stays smooth.
- **If insufficient**: the deeper lever is objective re-design — asymmetric reward scaling
  or a training-time constraint back-off — to make the economic signal visible without
  flipping the optimum.

### p126 — VERDICT: shaping didn't help; the smoking gun is run-to-run VARIANCE
- **Safety-margin shaping regressed (within noise)**: cv_viol 76 → **99**, cum_raw
  −128k → −149k, WM MV gain 0.950 → **0.861**. But this is **inside the noise band**.
- **The decisive finding — we've been measuring NOISE**: per-seed validation cum_raw
  ranges **−5,646 to −440,575** across this one run's 12 episodes (80× spread). And the
  cross-run metrics bounce with **no trend** over 6 runs: MV gain
  0.932/0.947/0.898/0.849/0.950/0.861 (±0.05), DV 0.753/0.792/0.772/0.761/0.859/0.775
  (~0.78), cv_viol 64.8/94.9/101/62.8/76.2/99.0 (±20). Single-knob A/Bs **cannot be
  attributed** — the variance dwarfs the effect.
- **Critic — structurally fine** (the user's question): fits target (`pred_target_r`
  0.983), MC grounding 92%, `return_scale` cleanly capped at 20.00. Entropy pins at the
  σ-floor (−0.101) from the first P3 iter. The critic faithfully fits a reward whose
  economic component is genuinely tiny — it is not the bottleneck.
- **Root cause of the variance + "passive actor" (vision-confirmed)**: the actor
  **under-reacts to measured disturbances** (MV moves right direction but too little/slow)
  → CV overshoots the limit by **6–7 °C** and sustains the −488 cliff → catastrophic
  episodes that dominate the mean. This is **downstream of the DV-gain under-read (0.78)**:
  the WM tells the actor a disturbance is only 78% as strong as it is. And the DV gain
  bounces run-to-run because the `wm_best` pick is **noise-driven** (gain-blind score).

### p127 — fix: gain-aware `wm_best` selection (the structural #1-priority lever)
- The `wm_best` fidelity score was correlation + convergence + recon — none directly
  measure the **CV open-loop gain**, so the pick rode noise and the frozen WM gain
  bounced 0.85–0.95. Add a **gain-fidelity term**: the CV-channel std-ratio of the k-step
  open-loop rollout (pred vs real, under real actions + DV teacher-forced). `min(ratio,1)`
  credits a faithful/over-reading gain fully and penalises only under-prediction (the
  actual bias). Recon-gated so an untrained, high-variance early checkpoint can't win on
  spurious CV variance. Weight `DREAMER_WM_FIDELITY_GAIN_WEIGHT=3.0` (default-on),
  gate `…GAIN_GATE_RECON=0.15`.
- **Why this is the root-cause fix**: it directly optimizes the control-relevant property
  (CV gain), so it should (a) **raise** the frozen WM gain — especially the DV (the
  under-reaction source) — and (b) **reduce run-to-run variance** by picking consistently
  high-gain checkpoints instead of noise spikes, which makes future fixes attributable.
  Serves the standing #1 priority (unbiased WM) directly.
- **Kept p126 as the baseline** (flat-top shaping) so the only new variable is the
  gain-aware selection. Unit-tested (0.78→0.78, 0.97→0.97, over-read capped at 1.0);
  curriculum smoke green.
- **Judge by**: MV gain → **>0.93** AND DV → **>0.85** AND **lower run-to-run spread**
  (the variance drop is itself the signal); then the actor's catastrophic-episode rate
  should fall (cum_raw spread tightens) as the DV under-reaction is corrected.
- **If the gain rises but the actor is still imprecise**: the next lever is the economic
  signal strength (the 770:1 asymmetry) — but fix the WM gain + variance first so it's
  measurable.

### p127 — VERDICT: gain-aware selection FAILED, 7-run plateau (p121–p127)
- Gain-aware changed the pick (iter110 vs p126's iter60) but the transfer gain stayed
  biased: **MV 0.882, DV 0.760**, disturbance r 0.347 (now *over*-predicts, pred/true
  1.61), cum_raw −156k, cv_viol 98, mv_viol 0.53 (MV oscillating), per-seed spread
  −15k…−459k. The `gain_fid` proxy (P1 random-action CV std-ratio) does **not** match the
  post-train isolated-step transfer gain → we optimized the **wrong proxy**.
- **4 evidence-backed root causes**: RC-W1 (sysID, #1) DV→CV gain structurally
  under-identified — FEED moves ~0.29 std (slow OU) + only ~1–5 sparse step events/episode
  on-policy vs MV's ±0.6 continuous PRBS → DV gain stuck 0.76 across all 7 runs. RC-W2
  (signal theory) WM-gain measurement is noisy (same-ckpt probe bounces 0.2–0.3). RC-A1
  (control/ML) economics-blind reward (bounded-remap slope 0.03 crushes econ +0.73→+0.022,
  ~9× below noise). RC-A2 (control) actor controls a biased plant (under-reads DV 24% →
  CV overshoot 6–7 °C). RC-M1 (meta) per-seed 30–80× spread on 3×3 val exceeds the effect
  size → runs un-attributable.

### p128 — fixes: R0 (val CI) + R1a (on-policy DV-PRBS) + R2a (economic shaping)
- **R0 (measurement unblock, RC-M1)**: `--val-episodes 3→4`, `--val-seeds 3→8` (32 vs 9
  rollouts) + a **mean ± 95 % CI** print and `cum_raw_reward_ci95_halfwidth` /
  `_n_rollouts` stored in `validation_summary.json` (sample sd, 1.96·s/√n). Makes the
  run-to-run variance — the p126 smoking gun — **measurable**, so the bundled p128 stays
  attributable.
- **R1a (WM root, RC-W1)**: drive the **measured DV** with the **same full-range,
  multi-timescale, stratified PRBS** the seed episodes use **throughout the clean Stage-1
  on-policy collection** (not just the evicted seed batch). Extracted the schedule core
  into a shared `_build_dv_prbs_schedule(env, cfg)`; `reset()` swaps in the DV-PRBS when
  the curriculum sets `env._dv_prbs_in_reset` (P1 + measured DV + `dv_prbs_onpolicy_in_p1`,
  default ON; OFF in P2/P3). Hidden/unmeasured disturbance stays OFF in P1, so the
  gain↔unmeasured-disturbance separation is preserved. Smoke: P1-reset DV events 4→11.
- **R2a (actor, RC-A1; greenlit)**: a **state-based economic potential** Φ_econ ∈ [0,1]
  folded into the shaping potential — Φ = Φ_safe + `shaping_econ_coef`·Φ_econ
  (default 0.5). Φ_econ is a per-channel linear ramp across each economically-weighted
  MV/CV's engineering band, oriented by the sign of its economic weight (the
  penalty-lowering direction), |w|-weighted, clamped at the band (zero gradient outside
  the limits = feasibility-aligned). Potential-based ⇒ **policy-invariant (Ng 1999) for
  any potential**, so — unlike the held R1b gain-loss — it is **safe on nonlinear plants**
  (only densifies the near-invisible economic gradient, never moves the optimum). Smoke:
  Φ_econ 0.95 at low MV vs 0.05 at high MV (correct for test_sim's +5.0 reflux-min weight).
- **Verification**: both new env-overrides (`DREAMER_DV_PRBS_ONPOLICY_IN_P1`,
  `DREAMER_SHAPING_ECON_COEF`) wired; p128 smoke green; curriculum freeze-partition smoke
  green on **both** backbones. **Not yet launched** (awaiting user go).
- **Judge by**: DV transfer gain **>0.85** (R1a) AND a **tighter validation CI / smaller
  per-seed spread** (R0+R1a) AND actor **cv_viol down** (R2a). The R0 CIs make the
  3-change bundle attributable.

### p128 — VERDICT: MV gain BEST-EVER (0.96) but DV still 0.76; actor poor; DV gain is THE root
- **WM transfer matrix**: **MV ratio 0.961** (real −0.28, WM −0.269) — best of the whole series
  (p117 0.78 → p125 0.95 → p128 0.96, near-unbiased). **DV ratio 0.764** (real +0.18, WM +0.1375)
  — UNCHANGED from the 8-run plateau (~0.76). R1a (on-policy DV-PRBS) did NOT move the DV gain.
- **R1a FALSIFIED + counter-productive**: direct regen showed the on-policy DV-PRBS flag-ON gives
  FEED(obs) std **0.73** (range ±1.65) vs the legacy sparse schedule flag-OFF std **1.20** (±3.5) —
  R1a *reduced* on-policy DV excitation, and even the larger legacy excitation never moved 0.76.
  ⇒ **excitation amplitude is NOT the binding constraint** (revert candidate).
- **DV bias is STRUCTURAL — every excitation/data fix already tried & failed**: seed DV-PRBS (24 eps),
  periodic MV-held DV-PRBS re-inject (every 10 iters, 2026-06-14), step-test re-inject (every 20),
  R1a on-policy — all keep MV-held DV-isolated data fresh in the buffer, gain still 0.76. EIV ruled
  out (obs[3] FEED SNR **11.25 dB** → attenuation ≈0.93, explains only ~7% of the 24% gap). Eviction
  ruled out (re-inject exists). **Root cause = DV is a SUBDOMINANT regressor**: its CV contribution
  (gain 0.18) is drowned by the MV-driven CV variance (gain 0.28, full-range action) in the
  autoencoder + CV-weighted recon/overshoot loss, so the categorical latent under-represents the
  small DV-driven CV component → open-loop DV gain plateaus ~0.76 (× EIV 0.93). The overshoot loss
  DOES teacher-force the real DV, but its CV-weighted MSE is still dominated by MV-driven CV.
- **Actor poor (vision-confirmed)**: REFLUX **passive** (~56-58%, barely moves), under-reacts to FEED
  steps → CV rides **1–3.5 °C above the 85.5 high limit for ~600 steps**. cv_viol mean **95.7**
  (healthy ~11), median per-ep 57.5, catastrophic tail (max 603, cum_raw −651k); cum_raw mean −144k
  ± 41.9k CI95 (n=36). Direct signature of the biased DV gain: frozen WM under-reads FEED 24% →
  actor under-compensates. `wm_gain_rel_err 0.039` is MV-only ⇒ all_pass=1 is a FALSE POSITIVE.
- **Critic structurally fine**: pred_target_r 0.99, MC loss ~2.5 engaged, return_scale capped 20,
  σ 0.219→0.13 (not floored). `rew_to_tgt_var` 0.001 = the known MC-measurement caveat. NOT the
  bottleneck. (adv_std decays 0.99→0.34.)
- **Disturbance head**: amplitude OK (pred_std 1.88 ≈ true_std 1.93, lag −2) but R² **−1.56** (r 0.56)
  — a slow mid-episode DRIFT/bias (what the user saw), consistent with under-read measured-FEED
  leaking into `d_t`. Largely downstream of the DV gain.
- **R2a (econ shaping) premature**: Φ_econ pushes CV toward the high limit (min-reflux econ) under a
  disturbance-blind WM → makes the high-side violations WORSE. Revert/margin-gate until WM unbiased.
- **UNIFYING ROOT CAUSE = the biased DV→CV gain (0.76)**: it is the common root of (1) the DV
  transfer bias, (2) the disturbance-head drift (FEED leaks into d_t), and (3) the actor
  under-reaction → CV high-side violation. Fix #1 (DV gain) addresses all three.
- **Next (proposed, awaiting approval)**: un-bias the DV gain by making the MV-held DV-isolated
  episodes a FIRST-CLASS WM-loss target (tag + oversample in the Stage-1 minibatch, mirroring the
  `expert` per-step flag) so the subdominant DV-driven CV stops being drowned — sim-agnostic, no
  linearity assumption (supervises the real CV response, unlike the held R1b scalar-gain loss).
  Revert R1a (counter-productive) + revert/margin-gate R2a. Re-judge DV gain, disturbance R², actor.

### p129 — fix: D1 DV-isolated minibatch oversampling (the DV-gain root); revert R1a + R2a
- **D1 (the DV-gain root cause)**: tag MV-held, DV-swept episodes (`collect_dv_prbs_episode`) as
  `dv_isolated` in `TrajectoryBuffer` (mirrors the per-episode `expert` flag pattern) and OVERSAMPLE
  them to a guaranteed floor fraction (`wm_dv_isolated_minibatch_frac`, default **0.3**) of the
  Stage-1 (P1/P2) WM minibatch. In those episodes ALL CV variance is DV-driven (and the DV is swept
  at large amplitude ⇒ EIV≈1), so the CV-weighted recon/overshoot loss supervises ∂CV/∂DV
  **undiluted** — directly attacking the p128 root cause (the DV is a subdominant regressor whose
  CV contribution, gain 0.18, was drowned by the MV-driven CV variance, gain 0.28). Gated to P1/P2
  by the caller so P3 imagination starts stay representative. Sim-agnostic (a fraction),
  backbone-agnostic (sampling is upstream of the WM), **no linearity assumption** (supervises the
  real CV response → safe on nonlinear ONNX sims — the property the held R1b scalar-gain loss lacks).
  Env: `DREAMER_WM_DV_ISOLATED_FRAC`. Realistic target: DV ratio 0.76 → ~0.90 (EIV 0.93 floor).
- **Cleanliness cull (per the standing mandate)**: **reverted R1a** (on-policy DV-PRBS in reset —
  falsified by p128: it *reduced* on-policy FEED excitation std 1.2→0.73 and didn't move the gain;
  removed the `_dv_prbs_in_reset` flag + reset hook + curriculum wiring + `dv_prbs_onpolicy_in_p1`
  knob + `DREAMER_DV_PRBS_ONPOLICY_IN_P1`; KEPT the clean `_build_dv_prbs_schedule` refactor that
  `collect_dv_prbs_episode` uses) and **reverted R2a** (econ shaping — premature: Φ_econ pushed CV
  toward the high limit under a disturbance-blind WM, worsening high-side violations; removed
  `_economic_potential`, the Φ_econ fold, the `shaping_econ_coef` knob + `DREAMER_SHAPING_ECON_COEF`).
  Net **−169/+103 lines**. So D1 is the only new variable vs p128 (R0's val CIs stay for attribution).
- **Verification**: p129 D1 smoke green (tag stored, oversample floor honoured — 54≥38 @frac0.3,
  empty-pool no-op, override works, R1a/R2a fully gone, helper kept); curriculum freeze-partition
  smoke green on **both** backbones. Launched env-free (tmux `mbrl_p129`), curriculum Stage-1 active,
  no env-overrides, clean startup.
- **Judge by**: **DV ss-gain ratio >0.85** (read `wm_dv_transfer_matrix.json`) — the primary signal;
  then disturbance-head **R² >0** (FEED stops leaking into `d_t`) and actor **cv_viol down** + tighter
  CI (all DOWNSTREAM of the DV gain). MV gain should HOLD ~0.96.
- **Deferred (Step 4, post-p129)**: actor reward redesign — re-introduce economic shaping ONLY after
  the WM is unbiased, ideally **CV-margin-gated** (reward econ only when the CV has safe headroom) so
  it doesn't chase economics into the constraint. On the todo list + memory.

### p129 — VERDICT: D1 FAILED (DV 0.765 ≈ p128 0.764); DV bias is an AUTOENCODER bottleneck (new decomp)
- **D1 did nothing**: DV ss-gain ratio **0.765** vs p128 0.764 — a 30% DV-isolated minibatch oversample
  (verified to fire: `run_plan.wm_dv_isolated_minibatch_frac=0.3`) moved the gain by **0.001**. MV
  slipped 0.961→**0.905**. This near-exact repeat across a big data-mix change is the decisive clue:
  the DV bias is **NOT data-limited**.
- **NEW DIAGNOSTIC — DV posterior-prior decomposition** (added this run, `compute_dv_posterior_prior_decomp`):
  drives a real DV step (via `sim.set_disturbance_offset`) with the MV held, teacher-forces the WM
  posterior+prior. Result: **real→posterior ×0.72–0.77, posterior→1-step ×1.00**. The DV gain dies
  **entirely in the AUTOENCODER** (encoder→categorical latent→decoder); the prior transition is
  *perfect*. (The MV decomp also flipped to "autoencoder" this run: real→post 0.854.) **This is why
  9 runs of data/excitation fixes failed** — seed DV-PRBS, two re-injectors, R1a, D1 all feed more
  data to a loss whose bottleneck is the representational capacity of the autoencoder. You cannot fix
  a categorical-bottleneck quantization limit with more data. Architecture fact: the decoder input is
  `[deter h, stoch z]` only — the measured DV feeds the *transition* but the decoder + reward/value
  heads never see it directly, so the small DV→CV gain must survive the lossy encode→decode.
- **ACTOR — structural root: the policy gradient is ≈ 0**. `actor_loss ≈ 0` for all of P3
  (0.005/−0.000/0.001/0.003…), σ **pinned at its 0.219 max** (`policy_log_std_max` ceiling, never
  tightened), `mv_viol 0.003` (MV barely moves), `pmpo_pos_frac ≈ 0.5` (symmetric), `cv_viol 104`,
  cum_raw −152k. The actor is passive because it gets **no gradient**, not by choice. Mechanism: the
  imagined-return variance is dominated by the **uncontrollable hidden-disturbance**, so the advantage
  is ~noise around 0 (`reinforce_actor_loss = −E[adv·logπ]` → ~0); the *controllable* economic+safety
  signal is below that noise floor (economics-blind reward, 770:1 asymmetry, bounded slope 0.03 → econ
  ~9× below noise) AND attenuated 24% by the biased WM DV gain. Critic is structurally fine
  (pred_target_r 0.99, MC engaged). So the actor is gated by (a) the reward shape and (b) the WM DV bias.
- **Disturbance head**: r 0.78 (up from p128 0.56), pred_std 1.61 vs true 1.93, R² still −1.59 (slow
  drift) — largely downstream of the DV gain.
- **Next (proposed, awaiting approval)**: (1) **WM DV-gain representation fix** — give the decoder +
  reward/value heads **direct access to the exogenous DV** (feedforward: `feat'/decode_in = [h, z, dv]`)
  so the measured-DV→CV gain skips the categorical bottleneck (both backbones; the confirmed root after
  9 data-fix failures). (2) **Actor**: margin-gated economic shaping + a disturbance-aware advantage
  baseline to lift the controllable signal above the disturbance noise — AFTER the WM DV fix. (3) Added
  the DV decomp as a permanent validation diagnostic (`wm_dv_posterior_prior_decomp.json`); propose an
  actor controllable-vs-uncontrollable return-variance diagnostic next.

### p130 — fix: DV→decoder+heads feedforward (WM gain root) + margin-gated econ shaping + actor diagnostics
- **Fix 1 (WM DV gain, the confirmed p129 root)**: the measured DV is now appended to the head-facing
  `feat` AND fed directly into the WM **decoder**, so the CV reconstruction is `g(h, z, dv)` — a DIRECT
  ∂CV/∂dv path that **skips the categorical bottleneck** where the DV→CV gain was dying (p129 decomp:
  real→post ×0.77, post→1step ×1.00). Feat layout `[h, z, dv, d_tail]`; decoder reads `[h, z, dv]`
  (contiguous front slice), DOB d-tail sliced off + re-added by `apply_dob` (factorisation preserved).
  Implemented on **BOTH backbones** (RSSM + TSSM, symmetric mirrors) + threaded through the state
  (img_step/obs_step carry `dv`), rollout_observed, decode, feat_dim, and the held-DV imagination (the
  value/reward/policy heads now SEE the disturbance → also the disturbance-aware baseline for the actor).
  Sim-adaptive: no-op when the plant has no measured DV (`dv_dim=0`). Knob `dv_feedforward` (default
  True) + `DREAMER_DV_FEEDFORWARD`; threaded TrainConfig→DreamerV4Config→both dyn configs + the
  bo_runner ONNX-export build (also fixed a latent omission: that build was missing `dv_dim`/`dv_indices`).
- **Fix 2a (actor reward, margin-gated econ shaping)**: re-added the economic potential Φ_econ but now
  **CV-margin-gated** — `Φ = Φ_safe + coef·gate·Φ_econ`, gate→0 at the constraint and →1 with safe
  headroom (`shaping_econ_margin_frac`, default 0.5) — the safe successor to the reverted R2a (which,
  ungated, pushed the CV into the high limit). Still potential-based ⇒ policy-invariant (Ng 1999).
  Knobs `shaping_econ_coef` (default 0.5) + `shaping_econ_margin_frac` + env overrides.
- **Fix 2b (disturbance-aware advantage baseline)**: subtract the per-horizon **batch-mean advantage**
  (a pure, unbiased control variate) so the uncontrollable common-mode return offset driven by the
  disturbance level is removed, leaving the controllable action-relative advantage → higher
  policy-gradient SNR. Pairs with Fix 1 (value baseline now conditions on the DV). Knob
  `actor_disturbance_baseline` (default True) + `DREAMER_ACTOR_DISTURBANCE_BASELINE`.
- **Diagnostic 2 (actor controllable-vs-uncontrollable)**: two new `train_log.jsonl` fields in the
  imagination diag — `imag_adv_action_corr` (|corr(advantage, action)|: ~0 ⇒ the advantage doesn't
  credit the action ⇒ no policy gradient on μ ⇒ passive actor — the p129 signature) and
  `imag_reward_dv_corr` (|corr(per-rollout reward, held DV)|: high ⇒ imagined reward is
  disturbance-dominated, the controllable signal buried).
- **Verification**: model forward both backbones (feat_dim 130, recon correct, decoder gets gradient);
  end-to-end WM loss + imagination both backbones; Fix-2a gate (econ add 0.025 near-limit vs 0.315
  mid-band); DOB vectorized==per-step equivalence (Δ=1.5e-7) + DV-input teacher-forcing intact;
  curriculum freeze-partition green both backbones; all overrides apply; full import check. `dv_dim=0`
  is a verified no-op.
- **Judge by**: **DV ss-gain ratio >0.85** (the root fix — `wm_dv_transfer_matrix.json`) AND the DV
  decomp's `real→posterior` jumping toward 1.0 (the autoencoder now represents ∂CV/∂dv); then
  `imag_adv_action_corr` rising off ~0, σ decaying off its max, `mv_viol`/economics improving, and
  `cv_viol` down — with the val CIs for attribution.

### p130 — VERDICT: MIXED — dv_feedforward lifted the DV autoencoder but D1 starved the MV + the active actor over-drove the biased WM
- **WM bias (the headline)**: MV ss-gain ratio **0.813** (DOWN from p129 0.905, p128 0.961 — MV bias
  INCREASED), DV **0.769** (open-loop ~unchanged). But the DECOMP is the real signal: MV real→post
  0.869, **DV real→post 0.859** (up from p129's 0.717–0.767) ⇒ **dv_feedforward lifted the DV
  autoencoder to MATCH the MV** (~0.86). Both channels now share a **~0.86 autoencoder ceiling**
  (categorical-latent contraction). Vision-confirmed: the DV transfer-matrix **band TIGHTENED**
  (variance DOWN — the user's read was right); the MV band stayed wide.
- **MV DECLINE root = D1 still on** (`wm_dv_isolated_minibatch_frac=0.3`): D1 oversamples MV-HELD
  DV-isolated episodes ⇒ STARVES the MV gain (MV autoencoder 0.945 p128 → 0.854 p129 [D1 era] →
  0.869 p130). D1 is **falsified** (p129 moved the gain 0.001), **superseded** by dv_feedforward, and
  **harmful** to the MV ⇒ remove.
- **Neural Kalman / disturbance (`d_t`) WORSE**: r 0.78→0.54, R² −1.6→−2.4, pred_std 1.61→**2.39** vs
  true 1.93 (flipped UNDER→OVER-predict), lag −2→+2. For this DOB-enabled sim the reported estimator
  is the DOB `d_t` (not the read-out head): a MORE-biased decoder `g` (MV gain 0.81) leaves a LARGER
  recon residual for `d_t` to absorb ⇒ `d_t` inflates. **Downstream of the WM gain bias** — the WM
  fix is the primary Kalman lever (Kalman gain `dob_gain_init` is unchanged, so the over-shoot is not
  from K). Separately, dv_feedforward put the measured dv in `feat`, contaminating the read-out HEAD
  (which must predict the UNMEASURED load) — a real correctness bug to harden.
- **Actor went PASSIVE→ACTIVE but worse**: Fix 2 WORKED — `imag_adv_action_corr` STARTED **0.735**
  (vs ~0 in p129, a real policy gradient on μ) — but it **collapses** to ~0.01 over P3, `cv_viol` 187
  (was 104), `mv_viol` 0.368 (was 0.003), `cum_raw` −242k±66k (worse than −152k). `imagined_return`
  RUNS AWAY −13→−187 (NOT a numerical cascade — `return_scale` capped 20, `critic_pred_target_r` 0.99,
  MC engaged: the ACTIVE actor genuinely drives the BIASED WM into imagined violations);
  `imag_reward_dv_corr` HIGH 0.3–0.86 (imagined reward disturbance-dominated). Vision: disturbance-
  driven MV excursions 5–10%pp, 30–60 step period, CV violates, under-reacting. **Root: the Fix-2
  actor activation is PREMATURE on a biased WM (0.81/0.77)** — temper the econ push so it doesn't
  over-drive while the WM heals.
- **Unifying root**: the WM autoencoder bias (~0.86) **+ D1 starving the MV**. The Kalman head and the
  actor are largely DOWNSTREAM. Keep dv_feedforward (net-positive for DV repr + variance) and fix the
  side-effects.

### p131 — fix: REMOVE D1 (WM gain) + de-contaminate the disturbance head from the measured dv (Kalman) + temper econ shaping (actor)
- **Fix A (WM bias — the primary lever)**: **remove the D1 DV-isolated oversampling** entirely
  (falsified p129, superseded by dv_feedforward, and actively starving the MV gain). Deleted the
  machinery per the cleanliness mandate: the buffer `dv_isolated` tag (`__init__`/`add_episode`/
  `sample`), the `wm_dv_isolated_minibatch_frac` knob, the `DREAMER_WM_DV_ISOLATED_FRAC` override, the
  two `dv_isolated=True` add-site tags (seed + reinject), and the `sample(dv_oversample_frac=…)` call
  site. **Kept** `collect_dv_prbs_episode` + the DV-PRBS seed/reinject episodes (the excitation is
  fine — the oversampling was the harm) and **kept dv_feedforward** (it lifted the DV autoencoder
  0.72→0.86 + tightened the DV band). Expectation: the MV autoencoder recovers toward ~0.95 while DV
  holds ~0.86.
- **Fix B (neural Kalman / disturbance head — de-contaminate)**: the disturbance head must predict the
  UNMEASURED load, but `dv_feedforward` routes the MEASURED dv into `feat` ⇒ the head conflated the
  two (p130: r 0.78→0.54, std flipped under→over). New `_mask_measured_dv_from_feat` ZEROES the
  measured-dv columns of `feat` (`feat[…, core:core+dv_feed]`, `core=deter+stoch_flat`) before the
  disturbance head ONLY — the decoder + actor/critic heads still get the dv feed-forward, and the head
  can still infer the load indirectly via the `(h, z)` latent (removes only the DIRECT measured-dv
  shortcut). Applied in `_disturbance_head_loss` (training) AND the validation read-out probe
  (`evaluation/wm_disturbance_prediction.py`). Backbone-agnostic; no-op when dv_feedforward is off or
  the plant has no DV. Knob `disturbance_head_exclude_dv` (default True) + `DREAMER_DISTURBANCE_HEAD_
  EXCLUDE_DV`. **NOTE**: for DOB-enabled sims the *reported* disturbance estimate is the DOB `d_t`, not
  the head — its p130 over-shoot is downstream of the WM gain bias, so **Fix A is the primary Kalman
  lever** and Fix B is correctness hardening (matters when the head is the estimator / latent-shaping).
- **Fix C (actor — temper the econ push)**: `shaping_econ_coef` **0.5→0.25**. Fix 2 (p130) made the
  actor ACTIVE (`imag_adv_action_corr` 0→0.74) but the 0.5 econ push over-drove it on the still-biased
  WM ⇒ oscillation + `imagined_return` runaway. A gentler 0.25 lets the now-active actor track the WM
  while the gain heals (Fix A), instead of chasing economics into the WM's bias. Kept Fix 2b
  (disturbance-aware advantage baseline — it helped) + the Diagnostic-2 canaries. Restore 0.5 once the
  MV gain is back >0.9.
- **Each subsystem keeps its own readout for attribution**: the DV decomp (WM), the disturbance R²/`d_t`
  amplitude (Kalman), `imag_adv_action_corr` (actor), plus the R0 val CIs.
- **Designed-but-deferred (p132 if p131 insufficient)**: A2 = break the **0.86 autoencoder ceiling**
  shared by both channels via `wm_recon_cv_weight` 4→8 OR `rssm_n_classes` 32→48 (capacity). Verify
  Fix A first — removing D1 may recover the MV toward 0.945 without it.
- **Verification**: p131 smoke green (config defaults econ=0.25/exclude_dv=True/D1-knob-gone; buffer
  `sample`/`add_episode` reject the removed kwargs; `_mask_measured_dv_from_feat` zeroes exactly the
  dv cols + no-op when off; `_disturbance_head_loss` invariant to dv cols with exclude on, sensitive
  with it off — A/B works). DV-input, DOB, and curriculum-partition smokes green on **both** backbones.
  (The `_smoke_curriculum_e2e` STAGE-2 banner check fails identically on clean master b296fab — a
  pre-existing tiny-budget stage-latch artefact, unrelated; it disables the disturbance head.)
- **Judge by**: **MV ss-gain ratio recovers >0.9** (`wm_transfer_matrix.json`) AND DV holds ~0.86 with
  the band tight (`wm_dv_transfer_matrix.json` + DV decomp real→post ~0.86); the disturbance R²
  improves off −2.38 (ideally `d_t` amplitude de-inflates toward true 1.93); `imag_adv_action_corr`
  stays >0.1 (no collapse) + `imagined_return` stops running away + `cv_viol` down — with the val CIs.

### p131 — VERDICT: PARTIAL SUCCESS — fixes worked; the residual MV bias is now a cleaner COMPOUNDING root
- **What WORKED** (all three fixes did their job): (1) **removing D1 FIXED the MV autoencoder** — MV decomp
  real→post **0.869→0.909** (exactly as predicted; D1 was starving the MV). (2) **DV open-loop transfer IMPROVED
  0.769→0.868** (dv_feedforward held it + D1 no longer starving). (3) **Actor improved** — `cum_raw`
  −242k→**−182k** (CI tighter 66k→43k), `cv_viol` 187→**121**, `imagined_return` runaway BOUNDED −187→−52,
  `imag_adv_action_corr` sustained better (0.51 start, oscillates 0.05–0.21, ends 0.149 vs p130's collapse to
  0.01) — the econ temper 0.5→0.25 helped.
- **What DIDN'T fully resolve**: MV **open-loop** gain only 0.813→**0.828** (NOT >0.9) — the bias **moved from the
  autoencoder to COMPOUNDING**: MV decomp post→1step **1.001** (1-step prior faithful) but 1step→openloop **0.876**
  (the free-running multi-step rollout contracts the gain). Disturbance R² −2.38→**−6.07** WORSE — but `r` 0.53
  stable and amplitude ratio now **0.97** (calibrated): the DOB `d_t` **DRIFTS positive while the true OU
  disturbance is negative** (vision) = a SIGN/DRIFT bias, not amplitude. `mv_viol` 0.368→3.76 up (actor moves MV
  more — economically good, but over-shoots the limit).
- **ROOT of the residual (decisive probe)**: `tools/_probe_sampling_gain.py` measured the open-loop MV→CV gain
  under **sampled vs expected latent**: real −3.21, **sample=True 0.79×**, **sample=False (expected) 0.32×**. The
  expected path COLLAPSES the gain → the gain lives in the **learned SAMPLED prior**; sampling is NOT the
  contraction (it's the opposite), so this is **weak steady-state supervision**, not a categorical Jensen/EIV
  artefact. The overshoot loss (THE gain supervisor) ALREADY runs at full horizon (auto-tune `wm_overshoot_len`=
  horizon=**55**, NOT the run_plan's pre-auto-tune 15) but its **uniform `/K` mean dilutes the settled tail (the DC
  gain) to ~1/K weight** — coef 0.3 too weak there. The DOB `d_t` drift + the actor mis-calibration are BOTH
  **downstream** of this MV-gain contraction (a biased `g` → MV-correlated recon residual → `d_t` absorbs an
  MV-driven bias → drifts; the actor imagines a gain-contracted WM). One root fixes #1 + #2 + #4.

### p132 — fix: steady-state TAIL-WEIGHTING of the latent-overshoot loss (the open-loop DC-gain supervisor)
- **The fix (single variable)**: new `wm_overshoot_tail_power` (default **2.0**, `DREAMER_WM_OVERSHOOT_TAIL_POWER`).
  The overshoot loss now weights rollout step `k` by `(k/K)^power` (Σw-normalised), concentrating the gain
  gradient on the **settled tail** where the contraction lives, instead of a uniform `/K` mean that dilutes the DC
  gain to ~1/K. Bounded magnitude (still a weighted mean — no term inflation, can't destabilise the WM); the noisy
  early transient (already covered by 1-step recon/KL) is de-emphasised (p=2 ⇒ last step ~2.9× its uniform weight,
  first step ~0). RSSM-only (the gain supervisor is RSSM-only; SF/TSSM unaffected). `power=0` recovers the exact
  uniform mean.
- **Theory**: this is Simulation-Error-Minimisation / latent-overshooting applied to the **DC gain** — the
  steady-state gain is the **zero-frequency** response, so a low-frequency (settled-tail) emphasis is the
  signal-theoretically correct way to fit it. The probe ruled out the sampling-EIV alternative (sample=False is
  worse), so strengthening the learned-prior supervision is the right lever, not a sampling change.
- **Why one change**: the DOB `d_t` drift (#2) and the actor mis-calibration (#4) are downstream of the MV-gain
  contraction, so NO separate DOB/actor fix is bundled — the WM gain recovery should de-drift `d_t` (shrinks the
  MV-correlated residual it integrates) and de-bias the actor's imagination. Kept `wm_overshoot_coef` at 0.3 to
  isolate the variable; p133 bumps it (and/or the A2 autoencoder-ceiling levers) only if the tail-weighting is
  insufficient.
- **Verification**: `/tmp/p132_smoke.py` green (config default 2.0; env override wired; weighting math — p=0 exact
  uniform 1/K, p=2 last-step 2.92× uniform + first-step ~0; loss runs finite for both powers). `world_model_loss`
  integration green on **both** backbones (`_smoke_wm_fixes`); curriculum + dv-input green both backbones. Also
  fixed the STALE `overshoot==0 by default` assertion in `_smoke_overshoot_critic.py` (p117 promoted the coef
  0→0.3). (Other p117-promotion stale assertions — `wm_recon_cv_weight==1.0`, the critic-identity composition —
  REMAIN in those smokes, out of p132 scope; noted for a cleanup pass.)
- **Judge by**: MV **1step→openloop** decomp ratio rises toward 1.0 AND the MV open-loop transfer ratio >0.9
  (`wm_transfer_matrix.json` + `wm_posterior_prior_decomp.json`); the DOB `d_t` stops drifting (R² up off −6.07, no
  sign flip in `wm_disturbance_prediction.png`); `cv_viol` down further — with the val CIs for attribution.

### p132 — VERDICT: tail-weighting FIXED the compounding, but the autoencoder is the residual ceiling; data is GOOD
- **Tail-weighting worked on its target**: MV **1step→openloop ×0.876→×0.964** (the compounding contraction is
  largely fixed, exactly as designed). But the **autoencoder regressed** (MV real→post 0.909→0.837 — mostly
  run-to-run variance: real→post bounces 0.84–0.945 across runs), so net MV open-loop **0.856** (~flat), and DV
  regressed to **0.686**. The lever flipped back to `autoencoder` for both channels (prior stays perfect ×1.000).
- **Disturbance** (detrended, control-relevant): **det R² 0.05 / det r 0.52** — far below p129's 0.59/0.78. The
  drift fell (drift_sd 0.64) but the **dynamic error is high** (dyn_sd 1.02), tracking the DV-gain regression.
- **Actor**: `cum_raw` **−144k** (↑ from −182k), `mv_viol` 0.11 (↓), `cv_viol` 99.5 — but `imag_adv_action_corr`
  **oscillates 0.01↔0.59** and `critic_rew_to_tgt_var` decays **0.019→0.002** (bootstrap re-dominates P3).
- **Two user hypotheses FALSIFIED by measurement (valuable):** (1) **MV/DV correlation in the injected data** —
  measured `corr(MV,DV)=+0.004`, var-ratio 1.35 ⇒ **decorrelated + balanced** (the env drives the DV in MV-PRBS
  episodes too, independently). The data is good; the bias is **model-side**. (2) **raise `wm_recon_cv_weight`** —
  measured the **CV is the *highest*-variance obs channel** (var 2.43 vs MV 1.01 / DV 1.39), already over-weighted
  4×; raising it worsens the autoencoding "cheat" (reconstruct the CV *level*, not the input→CV *gain*). Confirmed
  by p110 history (cv6→autoenc 0.783, cv3→0.815, cv1→0.844: **higher cv BACKFIRES**).
- **Unifying RCA**: the **MV autoencoder ~0.84 ≈ p106's cv1 0.844** ⇒ the MV ceiling is the **categorical
  autoencoder's fundamental small-signal-gain limit** (not data, not cv_weight). The **DV (0.67) sits *below* that
  ceiling** ⇒ the dv-feedforward (1-dim drowned in the ~1500-d decoder MLP) is underutilized. The **actor
  oscillation is downstream of the WM gain under-estimate** (control theory, p112-confirmed: an under-gained WM →
  the actor over-actuates in imagination → overshoots the real, higher-gain plant → reduced phase margin).

### p133 — fix: zero-init DV→obs decoder skip (DV gain) + stronger critic MC-grounding (actor stability)
- **WM (Fix A)**: a **zero-init `dv → obs` linear skip** in the decoder of BOTH backbones — `out = decoder([h,z,dv])
  + W·dv`, `W` zero-init ⇒ exact no-op at start, learns the clean `∂CV/∂dv` from the residual, giving the exogenous
  DV a direct, high-gradient path that **bypasses both the ~1/1500 MLP dilution and the categorical bottleneck**.
  Targets the worst gain (DV 0.67 < MV 0.84, which proves the dv-feedforward is *not* actually direct) and cleans
  the DOB innovation. Readout: **DV decomp real→post** (should rise toward/above the MV ceiling).
- **Critic (Fix B)**: `critic_mc_grounding_coef` **1.0→2.0** (now ~6.7× the imag CE 0.3). At 1.0 the real-economic
  grounding held early P3 (rew_to_tgt_var 0.019) but decayed to 0.002 (bootstrap re-dominated → the advantage
  oscillated). A stronger MC anchor pins the critic baseline to realised economics through all of P3 → a stable
  advantage → a calmer actor. Readout: `critic_rew_to_tgt_var` stays >0.015.
- **Why these two (separate readouts)**: the DV-skip reads via the DV decomp; the critic via `rew_to_tgt_var`. The
  Kalman (#2) improves downstream of the DV gain; the actor (#4) improves downstream of both (correct DV loop +
  grounded baseline). Kept tail-weighting, econ temper 0.25, dv_feedforward, detrend metric.
- **Verification**: p133 smoke green (critic coef 2.0; dv_skip present + zero-init no-op + sensitive-to-dv after
  training, both backbones; absent when `dv_dim=0`). dv-input / DOB (vectorized==per-step 1.5e-7) / curriculum
  green on both backbones.
- **Designed-but-deferred (the MV ceiling — the real WM frontier; user decides the bigger bet)**: `rssm_n_classes`
  32→48 (categorical capacity), OR a continuous-latent component, OR a direct **identified-gain-matching** aux loss
  (match the WM's N-step step-response asymptote to the `dynamics_id` gain). `wm_recon_cv_weight` is **not** the
  lever — do not raise it (the p110 + variance evidence says lower-is-better for the autoencoder).
- **Judge by**: DV decomp real→post up (skip works) + disturbance **det R²** up off 0.05 + `critic_rew_to_tgt_var`
  holds >0.015 through P3 + `imag_adv_action_corr` stops oscillating + `cum_raw`/`cv_viol` improve — val CIs for
  attribution.

> **p134–p140 detail lives in `/memories/repo/mbrl_open_items.md`** (the human doc was not backfilled). Short
> thread to p141: p137–p139 chased the disturbance channel (return_scale cap fix, dist_match supervision); **p140**
> (innovation-driven cont-disturbance posterior, "Option B") was the structural win — **MV WM gain best-ever
> (MV@H 0.94)** and the disturbance finally **ENCODED** (held-out localization det_r 0.025→**0.320**, ≈ the DOB's
> 0.354) — but it **regressed the actor**: worst-ever econ `cum_raw` **−196k**, `cv_viol` 110, `mv_tv` ~2300
> (all 9 seeds catastrophic). RCA: the encoded load now ROLLS STOCHASTICALLY in imagination →
> `imag_reward_dv_corr` 0.44 (imagined reward disturbance-dominated) → `imag_adv_action_corr` 0.095 (action signal
> buried) → actor thrash → `return_scale` pegs the cap → critic bootstrap cascade. Also: disturbance AMPLITUDE
> attenuated (`dist_match_loss` 0.48→0.55 didn't converge, c_dist ~9% of true), and the DV WM transient off
> (DV@H 1.00→0.89; instant t=0 jump from the static dv_skip).

### p141 — fixes: R1 deterministic cont-disturbance roll (actor) + R2 dist_match 0.3→0.6 (rejection) + R3 remove the static DV feedthrough (cleanup)
- **R1 (PRIMARY — the actor, control theory)**: roll the cont **disturbance block DETERMINISTICALLY** (prior MEAN,
  not a sample) in `img_step` of BOTH backbones (`cont_dist_deterministic_roll=True`). The cont disturbance is a
  **feedforward** signal — the actor needs the PREDICTED load, not a per-rollout sampled realization that injects
  uncontrollable noise into the imagined reward (p140: `imag_reward_dv_corr` 0.44 buried the action signal →
  `imag_adv_action_corr` 0.095 → thrash → cap cascade). Mirrors the DOB persistence roll (`d_t=A·d`, no sampling).
  The GAIN block stays sampled. Removes the per-rollout `c_dist` noise the disturbance-aware advantage baseline
  could not (it only cancels the measured-DV common-mode). Readout: `imag_adv_action_corr` ↑, `imag_reward_dv_corr`
  controllable, `return_scale` stops pegging the cap, `rew_to_tgt_var` holds.
- **R2 (rejection amplitude, variational)**: `dist_match_coef` auto **0.3→0.6**. At 0.3 the `cont_kl` (KL→OU prior,
  which can't see the future load) **fought** the dist_match supervisor → `c_dist` settled at ~9% of the true load
  amplitude (phase-correct but crushed) → weak feedforward → poor rejection. A stronger supervisor lets `c_dist`
  fit the full amplitude against the KL. Complements R1 (deterministic + full-amplitude = strong, predictable
  feedforward). Readout: `dist_match_loss` converges, `c_dist`/pred_std → toward the true load std, CV-obs
  disturbance det_r ↓ (better rejection).
- **R3 (real cleanup — remove the static DV→obs feedthrough)**: `dv_static_skip` **default OFF** (was the p132
  always-on skip). The memoryless `W·dv_t` is (a) a **physically-wrong instant t=0 feedthrough** (DV→CV has
  dead-time, not feedthrough) and (b) a **`gain_match` crutch** — `gain_match` measures the full decode
  (`decoder([h,z,c]) + skip`), so the skip let it be satisfied with a WEAK dynamic DV path (slow rise, DV@H 0.89).
  The cont **GAIN block + `gain_match`** (p134+) are the principled DYNAMIC gain mechanism that **supersedes** it;
  removing the skip forces the latent path to carry the real DV transfer (gain + dynamics). Retained as an ablation
  lever (`DREAMER_DV_STATIC_SKIP=1`) to verify the supersession. KEEP `dv_feedforward` (dv-in-decoder, p129) —
  separate concern, next candidate if the DV transient is still wrong.
- **Verification**: new `_smoke_cont_dist_roll.py` (R1: sampled `img_step` disturbance block == prior mean +
  gain block still stochastic + flag-off restores stochasticity; R3: `dv_skip is None` by default, restorable) +
  dist_match / rssm / tssm / dob smokes ALL green both backbones. No band-aids (no MV penalty, no extra cap raise).
- **Judge by**: actor `cum_raw`→0 / `cv_viol`↓ / `mv_tv`→~979 (oscillation gone) + `imag_adv_action_corr` ↑ &
  `return_scale` off the cap + `rew_to_tgt_var` >0.015 held (R1); disturbance pred amplitude ↑ & CV rejection ↑ &
  `dist_match_loss` converges (R2); DV WM step-response shape (no instant jump, gain by horizon) + MV gain held
  (R3) — val CIs for attribution.

### p141 — VERDICT: R3 win, R1 partial, R2 backfired; the cont-disturbance channel is a structural dead end
- **R3 ✅** (remove the static DV skip): DV gain HELD/improved (DV@H **0.94**, @4H 0.98, decomp "faithful") — the
  cont-gain + `gain_match` genuinely supersede the p132 feedthrough.
- **R1 ⚠️** (deterministic disturbance roll): `imag_reward_dv_corr` 0.44→0.26 but the cascade STILL fired
  (`return_scale`→49.5 cap, `rew_to_tgt_var`→0.0008). Eased the symptom, not the disease.
- **R2 ❌** (dist_match 0.3→0.6): **diverged** (0.48→0.92); `c_dist` amplitude up but phase tracking destroyed.
- **Disturbance encoding REGRESSED**: localize probe on `best.pt` held-out full-latent det_r **0.32(p140)→ −0.05**
  (train 0.81 → test −0.05 = pure overfit). The learned cont-disturbance channel has now **failed 5 runs straight**
  (p137–141); −0.05 < 0.1 = the documented DOB-fallback threshold.
- **WM MV gain flipped to OVER**: MV@H 1.14, @4H 1.37 (autoencoder over-reads + open-loop drift — the prior never
  settles). Run-to-run swing 0.82–1.14 @H. **Actor catastrophic**: cum_raw −147k, cv_viol 93, mv_tv 2201, critic_r 0.085.
- **UNIFYING STRUCTURAL ROOT**: the large unmeasured load (CV swings ±2σ, MV authority small) is (a) not cleanly
  identifiable by the learned latent (ν confounds load with the WM's own gain error) and (b) dominates the imagined
  reward; since it isn't in the latent, the critic can't baseline it out → advantages = noise → actor diverges →
  `return_scale` caps. ONE root (the load), four symptoms. → revert the disturbance to the classical DOB.

### p142 — fix #1: revert the disturbance to the classical DOB (keep cont-gain), drop the failed cont-disturbance channel
- **The call** (system-ID theory): the learned "inherent amortized-Kalman" cont-disturbance channel failed 5 runs;
  the classical **DOB** (first-order neural-Kalman observer) is the *optimal* linear estimator for a Gauss-Markov
  load and previously hit det_r **0.354**. Re-enable it (`DREAMER_DOB_ENABLED=1`) as the disturbance estimator +
  Scope-2 feedforward (d_t in `feat`). **Keep the cont GAIN block** (it works — DV faithful 0.96); **drop the cont
  DISTURBANCE block + dist_match** (auto when DOB on: `cont_dist_dim=0`, `dist_match_coef=0`).
- **Why it fixes all four** (one root): the DOB de-confounds the WM gain (`d_t` absorbs the load → `g` learns the
  clean input→CV gain, #1), puts a `d_t` state in `feat` the critic CAN use (#3 → #4), and gives clean feedforward
  (#2, the proven 0.354). `gain_match` pins `g` so `d_t` cleanly gets the load residual (no gain↔disturbance fight).
- **Workflow change (automatic, no manual data work)**: with the DOB on, the run switches from the non-staged
  continuous-latent curriculum to the **staged clean→disturbance curriculum** — the textbook sysID recipe: **Stage 1**
  collects a CLEAN seed buffer + trains `g` (incl. cont-gain + `gain_match`) on the unbiased gain; **Stage 2** freezes
  `g`, trains the DOB (A,K) on the disturbance-laden innovation; **Stage 3** trains the actor on the frozen WM +
  working observer. The clean Stage-1 should also tighten the MV gain (#1). The P87 readout head auto-retires
  (`disturbance_head_dim=0`). Sim-agnostic (A,K learned, `cv_indices` auto, no-op when `n_cv=0`); both backbones.
- **Deferred (#2, on the todo)**: a `d_t`-conditioned **advantage baseline** (control variate) to remove the
  uncontrollable load from the policy gradient — the cascade's true fix, added next once the DOB revert is attributed.
- **Verification**: new `_smoke_dob_cont_gain.py` (gain-only cont + DOB layout, dist_match=0, DOB d_t live, clean
  g↔dob freeze partition) + dob/dist_match/cont_dist_roll/curriculum/rssm/tssm smokes ALL green both backbones.
- **Judge by**: disturbance det_r → toward 0.354 (DOB d_t estimator) + MV gain @H toward 1.0 & stable @H==@4H
  (clean Stage-1) + `critic_r` recovers off 0.085 + actor `cum_raw`/`cv_viol`/`mv_tv` improve — val CIs.

---

## Real-sim controller runs (mbrl2 fork — separate `pNN` numbering)

> **2026-07-07 — the pivot.** Imagination is **deleted**; the WM(RSSM)+DOB is a **frozen observer** and the
> actor-critic trains on λ-returns from **real rollouts of the true simulator** (`_realsim_actor_critic_step`,
> `actor_train_source='realsim'`) with domain randomisation. These `p01…` rows are the **fork's own** runs (a fresh
> counter — NOT the imagination-era `p95–p142` above). Imagination-only metrics (`imag_adv_action_corr`, the
> `return_scale` cascade, the WM-@H-gain-as-seen-by-the-actor) no longer apply; judge on validation econ /
> CV-MV-violation / disturbance-rejection + the P3 actor-critic canaries (`critic_r`, `critic_rew_to_tgt_var`).

### p01 (`run_20260707_realsim1`) — VERDICT: FAILED, entropy-collapse @ iter153 — off-policy REINFORCE
- MV = high-freq **CHATTER** spanning 70-90 % of range (eval deterministic ⇒ it's the POLICY, not sampling);
  validation cum_raw ≈ **−560k**; critic_r 0.23, critic_rew_to_tgt_var 0.0023, adv_corr 0.013.
- **ROOT CAUSE (confirmed)**: the P3 loop sampled the **shared replay `buf`** (Phase-1/2 PRBS/random/step-test/expert
  seed actions + stale-policy eps) and did **vanilla REINFORCE `−(adv·logp)` on those OFF-POLICY actions**. REINFORCE
  is on-policy-only ⇒ the gradient pulls π toward imitating whatever full-range seed actions scored +adv = exactly the
  chatter. (Imagination was safe because `buf.sample` gave only START STATES; the actions were freshly sampled from the
  current policy INSIDE the WM rollout, and `batch['rew']` was ignored.) `mv_move_weights=0` in control_objective.json
  was a secondary contributor, not the primary lever. Observer/WM FINE (wm_gain 5.7 %, next-state r 0.48).
- **FIX (p02/p03)**: a dedicated rolling **on-policy buffer** `onpol_buf` (recent current-policy eps only) — the P3
  actor-critic samples IT, not `buf` — + certainty-equivalent belief (posterior MODE, sample=False, in both collection
  and eval).

### p02 — deleted (on-policy-buffer attempt; superseded/cleaned before a full verdict).

### p03 (`run_p03_wmopt`) — VERDICT: FAILED, entropy-collapse @ iter204 — INVERTED CRITIC, MV pinned high
- On-policy buffer + the wm-opt (compiled `img_rollout` + every-other-step aux, commit 3c26649). **VISION-confirmed**:
  MV/REFLUX **PINNED FLAT at max ~95-100 %** of the episode; CV crashed **~13-15 °C BELOW** band ~95 % of the time
  (too cold from max reflux). mean_mv_viol 448.9, mean_cv_viol 580.7, cum_raw −934235, best_p3_det_return **−1972**
  (WORSE than p01's −1235).
- **SMOKING GUNS**: **critic_r_observed −0.231 (INVERTED!)** (p01 was +0.229) ⇒ wrong-signed advantage ⇒ actor
  **maximizes** reflux (should minimize); critic_rew_to_tgt_var **0.0003** (healthy >0.015 = severe bootstrap
  dominance); return_scale **49.5** (pinned at cap); wm_gain_rel_err **0.123** (12.3 % MV — WORSE than p01's 5.7 %,
  regressed by the overshoot every-other-step gating). DV decomp gain_ratio_post_vs_real 0.933 (6.7 % under),
  dominant_lever "faithful", "residual TM bias is open-loop COMPOUNDING".
- **ROOT CAUSE**: the p02/p03 on-policy fix **cured the chatter but removed the critic's two stabilizers**:
  (a) **no MC-grounding** (the bootstrap-only λ-return drifts with no real anchor → rew_to_tgt_var 0.0003), and
  (b) the **on-policy buffer STARVED the critic of state diversity** — once the actor drifts to a corner, the buffer
  holds only corner states → the value head inverts. p01's diverse replay buffer kept critic +0.229. Budget was NOT the
  bottleneck (46 of ~444 P3-iter ceiling used; P1-cap = plateau-detector artifact).

### p04 (`run_p04_criticfix`) — fixes: MC-ground + critic-diversity buffer-split + un-gate overshoot + more exploration
- **Fix 1 (MC-ground the critic)**: re-added `critic_mc_grounding_coef` (default **1.0**, `DREAMER_CRITIC_MC_GROUNDING_COEF`;
  pruned during the imagination cull). The critic target is the λ-return CE **plus** `coef ×` the PURE discounted
  reward-to-go (λ=1, no bootstrap) CE — anchors V to realised economics so the advantage sign can't invert. Real-sim
  gives full real episodes ⇒ a cleaner MC anchor than the deleted imagination version.
- **Fix 2 (critic diversity — actor/critic buffer split)**: `_realsim_actor_critic_step(…, critic_batch=None)`. The
  ACTOR advantage + REINFORCE + entropy stay **on-policy** (`batch` from `onpol_buf`, unbiased policy gradient); the
  CRITIC λ-return CE + MC now train on the **DIVERSE shared replay** (`critic_batch` sampled from `buf` in the P3 loop).
  A value BASELINE is action-independent ⇒ off-policy replay is UNBIASED for REINFORCE, and it keeps the value head
  well-conditioned when the actor sits in a corner.
- **Fix 3 (un-gate the overshoot gain-supervisor)**: the overshoot loss (the OPEN-LOOP gain supervisor) runs **every**
  WM step again (p03's every-other gating slipped MV gain 5.7→12.3 % + let the DV compound open-loop). The held-rollout
  (drift, less gain-critical) stays every-other for the compiled-`img_rollout` speedup (~360→~461 ms, still 1.37×
  faster than pre-opt 676 ms).
- **Fix 4 (more early exploration)**: `policy_init_log_std` auto-tune **−2.0 (σ0.135) → −1.5 (σ0.22)** so the early
  on-policy data is diverse enough to keep the now-grounded critic conditioned before entropy can collapse
  (plant-adaptive `policy_log_std_max` still bounds it from above).
- **Verification**: `get_errors` clean; `_smoke_rssm test_sim` green incl. the new `critic_batch` split + MC-grounding
  path (critic_loss ≈ 2·log(255) confirms both CE terms present; overshoot runs every step). Launched env-free
  (tmux `mbrl2_p04`, CVD=0, DOB=1, dedicated venv).
- **Judge by**: **critic_r > 0** (NOT inverted — the primary canary) AND `critic_rew_to_tgt_var` > 0.015 AND
  `return_scale` NOT pinned AND entropy NOT collapsed AND MV NOT railed high AND `wm_gain_rel_err` back < ~0.07 — val CIs.

### p04 — CRASHED at the P2→P3 boundary (logging bug, not a training result)
- Crashed on the **first P3 log row** (iter 159, ~2.5 h into P1/P2) with `TypeError: Object of type Tensor is not JSON
  serializable` at `json.dumps(row)`. A stray non-JSON diagnostic value (0-d torch Tensor / numpy scalar) reached the
  per-iter train-log write. The row is normally all-scalar (base dict + the float() loss merge + the diag-per-head
  float()/str loop; the `_realsim_actor_critic_step` return dict is provably all 0-d scalars — reproduced with
  DOB+dv+bf16 autocast). p03 logged 46 P3 rows fine (44 keys, all float/int/str/None), so it is a latent heisenbug.
- **FIX (a5e147a, training-neutral)**: `_coerce_row_for_json(row)` coerces any torch/numpy value in the row to a Python
  scalar IN PLACE before `json.dumps` + prints a **one-time** `[log-row] coerced non-JSON …` warning NAMING the key (so
  p05 captures the true source). Fresh restart (init_from_ckpt only loads weights, doesn't skip phases) reproduces the
  same P1/P2 and clears P3. No p04 P3/validation result — the 4 critic fixes are evaluated in **p05**.

### p05 — relaunch of the p04 recipe (4 critic fixes) with the logging fix
- Identical recipe to p04 (MC-ground + critic-diversity split + un-gate overshoot + more exploration), env-free, DOB on.
- **Judge by**: same as p04 (`critic_r > 0`, `critic_rew_to_tgt_var` > 0.015, `return_scale` not pinned, entropy not
  collapsed, MV not railed, `wm_gain_rel_err` < ~0.07) — PLUS watch the workflow.log for a `[log-row] coerced …` line
  that would name the p04-crash root.

### p05 — VERDICT: same collapse as p03 — actor-collapse-ON-UNFREEZE (not a critic-health problem)
- MV pinned at the high limit, `entropy_collapse` early-stop @ iter193, `cum_raw −942k` (all 9 seeds −366k…−1274k).
- **The critic fixes WORKED at warmup** (`critic_rew_to_tgt_var ≈ 0.015` healthy for iters 149–158, actor frozen; p03
  was 0.0003). But the **actor collapses the instant it unfreezes** (iter 159): `actor_loss −498`, entropy → floor
  (−1.017), `return_scale` → cap (49.5), μ → MV=max. `critic_r = −0.04` is DOWNSTREAM (collapsed actor → no validation
  state diversity), not a primary inversion.
- **RCA**: the auto-tuned σ floor `σ_min = σ_max/2.5 = 0.0875` sits **below** the −0.817 entropy early-stop trip, so on
  unfreeze REINFORCE crashes σ to that floor → tiny σ makes `logp` explode → `actor_loss −498` → μ commits to the
  max-reflux corner → cascade. (The KL trust region was already tried + reverted in p136 — it chases the collapse.)
- The **logging fix (a5e147a) named the p04-crash root**: `[log-row] coerced … ['t_ac_s']` — the Fix 2 critic_batch
  loop reused `_t` (the AC-step timer) → `t_ac_s` became a Tensor. Fixed (renamed `_t`→`_ct`).
- WM MV gain now **over-reads** (ratio 1.15 / 15% — un-gating overshoot made it worse); DV gain near-perfect (0.99).

### p06 — fix: stop the actor-collapse-on-unfreeze (raise the σ floor) + gentler actor + stronger econ potential
- **(1) σ FLOOR** (`sigma_min_ratio` 2.5→1.6, σ_min 0.0875→0.137, entropy floor −1.017→**−0.57** > trip) + the auto-tune
  `max(2.0,…)` clamp → `max(1.3,…)`: σ can no longer collapse to a near-deterministic corner; on-policy exploration stays
  alive so the warmup-healthy critic keeps getting diverse states to guide μ.
- **(2) `phase3_train_steps_per_iter` 25→8**: the real-sim on-policy buffer is FIXED per iter, so 25 reuses overfit the
  corner in one unfreeze iter.
- **(3) `shaping_econ_coef` 0.25→0.5**: strengthen the EXISTING margin-gated economic potential
  (`_shaping_potential`/`_economic_potential`) — potential-based / policy-invariant (Ng 1999), sim-adaptive, validation on
  the UNSHAPED reward — so the tiny economic gradient is visible against the ~2500× larger cv-violation term driving
  μ→max. (Objective-preserving; the deeper reward-asymmetry redesign is deferred.)
- **Judge by**: the actor does NOT collapse — entropy stays > −0.817 (no early-stop), MV **explores** (not pinned at
  max), `return_scale` not pinned, `critic_r > 0` on validation (diverse states). If μ still drifts to max without fully
  collapsing → the reward asymmetry needs the bigger fix next.

### p06 — VERDICT: anti-collapse worked, but exposed a critic INVERSION → rail-to-rail bang-bang MV
- The σ-floor held (entropy pinned at the floor -0.571, no collapse to a corner) so the **MV now moves** — but the actor
  learned a degenerate **rail-to-rail BANG-BANG** policy (vision: REFLUX slams 15%↔85%; cum_raw -530k, mv_viol 422,
  cv_viol 98.6). `actor_loss` runs away on unfreeze (iter 169: -69 → -2039).
- **RCA: val `critic_r = -0.20` (INVERTED)** — a *different* inversion than p03. The p05 buffer-split (Fix 2) trains the
  critic ONLY on the diverse replay `buf`, but the ADVANTAGE baseline V is evaluated on the actor's **on-policy** states
  — states the critic never fit → V(on-policy) is a wrong extrapolation → wrong-signed advantage → bang-bang. The
  tanh-boundary `logp` explosion is the numerical signature of μ driven to the action rails. (The early-stop is
  performance-aware — it tripped with `adv_corr 0.044`, correctly flagging the degenerate policy.)
- **WM (confirmed):** MV gain near-perfect (ss ratio **0.99**); DV gain **0.91** (9% under) — decomp says the bias is the
  **autoencoder** (real→post 0.888) while the **dynamics/prior is perfect** (post→1step 0.9995): DV is a subdominant
  regressor drowned in the categorical latent. **DOB/Kalman:** detrended r **0.65** (dynamic OK) but a large slow
  **drift** (raw r negative), partly downstream of the DV-gain bias.

### p07 — fix: train the critic on the ON-POLICY states too (correct advantage → learn good control)
- The p05/p06 buffer-split made V an extrapolation on the actor's states. Fix (DreamerV3-faithful — the critic must fit
  where the actor acts): `_realsim_actor_critic_step` now trains the critic on **BOTH** the on-policy `batch` (advantage
  accuracy) AND the replay `critic_batch` (diversity, still prevents the p03 corner-starvation), MC-grounded on both.
  **NOT a move penalty** — a correct advantage lets the actor *learn* that bang-bang is worse than smooth low-reflux
  control (the user's constraint).
- **Judge by**: `critic_r > 0` (not inverted), `adv_corr` up off ~0.04, `actor_loss` bounded (no tanh-rail runaway), MV
  **smooth** (not bang-bang), CV in band with low reflux, cum_raw → 0. DV-gain representation fix + DOB drift are
  deferred follow-ups (the actor is the blocker; the observer bias is minor since the actor trains on the real plant).

### p07 — VERDICT: inversion FIXED (best run so far), residual actor OSCILLATION
- **The on-policy-critic fix WORKED.** Val `critic_r = **+0.204**` (was **-0.20** INVERTED in p06) — correct-signed,
  no more bang-bang. **`best_p3_det_return = -1223`** — the BEST since p01 (-1235); p03/p05/p06 were -1700…-1972.
  `cum_raw` -444k (p06 -578k), `mv_viol` 230 (p06 422), `wm_gain_rel_err` 0.045 (healthy). P3 ran 221 iters (vs ~45).
- **Residual: a high-frequency actor OSCILLATION (limit cycle).** Vision: MV toggles fast ~20%↔80% (NOT pinned at one
  rail = "less severe" than p06's bang-bang), CV sawtooths across the band, `cum_raw` spirals. Best policy is **early**
  (iter 186) then DEGRADES.
- **RCA — the critic is correct-signed but WEAK and DECAYS into bootstrap-dominance.** `critic_rew_to_tgt_var` decays
  0.0138 → **0.0002**; `return_scale` runs to its cap 49.5; `realsim_return` spirals -8 → -193. A weak/noisy advantage
  can't clearly credit *smooth-vs-oscillating* control → the actor settles into a high-gain limit cycle → CV violations
  → returns spiral → `return_scale` cascade → noisier advantage → sustained oscillation + degradation after iter 186.
  **Not a WM problem** (gain 4.5%, `wm_next_state_r` 0.447). Not the actor collapse anymore.

### p08 — fix: strengthen the critic's real-return grounding (keep the advantage accurate)
- **`critic_mc_grounding_coef` 1.0 → 2.0.** The on-policy critic fixed the *sign* but not the *strength*; the fix keeps
  V anchored to **realised economics** through ALL of P3 (not just the bootstrap) so the advantage keeps correctly
  penalising the oscillation's violations → the actor **learns smooth control from the objective** — **NOT a move
  penalty** (the user's constraint; the agent must learn that oscillation is bad control). Proven lever (imagination p133).
- **p08 carries ALL fixes**: on-policy critic (p07) + MIMO WM gain fix (cont-gain `gain_dim=2` + self-supervised input
  isolation 0.5, `gain_match` DISABLED = fully unsupervised) + MC-grounding 2.0. `test_sim` is LINEAR → a clean
  isolation-only test of the WM gain fix, and the MC-grounding boost targets the actor oscillation.
- **Judge by**: (actor) `critic_r > 0.3` (was 0.204), `critic_rew_to_tgt_var` stays > 0.015 (not → 0.0002),
  `return_scale` NOT pinned at 49.5, MV **smooth** (not oscillating), `cum_raw` → 0; (WM) `wm_input_isolation_loss`
  present + decreasing in P1/P2, `wm_dv_transfer` ratio > 0.93 (was 0.91), DV decomp real→post ~0.94+ (was 0.888), MV
  holds ~0.99; (DOB) raw r positive + drift down. If it still oscillates but less → escalate MC-grounding further or add
  a REINFORCE-variance lever (fresher on-policy data / lower `lr_actor`).

### p08 — VERDICT: MC-grounding is a BIG win (first all-gates-pass), but isolation-only REGRESSED the WM gains
- **The MC-grounding 2.0 fix worked well.** Val `critic_r` **0.204 → 0.733**, `wm_next_state_r` **0.447 → 0.721**, and
  p08 is the **FIRST run where ALL fidelity gates pass** (`critic_pass`, `wm_pass`, `wm_gain_pass`, `reward_pass`). No more
  hard bang-bang; `mv_violation` mean 230 → 112.
- **But two residual symptoms (user-reported): a WM bias and "the MV is noisy sometimes."** Both trace to ONE root cause.
- **WM bias:** the transfer matrix shows BOTH steady-state gains **under-estimated** — CV←MV real −0.32 wm −0.275 (**×0.86**,
  was 0.99 in p07) and CV←DV real +0.18 wm +0.141 (**×0.78**, was 0.888). Disabling `gain_match` (the isolation-only
  experiment) **regressed** both gains: `_wm_input_isolation_loss` enforces input *separation* but does not pin gain
  *magnitude*, so the cont GAIN block under-shoots.
- **MV noise (intermittent, condition-dependent):** per-episode `mv_violation` = [159, **21.6**, **425**]; seed box-plots
  smooth for s3/s5 (~40-58) but noisy for s6 (~240). Mechanism = the WM gain bias: under-est **MV gain (0.86)** → the actor
  believes its authority is weak → **over-actuates** (noisy MV, mv_viol ≫ baseline 0); under-est **DV gain (0.78)** →
  mis-predicts disturbance effects → **mis-reacts during disturbances** → the noise is condition/seed-dependent. As the
  curriculum ramps disturbances through P3 the mis-reaction worsens → `realsim_return` spirals (−2 → −216), `return_scale`
  → cap 49.5, best is **early** (iter 161) then degrades. Agent is still worse than the baseline (cum_raw −417k vs ~−130k).

### p09 — fix: re-enable `gain_match` (pin the WM gain magnitude the isolation loss can't)
- **Re-enable the identified-gain anchor ALONGSIDE isolation** (the intended MIMO design). `gain_match` pins magnitude
  (linear/identifiable plants — test_sim's identified gains −0.32/+0.18 exactly match the transfer-matrix truth), isolation
  keeps the MIMO structure + nonlinear generality. Removed the forced `gain_match_coef=0`; wired
  `_resolve_gain_match_targets(env, cfg)` just before the training loop (obs-norm fitted by then), guarded so a missing
  `dynamics_identification.json` falls back to isolation-only. Fixes BOTH symptoms via the shared root cause: correct WM
  gains → the actor no longer over-actuates (MV gain) or mis-reacts to disturbances (DV gain).
- **Judge by**: (WM) transfer matrix CV←MV → ~1.0×, CV←DV → ~0.95×+ (both were 0.86/0.78); `wm_gain_rel_err` ≪ 0.14;
  watch for the `[gain-match] targets …` banner in the log (confirms the anchor is active, not the fallback). (Actor) MV
  **smooth across all seeds** (not just some), `mv_violation` ≪ 112, agent beats the baseline, `realsim_return` doesn't
  spiral. If the WM is now correct but the actor still degrades late → that's an independent actor-stability issue for p10.

### p25 (`run_p25_tbptt`) — VERDICT: TBPTT stopped the P24 blow-up; DC-gain + DOB grounding still dead
- P1 completed (0 grad-skips, max `wm_grad` 40 vs P24 57 skips / 7e13). P1→P2 **capped GAIN_NOT_READY** (DC [0.72, 1.00]).
- Val (best.pt = P3 it176): MV ss/**@H ×0.74 / ×0.60**, DV ×1.16 / ×1.12; WM r 0.167; disturbance det_r 0.24; critic_r 0.087; econ −194 vs baseline −102. Entropy-collapse early-stop @345. `return_scale` 1.12→49.5 by P3 it185.
- **ROOT (observer):** (1) `RSSMState.detach()` every 16 of K=55 on the gain-match *asymptote* cut the DC-gain gradient (forward Huber loss ~0.002, transfer still ×0.74). (2) DOB-on forces `disturbance_head_dim=0` → buffer `n_dist=0` → `batch['dist']` missing → `dob_ground` identically 0 for all 54 P2 iters. **P19 was the same** — the P19 "grounding win" was not this loss.
- **ROOT (actor):** independent critic cascade (single twohot + λ-bootstrap + p95–p5 `return_scale` EMA). Stage 2 after the observer freeze is READY.

### p26 (`run_p26_gainbptt`, branch `grok` @ 1517d91) — VERDICT: observer GAIN healthy; DV residual + actor cascade remain
- **Observer WIN:** MV ss/@H **×0.973 / ×0.880** (`wm_gain_rel_err=0.026`, `wm_gain_pass=True`) vs P25 ×0.74/×0.60. Full-BPTT gain-match + live `n_dist=1` worked. P1 0 skip-storm. P2 `dob_ground` live (0→0.010). det_r **0.68** (P25 0.24).
- Residual DV ss **×0.866** (@H ×0.998). Absolute Huber on WM-norm targets under-weights the smaller DV (|tgt_mv|≈2.46 vs |tgt_dv|≈0.52). P1→P2 CAPPED `gain_not_ready` (DV DC 0.75 at iter 75); freeze still good enough that val GAIN passed.
- **Actor FAIL (independent, Stage 2):** `return_scale` 1.19→**49.5 cap**, `critic_rew_to_tgt_var` 0.041→**0.00015**, entropy-collapse @262 (`adv_corr=0.05`), critic_r 0.079, econ **−424 vs baseline −117**, MV reversal 0.665, noisy MV near the high limit. Same cascade as P25.
- Log noise (not training bugs): `[val] P1: shortcut-forcing loss did not drop` (RSSM `sf_loss≡0`); `[gate p1->p2] FAIL (gain_not_ready) … wm_ema_best=5.211 < 1.50` (wrong reason string).

### p27 (`run_p27_relgain_qens`) — VERDICT: P1 skip-storm abort; relative gain-match exploded the observer; actor/critic NEVER trained
- **Did not complete.** Early-stop `grad_skip_storm: 42 skips in last 100 iters` at P1 iter 50 (`wm_grad_norm=6.0e12`, recon 0.0039→0.496, `gain_match_loss` 2.6e-4→0.64, isolation 0.005→13.1). P2=0, P3=0. Validation used exploded `final.pt` (`best.pt` missing).
- **Observer FAIL vs P26:** MV ss/@H **×1.95 / ×1.90** (P26 ×0.97 / ×0.88); DV ss/@H ×1.13 / ×0.80 (P26 ×0.87 / ×1.00); `wm_gain_rel_err` 0.95 vs 0.026; autoencoder lever (real→post **×0.26**). Iter 49 was still healthy — the relative Huber DV boost detonated full-BPTT on the next step (same marginally-stable recurrence as P24).
- **Apparent actor win is the expert-BC launchpad.** econ **−59 vs baseline −77** (beats_baseline) and cum_raw −68k vs P26 −585k, but `critic_r=NaN`, `reward_head_r=NaN` (heads at twohot prior, pred_std 0), MV reversal 0.70. Closed-loop plots are P1 `bc_scale=0.15` static expert, not min-of-2 / freeze-return_scale.
- **ROOT:** `gain_match_relative=1` (only observer delta vs P26). Actor knobs (`n_critics=2`, `return_scale_freeze_after_warmup`) never ran.

### p28 (`run_p28_absgain_qens`, branch `cursor`) — FIX: restore P26 observer; keep untested actor knobs; recover P1 skip-storms
- Revert `gain_match_relative` default **1→0** (absolute Huber, P26 observer). The opt-in A/B path was later **removed** (P31 GPU-occupied) so future runs cannot re-enable it via env.
- Keep TD3 min-of-2 twohot critics + freeze `return_scale` after critic warmup (first real P3 test of the P26 cascade fix).
- P1 skip-storm: restore `wm_best.pt`, reset `opt_world` AdamW (drop 1e12 moments), cap P1 so the next iter is Stage 2. P2/P3 skip-storms still abort. `DREAMER_SKIP_STORM_RECOVER_P1=0` to disable.
- Same env stack as P26/P27 so the observer delta is only the relative-Huber revert. Run ALONE from `neural-APC-mbrl2-cursor`.
- **Judge by:** P1 completes (or recovers); MV ss/@H ~×1.0 ±0.1 (P26-class); P3 actually runs; `return_scale` not pinned at 49.5; `critic_rew_to_tgt_var` stays >0.015; no entropy-collapse; econ beats baseline AND the P27 BC score (−59); MV not railed/chattery.
- **Code follow-up (no GPU this session):** skip-storm recovery now closes `p1_gate_max_ext_steps` (`_force_p1_cap_at`) so a recovered P1 cannot re-open the quality-gate extension and re-explode gain-match. `gain_match_huber_beta` / `aux_tbptt_steps` / `wm_grad_skip_norm` promoted to TrainConfig + `ENV_OVERRIDES` (defaults unchanged). `gain_match_huber_beta<=0` auto-sets median |tgt| (opt-in sim-adaptive; not the P28 default).
- **Code follow-up 2 (no GPU this session):** (1) Curriculum freeze/DOB now re-applies on the **same iter** as `current_phase` changes — the loop-start latch left `g` trainable for the first P2 train step (full-BPTT gain-match on skip-storm-restored weights; also leaked one extra gain-match step on every healthy P1→P2). (2) Isolation TBPTT stride is sim-adaptive: dataclass 16 (test_sim K≈55) becomes `max(8, round(K/3.5))` unless `DREAMER_AUX_TBPTT_STEPS` is explicit. (3) Gain-match Huber β is not re-read from env on the hot path (env `0` = auto-median would overwrite the resolved β and pass β=0 into `smooth_l1`).
- **Code follow-up 3 (no GPU this session):** skip-storm recovery restored fidelity-peak `wm_best` (gain-blind / noise-led). P27's last healthy observer was iter 49 — restoring an early lucky spike would discard late-P1 excitation. Keep an in-memory last-ok snapshot on skip-free P1 iters whose recon is within 5× the best (unitless); restore that first, `wm_best` only as fallback (`wm_last_ok.pt` written only if a storm fires). Summary `actor_experiment_valid=False` when the freeze is `GAIN_NOT_READY` or skip-storm fell back to `wm_best`.
- **Code follow-up 4 (no GPU this session):** follow-up 3 was still undone on the next iter. Default `wm_best_restore_at_p2` reloads fidelity-peak `wm_best.pt` at P1→P2 (p124: required on a *healthy* P1). After skip-storm the next iter **is** P1→P2, so last-ok was overwritten by the same gain-blind spike. Skip the boundary wm_best reload when `skip_storm_p1_recovered`. Healthy P1 path unchanged. Knobs promoted to TrainConfig + `ENV_OVERRIDES`.
- **Code follow-up 5 (no GPU this session):** P1 re-inject EVERY (const/step 20, DV-PRBS 10) was a raw iter count — one test_sim buffer-lap fraction (`400k/(1220×5)≈66`). `_resolve_inject_cadence` now sets const/step/expert = `round(0.30×lap)` and DV-PRBS = `min(round(0.15×lap), warmup/4)` so a longer-τ plant (fewer episodes in the same step-cap) injects more often and a faster plant less often, while p122's "inject before typical wm_best" still holds. test_sim stays 20/10. `0` still disables. Gain-ready / wm-fidelity knobs promoted to TrainConfig + `ENV_OVERRIDES` (defaults unchanged; already unitless).
- **Code follow-up 7 (no GPU this session):** `wm_isolation_settle_episodes=24` was a **side-total** (24 MV-together + 24 DV-together) and isolation_buf cap ignored it (48 = baseline+dv_prbs+8). MIMO wrap dropped all but the last 48 settle episodes; MV settle PRBS'd every actuator at once and inherited the curriculum DV schedule from `reset()`. Auto 24 is now **per isolated input** (test_sim stays 24+24 / cap 48; distillation 4+1 → 96+24 / cap 120). Collectors hold other MVs / isolate one DV; long-hold settle suppresses curriculum DV + hidden OU (same as const-action seeds).
- **Code follow-up 8 (no GPU this session):** isolation_buf still ingested ordinary MIMO PRBS + all-DV PRBS. Follow-up 7 sized cap `max(baseline+dv_prbs+8, settle)` to keep test_sim at 48, but auto-tune baseline ~26 grew the cap to 58 so wrap left ~10 confounded all-DV episodes in front of the 48 settle — `wm_ss_match` trained on a MIMO mixture. Cap is settle-only (test_sim still 48; distillation still 120). Long-hold settle now also zeros process/measurement noise (`clean_steady_seeds`, same P89 gate as const-action / step-settle) so DC-gain ID is not errors-in-variables.
- **Code follow-up 9 (no GPU this session):** follow-up 8 still PRBS-stepped inside each settle episode (`seg_max = min(max(seg, 2K), T/4)` → ~11 holds of 110 steps on test_sim T=1220) and passed `action_std=baseline_seed_std` on the isolated MV. Random `seq_len=64` windows from `isolation_buf` straddled those steps (~half), so `wm_ss_match`'s `settle_var` gate down-weighted the DC-gain term; `_st_levels` was wired to `hold_level` of *other* MVs (no-op on test_sim). Isolation settle is now a **whole-episode constant hold** at the stratified `isolated_level` (`action_std=0`, others at 0). DV settle is one step at t=0 to `isolated_level × (span/2)`, MV held at 0. Ordinary MIMO PRBS / all-DV PRBS still cover transients in the main replay buffer.
- **Code follow-up 10 (no GPU this session):** follow-up 9 still multiplied the DV isolation step by `dv_prbs_op_frac` (`delta = isolated_level × 0.8 × span/2`). Isolation levels are already in MV-action units (`±constant_action_seed_op_band`), so DV |Δu| was 0.8× the matching MV step → smaller |ΔCV| → absolute isolation/ss-match MSE under-trained DV (same family as abs-Huber on unequal |tgt|, P26 DV ss ×0.87). Isolation DV step is now **MV-action-isomorphic** (`±1 ↔ ±half-span`); ordinary all-DV PRBS still uses `dv_prbs_op_frac`. Also: isolation sample windows are `max(seq_len, K+1)` so a slow plant with H > seq_len still reaches SS (test_sim seq_len ≥ H unchanged); isolation extra unroll is skipped when `g` is frozen (DOB curriculum P2).
- **Code follow-up 11 (no GPU this session):** follow-up 10 skipped only the isolation extra unroll. P2 `world_model_loss` still ran overshoot + held-rollout (~73% of the WM step) and full-BPTT gain-match (baseline + one K-step FD roll per input) after `g` was frozen. Those losses cannot update encoder/decoder/GRU/cont-gain; P2 is DOB A,K on recon + ground/reg. Skip the g-only aux when `_dynamics_g_trainable` is false (cadence counter still ticks). P1 unchanged.
- **Code follow-up 12 (no GPU this session):** overshoot + held-rollout `img_rollout` dropped posterior `c` (RSSMState constructed from `h,z` only → `img_step` zero-fills `prev.c`). The open-loop gain supervisor (~73% of the P1 WM step) therefore trained a **c=0 GRU path** while isolation / gain-match / actor / transfer-matrix start from posterior `c` (p20 family). `img_rollout(..., c0=)` now threads it; omit ≡ zeros (back-compat). test_sim recipe unchanged.
- **Code follow-up 13 (no GPU this session):** follow-up 10 grew *isolation_buf* windows to `max(seq_len, K+1)`, but P1/P2 still sampled the MAIN replay at `cfg.seq_len`. Overshoot (`K=min(K,T-1)`, needs future obs) and gain-match (`n_valid=T-K`) therefore truncated the identified settling length when `H >= seq_len` (slow plant or `DREAMER_SEQ_LEN` pin) — P25-family (forward Huber tiny, transfer-matrix DC dead). P1/P2 now sample `_wm_train_seq_len = max(seq_len, K+1)` (P3 on-policy stays `seq_len`). Gain-match no longer truncates open-loop `K` to `T-1` (held a/dv from the start; no future obs needed; test_sim T=64, K=55 start restriction unchanged). GPU probe uses the same T. Log: `wm_train_T=`. test_sim 64/55 unchanged.
- **Code follow-up 14 (no GPU this session):** follow-up 12 threaded posterior `c` into overshoot/held `img_rollout`, but sliced it from `rollout_observed(sample=True)` feat (the reparameterized sample). p20 already rolls *subsequent* gain at the prior mean (`cont_gain_deterministic_roll`); the **first GRU step** still saw `c_sampled` → supervisor trained `E[f(c_sampled)]` while isolation / actor / transfer-matrix start from the posterior MEAN (`sample=False`). Open-loop aux (overshoot, held, gain-match, 1-step steady) now start from `cont['post_mean']`. Recon still uses the sample. `_rssm_steady_consistency` also threads `c` + measured DV (was zero-filling both; gated off on the P28 recipe). test_sim recipe unchanged.

### p28 GPU VERDICT (`run_p28_absgain_qens` @ a7941be, 2026-08-25) — observer FAIL (gain-blind wm_best restore); actor INVALID
- Recipe on GPU was **pre-follow-up** (absolute Huber + n_critics=2 + freeze `return_scale` + skip-storm recover). Recover did **not** fire (0 skips). Follow-ups 1–14 were committed to `cursor/p28` from a CPU session and were **not** in this GPU job.
- **Observer FAIL vs P26.** P1 live gate MV DC **×1.94** / DV ×0.70, spread **×9.9**. Then `[p1→p2] loaded wm_best.pt (iter 60, gain badness 0.30) — discarded 37 iters`. P1-end r@H **0.50** → P2 iter 100 r@H **0.09**. Val `best.pt`: MV ss/@H **×0.52 / ×0.48**, DV **×1.56 / ×1.65**, `wm_gain_rel_err` 0.48, det_r 0.555 (P26 0.68). `wm_gain_pass=True` is the loose <2× gate — not healthy.
- **Why P26 skipped restore:** last P1 `wm_best` was iter 90, STAGE 2 at 98, gap 8 < min_gap 10. P28 fidelity peaked at 60 and never updated → restore fired. Follow-up 4 (skip restore *only after skip-storm*) would **not** have prevented this — P28 had no skip-storm.
- **Actor INVALID** (GAIN_NOT_READY freeze). Freeze `return_scale` **worked** (1.11→**2.29**, not 49.5). `critic_rew_to_tgt_var` still 0.038→0.00015; entropy-collapse @285; econ **−410 vs baseline −61**. Do not stack critic knobs on this freeze.
- **ROOT:** gain-blind `wm_best` P1→P2 restore on a healthy P1 (env/TrainConfig default ON). Not relative Huber (off). Not skip-storm.

### p29 (`run_p29_norestp2`, branch `cursor/p28`) — FIX: freeze end-of-P1 g (skip gain-blind wm_best restore)
- **Intended one attributed change vs origin:** `wm_best_restore_at_p2` default **True → False**. Log `[p1→p2] WM warm-restore SKIPPED (default-off …)`. Opt-in `DREAMER_WM_BEST_RESTORE_AT_P2=1`.
- Codebase also includes follow-ups 1–14 (never GPU-validated). If observer is not P26-class, isolate those next — do not stack more critic knobs.
- Keep min-of-2 + freeze `return_scale` so a GAIN-READY freeze is the first *valid* actor test of those knobs.
- Env-free, tmux `mbrl2_p29`, `output/test_sim/run_p29_norestp2`. Launch *intended* compile default-on (believed to match P26/P28); P26/P28 were actually eager (`DREAMER_COMPILE=0`, no compile banner).
- **Judge by:** SKIPPED restore (not `loaded wm_best.pt`); P1→P2 gain-probe MV in [0.8, 1.3]; val MV ss/@H ~×1.0 ±0.1, DV not ×1.56; P2 r@H stays ~0.5 (not 0.09). If still GAIN_NOT_READY: `actor_experiment_valid=False`, do not read econ as an actor result. If observer P26-class: `return_scale` stays near warmup, `critic_rew_to_tgt_var` >0.015, econ vs baseline **and** vs P27 BC −59.
- **LIVE (2026-08-26 P3 @iter 151, GPU still running): CONFOUNDED; actor INVALID.** Categorical leftover (`kl` 0.30–0.81, `jemb≡0`, recon **0.26–0.32** vs P28 0.003; wrap spike 0.65@49; overshoot ~0.10; `z_alive` ~94). Compile-on (iter-1 t_wm 1681 s). **Skip-restore DID fire** (`[p1→p2] WM warm-restore SKIPPED`) — still not a skip-restore A/B (observer is not P26). P1 CAPPED `not_plateaued` after iter-75 `gain_not_ready` DC **3.05@MV**; cap-time gain probe was skipped so `[actor] NOT an actor experiment` **never printed** and `actor_experiment_valid` would be True. P2 `dob_ground≈0.008` (n_dist live); P2→P3 PASS reward_mtp median=1.127. P3 started STAGE 3 on a GAIN_NOT_READY freeze — do not read econ. **Code (GPU occupied, no kill/relaunch):** cap-time gain probe on P1 CAPPED; `skip_invalid_p3=True` default (`[p3-skip]`, `early_stop_reason=p3_skipped_invalid_observer`); banner None-safe. Next GPU job after EXIT = env-free deterministic + eager (CUDA only).

### p29 GPU VERDICT (`run_p29_norestp2` EXIT=0 @ 15:12, 234 iters, entropy-collapse) — CONFOUNDED leftover class; skip-restore NOT judged
- **Liveness:** finished. GPU free. Wall ~6.2 h. `early_stop_reason=entropy_collapse_window` (30/30 below −0.163, `adv_corr=0.045`). `best.pt` iter 186 `det_return=−569`.
- **Confound (not skip-restore):** categorical (`kl` 0.30–0.72, `jemb≡0`, recon 0.50→0.29 @P1-end vs P28 0.003; wrap spike 0.65@49; `z_alive` 1024→93). Compile-on (`[dreamer_v4] torch.compile`, t_wm 1681 s iter 1 then ~148 s). Leftover `[env-override] dob_enabled=True`. Follow-ups 1–14 were in this GPU job but cannot be attributed.
- **Skip-restore DID fire** (`WM warm-restore SKIPPED`). Still not an A/B: observer is not P26-class.
- **Observer:** MV ss/@H **×1.10 / ×1.23** (sign OK, slightly fast/over). DV ss/@H **×0.56 / ×0.65** (direction OK, gain too small). `wm_gain_rel_err=0.097` MV-only printed HEALTHY. Decomp MV real→post ×0.82 / DV ×0.67 (autoencoder). det_r **0.725** / det_R² 0.43 (phase OK, amplitude poor). `wm_pred_converges_under_constant_action=0.0`. P2 `dob_ground` 0.019→0.008 live.
- **P1 gate:** iter 75 `gain_not_ready` worst=3.05@MV; CAPPED iter 97 without freeze-time probe. `p1_gain_not_ready_capped=false`, `actor_experiment_valid=true` (bookkeeping lie).
- **Actor INVALID.** Freeze `return_scale` 1.48→**4.85** (not 49.5 — KEEP). `critic_rew_to_tgt_var` 0.021→**0.00027**. `critic_r=0.193`. econ **−411 vs baseline −83**. Vision: pulse/chatter MV, CV high-side offset. Do not stack critic knobs.
- **ROOT:** env-free drop of `DREAMER_RSSM_LATENT_TYPE` + compile leftover. Not skip-restore.
- **Code this session:** `fidelity_gates` logs `wm_dv_gain_*` / `wm_dv_ss_ratio_worst` (MV `wm_gain_pass` unchanged).

### p30 (`run_p30_deteager`, branch `cursor/p28`) — FIX: env-free deterministic + eager (P29 leftover class)
- **One attributed change vs P29 GPU job:** TrainConfig defaults that P29 never had at launch — `rssm_latent_type=deterministic`, compile **eager**, `skip_invalid_p3=True` (cap-time gain probe). Do **not** pass leftover `DREAMER_COMPILE=0` / latent / DOB / act-hist.
- Skip-restore stays OFF. min-of-2 + freeze `return_scale` stay (valid actor test only if GAIN-READY).
- **Judge by:** train-start `latent=deterministic compile=eager`; no `torch.compile` banner; P1 `kl≈0` / `jemb>0` / recon like P28; `[p1→p2] WM warm-restore SKIPPED`; if CAPPED, cap-time gain-probe. Val MV ~×1.0 ±0.1, DV not ×0.56. If GAIN_NOT_READY: `[p3-skip]`. If GAIN-READY: `return_scale` near warmup, `rtgt>0.015`, econ vs baseline **and** vs P27 BC −59.
- **LIVE (2026-08-26 16:55, tmux `mbrl2_p30`, GPU ~19 GB):** still P1. Launch confirmed det+eager. **P1 iter 17:** recon **0.0098** `kl=0` `jemb=0.036` gmatch 0.0018 iso 0.0021 `ss_wmean=1.00` alive **1024** skip **0** (P28 iter 14 recon 0.012 / P26 0.010; P29 leftover recon ~0.29 alive ~93). t_wm ~160–165 s = P26/P28 eager, not P29 compile. Fidelity probe iter 10 `best_h=1/55` (gain-blind spike — restore is OFF so this is WATCH-only). Wait recon ~0.003 by ~iter 50, then skip-restore SKIPPED. **No second GPU job.** HEAD now has eager-path opt (batched isolation decode + vectorized overshoot MSE; same mean) for the *next* launch — P30 process is still `2f0aec9`.

### p30 GPU VERDICT (`run_p30_deteager` EXIT=0 @ ~17:36, 60 iters) — det+eager CONFIRMED; skip-storm cap froze P1 iter 18; P3 skipped
- **Launch succeeded** as the env-free leftover-class A/B. Train-start `latent=deterministic compile=eager skip_invalid_p3=True device=cuda bs=128`. `[resolved-cfg]` `restore_p2=False gain_match=1 dob_ground=2 isolation=1 ss_match=3 n_critics=2 rs_freeze=True`. No `torch.compile` banner. P1 `kl=0` `jemb>0` alive 1024, recon like P26/P28 until iter 18. Follow-ups 1–14 **were** in this GPU job (first time). Batched isolation/overshoot opt (`0a61872`) was **not** in the process (`2f0aec9`).
- **P1 skip-storm @iter 19:** recon 0.0045→**0.901**, gmatch 1.21, 7 skips, `wm_grad_norm=22` (not 1e12). Restore last-ok iter 18 **worked** (P2 recon 0.0125). Then `_force_p1_cap_at` **threw away remaining original P1** (~362k of ~754k steps). Cap-time probe GAIN_NOT_READY DC `[0.88, 1.44]` worst=`1.44@MV`. `[p1→p2] WM warm-restore SKIPPED`. `actor_experiment_valid=false`, `skip_invalid_p3=true`, `p1_gain_not_ready_capped=true`.
- **P2 fidelity-ES killed DOB:** first P2 probe iter 20 `r@H=0.757` gain_fid=0.923 → `wm_best` 5.644; `wm_fidelity_degradation` patience 40 (g frozen — probe cannot improve). P3 never started (correct; ES hit P2 first).
- **Observer FAIL vs P26.** MV ss/@H **×1.882 / ×1.678** (`wm_gain_rel_err=0.88`; `wm_gain_pass=True` is the loose <2× gate). Median of 10 OP curves ≈ **×1.06**; one outlier −2.745 pulls the mean. DV ss/@H **×0.770 / ×0.830**. Decomp MV real→post **×0.994** (posterior faithful; compounding). DV real→post **×1.019**, post→1step ×0.756 (prior). DOB det_r **0.051** / R² −0.091 (amplitude dead because frozen g is biased; `dob_ground` live 0.022→0.012 — not P25 dead-grounding). Actor INVALID (`critic_r=NaN`; plots are P1 expert-BC).
- **ROOT:** (1) cap-on-first-storm froze an under-trained observer (P26 needed ~90 P1 iters). Seven skipped steps of iter 19 were correct; 93 applied steps still detonated recon — last-ok restore is KEEP, the **cap** is the bug. (2) Independent: `wm_fidelity_degradation` only runs in P2 while curriculum freezes `g`.
- **Do not:** revive relative Huber; TBPTT on gain-match asymptote; stack critic knobs; train P3 on this freeze.

### p31 (`run_p31_stormcont`, branch `cursor/p28`) — FIX: continue P1 after first skip-storm; suppress frozen-g fidelity ES
- **One attributed GPU change vs P30:** `skip_storm_p1_cap_after=2` (TrainConfig default). First P1 skip-storm restores last-ok, resets AdamW, **keeps original P1 budget** with extension closed; second storm still `_force_p1_cap_at` → P2. Storm-time GAIN_NOT_READY does **not** stick if P1 continues. Same commit, not the A/B: suppress `wm_fidelity_degradation` while `_dynamics_g_trainable` is false (one-line log). Env-free. Batched isolation/overshoot (`0a61872`) now in the GPU job.
- **Judge by:** `[skip-storm] … continuing P1 (storm 1/2, extension closed)` not instant P2; recon returns to last-ok; P1 proceeds toward ~90 iters; skip-restore SKIPPED; if GAIN-READY then first valid actor test of min-of-2 + freeze `return_scale`. Val MV ss/@H ~×1.0 ±0.1 (P26 ×0.97/×0.88). If second storm caps GAIN_NOT_READY: `[p3-skip]`. `[wm-fidelity-ES] suppressed: dynamics g frozen` in P2; DOB should train full P2, det_r not 0.05 after 40 iters.
- **LIVE (2026-08-26 22:55, tmux `mbrl2_p31`, GPU ~19.9 GB, process `4ffe11f`):** env-free confirmed. Survived P30's iter-18–25 window **without** a skip-storm. Healthy stretch iter 40–57 recon **0.003–0.005**; wrap blip 58–59 recovered; `wm_best` iter **90** `best_h=55/55` gain_fid=0.844 raw 6.465. Iter 75 gate `gain_not_ready` DC `[0.63, 1.48]` 1.48@MV spread_x5.1 — extended P1 (correct). Iter 85 `not_plateaued` extended again. **Iter 95 detonation (NOT a skip-storm):** recon 0.0035→**0.708**, gmatch 1.20, iso 2.98, `wm_grad_norm` **66.6**, skip 2→**3** (window=1 < 5). Iter 97 **CAPPED** GAIN_NOT_READY DC `[0.08, 1.31]` worst=`0.08@DV` — froze exploded `g` into P2 (restore SKIPPED). P2 iter **119** recon **0.46** `dobg` 0.15 (grounding live) `best_h=0/55`. `skip_invalid_p3` should skip P3. **ROOT (next GPU, not this process):** skip-storm last-ok never fired because skip count was 1; quality-gate CAPPED still froze detonated weights. HEAD restores last-ok at P1→P2 when recon > 5× last-ok best, re-probes gain. **No second GPU job.** HEAD also has batched gain-match FD + numpy random collect + vectorized DOB Kalman (not in `4ffe11f`).

### p31 GPU VERDICT (`run_p31_stormcont` EXIT @ 23:23, 151 iters) — storm_cap=2 KEEP; detonated CAPPED freeze of exploded g; P3 skipped
- **Liveness:** finished. GPU free. Wall ~5.4 h. `early_stop_reason=p3_skipped_invalid_observer`. `actor_experiment_valid=false`. `p1_last_ok_iter=94`. Process `4ffe11f` (no P1→P2 last-ok restore).
- **Launch KEEP:** env-free det+eager `storm_cap=2`. Survived P30 skip-storm window **without** a storm (continue-after-storm still untested). Healthy P1 to iter 94: recon **0.003–0.005**, `kl=0`, `jemb>0`, `wm_best` iter 90 `best_h=55/55` gain_fid=0.844. Iter 11 recon 0.71 recovered; wrap 58–59 recovered — do **not** lower skip threshold. Iter 75 `gain_not_ready` 1.48@MV extended P1 (correct).
- **Observer FAIL vs P26.** Iter 95 detonated (recon **0.708**, gmatch 1.20, gnorm **66.6**, skip 2→3). Iter 97 CAPPED froze exploded g (DC `[0.08, 1.31]` 0.08@DV). P2 recon **0.38–0.48**, `dob_ground` 0.33→0.033 live. Val MV ss/@H **×1.35 / ×1.34** (direction OK, ~35% high). DV ss/@H **×0.113 / ×0.108** (direction OK, dead). `wm_gain_pass=True` is the loose <2× MV gate (`wm_gain_healthy=True` at rel_err 0.35); `wm_dv_gain_healthy=False` (`wm_dv_ss_ratio_worst=0.113`; loose `wm_dv_gain_pass=True` because rel_err 0.887<1). Decomp MV real→post ×0.41 / DV ×0.32 (autoencoder — exploded freeze, not a DV-FF verdict). det_r **0.113** / det_R² −0.10 (phase/amplitude dead; pred_std 0.64 vs true 1.93). `wm_pred_converges_under_constant_action=1.0` but constant-action SS gain far too small. Vision: MV curves overlap-ish with overshoot; DV nearly flat; DOB does not track plateaus.
- **Actor INVALID.** `[p3-skip]` KEEP. `critic_r=NaN`. econ **−110 vs baseline −120** is expert-BC (not an actor win). Vision: pulse/chatter MV, oscillatory CV — BC policy on a dead observer, not P3.
- **ROOT:** skip-storm needs >5 skips; one applied step (gnorm 66 ≪ `wm_grad_skip_norm` 1e4) detonated recon; quality-gate CAPPED froze exploded `g`. Iter 11/58 blips recovered — restore only at freeze when recon still detonated. `skip_storm_p1_cap_after=2` KEEP (did not cap early). Frozen-g fidelity ES KEEP (P2 completed; budget-deferral at 140/150). Do not revive relative Huber; do not stack critic knobs; do not lower skip count.
- **P32:** P1→P2 last-ok restore when freeze recon > 5× last-ok best (already HEAD `1fd7640`). Env-free. Also picks up batched gain-match FD + numpy P1/P2 collect + vectorized DOB Kalman (same Huber/recurrence; not in `4ffe11f`).

### p32 (`run_p32_detfreeze`, branch `cursor/p28`) — FIX: restore last-ok at P1 freeze when recon is detonated
- **One attributed GPU change vs P31:** at P1→P2, if freeze-iter recon is not `_recon_still_healthy` vs last-ok best, restore `wm_last_ok`, reset AdamW, re-probe gain (`[p1→p2] detonated freeze restored wm_last_ok`). Skip-storm still needs >5 skips (wrap blips recover). `skip_invalid_p3` still skips P3 if the **restored** probe is GAIN_NOT_READY. Env-free. Same commit family also has batched gain-match FD / numpy collect / vectorized Kalman (objective-unchanged).
- **Judge by:** if P1 detonates at cap like P31: `[p1→p2] detonated freeze restored wm_last_ok (iter ~94)` not SKIPPED; P2 recon last-ok class (~0.003 not 0.45); restored gain-probe not 0.08@DV. Val MV ss/@H ~×1.0 ±0.1 (P26 ×0.97/×0.88), DV not ×0.11. If restore still GAIN_NOT_READY: `[p3-skip]`. If GAIN-READY: first valid actor test of min-of-2 + freeze `return_scale` (rscale near warmup, `rtgt>0.015`, econ vs baseline **and** vs P27 BC −59). If P1 never detonates: even better — healthy freeze. Watch `[resolved-cfg] latent=deterministic compile=eager storm_cap=2`.
- **LIVE (2026-08-27 03:06, tmux `mbrl2_p32`, GPU ~19.5 GB, pid 4139864):** env-free det+eager CONFIRMED. Process `a138e1f`. **No second GPU job.**
  - Healthy stretch to iter 51: recon **0.003–0.005**, `kl=0`, `jemb>0`, skip **0**, alive 1024. `wm_best` iter **50** EMA 5.799, `best_h=55/55`, gain_fid=0.898. Wrap/gnorm blips recovered — do **not** lower skip threshold.
  - **Skip-storm 1/2 @ iter 53 GPU-confirmed continue-after-storm:** iter 52 recon **0.486** gnorm **17.7** applied (skip 0). Iter 53 recon **0.596** gnorm **6.28e15** skip **71** → restored `wm_last_ok` iter 51; **extension closed at 753960**. Iter 54–75 recon last-ok class (**0.002–0.006**). Continue-after-storm KEEP.
  - **P1→P2 @ iter 75 CAPPED GAIN_NOT_READY** despite healthy freeze recon **0.0026**. Gain-probe `DCgain_ratio[0.71,1.12]` worst=**0.71@DV** (MV 1.12 in band `[0.8,1.3]`), `unbiased=False not_noisy=True`. Cap **0** steps — `_skip_storm_continue_p1` had zeroed `p1_gate_max_ext_steps`. `[p1→p2] WM warm-restore SKIPPED` (skip-storm already restored last-ok; freeze recon healthy so detonated-freeze last-ok did **not** fire). `[gate-budget]` had p1_ext_cap **175924** before the storm closed it. P31 at the same iter-75 gate **extended** (1.48@MV) and reached ~90–94.
  - P2 @ iter **106** env 951600: recon **0.0036**, `kl=0`, `jemb` 0.010, `dobg` 0.005–0.010 live, `g` frozen. `skip_invalid_p3=True` → expect `[p3-skip]`. Actor INVALID (GAIN_NOT_READY). Full val axes at EXIT.
  - **ROOT (next GPU, not this process):** closing the quality-gate extension on storm 1 starved the remaining P1 that P26/P31 used to pin DV DC-gain. Mechanism: skip-storm continue → `p1_gate_max_ext_steps=0` → first gate at original P1 budget cannot EXTEND → CAPPED GAIN_NOT_READY with healthy recon. Storm 2 must still `_force_p1_cap_at` (P28: cap-now with open ext re-opens exploding BPTT). HEAD: keep original P1 **and** extension on storm 1; TSSM `img_rollout`; DV-PRBS DRY.

### p32 GPU VERDICT (`run_p32_detfreeze` EXIT=0 @ 03:22, 129 iters) — continue-after-storm KEEP; CAPPED 0.71@DV (ext closed); P3 skipped
- **Liveness:** finished. GPU free. Wall ~3.5 h (23:54→03:22). `early_stop_reason=p3_skipped_invalid_observer`. `actor_experiment_valid=false`. `p1_gain_not_ready_capped=true`. `p1_detonated_freeze_restored=false`. Process `a138e1f` (keep-ext **not** in this job).
- **Launch KEEP:** env-free det+eager `storm_cap=2`. Train-start `latent=deterministic compile=eager skip_invalid_p3=True device=cuda bs=128`. `[resolved-cfg]` `restore_p2=False gain_match=1 dob_ground=2 isolation=1 ss_match=3 n_critics=2 rs_freeze=True`. Skip-storm **1/2 @iter 53** GPU-confirmed (last-ok iter 51, recon 0.596→0.003, original P1 kept). Freeze recon **0.0026** healthy — detonated-freeze last-ok correctly silent. `[p3-skip]` KEEP. Frozen-g fidelity ES: P2 probes still improved `wm_best` (iter 80/90/110) so ES did not trip.
- **Observer FAIL vs P26 on DV (MV close).** Val `final.pt`: MV ss/@H **×1.088 / ×1.098** (sign OK, ~9% high; live probe 1.12 in `[0.8,1.3]`). DV ss/@H **×0.675 / ×0.754** (sign OK, under; live 0.71@DV). `wm_gain_rel_err=0.088` MV HEALTHY. `wm_dv_ss_ratio_worst=0.675` (`wm_dv_gain_healthy=True` is the loose rel_err 0.325<0.35 — GAIN_READY band still failed). Decomp MV real→post **×0.997** compounding; DV real→post **×1.011** prior (1-step ×0.89). det_r **0.109** / det_R² 0.011 (amplitude dead; pred_std 0.19 vs true 1.93). P2 recon **0.003–0.004**, `dobg` 0.003→0.016 live, `g` frozen. `wm_pred_converges_under_constant_action=1.0`. Vision: MV curves overlap-ish (WM slightly high SS); DV WM flattens ~×0.68; DOB near-zero vs true load steps.
- **Actor INVALID.** `[p3-skip]` KEEP. `critic_r=NaN` (V collapsed). econ **−22.8 vs baseline −81** is expert-BC (beats_baseline is **not** an actor win; MV viol 20.7, pulse/chatter). Do not stack critic knobs. Do not revive relative Huber. Do not lower skip threshold.
- **ROOT:** skip-storm continue closed `p1_gate_max_ext_steps` → first P1→P2 gate at original budget CAPPED with 0 ext while recon was healthy. P31 same iter-75 **extended**. Keep-ext already HEAD (`8205f4b`). Storm 2 still `_force_p1_cap_at` (P28).

### p33 (`run_p33_keepext`, branch `cursor/p28`) — FIX: keep P1 quality-gate extension after first skip-storm
- **One attributed GPU change vs P32 process `a138e1f`:** `_skip_storm_continue_p1` keeps `p1_gate_max_ext_steps` (already HEAD). Storm 2 still `_force_p1_cap_at`. Env-free. Same job also picks up isolation TBPTT chunked `img_rollout`, Stage-1 prior-core skip, TSSM `img_rollout`, DV-PRBS DRY (objective-unchanged; not in P32).
- **Step 4 resolved-cfg vs P32:** TrainConfig knobs **identical** (env-free det+eager `storm_cap=2`). Diff is the continue code-path, not a `DREAMER_*` override. Watch `[skip-storm] … extension kept` (not `extension closed`); `[gate-budget]` p1_ext_cap must still be usable at the first P1→P2 gate.
- **Judge by:** if storm 1/2 fires: log `extension kept <p1_ext_cap> steps`. If GAIN_NOT_READY at original P1 budget (~iter 75): **EXTEND** not `CAPPED … (cap 0 steps reached)`. Val MV ss/@H ~×1.0 ±0.1 (P26 ×0.97/×0.88; P32 ×1.09), DV not ×0.67. If GAIN-READY: first valid actor test of min-of-2 + freeze `return_scale` (rscale near warmup, `rtgt>0.015`, econ vs baseline **and** vs P32 BC −23 / P27 BC −59). If still GAIN_NOT_READY: `[p3-skip]`. Watch `[resolved-cfg] latent=deterministic compile=eager storm_cap=2`.
- **LIVE (2026-08-27 12:50 UTC, tmux `mbrl2_p33`, GPU ~19.4 GB, pid 4146603, launch `1cfb530` / keep-ext `8205f4b`):** env-free det+eager CONFIRMED. `[resolved-cfg] latent=deterministic restore_p2=False gain_match=1 dob_ground=2 isolation=1 ss_match=3 n_critics=2 rs_freeze=True skip_invalid_p3=True storm_cap=2 compile=eager`. Train-start `device=cuda bs=128`. `[gate-budget]` p1_ext_cap **175924**. **No skip-storm** (past P32 storm window). Wrap blip 46–48 recovered. `wm_best` iter 40 then 90/100/110 (P2 probes; g frozen).
  - Iter 75 **EXTENDED** (`FAIL not_plateaued`, `wm_ema_best=5.206`): +75396 to 829356 (cap 929884).
  - Iter 85 **EXTENDED** (`GAIN_NOT_READY` DC `[0.68,1.05]` worst=**0.68@DV**, MV 1.05 in `[0.8,1.3]`, `unbiased=False not_noisy=True`): +75396 to 904752.
  - Iter 97 **CAPPED** (`GAIN_NOT_READY` DC `[0.68,1.11]` worst=**0.68@DV**, MV 1.11) after **cap 175924 steps reached**. Freeze recon **0.0034**, skip **0**, alive 1022. `[p1→p2] WM warm-restore SKIPPED`.
  - **Keep-ext KEEP as mechanism** (P32 CAPPED at iter 75 with 0 ext). **Keep-ext FALSIFIED as the DV×0.68 lever:** extra ~22 P1 iters (85→97) left DV at 0.68 — same as P32 val ×0.675. Next attributed GPU change is **not** more P1 time.
  - P2 through iter **110**: recon **0.0039**, `dobg` 0.005 live, gmatch/iso/ss 0 (`g` frozen). `skip_invalid_p3` → expect `[p3-skip]`. Actor INVALID (GAIN_NOT_READY). Full val axes at EXIT.
  - **Do not lower skip threshold.** Do not revive relative Huber. Do not stack critic knobs. Do not train P3 on this freeze. **No second GPU job.**
  - GPU-occupied HEAD (not in this pid): skip unused P1 `agent_finetune_loss` when `reward_scale_loss_p1=0` (paper default; was 100 unused MTP forwards/iter); last-ok snapshot `copy_` reuse. Stage-1 skip `apply_dob`; isolation/overshoot/held RSSM-interface; CV-std cache. Do not retune `wm_overshoot_max_starts` / `gain_match_max_starts`.

### p33 GPU VERDICT (`run_p33_keepext` EXIT=0 @ 08:15, 151 iters) — keep-ext KEEP as mechanism; FALSIFIED as DV lever; P3 skipped
- **Liveness:** finished. GPU free. Wall ~4.3 h (03:55→08:15). `early_stop_reason=p3_skipped_invalid_observer`. `actor_experiment_valid=false`. `p1_gain_not_ready_capped=true`. `skip_storm_p1_n=0`. `p1_detonated_freeze_restored=false`. Process `1cfb530` (HEAD skip-MTP/`copy_` **not** in this job).
- **Launch KEEP:** env-free det+eager `storm_cap=2`. Train-start `latent=deterministic compile=eager skip_invalid_p3=True device=cuda bs=128`. `[resolved-cfg]` `restore_p2=False gain_match=1 dob_ground=2 isolation=1 ss_match=3 n_critics=2 rs_freeze=True`. Keep-ext **used** (iter 75/85 EXTEND, iter 97 CAPPED after 175924). `[p3-skip]` KEEP. Frozen-g fidelity ES: P2 probes still improved `wm_best` so ES did not trip. P2 recon **0.003–0.004**, `dobg` 0.005–0.015 live.
- **Observer FAIL vs P26 on DV (MV close; same as P32).** Val `final.pt`: MV ss/@H **×1.083 / ×1.093** (sign OK; live 1.11 in band). DV ss/@H **×0.660 / ×0.770** (sign OK, under; live **0.68@DV**). `wm_gain_rel_err=0.083` MV HEALTHY. `wm_dv_ss_ratio_worst=0.660`. Decomp MV real→post **×1.022** compounding; DV real→post **×1.011** **prior** (1-step ×0.90). det_r **0.398** / det_R² 0.042 (better than P32 0.11; amplitude still dead: pred_std 0.087 vs true 1.93). `wm_pred_converges_under_constant_action=1.0`. Gain-match targets |MV| **2.82** vs |DV| **0.49** (~5.8×) — abs isolation/ss-match MSE weights MV ~33× more (SysID: subdominant input drowned).
- **Actor INVALID.** `[p3-skip]` KEEP. `critic_r=NaN`. econ **−20.7 vs baseline −108** is expert-BC (beats_baseline is **not** an actor win; MV viol 20.7, pulse/chatter). Do not stack critic knobs. Do not revive relative Huber. Do not lower skip threshold. Do not train P3 on GAIN_NOT_READY.
- **ROOT:** extra P1 cannot pin DV when abs isolation/ss-match (and abs Huber gain-match) under-weight the smaller |ΔCV|. Keep-ext stays so GAIN_NOT_READY can extend. Next: inverse-variance reweight of isolation/ss-match. P34 AM/HM form exploded; P35 is mean-1 `w·err`. Not relative Huber.

### p34 (`run_p34_isovarnorm`, branch `cursor/p28`) — FAIL: AM/HM inv-var exploded at init
- **One attributed GPU change vs P33:** `wm_isolation_var_norm=True` with ``mean(err/scale)*mean(scale)``.
- **ABORT iter 1:** iso **7088** ss **5357** (P33 iter 1 iso 1.69), skip **99**, recon 0.851. `[early-stop] grad_skip_storm: 99 skips in last 100`. Validation of exploded `final.pt` killed.
- **ROOT:** untrained WM has similar *abs* err on quiet (near-zero hold) and loud sequences. ``mean(err/s)*mean(s)`` has mean(w)=AM/HM(scale) ≫ 1 → isolation loss 4000×. Not relative Huber. Not GPU occupancy.
- **Fix (P35):** ``mean(w·err)`` with ``w∝1/scale``, ``mean(w)=1`` + abs floor 1e-4 (norm CV)². Identity on constant scale; O(mean(err)) at init; still equalizes relative-gain gradient when err∝scale.

### p35 (`run_p35_isowmean1`, branch `cursor/p28`) — LIVE P2: mean-1 per-seq inv-var starved DC-gain
- **One attributed GPU change vs P33 abs isolation (P34 form discarded):** same `wm_isolation_var_norm=True` default, weights mean-normalized. Gain-match stays abs Huber. Env-free. Launch `fcfe750`.
- **Liveness (2026-08-27 ~09:57):** still RUNNING P2. GPU ~17.7 GB. **No second GPU job.** Storm **1/2 @iter 9** (gnorm 8.8e13, skip 70, restored last-ok iter 3). Storm **2/2 @iter 16** (gnorm 1.8e12, skip 155, restored last-ok iter 15) **capped P1** at 344040 steps. `[gate p1->p2] CAPPED GAIN_NOT_READY worst=0.01@DV` MV 0.78 (neither in `[0.8,1.3]`). Detonated-freeze restored last-ok iter 15 (recon 0.341 > 5× 0.0199). P2 recon ~0.08, `dobg` 0.07–0.11 live, g frozen. Expect `[p3-skip]`. Actor INVALID.
- **Observer path vs P33 (P1 only ~16 iters).** Iter 1 iso **0.0008** skip 0 (P34 iso 7088 — mean-1 did not explode). Then iso stayed ~0.0006 vs P33 iter-1 **1.69**. `wm_isolation_var_scale_ratio` **~22000** throughout (wmax ~2.5). Gain-match **stuck at 1.21** (P33 dropped to 0.002 by iter 9). Cap-time DV ×0.01, MV ×0.78.
- **ROOT (SysID):** per-sequence `cv_real.pow(2)` hits the 1e-4 floor on near-zero stratified isolation holds. Mean-1 then parks almost all weight on quiet sequences → loud MV/DV DC-gain teacher starved → full-BPTT gain-match never pins G → skip-storm. `scale_ratio` 22000 is quiet-vs-loud, not MV-vs-DV (~33 from \|tgt\| 2.82 vs 0.49). Median floor in `_invvar_reweight` is useless when the median IS the quiet floor.
- **HEAD (not in this pid):** isolation scale is **per isolated input** (identified \|tgt\|², else scatter-mean CV² of that input). Same mean-1 `w·err`. JSONL `wm_isolation_var_tgt_scale` + `scale_ratio` should be ~33 not 22000. CPU smoke identity. Next GPU after P35 EXIT: env-free P36. Do not revive relative Huber. Do not lower skip threshold. Do not stack critic knobs. Do not train P3 on this freeze.

### p35 GPU VERDICT (`run_p35_isowmean1` EXIT=0 @ 10:24, 70 iters) — per-seq `|CV|²` SUPERSEDED; mean-1 KEEP; P3 skipped
- **Liveness:** finished. GPU free. Wall ~1.2 h (09:12→10:24). `early_stop_reason=p3_skipped_invalid_observer`. `actor_experiment_valid=false`. `skip_storm_p1_n=2`. `p1_gain_not_ready_capped=true`. `p1_detonated_freeze_restored=true` (iter 15, recon 0.341 vs best 0.0199). Process `fcfe750`.
- **Launch KEEP (mechanisms, not the scale source):** env-free det+eager `storm_cap=2`. Train-start `latent=deterministic compile=eager skip_invalid_p3=True device=cuda bs=128`. `[resolved-cfg]` `restore_p2=False gain_match=1 dob_ground=2 isolation=1 ss_match=3 iso_varnorm=True n_critics=2 rs_freeze=True`. Mean-1 `w·err` did **not** explode (iter 1 iso 0.0008 skip 0 vs P34 7088). Storm 1/2 KEEP (last-ok iter 3). Storm 2/2 `_force_p1_cap_at` KEEP. Detonated-freeze last-ok KEEP. `[p3-skip]` KEEP.
- **Observer FAIL vs P26 and vs P33 (worse).** Val `final.pt`: MV ss/@H **×0.936 / ×0.756** (sign OK; live cap 0.78). DV ss/@H **×0.013 / ×0.016** (sign OK, dead; live **0.01@DV**). `wm_gain_rel_err=0.064` MV HEALTHY (gain-blind MV-only hid DV). `wm_dv_ss_ratio_worst=0.013` `wm_dv_gain_healthy=False`. Decomp MV real→post **×0.912** compounding; DV real→post **×0.916** **prior** (1-step ×0.375). det_r **0.275** / det_R² 0.075 (pred_std 0.32 vs true 1.93). P2 recon **0.056–0.093** (last-ok class, not P33 0.003). `wm_pred_converges_under_constant_action=1.0`. Vision: MV curves overlap (WM slightly low SS); DV WM flat ~0 vs real +0.18; DOB noisy around 0 vs true load steps.
- **Actor INVALID.** `[p3-skip]` KEEP. `critic_r=NaN` (V≡0). econ **−43 vs baseline −141** is expert-BC (beats_baseline is **not** an actor win; CV viol 11.6, MV viol 5.6, pulse/chatter). Do not stack critic knobs. Do not revive relative Huber. Do not lower skip threshold. Do not train P3 on GAIN_NOT_READY.
- **ROOT:** per-sequence `|CV|²` quiet-hold floor (scale_ratio ~22000) starved isolation/ss-match and left gain-match stuck at 1.21 → skip-storm 2/2 at iter 16 froze an un-identified G. Mean-1 formula KEEP (P34 AM/HM exploded; this did not). Scale source SUPERSEDED → per-input identified `|tgt|²` (HEAD `596b78e`).

### p36 GPU VERDICT (`run_p36_isoinpscale` EXIT=0 @ 11:44, 61 iters) — per-input `|G|²` fired; inv-var DISCARDED; P3 skipped
- **Liveness:** finished. GPU free. Wall ~0.9 h (10:52→11:44). `early_stop_reason=p3_skipped_invalid_observer`. `actor_experiment_valid=false`. `skip_storm_p1_n=2`. `p1_gain_not_ready_capped=true`. `p1_detonated_freeze_restored=true` (iter 6, recon 0.242 vs best 0.0207). Process `596b78e`.
- **Launch KEEP (scale source, not the reweight):** env-free det+eager `storm_cap=2`. `[resolved-cfg] iso_varnorm=True`. Per-input `|G|²` **did fire**: `tgt_scale=1`, `scale_ratio` 20 then **33.35** (not P35 22000). Iter-1 iso **0.125** (P35 0.0008, P33 **1.69**). Quiet-hold SUPERSEDE KEEP. Storm 1/2 KEEP (last-ok iter 3). Storm 2/2 `_force_p1_cap_at` KEEP. Detonated-freeze last-ok KEEP. `[p3-skip]` KEEP.
- **Observer FAIL vs P26 and vs P33 (P35-class freeze).** Storm **1/2 @iter 4** (gnorm **2.5e6**, skip 15, recon 0.82). Recovered iter 5–6 recon 0.02. Storm **2/2 @iter 7** (gnorm **4.2e18**, skip 91, recon 0.24) CAPPED at 286700. Cap-time GAIN_NOT_READY **0.00@DV** (MV 0.98 in `[0.8,1.3]`). Gain-match **stuck at 1.21** (P33 fell to 0.002 by iter 9). Val `final.pt`: MV ss/@H **×0.913 / ×0.945** (sign OK). DV ss/@H **×0.004 / ×0.004** (flat; live 0.00@DV). `wm_gain_rel_err=0.087` MV HEALTHY (gain-blind hid DV). `wm_dv_ss_ratio_worst=0.004` `wm_dv_gain_healthy=False`. Decomp MV/DV real→post ×0.86 **autoencoder** (undertrained freeze). det_r **0.076** / R² −0.036 (pred_std 0.31 vs true 1.93). P2 recon 0.063→0.028, `dob_ground` 0.12→0.020 live, `g` frozen. Vision: MV curves overlap; DV WM flat ~0 vs real +0.18; DOB oscillates around 0 vs true load steps.
- **Actor INVALID.** `[p3-skip]` KEEP. `critic_r=NaN`. econ **−29 vs baseline −120** is expert-BC (beats_baseline is **not** an actor win; CV viol 11.9, MV viol 4.0, pulse/chatter near a limit). Do not stack critic knobs. Do not revive relative Huber. Do not lower skip threshold. Do not train P3 on GAIN_NOT_READY.
- **ROOT:** inverse-variance isolation is a **relative-gain reweight**. `w∝1/|G|²` with ratio 33 = 33× DV isolation gradient — same class as P27 relative Huber (skip-storm). Isolation teacher 13× weaker than P33 abs (iso 0.125 vs 1.69) so gain-match never pins G, then a huge-grad step detonates P1 at iter 4/7. Per-input scale fixed P35's quiet-hold (ratio 33 not 22000) but did **not** restore P33 P1 health. **REVERT default `wm_isolation_var_norm=False`** (abs, P33). Do not try another isolation reweight formula.

### p37 GPU EXIT (`run_p37_isoabs` EXIT=0 @ 16:16, 151 iters) — abs isolation = P33 pin; extra-P1 detonated; `[p3-skip]`
- **Liveness:** finished. GPU free. Wall ~4.3 h (11:56→16:16). `early_stop_reason=p3_skipped_invalid_observer`. `actor_experiment_valid=false`. `skip_storm_p1_n=0`. `p1_gain_not_ready_capped=true`. `p1_detonated_freeze_restored=true` (iter 87). Launch `4d349fb` (unscaled abs; HEAD dcv_match **not** in pid).
- **Launch KEEP:** env-free det+eager `storm_cap=2`. `[resolved-cfg] iso_varnorm=False` gain_match=1 isolation=1 ss_match=3. tgt MV −2.81 DV 0.487. device=cuda bs=128. No leftover `DREAMER_*`. Keep-ext used (iter 75/85). Storm-cap unused (skip 0 through P32 window). Detonated-freeze last-ok KEEP. `[p3-skip]` KEEP.
- **P1 healthy through iter 87 (P33 class).** Iter 53 recon **0.0047 skip 0**. Iter-1 iso **1.33**. gmatch **1.20 → 0.0008 by iter 17**. Iter 19 wrap blip recovered (do **not** lower skip). Iter 75 EXTEND **0.68@DV** (MV 0.99). Iter 85 EXTEND **0.72@DV** (MV 0.98). Extra P1 then **silent detonation** iter 88 (recon 0.848, gnorm **62.4**, skip **0**). Iter 97 CAPPED exploded probe **0.00@DV / 0.59@MV**. Restored last-ok iter **87**: live `DCgain_ratio[0.71,1.00]` worst=**0.71@DV**.
- **Observer FAIL vs P26 on DV (MV pinned; P33-class).** Val `final.pt`: MV ss/@H **×0.981 / ×1.005** (sign OK; better than P33 ×1.08 and P26 @H ×0.88). DV ss/@H **×0.690 / ×0.783** (sign OK, under; live 0.71@DV; P33 ×0.660/×0.770). `wm_gain_rel_err=0.019` MV HEALTHY. `wm_dv_ss_ratio_worst=0.690` (`wm_dv_gain_healthy=True` is the loose rel_err 0.31<0.35 — GAIN_READY still failed). Decomp MV real→post **×1.008** compounding (open-loop ×0.874). DV real→post **×0.969** **faithful** (1-step ×0.894). det_r **0.370** / R² 0.038 (P33 0.398; amp dead: pred_std **0.094** vs true 1.93). P2 recon **0.003**, `dobg` 0.004→0.012 live, `g` frozen. `wm_pred_converges_under_constant_action=1.0`. Vision: MV curves overlap (×0.98); DV WM settles ~×0.69 vs real +0.18; DOB near-zero vs true load steps.
- **Actor INVALID.** `[p3-skip]` KEEP. `critic_r=NaN` (V≡0). econ **−24 vs baseline −106** is expert-BC (beats_baseline is **not** an actor win; CV viol 2.73, MV viol 18.0, pulse/chatter near a limit). Do not stack critic knobs. Do not revive relative Huber. Do not lower skip threshold. Do not train P3 on GAIN_NOT_READY. Do not try another isolation reweight. Extra P1 FALSIFIED as DV lever (again).
- **ROOT:** abs isolation/ss-match on isomorphic |Δu| drowns subdominant |ΔCV| (tgt |MV| 2.82 vs |DV| 0.49). Completes P1 (no skip-storm) and pins MV (×0.98) but DV plateaus ~×0.66–0.71 (P32/P33/P37). Extra P1 0.68→0.72 then detonated; last-ok 0.71. Inv-var (P34–P36) skip-stormed worse. **Next GPU launched:** env-free P38 `run_p38_isodcv` — isolation |ΔCV| excitation (`wm_isolation_dcv_match`). Not a loss reweight.
- **HEAD (not in this pid, P38 recipe):** inv-var A/B **REMOVED**. Abs MSE + **`wm_isolation_dcv_match=True`**: settle `Δu_i ∝ 1/|G_i|` (test_sim scale MV ~0.289 DV ~1.67). jsonl `p1_last_ok_iter` + `wm_isolation_loss`. Pre-iso resolve = scales only; post-seed always re-resolves Huber. Opt out `DREAMER_WM_ISOLATION_DCV_MATCH=0`.

### p38 GPU EXIT (`run_p38_isodcv` EXIT=0 @ 19:28, 102 iters) — match-at-`g_min` (no floor) FALSIFIED; `[p3-skip]`
- **Liveness:** finished. GPU free. Wall ~2.5 h (17:00→19:28). `early_stop_reason=p3_skipped_invalid_observer`. `actor_experiment_valid=false`. `skip_storm_p1_n=2`. `p1_gain_not_ready_capped=true`. `p1_detonated_freeze_restored=true` (iter 47). Launch `42bc7c2` (no floor; HEAD floor **not** in pid).
- **Launch KEEP:** env-free det+eager `storm_cap=2`. `[resolved-cfg] iso_dcv=True mv=['0.317'] dv=['1.67']` isolation=1 ss_match=3 gain_match=1. Applied edge |Δu| MV **0.19** (P37 0.60), DV **1.0** (P37 0.60). device=cuda bs=128. No leftover `DREAMER_*`.
- **P1 FAIL (P36 class).** gmatch **stuck 1.30** entire P1 (P37 →0.0008 by iter 17). iso/ss ~0.14/0.11. Storm **1/2 @iter 45** last-ok **13**. Storm **2/2 @iter 48** CAPPED 570960. Cap-time GAIN_NOT_READY **0.01@DV** (MV 1.12). Detonated-freeze last-ok iter **47** (gmatch still 1.30). P2 recon ~0.07–0.11 `g` frozen. `[p3-skip]` KEEP.
- **Observer FAIL vs P26/P37 (DV dead; MV overgain on untrained freeze).** Val `final.pt`: MV ss/@H **×1.253 / ×1.293** (sign OK, too large; freeze MV 1.12). DV ss/@H **×0.007 / ×0.007** (dead; P37 ×0.690/×0.783). `wm_gain_rel_err=0.253` MV HEALTHY (misleading). `wm_dv_ss_ratio_worst=0.007` (`wm_dv_gain_pass=true` is the loose rel_err<1 gate — GAIN_READY failed). Decomp MV/DV real→post ~×0.85/×0.82 autoencoder; open-loop ~0. Vision: MV too large/too fast; DV flat ~0. det_r **0.428** / R² 0.10 (amp dead: pred_std **0.158** vs true 1.93). `wm_pred_converges_under_constant_action=1.0` (wrong attractor). `wm_next_state_r=0.30` FAIL.
- **Actor INVALID.** `[p3-skip]` KEEP. `critic_r=NaN` (V≡0). econ **−35 vs baseline −108** is expert-BC (beats_baseline is **not** an actor win; CV viol 10.9, MV viol 8.5). Do not stack critic knobs. Do not revive relative Huber. Do not re-add inv-var. Do not lower skip threshold. Do not train P3 on GAIN_NOT_READY.
- **ROOT (SysID / signal):** match-at-`g_min` equalized |ΔCV| by **shrinking** the strong-|G| isolation step (MV |Δu| 0.19 vs P37 0.60). Isolation teacher too weak to pin G → gain-match Huber stuck at untrained |tgt| (~1.30). Same class as P36 weak teacher, different lever (data amplitude). Boosting DV to cube did **not** recover DV when MV SNR was starved. Val confirms CAPPED RCA.
- **Next GPU launched:** env-free P39 `run_p39_isodcvfloor` — HEAD scale **floor 1.0** (already default). Watch `[seed] dcv_match min_scale=1` edge_du MV=0.60 DV=1.0; gmatch falling like P37; skip 0 through iter 53.

### p39 GPU EXIT (`run_p39_isodcvfloor` EXIT=0 @ 00:28, 151 iters) — cube-boost FALSIFIED; floor KEEP as P1 form; `[p3-skip]`
- **Liveness:** finished. GPU free. Wall ~4.2 h (20:13→00:28). `early_stop_reason=p3_skipped_invalid_observer`. `actor_experiment_valid=false`. `skip_storm_p1_n=1`. `p1_gain_not_ready_capped=true`. `p1_detonated_freeze_restored=false`. Process `92c7662` (HEAD diag-off / span-audit **not** in pid).
- **Launch KEEP:** env-free det+eager `storm_cap=2`. `[resolved-cfg] iso_dcv=True min_scale=1.0 mv=['1'] dv=['1.67'] edge_du_mv=['0.6'] edge_du_dv=['1']` isolation=1 ss_match=3 gain_match=1. Applied edge |Δu| MV **0.60** / DV **1.0**. device=cuda bs=128. No leftover `DREAMER_*`. Floor KEEP as P1 form vs P38 (gmatch pinned 0.0002; extra-P1 stable). Storm **1/2 @iter 53** recovered last-ok 51, **extension kept**. Iter 75/85 EXTEND 0.71 then 0.69@DV. Extra-P1 **did not detonate**. Iter 97 CAPPED **0.70@DV** (MV 1.00) after 175924. Freeze recon **0.0031**. `[p3-skip]` KEEP.
- **Observer FAIL vs P26 on DV (MV pinned; P37-class).** Val `final.pt`: MV ss/@H **×0.954 / ×0.954** (sign OK; live 1.00 in band). DV ss/@H **×0.679 / ×0.785** (sign OK, under; live **0.70@DV**; P37 ×0.690/×0.783). `wm_gain_rel_err=0.046` MV HEALTHY. `wm_dv_ss_ratio_worst=0.679` (`wm_dv_gain_healthy=True` is the loose rel_err 0.32<0.35 — GAIN_READY still failed). Decomp MV real→post **×0.972** compounding (open-loop ×0.846). DV real→post **×1.002** **faithful** (1-step ×0.936). det_r **0.326** / R² 0.056 (amp dead: pred_std **0.156** vs true 1.93; P26 0.68 / 0.80). P2 recon **0.0025**, `dobg` live, `g` frozen. `wm_pred_converges_under_constant_action=1.0`. Vision: MV curves overlap (×0.95); DV WM settles ~×0.68 vs real +0.18; DOB near-zero vs true load steps.
- **Actor INVALID.** `[p3-skip]` KEEP. `critic_r=NaN`. econ **−44 vs baseline −91** is expert-BC (beats_baseline is **not** an actor win; CV viol 9.2, MV viol 20.9, pulse/chatter). Do not stack critic knobs. Do not revive relative Huber. Do not re-add inv-var. Do not lower skip threshold. Do not train P3 on GAIN_NOT_READY. Do not try another isolation reweight. Extra P1 FALSIFIED. Cube-boost FALSIFIED as DV pin (`|G_max|/|G_min|≈5.8 > 1/op_band≈1.67`).
- **ROOT (SysID, structural):** abs isolation/ss-match on mixed MV/DV drowns the subdominant |ΔCV|. P26 *also* ran isolation auto-enable (jsonl `wm_input_isolation_loss` 1.44 at iter 1; `run_plan` isolation=0 was the **pre-rewrite dump**). P28+ whole-episode holds + first-class `ss_match=3` made that teacher louder; DV ss fell P26 ×0.87 → P32–P39 ×0.66–0.70. Isolation cannot equalize |ΔCV| without P38 shrink or cube overflow. Floor KEEP as P1 form when isolation is opted in.
- **Correction:** do not treat P26 as "gain-match-only". Next GPU is the first env-free run with isolation **actually off**.

### p40 (`run_p40_gmatchonly`, branch `cursor/p28`) — FIX: stop auto-enabling isolation/ss-match
- **One attributed GPU change vs P39:** env-free isolation=0 / ss_match=0. Gain-match (identified G, per-input FD Huber) is the DC supervisor. Isolation stays opt-in (`DREAMER_WM_INPUT_ISOLATION_COEF=1` then auto-fills len/settle). Skip isolation_buf + isolated-settle seed when the teacher is off.
- **Not:** isolation reweight, extra-P1, no-floor, leftover `DREAMER_*`, diag probes, P3 on GAIN_NOT_READY.
- **Step 4 resolved-cfg vs P39:** `isolation=0 ss_match=0` (was 1 / 3). `gain_match=1 dob_ground=2` unchanged. No `[seed] isolated-settle`. No `[isolation-buf]`.
- **Liveness:** finished. GPU free. Wall ~3.6 h (01:14→04:53). 158 iters. `early_stop_reason=p3_skipped_invalid_observer`. `actor_experiment_valid=false`. `p1_gain_not_ready_capped=true`. `p1_detonated_freeze_restored=false`. `p1_last_ok_iter=104`. Process `73a5116` (recent-floor **not** in this job).
- **Launch KEEP:** env-free det+eager `storm_cap=2`. `[resolved-cfg] isolation=0 ss_match=0` gain_match=1 dob_ground=2. device=cuda bs=128. No leftover `DREAMER_*`. Storm **1/2 @iter 65** restored last-ok 63, **extension kept**. Isolation-off did **not** eliminate storms or extra-P1 detonation. Iter 75 not the quality gate (steps 696k < p1 754k). Iter **82** EXTEND `not_plateaued` ema_best=6.541 (iter-10 warmup; **gain-probe skipped**). Extra-P1 silent detonation **iter 84** (recon 0.482, gnorm 2.51, skip still 7). last_ok **overwrote 83→104** at iter 98 (recon 0.0068 < 5×best 0.0076). Iter **104 CAPPED GAIN_NOT_READY 0.75@DV** (MV **1.01**); freeze recon 0.0045 within 5× so detonated-freeze did **not** restore 83. P2 recon ~0.006, `dobg` live 0.007–0.013, `g` frozen. `[p3-skip]` KEEP.
- **Observer: MV pinned; isolation-off FALSIFIED as DV pin.** Val `final.pt`: MV ss/@H **×0.995 / ×0.964** (`wm_gain_rel_err=0.005` HEALTHY; live MV 1.01). DV ss/@H **×0.723 / ×0.743** (live **0.75@DV**; P39 ×0.679/×0.785; P26 ×0.87/×1.00). `wm_dv_ss_ratio_worst=0.723`. Decomp MV real→post **×0.946** compounding (open-loop ×0.851). DV real→post **×0.998** **faithful** (1-step ×0.918). det_r **0.490** / R² 0.152 (amp dead: pred_std **0.246** vs true 1.93; P39 0.326 / 0.156; P26 0.68 / 0.80). `wm_pred_converges_under_constant_action=1.0`. Isolation-off did **not** recover P26 DV. Extra-P1 last_ok overwrite confounded the freeze (recovered extra-P1 weights, not pre-blip 83).
- **Actor INVALID.** `[p3-skip]` KEEP. `critic_r=NaN`. econ **−78 vs baseline −76** is expert-BC (worse than open-loop; beats_baseline is **not** an actor result). Do not stack critic knobs. Do not revive relative Huber. Do not re-add inv-var. Do not lower skip threshold. Do not train P3 on GAIN_NOT_READY. Do not try another isolation reweight.
- **ROOT:** abs Huber gain-match still under-weights the smaller DV |ΔCV| when isolation is off; isolation was not the sole DV regression. First-gate `not_plateaued` on iter-10 `wm_ema_best=6.541` skipped the gain-probe on healthy original-budget weights, then extra-P1 detonated. Next GPU: recent-floor P1 gate (already HEAD) so the original-budget gate can run a gain-probe and possibly freeze **before** extra-P1.

### p41 (`run_p41_recentfloor`, branch `cursor/p28`) — FIX: P1 gate recent-floor (not warmup `wm_ema_best`)
- **One attributed GPU change vs P40 process `73a5116`:** P1→P2 fidelity floor is the **recent** EMA max (`_p1_fidelity_local_plateau`), not return to all-time `wm_score_ema_best`. Gain-probe can run at the original P1 budget (~iter 82). Isolation-off stays (env-free KEEP; FALSIFIED as DV pin).
- **Not:** isolation reweight, last_ok-overwrite change, leftover `DREAMER_*`, P3 on GAIN_NOT_READY. `last_only` / jsonl zeros already HEAD from P40-live (claimed objective-unchanged; in this pid).
- **Liveness:** finished. GPU free. tmux `mbrl2_p41` DEAD, pid **4185725** DEAD. Launch `acb8a7b`. 158 iters. `early_stop_reason=p3_skipped_invalid_observer`. `actor_experiment_valid=false`. `p1_gain_not_ready_capped=true`. `p1_detonated_freeze_restored=false`. `p1_last_ok_iter=104`. P2-end recon **0.0029** gmatch **0** dobg **0.012 live**.
- **Storm 1/2 @iter 41** (`[skip-storm] … last_ok iter 38, extension kept`). Iter 39–40 silent detonation: recon **0.398→0.590**, gmatch **1.12**, **gnorm 0.45/0.34**, skip **0**. Iter 41 skip 93, gnorm 1.47e7, restored 38. Recovered recon 0.002 by 42.
- **P40 last-ok overwrite already happened in original P1:** iter 57 recon **0.0887** / gnorm **1.99** / skip 0 (42× best ~0.0021); last_ok **overwrote 56→64** at recon **0.0098 < 5×**. Iter 66 wrap 0.096; iter 74 silent 0.281 skip 94. last_ok then advanced through extra-P1 to **104**. Freeze recon **0.0028** healthy → detonated-freeze does **not** restore 56 (same hole as P40 83→104).
- **First gate iter 82 (original budget 753960) GPU-confirmed:** recent-floor **KEEP as mechanism**. Printed **`[gate p1->p2] gain-probe`** (P40 `not_plateaued` / **no probe**). **FAIL GAIN_NOT_READY worst=0.76@DV** (MV **1.01**). recon **0.0058**. **EXTEND** to 829356.
- **Extra-P1 through 104:** **did not detonate** at P40 iter 84 (recon **0.0062** / gnorm **1.04**) nor P37 iter 88 (recon **0.013** / gnorm **0.57**). Iter 94 gnorm **34.8** with recon **0.0036**; recovered.
- **Second gate iter 94:** **gain-probe**. **FAIL 0.76@DV** (MV **1.03**). **EXTEND** to 904752. Extra P1 did not move DV.
- **CAPPED iter 104 (cap 175924):** cap-time **gain-probe**. **GAIN_NOT_READY worst=0.74@DV** (MV **1.02**; *slightly worse* than 0.76). Freeze recon **0.0028**. Warm-restore **SKIPPED**. STAGE 2 g FROZEN. Extra P1 FALSIFIED as DV lever (0.76→0.76→**0.74**).
- **P2→P3 iter 158:** PASS reward_mtp; **`[p3-skip]`**. Actor INVALID (GAIN_NOT_READY).
- **Observer: MV pinned; recent-floor FALSIFIED as DV pin.** Val `final.pt`: MV ss/@H **×0.986 / ×0.973** (`wm_gain_rel_err=0.014` HEALTHY). DV ss/@H **×0.700 / ×0.775** (live **0.74@DV**; P40 ×0.723/×0.743; P26 ×0.87/×1.00). `wm_dv_ss_ratio_worst=0.700` (loose `wm_dv_gain_healthy=True` — not a DV pin). Decomp MV real→post **×0.986** compounding. DV real→post **×0.976** **faithful**. det_r **0.079** / R² **0.006** (amp dead: pred_std **0.196** vs true 1.93; P40 **0.490** / 0.152). Correlation-only is not a pass. det_r collapsed vs P40 because freeze is post-overwrite last_ok **104**, not locked pre-spike **56**.
- **Actor INVALID.** `[p3-skip]` KEEP. `critic_r=+nan` (P3 never ran). econ **−34.7 vs baseline −102** is expert-BC. `cum_raw` **−46742**. Do not stack critic knobs. Do not revive relative Huber. Do not re-add inv-var. Do not lower skip threshold. Do not train P3 on GAIN_NOT_READY. Do not try another isolation reweight.
- **ROOT:** recent-floor unblocked the original-budget gain-probe (KEEP as mechanism) but extra P1 cannot pin DV and the freeze still used recovered last_ok 104. Isolation-off stays env-free default (P40). **P42 EXIT later FALSIFIED the "det_r collapsed because freeze was 104 not 56" story** (freeze-66 det_r 0.12, not P40 0.49). Next GPU after P41 was last-ok lock (`skip_storm_last_ok_lock_ratio=20`).

### p42 (`run_p42_lastoklock`, branch `cursor/p28`) — FIX: last-ok lock after silent recon spike
- **One attributed GPU change vs P41 process `acb8a7b`:** last-ok **locks** when recon > `skip_storm_last_ok_lock_ratio=20` × best (recon-only call site). Recovered recon must not overwrite the pre-spike snapshot. Skip-storm restore still unlocks. Wrap ~5–10× must not lock. Recent-floor + isolation-off stay (P41/P40 KEEP).
- **Not:** isolation reweight, extra-P1, leftover `DREAMER_*`, P3 on GAIN_NOT_READY.
- **Step 4 resolved-cfg vs P41:** identical banner `latent=deterministic restore_p2=False gain_match=1 dob_ground=2 isolation=0 ss_match=0 iso_dcv=off n_critics=2 rs_freeze=True skip_invalid_p3=True storm_cap=2 compile=eager`. Attributed delta: `skip_storm_last_ok_lock_ratio=20` now in `run_plan` (P41 plan `<MISSING>` — field not in that pid).
- **Liveness:** finished. GPU free. tmux `mbrl2_p42` DEAD, pid **4192010** DEAD. Launch `72f7b48`. 158 iters. `early_stop_reason=p3_skipped_invalid_observer`. `actor_experiment_valid=false`. `p1_gain_not_ready_capped=true`. `p1_detonated_freeze_restored=true`. `p1_last_ok_iter=66`. `p1_last_ok_locked=true`. P2-end recon **0.0022** gmatch **0** dobg **0.010 live**. Isolation=0; no leftover `DREAMER_*`.
- **Lock 20× fire GPU-confirmed KEEP as mechanism.** `[wm-last-ok] locked iter 66` on spike iter **67** recon **0.3066** (170× best 0.0018) skip **1**. last_ok **stayed 66** through extra-P1 and freeze. Wrap 5–10× untested (jump was 170×). Skip-storm unlock untested (skip stayed 1; no storm).
- **<5× overwrite-prevention untested.** Extra-P1 recon never dropped below 5× best (min **0.0255** = 14× at iter 104). last_ok would have stayed 66 even without the lock. Freeze-iter recon 14×>5× so old detonated-freeze would have restored 66 anyway. Lock's restore-even-if-healthy path untested.
- **Gates 82 / 94 / 104 all `gain-probe`** (recent-floor KEEP). 82 FAIL **0.01@DV** (MV 2.01, live detonated recon 0.92). 94 FAIL **3.26@MV** (DV 0.87 noisy). **104 PASS on LIVE extra-P1** worst=**0.81@DV** (MV 1.28) — declined.
- **Freeze GPU-confirmed last_ok 66:** `[p1→p2] detonated freeze restored wm_last_ok (iter 66)`. Re-probe **GAIN_NOT_READY 0.75@DV** (MV **1.19**). Warm-restore SKIPPED. First P2 recon **0.0025** (last_ok class). `wm_last_ok.pt` written at freeze.
- **P2→P3 iter 158:** PASS reward_mtp 0.89; **`[p3-skip]`**. Actor INVALID (GAIN_NOT_READY freeze).
- **Observer: MV overgain; lock FALSIFIED as DV pin and as P41 det_r-collapse fix.** Val `final.pt`: MV ss/@H **×1.179 / ×1.161** (`wm_gain_rel_err=0.179` HEALTHY at the p106 0.186 line — misleading; freeze MV 1.19). DV ss/@H **×0.737 / ×0.804** (live **0.75@DV**; P41 ×0.700/×0.775; P40 ×0.723/×0.743; P26 ×0.87/×1.00). `wm_dv_ss_ratio_worst=0.737` (loose `wm_dv_gain_healthy=True` — not a DV pin). Decomp MV real→post **×0.986** compounding (open-loop ×0.843). DV real→post **×1.003** **faithful**. det_r **0.124** / R² **0.013** (amp dead: pred_std **0.164** vs true 1.93; P41 **0.079** / 0.006; P40 **0.490** / 0.152; P26 **0.68**). Freeze-66 did **not** recover P40 det_r. P41 RCA that det_r collapsed because freeze was overwritten last_ok **104** (not pre-spike **56**) is **not supported** — P40 also froze last_ok **104** with det_r 0.490; P42 froze **66** and stayed P41-class (0.12).
- **Actor INVALID.** `[p3-skip]` KEEP. `critic_r=+nan` (P3 never ran). econ **−48.6 vs baseline −97.6** is expert-BC. `cum_raw` **−45229**. Do not stack critic knobs. Do not revive relative Huber. Do not re-add inv-var. Do not lower skip threshold. Do not train P3 on GAIN_NOT_READY. Do not try another isolation reweight.
- **ROOT:** lock KEEP as a 20× fire (held last_ok 66; freeze restored 66). Extra-P1 never reached <5× so overwrite-prevention was not the A/B. Freeze of pre-spike 66 is GAIN_NOT_READY 0.75@DV with MV **overgain ×1.18** (worse than P40/P41 ~×1.00) and det_r **0.12** (not P40 0.49). Isolation-off + recent-floor stay. Remaining DV drowning is abs Huber gain-match |tgt| MV **2.62** vs DV **0.51**. Persist-on-lock is crash-safety already HEAD — not an observer A/B.

### p43 (`run_p43_gmatchperbeta`, branch `cursor/p28`) — EXIT: per-input Huber β = |tgt_ij|
- **One attributed GPU change vs P42 process `72f7b48`:** gain-match Huber β is **per-element `|tgt_ij|`** (`gain_match_huber_per_input=True`). L1 saturation stays ±1 (not P27 relative Huber, which divides the residual by |tgt| before Huber → L1 grad 1/|tgt|). Isolation-off + last-ok lock + recent-floor stay.
- **Not:** relative Huber, isolation reweight, cube-boost, persist-on-lock A/B, leftover `DREAMER_*`, P3 on GAIN_NOT_READY, median-β (median of {2.62, 0.51} ≈ 1.56 ≈ scalar β=1).
- **Step 4 resolved-cfg vs P42:** identical banner except `huber_per_in=True`. `latent=deterministic restore_p2=False gain_match=1 dob_ground=2 isolation=0 ss_match=0 iso_dcv=off n_critics=2 rs_freeze=True skip_invalid_p3=True storm_cap=2 lock=20 compile=eager`.
- **Liveness:** finished. GPU free. tmux `mbrl2_p43` DEAD, pid **4465** DEAD. Launch `51b0f45`. 158 iters. Wall ~3.6 h (13:11→16:50). `early_stop_reason=p3_skipped_invalid_observer`. `actor_experiment_valid=false`. `p1_gain_not_ready_capped=true`. `p1_detonated_freeze_restored=true`. `p1_last_ok_iter=94`. `p1_last_ok_locked=true`. P2-end recon **0.0017** gmatch **0** (g frozen) dobg **0.011 rising** d_abs **0.007**. Isolation=0; `huber_per_in=True`; no leftover `DREAMER_*`. Settle **off** (P43 pid predates `6dd9627`).
  - Storm **1/2 @iter 74**: recon 0.520 / gmatch 0.683 / gnorm **1.4e7** / 60 skips. last_ok **locked iter 73** (20×) then skip-storm restored + **unlocked**; extension kept. Recovered iter 75 recon 0.0025 gnorm 0.18. **Not P27-class** (did not abort P1).
  - **First gate iter 82 gain-probe:** MV **×1.04** DV **×0.75** (P41@82 ×1.01/×0.76; P40 cap ×1.01/×0.75). `unbiased=False not_noisy=True` spread_x0.5. EXTEND. jsonl gmatch **1.6e-4** (mv 1.7e-4 / dv 1.5e-4 already equal). Iter-80 H=55 r=0.573 best_h=55/55 gain_fid=0.751. last_ok advancing unlocked (73→85).
  - **RCA (first gate, EXIT-confirmed):** per-input Huber **FALSIFIED as first-gate DV pin**. Huber residual was already ~0 before the β change (P40-class); reweighting a 1e-4 loss cannot move transfer-matrix DC. Do not try another Huber/isolation reweight. Extra P1 historically does not pin DV (P41 0.76→0.74). KEEP per-input Huber as P1-healthy form (not relative-Huber cousin).
  - **Second gate iter 94 gain-probe:** MV **×1.00** DV **×0.75** (unchanged vs iter 82). Extra P1 FALSIFIED as DV pin again. Iter-90 new wm_best EMA 6.747 (gain-blind; probe still 0.75@DV). Iter **95** silent recon spike **0.473** (gnorm **2.65**, no new skip — not skip-storm). Lock last_ok **94** (20×).
  - **Cap iter 104:** CAPPED **GAIN_NOT_READY 0.75@DV** (MV ×1.06). Detonated-freeze restored last_ok **94**; freeze probe **0.76@DV / MV ×1.02**. P2 completed (`g` frozen). `[p3-skip]` KEEP. Actor INVALID.
- **Observer: MV pinned; per-input Huber FALSIFIED as DV pin.** Val `final.pt`: MV ss/@H **×0.985 / ×0.993** (`wm_gain_rel_err=0.015` HEALTHY; live freeze MV ×1.02). DV ss/@H **×0.740 / ×0.849** (live **0.76@DV**; P42 ×0.737/×0.804; P40 ×0.723/×0.743; P26 ×0.87/×1.00). `wm_dv_ss_ratio_worst=0.740` (loose `wm_observer_gain_healthy=True` at rel_err 0.26<0.35 — GAIN_READY failed). Decomp MV real→post **×1.014** compounding (open-loop ×0.823). DV real→post **×1.005** **faithful**. det_r **−0.215** / R² **−0.064** (amp dead: pred_std **0.137** vs true 1.93; P42 0.124 / 0.164; P40 **0.490** / 0.246; P26 0.68). Vision: MV curves overlap (×0.99); DV WM settles ~×0.74 vs real +0.18; DOB near-zero vs true load steps. `wm_pred_converges_under_constant_action=1.0`. Printed `[val] WM observer gain: HEALTHY` is the loose 0.35 gate.
- **Actor INVALID.** `[p3-skip]` KEEP. `critic_r=+nan` (P3 never ran). econ **−66.3 vs baseline −102.4** is expert-BC (beats_baseline is **not** an actor win; CV viol 12.4, MV viol 13.0, pulse/chatter near a limit). Do not stack critic knobs. Do not revive relative Huber. Do not re-add inv-var. Do not lower skip threshold. Do not train P3 on GAIN_NOT_READY. Do not try another Huber/isolation reweight.
- **ROOT (SysID):** teacher FD from PRBS posterior (Huber ~1e-4, mv≈dv) ≠ rest-then-step probe (0.75@DV at ss **and** @H). Identified G matches TM real (MV −0.32, DV 0.18). Reweighting a saturated Huber cannot move DC. Extra P1 FALSIFIED (0.75→0.75→cap 0.75). Freeze of last_ok 94 did not recover P40 det_r (went **negative**). Isolation-off + recent-floor + lock stay. **Next GPU: P44 held settle** (`gain_match_settle_len` auto=H).

### p44 (`run_p44_gmatchsettle`, branch `cursor/p28`) — EXIT: held settle S=H FALSIFIED as DV pin
- **One attributed GPU change vs P43 process `51b0f45`:** hold a/dv `gain_match_settle_len` auto = horizon **before** the FD (TM rest-then-step IC). Gradful (P25). Per-input Huber stays (KEEP as P1 form; FALSIFIED as DV pin). Isolation-off + last-ok lock + recent-floor stay.
- **Not:** Huber/isolation reweight, extra-P1, leftover `DREAMER_*`, P3 on GAIN_NOT_READY. `DREAMER_GAIN_MATCH_SETTLE_LEN=-1` recovers P43 FD-from-posterior.
- **Step 4 resolved-cfg vs P43:** identical banner except `gmatch_settle=` **H** (P43 pid had no settle field / identity). `latent=deterministic restore_p2=False gain_match=1 dob_ground=2 isolation=0 ss_match=0 iso_dcv=off n_critics=2 rs_freeze=True skip_invalid_p3=True storm_cap=2 lock=20 huber_per_in=True compile=eager`.
- **Liveness:** finished. GPU free. tmux `mbrl2_p44` DEAD, pid **11360** DEAD. Launch `fc18ebf`. 120 iters. `early_stop_reason=p3_skipped_invalid_observer`. `actor_experiment_valid=false`. `p1_gain_not_ready_capped=true`. `p1_detonated_freeze_restored=true`. `p1_last_ok_iter=57`. `p1_last_ok_locked=false` (skip-storm unlock). `[resolved-cfg] gmatch_settle=55`.
  - Healthy through iter **57** (recon 0.0023 gmatch 0.0002). Iter **58** G_pred≈0 (mv Huber **1.3125** / dv **0.256** = `|tgt|-½β`), lock **57**. Storm **1/2 @65** then **2/2 @66** → `_force_p1_cap_at` **639280**. Gate **CAPPED GAIN_NOT_READY 0.76@DV** (MV ×0.94). Detonated-freeze probe 0.77@DV then restored last_ok **57**. Warm-restore SKIPPED. P2 g frozen. `[p3-skip]` KEEP.
- **Observer: S=H FALSIFIED as DV pin; REVERT env-free settle `-1`.** Val `final.pt` (last_ok 57): MV ss/@H **×0.926 / ×0.943** (`wm_gain_rel_err=0.074` HEALTHY; live freeze MV ×0.94; P43 ×0.985/×0.993 — earlier freeze). DV ss/@H **×0.751 / ×0.842** (live **0.76@DV**; P43 ×0.740/×0.849; P40 ×0.723/×0.743; P26 ×0.87). `wm_dv_ss_ratio_worst=0.751`. Decomp MV real→post **×0.995** compounding (open-loop ×0.700). DV real→post **×1.012** **faithful**. det_r **0.099** / R² **0.008** (amp dead: pred_std **0.111** vs true 1.93; P43 −0.215 / 0.137; P40 **0.490** / 0.246; P26 0.68). `wm_next_state_r=0.883` (correlation-only; not a gain pass). Printed `[val] WM observer gain: HEALTHY` is the loose 0.35 gate. GAIN_READY failed.
- **Actor INVALID.** `[p3-skip]` KEEP. `critic_r=+nan` (P3 never ran). econ **−38.5 vs baseline −141.0** is expert-BC. `cum_raw` **−38292**. Do not stack critic knobs. Do not revive relative Huber. Do not re-add inv-var. Do not lower skip threshold. Do not train P3 on GAIN_NOT_READY. Do not retry S=H. Do not set `gain_match_len`=4H as a Huber reweight.
- **ROOT (SysID):** extra WM prior-roll of a PRBS posterior detonated original P1 (G_pred≈0 then storm 2/2). Freeze of last_ok **57** is still GAIN_NOT_READY 0.76@DV — same class as P40–P43 0.74–0.75@DV. Teacher FD from excited IC (Huber~0, even S=H on P43 freeze FD DV×0.969) ≠ TM rest-then-step. Isolation-off + recent-floor + lock + per-input Huber stay. Settle default **REVERT `-1`** (already HEAD). **Next GPU: P45 rest-IC** (`DREAMER_GAIN_MATCH_REST_IC=1` only).

### p45 (`run_p45_restic`, branch `cursor/p28`) — EXIT: rest-IC GAIN-READY; first valid P3 FAIL (σ_min)
- **One attributed GPU change vs P43 identity (P44 settle REVERT):** `gain_match_rest_ic=True`. Real held-OP lookback encode → FD (TM `_settle_capture` post-step pairing). Skips P44 WM-held settle. Isolation loss stays 0. Collect settle = max(H, lookback), not `wm_tf_horizon`. Cache miss **aborts**.
- **Not:** S=H retry, Huber/isolation reweight, leftover `DREAMER_*`. Env-free settle stays `-1`.
- **Step 4 resolved-cfg vs P44:** identical banner except `gmatch_settle=-1` and `gmatch_rest=True`. `latent=deterministic restore_p2=False gain_match=1 dob_ground=2 isolation=0 ss_match=0 iso_dcv=off n_critics=2 rs_freeze=True skip_invalid_p3=True storm_cap=2 lock=20 huber_per_in=True compile=eager`.
- **Liveness:** finished. GPU free. tmux `mbrl2_p45` DEAD, pid **16548** DEAD. Launch `cf25923`. 208 iters. `early_stop_reason=entropy_collapse_window` (30/30 below thr=−0.163, adv_corr=0.045). `best.pt` iter **201**, det_return=−433. `actor_experiment_valid=true`. `p1_gain_not_ready_capped=false`. `p1_last_ok_iter=78` (detonated-freeze restored; probe 0.88@DV). `[resolved-cfg] gmatch_rest=True gmatch_settle=-1`. Storm **1/2 @iter 58–59** recovered; skip-storm **unlock GPU-confirmed**.
  - **P1→P2 iter 82 PASS** (GAIN-READY): worst **0.86@DV**, MV ×0.97 / @H ×0.91, DV ×0.86 / @H ×0.89, band [0.8, 1.3]. First GAIN-READY freeze since P40–P44 0.74–0.76@DV.
  - **P2→P3 iter 136 PASS** reward_mtp median **0.901**.
- **Observer: rest-IC KEEP / PROMOTE as DV pin.** Val `best.pt` (WM frozen = P1 freeze): MV ss/@H **×0.877 / ×0.887** (`wm_gain_rel_err=0.123` HEALTHY). DV ss/@H **×0.815 / ×0.875** (`wm_dv_ss_ratio_worst=0.815` HEALTHY; P44 ×0.751/×0.842; P26 ×0.87). Printed `[val] WM observer gain: HEALTHY`. Decomp MV compounding (open-loop ×0.75); DV posterior+1-step **faithful**. det_r **0.148** / R² **0.019** (amp-dead: pred_std **0.182** vs true 1.93; P40 **0.490**). Do **not** do rest-IC settle=`wm_tf_horizon` next (that was only if still 0.75@DV). Remaining observer hole: DOB amp-dead (stage-1 incomplete on disturbance) — not the next attributed GPU change.
- **Actor VALID and FAIL.** First valid P3 of min-of-2 + freeze `return_scale`. Freeze **KEEP** (1.15→**2.18**, not P26 49.5). min-of-2 **FALSIFIED as sufficient**. Iters 137–145 (critic warmup): entropy **stuck −0.363 = σ_min floor**; rtgt **HEALTHY 0.08–0.09**; adv_corr **0.38–0.41**; first eval mvv thousands (BC μ + DR). After unfreeze: actor_loss **−322**, rtgt **0.08→0.0004**, entropy briefly −0.106 then collapsed, mvv 1e5–4e5. Auto-tune `policy_init_log_std=-1.5`, `log_std_max=-1.520` (σ=0.219), `log_std_min=-1.782` (σ=0.168). Floor-aware ES thr = h_floor+0.20 = **−0.163**. Val econ **−216 vs baseline −92** FAIL; `critic_r=0.742` (correlation only); MV viol **182**, CV viol **33**, reversal 0.135 (smooth_pass); cum_raw **−279k**. Vision: TM direction correct, gains ~12%/19% low, WM too fast; DOB near-zero vs hidden load; closed-loop CV near SP on average but **MV limit-riding**.
- **ROOT (Control / ML):** P1/P2 expert-BC pins last-layer log_std residual to σ_min → P3 starts at entropy floor → warmup on railed on-policy → unfreeze REINFORCE explodes. Freeze rscale worked; stacking more critic knobs would miss the lever. **PROMOTE `gain_match_rest_ic=True`.** Next GPU: P3 log_std reset (μ BC kept).

### p46 (`run_p46_p3sigreset`, branch `cursor/p28`) — EXIT: σ-reset opens entropy; still limit-rides / re-collapses
- **One attributed GPU change vs P45 identity:** `DREAMER_P3_RESET_LOG_STD=1`. At P3 entry, zero the log_std half of the last Linear so σ = auto-tuned `policy_init_log_std` (−1.500); μ (BC launchpad) intact. Pid **24426** is **weights-only** (no Adam log_std-row zero — that is HEAD `77deae0`, not this process). Rest-IC is the TrainConfig default.
- **Not:** rest-IC settle=`wm_tf_horizon`, extra critic knobs, leftover `DREAMER_GAIN_MATCH_REST_IC`, compile/latent/DOB, collect-DV / P3 Kalman.
- **Step 4 resolved-cfg vs P45:** identical except `p3_sigreset=True`. Banner: `latent=deterministic restore_p2=False gain_match=1 dob_ground=2 isolation=0 ss_match=0 iso_dcv=off n_critics=2 rs_freeze=True skip_invalid_p3=True storm_cap=2 lock=20 huber_per_in=True gmatch_settle=-1 gmatch_rest=True p3_sigreset=True compile=eager`. Confirmed `[env-override] p3_reset_log_std=True` only.
- **Liveness:** finished. GPU free. tmux `mbrl2_p46` DEAD, pid **24426** DEAD. Launch `62cf1f5`. 220 iters. `early_stop_reason=entropy_collapse_window` (30/30 below thr=−0.163, latest=−0.290, adv_corr=0.048). `best.pt` iter **181**, det_return=**−611**. `actor_experiment_valid=true`. `p1_gain_not_ready_capped=false`. `p1_last_ok_iter=58` **locked** (skip 0; P45 unlocked to 78). Detonated-freeze restored last_ok **58**; freeze probe **0.81@DV** still GAIN-READY. `[p3] reset policy log_std residual → σ=init (−1.500)` (no Adam-moment line).
  - **P1→P2 iter 82 PASS** (GAIN-READY): worst **0.88@DV**, MV ×0.91 / @H ×0.87, DV ×0.88 / @H ×0.92. Second GAIN-READY since P40–P44 (P45 0.86@DV).
  - **P2→P3 iter 136 PASS** reward_mtp median **0.905**.
- **Observer: rest-IC KEEP.** Val `best.pt` (WM frozen = last_ok 58): MV ss/@H **×0.858 / ×0.854** (`wm_gain_rel_err=0.142` HEALTHY; P45 ×0.877/×0.887). DV ss/@H **×0.793 / ×0.842** (`wm_dv_ss_ratio_worst=0.793` HEALTHY; P45 ×0.815/×0.875; P26 ×0.87). Printed `[val] WM observer gain: HEALTHY`. Decomp MV compounding (open-loop ×0.739); DV posterior×0.964 / 1-step ×0.937 **faithful**. det_r **0.236** / R² **0.053** (amp-dead pred_std **0.293** vs true 1.93; P45 0.148 / 0.182). `wm_next_state_r=0.835` (correlation-only). Lock-without-unlock did not lose GAIN-READY.
- **Actor VALID and FAIL.** First P3 entropy **−0.101** (reset WORKED; P45 −0.363 = σ_min). Freeze rscale **KEEP** (1.11→**2.13**, not 49.5). Warmup rtgt HEALTHY 0.076 then **0.0007** after unfreeze. Frozen-actor first eval mvv **14104** (P45 6061) — opening σ did **not** un-rail BC μ. Unfreeze actor_loss **−357 / −422** (watch “no −300” FAIL). Entropy yanked iter 147 **−0.323** (Adam P2 NLL moments; weights-only) then recovered to **−0.101** @160 then re-collapsed **−0.363** by 183. Val econ **−256 vs baseline −129** FAIL (P45 −216 vs −92); `critic_r=0.698` (correlation only); MV viol **77**, CV viol **160**, reversal **0.409** (P45 0.135). cum_raw **−298k**.
- **ROOT (Control / ML):** last-Linear **is** the σ residual (entropy opened). Opening σ is **not** sufficient: frozen BC μ still limit-rides; unfreeze REINFORCE still explodes; rtgt still collapses. Weights-only first-unfreeze yank **CONFIRMED**. Second collapse after entropy returned to init is **not** leftover P1/P2 Adam. **Do not promote** `p3_reset_log_std`. Do not stack critic knobs. **Next GPU: P47** same `DREAMER_P3_RESET_LOG_STD=1` on HEAD (Adam log_std-row zero). If unfreeze still yanks → Adam was not the lever. If σ stays open but still limit-rides → **REVERT** reset; next is collect-DV / collect-DOB (train/serve).

### p47 (`run_p47_p3sigadam`, branch `cursor/p28`) — EXIT: Adam log_std zero FALSIFIED as yank lever
- **One attributed GPU change vs P46 pid:** HEAD `reset_log_std(opt)` zeros Adam `exp_avg`/`exp_avg_sq` on log_std rows (μ rows kept). Same `DREAMER_P3_RESET_LOG_STD=1`. Not a new knob.
- **Not:** collect-DV, P3 Kalman, extra critic knobs, leftover compile/latent/DOB, promote `p3_reset_log_std`.
- **Liveness:** finished. GPU free. tmux `mbrl2_p47` DEAD, pid **29434** DEAD. Launch `0acde1d`. 253 iters. `early_stop_reason=entropy_collapse_window` (29/30 below thr=−0.163, latest=−0.328, adv_corr=0.049). `best.pt` iter **196**, det_return=**−372**. `actor_experiment_valid=true`. Storm **2/2** capped P1 iter **80**; freeze last_ok **77**; detonated-freeze restored 77. `[resolved-cfg] p3_sigreset=True compile=eager`. `[env-override] p3_reset_log_std=True` only.
  - **P1→P2 iter 80 PASS** (GAIN-READY): worst **0.90@DV**, MV ×0.95/@H ×0.94, DV ×0.90/@H ×0.97. Freeze probe **0.91@DV**. Best first-gate DV of P45–P47.
  - **P2→P3 iter 134 PASS** reward_mtp median **0.909**. `[p3] reset … Adam log_std-row moments zeroed`.
- **Observer: rest-IC KEEP.** Live freeze 0.90@DV. Val TM/distpred **did not run** (`ImportError: resolve_wm_tf_knobs` — HEAD leftover whitelist landed while this pid's late import of `validate.py` raced a partial module). Gate + freeze probe stand in for TM. Do not treat missing JSON as observer FAIL.
- **Actor VALID and FAIL.** First P3 ent **−0.101** (reset WORKED). Freeze rscale **KEEP** (1.14→**2.05**). Warmup rtgt HEALTHY 0.059; mvv ~0 (not P46 14104). Unfreeze iter **145**: ent **−0.101→−0.336**, bc 0.001→**1.255**, actor_loss −6.2 then **−101/−308**. Adam zero **did not** stop the yank. rtgt → 0.0004 by best.pt; mvv **32420** @196. Val econ **−221 vs baseline −121** FAIL; `mv_reversal_rate=0.530` BANG-BANG; critic_r not emitted (val aborted after econ gates).
- **ROOT (Control / ML):** REINFORCE on the unfrozen actor yanks σ and μ — leftover P2 Adam moments were **not** the lever. Opening σ is still not sufficient. Train/serve hole remains: P3 collect streamed `_posterior_step` without measured DV / Kalman while `_realsim_actor_critic_step` re-encodes with both. **Do not promote** `p3_reset_log_std`. Adam-row zero KEEP as hygiene on the opt-in reset. Do not stack critic knobs. **Next GPU: P48** env-free collect DV+Kalman (no `DREAMER_P3_RESET_LOG_STD`).

### p48 (`run_p48_collectdv`, branch `cursor/p28`) — EXIT: collect-DV FALSIFIED as cascade lever
- **One attributed GPU change vs P47 identity:** `stream_serve_step` — P3 collect/val stream measured DV + Kalman (train/serve match). Env-free, `p3_sigreset=False`. Launch `790245a` (+ live HEAD leftovers, not wrap-unlock).
- **Not:** `DREAMER_P3_RESET_LOG_STD`. Extra critic knobs. Isolation reweight. Rest-IC settle=`wm_tf_horizon`. Wrap last-ok unlock (HEAD, not this pid).
- **Step 4 resolved-cfg vs P47:** env-free. `p3_sigreset=False`. `sigma_min_ratio=1.2` resolved. Banner: `latent=deterministic restore_p2=False gain_match=1 dob_ground=2 isolation=0 ss_match=0 iso_dcv=off n_critics=2 rs_freeze=True skip_invalid_p3=True storm_cap=2 lock=20 huber_per_in=True gmatch_settle=-1 gmatch_rest=True p3_sigreset=False compile=eager`. Confirmed `[p3] on-policy collect streams measured DV + Kalman`.
- **Liveness:** finished. GPU free. tmux `mbrl2_p48` DEAD, pid **34706** DEAD. 236 iters. `early_stop_reason=entropy_collapse_window` (30/30 below thr=−0.083, latest=−0.283, adv_corr=0.046). `best.pt` iter **161**, det_return=**−569**. `actor_experiment_valid=true`. Storm 0. `p1_last_ok_iter=24` **locked**. Detonated-freeze restored 24.
  - **P1→P2 iter 82 PASS** (GAIN-READY): live worst **0.89@DV**, MV ×0.91/@H ×0.92, DV ×0.89/@H ×0.94. Freeze probe **0.81@DV** (MV ×1.02/@H ×0.93, DV ×0.81/@H ×0.84). Wrap iter 25 recon 0.145 (43×, gnorm 5.26, skip 0) locked 24; iter 26 recon 0.009 never unlocked.
  - **P2→P3 iter 136 PASS** reward_mtp median **0.927**.
- **Observer:** rest-IC KEEP. Freeze 0.81@DV still GAIN-READY (band [0.8,1.3]) but discarded live 0.89. Val TM/distpred **did not run** (`ImportError: alloc_pinned_obs_host` — same late-import race as P47). Gate + freeze probe stand in. WM SS diagnostic: converges with SS bias. Do not treat missing JSON as observer FAIL.
- **Actor VALID and FAIL.** Collect-DV **did fire**. First P3 ent **−0.283** (σ_min_ratio 1.2; not P47 −0.101 reset). Freeze rscale **KEEP** (1.17→**2.51**). Warmup rtgt HEALTHY 0.059; mvv **17325** @146. Unfreeze **147**: actor_loss **−6.8**, logp **+0.75→−31**, bc 0.002→**1.028**; then **−429/−557** @152–153. rtgt → **0.0004**. Iter 166 ent −0.101 mvv **223031** raw **−2293**. ES @236. Same cascade class as P45–P47.
- **ROOT (Control / ML):** train/serve DV+Kalman is **not** the cascade lever. Freeze-24 is an observer confound (weaker than live 0.89) but P3 was still GAIN-READY. Do not promote `p3_reset_log_std`. Do not stack critic knobs. **KEEP** `stream_serve_step` as identity. **Next GPU: P49** original-P1 wrap last-ok unlock (HEAD default) so freeze can keep late-P1 (~80) not wrap-era 24.

### p49 (`run_p49_wrapunlock`, branch `cursor/p28`) — EXIT: wrap-unlock FALSIFIED as cascade lever
- **One attributed GPU change vs P48 pid:** original-P1 wrap recovery unlocks last-ok (`extra_p1=False` and recon back below 20×) so snapshots resume. Extra-P1 recovered basin stays locked (P40). Not a knob. Env-free. Launch `4ef9bcb`.
- **Not:** leftover `DREAMER_*`, `p3_reset_log_std`, extra critic knobs, raise lock_ratio, rest-IC settle=`wm_tf_horizon`.
- **Step 4 resolved-cfg vs P48:** identical banner. `p3_sigreset=False`. No `bc_mean=` field (P49 pid launched before that banner bit).
- **Liveness:** finished. GPU free. tmux `mbrl2_p49` DEAD, pid **41994** DEAD. 292 jsonl rows (P1 82 / P2 54 / P3 156). `early_stop_reason=entropy_collapse_window` (30/30 below thr=−0.083, latest=−0.283, adv_corr=0.047). `best.pt` iter **161**, det_return=**−1478**. `actor_experiment_valid=true`. Storm **1/2 @64–65** skip-storm unlocked; wrap-unlock **untested**. `p1_last_ok_iter=82` unlocked. No detonated-freeze. Warm-restore SKIPPED.
  - **P1→P2 iter 82 PASS** (GAIN-READY): worst **0.82@MV**, MV ×0.82/@H ×0.84, DV ×0.88/@H ×0.94. Freeze last_ok **82** ≈ live gate (not P48 freeze-24).
  - **P2→P3 iter 136/137 PASS.** First P3 ent **−0.283** = H(σ_min) at `sigma_min_ratio=1.2`. P2 `bc_loss` **−0.87…−1.13** peaked Gaussian NLL.
- **Observer: wrap-unlock KEEP as lock hygiene; FALSIFIED as cascade lever.** Val `best.pt`: MV ss/@H **×0.816 / ×0.830** (`wm_gain_rel_err=0.184` HEALTHY). DV ss/@H **×0.867 / ×0.924** (`wm_dv_ss_ratio_worst=0.867` HEALTHY). Printed `[val] WM observer gain: HEALTHY`. Decomp MV posterior ×0.994 / 1-step ×0.955 compounding (open-loop ×0.745). DV posterior ×0.972 / 1-step ×0.991 **faithful**. det_r **0.632** / R²_det **0.102** (amp-dead pred_std **0.213** vs true 1.93; P45 0.148 / 0.182; P46 0.236 / 0.293). `wm_next_state_r=0.866` (correlation-only). `critic_r=0.667`. Scripted-disturbance **skipped all seeds** (`TypeError: get_authority_target_frac() got an unexpected keyword argument 'cfg'` — leftover HEAD `validate.py` vs pid-stale `training_disturbance`; pin_eval was not in `4ef9bcb`). Fidelity `all_pass=True` was a **0.0 vs 0.0 false pass** (empty paired records). Seed-episode kpi econ mean **−714** (P45 paired **−216 vs −92**).
- **Actor VALID and FAIL.** Freeze rscale **KEEP** (1.15→**2.10**). Warmup 137–146: ent stuck −0.283, rtgt 0.04–0.05, mvv 0. Unfreeze **147**: bc 0.002→**1.24**, actor_loss −4.9 then −17/−55/−136; mvv **667k**; rtgt **0.048→0.0001**; ent briefly −0.11 then re-pinned **−0.283**. ES @253, jsonl continued to 292. cum_raw **−871k**. Same cascade class as P45–P48.
- **ROOT (Control / ML):** freeze-24 was **not** the actor confound. Late-P1 freeze (82 ≈ live gate) still cascades on first unfreeze REINFORCE. P1/P2 expert-BC **NLL** pins last-Linear log_std to σ_min. P46/P47 reset undoes the pin after the fact (FALSIFIED as sufficient). Collect-DV FALSIFIED. Wrap-unlock FALSIFIED as cascade lever. Freeze rscale KEEP. Do not stack critic knobs. Do not promote `p3_reset_log_std`. **Next GPU: env-free P50** `bc_mean_only=True` (HEAD default; P1/P2 MSE-on-μ). Watch P2 `bc_loss`>0; first P3 ent ≈ H(σ_init) not −0.283.

### p50 (`run_p50_bcmean`, branch `cursor/p28`) — EXIT: μ-only opened σ; val beats; unfreeze still yanked
- **One attributed GPU change vs P49 pid:** `bc_mean_only=True` (HEAD default `c900766`). P1/P2 expert-BC is MSE-on-μ so cloning does not pin last-Linear log_std to σ_min. P3 `expert_bc_p3_loss` was already MSE-on-μ. Opt out `DREAMER_BC_MEAN_ONLY=0`. Launch **`9cbf771`**. Env-free. tmux `mbrl2_p50`, pid **49389**, `CUDA_VISIBLE_DEVICES=0`. Started 16:34:23. EXIT=0.
- **Not:** leftover `DREAMER_*`, `p3_reset_log_std`, extra critic knobs, rest-IC settle=`wm_tf_horizon`, isolation reweight.
- **Step 4:** `[resolved-cfg] latent=deterministic restore_p2=False gain_match=1 dob_ground=2 dob_reg=0 isolation=0 ss_match=0 iso_dcv=off n_critics=2 rs_freeze=True skip_invalid_p3=True storm_cap=2 lock=20 huber_per_in=True gmatch_settle=-1 gmatch_rest=True p3_sigreset=False bc_mean=True compile=eager`. Process environ: only `CUDA_VISIBLE_DEVICES=0` + `PYTORCH_CUDA_ALLOC_CONF`. `[run] pinned eval modules at launch`.
- **Liveness:** finished. GPU free. tmux `mbrl2_p50` GONE, pid **49389** DEAD. 286 jsonl rows. `early_stop_reason=entropy_collapse_window` (30/30 below thr=−0.083, latest=−0.273, adv_corr=0.049). `best.pt` iter **191**, det_return=**−137**. `actor_experiment_valid=true`. skip **1** (not a storm). `p1_last_ok_iter=88` **locked**. Detonated-freeze restored 88. Wrap-unlock ×2 GPU-confirmed (56→58, 70→73).
  - **Gate 82 FAIL** `worst=1.37@MV` wrap-adjacent recon 0.0283. Extra-P1 **detonated iter 89** recon 0.6286. **Gate 94 FAIL** 0.69@DV noisy on detonated g.
  - **Gate 104 PASS live 0.85@DV**; freeze probe **GAIN-READY 0.90@DV** (MV ×0.99/@H ×0.94; DV ×0.90/@H ×0.93). STAGE 2 @104.
  - **P2→P3 PASS @158** reward_mtp median **0.926**. P2 `bc_loss` **MSE 0.000–0.015 never NLL**.
- **Observer KEEP (GAIN-READY freeze).** Val `best.pt`: MV ss/@H **×0.935 / ×0.877** (`wm_gain_rel_err=0.065` HEALTHY). DV ss/@H **×0.836 / ×0.868** (`wm_dv_ss_ratio_worst=0.836` HEALTHY). Printed `[val] WM observer gain: HEALTHY`. Decomp MV posterior ×1.002 / 1-step ×1.167 compounding (open-loop ×0.715). DV posterior ×1.003 / 1-step ×1.024 **faithful**. det_r **0.290** / detrended r **0.544** R²_det **0.141** (amp-dead pred_std **0.220** vs true 1.93). `wm_next_state_r=0.791`. `critic_r=0.686`. Scripted dist **9 real pairs** (pin_eval WORKED; not P49 TypeError / 0-vs-0).
- **Actor VALID: σ-open KEEP; train yank FALSIFIED-as-sufficient; val econ BEATS.** First P3 ent **−0.107** ≈ H(σ_init) (not −0.283). Freeze rscale **KEEP** (1.22→**2.33**). Warmup 159–168: ent −0.10, rtgt HEALTHY 0.04–0.05, logp_std ~0.56. Unfreeze **169**: ent **−0.107→−0.268**, bc 0.003→**0.93**, actor_loss **−6.4**, logp_std **0.56→53.6**. rtgt **0.05→0.0001**. Brief pin −0.283 @212–217 then recovered. best.pt **191** still open σ (−0.106) with logp_std 33 / rtgt 0.001. ES @286 latest −0.273. Paired scripted **agent −56 vs baseline −104 BEATS** (P45 −216 vs −92; P46 −256 vs −129; P47 −221 vs −121). Seed kpi mean **−74** (P49 **−714**). reversal **0.340** smooth_pass (not P47 0.53 bang-bang). cum_raw mean **−91k** (P49 −871k). `all_pass=True` on real pairs.
- **ROOT (Control / ML):** P1/P2 MSE-on-μ **does** leave σ at init (canary WORKED) and this pid is the **first valid P3 that beats baseline**. Unfreeze REINFORCE **still yanks log_std** (same class as P46/P47 after opening σ) and train ES still fires. Val win is `best.pt` 191 (σ still open) during the cascade, not a stable late-P3. Freeze rscale KEEP. Do not stack critic knobs. Do not promote `p3_reset_log_std`. **Next GPU: env-free P51** `p3_stop_grad_log_std=True` (HEAD default; REINFORCE trains μ only). Watch unfreeze 169-class: ent stays ≈−0.10, log_std grad 0.

### p51 (`run_p51_sglogstd`, branch `cursor/p28`) — EXIT: stop-grad held σ; μ-rail still cascaded
- **One attributed GPU change vs P50 pid:** `p3_stop_grad_log_std=True` (HEAD default `d6a4511`). P3 REINFORCE `log_prob_of` / entropy detach log_std so σ stays at `policy_init_log_std`; μ still trains. Opt out `DREAMER_P3_STOP_GRAD_LOG_STD=0`. Also carries P50 `bc_mean_only=True` and HEAD gain-probe last-ok (`f2c9092`, not in P50 pid). Env-free. tmux `mbrl2_p51`, pid **54815**, `CUDA_VISIBLE_DEVICES=0`. Started ~21:47 CDT. EXIT=0 (val finished ~01:52 CDT).
- **Not:** leftover `DREAMER_*`, `p3_reset_log_std`, extra critic knobs, rest-IC settle=`wm_tf_horizon`, `actor_kl_coef`.
- **Step 4 CONFIRMED:** `[resolved-cfg] … isolation=0 … gmatch_settle=-1 gmatch_rest=True p3_sigreset=False bc_mean=True p3_sglogstd=True compile=eager`. Process environ: only `CUDA_VISIBLE_DEVICES=0` + `PYTORCH_CUDA_ALLOC_CONF`.
- **Liveness:** finished. GPU free. tmux `mbrl2_p51` GONE, pid **54815** DEAD. 361 jsonl rows. `early_stop_reason=p3_plateau` (no >+1.0% over best_det_return=**−86.607** for 200 iters). `best.pt` / `final.pt` iter **161**. `actor_experiment_valid=true`. skip-storm **1/2**. `p1_last_ok_iter=82` **unlocked**.
- **P1 / wrap / skip-storm:** lock@**47**; skip-storm **1/2** @52 restore 47. Wrap lock@**66** unlock@**69**. Gate **GAIN-READY** 0.86@MV / 0.88@DV @**82**; last_ok **82 unlocked**. P1→P2 wm_best SKIPPED. P2 bc **MSE** never NLL. Wrap-recovery unlock **GPU-confirmed P51**. Skip-storm unlock **GPU-confirmed P51**.
- **P2→P3:** PASS @**136** reward_mtp median **0.922**. rscale **FREEZE 2.170 KEEP**. First P3 ent **−0.125** ≈ H(σ_init).
- **Unfreeze 147 (A/B canary):** warmup 146 ent **−0.141** `actor_logp_std` **0.54**. Iter 147: ent **−0.101 HELD** (P50 169 yanked **−0.107→−0.268**). `actor_logp_std` **0.54→38** (μ-rail logp variance, not σ). actor **−4.66**, bc 0.002→0.87, gnorm **9.65** (clip). **Stop-grad KEEP as unfreeze-yank block.**
- **Non-sticky floor:** first <−0.25 @294; **−0.282 @315** (σ_min ≈−0.283, tanh-rail geometry with frozen σ) then **recovered −0.106 @348**. 23-iter <−0.25 streak 296–318; entropy_collapse ES did **not** fire. EXIT is plateau @361, not σ-stick.
- **Cascade (μ-rail):** best.pt **161** already cascading (ent −0.102, logp_std **47**, rtgt **0.0057**, mvv **19k**, ret_w **−87**). Late P3: actor ±200, logp_mean **−227 @315**, logp_std ~150, rtgt **0.00015**, ret_w **−1822**, mvv **681k @361**, bc ~1.83. Same class as P50 best **191** during cascade.
- **Observer GAIN-READY freeze.** Val `best.pt`: MV ss/@H **×0.812 / ×0.782** (`wm_gain_rel_err=0.188` HEALTHY; P50 ×0.935/×0.877). DV ss/@H **×0.819 / ×0.855** (`wm_dv_ss_ratio_worst=0.819` HEALTHY; P50 ×0.836/×0.868). Decomp MV posterior ×0.978 / 1-step ×0.969 compounding (open-loop ×0.772). DV posterior ×0.942 / 1-step ×0.945 **faithful**. det_r **0.626** / R²_det **0.162** (amp-dead pred_std **0.264** vs true 1.93; P50 0.290 / 0.220). `wm_next_state_r=0.814`. `critic_r=0.660`. Scripted dist **9 real pairs**.
- **Actor VALID: yank-block KEEP; cascade FALSIFIED; val still beats, worse than P50.** Freeze rscale **KEEP**. Paired **agent −72 vs baseline −111 BEATS** (P50 **−56 vs −104**; 6/9 pairs). Seed kpi mean **−82** (P50 **−74**; P49 **−714**). reversal **0.143** smooth_pass (P50 0.340). cum_raw mean **−100k** (P50 −91k). `all_pass=True` on real pairs. Win is early `best.pt` 161 (14 iters after unfreeze) during the μ-rail, not a stable late-P3.
- **ROOT (Control / ML):** detaching σ **does** block the unfreeze yank and **does** let entropy recover from a delayed rail-geometry dip. Frozen-σ REINFORCE still rails **μ**: `(u−μ)/σ²` explodes (`actor_logp_std` 0.54→38 on the first unfreeze step; advantage_clip 8 and grad_clip 10 already saturate). Val beat is the same “early best.pt during cascade” class as P50. Do not stack critic knobs. Do not promote `p3_reset_log_std`. Do not revive `actor_kl_coef` (p136 FALSIFIED as a σ-collapse TR). **Next GPU: env-free P52** `p3_logp_clip=8` (clamp REINFORCE logp so railed pairs contribute ~0 μ-grad). Watch unfreeze 147-class: `actor_logp_std` may still jump in the **unclipped** jsonl diag; actor_loss / μ should not explode.

### p52 (`run_p52_logpclip`, branch `cursor/p28`) — EXIT: logp-clip delayed first-unfreeze rail; cascade + val FAIL
- **One attributed GPU change vs P51 pid:** `p3_logp_clip=8.0` (HEAD default `d910ee2`). Clamp `logp` in `-(adv * logp)` so frozen-σ `(u−μ)/σ²` cannot explode μ. Jsonl `actor_logp_*` stays **unclipped**. Opt out `DREAMER_P3_LOGP_CLIP=0`. Also carries P51 `p3_stop_grad_log_std=True` (KEEP as yank block) and P50 `bc_mean_only=True`. Env-free. tmux `mbrl2_p52`, pid **64705**, `CUDA_VISIBLE_DEVICES=0`. Started 02:01:47 CDT. EXIT=0 (val finished ~05:48 CDT).
- **Not:** leftover `DREAMER_*`, `p3_reset_log_std`, extra critic knobs, `actor_kl_coef`.
- **Step 4 CONFIRMED:** `[resolved-cfg] latent=deterministic restore_p2=False gain_match=1 dob_ground=2 dob_reg=0 isolation=0 ss_match=0 iso_dcv=off n_critics=2 rs_freeze=True skip_invalid_p3=True storm_cap=2 lock=20 huber_per_in=True gmatch_settle=-1 gmatch_rest=True p3_sigreset=False bc_mean=True p3_sglogstd=True p3_logpclip=8 compile=eager`. Process environ: only `CUDA_VISIBLE_DEVICES=0` + `PYTORCH_CUDA_ALLOC_CONF`.
- **Liveness:** finished. GPU free. tmux `mbrl2_p52` GONE, pid **64705** DEAD. 209 jsonl rows. `early_stop_reason=entropy_collapse_window` (30/30 below thr=**−0.083**, latest **−0.111**, `adv_corr=0.049`). `best.pt` / `final.pt` iter **176** det **−634**. `actor_experiment_valid=true`. skip-storm **0**. `p1_last_ok_iter=78` **unlocked**.
- **P1 / wrap:** lock@**38** wrap-unlock@**39**; lock@**79** wrap-unlock@**81**. Gate **GAIN-READY** @**82** last-ok probe **78** MV **0.95/@H 0.95** DV **0.84/@H 0.89**. Freeze restored last_ok **78**. P50 last-ok probe **GPU-confirmed in this pid**.
- **P2→P3:** STAGE 3 @**136**. First P3 @137 ent **−0.143**. rscale **FREEZE 2.833 KEEP**.
- **Unfreeze 147 (A/B canary):** ent **−0.101 HELD** (not yanked to σ_min). Unclipped logp_std **0.53→6.11** (not P51 **38**); actor **−0.77** (not **−4.66**). **Logp-clip KEEP as delayed first-unfreeze bound.**
- **Delayed μ-rail then collect:** **@165–169** unclipped std **80.8 / peak 100.7**; actor **−5.80**. Later std collapsed **0.88@196** while actor **−9.81** mvv **813k@196 / 841k@191** (past P51 681k@361). rtgt **0.00012**. ema **−2231@209**. skip 0. Entropy **−0.105 HELD** (stop-grad yank-block still holding; ES fired because init H≈−0.10 is already below thr −0.083 and adv_corr stayed <0.05).
- **Observer GAIN-READY freeze.** Val `best.pt`: MV ss/@H **×0.925 / ×0.927** (`wm_gain_rel_err=0.075` HEALTHY; P51 ×0.812/×0.782; P50 ×0.935/×0.877). DV ss/@H **×0.815 / ×0.860** (`wm_dv_ss_ratio_worst=0.815` HEALTHY; P51 ×0.819/×0.855). Decomp MV posterior ×0.996 / 1-step ×1.037 compounding (open-loop ×0.753). DV posterior ×0.998 / 1-step ×1.035 **faithful**. det_r **0.127** / R²_det **0.013** (amp-dead pred_std **0.102** vs true 1.93; P51 0.626 / 0.264). `wm_next_state_r=0.792`. `critic_r=0.638`. Scripted dist **9 real pairs**.
- **Actor VALID: delayed-rail KEEP; cascade FALSIFIED; val LOSES.** Freeze rscale **KEEP**. Paired **agent −377 vs baseline −87 FAIL** (0/9 pairs; P50 **−56 vs −104**  beats; P51 **−72 vs −111** beats). Seed kpi mean **−306** (P50 **−74**; P51 **−82**; P49 **−714**). reversal **0.451** smooth_pass (P51 0.143; P50 0.340). cum_raw mean **−373k** (P50 −91k; P51 −100k). `all_pass=False` (econ vs baseline). Win-class is gone: `best.pt` 176 already det **−634** (P51 best 161 det **−87**) during the delayed walk.
- **ROOT (Control / ML):** clamping REINFORCE logp at ±8 **does** delay the unclipped first-unfreeze rail (~18 iters) because |logp|>8 has zero clamp-grad. It does **not** stop in-support `(u−μ)/σ²` SGD (|logp|<8, |z|≲4) from walking μ onto the tanh rail. After μ has walked, unclipped std can look healthy (~0.9) while mvv/actor_loss explode. Delayed walk also **worsened** `best.pt` vs P51 (locked later at a more railed μ). Do not stack critic knobs. Do not promote `p3_reset_log_std`. Do not revive `actor_kl_coef` (p136 FALSIFIED as a σ-collapse TR; not wired in `_realsim_actor_critic_step`). **Next GPU: env-free P53** one attributed μ-walk limiter (PPO-style ratio clip vs a frozen unfreeze-μ snapshot). Watch unfreeze 147: unclipped logp_std stays ~0.5 and mvv stays warmup-class (~16k), not 813k.
- **HEAD (not in pid 64705):** `p3_logp_clip` is nats **per action dim**; summed logp clamp = `8×n_mv` (`_p3_logp_clip_bound`; 1-MV test_sim identity). SNR window and `identify_dynamics` / noise-config no longer invent τ=50 s / θ=5 s when SysID keys are missing. APCEnv schedule knobs + step-seed/shaping-safe/PRBS-seg/ss-window `ENV_OVERRIDES`. Do **not** switch APCEnv to `auto_derive`.

### p53 (`run_p53_muratio`, branch `cursor/p28`) — EXIT: PPO μ-ratio clip KEEP as walk limiter + cascade; actor champion
- **One attributed GPU change vs P52 pid:** `p3_mu_ratio_clip=0.2` (HEAD default `126f011`). PPO surrogate `min(ratio·adv, clip(ratio, 1−ε, 1+ε)·adv)` with `logp_old` from a **frozen deepcopy of the policy at the first `_realsim_actor_critic_step`**. Jsonl `actor_logp_*` stays unclipped. Opt out `DREAMER_P3_MU_RATIO_CLIP=0`. Also P52 `p3_logp_clip=8`, P51 `p3_stop_grad_log_std=True`, P50 `bc_mean_only=True`. Env-free. tmux `mbrl2_p53`, pid **75966**, `CUDA_VISIBLE_DEVICES=0`. Started 06:09 CDT. EXIT=0 (~09:54 CDT val done).
- **Not:** leftover `DREAMER_*`, tighter `p3_logp_clip`, `p3_reset_log_std`, extra critic knobs, `actor_kl_coef`, lag-copy.
- **Step 4 CONFIRMED:** `[resolved-cfg] … p3_sglogstd=True p3_logpclip=8 p3_muratio=0.2 compile=eager`. Process environ: only `CUDA_VISIBLE_DEVICES=0` + `PYTORCH_CUDA_ALLOC_CONF`.
- **Liveness:** finished. GPU free. tmux `mbrl2_p53` DEAD, pid **75966** DEAD. 262 jsonl rows. `early_stop_reason=entropy_collapse_window` (30/30 below thr=**−0.083**, latest **−0.101**, `adv_corr=0.047`). `best.pt` / `final.pt` iter **166** det **−46.2**. `actor_experiment_valid=true`. skip-storm **1/2 @73**. `p1_last_ok_iter=82` unlocked.
  - **P1→P2 @82 GAIN-READY** worst **0.88@DV** (MV 0.94/@H 0.93, DV 0.88/@H 0.93). vs P52 last-ok **78**.
  - **P2→P3 PASS @136** mtp median **0.909**. rscale **FREEZE 2.302 KEEP**.
  - **Unfreeze 147:** ent **−0.101 HELD**. Unclipped logp_std **0.52→6.71@149 then held ~0.47@262** (not P51 **38** / P52 delayed **80.8**). actor_loss bounded. mvv transient **56.8k@191** not **813k**. Passed P52 rail window **and** P52 ES iter **209**.
- **Observer GAIN-READY freeze.** Val `best.pt`: MV ss/@H **×0.900 / ×0.898** (`wm_gain_rel_err=0.100` HEALTHY; P52 ×0.925/×0.927; P50 ×0.935/×0.877). DV ss/@H **×0.848 / ×0.903** (`wm_dv_ss_ratio_worst=0.848` HEALTHY). Printed `[val] WM observer gain: HEALTHY`. Decomp MV posterior ×1.006 / 1-step ×1.010 compounding (open-loop ×0.714). DV posterior ×1.004 / 1-step ×0.992 **faithful**. det_r **0.279** / R²_det **0.046** (amp-dead pred_std **0.168** vs true 1.84; P50 0.290 / 0.220; P52 0.127 / 0.102). `wm_next_state_r=0.889`. `critic_r=0.681`. Scripted dist **9 real pairs**.
- **Actor VALID: μ-walk KEEP; cascade KEEP; val BEATS and new champion.** Freeze rscale **KEEP**. Paired **agent −13 vs baseline −98 BEATS** (**9/9**; P50 **−56 vs −104**; P51 **−72 vs −111**; P52 **−377 vs −87 FAIL**). Seed kpi mean **−13** (P50 **−74**; P51 **−82**; P52 **−306**). reversal **0.202** smooth_pass (P50 0.340; P51 0.143). cum_raw mean **−15.6k** (P50 −91k; P52 −373k). `all_pass=True`. best.pt **166** is still early-unfreeze (19 iters after 147) — same snapshot *class* as P50/P51, but this snapshot **does not** carry a railed μ.
- **ROOT (Control / ML):** PPO clip vs frozen unfreeze-μ **does** stop in-support `(u−μ)/σ²` SGD from walking μ onto the tanh rail (unclipped std held; mvv not 813k; val 9/9). Train ES is the **same entropy-threshold trap as P52** (H(σ_init)≈−0.101 already below thr −0.083 and adv_corr <0.05 for 30 iters) — **not** evidence the ratio clip failed. Do not stack critic knobs. Do not promote `p3_reset_log_std`. Do not revive `actor_kl_coef`. Do **not** tighter-clip `p3_logp_clip`. **Next GPU: env-free P54** `early_stop_entropy_collapse_floor_frac=0.25` (σ-band frac; not a stacked actor knob).
- **HEAD:** `p3_mu_ratio_clip=0.2` stays env-free default. Actor champion **P53**.

### p54 (`run_p54_esentband`, branch `cursor/p28`) — EXIT: ES σ-band floor KEEP as false-trip fix; FALSIFIED as actor-econ lever
- **One attributed GPU change vs P53 pid:** `early_stop_entropy_collapse_floor_frac=0.25` (HEAD default `f6739ac`). Trip at `H(σ_min)+0.25·(H(σ_max)−H(σ_min))` instead of `H(σ_min)+0.20 nats`. Env-free. tmux `mbrl2_p54`, pid **95956**, `CUDA_VISIBLE_DEVICES=0`. EXIT=0 (~15:13 CDT val done).
- **Not:** leftover `DREAMER_*`, tighter `p3_logp_clip`, `p3_reset_log_std`, extra critic knobs, `actor_kl_coef`, raising `min_adv_corr`.
- **Step 4 CONFIRMED:** `[resolved-cfg] … p3_muratio=0.2 es_ent_floor=0.25 compile=eager`. Process environ: only `CUDA_VISIBLE_DEVICES=0` + `PYTORCH_CUDA_ALLOC_CONF`.
- **Liveness:** finished. GPU free. tmux `mbrl2_p54` DEAD, pid **95956** DEAD. 446 jsonl rows. `early_stop_reason=p3_plateau` (no >+1.0% over best_det_return=**−36.189** for 200 iters). `best.pt` / `final.pt` iter **246** det **−36.2**. `actor_experiment_valid=true`. skip-storm **0**. `p1_last_ok_iter=82` unlocked.
  - **P1→P2 @82 GAIN-READY** worst **0.91@DV** (MV 0.93/@H 0.93, DV 0.91/@H 0.96). last_ok **82 unlocked**. wm_best iter **90** EMA **6.517**.
  - **P2→P3 PASS @136** mtp median **0.913**. rscale **FREEZE 2.062 KEEP**.
  - **P3 entropy:** first **−0.127** → open-σ **−0.101** (min **−0.136**). **310/310** below old trip **−0.083**; **0** below new **−0.238**. Passed P53 ES death @262. Unfreeze logp_std spike **1.38@150** then held **~0.38–0.55**. μ-ratio `clip_frac` **0.42@147 → 0.11@446**. rtgt 0.06→**0.001**.
- **Observer GAIN-READY freeze.** Val `best.pt`: MV ss/@H **×0.821 / ×0.815** (`wm_gain_rel_err=0.179` HEALTHY; P53 ×0.900/×0.898). DV ss/@H **×0.835 / ×0.874** (`wm_dv_ss_ratio_worst=0.835` HEALTHY). Printed `[val] WM observer gain: HEALTHY`. Decomp MV posterior ×0.991 / 1-step ×0.977 compounding (open-loop ×0.772). DV posterior ×0.969 / 1-step ×0.896 **prior**. det_r **0.280** / R²_det **0.051** (amp-dead pred_std vs true; P53 0.279). `wm_next_state_r=0.907`. `critic_r=0.704`. Vision: MV/DV both correct sign, gain too small (~×0.82/×0.83), dynamics too fast; DOB heavily attenuated; seed_10004 limit-riding/oscillation, seed_10005 tracking. Scripted dist **9 real pairs**.
- **Actor VALID: ES KEEP; extra P3 FALSIFIED as econ lever; val BEATS baseline, loses to P53.** Freeze rscale **KEEP**. Paired **agent −26 vs baseline −101 BEATS** (**9/9**; P53 **−13 vs −98**). Seed kpi mean **−26**. reversal **0.197** smooth_pass. `all_pass=True`. CV viol mean **5.11** / MV viol **11.84** (P53 2.48 / 6.41). best.pt **246** then 200-iter plateau — extra budget after P53's death did **not** beat P53.
- **ROOT (Control / ML):** σ-band ES **does** stop the H(σ_init) false trip (P53 `entropy_collapse_window`). Extra P3 time cannot leave the ε=0.2 ball around unfreeze-μ (`clip_frac` died to 0.11; det_return plateau). Do not stack critic knobs. Do not promote `p3_reset_log_std`. **Next GPU: env-free P55** recopy μ-ratio snapshot every P3 iter (`p3_mu_ratio_refresh_iters=1`). Watch `clip_frac` stay >~0.15 late P3; val paired vs P53 −13/−98; no P52 μ-rail (logp_std ~0.5 not 80).
- **HEAD (P55 includes):** `critic_mc_loss` logged; dead `pmpo_loss`/`kl_to` **REMOVED**; rest-IC CUDA-graph; refuse `DREAMER_ACTOR_LOSS=pmpo`.

### p55 (`run_p55_murefresh`, branch `cursor/p28`) — LAUNCH: PPO μ-ratio recopy each P3 iter
- **One attributed GPU change vs P54 pid:** `p3_mu_ratio_refresh_iters=1` (HEAD default `b321d89`). Recopy live μ at the first AC call of every P3 collect iter (window = `phase3_train_steps_per_iter` inner steps). 0 = P53 freeze-forever. First unfreeze epoch still clips vs unfreeze-μ (warmup does not step the actor). Opt out `DREAMER_P3_MU_RATIO_REFRESH=0`. Also P54 ES floor 0.25, P53 `p3_mu_ratio_clip=0.2`. Env-free. tmux `mbrl2_p55`, pid **103504**, `CUDA_VISIBLE_DEVICES=0`. Started 16:04 CDT.
- **Not:** leftover `DREAMER_*`, tighter `p3_logp_clip`, `p3_reset_log_std`, extra critic knobs, `actor_kl_coef`, loosening ε.
- **Step 4 CONFIRMED:** `[resolved-cfg] … p3_sglogstd=True p3_logpclip=8 p3_muratio=0.2 p3_murefresh=1 es_ent_floor=0.25 compile=eager`. Process environ: only `CUDA_VISIBLE_DEVICES=0` + `PYTORCH_CUDA_ALLOC_CONF`.
- **Liveness:** tmux `mbrl2_p55` LIVE, pid **103504** LIVE, `device=cuda`. Rest-IC CUDA-graph capture **failed** on first GPU attempt (`make_graphed_callables` autocast cache) → eager T-loop (same rest-IC as P54). HEAD after this pid wraps capture in `cache_enabled=False`. **Watch** as above. **Do not launch a second GPU job.**




### Sim-adaptive leftovers (env-free multi-sim; do not promote plants yet)

Already unitless / derived: held-rollout `win` clamped to `(K-1)//4` (identity 8 at K=55; K=15 was dead); isolation TBPTT `max(8, round(K/3.5))` (chunked `img_rollout`, same `h`-only `keep_c` cuts); `gain_match_len` / isolation len = H; **`gain_match_settle_len` default `-1` (P44 storm 2/2 REVERT; `0` still auto-H A/B)**; TM probe settle is `wm_tf_horizon(H)=max(80,4H)`; isolation sample window `max(seq_len, K+1)` (test_sim seq_len ≥ H unchanged); P1/P2 main WM sample `max(seq_len, K+1)` (overshoot/gain-match; P3 stays seq_len); gain-match open-loop `K` not truncated to `T-1`; `skip_storm_last_ok_recon_ratio=5` (recon/recon); `skip_storm_p1_cap_after=2` (storm count; **P32 GPU-confirmed** continue 1/2 @iter 53). Storm-1 continue now **keeps** `p1_gate_max_ext_steps` (P32 CAPPED 0.71@DV when it was closed); storm 2 still `_force_p1_cap_at`. Isolation / overshoot / held are **RSSM-interface** (rssm + tssm; SF no-op). HEAD not in P33 process; `p1_gate_wm_ema_min=1.5` (fidelity mix); inject EVERY = f(buffer lap) (test_sim 20/10); inject N = f(n_mv, n_dv) (test_sim 5/2/2/3); isolation settle = 24 **per input** (test_sim 24+24 / cap 48 settle-only; distillation 96+24 / cap 120). Isolation_buf holds only those settle episodes (not MIMO PRBS). Isolation settle is a whole-episode hold at a stratified level (`action_std=0`); DV step is MV-action-isomorphic (`isolated_level × span/2`). Opt-in: `gain_match_huber_beta<=0` → median |tgt| only when `gain_match_huber_per_input` is False (P43 default is per-input `|tgt_ij|`). Gain-ready band `[0.80, 1.30]` and wm-fidelity mix weights are unitless TrainConfig (were os.environ). `rssm_latent_type` default **deterministic** (P26 proven; P29 env-free drop of the leftover env-var). `compile_mode` default **eager**; `DREAMER_COMPILE` / `DREAMER_COMPILE_MODE` are in `ENV_OVERRIDES` (MODE was CLI-only). Intra-op threads and inductor workers scale with `sched_getaffinity` (host-adaptive). `skip_invalid_p3` default ON (unitless validity flag, not a plant unit). Stage-1 skips unused DOB prior-core when `dob_active=False` **and** skips `apply_dob` clone + ground/reg (`d_t≡0`; HEAD, not P33 process). SNR summary/WARN is **measured CV+DV** (`summary_scope`; constant aug channels excluded). Isolation jsonl splits detached `wm_isolation_mv_traj` / `wm_isolation_dv_traj`. jsonl `wm_gain_match_{mv,dv}_ratio` = mean `G_pred/G_tgt` (HEAD; P44 pid Huber-only; teacher FD, not TM). TM / postprior / smoke share `wm_tf_horizon` (was inlined `max(80,4H)`). Rest-IC encode `rollout_observed last_only return_feats=False` + Stage-1 `_posterior_step` (P46 pid; P45 pid 16548 still stacked). `gain_match_rest_ic` default **True** (P45 EXIT PROMOTE).

Still not sim-adaptive:
- `wm_fidelity_{warmup,patience}_iters` and `wm_probe_every_iters` stay in *iters* (one WM update ≈ one iter — plant-independent). KEEP.
- `dv_feedforward=True` still appends measured DV to **head** feat (decoder half is off). `dv_as_input` is the symmetric path; do not re-add decoder FF.
- `dv_prbs_seed_episodes=24` / `expert_seed_episodes=24` / `constant_action_seed_episodes=40` / `baseline_seed_episodes=16` are flat episode counts (each DV-PRBS episode already sweeps every DV; episode length already scales with τ). Isolation settle is already per-input. `phase3_onpolicy_buffer_eps=16` is an on-policy episode count (same). **Op-band fractions** (`baseline_seed_op_band=0.6`, `constant_action_seed_op_band=0.6`, `prbs_seed_op_band=0.95`) are unitless and now in `ENV_OVERRIDES` (`DREAMER_BASELINE_SEED_OP_BAND` / `DREAMER_CONST_ACTION_OP_BAND` / `DREAMER_PRBS_SEED_OP_BAND`). Env-free baseline still `min(0.6, PRBS)` unless the baseline field is explicit. **Const-action / step-settle / step-test MIMO OP** is no longer the all-MVs-equal diagonal: `_per_mv_hold_rows` permutes the same linspace per MV (`n_mv<=1` → None, test_sim scalar path + RNG unchanged). Step-test MV *events* step one channel (`primary_mv_pos`). Do not raise the 40-episode sentinel without a MIMO run.
- Abs Huber gain-match is count-equal per input but **scale-unequal** when `|G_mv|≫|G_dv|`. P43 EXIT: per-input β **equalized jsonl Huber** (mv≈dv ~1e-4) but val DV still **×0.740 / @H ×0.849**. Residual was already ~0 — β is not the DV lever. **P44 EXIT** S=H FALSIFIED (val DV ×0.751). **P45 EXIT** rest-IC **PROMOTE** (`gain_match_rest_ic=True`): first GAIN-READY since P40–P44; val DV ss/@H **×0.815 / ×0.875**. **P46 EXIT** second GAIN-READY; val DV **×0.793 / ×0.842** (lock last_ok 58). Do **not** revive relative Huber. Do not retry S=H. Do not set rest-IC settle=`wm_tf_horizon` (P45 was GAIN-READY without it). jsonl `*_ratio` ~×1 is teacher FD (not the TM gate). `DREAMER_GAIN_MATCH_REST_IC=0` reverts to PRBS-posterior FD.
- Isolation/ss-match is **off env-free** (P40 EXIT KEEP as default; **FALSIFIED as DV pin** — val DV ×0.723 not P26 ×0.87; P41 EXIT ×0.700; P42 EXIT ×0.737; P43 EXIT ×0.740). P08 auto-enable isolation=1 / ss_match=3 ran on P26 (jsonl iso 1.44; `run_plan` 0.0 was the pre-rewrite dump) and P32–P39. Abs MSE + whole-episode holds drowned DV (×0.87 → ×0.66–0.70) but turning it off did not recover P26 DV. Gain-match is the DC supervisor. Opt in `DREAMER_WM_INPUT_ISOLATION_COEF=1` (len/settle auto). When on: abs MSE only, inv-var **REMOVED**, `wm_isolation_dcv_match=True` with `wm_isolation_dcv_min_scale=1.0`. Structural `|G_max|/|G_min| > 1/op_band` cannot equalize |ΔCV| (P38 shrink FALSIFIED; P39 cube-boost FALSIFIED; P40 isolation-off FALSIFIED as DV pin; P41 recent-floor FALSIFIED as DV pin). Do **not** re-add `wm_isolation_var_norm`. Do not try another isolation reweight. May-2026 per-head/latent diag probes default **OFF**.
- `wm_overshoot_max_starts=24` / `gain_match_max_starts=6` are **per-sequence** caps. With gpu-calib `B=128`, total `Bm=B·S` is ~8× paper `B=16`. Do **not** retune until the observer is GAIN-READY (would stack a second observer change). Host-adaptive thread caps already scale with `sched_getaffinity`.
- Gain-probe log prints **per-input** `ss_pairs` **and `@H`** + `unbiased`/`not_noisy` (HEAD; P43 pid only printed ss). P31 live gates only had min/max+worst.
- P1→P2 **detonated-freeze last-ok restore** (P32 GPU job, P31 RCA, **P37 GPU-confirmed**): if freeze-iter recon > `skip_storm_last_ok_recon_ratio` × last-ok best, restore `wm_last_ok`, reset AdamW, re-probe gain. Unitless (recon/recon). P31 GPU froze exploded g (val DV ×0.11). P37 extra-P1 iter 88 gnorm 62.4 skip 0; restored iter 87 (0.71@DV). jsonl `p1_last_ok_iter`. Do not lower skip threshold.
- P1/P2 random collect is **numpy-only** (no `torch.inference_mode` / SF `a_history`). P3 on-policy still streams. P2 DOB Kalman scan is a closed-form mix (same recurrence; host-adaptive T×T cap `clamp(GPU_mem/1500, 4MiB, 64MiB)` — A10 24GB still **16 MiB**). Stage-1 skips `apply_dob` clone + ground/reg. CV-std tensor cached. P1 skips unused `agent_finetune_loss` when `reward_scale_loss_p1=0`. Last-ok snapshot `copy_` reuse; **disk persist on lock** (HEAD, not P42 pid); this pid wrote `wm_last_ok.pt` at P1→P2 freeze. Isolation is abs MSE (inv-var helpers deleted). Gain-match FD uses `img_rollout(..., last_only=True, out='obs')` (last-step Huber ≡ `decode(stack[:, -1])`; no unused K-stack; sequential `img_step` fallback **REMOVED**). jsonl splits `wm_gain_match_mv_loss` / `wm_gain_match_dv_loss` (no extra FD). Overshoot uses `out='obs'` and held-rollout `out='h'` (no unused F-stack; last_only materializes `out` once after the K-loop). P1 gate floor is **recent EMA max** (not return to warmup `wm_score_ema_best`; P40 RCA). **P50 RCA (HEAD, not pid):** when live recon is > `skip_storm_last_ok_recon_ratio` × best, the quality-gate / cap-time gain-probe runs on last-ok then restores live (P50 wrap-adjacent MV ×1.37 EXTEND → extra-P1 detonation). Healthy-recon gates unchanged. jsonl emits `wm_score_ema*` and `p1_recon_best`. jsonl always emits `wm_gain_match_loss` / isolation/ss keys (0 when teacher off). Banner `n_grad_skip` is cumulative inner-step skips; skip-storm window is `grad_skip_history` (cleared on restore). HEAD last-ok **locks** after recon > `skip_storm_last_ok_lock_ratio` × best (default 20, unitless; P40 RCA). Call site is recon-only (P41 spike iter 58 skip 1 still locks). Skip-storm restore unlocks. **P48 freeze:** original-P1 wrap recovery also unlocks (`extra_p1=False` and recon back below 20×) so last-ok can advance; extra-P1 recovered basin stays locked (P40). P48 pid locked iter **24** on a recovered wrap (recon 0.145 = 43×, gnorm 5.26, skip 0) and freeze restored 24 (0.81@DV) vs live gate 0.89@DV — wrap unlock was missing. **P50 GPU-confirmed wrap-recovery unlock** (lock iter 56 recon 0.1412 skip 0; unlock iter 58; lock@70 unlock@73). Do not raise lock_ratio or require gnorm/skip (that would miss P41). P41 pid `acb8a7b` did **not** have the lock (overwrite 56→104; val det_r 0.079). P42 EXIT froze last_ok **66** (val det_r 0.124, not P40 0.490).
- `[resolved-cfg]` prints `iso_dcv=off` when isolation/ss-match are off (P40); when the teacher is on it still prints `iso_dcv=` **and** `min_scale=` **and** `mv=`/`dv=` multipliers **and** `edge_du_mv=`/`edge_du_dv=` **and** `g_ratio=`/`smax=`/`equalize=`. Also prints `lock=` (`skip_storm_last_ok_lock_ratio`; P42), `huber_per_in=` (P43 per-input Huber β), `gmatch_settle=` (P44 held settle; dataclass **`-1` off** after storm 2/2 REVERT; `0` auto-resolves to H), `gmatch_rest=` (P45 EXIT default **True**), `gmatch_rest_cg=` (rest-IC CUDA-graph default **True**), `p3_sigreset=` (P46 opt-in; default False), `bc_mean=` (P50 default **True** MSE-on-μ), `p3_sglogstd=` (P51 default **True** stop-grad log_std), `p3_logpclip=` (P52 default **8** REINFORCE logp clamp), `p3_muratio=` (P53 default **0.2** PPO clip vs frozen unfreeze-μ), `p3_murefresh=` (P55 default **1** recopy each P3 iter; 0 = freeze-forever), and `es_ent_floor=` (P54 EXIT default **0.25** σ-band frac). `run_plan.isolation_dcv_scales` is the P38/P39 audit trail (inv-var knob removed). Pre-rewrite dump is dataclass defaults (P29 plan still shows `rssm_latent_type=categorical` / `gain_match_coef=0` even though training auto-enabled grounding).
- P3 collect/val streams `stream_serve_step` (measured DV + Kalman when `dob_active`) so served `feat` matches `_realsim_actor_critic_step` re-encode. **P48 EXIT** KEEP as identity; **FALSIFIED as cascade lever**. P47 EXIT Adam-complete **FALSIFIED as yank lever**. **P50 EXIT** μ-only KEEP as σ-open + first val econ beat (−56 vs −104); **FALSIFIED as yank lever** (unfreeze 169). **P51 EXIT** stop-grad log_std KEEP as unfreeze-yank + non-sticky-floor; **FALSIFIED as cascade lever** (μ-rail; paired −72 vs −111, worse than P50). **P52 EXIT** `p3_logp_clip=8` KEEP as delayed first-unfreeze bound; **FALSIFIED as cascade** (paired −377 vs −87 FAIL; in-support μ-walk). **P53 EXIT** `p3_mu_ratio_clip=0.2` KEEP as μ-walk limiter **and** cascade (paired **−13 vs −98** 9/9; actor champion). **P54 EXIT** ES floor frac KEEP as false-trip / **FALSIFIED as actor-econ lever** (paired −26 vs −101). **P55** recopy μ-ratio each P3 iter.
- **Operator-event schedule** `disturbance_authority_frac=0.65` / `disturbance_recovery_frac=0.20` / `disturbance_settle_steps=0` (auto) / `disturbance_quiet_frac=0.12` are TrainConfig + `ENV_OVERRIDES` (identity; leftover `AGENT_DISTURBANCE_*` still wins when DREAMER is unset). Dual-read at `build_training_disturbance_schedule`. Identifier JSON is process-cached (glob every `env.reset` was host-CPU). Unused curriculum/progressive/sat-monitor/intensity/init-offset helpers **REMOVED** (never called). `AGENT_DYNAMICS_JSON` / `AGENT_LOOKBACK_JSON` stay env-only (paths).
- Reward clip/cal + `DREAMER_DIAG_*` / `DREAMER_RUN_WM_DIAGNOSTIC` / `DREAMER_WM_DIAG_{N_STARTS,HORIZON}` are TrainConfig + `ENV_OVERRIDES` (identity defaults). `DREAMER_WM_DIAG_DEVICE` stays env-only (device picker). Auto-tune formula inputs `seed_target_cv_frac` / `seed_sigma_cap` / `pmpo_entropy_eta_v3` / `pmpo_entropy_sigma_ref` / `prbs_seg_min` / `prbs_seg_min_floor` are the same class (leftover `SEED_TARGET_CV_FRAC` / `PMPO_ENTROPY_COEF_BASELINE` / `PRBS_SEG_MIN` still win when the DREAMER_* field is not explicit). Inner-step counts `train_steps_per_iter` / `phase3_train_steps_per_iter` were already TrainConfig; now in `ENV_OVERRIDES`. **`sigma_min_ratio` default 1.2 now actually resolves** (`_resolve_policy_sigma_bounds`; leftover `max(1.3)` floor SUPERSEDED — P45/P46/P47 env-free used 1.3). Leftover `SIGMA_MAX_*` / `SIGMA_MIN_RATIO_OF_MAX` / `OBJ_REWARD_SCALE` still win when DREAMER not explicit. `DREAMER_OBJ_REWARD_SCALE` / `DREAMER_ATTN_IMPL` / leftover `DREAMER_FAST_ATTN` are in `ENV_OVERRIDES`. **Eval TM protocol** `wm_tf_{levels,span,step_frac,horizon,settle}` + val-suite gates `val_wm_{transfer,postprior,distpred}` are TrainConfig + `ENV_OVERRIDES` (identity; horizon/settle 0 = auto `max(80,4H)`). **Horizon formula** `horizon_settle_n_tau=4.0` / `horizon_max=120` + **IC DR** `init_randomization=True` / `frac=0.6` + **GPU-calib** `wm_overhead=1.30` / `gpu_target_util=0.80` / `gpu_max_bs=512` are TrainConfig + `ENV_OVERRIDES` (identity; `horizon_formula_knobs()` / `ic_randomization_knobs()` / `gpu_probe_knobs()` read TrainConfig then leftover env so changing the dataclass sizes H / IC span / B — BO no longer silently uses WM-only 1.0). `DREAMER_BATCH_SIZE` pins B and skips the probe (leftover `OBJ_BATCH_SIZE` still wins when DREAMER is unset). **Noise / hidden-load** `process_noise_amp_ramp` / `hidden_dist_*` / `hidden_ou_*` / `disturbance_prob_*` are TrainConfig + `ENV_OVERRIDES` (identity; dual-read at `noise_config` / `hidden_disturbance`). **Plant SNR** `sim_noise_adaptive` / `sim_ou_sigma_frac` / `sim_ou_gain_{cv,dv}` / `sim_meas_noise_{cv,dv}_frac` **and wrapper** `sim_noise_enabled` / `sim_noise_seed` / `sim_noise_jitter_pct` / `sim_domain_randomization` are TrainConfig + `ENV_OVERRIDES` (identity ON / 0.008 / 0.15 / 0.60 / 0.005 / 0.010 / jitter 0.20 / DR ON; dual-read at bake+wrap; leftover `SIM_*` still wins when DREAMER is unset). `DREAMER_DERIVED_OBSERVABLES` / `DREAMER_DERIVED_OBS_WINDOW` are TrainConfig + `ENV_OVERRIDES` (identity ON / window 0=auto 2τ/sr). `DREAMER_WM_DIAG_DEVICE` stays env-only. `SIM_NOISE_CONFIG_JSON` stays env-only (path).
- Rest-IC encode length is `lookback` (test_sim **128** sequential GRU steps × 100 inner WM steps → `t_wm` ~124 s). A few-τ window would be more sim-adaptive; that is a later A/B (not identity). **HEAD:** CUDA-graph of that T-loop (`gain_match_rest_ic_cuda_graph=True`, `make_graphed_callables` + GRU-grad canary; RSSM only). GPU identity smoke deferred while pid **95956** holds the A10; CPU path stays the Python loop (`CVD=""`). `DREAMER_GAIN_MATCH_REST_IC_CUDA_GRAPH=0` keeps eager.
- Identified plant `τ`/`θ` are TrainConfig `identified_tau_dominant` / `identified_dead_time` (P49 GPU-occupied; `single_run` writes identifier values; APCEnv caches; leftover `IDENTIFIED_*` still wins when the field is 0). Missing SysID keys no longer invent **50 s / 5 s** (`identify_dynamics` / `noise_config`; SNR window lookback/4). `AGENT_DYNAMICS_JSON` / `AGENT_LOOKBACK_JSON` stay env-only (paths).
- Reward-engine leftovers `objective_integral_{coef,windup,leak}` / `obj_auto_*` / `objective_{penalty,reward}_clip` (sentinel `<0` = adaptive) / `objective_violation_rate_coef='auto'` / `objective_penalty_sat_mode='tanh'` / `objective_feasibility_{cap,scale}` **and remaining `derive_auto_weights` formula knobs** (`obj_auto_{mv,cv}_violation_base` / rank_decay / mv_over_cv / typical_cv 0.10 / move_* / reversal / rate_coef_{div,min,max} / differentiable_depth / reward_clip_floor) **and `objective_use_normalized=True`** are TrainConfig + `ENV_OVERRIDES` (identity; dual-read leftover `OBJECTIVE_*` / `OBJ_AUTO_*` / `OBJ_USE_NORMALIZED` when DREAMER unset; `_explicit_fields` beats leftover). `obj_auto_cv_over_econ_ratio=0` follows margin. APCEnv jitter is dataclass **0.15 / 0.20** (`runtime_setpoint_{bounds,target}_jitter_frac`; leftover `RUNTIME_SETPOINT_*_JITTER_FRACTION`). Remaining schedule knobs **1–2 / 0.10 / 3 / 0.05** (`runtime_setpoint_{bounds,target}_changes_{min,max}` / `ramp_duration_frac` / `curriculum_warmup_frac` / `n_magnitude_strata` / `target_inside_margin_frac`) are TrainConfig + `ENV_OVERRIDES` (identity; `_runtime_setpoint_config_from_cfg`). `auto_derive` jitter is **0.15 / 0.20** (not leftover 0.25); dual-read DREAMER then leftover. Do **not** switch APCEnv to `auto_derive` (τ-derived change-count / ramp are not identity). `resolve_integral_config` / `compute_objective_components` take `cfg`. Auto-weights cache on quiet steps. Dead unused `estimate_reward_scale` **REMOVED**.
- Expert move-law leftovers `expert_move_frac` / `backoff_frac` / `econ_frac` / `loop_gain` / `ridge_frac` / `feas_scale` / `econ_scale` / `opt_iters` / `opt_lr` are TrainConfig + `ENV_OVERRIDES` (identity; dual-read leftover `DREAMER_EXPERT_*` in `apc_expert`; explicit cfg wins). `DREAMER_DYNAMICS_ID_JSON` stays env-only (path). CLI extras `DREAMER_PMPO_{ALPHA,BETA}` / `POLICY_TYPE` / `POLICY_INIT_LOG_STD` / `ACTOR_LOSS` / `GRAD_CLIP` / `MAE_PMAX` / `BASELINE_SEED_{EPS,STD}` / `RANDOM_SEED_EPS` / `EXPLORATION_SEED_EPS` / `DV_PRBS_{SEEDS,OP_FRAC}` / `PRBS_SEED_N_STRATA` / plant-derived `D_MODEL`/`N_LAYERS`/`LOOKBACK`/… are in `ENV_OVERRIDES` (identity; `single_run` used to drop them). Step-settle `|Δu|` / prefix frac, `shaping_safe_margin_frac`, PRBS-seg sentinel, `wm_ss_match_window_frac` are the same silent-drop class (identity 0.20/0.60 / 0.05/0.20 / 0.25 / 0 / 0.34). Do not switch APCEnv to `RuntimeSetpointConfig.auto_derive`.
- Validation `wm_gain_pass` stays **MV-only** (lineage). HEAD also emits `wm_observer_gain_pass` / `wm_observer_gain_healthy` = MV AND DV (no-DV plants copy MV). Prints `[val] WM observer gain:` so P29 HEALTHY-on-MV / DV ×0.56 cannot read as observer-healthy. `wm_dv_gain_*` + `wm_dv_ss_ratio_worst` unchanged (P29 HEALTHY MV rel_err 0.10 hid DV ×0.56; P31 0.35 hid ×0.11; P35 0.064 hid ×0.013).
- P36 EXIT (`run_p36_isoinpscale`): storm 2/2 CAPPED GAIN_NOT_READY 0.00@DV. Val MV ×0.91/@H ×0.95, DV ×0.004, det_r 0.076, `[p3-skip]`. Per-input `|G|²` fired (ratio 33) but inv-var DISCARDED then **REMOVED**. P37 EXIT (`run_p37_isoabs`): abs isolation, env-free, 151 iters. Iter 75/85 EXTEND 0.68 then 0.72@DV; extra-P1 silent detonation iter 88; last-ok iter 87. Val MV ss/@H **×0.981 / ×1.005**, DV **×0.690 / ×0.783**, det_r **0.370**, `[p3-skip]`. Abs isolation KEEP as P1-completing form; **FALSIFIED as DV pin**. Extra P1 FALSIFIED. P38 EXIT (`run_p38_isodcv`, 102 iters): match-at-`g_min` no floor FALSIFIED; val MV **×1.25** DV **×0.007** det_r 0.43 `[p3-skip]`. P39 EXIT (`run_p39_isodcvfloor`, 151 iters): floor 1.0 KEEP as P1 form; cube-boost FALSIFIED; val MV **×0.954 / ×0.954**, DV **×0.679 / ×0.785**, det_r **0.326**, `[p3-skip]`. P40 EXIT (`run_p40_gmatchonly`, 158 iters): isolation-off KEEP as env-free default; **FALSIFIED as DV pin**; val MV **×0.995 / ×0.964**, DV **×0.723 / ×0.743**, det_r **0.490**, `[p3-skip]`. P41 EXIT (`run_p41_recentfloor`, 158 iters): recent-floor KEEP as mechanism; **FALSIFIED as DV pin**; val MV **×0.986 / ×0.973**, DV **×0.700 / ×0.775**, det_r **0.079**, `[p3-skip]`. P42 EXIT (`run_p42_lastoklock`, 158 iters): lock KEEP as 20× fire; **FALSIFIED as DV pin / det_r fix**; val MV **×1.179 / ×1.161**, DV **×0.737 / ×0.804**, det_r **0.124**, `[p3-skip]`. P43 EXIT (`run_p43_gmatchperbeta`, 158 iters): per-input Huber KEEP as P1 form; **FALSIFIED as DV pin**; val MV **×0.985 / ×0.993**, DV **×0.740 / ×0.849**, det_r **−0.215**, `[p3-skip]`. P45 EXIT (`run_p45_restic`, 208 iters): rest-IC **PROMOTE**; first GAIN-READY; val MV **×0.877 / ×0.887**, DV **×0.815 / ×0.875**, det_r **0.148**; valid P3 FAIL econ −216 vs −92 (σ_min). **P46 EXIT** (`run_p46_p3sigreset`, 220 iters): second GAIN-READY; val MV **×0.858 / ×0.854**, DV **×0.793 / ×0.842**, det_r **0.236**; valid P3 FAIL econ **−256 vs −129** (σ opened then re-collapsed). P26 `run_plan` isolation=0 was the pre-rewrite dump (jsonl iso 1.44). Do not pass leftover compile/latent/DOB/`DREAMER_RSSM_IMAG_LATENT_MODE` env-vars.

### Graveyard (P24–P28 observer / actor levers)

| Lever | Tried | Status | Root-cause? |
|---|---|---|---|
| TBPTT `st.detach()` on gain-match asymptote | P24/P25 | REVERTED (full-BPTT) | YES — cut DC-gain gradient |
| Isolation TBPTT `h` only (`keep_c`), never in SS window | P24/P25 | KEPT | YES — bounds GRU without killing DC |
| Isolation TBPTT stride `max(8, round(K/3.5))` | P28 follow-up 2 | KEPT | YES — 16 was absolute steps, not f(K/τ) |
| Huge-grad skip (`wm_grad_skip_norm=1e4`) | P24 | KEPT | YES — garbage clipped step |
| Huber gain-match (abs, β=1) | P23/P26 | KEPT as scalar A/B (`DREAMER_GAIN_MATCH_HUBER_PER_INPUT=0`) | YES — MSE overshoot |
| Per-input Huber β = `|tgt_ij|` (L1 sat ±1) | P43 EXIT | KEEP as P1 form (not P27); **FALSIFIED as DV pin** | YES — gates 82/94/104 all 0.75@DV. Val MV ×0.985/@H ×0.993 DV ×0.740/@H ×0.849 det_r **−0.215** `[p3-skip]`. Freeze last_ok 94 → 0.76@DV. Storm 1/2 @74 recovered. |
| Gain-match FD from PRBS posterior (no held settle) | P26–P43; P44 REVERT | **KEPT** env-free (`gain_match_settle_len=-1`) | YES — Huber~0 at excited IC; TM rest-step 0.75@DV. P44 S=H storm 2/2. Identified G = TM real. DV @H ×0.849 vs ss ×0.740 (4H compounding); MV @H≈ss. |
| Gain-match FD from PRBS + WM held settle S=H | P44 EXIT (`run_p44_gmatchsettle`, `fc18ebf`, 120 iters) | **REVERTED** env-free default `-1` | YES — storm 2/2 @iter 66 after iter-58 G_pred≈0 (Huber sat \|tgt\|-½β). Lock 57 GPU-confirmed. CAPPED 0.76@DV (MV ×0.94). Val MV ×0.926/×0.943 DV ×0.751/×0.842 det_r 0.099 `[p3-skip]`. Extra prior-roll detonated original P1. Do not retry S=H. CPU probe P43 freeze: FD DV×0.969 still ≈×1; TM rest-step remains the gate. |
| Gain-match FD from real rest lookback encode (TM protocol IC) | P45 EXIT (`run_p45_restic`, `cf25923`, 208 iters) | **PROMOTE** env-free `gain_match_rest_ic=True` | YES — first GAIN-READY since P40–P44. Gate 82 0.86@DV. Val MV ×0.877/@H ×0.887 DV ×0.815/@H ×0.875 det_r 0.148. Storm 1/2 @58–59 recovered. Do not settle=`wm_tf_horizon` next. `DREAMER_GAIN_MATCH_REST_IC=0` reverts PRBS-posterior FD. |
| Rest-IC `rollout_observed` stacks unused `(N,L,F)` + Stage-1 `(N,L,n_cv)` zeros | P45 GPU-occupied | SUPERSEDED (`last_only=True`; HEAD `deb0236`, not pid 16548) | YES — rest-IC only needs last state; GRU recurrence identical; smoke last ≡ `stack[:, -1]`. Remaining `t_wm` is T sequential kernel launches (N-independent). |
| Rest-IC last_only still built `post.feat` every t + last Stage-1 zero-`d` tail | P45 GPU-occupied | SUPERSEDED (`return_feats=False`; feat once after the loop when feats are requested) | YES — rest-IC reads `h/z/c_mean` only; GRU identical; smoke last state ≡ full. Does not skip the GRU. Do not concat into main `sample=True`. |
| Rest-IC last_only still ran unused `prior_net` / `cont_prior_net` every t | P45 GPU-occupied | SUPERSEDED (`_posterior_step`; HEAD, not pid 16548) | YES — next GRU input is the posterior; prior heads unused when Kalman/two-pass off (Stage-1 P1). Smoke last `h/z/c_mean` ≡ full obs_step; `prior_net` \|g\|=0; GRU/post_net get grad. P2 `dob_active` still uses `obs_step`. |
| `actor_train_source` other than `realsim` (imagination) | mbrl2 leftover / p01 | **REFUSED** at `train()` start (`_require_realsim_actor`) | YES — skipped `onpol_buf` and trained P3 on shared replay (p01 MV-chatter). Override stays in `ENV_OVERRIDES` so a leftover env is visible then aborted. |
| Relative Huber (`gain_match_relative=1`) | P27 | REVERTED then **REMOVED** (P31 GPU-occupied) | YES — DV ~5× grad, skip-storm abort; A/B path deleted |
| Live `_replay_n_dist` for DOB ground | P25/P26 | KEPT | YES — grounding was dead |
| Full-BPTT gain-match | P26 | KEPT | YES — recovered MV ×0.97 |
| min-of-2 critics + freeze `return_scale` | P27/P28; **P45/P46 valid P3** | freeze **KEEP**; min-of-2 **FALSIFIED as sufficient** | YES — rscale ~2.1 (not 49.5). Warmup rtgt then collapse **also on open σ** (P46). Do not stack more critic knobs. |
| P3 `reset_log_std` (zero last-Linear log_std rows; μ kept) | P46 EXIT; **P47 EXIT** (`run_p47_p3sigadam`, `0acde1d`, 253 iters) | **KEEP-AS-OVERRIDE** (default False) | PARTIAL — residual **is** last Linear (ent −0.101). Opening σ **not** sufficient. P47 Adam-complete unfreeze still yanked 145 **−0.101→−0.336**. Econ **−221 vs −121**. Do not promote. Next: P48 collect DV+Kalman. |
| P3 reset_log_std weight-only (Adam P2 NLL moments live) | P47 EXIT | **FALSIFIED as yank lever** (zero log_std-row moments KEEP as reset hygiene) | YES — first `opt_actor.step` after warmup still yanks σ. REINFORCE on μ+log_std, not leftover Adam. |
| Leftover `os.environ.get` reward clip/cal + DIAG/WM-diag | P46 GPU-occupied → EXIT | SUPERSEDED (TrainConfig + `ENV_OVERRIDES`; identity defaults) | YES — worked but missing from `run_plan`. `DREAMER_WM_DIAG_DEVICE` stays env-only (device picker). |
| Leftover `os.environ.get` auto-tune formula inputs (`SEED_TARGET_CV_FRAC` / `SEED_SIGMA_CAP` / `PMPO_ENTROPY_COEF_BASELINE` / `PRBS_SEG_MIN*`) + inner-step counts not in whitelist | P47 GPU-occupied (pid 29434 P1) | SUPERSEDED (TrainConfig + `ENV_OVERRIDES`; identity defaults) | YES — same leftover-env class as clip/cal. Leftover names still win when DREAMER_* is not explicit. `DREAMER_TRAIN_STEPS_PER_ITER` / `DREAMER_P3_TRAIN_STEPS_PER_ITER` close a silent-drop gap. |
| Auto-tune `max(1.3, sigma_min_ratio)` + dual-read `DREAMER_SIGMA_MAX_*` / leftover `SIGMA_*` beating explicit cfg | P47 GPU-occupied (pid 29434 P1) | SUPERSEDED (`_resolve_policy_sigma_bounds`; leftover `SIGMA_*` only when not explicit) | YES — TrainConfig **1.2** (p10) never resolved; P45/P46/P47 env-free entropy floor **−0.363** = H(σ_max/1.3). Explicit `DREAMER_SIGMA_MIN_RATIO` was also floored. Next env-free launch uses 1.2 (not a P47 A/B). |
| Leftover `OBJ_REWARD_SCALE` + `DREAMER_ATTN_IMPL` CLI-only / constructor `DREAMER_FAST_ATTN` | P47 GPU-occupied | SUPERSEDED (TrainConfig `obj_reward_scale='auto'` + `ENV_OVERRIDES`; FAST_ATTN then ATTN_IMPL; CausalAttention `auto` device-only) | YES — `single_run` silently dropped ATTN_IMPL; OBJ scale missing from `run_plan`. Constructor re-read of FAST_ATTN on `auto` would beat explicit `DREAMER_ATTN_IMPL=auto`. Env-free identity. |
| Leftover eval `DREAMER_WM_TF_*` / `DREAMER_VAL_WM_*` `os.environ.get` | P47 GPU-occupied (pid 29434 P2) | SUPERSEDED (TrainConfig + `ENV_OVERRIDES`; identity) | YES — TM protocol + val-suite gates missing from `run_plan`. Horizon/settle 0 = auto `max(80,4H)` (already sim-adaptive). Levels/span/step_frac unitless. `DREAMER_WM_DIAG_DEVICE` stays env-only. |
| Leftover `DREAMER_HORIZON_SETTLE_NTAU` / `DREAMER_HORIZON_MAX` / `DREAMER_INIT_RANDOMIZATION` / `_FRAC` / `DREAMER_WM_OVERHEAD` | P47 GPU-occupied (pid 29434 P2) | SUPERSEDED (TrainConfig + `ENV_OVERRIDES`; identity 4.0 / 120 / ON / 0.6 / 1.30) | YES — sim-adaptive H formula + IC DR + GPU-calib overhead missing from `run_plan`. Dual-read at `derive_horizon` / sim `reset()`. WM probe now `gpu_probe_knobs()` (TrainConfig; P51 GPU-occupied). |
| P3 reset_log_std weight-only (Adam P2 NLL moments live) | P46 GPU-occupied (pid 24426 weights-only) | SUPERSEDED by P47 EXIT **FALSIFIED** | PARTIAL — hypothesis that first `opt_actor.step` re-collapses from leftover Adam. P47 zeroed those moments and still yanked. |
| P3 collect `_posterior_step`/`obs_step` with `dv=None` `obs=None` | P45–P47; **P48 EXIT** (`run_p48_collectdv`, 236 iters) | SUPERSEDED (`stream_serve_step`); **FALSIFIED as cascade lever** | YES — train/serve match. P48 still cascade (ent −0.283, rtgt 0.059→0.0004, actor_loss −557, mvv 17k/223k). Freeze-24 confound (0.81 vs live 0.89). KEEP serve. Next: wrap-unlock. |
| Leftover `DREAMER_DERIVED_OBSERVABLES` / `_OBS_WINDOW` `os.environ.get` | P48 GPU-occupied | SUPERSEDED (TrainConfig + `ENV_OVERRIDES`; identity ON / auto 2τ) | YES — same leftover-env class as clip/cal. Window 0 = auto. |
| Leftover `DREAMER_PROCESS_NOISE_AMP_RAMP` / `DREAMER_HIDDEN_DIST_*` / `HIDDEN_OU_*` / `DISTURBANCE_PROB_*` `os.environ.get` | P48 GPU-occupied | SUPERSEDED (TrainConfig + `ENV_OVERRIDES`; identity; dual-read) | YES — worked but missing from `run_plan`. Empty leftover ramp still = full noise. `DREAMER_HIDDEN_DIST_MODE` / OU path already deleted. Shape weights `0.5,0.3,0.2`; `p_revert` **0.7**. |
| Leftover `DREAMER_TARGET_UTIL` / `DREAMER_MAX_BS` / `OBJ_BATCH_SIZE` `os.environ.get` | P48 GPU-occupied | SUPERSEDED (TrainConfig `gpu_target_util`/`gpu_max_bs`/`batch_size` + `ENV_OVERRIDES`; identity 0.80 / 512; dual-read at probe) | YES — worked but missing from `run_plan`. `DREAMER_BATCH_SIZE` wins over leftover `OBJ_BATCH_SIZE`. Probe now `gpu_probe_knobs()` (TrainConfig defaults; P51 GPU-occupied). |
| Leftover `SIM_OU_*` / `SIM_MEAS_*` / `SIM_NOISE_ADAPTIVE` `os.environ.get` | P48 GPU-occupied | SUPERSEDED (TrainConfig `sim_noise_adaptive`/`sim_ou_*`/`sim_meas_noise_*` + `ENV_OVERRIDES`; identity ON / 0.008 / 0.15 / 0.60 / 0.005 / 0.010; dual-read at bake) | YES — worked but missing from `run_plan`. `DREAMER_SIM_*` beats leftover `SIM_*`. `SIM_NOISE_CONFIG_JSON` stays env-only (path). Comment that claimed adaptive default-OFF was stale (`get(..., '1')` was already ON). |
| Last-ok original-P1 wrap never unlocks | P48 freeze (`run_p48_collectdv`); P49 skip-storm; **P50 LIVE** wrap@57 | SUPERSEDED (original-P1 wrap recovery unlock; extra-P1 stay locked); skip-storm unlock **GPU-confirmed P45+P49**; wrap-recovery unlock **GPU-confirmed P50**; **FALSIFIED as cascade lever** (P49) | YES — P48 live 0.89@DV @82; freeze restored 24 → 0.81@DV. P49 last_ok **82**. P50 lock@56 recon 0.1412 (97×) skip 0 gnorm 1.20; unlocked wrap-recovery @58; last_ok **66**. Do not raise lock_ratio. |
| P1/P2 expert-BC Gaussian NLL | P45–P49 valid P3; **P50 EXIT** (`run_p50_bcmean`) | SUPERSEDED (`bc_mean_only=True`); KEEP as σ-open + val beat; **FALSIFIED as yank lever** | YES — see P50 EXIT. Next: P51 stop-grad log_std. |
| P3 REINFORCE trains log_std (unfreeze yank) | P50 EXIT unfreeze 169; **P51** (`run_p51_sglogstd`) | SUPERSEDED (`p3_stop_grad_log_std=True` default) | YES — P50 opened σ then REINFORCE yanked −0.107→−0.268 (logp_std 0.56→54). Detach σ in `log_prob_of` / entropy. Opt out `DREAMER_P3_STOP_GRAD_LOG_STD=0`. Not `p3_reset_log_std`. |
| Dead unused `estimate_reward_scale` leftover `REWARD_SCALE` / `REWARD_SCALE_TARGET` | P49 GPU-occupied | **REMOVED** | YES — no callers; leftover env-only. |
| Freeze last_ok 24 (wrap-era observer) as actor cascade confound | P48 EXIT; **P49 EXIT** (`run_p49_wrapunlock`, `4ef9bcb`) | **FALSIFIED as cascade lever** | YES — P49 last_ok **82** ≈ live gate 0.88@DV. Val MV ×0.816/×0.830 DV ×0.867/×0.924 det_r 0.632. Unfreeze 147 still bc 0.002→1.24, mvv 667k, rtgt 0.048→0.0001, ent −0.283. Seed kpi **−714**. Freeze rscale **2.10 KEEP**. Next: P1/P2 BC NLL pins σ_min. |
| Original-P1 wrap last-ok unlock | P49 EXIT; **P50 EXIT**; **P51**; **P52** wrap@81 | KEEP as lock hygiene; **FALSIFIED as cascade lever** (P49); wrap-recovery unlock **GPU-confirmed P50** @57–58/@70–73, **P51** @66–69, **P52** @81 (lock@79 recon 0.331 skip 0; iter 80 0.0554 still locked; unlock@81 recon 0.0226) | YES — P52 freeze last_ok **78** (healthy) not wrap-adjacent. Extra-P1 stay locked (P40). |
| P1 gain-probe on wrap/detonated live g | P50 LIVE gate 82 recon 0.0283 → 1.37@MV EXTEND; **P52 GPU-confirmed** gate 82 | SUPERSEDED (probe last-ok when recon >5× best; restore live); **GPU-confirmed P52** | YES — P52 live recon 0.0306 @82 → last-ok probe iter 78; GAIN-READY 0.95@MV / 0.84@DV. Wrap-unlock@81. Freeze restored 78. Identity when recon healthy (P45–P49, P51). |
| Late `import evaluation.validate` vs already-imported launch-time TM/rssm | P47/P48 val | SUPERSEDED (`pin_eval_modules_at_launch` at train start) | YES — P47 `ImportError: resolve_wm_tf_knobs`; P48 `alloc_pinned_obs_host`. HEAD leftover during a live pid raced val. Pin binds launch-time eval stack. |
| Late `import evaluation.validate` vs launch-time `training_disturbance` | P49 val (`TypeError: get_authority_target_frac(cfg=)`) | SUPERSEDED (pin also imports `utils.training_disturbance`; `cfg=` TypeError fallback; empty paired records **fail** beats_baseline) | YES — P49 launched `4ef9bcb` before pin. Leftover `2ec8345` added `cfg=` to HEAD validate.py. Empty `_dr` → 0.0 vs 0.0 **false all_pass**. Seed kpi −714. |
| Leftover `SIM_PARAM_RANDOMIZATION_PCT` overwritten by identifier bake | P48 GPU-occupied | SUPERSEDED (TrainConfig `sim_param_randomization_pct` sentinel −1=auto + `ENV_OVERRIDES`; leftover/DREAMER win at bake) | YES — wrap applied baked ~0.115 and ignored leftover. Env-free identity stays identifier-derived. |
| Leftover `AGENT_DISTURBANCE_{AUTHORITY,RECOVERY,QUIET}_FRAC` / `SETTLE_STEPS` `os.environ.get` | P49 GPU-occupied (pid 41994 P1) | SUPERSEDED (TrainConfig + `ENV_OVERRIDES`; identity 0.65 / 0.20 / 0.12 / 0=auto; dual-read at schedule) | YES — worked but missing from `run_plan`. Quiet clip 0.5 unchanged. `AGENT_DYNAMICS_JSON` / `AGENT_LOOKBACK_JSON` stay env-only. |
| Late `import evaluation.validate` vs already-imported launch-time TM/rssm | P47/P48 val | SUPERSEDED (`pin_eval_modules_at_launch` at train start) | YES — P47 `ImportError: resolve_wm_tf_knobs`; P48 `alloc_pinned_obs_host`. HEAD leftover during a live pid raced val. Pin binds launch-time eval stack. |
| Leftover `OBJECTIVE_*` / `OBJ_AUTO_*` `os.environ.get` every `env.step` | P49 GPU-occupied (pid 41994 P2) | SUPERSEDED (TrainConfig + `ENV_OVERRIDES`; identity 0.05 / 5 / 0.98 / soft-compensate ON / clip `−1`=adaptive / ratio `0`=follow margin; dual-read leftover when DREAMER unset; explicit cfg wins) | YES — worked but missing from `run_plan`. Integral dead-time damping prefers `identified_tau_dominant` / `identified_dead_time`. Auto-weights cache on `(obj_w id, dims, bounds)` (operator limit-steps miss). Dead `estimate_reward_scale` **REMOVED**. |
| Leftover `derive_auto_weights` `OBJ_AUTO_*` / `OBJ_USE_NORMALIZED` / APCEnv `RUNTIME_SETPOINT_*_JITTER` | P51 GPU-occupied (pid **54815** P1) | SUPERSEDED (TrainConfig + `ENV_OVERRIDES`; identity; dual-read leftover; explicit cfg wins) | YES — P49 covered integral/clip/margin; remaining formula knobs + jitter + `OBJ_USE_NORMALIZED` were still `_env_float`. Typical-CV default is **0.10** (docstring 0.05 was stale). Do **not** `auto_derive`. Not in P51 pid. |
| GPU-calib probe hard-coded env fallback / BO `wm_overhead_factor=1.0` | P51 GPU-occupied (pid **54815** P1) | SUPERSEDED (`gpu_probe_knobs()` TrainConfig 1.30/0.80/512 + leftover env; `pick_batch_size_for_plant` None sentinels) | YES — env-free `single_run` identity (still 1.30). BO omitted overhead and sized against WM-only 1.0. Changing the dataclass now sizes B. Do **not** `auto_derive`. Not in P51 pid. |
| `derive_horizon` / sim `reset()` hard-coded 4.0/120 / ON/0.6 env fallback | P51 GPU-occupied (pid **54815** P1) | SUPERSEDED (`horizon_formula_knobs()` / `ic_randomization_knobs()` TrainConfig then leftover env) | YES — ENV_OVERRIDES recorded the fields but changing the dataclass did not size H or the IC draw. Env-free identity (test_sim H=55). IC knobs do not import `train.py` during plant ID (`sys.modules` guard). Not in P51 pid. |
| `derive_episode_length` hard-coded 20 / 500 / 4000 | P51 GPU-occupied (pid **54815** P3) | SUPERSEDED (`episode_formula_knobs()` TrainConfig 20 / 500 / 4000 then leftover `DREAMER_EPISODE_*`) | YES — same leftover-env class as `horizon_formula_knobs`. Env-free identity (test_sim L=1220). Explicit `SIM_EPISODE_LENGTH` still wins. CVD="" smoke green. Not in P51 pid. |
| P3 REINFORCE stop-grad log_std (μ-only) | P51 EXIT (`run_p51_sglogstd`, `d6a4511`, 361 iters) | KEEP as unfreeze-yank + non-sticky floor; **FALSIFIED as cascade lever** | YES — 147 ent HELD −0.141→−0.101; dip −0.282 @315 recovered −0.106; ES p3_plateau not entropy_collapse. μ still railed (logp_std 0.54→38 @147; mvv 19k@161 / 681k@361; rtgt 0.00015). Val −72 vs −111 BEATS, worse than P50 −56 vs −104; best.pt 161 during cascade. rscale 2.17 KEEP. Do not stack critic knobs. Do not promote `p3_reset_log_std`. Do not revive `actor_kl_coef`. |
| P3 REINFORCE unclipped logp (μ-rail with frozen σ) | P51 EXIT; **P52 EXIT** (`run_p52_logpclip`, `d910ee2`, 209 iters) | KEEP as delayed first-unfreeze bound (`p3_logp_clip=8`); **FALSIFIED as cascade lever** | YES — 147 unclipped std **6.11** / actor **−0.77** vs P51 38 / −4.66. Then in-support SGD walked μ: std **80.8@165 / 100.7@169**; later std **0.88@196** actor **−9.81** mvv **813k**. Val **−377 vs −87 FAIL** (0/9); best.pt 176 det **−634**. Clip cannot stop \|logp\|<8. Next: μ-walk limiter (PPO ratio vs unfreeze snapshot). Opt out `DREAMER_P3_LOGP_CLIP=0`. |
| P3 in-support μ-walk (\|logp\|<8 SGD) | P52 EXIT; **P53 EXIT** (`run_p53_muratio`, `126f011`, 262 iters) | KEEP as μ-walk limiter (`p3_mu_ratio_clip=0.2`); **KEEP as cascade** | YES — logp_std held ~0.47@262 (not P52 80+). Val paired **−13 vs −98 BEATS 9/9** (beats P50 −56/−104). ES entropy-thr vs H(σ_init) still fires — not a cascade fail. Actor champion P53. Opt out `DREAMER_P3_MU_RATIO_CLIP=0`. Next: P54 σ-band ES frac. |
| P3 freeze-forever μ-ratio snapshot (P53 KEEP as walk limiter) | P53 EXIT; **P54 EXIT** (`run_p54_esentband`) | P55 tests recopy every P3 iter (`p3_mu_ratio_refresh_iters=1`); ε=0.2 KEEP | PARTIAL — P53 KEEP as cascade (val −13/−98). P54 extra P3 FALSIFIED as econ lever: `clip_frac` 0.42→0.11, det_return plateau 200 iters, val −26 vs −101. ε=0.2 vs P3-entry μ is a hard BC ball. 0 = freeze-forever. Opt out `DREAMER_P3_MU_RATIO_REFRESH=0`. |
| ES entropy floor margin 0.20 nats (implicit; not TrainConfig) | P52/P53 EXIT; **P54 EXIT** (`run_p54_esentband`, `f6739ac`, 446 iters) | KEEP as ES false-trip fix (`early_stop_entropy_collapse_floor_frac=0.25`); **FALSIFIED as actor-econ lever** | YES — 0/310 P3 iters below new trip −0.238; ES `p3_plateau` not `entropy_collapse`. Val paired **−26 vs −101** 9/9, worse than P53 **−13 vs −98**. Extra P3 after best.pt 246 was a freeze-forever μ-ratio ceiling (`clip_frac` 0.42→0.11). Opt out `DREAMER_ES_ENT_FLOOR_FRAC=0`. Next: P55 snapshot refresh. |
| p136 `actor_kl_coef` / `DREAMER_ACTOR_KL_COEF` (default 0; not wired in `_realsim_actor_critic_step`) | P54 GPU-occupied (pid **95956** P2) | **REMOVED** | YES — FALSIFIED as σ-collapse TR; whitelist was a false A/B (`run_plan` would change, actor would not). |
| `DREAMER_ACTOR_LOSS=pmpo` / `actor_loss_type` (P3 always inlines REINFORCE + μ-ratio) | P54 GPU-occupied (pid **95956** P3) | **REFUSED at `train()`** (false A/B) | YES — `pmpo_loss` / `reinforce_actor_loss` / policy `kl_to` **REMOVED** (no P3 call sites). Whitelist kept so an explicit override fails loud. `_actor_uses_prior_policy` still gates π_prior snapshot (unreachable in env-free). `pmpo_alpha`/`pmpo_beta` unused; stay in `run_plan`. |
| Real-sim P3 jsonl omitted `critic_mc_loss` / `critic_pred_target_r` / `critic_target_v_r` | P54 GPU-occupied (pid **95956** P3; diagnosis skill expected them) | SUPERSEDED (log detached MC CE + Pearson r; training graph unchanged) | YES — `training_diagnostics.csv` already listed the columns; `_realsim_actor_critic_step` never emitted the keys. P54 pid launched before this. CVD="" smoke: `critic_mc_loss>=0`. Not in P54 pid. |
| Rest-IC CUDA-graph canary `zero_grad` wipes in-flight grads | P54 GPU-occupied (pid **95956** P2) | SUPERSEDED (save/restore snapshot; identity when None) | YES — capture is first rest-IC forward so grads are None today; restore is identity. If capture moves later in a WM step, canary must not drop the live graph. CVD="" smoke green. Not in P54 pid. |
| CUDA replay `from_numpy`+`pin_memory` every P1 inner step | P54 GPU-occupied (pid **95956** P1) | SUPERSEDED (reuse pinned host + GPU dest **per slot**; identity) | YES — ~100 inner steps/iter allocated a fresh pinned tensor per key. Slots `replay`/`iso`/`critic` keep live graphs from aliasing (P3 actor on-policy vs critic replay share `obs` shape). Host also per-slot (non-blocking H2D race). Host-adaptive (CPU copies). Not in P54 pid. |
| Rest-IC last_only T-loop = 128 sequential kernel launches × 100 WM steps | P54 GPU-occupied (pid **95956** P1); **P55 LIVE** first GPU capture failed | SUPERSEDED (`gain_match_rest_ic_cuda_graph=True`; `make_graphed_callables` + GRU-grad canary; **P55 HEAD** `cache_enabled=False`) | PARTIAL — P55 pid **103504** capture `RuntimeError: make_graphed_callables does not support the autocast caching`. Eager T-loop (P54 rest-IC class). HEAD nested autocast `cache_enabled=False` records bf16; not in this pid. GPU identity still deferred. `DREAMER_GAIN_MATCH_REST_IC_CUDA_GRAPH=0` keeps the Python loop. |
| SNR window leftover τ=50 s when identifier missing | P52 GPU-occupied | SUPERSEDED (lookback/4 frames; leftover `IDENTIFIED_*` still wins when >0) | YES — 50 seconds is a plant unit. test_sim identity (τ=53 from identifier). |
| Identifier / noise-config invent τ=50 s / θ=5 s when SysID keys missing | P52 GPU-occupied | SUPERSEDED (0 → existing unitless floors; OU θ=0.02) | YES — `identify_dynamics` exported fake `IDENTIFIED_TAU_DOMINANT=50` and sized L/H/sr off a phantom plant. `plant_init.derive_all` already used 0. test_sim identity (keys present). |
| APCEnv schedule knobs dataclass-only / `auto_derive` leftover jitter 0.25 | P52 GPU-occupied (pid **64705** P1) | SUPERSEDED (TrainConfig + `ENV_OVERRIDES` identity 1–2 / 0.10 / 3 / 0.05; `auto_derive` jitter 0.15/0.20) | YES — worked but missing from `run_plan`. `from_spec` unused. Do **not** switch APCEnv to `auto_derive` (τ-derived change-count / ramp). CVD="" smoke green. Not in P52 pid. |
| TrainConfig-only `step_seed_delta_*` / `step_seed_prefix_frac_*` / `shaping_safe_margin_frac` / `prbs_seed_segment_steps*` / `wm_ss_match_window_frac` | P52 GPU-occupied (pid **64705** P1) | SUPERSEDED (`ENV_OVERRIDES`; identity 0.20/0.60 / 0.05/0.20 / 0.25 / 0=auto / 0.34) | YES — `single_run` silently dropped A/B (`STEP_SETTLE_FRAC` / `SHAPING_ECON_MARGIN` / `PRBS_SEG_MIN` siblings were already whitelisted). Auto-tune still honours `_explicit_fields` on PRBS-seg. Isolation/ss-match still off env-free. Not in P52 pid. |
| Leftover `DREAMER_EXPERT_{MOVE,BACKOFF,ECON}_FRAC` / `LOOP_GAIN` / `RIDGE_FRAC` / `FEAS_SCALE` / `ECON_SCALE` / `OPT_{ITERS,LR}` `os.environ.get` in `apc_expert` | P50 GPU-occupied (pid 49389 P1) | SUPERSEDED (TrainConfig + `ENV_OVERRIDES`; identity; dual-read leftover; explicit cfg wins) | YES — worked but missing from `run_plan`. Env-free BC seed path unchanged. `DREAMER_DYNAMICS_ID_JSON` stays env-only. |
| CLI-only `DREAMER_PMPO_{ALPHA,BETA}` / `POLICY_*` / `GRAD_CLIP` / seed counts / plant-derived arch not in `ENV_OVERRIDES` | P50 GPU-occupied (pid 49389 P1) | SUPERSEDED (same keys in `ENV_OVERRIDES`; identity; plant size still constructed first) | YES — `single_run` silently dropped A/B that `python -m training.train` honored via `_CLI_ONLY_ENV`. |
| APCEnv plant τ/θ only via leftover `IDENTIFIED_TAU_DOMINANT` env | P49 GPU-occupied | SUPERSEDED (TrainConfig `identified_tau_dominant` / `identified_dead_time`; cache; leftover when field is 0) | YES — `cfg.tau` was never a field so `_resolve_plant_timing` always env-fell-back. `single_run` / `bo_runner` write identifier values. PRBS seg fallback reads cfg first. |
| Unused `AGENT_DISTURBANCE_CURRICULUM` / `PROGRESSIVE` / sat-monitor / intensity / init-offset mix | P49 GPU-occupied | **REMOVED** (never called) | YES — no call sites; `apply_disturbance_schedule` never passed `mv_monitor`. |
| Identifier JSON glob every `env.reset` | P49 GPU-occupied | SUPERSEDED (process cache on cwd+path env) | YES — static for a `single_run`; P1 ~5 episodes/iter. Schedule RNG identity: quiet `uniform` stays the first draw. |
| Stage-1 unused `sigmoid·d` decay (`d` not a GRU input; `d_t≡0` after loop) | P48 GPU-occupied | SUPERSEDED (skip when `dob_active=False`; RSSM+TSSM) | YES — P1 rest-IC + main WM T-loop, 100 inner steps. P2 Kalman still needs `A·d`. Smoke last_only Stage-1 ≡ full. |
| Rest-IC last_only `unbind` of act/embed/dv | P48 GPU-occupied | SUPERSEDED (`[:, t]` slices) | YES — T extra Python tensor objects on the lookback=128 hot path. GRU identical. |
| Collect/val per-step `from_numpy` H2D obs row | P48 GPU-occupied | SUPERSEDED (pinned host + `copy_obs_row`; identity) | YES — persistent GPU dest already existed; pin stages non-blocking H2D on CUDA. CPU path unchanged. |
| Held-rollout `win=8` absolute steps | P46 GPU-occupied | SUPERSEDED (clamp `(K-1)//4`; identity 8 at K=55) | YES — K=15 made the two windows overlap so the loss returned 0. Unitless cap. Do not retune test_sim 8. |
| `DREAMER_BASELINE_SEED_OP_BAND` via `os.environ.get` inside `train()` | P46 GPU-occupied | SUPERSEDED (`baseline_seed_op_band` + `ENV_OVERRIDES`; const/PRBS bands whitelisted) | YES — env-free still `min(0.6, PRBS)`; explicit override now in `run_plan`. Const/PRBS bands were TrainConfig-only. |
| P1 skip-storm restore `wm_best` + AdamW reset | P28 | KEPT | PARTIAL — recovery without closing ext cap would re-enter exploding P1 |
| P1 skip-storm close `p1_gate_max_ext_steps` on **cap-now** (`_force_p1_cap_at`) | P28 follow-up | KEPT | YES — zeroed ext after setting `p1=now` re-opened the next-iter gate |
| Curriculum freeze on same iter as phase change | P28 follow-up 2 | KEPT | YES — first P2 step still trained `g` |
| P1 skip-storm restore last-ok (not wm_best) | P28 follow-up 3 | KEPT | YES — wm_best can discard late-P1 |
| Skip P1→P2 wm_best reload after skip-storm | P28 follow-up 4 | KEPT | YES — next-iter restore undid last-ok |
| P1→P2 wm_best restore on healthy P1 (default ON) | P28 GPU | REVERTED default OFF | YES — restored iter 60, val MV ×0.52; P26 skipped via min_gap |
| `rssm_latent_type` default `categorical` (env-free) | P29 live | REVERTED default `deterministic` | YES — P26/P28 env never promoted; P29 recon stuck 0.49, kl_loss=0.3, joint_embed≡0 |
| `torch.compile` default-on when `compile_mode=''` | P29 live | REVERTED default eager | YES — same leftover-env class; P26/P28 observer was `DREAMER_COMPILE=0`; P29 compiled |
| P1 inject EVERY = f(buffer lap) | P28 follow-up 5 | KEPT | YES — 20/10 were test_sim iters, not f(τ) |
| P1 inject N = f(n_mv, n_dv) | P28 follow-up 6 | KEPT | YES — 5/2/2/3 were 1-MV+1-DV counts |
| `_cfg_from_env` uses `ENV_OVERRIDES` | P28 follow-up 6 | KEPT | YES — train.py CLI silently dropped 130+ knobs |
| Isolation settle per-input + isolation_buf cap | P28 follow-up 7 | KEPT | YES — 24 was a side-total; cap 48 wrap-killed MIMO channels; MV settle was not isolated |
| Isolation_buf settle-only + clean_steady_seeds on long-hold | P28 follow-up 8 | KEPT | YES — auto-tune-inflated cap wrapped all-DV PRBS into ss-match; settle still had process/meas noise (P89 hole) |
| Isolation settle whole-episode hold at isolated_level, action_std=0 | P28 follow-up 9 | KEPT | YES — T/4 PRBS inside settle + MV dither + `_st_levels` on other MVs starved ss-match settle_var |
| Isolation DV step MV-action-isomorphic (`× span/2`, no extra `dv_prbs_op_frac`) | P28 follow-up 10 | KEPT | YES — extra op_frac shrank \|ΔDV\| vs \|ΔMV\| so abs isolation/ss-match under-trained DV |
| Isolation sample window `max(seq_len, K+1)` | P28 follow-up 10 | KEPT | YES — seq_len < H truncated the SS window on slow plants |
| P1/P2 WM sample `max(seq_len, K+1)`; gain-match K not truncated to T-1 | P28 follow-up 13 | KEPT | YES — isolation-only growth left overshoot/gain-match truncating DC-gain at seq_len-1 |
| Isolation extra unroll skipped when g frozen | P28 follow-up 10 | KEPT | YES — DOB-curriculum P2 cannot update frozen g (dead hot-path) |
| Extra P1 via keep-ext to pin DV DC-gain | P33 EXIT; P37 EXIT; P39 CAPPED; **P41 EXIT 0.74@DV** | FALSIFIED as DV lever (keep-ext KEEP as mechanism) | YES — P33 val DV ×0.66; P37 0.68→0.72 then detonated; P39 extra-P1 stable then CAPPED 0.70@DV; P41 original-budget 0.76 then cap **0.74** (healthy extra-P1, val DV ×0.700 vs P40 ×0.723) |
| Abs isolation/ss-match MSE on mixed MV/DV | P33 EXIT; P37 EXIT | KEPT as default (P36 REVERT); FALSIFIED as DV pin | YES — |tgt| MV 2.82 vs DV 0.49; P37 val MV ×0.98 DV ×0.69. Inv-var (P34–P36) skip-stormed worse. |
| Isolation inv-var (AM/HM, per-seq, per-input `|G|²`) | P34–P36 | DISCARDED then **REMOVED** (P37-live; like relative Huber) | YES — relative-gain reweight; P36 LUT fired (ratio 33) then storm 2/2 @iter 7, val DV ×0.004 |
| Isolation |ΔCV| excitation `Δu_i ∝ 1/|G_i|` match-at-`g_min` (no floor) | P38 EXIT (`run_p38_isodcv`, `42bc7c2`, 102 iters) | FALSIFIED as P1 form | YES — MV edge \|Δu\| 0.19 vs P37 0.60; gmatch stuck 1.30; storm 2/2 @iter 48; val MV ×1.25 DV ×0.007 det_r 0.43 `[p3-skip]`. Same weak-teacher class as P36. Cube-boost of DV did not recover DV. |
| Isolation |ΔCV| excitation with **scale floor 1.0** | P39 EXIT (`run_p39_isodcvfloor`, `92c7662`, 151 iters) | KEEP as P1 form when isolation on; FALSIFIED as DV pin | YES — `|G_max|/|G_min|>1/op_band` cannot equalize \|ΔCV\| without P38 shrink or cube overflow. Val MV ×0.954/@H ×0.954, DV ×0.679/@H ×0.785, det_r 0.326 `[p3-skip]`. Extra-P1 did not detonate. Floor is `wm_isolation_dcv_min_scale`. |
| P08 auto-enable isolation=1 / ss_match=3 on cont-gain | P26–P39 (P26 jsonl iso 1.44; run_plan 0 was pre-rewrite dump) | SUPERSEDED (P40 env-free off; opt-in) | PARTIAL — P26 DV ×0.87 with weak isolation; P28+ whole-ep drowned DV to ×0.66–0.70. P40 isolation-off val DV ×0.723 **FALSIFIED as DV pin**. KEEP as env-free default (MV ×0.995). Gain-match stays the DC supervisor. |
| Env-free isolation/ss-match off (gain-match-only) | P40 EXIT (`run_p40_gmatchonly`, `73a5116`, 158 iters) | KEEP as env-free default; **FALSIFIED as DV pin** | YES — val MV ×0.995/×0.964; DV ×0.723/×0.743 vs P26 ×0.87/×1.00 / P39 ×0.679/×0.785. det_r 0.490 amp-dead. Extra-P1 last_ok overwrite 83→104 confounded freeze. Isolation was not the sole DV regression. |
| `rssm_imag_latent_mode` / `DREAMER_RSSM_IMAG_LATENT_MODE` | imagination-era P70 | **REMOVED** (P39 GPU-occupied) | YES — actor imagination deleted; field was never read by RSSM/TSSM (`sample=` is explicit on observer rolls) |
| Const-action / step-settle all-MVs-equal diagonal | P39 GPU-occupied | SUPERSEDED (`_per_mv_hold_rows`; n_mv≤1 identity) | YES — joint SS on MIMO was the OP-space diagonal; isolation already covers per-input holds. Episode count unchanged. |
| Step-test all-MVs-equal hold + joint MV steps | P39 GPU-occupied | SUPERSEDED (`_per_mv_hold_rows` + one-MV events; n_mv≤1 identity) | YES — scalar `cur_u` filled every MV and each event stepped all MVs (confounded MIMO ∂CV/∂u_i). Faithful auditor matches. |
| SNR min/WARN over all `obs_dim` (incl. constant aug) | P39 log `min=-120dB` | SUPERSEDED (measured CV+DV summary) | YES — `CV0_tgt`/`tgt_on` are constant-by-design; true CV/DV +17.9/+14.8 dB |
| May-2026 `DREAMER_DIAG_PERHEAD_GRADS_EVERY` / `LATENT_STABILITY` default 10 | P39 GPU-occupied | SUPERSEDED (default 0) | YES — extra `autograd.grad(retain_graph=True)` + tokenizer encode every 10 iters; not `ENV_OVERRIDES`; probes belong in scratch. Opt in `=10`. |
| Late-P1 silent detonation (gnorm ~60 < skip 1e4) during extra P1 | P31 iter 95; P37 iter 88 | KEEP detonated-freeze last-ok (skip-storm silent) | YES — applied step explodes g; wrap blips (gnorm ~10) recover. Do not lower skip threshold. |
| Skip post-seed gain-match resolve after pre-iso dcv | P37-live HEAD | REVERTED (always re-resolve) | YES — would freeze WM-norm \|G\| before isolation+expert update cv_std; P38 Huber targets ≠ P37 |
| `mean(err/scale)*mean(scale)` isolation | P34 iter 1 | REVERTED (mean-1 `w·err`) | YES — AM/HM exploded iso 7088 skip 99 ES |
| Per-sequence `|CV|²` inv-var (quiet holds) | P35 EXIT | SUPERSEDED (per-input `|G|²`) | YES — scale_ratio ~22000, iso 0.0008, gmatch stuck 1.21, storm 2/2 cap 0.01@DV; val DV ×0.013 |
| P1 `agent_finetune_loss` when `reward_scale_loss_p1=0` | P33 GPU-occupied | SUPERSEDED (skip except log-last inner step) | YES — paper default 0; 100 unused MTP forwards/iter; jsonl `bc` still from last step |
| Fresh last-ok `state_dict` clone every P1 iter | P33 GPU-occupied | SUPERSEDED (`copy_` reuse) | YES — same weights; first alloc then in-place |
| Sequential `TrajectoryBuffer.sample` Python copy | P33 GPU-occupied | SUPERSEDED (fancy-index) | YES — B=128 windows were a host `for b` copy; same slices; CPU smoke identity |
| Isolation / P3 encode unused logit stacks | P33 GPU-occupied | SUPERSEDED (`store_aux=False`) | YES — no-grad feats-only paths still stacked logits/cont-KL stats (~64MB×2 × 100 inner steps); feats identity-checked |
| Isolation / overshoot / held RSSM-only gate | P33 GPU-occupied | SUPERSEDED (RSSM-interface: rssm + tssm) | YES — TSSM already had `img_rollout`; SF still no-op |
| Stage-1 `apply_dob` clone + ground/reg on `d_t≡0` | P33 GPU-occupied | SUPERSEDED (skip when `dob_active=False`) | YES — constant MSE, no gradient; clone of (B,T,D) every P1 WM step |
| Skip overshoot / held-rollout / gain-match when g frozen | P28 follow-up 11 | KEPT | YES — same dead hot-path inside `world_model_loss`; ~73% of WM step + full-BPTT FD rolls |
| `img_rollout` threads posterior `c` (overshoot/held) | P28 follow-up 12 | KEPT | YES — aux rolls started `c=None` → GRU saw zeros; supervisor ≠ isolation/gain-match/actor path (p20) |
| Open-loop aux start from posterior MEAN c (not feat sample) | P28 follow-up 14 | KEPT | YES — follow-up 12 sliced sampled c from feat; first GRU step still E[f(c_sampled)] ≠ f(mean) |
| Rewrite `run_plan.json → config` after promotions | P29 GPU-occupied | KEPT | YES — start-of-workflow dump is dataclass sentinels; env-free audits would miss auto-enables the same way they missed latent type |
| Honour explicit `DREAMER_DOB_GROUND_COEF=0` | P29 GPU-occupied | KEPT | YES — auto-enable used to re-arm grounding on A/B disable |
| `DREAMER_COMPILE_MODE` not in `ENV_OVERRIDES` | P29 GPU-occupied | KEPT (whitelisted) | YES — CLI-only; `single_run` silently dropped the documented opt-in |
| Inductor workers = ncpu (compile-on) | P29 live | KEPT (cap min(4, ncpu/4)) | YES — 20 workers starved the CPU sim (collect 22.6 s vs P28 17.6 s) |
| P29 live iter banner `sf≡0` / `img_ret` | P29 GPU-occupied | KEPT (print `kl`/`jemb`/`gmatch`/`iso`/`ss`) | YES — leftover latent type was only in jsonl |
| Isolation SS-match folded into iso (no jsonl key) | P28/P29 live | KEPT (log `wm_ss_match_loss` + `wmean` + traj) | YES — settle_var starvation was invisible; training objective unchanged |
| Numpy OpenBLAS all-core (MAX_THREADS=64) | P29 GPU-occupied | KEPT (cap to same nth as PyTorch intra-op) | YES — scipy-openblas64 ignores OMP after import; collect vs WM oversubscription |
| P1 CAPPED skips gain probe if plateau failed | P29 P3 | KEPT (cap-time re-probe) | YES — iter 75 GAIN_NOT_READY 3.05@MV vanished from the cap line; `actor_experiment_valid` would stay True |
| Warn-and-still-train P3 on GAIN_NOT_READY | P28/P29 | KEPT (`skip_invalid_p3=True`) | YES — hours of INVALID actor; observer val still runs on `final.pt` |
| MV-only `wm_gain_pass` hid DV ×0.56 | P29 val | KEEP MV-only field (lineage); SUPERSEDED as printed observer verdict (`wm_observer_gain_*` = MV AND DV) | YES — P29 HEALTHY rel_err 0.10 while DV ss ×0.56 |
| `DREAMER_CRITIC_IMAG_LOSS_COEF` whitelist (imagination CE) | mbrl2 real-sim | **REMOVED** from `ENV_OVERRIDES` (field kept inert) then **REMOVED** field + replay-anchor/`critic_anchor_*`/`critic_mc_tail_bootstrap` (P43 GPU-occupied) | YES — `_realsim_actor_critic_step` never reads them; override was a false A/B. Real-sim grounding is `critic_mc_grounding_coef` (now in `ENV_OVERRIDES`). |
| Gain-match FD clone-loop (`a_j = a_base.clone(); a_j[:, j] += step`) | P43 GPU-occupied | SUPERSEDED (`_gain_match_fd_held` broadcast) | YES — same (1+n_mv+n_dv, Bm, A) stack; smoke identity vs clone-loop |
| Isolation K-step per-decode + overshoot Python MSE loop (eager) | P30-live GPU-occupied | KEPT (batched decode / vectorized MSE, same mean) | YES — extra hot-path forwards; compile leftover was full-model not this |
| Sequential gain-match `_roll` (img_step K-loops) after RSSM+TSSM `img_rollout` | P40 GPU-occupied | **REMOVED** | YES — both RSSM-interface backbones expose `img_rollout`; else branch was dead |
| Gain-match `img_rollout` stacked unused (Bm, K, F) for last-step Huber | P40 GPU-occupied | SUPERSEDED (`last_only=True, out='obs'`) | YES — Huber is ΔCV at step K; GRU recurrence unchanged; smoke last_only ≡ stack[:, -1], batched Huber = seq |
| P1 gate return-to-global `wm_ema_best` (`not_plateaued` EXTEND) | P40 EXIT iter 82 | SUPERSEDED (recent-floor `_p1_fidelity_local_plateau`; **P41 GPU-confirmed**) | YES — warmup spike 6.541 blocked the gain probe at healthy iter 82; extra-P1 silent-detonated iter 84 (gnorm 2.51, skip-storm silent). Extra P1 already FALSIFIED as DV lever |
| P1 gate recent-floor (`_p1_fidelity_local_plateau`) | P41 EXIT CAPPED iter 104 | KEEP as mechanism; **FALSIFIED as DV pin** | YES — gates 82/94/104 all printed gain-probe (P40 `not_plateaued`). FAIL 0.76@DV then CAPPED **0.74@DV** (MV 1.02). Val MV ×0.986/×0.973 DV ×0.700/×0.775 det_r **0.079** (P40 0.490). Extra P1 FALSIFIED as DV lever (0.76→0.74). Extra-P1 healthy. |
| Extra-P1 last-ok overwrite after silent recon spike | P40 EXIT iter 98 (83→104); P41 EXIT original-P1 iter 64 (56→64 then 104); **P42 EXIT freeze 66**; **P45 storm 1/2 unlock**; **P49 storm 1/2 unlock** | KEEP lock as 20× fire; **FALSIFIED as DV pin / det_r-collapse fix**; <5× overwrite-prevention **untested**; skip-storm unlock **GPU-confirmed P45+P49** | YES — P42 locked last_ok **66** @iter 67 (170×) and freeze restored 66. Extra-P1 recon stayed ≥14× so overwrite never opened. Val MV **×1.179/×1.161** DV **×0.737/×0.804** det_r **0.124**. P45 locked 57 @iter 58 (recon 0.2864), skip-storm @59 restored and **unlocked**; last_ok advancing (77). P49 locked 64 @iter 65 (recon 0.3668), skip-storm unlocked; last_ok **82**. Wrap 5–10× untested. |
| Last-ok RAM-only until storm/freeze | P42 (`wm_last_ok.pt` absent after lock; written at P1→P2 freeze) | SUPERSEDED (`_persist_last_ok_ckpt` on lock; HEAD, not pid 4192010) | YES — crash after lock before freeze would lose the snapshot. This pid persisted at freeze. Objective unchanged. |
| jsonl gain-match scalar only | P42 EXIT | SUPERSEDED (`wm_gain_match_mv_loss` / `wm_gain_match_dv_loss` + `p1_recon_best`; HEAD, not pid 4192010) | PARTIAL — abs Huber mean drowns DV when \|tgt\| 2.62 vs 0.51; split is observability (loss unchanged). Do not revive relative Huber. |
| Overshoot/held stacked unused (Bm, K, F) | P40 GPU-occupied | SUPERSEDED (`out='obs'` / `out='h'`) | YES — pointwise decode ≡ batched; held drift only needs `h`; ~2 GB at B=128/starts=24/K=55; last_only materializes `out` once after the K-loop |
| jsonl omit isolation/ss keys when teacher off | P40 live (`73a5116`) | SUPERSEDED (emit 0 + `wm_gain_match_loss` alias) | YES — parsers read None while banner printed `iso 0` |
| Sequential DOB Kalman T-loop (P2 `d_t = A(1−K) d + u`) | P31 GPU-occupied | SUPERSEDED (closed-form mix) | YES — T sequential GPU launches per WM step; same recurrence |
| P1/P2 random collect still entered torch.inference_mode after obs_step skip | P31 GPU-occupied | SUPERSEDED (numpy-only helper) | YES — leftover GPU context + unused SF a_history concat |
| Relative Huber opt-in (`DREAMER_GAIN_MATCH_RELATIVE`) | P27/P28 leftover | REMOVED | YES — proven inferior; field + whitelist invited re-enable |
| `tools/_smoke_*.py` leftover `DREAMER_COMPILE=0` + mbrl-env path | P30-live GPU-occupied | KEPT (stripped; eager is default) | YES — same leftover-env class as P29 launch |
| Cap P1 on first skip-storm (`_force_p1_cap_at` immediately) | P30 | REVERTED default `skip_storm_p1_cap_after=2` | YES — restored last-ok then froze iter 18 of ~90; val MV mean ×1.88 (median ~×1.06 + OP outlier) |
| Continue P1 after first skip-storm (`skip_storm_p1_cap_after=2`) | P32 GPU (storm 1/2 @53) | KEPT | YES — restored last-ok iter 51, recon 0.596→0.003, original P1 budget kept; P30 capped on first; P31 never stormed |
| Close P1 extension on first skip-storm **continue** | P32 P1→P2 | REVERTED (keep ext until storm 2) | YES — healthy recon 0.0026 CAPPED GAIN_NOT_READY 0.71@DV with 0 ext; P31 same iter-75 gate extended; `[gate-budget]` p1_ext_cap was 175924 |
| Sequential isolation `img_step` K-loop | P32-live GPU-occupied | SUPERSEDED (chunked `img_rollout`, same TBPTT keep_c) | YES — compile-on per-step launch; eager img_step count unchanged; CPU identity delta 0 |
| Unused P1 prior-core harvest (`dob_enabled`, `dob_active=False`) | P32 GPU-occupied | SUPERSEDED (skip append) | YES — Stage-1 `d_t≡0` discarded the T-list; extra `prior.feat` slice every t |
| TSSM sequential `_roll` (no `img_rollout`) | P32 GPU-occupied | SUPERSEDED (`img_rollout`, same as RSSM) | YES — gain-match batched FD was RSSM-only; MIMO-width sequential on TSSM |
| Quality-gate CAPPED while recon detonated (skip-storm silent) | P31-live | SUPERSEDED (P1→P2 last-ok restore) | YES — iter 95 recon 0.71 / gnorm 66 / skip 2→3 (<5); cap froze exploded g (DV ×0.08); last-ok was iter ~94 |
| `wm_fidelity_degradation` while curriculum `g` is frozen | P30 | KEPT (suppress in P2) | YES — first P2 probe 5.644 + patience 40 killed DOB; g cannot improve a gain-blind score |











