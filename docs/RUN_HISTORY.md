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
| **MV WM gain** | `wm_transfer` ss-ratio (→1.0) | **p128** | **0.967** | dvprbs+econ; regressed to ~0.82 since (dv_ff/D1) — the open p132 target |
| **DV WM gain** | `wm_dv_transfer` ss-ratio (→1.0) | **p131** | **0.868** | the dv_feedforward (p130) + D1-removal (p131) win |
| **Disturbance (Kalman, dynamic)** | `r2_detrended` / `r_detrended` (→1) | **p129** | **0.593 / 0.778** | DETRENDED = control-relevant (see note). p128 ~tied (0.558 / 0.747); p127 amplitude-best (det 0.99). RAW R² is drift-dominated & **misleading** (p129 raw −1.59) |
| **Critic** | `critic_rew_to_tgt_var` (>0.015) | **p131** | **0.0079** | ⚠ NONE healthy yet — all runs <0.015 (bootstrap-dominated); p131 is merely the least-bad |
| **Actor** | `cum_raw` (→0) / `cv_viol` (→0) | **p124** | **−105k / 62.8** | p121 close (−110k / 64.8); wide CIs (±40–66k) so treat as a band, not a point |

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
| p119 | 2026-06-13 | **p117 + TWO independent fixes**: (1) **step-test re-injection in P1** (`STEP_TEST_INJECT`) → fixes the DV→CV gain bias (was 0.62 — DV step-test seeds were evicted before the WM froze); (2) **`DOB_REG_COEF 0.01→0.002`** → fixes the #2 disturbance amplitude (R² was −0.626). Independently measured (DV gain ratio vs disturbance R²) | _running_ | ⏳ DV gain ratio →~1 (+ MV holds) AND dist R²>0, WHILE all_pass + active/smooth actor hold p117 |

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









