# Training Run History — neural-apc-mbrl

Lab-notebook ledger of every training run: the **change/hypothesis**, the
**key results**, and the **conclusion/next action**. Maintained by the
`dreamer-training-diagnosis` skill — a new row is appended (or the run's row
updated) at the end of **every** run diagnosis/verdict. Newest at the bottom.

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

