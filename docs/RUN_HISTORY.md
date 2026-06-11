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

| Run | Date | Change / hypothesis vs prev | Headline result | Verdict |
|---|---|---|---|---|
| p106 | 2026-06-09 | MC-critic + γ0.97 + B3 + DV-input + WM recipe (the proven stack) | gain 0.186, reward_r 0.177, mv_tv 979, cv_viol ~11, econ −30.6 (+69%) | ✅ **BEST baseline (KNOWN-GOOD)** |
| p107 | 2026-06-09 | econ-led `OBJ_AUTO_CV_OVER_ECON_RATIO=1.0` + early-stop 120 | gain 0.094 (best WM) BUT mv_tv 4019, cv_viol 46, econ −108.5 | ❌ FAILED — constraint limit cycle |
| p108 | 2026-06-09 | econ-led 1.0 + integral-boost OFF (single-var ablation of p107) | cycle gone (mv_tv 1777, mv_viol 0) but parks outside limit: cv_viol 72, econ −131, gain 0.183 | ❌ econ-led not ready; p106 stays best |
| p109 | 2026-06-09 | WM-fix levers: recon_cv=6 + excitation=0.4 + P87 dist-head trains (no stop-grad) | gain 0.365 (worse), reward_r −0.086, cv_viol 54, decomp 0.783/–/0.660 | ❌ dist-head flooded the WM, regressed |
| p110 | 2026-06-10 | recon_cv=3 + excitation=0.4 + dist-head **stop-grad** | gain 0.311, reward_r 0.534, cv_viol 54, decomp 0.815/–/0.930 | ❌ still regressed vs p106 (recon_cv backfired) |
| p111 | 2026-06-10 | clean p106 replica (control) | — (killed before completion) | ⏹️ aborted |
| p112 | 2026-06-10 | Gd hidden-disturbance ON (realistic FOPDT load) | gain 0.357, actor best-but-oscillates, mv_tv 1855 | 🔬 omitted-variable confound visible |
| p113 | 2026-06-10 | **Exp A**: hidden-disturbance OFF (ablation) | gain 0.176, real→post 0.940, mv_tv 813 | 🎯 **DECISIVE** — omitted-variable attenuation confirmed |
| p114 | 2026-06-11 | **DOB Scope 1** (neural Kalman filter; d_t output-additive only) | gain 0.365 (NOT recovered), reward_r 0.024, mv_tv 1007, cv_viol 31.5, econ −48.7; decomp 0.798/**1.000**/0.850; dist r **+0.70** R² −0.55 | ⚠️ prior dynamics perfect + dist-corr positive + no oscillation, BUT gain not recovered (autoencoder) + **actor PASSIVE** (mv_viol 0.13 vs p106 35.9) |
| p115 | 2026-06-11 | **DOB + Scope 2** (d_t fed into feat) + excitation 0.6 + recon_cv 4 + P87 head retired | _running_ | ⏳ in progress |

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

### p115 — DOB + Scope 2 + open-loop excitation (IN PROGRESS)
- **Change vs p114**: Scope 2 (RSSMState/TSSMState `feat = [h, z_flat,
  d.detach()]` so the actor/critic/reward heads condition on d_t — explicit
  feed-forward; decoder still reads the clean core) + `WM_EXCITATION_BUFFER_FRAC=0.6`
  (open-loop step-tests de-confound gain↔d_t identifiability — p109/p110 used
  only 0.4 AND had the confounding head) + `WM_RECON_CV_WEIGHT=4.0` (autoencoder)
  + P87 head retired (`DISTURBANCE_LOSS_SCALE=0.0` → `disturbance_head_dim=0`;
  the DOB d_t replaces it).
- **Watch**: gain < 0.186 WITH disturbance on (does open-loop excitation recover
  it once the head confound is gone?); actor ACTIVE again (mv_tv ~979, not ~0);
  dist r stays positive; real→post → 1.0 (recon_cv).
- **Decision rule**: gain < 0.186 → excitation+DOB+Scope2 works, promote + add
  WM-freeze next. gain ~0.31–0.365 → partial-excitation-while-co-training is
  futile → build the pure open-loop WM+DOB **pretrain-then-FREEZE** workstream.
