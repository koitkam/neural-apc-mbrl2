# GOAL_PLAN — neural-APC-mbrl2

Living plan. Update every visit (live analysis and EXIT). Champions live in `docs/RUN_HISTORY.md`.

**Product:** simulator-agnostic neural APC — smooth CV on the economic limit without violating; faithful observer; unmeasured-load rejection. Envelope: learned observer + neural Kalman/DOB + neural actor-critic. No gray-box plant, no PID/LQR/MPC as the product, no DV-only FF.

Plant this cycle: `nonlinear_sim` HeatExchangerTower (P118 = job 2 of 2, **LIVE P2** after CAPPED GAIN_NOT_READY **0.68@MV**; P119 is the fair repeat of the same mechanism after the MV-Δu unit bug; then return-to-`test_sim`).

## Residual board vs champions (test_sim unless noted)

| Axis | Score we trust | Champion | Last VALID (P116) | This plant P117 | Status |
|---|---|---|---|---|---|
| R1 TM MV ss/@H / curve | val TM, not jsonl ×1 | P26 ×0.973 / @H ×0.880 | ×0.706 / ×0.711 | P117 CAPPED 0.71@DV; P118 freeze **0.68@MV** last_ok **108** — **do not score** (wrong-unit teacher) | OPEN |
| R1 TM DV ss/@H / curve | same | P64 ×0.893 / @H ×0.962 | ×0.692 / ×0.712 | same | OPEN |
| R1 compounding | 1step→OL | P64 ×0.85 | ×0.911 | n/a (INVALID freeze) | OPEN |
| R2 CV smoothness | worst-seed `cv_d2_rms_normed`≤0.05 AND `cv_reversal_rate`≤0.25 | — | P116 `smooth_pass` | n/a | OPEN |
| R2 headroom / viol | `cv_opt_headroom`, `cv_viol_frac` (not a P3-skip gate) | — | hugging 0.35; mv_viol 1.67 | n/a | OPEN |
| R3 Kalman | `det_r` + pred_std vs true | P26 det_r 0.68 | 0.676 / 1.479 vs 1.93 | n/a | OPEN |
| Actor econ | paired vs baseline **and** P64; only if `actor_experiment_valid` | P64 −4.54 vs −94 | −8.24 vs −109.83 VALID | **do not score** | OPEN |

## Metric audit (trusted vs on trial)

**Trusted:** val TM ss/@H/curve_iae (MV and DV); gain decomp real→post / post→1step / 1step→OL; `det_r` + pred_std vs true; paired econ vs baseline **and** champion when freeze GAIN-READY and P3 ran; CV d2/reversal (smooth_pass); `cv_opt_headroom` / `cv_viol_frac`. Skip-storm 5-level TM (`DCgain_ratio` / `@H` / `compound_ok`) is a freeze/RCA diagnostic, not a champ score. Launch `|du_mv|` vs teacher `step` (P119 print): O(step) = WM-norm; ≫5×step = engineering-unit bug.

**On trial / do not GPU-optimize:** jsonl teacher ×1 vs SysID median (tautology on OP-varying plants — P117); **P118 jsonl `gain_match_mv_ratio` vs local G** (teacher Δu was engineering `_prev_control` − WM-norm action → local MV G **0.015** vs identified **2.63**; extra-P1@98 jsonl MV **1.20** and cap last_ok@108 jsonl MV **0.71** are still vs G≈0.015, not TM); ss-ratio alone; mv_reversal as a smoothness gate; CV total variation as a smoothness gate; critic_r without `critic_rew_to_tgt_var`; raw dist R²; VALID 9/9 / GAIN-READY / all_pass / family-closed as “residual closed”; beating P64 to KEEP a smoothness/headroom/DR win; **`wm_grad_norm=inf` at a skip-storm restore** (P118@13) as a kill; **recon-spike ~3 iters after P1 inject** as a kill (13/23/33/45 then **0.935@61** after dv-prbs@60; **0.590@109** after 20× lock@108); **persist_rel spikes to 31** as a kill; **H=1 fidelity r** as a freeze score (`wm_best` @100 is gain-blind; warm-restore SKIPPED); **skip-storm 2 0.02@MV** as a last_ok freeze score; **extra-P1@98 live TM 0.11@MV** as the freeze score (cap used last_ok **108** **0.68@MV**).

P118 live: DV local G **−0.345** vs ident **−0.426** (same space, OK). MV local G is the unit bug, not equal-% OP variation (span **0.015** would be ~2× if the 2.23× SysID story were in WM-norm). LPV `wm_op_scale_dev` **0.52→0.99** then pinned **~0.86–0.91** while Huber MV target ≈0 — do not score LPV from this pid. Freeze TM **0.68@MV / 0.85@DV** is last_ok **108** (recon-best **0.0073**), not live detonated g.

## Closed families (do not N+1)

extra-P1 as freeze (P41; **P117 on this plant**); compounding-teacher Huber (P111 traj unpromoted, P115 k1 REVERT, P116 ol1 REVERT); isolation-off KEEP as default; 2TS-α as champ; kfeat as freeze; quiet_env as freeze; decoder FF; TSSM; GRU keep-h.

## Live this visit — P118 `opscale` (do not kill / do not relaunch / do not second GPU)

tmux `mbrl2_p118` pid **551549** sha **`ff5f84c`** `device=cuda` bs=128 compile=eager nvidia **~4152 MiB** (graph released). Env-free (`CUDA_VISIBLE_DEVICES=0`, no `DREAMER_*`). `[resolved-cfg] opscale=True` no ol1. STAGE 2 `g=84 dob=8` (g **FROZEN**). sps P2 **~220**. jsonl **117** P2. Heartbeat ~28 s/iter. Process 100% CPU. last_ok **108 locked**. `skip_invalid_p3=True` → `[p3-skip]` expected. **Do not score actor.**

**Teacher print (launch):** local G mean MV **0.0153** / DV **−0.345** vs identified **2.627 / −0.426**. jsonl NaN/Inf = **one** `wm_grad_norm=inf@13`. recon best **0.0073@106**. `wm_best` **iter 100** (gain-blind; warm-restore SKIPPED). `wm_op_scale_dev` **0.52→0.99** then **~0.86–0.91**. persist_rel **31@36** recovered; **4.89@98** (extra-P1) recovered.

**P1 gates (phase boundaries):**
- Orig-P1@**87 FAIL** last_ok **85** (live recon **0.1003** > 5× best **0.0120**) median 3/3 worsts **[0.34, 0.33, 0.36]** DC **[0.34, 0.81]** `@H[0.54, 0.80]` worst **0.34@MV** (`not_noisy=False`). DV **0.81/@H=0.80**. Extending to **875688**.
- Extra-P1@**98 FAIL** live recon **0.0138** (healthy — probed live, not last_ok) median 3/3 worsts **[0.06, 0.13, 0.11]** DC **[0.11, 0.75]** `@H[0.04, 0.64]` worst **0.11@MV**. DV **0.75/@H=0.64**. persist_rel **4.89**. Extending to **955296**. extra-P1 lottery **again** worse than orig-P1.
- Cap@**110** last_ok **108** (live recon **0.3364** = **46×** best **0.0073**; 20× lock@108 after recon **0.590@109**). median 3/3 worsts **[0.72, 0.66, 0.68]** DC **[0.68, 0.85]** `@H[0.77, 0.79]` worst **0.68@MV** `unbiased=False` `not_noisy=True` spread **×0.9** signflips=1. DV **0.85/@H=0.79**. Detonated-freeze restored last_ok **108**; re-probe **0.68@MV / 0.85@DV** still GAIN_NOT_READY. Warm-restore SKIPPED. Graph released.

**P2 LIVE @117:** first dobg **0.0209** skip **0** KEEP vs P95. `dob_A` **0.95257** held. `dob_K` **0.119→0.017@111→0.0018@117** SS **≈2.51→0.037** (crush in progress; P114-class early P2 — do not score Kalman yet). leftover `|d_slow|/|d|` **0.80@111→1.53@117**. `std_ratio` **144389@111 → 0.71@117**. recon **0.015→0.010** (g frozen). α **0.00391** (`1/(2T)`). P2→P3 ~step **1.314e6** (~**50** iters / ~25 min). `[p3-skip]` expected.

**Do not treat opscale as FALSIFIED** — MV teacher was not WM-norm. Freeze last_ok **108** **0.68@MV** is recon-best with G_tgt≈0.015, not identified TM. extra-P1 walking last_ok **85 (0.34@MV) → 108 (0.68@MV)** did not GAIN-READY.

HEAD (not this pid): `_wm_norm_realized_du` + snapshot/restore `_prev_cmd_norm` including **None**; launch prints `|du_mv|` vs teacher `step` and WARNINGs if G_MV≪ident **or** `|du_mv|`≫5×step. CVD="" locgfix smoke **PASSED**.

## Next job (after P118 EXIT or hard-fail — not while LIVE)

**P119 `locgfix`** — same mechanism as P118 (LPV + local rest-IC G), **one bugfix**: MV plant-FD Δu = `_prev_cmd_norm` (WM-norm; P61 realized/rate-limit), never engineering `_prev_control`. Snapshot/restore `_prev_cmd_norm` **including None**. Print WARNING if |G_MV| ≪ 5% of identified **or** `|du_mv|` ≫ 5× teacher step. Env-free. No new TrainConfig / no `DREAMER_*`. Tag `locgfix`. Session `mbrl2_p119`. Out-dir `output/nonlinear_sim/run_p119_locgfix`.

Predicted signature: rest-ic local G MV same order as SysID (~2, OP span ~2× not 0.015); `|du_mv|` ~ teacher step **0.4** not ~50; **no** WARNING; jsonl `gain_match_mv_ratio` ~O(1) not ±70 / −129; `wm_op_scale_dev` not pinned at tanh sat; skip-storm Inf@13-class, inject 20× lock@60-class, and skip-storm **0.02@MV** absent or rare; persist_rel not 31; then GAIN-READY vs P118 CAPPED **0.68@MV** / P117 **0.71@DV**.

Falsifier: still CAPPED ~0.68@MV **after** local MV G is WM-norm (WARNING absent; `|du_mv|`~step; jsonl MV ratio O(1)). Do **not** use this pid's 0.68@MV as the P119 falsifier — teacher units were wrong.

Do **not** extra-P1 N+1 / identity-relaunch P118 / rewrite the live recipe.

After a fair P119 VALID or a fair GAIN_NOT_READY with a **correct** local teacher: return-to-`test_sim` on the worst residual (cadence: 2 jobs on this plant then return). P118 is job 2 of 2 **but** the unit bug means P119 is the fair repeat of the same mechanism, not a third plant-hop.

## Ranked follow-ups (not this GPU)

1. **P119 locgfix** (above) — causal for R1 on this plant. HEAD is launch-ready.
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
- Signal: jsonl NaN/Inf = **one** skip-storm `wm_grad_norm=inf@13`. P2 first dobg **0.0209** skip **0** KEEP vs P95. GPU P2 **4.2 GB**. Disk `/home` 64% / 62G. P118 process **alive** (unlike P117 SIGKILL mid-P2).
- Control: do not score actor (`skip_invalid_p3`; freeze GAIN_NOT_READY).
- ML: LPV `wm_op_scale_dev` sat chasing a ~0 MV Huber target — not a test of gain-c G(op). Inf@13, persist 31, 20× lock@60 then **@108**, skip-storm 0.02@MV, extra-P1 0.11@MV are teacher-unit. `wm_best` @100 gain-blind — SKIPPED at P2. last_ok **108** recon-best with wrong G; freeze **0.68@MV / 0.85@DV** is not a P119 falsifier. extra-P1 live@98 **0.11@MV** was worse than orig-P1 last_ok **0.34@MV** — lottery closed.
- Plant: HeatExchangerTower `step` is engineering; APCEnv denorms. Teacher must use `_prev_cmd_norm`. Restore must not leave a post-FD leftover when the snapshot was `None`.
- Metric: jsonl MV ratio vs 0.015 is not val TM. Cap 0.68@MV is last_ok TM vs identified, still GAIN_NOT_READY because the teacher never trained identified-scale MV G.

P117 process died mid-P2 (SIGKILL). P118 P2 is healthy so far. Do not identity-relaunch P118.
