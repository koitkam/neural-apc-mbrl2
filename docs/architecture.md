# neural-apc-mbrl — World-Model + Actor-Critic Architecture

Living architecture reference for the model-based APC controller. Keep this in
sync with the code when the data flow changes (it is part of the repo on
purpose). Backbone-agnostic: the **RSSM** (default) and **TSSM** (transformer,
opt-in via `DREAMER_WORLD_MODEL_TYPE=tssm`) are duck-compatible — `TSSMState`
mirrors `RSSMState` (`.h`, `.z_logits`, `.z`, `.feat`, `.stoch_flat`) and both
expose `obs_step` / `img_step` / `decode` / `rollout_observed`, with
`feat = cat([h, stoch_flat])` and `decode(feat) → obs`.

Status legend: **[current]** = implemented & default · **[opt-in]** = implemented,
env-gated off · **[planned]** = designed, not yet built.

> **2026-06-11 (status 2026-08):** the neural-Kalman-filter / DOB disturbance
> observer (§3) is implemented in both backbones and is **default ON**
> (`TrainConfig.dob_enabled=True`; opt out `DREAMER_DOB_ENABLED=0`). Exp A
> (p113) showed the unmeasured load was an omitted variable attenuating the
> gain (hidden disturbance OFF recovered WM gain 0.36→0.18 and autoencoder
> real→posterior 0.77→0.94) — exactly what the DOB de-confounds.

> **2026-06-22 [SUPERSEDED 2026-08-18 — see below]:** the **continuous
> gain+disturbance latent (§3b, C3)** supersedes the DOB as the default direction
> (`DREAMER_CONT_LATENT_ENABLED=1`). One Gaussian latent fixes BOTH the
> subdominant DV-gain categorical-attenuation bias (≈0.85, via the C(1)
> gain-match) AND the unmeasured-disturbance estimate (an inherent amortized
> Kalman, **no DOB**). The DOB stays in the code as a one-flag fallback
> (`DREAMER_DOB_ENABLED=1`) until the cont disturbance is verified to recover
> (detrended r ≥ the DOB's 0.354). First run: p137. *(The cont **disturbance**
> channel FAILED to recover — p137–p141 held-out r ≤ 0 — and was reverted at
> p142; the cont latent now keeps only the **gain** block. See the 2026-08-18
> note.)*

> **2026-07-07 — neural-apc-mbrl2 fork (REAL-SIM controller):** the actor is no
> longer trained in WM imagination. **Imagination is deleted** (`imagination_step`
> / `_imagination_step_rssm`, ~880 lines removed). The WM (RSSM) + DOB are now a
> **frozen OBSERVER only**; the **actor-critic trains on λ-returns from REAL
> rollouts of the true simulator** (`_realsim_actor_critic_step`) with **domain
> randomisation** enabled at P3 (`set_domain_randomization(True)`). This removes
> the objective-mismatch / model-exploitation that drove the p106→p143 actor
> failures, grounds the critic in **real returns** (an MC anchor over full real
> episodes + a diverse-replay critic keep the value head well-conditioned; no
> `return_scale` cascade), and keeps DreamerV3's scale-invariant normalisation
> (symlog/twohot/percentile ⇒ fixed hyperparameters across sims). The
> imagination-specific parts of §1 (the Phase-3 rollout, imagination gain-rand)
> are **superseded**; the WM losses (recon/KL/DOB + overshoot/held-rollout) are
> KEPT — they train the OBSERVER, not the deleted imagination actor.
> `actor_train_source='realsim'` (default).

> **2026-08-18 — observer sharpened (GAIN + DISTURBANCE).** Terminology note: the
> RSSM is still a **world model** (a learned plant-dynamics model `g`/`f`; the
> `wm_*` diagnostics show it *imagining* open-loop step responses to measure its
> fidelity). Its **role** in this fork is a **frozen OBSERVER** (state +
> disturbance estimator) that the real-rollout actor-critic reads. Three updates:
> **(1) DOB reinstated as the default disturbance estimator (p142).** The
> continuous *disturbance* latent (2026-06-22) was dropped — it competed with the
> DOB for the same CV innovation (the gain↔disturbance identifiability confound)
> and never recovered (p137–p141 held-out r ≤ 0). The cont latent now keeps ONLY
> the **gain** block (the C(1) gain-match de-confounder); the neural-Kalman
> **DOB `d_t` owns the unmeasured load** (`TrainConfig.dob_enabled=True` auto-selects the
> GAIN-ONLY cont latent). **(2) Option 1 — symmetric per-input steady-state
> DC-gain ID (p18).** A first-class, input-symmetric `wm_ss_match` DC-gain
> objective (settledness-gated via `wm_ss_match_settle_var`, MV & DV identical)
> makes the observer's gain **bias-free** on the linear plant (MV ×0.92, DV ×1.14,
> ss ≈ at-horizon ⇒ stable open-loop). **(3) P19 — KalmanNet-style DOB grounding**
> (`DREAMER_DOB_GROUND_COEF`, §3). P18 showed the DOB *under-tracked* the load
> (d_t vs true r=0.42, ~0.3× amplitude): in Stage-2 the recon innovation alone
> under-drives `d_t` (the slow load is a small share of the recon MSE) and
> `dob_reg` opposes it, so imagination + the critic stayed disturbance-blind and
> the actor collapsed below the open-loop baseline. Grounding adds a direct
> `dob_ground_coef · ‖d_t − true_load‖²` (unit-matched via the running CV
> obs-norm std) that tunes the Kalman `A,K` to TRACK the load — the structural fix
> for the manual `dob_gain_init` amplitude tuning. `dob_reg_coef → 0` when
> grounding is on. First run: p19 (`run_p19_dobground`).

> **2026-08-24 — P25 RCA / P26 observer fixes (branch `grok`).** Two silent
> bugs made the observer look trained while staying biased. **(1) TBPTT killed
> DC-gain.** P24's `st.detach()` every 16 of K=55 on the gain-match *asymptote*
> roll left the forward Huber loss tiny but cut the gradient through the rise
> that sets transfer-matrix gain (P25 freeze MV ×0.74 / @H ×0.60). Gain-match
> now runs **full BPTT** (explosion defence = Huber + `wm_grad_skip_norm`);
> isolation may still TBPTT **`h` only** (`keep_c=True`) and **never inside the
> SS-match settle window** (applied *between* `img_rollout` chunks so
> compile-on fuses each chunk; eager `img_step` count unchanged). **(2) DOB grounding was dead code** whenever the
> DOB replaced the read-out head: `disturbance_head_dim` is forced to 0, the
> replay buffer stored `n_dist=0`, `batch['dist']` was missing, and
> `dob_ground` was identically 0 for every P2 iter of P19 *and* P25. Buffer
> width is now `n_cv` whenever DOB or `dob_ground_coef>0` (`_replay_n_dist`).
> Stage 2 (critic ensemble + freeze `return_scale`) waits until the P1→P2
> gain-probe is READY.

> **2026-08-25 — P26 verdict / P27 FAIL / P28 (branch `cursor`).** Observer GAIN
> was healthy on P26 (MV ss/@H ×0.97/×0.88) with residual DV ss ×0.87. **P27
> aborted in P1** (`grad_skip_storm` @iter 50, `wm_grad_norm` 6e12, recon
> 0.004→0.50, 42 skips). Relative Huber (`gain_match_relative=1`) over-weighted
> the small DV target and exploded full-BPTT; P2/P3 never ran. Validation
> `final.pt` is the **P1 expert-BC policy** (econ −59 vs baseline −77 looks
> “better” than P26’s cascaded RL, but `critic_r`/`reward_r` are NaN and MV
> gain is ×1.95). **P28:** revert relative gain-match to absolute (P26
> observer); keep min-of-2 critics + freeze `return_scale` (still untested);
> on a P1 skip-storm restore `wm_best.pt`, reset AdamW, cap P1 and continue
> to Stage 2 instead of aborting. **P28 follow-up:** the original cap only
> zeroed `p1_ext_steps`, which re-opened the P1 quality-gate extension and
> would re-run exploding full-BPTT gain-match. Recovery now also closes
> `p1_gate_max_ext_steps` (`_force_p1_cap_at`) so the next iter is Stage 2.
> **P28 follow-up 2:** closing the extension cap was not enough — the
> curriculum freeze latched at *loop start*, so the first P2 train step
> still had `g` trainable (one extra full-BPTT gain-match on the restored
> weights). Freeze/DOB now re-apply on the same iter as `current_phase`
> changes. Isolation TBPTT stride is sim-adaptive (`max(8, round(K/3.5))`,
> 16-of-55 on test_sim); Huber β is not re-read from env on the hot path
> (that undid `<=0` auto-median).
> **P28 follow-up 3:** skip-storm recovery used to restore fidelity-peak
> `wm_best.pt` (gain-blind EMA of correlation + std-ratio). P27's last
> healthy observer was iter 49; an early lucky `wm_best` would have
> discarded that late-P1 excitation. Recovery now keeps an in-memory
> last-ok snapshot on every skip-free P1 iter whose recon is within
> `skip_storm_last_ok_recon_ratio` (default 5×, unitless) of the best
> recon, and restores that first (writes `wm_last_ok.pt` only if a
> storm fires). Fallback is still `wm_best`. Summary
> records `actor_experiment_valid=False` when the freeze is
> `GAIN_NOT_READY` or skip-storm fell back to `wm_best`. Default
> `skip_invalid_p3=True` then **skips P3** (`[p3-skip]`,
> `early_stop_reason=p3_skipped_invalid_observer`); observer
> validation still runs on `final.pt`. P29 hid GAIN_NOT_READY
> behind a later `not_plateaued` cap (no freeze-time probe) so P3
> still started — the cap path now always re-probes gain.
> Opt out `DREAMER_SKIP_INVALID_P3=0`.
> **P28 follow-up 4:** last-ok restore was still undone on the next
> iter. At the time, default `DREAMER_WM_BEST_RESTORE_AT_P2=1` reloaded the
> fidelity-peak `wm_best.pt` at the P1→P2 boundary (healthy-P1 win,
> p124). **P28 GPU reverted that default to False** (see below). After a skip-storm the next iter *is* that boundary, so
> last-ok was overwritten by the same gain-blind spike follow-up 3
> exists to avoid. Recovery now skips the P1→P2 (and P2→P3) wm_best
> reload when `skip_storm_p1_recovered`. Healthy P1 still restores
> only if that knob is opted back on.
> **2026-08-26 — P28 GPU VERDICT / P29 / env-free latent drop.** `run_p28_absgain_qens` (@ a7941be,
> before follow-ups 1–14) restored gain-blind `wm_best` iter 60 on a
> *healthy* P1 (0 skip-storms). Val MV ss/@H ×0.52/×0.48 vs P26 ×0.97/×0.88.
> Follow-up 4 (skip restore only after skip-storm) would not have stopped
> this. **P29:** `wm_best_restore_at_p2` default **False** — freeze
> end-of-P1 `g`. Opt in with `DREAMER_WM_BEST_RESTORE_AT_P2=1`. Freeze
> `return_scale` did pin 2.29 (KEEP) but the P28 actor experiment is
> **invalid** (GAIN_NOT_READY).
> **P29 live (confounded):** env-free launch dropped
> `DREAMER_RSSM_LATENT_TYPE=deterministic` (P26/P28 env, never a
> TrainConfig default). P29 is **categorical** (`kl_loss` pinned at
> free_bits, `joint_embed_loss≡0`, recon 0.50→0.28 by P1 iter 30 vs
> P28 0.004; recon spiked 0.18→0.65 at iter 49 after buffer wrap).
> Not a skip-restore A/B. **Default is now `deterministic`.**
> `z_dim=32` / `zrank/1024` does **not** distinguish the two (same
> stoch_flat width). Read `kl_loss` vs `joint_embed_loss` or the
> train-start `latent=` banner. After gain-match resolve, train.py
> rewrites `run_plan.json → config` and prints `[resolved-cfg]` so
> env-free audits see auto-enabled `gain_match` / `dob_ground` /
> isolation, not the dataclass sentinels. P29's on-disk plan is the
> *pre-rewrite* dump (`rssm_latent_type=categorical`).
> **Compile leftover (same class):** `TrainConfig.compile_mode=''` was
> documented off, but `build_model` treated empty as default-on unless
> `DREAMER_COMPILE=0`. Env-free P29 compiled; P26/P28 (no compile
> banner) were eager. Default is now **eager**; opt in
> `DREAMER_COMPILE=1` or `DREAMER_COMPILE_MODE=default` (both now in
> `ENV_OVERRIDES` — `single_run` used to silently drop `DREAMER_COMPILE_MODE`).
> Train-start / `[resolved-cfg]` print `compile=`. Compile-on caps
> inductor worker threads to `min(4, ncpu/4)` so the CPU sim is not
> starved (P29: 20 workers on 20 cores, collect 22.6 s vs P28 17.6 s).
> **P29 EXIT=0 (2026-08-26):** categorical leftover + compile-on. Val MV
> ss/@H ×1.10/×1.23, DV ×0.56/×0.65 (autoencoder), det_r 0.725 (amplitude
> poor). Skip-restore DID fire — still not an A/B. Actor INVALID
> (`actor_experiment_valid` bookkeeping True; entropy-collapse @234; econ
> −411 vs −83). Freeze `return_scale` 1.48→4.85 (KEEP). HEAD already has
> deterministic + eager + `skip_invalid_p3`. **P30 LIVE** (`run_p30_deteager`,
> HEAD `2f0aec9`, 2026-08-26 16:01): env-free CUDA only. Train-start
> `latent=deterministic compile=eager skip_invalid_p3=True device=cuda bs=128`.
> `[resolved-cfg]` same + `gain_match=1 isolation=1 ss_match=3 n_critics=2
> rs_freeze=True restore_p2=False`. No leftover `[env-override]`, no
> `torch.compile` banner. Validation now logs `wm_dv_gain_*` so MV-only
> `wm_gain_pass` cannot hide a biased DV. **P30 P1 iter 17 (2026-08-26 16:55):**
> recon 0.0098, `kl=0`, `jemb=0.036`, alive 1024, skip 0 — tracking P26/P28
> (not P29 leftover). GPU occupied; no second job. HEAD eager-path opt
> (batched isolation decode + vectorized overshoot MSE) is for the *next*
> launch; the live process is still `2f0aec9`.
> **P30 EXIT (2026-08-26 ~17:36):** skip-storm @iter 19 restored last-ok
> (KEEP — P2 recon 0.0125) then **capped P1 at iter 18**, throwing away
> remaining original P1. Cap-time DC 1.44@MV. P2 `wm_fidelity_degradation`
> then killed DOB after ~40 iters (g frozen; first P2 probe cannot
> improve). Val MV mean ×1.88 (median ~×1.06 + OP outlier); DV ×0.77;
> det_r 0.05. Actor never ran (`skip_invalid_p3`). **P31:** first
> skip-storm **continues original P1** (`skip_storm_p1_cap_after=2`);
> storm 1 keeps the quality-gate extension (P32 CAPPED 0.71@DV when
> it was closed); storm 2 still `_force_p1_cap_at`. Storm-time
> GAIN_NOT_READY does not stick if P1 continues. Fidelity ES suppressed
> while `_dynamics_g_trainable` is false.
> **P31 EXIT (2026-08-26 23:23, 151 iters):** storm_cap=2 KEEP (healthy
> P1 to iter 94; no skip-storm). Iter 95 detonated (recon 0.71, gnorm 66,
> skip 2→3). CAPPED froze exploded g — val MV ×1.35, DV ×0.11, det_r
> 0.11. `[p3-skip]` KEEP. **P32** (`run_p32_detfreeze`): P1→P2 restores
> last-ok when freeze recon is detonated vs last-ok best (same 5×
> ratio); re-probes gain. Also batched gain-match FD, numpy P1/P2
> collect, vectorized DOB Kalman, relative Huber **removed**.
> **P32 EXIT (2026-08-27 03:22, `run_p32_detfreeze`, 129 iters):**
> skip-storm 1/2 @iter 53 KEEP. P1→P2 CAPPED GAIN_NOT_READY 0.71@DV
> (healthy recon 0.0026; continue closed extension). Val MV ss/@H
> ×1.088/×1.098, DV ×0.675/×0.754, det_r 0.109. `[p3-skip]` KEEP.
> **P33 EXIT (2026-08-27 08:15, `run_p33_keepext`, 151 iters):**
> keep-ext **used** (iter 75/85 EXTEND, iter 97 CAPPED after 175924).
> No skip-storm. Val MV ss/@H ×1.083/×1.093, DV ×0.660/×0.770,
> det_r 0.398 (amplitude still dead). `[p3-skip]` KEEP. Keep-ext KEEP
> as mechanism, **FALSIFIED as DV lever**. Abs isolation drowns DV
> **P36 EXIT (2026-08-27 11:44, `run_p36_isoinpscale`, 61 iters):**
> per-input `|G|²` fired (`tgt_scale=1`, scale_ratio 20–33) but inv-var
> **DISCARDED**: iso 0.125 vs P33 1.69, gmatch stuck 1.21, storm 2/2
> @iter 7, val MV ×0.91 DV ×0.004 det_r 0.076. `[p3-skip]`. Same class
> as P27 relative Huber (33× DV isolation grad). Abs isolation is the
> only path (inv-var A/B **REMOVED**, same as relative Huber). Do not
> try another isolation reweight. P37 EXIT (`run_p37_isoabs`): val MV
> ×0.98 DV ×0.69 det_r 0.37 `[p3-skip]`. Abs isolation KEEP as P1 form;
> FALSIFIED as DV pin. Next GPU: env-free P38 `run_p38_isodcv`.
> **P28 follow-up 6 (no GPU this session):** inject **N** was a raw
> episode count (const 5 / step-test 2 / DV-PRBS 2 / expert 3). That is
> one 1-MV+1-DV shot. `_resolve_inject_cadence` now also sets
> `max(sentinel, f(n_mv,n_dv))` so distillation (4 MV + 1 DV) injects
> 5/5/4/4. test_sim stays 5/2/2/3. **Also:** `training.train._cfg_from_env`
> now applies the shared `ENV_OVERRIDES` whitelist (it had silently
> dropped 130+ knobs including skip-storm / aux TBPTT / `n_critics`).
> Cascade / grad-skip early-stop env-vars were missing from the
> whitelist entirely (ignored by `single_run`).
> **P28 follow-up 7 (no GPU this session):** isolation settle was a
> flat 24 (all MVs PRBS'd together, all DVs swept together) and the
> isolation ring-buffer cap was 48 — MIMO wrap-killed per-channel
> settle. `wm_isolation_settle_episodes` is now **per isolated
> input** (test_sim 24+24 / cap 48 unchanged). Distillation 4+1
> emits 96+24 and the cap grows to 120. Long-hold MV settle holds
> every other MV, suppresses curriculum DV + hidden OU, and DV
> settle isolates one DV channel.
> **P28 follow-up 8 (no GPU this session):** isolation_buf still mixed
> MIMO PRBS + all-DV PRBS with settle. Follow-up 7 sized the cap
> `max(baseline+dv_prbs+8, settle)` so dataclass test_sim stayed 48,
> but `auto_tune_seed_buffer` raises baseline ~16→26 → cap 58, and
> wrap kept ~10 confounded all-DV episodes in front of the 48 settle
> (errors-in-variables / gain↔disturbance on `wm_ss_match`). Cap is
> now settle-only; ordinary PRBS stays in the main replay buffer.
> Long-hold settle also uses P89 `clean_steady_seeds` (process OU +
> measurement noise off) — const-action/step-settle already did;
> isolation settle is the buffer that actually trains DC-gain.
> **P28 follow-up 9 (no GPU this session):** follow-up 8 still
> PRBS-stepped *inside* each settle episode (seg capped at T/4 → ~11
> holds of 2K on test_sim) and dithered the isolated MV with
> `baseline_seed_std`, and wired `_st_levels` to *other* MVs (`hold_level`,
> a no-op on test_sim n_mv=1). Random `seq_len` windows from
> `isolation_buf` straddled those steps (~half) so `wm_ss_match`'s
> `settle_var` gate starved the DC-gain term. Isolation settle is now a
> **whole-episode constant hold** at the stratified `isolated_level`
> (`action_std=0`, others at 0). DV settle is one step at t=0 to
> `isolated_level × (span/2)` (MV-action units; follow-up 10 dropped the
> extra `dv_prbs_op_frac` shrink), MV held at 0.
> **P28 follow-up 10 (no GPU this session):** follow-up 9 reused
> `dv_prbs_op_frac` as the DV isolation amplitude (`delta = isolated_level
> × op_frac × span/2`). Isolation linspace is already in
> `[-constant_action_seed_op_band, +…]` like the isolated MV, so DV steps
> were 0.8× smaller than the matching MV action → smaller |ΔCV| →
> absolute isolation/ss-match MSE under-trained DV (same family as
> abs-Huber on unequal |tgt|). Isolation DV step is now MV-action-
> isomorphic (`±1 ↔ ±half-span`). Isolation sample windows are
> `max(seq_len, K+1)` so a slow plant with H > seq_len still reaches SS
> (test_sim seq_len ≥ H unchanged). Isolation extra unroll is skipped
> when `g` is frozen (DOB curriculum P2 — dead hot-path forward).
> **P34/P35/P36:** even with isomorphic |Δu|, WM-norm |ΔCV| still differs
> by ~|G| (test_sim gain-match |tgt| 2.82 vs 0.49). Isolation inv-var
> (`wm_isolation_var_norm`) tried to equalize that and **failed three
> formulas** (P34 explode, P35 quiet-hold, P36 33× DV skip-storm).
> Abs MSE is the only path (A/B **REMOVED**). Default
> `wm_isolation_dcv_match` scales isolation **excitation**
> `Δu_i ∝ 1/|G_i|` then clips `level*scale` to ±1. Scale is
> **floored at 1.0** so the strong-|G| teacher stays at op-band.
> Logged `scale` is a **multiplier** on
> `isolated_level ∈ [−op_band,+op_band]`; applied edge |Δu| is
> `min(1, op_band·scale)` (`run_plan.isolation_dcv_scales.edge_du_*`).
> P38 EXIT (`run_p38_isodcv`, **no floor**, 102 iters): MV scale 0.317 →
> edge |Δu| **0.19** (P37 0.60); DV scale 1.67 → edge |Δu| **1.0**.
> Match-at-`g_min` starved MV isolation SNR. Storm **1/2 @iter 45**
> (last-ok 13) then **2/2 @iter 48** CAPPED GAIN_NOT_READY **0.01@DV**
> (MV 1.12); gmatch stuck **1.30**; val MV ss/@H **×1.253 / ×1.293**,
> DV **×0.007 / ×0.007**, det_r **0.428** amp-dead, `[p3-skip]`.
> **FALSIFIED**. HEAD/P39 floor keeps MV edge |Δu|=0.60 and still
> boosts DV to the cube. P39 LIVE (`run_p39_isodcvfloor`, pid 4170377):
> `[resolved-cfg] min_scale=1.0 edge_du 0.60/1.0`; P1 iter 22 gmatch
> **0.0008** skip 0 (P37-class, not P38 stuck 1.30). Equal-|G| plants keep the op-band
> linspace.
> **P39 GPU-occupied (no second job):** const-action / step-settle
> seed used one linspace level on **every** MV (OP-space diagonal).
> Isolation settle already covers per-input holds. Joint SS on MIMO
> now uses independent per-MV permutations of the same linspace
> (`_per_mv_hold_rows`); `n_mv<=1` returns None so test_sim keeps the
> scalar path (same RNG, same episodes). Episode **count** stays 40.
> Not a loss reweight; not a test_sim recipe change.
> Not relative Huber.
> Pre-iso resolve is **only** for those scales; gain-match Huber
> targets always re-resolve after isolation+expert (P37 obs-norm freeze
> point — skipping would confound P38). P37 EXIT (`run_p37_isoabs`,
> launch `4d349fb`, 151 iters) ran unscaled abs. Iter 75/85 EXTEND 0.68
> then 0.72@DV; extra-P1 silent detonation iter 88; last-ok iter 87.
> Val MV ss/@H **×0.981 / ×1.005**, DV **×0.690 / ×0.783**, det_r
> **0.370**, `[p3-skip]`. Abs isolation completes P1 and pins MV;
> **FALSIFIED as DV pin**. Actor only on a GAIN-READY freeze. Opt out
> `DREAMER_WM_ISOLATION_DCV_MATCH=0`.
> **P28 follow-up 11 (no GPU this session):** follow-up 10 skipped only
> the *extra* isolation unroll. `world_model_loss` still ran overshoot,
> held-rollout, and full-BPTT gain-match every P2 iter (~73% of the WM
> step plus one K-step FD roll per input). Those losses train `g`
> (encoder/decoder/GRU/cont-gain), which curriculum P2 has frozen;
> P2 only needs recon + DOB ground/reg. Skip the g-only aux when
> `_dynamics_g_trainable` is false. P1 unchanged. test_sim recipe
> unchanged.
> **P28 follow-up 12 (no GPU this session):** overshoot + held-rollout
> call compiled `img_rollout(h, z, a, dv)` and constructed
> `RSSMState` **without posterior `c`**. `img_step` zero-fills
> `prev.c`, so the first GRU step (and therefore the whole open-loop
> gain supervisor — ~73% of the P1 WM step) trained a **c=0 path**
> while isolation / full-BPTT gain-match / the actor / transfer-matrix
> start from posterior `c` (p20 family: supervisor ≠ metric path).
> `img_rollout` now takes `c0`; overshoot/held slice it from feat
> like isolation. test_sim recipe unchanged. `c0=None` still
> zero-fills (back-compat).
> **P28 follow-up 13 (no GPU this session):** follow-up 10 grew only
> `isolation_buf` samples to `max(seq_len, K+1)`. P1/P2 still sampled
> the MAIN replay at `cfg.seq_len`, so overshoot (`K=min(K,T-1)`) and
> gain-match (`n_valid=T-K`) truncated the identified settling length
> whenever `H >= seq_len` (slow plant or `DREAMER_SEQ_LEN` pin) — the
> same DC-gain miss as TBPTT-on-asymptote (P25). P1/P2 now sample
> `_wm_train_seq_len = max(seq_len, K+1)` (P3 on-policy stays
> `seq_len`). Gain-match rolls the full open-loop `K` even when
> `T <= K` (held a/dv; no future obs). GPU calib probes the same T.
> test_sim (64 / H≈55) unchanged.
> **P28 follow-up 14 (no GPU this session):** follow-up 12 threaded
> posterior `c` into `img_rollout`, but sliced it from
> `rollout_observed(sample=True)` feat — the reparameterized *sample*.
> `cont_gain_deterministic_roll` already rolls *subsequent* gain at the
> prior mean (p20), so the first GRU step was the remaining
> `E[f(c_sampled)] ≠ f(mean)` hole vs isolation / actor / transfer-matrix
> (`sample=False`). Open-loop aux (overshoot, held, gain-match, 1-step
> steady) now start from `cont['post_mean']`. Recon still uses the
> sample. test_sim recipe unchanged.

---

## 1. Full architecture (training)

```mermaid
flowchart TB
  subgraph ENV["Plant + unmeasured load (env)"]
    PLANT["Sim plant g_true\nMV,DV -> CV via lag+deadtime"]
    GD["Hidden load L(t) -> Gd\n(dead-time + 1st-order lag)\nunmeasured d_cv  [current]"]
    PLANT --> OBS["obs = state + setpoints\n+ integral + derived"]
    GD --> OBS
  end

  subgraph WM["World model (RSSM default / TSSM opt-in)  — opt_world"]
    ENC["encoder / embed(obs)"]
    POST["posterior obs_step\n-> z (sees obs)"]
    PRIOR["prior img_step\n-> z_hat (no obs)"]
    CORE["deterministic core\nGRU (RSSM) / transformer (TSSM)\nstate h"]
    FEAT["feat = [h, z]"]
    DEC["decoder g(feat) -> obs_hat"]
    DOBS["disturbance state d_t\n(neural Kalman / DOB)  [current]"]
    ENC --> POST --> CORE --> FEAT
    CORE --> PRIOR
    FEAT --> DEC
    DOBS -. "CV = g(feat) + d_t" .-> DEC
  end

  subgraph HEADS["Heads"]
    REW["reward head\n(twohot, on feat)  — opt_world"]
    VAL["critic / value head\n(twohot V(feat))  — opt_critic"]
    TGT["target_value (EMA, frozen)"]
    POL["actor / policy head\npi(a | feat)  — opt_actor"]
    DHEAD["disturbance head\nreads feat (read-out)  [opt-in]\nsuperseded by d_t when DOB on"]
  end

  OBS --> ENC
  FEAT --> REW
  FEAT --> VAL
  FEAT --> POL
  FEAT --> DHEAD
  VAL -. EMA .-> TGT

  subgraph REALSIM["Real-sim controller (Phase 3) — actor+critic on the TRUE plant"]
    ROLLR["roll the REAL sim (DR on)\ncollect_episode -> {obs, act, rew}"]
    ENCR["frozen OBSERVER encode\nrollout_observed(sample=False) -> feat"]
    Lam["lambda-returns (TD-lambda)\nREAL reward + gamma*bootstrap(V)"]
    ROLLR --> ENCR --> Lam
  end

  POL --> ROLLR
  TGT --> Lam
  Lam --> ADV["advantage = return - V(feat)"]
  ADV --> POL
  Lam --> VAL

  POL --> ACT["action a_t (MV)"]
  ACT --> PLANT
  DOBS -. feedforward .-> POL
```

### Reading the diagram
- **World model** learns the plant from `obs`: `encoder → posterior z` (sees obs),
  `prior z_hat` (predicts z without obs — the imagination engine), deterministic
  core `h`, and `decoder g(feat) → obs_hat`. Trained by `opt_world`
  (recon + KL + overshoot/held-rollout). The **disturbance head** [opt-in] is a
  gradient-isolated read-out probe today; the **DOB `d_t`** [planned] replaces
  it with a real state (Section 3).
- **Critic** `V(feat)` [`opt_critic`] is trained on **λ-returns** (TD-λ,
  bootstrapped by the EMA `target_value`) computed from the **REAL** environment
  rewards, **plus a Monte-Carlo grounding term** — `critic_mc_grounding_coef ×`
  the pure discounted reward-to-go (λ=1, no bootstrap) CE — that anchors the
  value to realised economics so it cannot drift/invert (the p03 failure: a
  bootstrap-only λ-return let critic_r go **−0.23**). The critic trains on the
  **diverse shared replay** (a value baseline is action-independent ⇒ off-policy
  replay is unbiased and keeps the head conditioned when the actor sits in a
  corner), while the **actor** stays on-policy (`_realsim_actor_critic_step(…,
  critic_batch=<replay sample>)`).
- **Actor** `π(a|feat)` [`opt_actor`] is trained on the **advantage**
  `return − V(feat)` (÷ the percentile return-scale) via REINFORCE on the REAL
  taken action (`policy.log_prob_of`) from the **on-policy** buffer. It is the
  ONLY thing that drives `action → plant`.
- **Three optimizers are strictly partitioned** (verified by
  `tools/_smoke_grad_isolation.py`): `opt_world` (encoder/core/decoder + reward
  head [+ disturbance head]), `opt_actor` (policy), `opt_critic` (value).
  `target_value` and `prior_policy` are frozen (in no optimizer).

---

## 2. Inference / deployment (closed loop)

```mermaid
flowchart LR
  OBS["obs_t (CV,MV,DV,setpoints)"] --> ENC["encoder"]
  ENC --> POST["posterior obs_step -> feat_t (incl. d_t tail)"]
  POST --> POL["actor pi(a|feat) (deterministic mean)"]
  DOBS["d_t observer (neural Kalman)"] -->|"feedforward (in feat)"| POL
  POST -->|"innovation update"| DOBS
  POL --> MV["MV command"]
  MV --> PLANT["plant"]
  PLANT --> OBS
```

Only the **encoder + posterior + actor** run in closed loop at deploy time (the
critic is training-only). With the DOB enabled,
`d_t` is estimated online from the prediction error and fed forward to the actor
(it is appended to `feat`, so the deployed actor conditions on it directly).

---

## 3. [current] Neural Kalman filter / disturbance observer (DOB)

Implemented 2026-06-11; **default ON** (`TrainConfig.dob_enabled=True`, p142).
Opt out `DREAMER_DOB_ENABLED=0`. The unmeasured load is an **omitted variable**: the WM cannot attribute that CV
movement to any input it sees, so it under-fits the input→CV gain
(MV ratio ≈ 0.64, DV ratio ≈ 0.73 in p112) — which makes the actor over-actuate
and oscillate, and makes a read-out disturbance head unrecoverable. The fix is a
learned **predict–correct observer** (a neural Kalman filter / DOB) bolted onto
the shared `feat → decode` interface so it transfers to **both** backbones.

```mermaid
flowchart LR
  subgraph WMcore["WM process model"]
    G["g(feat) (input->CV gain)"]
  end
  CVOBS["CV_obs"] --> INNOV{"nu = CV_obs - CV_hat"}
  G --> PREDADD["CV_hat = g(feat) + A*d_(t-1)"]
  DPREV["d_(t-1)"] -->|"decay A"| PREDADD
  PREDADD --> INNOV
  INNOV -->|"learned gain K"| DT["d_t = A*d_(t-1) + K*nu"]
  DPREV -->|"decay A"| DT
  DT --> DECO["decoder: CV = g(feat) + d_t"]
  DT --> POLFF["actor (feedforward MV)"]
  DT --> IMGP["img_step: propagate d_t"]
```

- **Predict** (`img_step`, no obs): `d_t = A·d_{t-1}`; `CV_hat = g(feat) + d_t`.
- **Correct** (`obs_step`, real obs): `ν_t = CV_obs − (g + A·d_{t-1})`;
  `d_t = A·d_{t-1} + K·ν_t` (`K` = **learned** Kalman gain).
- **Output**: decoder `CV = g(h,z) + d_t`. `g` now learns the *true* gain because
  `d_t` absorbs the unexplained movement (de-confounds the attenuation). The
  recon loss compares `g(feat)+d_t` vs `obs`, and an L2 prior on `d_t`
  (`dob_reg_coef`, the Kalman "process-noise-is-small" assumption) keeps the
  model using `d_t` only for the genuine residual.
- **Grounding (KalmanNet, P19)**: the recon innovation alone under-drives `d_t`
  in Stage-2 (the slow load is a small share of the recon MSE and `dob_reg`
  opposes it, so the load amplitude is under-estimated — p18: `d_t` vs true load
  r=0.42, ~0.3× amplitude). When `dob_ground_coef > 0` a direct target
  `‖d_t − true_load‖²` (the sim's known hidden load, unit-matched to `d_t`'s
  normalized space via the running CV obs-norm std, threaded as `cfg._cv_obs_std`)
  tunes the Kalman `A,K` to TRACK the load — the structural replacement for the
  manual `dob_gain_init` amplitude tuning and the `dob_reg` prior
  (`dob_reg_coef → 0` when grounding is on). `DREAMER_DOB_GROUND_COEF`.
  **P25 RCA:** the replay buffer must store `n_dist = n_cv` whenever grounding
  or the DOB is on. Do **not** key storage off `disturbance_head_dim` (forced
  0 when the DOB replaces the head — that made P19/P25 `dob_ground` a silent
  no-op). `_replay_n_dist` is the single resolver; a missing/`shape`-mismatch
  target logs a one-shot warning instead of failing closed.
- **Disturbance estimate**: `d_t` itself is the estimate — `wm_disturbance_prediction`
  reads it directly (converted to engineering units via the obs-norm std),
  superseding the read-out head when DOB is on.
- **Feedforward (Scope 2, shipped 2026-06-11)**: `d_t` is appended to `feat`
  (`feat = [h, z_flat, d_t.detach()]`) so the actor / critic / reward heads
  condition on the disturbance estimate and pre-empt the load (prediction-error
  feedforward, not just feedback). `d_t` is **detached** into `feat` (the DOB is
  trained by the recon innovation, not head gradients); the **decoder reads only
  the core `[h, z_flat]`** (slices the d-tail) so the `g + d_t` factorisation is
  preserved. Scope 1 de-confounds the gain; Scope 2 adds the explicit feedforward.

Classical mapping: process model = learned WM dynamics; measurement model =
decoder; `K` = learned Kalman gain (per-CV, `sigmoid` ∈ (0,1)); `d_t` =
bias/disturbance state; holding `d_t` (decayed by `A`, per-CV `sigmoid` ∈ (0,1))
through imagination = the MPC "persistent disturbance" assumption, learned.
Implemented once at the shared `feat → decode` interface so RSSM + TSSM share the
observer math. Env knobs: `DREAMER_DOB_ENABLED` / `_REG_COEF` / `_DECAY_INIT` /
`_GAIN_INIT`. Verified by `tools/_smoke_dob.py` (both backbones: A/K bounded,
decay/correct, CV-only add, grad-isolated into `opt_world`).

---

## 3b. [current] Continuous GAIN latent (C3) — DV-gain de-confounder

> **GAIN-ONLY (2026-08-18):** the continuous **gain** block below is current; the
> continuous **disturbance** block it originally shipped with was reverted at p142
> (it competed with the DOB for the same CV innovation and never recovered). With
> `TrainConfig.dob_enabled=True` the cont latent auto-resolves to **gain-only**
> (`cont_dist_dim=0`, `dist_match` off) and the neural-Kalman **DOB `d_t` owns the
> unmeasured load** (§3). See the 2026-08-18 changelog note above.

Shipped 2026-06-22 (`DREAMER_CONT_LATENT_ENABLED=1`, first run p137). A small
**Gaussian latent alongside the categorical** (`_ContinuousLatent` in
`models/dreamer_v4_rssm.py`) gives the precision-critical **continuous** quantities
an un-quantized home that the 32-class categorical attenuates — the shared root
cause of (a) the subdominant DV-gain bias (≈0.85) and (b) the disturbance read-out
collapse when the DOB is removed (p136: head amplitude 2% of true). One change
fixes BOTH:

- **GAIN block** (`cont_gain_dim = n_cv·(n_mv+n_dv)`): inferred in-context from the
  lookback, feeds the GRU (so `h` carries the per-episode gain forward) **and**
  the decoder. Supervised by **C(1) gain-matching** (`_wm_gain_match_loss`): a
  finite-difference step-response asymptote (roll the prior K=`gain_match_len`
  steps, held baseline vs +`gain_match_step` per MV/DV input, ΔCV/step) matched to
  the identified steady-state gain in WM-normalized units
  (`gain_match_mv_target`/`gain_match_dv_target`, resolved by
  `_resolve_gain_match_targets` from `dynamics_identification.json` + obs-norm +
  action scale: `g_dv_norm = g_eng·obs_std[dv]/obs_std[cv]`,
  `g_mv_norm = g_eng·mv_action_scale/obs_std[cv]`). `sample=False` freezes the
  categorical so the gain gradient flows into the continuous channel + decoder —
  the un-cheatable DC supervisor.
- **DISTURBANCE block** (`cont_dist_dim = n_cv`): an **inherent amortized Kalman**.
  The posterior `q(c|h,embed,ν)` infers the load from the one-step CV
  **innovation** `ν = cv_obs − prior-CV-forecast` (the DOB residual that carries
  the load) — fed EXPLICITLY since **p140**, because a posterior on `[h,embed]`
  alone learned an excited-CV shortcut that died under closed-loop control (p139:
  closed-loop `det_r(c_dist) 0.03` while `det_r(ν) 0.32`). It is supervised
  toward the recorded true load (`dist_match_coef`, p138) and the prior `p(c|h)`
  rolls it forward (OU) via the KL-balanced continuous KL (`rssm_cont_kl_loss`).
  ν needs a prior decode (too costly per-step in the compiled loop), so
  `rollout_observed` computes it BATCHED across two compile-friendly passes
  (pass 1 harvests prior feats → one batched decode → ν; pass 2 re-rolls feeding
  ν so the `c` that feeds `h` is innovation-driven). In **imagination** the
  disturbance block rolls **DETERMINISTICALLY** (prior MEAN, not a sample —
  `cont_dist_deterministic_roll`, p140 RCA): it is a feedforward signal, so a
  per-rollout sample would inject uncontrollable noise into the imagined reward
  and bury the action signal (the gain block stays sampled).

  > **⚠ SUPERSEDED for the disturbance (p142, after 5 failed runs p137–141).** The
  > learned cont-disturbance block never reliably encoded the load on held-out
  > data (p141 held-out `det_r` 0.32→**−0.05**; `dist_match` *diverged* at 0.6) —
  > ν confounds the load with the WM's own gain error, so it is not cleanly
  > identifiable. The disturbance is reverted to the classical **DOB** (the
  > neural-Kalman observer, §3; proven `det_r` 0.354). **When `dob_enabled` AND
  > `cont_latent_enabled`, the config resolves to GAIN-ONLY cont** (`cont_dist_dim
  > =0`, `dist_match_coef=0`): the cont latent keeps only the GAIN block, the DOB
  > owns the load (`d_t` in `feat`, Scope-2), and `gain_match` pins `g` so `d_t`
  > cleanly gets the load residual (no gain↔disturbance fight). The staged
  > clean→disturbance curriculum (the textbook sysID recipe) activates
  > automatically with the DOB on.

`feat = [h, z_flat, c, (dv), (d)]`; the decoder reads `[h, z, c, (dv)]`. Both
blocks feed the GRU transition. `cont_gain_dim == cont_dist_dim == 0` ⇒
byte-identical to the pre-cont model (regression-verified). Env knobs:
`DREAMER_CONT_LATENT_ENABLED` / `_MIN_STD` / `_MAX_STD` / `_FREE_BITS` /
`_KL_SCALE` / `_GAIN_PERSIST_COEF`, `DREAMER_GAIN_MATCH_COEF` / `_LEN` /
`_MAX_STARTS` / `_STEP`, `DREAMER_DIST_MATCH_COEF` (the disturbance-match
supervision, auto-**0.6** when the cont disturbance block is on AND the DOB is
off — the DOB-off fallback only; p141 found 0.6 backfired so the disturbance is
now the DOB's job), `DREAMER_CONT_DIST_DET_ROLL` (deterministic imagination roll,
default on). Threaded through all 5 `DreamerV4Config` build sites.
**Both RSSM AND TSSM are runnable** (p137 RSSM live-validated; TSSM parity is a
real recompute impl — `cont_kl`, `gain_match_loss`, the innovation 2-pass all
smoke-pass on both backbones).

> **DV→decoder paths removed (p141 + p146)**: the measured DV drives the CV
> ONLY through the recurrent transition (like the MV) — no instantaneous decoder
> feedthrough. The p132 zero-init `W·dv_t` decoder skip (`dv_static_skip`) is
> **deleted** (p141: a memoryless feedthrough + `gain_match` crutch, superseded
> by the cont GAIN block + `gain_match`). The p129 `dv_feedforward` decoder half
> is **default OFF** (p146 RCA: appending `dv_t` to the decoder input made the WM
> DV response LEAD the plant and settle low; `DREAMER_DV_DECODER_FEEDFORWARD=1`
> restores it). The measured DV still feeds the head-facing `feat` so the
> actor/critic see the load.

### Continuous-latent curriculum (the simplified path)

When `cont_latent_enabled` (DOB off), the staged §4 curriculum is replaced by the
`_cont_curric` branch in `train()`: **WM-id (P1+P2): `g` TRAINS WITH the
disturbance present** — the cont gain channel + gain-match de-confound the gain
*inherently*, so there is **no clean-P1 / frozen-g-P2 staging** (that existed only
to protect the gain from the disturbance/DR confound), and the cont disturbance
channel learns from the disturbance being present — then **actor on the frozen WM
(P3)**. **mbrl2 real-sim:** DR is **off** during observer id (P1/P2, clean plant)
and **on in P3** (`set_domain_randomization(True)`) so the real-sim actor trains
on the randomised true plant for sim-to-real robustness.

---

## 4. [current] Staged clean→disturbance curriculum

Shipped 2026-06-12; **default ON** (`TrainConfig.curriculum_enabled=True`,
phased + DOB). Opt out `DREAMER_CURRICULUM_ENABLED=0`. Joint mode or
`dob_enabled=False` hard-disables with a warning. It is the textbook
system-identification recipe applied to the DOB:
**identify the plant `g` on clean data → identify the observer `(A,K)` on the
fixed plant → train the controller.** This removes the gain↔disturbance
identifiability confound that co-training `g` and `d_t` on disturbed closed-loop
data creates (p114/p115: `d_t` "steals" gain from `g`). The three stages map to
the existing phases P1/P2/P3 (budgeted by `phase{1,2,3}_frac`); the per-stage
freeze is `DreamerV4.set_world_model_trainable(g, dob, reward)` (toggles
`requires_grad`; `opt_world` skips frozen params) and `set_dob_active(...)`.

The DOB is built ON for the whole run so `feat` is always `core + n_cv` wide —
**no head-dim change at a stage boundary**. In Stage 1 the estimate is *suppressed*
(`d_t ≡ 0`), not removed — `rollout_observed` skips the prior-core harvest that
P2's batched DOB decode consumes (`dob_active=False`).

| | **Stage 1 = P1** (plant id) | **Stage 2 = P2** (observer id) | **Stage 3 = P3** (controller) |
|---|---|---|---|
| **trainable** | `g` (enc/dec/GRU/prior/post) + reward | DOB `(A,K)` + reward | actor + critic + reward |
| **frozen** | DOB `(A,K)` | **`g`** | **`g` AND DOB** |
| **DOB `d_t`** | suppressed (`≡0`) | active | active (feeds actor via `feat`) |
| **unmeasured disturbance** | **OFF** (prob 0) | **ON** (prob 1.0) | ON (prob 0.85) |
| **measured DV (FEED)** | ON | ON | ON |
| **domain randomization** | **OFF** (clean plant id) | **OFF** | **ON** (±10% τ/gain/dead-time — real-sim actor robustness) |
| **process+meas noise** | ramped 0→full (≈40% prog) | ~full | **full** (curriculum →1.0 in P3) |
| **collection** | random-action (open-loop) + seed buffer | random-action (now disturbed) + expert reinject | **on-policy** (actor closes loop) + eval + expert reinject |
| **what it learns** | unbiased input→CV **gain** | Kalman `(A,K)` on the fixed plant | reject disturbances (`d_t` feedforward) + economics |

- **Stage 1** — `d_t≡0` forces `g` to explain *all* CV movement, so the gain is
  identified with no omitted-variable escape hatch. The data is clean of the
  *unmeasured* load; the *measured* DV stays in (it is a WM input), and DR + a
  ramped noise curriculum keep the family realistic. Open-loop excitation comes
  from **random-action collection + the seed buffer** (small-noise holds + PRBS
  sweeps [+ const/step/expert when enabled]).
- **Stage 2** — `g` is frozen, so the recon innovation can only be reduced by
  `d_t`: the observer `(A,K)` is identified on the fixed plant (identifiable by
  construction). Reuses the P2 loss (`wm_total + agent_total`); BC also warms the
  actor as a free bonus. The disturbance is at max density (prob 1.0) so the
  observer sees plenty of residual.
- **Stage 3** — `g` and `(A,K)` are frozen (`_wm_frozen_now` drops `wm_total`);
  the actor/critic train on the static unbiased WM + working observer, **with
  disturbances + domain randomization on**, so the deployed controller is robust
  at runtime. The reward head keeps adapting.

> **Stage-1 excitation (verified p117):** the seed buffer is **settle-aware**, not
> random-only — it carries constant-action episodes (full-episode hold, 4–10×
> settle time), step-settle episodes (hold u₀→step→hold u₁ to episode end, `>>τ`
> tail, noise-free), step-test episodes, and PRBS with a slow hold ≈ `(θ+4τ)/sr`
> (≈4τ ≈ 98% settled); P1 re-injects const+step every 20 iters (anti-eviction).
> So the schedule **does** consider time-to-steady-state. **Empirically (p117
> clean Stage-1 gain probe): the dynamics are essentially perfectly identified**
> (posterior→1-step gain ratio ×0.998); the residual gain under-read sits in the
> **autoencoder** (real→posterior ×0.847) + mild compounding (×0.905), i.e. it is
> an encoder/decoder/latent-capacity bottleneck (the small CV step-gain gets
> squashed), **not** a non-settling/excitation problem. The lever is the
> CV-weighted recon (`wm_recon_cv_weight`) / per-CV obs-norm / larger latent —
> not more step-tests.
>
> **Note (2026-06-12):** the WM-only excitation **partition**
> (`wm_excitation_buffer_frac`) was **removed** — it was only ever drawn in the
> P3/joint WM-update path (inert in this phased curriculum), never demonstrably
> helped, and (per the p117 probe above) the dynamics are already fully
> identified without it. Stage-1 open-loop excitation comes from the settle-aware
> seed buffer + random-action collection + P1 const/step re-injection.

Verified by `tools/_smoke_curriculum.py` (per-stage `requires_grad` partition +
`dob_active` toggle + gradient isolation: S1 recon trains `g` not the DOB, S2
trains the DOB not `g`, S3 leaves the WM static) and `tools/_smoke_curriculum_e2e.py`
(real phased `train()`: all three stage transitions fire, recon finite through
the freezes, WM frozen by S3). Both backbones.

---

## 5. Code map

| Component | Where |
|---|---|
| RSSM (`obs_step`/`img_step`/`decode`/`rollout_observed`) | `models/dreamer_v4_rssm.py` |
| TSSM (transformer, duck-compatible) | `models/transformer_ssm.py` |
| Heads (reward/value/policy/disturbance), param groups | `models/dreamer_v4.py` (`parameters_world/_actor/_critic`) |
| WM loss (recon/KL/overshoot/held-rollout, disturbance) | `training/train.py` (`world_model_loss`, `_disturbance_head_loss`) |
| Real-sim λ-returns + MC grounding + actor/critic | `training/train.py` (`_realsim_actor_critic_step`) |
| Hidden load + Gd disturbance | `utils/hidden_disturbance.py` (`HiddenDisturbance`) |
| Neural Kalman filter / DOB (`d_t` state) | `models/dreamer_v4_rssm.py` + `models/transformer_ssm.py` (`dob_enabled`, `obs_step`/`img_step`/`apply_dob`); recon in `training/train.py:_rssm_world_model_loss` |
| Staged curriculum (per-stage freeze + DOB suppression) | `models/dreamer_v4.py` (`set_world_model_trainable`, `set_dob_active`); stage latch in `training/train.py` (`curriculum_enabled`, per-iter stage hook) |
| Gradient-isolation audit | `tools/_smoke_grad_isolation.py` |
| DOB smoke (both backbones) | `tools/_smoke_dob.py` |
| Curriculum smoke (unit + e2e) | `tools/_smoke_curriculum.py`, `tools/_smoke_curriculum_e2e.py` |
| Disturbance-prediction diagnostic | `evaluation/wm_disturbance_prediction.py` |
| WM gain / posterior-prior probes | `evaluation/wm_transfer_matrix.py`, `tools/wm_posterior_prior_probe.py` |
