# Reference papers

Outbound HTTPS is blocked from the training host so the PDFs could not be
mirrored automatically.  Download manually if needed:

| File name (suggested)                | URL                              |
|--------------------------------------|----------------------------------|
| `dreamerV4_2509.24527.pdf`           | https://arxiv.org/pdf/2509.24527 |
| `dreamerV3_2301.04104.pdf`           | https://arxiv.org/pdf/2301.04104 |

```bash
cd docs/papers
curl -fL -o dreamerV4_2509.24527.pdf https://arxiv.org/pdf/2509.24527
curl -fL -o dreamerV3_2301.04104.pdf https://arxiv.org/pdf/2301.04104
```

## Citations

* **Dreamer 4** — Hafner, Yan & Lillicrap, *Training Agents Inside of
  Scalable World Models*, arXiv:2509.24527, Sep 2025.
* **Dreamer V3** — Hafner, Pasukonis, Ba & Lillicrap, *Mastering Diverse
  Domains through World Models*, arXiv:2301.04104, Nature 2025.

## Key recipes used in this codebase

### From DreamerV3 (continuous-control prescriptions)

1. **Truncated-/squashed-Normal actor** with mean ∈ [-1, 1] (via `tanh`)
   and **σ ∈ [0.1, 1.0]** (i.e. `log_std ∈ [-2.3, 0]`).  Keeping σ
   bounded is described as a key stability fix that allows a single
   hyperparameter set to work across 150+ tasks.
2. **Percentile return scale**:
   `S = max(1, EMA(Per(R, 95) − Per(R, 5)))`; advantages are divided by
   `S` before the actor loss.  Robust to reward-spike tails (e.g. our
   cv-violation penalty), unlike a plain std EMA.
3. **Symlog reward / value targets** with a 2-hot critic — already
   implemented in `models.dreamer_v4.TwohotHead`.
4. **Entropy bonus** `η · H[π]` with `η = 3e-4` added to the actor loss
   (acts as a soft σ floor).
5. **Actor / critic learning rate** `3e-5`, **WM lr** `1e-4`,
   **batch size** `16` for proprio control.
6. **EMA target critic** with τ ≈ 0.02 — already implemented.

### From DreamerV4 (Phase 3 / PMPO specifics)

1. **Three-phase trainer** (WM pretrain / BC + MTP / PMPO actor) —
   already implemented.
2. **PMPO loss** (paper eq. 11): advantage-sign-split surrogate plus
   `β · KL(π ‖ π_prior)` to a frozen behavioural prior.  `β = 0.1`,
   `α = 0.5` are the defaults — already implemented in
   `models.dreamer_v4.pmpo_loss`.
3. **Shortcut-forcing world-model loss** (paper eq. 7) — implemented.
4. **MTP** (multi-token prediction) head of length `L = 8` — implemented.
5. **Continuous actions are NOT discussed in the V4 paper.**  V4
   inherits V3's continuous-control machinery (truncated-normal head,
   percentile return-norm, entropy bonus, lr 3e-5).  The codebase
   release confirms this.

## Stability fixes in this repo

The four "V3 prescriptions" above (log_std clamp, percentile return-norm,
entropy bonus, actor lr) are wired generically through `TrainConfig`
fields (`policy_log_std_min/max`, `return_norm`, `entropy_coef`,
`actor_lr`) so they apply uniformly across simulators.  Auto-derivation
hooks (e.g. `auto_weights.derive_auto_weights`) do not override them —
they remain the same recipe everywhere, matching V3's "single
hyperparameter set" goal.
