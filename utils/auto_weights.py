"""Auto-derive objective weights from priority rank, control-speed preset,
and identified per-MV dynamics.

Replaces hand-tuned ``mv_violation_weights``, ``cv_violation_weights``,
``mv_move_weights``, and ``mv/cv_target_weights`` in
``control_objective.json``.

Priority ordering (enforced by construction, April 2026 quadratic refactor;
move-cap added 2026-05-12):

    1. MV limits         (most important -- never violate)
    2. CV limits by rank (rank 1 highest, geometric decay per rank)
    3. Targets           (when targets_enabled)
    4. Economics         (mv/cv economic nudge, user-supplied weights)
    5. MV move           (steady-state actor-jitter cost)

Each tier strictly dominates the one below it at the typical
violation/deviation magnitude:
- MV violation > CV rank-1 violation by ``OBJ_AUTO_MV_OVER_CV_RATIO``.
- CV rank-N violation > targets+economics by ``OBJ_AUTO_VIOLATION_MARGIN``
  (rank-1 inflated by ``rank_decay^-(N-1)`` so the ladder holds).
- Target tracking > economics by ``OBJ_AUTO_VIOLATION_MARGIN``.
- Economics > move penalty by ``OBJ_AUTO_ECON_OVER_MOVE_RATIO``
  (NEW 2026-05-12 — symmetric with the CV>econ margin so move never
  out-weighs the user's economic objective).

MV/CV violation penalties are **quadratic** in the ReLU bound-violation
magnitude (``(max(0, lo - x))^2 + (max(0, x - hi))^2``). Target penalties
are **linear** (L1) in the target deviation. Adaptive scaling solves for
base weights so each tier dominates the one below it at the configured
``OBJ_AUTO_VIOLATION_TOLERANCE`` normalised violation magnitude.

- MV violation is sized so that, at the tolerance violation, its
  quadratic penalty dominates the CV tier by ``OBJ_AUTO_MV_OVER_CV_RATIO``.
- CV violation (rank 1) is sized so that, at the tolerance violation,
  its quadratic penalty dominates the economic budget by
  ``OBJ_AUTO_VIOLATION_MARGIN``. Subsequent ranks decay geometrically
  by ``OBJ_AUTO_CV_RANK_DECAY``.
- Economic weights are supplied by the user (in ``spec['weights']``).
- Target weights (for channels with ``runtime_setpoints.targets_enabled``
  = True) are auto-sized so the economic budget dominates the target
  budget by ``OBJ_AUTO_ECON_OVER_TARGET_RATIO``. If the user supplied
  no economic weights, target_base defaults to ``OBJ_AUTO_TARGET_BASE``.
- Move penalty per MV: ``MOVE_BASE * max(0.2, tau_i/median_tau) * speed_factor``.
  Slow MVs get more penalty; fast MVs less.

All magnitudes can be overridden via env vars:
- ``OBJ_AUTO_MV_VIOLATION_BASE``      (default 25.0; linear-equivalent floor)
- ``OBJ_AUTO_CV_VIOLATION_BASE``      (default 25.0; linear-equivalent floor)
- ``OBJ_AUTO_CV_RANK_DECAY``          (default 0.5)
- ``OBJ_AUTO_MOVE_BASE``              (default 0.1; floor only)
- ``OBJ_AUTO_MV_OVER_CV_RATIO``       (default 2.0)
- ``OBJ_AUTO_VIOLATION_TOLERANCE``    (default 0.02, i.e. 2% of normalised range)
- ``OBJ_AUTO_VIOLATION_MARGIN``       (default 2.0)
- ``OBJ_AUTO_CV_PENALTY_CAP_FRAC``    (default 0.5; cv_base cap vs reward_clip)
- ``OBJ_AUTO_TYPICAL_CV_VIOLATION``   (default 0.05; typical normalised violation)
- ``OBJ_AUTO_MOVE_OVER_CV_K``         (default 20.0; move_base ≤ cv_base / K legacy cap)
- ``OBJ_AUTO_ECON_OVER_MOVE_RATIO``   (default 2.0; econ_budget ≥ ratio × per-step move pen at typical jitter)
- ``OBJ_AUTO_ECON_OVER_TARGET_RATIO`` (default 2.0; econ budget vs target budget)
- ``OBJ_AUTO_TARGET_BASE``            (default 0.5; target_base floor if no econ)
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except Exception:
        return float(default)


def _cv_rank_order_from_list(raw, n_cv: int) -> List[int]:
    """Convert a list of CV names (``cv_0``, ``cv_1`` ...) into a per-state-
    index rank list (1 = most important). Missing names get tail ranks.
    """
    order = list(range(n_cv))
    if isinstance(raw, list) and raw:
        seen = []
        for name in raw:
            try:
                idx = int(str(name).lower().replace('cv_', ''))
            except Exception:
                continue
            if 0 <= idx < n_cv and idx not in seen:
                seen.append(idx)
        for i in range(n_cv):
            if i not in seen:
                seen.append(i)
        order = seen
    ranks = [0] * n_cv
    for rank_minus_one, state_idx in enumerate(order):
        if 0 <= state_idx < n_cv:
            ranks[state_idx] = rank_minus_one + 1
    for i in range(n_cv):
        if ranks[i] == 0:
            ranks[i] = n_cv
    return ranks


def _cv_rank_order(spec: Dict, n_cv: int) -> List[int]:
    """Map state-order CV index -> rank from ``spec['cv_priority']``."""
    raw = spec.get('cv_priority') if isinstance(spec, dict) else None
    return _cv_rank_order_from_list(raw, n_cv)


def _cv_rank_order_side(spec: Dict, n_cv: int, side: str) -> List[int]:
    """Per-bound-side rank order. ``side`` is ``'lo'`` or ``'hi'``.

    Reads ``spec['cv_priority_lo']`` / ``spec['cv_priority_hi']`` and
    falls back to ``spec['cv_priority']`` when the side-specific key is
    absent. Returns a per-state-index rank list (1 = most important).

    This lets a multi-CV objective express asymmetric urgency between
    over-limit and under-limit conditions per channel (e.g. tighter
    over-limit handling on a quality CV, tighter under-limit on a
    safety CV) without changing the legacy ``cv_priority`` schema. The
    derived ``cv_violation_weights_lo`` / ``cv_violation_weights_hi``
    are consumed by ``utils.objective_runtime.compute_objective_components``
    when ``OBJECTIVE_REWARD_MODE=dmc`` (linear) and produce a true
    asymmetric penalty: ``w_lo * max(0, lo - x) + w_hi * max(0, x - hi)``.
    """
    if not isinstance(spec, dict):
        return _cv_rank_order_from_list(None, n_cv)
    side = str(side).lower().strip()
    key = 'cv_priority_lo' if side == 'lo' else 'cv_priority_hi'
    raw = spec.get(key)
    if not isinstance(raw, list) or not raw:
        raw = spec.get('cv_priority')
    return _cv_rank_order_from_list(raw, n_cv)


def _load_dynamics_json() -> Optional[Dict]:
    path = os.environ.get('DYNAMICS_IDENTIFICATION_JSON', '').strip()
    if not path or not os.path.isfile(path):
        return None
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return None


def _per_mv_tau(n_mv: int, dynamics: Optional[Dict]) -> List[float]:
    """Return a list of τ estimates per MV index.  Falls back to uniform."""
    default_tau = 20.0
    if not isinstance(dynamics, dict):
        return [default_tau] * n_mv
    # The dynamics JSON stores per-pair estimates; aggregate max τ per MV
    # across its CV pairs so slow dynamics dominate the move penalty.
    pairs = dynamics.get('per_pair_estimates') or dynamics.get('pairs') or []
    per_mv: Dict[int, List[float]] = {}
    for p in pairs if isinstance(pairs, list) else []:
        try:
            mv_idx = int(p.get('mv_index', p.get('mv', -1)))
            tau = float(p.get('tau', p.get('tau_est', 0.0)))
        except Exception:
            continue
        if mv_idx < 0 or tau <= 0.0:
            continue
        per_mv.setdefault(mv_idx, []).append(tau)
    out = []
    for i in range(n_mv):
        taus = per_mv.get(i) or []
        out.append(float(np.median(taus)) if taus else default_tau)
    return out


def _safe_float(v, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return float(default)


def _sum_abs_weights(weights_section: Dict, key: str) -> float:
    """Sum of absolute weight values under ``weights[key]`` in the spec.

    Supports both dict-of-channels (``{"mv_0": 5.0, "mv_1": 2.0}``) and
    list representations, matching the formats accepted elsewhere in the
    objective runtime.
    """
    if not isinstance(weights_section, dict):
        return 0.0
    sub = weights_section.get(key)
    if isinstance(sub, dict):
        return float(sum(abs(_safe_float(v, 0.0)) for v in sub.values()))
    if isinstance(sub, (list, tuple)):
        return float(sum(abs(_safe_float(v, 0.0)) for v in sub))
    return 0.0


def derive_auto_weights(spec: Dict, n_mv: int, n_cv: int,
                        dynamics: Optional[Dict] = None) -> Dict:
    """Return a dict of auto-derived weight vectors.

    The returned weights are **adaptive** to (a) the user-supplied
    economic / target weights in ``spec['weights']`` and (b) the identified
    per-MV dynamics.  Priority ordering
    ``MV > CV-rank-1 > targets > economics`` is enforced by construction.

    Keys:
        mv_violation_weights: list[float] length n_mv (quadratic coeff)
        cv_violation_weights: list[float] length n_cv (quadratic coeff)
        mv_move_weights: list[float] length n_mv
        mv_target_weights: list[float] length n_mv (linear L1 coeff)
        cv_target_weights: list[float] length n_cv (linear L1 coeff)
        cv_ranks: list[int]
        tau_per_mv: list[float]
        violation_rate_coef: float (auto-derived DMC sliding-mode gain)
        econ_budget / target_budget / priority_budget: diagnostic scalars
    """
    mv_base_env = _env_float('OBJ_AUTO_MV_VIOLATION_BASE', 25.0)
    cv_base_env = _env_float('OBJ_AUTO_CV_VIOLATION_BASE', 25.0)
    rank_decay = _env_float('OBJ_AUTO_CV_RANK_DECAY', 0.5)
    mv_over_cv_ratio = max(1.0, _env_float('OBJ_AUTO_MV_OVER_CV_RATIO', 2.0))
    tolerance = max(1e-4, _env_float('OBJ_AUTO_VIOLATION_TOLERANCE', 0.02))
    margin = max(0.0, _env_float('OBJ_AUTO_VIOLATION_MARGIN', 2.0))
    econ_over_target_ratio = max(1.0, _env_float('OBJ_AUTO_ECON_OVER_TARGET_RATIO', 2.0))
    target_base_floor = _env_float('OBJ_AUTO_TARGET_BASE', 0.5)
    # Reward clip used to size the violation cap. Quadratic formulation
    # caps cv_base so the expected penalty at the *typical* violation
    # magnitude stays inside cap_frac of the reward clip, preventing the
    # adaptive formula from producing reward magnitudes that saturate.
    reward_clip = _env_float('OBJECTIVE_REWARD_CLIP', 250.0)
    cap_frac = max(0.01, _env_float('OBJ_AUTO_CV_PENALTY_CAP_FRAC', 0.5))
    typical_cv_viol_frac = max(1e-3, _env_float('OBJ_AUTO_TYPICAL_CV_VIOLATION', 0.10))
    # Soft saturation advisory threshold (no longer a hard cap).
    # The reward signal uses tanh saturation in objective_runtime.py, so
    # oversized cv_base values no longer create a flat-gradient cliff —
    # they simply mean a typical-magnitude violation produces a reward
    # near the clip. We log a notice when this happens so the user can
    # raise reward_clip if they need linear differentiation between
    # moderate and severe violations, but never override the hierarchy
    # the user asked for.
    cv_base_advisory = (cap_frac * reward_clip) / (typical_cv_viol_frac * typical_cv_viol_frac)
    # Linear-equivalent floor conversion: a user-set "base=25" linear
    # violation weight produced penalty = base * tolerance at tolerance.
    # For quadratic equivalence at the same tolerance, the quadratic base
    # must be base_linear / tolerance so that penalty_quad = base_quad *
    # tolerance² = base_linear * tolerance (same numeric value). This keeps
    # the env-var default "at least as punishing" as the previous linear
    # formulation, without requiring users to re-derive their overrides.
    cv_base_floor_quadratic = cv_base_env / tolerance
    mv_base_floor_quadratic = mv_base_env / tolerance
    # Adaptive move-penalty derivation (paper-aligned, simulator-agnostic).
    # Goal: at steady state the per-step move penalty incurred by actor
    # exploration noise alone equals a small fixed fraction of the reward
    # clip (default 0.5%). This keeps the move term invisible to
    # corrective control while still applying a small pressure toward
    # narrowing actor sigma at convergence. Formula:
    #   move_base = (target_cost_frac * reward_clip) / E[|delta_a|_noise]
    # where E[|delta_a|_noise] = sigma_ref * sqrt(2/pi) for a Gaussian
    # action perturbation between consecutive steps.
    # The legacy ``OBJ_AUTO_MOVE_OVER_CV_K`` env var is retained as a
    # fallback floor (so users with overrides keep their behaviour) and
    # ``OBJ_AUTO_MOVE_BASE`` becomes a hard absolute floor.
    move_cv_ratio_k = max(1.0, _env_float('OBJ_AUTO_MOVE_OVER_CV_K', 20.0))
    move_base_floor = _env_float('OBJ_AUTO_MOVE_BASE', 0.1)
    move_target_cost_frac = max(
        0.0, _env_float('OBJ_AUTO_MOVE_TARGET_COST_FRAC', 0.005)
    )
    move_sigma_ref = max(0.05, _env_float('OBJ_AUTO_MOVE_SIGMA_REF', 0.3))
    # E[|x|] for x ~ N(0, sigma) is sigma * sqrt(2/pi). The relevant
    # quantity is |a_t - a_{t-1}| where each is a noisy sample around the
    # actor mean; if exploration is i.i.d. across steps the difference is
    # sigma * sqrt(2), giving E[|delta|] = sigma * 2 / sqrt(pi). We use
    # the conservative single-sigma estimate so the penalty is a soft
    # floor rather than an upper bound.
    expected_step_jitter = move_sigma_ref * float(np.sqrt(2.0 / np.pi))
    move_base_adaptive = 0.0
    if move_target_cost_frac > 0.0 and expected_step_jitter > 1e-6:
        move_base_adaptive = (
            move_target_cost_frac * reward_clip / expected_step_jitter
        )

    # ----- Adaptive budget from spec weights + targets_enabled -----------
    # Strict priority hierarchy (highest-to-lowest, all measured at the
    # *typical* violation/deviation magnitude in normalised space):
    #
    #     1. MV bound violations
    #     2. CV bound violations, by ``cv_priority`` rank order
    #        (rank-1 strongest; rank-N still beats targets+economics)
    #     3. Target tracking (only for channels in ``targets_enabled``)
    #     4. Economic weights (user-set in control_objective.json)
    #
    # The hierarchy is enforced by construction:
    #
    #   target_base   = margin × max(econ_budget, target_floor)
    #     → target tracking dominates economics by ``margin``
    #
    #   cv_base × tolerance² × rank_decay^(N-1)  ≥  margin × priority_budget
    #     → even the *lowest-ranked* CV at a tolerance-magnitude violation
    #       beats targets+economics by ``margin``. Higher-ranked CVs
    #       stack on top via the geometric ``rank_decay`` ladder.
    #
    #   mv_base = mv_over_cv_ratio × cv_base
    #     → MV-limit penalty dominates rank-1 CV by ``mv_over_cv_ratio``.
    #
    # All three margins (margin, mv_over_cv_ratio, rank_decay) keep their
    # universal defaults; the user does not need to tune anything.
    weights_section: Dict = {}
    if isinstance(spec, dict):
        w = spec.get('weights')
        if isinstance(w, dict):
            weights_section = w

    mv_econ_total = _sum_abs_weights(weights_section, 'mv_economic')
    cv_econ_total = _sum_abs_weights(weights_section, 'cv_economic')
    # Max per-step economic benefit: pushing a normalised channel from one
    # bound to the other yields |x_clipped - 0.5| up to 0.5 per unit
    # weight. Economic is now clipped to bounds so this is a hard cap.
    econ_budget = 0.5 * (mv_econ_total + cv_econ_total)

    # Which targets are actually active? Only channels flagged in
    # runtime_setpoints.targets_enabled contribute to the target budget.
    cv_target_active, mv_target_active = _active_targets(spec, n_cv=n_cv, n_mv=n_mv)
    n_active_targets = int(sum(1 for v in cv_target_active + mv_target_active if v))

    # Targets dominate economics: each active target's per-step linear
    # penalty at the *typical* deviation (we use 1.0 = full normalised
    # range as the worst-case sizing point) must exceed econ_budget by
    # ``margin``. When the user supplied no economic weights, fall back
    # to ``target_base_floor`` so pure target-tracking tasks still have
    # a usable scale.
    if n_active_targets > 0:
        target_base_adaptive = margin * max(econ_budget, target_base_floor)
        target_base = float(target_base_adaptive)
    else:
        target_base = 0.0

    # Per-step target budget (linear L1, deviation in [0, 1]).
    target_budget = float(n_active_targets) * target_base * 1.0

    priority_budget = float(max(econ_budget, target_budget))

    # Tail-rank guard: the lowest-ranked CV must still beat the
    # priority_budget by ``margin``. With geometric rank decay this
    # means the rank-1 (highest-priority) base must be inflated by
    # ``rank_decay^-(N-1)``.
    n_cv_eff = max(1, int(n_cv))
    tail_factor = float(rank_decay ** max(0, n_cv_eff - 1))
    tail_factor = max(tail_factor, 1e-6)  # avoid div-by-zero for crazy N

    cv_base_adaptive = 0.0
    if priority_budget > 0.0:
        cv_base_adaptive = (margin * priority_budget) / (
            tolerance * tolerance * tail_factor
        )
    cv_base_raw = float(max(cv_base_floor_quadratic, cv_base_adaptive))
    # No hard cap: tanh saturation in the reward path means oversized
    # cv_base no longer produces a flat-gradient cliff. The hierarchy
    # (MV > CV(by rank) > targets > economics) takes precedence.
    cv_base = cv_base_raw
    cv_base_above_advisory = bool(cv_base > cv_base_advisory + 1e-9)
    if cv_base_above_advisory:
        try:
            print(
                f"[auto_weights] cv_violation_base={cv_base:.0f} exceeds the "
                f"advisory cap {cv_base_advisory:.0f} (=cap_frac*reward_clip/"
                f"typ_viol²). This is expected when economics or n_cv are "
                f"large; with tanh saturation the gradient is preserved and "
                f"the priority hierarchy is maintained. Raise "
                f"OBJECTIVE_REWARD_CLIP if you need linear differentiation "
                f"between moderate and severe violations."
            )
        except Exception:
            pass
    cv_base_capped = False  # legacy field, retained for diagnostics
    # Enforce priority #1: MV limits dominate CV limits by construction.
    mv_base = float(max(mv_base_floor_quadratic, mv_over_cv_ratio * cv_base))

    # ---- DMC reward-mode rescaling (2026-05-26) -----------------------
    # ``cv_base`` / ``mv_base`` are sized to make the QUADRATIC violation
    # tail (legacy mode) hit a useful gradient at typical operating
    # depths. In DMC mode the violation shape is LINEAR, so at the same
    # depth the un-rescaled base produces a penalty ~1/typ_depth times
    # larger than the legacy curve and tanh-saturates everywhere except
    # very small depths — destroying both the per-side asymmetry and the
    # depth gradient. We rescale by ``OBJ_AUTO_DMC_VIOLATION_SCALE``
    # (default = typical normalized depth = 0.05) so that at typical
    # depth ``base_dmc * depth`` matches ``base_legacy * depth^2`` in
    # magnitude. The MV/CV ratio is preserved (both bases scaled
    # identically) so priority MV > CV is maintained, and the priority
    # CV > economics is maintained by the same factor (economics already
    # capped relative to cv_base). Move tier is left untouched: it
    # becomes ~1/typ_jitter smaller in absolute reward magnitude under
    # the quadratic shape, but its anti-chatter relative gradient is
    # preserved and it stays well below the violation tier (priority
    # violation > move strengthened).
    reward_mode_spec = ''
    if isinstance(spec, dict):
        reward_mode_spec = str(spec.get('reward_mode', '') or '').strip().lower()
    # Env override beats spec, matching objective_runtime precedence.
    reward_mode_env = str(os.environ.get('OBJECTIVE_REWARD_MODE', '') or '').strip().lower()
    effective_reward_mode = reward_mode_env or reward_mode_spec or 'legacy'
    if effective_reward_mode not in ('legacy', 'dmc'):
        effective_reward_mode = 'legacy'
    if effective_reward_mode == 'dmc':
        # P57 (2026-05-27): tightened from 0.05 → 0.02 because P56 showed
        # the linear-penalty depth slope saturated the symlog support and
        # the critic-pessimism cascade. 0.02 halves the saturation depth
        # and shrinks the negative tail to give the symlog grid room.
        dmc_violation_scale = _env_float('OBJ_AUTO_DMC_VIOLATION_SCALE', 0.02)
        cv_base = float(cv_base * dmc_violation_scale)
        mv_base = float(mv_base * dmc_violation_scale)

    if dynamics is None:
        dynamics = _load_dynamics_json()

    ranks = _cv_rank_order(spec, n_cv)
    ranks_lo = _cv_rank_order_side(spec, n_cv, 'lo')
    ranks_hi = _cv_rank_order_side(spec, n_cv, 'hi')
    # Optional per-CV per-side explicit multiplier. Used to express
    # asymmetric hi-vs-lo urgency on a SINGLE-CV plant where the
    # rank-based scheme is degenerate (one CV → rank=1 on both sides).
    # Schema: ``"cv_side_scale": {"cv_0": {"lo": 0.4, "hi": 1.0}, ...}``
    # Missing CV / missing side defaults to 1.0 (no change).
    side_scale_raw = spec.get('cv_side_scale', {}) if isinstance(spec, dict) else {}
    if not isinstance(side_scale_raw, dict):
        side_scale_raw = {}

    def _side_scale(j: int, side: str) -> float:
        entry = side_scale_raw.get(f'cv_{j}')
        if not isinstance(entry, dict):
            return 1.0
        try:
            return float(entry.get(side, 1.0))
        except Exception:
            return 1.0
    taus = _per_mv_tau(n_mv, dynamics)
    median_tau = float(np.median(taus)) if taus else 1.0
    median_tau = max(1e-3, median_tau)

    cv_weights = [float(cv_base * (rank_decay ** max(0, r - 1))) for r in ranks]
    # Per-side CV weights derived from optional ``cv_priority_lo`` /
    # ``cv_priority_hi`` (rank-based, useful when N_CV > 1) AND
    # ``cv_side_scale`` (explicit per-side multiplier, useful when
    # N_CV == 1 where rank ordering is degenerate). When both keys are
    # absent the per-side weights collapse to ``cv_weights`` (=symmetric,
    # legacy behaviour). Consumed by ``OBJECTIVE_REWARD_MODE=dmc`` in
    # ``objective_runtime.py``; legacy mode keeps using the symmetric
    # ``cv_violation_weights`` so checkpoint calibrations remain valid.
    cv_weights_lo = [
        float(cv_base * (rank_decay ** max(0, r - 1)) * _side_scale(j, 'lo'))
        for j, r in enumerate(ranks_lo)
    ]
    cv_weights_hi = [
        float(cv_base * (rank_decay ** max(0, r - 1)) * _side_scale(j, 'hi'))
        for j, r in enumerate(ranks_hi)
    ]
    mv_weights = [float(mv_base) for _ in range(n_mv)]
    # Adaptive move_base: target a fixed-fraction-of-reward-clip steady-
    # state cost from actor jitter (sigma_ref). Three competing ceilings:
    #   1. ``move_base_adaptive``     — fixed cost-frac of reward_clip
    #      (sim-agnostic; keeps move pressure bounded vs reward scale).
    #   2. ``move_base_legacy_cap``   — cv_base / K (legacy historical cap).
    #   3. ``move_base_econ_cap``     — economics-dominance cap (NEW
    #      2026-05-12): the per-step move penalty incurred at the
    #      *typical* actor jitter must stay below ``econ_budget /
    #      OBJ_AUTO_ECON_OVER_MOVE_RATIO`` so the economics tier strictly
    #      dominates the move tier — symmetric with how CV bounds
    #      dominate economics by ``OBJ_AUTO_VIOLATION_MARGIN``. When
    #      ``econ_budget = 0`` (no economic weights) this cap is
    #      disabled and the legacy two-cap behaviour is preserved.
    # The minimum of the three is taken, then floored at
    # ``OBJ_AUTO_MOVE_BASE`` so move never goes to zero (preserves a
    # baseline pressure toward narrowing actor sigma at convergence).
    move_base_legacy_cap = float(cv_base / move_cv_ratio_k)
    econ_over_move_ratio = max(1.0,
        _env_float('OBJ_AUTO_ECON_OVER_MOVE_RATIO', 2.0))
    if econ_budget > 0.0 and expected_step_jitter > 1e-6:
        move_base_econ_cap = float(econ_budget / (
            econ_over_move_ratio * expected_step_jitter * max(1, n_mv)
        ))
    else:
        move_base_econ_cap = float('inf')
    move_base_uncapped_min = float(min(move_base_adaptive,
                                        move_base_legacy_cap,
                                        move_base_econ_cap))
    move_base = float(max(move_base_floor, move_base_uncapped_min))
    move_weights = [
        float(move_base * max(0.2, tau / median_tau))
        for tau in taus
    ]

    # ---- Adaptive DMC violation-rate coefficient (2026-05-26) ----
    # Sliding-mode "reaching law" gain auto-derived from identified
    # plant dynamics. Larger gain = faster constraint recovery; smaller
    # gain = less reaction to spurious growth signals from noise.
    # Trade-off: slow plants benefit more from anticipatory penalty
    # (many sample periods to recover), fast plants need less (the
    # static linear penalty alone reacts in time). The quadratic MV
    # move penalty in DMC mode already handles anti-jitter, so coef
    # can be moderately aggressive without inducing chatter.
    # Formula: coef = clip(median_tau / 4, 0.3, 1.5).
    # Override via env ``OBJ_AUTO_VIOLATION_RATE_COEF_DIVISOR`` (default 4),
    # ``OBJ_AUTO_VIOLATION_RATE_COEF_MIN`` (default 0.3),
    # ``OBJ_AUTO_VIOLATION_RATE_COEF_MAX`` (default 1.5).
    # Hard override via env ``OBJECTIVE_DMC_VIOLATION_RATE_COEF`` or
    # spec key ``violation_rate_coef`` (consumed in objective_runtime.py).
    rate_div = max(0.5, _env_float('OBJ_AUTO_VIOLATION_RATE_COEF_DIVISOR', 4.0))
    rate_min = max(0.0, _env_float('OBJ_AUTO_VIOLATION_RATE_COEF_MIN', 0.3))
    rate_max = max(rate_min, _env_float('OBJ_AUTO_VIOLATION_RATE_COEF_MAX', 1.5))
    violation_rate_coef = float(
        max(rate_min, min(rate_max, median_tau / rate_div))
    )
    mv_target_weights = [float(target_base if v else 0.0) for v in mv_target_active]
    cv_target_weights = [float(target_base if v else 0.0) for v in cv_target_active]

    return {
        'mv_violation_weights': mv_weights,
        'cv_violation_weights': cv_weights,
        'cv_violation_weights_lo': cv_weights_lo,
        'cv_violation_weights_hi': cv_weights_hi,
        'mv_move_weights': move_weights,
        'mv_target_weights': mv_target_weights,
        'cv_target_weights': cv_target_weights,
        'cv_ranks': ranks,
        'cv_ranks_lo': ranks_lo,
        'cv_ranks_hi': ranks_hi,
        'tau_per_mv': taus,
        'median_tau': median_tau,
        'violation_rate_coef': violation_rate_coef,
        'mv_violation_base': mv_base,
        'cv_violation_base': cv_base,
        'mv_violation_base_floor_linear': mv_base_env,
        'cv_violation_base_floor_linear': cv_base_env,
        'mv_violation_base_floor_quadratic': mv_base_floor_quadratic,
        'cv_violation_base_floor_quadratic': cv_base_floor_quadratic,
        'mv_violation_base_adaptive': float(mv_over_cv_ratio * cv_base),
        'cv_violation_base_adaptive': float(cv_base_adaptive),
        'cv_violation_base_raw': float(cv_base_raw),
        'cv_violation_base_advisory': float(cv_base_advisory),
        'cv_violation_base_above_advisory': bool(cv_base_above_advisory),
        'cv_violation_base_capped': bool(cv_base_capped),
        'cv_tail_rank_factor': float(tail_factor),
        'n_cv_for_tail_guard': int(n_cv_eff),
        'cv_penalty_cap_frac': float(cap_frac),
        'typical_cv_violation_frac': float(typical_cv_viol_frac),
        'reward_clip_used_for_cap': float(reward_clip),
        'move_over_cv_k': float(move_cv_ratio_k),
        'cv_rank_decay': rank_decay,
        'move_base': move_base,
        'move_base_adaptive': float(move_base_adaptive),
        'move_base_legacy_cap': float(move_base_legacy_cap),
        'move_base_econ_cap': float(move_base_econ_cap),
        'move_base_floor': float(move_base_floor),
        'move_base_active_cap': (
            'econ' if (econ_budget > 0.0
                       and move_base_econ_cap <= move_base_legacy_cap
                       and move_base_econ_cap <= move_base_adaptive)
            else 'adaptive' if move_base_adaptive <= move_base_legacy_cap
            else 'legacy_cv_over_k'
        ),
        'econ_over_move_ratio': float(econ_over_move_ratio),
        'econ_budget_per_step': float(econ_budget),
        'move_target_cost_frac': float(move_target_cost_frac),
        'move_sigma_ref': float(move_sigma_ref),
        'move_expected_step_jitter': float(expected_step_jitter),
        'mv_over_cv_ratio': mv_over_cv_ratio,
        'violation_tolerance': tolerance,
        'violation_margin': margin,
        'target_base': float(target_base),
        'target_base_floor': float(target_base_floor),
        'econ_over_target_ratio': float(econ_over_target_ratio),
        'n_active_targets': int(n_active_targets),
        'cv_target_active': [bool(v) for v in cv_target_active],
        'mv_target_active': [bool(v) for v in mv_target_active],
        'econ_budget': float(econ_budget),
        'target_budget': float(target_budget),
        'priority_budget': priority_budget,
        'mv_econ_total': float(mv_econ_total),
        'cv_econ_total': float(cv_econ_total),
    }


def _active_targets(spec: Dict, n_cv: int, n_mv: int) -> Tuple[List[bool], List[bool]]:
    """Return per-channel target-active flags derived from
    ``spec['runtime_setpoints']['targets_enabled']`` (for CVs) and the
    explicit presence of finite target values in ``spec['targets']``.

    A CV target is active only when:
      1. ``runtime_setpoints.targets_enabled`` includes True for that CV,
      2. AND ``spec['targets']['cvs']`` supplies a finite target value.

    An MV target is active only when the user **explicitly** supplied a
    finite target in ``spec['targets']['mvs']``. Coerced default values
    (e.g. all-zero ``weights.mv_target_values``) do not count.
    """
    cv_active = [False] * n_cv
    mv_active = [False] * n_mv
    if not isinstance(spec, dict):
        return cv_active, mv_active

    # ---- CV targets ----
    rs = spec.get('runtime_setpoints') or {}
    raw = rs.get('targets_enabled', False)
    if isinstance(raw, (list, tuple)):
        per_cv_flags = [bool(v) for v in list(raw)[:n_cv]]
        per_cv_flags = per_cv_flags + [False] * max(0, n_cv - len(per_cv_flags))
    else:
        per_cv_flags = [bool(raw)] * n_cv

    targets = spec.get('targets') or {}
    cv_tgts_explicit = targets.get('cvs') if isinstance(targets, dict) else None
    cv_target_values: List[float] = [float('nan')] * n_cv
    if isinstance(cv_tgts_explicit, dict):
        for k, v in cv_tgts_explicit.items():
            try:
                idx = int(str(k).lower().replace('cv_', ''))
                if 0 <= idx < n_cv:
                    cv_target_values[idx] = _safe_float(v, float('nan'))
            except Exception:
                continue
    elif isinstance(cv_tgts_explicit, list):
        for i, v in enumerate(cv_tgts_explicit[:n_cv]):
            cv_target_values[i] = _safe_float(v, float('nan'))
    for i in range(n_cv):
        cv_active[i] = bool(per_cv_flags[i] and np.isfinite(cv_target_values[i]))

    # ---- MV targets (explicit only; no targets_enabled gate in schema) ----
    mv_tgts_explicit = targets.get('mvs') if isinstance(targets, dict) else None
    mv_target_values: List[float] = [float('nan')] * n_mv
    if isinstance(mv_tgts_explicit, dict):
        for k, v in mv_tgts_explicit.items():
            try:
                idx = int(str(k).lower().replace('mv_', ''))
                if 0 <= idx < n_mv:
                    mv_target_values[idx] = _safe_float(v, float('nan'))
            except Exception:
                continue
    elif isinstance(mv_tgts_explicit, list):
        for i, v in enumerate(mv_tgts_explicit[:n_mv]):
            mv_target_values[i] = _safe_float(v, float('nan'))
    for i in range(n_mv):
        mv_active[i] = bool(np.isfinite(mv_target_values[i]))

    return cv_active, mv_active
