"""Auto-derive objective weights from priority rank, control-speed preset,
and identified per-MV dynamics.

Replaces hand-tuned ``mv_violation_weights``, ``cv_violation_weights``,
``mv_move_weights``, and ``mv/cv_target_weights`` in
``control_objective.json``.

Priority ordering (enforced by construction, April 2026 quadratic refactor):

    1. MV limits         (most important -- never violate)
    2. CV limits by rank (rank 1 highest, geometric decay per rank)
    3. Economics         (mv/cv economic nudge)
    4. Targets           (mv/cv target tracking, if targets_enabled)

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
- ``OBJ_AUTO_MOVE_OVER_CV_K``         (default 20.0; move_base = cv_base / K)
- ``OBJ_AUTO_ECON_OVER_TARGET_RATIO`` (default 2.0; econ budget vs target budget)
- ``OBJ_AUTO_TARGET_BASE``            (default 0.5; target_base floor if no econ)
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

from utils.control_speed import get_control_speed_factors


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except Exception:
        return float(default)


def _cv_rank_order(spec: Dict, n_cv: int) -> List[int]:
    """Map state-order CV index -> rank (1 = most important).

    Looks for ``spec['cv_priority']`` which is a list of CV names
    (``cv_0``, ``cv_1`` ...) in decreasing importance.  Missing names get
    the tail ranks.
    """
    raw = spec.get('cv_priority') if isinstance(spec, dict) else None
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
        # Append any missing CV indices at the end (lowest priority).
        for i in range(n_cv):
            if i not in seen:
                seen.append(i)
        order = seen
    # order[k] is the state-index of the k-th most important CV.
    # Return rank per state index.
    ranks = [0] * n_cv
    for rank_minus_one, state_idx in enumerate(order):
        if 0 <= state_idx < n_cv:
            ranks[state_idx] = rank_minus_one + 1
    for i in range(n_cv):
        if ranks[i] == 0:
            ranks[i] = n_cv  # any untouched gets last rank
    return ranks


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
                        dynamics: Optional[Dict] = None,
                        control_speed: Optional[str] = None) -> Dict:
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
        speed: str
        tau_per_mv: list[float]
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
    # Quadratic cap: cv_base * typical_violation² <= cap_frac * reward_clip
    cv_base_cap = (cap_frac * reward_clip) / (typical_cv_viol_frac * typical_cv_viol_frac)
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
    # The CV violation penalty is now quadratic in the normalised
    # bound-violation magnitude. For the CV-limit priority to dominate
    # economics AND targets, the per-step CV penalty at the tolerance
    # violation magnitude must exceed the per-step economic + target
    # budget by ``margin``:
    #
    #   cv_base * tolerance²  >=  margin * priority_budget
    #
    # priority_budget is the max of econ_budget and target_budget; we
    # then size target_base from econ_budget (if present) so the
    # hierarchy econ > targets holds automatically.
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

    # Derive target_base so econ dominates targets by ratio. If the user
    # supplied no economic weights (pure target-tracking task), fall back
    # to the floor so target tracking still has a usable penalty.
    if n_active_targets > 0:
        if econ_budget > 0.0:
            target_base_adaptive = econ_budget / (econ_over_target_ratio * n_active_targets)
            target_base = max(target_base_adaptive, 0.0)
        else:
            target_base = target_base_floor
    else:
        target_base = 0.0

    # Max per-step target penalty (linear L1). Deviation peaks at 1.0 in
    # normalised space (channel at one extreme of norm range, target at
    # the other).
    target_budget = float(n_active_targets) * target_base * 1.0

    priority_budget = float(max(econ_budget, target_budget))

    cv_base_adaptive = 0.0
    if priority_budget > 0.0:
        cv_base_adaptive = (margin * priority_budget) / (tolerance * tolerance)
    cv_base_raw = float(max(cv_base_floor_quadratic, cv_base_adaptive))
    cv_base = float(min(cv_base_raw, cv_base_cap))
    cv_base_capped = bool(cv_base < cv_base_raw - 1e-9)
    if cv_base_capped:
        try:
            print(
                f"[auto_weights] cv_violation_base capped: "
                f"adaptive={cv_base_raw:.2f} -> {cv_base:.2f} "
                f"(cap={cv_base_cap:.2f}, reward_clip={reward_clip:.1f}, "
                f"cap_frac={cap_frac:.2f}, typical_viol={typical_cv_viol_frac:.3f}). "
                f"Economic weights may exceed what the reward clip can express; "
                f"raise OBJECTIVE_REWARD_CLIP or shrink mv/cv_economic to keep "
                f"strict CV priority."
            )
        except Exception:
            pass
    # Enforce priority #1: MV limits dominate CV limits by construction.
    mv_base = float(max(mv_base_floor_quadratic, mv_over_cv_ratio * cv_base))

    if dynamics is None:
        dynamics = _load_dynamics_json()
    factors = get_control_speed_factors(control_speed or spec.get('control_speed'))

    ranks = _cv_rank_order(spec, n_cv)
    taus = _per_mv_tau(n_mv, dynamics)
    median_tau = float(np.median(taus)) if taus else 1.0
    median_tau = max(1e-3, median_tau)

    cv_weights = [float(cv_base * (rank_decay ** max(0, r - 1))) for r in ranks]
    mv_weights = [float(mv_base) for _ in range(n_mv)]
    # Adaptive move_base: target a fixed-fraction-of-reward-clip steady-
    # state cost from actor jitter (sigma_ref). The legacy cv_base/K
    # ratio is retained as a soft *upper* cap so move never exceeds the
    # historical formula -- prevents pathological numbers if reward_clip
    # or sigma_ref are mis-set. Hard floor at OBJ_AUTO_MOVE_BASE.
    move_base_legacy_cap = float(cv_base / move_cv_ratio_k)
    move_base = float(max(move_base_floor, min(move_base_adaptive, move_base_legacy_cap)))
    move_weights = [
        float(move_base * max(0.2, tau / median_tau) * factors.move_penalty_scale)
        for tau in taus
    ]
    mv_target_weights = [float(target_base if v else 0.0) for v in mv_target_active]
    cv_target_weights = [float(target_base if v else 0.0) for v in cv_target_active]

    return {
        'mv_violation_weights': mv_weights,
        'cv_violation_weights': cv_weights,
        'mv_move_weights': move_weights,
        'mv_target_weights': mv_target_weights,
        'cv_target_weights': cv_target_weights,
        'cv_ranks': ranks,
        'speed': factors.name,
        'tau_per_mv': taus,
        'median_tau': median_tau,
        'mv_violation_base': mv_base,
        'cv_violation_base': cv_base,
        'mv_violation_base_floor_linear': mv_base_env,
        'cv_violation_base_floor_linear': cv_base_env,
        'mv_violation_base_floor_quadratic': mv_base_floor_quadratic,
        'cv_violation_base_floor_quadratic': cv_base_floor_quadratic,
        'mv_violation_base_adaptive': float(mv_over_cv_ratio * cv_base),
        'cv_violation_base_adaptive': float(cv_base_adaptive),
        'cv_violation_base_raw': float(cv_base_raw),
        'cv_violation_base_cap': float(cv_base_cap),
        'cv_violation_base_capped': bool(cv_base_capped),
        'cv_penalty_cap_frac': float(cap_frac),
        'typical_cv_violation_frac': float(typical_cv_viol_frac),
        'reward_clip_used_for_cap': float(reward_clip),
        'move_over_cv_k': float(move_cv_ratio_k),
        'cv_rank_decay': rank_decay,
        'move_base': move_base,
        'move_base_adaptive': float(move_base_adaptive),
        'move_base_legacy_cap': float(move_base_legacy_cap),
        'move_base_floor': float(move_base_floor),
        'move_target_cost_frac': float(move_target_cost_frac),
        'move_sigma_ref': float(move_sigma_ref),
        'move_expected_step_jitter': float(expected_step_jitter),
        'speed_move_scale': factors.move_penalty_scale,
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
