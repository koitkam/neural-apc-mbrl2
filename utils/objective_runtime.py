"""Runtime objective computation helpers for generic control workflows.

Computes per-step reward from constraint-violation, MV-move, economic, and
production/energy terms, using simulator metadata and objective weights.

Key design notes (April 2026 refactor):
- MV and CV violation penalties are **quadratic in the (ReLU) bound-
  violation magnitude**. Penalty is exactly zero inside bounds and grows
  as ``(max(0, lo - x))^2 + (max(0, x - hi))^2``. The quadratic shape
  gives an increasing gradient as the violation grows, making limit
  compliance the dominant driver during optimization.
- MV and CV economic terms are **clipped to the current bounds** before
  the economic reward is computed, so there is no economic gradient
  outside the limits. Only the bound-violation term contributes to the
  gradient outside bounds.
- MV and CV target penalties (when auto-derived weights are non-zero)
  remain **linear** (L1) in the target deviation, keeping a soft drive
  toward the operating setpoint without dominating the quadratic bound
  penalties.
- Auto-scaling priority enforced in ``utils.auto_weights``:
  ``MV limits  >  CV limits (by rank)  >  economics  >  targets``.
- MV violation weights, CV violation weights, MV move weights, and
  CV/MV target weights are **auto-derived at runtime** via
  ``utils.auto_weights.derive_auto_weights`` if not already present in
  ``obj_w``. This means control_objective.json no longer needs
  ``*_violation_weights`` / ``mv_move_weights`` / ``*_target_weights``.
- Bounds and CV targets can come from a :class:`RuntimeSetpointManager`
  passed as ``setpoint_manager``; if provided, its ``current_mv_bounds``,
  ``current_cv_bounds`` and ``current_cv_targets`` take precedence over the
  static ``bounds`` argument so the objective tracks operator setpoint
  changes during an episode.
- ``estimate_reward_scale`` derives a per-step O(5) reward target so any
  plant trains with similar Q-value magnitudes.
"""

import os
from typing import Dict, List, Optional, Tuple

import numpy as np

from utils.state_normalization import state_value_in_mode


def _safe_float(v, default=0.0):
    try:
        return float(v)
    except Exception:
        return float(default)


def _finite(v: float, default: float = 0.0) -> float:
    out = _safe_float(v, default)
    return out if np.isfinite(out) else float(default)


def _clip(v: float, lo: float, hi: float) -> float:
    return float(np.clip(_finite(v, 0.0), float(lo), float(hi)))


def _saturate_one_sided(v: float, cap: float, mode: str) -> float:
    """Saturate a non-negative penalty at ``cap``.

    ``mode='hard'`` reproduces ``np.clip(v, 0, cap)``.
    ``mode='tanh'`` applies a smooth saturation ``cap * tanh(v / cap)`` so
    the gradient never vanishes for large ``v``. The output is always in
    ``[0, cap)`` for non-negative ``v`` and matches the linear region for
    ``v << cap``.
    """
    val = max(0.0, _finite(v, 0.0))
    cap = max(0.0, float(cap))
    if cap <= 0.0:
        return 0.0
    if mode == 'tanh':
        return float(cap * np.tanh(val / cap))
    return float(min(val, cap))


def _saturate_two_sided(v: float, cap: float, mode: str) -> float:
    """Saturate a signed term at ``[-cap, +cap]``.

    Symmetric counterpart of :func:`_saturate_one_sided`.
    """
    val = _finite(v, 0.0)
    cap = max(0.0, float(cap))
    if cap <= 0.0:
        return 0.0
    if mode == 'tanh':
        return float(cap * np.tanh(val / cap))
    return float(np.clip(val, -cap, cap))


def _resolve_mv_bounds(bounds: Dict, action_dim: int,
                       setpoint_manager=None) -> List[List[float]]:
    if setpoint_manager is not None:
        arr = getattr(setpoint_manager, 'current_mv_bounds', None)
        if arr is not None and len(arr) >= action_dim:
            return [[float(arr[i, 0]), float(arr[i, 1])] for i in range(action_dim)]
    mv_bounds = list(bounds.get('mv_bounds', []))
    if len(mv_bounds) < action_dim:
        mv_bounds = mv_bounds + [[0.0, 100.0] for _ in range(action_dim - len(mv_bounds))]
    return mv_bounds[:action_dim]


def _resolve_cv_bounds(bounds: Dict, cv_dim: int,
                       setpoint_manager=None) -> List[List[float]]:
    if setpoint_manager is not None:
        arr = getattr(setpoint_manager, 'current_cv_bounds', None)
        if arr is not None and len(arr) >= cv_dim:
            return [[float(arr[i, 0]), float(arr[i, 1])] for i in range(cv_dim)]
    cv_bounds = list(bounds.get('cv_bounds', []))
    if len(cv_bounds) < cv_dim:
        cv_bounds = cv_bounds + [[-1e12, 1e12] for _ in range(cv_dim - len(cv_bounds))]
    return cv_bounds[:cv_dim]


def _resolve_vector(values, length: int, default: float) -> List[float]:
    if not isinstance(values, list):
        values = []
    out = [_safe_float(v, default) for v in values]
    if len(out) < length:
        out = out + [float(default) for _ in range(length - len(out))]
    return out[:length]


def _resolve_ranges(raw_ranges, dim: int, fallback_bounds: List[List[float]]) -> List[List[float]]:
    out = []
    src = raw_ranges if isinstance(raw_ranges, list) else []
    for i in range(dim):
        if i < len(src) and isinstance(src[i], (list, tuple)) and len(src[i]) >= 2:
            lo = _safe_float(src[i][0], 0.0)
            hi = _safe_float(src[i][1], 1.0)
        elif i < len(fallback_bounds):
            lo = _safe_float(fallback_bounds[i][0], 0.0)
            hi = _safe_float(fallback_bounds[i][1], 1.0)
        else:
            lo, hi = 0.0, 1.0
        if hi <= lo:
            hi = lo + 1.0
        out.append([lo, hi])
    return out


def _normalize(value: float, lo: float, hi: float) -> float:
    return (float(value) - float(lo)) / max(1e-6, float(hi) - float(lo))


def _normalized_bounds(lo: float, hi: float, r_lo: float, r_hi: float) -> Tuple[float, float]:
    return _normalize(lo, r_lo, r_hi), _normalize(hi, r_lo, r_hi)


def _objective_uses_normalized(obj_w: Dict, terms: Dict) -> bool:
    if isinstance(terms, dict) and 'objective_use_normalized' in terms:
        return bool(int(_safe_float(terms.get('objective_use_normalized', 1), 1)))
    return bool(int(_safe_float(obj_w.get('objective_use_normalized', 1), 1)))


def _maybe_auto_weights(obj_w: Dict, n_mv: int, n_cv: int, spec: Optional[Dict]) -> Dict:
    """Fill in violation/move/target weight vectors if absent, using auto-derivation."""
    needs = (
        not obj_w.get('mv_violation_weights')
        or not obj_w.get('cv_violation_weights')
        or not obj_w.get('mv_move_weights')
        or not obj_w.get('cv_target_weights')
        or not obj_w.get('mv_target_weights')
    )
    if not needs:
        return obj_w
    try:
        from utils.auto_weights import derive_auto_weights
    except Exception:
        return obj_w
    spec_local = spec if isinstance(spec, dict) else {}
    auto = derive_auto_weights(spec_local, n_mv=n_mv, n_cv=n_cv)
    merged = dict(obj_w)
    for k in ('mv_violation_weights', 'cv_violation_weights', 'mv_move_weights',
              'mv_target_weights', 'cv_target_weights'):
        if not merged.get(k):
            merged[k] = list(auto.get(k) or [])
    return merged


def compute_objective_components(
    state,
    sim,
    control,
    prev_control,
    obj_w: Dict,
    bounds: Dict,
    terms: Dict = None,
    setpoint_manager=None,
    objective_spec: Optional[Dict] = None,
) -> Dict[str, float]:
    state = np.asarray(state, dtype='float32').reshape(-1)
    control = np.asarray(control, dtype='float32').reshape(-1)
    prev_control = np.asarray(prev_control, dtype='float32').reshape(-1)

    mv_dim = len(control)
    cv_indices = [int(i) for i in list(getattr(sim, 'cv_indices', []))]
    cv_dim = len(cv_indices)
    obj_w = _maybe_auto_weights(obj_w, n_mv=mv_dim, n_cv=cv_dim, spec=objective_spec)

    mv_bounds = _resolve_mv_bounds(bounds, action_dim=mv_dim, setpoint_manager=setpoint_manager)
    cv_bounds = _resolve_cv_bounds(bounds, cv_dim=cv_dim, setpoint_manager=setpoint_manager)
    use_normalized = _objective_uses_normalized(obj_w, terms)
    mv_norm_ranges = _resolve_ranges(getattr(sim, 'mv_normalization_ranges', []), mv_dim, mv_bounds)
    cv_norm_ranges = _resolve_ranges(getattr(sim, 'cv_normalization_ranges', []), cv_dim, cv_bounds)
    if not isinstance(terms, dict):
        terms = {}

    mv_violation_weights = _resolve_vector(obj_w.get('mv_violation_weights', []), mv_dim, 0.0)
    cv_violation_weights = _resolve_vector(obj_w.get('cv_violation_weights', []), cv_dim,
                                           _safe_float(obj_w.get('cv_violation', 0.0), 0.0))
    mv_move_weights = _resolve_vector(obj_w.get('mv_move_weights', []), mv_dim, 0.0)
    mv_economic_weights = _resolve_vector(obj_w.get('mv_economic_weights', []), mv_dim, 0.0)
    cv_economic_weights = _resolve_vector(obj_w.get('cv_economic_weights', []), cv_dim, 0.0)
    mv_target_weights = _resolve_vector(obj_w.get('mv_target_weights', []), mv_dim, 0.0)
    cv_target_weights = _resolve_vector(obj_w.get('cv_target_weights', []), cv_dim, 0.0)
    mv_target_values = _resolve_vector(obj_w.get('mv_target_values', []), mv_dim, 0.0)

    # CV targets: prefer setpoint manager (per-step authoritative).
    if setpoint_manager is not None:
        arr = getattr(setpoint_manager, 'current_cv_targets', None)
        if arr is not None and len(arr) >= cv_dim:
            cv_target_values = [float(arr[i]) for i in range(cv_dim)]
        else:
            cv_target_values = _resolve_vector(obj_w.get('cv_target_values', []), cv_dim, 0.0)
    else:
        cv_target_values = _resolve_vector(obj_w.get('cv_target_values', []), cv_dim, 0.0)

    # ---- MV violations (quadratic ReLU, zero inside bounds) ----
    mv_violation_per_channel = []
    for i in range(mv_dim):
        lo_i, hi_i = float(mv_bounds[i][0]), float(mv_bounds[i][1])
        r_lo, r_hi = mv_norm_ranges[i]
        u_term = float(control[i])
        lo_term, hi_term = lo_i, hi_i
        if use_normalized:
            lo_term, hi_term = _normalized_bounds(lo_term, hi_term, r_lo, r_hi)
            u_term = _normalize(u_term, r_lo, r_hi)
        lo_viol = max(0.0, lo_term - u_term)
        hi_viol = max(0.0, u_term - hi_term)
        violation = lo_viol * lo_viol + hi_viol * hi_viol
        mv_violation_per_channel.append(float(violation))
    mv_violation_penalty = float(
        np.sum(np.asarray(mv_violation_per_channel, dtype='float32')
               * np.asarray(mv_violation_weights, dtype='float32'))
    )

    # ---- CV violations (quadratic ReLU, zero inside bounds) ----
    cv_violation_per_channel = []
    for j, sidx in enumerate(cv_indices):
        if sidx < 0 or sidx >= len(state):
            cv_violation_per_channel.append(0.0)
            continue
        lo = float(cv_bounds[j][0])
        hi = float(cv_bounds[j][1])
        pv = float(state_value_in_mode(state, sim, sidx, use_normalized=use_normalized))
        r_lo, r_hi = cv_norm_ranges[j]
        if use_normalized:
            lo_n, hi_n = _normalized_bounds(lo, hi, r_lo, r_hi)
            lo_viol = max(0.0, lo_n - pv)
            hi_viol = max(0.0, pv - hi_n)
        else:
            lo_viol = max(0.0, lo - pv)
            hi_viol = max(0.0, pv - hi)
        violation = lo_viol * lo_viol + hi_viol * hi_viol
        cv_violation_per_channel.append(float(violation))
    cv_violation_penalty = float(
        np.sum(np.asarray(cv_violation_per_channel, dtype='float32')
               * np.asarray(cv_violation_weights, dtype='float32'))
    )
    cv_penalty = float(cv_violation_penalty)

    # ---- MV economic (clipped to bounds: no gradient outside limits) ----
    # Economic nudge only active inside [lo, hi]. Outside, the term is held
    # at its boundary value so the agent has zero economic incentive to
    # cross the limit -- only the quadratic violation penalty acts there.
    mv_economic_terms = []
    for i in range(mv_dim):
        lo_i, hi_i = float(mv_bounds[i][0]), float(mv_bounds[i][1])
        r_lo, r_hi = mv_norm_ranges[i]
        if use_normalized:
            u_raw = _normalize(float(control[i]), r_lo, r_hi)
            lo_term, hi_term = _normalized_bounds(lo_i, hi_i, r_lo, r_hi)
            u_clipped = float(np.clip(u_raw, lo_term, hi_term))
            term = u_clipped - 0.5
        else:
            u_clipped = float(np.clip(float(control[i]), lo_i, hi_i))
            term = u_clipped
        mv_economic_terms.append(float(term))
    mv_economic_penalty = float(
        np.sum(np.asarray(mv_economic_terms, dtype='float32')
               * np.asarray(mv_economic_weights, dtype='float32'))
    )

    # ---- MV move (linear, unchanged) ----
    mv_move_terms = []
    for i in range(mv_dim):
        r_lo, r_hi = mv_norm_ranges[i]
        if use_normalized:
            u_term = _normalize(float(control[i]), r_lo, r_hi)
            p_term = _normalize(float(prev_control[i]), r_lo, r_hi)
        else:
            u_term = float(control[i])
            p_term = float(prev_control[i])
        mv_move_terms.append(float(abs(u_term - p_term)))
    mv_move_penalty = float(
        np.sum(np.asarray(mv_move_terms, dtype='float32')
               * np.asarray(mv_move_weights, dtype='float32'))
    )

    # ---- CV economic (clipped to bounds: no gradient outside limits) ----
    cv_economic_terms = []
    for j, sidx in enumerate(cv_indices):
        if sidx < 0 or sidx >= len(state):
            cv_economic_terms.append(0.0)
            continue
        lo_j, hi_j = float(cv_bounds[j][0]), float(cv_bounds[j][1])
        r_lo, r_hi = cv_norm_ranges[j]
        y_raw = float(state_value_in_mode(state, sim, sidx, use_normalized=use_normalized))
        if use_normalized:
            lo_term, hi_term = _normalized_bounds(lo_j, hi_j, r_lo, r_hi)
            y_clipped = float(np.clip(y_raw, lo_term, hi_term))
            term = y_clipped - 0.5
        else:
            y_clipped = float(np.clip(y_raw, lo_j, hi_j))
            term = y_clipped
        cv_economic_terms.append(float(term))
    cv_economic_penalty = float(
        np.sum(np.asarray(cv_economic_terms, dtype='float32')
               * np.asarray(cv_economic_weights, dtype='float32'))
    )

    # ---- Target tracking (linear / L1 when enabled) ----
    mv_target_terms = []
    for i in range(mv_dim):
        r_lo, r_hi = mv_norm_ranges[i]
        if use_normalized:
            u_term = _normalize(float(control[i]), r_lo, r_hi)
            t_term = _normalize(float(mv_target_values[i]), r_lo, r_hi)
        else:
            u_term = float(control[i])
            t_term = float(mv_target_values[i])
        mv_target_terms.append(float(abs(u_term - t_term)))
    mv_target_penalty = float(
        np.sum(np.asarray(mv_target_terms, dtype='float32')
               * np.asarray(mv_target_weights, dtype='float32'))
    )

    cv_target_terms = []
    for j, sidx in enumerate(cv_indices):
        if sidx < 0 or sidx >= len(state):
            cv_target_terms.append(0.0)
            continue
        y_raw = float(state_value_in_mode(state, sim, sidx, use_normalized=False))
        if use_normalized:
            r_lo, r_hi = cv_norm_ranges[j]
            y_term = _normalize(y_raw, r_lo, r_hi)
            t_term = _normalize(float(cv_target_values[j]), r_lo, r_hi)
        else:
            y_term = y_raw
            t_term = float(cv_target_values[j])
        cv_target_terms.append(float(abs(y_term - t_term)))
    cv_target_penalty = float(
        np.sum(np.asarray(cv_target_terms, dtype='float32')
               * np.asarray(cv_target_weights, dtype='float32'))
    )

    production_idx = 0
    production_signal = 0.0
    production_term = 0.0
    energy_term = 0.0
    movement_term = float(np.mean(np.asarray(mv_move_terms, dtype='float32'))) if mv_move_terms else 0.0

    penalty_clip = float(os.environ.get('OBJECTIVE_PENALTY_CLIP', '250.0'))
    reward_clip = float(os.environ.get('OBJECTIVE_REWARD_CLIP', '250.0'))
    sat_mode = str(os.environ.get('OBJECTIVE_PENALTY_SAT_MODE', 'tanh')).strip().lower()
    if sat_mode not in ('hard', 'tanh'):
        sat_mode = 'tanh'

    mv_violation_penalty = _saturate_one_sided(mv_violation_penalty, penalty_clip, sat_mode)
    cv_violation_penalty = _saturate_one_sided(cv_violation_penalty, penalty_clip, sat_mode)
    mv_economic_penalty = _saturate_two_sided(mv_economic_penalty, penalty_clip, sat_mode)
    cv_economic_penalty = _saturate_two_sided(cv_economic_penalty, penalty_clip, sat_mode)
    mv_target_penalty = _saturate_one_sided(mv_target_penalty, penalty_clip, sat_mode)
    cv_target_penalty = _saturate_one_sided(cv_target_penalty, penalty_clip, sat_mode)
    mv_move_penalty = _saturate_one_sided(mv_move_penalty, penalty_clip, sat_mode)
    movement_term = _saturate_two_sided(movement_term, penalty_clip, sat_mode)

    reward = 0.0
    reward -= mv_violation_penalty
    reward -= cv_violation_penalty
    reward -= mv_economic_penalty
    reward -= cv_economic_penalty
    reward -= mv_target_penalty
    reward -= cv_target_penalty
    reward -= mv_move_penalty
    reward -= _safe_float(obj_w.get('movement', 0.0), 0.0) * movement_term
    if 'cv_violation' in obj_w:
        reward -= _safe_float(obj_w.get('cv_violation', 0.0), 0.0) * _saturate_one_sided(cv_penalty, penalty_clip, sat_mode)
    reward = _saturate_two_sided(reward, reward_clip, sat_mode)

    return {
        'prod_term': float(production_term),
        'energy_term': float(energy_term),
        'movement_term': float(movement_term),
        'mv_violation_per_channel': [float(x) for x in mv_violation_per_channel],
        'cv_violation_per_channel': [float(x) for x in cv_violation_per_channel],
        'mv_violation_penalty': float(mv_violation_penalty),
        'cv_violation_penalty': float(cv_violation_penalty),
        'mv_economic_terms': [float(x) for x in mv_economic_terms],
        'cv_economic_terms': [float(x) for x in cv_economic_terms],
        'mv_economic_penalty': float(mv_economic_penalty),
        'cv_economic_penalty': float(cv_economic_penalty),
        'mv_target_terms': [float(x) for x in mv_target_terms],
        'cv_target_terms': [float(x) for x in cv_target_terms],
        'mv_target_penalty': float(mv_target_penalty),
        'cv_target_penalty': float(cv_target_penalty),
        'mv_move_terms': [float(x) for x in mv_move_terms],
        'mv_move_penalty': float(mv_move_penalty),
        'cv_penalty': float(cv_penalty),
        'reward': float(reward),
        'production_signal': float(production_signal),
        'production_state_index': int(production_idx),
        'cv_penalties': [float(x) for x in cv_violation_per_channel],
    }


def estimate_reward_scale(obj_w: Dict, use_normalized: bool = True) -> Tuple[float, float]:
    """Estimate reward_scale and penalty_clip from the objective weights.

    Returns ``(reward_scale, penalty_clip)``.

    Updated for April 2026 quadratic refactor:
    - MV/CV violations are quadratic: penalty ~ weight * violation².
    - Targets are linear: penalty ~ weight * |deviation|.
    - Economic terms are clipped to bounds; contribution ~ weight * 0.5.
    """
    if not use_normalized:
        return 1.0, 250.0

    typical_cv_violation = 0.05
    worst_cv_violation = 0.20
    typical_mv_violation = 0.01
    worst_mv_violation = 0.10
    typical_target_dev = 0.10
    worst_target_dev = 0.25
    typical_econ_dev = 0.05
    worst_econ_dev = 0.25
    typical_move = 0.02
    worst_move = 0.10

    mv_tw = sum(abs(_safe_float(w)) for w in (obj_w.get('mv_target_weights') or []))
    cv_tw = sum(abs(_safe_float(w)) for w in (obj_w.get('cv_target_weights') or []))
    mv_ew = sum(abs(_safe_float(w)) for w in (obj_w.get('mv_economic_weights') or []))
    cv_ew = sum(abs(_safe_float(w)) for w in (obj_w.get('cv_economic_weights') or []))
    mv_mw = sum(abs(_safe_float(w)) for w in (obj_w.get('mv_move_weights') or []))
    cv_vw = sum(abs(_safe_float(w)) for w in (obj_w.get('cv_violation_weights') or []))
    mv_vw = sum(abs(_safe_float(w)) for w in (obj_w.get('mv_violation_weights') or []))
    scalar_cv_viol = abs(_safe_float(obj_w.get('cv_violation', 0.0), 0.0))

    est_typical = 0.0
    # Quadratic violation contributions.
    est_typical += cv_vw * (typical_cv_violation ** 2)
    est_typical += scalar_cv_viol * typical_cv_violation
    est_typical += mv_vw * (typical_mv_violation ** 2)
    # Linear target contributions.
    est_typical += (mv_tw + cv_tw) * typical_target_dev
    est_typical += (mv_ew + cv_ew) * typical_econ_dev
    est_typical += mv_mw * typical_move

    est_worst = 0.0
    est_worst += cv_vw * (worst_cv_violation ** 2)
    est_worst += scalar_cv_viol * worst_cv_violation
    est_worst += mv_vw * (worst_mv_violation ** 2)
    est_worst += (mv_tw + cv_tw) * worst_target_dev
    est_worst += (mv_ew + cv_ew) * worst_econ_dev
    est_worst += mv_mw * worst_move

    override_scale = os.environ.get('REWARD_SCALE', '').strip()
    override_clip = os.environ.get('OBJECTIVE_PENALTY_CLIP', '').strip()
    target_magnitude = float(os.environ.get('REWARD_SCALE_TARGET', '5.0'))

    if override_scale:
        try:
            reward_scale = max(1.0, float(override_scale))
        except ValueError:
            reward_scale = 1.0
    elif est_typical < 1e-8:
        reward_scale = 1.0
    else:
        reward_scale = float(np.clip(target_magnitude / est_typical, 1.0, 200.0))

    if override_clip:
        try:
            penalty_clip = max(10.0, float(override_clip))
        except ValueError:
            penalty_clip = 250.0
    else:
        worst_scaled = est_worst * reward_scale
        penalty_clip = float(np.clip(worst_scaled / 0.8, 50.0, 5000.0))

    return float(reward_scale), float(penalty_clip)
