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


def _maybe_auto_weights(obj_w: Dict, n_mv: int, n_cv: int, spec: Optional[Dict],
                        mv_bounds: Optional[list] = None,
                        cv_bounds: Optional[list] = None,
                        mv_norm_ranges: Optional[list] = None,
                        cv_norm_ranges: Optional[list] = None) -> Dict:
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
    auto = derive_auto_weights(spec_local, n_mv=n_mv, n_cv=n_cv,
                               mv_bounds=mv_bounds, cv_bounds=cv_bounds,
                               mv_norm_ranges=mv_norm_ranges,
                               cv_norm_ranges=cv_norm_ranges)
    merged = dict(obj_w)
    for k in ('mv_violation_weights', 'cv_violation_weights', 'mv_move_weights',
              'mv_target_weights', 'cv_target_weights',
              'cv_violation_weights_lo', 'cv_violation_weights_hi'):
        if not merged.get(k):
            merged[k] = list(auto.get(k) or [])
    # Stash scalar auto-derived knobs that have no per-channel form.
    # Consumed by ``compute_objective_components`` as the default in the
    # env > spec > auto-derived > 0.0 resolution chain for the DMC
    # sliding-mode rate term.
    if 'auto_violation_rate_coef' not in merged:
        merged['auto_violation_rate_coef'] = float(auto.get('violation_rate_coef', 0.0))
    return merged


def resolve_integral_config(objective_spec=None) -> Tuple[bool, float, float]:
    """Resolve the integral (accumulated-violation) term configuration.

    Shared single source of truth for both
    :func:`compute_objective_components` (which applies the penalty) and
    the environment (which sizes/normalises the exposed accumulator
    observation channel).  Precedence: env var > objective_spec key >
    default — identical to the in-function resolution.

    Returns ``(enabled, coef, windup)``.
    """
    spec = objective_spec if isinstance(objective_spec, dict) else {}

    def _r(env_key: str, spec_key: str, default):
        v = os.environ.get(env_key)
        if v is not None and str(v).strip() != '':
            return v
        if spec_key in spec and spec.get(spec_key) is not None:
            return spec.get(spec_key)
        return default

    # ON by default (opt-out): a small positive coefficient applies a
    # dwell penalty for SUSTAINED limit violation, attacking the passive
    # "park just outside the bound" attractor. Opt out by setting the coef
    # to 0 (env ``OBJECTIVE_INTEGRAL_COEF`` or spec ``integral_coef``).
    coef = float(_safe_float(_r('OBJECTIVE_INTEGRAL_COEF', 'integral_coef', 0.05), 0.05))
    windup = float(_safe_float(_r('OBJECTIVE_INTEGRAL_WINDUP', 'integral_windup', 5.0), 5.0))
    if windup <= 0.0:
        windup = 5.0
    return (coef > 0.0, coef, windup)


def resolve_integral_leak(objective_spec=None) -> float:
    """Resolve the in-band leak (bleed-off) factor for the integral
    accumulator.

    Anti-windup recovery: while a CV is actively violating its limit the
    accumulator integrates the violation depth in full (no leak — the
    sustained-violation pressure that defeats the passivity attractor is
    preserved).  Once the CV returns *in-band* (depth == 0) the accumulator
    is multiplied by this leak factor every step, so a transient excursion
    bleeds off exponentially (~``1/(1 - leak)`` step memory) instead of
    permanently taxing the rest of the episode.  This restores a positive
    "recovery" gradient for clearing a violation.

    Precedence: env ``OBJECTIVE_INTEGRAL_LEAK`` > spec ``integral_leak`` >
    default 0.98 (~50-step recovery memory).  Clamped to ``(0, 1]``; a
    value of 1.0 disables the leak (legacy hold-forever behaviour).
    """
    spec = objective_spec if isinstance(objective_spec, dict) else {}
    raw = os.environ.get('OBJECTIVE_INTEGRAL_LEAK')
    if raw is None or str(raw).strip() == '':
        raw = spec.get('integral_leak') if 'integral_leak' in spec else 0.98
    leak = float(_safe_float(raw, 0.98))
    if leak <= 0.0:
        leak = 0.98
    return float(min(1.0, leak))



def _shaping_linear_equiv_scale() -> float:
    """Quadratic→linear conversion scale for the integral / derivative
    shaping terms.

    The per-channel CV/MV violation weights (``cv_violation_base`` etc.)
    are sized for the **quadratic** instantaneous penalty ``w·depth²`` and
    are therefore very large (``base ≈ floor/tolerance``), so they dominate
    everything below them *at the tolerance-magnitude violation*.  The
    integral (accumulated depth ``I_t``) and derivative (depth growth) terms
    are **linear** in depth, so multiplying that quadratically-sized weight
    by a linear accumulator over-scales by ~``1/tolerance`` and pegs the
    penalty clip after a trivial dwell (e.g. ``I_t≈0.08`` here), turning the
    shaping term into an on/off cliff instead of a gradient.

    Rescaling the derived weight by the violation ``tolerance`` yields the
    **linear-equivalent** weight (``w·tolerance``): the shaping penalty then
    matches the quadratic base penalty exactly at a tolerance-magnitude
    violation and stays strictly bounded below it for smaller dwells,
    restoring a usable gradient.  Adaptive: it scales with the per-channel
    derived CV/MV violation weight.  Mirrors ``OBJ_AUTO_VIOLATION_TOLERANCE``
    (default 0.02).
    """
    try:
        return max(1e-4, float(os.environ.get('OBJ_AUTO_VIOLATION_TOLERANCE', 0.02)))
    except Exception:
        return 0.02


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
    prev_mv_violation_per_channel=None,
    prev_cv_violation_per_channel=None,
    prev_integral_cv_per_channel=None,
) -> Dict[str, float]:
    state = np.asarray(state, dtype='float32').reshape(-1)
    control = np.asarray(control, dtype='float32').reshape(-1)
    prev_control = np.asarray(prev_control, dtype='float32').reshape(-1)

    mv_dim = len(control)
    cv_indices = [int(i) for i in list(getattr(sim, 'cv_indices', []))]
    cv_dim = len(cv_indices)

    mv_bounds = _resolve_mv_bounds(bounds, action_dim=mv_dim, setpoint_manager=setpoint_manager)
    cv_bounds = _resolve_cv_bounds(bounds, cv_dim=cv_dim, setpoint_manager=setpoint_manager)
    mv_norm_ranges = _resolve_ranges(getattr(sim, 'mv_normalization_ranges', []), mv_dim, mv_bounds)
    cv_norm_ranges = _resolve_ranges(getattr(sim, 'cv_normalization_ranges', []), cv_dim, cv_bounds)

    # Auto-derive weights *after* bounds/ranges are known so the economic
    # budget sizing uses the same band-midpoint ``typical`` as the reward
    # engine (shared ``resolve_econ_typical``).
    obj_w = _maybe_auto_weights(obj_w, n_mv=mv_dim, n_cv=cv_dim, spec=objective_spec,
                                mv_bounds=mv_bounds, cv_bounds=cv_bounds,
                                mv_norm_ranges=mv_norm_ranges,
                                cv_norm_ranges=cv_norm_ranges)

    use_normalized = _objective_uses_normalized(obj_w, terms)
    if not isinstance(terms, dict):
        terms = {}

    mv_violation_weights = _resolve_vector(obj_w.get('mv_violation_weights', []), mv_dim, 0.0)
    cv_violation_weights = _resolve_vector(obj_w.get('cv_violation_weights', []), cv_dim,
                                           _safe_float(obj_w.get('cv_violation', 0.0), 0.0))
    # Per-side CV weights: asymmetric high vs low limit urgency derived
    # from optional ``cv_priority_lo`` / ``cv_priority_hi`` in the
    # objective spec.  Falls back to the symmetric vector when absent.
    # The quadratic CV violation penalty uses the symmetric
    # ``cv_violation_weights``; the per-side weights drive the urgency
    # scale (``max(w_lo, w_hi)``) of the optional derivative and integral
    # shaping terms.
    raw_lo = obj_w.get('cv_violation_weights_lo')
    if raw_lo:
        cv_violation_weights_lo = _resolve_vector(raw_lo, cv_dim, 0.0)
    else:
        cv_violation_weights_lo = list(cv_violation_weights)
    raw_hi = obj_w.get('cv_violation_weights_hi')
    if raw_hi:
        cv_violation_weights_hi = _resolve_vector(raw_hi, cv_dim, 0.0)
    else:
        cv_violation_weights_hi = list(cv_violation_weights)
    mv_move_weights = _resolve_vector(obj_w.get('mv_move_weights', []), mv_dim, 0.0)
    mv_economic_weights = _resolve_vector(obj_w.get('mv_economic_weights', []), mv_dim, 0.0)
    cv_economic_weights = _resolve_vector(obj_w.get('cv_economic_weights', []), cv_dim, 0.0)
    # Economic *typical* operating point (normalised [0, 1]). The economic
    # nudge measures deviation from this point (``x_norm - typical``)
    # instead of from the range midpoint. Shared single source of truth
    # with the adaptive weight derivation (``resolve_econ_typical``) so the
    # priority ladder stays consistent. Absent an explicit user value it
    # defaults to the normalised *band midpoint* (derived from bounds +
    # normalisation ranges), falling back to 0.5 only when those are
    # unavailable.
    try:
        from utils.auto_weights import resolve_econ_typical as _resolve_econ_typical
        mv_economic_typical, cv_economic_typical = _resolve_econ_typical(
            objective_spec, mv_dim, cv_dim,
            mv_bounds=mv_bounds, cv_bounds=cv_bounds,
            mv_norm_ranges=mv_norm_ranges, cv_norm_ranges=cv_norm_ranges)
    except Exception:
        mv_economic_typical = [0.5] * mv_dim
        cv_economic_typical = [0.5] * cv_dim
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

    # ---------------- Optional-term config resolution -------------------
    # Single reward shape (quadratic violations, L1 mv-move).  Two
    # OPTIONAL PID-style shaping terms can be switched on independently:
    #   - derivative (violation-rate) term  -> OBJECTIVE_VIOLATION_RATE_COEF
    #   - integral (accumulated-violation)  -> OBJECTIVE_INTEGRAL_COEF
    # Precedence for every knob: env var > objective_spec key > default.
    _spec_cfg = objective_spec if isinstance(objective_spec, dict) else {}

    def _resolve_cfg(env_key: str, spec_key: str, default):
        v = os.environ.get(env_key)
        if v is not None and str(v).strip() != '':
            return v
        if spec_key in _spec_cfg and _spec_cfg.get(spec_key) is not None:
            return _spec_cfg.get(spec_key)
        return default

    # ---- MV violations (quadratic in bound-violation depth) ----
    mv_violation_per_channel = []        # shaped magnitude (matches mode)
    mv_violation_per_channel_raw = []    # raw depth (lo_viol+hi_viol) for rate term
    mv_violation_penalty = 0.0
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
        lo_shaped, hi_shaped = lo_viol * lo_viol, hi_viol * hi_viol
        w_i = float(mv_violation_weights[i])
        mv_violation_penalty += w_i * (lo_shaped + hi_shaped)
        mv_violation_per_channel.append(float(lo_shaped + hi_shaped))
        mv_violation_per_channel_raw.append(float(lo_viol + hi_viol))
    mv_violation_penalty = float(mv_violation_penalty)

    # ---- CV violations (quadratic, symmetric per-side weights) ----
    cv_violation_per_channel = []
    cv_violation_per_channel_raw = []
    cv_violation_penalty = 0.0
    for j, sidx in enumerate(cv_indices):
        if sidx < 0 or sidx >= len(state):
            cv_violation_per_channel.append(0.0)
            cv_violation_per_channel_raw.append(0.0)
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
        lo_shaped, hi_shaped = lo_viol * lo_viol, hi_viol * hi_viol
        w_sym = float(cv_violation_weights[j])
        cv_violation_penalty += w_sym * (lo_shaped + hi_shaped)
        cv_violation_per_channel.append(float(lo_shaped + hi_shaped))
        cv_violation_per_channel_raw.append(float(lo_viol + hi_viol))
    cv_violation_penalty = float(cv_violation_penalty)
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
            term = u_clipped - float(mv_economic_typical[i])
        else:
            u_clipped = float(np.clip(float(control[i]), lo_i, hi_i))
            term = u_clipped
        mv_economic_terms.append(float(term))
    mv_economic_penalty = float(
        np.sum(np.asarray(mv_economic_terms, dtype='float32')
               * np.asarray(mv_economic_weights, dtype='float32'))
    )

    # ---- MV move (L1 |Δu|) ----
    mv_move_terms = []
    for i in range(mv_dim):
        r_lo, r_hi = mv_norm_ranges[i]
        if use_normalized:
            u_term = _normalize(float(control[i]), r_lo, r_hi)
            p_term = _normalize(float(prev_control[i]), r_lo, r_hi)
        else:
            u_term = float(control[i])
            p_term = float(prev_control[i])
        du = u_term - p_term
        mv_move_terms.append(float(abs(du)))
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
            term = y_clipped - float(cv_economic_typical[j])
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

    penalty_clip = float(os.environ.get('OBJECTIVE_PENALTY_CLIP', '50.0'))
    reward_clip = float(os.environ.get('OBJECTIVE_REWARD_CLIP', '50.0'))
    sat_mode = str(os.environ.get('OBJECTIVE_PENALTY_SAT_MODE', 'tanh')).strip().lower()
    if sat_mode not in ('hard', 'tanh'):
        sat_mode = 'tanh'
    # ---- Optional violation-rate (derivative) term ----
    # Sliding-mode "reaching law" shaping (Slotine & Li 1991): when the
    # per-channel violation depth is *growing*, add a penalty proportional
    # to the growth rate so the agent is pushed to arrest the excursion
    # before the depth term builds enough magnitude to dominate. One-sided
    # (no bonus for shrinking violations) so it cannot be exploited by
    # oscillating across a bound. CV channels use ``max(w_lo, w_hi)`` as
    # the urgency scale so the asymmetric per-side weights still drive it.
    # Precedence: ``OBJECTIVE_VIOLATION_RATE_COEF`` (env) > spec key
    # ``violation_rate_coef`` > default ``"auto"``. ON by default: the
    # default ``"auto"`` resolves to the dynamics-derived
    # ``auto_violation_rate_coef`` computed in ``utils.auto_weights``.
    # Opt out by setting the coefficient to ``0`` (env or spec).
    auto_rate_default = float(obj_w.get('auto_violation_rate_coef', 0.0))
    _rate_raw = _resolve_cfg('OBJECTIVE_VIOLATION_RATE_COEF', 'violation_rate_coef', 'auto')
    if isinstance(_rate_raw, str) and _rate_raw.strip().lower() == 'auto':
        violation_rate_coef = auto_rate_default
    else:
        violation_rate_coef = float(_safe_float(_rate_raw, 0.0))
    violation_rate_penalty = 0.0
    # Quadratic→linear conversion scale shared by the derivative (rate) and
    # integral shaping terms below; keeps their linear-in-depth penalties
    # bounded below the quadratic base penalty (prevents the dwell cliff).
    _shaping_lin = _shaping_linear_equiv_scale()
    if (violation_rate_coef > 0.0
            and prev_cv_violation_per_channel is not None):
        for j in range(cv_dim):
            prev_v = (float(prev_cv_violation_per_channel[j])
                      if j < len(prev_cv_violation_per_channel) else 0.0)
            cur_v = cv_violation_per_channel_raw[j]
            growth = max(0.0, cur_v - prev_v)
            w_lo = float(cv_violation_weights_lo[j])
            w_hi = float(cv_violation_weights_hi[j])
            # Linear-equivalent shaping weight (quadratic base × tolerance):
            # the rate term is linear in depth-growth, so size it against the
            # same tolerance-magnitude reference as the integral term.
            w_rate = max(w_lo, w_hi) * _shaping_lin
            violation_rate_penalty += violation_rate_coef * w_rate * growth
    if (violation_rate_coef > 0.0
            and prev_mv_violation_per_channel is not None):
        for i in range(mv_dim):
            prev_v = (float(prev_mv_violation_per_channel[i])
                      if i < len(prev_mv_violation_per_channel) else 0.0)
            cur_v = mv_violation_per_channel_raw[i]
            growth = max(0.0, cur_v - prev_v)
            w_rate_mv = float(mv_violation_weights[i]) * _shaping_lin
            violation_rate_penalty += (
                violation_rate_coef * w_rate_mv * growth)
    violation_rate_penalty = _saturate_one_sided(
        violation_rate_penalty, penalty_clip, sat_mode)

    # ---- Optional integral (accumulated-violation) term ----
    # PID-style "reset action": penalises SUSTAINED CV limit violation so
    # a passive policy that parks the CV just outside a limit pays a cost
    # that grows with dwell time (directly attacks the P74 passivity
    # attractor).  The accumulator I_t = clip(I_{t-1} + depth_t, 0, windup)
    # is anti-windup clamped and EXPOSED TO THE AGENT in the observation
    # (see DreamerEnv._build_obs_vec) so the reward stays Markov and the
    # world model can predict it.  CV-only (MVs are actuated within
    # bounds, so they do not drift out and accumulate).
    # ``OBJECTIVE_INTEGRAL_COEF`` (env) > spec ``integral_coef`` > default.
    # ON by default (see ``resolve_integral_config`` for the default coef);
    # opt out by setting the coefficient to ``0`` (env or spec).
    # Windup cap: ``OBJECTIVE_INTEGRAL_WINDUP`` > spec ``integral_windup``
    # > 5.0.  ``prev_integral_cv_per_channel`` carries I_{t-1} (owned and
    # reset per-episode by the env); the updated I_t is returned for the
    # env to store + expose next step.
    _intg_enabled, integral_coef, integral_windup = resolve_integral_config(objective_spec)
    integral_leak = resolve_integral_leak(objective_spec)
    integral_cv_per_channel = [0.0] * cv_dim
    integral_penalty = 0.0
    if integral_coef > 0.0:
        for j in range(cv_dim):
            prev_I = (float(prev_integral_cv_per_channel[j])
                      if (prev_integral_cv_per_channel is not None
                          and j < len(prev_integral_cv_per_channel)) else 0.0)
            depth = (cv_violation_per_channel_raw[j]
                     if j < len(cv_violation_per_channel_raw) else 0.0)
            if depth > 0.0:
                # Active violation: integrate depth in full (no leak) so the
                # sustained-violation pressure that defeats the passivity
                # attractor is preserved.
                I_t = float(np.clip(prev_I + depth, 0.0, integral_windup))
            else:
                # In-band: bleed the accumulator off so a transient excursion
                # recovers (~1/(1-leak) step memory) instead of permanently
                # taxing the episode — restores a recovery gradient.
                I_t = float(np.clip(prev_I * integral_leak, 0.0, integral_windup))
            integral_cv_per_channel[j] = I_t
            w_lo = float(cv_violation_weights_lo[j])
            w_hi = float(cv_violation_weights_hi[j])
            # Linear-equivalent shaping weight (quadratic base × tolerance):
            # keeps the accumulated-dwell penalty bounded below the quadratic
            # base penalty and proportional to the derived CV urgency.
            w_int = max(w_lo, w_hi) * _shaping_lin
            integral_penalty += integral_coef * w_int * I_t
    else:
        # Pass through the previous accumulator unchanged when the term is
        # off so the env's exposed channel stays well-defined (all-zero).
        if prev_integral_cv_per_channel is not None:
            for j in range(min(cv_dim, len(prev_integral_cv_per_channel))):
                integral_cv_per_channel[j] = float(prev_integral_cv_per_channel[j])
    integral_penalty = _saturate_one_sided(integral_penalty, penalty_clip, sat_mode)

    # In-band test (uses pre-saturation violations to avoid false
    # positives from tanh tail).
    in_band = (
        (sum(mv_violation_per_channel) + sum(cv_violation_per_channel))
        <= 1e-9
    )

    # ---- Feasibility gate (priority hierarchy, no toggle) ----
    # When the agent is outside ANY limit, exponentially suppress the
    # "optimisation" penalties (economic / target / move) so the only
    # active gradients are the limit-handling terms: violations plus the
    # derivative (reaching) and integral (dwell) terms. This makes the
    # MV/CV-limits > targets > economics ladder unconditional without a
    # reward-mode switch. ``feasibility = exp(-min(V, cap) / scale)`` with
    # ``V`` the summed pre-saturation quadratic violation depth across all
    # MV+CV channels; ``feasibility -> 1`` in-band and ``-> 0`` as the
    # excursion grows. ``cap`` bounds the argument so a huge excursion does
    # not underflow before the gate has fully closed.
    feas_cap = max(0.0, float(os.environ.get('OBJECTIVE_FEASIBILITY_CAP', '4.0')))
    feas_scale = max(1e-6, float(os.environ.get('OBJECTIVE_FEASIBILITY_SCALE', '0.08')))
    total_violation_norm = float(
        sum(mv_violation_per_channel) + sum(cv_violation_per_channel))
    feasibility = float(np.exp(-min(total_violation_norm, feas_cap) / feas_scale))
    # Apply the gate BEFORE saturation so the suppression acts on the raw
    # penalty magnitude (not the tanh-compressed value).
    mv_economic_penalty *= feasibility
    cv_economic_penalty *= feasibility
    mv_target_penalty *= feasibility
    cv_target_penalty *= feasibility
    mv_move_penalty *= feasibility
    movement_term *= feasibility

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
    reward -= mv_target_penalty
    reward -= cv_target_penalty
    reward -= mv_move_penalty
    reward -= violation_rate_penalty
    reward -= integral_penalty
    reward -= _safe_float(obj_w.get('movement', 0.0), 0.0) * movement_term
    if 'cv_violation' in obj_w:
        reward -= _safe_float(obj_w.get('cv_violation', 0.0), 0.0) * _saturate_one_sided(cv_penalty, penalty_clip, sat_mode)

    # Economic terms (always subtracted; sign convention encoded in weights).
    reward -= mv_economic_penalty
    reward -= cv_economic_penalty

    reward = _saturate_two_sided(reward, reward_clip, sat_mode)

    return {
        'prod_term': float(production_term),
        'energy_term': float(energy_term),
        'movement_term': float(movement_term),
        'mv_violation_per_channel': [float(x) for x in mv_violation_per_channel],
        'cv_violation_per_channel': [float(x) for x in cv_violation_per_channel],
        'mv_violation_per_channel_raw': [float(x) for x in mv_violation_per_channel_raw],
        'cv_violation_per_channel_raw': [float(x) for x in cv_violation_per_channel_raw],
        'mv_violation_penalty': float(mv_violation_penalty),
        'cv_violation_penalty': float(cv_violation_penalty),
        'cv_violation_weights_lo': [float(x) for x in cv_violation_weights_lo],
        'cv_violation_weights_hi': [float(x) for x in cv_violation_weights_hi],
        'reward_mode': 'legacy',
        'feasibility': float(feasibility),
        'total_violation_norm': float(total_violation_norm),
        'violation_rate_coef': float(violation_rate_coef),
        'violation_rate_penalty': float(violation_rate_penalty),
        'integral_coef': float(integral_coef),
        'integral_cv_per_channel': [float(x) for x in integral_cv_per_channel],
        'integral_penalty': float(integral_penalty),
        'mv_economic_terms': [float(x) for x in mv_economic_terms],
        'cv_economic_terms': [float(x) for x in cv_economic_terms],
        'mv_economic_typical': [float(x) for x in mv_economic_typical],
        'cv_economic_typical': [float(x) for x in cv_economic_typical],
        'mv_economic_penalty': float(mv_economic_penalty),
        'cv_economic_penalty': float(cv_economic_penalty),
        'in_band': bool(in_band),
        'in_band_bonus': 0.0,
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
