"""Training-time disturbance curriculum helpers for controller learning.

Functionality:
- Builds a per-episode disturbance schedule with realistic DV and CV
  perturbations so policies learn disturbance rejection during training.
- Applies active disturbances via the simulator's ``set_disturbance_offset``
  interface for persistence **and** immediate state modification for instant
  step visibility.

Design notes:
- Two disturbance sources are supported:
    * **measured DV** — a step change on a measured disturbance variable.
      The agent can observe the shift and must learn feed-forward rejection.
    * **unmeasured CV** — a step change on the process target underlying a
      controlled variable.  The agent cannot see the cause; it only observes
      the CV deviating from setpoint and must learn feedback rejection.
- Disturbances are instantaneous step changes (one-time shift at the trigger
  step) that persist for the remainder of the episode.  This mirrors real
  plant upsets (valve position change, feed composition shift) and forces
  policies to learn sustained disturbance rejection.
- Magnitudes auto-adapt to the simulation model via identified dynamics,
  DV→CV gains, and channel spans.
- Objective remains purely economic/constraint-driven; this module only changes
  trajectories seen by the agent.
"""

import glob
import json
import math
import os
from typing import Dict, List, Optional, Tuple

import numpy as np


def _bool_env(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() not in {"0", "false", "no", "off"}


def disturbance_curriculum_enabled(default: bool = True) -> bool:
    return _bool_env("AGENT_DISTURBANCE_CURRICULUM", default)


def disturbance_progressive_enabled(default: bool = True) -> bool:
    return _bool_env("AGENT_DISTURBANCE_PROGRESSIVE", default)


def _state_name(sim, idx: int) -> str:
    names = list(getattr(sim, 'state_variables', []))
    i = int(idx)
    if 0 <= i < len(names):
        return str(names[i])
    return f'S{i}'


def _channel_catalog(sim) -> Dict[str, List[Dict]]:
    cv_indices = [int(i) for i in list(getattr(sim, 'cv_indices', []))]
    dv_indices = [int(i) for i in list(getattr(sim, 'dv_indices', []))]
    cv_ranges = list(getattr(sim, 'cv_normalization_ranges', []))
    dv_ranges = list(getattr(sim, 'dv_normalization_ranges', []))

    out = {'cv': [], 'dv': []}
    for pos, idx in enumerate(cv_indices):
        bounds = None
        if 0 <= pos < len(cv_ranges):
            bounds = [float(cv_ranges[pos][0]), float(cv_ranges[pos][1])]
        out['cv'].append({'group': 'cv', 'pos': int(pos), 'index': int(idx), 'name': _state_name(sim, idx), 'bounds': bounds})

    for pos, idx in enumerate(dv_indices):
        bounds = None
        if 0 <= pos < len(dv_ranges):
            bounds = [float(dv_ranges[pos][0]), float(dv_ranges[pos][1])]
        out['dv'].append({'group': 'dv', 'pos': int(pos), 'index': int(idx), 'name': _state_name(sim, idx), 'bounds': bounds})
    return out


def _engineering_to_raw(value: float, ranges: List[List[float]], pos: int, state_is_normalized: bool) -> float:
    v = float(value)
    if not state_is_normalized:
        return v
    if 0 <= int(pos) < len(ranges):
        lo, hi = float(ranges[pos][0]), float(ranges[pos][1])
        if hi <= lo:
            hi = lo + 1.0
        return float(np.clip((v - lo) / (hi - lo), 0.0, 1.0))
    return v


def _state_group_value(state: np.ndarray, sim, group: str, pos: int, idx: int) -> float:
    raw = float(state[int(idx)])
    if not bool(getattr(sim, 'state_is_normalized', False)):
        return raw
    if group == 'cv':
        ranges = list(getattr(sim, 'cv_normalization_ranges', []))
    elif group == 'dv':
        ranges = list(getattr(sim, 'dv_normalization_ranges', []))
    else:
        ranges = []
    if 0 <= int(pos) < len(ranges):
        lo, hi = float(ranges[pos][0]), float(ranges[pos][1])
        if hi <= lo:
            hi = lo + 1.0
        return float(lo + raw * (hi - lo))
    return raw


def _load_identifier_context() -> Dict:
    roots = []
    setup_json = str(os.environ.get('CONTROL_SETUP_JSON', '')).strip()
    if setup_json:
        roots.append(os.path.dirname(os.path.abspath(setup_json)))

    cur = os.getcwd()
    for _ in range(6):
        roots.append(cur)
        nxt = os.path.dirname(cur)
        if nxt == cur:
            break
        cur = nxt

    explicit_dyn = str(os.environ.get('AGENT_DYNAMICS_JSON', '')).strip()
    explicit_lb = str(os.environ.get('AGENT_LOOKBACK_JSON', '')).strip()

    def _latest_match(patterns):
        candidates = []
        for r in roots:
            for pat in patterns:
                candidates.extend(glob.glob(os.path.join(r, pat)))
        candidates = [p for p in candidates if os.path.isfile(p)]
        if not candidates:
            return ''
        candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        return candidates[0]

    dyn_path = explicit_dyn if (explicit_dyn and os.path.isfile(explicit_dyn)) else _latest_match([
        'dynamics_identification_*.json',
        '**/dynamics_identification_*.json',
    ])
    lb_path = explicit_lb if (explicit_lb and os.path.isfile(explicit_lb)) else _latest_match([
        'lookback_identification_*.json',
        'lookback_identification_practical.json',
        '**/lookback_identification_*.json',
        '**/lookback_identification_practical.json',
    ])

    dyn = {}
    lb = {}
    if dyn_path:
        try:
            with open(dyn_path, 'r', encoding='utf-8') as f:
                dyn = json.load(f)
        except Exception:
            dyn = {}
    if lb_path:
        try:
            with open(lb_path, 'r', encoding='utf-8') as f:
                lb = json.load(f)
        except Exception:
            lb = {}

    dv_gain = {}
    mv_gain = {}
    rows = dyn.get('per_pair_estimates', []) if isinstance(dyn, dict) else []
    for r in rows if isinstance(rows, list) else []:
        try:
            if not bool(r.get('valid', False)):
                continue
            delta = abs(float(r.get('delta', 0.0)))
            amp = abs(float(r.get('amplitude', 0.0)))
            if delta <= 1e-8:
                continue
            g = amp / delta
            input_type = str(r.get('input_type', 'mv')).lower()
            if input_type == 'dv':
                bucket = dv_gain
                prefix = 'dv'
            else:
                bucket = mv_gain
                prefix = 'mv'
            keys = []
            name = str(r.get('input', r.get(prefix, ''))).strip()
            if name:
                keys.append(name)
            if 'input_index' in r:
                keys.append(f"{prefix}_{int(r.get('input_index', 0))}")
            for k in keys:
                bucket.setdefault(k, []).append(g)
        except Exception:
            continue

    dv_gain = {
        str(k): float(np.percentile(np.asarray(v, dtype='float64'), 70))
        for k, v in dv_gain.items() if v
    }
    mv_gain = {
        str(k): float(np.percentile(np.asarray(v, dtype='float64'), 70))
        for k, v in mv_gain.items() if v
    }

    return {
        'lookback': lb if isinstance(lb, dict) else {},
        'dynamics': dyn if isinstance(dyn, dict) else {},
        'dv_gain_to_cv': dv_gain,
        'mv_gain_to_cv': mv_gain,
    }


def load_mv_gain_to_cv() -> Dict[str, float]:
    """Public helper: return the MV->CV steady-state |gain| map from the
    latest identifier JSON, keyed by both the MV state-name and 'mv_{i}'.

    Used by training/validation to size dynamic CV-bound step changes so
    the demanded CV excursion is reachable within a fraction of MV travel.
    Returns an empty dict when no identifier context is available.
    """
    try:
        ctx = _load_identifier_context()
        gains = ctx.get('mv_gain_to_cv') if isinstance(ctx, dict) else None
        return dict(gains) if isinstance(gains, dict) else {}
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Shared MV-authority budget + saturation + intensity helpers
# ---------------------------------------------------------------------------
#
# These utilities are simulator-agnostic: they read only public sim metadata
# (mv_indices, mv_normalization_ranges, state_variables) plus the identifier
# context produced by ``_load_identifier_context``.  They are used by both
# the training-time disturbance scheduler (this module) and the validation-
# time scheduler in ``evaluation/validate_latent.py``, so any simulator that
# exposes the standard mv/cv/dv index + normalization-range attributes —
# including ONNX-driven surrogates and the analytic test_sim — works without
# code changes.


def compute_mv_authority_to_cv(sim, identifier_ctx: Optional[Dict] = None) -> float:
    """Estimate total MV->CV authority in CV engineering units.

    ``mv_gain_to_cv`` from the identifier is keyed by MV (collapsed across
    CVs) so we cannot break authority down per-CV without a per-pair
    rerun.  Returning a single conservative scalar (sum over all MVs of
    ``|gain_i| * mv_span_i``) is sufficient for the disturbance budgeter:
    we treat the total MV travel as the authority available to reject a
    composite CV excursion.

    Returns 0.0 when the identifier produced no usable MV gains, in which
    case the budgeter falls back to magnitude-only clipping.
    """
    if identifier_ctx is None:
        try:
            identifier_ctx = _load_identifier_context()
        except Exception:
            identifier_ctx = {}
    mv_gain_map = (identifier_ctx or {}).get('mv_gain_to_cv', {}) or {}
    mv_indices = [int(i) for i in list(getattr(sim, 'mv_indices', []) or [])]
    mv_ranges = list(getattr(sim, 'mv_normalization_ranges', []) or [])
    state_vars = list(getattr(sim, 'state_variables', []) or [])

    total = 0.0
    for mv_pos, mv_idx in enumerate(mv_indices):
        if mv_pos >= len(mv_ranges):
            continue
        try:
            mv_lo, mv_hi = float(mv_ranges[mv_pos][0]), float(mv_ranges[mv_pos][1])
        except (TypeError, ValueError, IndexError):
            continue
        mv_span = max(0.0, mv_hi - mv_lo)
        mv_state_name = ''
        if 0 <= mv_idx < len(state_vars):
            try:
                mv_state_name = str(state_vars[mv_idx])
            except Exception:
                mv_state_name = ''
        gain = 0.0
        try:
            gain = float(
                mv_gain_map.get(mv_state_name, 0.0)
                or mv_gain_map.get(f'mv_{mv_pos}', 0.0)
                or 0.0
            )
        except (TypeError, ValueError):
            gain = 0.0
        total += abs(gain) * mv_span
    return float(total)


def get_authority_target_frac(default: float = 0.65) -> float:
    """Fraction of total MV->CV authority the disturbance budget may consume.

    Default 0.65 leaves 35% headroom for OU drift, measurement noise, and
    transient overshoot.  Override via ``AGENT_DISTURBANCE_AUTHORITY_FRAC``.
    Set to 0 to disable the budget entirely (legacy behaviour).
    """
    try:
        val = float(os.environ.get('AGENT_DISTURBANCE_AUTHORITY_FRAC', default))
    except (TypeError, ValueError):
        val = default
    return float(np.clip(val, 0.0, 1.5))


def clamp_event_to_authority_budget(
    proposed_delta: float,
    cv_impact_per_unit: float,
    cumulative_cv_impact: float,
    mv_authority_cv: float,
    target_frac: float,
) -> Tuple[float, float]:
    """Clip a proposed event ``delta`` so cumulative CV impact stays under budget.

    ``cv_impact_per_unit`` is the conversion from event ``delta`` to CV
    engineering units (1.0 for unmeasured CV events, ``dv_gain`` for
    measured DV events).  The function returns ``(scaled_delta, achieved_cv_impact_signed)``.
    When ``mv_authority_cv`` is non-positive (no identifier gain) or
    ``target_frac`` is 0, the proposed delta is returned unchanged.
    """
    if mv_authority_cv <= 1e-9 or target_frac <= 0.0:
        return float(proposed_delta), float(proposed_delta) * float(cv_impact_per_unit)
    budget = float(mv_authority_cv) * float(target_frac)
    proposed_cv = float(proposed_delta) * float(cv_impact_per_unit)
    projected = float(cumulative_cv_impact) + proposed_cv
    # Allow accumulated impact up to +- budget; cap if the new event would
    # blow the budget on either side.
    if abs(projected) <= budget:
        return float(proposed_delta), proposed_cv
    sign = 1.0 if proposed_cv >= 0.0 else -1.0
    allowed_cv = sign * budget - float(cumulative_cv_impact)
    # If the cumulative is already over-budget on one side, clamp to a
    # small fraction of the original event in the *opposite* direction so
    # the channel can begin returning toward the budget envelope rather
    # than zeroing out (which previously starved late multi-event
    # episodes on low-gain plants of any meaningful state movement).
    # Cap at ``AGENT_DISTURBANCE_RECOVERY_FRAC`` (default 0.20) of the
    # proposed magnitude.
    try:
        recovery_frac = float(os.environ.get(
            'AGENT_DISTURBANCE_RECOVERY_FRAC', '0.20'))
    except (TypeError, ValueError):
        recovery_frac = 0.20
    recovery_frac = float(np.clip(recovery_frac, 0.0, 1.0))
    if (allowed_cv == 0.0) or (np.sign(allowed_cv) != sign):
        if recovery_frac <= 0.0:
            return 0.0, 0.0
        # Reverse direction with a small magnitude (pull cumulative back
        # toward zero).
        recover_delta = -sign * recovery_frac * abs(float(proposed_delta))
        recover_cv = recover_delta * float(cv_impact_per_unit)
        return float(recover_delta), float(recover_cv)
    scale = abs(allowed_cv) / max(1e-12, abs(proposed_cv))
    scale = float(np.clip(scale, 0.0, 1.0))
    return float(proposed_delta) * scale, proposed_cv * scale


class MVSaturationMonitor:
    """Track recent MV saturation and gate disturbance event firing.

    Sim-agnostic: reads only ``state[mv_indices[i]]`` and the simulator's
    ``mv_normalization_ranges``.  Maintains an exponential moving average of
    saturated-step indicators per MV and collapses to a worst-case scalar.

    Intended use::

        mon = MVSaturationMonitor(sim)
        ...
        mon.update(state)               # call after every sim.step
        if mon.should_suppress():
            # skip / shrink the next scheduled disturbance event
            ...

    The default suppression threshold is 0.20 (i.e. >20% of recent steps
    saturated).  Override via ``AGENT_DISTURBANCE_SUPPRESS_SAT_FRAC``.  The
    EMA window length defaults to the identified lookback (or 64 steps);
    override via ``AGENT_DISTURBANCE_SUPPRESS_WINDOW``.
    """

    def __init__(
        self,
        sim,
        edge_frac: float = 0.02,
        threshold: Optional[float] = None,
        window: Optional[int] = None,
    ) -> None:
        self.mv_indices = [int(i) for i in list(getattr(sim, 'mv_indices', []) or [])]
        self.mv_ranges = list(getattr(sim, 'mv_normalization_ranges', []) or [])
        self.is_normalized = bool(getattr(sim, 'state_is_normalized', False))
        self.edge_frac = float(np.clip(edge_frac, 1e-6, 0.49))
        try:
            thr_env = os.environ.get('AGENT_DISTURBANCE_SUPPRESS_SAT_FRAC')
            self.threshold = (
                float(thr_env) if thr_env is not None and threshold is None
                else float(threshold if threshold is not None else 0.20)
            )
        except (TypeError, ValueError):
            self.threshold = 0.20
        try:
            win_env = os.environ.get('AGENT_DISTURBANCE_SUPPRESS_WINDOW')
            win = int(win_env) if win_env is not None and window is None else int(
                window if window is not None else 64
            )
        except (TypeError, ValueError):
            win = 64
        self.window = max(8, int(win))
        self._ema = np.zeros(max(1, len(self.mv_indices)), dtype='float64')
        self._alpha = 1.0 / float(self.window)
        # Persistent saturation count (independent of EMA) so we can also
        # track per-episode saturation fraction for the intensity controller.
        self._sat_count = np.zeros(max(1, len(self.mv_indices)), dtype='int64')
        self._step_count = 0

    def reset_episode(self) -> None:
        self._ema[:] = 0.0
        self._sat_count[:] = 0
        self._step_count = 0

    def _instant_sat(self, state: np.ndarray) -> np.ndarray:
        out = np.zeros(max(1, len(self.mv_indices)), dtype='float64')
        for i, mv_idx in enumerate(self.mv_indices):
            if mv_idx < 0 or mv_idx >= len(state):
                continue
            if i >= len(self.mv_ranges):
                continue
            try:
                lo, hi = float(self.mv_ranges[i][0]), float(self.mv_ranges[i][1])
            except (TypeError, ValueError, IndexError):
                continue
            span = max(1e-6, hi - lo)
            if self.is_normalized:
                v = float(np.clip(state[mv_idx], 0.0, 1.0)) * span + lo
            else:
                v = float(state[mv_idx])
            margin = self.edge_frac * span
            if (v <= lo + margin) or (v >= hi - margin):
                out[i] = 1.0
        return out

    def update(self, state: np.ndarray) -> None:
        if not self.mv_indices:
            return
        inst = self._instant_sat(state)
        self._ema = (1.0 - self._alpha) * self._ema + self._alpha * inst
        self._sat_count += inst.astype('int64')
        self._step_count += 1

    def saturation_fraction(self) -> float:
        if self._step_count <= 0 or not self.mv_indices:
            return 0.0
        return float(np.max(self._sat_count) / float(self._step_count))

    def recent_saturation_fraction(self) -> float:
        if not self.mv_indices:
            return 0.0
        return float(np.max(self._ema))

    def should_suppress(self) -> bool:
        return self.recent_saturation_fraction() > self.threshold


class DisturbanceIntensityController:
    """Replaces the per-episode-index intensity lambda with an adaptive
    schedule that backs off when MV saturates and recovers when it doesn't.

    The base curriculum is identical to the previous lambda
    (warmup -> ramp from ``intensity_min`` to ``intensity_max`` over the
    estimated episode budget), but a multiplicative ``adaptive`` factor in
    ``[adaptive_min, 1.0]`` modulates the result based on per-episode MV
    saturation feedback.  Reduce by ``decay`` when episode saturation
    exceeds ``target_sat`` and recover by ``recovery`` (geometric, capped at
    1.0) when it stays below.
    """

    def __init__(
        self,
        intensity_min: float,
        intensity_max: float,
        warmup_episodes: int,
        total_episodes: int,
        progressive: bool = True,
    ) -> None:
        self.intensity_min = float(intensity_min)
        self.intensity_max = float(intensity_max)
        self.warmup_episodes = max(0, int(warmup_episodes))
        self.total_episodes = max(1, int(total_episodes))
        self.progressive = bool(progressive)
        try:
            self.target_sat = float(os.environ.get('AGENT_TARGET_SATURATION_FRAC', '0.20'))
        except (TypeError, ValueError):
            self.target_sat = 0.20
        try:
            self.decay = float(os.environ.get('AGENT_DISTURBANCE_INTENSITY_DECAY', '0.85'))
        except (TypeError, ValueError):
            self.decay = 0.85
        try:
            self.recovery = float(os.environ.get('AGENT_DISTURBANCE_INTENSITY_RECOVERY', '1.05'))
        except (TypeError, ValueError):
            self.recovery = 1.05
        try:
            self.adaptive_min = float(os.environ.get('AGENT_DISTURBANCE_INTENSITY_FLOOR', '0.30'))
        except (TypeError, ValueError):
            self.adaptive_min = 0.30
        self.target_sat = float(np.clip(self.target_sat, 0.0, 1.0))
        self.decay = float(np.clip(self.decay, 0.5, 1.0))
        self.recovery = float(np.clip(self.recovery, 1.0, 1.5))
        self.adaptive_min = float(np.clip(self.adaptive_min, 0.05, 1.0))
        self._adaptive = 1.0
        self._low_sat_streak = 0

    def get_intensity(self, episode_index: int) -> float:
        if not self.progressive:
            return self.intensity_max * self._adaptive
        ep = max(0, int(episode_index))
        if ep < self.warmup_episodes:
            return 0.0
        remaining = max(1.0, float(self.total_episodes - self.warmup_episodes - 1))
        frac = float(np.clip(float(ep - self.warmup_episodes) / remaining, 0.0, 1.0))
        base = self.intensity_min + (self.intensity_max - self.intensity_min) * frac
        return float(np.clip(base * self._adaptive, 0.0, self.intensity_max))

    def update_from_episode(self, sat_frac: float) -> None:
        """Adjust the adaptive multiplier from last episode's MV saturation."""
        sat = float(np.clip(sat_frac, 0.0, 1.0))
        if sat > self.target_sat:
            self._adaptive = float(np.clip(self._adaptive * self.decay, self.adaptive_min, 1.0))
            self._low_sat_streak = 0
        else:
            # Mild hysteresis: only recover after a few consecutive low-sat episodes
            self._low_sat_streak += 1
            if self._low_sat_streak >= 3:
                self._adaptive = float(np.clip(self._adaptive * self.recovery, self.adaptive_min, 1.0))
                self._low_sat_streak = 0

    @property
    def adaptive_factor(self) -> float:
        return float(self._adaptive)


def compute_episode_mv_saturation(
    mv_value_series: List[float],
    mv_lo: float,
    mv_hi: float,
    edge_frac: float = 0.02,
) -> float:
    """Compute MV saturation fraction over a single episode trace.

    Generic helper for training scripts: pass the raw MV-engineering-unit
    series captured during the episode and the bounds from
    ``mv_normalization_ranges[mv_pos]``.  Returns the fraction of steps the
    MV spent within ``edge_frac`` of either bound.
    """
    if not mv_value_series:
        return 0.0
    span = max(1e-6, float(mv_hi) - float(mv_lo))
    margin = float(edge_frac) * span
    arr = np.asarray(mv_value_series, dtype='float64')
    sat = np.logical_or(arr <= float(mv_lo) + margin, arr >= float(mv_hi) - margin)
    return float(np.mean(sat.astype('float32')))


# ---------------------------------------------------------------------------
# Mixed-mode episode initialization
# ---------------------------------------------------------------------------

def choose_episode_init_mode(
    rng: np.random.Generator,
    completed_episodes: int = 0,
    warmup_episodes: int = 5,
) -> str:
    """Select episode initialization mode.

    Returns one of:
      'center'  — (50%) normal reset near center operating point
      'explore' — (40%) wide random operating point
      'extreme' — (10%) near edge of operating envelope
    Early warmup episodes always use 'center'.
    """
    if completed_episodes < warmup_episodes:
        return 'center'
    # R8: shift mix toward explore/extreme so the policy rarely trains from
    # a comfortable center-of-envelope start. Constraint-only objectives with
    # strong MV economic pull (e.g. test_sim mv_economic=5, cv_econ=0) cause
    # the policy to park MV at its economic optimum; we want disturbances
    # hitting while MV is at that optimum, not at the midpoint.
    r = rng.uniform()
    if r < 0.30:
        return 'center'
    elif r < 0.75:
        return 'explore'
    else:
        return 'extreme'


def apply_episode_init_offsets(
    state: np.ndarray,
    sim,
    mode: str,
    rng: np.random.Generator,
) -> Dict:
    """Shift the post-reset state to a random operating point.

    Uses only configuration-derived metadata (indices, normalization ranges
    from control_setup.json attached by the sim factory) and system
    identification results.  Does **not** access simulator internals
    (u_actual, u_history, episode_array, etc.) so it works generically
    with any simulator backend including neural-network surrogates.

    For 'explore': DV offset across 15-85% of identified range, MV shifted
    to a moderate random offset.
    For 'extreme': DV near 5-20% from range edges, MV near opposite edges.
    For 'center': no changes.

    Returns a dict:
      - 'state':     the (in-place) modified state array
      - 'mv_values': np.ndarray of MV targets in engineering units
                     (caller should use for prev_control)
      - 'mode':      the mode that was applied
    """
    # Default MV values at midpoint of ranges (used when mode == 'center')
    mv_indices = [int(i) for i in list(getattr(sim, 'mv_indices', []))]
    mv_ranges = list(getattr(sim, 'mv_normalization_ranges', []))
    default_mv = np.array([
        0.5 * (float(mv_ranges[p][0]) + float(mv_ranges[p][1]))
        if p < len(mv_ranges) else 50.0
        for p in range(len(mv_indices))
    ], dtype='float32')

    if mode == 'center':
        return {'state': state, 'mv_values': default_mv, 'mode': mode}

    dv_indices = [int(i) for i in list(getattr(sim, 'dv_indices', []))]
    dv_ranges = list(getattr(sim, 'dv_normalization_ranges', []))
    is_normalized = bool(getattr(sim, 'state_is_normalized', False))

    # --- DV offsets ---------------------------------------------------------
    for dv_pos, dv_idx in enumerate(dv_indices):
        if dv_pos >= len(dv_ranges):
            continue
        dv_lo, dv_hi = float(dv_ranges[dv_pos][0]), float(dv_ranges[dv_pos][1])
        dv_center = (dv_lo + dv_hi) * 0.5
        dv_span = max(1e-6, dv_hi - dv_lo)

        if mode == 'explore':
            dv_target = float(rng.uniform(dv_lo + 0.15 * dv_span, dv_hi - 0.15 * dv_span))
        else:  # extreme
            if rng.uniform() < 0.5:
                dv_target = float(rng.uniform(dv_lo + 0.05 * dv_span, dv_lo + 0.20 * dv_span))
            else:
                dv_target = float(rng.uniform(dv_hi - 0.20 * dv_span, dv_hi - 0.05 * dv_span))

        dv_offset = dv_target - dv_center

        # Persistent DV offset via generic mixin API (if available).
        if hasattr(sim, 'set_disturbance_offset'):
            sim.set_disturbance_offset('dv', dv_pos, float(dv_offset))

        # Update observable state array.
        if is_normalized:
            state[dv_idx] = float(np.clip((dv_target - dv_lo) / dv_span, 0.0, 1.0))
        else:
            state[dv_idx] = float(np.clip(dv_target, dv_lo, dv_hi))

    # --- MV offsets ---------------------------------------------------------
    mv_values = np.copy(default_mv)
    for mv_pos, mv_idx in enumerate(mv_indices):
        if mv_pos >= len(mv_ranges):
            continue
        mv_lo, mv_hi = float(mv_ranges[mv_pos][0]), float(mv_ranges[mv_pos][1])
        mv_center = (mv_lo + mv_hi) * 0.5
        mv_span = max(1e-6, mv_hi - mv_lo)

        if mode == 'explore':
            mv_frac = float(rng.uniform(0.05, 0.35))
            mv_target = mv_center + float(rng.uniform(-mv_frac, mv_frac)) * mv_span
        else:  # extreme
            if rng.uniform() < 0.5:
                mv_target = float(rng.uniform(mv_lo + 0.05 * mv_span, mv_lo + 0.20 * mv_span))
            else:
                mv_target = float(rng.uniform(mv_hi - 0.20 * mv_span, mv_hi - 0.05 * mv_span))

        mv_target = float(np.clip(mv_target, mv_lo, mv_hi))
        mv_values[mv_pos] = mv_target

        if is_normalized:
            state[mv_idx] = float(np.clip((mv_target - mv_lo) / mv_span, 0.0, 1.0))
        else:
            state[mv_idx] = float(np.clip(mv_target, mv_lo, mv_hi))

    # --- CV offsets (init-only) --------------------------------------------
    # Why: the agent previously only saw CVs at boundaries via dynamics
    # (DV impact + MV pressure).  In ``extreme`` mode that produces only
    # one boundary side per episode and never starts at a boundary.  Add
    # a deliberate persistent CV offset on init so the policy trains
    # episodes that begin near a CV bound and must recover.
    #
    # Anti-saturation safety: cap each CV offset by
    # ``init_cv_offset_authority_frac`` (default 0.30) of the per-channel
    # MV->CV recoverable authority, so the agent always has \u2265 70 % of
    # MV travel free to defend against the offset.  Falls back to a span
    # cap when no MV gain is available, and is opt-out via env-var
    # ``AGENT_INIT_CV_OFFSET_FRAC=0``.
    cv_indices = [int(i) for i in list(getattr(sim, 'cv_indices', []) or [])]
    cv_ranges = list(getattr(sim, 'cv_normalization_ranges', []) or [])
    try:
        init_cv_auth_frac = float(os.environ.get(
            'AGENT_INIT_CV_OFFSET_FRAC', '0.30'))
    except (TypeError, ValueError):
        init_cv_auth_frac = 0.30
    init_cv_auth_frac = float(np.clip(init_cv_auth_frac, 0.0, 1.0))
    if cv_indices and cv_ranges and init_cv_auth_frac > 0.0:
        # Per-CV recoverable authority in CV engineering units:
        # sum_m |k_{m,c}| * 0.35 * mv_span_m  (35 % MV travel headroom).
        try:
            id_ctx = _load_identifier_context()
        except Exception:
            id_ctx = {}
        mv_gain_map = (id_ctx or {}).get('mv_gain_to_cv', {}) or {}
        state_vars = list(getattr(sim, 'state_variables', []) or [])
        # mv_gain_map is collapsed across CVs in the identifier context,
        # so we can only produce a single shared authority estimate.
        # Divide evenly across CVs as a conservative per-CV cap.
        total_auth = 0.0
        for mv_pos, mv_idx in enumerate(mv_indices):
            if mv_pos >= len(mv_ranges):
                continue
            mv_lo, mv_hi = float(mv_ranges[mv_pos][0]), float(mv_ranges[mv_pos][1])
            mv_span = max(0.0, mv_hi - mv_lo)
            mv_state_name = ''
            if 0 <= mv_idx < len(state_vars):
                try:
                    mv_state_name = str(state_vars[mv_idx])
                except Exception:
                    mv_state_name = ''
            try:
                gain = float(
                    mv_gain_map.get(mv_state_name, 0.0)
                    or mv_gain_map.get(f'mv_{mv_pos}', 0.0)
                    or 0.0
                )
            except (TypeError, ValueError):
                gain = 0.0
            total_auth += abs(gain) * 0.35 * mv_span
        per_cv_auth = total_auth / max(1, len(cv_indices))
        for cv_pos, cv_idx in enumerate(cv_indices):
            if cv_pos >= len(cv_ranges):
                continue
            cv_lo, cv_hi = float(cv_ranges[cv_pos][0]), float(cv_ranges[cv_pos][1])
            cv_center = (cv_lo + cv_hi) * 0.5
            cv_span = max(1e-6, cv_hi - cv_lo)
            # Magnitude budget: prefer the gain-based authority cap when
            # available (keeps MV ~70 % free), else fall back to a small
            # span cap so we never demand an excursion the actuator
            # cannot recover from.
            if per_cv_auth > 1e-9:
                cap = init_cv_auth_frac * per_cv_auth
            else:
                cap = init_cv_auth_frac * 0.30 * cv_span  # ~9 % span at default
            if mode == 'explore':
                cv_offset = float(rng.uniform(-1.0, 1.0)) * 0.6 * cap
            else:  # extreme
                sign = -1.0 if rng.uniform() < 0.5 else 1.0
                cv_offset = sign * float(rng.uniform(0.6, 1.0)) * cap
            # Hard clip so the resulting CV stays inside the normalisation
            # range with a small inward margin.
            cv_target_eu = cv_center + cv_offset
            inward = 0.05 * cv_span
            cv_target_eu = float(np.clip(cv_target_eu, cv_lo + inward, cv_hi - inward))
            cv_offset_clipped = cv_target_eu - cv_center
            if hasattr(sim, 'set_disturbance_offset'):
                # Persistent offset; sims that honor CV offsets observe
                # a state change immediately, others see it only via the
                # objective-side bounds delta.  Either way the agent
                # learns a boundary-near initial condition.
                sim.set_disturbance_offset('cv', cv_pos, float(cv_offset_clipped))
            # Mirror into the observable state so the seed buffer / WM
            # see the boundary-near initial CV right away.
            if is_normalized:
                state[cv_idx] = float(np.clip((cv_target_eu - cv_lo) / cv_span, 0.0, 1.0))
            else:
                state[cv_idx] = float(np.clip(cv_target_eu, cv_lo, cv_hi))

    return {'state': state, 'mv_values': mv_values, 'mode': mode}


def build_training_disturbance_schedule(
    episode_length: int,
    rng: np.random.Generator,
    max_events: int = 5,
    intensity: float = 1.0,
    sim=None,
) -> List[Dict]:
    """Build the per-episode operator-event schedule.

    Only measured-DV events are emitted.  Unmeasured-CV step events
    were removed in favour of the hidden OU disturbance process (see
    ``utils/hidden_disturbance.py``) because step-shaped CV bumps are
    Dirac spikes with no observable cue and capped ``sf_loss_flow``
    around 1.15 in P33.
    """
    ep_len = max(220, int(episode_length))

    channels = {'dv': [], 'cv': []}
    if sim is not None:
        channels = _channel_catalog(sim)
    dv_targets = list(channels.get('dv', []))
    # CV channels are still catalogued for the cv_span_ref reference
    # (used to size DV-driven CV impact); we never emit CV step events.
    cv_targets = list(channels.get('cv', []))
    targets = list(dv_targets)
    if not dv_targets:
        # Generic fallback if simulator metadata is unavailable.
        dv_targets = [{'group': 'dv', 'pos': 0, 'name': 'dv_0', 'bounds': [0.0, 100.0]}]
        targets = list(dv_targets)
        if not cv_targets:
            cv_targets = [{'group': 'cv', 'pos': 0, 'name': 'cv_0', 'bounds': [0.0, 100.0]}]

    id_ctx = _load_identifier_context()
    lookback = int((id_ctx.get('lookback', {}) or {}).get('identified_lookback', 0) or 0)
    tau_dom = float((id_ctx.get('dynamics', {}) or {}).get('tau_dominant_identified', 0.0) or 0.0)
    dead_time = float((id_ctx.get('dynamics', {}) or {}).get('dead_time_identified', 0.0) or 0.0)
    dyn_horizon = max(1.0, dead_time + tau_dom)
    dv_gain = id_ctx.get('dv_gain_to_cv', {}) if isinstance(id_ctx, dict) else {}

    intensity = float(max(0.05, min(1.6, intensity)))
    settle_steps_env = int(os.environ.get("AGENT_DISTURBANCE_SETTLE_STEPS", "0") or 0)
    if settle_steps_env > 0:
        settle = max(32, int(settle_steps_env))
    else:
        settle = int(max(32.0, round(max(0.55 * float(lookback), 1.15 * dyn_horizon))))
    settle = int(max(24, min(settle, int(0.30 * ep_len))))

    # Per-episode event count is randomised in [0, max_events] so the
    # agent sees both dense and sparse episodes — plus a configurable
    # fraction of fully-quiet episodes where no operator events fire and
    # the policy must hold its current operating point against only
    # OU / measurement noise.  Quiet fraction is controlled by
    # ``AGENT_DISTURBANCE_QUIET_FRAC`` (default 0.12 = ~1 in 8
    # episodes).  Dense episodes train constraint reaction under
    # overlapping transients; sparse episodes expose the reach-steady-
    # state regime; quiet episodes expose the agent to the drift/hold
    # task so the learned policy does not rely on disturbance arrival as
    # an exploration trigger.  Operator limit/target changes from the
    # RuntimeSetpointManager remain active during quiet episodes so the
    # agent still sees limit-tracking events without a concurrent
    # disturbance.
    try:
        quiet_frac = float(os.environ.get('AGENT_DISTURBANCE_QUIET_FRAC', '0.12'))
    except Exception:
        quiet_frac = 0.12
    quiet_frac = float(np.clip(quiet_frac, 0.0, 0.5))
    if rng.uniform() < quiet_frac:
        return []  # quiet episode — no disturbance events

    n_cap = max(1, int(ep_len // max(22, int(0.5 * settle))))
    max_events_eff = int(np.clip(max_events, 1, min(6, n_cap)))
    n_events = int(rng.integers(1, max_events_eff + 1))

    # Per-event spacing mixes three regimes so the agent sees both
    # transient and steady-state behaviour:
    #   overlap  ~0.6 - 1.0 x (tau+dead)  — events hit each other mid-transient
    #   settled  ~3.0 - 4.0 x (tau+dead)  — plant reaches ~95% steady state
    #   long     ~6.0 - 9.0 x (tau+dead)  — plant fully relaxed, quiet between
    # The "long" regime is new: it guarantees that within episodes that do
    # have disturbances, some fraction of inter-event windows expose truly
    # steady-state behaviour.  Without it the previous mix never dwelt
    # beyond 4 x dyn_horizon which is <99% of step response on slow plants.
    # Falls back to ``settle`` when dynamics are unknown.
    overlap_lo = max(16, int(0.60 * dyn_horizon))
    overlap_hi = max(overlap_lo + 1, int(1.00 * dyn_horizon))
    settled_lo = max(overlap_hi + 1, int(3.0 * dyn_horizon))
    settled_hi = max(settled_lo + 1, int(4.0 * dyn_horizon))
    long_lo = max(settled_hi + 1, int(6.0 * dyn_horizon))
    long_hi = max(long_lo + 1, int(9.0 * dyn_horizon))

    def _sample_recovery() -> int:
        if dyn_horizon <= 1.0:
            return max(26, int(0.95 * settle))
        r = float(rng.uniform())
        # 40% overlap, 40% settled, 20% long-recovery (quiet between events).
        if r < 0.40:
            return int(rng.integers(overlap_lo, overlap_hi + 1))
        elif r < 0.80:
            return int(rng.integers(settled_lo, settled_hi + 1))
        return int(rng.integers(long_lo, long_hi + 1))

    jitter_start = max(10, int(0.08 * ep_len))

    anchors = np.sort(rng.uniform(0.10, 0.90, size=max(2, n_events)))

    cv_widths = []
    for t in cv_targets:
        b = t.get('bounds')
        if isinstance(b, list) and len(b) >= 2:
            cv_widths.append(max(1e-6, float(b[1]) - float(b[0])))
    cv_span_ref = float(np.median(np.asarray(cv_widths, dtype='float64'))) if cv_widths else 10.0

    # Track cumulative offset per channel key to alternate direction and
    # prevent monotonic drift that saturates MVs.
    _cumulative_offset: Dict[str, float] = {}
    # Authority budget: cap cumulative CV-side impact across ALL events at
    # ``authority_target_frac`` of total MV->CV authority so the agent has
    # at least ~35% MV travel headroom over OU drift + transients.  Falls
    # back to magnitude-only clipping when the identifier produced no
    # usable MV gain.
    mv_authority_cv = compute_mv_authority_to_cv(sim, id_ctx) if sim is not None else 0.0
    authority_frac = get_authority_target_frac()
    _cumulative_cv_impact = 0.0  # signed sum across events (CV engineering units)
    schedule: List[Dict] = []
    last_start = 0
    for i in range(n_events):
        if dv_targets:
            target = dv_targets[int(rng.integers(0, len(dv_targets)))]
        else:
            target = targets[i % len(targets)]
        source = 'measured_dv'

        start = int(float(anchors[min(i, len(anchors) - 1)]) * ep_len) + int(rng.integers(-jitter_start, jitter_start + 1))
        start = max(last_start + _sample_recovery(), start)
        if start >= ep_len - 30:
            break

        b = target.get('bounds')
        span = 100.0
        if isinstance(b, list) and len(b) >= 2:
            span = max(1e-6, float(b[1]) - float(b[0]))

        # Curriculum-coupled violation probability: ramps from 0.25 (gentle
        # start) to 0.60 (matches the hardest BO validation profile,
        # ``holdout_b``) so training-time distribution tracks what the
        # policy will be scored on.  Replaces previous hard-coded 0.35.
        i01 = float(np.clip(intensity, 0.0, 1.0))
        # R5: raise violation-probability ceiling. Training should over-expose
        # relative to holdout_b (50% violation) so the policy is never
        # surprised by the constraint location at eval time.
        p_violation = float(np.clip(0.30 + 0.40 * i01, 0.30, 0.70))
        is_violation = bool(rng.uniform() < p_violation)
        # R1: remove intensity double-scaling. `frac` interpolation below already
        # uses `i01` to ramp magnitudes from safe to design peak, so applying
        # `intensity` a second time via `scale` shrank the realised magnitudes
        # ~1/intensity (roughly 10x too small early, 24%+ short even late).
        scale = float(rng.uniform(0.90, 1.10))
        # Direction-aware sign: if this channel has drifted significantly,
        # bias the next step in the opposite direction to prevent MV saturation.
        ch_key = f"{target.get('group', 'dv')}_{target.get('pos', 0)}"
        cum = _cumulative_offset.get(ch_key, 0.0)
        b_ref = target.get('bounds')
        ch_span = 100.0
        if isinstance(b_ref, list) and len(b_ref) >= 2:
            ch_span = max(1e-6, float(b_ref[1]) - float(b_ref[0]))
        drift_frac = abs(cum) / ch_span if ch_span > 1e-6 else 0.0
        if drift_frac > 0.15 and abs(cum) > 1e-9:
            sign = -1.0 if cum > 0 else 1.0
            if rng.uniform() < 0.20:
                sign = -sign
        else:
            sign = -1.0 if rng.uniform() < 0.5 else 1.0

        # Magnitude fractions interpolate linearly with curriculum intensity
        # so at ``intensity == 1`` the upper ends coincide with the ranges
        # used by validation's ``holdout_a`` profile.  This eliminates the
        # training<->validation distribution gap that let policies pass the
        # easy gate while failing harder profiles.  Each pair below is
        # ``(lo, hi)`` at intensity=1; at low intensity the ranges narrow
        # toward the historical training-safe values.
        if is_violation:
            lo = 0.08 + 0.10 * i01   # 0.08 -> 0.18
            hi = 0.16 + 0.22 * i01   # 0.16 -> 0.38
        else:
            lo = 0.03
            hi = 0.08 + 0.08 * i01   # 0.08 -> 0.16
        frac = float(rng.uniform(lo, hi))
        total_magnitude = float(sign * frac * span * scale)

        gain = float(
            dv_gain.get(
                str(target.get('name', '')),
                dv_gain.get(f"dv_{int(target.get('pos', 0))}", 0.0),
            ) or 0.0
        )
        if gain > 1e-8:
            if is_violation:
                impact_lo = 0.10 + 0.08 * i01   # 0.10 -> 0.18
                impact_hi = 0.22 + 0.33 * i01   # 0.22 -> 0.55
            else:
                impact_lo = 0.04
                impact_hi = 0.10 + 0.08 * i01   # 0.10 -> 0.18
            desired_cv_impact = (
                float(rng.uniform(impact_lo, impact_hi))
                * cv_span_ref
                * max(0.55, min(1.20, scale))
            )
            needed = float(desired_cv_impact) / float(gain)
            total_magnitude = float(sign * max(abs(total_magnitude), abs(needed)))

        # Clip expands with intensity to match the widest fraction
        # the curriculum can draw (0.22 -> 0.40 of span).
        dv_clip = (0.22 + 0.18 * i01) * span
        total_magnitude = float(np.clip(total_magnitude, -dv_clip, dv_clip))

        # --- Authority-budget clip --------------------------------------
        # Convert the proposed event delta to its CV-side impact and clamp
        # so the running |sum| of CV impacts stays under
        # ``authority_frac * mv_authority_cv``.  This guarantees the policy
        # always has > (1 - frac) of MV travel free to reject the
        # disturbance plus OU drift.
        gain_for_budget = float(
            dv_gain.get(
                str(target.get('name', '')),
                dv_gain.get(f"dv_{int(target.get('pos', 0))}", 0.0),
            ) or 0.0
        )
        cv_per_unit = abs(gain_for_budget) if gain_for_budget > 1e-8 else 0.0
        if cv_per_unit > 0.0 and mv_authority_cv > 1e-9 and authority_frac > 0.0:
            new_delta, achieved_cv = clamp_event_to_authority_budget(
                proposed_delta=float(total_magnitude),
                cv_impact_per_unit=float(cv_per_unit),
                cumulative_cv_impact=float(_cumulative_cv_impact),
                mv_authority_cv=float(mv_authority_cv),
                target_frac=float(authority_frac),
            )
            total_magnitude = float(new_delta)
            _cumulative_cv_impact += float(achieved_cv)

        if abs(total_magnitude) < 1e-6:
            # Budget consumed: skip emitting a no-op event but advance time
            # so the next anchor is honored.
            last_start = start
            continue

        # ----- Event shape selection -----------------------------------
        # Mix step / ramp / drift / oscillation so the agent learns to
        # reject more than just instantaneous bumps.  Shapes are realised
        # at apply-time via _shape_step_increment.
        # Probabilities curriculum-coupled: at low intensity the policy
        # mostly sees clean steps (easy to learn), at high intensity it
        # sees a mix that better matches plant reality (drifts, ramps,
        # oscillating disturbances).
        shape_roll = float(rng.uniform())
        if shape_roll < 0.55:
            shape = 'step'
            duration = 1
        elif shape_roll < 0.75:
            shape = 'ramp'
            duration = int(rng.integers(
                max(4, int(0.50 * dyn_horizon)),
                max(8, int(2.00 * dyn_horizon)) + 1,
            ))
        elif shape_roll < 0.90:
            shape = 'drift'
            duration = int(rng.integers(
                max(8, int(4.00 * dyn_horizon)),
                max(16, int(8.00 * dyn_horizon)) + 1,
            ))
        else:
            shape = 'oscillation'
            duration = int(rng.integers(
                max(12, int(3.00 * dyn_horizon)),
                max(24, int(6.00 * dyn_horizon)) + 1,
            ))
        # Keep the event entirely inside the episode.
        duration = max(1, min(int(duration), int(ep_len) - int(start) - 5))
        if duration < 2 and shape != 'step':
            shape = 'step'
            duration = 1
        period = max(2.0, float(2.0 * max(1.0, dyn_horizon)))

        event = {
            "name": f"{source}_{target.get('name', 'channel')}_{i + 1}",
            "target_group": str(target.get('group', 'dv')),
            "target_pos": int(target.get('pos', 0)),
            "start": int(start),
            "duration": int(duration),
            "shape": str(shape),
            "period": float(period),
            "source": str(source),
            "delta": float(total_magnitude),
            "_applied": False,
            "_is_violation": bool(is_violation),
        }
        schedule.append(event)

        # ----- DV co-movement -----------------------------------------
        # With moderate probability emit a SIMULTANEOUS event on a
        # different DV at half-amplitude so the observer/agent see
        # correlated multi-DV behaviour (common in industrial plants
        # where flow + composition shift together).
        if (
            source == 'measured_dv'
            and len(dv_targets) >= 2
            and rng.uniform() < 0.25
        ):
            other_candidates = [
                t for t in dv_targets if int(t.get('pos', -1)) != int(target.get('pos', -2))
            ]
            if other_candidates:
                co_target = other_candidates[int(rng.integers(0, len(other_candidates)))]
                co_b = co_target.get('bounds')
                co_span = (
                    max(1e-6, float(co_b[1]) - float(co_b[0]))
                    if isinstance(co_b, list) and len(co_b) >= 2
                    else 100.0
                )
                # Half-amplitude, possibly anti-correlated so the pair
                # exercises both same-direction and opposing pushes.
                co_sign = sign if rng.uniform() < 0.65 else -sign
                co_delta = float(np.clip(
                    co_sign * 0.5 * abs(total_magnitude) * (co_span / max(1e-6, span)),
                    -0.20 * co_span, 0.20 * co_span,
                ))
                co_gain = float(
                    dv_gain.get(
                        str(co_target.get('name', '')),
                        dv_gain.get(f"dv_{int(co_target.get('pos', 0))}", 0.0),
                    ) or 0.0
                )
                co_cv_per_unit = abs(co_gain) if co_gain > 1e-8 else 0.0
                if co_cv_per_unit > 0.0 and mv_authority_cv > 1e-9 and authority_frac > 0.0:
                    co_new_delta, co_achieved_cv = clamp_event_to_authority_budget(
                        proposed_delta=co_delta,
                        cv_impact_per_unit=co_cv_per_unit,
                        cumulative_cv_impact=float(_cumulative_cv_impact),
                        mv_authority_cv=float(mv_authority_cv),
                        target_frac=float(authority_frac),
                    )
                    co_delta = float(co_new_delta)
                    _cumulative_cv_impact += float(co_achieved_cv)
                if abs(co_delta) > 1e-6:
                    schedule.append({
                        "name": f"{event['name']}_co",
                        "target_group": str(co_target.get('group', 'dv')),
                        "target_pos": int(co_target.get('pos', 0)),
                        "start": int(start),
                        "duration": int(duration),
                        "shape": str(shape),
                        "period": float(period),
                        "source": "measured_dv",
                        "delta": float(co_delta),
                        "_applied": False,
                        "_is_violation": False,
                        "_co_movement": True,
                    })
                    co_ch_key = f"{co_target.get('group', 'dv')}_{co_target.get('pos', 0)}"
                    _cumulative_offset[co_ch_key] = (
                        _cumulative_offset.get(co_ch_key, 0.0) + co_delta
                    )
        # Update cumulative offset tracker for this channel.
        _cumulative_offset[ch_key] = cum + float(total_magnitude)
        last_start = start
        if start >= ep_len - 20:
            break

    return schedule


def _apply_event_to_state(state: np.ndarray, sim, event: Dict) -> None:
    channels = _channel_catalog(sim)

    # Backward compatibility for legacy schedules.
    legacy_target = str(event.get('target', '')).strip().lower()
    if legacy_target:
        if legacy_target in {'feed_flow', 'feed_comp'}:
            event = {**event, 'target_group': 'dv', 'target_pos': 0 if legacy_target == 'feed_flow' else 1}
        else:
            # Legacy CV-targeted events are no longer supported; the
            # hidden OU process replaces them.  Silently drop.
            return

    group = str(event.get('target_group', '')).strip().lower()
    pos = int(event.get('target_pos', -1))
    delta = float(event.get('delta', 0.0))

    # Only measured-DV events are emitted now; ignore stale CV entries.
    if group != 'dv':
        return

    if group not in channels:
        return
    group_channels = channels[group]
    if not (0 <= pos < len(group_channels)):
        return

    ch = group_channels[pos]
    idx = int(ch['index'])

    # --- Immediate state modification (for instant step visibility) ------
    cur = _state_group_value(state, sim, group=group, pos=pos, idx=idx)
    tgt = float(cur + delta)

    b = ch.get('bounds')
    if isinstance(b, list) and len(b) >= 2:
        tgt = float(np.clip(tgt, float(b[0]), float(b[1])))

    # If clipping consumed most of the intended delta, flip direction so the
    # disturbance moves away from the limit instead of being stuck at it.
    if abs(delta) > 1e-9:
        achieved_frac = abs(tgt - cur) / abs(delta)
        if achieved_frac < 0.55:
            tgt = float(cur - delta)
            if isinstance(b, list) and len(b) >= 2:
                tgt = float(np.clip(tgt, float(b[0]), float(b[1])))
            event['delta'] = float(-delta)

    # --- Persistent offset (sim-level) -----------------------------------
    achieved_delta = float(tgt - cur)
    if hasattr(sim, 'set_disturbance_offset'):
        current_offset = float(sim.get_disturbance_offset(group, pos)) if hasattr(sim, 'get_disturbance_offset') else 0.0
        sim.set_disturbance_offset(group, pos, current_offset + achieved_delta)

    ranges = list(getattr(sim, 'dv_normalization_ranges', []))
    state[idx] = _engineering_to_raw(
        tgt,
        ranges=ranges,
        pos=pos,
        state_is_normalized=bool(getattr(sim, 'state_is_normalized', False)),
    )


def _shape_step_increment(event: Dict, t_local: int, duration: int) -> float:
    """Per-step delta for ramp/drift/oscillation shapes.

    Returns the *incremental* magnitude to apply at relative time ``t_local``
    (0-indexed) given a total ``event['delta']`` and ``duration`` steps.
    For ``ramp`` and ``drift`` the increments sum to ``delta`` over the
    duration.  For ``oscillation`` the *cumulative* state offset traces
    ``delta * sin(2*pi*(t_local+1)/period)`` so the disturbance returns
    near zero at the end of the window.
    """
    shape = str(event.get('shape', 'step'))
    delta = float(event.get('delta', 0.0))
    duration = max(1, int(duration))
    if shape in ('ramp', 'drift'):
        return delta / float(duration)
    if shape == 'oscillation':
        period = max(2.0, float(event.get('period', max(2.0, duration / 2.0))))
        omega = 2.0 * math.pi / period
        return delta * (math.sin(omega * (t_local + 1)) - math.sin(omega * t_local))
    return delta  # step fallback


def _apply_incremental_to_state(
    state: np.ndarray, sim, target_group: str, target_pos: int, inc: float
) -> None:
    """Apply an incremental delta without flip/hold side-effects.

    Used by shaped (ramp/drift/oscillation) events — these emit many small
    pieces over their duration and do not need the saturation-flip logic
    that one-shot step events use.
    """
    channels = _channel_catalog(sim)
    group = str(target_group).strip().lower()
    if group not in channels:
        return
    group_channels = channels[group]
    if not (0 <= int(target_pos) < len(group_channels)):
        return
    ch = group_channels[int(target_pos)]
    idx = int(ch['index'])

    cur = _state_group_value(state, sim, group=group, pos=int(target_pos), idx=idx)
    tgt = float(cur + float(inc))
    b = ch.get('bounds')
    if isinstance(b, list) and len(b) >= 2:
        tgt = float(np.clip(tgt, float(b[0]), float(b[1])))

    achieved_inc = float(tgt - cur)
    if hasattr(sim, 'set_disturbance_offset'):
        try:
            current_offset = (
                float(sim.get_disturbance_offset(group, int(target_pos)))
                if hasattr(sim, 'get_disturbance_offset')
                else 0.0
            )
            sim.set_disturbance_offset(group, int(target_pos), current_offset + achieved_inc)
        except Exception:
            pass

    if group == 'cv':
        ranges = list(getattr(sim, 'cv_normalization_ranges', []))
    elif group == 'dv':
        ranges = list(getattr(sim, 'dv_normalization_ranges', []))
    else:
        ranges = []
    state[idx] = _engineering_to_raw(
        tgt,
        ranges=ranges,
        pos=int(target_pos),
        state_is_normalized=bool(getattr(sim, 'state_is_normalized', False)),
    )


def apply_disturbance_schedule(
    state: np.ndarray,
    sim,
    schedule: List[Dict],
    mv_monitor: Optional[MVSaturationMonitor] = None,
) -> List[str]:
    """Apply pending step disturbances at their trigger time.

    Each event is an instantaneous step applied once at ``event['start']``.
    The ``_applied`` flag prevents re-application on subsequent steps.

    When ``mv_monitor`` is supplied and reports ``should_suppress()``, the
    pending event is marked suppressed (delta zeroed) instead of fired.
    This guarantees that even with a poorly-calibrated schedule the agent
    is not asked to absorb yet another disturbance while its MV is already
    pinned to a limit.
    """
    if not schedule:
        return []

    k = int(sim.episode_counter)
    active = []
    for ev in schedule:
        if ev.get('_applied', False):
            continue
        shape = str(ev.get('shape', 'step'))
        duration = max(1, int(ev.get('duration', 1)))
        start = int(ev['start'])
        t_local = k - start
        if t_local < 0:
            continue
        if shape == 'step' or duration <= 1:
            if t_local != 0:
                continue
            if mv_monitor is not None and mv_monitor.should_suppress():
                ev['_applied'] = True
                ev['_suppressed'] = True
                ev['delta'] = 0.0
                continue
            _apply_event_to_state(state, sim, ev)
            ev['_applied'] = True
            active.append(str(ev.get("name", "disturbance")))
        else:
            # Shaped (ramp/drift/oscillation) events: apply incremental piece
            # for each step in [start, start + duration).  Saturation
            # suppression only freezes the current increment; the next step
            # will resume the trajectory.
            if t_local >= duration:
                continue
            if mv_monitor is not None and mv_monitor.should_suppress():
                if t_local == duration - 1:
                    ev['_applied'] = True
                    ev['_suppressed'] = True
                continue
            inc = _shape_step_increment(ev, int(t_local), int(duration))
            if abs(inc) > 1e-12:
                _apply_incremental_to_state(
                    state, sim,
                    target_group=str(ev.get('target_group', 'dv')),
                    target_pos=int(ev.get('target_pos', 0)),
                    inc=float(inc),
                )
                if t_local == 0:
                    active.append(str(ev.get("name", "disturbance")))
            if t_local == duration - 1:
                ev['_applied'] = True

    if active and hasattr(sim, "episode_array"):
        sim.episode_array[k] = state

    return active
