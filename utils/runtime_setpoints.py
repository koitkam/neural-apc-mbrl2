"""Runtime setpoint manager: dynamic MV/CV limits and CV targets during training.

Teaches the agent to track operator changes at runtime by randomly varying
the active limits and targets inside each training/validation episode.

Design:
- Base MV/CV bounds and (optional) CV targets come from the objective spec.
- At episode reset, :meth:`reset` schedules a small number of bound and
  target events.  Event count follows a curriculum (rare early, frequent
  late) to avoid flooding a fresh policy.
- :meth:`step` applies scheduled changes (step or ramp) during the episode.
- :meth:`get_augmented_obs_channels` returns the per-channel observation
  augmentation in [-1, 1] (the policy must see the current setpoints).
  V4 packed layout:
    * per MV ch: [lo, hi]                                  (2 * n_mv)
    * per CV ch: [lo, hi, target, target_active_flag]      (4 * n_cv)
  Per-step overhead is ``2 * n_mv + 4 * n_cv`` scalars.
- The reward/objective layer reads ``current_mv_bounds``, ``current_cv_bounds``
  and ``current_cv_targets`` through the manager instead of the static spec.

Schedule invariants (verified by Monte-Carlo tests in the test suite):
- **Both bounds always step.**  Each event applies a coherent directional
  shift (sign + magnitude) to BOTH edges of the band, plus a small width
  jitter, so disturbance plots show genuine band motion rather than one
  edge pinned to the base limit.
- **No same-channel time overlap.**  Events on the same (kind, channel) are
  resampled until they land in disjoint time windows; otherwise the
  per-event ``start_value`` (= base) would cause snap-back glitches when
  events collide.
- **Hard outer envelope.**  When ``mv_norm_ranges`` / ``cv_norm_ranges`` are
  provided (from the simulator's ``mv_normalization_ranges`` /
  ``cv_normalization_ranges``), bound events are hard-clamped inside the
  normalisation range.  Targets are also clamped inside the norm range.
  This guarantees the policy/observer never see inputs outside the window
  they were normalised against.  When norm ranges are absent, the
  scheduler falls back to extending the base box by 0.20*span (MV) /
  0.40*span (CV).
- **Target inside current bounds, continuously.**  After every event apply
  the active target is clamped against the current (possibly mutated) CV
  bounds with an inward margin.  "Target outside limits is not allowed"
  holds at every timestep, not just at schedule time.

The manager is simulator-agnostic: simulators never need to implement
``set_bounds`` / ``set_target``.  The manager holds the authoritative
"operator setpoint" and all consumers query it.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


def _clip_range(lo: float, hi: float, floor_lo: float, ceil_hi: float,
                min_span_frac: float = 0.20) -> Tuple[float, float]:
    """Clip (lo, hi) to the base [floor_lo, ceil_hi] with a minimum span."""
    full_span = max(1e-9, ceil_hi - floor_lo)
    min_span = min_span_frac * full_span
    lo = float(np.clip(lo, floor_lo, ceil_hi))
    hi = float(np.clip(hi, floor_lo, ceil_hi))
    if hi - lo < min_span:
        center = 0.5 * (lo + hi)
        lo = max(floor_lo, center - 0.5 * min_span)
        hi = min(ceil_hi, center + 0.5 * min_span)
        if hi - lo < min_span:  # base range too tight; clamp to base
            lo, hi = floor_lo, ceil_hi
    return lo, hi


@dataclass
class _ScheduledChange:
    """A single scheduled bounds/target change during an episode."""

    t_start: int
    t_end: int                  # equal to t_start for a pure step
    channel_kind: str           # 'cv_bounds' | 'mv_bounds' | 'cv_target'
    channel_index: int
    start_value: Tuple[float, ...]
    end_value: Tuple[float, ...]


@dataclass
class RuntimeSetpointConfig:
    """Auto-derived runtime-setpoint schedule.

    The user-facing JSON only needs ``targets_enabled`` (default False). It may
    be a bool (applies to every CV) or a list of bools, one per CV, to enable
    targets on selected channels.  MV/CV bounds variation is *always on*.
    """

    bounds_enabled: bool = True             # always on
    targets_enabled: bool = False           # opt-in
    bounds_changes_per_episode: Tuple[int, int] = (1, 2)
    target_changes_per_episode: Tuple[int, int] = (1, 2)
    bounds_jitter_fraction: float = 0.15
    target_jitter_fraction: float = 0.20
    # Targets must stay at least this fraction of the base CV span inward
    # from each bound.  Prevents scheduling a target that sits on the
    # limit (physically untrackable) and gives ``_apply_events_at`` a
    # soft inward margin when a bound move would otherwise swallow the
    # active target.
    target_inside_margin_frac: float = 0.05
    change_style: str = 'step'
    ramp_duration_fraction: float = 0.10
    # Fraction of total training over which NO setpoint changes are
    # scheduled (the agent first stabilises against base bounds).  Was
    # 0.25 — too long in 600 k-step runs (~150 k steps frozen-bounds),
    # so the actor specialises on a static envelope and has to relearn
    # later.  0.10 keeps a brief warm-up while exposing limit-tracking
    # early enough that policy collapse on the static distribution is
    # avoided.
    curriculum_warmup_fraction: float = 0.10
    # Number of magnitude strata for shift sampling.  When > 1 the
    # per-event magnitude is drawn deterministically from one of N
    # equally-spaced strata over ``[0.4*jitter, 1.0*jitter]``, cycled
    # per (kind, channel) across episodes so every magnitude regime
    # — small/mid/large — is exercised.  Without strata, repeated
    # uniform sampling clusters near the mean (≈0.7*jitter) and the
    # boundary regimes (smallest meaningful step, largest allowed step)
    # are statistically rare.  Set 0/1 to disable.
    n_magnitude_strata: int = 3

    @classmethod
    def auto_derive(cls,
                    targets_enabled: bool,
                    episode_length: int,
                    tau_dominant: Optional[float] = None,
                    dead_time: Optional[float] = None,
                    dt: float = 1.0) -> 'RuntimeSetpointConfig':
        """Pick sensible defaults from system dynamics.

        - Number of changes per episode is bounded so each change has at least
          ``5 * (tau + theta)`` settling time.
        - Ramp duration is ~ ``2 * (tau + theta) / episode_length`` (bounded).
        - Jitter and curriculum warmup use safe defaults; not user-tunable.
        """
        ep = max(1, int(episode_length))
        if tau_dominant is None or not math.isfinite(float(tau_dominant)) or float(tau_dominant) <= 0:
            tau_steps = max(1.0, 0.05 * ep)         # fallback: 5% of episode
        else:
            tau_steps = max(1.0, float(tau_dominant) / max(1e-6, float(dt)))
        theta_steps = 0.0 if dead_time is None else max(0.0, float(dead_time) / max(1e-6, float(dt)))
        settle = max(1.0, 5.0 * (tau_steps + theta_steps))
        max_changes = max(1, int(ep // settle))
        n_max = int(min(4, max(1, max_changes)))
        # At minimum one change so the agent always sees movement.
        n_min = 1 if n_max >= 1 else 0

        ramp_frac = float(np.clip(2.0 * (tau_steps + theta_steps) / ep, 0.05, 0.25))

        # Jitter fraction governs how big each limit step is, as a fraction
        # of the base bound span.  15% was too small to be a meaningful
        # learning signal on plants with narrow CV bands (e.g. 5 °C band
        # -> 0.75 °C step, vanishing under routine closed-loop noise).
        # 25% gives a clear step (~1.25 °C for a 5 °C band) without pushing
        # bounds past their physical limits (``_clip_range`` caps at base bounds).
        bounds_jitter = float(os.environ.get('RUNTIME_SETPOINT_BOUNDS_JITTER_FRACTION', '0.25'))
        bounds_jitter = float(np.clip(bounds_jitter, 0.05, 0.45))
        target_jitter = float(os.environ.get('RUNTIME_SETPOINT_TARGET_JITTER_FRACTION', '0.25'))
        target_jitter = float(np.clip(target_jitter, 0.05, 0.45))

        return cls(
            bounds_enabled=True,
            targets_enabled=bool(targets_enabled),
            bounds_changes_per_episode=(n_min, n_max),
            target_changes_per_episode=(n_min, n_max),
            bounds_jitter_fraction=bounds_jitter,
            target_jitter_fraction=target_jitter,
            change_style='step',
            ramp_duration_fraction=ramp_frac,
            curriculum_warmup_fraction=0.10,
        )

    # Backward-compat: code that still reads .enabled treats it as "bounds on".
    @property
    def enabled(self) -> bool:
        return bool(self.bounds_enabled or self.targets_enabled)


@dataclass
class RuntimeSetpointManager:
    """Holds the authoritative active MV/CV bounds + optional CV targets.

    Use :meth:`from_spec` to build from the normalised objective spec.  The
    manager is process-local, thread-unsafe, and intended to be owned by the
    training/validation episode loop.
    """

    n_mv: int
    n_cv: int
    base_mv_bounds: np.ndarray        # (n_mv, 2)
    base_cv_bounds: np.ndarray        # (n_cv, 2)
    base_cv_targets: np.ndarray       # (n_cv,) NaN when disabled
    cv_target_enabled: np.ndarray     # (n_cv,) bool
    cfg: RuntimeSetpointConfig = field(default_factory=RuntimeSetpointConfig)
    # Steady-state |MV -> CV| gain per (mv_idx, cv_idx).  Shape (n_mv, n_cv).
    # When unknown, entries are 0 and the scheduler falls back to pure
    # span-based sizing.  When non-zero, bound shifts are additionally
    # capped so the demanded CV excursion is reachable within ~35% of the
    # MV actuator range (prevents impossible-to-track bound requests).
    mv_gain_to_cv: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype='float32'))
    # Hard outer envelope for bound/target excursions.  When set (non-empty
    # and finite), the scheduler will never produce a bound or target outside
    # this box.  Defaults to an empty array meaning "use extended base box".
    # Source: simulator's mv_normalization_ranges / cv_normalization_ranges
    # so bound steps stay inside the same window the policy was trained on.
    mv_norm_bounds: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), dtype='float32'))
    cv_norm_bounds: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), dtype='float32'))

    # Active (possibly mutated) values.
    current_mv_bounds: np.ndarray = field(init=False)
    current_cv_bounds: np.ndarray = field(init=False)
    current_cv_targets: np.ndarray = field(init=False)

    _rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(0))
    _schedule: List[_ScheduledChange] = field(default_factory=list)
    _episode_length: int = 0
    _curriculum_fraction: float = 1.0
    # Per-(kind, channel) magnitude-stratum counter for stratified shift
    # sampling.  Cycles 0..N-1 across episodes so successive events on
    # the same channel land in successive strata.  Initial offset is
    # randomised at construction so seeds with different RNGs don't all
    # start in the same stratum.
    _mag_stratum_counter: Dict[Tuple[str, int], int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.base_mv_bounds = np.asarray(self.base_mv_bounds, dtype='float32').reshape(self.n_mv, 2)
        self.base_cv_bounds = np.asarray(self.base_cv_bounds, dtype='float32').reshape(self.n_cv, 2)
        self.base_cv_targets = np.asarray(self.base_cv_targets, dtype='float32').reshape(self.n_cv)
        self.cv_target_enabled = np.asarray(self.cv_target_enabled, dtype=bool).reshape(self.n_cv)
        gain = np.asarray(self.mv_gain_to_cv, dtype='float32')
        if gain.size == 0 or gain.ndim != 2 or gain.shape != (self.n_mv, self.n_cv):
            gain = np.zeros((self.n_mv, self.n_cv), dtype='float32')
        self.mv_gain_to_cv = gain
        # Validate / normalise norm-bound arrays.  Treat any malformed or
        # non-finite entry as "missing" so the scheduler falls back to the
        # extended base box for that channel.
        mvn = np.asarray(self.mv_norm_bounds, dtype='float32')
        if mvn.size == 0 or mvn.ndim != 2 or mvn.shape != (self.n_mv, 2):
            mvn = np.zeros((0, 2), dtype='float32')
        self.mv_norm_bounds = mvn
        cvn = np.asarray(self.cv_norm_bounds, dtype='float32')
        if cvn.size == 0 or cvn.ndim != 2 or cvn.shape != (self.n_cv, 2):
            cvn = np.zeros((0, 2), dtype='float32')
        self.cv_norm_bounds = cvn
        self.current_mv_bounds = self.base_mv_bounds.copy()
        self.current_cv_bounds = self.base_cv_bounds.copy()
        self.current_cv_targets = self.base_cv_targets.copy()

    # ---------------------------------------------------------------- builders
    @classmethod
    def from_spec(cls, spec: Dict, n_mv: int, n_cv: int,
                  seed: Optional[int] = None,
                  episode_length: int = 1000,
                  tau_dominant: Optional[float] = None,
                  dead_time: Optional[float] = None,
                  dt: float = 1.0,
                  mv_gain_to_cv: Optional[Sequence] = None,
                  mv_norm_ranges: Optional[Sequence] = None,
                  cv_norm_ranges: Optional[Sequence] = None) -> 'RuntimeSetpointManager':
        bounds_dict = spec.get('bounds', {}) if isinstance(spec, dict) else {}
        mv_list = bounds_dict.get('mv_bounds') or []
        cv_list = bounds_dict.get('cv_bounds') or []
        mv_arr = np.asarray(mv_list, dtype='float32').reshape(-1, 2) if mv_list else \
            np.tile([[0.0, 100.0]], (n_mv, 1)).astype('float32')
        cv_arr = np.asarray(cv_list, dtype='float32').reshape(-1, 2) if cv_list else \
            np.tile([[-1e12, 1e12]], (n_cv, 1)).astype('float32')
        if mv_arr.shape[0] < n_mv:
            mv_arr = np.vstack([mv_arr, np.tile([[0.0, 100.0]], (n_mv - mv_arr.shape[0], 1))])
        if cv_arr.shape[0] < n_cv:
            cv_arr = np.vstack([cv_arr, np.tile([[-1e12, 1e12]], (n_cv - cv_arr.shape[0], 1))])

        tgt_dict = (spec.get('targets') or {}).get('cvs') or {}
        tgt_list = (spec.get('weights') or {}).get('cv_target_values') or []
        enabled = np.zeros(n_cv, dtype=bool)
        targets = np.full(n_cv, np.nan, dtype='float32')
        if tgt_dict:
            for i in range(n_cv):
                key = f'cv_{i}'
                if key in tgt_dict:
                    targets[i] = float(tgt_dict[key])
                    enabled[i] = True
        elif tgt_list:
            for i, v in enumerate(tgt_list[:n_cv]):
                try:
                    fv = float(v)
                    if math.isfinite(fv):
                        targets[i] = fv
                        enabled[i] = True
                except Exception:
                    continue

        cfg_dict = spec.get('runtime_setpoints') or {}
        # `targets_enabled` may be a bool (applies to all CVs) or a per-CV list/vector.
        targets_enabled_raw = cfg_dict.get('targets_enabled', False)
        if isinstance(targets_enabled_raw, (list, tuple, np.ndarray)):
            per_cv = np.zeros(n_cv, dtype=bool)
            for i, v in enumerate(list(targets_enabled_raw)[:n_cv]):
                try:
                    per_cv[i] = bool(v)
                except Exception:
                    per_cv[i] = False
        else:
            per_cv = np.full(n_cv, bool(targets_enabled_raw), dtype=bool)
        # A target is only active when the user enabled it AND a finite target was provided.
        enabled = enabled & per_cv
        targets_enabled_any = bool(enabled.any())

        cfg = RuntimeSetpointConfig.auto_derive(
            targets_enabled=targets_enabled_any,
            episode_length=episode_length,
            tau_dominant=tau_dominant,
            dead_time=dead_time,
            dt=dt,
        )

        rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
        # Normalise mv_gain_to_cv argument.  Accept:
        #   - None / empty            -> zero matrix (span-only sizing)
        #   - 2D array (n_mv, n_cv)   -> used as-is
        #   - dict {name_or_mv_i: gain_scalar} with single-CV assumption
        #     -> broadcast to shape (n_mv, 1)
        gain_mat = np.zeros((n_mv, n_cv), dtype='float32')
        try:
            if isinstance(mv_gain_to_cv, dict):
                # Assume single-CV plant: column 0 gets the scalar per MV key.
                for k, v in mv_gain_to_cv.items():
                    try:
                        fv = float(v)
                    except Exception:
                        continue
                    if not math.isfinite(fv):
                        continue
                    key = str(k)
                    # Accept 'mv_{i}' keys directly.
                    if key.startswith('mv_'):
                        try:
                            idx = int(key.split('_', 1)[1])
                        except Exception:
                            continue
                        if 0 <= idx < n_mv and n_cv >= 1:
                            gain_mat[idx, 0] = abs(fv)
            elif mv_gain_to_cv is not None:
                arr = np.asarray(mv_gain_to_cv, dtype='float32')
                if arr.ndim == 1 and n_cv == 1 and arr.shape[0] == n_mv:
                    gain_mat[:, 0] = np.abs(arr)
                elif arr.ndim == 2 and arr.shape == (n_mv, n_cv):
                    gain_mat = np.abs(arr).astype('float32')
        except Exception:
            gain_mat = np.zeros((n_mv, n_cv), dtype='float32')

        # Normalise per-channel normalization ranges into (n, 2) float32
        # arrays.  Reject malformed or non-finite ranges per channel; the
        # scheduler falls back to the extended base box for those channels.
        def _coerce_ranges(seq, n_expected: int) -> np.ndarray:
            if seq is None:
                return np.zeros((0, 2), dtype='float32')
            try:
                arr = np.asarray(list(seq), dtype='float32')
            except Exception:
                return np.zeros((0, 2), dtype='float32')
            if arr.ndim != 2 or arr.shape[1] != 2 or arr.shape[0] < n_expected:
                return np.zeros((0, 2), dtype='float32')
            arr = arr[:n_expected].astype('float32')
            # Replace per-row non-finite or zero-span ranges with a sentinel
            # that __post_init__ will treat as missing for the whole array.
            for i in range(arr.shape[0]):
                lo, hi = float(arr[i, 0]), float(arr[i, 1])
                if (not math.isfinite(lo)) or (not math.isfinite(hi)) or hi - lo <= 0.0:
                    return np.zeros((0, 2), dtype='float32')
            return arr

        mv_norm_arr = _coerce_ranges(mv_norm_ranges, n_mv)
        cv_norm_arr = _coerce_ranges(cv_norm_ranges, n_cv)

        mgr = cls(
            n_mv=n_mv, n_cv=n_cv,
            base_mv_bounds=mv_arr[:n_mv], base_cv_bounds=cv_arr[:n_cv],
            base_cv_targets=targets, cv_target_enabled=enabled, cfg=cfg,
            mv_gain_to_cv=gain_mat,
            mv_norm_bounds=mv_norm_arr,
            cv_norm_bounds=cv_norm_arr,
        )
        mgr._rng = rng
        return mgr

    # ----------------------------------------------------------------- episode
    def reset(self, episode_length: int, curriculum_fraction: float = 1.0) -> None:
        """Reset active setpoints for a new episode and pre-schedule any changes."""
        self._episode_length = int(max(1, episode_length))
        self._curriculum_fraction = float(max(0.0, min(1.0, curriculum_fraction)))
        self.current_mv_bounds = self.base_mv_bounds.copy()
        self.current_cv_bounds = self.base_cv_bounds.copy()
        self.current_cv_targets = self.base_cv_targets.copy()
        self._schedule = []
        if not self.cfg.enabled:
            return
        # Curriculum: no changes until ``curriculum_warmup_fraction`` of training.
        if self._curriculum_fraction < self.cfg.curriculum_warmup_fraction:
            return

        effective_fraction = (self._curriculum_fraction - self.cfg.curriculum_warmup_fraction) / \
            max(1e-6, 1.0 - self.cfg.curriculum_warmup_fraction)
        effective_fraction = float(max(0.0, min(1.0, effective_fraction)))

        def _sample_event_count(bounds: Tuple[int, int]) -> int:
            lo, hi = int(bounds[0]), int(bounds[1])
            if hi <= lo:
                return max(0, lo)
            # Scale upper range with curriculum.
            scaled_hi = lo + int(round((hi - lo) * effective_fraction))
            return int(self._rng.integers(lo, scaled_hi + 1))

        self._schedule_bounds_changes(_sample_event_count(self.cfg.bounds_changes_per_episode))
        if self.cfg.targets_enabled:
            self._schedule_target_changes(_sample_event_count(self.cfg.target_changes_per_episode))
        # Apply any events scheduled at t=0 immediately.
        self._apply_events_at(0)

    def _pick_style(self) -> str:
        style = self.cfg.change_style.lower()
        if style == 'mixed':
            return 'step' if self._rng.random() < 0.5 else 'ramp'
        return style if style in ('step', 'ramp') else 'step'

    def _ramp_duration(self) -> int:
        return max(1, int(round(self._episode_length * self.cfg.ramp_duration_fraction)))

    def _sample_magnitude_fraction(self, kind: str, ch: int, jitter: float) -> float:
        """Stratified magnitude draw over ``[0.4*jitter, 1.0*jitter]``.

        Cycles through ``cfg.n_magnitude_strata`` equally-spaced strata
        per ``(kind, ch)`` so successive events on the same channel
        rotate through small / mid / large magnitudes.  Within a stratum,
        a small uniform jitter (±half the stratum width) is added so
        repeated visits to the same stratum still vary slightly.

        Set ``cfg.n_magnitude_strata <= 1`` to fall back to pure uniform
        sampling (legacy behaviour).
        """
        n_strata = int(getattr(self.cfg, 'n_magnitude_strata', 0) or 0)
        lo_frac = 0.4 * jitter
        hi_frac = 1.0 * jitter
        if n_strata <= 1:
            return float(self._rng.uniform(lo_frac, hi_frac))
        key = (kind, int(ch))
        idx = int(self._mag_stratum_counter.get(key, int(self._rng.integers(0, n_strata))))
        self._mag_stratum_counter[key] = (idx + 1) % n_strata
        # Stratum centers evenly spaced across [lo_frac, hi_frac].
        if n_strata == 1:
            center = 0.5 * (lo_frac + hi_frac)
            half_w = 0.5 * (hi_frac - lo_frac)
        else:
            step = (hi_frac - lo_frac) / float(n_strata)
            center = lo_frac + (idx + 0.5) * step
            half_w = 0.5 * step
        return float(np.clip(self._rng.uniform(center - half_w, center + half_w),
                              lo_frac, hi_frac))

    def _schedule_bounds_changes(self, n_events: int) -> None:
        """Schedule bounds-change events.

        ``n_events`` is the target count per channel kind (MV and CV).  When
        both MV and CV channels exist, we schedule independent events for
        each kind so every episode exercises both MV-limit and CV-limit
        changes (critical for the agent to learn to track each).  When only
        one kind has channels, all events go to that kind.
        """
        if n_events <= 0:
            return
        have_mv = self.n_mv > 0
        have_cv = self.n_cv > 0
        if not (have_mv or have_cv):
            return
        # Allocate events per kind.
        if have_mv and have_cv:
            mv_events = n_events
            cv_events = n_events
        elif have_mv:
            mv_events = n_events
            cv_events = 0
        else:
            mv_events = 0
            cv_events = n_events
        margin = max(1, int(0.1 * self._episode_length))
        jitter = self.cfg.bounds_jitter_fraction
        # Per-kind extension fraction: how far outside the base box the band
        # is allowed to slide.  Without this, _clip_range clips one of the
        # two bounds back to the base limit on every event, so visually only
        # ONE side of the band moves per event (the other is pinned to base).
        # That's why upper-bound steps were missing from disturbance plots
        # whenever the random sign favoured a positive (downward-loose)
        # shift.  Allowing a modest outward extension makes BOTH bounds
        # always visibly step.  CV bounds are operator-set targets with
        # no hard physical limit so they can extend further than MV bounds,
        # which are bounded by actuator travel.
        extension_by_kind = {'mv_bounds': 0.20, 'cv_bounds': 0.40}

        def _overlaps(kind: str, ch: int, t_a: int, t_b: int) -> bool:
            """True if [t_a, t_b] intersects any existing event on (kind, ch)."""
            for prev in self._schedule:
                if prev.channel_kind != kind or prev.channel_index != ch:
                    continue
                if t_a <= prev.t_end and prev.t_start <= t_b:
                    return True
            return False

        def _schedule_for(kind: str, count: int) -> None:
            n_ch = self.n_mv if kind == 'mv_bounds' else self.n_cv
            if n_ch == 0 or count <= 0:
                return
            ext_frac = float(extension_by_kind.get(kind, 0.20))
            for _ in range(count):
                ch = int(self._rng.integers(0, n_ch))
                base = (self.base_mv_bounds if kind == 'mv_bounds'
                        else self.base_cv_bounds)[ch]
                span = max(1e-6, float(base[1] - base[0]))
                # Coherent directional shift: pick a sign and move BOTH
                # bounds together by a random magnitude, plus a small
                # independent width jitter.  Previously each bound moved
                # independently in [-jitter, jitter], so the band CENTER
                # rarely moved (expected center_shift == 0, std ~ 0.06*span
                # at jitter=0.15).  With a coherent shift, the center now
                # moves by sign * uniform(0.4*jitter, 1.0*jitter) * span,
                # which actually forces the agent to re-centre MV action.
                sign = 1.0 if self._rng.random() < 0.5 else -1.0
                mag_frac = self._sample_magnitude_fraction(kind, ch, jitter)
                center_shift = sign * mag_frac * span

                # Adapt shift magnitude to MV-reachable CV envelope so the
                # demanded CV excursion is achievable and meaningful.  For
                # CV bounds only (MV bounds are not CV-referred).
                if kind == 'cv_bounds' and self.mv_gain_to_cv.size > 0 and self.n_mv > 0:
                    # Max CV excursion reachable by ~35% MV actuator travel:
                    # sum_m (|k_m,cv| * 0.35 * mv_span_m).  If the coherent
                    # shift exceeds this, clip so the agent has a chance
                    # to track.  If the span-based shift is vanishingly
                    # small compared to the reachable envelope, bump the
                    # lower end so the step is at least 10% of the
                    # reachable envelope (meaningful learning signal).
                    reachable = 0.0
                    for m in range(self.n_mv):
                        mv_span_m = max(1e-6,
                                        float(self.base_mv_bounds[m, 1]
                                              - self.base_mv_bounds[m, 0]))
                        reachable += float(self.mv_gain_to_cv[m, ch]) * 0.35 * mv_span_m
                    if reachable > 1e-6:
                        abs_shift = abs(center_shift)
                        # Cap at reachable envelope (agent can track).
                        if abs_shift > reachable:
                            abs_shift = reachable
                        # Floor at 25% of reachable so the change stands
                        # clearly above closed-loop noise and is visible
                        # on plots.  Never exceed the user-configured
                        # ``jitter * span`` ceiling so the knob remains
                        # meaningful on high-gain plants.
                        min_meaningful = min(0.25 * reachable, jitter * span)
                        if abs_shift < min_meaningful:
                            abs_shift = min_meaningful
                        center_shift = sign * abs_shift

                # Small independent width jitter so the band width varies
                # within +/- 30% of the jitter fraction.
                width_shift = float(self._rng.uniform(-0.3 * jitter, 0.3 * jitter)) * span
                # Outer envelope: prefer simulator-declared normalization
                # range so bound steps NEVER fall outside the window the
                # observer/policy was normalised against.  When norm range
                # is missing, fall back to extending the base box by
                # ext_frac*span (preserves the previous fix that lets BOTH
                # edges step instead of pinning one to the base).
                ext = ext_frac * span
                if kind == 'mv_bounds' and self.mv_norm_bounds.shape[0] == self.n_mv:
                    floor_lo = float(self.mv_norm_bounds[ch, 0])
                    ceil_hi = float(self.mv_norm_bounds[ch, 1])
                elif kind == 'cv_bounds' and self.cv_norm_bounds.shape[0] == self.n_cv:
                    floor_lo = float(self.cv_norm_bounds[ch, 0])
                    ceil_hi = float(self.cv_norm_bounds[ch, 1])
                else:
                    floor_lo = float(base[0]) - ext
                    ceil_hi = float(base[1]) + ext
                new_lo, new_hi = _clip_range(
                    float(base[0]) + center_shift - width_shift,
                    float(base[1]) + center_shift + width_shift,
                    floor_lo,
                    ceil_hi,
                )
                style = self._pick_style()
                # Sample a non-overlapping time window for this channel.
                # Without this, two events on the same channel can collide
                # and produce snap-back glitches in the bound trace because
                # _apply_events_at iterates events in schedule order using
                # each event's own start_value=base.  Reject up to 8 attempts.
                ramp_dur = self._ramp_duration()
                t_start = -1
                t_end = -1
                for _attempt in range(8):
                    cand_start = int(self._rng.integers(
                        margin,
                        max(margin + 1, self._episode_length - margin),
                    ))
                    cand_end = (cand_start if style == 'step'
                                else min(self._episode_length - 1,
                                         cand_start + ramp_dur))
                    if not _overlaps(kind, ch, cand_start, cand_end):
                        t_start, t_end = cand_start, cand_end
                        break
                if t_start < 0:
                    # All attempts collided; skip this event to keep the
                    # trace clean rather than emit a colliding event.
                    continue
                self._schedule.append(_ScheduledChange(
                    t_start=t_start, t_end=t_end, channel_kind=kind,
                    channel_index=ch,
                    start_value=(float(base[0]), float(base[1])),
                    end_value=(float(new_lo), float(new_hi)),
                ))

        _schedule_for('mv_bounds', mv_events)
        _schedule_for('cv_bounds', cv_events)

    def _schedule_target_changes(self, n_events: int) -> None:
        if n_events <= 0 or not bool(self.cv_target_enabled.any()):
            return
        enabled_idx = np.where(self.cv_target_enabled)[0]
        if enabled_idx.size == 0:
            return
        jitter = self.cfg.target_jitter_fraction
        margin = max(1, int(0.1 * self._episode_length))
        for _ in range(n_events):
            ch = int(self._rng.choice(enabled_idx))
            base_target = float(self.base_cv_targets[ch])
            lo_base = float(self.base_cv_bounds[ch, 0])
            hi_base = float(self.base_cv_bounds[ch, 1])
            span = max(1e-6, hi_base - lo_base)

            # Coherent directional target shift, mirroring the bounds scheduler
            # so the centre of the operating envelope actually moves.  The old
            # ``uniform(-jitter, jitter) * span`` call averaged to zero and
            # made many episodes effectively target-static.
            sign = 1.0 if self._rng.random() < 0.5 else -1.0
            mag_frac = self._sample_magnitude_fraction('cv_target', ch, jitter)
            shift = sign * mag_frac * span

            # Gain-aware envelope: ensure the demanded target excursion is
            # reachable within ~35 % of the MV travel.  Falls back to pure
            # span sizing when no MV gain is known.
            if self.mv_gain_to_cv.size > 0 and self.n_mv > 0:
                reachable = 0.0
                for m in range(self.n_mv):
                    mv_span_m = max(1e-6,
                                    float(self.base_mv_bounds[m, 1]
                                          - self.base_mv_bounds[m, 0]))
                    reachable += float(self.mv_gain_to_cv[m, ch]) * 0.35 * mv_span_m
                if reachable > 1e-6:
                    abs_shift = abs(shift)
                    if abs_shift > reachable:
                        abs_shift = reachable
                    min_meaningful = min(0.10 * reachable, jitter * span)
                    if abs_shift < min_meaningful:
                        abs_shift = min_meaningful
                    shift = sign * abs_shift

            # Targets must stay inside the base CV bounds with a small
            # inward margin (``target_inside_margin_frac``) so the target is
            # physically trackable without parking on a limit.  Individual
            # bound events can temporarily move the band; :meth:`_apply_events_at`
            # will additionally clamp the active target against the current
            # bounds each step so the "target outside limits is not allowed"
            # invariant holds continuously, not just at schedule time.
            inward = float(self.cfg.target_inside_margin_frac) * span
            tgt_lo = lo_base + inward
            tgt_hi = hi_base - inward
            if tgt_hi <= tgt_lo:
                tgt_lo = lo_base
                tgt_hi = hi_base
            # Hard clamp to the simulator normalization range so target
            # excursions never escape the window the observer/policy was
            # trained on.  When norm range is missing, leave the base-bound
            # window as the outer limit (legacy behavior).
            if self.cv_norm_bounds.shape[0] == self.n_cv:
                norm_lo = float(self.cv_norm_bounds[ch, 0])
                norm_hi = float(self.cv_norm_bounds[ch, 1])
                tgt_lo = max(tgt_lo, norm_lo)
                tgt_hi = min(tgt_hi, norm_hi)
                if tgt_hi <= tgt_lo:
                    tgt_lo, tgt_hi = norm_lo, norm_hi
            new_target = float(np.clip(base_target + shift, tgt_lo, tgt_hi))

            style = self._pick_style()
            ramp_dur = self._ramp_duration()
            # Reject overlapping target events on the same channel so the
            # target trace is monotone-piecewise per channel rather than a
            # snap-back staircase.
            def _target_overlaps(c: int, t_a: int, t_b: int) -> bool:
                for prev in self._schedule:
                    if prev.channel_kind != 'cv_target' or prev.channel_index != c:
                        continue
                    if t_a <= prev.t_end and prev.t_start <= t_b:
                        return True
                return False
            t_start = -1
            t_end = -1
            for _attempt in range(8):
                cand_start = int(self._rng.integers(margin, max(margin + 1, self._episode_length - margin)))
                cand_end = cand_start if style == 'step' else min(self._episode_length - 1, cand_start + ramp_dur)
                if not _target_overlaps(ch, cand_start, cand_end):
                    t_start, t_end = cand_start, cand_end
                    break
            if t_start < 0:
                continue
            self._schedule.append(_ScheduledChange(
                t_start=t_start, t_end=t_end, channel_kind='cv_target',
                channel_index=ch,
                start_value=(base_target,),
                end_value=(new_target,),
            ))

    def step(self, t: int) -> None:
        """Advance one timestep; apply scheduled changes whose window contains ``t``."""
        if not self._schedule:
            return
        self._apply_events_at(int(t))

    def _apply_events_at(self, t: int) -> None:
        for ev in self._schedule:
            if t < ev.t_start:
                continue
            # Interpolation factor 0..1 across [t_start, t_end]; clamp past t_end.
            if ev.t_end <= ev.t_start:
                frac = 1.0
            else:
                frac = float(np.clip((t - ev.t_start) / max(1, (ev.t_end - ev.t_start)), 0.0, 1.0))
            if ev.channel_kind in ('cv_bounds', 'mv_bounds'):
                s_lo, s_hi = ev.start_value
                e_lo, e_hi = ev.end_value
                cur_lo = s_lo + frac * (e_lo - s_lo)
                cur_hi = s_hi + frac * (e_hi - s_hi)
                if ev.channel_kind == 'cv_bounds':
                    self.current_cv_bounds[ev.channel_index] = [cur_lo, cur_hi]
                else:
                    self.current_mv_bounds[ev.channel_index] = [cur_lo, cur_hi]
            elif ev.channel_kind == 'cv_target':
                s_t = ev.start_value[0]
                e_t = ev.end_value[0]
                cur = s_t + frac * (e_t - s_t)
                self.current_cv_targets[ev.channel_index] = cur

        # Continuous invariant: every enabled CV target must lie strictly
        # inside the *current* CV bounds (with a small inward margin).  A
        # target change and a bound change can be scheduled independently,
        # so without this clamp a target could temporarily drift outside
        # the band.  "A target outside of limits is not allowed" — enforce
        # it here each step so the reward layer never sees an inconsistent
        # setpoint.
        if self.n_cv > 0 and bool(self.cv_target_enabled.any()):
            inward_frac = float(self.cfg.target_inside_margin_frac)
            for ch in range(self.n_cv):
                if not bool(self.cv_target_enabled[ch]):
                    continue
                lo = float(self.current_cv_bounds[ch, 0])
                hi = float(self.current_cv_bounds[ch, 1])
                span = max(1e-9, hi - lo)
                inward = inward_frac * span
                tgt_lo = lo + inward
                tgt_hi = hi - inward
                if tgt_hi <= tgt_lo:
                    tgt_lo, tgt_hi = lo, hi
                cur = float(self.current_cv_targets[ch])
                if not math.isfinite(cur):
                    continue
                if cur < tgt_lo:
                    self.current_cv_targets[ch] = tgt_lo
                elif cur > tgt_hi:
                    self.current_cv_targets[ch] = tgt_hi

    # ----------------------------------------------------- observation output
    def get_augmented_obs_channels(self) -> np.ndarray:
        """Return per-channel observation augmentation in [-1, 1] (V4 layout).

        Packed per-channel layout (Dreamer-V4 rewrite):
            For each MV ch: [mv_lo_norm, mv_hi_norm]                 -> 2 * n_mv
            For each CV ch: [cv_lo_norm, cv_hi_norm,
                             cv_target_norm, cv_target_active_flag]  -> 4 * n_cv

        Total channels = 2 * n_mv + 4 * n_cv.

        - All bound/target values are normalised against the base MV/CV span
          (mid-centred; 0.0 = base mid-band, +/-1.0 = at the base edges).
        - cv_target_active_flag in {0.0, 1.0}; the policy reads it as an
          explicit "track target" gate rather than overloading 0.0 to mean
          "disabled" (which collides with a legitimate centred target).
        - When the base CV bound is unbounded (|hi-lo| > 1e10), all four CV
          channels emit 0.0 and the active flag is forced 0.
        """
        out: List[float] = []
        # MV: per-channel (lo, hi) packed.
        for ch in range(self.n_mv):
            base_lo = float(self.base_mv_bounds[ch, 0])
            base_hi = float(self.base_mv_bounds[ch, 1])
            span = max(1e-6, base_hi - base_lo)
            mid = 0.5 * (base_lo + base_hi)
            lo = float(np.clip(2.0 * (self.current_mv_bounds[ch, 0] - mid) / span, -1.0, 1.0))
            hi = float(np.clip(2.0 * (self.current_mv_bounds[ch, 1] - mid) / span, -1.0, 1.0))
            out.extend([lo, hi])
        # CV: per-channel (lo, hi, target, target_active) packed.
        for ch in range(self.n_cv):
            base_lo = float(self.base_cv_bounds[ch, 0])
            base_hi = float(self.base_cv_bounds[ch, 1])
            if (not math.isfinite(base_lo) or not math.isfinite(base_hi)
                    or base_hi - base_lo > 1e10):
                out.extend([0.0, 0.0, 0.0, 0.0])
                continue
            span = max(1e-6, base_hi - base_lo)
            mid = 0.5 * (base_lo + base_hi)
            lo = float(np.clip(2.0 * (self.current_cv_bounds[ch, 0] - mid) / span, -1.0, 1.0))
            hi = float(np.clip(2.0 * (self.current_cv_bounds[ch, 1] - mid) / span, -1.0, 1.0))
            tgt_val = float(self.current_cv_targets[ch])
            active = bool(self.cv_target_enabled[ch]) and math.isfinite(tgt_val)
            if active:
                tgt = float(np.clip(2.0 * (tgt_val - mid) / span, -1.0, 1.0))
                flag = 1.0
            else:
                tgt = 0.0
                flag = 0.0
            out.extend([lo, hi, tgt, flag])
        return np.asarray(out, dtype='float32')

    @property
    def aug_obs_dim(self) -> int:
        # 2 per MV (lo, hi); 4 per CV (lo, hi, target, target_active_flag).
        return 2 * self.n_mv + 4 * self.n_cv

    # ------------------------------------------------------------- diagnostics
    def describe(self) -> Dict:
        return {
            'n_mv': int(self.n_mv),
            'n_cv': int(self.n_cv),
            'base_mv_bounds': self.base_mv_bounds.tolist(),
            'base_cv_bounds': self.base_cv_bounds.tolist(),
            'cv_target_enabled': self.cv_target_enabled.tolist(),
            'base_cv_targets': self.base_cv_targets.tolist(),
            'config': self.cfg.__dict__,
            'aug_obs_dim': int(self.aug_obs_dim),
        }
