"""Hidden, dynamics-respecting unmeasured-LOAD disturbance process.

Models the unobservable upstream upsets a deployed APC controller must reject
(feed-composition shifts, ambient changes, fouling, slugs) as a true unmeasured
LOAD with its OWN dynamics — NOT a step added directly onto the controlled
variable.

The single ``HiddenDisturbance`` class (2026-06-10 consolidation) does:

  L(t)  — a per-episode schedule of discrete unmeasured LOAD events
          (step / ramp / pulse; isolated or overlapping; reverting or held),
          magnitude capped at a fraction of the agent's MV->CV authority.  This
          is the upset ORIGIN and MAY be a sharp step (a feed valve switches).
  Gd    — the disturbance->CV transfer function, FOPDT with UNIT DC gain:
          ``Gd(s) = e^{-theta_d s} / (tau_d s + 1)``.  ``theta_d`` (transport
          delay) and ``tau_d`` (lag) are sampled per-episode ADAPTIVELY from the
          identified plant timing (dead_time, tau_dom).
  d_cv  — ``Gd`` applied to ``L``: the SMOOTH, DELAYED contribution actually
          added to the CV output.  Unit DC gain preserves the authority cap.

The OU state / load schedule is **never exposed** to the agent or the world
model — it is a true unmeasured disturbance.

Why this design (the p33/p110 finding)
---------------------------------------
Adding the disturbance DIRECTLY (un-lagged) to the CV is a step/Dirac on the
controlled variable: the world model has zero observable signal predicting the
spike, so it incurs a loss/gain-bias floor that no amount of training removes,
and the feed-forward disturbance head cannot predict it.  Routing the load
through ``Gd`` makes the CV effect a first-order response — smooth and
autocorrelated — which the WM CAN learn from its lookback context and the head
CAN predict.

Sizing rationale (all sim-adaptive, env-overridable)
-----------------------------------------------------
- ``tau_d`` ~ ``U(0.5, 1.0) * tau_dom``: comparable to the plant lag, so the
  disturbance is clearly learnable (not high-frequency noise) yet distinct.
- ``theta_d`` ~ ``U(0.5, 1.5) * dead_time`` (floored at >=1 step): a realistic
  transport delay — a real upset takes time to reach the CV.
- Load amplitude <= ``amp_frac * mv_authority_cv`` (default 0.10) so the policy
  always has headroom to reject; unit-DC-gain ``Gd`` preserves this at the CV.
- Per-episode Bernoulli toggle: with probability ``DREAMER_DISTURBANCE_PROB``
  the episode has this hidden disturbance; otherwise it is clean.

This module is intentionally sim-agnostic: it only touches CV channels
identified through ``sim.cv_indices`` and respects
``sim.state_is_normalized`` for unit handling.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

import numpy as np


def _env_float(name: str, default: float) -> float:
    """Read a float env var with a fallback (never raises)."""
    try:
        return float(os.environ.get(name, str(default)))
    except Exception:
        return float(default)


def _env_pair(name: str, lo_default: float, hi_default: float) -> tuple:
    """Read a ``"lo:hi"`` env var as a (lo, hi) float pair (never raises).

    Returns ``(lo_default, hi_default)`` when unset/malformed; if a single
    value is given it is used for both ends; ensures ``hi >= lo``.
    """
    raw = os.environ.get(name, '').strip()
    if not raw:
        return float(lo_default), float(hi_default)
    try:
        if ':' in raw:
            a, b = raw.split(':')
            lo, hi = float(a), float(b)
        else:
            lo = hi = float(raw)
        if hi < lo:
            hi = lo
        return lo, hi
    except Exception:
        return float(lo_default), float(hi_default)


def hidden_disturbance_enabled(default: bool = True) -> bool:
    """Whether the hidden CV disturbance is active.

    Default: ON.  The legacy step-shaped unmeasured-CV events are gone
    (run p33 showed they capped ``sf_loss`` at ~1.15 by being Dirac
    spikes the WM has no observation to predict).  Off-switch only:
    set ``DREAMER_HIDDEN_DISTURBANCE=0`` to disable for ablations.
    """
    raw = os.environ.get('DREAMER_HIDDEN_DISTURBANCE')
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() not in {'0', 'false', 'no', 'off', ''}


def curriculum_amp_scale(progress: float, phase: Optional[int] = None) -> float:
    """Amplitude curriculum scale (≤cap) as a function of training progress.

    Reads ``DREAMER_HIDDEN_OU_AMP_RAMP="<start_frac>:<reach_full_at>"``.
    ``start_frac``: amplitude scale at ``progress=0`` (e.g. 0.1 = 10% of
    nominal).  ``reach_full_at``: progress fraction at which the scale
    reaches the cap (e.g. 0.4 = full amplitude by 40% of training).

    Phase-aware cap (P38, 2026-05-22):
      - P1/P2 (or phase unknown): ``DREAMER_HIDDEN_OU_AMP_MAX_SCALE``
        (default 0.2). Keeps the WM in a regime where it can learn
        base dynamics first; the critic only sees lightly-disturbed
        imagined starts.
      - P3: ``DREAMER_HIDDEN_OU_AMP_MAX_SCALE_P3`` (default 1.0). The
        WM is frozen and the actor must learn to reject
        realistic-magnitude upsets.

    Default (env unset or malformed ramp): cap value — no curriculum,
    cap still applies.

    The OU is still hidden — only its magnitude is shaped over time.
    """
    if phase is not None and int(phase) >= 3:
        env_key = 'DREAMER_HIDDEN_OU_AMP_MAX_SCALE_P3'
        cap_default = 1.0
    else:
        env_key = 'DREAMER_HIDDEN_OU_AMP_MAX_SCALE'
        cap_default = 0.2
    try:
        cap = float(os.environ.get(env_key, str(cap_default)))
    except Exception:
        cap = cap_default
    cap = float(np.clip(cap, 0.0, 1.0))
    raw = os.environ.get('DREAMER_HIDDEN_OU_AMP_RAMP', '0.1:0.4').strip()
    if not raw:
        return cap
    try:
        start_str, reach_str = raw.split(':')
        start = float(start_str)
        reach = float(reach_str)
    except Exception:
        return cap
    p = float(np.clip(progress, 0.0, 1.0))
    start = float(np.clip(start, 0.0, 1.0))
    reach = float(np.clip(reach, 1e-6, 1.0))
    if p >= reach:
        return cap
    # Linear ramp from start at p=0 to 1.0 at p=reach, then clamped at cap.
    val = start + (1.0 - start) * (p / reach)
    return float(min(val, cap))


def _sample_amp_jitter(rng: np.random.Generator) -> float:
    """Per-episode amplitude DR factor (multiplier on amp_frac).

    Reads ``DREAMER_HIDDEN_OU_AMP_JITTER="<lo>:<hi>"`` (uniform band).
    Default (unset): ``1.0`` — no jitter.

    Part of Stage C #5: domain randomization across OU amplitude so the
    policy is robust to a *family* of disturbance magnitudes, not a
    single fixed amplitude.
    """
    # Default ON (P37 onward): uniform ±60% around nominal amplitude.
    # Set ``DREAMER_HIDDEN_OU_AMP_JITTER=1.0:1.0`` to disable.
    raw = os.environ.get('DREAMER_HIDDEN_OU_AMP_JITTER', '0.6:1.6').strip()
    if not raw:
        return 1.0
    try:
        lo_str, hi_str = raw.split(':')
        lo = float(lo_str); hi = float(hi_str)
        if hi <= lo:
            return float(lo)
        return float(rng.uniform(lo, hi))
    except Exception:
        return 1.0


def _sample_drift_frac(rng: np.random.Generator) -> float:
    """Per-episode constant drift offset (as a fraction of amp).

    Reads ``DREAMER_HIDDEN_OU_DRIFT_FRAC="<max>"`` (uniform in ``[-max,+max]``).
    Default (unset): ``0.0`` — zero-mean OU (no drift).

    Adds a constant bias to the OU mean per episode so the policy must
    handle not just zero-mean random walks but also a slowly-shifted
    operating-point bias.  Part of Stage C #5.
    """
    # Default ON (P37 onward): up to ±40% of nominal amplitude as a
    # constant per-episode mean offset.  Set ``DREAMER_HIDDEN_OU_DRIFT_FRAC=0``
    # to disable.
    raw = os.environ.get('DREAMER_HIDDEN_OU_DRIFT_FRAC', '0.4').strip()
    if not raw:
        return 0.0
    try:
        mx = float(raw)
    except Exception:
        return 0.0
    if mx <= 0.0:
        return 0.0
    return float(rng.uniform(-mx, mx))


def get_phase_disturbance_prob(
    phase: int,
    wm_best_score: Optional[float] = None,
    phase_progress: Optional[float] = None,
) -> float:
    """Return the per-episode probability that hidden disturbance fires.

    **Adaptive triggering (P38, 2026-05-22).**  Observable forcing
    (setpoint steps, DV ramps, MV exploration) fires on 100% of
    episodes; the hidden OU is a much harder learning signal because
    the WM has no observation to attribute it to.  Per-phase curriculum:

      - **P1 (WM training)**: adaptive on ``wm_best_score``. Interpolates
        between ``DREAMER_HIDDEN_OU_PROB_MIN`` (default 0.05) at score=0
        and ``DREAMER_DISTURBANCE_PROB_WM`` (default 0.10) at
        score=``DREAMER_HIDDEN_OU_PROB_TARGET_SCORE`` (default 2.0).
      - **P2 (critic training)**: ramp on ``phase_progress`` from the
        P1/P2 cap (``DREAMER_DISTURBANCE_PROB_WM``, 0.10) to
        ``DREAMER_DISTURBANCE_PROB_P2`` (default 0.20). Reaches the P2
        cap by ``DREAMER_HIDDEN_OU_PROB_P2_RAMP_REACH`` (default 0.5 =
        midpoint of P2). Rationale: critic learns value of imagined
        rollouts that start from buffered real states. If the buffer
        only contains lightly-disturbed states (P1 cap), the critic
        never learns values for the disturbed manifold. Ramping P2
        gradually broadens buffer coverage without destabilising the
        still-updating WM.
      - **P3 (actor + critic)**: ramp on ``phase_progress`` from the P2
        cap (0.20) to ``DREAMER_DISTURBANCE_PROB_AGENT`` (default 0.30,
        P89: was 0.50 — 50 % hidden-upset episodes corrupted too much of
        the actor's gradient and never let the CV settle; a realistic
        plant sees occasional upsets, ~20-30 % of the time).
        Reaches the P3 cap by ``DREAMER_HIDDEN_OU_PROB_P3_RAMP_REACH``
        (default 0.5 = midpoint of P3). Rationale: actor needs to learn
        observable tracking before robust rejection; a step to 0.50 from
        day one of P3 corrupts ~half of its gradient signal on a
        randomly-initialised policy.

    Backward compatible: omitting ``phase_progress`` returns the
    phase-end cap (P1 still uses wm_best_score adaptation when supplied).
    """
    # ---- P3 ----
    if int(phase) >= 3:
        try:
            p3_cap = float(np.clip(
                float(os.environ.get('DREAMER_DISTURBANCE_PROB_AGENT', '0.3')),
                0.0, 1.0))
        except Exception:
            p3_cap = 0.3
        try:
            p3_floor = float(np.clip(
                float(os.environ.get('DREAMER_DISTURBANCE_PROB_P2', '0.2')),
                0.0, 1.0))
        except Exception:
            p3_floor = 0.2
        if phase_progress is None:
            return p3_cap
        try:
            reach = float(np.clip(
                float(os.environ.get(
                    'DREAMER_HIDDEN_OU_PROB_P3_RAMP_REACH', '0.5')),
                1e-6, 1.0))
        except Exception:
            reach = 0.5
        pp = float(np.clip(phase_progress, 0.0, 1.0))
        if pp >= reach:
            return p3_cap
        return float(p3_floor + (p3_cap - p3_floor) * (pp / reach))

    # P1/P2 share the static WM cap (also the P2 floor).
    raw_wm = os.environ.get('DREAMER_DISTURBANCE_PROB_WM', '0.10')
    try:
        wm_cap = float(np.clip(float(raw_wm), 0.0, 1.0))
    except Exception:
        wm_cap = 0.10

    # ---- P2 ----
    if int(phase) == 2:
        try:
            p2_cap = float(np.clip(
                float(os.environ.get('DREAMER_DISTURBANCE_PROB_P2', '0.2')),
                0.0, 1.0))
        except Exception:
            p2_cap = 0.2
        if p2_cap < wm_cap:
            p2_cap = wm_cap
        if phase_progress is None:
            return p2_cap
        try:
            reach = float(np.clip(
                float(os.environ.get(
                    'DREAMER_HIDDEN_OU_PROB_P2_RAMP_REACH', '0.5')),
                1e-6, 1.0))
        except Exception:
            reach = 0.5
        pp = float(np.clip(phase_progress, 0.0, 1.0))
        if pp >= reach:
            return p2_cap
        return float(wm_cap + (p2_cap - wm_cap) * (pp / reach))

    # ---- P1 (adaptive on wm_best_score) ----
    if wm_best_score is None:
        return wm_cap
    try:
        p_min = float(os.environ.get('DREAMER_HIDDEN_OU_PROB_MIN', '0.05'))
    except Exception:
        p_min = 0.05
    try:
        p_max = float(os.environ.get(
            'DREAMER_HIDDEN_OU_PROB_MAX', str(wm_cap)))
    except Exception:
        p_max = wm_cap
    try:
        target = float(os.environ.get(
            'DREAMER_HIDDEN_OU_PROB_TARGET_SCORE', '2.0'))
    except Exception:
        target = 2.0
    p_min = float(np.clip(p_min, 0.0, 1.0))
    p_max = float(np.clip(p_max, 0.0, 1.0))
    if p_max < p_min:
        p_max = p_min
    target = float(max(target, 1e-6))
    score = float(max(0.0, wm_best_score))
    frac = float(np.clip(score / target, 0.0, 1.0))
    return float(p_min + (p_max - p_min) * frac)


# NOTE (2026-06-10 consolidation): the legacy ``HiddenDisturbanceProcess`` (OU
# bias added DIRECTLY to the CV output) and the instant-event
# ``HiddenDisturbanceSchedule`` (also output-additive) were REMOVED.  Both
# injected the disturbance straight onto the CV un-lagged — physically a
# step/Dirac on the controlled variable, which (a) is unlike any real plant
# upset, (b) is an unlearnable target for the world model (no observable signal
# precedes a Dirac → a permanent WM loss/gain bias, the p33/p110 finding), and
# (c) is unpredictable for the feed-forward disturbance head.  They are replaced
# by the single ``HiddenDisturbance`` class below: a hidden LOAD (the unmeasured
# upset events) propagated to the CV through its OWN disturbance transfer
# function ``Gd`` = dead-time + first-order lag (adaptive from the identified
# plant), i.e. a true unmeasured DV with hidden dynamics.


def _cv_amp_caps(sim, amp_frac: float,
                  identifier_ctx: Optional[Dict] = None):
    """Per-CV steady-state disturbance amplitude cap.

    Returns ``(amp, is_normalized, spans)`` with ``amp`` aligned to
    ``sim.cv_indices``: ``amp[pos] = amp_frac * min(cv_span, MV->CV
    authority)`` so a hidden disturbance never exceeds a fraction of what
    the agent can rebalance by moving the MV across its op-band.  Converted
    to normalized units when the sim publishes normalized state.  Shared by
    the legacy OU process and the realistic event schedule.
    """
    cv_indices = [int(i) for i in list(getattr(sim, 'cv_indices', []))]
    is_norm = bool(getattr(sim, 'state_is_normalized', False))
    n_cv = len(cv_indices)
    if n_cv == 0:
        return np.zeros(0, dtype='float64'), is_norm, np.zeros(0, dtype='float64')
    cv_ranges = list(getattr(sim, 'cv_normalization_ranges', []))
    spans = np.ones(n_cv, dtype='float64')
    for pos in range(n_cv):
        if 0 <= pos < len(cv_ranges):
            lo, hi = float(cv_ranges[pos][0]), float(cv_ranges[pos][1])
            if hi > lo:
                spans[pos] = hi - lo
    mv_authority_cv = 0.0
    try:
        from utils.training_disturbance import compute_mv_authority_to_cv
        mv_authority_cv = float(
            compute_mv_authority_to_cv(sim, identifier_ctx) or 0.0)
    except Exception:
        mv_authority_cv = 0.0
    amp_eng = np.empty(n_cv, dtype='float64')
    for pos in range(n_cv):
        cap_authority = (amp_frac * mv_authority_cv
                          if mv_authority_cv > 1e-9 else float('inf'))
        amp_eng[pos] = min(amp_frac * spans[pos], cap_authority)
    if is_norm:
        amp = amp_eng / np.maximum(spans, 1e-9)
    else:
        amp = amp_eng
    return amp.astype('float64'), is_norm, spans


class HiddenDisturbance:
    """Hidden unmeasured LOAD disturbance, propagated to the CV through its OWN
    disturbance transfer function ``Gd`` (dead-time + first-order lag).

    Models a true unmeasured DV / load upset (feed-composition shift, ambient
    change, fouling, slug) the way a real plant experiences it — NOT a step
    added straight onto the controlled variable:

      L(t)  — a sim-adaptive schedule of discrete unmeasured LOAD events
              (``step`` = a feed/valve switches; ``ramp`` = a gradual drift;
              ``pulse`` = a transient slug), isolated or overlapping, reverting
              or permanently held, magnitude capped at ``amp_frac`` of the
              agent's MV->CV authority.  This is the upset ORIGIN — it MAY be a
              sharp step.
      Gd    — the disturbance->CV transfer function, FOPDT with UNIT DC gain:
              ``Gd(s) = e^{-theta_d s} / (tau_d s + 1)``.  ``theta_d`` (transport
              delay) and ``tau_d`` (lag) are sampled per episode ADAPTIVELY from
              the identified plant timing (``dead_time``, ``tau_dom``).
      d_cv  — ``Gd`` applied to ``L``: the SMOOTH, DELAYED contribution actually
              added to the CV.  Unit DC gain keeps the steady-state magnitude
              within the authority cap.

    Output-additive (DMC-style output disturbance): ``d_cv`` is added to the CV
    AFTER the plant step because ``Gd`` already encodes the full disturbance->CV
    path (the plant lag is NOT re-applied).  Because ``d_cv`` is a first-order
    response (never an instantaneous step on the CV), the world model CAN learn
    it from its lookback context and the feed-forward disturbance head CAN
    predict it (the p33/p110 Dirac-floor fix).

    NEVER exposed to the agent or WM (truly unmeasured).  Stable drop-in
    interface — ``is_empty()``, ``step(state)`` (in place; advances an internal
    counter), ``last_applied`` (the per-``cv_indices`` ``d_cv`` just injected),
    ``summary()`` — so the disturbance-head target (env trace), the validation
    disturbance-prediction diagnostic, and the rejection plot all read the same
    ``last_applied`` signal with no extra wiring.

    Env knobs (all optional; sim-adaptive defaults):
      ``DREAMER_HIDDEN_DIST_SETTLE_NTAU``   load-event settle = dead + N*tau (4)
      ``DREAMER_HIDDEN_DIST_MAX_EVENTS``    cap on load events / episode     (6)
      ``DREAMER_HIDDEN_DIST_P_ISOLATED``    P(gap >= settle)                 (0.5)
      ``DREAMER_HIDDEN_DIST_P_REVERT``      P(load event reverts)            (0.7)
      ``DREAMER_HIDDEN_DIST_SHAPE_WEIGHTS`` "step,ramp,pulse"         (0.5,0.3,0.2)
      ``DREAMER_HIDDEN_DIST_SPREAD``        spread events across the episode (1)
      ``DREAMER_HIDDEN_DIST_TAU_FRAC``      "lo:hi": tau_d=U(lo,hi)*tau_dom (0.5:1.0)
      ``DREAMER_HIDDEN_DIST_DEADTIME_FRAC`` "lo:hi": theta_d=U(lo,hi)*dead   (0.5:1.5)
    """

    def __init__(self, rng: np.random.Generator, sim, *,
                 tau_dom: float, sample_rate: float, dead_time: float,
                 episode_length: int, amp_frac: float = 0.10,
                 drift_frac: float = 0.0,
                 identifier_ctx: Optional[Dict] = None) -> None:
        self.sim = sim
        self.cv_indices = [int(i) for i in list(getattr(sim, 'cv_indices', []))]
        n_cv = len(self.cv_indices)
        self._rng = rng
        self._t = -1
        self.last_applied = np.zeros(max(n_cv, 0), dtype='float64')
        amp, is_norm, _spans = _cv_amp_caps(sim, float(amp_frac), identifier_ctx)
        self._is_norm = is_norm
        self.amp_cap = amp
        self.events: List[Dict] = []
        self._settle = 0.0
        self._tau_steps = 0.0
        # Gd state (per CV channel) — populated below when active.
        self.drift = np.zeros(max(n_cv, 0), dtype='float64')
        self._y = np.zeros(max(n_cv, 0), dtype='float64')
        self._delay: List = []
        self.tau_d_steps = np.zeros(max(n_cv, 0), dtype='float64')
        self.theta_d_steps = np.zeros(max(n_cv, 0), dtype='int64')
        self._alpha = np.zeros(max(n_cv, 0), dtype='float64')
        if n_cv == 0 or (amp.size and float(np.max(np.abs(amp))) <= 0.0):
            return

        sr = max(1e-6, float(sample_rate))
        tau_steps = max(1.0, float(tau_dom) / sr)         # agent steps
        dead_steps = max(0.0, float(dead_time) / sr)
        n_settle = _env_float('DREAMER_HIDDEN_DIST_SETTLE_NTAU', 4.0)
        settle = max(2.0, dead_steps + n_settle * tau_steps)
        self._settle = float(settle)
        self._tau_steps = float(tau_steps)
        T = int(episode_length)
        max_events = int(_env_float('DREAMER_HIDDEN_DIST_MAX_EVENTS', 6.0))
        p_isolated = float(np.clip(
            _env_float('DREAMER_HIDDEN_DIST_P_ISOLATED', 0.5), 0.0, 1.0))
        p_revert = float(np.clip(
            _env_float('DREAMER_HIDDEN_DIST_P_REVERT', 0.7), 0.0, 1.0))
        shapes = ['step', 'ramp', 'pulse']
        weights = self._shape_weights()

        # ----- Gd: per-channel disturbance transfer function (FOPDT) -----
        # tau_d ~ comparable to the plant dominant lag (smooth + learnable);
        # theta_d ~ a fraction-to-multiple of the plant dead time (transport
        # delay), floored at >=1 step so every disturbance is delayed.  Both
        # sim-adaptive (from the identified plant) + env-overridable.  UNIT DC
        # gain (the lag has no extra gain) so the authority-based amp cap on
        # the load is preserved at the CV.
        tau_lo, tau_hi = _env_pair('DREAMER_HIDDEN_DIST_TAU_FRAC', 0.5, 1.0)
        dt_lo, dt_hi = _env_pair('DREAMER_HIDDEN_DIST_DEADTIME_FRAC', 0.5, 1.5)
        self.tau_d_steps = np.maximum(
            rng.uniform(tau_lo, tau_hi, size=n_cv) * tau_steps, 1.0)
        self._alpha = np.clip(1.0 / self.tau_d_steps, 1e-3, 1.0)
        theta = np.round(rng.uniform(dt_lo, dt_hi, size=n_cv) * dead_steps)
        self.theta_d_steps = np.maximum(theta, 1.0).astype('int64')
        # Per-episode constant LOAD bias (operating-point shift), within cap.
        self.drift = (float(drift_frac) * self.amp_cap).astype('float64')
        # Initialise Gd at the drift steady state (the bias pre-exists the
        # episode — no artificial ramp-in at t=0).
        self._y = self.drift.copy()
        from collections import deque
        self._delay = [
            deque([float(self.drift[p])] * (int(self.theta_d_steps[p]) + 1),
                  maxlen=int(self.theta_d_steps[p]) + 1)
            for p in range(n_cv)
        ]

        n_budget = int(np.clip(round(T / (1.5 * settle)), 1, max(1, max_events)))
        # Spread mode (DEFAULT): distribute LOAD-event starts UNIFORMLY across
        # the whole episode (a realistic sequence of distinct upsets) instead of
        # the legacy front-loaded sequential placement.  Set
        # ``DREAMER_HIDDEN_DIST_SPREAD=0`` to restore the legacy placement.
        spread = str(os.environ.get('DREAMER_HIDDEN_DIST_SPREAD', '1')).strip() \
            .lower() not in ('0', 'false', 'no', 'off', '')
        events: List[Dict] = []
        if spread:
            earliest = float(rng.uniform(0.0, settle))
            latest = max(earliest + 1.0, 0.92 * T - tau_steps)
            min_gap = max(2.0, 0.6 * settle)
            starts: List[float] = []
            for _ in range(max(40, 12 * n_budget)):
                if len(starts) >= n_budget:
                    break
                s = float(rng.uniform(earliest, latest))
                if all(abs(s - s0) >= min_gap for s0 in starts):
                    starts.append(s)
            starts.sort()
            for t in starts:
                events.append(self._make_event(
                    rng, float(t), n_cv, shapes, weights, tau_steps,
                    settle, p_revert))
        else:
            t = float(rng.uniform(0.0, settle))
            while len(events) < n_budget and t < (T - tau_steps):
                ev = self._make_event(
                    rng, float(t), n_cv, shapes, weights, tau_steps,
                    settle, p_revert)
                events.append(ev)
                dur = ev['rise'] + ev['hold'] + ev['fall']
                if rng.uniform() < p_isolated:
                    gap = settle * (1.0 + float(rng.uniform(0.0, 1.0)))
                else:
                    gap = settle * float(rng.uniform(0.2, 0.8))
                t = t + dur + gap
        self.events = events

    def _make_event(self, rng: np.random.Generator, t: float, n_cv: int,
                     shapes: List[str], weights: np.ndarray, tau_steps: float,
                     settle: float, p_revert: float) -> Dict:
        """Draw one LOAD event (shape / magnitude / timing) starting at ``t``."""
        pos = int(rng.integers(0, n_cv))
        shape = str(rng.choice(shapes, p=weights))
        sign = 1.0 if rng.uniform() < 0.5 else -1.0
        mag = sign * float(rng.uniform(0.4, 1.0)) * float(self.amp_cap[pos])
        if shape == 'step':
            rise = 0.0
        elif shape == 'pulse':
            rise = float(rng.uniform(0.3, 0.8)) * tau_steps
        else:  # ramp
            rise = float(rng.uniform(0.5, 1.5)) * tau_steps
        if shape == 'pulse':
            hold = float(rng.uniform(0.5, 1.5)) * tau_steps
            revert = True
        else:  # step / ramp held toward steady state
            hold = float(rng.uniform(0.5, 2.0)) * settle
            revert = bool(rng.uniform() < p_revert)
        fall = (float(rng.uniform(0.5, 1.5)) * tau_steps) if revert else 0.0
        return {
            'pos': pos, 'shape': shape, 'mag': float(mag),
            'start': float(t), 'rise': float(rise), 'hold': float(hold),
            'fall': float(fall), 'revert': bool(revert),
        }

    def _shape_weights(self) -> np.ndarray:
        # step / ramp / pulse only.  Gd (dead-time + lag) provides the
        # smoothing, so a sharp ``step`` LOAD is fine — it produces a smooth
        # first-order CV response, not a CV discontinuity.  (The legacy
        # ``ou_drift`` shape — a per-step random walk that read as
        # high-frequency noise — is removed.)
        default = np.array([0.5, 0.3, 0.2], dtype='float64')
        raw = os.environ.get('DREAMER_HIDDEN_DIST_SHAPE_WEIGHTS', '').strip()
        if raw:
            try:
                w = np.array([float(x) for x in raw.split(',')], dtype='float64')
                if w.size == 3 and w.sum() > 0:
                    return w / w.sum()
            except Exception:
                pass
        return default

    def is_empty(self) -> bool:
        return len(self.cv_indices) == 0 or len(self.events) == 0

    def _load_value(self, ev: Dict, local: float) -> float:
        """Raw LOAD contributed by ``ev`` at local time ``local`` (>=0)."""
        mag = ev['mag']
        rise = ev['rise']
        hold = ev['hold']
        fall = ev['fall']
        if local < rise:
            return mag * (local / rise) if rise > 0 else mag
        if local < rise + hold:
            return mag
        if ev['revert']:
            f = local - (rise + hold)
            if f < fall:
                return mag * (1.0 - f / fall) if fall > 0 else 0.0
            return 0.0
        return mag  # permanent hold (non-reverting step/ramp)

    def step(self, state: np.ndarray) -> None:
        """Advance one step: build the raw LOAD, filter it through ``Gd``
        (dead-time + first-order lag), and add the resulting smooth ``d_cv`` to
        the CV channels of ``state`` (in place)."""
        n_cv = len(self.cv_indices)
        if n_cv == 0:
            return
        self._t += 1
        t = self._t
        # Raw unmeasured LOAD = constant bias + active events (the upset origin).
        load = self.drift.astype('float64').copy()
        for ev in self.events:
            local = t - ev['start']
            if local < 0:
                continue
            load[ev['pos']] += self._load_value(ev, float(local))
        # Gd per channel: dead-time delay, then first-order lag (unit DC gain).
        d_cv = np.zeros(n_cv, dtype='float64')
        for pos in range(n_cv):
            self._delay[pos].append(float(load[pos]))
            u_delayed = float(self._delay[pos][0])    # load theta_d steps ago
            self._y[pos] += self._alpha[pos] * (u_delayed - self._y[pos])
            d_cv[pos] = self._y[pos]
        self.last_applied = d_cv
        for pos, idx in enumerate(self.cv_indices):
            if d_cv[pos] != 0.0:
                state[int(idx)] = float(state[int(idx)]) + float(d_cv[pos])
        # If the sim publishes state in normalized [0,1] space, clip back in to
        # keep downstream consumers (encoder, reward) well-defined.
        if self._is_norm:
            for idx in self.cv_indices:
                state[int(idx)] = float(np.clip(state[int(idx)], 0.0, 1.0))

    def summary(self) -> Dict:
        return {
            'mode': 'load_through_Gd',
            'n_events': len(self.events),
            'cv_indices': list(self.cv_indices),
            'settle_steps': float(self._settle),
            'tau_steps': float(self._tau_steps),
            'tau_d_steps': self.tau_d_steps.tolist(),
            'theta_d_steps': self.theta_d_steps.tolist(),
            'drift': self.drift.tolist(),
            'events': [
                {k: ev[k] for k in ('pos', 'shape', 'mag', 'start',
                                     'rise', 'hold', 'fall', 'revert')}
                for ev in self.events
            ],
        }


def maybe_build_hidden_disturbance(
    rng: np.random.Generator,
    sim,
    *,
    tau_dom: float,
    sample_rate: float,
    prob: float,
    identifier_ctx: Optional[Dict] = None,
    amp_frac: float = 0.10,
    force: bool = False,
    progress: float = 0.0,
    phase: Optional[int] = None,
    dead_time: float = 0.0,
    episode_length: int = 0,
):
    """Bernoulli-toggle the per-episode hidden LOAD disturbance.

    Returns ``None`` for clean episodes so the WM also sees some clean
    data and can sharpen its base dynamics estimate.  Set ``force=True``
    (validation path) to bypass the Bernoulli toggle and always build.

    ``progress`` is the training progress in ``[0, 1]``; combined with
    the amplitude curriculum (see ``curriculum_amp_scale``) and the
    per-episode jitter / drift DR knobs to produce the effective
    ``amp_frac`` and ``drift_frac`` for this episode.  ``phase`` (optional)
    selects the phase-aware amp cap (P1/P2 vs P3).

    Single consolidated model (2026-06-10): a hidden unmeasured LOAD (the
    upset events) propagated to the CV through its OWN disturbance transfer
    function ``Gd`` (dead-time + first-order lag), adaptive from the identified
    plant — see ``HiddenDisturbance``.  Needs ``episode_length`` to schedule
    the load events.
    """
    if not hidden_disturbance_enabled(default=True):
        return None
    if int(episode_length) <= 0:
        # The consolidated model schedules LOAD events across the episode, so a
        # disturbance needs a positive horizon.  (Every live caller passes
        # cfg.episode_length > 0; this guards degenerate/unit-test inputs.)
        return None
    if not force:
        if prob <= 0.0:
            return None
        if rng.uniform() >= float(prob):
            return None
    # Effective amp_frac = nominal × curriculum scale × per-episode jitter.
    curr_scale = curriculum_amp_scale(float(progress), phase=phase)
    jitter = _sample_amp_jitter(rng)
    eff_amp_frac = float(amp_frac) * float(curr_scale) * float(jitter)
    drift_frac = _sample_drift_frac(rng)
    dist = HiddenDisturbance(
        rng=rng,
        sim=sim,
        tau_dom=float(tau_dom),
        sample_rate=float(sample_rate),
        dead_time=float(dead_time),
        episode_length=int(episode_length),
        amp_frac=eff_amp_frac,
        drift_frac=drift_frac,
        identifier_ctx=identifier_ctx,
    )
    if dist.is_empty():
        return None
    return dist
