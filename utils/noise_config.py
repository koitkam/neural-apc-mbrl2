"""Build noise and domain-randomization configuration from identified dynamics.

Derives all noise amplitudes, OU parameters, and domain-randomization ranges
from the system identification results (dynamics, lookback, channel metadata)
rather than hard-coding them inside each simulator.  This makes noise
configuration **simulator-independent** — any new simulator that exposes
the standard ``reset() / step()`` API can be used without changes.

Usage
-----
After dynamics identification the staged runner (or any caller) invokes
:func:`build_noise_config` to produce a JSON-serializable dict.  This dict
is either passed to :func:`create_sim` directly or propagated via the
``SIM_NOISE_CONFIG_JSON`` environment variable so that ``SimNoiseWrapper``
can pick it up automatically.

The config encodes:

* Per-channel **OU process noise** — temporally correlated drift sized to a
  fraction of the measured response amplitude.
* Per-channel **measurement (white) noise** — sized as a fraction of
  the channel span.
* **Domain-randomization percentage** — scaled to the relative uncertainty
  implied by the dynamics spread (multiple MVs, different taus / gains).
* **OU theta** (mean-reversion rate) derived from identified tau.

All amplitudes are expressed in **engineering units** so they are independent
of whether the simulator normalises its state vectors or not.
"""

import json
import os
from typing import Any, Dict, List, Optional

import numpy as np


# ── Defaults (overridable via TrainConfig / env) ─────────────────────────
# Canonical DREAMER_SIM_* land in run_plan via ENV_OVERRIDES.  Leftover
# SIM_* names still work when the DREAMER field is unset.  Dual-read at
# bake time (phase 1a, before TrainConfig exists).

_DEFAULT_OU_SIGMA_FRAC = 0.008       # OU sigma = fraction of channel span
_DEFAULT_OU_GAIN_CV = 0.15           # OU gain multiplier for CV channels
_DEFAULT_OU_GAIN_DV = 0.60           # OU gain multiplier for DV channels
_DEFAULT_MEAS_NOISE_CV_FRAC = 0.005  # measurement noise sigma = frac of CV span
_DEFAULT_MEAS_NOISE_DV_FRAC = 0.010  # measurement noise sigma = frac of DV span
_DEFAULT_NOISE_ADAPTIVE = True       # SNR-weighted OU gain + meas-sigma cap
_DEFAULT_DR_PCT = 0.10               # domain-randomization ±%
_DEFAULT_DR_MAX = 0.30               # upper cap for DR %

_BOOL_OFF = ('0', 'false', 'off', 'no', 'n', 'f', '')


def _snr_explicit(cfg, field: str) -> bool:
    if cfg is None:
        return False
    return field in (getattr(cfg, '_explicit_fields', set()) or set())


def _snr_float(cfg, field: str, dreamer_key: str, leftover_key: str,
               default: float) -> float:
    """Explicit TrainConfig, else DREAMER_*, else leftover SIM_*, else default."""
    if _snr_explicit(cfg, field):
        try:
            return float(getattr(cfg, field))
        except Exception:
            return float(default)
    d_raw = os.environ.get(dreamer_key)
    if d_raw not in (None, ''):
        try:
            return float(d_raw)
        except Exception:
            pass
    l_raw = os.environ.get(leftover_key)
    if l_raw not in (None, ''):
        try:
            return float(l_raw)
        except Exception:
            pass
    if cfg is not None:
        try:
            return float(getattr(cfg, field, default))
        except Exception:
            pass
    return float(default)


def _snr_bool(cfg, field: str, dreamer_key: str, leftover_key: str,
              default: bool) -> bool:
    """Same precedence as ``_snr_float``.  Leftover empty string is OFF."""
    if _snr_explicit(cfg, field):
        return bool(getattr(cfg, field))
    d_raw = os.environ.get(dreamer_key)
    if d_raw not in (None, ''):
        return str(d_raw).strip().lower() not in _BOOL_OFF
    if leftover_key in os.environ:
        return str(os.environ.get(leftover_key, '')).strip().lower() not in _BOOL_OFF
    if cfg is not None:
        return bool(getattr(cfg, field, default))
    return bool(default)


def resolve_sim_snr_knobs(cfg=None) -> Dict[str, Any]:
    """Unitless plant-SNR knobs used by ``build_noise_config``.

    Identity defaults match the historical ``os.environ.get`` fallbacks.
    ``SIM_NOISE_CONFIG_JSON`` stays env-only (baked path, not a knob).
    """
    return {
        'adaptive': _snr_bool(
            cfg, 'sim_noise_adaptive',
            'DREAMER_SIM_NOISE_ADAPTIVE', 'SIM_NOISE_ADAPTIVE',
            _DEFAULT_NOISE_ADAPTIVE),
        'ou_sigma_frac': _snr_float(
            cfg, 'sim_ou_sigma_frac',
            'DREAMER_SIM_OU_SIGMA_FRAC', 'SIM_OU_SIGMA_FRAC',
            _DEFAULT_OU_SIGMA_FRAC),
        'ou_gain_cv': _snr_float(
            cfg, 'sim_ou_gain_cv',
            'DREAMER_SIM_OU_GAIN_CV', 'SIM_OU_GAIN_CV',
            _DEFAULT_OU_GAIN_CV),
        'ou_gain_dv': _snr_float(
            cfg, 'sim_ou_gain_dv',
            'DREAMER_SIM_OU_GAIN_DV', 'SIM_OU_GAIN_DV',
            _DEFAULT_OU_GAIN_DV),
        'meas_cv_frac': _snr_float(
            cfg, 'sim_meas_noise_cv_frac',
            'DREAMER_SIM_MEAS_NOISE_CV_FRAC', 'SIM_MEAS_NOISE_CV_FRAC',
            _DEFAULT_MEAS_NOISE_CV_FRAC),
        'meas_dv_frac': _snr_float(
            cfg, 'sim_meas_noise_dv_frac',
            'DREAMER_SIM_MEAS_NOISE_DV_FRAC', 'SIM_MEAS_NOISE_DV_FRAC',
            _DEFAULT_MEAS_NOISE_DV_FRAC),
    }


_RUNTIME_BOOL_OFF = ('0', 'false', 'off', 'no', 'n', 'f')
_DEFAULT_NOISE_JITTER_PCT = 0.20
_DEFAULT_NOISE_ENABLED = True
_DEFAULT_DOMAIN_RANDOMIZATION = True


def _snr_str(cfg, field: str, dreamer_key: str, leftover_key: str,
             default: str = '') -> str:
    """Explicit TrainConfig, else DREAMER_*, else leftover, else default.

    Empty string = unset (unseeded RNG).  Explicit empty still wins.
    """
    if _snr_explicit(cfg, field):
        return str(getattr(cfg, field, default) or '')
    d_raw = os.environ.get(dreamer_key)
    if d_raw not in (None, ''):
        return str(d_raw).strip()
    l_raw = os.environ.get(leftover_key)
    if l_raw not in (None, ''):
        return str(l_raw).strip()
    if cfg is not None:
        return str(getattr(cfg, field, default) or '')
    return str(default or '')


def _snr_flag_empty_on(cfg, field: str, dreamer_key: str, leftover_keys,
                       default: bool) -> bool:
    """Bool with DomainRandomizer / ``SIM_NOISE_ENABLED`` identity.

    Unset → default.  Empty leftover value is ON (not in the off-set).
    ``leftover_keys`` is a string or a sequence (first set leftover wins).
    """
    if _snr_explicit(cfg, field):
        return bool(getattr(cfg, field))
    d_raw = os.environ.get(dreamer_key)
    if d_raw not in (None, ''):
        return str(d_raw).strip().lower() not in _RUNTIME_BOOL_OFF
    if isinstance(leftover_keys, str):
        leftover_keys = (leftover_keys,)
    for leftover_key in leftover_keys:
        if leftover_key not in os.environ:
            continue
        return str(os.environ.get(leftover_key, '')).strip().lower() not in _RUNTIME_BOOL_OFF
    if cfg is not None:
        return bool(getattr(cfg, field, default))
    return bool(default)


def _snr_float_multi(cfg, field: str, dreamer_key: str, leftover_keys,
                     default: float) -> float:
    """Like ``_snr_float`` with several leftover names (first non-empty wins)."""
    if _snr_explicit(cfg, field):
        try:
            return float(getattr(cfg, field))
        except Exception:
            return float(default)
    d_raw = os.environ.get(dreamer_key)
    if d_raw not in (None, ''):
        try:
            return float(d_raw)
        except Exception:
            pass
    if isinstance(leftover_keys, str):
        leftover_keys = (leftover_keys,)
    for leftover_key in leftover_keys:
        l_raw = os.environ.get(leftover_key)
        if l_raw not in (None, ''):
            try:
                return float(l_raw)
            except Exception:
                pass
    if cfg is not None:
        try:
            return float(getattr(cfg, field, default))
        except Exception:
            pass
    return float(default)


def resolve_sim_runtime_knobs(cfg=None) -> Dict[str, Any]:
    """Runtime wrapper knobs (seed / jitter / enable / DR).

    Identity defaults match ``SimNoiseWrapper`` / ``DomainRandomizer``.
    ``SIM_NOISE_AMPLITUDE_JITTER_PCT`` is the dead name ``clean_mode`` still
    writes — dual-read so SysID actually zeros jitter.
    Leftover ``DREAMER_DOMAIN_RANDOMIZATION`` (comments only; never read)
    is an alias of ``SIM_DOMAIN_RANDOMIZATION``.
    """
    jitter = _snr_float_multi(
        cfg, 'sim_noise_jitter_pct',
        'DREAMER_SIM_NOISE_JITTER_PCT',
        ('SIM_NOISE_JITTER_PCT', 'SIM_NOISE_AMPLITUDE_JITTER_PCT'),
        _DEFAULT_NOISE_JITTER_PCT)
    return {
        'noise_enabled': _snr_flag_empty_on(
            cfg, 'sim_noise_enabled',
            'DREAMER_SIM_NOISE_ENABLED', 'SIM_NOISE_ENABLED',
            _DEFAULT_NOISE_ENABLED),
        'noise_seed': _snr_str(
            cfg, 'sim_noise_seed',
            'DREAMER_SIM_NOISE_SEED', 'SIM_NOISE_SEED', ''),
        'jitter_pct': float(np.clip(float(jitter), 0.0, 0.5)),
        'domain_randomization': _snr_flag_empty_on(
            cfg, 'sim_domain_randomization',
            'DREAMER_SIM_DOMAIN_RANDOMIZATION',
            ('DREAMER_DOMAIN_RANDOMIZATION', 'SIM_DOMAIN_RANDOMIZATION'),
            _DEFAULT_DOMAIN_RANDOMIZATION),
        'domain_randomization_seed': _snr_str(
            cfg, 'sim_domain_randomization_seed',
            'DREAMER_SIM_DOMAIN_RANDOMIZATION_SEED',
            'SIM_DOMAIN_RANDOMIZATION_SEED', ''),
    }


def _span(bounds):
    lo, hi = float(bounds[0]), float(bounds[1])
    return max(1e-6, abs(hi - lo))


def noise_curriculum_scale(progress: float,
                            phase: Optional[int] = None,
                            cfg=None) -> float:
    """P1 process+measurement noise amplitude curriculum (P89, 2026-06-06).

    Returns a multiplier in ``[0, 1]`` applied to BOTH the OU process noise
    and the white measurement noise (via ``SimNoiseWrapper.set_noise_scale``)
    so the world model can learn the clean base dynamics + a held-action
    fixed point first, then face progressively realistic noise.

    Rationale (P89 RCA): the WM never converged under a held action because
    its training data never contained a clean settled trajectory \u2014 process
    OU (≈1.3 % span, ~133-step correlation) + measurement noise are on in
    100 % of episodes, so the plant never sits still.  Ramping noise from ~0
    lets P1 establish the attractor before noise is added.

    Schedule: TrainConfig ``process_noise_amp_ramp`` / leftover
    ``DREAMER_PROCESS_NOISE_AMP_RAMP="<start>:<reach>"`` (default
    ``"0.0:0.4"`` \u2014 start fully clean, reach full noise by 40 % progress).
    ``start``: scale at ``progress=0``.  ``reach``: progress fraction at which
    the scale reaches 1.0.  Phase-aware: **P3 always returns 1.0** (the WM is
    frozen and the actor must learn to reject realistic-magnitude noise);
    P1/P2 follow the ramp.  Disable the curriculum entirely (full noise from
    step 0, legacy behaviour) with ramp ``"1.0:1e-6"`` or by setting
    ``process_noise_curriculum=False`` on the cfg.

    Dual-read: explicit TrainConfig wins; leftover env wins if not explicit;
    else cfg default.  Unset → ``0.0:0.4``.  Empty leftover env → 1.0
    (full noise, same as the historical ``os.environ.get`` empty-string path).
    """
    if phase is not None and int(phase) >= 3:
        return 1.0
    from utils.hidden_disturbance import _knob_raw
    raw, leftover = _knob_raw(cfg, 'process_noise_amp_ramp',
                              'DREAMER_PROCESS_NOISE_AMP_RAMP')
    if leftover and (raw is None or str(raw).strip() == ''):
        return 1.0
    text = '0.0:0.4' if raw is None else str(raw).strip()
    if not text:
        return 1.0
    try:
        start_str, reach_str = text.split(':')
        start = float(start_str)
        reach = float(reach_str)
    except Exception:
        return 1.0
    p = float(np.clip(progress, 0.0, 1.0))
    start = float(np.clip(start, 0.0, 1.0))
    reach = float(np.clip(reach, 1e-6, 1.0))
    if p >= reach:
        return 1.0
    return float(min(1.0, start + (1.0 - start) * (p / reach)))


def _theta_from_tau(tau_dominant: float, sample_rate: int = 1) -> float:
    """Derive OU mean-reversion rate from the dominant time constant.

    A faster process (small tau) should have faster-reverting noise so it
    doesn't mask the real dynamics.  A slower process tolerates longer
    drift.  We target ~10 % of tau as the OU characteristic time.

    The returned ``theta`` is consumed inside the OU update together with
    a hard-coded ``dt = 0.01`` (see ``SimNoiseWrapper``), so the effective
    per-step contraction is ``theta * dt`` which is always well below 1
    even for large ``theta``.  Hence no upper clip is applied — processes
    with small ``tau`` (e.g. soft-sensor with tau ≈ 11 s at 5 s sample
    rate) are allowed to use the fast-decorrelation ratio the formula
    returns.  Only a small positive floor is kept for numerical safety
    (``theta = 0`` would turn the OU into a random walk).
    """
    tau = max(1.0, float(tau_dominant))
    characteristic = 0.10 * tau
    dt = max(1e-3, float(sample_rate))
    ratio = dt / characteristic
    return float(max(ratio, 0.02))


def _ou_sigma_for_channel(
    span: float,
    response_amplitude: float,
    base_frac: float,
) -> float:
    """Size OU volatility relative to the channel's observed response.

    If the identification measured a small response amplitude, noise should
    be proportionally smaller so it stays below the signal.  We take the
    geometric mean of ``base_frac * span`` and ``0.3 * |amplitude|`` to
    balance physical range with observed sensitivity.
    """
    from_span = base_frac * span
    from_amp = 0.30 * abs(response_amplitude) if abs(response_amplitude) > 1e-8 else from_span
    return float(max(1e-6, (from_span * from_amp) ** 0.5))


def _domain_randomization_pct(
    per_mv_dynamics: Dict,
    per_cv_dynamics: Dict,
) -> float:
    """Derive domain-randomization ± % from dynamics spread.

    If identified dynamics show large spread across MVs / CVs the process
    has more structural uncertainty and warrants stronger randomization.
    """
    taus = []
    for d in list(per_mv_dynamics.values()) + list(per_cv_dynamics.values()):
        if isinstance(d, dict) and 'tau' in d:
            taus.append(float(d['tau']))
    if len(taus) < 2:
        return _DEFAULT_DR_PCT
    tau_arr = np.array(taus)
    cv_tau = float(np.std(tau_arr) / max(1e-6, np.mean(tau_arr)))
    pct = float(np.clip(
        _DEFAULT_DR_PCT + 0.5 * cv_tau,
        _DEFAULT_DR_PCT,
        _DEFAULT_DR_MAX,
    ))
    return round(pct, 3)


# ── Public API ────────────────────────────────────────────────────────────

def build_noise_config(
    dynamics_json: Optional[Dict] = None,
    lookback_json: Optional[Dict] = None,
    state_variables: Optional[List[str]] = None,
    cv_indices: Optional[List[int]] = None,
    dv_indices: Optional[List[int]] = None,
    mv_indices: Optional[List[int]] = None,
    state_normalization_ranges: Optional[List[List[float]]] = None,
    cv_normalization_ranges: Optional[List[List[float]]] = None,
    dv_normalization_ranges: Optional[List[List[float]]] = None,
    sample_rate: int = 1,
    noise_stdv: float = 0.03,
    cfg=None,
) -> Dict[str, Any]:
    """Build a complete noise + domain-randomization config from dynamics.

    Parameters
    ----------
    dynamics_json : dict, optional
        Parsed output of ``control_dynamics_identifier``.
    lookback_json : dict, optional
        Parsed output of ``control_lookback_identifier``.
    state_variables : list of str
        Channel names in state-vector order.
    cv_indices, dv_indices, mv_indices : list of int
        Standard index lists from ``resolve_sim_metadata``.
    state_normalization_ranges : list of [lo, hi]
        Engineering-unit bounds per state channel.
    cv_normalization_ranges, dv_normalization_ranges : list of [lo, hi]
        Engineering-unit bounds per CV / DV channel.
    sample_rate : int
        Simulator sample rate (seconds).
    noise_stdv : float
        Baseline noise level requested by the caller.
    cfg : TrainConfig, optional
        When present, explicit SNR fields win over env.  Phase 1a bake
        passes ``None`` and dual-reads ``DREAMER_SIM_*`` / leftover
        ``SIM_*``.

    Returns
    -------
    dict
        JSON-serializable config with keys ``ou_noise``,
        ``measurement_noise``, ``domain_randomization``, and ``metadata``.
    """
    dyn = dynamics_json or {}
    lb = lookback_json or {}
    state_vars = list(state_variables or [])
    cv_idx = list(cv_indices or [])
    dv_idx = list(dv_indices or [])
    mv_idx = list(mv_indices or [])
    state_ranges = list(state_normalization_ranges or [])
    cv_ranges = list(cv_normalization_ranges or [])
    dv_ranges = list(dv_normalization_ranges or [])

    tau_dom = float(dyn.get('tau_dominant_identified', 50.0) or 50.0)
    dead_time = float(dyn.get('dead_time_identified', 5.0) or 5.0)
    per_mv = dyn.get('per_mv_dynamics', {}) or {}
    per_cv = dyn.get('per_cv_dynamics', {}) or {}
    per_pair = dyn.get('per_pair_estimates', []) or []

    theta = _theta_from_tau(tau_dom, sample_rate=sample_rate)
    snr = resolve_sim_snr_knobs(cfg)
    runtime = resolve_sim_runtime_knobs(cfg)
    base_sigma_frac = float(snr['ou_sigma_frac'])

    # ── Build per-CV response amplitude map ──────────────────────────────
    cv_amp: Dict[int, float] = {}
    for row in per_pair:
        if not row.get('valid', False):
            continue
        amp = abs(float(row.get('amplitude', 0.0)))
        if amp < 1e-9:
            continue
        cv_name = str(row.get('cv', ''))
        for i, vi in enumerate(cv_idx):
            if vi < len(state_vars) and state_vars[vi] == cv_name:
                cv_amp[vi] = max(cv_amp.get(vi, 0.0), amp)

    # ── per-DV response amplitude map ────────────────────────────────────
    dv_amp: Dict[int, float] = {}
    for row in per_pair:
        if not row.get('valid', False):
            continue
        if row.get('input_type') != 'dv':
            continue
        dv_name = str(row.get('input', '') or row.get('mv', ''))
        amp = abs(float(row.get('delta', 0.0)))
        if amp < 1e-9:
            continue
        for i, di in enumerate(dv_idx):
            if di < len(state_vars) and state_vars[di] == dv_name:
                dv_amp[di] = max(dv_amp.get(di, 0.0), amp)

    # ── OU noise entries ─────────────────────────────────────────────────
    # Adaptive SNR weighting is env-free ON (TrainConfig
    # ``sim_noise_adaptive``).  Leftover ``SIM_NOISE_ADAPTIVE=0`` still
    # disables it.  Comment that claimed default-OFF was stale — the
    # historical ``os.environ.get(..., '1')`` path was already ON.
    adaptive = bool(snr['adaptive'])
    base_gain_cv = float(snr['ou_gain_cv'])
    base_gain_dv = float(snr['ou_gain_dv'])

    def _snr_gain(base_gain: float, amp: float, meas_sigma: float) -> float:
        if not adaptive:
            return base_gain
        # Scale gain by identified signal-to-noise: low-SNR channels get
        # reduced process noise so the signal is not masked; high-SNR
        # channels keep the full default gain.
        if meas_sigma <= 0 or amp <= 0:
            return base_gain
        snr = float(amp) / float(meas_sigma)
        scale = float(np.clip(snr / 5.0, 0.3, 1.5))
        return float(base_gain * scale)

    ou_noise: List[Dict] = []

    # CV channels
    for i, idx in enumerate(cv_idx):
        bounds = cv_ranges[i] if i < len(cv_ranges) else (
            state_ranges[idx] if idx < len(state_ranges) else [0.0, 100.0]
        )
        span = _span(bounds)
        amp = cv_amp.get(idx, 0.05 * span)
        sigma = _ou_sigma_for_channel(span, amp, base_sigma_frac)
        sigma = float(max(sigma, noise_stdv * 0.3))
        # Rough measurement-sigma estimate used only for SNR scaling below
        cv_meas_frac_tmp = float(snr['meas_cv_frac'])
        meas_sigma_est = max(cv_meas_frac_tmp * span, noise_stdv * 0.6)
        gain = _snr_gain(base_gain_cv, amp, meas_sigma_est)
        ou_noise.append({
            'index': int(idx),
            'sigma': round(float(sigma), 6),
            'gain': round(float(gain), 4),
            'bounds': [float(bounds[0]), float(bounds[1])],
            'theta': round(theta, 4),
            'dt': 0.01,
            'channel_name': state_vars[idx] if idx < len(state_vars) else f'cv_{i}',
            'channel_type': 'cv',
        })

    # DV channels
    for i, idx in enumerate(dv_idx):
        bounds = dv_ranges[i] if i < len(dv_ranges) else (
            state_ranges[idx] if idx < len(state_ranges) else [0.0, 100.0]
        )
        span = _span(bounds)
        amp = dv_amp.get(idx, 0.10 * span)
        sigma = _ou_sigma_for_channel(span, amp, base_sigma_frac * 1.2)
        sigma = float(max(sigma, noise_stdv * 0.5))
        dv_meas_frac_tmp = float(snr['meas_dv_frac'])
        meas_sigma_est = max(dv_meas_frac_tmp * span, noise_stdv * 1.0)
        gain = _snr_gain(base_gain_dv, amp, meas_sigma_est)
        ou_noise.append({
            'index': int(idx),
            'sigma': round(float(sigma), 6),
            'gain': round(float(gain), 4),
            'bounds': [float(bounds[0]), float(bounds[1])],
            'theta': round(theta, 4),
            'dt': 0.01,
            'channel_name': state_vars[idx] if idx < len(state_vars) else f'dv_{i}',
            'channel_type': 'dv',
        })

    # ── Measurement noise entries ────────────────────────────────────────
    meas_noise: List[Dict] = []

    cv_meas_frac = float(snr['meas_cv_frac'])
    dv_meas_frac = float(snr['meas_dv_frac'])

    for i, idx in enumerate(cv_idx):
        bounds = cv_ranges[i] if i < len(cv_ranges) else (
            state_ranges[idx] if idx < len(state_ranges) else [0.0, 100.0]
        )
        span = _span(bounds)
        sigma = float(max(cv_meas_frac * span, noise_stdv * 0.6))
        if adaptive:
            amp = cv_amp.get(idx, 0.0)
            if amp > 1e-6:
                sigma = float(min(sigma, 0.25 * amp))
        meas_noise.append({
            'index': int(idx),
            'sigma': round(float(sigma), 6),
            'bounds': [float(bounds[0]), float(bounds[1])],
            'channel_name': state_vars[idx] if idx < len(state_vars) else f'cv_{i}',
            'channel_type': 'cv',
        })

    for i, idx in enumerate(dv_idx):
        bounds = dv_ranges[i] if i < len(dv_ranges) else (
            state_ranges[idx] if idx < len(state_ranges) else [0.0, 100.0]
        )
        span = _span(bounds)
        sigma = float(max(dv_meas_frac * span, noise_stdv * 1.0))
        if adaptive:
            amp = dv_amp.get(idx, 0.0)
            if amp > 1e-6:
                sigma = float(min(sigma, 0.25 * amp))
        meas_noise.append({
            'index': int(idx),
            'sigma': round(float(sigma), 6),
            'bounds': [float(bounds[0]), float(bounds[1])],
            'channel_name': state_vars[idx] if idx < len(state_vars) else f'dv_{i}',
            'channel_type': 'dv',
        })

    # ── Domain randomization ─────────────────────────────────────────────
    dr_pct = _domain_randomization_pct(per_mv, per_cv)
    dr_on = bool(runtime['domain_randomization'])

    config = {
        'ou_noise': ou_noise,
        'measurement_noise': meas_noise,
        'domain_randomization': {
            'enabled': dr_on,
            'param_randomization_pct': round(dr_pct, 4),
        },
        'metadata': {
            'tau_dominant': round(tau_dom, 3),
            'dead_time': round(dead_time, 3),
            'theta_derived': round(theta, 4),
            'base_sigma_frac': round(base_sigma_frac, 6),
            'sim_noise_adaptive': bool(adaptive),
            'sim_ou_gain_cv': round(base_gain_cv, 6),
            'sim_ou_gain_dv': round(base_gain_dv, 6),
            'sim_meas_noise_cv_frac': round(cv_meas_frac, 6),
            'sim_meas_noise_dv_frac': round(dv_meas_frac, 6),
            'sim_noise_enabled': bool(runtime['noise_enabled']),
            'sim_noise_jitter_pct': round(float(runtime['jitter_pct']), 6),
            'sim_domain_randomization': dr_on,
            'noise_stdv_input': round(noise_stdv, 6),
            'sample_rate': int(sample_rate),
            'cv_channels': len(cv_idx),
            'dv_channels': len(dv_idx),
            'mv_channels': len(mv_idx),
            'source': 'control_noise_config.build_noise_config',
        },
    }
    return config


def save_noise_config(config: Dict, path: str) -> str:
    """Write noise config to JSON and set ``SIM_NOISE_CONFIG_JSON`` env var."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2)
    os.environ['SIM_NOISE_CONFIG_JSON'] = os.path.abspath(path)
    return os.path.abspath(path)


def load_noise_config(path: Optional[str] = None) -> Optional[Dict]:
    """Load noise config from a file path or ``SIM_NOISE_CONFIG_JSON`` env.

    Returns *None* if no config is available (fall back to sim-provided).
    """
    p = path or os.environ.get('SIM_NOISE_CONFIG_JSON', '').strip()
    if not p or not os.path.isfile(p):
        return None
    try:
        with open(p, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return None


def build_noise_config_from_sim(sim, dynamics_json=None, lookback_json=None,
                                noise_stdv: float = 0.03, cfg=None,
                                ) -> Dict[str, Any]:
    """Convenience: build noise config using metadata already on a sim.

    This is the bridge for callers that have a simulator instance with
    attached metadata but want dynamics-driven noise instead of the
    sim's built-in ``noise_config``.
    """
    return build_noise_config(
        dynamics_json=dynamics_json,
        lookback_json=lookback_json,
        state_variables=list(getattr(sim, 'state_variables', [])),
        cv_indices=list(getattr(sim, 'cv_indices', [])),
        dv_indices=list(getattr(sim, 'dv_indices', [])),
        mv_indices=list(getattr(sim, 'mv_indices', [])),
        state_normalization_ranges=list(getattr(sim, 'state_normalization_ranges', [])),
        cv_normalization_ranges=list(getattr(sim, 'cv_normalization_ranges', [])),
        dv_normalization_ranges=list(getattr(sim, 'dv_normalization_ranges', [])),
        sample_rate=int(getattr(sim, 'sample_rate', 1)),
        noise_stdv=noise_stdv,
        cfg=cfg,
    )
