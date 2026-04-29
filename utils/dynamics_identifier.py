"""Identify process time constants and dead time via simulator step tests.

Functionality:
- Runs open-loop MV and DV step experiments on the configured simulator.
- Estimates per MV/CV FOPDT-like dead time and time constant.
- Aggregates robust dominant values for physics-informed lookback selection.
- Adaptive pre-settle: when pre_steps<=0, a probe run auto-detects how
  long the simulator needs to reach steady state from reset, making the
  identifier robust to any system dynamics.
- Per-pair sanity guards reject estimates where the response is clipped by
  the experiment window or the time constant is implausibly small.
- Robust median aggregation reduces sensitivity to outlier experiments.

Inputs:
- CLI args: `--output`, `--pre-steps`, `--post-steps`, `--step-size`,
  `--repeats`, `--noise-stdv`.
- Simulator source and metadata from `control_sim_factory`.

Outputs:
- JSON file with dominant tau/dead-time estimates and all pairwise results.

Main steps:
1) Probe simulator metadata to get MV/CV channels.
2) Execute repeated positive/negative MV and DV step tests.
3) Smooth CV trajectories and estimate FOPDT parameters.
4) Save per-pair and aggregated dynamics identification summary.
"""

import argparse
import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from utils.objective_config import load_objective_spec
from utils.sim_factory import create_sim, resolve_sim_metadata, require_sim_metadata


def _moving_average(x: np.ndarray, window: int = 5) -> np.ndarray:
    if window <= 1 or x.size < window:
        return x.copy()
    # Edge-replicated padding: np.convolve(mode='same') pads with ZEROS,
    # which corrupts the DC value at the start and end of the signal and
    # produced heavily asymmetric apparent step amplitudes in FOPDT fits.
    # Replicating the edge values keeps the moving average accurate at
    # both ends.
    pad = window // 2
    xpad = np.pad(np.asarray(x, dtype=np.float64), pad, mode='edge')
    kernel = np.ones(window, dtype=np.float64) / float(window)
    out = np.convolve(xpad, kernel, mode='valid')
    # 'valid' length = len(xpad) - window + 1 = len(x) + 2*pad - window + 1;
    # for odd window this equals len(x); for even window it's len(x)+1 so
    # trim the final sample to preserve input length.
    if out.size != x.size:
        out = out[: x.size]
    return out.astype(np.float64)


def _estimate_fopdt(y: np.ndarray, step_index: int, deadband_frac: float = 0.05) -> Dict:
    y = np.asarray(y, dtype=np.float64)
    y0 = float(np.mean(y[max(0, step_index - 40):step_index]))
    yf = float(np.mean(y[-60:]))
    amp = yf - y0

    if abs(amp) < 1e-6:
        return {
            'valid': False,
            'y0': y0,
            'yf': yf,
            'amplitude': amp,
            'dead_time': None,
            'tau': None,
        }

    sign = 1.0 if amp >= 0 else -1.0
    threshold = y0 + sign * max(deadband_frac * abs(amp), 1e-4)
    tau_target = y0 + sign * (0.632 * abs(amp))

    post = y[step_index:]
    dead_idx = None
    tau_idx = None

    for i, val in enumerate(post):
        if dead_idx is None and ((val >= threshold) if sign > 0 else (val <= threshold)):
            dead_idx = i
        if tau_idx is None and ((val >= tau_target) if sign > 0 else (val <= tau_target)):
            tau_idx = i
        if dead_idx is not None and tau_idx is not None:
            break

    if dead_idx is None or tau_idx is None or tau_idx < dead_idx:
        return {
            'valid': False,
            'y0': y0,
            'yf': yf,
            'amplitude': amp,
            'dead_time': None,
            'tau': None,
        }

    dead_time = float(dead_idx)
    tau = float(tau_idx - dead_idx)
    return {
        'valid': True,
        'y0': y0,
        'yf': yf,
        'amplitude': amp,
        'dead_time': dead_time,
        'tau': tau,
    }


def _get_cv_indices(meta: Dict[str, Any]) -> List[int]:
    cv_idxs = [int(x) for x in meta.get('cv_indices', []) if x is not None]

    # Backward-compatible fallback for simulators exposing named PV indices only.
    if not cv_idxs:
        cv_idxs = []
    for key in ['top_pv_index', 'bottom_pv_index', 'pressure_pv_index']:
        if cv_idxs:
            break
        v = meta.get(key)
        if v is not None and int(v) not in cv_idxs:
            cv_idxs.append(int(v))

    if not cv_idxs:
        env_cv = os.environ.get('SIM_CV_INDICES_JSON', '')
        if env_cv:
            try:
                cv_idxs = [int(x) for x in json.loads(env_cv)]
            except Exception:
                cv_idxs = []

    if not cv_idxs:
        raise ValueError('No CV indices available. Set SIM_CV_INDICES_JSON or provide simulator CV index attributes.')
    return cv_idxs


def _resolve_mv_bounds(action_dim: int) -> List[List[float]]:
    try:
        spec = load_objective_spec()
        bounds = spec.get('bounds', {}) if isinstance(spec.get('bounds', {}), dict) else {}
        mv_bounds = list(bounds.get('mv_bounds', []))
    except Exception:
        mv_bounds = []

    if len(mv_bounds) < action_dim:
        mv_bounds = mv_bounds + [[0.0, 100.0] for _ in range(action_dim - len(mv_bounds))]

    out = []
    for i in range(action_dim):
        try:
            lo = float(mv_bounds[i][0])
            hi = float(mv_bounds[i][1])
        except Exception:
            lo, hi = 0.0, 100.0
        if hi <= lo:
            hi = lo + 1.0
        out.append([lo, hi])
    return out


def _resolve_dv_bounds(meta: Dict[str, Any], dv_count: int) -> List[List[float]]:
    raw = list(meta.get('dv_normalization_ranges', []))
    out = []
    for i in range(dv_count):
        if i < len(raw) and isinstance(raw[i], (list, tuple)) and len(raw[i]) >= 2:
            lo = float(raw[i][0])
            hi = float(raw[i][1])
            if hi <= lo:
                hi = lo + 1.0
            out.append([lo, hi])
        else:
            out.append([0.0, 100.0])
    return out


def _state_value_to_engineering(state: np.ndarray, state_index: int, meta: Dict[str, Any]) -> float:
    raw = float(state[state_index])
    if not bool(meta.get('state_is_normalized', False)):
        return raw

    mv_idxs = [int(x) for x in meta.get('mv_indices', [])]
    cv_idxs = [int(x) for x in _get_cv_indices(meta)]
    dv_idxs = [int(x) for x in meta.get('dv_indices', [])]

    if state_index in mv_idxs:
        j = mv_idxs.index(state_index)
        ranges = list(meta.get('mv_normalization_ranges', []))
    elif state_index in cv_idxs:
        j = cv_idxs.index(state_index)
        ranges = list(meta.get('cv_normalization_ranges', []))
    elif state_index in dv_idxs:
        j = dv_idxs.index(state_index)
        ranges = list(meta.get('dv_normalization_ranges', []))
    else:
        return raw

    if 0 <= j < len(ranges) and isinstance(ranges[j], (list, tuple)) and len(ranges[j]) >= 2:
        lo = float(ranges[j][0])
        hi = float(ranges[j][1])
        if hi <= lo:
            hi = lo + 1.0
        return float(lo + raw * (hi - lo))
    return raw


def _effective_delta(span: float, requested_step: float, noise_stdv: float) -> float:
    base = abs(float(requested_step))
    # Keep the perturbation large enough to dominate noise, sensor chatter,
    # and any residual non-steadiness from the pre-settle phase. We want a
    # clearly supra-noise excursion so FOPDT fitting converges cleanly.
    # 15% span gives ~2 orders of magnitude headroom over typical sensor
    # noise (~0.01 span) and is still well within bounds (<<100% span).
    min_from_range = 0.15 * max(1e-6, float(span))
    min_from_noise = 4.0 * abs(float(noise_stdv)) * max(1e-6, float(span))
    return float(max(base, min_from_range, min_from_noise, 1e-3))


def _detect_settle_time(noise_stdv: float = 0.0,
                        max_budget: int = 2000,
                        window: int = 30) -> int:
    """Detect how many steps the simulator needs to reach steady state from reset.

    Runs a single episode with constant inputs and monitors when CV drift
    between consecutive windows becomes negligible.  Returns the step count
    at which all CVs have converged.  Works for *any* simulator — no
    hard-coded knowledge of system dynamics.
    """
    sim = create_sim(episode_length=max_budget + 10, sample_rate=1, noise_stdv=noise_stdv)
    meta = resolve_sim_metadata(sim)
    cv_idxs = _get_cv_indices(meta)
    mv_idxs = [int(x) for x in meta.get('mv_indices', []) if x is not None]

    state, _done = sim.reset()
    u = np.array([_state_value_to_engineering(state, i, meta) for i in mv_idxs],
                 dtype=np.float32)

    cv_history: List[List[float]] = []
    for _ in range(max_budget):
        state, done = sim.step(u)
        cv_history.append([_state_value_to_engineering(state, i, meta) for i in cv_idxs])
        if done:
            break

    cv_arr = np.asarray(cv_history, dtype=np.float64)
    n_steps = cv_arr.shape[0]

    if n_steps < 2 * window:
        return n_steps

    settle_times: List[int] = []
    for cv_i in range(cv_arr.shape[1]):
        series = cv_arr[:, cv_i]
        overall_range = float(np.max(series) - np.min(series))
        if overall_range < 1e-10:
            settle_times.append(0)
            continue

        # Settled when drift between consecutive windows < 1% of overall range.
        drift_thr = 0.01 * overall_range
        settled_at = n_steps
        for start in range(0, n_steps - 2 * window + 1, window):
            mean_a = float(np.mean(series[start:start + window]))
            mean_b = float(np.mean(series[start + window:start + 2 * window]))
            if abs(mean_b - mean_a) < drift_thr:
                settled_at = start + 2 * window
                break
        settle_times.append(settled_at)

    detected = max(settle_times) if settle_times else n_steps
    print(f"[SETTLE-DETECT] settle_time={detected} steps  (budget={max_budget}, window={window})")
    return detected


def _safe_state_name(meta: Dict[str, Any], state_index: int, prefix: str) -> str:
    state_names = meta.get('state_variables') or []
    if 0 <= int(state_index) < len(state_names):
        return str(state_names[int(state_index)])
    return f'{prefix}_{int(state_index)}'


def _apply_dv_perturbation(sim, state_index: int, value: float, state_name: str,
                           dv_pos: int = -1, offset_delta: Optional[float] = None) -> bool:
    # Preferred path: use the sim's native disturbance-offset interface
    # (DisturbanceOffsetMixin.set_disturbance_offset).  This drives the
    # simulator's own rest point (e.g. a first-order lag's target value)
    # instead of overwriting the state array every step, which yields a
    # clean, symmetric step response. Requires the caller to pass the
    # position within the DV group and the offset from the channel's
    # natural baseline.
    if dv_pos >= 0 and offset_delta is not None:
        fn = getattr(sim, 'set_disturbance_offset', None)
        if callable(fn):
            try:
                fn('dv', int(dv_pos), float(offset_delta))
                return True
            except Exception:
                pass

    # Generic interface priority: explicit simulator hook, env attr map, then common in-memory state containers.
    for method_name in ('set_disturbance_by_state_index', 'set_state_by_index'):
        fn = getattr(sim, method_name, None)
        if callable(fn):
            try:
                fn(int(state_index), float(value))
                return True
            except Exception:
                pass

    raw_map = os.environ.get('SIM_DV_PERTURB_ATTR_MAP_JSON', '').strip()
    if raw_map:
        try:
            attr_map = json.loads(raw_map)
            if isinstance(attr_map, dict):
                attr_name = attr_map.get(state_name, '')
                if attr_name and hasattr(sim, str(attr_name)):
                    setattr(sim, str(attr_name), float(value))
                    return True
        except Exception:
            pass

    if hasattr(sim, 'episode_array') and hasattr(sim, 'episode_counter'):
        try:
            row = int(sim.episode_counter)
            sim.episode_array[row, int(state_index)] = float(value)
            return True
        except Exception:
            pass

    for attr_name in ('state', 'state_vector', 'current_state'):
        if hasattr(sim, attr_name):
            try:
                arr = getattr(sim, attr_name)
                arr[int(state_index)] = float(value)
                return True
            except Exception:
                pass

    return False


def _single_step_experiment(
    input_type: str,
    input_index: int,
    delta: float,
    pre_steps: int,
    post_steps: int,
    noise_stdv: float,
    mv_bounds: List[List[float]],
    dv_bounds: List[List[float]],
) -> Dict:
    total_steps = pre_steps + post_steps + 1
    sim = create_sim(episode_length=total_steps + 5, sample_rate=1, noise_stdv=noise_stdv)
    meta = resolve_sim_metadata(sim)
    require_sim_metadata(sim, ['mv_indices'])

    mv_idxs = list(meta['mv_indices'])
    cv_idxs = _get_cv_indices(meta)
    dv_idxs = [int(x) for x in meta.get('dv_indices', []) if x is not None]

    state, done = sim.reset()

    # Current workflow is configured around engineering-unit channels.
    # For normalized-state simulators we denormalize channel values using metadata ranges.
    u0 = np.array([_state_value_to_engineering(state, i, meta) for i in mv_idxs], dtype=np.float32)
    u = u0.copy()

    target_dv = None
    dv_state_index = None
    dv_name = None
    dv_apply_ok = True

    if input_type == 'mv':
        lo, hi = mv_bounds[input_index]
        base = float(u[input_index])
        u[input_index] = float(np.clip(base + delta, lo, hi))
        applied_delta = float(u[input_index] - base)
        input_name = _safe_state_name(meta, mv_idxs[input_index], 'MV')
    else:
        if input_index >= len(dv_idxs):
            raise ValueError(f'DV index {input_index} out of bounds for simulator dv_indices={dv_idxs}')
        dv_state_index = int(dv_idxs[input_index])
        dv_name = _safe_state_name(meta, dv_state_index, 'DV')
        dv_base = float(_state_value_to_engineering(state, dv_state_index, meta))
        lo, hi = dv_bounds[input_index]
        target_dv = float(np.clip(dv_base + delta, lo, hi))
        applied_delta = float(target_dv - dv_base)
        input_name = dv_name

    # Pre-settle: hold all inputs at BASELINE values so the system reaches
    # steady state before the step change. For DV channels we explicitly
    # set the disturbance offset to 0 so any first-order-lag rest point in
    # the sim matches the observed baseline.
    y_hist = []
    dv_pos = int(input_index) if input_type == 'dv' else -1
    for _ in range(pre_steps):
        if input_type == 'dv' and dv_state_index is not None:
            dv_apply_ok = dv_apply_ok and _apply_dv_perturbation(
                sim, dv_state_index,
                float(_state_value_to_engineering(state, dv_state_index, meta)),
                dv_name, dv_pos=dv_pos, offset_delta=0.0,
            )
        state, done = sim.step(u0)
        y_hist.append([_state_value_to_engineering(state, i, meta) for i in cv_idxs])
        if done:
            break

    # Post-step: apply the step change (MV via u, DV via target_dv).
    for _ in range(post_steps):
        if input_type == 'dv' and dv_state_index is not None and target_dv is not None:
            dv_apply_ok = dv_apply_ok and _apply_dv_perturbation(
                sim, dv_state_index, target_dv, dv_name,
                dv_pos=dv_pos, offset_delta=float(applied_delta),
            )
        state, done = sim.step(u)
        y_hist.append([_state_value_to_engineering(state, i, meta) for i in cv_idxs])
        if done:
            break

    y_arr = np.asarray(y_hist, dtype=np.float64)
    state_names = meta.get('state_variables') or [f'S{i}' for i in range(max(max(mv_idxs), max(cv_idxs)) + 1)]
    out = {
        'u0': u0.tolist(),
        'u1': u.tolist(),
        'y': y_arr,
        'step_index': pre_steps,
        'input_type': str(input_type),
        'input_name': str(input_name),
        'delta_applied': float(applied_delta),
        'cv_names': [state_names[i] if i < len(state_names) else f'CV_{j}' for j, i in enumerate(cv_idxs)],
    }
    if input_type == 'mv':
        out['mv_name'] = str(input_name)
        out['mv'] = str(input_name)
    else:
        out['dv_name'] = str(input_name)
        out['dv'] = str(input_name)
        out['dv_apply_ok'] = bool(dv_apply_ok)
    return out


def _compute_per_channel_dynamics(per_pair: List[Dict]) -> Tuple[Dict[str, Dict], Dict[str, Dict]]:
    """Compute per-MV and per-CV dominant dynamics from valid per-pair estimates."""
    by_mv: Dict[str, Dict[str, List]] = {}
    by_cv: Dict[str, Dict[str, List]] = {}
    for row in per_pair:
        if not row.get('valid', False) or row.get('reject_reason'):
            continue
        tau_v = float(row['tau'])
        dt_v = float(row['dead_time'])
        if row.get('input_type') == 'mv':
            name = row.get('mv', '')
            by_mv.setdefault(name, {'tau': [], 'dead_time': []})
            by_mv[name]['tau'].append(tau_v)
            by_mv[name]['dead_time'].append(dt_v)
        cv = row.get('cv', '')
        by_cv.setdefault(cv, {'tau': [], 'dead_time': []})
        by_cv[cv]['tau'].append(tau_v)
        by_cv[cv]['dead_time'].append(dt_v)
    per_mv = {
        name: {'tau': float(np.median(v['tau'])), 'dead_time': float(np.median(v['dead_time']))}
        for name, v in by_mv.items()
    }
    per_cv = {
        name: {'tau': float(np.median(v['tau'])), 'dead_time': float(np.median(v['dead_time']))}
        for name, v in by_cv.items()
    }
    return per_mv, per_cv


def identify_dynamics(
    pre_steps: int = 0,
    post_steps: int = 600,
    step_size: float = 10.0,
    dv_step_size: float = None,
    repeats: int = 4,
    noise_stdv: float = 0.0,
    include_dv: bool = True,
    clean_mode: bool = True,
) -> Dict:
    """Identify FOPDT dynamics via open-loop step experiments.

    When *pre_steps* <= 0 the settle time is auto-detected from a probe
    run, making the identifier robust to any simulator dynamics.

    With ``clean_mode=True`` (the default) the identifier disables sim
    process/measurement noise AND domain randomisation for the duration
    of the experiment. We want a single deterministic plant answer
    (plant IS linear on test_sim; any variation between +/- step
    directions must come from physics, not Monte-Carlo noise).
    Noise + randomisation remain active during training so the policy
    still learns to cope with the full stochastic envelope.
    """
    prev_env = None
    if clean_mode:
        # Force noise off for the entire identification procedure. Keep a
        # snapshot so we can restore the training-time env vars afterward.
        # We also pin the randomization seed so every repeat and every
        # channel experiment starts from the *same* initial conditions.
        # Otherwise, nonlinear plants (e.g. the softsensor_lab ONNX model)
        # linearise around different operating points per run, producing
        # apparent gain/tau scatter that is neither measurement noise nor
        # real physics.
        prev_env = {
            k: os.environ.get(k)
            for k in (
                'SIM_DOMAIN_RANDOMIZATION',
                'DISTILLATION_DOMAIN_RANDOMIZATION',
                'SIM_NOISE_AMPLITUDE_JITTER_PCT',
                'SIM_NOISE_ENABLED',
                'SIM_DOMAIN_RANDOMIZATION_SEED',
            )
        }
        os.environ['SIM_DOMAIN_RANDOMIZATION'] = '0'
        os.environ['DISTILLATION_DOMAIN_RANDOMIZATION'] = '0'
        os.environ['SIM_NOISE_AMPLITUDE_JITTER_PCT'] = '0'
        os.environ['SIM_NOISE_ENABLED'] = '0'
        os.environ['SIM_DOMAIN_RANDOMIZATION_SEED'] = '1337'
        noise_stdv = 0.0
    try:
        return _identify_dynamics_inner(
            pre_steps=pre_steps,
            post_steps=post_steps,
            step_size=step_size,
            dv_step_size=dv_step_size,
            repeats=repeats,
            noise_stdv=noise_stdv,
            include_dv=include_dv,
        )
    finally:
        if clean_mode and prev_env is not None:
            for k, v in prev_env.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v


def _identify_dynamics_inner(
    pre_steps: int = 0,
    post_steps: int = 300,
    step_size: float = 10.0,
    dv_step_size: float = None,
    repeats: int = 2,
    noise_stdv: float = 0.0,
    include_dv: bool = True,
) -> Dict:
    # ── Adaptive pre-settle ───────────────────────────────────────────────
    auto_settle = pre_steps <= 0
    if auto_settle:
        settle_time = _detect_settle_time(noise_stdv=noise_stdv)
        pre_steps = max(int(settle_time * 1.5), 60)
        # Also ensure post_steps is long enough to capture the response.
        post_steps = max(post_steps, int(settle_time * 1.5))
        print(f"[DYNAMICS] Adaptive pre_steps={pre_steps}, post_steps={post_steps}")
    # ──────────────────────────────────────────────────────────────────────

    per_pair = []
    all_tau = []
    all_dead = []

    probe = create_sim(episode_length=20, sample_rate=1, noise_stdv=noise_stdv)
    probe_meta = resolve_sim_metadata(probe)
    mv_idxs = [int(x) for x in probe_meta.get('mv_indices', []) if x is not None]
    dv_idxs = [int(x) for x in probe_meta.get('dv_indices', []) if x is not None]
    mv_count = len(mv_idxs)
    if mv_count == 0:
        mv_count = int(os.environ.get('SIM_ACTION_DIM', '3'))

    mv_bounds = _resolve_mv_bounds(mv_count)
    dv_bounds = _resolve_dv_bounds(probe_meta, len(dv_idxs))

    if dv_step_size is None:
        dv_step_size = float(step_size)

    valid_counts = {'mv': 0, 'dv': 0}

    for mv_i in range(mv_count):
        span = float(max(1e-6, mv_bounds[mv_i][1] - mv_bounds[mv_i][0]))
        eff = _effective_delta(span=span, requested_step=float(step_size), noise_stdv=float(noise_stdv))
        for sign in (-1.0, 1.0):
            delta = sign * eff
            for rep in range(repeats):
                run = _single_step_experiment(
                    input_type='mv',
                    input_index=mv_i,
                    delta=delta,
                    pre_steps=pre_steps,
                    post_steps=post_steps,
                    noise_stdv=noise_stdv,
                    mv_bounds=mv_bounds,
                    dv_bounds=dv_bounds,
                )
                y = run['y']
                for cv_i, cv_name in enumerate(run['cv_names']):
                    ys = _moving_average(y[:, cv_i], window=5)
                    est = _estimate_fopdt(ys, step_index=run['step_index'])
                    row = {
                        'input': run['input_name'],
                        'input_type': 'mv',
                        'mv': run['input_name'],
                        'cv': cv_name,
                        'input_index': int(mv_i),
                        'delta': float(delta),
                        'repeat': rep + 1,
                        **est,
                    }
                    per_pair.append(row)
                    if est['valid']:
                        tau_v = float(est['tau'])
                        dt_v = float(est['dead_time'])
                        # Sanity guard: reject if response is clipped by
                        # experiment window or tau is implausibly small.
                        if tau_v < 1 or (dt_v + tau_v) > 0.7 * post_steps:
                            row['valid'] = False
                            row['reject_reason'] = 'sanity_guard'
                        else:
                            all_tau.append(tau_v)
                            all_dead.append(dt_v)
                            valid_counts['mv'] += 1

    if include_dv and dv_idxs:
        for dv_i in range(len(dv_idxs)):
            span = float(max(1e-6, dv_bounds[dv_i][1] - dv_bounds[dv_i][0]))
            eff = _effective_delta(span=span, requested_step=float(dv_step_size), noise_stdv=float(noise_stdv))
            for sign in (-1.0, 1.0):
                delta = sign * eff
                for rep in range(repeats):
                    run = _single_step_experiment(
                        input_type='dv',
                        input_index=dv_i,
                        delta=delta,
                        pre_steps=pre_steps,
                        post_steps=post_steps,
                        noise_stdv=noise_stdv,
                        mv_bounds=mv_bounds,
                        dv_bounds=dv_bounds,
                    )
                    y = run['y']
                    for cv_i, cv_name in enumerate(run['cv_names']):
                        ys = _moving_average(y[:, cv_i], window=5)
                        est = _estimate_fopdt(ys, step_index=run['step_index'])
                        row = {
                            'input': run['input_name'],
                            'input_type': 'dv',
                            'dv': run['input_name'],
                            'cv': cv_name,
                            'input_index': int(dv_i),
                            'delta': float(delta),
                            'repeat': rep + 1,
                            'dv_apply_ok': bool(run.get('dv_apply_ok', True)),
                            **est,
                        }
                        per_pair.append(row)
                        if est['valid']:
                            tau_v = float(est['tau'])
                            dt_v = float(est['dead_time'])
                            if tau_v < 1 or (dt_v + tau_v) > 0.7 * post_steps:
                                row['valid'] = False
                                row['reject_reason'] = 'sanity_guard'
                            else:
                                all_tau.append(tau_v)
                                all_dead.append(dt_v)
                                valid_counts['dv'] += 1

    if not dv_idxs:
        print("[DYNAMICS] No DV indices detected — DV experiments skipped.")

    # ── MIMO retry: extend window for inputs with no valid estimates ───────
    failed_mv_indices = set(range(mv_count))
    failed_dv_indices = set(range(len(dv_idxs))) if (include_dv and dv_idxs) else set()
    for row in per_pair:
        if row.get('valid', False) and not row.get('reject_reason'):
            idx = row.get('input_index')
            if idx is not None:
                if row['input_type'] == 'mv':
                    failed_mv_indices.discard(int(idx))
                elif row['input_type'] == 'dv':
                    failed_dv_indices.discard(int(idx))

    if failed_mv_indices or failed_dv_indices:
        extended_post = max(post_steps * 3, 900)
        retry_tag = f"extended_post={extended_post}"

        if failed_mv_indices:
            print(f"[DYNAMICS] Retry {len(failed_mv_indices)} MV channel(s) "
                  f"with no valid estimates ({retry_tag})")
            for mv_i in sorted(failed_mv_indices):
                span = float(max(1e-6, mv_bounds[mv_i][1] - mv_bounds[mv_i][0]))
                eff = _effective_delta(span=span, requested_step=float(step_size),
                                      noise_stdv=float(noise_stdv))
                for sign in (-1.0, 1.0):
                    delta = sign * eff
                    for rep in range(repeats):
                        run = _single_step_experiment(
                            input_type='mv', input_index=mv_i, delta=delta,
                            pre_steps=pre_steps, post_steps=extended_post,
                            noise_stdv=noise_stdv, mv_bounds=mv_bounds, dv_bounds=dv_bounds,
                        )
                        y = run['y']
                        for cv_i, cv_name in enumerate(run['cv_names']):
                            ys = _moving_average(y[:, cv_i], window=5)
                            est = _estimate_fopdt(ys, step_index=run['step_index'])
                            row = {
                                'input': run['input_name'],
                                'input_type': 'mv',
                                'mv': run['input_name'],
                                'cv': cv_name,
                                'input_index': int(mv_i),
                                'delta': float(delta),
                                'repeat': rep + 1,
                                'retry': True,
                                **est,
                            }
                            per_pair.append(row)
                            if est['valid']:
                                tau_v = float(est['tau'])
                                dt_v = float(est['dead_time'])
                                if tau_v < 1 or (dt_v + tau_v) > 0.7 * extended_post:
                                    row['valid'] = False
                                    row['reject_reason'] = 'sanity_guard'
                                else:
                                    all_tau.append(tau_v)
                                    all_dead.append(dt_v)
                                    valid_counts['mv'] += 1

        if failed_dv_indices:
            print(f"[DYNAMICS] Retry {len(failed_dv_indices)} DV channel(s) "
                  f"with no valid estimates ({retry_tag})")
            for dv_i in sorted(failed_dv_indices):
                span = float(max(1e-6, dv_bounds[dv_i][1] - dv_bounds[dv_i][0]))
                eff = _effective_delta(span=span, requested_step=float(dv_step_size),
                                      noise_stdv=float(noise_stdv))
                for sign in (-1.0, 1.0):
                    delta = sign * eff
                    for rep in range(repeats):
                        run = _single_step_experiment(
                            input_type='dv', input_index=dv_i, delta=delta,
                            pre_steps=pre_steps, post_steps=extended_post,
                            noise_stdv=noise_stdv, mv_bounds=mv_bounds, dv_bounds=dv_bounds,
                        )
                        y = run['y']
                        for cv_i, cv_name in enumerate(run['cv_names']):
                            ys = _moving_average(y[:, cv_i], window=5)
                            est = _estimate_fopdt(ys, step_index=run['step_index'])
                            row = {
                                'input': run['input_name'],
                                'input_type': 'dv',
                                'dv': run['input_name'],
                                'cv': cv_name,
                                'input_index': int(dv_i),
                                'delta': float(delta),
                                'repeat': rep + 1,
                                'retry': True,
                                'dv_apply_ok': bool(run.get('dv_apply_ok', True)),
                                **est,
                            }
                            per_pair.append(row)
                            if est['valid']:
                                tau_v = float(est['tau'])
                                dt_v = float(est['dead_time'])
                                if tau_v < 1 or (dt_v + tau_v) > 0.7 * extended_post:
                                    row['valid'] = False
                                    row['reject_reason'] = 'sanity_guard'
                                else:
                                    all_tau.append(tau_v)
                                    all_dead.append(dt_v)
                                    valid_counts['dv'] += 1

    # ── Per-channel dynamics ──────────────────────────────────────────────
    per_mv_dynamics, per_cv_dynamics = _compute_per_channel_dynamics(per_pair)

    # ── Aggregation ──────────────────────────────────────────────────────
    # For MIMO (multiple MVs with valid estimates), use the slowest (max)
    # per-MV dynamics to ensure lookback/episode sizing accommodates every loop.
    if per_mv_dynamics and len(per_mv_dynamics) > 1:
        tau_dominant = max(v['tau'] for v in per_mv_dynamics.values())
        dead_time = max(v['dead_time'] for v in per_mv_dynamics.values())
        print(f"[DYNAMICS] MIMO aggregation ({len(per_mv_dynamics)} MVs): "
              f"tau_dominant={tau_dominant:.1f} dead_time={dead_time:.1f}")
        for name, d in sorted(per_mv_dynamics.items()):
            print(f"  MV '{name}': tau={d['tau']:.1f}  dead_time={d['dead_time']:.1f}")
    elif all_tau:
        tau_dominant = float(np.median(np.asarray(all_tau)))
        dead_time = float(np.median(np.asarray(all_dead)))
    else:
        tau_dominant = 75.0
        dead_time = 8.0
        print("[DYNAMICS] WARNING: No valid FOPDT estimates — "
              "using conservative defaults tau=75, dead_time=8.")

    # Fastest dynamics across the full transfer-function matrix (MV+DV → CV).
    # Used downstream for scan-rate ceiling (sampling must resolve the
    # fastest channel, not the slowest).
    if all_tau:
        tau_fastest = float(np.min(np.asarray(all_tau)))
        dead_time_fastest = float(np.min(np.asarray(all_dead)))
    else:
        tau_fastest = tau_dominant
        dead_time_fastest = dead_time

    return {
        'tau_dominant_identified': tau_dominant,
        'dead_time_identified': dead_time,
        'tau_fastest_identified': tau_fastest,
        'dead_time_fastest_identified': dead_time_fastest,
        'valid_estimate_count': len(all_tau),
        'valid_estimate_count_by_input_type': valid_counts,
        'per_mv_dynamics': per_mv_dynamics,
        'per_cv_dynamics': per_cv_dynamics,
        'auto_settle_used': bool(auto_settle),
        'experiment_config': {
            'pre_steps': int(pre_steps),
            'post_steps': int(post_steps),
            'step_size': float(step_size),
            'dv_step_size': float(dv_step_size),
            'repeats': int(repeats),
            'noise_stdv': float(noise_stdv),
            'include_dv': bool(include_dv),
            'unit_policy': 'engineering_units',
        },
        'per_pair_estimates': per_pair,
        'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
    }


def identify_and_save_dynamics(
    output_path: str,
    pre_steps: int = 0,
    post_steps: int = 600,
    step_size: float = 10.0,
    dv_step_size: float = None,
    repeats: int = 4,
    noise_stdv: float = 0.0,
    include_dv: bool = True,
    metadata: Dict = None,
    clean_mode: bool = True,
) -> Dict:
    result = identify_dynamics(
        pre_steps=pre_steps,
        post_steps=post_steps,
        step_size=step_size,
        dv_step_size=dv_step_size,
        repeats=repeats,
        noise_stdv=noise_stdv,
        include_dv=include_dv,
        clean_mode=clean_mode,
    )
    if metadata:
        result['metadata'] = metadata

    output_dir = os.path.dirname(os.path.abspath(output_path))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=4)

    return result


def main():
    parser = argparse.ArgumentParser(description='Identify process dynamics by open-loop simulator experiments.')
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--pre-steps', type=int, default=0,
                        help='Pre-settle steps before step (0 = auto-detect)')
    parser.add_argument('--post-steps', type=int, default=300)
    parser.add_argument('--step-size', type=float, default=10.0)
    parser.add_argument('--dv-step-size', type=float, default=None)
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--noise-stdv', type=float, default=0.0)
    parser.add_argument('--disable-dv', action='store_true', help='Disable DV->CV perturbation experiments.')
    args = parser.parse_args()

    info = identify_and_save_dynamics(
        output_path=args.output,
        pre_steps=args.pre_steps,
        post_steps=args.post_steps,
        step_size=args.step_size,
        dv_step_size=args.dv_step_size,
        repeats=args.repeats,
        noise_stdv=args.noise_stdv,
        include_dv=(not args.disable_dv),
    )

    print(f"tau_dominant_identified: {info['tau_dominant_identified']:.3f}")
    print(f"dead_time_identified: {info['dead_time_identified']:.3f}")
    print(f"valid_estimate_count: {info['valid_estimate_count']}")
    print(f"saved: {args.output}")


if __name__ == '__main__':
    main()
