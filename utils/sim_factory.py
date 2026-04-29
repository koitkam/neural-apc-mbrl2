"""Create simulator instances and normalize metadata for workflow scripts.

Functionality:
- Loads simulator module/class from `CONTROL_SETUP_JSON` or `SIM_MODEL_*` envs.
- Instantiates simulator with tolerant constructor filtering.
- Resolves and attaches standardized metadata/aliases expected by training and
    validation scripts.

Inputs:
- Setup file: `CONTROL_SETUP_JSON`.
- Env overrides: `SIM_MODEL_MODULE`, `SIM_MODEL_CLASS`,
    `SIM_MODEL_KWARGS_JSON`, `SIM_STATE_VARIABLES_JSON`, `SIM_MV_INDICES_JSON`,
    `SIM_CV_INDICES_JSON`, `SIM_DV_INDICES_JSON`.
- Runtime constructor args: `episode_length`, `sample_rate`, `noise_stdv`.

Outputs:
- Simulator instance exposing `reset()`/`step()` and attached metadata fields.
- Helper outputs from `resolve_sim_metadata()` for diagnostics and validation.

Main steps:
1) Read setup/env for simulator class and constructor kwargs.
2) Instantiate simulator using accepted constructor parameters.
3) Resolve metadata from simulator attributes or env/config fallbacks.
4) Attach compatibility aliases (`reflux_mv_index`, etc.) and validate fields.

Required simulator API:
- Constructor accepts at least `episode_length`.
- Methods: `reset() -> (state, done)` and `step(action) -> (state, done)`.
"""

import importlib
import inspect
import json
import os
import glob
from typing import Any, Dict, List, Optional


def _load_setup_file() -> Dict[str, Any]:
    path = os.environ.get('CONTROL_SETUP_JSON', '')
    if not path:
        return {}
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _env_json(name: str, default):
    raw = os.environ.get(name, '')
    if not raw:
        return default
    try:
        return json.loads(raw)
    except Exception:
        return default


def _load_sim_class():
    setup = _load_setup_file()
    sim_cfg = setup.get('simulator', {}) if isinstance(setup.get('simulator', {}), dict) else {}
    module_name = os.environ.get('SIM_MODEL_MODULE', sim_cfg.get('module', ''))
    class_name = os.environ.get('SIM_MODEL_CLASS', sim_cfg.get('class', ''))

    candidates = []
    if module_name and class_name:
        candidates.append((module_name, class_name))

    # Auto-discover simulator class from simulation setup files when explicit
    # env/setup values are absent.
    if not candidates:
        project_dir = os.path.dirname(os.path.abspath(__file__))
        setup_paths = sorted(glob.glob(os.path.join(project_dir, 'simulation', '*', 'control_setup.json')))
        if len(setup_paths) > 1 and not os.environ.get('CONTROL_SETUP_JSON', '').strip():
            sim_names = [os.path.basename(os.path.dirname(p)) for p in setup_paths]
            raise RuntimeError(
                f'Multiple simulations found: {sim_names}. '
                'Set --simulation-dir or CONTROL_SETUP_JSON to select one.'
            )
        for p in setup_paths:
            try:
                with open(p, 'r', encoding='utf-8') as f:
                    cfg = json.load(f)
                sim_auto = cfg.get('simulator', {}) if isinstance(cfg.get('simulator', {}), dict) else {}
                m = str(sim_auto.get('module', '')).strip()
                c = str(sim_auto.get('class', '')).strip()
                if m and c:
                    candidates.append((m, c))
            except Exception:
                continue

    last_exc = None
    for mod_name, cls_name in candidates:
        try:
            mod = importlib.import_module(mod_name)
            return getattr(mod, cls_name)
        except Exception as exc:
            last_exc = exc

    raise RuntimeError(
        'Unable to resolve simulator class from CONTROL_SETUP_JSON/SIM_MODEL_* '
        'or discovered simulation setup files. Provide simulator.module/class in control_setup.json '
        'or set SIM_MODEL_MODULE and SIM_MODEL_CLASS.'
    ) from last_exc


def _apply_setup_noise_and_randomization_env():
    """Seed os.environ from the ``noise_and_randomization`` block of
    control_setup.json so simulators, :class:`DomainRandomizer`, and
    :class:`SimNoiseWrapper` all see consistent defaults.

    Explicit env vars (exported by the user or set by the dynamics
    identifier's ``clean_mode``) always win — we only fill in gaps via
    ``os.environ.setdefault``. Defaults when the block is absent leave
    everything enabled (domain randomization ON, process noise ON,
    measurement noise ON).
    """
    setup = _load_setup_file()
    block = setup.get('noise_and_randomization', {})
    if not isinstance(block, dict):
        return

    def _as_bool_env(key: str, value: Any) -> Optional[str]:
        if isinstance(value, bool):
            return '1' if value else '0'
        if isinstance(value, (int, float)):
            return '1' if value else '0'
        if isinstance(value, str):
            return value.strip()
        return None

    mapping = {
        'domain_randomization': 'SIM_DOMAIN_RANDOMIZATION',
        'param_randomization_pct': 'SIM_PARAM_RANDOMIZATION_PCT',
        'noise_enabled': 'SIM_NOISE_ENABLED',
    }
    for key, env_name in mapping.items():
        if key not in block:
            continue
        val = block.get(key)
        if val is None:
            continue
        if key in ('domain_randomization', 'noise_enabled'):
            env_val = _as_bool_env(env_name, val)
        else:
            env_val = str(val)
        if env_val is None:
            continue
        os.environ.setdefault(env_name, env_val)


def _constructor_kwargs(episode_length: int, sample_rate: int, noise_stdv: float):
    setup = _load_setup_file()
    sim_cfg = setup.get('simulator', {}) if isinstance(setup.get('simulator', {}), dict) else {}
    file_kwargs = sim_cfg.get('kwargs', {}) if isinstance(sim_cfg.get('kwargs', {}), dict) else {}
    extra = _env_json('SIM_MODEL_KWARGS_JSON', file_kwargs)
    kwargs = dict(extra) if isinstance(extra, dict) else {}
    kwargs.setdefault('episode_length', episode_length)
    kwargs.setdefault('sample_rate', sample_rate)
    kwargs.setdefault('noise_stdv', noise_stdv)
    return kwargs


def create_sim(episode_length: int, sample_rate: int = 1, noise_stdv: float = 0.03,
               noise_config: Optional[Dict] = None):
    """Create a simulator instance wrapped with noise injection.

    Parameters
    ----------
    noise_config : dict, optional
        External noise configuration (from :func:`control_noise_config.build_noise_config`).
        When provided it overrides both ``SIM_NOISE_CONFIG_JSON`` and the
        simulator's built-in ``noise_config``.  When *None*, the wrapper
        checks for the env var and finally falls back to ``sim.noise_config``.
    """
    from utils.sim_noise import SimNoiseWrapper

    # Propagate control_setup.json noise/randomization defaults into the
    # environment before the sim is constructed so DomainRandomizer picks
    # them up. Existing env vars (e.g. set by the dynamics identifier's
    # clean_mode) take precedence.
    _apply_setup_noise_and_randomization_env()

    sim_cls = _load_sim_class()
    kwargs = _constructor_kwargs(episode_length=episode_length, sample_rate=sample_rate, noise_stdv=noise_stdv)

    # Be tolerant to constructor signatures of arbitrary simulator classes.
    try:
        sig = inspect.signature(sim_cls.__init__)
        accepted = set(sig.parameters.keys())
        accepted.discard('self')
        safe_kwargs = {k: v for k, v in kwargs.items() if k in accepted}
    except Exception:
        safe_kwargs = kwargs

    sim = sim_cls(**safe_kwargs)
    attach_standard_metadata(sim)
    sim = _SimSanityWrapper(sim)
    sim = SimNoiseWrapper(sim, noise_config=noise_config)
    return sim


def _state_variables(sim) -> List[str]:
    setup = _load_setup_file()
    io_cfg = setup.get('io', {}) if isinstance(setup.get('io', {}), dict) else {}
    names = getattr(sim, 'state_variables', None)
    if isinstance(names, list) and names:
        return [str(x) for x in names]
    env_names = _env_json('SIM_STATE_VARIABLES_JSON', io_cfg.get('state_variables', []))
    if isinstance(env_names, list) and env_names:
        return [str(x) for x in env_names]
    return []


def _mv_indices(sim) -> List[int]:
    setup = _load_setup_file()
    io_cfg = setup.get('io', {}) if isinstance(setup.get('io', {}), dict) else {}
    mv = getattr(sim, 'mv_indices', None)
    if isinstance(mv, list) and mv:
        return [int(x) for x in mv]
    env_mv = _env_json('SIM_MV_INDICES_JSON', io_cfg.get('mv_indices', []))
    if isinstance(env_mv, list) and env_mv:
        return [int(x) for x in env_mv]
    return []


def _coerce_ranges(raw) -> List[List[float]]:
    if not isinstance(raw, list):
        return []
    out = []
    for item in raw:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        try:
            lo = float(item[0])
            hi = float(item[1])
        except Exception:
            continue
        out.append([lo, hi])
    return out


def resolve_sim_metadata(sim) -> Dict[str, Any]:
    setup = _load_setup_file()
    io_cfg = setup.get('io', {}) if isinstance(setup.get('io', {}), dict) else {}
    state_vars = _state_variables(sim)
    mv = _mv_indices(sim)

    sim_cv = getattr(sim, 'cv_indices', None)
    if isinstance(sim_cv, list) and sim_cv:
        cv_idxs = [int(x) for x in sim_cv]
    else:
        cv_idxs = _env_json('SIM_CV_INDICES_JSON', io_cfg.get('cv_indices', []))
        if not isinstance(cv_idxs, list):
            cv_idxs = []
        cv_idxs = [int(x) for x in cv_idxs if isinstance(x, (int, float)) or str(x).lstrip('-').isdigit()]

    top = getattr(sim, 'top_pv_index', None)
    bottom = getattr(sim, 'bottom_pv_index', None)
    pressure = getattr(sim, 'pressure_pv_index', None)

    if top is None and len(cv_idxs) >= 1:
        top = int(cv_idxs[0])
    if bottom is None and len(cv_idxs) >= 2:
        bottom = int(cv_idxs[1])
    if pressure is None and len(cv_idxs) >= 3:
        pressure = int(cv_idxs[2])

    if not cv_idxs:
        inferred = []
        for v in (top, bottom, pressure):
            if v is not None:
                inferred.append(int(v))
        cv_idxs = inferred

    sim_dv = getattr(sim, 'dv_indices', None)
    if isinstance(sim_dv, list) and sim_dv:
        dv_idxs = [int(x) for x in sim_dv]
    else:
        dv_idxs = _env_json('SIM_DV_INDICES_JSON', io_cfg.get('dv_indices', []))
        if not isinstance(dv_idxs, list):
            dv_idxs = []
        dv_idxs = [int(x) for x in dv_idxs if isinstance(x, (int, float)) or str(x).lstrip('-').isdigit()]

    def _ranges_from(attr_name: str, cfg_key: str) -> List[List[float]]:
        sim_ranges = getattr(sim, attr_name, None)
        coerced = _coerce_ranges(sim_ranges) if sim_ranges else []
        if coerced:
            return coerced
        return _coerce_ranges(io_cfg.get(cfg_key, []))

    mv_norm_ranges = _ranges_from('mv_normalization_ranges', 'mv_normalization_ranges')
    cv_norm_ranges = _ranges_from('cv_normalization_ranges', 'cv_normalization_ranges')
    dv_norm_ranges = _ranges_from('dv_normalization_ranges', 'dv_normalization_ranges')
    state_norm_ranges = _ranges_from('state_normalization_ranges', 'state_normalization_ranges')
    state_is_normalized = bool(getattr(sim, 'state_is_normalized', io_cfg.get('state_is_normalized', False)))

    return {
        'state_variables': state_vars,
        'state_dim': len(state_vars) if state_vars else None,
        'mv_indices': [int(x) for x in mv],
        'cv_indices': [int(x) for x in cv_idxs],
        'dv_indices': [int(x) for x in dv_idxs],
        'mv_normalization_ranges': mv_norm_ranges,
        'cv_normalization_ranges': cv_norm_ranges,
        'dv_normalization_ranges': dv_norm_ranges,
        'state_normalization_ranges': state_norm_ranges,
        'state_is_normalized': state_is_normalized,
        'action_dim': len(mv),
        'top_pv_index': top,
        'bottom_pv_index': bottom,
        'pressure_pv_index': pressure,
    }


def attach_standard_metadata(sim):
    meta = resolve_sim_metadata(sim)
    for k, v in meta.items():
        if v is None:
            continue
        if not hasattr(sim, k):
            setattr(sim, k, v)
    if not hasattr(sim, 'state_variables') and meta['state_variables']:
        sim.state_variables = meta['state_variables']
    if not hasattr(sim, 'mv_indices') and meta['mv_indices']:
        sim.mv_indices = meta['mv_indices']

    # Backward-compatible aliases expected by existing control scripts.
    if hasattr(sim, 'mv_indices') and len(sim.mv_indices) >= 3:
        if not hasattr(sim, 'reflux_mv_index'):
            sim.reflux_mv_index = int(sim.mv_indices[0])
        if not hasattr(sim, 'boilup_mv_index'):
            sim.boilup_mv_index = int(sim.mv_indices[1])
        if not hasattr(sim, 'cooling_mv_index'):
            sim.cooling_mv_index = int(sim.mv_indices[2])
    if hasattr(sim, 'mv_indices') and len(sim.mv_indices) >= 4:
        if not hasattr(sim, 'feed_flow_mv_index'):
            sim.feed_flow_mv_index = int(sim.mv_indices[3])

    if not hasattr(sim, 'top_pv_index') and meta.get('top_pv_index') is not None:
        sim.top_pv_index = int(meta['top_pv_index'])
    if not hasattr(sim, 'bottom_pv_index') and meta.get('bottom_pv_index') is not None:
        sim.bottom_pv_index = int(meta['bottom_pv_index'])
    if not hasattr(sim, 'pressure_pv_index') and meta.get('pressure_pv_index') is not None:
        sim.pressure_pv_index = int(meta['pressure_pv_index'])

    return sim


def require_sim_metadata(sim, required_fields: List[str]):
    missing = [k for k in required_fields if not hasattr(sim, k) or getattr(sim, k) is None]
    if missing:
        raise ValueError(
            'Simulator is missing required metadata fields: '
            + ', '.join(missing)
            + '. Provide attributes on the simulator class or set env overrides '
            + '(SIM_MV_INDICES_JSON, SIM_CV_INDICES_JSON, SIM_DV_INDICES_JSON, SIM_STATE_VARIABLES_JSON).'
        )


class _SimSanityWrapper:
    """Transparent wrapper that validates simulator outputs.

    Guards against three classes of silent failures:
    - NaN / Inf values in state (sanitised to the simulator bounds' midpoint)
    - state-vector shape mismatches (logged once)
    - wrong types returned from ``reset`` / ``step``

    The wrapper forwards all non-intercepted attributes to the underlying
    simulator so downstream metadata access still works.
    """

    def __init__(self, inner):
        import numpy as _np
        self._inner = inner
        self._np = _np
        self._logged_shape_mismatch = False
        self._logged_nan = False

    def __getattr__(self, item):
        return getattr(self._inner, item)

    def _sanitise(self, state):
        import numpy as _np
        arr = _np.asarray(state, dtype='float32').reshape(-1)
        if not _np.all(_np.isfinite(arr)):
            if not self._logged_nan:
                import sys as _sys
                print('[sim_sanity] WARNING: simulator returned NaN/Inf in state '
                      '(sanitising this and future occurrences silently).', file=_sys.stderr)
                self._logged_nan = True
            # Replace with per-channel midpoint using normalisation ranges if available.
            ranges = getattr(self._inner, 'state_normalization_ranges', None)
            if isinstance(ranges, list) and len(ranges) == len(arr):
                mid = _np.array([(float(r[0]) + float(r[1])) * 0.5 for r in ranges], dtype='float32')
                arr = _np.where(_np.isfinite(arr), arr, mid)
            else:
                arr = _np.nan_to_num(arr, nan=0.0, posinf=1e6, neginf=-1e6)
        expected = getattr(self._inner, 'state_variables', None)
        if isinstance(expected, list) and expected and len(arr) != len(expected):
            if not self._logged_shape_mismatch:
                import sys as _sys
                print(f'[sim_sanity] WARNING: state length {len(arr)} != '
                      f'declared state_variables length {len(expected)}.', file=_sys.stderr)
                self._logged_shape_mismatch = True
        return arr

    def reset(self, *args, **kwargs):
        out = self._inner.reset(*args, **kwargs)
        if isinstance(out, tuple) and len(out) == 2:
            state, done = out
            return self._sanitise(state), bool(done)
        return self._sanitise(out), False

    def step(self, action):
        out = self._inner.step(action)
        if isinstance(out, tuple) and len(out) == 2:
            state, done = out
            return self._sanitise(state), bool(done)
        return self._sanitise(out), False
