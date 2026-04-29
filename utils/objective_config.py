"""Load and normalize control objective configuration.

New schema (April 2026 refactor):

    {
      "control_speed": "normal",        # aggressive | normal | slow
      "objective_use_normalized": 1,
      "bounds": {
        "mvs": {"mv_0": [0, 100], ...},
        "outputs": {"cv_0": [68, 96], ...}
      },
      "cv_priority": ["cv_0", "cv_1"],  # ranked, index 0 = most important
      "targets": {                      # optional
        "mvs": {"mv_0": 50.0},
        "cvs": {"cv_0": 82.0}
      },
      "weights": {                      # only economic/scalar terms here
        "mv_economic": {"mv_0": 0.1},   # optional, per-MV (dict or list)
        "cv_economic": {"cv_0": 0.0}
      },
      "runtime_setpoints": {
        "targets_enabled": false   # bool (all CVs) or list of bools, one per CV.
      }                             # MV/CV bounds variation is always on; targets are opt-in.
    }

Violation and move weights are **auto-derived at runtime** from
``cv_priority`` + ``control_speed`` + identified per-MV tau (see
:mod:`utils.auto_weights`) and are NOT part of the JSON.

Legacy fields (``*_violation_weights``, ``mv_move_weights``) are still parsed
if present for backward compatibility but a warning is printed.
"""

import json
import os
import re
import sys
from typing import Any, Dict

LEGACY_WEIGHT_KEYS = (
    'mv_violation_weights', 'cv_violation_weights', 'mv_move_weights',
)


def _safe_float(v: Any, d: float) -> float:
    try:
        return float(v)
    except Exception:
        return float(d)


def _load_file_config() -> Dict[str, Any]:
    path = os.environ.get('CONTROL_OBJECTIVE_JSON', '')
    if not path:
        return {}
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _default_spec() -> Dict[str, Any]:
    return {
        'control_speed': 'normal',
        'objective_use_normalized': 1,
        'bounds': {
            'mvs': {},
            'outputs': {},
        },
        'cv_priority': [],
        'targets': {'mvs': {}, 'cvs': {}},
        'weights': {
            'mv_economic': {},
            'cv_economic': {},
        },
        'runtime_setpoints': {
            'targets_enabled': False,
        },
    }


def _merge(base: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in patch.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def _coerce_named_bounds(raw: Any, prefix: str) -> Dict[str, list]:
    """Coerce a dict of {name: [lo, hi]} keeping named keys authoritative.

    Also accepts a plain list; auto-names entries ``{prefix}_0``, ``{prefix}_1`` ...
    """
    if isinstance(raw, list):
        out = {}
        for i, v in enumerate(raw):
            if isinstance(v, (list, tuple)) and len(v) >= 2:
                out[f'{prefix}_{i}'] = [_safe_float(v[0], 0.0), _safe_float(v[1], 1.0)]
        return out
    if not isinstance(raw, dict):
        return {}

    def _auto(k):
        return bool(re.fullmatch(rf'{prefix}_\d+', str(k).strip().lower()))

    named = {k: v for k, v in raw.items() if not _auto(str(k))}
    src = named if named else raw
    out = {}
    for k, v in src.items():
        if isinstance(v, (list, tuple)) and len(v) >= 2:
            out[str(k)] = [_safe_float(v[0], 0.0), _safe_float(v[1], 1.0)]
    return out


def _sorted_bound_values(bd: Dict[str, list]) -> list:
    return [list(v) for _, v in sorted(bd.items())]


def _coerce_named_scalar(raw: Any, prefix: str) -> Dict[str, float]:
    if isinstance(raw, list):
        return {f'{prefix}_{i}': _safe_float(v, 0.0) for i, v in enumerate(raw)}
    if not isinstance(raw, dict):
        return {}
    return {str(k): _safe_float(v, 0.0) for k, v in raw.items()}


def _coerce_cv_priority(raw: Any) -> list:
    if not isinstance(raw, list):
        return []
    out = []
    for item in raw:
        s = str(item).strip().lower()
        if re.fullmatch(r'cv_\d+', s):
            out.append(s)
    return out


def _warn_legacy_weights(file_cfg: Dict[str, Any]) -> None:
    w = file_cfg.get('weights') if isinstance(file_cfg, dict) else None
    if not isinstance(w, dict):
        return
    present = [k for k in LEGACY_WEIGHT_KEYS if k in w]
    if present:
        print(
            f'[objective_config] NOTICE: legacy weight keys {present} found in '
            f'control_objective.json - these are now auto-derived from '
            f'cv_priority + control_speed + identified dynamics and will be '
            f'ignored. You can safely remove them.',
            file=sys.stderr,
        )


def _coerce_spec(spec: Dict[str, Any]) -> Dict[str, Any]:
    bounds_in = spec.get('bounds') or {}
    mvs_in = bounds_in.get('mvs') or bounds_in.get('inputs') or {}
    cvs_in = bounds_in.get('outputs') or {}
    if not mvs_in and isinstance(bounds_in.get('mv_bounds'), list):
        mvs_in = bounds_in.get('mv_bounds')
    if not cvs_in and isinstance(bounds_in.get('cv_bounds'), list):
        cvs_in = bounds_in.get('cv_bounds')
    mvs = _coerce_named_bounds(mvs_in, 'mv')
    cvs = _coerce_named_bounds(cvs_in, 'cv')

    mv_bounds_list = _sorted_bound_values(mvs)
    cv_bounds_list = _sorted_bound_values(cvs)

    targets_in = spec.get('targets') or {}
    w_in = spec.get('weights') or {}
    if not targets_in.get('mvs') and (w_in.get('mv_target_values') is not None):
        targets_in = {**targets_in, 'mvs': _coerce_named_scalar(w_in.get('mv_target_values'), 'mv')}
    if not targets_in.get('cvs') and (w_in.get('cv_target_values') is not None):
        targets_in = {**targets_in, 'cvs': _coerce_named_scalar(w_in.get('cv_target_values'), 'cv')}
    mv_targets = _coerce_named_scalar(targets_in.get('mvs') or {}, 'mv')
    cv_targets = _coerce_named_scalar(targets_in.get('cvs') or {}, 'cv')

    mv_econ = _coerce_named_scalar(w_in.get('mv_economic') or w_in.get('mv_economic_weights') or {}, 'mv')
    cv_econ = _coerce_named_scalar(w_in.get('cv_economic') or w_in.get('cv_economic_weights') or {}, 'cv')

    rs_in = spec.get('runtime_setpoints') or {}
    default_rs = _default_spec()['runtime_setpoints']
    rs = {**default_rs, **{k: v for k, v in rs_in.items() if v is not None}}

    speed = str(spec.get('control_speed') or 'normal').strip().lower()
    if speed not in ('aggressive', 'normal', 'slow'):
        speed = 'normal'

    n_mv = len(mvs)
    n_cv = len(cvs)
    out = {
        'control_speed': speed,
        'objective_use_normalized': 1 if int(_safe_float(spec.get('objective_use_normalized', 1), 1)) != 0 else 0,
        'bounds': {
            'mvs': mvs,
            'outputs': cvs,
            'mv_bounds': mv_bounds_list,
            'cv_bounds': cv_bounds_list,
            'inputs': dict(mvs),
        },
        'cv_priority': _coerce_cv_priority(spec.get('cv_priority') or []),
        'targets': {
            'mvs': mv_targets,
            'cvs': cv_targets,
        },
        'weights': {
            'mv_economic': mv_econ,
            'cv_economic': cv_econ,
            'mv_economic_weights': [mv_econ.get(f'mv_{i}', 0.0) for i in range(n_mv)],
            'cv_economic_weights': [cv_econ.get(f'cv_{i}', 0.0) for i in range(n_cv)],
            'mv_target_values': [mv_targets.get(f'mv_{i}', 0.0) for i in range(n_mv)],
            'cv_target_values': [cv_targets.get(f'cv_{i}', 0.0) for i in range(n_cv)],
            'mv_target_weights': [0.0 for _ in range(n_mv)],
            'cv_target_weights': [0.0 for _ in range(n_cv)],
            'objective_use_normalized': 1 if int(_safe_float(spec.get('objective_use_normalized', 1), 1)) != 0 else 0,
        },
        'runtime_setpoints': rs,
    }
    return out


def load_objective_spec() -> Dict[str, Any]:
    spec = _default_spec()
    file_cfg = _load_file_config()
    if file_cfg:
        _warn_legacy_weights(file_cfg)
        spec = _merge(spec, file_cfg)

    if os.environ.get('CONTROL_SPEED'):
        spec['control_speed'] = os.environ['CONTROL_SPEED']
    spec['objective_use_normalized'] = 1 if int(_safe_float(
        os.environ.get('OBJ_USE_NORMALIZED', spec.get('objective_use_normalized', 1)),
        spec.get('objective_use_normalized', 1),
    )) != 0 else 0

    return _coerce_spec(spec)
