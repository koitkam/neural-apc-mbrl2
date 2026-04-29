"""Compute and persist physics-informed latent lookback and scan-rate settings.

Functionality:
- Converts process dynamics (`tau_dominant`, `dead_time`) into a nominal
    lookback horizon and a nearby candidate set.
- Derives a recommended `sample_rate` so long horizons can be subsampled to a
    manageable number of model input points.
- Supports explicit seed mode or inferred seed from time-constant formula.

Inputs:
- CLI args: `--seed`, `--min`, `--max`, `--tau-dominant`, `--dead-time`,
    `--horizon-multiplier`, `--target-sampled-points`, `--max-scan-rate`, `--output`.

Outputs:
- JSON report containing `identified_lookback`, `identified_sample_rate`, candidate
    lookbacks, per-candidate sample rates, seed source, scalarization policy,
    process constants, and timestamp.

Main steps:
1) Compute seed from explicit input or inferred formula.
2) Generate candidate list around seed and clamp to min/max.
3) Build feature-wise vectors from pairing/simulator metadata.
4) Scalarize to single values (`max` raw lookback vector, `min` scan-rate vector).
5) Save deterministic summary for workflow runners.
"""

import argparse
import json
import os
import time
import math
from typing import Any, Dict, List

from utils.sim_factory import create_sim, resolve_sim_metadata


def _dedupe_sorted_int(values: List[int]) -> List[int]:
    return sorted({int(v) for v in values if int(v) > 0})


def _safe_name(name: Any) -> str:
    return ''.join(ch for ch in str(name).lower() if ch.isalnum() or ch == '_')


def _calc_seed_from_process(tau_dominant: float, dead_time: float, horizon_multiplier: float) -> int:
    # Horizon ~= dead time + multiplier * dominant time constant.
    return max(1, int(math.ceil(float(dead_time) + float(horizon_multiplier) * float(tau_dominant))))


def _parse_json_dict(raw: str) -> Dict[str, float]:
    if not raw:
        return {}
    try:
        obj = json.loads(raw)
        if not isinstance(obj, dict):
            return {}
        out = {}
        for k, v in obj.items():
            out[str(k)] = float(v)
        return out
    except Exception:
        return {}


def _robust_p80(values: List[float]) -> float:
    arr = sorted(float(v) for v in values)
    if not arr:
        return 0.0
    idx = int(max(0, min(len(arr) - 1, round(0.8 * (len(arr) - 1)))))
    return float(arr[idx])


def _parse_dynamics_json(path: str) -> List[Dict[str, Any]]:
    if not path:
        return []
    try:
        with open(path, 'r', encoding='utf-8') as f:
            obj = json.load(f)
        if isinstance(obj, dict):
            rows = obj.get('per_pair_estimates', [])
            return rows if isinstance(rows, list) else []
        if isinstance(obj, list):
            return obj
        return []
    except Exception:
        return []


def _median(values: List[float]) -> float:
    if not values:
        return 0.0
    arr = sorted(float(v) for v in values)
    n = len(arr)
    mid = n // 2
    if n % 2 == 1:
        return float(arr[mid])
    return float(0.5 * (arr[mid - 1] + arr[mid]))


def derive_pairing_lookback_sample_rate(
    per_pair_estimates: List[Dict[str, Any]],
    min_lb: int,
    max_lb: int,
    horizon_multiplier: float,
    target_sampled_points: int,
    max_sample_rate: int,
) -> Dict[str, Any]:
    grouped = {}
    for row in per_pair_estimates or []:
        try:
            if not row.get('valid', False):
                continue
            input_name = str(row.get('input', row.get('mv', row.get('dv', '')))).strip()
            input_type = str(row.get('input_type', 'mv')).strip().lower()
            if input_type not in {'mv', 'dv'}:
                input_type = 'mv'
            cv = str(row.get('cv', '')).strip()
            tau = float(row.get('tau', 0.0))
            dead = float(row.get('dead_time', 0.0))
            amp = float(row.get('amplitude', 0.0))
            delta = float(row.get('delta', 0.0))
            if not input_name or not cv or tau < 0.0 or dead < 0.0:
                continue
        except Exception:
            continue
        grouped.setdefault((input_name, input_type, cv), []).append(
            {
                'tau': tau,
                'dead': dead,
                'amp': amp,
                'delta': delta,
            }
        )

    pair_rows = []
    gain_metrics = []
    for (input_name, input_type, cv), rows in grouped.items():
        pos = [r['amp'] for r in rows if r['delta'] > 0]
        neg = [r['amp'] for r in rows if r['delta'] < 0]
        if pos and neg:
            gm = abs(_median(pos) - _median(neg))
        else:
            gm = _median([abs(r['amp']) for r in rows])
        pair_rows.append({'input': input_name, 'input_type': input_type, 'cv': cv, 'rows': rows, 'gain_metric': float(gm)})
        gain_metrics.append(float(gm))

    adaptive_thr = max(0.25, 0.15 * max(gain_metrics) if gain_metrics else 0.25)

    active_pairs = []
    inactive_pairs = []
    for pr in pair_rows:
        rows = pr['rows']
        if pr['gain_metric'] < adaptive_thr:
            inactive_pairs.append({'input': pr['input'], 'input_type': pr['input_type'], 'cv': pr['cv'], 'gain_metric': pr['gain_metric']})
            continue

        pair_lbs = []
        for r in rows:
            lb = _calc_seed_from_process(
                tau_dominant=max(0.0, float(r['tau'])),
                dead_time=max(0.0, float(r['dead'])),
                horizon_multiplier=horizon_multiplier,
            )
            lb = max(min_lb, min(max_lb, int(lb)))
            pair_lbs.append(int(lb))

        if not pair_lbs:
            continue

        longest_lb = int(max(pair_lbs))
        fastest_lb = int(min(pair_lbs))
        active_pairs.append(
            {
                'input': pr['input'],
                'input_type': pr['input_type'],
                'cv': pr['cv'],
                'gain_metric': float(pr['gain_metric']),
                'pair_lookbacks': pair_lbs,
                'longest_lb': longest_lb,
                'fastest_lb': fastest_lb,
            }
        )

    by_cv = {}
    by_mv = {}
    by_dv = {}
    for p in active_pairs:
        cv = p['cv']
        input_name = p['input']
        input_type = p['input_type']
        by_cv.setdefault(cv, {'longest': [], 'fastest': []})
        by_cv[cv]['longest'].append(int(p['longest_lb']))
        by_cv[cv]['fastest'].append(int(p['fastest_lb']))
        if input_type == 'dv':
            by_dv.setdefault(input_name, {'longest': [], 'fastest': []})
            by_dv[input_name]['longest'].append(int(p['longest_lb']))
            by_dv[input_name]['fastest'].append(int(p['fastest_lb']))
        else:
            by_mv.setdefault(input_name, {'longest': [], 'fastest': []})
            by_mv[input_name]['longest'].append(int(p['longest_lb']))
            by_mv[input_name]['fastest'].append(int(p['fastest_lb']))

    longest_by_cv = {k: int(max(v['longest'])) for k, v in by_cv.items()}
    fastest_by_cv = {k: int(min(v['fastest'])) for k, v in by_cv.items()}
    longest_by_mv = {k: int(max(v['longest'])) for k, v in by_mv.items()}
    fastest_by_mv = {k: int(min(v['fastest'])) for k, v in by_mv.items()}
    longest_by_dv = {k: int(max(v['longest'])) for k, v in by_dv.items()}
    fastest_by_dv = {k: int(min(v['fastest'])) for k, v in by_dv.items()}

    scan_by_cv = {
        k: int(
            derive_sample_rate_for_lookback(
                lookback=v,
                target_sampled_points=target_sampled_points,
                max_sample_rate=max_sample_rate,
            )
        )
        for k, v in fastest_by_cv.items()
    }
    scan_by_mv = {
        k: int(
            derive_sample_rate_for_lookback(
                lookback=v,
                target_sampled_points=target_sampled_points,
                max_sample_rate=max_sample_rate,
            )
        )
        for k, v in fastest_by_mv.items()
    }
    scan_by_dv = {
        k: int(
            derive_sample_rate_for_lookback(
                lookback=v,
                target_sampled_points=target_sampled_points,
                max_sample_rate=max_sample_rate,
            )
        )
        for k, v in fastest_by_dv.items()
    }

    return {
        'gain_metric_threshold': float(adaptive_thr),
        'active_pair_count': int(len(active_pairs)),
        'inactive_pair_count': int(len(inactive_pairs)),
        'inactive_pairs': inactive_pairs,
        'active_pairs': active_pairs,
        'lookback_by_cv_raw': longest_by_cv,
        'sample_rate_by_cv': scan_by_cv,
        'lookback_by_mv_raw': longest_by_mv,
        'sample_rate_by_mv': scan_by_mv,
        'lookback_by_dv_raw': longest_by_dv,
        'sample_rate_by_dv': scan_by_dv,
    }


def _derive_process_constants_by_key(per_pair_estimates: List[Dict[str, Any]], key: str) -> Dict[str, Dict[str, float]]:
    by_tau = {}
    by_dead = {}
    for row in per_pair_estimates or []:
        try:
            if not row.get('valid', False):
                continue
            name = str(row.get(key, '')).strip()
            if not name:
                continue
            tau = float(row.get('tau'))
            dead = float(row.get('dead_time'))
            if not (tau > 0.0 and dead >= 0.0):
                continue
        except Exception:
            continue
        by_tau.setdefault(name, []).append(tau)
        by_dead.setdefault(name, []).append(dead)

    tau_map = {k: _robust_p80(v) for k, v in by_tau.items()}
    dead_map = {k: _robust_p80(v) for k, v in by_dead.items()}
    return {
        f'tau_per_{key}': tau_map,
        f'dead_time_per_{key}': dead_map,
    }


def derive_cv_process_constants(per_pair_estimates: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    out = _derive_process_constants_by_key(per_pair_estimates, key='cv')
    return {
        'tau_per_cv': out.get('tau_per_cv', {}),
        'dead_time_per_cv': out.get('dead_time_per_cv', {}),
    }


def derive_mv_process_constants(per_pair_estimates: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    out = _derive_process_constants_by_key(per_pair_estimates, key='mv')
    return {
        'tau_per_mv': out.get('tau_per_mv', {}),
        'dead_time_per_mv': out.get('dead_time_per_mv', {}),
    }


def derive_dv_process_constants(per_pair_estimates: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    out = _derive_process_constants_by_key(per_pair_estimates, key='dv')
    return {
        'tau_per_dv': out.get('tau_per_dv', {}),
        'dead_time_per_dv': out.get('dead_time_per_dv', {}),
    }


def build_feature_lookback_vector(
    sim_meta: Dict[str, Any],
    raw_lookback_by_cv: Dict[str, int],
    raw_lookback_by_mv: Dict[str, int],
    raw_lookback_by_dv: Dict[str, int],
    global_lookback_raw: int,
    global_sample_rate: int,
    target_sampled_points: int,
    max_sample_rate: int,
) -> Dict[str, Any]:
    state_vars = list(sim_meta.get('state_variables', []))
    state_dim = int(sim_meta.get('state_dim', len(state_vars) if state_vars else 0))
    if state_dim <= 0:
        return {}

    cv_indices = [int(i) for i in sim_meta.get('cv_indices', []) if 0 <= int(i) < state_dim]
    mv_indices = [int(i) for i in sim_meta.get('mv_indices', []) if 0 <= int(i) < state_dim]
    dv_indices = [int(i) for i in sim_meta.get('dv_indices', []) if 0 <= int(i) < state_dim]

    max_cv_raw = max(raw_lookback_by_cv.values()) if raw_lookback_by_cv else int(global_lookback_raw)
    raw_vec = [int(global_lookback_raw)] * state_dim

    def _name_map(raw_map: Dict[str, int]) -> Dict[str, int]:
        return {_safe_name(k): int(v) for k, v in (raw_map or {}).items()}

    def _pick_value(name: str, name_map: Dict[str, int]):
        if not name_map:
            return None
        s = _safe_name(name)
        if s in name_map:
            return int(name_map[s])
        for k, v in name_map.items():
            if k and (k in s or s in k):
                return int(v)
        return None

    # Match CV/MV names from metadata to channel-specific lookbacks when possible.
    cv_by_safe = {_safe_name(k): int(v) for k, v in raw_lookback_by_cv.items()}
    mv_by_safe = _name_map(raw_lookback_by_mv)
    dv_by_safe = _name_map(raw_lookback_by_dv)
    fallback_cv_vals = list(raw_lookback_by_cv.values())
    fallback_mv_vals = list(raw_lookback_by_mv.values())
    fallback_dv_vals = list(raw_lookback_by_dv.values())
    for i, sidx in enumerate(cv_indices):
        chosen = None
        if sidx < len(state_vars):
            chosen = _pick_value(state_vars[sidx], cv_by_safe)
        if chosen is None and i < len(fallback_cv_vals):
            chosen = int(fallback_cv_vals[i])
        if chosen is None:
            chosen = max_cv_raw
        raw_vec[sidx] = int(max(1, chosen))

    # MV suggestion: use MV-specific dynamics when available, otherwise max CV memory.
    for i, sidx in enumerate(mv_indices):
        chosen = None
        if sidx < len(state_vars):
            chosen = _pick_value(state_vars[sidx], mv_by_safe)
        if chosen is None and i < len(fallback_mv_vals):
            chosen = int(fallback_mv_vals[i])
        if chosen is None:
            chosen = int(max_cv_raw)
        raw_vec[sidx] = int(max(1, chosen))

    # Disturbance channels use identified DV dynamics when available.
    for i, sidx in enumerate(dv_indices):
        chosen = None
        if sidx < len(state_vars):
            chosen = _pick_value(state_vars[sidx], dv_by_safe)
        if chosen is None and i < len(fallback_dv_vals):
            chosen = int(fallback_dv_vals[i])
        if chosen is None:
            chosen = int(max_cv_raw)
        raw_vec[sidx] = int(max(1, chosen))

    sr_global = max(1, int(global_sample_rate))
    sr_vec = [
        int(
            derive_sample_rate_for_lookback(
                lookback=v,
                target_sampled_points=int(target_sampled_points),
                max_sample_rate=int(max_sample_rate),
            )
        )
        for v in raw_vec
    ]
    sampled_vec = [max(1, int(math.ceil(v / float(sr)))) for v, sr in zip(raw_vec, sr_vec)]
    return {
        'feature_lookback_vector_raw': raw_vec,
        'feature_lookback_vector': sampled_vec,
        'feature_sample_rate_vector': sr_vec,
        'feature_vector_sample_rate': sr_global,
        'feature_vector_policy': 'CV-specific horizons/rates for CVs; MV/feed use max CV horizon/rate for delayed-coupling safety.',
        'feature_vector_cv_lookback_by_name_raw': raw_lookback_by_cv,
        'feature_vector_mv_lookback_by_name_raw': raw_lookback_by_mv,
        'feature_vector_dv_lookback_by_name_raw': raw_lookback_by_dv,
    }


def _dynamics_sample_rate_ceiling(dead_time: float, tau_dominant: float) -> int:
    """Compute the maximum allowable sample_rate from process dynamics.

    Ensures the observer can resolve the dead time and the fastest
    significant time constant.  Rules (process-control Nyquist):
      - sample_rate <= dead_time  (at least one sample per dead-time interval)
      - sample_rate <= 0.1 * tau_dominant  (≥10 samples per dominant τ)
    Returns 1 when either constraint would force sub-1 values.
    """
    constraints = []
    if dead_time > 0:
        constraints.append(dead_time)
    if tau_dominant > 0:
        constraints.append(0.1 * tau_dominant)
    if not constraints:
        return 256  # no dynamics info → no constraint
    return max(1, int(math.floor(min(constraints))))


def derive_sample_rate_for_lookback(
    lookback: int,
    target_sampled_points: int = 12,
    max_sample_rate: int = 256,
    min_sample_rate: int = 1,
) -> int:
    lb = max(1, int(lookback))
    tgt = max(1, int(target_sampled_points))
    min_sr = max(1, int(min_sample_rate))
    max_sr = max(min_sr, int(max_sample_rate))
    scan = int(math.ceil(lb / float(tgt)))
    return max(min_sr, min(max_sr, scan))


def identify_lookback(
    seed: int,
    min_lb: int,
    max_lb: int,
    tau_dominant: float = 75.0,
    dead_time: float = 8.0,
    horizon_multiplier: float = 2.0,
    target_sampled_points: int = 12,
    max_sample_rate: int = 256,
    per_pair_estimates: List[Dict[str, Any]] = None,
    tau_per_cv: Dict[str, float] = None,
    dead_time_per_cv: Dict[str, float] = None,
    tau_per_mv: Dict[str, float] = None,
    dead_time_per_mv: Dict[str, float] = None,
    tau_per_dv: Dict[str, float] = None,
    dead_time_per_dv: Dict[str, float] = None,
    tau_fastest: float = None,
    dead_time_fastest: float = None,
) -> Dict:
    min_lb = max(1, int(min_lb))
    max_lb = max(min_lb, int(max_lb))
    source = 'explicit_seed'
    inferred_seed = None

    # ── Resolve fastest dynamics for scan-rate ceiling ──────────────────
    # The ceiling must be driven by the SMALLEST time constant / dead time
    # in the full transfer-function matrix so the observer can resolve the
    # fastest channel.  tau_dominant (largest) is used for the lookback.
    _tau_fast = tau_fastest
    _dt_fast = dead_time_fastest
    if _tau_fast is None or _dt_fast is None:
        # Attempt to derive from per_pair_estimates.
        _valid_taus = []
        _valid_dts = []
        for row in (per_pair_estimates or []):
            if row.get('valid', False):
                try:
                    _valid_taus.append(float(row['tau']))
                    _valid_dts.append(float(row['dead_time']))
                except (KeyError, ValueError, TypeError):
                    pass
        if _valid_taus:
            _tau_fast = _tau_fast if _tau_fast is not None else min(_valid_taus)
            _dt_fast = _dt_fast if _dt_fast is not None else min(_valid_dts)
        else:
            _tau_fast = _tau_fast if _tau_fast is not None else tau_dominant
            _dt_fast = _dt_fast if _dt_fast is not None else dead_time

    # ── Dynamics-aware scan-rate ceiling ────────────────────────────────
    dynamics_ceiling = _dynamics_sample_rate_ceiling(_dt_fast, _tau_fast)
    effective_max_sr = min(int(max_sample_rate), dynamics_ceiling)

    if seed is None or int(seed) <= 0:
        inferred_seed = _calc_seed_from_process(
            tau_dominant=tau_dominant,
            dead_time=dead_time,
            horizon_multiplier=horizon_multiplier,
        )
        s = max(min_lb, min(max_lb, int(inferred_seed)))
        source = 'process_time_constants'
    else:
        s = max(min_lb, min(max_lb, int(seed)))

    raw = [
        round(s * 0.5),
        round(s * 0.75),
        s,
        round(s * 1.25),
        round(s * 1.5),
        round(s * 2.0),
        round(s * 2.5),
        s - 4,
        s + 4,
    ]
    clipped = [max(min_lb, min(max_lb, int(v))) for v in raw]
    candidates = _dedupe_sorted_int(clipped)
    identified = s
    identified_sample_rate = derive_sample_rate_for_lookback(
        identified,
        target_sampled_points=target_sampled_points,
        max_sample_rate=effective_max_sr,
    )
    candidate_sample_rates = {
        str(lb): derive_sample_rate_for_lookback(
            lb,
            target_sampled_points=target_sampled_points,
            max_sample_rate=effective_max_sr,
        )
        for lb in candidates
    }

    # Build CV/MV-wise lookbacks from per-pair dynamics if available.
    cv_tau = dict(tau_per_cv or {})
    cv_dead = dict(dead_time_per_cv or {})
    mv_tau = dict(tau_per_mv or {})
    mv_dead = dict(dead_time_per_mv or {})
    dv_tau = dict(tau_per_dv or {})
    dv_dead = dict(dead_time_per_dv or {})
    pairwise_info = {}
    pairwise_longest_by_cv = {}
    pairwise_scan_by_cv = {}
    pairwise_longest_by_mv = {}
    pairwise_scan_by_mv = {}
    pairwise_longest_by_dv = {}
    pairwise_scan_by_dv = {}
    if per_pair_estimates:
        cv_derived = derive_cv_process_constants(per_pair_estimates)
        mv_derived = derive_mv_process_constants(per_pair_estimates)
        dv_derived = derive_dv_process_constants(per_pair_estimates)
        cv_tau = {**cv_derived.get('tau_per_cv', {}), **cv_tau}
        cv_dead = {**cv_derived.get('dead_time_per_cv', {}), **cv_dead}
        mv_tau = {**mv_derived.get('tau_per_mv', {}), **mv_tau}
        mv_dead = {**mv_derived.get('dead_time_per_mv', {}), **mv_dead}
        dv_tau = {**dv_derived.get('tau_per_dv', {}), **dv_tau}
        dv_dead = {**dv_derived.get('dead_time_per_dv', {}), **dv_dead}

        pairwise_info = derive_pairing_lookback_sample_rate(
            per_pair_estimates=per_pair_estimates,
            min_lb=min_lb,
            max_lb=max_lb,
            horizon_multiplier=horizon_multiplier,
            target_sampled_points=target_sampled_points,
            max_sample_rate=effective_max_sr,
        )
        pairwise_longest_by_cv = dict(pairwise_info.get('lookback_by_cv_raw', {}))
        pairwise_scan_by_cv = dict(pairwise_info.get('sample_rate_by_cv', {}))
        pairwise_longest_by_mv = dict(pairwise_info.get('lookback_by_mv_raw', {}))
        pairwise_scan_by_mv = dict(pairwise_info.get('sample_rate_by_mv', {}))
        pairwise_longest_by_dv = dict(pairwise_info.get('lookback_by_dv_raw', {}))
        pairwise_scan_by_dv = dict(pairwise_info.get('sample_rate_by_dv', {}))

    raw_by_cv = {}
    sampled_by_cv = {}
    scan_by_cv = {}
    cv_names = sorted(set(cv_tau.keys()) | set(cv_dead.keys()) | set(pairwise_longest_by_cv.keys()))
    for cv_name in cv_names:
        if cv_name in pairwise_longest_by_cv:
            lb_i = int(pairwise_longest_by_cv[cv_name])
            sr_i = int(min(pairwise_scan_by_cv.get(cv_name, effective_max_sr), effective_max_sr))
            if sr_i < 1:
                sr_i = derive_sample_rate_for_lookback(lb_i, target_sampled_points=target_sampled_points, max_sample_rate=effective_max_sr)
        else:
            tau_i = float(cv_tau.get(cv_name, tau_dominant))
            dead_i = float(cv_dead.get(cv_name, dead_time))
            lb_i = max(min_lb, min(max_lb, _calc_seed_from_process(tau_i, dead_i, horizon_multiplier)))
            sr_i = derive_sample_rate_for_lookback(lb_i, target_sampled_points=target_sampled_points, max_sample_rate=effective_max_sr)
        raw_by_cv[cv_name] = int(lb_i)
        scan_by_cv[cv_name] = int(sr_i)
        sampled_by_cv[cv_name] = int(math.ceil(lb_i / float(sr_i)))

    raw_by_mv = {}
    sampled_by_mv = {}
    scan_by_mv = {}
    mv_names = sorted(set(mv_tau.keys()) | set(mv_dead.keys()) | set(pairwise_longest_by_mv.keys()))
    for mv_name in mv_names:
        if mv_name in pairwise_longest_by_mv:
            lb_i = int(pairwise_longest_by_mv[mv_name])
            sr_i = int(min(pairwise_scan_by_mv.get(mv_name, effective_max_sr), effective_max_sr))
            if sr_i < 1:
                sr_i = derive_sample_rate_for_lookback(lb_i, target_sampled_points=target_sampled_points, max_sample_rate=effective_max_sr)
        else:
            tau_i = float(mv_tau.get(mv_name, tau_dominant))
            dead_i = float(mv_dead.get(mv_name, dead_time))
            lb_i = max(min_lb, min(max_lb, _calc_seed_from_process(tau_i, dead_i, horizon_multiplier)))
            sr_i = derive_sample_rate_for_lookback(lb_i, target_sampled_points=target_sampled_points, max_sample_rate=effective_max_sr)
        raw_by_mv[mv_name] = int(lb_i)
        scan_by_mv[mv_name] = int(sr_i)
        sampled_by_mv[mv_name] = int(math.ceil(lb_i / float(sr_i)))

    raw_by_dv = {}
    sampled_by_dv = {}
    scan_by_dv = {}
    dv_names = sorted(set(dv_tau.keys()) | set(dv_dead.keys()) | set(pairwise_longest_by_dv.keys()))
    for dv_name in dv_names:
        if dv_name in pairwise_longest_by_dv:
            lb_i = int(pairwise_longest_by_dv[dv_name])
            sr_i = int(min(pairwise_scan_by_dv.get(dv_name, effective_max_sr), effective_max_sr))
            if sr_i < 1:
                sr_i = derive_sample_rate_for_lookback(lb_i, target_sampled_points=target_sampled_points, max_sample_rate=effective_max_sr)
        else:
            tau_i = float(dv_tau.get(dv_name, tau_dominant))
            dead_i = float(dv_dead.get(dv_name, dead_time))
            lb_i = max(min_lb, min(max_lb, _calc_seed_from_process(tau_i, dead_i, horizon_multiplier)))
            sr_i = derive_sample_rate_for_lookback(lb_i, target_sampled_points=target_sampled_points, max_sample_rate=effective_max_sr)
        raw_by_dv[dv_name] = int(lb_i)
        scan_by_dv[dv_name] = int(sr_i)
        sampled_by_dv[dv_name] = int(math.ceil(lb_i / float(sr_i)))

    max_cv_raw = int(max(raw_by_cv.values()) if raw_by_cv else identified)
    max_cv_sr = int(derive_sample_rate_for_lookback(max_cv_raw, target_sampled_points=target_sampled_points, max_sample_rate=effective_max_sr))
    max_cv_sampled = int(math.ceil(max_cv_raw / float(max_cv_sr)))

    feature_info = {}
    try:
        sim = create_sim(episode_length=10, sample_rate=1, noise_stdv=0.0)
        sim_meta = resolve_sim_metadata(sim)
        feature_info = build_feature_lookback_vector(
            sim_meta=sim_meta,
            raw_lookback_by_cv=raw_by_cv,
            raw_lookback_by_mv=raw_by_mv,
            raw_lookback_by_dv=raw_by_dv,
            global_lookback_raw=int(identified),
            global_sample_rate=int(identified_sample_rate),
            target_sampled_points=int(target_sampled_points),
            max_sample_rate=int(effective_max_sr),
        )
        if feature_info:
            feature_info['feature_vector_state_variables'] = sim_meta.get('state_variables', [])
            feature_info['feature_vector_mv_indices'] = sim_meta.get('mv_indices', [])
            feature_info['feature_vector_cv_indices'] = sim_meta.get('cv_indices', [])
            feature_info['feature_vector_dv_indices'] = sim_meta.get('dv_indices', [])
    except Exception:
        feature_info = {}

    scalar_lookback = int(identified)
    scalar_sample_rate = int(identified_sample_rate)
    try:
        flv_raw = feature_info.get('feature_lookback_vector_raw', None)
        if isinstance(flv_raw, list) and flv_raw:
            scalar_lookback = int(max(int(v) for v in flv_raw))
        fsv = feature_info.get('feature_sample_rate_vector', None)
        if isinstance(fsv, list) and fsv:
            scalar_sample_rate = int(max(1, min(int(v) for v in fsv)))
    except Exception:
        scalar_lookback = int(identified)
        scalar_sample_rate = int(identified_sample_rate)

    scalar_lookback = int(max(min_lb, min(max_lb, scalar_lookback)))
    scalar_sample_rate = int(max(1, min(int(effective_max_sr), scalar_sample_rate)))

    result = {
        'identified_lookback': int(scalar_lookback),
        'identified_sample_rate': int(scalar_sample_rate),
        'identified_sampled_points': int(math.ceil(scalar_lookback / float(scalar_sample_rate))),
        'scalarization_policy': {
            'lookback': 'max(feature_lookback_vector_raw)  — driven by LARGEST tau (slowest dynamics)',
            'sample_rate': 'min(feature_sample_rate_vector)  — driven by SMALLEST tau (fastest dynamics)',
            'lookback_bounds': [int(min_lb), int(max_lb)],
            'sample_rate_bounds': [1, int(effective_max_sr)],
            'dynamics_ceiling': int(dynamics_ceiling),
            'dynamics_ceiling_rule': 'min(dead_time_fastest, 0.1 * tau_fastest)',
            'tau_fastest_used': float(_tau_fast),
            'dead_time_fastest_used': float(_dt_fast),
        },
        'identified_lookback_by_cv_raw': raw_by_cv,
        'identified_lookback_by_cv': sampled_by_cv,
        'identified_sample_rate_by_cv': scan_by_cv,
        'identified_lookback_by_mv_raw': raw_by_mv,
        'identified_lookback_by_mv': sampled_by_mv,
        'identified_sample_rate_by_mv': scan_by_mv,
        'identified_lookback_by_dv_raw': raw_by_dv,
        'identified_lookback_by_dv': sampled_by_dv,
        'identified_sample_rate_by_dv': scan_by_dv,
        'mv_lookback_recommendation_raw': max_cv_raw,
        'mv_lookback_recommendation': max_cv_sampled,
        'candidate_lookbacks': candidates,
        'candidate_sample_rates': candidate_sample_rates,
        'lookback_min': min_lb,
        'lookback_max': max_lb,
        'sample_rate_rule': {
            'target_sampled_points': int(target_sampled_points),
            'max_sample_rate': int(effective_max_sr),
            'dynamics_ceiling': int(dynamics_ceiling),
            'formula': 'sample_rate = clamp(ceil(lookback / target_sampled_points), 1, min(max_sample_rate, dynamics_ceiling))',
        },
        'seed_input': int(seed) if seed is not None else 0,
        'seed_source': source,
        'inferred_seed_from_process': inferred_seed,
        'process_time_constants': {
            'tau_dominant': float(tau_dominant),
            'dead_time': float(dead_time),
            'horizon_multiplier': float(horizon_multiplier),
            'formula': 'ceil(dead_time + horizon_multiplier * tau_dominant)',
        },
        'mv_policy_note': 'Set MV lookback to max CV lookback for robustness to delayed/coupled MV-CV dynamics.',
        'pairing_identification': pairwise_info,
        'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    result.update(feature_info)
    return result


def identify_and_save_lookback(
    seed: int,
    min_lb: int,
    max_lb: int,
    output_path: str,
    metadata: Dict = None,
    tau_dominant: float = 75.0,
    dead_time: float = 8.0,
    horizon_multiplier: float = 2.0,
    target_sampled_points: int = 12,
    max_sample_rate: int = 256,
    per_pair_estimates: List[Dict[str, Any]] = None,
    tau_per_cv: Dict[str, float] = None,
    dead_time_per_cv: Dict[str, float] = None,
    tau_per_mv: Dict[str, float] = None,
    dead_time_per_mv: Dict[str, float] = None,
    tau_per_dv: Dict[str, float] = None,
    dead_time_per_dv: Dict[str, float] = None,
    tau_fastest: float = None,
    dead_time_fastest: float = None,
) -> Dict:
    result = identify_lookback(
        seed=seed,
        min_lb=min_lb,
        max_lb=max_lb,
        tau_dominant=tau_dominant,
        dead_time=dead_time,
        horizon_multiplier=horizon_multiplier,
        target_sampled_points=target_sampled_points,
        max_sample_rate=max_sample_rate,
        per_pair_estimates=per_pair_estimates,
        tau_per_cv=tau_per_cv,
        dead_time_per_cv=dead_time_per_cv,
        tau_per_mv=tau_per_mv,
        dead_time_per_mv=dead_time_per_mv,
        tau_per_dv=tau_per_dv,
        dead_time_per_dv=dead_time_per_dv,
        tau_fastest=tau_fastest,
        dead_time_fastest=dead_time_fastest,
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
    parser = argparse.ArgumentParser(description='Identify and persist physics-informed lookback candidates.')
    parser.add_argument('--seed', type=int, default=0, help='Lookback seed in samples. 0 uses legacy defaults.')
    parser.add_argument('--min', dest='min_lb', type=int, default=4)
    parser.add_argument('--max', dest='max_lb', type=int, default=4096)
    parser.add_argument('--tau-dominant', type=float, default=75.0)
    parser.add_argument('--dead-time', type=float, default=8.0)
    parser.add_argument('--horizon-multiplier', type=float, default=2.0)
    parser.add_argument('--target-sampled-points', type=int, default=12)
    parser.add_argument('--max-scan-rate', type=int, default=256)
    parser.add_argument('--dynamics-json', type=str, default='', help='Optional dynamics JSON containing per_pair_estimates.')
    parser.add_argument('--tau-per-cv-json', type=str, default='', help='Optional JSON dict of {cv_name: tau}.')
    parser.add_argument('--dead-time-per-cv-json', type=str, default='', help='Optional JSON dict of {cv_name: dead_time}.')
    parser.add_argument('--tau-per-mv-json', type=str, default='', help='Optional JSON dict of {mv_name: tau}.')
    parser.add_argument('--dead-time-per-mv-json', type=str, default='', help='Optional JSON dict of {mv_name: dead_time}.')
    parser.add_argument('--tau-per-dv-json', type=str, default='', help='Optional JSON dict of {dv_name: tau}.')
    parser.add_argument('--dead-time-per-dv-json', type=str, default='', help='Optional JSON dict of {dv_name: dead_time}.')
    parser.add_argument('--output', type=str, required=True, help='Output JSON path for identified lookback.')
    args = parser.parse_args()

    result = identify_and_save_lookback(
        seed=args.seed,
        min_lb=args.min_lb,
        max_lb=args.max_lb,
        output_path=args.output,
        tau_dominant=args.tau_dominant,
        dead_time=args.dead_time,
        horizon_multiplier=args.horizon_multiplier,
        target_sampled_points=args.target_sampled_points,
        max_sample_rate=args.max_sample_rate,
        per_pair_estimates=_parse_dynamics_json(args.dynamics_json),
        tau_per_cv=_parse_json_dict(args.tau_per_cv_json),
        dead_time_per_cv=_parse_json_dict(args.dead_time_per_cv_json),
        tau_per_mv=_parse_json_dict(args.tau_per_mv_json),
        dead_time_per_mv=_parse_json_dict(args.dead_time_per_mv_json),
        tau_per_dv=_parse_json_dict(args.tau_per_dv_json),
        dead_time_per_dv=_parse_json_dict(args.dead_time_per_dv_json),
    )

    print(f"Identified lookback: {result['identified_lookback']}")
    print(f"Identified sample_rate: {result['identified_sample_rate']}")
    print(f"Sampled points per window: {result['identified_sampled_points']}")
    print(f"Candidates: {result['candidate_lookbacks']}")
    print(f"Saved: {args.output}")


if __name__ == '__main__':
    main()
