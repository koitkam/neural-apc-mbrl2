"""Validation for trained DreamerV4 controllers.

Design — slim by intent:

  • Reuses the training APCEnv (same disturbance schedule, same objective, same
    setpoint manager) so eval distribution matches training distribution; the
    only difference is a held-out RNG seed and deterministic actor.  This
    eliminates ~2000 lines of channel cataloguing / holdout-profile code from
    the legacy validator (`neural-apc-pytorch/evaluation/validate_latent.py`).

  • Loads ``final.pt`` (or any ``--ckpt``) and runs ``--episodes`` per seed
    over ``--seeds`` seeds (default 3).

  • Records per-step CSV: state, MV, CV, reward (raw + scaled), reward
    components, action bin index.

  • Plots:  CV trajectories with bound bands and disturbance markers, MV
            trajectories with bound bands, per-step reward, cumulative reward.

Usage::

    python -m evaluation.validate \\
        --controller-dir _runs/test_sim_20260429_143935 \\
        --simulation-dir simulation/test_sim          # optional, auto-read

Outputs land in ``<controller-dir>/validation/`` so they sit next to
``train_log.jsonl`` and ``run_plan.json``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

import matplotlib
matplotlib.use('Agg')  # type: ignore
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_run_plan(controller_dir: Path) -> Dict:
    """Look up run_plan.json: in the controller dir, then walk parents.

    The BO workflow writes run_plan.json at the workflow root, while
    controller_dir is typically a per-trial or final/ subfolder.
    """
    for d in [controller_dir, *controller_dir.parents]:
        plan_path = d / 'run_plan.json'
        if plan_path.exists():
            with open(plan_path, 'r') as f:
                return json.load(f)
        # stop at repo root or filesystem root
        if d.name in ('output', '') or d == d.parent:
            break
    return {}


def _resolve_sim_dir(arg: str | None, controller_dir: Path,
                      run_plan: Dict) -> Path:
    repo = Path(__file__).resolve().parent.parent
    if arg:
        p = Path(arg)
        if p.is_absolute() and p.exists():
            return p
        cand = repo / arg
        if cand.exists():
            return cand
        cand2 = repo / 'simulation' / arg
        if cand2.exists():
            return cand2
        raise FileNotFoundError(f'Cannot resolve --simulation-dir: {arg}')
    sim_dir = run_plan.get('simulation_dir')
    if sim_dir and Path(sim_dir).exists():
        return Path(sim_dir)
    raise FileNotFoundError(
        f'Cannot infer simulation_dir from {controller_dir}/run_plan.json — '
        f'pass --simulation-dir explicitly.')


def _ss_gain_rel_errs(pairs: Dict) -> List[float]:
    """``|wm-real|/|real|`` per pair with a non-tiny real SS gain."""
    rel_errs: List[float] = []
    for v in (pairs or {}).values():
        rg = abs(float(v.get('real_ss_gain', 0.0)))
        if rg > 1e-6:
            rel_errs.append(abs(float(v.get('ss_gain_abs_err', 0.0))) / rg)
    return rel_errs


def _merge_observer_gain_gate(gate: Dict, dv_gate: Optional[Dict]) -> Dict:
    """Keep MV-only ``wm_gain_*`` for lineage; AND DV into observer-wide keys.

    P29 printed ``wm_gain_healthy=True`` at MV rel_err=0.10 while DV ss
    was ×0.56. ``wm_gain_pass`` stays MV-only. ``wm_observer_gain_*`` is
    the control-relevant observer verdict (MV and DV both in band).
    No-DV plants copy the MV flags.
    """
    mv_pass = bool(gate.get('wm_gain_pass'))
    mv_healthy = bool(gate.get('wm_gain_healthy'))
    if dv_gate:
        gate['wm_observer_gain_pass'] = (
            mv_pass and bool(dv_gate.get('wm_dv_gain_pass')))
        gate['wm_observer_gain_healthy'] = (
            mv_healthy and bool(dv_gate.get('wm_dv_gain_healthy')))
    else:
        gate['wm_observer_gain_pass'] = mv_pass
        gate['wm_observer_gain_healthy'] = mv_healthy
    return gate


def _gain_status(healthy: bool, passed: bool) -> str:
    if healthy:
        return 'HEALTHY'
    if passed:
        return 'PASS'
    return 'FAIL'


def _dv_gain_gate_from_json(path: Path) -> Optional[Dict]:
    """MV-only ``wm_gain_*`` hid P29 DV ss ×0.56 behind HEALTHY MV rel_err.

    Same thresholds as the MV gate (pass <1.0, healthy <0.35). Does **not**
    change ``wm_gain_pass`` so lineage comparisons stay MV-only.
    """
    if not path.exists():
        return None
    try:
        dv = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    pairs = dv.get('pairs') or {}
    rel_errs = _ss_gain_rel_errs(pairs)
    if not rel_errs:
        return None
    ratios = []
    for v in pairs.values():
        r = v.get('ss_gain_ratio_wm_over_real')
        if r is None:
            continue
        rf = float(r)
        if np.isfinite(rf):
            ratios.append(rf)
    mean_err = float(np.mean(rel_errs))
    gate = {
        'wm_dv_gain_rel_err': mean_err,
        'wm_dv_gain_rel_err_max': float(np.max(rel_errs)),
        'wm_dv_gain_pass': bool(mean_err < 1.0),
        'wm_dv_gain_healthy': bool(mean_err < 0.35),
        'n_dv_pairs': len(rel_errs),
    }
    if ratios:
        gate['wm_dv_ss_ratio_worst'] = float(
            max(ratios, key=lambda r: abs(r - 1.0)))
    return gate


def _episode_disturbance_markers(schedule: List[Dict], sample_rate: int = 1
                                  ) -> List[Dict]:
    """Flatten schedule events into ``(start_step, label)`` markers."""
    out = []
    for ev in (schedule or []):
        try:
            start = int(ev.get('start', 0))
            name = ev.get('name') or ev.get('group') or 'event'
            out.append({'start': start, 'label': str(name)})
        except Exception:
            continue
    return out


# CV smoothness hygiene (all_pass).  Second-difference chatter + CV
# direction flips — NOT MV reversal.  MV chatter is allowed as long as
# the CV stays smooth and can sit on the economic limit.
CV_D2_RMS_GATE = 0.05
CV_REVERSAL_GATE = 0.25
# Informational only (not all_pass).  Fraction of steps outside [lo, hi].
CV_VIOL_FRAC_INFO = 0.02
# Residual *targets* (docs / residual_board), not all_pass gates.
CV_D2_RMS_TARGET = 0.01
CV_REVERSAL_TARGET = 0.10
CV_OPT_HEADROOM_TARGET = 0.15
# R3 return-to-limit (residual, not all_pass).  Distances are / bound
# width; times are a fraction of the event window (that window is 5τ
# when the identifier is present, else a fixed step count).  No
# engineering units, no test_sim magic defaults.
RETURN_BAND_NORMED = 0.05          # same as settle_band: 5% of span
RETURN_LATE_FRAC = 0.2             # last 20% of window ≈ 1τ if window=5τ
RETURN_HOLD_FRAC = 0.1             # hold ≈ 0.5τ to count as reached
CV_RETURN_HEADROOM_TARGET = 0.15   # same shape as opt-headroom target
CV_RETURN_TIME_FRAC_TARGET = 0.4   # back within ~2τ of a 5τ window


def _finite_floats(vals) -> List[float]:
    out: List[float] = []
    for v in vals or []:
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if np.isfinite(f):
            out.append(f)
    return out


def _reversal_rate(series, width: float) -> float:
    """Sign-changes per step on a 0.1%-of-width deadband."""
    col = np.asarray(series, dtype='float64').reshape(-1)
    if col.size < 2:
        return 0.0
    rng = float(width) if np.isfinite(width) and width > 1e-9 else (
        float(np.nanmax(col) - np.nanmin(col)) or 1.0)
    d = np.diff(col)
    deadband = 1e-3 * max(rng, 1e-12)
    signed = np.where(np.abs(d) > deadband, np.sign(d), 0.0)
    nz = signed[signed != 0.0]
    flips = int(np.sum(np.abs(np.diff(nz)) > 1.0)) if nz.size >= 2 else 0
    return float(flips) / float(max(1, len(d)))


def _cv_side_scale_pair(side_scale, k: int) -> Tuple[Optional[float], Optional[float]]:
    """Per-CV (lo, hi) multipliers from ``cv_side_scale`` JSON."""
    if side_scale is None:
        return None, None
    entry = None
    if isinstance(side_scale, dict):
        entry = side_scale.get(f'cv_{k}')
        if entry is None:
            entry = side_scale.get(k)
        if entry is None and str(k) in side_scale:
            entry = side_scale.get(str(k))
    elif isinstance(side_scale, (list, tuple)) and 0 <= k < len(side_scale):
        entry = side_scale[k]
    if isinstance(entry, dict):
        try:
            return float(entry.get('lo', 1.0)), float(entry.get('hi', 1.0))
        except (TypeError, ValueError):
            return None, None
    if isinstance(entry, (list, tuple)) and len(entry) >= 2:
        try:
            return float(entry[0]), float(entry[1])
        except (TypeError, ValueError):
            return None, None
    return None, None


def preferred_cv_side(ep: Dict, k: int) -> Tuple[str, bool]:
    """Which bound is the money limit for CV ``k``.

    1. ``cv_economic_weights[k]``: ``>0`` prefer LO, ``<0`` prefer HI.
    2. Else ``cv_side_scale``: ``hi > lo`` → HI.
    3. Else unknown → nearest-bound gap, ``preferred_unknown=True``.
    """
    econ = ep.get('obj_cv_economic_weights') or []
    if k < len(econ):
        try:
            w = float(econ[k])
        except (TypeError, ValueError):
            w = 0.0
        if abs(w) > 1e-12:
            return ('lo' if w > 0.0 else 'hi', False)
    lo_s, hi_s = _cv_side_scale_pair(ep.get('obj_cv_side_scale'), k)
    if lo_s is not None and hi_s is not None:
        if hi_s > lo_s + 1e-12:
            return ('hi', False)
        if lo_s > hi_s + 1e-12:
            return ('lo', False)
    return ('unknown', True)


def _load_cv_side_scale_fallback() -> Dict:
    """``_coerce_spec`` drops ``cv_side_scale``; read the live JSON."""
    path = os.environ.get('CONTROL_OBJECTIVE_JSON', '')
    if not path:
        return {}
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        side = (data or {}).get('cv_side_scale') or {}
        return side if isinstance(side, dict) else {}
    except Exception:
        return {}


def _episode_objective_meta(env) -> Dict:
    """Attach objective weights so episode metrics stay env-free later."""
    obj_w = getattr(env, 'obj_w', None) or {}
    spec = getattr(env, 'obj_spec', None) or {}
    side = spec.get('cv_side_scale') if isinstance(spec, dict) else None
    if not side:
        side = _load_cv_side_scale_fallback()
    return {
        'obj_mv_economic_weights': list(obj_w.get('mv_economic_weights') or []),
        'obj_cv_economic_weights': list(obj_w.get('cv_economic_weights') or []),
        'obj_cv_side_scale': side or {},
        'obj_cv_violation_weights': list(obj_w.get('cv_violation_weights') or []),
        'obj_cv_violation_weights_lo': list(
            obj_w.get('cv_violation_weights_lo') or []),
        'obj_cv_violation_weights_hi': list(
            obj_w.get('cv_violation_weights_hi') or []),
    }


def limit_gap_normed(y, lo: float, hi: float, side: str) -> np.ndarray:
    """Distance to the economic bound, divided by bound width.

    0 = on the preferred limit.  Positive = inside the band, away from
    that limit.  Negative = past the preferred bound (money-side
    violation).  ``side`` is ``hi`` / ``lo`` / ``unknown`` (nearest).
    """
    arr = np.asarray(y, dtype='float64').reshape(-1)
    width = float(hi) - float(lo)
    if (not np.isfinite(width)) or width <= 1e-12:
        return np.zeros_like(arr)
    if side == 'hi':
        return (float(hi) - arr) / width
    if side == 'lo':
        return (arr - float(lo)) / width
    return np.minimum(arr - float(lo), float(hi) - arr) / width


def return_to_limit_on_window(
    y,
    lo: float,
    hi: float,
    side: str,
    *,
    band: float = RETURN_BAND_NORMED,
    late_frac: float = RETURN_LATE_FRAC,
    hold_frac: float = RETURN_HOLD_FRAC,
) -> Dict[str, float]:
    """Plant-agnostic post-event return to the economic limit.

    Time is a fraction of *this window* (caller sizes the window from τ).
    Headroom is mean late-window gap / bound width on feasible samples.
    """
    arr = np.asarray(y, dtype='float64').reshape(-1)
    n = int(arr.size)
    empty = {
        'cv_return_headroom': float('nan'),
        'cv_return_time_frac': 1.0,
        'cv_return_viol_frac': float('nan'),
        'cv_return_reached': False,
        'cv_return_band_normed': float(band),
    }
    if n < 2:
        return empty
    width = float(hi) - float(lo)
    if (not np.isfinite(width)) or width <= 1e-12:
        return empty
    gap = limit_gap_normed(arr, lo, hi, side)
    viol = (arr > float(hi)) | (arr < float(lo))
    in_band = ((~viol) & (gap >= -1e-12)
               & (gap <= float(band)))
    hold = max(1, int(round(float(hold_frac) * n)))
    run = 0
    reached_at = None
    for i, ok in enumerate(in_band.tolist()):
        if ok:
            run += 1
            if run >= hold:
                reached_at = i - hold + 1
                break
        else:
            run = 0
    time_frac = (1.0 if reached_at is None
                 else float(reached_at) / float(max(1, n)))
    late_n = max(1, int(round(float(late_frac) * n)))
    late_gap = gap[-late_n:]
    late_viol = viol[-late_n:]
    feasible = late_gap[~late_viol]
    if feasible.size:
        head = float(np.mean(np.maximum(feasible, 0.0)))
    else:
        head = float('nan')
    return {
        'cv_return_headroom': head,
        'cv_return_time_frac': float(time_frac),
        'cv_return_viol_frac': float(np.mean(late_viol)),
        'cv_return_reached': reached_at is not None,
        'cv_return_band_normed': float(band),
    }


def _base_cv_lo_hi(ep: Dict, k: int, y: np.ndarray) -> Tuple[float, float]:
    cv_bounds = ep.get('cv_bounds') or []
    b = cv_bounds[k] if k < len(cv_bounds) else None
    if (isinstance(b, (list, tuple)) and len(b) >= 2
            and np.isfinite(b[0]) and np.isfinite(b[1])
            and float(b[1]) > float(b[0])):
        return float(b[0]), float(b[1])
    lo = float(np.nanmin(y)) if y.size else 0.0
    hi = float(np.nanmax(y)) if y.size else 1.0
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return 0.0, 1.0
    return lo, hi


def _metric_get(row: Dict, *keys) -> Optional[float]:
    for k in keys:
        if row is None:
            break
        if k in row and row[k] is not None:
            try:
                f = float(row[k])
            except (TypeError, ValueError):
                continue
            if np.isfinite(f):
                return f
    return None


def _cv_quality_from_kpi_row(row: Dict, *, prefix: str = '') -> Dict:
    """Normalise seed-KPI / episode-metrics rows to CV-quality scalars."""
    p = prefix
    return {
        'seed': row.get('seed') if isinstance(row, dict) else None,
        'd2': _metric_get(row, f'{p}cv_d2_rms_normed', 'cv_d2_rms_normed',
                          f'{p}kpi_cv_d2_rms_normed'),
        'rev': _metric_get(row, f'{p}cv_reversal_rate', 'cv_reversal_rate',
                           f'{p}kpi_cv_reversal_rate'),
        'tv': _metric_get(row, f'{p}cv_tv_per_step_normed',
                          'cv_tv_per_step_normed'),
        'headroom': _metric_get(row, f'{p}cv_opt_headroom', 'cv_opt_headroom'),
        'viol_frac': _metric_get(row, f'{p}cv_viol_frac', 'cv_viol_frac'),
        'viol_depth': _metric_get(row, f'{p}cv_viol_depth_normed',
                                  'cv_viol_depth_normed'),
        'mv_rev': _metric_get(row, f'{p}mv_reversal_rate', 'mv_reversal_rate',
                              f'{p}kpi_mv_reversal_rate'),
        'econ': _metric_get(row, f'{p}economic_score', 'economic_score',
                            f'{p}kpi_economic_score'),
        'iae': _metric_get(row, f'{p}iae_normed_mean', 'iae_normed_mean'),
    }


def _worst_seed_cv_quality(rows: List[Dict]) -> Dict[str, float]:
    """Worst-seed (max) CV chatter / headroom; mean MV reversal (diagnostic)."""
    by_seed: Dict[object, List[Dict]] = {}
    for i, row in enumerate(rows or []):
        seed = row.get('seed')
        if seed is None:
            seed = i
        by_seed.setdefault(seed, []).append(row)

    def _seed_max(key: str) -> List[float]:
        out: List[float] = []
        for grp in by_seed.values():
            vals = _finite_floats([r.get(key) for r in grp])
            if vals:
                out.append(float(max(vals)))
        return out

    def _seed_mean(key: str) -> List[float]:
        out: List[float] = []
        for grp in by_seed.values():
            vals = _finite_floats([r.get(key) for r in grp])
            if vals:
                out.append(float(np.mean(vals)))
        return out

    d2s = _seed_max('d2')
    revs = _seed_max('rev')
    tvs = _seed_max('tv')
    heads = _seed_max('headroom')
    vfracs = _seed_max('viol_frac')
    vdepths = _seed_max('viol_depth')
    mv_revs = _seed_mean('mv_rev')
    econs = _seed_mean('econ')
    return {
        'n_seeds': int(len(by_seed)),
        'cv_d2_rms_normed_worst_seed': (
            float(max(d2s)) if d2s else float('nan')),
        'cv_reversal_rate_worst_seed': (
            float(max(revs)) if revs else float('nan')),
        'cv_tv_per_step_normed_worst_seed': (
            float(max(tvs)) if tvs else float('nan')),
        'cv_opt_headroom_worst_seed': (
            float(max(heads)) if heads else float('nan')),
        'cv_opt_headroom_mean': (
            float(np.mean(_seed_mean('headroom'))) if _seed_mean('headroom')
            else float('nan')),
        'cv_viol_frac_worst_seed': (
            float(max(vfracs)) if vfracs else float('nan')),
        'cv_viol_depth_normed_worst_seed': (
            float(max(vdepths)) if vdepths else float('nan')),
        'mv_reversal_rate_observed': (
            float(np.mean(mv_revs)) if mv_revs else float('nan')),
        'agent_economic_score': (
            float(np.mean(econs)) if econs else float('nan')),
        'has_cv_smoothness': bool(d2s or revs),
    }


def _cv_smooth_pass(worst_d2: float, worst_rev: float,
                    has_cv: bool) -> bool:
    """Missing CV scores do not fail (legacy JSON); present scores gate."""
    if not has_cv:
        return True
    d2_ok = (not np.isfinite(worst_d2)) or (worst_d2 <= CV_D2_RMS_GATE)
    rev_ok = (not np.isfinite(worst_rev)) or (worst_rev <= CV_REVERSAL_GATE)
    return bool(d2_ok and rev_ok)


def control_quality_gates(
    disturbance_records: Optional[List[Dict]] = None,
    seed_metrics: Optional[List[Dict]] = None,
) -> Dict:
    """Paired scripted-disturbance agent vs open-loop baseline.

    Empty records must **not** pass (P49: leftover ``cfg=`` TypeError
    skipped every scripted episode → 0.0 vs 0.0 falsely PASSED
    ``beats_baseline``).  Seed-episode KPIs have no paired baseline, so
    they can fill CV-quality / agent econ for the log but cannot pass
    ``beats_baseline``.

    ``smooth_pass`` is **CV** chatter (d2 RMS + CV reversal), not MV
    reversal.  MV oscillation is allowed; it is recorded as a diagnostic.
    ``cv_opt_headroom`` is a residual, not an ``all_pass`` gate.
    """
    _dr = list(disturbance_records or [])
    out: Dict = {
        'cv_d2_rms_normed_max': CV_D2_RMS_GATE,
        'cv_reversal_rate_max': CV_REVERSAL_GATE,
        'cv_viol_frac_info': CV_VIOL_FRAC_INFO,
        # Obsolete as a gate; kept so old readers still see the key.
        'mv_reversal_rate_max': 0.5,
        'n_scripted_pairs': len(_dr),
        'mv_oscillation_allowed': True,
    }
    quality_rows: List[Dict] = []
    econs: List[float] = []
    base_econs: List[float] = []
    if _dr:
        for r in _dr:
            q = _cv_quality_from_kpi_row(r.get('episode_metrics_agent') or {})
            q['seed'] = r.get('seed')
            quality_rows.append(q)
            if q.get('econ') is not None:
                econs.append(float(q['econ']))
            be = _metric_get(r.get('episode_metrics_baseline') or {},
                             'economic_score')
            if be is not None:
                base_econs.append(float(be))
    elif seed_metrics:
        for r in seed_metrics:
            q = _cv_quality_from_kpi_row(r, prefix='kpi_')
            # Rows already flattened with kpi_*; also try unprefixed.
            if q.get('d2') is None:
                q = _cv_quality_from_kpi_row(r)
            quality_rows.append(q)
            if q.get('econ') is not None:
                econs.append(float(q['econ']))

    agg = _worst_seed_cv_quality(quality_rows)
    worst_d2 = float(agg['cv_d2_rms_normed_worst_seed'])
    worst_rev = float(agg['cv_reversal_rate_worst_seed'])
    worst_vfrac = float(agg['cv_viol_frac_worst_seed'])
    smooth = _cv_smooth_pass(worst_d2, worst_rev, bool(agg['has_cv_smoothness']))
    limit_pass = (
        bool(np.isfinite(worst_vfrac) and worst_vfrac <= CV_VIOL_FRAC_INFO)
        if agg['has_cv_smoothness'] and np.isfinite(worst_vfrac) else True
    )
    agent_econ = (float(np.mean(econs)) if econs
                  else float(agg['agent_economic_score']))
    out.update(agg)
    out.update({
        'cv_d2_rms_normed_observed': worst_d2,
        'cv_reversal_rate_observed': worst_rev,
        'cv_opt_headroom_observed': float(agg['cv_opt_headroom_mean']),
        'cv_viol_frac_observed': worst_vfrac,
        'agent_economic_score': agent_econ,
        'smooth_pass': bool(smooth),
        'cv_smooth_pass': bool(smooth),
        'cv_limit_pass': bool(limit_pass),
    })
    if not _dr:
        out.update({
            'baseline_economic_score': float('nan'),
            'beats_baseline_pass': False,
            'control_gate_skipped': 'no_scripted_disturbance_pairs',
        })
        return out
    base_econ = float(np.mean(base_econs)) if base_econs else 0.0
    out.update({
        'baseline_economic_score': base_econ,
        'beats_baseline_pass': bool(agent_econ >= base_econ),
    })
    return out


def _pair_ss_and_curve(pairs: Optional[Dict]) -> Dict[str, float]:
    """Aggregate TM pair dicts: ss-ratio + curve IAE (mean / worst)."""
    ss: List[float] = []
    iae: List[float] = []
    for v in (pairs or {}).values():
        r = v.get('ss_gain_ratio_wm_over_real')
        try:
            rf = float(r)
        except (TypeError, ValueError):
            rf = float('nan')
        if np.isfinite(rf):
            ss.append(rf)
        c = v.get('curve_iae_normed')
        try:
            cf = float(c)
        except (TypeError, ValueError):
            cf = float('nan')
        if np.isfinite(cf):
            iae.append(cf)
    def _worst_ss(vals: List[float]) -> float:
        if not vals:
            return float('nan')
        return float(max(vals, key=lambda x: abs(x - 1.0)))
    return {
        'ss_ratio_mean': float(np.mean(ss)) if ss else float('nan'),
        'ss_ratio_worst': _worst_ss(ss),
        'curve_iae_mean': float(np.mean(iae)) if iae else float('nan'),
        'curve_iae_worst': float(max(iae)) if iae else float('nan'),
        'n_pairs': int(len(ss) or len(iae)),
    }


def _distpred_amp(dp: Optional[Dict]) -> Dict[str, float]:
    dp = dp or {}
    chs = dp.get('per_channel') or []
    pred = _finite_floats([c.get('pred_std') for c in chs])
    true = _finite_floats([c.get('true_std') for c in chs])
    det = dp.get('mean_pearson_r_detrended')
    try:
        det_f = float(det)
    except (TypeError, ValueError):
        det_f = float('nan')
    return {
        'det_r': det_f if np.isfinite(det_f) else float('nan'),
        'pred_std': float(np.mean(pred)) if pred else float('nan'),
        'true_std': float(np.mean(true)) if true else float('nan'),
    }


def _closed_loop_dr_scores(disturbance_records: Optional[List[Dict]]
                            ) -> Dict[str, float]:
    ratios: List[float] = []
    ovs: List[float] = []
    ret_h: List[float] = []
    ret_t: List[float] = []
    ret_v: List[float] = []
    ret_vs_base: List[float] = []
    n_reached = 0
    n_events = 0
    for r in disturbance_records or []:
        am = r.get('episode_metrics_agent') or {}
        bm = r.get('episode_metrics_baseline') or {}
        ai = _metric_get(am, 'iae_normed_mean')
        bi = _metric_get(bm, 'iae_normed_mean')
        if ai is not None and bi is not None and abs(bi) > 1e-12:
            ratios.append(float(ai) / float(bi))
        ev = r.get('event_response') or {}
        ov = ((ev.get('overshoot_normed') or {}).get('p90'))
        ovf = _metric_get({'p90': ov}, 'p90')
        if ovf is not None:
            ovs.append(ovf)
        rh = _metric_get(ev.get('return_headroom') or {}, 'mean', 'max')
        rt = _metric_get(ev.get('return_time_frac') or {}, 'mean', 'max')
        rv = _metric_get(ev.get('return_viol_frac') or {}, 'mean', 'max')
        if rh is not None:
            ret_h.append(rh)
        if rt is not None:
            ret_t.append(rt)
        if rv is not None:
            ret_v.append(rv)
        try:
            n_reached += int(ev.get('n_return_reached') or 0)
            n_events += int(ev.get('n_return_events') or 0)
        except (TypeError, ValueError):
            pass
        bev = r.get('event_response_baseline') or {}
        bh = _metric_get(bev.get('return_headroom') or {}, 'mean', 'max')
        if rh is not None and bh is not None and abs(bh) > 1e-3:
            ret_vs_base.append(float(rh) / float(bh))
    return {
        'iae_agent_over_baseline_mean': (
            float(np.mean(ratios)) if ratios else float('nan')),
        'iae_agent_over_baseline_worst': (
            float(max(ratios)) if ratios else float('nan')),
        'overshoot_p90_mean': float(np.mean(ovs)) if ovs else float('nan'),
        'n_pairs': int(len(ratios)),
        'cv_return_headroom_mean': (
            float(np.mean(ret_h)) if ret_h else float('nan')),
        'cv_return_headroom_worst': (
            float(max(ret_h)) if ret_h else float('nan')),
        'cv_return_time_frac_mean': (
            float(np.mean(ret_t)) if ret_t else float('nan')),
        'cv_return_time_frac_worst': (
            float(max(ret_t)) if ret_t else float('nan')),
        'cv_return_viol_frac_mean': (
            float(np.mean(ret_v)) if ret_v else float('nan')),
        'cv_return_headroom_vs_baseline_mean': (
            float(np.mean(ret_vs_base)) if ret_vs_base else float('nan')),
        'n_return_reached': int(n_reached),
        'n_return_events': int(n_events),
    }


def build_residual_board(
    *,
    fidelity_gates: Optional[Dict] = None,
    mv_tf: Optional[Dict] = None,
    dv_tf: Optional[Dict] = None,
    postprior: Optional[Dict] = None,
    distpred: Optional[Dict] = None,
    disturbance_records: Optional[List[Dict]] = None,
    seed_metrics: Optional[List[Dict]] = None,
) -> Dict:
    """Standing R1/R2/R3 board.  Never 'closed' because a knob family died."""
    fg = fidelity_gates or {}
    mv_agg = _pair_ss_and_curve((mv_tf or {}).get('pairs'))
    dv_agg = _pair_ss_and_curve((dv_tf or {}).get('pairs'))
    pp = postprior or {}
    try:
        compound = float(pp.get('decomp_1step_to_openloop'))
    except (TypeError, ValueError):
        compound = float('nan')
    amp = _distpred_amp(distpred)
    dr = _closed_loop_dr_scores(disturbance_records)
    cq = control_quality_gates(disturbance_records, seed_metrics=seed_metrics)
    return {
        'never_retire_because_family_closed': True,
        'mv_oscillation_allowed': True,
        'refactors_allowed': True,
        'plan_while_live': True,
        'overall_goal': (
            'Smooth CV on the economic limit without violating; faithful '
            'observer TM; unmeasured-load rejection with an aggressive '
            'return to that limit after a disturbance.  Neural observer + '
            'neural Kalman/DOB + neural actor-critic.'
        ),
        'history_files': [
            'docs/GOAL_PLAN.md',
            'docs/RUN_HISTORY.md',
        ],
        'note': (
            'MV chatter is allowed.  Fail/optimize on CV smoothness + '
            'limit hugging + TM shape + unmeasured DR.  Compare these '
            'scores to current champions in docs/RUN_HISTORY.md and to '
            'docs/GOAL_PLAN.md — do not freeze a past run as the lock.  '
            'VALID 9/9 / GAIN-READY / family-closed do not close R1/R2/R3.  '
            'Do not require beating the current econ champion to KEEP a '
            'CV-smoothness, headroom, or DR win.  Bigger '
            'observer/Kalman/actor-critic/loss/gate/metric refactors are '
            'in-scope when knob N+1 cannot serve the overall goal.  While '
            'a run is LIVE, analyze and update docs/GOAL_PLAN.md; do not '
            'start a second GPU job.'
        ),
        'metric_audit': {
            'required_every_exit': True,
            'also_during_live_analysis': True,
            'question': (
                'If this metric improved, would a smooth CV actually sit '
                'closer to the economic limit with a faithful observer and '
                'unmeasured-load rejection?'
            ),
            'known_insufficient': [
                'mv_reversal as smoothness gate (diagnostic only)',
                'cv_tv as smoothness gate (slow ride to the limit is good)',
                'ss-ratio alone (DC freeze-gate; also score curve_iae)',
                'jsonl teacher gain x1 (tautology vs val TM)',
                'critic_r Pearson without critic_rew_to_tgt_var',
                'raw disturbance R2 (use det_r + pred_std vs true)',
                'event IAE without cv_return_headroom / cv_return_time_frac',
                'VALID 9/9 / GAIN-READY / all_pass / family-closed as residual-closed',
                'requiring current econ champion to KEEP a smoothness/headroom/DR win',
            ],
            'if_no': (
                'Do not spend GPU improving it. Fix or replace the metric / '
                'loss / gate first; that work may be a large refactor. '
                'Write the conclusion in docs/GOAL_PLAN.md.'
            ),
        },
        'quality_targets_not_all_pass': {
            'cv_d2_rms_normed': CV_D2_RMS_TARGET,
            'cv_reversal_rate': CV_REVERSAL_TARGET,
            'cv_opt_headroom': CV_OPT_HEADROOM_TARGET,
            'cv_viol_frac': 0.0,
            'tm_curve_iae_normed': 0.0,
            'tm_ss_ratio': 1.0,
            'kalman_amp_ratio': 1.0,
            'cv_return_headroom': CV_RETURN_HEADROOM_TARGET,
            'cv_return_time_frac': CV_RETURN_TIME_FRAC_TARGET,
        },
        'r1_observer_tm': {
            'residual': 'Observer transfer-matrix shape + gain',
            'mv_ss_ratio_mean': mv_agg['ss_ratio_mean'],
            'mv_ss_ratio_worst': mv_agg['ss_ratio_worst'],
            'mv_curve_iae_mean': mv_agg['curve_iae_mean'],
            'mv_curve_iae_worst': mv_agg['curve_iae_worst'],
            'dv_ss_ratio_mean': dv_agg['ss_ratio_mean'],
            'dv_ss_ratio_worst': dv_agg['ss_ratio_worst'],
            'dv_curve_iae_mean': dv_agg['curve_iae_mean'],
            'dv_curve_iae_worst': dv_agg['curve_iae_worst'],
            'compound_1step_to_openloop': compound,
            'compare_to': 'docs/RUN_HISTORY.md BEST-RUN BASELINES and docs/GOAL_PLAN.md',
        },
        'r2_cv_quality': {
            'residual': (
                'CV must stay smooth and close to the economic limit '
                'without violating.  MV oscillation is allowed; mid-band '
                'CV is lost optimization potential.'
            ),
            'cv_d2_rms_normed_worst_seed': cq.get(
                'cv_d2_rms_normed_worst_seed'),
            'cv_reversal_rate_worst_seed': cq.get(
                'cv_reversal_rate_worst_seed'),
            'cv_tv_per_step_normed_worst_seed': cq.get(
                'cv_tv_per_step_normed_worst_seed'),
            'cv_opt_headroom_mean': cq.get('cv_opt_headroom_mean'),
            'cv_opt_headroom_worst_seed': cq.get(
                'cv_opt_headroom_worst_seed'),
            'cv_viol_frac_worst_seed': cq.get('cv_viol_frac_worst_seed'),
            'cv_viol_depth_normed_worst_seed': cq.get(
                'cv_viol_depth_normed_worst_seed'),
            'mv_reversal_rate_observed': cq.get('mv_reversal_rate_observed'),
            'cv_smooth_pass': cq.get('cv_smooth_pass'),
            'cv_limit_pass': cq.get('cv_limit_pass'),
            'smooth_pass_is_cv_only': True,
        },
        'r3_unmeasured_dr': {
            'residual': (
                'Kalman pred_std vs true + det_r; closed-loop event IAE '
                'agent/baseline; CV returns to the economic limit after '
                'the event (cv_return_headroom / cv_return_time_frac), '
                'not a mid-band settle.'
            ),
            'kalman_det_r': amp['det_r'],
            'kalman_pred_std': amp['pred_std'],
            'kalman_true_std': amp['true_std'],
            'iae_agent_over_baseline_mean': dr['iae_agent_over_baseline_mean'],
            'iae_agent_over_baseline_worst': dr[
                'iae_agent_over_baseline_worst'],
            'overshoot_p90_mean': dr['overshoot_p90_mean'],
            'cv_return_headroom_mean': dr['cv_return_headroom_mean'],
            'cv_return_headroom_worst': dr['cv_return_headroom_worst'],
            'cv_return_time_frac_mean': dr['cv_return_time_frac_mean'],
            'cv_return_time_frac_worst': dr['cv_return_time_frac_worst'],
            'cv_return_viol_frac_mean': dr['cv_return_viol_frac_mean'],
            'cv_return_headroom_vs_baseline_mean': dr[
                'cv_return_headroom_vs_baseline_mean'],
            'n_return_reached': dr['n_return_reached'],
            'n_return_events': dr['n_return_events'],
            'not_all_pass': True,
            'compare_to': 'docs/RUN_HISTORY.md BEST-RUN BASELINES and docs/GOAL_PLAN.md',
        },
        'fidelity_all_pass': fg.get('all_pass'),
        'beats_baseline_pass': fg.get('beats_baseline_pass',
                                      cq.get('beats_baseline_pass')),
    }


def _jsonable(x):
    """JSON dump helper: numpy / NaN → list / None."""
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return _jsonable(x.tolist())
    if isinstance(x, (np.floating, float)):
        xf = float(x)
        return xf if np.isfinite(xf) else None
    if isinstance(x, (np.integer, int)) and not isinstance(x, bool):
        return int(x)
    if isinstance(x, (np.bool_, bool)):
        return bool(x)
    return x


# ---------------------------------------------------------------------------
# Scripted disturbance schedule (deterministic, for rejection plots)
# ---------------------------------------------------------------------------

def build_scripted_disturbance_schedule(env, *, n_events: int = 0,
                                         magnitude_frac: float = 0.0,
                                         profile: str = 'holdout_a',
                                         seed: int = 2026,
                                         ) -> List[Dict]:
    """Build a plant-aware validation disturbance schedule.

    Mirrors ``neural-apc-pytorch``'s ``build_output_disturbance_schedule``
    (single ``holdout_a`` profile) but simplified for V4: no profile
    branching, no extra knobs.  The schedule is realistic — magnitudes
    auto-adapt to channel spans, identified plant dynamics (τ, dead-time,
    DV→CV gains) and the MV→CV authority budget — so the controller is
    asked to reject only physically-rejectable disturbances.

    Mix of event types:
      * **measured DV**  — first event guaranteed; sized to deliver a
        target CV impact via the identified DV→CV gain.
      * **unmeasured CV** — second event guaranteed; pushes a CV directly,
        with violation events allowed outside the bound band so the
        controller must drive recovery.

    Each event is a one-shot step held for the rest of the episode.
    Output is a list of dicts compatible with ``env._schedule`` and
    ``utils.training_disturbance.apply_disturbance_schedule``.

    The ``n_events``/``magnitude_frac`` arguments are accepted for
    backward compatibility only — they are ignored in favour of the
    plant-aware draw.
    """
    from utils.training_disturbance import (
        _channel_catalog, _load_identifier_context,
        compute_mv_authority_to_cv, get_authority_target_frac,
        clamp_event_to_authority_budget,
    )
    rng = np.random.default_rng(int(seed))

    catalog = _channel_catalog(env.sim)
    dv_targets = list(catalog.get('dv', []))
    cv_targets = list(catalog.get('cv', []))
    if not dv_targets and not cv_targets:
        return []

    id_ctx = _load_identifier_context()
    lookback = int((id_ctx.get('lookback', {}) or {}).get('identified_lookback', 0) or 0)
    tau_dom = float((id_ctx.get('dynamics', {}) or {}).get('tau_dominant_identified', 0.0) or 0.0)
    dead_time = float((id_ctx.get('dynamics', {}) or {}).get('dead_time_identified', 0.0) or 0.0)
    dv_gain = id_ctx.get('dv_gain_to_cv', {}) if isinstance(id_ctx, dict) else {}

    ep_len = int(env.cfg.episode_length)
    dyn_horizon = max(1.0, dead_time + tau_dom)
    settle = int(max(40.0, round(max(0.6 * float(lookback), 1.25 * dyn_horizon))))
    min_gap = max(14, int(0.95 * settle))

    # holdout_a profile (validation default).  Only measured-DV events
    # are scripted — the unmeasured CV disturbance is now provided
    # automatically by the hidden OU process attached to the env
    # (``env._hidden_disturbance`` with ``_hidden_disturbance_force=True``),
    # which fires on every validation episode.
    n_total = int(max(3, min(14, ep_len // max(36, settle))))
    p_violation = 0.45

    earliest = max(8, int(0.06 * ep_len))
    latest = max(earliest + 1, int(0.92 * ep_len))
    starts: List[int] = []
    for _ in range(max(50, 15 * n_total)):
        if len(starts) >= n_total:
            break
        s = int(rng.integers(earliest, latest + 1))
        if all(abs(s - s0) >= min_gap for s0 in starts):
            starts.append(s)
    starts.sort()
    if not starts:
        starts = [earliest]

    cv_widths = []
    for ch in cv_targets:
        b = ch.get('bounds')
        if isinstance(b, list) and len(b) >= 2:
            cv_widths.append(max(1e-6, float(b[1]) - float(b[0])))
    cv_span_ref = (float(np.median(np.asarray(cv_widths, dtype='float64')))
                    if cv_widths else 10.0)

    mv_authority_cv = compute_mv_authority_to_cv(env.sim, id_ctx)
    # P49 leftover race: HEAD validate.py passed cfg= while the live pid
    # still had launch-time get_authority_target_frac(default=) only.
    # TypeError skipped every scripted-disturbance episode → empty
    # paired records → 0.0 vs 0.0 falsely PASSED beats_baseline.
    try:
        authority_frac = get_authority_target_frac(
            cfg=getattr(env, 'cfg', None))
    except TypeError:
        authority_frac = get_authority_target_frac()
    cumulative_offset: Dict[str, float] = {}
    cumulative_cv_impact = 0.0

    schedule: List[Dict] = []
    if not dv_targets:
        return schedule
    for i, start in enumerate(starts):
        target = dv_targets[int(rng.integers(0, len(dv_targets)))]
        source, target_group = 'measured_dv', 'dv'

        is_violation = bool(rng.uniform() < p_violation)
        intent = 'violation' if is_violation else 'economic'

        # Direction: anti-drift bias when this channel has accumulated
        # > 15% of its span in one direction.
        ch_key = f"{target_group}_{int(target.get('pos', 0))}"
        cum = cumulative_offset.get(ch_key, 0.0)
        b_ref = target.get('bounds')
        ch_span = (max(1e-6, float(b_ref[1]) - float(b_ref[0]))
                   if isinstance(b_ref, list) and len(b_ref) >= 2 else 1.0)
        if abs(cum) / ch_span > 0.15 and abs(cum) > 1e-9:
            sign = -1.0 if cum > 0 else 1.0
            if rng.uniform() < 0.20:
                sign = -sign
        else:
            sign = -1.0 if rng.uniform() < 0.5 else 1.0

        span = ch_span
        frac = float(rng.uniform(0.05, 0.16) if not is_violation
                     else rng.uniform(0.18, 0.38))
        mag = sign * frac * span
        gain = float(dv_gain.get(str(target.get('name', '')),
                                  dv_gain.get(f"dv_{int(target.get('pos', 0))}", 0.0))
                      or 0.0)
        if gain > 1e-8:
            desired_cv = float(rng.uniform(0.08, 0.18) if not is_violation
                                else rng.uniform(0.30, 0.55)) * cv_span_ref
            needed = desired_cv / gain
            mag = sign * max(abs(mag), abs(needed))
        allow_oob = False
        cv_per_unit = abs(gain) if gain > 1e-8 else 0.0

        # Authority-budget clip (shared with training).
        if cv_per_unit > 0.0 and mv_authority_cv > 1e-9 and authority_frac > 0.0:
            new_delta, achieved = clamp_event_to_authority_budget(
                proposed_delta=float(mag),
                cv_impact_per_unit=float(cv_per_unit),
                cumulative_cv_impact=float(cumulative_cv_impact),
                mv_authority_cv=float(mv_authority_cv),
                target_frac=float(authority_frac),
                cfg=getattr(env, 'cfg', None),
            )
            mag = float(new_delta)
            cumulative_cv_impact += float(achieved)

        if abs(mag) < 1e-6:
            continue

        color = '#e76f51' if is_violation else '#2a9d8f'
        ch_name = str(target.get('name', f'{target_group}_{int(target.get("pos", 0))}'))
        schedule.append({
            'name': f'{source}_{ch_name}_{intent}_{i + 1}',
            'start': int(start),
            'target_group': target_group,
            'target_pos': int(target.get('pos', 0)),
            'target_state_index': int(target.get('state_index', 0)),
            'target_name': ch_name,
            'source': source,
            'intent': intent,
            'delta': float(mag),
            'allow_out_of_bounds': bool(allow_oob),
            'color': color,
            'duration': int(ep_len - int(start)),
            'shape': 'step',
            'period': float(max(2.0, ep_len)),
            '_applied': False,
            '_is_violation': bool(is_violation),
        })
        cumulative_offset[ch_key] = cum + float(mag)

    schedule.sort(key=lambda x: int(x.get('start', 0)))
    return schedule


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_episode(env, model, device, *, deterministic: bool, seed_offset: int = 0
                ) -> Dict:
    """Run one full episode under the trained actor (deterministic by default).

    Returns a dict of arrays + metadata for plotting and metrics.
    """
    obs_window = env.reset(exploration=False)
    schedule = list(env._schedule)
    return _run_episode_with_window(env, model, device, obs_window, schedule,
                                     deterministic=deterministic)


def run_scripted_episode(env, model, device, *, deterministic: bool,
                          schedule: List[Dict]) -> Dict:
    """Run an episode where the disturbance schedule is replaced by
    ``schedule`` (typically the deterministic one from
    ``build_scripted_disturbance_schedule``).  Used for the
    disturbance-rejection plot.
    """
    obs_window = env.reset(exploration=False)
    # Override the schedule the env built in reset() with our scripted one.
    env._schedule = list(schedule)
    return _run_episode_with_window(env, model, device, obs_window,
                                     env._schedule,
                                     deterministic=deterministic)


def _run_episode_with_window(env, model, device, obs_window, schedule, *,
                              deterministic: bool) -> Dict:
    T = env.cfg.episode_length
    state_dim = env.state_dim
    action_dim = env.action_dim

    states = np.zeros((T, state_dim), dtype='float32')
    actions_norm = np.zeros((T, action_dim), dtype='float32')
    controls = np.zeros((T, action_dim), dtype='float32')
    raw_rewards = np.zeros(T, dtype='float32')
    scaled_rewards = np.zeros(T, dtype='float32')
    cv_violations = np.zeros(T, dtype='float32')
    mv_violations = np.zeros(T, dtype='float32')
    # Per-step active MV/CV bounds and CV targets so the plot can
    # render the bound-change schedule the operator/agent saw.  The
    # base bounds (constant) are recorded separately in the rollout
    # dict; these are the *current* values that vary across the
    # episode whenever ``RuntimeSetpointManager`` schedules a change.
    n_mv_aux = int(getattr(env.setpoint_mgr, 'n_mv', 0))
    n_cv_aux = int(getattr(env.setpoint_mgr, 'n_cv', 0))
    current_mv_bounds_t = np.zeros((T, n_mv_aux, 2), dtype='float32')
    current_cv_bounds_t = np.zeros((T, n_cv_aux, 2), dtype='float32')
    current_cv_targets_t = np.zeros((T, n_cv_aux), dtype='float32')
    # Per-step hidden (unmeasured) OU disturbance offset injected into each
    # CV channel (aligned to ``env.cv_indices``).  Surfaces the otherwise-
    # invisible disturbance the agent had to reject.
    n_cv_h = len(env.cv_indices)
    hidden_dist_t = np.zeros((T, n_cv_h), dtype='float32')

    # V4 streaming inference: maintain a rolling action history alongside
    # the env-provided observation window. At each step we encode the
    # window through the tokenizer, run the dynamics transformer with
    # context-noise corruption (τ = 1 − τ_ctx), and read the agent-register
    # hidden state at the latest position to feed the policy head.
    cfg = env.cfg
    L = cfg.lookback
    a_history = np.zeros((L, action_dim), dtype='float32')
    d_min = 1.0 / cfg.k_max
    tau_ctx_val = 1.0 - cfg.tau_ctx

    # 'tssm' (transformer-SSM) shares the RSSM interface (initial_state/obs_step
    # /img_step/decode), so it uses the same open-loop rollout path here.
    _is_rssm = getattr(model, 'world_model_type', 'sf_transformer') in ('rssm', 'tssm')
    _rssm_state = (model.dynamics.initial_state(1, device)
                   if _is_rssm else None)
    _rssm_prev_a = (torch.zeros(1, action_dim, device=device)
                    if _is_rssm else None)
    _serve_step = None
    _o = None
    _o_host = None
    _get_serve_cg = None
    if _is_rssm:
        from models.dreamer_v4_rssm import (
            alloc_pinned_obs_host, copy_obs_row,
            stream_serve_step as _serve_step,
            get_collect_serve_cuda_graph as _get_serve_cg)
        _o = torch.empty(1, int(obs_window.shape[-1]), device=device,
                         dtype=torch.float32)
        _o_host = alloc_pinned_obs_host(device, int(obs_window.shape[-1]))

    _use_cuda_amp = (device.type == 'cuda')
    with torch.inference_mode(), torch.amp.autocast(
            device_type=device.type, dtype=torch.bfloat16,
            enabled=_use_cuda_amp):
        # Reuse the P3 collect graph when present (same bf16 autocast).
        # CPU / TSSM / capture-fail stay eager. ``copy_`` into static
        # prev_a (no per-step rebind).
        _serve_cg = None
        if _is_rssm and _rssm_state is not None and _get_serve_cg is not None:
            _serve_cg = _get_serve_cg(
                model.dynamics, _rssm_state, device,
                int(obs_window.shape[-1]), int(action_dim))
            if _serve_cg is not None:
                _serve_cg.reset(_rssm_state)
        for t in range(T):
            if _is_rssm:
                # Certainty-equivalent belief + the same DV/Kalman feat
                # ``collect_episode`` / ``_realsim_actor_critic_step`` use.
                if _serve_cg is not None:
                    copy_obs_row(_serve_cg.obs, obs_window[-1], _o_host)
                    agent_hid = _serve_cg.replay()
                else:
                    copy_obs_row(_o, obs_window[-1], _o_host)
                    _rssm_state = _serve_step(
                        model.dynamics, _rssm_state, _rssm_prev_a, _o,
                        sample=False)
                    agent_hid = _rssm_state.feat
            else:
                ow = torch.from_numpy(obs_window).to(device)
                a_ctx = torch.from_numpy(a_history).to(device)
                z_ctx = model.tokenizer.encode(ow).unsqueeze(0)
                tau = torch.full((1, L), tau_ctx_val, device=device,
                                  dtype=z_ctx.dtype)
                d = torch.full((1, L), d_min, device=device,
                                dtype=z_ctx.dtype)
                out = model.dynamics(z_ctx, tau, d, a_ctx.unsqueeze(0))
                agent_hid = out['agent_hid'][:, -1]
            action_t, _, _ = model.policy(agent_hid,
                                            deterministic=deterministic)
            a_np = action_t.float().squeeze(0).cpu().numpy().astype('float32')
            if _is_rssm:
                _prev = (_serve_cg.prev_a if _serve_cg is not None
                         else _rssm_prev_a)
                _prev.copy_(
                    action_t.detach().to(dtype=_prev.dtype).reshape(1, -1))
            next_window, scaled_r, done, info = env.step(a_np)
            comps = info.get('reward_components', {}) or {}
            # Record the *raw* (physical-units) state so plots/npz read true
            # plant values, not the post-standardizer z-scores that the
            # tokenizer sees.  Falls back to the normalized obs slice if the
            # env did not expose ``raw_state`` for back-compat.
            raw_st = info.get('raw_state')
            if raw_st is None:
                states[t] = next_window[-1, :state_dim]
            else:
                arr = np.asarray(raw_st, dtype='float32').reshape(-1)
                states[t, :min(state_dim, arr.shape[0])] = arr[:state_dim]
            actions_norm[t] = a_np
            controls[t] = np.asarray(env._prev_control, dtype='float32')
            raw_rewards[t] = float(info.get('raw_reward', 0.0))
            scaled_rewards[t] = float(scaled_r)
            cv_violations[t] = float(comps.get('cv_violation_penalty', 0.0))
            mv_violations[t] = float(comps.get('mv_violation_penalty', 0.0))
            hd = info.get('hidden_disturbance')
            if hd is not None:
                hd = np.asarray(hd, dtype='float32').reshape(-1)
                hidden_dist_t[t, :min(n_cv_h, hd.shape[0])] = hd[:n_cv_h]
            if n_mv_aux > 0:
                current_mv_bounds_t[t] = np.asarray(
                    env.setpoint_mgr.current_mv_bounds, dtype='float32')
            if n_cv_aux > 0:
                current_cv_bounds_t[t] = np.asarray(
                    env.setpoint_mgr.current_cv_bounds, dtype='float32')
                current_cv_targets_t[t] = np.asarray(
                    env.setpoint_mgr.current_cv_targets, dtype='float32')
            if not _is_rssm:
                a_history = np.concatenate(
                    [a_history[1:], a_np[None, :]], axis=0)
            obs_window = next_window
            if done:
                break

    return {
        'states': states[:t + 1],
        'actions_norm': actions_norm[:t + 1],
        'controls': controls[:t + 1],
        'raw_rewards': raw_rewards[:t + 1],
        'scaled_rewards': scaled_rewards[:t + 1],
        'cv_violations': cv_violations[:t + 1],
        'mv_violations': mv_violations[:t + 1],
        'cum_reward': float(np.cumsum(scaled_rewards[:t + 1])[-1]),
        'cum_raw_reward': float(np.cumsum(raw_rewards[:t + 1])[-1]),
        'mean_cv_violation': float(cv_violations[:t + 1].mean()),
        'mean_mv_violation': float(mv_violations[:t + 1].mean()),
        'schedule': schedule,
        'episode_length': int(t + 1),
        'sample_rate': int(env.cfg.sample_rate),
        'cv_indices': list(env.cv_indices),
        'mv_indices': [int(x) for x in (env.meta.get('mv_indices') or [])],
        'dv_indices': [int(x) for x in (env.meta.get('dv_indices') or [])],
        'state_variables': list(env.meta.get('state_variables') or []),
        'mv_norm_ranges': [list(b) for b in env.mv_norm_ranges],
        'cv_norm_ranges': [list(b) for b in env.cv_norm_ranges],
        'mv_bounds': [list(b) for b in
                       getattr(env.setpoint_mgr, 'base_mv_bounds', np.zeros((0, 2)))],
        'cv_bounds': [list(b) for b in
                       getattr(env.setpoint_mgr, 'base_cv_bounds', np.zeros((0, 2)))],
        'cv_targets': [float(x) for x in
                        getattr(env.setpoint_mgr, 'base_cv_targets', [])],
        'cv_target_enabled': [bool(x) for x in
                                getattr(env.setpoint_mgr,
                                          'cv_target_enabled', [])],
        'current_mv_bounds_t': current_mv_bounds_t[:t + 1],
        'current_cv_bounds_t': current_cv_bounds_t[:t + 1],
        'current_cv_targets_t': current_cv_targets_t[:t + 1],
        'hidden_disturbance_t': hidden_dist_t[:t + 1],
        'reward_scale': float(env.reward_scale),
        **_episode_objective_meta(env),
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _add_disturbance_markers(ax, schedule: List[Dict], color='red', alpha=0.20):
    for ev in (schedule or []):
        try:
            x = float(ev.get('start', 0))
        except Exception:
            continue
        ax.axvline(x, color=color, linestyle='--', linewidth=0.7, alpha=alpha)


# ---------------------------------------------------------------------------
# Episode metrics  (process-control standard, simulator-agnostic)
# ---------------------------------------------------------------------------

def _cv_active_target(ep: Dict, cv_row: int) -> np.ndarray | None:
    """Per-step target for CV ``cv_row`` if a target is enabled, else None.

    Returns a (T,) array — uses the time-varying ``current_cv_targets_t``
    when the setpoint manager scheduled changes, else falls back to the
    constant ``cv_targets`` value.
    """
    enabled = ep.get('cv_target_enabled') or []
    if cv_row >= len(enabled) or not bool(enabled[cv_row]):
        return None
    cur_raw = ep.get('current_cv_targets_t')
    cur = (np.asarray(cur_raw, dtype='float32')
            if cur_raw is not None and len(cur_raw) > 0
            else np.zeros((0, 0), dtype='float32'))
    if cur.ndim == 2 and cv_row < cur.shape[1] and np.isfinite(cur[:, cv_row]).any():
        return cur[:, cv_row].astype('float32')
    base = ep.get('cv_targets') or []
    if cv_row < len(base) and np.isfinite(base[cv_row]):
        T = ep['episode_length']
        return np.full(T, float(base[cv_row]), dtype='float32')
    return None


def compute_episode_metrics(ep: Dict) -> Dict[str, float]:
    """V3-equivalent operator-facing metrics, plus tracking IAE/ITAE/ISE.

    All metrics are simulator-agnostic — they derive from the rollout
    arrays, the bound boxes recorded with the episode, and (where
    available) the per-step active CV target.  No simulator-specific
    column names or knobs.
    """
    states = ep['states']
    controls = ep['controls']
    cv_idx = ep.get('cv_indices') or []
    mv_bounds = ep.get('mv_bounds') or []
    cv_bounds = ep.get('cv_bounds') or []
    T = int(ep['episode_length'])
    n_mv = controls.shape[1]

    # --- MV economic / actuator-health metrics ---------------------------
    mv_tv = float(np.sum(np.abs(np.diff(controls, axis=0)))) if T > 1 else 0.0
    activity_ratios: List[float] = []
    hugging_scores: List[float] = []
    usage_scores: List[float] = []
    reversal_rates: List[float] = []
    for j in range(n_mv):
        col = controls[:, j].astype('float64')
        b = mv_bounds[j] if j < len(mv_bounds) else None
        if (isinstance(b, list) and len(b) >= 2
                and np.isfinite(b[0]) and np.isfinite(b[1])
                and b[1] > b[0]):
            lo, hi = float(b[0]), float(b[1])
            rng = hi - lo
        else:
            rng = float(np.nanmax(col) - np.nanmin(col)) or 1.0
            lo, hi = float(np.nanmin(col)), float(np.nanmax(col))
        if rng > 1e-9:
            activity_ratios.append(float(np.nanstd(col) / rng))
            mean_col = float(np.nanmean(col))
            hugging_scores.append(float(max(0.0,
                min((mean_col - lo) / rng, (hi - mean_col) / rng))))
            usage_scores.append(float(
                (np.nanmax(col) - np.nanmin(col)) / rng))
        reversal_rates.append(_reversal_rate(col, rng))

    # --- CV tracking: IAE / ITAE / ISE per CV (where target enabled) ----
    iae_per_cv: List[float] = []
    itae_per_cv: List[float] = []
    ise_per_cv: List[float] = []
    for k, cidx in enumerate(cv_idx):
        if cidx >= states.shape[1]:
            continue
        tgt = _cv_active_target(ep, k)
        if tgt is None:
            continue
        err = states[:T, cidx].astype('float64') - tgt[:T].astype('float64')
        # Normalise by CV bound width so cross-CV / cross-sim comparison is meaningful.
        b = cv_bounds[k] if k < len(cv_bounds) else None
        if (isinstance(b, list) and len(b) >= 2
                and np.isfinite(b[0]) and np.isfinite(b[1])
                and b[1] > b[0]):
            denom = float(b[1]) - float(b[0])
        else:
            denom = float(np.nanstd(states[:T, cidx])) or 1.0
        e = err / max(1e-9, denom)
        iae_per_cv.append(float(np.sum(np.abs(e))))
        itae_per_cv.append(float(np.sum(np.arange(T) * np.abs(e))))
        ise_per_cv.append(float(np.sum(e ** 2)))

    # --- CV quality: smoothness + limit hugging (MV chatter is allowed) -
    cv_d2_per: List[float] = []
    cv_rev_per: List[float] = []
    cv_tv_per: List[float] = []
    cv_head_per: List[float] = []
    cv_vfrac_per: List[float] = []
    cv_vdepth_per: List[float] = []
    cv_sides: List[str] = []
    any_pref_unknown = False
    for k, cidx in enumerate(cv_idx):
        if cidx >= states.shape[1]:
            continue
        y = states[:T, int(cidx)].astype('float64')
        lo, hi = _base_cv_lo_hi(ep, k, y)
        width = max(1e-9, hi - lo)
        side, unk = preferred_cv_side(ep, k)
        cv_sides.append(side)
        any_pref_unknown = any_pref_unknown or bool(unk)
        if T >= 3:
            d2 = np.diff(y, n=2)
            d2_rms = float(np.sqrt(np.mean(d2 * d2))) if d2.size else 0.0
        else:
            d2_rms = 0.0
        cv_d2_per.append(d2_rms / width)
        cv_rev_per.append(_reversal_rate(y, width))
        if T > 1:
            cv_tv_per.append(
                float(np.mean(np.abs(np.diff(y)))) / width)
        else:
            cv_tv_per.append(0.0)
        over_hi = np.maximum(y - hi, 0.0)
        over_lo = np.maximum(lo - y, 0.0)
        depth = (over_hi + over_lo) / width
        viol = depth > 0.0
        cv_vfrac_per.append(float(np.mean(viol)) if T > 0 else 0.0)
        if np.any(viol):
            cv_vdepth_per.append(float(np.mean(depth[viol])))
        else:
            cv_vdepth_per.append(0.0)
        feasible = ~viol
        if side == 'hi':
            gap = (hi - y) / width
        elif side == 'lo':
            gap = (y - lo) / width
        else:
            gap = np.minimum(y - lo, hi - y) / width
        if np.any(feasible):
            cv_head_per.append(float(np.mean(gap[feasible])))
        # else: all violating — headroom undefined for this CV

    def _worst(xs: List[float], default: float = 0.0) -> float:
        vals = _finite_floats(xs)
        return float(max(vals)) if vals else default

    return {
        'cv_violation_mean': float(ep.get('mean_cv_violation', 0.0)),
        'mv_violation_mean': float(ep.get('mean_mv_violation', 0.0)),
        'mv_tv': mv_tv,
        'mv_activity_ratio': float(np.mean(activity_ratios)) if activity_ratios else 0.0,
        'mv_bound_hugging_score': float(np.min(hugging_scores)) if hugging_scores else 1.0,
        'mv_bound_usage': float(np.mean(usage_scores)) if usage_scores else 0.0,
        'mv_reversal_rate': float(np.mean(reversal_rates)) if reversal_rates else 0.0,
        'economic_score': float(np.mean(ep['raw_rewards'])) if T > 0 else 0.0,
        'cum_raw_reward': float(ep.get('cum_raw_reward', 0.0)),
        'iae_normed_mean': float(np.mean(iae_per_cv)) if iae_per_cv else 0.0,
        'itae_normed_mean': float(np.mean(itae_per_cv)) if itae_per_cv else 0.0,
        'ise_normed_mean': float(np.mean(ise_per_cv)) if ise_per_cv else 0.0,
        'iae_normed_per_cv': iae_per_cv,
        'itae_normed_per_cv': itae_per_cv,
        'ise_normed_per_cv': ise_per_cv,
        # CV quality (episode = worst CV).  MV reversal is diagnostic only.
        'cv_d2_rms_normed': _worst(cv_d2_per),
        'cv_reversal_rate': _worst(cv_rev_per),
        'cv_tv_per_step_normed': _worst(cv_tv_per),
        'cv_opt_headroom': _worst(cv_head_per, default=float('nan')),
        'cv_viol_frac': _worst(cv_vfrac_per),
        'cv_viol_depth_normed': _worst(cv_vdepth_per),
        'cv_preferred_side': ','.join(cv_sides) if cv_sides else 'unknown',
        'cv_preferred_unknown': bool(any_pref_unknown or not cv_sides),
        'cv_d2_rms_normed_per_cv': cv_d2_per,
        'cv_reversal_rate_per_cv': cv_rev_per,
        'cv_opt_headroom_per_cv': cv_head_per,
    }


def compute_event_response_metrics(ep: Dict, *, settle_band: float = 0.05,
                                     window_steps: int | None = None
                                     ) -> Dict[str, object]:
    """Per-disturbance response on the most-impacted CV, plus return-to-limit.

    Window ``[start, start + window_steps]`` (default = 5τ if identified,
    else 200), clipped at the next scripted event.  Per CV:
      * ``peak_overshoot`` / ``settle_steps`` / ``iae_window`` vs *pre-event
        baseline* (IAE-to-pre-event can look good while the CV sits
        mid-band — that is not return-to-limit).
      * ``cv_return_headroom`` — late-window gap to the *economic* bound
        / bound width (0 = on the limit).
      * ``cv_return_time_frac`` — first time inside ``RETURN_BAND_NORMED``
        of that bound, as a fraction of the window (0 = immediate, 1 =
        never).  Window is already τ-scaled, so the fraction is
        plant-agnostic.

    Overshoot aggregates use the most-impacted CV.  Return aggregates use
    the worst return (farthest from the economic limit) per event.
    """
    states = ep['states']
    cv_idx = ep.get('cv_indices') or []
    cv_bounds = ep.get('cv_bounds') or []
    T = int(ep['episode_length'])

    # Default window: 5 × identified τ (sample-rate scaled) if known,
    # else 200 steps.  Falls back to the schedule-builder's heuristic
    # when no identifier context is on disk.
    if window_steps is None:
        window_steps = 200
        try:
            from utils.training_disturbance import _load_identifier_context
            ctx = _load_identifier_context()
            tau = float((ctx.get('dynamics', {}) or {})
                          .get('tau_dominant_identified', 0.0) or 0.0)
            sr = max(1, int(ep.get('sample_rate', 1)))
            if tau > 0:
                window_steps = max(60, min(400, int(round(5.0 * tau / sr))))
        except Exception:
            pass

    sched_starts: List[int] = []
    for ev in ep.get('schedule') or []:
        try:
            s = int(ev.get('start', 0))
        except Exception:
            continue
        if 0 < s < T - 5:
            sched_starts.append(s)
    sched_starts.sort()

    events = []
    for ev in ep.get('schedule') or []:
        try:
            start = int(ev.get('start', 0))
        except Exception:
            continue
        if start <= 0 or start >= T - 5:
            continue
        nxt = [s for s in sched_starts if s > start]
        clip_end = nxt[0] if nxt else T
        end = min(T, start + int(window_steps), clip_end)
        if end - start < 8:
            continue
        pre = states[max(0, start - 20):start]
        if pre.size == 0:
            continue
        per_cv = []
        for k, cidx in enumerate(cv_idx):
            if cidx >= states.shape[1]:
                continue
            base = float(np.mean(pre[:, cidx]))
            dev = states[start:end, cidx].astype('float64') - base
            if dev.size == 0:
                continue
            b = cv_bounds[k] if k < len(cv_bounds) else None
            if (isinstance(b, list) and len(b) >= 2
                    and np.isfinite(b[0]) and np.isfinite(b[1])
                    and b[1] > b[0]):
                denom = float(b[1]) - float(b[0])
            else:
                denom = float(np.nanstd(states[:T, cidx])) or 1.0
            dev_n = dev / max(1e-9, denom)
            ovs = float(dev_n[int(np.argmax(np.abs(dev_n)))])
            band = settle_band  # already normalised
            settle = None
            for i in range(dev_n.size - 1, -1, -1):
                if abs(dev_n[i]) > band:
                    settle = i + 1
                    break
            if settle == 0:
                settle = 0
            iae_w = float(np.sum(np.abs(dev_n)))
            y_win = states[start:end, int(cidx)].astype('float64')
            lo, hi = _base_cv_lo_hi(ep, k, y_win)
            side, _unk = preferred_cv_side(ep, k)
            ret = return_to_limit_on_window(
                y_win, lo, hi, side, band=float(settle_band))
            per_cv.append({
                'cv_row': k, 'cv_index': int(cidx),
                'peak_overshoot_normed': ovs,
                'settle_steps': settle,
                'iae_window_normed': iae_w,
                'preferred_side': side,
                **ret,
            })
        if not per_cv:
            continue
        worst = max(per_cv, key=lambda r: abs(r['peak_overshoot_normed']))

        def _return_rank(r: Dict) -> Tuple[float, float]:
            h = r.get('cv_return_headroom')
            try:
                hf = float(h)
            except (TypeError, ValueError):
                hf = float('nan')
            if not np.isfinite(hf):
                hf = 1.0 + float(r.get('cv_return_viol_frac') or 1.0)
            tf = float(r.get('cv_return_time_frac') or 1.0)
            return (hf, tf)

        worst_ret = max(per_cv, key=_return_rank)
        events.append({
            'name': str(ev.get('name', 'event')),
            'start': start, 'end': end,
            'intent': str(ev.get('intent', '')),
            'source': str(ev.get('source', '')),
            'delta': float(ev.get('delta', 0.0)),
            'worst_cv': worst,
            'worst_return': worst_ret,
            'per_cv': per_cv,
        })

    def _agg(key: str) -> Dict[str, float]:
        vals = [abs(e['worst_cv'][key]) if isinstance(e['worst_cv'][key], (int, float))
                else 0.0 for e in events]
        if not vals:
            return {'median': 0.0, 'p90': 0.0, 'max': 0.0, 'n': 0}
        a = np.asarray(vals, dtype='float64')
        return {'median': float(np.median(a)),
                'p90': float(np.percentile(a, 90)),
                'max': float(np.max(a)),
                'n': int(a.size)}

    def _agg_return(key: str) -> Dict[str, float]:
        raw: List[float] = []
        for e in events:
            wr = e.get('worst_return') or {}
            v = wr.get(key)
            try:
                f = float(v)
            except (TypeError, ValueError):
                continue
            if np.isfinite(f):
                raw.append(f)
        if not raw:
            return {'mean': float('nan'), 'median': float('nan'),
                    'p90': float('nan'), 'max': float('nan'), 'n': 0}
        a = np.asarray(raw, dtype='float64')
        return {'mean': float(np.mean(a)),
                'median': float(np.median(a)),
                'p90': float(np.percentile(a, 90)),
                'max': float(np.max(a)),
                'n': int(a.size)}

    settle_vals = [e['worst_cv']['settle_steps'] for e in events
                    if e['worst_cv']['settle_steps'] is not None]
    settle_agg = ({'median': float(np.median(settle_vals)),
                   'p90': float(np.percentile(settle_vals, 90)),
                   'max': float(np.max(settle_vals)),
                   'n_settled': int(len(settle_vals)),
                   'n_total': int(len(events))}
                  if settle_vals else
                  {'median': 0.0, 'p90': 0.0, 'max': 0.0,
                   'n_settled': 0, 'n_total': int(len(events))})
    n_reached = int(sum(
        1 for e in events
        if (e.get('worst_return') or {}).get('cv_return_reached')))

    return {
        'window_steps': int(window_steps),
        'settle_band_normed': float(settle_band),
        'overshoot_normed': _agg('peak_overshoot_normed'),
        'iae_window_normed': _agg('iae_window_normed'),
        'settle_steps': settle_agg,
        'return_headroom': _agg_return('cv_return_headroom'),
        'return_time_frac': _agg_return('cv_return_time_frac'),
        'return_viol_frac': _agg_return('cv_return_viol_frac'),
        'n_return_reached': n_reached,
        'n_return_events': int(len(events)),
        'events': events,
    }


# ---------------------------------------------------------------------------
# Baselines  (constant-MV; simulator-agnostic, no controller needed)
# ---------------------------------------------------------------------------

def run_constant_mv_episode(env, *, schedule: List[Dict],
                              mv_norm: float = 0.0,
                              dv_override: Optional[np.ndarray] = None) -> Dict:
    """Replay ``schedule`` with a frozen MV at ``mv_norm`` (default = mid).

    Used as the lower baseline on the disturbance-rejection plot — shows
    what the plant does with no control intervention.  Reuses the same
    APCEnv so noise/setpoint/objective code paths are identical to the
    agent rollout (only the action source differs).

    Parameters
    ----------
    dv_override : (T, n_dv) array, optional
        If provided, overwrites the simulator's DV channel(s) after each
        step with ``dv_override[t]``.  Used by the disturbance-rejection
        plot so the baseline experiences the *exact same* uncontrolled
        external disturbance trace as the agent run, making the
        baseline-vs-agent CV overlay an apples-to-apples comparison
        (DV is external by definition; the agent has no influence over
        it, so any divergence between the two runs' DV streams is
        purely noise-RNG drift between the two SimNoiseWrapper
        instances).  Without this override the user sees subtly
        different DV curves and can't tell whether the CV difference
        is due to the controller or due to a different disturbance.
    """
    obs_window = env.reset(exploration=False)
    env._schedule = list(schedule)
    T = env.cfg.episode_length
    state_dim = env.state_dim
    action_dim = env.action_dim
    states = np.zeros((T, state_dim), dtype='float32')
    controls = np.zeros((T, action_dim), dtype='float32')
    raw_rewards = np.zeros(T, dtype='float32')
    cv_violations = np.zeros(T, dtype='float32')
    mv_violations = np.zeros(T, dtype='float32')
    n_mv_aux = int(getattr(env.setpoint_mgr, 'n_mv', 0))
    n_cv_aux = int(getattr(env.setpoint_mgr, 'n_cv', 0))
    cur_mv_b = np.zeros((T, n_mv_aux, 2), dtype='float32')
    cur_cv_b = np.zeros((T, n_cv_aux, 2), dtype='float32')
    cur_cv_t = np.zeros((T, n_cv_aux), dtype='float32')

    a = np.full(action_dim, float(mv_norm), dtype='float32')
    # Resolve the underlying sim object so we can write back into its
    # state buffer when a DV override is provided.  ``env.sim`` is the
    # SimNoiseWrapper; its ``_sim`` is the raw simulator class instance
    # which holds ``episode_array`` and the live state.
    bare_sim = env.sim
    for _ in range(4):
        inner = getattr(bare_sim, '_sim', None)
        if inner is None:
            break
        bare_sim = inner
    dv_idx_list = list(env.meta.get('dv_indices') or [])
    use_dv_override = (dv_override is not None
                        and len(dv_idx_list) > 0
                        and len(dv_override) >= T)
    for t in range(T):
        next_window, _, done, info = env.step(a)
        comps = info.get('reward_components', {}) or {}
        # If a DV override is provided, force the underlying sim's
        # state to carry the agent's recorded DV trajectory at this
        # step *before* the next step uses it for dynamics.  This makes
        # the baseline experience the identical external disturbance
        # the agent saw, so the CV overlay is an apples-to-apples
        # comparison.
        if use_dv_override:
            try:
                ep_arr = getattr(bare_sim, 'episode_array', None)
                # After env.step() returns, the underlying sim has
                # advanced ``episode_counter`` and written the new row.
                # The *next* step reads ``prev = episode_array[
                # episode_counter]`` — so overwrite that exact row.
                idx = int(getattr(bare_sim, 'episode_counter', t + 1))
                if ep_arr is not None:
                    idx = max(0, min(idx, ep_arr.shape[0] - 1))
                ov = np.asarray(dv_override[t], dtype='float32').reshape(-1)
                for j, di in enumerate(dv_idx_list):
                    if j >= ov.shape[0]:
                        break
                    val = float(ov[j])
                    if ep_arr is not None and idx >= 0 and di < ep_arr.shape[1]:
                        ep_arr[idx, di] = val
            except Exception:
                pass
        raw_st = info.get('raw_state')
        if raw_st is None:
            states[t] = next_window[-1, :state_dim]
        else:
            arr = np.asarray(raw_st, dtype='float32').reshape(-1)
            states[t, :min(state_dim, arr.shape[0])] = arr[:state_dim]
        # Mirror the DV override into the recorded state so plots /
        # CSV report the agent-aligned DV (must come AFTER the
        # raw_st recording, which would otherwise clobber it).
        if use_dv_override:
            ov = np.asarray(dv_override[t], dtype='float32').reshape(-1)
            for j, di in enumerate(dv_idx_list):
                if j >= ov.shape[0]:
                    break
                if di < states.shape[1]:
                    states[t, di] = float(ov[j])
        controls[t] = np.asarray(env._prev_control, dtype='float32')
        raw_rewards[t] = float(info.get('raw_reward', 0.0))
        cv_violations[t] = float(comps.get('cv_violation_penalty', 0.0))
        mv_violations[t] = float(comps.get('mv_violation_penalty', 0.0))
        if n_mv_aux > 0:
            cur_mv_b[t] = np.asarray(env.setpoint_mgr.current_mv_bounds,
                                       dtype='float32')
        if n_cv_aux > 0:
            cur_cv_b[t] = np.asarray(env.setpoint_mgr.current_cv_bounds,
                                       dtype='float32')
            cur_cv_t[t] = np.asarray(env.setpoint_mgr.current_cv_targets,
                                       dtype='float32')
        obs_window = next_window
        if done:
            break

    return {
        'states': states[:t + 1], 'controls': controls[:t + 1],
        'raw_rewards': raw_rewards[:t + 1],
        'scaled_rewards': raw_rewards[:t + 1] * float(env.reward_scale),
        'cv_violations': cv_violations[:t + 1],
        'mv_violations': mv_violations[:t + 1],
        'cum_raw_reward': float(np.sum(raw_rewards[:t + 1])),
        'cum_reward': float(np.sum(raw_rewards[:t + 1] * float(env.reward_scale))),
        'mean_cv_violation': float(cv_violations[:t + 1].mean()),
        'mean_mv_violation': float(mv_violations[:t + 1].mean()),
        'episode_length': int(t + 1),
        'sample_rate': int(env.cfg.sample_rate),
        'cv_indices': list(env.cv_indices),
        'mv_indices': [int(x) for x in (env.meta.get('mv_indices') or [])],
        'dv_indices': [int(x) for x in (env.meta.get('dv_indices') or [])],
        'state_variables': list(env.meta.get('state_variables') or []),
        'mv_bounds': [list(b) for b in
                       getattr(env.setpoint_mgr, 'base_mv_bounds', np.zeros((0, 2)))],
        'cv_bounds': [list(b) for b in
                       getattr(env.setpoint_mgr, 'base_cv_bounds', np.zeros((0, 2)))],
        'cv_targets': [float(x) for x in
                        getattr(env.setpoint_mgr, 'base_cv_targets', [])],
        'cv_target_enabled': [bool(x) for x in
                                getattr(env.setpoint_mgr,
                                          'cv_target_enabled', [])],
        'current_mv_bounds_t': cur_mv_b[:t + 1],
        'current_cv_bounds_t': cur_cv_b[:t + 1],
        'current_cv_targets_t': cur_cv_t[:t + 1],
        'reward_scale': float(env.reward_scale),
        'schedule': schedule,
        'mv_norm_ranges': [list(b) for b in env.mv_norm_ranges],
        'cv_norm_ranges': [list(b) for b in env.cv_norm_ranges],
        **_episode_objective_meta(env),
    }


# ---------------------------------------------------------------------------
# CSV / schedule.txt artefacts  (operator-friendly, opens in Excel)
# ---------------------------------------------------------------------------

def _episode_to_dataframe_dict(ep: Dict, suffix: str = '') -> Dict[str, np.ndarray]:
    """Build a column-name → array dict suitable for pandas.DataFrame.

    Naming follows the V3 convention so spreadsheets / plotting scripts
    transfer over: ``MV_<name>``, ``CV_<name>``, ``DV_<name>``,
    ``MV_<name>_bound_lo/hi``, ``CV_<name>_bound_lo/hi``,
    ``CV_<name>_target``.  ``suffix`` is appended to every column when
    a baseline is being merged into the agent's CSV.
    """
    T = ep['episode_length']
    states = ep['states']
    controls = ep['controls']
    cv_idx = ep.get('cv_indices') or []
    mv_idx = ep.get('mv_indices') or []
    dv_idx = ep.get('dv_indices') or []
    names = ep.get('state_variables') or []
    def _arr(key, dtype='float32'):
        v = ep.get(key)
        if v is None or len(v) == 0:
            return np.zeros((0,), dtype=dtype)
        return np.asarray(v, dtype=dtype)
    cur_mv_b = _arr('current_mv_bounds_t')
    cur_cv_b = _arr('current_cv_bounds_t')
    cur_cv_t = _arr('current_cv_targets_t')

    def _nm(i: int, default: str) -> str:
        s = names[i] if 0 <= i < len(names) and names[i] else default
        return str(s).replace('/', '_').replace(' ', '_')

    out: Dict[str, np.ndarray] = {f'time_step{suffix}': np.arange(T, dtype='int64')}
    for k, i in enumerate(mv_idx):
        nm = _nm(i, f'MV{i}')
        if k < controls.shape[1]:
            out[f'MV_{nm}{suffix}'] = controls[:T, k]
        if cur_mv_b.ndim == 3 and k < cur_mv_b.shape[1]:
            out[f'MV_{nm}_bound_lo{suffix}'] = cur_mv_b[:T, k, 0]
            out[f'MV_{nm}_bound_hi{suffix}'] = cur_mv_b[:T, k, 1]
    for i in dv_idx:
        if i < states.shape[1]:
            out[f'DV_{_nm(i, f"DV{i}")}{suffix}'] = states[:T, i]
    for k, i in enumerate(cv_idx):
        if i < states.shape[1]:
            nm = _nm(i, f'CV{i}')
            out[f'CV_{nm}{suffix}'] = states[:T, i]
            if cur_cv_b.ndim == 3 and k < cur_cv_b.shape[1]:
                out[f'CV_{nm}_bound_lo{suffix}'] = cur_cv_b[:T, k, 0]
                out[f'CV_{nm}_bound_hi{suffix}'] = cur_cv_b[:T, k, 1]
            tgt = _cv_active_target(ep, k)
            if tgt is not None:
                out[f'CV_{nm}_target{suffix}'] = tgt[:T]
    out[f'raw_reward{suffix}'] = ep.get('raw_rewards', np.zeros(T, dtype='float32'))
    out[f'cv_violation_penalty{suffix}'] = ep.get('cv_violations', np.zeros(T, dtype='float32'))
    out[f'mv_violation_penalty{suffix}'] = ep.get('mv_violations', np.zeros(T, dtype='float32'))
    out[f'cum_raw_reward{suffix}'] = np.cumsum(out[f'raw_reward{suffix}']).astype('float64')
    return out


def write_episode_csv(ep_agent: Dict, ep_baseline: Dict | None,
                       csv_path: Path) -> None:
    """Write a single CSV with agent columns + (optional) baseline columns.

    Falls back to plain numpy.savetxt when pandas is unavailable.
    """
    cols = _episode_to_dataframe_dict(ep_agent)
    if ep_baseline is not None and ep_baseline.get('episode_length', 0) > 0:
        cols.update(_episode_to_dataframe_dict(ep_baseline, suffix='_baseline'))
    try:
        import pandas as pd
        T = max(int(v.shape[0]) for v in cols.values())
        norm = {k: (np.pad(v, (0, T - v.shape[0]),
                            constant_values=np.nan).astype('float64')
                     if v.shape[0] < T else v.astype('float64'))
                for k, v in cols.items()}
        pd.DataFrame(norm).to_csv(csv_path, index=False, float_format='%.6g')
    except Exception:
        # Fallback: header + savetxt
        header = ','.join(cols.keys())
        T = max(int(v.shape[0]) for v in cols.values())
        rows = np.full((T, len(cols)), np.nan, dtype='float64')
        for j, (_, v) in enumerate(cols.items()):
            rows[:v.shape[0], j] = v.astype('float64')
        np.savetxt(csv_path, rows, header=header, comments='',
                    delimiter=',', fmt='%.6g')


def write_schedule_txt(schedule: List[Dict], out_path: Path) -> None:
    """V3-style human-readable disturbance-schedule listing."""
    lines = ['Disturbance schedule (validation, holdout_a profile):']
    for i, ev in enumerate(schedule, start=1):
        lines.append(
            f"{i:2d}. t={int(ev.get('start', 0)):5d}  "
            f"{str(ev.get('source', '?')):<14s}  "
            f"target={str(ev.get('target_name', '?')):<24s}  "
            f"intent={str(ev.get('intent', '?')):<10s}  "
            f"delta={float(ev.get('delta', 0.0)):+.4g}"
        )
    out_path.write_text('\n'.join(lines) + '\n')


def plot_episode(ep: Dict, out_path: Path, title: str = '') -> None:
    states = ep['states']
    controls = ep['controls']
    cv_idx = ep['cv_indices']
    cv_norm = ep['cv_norm_ranges']
    mv_norm = ep['mv_norm_ranges']
    schedule = ep['schedule']
    T = ep['episode_length']
    t_arr = np.arange(T)

    n_cv = len(cv_idx)
    n_mv = controls.shape[1]
    n_rows = max(1, n_cv) + max(1, n_mv) + 2  # +rewards +cum
    fig, axes = plt.subplots(n_rows, 1, figsize=(12, 2.0 * n_rows), sharex=True)
    if n_rows == 1:
        axes = [axes]

    row = 0
    # CVs with bound bands
    for j, cidx in enumerate(cv_idx):
        ax = axes[row]; row += 1
        if cidx < states.shape[1]:
            ax.plot(t_arr, states[:, cidx], color='C0', lw=1.0, label=f'CV[{cidx}]')
        if j < len(cv_norm):
            lo, hi = cv_norm[j]
            ax.axhline(lo, color='gray', lw=0.6, ls=':')
            ax.axhline(hi, color='gray', lw=0.6, ls=':')
            ax.fill_between(t_arr, lo, hi, color='gray', alpha=0.05)
        _add_disturbance_markers(ax, schedule)
        ax.set_ylabel(f'CV[{cidx}]')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)

    # MVs with bound bands
    for j in range(n_mv):
        ax = axes[row]; row += 1
        ax.plot(t_arr, controls[:, j], color='C1', lw=1.0, label=f'MV[{j}]')
        if j < len(mv_norm):
            lo, hi = mv_norm[j]
            ax.axhline(lo, color='gray', lw=0.6, ls=':')
            ax.axhline(hi, color='gray', lw=0.6, ls=':')
            ax.fill_between(t_arr, lo, hi, color='gray', alpha=0.05)
        _add_disturbance_markers(ax, schedule)
        ax.set_ylabel(f'MV[{j}]')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)

    # Per-step reward
    ax = axes[row]; row += 1
    ax.plot(t_arr, ep['raw_rewards'], color='C2', lw=0.9, label='raw reward')
    ax.plot(t_arr, ep['scaled_rewards'], color='C3', lw=0.9, alpha=0.6,
            label=f"scaled (×{ep['reward_scale']:.2f})")
    ax.axhline(0, color='gray', lw=0.5, ls='-', alpha=0.5)
    _add_disturbance_markers(ax, schedule)
    ax.set_ylabel('reward')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)

    # Cumulative reward
    ax = axes[row]; row += 1
    ax.plot(t_arr, np.cumsum(ep['raw_rewards']), color='C2', lw=1.0,
            label=f"raw cum (final={ep['cum_raw_reward']:+.1f})")
    ax.plot(t_arr, np.cumsum(ep['scaled_rewards']), color='C3', lw=1.0,
            alpha=0.6,
            label=f"scaled cum (final={ep['cum_reward']:+.1f})")
    _add_disturbance_markers(ax, schedule)
    ax.set_ylabel('cum reward')
    ax.set_xlabel('step')
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def plot_disturbance_rejection(ep: Dict, out_path: Path, title: str = '',
                                  *, ep_baseline: Dict | None = None,
                                  event_metrics: Dict | None = None) -> None:
    """Plant-style disturbance-rejection plot.

    Mirrors ``plot_channels`` from ``neural-apc-pytorch/evaluation/
    validate_latent.py`` so the operator sees a real plant view:
      - one row per channel (MV, DV, CV) with semantic state names,
      - CV bound band shaded orange + red dashed low/high lines,
      - MV red dashed low/high lines + gray dotted normalization range,
      - dark-green dash-dot CV target setpoint,
      - vertical disturbance markers with ▲/▼ direction arrow at top,
      - **constant-MV baseline overlay** (dashed grey) on every MV/CV row
        when ``ep_baseline`` is provided — makes "is the agent doing
        anything?" answerable at a glance,
      - **CV tracking-error subplot** (replaces the legacy cum-reward
        subplot, which was dominated by violation-penalty steps and
        carried no operator information),
      - per-event response annotations (±overshoot, settle steps) when
        ``event_metrics`` is provided.
    """
    states = ep['states']
    controls = ep['controls']
    cv_idx = ep['cv_indices']
    mv_idx = ep.get('mv_indices') or []
    dv_idx = ep.get('dv_indices') or []
    state_names = ep.get('state_variables') or []
    cv_norm = ep['cv_norm_ranges']
    mv_norm = ep['mv_norm_ranges']
    cv_bounds = ep.get('cv_bounds') or []
    mv_bounds = ep.get('mv_bounds') or []
    cv_targets = ep.get('cv_targets') or []
    cv_target_enabled = ep.get('cv_target_enabled') or []
    # Per-step (T, n_ch[, 2]) traces of the operator-active bounds /
    # targets.  Empty / shape-(T, 0, 2) when the setpoint manager did
    # not run, in which case the plot falls back to the constant base
    # bounds only.
    def _arr_t(key, dtype='float32'):
        v = ep.get(key)
        if v is None or len(v) == 0:
            return np.zeros((0,), dtype=dtype)
        return np.asarray(v, dtype=dtype)
    cur_mv_b_t = _arr_t('current_mv_bounds_t')
    cur_cv_b_t = _arr_t('current_cv_bounds_t')
    cur_cv_tgt_t = _arr_t('current_cv_targets_t')
    hidden_dist_t = _arr_t('hidden_disturbance_t')
    schedule = ep['schedule']
    T = ep['episode_length']
    t_arr = np.arange(T)

    def _name(i: int, default: str) -> str:
        return state_names[i] if 0 <= i < len(state_names) and state_names[i] else default

    # Build channel rows: MVs first (operator manipulated), then DVs
    # (uncontrolled drivers — what the disturbance actually injects), then
    # CVs (controlled outputs).  Same order/grouping as the legacy plot.
    channels: List[Dict] = []
    for k, i in enumerate(mv_idx):
        bounds = mv_bounds[k] if k < len(mv_bounds) else None
        norm = mv_norm[k] if k < len(mv_norm) else None
        bounds_t = (cur_mv_b_t[:, k] if (cur_mv_b_t.ndim == 3
                                          and k < cur_mv_b_t.shape[1])
                     else None)
        channels.append({'group': 'mv', 'series': controls[:, k] if k < controls.shape[1] else None,
                         'label': _name(i, f'MV[{i}]'), 'bounds': bounds,
                         'bounds_t': bounds_t,
                         'norm': norm, 'target': None, 'target_t': None,
                         'color': '#1f77b4'})
    for i in dv_idx:
        if i >= states.shape[1]:
            continue
        channels.append({'group': 'dv', 'series': states[:, i],
                         'label': _name(i, f'DV[{i}]'), 'bounds': None,
                         'bounds_t': None,
                         'norm': None, 'target': None, 'target_t': None,
                         'color': '#9467bd'})
    for k, i in enumerate(cv_idx):
        if i >= states.shape[1]:
            continue
        bounds = cv_bounds[k] if k < len(cv_bounds) else None
        norm = cv_norm[k] if k < len(cv_norm) else None
        tgt_enabled = (k < len(cv_target_enabled)
                        and bool(cv_target_enabled[k]))
        target = (cv_targets[k] if (tgt_enabled and k < len(cv_targets))
                   else None)
        bounds_t = (cur_cv_b_t[:, k] if (cur_cv_b_t.ndim == 3
                                          and k < cur_cv_b_t.shape[1])
                     else None)
        target_t = (cur_cv_tgt_t[:, k] if (tgt_enabled
                                            and cur_cv_tgt_t.ndim == 2
                                            and k < cur_cv_tgt_t.shape[1])
                     else None)
        channels.append({'group': 'cv', 'series': states[:, i],
                         'label': _name(i, f'CV[{i}]'), 'bounds': bounds,
                         'bounds_t': bounds_t,
                         'norm': norm, 'target': target,
                         'target_t': target_t,
                         'cv_ord': k,
                         'color': '#2ca02c'})

    # Dedicated UNMEASURED (hidden) disturbance INPUT rows — one per CV that
    # actually received a hidden disturbance.  Plots the raw injected offset
    # (engineering units, zero baseline) as its OWN panel so the operator can
    # directly SEE the unmeasured load the controller had to reject and line
    # it up against the CV-response panel above (the agent/WM never observe
    # this signal).  With the P90 realistic schedule this reveals the event
    # SHAPE (step / ramp / pulse / drift), timing, and persistence.
    if hidden_dist_t.ndim == 2 and hidden_dist_t.shape[0] > 0:
        for k, i in enumerate(cv_idx):
            if k >= hidden_dist_t.shape[1]:
                continue
            col = np.asarray(hidden_dist_t[:, k], dtype='float32')
            if not np.any(np.abs(col) > 1e-9):
                continue
            channels.append({'group': 'hidden', 'series': col,
                             'label': f'{_name(i, f"CV[{i}]")} — unmeasured '
                                       f'disturbance input (hidden)',
                             'bounds': None, 'bounds_t': None, 'norm': None,
                             'target': None, 'target_t': None,
                             'color': '#e76f51'})

    n_rows = max(1, len(channels)) + 2  # + cum reward + reward/violation companion
    fig, axes = plt.subplots(n_rows, 1,
                              figsize=(13, max(4.0, 2.0 * n_rows)),
                              sharex=True)
    if n_rows == 1:
        axes = [axes]

    # Per-event annotations (CV settle/overshoot) for the summary record.
    annotations: List[Dict] = []
    for ev in schedule:
        st = int(ev.get('start', 0))
        if st >= T - 5:
            continue
        for j, cidx in enumerate(cv_idx):
            if cidx >= states.shape[1]:
                continue
            pre = states[max(0, st - 20):st, cidx]
            post = states[st:min(T, st + 200), cidx]
            if pre.size == 0 or post.size == 0:
                continue
            base = float(np.mean(pre))
            dev = post - base
            ovr = float(dev[np.argmax(np.abs(dev))]) if dev.size else 0.0
            band = max(1e-6, 0.05 * (np.max(np.abs(pre)) if pre.size else 1.0))
            settled = np.where(np.abs(dev) <= band)[0]
            settle_t = int(settled[0]) if settled.size else int(post.size)
            annotations.append({'cv_row': j, 'start': st,
                                 'overshoot': ovr, 'settle_steps': settle_t,
                                 'name': ev.get('name', 'step')})

    def _draw_disturbance_markers(ax) -> None:
        ylo, yhi = ax.get_ylim()
        y_top = yhi - 0.04 * (yhi - ylo)
        for ev in schedule:
            st = int(ev.get('start', 0))
            color = ev.get('color') or (
                '#ff7f0e' if 'violation' in str(ev.get('intent', '')).lower()
                else '#17a2b8')
            ax.axvline(st, color=color, alpha=0.50, linewidth=1.2,
                        linestyle='--')
            delta = float(ev.get('delta', 0.0))
            label = '\u25B2' if delta > 0 else '\u25BC'
            ax.text(st, y_top, label, color=color, fontsize=8, ha='center',
                     va='top', fontweight='bold', clip_on=True)

    for r, ch in enumerate(channels):
        ax = axes[r]
        series = ch['series']
        if series is None or len(series) == 0:
            ax.set_ylabel(ch['label'])
            continue
        ax.plot(t_arr[:len(series)], series, color=ch['color'], lw=1.2,
                label=ch['label'])

        # Dedicated unmeasured-disturbance INPUT panel: zero baseline + fill
        # so the injected load's shape/magnitude is unmistakable.  No bounds/
        # target/baseline overlays apply (the rest of the loop no-ops for this
        # group since bounds/norm/target are None).
        if ch.get('group') == 'hidden':
            m = len(series)
            ax.axhline(0.0, color='#6c757d', lw=0.8, ls='-', alpha=0.6)
            ax.fill_between(t_arr[:m], 0.0, series[:m],
                             color=ch['color'], alpha=0.20)
            ax.set_ylabel('Δ disturbance\n(eng. units)', fontsize=8)
            ax.legend(loc='upper right', fontsize=7, framealpha=0.6)
            ax.grid(True, alpha=0.3)
            continue

        # Baseline overlay (constant-MV episode under same schedule).
        # The baseline is the same for every seed and channel: it shows
        # what the plant does with no control, so the operator can see
        # at a glance how much the agent recovered the deviation.
        if ep_baseline is not None:
            base_series = None
            grp = ch.get('group')
            # Direct alignment: channels were built in MV/DV/CV order
            # from ep itself, so the baseline rollout (built the same
            # way) lines up positionally.
            mv_idx_a = ep.get('mv_indices') or []
            dv_idx_a = ep.get('dv_indices') or []
            cv_idx_a = ep.get('cv_indices') or []
            if grp == 'mv' and r < len(mv_idx_a):
                k = r
                if k < ep_baseline['controls'].shape[1]:
                    base_series = ep_baseline['controls'][:, k]
            elif grp == 'dv':
                # DV row index in channels = len(mv) + (offset within DVs)
                k = r - len(mv_idx_a)
                if 0 <= k < len(dv_idx_a):
                    cidx = dv_idx_a[k]
                    if cidx < ep_baseline['states'].shape[1]:
                        base_series = ep_baseline['states'][:, cidx]
            elif grp == 'cv':
                k = r - len(mv_idx_a) - len(dv_idx_a)
                if 0 <= k < len(cv_idx_a):
                    cidx = cv_idx_a[k]
                    if cidx < ep_baseline['states'].shape[1]:
                        base_series = ep_baseline['states'][:, cidx]
            if base_series is not None and len(base_series) > 0:
                m = min(len(base_series), len(t_arr))
                ax.plot(t_arr[:m], base_series[:m], color='#888888',
                         lw=1.0, ls='--', alpha=0.85,
                         label='baseline (no control)')

        bounds = ch.get('bounds')
        if bounds is not None and len(bounds) >= 2 and \
           np.isfinite(bounds[0]) and np.isfinite(bounds[1]) and \
           bounds[1] > bounds[0] and abs(bounds[0]) < 1e9 and abs(bounds[1]) < 1e9:
            lo_b, hi_b = float(bounds[0]), float(bounds[1])
            if ch['group'] == 'cv':
                ax.axhspan(lo_b, hi_b, color='#ffcc80', alpha=0.18,
                            label='CV base band')
                ax.axhline(lo_b, color='#d32f2f', linestyle='--', linewidth=1.0,
                            label='CV base low')
                ax.axhline(hi_b, color='#d32f2f', linestyle='--', linewidth=1.0,
                            label='CV base high')
            else:
                ax.axhline(lo_b, color='r', linestyle='--', linewidth=1.0,
                            label='Base low')
                ax.axhline(hi_b, color='r', linestyle='--', linewidth=1.0,
                            label='Base high')

        # Per-step active bounds (operator schedule) — overlays the
        # base bound box.  Shown as a step trace (post-step semantics)
        # so the operator can see exactly when each bound moved.
        bounds_t = ch.get('bounds_t')
        if (bounds_t is not None
                and isinstance(bounds_t, np.ndarray)
                and bounds_t.ndim == 2
                and bounds_t.shape[1] >= 2
                and bounds_t.shape[0] >= 1):
            n = min(bounds_t.shape[0], len(t_arr))
            lo_arr = np.asarray(bounds_t[:n, 0], dtype='float32')
            hi_arr = np.asarray(bounds_t[:n, 1], dtype='float32')
            t_seg = t_arr[:n]
            if ch['group'] == 'cv':
                ax.fill_between(t_seg, lo_arr, hi_arr,
                                  step='post', color='#fb8c00', alpha=0.10,
                                  label='CV active band')
                ax.step(t_seg, lo_arr, where='post', color='#b71c1c',
                         linewidth=1.6, label='CV active low')
                ax.step(t_seg, hi_arr, where='post', color='#b71c1c',
                         linewidth=1.6, label='CV active high')
            else:
                ax.step(t_seg, lo_arr, where='post', color='#c62828',
                         linewidth=1.4, label='MV active low')
                ax.step(t_seg, hi_arr, where='post', color='#c62828',
                         linewidth=1.4, label='MV active high')

        norm = ch.get('norm')
        if norm is not None and len(norm) >= 2 and \
           np.isfinite(norm[0]) and np.isfinite(norm[1]):
            ax.axhline(float(norm[0]), color='#6c757d', linestyle=':',
                        linewidth=1.0, label='Norm low')
            ax.axhline(float(norm[1]), color='#6c757d', linestyle=':',
                        linewidth=1.0, label='Norm high')

        target = ch.get('target')
        target_t = ch.get('target_t')
        if (target_t is not None
                and isinstance(target_t, np.ndarray)
                and target_t.ndim == 1
                and target_t.size >= 1
                and np.any(np.isfinite(target_t))):
            n = min(target_t.size, len(t_arr))
            ax.step(t_arr[:n], np.asarray(target_t[:n], dtype='float32'),
                     where='post', color='#1b5e20', linewidth=1.6,
                     linestyle='-.', label='Target (active)')
        elif target is not None and np.isfinite(target):
            ax.axhline(float(target), color='#1b5e20', linestyle='-.',
                        linewidth=1.4, label=f'Target ({float(target):g})')

        # Unmeasured (hidden OU) disturbance overlay — only on CV rows.
        # The disturbance is added to the CV state and is invisible to the
        # agent/WM; plotting ``ref + offset`` lets the operator see the
        # magnitude/shape of what the controller had to reject.  ``ref`` is
        # the active target if enabled, else the active-band centre, else
        # the series mean, so the dotted trace sits on the CV's own scale.
        cv_ord = ch.get('cv_ord')
        if (ch.get('group') == 'cv' and cv_ord is not None
                and hidden_dist_t.ndim == 2
                and cv_ord < hidden_dist_t.shape[1]
                and np.any(np.abs(hidden_dist_t[:, cv_ord]) > 1e-9)):
            hcol = np.asarray(hidden_dist_t[:, cv_ord], dtype='float32')
            nH = min(hcol.shape[0], len(t_arr))
            # Reference level the offset is drawn relative to.
            if (target_t is not None and isinstance(target_t, np.ndarray)
                    and target_t.ndim == 1 and target_t.size >= 1
                    and np.any(np.isfinite(target_t))):
                ref = np.asarray(target_t[:nH], dtype='float32')
            elif target is not None and np.isfinite(target):
                ref = np.full(nH, float(target), dtype='float32')
            elif (bounds is not None and len(bounds) >= 2
                    and np.isfinite(bounds[0]) and np.isfinite(bounds[1])):
                ref = np.full(nH, 0.5 * (float(bounds[0]) + float(bounds[1])),
                              dtype='float32')
            else:
                fin = series[np.isfinite(series)] if isinstance(series, np.ndarray) else None
                ref = np.full(nH, float(np.mean(fin)) if fin is not None and fin.size else 0.0,
                              dtype='float32')
            ax.plot(t_arr[:nH], ref + hcol[:nH], color='#ff7f0e',
                     lw=1.1, ls=':', alpha=0.85,
                     label='unmeasured disturbance (hidden)')


        finite = series[np.isfinite(series)] if isinstance(series, np.ndarray) else None
        lo = float(np.min(finite)) if finite is not None and finite.size else None
        hi = float(np.max(finite)) if finite is not None and finite.size else None
        for ref in (bounds, norm):
            if ref is not None and len(ref) >= 2 and np.isfinite(ref[0]) and np.isfinite(ref[1]):
                rlo, rhi = float(ref[0]), float(ref[1])
                if abs(rlo) < 1e9 and abs(rhi) < 1e9:
                    lo = rlo if lo is None else min(lo, rlo)
                    hi = rhi if hi is None else max(hi, rhi)
        if lo is not None and hi is not None and hi > lo:
            pad = 0.06 * (hi - lo)
            ax.set_ylim(lo - pad, hi + pad)

        _draw_disturbance_markers(ax)
        ax.set_ylabel(ch['label'])
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)

    ax = axes[-2]
    # CV tracking-error trace.  Replaces cum-reward (which was
    # dominated by the violation-penalty step and carried no actionable
    # operator information).  For each CV with a target enabled we plot
    # |CV_t - target_t| / bound_width on the left axis, and the running
    # IAE (sum of normalised |error|) on a twin axis on the right.
    cv_idx_local = ep.get('cv_indices') or []
    cv_bounds_local = ep.get('cv_bounds') or []
    states_local = ep['states']
    plotted_any_err = False
    cum_iae = np.zeros_like(t_arr, dtype='float64')
    for k, cidx in enumerate(cv_idx_local):
        if cidx >= states_local.shape[1]:
            continue
        tgt = _cv_active_target(ep, k)
        if tgt is None:
            continue
        b = cv_bounds_local[k] if k < len(cv_bounds_local) else None
        if (isinstance(b, list) and len(b) >= 2 and np.isfinite(b[0])
                and np.isfinite(b[1]) and b[1] > b[0]):
            denom = float(b[1]) - float(b[0])
        else:
            denom = float(np.nanstd(states_local[:T, cidx])) or 1.0
        n = min(T, tgt.shape[0])
        err = (states_local[:n, cidx].astype('float64')
               - tgt[:n].astype('float64')) / max(1e-9, denom)
        ax.plot(t_arr[:n], np.abs(err), lw=1.0, label=f'|err| CV[{cidx}]')
        cum_iae[:n] += np.cumsum(np.abs(err))
        plotted_any_err = True
        if ep_baseline is not None:
            sb = ep_baseline['states']
            tb = _cv_active_target(ep_baseline, k)
            if tb is not None and cidx < sb.shape[1]:
                m = min(sb.shape[0], tb.shape[0], len(t_arr))
                err_b = (sb[:m, cidx].astype('float64')
                          - tb[:m].astype('float64')) / max(1e-9, denom)
                ax.plot(t_arr[:m], np.abs(err_b), lw=0.9, ls='--',
                         color='#888888', alpha=0.8,
                         label=f'|err| CV[{cidx}] baseline')
    ax.axhline(0.0, color='gray', lw=0.5, ls='-', alpha=0.5)
    if plotted_any_err:
        ax2 = ax.twinx()
        ax2.plot(t_arr, cum_iae, color='#1f77b4', lw=1.2,
                  label=f'cum IAE (final={cum_iae[-1]:.2f})')
        if ep_baseline is not None:
            cum_iae_b = np.zeros(t_arr.shape[0], dtype='float64')
            sb = ep_baseline['states']
            for k, cidx in enumerate(cv_idx_local):
                if cidx >= sb.shape[1]:
                    continue
                tb = _cv_active_target(ep_baseline, k)
                if tb is None:
                    continue
                b = cv_bounds_local[k] if k < len(cv_bounds_local) else None
                denom = ((float(b[1]) - float(b[0]))
                           if (isinstance(b, list) and len(b) >= 2
                                and np.isfinite(b[0]) and np.isfinite(b[1])
                                and b[1] > b[0])
                           else (float(np.nanstd(sb[:, cidx])) or 1.0))
                m = min(sb.shape[0], tb.shape[0], len(t_arr))
                err = np.abs((sb[:m, cidx].astype('float64')
                                - tb[:m].astype('float64'))
                               / max(1e-9, denom))
                cum_iae_b[:m] += np.cumsum(err)
            ax2.plot(t_arr, cum_iae_b, color='#888888', lw=1.0, ls='--',
                      label=f'cum IAE baseline (final={cum_iae_b[-1]:.2f})')
        ax2.set_ylabel('cum IAE (normed)', color='#1f77b4')
        ax2.tick_params(axis='y', labelcolor='#1f77b4')
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, loc='upper left', fontsize=8)
    else:
        # Fallback when no CV target is enabled — keep the cum-reward
        # trace so the subplot is still informative.
        ax.plot(t_arr, np.cumsum(ep['raw_rewards']), color='C2', lw=1.0,
                 label=f"raw cum (final={ep['cum_raw_reward']:+.1f})")
        ax.legend(loc='upper left', fontsize=8)
    _draw_disturbance_markers(ax)
    ax.set_ylabel('|err| (normed)')
    ax.grid(True, alpha=0.3)

    # Reward / violation companion: instantaneous raw reward (left axis) +
    # cumulative CV-violation count (right axis).  Frames the cum-reward
    # subplot above by exposing where penalties are coming from.
    ax = axes[-1]
    ax.plot(t_arr, ep['raw_rewards'], color='#555555', lw=0.8, alpha=0.85,
            label='raw reward (per step)')
    ax.axhline(0.0, color='#888888', linestyle=':', linewidth=0.8)
    ax.set_ylabel('raw r/step')
    ax.grid(True, alpha=0.3)
    cv_v_raw = ep.get('cv_violations')
    cv_v = (np.asarray(cv_v_raw, dtype='float64')
            if cv_v_raw is not None and len(cv_v_raw) > 0
            else np.zeros_like(t_arr, dtype='float64'))
    if cv_v.size == t_arr.size and cv_v.size > 0:
        cv_count = np.cumsum((cv_v > 1e-9).astype('float64'))
        ax2 = ax.twinx()
        ax2.plot(t_arr, cv_count, color='#d32f2f', lw=1.2,
                  label=f'cum CV viol (final={int(cv_count[-1])})')
        ax2.set_ylabel('cum CV viol', color='#d32f2f')
        ax2.tick_params(axis='y', labelcolor='#d32f2f')
        # Combined legend
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, loc='upper left', fontsize=8)
    else:
        ax.legend(loc='upper left', fontsize=8)
    _draw_disturbance_markers(ax)
    ax.set_xlabel('time step')

    fig.suptitle(title, fontsize=11, y=0.995)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    fig.savefig(out_path, dpi=130)
    plt.close(fig)

    return annotations  # caller stashes into summary metrics


def plot_summary(seed_results: List[List[Dict]], out_path: Path,
                  title: str = '') -> None:
    """Cross-seed summary: cum-reward distribution + violation rates."""
    cum = np.array([[ep['cum_raw_reward'] for ep in seed_eps]
                    for seed_eps in seed_results], dtype='float64')
    cv_v = np.array([[ep['mean_cv_violation'] for ep in seed_eps]
                     for seed_eps in seed_results], dtype='float64')
    mv_v = np.array([[ep['mean_mv_violation'] for ep in seed_eps]
                     for seed_eps in seed_results], dtype='float64')

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].boxplot(cum.T, tick_labels=[f's{i}' for i in range(cum.shape[0])])
    axes[0].set_title(f'Cum raw reward per episode\n'
                      f'overall mean={cum.mean():+.2f} ± {cum.std():.2f}')
    axes[0].set_ylabel('cum reward')
    axes[0].grid(True, alpha=0.3)

    axes[1].boxplot(cv_v.T, tick_labels=[f's{i}' for i in range(cv_v.shape[0])])
    axes[1].set_title(f'Mean CV violation\nmean={cv_v.mean():.4f}')
    axes[1].set_ylabel('cv penalty')
    axes[1].grid(True, alpha=0.3)

    axes[2].boxplot(mv_v.T, tick_labels=[f's{i}' for i in range(mv_v.shape[0])])
    axes[2].set_title(f'Mean MV violation\nmean={mv_v.mean():.4f}')
    axes[2].set_ylabel('mv penalty')
    axes[2].grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_validation(*,
                   controller_dir: Path | str,
                   simulation_dir: Path | str | None = None,
                   ckpt: str = 'final.pt',
                   episodes: int = 3, seeds: int = 3,
                   out: Path | str | None = None,
                   deterministic: bool = True) -> Dict:
    """Validate ``controller_dir/<ckpt>`` and write plots + summary.json.

    This is the programmatic entry point used by the workflow runner; the
    CLI ``main()`` simply parses argv and calls this.
    """
    controller_dir = Path(controller_dir).resolve()
    if not controller_dir.exists():
        raise FileNotFoundError(controller_dir)
    from utils.training_disturbance import bind_identifier_out_dir
    bind_identifier_out_dir(controller_dir)
    ckpt_path = controller_dir / ckpt
    if not ckpt_path.exists():
        raise FileNotFoundError(ckpt_path)

    out_dir = Path(out).resolve() if out else controller_dir / 'validation'
    out_dir.mkdir(parents=True, exist_ok=True)

    repo = Path(__file__).resolve().parent.parent
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))

    # Ensure validation uses the same noise config the model was trained
    # against.  ``single_run`` writes ``noise_config.json`` to the
    # controller dir and exports ``SIM_NOISE_CONFIG_JSON`` for the
    # in-process train+validate flow, but standalone ``python -m
    # evaluation.validate`` invocations (and the workflow runner) start
    # without that env var set.  Falling back to "no noise" silently
    # produces flat baseline traces that don't match the training
    # distribution (diagnosed 2026-05-06).  Load the file explicitly.
    if not os.environ.get('SIM_NOISE_CONFIG_JSON', '').strip():
        nc_path = controller_dir / 'noise_config.json'
        if nc_path.exists():
            os.environ['SIM_NOISE_CONFIG_JSON'] = str(nc_path.resolve())
            print(f'[val] noise_config: loaded {nc_path} '
                  '(SIM_NOISE_CONFIG_JSON was unset)', flush=True)

    run_plan = _load_run_plan(controller_dir)
    sim_dir = _resolve_sim_dir(str(simulation_dir) if simulation_dir else None,
                                controller_dir, run_plan)

    os.environ['CONTROL_SETUP_JSON'] = str(sim_dir / 'control_setup.json')
    os.environ['CONTROL_OBJECTIVE_JSON'] = str(sim_dir / 'control_objective.json')
    os.environ['SIMULATION_DIR'] = str(sim_dir)
    if 'sample_rate' in run_plan:
        os.environ['SIM_SAMPLE_RATE'] = str(run_plan['sample_rate'])
    if 'episode_length' in run_plan:
        os.environ['SIM_EPISODE_LENGTH'] = str(run_plan['episode_length'])
    if 'tau' in run_plan:
        os.environ['IDENTIFIED_TAU_DOMINANT'] = f"{run_plan['tau']:g}"
    if 'dead_time' in run_plan:
        os.environ['IDENTIFIED_DEAD_TIME'] = f"{run_plan['dead_time']:g}"

    from training.train import TrainConfig, APCEnv
    from models.dreamer_v4 import DreamerV4, dreamer_v4_config_from_train

    print(f'[val] controller: {controller_dir}', flush=True)
    print(f'[val] simulation: {sim_dir}', flush=True)
    print(f'[val] ckpt: {ckpt_path}  deterministic={deterministic}', flush=True)

    ckpt_obj = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    cfg_dict = ckpt_obj.get('cfg') or {}
    valid_keys = set(TrainConfig.__dataclass_fields__.keys())
    cfg = TrainConfig(**{k: v for k, v in cfg_dict.items() if k in valid_keys})

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DreamerV4(dreamer_v4_config_from_train(cfg)).to(device)
    # Checkpoints saved while ``torch.compile`` was active have keys
    # prefixed with ``_orig_mod.`` (e.g. ``tokenizer._orig_mod.encoder...``)
    # because ``torch.compile`` wraps the module in ``OptimizedModule``.
    # Strip the prefix so the bare DreamerV4 can load.
    sd = ckpt_obj['model']
    if any('._orig_mod.' in k for k in sd):
        sd = {k.replace('._orig_mod.', '.'): v for k, v in sd.items()}
    model.load_state_dict(sd)
    model.eval()

    seed_results: List[List[Dict]] = []
    metrics_records: List[Dict] = []
    disturbance_records: List[Dict] = []
    # Restore the env-side obs normalizer stats saved with the checkpoint
    # (added 2026-05-03 — the trainer applies running standardization to
    # obs before the tokenizer; evaluation must use the same stats with
    # learning frozen so the model sees the distribution it was trained
    # against).  Older checkpoints without 'obs_norm' fall back to the
    # default (mean=0, var=1) stats — which is also what those models
    # were effectively trained with.
    obs_norm_state = ckpt_obj.get('obs_norm') if isinstance(ckpt_obj, dict) else None
    # Seed plan: FOUR consolidated disturbance-rejection plots (one per seed,
    # ``int(seeds)+1`` total = 4 by default), each with the FULL feature set:
    #   * the measured-DV scripted schedule (events spread across the episode), and
    #   * the unmeasured/hidden disturbance at FULL phase-3 amplitude, ALSO spread
    #     across the whole episode (TrainConfig ``hidden_dist_spread``, default
    #     ON in training too) so it reads as a realistic load active start->end
    #     instead of a few front-loaded events holding a DC offset.
    # The seeds differ only by RNG draw, so the four plots show the same realistic
    # operating regime under different disturbance realisations.  The PASS/FAIL
    # fidelity gates run on a separate CLEAN env (``_disturbance_prob_override=0``)
    # and are unaffected by this.
    n_plots = int(seeds) + 1
    seed_plan: List[Tuple[int, bool]] = [(10_000 + s, True) for s in range(n_plots)]
    # Pin spread ON for val plots even if an A/B set training spread=0.
    # Explicit cfg — do not poke leftover ``DREAMER_HIDDEN_DIST_SPREAD``.
    from utils.hidden_disturbance import force_val_hidden_dist_spread
    with force_val_hidden_dist_spread(cfg):
      for seed, unmeasured_full in seed_plan:
        rng = np.random.default_rng(seed)
        env = APCEnv(cfg, rng)
        # Force the hidden disturbance on every validation episode (always test
        # the agent's rejection skill) and lift the curriculum so it runs at full
        # phase-3 amplitude (curriculum_amp_scale cap 1.0, ramp at progress=1).
        env._hidden_disturbance_force = True
        env._current_phase = 3
        env._training_progress = 1.0
        if obs_norm_state is not None:
            try:
                env.set_obs_norm_stats(
                    mean=np.asarray(obs_norm_state.get('mean')),
                    var=np.asarray(obs_norm_state.get('var')),
                    count=float(obs_norm_state.get('count', 1.0)),
                    learn=False,
                )
            except Exception as e:
                print(f'[val] obs_norm restore skipped: {e!r}', flush=True)
        # Use the calibrated reward scale from training when available.
        cal_path = controller_dir / 'reward_calibration.json'
        if cal_path.exists():
            try:
                with open(cal_path, 'r') as f:
                    env.reward_scale = float(json.load(f).get('reward_scale', 1.0))
            except Exception:
                env.reward_scale = 1.0

        per_seed_dir = out_dir / f'seed_{seed:05d}'
        per_seed_dir.mkdir(parents=True, exist_ok=True)
        _ttl_sfx = ''

        eps = []
        for e in range(int(episodes)):
            ep = run_episode(env, model, device, deterministic=deterministic)
            # Per-episode rollouts feed the cross-seed summary + metrics.json;
            # the barebones per-episode plot (CV/MV/reward only, no DV/hidden
            # rows) is intentionally NOT emitted — the consolidated, full-feature
            # ``disturbance_rejection.png`` below is the single per-seed plot.
            eps.append(ep)
            metrics_records.append({
                'seed': seed, 'episode': e,
                'cum_raw_reward': ep['cum_raw_reward'],
                'cum_scaled_reward': ep['cum_reward'],
                'mean_cv_violation': ep['mean_cv_violation'],
                'mean_mv_violation': ep['mean_mv_violation'],
                'episode_length': ep['episode_length'],
                'n_disturbance_events': len(ep['schedule']),
                # V3-parity per-episode KPIs.
                **{f'kpi_{k}': v for k, v in
                    compute_episode_metrics(ep).items()
                    if not isinstance(v, list)},
            })

        # ---- Disturbance-rejection plot (plant-aware holdout_a profile) ----
        try:
            scripted = build_scripted_disturbance_schedule(env, seed=seed)
            ep_d = run_scripted_episode(env, model, device,
                                         deterministic=deterministic,
                                         schedule=scripted)
            # Constant-MV baseline replay under the same scripted
            # schedule.  Same env (so noise / setpoint / objective are
            # identical), same seed offset, frozen action at the bound
            # midpoint.  Makes "is the agent doing anything?" a one-glance
            # answer on the rejection plot.
            try:
                # Reset per-event bookkeeping flags so the baseline run
                # actually re-applies the disturbance schedule.  The
                # agent run mutates ``_applied`` / ``_hold_until`` /
                # ``_hold_state_idx`` / ``_hold_value_raw`` in-place;
                # without this reset the baseline sees zero disturbances
                # → flat DV → CV at steady state with only measurement
                # noise → "is the agent doing anything?" plot shows a
                # near-flat baseline that misleads the operator (mirrors
                # the legacy V3 fix at validate_latent.py:2299).
                for ev in scripted:
                    ev['_applied'] = False
                    ev.pop('_hold_until', None)
                    ev.pop('_hold_state_idx', None)
                    ev.pop('_hold_value_raw', None)
                env_b = APCEnv(cfg, np.random.default_rng(seed + 1))
                env_b._hidden_disturbance_force = True
                if unmeasured_full:
                    env_b._current_phase = 3
                    env_b._training_progress = 1.0
                if obs_norm_state is not None:
                    try:
                        env_b.set_obs_norm_stats(
                            mean=np.asarray(obs_norm_state.get('mean')),
                            var=np.asarray(obs_norm_state.get('var')),
                            count=float(obs_norm_state.get('count', 1.0)),
                            learn=False)
                    except Exception:
                        pass
                env_b.reward_scale = env.reward_scale
                # DV is by definition an external disturbance the agent
                # can't influence.  Force the baseline simulator to
                # follow the *agent's* recorded DV trajectory so the
                # baseline-vs-agent CV overlay isolates "what would
                # happen with no control under the same external
                # disturbance" — the only meaningful comparison.
                # Without the override, the baseline's noise wrapper
                # produces an independent OU+measurement-noise stream
                # on the DV channel, so the two runs see different
                # external disturbances and the overlay is misleading.
                _dv_idx = list(env.meta.get('dv_indices') or [])
                _agent_dv = (ep_d['states'][:, _dv_idx]
                              if (_dv_idx and ep_d['states'].size)
                              else None)
                ep_b = run_constant_mv_episode(env_b, schedule=scripted,
                                                  mv_norm=0.0,
                                                  dv_override=_agent_dv)
            except Exception as _be:
                print(f'[val] baseline replay skipped (seed {seed}): {_be!r}',
                      flush=True)
                ep_b = None

            # Per-event response metrics + episode-level KPIs (V3 parity).
            ep_metrics = compute_episode_metrics(ep_d)
            ev_metrics = compute_event_response_metrics(ep_d)
            base_metrics = (compute_episode_metrics(ep_b)
                              if ep_b is not None else None)
            ev_metrics_b = (compute_event_response_metrics(ep_b)
                              if ep_b is not None else None)

            d_title = (f'seed={seed}  scripted disturbance rejection  '
                       f'cum_raw={ep_d["cum_raw_reward"]:+.2f}  '
                       f'IAE={ep_metrics["iae_normed_mean"]:.2f}  '
                       f'overshoot_max={ev_metrics["overshoot_normed"]["max"]:.3f}'
                       + (f'   |  baseline IAE={base_metrics["iae_normed_mean"]:.2f}'
                          if base_metrics is not None else '')
                       + _ttl_sfx)
            ann = plot_disturbance_rejection(
                ep_d, per_seed_dir / 'disturbance_rejection.png',
                title=d_title, ep_baseline=ep_b, event_metrics=ev_metrics)

            # Operator-friendly artefacts: schedule.txt + CSV (with
            # baseline columns merged in).  Both are pure post-processing
            # — they read the rollout dicts and write text files.
            try:
                write_schedule_txt(scripted,
                                    per_seed_dir / 'disturbance_schedule.txt')
            except Exception as _se:
                print(f'[val] schedule.txt skipped (seed {seed}): {_se!r}',
                      flush=True)
            try:
                write_episode_csv(ep_d, ep_b,
                                   per_seed_dir / 'disturbance_rollout.csv')
            except Exception as _ce:
                print(f'[val] rollout.csv skipped (seed {seed}): {_ce!r}',
                      flush=True)
            # Persist the raw trajectory for offline analysis (added
            # 2026-05-03 — diagnosing constant-action collapse required
            # re-running the eval rollouts because the PNG is the only
            # artefact, which is wasteful and lossy).
            try:
                npz_path = per_seed_dir / 'disturbance_rejection.npz'
                # Convert lists/dicts in the schedule to a JSON blob so
                # np.savez doesn't choke on object arrays.
                sched_json = json.dumps(ep_d.get('schedule', []), default=str)
                np.savez(
                    npz_path,
                    states=np.asarray(ep_d['states'], dtype='float32'),
                    actions_norm=np.asarray(ep_d['actions_norm'], dtype='float32'),
                    controls=np.asarray(ep_d['controls'], dtype='float32'),
                    raw_rewards=np.asarray(ep_d['raw_rewards'], dtype='float32'),
                    scaled_rewards=np.asarray(ep_d['scaled_rewards'], dtype='float32'),
                    cv_violations=np.asarray(ep_d['cv_violations'], dtype='float32'),
                    mv_violations=np.asarray(ep_d['mv_violations'], dtype='float32'),
                    cv_indices=np.asarray(ep_d.get('cv_indices', []), dtype='int64'),
                    mv_indices=np.asarray(ep_d.get('mv_indices', []), dtype='int64'),
                    dv_indices=np.asarray(ep_d.get('dv_indices', []), dtype='int64'),
                    mv_norm_ranges=np.asarray(ep_d.get('mv_norm_ranges', []),
                                                dtype='float32'),
                    cv_norm_ranges=np.asarray(ep_d.get('cv_norm_ranges', []),
                                                dtype='float32'),
                    mv_bounds=np.asarray(ep_d.get('mv_bounds', []),
                                           dtype='float32'),
                    cv_bounds=np.asarray(ep_d.get('cv_bounds', []),
                                           dtype='float32'),
                    cv_targets=np.asarray(ep_d.get('cv_targets', []),
                                            dtype='float32'),
                    cv_target_enabled=np.asarray(
                        ep_d.get('cv_target_enabled', []), dtype=bool),
                    current_mv_bounds_t=np.asarray(
                        ep_d.get('current_mv_bounds_t',
                                   np.zeros((0, 0, 2))), dtype='float32'),
                    current_cv_bounds_t=np.asarray(
                        ep_d.get('current_cv_bounds_t',
                                   np.zeros((0, 0, 2))), dtype='float32'),
                    current_cv_targets_t=np.asarray(
                        ep_d.get('current_cv_targets_t',
                                   np.zeros((0, 0))), dtype='float32'),
                    hidden_disturbance_t=np.asarray(
                        ep_d.get('hidden_disturbance_t',
                                   np.zeros((0, 0))), dtype='float32'),
                    sample_rate=np.asarray([int(ep_d.get('sample_rate', 1))],
                                             dtype='int64'),
                    episode_length=np.asarray([int(ep_d.get('episode_length',
                                                              len(ep_d['raw_rewards'])))],
                                                dtype='int64'),
                    state_variables=np.asarray(
                        ep_d.get('state_variables', []), dtype=object),
                    schedule_json=np.asarray([sched_json], dtype=object),
                )
            except Exception as ee:
                print(f'[val] disturbance_rejection.npz skipped '
                      f'(seed {seed}): {ee!r}', flush=True)
            disturbance_records.append({
                'seed': seed,
                'cum_raw_reward': ep_d['cum_raw_reward'],
                'mean_cv_violation': ep_d['mean_cv_violation'],
                'mean_mv_violation': ep_d['mean_mv_violation'],
                'event_annotations': ann or [],
                'schedule': ep_d['schedule'],
                # V3-parity episode KPIs (mv_tv, mv_activity_ratio,
                # mv_bound_hugging_score, mv_bound_usage,
                # mv_reversal_rate, economic_score, IAE/ITAE/ISE).
                'episode_metrics_agent': ep_metrics,
                'episode_metrics_baseline': base_metrics,
                # Per-event overshoot / settle / IAE_window + return-to-limit.
                'event_response': ev_metrics,
                'event_response_baseline': ev_metrics_b,
            })
        except Exception as e:
            import traceback
            print(f'[val] scripted-disturbance episode skipped (seed {seed}): {e!r}',
                  flush=True)
            traceback.print_exc()

        seed_results.append(eps)
        print(f'[val] seed {seed}: {len(eps)} episodes done', flush=True)

    plot_summary(seed_results, out_dir / 'summary.png',
                  title=f'{controller_dir.name}  validation summary  '
                        f'({len(seed_results)} seeds × {episodes} eps)')

    # ---- Training-stage + WM-fidelity diagnostics ------------------------
    # Run once per validation invocation on a fresh env so we don't pay
    # per-seed cost.  Tells the operator which training stage (P1 WM,
    # P2 reward MTP / BC, P3 actor-critic) is the bottleneck.
    try:
        from evaluation.diagnostics import compute_training_diagnostics
        diag_env = APCEnv(cfg, np.random.default_rng(99_999))
        # WM-fidelity probe: disable hidden OU so the WM is scored on
        # base-plant dynamics, not augmented-system dynamics.
        diag_env._disturbance_prob_override = 0.0
        if obs_norm_state is not None:
            try:
                diag_env.set_obs_norm_stats(
                    mean=np.asarray(obs_norm_state.get('mean')),
                    var=np.asarray(obs_norm_state.get('var')),
                    count=float(obs_norm_state.get('count', 1.0)),
                    learn=False,
                )
            except Exception:
                pass
        diag = compute_training_diagnostics(
            controller_dir=controller_dir,
            env=diag_env,
            model=model,
            device=device,
            out_dir=out_dir,
            k_max=int(getattr(cfg, 'horizon', 32)),
            gamma=float(getattr(cfg, 'gamma', 0.997)),
        )
        flags = (diag.get('stage_metrics') or {}).get('flags') or []
        if flags:
            print('[val] training-stage flags:', flush=True)
            for fl in flags:
                print(f'        - {fl}', flush=True)

        # Internal-fidelity gates (2026-05-06).  Independent of the
        # economic / disturbance-rejection plots, these flag whether the
        # *world model itself* is usable.  Thresholds chosen from the
        # validate-iter140 RCA: a healthy WM should yield at least
        # weak positive correlation between predictions and real
        # next-state / next-reward / Monte-Carlo return.  Anything
        # below these floors means downstream actor learning is
        # mathematically guaranteed to fail.
        try:
            wm = diag.get('wm_fidelity', {}) or {}
            rw = diag.get('reward_fidelity', {}) or {}
            cc = diag.get('critic_calib', {}) or {}
            wm_r1 = float(((wm.get('per_offset') or {}).get('1') or {}).get('r_mean', 0.0))
            rw_r0 = float(((rw.get('per_offset') or {}).get('0') or {}).get('r', 0.0))
            critic_r = float(cc.get('r_pearson', 0.0))
            fidelity_gates = {
                'wm_next_state_r_min': 0.5,
                'reward_head_r_min': 0.3,
                'critic_r_min': 0.3,
                'wm_next_state_r_observed': wm_r1,
                'reward_head_r_observed': rw_r0,
                'critic_r_observed': critic_r,
                'wm_pass': bool(wm_r1 >= 0.5),
                'reward_pass': bool(rw_r0 >= 0.3),
                'critic_pass': bool(critic_r >= 0.3),
            }
            # CONTROL-QUALITY gates.  Internal WM/critic floors can PASS
            # while the actor chatters.  MV oscillation is ALLOWED; the
            # CV must stay smooth (d2 / CV reversal) and beats_baseline
            # still flags a policy no better than doing nothing.
            # cv_opt_headroom is a residual, not an all_pass gate.
            try:
                _cq = control_quality_gates(
                    locals().get('disturbance_records') or [],
                    seed_metrics=locals().get('metrics_records') or [],
                )
                fidelity_gates.update(_cq)
            except Exception as _cge:
                fidelity_gates['smooth_pass'] = False
                fidelity_gates['beats_baseline_pass'] = False
                fidelity_gates['control_gate_error'] = repr(_cge)
            fidelity_gates['all_pass'] = bool(
                fidelity_gates['wm_pass']
                and fidelity_gates['reward_pass']
                and fidelity_gates['critic_pass']
                and fidelity_gates['smooth_pass']
                and fidelity_gates['beats_baseline_pass']
            )
            diag['fidelity_gates'] = fidelity_gates
            if not fidelity_gates['all_pass']:
                print('[val] internal-fidelity gates FAILED:', flush=True)
                if not fidelity_gates['wm_pass']:
                    print(f'        - WM next-state r={wm_r1:+.3f} < 0.5'
                          ' (encoder/dynamics not learning plant)', flush=True)
                if not fidelity_gates['reward_pass']:
                    print(f'        - reward head r={rw_r0:+.3f} < 0.3'
                          ' (reward MTP uncorrelated with truth)', flush=True)
                if not fidelity_gates['critic_pass']:
                    print(f'        - critic V vs MC r={critic_r:+.3f} < 0.3'
                          ' (value head uncorrelated with returns)', flush=True)
                if not fidelity_gates.get('smooth_pass', True):
                    print(f'        - CV smoothness FAIL: '
                          f'cv_d2_rms_normed='
                          f'{fidelity_gates.get("cv_d2_rms_normed_observed", float("nan")):.4f}'
                          f' (gate ≤ {CV_D2_RMS_GATE})  cv_reversal_rate='
                          f'{fidelity_gates.get("cv_reversal_rate_observed", float("nan")):.3f}'
                          f' (gate ≤ {CV_REVERSAL_GATE}); '
                          f'mv_reversal='
                          f'{fidelity_gates.get("mv_reversal_rate_observed", float("nan")):.3f}'
                          ' is diagnostic only (MV oscillation allowed)',
                          flush=True)
                if not fidelity_gates.get('beats_baseline_pass', True):
                    if fidelity_gates.get('control_gate_skipped'):
                        print('        - no scripted agent/baseline pairs '
                              f'({fidelity_gates.get("control_gate_skipped")}); '
                              'cannot claim beats-baseline (P49 0-vs-0 false pass)',
                              flush=True)
                    else:
                        print(f'        - agent economic_score='
                              f'{fidelity_gates.get("agent_economic_score", 0.0):+.4f}'
                              f' < baseline={fidelity_gates.get("baseline_economic_score", 0.0):+.4f}'
                              ' (policy WORSE than open-loop baseline)', flush=True)
            else:
                print(f'[val] internal-fidelity gates PASSED '
                      f'(wm_r={wm_r1:+.3f} rw_r={rw_r0:+.3f} '
                      f'critic_r={critic_r:+.3f} '
                      f'cv_d2={fidelity_gates.get("cv_d2_rms_normed_observed", float("nan")):.4f} '
                      f'cv_rev={fidelity_gates.get("cv_reversal_rate_observed", float("nan")):.3f} '
                      f'mv_rev={fidelity_gates.get("mv_reversal_rate_observed", float("nan")):.3f}'
                      f' [MV osc allowed])',
                      flush=True)
        except Exception as _ge:
            print(f'[val] fidelity-gate computation skipped: {_ge!r}',
                  flush=True)
            diag['fidelity_gates'] = {'error': repr(_ge)}
    except Exception as e:
        print(f'[val] diagnostics skipped: {e}', flush=True)

    # ---- WM transfer-function (step-response) matrix ---------------------
    # DMC-style per-MV/CV step-response curves (WM vs real sim) averaged over
    # the operating region with a min/max variation band.  Directly measures
    # whether the world model captured the true GAINS + DYNAMICS (the
    # correlation-based fidelity probe does NOT).  Gated ON by default; skip
    # with DREAMER_VAL_WM_TRANSFER=0 (TrainConfig ``val_wm_transfer``).
    from evaluation.wm_transfer_matrix import (
        compute_and_plot, resolve_wm_tf_knobs, val_diag_enabled, wm_tf_roll_len)
    if val_diag_enabled(cfg, 'val_wm_transfer', 'DREAMER_VAL_WM_TRANSFER'):
        try:
            tf_env = APCEnv(cfg, np.random.default_rng(77_777))
            tf_env._disturbance_prob_override = 0.0
            tf_obs_std = None
            if obs_norm_state is not None:
                try:
                    _var = np.asarray(obs_norm_state.get('var'), dtype='float32')
                    tf_obs_std = np.clip(np.sqrt(np.maximum(_var, 1e-6)), 1e-3, None)
                    tf_env.set_obs_norm_stats(
                        mean=np.asarray(obs_norm_state.get('mean')), var=_var,
                        count=float(obs_norm_state.get('count', 1.0)),
                        learn=False)
                except Exception:
                    pass
            tf_result = compute_and_plot(
                model, tf_env, cfg, device, out_dir, obs_std=tf_obs_std,
                title=f'{controller_dir.name}  observer transfer matrix')
            # GAIN-FIDELITY GATE (control-relevant; the correlation-based
            # wm_next_state_r does NOT measure gain).  Mean relative SS-gain
            # error across MV/CV pairs; a WM usable for control needs the gain
            # within ~2× of the real plant (rel_err < 1.0; healthy < 0.35).
            try:
                pairs = (tf_result or {}).get('pairs', {}) if tf_result else {}
                rel_errs = _ss_gain_rel_errs(pairs)
                if rel_errs:
                    gain_rel_err = float(np.mean(rel_errs))
                    gate = {
                        'wm_gain_rel_err': gain_rel_err,
                        'wm_gain_rel_err_max': float(np.max(rel_errs)),
                        'wm_gain_pass': bool(gain_rel_err < 1.0),
                        'wm_gain_healthy': bool(gain_rel_err < 0.35),
                        'n_pairs': len(rel_errs),
                    }
                    # DV is a separate JSON (not in MV wm_gain_rel_err). P29
                    # printed wm_gain_healthy=True at MV rel_err=0.10 while
                    # DV ss was ×0.56 — the MV-only aggregate hid it.
                    dv_gate = _dv_gain_gate_from_json(
                        out_dir / 'wm_dv_transfer_matrix.json')
                    if dv_gate:
                        gate.update(dv_gate)
                    _merge_observer_gain_gate(gate, dv_gate)
                    if isinstance(locals().get('fidelity_gates'), dict):
                        fidelity_gates.update(gate)
                    else:
                        fidelity_gates = gate
                    mv_status = _gain_status(
                        gate['wm_gain_healthy'], gate['wm_gain_pass'])
                    print(f'[val] WM MV gain fidelity: rel_err={gain_rel_err:.2f} '
                          f'({mv_status}; lineage wm_gain_pass is MV-only)',
                          flush=True)
                    if dv_gate:
                        dv_status = _gain_status(
                            dv_gate['wm_dv_gain_healthy'],
                            dv_gate['wm_dv_gain_pass'])
                        print(f'[val] WM DV gain fidelity: rel_err='
                              f'{dv_gate["wm_dv_gain_rel_err"]:.2f} '
                              f'ss_ratio_worst='
                              f'{dv_gate.get("wm_dv_ss_ratio_worst", float("nan")):.2f} '
                              f'({dv_status})',
                              flush=True)
                    obs_status = _gain_status(
                        gate['wm_observer_gain_healthy'],
                        gate['wm_observer_gain_pass'])
                    print(f'[val] WM observer gain: {obs_status} '
                          f'(MV+DV; correlation gates can pass while this '
                          f'fails — gain is the control-relevant metric)',
                          flush=True)
            except Exception as _ge:
                print(f'[val] WM gain-gate skipped: {_ge!r}', flush=True)
        except Exception as e:
            print(f'[val] WM transfer matrix skipped: {e!r}', flush=True)

    # ---- WM posterior-vs-prior gain-lag decomposition --------------------
    # Localises WHERE the WM loses the steady-state gain (autoencoder vs the
    # prior<->posterior gap vs open-loop compounding) so the right lever is
    # obvious from the saved artefact alone — no manual probe re-run.  Gated
    # ON by default (RSSM/TSSM only); skip with DREAMER_VAL_WM_POSTPRIOR=0
    # (TrainConfig ``val_wm_postprior``).
    # Reuses a fresh disturbance-free env; guarded so it never breaks a run.
    if val_diag_enabled(cfg, 'val_wm_postprior', 'DREAMER_VAL_WM_POSTPRIOR'):
        try:
            from tools.wm_posterior_prior_probe import compute_posterior_prior_decomp
            pp_env = APCEnv(cfg, np.random.default_rng(43_210))
            pp_env._disturbance_prob_override = 0.0
            if obs_norm_state is not None:
                try:
                    pp_env.set_obs_norm_stats(
                        mean=np.asarray(obs_norm_state.get('mean')),
                        var=np.asarray(obs_norm_state.get('var')),
                        count=float(obs_norm_state.get('count', 1.0)),
                        learn=False)
                except Exception:
                    pass
            _tf_h = resolve_wm_tf_knobs(cfg)['horizon']
            _pp_h = wm_tf_roll_len(cfg, _tf_h)
            pp_res = compute_posterior_prior_decomp(
                model, pp_env, cfg, device, horizon=_pp_h, settle=_pp_h)
            with open(out_dir / 'wm_posterior_prior_decomp.json', 'w') as f:
                json.dump(pp_res, f, indent=2)
            if pp_res.get('enabled'):
                print(f'[val] WM posterior/prior gain decomp: '
                      f'real->post x{pp_res["decomp_real_to_posterior"]:.3f}, '
                      f'post->1step x{pp_res["decomp_posterior_to_1step"]:.3f}, '
                      f'1step->openloop x{pp_res["decomp_1step_to_openloop"]:.3f} '
                      f'| lever={pp_res["dominant_lever"]} -> '
                      f'{out_dir}/wm_posterior_prior_decomp.json', flush=True)
            else:
                print(f'[val] WM posterior/prior decomp: not applicable '
                      f'({pp_res.get("reason")})', flush=True)
        except Exception as _ppe:
            print(f'[val] WM posterior/prior decomp skipped: {_ppe!r}', flush=True)

    # ---- WM DV→CV posterior-vs-prior gain-lag decomposition (2026-06-19) ----
    # The MV decomp above localises the MV gain loss; this is the DV analogue —
    # it localises WHERE the measured-DV→CV gain is lost (autoencoder vs prior).
    # p129 RCA: across a 9-run DV-bias plateau the DV gain (~0.76) had NEVER
    # been localised; this decomp showed real→post ×0.767 / post→1step ×1.002
    # ⇒ the DV gain dies ENTIRELY in the autoencoder (not the prior, not data),
    # explaining why every excitation/data fix failed.  Saved as
    # wm_dv_posterior_prior_decomp.json.  ON by default (RSSM/TSSM + DV-as-input
    # + sim.set_disturbance_offset); shares the DREAMER_VAL_WM_POSTPRIOR gate.
    if val_diag_enabled(cfg, 'val_wm_postprior', 'DREAMER_VAL_WM_POSTPRIOR'):
        try:
            from tools.wm_posterior_prior_probe import compute_dv_posterior_prior_decomp
            dpp_env = APCEnv(cfg, np.random.default_rng(43_211))
            dpp_env._disturbance_prob_override = 0.0
            if obs_norm_state is not None:
                try:
                    dpp_env.set_obs_norm_stats(
                        mean=np.asarray(obs_norm_state.get('mean')),
                        var=np.asarray(obs_norm_state.get('var')),
                        count=float(obs_norm_state.get('count', 1.0)),
                        learn=False)
                except Exception:
                    pass
            _tf_h2 = resolve_wm_tf_knobs(cfg)['horizon']
            _dpp_h = wm_tf_roll_len(cfg, _tf_h2)
            dvpp_res = compute_dv_posterior_prior_decomp(
                model, dpp_env, cfg, device, horizon=_dpp_h, settle=_dpp_h)
            with open(out_dir / 'wm_dv_posterior_prior_decomp.json', 'w') as f:
                json.dump(dvpp_res, f, indent=2)
            if dvpp_res.get('enabled'):
                print(f'[val] WM DV→CV posterior/prior gain decomp: '
                      f'real->post x{dvpp_res["decomp_real_to_posterior"]:.3f}, '
                      f'post->1step x{dvpp_res["decomp_posterior_to_1step"]:.3f} '
                      f'| lever={dvpp_res["dominant_lever"]} -> '
                      f'{out_dir}/wm_dv_posterior_prior_decomp.json', flush=True)
            else:
                print(f'[val] WM DV→CV posterior/prior decomp: not applicable '
                      f'({dvpp_res.get("reason")})', flush=True)
        except Exception as _dppe:
            print(f'[val] WM DV→CV posterior/prior decomp skipped: {_dppe!r}', flush=True)

    # ----- WM unmeasured-disturbance PREDICTION diagnostic (2026-06-10) -----
    # How well does the WM disturbance-estimator head (P87 feed-forward model)
    # predict the TRUE hidden OU disturbance the agent can't see?  Rolls one
    # forced-disturbance episode, runs the head over the streamed WM posterior,
    # and scores pred-vs-true per CV channel (NRMSE / r / R² / lead-lag).  Saves
    # wm_disturbance_prediction.{json,png}.  ON by default (RSSM/TSSM + head);
    # skip with DREAMER_VAL_WM_DISTPRED=0 (TrainConfig ``val_wm_distpred``).
    if val_diag_enabled(cfg, 'val_wm_distpred', 'DREAMER_VAL_WM_DISTPRED'):
        try:
            from evaluation.wm_disturbance_prediction import (
                compute_disturbance_prediction, plot_disturbance_prediction)
            dp_env = APCEnv(cfg, np.random.default_rng(51_234))
            if obs_norm_state is not None:
                try:
                    dp_env.set_obs_norm_stats(
                        mean=np.asarray(obs_norm_state.get('mean')),
                        var=np.asarray(obs_norm_state.get('var')),
                        count=float(obs_norm_state.get('count', 1.0)),
                        learn=False)
                except Exception:
                    pass
            dp_res = compute_disturbance_prediction(
                model, dp_env, cfg, device, deterministic=deterministic)
            with open(out_dir / 'wm_disturbance_prediction.json', 'w') as f:
                json.dump(dp_res, f, indent=2)
            if dp_res.get('enabled'):
                plot_disturbance_prediction(
                    dp_res, out_dir / 'wm_disturbance_prediction.png')
                print(f'[val] WM disturbance prediction: '
                      f'mean NRMSE={dp_res["mean_nrmse"]:.3f} '
                      f'r={dp_res["mean_pearson_r"]:.3f} '
                      f'R²={dp_res["mean_r2"]:.3f} | '
                      f'DETRENDED (control-relevant) '
                      f'r={dp_res.get("mean_pearson_r_detrended", float("nan")):.3f} '
                      f'R²={dp_res.get("mean_r2_detrended", float("nan")):.3f} '
                      f'(drift_sd={dp_res.get("mean_drift_err_std", float("nan")):.3f} '
                      f'dyn_sd={dp_res.get("mean_dyn_err_std", float("nan")):.3f}) -> '
                      f'{out_dir}/wm_disturbance_prediction.png', flush=True)
            else:
                print(f'[val] WM disturbance prediction: not applicable '
                      f'({dp_res.get("reason")})', flush=True)
        except Exception as _dpe:
            print(f'[val] WM disturbance prediction skipped: {_dpe!r}', flush=True)

    cum = np.array([m['cum_raw_reward'] for m in metrics_records])
    cv_v = np.array([m['mean_cv_violation'] for m in metrics_records])
    mv_v = np.array([m['mean_mv_violation'] for m in metrics_records])
    dv_tf_loaded = None
    try:
        _dv_tf_path = out_dir / 'wm_dv_transfer_matrix.json'
        if _dv_tf_path.exists():
            dv_tf_loaded = json.loads(_dv_tf_path.read_text())
    except Exception:
        dv_tf_loaded = None
    residual_board = build_residual_board(
        fidelity_gates=locals().get('fidelity_gates'),
        mv_tf=locals().get('tf_result'),
        dv_tf=dv_tf_loaded,
        postprior=locals().get('pp_res'),
        distpred=locals().get('dp_res'),
        disturbance_records=disturbance_records,
        seed_metrics=metrics_records,
    )
    residual_board = _jsonable(residual_board)
    try:
        with open(out_dir / 'residual_board.json', 'w') as f:
            json.dump(residual_board, f, indent=2)
        r1 = residual_board.get('r1_observer_tm') or {}
        r2 = residual_board.get('r2_cv_quality') or {}
        r3 = residual_board.get('r3_unmeasured_dr') or {}
        print('[val] residual board (MV osc allowed; family-closed ≠ closed):',
              flush=True)
        print(f'        R1 TM  mv_ss={r1.get("mv_ss_ratio_mean")} '
              f'dv_ss={r1.get("dv_ss_ratio_mean")} '
              f'curve_iae_mv={r1.get("mv_curve_iae_mean")} '
              f'1step→OL={r1.get("compound_1step_to_openloop")}',
              flush=True)
        print(f'        R2 CV  d2={r2.get("cv_d2_rms_normed_worst_seed")} '
              f'rev={r2.get("cv_reversal_rate_worst_seed")} '
              f'headroom={r2.get("cv_opt_headroom_mean")} '
              f'viol_frac={r2.get("cv_viol_frac_worst_seed")} '
              f'mv_rev={r2.get("mv_reversal_rate_observed")} '
              f'(diagnostic)',
              flush=True)
        print(f'        R3 DR  det_r={r3.get("kalman_det_r")} '
              f'pred_std={r3.get("kalman_pred_std")} '
              f'true_std={r3.get("kalman_true_std")} '
              f'iae_ratio={r3.get("iae_agent_over_baseline_mean")} '
              f'return_headroom={r3.get("cv_return_headroom_mean")} '
              f'return_time_frac={r3.get("cv_return_time_frac_mean")} '
              f'-> {out_dir}/residual_board.json',
              flush=True)
    except Exception as _rbe:
        print(f'[val] residual_board skipped: {_rbe!r}', flush=True)
        residual_board = {'error': repr(_rbe)}
    summary = {
        'controller_dir': str(controller_dir),
        'simulation_dir': str(sim_dir),
        'ckpt': str(ckpt_path),
        'deterministic': deterministic,
        'policy_type': str(getattr(cfg, 'policy_type', 'continuous')),
        'actor_loss_type': str(getattr(cfg, 'actor_loss_type', 'reinforce')),
        'policy_init_log_std': float(getattr(cfg, 'policy_init_log_std', -0.5)),
        'n_seeds': int(seeds),
        'episodes_per_seed': int(episodes),
        'n_episodes_total': int(len(metrics_records)),
        'cum_raw_reward_mean': float(cum.mean()),
        'cum_raw_reward_std': float(cum.std()),
        'cum_raw_reward_min': float(cum.min()),
        'cum_raw_reward_max': float(cum.max()),
        'mean_cv_violation_mean': float(cv_v.mean()),
        'mean_mv_violation_mean': float(mv_v.mean()),
        'fidelity_gates': locals().get('fidelity_gates', None),
        'wm_posterior_prior_decomp': locals().get('pp_res', None),
        'wm_dv_posterior_prior_decomp': locals().get('dvpp_res', None),
        'wm_disturbance_prediction': locals().get('dp_res', None),
        'residual_board': residual_board,
        'episodes': metrics_records,
        'disturbance_rejection': disturbance_records,
    }
    with open(out_dir / 'validation_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print('[val] done.', flush=True)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description='Validate a trained DreamerV4 controller against held-out '
                    'episodes drawn from the training disturbance distribution.')
    parser.add_argument('--controller-dir', '-c', required=True,
                        help='Output directory of a training run (contains '
                             'final.pt, run_plan.json).')
    parser.add_argument('--simulation-dir', '-s', default=None,
                        help='Override the simulation directory '
                             '(default: read from run_plan.json).')
    parser.add_argument('--ckpt', default='final.pt',
                        help='Checkpoint filename within --controller-dir.')
    parser.add_argument('--episodes', type=int, default=3,
                        help='Episodes per seed.')
    parser.add_argument('--seeds', type=int, default=3,
                        help='Number of validation seeds.')
    parser.add_argument('--out', default=None,
                        help='Validation output dir (default: '
                             '<controller-dir>/validation).')
    parser.add_argument('--stochastic', action='store_true',
                        help='Sample actions stochastically '
                             '(default: deterministic argmax).')
    args = parser.parse_args()
    summary = run_validation(controller_dir=args.controller_dir,
                             simulation_dir=args.simulation_dir,
                             ckpt=args.ckpt,
                             episodes=args.episodes, seeds=args.seeds,
                             out=args.out,
                             deterministic=not args.stochastic)
    print(json.dumps({k: v for k, v in summary.items() if k != 'episodes'},
                     indent=2), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
