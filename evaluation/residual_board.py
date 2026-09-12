"""Standing R1/R2/R3 residual board written at validation.

Scores the product (smooth CV on the economic limit, faithful observer,
unmeasured-load rejection) — not VALID 9/9 / GAIN-READY / all_pass.
``smooth_pass`` is CV d2/reversal only; ``mv_reversal`` is diagnostic.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

CV_D2_RMS_MAX = 0.05
CV_REVERSAL_MAX = 0.25
CV_RETURN_HEADROOM_BAND = 0.10
# Parked #11 gate (not in smooth_pass). Crossings of y_econ per identified τ.
CV_LIMIT_ORBIT_MAX = 0.15
# Pearson r is scale-invariant (P127 r=+0.318 with slope 289 / V−81 vs G−15318).
# Order-1 same-sign slope: [0.25, 4] ≈ |log10(s)| ≲ 0.6.
CRITIC_R_MIN = 0.3
CRITIC_SLOPE_LO = 0.25
CRITIC_SLOPE_HI = 4.0


def _f(x: Any, default: float = float('nan')) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return default
    return v if np.isfinite(v) else default


def critic_fidelity_pass(r_pearson, slope_g_on_v) -> bool:
    """Val critic gate: Pearson AND order-1 same-sign slope.

    ``r>=0.3`` alone is on trial (P127 PASSED it while V was 200× compressed).
    Negative slope is inversion (P125 **−15**). P3-skip / nan → False.
    """
    r = _f(r_pearson)
    s = _f(slope_g_on_v)
    if not (np.isfinite(r) and np.isfinite(s)):
        return False
    return r >= CRITIC_R_MIN and CRITIC_SLOPE_LO <= s <= CRITIC_SLOPE_HI


def _json_load(path: Path) -> Optional[Dict]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def curve_iae_normed(wm_mean, real_mean, real_ss: float) -> float:
    """Mean |wm−real| / |real SS|. Shape residual; ss-ratio is DC only."""
    wm = np.asarray(wm_mean, dtype='float64').reshape(-1)
    real = np.asarray(real_mean, dtype='float64').reshape(-1)
    n = int(min(wm.size, real.size))
    if n == 0:
        return float('nan')
    iae = float(np.mean(np.abs(wm[:n] - real[:n])))
    den = abs(float(real_ss))
    if not np.isfinite(den) or den < 1e-9:
        den = float(np.mean(np.abs(real[:n]))) or 1.0
    return iae / max(1e-9, den)


def _pair_cell(pairs: Dict) -> Tuple[str, Dict]:
    if not pairs:
        return '', {}
    best_k, best_v, best_d = '', {}, -1.0
    for k, v in pairs.items():
        if not isinstance(v, dict):
            continue
        r = _f(v.get('ss_gain_ratio_wm_over_real'))
        d = abs(r - 1.0) if np.isfinite(r) else 0.0
        if d >= best_d:
            best_k, best_v, best_d = str(k), v, d
    return best_k, best_v


def _tm_axis(tm: Optional[Dict]) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        'ss_ratio': float('nan'),
        'h_ratio': float('nan'),
        'curve_iae_normed': float('nan'),
        'real_ss_gain': float('nan'),
        'pair': None,
    }
    if not tm:
        return out
    key, cell = _pair_cell(tm.get('pairs') or {})
    if not cell:
        return out
    out['pair'] = key
    out['ss_ratio'] = _f(cell.get('ss_gain_ratio_wm_over_real'))
    out['h_ratio'] = _f(cell.get('gain_ratio_at_h'))
    out['real_ss_gain'] = _f(cell.get('real_ss_gain'))
    stored = cell.get('curve_iae_normed')
    if stored is not None and np.isfinite(_f(stored)):
        out['curve_iae_normed'] = _f(stored)
    else:
        wm = (cell.get('wm') or {}).get('mean')
        real = (cell.get('real') or {}).get('mean')
        out['curve_iae_normed'] = curve_iae_normed(
            wm, real, _f(cell.get('real_ss_gain'), 0.0))
    return out


def _median(xs: List[float]) -> float:
    vals = [float(x) for x in xs if np.isfinite(float(x))]
    if not vals:
        return float('nan')
    return float(np.median(np.asarray(vals, dtype='float64')))


def _max_finite(xs: List[float]) -> float:
    vals = [float(x) for x in xs if np.isfinite(float(x))]
    if not vals:
        return float('nan')
    return float(np.max(np.asarray(vals, dtype='float64')))


def _mean_finite(xs: List[float]) -> float:
    vals = [float(x) for x in xs if np.isfinite(float(x))]
    if not vals:
        return float('nan')
    return float(np.mean(np.asarray(vals, dtype='float64')))


def cv_smooth_pass(worst_d2: float, worst_rev: float) -> bool:
    return bool(np.isfinite(worst_d2) and np.isfinite(worst_rev)
                and worst_d2 <= CV_D2_RMS_MAX
                and worst_rev <= CV_REVERSAL_MAX)


def build_residual_board(out_dir: Path, summary: Optional[Dict] = None
                         ) -> Dict[str, Any]:
    """Assemble R1/R2/R3 from val artefacts. Missing axes stay NaN."""
    out_dir = Path(out_dir)
    if summary is None:
        summary = _json_load(out_dir / 'validation_summary.json') or {}
    tm_mv = _json_load(out_dir / 'wm_transfer_matrix.json')
    tm_dv = _json_load(out_dir / 'wm_dv_transfer_matrix.json')
    decomp = (_json_load(out_dir / 'wm_posterior_prior_decomp.json')
              or summary.get('wm_posterior_prior_decomp') or {})
    dist = (_json_load(out_dir / 'wm_disturbance_prediction.json')
            or summary.get('wm_disturbance_prediction') or {})
    mv = _tm_axis(tm_mv)
    dv = _tm_axis(tm_dv)

    r1 = {
        'mv_ss_ratio': mv['ss_ratio'],
        'mv_h_ratio': mv['h_ratio'],
        'mv_curve_iae_normed': mv['curve_iae_normed'],
        'mv_pair': mv['pair'],
        'dv_ss_ratio': dv['ss_ratio'],
        'dv_h_ratio': dv['h_ratio'],
        'dv_curve_iae_normed': dv['curve_iae_normed'],
        'dv_pair': dv['pair'],
        'ol_1step_ratio': _f(decomp.get('decomp_1step_to_openloop')),
        'real_to_post': _f(decomp.get('decomp_real_to_posterior')),
        'post_to_1step': _f(decomp.get('decomp_posterior_to_1step')),
        'ol_vs_real': _f(decomp.get('gain_ratio_openloop_vs_real')),
        'dominant_lever': decomp.get('dominant_lever'),
    }

    dr = list(summary.get('disturbance_rejection') or [])
    seeds = list(summary.get('episodes') or [])
    d2s, revs, orbits, heads, viols = [], [], [], [], []
    for rec in dr:
        m = rec.get('episode_metrics_agent') or {}
        d2s.append(_f(m.get('cv_d2_rms_normed')))
        revs.append(_f(m.get('cv_reversal_rate')))
        orbits.append(_f(m.get('cv_limit_orbit_rate', m.get('cv_limit_orbit_frac'))))
        heads.append(_f(m.get('cv_opt_headroom')))
        viols.append(_f(m.get('cv_viol_frac')))
    if not any(np.isfinite(x) for x in d2s):
        d2s, revs, orbits, heads, viols = [], [], [], [], []
        for rec in seeds:
            d2s.append(_f(rec.get('kpi_cv_d2_rms_normed')))
            revs.append(_f(rec.get('kpi_cv_reversal_rate')))
            orbits.append(_f(rec.get('kpi_cv_limit_orbit_rate',
                                    rec.get('kpi_cv_limit_orbit_frac'))))
            heads.append(_f(rec.get('kpi_cv_opt_headroom')))
            viols.append(_f(rec.get('kpi_cv_viol_frac')))

    ret_heads, ret_fracs, iaes = [], [], []
    for rec in dr:
        ev = rec.get('event_response') or {}
        ret_heads.append(_f((ev.get('cv_return_headroom') or {}).get('median')
                            if isinstance(ev.get('cv_return_headroom'), dict)
                            else ev.get('cv_return_headroom')))
        ret_fracs.append(_f((ev.get('cv_return_time_frac') or {}).get('median')
                            if isinstance(ev.get('cv_return_time_frac'), dict)
                            else ev.get('cv_return_time_frac')))
        iae = ev.get('iae_window_normed')
        if isinstance(iae, dict):
            iaes.append(_f(iae.get('median')))
        else:
            iaes.append(_f(iae))

    json_d2_ok = any(np.isfinite(x) for x in d2s)
    json_orbit_ok = any(np.isfinite(x) for x in orbits)
    json_ret_ok = any(np.isfinite(x) for x in ret_heads)
    json_iae_ok = any(np.isfinite(x) for x in iaes)
    # Backfill missing axes independently. JSON IAE medians are the
    # val-suite score; npz recomputes a different window sum and must
    # not overwrite a finite JSON IAE just because return-to-limit
    # keys were added later (P119 board 64.36 vs summary 20.3).
    if ((not json_d2_ok) or (not json_orbit_ok)
            or (not json_ret_ok) or (not json_iae_ok)):
        npz = _metrics_from_val_npz(out_dir)
        if npz is not None:
            if not json_d2_ok:
                d2s, revs, heads, viols = npz[:4]
            if not json_orbit_ok:
                orbits = npz[7] if len(npz) > 7 else []
            if not json_ret_ok:
                ret_heads, ret_fracs = npz[4], npz[5]
            if not json_iae_ok:
                iaes = npz[6]

    worst_d2 = _max_finite(d2s)
    worst_rev = _max_finite(revs)
    worst_orbit = _max_finite(orbits)
    r2 = {
        'cv_d2_rms_normed_worst_seed': worst_d2,
        'cv_reversal_rate_worst_seed': worst_rev,
        'cv_limit_orbit_rate_worst_seed': worst_orbit,
        'cv_limit_orbit_rate_max': CV_LIMIT_ORBIT_MAX,
        'cv_opt_headroom_mean': _mean_finite(heads),
        'cv_viol_frac_mean': _mean_finite(viols),
        'smooth_pass': cv_smooth_pass(worst_d2, worst_rev),
        'smooth_pass_rule': (
            f'worst-seed cv_d2_rms_normed<={CV_D2_RMS_MAX} AND '
            f'cv_reversal_rate<={CV_REVERSAL_MAX} (not mv_reversal; '
            f'cv_limit_orbit_rate is diagnostic until #11)'),
        'orbit_note': (
            'cv_limit_orbit_rate = (CV−y_econ) sign-changes × (τ/sr) / T '
            '(crossings per identified τ). Diagnostic only; not a gate. '
            '1τ-window ≥2 frac FALSIFIED on P125 hunt (period ~2τ). '
            f'Parked #11 threshold {CV_LIMIT_ORBIT_MAX} /τ.'),
        'mv_reversal_rate_observed': _f(
            ((summary.get('fidelity_gates') or {}).get(
                'mv_reversal_rate_observed'))),
        'econ_side': None,
    }
    for rec in dr:
        side = (rec.get('episode_metrics_agent') or {}).get('cv_econ_side')
        if side:
            r2['econ_side'] = side
            break
    if r2['econ_side'] is None:
        rg = mv.get('real_ss_gain')
        if np.isfinite(_f(rg)) and abs(_f(rg)) > 1e-9:
            r2['econ_side'] = 'hi' if _f(rg) < 0 else 'lo'

    per = (dist.get('per_channel') or [None])[0] or {}
    r3 = {
        'det_r': _f(dist.get('mean_pearson_r_detrended')),
        'r2_det': _f(dist.get('mean_r2_detrended')),
        'pred_std': _f(per.get('pred_std')),
        'true_std': _f(per.get('true_std')),
        'event_iae_median_of_medians': _median(iaes),
        'cv_return_headroom': _median(ret_heads),
        'cv_return_time_frac': _median(ret_fracs),
    }

    gates = summary.get('fidelity_gates') or {}
    cc = {}
    diag_js = _json_load(out_dir / 'wm_diagnostics.json') or {}
    if isinstance(diag_js.get('critic_calib'), dict):
        cc = diag_js['critic_calib']
    board = {
        'schema': 'residual_board.v1',
        'controller_dir': summary.get('controller_dir'),
        'simulation_dir': summary.get('simulation_dir'),
        'ckpt': summary.get('ckpt'),
        'r1': r1,
        'r2': r2,
        'r3': r3,
        'diagnostic': {
            'actor_economic_score': _f(gates.get('agent_economic_score')),
            'baseline_economic_score': _f(gates.get('baseline_economic_score')),
            'beats_baseline_pass': bool(gates.get('beats_baseline_pass')),
            'critic_r': _f(gates.get('critic_r_observed')),
            # Scale, not Pearson: P127 r=+0.318 with slope_g_on_v=289 / V−81 vs G−15318.
            'critic_v_mean': _f(cc.get('v_mean', gates.get('critic_v_mean'))),
            'critic_g_mean': _f(cc.get('g_mean', gates.get('critic_g_mean'))),
            'critic_g_raw_mean': _f(
                cc.get('g_raw_mean', gates.get('critic_g_raw_mean'))),
            'critic_slope_g_on_v': _f(
                cc.get('slope_g_on_v', gates.get('critic_slope_g_on_v'))),
            'critic_nmae': _f(cc.get('nmae', gates.get('critic_nmae'))),
            # Recompute — do not copy Pearson-only summary.critic_pass (P127 True).
            'critic_pass': critic_fidelity_pass(
                gates.get('critic_r_observed'),
                cc.get('slope_g_on_v', gates.get('critic_slope_g_on_v'))),
            'all_pass': gates.get('all_pass'),
            'n_scripted_pairs': gates.get('n_scripted_pairs'),
        },
        'note': (
            'R1 = observer TM (ss + @H + curve_iae, MV and DV) and 1step→OL. '
            'R2 = CV smoothness (d2/reversal) and limit hugging/viol; '
            'cv_limit_orbit_rate is diagnostic (not a gate). '
            'R3 = Kalman det_r + pred_std vs true and DR return-to-limit. '
            'Critic: critic_pass = r≥0.3 AND slope_g_on_v in [0.25, 4] on '
            'matched training units (bound-shaped econ; g_raw_mean is '
            'honest-econ RCA, not the gate). '
            'Pearson without slope is not residual-closed. '
            'Do not treat VALID/GAIN-READY/all_pass/family-closed as residual-closed.'
        ),
    }
    return board


def _metrics_from_val_npz(out_dir: Path):
    """Backfill CV KPIs from ``seed_*/disturbance_rejection.npz`` (pre-R2 vals)."""
    try:
        from evaluation.validate import (
            compute_episode_metrics, compute_event_response_metrics)
    except Exception:
        return None
    d2s, revs, heads, viols = [], [], [], []
    ret_h, ret_t, iaes = [], [], []
    orbits: List[float] = []
    pid = (_json_load(Path(out_dir) / 'plant_id.json')
           or _json_load(Path(out_dir).parent / 'plant_id.json')
           or {})
    tau_dom = _f(pid.get('tau'), 0.0)
    found = False
    for npz_path in sorted(Path(out_dir).glob('seed_*/disturbance_rejection.npz')):
        try:
            z = np.load(npz_path, allow_pickle=True)
        except (OSError, ValueError):
            continue
        try:
            sched_raw = z['schedule_json']
            if hasattr(sched_raw, 'tolist'):
                sched_raw = sched_raw.tolist()
            if isinstance(sched_raw, (list, np.ndarray)) and len(sched_raw) == 1:
                sched_raw = sched_raw[0]
            if isinstance(sched_raw, bytes):
                sched_raw = sched_raw.decode('utf-8')
            schedule = json.loads(str(sched_raw)) if sched_raw else []
            n_cv = max(1, len(np.asarray(z['cv_indices']).tolist()))
            gsign = 0.0
            tm = _json_load(Path(out_dir) / 'wm_transfer_matrix.json')
            if tm:
                _k, cell = _pair_cell(tm.get('pairs') or {})
                rg = _f(cell.get('real_ss_gain'))
                if np.isfinite(rg) and abs(rg) > 1e-9:
                    gsign = float(np.sign(rg))
            ep = {
                'states': np.asarray(z['states']),
                'controls': np.asarray(z['controls']),
                'episode_length': int(np.asarray(z['episode_length']).reshape(-1)[0]),
                'cv_indices': [int(x) for x in np.asarray(z['cv_indices']).tolist()],
                'mv_bounds': np.asarray(z['mv_bounds']).tolist(),
                'cv_bounds': np.asarray(z['cv_bounds']).tolist(),
                'cv_target_enabled': (
                    np.asarray(z['cv_target_enabled']).tolist()
                    if 'cv_target_enabled' in z.files else []),
                'raw_rewards': np.asarray(z['raw_rewards']),
                'mean_cv_violation': float(np.mean(z['cv_violations']))
                    if 'cv_violations' in z.files else 0.0,
                'mean_mv_violation': float(np.mean(z['mv_violations']))
                    if 'mv_violations' in z.files else 0.0,
                'cum_raw_reward': float(np.sum(z['raw_rewards'])),
                'schedule': schedule,
                'sample_rate': int(np.asarray(z['sample_rate']).reshape(-1)[0])
                    if 'sample_rate' in z.files else 1,
                'cv_side_scale': [{'lo': 0.4, 'hi': 1.0}] * n_cv,
                'mv_cv_gain_sign': gsign,
                'tau_dominant': tau_dom,
            }
        except Exception:
            continue
        m = compute_episode_metrics(ep)
        ev = compute_event_response_metrics(ep)
        found = True
        d2s.append(_f(m.get('cv_d2_rms_normed')))
        revs.append(_f(m.get('cv_reversal_rate')))
        orbits.append(_f(m.get('cv_limit_orbit_rate')))
        heads.append(_f(m.get('cv_opt_headroom')))
        viols.append(_f(m.get('cv_viol_frac')))
        rh = ev.get('cv_return_headroom') or {}
        rt = ev.get('cv_return_time_frac') or {}
        iae = ev.get('iae_window_normed') or {}
        ret_h.append(_f(rh.get('median') if isinstance(rh, dict) else rh))
        ret_t.append(_f(rt.get('median') if isinstance(rt, dict) else rt))
        iaes.append(_f(iae.get('median') if isinstance(iae, dict) else iae))
    if not found:
        return None
    return d2s, revs, heads, viols, ret_h, ret_t, iaes, orbits


def write_residual_board(out_dir: Path, summary: Optional[Dict] = None
                         ) -> Dict[str, Any]:
    board = build_residual_board(out_dir, summary=summary)
    path = Path(out_dir) / 'residual_board.json'
    path.write_text(json.dumps(board, indent=2, allow_nan=True))
    return board
