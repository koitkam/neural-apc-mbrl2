"""CPU tests: CV smoothness + opt-headroom (MV chatter allowed).

The product goal is a CV that sits smoothly on the economic limit.
MV oscillation is allowed; it is a diagnostic, not an all_pass gate.
CV total variation is reported but not gated (a slow ride to the limit
is good).  cv_opt_headroom is a residual, not a skip gate.

Run:
  CUDA_VISIBLE_DEVICES="" PYTHONPATH=$PWD python tools/_test_cv_quality_metrics.py
"""
from __future__ import annotations

import numpy as np

from evaluation.validate import (
    CV_D2_RMS_GATE,
    CV_REVERSAL_GATE,
    build_residual_board,
    compute_episode_metrics,
    control_quality_gates,
    preferred_cv_side,
)
from evaluation.wm_transfer_matrix import _curve_shape_scores

LO, HI = 78.5, 85.5
WIDTH = HI - LO
T = 200
MID = 0.5 * (LO + HI)


def _make_ep(cv, mv, *, cv_econ=None, side_scale=None):
    cv = np.asarray(cv, dtype='float64').reshape(-1)
    mv = np.asarray(mv, dtype='float64').reshape(-1)
    n = int(cv.shape[0])
    return {
        'states': cv.reshape(n, 1).astype('float32'),
        'controls': mv.reshape(n, 1).astype('float32'),
        'cv_indices': [0],
        'cv_bounds': [[LO, HI]],
        'mv_bounds': [[20.0, 80.0]],
        'episode_length': n,
        'mean_cv_violation': 0.0,
        'mean_mv_violation': 0.0,
        'raw_rewards': np.zeros(n, dtype='float32'),
        'cum_raw_reward': 0.0,
        'obj_cv_economic_weights': (
            list(cv_econ) if cv_econ is not None else [0.0]),
        'obj_cv_side_scale': (
            side_scale if side_scale is not None
            else {'cv_0': {'lo': 0.4, 'hi': 1.0}}),
    }


def _square_mv(n: int = T) -> np.ndarray:
    return np.where((np.arange(n) // 1) % 2 == 0, 20.0, 80.0)


def _check(cond: bool, msg: str, ok: list) -> None:
    if cond:
        return
    print(f'FAIL: {msg}')
    ok.append(False)


def main() -> int:
    ok: list = []
    hug = np.full(T, HI - 0.05, dtype='float64')
    mid = np.full(T, MID, dtype='float64')
    over = np.full(T, HI + 1.0, dtype='float64')
    ramp = np.linspace(MID, HI - 0.05, T)
    bang_cv = np.where(np.arange(T) % 2 == 0, LO + 0.2, HI - 0.2)
    quiet_mv = np.full(T, 50.0)

    # 1. Smooth CV hugging HI, chattery MV → MV reversal high, CV smooth,
    #    headroom small, smooth_pass True (P100-class must not fail).
    m1 = compute_episode_metrics(_make_ep(hug, _square_mv()))
    print(f'[1 hug+chatter MV] mv_rev={m1["mv_reversal_rate"]:.3f} '
          f'cv_d2={m1["cv_d2_rms_normed"]:.5f} '
          f'cv_rev={m1["cv_reversal_rate"]:.3f} '
          f'headroom={m1["cv_opt_headroom"]:.4f} '
          f'side={m1["cv_preferred_side"]}')
    _check(m1['mv_reversal_rate'] > 0.5, 'chattery MV should reverse often', ok)
    _check(m1['cv_d2_rms_normed'] < 1e-6, 'constant CV must have ~0 d2', ok)
    _check(m1['cv_reversal_rate'] < 1e-9, 'constant CV must not reverse', ok)
    _check(m1['cv_opt_headroom'] < 0.02, 'hugging HI should have small headroom', ok)
    _check(m1['cv_preferred_side'] == 'hi', 'side_scale hi>lo → prefer HI', ok)
    g1 = control_quality_gates([{
        'seed': 1,
        'episode_metrics_agent': {
            **m1, 'economic_score': -5.0, 'mv_reversal_rate': 0.501,
        },
        'episode_metrics_baseline': {'economic_score': -90.0},
    }])
    _check(g1['smooth_pass'] is True,
           f'P100-class mv_rev=0.501 must not fail smooth_pass ({g1})', ok)
    _check(g1['beats_baseline_pass'] is True, 'agent -5 vs baseline -90', ok)

    # 2. Smooth slow ramp toward HI → TV > 0, d2 still tiny, pass.
    m2 = compute_episode_metrics(_make_ep(ramp, quiet_mv))
    print(f'[2 slow ramp] tv={m2["cv_tv_per_step_normed"]:.5f} '
          f'd2={m2["cv_d2_rms_normed"]:.5e} rev={m2["cv_reversal_rate"]:.3f}')
    _check(m2['cv_tv_per_step_normed'] > m1['cv_tv_per_step_normed'],
           'ramp TV should exceed a parked CV', ok)
    _check(m2['cv_d2_rms_normed'] <= CV_D2_RMS_GATE,
           'linear ramp must pass d2 gate (TV is not a gate)', ok)
    g2 = control_quality_gates([{
        'seed': 2,
        'episode_metrics_agent': {**m2, 'economic_score': -6.0},
        'episode_metrics_baseline': {'economic_score': -90.0},
    }])
    _check(g2['smooth_pass'] is True, 'slow ramp must pass cv_smooth', ok)

    # 3. CV bang-bang inside the band → fail cv_smooth_pass.
    m3 = compute_episode_metrics(_make_ep(bang_cv, quiet_mv))
    print(f'[3 CV bang-bang] d2={m3["cv_d2_rms_normed"]:.4f} '
          f'rev={m3["cv_reversal_rate"]:.3f}')
    _check(m3['cv_d2_rms_normed'] > CV_D2_RMS_GATE,
           'CV bang-bang d2 must exceed the gate', ok)
    _check(m3['cv_reversal_rate'] > CV_REVERSAL_GATE,
           'CV bang-bang reversal must exceed the gate', ok)
    g3 = control_quality_gates([{
        'seed': 3,
        'episode_metrics_agent': {**m3, 'economic_score': -8.0},
        'episode_metrics_baseline': {'economic_score': -90.0},
    }])
    _check(g3['smooth_pass'] is False, 'CV bang-bang must fail smooth_pass', ok)
    _check(g3['beats_baseline_pass'] is True,
           'smoothness fail must not hide a real econ win', ok)

    # 4. CV mid-band, no viol → high headroom (lost potential), smoothness pass.
    m4 = compute_episode_metrics(_make_ep(mid, quiet_mv))
    print(f'[4 mid-band] headroom={m4["cv_opt_headroom"]:.3f} '
          f'viol={m4["cv_viol_frac"]:.3f}')
    _check(m4['cv_opt_headroom'] > 0.4, 'mid-band is lost opt potential', ok)
    _check(m4['cv_viol_frac'] < 1e-12, 'mid-band must not violate', ok)
    g4 = control_quality_gates([{
        'seed': 4,
        'episode_metrics_agent': {**m4, 'economic_score': -40.0},
        'episode_metrics_baseline': {'economic_score': -90.0},
    }])
    _check(g4['smooth_pass'] is True, 'parked mid-band is smooth (residual is headroom)', ok)
    _check(g4.get('all_pass') is None,
           'control_quality_gates must not emit all_pass (caller ANDs WM floors)', ok)

    # 5. CV over HI bound → viol_frac > 0; constant so smoothness still passes.
    m5 = compute_episode_metrics(_make_ep(over, quiet_mv))
    print(f'[5 over HI] viol_frac={m5["cv_viol_frac"]:.3f} '
          f'depth={m5["cv_viol_depth_normed"]:.3f} '
          f'd2={m5["cv_d2_rms_normed"]:.5f}')
    _check(m5['cv_viol_frac'] > 0.99, 'constant over-bound should violate every step', ok)
    _check(m5['cv_d2_rms_normed'] < 1e-6, 'violation is not a smoothness fail', ok)
    g5 = control_quality_gates([{
        'seed': 5,
        'episode_metrics_agent': {**m5, 'economic_score': -12.0},
        'episode_metrics_baseline': {'economic_score': -90.0},
    }])
    _check(g5['smooth_pass'] is True, 'limit violation is cv_limit_pass, not smooth_pass', ok)
    _check(g5['cv_limit_pass'] is False, 'over-bound must fail informational cv_limit_pass', ok)

    # 6. Preferred side: side_scale hi>lo → HI; cv_economic_weights>0 → LO.
    side_hi, unk_hi = preferred_cv_side(_make_ep(hug, quiet_mv), 0)
    side_lo, unk_lo = preferred_cv_side(
        _make_ep(hug, quiet_mv, cv_econ=[2.0],
                 side_scale={'cv_0': {'lo': 0.4, 'hi': 1.0}}), 0)
    print(f'[6 preferred] side_scale={side_hi}/{unk_hi}  econ+={side_lo}/{unk_lo}')
    _check(side_hi == 'hi' and unk_hi is False, 'side_scale hi>lo → HI', ok)
    _check(side_lo == 'lo' and unk_lo is False, 'econ weight >0 overrides to LO', ok)
    hug_lo = np.full(T, LO + 0.05)
    m_lo = compute_episode_metrics(_make_ep(
        hug_lo, quiet_mv, cv_econ=[2.0]))
    m_hi_wrong = compute_episode_metrics(_make_ep(
        hug, quiet_mv, cv_econ=[2.0]))
    _check(m_lo['cv_opt_headroom'] < 0.02, 'prefer LO: sitting on LO is small headroom', ok)
    _check(m_hi_wrong['cv_opt_headroom'] > 0.9,
           'prefer LO: sitting on HI is lost potential', ok)

    # 7. TM curve IAE.
    t = np.linspace(0.0, 1.0, 40)
    real = 1.0 - np.exp(-5.0 * t)
    ident = _curve_shape_scores(real, real)
    scaled = _curve_shape_scores(2.0 * real, real)
    print(f'[7 TM curve] ident={ident["curve_iae_normed"]:.4e} '
          f'scaled={scaled["curve_iae_normed"]:.3f}')
    _check(ident['curve_iae_normed'] < 1e-12, 'identical curves IAE≈0', ok)
    _check(scaled['curve_iae_normed'] > 0.5, '2× scale mismatch must score IAE>0', ok)

    # Residual board must keep P64 lock and not treat family-closed as done.
    board = build_residual_board(
        fidelity_gates={'all_pass': True, 'beats_baseline_pass': True},
        mv_tf={'pairs': {'cv0<-mv0': {
            'ss_gain_ratio_wm_over_real': 0.927,
            'curve_iae_normed': 0.12,
        }}},
        dv_tf={'pairs': {'cv0<-dv0': {
            'ss_gain_ratio_wm_over_real': 0.893,
            'curve_iae_normed': 0.20,
        }}},
        postprior={'decomp_1step_to_openloop': 0.85},
        distpred={'mean_pearson_r_detrended': 0.352,
                  'per_channel': [{'pred_std': 0.608, 'true_std': 1.93}]},
        disturbance_records=[{
            'seed': 0,
            'episode_metrics_agent': {**m1, 'iae_normed_mean': 10.0,
                                      'economic_score': -4.54},
            'episode_metrics_baseline': {'iae_normed_mean': 80.0,
                                         'economic_score': -94.0},
            'event_response': {'overshoot_normed': {'p90': 0.2}},
        }],
    )
    _check(board['never_retire_because_family_closed'] is True,
           'board must refuse family-closed as residual-closed', ok)
    _check(board['mv_oscillation_allowed'] is True, 'MV osc allowed flag', ok)
    _check(board.get('refactors_allowed') is True, 'refactors_allowed on board', ok)
    _check(board.get('plan_while_live') is True, 'plan_while_live on board', ok)
    _check(board.get('metric_audit', {}).get('required_every_exit') is True,
           'metric audit required every EXIT', ok)
    _check('lock_p64' not in board, 'board must not freeze a past-run lock', ok)
    _check('docs/GOAL_PLAN.md' in (board.get('history_files') or []),
           'board must point at the living plan', ok)
    _check(board['r2_cv_quality']['smooth_pass_is_cv_only'] is True,
           'R2 smooth_pass is CV-only', ok)
    _check(board['r2_cv_quality']['cv_smooth_pass'] is True,
           'hug+chatter MV must be a CV-smooth win on the board', ok)

    nfail = len(ok)
    print('PASS' if nfail == 0 else f'FAILED ({nfail} checks)')
    return 0 if nfail == 0 else 1


if __name__ == '__main__':
    raise SystemExit(main())
