"""CPU tests: plant-agnostic return-to-limit (R3).

Distances are / bound width.  Times are a fraction of the event window
(the window is 5τ in live val).  No engineering units.

IAE vs the pre-event baseline can look good while the CV sits mid-band.
cv_return_headroom / cv_return_time_frac are the scores for "aggressive
return to the economic limit."

Run:
  CUDA_VISIBLE_DEVICES="" PYTHONPATH=$PWD python tools/_test_return_to_limit.py
"""
from __future__ import annotations

import numpy as np

from evaluation.validate import (
    build_residual_board,
    compute_event_response_metrics,
    limit_gap_normed,
    return_to_limit_on_window,
)


LO, HI = 78.5, 85.5
WIDTH = HI - LO
T = 200
START = 40
WIN = 100
MID = 0.5 * (LO + HI)


def _make_ep(cv, *, start: int = START, cv_econ=None):
    cv = np.asarray(cv, dtype='float64').reshape(-1)
    n = int(cv.shape[0])
    return {
        'states': cv.reshape(n, 1).astype('float32'),
        'controls': np.full((n, 1), 50.0, dtype='float32'),
        'cv_indices': [0],
        'cv_bounds': [[LO, HI]],
        'mv_bounds': [[20.0, 80.0]],
        'episode_length': n,
        'sample_rate': 4,
        'schedule': [{'start': int(start), 'name': 'step', 'delta': 1.0}],
        'mean_cv_violation': 0.0,
        'mean_mv_violation': 0.0,
        'raw_rewards': np.zeros(n, dtype='float32'),
        'cum_raw_reward': 0.0,
        'obj_cv_economic_weights': (
            list(cv_econ) if cv_econ is not None else [-1.0]),
        'obj_cv_side_scale': {'cv_0': {'lo': 0.4, 'hi': 1.0}},
    }


def _check(cond: bool, msg: str, ok: list) -> None:
    if cond:
        return
    print(f'FAIL: {msg}')
    ok.append(False)


def _worst_ret(ev: dict) -> dict:
    return (ev.get('events') or [{}])[0].get('worst_return') or {}


def main() -> int:
    ok: list = []
    hug = np.full(T, HI - 0.05, dtype='float64')
    mid = np.full(T, MID, dtype='float64')

    # Kernel: HI-prefer, sitting on HI → gap ≈ 0.05/width.
    gap_hug = limit_gap_normed(hug[:10], LO, HI, 'hi')
    gap_mid = limit_gap_normed(mid[:10], LO, HI, 'hi')
    print(f'[0 kernel] hug_gap={gap_hug.mean():.4f} mid_gap={gap_mid.mean():.4f}')
    _check(float(gap_hug.mean()) < 0.02, 'hug HI → small gap/width', ok)
    _check(abs(float(gap_mid.mean()) - 0.5) < 0.02, 'mid-band → gap≈0.5', ok)
    lo_gap = limit_gap_normed(np.full(8, LO + 0.05), LO, HI, 'lo')
    _check(float(lo_gap.mean()) < 0.02, 'prefer LO sitting on LO', ok)

    # Scale invariance: same shape on 10× width → same time_frac + headroom.
    y_small = np.linspace(0.0, 1.0, 80)  # 0=lo, 1=hi
    a = return_to_limit_on_window(y_small, 0.0, 1.0, 'hi')
    b = return_to_limit_on_window(10.0 * y_small, 0.0, 10.0, 'hi')
    print(f'[0 scale] small t={a["cv_return_time_frac"]:.3f} '
          f'h={a["cv_return_headroom"]:.3f}  '
          f'wide t={b["cv_return_time_frac"]:.3f} '
          f'h={b["cv_return_headroom"]:.3f}')
    _check(abs(a['cv_return_time_frac'] - b['cv_return_time_frac']) < 1e-9,
           'time_frac must be width-invariant', ok)
    _check(abs(a['cv_return_headroom'] - b['cv_return_headroom']) < 1e-9,
           'headroom must be width-invariant', ok)

    # 1. Already on HI: IAE-to-pre-event is ~0 AND return is good.
    ev1 = compute_event_response_metrics(_make_ep(hug), window_steps=WIN)
    r1 = _worst_ret(ev1)
    print(f'[1 already-on-limit] head={r1.get("cv_return_headroom")} '
          f'time={r1.get("cv_return_time_frac")} '
          f'reached={r1.get("cv_return_reached")}')
    _check(r1.get('cv_return_reached') is True, 'already on HI must reach', ok)
    _check(float(r1['cv_return_headroom']) < 0.02, 'late headroom tiny', ok)
    _check(float(r1['cv_return_time_frac']) < 0.05, 'immediate return', ok)

    # 2. Pre-event MID, recover to MID: IAE-to-pre-event looks great,
    #    return-to-limit does not (the gamed case).
    ev2 = compute_event_response_metrics(_make_ep(mid), window_steps=WIN)
    r2 = _worst_ret(ev2)
    iae2 = (ev2.get('events') or [{}])[0].get('worst_cv', {}).get(
        'iae_window_normed', 1.0)
    print(f'[2 mid-band recover] head={r2.get("cv_return_headroom"):.3f} '
          f'time={r2.get("cv_return_time_frac"):.3f} iae_pre={iae2:.4f}')
    _check(float(iae2) < 1e-6, 'flat mid-band IAE vs pre-event ≈ 0', ok)
    _check(float(r2['cv_return_headroom']) > 0.4,
           'mid-band park is lost opt potential', ok)
    _check(r2.get('cv_return_reached') is False, 'never enters limit band', ok)
    _check(float(r2['cv_return_time_frac']) >= 0.999, 'time_frac=1 if never', ok)

    # 3. Knocked off HI, back to HI quickly.
    y3 = hug.copy()
    y3[START:START + 8] = MID
    ev3 = compute_event_response_metrics(_make_ep(y3), window_steps=WIN)
    r3 = _worst_ret(ev3)
    print(f'[3 fast return] head={r3.get("cv_return_headroom"):.4f} '
          f'time={r3.get("cv_return_time_frac"):.3f} '
          f'reached={r3.get("cv_return_reached")}')
    _check(r3.get('cv_return_reached') is True, 'fast return must reach', ok)
    _check(float(r3['cv_return_headroom']) < 0.02, 'late window back on HI', ok)
    _check(float(r3['cv_return_time_frac']) < 0.25, 'aggressive time_frac', ok)

    # 4. Slow crawl: drop to MID at event, linear back to HI over the window.
    y4 = hug.copy()
    y4[START:START + WIN] = np.linspace(MID, HI - 0.05, WIN)
    ev4 = compute_event_response_metrics(_make_ep(y4), window_steps=WIN)
    r4 = _worst_ret(ev4)
    print(f'[4 slow crawl] head={r4.get("cv_return_headroom"):.3f} '
          f'time={r4.get("cv_return_time_frac"):.3f}')
    _check(float(r4['cv_return_time_frac']) > float(r3['cv_return_time_frac']),
           'slow crawl must be slower than the fast return', ok)
    _check(float(r4['cv_return_time_frac']) > 0.5,
           'arriving only at the end is not aggressive', ok)

    # 5. Residual board surfaces the keys (not an all_pass gate).
    board = build_residual_board(
        disturbance_records=[{
            'seed': 0,
            'episode_metrics_agent': {
                'iae_normed_mean': 10.0, 'economic_score': -8.0,
                'cv_d2_rms_normed': 0.0, 'cv_reversal_rate': 0.0,
            },
            'episode_metrics_baseline': {
                'iae_normed_mean': 80.0, 'economic_score': -90.0,
            },
            'event_response': ev3,
            'event_response_baseline': ev2,
        }],
    )
    r3b = board['r3_unmeasured_dr']
    print(f'[5 board] head={r3b.get("cv_return_headroom_mean")} '
          f'time={r3b.get("cv_return_time_frac_mean")} '
          f'vs_base={r3b.get("cv_return_headroom_vs_baseline_mean")}')
    _check(r3b.get('not_all_pass') is True, 'return-to-limit is not all_pass', ok)
    _check(r3b.get('cv_return_headroom_mean') is not None
           and np.isfinite(float(r3b['cv_return_headroom_mean'])),
           'board must carry return headroom', ok)
    _check(float(r3b['cv_return_headroom_mean'])
           < float(r3b['cv_return_headroom_vs_baseline_mean'] or 99)
           or float(r3b['cv_return_headroom_vs_baseline_mean']) < 0.5,
           'fast-return agent vs mid-band baseline should beat on ratio', ok)
    audit = board.get('metric_audit') or {}
    _check(any('cv_return_headroom' in s for s in (audit.get('known_insufficient') or [])),
           'metric audit must name IAE-without-return', ok)
    _check(board['quality_targets_not_all_pass'].get('cv_return_headroom') == 0.15,
           'headroom target on the board', ok)

    nfail = len(ok)
    print('PASS' if nfail == 0 else f'FAILED ({nfail} checks)')
    return 0 if nfail == 0 else 1


if __name__ == '__main__':
    raise SystemExit(main())
