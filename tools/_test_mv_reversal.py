"""Focused test: adaptive MV reversal (oscillation) suppression.

Proves the design goal — a FAST controller is NOT penalised, only self-induced
oscillation is:
  * a fast full-range MONOTONIC move (du=+0.5, +0.5 normalized) costs EXACTLY
    zero reversal penalty  -> the agent can still move as fast as it needs.
  * a full-range BANG-BANG reversal (du=+0.5 then -0.5) costs a large penalty.
  * the auto-derived per-MV reversal weight is HARD-capped at cv_base so CV
    control + economics always strictly dominate ("not too strong").

Run:
  CUDA_VISIBLE_DEVICES="" PYTHONPATH=$PWD \
    ~/neural-APC-mbrl2-env/bin/python tools/_test_mv_reversal.py
"""
import os
from pathlib import Path

import numpy as np

SIM = Path('simulation/test_sim').resolve()
os.environ['CONTROL_SETUP_JSON'] = str(SIM / 'control_setup.json')
os.environ['CONTROL_OBJECTIVE_JSON'] = str(SIM / 'control_objective.json')
os.environ['SIMULATION_DIR'] = str(SIM)

from training.train import TrainConfig, APCEnv                 # noqa: E402
from utils.objective_runtime import compute_objective_components  # noqa: E402
from utils.auto_weights import derive_auto_weights             # noqa: E402


def _rev_penalty(env, pp, p, u):
    """Reversal penalty for MV history prev_prev=pp, prev=p, control=u (eng %)."""
    comps = compute_objective_components(
        state=np.array([82.0, 82.0, u, 100.0], dtype='float32'),
        sim=env.sim,
        control=np.array([u], dtype='float32'),
        prev_control=np.array([p], dtype='float32'),
        obj_w=env.obj_w, bounds=env.bounds,
        setpoint_manager=env.setpoint_mgr,
        objective_spec=env.obj_spec,
        prev_prev_control=np.array([pp], dtype='float32'),
    )
    return float(comps['mv_reversal_penalty']), comps['mv_reversal_terms']


def main() -> int:
    cfg = TrainConfig()
    cfg.episode_length = 200
    cfg.sample_rate = 1
    env = APCEnv(cfg, np.random.default_rng(0))
    env.reset()

    # MV range is [20, 80] (span 60); du of 30% eng = 0.5 normalized.
    ramp_pen, ramp_terms = _rev_penalty(env, 20.0, 50.0, 80.0)   # +0.5, +0.5 (fast, monotonic)
    bang_pen, bang_terms = _rev_penalty(env, 40.0, 70.0, 40.0)   # +0.5, -0.5 (reversal)
    small_pen, _ = _rev_penalty(env, 49.0, 50.0, 49.0)           # +0.017, -0.017 (tiny dither)

    print(f'[fast monotonic ramp] du=+0.5,+0.5  osc_term={ramp_terms[0]:.4f}  penalty={ramp_pen:.3f}')
    print(f'[full bang-bang     ] du=+0.5,-0.5  osc_term={bang_terms[0]:.4f}  penalty={bang_pen:.3f}')
    print(f'[tiny dither        ] du=+.017,-.017              penalty={small_pen:.3f}')

    auto = derive_auto_weights(
        {'cv_priority': ['cv_0'], 'weights': {'mv_economic': {'mv_0': 5.0}}},
        n_mv=1, n_cv=1,
        mv_bounds=[[20.0, 80.0]], cv_bounds=[[78.5, 85.5]],
        mv_norm_ranges=[[20.0, 80.0]], cv_norm_ranges=[[78.5, 85.5]])
    rev_w = float((auto.get('mv_reversal_weights') or [0.0])[0])
    cv_b = float((auto.get('cv_violation_weights') or [0.0])[0])
    print(f'[auto weights] mv_reversal_weight={rev_w:.1f}  cv_violation_weight={cv_b:.1f}  '
          f'(capped: {rev_w <= cv_b + 1e-6})')

    ok = True
    if not (abs(ramp_pen) < 1e-6):
        print(f'FAIL: fast monotonic ramp incurred a reversal penalty {ramp_pen:.4f} (must be 0)')
        ok = False
    if not (abs(ramp_terms[0]) < 1e-9):
        print(f'FAIL: monotonic osc term {ramp_terms[0]:.4f} != 0')
        ok = False
    if not (bang_pen > 50.0):
        print(f'FAIL: bang-bang penalty {bang_pen:.3f} too small to deter oscillation')
        ok = False
    if not (bang_pen > 100.0 * max(small_pen, 1e-9)):
        print(f'FAIL: bang-bang {bang_pen:.3f} not >> tiny-dither {small_pen:.3f} '
              '(should be quadratic in amplitude)')
        ok = False
    if not (rev_w > 0.0 and rev_w <= cv_b + 1e-6):
        print(f'FAIL: reversal weight {rev_w:.1f} not in (0, cv_base={cv_b:.1f}]')
        ok = False

    print('PASS' if ok else 'FAILED')
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
