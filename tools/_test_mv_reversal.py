"""P125: MV reversal penalty is OFF; CV sticky-tanh hunting is ON.

Proves GOAL_PLAN #4:
  * a full-range MV bang-bang costs EXACTLY zero reversal penalty
    (MV chatter is allowed).
  * a monotonic CV ride (one sign) costs zero.
  * a P116-class opposite move after a sticky sign costs a large penalty.
  * a hold then opposite move still counts (sticky does not clear on hold).
  * auto-derived ``mv_reversal_weights`` are zero; ``cv_reversal_weights``
    are HARD-capped at cv_base.

Run:
  CUDA_VISIBLE_DEVICES="" PYTHONPATH=$PWD \\
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
from utils.objective_runtime import (                          # noqa: E402
    compute_objective_components, _CV_REV_STICKY_THRESH)
from utils.auto_weights import derive_auto_weights             # noqa: E402


def _comps(env, state_cv, u, prev_u, prev_cv, sticky, prev_prev_u=None):
    st = np.array([state_cv, state_cv, u, 100.0], dtype='float32')
    return compute_objective_components(
        state=st,
        sim=env.sim,
        control=np.array([u], dtype='float32'),
        prev_control=np.array([prev_u], dtype='float32'),
        obj_w=env.obj_w, bounds=env.bounds,
        setpoint_manager=env.setpoint_mgr,
        objective_spec=env.obj_spec,
        prev_prev_control=np.array(
            [prev_u if prev_prev_u is None else prev_prev_u], dtype='float32'),
        prev_cv=np.array([prev_cv], dtype='float32'),
        prev_cv_reversal_sticky=np.array([sticky], dtype='float32'),
    )


def main() -> int:
    cfg = TrainConfig()
    cfg.episode_length = 200
    cfg.sample_rate = 1
    env = APCEnv(cfg, np.random.default_rng(0))
    env.reset()

    # MV bang-bang must not pay (weights zero). +0.5 then -0.5 normalised.
    bang = _comps(env, 82.0, 40.0, 70.0, 82.0, 0.0, prev_prev_u=40.0)
    print(f'[MV bang-bang] mv_pen={bang["mv_reversal_penalty"]:.6f} '
          f'cv_pen={bang["cv_reversal_penalty"]:.6f} '
          f'mv_term={bang["mv_reversal_terms"][0]:.4f}')

    # First CV move +0.03 (≫ deadband 0.007): sets sticky, penalty 0.
    first = _comps(env, 82.03, 50.0, 50.0, 82.00, 0.0)
    sticky1 = float(first['cv_reversal_sticky'][0])
    print(f'[CV first +d] cv_pen={first["cv_reversal_penalty"]:.6f} '
          f'term={first["cv_reversal_terms"][0]:.4f} sticky={sticky1:.3f} '
          f'thresh={_CV_REV_STICKY_THRESH:.3f}')

    # Opposite move: must pay.
    opp = _comps(env, 82.00, 50.0, 50.0, 82.03, sticky1)
    print(f'[CV opposite] cv_pen={opp["cv_reversal_penalty"]:.6f} '
          f'term={opp["cv_reversal_terms"][0]:.4f}')

    # Monotonic second +d: free.
    ride = _comps(env, 82.06, 50.0, 50.0, 82.03, sticky1)
    print(f'[CV monotonic] cv_pen={ride["cv_reversal_penalty"]:.6f} '
          f'term={ride["cv_reversal_terms"][0]:.4f}')

    # Hold then opposite: hold keeps sticky, reverse pays.
    hold = _comps(env, 82.03, 50.0, 50.0, 82.03, sticky1)
    sticky_hold = float(hold['cv_reversal_sticky'][0])
    after_hold = _comps(env, 82.00, 50.0, 50.0, 82.03, sticky_hold)
    print(f'[CV hold] pen={hold["cv_reversal_penalty"]:.6f} sticky={sticky_hold:.3f}')
    print(f'[CV after hold] pen={after_hold["cv_reversal_penalty"]:.6f} '
          f'term={after_hold["cv_reversal_terms"][0]:.4f}')

    auto = derive_auto_weights(
        {'cv_priority': ['cv_0'], 'weights': {'mv_economic': {'mv_0': 5.0}}},
        n_mv=1, n_cv=1,
        mv_bounds=[[20.0, 80.0]], cv_bounds=[[78.5, 85.5]],
        mv_norm_ranges=[[20.0, 80.0]], cv_norm_ranges=[[78.5, 85.5]],
        cfg=cfg)
    mv_w = float((auto.get('mv_reversal_weights') or [0.0])[0])
    cv_w = float((auto.get('cv_reversal_weights') or [0.0])[0])
    cv_b = float((auto.get('cv_violation_weights') or [0.0])[0])
    print(f'[auto weights] mv_reversal={mv_w:.1f} cv_reversal={cv_w:.1f} '
          f'cv_base={cv_b:.1f} (cv capped: {cv_w <= cv_b + 1e-6})')

    ok = True
    if abs(bang['mv_reversal_penalty']) > 1e-6:
        print(f'FAIL: MV bang-bang paid {bang["mv_reversal_penalty"]:.4f}')
        ok = False
    if abs(first['cv_reversal_penalty']) > 1e-6:
        print(f'FAIL: first CV move paid {first["cv_reversal_penalty"]:.4f}')
        ok = False
    if not (sticky1 >= _CV_REV_STICKY_THRESH):
        print(f'FAIL: first move did not set sticky {sticky1}')
        ok = False
    if not (opp['cv_reversal_penalty'] > 50.0):
        print(f'FAIL: opposite-move penalty {opp["cv_reversal_penalty"]:.3f} too small')
        ok = False
    if abs(ride['cv_reversal_penalty']) > 1e-6:
        print(f'FAIL: monotonic ride paid {ride["cv_reversal_penalty"]:.4f}')
        ok = False
    if abs(hold['cv_reversal_penalty']) > 1e-6:
        print(f'FAIL: hold paid {hold["cv_reversal_penalty"]:.4f}')
        ok = False
    if abs(sticky_hold - sticky1) > 1e-6:
        print(f'FAIL: hold cleared sticky {sticky_hold} vs {sticky1}')
        ok = False
    if not (after_hold['cv_reversal_penalty'] > 50.0):
        print(f'FAIL: after-hold reverse {after_hold["cv_reversal_penalty"]:.3f} too small')
        ok = False
    if abs(mv_w) > 1e-9:
        print(f'FAIL: mv_reversal_weights {mv_w} not zero')
        ok = False
    if not (cv_w > 0.0 and cv_w <= cv_b + 1e-6):
        print(f'FAIL: cv_reversal_weight {cv_w:.1f} not in (0, cv_base={cv_b:.1f}]')
        ok = False
    if abs(cv_w - cv_b) > 1e-6:
        print(f'FAIL: P126 default gain 1.0 should sit at cap '
              f'cv_reversal={cv_w:.1f} vs cv_base={cv_b:.1f}')
        ok = False

    # #8 split: opposite-move econ less negative than hunt; mono/hold equal.
    if not np.isfinite(opp['reward_econ']):
        print('FAIL: opposite reward_econ non-finite')
        ok = False
    if not (opp['reward_econ'] > opp['reward'] + 1.0):
        print(f'FAIL: opposite econ {opp["reward_econ"]:.3f} not milder '
              f'than hunt {opp["reward"]:.3f}')
        ok = False
    if abs(ride['reward'] - ride['reward_econ']) > 1e-6:
        print(f'FAIL: monotonic streams differ {ride["reward"]:.4f} vs '
              f'{ride["reward_econ"]:.4f}')
        ok = False
    if abs(hold['reward'] - hold['reward_econ']) > 1e-6:
        print(f'FAIL: hold streams differ {hold["reward"]:.4f} vs '
              f'{hold["reward_econ"]:.4f}')
        ok = False
    post_sat_add = float(opp['reward']) + float(opp['cv_reversal_penalty'])
    if abs(float(opp['reward_econ']) - post_sat_add) < 1e-3:
        print(f'FAIL: reward_econ is post-sat add-back {post_sat_add:.3f}')
        ok = False

    # env.step: raw_reward reversal-free; raw_hunt = hunting sat; trace copy.
    env._shaping_enabled = False
    env.reset()
    _, _r0, _, info0 = env.step(np.zeros((env.action_dim,), dtype='float32'))
    c0 = info0['reward_components']
    if 'raw_hunt' not in info0:
        print('FAIL: info missing raw_hunt')
        ok = False
    if 'reward_econ_train' not in info0:
        print('FAIL: info missing reward_econ_train')
        ok = False
    if abs(float(info0['raw_reward']) - float(c0['reward_econ'])) > 1e-3:
        print(f'FAIL: raw_reward {info0["raw_reward"]:.4f} != '
              f'reward_econ {c0["reward_econ"]:.4f}')
        ok = False
    if abs(float(info0['raw_hunt']) - float(c0['reward'])) > 1e-3:
        print(f'FAIL: raw_hunt {info0["raw_hunt"]:.4f} != '
              f'reward {c0["reward"]:.4f}')
        ok = False
    tlen = 1
    e1 = env.pop_episode_rew_econ(tlen)
    e2 = env.pop_episode_rew_econ(tlen)
    if e1 is None or e2 is None or not np.allclose(e1, e2):
        print('FAIL: pop_episode_rew_econ did not copy-without-clear')
        ok = False
    env.reset()
    if env.pop_episode_rew_econ(1) is not None:
        print('FAIL: reset() did not clear rew_econ trace')
        ok = False

    print('PASS' if ok else 'FAILED')
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
