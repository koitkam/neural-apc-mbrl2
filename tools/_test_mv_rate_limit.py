"""Focused test: the DCS output-velocity (rate) limit in APCEnv.step.

Builds a real test_sim APCEnv and drives it with a full-range ALTERNATING
+-1 command (the degenerate bang-bang the actor learns).  Asserts:
  * with the rate limit ON  (auto): the APPLIED normalized command slews by
    at most `rate` per step  ->  bang-bang is physically impossible.
  * with the rate limit OFF (-1) : the applied command swings the full +-1
    every step  ->  proves the limit is what suppresses the swing.

Run:
  CUDA_VISIBLE_DEVICES="" PYTHONPATH=$PWD \
    ~/neural-APC-mbrl2-env/bin/python tools/_test_mv_rate_limit.py
"""
import os
from pathlib import Path

import numpy as np

SIM = Path('simulation/test_sim').resolve()
os.environ['CONTROL_SETUP_JSON'] = str(SIM / 'control_setup.json')
os.environ['CONTROL_OBJECTIVE_JSON'] = str(SIM / 'control_objective.json')
os.environ['SIMULATION_DIR'] = str(SIM)

from training.train import TrainConfig, APCEnv  # noqa: E402


def _drive(rate_cfg: float):
    cfg = TrainConfig()
    cfg.episode_length = 200
    cfg.sample_rate = 1
    cfg.mv_rate_limit = rate_cfg
    env = APCEnv(cfg, np.random.default_rng(0))
    env.reset()
    rate = env._mv_rate_limit()
    prev = None
    max_slew = 0.0
    for t in range(30):
        a = np.ones(env.action_dim, dtype='float32') * (1.0 if t % 2 == 0 else -1.0)
        env.step(a)
        cmd = env._prev_cmd_norm.copy()
        if prev is not None:
            max_slew = max(max_slew, float(np.max(np.abs(cmd - prev))))
        prev = cmd
    return rate, max_slew


def main() -> int:
    rate_on, slew_on = _drive(0.0)          # auto
    rate_off, slew_off = _drive(-1.0)       # OFF

    print(f'[rate-limit ON ] auto rate={rate_on:.3f}  max applied slew/step={slew_on:.3f}')
    print(f'[rate-limit OFF] rate={rate_off:.3f}  max applied slew/step={slew_off:.3f}')

    ok = True
    # ON: applied slew must never exceed the rate (+ tiny fp eps).
    if not (slew_on <= rate_on + 1e-5):
        print(f'FAIL: ON slew {slew_on:.3f} exceeds rate {rate_on:.3f}')
        ok = False
    # ON: rate must be a sane fraction of the [-1,1] range (auto clip [0.05,0.4]).
    if not (0.05 - 1e-9 <= rate_on <= 0.4 + 1e-9):
        print(f'FAIL: auto rate {rate_on:.3f} outside [0.05, 0.4]')
        ok = False
    # OFF: full alternating swing -> applied slew ~ 2.0 (from +1 to -1).
    if not (slew_off > 1.5):
        print(f'FAIL: OFF slew {slew_off:.3f} not full-range (rate limit leaking)')
        ok = False
    # ON must be dramatically smoother than OFF.
    if not (slew_on < 0.5 * slew_off):
        print(f'FAIL: ON slew {slew_on:.3f} not much smaller than OFF {slew_off:.3f}')
        ok = False

    print('PASS' if ok else 'FAILED')
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
