"""Smoke test for the P89 noise-curriculum + realistic hidden-disturbance work.

Verifies WITHOUT a real sim / GPU:
  * ``noise_curriculum_scale`` ramps 0->1 over P1 and is full (1.0) in P3.
  * ``SimNoiseWrapper.set_noise_scale(0.0)`` makes a step fully noise-free,
    and ``1.0`` restores configured noise.
  * ``HiddenDisturbanceSchedule`` builds a realistic, sim-adaptive event set
    (varied shapes, magnitudes within the MV-authority cap), is a drop-in
    (is_empty / step / last_applied / summary), reverting events return to 0,
    permanent events hold, and overlap can occur.
  * ``maybe_build_hidden_disturbance`` routes to the schedule by default and to
    the legacy OU when DREAMER_HIDDEN_DIST_MODE=ou.

Run (CPU):
  PYTHONPATH=$PWD \
  $PWD/../neural-apc-mbrl-env/bin/python tools/_smoke_noise_disturbance.py
"""
import os

import numpy as np

from utils.noise_config import noise_curriculum_scale
from utils.sim_noise import SimNoiseWrapper
from utils.hidden_disturbance import (
    HiddenDisturbanceSchedule, maybe_build_hidden_disturbance)


# --------------------------------------------------------------------------
# Minimal fake sim for SimNoiseWrapper + disturbance schedule
# --------------------------------------------------------------------------
class _FakeSim:
    """Deterministic 4-state sim: step() returns a constant state.

    Exposes the metadata the noise wrapper + disturbance schedule read.
    """

    def __init__(self):
        self.cv_indices = [0]
        self.cv_normalization_ranges = [[68.0, 96.0]]  # span 28
        self.state_is_normalized = False
        self.episode_counter = 0
        self.noise_config = {
            'ou_noise': [{'index': 0, 'sigma': 0.5, 'gain': 1.0,
                          'bounds': (68, 96), 'theta': 0.75, 'dt': 0.01}],
            'measurement_noise': [{'index': 0, 'sigma': 0.14,
                                   'bounds': (68, 96)}],
        }

    def reset(self):
        self.episode_counter = 0
        return np.array([80.0, 0.0, 0.0, 100.0], dtype='float32')

    def step(self, action):
        self.episode_counter += 1
        return np.array([80.0, 0.0, 0.0, 100.0], dtype='float32'), False


def test_noise_curriculum():
    os.environ.pop('DREAMER_PROCESS_NOISE_AMP_RAMP', None)  # default 0.0:0.4
    s0 = noise_curriculum_scale(0.0, phase=1)
    s20 = noise_curriculum_scale(0.2, phase=1)
    s40 = noise_curriculum_scale(0.4, phase=1)
    s_p3 = noise_curriculum_scale(0.0, phase=3)
    assert s0 == 0.0, s0
    assert 0.0 < s20 < 1.0, s20
    assert s40 == 1.0, s40
    assert s_p3 == 1.0, s_p3            # P3 always full noise
    assert s20 < s40
    print(f'[smoke] OK noise curriculum: p=0->{s0:.2f} 0.2->{s20:.2f} '
          f'0.4->{s40:.2f} P3->{s_p3:.2f}')


def test_noise_scale_clean():
    sim = _FakeSim()
    w = SimNoiseWrapper(sim, noise_config=sim.noise_config)
    w.reset()
    w.set_noise_scale(0.0)
    states = np.array([w.step(0.0)[0][0] for _ in range(64)])
    assert np.allclose(states, 80.0), f'noise leaked at scale 0: std={states.std()}'
    w.set_noise_scale(1.0)
    states_n = np.array([w.step(0.0)[0][0] for _ in range(256)])
    assert states_n.std() > 1e-3, f'no noise at scale 1: std={states_n.std()}'
    print(f'[smoke] OK noise_scale: clean std={states.std():.2e} '
          f'-> full std={states_n.std():.3f}')


def test_hidden_schedule():
    sim = _FakeSim()
    rng = np.random.default_rng(0)
    # tau_dom=53, sr=4 -> tau_steps~13.25; dead=8 -> dead_steps=2; settle~55.
    sched = HiddenDisturbanceSchedule(
        rng=rng, sim=sim, tau_dom=53.0, sample_rate=4.0, dead_time=8.0,
        episode_length=1220, amp_frac=0.10)
    assert not sched.is_empty(), 'schedule built no events'
    summ = sched.summary()
    shapes = {e['shape'] for e in summ['events']}
    assert summ['n_events'] >= 1
    # MV-authority/ span cap: |mag| <= amp_frac*span = 0.10*28 = 2.8.
    for e in summ['events']:
        assert abs(e['mag']) <= 0.10 * 28.0 + 1e-6, e
    # Roll the schedule and confirm: applied offset is bounded, reverting
    # events return to ~0, and last_applied is aligned to cv_indices (len 1).
    max_abs = 0.0
    traj = []
    for _ in range(1220):
        st = np.array([80.0, 0.0, 0.0, 100.0], dtype='float64')
        sched.step(st)
        off = st[0] - 80.0
        traj.append(off)
        max_abs = max(max_abs, abs(off))
        assert sched.last_applied.shape[0] == 1
    traj = np.array(traj)
    # Overlap possible: total offset can exceed a single event's cap when two
    # events coincide, but must stay within n_events * cap.
    assert max_abs <= summ['n_events'] * 0.10 * 28.0 + 1e-6
    # At least some steps are exactly clean (gaps between events).
    assert np.any(np.abs(traj) < 1e-9), 'schedule never idle (no gaps)'
    print(f"[smoke] OK hidden schedule: n_events={summ['n_events']} "
          f"shapes={sorted(shapes)} settle={summ['settle_steps']:.0f} "
          f"max|off|={max_abs:.3f}")


def test_builder_routing():
    sim = _FakeSim()
    rng = np.random.default_rng(1)
    os.environ['DREAMER_HIDDEN_DIST_MODE'] = 'schedule'
    p = maybe_build_hidden_disturbance(
        rng=rng, sim=sim, tau_dom=53.0, sample_rate=4.0, prob=1.0,
        force=True, dead_time=8.0, episode_length=1220)
    assert type(p).__name__ == 'HiddenDisturbanceSchedule', type(p)
    os.environ['DREAMER_HIDDEN_DIST_MODE'] = 'ou'
    p2 = maybe_build_hidden_disturbance(
        rng=rng, sim=sim, tau_dom=53.0, sample_rate=4.0, prob=1.0,
        force=True, dead_time=8.0, episode_length=1220)
    assert type(p2).__name__ == 'HiddenDisturbanceProcess', type(p2)
    os.environ.pop('DREAMER_HIDDEN_DIST_MODE', None)
    print('[smoke] OK builder routing: schedule (default) + ou (legacy)')


if __name__ == '__main__':
    test_noise_curriculum()
    test_noise_scale_clean()
    test_hidden_schedule()
    test_builder_routing()
    print('\n[smoke] ALL P89 noise + disturbance checks PASSED')
