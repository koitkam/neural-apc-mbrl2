"""Smoke test for the P89 noise-curriculum + realistic hidden-disturbance work.

Verifies WITHOUT a real sim / GPU:
  * ``noise_curriculum_scale`` ramps 0->1 over P1 and is full (1.0) in P3.
  * ``SimNoiseWrapper.set_noise_scale(0.0)`` makes a step fully noise-free,
    and ``1.0`` restores configured noise.
  * ``HiddenDisturbance`` builds a realistic, sim-adaptive LOAD-event set
    (varied shapes, magnitudes within the MV-authority cap), is a drop-in
    (is_empty / step / last_applied / summary), and the CV effect ``d_cv`` is
    SMOOTH + DELAYED (filtered through Gd = dead-time + first-order lag) — i.e.
    NO instantaneous step on the CV even for a ``step`` LOAD event.
  * ``maybe_build_hidden_disturbance`` builds the single consolidated model.

Run (CPU):
  PYTHONPATH=$PWD \
  $PWD/../neural-apc-mbrl-env/bin/python tools/_smoke_noise_disturbance.py
"""
import os

import numpy as np

from utils.noise_config import noise_curriculum_scale
from utils.sim_noise import SimNoiseWrapper
from utils.hidden_disturbance import (
    HiddenDisturbance, maybe_build_hidden_disturbance)


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


def test_hidden_load_through_gd():
    sim = _FakeSim()
    rng = np.random.default_rng(0)
    # tau_dom=53, sr=4 -> tau_steps~13.25; dead=8 -> dead_steps=2; settle~55.
    dist = HiddenDisturbance(
        rng=rng, sim=sim, tau_dom=53.0, sample_rate=4.0, dead_time=8.0,
        episode_length=1220, amp_frac=0.10, drift_frac=0.0)
    assert not dist.is_empty(), 'built no load events'
    summ = dist.summary()
    assert summ['mode'] == 'load_through_Gd', summ['mode']
    shapes = {e['shape'] for e in summ['events']}
    assert 'ou_drift' not in shapes, 'ou_drift must be gone'
    assert shapes <= {'step', 'ramp', 'pulse'}, shapes
    assert summ['n_events'] >= 1
    # Gd params present + sane (theta_d>=1 step, tau_d>=1 step).
    assert summ['theta_d_steps'][0] >= 1
    assert summ['tau_d_steps'][0] >= 1.0
    # LOAD-event magnitude cap: |mag| <= amp_frac*span = 0.10*28 = 2.8.
    for e in summ['events']:
        assert abs(e['mag']) <= 0.10 * 28.0 + 1e-6, e
    # Roll and confirm: (a) last_applied aligned to cv_indices (len 1),
    # (b) the CV effect is SMOOTH (no instantaneous jump — bounded per-step
    # delta, because Gd's first-order lag rate is alpha=1/tau_d_steps), and
    # (c) a clean warm-up (d_cv starts at ~0 with drift_frac=0).
    tau_d = float(summ['tau_d_steps'][0])
    cap = 0.10 * 28.0
    alpha = 1.0 / tau_d
    # max single-step |Δd_cv| = alpha * |u_delayed - y| <= alpha * (peak load).
    # Peak load <= n_events*cap (overlap); bound the per-step jump generously.
    max_step_jump = alpha * (summ['n_events'] * cap) + 1e-6
    traj = []
    prev = 0.0
    max_abs = 0.0
    max_jump = 0.0
    for _ in range(1220):
        st = np.array([80.0, 0.0, 0.0, 100.0], dtype='float64')
        dist.step(st)
        off = st[0] - 80.0
        traj.append(off)
        max_abs = max(max_abs, abs(off))
        max_jump = max(max_jump, abs(off - prev))
        prev = off
        assert dist.last_applied.shape[0] == 1
    traj = np.array(traj)
    assert max_abs <= summ['n_events'] * cap + 1e-6, max_abs
    # THE key realism check: the CV effect never jumps like a raw step would.
    # A raw step LOAD would put |Δ|=cap (~2.8) in one step; Gd keeps it small.
    assert max_jump <= max_step_jump, (
        f'CV effect jumped {max_jump:.3f} > Gd bound {max_step_jump:.3f} '
        f'(disturbance not properly lagged!)')
    assert abs(traj[0]) < 1e-9, f'd_cv should start at ~0, got {traj[0]}'
    assert np.any(np.abs(traj) < 1e-9), 'never idle (no gaps)'
    print(f"[smoke] OK hidden load->Gd: n_events={summ['n_events']} "
          f"shapes={sorted(shapes)} tau_d={tau_d:.1f} theta_d={summ['theta_d_steps'][0]} "
          f"max|d_cv|={max_abs:.3f} max|Δd_cv|={max_jump:.4f} (<= {max_step_jump:.4f})")


def test_gd_smooths_a_pure_step():
    """A pure step LOAD must produce a SMOOTH first-order CV response (the core
    realism fix): no instantaneous CV jump, monotonic rise toward the cap."""
    sim = _FakeSim()
    rng = np.random.default_rng(7)
    os.environ['DREAMER_HIDDEN_DIST_SHAPE_WEIGHTS'] = '1,0,0'   # all step loads
    os.environ['DREAMER_HIDDEN_DIST_MAX_EVENTS'] = '1'
    try:
        dist = HiddenDisturbance(
            rng=rng, sim=sim, tau_dom=53.0, sample_rate=4.0, dead_time=8.0,
            episode_length=1220, amp_frac=0.10, drift_frac=0.0)
    finally:
        os.environ.pop('DREAMER_HIDDEN_DIST_SHAPE_WEIGHTS', None)
        os.environ.pop('DREAMER_HIDDEN_DIST_MAX_EVENTS', None)
    summ = dist.summary()
    assert summ['events'] and summ['events'][0]['shape'] == 'step'
    tau_d = float(summ['tau_d_steps'][0])
    traj = []
    for _ in range(1220):
        st = np.array([80.0, 0.0, 0.0, 100.0], dtype='float64')
        dist.step(st)
        traj.append(st[0] - 80.0)
    traj = np.array(traj)
    # The biggest single-step change must be much smaller than the step's
    # magnitude (a raw step would be the full magnitude in one sample).
    mag = abs(summ['events'][0]['mag'])
    biggest_jump = float(np.max(np.abs(np.diff(traj))))
    assert biggest_jump < 0.5 * mag, (
        f'step LOAD produced a {biggest_jump:.3f} CV jump vs mag {mag:.3f} '
        f'— Gd did not smooth it')
    print(f"[smoke] OK Gd smooths a pure step: mag={mag:.3f} "
          f"max|Δd_cv|={biggest_jump:.4f} tau_d={tau_d:.1f} (smooth first-order)")


def test_builder_single_model():
    sim = _FakeSim()
    rng = np.random.default_rng(1)
    p = maybe_build_hidden_disturbance(
        rng=rng, sim=sim, tau_dom=53.0, sample_rate=4.0, prob=1.0,
        force=True, dead_time=8.0, episode_length=1220)
    assert type(p).__name__ == 'HiddenDisturbance', type(p)
    assert not p.is_empty()
    # No episode_length -> no events schedulable -> None (clean).
    p0 = maybe_build_hidden_disturbance(
        rng=rng, sim=sim, tau_dom=53.0, sample_rate=4.0, prob=1.0,
        force=True, dead_time=8.0, episode_length=0)
    assert p0 is None, 'episode_length=0 should yield no disturbance'
    print('[smoke] OK builder: single consolidated HiddenDisturbance model')


if __name__ == '__main__':
    test_noise_curriculum()
    test_noise_scale_clean()
    test_hidden_load_through_gd()
    test_gd_smooths_a_pure_step()
    test_builder_single_model()
    print('\n[smoke] ALL noise + (Gd-filtered) disturbance checks PASSED')
