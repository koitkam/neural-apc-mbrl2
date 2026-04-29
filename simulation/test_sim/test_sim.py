"""Fast SISO distillation-like simulator for workflow validation.

Functionality:
- Simulates one controlled temperature CV with one manipulated reflux MV
  and one measured feed-rate DV.
- Pure deterministic physics: actuator lag, MV dead time, first-order
  process lag, DV mean-reversion.
- All stochastic elements (process noise, measurement noise, noise amplitude
  jitter) are injected externally by ``SimNoiseWrapper`` using a dynamics-
  derived config from ``control_noise_config``.  Domain randomization of
  internal parameters (taus, gains, dead times) is controlled by the
  ``DomainRandomizer`` whose range can be overridden externally.

Inputs:
- Constructor: episode_length, sample_rate.
- Optional env vars: SIM_DOMAIN_RANDOMIZATION, SIM_PARAM_RANDOMIZATION_PCT,
    SIM_DOMAIN_RANDOMIZATION_SEED (and legacy DISTILLATION_* aliases).

Outputs:
- reset() -> (state, done)
- step(action) -> (state, done)

Normalization ranges:
- ``mv_normalization_ranges`` / ``cv_normalization_ranges`` /
  ``dv_normalization_ranges`` are *per-group* (one entry per MV/CV/DV).
- ``state_normalization_ranges`` is *per state-vector slot* (one entry per
  element in the state returned by ``reset()`` / ``step()``); some slots
  duplicate per-group entries because they describe the same channel from
  a different view.
- When changing any range, update BOTH the per-group list AND the matching
  slot(s) in ``state_normalization_ranges``.  These two views are kept in
  sync by hand because the state-vector layout differs between sims.
"""

import datetime
import os

import numpy as np
import pandas as pd

from utils.sim_noise import DisturbanceOffsetMixin, DomainRandomizer


class TestSimTower(DisturbanceOffsetMixin):
    # CV physics consumes `self._cv_offsets` in `step()` (see
    # `_temp_target(...) + self._cv_offsets.get(...)`), so unmeasured-CV
    # disturbances produce a sustained, dynamics-respecting response
    # without needing the post-step hold in the disturbance layer.
    honors_cv_disturbance_offsets: bool = True

    """Single-loop temperature control process for fast training checks."""

    def __init__(
        self,
        episode_length,
        sample_rate=1,
        noise_stdv=0.02,  # accepted for backward compat; ignored (noise is external)
        tau_temp=52.0,
        tau_actuator=5.0,
        mv_deadtime_steps=3,
        target_coeffs=None,
        domain_randomization=None,
        param_randomization_pct=None,
        randomization_seed=None,
    ):
        self.sample_rate = int(sample_rate)
        self.episode_length = int(episode_length)
        self.episode_counter = 0
        self.done = False

        self.state_variables = [
            'CONTROL_TEMP_PV_C',
            'CONTROL_TEMP_SV_C',
            'REFLUX_MV_%',
            'FEED_DV_kg_h',
        ]

        self.control_temp_pv_index = 0
        self.control_temp_sv_index = 1
        self.reflux_mv_index = 2
        self.feed_dv_index = 3

        self.mv_indices = [self.reflux_mv_index]
        self.cv_indices = [self.control_temp_pv_index]
        self.dv_indices = [self.feed_dv_index]

        self.state_is_normalized = False

        self.episode_array = np.zeros(
            (self.episode_length + 1, len(self.state_variables)),
            dtype='float32',
        )

        self.reflux_limits = (20.0, 80.0)
        self.feed_limits = (60.0, 140.0)
        self.temp_limits = (68.0, 96.0)

        # Normalization ranges exposed as single source of truth for sim_factory.
        # MV norm range is intentionally wider than operator bounds [20,80] so
        # that runtime bound-step events have physical headroom; reflux=0 is
        # not part of the simulator dynamics so we cap the lower edge at 10.
        self.mv_normalization_ranges = [[10.0, 90.0]]
        self.cv_normalization_ranges = [[68.0, 96.0]]
        self.dv_normalization_ranges = [[60.0, 140.0]]
        self.state_normalization_ranges = [
            [68.0, 96.0],
            [68.0, 96.0],
            [10.0, 90.0],
            [60.0, 140.0],
        ]

        self.base_tau_temp = float(tau_temp)
        self.base_tau_actuator = float(tau_actuator)
        self.base_mv_deadtime_steps = int(max(0, mv_deadtime_steps))

        default_coeffs = {
            'bias': 82.0,
            'k_u': -16.0,
            'k_d': 18.0,
        }
        c = target_coeffs if isinstance(target_coeffs, dict) else {}
        self.base_target_coeffs = {
            'bias': float(c.get('bias', default_coeffs['bias'])),
            'k_u': float(c.get('k_u', default_coeffs['k_u'])),
            'k_d': float(c.get('k_d', default_coeffs['k_d'])),
        }

        # --- Domain randomizer (generic utility) --------------------------
        self._randomizer = DomainRandomizer(
            env_prefixes=['SIM', 'DISTILLATION'],
            domain_randomization=domain_randomization,
            param_randomization_pct=param_randomization_pct,
            randomization_seed=randomization_seed,
        )

        self.tau_temp = self.base_tau_temp
        self.tau_actuator = self.base_tau_actuator
        self.mv_deadtime_steps = self.base_mv_deadtime_steps
        self.target_coeffs = dict(self.base_target_coeffs)

        self.u_actual = np.array([50.0], dtype='float32')
        self.u_history = []

        # --- Disturbance offsets (from mixin) -----------------------------
        self._init_disturbance_offsets()

    def _sample_episode_dynamics(self):
        rs = self._randomizer.rand_scale
        self.tau_temp = float(self.base_tau_temp * rs())
        self.tau_actuator = float(self.base_tau_actuator * rs())

        dt_scale = rs()
        self.mv_deadtime_steps = int(max(0, round(self.base_mv_deadtime_steps * dt_scale)))

        self.target_coeffs = {
            'bias': float(self.base_target_coeffs['bias']),
            'k_u': float(self.base_target_coeffs['k_u'] * rs()),
            'k_d': float(self.base_target_coeffs['k_d'] * rs()),
        }

    def reset(self):
        self.episode_counter = 0
        self.done = False
        self.reset_disturbance_offsets()
        self._sample_episode_dynamics()

        _rng = self._randomizer.rng
        sv = 82.0
        temp = float(np.clip(sv + _rng.standard_normal() * 0.7, self.temp_limits[0], self.temp_limits[1]))
        reflux = float(np.clip(50.0 + _rng.standard_normal() * 2.5, self.reflux_limits[0], self.reflux_limits[1]))
        feed = float(np.clip(100.0 + _rng.standard_normal() * 3.0, self.feed_limits[0], self.feed_limits[1]))

        self.u_actual = np.array([reflux], dtype='float32')
        self.u_history = [self.u_actual.copy() for _ in range(self.mv_deadtime_steps + 1)]

        x0 = np.array([temp, sv, reflux, feed], dtype='float32')
        self.episode_array[:] = 0.0
        self.episode_array[0] = x0
        return x0.copy(), self.done

    def _temp_target(self, delayed_reflux, feed):
        u_dev = (float(delayed_reflux) - 50.0) / 50.0
        d_dev = (float(feed) - 100.0) / 100.0
        return float(
            self.target_coeffs['bias']
            + self.target_coeffs['k_u'] * u_dev
            + self.target_coeffs['k_d'] * d_dev
        )

    def step(self, action):
        if np.isscalar(action):
            action = np.array([action], dtype='float32')
        else:
            action = np.asarray(action, dtype='float32').reshape(-1)

        if action.shape[0] != 1:
            raise ValueError('Action must contain exactly 1 value: reflux')

        reflux_cmd = float(np.clip(action[0], self.reflux_limits[0], self.reflux_limits[1]))

        prev = self.episode_array[self.episode_counter]
        temp = float(prev[self.control_temp_pv_index])
        sv = float(prev[self.control_temp_sv_index])
        feed = float(prev[self.feed_dv_index])

        alpha_u = self.sample_rate / max(self.tau_actuator, float(self.sample_rate))
        self.u_actual[0] = float(self.u_actual[0] + alpha_u * (reflux_cmd - self.u_actual[0]))

        self.u_history.append(self.u_actual.copy())
        keep = self.mv_deadtime_steps + 1
        if len(self.u_history) > keep:
            self.u_history = self.u_history[-keep:]

        if self.mv_deadtime_steps >= len(self.u_history):
            delayed_reflux = float(self.u_history[0][0])
        else:
            delayed_reflux = float(self.u_history[-(self.mv_deadtime_steps + 1)][0])

        feed_ref = 100.0 + self._dv_offsets.get(self.feed_dv_index, 0.0)
        feed = float(np.clip(feed + 0.08 * (feed_ref - feed), self.feed_limits[0], self.feed_limits[1]))

        target = self._temp_target(delayed_reflux, feed) + self._cv_offsets.get(self.control_temp_pv_index, 0.0)
        temp = float(temp + (self.sample_rate / max(self.tau_temp, float(self.sample_rate))) * (target - temp))

        self.episode_counter += 1
        self.episode_array[self.episode_counter] = prev
        self.episode_array[self.episode_counter, self.control_temp_pv_index] = float(np.clip(temp, self.temp_limits[0], self.temp_limits[1]))
        self.episode_array[self.episode_counter, self.control_temp_sv_index] = sv
        self.episode_array[self.episode_counter, self.reflux_mv_index] = float(self.u_actual[0])
        self.episode_array[self.episode_counter, self.feed_dv_index] = float(np.clip(feed, self.feed_limits[0], self.feed_limits[1]))

        if self.episode_counter > self.episode_length - 1:
            self.done = True

        return self.episode_array[self.episode_counter].copy(), self.done

    def save_eps(self, folder='data/raw_data', prefix='test_sim'):
        os.makedirs(folder, exist_ok=True)
        timestamp = [datetime.datetime(2023, 3, 1)]
        for t in range(1, self.episode_array.shape[0]):
            timestamp.append(timestamp[t - 1] + datetime.timedelta(seconds=self.sample_rate))

        tme = pd.DataFrame(timestamp, columns=['TimeStamp'])
        eps = pd.DataFrame(self.episode_array, columns=self.state_variables)
        data = pd.concat([tme, eps], axis=1)
        data.to_csv(os.path.join(folder, f'{prefix}.csv'), index=False)
