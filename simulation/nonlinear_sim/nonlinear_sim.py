"""Fast SISO NONLINEAR steam-heated exchanger simulator for WM-gain validation.

A small, realistic process-industry example whose input→CV GAIN varies with
operating point (unlike ``test_sim``, which is linear).  It exists to test that
the self-supervised steady-state (DC-gain) match trains an UNBIASED world model
where the identified-gain ``gain_match`` cannot (a single scalar gain is wrong
for a nonlinear plant).

Process:
- Outlet temperature CV heated by a steam coil whose valve is an EQUAL-PERCENTAGE
  control valve (the ubiquitous industrial characteristic): steam flow
  ``f(x) = R^(x-1)`` with fractional opening ``x`` and rangeability ``R`` — so
  ``dCV/dvalve`` grows ~exponentially with valve opening (~2.5x across the band).
- Feed flow DV cools the exchanger HYPERBOLICALLY (``T ∝ 1/feed``): a second,
  independent nonlinearity, and the MV/DV gains are cross-coupled (each depends
  on the other's operating point).
- Actuator lag, MV transport dead time, first-order thermal lag, DV
  mean-reversion — identical dynamic structure to ``test_sim``.

Structure (interface, mixin, DR, disturbance offsets, normalization-range
bookkeeping) is intentionally IDENTICAL to ``simulation/test_sim/test_sim.py``;
only the steady-state map ``_temp_target`` is nonlinear.

Inputs:
- Constructor: episode_length, sample_rate.
- Optional env vars: DREAMER_SIM_DOMAIN_RANDOMIZATION,
    DREAMER_SIM_PARAM_RANDOMIZATION_PCT, DREAMER_SIM_DOMAIN_RANDOMIZATION_SEED.
    Leftover SIM_* names are ignored.

Outputs:
- reset() -> (state, done)
- step(action) -> (state, done)

Normalization ranges: see the note in ``test_sim.py`` — per-group lists and the
per-slot ``state_normalization_ranges`` are kept in sync by hand.
"""

import datetime
import os

import numpy as np
import pandas as pd

from utils.sim_noise import DisturbanceOffsetMixin, DomainRandomizer
from utils.initial_conditions import sample_initial_value


class HeatExchangerTower(DisturbanceOffsetMixin):
    # CV physics consumes ``self._cv_offsets`` in ``step()`` so unmeasured-CV
    # disturbances produce a sustained, dynamics-respecting response.
    honors_cv_disturbance_offsets: bool = True

    """Single-loop NONLINEAR steam-heated exchanger for WM-gain validation."""

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
            'OUTLET_TEMP_PV_C',
            'OUTLET_TEMP_SV_C',
            'STEAM_VALVE_MV_%',
            'FEED_FLOW_DV_kg_h',
        ]

        self.outlet_temp_pv_index = 0
        self.outlet_temp_sv_index = 1
        self.steam_valve_mv_index = 2
        self.feed_flow_dv_index = 3

        self.mv_indices = [self.steam_valve_mv_index]
        self.cv_indices = [self.outlet_temp_pv_index]
        self.dv_indices = [self.feed_flow_dv_index]

        self.state_is_normalized = False

        self.episode_array = np.zeros(
            (self.episode_length + 1, len(self.state_variables)),
            dtype='float32',
        )

        self.valve_limits = (20.0, 80.0)
        self.feed_limits = (60.0, 140.0)
        self.temp_limits = (60.0, 120.0)

        # Normalization ranges (single source of truth for sim_factory).  The
        # valve MV norm range is wider than the operator bounds [20,80] so
        # runtime bound-step events have headroom.
        self.mv_normalization_ranges = [[10.0, 90.0]]
        self.cv_normalization_ranges = [[60.0, 120.0]]
        self.dv_normalization_ranges = [[60.0, 140.0]]
        self.state_normalization_ranges = [
            [60.0, 120.0],
            [60.0, 120.0],
            [10.0, 90.0],
            [60.0, 140.0],
        ]

        self.base_tau_temp = float(tau_temp)
        self.base_tau_actuator = float(tau_actuator)
        self.base_mv_deadtime_steps = int(max(0, mv_deadtime_steps))

        # NONLINEAR steady-state map coefficients:
        #   T_ss = t_in + heat_gain * valve_flow(x) * (feed_ref / feed)
        # with the EQUAL-PERCENTAGE valve ``valve_flow(x) = R^(x-1)`` (fractional
        # opening x = valve%/100, rangeability R).  At the nominal operating
        # point (valve 50 %, feed 100) this gives T_ss = 45 + 126*0.316 ≈ 85 C.
        default_coeffs = {
            't_in': 45.0,           # inlet / ambient temperature (C)
            'heat_gain': 126.0,     # steam→temperature scale (C)
            'rangeability': 10.0,   # equal-% valve rangeability R
            'feed_ref': 100.0,      # feed at which the coupling term is unity
        }
        c = target_coeffs if isinstance(target_coeffs, dict) else {}
        self.base_target_coeffs = {
            't_in': float(c.get('t_in', default_coeffs['t_in'])),
            'heat_gain': float(c.get('heat_gain', default_coeffs['heat_gain'])),
            'rangeability': float(c.get('rangeability',
                                        default_coeffs['rangeability'])),
            'feed_ref': float(c.get('feed_ref', default_coeffs['feed_ref'])),
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

        # MV dead-time variance: force at least +/-1-step integer noise so the
        # WM sees multiple delay realisations even for small base values.  When
        # DR is disabled the noise collapses to 0 deterministically.
        rng = self._randomizer.rng
        if self._randomizer.enabled and self.base_mv_deadtime_steps > 0:
            span = max(1, int(round(self._randomizer.frac
                                     * float(self.base_mv_deadtime_steps))))
            jitter = int(rng.integers(-span, span + 1))
            self.mv_deadtime_steps = int(max(0, self.base_mv_deadtime_steps + jitter))
        else:
            self.mv_deadtime_steps = int(max(0, self.base_mv_deadtime_steps))

        # Randomize the overall heat gain and the valve rangeability (the gain
        # SHAPE) so the WM sees plant-to-plant nonlinearity variability; keep
        # t_in / feed_ref fixed (they set the operating envelope).
        self.target_coeffs = {
            't_in': float(self.base_target_coeffs['t_in']),
            'heat_gain': float(self.base_target_coeffs['heat_gain'] * rs()),
            'rangeability': float(np.clip(
                self.base_target_coeffs['rangeability'] * rs(), 3.0, 50.0)),
            'feed_ref': float(self.base_target_coeffs['feed_ref']),
        }

    def reset(self):
        self.episode_counter = 0
        self.done = False
        self.reset_disturbance_offsets()
        self._randomizer.sample_episode(
            n_dvs=1,
            identified_tau=float(self.base_tau_temp),
            identified_dead_time=float(self.base_mv_deadtime_steps),
        )
        self._sample_episode_dynamics()

        _rng = self._randomizer.rng
        sv = 85.0
        temp = sample_initial_value(
            _rng, nominal=sv, bounds=self.temp_limits, legacy_sigma=0.7,
        )
        valve = sample_initial_value(
            _rng, nominal=50.0, bounds=self.valve_limits, legacy_sigma=2.5,
        )
        feed = sample_initial_value(
            _rng, nominal=100.0, bounds=self.feed_limits, legacy_sigma=3.0,
        )

        self.u_actual = np.array([valve], dtype='float32')
        self.u_history = [self.u_actual.copy() for _ in range(self.mv_deadtime_steps + 1)]

        x0 = np.array([temp, sv, valve, feed], dtype='float32')
        self.episode_array[:] = 0.0
        self.episode_array[0] = x0
        return x0.copy(), self.done

    def _valve_flow(self, valve_pct):
        """Equal-percentage control-valve characteristic: fractional steam flow
        ``R^(x-1)`` for fractional opening ``x = valve%/100`` (clamped to the
        physical valve travel).  This is the source of the MV-gain nonlinearity.
        """
        x = float(np.clip(valve_pct, 0.0, 100.0)) / 100.0
        R = max(1.0001, float(self.target_coeffs['rangeability']))
        return float(R ** (x - 1.0))

    def _temp_target(self, delayed_valve, feed):
        # NONLINEAR steady-state outlet temperature: equal-% valve heating,
        # hyperbolic feed-flow cooling.  The local MV gain d(target)/d(valve)
        # scales with the steam flow (~exponential in valve) and inversely with
        # feed → varies with operating point (the point of this sim).
        flow = self._valve_flow(delayed_valve)
        feed_ref = float(self.target_coeffs['feed_ref'])
        feed_c = float(np.clip(feed, self.feed_limits[0], self.feed_limits[1]))
        return float(
            self.target_coeffs['t_in']
            + self.target_coeffs['heat_gain'] * flow * (feed_ref / feed_c)
        )

    def step(self, action):
        if np.isscalar(action):
            action = np.array([action], dtype='float32')
        else:
            action = np.asarray(action, dtype='float32').reshape(-1)

        if action.shape[0] != 1:
            raise ValueError('Action must contain exactly 1 value: steam valve')

        valve_cmd = float(np.clip(action[0], self.valve_limits[0], self.valve_limits[1]))

        prev = self.episode_array[self.episode_counter]
        temp = float(prev[self.outlet_temp_pv_index])
        sv = float(prev[self.outlet_temp_sv_index])
        feed = float(prev[self.feed_flow_dv_index])

        alpha_u = self.sample_rate / max(self.tau_actuator, float(self.sample_rate))
        self.u_actual[0] = float(self.u_actual[0] + alpha_u * (valve_cmd - self.u_actual[0]))

        self.u_history.append(self.u_actual.copy())
        keep = self.mv_deadtime_steps + 1
        if len(self.u_history) > keep:
            self.u_history = self.u_history[-keep:]

        if self.mv_deadtime_steps >= len(self.u_history):
            delayed_valve = float(self.u_history[0][0])
        else:
            delayed_valve = float(self.u_history[-(self.mv_deadtime_steps + 1)][0])

        feed_ref = 100.0 + self._dv_offsets.get(self.feed_flow_dv_index, 0.0)
        feed = float(np.clip(feed + 0.08 * (feed_ref - feed), self.feed_limits[0], self.feed_limits[1]))

        target = self._temp_target(delayed_valve, feed) + self._cv_offsets.get(self.outlet_temp_pv_index, 0.0)
        temp = float(temp + (self.sample_rate / max(self.tau_temp, float(self.sample_rate))) * (target - temp))

        self.episode_counter += 1
        self.episode_array[self.episode_counter] = prev
        self.episode_array[self.episode_counter, self.outlet_temp_pv_index] = float(np.clip(temp, self.temp_limits[0], self.temp_limits[1]))
        self.episode_array[self.episode_counter, self.outlet_temp_sv_index] = sv
        self.episode_array[self.episode_counter, self.steam_valve_mv_index] = float(self.u_actual[0])
        self.episode_array[self.episode_counter, self.feed_flow_dv_index] = float(np.clip(feed, self.feed_limits[0], self.feed_limits[1]))

        if self.episode_counter > self.episode_length - 1:
            self.done = True

        return self.episode_array[self.episode_counter].copy(), self.done

    def save_eps(self, folder='data/raw_data', prefix='nonlinear_sim'):
        os.makedirs(folder, exist_ok=True)
        timestamp = [datetime.datetime(2023, 3, 1)]
        for t in range(1, self.episode_array.shape[0]):
            timestamp.append(timestamp[t - 1] + datetime.timedelta(seconds=self.sample_rate))

        tme = pd.DataFrame(timestamp, columns=['TimeStamp'])
        eps = pd.DataFrame(self.episode_array, columns=self.state_variables)
        data = pd.concat([tme, eps], axis=1)
        data.to_csv(os.path.join(folder, f'{prefix}.csv'), index=False)
