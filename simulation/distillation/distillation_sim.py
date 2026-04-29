"""Distillation tower simulator used by latent-control training/validation.

Functionality:
- Simulates a simplified MIMO distillation process with four manipulated
    variables (reflux, boilup, cooling, feed_flow) and feed composition DV.
- Pure deterministic physics: actuator lag, MV dead-time delays, first-order
    process dynamics.
- All stochastic elements (process noise, measurement noise, noise amplitude
    jitter) are injected externally by ``SimNoiseWrapper`` using a dynamics-
    derived config from ``control_noise_config``.  Domain randomization of
    internal parameters (taus, gains, dead times) is controlled by the
    ``DomainRandomizer`` whose range can be overridden externally.

Inputs:
- Constructor parameters: ``episode_length``, ``sample_rate``.
- Optional env controls:
    - ``DISTILLATION_DOMAIN_RANDOMIZATION`` (default enabled)
    - ``DISTILLATION_PARAM_RANDOMIZATION_PCT`` (default ``0.10``)
    - ``DISTILLATION_DOMAIN_RANDOMIZATION_SEED`` (for reproducibility)

Outputs:
- ``reset() -> (state, done)`` and ``step(action) -> (state, done)``.
- Optional saved episode CSV/plots via ``save_eps()``.

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
import matplotlib.pyplot as plt

from utils.sim_noise import DisturbanceOffsetMixin, DomainRandomizer


class DistillationTower(DisturbanceOffsetMixin):
    # CV physics consumes `self._cv_offsets` in `step()` for all three
    # CV channels (top/bottom/pressure targets), so unmeasured-CV
    # disturbances produce a sustained, dynamics-respecting response
    # without needing the post-step hold in the disturbance layer.
    honors_cv_disturbance_offsets: bool = True

    """Simple MIMO distillation tower simulator for RL training."""

    def __init__(
        self,
        episode_length,
        sample_rate=1,
        noise_stdv=0.01,  # accepted for backward compat; ignored (noise is external)
        tau_top=55.0,
        tau_bottom=75.0,
        tau_pressure=20.0,
        tau_actuator=8.0,
        mv_deadtime_steps=None,
        target_coeffs=None,
        domain_randomization=None,
        param_randomization_pct=None,
        randomization_seed=None,
        enforce_min_meas_noise=True,  # accepted for backward compat; ignored
    ):
        self.sample_rate = sample_rate
        self.episode_length = episode_length
        self.episode_counter = 0
        self.done = False

        self.state_variables = [
            'TOP_COMP_PV_%',
            'TOP_COMP_SV_%',
            'BOTTOM_COMP_PV_%',
            'BOTTOM_COMP_SV_%',
            'PRESSURE_PV_kPa',
            'PRESSURE_SV_kPa',
            'REFLUX_MV_%',
            'BOILUP_MV_%',
            'COOLING_MV_%',
            'FEED_FLOW_PV_%',
            'FEED_COMP_PV_%',
        ]

        self.top_pv_index = self.state_variables.index('TOP_COMP_PV_%')
        self.top_sv_index = self.state_variables.index('TOP_COMP_SV_%')
        self.bottom_pv_index = self.state_variables.index('BOTTOM_COMP_PV_%')
        self.bottom_sv_index = self.state_variables.index('BOTTOM_COMP_SV_%')
        self.pressure_pv_index = self.state_variables.index('PRESSURE_PV_kPa')
        self.pressure_sv_index = self.state_variables.index('PRESSURE_SV_kPa')
        self.reflux_mv_index = self.state_variables.index('REFLUX_MV_%')
        self.boilup_mv_index = self.state_variables.index('BOILUP_MV_%')
        self.cooling_mv_index = self.state_variables.index('COOLING_MV_%')
        self.feed_flow_index = self.state_variables.index('FEED_FLOW_PV_%')
        self.feed_comp_index = self.state_variables.index('FEED_COMP_PV_%')

        self.mv_indices = [
            self.reflux_mv_index,
            self.boilup_mv_index,
            self.cooling_mv_index,
            self.feed_flow_index,
        ]

        # Controlled and Disturbance variable indices, exposed on the sim as the
        # single source of truth (sim_factory prefers these over control_setup.json).
        self.cv_indices = [
            self.top_pv_index,
            self.bottom_pv_index,
            self.pressure_pv_index,
        ]
        self.dv_indices = [self.feed_comp_index]

        self.state_is_normalized = False

        # Normalization ranges (engineering units). One entry per channel.
        # MV norm ranges are wider than the operator bounds in
        # control_objective.json so that runtime bound-step events have
        # physical headroom.  Lower edges stay above 0 because operating
        # reflux/boilup/cooling/feed at 0 is not part of the simulator's
        # learned dynamics.
        self.mv_normalization_ranges = [
            [2.0, 98.0],   # reflux
            [2.0, 98.0],   # boilup
            [2.0, 98.0],   # cooling
            [25.0, 75.0],  # feed_flow
        ]
        self.cv_normalization_ranges = [
            [60.0, 100.0],   # top comp
            [60.0, 100.0],   # bottom comp
            [120.0, 240.0],  # pressure
        ]
        self.dv_normalization_ranges = [
            [30.0, 70.0],  # feed composition
        ]
        self.state_normalization_ranges = [
            [60.0, 100.0],   # TOP_COMP_PV_%
            [60.0, 100.0],   # TOP_COMP_SV_%
            [60.0, 100.0],   # BOTTOM_COMP_PV_%
            [60.0, 100.0],   # BOTTOM_COMP_SV_%
            [120.0, 240.0],  # PRESSURE_PV_kPa
            [120.0, 240.0],  # PRESSURE_SV_kPa
            [2.0, 98.0],     # REFLUX_MV_%
            [2.0, 98.0],     # BOILUP_MV_%
            [2.0, 98.0],     # COOLING_MV_%
            [25.0, 75.0],    # FEED_FLOW_PV_%
            [30.0, 70.0],    # FEED_COMP_PV_%
        ]

        self.episode_array = np.zeros(
            (self.episode_length + 1, len(self.state_variables)),
            dtype='float32',
        )

        self.u_actual = np.array([50.0, 50.0, 50.0, 50.0], dtype='float32')

        # Base first-order dynamics. Per-episode domain randomization perturbs
        # gains, taus, and dead times around these nominal values.
        self.base_tau_top = float(tau_top)
        self.base_tau_bottom = float(tau_bottom)
        self.base_tau_pressure = float(tau_pressure)
        self.base_tau_actuator = float(tau_actuator)

        # MV transport/actuation dead times (in scan steps) for
        # reflux/boilup/cooling/feed_flow.
        if mv_deadtime_steps is None:
            self.base_mv_deadtime_steps = np.array([6, 8, 5, 4], dtype='int32')
        else:
            dt = np.asarray(mv_deadtime_steps, dtype='int32').reshape(-1)
            if dt.size < 4:
                pad = np.array([6, 8, 5, 4], dtype='int32')
                dt = np.concatenate([dt, pad[dt.size:]])
            self.base_mv_deadtime_steps = np.maximum(0, dt[:4])

        # Nominal steady-state target mapping coefficients.
        default_coeffs = {
            'top': {
                'bias': 86.0,
                'mv': np.array([0.16, 0.10, -0.08], dtype='float32'),
                'feed_flow': -8.0,
                'feed_comp': 7.0,
            },
            'bottom': {
                'bias': 90.0,
                'mv': np.array([-0.13, 0.17, -0.04], dtype='float32'),
                'feed_flow': -7.0,
                'feed_comp': -6.0,
            },
            'pressure': {
                'bias': 170.0,
                'mv': np.array([0.0, 0.24, -0.30], dtype='float32'),
                'feed_flow': 10.0,
                'feed_comp': 2.5,
            },
        }

        self.base_target_coeffs = self._normalize_target_coeffs(target_coeffs, default_coeffs)

        # Domain randomization controls.
        self._randomizer = DomainRandomizer(
            env_prefixes=['DISTILLATION'],
            domain_randomization=domain_randomization,
            param_randomization_pct=param_randomization_pct,
            randomization_seed=randomization_seed,
        )

        self.tau_top = self.base_tau_top
        self.tau_bottom = self.base_tau_bottom
        self.tau_pressure = self.base_tau_pressure
        self.tau_actuator = self.base_tau_actuator
        self.mv_deadtime_steps = self.base_mv_deadtime_steps.copy()
        self.target_coeffs = self.base_target_coeffs
        self.u_history = []

        # --- Disturbance offsets (from mixin) -----------------------------
        self._init_disturbance_offsets()

    def _normalize_target_coeffs(self, user_coeffs, defaults):
        if not isinstance(user_coeffs, dict):
            return defaults

        out = {}
        for key in ('top', 'bottom', 'pressure'):
            src = user_coeffs.get(key, {}) if isinstance(user_coeffs.get(key, {}), dict) else {}
            d = defaults[key]

            bias = float(src.get('bias', d['bias']))

            mv = src.get('mv', d['mv'])
            mv_arr = np.asarray(mv, dtype='float32').reshape(-1)
            if mv_arr.size < 3:
                mv_arr = np.concatenate([mv_arr, np.asarray(d['mv'], dtype='float32')[mv_arr.size:]])
            mv_arr = mv_arr[:3].astype('float32')

            feed_flow = float(src.get('feed_flow', d['feed_flow']))
            feed_comp = float(src.get('feed_comp', d['feed_comp']))

            out[key] = {
                'bias': bias,
                'mv': mv_arr,
                'feed_flow': feed_flow,
                'feed_comp': feed_comp,
            }

        return out

    def _rand_scale(self, size=None):
        return self._randomizer.rand_scale(size=size)

    def _sample_episode_dynamics(self):
        # Time constants
        self.tau_top = float(self.base_tau_top * self._rand_scale())
        self.tau_bottom = float(self.base_tau_bottom * self._rand_scale())
        self.tau_pressure = float(self.base_tau_pressure * self._rand_scale())
        self.tau_actuator = float(self.base_tau_actuator * self._rand_scale())

        # Dead times (integer scan steps)
        dt_scale = np.asarray(self._rand_scale(size=4), dtype='float32')
        varied_dt = np.rint(self.base_mv_deadtime_steps.astype('float32') * dt_scale).astype('int32')
        self.mv_deadtime_steps = np.maximum(0, varied_dt)

        # Gain randomization around nominal process map
        gain_scale = np.asarray(self._rand_scale(size=3), dtype='float32')
        ff_scale = np.asarray(self._rand_scale(size=3), dtype='float32')
        fc_scale = np.asarray(self._rand_scale(size=3), dtype='float32')

        self.target_coeffs = {
            'top': {
                'bias': float(self.base_target_coeffs['top']['bias']),
                'mv': (self.base_target_coeffs['top']['mv'] * gain_scale).astype('float32'),
                'feed_flow': float(self.base_target_coeffs['top']['feed_flow'] * ff_scale[0]),
                'feed_comp': float(self.base_target_coeffs['top']['feed_comp'] * fc_scale[0]),
            },
            'bottom': {
                'bias': float(self.base_target_coeffs['bottom']['bias']),
                'mv': (self.base_target_coeffs['bottom']['mv'] * gain_scale).astype('float32'),
                'feed_flow': float(self.base_target_coeffs['bottom']['feed_flow'] * ff_scale[1]),
                'feed_comp': float(self.base_target_coeffs['bottom']['feed_comp'] * fc_scale[1]),
            },
            'pressure': {
                'bias': float(self.base_target_coeffs['pressure']['bias']),
                'mv': (self.base_target_coeffs['pressure']['mv'] * gain_scale).astype('float32'),
                'feed_flow': float(self.base_target_coeffs['pressure']['feed_flow'] * ff_scale[2]),
                'feed_comp': float(self.base_target_coeffs['pressure']['feed_comp'] * fc_scale[2]),
            },
        }

    def reset(self):
        """Initialize tower state and return first observation."""
        self.episode_counter = 0
        self.done = False
        self.reset_disturbance_offsets()
        self._sample_episode_dynamics()

        _rng = self._randomizer.rng
        top_sv = 95.0
        bottom_sv = 93.0
        pressure_sv = 175.0

        top_pv = np.clip(top_sv + _rng.standard_normal() * 1.5, 80.0, 99.8)
        bottom_pv = np.clip(bottom_sv + _rng.standard_normal() * 2.0, 75.0, 99.5)
        pressure_pv = np.clip(pressure_sv + _rng.standard_normal() * 2.0, 150.0, 210.0)

        reflux_mv = np.clip(50 + _rng.standard_normal() * 4, 0, 100)
        boilup_mv = np.clip(50 + _rng.standard_normal() * 4, 0, 100)
        cooling_mv = np.clip(50 + _rng.standard_normal() * 4, 0, 100)

        feed_flow = np.clip(50 + _rng.standard_normal() * 2, 30, 70)
        feed_comp = np.clip(50 + _rng.standard_normal() * 2, 30, 70)

        initial_state = np.array([
            top_pv,
            top_sv,
            bottom_pv,
            bottom_sv,
            pressure_pv,
            pressure_sv,
            reflux_mv,
            boilup_mv,
            cooling_mv,
            feed_flow,
            feed_comp,
        ], dtype='float32')

        self.u_actual = np.array([reflux_mv, boilup_mv, cooling_mv, feed_flow], dtype='float32')
        max_dt = int(np.max(self.mv_deadtime_steps)) if len(self.mv_deadtime_steps) else 0
        self.u_history = [self.u_actual.copy() for _ in range(max_dt + 1)]
        self.episode_array[:] = 0.0
        self.episode_array[0] = initial_state

        return initial_state.copy(), self.done

    def _compute_targets(self, u, feed_flow, feed_comp):
        u_mv = np.asarray(u[:3], dtype='float32')
        feed_flow_dev = (feed_flow - 50.0) / 50.0
        feed_comp_dev = (feed_comp - 50.0) / 50.0

        top_cfg = self.target_coeffs['top']
        bot_cfg = self.target_coeffs['bottom']
        pre_cfg = self.target_coeffs['pressure']

        top_target = (
            float(top_cfg['bias'])
            + float(np.dot(top_cfg['mv'], u_mv))
            + float(top_cfg['feed_flow']) * feed_flow_dev
            + float(top_cfg['feed_comp']) * feed_comp_dev
        )

        bottom_target = (
            float(bot_cfg['bias'])
            + float(np.dot(bot_cfg['mv'], u_mv))
            + float(bot_cfg['feed_flow']) * feed_flow_dev
            + float(bot_cfg['feed_comp']) * feed_comp_dev
        )

        pressure_target = (
            float(pre_cfg['bias'])
            + float(np.dot(pre_cfg['mv'], u_mv))
            + float(pre_cfg['feed_flow']) * feed_flow_dev
            + float(pre_cfg['feed_comp']) * feed_comp_dev
        )

        return top_target, bottom_target, pressure_target

    def step(self, action):
        """Step one simulation interval with 4 MVs: [reflux, boilup, cooling, feed_flow]."""
        if np.isscalar(action):
            action = np.array([action, self.u_actual[1], self.u_actual[2], self.u_actual[3]], dtype='float32')
        else:
            action = np.asarray(action, dtype='float32').reshape(-1)

        if action.shape[0] != 4:
            raise ValueError('Action must contain 4 values: reflux, boilup, cooling, feed_flow')

        action_lo = np.array([0.0, 0.0, 0.0, 35.0], dtype='float32')
        action_hi = np.array([100.0, 100.0, 100.0, 65.0], dtype='float32')
        action = np.clip(action, action_lo, action_hi)

        prev = self.episode_array[self.episode_counter]
        top_pv = float(prev[self.top_pv_index])
        bottom_pv = float(prev[self.bottom_pv_index])
        pressure_pv = float(prev[self.pressure_pv_index])
        feed_flow = float(prev[self.feed_flow_index])
        feed_comp = float(prev[self.feed_comp_index])

        alpha_u = self.sample_rate / max(self.tau_actuator, self.sample_rate)
        self.u_actual = self.u_actual + alpha_u * (action - self.u_actual)

        # Push new actuator state into dead-time history.
        self.u_history.append(self.u_actual.copy())
        max_dt = int(np.max(self.mv_deadtime_steps)) if len(self.mv_deadtime_steps) else 0
        keep = max_dt + 1
        if len(self.u_history) > keep:
            self.u_history = self.u_history[-keep:]

        delayed_u = np.zeros_like(self.u_actual)
        for i in range(4):
            dt = int(self.mv_deadtime_steps[i])
            if dt >= len(self.u_history):
                delayed_u[i] = float(self.u_history[0][i])
            else:
                delayed_u[i] = float(self.u_history[-(dt + 1)][i])

        feed_flow = float(np.clip(delayed_u[3], 35.0, 65.0))
        feed_comp = float(np.clip(feed_comp, 30.0, 70.0))

        top_target, bottom_target, pressure_target = self._compute_targets(
            delayed_u,
            feed_flow,
            feed_comp,
        )

        # Apply persistent CV offsets to process targets.
        top_target += self._cv_offsets.get(self.top_pv_index, 0.0)
        bottom_target += self._cv_offsets.get(self.bottom_pv_index, 0.0)
        pressure_target += self._cv_offsets.get(self.pressure_pv_index, 0.0)

        top_pv = top_pv + (self.sample_rate / self.tau_top) * (top_target - top_pv)
        bottom_pv = bottom_pv + (self.sample_rate / self.tau_bottom) * (bottom_target - bottom_pv)
        pressure_pv = pressure_pv + (self.sample_rate / self.tau_pressure) * (pressure_target - pressure_pv)

        self.episode_counter += 1
        self.episode_array[self.episode_counter] = prev

        self.episode_array[self.episode_counter, self.top_pv_index] = np.clip(top_pv, 60.0, 100.0)
        self.episode_array[self.episode_counter, self.bottom_pv_index] = np.clip(bottom_pv, 60.0, 100.0)
        self.episode_array[self.episode_counter, self.pressure_pv_index] = np.clip(pressure_pv, 120.0, 240.0)
        self.episode_array[self.episode_counter, self.reflux_mv_index] = self.u_actual[0]
        self.episode_array[self.episode_counter, self.boilup_mv_index] = self.u_actual[1]
        self.episode_array[self.episode_counter, self.cooling_mv_index] = self.u_actual[2]
        self.episode_array[self.episode_counter, self.feed_flow_index] = np.clip(feed_flow, 35.0, 65.0)
        self.episode_array[self.episode_counter, self.feed_comp_index] = np.clip(feed_comp, 30.0, 70.0)

        state_ = self.episode_array[self.episode_counter].copy()

        if self.episode_counter > self.episode_length - 1:
            self.done = True

        return state_, self.done

    def save_eps(self, plot=False, folder='data/raw_data', prefix='distillation'):
        """Save simulated trajectory to csv chunks and optional plots."""
        os.makedirs(folder, exist_ok=True)

        timestamp = [datetime.datetime(2023, 3, 1)]
        for t in range(1, self.episode_array.shape[0]):
            timestamp.append(timestamp[t - 1] + datetime.timedelta(seconds=self.sample_rate))

        tme = pd.DataFrame(timestamp, columns=['TimeStamp'])
        eps = pd.DataFrame(self.episode_array, columns=self.state_variables)
        data = pd.concat([tme, eps], axis=1)

        if data.shape[0] > 10000:
            n_chunks = int(np.ceil(data.shape[0] / 10000))
            for i in range(n_chunks):
                chunk = data[i * 10000:(i + 1) * 10000]
                chunk.to_csv(
                    os.path.join(folder, f'{prefix}_{i}.csv'),
                    sep=',',
                    index=False,
                    header=True,
                )
        else:
            data.to_csv(
                os.path.join(folder, f'{prefix}.csv'),
                sep=',',
                index=False,
                header=True,
            )

        if plot:
            fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
            axes[0].plot(eps['TOP_COMP_SV_%'], label='Top SV')
            axes[0].plot(eps['TOP_COMP_PV_%'], label='Top PV')
            axes[0].set_ylabel('Top %')
            axes[0].legend(loc='lower right')

            axes[1].plot(eps['BOTTOM_COMP_SV_%'], label='Bottom SV')
            axes[1].plot(eps['BOTTOM_COMP_PV_%'], label='Bottom PV')
            axes[1].set_ylabel('Bottom %')
            axes[1].legend(loc='lower right')

            axes[2].plot(eps['PRESSURE_SV_kPa'], label='P SV')
            axes[2].plot(eps['PRESSURE_PV_kPa'], label='P PV')
            axes[2].set_ylabel('kPa')
            axes[2].set_xlabel('Time (s)')
            axes[2].legend(loc='lower right')

            plt.tight_layout()
            plt.show()
