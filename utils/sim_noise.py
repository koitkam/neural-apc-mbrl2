"""Generic noise, domain randomization, and disturbance offset utilities.

Provides reusable building blocks for adding stochastic elements to any
deterministic simulator used in the control workflow:

- **OUActionNoise** — Ornstein-Uhlenbeck temporally-correlated noise process.
- **DomainRandomizer** — Per-episode parameter perturbation with env-var
  controls for enable/disable, range, and reproducibility seed.
- **DisturbanceOffsetMixin** — Mixin providing ``set_disturbance_offset`` /
  ``reset_disturbance_offsets`` interface consumed by the disturbance
  curriculum module.
- **SimNoiseWrapper** — Transparent wrapper around any deterministic simulator
  that adds process noise (OU), DV drift noise (OU), and measurement noise
  (white) after each ``step()`` call.  Training / validation code does not
  need to change — all noise is injected via the wrapper.

Noise configuration
-------------------
Each simulator exposes a ``noise_config`` dict attribute built in its
constructor.  The wrapper reads it automatically::

    noise_config = {
        'ou_noise': [
            {'index': 0, 'sigma': 0.01, 'gain': 0.35, 'bounds': (68, 96)},
        ],
        'measurement_noise': [
            {'index': 0, 'sigma': 0.03, 'bounds': (68, 96)},
        ],
    }
"""

import os
from typing import Any, Dict, List, Optional, Tuple

import json
import numpy as np


# ---------------------------------------------------------------------------
# Ornstein-Uhlenbeck noise process
# ---------------------------------------------------------------------------

class OUActionNoise:
    """Temporally-correlated Ornstein-Uhlenbeck noise generator.

    Args:
        mean: Long-term mean level (numpy array).
        std_deviation: Volatility (same shape as *mean*).
        theta: Mean-reversion rate (default 0.15).
        dt: Integration time-step (default 0.01).
        x_initial: Optional starting state; zeros if *None*.
        rng: Optional ``np.random.Generator`` for reproducible noise.
            Falls back to a private unseeded generator if *None*.
    """

    def __init__(self, mean, std_deviation, theta=0.15, dt=1e-2, x_initial=None,
                 rng: Optional[np.random.Generator] = None):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.x_initial = x_initial
        self._rng = rng if rng is not None else np.random.default_rng()
        self.reset()

    def __call__(self):
        x = (
            self.x_prev
            + self.theta * (self.mean - self.x_prev) * self.dt
            + self.std_dev * np.sqrt(self.dt) * self._rng.standard_normal(size=self.mean.shape)
        )
        self.x_prev = x
        return x

    def reset(self):
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = np.zeros_like(self.mean)

    def __repr__(self):
        return (
            f'OUActionNoise(theta={self.theta}, dt={self.dt}, '
            f'std_dev={self.std_dev})'
        )


# ---------------------------------------------------------------------------
# Domain randomizer
# ---------------------------------------------------------------------------

class DomainRandomizer:
    """Per-episode parameter randomization utility.

    Reads enable/disable, range, and seed from environment variables with a
    configurable prefix list.  The first prefix that has a set env var wins;
    if none are set the built-in defaults apply.

    Args:
        env_prefixes: Ordered list of env-var prefixes to check, e.g.
            ``['SIM', 'DISTILLATION']``.  For prefix ``SIM`` the env vars
            ``SIM_DOMAIN_RANDOMIZATION``, ``SIM_PARAM_RANDOMIZATION_PCT``,
            and ``SIM_DOMAIN_RANDOMIZATION_SEED`` are inspected.
        domain_randomization: Explicit override for enable/disable (*None*
            falls back to env var, default enabled).
        param_randomization_pct: Explicit override for the ±% range (*None*
            falls back to env var, default 0.10).
        randomization_seed: Explicit seed (*None* falls back to env var,
            default un-seeded).
    """

    def __init__(
        self,
        env_prefixes: Optional[List[str]] = None,
        domain_randomization: Optional[bool] = None,
        param_randomization_pct: Optional[float] = None,
        randomization_seed=None,
    ):
        prefixes = list(env_prefixes or ['SIM'])

        # --- enabled ---
        env_val = None
        for pfx in prefixes:
            v = os.environ.get(f'{pfx}_DOMAIN_RANDOMIZATION')
            if v is not None:
                env_val = v
                break
        if domain_randomization is not None:
            self.enabled = bool(domain_randomization)
        elif env_val is not None:
            self.enabled = str(env_val).strip().lower() not in {
                '0', 'false', 'no', 'off',
            }
        else:
            self.enabled = True

        # --- fraction ---
        env_pct = None
        for pfx in prefixes:
            v = os.environ.get(f'{pfx}_PARAM_RANDOMIZATION_PCT')
            if v is not None:
                env_pct = v
                break
        if param_randomization_pct is not None:
            self.frac = float(param_randomization_pct)
        elif env_pct is not None:
            self.frac = float(env_pct)
        else:
            self.frac = 0.10
        self.frac = float(np.clip(self.frac, 0.0, 0.5))

        # --- seed / rng ---
        env_seed = ''
        for pfx in prefixes:
            v = os.environ.get(f'{pfx}_DOMAIN_RANDOMIZATION_SEED', '').strip()
            if v:
                env_seed = v
                break
        seed_str = (
            str(randomization_seed).strip()
            if randomization_seed is not None
            else env_seed
        )
        self.rng = (
            np.random.default_rng(int(seed_str))
            if seed_str
            else np.random.default_rng()
        )

        # --- load noise_and_randomization block from CONTROL_SETUP_JSON ---
        # Extra per-episode knobs (output gain/bias, input jitter, actuator
        # lag, MV dead-time, DV mean-shift) are optional and plant-agnostic.
        # Sim classes opt-in by calling ``sample_episode()`` in ``reset()``
        # and consuming the resulting fields (output_gain, output_bias_frac,
        # input_jitter_std_frac, actuator_tau_steps, mv_deadtime_steps,
        # dv_mean_shifts_frac).
        self._config_block: Dict[str, Any] = self._load_config_block()

        # Per-episode sampled values — initialised to neutral defaults so
        # sims that never call ``sample_episode()`` behave as if DR = off
        # for these dimensions (backward compatible).
        self.output_gain: float = 1.0
        self.output_bias_frac: float = 0.0
        self.input_jitter_std_frac: float = 0.0
        self.actuator_tau_steps: float = 0.0
        self.mv_deadtime_steps: int = 0
        self.dv_mean_shifts_frac: Dict[int, float] = {}

    @staticmethod
    def _load_config_block() -> Dict[str, Any]:
        """Read ``noise_and_randomization`` block from CONTROL_SETUP_JSON.

        Returns an empty dict when the file or key is absent so callers
        always get a usable dict.
        """
        path = os.environ.get('CONTROL_SETUP_JSON', '').strip()
        if not path or not os.path.isfile(path):
            return {}
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception:
            return {}
        block = data.get('noise_and_randomization', {}) if isinstance(data, dict) else {}
        return block if isinstance(block, dict) else {}

    def rand_scale(self, size=None):
        """Return uniform random scale(s) in ``[1 - frac, 1 + frac]``."""
        if not self.enabled or self.frac <= 0.0:
            if size is None:
                return 1.0
            return np.ones(size, dtype='float32')
        lo = 1.0 - self.frac
        hi = 1.0 + self.frac
        return self.rng.uniform(lo, hi, size=size)

    def sample_episode(self, *, n_dvs: int = 0,
                       identified_tau: Optional[float] = None,
                       identified_dead_time: Optional[float] = None) -> None:
        """Refresh per-episode randomised plant-wrapper parameters.

        Sims should call this once in ``reset()``. All values are neutral
        (gain=1, bias=0, tau=0, delay=0) when domain randomisation is
        disabled. ``identified_tau`` / ``identified_dead_time`` (in sample
        steps) let the randomiser derive sensible defaults for actuator
        lag and MV dead-time when the user did not specify them in the
        control_setup.json block.

        The sampled fields are plant-agnostic and intended for ONNX-backed
        simulators where the model weights cannot themselves be perturbed:
        gain/bias wrap the model output, input jitter wraps its inputs,
        actuator lag + dead-time shape how the commanded MV reaches the
        model's input window, and DV mean-shift perturbs the rest point
        of each DV's OU drift.
        """
        if not self.enabled or self.frac <= 0.0:
            self.output_gain = 1.0
            self.output_bias_frac = 0.0
            self.input_jitter_std_frac = 0.0
            self.actuator_tau_steps = 0.0
            self.mv_deadtime_steps = 0
            self.dv_mean_shifts_frac = {i: 0.0 for i in range(max(0, int(n_dvs)))}
            return

        cfg = self._config_block

        def _cfg_float(key: str, default: float) -> float:
            """Return cfg[key] as float, treating missing / null / blank
            as ``default``. Keeps explicit ``"null"`` or JSON ``null`` in
            control_setup.json as "use the adaptive default" without
            crashing on ``float(None)``."""
            val = cfg.get(key, None) if isinstance(cfg, dict) else None
            if val is None:
                return float(default)
            try:
                return float(val)
            except (TypeError, ValueError):
                return float(default)

        # Identified plant-dynamics fallbacks (env vars exported by the
        # workflow after dynamics ID); only consulted when control_setup
        # leaves the knob unset. This keeps per-plant defaults adaptive
        # — a fast plant gets small actuator tau, a slow plant gets more.
        env_tau = os.environ.get('SIM_IDENTIFIED_TAU_DOMINANT', '').strip()
        env_dead = os.environ.get('SIM_IDENTIFIED_DEAD_TIME', '').strip()
        if identified_tau is None and env_tau:
            try:
                identified_tau = float(env_tau)
            except Exception:
                identified_tau = None
        if identified_dead_time is None and env_dead:
            try:
                identified_dead_time = float(env_dead)
            except Exception:
                identified_dead_time = None

        # Output gain ±output_gain_pct (defaults to self.frac so it tracks
        # the top-level DR magnitude).
        gain_pct = _cfg_float('output_gain_pct', self.frac)
        gain_pct = float(np.clip(gain_pct, 0.0, 0.5))
        self.output_gain = 1.0 + float(self.rng.uniform(-gain_pct, gain_pct))

        # Output bias: std expressed as fraction of output range, Gaussian.
        bias_pct = max(0.0, _cfg_float('output_bias_pct', 0.02))
        self.output_bias_frac = float(self.rng.normal(0.0, bias_pct))

        # Per-step input jitter std as fraction of input range.
        input_jit = max(0.0, _cfg_float('input_jitter_pct', 0.005))
        self.input_jitter_std_frac = float(input_jit)

        # Actuator first-order lag (in sample steps).
        actuator_cfg_raw = cfg.get('actuator_tau_steps', None) if isinstance(cfg, dict) else None
        if actuator_cfg_raw is None and identified_tau is not None:
            # ~8 % of the dominant plant tau, floored at 1 step, capped at
            # 20 so a very slow plant doesn't get an absurd actuator lag.
            base_tau = float(np.clip(0.08 * float(identified_tau), 1.0, 20.0))
        else:
            try:
                base_tau = float(actuator_cfg_raw) if actuator_cfg_raw is not None else 0.0
            except (TypeError, ValueError):
                base_tau = 0.0
        # Randomise ±frac around the base so different episodes get
        # different actuator characteristics.
        if base_tau > 0.0:
            self.actuator_tau_steps = float(max(
                0.0,
                base_tau * (1.0 + float(self.rng.uniform(-self.frac, self.frac))),
            ))
        else:
            self.actuator_tau_steps = 0.0

        # MV dead-time jitter (integer, drawn uniformly in [0, max]).
        dead_cfg_raw = cfg.get('mv_deadtime_max_steps', None) if isinstance(cfg, dict) else None
        if dead_cfg_raw is None and identified_dead_time is not None:
            # Up to 50 % of identified dead-time, at least 1 step so the
            # draw is non-degenerate.
            dead_max = int(max(1, round(0.5 * float(identified_dead_time))))
        else:
            try:
                dead_max = int(dead_cfg_raw) if dead_cfg_raw is not None else 0
            except (TypeError, ValueError):
                dead_max = 0
        if dead_max > 0:
            self.mv_deadtime_steps = int(self.rng.integers(0, dead_max + 1))
        else:
            self.mv_deadtime_steps = 0

        # Per-DV rest-point shift (fraction of DV range, uniform ±shift).
        dv_shift_pct = max(0.0, _cfg_float('dv_mean_shift_pct', 0.0))
        if dv_shift_pct > 0.0 and n_dvs > 0:
            self.dv_mean_shifts_frac = {
                i: float(self.rng.uniform(-dv_shift_pct, dv_shift_pct))
                for i in range(int(n_dvs))
            }
        else:
            self.dv_mean_shifts_frac = {i: 0.0 for i in range(max(0, int(n_dvs)))}

    def apply_output_gain_bias(self, y: float, output_range: float) -> float:
        """Apply sampled gain & bias to a scalar model output.

        ``output_range`` is the engineering-unit span (hi-lo) of the CV;
        the stored bias_frac is scaled by it so the bias magnitude is
        meaningful relative to each plant's output scale.
        """
        if not self.enabled:
            return float(y)
        return float(self.output_gain) * float(y) + float(self.output_bias_frac) * float(output_range)

    def jitter_inputs(self, window: np.ndarray,
                      input_ranges: np.ndarray) -> np.ndarray:
        """Additive Gaussian jitter on a (lookback, n_inputs) array.

        ``input_ranges`` is a 1-D array of per-channel engineering spans;
        jitter std is ``input_jitter_std_frac * range`` independently per
        channel. Noise is redrawn every call (per step).
        """
        if not self.enabled or self.input_jitter_std_frac <= 0.0:
            return window
        ranges = np.asarray(input_ranges, dtype=np.float64).reshape(-1)
        if ranges.size != window.shape[-1]:
            return window
        noise = self.rng.standard_normal(size=window.shape).astype(window.dtype)
        scale = (self.input_jitter_std_frac * ranges).astype(window.dtype)
        return window + noise * scale


# ---------------------------------------------------------------------------
# Disturbance offset mixin
# ---------------------------------------------------------------------------

class DisturbanceOffsetMixin:
    """Mixin providing persistent disturbance-offset interface.

    Simulators that inherit from (or compose) this mixin expose
    ``set_disturbance_offset`` and ``reset_disturbance_offsets``, consumed
    by the disturbance curriculum in ``control_training_disturbance.py`` and
    ``control_validate_latent.py``.

    The simulator must also expose ``cv_indices`` and ``dv_indices`` lists.

    Subclasses whose physics already read ``self._cv_offsets`` each step
    (so that unmeasured-CV disturbances produce a sustained, dynamics-
    respecting response) should override
    ``honors_cv_disturbance_offsets = True``.  Simulators that recompute
    CVs from an external black-box (e.g. an ONNX soft-sensor) and cannot
    inject the offset into their recomputation must leave the default
    ``False`` so the validation/training disturbance layer knows to
    re-assert the injected CV value for a hold horizon.
    """

    # Default: assume the simulator does NOT apply `_cv_offsets` inside
    # its physics; the disturbance layer must hold the CV value manually
    # after each step.  Override to True in subclasses where
    # `_cv_offsets` is consumed inside `step()`.
    honors_cv_disturbance_offsets: bool = False

    def _init_disturbance_offsets(self):
        self._dv_offsets: Dict[int, float] = {}
        self._cv_offsets: Dict[int, float] = {}

    def set_disturbance_offset(self, group: str, pos: int, delta: float):
        """Set an absolute persistent offset for a DV or CV channel.

        The offset replaces any previous value for the same channel (it is
        **not** cumulative).  The caller is responsible for computing the
        desired total offset relative to the channel's baseline.

        Args:
            group: ``'dv'`` or ``'cv'``.
            pos: Position within the group (0-based).
            delta: Engineering-unit total offset from baseline.
        """
        if group == 'dv':
            indices = getattr(self, 'dv_indices', [])
        elif group == 'cv':
            indices = getattr(self, 'cv_indices', [])
        else:
            return
        if 0 <= pos < len(indices):
            idx = int(indices[pos])
            bucket = self._dv_offsets if group == 'dv' else self._cv_offsets
            bucket[idx] = float(delta)

    def get_disturbance_offset(self, group: str, pos: int) -> float:
        """Return the current absolute offset for a DV or CV channel."""
        if group == 'dv':
            indices = getattr(self, 'dv_indices', [])
            bucket = self._dv_offsets
        elif group == 'cv':
            indices = getattr(self, 'cv_indices', [])
            bucket = self._cv_offsets
        else:
            return 0.0
        if 0 <= pos < len(indices):
            return float(bucket.get(int(indices[pos]), 0.0))
        return 0.0

    def reset_disturbance_offsets(self):
        """Clear all disturbance offsets (called automatically on reset)."""
        self._dv_offsets.clear()
        self._cv_offsets.clear()


# ---------------------------------------------------------------------------
# Generic noise wrapper
# ---------------------------------------------------------------------------

class _OUSource:
    """Internal: one OU noise source bound to a state-variable index."""
    __slots__ = ('index', 'base_gain', 'gain', 'lo', 'hi', 'ou')

    def __init__(self, index: int, sigma: float, gain: float,
                 bounds: Tuple[float, float],
                 theta: float = 0.15, dt: float = 0.01,
                 rng: Optional[np.random.Generator] = None):
        self.index = int(index)
        self.base_gain = float(gain)
        self.gain = self.base_gain
        self.lo, self.hi = float(bounds[0]), float(bounds[1])
        self.ou = OUActionNoise(
            mean=np.zeros(1),
            std_deviation=np.array([float(sigma)]),
            theta=theta,
            dt=dt,
            rng=rng,
        )

    def sample(self) -> float:
        return float(self.ou()[0]) * self.gain

    def reset(self):
        self.ou.reset()


class _MeasNoise:
    """Internal: white noise source bound to a state-variable index."""
    __slots__ = ('index', 'base_sigma', 'sigma', 'lo', 'hi')

    def __init__(self, index: int, sigma: float,
                 bounds: Tuple[float, float]):
        self.index = int(index)
        self.base_sigma = float(sigma)
        self.sigma = self.base_sigma
        self.lo, self.hi = float(bounds[0]), float(bounds[1])


class SimNoiseWrapper:
    """Transparent wrapper adding generic noise after each ``sim.step()``.

    The wrapper reads noise configuration from (in priority order):

    1. An explicit ``noise_config`` dict passed to the constructor.
    2. The ``SIM_NOISE_CONFIG_JSON`` environment variable (path to JSON).
    3. ``sim.noise_config`` (legacy: simulator-provided config).

    ``step()`` delegates to the underlying sim, then applies noise and
    writes back to ``sim.episode_array``.  All other attribute access is
    proxied to the underlying sim, so callers need no changes.

    If no noise configuration is found via any source the wrapper is a
    transparent no-op.

    Per-episode noise amplitude jitter
    -----------------------------------
    On each ``reset()`` the wrapper multiplies every OU gain and measurement
    sigma by a uniform random factor in ``[1 - noise_jitter_pct,
    1 + noise_jitter_pct]``.  This prevents the agent from overfitting to a
    single noise profile.  Controlled by the ``SIM_NOISE_JITTER_PCT`` env
    var (default **0.20**, i.e. ±20 %).

    Domain-randomization passthrough
    ---------------------------------
    When the external noise config contains a ``domain_randomization`` key
    and the underlying sim has a ``_randomizer`` attribute, the wrapper
    updates the randomizer's ``frac`` (±%) on construction so the sim's own
    ``_sample_episode_dynamics`` uses the dynamics-derived range.
    """

    def __init__(self, sim, noise_config: Optional[Dict] = None,
                 noise_seed: Optional[int] = None):
        self._sim = sim

        # --- Resolve noise config (external → env → sim-provided) ---------
        cfg = noise_config
        if cfg is None:
            env_path = os.environ.get('SIM_NOISE_CONFIG_JSON', '').strip()
            if env_path and os.path.isfile(env_path):
                try:
                    import json as _json
                    with open(env_path, 'r', encoding='utf-8') as _f:
                        cfg = _json.load(_f)
                except Exception:
                    cfg = None
        if cfg is None:
            cfg = getattr(sim, 'noise_config', None) or {}

        # --- Local RNG for all noise operations ---------------------------
        seed_str = os.environ.get('SIM_NOISE_SEED', '').strip()
        if noise_seed is not None:
            self._rng = np.random.default_rng(int(noise_seed))
        elif seed_str:
            self._rng = np.random.default_rng(int(seed_str))
        else:
            self._rng = np.random.default_rng()

        # --- Noise amplitude jitter per episode ---------------------------
        jitter_str = os.environ.get('SIM_NOISE_JITTER_PCT', '').strip()
        self._noise_jitter_pct = float(jitter_str) if jitter_str else 0.20
        self._noise_jitter_pct = float(np.clip(self._noise_jitter_pct, 0.0, 0.5))

        self._ou_sources: List[_OUSource] = []
        for entry in cfg.get('ou_noise', []):
            self._ou_sources.append(_OUSource(
                index=entry['index'],
                sigma=entry.get('sigma', 0.01),
                gain=entry.get('gain', 1.0),
                bounds=entry.get('bounds', (-1e9, 1e9)),
                theta=entry.get('theta', 0.15),
                dt=entry.get('dt', 0.01),
                rng=self._rng,
            ))

        self._meas_noise: List[_MeasNoise] = []
        for entry in cfg.get('measurement_noise', []):
            self._meas_noise.append(_MeasNoise(
                index=entry['index'],
                sigma=entry.get('sigma', 0.01),
                bounds=entry.get('bounds', (-1e9, 1e9)),
            ))

        self._has_noise = bool(self._ou_sources or self._meas_noise)

        # Global per-episode noise amplitude scale (P89, 2026-06-06).
        # Multiplies BOTH OU process noise and white measurement noise on
        # every step.  Set to 0.0 to make an episode fully noise-free (the
        # clean held-action steady-state seeds need this so the WM gets pure
        # fixed-point supervision), or to a curriculum value in (0, 1] to ramp
        # noise in over P1 (``noise_curriculum_scale``).  Default 1.0 = full
        # configured noise (legacy behaviour).
        self._noise_scale: float = 1.0

        # --- Apply domain-randomization % to sim's randomizer -------------
        dr_cfg = cfg.get('domain_randomization')
        if isinstance(dr_cfg, dict):
            randomizer = getattr(sim, '_randomizer', None)
            if randomizer is not None and hasattr(randomizer, 'frac'):
                if dr_cfg.get('enabled', True):
                    randomizer.enabled = True
                    randomizer.frac = float(np.clip(
                        float(dr_cfg.get('param_randomization_pct', randomizer.frac)),
                        0.0, 0.5,
                    ))
                else:
                    randomizer.enabled = False

    # -- Intercepted methods -----------------------------------------------

    def set_noise_scale(self, scale: float) -> None:
        """Set the global noise amplitude multiplier for subsequent steps.

        ``scale=0.0`` → fully noise-free (clean steady-state seeds);
        ``scale=1.0`` → full configured noise.  Values in between ramp the
        process-OU + measurement noise together (P1 noise curriculum).
        Persists until changed; ``reset()`` does NOT clear it (the caller
        — APCEnv.reset — re-sets it every episode), so a value set after a
        reset stays in force for that episode.
        """
        try:
            self._noise_scale = float(max(0.0, scale))
        except Exception:
            self._noise_scale = 1.0

    def _jitter_noise_amplitudes(self):
        """Randomize noise gains/sigmas within ±jitter_pct of base values."""
        pct = self._noise_jitter_pct
        if pct <= 0.0:
            return
        for ou in self._ou_sources:
            ou.gain = float(ou.base_gain * self._rng.uniform(1.0 - pct, 1.0 + pct))
        for mn in self._meas_noise:
            mn.sigma = float(mn.base_sigma * self._rng.uniform(1.0 - pct, 1.0 + pct))

    def step(self, action):
        state, done = self._sim.step(action)
        scale = self._noise_scale
        if self._has_noise and not done and scale > 0.0:
            for ou in self._ou_sources:
                state[ou.index] = float(np.clip(
                    float(state[ou.index]) + ou.sample() * scale,
                    ou.lo, ou.hi,
                ))
            for mn in self._meas_noise:
                state[mn.index] = float(np.clip(
                    float(state[mn.index])
                    + self._rng.standard_normal() * mn.sigma * scale,
                    mn.lo, mn.hi,
                ))
            if hasattr(self._sim, 'episode_array'):
                self._sim.episode_array[self._sim.episode_counter] = state
        return state, done

    def reset(self):
        result = self._sim.reset()
        for ou in self._ou_sources:
            ou.reset()
        self._jitter_noise_amplitudes()
        return result

    # -- Transparent proxy -------------------------------------------------

    def __getattr__(self, name):
        return getattr(self._sim, name)

    def __repr__(self):
        return f'SimNoiseWrapper({self._sim!r})'
