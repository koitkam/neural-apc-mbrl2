"""ONNX-based soft sensor simulator for neural APC training.

Wraps a pre-trained soft sensor ONNX model as a gym-like environment.
The agent manipulates ACY_FIC631_SP (flow setpoint) to drive the
predicted lab value ACYNMP_1259_LV toward a target.

The remaining 4 model inputs are treated as disturbance variables that
evolve via mean-reverting (Ornstein–Uhlenbeck) random walks based on
historical operating statistics.

State vector layout (7 elements):
  [0] LAB_PV       – soft sensor predicted lab value  (CV)
  [1] LAB_SV       – lab target setpoint              (SV, constant)
  [2] FIC631_SP_MV – flow 631 setpoint                (MV)
  [3] FZR602A_DV   – DV: freezer A output
  [4] FIC660_DV    – DV: flow 660
  [5] FIC638_OUT_DV– DV: flow 638 output
  [6] TIC631_MV_DV – DV: temperature 631 measurement

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
- For this ONNX-backed simulator the ranges reflect the model's training
  data distribution and should not be widened beyond it (the soft sensor
  cannot extrapolate).
"""

import json
import os

import numpy as np

from utils.sim_noise import DisturbanceOffsetMixin, DomainRandomizer

# ---------------------------------------------------------------------------
# Historical operating statistics (from Foxboro DCS data 2025-01-01 → 2026-04-14)
# ---------------------------------------------------------------------------
_DV_STATS = {
    # tag_name: (mean, std, min, max)
    'ACY_FZR602A_MV': (2.1429, 0.7467, -0.0076, 3.6466),
    'ACY_FIC660_MV':  (0.1150, 0.1337, -0.0380, 0.5812),
    'ACY_FIC638_OUT': (66.911, 15.770, 46.290, 100.00),
    'ACY_TIC631_MV':  (147.316, 13.257, 119.30, 182.03),
}

# Soft sensor normalization (from metadata.json)
_NORM = {
    'ACY_FIC631_SP':  (0.0, 387.705719),
    'ACY_FZR602A_MV': (-0.007647828, 3.646620035),
    'ACY_FIC660_MV':  (-0.038005721, 0.581155062),
    'ACY_FIC638_OUT': (46.29022217, 100.0),
    'ACY_TIC631_MV':  (119.29788903545455, 182.0320011394366),
}
_OUT_NORM = (5.0, 75.0)  # (min, max) for ACYNMP_1259_LV

# Input order expected by the ONNX model
_INPUT_ORDER = [
    'ACY_FIC631_SP',
    'ACY_FZR602A_MV',
    'ACY_FIC660_MV',
    'ACY_FIC638_OUT',
    'ACY_TIC631_MV',
]

_MV_TAG = 'ACY_FIC631_SP'
_DV_TAGS = [t for t in _INPUT_ORDER if t != _MV_TAG]


def _ensure_nvidia_libs_on_path() -> None:
    """Preload NVIDIA runtime libs so onnxruntime's CUDA EP can find them.

    ``onnxruntime-gpu`` looks up ``libcudnn.so.9`` etc via the dynamic
    loader. Our venv ships these under ``nvidia/<pkg>/lib`` (pip wheels)
    but nothing adds them to ``LD_LIBRARY_PATH``. Setting the env var at
    runtime is too late (the loader has already cached it), so we instead
    ``dlopen`` each lib with ``RTLD_GLOBAL`` — symbols then become visible
    to subsequent ``dlopen`` calls by onnxruntime's provider shim.
    """
    import importlib.util as _ilu
    import ctypes as _ct

    # Order matters: cudnn_graph depends on cudnn_ops which depends on cudnn,
    # cublas depends on cublasLt, etc. Load leaf deps before their dependents.
    # Missing libs are silently skipped (env without GPU wheels just falls
    # back to CPU provider).
    lib_order = [
        ('nvidia.cuda_runtime', ['libcudart.so.12']),
        ('nvidia.cublas', ['libcublasLt.so.12', 'libcublas.so.12']),
        ('nvidia.cuda_nvrtc', ['libnvrtc.so.12']),
        ('nvidia.cudnn', [
            'libcudnn.so.9',
            'libcudnn_graph.so.9',
            'libcudnn_ops.so.9',
            'libcudnn_adv.so.9',
            'libcudnn_cnn.so.9',
            'libcudnn_engines_precompiled.so.9',
            'libcudnn_engines_runtime_compiled.so.9',
            'libcudnn_heuristic.so.9',
        ]),
    ]
    for pkg, libs in lib_order:
        try:
            spec = _ilu.find_spec(pkg)
        except Exception:
            spec = None
        if spec is None or not spec.submodule_search_locations:
            continue
        lib_dir = os.path.join(list(spec.submodule_search_locations)[0], 'lib')
        for name in libs:
            path = os.path.join(lib_dir, name)
            if not os.path.isfile(path):
                continue
            try:
                _ct.CDLL(path, mode=_ct.RTLD_GLOBAL)
            except OSError:
                # A single failure is non-fatal: CUDA EP creation will warn
                # and we fall back to CPU below.
                pass


def _load_onnx_session(onnx_path: str):
    """Lazy-load ONNX Runtime inference session.

    Prefer CUDA execution provider when available (onnxruntime-gpu installed
    and CUDA device present); fall back to CPU when unavailable. Per-step
    ONNX inference is the dominant simulator cost, so running on GPU frees
    the CPU bottleneck during SAC rollouts.
    """
    _ensure_nvidia_libs_on_path()
    import onnxruntime as ort
    opts = ort.SessionOptions()
    opts.inter_op_num_threads = 1
    opts.intra_op_num_threads = 1
    # Honour explicit override for debugging; otherwise try CUDA then CPU.
    force_cpu = os.environ.get('SOFTSENSOR_ONNX_CPU', '').strip().lower() in {'1', 'true', 'yes', 'on'}
    if force_cpu:
        providers = ['CPUExecutionProvider']
    else:
        available = set(ort.get_available_providers())
        providers = []
        if 'CUDAExecutionProvider' in available:
            providers.append('CUDAExecutionProvider')
        providers.append('CPUExecutionProvider')
    return ort.InferenceSession(onnx_path, opts, providers=providers)


class SoftSensorLabSim(DisturbanceOffsetMixin):
    """Soft sensor plant model for APC agent training.

    The ONNX model expects shape (1, lookback_steps, 5) — a window of
    normalised input history.  Each call to ``step()`` pushes one new
    row into the window, runs inference, and returns the predicted lab
    value as the CV.
    """

    def __init__(
        self,
        episode_length: int = 1000,
        sample_rate: int = 1,
        noise_stdv: float = 0.02,
        lab_target: float = 33.0,
        onnx_path: str | None = None,
        lookback_steps: int = 45,
        dv_tau: float = 40.0,
        domain_randomization=None,
        param_randomization_pct=None,
        randomization_seed=None,
        **kwargs,
    ):
        self.sample_rate = int(sample_rate)
        self.episode_length = int(episode_length)
        self.episode_counter = 0
        self.done = False
        self.lookback_steps = int(lookback_steps)
        self.lab_target = float(lab_target)
        self.dv_tau = float(dv_tau)

        # State vector: CV, SV, MV, 4×DV
        self.state_variables = [
            'LAB_PV',
            'LAB_SV',
            'FIC631_SP_MV',
            'FZR602A_DV',
            'FIC660_DV',
            'FIC638_OUT_DV',
            'TIC631_MV_DV',
        ]

        self.mv_indices = [2]       # FIC631_SP_MV
        self.cv_indices = [0]       # LAB_PV
        self.dv_indices = [3, 4, 5, 6]

        self.state_is_normalized = False

        self.episode_array = np.zeros(
            (self.episode_length + 1, len(self.state_variables)),
            dtype='float32',
        )

        self.mv_limits = (0.0, 387.7)
        self.lab_limits = (5.0, 75.0)

        # Normalization ranges exposed as single source of truth for sim_factory.
        self.mv_normalization_ranges = [[0.0, 387.7]]
        self.cv_normalization_ranges = [[5.0, 75.0]]
        self.dv_normalization_ranges = [
            [-0.008, 3.647],
            [-0.038, 0.582],
            [46.29, 100.0],
            [119.30, 182.03],
        ]
        self.state_normalization_ranges = [
            [5.0, 75.0],
            [5.0, 75.0],
            [0.0, 387.7],
            [-0.008, 3.647],
            [-0.038, 0.582],
            [46.29, 100.0],
            [119.30, 182.03],
        ]

        # --- Domain randomizer ---
        self._randomizer = DomainRandomizer(
            env_prefixes=['SIM'],
            domain_randomization=domain_randomization,
            param_randomization_pct=param_randomization_pct,
            randomization_seed=randomization_seed,
        )

        # --- Load ONNX model ---
        if onnx_path is None:
            sim_dir = os.path.dirname(os.path.abspath(__file__))
            onnx_path = os.path.join(sim_dir, 'softsensor.onnx')
        self._session = _load_onnx_session(onnx_path)
        self._input_name = self._session.get_inputs()[0].name
        self._output_name = self._session.get_outputs()[0].name

        # Raw (un-normalised) input history buffer: (lookback_steps, 5)
        self._raw_history = np.zeros(
            (self.lookback_steps, len(_INPUT_ORDER)),
            dtype='float32',
        )

        # --- Disturbance offsets (from mixin) ---
        self._init_disturbance_offsets()

        # Per-input engineering spans used by input-jitter in the ONNX
        # input window. Order matches ``_INPUT_ORDER``.
        self._input_spans = np.asarray(
            [float(_NORM[tag][1] - _NORM[tag][0]) for tag in _INPUT_ORDER],
            dtype='float64',
        )
        # Actuator lag + MV dead-time buffer state (populated in reset()).
        self._mv_actual = 0.0
        self._mv_cmd_history: list[float] = []

    # ----- helpers -----

    def _normalise_history(self) -> np.ndarray:
        """Min-max normalise the raw history buffer → (1, lookback, 5).

        When :attr:`_randomizer.input_jitter_std_frac` > 0 a per-step
        Gaussian jitter (fraction of each input's engineering-unit range)
        is added to the raw window *before* normalisation. This emulates
        sensor noise the softsensor ONNX model was never trained on.
        """
        raw = self._raw_history
        if getattr(self._randomizer, 'input_jitter_std_frac', 0.0) > 0.0:
            raw = self._randomizer.jitter_inputs(raw.copy(), self._input_spans)
        normed = np.zeros_like(raw)
        for i, tag in enumerate(_INPUT_ORDER):
            lo, hi = _NORM[tag]
            span = hi - lo if hi != lo else 1.0
            normed[:, i] = (raw[:, i] - lo) / span
        return normed[np.newaxis, :, :]  # (1, lookback, 7)

    def _predict_lab(self) -> float:
        """Run ONNX inference and return denormalised lab value.

        Applies per-episode output gain & bias (from DomainRandomizer)
        to emulate plant-gain drift and lab-calibration offset without
        retraining the frozen ONNX predictor.
        """
        inp = self._normalise_history()
        out_norm = self._session.run(
            [self._output_name], {self._input_name: inp}
        )[0]
        # Denormalise: output was min-max scaled to [0, 1]
        val = float(out_norm.flat[0]) * (_OUT_NORM[1] - _OUT_NORM[0]) + _OUT_NORM[0]
        val = self._randomizer.apply_output_gain_bias(
            val, output_range=(_OUT_NORM[1] - _OUT_NORM[0]),
        )
        return float(np.clip(val, self.lab_limits[0], self.lab_limits[1]))

    def _init_dv_values(self, rng: np.random.Generator) -> dict[str, float]:
        """Sample initial DV values near their historical means."""
        vals = {}
        for tag in _DV_TAGS:
            mean, std, lo, hi = _DV_STATS[tag]
            v = float(np.clip(mean + rng.standard_normal() * std * 0.3, lo, hi))
            vals[tag] = v
        return vals

    def _step_dv(self, tag: str, current: float, rng: np.random.Generator) -> float:
        """One OU step for a DV: mean-revert + noise.

        The rest point is the historical mean plus (a) the per-episode
        mean shift sampled by DomainRandomizer — emulates operating-point
        drift between commissioning and deployment — and (b) any explicit
        disturbance offset set by the curriculum module.
        """
        mean, std, lo, hi = _DV_STATS[tag]
        dv_pos = _DV_TAGS.index(tag)
        dv_range = float(hi - lo)
        mean_shift = float(self._randomizer.dv_mean_shifts_frac.get(dv_pos, 0.0)) * dv_range
        ref = mean + mean_shift + self._dv_offsets.get(
            dv_pos + 3, 0.0  # DV state indices start at 3
        )
        alpha = self.sample_rate / max(self.dv_tau, float(self.sample_rate))
        # Stochastic OU kick honours SIM_NOISE_ENABLED so the dynamics
        # identifier (which sets it to '0') sees a deterministic plant while
        # training keeps the realistic stochastic envelope.
        noise_on = str(os.environ.get('SIM_NOISE_ENABLED', '1')).strip().lower() not in {
            '0', 'false', 'no', 'off',
        }
        noise = rng.standard_normal() * std * 0.05 if noise_on else 0.0
        new = float(current + alpha * (ref - current) + noise)
        return float(np.clip(new, lo, hi))

    # ----- gym-like interface -----

    def reset(self):
        self.episode_counter = 0
        self.done = False
        self.reset_disturbance_offsets()

        # Refresh per-episode randomised plant-wrapper parameters
        # (output gain/bias, input jitter std, actuator lag, MV dead-time,
        # DV mean-shift). All become neutral when domain randomisation is
        # disabled via control_setup or SIM_DOMAIN_RANDOMIZATION=0.
        self._randomizer.sample_episode(n_dvs=len(_DV_TAGS))

        rng = self._randomizer.rng

        # Initial MV (FIC631_SP: mean=75.6, std=80.5 — use tighter init)
        mv_init = float(np.clip(
            75.6 + rng.standard_normal() * 10.0,
            self.mv_limits[0], self.mv_limits[1],
        ))

        # Actuator lag + dead-time buffers start at the initial MV so
        # the history window is steady-state.
        self._mv_actual = float(mv_init)
        delay_len = int(self._randomizer.mv_deadtime_steps) + 1
        self._mv_cmd_history = [float(mv_init)] * delay_len

        # Initial DVs
        dv_vals = self._init_dv_values(rng)

        # Fill history buffer with steady-state-ish values
        for t in range(self.lookback_steps):
            row = np.zeros(len(_INPUT_ORDER), dtype='float32')
            for i, tag in enumerate(_INPUT_ORDER):
                if tag == _MV_TAG:
                    row[i] = mv_init
                else:
                    row[i] = dv_vals[tag]
            self._raw_history[t] = row

        # Run prediction for initial lab PV
        lab_pv = self._predict_lab()

        # Build state vector
        x0 = np.array([
            lab_pv,
            self.lab_target,
            mv_init,
            dv_vals['ACY_FZR602A_MV'],
            dv_vals['ACY_FIC660_MV'],
            dv_vals['ACY_FIC638_OUT'],
            dv_vals['ACY_TIC631_MV'],
        ], dtype='float32')

        self.episode_array[:] = 0.0
        self.episode_array[0] = x0
        return x0.copy(), self.done

    def step(self, action):
        if np.isscalar(action):
            action = np.array([action], dtype='float32')
        else:
            action = np.asarray(action, dtype='float32').reshape(-1)

        if action.shape[0] != 1:
            raise ValueError('Action must contain exactly 1 value: FIC631_SP')

        rng = self._randomizer.rng
        prev = self.episode_array[self.episode_counter]

        # Clip MV command
        mv_cmd = float(np.clip(action[0], self.mv_limits[0], self.mv_limits[1]))

        # Pass the command through the randomised MV dead-time buffer so
        # the value that reaches the actuator is delayed by 0..mv_dt_max
        # steps (emulates transport/communication delay on the real unit).
        self._mv_cmd_history.append(mv_cmd)
        delay = int(self._randomizer.mv_deadtime_steps)
        if len(self._mv_cmd_history) > delay + 1:
            self._mv_cmd_history = self._mv_cmd_history[-(delay + 1):]
        delayed_cmd = float(self._mv_cmd_history[0])

        # Apply first-order actuator lag: mv_actual = mv_actual + alpha * (delayed_cmd - mv_actual)
        actuator_tau = float(self._randomizer.actuator_tau_steps)
        if actuator_tau > 0.0:
            alpha_u = self.sample_rate / max(actuator_tau, float(self.sample_rate))
            self._mv_actual = float(
                self._mv_actual + alpha_u * (delayed_cmd - self._mv_actual)
            )
        else:
            self._mv_actual = delayed_cmd
        mv_effective = float(np.clip(
            self._mv_actual, self.mv_limits[0], self.mv_limits[1],
        ))

        # Step DVs (OU dynamics)
        dv_new = {}
        for idx, tag in enumerate(_DV_TAGS):
            dv_new[tag] = self._step_dv(tag, float(prev[3 + idx]), rng)

        # Build new raw input row using the post-actuator MV so the ONNX
        # window sees the same value the "real" valve is delivering.
        new_row = np.zeros(len(_INPUT_ORDER), dtype='float32')
        for i, tag in enumerate(_INPUT_ORDER):
            if tag == _MV_TAG:
                new_row[i] = mv_effective
            else:
                new_row[i] = dv_new[tag]

        # Shift history and append
        self._raw_history = np.roll(self._raw_history, -1, axis=0)
        self._raw_history[-1] = new_row

        # Predict lab value
        lab_pv = self._predict_lab()

        # Advance — the reported MV in state is the effective (post-lag
        # and post-dead-time) value so the observer sees what the plant
        # actually receives, consistent with the ONNX input window.
        self.episode_counter += 1
        state = np.array([
            lab_pv,
            self.lab_target,
            mv_effective,
            dv_new['ACY_FZR602A_MV'],
            dv_new['ACY_FIC660_MV'],
            dv_new['ACY_FIC638_OUT'],
            dv_new['ACY_TIC631_MV'],
        ], dtype='float32')
        self.episode_array[self.episode_counter] = state

        if self.episode_counter >= self.episode_length:
            self.done = True

        return state.copy(), self.done
