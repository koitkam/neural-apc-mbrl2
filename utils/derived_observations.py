"""Derived observable features for POMDP belief-state augmentation.

Stage B of the 2026-05-21 robust-controller redesign: even with a hidden
upstream disturbance and a long lookback, the policy still has to infer
"what the unmeasured disturbance is doing" from raw CV history.  The WM
can do this with attention, but the actor-head only sees a single
hidden vector per step, and the value head needs scale-invariant
features to estimate long-horizon return.

We therefore inject three cheap, plant-agnostic derived statistics per
CV channel into the augmented-observation block:

  1. ``int_err``: time-averaged tracking error
     ``mean_window(cv - sp)``, squashed through ``tanh`` so the channel
     is bounded in ``[-1, 1]`` regardless of CV units.  Encodes
     persistent steady-state offset (an "integrator" signature).
  2. ``dcv``: one-step CV difference ``cv(t) - cv(t-1)``, normalized
     by the running standard deviation of CV.  Encodes immediate trend
     direction — what a P-on-derivative controller would see.
  3. ``cv_var``: ``log1p(var_window(cv))``, the within-window variance.
     Encodes "is something perturbing me right now?" — proxy for
     unmeasured-disturbance amplitude estimation.

All three are zero-mean / bounded by construction so the running obs
normalizer in ``APCEnv._update_obs_norm`` does not have to compensate
for arbitrary CV scaling.

Toggle via ``DREAMER_DERIVED_OBSERVABLES=1`` (default OFF — preserves
backward compat with existing checkpoints).  Window size is controlled
by ``DREAMER_DERIVED_OBS_WINDOW`` (default 32 agent steps ≈ τ_dom for
test_sim at sample_rate=4).
"""

from __future__ import annotations

import os
from collections import deque
from typing import Deque, Optional

import numpy as np


def derived_observables_enabled(*, default: bool = True) -> bool:
    """Return True iff the derived-observables block is active.

    Default ON (P37 onward).  Set ``DREAMER_DERIVED_OBSERVABLES=0`` to
    disable for ablations or when loading a checkpoint trained with
    the legacy obs layout (different ``obs_dim``).
    """
    raw = os.environ.get('DREAMER_DERIVED_OBSERVABLES', '').strip()
    if not raw:
        return bool(default)
    return raw not in ('0', 'false', 'False', 'off', 'OFF', 'no')


def derived_observables_window(default: int = 32,
                               *,
                               tau: Optional[float] = None,
                               sample_rate: Optional[float] = None) -> int:
    """Window length (in agent steps) for the rolling derived features.

    Resolution order:
      1. ``DREAMER_DERIVED_OBS_WINDOW`` env var if set (operator override).
      2. ``round(2 * tau / sample_rate)`` clamped to ``[8, 128]`` when
         both ``tau`` and ``sample_rate`` are provided (auto-tune per
         plant — covers ~2 dominant time-constants of context).
      3. Static ``default`` fallback.
    """
    raw = os.environ.get('DREAMER_DERIVED_OBS_WINDOW', '').strip()
    if raw:
        try:
            return max(2, int(raw))
        except Exception:
            pass
    if tau is not None and sample_rate is not None:
        try:
            sr = max(1.0, float(sample_rate))
            auto = int(round(2.0 * float(tau) / sr))
            return int(max(8, min(128, auto)))
        except Exception:
            pass
    return int(default)


class DerivedFeatures:
    """Rolling per-CV derived statistics for belief-state augmentation.

    Use:
        df = DerivedFeatures(n_cv=N, window=32)
        df.reset(cv0)
        # each env step (after sim step but before _build_obs_vec):
        df.update(cv_now, sp_now)
        feats = df.features()   # shape (3 * n_cv,)
    """

    FEATS_PER_CV = 3

    def __init__(self, n_cv: int, window: int = 32) -> None:
        self.n_cv = int(n_cv)
        self.window = max(2, int(window))
        self._cv_buf: Deque[np.ndarray] = deque(maxlen=self.window)
        self._err_buf: Deque[np.ndarray] = deque(maxlen=self.window)
        self._last_cv: Optional[np.ndarray] = None

    @property
    def feat_dim(self) -> int:
        return int(self.n_cv * self.FEATS_PER_CV)

    def reset(self, cv0: Optional[np.ndarray] = None) -> None:
        self._cv_buf.clear()
        self._err_buf.clear()
        if cv0 is not None:
            x = np.asarray(cv0, dtype='float64').reshape(self.n_cv)
            self._cv_buf.append(x)
            self._last_cv = x.copy()
        else:
            self._last_cv = None

    def update(self, cv: np.ndarray, sp: np.ndarray) -> None:
        cv_v = np.asarray(cv, dtype='float64').reshape(self.n_cv)
        sp_v = np.asarray(sp, dtype='float64').reshape(self.n_cv)
        # Replace any NaN setpoints (disabled CV-target slots) with cv
        # itself so the error contribution is zero rather than NaN.
        bad = ~np.isfinite(sp_v)
        if bad.any():
            sp_v = np.where(bad, cv_v, sp_v)
        self._cv_buf.append(cv_v)
        self._err_buf.append(cv_v - sp_v)
        self._last_cv = cv_v.copy()

    def features(self) -> np.ndarray:
        if self.n_cv <= 0:
            return np.zeros(0, dtype='float32')
        out = np.zeros(self.feat_dim, dtype='float32')
        if not self._cv_buf:
            return out
        cv_arr = np.stack(self._cv_buf, axis=0)  # (T, n_cv)
        err_arr = np.stack(self._err_buf, axis=0)  # (T, n_cv)
        # Feature 1: tanh(mean_err / scale_per_cv).  Scale = running
        # std of cv (so tracking error is in plant-natural units).
        cv_std = cv_arr.std(axis=0) + 1e-6
        mean_err = err_arr.mean(axis=0)
        f1 = np.tanh(mean_err / cv_std)
        # Feature 2: one-step difference, normalized by running std.
        if cv_arr.shape[0] >= 2:
            dcv = cv_arr[-1] - cv_arr[-2]
        else:
            dcv = np.zeros(self.n_cv, dtype='float64')
        f2 = np.tanh(dcv / cv_std)
        # Feature 3: log1p(variance) — bounded growth, always ≥ 0.
        cv_var = cv_arr.var(axis=0)
        f3 = np.log1p(np.maximum(cv_var, 0.0)).astype('float64')
        # Interleave per CV: [int_err_cv0, dcv_cv0, var_cv0, int_err_cv1, ...]
        stacked = np.stack([f1, f2, f3], axis=1)  # (n_cv, 3)
        out[:] = stacked.reshape(-1).astype('float32')
        return out
