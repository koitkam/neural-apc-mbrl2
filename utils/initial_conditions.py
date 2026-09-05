"""Per-episode initial-condition randomization for plant simulators.

Until 2026-05-21 every simulator's ``reset()`` started PVs/MVs/DVs from a
narrow Gaussian (σ ≈ 0.7–4.0) around a single fixed nominal operating
point.  The world model therefore only ever saw t=0 state distributions
near one location in state-space and extrapolated badly when the trained
controller arrived at a different operating point at deployment.  The
diagnostic verdict on p31 (real plant converges 75–88% under held action,
WM converges 0%) confirmed the WM was generalising poorly outside the
seed distribution.

This helper replaces the narrow Gaussian with a wide *uniform* draw
centred on the same nominal but covering a configurable fraction of the
variable's bounded operating range.  It is sim-adaptive by construction:
each simulator already knows its own ``(lo, hi)`` bounds for every
variable, so this helper only needs the bounds and the legacy nominal /
σ — no per-sim configuration is required.

Env vars / TrainConfig
----------------------
* ``DREAMER_INIT_RANDOMIZATION`` / ``TrainConfig.init_randomization`` —
  master switch.  Default ON.  Set to ``0`` to restore the legacy
  narrow-Gaussian behaviour.
* ``DREAMER_INIT_RANDOMIZATION_FRAC`` / ``TrainConfig.init_randomization_frac``
  — fraction of the bounded range used for the wide uniform draw.
  Default ``0.6`` = uniform over a 60% slice of ``(hi - lo)`` centred
  on the legacy nominal.

Simulators have no cfg at ``reset()``.  ``ic_randomization_knobs()``
reads TrainConfig defaults then leftover env (identity ON / 0.6).
ENV_OVERRIDES records the same keys in ``run_plan``.  Env is read
fresh every call so tests can flip it without restarting.
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import numpy as np

_IC_OFF = ('0', 'false', 'no', 'off')
_TC_IC: Optional[Tuple[bool, float]] = None


def ic_randomization_knobs() -> Tuple[bool, float]:
    """Master switch / span fraction for sim ``reset()`` IC draws.

    Simulators have no ``TrainConfig`` at ``reset()``, so this used to
    hard-code leftover-env fallbacks ``1`` / ``0.6``.  Identity for
    env-free: those numbers **are** the TrainConfig defaults.  Changing
    the dataclass now actually widens/narrows the IC draw.  Leftover
    ``DREAMER_INIT_RANDOMIZATION`` / ``DREAMER_INIT_RANDOMIZATION_FRAC``
    still win when set (same keys as ``ENV_OVERRIDES``).  Env is read
    fresh every call so tests can flip it without restarting.
    """
    global _TC_IC
    if _TC_IC is None:
        enabled, frac = True, 0.6
        # Do not import training.train here: plant ID calls reset()
        # before single_run imports TrainConfig.  Once train.py is in
        # sys.modules, dataclass defaults win over the hardcoded 1/0.6.
        import sys
        mod = sys.modules.get('training.train')
        cfg_cls = getattr(mod, 'TrainConfig', None) if mod is not None else None
        if cfg_cls is not None:
            cfg = cfg_cls()
            enabled = bool(getattr(cfg, 'init_randomization', True))
            frac = float(getattr(cfg, 'init_randomization_frac', 0.6) or 0.6)
            _TC_IC = (enabled, frac)
        else:
            enabled, frac = True, 0.6
    else:
        enabled, frac = _TC_IC
    raw = os.environ.get('DREAMER_INIT_RANDOMIZATION', '').strip()
    if raw:
        enabled = raw.lower() not in _IC_OFF
    raw = os.environ.get('DREAMER_INIT_RANDOMIZATION_FRAC', '').strip()
    if raw:
        try:
            frac = float(raw)
        except ValueError:
            pass
    return bool(enabled), float(np.clip(frac, 0.05, 0.95))


def _enabled() -> bool:
    return ic_randomization_knobs()[0]


def _frac() -> float:
    return ic_randomization_knobs()[1]


def sample_initial_value(
    rng: np.random.Generator,
    *,
    nominal: float,
    bounds,
    legacy_sigma: float,
) -> float:
    """Sample one initial PV / MV / DV value.

    When ``DREAMER_INIT_RANDOMIZATION=1`` (default), draws uniformly from
    ``[nominal - half, nominal + half]`` where
    ``half = 0.5 * frac * (hi - lo)``, then clips to ``bounds``.  This
    keeps the legacy nominal as the centre of the distribution while
    widening the spread enough to give the WM real coverage of the
    operating envelope.

    When disabled, falls back to ``nominal + rng.standard_normal() *
    legacy_sigma`` clipped to ``bounds`` (the legacy behaviour every sim
    used before 2026-05-21).
    """
    lo = float(bounds[0])
    hi = float(bounds[1])
    if not _enabled():
        return float(np.clip(
            nominal + rng.standard_normal() * float(legacy_sigma),
            lo, hi,
        ))
    half = 0.5 * _frac() * (hi - lo)
    val = float(rng.uniform(nominal - half, nominal + half))
    return float(np.clip(val, lo, hi))
