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

Env vars
--------
* ``DREAMER_INIT_RANDOMIZATION`` — master switch.  Default ``1`` (on).
  Set to ``0`` to restore the legacy narrow-Gaussian behaviour.
* ``DREAMER_INIT_RANDOMIZATION_FRAC`` — fraction of the bounded range
  used for the wide uniform draw.  Default ``0.6`` = uniform over a
  60% slice of ``(hi - lo)`` centred on the legacy nominal.

Both are read fresh on every call so test code can flip them without
restarting the process.
"""

from __future__ import annotations

import os

import numpy as np


def _enabled() -> bool:
    v = str(os.environ.get('DREAMER_INIT_RANDOMIZATION', '1')).strip().lower()
    return v not in {'0', 'false', 'no', 'off'}


def _frac() -> float:
    try:
        f = float(os.environ.get('DREAMER_INIT_RANDOMIZATION_FRAC', '0.6'))
    except Exception:
        f = 0.6
    return float(np.clip(f, 0.05, 0.95))


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
