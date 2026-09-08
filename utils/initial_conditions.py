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

TrainConfig
-----------
* ``TrainConfig.init_randomization`` — master switch.  Default ON.
  ``DREAMER_INIT_RANDOMIZATION=0`` at launch still opts out via
  ``ENV_OVERRIDES`` then ``bind_ic_randomization_from_cfg``.
* ``TrainConfig.init_randomization_frac`` — fraction of the bounded
  range used for the wide uniform draw.  Default ``0.6``.

Simulators have no cfg at ``reset()``.  ``bind_ic_randomization_from_cfg``
pins the live TrainConfig after ``ENV_OVERRIDES``.  Unbound (plant ID
before TrainConfig) uses identity ON / 0.6.  Leftover
``DREAMER_INIT_RANDOMIZATION*`` at every ``reset()`` is **REMOVED**.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

_TC_IC: Optional[Tuple[bool, float]] = None


def _set_ic(enabled: bool, frac: float) -> Tuple[bool, float]:
    global _TC_IC
    pair = (bool(enabled), float(np.clip(frac, 0.05, 0.95)))
    _TC_IC = pair
    return pair


def bind_ic_randomization_from_cfg(cfg) -> Tuple[bool, float]:
    """Pin sim-reset IC knobs from the live TrainConfig.

    ``reset()`` has no cfg.  A/B is ``ENV_OVERRIDES`` then this bind,
    not leftover ``DREAMER_INIT_RANDOMIZATION*`` on every reset.
    """
    enabled = bool(getattr(cfg, 'init_randomization', True))
    frac = float(getattr(cfg, 'init_randomization_frac', 0.6) or 0.6)
    return _set_ic(enabled, frac)


def reset_ic_randomization_bind() -> None:
    """Drop the pinned pair (smokes).  Next ``ic_randomization_knobs`` re-reads defaults."""
    global _TC_IC
    _TC_IC = None


def ic_randomization_knobs() -> Tuple[bool, float]:
    """Master switch / span fraction for sim ``reset()`` IC draws.

    Simulators have no ``TrainConfig`` at ``reset()``.  After
    ``bind_ic_randomization_from_cfg`` the live cfg wins.  Before that,
    dataclass defaults (identity ON / 0.6) or the same hardcoded pair
    during plant ID (``training.train`` not imported yet).
    """
    global _TC_IC
    if _TC_IC is not None:
        return _TC_IC
    enabled, frac = True, 0.6
    import sys
    mod = sys.modules.get('training.train')
    cfg_cls = getattr(mod, 'TrainConfig', None) if mod is not None else None
    if cfg_cls is not None:
        cfg = cfg_cls()
        enabled = bool(getattr(cfg, 'init_randomization', True))
        frac = float(getattr(cfg, 'init_randomization_frac', 0.6) or 0.6)
    return _set_ic(enabled, frac)


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

    When IC randomization is on (default), draws uniformly from
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
