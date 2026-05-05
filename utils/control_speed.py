"""Control-speed preset → numerical factors used by the auto-weight
derivation and hyperparameter initialization.

The preset lives in ``control_objective.json`` under the top-level key
``control_speed`` (values: ``aggressive`` | ``normal`` | ``slow``).  A single
preset drives:

* per-MV move penalty magnitude (slow → high move penalty, aggressive → low)
* observer/agent learning-rate scaling (slow → lower LR)
* disturbance curriculum aggressiveness

All consumers should read through :func:`get_control_speed_factors` so the
mapping stays in one place.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class ControlSpeedFactors:
    """Numerical factors derived from a control-speed preset."""

    name: str
    move_penalty_scale: float        # multiplies the per-MV move penalty
    agent_lr_scale: float            # multiplies the default agent LR
    exploration_scale: float         # multiplies entropy / initial alpha
    disturbance_intensity_scale: float


_PRESETS: Dict[str, ControlSpeedFactors] = {
    'aggressive': ControlSpeedFactors(
        name='aggressive',
        move_penalty_scale=0.2,
        agent_lr_scale=1.3,
        exploration_scale=1.2,
        disturbance_intensity_scale=1.1,
    ),
    'normal': ControlSpeedFactors(
        name='normal',
        move_penalty_scale=1.0,
        agent_lr_scale=1.0,
        exploration_scale=1.0,
        disturbance_intensity_scale=1.0,
    ),
    'slow': ControlSpeedFactors(
        name='slow',
        move_penalty_scale=4.0,
        agent_lr_scale=0.7,
        exploration_scale=0.9,
        disturbance_intensity_scale=0.85,
    ),
}


def get_control_speed_factors(name: str | None = None) -> ControlSpeedFactors:
    """Return the factor bundle for a preset name.

    Falls back to ``normal`` for anything unrecognised.  Env var
    ``CONTROL_SPEED`` takes precedence when ``name`` is ``None``.
    """
    if name is None:
        name = os.environ.get('CONTROL_SPEED', '').strip() or None
    key = str(name or 'normal').strip().lower()
    return _PRESETS.get(key, _PRESETS['normal'])


def list_presets() -> Dict[str, ControlSpeedFactors]:
    """Return a copy of the preset table (for diagnostic/report use)."""
    return dict(_PRESETS)
