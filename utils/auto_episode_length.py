"""Derive a reasonable training episode length from identified dynamics.

The goal is to give slow plants (large tau + dead_time) enough samples per
episode to see the step response settle, while keeping fast plants from
wasting training time on padded tails.  Resolution order:

1. Explicit env ``SIM_EPISODE_LENGTH`` (user / BO override) — always wins.
2. Identified dynamics: ``k * (tau + dead_time)`` steps, floored + ceilinged.
3. Fallback: ``default_fallback`` (1000).

All numbers come from :mod:`utils.dynamics_identifier` via the
``IDENTIFIED_TAU_DOMINANT`` / ``IDENTIFIED_DEAD_TIME`` environment variables
that :mod:`workflow.bo_runner` sets after identification.
"""

from __future__ import annotations

import math
import os
from typing import Optional, Tuple


def _safe_float(x, default=0.0):
    try:
        v = float(x)
        return v if math.isfinite(v) else float(default)
    except Exception:
        return float(default)


def derive_episode_length(
    default_fallback: int = 1000,
    settle_multiple: float = 20.0,
    min_length: int = 500,
    max_length: int = 4000,
) -> Tuple[int, str]:
    """Return ``(episode_length, source)``.

    - ``source='env'``: user-set SIM_EPISODE_LENGTH.
    - ``source='auto:{k}x_tau_plus_dt'``: derived from identified dynamics.
    - ``source='default'``: fallback (no identification available).
    """
    env_raw = os.environ.get('SIM_EPISODE_LENGTH', '').strip()
    if env_raw:
        try:
            v = int(float(env_raw))
            if v > 0:
                return v, 'env'
        except Exception:
            pass

    tau = _safe_float(os.environ.get('IDENTIFIED_TAU_DOMINANT', '0'), 0.0)
    dt = _safe_float(os.environ.get('IDENTIFIED_DEAD_TIME', '0'), 0.0)
    dyn_horizon = max(0.0, tau + dt)
    if dyn_horizon > 1e-6:
        v = int(round(settle_multiple * dyn_horizon))
        v = max(int(min_length), min(int(max_length), v))
        return v, f'auto:{settle_multiple:g}x_tau_plus_dt'

    return int(default_fallback), 'default'


def trainer_auto_tuned_block(episode_length: int, episode_length_source: str) -> dict:
    """Build the ``auto_tuned`` sub-dict embedded in ``controller_config.json``.

    This is the single authoritative location for trainer-side auto-tuned
    values (episode length + the identified dynamics that drove it).  Consumers
    (validation, runner) read from it rather than writing their own copies.
    """
    def _maybe(key: str):
        raw = os.environ.get(key)
        if raw is None or str(raw).strip() == '':
            return None
        try:
            v = float(raw)
            return v if math.isfinite(v) else None
        except Exception:
            return None

    return {
        'sim_episode_length': int(episode_length),
        'sim_episode_length_source': str(episode_length_source),
        'identified_tau_dominant': _maybe('IDENTIFIED_TAU_DOMINANT'),
        'identified_dead_time': _maybe('IDENTIFIED_DEAD_TIME'),
        'identified_lookback_seed': _maybe('IDENTIFIED_LOOKBACK_SEED'),
        'sim_noise_stdv': _maybe('SIM_NOISE_STDV'),
    }
