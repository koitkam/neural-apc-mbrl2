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


def derive_horizon(
    tau: float,
    dead_time: float,
    sample_rate: int,
    settle_n_tau: float = 4.0,
    min_h: int = 15,
    max_h: int = 120,
) -> Tuple[int, str]:
    """Return ``(horizon, source)`` — the imagination horizon in AGENT steps.

    The horizon is sized to the identified *time to steady state* so the
    actor/critic can credit the full settling response of the slowest loop
    (and, for limit-tracking, see the consequence of riding vs not-riding a
    moved operator limit over the whole transient).  The settling time uses
    the textbook first-order 2% criterion ``t_settle = dead_time + 4*tau``
    (raw sim steps), divided by ``sample_rate`` to convert to agent steps:

        ``H = round((dead_time + settle_n_tau * tau) / sample_rate)``

    Resolution order:
    - ``source='auto:{n}tau_settle'``: derived from identified dynamics.
    - ``source='default'``: paper floor (15) when no dynamics are available.

    Floored at ``min_h`` (the DreamerV3/V4 paper default, 15) so fast plants
    never go below the paper minimum, and capped at ``max_h`` (env
    ``DREAMER_HORIZON_MAX``) to bound imagination compute / WM-rollout error.
    ``settle_n_tau`` is overridable via ``DREAMER_HORIZON_SETTLE_NTAU``.  An
    explicit ``DREAMER_HORIZON`` still hard-overrides downstream via the
    env-override layer.
    """
    try:
        n_tau = float(os.environ.get('DREAMER_HORIZON_SETTLE_NTAU', '').strip()
                      or settle_n_tau)
    except Exception:
        n_tau = settle_n_tau
    try:
        cap = int(float(os.environ.get('DREAMER_HORIZON_MAX', '').strip() or max_h))
    except Exception:
        cap = int(max_h)
    cap = max(int(min_h), cap)

    tau_v = _safe_float(tau, 0.0)
    dt_v = _safe_float(dead_time, 0.0)
    sr = max(1, int(sample_rate or 1))
    t_settle = dt_v + n_tau * tau_v
    if t_settle > 1e-6:
        h = int(round(t_settle / sr))
        h = max(int(min_h), min(cap, h))
        return h, f'auto:{n_tau:g}tau_settle'
    return int(min_h), 'default'


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
