"""Derive a reasonable training episode length from identified dynamics.

The goal is to give slow plants (large tau + dead_time) enough samples per
episode to see the step response settle, while keeping fast plants from
wasting training time on padded tails.  Resolution order:

1. Canonical ``DREAMER_EPISODE_LENGTH`` (explicit pin).
2. Identified dynamics: ``k * (tau + dead_time)`` steps, floored + ceilinged.
3. Fallback: ``default_fallback`` (1000).

Leftover ``SIM_EPISODE_LENGTH`` is ignored at derive time (P92-live;
login leftover was a silent A/B).  ``single_run`` / BO still WRITE
``SIM_EPISODE_LENGTH`` after derivation as IPC for env / validate
(same class as ``SIM_NOISE_CONFIG_JSON``).

All numbers come from :mod:`utils.dynamics_identifier` via the
``IDENTIFIED_TAU_DOMINANT`` / ``IDENTIFIED_DEAD_TIME`` environment variables
that :mod:`workflow.bo_runner` sets after identification.  Formula knobs
(``k``, min, max) are TrainConfig ``episode_settle_multiple`` /
``episode_min_length`` / ``episode_max_length`` (identity 20 / 500 / 4000)
via ``episode_formula_knobs()``.
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


def episode_formula_knobs() -> Tuple[float, int, int]:
    """``k`` / ``min_length`` / ``max_length`` for ``derive_episode_length``.

    ``single_run`` / BO call ``derive_episode_length`` before a plant-filled
    ``TrainConfig`` exists, so the formula used to hard-code ``20 / 500 /
    4000``.  Identity for env-free: those numbers **are** the TrainConfig
    defaults.  Changing the dataclass now actually sizes L.  Canonical
    ``DREAMER_EPISODE_SETTLE_MULTIPLE`` / ``DREAMER_EPISODE_MIN_LENGTH`` /
    ``DREAMER_EPISODE_MAX_LENGTH`` still apply when set (same keys as
    ``ENV_OVERRIDES``; probe-before-cfg A/B).  Does **not** call
    ``apply_dreamer_env_overrides`` (would reprint ``[env-override]``).
    Leftover ``SIM_EPISODE_LENGTH`` is ignored.
    """
    k = 20.0
    min_len = 500
    max_len = 4000
    try:
        from training.train import TrainConfig
        cfg = TrainConfig()
        k = float(getattr(cfg, 'episode_settle_multiple', k) or k)
        min_len = int(getattr(cfg, 'episode_min_length', min_len) or min_len)
        max_len = int(getattr(cfg, 'episode_max_length', max_len) or max_len)
    except Exception:
        pass
    raw = os.environ.get('DREAMER_EPISODE_SETTLE_MULTIPLE', '').strip()
    if raw:
        try:
            k = float(raw)
        except ValueError:
            pass
    raw = os.environ.get('DREAMER_EPISODE_MIN_LENGTH', '').strip()
    if raw:
        try:
            min_len = int(float(raw))
        except ValueError:
            pass
    raw = os.environ.get('DREAMER_EPISODE_MAX_LENGTH', '').strip()
    if raw:
        try:
            max_len = int(float(raw))
        except ValueError:
            pass
    min_len = max(2, int(min_len))
    max_len = max(min_len, int(max_len))
    return float(k), min_len, max_len


def derive_episode_length(
    default_fallback: int = 1000,
    settle_multiple: Optional[float] = None,
    min_length: Optional[int] = None,
    max_length: Optional[int] = None,
) -> Tuple[int, str]:
    """Return ``(episode_length, source)``.

    - ``source='env:DREAMER_EPISODE_LENGTH'``: canonical explicit pin.
    - ``source='auto:{k}x_tau_plus_dt'``: derived from identified dynamics.
    - ``source='default'``: fallback (no identification available).

    Formula inputs come from ``episode_formula_knobs()`` (TrainConfig
    then canonical ``DREAMER_EPISODE_*``).  Explicit args override the
    dataclass.  Leftover ``SIM_EPISODE_LENGTH`` is ignored (P92-live).
    """
    pin = os.environ.get('DREAMER_EPISODE_LENGTH', '').strip()
    if pin:
        try:
            v = int(float(pin))
            if v > 0:
                return v, 'env:DREAMER_EPISODE_LENGTH'
        except Exception:
            pass

    kn_k, kn_min, kn_max = episode_formula_knobs()
    k = kn_k if settle_multiple is None else float(settle_multiple)
    min_len = kn_min if min_length is None else int(min_length)
    max_len = kn_max if max_length is None else int(max_length)
    min_len = max(2, int(min_len))
    max_len = max(min_len, int(max_len))

    tau = _safe_float(os.environ.get('IDENTIFIED_TAU_DOMINANT', '0'), 0.0)
    dt = _safe_float(os.environ.get('IDENTIFIED_DEAD_TIME', '0'), 0.0)
    dyn_horizon = max(0.0, tau + dt)
    if dyn_horizon > 1e-6:
        v = int(round(k * dyn_horizon))
        v = max(int(min_len), min(int(max_len), v))
        return v, f'auto:{k:g}x_tau_plus_dt'

    return int(default_fallback), 'default'


def horizon_formula_knobs() -> Tuple[float, int]:
    """``settle_n_tau`` / ``horizon_max`` for ``derive_horizon``.

    ``single_run`` / BO call ``derive_horizon`` before a plant-filled
    ``TrainConfig`` exists, so the formula used to hard-code ``4.0 / 120``.
    Identity for env-free: those numbers **are** the TrainConfig defaults.
    Changing the dataclass now actually sizes H.  Leftover
    ``DREAMER_HORIZON_SETTLE_NTAU`` / ``DREAMER_HORIZON_MAX`` still win
    when set (same keys as ``ENV_OVERRIDES``).  Does **not** call
    ``apply_dreamer_env_overrides``.
    """
    n_tau = 4.0
    max_h = 120
    try:
        from training.train import TrainConfig
        cfg = TrainConfig()
        n_tau = float(getattr(cfg, 'horizon_settle_n_tau', n_tau) or n_tau)
        max_h = int(getattr(cfg, 'horizon_max', max_h) or max_h)
    except Exception:
        pass
    raw = os.environ.get('DREAMER_HORIZON_SETTLE_NTAU', '').strip()
    if raw:
        try:
            n_tau = float(raw)
        except ValueError:
            pass
    raw = os.environ.get('DREAMER_HORIZON_MAX', '').strip()
    if raw:
        try:
            max_h = int(float(raw))
        except ValueError:
            pass
    return float(n_tau), max(15, int(max_h))


def derive_horizon(
    tau: float,
    dead_time: float,
    sample_rate: int,
    settle_n_tau: Optional[float] = None,
    min_h: int = 15,
    max_h: Optional[int] = None,
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
    - explicit ``settle_n_tau`` / ``max_h`` args (win over leftover env)
    - leftover ``DREAMER_HORIZON_SETTLE_NTAU`` / ``DREAMER_HORIZON_MAX``
      via ``horizon_formula_knobs()`` when the arg is None
    - TrainConfig ``horizon_settle_n_tau`` / ``horizon_max`` (identity 4.0 / 120)
    - ``source='auto:{n}tau_settle'``: derived from identified dynamics.
    - ``source='default'``: paper floor (15) when no dynamics are available.

    Floored at ``min_h`` (the DreamerV3/V4 paper default, 15) so fast plants
    never go below the paper minimum, and capped at ``max_h``.  An explicit
    ``DREAMER_HORIZON`` still hard-overrides downstream via ENV_OVERRIDES.
    """
    kn_tau, kn_max = horizon_formula_knobs()
    n_tau = kn_tau if settle_n_tau is None else float(settle_n_tau)
    cap = kn_max if max_h is None else int(max_h)
    # Explicit args override knobs (P92-live leftover-class: leftover
    # ``DREAMER_HORIZON_*`` used to win over an explicit arg after knobs
    # already applied it).  Leftover env still applies via
    # ``horizon_formula_knobs()`` when the arg is None.
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
