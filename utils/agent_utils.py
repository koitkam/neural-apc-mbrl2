"""Slim agent utilities for DreamerV4.

Stripped from neural-apc-pytorch's full agent_utils to keep only the helpers
needed by the V4 trainer:

- action mapping ([0, 1] -> engineering MV via current bounds)
- objective spec / weights / bounds loaders

All observer-related helpers, PPO/SAC reward wrappers, GAE, action history,
and ONNX export shims are intentionally removed.  V4 has a single algorithm
and a single integrated ONNX artifact.
"""

import numpy as np

from utils.objective_config import load_objective_spec


# ---------------------------------------------------------------------------
# Objective helpers
# ---------------------------------------------------------------------------

def load_objective_weights():
    return load_objective_spec()['weights']


def load_objective_bounds():
    return load_objective_spec()['bounds']


def load_full_objective_spec():
    """Return the full normalised control objective spec (new-schema keys)."""
    return load_objective_spec()


# ---------------------------------------------------------------------------
# Action / control conversion
# ---------------------------------------------------------------------------

def action_to_control(action, bounds, setpoint_manager=None):
    """Map normalised [0, 1] action vector to engineering-unit MV values.

    The V4 actor emits actions in [-1, 1]; the trainer rescales to [0, 1]
    before calling this function.  When a ``setpoint_manager`` is provided
    and exposes ``current_mv_bounds`` of shape ``(n_mv, 2)``, the live
    bounds are used instead of the static ``bounds['mv_bounds']`` so that
    operator limit changes are tracked by the controller (not only by the
    violation penalty).  Runtime bounds are clipped to the static base
    bounds so the agent can never drive the actuator past its physical
    limits.
    """
    action = np.clip(np.asarray(action, dtype='float32').reshape(-1), 0.0, 1.0)
    mv_bounds = list(bounds.get('mv_bounds', []))
    if len(mv_bounds) < len(action) and isinstance(bounds.get('mvs'), dict):
        mvs = bounds['mvs']
        mv_bounds = [list(mvs[k]) for k in sorted(mvs)]
    if len(mv_bounds) < len(action):
        mv_bounds = mv_bounds + [[0.0, 100.0] for _ in range(len(action) - len(mv_bounds))]
    mv_bounds = mv_bounds[:len(action)]
    lo = np.array([b[0] for b in mv_bounds], dtype='float32')
    hi = np.array([b[1] for b in mv_bounds], dtype='float32')
    if setpoint_manager is not None:
        try:
            rt = np.asarray(getattr(setpoint_manager, 'current_mv_bounds'), dtype='float32')
            if rt.shape == (len(action), 2):
                rt_lo = np.maximum(rt[:, 0], lo)
                rt_hi = np.minimum(rt[:, 1], hi)
                valid = rt_hi > rt_lo
                lo = np.where(valid, rt_lo, lo)
                hi = np.where(valid, rt_hi, hi)
        except Exception:
            pass
    return lo + action * (hi - lo)
