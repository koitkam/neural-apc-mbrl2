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
    before calling this function.

    **Bound-step continuity fix (P60, 2026-05-28).**  Historically the
    mapping rescaled against ``setpoint_manager.current_mv_bounds`` so
    operator-driven MV-limit changes "tracked" through the controller.
    That coupling caused a spurious discontinuous MV jump every time the
    runtime MV bounds stepped, even when the previous MV value was
    strictly interior to the new bounds: the same normalised action
    re-mapped onto a shifted [lo, hi] window produces a different raw
    MV.  Validation runs visibly showed MV stepping up by several percent
    at every MV-bound-step event regardless of whether clipping was
    needed (`run_p59` ep_00, t=247: MV 52.1 → 55.5 when bounds shifted
    from [20, 80] → [27.24, 87.58] with MV already deep in the
    interior).

    The fix: always rescale against the *static* base MV bounds.  The
    runtime ``current_mv_bounds`` continue to drive:
      - the bounds info channels in the observation (so the policy can
        choose to respond to a bound change),
      - the violation penalty (so the policy is incentivised to respect
        the new limits),
    but they no longer warp the action→MV mapping itself.  Net effect:
    when the runtime bounds shift but the prior MV remains valid, the
    agent's MV stays where it was unless the policy intentionally moves
    it.

    ``setpoint_manager`` is kept in the signature for backward
    compatibility; it is ignored.
    """
    del setpoint_manager  # intentionally unused; see docstring.
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
    return lo + action * (hi - lo)
