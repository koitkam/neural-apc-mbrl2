"""Adaptive steady-state APC experts for grounding the deterministic policy mean.

Why this exists
---------------
Across P59-P80 the Dreamer machinery was healthy (WM r~0.77, reward-head r~0.96,
critic r~0.82) yet the deterministic policy mean converged to a worse-than-do-
nothing operating policy (mv_activity~0.013 but mv_tv~729, bound_hug~0.49).  Root
cause: imagination exploitation under bootstrap dominance
(critic_rew_to_tgt_var -> ~0.001).  Exploration (sigma) is fine, gamma is paper-
faithful; the policy-improvement target simply lands the deterministic mean in the
wrong basin and the reward's deceptive geometry (flat, gated economics + a knife-
edge constraint) gives nothing to climb back out with.

These experts provide a *dense, correct-direction anchor* for the policy mean.
Behaviour-cloning toward the expert (Cursor stabilizer #6) converts a deceptive
global search into local refinement around a known-good operating policy.  The
expert is ONLY a ``bc_scale``-weighted regulariser on the deterministic mean -- RL
still owns the true (nonlinear) optimum via the real reward + world model.

The APC objective (NO setpoint)
-------------------------------
``utils/objective_runtime.py`` reward = -(MV/CV bound-violation quadratics)
- (move) - phi * (economic linear term), where ``phi`` is a feasibility gate that
switches economics OFF whenever a constraint is violated.  The optimum therefore
sits on the binding-constraint edge (maximum economics without violation).  There
is no setpoint; the expert is a constraint-aware *economic mover*.

Two pluggable target generators (both consume the SAME real steady-state data)
------------------------------------------------------------------------------
``gather_steadystate_samples`` runs a clean constant-action sweep on a fresh
simulator (like dynamics identification) and records settled ``(MV, DV, CV)`` in
engineering units -- real plant equilibria, NOT world-model rollouts (the WM's
steady-state fidelity is the known weak spot, so it is deliberately not used here).

  * :class:`GainScheduleExpert` -- robust *gain scheduling* over the operating
    region.  Fits a set of operating-point anchors ``{(MV_op, G_op)}`` by local
    linear regression of the sampled equilibria (falling back to the single-OP
    identified gain when sampling is unavailable), then interpolates the signed
    gain ``G(OP)`` at the current MV and issues a damped-pseudo-inverse,
    feasibility-gated economic move.  Handles smooth nonlinearity because the gain
    tracks the operating point.

  * :class:`NNSteadyStateExpert` -- a small MLP surrogate ``h_theta(MV, DV) ->
    CV_ss`` trained on the SAME real equilibria, then per-step the MV target is
    obtained by projected-gradient maximisation of the LITERAL reward through the
    surrogate.  Captures arbitrary curvature and a region-dependent optimum.

Both implement :class:`SteadyStateExpert` so they are drop-in swaps; the trainer
selects via ``expert_type in {'none', 'static', 'nn'}``.

Adaptivity / sim-agnosticism
----------------------------
All inputs are per-sim and already produced (objective spec + identified gains +
the SS sweep).  No plant-specific constants.  SISO and MIMO use the identical code
path; the pseudo-inverse / surrogate degenerate gracefully to the scalar case.
Dimensionless knobs are overridable via ``DREAMER_EXPERT_*`` env vars.
"""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


def _env_float(key: str, default: float) -> float:
    v = os.environ.get(key, '')
    if v is None or str(v).strip() == '':
        return float(default)
    try:
        return float(v)
    except ValueError:
        return float(default)


# ---------------------------------------------------------------------------
# Name <-> index helpers
# ---------------------------------------------------------------------------

def _name_index_map(names: Sequence[str]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for i, nm in enumerate(names):
        if nm is None:
            continue
        s = str(nm)
        out.setdefault(s, i)
        out.setdefault(s.strip().lower(), i)
    return out


def _lookup_index(name: Any, idx_map: Dict[str, int]) -> Optional[int]:
    if name is None:
        return None
    s = str(name)
    if s in idx_map:
        return idx_map[s]
    return idx_map.get(s.strip().lower())


# ---------------------------------------------------------------------------
# Signed steady-state gain matrices from dynamics_identification.json
# ---------------------------------------------------------------------------

def build_gain_matrices(
    dyn_id: Dict[str, Any],
    mv_names: Sequence[str],
    cv_names: Sequence[str],
    dv_names: Sequence[str],
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Assemble signed steady-state gain matrices from per-pair FOPDT estimates.

    Returns ``(G, Gd, info)`` where ``G[j, i] = dCV_j / dMV_i`` (eng units, shape
    ``(n_cv, n_mv)``) and ``Gd[j, k] = dCV_j / dDV_k``.  Missing pairs stay 0.0;
    repeats are reduced by the median (robust to a single bad FOPDT fit).
    """
    n_mv, n_cv, n_dv = len(mv_names), len(cv_names), len(dv_names)
    mv_map = _name_index_map(mv_names)
    cv_map = _name_index_map(cv_names)
    dv_map = _name_index_map(dv_names)

    mv_acc: Dict[Tuple[int, int], List[float]] = {}
    dv_acc: Dict[Tuple[int, int], List[float]] = {}
    n_valid_mv = n_valid_dv = 0

    for row in (dyn_id.get('per_pair_estimates', []) or []):
        if not isinstance(row, dict):
            continue
        if not row.get('valid', False) or row.get('reject_reason'):
            continue
        try:
            y0 = float(row.get('y0'))
            yf = float(row.get('yf'))
        except (TypeError, ValueError):
            continue
        amp = row.get('amplitude', None)
        amp = float(amp) if amp is not None else (yf - y0)
        delta = row.get('delta_applied', None)
        if delta is None:
            delta = row.get('delta', None)
        try:
            delta = float(delta)
        except (TypeError, ValueError):
            continue
        if abs(delta) < 1e-9:
            continue
        gain = amp / delta

        cv_idx = _lookup_index(row.get('cv'), cv_map)
        if cv_idx is None:
            cv_idx = _lookup_index(row.get('cv_name'), cv_map)
        if cv_idx is None:
            continue

        itype = str(row.get('input_type', 'mv')).lower()
        if itype == 'mv':
            mv_idx = _lookup_index(row.get('mv'), mv_map)
            if mv_idx is None:
                mv_idx = _lookup_index(row.get('input_name'), mv_map)
            if mv_idx is None:
                ii = row.get('input_index', None)
                if isinstance(ii, int) and 0 <= ii < n_mv:
                    mv_idx = ii
            if mv_idx is None:
                continue
            mv_acc.setdefault((cv_idx, mv_idx), []).append(gain)
            n_valid_mv += 1
        elif itype == 'dv':
            dv_idx = _lookup_index(row.get('dv'), dv_map)
            if dv_idx is None:
                dv_idx = _lookup_index(row.get('input_name'), dv_map)
            if dv_idx is None:
                ii = row.get('input_index', None)
                if isinstance(ii, int) and 0 <= ii < n_dv:
                    dv_idx = ii
            if dv_idx is None:
                continue
            dv_acc.setdefault((cv_idx, dv_idx), []).append(gain)
            n_valid_dv += 1

    G = np.zeros((n_cv, n_mv), dtype='float64')
    for (j, i), vals in mv_acc.items():
        G[j, i] = float(np.median(vals))
    if n_dv > 0:
        Gd = np.zeros((n_cv, n_dv), dtype='float64')
        for (j, k), vals in dv_acc.items():
            Gd[j, k] = float(np.median(vals))
    else:
        Gd = np.zeros((n_cv, 0), dtype='float64')

    info = {
        'n_mv': n_mv, 'n_cv': n_cv, 'n_dv': n_dv,
        'mv_pairs_covered': int(len(mv_acc)),
        'mv_pairs_total': int(n_cv * n_mv),
        'dv_pairs_covered': int(len(dv_acc)),
        'valid_mv_rows': int(n_valid_mv),
        'valid_dv_rows': int(n_valid_dv),
        'G': G.tolist(), 'Gd': Gd.tolist(),
    }
    return G, Gd, info


# ---------------------------------------------------------------------------
# Shared steady-state sampler (real plant equilibria, clean constant holds)
# ---------------------------------------------------------------------------

def gather_steadystate_samples(
    *,
    mv_bounds_eu: np.ndarray,
    n_grid: int = 7,
    max_samples: int = 81,
    settle_steps: Optional[int] = None,
    sample_rate: int = 1,
    seed: int = 0,
) -> Dict[str, np.ndarray]:
    """Sweep constant MV holds on a fresh sim; record settled ``(MV, DV, CV)``.

    The sweep is deterministic given ``seed``.  For ``n_mv <= 2`` a full grid is
    used; for higher dimensions a quasi-random Latin-hypercube-style sample keeps
    the count under ``max_samples``.  All values are returned in engineering
    units (denormalised via simulator metadata).  DV channels are held at their
    reset/nominal value (DV-level sweeps are left to the identified ``Gd``
    feedforward, which already covers DV->CV).

    Returns ``{'mv': (N, n_mv), 'dv': (N, n_dv), 'cv': (N, n_cv), 'meta': {...}}``.
    Raises ``RuntimeError`` if the simulator cannot be constructed.
    """
    from utils.sim_factory import create_sim, resolve_sim_metadata
    from utils.dynamics_identifier import (
        _state_value_to_engineering, _get_cv_indices, _detect_settle_time,
    )

    mv_bounds_eu = np.asarray(mv_bounds_eu, dtype='float64').reshape(-1, 2)
    n_mv = mv_bounds_eu.shape[0]

    if settle_steps is None:
        try:
            settle_steps = int(_detect_settle_time())
        except Exception:
            settle_steps = 400
    settle_steps = max(20, int(settle_steps))
    total_steps = settle_steps + 5

    lo = mv_bounds_eu[:, 0]
    hi = mv_bounds_eu[:, 1]
    rng = np.random.default_rng(int(seed))
    if n_mv <= 2:
        axes = [np.linspace(lo[i], hi[i], n_grid) for i in range(n_mv)]
        mesh = np.meshgrid(*axes, indexing='ij')
        mv_grid = np.stack([m.reshape(-1) for m in mesh], axis=1)
    else:
        n = min(max_samples, max(n_grid * n_mv, 16))
        cols = []
        for i in range(n_mv):
            edges = np.linspace(0.0, 1.0, n + 1)
            u = rng.uniform(edges[:-1], edges[1:])
            rng.shuffle(u)
            cols.append(lo[i] + u * (hi[i] - lo[i]))
        mv_grid = np.stack(cols, axis=1)
    if mv_grid.shape[0] > max_samples:
        sel = rng.choice(mv_grid.shape[0], size=max_samples, replace=False)
        mv_grid = mv_grid[np.sort(sel)]

    sim = create_sim(episode_length=total_steps, sample_rate=max(1, int(sample_rate)))
    meta = resolve_sim_metadata(sim)
    cv_idxs = [int(x) for x in _get_cv_indices(meta)]
    dv_idxs = [int(x) for x in meta.get('dv_indices', []) if x is not None]
    n_cv = len(cv_idxs)
    n_dv = len(dv_idxs)

    mv_out, cv_out, dv_out = [], [], []
    for u_eu in mv_grid:
        state = sim.reset()
        if isinstance(state, tuple):
            state = state[0]
        state = np.asarray(state, dtype='float64').reshape(-1)
        u = np.asarray(u_eu, dtype='float32').reshape(-1)
        for _ in range(settle_steps):
            step_out = sim.step(u)
            state = step_out[0] if isinstance(step_out, tuple) else step_out
            state = np.asarray(state, dtype='float64').reshape(-1)
        cv_eu = [float(_state_value_to_engineering(state, ci, meta)) for ci in cv_idxs]
        dv_eu = [float(_state_value_to_engineering(state, di, meta)) for di in dv_idxs]
        mv_out.append([float(x) for x in u_eu])
        cv_out.append(cv_eu)
        dv_out.append(dv_eu)

    return {
        'mv': np.asarray(mv_out, dtype='float64'),
        'dv': (np.asarray(dv_out, dtype='float64')
               if n_dv > 0 else np.zeros((len(mv_out), 0))),
        'cv': np.asarray(cv_out, dtype='float64'),
        'meta': {'n_mv': n_mv, 'n_cv': n_cv, 'n_dv': n_dv,
                 'settle_steps': settle_steps, 'n_samples': len(mv_out)},
    }


# ---------------------------------------------------------------------------
# Objective-spec resolution shared by both experts
# ---------------------------------------------------------------------------

def _resolve_priority_weights(cv_priority: Sequence[str],
                              cv_names: Sequence[str]) -> np.ndarray:
    n = len(cv_names)
    w = np.ones(n, dtype='float64')
    if cv_priority:
        name_map = _name_index_map(cv_names)
        ranked = [idx for nm in cv_priority
                  if (idx := _lookup_index(nm, name_map)) is not None]
        if ranked:
            n_rank = len(ranked)
            for r, idx in enumerate(ranked):
                w[idx] = float(n_rank - r)
            for idx in (i for i in range(n) if i not in ranked):
                w[idx] = 0.5
    m = float(np.mean(w)) if n > 0 else 1.0
    return w / m if m > 1e-9 else w


def _resolve_side_scales(spec: Dict[str, Any],
                         cv_names: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
    n = len(cv_names)
    lo = np.ones(n, dtype='float64')
    hi = np.ones(n, dtype='float64')
    side = (spec or {}).get('cv_side_scale', {}) or {}
    if isinstance(side, dict):
        name_map = _name_index_map(cv_names)
        for key, val in side.items():
            idx = _lookup_index(key, name_map)
            if idx is None or not isinstance(val, dict):
                continue
            try:
                lo[idx] = float(val.get('lo', 1.0))
                hi[idx] = float(val.get('hi', 1.0))
            except (TypeError, ValueError):
                continue
    return lo, hi


def _resolve_econ_signs(obj_w: Dict[str, Any], n_mv: int) -> np.ndarray:
    signs = np.zeros(n_mv, dtype='float64')
    vec = obj_w.get('mv_economic_weights', None)
    if vec is None:
        return signs
    try:
        arr = np.asarray(vec, dtype='float64').reshape(-1)
    except (TypeError, ValueError):
        return signs
    for i in range(min(n_mv, arr.shape[0])):
        signs[i] = float(np.sign(arr[i]))
    return signs


def _resolve_cv_violation_weights(obj_w: Dict[str, Any], n_cv: int) -> np.ndarray:
    vec = obj_w.get('cv_violation_weights', None)
    out = np.ones(n_cv, dtype='float64')
    if vec is not None:
        try:
            arr = np.asarray(vec, dtype='float64').reshape(-1)
            for j in range(min(n_cv, arr.shape[0])):
                out[j] = float(arr[j])
        except (TypeError, ValueError):
            pass
    return out


# ---------------------------------------------------------------------------
# Pluggable expert interface
# ---------------------------------------------------------------------------

class SteadyStateExpert(ABC):
    """Base class: rate-limited MV mover that wraps a target generator.

    Subclasses implement :meth:`_target_mv_eu` returning the *absolute* desired
    MV (engineering units) for the current CVs; the base class handles rate-
    limiting from the previous MV, bound clipping, and the mapping to the actor's
    normalised ``[-1, 1]`` action space.
    """

    def __init__(self, *, mv_bounds: np.ndarray, move_frac: float = 0.30):
        self.mv_bounds = np.asarray(mv_bounds, dtype='float64').reshape(-1, 2)
        self.n_mv = self.mv_bounds.shape[0]
        self.mv_span = np.maximum(self.mv_bounds[:, 1] - self.mv_bounds[:, 0], 1e-9)
        self.move_frac = _env_float('DREAMER_EXPERT_MOVE_FRAC', move_frac)
        self._u = self.mv_bounds.mean(axis=1).astype('float64')

    # ---- lifecycle -----------------------------------------------------
    def reset(self, u0_eu: Optional[np.ndarray] = None) -> None:
        if u0_eu is not None:
            self._u = np.clip(np.asarray(u0_eu, dtype='float64').reshape(-1),
                              self.mv_bounds[:, 0], self.mv_bounds[:, 1])
        else:
            self._u = self.mv_bounds.mean(axis=1).astype('float64')
        self._on_reset()

    def _on_reset(self) -> None:  # optional hook
        pass

    def step_eu(self, cv_eu: np.ndarray, *,
                cv_bounds: Optional[np.ndarray] = None,
                dv_eu: Optional[np.ndarray] = None) -> np.ndarray:
        target = np.asarray(
            self._target_mv_eu(np.asarray(cv_eu, dtype='float64').reshape(-1),
                               cv_bounds, dv_eu),
            dtype='float64').reshape(-1)
        move_max = self.move_frac * self.mv_span
        du = np.clip(target - self._u, -move_max, move_max)
        self._u = np.clip(self._u + du, self.mv_bounds[:, 0], self.mv_bounds[:, 1])
        return self._u.copy()

    def action_norm(self, u_eu: Optional[np.ndarray] = None) -> np.ndarray:
        """Map an engineering MV to the actor's normalised ``[-1, 1]`` space.

        Uses STATIC base MV bounds, matching ``utils.agent_utils.action_to_control``.
        """
        u = self._u if u_eu is None else np.asarray(u_eu, dtype='float64').reshape(-1)
        a01 = (u - self.mv_bounds[:, 0]) / self.mv_span
        return np.clip(2.0 * a01 - 1.0, -1.0, 1.0).astype('float32')

    def is_usable(self) -> bool:
        return True

    @abstractmethod
    def _target_mv_eu(self, cv_eu: np.ndarray,
                      cv_bounds: Optional[np.ndarray],
                      dv_eu: Optional[np.ndarray]) -> np.ndarray:
        ...


# ---------------------------------------------------------------------------
# Static gain-scheduled expert
# ---------------------------------------------------------------------------

class GainScheduleExpert(SteadyStateExpert):
    """Constraint-aware economic mover with gain scheduling over the OP region.

    ``anchors`` is a list of ``(op_mv_eu, G)`` pairs; the active gain at the
    current MV is the inverse-distance-weighted blend of the anchor gains
    (distances measured in normalised MV space).  A single anchor reproduces the
    classic single-operating-point expert.
    """

    def __init__(
        self,
        *,
        mv_bounds: np.ndarray,
        cv_bounds: np.ndarray,
        cv_priority_weight: np.ndarray,
        cv_side_lo: np.ndarray,
        cv_side_hi: np.ndarray,
        mv_econ_sign: np.ndarray,
        anchors: List[Tuple[np.ndarray, np.ndarray]],
        Gd: Optional[np.ndarray] = None,
        backoff_frac: float = 0.12,
        econ_frac: float = 0.02,
        move_frac: float = 0.30,
        loop_gain: float = 0.6,
        ridge_frac: float = 0.05,
        feas_scale: float = 0.02,
    ) -> None:
        super().__init__(mv_bounds=mv_bounds, move_frac=move_frac)
        self.cv_bounds = np.asarray(cv_bounds, dtype='float64').reshape(-1, 2)
        self.n_cv = self.cv_bounds.shape[0]
        self.cv_span = np.maximum(self.cv_bounds[:, 1] - self.cv_bounds[:, 0], 1e-9)
        self.cv_priority_weight = np.asarray(cv_priority_weight, dtype='float64').reshape(-1)
        self.cv_side_lo = np.asarray(cv_side_lo, dtype='float64').reshape(-1)
        self.cv_side_hi = np.asarray(cv_side_hi, dtype='float64').reshape(-1)
        self.mv_econ_sign = np.asarray(mv_econ_sign, dtype='float64').reshape(-1)

        self.backoff_frac = _env_float('DREAMER_EXPERT_BACKOFF_FRAC', backoff_frac)
        self.econ_frac = _env_float('DREAMER_EXPERT_ECON_FRAC', econ_frac)
        self.loop_gain = _env_float('DREAMER_EXPERT_LOOP_GAIN', loop_gain)
        self.ridge_frac = _env_float('DREAMER_EXPERT_RIDGE_FRAC', ridge_frac)
        self.feas_scale = _env_float('DREAMER_EXPERT_FEAS_SCALE', feas_scale)

        self.anchor_ops = [np.asarray(op, dtype='float64').reshape(-1) for op, _ in anchors]
        self.anchor_G = [np.asarray(G, dtype='float64').reshape(self.n_cv, self.n_mv)
                         for _, G in anchors]
        if Gd is None or np.asarray(Gd).size == 0:
            self.Gd = np.zeros((self.n_cv, 0), dtype='float64')
        else:
            self.Gd = np.asarray(Gd, dtype='float64').reshape(self.n_cv, -1)
        self._dv_prev: Optional[np.ndarray] = None

    # ---- gain scheduling ----------------------------------------------
    def _G_at(self, u_eu: np.ndarray) -> np.ndarray:
        if len(self.anchor_G) == 1:
            return self.anchor_G[0]
        un = (u_eu - self.mv_bounds[:, 0]) / self.mv_span
        wts = np.empty(len(self.anchor_ops), dtype='float64')
        for k, op in enumerate(self.anchor_ops):
            opn = (op - self.mv_bounds[:, 0]) / self.mv_span
            d2 = float(np.sum((un - opn) ** 2))
            wts[k] = 1.0 / (d2 + 1e-6)
        wts /= max(1e-12, float(np.sum(wts)))
        G = np.zeros((self.n_cv, self.n_mv), dtype='float64')
        for k, Gk in enumerate(self.anchor_G):
            G += wts[k] * Gk
        return G

    @staticmethod
    def _damped_pinv(G: np.ndarray, ridge_frac: float) -> np.ndarray:
        n_cv, n_mv = G.shape
        if n_cv == 0 or n_mv == 0:
            return np.zeros((n_mv, n_cv), dtype='float64')
        fro2 = float(np.sum(G * G))
        lam = max(1e-12, ridge_frac * (fro2 / max(1, n_cv)))
        GGt = G @ G.T + lam * np.eye(n_cv)
        try:
            inv = np.linalg.inv(GGt)
        except np.linalg.LinAlgError:
            inv = np.linalg.pinv(GGt)
        return G.T @ inv

    def gain_condition_number(self) -> float:
        G = self.anchor_G[0] if self.anchor_G else np.zeros((0, 0))
        if G.size == 0:
            return 1.0
        try:
            s = np.linalg.svd(G, compute_uv=False)
            s = s[s > 1e-12]
            return float(s.max() / s.min()) if s.size else float('inf')
        except np.linalg.LinAlgError:
            return float('inf')

    def is_usable(self) -> bool:
        if not self.anchor_G:
            return False
        cv_reachable = np.any(np.abs(self.anchor_G[0]) > 1e-12, axis=1)
        return bool(np.all(cv_reachable))

    def _on_reset(self) -> None:
        self._dv_prev = None

    def _feasibility(self, cv_eu: np.ndarray, cv_lo: np.ndarray,
                     cv_hi: np.ndarray) -> float:
        over = np.maximum(0.0, cv_eu - cv_hi) / self.cv_span
        under = np.maximum(0.0, cv_lo - cv_eu) / self.cv_span
        V = min(float(np.sum(over * over + under * under)), 4.0)
        return float(np.exp(-V / max(1e-6, self.feas_scale)))

    def _target_mv_eu(self, cv_eu, cv_bounds, dv_eu) -> np.ndarray:
        if cv_bounds is not None:
            cb = np.asarray(cv_bounds, dtype='float64').reshape(-1, 2)
            cv_lo, cv_hi = cb[:, 0], cb[:, 1]
        else:
            cv_lo, cv_hi = self.cv_bounds[:, 0], self.cv_bounds[:, 1]

        # (1) Constraint pressure -> desired CV correction (signed).
        beta = self.backoff_frac * self.cv_span
        over = np.maximum(0.0, cv_eu - (cv_hi - beta))
        under = np.maximum(0.0, (cv_lo + beta) - cv_eu)
        w_hi = self.cv_priority_weight * self.cv_side_hi
        w_lo = self.cv_priority_weight * self.cv_side_lo
        d_cv_desired = (w_lo * under) - (w_hi * over)

        # (2) Decouple through the OP-scheduled, damped gain pseudo-inverse.
        Gpinv = self._damped_pinv(self._G_at(self._u), self.ridge_frac)
        du_con = self.loop_gain * (Gpinv @ d_cv_desired)

        # (3) Economic creep, feasibility-gated.
        phi = self._feasibility(cv_eu, cv_lo, cv_hi)
        du_econ = -phi * self.econ_frac * self.mv_span * self.mv_econ_sign

        # (4) Disturbance feedforward.
        du_ff = np.zeros(self.n_mv, dtype='float64')
        if dv_eu is not None and self.Gd.shape[1] > 0:
            dv_now = np.asarray(dv_eu, dtype='float64').reshape(-1)
            if self._dv_prev is not None and dv_now.shape == self._dv_prev.shape:
                du_ff = -(Gpinv @ (self.Gd @ (dv_now - self._dv_prev)))
            self._dv_prev = dv_now

        return self._u + du_con + du_econ + du_ff


# ---------------------------------------------------------------------------
# NN steady-state surrogate expert
# ---------------------------------------------------------------------------

class _SSSurrogate:
    """Small MLP ``(MV, DV) -> CV`` with input/output standardisation."""

    def __init__(self, n_mv: int, n_dv: int, n_cv: int, hidden: int = 64):
        import torch.nn as nn
        self.n_mv, self.n_dv, self.n_cv = n_mv, n_dv, n_cv
        self.is_trained = False
        in_dim = n_mv + n_dv
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, n_cv),
        )
        self._x_mean = None
        self._x_std = None
        self._y_mean = None
        self._y_std = None

    def _featurise(self, mv, dv):
        import torch
        if dv is None or self.n_dv == 0:
            return mv
        return torch.cat([mv, dv], dim=-1)

    def predict_torch(self, mv_eu, dv_eu):
        x = self._featurise(mv_eu, dv_eu)
        xn = (x - self._x_mean) / self._x_std
        yn = self.net(xn)
        return yn * self._y_std + self._y_mean

    def train_from_samples(self, samples: Dict[str, np.ndarray], *,
                           epochs: int = 400, lr: float = 1e-3,
                           seed: int = 0) -> Dict[str, float]:
        import torch
        torch.manual_seed(int(seed))
        mv = torch.tensor(samples['mv'], dtype=torch.float32)
        cv = torch.tensor(samples['cv'], dtype=torch.float32)
        dv = (torch.tensor(samples['dv'], dtype=torch.float32)
              if (self.n_dv > 0 and samples['dv'].shape[1] == self.n_dv) else None)
        x = self._featurise(mv, dv)
        self._x_mean = x.mean(dim=0, keepdim=True)
        self._x_std = x.std(dim=0, keepdim=True).clamp_min(1e-6)
        self._y_mean = cv.mean(dim=0, keepdim=True)
        self._y_std = cv.std(dim=0, keepdim=True).clamp_min(1e-6)
        xn = (x - self._x_mean) / self._x_std
        yn = (cv - self._y_mean) / self._y_std
        opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        lossf = torch.nn.MSELoss()
        final = 0.0
        n = xn.shape[0]
        bs = min(64, n)
        for _ in range(int(epochs)):
            perm = torch.randperm(n)
            ep_loss = 0.0
            for s in range(0, n, bs):
                idx = perm[s:s + bs]
                opt.zero_grad()
                pred = self.net(xn[idx])
                loss = lossf(pred, yn[idx])
                loss.backward()
                opt.step()
                ep_loss += float(loss.detach()) * idx.shape[0]
            final = ep_loss / max(1, n)
        with torch.no_grad():
            pred_eu = self.net(xn) * self._y_std + self._y_mean
            ss_res = float(((pred_eu - cv) ** 2).sum())
            ss_tot = float(((cv - cv.mean(dim=0, keepdim=True)) ** 2).sum()) + 1e-9
        self.is_trained = True
        return {'final_mse_norm': float(final),
                'r2_insample': float(1.0 - ss_res / ss_tot),
                'n_samples': int(n)}


class NNSteadyStateExpert(SteadyStateExpert):
    """MLP steady-state surrogate ``h(MV, DV) -> CV_ss`` + reward optimiser.

    The surrogate is trained on real settled equilibria (see
    :func:`gather_steadystate_samples`).  Per step the MV target is obtained by
    projected-gradient maximisation of the LITERAL reward (constraint quadratics
    + feasibility-gated economics) through the surrogate, so it captures
    arbitrary curvature and a region-dependent optimum.
    """

    def __init__(
        self,
        *,
        mv_bounds: np.ndarray,
        cv_bounds: np.ndarray,
        cv_violation_weight: np.ndarray,
        cv_priority_weight: np.ndarray,
        cv_side_lo: np.ndarray,
        cv_side_hi: np.ndarray,
        mv_econ_weight: np.ndarray,
        surrogate: _SSSurrogate,
        nominal_dv_eu: np.ndarray,
        backoff_frac: float = 0.12,
        econ_scale: float = 1.0,
        move_frac: float = 0.30,
        opt_iters: int = 40,
        opt_lr: float = 0.1,
        feas_scale: float = 0.02,
    ) -> None:
        super().__init__(mv_bounds=mv_bounds, move_frac=move_frac)
        self.cv_bounds = np.asarray(cv_bounds, dtype='float64').reshape(-1, 2)
        self.n_cv = self.cv_bounds.shape[0]
        self.cv_span = np.maximum(self.cv_bounds[:, 1] - self.cv_bounds[:, 0], 1e-9)
        self.cv_violation_weight = np.asarray(cv_violation_weight, dtype='float64').reshape(-1)
        self.cv_priority_weight = np.asarray(cv_priority_weight, dtype='float64').reshape(-1)
        self.cv_side_lo = np.asarray(cv_side_lo, dtype='float64').reshape(-1)
        self.cv_side_hi = np.asarray(cv_side_hi, dtype='float64').reshape(-1)
        self.mv_econ_weight = np.asarray(mv_econ_weight, dtype='float64').reshape(-1)
        self.surrogate = surrogate
        self.nominal_dv_eu = np.asarray(nominal_dv_eu, dtype='float64').reshape(-1)

        self.backoff_frac = _env_float('DREAMER_EXPERT_BACKOFF_FRAC', backoff_frac)
        self.econ_scale = _env_float('DREAMER_EXPERT_ECON_SCALE', econ_scale)
        self.opt_iters = int(_env_float('DREAMER_EXPERT_OPT_ITERS', float(opt_iters)))
        self.opt_lr = _env_float('DREAMER_EXPERT_OPT_LR', opt_lr)
        self.feas_scale = _env_float('DREAMER_EXPERT_FEAS_SCALE', feas_scale)
        self._dv_eu: Optional[np.ndarray] = None

    def is_usable(self) -> bool:
        return self.surrogate is not None and self.surrogate.is_trained

    def _on_reset(self) -> None:
        self._dv_eu = None

    def _target_mv_eu(self, cv_eu, cv_bounds, dv_eu) -> np.ndarray:
        import torch
        if cv_bounds is not None:
            cb = np.asarray(cv_bounds, dtype='float64').reshape(-1, 2)
            cv_lo = torch.tensor(cb[:, 0], dtype=torch.float32)
            cv_hi = torch.tensor(cb[:, 1], dtype=torch.float32)
        else:
            cv_lo = torch.tensor(self.cv_bounds[:, 0], dtype=torch.float32)
            cv_hi = torch.tensor(self.cv_bounds[:, 1], dtype=torch.float32)

        dv = (np.asarray(dv_eu, dtype='float64').reshape(-1)
              if dv_eu is not None else self.nominal_dv_eu)
        if dv.shape[0] != self.nominal_dv_eu.shape[0]:
            dv = self.nominal_dv_eu
        dv_t = torch.tensor(dv, dtype=torch.float32)

        lo = torch.tensor(self.mv_bounds[:, 0], dtype=torch.float32)
        hi = torch.tensor(self.mv_bounds[:, 1], dtype=torch.float32)
        span = torch.tensor(self.cv_span, dtype=torch.float32)
        w_cv = torch.tensor(self.cv_violation_weight * self.cv_priority_weight,
                            dtype=torch.float32)
        side_lo = torch.tensor(self.cv_side_lo, dtype=torch.float32)
        side_hi = torch.tensor(self.cv_side_hi, dtype=torch.float32)
        w_econ = torch.tensor(self.mv_econ_weight, dtype=torch.float32)
        beta = torch.tensor(self.backoff_frac, dtype=torch.float32) * span

        u = torch.tensor(self._u, dtype=torch.float32, requires_grad=True)
        opt = torch.optim.Adam([u], lr=self.opt_lr)
        for _ in range(max(1, self.opt_iters)):
            opt.zero_grad()
            u_clamped = torch.clamp(u, lo, hi)
            cv_pred = self.surrogate.predict_torch(u_clamped, dv_t)
            over = torch.relu(cv_pred - (cv_hi - beta)) / span
            under = torch.relu((cv_lo + beta) - cv_pred) / span
            viol = (w_cv * side_hi * over ** 2).sum() + (w_cv * side_lo * under ** 2).sum()
            with torch.no_grad():
                V = float(torch.clamp((over ** 2 + under ** 2).sum(), max=4.0))
                phi = float(np.exp(-V / max(1e-6, self.feas_scale)))
            econ = self.econ_scale * phi * (w_econ * u_clamped).sum()
            loss = viol + econ
            loss.backward()
            opt.step()
        return torch.clamp(u.detach(), lo, hi).numpy().astype('float64')


# ---------------------------------------------------------------------------
# Anchor fitting for the gain schedule (local linear regression on samples)
# ---------------------------------------------------------------------------

def fit_gain_anchors(
    samples: Dict[str, np.ndarray],
    *,
    mv_bounds_eu: np.ndarray,
    n_anchor: int = 5,
    min_neighbors: int = 6,
) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], Dict[str, Any]]:
    """Fit ``{(op_mv, G)}`` anchors via local linear regression of CV on MV.

    Anchor OPs are placed on a grid along each MV axis (or the full grid for
    ``n_mv<=2``).  At each anchor the local gain ``G`` is the least-squares
    Jacobian ``dCV/dMV`` fit from the nearest sampled equilibria.  When an anchor
    has too few neighbours it inherits the global-regression gain so the schedule
    never has holes.
    """
    mv = np.asarray(samples['mv'], dtype='float64')
    cv = np.asarray(samples['cv'], dtype='float64')
    mv_bounds_eu = np.asarray(mv_bounds_eu, dtype='float64').reshape(-1, 2)
    n_mv = mv.shape[1]
    n_cv = cv.shape[1]
    lo, hi = mv_bounds_eu[:, 0], mv_bounds_eu[:, 1]
    span = np.maximum(hi - lo, 1e-9)

    def _local_gain(center_norm: np.ndarray, k: int) -> Optional[np.ndarray]:
        mvn = (mv - lo) / span
        d2 = np.sum((mvn - center_norm) ** 2, axis=1)
        order = np.argsort(d2)
        sel = order[:max(k, min_neighbors)]
        X = mv[sel]
        Y = cv[sel]
        if X.shape[0] < n_mv + 1:
            return None
        Xc = X - X.mean(axis=0, keepdims=True)
        Yc = Y - Y.mean(axis=0, keepdims=True)
        lam = 1e-6 * float(np.trace(Xc.T @ Xc) / max(1, n_mv))
        A = Xc.T @ Xc + lam * np.eye(n_mv)
        try:
            Gt = np.linalg.solve(A, Xc.T @ Yc)   # (n_mv, n_cv)
        except np.linalg.LinAlgError:
            return None
        return Gt.T  # (n_cv, n_mv)

    global_G = _local_gain(np.full(n_mv, 0.5), k=mv.shape[0])
    if global_G is None:
        global_G = np.zeros((n_cv, n_mv), dtype='float64')

    if n_mv <= 2:
        side = max(2, int(round(n_anchor ** (1.0 / n_mv))))
        axes = [np.linspace(0.15, 0.85, side) for _ in range(n_mv)]
        mesh = np.meshgrid(*axes, indexing='ij')
        centres_norm = np.stack([m.reshape(-1) for m in mesh], axis=1)
    else:
        centres_norm = np.linspace(0.15, 0.85, n_anchor)[:, None] * np.ones((1, n_mv))

    anchors: List[Tuple[np.ndarray, np.ndarray]] = []
    n_local = 0
    k = max(min_neighbors, mv.shape[0] // max(1, centres_norm.shape[0]))
    for c in centres_norm:
        G = _local_gain(c, k=k)
        if G is None:
            G = global_G
        else:
            n_local += 1
        op_eu = lo + c * span
        anchors.append((op_eu, G))

    info = {
        'n_anchor': len(anchors),
        'n_local_fit': int(n_local),
        'global_G': global_G.tolist(),
        'anchor_ops': [a[0].tolist() for a in anchors],
        'anchor_G': [a[1].tolist() for a in anchors],
    }
    return anchors, info


# ---------------------------------------------------------------------------
# Top-level builders
# ---------------------------------------------------------------------------

def load_dynamics_id(out_dir: Optional[str]) -> Optional[Dict[str, Any]]:
    cand = os.environ.get('DREAMER_DYNAMICS_ID_JSON', '').strip()
    paths: List[str] = []
    if cand:
        paths.append(cand)
    if out_dir:
        paths.append(os.path.join(str(out_dir), 'plant_id',
                                  'dynamics_identification.json'))
    for p in paths:
        try:
            if p and os.path.isfile(p):
                with open(p, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except (OSError, ValueError):
            continue
    return None


def build_static_expert(
    *,
    dyn_id: Dict[str, Any],
    obj_spec: Dict[str, Any],
    obj_w: Dict[str, Any],
    mv_bounds: np.ndarray,
    cv_bounds: np.ndarray,
    mv_names: Sequence[str],
    cv_names: Sequence[str],
    dv_names: Sequence[str],
    samples: Optional[Dict[str, np.ndarray]] = None,
    n_anchor: int = 5,
) -> Tuple[Optional[GainScheduleExpert], Dict[str, Any]]:
    """Build a gain-scheduled static expert.

    When ``samples`` (a steady-state sweep) is supplied, the gain schedule is fit
    from real equilibria (robust over the operating region).  Otherwise the
    single-OP identified gain is used as the sole anchor.
    """
    G_id, Gd, ginfo = build_gain_matrices(dyn_id, mv_names, cv_names, dv_names)
    cv_pw = _resolve_priority_weights((obj_spec or {}).get('cv_priority', []) or [], cv_names)
    side_lo, side_hi = _resolve_side_scales(obj_spec, cv_names)
    econ_sign = _resolve_econ_signs(obj_w, len(mv_names))
    mv_bounds = np.asarray(mv_bounds, dtype='float64').reshape(-1, 2)

    if samples is not None and samples.get('mv') is not None and samples['mv'].shape[0] >= 4:
        anchors, ainfo = fit_gain_anchors(samples, mv_bounds_eu=mv_bounds, n_anchor=n_anchor)
        ginfo['schedule'] = ainfo
        ginfo['gain_source'] = 'fit_from_samples'
        # Sign-guard: if any fitted anchor disagrees in sign with the identified
        # gain (where the latter is non-zero), trust the identified single-OP
        # gain (guards against degenerate local fits in noisy regions).
        if np.any(np.abs(G_id) > 1e-9):
            sref = np.sign(G_id)
            m = np.abs(G_id) > 1e-9
            bad = any(np.any((np.sign(Gk) * sref)[m] < 0) for _, Gk in anchors)
            if bad:
                anchors = [(mv_bounds.mean(axis=1), G_id)]
                ginfo['gain_source'] = 'identified_single_op (sign-guard tripped)'
    else:
        anchors = [(mv_bounds.mean(axis=1), G_id)]
        ginfo['gain_source'] = 'identified_single_op'

    expert = GainScheduleExpert(
        mv_bounds=mv_bounds, cv_bounds=cv_bounds,
        cv_priority_weight=cv_pw, cv_side_lo=side_lo, cv_side_hi=side_hi,
        mv_econ_sign=econ_sign, anchors=anchors, Gd=Gd,
    )
    ginfo['gain_condition_number'] = expert.gain_condition_number()
    ginfo['usable'] = expert.is_usable()
    ginfo['cv_priority_weight'] = cv_pw.tolist()
    ginfo['mv_econ_sign'] = econ_sign.tolist()
    if not expert.is_usable():
        return None, ginfo
    return expert, ginfo


def build_nn_expert(
    *,
    obj_spec: Dict[str, Any],
    obj_w: Dict[str, Any],
    mv_bounds: np.ndarray,
    cv_bounds: np.ndarray,
    cv_names: Sequence[str],
    samples: Dict[str, np.ndarray],
    epochs: int = 400,
    seed: int = 0,
) -> Tuple[Optional[NNSteadyStateExpert], Dict[str, Any]]:
    """Train an MLP steady-state surrogate and wrap it in an NN expert."""
    mv = np.asarray(samples['mv'], dtype='float64')
    cv = np.asarray(samples['cv'], dtype='float64')
    dv = np.asarray(samples.get('dv', np.zeros((mv.shape[0], 0))), dtype='float64')
    n_mv, n_cv, n_dv = mv.shape[1], cv.shape[1], dv.shape[1]

    info: Dict[str, Any] = {'n_mv': n_mv, 'n_cv': n_cv, 'n_dv': n_dv,
                            'n_samples': int(mv.shape[0])}
    if mv.shape[0] < max(8, n_mv + n_dv + 2):
        info['usable'] = False
        info['reason'] = 'insufficient_samples'
        return None, info

    surrogate = _SSSurrogate(n_mv, n_dv, n_cv)
    train_info = surrogate.train_from_samples(samples, epochs=epochs, seed=seed)
    info.update(train_info)

    cv_pw = _resolve_priority_weights((obj_spec or {}).get('cv_priority', []) or [], cv_names)
    side_lo, side_hi = _resolve_side_scales(obj_spec, cv_names)
    cv_vw = _resolve_cv_violation_weights(obj_w, n_cv)
    econ_w = np.zeros(n_mv, dtype='float64')
    vec = obj_w.get('mv_economic_weights', None)
    if vec is not None:
        arr = np.asarray(vec, dtype='float64').reshape(-1)
        econ_w[:min(n_mv, arr.shape[0])] = arr[:min(n_mv, arr.shape[0])]
    nominal_dv = dv.mean(axis=0) if n_dv > 0 else np.zeros(0)

    expert = NNSteadyStateExpert(
        mv_bounds=mv_bounds, cv_bounds=cv_bounds,
        cv_violation_weight=cv_vw, cv_priority_weight=cv_pw,
        cv_side_lo=side_lo, cv_side_hi=side_hi,
        mv_econ_weight=econ_w, surrogate=surrogate, nominal_dv_eu=nominal_dv,
    )
    info['usable'] = expert.is_usable()
    if not expert.is_usable():
        return None, info
    return expert, info


def build_expert(
    *,
    expert_type: str,
    out_dir: Optional[str],
    obj_spec: Dict[str, Any],
    obj_w: Dict[str, Any],
    mv_bounds: np.ndarray,
    cv_bounds: np.ndarray,
    mv_names: Sequence[str],
    cv_names: Sequence[str],
    dv_names: Sequence[str],
    use_ss_samples: bool = True,
    seed: int = 0,
) -> Tuple[Optional[SteadyStateExpert], Dict[str, Any]]:
    """Top-level dispatcher used by the trainer.

    ``expert_type`` in ``{'none', 'static', 'nn'}``.  Loads the identified gains,
    optionally gathers a real steady-state sweep, and constructs the requested
    expert.  Returns ``(expert_or_None, info)``; ``info`` is always JSON-safe for
    logging next to the run artefacts.
    """
    et = str(expert_type or 'none').strip().lower()
    info: Dict[str, Any] = {'expert_type': et}
    if et in ('', 'none'):
        info['usable'] = False
        return None, info

    dyn_id = load_dynamics_id(out_dir)
    mv_bounds = np.asarray(mv_bounds, dtype='float64').reshape(-1, 2)

    samples = None
    if use_ss_samples:
        try:
            samples = gather_steadystate_samples(mv_bounds_eu=mv_bounds, seed=seed)
            info['ss_sampling'] = samples['meta']
        except Exception as exc:  # sampling is best-effort; fall back gracefully
            info['ss_sampling_error'] = repr(exc)
            samples = None

    if et == 'static':
        if dyn_id is None:
            info['usable'] = False
            info['reason'] = 'no_dynamics_id'
            return None, info
        expert, ginfo = build_static_expert(
            dyn_id=dyn_id, obj_spec=obj_spec, obj_w=obj_w,
            mv_bounds=mv_bounds, cv_bounds=cv_bounds,
            mv_names=mv_names, cv_names=cv_names, dv_names=dv_names,
            samples=samples,
        )
        info.update(ginfo)
        return expert, info

    if et == 'nn':
        if samples is None:
            info['usable'] = False
            info['reason'] = 'no_ss_samples'
            return None, info
        expert, ninfo = build_nn_expert(
            obj_spec=obj_spec, obj_w=obj_w,
            mv_bounds=mv_bounds, cv_bounds=cv_bounds,
            cv_names=cv_names, samples=samples, seed=seed,
        )
        info.update(ninfo)
        return expert, info

    info['usable'] = False
    info['reason'] = f'unknown_expert_type:{et}'
    return None, info
