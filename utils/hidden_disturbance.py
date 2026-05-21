"""Hidden, dynamics-respecting unmeasured-CV disturbance process.

Replaces the step-shaped ``unmeasured_cv`` events from
``training_disturbance.py`` with a per-episode Ornstein-Uhlenbeck (OU)
process that injects a smoothly evolving bias into each CV state
channel.  The OU state is **never exposed** to the agent or the world
model — it is a true unmeasured disturbance, exactly mirroring the
unobservable upstream upsets a deployed APC controller must reject.

Why this exists
---------------
Step disturbances are mathematically Diracs in the dynamics: the world
model has zero observable signal predicting the spike, so it incurs a
loss floor that no amount of training can drive away.  Empirically (run
p33, WM-only, 1M env-steps) ``sf_loss_flow`` plateaus at ~1.15
regardless of training time when the disturbance schedule emits
unmeasured CV steps.

With a hidden OU process the disturbance evolves on a timescale
``tau_dist`` that the WM CAN model implicitly via its lookback context
(transformer's posterior gradually picks up on the unexplained CV
drift).  The driving noise ``w_t`` is small and white, so the
unlearnable-noise floor is bounded by ``Var(w_t)`` per step, not the
event amplitude.

Sizing rationale
----------------
- ``tau_dist`` ~ Uniform(0.15, 0.5) * tau_dom: well below the dominant
  plant time-constant so the disturbance is distinguishable from plant
  dynamics, and ≥ a few sample periods so it is representable.
- Steady-state amplitude ≤ ``amp_frac * mv_authority_cv``: capped at a
  fraction of the agent's MV-to-CV authority so the policy always has
  headroom to reject.  Default ``amp_frac=0.10``.
- Per-episode Bernoulli toggle: with probability ``DREAMER_DISTURBANCE_PROB``
  the episode has this hidden disturbance; otherwise it is clean.
  Recommended split: 0.3 during WM-training phases (P1/P2), 0.5 during
  agent-training phase (P3).  Set via env vars
  ``DREAMER_DISTURBANCE_PROB_WM`` and ``DREAMER_DISTURBANCE_PROB_AGENT``.

This module is intentionally sim-agnostic: it only touches CV channels
identified through ``sim.cv_indices`` and respects
``sim.state_is_normalized`` for unit handling.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

import numpy as np


def hidden_disturbance_enabled(default: bool = True) -> bool:
    """Whether the hidden CV disturbance is active.

    Default: ON.  The legacy step-shaped unmeasured-CV events are gone
    (run p33 showed they capped ``sf_loss_flow`` at ~1.15 by being Dirac
    spikes the WM has no observation to predict).  Off-switch only:
    set ``DREAMER_HIDDEN_DISTURBANCE=0`` to disable for ablations.
    """
    raw = os.environ.get('DREAMER_HIDDEN_DISTURBANCE')
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() not in {'0', 'false', 'no', 'off', ''}


def get_phase_disturbance_prob(phase: int) -> float:
    """Return the per-episode probability that hidden disturbance fires.

    Defaults: 0.3 in P1/P2 (WM training), 0.5 in P3 (critic+actor).
    Override via env vars ``DREAMER_DISTURBANCE_PROB_WM`` and
    ``DREAMER_DISTURBANCE_PROB_AGENT``.
    """
    if int(phase) >= 3:
        raw = os.environ.get('DREAMER_DISTURBANCE_PROB_AGENT', '0.5')
        default = 0.5
    else:
        raw = os.environ.get('DREAMER_DISTURBANCE_PROB_WM', '0.3')
        default = 0.3
    try:
        return float(np.clip(float(raw), 0.0, 1.0))
    except Exception:
        return default


class HiddenDisturbanceProcess:
    """Per-episode OU process injecting a hidden bias into CV channels.

    State (per CV channel):
        d_{t+1} = (1 - alpha) * d_t + sigma_w * sqrt(2*alpha) * eps_t,
        eps_t ~ N(0, 1)
    With ``alpha = sample_rate / tau_dist`` this yields a stationary
    Gaussian with std = ``sigma_w`` and autocorrelation time ``tau_dist``.
    """

    def __init__(
        self,
        rng: np.random.Generator,
        sim,
        *,
        tau_dom: float,
        sample_rate: float,
        amp_frac: float = 0.10,
        tau_frac_range: tuple = (0.15, 0.5),
        identifier_ctx: Optional[Dict] = None,
    ) -> None:
        self.sim = sim
        cv_indices = [int(i) for i in list(getattr(sim, 'cv_indices', []))]
        if not cv_indices:
            self.cv_indices: List[int] = []
            self.amp: np.ndarray = np.zeros(0, dtype='float64')
            self.alpha: np.ndarray = np.zeros(0, dtype='float64')
            self.d: np.ndarray = np.zeros(0, dtype='float64')
            self.tau_dist: np.ndarray = np.zeros(0, dtype='float64')
            self._is_normalized: bool = bool(getattr(sim, 'state_is_normalized', False))
            return

        self.cv_indices = cv_indices
        n_cv = len(cv_indices)
        self._is_normalized = bool(getattr(sim, 'state_is_normalized', False))

        # Per-channel CV span (engineering units; or [0,1] if normalized).
        cv_ranges = list(getattr(sim, 'cv_normalization_ranges', []))
        spans = np.ones(n_cv, dtype='float64')
        for pos in range(n_cv):
            if 0 <= pos < len(cv_ranges):
                lo, hi = float(cv_ranges[pos][0]), float(cv_ranges[pos][1])
                if hi > lo:
                    spans[pos] = hi - lo

        # MV->CV authority cap (engineering units) — disturbance amp ≤
        # ``amp_frac`` of what the agent can rebalance by moving the MV
        # across its op-band.  Falls back to ``amp_frac * cv_span`` if
        # authority is unknown.
        mv_authority_cv = 0.0
        try:
            from utils.training_disturbance import compute_mv_authority_to_cv
            mv_authority_cv = float(
                compute_mv_authority_to_cv(sim, identifier_ctx) or 0.0
            )
        except Exception:
            mv_authority_cv = 0.0

        # Per-channel steady-state amplitude (1 standard deviation).
        amp_eng = np.empty(n_cv, dtype='float64')
        for pos in range(n_cv):
            cap_authority = (
                amp_frac * mv_authority_cv if mv_authority_cv > 1e-9 else float('inf')
            )
            cap_span = amp_frac * spans[pos]
            amp_eng[pos] = min(cap_span, cap_authority)
        # If the sim exposes its state in normalized [0,1] space, convert
        # the engineering-units amplitude to normalized units so adding
        # ``d`` to the CV state has the intended physical magnitude.
        if self._is_normalized:
            self.amp = amp_eng / np.maximum(spans, 1e-9)
        else:
            self.amp = amp_eng

        # Per-channel tau_dist sampled per episode within configured band.
        lo_frac, hi_frac = float(tau_frac_range[0]), float(tau_frac_range[1])
        tau_doms = max(1e-6, float(tau_dom))
        self.tau_dist = rng.uniform(lo_frac, hi_frac, size=n_cv) * tau_doms
        # alpha = sample_rate / tau_dist, clamped to (0, 1].
        sr = max(1e-6, float(sample_rate))
        self.alpha = np.clip(sr / np.maximum(self.tau_dist, 3.0 * sr), 1e-3, 1.0)

        # Initialize OU at the stationary distribution so episode-start
        # behaviour matches mid-episode.
        self.d = rng.normal(0.0, 1.0, size=n_cv) * self.amp
        self._rng = rng

    def is_empty(self) -> bool:
        return len(self.cv_indices) == 0

    def step(self, state: np.ndarray) -> None:
        """Advance the OU one step and add it to the CV channels of ``state``.

        Modifies ``state`` in place.
        """
        if self.is_empty():
            return
        # OU update with variance-preserving driving noise:
        #     d <- (1-alpha) d + sigma_w * sqrt(2*alpha - alpha^2) eps
        # The factor sqrt(2*alpha - alpha^2) keeps Var(d) = amp^2 at
        # steady state for any alpha in (0, 1].
        a = self.alpha
        eps = self._rng.normal(0.0, 1.0, size=self.d.shape)
        sigma_drive = np.sqrt(np.maximum(2.0 * a - a * a, 1e-12)) * self.amp
        self.d = (1.0 - a) * self.d + sigma_drive * eps
        for pos, idx in enumerate(self.cv_indices):
            state[int(idx)] = float(state[int(idx)]) + float(self.d[pos])
        # If the sim publishes state in normalized [0,1] space, clip back
        # in to keep downstream consumers (encoder, reward) well-defined.
        if self._is_normalized:
            for idx in self.cv_indices:
                state[int(idx)] = float(np.clip(state[int(idx)], 0.0, 1.0))

    def summary(self) -> Dict:
        return {
            'cv_indices': list(self.cv_indices),
            'tau_dist': self.tau_dist.tolist(),
            'amp': self.amp.tolist(),
            'alpha': self.alpha.tolist(),
        }


def maybe_build_hidden_disturbance(
    rng: np.random.Generator,
    sim,
    *,
    tau_dom: float,
    sample_rate: float,
    prob: float,
    identifier_ctx: Optional[Dict] = None,
    amp_frac: float = 0.10,
    force: bool = False,
) -> Optional[HiddenDisturbanceProcess]:
    """Bernoulli-toggle the per-episode hidden disturbance process.

    Returns ``None`` for clean episodes so the WM also sees some clean
    data and can sharpen its base dynamics estimate.  Set ``force=True``
    (validation path) to bypass the Bernoulli toggle and always build.
    """
    if not hidden_disturbance_enabled(default=True):
        return None
    if not force:
        if prob <= 0.0:
            return None
        if rng.uniform() >= float(prob):
            return None
    proc = HiddenDisturbanceProcess(
        rng=rng,
        sim=sim,
        tau_dom=float(tau_dom),
        sample_rate=float(sample_rate),
        amp_frac=float(amp_frac),
        identifier_ctx=identifier_ctx,
    )
    if proc.is_empty():
        return None
    return proc
