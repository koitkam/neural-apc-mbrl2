"""Plant-tied derivations for trainer hyperparameters.

These helpers replace fixed paper defaults with values that adapt to the
simulator's identified dynamics and channel complexity.  They are used by
both ``workflow/single_run.py`` (single-run) and ``workflow/bo_runner.py`` (BO
seed for the first trial).

Paper-defaults are kept as floors / minimums — we only *grow* values when
the plant clearly demands more capacity / context.
"""

from __future__ import annotations

import math
from typing import Dict, Any


# ---------------------------------------------------------------------------
# Sample rate
# ---------------------------------------------------------------------------

def derive_sample_rate(tau_fast: float, dead_fast: float,
                       default: int = 5,
                       min_sr: int = 1, max_sr: int = 60) -> int:
    """Choose a sample rate that resolves the *fastest* identified channel.

    Two criteria, both must be satisfied:
      - ≥ 10 samples per fastest time constant     (sr ≤ τ_fast / 10)
      - ≥ 2 samples within the fastest dead time   (sr ≤ θ_fast / 2)

    The smaller (tighter) of the two wins.  Falls back to ``default`` when
    no identified value is positive.
    """
    if not (tau_fast and tau_fast > 0):
        return int(default)
    sr_tau = max(min_sr, int(round(tau_fast / 10.0)))
    if dead_fast and dead_fast > 0:
        sr_dead = max(min_sr, int(round(dead_fast / 2.0)))
        sr = min(sr_tau, sr_dead)
    else:
        sr = sr_tau
    return int(max(min_sr, min(max_sr, sr)))


# ---------------------------------------------------------------------------
# Model size from channel + dynamics complexity
# ---------------------------------------------------------------------------

def complexity_score(*, n_mv: int, n_cv: int, n_dv: int, state_dim: int,
                     tau_dom: float, tau_fast: float) -> float:
    """Composite plant complexity score (used to select model size).

    Components:
      - channel count          : (n_mv + n_dv) * n_cv
      - multiscale dynamics    : 2 * (τ_dom / τ_fast - 1)
      - state-dim contribution : 0.5 * state_dim
    """
    n_mv = max(0, int(n_mv))
    n_cv = max(0, int(n_cv))
    n_dv = max(0, int(n_dv))
    state_dim = max(0, int(state_dim))
    tau_dom = float(tau_dom) if tau_dom else 0.0
    tau_fast = float(tau_fast) if tau_fast else tau_dom
    n_channels = max(1, n_mv + n_dv) * max(1, n_cv)
    multiscale = (tau_dom / tau_fast) if tau_fast > 0 else 1.0
    return float(n_channels) + 2.0 * max(0.0, multiscale - 1.0) + 0.5 * state_dim


def derive_model_size(*, n_mv: int, n_cv: int, n_dv: int, state_dim: int,
                      tau_dom: float, tau_fast: float,
                      sample_rate: int = 1) -> str:
    """Map complexity score to {S, M, L}.

    Thresholds chosen so:
      - 1×1 SISO plants with state_dim ≤ 4 → S
      - moderate MIMO (≤ 4×4, single-scale) → M
      - large MIMO or strongly multi-scale → L

    Long-τ escalation (2026-05-06): plants whose normalized settling
    time (``3τ + θ ≈ 3·tau_dom``) covers many sample steps need more
    transformer capacity to model the long-range dependency, even
    when the channel count is small.  We escalate one tier when
    ``settling_samples ≥ 25`` (e.g. test_sim: τ=53, sr=4 ⇒ 40 samples
    ⇒ S→M).  This covers the under-capacity case diagnosed in
    run_p0adapt where SISO test_sim was assigned S despite τ=53.
    """
    score = complexity_score(
        n_mv=n_mv, n_cv=n_cv, n_dv=n_dv, state_dim=state_dim,
        tau_dom=tau_dom, tau_fast=tau_fast,
    )
    if score <= 4.0:
        size = 'S'
    elif score <= 12.0:
        size = 'M'
    else:
        size = 'L'

    sr = max(1, int(sample_rate))
    settling_samples = (3.0 * float(max(0.0, tau_dom))) / sr
    if settling_samples >= 25.0:
        size = {'S': 'M', 'M': 'L', 'L': 'L'}[size]
    return size


# ---------------------------------------------------------------------------
# Sequence length for WM training (must cover ≥ 1 settling time)
# ---------------------------------------------------------------------------

def derive_seq_len(tau_dom: float, dead_time: float, sample_rate: int,
                   paper_default: int = 64, max_len: int = 256) -> int:
    """Sequence length for world-model training.

    Paper default is 64 (Hafner et al. 2024 §C).  We keep that as a floor
    and grow it to cover ``≥2 × settling`` (i.e. enough room within a
    single training sub-sequence for the WM to see *both* a transient
    response *and* the relaxation back to a steady state).  Output is
    snapped up to the next power of two (≥64) so GPU shapes stay clean,
    capped at ``max_len`` to bound memory.

    Rationale: with ``seq_len = settling``, the WM frequently trains on
    windows that contain only the ramp-up half of a step response, never
    seeing the gain converge.  Doubling to ``2 × settling`` guarantees
    that every training window contains at least one full step
    transient, which is what the dynamics transformer needs to learn
    open-loop gain (not just slope).
    """
    sr = max(1, int(sample_rate))
    settling_samples = int(math.ceil((3.0 * float(tau_dom) + float(dead_time)) / sr))
    target = 2 * max(1, settling_samples)
    n = max(int(paper_default), target)
    # snap up to next power of two (≥6 ⇒ ≥64) for clean attention shapes
    pw = 1 << max(6, int(math.ceil(math.log2(max(2, n)))))
    return int(min(max_len, pw))


# ---------------------------------------------------------------------------
# Denoising granularity (k_max) for the shortcut-forcing dynamics module
# ---------------------------------------------------------------------------

def derive_k_max(model_size: str, complexity_score: float,
                 paper_default: int = 4) -> int:
    """Number of shortcut-forcing denoising sub-steps (``d_min = 1/k_max``).

    ``k_max`` controls how many iterative denoising steps the dynamics
    transformer can take when ``imagine_next_z`` rolls out a future z.
    Larger ``k_max`` = finer denoising trajectory = better latent
    fidelity, at the cost of K× inference compute per imagined step.

    No clean closed-form mapping exists from plant dynamics to optimal
    ``k_max``; we tie it to model capacity (S/M → 4, L → 8) and bump it
    one tier for very complex plants (``complexity_score ≥ 5``) where
    the dynamics transformer is being asked to fit a richer manifold.
    Hard cap at 16 to keep the imagination roll-out cost bounded.
    """
    base = {'S': 4, 'M': 4, 'L': 8}.get(model_size, int(paper_default))
    if float(complexity_score) >= 5.0:
        base = min(16, base * 2)
    return int(max(1, base))


# ---------------------------------------------------------------------------
# Step budgets (BO trial / final retrain)
# ---------------------------------------------------------------------------

def derive_step_budgets(*, episode_length: int, complexity_score: float,
                        trial_eps_base: int = 40,
                        final_eps_multiplier: int = 10,
                        trial_floor: int = 50_000,
                        trial_cap: int = 250_000,
                        final_floor: int = 200_000,
                        final_cap: int = 2_000_000) -> Dict[str, Any]:
    """Plant-tied trial / final step budgets for BO.

    Unit of work is *episodes*, scaled by complexity:

        trial_eps = trial_eps_base * max(1.0, complexity_score / 4)
        final_eps = final_eps_multiplier * trial_eps

    then converted to steps via ``episode_length`` and clamped to the
    floor / cap envelope.  Floors guarantee enough buffer fill + actor
    settling on simple plants; caps protect against runaway budgets on
    very long episodes.
    """
    ep_len = max(1, int(episode_length))
    factor = max(1.0, float(complexity_score) / 4.0)
    trial_eps = int(round(trial_eps_base * factor))
    final_eps = int(round(final_eps_multiplier * trial_eps))
    trial_steps_raw = trial_eps * ep_len
    final_steps_raw = final_eps * ep_len
    trial_steps = int(max(trial_floor, min(trial_cap, trial_steps_raw)))
    final_steps = int(max(final_floor, min(final_cap, final_steps_raw)))
    return {
        'trial_steps': trial_steps,
        'final_steps': final_steps,
        'trial_episodes_target': trial_eps,
        'final_episodes_target': final_eps,
        'episode_length': ep_len,
        'complexity_factor': factor,
        'trial_steps_raw': trial_steps_raw,
        'final_steps_raw': final_steps_raw,
        'source': 'auto:plant_tied',
        'envelope': {
            'trial_floor': trial_floor, 'trial_cap': trial_cap,
            'final_floor': final_floor, 'final_cap': final_cap,
        },
    }


# ---------------------------------------------------------------------------
# Convenience: full derivation block from dynamics report + sim metadata
# ---------------------------------------------------------------------------

def derive_all(dyn_report: Dict[str, Any], sim_meta: Dict[str, Any],
               *, sample_rate_override: int = 0) -> Dict[str, Any]:
    """Compute (sample_rate, model_size, seq_len, score) plus the inputs.

    ``sample_rate_override > 0`` forces that sample rate (useful when the
    simulator is hard-coded to a fixed scan rate); otherwise it is
    derived from the fastest identified dynamics.
    """
    tau_dom = float(dyn_report.get('tau_dominant_identified',
                                    dyn_report.get('tau_dominant', 0.0)) or 0.0)
    dead_dom = float(dyn_report.get('dead_time_identified',
                                     dyn_report.get('dead_time', 0.0)) or 0.0)
    tau_fast = float(dyn_report.get('tau_fastest_identified', tau_dom) or tau_dom)
    dead_fast = float(dyn_report.get('dead_time_fastest_identified', dead_dom) or dead_dom)

    n_mv = len((sim_meta or {}).get('mv_indices', []) or [])
    n_cv = len((sim_meta or {}).get('cv_indices', []) or [])
    n_dv = len((sim_meta or {}).get('dv_indices', []) or [])
    state_dim = int((sim_meta or {}).get('state_dim') or 0)

    if sample_rate_override and int(sample_rate_override) > 0:
        sr = int(sample_rate_override)
        sr_source = 'override'
    else:
        sr = derive_sample_rate(tau_fast, dead_fast)
        sr_source = 'auto:tau_fast/10_or_dead_fast/2'

    model_size = derive_model_size(
        n_mv=n_mv, n_cv=n_cv, n_dv=n_dv, state_dim=state_dim,
        tau_dom=tau_dom, tau_fast=tau_fast,
        sample_rate=sr,
    )
    score = complexity_score(
        n_mv=n_mv, n_cv=n_cv, n_dv=n_dv, state_dim=state_dim,
        tau_dom=tau_dom, tau_fast=tau_fast,
    )
    seq_len = derive_seq_len(tau_dom, dead_dom, sr)
    k_max = derive_k_max(model_size, score)

    return {
        'sample_rate': int(sr),
        'sample_rate_source': sr_source,
        'model_size': model_size,
        'complexity_score': float(score),
        'seq_len': int(seq_len),
        'k_max': int(k_max),
        'inputs': {
            'tau_dom': tau_dom, 'dead_dom': dead_dom,
            'tau_fast': tau_fast, 'dead_fast': dead_fast,
            'n_mv': n_mv, 'n_cv': n_cv, 'n_dv': n_dv,
            'state_dim': state_dim,
        },
    }
