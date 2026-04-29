"""Plant-tied derivations for trainer hyperparameters.

These helpers replace fixed paper defaults with values that adapt to the
simulator's identified dynamics and channel complexity.  They are used by
both ``workflow/run.py`` (single-run) and ``workflow/runner.py`` (BO
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
                      tau_dom: float, tau_fast: float) -> str:
    """Map complexity score to {S, M, L}.

    Thresholds chosen so:
      - 1×1 SISO plants with state_dim ≤ 4 → S
      - moderate MIMO (≤ 4×4, single-scale) → M
      - large MIMO or strongly multi-scale → L
    """
    score = complexity_score(
        n_mv=n_mv, n_cv=n_cv, n_dv=n_dv, state_dim=state_dim,
        tau_dom=tau_dom, tau_fast=tau_fast,
    )
    if score <= 4.0:
        return 'S'
    if score <= 12.0:
        return 'M'
    return 'L'


# ---------------------------------------------------------------------------
# Sequence length for WM training (must cover ≥ 1 settling time)
# ---------------------------------------------------------------------------

def derive_seq_len(tau_dom: float, dead_time: float, sample_rate: int,
                   paper_default: int = 64, max_len: int = 256) -> int:
    """Sequence length for world-model training.

    Paper default is 64 (Hafner et al. 2024 §C).  We keep that as a floor
    and only grow it to cover at least one full settling time
    (``3τ + θ`` samples) when the plant is slow relative to the sample rate.
    """
    sr = max(1, int(sample_rate))
    settling_samples = int(math.ceil((3.0 * float(tau_dom) + float(dead_time)) / sr))
    return int(min(max_len, max(paper_default, settling_samples)))


# ---------------------------------------------------------------------------
# Adaptive batch size from GPU memory headroom
# ---------------------------------------------------------------------------

def derive_batch_size(model_size: str,
                      paper_default: int = 16,
                      target_util: float = 0.5,
                      min_bs: int = 16, max_bs: int = 128) -> Dict[str, Any]:
    """Pick a batch size that uses ~``target_util`` of the GPU.

    Empirical per-batch peak memory (test_sim, seq_len=64, horizon=42, bf16):
      S ≈ 220 MB / batch
      M ≈ 330 MB / batch
      L ≈ 800 MB / batch

    On CPU or no-CUDA we fall back to the paper default.  We always use
    powers of two for batch and clamp to ``[paper_default, max_bs]`` so
    the recipe stays a strict superset of the paper.
    """
    per_batch_mb = {'S': 220, 'M': 330, 'L': 800}.get(model_size, 330)
    info: Dict[str, Any] = {'model_size': model_size,
                            'per_batch_mb': per_batch_mb,
                            'paper_default': paper_default,
                            'target_util': target_util,
                            'min_bs': min_bs, 'max_bs': max_bs}
    try:
        import torch  # local import to keep utils import-light
        if not torch.cuda.is_available():
            info.update({'batch_size': paper_default,
                         'source': 'cpu_fallback', 'gpu_total_gb': 0.0})
            return info
        free_b, total_b = torch.cuda.mem_get_info(0)
        total_gb = total_b / (1024 ** 3)
        budget_mb = target_util * total_b / (1024 ** 2)
        raw_bs = max(paper_default, int(budget_mb // per_batch_mb))
        # snap to nearest power of two, clamped to [paper_default, max_bs]
        bs_pow = 1 << max(int(math.log2(paper_default)),
                          int(math.floor(math.log2(max(raw_bs, paper_default)))))
        bs = int(min(max_bs, max(min_bs, bs_pow)))
        info.update({'batch_size': bs, 'source': 'auto:gpu_headroom',
                     'gpu_total_gb': total_gb,
                     'budget_mb': budget_mb, 'raw_bs': raw_bs})
        return info
    except Exception as e:
        info.update({'batch_size': paper_default, 'source': f'fallback:{e!r}',
                     'gpu_total_gb': 0.0})
        return info


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
    )
    score = complexity_score(
        n_mv=n_mv, n_cv=n_cv, n_dv=n_dv, state_dim=state_dim,
        tau_dom=tau_dom, tau_fast=tau_fast,
    )
    seq_len = derive_seq_len(tau_dom, dead_dom, sr)

    return {
        'sample_rate': int(sr),
        'sample_rate_source': sr_source,
        'model_size': model_size,
        'complexity_score': float(score),
        'seq_len': int(seq_len),
        'inputs': {
            'tau_dom': tau_dom, 'dead_dom': dead_dom,
            'tau_fast': tau_fast, 'dead_fast': dead_fast,
            'n_mv': n_mv, 'n_cv': n_cv, 'n_dv': n_dv,
            'state_dim': state_dim,
        },
    }
