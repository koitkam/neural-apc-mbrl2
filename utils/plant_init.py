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
# Adaptive batch size from GPU memory headroom
# ---------------------------------------------------------------------------

def derive_batch_size(model_size: str,
                      paper_default: int = 16,
                      target_util: float = 0.65,
                      min_bs: int = 16, max_bs: int = 256,
                      horizon: int = 42,
                      horizon_ref: int = 42,
                      seq_len: int = 64,
                      seq_len_ref: int = 64,
                      lookback: int = 64,
                      lookback_ref: int = 64) -> Dict[str, Any]:
    """Pick a batch size that uses ~``target_util`` of the GPU.

    Reference per-sample peak memory at:
      model_size=M, seq_len=64, lookback=64, horizon=42, bf16, manual attn
        S ≈ 370 MB,  M ≈ 540 MB,  L ≈ 1050 MB

    Cross-sim recalibration 2026-05-24 (tools/gpu_calibrate.py): measured
    actual wm fwd+bwd peaks on three plants at their derived L configs
    (bs=16, seq=64–128, lb=82–165, hz=15, SDPA bf16):
      * test_sim       L: 989 MB/sample (predicted 545 → 1.81× under)
      * distillation   L: 1032 MB/sample (predicted 749 → 1.38× under)
      * softsensor_lab L: 488 MB/sample (predicted 186 → 2.62× under)
    Backed-out base_L per sim varies 1019–1939 MB because the linear
    seq×lookback×horizon scaling under-models attention KV / Z-token
    overhead for shorter contexts. We pick base_L=1050 (≈ test_sim
    measurement + small margin) and pair it with a conservative
    ``target_util=0.65`` headroom so all three sims land at bs≤32 with
    predicted peaks ≤16 GiB on a 22 GiB A10. Prior baselines (S=260,
    M=380, L=740) under-predicted by 1.4-2.6× across sims.

    Per-sample memory is scaled by:
      * ``seq_len / seq_len_ref``    (linear — WM token sequence length)
      * ``lookback / lookback_ref``  (linear — encoder + GRU context)
      * ``horizon / horizon_ref``    (linear — imagined-trajectory graph)
      * ``speed_factor``             (~0.55 with SDPA, ~0.30 with +compile)

    On CPU or no-CUDA we fall back to the paper default. Batch is snapped
    to powers of two in ``[paper_default, max_bs]`` so the recipe stays a
    strict superset of the paper.

    Env-var overrides (in precedence order):
      * ``DREAMER_MAX_BS``       — hard cap on the picked batch
      * ``DREAMER_TARGET_UTIL``  — GPU-headroom fraction (default 0.65)
      * ``DREAMER_PER_BATCH_MB`` — explicit per-sample baseline (advanced)
    """
    import os
    env_util = os.environ.get('DREAMER_TARGET_UTIL', '').strip()
    if env_util:
        try:
            target_util = max(0.1, min(0.95, float(env_util)))
        except ValueError:
            pass
    env_max_bs = os.environ.get('DREAMER_MAX_BS', '').strip()
    if env_max_bs:
        try:
            max_bs = max(min_bs, int(env_max_bs))
        except ValueError:
            pass
    base_per_batch_mb = {'S': 370, 'M': 540, 'L': 1050}.get(model_size, 540)
    # When the fast attention + compile paths are on, peak memory is roughly
    # 30% of the manual+eager baseline (measured: M-size W-M fwd+bwd drops
    # from ~290 MB/sample to ~70 MB/sample with bf16+SDPA+compile). Scale
    # the per-batch budget down so the auto-derived batch size grows to
    # match the freed headroom.
    # The fast attention + compile paths drop peak memory to ~30% of the
    # manual+eager baseline. Default policy (2026-05-12): SDPA is on
    # whenever a CUDA device is available unless explicitly disabled via
    # DREAMER_FAST_ATTN=0/manual.
    _fast_env = os.environ.get('DREAMER_FAST_ATTN', '').strip().lower()
    if _fast_env in ('0', 'false', 'manual', 'off'):
        fast_on = False
    elif _fast_env in ('1', 'true', 'sdpa', 'on'):
        fast_on = True
    else:
        try:
            import torch as _torch
            fast_on = bool(_torch.cuda.is_available())
        except Exception:
            fast_on = False
    compile_on = os.environ.get('DREAMER_COMPILE', '').strip() in ('1','true','True')
    speed_factor = 1.0
    if fast_on:
        speed_factor *= 0.55
    if compile_on:
        speed_factor *= 0.55
    base_per_batch_mb = float(base_per_batch_mb) * speed_factor
    # Optional explicit override (advanced).
    env_pb = os.environ.get('DREAMER_PER_BATCH_MB', '').strip()
    if env_pb:
        try:
            base_per_batch_mb = max(8.0, float(env_pb))
        except ValueError:
            pass
    seq_scale = max(1, int(seq_len)) / max(1, int(seq_len_ref))
    lb_scale = max(1, int(lookback)) / max(1, int(lookback_ref))
    hz_scale = max(1, int(horizon)) / max(1, int(horizon_ref))
    per_batch_mb = float(base_per_batch_mb) * seq_scale * lb_scale * hz_scale
    info: Dict[str, Any] = {'model_size': model_size,
                            'per_batch_mb': per_batch_mb,
                            'per_batch_base_mb': base_per_batch_mb,
                            'horizon': horizon, 'horizon_ref': horizon_ref,
                            'seq_len': seq_len, 'seq_len_ref': seq_len_ref,
                            'lookback': lookback, 'lookback_ref': lookback_ref,
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
