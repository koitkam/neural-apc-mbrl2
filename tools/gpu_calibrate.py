"""Cross-sim GPU per-sample memory calibration.

Builds the actual training model (same code path as `workflow.single_run`)
for a given simulation, runs ONE forward + backward of `world_model_loss`
at bs=4 on synthetic data, and measures peak GPU memory.  Reports
measured per-sample MB and recommends an empirical batch size.

Usage::

    python -m tools.gpu_calibrate --simulation-dir simulation/test_sim

Skips dynamics ID if `--plant-id` points to an existing plant_id.json
from a prior run (re-uses tau/dead_time/lookback).

Also exposes the public helpers consumed by `workflow.single_run` /
`workflow.bo_runner` for on-the-fly empirical batch sizing:
``measure_per_sample_mb``, ``pick_batch_size_empirical``, and the
cached high-level ``pick_batch_size_for_plant``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import torch


# ───────────────────────────────────────────────────────────────────────────
# Public helpers (also used by workflow.single_run for on-the-fly empirical
# batch-size selection — see DREAMER_GPU_CALIBRATE).
# ───────────────────────────────────────────────────────────────────────────

def measure_per_sample_mb(cfg, bs_probe: int = 4) -> dict:
    """Run one fwd+bwd of ``world_model_loss`` on synthetic data and
    return the measured per-sample GPU peak in MB.

    ``cfg`` must be a populated ``training.train.TrainConfig`` with
    ``obs_dim``, ``action_dim``, ``seq_len``, ``lookback``, ``horizon``,
    and architecture fields set.  ``cfg.batch_size`` is ignored — we use
    ``bs_probe`` for the synthetic batch.

    Cost: ~5-15 s on an A10 (model build + warmup + measured pass).
    Returns ``{'peak_mb': float, 'per_sample_mb': float, 'bs_probe': int}``
    or ``{'error': str}`` on CPU / OOM.
    """
    if not torch.cuda.is_available():
        return {'error': 'no_cuda'}
    from training.train import build_model, world_model_loss
    device = torch.device('cuda')
    obs_dim = int(getattr(cfg, 'obs_dim', 0) or 0)
    action_dim = int(getattr(cfg, 'action_dim', 0) or 0)
    if obs_dim <= 0 or action_dim <= 0:
        return {'error': f'cfg.obs_dim/action_dim not set ({obs_dim},{action_dim})'}
    seq_len = int(cfg.seq_len)
    # Phase 2 (2026-05-24): replay buffer no longer carries the L axis;
    # ``world_model_loss`` consumes obs of shape (B, T, D).  The probe
    # tensor must match or unpacking fails with
    # ``ValueError: too many values to unpack (expected 3)`` and the
    # caller falls back to the paper-default batch size.
    mtp_length = max(1, int(getattr(cfg, 'mtp_length', 8)))
    try:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        model = build_model(cfg).to(device)
        model.train()
        B = int(bs_probe)
        obs = torch.randn(B, seq_len, obs_dim, device=device)
        act = torch.randn(B, seq_len, action_dim, device=device).clamp_(-1, 1)
        fut_rew = torch.randn(B, seq_len, mtp_length, device=device)
        batch = {'obs': obs, 'act': act, 'fut_rew': fut_rew,
                 'rew': torch.randn(B, seq_len, device=device)}
        # Warmup pass.
        losses, _, _ = world_model_loss(model, batch, cfg)
        total = sum(v for v in losses.values()
                    if isinstance(v, torch.Tensor) and v.requires_grad)
        total.backward()
        model.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        # Measured pass.
        losses, _, _ = world_model_loss(model, batch, cfg)
        total = sum(v for v in losses.values()
                    if isinstance(v, torch.Tensor) and v.requires_grad)
        total.backward()
        torch.cuda.synchronize()
        peak_b = torch.cuda.max_memory_allocated()
        peak_mb = peak_b / (1024 ** 2)
        per_sample_mb = peak_mb / max(1, B)
        # Cleanup.
        del model, obs, act, fut_rew, batch, losses, total
        torch.cuda.empty_cache()
        return {'peak_mb': float(peak_mb),
                'per_sample_mb': float(per_sample_mb),
                'bs_probe': B}
    except torch.cuda.OutOfMemoryError as e:
        torch.cuda.empty_cache()
        return {'error': f'oom@bs_probe={bs_probe}: {e}'}
    except Exception as e:
        torch.cuda.empty_cache()
        return {'error': f'{type(e).__name__}: {e}'}


def pick_batch_size_empirical(*, model_size: str, seq_len: int, lookback: int,
                               horizon: int, k_max: int, sample_rate: int,
                               obs_dim: int, action_dim: int,
                               paper_default: int = 16, max_bs: int = 512,
                               target_util: float = 0.65,
                               bs_probe: int = 4,
                               wm_overhead_factor: float = 1.0) -> dict:
    """Empirically size the batch by measuring actual WM fwd+bwd peak
    memory at ``bs_probe`` and projecting linearly.

    The probe builds the **actual world-model backbone** that the run will
    use: ``world_model_type`` and the ``rssm_*`` dims are read from the
    ``DREAMER_WORLD_MODEL_TYPE`` / ``DREAMER_RSSM_*`` env vars (falling
    back to the ``TrainConfig`` defaults, currently the DreamerV3 RSSM).
    This keeps the memory measurement faithful when the backbone or its
    latent sizes are overridden at launch.

    ``wm_overhead_factor`` (>=1.0) inflates the measured **WM-only** peak
    to reserve for memory the probe does NOT see: actor+critic+optimizer
    state and the Phase-3 imagination rollout.  Default 1.0 = WM-only;
    pass e.g. 1.30 to reserve 30% for actor/critic/opt + P3 imagination.

    ``max_bs`` default is 512 (the RSSM backbone is ~4x lighter per sample
    than the legacy SF-transformer, so the ceiling no longer needs to clip
    at 256); on a memory-bound card the ``target_util`` budget still pins
    the choice well below this.

    Returns a ``bs_info`` dict with ``batch_size``, ``source``,
    ``per_sample_mb_measured``, ``predicted_peak_gib``, etc.  On CPU or
    probe failure, falls back to ``paper_default`` (no formula).
    """
    from training.train import TrainConfig
    from workflow.bo_runner import MODEL_SIZE_PRESETS
    arch = MODEL_SIZE_PRESETS[model_size]
    cfg = TrainConfig(
        d_model=arch['d_model'], n_layers=arch['n_layers'],
        n_heads=arch['n_heads'], z_dim=arch['z_dim'],
        n_register=arch['n_register'], tok_hidden=arch['tok_hidden'],
        head_hidden=arch['head_hidden'],
        lookback=int(lookback), sample_rate=int(sample_rate),
        episode_length=1000, total_steps=1000,
        horizon=int(horizon), seq_len=int(seq_len), k_max=int(k_max),
        batch_size=int(bs_probe), out_dir='/tmp/gpu_calib_inplace',
        obs_dim=int(obs_dim), action_dim=int(action_dim),
    )
    # Mirror the world-model backbone the run will actually build so the
    # measured per-sample memory matches.  Env overrides take precedence
    # over the TrainConfig defaults (which already select the RSSM).
    wmt = (os.environ.get('DREAMER_WORLD_MODEL_TYPE', '').strip()
           or str(getattr(cfg, 'world_model_type', 'rssm')))
    cfg.world_model_type = wmt
    if wmt == 'rssm':
        def _envint(name: str, cur) -> int:
            v = os.environ.get(name, '').strip()
            try:
                return int(v) if v else int(cur)
            except ValueError:
                return int(cur)
        cfg.rssm_deter_dim = _envint('DREAMER_RSSM_DETER_DIM', cfg.rssm_deter_dim)
        cfg.rssm_n_categoricals = _envint(
            'DREAMER_RSSM_N_CATEGORICALS', cfg.rssm_n_categoricals)
        cfg.rssm_n_classes = _envint(
            'DREAMER_RSSM_N_CLASSES', cfg.rssm_n_classes)
        cfg.rssm_embed_dim = _envint('DREAMER_RSSM_EMBED_DIM', cfg.rssm_embed_dim)
        cfg.rssm_hidden_dim = _envint(
            'DREAMER_RSSM_HIDDEN_DIM', cfg.rssm_hidden_dim)
        # Mirror the #2 latent-overshooting knobs (P88) so the probe's
        # ``world_model_loss`` builds the SAME open-loop prior-rollout graph
        # the run will (B x max_starts x len GRU steps retained for backward
        # = the dominant new activation cost).  Without this the probe runs
        # overshoot OFF and under-measures per-sample memory -> oversized batch
        # near the OOM ceiling.  RSSM-only (SF no-ops the term).
        def _envfloat(name: str, cur) -> float:
            v = os.environ.get(name, '').strip()
            try:
                return float(v) if v else float(cur)
            except ValueError:
                return float(cur)
        cfg.wm_overshoot_coef = _envfloat(
            'DREAMER_WM_OVERSHOOT_COEF', cfg.wm_overshoot_coef)
        cfg.wm_overshoot_len = _envint(
            'DREAMER_WM_OVERSHOOT_LEN', cfg.wm_overshoot_len)
        cfg.wm_overshoot_max_starts = _envint(
            'DREAMER_WM_OVERSHOOT_MAX_STARTS', cfg.wm_overshoot_max_starts)
    info = {'model_size': model_size, 'seq_len': seq_len, 'lookback': lookback,
            'horizon': horizon, 'paper_default': paper_default,
            'target_util': target_util, 'min_bs': paper_default, 'max_bs': max_bs,
            'world_model_type': wmt,
            'wm_overhead_factor': wm_overhead_factor, 'bs_probe': bs_probe}
    # Env-var overrides.
    env_util = os.environ.get('DREAMER_TARGET_UTIL', '').strip()
    if env_util:
        try:
            target_util = max(0.1, min(0.95, float(env_util)))
            info['target_util'] = target_util
        except ValueError:
            pass
    env_max_bs = os.environ.get('DREAMER_MAX_BS', '').strip()
    if env_max_bs:
        try:
            max_bs = max(paper_default, int(env_max_bs))
            info['max_bs'] = max_bs
        except ValueError:
            pass
    if not torch.cuda.is_available():
        info.update({'batch_size': paper_default, 'source': 'cpu_fallback',
                     'gpu_total_gb': 0.0})
        return info
    meas = measure_per_sample_mb(cfg, bs_probe=bs_probe)
    if 'error' in meas:
        # Probe failed (e.g. OOM at bs_probe).  Fall back to the paper
        # default — no formula to rely on.
        info.update({'batch_size': paper_default,
                     'source': f'paper_default_fallback:{meas["error"]}',
                     'gpu_total_gb': 0.0})
        return info
    per_sample_mb = float(meas['per_sample_mb']) * float(wm_overhead_factor)
    free_b, total_b = torch.cuda.mem_get_info(0)
    gpu_total_gb = total_b / (1024 ** 3)
    budget_mb = target_util * total_b / (1024 ** 2)
    raw_bs = max(paper_default, int(budget_mb // max(1.0, per_sample_mb)))
    bs_pow = 1 << max(int(math.log2(paper_default)),
                      int(math.floor(math.log2(max(raw_bs, paper_default)))))
    bs = int(min(max_bs, max(paper_default, bs_pow)))
    info.update({
        'batch_size': bs, 'source': 'empirical:gpu_calibrate',
        'per_batch_mb': per_sample_mb,
        'per_sample_mb_measured': float(meas['per_sample_mb']),
        'peak_mb_at_probe': float(meas['peak_mb']),
        'gpu_total_gb': gpu_total_gb, 'budget_mb': budget_mb,
        'raw_bs': raw_bs,
        'predicted_peak_gib': bs * per_sample_mb / 1024,
    })
    return info


# Module-level cache so BO trials sharing the same (size, seq, lb, hz) don't
# pay the probe cost repeatedly.
_PROBE_CACHE: dict = {}
_OBS_DIM_CACHE: dict = {}


def _resolve_obs_action_dim(lookback: int, sample_rate: int,
                             seq_len: int, horizon: int, k_max: int,
                             episode_length: int) -> tuple[int, int]:
    """Build a throw-away APCEnv to discover ``(obs_dim, action_dim)``
    for the current SIMULATION_DIR / CONTROL_SETUP_JSON / env overrides.
    Cached per plant config so repeated BO trials don't rebuild the env.
    """
    key = (lookback, sample_rate, seq_len, horizon, k_max, episode_length,
           os.environ.get('SIMULATION_DIR', ''),
           os.environ.get('CONTROL_SETUP_JSON', ''))
    if key in _OBS_DIM_CACHE:
        return _OBS_DIM_CACHE[key]
    import numpy as _np
    from training.train import TrainConfig, APCEnv
    probe_cfg = TrainConfig(
        lookback=int(lookback), sample_rate=int(sample_rate),
        episode_length=int(episode_length), total_steps=1000,
        horizon=int(horizon), seq_len=int(seq_len), k_max=int(k_max),
        batch_size=4, out_dir='/tmp/_obs_dim_probe',
    )
    env = APCEnv(probe_cfg, _np.random.default_rng(0))
    od, ad = int(env.obs_dim), int(env.action_dim)
    del env
    _OBS_DIM_CACHE[key] = (od, ad)
    return od, ad


def pick_batch_size_for_plant(*, model_size: str, seq_len: int, lookback: int,
                               horizon: int, k_max: int, sample_rate: int,
                               episode_length: int,
                               paper_default: int = 16, max_bs: int = 512,
                               target_util: float = 0.65,
                               bs_probe: int = 4,
                               wm_overhead_factor: float = 1.0) -> dict:
    """High-level entry: discover ``(obs_dim, action_dim)`` from the
    current simulation env and pick a batch size empirically.  Result is
    cached on ``(model_size, seq_len, lookback, horizon)`` so repeated BO
    trials with identical shape skip the probe.  Returns the same
    ``bs_info`` dict as :func:`pick_batch_size_empirical`.
    """
    cache_key = (model_size, int(seq_len), int(lookback), int(horizon),
                 int(k_max), os.environ.get('SIMULATION_DIR', ''),
                 os.environ.get('DREAMER_MAX_BS', ''),
                 os.environ.get('DREAMER_TARGET_UTIL', ''),
                 os.environ.get('DREAMER_WORLD_MODEL_TYPE', ''),
                 os.environ.get('DREAMER_RSSM_DETER_DIM', ''),
                 os.environ.get('DREAMER_RSSM_N_CATEGORICALS', ''),
                 os.environ.get('DREAMER_RSSM_N_CLASSES', ''),
                 os.environ.get('DREAMER_WM_OVERSHOOT_COEF', ''),
                 os.environ.get('DREAMER_WM_OVERSHOOT_LEN', ''),
                 os.environ.get('DREAMER_WM_OVERSHOOT_MAX_STARTS', ''),
                 float(wm_overhead_factor))
    if cache_key in _PROBE_CACHE:
        info = dict(_PROBE_CACHE[cache_key])
        info['source'] = info.get('source', 'empirical:gpu_calibrate') + ':cached'
        return info
    od, ad = _resolve_obs_action_dim(lookback=lookback, sample_rate=sample_rate,
                                      seq_len=seq_len, horizon=horizon,
                                      k_max=k_max, episode_length=episode_length)
    info = pick_batch_size_empirical(
        model_size=model_size, seq_len=seq_len, lookback=lookback,
        horizon=horizon, k_max=k_max, sample_rate=sample_rate,
        obs_dim=od, action_dim=ad,
        paper_default=paper_default, max_bs=max_bs,
        target_util=target_util, bs_probe=bs_probe,
        wm_overhead_factor=wm_overhead_factor)
    info['obs_dim'] = od
    info['action_dim'] = ad
    _PROBE_CACHE[cache_key] = info
    return info


def _resolve_sim_dir(arg: str) -> Path:
    repo = Path(__file__).resolve().parent.parent
    p = Path(arg)
    if p.is_absolute() and p.exists():
        return p
    cand = repo / arg
    if cand.exists():
        return cand
    cand2 = repo / 'simulation' / arg
    if cand2.exists():
        return cand2
    raise FileNotFoundError(f"cannot find sim dir for '{arg}'")


def _read_setup_sample_rate(setup_path: Path, default: int = 5) -> int:
    try:
        with open(setup_path, 'r', encoding='utf-8') as f:
            d = json.load(f)
        for k in ('sample_rate',):
            if d.get(k) is not None:
                return int(d[k])
        sim = d.get('simulator') or {}
        if sim.get('sample_rate') is not None:
            return int(sim['sample_rate'])
        kwargs = (sim.get('kwargs') or {})
        if kwargs.get('sample_rate') is not None:
            return int(kwargs['sample_rate'])
        return int(default)
    except Exception:
        return int(default)


def calibrate(sim_dir: Path, bs_probe: int = 4,
              plant_id_cache: Path | None = None,
              run_plan_cache: Path | None = None,
              force_model_size: str | None = None,
              force_seq_len: int | None = None,
              force_horizon: int | None = None) -> dict:
    repo = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(repo))
    setup_path = sim_dir / 'control_setup.json'
    obj_path = sim_dir / 'control_objective.json'
    assert setup_path.exists() and obj_path.exists(), \
        f'missing setup/objective in {sim_dir}'
    os.environ['CONTROL_SETUP_JSON'] = str(setup_path)
    os.environ['CONTROL_OBJECTIVE_JSON'] = str(obj_path)
    os.environ['SIMULATION_DIR'] = str(sim_dir)
    os.environ.setdefault('SEED', '0')

    # ── Dynamics + lookback ──
    if plant_id_cache and plant_id_cache.exists():
        plant = json.loads(plant_id_cache.read_text())
        tau = float(plant['tau'])
        dead = float(plant['dead_time'])
        tau_fast = float(plant.get('tau_fast', tau))
        dead_fast = float(plant.get('dead_time_fast', dead))
        lookback = int(plant['lookback'])
        # Build a dyn-raw stub good enough for derive_all (needs same fields
        # identify_dynamics returns).  Re-running ID would take minutes; the
        # cached file is a faithful snapshot from a prior single_run.
        # For derive_all we need: per-channel tau/dead_time/sign for sim_meta.
        # We'll fake them as scalars; complexity_score uses dominant only.
        dyn = {
            'tau_dom': tau, 'dead_time': dead,
            'tau_fast': tau_fast, 'dead_time_fast': dead_fast,
        }
        sr_setup = _read_setup_sample_rate(setup_path, default=4)
    else:
        from workflow._plant_prepare import (
            identify_dynamics, identify_lookback,
        )
        tmp_plant_dir = Path('/tmp') / f'gpu_calib_plant_{sim_dir.name}_{int(time.time())}'
        tmp_plant_dir.mkdir(parents=True, exist_ok=True)
        print(f'[gpu-calib] running dynamics ID (may take a few minutes)...',
              flush=True)
        plant_info = identify_dynamics(tmp_plant_dir)
        dyn = plant_info['dynamics_raw']
        tau = plant_info['tau']
        dead = plant_info['dead_time']
        tau_fast = plant_info['tau_fast']
        dead_fast = plant_info['dead_time_fast']
        sr_setup = _read_setup_sample_rate(setup_path, default=4)
        os.environ['SIM_SAMPLE_RATE'] = str(sr_setup)
        os.environ['IDENTIFIED_TAU_DOMINANT'] = f'{tau:g}'
        os.environ['IDENTIFIED_DEAD_TIME'] = f'{dead:g}'
        lb_info = identify_lookback(tmp_plant_dir, tau=tau, dead_time=dead,
                                     sample_rate=sr_setup, dynamics_raw=dyn,
                                     tau_fast=tau_fast,
                                     dead_time_fast=dead_fast)
        lookback = int(lb_info['lookback'])

    os.environ['SIM_SAMPLE_RATE'] = str(sr_setup)
    os.environ['IDENTIFIED_TAU_DOMINANT'] = f'{tau:g}'
    os.environ['IDENTIFIED_DEAD_TIME'] = f'{dead:g}'

    # ── Plant-tied derivations (model_size, seq_len, k_max) ──
    from utils.sim_factory import create_sim, resolve_sim_metadata
    from utils.plant_init import derive_all
    tmp_sim = create_sim(episode_length=10, sample_rate=max(1, sr_setup))
    sim_meta = resolve_sim_metadata(tmp_sim)
    # derive_all expects a dict with per-channel tau lists; rebuild from
    # cached scalars when running from cache.
    if isinstance(dyn, dict) and 'tau_dom' in dyn and 'tau_per_pair' not in dyn:
        # Minimal shape sufficient for derive_all complexity score.
        dyn_for_derive = {
            'tau_dom': dyn['tau_dom'], 'dead_time': dyn['dead_time'],
            'tau_fast': dyn['tau_fast'], 'dead_time_fast': dyn['dead_time_fast'],
            'tau_per_pair': [[dyn['tau_dom']]],
            'dead_time_per_pair': [[dyn['dead_time']]],
            'sign_per_pair': [[1]],
        }
    else:
        dyn_for_derive = dyn
    try:
        derived = derive_all(dyn_for_derive, sim_meta,
                              sample_rate_override=sr_setup)
        model_size = derived['model_size']
        seq_len = int(derived['seq_len'])
        k_max = int(derived['k_max'])
    except Exception as e:
        print(f'[gpu-calib] derive_all failed ({e!r}); using defaults', flush=True)
        from utils.plant_init import derive_model_size
        n_mv = len(sim_meta.get('mv_names', []))
        n_cv = len(sim_meta.get('cv_names', []))
        n_dv = len(sim_meta.get('dv_names', []))
        state_dim = int(sim_meta.get('state_dim', 4))
        model_size = derive_model_size(
            n_mv=n_mv, n_cv=n_cv, n_dv=n_dv, state_dim=state_dim,
            tau_dom=tau, tau_fast=tau_fast, sample_rate=sr_setup)
        seq_len = 64
        k_max = 8

    horizon = 15
    # If a run_plan.json from a prior real run is provided, use its
    # resolved cfg as ground-truth (it captures the same env-override
    # path as production).
    if run_plan_cache and run_plan_cache.exists():
        rp = json.loads(run_plan_cache.read_text())
        rp_cfg = rp.get('config', rp)
        model_size = rp.get('model_size') or model_size
        seq_len = int(rp.get('seq_len') or seq_len)
        horizon = int(rp.get('horizon') or horizon)
        lookback = int(rp.get('lookback') or lookback)
        k_max = int(rp.get('k_max') or k_max)
    if force_model_size:
        model_size = force_model_size
    if force_seq_len:
        seq_len = int(force_seq_len)
    if force_horizon:
        horizon = int(force_horizon)
    action_dim = len(sim_meta.get('mv_indices', [])) or 1
    # obs_dim is plant-derived and depends on the SetpointManager + aug
    # features.  Use the cached resolver from above.
    obs_dim, action_dim = _resolve_obs_action_dim(
        lookback=lookback, sample_rate=sr_setup, seq_len=seq_len,
        horizon=horizon, k_max=k_max, episode_length=1000)

    # ── Build TrainConfig + measure ──
    from training.train import TrainConfig
    from workflow.bo_runner import MODEL_SIZE_PRESETS
    arch = MODEL_SIZE_PRESETS[model_size]
    cfg = TrainConfig(
        d_model=arch['d_model'], n_layers=arch['n_layers'],
        n_heads=arch['n_heads'], z_dim=arch['z_dim'],
        n_register=arch['n_register'], tok_hidden=arch['tok_hidden'],
        head_hidden=arch['head_hidden'],
        lookback=lookback, sample_rate=sr_setup,
        episode_length=1000, total_steps=1000,
        horizon=horizon, seq_len=seq_len, k_max=k_max,
        batch_size=bs_probe, out_dir='/tmp/gpu_calib_out',
        obs_dim=obs_dim, action_dim=action_dim,
    )

    if not torch.cuda.is_available():
        return {'error': 'no CUDA available', 'sim': sim_dir.name}

    meas = measure_per_sample_mb(cfg, bs_probe=bs_probe)
    if 'error' in meas:
        return {'error': meas['error'], 'sim': sim_dir.name}
    peak_mb = float(meas['peak_mb'])
    per_sample_mb = float(meas['per_sample_mb'])

    # Recommended bs under empirical per_sample using
    # pick_batch_size_empirical's snap rule (target_util=0.65, [16, 256]).
    rec = pick_batch_size_empirical(
        model_size=model_size, seq_len=seq_len, lookback=lookback,
        horizon=horizon, k_max=k_max, sample_rate=sr_setup,
        obs_dim=obs_dim, action_dim=action_dim, bs_probe=bs_probe)
    emp_bs = int(rec['batch_size'])
    target_util = float(rec['target_util'])
    gpu_total_gb = float(rec.get('gpu_total_gb', 0.0))

    return {
        'sim': sim_dir.name,
        'model_size': model_size, 'seq_len': seq_len, 'lookback': lookback,
        'horizon': horizon, 'obs_dim': obs_dim, 'action_dim': action_dim,
        'k_max': k_max,
        'bs_probe': bs_probe,
        'peak_mb_at_bs_probe': peak_mb,
        'per_sample_mb_measured': per_sample_mb,
        'recommended_bs': emp_bs,
        'gpu_total_gb': gpu_total_gb,
        'target_util_used': target_util,
        'predicted_peak_gib_at_rec_bs': emp_bs * per_sample_mb / 1024,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--simulation-dir', '-s', required=True)
    parser.add_argument('--plant-id', default=None,
                        help='Path to a cached plant_id.json (skip dynamics ID).')
    parser.add_argument('--run-plan', default=None,
                        help='Path to a cached run_plan.json (use its resolved '
                             'cfg as ground-truth: model_size/seq_len/horizon/'
                             'lookback/k_max).')
    parser.add_argument('--model-size', default=None, choices=['S','M','L'])
    parser.add_argument('--seq-len', type=int, default=None)
    parser.add_argument('--horizon', type=int, default=None)
    parser.add_argument('--bs-probe', type=int, default=4)
    parser.add_argument('--json-out', default=None,
                        help='Write result to this JSON file.')
    args = parser.parse_args()
    sim_dir = _resolve_sim_dir(args.simulation_dir)
    plant_cache = Path(args.plant_id) if args.plant_id else None
    rp_cache = Path(args.run_plan) if args.run_plan else None
    result = calibrate(sim_dir, bs_probe=args.bs_probe,
                        plant_id_cache=plant_cache,
                        run_plan_cache=rp_cache,
                        force_model_size=args.model_size,
                        force_seq_len=args.seq_len,
                        force_horizon=args.horizon)
    print(json.dumps(result, indent=2))
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(result, indent=2))
    return 0


if __name__ == '__main__':
    sys.exit(main())
