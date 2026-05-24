"""Cross-sim GPU per-sample memory calibration.

Builds the actual training model (same code path as `workflow.single_run`)
for a given simulation, runs ONE forward + backward of `world_model_loss`
at bs=4 on synthetic data, and measures peak GPU memory.  Reports actual
per-sample MB and what `derive_batch_size` would pick under the current
baselines vs the empirical value.

Usage::

    python -m tools.gpu_calibrate --simulation-dir simulation/test_sim

Skips dynamics ID if `--plant-id` points to an existing plant_id.json
from a prior run (re-uses tau/dead_time/lookback).

2026-05-24: created during the P44→P45 cross-sim GPU auto-tune work.
P43 observed bs=16 → 12.4 GiB on test_sim (L, seq=128, lb=120, hz=15) =
793 MB/sample; the current L baseline of 740 MB under-predicts by 1.46×.
This tool gives us a measured per-sim number so the baselines aren't
chosen on one anecdote.
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
    lookback = int(cfg.lookback)
    mtp_length = max(1, int(getattr(cfg, 'mtp_length', 8)))
    try:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        model = build_model(cfg).to(device)
        model.train()
        B = int(bs_probe)
        obs = torch.randn(B, seq_len, lookback, obs_dim, device=device)
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
                               paper_default: int = 16, max_bs: int = 256,
                               target_util: float = 0.65,
                               bs_probe: int = 4,
                               wm_overhead_factor: float = 1.0) -> dict:
    """Empirically size the batch by measuring actual WM fwd+bwd peak
    memory at ``bs_probe`` and projecting linearly.

    ``wm_overhead_factor`` (>=1.0) accounts for the extra memory of
    actor+critic+optimizer state on top of the measured WM peak.  Default
    1.0 = WM-only; pass e.g. 1.15 to reserve 15% for actor/critic/opt.

    Returns a dict with the same shape as ``derive_batch_size`` for
    seamless drop-in, plus ``per_sample_mb_measured`` and
    ``source='empirical:gpu_calibrate'``.
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
    info = {'model_size': model_size, 'seq_len': seq_len, 'lookback': lookback,
            'horizon': horizon, 'paper_default': paper_default,
            'target_util': target_util, 'min_bs': paper_default, 'max_bs': max_bs,
            'wm_overhead_factor': wm_overhead_factor, 'bs_probe': bs_probe}
    # Env-var overrides (parity with derive_batch_size).
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
        # Fall back to formula on probe failure.
        from utils.plant_init import derive_batch_size as _dbs
        fb = _dbs(model_size, horizon=horizon, seq_len=seq_len, lookback=lookback,
                  paper_default=paper_default, max_bs=max_bs, target_util=target_util)
        fb['source'] = f'empirical_fallback:{meas["error"]}->{fb.get("source","auto")}'
        return fb
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
    from utils.plant_init import derive_all, derive_batch_size
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
    # features.  Easiest correct way: build a real APCEnv briefly.
    from training.train import TrainConfig, build_model, world_model_loss
    from training.train import APCEnv
    import numpy as np
    tmp_cfg_for_env = TrainConfig(
        lookback=lookback, sample_rate=sr_setup,
        episode_length=1000, total_steps=1000,
        horizon=horizon, seq_len=seq_len, k_max=k_max,
        batch_size=bs_probe, out_dir='/tmp/gpu_calib_out',
    )
    _tmp_rng = np.random.default_rng(0)
    _tmp_env = APCEnv(tmp_cfg_for_env, _tmp_rng)
    obs_dim = int(_tmp_env.obs_dim)
    action_dim = int(_tmp_env.action_dim)
    del _tmp_env

    # ── Build TrainConfig + model ──
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

    device = torch.device('cuda')
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model = build_model(cfg).to(device)
    model.train()

    # ── Synthetic batch matching world_model_loss expectations ──
    B, T, L, D = bs_probe, seq_len, lookback, obs_dim
    A = action_dim
    obs = torch.randn(B, T, L, D, device=device)
    act = torch.randn(B, T, A, device=device).clamp_(-1, 1)
    # reward MTP target shape: (B, T, mtp_length)
    fut_rew = torch.randn(B, T, max(1, int(cfg.mtp_length)), device=device)
    batch = {'obs': obs, 'act': act, 'fut_rew': fut_rew}
    # Some loss helpers expect 'rew' as well; include both forms.
    batch['rew'] = torch.randn(B, T, device=device)

    # Warmup pass (allocate workspaces) and reset peak.
    losses, _, _ = world_model_loss(model, batch, cfg)
    total = sum(v for k, v in losses.items()
                if isinstance(v, torch.Tensor) and v.requires_grad)
    total.backward()
    model.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    # Measured pass.
    losses, _, _ = world_model_loss(model, batch, cfg)
    total = sum(v for k, v in losses.items()
                if isinstance(v, torch.Tensor) and v.requires_grad)
    total.backward()
    torch.cuda.synchronize()
    peak_b = torch.cuda.max_memory_allocated()
    peak_mb = peak_b / (1024 ** 2)
    per_sample_mb = peak_mb / bs_probe

    # ── What derive_batch_size currently picks for this sim ──
    bs_info = derive_batch_size(model_size, horizon=horizon,
                                 seq_len=seq_len, lookback=lookback)
    free_b, total_b = torch.cuda.mem_get_info(0)
    gpu_total_gb = total_b / (1024 ** 3)

    # Recommended bs under empirical per_sample (snapped to power of 2, [16, 256]).
    target_util = 0.75
    budget_mb = target_util * total_b / (1024 ** 2)
    raw_bs = max(8, int(budget_mb // per_sample_mb))
    emp_pow = 1 << max(3, int(math.floor(math.log2(max(raw_bs, 8)))))
    emp_bs = int(min(256, max(8, emp_pow)))

    del model, obs, act, fut_rew, batch, losses, total
    torch.cuda.empty_cache()

    return {
        'sim': sim_dir.name,
        'model_size': model_size, 'seq_len': seq_len, 'lookback': lookback,
        'horizon': horizon, 'obs_dim': obs_dim, 'action_dim': action_dim,
        'k_max': k_max,
        'bs_probe': bs_probe,
        'peak_mb_at_bs_probe': peak_mb,
        'per_sample_mb_measured': per_sample_mb,
        'per_sample_mb_formula': bs_info.get('per_batch_mb'),
        'formula_undercalibration': (per_sample_mb /
                                      max(1.0, bs_info.get('per_batch_mb', 1.0))),
        'current_bs_formula': bs_info.get('batch_size'),
        'recommended_bs_empirical': emp_bs,
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
