"""Data-audit v2 — re-collect with raw state taps + DR/disturbance probes.

Patches APCEnv to capture:
  * raw sim state (pre-normalization) per step
  * DR realization per episode (output_gain, bias, actuator_tau, etc.)
  * hidden-disturbance firing flag + amplitude
  * curriculum disturbance schedule length

Then per domain, plots:
  * MV histogram (combined + per-channel)
  * CV trajectory in PHYSICAL units with bounds overlaid
  * 2D MV x CV occupancy in physical units
  * steady-state convergence test (does CV stabilize?)
  * DR distribution check

Sim-agnostic usage:
    python tools/audit_data_generation_v2.py --sim test_sim
    python tools/audit_data_generation_v2.py --sim distillation \
        --tau 55 --dead 6 --episode-len 900
    python tools/audit_data_generation_v2.py --sim test_sim \
        --source-run output/test_sim/run_20260523_p43_data_fixed
"""
from __future__ import annotations
import argparse, json, os, sys, math, time
from pathlib import Path
from typing import Dict, List
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO = Path('/home/koitkam/neural-apc-dreamerV4')
sys.path.insert(0, str(REPO))

# ---------- CLI / sim-agnostic config -------------------------------------
_p = argparse.ArgumentParser(allow_abbrev=False, add_help=True,
                              description='Sim-agnostic data-buffer audit.')
_p.add_argument('--sim', default=os.environ.get('AUDIT_SIM_NAME', 'test_sim'),
                help='Simulation folder name under simulation/ (default test_sim).')
_p.add_argument('--source-run', default=os.environ.get('AUDIT_SOURCE_RUN', ''),
                help='Prior run dir to harvest tau/dead/ep_len/noise_config; '
                     'auto-pick latest run_* if blank.')
_p.add_argument('--tau', type=float,
                default=float(os.environ.get('AUDIT_TAU', '55')),
                help='Identified dominant time constant (sample steps).')
_p.add_argument('--dead', type=float,
                default=float(os.environ.get('AUDIT_DEAD', '8')),
                help='Identified dead time (sample steps).')
_p.add_argument('--sample-rate', type=int,
                default=int(os.environ.get('AUDIT_SAMPLE_RATE', '4')))
_p.add_argument('--episode-len', type=int,
                default=int(os.environ.get('AUDIT_EPISODE_LEN', '1220')))
_p.add_argument('--lookback', type=int,
                default=int(os.environ.get('AUDIT_LOOKBACK', '120')))
_p.add_argument('--seed', type=int, default=int(os.environ.get('SEED', '12345')))
_p.add_argument('--n-eps', type=int, default=8,
                help='Episodes per domain (per-MV/per-CV stats use the same pool).')
_args = _p.parse_args()

SIM_NAME = _args.sim
SIM_DIR  = REPO / 'simulation' / SIM_NAME
if not (SIM_DIR / 'control_setup.json').exists():
    raise SystemExit(f'[v2] sim folder {SIM_DIR} has no control_setup.json')

if _args.source_run.strip():
    SOURCE_RUN = Path(_args.source_run)
else:
    _runs = sorted((REPO / 'output' / SIM_NAME).glob('run_*'),
                   key=lambda p: p.stat().st_mtime if p.exists() else 0)
    SOURCE_RUN = _runs[-1] if _runs else None

os.environ['CONTROL_SETUP_JSON']     = str(SIM_DIR / 'control_setup.json')
os.environ['CONTROL_OBJECTIVE_JSON'] = str(SIM_DIR / 'control_objective.json')
os.environ['SIMULATION_DIR']         = str(SIM_DIR)
os.environ['SEED']                   = str(_args.seed)
if SOURCE_RUN is not None and (SOURCE_RUN / 'noise_config.json').exists():
    os.environ['SIM_NOISE_CONFIG_JSON'] = str(SOURCE_RUN / 'noise_config.json')

if SOURCE_RUN is not None and (SOURCE_RUN / 'run_plan.json').exists():
    plan = json.loads((SOURCE_RUN / 'run_plan.json').read_text())
    TAU         = float(plan.get('tau', _args.tau))
    DEAD        = float(plan.get('dead_time', _args.dead))
    SAMPLE_RATE = int(plan.get('sample_rate', _args.sample_rate))
    EPISODE_LEN = int(plan['config'].get('episode_length', _args.episode_len))
    LOOKBACK    = int(plan['config'].get('lookback', _args.lookback))
    _params_src = f'source_run={SOURCE_RUN.name}'
else:
    TAU         = float(_args.tau)
    DEAD        = float(_args.dead)
    SAMPLE_RATE = int(_args.sample_rate)
    EPISODE_LEN = int(_args.episode_len)
    LOOKBACK    = int(_args.lookback)
    _params_src = 'cli/env defaults (no prior run)'
os.environ['SIM_SAMPLE_RATE']         = str(SAMPLE_RATE)
os.environ['IDENTIFIED_TAU_DOMINANT'] = f'{TAU:g}'
os.environ['IDENTIFIED_DEAD_TIME']    = f'{DEAD:g}'
os.environ['SIM_EPISODE_LENGTH']      = str(EPISODE_LEN)

OUT = REPO / f'output/{SIM_NAME}/_data_audit_v2_{time.strftime("%Y%m%d_%H%M%S")}'
(OUT / 'plots').mkdir(parents=True, exist_ok=True)
print(f'[v2] sim={SIM_NAME} writing to {OUT}', flush=True)
print(f'[v2] plant tau={TAU} dead={DEAD} sr={SAMPLE_RATE} ep_len={EPISODE_LEN} '
      f'lookback={LOOKBACK}  source={_params_src}',
      flush=True)

# ---- Build env, then monkey-patch step/reset to log raw state -------------
from training.train import (TrainConfig, APCEnv,
    collect_baseline_episode, collect_prbs_episode,
    collect_constant_action_episode, collect_step_settle_episode)

cfg = TrainConfig(lookback=LOOKBACK, sample_rate=SAMPLE_RATE,
                  episode_length=EPISODE_LEN)
cfg.prbs_seed_n_strata = 8
cfg.constant_action_seed_op_band = 0.6
rng = np.random.default_rng(int(os.environ['SEED']))
env = APCEnv(cfg, rng)

# Patch step/reset to capture raw state + DR + disturbance
_raw_states: List[np.ndarray] = []
_episode_meta: List[Dict] = []
_orig_reset      = env.reset
_orig_step       = env.step
_orig_build_obs  = env._build_obs_vec

def _find_dr_obj(sim):
    """Walk wrapper chain to find a DomainRandomizer instance."""
    seen = set(); cur = sim
    for _ in range(8):
        if cur is None or id(cur) in seen: break
        seen.add(id(cur))
        # Both TestSimTower and DistillationTower expose the wrapper-level
        # DomainRandomizer as ``_randomizer``; older sims may use other names.
        for attr in ('_randomizer', 'dr', '_dr',
                     'domain_randomizer', '_domain_randomizer',
                     '_domain', 'domain'):
            if hasattr(cur, attr):
                obj = getattr(cur, attr)
                if obj is not None and hasattr(obj, 'output_gain'):
                    return obj
        cur = (getattr(cur, '_inner', None) or getattr(cur, '_sim', None)
               or getattr(cur, 'inner', None) or getattr(cur, 'wrapped', None))
    return None

def _dr_dump_attrs(sim):
    """Diagnostic: find any per-episode-randomized attribute on any wrapper."""
    cur = sim; chain = []
    while cur is not None and len(chain) < 8:
        chain.append({'cls': type(cur).__name__,
                      'attrs': sorted(a for a in vars(cur).keys()
                                       if not a.startswith('__'))[:25]
                                if hasattr(cur, '__dict__') else []})
        cur = (getattr(cur, '_inner', None) or getattr(cur, '_sim', None)
               or getattr(cur, 'inner', None) or getattr(cur, 'wrapped', None))
    return chain

def build_obs_hook(state):
    # Capture raw physical state BEFORE normalization
    try:
        s = np.asarray(state, dtype='float32').reshape(-1).copy()
        if _raw_states:
            _raw_states[-1].append(s)
    except Exception:
        pass
    return _orig_build_obs(state)

def reset_hook(*a, **kw):
    # Start a new raw-state collection bucket BEFORE reset (which calls build_obs)
    _raw_states.append([])
    out = _orig_reset(*a, **kw)
    sim = env.sim
    meta = {'sched_len': len(env._schedule or []),
            'hidden_active': env._hidden_disturbance is not None}
    dr = _find_dr_obj(sim)
    # Also probe plant-internal randomized fields (test_sim / distillation
    # randomize tau / k_u / k_d directly inside _sample_episode_dynamics)
    inner = sim
    for _ in range(6):
        nxt = (getattr(inner, '_inner', None) or getattr(inner, '_sim', None)
               or getattr(inner, 'inner', None) or getattr(inner, 'wrapped', None))
        if nxt is None: break
        inner = nxt
    for attr in ('tau_temp', 'tau_actuator', 'mv_deadtime_steps',
                 'target_coeffs'):
        v = getattr(inner, attr, None)
        if v is None: continue
        if isinstance(v, dict):
            for k, kv in v.items():
                try: meta[f'plant_{attr}_{k}'] = float(kv)
                except Exception: pass
        else:
            try: meta[f'plant_{attr}'] = float(v)
            except Exception: pass
    if dr is not None:
        meta.update({
            'dr_output_gain':   float(getattr(dr, 'output_gain', 1.0)),
            'dr_output_bias':   float(getattr(dr, 'output_bias_frac', 0.0)),
            'dr_input_jitter':  float(getattr(dr, 'input_jitter_std', 0.0)),
            'dr_actuator_tau':  float(getattr(dr, 'actuator_tau_steps', 0.0)),
            'dr_mv_deadtime':   float(getattr(dr, 'mv_deadtime_steps', 0.0)),
            'dr_enabled':       bool(getattr(dr, 'enabled', True)),
        })
    hd = env._hidden_disturbance
    if hd is not None:
        meta['hidden_amp'] = float(getattr(hd, 'amplitude', 0.0))
        meta['hidden_tau'] = float(getattr(hd, 'tau', 0.0))
    _episode_meta.append(meta)
    return out

env.reset          = reset_hook
env._build_obs_vec = build_obs_hook

# Diagnostic: dump sim wrapper chain so we know where DR lives
print('[v2] sim wrapper chain:', flush=True)
for i, info in enumerate(_dr_dump_attrs(env.sim)):
    print(f'   [{i}] {info["cls"]}: {info["attrs"]}', flush=True)

# ---- Collect N per domain --------------------------------------------------
N = 8
op_band = 0.6
const_levels = np.linspace(-op_band, op_band, N, dtype='float32')

def collect_random_episode():
    ow = env.reset(exploration=True); T = cfg.episode_length
    obs_buf = np.zeros((T, env.obs_dim), 'float32')
    act_buf = np.zeros((T, env.action_dim), 'float32')
    rew_buf = np.zeros(T, 'float32'); cont_buf = np.ones(T, 'float32')
    for t in range(T):
        obs_buf[t] = ow[-1]
        a = env.rng.uniform(-1.0, 1.0, size=env.action_dim).astype('float32')
        ow, r, d, _ = env.step(a)
        act_buf[t] = a; rew_buf[t] = r
        cont_buf[t] = 0.0 if (d and t == T-1) else 1.0
        if d: break
    return {'obs': obs_buf, 'act': act_buf, 'rew': rew_buf, 'cont': cont_buf}

# domain -> list of (episode_dict, raw_states_array, meta_dict)
runs = {}

def collect_n(name, factory_iter):
    items = []
    for i, fn in enumerate(factory_iter):
        ep_start = len(_episode_meta)
        # Track raw-state bucket index pre-collection (build_obs_hook appends)
        bucket_start = len(_raw_states)
        try:
            ep = fn()
        except Exception as e:
            print(f'[v2] {name} ep{i} FAIL: {e}', flush=True); continue
        meta = _episode_meta[ep_start] if ep_start < len(_episode_meta) else {}
        # P43 (2026-05-23): re-poll _hidden_disturbance AFTER the
        # collection so const_action / step_settle (which null it
        # before stepping) report correct fire rate.
        meta['hidden_active_post'] = (env._hidden_disturbance is not None)
        meta['sched_len_post'] = len(env._schedule or [])
        # Latest bucket (last entry in _raw_states) holds this episode's frames
        raw_list = _raw_states[bucket_start] if bucket_start < len(_raw_states) else []
        # Drop the first frame (initial state) so raw aligns with ep['act'] of length T
        raw_list_trim = raw_list[1:] if len(raw_list) > ep['act'].shape[0] else raw_list
        raw = np.stack(raw_list_trim, axis=0) if raw_list_trim else None
        items.append((ep, raw, meta))
    runs[name] = items
    print(f'[v2] {name}: {len(items)} eps  raw_shape={items[0][1].shape if items and items[0][1] is not None else None}',
          flush=True)

# P43: mirror training/train.py P1 loop — stratified baseline_seed centres
# over [-0.6, +0.6] so audit reflects the buffer the WM actually trains on.
_b_edges = np.linspace(-0.6, +0.6, N + 1)
_b_centres = np.random.default_rng(0).uniform(_b_edges[:-1], _b_edges[1:])
collect_n('baseline_seed',
    [lambda c=float(c): collect_baseline_episode(env, cfg, action_std=0.05,
                                                  center=c)
     for c in _b_centres])
collect_n('prbs_seed',
    [lambda: collect_prbs_episode(env, cfg, action_std=0.05, op_band=0.95) for _ in range(N)])
collect_n('const_action_seed',
    [(lambda lv=float(L): collect_constant_action_episode(env, cfg, action_level=lv))
     for L in const_levels])
collect_n('step_settle_seed',
    [(lambda: collect_step_settle_episode(env, cfg,
         action_start=float(env.rng.uniform(-0.5, 0.5)),
         action_end=float(env.rng.uniform(-0.5, 0.5)),
         switch_step=int(env.rng.uniform(0.3, 0.7) * EPISODE_LEN)))
     for _ in range(N)])
collect_n('random_action', [collect_random_episode for _ in range(N)])

# CV/MV bounds (physical) -- read off env. Per-channel arrays; element 0
# is what existing plots/summary fields refer to for backward compat.
mv_norm = [(float(lo), float(hi)) for lo, hi in env.mv_norm_ranges]
cv_norm = [(float(lo), float(hi)) for lo, hi in env.cv_norm_ranges]
mv_lo_n, mv_hi_n = mv_norm[0]
cv_lo_n, cv_hi_n = cv_norm[0]
print(f'[v2] MV[0] norm-bounds [{mv_lo_n}, {mv_hi_n}]  '
      f'CV[0] norm-bounds [{cv_lo_n}, {cv_hi_n}]  '
      f'action_dim={env.action_dim}  n_cv={len(env.cv_indices)}',
      flush=True)

# Read objective CV target if available
import json as _j
obj = _j.loads((SIM_DIR / 'control_objective.json').read_text())
print(f'[v2] objective top-level keys: {list(obj.keys())[:6]}', flush=True)

# (A) Operating bounds: prefer ``control_objective.bounds.{mvs,outputs}`` over
# the wider ``mv_norm_ranges`` / ``cv_norm_ranges``. The operating region is
# what the controller actually has to cover; using the (often much wider)
# normalization range over-states coverage gaps. Falls back to norm ranges
# if the objective lacks bounds — keeps the audit working for legacy plants.
_obj_b = obj.get('bounds', {}) if isinstance(obj, dict) else {}
_mv_b_raw = _obj_b.get('mvs', {})  if isinstance(_obj_b.get('mvs'),     dict) else {}
_cv_b_raw = _obj_b.get('outputs', {}) if isinstance(_obj_b.get('outputs'), dict) else {}

def _resolve_op_bounds(norm_list, b_raw, prefix):
    """Return per-channel (lo, hi) operating bounds.

    ``b_raw`` keys may be ``mv_0``/``cv_0``/``output_0`` or numeric strings.
    Falls back to ``norm_list[i]`` when missing.
    """
    out = []
    for i, (lo, hi) in enumerate(norm_list):
        for key in (f'{prefix}_{i}', str(i), f'output_{i}'):
            if key in b_raw:
                try:
                    pair = b_raw[key]
                    out.append((float(pair[0]), float(pair[1]))); break
                except Exception:
                    pass
        else:
            out.append((lo, hi))
    return out

mv_op = _resolve_op_bounds(mv_norm, _mv_b_raw, 'mv')
cv_op = _resolve_op_bounds(cv_norm, _cv_b_raw, 'cv')
print(f'[v2] MV op-bounds (per-channel): {mv_op}', flush=True)
print(f'[v2] CV op-bounds (per-channel): {cv_op}', flush=True)

# Channel-0 operating bounds used as default plot axes (existing plot
# blocks below reference ``mv_lo``/``mv_hi``/``cv_lo``/``cv_hi``).
mv_lo, mv_hi = mv_op[0]
cv_lo, cv_hi = cv_op[0]

# ---- Analysis: physical-units coverage ------------------------------------
COLORS = {'baseline_seed': '#1f77b4', 'prbs_seed': '#ff7f0e',
          'const_action_seed': '#2ca02c', 'step_settle_seed': '#d62728',
          'random_action': '#9467bd'}

def _hist_stats(arr, lo, hi, bins=20):
    """Return (in_band_frac, bin_cov_in_op) for ``arr`` over ``[lo, hi]``."""
    if arr.size == 0 or hi <= lo:
        return 0.0, 0.0
    in_band = float(((arr >= lo) & (arr <= hi)).mean())
    h, _ = np.histogram(arr, bins=bins, range=(lo, hi))
    return in_band, float((h > 0).mean())

summary = {}
# Track per-MV / per-CV combined arrays for (B) per-channel reporting.
combined_mv_per: List[List[np.ndarray]] = [[] for _ in mv_norm]
combined_cv_per: List[List[np.ndarray]] = [[] for _ in cv_norm]

for name, items in runs.items():
    cvs_all = []
    mvs_all = []
    dr_gains, dr_biases, dr_lags, hidden_flags, sched_lens = [], [], [], [], []
    cv_first_50pct, cv_last_25pct = [], []   # mean of first 50% vs last 25% (settling)
    for ep, raw, meta in items:
        # Per-MV (action -> physical)
        for j in range(env.action_dim):
            mlo, mhi = mv_norm[j]
            a_j = ep['act'][:, j]
            mv_phys_j = mlo + 0.5 * (a_j + 1.0) * (mhi - mlo)
            if j == 0:
                mvs_all.append(mv_phys_j)
            combined_mv_per[j].append(mv_phys_j)
        # Per-CV (raw state)
        if raw is not None and raw.shape[1] > 0:
            for j, cv_idx in enumerate(env.cv_indices):
                cv_j = raw[:, cv_idx]
                if j == 0:
                    cvs_all.append(cv_j)
                    T = len(cv_j)
                    cv_first_50pct.append(float(cv_j[:T//2].mean()))
                    cv_last_25pct.append(float(cv_j[-T//4:].mean()))
                combined_cv_per[j].append(cv_j)
        dr_gains.append(meta.get('dr_output_gain', np.nan))
        dr_biases.append(meta.get('dr_output_bias', np.nan))
        dr_lags.append(meta.get('dr_actuator_tau', np.nan))
        hidden_flags.append(int(meta.get('hidden_active_post',
                                          meta.get('hidden_active', False))))
        sched_lens.append(int(meta.get('sched_len', 0)))

    cv_arr = np.concatenate(cvs_all) if cvs_all else np.array([])
    mv_arr = np.concatenate(mvs_all) if mvs_all else np.array([])

    # Channel-0 stats over *operating* bounds (A).
    mv_lo0_op, mv_hi0_op = mv_op[0]
    cv_lo0_op, cv_hi0_op = cv_op[0]
    mv_in_band, mv_cov = _hist_stats(mv_arr, mv_lo0_op, mv_hi0_op)
    cv_in_band, cv_cov = _hist_stats(cv_arr, cv_lo0_op, cv_hi0_op)

    # (B) per-MV / per-CV stats for this domain.
    per_mv = []
    for j, (mlo, mhi) in enumerate(mv_op):
        per_ep = [(mv_norm[j][0]
                   + 0.5*(ep['act'][:, j]+1.0)*(mv_norm[j][1]-mv_norm[j][0]))
                  for ep, _, _ in items]
        a = np.concatenate(per_ep) if per_ep else np.array([])
        ib, cv_ = _hist_stats(a, mlo, mhi)
        per_mv.append({
            'op_lo': mlo, 'op_hi': mhi,
            'min': float(a.min()) if a.size else None,
            'max': float(a.max()) if a.size else None,
            'std': float(a.std()) if a.size else None,
            'in_op_frac': ib, 'bin_coverage_in_op': cv_,
        })
    per_cv = []
    for j, (clo, chi) in enumerate(cv_op):
        cv_idx = env.cv_indices[j]
        per_ep = [raw[:, cv_idx] for _, raw, _ in items if raw is not None]
        c = np.concatenate(per_ep) if per_ep else np.array([])
        ib, cv_ = _hist_stats(c, clo, chi)
        per_cv.append({
            'op_lo': clo, 'op_hi': chi,
            'min': float(c.min()) if c.size else None,
            'max': float(c.max()) if c.size else None,
            'std': float(c.std()) if c.size else None,
            'in_op_frac': ib, 'bin_coverage_in_op': cv_,
        })

    summary[name] = {
        'n_eps': len(items),
        # Backward-compat channel-0 entries (now using operating bounds for coverage):
        'mv_phys': {'min': float(mv_arr.min()) if mv_arr.size else None,
                    'max': float(mv_arr.max()) if mv_arr.size else None,
                    'mean': float(mv_arr.mean()) if mv_arr.size else None,
                    'std': float(mv_arr.std()) if mv_arr.size else None,
                    'in_band_frac': mv_in_band, 'bin_coverage_in_band': mv_cov,
                    'op_lo': mv_lo0_op, 'op_hi': mv_hi0_op},
        'cv_phys': {'min': float(cv_arr.min()) if cv_arr.size else None,
                    'max': float(cv_arr.max()) if cv_arr.size else None,
                    'mean': float(cv_arr.mean()) if cv_arr.size else None,
                    'std':  float(cv_arr.std())  if cv_arr.size else None,
                    'in_band_frac': cv_in_band,
                    'bin_coverage_in_band': cv_cov,
                    'cv_lo': cv_lo0_op, 'cv_hi': cv_hi0_op},
        'per_mv': per_mv,    # (B) per-channel stats over operating bounds
        'per_cv': per_cv,
        'cv_first_half_mean':  float(np.mean(cv_first_50pct)) if cv_first_50pct else None,
        'cv_last_quarter_mean': float(np.mean(cv_last_25pct))  if cv_last_25pct else None,
        'cv_settling_shift':   (float(np.mean(cv_last_25pct) - np.mean(cv_first_50pct))
                                if cv_first_50pct and cv_last_25pct else None),
        'dr_output_gain': {'mean': float(np.nanmean(dr_gains)),
                           'std': float(np.nanstd(dr_gains)),
                           'min': float(np.nanmin(dr_gains)),
                           'max': float(np.nanmax(dr_gains))},
        'dr_output_bias': {'mean': float(np.nanmean(dr_biases)),
                           'std': float(np.nanstd(dr_biases))},
        'dr_actuator_tau': {'mean': float(np.nanmean(dr_lags)),
                            'std': float(np.nanstd(dr_lags))},
        'hidden_disturbance_fire_rate': float(np.mean(hidden_flags)),
        'curriculum_schedule_len': {'mean': float(np.mean(sched_lens)),
                                    'min': int(np.min(sched_lens)),
                                    'max': int(np.max(sched_lens))},
    }

# Combined MV physical histogram (channel 0, over operating bounds).
mv_lo0_op, mv_hi0_op = mv_op[0]
cv_lo0_op, cv_hi0_op = cv_op[0]
all_mv = np.concatenate([np.concatenate([mv_norm[0][0]
                            + 0.5*(ep['act'][:,0]+1)*(mv_norm[0][1]-mv_norm[0][0])
                          for ep,_,_ in items])
                          for items in runs.values() if items])
all_cv = np.concatenate([np.concatenate([raw[:, env.cv_indices[0]]
                                          for ep,raw,_ in items if raw is not None])
                          for items in runs.values() if items])
h_mv, _ = np.histogram(all_mv, bins=20, range=(mv_lo0_op, mv_hi0_op))
h_cv, _ = np.histogram(all_cv, bins=20, range=(cv_lo0_op, cv_hi0_op))

# (B) per-channel combined coverage for ALL MV/CV channels.
combined_per_mv = []
for j, (mlo, mhi) in enumerate(mv_op):
    a = np.concatenate(combined_mv_per[j]) if combined_mv_per[j] else np.array([])
    h, _ = np.histogram(a, bins=20, range=(mlo, mhi)) if a.size else (np.zeros(20, int), None)
    combined_per_mv.append({
        'op_lo': mlo, 'op_hi': mhi,
        'bin_counts': h.tolist(),
        'bin_coverage_in_op': float((h > 0).mean()),
        'p10_count': int(np.quantile(h, 0.1)) if h.size else 0,
        'min_count': int(h.min()) if h.size else 0,
        'max_count': int(h.max()) if h.size else 0,
        'ratio_max_min': (float(h.max() / max(1, h.min())) if h.size else 0.0),
    })
combined_per_cv = []
for j, (clo, chi) in enumerate(cv_op):
    a = np.concatenate(combined_cv_per[j]) if combined_cv_per[j] else np.array([])
    h, _ = np.histogram(a, bins=20, range=(clo, chi)) if a.size else (np.zeros(20, int), None)
    combined_per_cv.append({
        'op_lo': clo, 'op_hi': chi,
        'bin_counts': h.tolist(),
        'bin_coverage_in_op': float((h > 0).mean()),
        'p10_count': int(np.quantile(h, 0.1)) if h.size else 0,
        'min_count': int(h.min()) if h.size else 0,
        'max_count': int(h.max()) if h.size else 0,
        'ratio_max_min': (float(h.max() / max(1, h.min())) if h.size else 0.0),
    })

summary['combined'] = {
    'mv_phys_bin_counts': h_mv.tolist(),
    'mv_phys_bin_coverage': float((h_mv > 0).mean()),
    'mv_phys_p10_count': int(np.quantile(h_mv, 0.1)),
    'mv_phys_min_count': int(h_mv.min()),
    'cv_phys_bin_counts': h_cv.tolist(),
    'cv_phys_bin_coverage': float((h_cv > 0).mean()),
    'cv_phys_p10_count': int(np.quantile(h_cv, 0.1)),
    'cv_phys_min_count': int(h_cv.min()),
    'per_mv': combined_per_mv,
    'per_cv': combined_per_cv,
}

(OUT / 'summary.json').write_text(json.dumps(summary, indent=2, default=float))

# ---- Plots ---------------------------------------------------------------
# 1. Physical MV histogram per domain
fig, ax = plt.subplots(2, 1, figsize=(10, 7))
for name, items in runs.items():
    mv = np.concatenate([mv_lo + 0.5*(ep['act'][:,0]+1)*(mv_hi-mv_lo)
                          for ep,_,_ in items]) if items else np.array([])
    ax[0].hist(mv, bins=40, range=(mv_lo, mv_hi), alpha=0.45, density=True,
                label=name, color=COLORS[name])
ax[0].set_xlabel(f'MV physical (bounds {mv_lo}, {mv_hi})')
ax[0].legend(fontsize=8, ncol=3)
ax[0].set_title('MV physical distribution per domain')
# combined
ctrs = np.linspace(mv_lo, mv_hi, 20)
ax[1].bar(ctrs, h_mv, width=(mv_hi-mv_lo)/22, color='#444', alpha=0.85)
ax[1].set_yscale('log')
ax[1].set_xlabel('MV physical')
ax[1].set_title(f'COMBINED MV physical coverage (log y)\n'
                f'min bin count={h_mv.min()}  p10={int(np.quantile(h_mv,0.1))}  '
                f'max={h_mv.max()}  ratio max/min={h_mv.max()/max(1,h_mv.min()):.0f}x')
fig.tight_layout()
fig.savefig(OUT / 'plots' / '01_mv_physical.png', dpi=130); plt.close(fig)

# 2. Physical CV histogram per domain + combined
fig, ax = plt.subplots(2, 1, figsize=(10, 7))
for name, items in runs.items():
    if not items: continue
    cv = [raw[:, env.cv_indices[0]] for _, raw, _ in items if raw is not None]
    if not cv: continue
    cv = np.concatenate(cv)
    ax[0].hist(cv, bins=40, range=(cv_lo, cv_hi), alpha=0.45, density=True,
                label=name, color=COLORS[name])
ax[0].set_xlabel(f'CV physical (bounds {cv_lo}, {cv_hi})')
ax[0].legend(fontsize=8, ncol=3)
ax[0].set_title('CV physical distribution per domain')
ctrs = np.linspace(cv_lo, cv_hi, 20)
ax[1].bar(ctrs, h_cv, width=(cv_hi-cv_lo)/22, color='#444', alpha=0.85)
ax[1].set_yscale('log')
ax[1].set_xlabel('CV physical')
ax[1].set_title(f'COMBINED CV physical coverage (log y)\n'
                f'min bin count={h_cv.min()}  p10={int(np.quantile(h_cv,0.1))}  '
                f'max={h_cv.max()}  ratio max/min={h_cv.max()/max(1,h_cv.min()):.0f}x')
fig.tight_layout()
fig.savefig(OUT / 'plots' / '02_cv_physical.png', dpi=130); plt.close(fig)

# 3. 2D MV-CV occupancy (physical)
fig, axes = plt.subplots(1, len(runs), figsize=(3.6*len(runs), 3.8),
                          sharex=True, sharey=True, squeeze=False)
for k, (name, items) in enumerate(runs.items()):
    ax = axes[0, k]
    if not items: ax.set_title(f'{name}\n(empty)'); continue
    mvs, cvs = [], []
    for ep, raw, _ in items:
        if raw is None: continue
        mvs.append(mv_lo + 0.5*(ep['act'][:,0]+1)*(mv_hi-mv_lo))
        cvs.append(raw[:, env.cv_indices[0]])
    if not mvs: continue
    mvs = np.concatenate(mvs); cvs = np.concatenate(cvs)
    n = min(len(mvs), len(cvs))
    mvs = mvs[:n]; cvs = cvs[:n]
    H, xe, ye = np.histogram2d(mvs, cvs, bins=20,
                                range=[[mv_lo, mv_hi], [cv_lo, cv_hi]])
    ax.imshow(np.log1p(H.T), origin='lower', aspect='auto',
              extent=[mv_lo, mv_hi, cv_lo, cv_hi], cmap='viridis')
    ax.set_title(name, fontsize=9); ax.set_xlabel('MV phys')
    if k == 0: ax.set_ylabel('CV phys')
fig.suptitle('2D MV × CV physical occupancy (log scale; brighter=more visits)')
fig.tight_layout()
fig.savefig(OUT / 'plots' / '03_mv_cv_occupancy_physical.png', dpi=130); plt.close(fig)

# 4. Timeseries (CV physical) — 3 episodes per domain
for name, items in runs.items():
    if not items: continue
    n_show = min(3, len(items))
    fig, axes = plt.subplots(n_show, 2, figsize=(11, 2.0*n_show+1),
                              sharex='col', squeeze=False)
    for i in range(n_show):
        ep, raw, meta = items[i]
        a = mv_lo + 0.5 * (ep['act'][:,0] + 1.0) * (mv_hi - mv_lo)
        axes[i, 0].plot(a, lw=0.7, color=COLORS[name])
        axes[i, 0].set_ylim(mv_lo - 5, mv_hi + 5)
        axes[i, 0].axhline(mv_lo, c='k', lw=0.3, alpha=0.4)
        axes[i, 0].axhline(mv_hi, c='k', lw=0.3, alpha=0.4)
        axes[i, 0].set_ylabel(f'ep{i}\nMV phys')
        if raw is not None:
            cv = raw[:, env.cv_indices[0]]
            axes[i, 1].plot(cv, lw=0.7, color=COLORS[name])
            axes[i, 1].set_ylim(cv_lo - 2, cv_hi + 2)
            axes[i, 1].axhline(cv_lo, c='k', lw=0.3, alpha=0.4)
            axes[i, 1].axhline(cv_hi, c='k', lw=0.3, alpha=0.4)
            axes[i, 1].set_ylabel('CV phys')
            title = f'sched={meta.get("sched_len",0)} hidden={int(meta.get("hidden_active",0))} '
            title += f'gain={meta.get("dr_output_gain",1.0):.2f}'
            axes[i, 1].set_title(title, fontsize=8)
    axes[-1, 0].set_xlabel('agent step')
    axes[-1, 1].set_xlabel('agent step')
    fig.suptitle(f'{name}: 3 representative episodes')
    fig.tight_layout()
    fig.savefig(OUT / 'plots' / f'04_ts_{name}.png', dpi=130); plt.close(fig)

# 5. SS settling test: for const_action_seed, plot CV[-100:] mean vs action_level
items = runs['const_action_seed']
if items:
    levels = []; ss_cv = []
    for ep, raw, _ in items:
        if raw is None: continue
        levels.append(float(ep['act'][0, 0]))
        ss_cv.append(float(raw[-100:, env.cv_indices[0]].mean()))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(levels, ss_cv, color=COLORS['const_action_seed'])
    ax.axhline(cv_lo, c='k', lw=0.4); ax.axhline(cv_hi, c='k', lw=0.4)
    ax.set_xlabel('action level (normalized)')
    ax.set_ylabel(f'CV phys (last 100 steps mean)\nbounds=[{cv_lo}, {cv_hi}]')
    ax.set_title('const_action_seed: steady-state CV vs held action')
    fig.tight_layout()
    fig.savefig(OUT / 'plots' / '05_const_action_ss.png', dpi=130); plt.close(fig)

print(f'[v2] DONE -> {OUT}', flush=True)
print(json.dumps({k: {kk: vv for kk, vv in v.items()
                       if kk in ('n_eps', 'mv_phys', 'cv_phys',
                                  'cv_settling_shift',
                                  'hidden_disturbance_fire_rate',
                                  'curriculum_schedule_len',
                                  'dr_output_gain', 'dr_actuator_tau')}
                  for k, v in summary.items() if k != 'combined'},
                 indent=2, default=float))
print('\nCOMBINED:')
print(json.dumps(summary['combined'], indent=2, default=float))

# (C) Sanity-check: warn loudly if any DR field is all-NaN across all
# domains. This catches silent regressions (e.g. wrapper-chain walker
# misses a renamed attribute, control_setup lacks the NR block, or
# DREAMER_DOMAIN_RANDOMIZATION=0 is left set in the env).
warnings: List[str] = []
for fld, label in (('dr_output_gain',  'output_gain'),
                   ('dr_output_bias',  'output_bias'),
                   ('dr_actuator_tau', 'actuator_tau')):
    means = [summary[k][fld].get('mean') for k in summary
             if k != 'combined' and fld in summary[k]]
    finite = [m for m in means if m is not None and m == m]  # NaN check
    if not finite:
        warnings.append(
            f'all-NaN DR field {label!r} across every domain — '
            f'check (a) DomainRandomizer attribute name in the sim '
            f'(walker looks for `_randomizer`/`dr`/`domain_randomizer`), '
            f'(b) control_setup.json has `noise_and_randomization` block, '
            f'(c) DREAMER_DOMAIN_RANDOMIZATION / SIM_DOMAIN_RANDOMIZATION are not 0.')
# Per-CV operating coverage warnings
for j, rec in enumerate(summary['combined']['per_cv']):
    if rec['bin_coverage_in_op'] < 0.5:
        warnings.append(
            f'CV[{j}] op-coverage only {rec["bin_coverage_in_op"]:.2f} over '
            f'[{rec["op_lo"]}, {rec["op_hi"]}]; min_count={rec["min_count"]} '
            f'— buffer may under-represent part of the operating envelope.')
    if rec['ratio_max_min'] > 50.0:
        warnings.append(
            f'CV[{j}] bin imbalance {rec["ratio_max_min"]:.1f}x '
            f'(max/min counts) over op[{rec["op_lo"]:.1f},{rec["op_hi"]:.1f}] '
            f'— buffer is heavily skewed; consider widening MV op-band or '
            f'enriching seed diversity for this CV.')
for j, rec in enumerate(summary['combined']['per_mv']):
    if rec['bin_coverage_in_op'] < 0.5:
        warnings.append(
            f'MV[{j}] op-coverage only {rec["bin_coverage_in_op"]:.2f} over '
            f'[{rec["op_lo"]}, {rec["op_hi"]}]; min_count={rec["min_count"]} '
            f'— consider widening baseline_seed_op_band or PRBS op_band.')
    if rec['ratio_max_min'] > 20.0:
        warnings.append(
            f'MV[{j}] bin imbalance {rec["ratio_max_min"]:.1f}x '
            f'(max/min counts); buffer is heavily skewed.')

if warnings:
    print('\n[v2] WARNINGS:')
    for w in warnings:
        print(f'  ! {w}')
    (OUT / 'warnings.txt').write_text('\n'.join(warnings) + '\n')
else:
    print('\n[v2] no warnings.')

# (B) Per-channel summary print (compact).
print('\nPER-MV (combined over all domains):')
for j, rec in enumerate(summary['combined']['per_mv']):
    print(f'  MV[{j}] op[{rec["op_lo"]:.1f},{rec["op_hi"]:.1f}] '
          f'bin_cov={rec["bin_coverage_in_op"]:.2f} '
          f'min_count={rec["min_count"]} ratio={rec["ratio_max_min"]:.1f}x')
print('PER-CV (combined over all domains):')
for j, rec in enumerate(summary['combined']['per_cv']):
    print(f'  CV[{j}] op[{rec["op_lo"]:.1f},{rec["op_hi"]:.1f}] '
          f'bin_cov={rec["bin_coverage_in_op"]:.2f} '
          f'min_count={rec["min_count"]} ratio={rec["ratio_max_min"]:.1f}x')
