"""Faithful RSSM training-data audit.

Unlike ``audit_data_generation_v2.py`` (a fixed-N qualitative coverage
probe), this tool reconstructs the **exact** seed replay buffer a given
run trains its RSSM world model on, by:

  1. Building ``APCEnv`` from the same ``control_setup`` / noise-config /
     identifier artifacts the run used (``--source-run``).
  2. Running ``auto_tune_seed_buffer`` (or loading the run's resolved
     ``auto_tune_seed_buffer.json``) so ``baseline_seed_action_std``,
     the multi-timescale PRBS segmentation, and every seed-episode count
     match the live run.
  3. Replaying the *identical* seed-fill loop from
     ``training.train.train`` (baseline → random → PRBS → const/step →
     step-test) with the same RNG seed and episode proportions.

It taps raw (pre-normalization) sim state per step, then answers the
three questions the WM-data audit must settle:

  Q1  RANGE COVERAGE  — does the buffer span the whole MV / CV / DV
      operating envelope (per-channel histogram, bin coverage, edge
      coverage, imbalance ratio), weighted by the real episode mix?

  Q2  DISTURBANCE MIX — is there a good blend of PRBS vs other excitation
      (composition table), and does the excitation cover BOTH the fast
      and the slow plant timescale (hold-time distribution + action PSD
      vs the plant corner frequency)?

  Q3  GAIN + DYNAMICS LEARNABILITY — are there enough clean isolated MV
      steps and DV steps, is DV decorrelated from MV (so ∂CV/∂MV and
      ∂CV/∂DV are separable), and is there enough settled steady-state
      data for the RSSM fixed point?

Usage:
    python tools/audit_rssm_training_data.py \
        --sim test_sim \
        --source-run output/test_sim/run_20260530_p68_rssm_baseline

Outputs a markdown report + plots + summary.json under
``output/<sim>/_rssm_data_audit_<ts>/``.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
_p = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
_p.add_argument('--sim', default=os.environ.get('AUDIT_SIM_NAME', 'test_sim'))
_p.add_argument('--source-run', default=os.environ.get('AUDIT_SOURCE_RUN', ''),
                help='Run dir to harvest plant-id / noise-config / '
                     'auto-tune from; auto-pick latest run_* if blank.')
_p.add_argument('--seed', type=int, default=int(os.environ.get('SEED', '0')))
_p.add_argument('--use-saved-autotune', action='store_true', default=True,
                help='Apply the run-saved auto_tune_seed_buffer.json values '
                     '(guarantees buffer matches the live run).')
_args = _p.parse_args()

SIM_NAME = _args.sim
SIM_DIR = REPO / 'simulation' / SIM_NAME
if not (SIM_DIR / 'control_setup.json').exists():
    raise SystemExit(f'[audit] no control_setup.json under {SIM_DIR}')

if _args.source_run.strip():
    SOURCE_RUN = Path(_args.source_run)
    if not SOURCE_RUN.is_absolute():
        SOURCE_RUN = REPO / SOURCE_RUN
else:
    _runs = sorted((REPO / 'output' / SIM_NAME).glob('run_*'),
                   key=lambda p: p.stat().st_mtime if p.exists() else 0)
    SOURCE_RUN = _runs[-1] if _runs else None
if SOURCE_RUN is None or not SOURCE_RUN.exists():
    raise SystemExit('[audit] need a --source-run with run_plan.json')

# --------------------------------------------------------------------------
# Environment wiring — point every loader at the run's own artifacts so the
# regenerated buffer is bit-faithful to the live run.
# --------------------------------------------------------------------------
os.environ['CONTROL_SETUP_JSON'] = str(SIM_DIR / 'control_setup.json')
os.environ['CONTROL_OBJECTIVE_JSON'] = str(SIM_DIR / 'control_objective.json')
os.environ['SIMULATION_DIR'] = str(SIM_DIR)
os.environ['SEED'] = str(_args.seed)
if (SOURCE_RUN / 'noise_config.json').exists():
    os.environ['SIM_NOISE_CONFIG_JSON'] = str(SOURCE_RUN / 'noise_config.json')
_dyn = SOURCE_RUN / 'plant_id' / 'dynamics_identification.json'
_lb = SOURCE_RUN / 'plant_id' / 'lookback_identification.json'
if _dyn.exists():
    os.environ['AGENT_DYNAMICS_JSON'] = str(_dyn)
if _lb.exists():
    os.environ['AGENT_LOOKBACK_JSON'] = str(_lb)

plan = json.loads((SOURCE_RUN / 'run_plan.json').read_text())
TAU = float(plan.get('tau') or 0.0)
DEAD = float(plan.get('dead_time') or 0.0)
TAU_FAST = float(plan.get('tau_fast') or TAU)
SAMPLE_RATE = int(plan.get('sample_rate') or 1)
EPISODE_LEN = int(plan['config'].get('episode_length'))
LOOKBACK = int(plan['config'].get('lookback'))
os.environ.setdefault('SIM_SAMPLE_RATE', str(SAMPLE_RATE))
os.environ.setdefault('IDENTIFIED_TAU_DOMINANT', f'{TAU:g}')
os.environ.setdefault('IDENTIFIED_DEAD_TIME', f'{DEAD:g}')

TS = time.strftime('%Y%m%d_%H%M%S')
OUT = REPO / f'output/{SIM_NAME}/_rssm_data_audit_{TS}'
(OUT / 'plots').mkdir(parents=True, exist_ok=True)
print(f'[audit] sim={SIM_NAME} source={SOURCE_RUN.name} -> {OUT}', flush=True)
print(f'[audit] tau={TAU} tau_fast={TAU_FAST} dead={DEAD} sr={SAMPLE_RATE} '
      f'ep_len={EPISODE_LEN} lookback={LOOKBACK}', flush=True)

# --------------------------------------------------------------------------
# Build cfg from run_plan, then env.
# --------------------------------------------------------------------------
from dataclasses import fields as _dc_fields  # noqa: E402
from training.train import (  # noqa: E402
    TrainConfig, APCEnv, auto_tune_seed_buffer,
    collect_baseline_episode, collect_episode, collect_prbs_episode,
    _seed_one_const_or_step, collect_step_test_episode,
)

_valid = {f.name for f in _dc_fields(TrainConfig)}
_cfg_kw = {k: v for k, v in plan['config'].items()
           if k in _valid and v is not None}
cfg = TrainConfig(**_cfg_kw)
cfg.lookback = LOOKBACK
cfg.sample_rate = SAMPLE_RATE
cfg.episode_length = EPISODE_LEN

rng = np.random.default_rng(_args.seed)
env = APCEnv(cfg, rng)

# --------------------------------------------------------------------------
# Apply auto-tune (saved values preferred — guarantees buffer == live run).
# --------------------------------------------------------------------------
AUTO_PATH = SOURCE_RUN / 'auto_tune_seed_buffer.json'
auto_applied: Dict[str, float] = {}
if _args.use_saved_autotune and AUTO_PATH.exists():
    saved = json.loads(AUTO_PATH.read_text())
    for field, info in saved.items():
        if field in _valid and isinstance(info, dict) and 'value' in info:
            setattr(cfg, field, info['value'])
            auto_applied[field] = info['value']
    print('[audit] applied SAVED auto-tune values', flush=True)
else:
    auto = auto_tune_seed_buffer(env, cfg)
    for field, info in auto.items():
        if field in _valid:
            setattr(cfg, field, info['value'])
            auto_applied[field] = info['value']
    print('[audit] applied FRESH auto-tune values', flush=True)
for k, v in auto_applied.items():
    print(f'    {k:32s} = {v}', flush=True)

# --------------------------------------------------------------------------
# Raw-state tap (reuse v2 pattern): capture pre-normalization sim state.
# --------------------------------------------------------------------------
_raw_states: List[List[np.ndarray]] = []
# Per-step UNMEASURED (hidden) disturbance signal, captured parallel to
# _raw_states so it can be correlated against MV / DV.  The WM must be able to
# tell the measured DV apart from the unmeasured load -> they must be
# uncorrelated; likewise disturbances vs MV, and MV vs MV (MIMO gain ID).
_hidden_states: List[List[np.ndarray]] = []
_orig_reset = env.reset
_orig_build_obs = env._build_obs_vec
_ep_meta: List[Dict] = []


def _build_obs_hook(state):
    try:
        s = np.asarray(state, dtype='float32').reshape(-1).copy()
        if _raw_states:
            _raw_states[-1].append(s)
        # tap the per-step hidden-disturbance offset (aligned to env.cv_indices)
        if _hidden_states:
            h = np.zeros(len(env.cv_indices), dtype='float32')
            hd = getattr(env, '_hidden_disturbance', None)
            if hd is not None:
                la = np.asarray(getattr(hd, 'last_applied', []),
                                dtype='float32').reshape(-1)
                for p, idx in enumerate(getattr(hd, 'cv_indices', [])):
                    if p < la.shape[0] and idx in env.cv_indices:
                        h[env.cv_indices.index(idx)] = la[p]
            _hidden_states[-1].append(h)
    except Exception:
        pass
    return _orig_build_obs(state)


def _reset_hook(*a, **kw):
    _raw_states.append([])
    _hidden_states.append([])
    out = _orig_reset(*a, **kw)
    _ep_meta.append({
        'sched_len': len(env._schedule or []),
        'hidden_active': env._hidden_disturbance is not None,
    })
    return out


env.reset = _reset_hook
env._build_obs_vec = _build_obs_hook

# --------------------------------------------------------------------------
# Channel metadata.
# --------------------------------------------------------------------------
state_vars = list(env.meta.get('state_variables') or [])
cv_idx = list(env.cv_indices)
dv_idx = list(env.meta.get('dv_indices') or [])
mv_idx = list(env.meta.get('mv_indices') or [])
A = int(env.action_dim)
mv_norm = [(float(lo), float(hi)) for lo, hi in env.mv_norm_ranges]

# Physical operating bounds from control_objective (preferred) else norm.
obj = json.loads((SIM_DIR / 'control_objective.json').read_text())
_obj_b = obj.get('bounds', {}) if isinstance(obj, dict) else {}


def _bounds_for(idx_list, key, fallback_ranges):
    raw = _obj_b.get(key)
    out = []
    if isinstance(raw, dict) and raw:
        keys = sorted(raw)
        for i in range(len(idx_list)):
            try:
                pair = raw[keys[i]] if i < len(keys) else None
                out.append((float(pair[0]), float(pair[1])))
            except Exception:
                out.append(fallback_ranges[i] if i < len(fallback_ranges)
                           else (0.0, 100.0))
    elif isinstance(raw, (list, tuple)) and raw:
        for i in range(len(idx_list)):
            try:
                out.append((float(raw[i][0]), float(raw[i][1])))
            except Exception:
                out.append(fallback_ranges[i] if i < len(fallback_ranges)
                           else (0.0, 100.0))
    else:
        out = list(fallback_ranges[:len(idx_list)]) or [(0.0, 100.0)] * len(idx_list)
    return out


cv_norm = [(float(lo), float(hi)) for lo, hi in env.cv_norm_ranges]
mv_op = _bounds_for(mv_idx, 'mvs', mv_norm)
cv_op = _bounds_for(cv_idx, 'outputs', cv_norm)
# DV physical bounds from the noise-config OU bounds (authoritative for
# this sim) then control_setup io ranges.
dv_op: List[Tuple[float, float]] = []
try:
    ncfg = json.loads((SOURCE_RUN / 'noise_config.json').read_text())
    dv_bounds_by_idx = {int(d['index']): tuple(d['bounds'])
                        for d in ncfg.get('ou_noise', [])
                        if str(d.get('channel_type')) == 'dv'}
except Exception:
    dv_bounds_by_idx = {}
for di in dv_idx:
    dv_op.append(tuple(map(float, dv_bounds_by_idx.get(di, (0.0, 100.0)))))

print(f'[audit] state_vars={state_vars}', flush=True)
print(f'[audit] cv_idx={cv_idx} op={cv_op}  dv_idx={dv_idx} op={dv_op}  '
      f'mv_dim={A} op={mv_op}', flush=True)


def _mv_phys(a_col: np.ndarray, j: int) -> np.ndarray:
    lo, hi = mv_op[j]
    return lo + 0.5 * (np.clip(a_col, -1.0, 1.0) + 1.0) * (hi - lo)


# --------------------------------------------------------------------------
# Replicate the EXACT seed-fill loop (training.train.train).
# Each entry: (episode_type, episode_dict, raw_state_array).
# --------------------------------------------------------------------------
EPISODES: List[Tuple[str, Dict, Optional[np.ndarray]]] = []
# Per-episode hidden-disturbance signal array (T, n_cv), index-aligned w/ EPISODES.
HIDDEN_BY_EP: List[Optional[np.ndarray]] = []


def _harvest(ep_type: str, ep: Dict):
    raw_list = _raw_states[-1] if _raw_states else []
    hid_list = _hidden_states[-1] if _hidden_states else []
    # First captured frame is the reset state; drop so signals align with act.
    if len(raw_list) > ep['act'].shape[0]:
        raw_list = raw_list[1:]
    if len(hid_list) > ep['act'].shape[0]:
        hid_list = hid_list[1:]
    raw = np.stack(raw_list, axis=0) if raw_list else None
    hid = np.stack(hid_list, axis=0) if hid_list else None
    EPISODES.append((ep_type, ep, raw))
    HIDDEN_BY_EP.append(hid)


n_baseline = int(getattr(cfg, 'baseline_seed_episodes', 0))
baseline_std = float(getattr(cfg, 'baseline_seed_action_std', 0.05))
n_random = int(getattr(cfg, 'random_seed_episodes', 0))
n_prbs = int(getattr(cfg, 'exploration_seed_episodes', 0))
prbs_band = float(getattr(cfg, 'prbs_seed_op_band', 0.95))
baseline_band = float(os.environ.get(
    'DREAMER_BASELINE_SEED_OP_BAND', str(min(0.6, prbs_band))))

# --- baseline (stratified centres) -------------------------------------
if n_baseline > 0:
    edges = np.linspace(-baseline_band, +baseline_band, n_baseline + 1)
    centres = env.rng.uniform(edges[:-1], edges[1:]).astype('float32')
    env.rng.shuffle(centres)
    for i in range(n_baseline):
        ep = collect_baseline_episode(env, cfg, action_std=baseline_std,
                                      center=float(centres[i]))
        _harvest('baseline', ep)
print(f'[audit] baseline={n_baseline}', flush=True)

# --- random ------------------------------------------------------------
for _ in range(max(0, n_random)):
    ep = collect_episode(env, None, None, cfg, random_action=True)
    _harvest('random', ep)
print(f'[audit] random={n_random}', flush=True)

# --- PRBS --------------------------------------------------------------
for _ in range(max(0, n_prbs)):
    ep = collect_prbs_episode(env, cfg, action_std=baseline_std,
                              op_band=prbs_band)
    _harvest('prbs', ep)
print(f'[audit] prbs={n_prbs}', flush=True)

# --- const-action / step-settle (interleaved) --------------------------
n_const = int(getattr(cfg, 'constant_action_seed_episodes', 0))
const_band = float(getattr(cfg, 'constant_action_seed_op_band', 0.6))
step_frac = float(np.clip(getattr(cfg, 'step_settle_seed_fraction', 0.0),
                          0.0, 1.0))
n_const_emit = n_step_settle_emit = 0
if n_const > 0:
    levels = np.linspace(-const_band, const_band, n_const, dtype='float32')
    jitter = env.rng.uniform(-0.05, 0.05, size=levels.shape).astype('float32')
    levels = np.clip(levels + jitter * const_band, -1.0, 1.0)
    n_step = int(round(step_frac * n_const))
    do_step = np.zeros(n_const, dtype=bool)
    if n_step > 0:
        do_step[np.linspace(0, n_const - 1, n_step, dtype=int)] = True
    for i, lvl in enumerate(levels):
        ep = _seed_one_const_or_step(env, cfg, level=float(lvl),
                                     do_step=bool(do_step[i]))
        if do_step[i]:
            _harvest('step_settle', ep)
            n_step_settle_emit += 1
        else:
            _harvest('const_action', ep)
            n_const_emit += 1
print(f'[audit] const_action={n_const_emit} step_settle={n_step_settle_emit}',
      flush=True)

# --- step-test ---------------------------------------------------------
n_st_floor = int(getattr(cfg, 'step_test_seed_episodes', 0))
n_per_ch = int(getattr(cfg, 'step_test_episodes_per_channel', 0))
n_mv = len(mv_idx)
n_dv = len(dv_idx)
n_channels = max(1, n_mv + n_dv)
n_st = max(n_st_floor, n_per_ch * n_channels)
if n_st > 0:
    st_levels = np.linspace(-const_band, const_band, n_st, dtype='float32')
    st_jit = env.rng.uniform(-0.05, 0.05, size=st_levels.shape).astype('float32')
    st_levels = np.clip(st_levels + st_jit * const_band, -1.0, 1.0)
    for ep_idx, lvl in enumerate(st_levels):
        primary = (ep_idx % n_dv) if n_dv > 0 else -1
        ep = collect_step_test_episode(env, cfg, initial_level=float(lvl),
                                       primary_dv_pos=int(primary))
        _harvest('step_test', ep)
print(f'[audit] step_test={n_st}', flush=True)

TYPES = ['baseline', 'random', 'prbs', 'const_action', 'step_settle',
         'step_test']
COLORS = {'baseline': '#1f77b4', 'random': '#9467bd', 'prbs': '#ff7f0e',
          'const_action': '#2ca02c', 'step_settle': '#d62728',
          'step_test': '#8c564b'}

# --------------------------------------------------------------------------
# Aggregate arrays per type.
# --------------------------------------------------------------------------
by_type: Dict[str, Dict[str, list]] = {
    t: {'act': [], 'cv': [[] for _ in cv_idx], 'dv': [[] for _ in dv_idx],
        'n_ep': 0, 'n_steps': 0}
    for t in TYPES}
for ep_type, ep, raw in EPISODES:
    d = by_type[ep_type]
    d['n_ep'] += 1
    d['n_steps'] += ep['act'].shape[0]
    d['act'].append(ep['act'])
    if raw is not None:
        for j, ci in enumerate(cv_idx):
            if ci < raw.shape[1]:
                d['cv'][j].append(raw[:, ci])
        for j, di in enumerate(dv_idx):
            if di < raw.shape[1]:
                d['dv'][j].append(raw[:, di])

total_steps = sum(d['n_steps'] for d in by_type.values())
total_eps = sum(d['n_ep'] for d in by_type.values())


def _cat(lst):
    return np.concatenate(lst) if lst else np.array([])


def _cov_stats(arr, lo, hi, bins=20):
    if arr.size == 0 or hi <= lo:
        return {'min': None, 'max': None, 'mean': None, 'std': None,
                'p1': None, 'p99': None, 'in_band_frac': 0.0,
                'bin_cov': 0.0, 'edge_cov': 0.0, 'imbalance': 0.0}
    h, _ = np.histogram(arr, bins=bins, range=(lo, hi))
    edge = float((h[0] > 0) and (h[-1] > 0))
    nz = h[h > 0]
    imbalance = float(h.max() / max(1, nz.min())) if nz.size else 0.0
    return {
        'min': float(arr.min()), 'max': float(arr.max()),
        'mean': float(arr.mean()), 'std': float(arr.std()),
        'p1': float(np.percentile(arr, 1)),
        'p99': float(np.percentile(arr, 99)),
        'in_band_frac': float(((arr >= lo) & (arr <= hi)).mean()),
        'bin_cov': float((h > 0).mean()), 'edge_cov': edge,
        'imbalance': imbalance,
    }


# --------------------------------------------------------------------------
# Q1: range coverage (buffer-weighted, combined over all types).
# --------------------------------------------------------------------------
all_act = [[] for _ in range(A)]
all_cv = [[] for _ in cv_idx]
all_dv = [[] for _ in dv_idx]
for ep_type, ep, raw in EPISODES:
    for j in range(A):
        all_act[j].append(_mv_phys(ep['act'][:, j], j))
    if raw is not None:
        for j, ci in enumerate(cv_idx):
            if ci < raw.shape[1]:
                all_cv[j].append(raw[:, ci])
        for j, di in enumerate(dv_idx):
            if di < raw.shape[1]:
                all_dv[j].append(raw[:, di])

coverage = {'mv': [], 'cv': [], 'dv': []}
for j in range(A):
    coverage['mv'].append(
        {'op': mv_op[j], **_cov_stats(_cat(all_act[j]), *mv_op[j])})
for j in range(len(cv_idx)):
    coverage['cv'].append(
        {'op': cv_op[j], **_cov_stats(_cat(all_cv[j]), *cv_op[j])})
for j in range(len(dv_idx)):
    coverage['dv'].append(
        {'op': dv_op[j], **_cov_stats(_cat(all_dv[j]), *dv_op[j])})

# --------------------------------------------------------------------------
# Q2: disturbance mix — hold-time + spectral content.
# --------------------------------------------------------------------------
# Settle/transient classification thresholds (agent steps).
settle_steps = max(8, int(round(4.0 * TAU / max(1, SAMPLE_RATE))))   # ~4tau
transient_steps = max(2, int(round(TAU / (3.0 * max(1, SAMPLE_RATE)))))  # ~tau/3
corner_f = 1.0 / (2.0 * np.pi * max(1e-6, TAU / max(1, SAMPLE_RATE)))  # cyc/step


def _hold_runs(act_2d: np.ndarray, eps: float = 1e-3) -> List[int]:
    """Run-lengths of (near-)constant action (channel-0). Clean held
    seeds (const/step/step-test) yield true segment lengths; noisy seeds
    yield length-1 runs (correctly: no held steady state)."""
    a = act_2d[:, 0]
    if a.size == 0:
        return []
    chg = np.abs(np.diff(a)) > eps
    runs = []
    cur = 1
    for c in chg:
        if c:
            runs.append(cur)
            cur = 1
        else:
            cur += 1
    runs.append(cur)
    return runs


def _settled_fraction(act_2d: np.ndarray, eps: float = 1e-3) -> float:
    """Fraction of steps inside a settled window: action range over the
    trailing ``settle_steps`` < eps (i.e. plant has time to reach SS)."""
    a = act_2d[:, 0]
    T = a.size
    if T <= settle_steps:
        return 0.0
    settled = 0
    for t in range(settle_steps, T):
        w = a[t - settle_steps:t + 1]
        if (w.max() - w.min()) < eps:
            settled += 1
    return settled / (T - settle_steps)


def _psd(act_2d: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Welch-ish PSD of channel-0 action (detrended, Hann, averaged over
    50%-overlap segments)."""
    a = act_2d[:, 0].astype('float64')
    n = a.size
    seg = min(256, n)
    if seg < 16:
        return np.array([]), np.array([])
    step = seg // 2
    win = np.hanning(seg)
    acc = None
    cnt = 0
    for s in range(0, n - seg + 1, step):
        x = a[s:s + seg]
        x = (x - x.mean()) * win
        p = np.abs(np.fft.rfft(x)) ** 2
        acc = p if acc is None else acc + p
        cnt += 1
    if cnt == 0 or acc is None:
        return np.array([]), np.array([])
    freqs = np.fft.rfftfreq(seg, d=1.0)  # cycles/step
    return freqs, acc / cnt


hold_by_type: Dict[str, List[int]] = {}
settled_by_type: Dict[str, float] = {}
for t in TYPES:
    runs = []
    sfracs = []
    for ep_type, ep, raw in EPISODES:
        if ep_type != t:
            continue
        runs.extend(_hold_runs(ep['act']))
        sfracs.append(_settled_fraction(ep['act']))
    hold_by_type[t] = runs
    settled_by_type[t] = float(np.mean(sfracs)) if sfracs else 0.0

# Buffer-wide settled fraction (step-weighted).
settled_overall = 0.0
if total_steps:
    settled_overall = sum(settled_by_type[t] * by_type[t]['n_steps']
                          for t in TYPES) / total_steps

# Excitation-timescale: pooled hold-times of the CLEAN held seeds
# (const + step_settle + step_test) classify transient vs settled holds.
clean_holds = (hold_by_type['const_action'] + hold_by_type['step_settle']
               + hold_by_type['step_test'])
ch = np.asarray(clean_holds, dtype=float) if clean_holds else np.array([])
hold_timescale = {
    'n_holds': int(ch.size),
    'transient_frac': float((ch < transient_steps).mean()) if ch.size else 0.0,
    'mid_frac': float(((ch >= transient_steps) & (ch < settle_steps)).mean())
    if ch.size else 0.0,
    'settled_frac': float((ch >= settle_steps).mean()) if ch.size else 0.0,
    'median_hold': float(np.median(ch)) if ch.size else 0.0,
    'max_hold': float(ch.max()) if ch.size else 0.0,
    'transient_steps_thresh': transient_steps,
    'settle_steps_thresh': settle_steps,
}

# Action PSD pooled over the excitation seeds (prbs + step_test).
psd_freqs, psd_pow = None, None
psd_act = [ep['act'] for et, ep, _ in EPISODES if et in ('prbs', 'step_test')]
if psd_act:
    pooled = np.concatenate(psd_act, axis=0)
    psd_freqs, psd_pow = _psd(pooled)
psd_lowfreq_frac = 0.0
if psd_pow is not None and psd_pow.size:
    mask = psd_freqs <= corner_f
    psd_lowfreq_frac = float(psd_pow[mask].sum() / max(1e-12, psd_pow.sum()))

# --------------------------------------------------------------------------
# Q3: gain + dynamics learnability.
# --------------------------------------------------------------------------
# (a) Clean isolated MV steps (step_test + step_settle): count + magnitude.
mv_steps = []  # (phys_delta, sign)
for et, ep, raw in EPISODES:
    if et not in ('step_test', 'step_settle'):
        continue
    a0 = ep['act'][:, 0]
    chg = np.where(np.abs(np.diff(a0)) > 1e-3)[0]
    for k in chg:
        d_norm = float(a0[k + 1] - a0[k])
        lo, hi = mv_op[0]
        d_phys = 0.5 * d_norm * (hi - lo)
        mv_steps.append(d_phys)
mv_steps = np.asarray(mv_steps, dtype=float)
mv_step_inventory = {
    'n_steps': int(mv_steps.size),
    'median_abs_phys': float(np.median(np.abs(mv_steps))) if mv_steps.size else 0.0,
    'min_abs_phys': float(np.min(np.abs(mv_steps))) if mv_steps.size else 0.0,
    'max_abs_phys': float(np.max(np.abs(mv_steps))) if mv_steps.size else 0.0,
    'up_frac': float((mv_steps > 0).mean()) if mv_steps.size else 0.0,
}

# (b) DV excitation: range covered + explicit DV step events in step-test.
dv_excitation = []
for j, di in enumerate(dv_idx):
    arr = _cat(all_dv[j])
    lo, hi = dv_op[j]
    # DV step events: count large jumps (> 5% of DV span) in step-test eps.
    n_dv_events = 0
    span = hi - lo
    for et, ep, raw in EPISODES:
        if et != 'step_test' or raw is None or di >= raw.shape[1]:
            continue
        dv_t = raw[:, di]
        jumps = np.abs(np.diff(dv_t)) > 0.05 * span
        n_dv_events += int(jumps.sum())
    dv_excitation.append({
        'name': state_vars[di] if di < len(state_vars) else f'dv_{j}',
        'op': (lo, hi),
        'min': float(arr.min()) if arr.size else None,
        'max': float(arr.max()) if arr.size else None,
        'std': float(arr.std()) if arr.size else None,
        'range_cov_frac': (float((arr.max() - arr.min()) / max(1e-9, span))
                           if arr.size else 0.0),
        'bin_cov': _cov_stats(arr, lo, hi)['bin_cov'],
        'n_step_events_step_test': n_dv_events,
    })

# (c) Excitation DECORRELATION: for the WM to identify ∂CV/∂MV vs ∂CV/∂DV vs
# the unmeasured-disturbance signature, the excitation sources must be mutually
# UNCORRELATED (and MVs mutually uncorrelated for per-MV gain identifiability).
# Build per-step physical series for every MV / DV / hidden channel, concatenate
# across the whole seed buffer, then Pearson-correlate every cross-group pair.
def _series_or_none(parts):
    if not parts:
        return None
    v = _cat(parts)
    return v if (v.size > 10 and float(v.std()) > 1e-9) else None


_mv_series = {
    f'MV{j}': _series_or_none([_mv_phys(ep['act'][:, j], j)
                              for et, ep, raw in EPISODES
                              if ep['act'].shape[1] > j])
    for j in range(A)}
_dv_series = {
    f'DV{k}': _series_or_none([raw[:, di] for et, ep, raw in EPISODES
                              if raw is not None and di < raw.shape[1]])
    for k, di in enumerate(dv_idx)}
_hid_series = {
    f'HID_CV{c}': _series_or_none(
        [HIDDEN_BY_EP[i][:, c] for i in range(len(EPISODES))
         if HIDDEN_BY_EP[i] is not None and c < HIDDEN_BY_EP[i].shape[1]])
    for c in range(len(cv_idx))}


def _corr_between(a, b):
    if a is None or b is None:
        return None
    n = min(a.size, b.size)
    if n <= 10:
        return None
    aa, bb = a[:n], b[:n]
    if float(aa.std()) < 1e-9 or float(bb.std()) < 1e-9:
        return None
    return float(np.corrcoef(aa, bb)[0, 1])


def _max_abs_cross(group_a, group_b, same=False):
    keys_a, keys_b = list(group_a), list(group_b)
    best, best_pair = None, None
    for ia, ka in enumerate(keys_a):
        for ib, kb in enumerate(keys_b):
            if same and ib <= ia:
                continue
            c = _corr_between(group_a[ka], group_b[kb])
            if c is None:
                continue
            if best is None or abs(c) > abs(best):
                best, best_pair = c, [ka, kb]
    return best, best_pair


# back-compat scalar (MV0 x DV0) still drives the occupancy plot title
mv_dv_corr = _corr_between(_mv_series.get('MV0'), _dv_series.get('DV0'))
mv_dv_max, mv_dv_pair = _max_abs_cross(_mv_series, _dv_series)
mv_hid_max, mv_hid_pair = _max_abs_cross(_mv_series, _hid_series)
dv_hid_max, dv_hid_pair = _max_abs_cross(_dv_series, _hid_series)
mv_mv_max, mv_mv_pair = _max_abs_cross(_mv_series, _mv_series, same=True)

_hid_all = [HIDDEN_BY_EP[i] for i in range(len(EPISODES))
            if HIDDEN_BY_EP[i] is not None]
if _hid_all:
    _hid_cat = np.concatenate(
        [h.reshape(h.shape[0], -1) for h in _hid_all], axis=0)
    hidden_active_step_frac = float(np.mean(np.abs(_hid_cat) > 1e-6))
else:
    hidden_active_step_frac = 0.0

excitation_corr = {
    'mv_dv_max_abs': mv_dv_max, 'mv_dv_pair': mv_dv_pair,
    'mv_hidden_max_abs': mv_hid_max, 'mv_hidden_pair': mv_hid_pair,
    'dv_hidden_max_abs': dv_hid_max, 'dv_hidden_pair': dv_hid_pair,
    'mv_mv_max_abs': mv_mv_max, 'mv_mv_pair': mv_mv_pair,
    'hidden_active_step_frac': hidden_active_step_frac,
    'n_mv': A, 'n_dv': len(dv_idx), 'n_cv': len(cv_idx),
}

# (d) Steady-state gain sampling: held-MV vs settled-CV scatter
# (const_action late-window SS).
ss_pairs = []  # (mv_phys, cv_phys)
for et, ep, raw in EPISODES:
    if et != 'const_action' or raw is None or cv_idx[0] >= raw.shape[1]:
        continue
    mv_ss = float(_mv_phys(ep['act'][-100:, 0], 0).mean())
    cv_ss = float(raw[-100:, cv_idx[0]].mean())
    ss_pairs.append((mv_ss, cv_ss))
ss_pairs = np.asarray(ss_pairs) if ss_pairs else np.zeros((0, 2))

# --------------------------------------------------------------------------
# Composition table.
# --------------------------------------------------------------------------
composition = []
for t in TYPES:
    d = by_type[t]
    composition.append({
        'type': t, 'n_ep': d['n_ep'], 'n_steps': d['n_steps'],
        'pct_steps': (100.0 * d['n_steps'] / total_steps) if total_steps else 0.0,
        'settled_frac': settled_by_type[t],
        'hidden_fire_rate': float(np.mean(
            [m['hidden_active'] for (et, _, _), m in zip(EPISODES, _ep_meta)
             if et == t] or [0.0])),
    })

prbs_like = by_type['prbs']['n_steps'] + by_type['random']['n_steps']
other_dist = (by_type['baseline']['n_steps'] + by_type['const_action']['n_steps']
              + by_type['step_settle']['n_steps'] + by_type['step_test']['n_steps'])

summary = {
    'source_run': SOURCE_RUN.name,
    'plant': {'tau': TAU, 'tau_fast': TAU_FAST, 'dead_time': DEAD,
              'sample_rate': SAMPLE_RATE, 'episode_length': EPISODE_LEN,
              'corner_freq_cyc_per_step': corner_f},
    'auto_tuned': auto_applied,
    'totals': {'n_episodes': total_eps, 'n_steps': total_steps,
               'prbs_like_step_pct': 100.0 * prbs_like / max(1, total_steps),
               'structured_step_pct': 100.0 * other_dist / max(1, total_steps)},
    'composition': composition,
    'coverage': coverage,
    'hold_timescale': hold_timescale,
    'settled_overall_frac': settled_overall,
    'psd_lowfreq_frac': psd_lowfreq_frac,
    'mv_step_inventory': mv_step_inventory,
    'dv_excitation': dv_excitation,
    'mv_dv_corr': mv_dv_corr,
    'excitation_corr': excitation_corr,
    'n_ss_gain_samples': int(ss_pairs.shape[0]),
}
(OUT / 'summary.json').write_text(json.dumps(summary, indent=2, default=float))

# --------------------------------------------------------------------------
# Plots.
# --------------------------------------------------------------------------
# 1. Per-channel coverage histograms (MV, CV, DV) buffer-weighted.
nrows = A + len(cv_idx) + len(dv_idx)
fig, axes = plt.subplots(nrows, 1, figsize=(10, 2.3 * nrows), squeeze=False)
r = 0
for j in range(A):
    arr = _cat(all_act[j])
    lo, hi = mv_op[j]
    axes[r, 0].hist(arr, bins=40, range=(lo, hi), color='#1f77b4', alpha=0.8)
    axes[r, 0].axvline(lo, c='k', lw=0.5); axes[r, 0].axvline(hi, c='k', lw=0.5)
    c = coverage['mv'][j]
    axes[r, 0].set_title(f"MV[{j}] phys  bin_cov={c['bin_cov']:.2f} "
                         f"edge={c['edge_cov']:.0f} imbalance={c['imbalance']:.0f}x "
                         f"range=[{c['min']:.1f},{c['max']:.1f}] op=[{lo:.0f},{hi:.0f}]")
    r += 1
for j in range(len(cv_idx)):
    arr = _cat(all_cv[j])
    lo, hi = cv_op[j]
    axes[r, 0].hist(arr, bins=40, range=(lo, hi), color='#2ca02c', alpha=0.8)
    axes[r, 0].axvline(lo, c='k', lw=0.5); axes[r, 0].axvline(hi, c='k', lw=0.5)
    c = coverage['cv'][j]
    nm = state_vars[cv_idx[j]] if cv_idx[j] < len(state_vars) else f'cv_{j}'
    axes[r, 0].set_title(f"CV {nm}  bin_cov={c['bin_cov']:.2f} "
                         f"imbalance={c['imbalance']:.0f}x "
                         f"range=[{c['min']:.1f},{c['max']:.1f}] op=[{lo:.0f},{hi:.0f}]")
    r += 1
for j in range(len(dv_idx)):
    arr = _cat(all_dv[j])
    lo, hi = dv_op[j]
    axes[r, 0].hist(arr, bins=40, range=(lo, hi), color='#ff7f0e', alpha=0.8)
    axes[r, 0].axvline(lo, c='k', lw=0.5); axes[r, 0].axvline(hi, c='k', lw=0.5)
    c = coverage['dv'][j]
    nm = state_vars[dv_idx[j]] if dv_idx[j] < len(state_vars) else f'dv_{j}'
    axes[r, 0].set_title(f"DV {nm}  bin_cov={c['bin_cov']:.2f} "
                         f"range=[{c['min']:.1f},{c['max']:.1f}] op=[{lo:.0f},{hi:.0f}]")
    r += 1
fig.suptitle('Buffer-weighted per-channel coverage (physical units)')
fig.tight_layout()
fig.savefig(OUT / 'plots' / '01_coverage.png', dpi=130)
plt.close(fig)

# 2. Composition + hold-time distribution.
fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
pcts = [c['pct_steps'] for c in composition]
ax[0].bar(TYPES, pcts, color=[COLORS[t] for t in TYPES])
ax[0].set_ylabel('% of buffer steps')
ax[0].set_title('Seed-buffer composition (step-weighted)')
ax[0].tick_params(axis='x', rotation=30)
for t in ('const_action', 'step_settle', 'step_test', 'prbs'):
    h = np.asarray(hold_by_type[t], dtype=float)
    h = h[h > 1]
    if h.size:
        ax[1].hist(h, bins=40, alpha=0.5, label=t, color=COLORS[t])
ax[1].axvline(transient_steps, c='b', ls='--', lw=1, label=f'~tau/3={transient_steps}')
ax[1].axvline(settle_steps, c='r', ls='--', lw=1, label=f'~4tau={settle_steps}')
ax[1].set_xlabel('hold-time (agent steps)')
ax[1].set_title('Held-segment length distribution (timescale excitation)')
ax[1].legend(fontsize=8)
fig.tight_layout()
fig.savefig(OUT / 'plots' / '02_composition_holdtime.png', dpi=130)
plt.close(fig)

# 3. Action PSD vs plant corner.
if psd_pow is not None and psd_pow.size:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.loglog(psd_freqs[1:], psd_pow[1:], color='#ff7f0e')
    ax.axvline(corner_f, c='r', ls='--', label=f'plant corner f={corner_f:.4f}')
    ax.set_xlabel('frequency (cycles/agent-step)')
    ax.set_ylabel('action power')
    ax.set_title(f'Excitation PSD (prbs+step_test)  '
                 f'low-freq power frac={psd_lowfreq_frac:.2f}')
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / 'plots' / '03_action_psd.png', dpi=130)
    plt.close(fig)

# 4. MV x DV occupancy + SS gain scatter.
fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
if dv_idx:
    mv_all = _cat([_mv_phys(ep['act'][:, 0], 0) for et, ep, raw in EPISODES
                   if raw is not None and dv_idx[0] < raw.shape[1]])
    dv_all = _cat([raw[:, dv_idx[0]] for et, ep, raw in EPISODES
                   if raw is not None and dv_idx[0] < raw.shape[1]])
    n = min(mv_all.size, dv_all.size)
    if n:
        H, xe, ye = np.histogram2d(
            mv_all[:n], dv_all[:n], bins=25,
            range=[list(mv_op[0]), list(dv_op[0])])
        ax[0].imshow(np.log1p(H.T), origin='lower', aspect='auto',
                     extent=[*mv_op[0], *dv_op[0]], cmap='viridis')
        ax[0].set_xlabel('MV phys'); ax[0].set_ylabel('DV phys')
        ax[0].set_title(f'MV x DV occupancy (corr={mv_dv_corr:.3f})')
if ss_pairs.shape[0]:
    ax[1].scatter(ss_pairs[:, 0], ss_pairs[:, 1], color='#2ca02c')
    ax[1].axhline(cv_op[0][0], c='k', lw=0.4)
    ax[1].axhline(cv_op[0][1], c='k', lw=0.4)
    ax[1].set_xlabel('held MV phys (last-100 mean)')
    ax[1].set_ylabel('settled CV phys (last-100 mean)')
    ax[1].set_title('Steady-state gain sampling (const_action)')
fig.tight_layout()
fig.savefig(OUT / 'plots' / '04_gain_learnability.png', dpi=130)
plt.close(fig)

# --------------------------------------------------------------------------
# Markdown report.
# --------------------------------------------------------------------------
def _verdict(ok: bool) -> str:
    return 'PASS' if ok else 'CONCERN'


lines: List[str] = []
lines.append(f'# RSSM Training-Data Audit — {SIM_NAME}\n')
lines.append(f'- Source run: `{SOURCE_RUN.name}`')
lines.append(f'- Plant: tau={TAU}, tau_fast={TAU_FAST}, dead_time={DEAD}, '
             f'sample_rate={SAMPLE_RATE}, episode_length={EPISODE_LEN}')
lines.append(f'- Buffer: {total_eps} episodes, {total_steps:,} steps')
lines.append(f'- Generated: {TS}\n')

lines.append('## Auto-tuned seed parameters (applied)\n')
for k, v in auto_applied.items():
    lines.append(f'- `{k}` = {v}')
lines.append('')

lines.append('## Buffer composition\n')
lines.append('| type | episodes | steps | % buffer | settled frac | '
             'hidden-dist fire |')
lines.append('|---|---:|---:|---:|---:|---:|')
for c in composition:
    lines.append(f"| {c['type']} | {c['n_ep']} | {c['n_steps']:,} | "
                 f"{c['pct_steps']:.1f}% | {c['settled_frac']:.2f} | "
                 f"{c['hidden_fire_rate']:.2f} |")
lines.append(f"\n- PRBS-like (prbs+random) excitation: "
             f"**{summary['totals']['prbs_like_step_pct']:.1f}%** of steps")
lines.append(f"- Structured (baseline/const/step/step-test): "
             f"**{summary['totals']['structured_step_pct']:.1f}%** of steps\n")

# Q1
mv_ok = all(c['bin_cov'] >= 0.9 and c['edge_cov'] >= 1 for c in coverage['mv'])
cv_ok = all(c['bin_cov'] >= 0.7 for c in coverage['cv'])
dv_ok = all(d['range_cov_frac'] >= 0.5 for d in dv_excitation) if dv_excitation else True
lines.append(f'## Q1 — Range coverage: {_verdict(mv_ok and cv_ok and dv_ok)}\n')
lines.append('| channel | op-band | observed range | bin cov | edge | imbalance |')
lines.append('|---|---|---|---:|---:|---:|')
for j, c in enumerate(coverage['mv']):
    lines.append(f"| MV[{j}] | [{c['op'][0]:.0f},{c['op'][1]:.0f}] | "
                 f"[{c['min']:.1f},{c['max']:.1f}] | {c['bin_cov']:.2f} | "
                 f"{c['edge_cov']:.0f} | {c['imbalance']:.0f}x |")
for j, c in enumerate(coverage['cv']):
    nm = state_vars[cv_idx[j]] if cv_idx[j] < len(state_vars) else f'cv_{j}'
    lines.append(f"| CV {nm} | [{c['op'][0]:.0f},{c['op'][1]:.0f}] | "
                 f"[{c['min']:.1f},{c['max']:.1f}] | {c['bin_cov']:.2f} | "
                 f"{c['edge_cov']:.0f} | {c['imbalance']:.0f}x |")
for j, c in enumerate(coverage['dv']):
    nm = state_vars[dv_idx[j]] if dv_idx[j] < len(state_vars) else f'dv_{j}'
    lines.append(f"| DV {nm} | [{c['op'][0]:.0f},{c['op'][1]:.0f}] | "
                 f"[{c['min']:.1f},{c['max']:.1f}] | {c['bin_cov']:.2f} | "
                 f"{c['edge_cov']:.0f} | {c['imbalance']:.0f}x |")
lines.append('')

# Q2
ts = hold_timescale
mix_ok = (summary['totals']['prbs_like_step_pct'] >= 5.0
          and ts['transient_frac'] > 0.0 and ts['settled_frac'] > 0.0)
lines.append(f'## Q2 — Disturbance mix & timescale: {_verdict(mix_ok)}\n')
lines.append(f"- Held-segment timescale split (clean seeds, n={ts['n_holds']}): "
             f"transient(<{ts['transient_steps_thresh']})={ts['transient_frac']:.2f}, "
             f"mid={ts['mid_frac']:.2f}, "
             f"settled(>={ts['settle_steps_thresh']})={ts['settled_frac']:.2f}")
lines.append(f"- Median hold={ts['median_hold']:.0f} steps, "
             f"max hold={ts['max_hold']:.0f} steps")
lines.append(f"- Buffer settled-fraction (step-weighted): {settled_overall:.2f}")
lines.append(f"- Excitation PSD low-frequency power (<= plant corner "
             f"{corner_f:.4f} cyc/step): {psd_lowfreq_frac:.2f}")
lines.append('')

# Q3
_xc = excitation_corr


def _decorr_ok(v):
    return v is None or abs(v) < 0.5


decorr_ok = (_decorr_ok(_xc['mv_dv_max_abs'])
             and _decorr_ok(_xc['mv_hidden_max_abs'])
             and _decorr_ok(_xc['dv_hidden_max_abs'])
             and _decorr_ok(_xc['mv_mv_max_abs']))
gain_ok = (mv_step_inventory['n_steps'] >= 10
           and decorr_ok
           and ss_pairs.shape[0] >= 5)
lines.append(f'## Q3 — Gain & dynamics learnability: {_verdict(gain_ok)}\n')
lines.append(f"- Clean isolated MV steps: {mv_step_inventory['n_steps']} "
             f"(|Δ| phys median={mv_step_inventory['median_abs_phys']:.2f}, "
             f"range [{mv_step_inventory['min_abs_phys']:.2f},"
             f"{mv_step_inventory['max_abs_phys']:.2f}], "
             f"up-frac={mv_step_inventory['up_frac']:.2f})")
for d in dv_excitation:
    lines.append(f"- DV {d['name']}: range_cov={d['range_cov_frac']:.2f}, "
                 f"bin_cov={d['bin_cov']:.2f}, "
                 f"explicit step events (step-test)={d['n_step_events_step_test']}")


def _fc(v):
    return f"{v:+.3f}" if v is not None else "n/a"


lines.append("- Excitation decorrelation (|corr|<0.3 good, >=0.5 CONCERN — "
             "sources must be uncorrelated so the WM can attribute ∂CV to the "
             "right cause):")
lines.append(f"    MV–DV      max|corr|={_fc(_xc['mv_dv_max_abs'])} "
             f"{_xc['mv_dv_pair'] or ''}")
lines.append(f"    MV–hidden  max|corr|={_fc(_xc['mv_hidden_max_abs'])} "
             f"{_xc['mv_hidden_pair'] or ''}")
lines.append(f"    DV–hidden  max|corr|={_fc(_xc['dv_hidden_max_abs'])} "
             f"{_xc['dv_hidden_pair'] or ''}")
lines.append(f"    MV–MV      max|corr|={_fc(_xc['mv_mv_max_abs'])} "
             f"{_xc['mv_mv_pair'] or '(SISO / n/a)'}")
lines.append(f"    hidden active in {_xc['hidden_active_step_frac'] * 100:.1f}% "
             f"of seed steps (if ~0 the DV/hidden decorrelation is trivial — "
             f"they rarely co-excite in the SEED buffer; on-policy data not "
             f"audited here)")
lines.append(f"- Steady-state gain samples (const-action): {ss_pairs.shape[0]}")
lines.append('')

lines.append('## Plots\n')
for p in ('01_coverage.png', '02_composition_holdtime.png',
          '03_action_psd.png', '04_gain_learnability.png'):
    lines.append(f'- `plots/{p}`')
lines.append('')

(OUT / 'audit_report.md').write_text('\n'.join(lines))
print('\n' + '\n'.join(lines), flush=True)
print(f'\n[audit] DONE -> {OUT}', flush=True)
