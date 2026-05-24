"""Data-generation quality audit for DreamerV4 seed/collection pipeline.

Generates a small batch of episodes from each "data domain":
  - baseline_seed      (low-noise around mid-MV)
  - prbs_seed          (stratified PRBS sweep)
  - const_action_seed  (single op-point, steady-state probe)
  - step_settle_seed   (u0 then u1 transient + settled tail)
  - random             (uniform [-1,1], the P1/P2 collection action)

For each domain:
  - histograms of MV, CV, SP (delta-from-target) and 2D MV x CV occupancy
  - autocorrelation of actions (how 'random' is random)
  - per-step coverage of operating quadrants
  - SNR per CV channel from successive differences
  - timeseries plots of 3 representative episodes
  - DR realization (per-episode output_gain / bias)

Outputs:  /tmp/data_audit_<timestamp>/{plots/, summary.json, report.md}

This is a read-only audit — no training is performed.
"""
from __future__ import annotations

import json
import os
import sys
import math
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ---- Repo setup (mirror single_run.py prelude) -----------------------------
REPO = Path('/home/koitkam/neural-apc-dreamerV4')
sys.path.insert(0, str(REPO))

SIM_DIR = REPO / 'simulation' / 'test_sim'
# Reuse an existing run's identification artifacts so we mirror real training.
SOURCE_RUN = REPO / 'output' / 'test_sim' / 'run_20260523_p42_lambda090'
assert SOURCE_RUN.exists(), f'expected source run not found: {SOURCE_RUN}'

os.environ['CONTROL_SETUP_JSON']     = str(SIM_DIR / 'control_setup.json')
os.environ['CONTROL_OBJECTIVE_JSON'] = str(SIM_DIR / 'control_objective.json')
os.environ['SIMULATION_DIR']         = str(SIM_DIR)
os.environ['SEED']                   = '12345'
# Point at the noise + episode-length settings the real P42 run resolved to.
noise_json = SOURCE_RUN / 'noise_config.json'
if noise_json.exists():
    os.environ['SIM_NOISE_CONFIG_JSON'] = str(noise_json)
# Read tau/dt/sr/episode_length/lookback from the source run plan
plan = json.loads((SOURCE_RUN / 'run_plan.json').read_text())
TAU         = float(plan.get('tau', 53.0))
DEAD        = float(plan.get('dead_time', 8.0))
SAMPLE_RATE = int(plan.get('sample_rate', 4))
EPISODE_LEN = int(plan['config']['episode_length'])
LOOKBACK    = int(plan['config']['lookback'])
os.environ['SIM_SAMPLE_RATE']      = str(SAMPLE_RATE)
os.environ['IDENTIFIED_TAU_DOMINANT'] = f'{TAU:g}'
os.environ['IDENTIFIED_DEAD_TIME']    = f'{DEAD:g}'
os.environ['SIM_EPISODE_LENGTH']      = str(EPISODE_LEN)
# Keep DR + hidden disturbance ON to mirror training conditions
# (defaults are already ON; just don't disable them).

OUT_DIR = REPO / f'output/test_sim/_data_audit_{time.strftime("%Y%m%d_%H%M%S")}'
(OUT_DIR / 'plots').mkdir(parents=True, exist_ok=True)
print(f'[audit] writing to {OUT_DIR}', flush=True)
print(f'[audit] plant tau={TAU} dead={DEAD} sr={SAMPLE_RATE} '
      f'episode_len={EPISODE_LEN} lookback={LOOKBACK}', flush=True)

# ---- Build env using the same TrainConfig path as training ----------------
from training.train import (
    TrainConfig, APCEnv,
    collect_baseline_episode, collect_prbs_episode,
    collect_constant_action_episode, collect_step_settle_episode,
)

cfg = TrainConfig(
    lookback=LOOKBACK,
    sample_rate=SAMPLE_RATE,
    episode_length=EPISODE_LEN,
)
# Match the live PRBS stratification width to what training used.
cfg.prbs_seed_n_strata = 8
cfg.constant_action_seed_op_band = 0.6

rng = np.random.default_rng(int(os.environ['SEED']))
env = APCEnv(cfg, rng)
print(f'[audit] env built: action_dim={env.action_dim} state_dim={env.state_dim} '
      f'obs_dim={env.obs_dim} aug_dim={env.aug_obs_dim} cv_idx={env.cv_indices}',
      flush=True)
print(f'[audit] mv_norm_ranges={env.mv_norm_ranges}', flush=True)
print(f'[audit] cv_norm_ranges={env.cv_norm_ranges}', flush=True)

# ---- Random-action collector for P1/P2 mirror -----------------------------
def collect_random_episode(env: APCEnv, cfg: TrainConfig) -> Dict[str, np.ndarray]:
    """P1/P2 collection: uniform [-1, 1] per step (mirrors training/train.py)."""
    obs_window = env.reset(exploration=True)
    T = cfg.episode_length
    D = env.obs_dim
    # Phase 2 (2026-05-24): per-step storage (T, D), no L axis.
    obs_buf  = np.zeros((T, D), dtype='float32')
    act_buf  = np.zeros((T, env.action_dim), dtype='float32')
    rew_buf  = np.zeros(T, dtype='float32')
    cont_buf = np.ones(T, dtype='float32')
    for t in range(T):
        obs_buf[t] = obs_window[-1]
        a = env.rng.uniform(-1.0, 1.0, size=env.action_dim).astype('float32')
        nxt, r, done, _ = env.step(a)
        act_buf[t] = a
        rew_buf[t] = r
        cont_buf[t] = 0.0 if (done and t == T - 1) else 1.0
        obs_window = nxt
        if done:
            break
    return {'obs': obs_buf, 'act': act_buf, 'rew': rew_buf, 'cont': cont_buf}

# ---- Collect N episodes per domain ----------------------------------------
N_PER_DOMAIN = 12
op_band = 0.6
const_levels = np.linspace(-op_band, op_band, N_PER_DOMAIN, dtype='float32')

DOMAINS = {}
DOMAINS['baseline_seed']      = [
    lambda: collect_baseline_episode(env, cfg, action_std=0.05)
    for _ in range(N_PER_DOMAIN)]
DOMAINS['prbs_seed']          = [
    lambda: collect_prbs_episode(env, cfg, action_std=0.05, op_band=0.95)
    for _ in range(N_PER_DOMAIN)]
DOMAINS['const_action_seed']  = [
    (lambda lv=float(L): collect_constant_action_episode(env, cfg, action_level=lv))
    for L in const_levels]
DOMAINS['step_settle_seed']   = [
    lambda: collect_step_settle_episode(
        env, cfg, action_start=float(env.rng.uniform(-0.5, 0.5)),
        action_end=float(env.rng.uniform(-0.5, 0.5)),
        switch_step=int(env.rng.uniform(0.3, 0.7) * EPISODE_LEN),
    )
    for _ in range(N_PER_DOMAIN)]
DOMAINS['random_action']      = [
    lambda: collect_random_episode(env, cfg)
    for _ in range(N_PER_DOMAIN)]

EPISODES = {}  # name -> list of dicts
for name, makers in DOMAINS.items():
    eps = []
    for i, mk in enumerate(makers):
        try:
            ep = mk()
            eps.append(ep)
        except Exception as e:
            print(f'[audit] {name} ep {i} FAILED: {e}', flush=True)
    EPISODES[name] = eps
    print(f'[audit] collected {len(eps)} episodes for {name} '
          f'(T={eps[0]["act"].shape[0] if eps else 0})', flush=True)

# ---- Analysis helpers ------------------------------------------------------
def stack_actions(eps): return np.concatenate([e['act'] for e in eps], axis=0)
def stack_rewards(eps): return np.concatenate([e['rew'] for e in eps], axis=0)
# Pull the "raw" current-step obs out of the lookback window (last row).
def stack_current_obs(eps):
    return np.concatenate([e['obs'][:, -1, :] for e in eps], axis=0)

# Identify CV/SP indices in the obs vector.
# obs = [state_dim | aug_obs (setpoint mgr) | derived (optional)]
# We'll just grab the CV columns from the state block (cv_indices map into state).
cv_state_idx = list(env.cv_indices)

def autocorr_lag1(x: np.ndarray) -> float:
    if x.size < 4: return float('nan')
    x = x - x.mean()
    den = (x ** 2).sum()
    if den <= 0: return float('nan')
    return float((x[:-1] * x[1:]).sum() / den)

def snr_db_from_signal(y: np.ndarray) -> float:
    """SNR estimate: var(low-pass(y))/var(high-freq residual)."""
    if y.size < 8: return float('nan')
    y = np.asarray(y, dtype='float64')
    # Smooth with a centered window ~= one settling time.
    w = max(3, int(round(TAU / SAMPLE_RATE)))
    k = np.ones(w) / w
    smooth = np.convolve(y, k, mode='same')
    noise = y - smooth
    vs = float(np.var(smooth))
    vn = float(np.var(noise))
    if vn <= 0 or vs <= 0: return float('nan')
    return 10.0 * math.log10(vs / vn)

# Per-domain summary stats
summary = {'plant': {'tau': TAU, 'dead': DEAD, 'sr': SAMPLE_RATE,
                     'episode_len': EPISODE_LEN, 'lookback': LOOKBACK},
           'domains': {}}

for name, eps in EPISODES.items():
    if not eps:
        summary['domains'][name] = {'error': 'no episodes'}
        continue
    acts = stack_actions(eps)   # (N*T, A)
    obs  = stack_current_obs(eps)  # (N*T, obs_dim)
    rews = stack_rewards(eps)

    A = acts.shape[1]
    # MV coverage stats per channel
    mv_stats = []
    for a in range(A):
        col = acts[:, a]
        hist, edges = np.histogram(col, bins=20, range=(-1.0, 1.0))
        cov_frac = float((hist > 0).mean())  # bin coverage
        # 5/50/95 quantiles
        q05, q50, q95 = np.quantile(col, [0.05, 0.50, 0.95])
        # autocorr lag-1 within episodes (median over episodes)
        per_ep_ac = [autocorr_lag1(e['act'][:, a]) for e in eps]
        ac1 = float(np.nanmedian(per_ep_ac))
        # fraction at saturation
        sat = float(((col <= -0.99) | (col >= 0.99)).mean())
        mv_stats.append({
            'channel': a, 'mean': float(col.mean()), 'std': float(col.std()),
            'q05': float(q05), 'q50': float(q50), 'q95': float(q95),
            'bin_coverage_frac': cov_frac, 'autocorr_lag1': ac1,
            'saturation_frac': sat,
            'hist_bin_edges': edges.tolist(),
            'hist_counts': hist.tolist(),
        })

    # CV coverage (use raw state block at cv_indices). CV is in *physical* units
    # in obs (built via _build_obs_vec).  Map to [0,1] using each channel's
    # normalization range so we get a uniform histogram across plants.
    cv_stats = []
    for j, ci in enumerate(cv_state_idx):
        if ci >= obs.shape[1]: continue
        col = obs[:, ci]
        lo, hi = env.cv_norm_ranges[j] if j < len(env.cv_norm_ranges) else (col.min(), col.max())
        rng_w = max(1e-9, hi - lo)
        norm = (col - lo) / rng_w
        hist, edges = np.histogram(norm, bins=20, range=(-0.5, 1.5))
        cov = float(((hist > 0) & (edges[:-1] >= 0) & (edges[1:] <= 1)).mean())
        # within-band fraction
        in_band = float(((norm >= 0) & (norm <= 1)).mean())
        # SNR from one representative channel of one representative episode
        ep0_col = eps[0]['obs'][:, -1, ci]
        snr = snr_db_from_signal(ep0_col)
        cv_stats.append({
            'cv_index': int(ci), 'phys_min': float(col.min()), 'phys_max': float(col.max()),
            'norm_lo': float(lo), 'norm_hi': float(hi),
            'in_band_frac': in_band,
            'bin_coverage_frac_in_band': cov,
            'snr_db_ch0': snr,
            'hist_bin_edges': edges.tolist(),
            'hist_counts': hist.tolist(),
        })

    # Reward stats
    rew_stats = {
        'mean': float(rews.mean()), 'std': float(rews.std()),
        'min': float(rews.min()), 'max': float(rews.max()),
        'p05': float(np.quantile(rews, 0.05)),
        'p50': float(np.quantile(rews, 0.50)),
        'p95': float(np.quantile(rews, 0.95)),
        'negative_frac': float((rews < 0).mean()),
    }

    summary['domains'][name] = {
        'n_episodes': len(eps),
        'total_steps': int(acts.shape[0]),
        'mv_stats': mv_stats,
        'cv_stats': cv_stats,
        'rew_stats': rew_stats,
    }

# Total cross-domain MV coverage (the most important check)
all_acts = np.concatenate([stack_actions(e) for e in EPISODES.values() if e],
                            axis=0)
combined_mv_cov = []
for a in range(env.action_dim):
    hist, _ = np.histogram(all_acts[:, a], bins=20, range=(-1.0, 1.0))
    combined_mv_cov.append({
        'channel': a, 'bin_coverage_frac': float((hist > 0).mean()),
        'counts': hist.tolist(),
        'min_bin_count': int(hist.min()),
        'p10_bin_count': int(np.quantile(hist, 0.1)),
    })
summary['combined_mv_coverage'] = combined_mv_cov

# Write JSON summary
(OUT_DIR / 'summary.json').write_text(json.dumps(summary, indent=2, default=float))
print(f'[audit] summary.json written ({len(json.dumps(summary))} bytes)', flush=True)

# ---- Plots -----------------------------------------------------------------
COLORS = {'baseline_seed': '#1f77b4', 'prbs_seed': '#ff7f0e',
          'const_action_seed': '#2ca02c', 'step_settle_seed': '#d62728',
          'random_action': '#9467bd'}

# Plot 1: MV histograms per domain x channel
fig, ax = plt.subplots(env.action_dim, 1, figsize=(8, 2.0 * env.action_dim + 1),
                        sharex=True, squeeze=False)
for a in range(env.action_dim):
    for name, eps in EPISODES.items():
        if not eps: continue
        acts = stack_actions(eps)[:, a]
        ax[a, 0].hist(acts, bins=40, range=(-1, 1), alpha=0.45,
                      label=name, color=COLORS[name], density=True)
    ax[a, 0].set_ylabel(f'MV ch{a}')
    ax[a, 0].axvline(-op_band, ls=':', c='k', alpha=0.4)
    ax[a, 0].axvline(+op_band, ls=':', c='k', alpha=0.4)
ax[0, 0].legend(loc='upper center', ncol=3, fontsize=8)
ax[-1, 0].set_xlabel('MV (normalized, ±1)')
fig.suptitle('MV action distribution per domain (dotted = const_action_seed_op_band ±0.6)')
fig.tight_layout()
fig.savefig(OUT_DIR / 'plots' / '01_mv_distribution.png', dpi=130)
plt.close(fig)

# Plot 2: CV occupancy per domain
n_cv = len(cv_state_idx)
fig, ax = plt.subplots(n_cv, 1, figsize=(8, 2.0 * n_cv + 1),
                        sharex=True, squeeze=False)
for j, ci in enumerate(cv_state_idx):
    lo, hi = env.cv_norm_ranges[j] if j < len(env.cv_norm_ranges) else (0, 100)
    for name, eps in EPISODES.items():
        if not eps: continue
        obs = stack_current_obs(eps)
        col = obs[:, ci]
        norm = (col - lo) / max(1e-9, hi - lo)
        ax[j, 0].hist(norm, bins=40, range=(-0.2, 1.2), alpha=0.45,
                      label=name, color=COLORS[name], density=True)
    ax[j, 0].set_ylabel(f'CV ch{j} (idx={ci})')
    ax[j, 0].axvline(0, ls=':', c='k', alpha=0.4)
    ax[j, 0].axvline(1, ls=':', c='k', alpha=0.4)
ax[0, 0].legend(loc='upper center', ncol=3, fontsize=8)
ax[-1, 0].set_xlabel('CV (normalized to its declared bounds; 0=lo, 1=hi)')
fig.suptitle('CV state distribution per domain')
fig.tight_layout()
fig.savefig(OUT_DIR / 'plots' / '02_cv_distribution.png', dpi=130)
plt.close(fig)

# Plot 3: 2D MV-CV occupancy heatmap (channel 0 only)
if n_cv >= 1:
    ci = cv_state_idx[0]
    lo, hi = env.cv_norm_ranges[0]
    fig, axes = plt.subplots(1, len(EPISODES), figsize=(3.5 * len(EPISODES), 3.5),
                              sharex=True, sharey=True, squeeze=False)
    for k, (name, eps) in enumerate(EPISODES.items()):
        ax = axes[0, k]
        if not eps:
            ax.set_title(f'{name}\n(empty)'); continue
        acts = stack_actions(eps)[:, 0]
        obs  = stack_current_obs(eps)[:, ci]
        cv_n = (obs - lo) / max(1e-9, hi - lo)
        H, xe, ye = np.histogram2d(acts, cv_n, bins=20,
                                    range=[[-1, 1], [-0.2, 1.2]])
        ax.imshow(np.log1p(H.T), origin='lower', aspect='auto',
                  extent=[-1, 1, -0.2, 1.2], cmap='viridis')
        ax.set_title(name, fontsize=9)
        ax.set_xlabel('MV')
        if k == 0: ax.set_ylabel('CV (normalized)')
    fig.suptitle('2D MV × CV occupancy (log scale)')
    fig.tight_layout()
    fig.savefig(OUT_DIR / 'plots' / '03_mv_cv_occupancy.png', dpi=130)
    plt.close(fig)

# Plot 4: timeseries — 3 episodes from each domain side-by-side
for name, eps in EPISODES.items():
    if not eps: continue
    n_show = min(3, len(eps))
    fig, axes = plt.subplots(n_show, 2, figsize=(11, 2.0 * n_show + 1),
                              sharex='col', squeeze=False)
    for i in range(n_show):
        a_ts = eps[i]['act'][:, 0]
        # CV in physical units for channel 0
        cv_ts = eps[i]['obs'][:, -1, cv_state_idx[0]]
        axes[i, 0].plot(a_ts, lw=0.8, color=COLORS[name])
        axes[i, 0].set_ylabel(f'ep{i}\nMV')
        axes[i, 0].set_ylim(-1.05, 1.05)
        axes[i, 0].axhline(0, c='k', lw=0.3, alpha=0.3)
        axes[i, 1].plot(cv_ts, lw=0.8, color=COLORS[name])
        axes[i, 1].set_ylabel('CV (phys)')
    axes[-1, 0].set_xlabel('agent step')
    axes[-1, 1].set_xlabel('agent step')
    fig.suptitle(f'{name}: 3 representative episodes (MV left, CV right)')
    fig.tight_layout()
    fig.savefig(OUT_DIR / 'plots' / f'04_ts_{name}.png', dpi=130)
    plt.close(fig)

# Plot 5: Combined MV coverage across all domains
fig, ax = plt.subplots(env.action_dim, 1, figsize=(8, 2.0 * env.action_dim + 1),
                        squeeze=False)
for a in range(env.action_dim):
    counts, edges = np.histogram(all_acts[:, a], bins=20, range=(-1, 1))
    ctrs = 0.5 * (edges[:-1] + edges[1:])
    ax[a, 0].bar(ctrs, counts, width=(edges[1]-edges[0])*0.95,
                  color='#444', alpha=0.85)
    ax[a, 0].set_ylabel(f'MV ch{a}\ncount')
    ax[a, 0].axvline(-op_band, ls=':', c='r', alpha=0.6)
    ax[a, 0].axvline(+op_band, ls=':', c='r', alpha=0.6)
    ax[a, 0].set_yscale('log')
ax[-1, 0].set_xlabel('MV (normalized)')
fig.suptitle('COMBINED MV coverage across all domains (log y; red = op_band)')
fig.tight_layout()
fig.savefig(OUT_DIR / 'plots' / '05_combined_mv_coverage.png', dpi=130)
plt.close(fig)

# Plot 6: Reward distribution per domain
fig, ax = plt.subplots(1, 1, figsize=(9, 4))
for name, eps in EPISODES.items():
    if not eps: continue
    r = stack_rewards(eps)
    ax.hist(r, bins=80, alpha=0.5, label=name, color=COLORS[name],
            density=True)
ax.legend(fontsize=8)
ax.set_xlabel('raw reward')
ax.set_yscale('log')
ax.set_title('Reward distribution per domain (log y)')
fig.tight_layout()
fig.savefig(OUT_DIR / 'plots' / '06_reward_distribution.png', dpi=130)
plt.close(fig)

print(f'[audit] plots written to {OUT_DIR/"plots"}', flush=True)

# ---- Markdown report -------------------------------------------------------
md = []
md.append('# Data-Generation Audit\n')
md.append(f'- Plant: τ={TAU} dead={DEAD} sr={SAMPLE_RATE} ep_len={EPISODE_LEN}\n')
md.append(f'- Source run (for noise/dynamics): `{SOURCE_RUN.name}`\n')
md.append(f'- N per domain: {N_PER_DOMAIN}\n\n')

md.append('## MV coverage per domain (channel 0)\n\n')
md.append('| Domain | mean | std | q05 | q50 | q95 | bin-cov | ac(1) | sat |\n')
md.append('|---|---:|---:|---:|---:|---:|---:|---:|---:|\n')
for name, info in summary['domains'].items():
    if 'mv_stats' not in info: continue
    s = info['mv_stats'][0]
    md.append(f'| {name} | {s["mean"]:+.3f} | {s["std"]:.3f} | '
              f'{s["q05"]:+.3f} | {s["q50"]:+.3f} | {s["q95"]:+.3f} | '
              f'{s["bin_coverage_frac"]:.2f} | {s["autocorr_lag1"]:+.3f} | '
              f'{s["saturation_frac"]:.3f} |\n')

md.append('\n## CV coverage per domain (channel 0, normalized to declared bounds)\n\n')
md.append('| Domain | in-band frac | bin-cov in band | SNR ch0 (dB) |\n')
md.append('|---|---:|---:|---:|\n')
for name, info in summary['domains'].items():
    if 'cv_stats' not in info or not info['cv_stats']: continue
    s = info['cv_stats'][0]
    md.append(f'| {name} | {s["in_band_frac"]:.3f} | '
              f'{s["bin_coverage_frac_in_band"]:.3f} | '
              f'{s["snr_db_ch0"]:.1f} |\n')

md.append('\n## Reward stats per domain\n\n')
md.append('| Domain | mean | std | p05 | p50 | p95 | <0 frac |\n')
md.append('|---|---:|---:|---:|---:|---:|---:|\n')
for name, info in summary['domains'].items():
    if 'rew_stats' not in info: continue
    r = info['rew_stats']
    md.append(f'| {name} | {r["mean"]:.2f} | {r["std"]:.2f} | '
              f'{r["p05"]:.2f} | {r["p50"]:.2f} | {r["p95"]:.2f} | '
              f'{r["negative_frac"]:.3f} |\n')

md.append('\n## Combined MV coverage (ALL domains, channel 0)\n\n')
c = summary['combined_mv_coverage'][0]
md.append(f'- bin coverage: {c["bin_coverage_frac"]:.2f} (of 20 bins in [-1,+1])\n')
md.append(f'- min bin count: {c["min_bin_count"]}\n')
md.append(f'- p10 bin count: {c["p10_bin_count"]}\n')
md.append(f'- counts: `{c["counts"]}`\n')

(OUT_DIR / 'report.md').write_text(''.join(md))
print(f'[audit] report.md written. DONE -> {OUT_DIR}', flush=True)
