"""Validation for trained DreamerV4 controllers.

Design — slim by intent:

  • Reuses the training APCEnv (same disturbance schedule, same objective, same
    setpoint manager) so eval distribution matches training distribution; the
    only difference is a held-out RNG seed and deterministic actor.  This
    eliminates ~2000 lines of channel cataloguing / holdout-profile code from
    the legacy validator (`neural-apc-pytorch/evaluation/validate_latent.py`).

  • Loads ``final.pt`` (or any ``--ckpt``) and runs ``--episodes`` per seed
    over ``--seeds`` seeds (default 3).

  • Records per-step CSV: state, MV, CV, reward (raw + scaled), reward
    components, action bin index.

  • Plots:  CV trajectories with bound bands and disturbance markers, MV
            trajectories with bound bands, per-step reward, cumulative reward.

Usage::

    python -m evaluation.validate \\
        --controller-dir _runs/test_sim_20260429_143935 \\
        --simulation-dir simulation/test_sim          # optional, auto-read

Outputs land in ``<controller-dir>/validation/`` so they sit next to
``train_log.jsonl`` and ``run_plan.json``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

import matplotlib
matplotlib.use('Agg')  # type: ignore
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_run_plan(controller_dir: Path) -> Dict:
    """Look up run_plan.json: in the controller dir, then walk parents.

    The BO workflow writes run_plan.json at the workflow root, while
    controller_dir is typically a per-trial or final/ subfolder.
    """
    for d in [controller_dir, *controller_dir.parents]:
        plan_path = d / 'run_plan.json'
        if plan_path.exists():
            with open(plan_path, 'r') as f:
                return json.load(f)
        # stop at repo root or filesystem root
        if d.name in ('output', '') or d == d.parent:
            break
    return {}


def _resolve_sim_dir(arg: str | None, controller_dir: Path,
                      run_plan: Dict) -> Path:
    repo = Path(__file__).resolve().parent.parent
    if arg:
        p = Path(arg)
        if p.is_absolute() and p.exists():
            return p
        cand = repo / arg
        if cand.exists():
            return cand
        cand2 = repo / 'simulation' / arg
        if cand2.exists():
            return cand2
        raise FileNotFoundError(f'Cannot resolve --simulation-dir: {arg}')
    sim_dir = run_plan.get('simulation_dir')
    if sim_dir and Path(sim_dir).exists():
        return Path(sim_dir)
    raise FileNotFoundError(
        f'Cannot infer simulation_dir from {controller_dir}/run_plan.json — '
        f'pass --simulation-dir explicitly.')


def _episode_disturbance_markers(schedule: List[Dict], sample_rate: int = 1
                                  ) -> List[Dict]:
    """Flatten schedule events into ``(start_step, label)`` markers."""
    out = []
    for ev in (schedule or []):
        try:
            start = int(ev.get('start', 0))
            name = ev.get('name') or ev.get('group') or 'event'
            out.append({'start': start, 'label': str(name)})
        except Exception:
            continue
    return out


def _bin_index_from_action(action_t: torch.Tensor, n_action_bins: int) -> int:
    """Recover the discrete bin index from a continuous action ∈ [-1, 1]."""
    a = float(np.clip(np.asarray(action_t.detach().cpu()).ravel()[0], -1.0, 1.0))
    bin_centres = np.linspace(-1.0, 1.0, n_action_bins)
    return int(np.argmin(np.abs(bin_centres - a)))


# ---------------------------------------------------------------------------
# Scripted disturbance schedule (deterministic, for rejection plots)
# ---------------------------------------------------------------------------

def build_scripted_disturbance_schedule(env, *, n_events: int = 3,
                                         magnitude_frac: float = 0.10
                                         ) -> List[Dict]:
    """Build a small, deterministic step-disturbance schedule.

    Mirrors the ``neural-apc-pytorch`` validator's disturbance-rejection
    test: a handful of equally-spaced step changes on the first available
    DV (or the first CV when no DV exists).  Each step is held for the
    rest of the episode (``shape='step'``) so the controller's settling
    behaviour is visible.  Magnitudes are a fraction of channel span
    with alternating sign to exercise both directions.

    The returned list slots directly into ``env._schedule`` (overwrite
    after ``env.reset()`` and the env's ``_apply_disturbance`` will pick
    them up).
    """
    T = int(env.cfg.episode_length)
    # Reserve a settle window equal to ~one tau before the first step and
    # at the end so the response is fully visible.
    first = max(int(0.20 * T), 1)
    last = max(first + 1, int(0.85 * T))
    starts = np.linspace(first, last, n_events).round().astype(int).tolist()

    # Pick a target channel: prefer DV-classified state index (states
    # outside cv_indices), fall back to the first CV.
    sim = env.sim
    meta = env.meta
    state_dim = int(meta.get('state_dim', env.state_dim))
    cv_idx = set(env.cv_indices)
    # Heuristic: any state index not a CV is treated as DV-like.
    dv_candidates = [i for i in range(state_dim) if i not in cv_idx]
    if dv_candidates:
        target_pos = int(dv_candidates[0])
        target_group = 'dv'
    else:
        target_pos = int(env.cv_indices[0]) if env.cv_indices else 0
        target_group = 'cv'

    # Magnitude: fraction of normalisation span (or fall back to 1.0).
    norm_ranges = (env.mv_norm_ranges if target_group == 'mv'
                    else env.cv_norm_ranges)
    span = 1.0
    if target_group == 'cv' and env.cv_norm_ranges:
        lo, hi = env.cv_norm_ranges[0]
        span = max(1e-6, float(hi - lo))
    elif target_group == 'dv' and env.cv_norm_ranges:
        # No DV norm ranges in metadata; use CV span as a proxy.
        lo, hi = env.cv_norm_ranges[0]
        span = max(1e-6, float(hi - lo))

    schedule: List[Dict] = []
    for i, s in enumerate(starts):
        sign = 1.0 if (i % 2 == 0) else -1.0
        delta = float(sign * magnitude_frac * span)
        schedule.append({
            'name': f'scripted_step_{i + 1}',
            'target_group': target_group,
            'target_pos': target_pos,
            'start': int(s),
            'duration': int(T - int(s)),
            'shape': 'step',
            'period': float(max(2.0, T)),
            'source': 'scripted',
            'delta': delta,
            '_applied': False,
            '_is_violation': False,
        })
    return schedule


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_episode(env, model, device, *, deterministic: bool, seed_offset: int = 0
                ) -> Dict:
    """Run one full episode under the trained actor (deterministic by default).

    Returns a dict of arrays + metadata for plotting and metrics.
    """
    obs_window = env.reset(exploration=False)
    schedule = list(env._schedule)
    return _run_episode_with_window(env, model, device, obs_window, schedule,
                                     deterministic=deterministic)


def run_scripted_episode(env, model, device, *, deterministic: bool,
                          schedule: List[Dict]) -> Dict:
    """Run an episode where the disturbance schedule is replaced by
    ``schedule`` (typically the deterministic one from
    ``build_scripted_disturbance_schedule``).  Used for the
    disturbance-rejection plot.
    """
    obs_window = env.reset(exploration=False)
    # Override the schedule the env built in reset() with our scripted one.
    env._schedule = list(schedule)
    return _run_episode_with_window(env, model, device, obs_window,
                                     env._schedule,
                                     deterministic=deterministic)


def _run_episode_with_window(env, model, device, obs_window, schedule, *,
                              deterministic: bool) -> Dict:
    T = env.cfg.episode_length
    state_dim = env.state_dim
    action_dim = env.action_dim

    states = np.zeros((T, state_dim), dtype='float32')
    actions_norm = np.zeros((T, action_dim), dtype='float32')
    controls = np.zeros((T, action_dim), dtype='float32')
    raw_rewards = np.zeros(T, dtype='float32')
    scaled_rewards = np.zeros(T, dtype='float32')
    cv_violations = np.zeros(T, dtype='float32')
    mv_violations = np.zeros(T, dtype='float32')

    # V4 streaming inference: maintain a rolling action history alongside
    # the env-provided observation window. At each step we encode the
    # window through the tokenizer, run the dynamics transformer with
    # context-noise corruption (τ = 1 − τ_ctx), and read the agent-register
    # hidden state at the latest position to feed the policy head.
    cfg = env.cfg
    L = cfg.lookback
    a_history = np.zeros((L, action_dim), dtype='float32')
    d_min = 1.0 / cfg.k_max
    tau_ctx_val = 1.0 - cfg.tau_ctx

    for t in range(T):
        ow = torch.from_numpy(obs_window).to(device)
        a_ctx = torch.from_numpy(a_history).to(device)
        with torch.no_grad():
            with torch.amp.autocast(device_type=device.type,
                                     dtype=torch.bfloat16,
                                     enabled=(device.type == 'cuda')):
                z_ctx = model.tokenizer.encode(ow).unsqueeze(0)
                tau = torch.full((1, L), tau_ctx_val, device=device,
                                  dtype=z_ctx.dtype)
                d = torch.full((1, L), d_min, device=device,
                                dtype=z_ctx.dtype)
                out = model.dynamics(z_ctx, tau, d, a_ctx.unsqueeze(0))
                agent_hid = out['agent_hid'][:, -1]
                action_t, _, _ = model.policy(agent_hid,
                                                deterministic=deterministic)
        a_np = action_t.float().squeeze(0).cpu().numpy().astype('float32')
        next_window, scaled_r, done, info = env.step(a_np)
        comps = info.get('reward_components', {}) or {}
        states[t] = next_window[-1, :state_dim]
        actions_norm[t] = a_np
        controls[t] = np.asarray(env._prev_control, dtype='float32')
        raw_rewards[t] = float(info.get('raw_reward', 0.0))
        scaled_rewards[t] = float(scaled_r)
        cv_violations[t] = float(comps.get('cv_violation_penalty', 0.0))
        mv_violations[t] = float(comps.get('mv_violation_penalty', 0.0))
        a_history = np.concatenate([a_history[1:], a_np[None, :]], axis=0)
        obs_window = next_window
        if done:
            break

    return {
        'states': states[:t + 1],
        'actions_norm': actions_norm[:t + 1],
        'controls': controls[:t + 1],
        'raw_rewards': raw_rewards[:t + 1],
        'scaled_rewards': scaled_rewards[:t + 1],
        'cv_violations': cv_violations[:t + 1],
        'mv_violations': mv_violations[:t + 1],
        'cum_reward': float(np.cumsum(scaled_rewards[:t + 1])[-1]),
        'cum_raw_reward': float(np.cumsum(raw_rewards[:t + 1])[-1]),
        'mean_cv_violation': float(cv_violations[:t + 1].mean()),
        'mean_mv_violation': float(mv_violations[:t + 1].mean()),
        'schedule': schedule,
        'episode_length': int(t + 1),
        'sample_rate': int(env.cfg.sample_rate),
        'cv_indices': list(env.cv_indices),
        'mv_indices': [int(x) for x in (env.meta.get('mv_indices') or [])],
        'dv_indices': [int(x) for x in (env.meta.get('dv_indices') or [])],
        'state_variables': list(env.meta.get('state_variables') or []),
        'mv_norm_ranges': [list(b) for b in env.mv_norm_ranges],
        'cv_norm_ranges': [list(b) for b in env.cv_norm_ranges],
        'mv_bounds': [list(b) for b in
                       getattr(env.setpoint_mgr, 'base_mv_bounds', np.zeros((0, 2)))],
        'cv_bounds': [list(b) for b in
                       getattr(env.setpoint_mgr, 'base_cv_bounds', np.zeros((0, 2)))],
        'cv_targets': [float(x) for x in
                        getattr(env.setpoint_mgr, 'base_cv_targets', [])],
        'reward_scale': float(env.reward_scale),
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _add_disturbance_markers(ax, schedule: List[Dict], color='red', alpha=0.20):
    for ev in (schedule or []):
        try:
            x = float(ev.get('start', 0))
        except Exception:
            continue
        ax.axvline(x, color=color, linestyle='--', linewidth=0.7, alpha=alpha)


def plot_episode(ep: Dict, out_path: Path, title: str = '') -> None:
    states = ep['states']
    controls = ep['controls']
    cv_idx = ep['cv_indices']
    cv_norm = ep['cv_norm_ranges']
    mv_norm = ep['mv_norm_ranges']
    schedule = ep['schedule']
    T = ep['episode_length']
    t_arr = np.arange(T)

    n_cv = len(cv_idx)
    n_mv = controls.shape[1]
    n_rows = max(1, n_cv) + max(1, n_mv) + 2  # +rewards +cum
    fig, axes = plt.subplots(n_rows, 1, figsize=(12, 2.0 * n_rows), sharex=True)
    if n_rows == 1:
        axes = [axes]

    row = 0
    # CVs with bound bands
    for j, cidx in enumerate(cv_idx):
        ax = axes[row]; row += 1
        if cidx < states.shape[1]:
            ax.plot(t_arr, states[:, cidx], color='C0', lw=1.0, label=f'CV[{cidx}]')
        if j < len(cv_norm):
            lo, hi = cv_norm[j]
            ax.axhline(lo, color='gray', lw=0.6, ls=':')
            ax.axhline(hi, color='gray', lw=0.6, ls=':')
            ax.fill_between(t_arr, lo, hi, color='gray', alpha=0.05)
        _add_disturbance_markers(ax, schedule)
        ax.set_ylabel(f'CV[{cidx}]')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)

    # MVs with bound bands
    for j in range(n_mv):
        ax = axes[row]; row += 1
        ax.plot(t_arr, controls[:, j], color='C1', lw=1.0, label=f'MV[{j}]')
        if j < len(mv_norm):
            lo, hi = mv_norm[j]
            ax.axhline(lo, color='gray', lw=0.6, ls=':')
            ax.axhline(hi, color='gray', lw=0.6, ls=':')
            ax.fill_between(t_arr, lo, hi, color='gray', alpha=0.05)
        _add_disturbance_markers(ax, schedule)
        ax.set_ylabel(f'MV[{j}]')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)

    # Per-step reward
    ax = axes[row]; row += 1
    ax.plot(t_arr, ep['raw_rewards'], color='C2', lw=0.9, label='raw reward')
    ax.plot(t_arr, ep['scaled_rewards'], color='C3', lw=0.9, alpha=0.6,
            label=f"scaled (×{ep['reward_scale']:.2f})")
    ax.axhline(0, color='gray', lw=0.5, ls='-', alpha=0.5)
    _add_disturbance_markers(ax, schedule)
    ax.set_ylabel('reward')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)

    # Cumulative reward
    ax = axes[row]; row += 1
    ax.plot(t_arr, np.cumsum(ep['raw_rewards']), color='C2', lw=1.0,
            label=f"raw cum (final={ep['cum_raw_reward']:+.1f})")
    ax.plot(t_arr, np.cumsum(ep['scaled_rewards']), color='C3', lw=1.0,
            alpha=0.6,
            label=f"scaled cum (final={ep['cum_reward']:+.1f})")
    _add_disturbance_markers(ax, schedule)
    ax.set_ylabel('cum reward')
    ax.set_xlabel('step')
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def plot_disturbance_rejection(ep: Dict, out_path: Path, title: str = '') -> None:
    """Plant-style disturbance-rejection plot.

    Mirrors ``plot_channels`` from ``neural-apc-pytorch/evaluation/
    validate_latent.py`` so the operator sees a real plant view:
      - one row per channel (MV, DV, CV) with semantic state names,
      - CV bound band shaded orange + red dashed low/high lines,
      - MV red dashed low/high lines + gray dotted normalization range,
      - dark-green dash-dot CV target setpoint,
      - vertical disturbance markers with ▲/▼ direction arrow at top,
      - cum-reward subplot at the bottom.
    """
    states = ep['states']
    controls = ep['controls']
    cv_idx = ep['cv_indices']
    mv_idx = ep.get('mv_indices') or []
    dv_idx = ep.get('dv_indices') or []
    state_names = ep.get('state_variables') or []
    cv_norm = ep['cv_norm_ranges']
    mv_norm = ep['mv_norm_ranges']
    cv_bounds = ep.get('cv_bounds') or []
    mv_bounds = ep.get('mv_bounds') or []
    cv_targets = ep.get('cv_targets') or []
    schedule = ep['schedule']
    T = ep['episode_length']
    t_arr = np.arange(T)

    def _name(i: int, default: str) -> str:
        return state_names[i] if 0 <= i < len(state_names) and state_names[i] else default

    # Build channel rows: MVs first (operator manipulated), then DVs
    # (uncontrolled drivers — what the disturbance actually injects), then
    # CVs (controlled outputs).  Same order/grouping as the legacy plot.
    channels: List[Dict] = []
    for k, i in enumerate(mv_idx):
        bounds = mv_bounds[k] if k < len(mv_bounds) else None
        norm = mv_norm[k] if k < len(mv_norm) else None
        channels.append({'group': 'mv', 'series': controls[:, k] if k < controls.shape[1] else None,
                         'label': _name(i, f'MV[{i}]'), 'bounds': bounds,
                         'norm': norm, 'target': None, 'color': '#1f77b4'})
    for i in dv_idx:
        if i >= states.shape[1]:
            continue
        channels.append({'group': 'dv', 'series': states[:, i],
                         'label': _name(i, f'DV[{i}]'), 'bounds': None,
                         'norm': None, 'target': None, 'color': '#9467bd'})
    for k, i in enumerate(cv_idx):
        if i >= states.shape[1]:
            continue
        bounds = cv_bounds[k] if k < len(cv_bounds) else None
        norm = cv_norm[k] if k < len(cv_norm) else None
        target = cv_targets[k] if k < len(cv_targets) else None
        channels.append({'group': 'cv', 'series': states[:, i],
                         'label': _name(i, f'CV[{i}]'), 'bounds': bounds,
                         'norm': norm, 'target': target, 'color': '#2ca02c'})

    n_rows = max(1, len(channels)) + 1  # +reward
    fig, axes = plt.subplots(n_rows, 1,
                              figsize=(13, max(4.0, 2.0 * n_rows)),
                              sharex=True)
    if n_rows == 1:
        axes = [axes]

    # Per-event annotations (CV settle/overshoot) for the summary record.
    annotations: List[Dict] = []
    for ev in schedule:
        st = int(ev.get('start', 0))
        if st >= T - 5:
            continue
        for j, cidx in enumerate(cv_idx):
            if cidx >= states.shape[1]:
                continue
            pre = states[max(0, st - 20):st, cidx]
            post = states[st:min(T, st + 200), cidx]
            if pre.size == 0 or post.size == 0:
                continue
            base = float(np.mean(pre))
            dev = post - base
            ovr = float(dev[np.argmax(np.abs(dev))]) if dev.size else 0.0
            band = max(1e-6, 0.05 * (np.max(np.abs(pre)) if pre.size else 1.0))
            settled = np.where(np.abs(dev) <= band)[0]
            settle_t = int(settled[0]) if settled.size else int(post.size)
            annotations.append({'cv_row': j, 'start': st,
                                 'overshoot': ovr, 'settle_steps': settle_t,
                                 'name': ev.get('name', 'step')})

    def _draw_disturbance_markers(ax) -> None:
        ylo, yhi = ax.get_ylim()
        y_top = yhi - 0.04 * (yhi - ylo)
        for ev in schedule:
            st = int(ev.get('start', 0))
            color = ev.get('color') or (
                '#ff7f0e' if 'violation' in str(ev.get('intent', '')).lower()
                else '#17a2b8')
            ax.axvline(st, color=color, alpha=0.50, linewidth=1.2,
                        linestyle='--')
            delta = float(ev.get('delta', 0.0))
            label = '\u25B2' if delta > 0 else '\u25BC'
            ax.text(st, y_top, label, color=color, fontsize=8, ha='center',
                     va='top', fontweight='bold', clip_on=True)

    for r, ch in enumerate(channels):
        ax = axes[r]
        series = ch['series']
        if series is None or len(series) == 0:
            ax.set_ylabel(ch['label'])
            continue
        ax.plot(t_arr[:len(series)], series, color=ch['color'], lw=1.2,
                label=ch['label'])

        bounds = ch.get('bounds')
        if bounds is not None and len(bounds) >= 2 and \
           np.isfinite(bounds[0]) and np.isfinite(bounds[1]) and \
           bounds[1] > bounds[0] and abs(bounds[0]) < 1e9 and abs(bounds[1]) < 1e9:
            lo_b, hi_b = float(bounds[0]), float(bounds[1])
            if ch['group'] == 'cv':
                ax.axhspan(lo_b, hi_b, color='#ffcc80', alpha=0.18,
                            label='CV bound band')
                ax.axhline(lo_b, color='#d32f2f', linestyle='--', linewidth=1.4,
                            label='CV low bound')
                ax.axhline(hi_b, color='#d32f2f', linestyle='--', linewidth=1.4,
                            label='CV high bound')
            else:
                ax.axhline(lo_b, color='r', linestyle='--', linewidth=1.0,
                            label='Low bound')
                ax.axhline(hi_b, color='r', linestyle='--', linewidth=1.0,
                            label='High bound')

        norm = ch.get('norm')
        if norm is not None and len(norm) >= 2 and \
           np.isfinite(norm[0]) and np.isfinite(norm[1]):
            ax.axhline(float(norm[0]), color='#6c757d', linestyle=':',
                        linewidth=1.0, label='Norm low')
            ax.axhline(float(norm[1]), color='#6c757d', linestyle=':',
                        linewidth=1.0, label='Norm high')

        target = ch.get('target')
        if target is not None and np.isfinite(target):
            ax.axhline(float(target), color='#1b5e20', linestyle='-.',
                        linewidth=1.4, label=f'Target ({float(target):g})')

        # Pad y-limits to include bounds + norm so steps stay visible.
        finite = series[np.isfinite(series)] if isinstance(series, np.ndarray) else None
        lo = float(np.min(finite)) if finite is not None and finite.size else None
        hi = float(np.max(finite)) if finite is not None and finite.size else None
        for ref in (bounds, norm):
            if ref is not None and len(ref) >= 2 and np.isfinite(ref[0]) and np.isfinite(ref[1]):
                rlo, rhi = float(ref[0]), float(ref[1])
                if abs(rlo) < 1e9 and abs(rhi) < 1e9:
                    lo = rlo if lo is None else min(lo, rlo)
                    hi = rhi if hi is None else max(hi, rhi)
        if lo is not None and hi is not None and hi > lo:
            pad = 0.06 * (hi - lo)
            ax.set_ylim(lo - pad, hi + pad)

        _draw_disturbance_markers(ax)
        ax.set_ylabel(ch['label'])
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)

    ax = axes[-1]
    ax.plot(t_arr, np.cumsum(ep['raw_rewards']), color='C2', lw=1.0,
            label=f"raw cum (final={ep['cum_raw_reward']:+.1f})")
    _draw_disturbance_markers(ax)
    ax.set_ylabel('cum reward')
    ax.set_xlabel('time step')
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=11, y=0.995)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    fig.savefig(out_path, dpi=130)
    plt.close(fig)

    return annotations  # caller stashes into summary metrics


def plot_summary(seed_results: List[List[Dict]], out_path: Path,
                  title: str = '') -> None:
    """Cross-seed summary: cum-reward distribution + violation rates."""
    cum = np.array([[ep['cum_raw_reward'] for ep in seed_eps]
                    for seed_eps in seed_results], dtype='float64')
    cv_v = np.array([[ep['mean_cv_violation'] for ep in seed_eps]
                     for seed_eps in seed_results], dtype='float64')
    mv_v = np.array([[ep['mean_mv_violation'] for ep in seed_eps]
                     for seed_eps in seed_results], dtype='float64')

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].boxplot(cum.T, tick_labels=[f's{i}' for i in range(cum.shape[0])])
    axes[0].set_title(f'Cum raw reward per episode\n'
                      f'overall mean={cum.mean():+.2f} ± {cum.std():.2f}')
    axes[0].set_ylabel('cum reward')
    axes[0].grid(True, alpha=0.3)

    axes[1].boxplot(cv_v.T, tick_labels=[f's{i}' for i in range(cv_v.shape[0])])
    axes[1].set_title(f'Mean CV violation\nmean={cv_v.mean():.4f}')
    axes[1].set_ylabel('cv penalty')
    axes[1].grid(True, alpha=0.3)

    axes[2].boxplot(mv_v.T, tick_labels=[f's{i}' for i in range(mv_v.shape[0])])
    axes[2].set_title(f'Mean MV violation\nmean={mv_v.mean():.4f}')
    axes[2].set_ylabel('mv penalty')
    axes[2].grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_validation(*,
                   controller_dir: Path | str,
                   simulation_dir: Path | str | None = None,
                   ckpt: str = 'final.pt',
                   episodes: int = 3, seeds: int = 3,
                   out: Path | str | None = None,
                   deterministic: bool = True) -> Dict:
    """Validate ``controller_dir/<ckpt>`` and write plots + summary.json.

    This is the programmatic entry point used by the workflow runner; the
    CLI ``main()`` simply parses argv and calls this.
    """
    controller_dir = Path(controller_dir).resolve()
    if not controller_dir.exists():
        raise FileNotFoundError(controller_dir)
    ckpt_path = controller_dir / ckpt
    if not ckpt_path.exists():
        raise FileNotFoundError(ckpt_path)

    out_dir = Path(out).resolve() if out else controller_dir / 'validation'
    out_dir.mkdir(parents=True, exist_ok=True)

    repo = Path(__file__).resolve().parent.parent
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))

    run_plan = _load_run_plan(controller_dir)
    sim_dir = _resolve_sim_dir(str(simulation_dir) if simulation_dir else None,
                                controller_dir, run_plan)

    os.environ['CONTROL_SETUP_JSON'] = str(sim_dir / 'control_setup.json')
    os.environ['CONTROL_OBJECTIVE_JSON'] = str(sim_dir / 'control_objective.json')
    os.environ['SIMULATION_DIR'] = str(sim_dir)
    if 'sample_rate' in run_plan:
        os.environ['SIM_SAMPLE_RATE'] = str(run_plan['sample_rate'])
    if 'episode_length' in run_plan:
        os.environ['SIM_EPISODE_LENGTH'] = str(run_plan['episode_length'])
    if 'tau' in run_plan:
        os.environ['IDENTIFIED_TAU_DOMINANT'] = f"{run_plan['tau']:g}"
    if 'dead_time' in run_plan:
        os.environ['IDENTIFIED_DEAD_TIME'] = f"{run_plan['dead_time']:g}"

    from training.train import TrainConfig, APCEnv
    from models.dreamer_v4 import DreamerV4, DreamerV4Config

    print(f'[val] controller: {controller_dir}', flush=True)
    print(f'[val] simulation: {sim_dir}', flush=True)
    print(f'[val] ckpt: {ckpt_path}  deterministic={deterministic}', flush=True)

    ckpt_obj = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    cfg_dict = ckpt_obj.get('cfg') or {}
    valid_keys = set(TrainConfig.__dataclass_fields__.keys())
    cfg = TrainConfig(**{k: v for k, v in cfg_dict.items() if k in valid_keys})

    model_cfg = DreamerV4Config(
        obs_dim=cfg.obs_dim, action_dim=cfg.action_dim, lookback=cfg.lookback,
        tok_hidden=cfg.tok_hidden, z_dim=cfg.z_dim, mae_p_max=cfg.mae_p_max,
        d_model=cfg.d_model, n_layers=cfg.n_layers, n_heads=cfg.n_heads,
        ff_mult=cfg.ff_mult, n_register=cfg.n_register,
        k_max=cfg.k_max, tau_n_bins=cfg.tau_n_bins, soft_cap=cfg.soft_cap,
        n_action_bins=cfg.n_action_bins,
        head_hidden=cfg.head_hidden, head_n_layers=cfg.head_n_layers,
        mtp_length=max(1, int(getattr(cfg, 'mtp_length', 1))),
        attn_impl=getattr(cfg, 'attn_impl', 'auto'),
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DreamerV4(model_cfg).to(device)
    # Checkpoints saved while ``torch.compile`` was active have keys
    # prefixed with ``_orig_mod.`` (e.g. ``tokenizer._orig_mod.encoder...``)
    # because ``torch.compile`` wraps the module in ``OptimizedModule``.
    # Strip the prefix so the bare DreamerV4 can load.
    sd = ckpt_obj['model']
    if any('._orig_mod.' in k for k in sd):
        sd = {k.replace('._orig_mod.', '.'): v for k, v in sd.items()}
    model.load_state_dict(sd)
    model.eval()

    seed_results: List[List[Dict]] = []
    metrics_records: List[Dict] = []
    disturbance_records: List[Dict] = []
    for s in range(int(seeds)):
        seed = 10_000 + s  # held-out from training (which used SEED=0..N).
        rng = np.random.default_rng(seed)
        env = APCEnv(cfg, rng)
        # Use the calibrated reward scale from training when available.
        cal_path = controller_dir / 'reward_calibration.json'
        if cal_path.exists():
            try:
                with open(cal_path, 'r') as f:
                    env.reward_scale = float(json.load(f).get('reward_scale', 1.0))
            except Exception:
                env.reward_scale = 1.0

        per_seed_dir = out_dir / f'seed_{seed:05d}'
        per_seed_dir.mkdir(parents=True, exist_ok=True)

        eps = []
        for e in range(int(episodes)):
            ep = run_episode(env, model, device, deterministic=deterministic)
            title = (f'seed={seed} ep={e}  T={ep["episode_length"]}  '
                     f'cum_raw={ep["cum_raw_reward"]:+.2f}  '
                     f'mean_cv_v={ep["mean_cv_violation"]:.4f}  '
                     f'mean_mv_v={ep["mean_mv_violation"]:.4f}')
            plot_episode(ep, per_seed_dir / f'ep_{e:02d}.png', title=title)
            eps.append(ep)
            metrics_records.append({
                'seed': seed, 'episode': e,
                'cum_raw_reward': ep['cum_raw_reward'],
                'cum_scaled_reward': ep['cum_reward'],
                'mean_cv_violation': ep['mean_cv_violation'],
                'mean_mv_violation': ep['mean_mv_violation'],
                'episode_length': ep['episode_length'],
                'n_disturbance_events': len(ep['schedule']),
            })

        # ---- Disturbance-rejection plot (scripted, deterministic schedule) ----
        try:
            scripted = build_scripted_disturbance_schedule(env, n_events=3,
                                                            magnitude_frac=0.10)
            ep_d = run_scripted_episode(env, model, device,
                                         deterministic=deterministic,
                                         schedule=scripted)
            d_title = (f'seed={seed}  scripted disturbance rejection  '
                       f'cum_raw={ep_d["cum_raw_reward"]:+.2f}  '
                       f'mean_cv_v={ep_d["mean_cv_violation"]:.4f}')
            ann = plot_disturbance_rejection(
                ep_d, per_seed_dir / 'disturbance_rejection.png', title=d_title)
            disturbance_records.append({
                'seed': seed,
                'cum_raw_reward': ep_d['cum_raw_reward'],
                'mean_cv_violation': ep_d['mean_cv_violation'],
                'mean_mv_violation': ep_d['mean_mv_violation'],
                'event_annotations': ann or [],
                'schedule': ep_d['schedule'],
            })
        except Exception as e:
            print(f'[val] scripted-disturbance episode skipped (seed {seed}): {e!r}',
                  flush=True)

        seed_results.append(eps)
        print(f'[val] seed {seed}: {len(eps)} episodes done', flush=True)

    plot_summary(seed_results, out_dir / 'summary.png',
                  title=f'{controller_dir.name}  validation summary  '
                        f'({seeds} seeds × {episodes} eps)')

    cum = np.array([m['cum_raw_reward'] for m in metrics_records])
    cv_v = np.array([m['mean_cv_violation'] for m in metrics_records])
    mv_v = np.array([m['mean_mv_violation'] for m in metrics_records])
    summary = {
        'controller_dir': str(controller_dir),
        'simulation_dir': str(sim_dir),
        'ckpt': str(ckpt_path),
        'deterministic': deterministic,
        'n_seeds': int(seeds),
        'episodes_per_seed': int(episodes),
        'n_episodes_total': int(len(metrics_records)),
        'cum_raw_reward_mean': float(cum.mean()),
        'cum_raw_reward_std': float(cum.std()),
        'cum_raw_reward_min': float(cum.min()),
        'cum_raw_reward_max': float(cum.max()),
        'mean_cv_violation_mean': float(cv_v.mean()),
        'mean_mv_violation_mean': float(mv_v.mean()),
        'episodes': metrics_records,
        'disturbance_rejection': disturbance_records,
    }
    with open(out_dir / 'validation_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print('[val] done.', flush=True)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description='Validate a trained DreamerV4 controller against held-out '
                    'episodes drawn from the training disturbance distribution.')
    parser.add_argument('--controller-dir', '-c', required=True,
                        help='Output directory of a training run (contains '
                             'final.pt, run_plan.json).')
    parser.add_argument('--simulation-dir', '-s', default=None,
                        help='Override the simulation directory '
                             '(default: read from run_plan.json).')
    parser.add_argument('--ckpt', default='final.pt',
                        help='Checkpoint filename within --controller-dir.')
    parser.add_argument('--episodes', type=int, default=3,
                        help='Episodes per seed.')
    parser.add_argument('--seeds', type=int, default=3,
                        help='Number of validation seeds.')
    parser.add_argument('--out', default=None,
                        help='Validation output dir (default: '
                             '<controller-dir>/validation).')
    parser.add_argument('--stochastic', action='store_true',
                        help='Sample actions stochastically '
                             '(default: deterministic argmax).')
    args = parser.parse_args()
    summary = run_validation(controller_dir=args.controller_dir,
                             simulation_dir=args.simulation_dir,
                             ckpt=args.ckpt,
                             episodes=args.episodes, seeds=args.seeds,
                             out=args.out,
                             deterministic=not args.stochastic)
    print(json.dumps({k: v for k, v in summary.items() if k != 'episodes'},
                     indent=2), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
