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

def build_scripted_disturbance_schedule(env, *, n_events: int = 0,
                                         magnitude_frac: float = 0.0,
                                         profile: str = 'holdout_a',
                                         seed: int = 2026,
                                         ) -> List[Dict]:
    """Build a plant-aware validation disturbance schedule.

    Mirrors ``neural-apc-pytorch``'s ``build_output_disturbance_schedule``
    (single ``holdout_a`` profile) but simplified for V4: no profile
    branching, no extra knobs.  The schedule is realistic — magnitudes
    auto-adapt to channel spans, identified plant dynamics (τ, dead-time,
    DV→CV gains) and the MV→CV authority budget — so the controller is
    asked to reject only physically-rejectable disturbances.

    Mix of event types:
      * **measured DV**  — first event guaranteed; sized to deliver a
        target CV impact via the identified DV→CV gain.
      * **unmeasured CV** — second event guaranteed; pushes a CV directly,
        with violation events allowed outside the bound band so the
        controller must drive recovery.

    Each event is a one-shot step held for the rest of the episode.
    Output is a list of dicts compatible with ``env._schedule`` and
    ``utils.training_disturbance.apply_disturbance_schedule``.

    The ``n_events``/``magnitude_frac`` arguments are accepted for
    backward compatibility only — they are ignored in favour of the
    plant-aware draw.
    """
    from utils.training_disturbance import (
        _channel_catalog, _load_identifier_context,
        compute_mv_authority_to_cv, get_authority_target_frac,
        clamp_event_to_authority_budget,
    )
    rng = np.random.default_rng(int(seed))

    catalog = _channel_catalog(env.sim)
    dv_targets = list(catalog.get('dv', []))
    cv_targets = list(catalog.get('cv', []))
    if not dv_targets and not cv_targets:
        return []

    id_ctx = _load_identifier_context()
    lookback = int((id_ctx.get('lookback', {}) or {}).get('identified_lookback', 0) or 0)
    tau_dom = float((id_ctx.get('dynamics', {}) or {}).get('tau_dominant_identified', 0.0) or 0.0)
    dead_time = float((id_ctx.get('dynamics', {}) or {}).get('dead_time_identified', 0.0) or 0.0)
    dv_gain = id_ctx.get('dv_gain_to_cv', {}) if isinstance(id_ctx, dict) else {}

    ep_len = int(env.cfg.episode_length)
    dyn_horizon = max(1.0, dead_time + tau_dom)
    settle = int(max(40.0, round(max(0.6 * float(lookback), 1.25 * dyn_horizon))))
    min_gap = max(14, int(0.95 * settle))

    # holdout_a profile (validation default).  Single profile — keep it
    # simple: number of events scales with episode length, half violation,
    # 40% of CV-class events go to unmeasured-CV.
    n_total = int(max(3, min(14, ep_len // max(36, settle))))
    p_violation = 0.45
    p_cv_unmeasured = 0.40

    earliest = max(8, int(0.06 * ep_len))
    latest = max(earliest + 1, int(0.92 * ep_len))
    starts: List[int] = []
    for _ in range(max(50, 15 * n_total)):
        if len(starts) >= n_total:
            break
        s = int(rng.integers(earliest, latest + 1))
        if all(abs(s - s0) >= min_gap for s0 in starts):
            starts.append(s)
    starts.sort()
    if not starts:
        starts = [earliest]

    cv_widths = []
    for ch in cv_targets:
        b = ch.get('bounds')
        if isinstance(b, list) and len(b) >= 2:
            cv_widths.append(max(1e-6, float(b[1]) - float(b[0])))
    cv_span_ref = (float(np.median(np.asarray(cv_widths, dtype='float64')))
                    if cv_widths else 10.0)

    mv_authority_cv = compute_mv_authority_to_cv(env.sim, id_ctx)
    authority_frac = get_authority_target_frac()
    cumulative_offset: Dict[str, float] = {}
    cumulative_cv_impact = 0.0

    schedule: List[Dict] = []
    for i, start in enumerate(starts):
        # Force a useful diagnostic mix on the first two events:
        # i=0 → measured DV, i=1 → unmeasured CV.
        if dv_targets and cv_targets and i == 0:
            target = dv_targets[int(rng.integers(0, len(dv_targets)))]
            source, target_group = 'measured_dv', 'dv'
        elif dv_targets and cv_targets and i == 1:
            target = cv_targets[int(rng.integers(0, len(cv_targets)))]
            source, target_group = 'unmeasured_cv', 'cv'
        else:
            use_cv = bool(cv_targets) and (rng.uniform() < p_cv_unmeasured)
            if use_cv or not dv_targets:
                target = cv_targets[int(rng.integers(0, len(cv_targets)))]
                source, target_group = 'unmeasured_cv', 'cv'
            else:
                target = dv_targets[int(rng.integers(0, len(dv_targets)))]
                source, target_group = 'measured_dv', 'dv'

        is_violation = bool(rng.uniform() < p_violation)
        intent = 'violation' if is_violation else 'economic'

        # Direction: anti-drift bias when this channel has accumulated
        # > 15% of its span in one direction.
        ch_key = f"{target_group}_{int(target.get('pos', 0))}"
        cum = cumulative_offset.get(ch_key, 0.0)
        b_ref = target.get('bounds')
        ch_span = (max(1e-6, float(b_ref[1]) - float(b_ref[0]))
                   if isinstance(b_ref, list) and len(b_ref) >= 2 else 1.0)
        if abs(cum) / ch_span > 0.15 and abs(cum) > 1e-9:
            sign = -1.0 if cum > 0 else 1.0
            if rng.uniform() < 0.20:
                sign = -sign
        else:
            sign = -1.0 if rng.uniform() < 0.5 else 1.0

        span = ch_span
        if source == 'measured_dv':
            frac = float(rng.uniform(0.05, 0.16) if not is_violation
                         else rng.uniform(0.18, 0.38))
            mag = sign * frac * span
            gain = float(dv_gain.get(str(target.get('name', '')),
                                      dv_gain.get(f"dv_{int(target.get('pos', 0))}", 0.0))
                          or 0.0)
            if gain > 1e-8:
                desired_cv = float(rng.uniform(0.08, 0.18) if not is_violation
                                    else rng.uniform(0.30, 0.55)) * cv_span_ref
                needed = desired_cv / gain
                mag = sign * max(abs(mag), abs(needed))
            allow_oob = False
            cv_per_unit = abs(gain) if gain > 1e-8 else 0.0
        else:  # unmeasured_cv
            frac = float(rng.uniform(0.08, 0.18) if not is_violation
                         else rng.uniform(0.30, 0.60))
            mag = sign * frac * span
            allow_oob = True
            cv_per_unit = 1.0

        # Authority-budget clip (shared with training).
        if cv_per_unit > 0.0 and mv_authority_cv > 1e-9 and authority_frac > 0.0:
            new_delta, achieved = clamp_event_to_authority_budget(
                proposed_delta=float(mag),
                cv_impact_per_unit=float(cv_per_unit),
                cumulative_cv_impact=float(cumulative_cv_impact),
                mv_authority_cv=float(mv_authority_cv),
                target_frac=float(authority_frac),
            )
            mag = float(new_delta)
            cumulative_cv_impact += float(achieved)

        if abs(mag) < 1e-6:
            continue

        color = '#e76f51' if is_violation else '#2a9d8f'
        ch_name = str(target.get('name', f'{target_group}_{int(target.get("pos", 0))}'))
        schedule.append({
            'name': f'{source}_{ch_name}_{intent}_{i + 1}',
            'start': int(start),
            'target_group': target_group,
            'target_pos': int(target.get('pos', 0)),
            'target_state_index': int(target.get('state_index', 0)),
            'target_name': ch_name,
            'source': source,
            'intent': intent,
            'delta': float(mag),
            'allow_out_of_bounds': bool(allow_oob),
            'color': color,
            'duration': int(ep_len - int(start)),
            'shape': 'step',
            'period': float(max(2.0, ep_len)),
            '_applied': False,
            '_is_violation': bool(is_violation),
        })
        cumulative_offset[ch_key] = cum + float(mag)

    schedule.sort(key=lambda x: int(x.get('start', 0)))
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
    # Per-step active MV/CV bounds and CV targets so the plot can
    # render the bound-change schedule the operator/agent saw.  The
    # base bounds (constant) are recorded separately in the rollout
    # dict; these are the *current* values that vary across the
    # episode whenever ``RuntimeSetpointManager`` schedules a change.
    n_mv_aux = int(getattr(env.setpoint_mgr, 'n_mv', 0))
    n_cv_aux = int(getattr(env.setpoint_mgr, 'n_cv', 0))
    current_mv_bounds_t = np.zeros((T, n_mv_aux, 2), dtype='float32')
    current_cv_bounds_t = np.zeros((T, n_cv_aux, 2), dtype='float32')
    current_cv_targets_t = np.zeros((T, n_cv_aux), dtype='float32')

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
        # Record the *raw* (physical-units) state so plots/npz read true
        # plant values, not the post-standardizer z-scores that the
        # tokenizer sees.  Falls back to the normalized obs slice if the
        # env did not expose ``raw_state`` for back-compat.
        raw_st = info.get('raw_state')
        if raw_st is None:
            states[t] = next_window[-1, :state_dim]
        else:
            arr = np.asarray(raw_st, dtype='float32').reshape(-1)
            states[t, :min(state_dim, arr.shape[0])] = arr[:state_dim]
        actions_norm[t] = a_np
        controls[t] = np.asarray(env._prev_control, dtype='float32')
        raw_rewards[t] = float(info.get('raw_reward', 0.0))
        scaled_rewards[t] = float(scaled_r)
        cv_violations[t] = float(comps.get('cv_violation_penalty', 0.0))
        mv_violations[t] = float(comps.get('mv_violation_penalty', 0.0))
        if n_mv_aux > 0:
            current_mv_bounds_t[t] = np.asarray(
                env.setpoint_mgr.current_mv_bounds, dtype='float32')
        if n_cv_aux > 0:
            current_cv_bounds_t[t] = np.asarray(
                env.setpoint_mgr.current_cv_bounds, dtype='float32')
            current_cv_targets_t[t] = np.asarray(
                env.setpoint_mgr.current_cv_targets, dtype='float32')
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
        'cv_target_enabled': [bool(x) for x in
                                getattr(env.setpoint_mgr,
                                          'cv_target_enabled', [])],
        'current_mv_bounds_t': current_mv_bounds_t[:t + 1],
        'current_cv_bounds_t': current_cv_bounds_t[:t + 1],
        'current_cv_targets_t': current_cv_targets_t[:t + 1],
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
    cv_target_enabled = ep.get('cv_target_enabled') or []
    # Per-step (T, n_ch[, 2]) traces of the operator-active bounds /
    # targets.  Empty / shape-(T, 0, 2) when the setpoint manager did
    # not run, in which case the plot falls back to the constant base
    # bounds only.
    cur_mv_b_t = np.asarray(ep.get('current_mv_bounds_t') or [],
                              dtype='float32')
    cur_cv_b_t = np.asarray(ep.get('current_cv_bounds_t') or [],
                              dtype='float32')
    cur_cv_tgt_t = np.asarray(ep.get('current_cv_targets_t') or [],
                                dtype='float32')
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
        bounds_t = (cur_mv_b_t[:, k] if (cur_mv_b_t.ndim == 3
                                          and k < cur_mv_b_t.shape[1])
                     else None)
        channels.append({'group': 'mv', 'series': controls[:, k] if k < controls.shape[1] else None,
                         'label': _name(i, f'MV[{i}]'), 'bounds': bounds,
                         'bounds_t': bounds_t,
                         'norm': norm, 'target': None, 'target_t': None,
                         'color': '#1f77b4'})
    for i in dv_idx:
        if i >= states.shape[1]:
            continue
        channels.append({'group': 'dv', 'series': states[:, i],
                         'label': _name(i, f'DV[{i}]'), 'bounds': None,
                         'bounds_t': None,
                         'norm': None, 'target': None, 'target_t': None,
                         'color': '#9467bd'})
    for k, i in enumerate(cv_idx):
        if i >= states.shape[1]:
            continue
        bounds = cv_bounds[k] if k < len(cv_bounds) else None
        norm = cv_norm[k] if k < len(cv_norm) else None
        tgt_enabled = (k < len(cv_target_enabled)
                        and bool(cv_target_enabled[k]))
        target = (cv_targets[k] if (tgt_enabled and k < len(cv_targets))
                   else None)
        bounds_t = (cur_cv_b_t[:, k] if (cur_cv_b_t.ndim == 3
                                          and k < cur_cv_b_t.shape[1])
                     else None)
        target_t = (cur_cv_tgt_t[:, k] if (tgt_enabled
                                            and cur_cv_tgt_t.ndim == 2
                                            and k < cur_cv_tgt_t.shape[1])
                     else None)
        channels.append({'group': 'cv', 'series': states[:, i],
                         'label': _name(i, f'CV[{i}]'), 'bounds': bounds,
                         'bounds_t': bounds_t,
                         'norm': norm, 'target': target,
                         'target_t': target_t,
                         'color': '#2ca02c'})

    n_rows = max(1, len(channels)) + 2  # + cum reward + reward/violation companion
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
                            label='CV base band')
                ax.axhline(lo_b, color='#d32f2f', linestyle='--', linewidth=1.0,
                            label='CV base low')
                ax.axhline(hi_b, color='#d32f2f', linestyle='--', linewidth=1.0,
                            label='CV base high')
            else:
                ax.axhline(lo_b, color='r', linestyle='--', linewidth=1.0,
                            label='Base low')
                ax.axhline(hi_b, color='r', linestyle='--', linewidth=1.0,
                            label='Base high')

        # Per-step active bounds (operator schedule) — overlays the
        # base bound box.  Shown as a step trace (post-step semantics)
        # so the operator can see exactly when each bound moved.
        bounds_t = ch.get('bounds_t')
        if (bounds_t is not None
                and isinstance(bounds_t, np.ndarray)
                and bounds_t.ndim == 2
                and bounds_t.shape[1] >= 2
                and bounds_t.shape[0] >= 1):
            n = min(bounds_t.shape[0], len(t_arr))
            lo_arr = np.asarray(bounds_t[:n, 0], dtype='float32')
            hi_arr = np.asarray(bounds_t[:n, 1], dtype='float32')
            t_seg = t_arr[:n]
            if ch['group'] == 'cv':
                ax.fill_between(t_seg, lo_arr, hi_arr,
                                  step='post', color='#fb8c00', alpha=0.10,
                                  label='CV active band')
                ax.step(t_seg, lo_arr, where='post', color='#b71c1c',
                         linewidth=1.6, label='CV active low')
                ax.step(t_seg, hi_arr, where='post', color='#b71c1c',
                         linewidth=1.6, label='CV active high')
            else:
                ax.step(t_seg, lo_arr, where='post', color='#c62828',
                         linewidth=1.4, label='MV active low')
                ax.step(t_seg, hi_arr, where='post', color='#c62828',
                         linewidth=1.4, label='MV active high')

        norm = ch.get('norm')
        if norm is not None and len(norm) >= 2 and \
           np.isfinite(norm[0]) and np.isfinite(norm[1]):
            ax.axhline(float(norm[0]), color='#6c757d', linestyle=':',
                        linewidth=1.0, label='Norm low')
            ax.axhline(float(norm[1]), color='#6c757d', linestyle=':',
                        linewidth=1.0, label='Norm high')

        target = ch.get('target')
        target_t = ch.get('target_t')
        if (target_t is not None
                and isinstance(target_t, np.ndarray)
                and target_t.ndim == 1
                and target_t.size >= 1
                and np.any(np.isfinite(target_t))):
            n = min(target_t.size, len(t_arr))
            ax.step(t_arr[:n], np.asarray(target_t[:n], dtype='float32'),
                     where='post', color='#1b5e20', linewidth=1.6,
                     linestyle='-.', label='Target (active)')
        elif target is not None and np.isfinite(target):
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

    ax = axes[-2]
    ax.plot(t_arr, np.cumsum(ep['raw_rewards']), color='C2', lw=1.0,
            label=f"raw cum (final={ep['cum_raw_reward']:+.1f})")
    _draw_disturbance_markers(ax)
    ax.set_ylabel('cum reward')
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)

    # Reward / violation companion: instantaneous raw reward (left axis) +
    # cumulative CV-violation count (right axis).  Frames the cum-reward
    # subplot above by exposing where penalties are coming from.
    ax = axes[-1]
    ax.plot(t_arr, ep['raw_rewards'], color='#555555', lw=0.8, alpha=0.85,
            label='raw reward (per step)')
    ax.axhline(0.0, color='#888888', linestyle=':', linewidth=0.8)
    ax.set_ylabel('raw r/step')
    ax.grid(True, alpha=0.3)
    cv_v_raw = ep.get('cv_violations')
    cv_v = (np.asarray(cv_v_raw, dtype='float64')
            if cv_v_raw is not None and len(cv_v_raw) > 0
            else np.zeros_like(t_arr, dtype='float64'))
    if cv_v.size == t_arr.size and cv_v.size > 0:
        cv_count = np.cumsum((cv_v > 1e-9).astype('float64'))
        ax2 = ax.twinx()
        ax2.plot(t_arr, cv_count, color='#d32f2f', lw=1.2,
                  label=f'cum CV viol (final={int(cv_count[-1])})')
        ax2.set_ylabel('cum CV viol', color='#d32f2f')
        ax2.tick_params(axis='y', labelcolor='#d32f2f')
        # Combined legend
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, loc='upper left', fontsize=8)
    else:
        ax.legend(loc='upper left', fontsize=8)
    _draw_disturbance_markers(ax)
    ax.set_xlabel('time step')

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
        policy_type=str(getattr(cfg, 'policy_type', 'continuous')),
        policy_init_log_std=float(getattr(cfg, 'policy_init_log_std', -0.5)),
        policy_log_std_min=float(getattr(cfg, 'policy_log_std_min', -2.3)),
        policy_log_std_max=float(getattr(cfg, 'policy_log_std_max', 0.0)),
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
    # Restore the env-side obs normalizer stats saved with the checkpoint
    # (added 2026-05-03 — the trainer applies running standardization to
    # obs before the tokenizer; evaluation must use the same stats with
    # learning frozen so the model sees the distribution it was trained
    # against).  Older checkpoints without 'obs_norm' fall back to the
    # default (mean=0, var=1) stats — which is also what those models
    # were effectively trained with.
    obs_norm_state = ckpt_obj.get('obs_norm') if isinstance(ckpt_obj, dict) else None
    for s in range(int(seeds)):
        seed = 10_000 + s  # held-out from training (which used SEED=0..N).
        rng = np.random.default_rng(seed)
        env = APCEnv(cfg, rng)
        if obs_norm_state is not None:
            try:
                env.set_obs_norm_stats(
                    mean=np.asarray(obs_norm_state.get('mean')),
                    var=np.asarray(obs_norm_state.get('var')),
                    count=float(obs_norm_state.get('count', 1.0)),
                    learn=False,
                )
            except Exception as e:
                print(f'[val] obs_norm restore skipped: {e!r}', flush=True)
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

        # ---- Disturbance-rejection plot (plant-aware holdout_a profile) ----
        try:
            scripted = build_scripted_disturbance_schedule(env, seed=seed)
            ep_d = run_scripted_episode(env, model, device,
                                         deterministic=deterministic,
                                         schedule=scripted)
            d_title = (f'seed={seed}  scripted disturbance rejection  '
                       f'cum_raw={ep_d["cum_raw_reward"]:+.2f}  '
                       f'mean_cv_v={ep_d["mean_cv_violation"]:.4f}')
            ann = plot_disturbance_rejection(
                ep_d, per_seed_dir / 'disturbance_rejection.png', title=d_title)
            # Persist the raw trajectory for offline analysis (added
            # 2026-05-03 — diagnosing constant-action collapse required
            # re-running the eval rollouts because the PNG is the only
            # artefact, which is wasteful and lossy).
            try:
                npz_path = per_seed_dir / 'disturbance_rejection.npz'
                # Convert lists/dicts in the schedule to a JSON blob so
                # np.savez doesn't choke on object arrays.
                sched_json = json.dumps(ep_d.get('schedule', []), default=str)
                np.savez(
                    npz_path,
                    states=np.asarray(ep_d['states'], dtype='float32'),
                    actions_norm=np.asarray(ep_d['actions_norm'], dtype='float32'),
                    controls=np.asarray(ep_d['controls'], dtype='float32'),
                    raw_rewards=np.asarray(ep_d['raw_rewards'], dtype='float32'),
                    scaled_rewards=np.asarray(ep_d['scaled_rewards'], dtype='float32'),
                    cv_violations=np.asarray(ep_d['cv_violations'], dtype='float32'),
                    mv_violations=np.asarray(ep_d['mv_violations'], dtype='float32'),
                    cv_indices=np.asarray(ep_d.get('cv_indices', []), dtype='int64'),
                    mv_indices=np.asarray(ep_d.get('mv_indices', []), dtype='int64'),
                    dv_indices=np.asarray(ep_d.get('dv_indices', []), dtype='int64'),
                    mv_norm_ranges=np.asarray(ep_d.get('mv_norm_ranges', []),
                                                dtype='float32'),
                    cv_norm_ranges=np.asarray(ep_d.get('cv_norm_ranges', []),
                                                dtype='float32'),
                    mv_bounds=np.asarray(ep_d.get('mv_bounds', []),
                                           dtype='float32'),
                    cv_bounds=np.asarray(ep_d.get('cv_bounds', []),
                                           dtype='float32'),
                    cv_targets=np.asarray(ep_d.get('cv_targets', []),
                                            dtype='float32'),
                    cv_target_enabled=np.asarray(
                        ep_d.get('cv_target_enabled', []), dtype=bool),
                    current_mv_bounds_t=np.asarray(
                        ep_d.get('current_mv_bounds_t',
                                   np.zeros((0, 0, 2))), dtype='float32'),
                    current_cv_bounds_t=np.asarray(
                        ep_d.get('current_cv_bounds_t',
                                   np.zeros((0, 0, 2))), dtype='float32'),
                    current_cv_targets_t=np.asarray(
                        ep_d.get('current_cv_targets_t',
                                   np.zeros((0, 0))), dtype='float32'),
                    sample_rate=np.asarray([int(ep_d.get('sample_rate', 1))],
                                             dtype='int64'),
                    episode_length=np.asarray([int(ep_d.get('episode_length',
                                                              len(ep_d['raw_rewards'])))],
                                                dtype='int64'),
                    state_variables=np.asarray(
                        ep_d.get('state_variables', []), dtype=object),
                    schedule_json=np.asarray([sched_json], dtype=object),
                )
            except Exception as ee:
                print(f'[val] disturbance_rejection.npz skipped '
                      f'(seed {seed}): {ee!r}', flush=True)
            disturbance_records.append({
                'seed': seed,
                'cum_raw_reward': ep_d['cum_raw_reward'],
                'mean_cv_violation': ep_d['mean_cv_violation'],
                'mean_mv_violation': ep_d['mean_mv_violation'],
                'event_annotations': ann or [],
                'schedule': ep_d['schedule'],
            })
        except Exception as e:
            import traceback
            print(f'[val] scripted-disturbance episode skipped (seed {seed}): {e!r}',
                  flush=True)
            traceback.print_exc()

        seed_results.append(eps)
        print(f'[val] seed {seed}: {len(eps)} episodes done', flush=True)

    plot_summary(seed_results, out_dir / 'summary.png',
                  title=f'{controller_dir.name}  validation summary  '
                        f'({seeds} seeds × {episodes} eps)')

    # ---- Training-stage + WM-fidelity diagnostics ------------------------
    # Run once per validation invocation on a fresh env so we don't pay
    # per-seed cost.  Tells the operator which training stage (P1 WM,
    # P2 reward MTP / BC, P3 PMPO) is the bottleneck.
    try:
        from evaluation.diagnostics import compute_training_diagnostics
        diag_env = APCEnv(cfg, np.random.default_rng(99_999))
        if obs_norm_state is not None:
            try:
                diag_env.set_obs_norm_stats(
                    mean=np.asarray(obs_norm_state.get('mean')),
                    var=np.asarray(obs_norm_state.get('var')),
                    count=float(obs_norm_state.get('count', 1.0)),
                    learn=False,
                )
            except Exception:
                pass
        diag = compute_training_diagnostics(
            controller_dir=controller_dir,
            env=diag_env,
            model=model,
            device=device,
            out_dir=out_dir,
            k_max=int(getattr(cfg, 'horizon', 32)),
            gamma=float(getattr(cfg, 'gamma', 0.997)),
        )
        flags = (diag.get('stage_metrics') or {}).get('flags') or []
        if flags:
            print('[val] training-stage flags:', flush=True)
            for fl in flags:
                print(f'        - {fl}', flush=True)
    except Exception as e:
        print(f'[val] diagnostics skipped: {e}', flush=True)

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
