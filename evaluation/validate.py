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
from typing import Dict, List, Optional, Tuple

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

    # holdout_a profile (validation default).  Only measured-DV events
    # are scripted — the unmeasured CV disturbance is now provided
    # automatically by the hidden OU process attached to the env
    # (``env._hidden_disturbance`` with ``_hidden_disturbance_force=True``),
    # which fires on every validation episode.
    n_total = int(max(3, min(14, ep_len // max(36, settle))))
    p_violation = 0.45

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
    if not dv_targets:
        return schedule
    for i, start in enumerate(starts):
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
    # Per-step hidden (unmeasured) OU disturbance offset injected into each
    # CV channel (aligned to ``env.cv_indices``).  Surfaces the otherwise-
    # invisible disturbance the agent had to reject.
    n_cv_h = len(env.cv_indices)
    hidden_dist_t = np.zeros((T, n_cv_h), dtype='float32')

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

    _is_rssm = getattr(model, 'world_model_type', 'sf_transformer') == 'rssm'
    _rssm_state = (model.dynamics.initial_state(1, device)
                   if _is_rssm else None)
    _rssm_prev_a = (torch.zeros(1, action_dim, device=device)
                    if _is_rssm else None)

    for t in range(T):
        ow = torch.from_numpy(obs_window).to(device)
        a_ctx = torch.from_numpy(a_history).to(device)
        with torch.no_grad():
            with torch.amp.autocast(device_type=device.type,
                                     dtype=torch.bfloat16,
                                     enabled=(device.type == 'cuda')):
                if _is_rssm:
                    _o = torch.from_numpy(
                        obs_window[-1]).to(device).unsqueeze(0)
                    _emb = model.dynamics.embed(_o)
                    _post, _ = model.dynamics.obs_step(
                        _rssm_state, _rssm_prev_a, _emb, sample=True)
                    agent_hid = _post.feat
                    _rssm_state = _post
                else:
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
        if _is_rssm:
            _rssm_prev_a = torch.from_numpy(a_np).to(device).unsqueeze(0)
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
        hd = info.get('hidden_disturbance')
        if hd is not None:
            hd = np.asarray(hd, dtype='float32').reshape(-1)
            hidden_dist_t[t, :min(n_cv_h, hd.shape[0])] = hd[:n_cv_h]
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
        'hidden_disturbance_t': hidden_dist_t[:t + 1],
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


# ---------------------------------------------------------------------------
# Episode metrics  (process-control standard, simulator-agnostic)
# ---------------------------------------------------------------------------

def _cv_active_target(ep: Dict, cv_row: int) -> np.ndarray | None:
    """Per-step target for CV ``cv_row`` if a target is enabled, else None.

    Returns a (T,) array — uses the time-varying ``current_cv_targets_t``
    when the setpoint manager scheduled changes, else falls back to the
    constant ``cv_targets`` value.
    """
    enabled = ep.get('cv_target_enabled') or []
    if cv_row >= len(enabled) or not bool(enabled[cv_row]):
        return None
    cur_raw = ep.get('current_cv_targets_t')
    cur = (np.asarray(cur_raw, dtype='float32')
            if cur_raw is not None and len(cur_raw) > 0
            else np.zeros((0, 0), dtype='float32'))
    if cur.ndim == 2 and cv_row < cur.shape[1] and np.isfinite(cur[:, cv_row]).any():
        return cur[:, cv_row].astype('float32')
    base = ep.get('cv_targets') or []
    if cv_row < len(base) and np.isfinite(base[cv_row]):
        T = ep['episode_length']
        return np.full(T, float(base[cv_row]), dtype='float32')
    return None


def compute_episode_metrics(ep: Dict) -> Dict[str, float]:
    """V3-equivalent operator-facing metrics, plus tracking IAE/ITAE/ISE.

    All metrics are simulator-agnostic — they derive from the rollout
    arrays, the bound boxes recorded with the episode, and (where
    available) the per-step active CV target.  No simulator-specific
    column names or knobs.
    """
    states = ep['states']
    controls = ep['controls']
    cv_idx = ep.get('cv_indices') or []
    mv_bounds = ep.get('mv_bounds') or []
    cv_bounds = ep.get('cv_bounds') or []
    T = int(ep['episode_length'])
    n_mv = controls.shape[1]

    # --- MV economic / actuator-health metrics ---------------------------
    mv_tv = float(np.sum(np.abs(np.diff(controls, axis=0)))) if T > 1 else 0.0
    activity_ratios: List[float] = []
    hugging_scores: List[float] = []
    usage_scores: List[float] = []
    reversal_rates: List[float] = []
    for j in range(n_mv):
        col = controls[:, j].astype('float64')
        b = mv_bounds[j] if j < len(mv_bounds) else None
        if (isinstance(b, list) and len(b) >= 2
                and np.isfinite(b[0]) and np.isfinite(b[1])
                and b[1] > b[0]):
            lo, hi = float(b[0]), float(b[1])
            rng = hi - lo
        else:
            rng = float(np.nanmax(col) - np.nanmin(col)) or 1.0
            lo, hi = float(np.nanmin(col)), float(np.nanmax(col))
        if rng > 1e-9:
            activity_ratios.append(float(np.nanstd(col) / rng))
            mean_col = float(np.nanmean(col))
            hugging_scores.append(float(max(0.0,
                min((mean_col - lo) / rng, (hi - mean_col) / rng))))
            usage_scores.append(float(
                (np.nanmax(col) - np.nanmin(col)) / rng))
        # Reversal rate: sign-changes per step on a 0.1%-of-range deadband.
        if T > 1:
            d = np.diff(col)
            deadband = 1e-3 * rng
            signed = np.where(np.abs(d) > deadband, np.sign(d), 0.0)
            nz = signed[signed != 0.0]
            flips = int(np.sum(np.abs(np.diff(nz)) > 1.0)) if nz.size >= 2 else 0
            reversal_rates.append(float(flips) / float(max(1, len(d))))

    # --- CV tracking: IAE / ITAE / ISE per CV (where target enabled) ----
    iae_per_cv: List[float] = []
    itae_per_cv: List[float] = []
    ise_per_cv: List[float] = []
    for k, cidx in enumerate(cv_idx):
        if cidx >= states.shape[1]:
            continue
        tgt = _cv_active_target(ep, k)
        if tgt is None:
            continue
        err = states[:T, cidx].astype('float64') - tgt[:T].astype('float64')
        # Normalise by CV bound width so cross-CV / cross-sim comparison is meaningful.
        b = cv_bounds[k] if k < len(cv_bounds) else None
        if (isinstance(b, list) and len(b) >= 2
                and np.isfinite(b[0]) and np.isfinite(b[1])
                and b[1] > b[0]):
            denom = float(b[1]) - float(b[0])
        else:
            denom = float(np.nanstd(states[:T, cidx])) or 1.0
        e = err / max(1e-9, denom)
        iae_per_cv.append(float(np.sum(np.abs(e))))
        itae_per_cv.append(float(np.sum(np.arange(T) * np.abs(e))))
        ise_per_cv.append(float(np.sum(e ** 2)))

    return {
        'cv_violation_mean': float(ep.get('mean_cv_violation', 0.0)),
        'mv_violation_mean': float(ep.get('mean_mv_violation', 0.0)),
        'mv_tv': mv_tv,
        'mv_activity_ratio': float(np.mean(activity_ratios)) if activity_ratios else 0.0,
        'mv_bound_hugging_score': float(np.min(hugging_scores)) if hugging_scores else 1.0,
        'mv_bound_usage': float(np.mean(usage_scores)) if usage_scores else 0.0,
        'mv_reversal_rate': float(np.mean(reversal_rates)) if reversal_rates else 0.0,
        'economic_score': float(np.mean(ep['raw_rewards'])) if T > 0 else 0.0,
        'cum_raw_reward': float(ep.get('cum_raw_reward', 0.0)),
        'iae_normed_mean': float(np.mean(iae_per_cv)) if iae_per_cv else 0.0,
        'itae_normed_mean': float(np.mean(itae_per_cv)) if itae_per_cv else 0.0,
        'ise_normed_mean': float(np.mean(ise_per_cv)) if ise_per_cv else 0.0,
        'iae_normed_per_cv': iae_per_cv,
        'itae_normed_per_cv': itae_per_cv,
        'ise_normed_per_cv': ise_per_cv,
    }


def compute_event_response_metrics(ep: Dict, *, settle_band: float = 0.05,
                                     window_steps: int | None = None
                                     ) -> Dict[str, object]:
    """Per-disturbance response metrics on the *most-impacted* CV.

    For each scheduled event we look at the rollout window
    ``[start, start + window_steps]`` (default = 5τ if available, else
    200) and compute on each CV:
      * ``peak_overshoot`` — max signed deviation from pre-event baseline,
        normalised by CV bound width when available.
      * ``settle_steps``   — first sample where |dev| stays inside
        ``settle_band × bound_width`` for the rest of the window.
        ``None`` if it never settles.
      * ``iae_window``     — sum |dev| / bound_width across the window.

    The most-impacted CV (largest |overshoot|) is reported per event;
    aggregates (median, p90, max) across events are returned at the top.
    """
    states = ep['states']
    cv_idx = ep.get('cv_indices') or []
    cv_bounds = ep.get('cv_bounds') or []
    T = int(ep['episode_length'])

    # Default window: 5 × identified τ (sample-rate scaled) if known,
    # else 200 steps.  Falls back to the schedule-builder's heuristic
    # when no identifier context is on disk.
    if window_steps is None:
        window_steps = 200
        try:
            from utils.training_disturbance import _load_identifier_context
            ctx = _load_identifier_context()
            tau = float((ctx.get('dynamics', {}) or {})
                          .get('tau_dominant_identified', 0.0) or 0.0)
            sr = max(1, int(ep.get('sample_rate', 1)))
            if tau > 0:
                window_steps = max(60, min(400, int(round(5.0 * tau / sr))))
        except Exception:
            pass

    events = []
    for ev in ep.get('schedule') or []:
        try:
            start = int(ev.get('start', 0))
        except Exception:
            continue
        if start <= 0 or start >= T - 5:
            continue
        end = min(T, start + int(window_steps))
        pre = states[max(0, start - 20):start]
        if pre.size == 0:
            continue
        per_cv = []
        for k, cidx in enumerate(cv_idx):
            if cidx >= states.shape[1]:
                continue
            base = float(np.mean(pre[:, cidx]))
            dev = states[start:end, cidx].astype('float64') - base
            if dev.size == 0:
                continue
            b = cv_bounds[k] if k < len(cv_bounds) else None
            if (isinstance(b, list) and len(b) >= 2
                    and np.isfinite(b[0]) and np.isfinite(b[1])
                    and b[1] > b[0]):
                denom = float(b[1]) - float(b[0])
            else:
                denom = float(np.nanstd(states[:T, cidx])) or 1.0
            dev_n = dev / max(1e-9, denom)
            ovs = float(dev_n[int(np.argmax(np.abs(dev_n)))])
            band = settle_band  # already normalised
            settle = None
            for i in range(dev_n.size - 1, -1, -1):
                if abs(dev_n[i]) > band:
                    settle = i + 1
                    break
            if settle == 0:
                settle = 0
            iae_w = float(np.sum(np.abs(dev_n)))
            per_cv.append({'cv_row': k, 'cv_index': int(cidx),
                            'peak_overshoot_normed': ovs,
                            'settle_steps': settle,
                            'iae_window_normed': iae_w})
        if not per_cv:
            continue
        worst = max(per_cv, key=lambda r: abs(r['peak_overshoot_normed']))
        events.append({
            'name': str(ev.get('name', 'event')),
            'start': start, 'end': end,
            'intent': str(ev.get('intent', '')),
            'source': str(ev.get('source', '')),
            'delta': float(ev.get('delta', 0.0)),
            'worst_cv': worst,
            'per_cv': per_cv,
        })

    def _agg(key: str) -> Dict[str, float]:
        vals = [abs(e['worst_cv'][key]) if isinstance(e['worst_cv'][key], (int, float))
                else 0.0 for e in events]
        if not vals:
            return {'median': 0.0, 'p90': 0.0, 'max': 0.0, 'n': 0}
        a = np.asarray(vals, dtype='float64')
        return {'median': float(np.median(a)),
                'p90': float(np.percentile(a, 90)),
                'max': float(np.max(a)),
                'n': int(a.size)}

    settle_vals = [e['worst_cv']['settle_steps'] for e in events
                    if e['worst_cv']['settle_steps'] is not None]
    settle_agg = ({'median': float(np.median(settle_vals)),
                   'p90': float(np.percentile(settle_vals, 90)),
                   'max': float(np.max(settle_vals)),
                   'n_settled': int(len(settle_vals)),
                   'n_total': int(len(events))}
                  if settle_vals else
                  {'median': 0.0, 'p90': 0.0, 'max': 0.0,
                   'n_settled': 0, 'n_total': int(len(events))})

    return {
        'window_steps': int(window_steps),
        'settle_band_normed': float(settle_band),
        'overshoot_normed': _agg('peak_overshoot_normed'),
        'iae_window_normed': _agg('iae_window_normed'),
        'settle_steps': settle_agg,
        'events': events,
    }


# ---------------------------------------------------------------------------
# Baselines  (constant-MV; simulator-agnostic, no controller needed)
# ---------------------------------------------------------------------------

def run_constant_mv_episode(env, *, schedule: List[Dict],
                              mv_norm: float = 0.0,
                              dv_override: Optional[np.ndarray] = None) -> Dict:
    """Replay ``schedule`` with a frozen MV at ``mv_norm`` (default = mid).

    Used as the lower baseline on the disturbance-rejection plot — shows
    what the plant does with no control intervention.  Reuses the same
    APCEnv so noise/setpoint/objective code paths are identical to the
    agent rollout (only the action source differs).

    Parameters
    ----------
    dv_override : (T, n_dv) array, optional
        If provided, overwrites the simulator's DV channel(s) after each
        step with ``dv_override[t]``.  Used by the disturbance-rejection
        plot so the baseline experiences the *exact same* uncontrolled
        external disturbance trace as the agent run, making the
        baseline-vs-agent CV overlay an apples-to-apples comparison
        (DV is external by definition; the agent has no influence over
        it, so any divergence between the two runs' DV streams is
        purely noise-RNG drift between the two SimNoiseWrapper
        instances).  Without this override the user sees subtly
        different DV curves and can't tell whether the CV difference
        is due to the controller or due to a different disturbance.
    """
    obs_window = env.reset(exploration=False)
    env._schedule = list(schedule)
    T = env.cfg.episode_length
    state_dim = env.state_dim
    action_dim = env.action_dim
    states = np.zeros((T, state_dim), dtype='float32')
    controls = np.zeros((T, action_dim), dtype='float32')
    raw_rewards = np.zeros(T, dtype='float32')
    cv_violations = np.zeros(T, dtype='float32')
    mv_violations = np.zeros(T, dtype='float32')
    n_mv_aux = int(getattr(env.setpoint_mgr, 'n_mv', 0))
    n_cv_aux = int(getattr(env.setpoint_mgr, 'n_cv', 0))
    cur_mv_b = np.zeros((T, n_mv_aux, 2), dtype='float32')
    cur_cv_b = np.zeros((T, n_cv_aux, 2), dtype='float32')
    cur_cv_t = np.zeros((T, n_cv_aux), dtype='float32')

    a = np.full(action_dim, float(mv_norm), dtype='float32')
    # Resolve the underlying sim object so we can write back into its
    # state buffer when a DV override is provided.  ``env.sim`` is the
    # SimNoiseWrapper; its ``_sim`` is the raw simulator class instance
    # which holds ``episode_array`` and the live state.
    bare_sim = env.sim
    for _ in range(4):
        inner = getattr(bare_sim, '_sim', None)
        if inner is None:
            break
        bare_sim = inner
    dv_idx_list = list(env.meta.get('dv_indices') or [])
    use_dv_override = (dv_override is not None
                        and len(dv_idx_list) > 0
                        and len(dv_override) >= T)
    for t in range(T):
        next_window, _, done, info = env.step(a)
        comps = info.get('reward_components', {}) or {}
        # If a DV override is provided, force the underlying sim's
        # state to carry the agent's recorded DV trajectory at this
        # step *before* the next step uses it for dynamics.  This makes
        # the baseline experience the identical external disturbance
        # the agent saw, so the CV overlay is an apples-to-apples
        # comparison.
        if use_dv_override:
            try:
                ep_arr = getattr(bare_sim, 'episode_array', None)
                # After env.step() returns, the underlying sim has
                # advanced ``episode_counter`` and written the new row.
                # The *next* step reads ``prev = episode_array[
                # episode_counter]`` — so overwrite that exact row.
                idx = int(getattr(bare_sim, 'episode_counter', t + 1))
                if ep_arr is not None:
                    idx = max(0, min(idx, ep_arr.shape[0] - 1))
                ov = np.asarray(dv_override[t], dtype='float32').reshape(-1)
                for j, di in enumerate(dv_idx_list):
                    if j >= ov.shape[0]:
                        break
                    val = float(ov[j])
                    if ep_arr is not None and idx >= 0 and di < ep_arr.shape[1]:
                        ep_arr[idx, di] = val
            except Exception:
                pass
        raw_st = info.get('raw_state')
        if raw_st is None:
            states[t] = next_window[-1, :state_dim]
        else:
            arr = np.asarray(raw_st, dtype='float32').reshape(-1)
            states[t, :min(state_dim, arr.shape[0])] = arr[:state_dim]
        # Mirror the DV override into the recorded state so plots /
        # CSV report the agent-aligned DV (must come AFTER the
        # raw_st recording, which would otherwise clobber it).
        if use_dv_override:
            ov = np.asarray(dv_override[t], dtype='float32').reshape(-1)
            for j, di in enumerate(dv_idx_list):
                if j >= ov.shape[0]:
                    break
                if di < states.shape[1]:
                    states[t, di] = float(ov[j])
        controls[t] = np.asarray(env._prev_control, dtype='float32')
        raw_rewards[t] = float(info.get('raw_reward', 0.0))
        cv_violations[t] = float(comps.get('cv_violation_penalty', 0.0))
        mv_violations[t] = float(comps.get('mv_violation_penalty', 0.0))
        if n_mv_aux > 0:
            cur_mv_b[t] = np.asarray(env.setpoint_mgr.current_mv_bounds,
                                       dtype='float32')
        if n_cv_aux > 0:
            cur_cv_b[t] = np.asarray(env.setpoint_mgr.current_cv_bounds,
                                       dtype='float32')
            cur_cv_t[t] = np.asarray(env.setpoint_mgr.current_cv_targets,
                                       dtype='float32')
        obs_window = next_window
        if done:
            break

    return {
        'states': states[:t + 1], 'controls': controls[:t + 1],
        'raw_rewards': raw_rewards[:t + 1],
        'scaled_rewards': raw_rewards[:t + 1] * float(env.reward_scale),
        'cv_violations': cv_violations[:t + 1],
        'mv_violations': mv_violations[:t + 1],
        'cum_raw_reward': float(np.sum(raw_rewards[:t + 1])),
        'cum_reward': float(np.sum(raw_rewards[:t + 1] * float(env.reward_scale))),
        'mean_cv_violation': float(cv_violations[:t + 1].mean()),
        'mean_mv_violation': float(mv_violations[:t + 1].mean()),
        'episode_length': int(t + 1),
        'sample_rate': int(env.cfg.sample_rate),
        'cv_indices': list(env.cv_indices),
        'mv_indices': [int(x) for x in (env.meta.get('mv_indices') or [])],
        'dv_indices': [int(x) for x in (env.meta.get('dv_indices') or [])],
        'state_variables': list(env.meta.get('state_variables') or []),
        'mv_bounds': [list(b) for b in
                       getattr(env.setpoint_mgr, 'base_mv_bounds', np.zeros((0, 2)))],
        'cv_bounds': [list(b) for b in
                       getattr(env.setpoint_mgr, 'base_cv_bounds', np.zeros((0, 2)))],
        'cv_targets': [float(x) for x in
                        getattr(env.setpoint_mgr, 'base_cv_targets', [])],
        'cv_target_enabled': [bool(x) for x in
                                getattr(env.setpoint_mgr,
                                          'cv_target_enabled', [])],
        'current_mv_bounds_t': cur_mv_b[:t + 1],
        'current_cv_bounds_t': cur_cv_b[:t + 1],
        'current_cv_targets_t': cur_cv_t[:t + 1],
        'reward_scale': float(env.reward_scale),
        'schedule': schedule,
        'mv_norm_ranges': [list(b) for b in env.mv_norm_ranges],
        'cv_norm_ranges': [list(b) for b in env.cv_norm_ranges],
    }


# ---------------------------------------------------------------------------
# CSV / schedule.txt artefacts  (operator-friendly, opens in Excel)
# ---------------------------------------------------------------------------

def _episode_to_dataframe_dict(ep: Dict, suffix: str = '') -> Dict[str, np.ndarray]:
    """Build a column-name → array dict suitable for pandas.DataFrame.

    Naming follows the V3 convention so spreadsheets / plotting scripts
    transfer over: ``MV_<name>``, ``CV_<name>``, ``DV_<name>``,
    ``MV_<name>_bound_lo/hi``, ``CV_<name>_bound_lo/hi``,
    ``CV_<name>_target``.  ``suffix`` is appended to every column when
    a baseline is being merged into the agent's CSV.
    """
    T = ep['episode_length']
    states = ep['states']
    controls = ep['controls']
    cv_idx = ep.get('cv_indices') or []
    mv_idx = ep.get('mv_indices') or []
    dv_idx = ep.get('dv_indices') or []
    names = ep.get('state_variables') or []
    def _arr(key, dtype='float32'):
        v = ep.get(key)
        if v is None or len(v) == 0:
            return np.zeros((0,), dtype=dtype)
        return np.asarray(v, dtype=dtype)
    cur_mv_b = _arr('current_mv_bounds_t')
    cur_cv_b = _arr('current_cv_bounds_t')
    cur_cv_t = _arr('current_cv_targets_t')

    def _nm(i: int, default: str) -> str:
        s = names[i] if 0 <= i < len(names) and names[i] else default
        return str(s).replace('/', '_').replace(' ', '_')

    out: Dict[str, np.ndarray] = {f'time_step{suffix}': np.arange(T, dtype='int64')}
    for k, i in enumerate(mv_idx):
        nm = _nm(i, f'MV{i}')
        if k < controls.shape[1]:
            out[f'MV_{nm}{suffix}'] = controls[:T, k]
        if cur_mv_b.ndim == 3 and k < cur_mv_b.shape[1]:
            out[f'MV_{nm}_bound_lo{suffix}'] = cur_mv_b[:T, k, 0]
            out[f'MV_{nm}_bound_hi{suffix}'] = cur_mv_b[:T, k, 1]
    for i in dv_idx:
        if i < states.shape[1]:
            out[f'DV_{_nm(i, f"DV{i}")}{suffix}'] = states[:T, i]
    for k, i in enumerate(cv_idx):
        if i < states.shape[1]:
            nm = _nm(i, f'CV{i}')
            out[f'CV_{nm}{suffix}'] = states[:T, i]
            if cur_cv_b.ndim == 3 and k < cur_cv_b.shape[1]:
                out[f'CV_{nm}_bound_lo{suffix}'] = cur_cv_b[:T, k, 0]
                out[f'CV_{nm}_bound_hi{suffix}'] = cur_cv_b[:T, k, 1]
            tgt = _cv_active_target(ep, k)
            if tgt is not None:
                out[f'CV_{nm}_target{suffix}'] = tgt[:T]
    out[f'raw_reward{suffix}'] = ep.get('raw_rewards', np.zeros(T, dtype='float32'))
    out[f'cv_violation_penalty{suffix}'] = ep.get('cv_violations', np.zeros(T, dtype='float32'))
    out[f'mv_violation_penalty{suffix}'] = ep.get('mv_violations', np.zeros(T, dtype='float32'))
    out[f'cum_raw_reward{suffix}'] = np.cumsum(out[f'raw_reward{suffix}']).astype('float64')
    return out


def write_episode_csv(ep_agent: Dict, ep_baseline: Dict | None,
                       csv_path: Path) -> None:
    """Write a single CSV with agent columns + (optional) baseline columns.

    Falls back to plain numpy.savetxt when pandas is unavailable.
    """
    cols = _episode_to_dataframe_dict(ep_agent)
    if ep_baseline is not None and ep_baseline.get('episode_length', 0) > 0:
        cols.update(_episode_to_dataframe_dict(ep_baseline, suffix='_baseline'))
    try:
        import pandas as pd
        T = max(int(v.shape[0]) for v in cols.values())
        norm = {k: (np.pad(v, (0, T - v.shape[0]),
                            constant_values=np.nan).astype('float64')
                     if v.shape[0] < T else v.astype('float64'))
                for k, v in cols.items()}
        pd.DataFrame(norm).to_csv(csv_path, index=False, float_format='%.6g')
    except Exception:
        # Fallback: header + savetxt
        header = ','.join(cols.keys())
        T = max(int(v.shape[0]) for v in cols.values())
        rows = np.full((T, len(cols)), np.nan, dtype='float64')
        for j, (_, v) in enumerate(cols.items()):
            rows[:v.shape[0], j] = v.astype('float64')
        np.savetxt(csv_path, rows, header=header, comments='',
                    delimiter=',', fmt='%.6g')


def write_schedule_txt(schedule: List[Dict], out_path: Path) -> None:
    """V3-style human-readable disturbance-schedule listing."""
    lines = ['Disturbance schedule (validation, holdout_a profile):']
    for i, ev in enumerate(schedule, start=1):
        lines.append(
            f"{i:2d}. t={int(ev.get('start', 0)):5d}  "
            f"{str(ev.get('source', '?')):<14s}  "
            f"target={str(ev.get('target_name', '?')):<24s}  "
            f"intent={str(ev.get('intent', '?')):<10s}  "
            f"delta={float(ev.get('delta', 0.0)):+.4g}"
        )
    out_path.write_text('\n'.join(lines) + '\n')


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


def plot_disturbance_rejection(ep: Dict, out_path: Path, title: str = '',
                                  *, ep_baseline: Dict | None = None,
                                  event_metrics: Dict | None = None) -> None:
    """Plant-style disturbance-rejection plot.

    Mirrors ``plot_channels`` from ``neural-apc-pytorch/evaluation/
    validate_latent.py`` so the operator sees a real plant view:
      - one row per channel (MV, DV, CV) with semantic state names,
      - CV bound band shaded orange + red dashed low/high lines,
      - MV red dashed low/high lines + gray dotted normalization range,
      - dark-green dash-dot CV target setpoint,
      - vertical disturbance markers with ▲/▼ direction arrow at top,
      - **constant-MV baseline overlay** (dashed grey) on every MV/CV row
        when ``ep_baseline`` is provided — makes "is the agent doing
        anything?" answerable at a glance,
      - **CV tracking-error subplot** (replaces the legacy cum-reward
        subplot, which was dominated by violation-penalty steps and
        carried no operator information),
      - per-event response annotations (±overshoot, settle steps) when
        ``event_metrics`` is provided.
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
    def _arr_t(key, dtype='float32'):
        v = ep.get(key)
        if v is None or len(v) == 0:
            return np.zeros((0,), dtype=dtype)
        return np.asarray(v, dtype=dtype)
    cur_mv_b_t = _arr_t('current_mv_bounds_t')
    cur_cv_b_t = _arr_t('current_cv_bounds_t')
    cur_cv_tgt_t = _arr_t('current_cv_targets_t')
    hidden_dist_t = _arr_t('hidden_disturbance_t')
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
                         'cv_ord': k,
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

        # Baseline overlay (constant-MV episode under same schedule).
        # The baseline is the same for every seed and channel: it shows
        # what the plant does with no control, so the operator can see
        # at a glance how much the agent recovered the deviation.
        if ep_baseline is not None:
            base_series = None
            grp = ch.get('group')
            # Direct alignment: channels were built in MV/DV/CV order
            # from ep itself, so the baseline rollout (built the same
            # way) lines up positionally.
            mv_idx_a = ep.get('mv_indices') or []
            dv_idx_a = ep.get('dv_indices') or []
            cv_idx_a = ep.get('cv_indices') or []
            if grp == 'mv' and r < len(mv_idx_a):
                k = r
                if k < ep_baseline['controls'].shape[1]:
                    base_series = ep_baseline['controls'][:, k]
            elif grp == 'dv':
                # DV row index in channels = len(mv) + (offset within DVs)
                k = r - len(mv_idx_a)
                if 0 <= k < len(dv_idx_a):
                    cidx = dv_idx_a[k]
                    if cidx < ep_baseline['states'].shape[1]:
                        base_series = ep_baseline['states'][:, cidx]
            elif grp == 'cv':
                k = r - len(mv_idx_a) - len(dv_idx_a)
                if 0 <= k < len(cv_idx_a):
                    cidx = cv_idx_a[k]
                    if cidx < ep_baseline['states'].shape[1]:
                        base_series = ep_baseline['states'][:, cidx]
            if base_series is not None and len(base_series) > 0:
                m = min(len(base_series), len(t_arr))
                ax.plot(t_arr[:m], base_series[:m], color='#888888',
                         lw=1.0, ls='--', alpha=0.85,
                         label='baseline (no control)')

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

        # Unmeasured (hidden OU) disturbance overlay — only on CV rows.
        # The disturbance is added to the CV state and is invisible to the
        # agent/WM; plotting ``ref + offset`` lets the operator see the
        # magnitude/shape of what the controller had to reject.  ``ref`` is
        # the active target if enabled, else the active-band centre, else
        # the series mean, so the dotted trace sits on the CV's own scale.
        cv_ord = ch.get('cv_ord')
        if (ch.get('group') == 'cv' and cv_ord is not None
                and hidden_dist_t.ndim == 2
                and cv_ord < hidden_dist_t.shape[1]
                and np.any(np.abs(hidden_dist_t[:, cv_ord]) > 1e-9)):
            hcol = np.asarray(hidden_dist_t[:, cv_ord], dtype='float32')
            nH = min(hcol.shape[0], len(t_arr))
            # Reference level the offset is drawn relative to.
            if (target_t is not None and isinstance(target_t, np.ndarray)
                    and target_t.ndim == 1 and target_t.size >= 1
                    and np.any(np.isfinite(target_t))):
                ref = np.asarray(target_t[:nH], dtype='float32')
            elif target is not None and np.isfinite(target):
                ref = np.full(nH, float(target), dtype='float32')
            elif (bounds is not None and len(bounds) >= 2
                    and np.isfinite(bounds[0]) and np.isfinite(bounds[1])):
                ref = np.full(nH, 0.5 * (float(bounds[0]) + float(bounds[1])),
                              dtype='float32')
            else:
                fin = series[np.isfinite(series)] if isinstance(series, np.ndarray) else None
                ref = np.full(nH, float(np.mean(fin)) if fin is not None and fin.size else 0.0,
                              dtype='float32')
            ax.plot(t_arr[:nH], ref + hcol[:nH], color='#ff7f0e',
                     lw=1.1, ls=':', alpha=0.85,
                     label='unmeasured disturbance (hidden)')


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
    # CV tracking-error trace.  Replaces cum-reward (which was
    # dominated by the violation-penalty step and carried no actionable
    # operator information).  For each CV with a target enabled we plot
    # |CV_t - target_t| / bound_width on the left axis, and the running
    # IAE (sum of normalised |error|) on a twin axis on the right.
    cv_idx_local = ep.get('cv_indices') or []
    cv_bounds_local = ep.get('cv_bounds') or []
    states_local = ep['states']
    plotted_any_err = False
    cum_iae = np.zeros_like(t_arr, dtype='float64')
    for k, cidx in enumerate(cv_idx_local):
        if cidx >= states_local.shape[1]:
            continue
        tgt = _cv_active_target(ep, k)
        if tgt is None:
            continue
        b = cv_bounds_local[k] if k < len(cv_bounds_local) else None
        if (isinstance(b, list) and len(b) >= 2 and np.isfinite(b[0])
                and np.isfinite(b[1]) and b[1] > b[0]):
            denom = float(b[1]) - float(b[0])
        else:
            denom = float(np.nanstd(states_local[:T, cidx])) or 1.0
        n = min(T, tgt.shape[0])
        err = (states_local[:n, cidx].astype('float64')
               - tgt[:n].astype('float64')) / max(1e-9, denom)
        ax.plot(t_arr[:n], np.abs(err), lw=1.0, label=f'|err| CV[{cidx}]')
        cum_iae[:n] += np.cumsum(np.abs(err))
        plotted_any_err = True
        if ep_baseline is not None:
            sb = ep_baseline['states']
            tb = _cv_active_target(ep_baseline, k)
            if tb is not None and cidx < sb.shape[1]:
                m = min(sb.shape[0], tb.shape[0], len(t_arr))
                err_b = (sb[:m, cidx].astype('float64')
                          - tb[:m].astype('float64')) / max(1e-9, denom)
                ax.plot(t_arr[:m], np.abs(err_b), lw=0.9, ls='--',
                         color='#888888', alpha=0.8,
                         label=f'|err| CV[{cidx}] baseline')
    ax.axhline(0.0, color='gray', lw=0.5, ls='-', alpha=0.5)
    if plotted_any_err:
        ax2 = ax.twinx()
        ax2.plot(t_arr, cum_iae, color='#1f77b4', lw=1.2,
                  label=f'cum IAE (final={cum_iae[-1]:.2f})')
        if ep_baseline is not None:
            cum_iae_b = np.zeros(t_arr.shape[0], dtype='float64')
            sb = ep_baseline['states']
            for k, cidx in enumerate(cv_idx_local):
                if cidx >= sb.shape[1]:
                    continue
                tb = _cv_active_target(ep_baseline, k)
                if tb is None:
                    continue
                b = cv_bounds_local[k] if k < len(cv_bounds_local) else None
                denom = ((float(b[1]) - float(b[0]))
                           if (isinstance(b, list) and len(b) >= 2
                                and np.isfinite(b[0]) and np.isfinite(b[1])
                                and b[1] > b[0])
                           else (float(np.nanstd(sb[:, cidx])) or 1.0))
                m = min(sb.shape[0], tb.shape[0], len(t_arr))
                err = np.abs((sb[:m, cidx].astype('float64')
                                - tb[:m].astype('float64'))
                               / max(1e-9, denom))
                cum_iae_b[:m] += np.cumsum(err)
            ax2.plot(t_arr, cum_iae_b, color='#888888', lw=1.0, ls='--',
                      label=f'cum IAE baseline (final={cum_iae_b[-1]:.2f})')
        ax2.set_ylabel('cum IAE (normed)', color='#1f77b4')
        ax2.tick_params(axis='y', labelcolor='#1f77b4')
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, loc='upper left', fontsize=8)
    else:
        # Fallback when no CV target is enabled — keep the cum-reward
        # trace so the subplot is still informative.
        ax.plot(t_arr, np.cumsum(ep['raw_rewards']), color='C2', lw=1.0,
                 label=f"raw cum (final={ep['cum_raw_reward']:+.1f})")
        ax.legend(loc='upper left', fontsize=8)
    _draw_disturbance_markers(ax)
    ax.set_ylabel('|err| (normed)')
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

    # Ensure validation uses the same noise config the model was trained
    # against.  ``single_run`` writes ``noise_config.json`` to the
    # controller dir and exports ``SIM_NOISE_CONFIG_JSON`` for the
    # in-process train+validate flow, but standalone ``python -m
    # evaluation.validate`` invocations (and the workflow runner) start
    # without that env var set.  Falling back to "no noise" silently
    # produces flat baseline traces that don't match the training
    # distribution (diagnosed 2026-05-06).  Load the file explicitly.
    if not os.environ.get('SIM_NOISE_CONFIG_JSON', '').strip():
        nc_path = controller_dir / 'noise_config.json'
        if nc_path.exists():
            os.environ['SIM_NOISE_CONFIG_JSON'] = str(nc_path.resolve())
            print(f'[val] noise_config: loaded {nc_path} '
                  '(SIM_NOISE_CONFIG_JSON was unset)', flush=True)

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
        world_model_type=str(getattr(cfg, 'world_model_type', 'sf_transformer')),
        rssm_deter_dim=int(getattr(cfg, 'rssm_deter_dim', 512)),
        rssm_n_categoricals=int(getattr(cfg, 'rssm_n_categoricals', 32)),
        rssm_n_classes=int(getattr(cfg, 'rssm_n_classes', 32)),
        rssm_embed_dim=int(getattr(cfg, 'rssm_embed_dim', 256)),
        rssm_hidden_dim=int(getattr(cfg, 'rssm_hidden_dim', 256)),
        rssm_unimix=float(getattr(cfg, 'rssm_unimix', 0.01)),
        disturbance_head_dim=int(getattr(cfg, 'disturbance_head_dim', 0) or 0),
        disturbance_head_hidden=int(getattr(cfg, 'disturbance_head_hidden', 0) or 0),
        disturbance_head_layers=int(getattr(cfg, 'disturbance_head_layers', 2) or 2),
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
    # Seed plan: the standard held-out seeds at the curriculum-suppressed
    # hidden-disturbance amplitude (phase=1/progress=0, ~10% of nominal —
    # keeps these comparable across runs), PLUS one dedicated seed at the
    # FULL phase-3 unmeasured-disturbance magnitude so the operator can see
    # how the agent rejects a realistic unmeasured load.  Gated by
    # ``DREAMER_VAL_UNMEASURED_SEED`` (default ON).
    seed_plan: List[Tuple[int, bool]] = [(10_000 + s, False)
                                          for s in range(int(seeds))]
    if str(os.environ.get('DREAMER_VAL_UNMEASURED_SEED', '1')).strip().lower() \
            not in ('0', 'false', 'no', 'off'):
        seed_plan.append((10_000 + int(seeds), True))
    for seed, unmeasured_full in seed_plan:
        rng = np.random.default_rng(seed)
        env = APCEnv(cfg, rng)
        # Force the hidden OU disturbance on every validation episode so
        # the agent is always tested on its disturbance-rejection skill.
        env._hidden_disturbance_force = True
        if unmeasured_full:
            # Lift the curriculum so the hidden OU runs at full phase-3
            # amplitude (curriculum_amp_scale cap 1.0, ramp at progress=1).
            env._current_phase = 3
            env._training_progress = 1.0
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

        per_seed_dir = out_dir / (f'seed_{seed:05d}_unmeasured'
                                   if unmeasured_full else f'seed_{seed:05d}')
        per_seed_dir.mkdir(parents=True, exist_ok=True)
        _ttl_sfx = '  [FULL unmeasured disturbance]' if unmeasured_full else ''

        eps = []
        for e in range(int(episodes)):
            ep = run_episode(env, model, device, deterministic=deterministic)
            title = (f'seed={seed} ep={e}  T={ep["episode_length"]}  '
                     f'cum_raw={ep["cum_raw_reward"]:+.2f}  '
                     f'mean_cv_v={ep["mean_cv_violation"]:.4f}  '
                     f'mean_mv_v={ep["mean_mv_violation"]:.4f}{_ttl_sfx}')
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
                # V3-parity per-episode KPIs.
                **{f'kpi_{k}': v for k, v in
                    compute_episode_metrics(ep).items()
                    if not isinstance(v, list)},
            })

        # ---- Disturbance-rejection plot (plant-aware holdout_a profile) ----
        try:
            scripted = build_scripted_disturbance_schedule(env, seed=seed)
            ep_d = run_scripted_episode(env, model, device,
                                         deterministic=deterministic,
                                         schedule=scripted)
            # Constant-MV baseline replay under the same scripted
            # schedule.  Same env (so noise / setpoint / objective are
            # identical), same seed offset, frozen action at the bound
            # midpoint.  Makes "is the agent doing anything?" a one-glance
            # answer on the rejection plot.
            try:
                # Reset per-event bookkeeping flags so the baseline run
                # actually re-applies the disturbance schedule.  The
                # agent run mutates ``_applied`` / ``_hold_until`` /
                # ``_hold_state_idx`` / ``_hold_value_raw`` in-place;
                # without this reset the baseline sees zero disturbances
                # → flat DV → CV at steady state with only measurement
                # noise → "is the agent doing anything?" plot shows a
                # near-flat baseline that misleads the operator (mirrors
                # the legacy V3 fix at validate_latent.py:2299).
                for ev in scripted:
                    ev['_applied'] = False
                    ev.pop('_hold_until', None)
                    ev.pop('_hold_state_idx', None)
                    ev.pop('_hold_value_raw', None)
                env_b = APCEnv(cfg, np.random.default_rng(seed + 1))
                env_b._hidden_disturbance_force = True
                if unmeasured_full:
                    env_b._current_phase = 3
                    env_b._training_progress = 1.0
                if obs_norm_state is not None:
                    try:
                        env_b.set_obs_norm_stats(
                            mean=np.asarray(obs_norm_state.get('mean')),
                            var=np.asarray(obs_norm_state.get('var')),
                            count=float(obs_norm_state.get('count', 1.0)),
                            learn=False)
                    except Exception:
                        pass
                env_b.reward_scale = env.reward_scale
                # DV is by definition an external disturbance the agent
                # can't influence.  Force the baseline simulator to
                # follow the *agent's* recorded DV trajectory so the
                # baseline-vs-agent CV overlay isolates "what would
                # happen with no control under the same external
                # disturbance" — the only meaningful comparison.
                # Without the override, the baseline's noise wrapper
                # produces an independent OU+measurement-noise stream
                # on the DV channel, so the two runs see different
                # external disturbances and the overlay is misleading.
                _dv_idx = list(env.meta.get('dv_indices') or [])
                _agent_dv = (ep_d['states'][:, _dv_idx]
                              if (_dv_idx and ep_d['states'].size)
                              else None)
                ep_b = run_constant_mv_episode(env_b, schedule=scripted,
                                                  mv_norm=0.0,
                                                  dv_override=_agent_dv)
            except Exception as _be:
                print(f'[val] baseline replay skipped (seed {seed}): {_be!r}',
                      flush=True)
                ep_b = None

            # Per-event response metrics + episode-level KPIs (V3 parity).
            ep_metrics = compute_episode_metrics(ep_d)
            ev_metrics = compute_event_response_metrics(ep_d)
            base_metrics = (compute_episode_metrics(ep_b)
                              if ep_b is not None else None)

            d_title = (f'seed={seed}  scripted disturbance rejection  '
                       f'cum_raw={ep_d["cum_raw_reward"]:+.2f}  '
                       f'IAE={ep_metrics["iae_normed_mean"]:.2f}  '
                       f'overshoot_max={ev_metrics["overshoot_normed"]["max"]:.3f}'
                       + (f'   |  baseline IAE={base_metrics["iae_normed_mean"]:.2f}'
                          if base_metrics is not None else '')
                       + _ttl_sfx)
            ann = plot_disturbance_rejection(
                ep_d, per_seed_dir / 'disturbance_rejection.png',
                title=d_title, ep_baseline=ep_b, event_metrics=ev_metrics)

            # Operator-friendly artefacts: schedule.txt + CSV (with
            # baseline columns merged in).  Both are pure post-processing
            # — they read the rollout dicts and write text files.
            try:
                write_schedule_txt(scripted,
                                    per_seed_dir / 'disturbance_schedule.txt')
            except Exception as _se:
                print(f'[val] schedule.txt skipped (seed {seed}): {_se!r}',
                      flush=True)
            try:
                write_episode_csv(ep_d, ep_b,
                                   per_seed_dir / 'disturbance_rollout.csv')
            except Exception as _ce:
                print(f'[val] rollout.csv skipped (seed {seed}): {_ce!r}',
                      flush=True)
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
                    hidden_disturbance_t=np.asarray(
                        ep_d.get('hidden_disturbance_t',
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
                # V3-parity episode KPIs (mv_tv, mv_activity_ratio,
                # mv_bound_hugging_score, mv_bound_usage,
                # mv_reversal_rate, economic_score, IAE/ITAE/ISE).
                'episode_metrics_agent': ep_metrics,
                'episode_metrics_baseline': base_metrics,
                # Per-event overshoot / settle / IAE_window summary.
                'event_response': ev_metrics,
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
                        f'({len(seed_results)} seeds × {episodes} eps)')

    # ---- Training-stage + WM-fidelity diagnostics ------------------------
    # Run once per validation invocation on a fresh env so we don't pay
    # per-seed cost.  Tells the operator which training stage (P1 WM,
    # P2 reward MTP / BC, P3 actor-critic) is the bottleneck.
    try:
        from evaluation.diagnostics import compute_training_diagnostics
        diag_env = APCEnv(cfg, np.random.default_rng(99_999))
        # WM-fidelity probe: disable hidden OU so the WM is scored on
        # base-plant dynamics, not augmented-system dynamics.
        diag_env._disturbance_prob_override = 0.0
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

        # Internal-fidelity gates (2026-05-06).  Independent of the
        # economic / disturbance-rejection plots, these flag whether the
        # *world model itself* is usable.  Thresholds chosen from the
        # validate-iter140 RCA: a healthy WM should yield at least
        # weak positive correlation between predictions and real
        # next-state / next-reward / Monte-Carlo return.  Anything
        # below these floors means downstream actor learning is
        # mathematically guaranteed to fail.
        try:
            wm = diag.get('wm_fidelity', {}) or {}
            rw = diag.get('reward_fidelity', {}) or {}
            cc = diag.get('critic_calib', {}) or {}
            wm_r1 = float(((wm.get('per_offset') or {}).get('1') or {}).get('r_mean', 0.0))
            rw_r0 = float(((rw.get('per_offset') or {}).get('0') or {}).get('r', 0.0))
            critic_r = float(cc.get('r_pearson', 0.0))
            fidelity_gates = {
                'wm_next_state_r_min': 0.5,
                'reward_head_r_min': 0.3,
                'critic_r_min': 0.3,
                'wm_next_state_r_observed': wm_r1,
                'reward_head_r_observed': rw_r0,
                'critic_r_observed': critic_r,
                'wm_pass': bool(wm_r1 >= 0.5),
                'reward_pass': bool(rw_r0 >= 0.3),
                'critic_pass': bool(critic_r >= 0.3),
            }
            fidelity_gates['all_pass'] = bool(
                fidelity_gates['wm_pass']
                and fidelity_gates['reward_pass']
                and fidelity_gates['critic_pass']
            )
            diag['fidelity_gates'] = fidelity_gates
            if not fidelity_gates['all_pass']:
                print('[val] internal-fidelity gates FAILED:', flush=True)
                if not fidelity_gates['wm_pass']:
                    print(f'        - WM next-state r={wm_r1:+.3f} < 0.5'
                          ' (encoder/dynamics not learning plant)', flush=True)
                if not fidelity_gates['reward_pass']:
                    print(f'        - reward head r={rw_r0:+.3f} < 0.3'
                          ' (reward MTP uncorrelated with truth)', flush=True)
                if not fidelity_gates['critic_pass']:
                    print(f'        - critic V vs MC r={critic_r:+.3f} < 0.3'
                          ' (value head uncorrelated with returns)', flush=True)
            else:
                print(f'[val] internal-fidelity gates PASSED '
                      f'(wm_r={wm_r1:+.3f} rw_r={rw_r0:+.3f} '
                      f'critic_r={critic_r:+.3f})', flush=True)
        except Exception as _ge:
            print(f'[val] fidelity-gate computation skipped: {_ge!r}',
                  flush=True)
            diag['fidelity_gates'] = {'error': repr(_ge)}
    except Exception as e:
        print(f'[val] diagnostics skipped: {e}', flush=True)

    # ---- WM transfer-function (step-response) matrix ---------------------
    # DMC-style per-MV/CV step-response curves (WM vs real sim) averaged over
    # the operating region with a min/max variation band.  Directly measures
    # whether the world model captured the true GAINS + DYNAMICS (the
    # correlation-based fidelity probe does NOT).  Gated ON by default; skip
    # with DREAMER_VAL_WM_TRANSFER=0.
    if os.environ.get('DREAMER_VAL_WM_TRANSFER', '1').strip() not in ('0', 'false', 'False'):
        try:
            from evaluation.wm_transfer_matrix import compute_and_plot
            tf_env = APCEnv(cfg, np.random.default_rng(77_777))
            tf_env._disturbance_prob_override = 0.0
            tf_obs_std = None
            if obs_norm_state is not None:
                try:
                    _var = np.asarray(obs_norm_state.get('var'), dtype='float32')
                    tf_obs_std = np.clip(np.sqrt(np.maximum(_var, 1e-6)), 1e-3, None)
                    tf_env.set_obs_norm_stats(
                        mean=np.asarray(obs_norm_state.get('mean')), var=_var,
                        count=float(obs_norm_state.get('count', 1.0)),
                        learn=False)
                except Exception:
                    pass
            tf_result = compute_and_plot(
                model, tf_env, cfg, device, out_dir, obs_std=tf_obs_std,
                title=f'{controller_dir.name}  WM transfer matrix')
            # GAIN-FIDELITY GATE (control-relevant; the correlation-based
            # wm_next_state_r does NOT measure gain).  Mean relative SS-gain
            # error across MV/CV pairs; a WM usable for control needs the gain
            # within ~2× of the real plant (rel_err < 1.0; healthy < 0.35).
            try:
                pairs = (tf_result or {}).get('pairs', {}) if tf_result else {}
                rel_errs = []
                for v in pairs.values():
                    rg = abs(float(v.get('real_ss_gain', 0.0)))
                    if rg > 1e-6:
                        rel_errs.append(abs(float(v.get('ss_gain_abs_err', 0.0))) / rg)
                if rel_errs:
                    gain_rel_err = float(np.mean(rel_errs))
                    gate = {
                        'wm_gain_rel_err': gain_rel_err,
                        'wm_gain_rel_err_max': float(np.max(rel_errs)),
                        'wm_gain_pass': bool(gain_rel_err < 1.0),
                        'wm_gain_healthy': bool(gain_rel_err < 0.35),
                        'n_pairs': len(rel_errs),
                    }
                    if isinstance(locals().get('fidelity_gates'), dict):
                        fidelity_gates.update(gate)
                    else:
                        fidelity_gates = gate
                    status = ('HEALTHY' if gate['wm_gain_healthy']
                              else ('PASS' if gate['wm_gain_pass'] else 'FAIL'))
                    print(f'[val] WM gain fidelity: rel_err={gain_rel_err:.2f} '
                          f'({status}; correlation gates can pass while this '
                          f'fails — gain is the control-relevant metric)',
                          flush=True)
            except Exception as _ge:
                print(f'[val] WM gain-gate skipped: {_ge!r}', flush=True)
        except Exception as e:
            print(f'[val] WM transfer matrix skipped: {e!r}', flush=True)

    cum = np.array([m['cum_raw_reward'] for m in metrics_records])
    cv_v = np.array([m['mean_cv_violation'] for m in metrics_records])
    mv_v = np.array([m['mean_mv_violation'] for m in metrics_records])
    summary = {
        'controller_dir': str(controller_dir),
        'simulation_dir': str(sim_dir),
        'ckpt': str(ckpt_path),
        'deterministic': deterministic,
        'policy_type': str(getattr(cfg, 'policy_type', 'continuous')),
        'actor_loss_type': str(getattr(cfg, 'actor_loss_type', 'reinforce')),
        'policy_init_log_std': float(getattr(cfg, 'policy_init_log_std', -0.5)),
        'n_seeds': int(seeds),
        'episodes_per_seed': int(episodes),
        'n_episodes_total': int(len(metrics_records)),
        'cum_raw_reward_mean': float(cum.mean()),
        'cum_raw_reward_std': float(cum.std()),
        'cum_raw_reward_min': float(cum.min()),
        'cum_raw_reward_max': float(cum.max()),
        'mean_cv_violation_mean': float(cv_v.mean()),
        'mean_mv_violation_mean': float(mv_v.mean()),
        'fidelity_gates': locals().get('fidelity_gates', None),
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
