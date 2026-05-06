"""Training-stage diagnostics for DreamerV4 controllers.

Aggregates four families of metrics into a single ``wm_diagnostics.json``
+ ``wm_diagnostics.png`` per validation run so the operator can tell at
a glance *which* training stage failed when a controller scores poorly:

  * **stage_metrics**   parsed from ``train_log.jsonl`` (per-phase).
                        Shows P1 (recon/sf) convergence, P2 (reward MTP +
                        BC) convergence, and P3 (actor / critic / entropy
                        / imagined return) progression.
                        Failure modes detected:
                          - WM not learning: P1 ``sf_loss`` does not drop.
                          - Reward head not learning: P2 ``reward_mtp_loss``
                            stays at the random baseline (≈ log(K)).
                          - Policy collapse: P3 ``ent`` stays at log(n_bins).
                          - Optimization unstable: ``n_grad_skip`` > 0.

  * **wm_fidelity**     k-step open-loop rollout of the world model
                        against the real plant on a held-out episode.
                        We feed the *real* action sequence to the
                        dynamics transformer and compare predicted obs
                        against the real obs at each future offset
                        (per-channel MAE + Pearson r at offsets 1,
                        H/4, H/2, H).  When the WM is broken (e.g. the
                        tokenizer collapsed) the open-loop trajectory
                        diverges from the plant within a few steps.

  * **reward_fidelity** Predicted reward (reward MTP head) vs realized
                        reward at each offset 0..L-1.  Reports per-offset
                        Pearson r + MAE.  When the head is action-blind
                        the correlation stays near 0 even for offset 0
                        (current-step prediction).

  * **critic_calib**    Predicted V(s_t) vs realized return-to-go on
                        the held-out episode (Pearson r + MAE).  When
                        critic regression failed, r ≈ 0 even though
                        ``critic_loss`` looked low (it can converge to
                        the marginal mean and still have r ≈ 0).

  * **policy_dist**     Action histogram + entropy over the held-out
                        rollout.  Catches deterministic-argmax collapse
                        and uniform-prior-pinning at log(n_bins).

The ``compute_training_diagnostics`` function below returns a dict and
optionally writes the JSON / PNG.  Designed to be called from
``evaluation.validate.run_validation`` after the per-seed loop.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Stage metrics from train_log.jsonl
# ---------------------------------------------------------------------------

def _parse_train_log(jsonl_path: Path,
                       policy_type: str = 'continuous',
                       n_action_bins: int = 21,
                       actor_loss_type: str = 'reinforce',
                       ) -> Tuple[List[Dict], Dict]:
    """Parse ``train_log.jsonl`` and bucket rows by phase.

    Returns ``(rows, summary)`` where ``rows`` is the parsed list and
    ``summary`` is a dict ``{ 'p1': {...}, 'p2': {...}, 'p3': {...} }``
    with first/last/min/max of each loss-like column per phase.
    Flags adapt to ``policy_type`` (continuous / discrete) and
    ``actor_loss_type`` (reinforce / pmpo) so labels match the
    actually-trained policy.
    """
    rows: List[Dict] = []
    if not jsonl_path.exists():
        return rows, {'error': f'{jsonl_path.name} not found'}
    with open(jsonl_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    if not rows:
        return rows, {'error': 'train_log.jsonl is empty'}

    by_phase: Dict[int, List[Dict]] = {1: [], 2: [], 3: []}
    for r in rows:
        ph = int(r.get('phase', 0) or 0)
        if ph in by_phase:
            by_phase[ph].append(r)

    def _stat(rs: List[Dict], key: str) -> Dict[str, float]:
        vals = [float(r.get(key, 0.0) or 0.0) for r in rs
                if r.get(key) is not None]
        if not vals:
            return {}
        a = np.asarray(vals, dtype='float64')
        return {
            'first': float(a[0]),
            'last': float(a[-1]),
            'min': float(np.min(a)),
            'max': float(np.max(a)),
            'median': float(np.median(a)),
        }

    summary = {'n_iters_total': len(rows),
                'policy_type': str(policy_type),
                'actor_loss_type': str(actor_loss_type),
                'p1': {}, 'p2': {}, 'p3': {}}
    keys = ['recon_loss', 'sf_loss', 'reward_mtp_loss', 'bc_loss',
            'actor_loss', 'critic_loss', 'entropy_mean',
            'imagined_return_mean', 'imagined_reward_mean',
            # Both naming schemes are emitted by the trainer:
            # ``actor_*`` from ``reinforce_actor_loss`` and ``pmpo_*``
            # as back-compat aliases.  Keep both so old/new logs work.
            'actor_kl_pen', 'pmpo_kl', 'pmpo_pos_frac', 'n_grad_skip',
            'ema_return', 'return_window_mean']
    for ph_id, ph_key in ((1, 'p1'), (2, 'p2'), (3, 'p3')):
        rs = by_phase[ph_id]
        summary[ph_key]['n_iters'] = int(len(rs))
        for k in keys:
            s = _stat(rs, k)
            if s:
                summary[ph_key][k] = s

    # Health flags
    flags: List[str] = []
    p1 = summary['p1']
    if p1.get('n_iters', 0) > 0 and 'sf_loss' in p1:
        if p1['sf_loss']['last'] >= 0.95 * p1['sf_loss']['first']:
            flags.append('P1: shortcut-forcing loss did not drop '
                          '(WM not learning dynamics)')
    p2 = summary['p2']
    if p2.get('n_iters', 0) > 0 and 'reward_mtp_loss' in p2:
        # Random baseline is ~ log(twohot K) ~ log(255) ≈ 5.5 for the
        # default 255-bin head; use 0.85 of first as 'no progress'.
        if p2['reward_mtp_loss']['last'] >= 0.95 * p2['reward_mtp_loss']['first']:
            flags.append('P2: reward MTP loss did not drop '
                          '(reward head not learning)')
    p3 = summary['p3']
    if p3.get('n_iters', 0) > 0:
        if 'entropy_mean' in p3:
            ent_first = p3['entropy_mean']['first']
            ent_last = p3['entropy_mean']['last']
            if str(policy_type).lower() == 'discrete':
                # log(n_bins) is the uniform-entropy upper bound for
                # discrete policies; flag if we never moved off it.
                ent_uniform = float(np.log(max(2, int(n_action_bins))))
                if ent_last >= 0.97 * ent_uniform:
                    flags.append(f'P3: discrete policy entropy stayed near uniform '
                                  f'(last={ent_last:.3f} ~ log({n_action_bins})='
                                  f'{ent_uniform:.3f})')
            else:
                # Continuous TanhNormal: flag (a) collapse to near-zero
                # std and (b) failure to drop below init by end of P3.
                if ent_last <= ent_first - 1.5:
                    flags.append(f'P3: continuous policy entropy collapsed '
                                  f'(first={ent_first:.3f} → last={ent_last:.3f}; '
                                  f'std → 0)')
                elif ent_last >= ent_first + 0.1:
                    flags.append(f'P3: continuous policy entropy did not drop '
                                  f'(first={ent_first:.3f} → last={ent_last:.3f}; '
                                  f'no exploration narrowing)')
        if 'critic_loss' in p3:
            if p3['critic_loss']['last'] >= 0.85 * p3['critic_loss']['first']:
                flags.append('P3: critic loss did not drop')
        if 'n_grad_skip' in p3 and p3['n_grad_skip']['max'] > 0:
            flags.append(f'P3: {int(p3["n_grad_skip"]["max"])} grad-clip '
                          f'skips (NaN/Inf in actor or critic gradient)')
        # Advantage-sign skew: ``pmpo_pos_frac`` (alias under REINFORCE)
        # is the fraction of imagined transitions with adv >= 0.  Both
        # extremes indicate trouble: ~0 means critic baseline above all
        # returns (over-optimistic value), ~1 means below all returns.
        if 'pmpo_pos_frac' in p3:
            pf_last = p3['pmpo_pos_frac']['last']
            pf_med = p3['pmpo_pos_frac'].get('median', pf_last)
            if pf_med <= 0.1:
                flags.append(f'P3: advantage-positive fraction near zero '
                              f'(median={pf_med:.3f}); critic baseline '
                              f'over-optimistic vs imagined returns')
            elif pf_med >= 0.9:
                flags.append(f'P3: advantage-positive fraction near one '
                              f'(median={pf_med:.3f}); critic baseline '
                              f'under-pessimistic vs imagined returns')
    summary['flags'] = flags
    return rows, summary


# ---------------------------------------------------------------------------
# World-model k-step rollout fidelity
# ---------------------------------------------------------------------------

def _wm_kstep_rollout(model, env, device, *, k_max: int = 32,
                       n_starts: int = 16, seed: int = 12345) -> Dict:
    """Open-loop k-step rollout of the world model on a held-out episode.

    Strategy: collect one full real episode under a *random* policy so
    the action sequence is informative (not constant), then for each of
    ``n_starts`` start indices ``t``:
      1. Encode the lookback window ending at ``t`` through the
         tokenizer, get ``z_ctx`` of length L.
      2. Step the dynamics transformer forward ``k`` times appending the
         *real* action ``a_{t}, ..., a_{t+k-1}`` to the action history,
         decoding ``z_{t+1}, ..., z_{t+k}`` to predicted obs each step.
      3. Compare predicted obs to real obs at offsets 1, k/4, k/2, k.

    Returns per-offset MAE + Pearson r averaged over channels and starts.
    """
    cfg = env.cfg
    L = cfg.lookback
    obs_dim = env.obs_dim
    action_dim = env.action_dim
    T = cfg.episode_length
    rng = np.random.default_rng(int(seed))

    # 1. Collect a real random-action episode.
    ow = env.reset(exploration=False)
    real_obs = np.zeros((T, obs_dim), dtype='float32')
    real_act = np.zeros((T, action_dim), dtype='float32')
    for t in range(T):
        a = rng.uniform(-1.0, 1.0, size=(action_dim,)).astype('float32')
        ow_next, _, done, _ = env.step(a)
        real_obs[t] = ow_next[-1]      # last row = freshest obs
        real_act[t] = a
        if done:
            T = t + 1
            real_obs = real_obs[:T]
            real_act = real_act[:T]
            break

    if T < L + k_max + 4:
        return {'error': f'episode too short for k-step rollout '
                          f'(T={T}, need >= L+k+4 = {L + k_max + 4})'}

    # 2. Build start indices uniformly from [L, T - k_max - 1].
    starts = rng.integers(L, T - k_max - 1, size=int(n_starts))

    # 3. For each start, do a closed-loop encode + open-loop dynamics roll.
    d_min = 1.0 / cfg.k_max
    tau_ctx_val = 1.0 - cfg.tau_ctx
    pred_obs_all: List[np.ndarray] = []      # per start: (k, obs_dim)
    real_obs_all: List[np.ndarray] = []
    for s in starts:
        s = int(s)
        # Initial lookback windows.
        # Real obs at indices s-L .. s-1 (the lookback ending at t=s-1).
        ow_window = real_obs[s - L:s]                    # (L, D)
        a_window = real_act[s - L:s]                     # (L, A)
        pred_per_start = np.zeros((k_max, obs_dim), dtype='float32')
        for kk in range(k_max):
            with torch.no_grad():
                ow_t = torch.from_numpy(ow_window).to(device)
                a_t = torch.from_numpy(a_window).to(device)
                z_ctx = model.tokenizer.encode(ow_t).unsqueeze(0)
                tau = torch.full((1, L), tau_ctx_val, device=device,
                                  dtype=z_ctx.dtype)
                d = torch.full((1, L), d_min, device=device,
                                dtype=z_ctx.dtype)
                out = model.dynamics(z_ctx, tau, d, a_t.unsqueeze(0))
                # ``z1_hat`` is the per-step predicted clean next-z.
                # We take the last position which is the prediction for
                # ``z_{t+1}`` given the lookback ending at ``t``.
                z_next = out['z1_hat'][:, -1]                # (1, z)
                obs_hat = model.tokenizer.decode(
                    z_next).squeeze(0).float().cpu().numpy()
            pred_per_start[kk] = obs_hat[:obs_dim]

            # Slide windows: append predicted obs (open-loop) and real
            # action (we are conditioning on the *real* action sequence
            # so the WM is judged on its predictive capacity, not on
            # action-distribution mismatch).
            ow_window = np.concatenate(
                [ow_window[1:], obs_hat[None, :obs_dim]], axis=0)
            a_window = np.concatenate(
                [a_window[1:], real_act[s + kk:s + kk + 1]], axis=0)

        real_seg = real_obs[s:s + k_max]
        pred_obs_all.append(pred_per_start)
        real_obs_all.append(real_seg)

    pred = np.stack(pred_obs_all, axis=0)      # (n_starts, k_max, D)
    real = np.stack(real_obs_all, axis=0)

    def _per_offset_metrics(off: int) -> Dict[str, float]:
        p = pred[:, off - 1, :]
        r = real[:, off - 1, :]
        mae = float(np.mean(np.abs(p - r)))
        # Pearson r per channel, averaged over channels with non-zero
        # std on either side.
        rs: List[float] = []
        for c in range(p.shape[1]):
            if p[:, c].std() < 1e-8 or r[:, c].std() < 1e-8:
                continue
            rs.append(float(np.corrcoef(p[:, c], r[:, c])[0, 1]))
        return {
            'mae': mae,
            'r_mean': float(np.mean(rs)) if rs else float('nan'),
            'r_min': float(np.min(rs)) if rs else float('nan'),
            'n_active_channels': int(len(rs)),
        }

    offsets = sorted(set([1, max(1, k_max // 4), max(1, k_max // 2),
                          max(1, k_max)]))
    return {
        'k_max': int(k_max),
        'n_starts': int(n_starts),
        'episode_length_used': int(T),
        'per_offset': {str(off): _per_offset_metrics(off) for off in offsets},
        # Persist raw arrays (small) so plots can re-render without re-running.
        'pred_obs': pred.astype('float32').tolist(),
        'real_obs': real.astype('float32').tolist(),
    }


# ---------------------------------------------------------------------------
# Reward MTP head fidelity
# ---------------------------------------------------------------------------

def _reward_mtp_fidelity(model, env, device, *,
                          n_starts: int = 32,
                          seed: int = 23456) -> Dict:
    """Predict reward at offsets 0..L-1 vs realized reward."""
    cfg = env.cfg
    L = cfg.lookback
    T = cfg.episode_length
    obs_dim = env.obs_dim
    action_dim = env.action_dim
    rng = np.random.default_rng(int(seed))

    ow = env.reset(exploration=False)
    real_obs = np.zeros((T, obs_dim), dtype='float32')
    real_rew = np.zeros((T,), dtype='float32')
    real_act = np.zeros((T, action_dim), dtype='float32')
    for t in range(T):
        a = rng.uniform(-1.0, 1.0, size=(action_dim,)).astype('float32')
        ow_next, scaled_r, done, info = env.step(a)
        real_obs[t] = ow_next[-1]
        real_rew[t] = float(info.get('raw_reward', scaled_r))
        real_act[t] = a
        if done:
            T = t + 1
            break

    L_mtp = int(getattr(model.reward, 'mtp_length', 1))
    if T < L + L_mtp + 4:
        return {'error': f'episode too short ({T})'}

    starts = rng.integers(L, T - L_mtp - 1, size=int(n_starts))
    d_min = 1.0 / cfg.k_max
    tau_ctx_val = 1.0 - cfg.tau_ctx

    pred_rew = np.zeros((len(starts), L_mtp), dtype='float32')
    targ_rew = np.zeros((len(starts), L_mtp), dtype='float32')
    for i, s in enumerate(starts):
        s = int(s)
        ow_window = real_obs[s - L:s]
        a_window = real_act[s - L:s]
        with torch.no_grad():
            ow_t = torch.from_numpy(ow_window).to(device)
            a_t = torch.from_numpy(a_window).to(device)
            z_ctx = model.tokenizer.encode(ow_t).unsqueeze(0)
            tau = torch.full((1, L), tau_ctx_val, device=device,
                              dtype=z_ctx.dtype)
            d = torch.full((1, L), d_min, device=device,
                            dtype=z_ctx.dtype)
            out = model.dynamics(z_ctx, tau, d, a_t.unsqueeze(0))
            agent_hid = out['agent_hid'][:, -1]                # (1, D)
            rew_logits = model.reward.forward_mtp(agent_hid)   # (1, L, K)
            r_hat = model.reward.expectation(rew_logits).squeeze(0)
            pred_rew[i] = r_hat.float().cpu().numpy()[:L_mtp]
        targ_rew[i] = real_rew[s:s + L_mtp]

    per_offset: Dict[str, Dict] = {}
    for off in range(L_mtp):
        p = pred_rew[:, off]
        t = targ_rew[:, off]
        if p.std() < 1e-8 or t.std() < 1e-8:
            r = float('nan')
        else:
            r = float(np.corrcoef(p, t)[0, 1])
        per_offset[str(off)] = {
            'mae': float(np.mean(np.abs(p - t))),
            'r': r,
            'pred_mean': float(p.mean()),
            'pred_std': float(p.std()),
            'real_mean': float(t.mean()),
            'real_std': float(t.std()),
        }
    return {
        'mtp_length': int(L_mtp),
        'n_starts': int(len(starts)),
        'per_offset': per_offset,
    }


# ---------------------------------------------------------------------------
# Critic calibration on a real episode
# ---------------------------------------------------------------------------

def _critic_calibration(model, env, device, *,
                          gamma: float = 0.997,
                          seed: int = 34567) -> Dict:
    """V(s_t) prediction vs realized return-to-go on one real episode."""
    cfg = env.cfg
    L = cfg.lookback
    T = cfg.episode_length
    obs_dim = env.obs_dim
    action_dim = env.action_dim
    rng = np.random.default_rng(int(seed))

    ow = env.reset(exploration=False)
    real_obs = np.zeros((T, obs_dim), dtype='float32')
    real_rew = np.zeros((T,), dtype='float32')
    real_act = np.zeros((T, action_dim), dtype='float32')
    for t in range(T):
        a = rng.uniform(-1.0, 1.0, size=(action_dim,)).astype('float32')
        ow_next, scaled_r, done, info = env.step(a)
        real_obs[t] = ow_next[-1]
        real_rew[t] = float(info.get('raw_reward', scaled_r))
        real_act[t] = a
        if done:
            T = t + 1
            real_obs = real_obs[:T]
            real_rew = real_rew[:T]
            real_act = real_act[:T]
            break

    if T < L + 8:
        return {'error': f'episode too short ({T})'}

    # Discounted return-to-go (use the raw plant reward so calibration
    # is reported in the same units as the trainer's reward MTP head).
    G = np.zeros((T,), dtype='float64')
    G[-1] = real_rew[-1]
    for t in range(T - 2, -1, -1):
        G[t] = real_rew[t] + gamma * G[t + 1]

    d_min = 1.0 / cfg.k_max
    tau_ctx_val = 1.0 - cfg.tau_ctx
    starts = np.arange(L, T - 1, dtype='int64')
    if starts.size > 256:
        idx = rng.choice(starts.size, 256, replace=False)
        starts = starts[np.sort(idx)]

    v_pred = np.zeros((starts.size,), dtype='float32')
    g_real = np.zeros((starts.size,), dtype='float32')
    for i, s in enumerate(starts):
        s = int(s)
        ow_window = real_obs[s - L:s]
        a_window = real_act[s - L:s]
        with torch.no_grad():
            ow_t = torch.from_numpy(ow_window).to(device)
            a_t = torch.from_numpy(a_window).to(device)
            z_ctx = model.tokenizer.encode(ow_t).unsqueeze(0)
            tau = torch.full((1, L), tau_ctx_val, device=device,
                              dtype=z_ctx.dtype)
            d = torch.full((1, L), d_min, device=device,
                            dtype=z_ctx.dtype)
            out = model.dynamics(z_ctx, tau, d, a_t.unsqueeze(0))
            agent_hid = out['agent_hid'][:, -1]
            v_logits = model.value(agent_hid)
            v_pred[i] = float(model.value.expectation(v_logits))
        g_real[i] = float(G[s])

    if v_pred.std() < 1e-8 or g_real.std() < 1e-8:
        r = float('nan')
    else:
        r = float(np.corrcoef(v_pred, g_real)[0, 1])
    return {
        'gamma': float(gamma),
        'n_points': int(starts.size),
        'v_mean': float(v_pred.mean()),
        'v_std': float(v_pred.std()),
        'g_mean': float(g_real.mean()),
        'g_std': float(g_real.std()),
        'mae': float(np.mean(np.abs(v_pred - g_real))),
        'r_pearson': r,
    }


# ---------------------------------------------------------------------------
# Policy distribution
# ---------------------------------------------------------------------------

def _policy_distribution(model, env, device, *, deterministic: bool,
                          seed: int = 45678) -> Dict:
    """Action histogram over a real on-policy rollout."""
    cfg = env.cfg
    L = cfg.lookback
    T = cfg.episode_length
    action_dim = env.action_dim
    n_bins = int(getattr(cfg, 'n_action_bins', 21))

    ow = env.reset(exploration=False)
    a_history = np.zeros((L, action_dim), dtype='float32')
    actions = np.zeros((T, action_dim), dtype='float32')
    d_min = 1.0 / cfg.k_max
    tau_ctx_val = 1.0 - cfg.tau_ctx
    for t in range(T):
        with torch.no_grad():
            ow_t = torch.from_numpy(ow).to(device)
            a_t = torch.from_numpy(a_history).to(device)
            z_ctx = model.tokenizer.encode(ow_t).unsqueeze(0)
            tau = torch.full((1, L), tau_ctx_val, device=device,
                              dtype=z_ctx.dtype)
            d = torch.full((1, L), d_min, device=device,
                            dtype=z_ctx.dtype)
            out = model.dynamics(z_ctx, tau, d, a_t.unsqueeze(0))
            agent_hid = out['agent_hid'][:, -1]
            a_act, _, _ = model.policy(agent_hid, deterministic=deterministic)
        a_np = a_act.float().squeeze(0).cpu().numpy()
        actions[t] = a_np
        ow, _, done, _ = env.step(a_np)
        a_history = np.concatenate([a_history[1:], a_np[None, :]], axis=0)
        if done:
            T = t + 1
            actions = actions[:T]
            break

    # Per-channel histogram in [-1, +1].
    edges = np.linspace(-1.0, 1.0, n_bins + 1)
    hist_per_ch: List[Dict] = []
    for c in range(action_dim):
        h, _ = np.histogram(actions[:, c], bins=edges)
        h = h.astype('float64') / max(1, h.sum())
        # Empirical entropy in nats (for comparison to log(n_bins)).
        nz = h[h > 0]
        emp_ent = float(-np.sum(nz * np.log(nz)))
        hist_per_ch.append({
            'channel': c,
            'hist': h.tolist(),
            'edges': edges.tolist(),
            'empirical_entropy': emp_ent,
            'log_n_bins': float(math.log(n_bins)),
            'mean': float(actions[:, c].mean()),
            'std': float(actions[:, c].std()),
            'unique_values': int(np.unique(np.round(actions[:, c], 4)).size),
        })
    return {
        'deterministic': bool(deterministic),
        'n_action_bins': int(n_bins),
        'episode_length_used': int(T),
        'per_channel': hist_per_ch,
    }


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------

def compute_training_diagnostics(*,
                                   controller_dir: Path,
                                   env,
                                   model,
                                   device: torch.device,
                                   out_dir: Path,
                                   k_max: int = 32,
                                   gamma: float = 0.997,
                                   ) -> Dict:
    """Run all five diagnostic families and dump JSON + PNG into ``out_dir``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    diag: Dict = {'controller_dir': str(controller_dir),
                   'k_max': int(k_max),
                   'gamma': float(gamma)}

    # 1. Stage metrics.
    cfg = getattr(env, 'cfg', None)
    rows, stage = _parse_train_log(
        Path(controller_dir) / 'train_log.jsonl',
        policy_type=str(getattr(cfg, 'policy_type', 'continuous')),
        n_action_bins=int(getattr(cfg, 'n_action_bins', 21)),
        actor_loss_type=str(getattr(cfg, 'actor_loss_type', 'reinforce')),
    )
    diag['stage_metrics'] = stage

    # 2. WM fidelity.
    try:
        diag['wm_fidelity'] = _wm_kstep_rollout(
            model, env, device, k_max=k_max, n_starts=16)
    except Exception as e:
        diag['wm_fidelity'] = {'error': repr(e)}

    # 3. Reward MTP fidelity.
    try:
        diag['reward_fidelity'] = _reward_mtp_fidelity(model, env, device)
    except Exception as e:
        diag['reward_fidelity'] = {'error': repr(e)}

    # 4. Critic calibration.
    try:
        diag['critic_calib'] = _critic_calibration(
            model, env, device, gamma=gamma)
    except Exception as e:
        diag['critic_calib'] = {'error': repr(e)}

    # 5. Policy distribution (deterministic, since validation is too).
    try:
        diag['policy_dist'] = _policy_distribution(
            model, env, device, deterministic=True)
    except Exception as e:
        diag['policy_dist'] = {'error': repr(e)}

    # Drop bulky raw arrays before writing JSON to keep summaries small.
    wm_lite = {k: v for k, v in (diag.get('wm_fidelity') or {}).items()
                if k not in ('pred_obs', 'real_obs')}
    diag_for_json = dict(diag)
    diag_for_json['wm_fidelity'] = wm_lite
    # Strip per-channel hist from policy_dist (kept only in plot pipeline).
    pd = diag_for_json.get('policy_dist') or {}
    if isinstance(pd, dict) and 'per_channel' in pd:
        pd_lite = dict(pd)
        pd_lite['per_channel'] = [
            {k: v for k, v in ch.items() if k not in ('hist', 'edges')}
            for ch in pd['per_channel']
        ]
        diag_for_json['policy_dist'] = pd_lite

    with open(out_dir / 'wm_diagnostics.json', 'w') as f:
        json.dump(diag_for_json, f, indent=2, default=str)

    _plot_diagnostics(diag, out_dir / 'wm_diagnostics.png',
                       title=f'{Path(controller_dir).name}  diagnostics')
    return diag_for_json


def _plot_diagnostics(diag: Dict, out_path: Path, title: str = '') -> None:
    """4-panel diagnostic plot."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # ---- (0,0) WM fidelity: pred vs real for one start, channel 0
    ax = axes[0, 0]
    wm = diag.get('wm_fidelity') or {}
    if 'pred_obs' in wm and 'real_obs' in wm:
        pred = np.asarray(wm['pred_obs'])
        real = np.asarray(wm['real_obs'])
        if pred.size and real.size:
            ax.plot(real[0, :, 0], color='#1976d2', lw=1.6, label='real (ch 0)')
            ax.plot(pred[0, :, 0], color='#d81b60', lw=1.4, ls='--',
                    label='WM pred (ch 0)')
            ax.set_xlabel('k-step open-loop offset')
            ax.set_ylabel('obs[0] (norm)')
        ax.legend(loc='best', fontsize=8)
    ax.set_title('WM open-loop fidelity (1 start, channel 0)')
    ax.grid(alpha=0.3)

    # ---- (0,1) Reward MTP scatter: pred vs real at offset 0
    ax = axes[0, 1]
    rw = diag.get('reward_fidelity') or {}
    per_off = rw.get('per_offset') or {}
    if per_off:
        offs = sorted(int(k) for k in per_off.keys())
        rs = [per_off[str(o)].get('r', float('nan')) for o in offs]
        maes = [per_off[str(o)].get('mae', float('nan')) for o in offs]
        ax.bar([o - 0.2 for o in offs], rs, width=0.4, color='#388e3c',
                label='Pearson r')
        ax2 = ax.twinx()
        ax2.bar([o + 0.2 for o in offs], maes, width=0.4, color='#fb8c00',
                  alpha=0.7, label='MAE')
        ax2.set_ylabel('MAE', color='#fb8c00')
        ax.set_ylabel('Pearson r', color='#388e3c')
        ax.set_xlabel('MTP offset')
        ax.set_ylim(-0.05, 1.05)
        ax.legend(loc='upper left', fontsize=8)
        ax2.legend(loc='upper right', fontsize=8)
    ax.set_title('Reward MTP head fidelity per offset')
    ax.grid(alpha=0.3)

    # ---- (1,0) Critic calibration scatter
    ax = axes[1, 0]
    cc = diag.get('critic_calib') or {}
    if 'r_pearson' in cc:
        ax.text(0.05, 0.85,
                 f"Pearson r = {cc.get('r_pearson', float('nan')):.3f}\n"
                 f"MAE       = {cc.get('mae', float('nan')):.3f}\n"
                 f"V mean    = {cc.get('v_mean', 0):+.2f}  (std {cc.get('v_std', 0):.2f})\n"
                 f"G mean    = {cc.get('g_mean', 0):+.2f}  (std {cc.get('g_std', 0):.2f})\n"
                 f"n         = {cc.get('n_points', 0)}",
                 transform=ax.transAxes, fontsize=10, family='monospace',
                 verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='#fffde7'))
    ax.set_title('Critic calibration  V(s) vs return-to-go')
    ax.set_axis_off()

    # ---- (1,1) Policy histogram (channel 0, deterministic)
    ax = axes[1, 1]
    pd = diag.get('policy_dist') or {}
    chs = pd.get('per_channel') or []
    if chs:
        ch0 = chs[0]
        h = np.asarray(ch0.get('hist', []))
        e = np.asarray(ch0.get('edges', []))
        if h.size and e.size:
            centers = 0.5 * (e[:-1] + e[1:])
            ax.bar(centers, h, width=(e[1] - e[0]) * 0.9,
                    color='#5e35b1')
        ax.set_xlabel('action[0] (norm)')
        ax.set_ylabel('frequency')
        ax.text(0.02, 0.95,
                 f"empirical H = {ch0.get('empirical_entropy', float('nan')):.3f}\n"
                 f"log(n_bins) = {ch0.get('log_n_bins', float('nan')):.3f}\n"
                 f"unique vals = {ch0.get('unique_values', 0)}",
                 transform=ax.transAxes, fontsize=9, family='monospace',
                 verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='#fffde7'))
    ax.set_title('Policy action distribution (deterministic)')
    ax.grid(alpha=0.3)

    fig.suptitle(title, fontsize=11, y=0.995)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
