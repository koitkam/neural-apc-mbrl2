"""WM unmeasured-disturbance PREDICTION diagnostic (feed-forward model check).

Measures how well the world-model's auxiliary disturbance-estimator head
(``model.disturbance``, P87) predicts the TRUE unmeasured (hidden OU) CV
disturbance that the agent cannot see directly — i.e. how good the learned
feed-forward / disturbance-observer model is.

It rolls ONE validation episode with the hidden disturbance FORCED on at full
phase-3 amplitude, streams the WM posterior feature at every step (exactly the
feature the policy + the head consume), runs the head to get the predicted
per-CV disturbance, and compares it against the env-recorded true hidden trace.

Per CV channel it reports: RMSE, NRMSE (RMSE/std(true)), Pearson r, R², and the
best lead/lag (cross-correlation peak offset — does the head predict early or
late?).  Saves ``wm_disturbance_prediction.json`` + a per-channel time-series
PNG (true vs predicted).

RSSM/TSSM only (needs ``obs_step`` + a disturbance head).  CPU-safe.  Returns
``{'enabled': False, 'reason': ...}`` when not applicable so it never breaks a
validation run.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional

import numpy as np
import torch


def _is_rssm(model) -> bool:
    return getattr(model, 'world_model_type', 'sf_transformer') in ('rssm', 'tssm')


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.std() < 1e-9 or b.std() < 1e-9:
        return float('nan')
    return float(np.corrcoef(a, b)[0, 1])


def _best_lead_lag(pred: np.ndarray, true: np.ndarray, max_lag: int) -> Dict:
    """Cross-correlate ``pred`` against ``true`` over ±``max_lag`` steps.

    Returns the lag (in steps) of peak correlation and that correlation.
    Positive lag ⇒ pred must be shifted forward to match true ⇒ the head
    predicts LATE (lags the disturbance); negative ⇒ it LEADS.
    """
    best_lag, best_r = 0, -2.0
    n = len(true)
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            a, b = pred[-lag:], true[:n + lag]
        elif lag > 0:
            a, b = pred[:n - lag], true[lag:]
        else:
            a, b = pred, true
        if len(a) < 8:
            continue
        r = _safe_corr(a, b)
        if r == r and r > best_r:
            best_r, best_lag = r, lag
    return {'best_lag_steps': int(best_lag),
            'best_lag_corr': (float(best_r) if best_r > -2.0 else float('nan'))}


@torch.no_grad()
def compute_disturbance_prediction(model, env, cfg, device, *,
                                   deterministic: bool = True,
                                   warmup_frac: float = 0.1) -> Dict:
    """Roll one forced-disturbance episode and score the head's prediction.

    The caller is expected to have built ``env`` from the run cfg; this fn
    forces the hidden disturbance on at full P3 amplitude itself so it is
    self-contained.  Returns a JSON-able dict (metrics + per-step traces).
    """
    head = getattr(model, 'disturbance', None)
    dob = bool(getattr(getattr(model, 'dynamics', None), 'dob_enabled', False))
    if head is None and not dob:
        return {'enabled': False,
                'reason': 'no disturbance head and DOB disabled'}
    if not _is_rssm(model):
        return {'enabled': False, 'reason': 'disturbance diagnostic is RSSM/TSSM only'}

    # Force the hidden disturbance on, full phase-3 amplitude, spread across the
    # whole episode (mirrors the disturbance-rejection plot env setup).
    _spread_prev = os.environ.get('DREAMER_HIDDEN_DIST_SPREAD')
    os.environ['DREAMER_HIDDEN_DIST_SPREAD'] = '1'
    try:
        env._hidden_disturbance_force = True
        env._current_phase = 3
        env._training_progress = 1.0
        env._disturbance_prob_override = 1.0

        T = int(cfg.episode_length)
        action_dim = int(env.action_dim)
        n_cv = len(env.cv_indices)
        L = int(cfg.lookback)

        obs_window = env.reset(exploration=False)
        rssm = model.dynamics
        state = rssm.initial_state(1, device)
        prev_a = torch.zeros(1, action_dim, device=device)

        # DOB (neural Kalman filter): when the WM carries the disturbance state
        # d_t, IT is the estimate (read d_t, converted to engineering units via
        # the obs-norm std at each CV index) — strictly better than the read-out
        # head.  Else fall back to the head.
        dob_on = bool(getattr(rssm, 'dob_enabled', False))
        cv_std = np.ones(n_cv, dtype='float32')
        if dob_on:
            try:
                _ons = env.get_obs_norm_stats()
                _var = np.asarray(_ons.get('var'), dtype='float64')
                for c, ci in enumerate(env.cv_indices):
                    if 0 <= int(ci) < _var.shape[0]:
                        cv_std[c] = float(np.sqrt(max(_var[int(ci)], 1e-12)))
            except Exception:
                pass
        estimator = 'dob_d_t' if dob_on else 'readout_head'

        true_d = np.zeros((T, n_cv), dtype='float32')
        pred_d = np.zeros((T, n_cv), dtype='float32')

        t_final = 0
        for t in range(T):
            o = torch.from_numpy(obs_window[-1]).to(device).unsqueeze(0)
            with torch.amp.autocast(device_type=device.type,
                                     dtype=torch.bfloat16,
                                     enabled=(device.type == 'cuda')):
                emb = rssm.embed(o)
                # Pass obs=o so the DOB innovation/correction runs when enabled.
                post, _ = rssm.obs_step(state, prev_a, emb, sample=True, obs=o)
                feat = post.feat
                if dob_on and getattr(post, 'd', None) is not None:
                    # d_t is in normalized obs space -> engineering units.
                    d_norm = post.d.float().squeeze(0).cpu().numpy()
                    dpred = d_norm[:n_cv] * cv_std[:n_cv]
                elif head is not None:
                    # read-out head: predicts the per-CV disturbance from feat.
                    dpred = head(feat).float().squeeze(0).cpu().numpy()
                else:
                    dpred = np.zeros(n_cv, dtype='float32')
                action_t, _, _ = model.policy(feat, deterministic=deterministic)
            pred_d[t, :min(n_cv, dpred.shape[0])] = dpred[:n_cv]
            a_np = action_t.float().squeeze(0).cpu().numpy().astype('float32')
            prev_a = torch.from_numpy(a_np).to(device).unsqueeze(0)
            state = post

            next_window, _r, done, info = env.step(a_np)
            hd = info.get('hidden_disturbance')
            if hd is not None:
                hd = np.asarray(hd, dtype='float32').reshape(-1)
                true_d[t, :min(n_cv, hd.shape[0])] = hd[:n_cv]
            obs_window = next_window
            t_final = t
            if done:
                break

        true_d = true_d[:t_final + 1]
        pred_d = pred_d[:t_final + 1]
        n = true_d.shape[0]
        w0 = int(max(L, warmup_frac * n))   # skip WM/posterior warm-up
        w0 = min(w0, max(0, n - 8))
        sr = int(getattr(cfg, 'sample_rate', 1) or 1)
        max_lag = max(2, int(round((env._resolve_plant_timing()[1] or 8) / sr))
                      if hasattr(env, '_resolve_plant_timing') else 8)

        sv = list(env.meta.get('state_variables', []) or [])
        per_channel: List[Dict] = []
        for c in range(n_cv):
            tr = true_d[w0:, c]
            pr = pred_d[w0:, c]
            rmse = float(np.sqrt(np.mean((pr - tr) ** 2)))
            std_t = float(tr.std())
            ss_res = float(np.sum((tr - pr) ** 2))
            ss_tot = float(np.sum((tr - tr.mean()) ** 2))
            r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else float('nan')
            ci = int(env.cv_indices[c])
            per_channel.append({
                'cv_index': ci,
                'cv_name': sv[ci] if 0 <= ci < len(sv) else f'CV{c}',
                'rmse': rmse,
                'nrmse': float(rmse / std_t) if std_t > 1e-9 else float('nan'),
                'pearson_r': _safe_corr(pr, tr),
                'r2': r2,
                'true_std': std_t,
                'pred_std': float(pr.std()),
                **_best_lead_lag(pr, tr, max_lag),
            })

        def _m(key):
            vals = [c[key] for c in per_channel if c[key] == c[key]]
            return float(np.mean(vals)) if vals else float('nan')

        return {
            'enabled': True,
            'episode_length': int(n),
            'warmup_skipped': int(w0),
            'n_cv': int(n_cv),
            'mean_nrmse': _m('nrmse'),
            'mean_pearson_r': _m('pearson_r'),
            'mean_r2': _m('r2'),
            'per_channel': per_channel,
            # raw traces (for the plot + offline re-analysis)
            'true_disturbance_t': true_d.tolist(),
            'pred_disturbance_t': pred_d.tolist(),
            'cv_indices': [int(x) for x in env.cv_indices],
            'sample_rate': sr,
            'estimator': estimator,
            'stop_grad': bool(getattr(cfg, 'disturbance_head_stop_grad', True)),
            'disturbance_loss_scale': float(getattr(cfg, 'disturbance_loss_scale', 0.0)),
        }
    finally:
        if _spread_prev is None:
            os.environ.pop('DREAMER_HIDDEN_DIST_SPREAD', None)
        else:
            os.environ['DREAMER_HIDDEN_DIST_SPREAD'] = _spread_prev


def plot_disturbance_prediction(result: Dict, out_path) -> bool:
    """Per-CV time-series plot: true vs predicted unmeasured disturbance."""
    if not result.get('enabled'):
        return False
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception:
        return False

    true_d = np.asarray(result['true_disturbance_t'], dtype='float32')
    pred_d = np.asarray(result['pred_disturbance_t'], dtype='float32')
    per_ch = result['per_channel']
    n_cv = int(result['n_cv'])
    w0 = int(result.get('warmup_skipped', 0))
    sr = int(result.get('sample_rate', 1) or 1)
    t_ax = np.arange(true_d.shape[0]) * sr

    fig, axes = plt.subplots(n_cv, 1, figsize=(11, 2.8 * n_cv + 0.6),
                             squeeze=False)
    for c in range(n_cv):
        ax = axes[c][0]
        m = per_ch[c]
        ax.fill_between(t_ax, true_d[:, c], 0, color='tab:orange', alpha=0.25,
                        label='true hidden disturbance')
        ax.plot(t_ax, true_d[:, c], color='tab:orange', lw=1.3)
        ax.plot(t_ax, pred_d[:, c], color='tab:blue', lw=1.2, ls='--',
                label='WM head prediction')
        if w0 > 0:
            ax.axvspan(0, w0 * sr, color='gray', alpha=0.12)
            ax.axvline(w0 * sr, color='gray', lw=0.6, ls=':')
        ax.axhline(0, color='k', lw=0.5)
        ax.set_ylabel(m.get('cv_name', f'CV{c}'))
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc='upper right')
        ax.set_title(
            f"{m.get('cv_name', f'CV{c}')}: "
            f"NRMSE={m['nrmse']:.2f}  r={m['pearson_r']:.2f}  "
            f"R²={m['r2']:.2f}  lag={m['best_lag_steps']}step "
            f"(r@lag={m['best_lag_corr']:.2f})", fontsize=9)
    axes[-1][0].set_xlabel('step')
    sg = 'read-out probe (stop-grad)' if result.get('stop_grad') else 'latent-shaping'
    est = result.get('estimator', 'readout_head')
    fig.suptitle(
        f"WM unmeasured-disturbance prediction [{est}] — mean NRMSE={result['mean_nrmse']:.2f} "
        f"r={result['mean_pearson_r']:.2f} R²={result['mean_r2']:.2f}  [{sg}]",
        fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    try:
        fig.savefig(str(out_path), dpi=110)
    finally:
        plt.close(fig)
    return True
