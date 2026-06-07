"""DMC-style world-model transfer-function (step-response) matrix.

WHAT THIS MEASURES (and why it exists)
--------------------------------------
The correlation-based WM-fidelity probe (`per_offset r`, `best_h`) only tells
us whether the world model's predictions move *together* with the plant — it is
scale-invariant and says NOTHING about whether the model captured the correct
**gains** (how much CV moves per unit MV) or the correct settling **dynamics**
across the operating region.  For control that is the property that actually
matters.  This diagnostic measures it directly.

For every MV->CV pair we step the MV by a fixed amount from a *settled*
operating point, hold it, and record the CV response in BOTH:
  * the world model (open-loop imagination), and
  * the real simulator (ground truth),
from the identical settled state.  The response is normalised by the
engineering MV step to give a transfer-function gain curve ``g_ij(t)``
(units: ΔCV-eng per ΔMV-eng; its asymptote is the steady-state gain).

Because the plant is nonlinear, we repeat the step at several operating points
across the region (and in both directions) and aggregate into a MEAN curve plus
a MIN/MAX envelope — exactly the "average transfer function and the maximum
variation around it" a DMC step-response model would show.  Overlaying the WM
curve on the real-sim curve makes this a direct, quantitative WM-fidelity plot.

SCOPE (v1): MV->CV transfer functions (the agent's actuators).  DV->CV is left
for a follow-up (DV steps must be injected through the env disturbance schedule,
not the action vector).  Sim-agnostic: any MV/CV count.

Standalone use:
  PYTHONPATH=$PWD \
  $PWD/../neural-apc-mbrl-env/bin/python -m evaluation.wm_transfer_matrix \
      --run-dir output/test_sim/run_XXXX
"""
from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Reuse the proven, identical rollout machinery the steady-state diagnostic
# uses so the WM/real comparison matches training exactly.
from tools.wm_steady_state_diagnostic import (
    _imagine_open_loop, _imagine_open_loop_rssm, _is_rssm_model, _quiet_env,
)


def _settle_capture(env, base_action: np.ndarray, settle_steps: int,
                    L: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Settle the env at ``base_action`` and capture the lookback window.

    Returns ``(lookback_obs (L,O), lookback_act (L,A), base_control_eng (A,),
    settled_obs (O,))``.  Leaves the env AT the settled state so a real
    open-loop step rollout can continue directly from here.
    """
    _quiet_env(env)
    env.reset(exploration=False)
    env._schedule = []                 # see _quiet_env rationale
    env._hidden_disturbance = None
    obs_hist: List[np.ndarray] = []
    for _ in range(settle_steps):
        ow, _, done, _ = env.step(base_action)
        obs_hist.append(ow[-1].copy())
        if done:                       # episode cap — re-settle on a fresh env
            env.reset(exploration=False)
            env._schedule = []
            env._hidden_disturbance = None
    obs_arr = np.asarray(obs_hist, dtype='float32')
    Lc = min(L, obs_arr.shape[0])
    lookback_obs = obs_arr[-Lc:].copy()
    lookback_act = np.tile(base_action.astype('float32'), (Lc, 1))
    base_control = np.asarray(getattr(env, '_prev_control',
                                      np.zeros(env.action_dim)),
                              dtype='float32').copy()
    tail = obs_arr[-max(1, settle_steps // 5):]
    settled_obs = tail.mean(axis=0)
    return lookback_obs, lookback_act, base_control, settled_obs


def _real_step_rollout(env, action: np.ndarray, horizon: int,
                       ) -> Tuple[np.ndarray, np.ndarray]:
    """Step the (already-settled) env ``horizon`` steps under a constant
    ``action`` and return ``(real_obs (horizon,O), stepped_control_eng (A,))``.
    """
    obs_dim = env.obs_dim
    out = np.zeros((horizon, obs_dim), dtype='float32')
    stepped_control = None
    for kk in range(horizon):
        ow, _, done, _ = env.step(action)
        out[kk] = ow[-1]
        if kk == 0:
            stepped_control = np.asarray(
                getattr(env, '_prev_control', np.zeros(env.action_dim)),
                dtype='float32').copy()
        if done:
            out[kk + 1:] = out[kk]
            break
    if stepped_control is None:
        stepped_control = np.asarray(
            getattr(env, '_prev_control', np.zeros(env.action_dim)),
            dtype='float32').copy()
    return out, stepped_control


def _wm_rollout(model, lookback_obs, lookback_act, act_seq, horizon, device,
                k_max: int) -> np.ndarray:
    if _is_rssm_model(model):
        return _imagine_open_loop_rssm(
            model, lookback_obs, lookback_act, act_seq, horizon, device)
    import torch
    with torch.no_grad():
        z_hist = model.tokenizer.encode(
            torch.from_numpy(lookback_obs).to(device))
        a_hist = torch.from_numpy(lookback_act).to(device)
    return _imagine_open_loop(model, z_hist, a_hist, act_seq, horizon,
                              k_max, device)


def _dv_settle_step_rollout(env, base_action: np.ndarray, dv_pos: int,
                            dv_offset_eng: float, settle_steps: int,
                            horizon: int, L: int,
                            ) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                                       np.ndarray, bool]:
    """Settle the env at ``base_action`` with the DV at nominal, capture the
    lookback, then STEP the measured DV (``set_disturbance_offset('dv', pos,
    offset_eng)``) and roll out ``horizon`` steps holding ``base_action``.

    Returns ``(lookback_obs (L,O), lookback_act (L,A), pre_obs (O,),
    real_obs (horizon,O), applied)`` where ``applied`` is False if the sim has
    no ``set_disturbance_offset`` (DV→CV then skipped for that sim).
    """
    obs_dim = env.obs_dim

    def _nominal():
        _quiet_env(env)
        env.reset(exploration=False)
        env._schedule = []
        env._hidden_disturbance = None
        try:
            env.sim.set_disturbance_offset('dv', dv_pos, 0.0)
        except Exception:
            return False
        return True

    if not _nominal():
        return (np.zeros((0, obs_dim), 'float32'), np.zeros((0, env.action_dim),
                'float32'), np.zeros(obs_dim, 'float32'),
                np.zeros((horizon, obs_dim), 'float32'), False)
    obs_hist: List[np.ndarray] = []
    for _ in range(settle_steps):
        ow, _, done, _ = env.step(base_action)
        obs_hist.append(ow[-1].copy())
        if done:
            _nominal()
    obs_arr = np.asarray(obs_hist, dtype='float32')
    Lc = min(L, obs_arr.shape[0])
    lookback_obs = obs_arr[-Lc:].copy()
    lookback_act = np.tile(base_action.astype('float32'), (Lc, 1))
    pre = obs_arr[-1].copy()
    # Step the measured DV (engineering units) and roll out.
    try:
        env.sim.set_disturbance_offset('dv', dv_pos, float(dv_offset_eng))
    except Exception:
        return lookback_obs, lookback_act, pre, np.tile(pre, (horizon, 1)), False
    real = np.zeros((horizon, obs_dim), dtype='float32')
    for kk in range(horizon):
        ow, _, done, _ = env.step(base_action)
        real[kk] = ow[-1]
        if done:
            real[kk + 1:] = real[kk]
            break
    return lookback_obs, lookback_act, pre, real, True


def compute_dv_transfer_matrix(model, env, cfg, device, *,
                               obs_std: Optional[np.ndarray] = None,
                               n_levels: int = 3, step_frac: float = 0.4,
                               horizon: int = 0, settle_steps: int = 0,
                               seed: int = 20260607) -> Dict:
    """WM-vs-real **DV→CV** step-response matrix (Option B DV-as-input only).

    Mirrors ``compute_transfer_matrix`` but the exogenous input is a measured
    DV (stepped via the sim disturbance offset), not an MV action.  The WM side
    holds the stepped DV constant over imagination via ``dv_hold_override`` —
    so this measures whether the world model learned ∂CV/∂DV.  Only meaningful
    when the WM was built with DV-as-input (``dynamics.dv_dim > 0``); returns an
    empty result otherwise.  Sim-agnostic: needs ``sim.set_disturbance_offset``
    and ``sim.dv_normalization_ranges`` (skips DVs lacking them).
    """
    rssm = getattr(model, 'dynamics', None)
    dv_dim = int(getattr(rssm, 'dv_dim', 0) or 0)
    result: Dict = {'cv_names': [], 'dv_names': [], 'pairs': {}, 'enabled': False}
    if dv_dim <= 0 or not _is_rssm_model(model):
        return result
    dv_obs_idx = [int(x) for x in list(getattr(rssm, 'dv_index_t', []))]
    if not dv_obs_idx:
        return result
    cv_idx = list(env.cv_indices)
    n_mv = int(env.action_dim)
    obs_dim = int(env.obs_dim)
    H = int(horizon) if horizon > 0 else max(40, int(1.5 * int(getattr(cfg, 'horizon', 30))))
    S = int(settle_steps) if settle_steps > 0 else H
    L = min(int(getattr(cfg, 'lookback', 64)), S)
    if obs_std is None:
        obs_std = np.ones(obs_dim, dtype='float32')
    dv_ranges = list(getattr(env.sim, 'dv_normalization_ranges', []) or [])
    cv_names = list(env.meta.get('cv_names') or [f'CV{c}' for c in cv_idx])
    dv_names = list(env.meta.get('dv_names') or
                    [f'DV{p}' for p in range(len(dv_obs_idx))])
    rng = np.random.default_rng(seed)
    # Operating points: a few MV levels (region coverage), DV stepped +/-.
    levels = np.linspace(-0.4, 0.4, max(1, n_levels))
    cells: Dict[str, Dict] = {}
    t_axis = list(range(H))
    for p, dv_oi in enumerate(dv_obs_idx):
        span = 1.0
        if p < len(dv_ranges) and len(dv_ranges[p]) == 2:
            span = float(dv_ranges[p][1]) - float(dv_ranges[p][0])
        step_eng = step_frac * abs(span)
        for lev in levels:
            base_action = np.full(n_mv, float(lev), dtype='float32')
            for d in (+1.0, -1.0):
                lb_obs, lb_act, pre, real, ok = _dv_settle_step_rollout(
                    env, base_action, p, d * step_eng, S, H, L)
                if not ok or lb_obs.shape[0] == 0:
                    continue
                d_dv_eng = float((real[-1, dv_oi] - pre[dv_oi])
                                 * (obs_std[dv_oi] if dv_oi < len(obs_std) else 1.0))
                if abs(d_dv_eng) < 1e-9:
                    continue
                # WM: hold ALL DVs at base, set the stepped one to its realized
                # normalized value, imagine under the held base action.
                dv_override = lb_obs[-1, dv_obs_idx].astype('float32').copy()
                dv_override[p] = float(real[-1, dv_oi])
                act_seq = np.tile(base_action, (H, 1)).astype('float32')
                pred = _imagine_open_loop_rssm(
                    model, lb_obs, lb_act, act_seq, H, device,
                    dv_hold_override=dv_override)
                for ci, c in enumerate(cv_idx):
                    sd = float(obs_std[c]) if c < len(obs_std) else 1.0
                    g_wm = (pred[:, c] - pre[c]) * sd / d_dv_eng
                    g_real = (real[:, c] - pre[c]) * sd / d_dv_eng
                    key = f'{cv_names[ci]}<-{dv_names[p]}'
                    cells.setdefault(key, {'wm': [], 'real': []})
                    cells[key]['wm'].append(g_wm.astype('float32'))
                    cells[key]['real'].append(g_real.astype('float32'))
    # restore nominal DV before returning
    for p in range(len(dv_obs_idx)):
        try:
            env.sim.set_disturbance_offset('dv', p, 0.0)
        except Exception:
            pass

    def _agg(curves: List[np.ndarray]) -> Dict[str, object]:
        arr = np.stack(curves, axis=0)
        ss = arr[:, max(1, int(0.8 * arr.shape[1])):].mean(axis=1)
        return {'mean': arr.mean(axis=0).tolist(), 'lo': arr.min(axis=0).tolist(),
                'hi': arr.max(axis=0).tolist(), 'ss_gain_mean': float(ss.mean()),
                'n': int(arr.shape[0])}

    result['cv_names'] = cv_names
    result['dv_names'] = dv_names
    result['t'] = t_axis
    result['horizon'] = H
    result['enabled'] = bool(cells)
    for key, cur in cells.items():
        if not cur['wm'] or not cur['real']:
            continue
        wm = _agg(cur['wm'])
        real = _agg(cur['real'])
        rg = real['ss_gain_mean']
        wg = wm['ss_gain_mean']
        result['pairs'][key] = {
            'wm': wm, 'real': real, 'wm_ss_gain': wg, 'real_ss_gain': rg,
            'ss_gain_ratio_wm_over_real': (wg / rg) if abs(rg) > 1e-9 else float('nan'),
            'ss_gain_abs_err': abs(wg - rg)}
    return result


def compute_transfer_matrix(model, env, cfg, device, *,
                            obs_std: Optional[np.ndarray] = None,
                            n_levels: int = 5, level_span: float = 0.6,
                            step_frac: float = 0.4, horizon: int = 0,
                            settle_steps: int = 0, max_starts_note: str = '',
                            seed: int = 20260605) -> Dict:
    """Build the WM-vs-real step-response matrix over the operating region.

    Returns a nested dict keyed ``f'{cv}<-{mv}'`` plus metadata.  Each cell
    holds the time axis and the mean / min / max engineering-gain curves for
    both the world model and the real simulator, with steady-state gains.
    """
    rng = np.random.default_rng(seed)
    cv_idx = list(env.cv_indices)
    n_mv = int(env.action_dim)
    obs_dim = int(env.obs_dim)
    H = int(horizon) if horizon > 0 else max(40, int(1.5 * int(getattr(cfg, 'horizon', 30))))
    S = int(settle_steps) if settle_steps > 0 else H
    L = min(int(getattr(cfg, 'lookback', 64)), S)
    k_max = int(getattr(cfg, 'k_max', 4))
    if obs_std is None:
        obs_std = np.ones(obs_dim, dtype='float32')
    levels = np.linspace(-abs(level_span), abs(level_span), max(1, n_levels))
    directions = (+abs(step_frac), -abs(step_frac))

    cv_names = list(env.meta.get('cv_names') or
                    [f'CV{c}' for c in cv_idx])
    mv_names = list(env.meta.get('mv_names') or
                    [f'MV{j}' for j in range(n_mv)])

    cells: Dict[str, Dict] = {}
    t_axis = list(range(H))
    for j in range(n_mv):
        for lev in levels:
            base_action = np.zeros(n_mv, dtype='float32')
            base_action[j] = float(lev)
            for d in directions:
                stepped = base_action.copy()
                stepped[j] = float(np.clip(lev + d, -1.0, 1.0))
                if abs(stepped[j] - base_action[j]) < 1e-6:
                    continue  # clipped to no-op at the rail
                # Settle, WM rollout, then real rollout (env is left settled).
                lb_obs, lb_act, base_ctrl, _settled = _settle_capture(
                    env, base_action, S, L)
                act_seq = np.tile(stepped, (H, 1)).astype('float32')
                pred_obs = _wm_rollout(model, lb_obs, lb_act, act_seq, H,
                                       device, k_max)
                real_obs, stepped_ctrl = _real_step_rollout(env, stepped, H)
                d_mv_eng = float(stepped_ctrl[j] - base_ctrl[j])
                if abs(d_mv_eng) < 1e-9:
                    continue
                pre = real_obs[0]      # response measured vs first settled obs
                for ci, c in enumerate(cv_idx):
                    sd = float(obs_std[c]) if c < len(obs_std) else 1.0
                    # ΔCV engineering = ΔCV_norm * channel_std; transfer gain
                    # = ΔCV_eng / ΔMV_eng.
                    g_wm = (pred_obs[:, c] - pre[c]) * sd / d_mv_eng
                    g_real = (real_obs[:, c] - real_obs[0, c]) * sd / d_mv_eng
                    key = f'{cv_names[ci]}<-{mv_names[j]}'
                    cells.setdefault(key, {'wm': [], 'real': []})
                    cells[key]['wm'].append(g_wm.astype('float32'))
                    cells[key]['real'].append(g_real.astype('float32'))

    def _agg(curves: List[np.ndarray]) -> Dict[str, List[float]]:
        arr = np.stack(curves, axis=0)                 # (N, H)
        ss = arr[:, max(1, int(0.8 * arr.shape[1])):].mean(axis=1)
        return {
            'mean': arr.mean(axis=0).tolist(),
            'lo': arr.min(axis=0).tolist(),
            'hi': arr.max(axis=0).tolist(),
            'ss_gain_mean': float(ss.mean()),
            'ss_gain_lo': float(ss.min()),
            'ss_gain_hi': float(ss.max()),
            'n': int(arr.shape[0]),
        }

    result: Dict = {
        't': t_axis, 'horizon': H, 'settle_steps': S, 'n_levels': n_levels,
        'level_span': level_span, 'step_frac': step_frac,
        'cv_names': cv_names, 'mv_names': mv_names, 'pairs': {},
    }
    for key, cur in cells.items():
        if not cur['wm'] or not cur['real']:
            continue
        wm = _agg(cur['wm'])
        real = _agg(cur['real'])
        # Quantitative fidelity: WM vs real steady-state gain ratio + error.
        rg = real['ss_gain_mean']
        wg = wm['ss_gain_mean']
        gain_ratio = (wg / rg) if abs(rg) > 1e-9 else float('nan')
        result['pairs'][key] = {
            'wm': wm, 'real': real,
            'wm_ss_gain': wg, 'real_ss_gain': rg,
            'ss_gain_ratio_wm_over_real': gain_ratio,
            'ss_gain_abs_err': abs(wg - rg),
        }
    return result


def plot_transfer_matrix(result: Dict, out_path: Path, title: str = '') -> None:
    """Render the WM-vs-real step-response matrix (CV rows × MV cols)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    cv_names = result['cv_names']
    mv_names = result['mv_names']
    t = np.asarray(result['t'], dtype='float32')
    n_cv = len(cv_names)
    n_mv = len(mv_names)
    # Floor the figure size so the title / legend / caption always have room
    # (a 1×1 SISO grid is otherwise too small and clips the suptitle).
    fig_w = max(7.0, 4.2 * n_mv)
    fig_h = max(4.8, 3.0 * n_cv + 1.3)
    fig, axes = plt.subplots(n_cv, n_mv, figsize=(fig_w, fig_h),
                             squeeze=False)
    for ci, cvn in enumerate(cv_names):
        for j, mvn in enumerate(mv_names):
            ax = axes[ci][j]
            key = f'{cvn}<-{mvn}'
            cell = result['pairs'].get(key)
            if not cell:
                ax.set_visible(False)
                continue
            for who, color in (('real', 'k'), ('wm', 'C0')):
                m = np.asarray(cell[who]['mean'], dtype='float32')
                lo = np.asarray(cell[who]['lo'], dtype='float32')
                hi = np.asarray(cell[who]['hi'], dtype='float32')
                lbl = 'real sim' if who == 'real' else 'world model'
                ax.plot(t, m, color=color, lw=1.8,
                        ls='-' if who == 'real' else '--', label=lbl)
                ax.fill_between(t, lo, hi, color=color, alpha=0.15,
                                linewidth=0)
            ax.axhline(0.0, color='grey', lw=0.6, alpha=0.6)
            ratio = cell.get('ss_gain_ratio_wm_over_real', float('nan'))
            ax.set_title(
                f'{key}\n'
                f'SS gain  real={cell["real_ss_gain"]:+.3g}  '
                f'wm={cell["wm_ss_gain"]:+.3g}  (wm/real={ratio:.2f})',
                fontsize=8)
            if ci == n_cv - 1:
                ax.set_xlabel('step (samples after the MV step)')
            if j == 0:
                ax.set_ylabel('ΔCV / ΔMV  (eng units)')
            ax.grid(alpha=0.25)
    # One shared, explicit colour-coding legend (proxy handles) so every
    # subplot is unambiguous — replaces the single per-axis legend that left
    # the shaded band undefined.
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    legend_handles = [
        Line2D([0], [0], color='k', lw=1.8, ls='-',
               label='real sim (ground truth)'),
        Line2D([0], [0], color='C0', lw=1.8, ls='--',
               label='world model (imagination)'),
        Patch(facecolor='grey', alpha=0.2,
              label='shaded = min–max across operating region'),
        Line2D([0], [0], color='grey', lw=0.6, label='zero gain'),
    ]
    fig.legend(handles=legend_handles, loc='lower center', ncol=4,
               fontsize=8, framealpha=0.9, bbox_to_anchor=(0.5, 0.085))
    sup = title or ('World-model vs real-plant transfer-function matrix '
                    '(MV→CV step response)')
    fig.suptitle(sup, fontsize=11, y=0.99)
    # Two-line "how to read it" caption (the colour key lives in the legend
    # above).  Kept short per line so it never overflows a small SISO figure.
    fig.text(
        0.5, 0.02,
        'How to read: world model (blue dashed) should OVERLAP real sim '
        '(black solid).\nFlatter/smaller ⇒ WM gain too small; opposite sign ⇒ '
        'wrong direction.  Cell titles: SS gains + wm/real ratio (1.0 = exact).',
        ha='center', va='bottom', fontsize=8)
    fig.tight_layout(rect=(0, 0.16, 1, 0.95))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def plot_dv_transfer_matrix(result: Dict, out_path: Path,
                            title: str = '') -> None:
    """Render the WM-vs-real **DV→CV** step-response matrix (CV rows × DV cols).

    Same colour key as the MV plot: black solid = real sim, blue dashed = world
    model, shaded = min–max across operating points.  Gains are ΔCV/ΔDV (eng).
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    cv_names = result['cv_names']
    dv_names = result['dv_names']
    t = np.asarray(result.get('t', []), dtype='float32')
    n_cv, n_dv = len(cv_names), len(dv_names)
    fig_w = max(7.0, 4.2 * n_dv)
    fig_h = max(4.8, 3.0 * n_cv + 1.3)
    fig, axes = plt.subplots(n_cv, n_dv, figsize=(fig_w, fig_h), squeeze=False)
    for ci, cvn in enumerate(cv_names):
        for j, dvn in enumerate(dv_names):
            ax = axes[ci][j]
            cell = result['pairs'].get(f'{cvn}<-{dvn}')
            if not cell:
                ax.set_visible(False)
                continue
            for who, color in (('real', 'k'), ('wm', 'C0')):
                m = np.asarray(cell[who]['mean'], dtype='float32')
                lo = np.asarray(cell[who]['lo'], dtype='float32')
                hi = np.asarray(cell[who]['hi'], dtype='float32')
                ax.plot(t, m, color=color, lw=1.8,
                        ls='-' if who == 'real' else '--',
                        label='real sim' if who == 'real' else 'world model')
                ax.fill_between(t, lo, hi, color=color, alpha=0.15, linewidth=0)
            ax.axhline(0.0, color='grey', lw=0.6, alpha=0.6)
            ratio = cell.get('ss_gain_ratio_wm_over_real', float('nan'))
            ax.set_title(f'{cvn}<-{dvn}\nSS gain  real={cell["real_ss_gain"]:+.3g}  '
                         f'wm={cell["wm_ss_gain"]:+.3g}  (wm/real={ratio:.2f})',
                         fontsize=8)
            if ci == n_cv - 1:
                ax.set_xlabel('step (samples after the DV step)')
            if j == 0:
                ax.set_ylabel('ΔCV / ΔDV  (eng units)')
            ax.grid(alpha=0.25)
    legend_handles = [
        Line2D([0], [0], color='k', lw=1.8, ls='-', label='real sim (ground truth)'),
        Line2D([0], [0], color='C0', lw=1.8, ls='--', label='world model (DV held)'),
        Patch(facecolor='grey', alpha=0.2, label='shaded = min–max across operating region'),
        Line2D([0], [0], color='grey', lw=0.6, label='zero gain'),
    ]
    fig.legend(handles=legend_handles, loc='lower center', ncol=4, fontsize=8,
               framealpha=0.9, bbox_to_anchor=(0.5, 0.085))
    sup = title or ('World-model vs real-plant DV→CV transfer matrix '
                    '(measured-disturbance step response)')
    fig.suptitle(sup, fontsize=11, y=0.99)
    fig.text(0.5, 0.02,
             'How to read: world model (blue dashed) should OVERLAP real sim '
             '(black solid).\nThe WM holds the stepped DV constant (Option B '
             'feedforward); flatter/smaller ⇒ WM ∂CV/∂DV too small.',
             ha='center', va='bottom', fontsize=8)
    fig.tight_layout(rect=(0, 0.16, 1, 0.95))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def compute_and_plot(model, env, cfg, device, out_dir: Path, *,
                     obs_std: Optional[np.ndarray] = None,
                     title: str = '') -> Optional[Dict]:
    """Convenience wrapper for validation: compute, plot, and dump JSON.

    Fully guarded — returns ``None`` (and prints) on any failure so it can
    never break a validation run.  Knobs via env:
      DREAMER_WM_TF_LEVELS, _SPAN, _STEP_FRAC, _HORIZON, _SETTLE.
    """
    try:
        n_levels = int(os.environ.get('DREAMER_WM_TF_LEVELS', '5'))
        span = float(os.environ.get('DREAMER_WM_TF_SPAN', '0.6'))
        step_frac = float(os.environ.get('DREAMER_WM_TF_STEP_FRAC', '0.4'))
        horizon = int(os.environ.get('DREAMER_WM_TF_HORIZON', '0'))
        settle = int(os.environ.get('DREAMER_WM_TF_SETTLE', '0'))
        result = compute_transfer_matrix(
            model, env, cfg, device, obs_std=obs_std, n_levels=n_levels,
            level_span=span, step_frac=step_frac, horizon=horizon,
            settle_steps=settle)
        out_dir = Path(out_dir)
        plot_transfer_matrix(result, out_dir / 'wm_transfer_matrix.png',
                             title=title)
        with open(out_dir / 'wm_transfer_matrix.json', 'w') as f:
            json.dump(result, f, indent=2)
        # DV→CV matrix (Option B DV-as-input only; empty/no-op otherwise).
        # Separately guarded so a DV-side failure never loses the MV result.
        try:
            dv_result = compute_dv_transfer_matrix(
                model, env, cfg, device, obs_std=obs_std, n_levels=3,
                step_frac=step_frac, horizon=horizon, settle_steps=settle)
            if dv_result.get('enabled') and dv_result.get('pairs'):
                plot_dv_transfer_matrix(
                    dv_result, out_dir / 'wm_dv_transfer_matrix.png', title=title)
                with open(out_dir / 'wm_dv_transfer_matrix.json', 'w') as f:
                    json.dump(dv_result, f, indent=2)
                _dvp = dv_result['pairs']
                print(f'[val] WM DV→CV transfer matrix: {len(_dvp)} DV/CV '
                      f'pair(s) -> {out_dir}/wm_dv_transfer_matrix.png',
                      flush=True)
        except Exception as _dve:
            print(f'[val] WM DV→CV transfer matrix skipped ({_dve!r})',
                  flush=True)
        # Concise fidelity summary to the log.
        pairs = result.get('pairs', {})
        worst = None
        for k, v in pairs.items():
            err = abs(v.get('ss_gain_abs_err', 0.0))
            if worst is None or err > worst[1]:
                worst = (k, err)
        n = len(pairs)
        msg = (f'[val] WM transfer matrix: {n} MV/CV pair(s) -> '
               f'{out_dir}/wm_transfer_matrix.png')
        if worst is not None:
            wk = worst[0]
            wv = pairs[wk]
            msg += (f'  | worst gain mismatch {wk}: '
                    f'real={wv["real_ss_gain"]:+.3g} wm={wv["wm_ss_gain"]:+.3g}')
        print(msg, flush=True)
        return result
    except Exception as e:  # never break validation
        import traceback
        print(f'[val] WM transfer matrix skipped: {e!r}', flush=True)
        traceback.print_exc()
        return None


def main() -> int:
    import argparse
    import torch
    from tools.wm_steady_state_diagnostic import (
        _find_ckpt, _load_model, _pick_device,
    )
    from training.train import APCEnv

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run-dir', required=True)
    ap.add_argument('--ckpt', default=None)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    run_dir = Path(args.run_dir).resolve()
    ckpt_path = _find_ckpt(run_dir, args.ckpt)
    device, _ = _pick_device()
    model, cfg, on = _load_model(ckpt_path, device)
    model.eval()

    env = APCEnv(cfg, np.random.default_rng(99_999))
    obs_std = None
    if on and on.get('var') is not None:
        var = np.asarray(on['var'], dtype='float32')
        obs_std = np.clip(np.sqrt(np.maximum(var, 1e-6)), 1e-3, None)
        try:
            env.set_obs_norm_stats(mean=np.asarray(on.get('mean')), var=var,
                                   count=float(on.get('count', 1.0)),
                                   learn=False)
        except Exception:
            pass
    out_dir = Path(args.out).resolve() if args.out else run_dir / 'validation'
    res = compute_and_plot(model, env, cfg, device, out_dir, obs_std=obs_std,
                           title=f'{run_dir.name}  WM transfer matrix')
    return 0 if res is not None else 1


if __name__ == '__main__':
    raise SystemExit(main())
