"""WM steady-state imagination diagnostic.

Probes whether the trained world model, asked to imagine far past its
training horizon, produces:

  (1) a CONVERGING trajectory in decoded-obs space (sign of stable
      latent dynamics, not divergence);
  (2) a steady-state value that MATCHES the real simulator's
      steady-state from the same initial state under the same
      constant action (sign of correct steady-state physics, not just
      stable but-wrong dynamics);
  (3) reasonable per-step prediction accuracy over the *training*
      imagination horizon ``H`` so we can see whether 1+2 actually
      matter or whether the WM is already lost by step H.

Three rollout protocols are run from the same N starting states:

  * ``replay``  — apply the same actions the real env took, measures
                  pure dynamics-prediction accuracy.
  * ``zero``    — hold action=0 (mid-MV), measures whether WM
                  converges to a sensible steady-state.
  * ``constant`` — hold a randomly-sampled but persistent action,
                  same convergence question at a different operating
                  point.

For each protocol we report per-step MAE in the env's CV channel(s),
final-step (steady-state) error vs. simulator ground truth, and a
"converged" flag (true if obs std over the last 20 steps < ε).

GPU/CPU selection: auto-detects GPU utilization via ``nvidia-smi`` and
falls back to CPU if utilization >50% or memory_used/total >0.5. Can
be forced with ``DREAMER_WM_DIAG_DEVICE={cpu,cuda}``.

CLI::

    python tools/wm_steady_state_diagnostic.py \
        --run-dir output/test_sim/run_xxx \
        --ckpt   ckpt_iter_00160.pt        # optional, default = best.pt or final.pt
        --n-starts 8 \
        --horizon 200

Outputs: ``<run-dir>/wm_steady_state_diagnostic.json`` and (if
matplotlib is available) ``wm_steady_state_diagnostic.png``.

Importable as ``run_wm_steady_state_diagnostic(run_dir, ...)`` for
the end-of-training hook in ``training/train.py``.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

# Allow `import training.train` from the repo root.
_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------

def _gpu_busy(util_threshold_pct: float = 50.0,
              mem_threshold_frac: float = 0.5) -> Tuple[bool, str]:
    """Return ``(is_busy, reason_str)`` from ``nvidia-smi``.

    ``is_busy=True`` means the script should fall back to CPU. Quiet
    fallback also on any error (no nvidia driver, no GPU at all, etc.).
    """
    if not torch.cuda.is_available():
        return True, 'cuda_unavailable'
    try:
        out = subprocess.check_output(
            ['nvidia-smi',
             '--query-gpu=utilization.gpu,memory.used,memory.total',
             '--format=csv,noheader,nounits'],
            stderr=subprocess.DEVNULL, timeout=5).decode()
    except Exception as e:
        return True, f'nvidia_smi_failed:{e!r}'
    lines = [ln.strip() for ln in out.strip().splitlines() if ln.strip()]
    if not lines:
        return True, 'nvidia_smi_no_lines'
    # Take the busiest GPU (some systems have multiple).
    parts = [ln.split(',') for ln in lines]
    util = max(float(p[0]) for p in parts)
    mem_used = max(float(p[1]) for p in parts)
    mem_total = max(float(p[2]) for p in parts)
    mem_frac = mem_used / max(mem_total, 1.0)
    busy = (util > util_threshold_pct) or (mem_frac > mem_threshold_frac)
    reason = (f'util={util:.0f}% mem_used={mem_used:.0f}/{mem_total:.0f}MiB '
              f'({mem_frac*100:.0f}%)')
    return bool(busy), reason


def _pick_device() -> Tuple[torch.device, str]:
    """Honour ``DREAMER_WM_DIAG_DEVICE`` override; else auto-detect."""
    forced = os.environ.get('DREAMER_WM_DIAG_DEVICE', '').strip().lower()
    if forced in ('cpu',):
        return torch.device('cpu'), 'forced_cpu'
    if forced in ('cuda', 'gpu'):
        if not torch.cuda.is_available():
            return torch.device('cpu'), 'forced_cuda_unavailable_fallback_cpu'
        return torch.device('cuda'), 'forced_cuda'
    busy, reason = _gpu_busy()
    if busy:
        return torch.device('cpu'), f'cpu_auto_gpu_busy({reason})'
    return torch.device('cuda'), f'cuda_auto_gpu_free({reason})'


# ---------------------------------------------------------------------------
# Checkpoint discovery + loading
# ---------------------------------------------------------------------------

def _find_ckpt(run_dir: Path, ckpt_name: Optional[str] = None) -> Path:
    """Resolve a checkpoint to load.

    Priority: ``ckpt_name`` if given → ``final.pt`` → ``best.pt`` →
    latest ``ckpt_iter_NNNNN.pt``.
    """
    if ckpt_name:
        p = run_dir / ckpt_name
        if not p.exists():
            raise FileNotFoundError(f'requested ckpt not found: {p}')
        return p
    for name in ('final.pt', 'best.pt'):
        p = run_dir / name
        if p.exists():
            return p
    iters = sorted(run_dir.glob('ckpt_iter_*.pt'))
    if iters:
        return iters[-1]
    raise FileNotFoundError(f'no checkpoint in {run_dir}')


def _load_model(ckpt_path: Path, device: torch.device):
    """Load DreamerV4 model + cfg from a checkpoint."""
    from training.train import TrainConfig
    from models.dreamer_v4 import DreamerV4, DreamerV4Config

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
        # 'sdpa' is significantly faster on CPU than 'manual' (uses torch's
        # fused scaled_dot_product_attention which has a vectorised CPU path).
        attn_impl='sdpa',
    )
    model = DreamerV4(model_cfg).to(device)
    sd = ckpt_obj['model']
    if any('._orig_mod.' in k for k in sd):
        sd = {k.replace('._orig_mod.', '.'): v for k, v in sd.items()}
    model.load_state_dict(sd)
    model.eval()

    obs_norm = ckpt_obj.get('obs_norm') if isinstance(ckpt_obj, dict) else None
    return model, cfg, obs_norm


# ---------------------------------------------------------------------------
# Rollout: imagined WM trajectory vs simulator ground truth
# ---------------------------------------------------------------------------

@torch.no_grad()
def _imagine_open_loop(model, z_history: torch.Tensor,
                        a_history: torch.Tensor,
                        action_seq: np.ndarray, n_steps: int,
                        k_max: int, device: torch.device
                        ) -> np.ndarray:
    """Roll the WM forward ``n_steps`` open-loop.

    Inputs:
      ``z_history``  (L, z_dim)  initial encoded lookback latents
      ``a_history``  (L, A)      initial action history
      ``action_seq`` (n_steps, A) actions to apply at each imagined step
      ``n_steps``    int         imagination length

    Returns ``pred_obs`` of shape ``(n_steps, obs_dim)``.
    """
    z_hist = z_history.clone()                                   # (L, z)
    a_hist = a_history.clone()                                   # (L, A)
    obs_dim = model.tokenizer.obs_dim
    out = np.zeros((n_steps, obs_dim), dtype='float32')
    for kk in range(n_steps):
        a_step = torch.from_numpy(action_seq[kk]).to(device).unsqueeze(0)  # (1, A)
        z_next = model.imagine_next_z(
            z_hist.unsqueeze(0), a_step,
            k_steps=k_max, tau_ctx=None,
            action_history=a_hist.unsqueeze(0)).squeeze(0)        # (z,)
        obs_hat = model.tokenizer.decode(
            z_next.unsqueeze(0)).squeeze(0).float().cpu().numpy()
        out[kk] = obs_hat[:obs_dim]
        z_hist = torch.cat([z_hist[1:], z_next.unsqueeze(0)], dim=0)
        a_hist = torch.cat([a_hist[1:], a_step], dim=0)
    return out


def _is_rssm_model(model) -> bool:
    # 'tssm' (transformer-SSM, neural-apc-mbrl) implements the SAME interface as
    # the RSSM (initial_state/obs_step/img_step/decode/feat), so the RSSM
    # open-loop rollout protocol applies to it unchanged.
    return getattr(model, 'world_model_type', 'sf_transformer') in ('rssm', 'tssm')


@torch.no_grad()
def _imagine_open_loop_rssm(model, lookback_obs: np.ndarray,
                             lookback_act: np.ndarray,
                             action_seq: np.ndarray, n_steps: int,
                             device: torch.device,
                             dv_hold_override: Optional[np.ndarray] = None,
                             ) -> np.ndarray:
    """RSSM open-loop rollout (mirrors training warmup + imagination).

    Warm-start the posterior over the real lookback via teacher-forced
    ``obs_step`` (action convention ``act[l]`` paired with ``obs[l]``),
    then advance the PRIOR with ``img_step`` under ``action_seq`` and
    decode each imagined feature.  Operates entirely in normalized-obs
    space (decoder output is directly comparable to ``env.step`` obs).

    DV-as-input (Option B): when the WM has ``dv_dim > 0`` the measured-DV
    channels are teacher-forced from the lookback during warm-start and HELD
    CONSTANT over imagination (MPC persistence) — matching training.
    ``dv_hold_override`` (dv_dim,) replaces the held DV value (used by the
    transfer matrix for the DV→CV step response); ``None`` holds the DV at its
    last lookback value.
    """
    rssm = model.dynamics
    obs_dim = rssm.obs_dim
    L = lookback_obs.shape[0]
    _dv_on = int(getattr(rssm, 'dv_dim', 0) or 0) > 0
    state = rssm.initial_state(1, device)
    for l in range(L):
        o = torch.from_numpy(lookback_obs[l]).to(device).unsqueeze(0)
        a = torch.from_numpy(lookback_act[l]).to(device).unsqueeze(0)
        emb = rssm.embed(o)
        dv = o.index_select(-1, rssm.dv_index_t) if _dv_on else None
        post, _ = rssm.obs_step(state, a, emb, dv=dv, sample=True)
        state = post
    # Held DV over imagination: override (DV→CV step) else last-lookback value.
    dv_hold = None
    if _dv_on:
        if dv_hold_override is not None:
            dv_hold = torch.from_numpy(
                np.asarray(dv_hold_override, dtype='float32')
            ).to(device).reshape(1, -1)
        else:
            dv_hold = (torch.from_numpy(lookback_obs[L - 1]).to(device)
                       .unsqueeze(0).index_select(-1, rssm.dv_index_t))
    out = np.zeros((n_steps, obs_dim), dtype='float32')
    for kk in range(n_steps):
        a_step = torch.from_numpy(action_seq[kk]).to(device).unsqueeze(0)
        state = rssm.img_step(state, a_step, dv=dv_hold, sample=True)
        obs_hat = rssm.decode(state.feat).squeeze(0).float().cpu().numpy()
        out[kk] = obs_hat[:obs_dim]
    return out


def _real_open_loop(env, cfg, lookback_obs: np.ndarray,
                     lookback_act: np.ndarray, action_seq: np.ndarray,
                     n_steps: int) -> np.ndarray:
    """Run the *real* simulator from the same lookback state under the
    same ``action_seq``.  Returns ``real_obs`` of shape
    ``(n_steps, obs_dim)``.

    The env is the same one used to collect the lookback (state is
    already at the right point); we just keep stepping.
    """
    obs_dim = env.obs_dim
    real = np.zeros((n_steps, obs_dim), dtype='float32')
    for kk in range(n_steps):
        ow_next, _, done, _ = env.step(action_seq[kk])
        real[kk] = ow_next[-1]
        if done:
            real[kk + 1:] = real[kk]  # pad with last seen
            break
    return real


def _convergence_stats(traj: np.ndarray, tail_frac: float = 0.2,
                        eps_std: float = 0.05) -> Dict[str, float]:
    """Did the trajectory converge?

    Looks at the last ``tail_frac`` of steps; reports the per-channel
    std and the mean drift between halves of the tail.  ``converged``
    is True if max-channel-std in the tail < ``eps_std``.
    """
    T = traj.shape[0]
    tail = traj[-max(2, int(T * tail_frac)):]                   # (Tt, D)
    half = tail.shape[0] // 2
    return {
        'tail_std_max': float(tail.std(axis=0).max()),
        'tail_std_mean': float(tail.std(axis=0).mean()),
        'tail_drift': float(np.abs(tail[half:].mean(0)
                                    - tail[:half].mean(0)).max()),
        'final': tail[-1].tolist(),
        'converged': bool(tail.std(axis=0).max() < eps_std),
    }


def _quiet_env(env) -> None:
    """Disable all stochastic sources on a constructed APCEnv in-place.

    The diagnostic is supposed to probe a *deterministic* steady-state
    response (WM convergence under constant action when nothing else
    is moving).  ``APCEnv.reset(exploration=False)`` would otherwise
    schedule curriculum disturbances ~88% of the time and the underlying
    ``SimNoiseWrapper`` keeps applying OU + measurement noise on every
    step.  Both contaminate the convergence signal.

    This helper zeroes the noise sources on the sim wrapper and disables
    the ``DomainRandomizer`` so subsequent ``env.reset()`` / ``env.step()``
    calls produce a fully deterministic noise-free trajectory.  Callers
    must additionally clear ``env._schedule = []`` after every
    ``env.reset()`` (see ``_run_protocol``) because ``reset()`` rebuilds
    the schedule from the curriculum.
    """
    sim = env.sim
    # Wipe OU + measurement noise channels on the SimNoiseWrapper.
    if hasattr(sim, '_ou_sources'):
        sim._ou_sources = []
    if hasattr(sim, '_meas_noise'):
        sim._meas_noise = []
    if hasattr(sim, '_has_noise'):
        sim._has_noise = False
    # Disable domain randomization so plant tau / gain stay at base.
    rd = getattr(sim, '_randomizer', None)
    if rd is None:
        # SimNoiseWrapper proxies most attribute access to the inner sim;
        # try the inner sim explicitly in case the wrapper does not
        # surface ``_randomizer`` directly.
        inner = getattr(sim, '_sim', None)
        if inner is not None:
            rd = getattr(inner, '_randomizer', None)
    if rd is not None and hasattr(rd, 'frac'):
        rd.enabled = False
        rd.frac = 0.0
    # Disable hidden OU CV disturbance for the probe — the diagnostic
    # measures WM convergence under deterministic dynamics.
    if hasattr(env, '_disturbance_prob_override'):
        env._disturbance_prob_override = 0.0
    if hasattr(env, '_hidden_disturbance_force'):
        env._hidden_disturbance_force = False
    if hasattr(env, '_hidden_disturbance'):
        env._hidden_disturbance = None


def _run_protocol(env, model, cfg, device: torch.device,
                   rng: np.random.Generator, *,
                   protocol: str, n_starts: int, lookback_steps: int,
                   horizon: int, k_max: int) -> Dict:
    """Run one rollout protocol and aggregate stats over ``n_starts``."""
    L = lookback_steps
    obs_dim = env.obs_dim
    action_dim = env.action_dim
    # (c) value-equivalence: also evaluate convergence / MAE truncated to the
    # TRAINING horizon H, where the policy actually queries the WM.  The full
    # ``horizon`` (≫ H) probe is a structural drift signal, not the metric the
    # cascade depends on; reporting at H removes the false-alarm 0% headline.
    H_train = int(max(2, min(int(getattr(cfg, 'horizon', 15)), horizon)))

    # Collect a single long real episode under small noise to seed
    # the lookback windows for all starts.
    seed_sigma = 0.05
    env.reset(exploration=False)
    env._schedule = []  # see ``_quiet_env`` rationale; defensive clear
    env._hidden_disturbance = None  # same rationale
    ep_obs, ep_act = [], []
    for _ in range(L + n_starts * 4):
        a = rng.normal(0.0, seed_sigma, size=(action_dim,)).astype('float32')
        np.clip(a, -1.0, 1.0, out=a)
        ow_next, _, done, _ = env.step(a)
        ep_obs.append(ow_next[-1].copy())
        ep_act.append(a.copy())
        if done:
            break
    ep_obs = np.asarray(ep_obs, dtype='float32')
    ep_act = np.asarray(ep_act, dtype='float32')
    T = ep_obs.shape[0]
    if T < L + 4:
        return {'error': f'seed episode too short: {T} < L+4={L+4}'}

    starts = rng.integers(L, max(L + 1, T - 1), size=int(n_starts))

    per_start_records: List[Dict] = []
    for sidx, s in enumerate(starts):
        s = int(s)
        lookback_obs = ep_obs[s - L:s].copy()
        lookback_act = ep_act[s - L:s].copy()

        # Build the action sequence for the requested protocol.
        if protocol == 'replay':
            # Re-run the seed-noise pattern (small N(0, 0.05)) starting
            # from the same state. The lookback already used small noise
            # so this is the "natural continuation" of the trajectory.
            act_seq = rng.normal(0.0, seed_sigma,
                                  size=(horizon, action_dim)).astype('float32')
            np.clip(act_seq, -1.0, 1.0, out=act_seq)
        elif protocol == 'zero':
            act_seq = np.zeros((horizon, action_dim), dtype='float32')
        elif protocol == 'constant':
            a_const = rng.uniform(-0.5, 0.5,
                                   size=(action_dim,)).astype('float32')
            act_seq = np.tile(a_const, (horizon, 1))
        else:
            raise ValueError(f'unknown protocol: {protocol}')

        # ---- WM imagination ----
        if _is_rssm_model(model):
            pred_obs = _imagine_open_loop_rssm(
                model, lookback_obs, lookback_act, act_seq, horizon, device)
        else:
            with torch.no_grad():
                z_hist = model.tokenizer.encode(
                    torch.from_numpy(lookback_obs).to(device))         # (L, z)
                a_hist = torch.from_numpy(lookback_act).to(device)     # (L, A)
            pred_obs = _imagine_open_loop(model, z_hist, a_hist,
                                           act_seq, horizon, k_max, device)

        # ---- Real-sim ground truth: re-create env at same lookback state ----
        # The simulator is deterministic given action sequence; we
        # need to fast-forward a fresh env to the same start state.
        # Cheapest way: reset the existing env and replay the entire
        # seed-noise prefix up to start ``s`` so the sim is at the
        # same internal state, then step ``horizon`` steps with
        # ``act_seq``.
        env.reset(exploration=False)
        env._schedule = []  # defensive: keep the probe disturbance-free
        env._hidden_disturbance = None
        for t in range(s):
            env.step(ep_act[t])
        real_obs = _real_open_loop(env, cfg, lookback_obs, lookback_act,
                                    act_seq, horizon)

        # ---- Metrics ----
        err_per_step = np.abs(pred_obs - real_obs).mean(axis=1)     # (horizon,)
        per_start_records.append({
            'start_idx': s,
            'protocol': protocol,
            'mae_step1': float(err_per_step[0]),
            'mae_step5': float(err_per_step[min(4, horizon - 1)]),
            'mae_step15': float(err_per_step[min(14, horizon - 1)]),
            'mae_step42': float(err_per_step[min(41, horizon - 1)])
                            if horizon > 41 else None,
            'mae_final': float(err_per_step[-1]),
            'mae_at_H_train': float(err_per_step[min(H_train - 1, horizon - 1)]),
            'pred_convergence': _convergence_stats(pred_obs),
            'real_convergence': _convergence_stats(real_obs),
            'pred_convergence_at_H_train': _convergence_stats(pred_obs[:H_train]),
            'real_convergence_at_H_train': _convergence_stats(real_obs[:H_train]),
            'pred_final': pred_obs[-1].tolist(),
            'real_final': real_obs[-1].tolist(),
            'pred_traj': pred_obs.astype('float32').tolist(),
            'real_traj': real_obs.astype('float32').tolist(),
            'action_seq': act_seq.astype('float32').tolist(),
        })

    # Aggregate.
    def _agg(key):
        vals = [r[key] for r in per_start_records if r.get(key) is not None]
        return float(np.mean(vals)) if vals else None

    pred_conv_rate = float(np.mean(
        [1.0 if r['pred_convergence']['converged'] else 0.0
         for r in per_start_records]))
    real_conv_rate = float(np.mean(
        [1.0 if r['real_convergence']['converged'] else 0.0
         for r in per_start_records]))
    pred_conv_rate_H = float(np.mean(
        [1.0 if r['pred_convergence_at_H_train']['converged'] else 0.0
         for r in per_start_records]))
    real_conv_rate_H = float(np.mean(
        [1.0 if r['real_convergence_at_H_train']['converged'] else 0.0
         for r in per_start_records]))
    ss_err = float(np.mean(
        [np.abs(np.asarray(r['pred_final'])
                 - np.asarray(r['real_final'])).mean()
         for r in per_start_records]))

    return {
        'protocol': protocol,
        'n_starts': len(per_start_records),
        'horizon': horizon,
        'horizon_train': H_train,
        'agg': {
            'mae_step1': _agg('mae_step1'),
            'mae_step5': _agg('mae_step5'),
            'mae_step15': _agg('mae_step15'),
            'mae_step42': _agg('mae_step42'),
            'mae_final': _agg('mae_final'),
            'mae_at_H_train': _agg('mae_at_H_train'),
            'steady_state_err_mean': ss_err,
            'pred_convergence_rate': pred_conv_rate,
            'real_convergence_rate': real_conv_rate,
            'pred_convergence_rate_at_H_train': pred_conv_rate_H,
            'real_convergence_rate_at_H_train': real_conv_rate_H,
        },
        'per_start': per_start_records,
    }


# ---------------------------------------------------------------------------
# Top-level entry
# ---------------------------------------------------------------------------

def run_wm_steady_state_diagnostic(run_dir: Path,
                                     ckpt_name: Optional[str] = None,
                                     n_starts: int = 8,
                                     horizon: int = 200,
                                     seed: int = 20260520,
                                     protocols: Optional[Tuple[str, ...]] = None,
                                     noise_free: bool = True,
                                     output_dir: Optional[Path] = None,
                                     ) -> Dict:
    """Run the diagnostic; write JSON (+ plot if matplotlib available).

    Args:
        output_dir: directory the ``wm_steady_state_diagnostic.{json,png}``
            are written to.  Defaults to ``run_dir`` (legacy/CLI).  The inline
            training caller passes ``run_dir/'validation'`` so the WM probes
            live alongside the other validation artifacts (P89).  The ckpt is
            still located under ``run_dir``.
        noise_free: when True (default), zero out all stochastic sources
            on the env (OU process noise, measurement noise, curriculum
            disturbances, domain randomization) so the probe measures a
            deterministic steady-state response.  Pass ``False`` to keep
            the noise/disturbance regime that the WM was trained under
            (legacy behaviour; results are then a mix of WM extrapolation
            error and stochastic plant variance and should not be read as
            a pure WM-quality metric).

    Returns the result dict.
    """
    run_dir = Path(run_dir)
    out_dir = Path(output_dir) if output_dir is not None else run_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    device, dev_reason = _pick_device()
    ckpt_path = _find_ckpt(run_dir, ckpt_name)
    print(f'[wm-ss-diag] ckpt={ckpt_path.name}  device={device}  ({dev_reason})',
          flush=True)

    from training.train import APCEnv
    model, cfg, obs_norm = _load_model(ckpt_path, device)
    k_max = int(cfg.k_max)
    L = int(cfg.lookback)
    print(f'[wm-ss-diag] cfg: lookback={L} k_max={k_max} horizon_train={cfg.horizon} '
          f'obs_dim={cfg.obs_dim} action_dim={cfg.action_dim}',
          flush=True)

    rng = np.random.default_rng(int(seed))
    env = APCEnv(cfg, rng)
    if noise_free:
        _quiet_env(env)
        print('[wm-ss-diag] noise_free=True: OU + measurement noise + DR + '
              'disturbance schedule disabled on env', flush=True)
    if obs_norm is not None:
        try:
            env.set_obs_norm_stats(
                mean=np.asarray(obs_norm.get('mean')),
                var=np.asarray(obs_norm.get('var')),
                count=float(obs_norm.get('count', 1.0)),
                learn=False)
        except Exception as e:
            print(f'[wm-ss-diag] obs_norm restore skipped: {e!r}', flush=True)
    # Reward-scale doesn't matter for prediction-accuracy probing, but set
    # it for consistency with how the model was trained.
    cal_path = run_dir / 'reward_calibration.json'
    if cal_path.exists():
        try:
            with open(cal_path) as f:
                env.reward_scale = float(json.load(f).get('reward_scale', 1.0))
        except Exception:
            pass

    if protocols is None:
        protocols = ('replay', 'zero', 'constant')
    results: Dict[str, Dict] = {}
    for protocol in protocols:
        print(f'[wm-ss-diag] running protocol={protocol} '
              f'n_starts={n_starts} horizon={horizon} ...', flush=True)
        results[protocol] = _run_protocol(
            env, model, cfg, device, rng,
            protocol=protocol, n_starts=n_starts,
            lookback_steps=L, horizon=horizon, k_max=k_max)

    # Build top-level verdict.
    H_train = int(cfg.horizon)
    verdict = {
        'horizon_train': H_train,
        'noise_free': bool(noise_free),
        'horizon_probe': horizon,
        'device': str(device),
        'device_reason': dev_reason,
        'ckpt': ckpt_path.name,
        'per_protocol_mae_at_H_train': {
            p: results[p]['agg'].get(f'mae_step{H_train}',
                                       results[p]['agg'].get('mae_final'))
            for p in results
        },
        'steady_state_err_zero_action':
            results.get('zero', {}).get('agg', {}).get('steady_state_err_mean'),
        'steady_state_err_constant_action':
            results.get('constant', {}).get('agg', {}).get('steady_state_err_mean'),
        'wm_pred_converges_under_zero_action':
            results.get('zero', {}).get('agg', {}).get('pred_convergence_rate'),
        'wm_pred_converges_under_constant_action':
            results.get('constant', {}).get('agg', {}).get('pred_convergence_rate'),
        # (c) value-equivalence: convergence judged at the TRAINING horizon H
        # (where the policy queries the WM) — the cascade-relevant metric.
        # The full-horizon rates above remain a structural drift signal.
        'wm_pred_converges_under_zero_action_at_H_train':
            results.get('zero', {}).get('agg', {})
                   .get('pred_convergence_rate_at_H_train'),
        'wm_pred_converges_under_constant_action_at_H_train':
            results.get('constant', {}).get('agg', {})
                   .get('pred_convergence_rate_at_H_train'),
        'per_protocol_mae_at_H_train_step':
            {p: results[p]['agg'].get('mae_at_H_train') for p in results},
    }
    out = {'verdict': verdict, 'results': results}
    out_path = out_dir / 'wm_steady_state_diagnostic.json'
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'[wm-ss-diag] wrote {out_path}', flush=True)

    # Optional plot.
    try:
        _save_plot(out_dir, out)
    except Exception as e:
        print(f'[wm-ss-diag] plot skipped: {e!r}', flush=True)

    _print_summary(verdict, results)
    return out


def _save_plot(run_dir: Path, out: Dict) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    protos = [p for p in ('replay', 'zero', 'constant')
              if p in out['results']]
    if not protos:
        return
    fig, axes = plt.subplots(len(protos), 1, figsize=(11, 3 * len(protos)),
                               sharex=True, squeeze=False)
    axes = axes[:, 0]
    for ax, protocol in zip(axes, protos):
        r = out['results'].get(protocol, {})
        per = r.get('per_start', [])
        if not per:
            continue
        # Plot CV channel (channel 0 by convention) for each start.
        for rec in per[:4]:  # avoid overplotting
            pred = np.asarray(rec['pred_traj'])
            real = np.asarray(rec['real_traj'])
            ax.plot(real[:, 0], color='C0', alpha=0.6,
                     label='real (sim)' if rec is per[0] else None)
            ax.plot(pred[:, 0], color='C3', alpha=0.6, ls='--',
                     label='imagined (WM)' if rec is per[0] else None)
        ax.set_title(f'{protocol}: CV ch0 over {r["horizon"]}-step rollout '
                      f'(H_train={out["verdict"]["horizon_train"]})')
        ax.set_ylabel('obs[0]')
        ax.grid(alpha=0.3)
        ax.axvline(out['verdict']['horizon_train'], color='gray',
                    lw=0.5, ls=':', label='H_train')
        if protocol == 'replay':
            ax.legend(fontsize=8, loc='best')
    axes[-1].set_xlabel('imagination step')
    fig.suptitle(f'WM steady-state diagnostic: {run_dir.name}',
                  fontsize=10)
    fig.tight_layout()
    out_png = run_dir / 'wm_steady_state_diagnostic.png'
    fig.savefig(out_png, dpi=110)
    plt.close(fig)
    print(f'[wm-ss-diag] wrote {out_png}', flush=True)


def _print_summary(verdict: Dict, results: Dict) -> None:
    print()
    print('=' * 72)
    print(f'WM steady-state diagnostic — ckpt={verdict["ckpt"]}  '
          f'device={verdict["device"]}')
    print('=' * 72)
    H = verdict['horizon_train']
    print(f'  training imagination horizon H = {H}')
    print(f'  probe horizon                  = {verdict["horizon_probe"]}')
    print()
    print('  per-protocol per-step MAE (lower=better):')
    print('  protocol  | step1   step5   step15  step42  final')
    for p in ('replay', 'zero', 'constant'):
        a = results.get(p, {}).get('agg', {})
        if not a:
            continue
        def _fmt(v): return f'{v:7.4f}' if v is not None else '   n/a '
        print(f'  {p:9s} | {_fmt(a.get("mae_step1"))} {_fmt(a.get("mae_step5"))} '
              f'{_fmt(a.get("mae_step15"))} {_fmt(a.get("mae_step42"))} '
              f'{_fmt(a.get("mae_final"))}')
    print()
    print('  steady-state behaviour (the APC-specific test):')
    for p in ('zero', 'constant'):
        a = results.get(p, {}).get('agg', {})
        if not a:
            continue
        ss = a.get('steady_state_err_mean')
        wm_conv = a.get('pred_convergence_rate', 0.0)
        sim_conv = a.get('real_convergence_rate', 0.0)
        wm_conv_H = a.get('pred_convergence_rate_at_H_train', 0.0) or 0.0
        print(f'  {p:9s} action: WM converged in {wm_conv*100:5.1f}% of starts '
              f'(@H={H}: {wm_conv_H*100:5.1f}%), '
              f'sim converged in {sim_conv*100:5.1f}%, '
              f'steady-state err = {ss:.4f}' if ss is not None else '')
    print()
    # Verdict heuristic.  Judged at the TRAINING horizon H (value-equivalence:
    # the WM only needs to be accurate where the policy queries it); the full-
    # horizon rate is reported above as a structural-drift signal.
    ss_zero = verdict.get('steady_state_err_zero_action') or float('inf')
    wm_conv_zero = (verdict.get('wm_pred_converges_under_zero_action_at_H_train')
                    or verdict.get('wm_pred_converges_under_zero_action') or 0.0)
    if wm_conv_zero >= 0.8 and ss_zero < 0.2:
        print('  VERDICT: WM steady-state representation HEALTHY '
              '(converges, low SS error)')
    elif wm_conv_zero >= 0.5:
        print('  VERDICT: WM converges but with noticeable SS bias '
              '(check controller robustness)')
    else:
        print('  VERDICT: WM imagined trajectories DO NOT CONVERGE '
              'under constant action — actor sees only transient '
              'dynamics in long imagined rollouts')
    print('=' * 72)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--run-dir', required=True, type=Path)
    ap.add_argument('--ckpt', default=None,
                     help='checkpoint filename inside run-dir '
                          '(default: final.pt > best.pt > latest ckpt_iter)')
    ap.add_argument('--n-starts', type=int, default=8)
    ap.add_argument('--horizon', type=int, default=200,
                     help='imagination horizon for the probe (env steps); '
                          'should be ≫ training horizon and ≥ several τ '
                          'to test steady-state convergence')
    ap.add_argument('--seed', type=int, default=20260520)
    ap.add_argument('--protocols', default='replay,zero,constant',
                     help='comma-separated subset of {replay,zero,constant}')
    ap.add_argument('--with-noise', action='store_true',
                     help='keep OU + measurement noise + curriculum '
                          'disturbances + domain randomization active during '
                          'the probe (legacy behaviour). Default is a fully '
                          'noise-free deterministic probe.')
    args = ap.parse_args()
    protocols = tuple(p.strip() for p in args.protocols.split(',') if p.strip())
    run_wm_steady_state_diagnostic(args.run_dir, ckpt_name=args.ckpt,
                                    n_starts=args.n_starts,
                                    horizon=args.horizon, seed=args.seed,
                                    protocols=protocols,
                                    noise_free=not args.with_noise)


if __name__ == '__main__':
    _cli()
