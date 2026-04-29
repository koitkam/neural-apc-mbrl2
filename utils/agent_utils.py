"""Shared utilities for agent training scripts and validation.

Consolidates functions across all agent training and validation scripts.
"""
import glob
import os

import numpy as np
import torch

from models.observers import MambaLikeObserver, TransformerObserver
from models.export_onnx import export_policy_onnx
from utils.objective_config import load_objective_spec
from utils.objective_runtime import compute_objective_components
from utils.time_sampling import (
    mask_window_by_feature_lookback,
    sample_history_window_feature_scan,
)


# ---------------------------------------------------------------------------
# Observer helpers
# ---------------------------------------------------------------------------

def build_observer_from_config(cfg: dict, device: torch.device = None):
    """Instantiate an observer model from a saved config dict."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if cfg.get('observer_type') == 'transformer':
        model = TransformerObserver(
            state_dim=cfg['state_dim'],
            lookback=cfg['lookback'],
            latent_dim=cfg['latent_dim'],
            d_model=cfg['d_model'],
            num_heads=cfg['num_heads'],
            num_layers=cfg['num_layers'],
            ff_dim=cfg['ff_dim'],
            dropout=0.1,
        )
    else:
        model = MambaLikeObserver(
            state_dim=cfg['state_dim'],
            lookback=cfg['lookback'],
            latent_dim=cfg['latent_dim'],
            d_model=cfg['d_model'],
            num_layers=cfg['num_layers'],
            kernel_size=cfg['kernel_size'],
        )
    model.to(device)
    model.eval()
    return model


def latest_observer_dir(project_dir: str, variant: str = 'ssm_mlp') -> str:
    """Find the most recent observer directory."""
    if variant in ('ssm_mlp', 'ssm'):
        paths = glob.glob(os.path.join(project_dir, 'OBSERVER_SSM*'))
        paths += glob.glob(os.path.join(project_dir, 'OBSERVER_TRANSFORMER*'))
    else:
        paths = glob.glob(os.path.join(project_dir, 'OBSERVER_TRANSFORMER*'))
    paths = [p for p in paths if os.path.isdir(p)]
    if not paths:
        prefix = 'OBSERVER_SSM* / OBSERVER_TRANSFORMER*' if variant in ('ssm_mlp', 'ssm') else 'OBSERVER_TRANSFORMER*'
        raise FileNotFoundError(f'No {prefix} directory found. Train observer first.')
    paths.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return paths[0]


@torch.inference_mode()
def encode_observation(observer, obs_window_np: np.ndarray, device: torch.device) -> np.ndarray:
    """Encode a single observation window to latent vector.

    Args:
        observer: Observer model (eval mode).
        obs_window_np: (1, lookback, state_dim) numpy array.
        device: Torch device.

    Returns:
        (latent_dim,) numpy array.
    """
    x = torch.from_numpy(obs_window_np).float().to(device)
    return observer.encode(x).cpu().numpy()[0]


# ---------------------------------------------------------------------------
# Objective helpers
# ---------------------------------------------------------------------------

def load_objective_weights():
    return load_objective_spec()['weights']


def load_objective_bounds():
    return load_objective_spec()['bounds']


def load_full_objective_spec():
    """Return the full normalised control objective spec (new-schema keys)."""
    return load_objective_spec()


def sanitize_persisted_objective_bounds(bounds):
    """Clean and validate objective bounds for JSON persistence."""
    if not isinstance(bounds, dict):
        return {}

    def _is_sane_pair(v):
        if not (isinstance(v, (list, tuple)) and len(v) >= 2):
            return False
        try:
            lo, hi = float(v[0]), float(v[1])
        except Exception:
            return False
        return np.isfinite(lo) and np.isfinite(hi) and (hi > lo) and (abs(lo) < 1e9) and (abs(hi) < 1e9)

    def _is_auto_key(k, prefix):
        s = str(k).strip().lower()
        return s.startswith(prefix + '_') and s[len(prefix) + 1:].isdigit()

    def _sanitize_map(d, prefix):
        if not isinstance(d, dict):
            return {}
        named = {k: [float(v[0]), float(v[1])] for k, v in d.items() if (not _is_auto_key(k, prefix)) and _is_sane_pair(v)}
        if named:
            return named
        return {k: [float(v[0]), float(v[1])] for k, v in d.items() if _is_sane_pair(v)}

    mv_bounds = [[float(v[0]), float(v[1])] for v in list(bounds.get('mv_bounds', [])) if _is_sane_pair(v)]
    cv_bounds = [[float(v[0]), float(v[1])] for v in list(bounds.get('cv_bounds', [])) if _is_sane_pair(v)]
    mvs = _sanitize_map(bounds.get('mvs', {}), 'mv')
    outputs = _sanitize_map(bounds.get('outputs', {}), 'cv')

    if (not mv_bounds) and mvs:
        mv_bounds = [list(v) for _, v in sorted(mvs.items())]
    if (not cv_bounds) and outputs:
        cv_bounds = [list(v) for _, v in sorted(outputs.items())]

    return {
        'mv_bounds': mv_bounds,
        'cv_bounds': cv_bounds,
        'mvs': mvs,
        'outputs': outputs,
    }


# ---------------------------------------------------------------------------
# Action / control conversion
# ---------------------------------------------------------------------------

def action_to_control(action, bounds, setpoint_manager=None):
    """Map normalised [0,1] action vector to engineering-unit MV values.

    When a ``setpoint_manager`` is provided and exposes runtime MV bounds
    (``current_mv_bounds`` of shape (n_mv, 2)), the live bounds are used
    instead of the static ``bounds['mv_bounds']`` so that operator limit
    changes are actually tracked by the controller (not only by the
    violation penalty).  Runtime bounds are clipped to the static base
    bounds so the agent can never drive the actuator past its physical
    limits.
    """
    action = np.clip(action, 0.0, 1.0)
    mv_bounds = list(bounds.get('mv_bounds', []))
    if len(mv_bounds) < len(action):
        mv_bounds = mv_bounds + [[0.0, 100.0] for _ in range(len(action) - len(mv_bounds))]
    mv_bounds = mv_bounds[:len(action)]
    lo = np.array([b[0] for b in mv_bounds], dtype='float32')
    hi = np.array([b[1] for b in mv_bounds], dtype='float32')
    if setpoint_manager is not None:
        try:
            rt = np.asarray(getattr(setpoint_manager, 'current_mv_bounds'), dtype='float32')
            if rt.shape == (len(action), 2):
                # Clip runtime bounds to static physical limits so the
                # controller cannot be driven outside the actuator envelope.
                rt_lo = np.maximum(rt[:, 0], lo)
                rt_hi = np.minimum(rt[:, 1], hi)
                # Guard against inverted bounds from misconfiguration.
                valid = rt_hi > rt_lo
                lo = np.where(valid, rt_lo, lo)
                hi = np.where(valid, rt_hi, hi)
        except Exception:
            pass
    return lo + action * (hi - lo)


# ---------------------------------------------------------------------------
# Reward functions
# ---------------------------------------------------------------------------

def ppo_reward_fn(state, sim, control, prev_control, obj_w, bounds, reward_scale=1.0, reward_clip=250.0,
                  setpoint_manager=None, objective_spec=None):
    """Scaled and clipped reward for on-policy algorithms (PPO)."""
    comp = compute_objective_components(state, sim, control, prev_control, obj_w, bounds,
                                        setpoint_manager=setpoint_manager, objective_spec=objective_spec)
    r = float(comp['reward']) * float(reward_scale)
    scaled = float(np.clip(r, -reward_clip, reward_clip)) if np.isfinite(r) else -float(reward_clip)
    return scaled, comp


def sac_reward_fn(state, sim, control, prev_control, obj_w, bounds,
                  setpoint_manager=None, objective_spec=None):
    """Raw reward for off-policy algorithms (SAC, TD3)."""
    comp = compute_objective_components(state, sim, control, prev_control, obj_w, bounds,
                                        setpoint_manager=setpoint_manager, objective_spec=objective_spec)
    r = float(comp['reward'])
    if not np.isfinite(r):
        r = -100.0
    return r, comp


# ---------------------------------------------------------------------------
# GAE (PPO only)
# ---------------------------------------------------------------------------

def compute_gae(rewards, values, dones, next_value, gamma=0.99, lam=0.95):
    n = len(rewards)
    adv = np.zeros(n, dtype='float32')
    last = 0.0
    for t in reversed(range(n)):
        nonterminal = 1.0 - dones[t]
        next_v = next_value if t == n - 1 else values[t + 1]
        delta = rewards[t] + gamma * next_v * nonterminal - values[t]
        last = delta + gamma * lam * nonterminal * last
        adv[t] = last
    returns = adv + values
    return returns, adv


# ---------------------------------------------------------------------------
# SAC helpers
# ---------------------------------------------------------------------------

def soft_update(target: torch.nn.Module, source: torch.nn.Module, tau: float):
    """Polyak-average update of target network parameters."""
    for tp, sp in zip(target.parameters(), source.parameters()):
        tp.data.copy_((1.0 - tau) * tp.data + tau * sp.data)


# ---------------------------------------------------------------------------
# Action-feedback history helpers
# ---------------------------------------------------------------------------

def extract_mv_temporal_vectors(feature_lookbacks, feature_sample_rates, mv_indices, default_lookback, default_sample_rate):
    mv_lbs, mv_srs = [], []
    for idx in list(mv_indices):
        mv_lbs.append(int(feature_lookbacks[int(idx)]) if 0 <= int(idx) < len(feature_lookbacks) else int(default_lookback))
        mv_srs.append(int(feature_sample_rates[int(idx)]) if 0 <= int(idx) < len(feature_sample_rates) else int(default_sample_rate))
    return np.asarray(mv_lbs, dtype='int32'), np.asarray(mv_srs, dtype='int32')


def init_action_history(action, lookback_horizon):
    return np.repeat(np.asarray(action, dtype='float32')[None, :], int(lookback_horizon), axis=0)


def update_action_history(history, action):
    history[:-1] = history[1:]
    history[-1] = np.asarray(action, dtype='float32')


def build_action_feedback(raw_action_hist, sample_rate, sampled_lookback, mv_lookbacks, mv_sample_rates):
    act_hist = sample_history_window_feature_scan(
        raw_action_hist, base_sample_rate=sample_rate, output_len=sampled_lookback,
        feature_sample_rates=mv_sample_rates,
    )
    act_hist = mask_window_by_feature_lookback(act_hist, mv_lookbacks)
    return act_hist.reshape(-1).astype('float32')
