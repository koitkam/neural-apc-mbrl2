"""DreamerV4 trainer for APC — paper-faithful, single algorithm.

Reference: Hafner et al. (2024), arXiv:2407.04693.

Training loop:
  1. Collect ``EP_PER_ITER`` episodes from the env using the current actor
     (sampled / stochastic actions in [-1, 1]).
  2. With probability ``EXPLORATION_RATIO``, also collect episodes with
     uniform-random actions (under the same disturbance schedule) into a
     separate exploration buffer.
  3. Train ``TRAIN_STEPS_PER_ITER`` updates: each update samples a
     50/50 mix of policy / exploration sequences, runs a WM update, then
     an actor-critic update in latent imagination (horizon ``H``).
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from models.dreamer_v4 import (
    DreamerV4, DreamerV4Config, RSSMConfig,
    free_bits_kl, symlog,
)
from utils.sim_factory import create_sim, resolve_sim_metadata
from utils.objective_runtime import compute_objective_components
from utils.runtime_setpoints import RuntimeSetpointManager, RuntimeSetpointConfig
from utils.training_disturbance import build_training_disturbance_schedule
from utils.agent_utils import (
    load_objective_weights, load_objective_bounds, load_full_objective_spec,
    action_to_control,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    # Architecture (model_size preset is set by BO; defaults here = M)
    deter_dim: int = 512
    embed_dim: int = 512
    hidden_dim: int = 512
    n_categoricals: int = 32
    n_classes: int = 32

    # Plant / windowing
    lookback: int = 32
    sample_rate: int = 5
    episode_length: int = 600

    # Training
    total_steps: int = 100_000
    train_steps_per_iter: int = 100
    ep_per_iter: int = 5
    seq_len: int = 64
    batch_size: int = 16
    horizon: int = 15

    # Optimizers
    lr_world: float = 1e-4
    lr_actor: float = 3e-5
    lr_critic: float = 3e-5
    grad_clip: float = 1000.0

    # Loss weights
    free_nats: float = 1.0
    kl_scale: float = 1.0
    recon_scale: float = 1.0
    reward_scale: float = 1.0
    cont_scale: float = 1.0
    actor_entropy_scale: float = 3e-4

    # Returns
    gamma: float = 0.997
    gae_lambda: float = 0.95
    target_critic_tau: float = 0.02

    # Buffers
    buffer_capacity_steps: int = 200_000
    explore_buffer_ratio: float = 0.5     # 50/50 mix in WM batches
    explore_episode_ratio: float = 0.25   # 1 in 4 collected episodes is random-action

    # I/O
    out_dir: str = ''
    log_every: int = 1
    save_every_iters: int = 20

    # Resolved at build-time (set by builder)
    obs_dim: int = 0
    action_dim: int = 0
    aug_obs_dim: int = 0
    state_dim: int = 0
    n_action_bins: int = 21


# ---------------------------------------------------------------------------
# Env wrapper — stacks state + aug-obs, builds lookback window, computes reward
# ---------------------------------------------------------------------------

class APCEnv:
    """Slim env wrapper around the carryover simulator.

    Step output is ``(obs_window, reward, done, info)`` where:

      - ``obs_window`` is shape ``(lookback, obs_dim)`` with most recent step last.
      - ``obs_dim = state_dim + aug_obs_dim``.
      - ``done`` is True at end of episode.
    """

    def __init__(self, cfg: TrainConfig, rng: np.random.Generator):
        self.cfg = cfg
        self.rng = rng
        self.sim = create_sim(episode_length=cfg.episode_length,
                              sample_rate=cfg.sample_rate)
        self.meta = resolve_sim_metadata(self.sim)
        self.action_dim = int(self.meta['action_dim'])
        self.state_dim = int(self.meta['state_dim'] or 0)
        self.cv_indices = list(self.meta['cv_indices'])
        self.mv_norm_ranges = list(self.meta['mv_normalization_ranges'])
        self.cv_norm_ranges = list(self.meta['cv_normalization_ranges'])

        self.bounds = load_objective_bounds() or {}
        self.obj_w = load_objective_weights() or {}
        self.obj_spec = load_full_objective_spec() or {}

        # Setpoint manager — packed aug-obs.
        mvs_raw = self.bounds.get('mvs')
        if isinstance(mvs_raw, dict) and mvs_raw:
            mvb = np.asarray([mvs_raw[k] for k in sorted(mvs_raw)],
                             dtype='float32').reshape(-1, 2)
        elif isinstance(mvs_raw, (list, tuple)) and len(mvs_raw) > 0:
            mvb = np.asarray(mvs_raw, dtype='float32').reshape(-1, 2)
        else:
            mv_bounds_list = list(self.bounds.get('mv_bounds', []))
            if mv_bounds_list:
                mvb = np.asarray(mv_bounds_list, dtype='float32').reshape(-1, 2)
            else:
                mvb = np.tile(np.array([[0.0, 100.0]], dtype='float32'),
                              (self.action_dim, 1))
        cv_bounds_raw = self.bounds.get('outputs') or self.bounds.get('cvs') or {}
        if isinstance(cv_bounds_raw, dict) and cv_bounds_raw:
            cvb = np.asarray([cv_bounds_raw[k] for k in sorted(cv_bounds_raw)],
                             dtype='float32').reshape(-1, 2)
        elif isinstance(cv_bounds_raw, (list, tuple)) and len(cv_bounds_raw) > 0:
            cvb = np.asarray(cv_bounds_raw, dtype='float32').reshape(-1, 2)
        else:
            cv_bounds_list = list(self.bounds.get('cv_bounds', []))
            if cv_bounds_list:
                cvb = np.asarray(cv_bounds_list, dtype='float32').reshape(-1, 2)
            else:
                cvb = np.tile(np.array([[0.0, 100.0]], dtype='float32'),
                              (max(1, len(self.cv_indices)), 1))
        n_cv = cvb.shape[0]

        # Targets from objective spec.
        rt_spec = (self.obj_spec or {}).get('runtime_setpoints', {}) or {}
        targets_enabled_spec = rt_spec.get('targets_enabled', False)
        if isinstance(targets_enabled_spec, bool):
            cv_target_enabled = np.array([targets_enabled_spec] * n_cv, dtype=bool)
        else:
            te = list(targets_enabled_spec) + [False] * n_cv
            cv_target_enabled = np.array(te[:n_cv], dtype=bool)
        cv_targets = np.array(
            [0.5 * (cvb[i, 0] + cvb[i, 1]) for i in range(n_cv)], dtype='float32')

        self.setpoint_mgr = RuntimeSetpointManager(
            n_mv=self.action_dim, n_cv=n_cv,
            base_mv_bounds=mvb,
            base_cv_bounds=cvb,
            base_cv_targets=cv_targets,
            cv_target_enabled=cv_target_enabled,
            mv_norm_bounds=np.asarray(self.mv_norm_ranges, dtype='float32')
                if self.mv_norm_ranges else np.zeros((0, 2), dtype='float32'),
            cv_norm_bounds=np.asarray(self.cv_norm_ranges, dtype='float32')
                if self.cv_norm_ranges else np.zeros((0, 2), dtype='float32'),
        )
        self.setpoint_mgr._rng = rng

        self.aug_obs_dim = int(self.setpoint_mgr.aug_obs_dim)
        self.obs_dim = self.state_dim + self.aug_obs_dim

        self._window: Optional[np.ndarray] = None
        self._t = 0
        self._prev_control = np.zeros(self.action_dim, dtype='float32')
        self._schedule: List[Dict] = []
        # Reward scale (auto-calibrated in train(); see calibrate_reward_scale).
        self.reward_scale: float = 1.0

    # ---- helpers ----
    def _build_obs_vec(self, state: np.ndarray) -> np.ndarray:
        aug = self.setpoint_mgr.get_augmented_obs_channels()
        return np.concatenate([np.asarray(state, dtype='float32').reshape(-1), aug], axis=0)

    # ---- API ----
    def reset(self, *, exploration: bool = False) -> np.ndarray:
        state = self.sim.reset()
        if isinstance(state, tuple):
            state = state[0]
        state = np.asarray(state, dtype='float32').reshape(-1)
        self._t = 0
        self._prev_control = np.zeros(self.action_dim, dtype='float32')
        # Curriculum fraction is the global progress; pass 1.0 for now and
        # let the BO/outer loop tune episode length. This keeps the trainer
        # paper-aligned (no auto-tuning band-aids inside the loop).
        self.setpoint_mgr.reset(episode_length=self.cfg.episode_length,
                                curriculum_fraction=1.0)
        intensity = 1.0 if not exploration else 1.2
        self._schedule = build_training_disturbance_schedule(
            episode_length=self.cfg.episode_length,
            rng=self.rng,
            intensity=intensity,
            sim=self.sim,
        )
        # Apply scheduled DV/CV events through the sim's interfaces (if any).
        # Disturbance step injection happens inside _apply_disturbance() per step.
        obs_vec = self._build_obs_vec(state)
        self._window = np.tile(obs_vec, (self.cfg.lookback, 1)).astype('float32')
        return self._window.copy()

    def _apply_disturbance(self) -> None:
        """Apply scheduled disturbance events at current step.

        Many simulators expose ``set_disturbance(channel, value)`` or accept
        DV via reset; here we keep the trainer simulator-agnostic by calling
        ``sim.apply_disturbance(t, schedule)`` if available.  Otherwise the
        schedule is informational only (the SimNoiseWrapper handles
        measurement / actuator noise).
        """
        fn = getattr(self.sim, 'apply_disturbance', None)
        if callable(fn):
            try:
                fn(self._t, self._schedule)
            except Exception:
                pass

    def step(self, action_norm: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        action_norm = np.asarray(action_norm, dtype='float32').reshape(self.action_dim)
        # V4 actor emits actions in [-1, 1]; action_to_control expects [0, 1].
        action_01 = 0.5 * (np.clip(action_norm, -1.0, 1.0) + 1.0)
        control = action_to_control(action_01, self.bounds, self.setpoint_mgr)
        self._apply_disturbance()
        # Also update setpoint manager schedule (limit/target changes).
        self.setpoint_mgr.step(self._t)
        next_state = self.sim.step(control)
        if isinstance(next_state, tuple):
            # Some sims return (state, reward, done, info) — discard sim's reward.
            next_state = next_state[0]
        next_state = np.asarray(next_state, dtype='float32').reshape(-1)
        comps = compute_objective_components(
            state=next_state, sim=self.sim,
            control=control, prev_control=self._prev_control,
            obj_w=self.obj_w, bounds=self.bounds,
            setpoint_manager=self.setpoint_mgr,
            objective_spec=self.obj_spec,
        )
        raw_reward = float(comps['reward'])
        reward = raw_reward * float(self.reward_scale)
        self._prev_control = np.asarray(control, dtype='float32')
        self._t += 1
        done = self._t >= self.cfg.episode_length
        # Roll the window: drop oldest, append newest.
        obs_vec = self._build_obs_vec(next_state)
        self._window = np.concatenate([self._window[1:], obs_vec[None, :]], axis=0)
        info = {'reward_components': comps, 't': self._t,
                'raw_reward': raw_reward}
        return self._window.copy(), reward, done, info


# ---------------------------------------------------------------------------
# Replay buffer — stores per-step (obs_window, action, reward, cont)
# ---------------------------------------------------------------------------

class TrajectoryBuffer:
    """Episode-major buffer. Stores complete episodes, samples sequences."""

    def __init__(self, capacity_eps: int, episode_length: int,
                 lookback: int, obs_dim: int, action_dim: int):
        self.capacity_eps = int(capacity_eps)
        self.T = int(episode_length)
        self.L = int(lookback)
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.obs = np.zeros((capacity_eps, self.T, self.L, self.obs_dim), dtype='float32')
        self.act = np.zeros((capacity_eps, self.T, self.action_dim), dtype='float32')
        self.rew = np.zeros((capacity_eps, self.T), dtype='float32')
        self.cont = np.ones((capacity_eps, self.T), dtype='float32')
        self.filled = 0
        self.write = 0

    def add_episode(self, obs: np.ndarray, act: np.ndarray, rew: np.ndarray,
                    cont: np.ndarray) -> None:
        T = obs.shape[0]
        assert T == self.T, f"episode length mismatch: {T} vs {self.T}"
        i = self.write
        self.obs[i] = obs
        self.act[i] = act
        self.rew[i] = rew
        self.cont[i] = cont
        self.write = (self.write + 1) % self.capacity_eps
        self.filled = min(self.filled + 1, self.capacity_eps)

    def sample(self, batch_size: int, seq_len: int, rng: np.random.Generator
               ) -> Dict[str, np.ndarray]:
        if self.filled == 0:
            raise ValueError("empty buffer")
        ep_idx = rng.integers(0, self.filled, size=batch_size)
        max_start = self.T - seq_len
        if max_start <= 0:
            starts = np.zeros(batch_size, dtype=np.int64)
        else:
            starts = rng.integers(0, max_start + 1, size=batch_size)
        out_obs = np.zeros((batch_size, seq_len, self.L, self.obs_dim), dtype='float32')
        out_act = np.zeros((batch_size, seq_len, self.action_dim), dtype='float32')
        out_rew = np.zeros((batch_size, seq_len), dtype='float32')
        out_cont = np.zeros((batch_size, seq_len), dtype='float32')
        for b in range(batch_size):
            s = starts[b]
            out_obs[b] = self.obs[ep_idx[b], s:s + seq_len]
            out_act[b] = self.act[ep_idx[b], s:s + seq_len]
            out_rew[b] = self.rew[ep_idx[b], s:s + seq_len]
            out_cont[b] = self.cont[ep_idx[b], s:s + seq_len]
        return {'obs': out_obs, 'act': out_act, 'rew': out_rew, 'cont': out_cont}


# ---------------------------------------------------------------------------
# Episode collection
# ---------------------------------------------------------------------------

@torch.no_grad()
def collect_episode(env: APCEnv, model: DreamerV4, device: torch.device,
                    cfg: TrainConfig, *, random_action: bool = False,
                    deterministic: bool = False) -> Dict[str, np.ndarray]:
    obs_window = env.reset(exploration=random_action)
    T, L, D = cfg.episode_length, cfg.lookback, env.obs_dim
    obs_buf = np.zeros((T, L, D), dtype='float32')
    act_buf = np.zeros((T, env.action_dim), dtype='float32')
    rew_buf = np.zeros(T, dtype='float32')
    cont_buf = np.ones(T, dtype='float32')

    h, z = model.rssm.initial_state(1, device)
    prev_action = torch.zeros(1, env.action_dim, device=device)

    for t in range(T):
        obs_buf[t] = obs_window
        ow = torch.from_numpy(obs_window).to(device).unsqueeze(0)
        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=(device.type == 'cuda')):
            h, z, _, _ = model.rssm.observe_step(ow, prev_action, h, z)
            if random_action:
                a = (env.rng.uniform(-1.0, 1.0, size=(env.action_dim,))).astype('float32')
                action_t = torch.from_numpy(a).to(device).unsqueeze(0)
            else:
                latent = torch.cat([h, z], dim=-1)
                action_t, _, _ = model.actor(latent, deterministic=deterministic)
        a_np = action_t.float().squeeze(0).cpu().numpy().astype('float32')
        next_window, reward, done, _ = env.step(a_np)
        act_buf[t] = a_np
        rew_buf[t] = reward
        cont_buf[t] = 0.0 if done and t == T - 1 else 1.0
        prev_action = action_t
        obs_window = next_window
        if done:
            break
    return {'obs': obs_buf, 'act': act_buf, 'rew': rew_buf, 'cont': cont_buf}


# ---------------------------------------------------------------------------
# World model loss
# ---------------------------------------------------------------------------

def world_model_step(model: DreamerV4, batch: Dict[str, torch.Tensor],
                     cfg: TrainConfig) -> Tuple[Dict[str, torch.Tensor],
                                                 torch.Tensor, torch.Tensor]:
    """Run sequence rollout with posterior; return (losses, h_seq, z_seq).

    Returned ``h_seq`` and ``z_seq`` are detached for use as starting states
    in actor-critic imagination.
    """
    obs = batch['obs']            # (B, T, L, D)
    act = batch['act']            # (B, T, A)
    rew = batch['rew']            # (B, T)
    cont = batch['cont']          # (B, T)
    B, T = obs.shape[:2]
    device = obs.device

    h, z = model.rssm.initial_state(B, device)
    h_list, z_list = [], []
    post_logits_list, prior_logits_list = [], []

    # Use act[:, t-1] as the action that drove transition into step t.
    # For t=0 we use a zero action.
    # Sequential rollout: only the recurrent observe_step stays per-step;
    # all decode/reward/cont head forwards are batched once after.
    for t in range(T):
        prev_a = act[:, t - 1] if t > 0 else torch.zeros_like(act[:, 0])
        h, z, post, prior = model.rssm.observe_step(obs[:, t], prev_a, h, z)
        h_list.append(h)
        z_list.append(z)
        post_logits_list.append(post)
        prior_logits_list.append(prior)

    h_seq = torch.stack(h_list, dim=1)               # (B, T, deter)
    z_seq = torch.stack(z_list, dim=1)               # (B, T, stoch)
    post_seq = torch.stack(post_logits_list, dim=1)  # (B, T, stoch)
    prior_seq = torch.stack(prior_logits_list, dim=1)

    # Batched head forwards across (B*T) — one CUDA kernel per head instead of T.
    h_flat = h_seq.reshape(B * T, -1)
    z_flat = z_seq.reshape(B * T, -1)
    decoded = model.rssm.decode(h_flat, z_flat).view(B, T, -1)
    reward_logits = model.rssm.predict_reward(h_flat, z_flat).view(B, T, -1)
    cont_logits = model.rssm.predict_cont(h_flat, z_flat).view(B, T, -1).squeeze(-1)

    # --- Reconstruction loss (symlog MSE on the latest frame of each window) ---
    target_obs = obs[:, :, -1, :]   # (B, T, D) — predicted-step observation
    recon_loss = F.mse_loss(symlog(decoded), symlog(target_obs))

    # --- Reward loss (twohot) ---
    flat_logits = reward_logits.reshape(-1, reward_logits.shape[-1])
    flat_target = rew.reshape(-1)
    reward_loss = model.rssm.reward_head.loss(flat_logits, flat_target).mean()

    # --- Continuation loss (BCE) ---
    cont_loss = F.binary_cross_entropy_with_logits(cont_logits, cont)

    # --- KL with free bits ---
    flat_post = post_seq.reshape(B * T, -1)
    flat_prior = prior_seq.reshape(B * T, -1)
    kl_loss = free_bits_kl(flat_post, flat_prior,
                           model.cfg.rssm.n_categoricals,
                           model.cfg.rssm.n_classes,
                           free_nats=cfg.free_nats)

    total = (cfg.recon_scale * recon_loss
             + cfg.reward_scale * reward_loss
             + cfg.cont_scale * cont_loss
             + cfg.kl_scale * kl_loss)

    losses = {
        'wm_total': total,
        'wm_recon': recon_loss.detach(),
        'wm_reward': reward_loss.detach(),
        'wm_cont': cont_loss.detach(),
        'wm_kl': kl_loss.detach(),
    }
    return losses, h_seq.detach(), z_seq.detach()


# ---------------------------------------------------------------------------
# Actor-critic imagination
# ---------------------------------------------------------------------------

def actor_critic_step(model: DreamerV4, h0: torch.Tensor, z0: torch.Tensor,
                      cfg: TrainConfig) -> Dict[str, torch.Tensor]:
    """Imagine ``H`` steps from ``(h0, z0)`` and compute actor + critic losses.

    Per-trajectory advantage normalization (paper canonical).
    """
    device = h0.device
    BT = h0.shape[0]
    H = cfg.horizon

    h, z = h0, z0
    h_list, z_list, log_prob_list, ent_list = [h], [z], [], []
    for t in range(H):
        latent = torch.cat([h, z], dim=-1)
        action, log_prob, entropy = model.actor(latent)
        h, z, _ = model.rssm.imagine_step(h, z, action)
        h_list.append(h)
        z_list.append(z)
        log_prob_list.append(log_prob)
        ent_list.append(entropy)

    H_h = torch.stack(h_list, dim=1)        # (BT, H+1, deter)
    H_z = torch.stack(z_list, dim=1)        # (BT, H+1, stoch)
    log_probs = torch.stack(log_prob_list, dim=1)  # (BT, H)
    entropies = torch.stack(ent_list, dim=1)        # (BT, H)

    flat = torch.cat([H_h, H_z], dim=-1)          # (BT, H+1, latent)
    # Predicted reward / continuation along the horizon.
    flat2 = flat.reshape(BT * (H + 1), -1)
    rew_logits = model.rssm.reward_head(flat2)
    rewards = model.rssm.reward_head.expectation(rew_logits).view(BT, H + 1)
    cont_logits = model.rssm.cont_head(flat2).view(BT, H + 1)
    cont_pred = torch.sigmoid(cont_logits).detach()

    # Critic values (target critic for bootstrap, current for loss).
    target_values = model.target_critic.value(flat2).view(BT, H + 1).detach()
    # λ-returns (GAE-style on imagined horizon, paper §C).
    gamma = cfg.gamma * cont_pred                  # (BT, H+1)
    lam = cfg.gae_lambda
    returns = torch.zeros_like(target_values)
    returns[:, -1] = target_values[:, -1]
    for t in reversed(range(H)):
        bootstrap = (1.0 - lam) * target_values[:, t + 1] + lam * returns[:, t + 1]
        returns[:, t] = rewards[:, t + 1] + gamma[:, t + 1] * bootstrap

    # Critic loss: twohot CE between current critic logits and λ-return.
    target_returns = returns[:, :-1].detach()       # (BT, H)
    flat_latents_for_critic = flat[:, :-1].reshape(BT * H, -1)
    critic_logits = model.critic(flat_latents_for_critic)
    critic_loss = model.critic.head.loss(critic_logits,
                                         target_returns.reshape(-1)).mean()

    # Actor: paper §C global return normalization.
    # advantage = (R^lambda - v(z)) / max(1, EMA(P95 - P5))
    # — global across the imagined batch, EMA-tracked across updates.
    with torch.no_grad():
        baseline = model.critic.value(flat_latents_for_critic).view(BT, H)
        adv_raw = target_returns - baseline
        scale = model.update_return_scale(target_returns).clamp_min(1.0)
        adv_norm = adv_raw / scale

    actor_loss = -(log_probs * adv_norm).mean() \
                 - cfg.actor_entropy_scale * entropies.mean()

    return {
        'actor_loss': actor_loss,
        'critic_loss': critic_loss,
        'imagined_return_mean': target_returns.mean().detach(),
        'imagined_reward_mean': rewards[:, 1:].mean().detach(),
        'entropy_mean': entropies.mean().detach(),
        'adv_std_mean': adv_raw.std(dim=1).mean().detach(),
        'return_scale': scale.detach().squeeze(),
    }


# ---------------------------------------------------------------------------
# Main trainer
# ---------------------------------------------------------------------------

def build_model(cfg: TrainConfig) -> DreamerV4:
    rssm_cfg = RSSMConfig(
        obs_dim=cfg.obs_dim, action_dim=cfg.action_dim, lookback=cfg.lookback,
        deter_dim=cfg.deter_dim, embed_dim=cfg.embed_dim, hidden_dim=cfg.hidden_dim,
        n_categoricals=cfg.n_categoricals, n_classes=cfg.n_classes,
        free_nats=cfg.free_nats,
    )
    model_cfg = DreamerV4Config(rssm=rssm_cfg, n_action_bins=cfg.n_action_bins,
                                actor_hidden=cfg.hidden_dim,
                                critic_hidden=cfg.hidden_dim)
    return DreamerV4(model_cfg)


def calibrate_reward_scale(env: 'APCEnv', rng: np.random.Generator,
                            n_steps: int = 1500,
                            target_std: float = 1.0,
                            min_scale: float = 1.0,
                            max_scale: float = 1000.0) -> Dict[str, float]:
    """Empirically choose a per-step reward scale.

    Runs a short rollout under uniform-random actions and the same training
    disturbance schedule, measures the std of raw per-step rewards, and
    returns ``target_std / std``.  Floored at 1.0 so plants whose natural
    rewards already span O(1) are left untouched.

    The ``[-20, +20]`` symlog twohot support of the reward head (paper §B)
    represents per-step rewards with the highest resolution near 0; with
    target_std≈1.0, ~95% of returns fall inside |r|<2 → ~64 of the 255 bins
    carry meaningful probability.  This is the regime the paper is
    calibrated for.
    """
    if env.reward_scale != 1.0:
        env.reward_scale = 1.0  # measure raw magnitudes
    raw_rewards: List[float] = []
    env.reset(exploration=True)
    for _ in range(int(n_steps)):
        a = rng.uniform(-1.0, 1.0, size=(env.action_dim,)).astype('float32')
        _, _, done, info = env.step(a)
        raw_rewards.append(float(info.get('raw_reward', 0.0)))
        if done:
            env.reset(exploration=True)
    arr = np.asarray(raw_rewards, dtype='float64')
    std = float(arr.std())
    mean = float(arr.mean())
    if std < 1e-8:
        scale = 1.0
    else:
        scale = float(target_std / std)
    scale = float(np.clip(scale, min_scale, max_scale))
    env.reward_scale = scale
    return {
        'reward_scale': scale,
        'raw_std': std,
        'raw_mean': mean,
        'raw_min': float(arr.min()),
        'raw_max': float(arr.max()),
        'target_std': float(target_std),
        'n_steps': int(n_steps),
    }


def train(cfg: TrainConfig, on_iter_end=None) -> Dict:
    """Train DreamerV4.

    Parameters
    ----------
    cfg : TrainConfig
    on_iter_end : optional callable ``(iter:int, env_steps:int, ema_return:float)
                  -> bool``.  If it returns True, training stops early — used
                  by the Optuna BO runner to support trial pruning.
    """
    rng = np.random.default_rng(int(os.environ.get('SEED', '0')))
    torch.manual_seed(int(os.environ.get('SEED', '0')))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # GPU performance knobs (A10 / Ampere+).  These are no-ops on CPU.
    if device.type == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision('high')
        except Exception:
            pass

    # Build env first so we know obs_dim/action_dim.
    env = APCEnv(cfg, rng)
    cfg.action_dim = env.action_dim
    cfg.state_dim = env.state_dim
    cfg.aug_obs_dim = env.aug_obs_dim
    cfg.obs_dim = env.obs_dim

    # Reward scaling: paper §B's symlog-twohot reward head is calibrated for
    # per-step rewards spanning O(1).  APC objectives often produce O(0.01),
    # which leaves the head pinned at the uniform-prior loss.  Either:
    #   • OBJ_REWARD_SCALE=<float>  — explicit override (no calibration)
    #   • OBJ_REWARD_SCALE=auto / unset — short calibration rollout to
    #     match a target raw-reward std of 1.0 (clamped ≥1.0).
    obj_scale_env = os.environ.get('OBJ_REWARD_SCALE', 'auto').strip().lower()
    if obj_scale_env in ('', 'auto', '1', 'on', 'true'):
        if obj_scale_env in ('1', 'on', 'true'):
            # legacy boolean ON → still calibrate adaptively
            pass
        cal = calibrate_reward_scale(env, rng)
        print(f"[reward-scale] auto-calibrated: scale={cal['reward_scale']:.3f}  "
              f"raw_std={cal['raw_std']:.5f} raw_range=[{cal['raw_min']:.4f},"
              f"{cal['raw_max']:.4f}]  target_std={cal['target_std']:.2f}",
              flush=True)
        out_dir_pre = Path(cfg.out_dir or '.')
        out_dir_pre.mkdir(parents=True, exist_ok=True)
        try:
            with open(out_dir_pre / 'reward_calibration.json', 'w') as f:
                json.dump(cal, f, indent=2)
        except Exception:
            pass
    elif obj_scale_env in ('off', '0', 'none', 'false'):
        env.reward_scale = 1.0
        print('[reward-scale] disabled (raw rewards passed through)', flush=True)
    else:
        try:
            env.reward_scale = float(obj_scale_env)
            print(f'[reward-scale] explicit override: scale={env.reward_scale:.3f}',
                  flush=True)
        except Exception:
            env.reward_scale = 1.0
            print(f'[reward-scale] could not parse OBJ_REWARD_SCALE={obj_scale_env!r}, '
                  f'defaulting to 1.0', flush=True)

    model = build_model(cfg).to(device)

    opt_world = torch.optim.AdamW(model.parameters_world(), lr=cfg.lr_world,
                                  eps=1e-8, weight_decay=0.0)
    opt_actor = torch.optim.AdamW(model.parameters_actor(), lr=cfg.lr_actor,
                                  eps=1e-8, weight_decay=0.0)
    opt_critic = torch.optim.AdamW(model.parameters_critic(), lr=cfg.lr_critic,
                                   eps=1e-8, weight_decay=0.0)

    capacity_eps = max(1, cfg.buffer_capacity_steps // cfg.episode_length)
    policy_buf = TrajectoryBuffer(capacity_eps, cfg.episode_length,
                                  cfg.lookback, cfg.obs_dim, cfg.action_dim)
    explore_buf = TrajectoryBuffer(max(1, capacity_eps // 2),
                                   cfg.episode_length,
                                   cfg.lookback, cfg.obs_dim, cfg.action_dim)

    out_dir = Path(cfg.out_dir or '.')
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / 'train_log.jsonl'
    log_f = open(log_path, 'a')

    total_env_steps = 0
    total_iters = 0
    start_time = time.time()
    last_log_time = start_time
    last_log_steps = 0
    ema_return = None

    # Header line in the .log so we can correlate with absolute time.
    print(f"# train start: {time.strftime('%Y-%m-%d %H:%M:%S')} "
          f"device={device.type} cfg.batch_size={cfg.batch_size} "
          f"seq_len={cfg.seq_len} horizon={cfg.horizon} "
          f"deter={cfg.deter_dim} embed={cfg.embed_dim} hidden={cfg.hidden_dim} "
          f"K={cfg.n_categoricals} C={cfg.n_classes} "
          f"lookback={cfg.lookback} ep_len={cfg.episode_length}",
          flush=True)

    # Seed buffers: collect a few exploration episodes first so WM can train.
    for _ in range(2):
        ep = collect_episode(env, model, device, cfg, random_action=True)
        explore_buf.add_episode(ep['obs'], ep['act'], ep['rew'], ep['cont'])
        total_env_steps += cfg.episode_length

    while total_env_steps < cfg.total_steps:
        # ----- 1. Collect episodes -----
        for _ in range(cfg.ep_per_iter):
            is_explore = rng.uniform() < cfg.explore_episode_ratio
            ep = collect_episode(env, model, device, cfg, random_action=is_explore)
            target = explore_buf if is_explore else policy_buf
            target.add_episode(ep['obs'], ep['act'], ep['rew'], ep['cont'])
            total_env_steps += cfg.episode_length
            ret = float(ep['rew'].sum())
            ema_return = ret if ema_return is None else 0.95 * ema_return + 0.05 * ret

        if policy_buf.filled == 0:
            # Until we have at least one policy episode, draw the WM mix from
            # exploration only.
            mix_explore = 1.0
        else:
            mix_explore = cfg.explore_buffer_ratio if explore_buf.filled > 0 else 0.0

        # ----- 2. Train -----
        for _ in range(cfg.train_steps_per_iter):
            n_explore = int(round(mix_explore * cfg.batch_size))
            n_policy = cfg.batch_size - n_explore
            parts = []
            if n_policy > 0 and policy_buf.filled > 0:
                parts.append(policy_buf.sample(n_policy, cfg.seq_len, rng))
            if n_explore > 0 and explore_buf.filled > 0:
                parts.append(explore_buf.sample(n_explore, cfg.seq_len, rng))
            batch_np = {k: np.concatenate([p[k] for p in parts], axis=0) for k in parts[0]}
            batch = {}
            for k, v in batch_np.items():
                t = torch.from_numpy(v)
                if device.type == 'cuda':
                    t = t.pin_memory().to(device, non_blocking=True)
                else:
                    t = t.to(device)
                batch[k] = t

            # World model
            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16,
                                    enabled=(device.type == 'cuda')):
                wm_losses, h_seq, z_seq = world_model_step(model, batch, cfg)
            opt_world.zero_grad(set_to_none=True)
            wm_losses['wm_total'].backward()
            wm_grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters_world(), cfg.grad_clip)
            opt_world.step()

            # Actor-critic in latent imagination, starting from posterior states.
            B, T = h_seq.shape[:2]
            h0 = h_seq.reshape(B * T, -1)
            z0 = z_seq.reshape(B * T, -1)
            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16,
                                    enabled=(device.type == 'cuda')):
                ac_losses = actor_critic_step(model, h0, z0, cfg)
            opt_actor.zero_grad(set_to_none=True)
            ac_losses['actor_loss'].backward(retain_graph=True)
            actor_grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters_actor(), cfg.grad_clip)
            opt_actor.step()
            opt_critic.zero_grad(set_to_none=True)
            ac_losses['critic_loss'].backward()
            critic_grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters_critic(), cfg.grad_clip)
            opt_critic.step()

            model.update_target(cfg.target_critic_tau)

        total_iters += 1
        if total_iters % cfg.log_every == 0:
            now = time.time()
            iter_dt = now - last_log_time
            steps_dt = total_env_steps - last_log_steps
            sps = (steps_dt / iter_dt) if iter_dt > 0 else 0.0
            gpu_mem_mb = None
            gpu_mem_peak_mb = None
            if device.type == 'cuda':
                try:
                    gpu_mem_mb = torch.cuda.memory_allocated(device) / (1024 ** 2)
                    gpu_mem_peak_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
                    torch.cuda.reset_peak_memory_stats(device)
                except Exception:
                    pass
            row = {
                'iter': total_iters,
                'env_steps': total_env_steps,
                'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
                'wallclock_s': now - start_time,
                'iter_time_s': iter_dt,
                'env_steps_per_s': sps,
                'ema_return': float(ema_return) if ema_return is not None else None,
                'mix_explore': float(mix_explore),
                'policy_eps': policy_buf.filled,
                'explore_eps': explore_buf.filled,
                'wm_grad_norm': float(wm_grad_norm.detach().item()
                                      if torch.is_tensor(wm_grad_norm) else wm_grad_norm),
                'actor_grad_norm': float(actor_grad_norm.detach().item()
                                         if torch.is_tensor(actor_grad_norm) else actor_grad_norm),
                'critic_grad_norm': float(critic_grad_norm.detach().item()
                                          if torch.is_tensor(critic_grad_norm) else critic_grad_norm),
                'gpu_mem_mb': gpu_mem_mb,
                'gpu_mem_peak_mb': gpu_mem_peak_mb,
            }
            for k, v in wm_losses.items():
                row[k] = float(v.detach().item() if torch.is_tensor(v) else v)
            for k, v in ac_losses.items():
                row[k] = float(v.detach().item() if torch.is_tensor(v) else v)
            log_f.write(json.dumps(row) + '\n')
            log_f.flush()
            print(f"[{row['timestamp']}] iter {total_iters:4d} "
                  f"steps {total_env_steps:6d} sps {sps:5.1f} "
                  f"ret_ema {row['ema_return']:.2f} "
                  f"recon {row['wm_recon']:.4f} kl {row['wm_kl']:.4f} "
                  f"actor {row['actor_loss']:+.4f} critic {row['critic_loss']:.4f} "
                  f"ent {row.get('entropy_mean', 0.0):.3f} "
                  f"adv_std {row.get('adv_std_mean', 0.0):.3f} "
                  f"gn(w/a/c) {row['wm_grad_norm']:.2f}/"
                  f"{row['actor_grad_norm']:.3f}/{row['critic_grad_norm']:.2f} "
                  f"img_ret {row.get('imagined_return_mean', 0.0):+.3f}",
                  flush=True)
            last_log_time = now
            last_log_steps = total_env_steps

        if on_iter_end is not None:
            try:
                stop = bool(on_iter_end(int(total_iters),
                                         int(total_env_steps),
                                         float(ema_return) if ema_return is not None else 0.0))
            except Exception:
                stop = False
            if stop:
                print(f'[train] early-stop requested by callback at iter {total_iters}',
                      flush=True)
                break
        if total_iters % cfg.save_every_iters == 0:
            torch.save({'model': model.state_dict(),
                        'cfg': asdict(cfg)},
                       out_dir / f'ckpt_iter_{total_iters:05d}.pt')

    log_f.close()
    final_path = out_dir / 'final.pt'
    torch.save({'model': model.state_dict(), 'cfg': asdict(cfg)}, final_path)
    return {'final_ckpt': str(final_path), 'iters': total_iters,
            'env_steps': total_env_steps,
            'final_ema_return': float(ema_return) if ema_return is not None else None}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cfg_from_env() -> TrainConfig:
    cfg = TrainConfig()
    for name, attr, cast in [
        ('DREAMER_DETER_DIM', 'deter_dim', int),
        ('DREAMER_EMBED_DIM', 'embed_dim', int),
        ('DREAMER_HIDDEN_DIM', 'hidden_dim', int),
        ('DREAMER_N_CATEGORICALS', 'n_categoricals', int),
        ('DREAMER_N_CLASSES', 'n_classes', int),
        ('DREAMER_LOOKBACK', 'lookback', int),
        ('DREAMER_HORIZON', 'horizon', int),
        ('DREAMER_SEQ_LEN', 'seq_len', int),
        ('DREAMER_BATCH_SIZE', 'batch_size', int),
        ('DREAMER_FREE_NATS', 'free_nats', float),
        ('DREAMER_ACTOR_ENTROPY', 'actor_entropy_scale', float),
        ('AGENT_TOTAL_STEPS', 'total_steps', int),
        ('SIM_EPISODE_LENGTH', 'episode_length', int),
        ('SIM_SAMPLE_RATE', 'sample_rate', int),
        ('CONTROLLER_OUT_DIR', 'out_dir', str),
    ]:
        v = os.environ.get(name)
        if v is not None and v != '':
            setattr(cfg, attr, cast(v))
    return cfg


if __name__ == '__main__':
    cfg = _cfg_from_env()
    summary = train(cfg)
    print(json.dumps(summary, indent=2))
