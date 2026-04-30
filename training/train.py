"""Three-phase Dreamer 4 trainer for APC.

Reference: Hafner, Yan, Lillicrap (2025) — "Training Agents Inside of
Scalable World Models", arXiv:2509.24527.

Algorithm 1 (paper-faithful, adapted to single-task online APC sim):

    Phase 1 — pretrain world model
        Collect random-action episodes; train tokenizer recon (eq. 5)
        + dynamics shortcut forcing (eq. 7).

    Phase 2 — agent finetune
        Continue collecting random-action episodes (paper-strict for the
        offline-data setting); keep eq. 5 + eq. 7 live and add policy +
        reward multi-token-prediction heads (eq. 9).

    Phase 3 — imagination training
        Freeze tokenizer + dynamics + reward head. Snapshot the current
        policy as the PMPO prior. Sample dataset contexts from the
        replay buffer; imagine H steps using K=4 shortcut sampling per
        step; train the value head with TD-λ (eq. 10) and the policy
        head with PMPO (eq. 11). Run periodic evaluation episodes for
        the return-window score (no buffer writes).

Phase budget: ``cfg.phaseN_frac`` of ``cfg.total_steps`` for N ∈ {1, 2, 3}.
"""

from __future__ import annotations

import json
import math
import os
import time
from collections import deque
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from models.dreamer_v4 import (
    DreamerV4, DreamerV4Config,
    shortcut_forcing_loss, pmpo_loss,
)
from utils.sim_factory import create_sim, resolve_sim_metadata
from utils.objective_runtime import compute_objective_components
from utils.runtime_setpoints import RuntimeSetpointManager, RuntimeSetpointConfig
from utils.training_disturbance import (
    build_training_disturbance_schedule,
    apply_disturbance_schedule,
)
from utils.agent_utils import (
    load_objective_weights, load_objective_bounds, load_full_objective_spec,
    action_to_control,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    # ----- Tokenizer -----
    tok_hidden: int = 256
    z_dim: int = 24
    mae_p_max: float = 0.5

    # ----- Dynamics transformer -----
    d_model: int = 256
    n_layers: int = 6
    n_heads: int = 8
    ff_mult: int = 4
    n_register: int = 4
    k_max: int = 4
    tau_n_bins: int = 32
    soft_cap: float = 50.0

    # ----- Heads -----
    n_action_bins: int = 21
    head_hidden: int = 256
    head_n_layers: int = 2

    # ----- Plant / windowing -----
    lookback: int = 32        # transformer context length T_ctx
    sample_rate: int = 5
    episode_length: int = 600

    # ----- Training overall -----
    total_steps: int = 100_000
    train_steps_per_iter: int = 100
    ep_per_iter: int = 5
    seq_len: int = 64
    batch_size: int = 16
    horizon: int = 15

    # ----- Phase budget fractions (paper Algorithm 1) -----
    phase1_frac: float = 0.4
    phase2_frac: float = 0.2
    phase3_frac: float = 0.4

    # ----- Optimizers -----
    lr_world: float = 1e-4
    lr_actor: float = 3e-5
    lr_critic: float = 3e-5
    grad_clip: float = 1000.0

    # ----- Loss weights -----
    recon_scale: float = 1.0
    sf_scale: float = 1.0
    reward_scale_loss: float = 1.0   # Phase-2 reward MTP weight
    bc_scale: float = 1.0            # Phase-2 policy BC weight

    # ----- MTP -----
    mtp_length: int = 8              # paper L=8 (Phase-2 multi-token prediction)

    # ----- PMPO (Phase 3) -----
    pmpo_alpha: float = 0.5
    pmpo_beta: float = 0.1

    # ----- Returns -----
    gamma: float = 0.997
    gae_lambda: float = 0.95
    target_critic_tau: float = 0.02
    tau_ctx: float = 0.1             # context-noise corruption at inference

    # ----- Buffer -----
    buffer_capacity_steps: int = 200_000

    # ----- Phase-3 evaluation cadence -----
    phase3_eval_every_iters: int = 5

    # ----- I/O -----
    out_dir: str = ''
    log_every: int = 1
    save_every_iters: int = 20

    # ----- Speedups (DREAMER_FAST_ATTN=1, DREAMER_COMPILE=1) -----
    attn_impl: str = 'auto'          # 'auto'|'manual'|'sdpa'
    compile_mode: str = ''           # '' (off) | 'default' | 'reduce-overhead' | 'max-autotune'

    # ----- Resolved at build-time -----
    obs_dim: int = 0
    action_dim: int = 0
    aug_obs_dim: int = 0
    state_dim: int = 0


# ---------------------------------------------------------------------------
# Env wrapper — stacks state + aug-obs, builds lookback window, computes reward
# ---------------------------------------------------------------------------

class APCEnv:
    """Slim env wrapper around the carryover simulator.

    ``obs_window`` shape ``(lookback, obs_dim)``; ``obs_dim = state_dim + aug_obs_dim``.
    The lookback window is maintained for streaming inference (the V4
    dynamics transformer attends over the last ``lookback`` frames of
    encoded latents).
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
        self.reward_scale: float = 1.0
        self._last_cv_violation_sum: float = 0.0
        self._last_mv_violation_sum: float = 0.0

    def _build_obs_vec(self, state: np.ndarray) -> np.ndarray:
        aug = self.setpoint_mgr.get_augmented_obs_channels()
        return np.concatenate([np.asarray(state, dtype='float32').reshape(-1),
                                aug], axis=0)

    def reset(self, *, exploration: bool = False) -> np.ndarray:
        state = self.sim.reset()
        if isinstance(state, tuple):
            state = state[0]
        state = np.asarray(state, dtype='float32').reshape(-1)
        self._t = 0
        self._prev_control = np.zeros(self.action_dim, dtype='float32')
        self._last_cv_violation_sum = 0.0
        self._last_mv_violation_sum = 0.0
        self.setpoint_mgr.reset(episode_length=self.cfg.episode_length,
                                curriculum_fraction=1.0)
        intensity = 1.0 if not exploration else 1.2
        self._schedule = build_training_disturbance_schedule(
            episode_length=self.cfg.episode_length,
            rng=self.rng,
            intensity=intensity,
            sim=self.sim,
        )
        obs_vec = self._build_obs_vec(state)
        self._window = np.tile(obs_vec, (self.cfg.lookback, 1)).astype('float32')
        return self._window.copy()

    def step(self, action_norm: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        action_norm = np.asarray(action_norm, dtype='float32').reshape(self.action_dim)
        action_01 = 0.5 * (np.clip(action_norm, -1.0, 1.0) + 1.0)
        control = action_to_control(action_01, self.bounds, self.setpoint_mgr)
        self.setpoint_mgr.step(self._t)
        next_state = self.sim.step(control)
        if isinstance(next_state, tuple):
            next_state = next_state[0]
        next_state = np.asarray(next_state, dtype='float32').reshape(-1)
        try:
            apply_disturbance_schedule(next_state, self.sim, self._schedule)
        except Exception:
            pass
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
        obs_vec = self._build_obs_vec(next_state)
        self._window = np.concatenate([self._window[1:], obs_vec[None, :]], axis=0)
        self._last_cv_violation_sum += float(comps.get('cv_violation_penalty', 0.0))
        self._last_mv_violation_sum += float(comps.get('mv_violation_penalty', 0.0))
        info = {'reward_components': comps, 't': self._t,
                'raw_reward': raw_reward}
        return self._window.copy(), reward, done, info


# ---------------------------------------------------------------------------
# Replay buffer — episode-major
# ---------------------------------------------------------------------------

class TrajectoryBuffer:
    def __init__(self, capacity_eps: int, episode_length: int,
                 lookback: int, obs_dim: int, action_dim: int):
        self.capacity_eps = int(capacity_eps)
        self.T = int(episode_length)
        self.L = int(lookback)
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.obs = np.zeros((capacity_eps, self.T, self.L, self.obs_dim),
                            dtype='float32')
        self.act = np.zeros((capacity_eps, self.T, self.action_dim),
                            dtype='float32')
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
        out_obs = np.zeros((batch_size, seq_len, self.L, self.obs_dim),
                           dtype='float32')
        out_act = np.zeros((batch_size, seq_len, self.action_dim),
                           dtype='float32')
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
# Episode collection (V4 streaming inference)
# ---------------------------------------------------------------------------

@torch.no_grad()
def collect_episode(env: APCEnv, model: DreamerV4, device: torch.device,
                    cfg: TrainConfig, *, random_action: bool = False,
                    deterministic: bool = False) -> Dict[str, np.ndarray]:
    """Collect one episode using V4 model for inference.

    Streaming inference path (per timestep):
      1. Encode the last ``lookback`` observations through the tokenizer
         → ``z_ctx`` (clean past latents).
      2. Run the dynamics transformer over ``(z_ctx, a_ctx)`` with τ=1−τ_ctx
         (slight context noise) and d=d_min; read the agent-register
         hidden state at the latest time slot.
      3. Sample (or argmax) the action from the policy head.
    """
    obs_window = env.reset(exploration=random_action)
    T, L, D = cfg.episode_length, cfg.lookback, env.obs_dim
    obs_buf = np.zeros((T, L, D), dtype='float32')
    act_buf = np.zeros((T, env.action_dim), dtype='float32')
    rew_buf = np.zeros(T, dtype='float32')
    cont_buf = np.ones(T, dtype='float32')

    a_history = np.zeros((L, env.action_dim), dtype='float32')
    d_min = 1.0 / cfg.k_max
    tau_ctx_val = 1.0 - cfg.tau_ctx

    for t in range(T):
        obs_buf[t] = obs_window
        if random_action:
            a_np = env.rng.uniform(-1.0, 1.0,
                                    size=(env.action_dim,)).astype('float32')
        else:
            ow = torch.from_numpy(obs_window).to(device)            # (L, D)
            a_ctx = torch.from_numpy(a_history).to(device)           # (L, A)
            with torch.amp.autocast(device_type=device.type,
                                     dtype=torch.bfloat16,
                                     enabled=(device.type == 'cuda')):
                z_ctx = model.tokenizer.encode(ow).unsqueeze(0)     # (1, L, z)
                tau = torch.full((1, L), tau_ctx_val, device=device,
                                  dtype=z_ctx.dtype)
                d = torch.full((1, L), d_min, device=device,
                                dtype=z_ctx.dtype)
                out = model.dynamics(z_ctx, tau, d, a_ctx.unsqueeze(0))
                agent_hid = out['agent_hid'][:, -1]                  # (1, D)
                action_t, _, _ = model.policy(agent_hid,
                                                deterministic=deterministic)
            a_np = action_t.float().squeeze(0).cpu().numpy().astype('float32')
        next_window, reward, done, _ = env.step(a_np)
        act_buf[t] = a_np
        rew_buf[t] = reward
        cont_buf[t] = 0.0 if done and t == T - 1 else 1.0
        a_history = np.concatenate([a_history[1:], a_np[None, :]], axis=0)
        obs_window = next_window
        if done:
            break
    return {'obs': obs_buf, 'act': act_buf, 'rew': rew_buf, 'cont': cont_buf}


# ---------------------------------------------------------------------------
# Phase 1 / 2 — World model loss (tokenizer recon + shortcut forcing)
# ---------------------------------------------------------------------------

def world_model_loss(model: DreamerV4, batch: Dict[str, torch.Tensor],
                      cfg: TrainConfig
                      ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor,
                                  torch.Tensor]:
    """Eq. 5 + Eq. 7. Returns (losses, z_clean, agent_hid).

    ``z_clean``  : (B, T, z_dim)  — frozen-tokenizer outputs (no MAE).
    ``agent_hid``: (B, T, d_model) — agent register hidden state from a
                    *clean* dynamics pass (used by Phase 2 BC heads).
    """
    obs = batch['obs']                     # (B, T, L, D)
    act = batch['act']                     # (B, T, A)
    obs_cur = obs[:, :, -1, :]             # (B, T, D)

    # Tokenizer with MAE: recon the masked obs.
    z_mae, recon = model.tokenizer.forward_with_mae(obs_cur)
    recon_loss = model.tokenizer.recon_loss(obs_cur, recon)

    # Shortcut forcing target = clean tokenizer output (no MAE).
    z_clean = model.tokenizer.encode(obs_cur)            # (B, T, z_dim)
    sf_loss, sf_diag = shortcut_forcing_loss(model.dynamics,
                                                z_clean.detach(), act)

    # Compute agent_hid from a clean dynamics pass (τ=1, d=d_min). Used by
    # the Phase 2 BC + reward MTP heads.
    B, T = z_clean.shape[:2]
    device = z_clean.device
    tau_clean = torch.full((B, T), 1.0, device=device, dtype=z_clean.dtype)
    d_min = torch.full((B, T), 1.0 / cfg.k_max, device=device,
                        dtype=z_clean.dtype)
    out_clean = model.dynamics(z_clean, tau_clean, d_min, act)
    agent_hid = out_clean['agent_hid']

    losses: Dict[str, torch.Tensor] = {
        'recon_loss': recon_loss,
        'sf_loss': sf_loss,
        'wm_total': cfg.recon_scale * recon_loss + cfg.sf_scale * sf_loss,
    }
    losses.update({k: v for k, v in sf_diag.items()})
    return losses, z_clean.detach(), agent_hid


def agent_finetune_loss(model: DreamerV4, batch: Dict[str, torch.Tensor],
                         agent_hid: torch.Tensor, cfg: TrainConfig
                         ) -> Dict[str, torch.Tensor]:
    """Eq. 9: policy MTP (BC) + reward MTP, length L = ``cfg.mtp_length``.

    For each context position ``t`` we read the agent-register hidden state
    ``agent_hid[t]`` and predict the next ``L`` actions and rewards
    ``(a_{t+1..t+L}, r_{t+1..t+L})``. Per the paper this is implemented as
    one head with ``L`` parallel output projections (shared trunk).
    """
    act = batch['act']                       # (B, T, A)
    rew = batch['rew']                       # (B, T)
    B, T, A = act.shape
    L_mtp = max(1, int(cfg.mtp_length))
    # Number of context positions for which all L future targets exist.
    T_ctx = T - L_mtp
    if T_ctx <= 0:
        # Sequence too short — fall back to L=1 BC on whatever we have.
        T_ctx = max(1, T - 1)
        L_mtp_eff = min(L_mtp, T - T_ctx)
    else:
        L_mtp_eff = L_mtp

    feat = agent_hid[:, :T_ctx].reshape(-1, agent_hid.shape[-1])  # (B*T_ctx,D)

    # Build (B, T_ctx, L, A) future-action and (B, T_ctx, L) future-reward
    # tensors via a strided slice — no Python loop, fully vectorised.
    fut_act = torch.stack(
        [act[:, 1 + l : 1 + l + T_ctx] for l in range(L_mtp_eff)], dim=2
    )                                                       # (B, T_ctx, L, A)
    fut_rew = torch.stack(
        [rew[:, 1 + l : 1 + l + T_ctx] for l in range(L_mtp_eff)], dim=2
    )                                                       # (B, T_ctx, L)
    fut_act = fut_act.reshape(B * T_ctx, L_mtp_eff, A)
    fut_rew = fut_rew.reshape(B * T_ctx, L_mtp_eff)

    # Policy MTP (BC over L future actions)
    if L_mtp_eff < model.policy.mtp_length:
        # Pad target with zeros so we can use logits_mtp; mask out the pad in loss.
        pad_act = torch.zeros(B * T_ctx,
                               model.policy.mtp_length - L_mtp_eff, A,
                               device=fut_act.device, dtype=fut_act.dtype)
        fut_act_full = torch.cat([fut_act, pad_act], dim=1)
        bc_lp_full = model.policy.log_prob_of_mtp(feat, fut_act_full)
        bc_loss = -bc_lp_full[:, :L_mtp_eff].mean()
    else:
        bc_lp = model.policy.log_prob_of_mtp(feat, fut_act)        # (BT, L)
        bc_loss = -bc_lp.mean()

    # Reward MTP (twohot CE over L future rewards)
    rew_logits_all = model.reward.forward_mtp(feat)           # (BT, L, K)
    if L_mtp_eff < model.reward.mtp_length:
        rew_logits_all = rew_logits_all[:, :L_mtp_eff]
    rew_loss_per = model.reward.loss_mtp(rew_logits_all, fut_rew)  # (BT, L)
    reward_mtp_loss = rew_loss_per.mean()

    total = cfg.bc_scale * bc_loss + cfg.reward_scale_loss * reward_mtp_loss
    return {
        'bc_loss': bc_loss.detach(),
        'reward_mtp_loss': reward_mtp_loss.detach(),
        'agent_total': total,
    }


# ---------------------------------------------------------------------------
# Phase 3 — Imagination training (PMPO + TD-λ)
# ---------------------------------------------------------------------------

def imagination_step(model: DreamerV4, batch: Dict[str, torch.Tensor],
                      cfg: TrainConfig) -> Dict[str, torch.Tensor]:
    """Phase 3: roll out H imagined steps, compute PMPO + TD-λ.

    Transformer + tokenizer + reward head are kept frozen here (callers
    only pass ``parameters_actor()`` and ``parameters_critic()`` to the
    optimizer in Phase 3). Imagination uses the V4 K=4 shortcut sampler.
    """
    obs = batch['obs']                       # (B, T, L, D)
    act = batch['act']                       # (B, T, A)
    B, T = obs.shape[:2]
    H = int(cfg.horizon)
    device = obs.device

    # Encode the buffered context (frozen tokenizer).
    with torch.no_grad():
        obs_cur = obs[:, :, -1, :]
        z_ctx = model.tokenizer.encode(obs_cur)              # (B, T, z)
    a_ctx = act

    # Sliding history.
    z_history = z_ctx
    a_history = a_ctx
    d_min_v = 1.0 / cfg.k_max

    imagined_rewards: List[torch.Tensor] = []
    imagined_logp: List[torch.Tensor] = []      # currently informational only
    imagined_entropy: List[torch.Tensor] = []
    imagined_actions: List[torch.Tensor] = []
    imagined_values: List[torch.Tensor] = []
    imagined_target_v: List[torch.Tensor] = []
    imagined_agent_hid: List[torch.Tensor] = []

    for h in range(H):
        T_now = z_history.shape[1]
        # 1. Clean dynamics pass to obtain agent_hid at the latest position.
        with torch.no_grad():
            tau_clean = torch.full((B, T_now), 1.0, device=device,
                                     dtype=z_history.dtype)
            d_min_t = torch.full((B, T_now), d_min_v, device=device,
                                  dtype=z_history.dtype)
            out = model.dynamics(z_history, tau_clean, d_min_t, a_history)
            agent_hid_t = out['agent_hid'][:, -1]            # (B, D)

        # 2. Policy sample (gradients flow through this).
        action_t, logp_t, ent_t = model.policy(agent_hid_t)
        # 3. Frozen reward + target value.
        with torch.no_grad():
            r_logits = model.reward(agent_hid_t)
            reward_pred = model.reward.expectation(r_logits)        # (B,)
            tv_logits = model.target_value(agent_hid_t)
            target_v_pred = model.target_value.expectation(tv_logits)
        # 4. Current value head (with grad).
        value_logits = model.value(agent_hid_t)              # (B, n_bins)

        # 5. Sample next z via K=4 shortcut forcing.
        with torch.no_grad():
            z_next = model.imagine_next_z(z_history, action_t,
                                            k_steps=cfg.k_max,
                                            tau_ctx=cfg.tau_ctx)     # (B, z)

        # 6. Slide history (keep at most T tokens).
        z_history = torch.cat([z_history[:, -(T - 1):],
                                 z_next.unsqueeze(1)], dim=1)
        a_history = torch.cat([a_history[:, -(T - 1):],
                                 action_t.unsqueeze(1)], dim=1)

        imagined_rewards.append(reward_pred)
        imagined_logp.append(logp_t)
        imagined_entropy.append(ent_t)
        imagined_actions.append(action_t)
        imagined_values.append(value_logits)
        imagined_target_v.append(target_v_pred)
        imagined_agent_hid.append(agent_hid_t)

    rewards = torch.stack(imagined_rewards, dim=1)            # (B, H)
    target_values = torch.stack(imagined_target_v, dim=1)     # (B, H)
    actions = torch.stack(imagined_actions, dim=1)            # (B, H, A)
    entropies = torch.stack(imagined_entropy, dim=1)          # (B, H)
    agent_hids = torch.stack(imagined_agent_hid, dim=1)       # (B, H, D)
    value_logits_seq = torch.stack(imagined_values, dim=1)    # (B, H, n_bins)

    # λ-returns (eq. 10).
    gamma = cfg.gamma
    lam = cfg.gae_lambda
    returns = torch.zeros_like(target_values)
    returns[:, -1] = target_values[:, -1]
    for t in reversed(range(H - 1)):
        bootstrap = (1.0 - lam) * target_values[:, t + 1] + lam * returns[:, t + 1]
        returns[:, t] = rewards[:, t] + gamma * bootstrap
    target_returns = returns.detach()

    # Critic loss (twohot CE).
    val_logits_flat = value_logits_seq.reshape(-1, value_logits_seq.shape[-1])
    val_target_flat = target_returns.reshape(-1)
    critic_loss = model.value.loss(val_logits_flat, val_target_flat).mean()

    # Advantage (current critic baseline) → PMPO (eq. 11).
    with torch.no_grad():
        baseline = model.value.expectation(
            model.value(agent_hids.reshape(-1, agent_hids.shape[-1]))
        ).view(B, H)
        adv_raw = target_returns - baseline
        scale = model.update_return_scale(target_returns).clamp_min(1.0)

    feat_flat = agent_hids.reshape(-1, agent_hids.shape[-1])
    actions_flat = actions.reshape(-1, actions.shape[-1])
    adv_flat = (adv_raw / scale).reshape(-1).detach()

    actor_loss, pmpo_diag = pmpo_loss(model.policy, model.prior_policy,
                                       feat_flat, actions_flat, adv_flat,
                                       alpha=cfg.pmpo_alpha,
                                       beta=cfg.pmpo_beta)

    diag = {
        'actor_loss': actor_loss,
        'critic_loss': critic_loss,
        'imagined_return_mean': target_returns.mean().detach(),
        'imagined_reward_mean': rewards.mean().detach(),
        'entropy_mean': entropies.mean().detach(),
        'adv_std_mean': adv_raw.std(dim=1).mean().detach(),
        'return_scale': scale.detach().squeeze(),
    }
    diag.update(pmpo_diag)
    return diag


# ---------------------------------------------------------------------------
# Model + reward calibration
# ---------------------------------------------------------------------------

def build_model(cfg: TrainConfig) -> DreamerV4:
    model_cfg = DreamerV4Config(
        obs_dim=cfg.obs_dim, action_dim=cfg.action_dim, lookback=cfg.lookback,
        tok_hidden=cfg.tok_hidden, z_dim=cfg.z_dim, mae_p_max=cfg.mae_p_max,
        d_model=cfg.d_model, n_layers=cfg.n_layers, n_heads=cfg.n_heads,
        ff_mult=cfg.ff_mult, n_register=cfg.n_register,
        k_max=cfg.k_max, tau_n_bins=cfg.tau_n_bins, soft_cap=cfg.soft_cap,
        attn_impl=cfg.attn_impl,
        n_action_bins=cfg.n_action_bins,
        head_hidden=cfg.head_hidden, head_n_layers=cfg.head_n_layers,
        mtp_length=max(1, int(cfg.mtp_length)),
    )
    model = DreamerV4(model_cfg)
    # Optional torch.compile (set via TrainConfig.compile_mode or env var).
    cm = (cfg.compile_mode or '').strip()
    if not cm:
        env_cm = os.environ.get('DREAMER_COMPILE', '').strip()
        if env_cm in ('1', 'true', 'True'):
            cm = 'default'
        elif env_cm:
            cm = env_cm
    if cm:
        model.maybe_compile(mode=cm)
    return model


def calibrate_reward_scale(env: 'APCEnv', rng: np.random.Generator,
                            n_steps: int = 1500,
                            target_std: float = 1.0,
                            min_scale: float = 1.0,
                            max_scale: float = 1000.0) -> Dict[str, float]:
    """Empirically choose a per-step reward scale to match V4's twohot range."""
    if env.reward_scale != 1.0:
        env.reward_scale = 1.0
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
        'reward_scale': scale, 'raw_std': std, 'raw_mean': mean,
        'raw_min': float(arr.min()), 'raw_max': float(arr.max()),
        'target_std': float(target_std), 'n_steps': int(n_steps),
    }


# ---------------------------------------------------------------------------
# Diagnostics plot
# ---------------------------------------------------------------------------

def _save_training_diagnostics_plot(log_path: Path, out_path: Path) -> None:
    """Plot a 2x3 diagnostic grid from train_log.jsonl.

    Panels: (1) ema_return + return_window_mean, (2) WM losses (recon/sf),
    (3) phase 2/3 losses (bc/reward_mtp/actor/critic), (4) entropy & adv_std,
    (5) grad norms (w/a/c) + skip count, (6) violations.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    rows = []
    with open(log_path, 'r') as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    if not rows:
        return
    steps = [r.get('env_steps', 0) for r in rows]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)

    ax = axes[0, 0]
    ax.plot(steps, [r.get('ema_return') for r in rows], label='ema_return', lw=1.0)
    rw = [r.get('return_window_mean') for r in rows]
    if any(v is not None for v in rw):
        ax.plot(steps, rw, label='return_window_mean', lw=1.2, color='C3')
    # Phase boundaries (vertical lines).
    phases = [r.get('phase') for r in rows]
    for i in range(1, len(phases)):
        if phases[i] is not None and phases[i] != phases[i - 1]:
            ax.axvline(steps[i], color='gray', lw=0.5, ls=':',
                        alpha=0.5)
    ax.axhline(0, color='gray', lw=0.5)
    ax.set_ylabel('return'); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax.set_title('Returns')

    ax = axes[0, 1]
    for k, c in [('recon_loss', 'C0'), ('sf_loss', 'C1'),
                  ('sf_loss_flow', 'C2'), ('sf_loss_boot', 'C3')]:
        vals = [r.get(k) for r in rows]
        if any(v is not None for v in vals):
            ax.plot(steps, vals, label=k, lw=1.0, color=c)
    ax.set_yscale('symlog', linthresh=1e-3)
    ax.set_ylabel('WM losses'); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax.set_title('World model losses')

    ax = axes[0, 2]
    for k, c in [('bc_loss', 'C0'), ('reward_mtp_loss', 'C1'),
                  ('actor_loss', 'C2'), ('critic_loss', 'C3')]:
        vals = [r.get(k) for r in rows]
        if any(v is not None for v in vals):
            ax.plot(steps, vals, label=k, lw=1.0, color=c)
    ax.set_ylabel('Phase 2/3 losses'); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax.set_title('Agent / RL losses')

    ax = axes[1, 0]
    ax.plot(steps, [r.get('entropy_mean') for r in rows], label='entropy', lw=1.0)
    ax2 = ax.twinx()
    ax2.plot(steps, [r.get('adv_std_mean') for r in rows], color='C3',
              label='adv_std', lw=1.0)
    ax.set_ylabel('entropy'); ax2.set_ylabel('adv_std', color='C3')
    ax.legend(loc='upper left', fontsize=8); ax.grid(alpha=0.3)
    ax.set_title('Policy entropy / advantage std')

    ax = axes[1, 1]
    for k, c in [('wm_grad_norm', 'C0'), ('actor_grad_norm', 'C1'),
                  ('critic_grad_norm', 'C2')]:
        ax.plot(steps, [r.get(k) for r in rows], label=k, lw=1.0, color=c)
    ax.set_yscale('symlog', linthresh=1e-2)
    ax.set_ylabel('grad norm'); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    skips = [r.get('n_grad_skip', 0) for r in rows]
    ax.set_title(f'Grad norms (skip total={skips[-1] if skips else 0})')

    ax = axes[1, 2]
    cv = [r.get('iter_cv_violation_mean') for r in rows]
    mv = [r.get('iter_mv_violation_mean') for r in rows]
    if any(v is not None for v in cv):
        ax.plot(steps, cv, label='cv_v', lw=1.0, color='C3')
    if any(v is not None for v in mv):
        ax.plot(steps, mv, label='mv_v', lw=1.0, color='C1')
    ax.set_ylabel('mean violation'); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax.set_title('Violations (per-iter mean)')

    for ax in axes[1, :]:
        ax.set_xlabel('env_steps')
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main trainer (three explicit phases)
# ---------------------------------------------------------------------------

def train(cfg: TrainConfig, on_iter_end=None) -> Dict:
    rng = np.random.default_rng(int(os.environ.get('SEED', '0')))
    torch.manual_seed(int(os.environ.get('SEED', '0')))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if device.type == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision('high')
        except Exception:
            pass

    env = APCEnv(cfg, rng)
    cfg.action_dim = env.action_dim
    cfg.state_dim = env.state_dim
    cfg.aug_obs_dim = env.aug_obs_dim
    cfg.obs_dim = env.obs_dim

    # ---- Reward calibration (V4 reward head expects O(1) per-step rewards) ----
    obj_scale_env = os.environ.get('OBJ_REWARD_SCALE', 'auto').strip().lower()
    if obj_scale_env in ('', 'auto', '1', 'on', 'true'):
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
        try:
            plan_path = out_dir_pre / 'run_plan.json'
            if plan_path.exists():
                with open(plan_path) as f:
                    plan = json.load(f)
                plan['reward_calibration'] = cal
                with open(plan_path, 'w') as f:
                    json.dump(plan, f, indent=2)
        except Exception:
            pass
    elif obj_scale_env in ('off', '0', 'none', 'false'):
        env.reward_scale = 1.0
    else:
        try:
            env.reward_scale = float(obj_scale_env)
        except Exception:
            env.reward_scale = 1.0

    model = build_model(cfg).to(device)

    # Square-root LR scaling for adaptive batch (kept from V3 trainer).
    bs_ref = 16
    lr_scale = math.sqrt(max(1, cfg.batch_size) / float(bs_ref)) \
                if cfg.batch_size > bs_ref else 1.0
    eff_lr_world = cfg.lr_world * lr_scale
    eff_lr_actor = cfg.lr_actor * lr_scale
    eff_lr_critic = cfg.lr_critic * lr_scale
    if abs(lr_scale - 1.0) > 1e-9:
        print(f'[lr-scale] batch_size={cfg.batch_size} ref={bs_ref} '
              f'-> sqrt lr_scale={lr_scale:.3f}', flush=True)

    opt_world = torch.optim.AdamW(model.parameters_world(), lr=eff_lr_world,
                                  eps=1e-8, weight_decay=0.0)
    opt_actor = torch.optim.AdamW(model.parameters_actor(), lr=eff_lr_actor,
                                  eps=1e-8, weight_decay=0.0)
    opt_critic = torch.optim.AdamW(model.parameters_critic(), lr=eff_lr_critic,
                                   eps=1e-8, weight_decay=0.0)

    capacity_eps = max(1, cfg.buffer_capacity_steps // cfg.episode_length)
    buf = TrajectoryBuffer(capacity_eps, cfg.episode_length,
                            cfg.lookback, cfg.obs_dim, cfg.action_dim)

    out_dir = Path(cfg.out_dir or '.')
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / 'train_log.jsonl'
    log_f = open(log_path, 'a')

    # Phase budgets.
    p1 = int(cfg.phase1_frac * cfg.total_steps)
    p2 = int(cfg.phase2_frac * cfg.total_steps)
    p3 = cfg.total_steps - p1 - p2     # absorb rounding

    print(f"# train start: {time.strftime('%Y-%m-%d %H:%M:%S')} "
          f"device={device.type} bs={cfg.batch_size} "
          f"seq_len={cfg.seq_len} horizon={cfg.horizon} "
          f"d_model={cfg.d_model} layers={cfg.n_layers} heads={cfg.n_heads} "
          f"z_dim={cfg.z_dim} lookback={cfg.lookback} "
          f"phases={p1}/{p2}/{p3}",
          flush=True)

    total_env_steps = 0
    total_iters = 0
    start_time = time.time()
    last_log_time = start_time
    last_log_steps = 0
    ema_return: Optional[float] = None
    return_window: 'deque[float]' = deque(maxlen=10)
    iter_returns: List[float] = []
    iter_cv_violations: List[float] = []
    iter_mv_violations: List[float] = []
    iter_raw_returns: List[float] = []
    n_grad_skip = 0
    current_phase = 1
    t_collect_acc = 0.0
    t_sample_acc = 0.0
    t_wm_acc = 0.0
    t_ac_acc = 0.0

    def _phase_for(env_steps: int) -> int:
        if env_steps < p1:
            return 1
        if env_steps < p1 + p2:
            return 2
        return 3

    # Seed buffer (always with random actions; needed before any training step).
    for _ in range(2):
        ep = collect_episode(env, model, device, cfg, random_action=True)
        buf.add_episode(ep['obs'], ep['act'], ep['rew'], ep['cont'])
        total_env_steps += cfg.episode_length

    # Cached optimizer set per phase.
    while total_env_steps < cfg.total_steps:
        new_phase = _phase_for(total_env_steps)
        if new_phase != current_phase:
            print(f'[phase] transition {current_phase} -> {new_phase} '
                  f'at env_steps={total_env_steps}', flush=True)
            current_phase = new_phase
            if current_phase == 3:
                # Snapshot the prior policy (PMPO behavioural prior, eq. 11).
                model.snapshot_prior_policy()

        # ----- Collection -----
        # Phases 1 & 2: random-action episodes append to the buffer.
        # Phase 3: only periodic deterministic eval episodes (no buffer write).
        _t = time.time()
        if current_phase in (1, 2):
            for _ in range(cfg.ep_per_iter):
                ep = collect_episode(env, model, device, cfg, random_action=True)
                buf.add_episode(ep['obs'], ep['act'], ep['rew'], ep['cont'])
                total_env_steps += cfg.episode_length
                ret = float(ep['rew'].sum())
                # Don't include phase-1/2 random-action returns in the
                # return_window — they are not policy returns.
                ema_return = (ret if ema_return is None
                                else 0.95 * ema_return + 0.05 * ret)
        else:
            # Phase 3: optional eval episode every N iters (deterministic policy).
            if (total_iters % max(1, cfg.phase3_eval_every_iters)) == 0:
                ep = collect_episode(env, model, device, cfg,
                                       random_action=False, deterministic=True)
                total_env_steps += cfg.episode_length
                ret = float(ep['rew'].sum())
                ema_return = (ret if ema_return is None
                                else 0.95 * ema_return + 0.05 * ret)
                return_window.append(ret)
                iter_returns.append(ret)
                rs = float(env.reward_scale) if env.reward_scale else 1.0
                iter_raw_returns.append(ret / rs if rs else ret)
                iter_cv_violations.append(float(getattr(env,
                                            '_last_cv_violation_sum', 0.0)))
                iter_mv_violations.append(float(getattr(env,
                                            '_last_mv_violation_sum', 0.0)))
            else:
                total_env_steps += cfg.episode_length  # stay paced with budget
        t_collect_acc += time.time() - _t

        # ----- Train -----
        wm_grad_norm = torch.tensor(0.0)
        actor_grad_norm = torch.tensor(0.0)
        critic_grad_norm = torch.tensor(0.0)
        wm_losses: Dict[str, torch.Tensor] = {}
        ag_losses: Dict[str, torch.Tensor] = {}
        ac_losses: Dict[str, torch.Tensor] = {}

        for _ in range(cfg.train_steps_per_iter):
            _t = time.time()
            batch_np = buf.sample(cfg.batch_size, cfg.seq_len, rng)
            batch: Dict[str, torch.Tensor] = {}
            for k, v in batch_np.items():
                t = torch.from_numpy(v)
                if device.type == 'cuda':
                    t = t.pin_memory().to(device, non_blocking=True)
                else:
                    t = t.to(device)
                batch[k] = t
            t_sample_acc += time.time() - _t

            if current_phase in (1, 2):
                # World-model losses (always live in P1 + P2).
                _t = time.time()
                with torch.amp.autocast(device_type=device.type,
                                          dtype=torch.bfloat16,
                                          enabled=(device.type == 'cuda')):
                    wm_losses, _, agent_hid = world_model_loss(model, batch, cfg)
                # Phase 2: also update reward + policy via MTP (eq. 9).
                if current_phase == 2:
                    with torch.amp.autocast(device_type=device.type,
                                              dtype=torch.bfloat16,
                                              enabled=(device.type == 'cuda')):
                        ag_losses = agent_finetune_loss(model, batch,
                                                          agent_hid, cfg)
                    total_loss = wm_losses['wm_total'] + ag_losses['agent_total']
                else:
                    total_loss = wm_losses['wm_total']

                opt_world.zero_grad(set_to_none=True)
                if current_phase == 2:
                    opt_actor.zero_grad(set_to_none=True)
                total_loss.backward()
                wm_grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters_world(), cfg.grad_clip)
                if current_phase == 2:
                    actor_grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters_actor(), cfg.grad_clip)
                if not torch.isfinite(wm_grad_norm):
                    n_grad_skip += 1
                    opt_world.zero_grad(set_to_none=True)
                    if current_phase == 2:
                        opt_actor.zero_grad(set_to_none=True)
                else:
                    opt_world.step()
                    if current_phase == 2 and torch.isfinite(actor_grad_norm):
                        opt_actor.step()
                t_wm_acc += time.time() - _t

            else:  # Phase 3 — imagination RL
                _t = time.time()
                with torch.amp.autocast(device_type=device.type,
                                          dtype=torch.bfloat16,
                                          enabled=(device.type == 'cuda')):
                    ac_losses = imagination_step(model, batch, cfg)
                opt_actor.zero_grad(set_to_none=True)
                opt_critic.zero_grad(set_to_none=True)
                (ac_losses['actor_loss'] + ac_losses['critic_loss']).backward()
                actor_grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters_actor(), cfg.grad_clip)
                critic_grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters_critic(), cfg.grad_clip)
                if (not torch.isfinite(actor_grad_norm)
                        or not torch.isfinite(critic_grad_norm)):
                    n_grad_skip += 1
                    opt_actor.zero_grad(set_to_none=True)
                    opt_critic.zero_grad(set_to_none=True)
                else:
                    opt_actor.step()
                    opt_critic.step()
                model.update_target(cfg.target_critic_tau)
                t_ac_acc += time.time() - _t

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
                'phase': current_phase,
                'env_steps': total_env_steps,
                'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
                'wallclock_s': now - start_time,
                'iter_time_s': iter_dt,
                'env_steps_per_s': sps,
                'ema_return': float(ema_return) if ema_return is not None else None,
                'buf_eps': buf.filled,
                'wm_grad_norm': float(wm_grad_norm.detach().item()
                                        if torch.is_tensor(wm_grad_norm)
                                        else wm_grad_norm),
                'actor_grad_norm': float(actor_grad_norm.detach().item()
                                           if torch.is_tensor(actor_grad_norm)
                                           else actor_grad_norm),
                'critic_grad_norm': float(critic_grad_norm.detach().item()
                                            if torch.is_tensor(critic_grad_norm)
                                            else critic_grad_norm),
                'gpu_mem_mb': gpu_mem_mb,
                'gpu_mem_peak_mb': gpu_mem_peak_mb,
                't_collect_s': t_collect_acc,
                't_sample_s': t_sample_acc,
                't_wm_s': t_wm_acc,
                't_ac_s': t_ac_acc,
                'return_window_mean': (float(np.mean(return_window))
                                        if return_window else None),
                'return_window_std': (float(np.std(return_window))
                                        if len(return_window) > 1 else None),
                'return_window_n': int(len(return_window)),
                'iter_return_mean': (float(np.mean(iter_returns))
                                      if iter_returns else None),
                'iter_raw_return_mean': (float(np.mean(iter_raw_returns))
                                          if iter_raw_returns else None),
                'iter_cv_violation_mean': (float(np.mean(iter_cv_violations))
                                            if iter_cv_violations else None),
                'iter_mv_violation_mean': (float(np.mean(iter_mv_violations))
                                            if iter_mv_violations else None),
                'n_grad_skip': int(n_grad_skip),
                'buf_fill_pct': float(buf.filled) / max(1, buf.capacity_eps),
            }
            for k, v in {**wm_losses, **ag_losses, **ac_losses}.items():
                row[k] = float(v.detach().item() if torch.is_tensor(v) else v)
            log_f.write(json.dumps(row) + '\n')
            log_f.flush()
            rwm = row.get('return_window_mean')
            rwm_str = f"{rwm:+.2f}" if rwm is not None else 'n/a'
            ema_str = (f"{row['ema_return']:.2f}"
                        if row['ema_return'] is not None else 'n/a')
            print(f"[{row['timestamp']}] P{current_phase} iter {total_iters:4d} "
                  f"steps {total_env_steps:6d} sps {sps:5.1f} "
                  f"ret_ema {ema_str} ret_w {rwm_str} "
                  f"recon {row.get('recon_loss', 0.0):.4f} "
                  f"sf {row.get('sf_loss', 0.0):.4f} "
                  f"bc {row.get('bc_loss', 0.0):.4f} "
                  f"actor {row.get('actor_loss', 0.0):+.4f} "
                  f"critic {row.get('critic_loss', 0.0):.4f} "
                  f"ent {row.get('entropy_mean', 0.0):.3f} "
                  f"img_ret {row.get('imagined_return_mean', 0.0):+.3f} "
                  f"skip {row.get('n_grad_skip', 0)}",
                  flush=True)
            last_log_time = now
            last_log_steps = total_env_steps
            t_collect_acc = t_sample_acc = t_wm_acc = t_ac_acc = 0.0
            iter_returns = []
            iter_cv_violations = []
            iter_mv_violations = []
            iter_raw_returns = []

        if on_iter_end is not None:
            try:
                stop = bool(on_iter_end(int(total_iters),
                                          int(total_env_steps),
                                          float(ema_return)
                                          if ema_return is not None else 0.0))
            except Exception:
                stop = False
            if stop:
                print(f'[train] early-stop requested at iter {total_iters}',
                      flush=True)
                break
        if total_iters % cfg.save_every_iters == 0:
            torch.save({'model': model.state_dict(), 'cfg': asdict(cfg)},
                       out_dir / f'ckpt_iter_{total_iters:05d}.pt')

    log_f.close()
    final_path = out_dir / 'final.pt'
    torch.save({'model': model.state_dict(), 'cfg': asdict(cfg)}, final_path)

    try:
        _save_training_diagnostics_plot(log_path, out_dir / 'training_diagnostics.png')
    except Exception as e:
        print(f'[train] training_diagnostics.png skipped: {e!r}', flush=True)

    return {
        'final_ckpt': str(final_path), 'iters': total_iters,
        'env_steps': total_env_steps,
        'final_ema_return': (float(ema_return)
                              if ema_return is not None else None),
        'final_return_window_mean': (float(np.mean(return_window))
                                       if return_window else None),
        'final_return_window_std': (float(np.std(return_window))
                                      if len(return_window) > 1 else None),
        'final_return_window_n': int(len(return_window)),
        'n_grad_skip': int(n_grad_skip),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cfg_from_env() -> TrainConfig:
    cfg = TrainConfig()
    for name, attr, cast in [
        ('DREAMER_D_MODEL', 'd_model', int),
        ('DREAMER_N_LAYERS', 'n_layers', int),
        ('DREAMER_N_HEADS', 'n_heads', int),
        ('DREAMER_FF_MULT', 'ff_mult', int),
        ('DREAMER_N_REGISTER', 'n_register', int),
        ('DREAMER_Z_DIM', 'z_dim', int),
        ('DREAMER_TOK_HIDDEN', 'tok_hidden', int),
        ('DREAMER_HEAD_HIDDEN', 'head_hidden', int),
        ('DREAMER_K_MAX', 'k_max', int),
        ('DREAMER_LOOKBACK', 'lookback', int),
        ('DREAMER_HORIZON', 'horizon', int),
        ('DREAMER_SEQ_LEN', 'seq_len', int),
        ('DREAMER_BATCH_SIZE', 'batch_size', int),
        ('DREAMER_PHASE1_FRAC', 'phase1_frac', float),
        ('DREAMER_PHASE2_FRAC', 'phase2_frac', float),
        ('DREAMER_PHASE3_FRAC', 'phase3_frac', float),
        ('DREAMER_PMPO_BETA', 'pmpo_beta', float),
        ('DREAMER_MAE_PMAX', 'mae_p_max', float),
        ('DREAMER_MTP_LENGTH', 'mtp_length', int),
        ('DREAMER_ATTN_IMPL', 'attn_impl', str),
        ('DREAMER_COMPILE_MODE', 'compile_mode', str),
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
