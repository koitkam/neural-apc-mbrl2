"""Smoke test for the 2026-06-09 WM-fix workstream levers:

  * #6 per-channel CV-weighted recon (``wm_recon_cv_weight``)
  * #5 WM freeze-after-pretrain (``wm_freeze_after_iters``) — freeze mechanism
  * #4 BC expert-return tracking (``bc_track_expert_every``) — config plumbing
  * ENV_OVERRIDES wiring for the new DREAMER_* knobs.
  (#7 WM-only excitation partition was REMOVED 2026-06-12 — never helped; the
   settle-aware seed buffer covers open-loop gain.)

Verifies, for BOTH backbones where applicable, WITHOUT a real env:
  * default cfg => every lever OFF / identity (no change vs p106 baseline).
  * #6 engaged => weighted recon finite, differs from uniform, up-weights the
    CV channel, backprops; world_model_loss stays finite (RSSM + SF).
  * #5 => freezing model.dynamics leaves the reward head (parameters_world)
    still trainable.
  * env overrides map each knob onto the right cfg field.

Run (CPU, do not disturb a live GPU run):
  CUDA_VISIBLE_DEVICES="" PYTHONPATH=$PWD \
    /home/koitkam/neural-APC-mbrl2-env/bin/python tools/_smoke_wm_fixes.py
"""
import os
import numpy as np
import torch
import torch.nn.functional as F

from training.train import (TrainConfig, build_model, world_model_loss,
                            TrajectoryBuffer, _weighted_recon_mse)


def _mk(obs_dim=6, action_dim=2, wm_type='rssm', cv_idx=(2,)):
    torch.manual_seed(0)
    cfg = TrainConfig()
    cfg.obs_dim = obs_dim
    cfg.action_dim = action_dim
    cfg.lookback = 8
    cfg.world_model_type = wm_type
    cfg.compile_mode = 'none'          # CPU smoke: skip slow torch.compile
    cfg.rssm_deter_dim = 64
    cfg.rssm_n_categoricals = 8
    cfg.rssm_n_classes = 8
    cfg.rssm_embed_dim = 32
    cfg.rssm_hidden_dim = 32
    cfg.d_model = 64
    cfg.head_hidden = 64
    cfg.head_n_layers = 2
    cfg.mtp_length = 4
    cfg.horizon = 4
    cfg.seq_len = 16
    cfg.cv_obs_indices = tuple(cv_idx)
    model = build_model(cfg)
    B, T = 3, cfg.seq_len
    batch = {
        'obs': torch.randn(B, T, obs_dim),
        'act': torch.rand(B, T, action_dim) * 2 - 1,
        'rew': torch.randn(B, T),
        'cont': torch.ones(B, T),
        'expert': (torch.rand(B, T) > 0.5).float(),
    }
    return cfg, model, batch


def _test_weighted_recon():
    print('\n=== #6 per-channel CV-weighted recon (helper) ===')
    cfg = TrainConfig()
    cfg.cv_obs_indices = (2,)
    torch.manual_seed(1)
    recon = torch.randn(4, 10, 6, requires_grad=True)
    target = torch.randn(4, 10, 6)

    # default weight 1.0 => byte-identical to F.mse_loss (identity).
    cfg.wm_recon_cv_weight = 1.0
    w1 = _weighted_recon_mse(recon, target, cfg)
    m1 = F.mse_loss(recon, target)
    assert torch.allclose(w1, m1), f'cv_weight=1.0 must equal F.mse_loss: {w1} vs {m1}'
    print(f'[smoke] OK  cv_weight=1.0 identity to F.mse_loss ({float(w1.detach()):.5f})')

    # empty cv indices => identity even with weight != 1.0.
    cfg.cv_obs_indices = ()
    cfg.wm_recon_cv_weight = 5.0
    w_empty = _weighted_recon_mse(recon, target, cfg)
    assert torch.allclose(w_empty, m1), 'empty cv_obs_indices must be identity'
    print('[smoke] OK  empty cv_obs_indices identity to F.mse_loss')

    # engaged: differs from uniform, finite, up-weights the CV channel error.
    cfg.cv_obs_indices = (2,)
    cfg.wm_recon_cv_weight = 5.0
    w5 = _weighted_recon_mse(recon, target, cfg)
    assert torch.isfinite(w5).all(), 'weighted recon must be finite'
    assert not torch.allclose(w5, m1), 'cv_weight=5 must differ from uniform'
    w5.backward()
    assert recon.grad is not None and torch.isfinite(recon.grad).all(), \
        'weighted recon must backprop'
    # the CV channel (idx 2) gradient magnitude must exceed a non-CV channel's
    # for the same per-element error scale (it is up-weighted).
    g = recon.grad.abs().mean(dim=(0, 1))            # per-channel mean |grad|
    se = (recon.detach() - target).pow(2).mean(dim=(0, 1))
    # normalise by raw error so we compare WEIGHT, not the random error size.
    cv_w_eff = float(g[2] / se[2].clamp_min(1e-6))
    other_w_eff = float(g[0] / se[0].clamp_min(1e-6))
    assert cv_w_eff > other_w_eff, \
        f'CV channel must be up-weighted: cv={cv_w_eff:.3f} other={other_w_eff:.3f}'
    print(f'[smoke] OK  CV channel up-weighted (cv_eff={cv_w_eff:.3f} > '
          f'other_eff={other_w_eff:.3f})')


def _test_recon_integration(wm_type):
    print(f'\n=== #6 world_model_loss integration ({wm_type}) ===')
    # default 1.0 => finite (identity path).
    cfg, model, batch = _mk(wm_type=wm_type)
    cfg.wm_recon_cv_weight = 1.0
    losses0, _, _ = world_model_loss(model, batch, cfg)
    r0 = float(losses0['recon_loss'])
    assert np.isfinite(r0), 'default recon_loss must be finite'
    print(f'[smoke] OK  default recon_loss finite ({r0:.5f}) ({wm_type})')

    # engaged => finite + backprops through wm_total.
    cfg, model, batch = _mk(wm_type=wm_type)
    cfg.wm_recon_cv_weight = 6.0
    losses1, _, _ = world_model_loss(model, batch, cfg)
    r1 = float(losses1['recon_loss'])
    assert np.isfinite(r1), 'weighted recon_loss must be finite'
    assert torch.isfinite(losses1['wm_total']).all()
    losses1['wm_total'].backward()
    print(f'[smoke] OK  cv_weight=6 recon_loss finite + backprops '
          f'({r1:.5f}) ({wm_type})')


def _test_freeze_mechanism():
    print('\n=== #5 freeze-after-pretrain (freeze mechanism) ===')
    cfg, model, _ = _mk(wm_type='rssm')
    # freeze the WM core exactly as the train loop does.
    for p in model.dynamics.parameters():
        p.requires_grad_(False)
    assert all(not p.requires_grad for p in model.dynamics.parameters()), \
        'all dynamics params must be frozen'
    # the reward head (in parameters_world) must remain trainable.
    n_world_trainable = sum(1 for p in model.parameters_world()
                            if p.requires_grad)
    assert n_world_trainable > 0, \
        'reward head (parameters_world) must remain trainable after WM freeze'
    print(f'[smoke] OK  dynamics frozen, {n_world_trainable} world params '
          f'(reward head) still trainable')


def _test_defaults_and_env():
    print('\n=== defaults OFF + ENV_OVERRIDES wiring ===')
    cfg = TrainConfig()
    assert cfg.wm_freeze_after_iters == 0, 'freeze must default 0 (off)'
    assert cfg.wm_best_restore_at_p2 is False
    assert cfg.wm_best_restore_at_p3 is False
    assert int(cfg.wm_best_restore_min_gap) == 10
    print('[smoke] OK  wm_best_restore_at_p2 default OFF; freeze-after-iters OFF')

    from workflow._plant_prepare import apply_dreamer_env_overrides
    env_map = {
        'DREAMER_WM_FREEZE_AFTER_ITERS': ('40', 'wm_freeze_after_iters', 40),
        'DREAMER_WM_RECON_CV_WEIGHT': ('8.0', 'wm_recon_cv_weight', 8.0),
        'DREAMER_BC_TRACK_EXPERT_EVERY': ('5', 'bc_track_expert_every', 5),
        'DREAMER_EXPERT_BC_P3_FLOOR': ('0.0', 'expert_bc_p3_floor', 0.0),
        'DREAMER_WM_BEST_RESTORE_AT_P2': ('1', 'wm_best_restore_at_p2', True),
        'DREAMER_WM_BEST_RESTORE_MIN_GAP': ('7', 'wm_best_restore_min_gap', 7),
        'DREAMER_CONST_ACTION_INJECT_EVERY': ('7', 'const_action_inject_every', 7),
        'DREAMER_DV_PRBS_INJECT_EVERY': ('5', 'dv_prbs_inject_every', 5),
        'DREAMER_GAIN_READY_LO': ('0.75', 'gain_ready_lo', 0.75),
        'DREAMER_P1_GAIN_GATE': ('0', 'p1_gain_gate', False),
        'DREAMER_WM_PROBE_EVERY_ITERS': ('8', 'wm_probe_every_iters', 8),
        'DREAMER_ES_GRADSKIP_MAX': ('11', 'early_stop_grad_skip_max', 11),
        'DREAMER_STEP_TEST_INJECT_N': ('7', 'step_test_inject_n', 7),
        'DREAMER_LR_WORLD': ('3e-4', 'lr_world', 3e-4),
    }
    for k, (val, _f, _e) in env_map.items():
        os.environ[k] = val
    try:
        cfg2 = TrainConfig()
        apply_dreamer_env_overrides(cfg2)
        for k, (val, field, expected) in env_map.items():
            got = getattr(cfg2, field)
            assert got == expected, f'{k} -> {field}: expected {expected}, got {got}'
        print('[smoke] OK  env overrides map to the right cfg fields')
    finally:
        for k in env_map:
            os.environ.pop(k, None)

    from dataclasses import fields
    from workflow._plant_prepare import ENV_OVERRIDES
    names = {f.name for f in fields(TrainConfig)}
    missing = [k for k, (field, _) in ENV_OVERRIDES.items() if field not in names]
    assert not missing, f'ENV_OVERRIDES fields missing on TrainConfig: {missing}'
    print(f'[smoke] OK  all {len(ENV_OVERRIDES)} ENV_OVERRIDES map to TrainConfig')


def _test_buffer_sample_identity():
    print('\n=== TrajectoryBuffer.sample fancy-index identity ===')
    T, D, A, N, n_dist = 32, 6, 2, 11, 1
    buf = TrajectoryBuffer(N, T, D, A, n_dist=n_dist)
    for i in range(N):
        obs = np.full((T, D), float(i), dtype='float32')
        obs[:, 0] = np.arange(T, dtype='float32')
        act = np.full((T, A), float(i) + 0.5, dtype='float32')
        rew = np.arange(T, dtype='float32') + i
        cont = np.ones(T, dtype='float32')
        expert = (np.arange(T) == i).astype('float32')
        dist = np.full((T, n_dist), float(i) + 0.25, dtype='float32')
        buf.add_episode(obs, act, rew, cont, expert=expert, dist=dist)
    B, S = 8, 12
    rng_a = np.random.default_rng(7)
    rng_b = np.random.default_rng(7)
    got = buf.sample(B, S, rng_a)
    ep_idx = rng_b.integers(0, buf.filled, size=B)
    max_start = buf.T - S
    starts = (np.zeros(B, dtype=np.int64) if max_start <= 0
              else rng_b.integers(0, max_start + 1, size=B))
    want_obs = np.zeros((B, S, D), dtype='float32')
    want_act = np.zeros((B, S, A), dtype='float32')
    want_rew = np.zeros((B, S), dtype='float32')
    want_cont = np.zeros((B, S), dtype='float32')
    want_exp = np.zeros((B, S), dtype='float32')
    want_dist = np.zeros((B, S, n_dist), dtype='float32')
    for b in range(B):
        s = starts[b]
        want_obs[b] = buf.obs[ep_idx[b], s:s + S]
        want_act[b] = buf.act[ep_idx[b], s:s + S]
        want_rew[b] = buf.rew[ep_idx[b], s:s + S]
        want_cont[b] = buf.cont[ep_idx[b], s:s + S]
        want_exp[b] = buf.expert[ep_idx[b], s:s + S]
        want_dist[b] = buf.dist[ep_idx[b], s:s + S]
    assert np.array_equal(got['obs'], want_obs)
    assert np.array_equal(got['act'], want_act)
    assert np.array_equal(got['rew'], want_rew)
    assert np.array_equal(got['cont'], want_cont)
    assert np.array_equal(got['expert'], want_exp)
    assert np.array_equal(got['dist'], want_dist)
    print('[smoke] OK  buffer sample matches per-row slice copy')


if __name__ == '__main__':
    _test_buffer_sample_identity()
    _test_weighted_recon()
    for wm in ('rssm', 'sf_transformer'):
        _test_recon_integration(wm)
    _test_freeze_mechanism()
    _test_defaults_and_env()
    print('\n[smoke] ALL WM-FIX SMOKE TESTS PASSED')
