"""CPU smoke + correctness test for the transformer-SSM (TSSM) backbone.

neural-apc-mbrl, 2026-06-06.  GPU-FREE — runs on a tiny model on CPU, so it does
not touch any live GPU training run.  The TSSM is NOT yet wired into build_model
dispatch; this exercises the backbone directly.

Verifies:
  1. Interface shapes match RSSMDynamics (embed / rollout_observed / img_step /
     obs_step / decode / feat_dim), and the state duck-types RSSMState
     (.h, .z_logits, .z, .feat, .stoch_flat).
  2. KL-ability: post_logits / prior_logits have shape (B, T, K, C) and are finite.
  3. Straight-through gradient REACHES the transformer + prior_net (the overshoot
     and held-rollout losses rely on sample=True grad flowing to the prior).
  4. Determinism: sample=False img_step is deterministic.
  5. CORRECTNESS GATE (the future KV-cache must match this): stepwise img_step
     over a fixed (z, action) sequence == a single full-sequence causal-
     transformer forward over the same tokens (h_stepwise ≈ h_full).

Run:
  PYTHONPATH=$PWD \
  $PWD/../neural-apc-mbrl-env/bin/python tools/_smoke_tssm.py
"""
import torch

from models.transformer_ssm import (TransformerSSMConfig,
                                     TransformerSSMDynamics, TSSMState)


def _mk(seed=0):
    torch.manual_seed(seed)
    cfg = TransformerSSMConfig(
        obs_dim=6, action_dim=2, deter_dim=32, n_categoricals=4, n_classes=4,
        embed_dim=16, n_layers=2, n_heads=4, max_seq_len=64)
    return cfg, TransformerSSMDynamics(cfg).eval()  # eval -> dropout off


def test_interface_shapes():
    cfg, m = _mk()
    B, T = 3, 7
    obs = torch.randn(B, T, cfg.obs_dim)
    act = torch.rand(B, T, cfg.action_dim) * 2 - 1
    feats, post_lg, prior_lg, last = m.rollout_observed(obs, act, sample=True)
    F = m.feat_dim
    assert feats.shape == (B, T, F), feats.shape
    assert post_lg.shape == (B, T, cfg.n_categoricals, cfg.n_classes), post_lg.shape
    assert prior_lg.shape == (B, T, cfg.n_categoricals, cfg.n_classes), prior_lg.shape
    assert torch.isfinite(feats).all() and torch.isfinite(post_lg).all()
    # state duck-types RSSMState
    assert last.feat.shape == (B, F)
    assert last.stoch_flat.shape == (B, cfg.n_categoricals * cfg.n_classes)
    assert last.h.shape == (B, cfg.deter_dim)
    # decode round-trips feat -> obs
    dec = m.decode(feats)
    assert dec.shape == (B, T, cfg.obs_dim), dec.shape
    print(f"[smoke] OK interface shapes: feat_dim={F} "
          f"feats{tuple(feats.shape)} logits{tuple(post_lg.shape)}")


def test_st_grad_reaches_prior_and_transformer():
    cfg, m = _mk()
    B = 3
    state = m.initial_state(B, torch.device('cpu'))
    # roll a few prior steps under a held action (sample=True straight-through)
    total = torch.zeros(())
    for _ in range(4):
        state = m.img_step(state, torch.rand(B, cfg.action_dim) * 2 - 1,
                           sample=True)
        total = total + m.decode(state.feat).pow(2).mean()
    m.zero_grad(set_to_none=True)
    total.backward()
    prior_g = sum(float(p.grad.abs().sum()) for p in m.prior_net.parameters()
                  if p.grad is not None)
    tf_g = sum(float(p.grad.abs().sum()) for p in m.transformer.parameters()
               if p.grad is not None)
    tok_g = sum(float(p.grad.abs().sum()) for p in m.token_proj.parameters()
                if p.grad is not None)
    assert prior_g > 0.0, "ST grad did NOT reach prior_net"
    assert tf_g > 0.0, "grad did NOT reach the transformer"
    assert tok_g > 0.0, "grad did NOT reach token_proj (z->token path broken)"
    print(f"[smoke] OK ST grad reaches prior_net (|g|={prior_g:.3f}), "
          f"transformer (|g|={tf_g:.3f}), token_proj (|g|={tok_g:.3f})")


def test_determinism_mode():
    cfg, m = _mk()
    B = 3
    s0 = m.initial_state(B, torch.device('cpu'))
    a = torch.rand(B, cfg.action_dim) * 2 - 1
    with torch.no_grad():
        h1 = m.img_step(s0, a, sample=False).h
        h2 = m.img_step(s0, a, sample=False).h
    assert torch.allclose(h1, h2, atol=1e-6), "sample=False img_step not deterministic"
    print("[smoke] OK sample=False img_step deterministic")


def test_stepwise_equals_full_sequence():
    """CORRECTNESS GATE for the future KV-cache: stepwise img_step over a fixed
    (z, action) sequence must equal a single full-sequence causal forward over
    the same tokens.  Builds the tokens from a FIXED z-sequence (sample=False so
    z is deterministic) and compares h at every step."""
    cfg, m = _mk()
    B, K = 2, 8
    with torch.no_grad():
        # Fixed action sequence; z evolves deterministically (sample=False).
        acts = torch.rand(B, K, cfg.action_dim) * 2 - 1
        # ---- stepwise (windowed recompute) ----
        state = m.initial_state(B, torch.device('cpu'))
        h_step, tokens = [], []
        for t in range(K):
            tok = m._build_token(state.z, acts[:, t])     # token from prev.z
            tokens.append(tok)
            state = m.img_step(state, acts[:, t], sample=False)
            h_step.append(state.h)
        h_step = torch.stack(h_step, dim=1)               # (B, K, d)
        # ---- full sequence over the SAME tokens, single forward ----
        window = torch.stack(tokens, dim=1)               # (B, K, d)
        h_full = m._encode_window(window)                 # (B, K, d)
    max_err = float((h_step - h_full).abs().max())
    assert max_err < 1e-4, f"stepwise != full-sequence (max_err={max_err})"
    print(f"[smoke] OK stepwise img_step == full-sequence forward "
          f"(max_err={max_err:.2e}) -- KV-cache target validated")


def test_end_to_end_dreamer_tssm():
    """Build a full DreamerV4(world_model_type='tssm') and confirm the whole
    RSSM pipeline works on it: WM loss + imagination step run + finite + the
    disturbance estimator head is present (granted automatically by the
    feat-dim build branch)."""
    from training.train import (TrainConfig, build_model, world_model_loss,
                                imagination_step)
    torch.manual_seed(0)
    cfg = TrainConfig()
    cfg.obs_dim, cfg.action_dim = 6, 2
    cfg.world_model_type = 'tssm'
    cfg.tssm_d_model, cfg.tssm_n_layers, cfg.tssm_n_heads = 48, 2, 4
    cfg.rssm_n_categoricals, cfg.rssm_n_classes, cfg.rssm_embed_dim = 4, 4, 16
    cfg.lookback, cfg.seq_len, cfg.horizon = 8, 16, 4
    cfg.mtp_length = 4
    cfg.disturbance_head_dim = 1          # unmeasured-disturbance estimator
    cfg.compile_mode = 'off'
    model = build_model(cfg)
    assert model.world_model_type == 'tssm'
    assert type(model.dynamics).__name__ == 'TransformerSSMDynamics'
    assert model.disturbance is not None, "disturbance head NOT built for TSSM"
    B, T = 3, cfg.seq_len
    batch = {
        'obs': torch.randn(B, T, cfg.obs_dim),
        'act': torch.rand(B, T, cfg.action_dim) * 2 - 1,
        'rew': torch.randn(B, T),
        'cont': torch.ones(B, T),
        'expert': torch.zeros(B, T),
        'dist': torch.randn(B, T, 1),
    }
    losses, _, _ = world_model_loss(model, batch, cfg)
    assert torch.isfinite(losses['wm_total']).all()
    assert 'disturbance_loss' in losses, "disturbance loss missing for TSSM"
    # overshoot/held-rollout no-op for TSSM (documented compat decision)
    assert float(losses.get('wm_overshoot_loss', 0.0)) == 0.0
    assert float(losses.get('wm_held_rollout_loss', 0.0)) == 0.0
    losses['wm_total'].backward()
    diag = imagination_step(model, batch, cfg)
    assert torch.isfinite(diag['critic_loss']).all()
    assert torch.isfinite(diag['actor_loss']).all()
    print("[smoke] OK end-to-end DreamerV4(tssm): WM loss + imagination run, "
          "disturbance head built, overshoot/held no-op (compat)")


def test_diagnostics_probes_route_tssm():
    """The WM fidelity / critic-calibration / feat-from-window probes must treat
    TSSM as rssm-interface (not route it into the SF tokenizer path, which would
    AttributeError on model.tokenizer=None)."""
    from evaluation.diagnostics import _is_rssm_like, _rssm_feat_from_window
    from tools.wm_steady_state_diagnostic import _is_rssm_model
    from training.train import TrainConfig, build_model
    import numpy as np
    torch.manual_seed(0)
    cfg = TrainConfig()
    cfg.obs_dim, cfg.action_dim = 6, 2
    cfg.world_model_type = 'tssm'
    cfg.tssm_d_model, cfg.tssm_n_layers, cfg.tssm_n_heads = 48, 2, 4
    cfg.rssm_n_categoricals, cfg.rssm_n_classes, cfg.rssm_embed_dim = 4, 4, 16
    cfg.compile_mode = 'off'
    model = build_model(cfg)
    assert _is_rssm_like(model), "diagnostics._is_rssm_like(tssm) should be True"
    assert _is_rssm_model(model), "steady_state._is_rssm_model(tssm) should be True"
    # _rssm_feat_from_window uses only interface methods -> must work for TSSM.
    L = 8
    feat = _rssm_feat_from_window(
        model, np.random.randn(L, cfg.obs_dim).astype('float32'),
        (np.random.rand(L, cfg.action_dim) * 2 - 1).astype('float32'),
        torch.device('cpu'))
    assert feat.shape == (1, model.dynamics.feat_dim), feat.shape
    print("[smoke] OK diagnostics probes route TSSM as rssm-interface "
          "(_is_rssm_like + _is_rssm_model True; feat_from_window works)")


if __name__ == '__main__':
    test_interface_shapes()
    test_st_grad_reaches_prior_and_transformer()
    test_determinism_mode()
    test_stepwise_equals_full_sequence()
    test_end_to_end_dreamer_tssm()
    test_diagnostics_probes_route_tssm()
    print("\n[smoke] ALL TSSM checks PASSED")
