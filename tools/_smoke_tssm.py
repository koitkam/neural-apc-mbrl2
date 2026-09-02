"""CPU smoke + correctness test for the transformer-SSM (TSSM) backbone.

GPU-FREE — tiny model on CPU; safe alongside a live GPU training run.

Verifies:
  1. Interface shapes match RSSMDynamics (embed / rollout_observed / img_step /
     img_rollout / obs_step / decode / feat_dim), and the state duck-types
     RSSMState (.h, .z_logits, .z, .feat, .stoch_flat).
  2. KL-ability: post_logits / prior_logits have shape (B, T, K, C) and are finite.
  3. Straight-through gradient REACHES the transformer + prior_net (the overshoot
     and held-rollout losses rely on sample=True grad flowing to the prior).
  4. Determinism: sample=False img_step is deterministic.
  5. CORRECTNESS GATE: stepwise img_step over a fixed (z, action) sequence == a
     single full-sequence causal-transformer forward over the same tokens.
  6. ``img_rollout`` ≡ sequential ``img_step`` (gain-match batched FD path).

Run:
  CUDA_VISIBLE_DEVICES="" PYTHONPATH=$PWD \\
  ~/neural-APC-mbrl2-env/bin/python tools/_smoke_tssm.py
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
    feats, post_lg, prior_lg, last, *_ = m.rollout_observed(obs, act, sample=True)
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
    tf_g = sum(float(p.grad.abs().sum()) for p in m.blocks.parameters()
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


def test_img_rollout_equals_img_step():
    """Gain-match batched FD: img_rollout ≡ sequential img_step (fresh cache)."""
    cfg, m = _mk()
    B, K = 3, 5
    torch.manual_seed(0)
    h0 = torch.randn(B, cfg.deter_dim)
    z0 = torch.zeros(B, cfg.n_categoricals, cfg.n_classes)
    z0[..., 0] = 1.0
    acts = torch.rand(B, K, cfg.action_dim) * 2 - 1
    with torch.no_grad():
        roll = m.img_rollout(h0, z0, acts, sample=False)
        state = TSSMState(
            h=h0.clone(),
            z_logits=torch.zeros(B, cfg.n_categoricals, cfg.n_classes),
            z=z0.clone(), kv_cache=None, pos=0)
        seq = []
        for k in range(K):
            state = m.img_step(
                state, acts[:, k], sample=False)
            seq.append(state.feat)
        seq = torch.stack(seq, dim=1)
    max_err = float((roll - seq).abs().max())
    assert max_err < 1e-6, f"img_rollout != img_step (max_err={max_err})"
    last = m.img_rollout(h0, z0, acts, sample=False, last_only=True)
    last_err = float((last - roll[:, -1]).detach().abs().max())
    assert last_err < 1e-6, f"last_only != stack[:, -1] (max_err={last_err})"
    try:
        m.img_rollout(h0, z0, acts, sample=False, out='h')
        raise AssertionError("out='h' should be removed")
    except ValueError as exc:
        assert 'out' in str(exc)
    obs_roll = m.img_rollout(h0, z0, acts, sample=False, out='obs')
    obs_err = float((obs_roll - m.decode(roll)).detach().abs().max())
    assert obs_err < 1e-5, f"out='obs' != decode(feat) (max_err={obs_err})"
    last_obs = m.img_rollout(h0, z0, acts, sample=False, last_only=True, out='obs')
    last_obs_err = float((last_obs - obs_roll[:, -1]).detach().abs().max())
    assert last_obs_err < 1e-5, f"last_only out='obs' != stack[:, -1] (max_err={last_obs_err})"
    last_s, st = m.img_rollout(
        h0, z0, acts, sample=False, last_only=True, out='obs',
        return_state=True)
    assert float((last_s - last_obs).detach().abs().max()) < 1e-6
    acts2 = torch.rand(B, 2, cfg.action_dim) * 2 - 1
    cont = m.img_rollout(
        st.h, st.z, acts2, sample=False, last_only=True, out='obs',
        c0=st.c, prev_state=st.detach())
    long = m.img_rollout(
        h0, z0, torch.cat([acts, acts2], dim=1), sample=False,
        last_only=True, out='obs')
    cont_err = float((cont - long).detach().abs().max())
    assert cont_err < 1e-5, f'prev_state continue != long roll (max_err={cont_err})'
    print(f"[smoke] OK img_rollout ≡ sequential img_step (max_err={max_err:.2e}); "
          f"last_only ≡ stack[:, -1] (max_err={last_err:.2e}); "
          f"out=obs identity (obs={obs_err:.2e} last_obs={last_obs_err:.2e})")
    want_in = (int(m.stoch_flat_dim) + int(m.recurrence_c_dim)
               + int(cfg.action_dim) + int(m.dv_dim))
    assert int(m.recurrence_c_dim) == int(m.cont_dim)
    assert int(m.token_proj.in_features) == want_in, (
        f'token_proj in={m.token_proj.in_features} want {want_in}')


def test_img_step_det_roll_skips_sample():
    """Gain det-roll: sample=True prior c is the mean; skip discarded randn."""
    torch.manual_seed(0)
    cfg = TransformerSSMConfig(
        obs_dim=6, action_dim=2, deter_dim=32, n_categoricals=4, n_classes=4,
        embed_dim=16, n_layers=2, n_heads=4, max_seq_len=64,
        cont_gain_dim=2)
    m = TransformerSSMDynamics(cfg).eval()
    B = 3
    state = m.initial_state(B, torch.device('cpu'))
    a = torch.zeros(B, cfg.action_dim)
    samples = []
    orig = m.cont_prior_net.forward

    def _spy(x, sample=True):
        samples.append(bool(sample))
        return orig(x, sample=sample)

    m.cont_prior_net.forward = _spy
    s1 = m.img_step(state, a, sample=True)
    s0 = m.img_step(state, a, sample=False)
    assert samples == [False, False], samples
    assert torch.allclose(s1.c, s0.c)
    print("[smoke] OK TSSM img_step det-roll skips discarded prior-c sample")


def test_initial_state_zeros_cache():
    """``initial_state`` reuses zero/one-hot ICs; img_step does not write them."""
    cfg, m = _mk()
    B = 3
    device = torch.device('cpu')
    s1 = m.initial_state(B, device)
    s2 = m.initial_state(B, device)
    assert s1.h is s2.h
    assert s1.z is s2.z
    assert s1.z_logits is s2.z_logits
    assert float(s1.z[..., 0].min()) == 1.0
    h_before = s1.h.clone()
    a = torch.zeros(B, cfg.action_dim)
    _ = m.img_step(s1, a, sample=False)
    assert torch.equal(s1.h, h_before)
    assert m.initial_state(B, device).h is s1.h
    print("[smoke] OK TSSM initial_state zero/one-hot cache")


def test_store_aux_feats_identity():
    """Isolation encode may drop logit stacks; feats must match the full pass."""
    cfg, m = _mk()
    B, T = 2, 6
    obs = torch.randn(B, T, cfg.obs_dim)
    act = torch.rand(B, T, cfg.action_dim) * 2 - 1
    with torch.no_grad():
        f_full, post, prior, *_ = m.rollout_observed(obs, act, sample=False)
        f_iso, post2, prior2, *_ = m.rollout_observed(
            obs, act, sample=False, store_aux=False)
    assert post is not None and prior is not None
    assert post2 is None and prior2 is None
    err = float((f_full - f_iso).abs().max())
    assert err < 1e-6, f"store_aux=False feats drifted (max_err={err})"
    with torch.no_grad():
        f_last, _, _, st_last, *_ = m.rollout_observed(
            obs, act, sample=False, store_aux=False, last_only=True)
        _, _, _, st_full, *_ = m.rollout_observed(
            obs, act, sample=False, store_aux=False)
    last_err = float((f_last[:, 0] - f_full[:, -1]).abs().max())
    assert last_err < 1e-6, f'last_only feats != stack[:, -1] (max_err={last_err})'
    from models.dreamer_v4_rssm import _append_decode_core, _stack_decode_core
    dec_in = (m.deter_dim + m.stoch_flat_dim + m.cont_dim + m._dv_feed_dim)
    emb = m.embed(obs)
    dvs = (obs.index_select(-1, m.dv_index_t) if m.dv_dim > 0 else None)
    st = m.initial_state(B, obs.device)
    feat_l, hh, zz, cc, ddv = [], [], [], [], []
    for t in range(T):
        dv_t = None if dvs is None else dvs[:, t]
        post, _ = m.obs_step(
            st, act[:, t], emb[:, t], dv=dv_t, sample=False, obs=None)
        feat_l.append(post.feat[..., :dec_in])
        _append_decode_core(hh, zz, cc, ddv, post)
        st = post
    stack_err = float(
        (_stack_decode_core(hh, zz, cc, ddv) - torch.stack(feat_l, 1)
         ).detach().abs().max())
    assert stack_err < 1e-6, f'_stack_decode_core != feat[..., :dec_in] ({stack_err})'
    h_err = float((st_last.h - st_full.h).abs().max())
    assert h_err < 1e-6, h_err
    _, _, _, st_nf, *_ = m.rollout_observed(
        obs, act, sample=False, store_aux=False, last_only=True,
        return_feats=False)
    nf_h = float((st_nf.h.detach() - st_full.h).abs().max())
    assert nf_h < 1e-6, nf_h
    # dob off (default): last_only already uses _posterior_step.  Confirm
    # prior_net is unused on that path.
    m.train()
    m.zero_grad(set_to_none=True)
    _, _, _, st_b, *_ = m.rollout_observed(
        obs, act, sample=False, store_aux=False, last_only=True,
        return_feats=False)
    st_b.h.sum().backward()
    prior_g = sum(float(p.grad.abs().sum()) for p in m.prior_net.parameters()
                  if p.grad is not None)
    post_g = sum(float(p.grad.abs().sum()) for p in m.post_net.parameters()
                 if p.grad is not None)
    assert prior_g == 0.0, f'TSSM last_only used prior_net (|g|={prior_g})'
    assert post_g > 0.0, 'TSSM last_only lost post_net gradient'
    print(f"[smoke] OK store_aux=False feats identity (max_err={err:.2e}); "
          f"observed last_only ≡ stack[:, -1] (feat={last_err:.2e} h={h_err:.2e}); "
          f"stack-core ≡ feat slice ({stack_err:.2e}); "
          f"return_feats=False h={nf_h:.2e}; Stage-1 prior_net |g|={prior_g:.1f} "
          f"post_net |g|={post_g:.3f}")


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
                                _realsim_actor_critic_step)
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
    cfg.wm_overshoot_len = 4
    cfg.wm_held_rollout_len = 8
    cfg.wm_held_rollout_win = 2
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
    # RSSM-interface: overshoot runs; held is every-other so the first WM
    # step stays 0 (same cadence as RSSM).
    assert float(losses.get('wm_overshoot_loss', 0.0)) > 0.0
    assert float(losses.get('wm_held_rollout_loss', 0.0)) == 0.0
    losses['wm_total'].backward()
    diag = _realsim_actor_critic_step(model, batch, cfg)
    assert torch.isfinite(diag['critic_loss']).all()
    assert torch.isfinite(diag['actor_loss']).all()
    print("[smoke] OK end-to-end DreamerV4(tssm): WM loss + real-sim actor, "
          "disturbance head built, overshoot live (held every-other skip)")


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
    test_img_rollout_equals_img_step()
    test_img_step_det_roll_skips_sample()
    test_initial_state_zeros_cache()
    test_store_aux_feats_identity()
    test_stepwise_equals_full_sequence()
    test_end_to_end_dreamer_tssm()
    test_diagnostics_probes_route_tssm()
    print("\n[smoke] ALL TSSM checks PASSED")
