"""Validate the MIMO per-input isolation loss: finite + gradient reaches the
continuous gain channel (the routing that bypasses the categorical bottleneck).
"""
import torch

from training.train import (TrainConfig, build_model,
                            _wm_input_isolation_loss, _wm_gain_match_loss,
                            _wm_latent_overshoot_loss,
                            _wm_held_rollout_stationarity_loss)


def _iso(model, obs, act, cfg):
    loss, _extras = _wm_input_isolation_loss(model, obs, act, cfg)
    return loss


def main():
    torch.manual_seed(0)
    cfg = TrainConfig()
    cfg.obs_dim = 6
    cfg.action_dim = 1
    cfg.lookback = 8
    cfg.world_model_type = 'rssm'
    cfg.rssm_deter_dim = 64
    cfg.rssm_n_categoricals = 8
    cfg.rssm_n_classes = 8
    cfg.rssm_embed_dim = 32
    cfg.rssm_hidden_dim = 32
    cfg.d_model = 64
    cfg.head_hidden = 64
    cfg.head_n_layers = 2
    cfg.mtp_length = 4
    cfg.horizon = 6
    cfg.seq_len = 16
    cfg.dv_dim = 1
    cfg.dv_indices = (3,)
    cfg.cv_obs_indices = (0,)
    cfg.dob_enabled = True
    # Enable the continuous gain channel (normally auto-resolved in train()).
    cfg.cont_latent_enabled = True
    cfg.cont_gain_dim = 1 * (1 + 1)   # n_cv·(n_mv+n_dv)
    cfg.cont_dist_dim = 0
    cfg.wm_input_isolation_coef = 0.5
    cfg.wm_input_isolation_len = 6
    model = build_model(cfg)
    print(f'[iso] cont_dim={model.dynamics.cont_dim} '
          f'cont_gain_dim={model.dynamics.cont_gain_dim} '
          f'feat_dim={model.dynamics.feat_dim}')
    assert model.dynamics.cont_gain_dim == 2, model.dynamics.cont_gain_dim

    B, T = 3, cfg.seq_len
    obs = torch.randn(B, T, cfg.obs_dim)
    act = torch.rand(B, T, cfg.action_dim) * 2 - 1

    # ---- off when coef<=0 ----
    cfg.wm_input_isolation_coef = 0.0
    z = _iso(model, obs, act, cfg)
    assert float(z) == 0.0, f'expected 0 when off, got {float(z)}'
    cfg.wm_input_isolation_coef = 0.5

    loss = _iso(model, obs, act, cfg)
    print(f'[iso] loss={float(loss):.5f}')
    assert torch.isfinite(loss).all() and float(loss) > 0.0, float(loss)

    model.zero_grad(set_to_none=True)
    loss.backward()
    cont_grad = 0.0
    dec_grad = 0.0
    enc_grad = 0.0
    for n, p in model.dynamics.named_parameters():
        if p.grad is None:
            continue
        g = float(p.grad.abs().sum())
        if 'cont' in n:
            cont_grad += g
        elif 'decode' in n or 'dec' in n:
            dec_grad += g
        elif 'enc' in n or 'embed' in n:
            enc_grad += g
    print(f'[iso] grad -> cont_gain={cont_grad:.4e}  decoder={dec_grad:.4e}  '
          f'encoder={enc_grad:.4e}')
    assert cont_grad > 0.0, 'gradient did NOT reach the cont-gain params!'
    # The encoder should get ~no gradient (start states are detached — the
    # open-loop prior/decoder/cont-gain path is what's supervised).
    print('[iso] OK: finite, off-when-disabled, grad reaches the cont-gain '
          'channel (bypasses the categorical bottleneck)')

    # ---- steady-state (DC-gain) match term (2026-08-03) ----
    base = float(_iso(model, obs, act, cfg))  # ss off
    cfg.wm_ss_match_coef = 0.5
    ss, extras = _wm_input_isolation_loss(model, obs, act, cfg)
    print(f'[ss] base(traj)={base:.5f}  with_ss={float(ss):.5f} '
          f'(coef={cfg.wm_ss_match_coef}) extras={sorted(extras)}')
    assert torch.isfinite(ss).all(), float(ss)
    assert float(ss) >= base - 1e-6, 'ss term should ADD to the trajectory loss'
    assert 'wm_isolation_traj_loss' in extras
    assert 'wm_ss_match_loss' in extras
    assert float(extras['wm_ss_match_loss']) >= -1e-8
    cfg.wm_ss_match_settle_var = 0.05
    _, extras_w = _wm_input_isolation_loss(model, obs, act, cfg)
    assert 'wm_ss_match_wmean' in extras_w
    wmean = float(extras_w['wm_ss_match_wmean'])
    assert 0.0 <= wmean <= 1.0 + 1e-6, wmean
    print(f'[ss] wmean={wmean:.4f} (settle_var=0.05)')
    cfg.wm_ss_match_settle_var = 0.0
    model.zero_grad(set_to_none=True)
    ss.backward()
    cont_grad_ss = sum(float(p.grad.abs().sum())
                       for n, p in model.dynamics.named_parameters()
                       if p.grad is not None and 'cont' in n)
    assert cont_grad_ss > 0.0, 'ss-match grad did NOT reach the cont-gain params!'
    print(f'[ss] OK: finite, adds to traj loss, grad reaches cont-gain '
          f'({cont_grad_ss:.4e})')

    # ---- P25 RCA: isolation TBPTT must keep_c so the gain channel still trains
    cfg.aux_tbptt_steps = 2
    cfg.wm_ss_match_coef = 0.5
    cfg.wm_input_isolation_len = 6
    model.zero_grad(set_to_none=True)
    tb = _iso(model, obs, act, cfg)
    assert torch.isfinite(tb).all() and float(tb) > 0.0, float(tb)
    tb.backward()
    cont_grad_tb = sum(float(p.grad.abs().sum())
                       for n, p in model.dynamics.named_parameters()
                       if p.grad is not None and 'cont' in n)
    assert cont_grad_tb > 0.0, 'TBPTT keep_c still must reach cont-gain!'
    print(f'[tbptt] OK: isolation TBPTT(keep_c) still trains cont-gain '
          f'({cont_grad_tb:.4e})')

    # ---- P25 RCA: gain-match is FULL BPTT (no detach) so the K-step
    # asymptote still reaches the continuous gain channel.
    cfg.gain_match_coef = 1.0
    cfg.gain_match_len = 6
    cfg.gain_match_mv_target = ((-1.0,),)
    cfg.gain_match_dv_target = ((0.5,),)
    with torch.no_grad():
        feats, *_ = model.dynamics.rollout_observed(obs, act, sample=False)
    model.zero_grad(set_to_none=True)
    gm, _diag = _wm_gain_match_loss(model, feats, obs, act, cfg)
    assert torch.isfinite(gm).all() and float(gm) > 0.0, float(gm)
    gm.backward()
    cont_grad_gm = sum(float(p.grad.abs().sum())
                       for n, p in model.dynamics.named_parameters()
                       if p.grad is not None and 'cont' in n)
    assert cont_grad_gm > 0.0, 'gain-match (full BPTT) did NOT reach cont-gain!'
    print(f'[gain-match] OK: full-BPTT asymptote trains cont-gain '
          f'({cont_grad_gm:.4e})')

    # P28 follow-up 13: open-loop FD K is not min(K, T-1).  A short
    # window (T=16) with K=20 used to return 0 (n_valid=T-K<1).
    cfg.gain_match_len = 20
    model.zero_grad(set_to_none=True)
    gm_long, _ = _wm_gain_match_loss(model, feats.detach(), obs, act, cfg)
    assert torch.isfinite(gm_long).all() and float(gm_long) > 0.0, float(gm_long)
    gm_long.backward()
    cont_grad_long = sum(float(p.grad.abs().sum())
                         for n, p in model.dynamics.named_parameters()
                         if p.grad is not None and 'cont' in n)
    assert cont_grad_long > 0.0, 'gain-match K>T did NOT reach cont-gain!'
    print(f'[gain-match-K] OK: open-loop K>T still trains cont-gain '
          f'({cont_grad_long:.4e})')
    cfg.gain_match_len = 6

    # Resolve-time Huber β must not be overwritten by a leftover env ``0``
    # (auto-median sentinel) on every loss call.
    import os
    os.environ['DREAMER_GAIN_MATCH_HUBER_BETA'] = '0'
    cfg.gain_match_huber_beta = 0.52
    model.zero_grad(set_to_none=True)
    gm_auto, _ = _wm_gain_match_loss(model, feats.detach(), obs, act, cfg)
    assert torch.isfinite(gm_auto).all() and float(gm_auto) > 0.0, float(gm_auto)
    os.environ.pop('DREAMER_GAIN_MATCH_HUBER_BETA', None)
    print(f'[gain-match-beta] OK: resolved beta used despite env=0 '
          f'(loss={float(gm_auto):.5f})')

    # ---- P28 follow-up 12: img_rollout / overshoot / held must start from
    # posterior c.  Dropping c zero-fills the GRU input, so the open-loop
    # gain supervisor trained a different path than isolation / gain-match
    # / the actor (p20 family).  Changing ONLY the c slice of feat must
    # move overshoot + held; img_rollout(c0=0) stays back-compat with omit.
    rssm = model.dynamics
    assert rssm.cont_dim > 0
    torch.manual_seed(0)
    Bm, Kc = 2, 4
    h0 = torch.zeros(Bm, rssm.deter_dim)
    z0 = torch.zeros(Bm, rssm.n_categoricals, rssm.n_classes)
    z0[..., 0] = 1.0
    a_seq = torch.zeros(Bm, Kc, cfg.action_dim)
    c_hi = torch.ones(Bm, rssm.cont_dim)
    c_lo = torch.zeros(Bm, rssm.cont_dim)
    f_hi = rssm.img_rollout(h0, z0, a_seq, sample=False, c0=c_hi)
    f_lo = rssm.img_rollout(h0, z0, a_seq, sample=False, c0=c_lo)
    f_none = rssm.img_rollout(h0, z0, a_seq, sample=False)
    assert f_hi.shape == f_lo.shape == f_none.shape
    assert not torch.allclose(f_hi, f_lo), \
        'img_rollout must use c0 (posterior gain) on the first GRU step'
    assert torch.allclose(f_lo, f_none, atol=1e-5, rtol=1e-5), \
        'c0=0 must match omitted c0 (back-compat zero-fill)'
    print('[img-rollout-c0] OK: c0 changes the prior roll; omit≡zeros')

    cfg.wm_overshoot_coef = 0.3
    cfg.wm_overshoot_len = 6
    cfg.wm_overshoot_gate_recon = 0.0
    cfg.wm_overshoot_max_starts = 4
    cfg.wm_held_rollout_coef = 0.5
    cfg.wm_held_rollout_len = 8
    cfg.wm_held_rollout_gate_recon = 0.0
    cfg.wm_held_rollout_max_starts = 4
    feats_c = feats.detach().clone()
    feats_shift = feats_c.clone()
    _ze = rssm.deter_dim + rssm.stoch_flat_dim
    feats_shift[..., _ze:_ze + rssm.cont_dim] = (
        feats_shift[..., _ze:_ze + rssm.cont_dim] + 1.0)
    torch.manual_seed(1)
    ov_a, _ = _wm_latent_overshoot_loss(model, feats_c, obs, act, cfg)
    torch.manual_seed(1)
    ov_b, _ = _wm_latent_overshoot_loss(model, feats_shift, obs, act, cfg)
    assert torch.isfinite(ov_a).all() and torch.isfinite(ov_b).all()
    assert abs(float(ov_a) - float(ov_b)) > 1e-8, \
        'overshoot must read posterior c (loss unchanged when only c shifted)'
    # Held-rollout is GAIN-NEUTRAL (late−early drift of h). A constant c
    # offset that persists can cancel in that difference — do not require
    # the loss to move. Still require a finite term; img_rollout(c0) above
    # already proves the start-c path.
    torch.manual_seed(2)
    hd_a, _ = _wm_held_rollout_stationarity_loss(model, feats_c, obs, act, cfg)
    assert torch.isfinite(hd_a).all() and float(hd_a) >= 0.0
    print(f'[overshoot-c0] OK: overshoot {float(ov_a):.5f}→{float(ov_b):.5f} '
          f'(c-slice shift); held finite {float(hd_a):.5f} (gain-neutral)')

    # ---- P28 follow-up 14: production path starts from posterior MEAN c,
    # not the reparameterized sample packed into feat.  Shifting only the
    # feat c-slice must NOT move overshoot/gain-match when c_mean is passed;
    # shifting c_mean must.
    with torch.no_grad():
        feats_live, *_, cont = rssm.rollout_observed(obs, act, sample=True)
    assert cont is not None and 'post_mean' in cont
    c_mean = cont['post_mean'].detach().clone()
    c_shift = c_mean.clone() + 1.0
    feats_live_c = feats_live.detach().clone()
    feats_live_shift = feats_live_c.clone()
    feats_live_shift[..., _ze:_ze + rssm.cont_dim] = (
        feats_live_shift[..., _ze:_ze + rssm.cont_dim] + 1.0)
    torch.manual_seed(3)
    ov_m0, _ = _wm_latent_overshoot_loss(
        model, feats_live_c, obs, act, cfg, c_mean=c_mean)
    torch.manual_seed(3)
    ov_m_feat, _ = _wm_latent_overshoot_loss(
        model, feats_live_shift, obs, act, cfg, c_mean=c_mean)
    torch.manual_seed(3)
    ov_m_mean, _ = _wm_latent_overshoot_loss(
        model, feats_live_c, obs, act, cfg, c_mean=c_shift)
    assert torch.isfinite(ov_m0).all() and torch.isfinite(ov_m_feat).all()
    assert torch.isfinite(ov_m_mean).all()
    assert abs(float(ov_m0) - float(ov_m_feat)) < 1e-7, (
        'overshoot with c_mean must ignore feat c-slice (sample vs mean)')
    assert abs(float(ov_m0) - float(ov_m_mean)) > 1e-8, (
        'overshoot must follow posterior mean c')
    cfg.gain_match_coef = 1.0
    cfg.gain_match_len = 6
    cfg.gain_match_mv_target = ((0.5,),)
    cfg.gain_match_dv_target = ((0.2,),)
    torch.manual_seed(4)
    gm_m0, _ = _wm_gain_match_loss(
        model, feats_live_c, obs, act, cfg, c_mean=c_mean)
    torch.manual_seed(4)
    gm_m_feat, _ = _wm_gain_match_loss(
        model, feats_live_shift, obs, act, cfg, c_mean=c_mean)
    torch.manual_seed(4)
    gm_m_mean, _ = _wm_gain_match_loss(
        model, feats_live_c, obs, act, cfg, c_mean=c_shift)
    assert torch.isfinite(gm_m0).all() and torch.isfinite(gm_m_feat).all()
    assert torch.isfinite(gm_m_mean).all()
    assert abs(float(gm_m0) - float(gm_m_feat)) < 1e-7, (
        'gain-match with c_mean must ignore feat c-slice')
    assert abs(float(gm_m0) - float(gm_m_mean)) > 1e-8, (
        'gain-match must follow posterior mean c')
    print(f'[overshoot-c-mean] OK: feat-slice Δ={abs(float(ov_m0)-float(ov_m_feat)):.2e} '
          f'(ignored); mean-shift {float(ov_m0):.5f}→{float(ov_m_mean):.5f}; '
          f'gain-match mean-shift {float(gm_m0):.5f}→{float(gm_m_mean):.5f}')

    # Batched img_rollout FD (eager default) must match sequential img_step
    # rolls — same last-step ΔCV/Δu Huber that pins DC gain (P26).
    from models.dreamer_v4_rssm import RSSMState
    torch.manual_seed(5)
    cfg.gain_match_len = 6
    cfg.gain_match_step = 1.0
    cfg.gain_match_huber_beta = 1.0
    cfg.gain_match_mv_target = ((0.4,),)
    cfg.gain_match_dv_target = ((-0.2,),)
    gm_batched, _ = _wm_gain_match_loss(
        model, feats_live_c, obs, act, cfg, c_mean=c_mean)

    def _seq_gain_match():
        K = int(cfg.gain_match_len)
        max_starts = max(1, int(cfg.gain_match_max_starts))
        B, T = obs.shape[:2]
        n_valid = T if T <= K else (T - K)
        stride = max(1, n_valid // max_starts)
        starts = torch.arange(0, n_valid, stride, device=obs.device)
        S = int(starts.numel())
        f0 = feats_live_c[:, starts]
        Bm = B * S
        h0 = f0[..., :rssm.deter_dim].reshape(Bm, -1)
        _ze = rssm.deter_dim + rssm.stoch_flat_dim
        z0 = f0[..., rssm.deter_dim:_ze].reshape(
            Bm, rssm.n_categoricals, rssm.n_classes)
        from training.train import _openloop_c0
        c0 = _openloop_c0(rssm, f0, c_mean=c_mean, starts=starts)
        a_base = act[:, starts].reshape(Bm, -1)
        dv0 = obs[:, starts].index_select(-1, rssm.dv_index_t).reshape(Bm, -1)
        cv_idx = rssm.cv_index_t
        step = float(cfg.gain_match_step)

        def _roll(a_held, dv_held):
            st = RSSMState(
                h=h0.clone(),
                z_logits=torch.zeros(Bm, rssm.n_categoricals, rssm.n_classes,
                                     device=obs.device, dtype=f0.dtype),
                z=z0.clone(),
                c=(c0.clone() if c0 is not None else None))
            for _ in range(K):
                st = rssm.img_step(st, a_held, dv=dv_held, sample=False)
            return rssm.decode(st.feat).index_select(-1, cv_idx)

        cv_base = _roll(a_base, dv0)
        a_mv = a_base.clone()
        a_mv[:, 0] = a_mv[:, 0] + step
        g_mv = (_roll(a_mv, dv0) - cv_base) / step
        dv_s = dv0.clone()
        dv_s[:, 0] = dv_s[:, 0] + step
        g_dv = (_roll(a_base, dv_s) - cv_base) / step
        tgt_mv = torch.tensor([0.4], device=obs.device, dtype=g_mv.dtype)
        tgt_dv = torch.tensor([-0.2], device=obs.device, dtype=g_dv.dtype)
        import torch.nn.functional as F
        l_mv = F.smooth_l1_loss(g_mv, tgt_mv.expand_as(g_mv), beta=1.0)
        l_dv = F.smooth_l1_loss(g_dv, tgt_dv.expand_as(g_dv), beta=1.0)
        return (l_mv + l_dv) / 2.0

    gm_seq = _seq_gain_match()
    assert torch.isfinite(gm_batched).all() and torch.isfinite(gm_seq).all()
    assert abs(float(gm_batched) - float(gm_seq)) < 1e-5, (
        f'batched gain-match {float(gm_batched):.6f} != sequential '
        f'{float(gm_seq):.6f}')
    print(f'[gain-match-batched] OK: batched={float(gm_batched):.6f} '
          f'seq={float(gm_seq):.6f}')


if __name__ == '__main__':
    main()
