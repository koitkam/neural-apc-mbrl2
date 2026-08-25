"""Validate the MIMO per-input isolation loss: finite + gradient reaches the
continuous gain channel (the routing that bypasses the categorical bottleneck).
"""
import torch

from training.train import (TrainConfig, build_model,
                            _wm_input_isolation_loss, _wm_gain_match_loss)


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
    z = _wm_input_isolation_loss(model, obs, act, cfg)
    assert float(z) == 0.0, f'expected 0 when off, got {float(z)}'
    cfg.wm_input_isolation_coef = 0.5

    loss = _wm_input_isolation_loss(model, obs, act, cfg)
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
    base = float(_wm_input_isolation_loss(model, obs, act, cfg))  # ss off
    cfg.wm_ss_match_coef = 0.5
    ss = _wm_input_isolation_loss(model, obs, act, cfg)
    print(f'[ss] base(traj)={base:.5f}  with_ss={float(ss):.5f} '
          f'(coef={cfg.wm_ss_match_coef})')
    assert torch.isfinite(ss).all(), float(ss)
    assert float(ss) >= base - 1e-6, 'ss term should ADD to the trajectory loss'
    model.zero_grad(set_to_none=True)
    ss.backward()
    cont_grad_ss = sum(float(p.grad.abs().sum())
                       for n, p in model.dynamics.named_parameters()
                       if p.grad is not None and 'cont' in n)
    assert cont_grad_ss > 0.0, 'ss-match grad did NOT reach the cont-gain params!'
    print(f'[ss] OK: finite, adds to traj loss, grad reaches cont-gain '
          f'({cont_grad_ss:.4e})')

    # ---- P25 RCA: isolation TBPTT must keep_c so the gain channel still trains
    import os
    os.environ['DREAMER_AUX_TBPTT_STEPS'] = '2'
    cfg.wm_ss_match_coef = 0.5
    cfg.wm_input_isolation_len = 6
    model.zero_grad(set_to_none=True)
    tb = _wm_input_isolation_loss(model, obs, act, cfg)
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

    # ---- P26 RCA / P27: RELATIVE Huber equalizes MV vs subdominant DV ----
    # Absolute Huber on |tgt_mv|>>|tgt_dv| under-weights the DV residual.
    # Relative (err = (g-tgt)/|tgt|) makes a same-ratio error cost the same.
    cfg.gain_match_relative = 0.0
    model.zero_grad(set_to_none=True)
    gm_abs, _ = _wm_gain_match_loss(model, feats.detach(), obs, act, cfg)
    cfg.gain_match_relative = 1.0
    model.zero_grad(set_to_none=True)
    gm_rel, _ = _wm_gain_match_loss(model, feats.detach(), obs, act, cfg)
    assert torch.isfinite(gm_rel).all() and float(gm_rel) > 0.0, float(gm_rel)
    gm_rel.backward()
    cont_grad_rel = sum(float(p.grad.abs().sum())
                        for n, p in model.dynamics.named_parameters()
                        if p.grad is not None and 'cont' in n)
    assert cont_grad_rel > 0.0, 'relative gain-match did NOT reach cont-gain!'
    print(f'[gain-match-rel] OK: abs={float(gm_abs):.5f} rel={float(gm_rel):.5f} '
          f'cont_grad={cont_grad_rel:.4e}')


if __name__ == '__main__':
    main()
