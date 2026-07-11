"""Validate the MIMO per-input isolation loss: finite + gradient reaches the
continuous gain channel (the routing that bypasses the categorical bottleneck).
"""
import torch

from training.train import (TrainConfig, build_model,
                            _wm_input_isolation_loss)


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


if __name__ == '__main__':
    main()
