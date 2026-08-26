"""Smoke test for the DETERMINISTIC continuous latent (bias-free observer core,
2026-08-12).

Verifies, WITHOUT a real env, that:
  1. ``_CategoricalLatent(latent_type='deterministic')`` emits a CONTINUOUS
     tanh latent (values in (-1,1), has negatives — NOT a one-hot/simplex),
     while 'categorical' stays one-hot.
  2. ``rssm_joint_embed_loss`` (plain latent-space MSE) is finite, >=0, and 0
     when prior==posterior.
  3. ``build_model`` propagates ``rssm_latent_type`` to BOTH backbones'
     prior/post nets (RSSM + TSSM share the head).
  4. ``world_model_loss`` is finite on RSSM AND TSSM for categorical AND
     deterministic; deterministic has kl_loss==0 + joint_embed_loss>0 (the
     imagination consistency), categorical has joint_embed_loss==0 + kl>0;
     ``wm_total.backward()`` works; P3 imagination is finite for deterministic.

Run:
  cd ~/neural-APC-mbrl2 && CUDA_VISIBLE_DEVICES="" DREAMER_COMPILE=0 PYTHONPATH=$PWD \
    ~/neural-APC-mbrl2-env/bin/python tools/_smoke_deterministic.py
"""
import torch

from models.dreamer_v4_rssm import _CategoricalLatent, rssm_joint_embed_loss
from training.train import (TrainConfig, build_model, world_model_loss,
                            _realsim_actor_critic_step, _resolve_compile_mode,
                            _batch_np_to_device, _host_cpu_count,
                            _limit_blas_threads)


def _small_cfg(obs_dim=6, action_dim=2, wm_type='rssm', latent_type='categorical'):
    cfg = TrainConfig()
    cfg.obs_dim = obs_dim
    cfg.action_dim = action_dim
    cfg.lookback = 8
    cfg.world_model_type = wm_type
    cfg.rssm_deter_dim = 64
    cfg.rssm_n_categoricals = 8
    cfg.rssm_n_classes = 8
    cfg.rssm_embed_dim = 32
    cfg.rssm_hidden_dim = 32
    cfg.rssm_latent_type = latent_type
    cfg.tssm_d_model = 64
    cfg.tssm_n_layers = 2
    cfg.tssm_n_heads = 4
    cfg.tssm_max_seq_len = 64
    cfg.d_model = 64
    cfg.head_hidden = 64
    cfg.head_n_layers = 2
    cfg.mtp_length = 4
    cfg.horizon = 4
    cfg.seq_len = 16
    return cfg


def _check_head():
    torch.manual_seed(0)
    x = torch.randn(4, 5, 16)
    K, C = 8, 8
    cat = _CategoricalLatent(16, K, C, hidden_dim=32, latent_type='categorical')
    _, z_cat = cat(x)
    z_cat = z_cat.view(4, 5, K, C).detach()
    assert (z_cat.max(dim=-1).values > 0.99).all(), 'categorical must be ~one-hot'
    det = _CategoricalLatent(16, K, C, hidden_dim=32, latent_type='deterministic')
    _, z = det(x)
    z = z.detach()
    assert float(z.abs().max()) < 1.0, 'deterministic latent must be tanh-bounded (<1)'
    assert float(z.min()) < 0.0, 'deterministic latent must be continuous (has negatives)'
    assert torch.isfinite(z).all()
    lg = torch.randn(4, 5, K, C)
    assert float(rssm_joint_embed_loss(lg, lg)) < 1e-6, 'JE(self)!=0'
    je = rssm_joint_embed_loss(torch.randn(4, 5, K, C), torch.randn(4, 5, K, C))
    assert torch.isfinite(je) and float(je) > 0.0, 'JE must be finite >0'
    print('[smoke] OK  _CategoricalLatent: categorical=one-hot, deterministic=continuous '
          'tanh; joint_embed_loss (MSE) finite')


def _check_backbone(wm_type):
    B, obs_dim, action_dim = 3, 6, 2
    for latent_type in ('categorical', 'deterministic'):
        torch.manual_seed(0)
        cfg = _small_cfg(obs_dim, action_dim, wm_type, latent_type)
        model = build_model(cfg)
        assert model.dynamics.prior_net.latent_type == latent_type, \
            f'{wm_type} prior_net latent_type not propagated'
        assert model.dynamics.post_net.latent_type == latent_type, \
            f'{wm_type} post_net latent_type not propagated'
        T = cfg.seq_len
        batch = {
            'obs': torch.randn(B, T, obs_dim),
            'act': torch.rand(B, T, action_dim) * 2 - 1,
            'rew': torch.randn(B, T),
            'cont': torch.ones(B, T),
            'expert': (torch.rand(B, T) > 0.5).float(),
        }
        losses, _z, agent_hid = world_model_loss(model, batch, cfg)
        assert 'joint_embed_loss' in losses and 'kl_loss' in losses
        for k, v in losses.items():
            if torch.is_tensor(v) and v.is_floating_point():
                assert torch.isfinite(v).all(), f'NON-FINITE {k}'
        je = float(losses['joint_embed_loss'].detach())
        kl = float(losses['kl_loss'].detach())
        if latent_type == 'deterministic':
            assert kl == 0.0, f'{wm_type}/det: kl should be 0 (no variational KL)'
            assert je > 0.0, f'{wm_type}/det: joint_embed should be >0'
        else:
            assert je == 0.0, f'{wm_type}/cat: joint_embed should be 0 (KL used)'
            assert kl > 0.0, f'{wm_type}/cat: kl should be >0'
        losses['wm_total'].backward()
        wt, rc = float(losses['wm_total'].detach()), float(losses['recon_loss'].detach())
        print(f'[smoke] OK  {wm_type}/{latent_type}: wm_total={wt:.4f} recon={rc:.4f} '
              f'kl={kl:.4f} joint_embed={je:.4f}  backward OK')
        if latent_type == 'deterministic':
            diag = _realsim_actor_critic_step(model, batch, cfg)
            for k, v in diag.items():
                if torch.is_tensor(v) and v.is_floating_point():
                    assert torch.isfinite(v).all(), f'P3 NON-FINITE {k}'
            print(f'[smoke] OK  {wm_type}/deterministic P3 imagination finite '
                  f'(actor_loss={float(diag["actor_loss"]):.3f})')


def _check_compile_default_eager():
    import os
    saved = os.environ.pop('DREAMER_COMPILE', None)
    saved_m = os.environ.pop('DREAMER_COMPILE_MODE', None)
    try:
        cfg = TrainConfig()
        assert cfg.compile_mode == ''
        assert _resolve_compile_mode(cfg) == '', (
            'env-free compile must be eager (P29 leftover default-on)'
        )
        os.environ['DREAMER_COMPILE'] = '0'
        assert _resolve_compile_mode(cfg) == ''
        os.environ['DREAMER_COMPILE'] = '1'
        assert _resolve_compile_mode(cfg) == 'default'
        os.environ['DREAMER_COMPILE'] = 'reduce-overhead'
        assert _resolve_compile_mode(cfg) == 'reduce-overhead'
        os.environ.pop('DREAMER_COMPILE', None)
        os.environ['DREAMER_COMPILE_MODE'] = 'reduce-overhead'
        assert _resolve_compile_mode(TrainConfig()) == 'reduce-overhead'
        os.environ.pop('DREAMER_COMPILE_MODE', None)
        off = TrainConfig()
        off.compile_mode = 'off'
        off._explicit_fields = {'compile_mode'}
        os.environ['DREAMER_COMPILE'] = '1'
        assert _resolve_compile_mode(off) == '', (
            'explicit compile_mode=off must beat DREAMER_COMPILE=1'
        )
        os.environ['DREAMER_COMPILE'] = '1'
        from workflow._plant_prepare import (
            apply_dreamer_env_overrides, _as_compile_mode)
        assert _as_compile_mode('1') == 'default'
        assert _as_compile_mode('0') == ''
        cfg2 = TrainConfig()
        apply_dreamer_env_overrides(cfg2)
        assert cfg2.compile_mode == 'default', cfg2.compile_mode
        assert 'compile_mode' in getattr(cfg2, '_explicit_fields', set())
        print('[smoke] OK  compile default eager; DREAMER_COMPILE=1 opt-in '
              '(whitelist + MODE)')
    finally:
        if saved is None:
            os.environ.pop('DREAMER_COMPILE', None)
        else:
            os.environ['DREAMER_COMPILE'] = saved
        if saved_m is None:
            os.environ.pop('DREAMER_COMPILE_MODE', None)
        else:
            os.environ['DREAMER_COMPILE_MODE'] = saved_m


def main():
    assert TrainConfig().rssm_latent_type == 'deterministic', (
        'TrainConfig default must be deterministic (P26 observer / P29 env-free drop)'
    )
    c = TrainConfig()
    assert c.wm_best_restore_at_p2 is False
    assert int(c.n_critics) == 2
    assert c.return_scale_freeze_after_warmup is True
    _check_compile_default_eager()
    import numpy as np
    y = _batch_np_to_device(
        {'obs': np.zeros((2, 4, 3), dtype='float32')}, torch.device('cpu'))
    assert y['obs'].shape == (2, 4, 3)
    assert _host_cpu_count() >= 1
    _btag = _limit_blas_threads(2)
    assert isinstance(_btag, str) and _btag
    print(f'[smoke] OK  _batch_np_to_device CPU + host_cpu_count + blas={_btag}')
    _check_head()
    for wm in ('rssm', 'tssm'):
        _check_backbone(wm)
    print('[smoke] ALL PASS')


if __name__ == '__main__':
    main()
