"""Smoke test for the SimNorm soft-simplicial latent + joint-embedding loss
(round-9 WM structural fix, 2026-08-11).

Verifies, WITHOUT a real env, that:
  1. ``_CategoricalLatent(latent_type='simnorm')`` emits SOFT simplices
     (per-group sum-to-1, NOT hard one-hot) while 'categorical' stays one-hot.
  2. ``build_model`` propagates ``rssm_latent_type`` to BOTH backbones'
     prior/post nets (RSSM + TSSM share the head — the transfer contract).
  3. ``world_model_loss`` is finite, exposes ``joint_embed_loss`` (>0 when the
     coef is on, ==0 when off), and ``wm_total.backward()`` works, for
     categorical AND simnorm, on RSSM AND TSSM.

Run:
  cd ~/neural-APC-mbrl2 && CUDA_VISIBLE_DEVICES="" PYTHONPATH=$PWD \
    ~/neural-APC-mbrl2-env/bin/python tools/_smoke_simnorm.py
"""
import torch

from models.dreamer_v4_rssm import _CategoricalLatent, rssm_joint_embed_loss
from training.train import (TrainConfig, build_model, world_model_loss,
                            _realsim_actor_critic_step)


def _small_cfg(obs_dim=6, action_dim=2, wm_type='rssm',
               latent_type='categorical', je_coef=0.0):
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
    cfg.rssm_simnorm_temp = 1.0
    cfg.rssm_joint_embed_coef = je_coef
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
    # categorical: hard straight-through one-hot (each group has one ~1 entry;
    # tiny float noise from ``sample_oh + probs - probs.detach()``).
    cat = _CategoricalLatent(16, K, C, hidden_dim=32, latent_type='categorical')
    _, z_cat = cat(x)
    z_cat = z_cat.view(4, 5, K, C).detach()
    grp_sum = z_cat.sum(-1)
    assert torch.allclose(grp_sum, torch.ones_like(grp_sum), atol=1e-4), \
        'cat not sum-1'
    assert (z_cat.max(dim=-1).values > 0.99).all(), \
        'categorical must be ~hard one-hot (a dominant ~1 per group)'
    # simnorm: soft simplices (sum-1 per group, strictly interior => not one-hot).
    sn = _CategoricalLatent(16, K, C, hidden_dim=32, latent_type='simnorm',
                            simnorm_temp=1.0)
    _, z_sn = sn(x)
    z_sn = z_sn.view(4, 5, K, C).detach()
    grp_sum = z_sn.sum(-1)
    assert torch.allclose(grp_sum, torch.ones_like(grp_sum), atol=1e-5), \
        'simnorm not sum-1 per group'
    assert float(z_sn.max()) < 0.999 and float(z_sn.min()) > 0.0, \
        'simnorm must be SOFT (strictly interior), not one-hot'
    assert torch.isfinite(z_sn).all()
    # joint-embedding loss: finite, non-negative, zero when prior==post.
    lg = torch.randn(4, 5, K, C)
    assert float(rssm_joint_embed_loss(lg, lg)) < 1e-6, 'JE(self)!=0'
    je = rssm_joint_embed_loss(torch.randn(4, 5, K, C), torch.randn(4, 5, K, C))
    assert torch.isfinite(je) and float(je) > 0.0, 'JE must be finite >0'
    print('[smoke] OK  _CategoricalLatent: categorical=one-hot, '
          'simnorm=soft-simplex; joint_embed_loss finite')


def _check_backbone(wm_type):
    B, obs_dim, action_dim = 3, 6, 2
    for latent_type, je_coef in (('categorical', 0.0), ('simnorm', 1.0)):
        torch.manual_seed(0)
        cfg = _small_cfg(obs_dim, action_dim, wm_type, latent_type, je_coef)
        model = build_model(cfg)
        # transfer contract: the shared head carries the mode in BOTH backbones.
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
        assert 'joint_embed_loss' in losses, 'joint_embed_loss missing'
        for k, v in losses.items():
            if torch.is_tensor(v) and v.is_floating_point():
                assert torch.isfinite(v).all(), f'NON-FINITE {k}'
        je = float(losses['joint_embed_loss'])
        if je_coef > 0.0:
            assert je > 0.0, f'{wm_type}/{latent_type}: JE should be >0'
        else:
            assert je == 0.0, f'{wm_type}/{latent_type}: JE should be 0 (off)'
        losses['wm_total'].backward()
        wt, rc, kl = (float(losses['wm_total'].detach()),
                      float(losses['recon_loss'].detach()),
                      float(losses['kl_loss'].detach()))
        print(f'[smoke] OK  {wm_type}/{latent_type}: wm_total={wt:.4f} '
              f'recon={rc:.4f} kl={kl:.4f} '
              f'joint_embed={je:.4f}  backward OK')
        # P3 imagination (img_step with the SOFT latent) must also be finite.
        if latent_type == 'simnorm':
            diag = _realsim_actor_critic_step(model, batch, cfg)
            for k, v in diag.items():
                if torch.is_tensor(v) and v.is_floating_point():
                    assert torch.isfinite(v).all(), f'P3 NON-FINITE {k}'
            print(f'[smoke] OK  {wm_type}/simnorm P3 imagination finite '
                  f'(actor_loss={float(diag["actor_loss"]):.3f} '
                  f'imag_return={float(diag["imagined_return_mean"]):.3f})')


def main():
    _check_head()
    for wm in ('rssm', 'tssm'):
        _check_backbone(wm)
    print('[smoke] ALL PASS')


if __name__ == '__main__':
    main()
