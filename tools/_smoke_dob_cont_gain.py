"""Smoke for the p142 config: cont GAIN block + DOB, cont DISTURBANCE block OFF.

The disturbance was reverted from the (failed, 5-run) learned cont-disturbance
channel back to the classical neural-Kalman DOB.  The cont latent keeps ONLY the
GAIN block (the C(1) gain-match de-confounder); the DOB owns the unmeasured load
+ Scope-2 (d_t in feat).  Verifies, BOTH backbones, WITHOUT a real env:

  * model builds with cont_gain_dim>0 AND cont_dist_dim=0 AND dob_enabled=True;
  * cont_dim == cont_gain_dim (no disturbance block); the DOB d-tail IS in feat
    (feat_dim == deter+stoch+cont_gain+dv + n_cv);
  * world_model_loss runs + backward; dist_match_loss == 0 (no cont-dist channel
    to supervise) while the DOB d_t is live (dob_d_absmean > 0);
  * the gain↔disturbance partition is clean: the cont-gain params live in the
    ``g`` group (train/freeze with the plant model), the DOB A/K in the ``dob``
    group — so the staged curriculum freezes g (incl. cont-gain) in Stage 2 while
    the DOB keeps training, exactly the identifiability separation we want.
"""
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from training.train import TrainConfig, build_model, world_model_loss  # noqa: E402


def _build(wm_type, cv_idx=(0,)):
    torch.manual_seed(0)
    cfg = TrainConfig()
    cfg.obs_dim = 6
    cfg.action_dim = 2
    cfg.lookback = 8
    cfg.world_model_type = wm_type
    cfg.compile_mode = 'none'
    cfg.rssm_deter_dim = 64
    cfg.rssm_n_categoricals = 8
    cfg.rssm_n_classes = 8
    cfg.rssm_embed_dim = 32
    cfg.rssm_hidden_dim = 32
    cfg.tssm_d_model = 64
    cfg.d_model = 64
    cfg.head_hidden = 64
    cfg.head_n_layers = 2
    cfg.mtp_length = 4
    cfg.horizon = 4
    cfg.seq_len = 16
    # p142 config: GAIN block on, DISTURBANCE block OFF, DOB owns the load.
    cfg.cont_gain_dim = 2
    cfg.cont_dist_dim = 0
    cfg.dist_match_coef = 0.0
    cfg.disturbance_head_dim = 0      # retired (DOB is the estimator)
    cfg.dob_enabled = True
    cfg.dob_reg_coef = 0.01
    cfg.cv_obs_indices = tuple(cv_idx)
    model = build_model(cfg)
    B, T = 3, cfg.seq_len
    batch = {
        'obs': torch.randn(B, T, cfg.obs_dim),
        'act': torch.rand(B, T, cfg.action_dim) * 2 - 1,
        'rew': torch.randn(B, T),
        'cont': torch.ones(B, T),
        'expert': (torch.rand(B, T) > 0.5).float(),
        'dist': torch.randn(B, T, 1),
    }
    return cfg, model, batch


def run(wm_type):
    print(f'\n===== {wm_type} =====')
    cfg, model, batch = _build(wm_type)
    dyn = model.dynamics

    # (1) layout: cont = gain-only; DOB d-tail present.
    assert dyn.cont_gain_dim == 2 and dyn.cont_dist_dim == 0, \
        f'expected gain-only cont, got gain={dyn.cont_gain_dim} dist={dyn.cont_dist_dim}'
    assert dyn.cont_dim == dyn.cont_gain_dim, 'cont_dim must equal cont_gain_dim'
    assert getattr(dyn, '_cont_post_uses_innov', False) is False, \
        'innovation 2-pass must be OFF when cont_dist_dim==0'
    core = dyn.deter_dim + dyn.stoch_flat_dim + dyn.cont_dim
    dv_feed = getattr(dyn, '_dv_feed_dim', 0)
    expect_feat = core + dv_feed + dyn.n_cv      # + DOB d-tail (Scope 2)
    assert dyn.feat_dim == expect_feat, \
        f'feat_dim {dyn.feat_dim} != core+dv+n_cv {expect_feat}'
    print(f'[smoke] OK  layout: cont gain-only (dim {dyn.cont_gain_dim}), '
          f'DOB d-tail in feat (feat_dim={dyn.feat_dim}, n_cv={dyn.n_cv})')

    # (2) WM loss runs; dist_match == 0; DOB d_t live; backward reaches cont+dob.
    model.zero_grad(set_to_none=True)
    losses, _, _ = world_model_loss(model, batch, cfg)
    assert float(losses['dist_match_loss']) == 0.0, \
        f'dist_match must be 0 with no cont-dist channel, got {float(losses["dist_match_loss"])}'
    assert float(losses['dob_d_absmean']) > 0.0, 'DOB d_t should be live (>0)'
    assert torch.isfinite(losses['wm_total']).all(), 'wm_total non-finite'
    losses['wm_total'].backward()
    cont_net = getattr(dyn, 'cont_post_net', None) or getattr(dyn, 'cont_prior_net', None)
    gsum = sum(p.grad.abs().sum().item() for p in cont_net.parameters()
               if p.grad is not None)
    assert gsum > 0.0, 'no gradient reached the cont-gain net'
    print(f'[smoke] OK  WM loss: dist_match=0, dob_d_absmean='
          f'{float(losses["dob_d_absmean"]):.4f}, grad->cont-gain |g|={gsum:.3f}')

    # (3) staged-curriculum partition: cont-gain in `g`, DOB A/K in `dob`.
    fz = model.set_world_model_trainable(g=True, dob=False, reward=True)
    cont_req = [p.requires_grad for p in cont_net.parameters()]
    dob_req = [getattr(dyn, n).requires_grad for n in ('dob_log_decay', 'dob_log_gain')
               if getattr(dyn, n, None) is not None]
    assert all(cont_req), 'cont-gain must be TRAINABLE when g=True'
    assert not any(dob_req), 'DOB A/K must be FROZEN when dob=False (Stage-1)'
    fz2 = model.set_world_model_trainable(g=False, dob=True, reward=True)
    cont_req2 = [p.requires_grad for p in cont_net.parameters()]
    dob_req2 = [getattr(dyn, n).requires_grad for n in ('dob_log_decay', 'dob_log_gain')
                if getattr(dyn, n, None) is not None]
    assert not any(cont_req2), 'cont-gain must FREEZE with g (Stage-2 g frozen)'
    assert all(dob_req2), 'DOB A/K must TRAIN when dob=True (Stage-2)'
    print(f'[smoke] OK  partition: Stage-1 g={fz["g"]}(cont-gain trainable)/'
          f'dob={fz["dob"]}(frozen); Stage-2 g frozen / dob trains — clean '
          f'gain↔disturbance separation')


if __name__ == '__main__':
    run('rssm')
    run('tssm')
    print('\n[smoke] ALL DOB+CONT-GAIN (p142) CHECKS PASSED both backbones')
