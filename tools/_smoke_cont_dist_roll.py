"""Targeted smoke for the cont-latent deterministic-roll fixes (both backbones):

  R1 — deterministic cont-DISTURBANCE roll in imagination (p140 RCA).
  R2 — deterministic cont-GAIN roll in imagination (p20 observer-bias RCA):
       ``img_step(sample=True)`` must return the GAIN block of ``c`` equal to
       its prior MEAN so the strong sample=True gain supervisor trains the
       actor's sample=False (mean) belief.  With
       ``cont_gain_deterministic_roll=False`` the gain block varies again (the
       flag genuinely gates it).  Both blocks default to deterministic.
"""
import torch
from training.train import TrainConfig, build_model


def _build(wm_type, *, det_roll=True, gain_det_roll=True, dv=False):
    torch.manual_seed(0)
    cfg = TrainConfig()
    cfg.obs_dim = 6
    cfg.action_dim = 2
    cfg.lookback = 8
    cfg.world_model_type = wm_type
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
    cfg.disturbance_head_dim = 0
    cfg.cont_latent_enabled = True
    cfg.cont_gain_dim = 2
    cfg.cont_dist_dim = 1                  # n_cv = 1
    cfg.cv_obs_indices = (0,)
    cfg.cont_dist_deterministic_roll = det_roll
    cfg.cont_gain_deterministic_roll = gain_det_roll
    if dv:
        cfg.dv_as_input = True
        cfg.dv_feedforward = True
        cfg.dv_dim = 1
        cfg.dv_indices = (5,)
    return cfg, build_model(cfg)


def _img2(model, *, dv_dim=0, B=3):
    """Two sampled img_steps from the SAME state + action (so ``h`` and the
    prior MEAN are identical) — the only difference is the internal sampling
    noise, which isolates the deterministic-roll behaviour."""
    dyn = model.dynamics
    st = dyn.initial_state(B, torch.device('cpu'))
    a = torch.rand(B, model.cfg.action_dim) * 2 - 1
    dv = torch.randn(B, dv_dim) if dv_dim > 0 else None
    s1 = dyn.img_step(st, a, dv=dv, sample=True)
    s2 = dyn.img_step(st, a, dv=dv, sample=True)
    return s1, s2


def run(wm_type):
    print(f'\n===== {wm_type} =====')
    cfg, model = _build(wm_type, det_roll=True, gain_det_roll=True)
    dyn = model.dynamics
    g = dyn.cont_gain_dim

    # R1+R2 (both flags default ON): two sampled img_steps from the SAME
    # state+action.  BOTH the disturbance AND the gain block must be IDENTICAL
    # (== prior mean, deterministic) across the two samples.
    s1, s2 = _img2(model)
    assert torch.allclose(s1.c[..., g:], s1.c_mean[..., g:]), \
        'R1: disturbance block of sampled c != prior mean'
    assert torch.allclose(s1.c[..., g:], s2.c[..., g:]), \
        'R1: disturbance block varied across samples (not deterministic)'
    assert torch.allclose(s1.c[..., :g], s1.c_mean[..., :g]), \
        'R2: gain block of sampled c != prior mean'
    assert torch.allclose(s1.c[..., :g], s2.c[..., :g]), \
        'R2: gain block varied across samples (should be deterministic)'
    print('[smoke] OK  R1+R2 det-roll: BOTH gain and disturbance blocks = '
          'prior mean (deterministic) under sample=True')

    # R2 (gain flag off): the gain block must VARY again while the disturbance
    # block stays deterministic (flags gate independently).
    _, model_goff = _build(wm_type, det_roll=True, gain_det_roll=False)
    g1, g2 = _img2(model_goff)
    assert (g1.c[..., :g] - g2.c[..., :g]).abs().max().item() > 1e-4, \
        'R2: gain_det_roll=False should leave the gain block stochastic'
    assert torch.allclose(g1.c[..., g:], g2.c[..., g:]), \
        'R2: disturbance block should stay deterministic when only gain flag off'
    print('[smoke] OK  R2 gain_det_roll=False -> gain block stochastic again '
          '(flag gates it independently)')

    # R1 (dist flag off): disturbance block must VARY again.
    _, model_off = _build(wm_type, det_roll=False, gain_det_roll=True)
    o1, o2 = _img2(model_off)
    assert (o1.c[..., g:] - o2.c[..., g:]).abs().max().item() > 1e-4, \
        'R1: det_roll=False should leave the disturbance block stochastic'
    print('[smoke] OK  R1 det_roll=False -> disturbance block stochastic again '
          '(flag gates it)')


if __name__ == '__main__':
    run('rssm')
    run('tssm')
    print('\n[smoke] ALL CONT-ROLL (p140 R1 + p20 R2) CHECKS PASSED both backbones')
