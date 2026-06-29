"""Targeted smoke for the p140 RCA fixes (both backbones):

  R1 — deterministic cont-DISTURBANCE roll in imagination.  ``img_step(sample=
       True)`` must return the DISTURBANCE block of ``c`` equal to its prior
       MEAN (no per-rollout sampling noise = clean feedforward) while the GAIN
       block stays STOCHASTIC.  With ``cont_dist_deterministic_roll=False`` the
       disturbance block varies again (the flag genuinely gates it).

  R3 — the static DV->obs feedthrough skip is REMOVED by default
       (``dv_static_skip=False`` => ``dynamics.dv_skip is None``) and restored
       only as an ablation lever (``dv_static_skip=True`` => module present).
"""
import torch
from training.train import TrainConfig, build_model


def _build(wm_type, *, det_roll=True, dv=False, dv_static_skip=False):
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
    cfg.dv_static_skip = dv_static_skip
    if dv:
        cfg.dv_as_input = True
        cfg.dv_feedforward = True
        cfg.dv_dim = 1
        cfg.dv_indices = (5,)
    return cfg, build_model(cfg)


def _img2(model, *, det_roll, dv_dim=0, B=3):
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
    cfg, model = _build(wm_type, det_roll=True)
    dyn = model.dynamics
    g = dyn.cont_gain_dim

    # R1: two sampled img_steps from the SAME state+action.  Disturbance block
    # must be IDENTICAL (== prior mean, deterministic) across both; gain block
    # STOCHASTIC (the only varying part).
    s1, s2 = _img2(model, det_roll=True)
    d1, d2 = s1.c[..., g:], s2.c[..., g:]
    assert torch.allclose(s1.c[..., g:], s1.c_mean[..., g:]), \
        'R1: disturbance block of sampled c != prior mean'
    assert torch.allclose(d1, d2), \
        'R1: disturbance block varied across samples (not deterministic)'
    gain1, gain2 = s1.c[..., :g], s2.c[..., :g]
    assert (gain1 - gain2).abs().max().item() > 1e-4, \
        'R1: gain block did NOT vary (should stay stochastic)'
    print('[smoke] OK  R1 det-roll: disturbance block = prior mean (clean '
          'feedforward); gain block still stochastic')

    # R1 (flag off): disturbance block must VARY again.
    _, model_off = _build(wm_type, det_roll=False)
    o1, o2 = _img2(model_off, det_roll=False)
    assert (o1.c[..., g:] - o2.c[..., g:]).abs().max().item() > 1e-4, \
        'R1: det_roll=False should leave the disturbance block stochastic'
    print('[smoke] OK  R1 det_roll=False -> disturbance block stochastic again '
          '(flag gates it)')

    # R3: dv_skip removed by default; present only when re-enabled.
    _, m_nodv = _build(wm_type, dv=True, dv_static_skip=False)
    assert m_nodv.dynamics.dv_skip is None, \
        'R3: dv_skip should be None by default (dv_static_skip=False)'
    _, m_skip = _build(wm_type, dv=True, dv_static_skip=True)
    assert m_skip.dynamics.dv_skip is not None, \
        'R3: dv_static_skip=True should restore the skip module'
    print('[smoke] OK  R3 dv_skip removed by default; restorable via '
          'dv_static_skip=True (ablation lever)')


if __name__ == '__main__':
    run('rssm')
    run('tssm')
    print('\n[smoke] ALL CONT-DIST-ROLL (p140 R1+R3) CHECKS PASSED both backbones')
