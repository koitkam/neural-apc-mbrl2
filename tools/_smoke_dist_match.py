"""Targeted smoke for Fix A: C(2) disturbance-match (c_dist supervision).

Builds a cont-latent-ON model (both backbones), runs the WM loss with a
non-trivial 'dist' target, and asserts:
  * dist_match_loss is present, finite, and > 0 (the term engaged),
  * wm_total.backward() runs,
  * the gradient REACHES the cont posterior net (the loss shapes the latent),
  * cont-OFF or dist_match_coef=0 => dist_match_loss == 0 (clean no-op).
"""
import torch
from training.train import TrainConfig, build_model, world_model_loss


def _build(wm_type, cont_on, dm_coef):
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
    cfg.disturbance_head_dim = 0          # isolate dist_match (no head term)
    if cont_on:
        cfg.cont_latent_enabled = True
        cfg.cont_gain_dim = 2
        cfg.cont_dist_dim = 1             # n_cv = 1
        cfg.dist_match_coef = dm_coef
    return cfg, build_model(cfg)


def _batch(cfg, with_dist=True, n_dist=1, const_dist=False):
    B, T = 3, cfg.seq_len
    b = {
        'obs': torch.randn(B, T, cfg.obs_dim),
        'act': torch.rand(B, T, cfg.action_dim) * 2 - 1,
        'rew': torch.randn(B, T),
        'cont': torch.ones(B, T),
        'expert': (torch.rand(B, T) > 0.5).float(),
    }
    if with_dist:
        b['dist'] = (torch.zeros(B, T, n_dist) if const_dist
                     else torch.randn(B, T, n_dist))
    return b


def run(wm_type):
    print(f'\n===== {wm_type} =====')
    # (1) cont ON + dist_match ON + varying dist -> term engages
    cfg, model = _build(wm_type, cont_on=True, dm_coef=0.3)
    losses, _, _ = world_model_loss(model, _batch(cfg), cfg)
    dml = float(losses['dist_match_loss'])
    assert 'dist_match_loss' in losses, 'dist_match_loss missing from losses'
    assert torch.isfinite(losses['dist_match_loss']).all(), 'dist_match non-finite'
    assert dml > 0.0, f'dist_match_loss should be >0 when engaged, got {dml}'
    print(f'[smoke] OK  dist_match engaged: dist_match_loss={dml:.4f} '
          f'wm_total={float(losses["wm_total"]):.4f}')

    # (2) gradient reaches the cont posterior net (shapes the latent)
    model.zero_grad(set_to_none=True)
    losses2, _, _ = world_model_loss(model, _batch(cfg), cfg)
    losses2['wm_total'].backward()
    rssm = model.dynamics
    post_net = getattr(rssm, 'cont_post_net', None)
    assert post_net is not None, 'cont_post_net not found'
    gsum = sum(p.grad.abs().sum().item() for p in post_net.parameters()
               if p.grad is not None)
    assert gsum > 0.0, 'no gradient reached cont_post_net'
    print(f'[smoke] OK  wm_total.backward(); grad reaches cont_post_net '
          f'(|g|={gsum:.3f})')

    # (3) const (zero-variance) dist -> dvar gate -> dist_match == 0
    cfg3, model3 = _build(wm_type, cont_on=True, dm_coef=0.3)
    l3, _, _ = world_model_loss(model3, _batch(cfg3, const_dist=True), cfg3)
    assert float(l3['dist_match_loss']) == 0.0, \
        f'const dist should gate dist_match to 0, got {float(l3["dist_match_loss"])}'
    print('[smoke] OK  zero-variance dist -> dist_match gated to 0')

    # (4) coef 0 -> no-op
    cfg4, model4 = _build(wm_type, cont_on=True, dm_coef=0.0)
    l4, _, _ = world_model_loss(model4, _batch(cfg4), cfg4)
    assert float(l4['dist_match_loss']) == 0.0, 'coef=0 should give dist_match 0'
    print('[smoke] OK  dist_match_coef=0 -> no-op')

    # (5) cont OFF -> dist_match 0 (byte-clean)
    cfg5, model5 = _build(wm_type, cont_on=False, dm_coef=0.0)
    l5, _, _ = world_model_loss(model5, _batch(cfg5), cfg5)
    assert float(l5.get('dist_match_loss', 0.0)) == 0.0, 'cont-off dist_match !=0'
    print('[smoke] OK  cont-OFF -> dist_match 0 (no-op)')


if __name__ == '__main__':
    run('rssm')
    run('tssm')
    print('\n[smoke] ALL DIST-MATCH (Fix A) CHECKS PASSED both backbones')
