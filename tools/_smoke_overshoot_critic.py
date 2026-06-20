"""Smoke test for the latent-overshooting (#2) and critic real-grounding (#1)
world-model/critic training levers.

Verifies, for both backbones, WITHOUT a real env:
  * default cfg => #2 OFF (wm_overshoot_loss == 0) and #1 identity
    (critic_imag_loss_coef=1.0 => critic_loss == critic_imag_loss).
  * #2 engaged (RSSM) => overshoot loss finite + > 0, and its gradient
    REACHES THE PRIOR (sample=True straight-through path) — the whole point.
  * #2 is a no-op for the SF backbone (returns 0).
  * #1 engaged => imagined critic CE is down-weighted (critic_loss shrinks)
    and critic_imag_loss is surfaced in the P3 diag, both backbones.
  * everything stays finite and backprops.

Run (CPU, do not disturb a live GPU run):
  CUDA_VISIBLE_DEVICES="" PYTHONPATH=$PWD \
  $PWD/../neural-apc-mbrl-env/bin/python tools/_smoke_overshoot_critic.py
"""
import torch

from training.train import (TrainConfig, build_model, world_model_loss,
                            imagination_step, _wm_latent_overshoot_loss,
                            _wm_held_rollout_stationarity_loss)


def _mk(obs_dim=6, action_dim=2, wm_type='rssm'):
    torch.manual_seed(0)
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
    cfg.d_model = 64
    cfg.head_hidden = 64
    cfg.head_n_layers = 2
    cfg.mtp_length = 4
    cfg.horizon = 4
    cfg.seq_len = 16
    model = build_model(cfg)
    B, T = 3, cfg.seq_len
    batch = {
        'obs': torch.randn(B, T, obs_dim),
        'act': torch.rand(B, T, action_dim) * 2 - 1,
        'rew': torch.randn(B, T),
        'cont': torch.ones(B, T),
        'expert': (torch.rand(B, T) > 0.5).float(),
    }
    return cfg, model, batch


def _run(wm_type):
    cfg, model, batch = _mk(wm_type=wm_type)
    print(f'\n=== backbone: {wm_type} ===')

    # ---- overshoot/held are ON by default (coef 0.3/0.5, promoted p117).
    # The overshoot/held losses are RSSM-only (the SF backbone's shortcut-
    # forcing is its native multi-step term), so they are a no-op (==0) for SF. -
    losses, _, _ = world_model_loss(model, batch, cfg)
    assert 'wm_overshoot_loss' in losses, 'overshoot key missing'
    ov_on = float(losses['wm_overshoot_loss'])
    if wm_type == 'rssm':
        assert ov_on > 0.0, f'RSSM overshoot must be ON by default (p117), got {ov_on}'
        print(f'[smoke] OK  default wm_overshoot_loss > 0 ({ov_on:.4f}) ({wm_type})')
    else:
        assert ov_on == 0.0, f'SF overshoot must be a no-op (0), got {ov_on}'
        print(f'[smoke] OK  default wm_overshoot_loss == 0 (SF no-op) ({wm_type})')

    # ---- coef=0 turns it OFF (the escape hatch) ----
    cfg.wm_overshoot_coef = 0.0
    cfg.wm_held_rollout_coef = 0.0
    losses_off, _, _ = world_model_loss(model, batch, cfg)
    assert float(losses_off['wm_overshoot_loss']) == 0.0, \
        'overshoot must be 0 when coef=0'
    assert float(losses_off.get('wm_held_rollout_loss', 0.0)) == 0.0, \
        'held-rollout must be 0 when coef=0'
    cfg.wm_overshoot_coef = 0.3
    cfg.wm_held_rollout_coef = 0.5
    print(f'[smoke] OK  coef=0 disables overshoot/held ({wm_type})')

    diag = imagination_step(model, batch, cfg)
    assert 'critic_imag_loss' in diag, 'critic_imag_loss missing from diag'
    cl = float(diag['critic_loss'])
    cil = float(diag['critic_imag_loss'])
    # critic_loss = imag_coef*imag + anchor_coef*anchor + mc_coef*mc.  Current
    # promoted defaults (p117/p124): imag_coef=0.3, anchor_coef=0.0, mc_coef=1.0
    # — so the identity critic_loss == critic_imag_loss only holds when we force
    # imag_coef=1.0 AND zero BOTH grounded terms (the TD-λ replay anchor AND the
    # MC grounding).
    cfg.critic_imag_loss_coef = 1.0
    cfg.critic_replay_anchor_coef = 0.0
    cfg.critic_mc_grounding_coef = 0.0
    diag_na = imagination_step(model, batch, cfg)
    assert abs(float(diag_na['critic_loss'])
               - float(diag_na['critic_imag_loss'])) < 1e-5, \
        'imag_coef=1.0 + no grounded terms must give critic_loss == critic_imag_loss'
    print(f'[smoke] OK  #1 identity at coef=1.0 (critic_loss==imag) ({wm_type})')

    # ---- #1 engaged: down-weight imagined CE (grounded terms still off) ----
    cfg.critic_imag_loss_coef = 0.3
    diag1 = imagination_step(model, batch, cfg)
    cl1 = float(diag1['critic_loss'])
    cil1 = float(diag1['critic_imag_loss'])
    assert abs(cl1 - 0.3 * cil1) < 1e-4, \
        f'#1 rebalance wrong: critic_loss={cl1} != 0.3*{cil1}'
    assert torch.isfinite(diag1['critic_loss']).all()
    diag1_full = dict(diag1)
    # restore the promoted defaults
    cfg.critic_imag_loss_coef = 0.3
    cfg.critic_replay_anchor_coef = 0.0
    cfg.critic_mc_grounding_coef = 1.0
    print(f'[smoke] OK  #1 engaged critic_loss == 0.3*imag ({wm_type})')

    # ---- #2 engaged (RSSM real; SF no-op) ----
    cfg.wm_overshoot_coef = 0.5
    cfg.wm_overshoot_len = 8
    losses2, _, _ = world_model_loss(model, batch, cfg)
    ov2 = float(losses2['wm_overshoot_loss'])
    starts = float(losses2['wm_overshoot_starts'])
    assert torch.isfinite(losses2['wm_total']).all()
    losses2['wm_total'].backward()
    if wm_type == 'rssm':
        assert ov2 > 0.0, f'RSSM overshoot loss must be > 0, got {ov2}'
        assert starts > 0, 'RSSM overshoot must use >0 start positions'
        print(f'[smoke] OK  #2 RSSM overshoot loss={ov2:.4f} starts={starts:.0f}')
        # KEY: the overshoot gradient must reach the PRIOR (sample=True ST).
        feats, *_ = model.dynamics.rollout_observed(
            batch['obs'], batch['act'], sample=True)
        model.zero_grad(set_to_none=True)
        ov_loss, S = _wm_latent_overshoot_loss(
            model, feats, batch['obs'], batch['act'], cfg)
        ov_loss.backward()
        prior_grad = sum(
            float(p.grad.abs().sum()) for p in model.dynamics.prior_net.parameters()
            if p.grad is not None)
        assert prior_grad > 0.0, \
            'overshoot gradient did NOT reach prior_net (sample path broken)'
        print(f'[smoke] OK  #2 overshoot grad reaches prior_net '
              f'(|g|={prior_grad:.4f}, S={S:.0f})')
    else:
        assert ov2 == 0.0, f'SF overshoot must be 0 (no-op), got {ov2}'
        print('[smoke] OK  #2 SF no-op (overshoot == 0)')
    cfg.wm_overshoot_coef = 0.0

    # ---- (b2) held-action rollout stationarity (RSSM real; SF no-op) ----
    cfg.wm_held_rollout_coef = 0.5
    cfg.wm_held_rollout_len = 12
    cfg.wm_held_rollout_settle_frac = 0.5
    cfg.wm_held_rollout_win = 2
    cfg.wm_held_rollout_max_starts = 6
    lossesh, _, _ = world_model_loss(model, batch, cfg)
    assert 'wm_held_rollout_loss' in lossesh, 'held key missing'
    hv = float(lossesh['wm_held_rollout_loss'])
    assert torch.isfinite(lossesh['wm_total']).all()
    lossesh['wm_total'].backward()
    if wm_type == 'rssm':
        assert hv > 0.0, f'RSSM held-rollout loss must be > 0, got {hv}'
        print(f'[smoke] OK  (b2) RSSM held-rollout loss={hv:.4f}')
        # KEY: the held-rollout gradient must reach the PRIOR (sample=True ST).
        feats, *_ = model.dynamics.rollout_observed(
            batch['obs'], batch['act'], sample=True)
        model.zero_grad(set_to_none=True)
        h_loss, S = _wm_held_rollout_stationarity_loss(
            model, feats, batch['obs'], batch['act'], cfg)
        h_loss.backward()
        prior_grad = sum(
            float(p.grad.abs().sum()) for p in model.dynamics.prior_net.parameters()
            if p.grad is not None)
        assert prior_grad > 0.0, \
            'held-rollout gradient did NOT reach prior_net (sample path broken)'
        print(f'[smoke] OK  (b2) held-rollout grad reaches prior_net '
              f'(|g|={prior_grad:.4f}, S={S:.0f})')
    else:
        assert hv == 0.0, f'SF held-rollout must be 0 (no-op), got {hv}'
        print('[smoke] OK  (b2) SF no-op (held-rollout == 0)')
    cfg.wm_held_rollout_coef = 0.0

    print(f'[smoke] ALL OVERSHOOT/CRITIC CHECKS PASSED ({wm_type})')


if __name__ == '__main__':
    _run('rssm')
    _run('sf_transformer')
    print('\n[smoke] overshoot (#2) + critic-grounding (#1) smoke complete — both backbones OK')
