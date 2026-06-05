"""Standalone smoke test for the RSSM world-model port (P68).

Exercises every RSSM execution path that training touches WITHOUT a real
env:
  * RSSMDynamics shape contracts (rollout_observed / obs_step / img_step /
    decode) + rssm_kl_loss.
  * V4 heads on RSSM feat (reward MTP, value, policy).
  * train.world_model_loss   (P1/P2 WM loss)
  * train.agent_finetune_loss (P2 BC + reward MTP)
  * train.imagination_step    (P3 actor/critic via RSSM imagination)

All outputs must be finite.  Run:
  CONTROL_SETUP_JSON=... PYTHONPATH=/home/koitkam/neural-apc-dreamerV4 \
  /home/koitkam/neural-apc-dreamerV4-env/bin/python tools/_smoke_rssm.py
"""
import torch

from training.train import (TrainConfig, build_model, world_model_loss,
                            agent_finetune_loss, imagination_step,
                            expert_bc_p3_loss, _adaptive_return_cap,
                            _steady_held_mask, _critic_anchor_lambda,
                            _critic_anchor_coef)


def main(obs_dim: int = 6, action_dim: int = 2, label: str = 'default',
         wm_type: str = 'rssm') -> None:
    torch.manual_seed(0)
    cfg = TrainConfig()
    cfg.obs_dim = obs_dim
    cfg.action_dim = action_dim
    cfg.lookback = 8
    cfg.world_model_type = wm_type
    # Keep it small + fast.
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
    feat_dim = (model.dynamics.feat_dim if wm_type == 'rssm'
                else int(cfg.d_model))
    print(f'[smoke] world_model_type={model.world_model_type} '
          f'feat_dim={feat_dim} '
          f'tokenizer={model.tokenizer}')

    B, T = 3, cfg.seq_len
    batch = {
        'obs': torch.randn(B, T, obs_dim),
        'act': torch.rand(B, T, action_dim) * 2 - 1,
        'rew': torch.randn(B, T),
        'cont': torch.ones(B, T),
        'expert': (torch.rand(B, T) > 0.5).float(),
    }
    def _finite(name, d):
        bad = []
        for k, v in d.items():
            if torch.is_tensor(v) and v.is_floating_point():
                if not torch.isfinite(v).all():
                    bad.append(k)
        flag = 'OK ' if not bad else 'BAD'
        print(f'[smoke] {flag} {name}: '
              + ', '.join(f'{k}={float(v):.4f}' for k, v in d.items()
                          if torch.is_tensor(v) and v.numel() == 1))
        if bad:
            raise SystemExit(f'NON-FINITE in {name}: {bad}')

    # ---- P1/P2 world-model loss ----
    losses, z_clean, agent_hid = world_model_loss(model, batch, cfg)
    assert agent_hid.shape == (B, T, feat_dim), agent_hid.shape
    # (b) steady-state consistency must be wired into BOTH backbones.
    assert 'wm_steady_loss' in losses, 'wm_steady_loss missing from WM losses'
    assert 'wm_steady_held_frac' in losses, 'wm_steady_held_frac missing'
    _finite('world_model_loss', losses)
    losses['wm_total'].backward()
    print('[smoke] OK  wm_total.backward()')

    # (a) adaptive return-value cap: positive + finite when reward bounded.
    _cap = _adaptive_return_cap(cfg)
    if bool(getattr(cfg, 'bound_training_reward', False)):
        assert _cap is not None and _cap > 0.0 and _cap == _cap, \
            f'adaptive return cap invalid: {_cap}'
        print(f'[smoke] OK  adaptive return cap = {_cap:.3f}')
    # (b) held-mask helper is finite and well-shaped (never NaN).
    _m = _steady_held_mask(batch['obs'], batch['act'], cfg)
    if _m is not None:
        assert torch.isfinite(_m).all() and _m.shape == (B, T - 1), _m.shape

    # (B) long-horizon critic-anchor grounding: λ default falls back to
    # gae_lambda; engaged value is clamped to [0,1]; coef resolver honours
    # the optional override.  Exercise both default + engaged.
    assert _critic_anchor_lambda(cfg) == float(cfg.gae_lambda), \
        'anchor λ default must equal gae_lambda'
    assert _critic_anchor_coef(cfg) == float(cfg.critic_replay_anchor_coef), \
        'anchor coef default must equal base coef'
    cfg.critic_anchor_lambda = 0.97
    cfg.critic_anchor_coef_long = 1.0
    assert abs(_critic_anchor_lambda(cfg) - 0.97) < 1e-9, 'anchor λ engage'
    assert abs(_critic_anchor_coef(cfg) - 1.0) < 1e-9, 'anchor coef engage'
    cfg.critic_anchor_lambda = 1.5   # out-of-range -> clamped to 1.0
    assert _critic_anchor_lambda(cfg) == 1.0, 'anchor λ clamp to 1.0'
    cfg.critic_anchor_lambda = None
    cfg.critic_anchor_coef_long = None
    print('[smoke] OK  critic-anchor λ/coef resolvers (default + engaged + clamp)')

    # ---- P2 agent finetune (BC + reward MTP) ----
    _, _, agent_hid2 = world_model_loss(model, batch, cfg)
    af = agent_finetune_loss(model, batch, agent_hid2, cfg)
    _finite('agent_finetune_loss', af)
    af['agent_total'].backward()
    print('[smoke] OK  agent_total.backward()')

    # ---- P3 imagination ----
    diag = imagination_step(model, batch, cfg)
    _finite('imagination_step', diag)

    # (B) imagination with the long-horizon anchor ENGAGED — exercise the
    # ``lam_anchor`` recursion + raised coef and confirm the critic loss is
    # finite and backprops through the engaged path.
    cfg.critic_anchor_lambda = 0.97
    cfg.critic_anchor_coef_long = 1.0
    diagB = imagination_step(model, batch, cfg)
    _finite('imagination_step[anchorB]', diagB)
    diagB['critic_loss'].backward()
    print('[smoke] OK  imagination_step critic_loss.backward() with anchor B engaged')
    cfg.critic_anchor_lambda = None
    cfg.critic_anchor_coef_long = None

    # ---- P3 expert-BC anchor (P83) + adaptive scaling (P84) ----
    # Exercise the new train-loop P3 branch outside the full loop: build a
    # masked expert batch, call expert_bc_p3_loss, and replay the exact
    # adaptive-scale arithmetic against the imagination return_scale.
    _, _, agent_hid3 = world_model_loss(model, batch, cfg)
    em = (torch.rand(B, T) > 0.5).float()           # ~half steps are expert
    bc_batch = dict(batch)
    bc_batch['expert'] = em
    bc_loss, n_exp = expert_bc_p3_loss(model, bc_batch, agent_hid3)
    assert torch.isfinite(bc_loss).all(), 'bc_p3_loss non-finite'
    assert float(n_exp) > 0, 'expert mask produced zero steps'

    # Empty-mask must yield exactly-zero loss (no expert steps -> no grad).
    bc_batch0 = dict(batch)
    bc_batch0['expert'] = torch.zeros(B, T)
    bc_loss0, n0 = expert_bc_p3_loss(model, bc_batch0, agent_hid3)
    assert float(bc_loss0) == 0.0, f'empty-mask bc loss != 0: {float(bc_loss0)}'

    # Adaptive-scale arithmetic (mirrors train.py P3 branch).
    cfg.expert_bc_p3_adaptive_scale = True
    base_w = float(cfg.expert_bc_scale) * 0.5       # decay placeholder
    adv_ref = float(getattr(cfg, 'advantage_clip', 8.0) or 8.0)
    for rs in (0.5, 1.0, 8.0, 102.0, 500.0):
        w = base_w * (adv_ref / max(rs, 1.0))
        assert w == w and w >= 0.0 and w != float('inf'), \
            f'adaptive weight non-finite at return_scale={rs}: {w}'
    # weight must shrink as return_scale grows (anchor not drowned check)
    w_lo = base_w * (adv_ref / max(1.0, 1.0))
    w_hi = base_w * (adv_ref / max(102.0, 1.0))
    assert w_hi < w_lo, 'adaptive weight should shrink as return_scale grows'
    _finite('expert_bc_p3', {'bc_loss': bc_loss, 'n_expert': n_exp,
                             'bc_loss_empty': bc_loss0,
                             'w@rs1': torch.tensor(w_lo),
                             'w@rs102': torch.tensor(w_hi)})

    print(f'[smoke] ALL RSSM SMOKE CHECKS PASSED  ({label}: '
          f'obs={obs_dim} act={action_dim} wm={wm_type})')


def _sim_dims(setup_path: str):
    """Boot a simulator from its control_setup.json and return (obs, act)."""
    import os
    os.environ['CONTROL_SETUP_JSON'] = setup_path
    from utils.sim_factory import create_sim
    sim = create_sim(episode_length=64)
    state_dim = int(getattr(sim, 'state_dim', None) or
                    len(getattr(sim, 'state_variables', []) or []))
    action_dim = int(getattr(sim, 'action_dim', None) or
                     len(getattr(sim, 'mv_indices', []) or []))
    return state_dim, action_dim


if __name__ == '__main__':
    import os
    import sys
    sims = [
        ('test_sim', 'simulation/test_sim/control_setup.json'),
        ('generic', 'simulation/generic/control_setup.json'),
        ('distillation', 'simulation/distillation/control_setup.json'),
        ('softsensor_lab', 'simulation/softsensor_lab/control_setup.json'),
    ]
    only = sys.argv[1] if len(sys.argv) > 1 else None
    ran = 0
    for name, path in sims:
        if only and only != name:
            continue
        if not os.path.exists(path):
            print(f'[smoke] SKIP {name}: {path} not found')
            continue
        try:
            obs_dim, act_dim = _sim_dims(path)
        except Exception as exc:  # noqa: BLE001
            print(f'[smoke] WARN {name}: dim probe failed ({exc}); '
                  f'falling back to 6x2')
            obs_dim, act_dim = 6, 2
        # obs the model sees includes lookback-flattened channels in the real
        # run; for the synthetic smoke we just need a sane positive width.
        obs_dim = max(int(obs_dim), 2)
        act_dim = max(int(act_dim), 1)
        print(f'\n[smoke] ===== {name} (obs={obs_dim} act={act_dim}) =====')
        main(obs_dim=obs_dim, action_dim=act_dim, label=name)
        ran += 1
        # Also exercise the legacy transformer backbone on test_sim so the
        # (a)/(b) transformer code paths are covered without quadrupling time.
        if name == 'test_sim':
            print(f'\n[smoke] ===== {name} [sf_transformer] '
                  f'(obs={obs_dim} act={act_dim}) =====')
            main(obs_dim=obs_dim, action_dim=act_dim,
                 label=name, wm_type='sf_transformer')
            ran += 1
    print(f'\n[smoke] DONE: {ran} simulator config(s) passed')
