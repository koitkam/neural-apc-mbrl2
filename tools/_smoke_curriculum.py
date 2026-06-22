"""Smoke test for the staged clean->disturbance curriculum (2026-06-12).

Verifies, WITHOUT a real env, for BOTH backbones (rssm + tssm), the three
curriculum stages' freeze partition + DOB suppression + GRADIENT ISOLATION —
the properties that make the staged Kalman/DOB identification correct:

  Stage 1 (clean WM): g trainable, DOB frozen + SUPPRESSED (d_t==0) ->
    recon backward trains g (the plant) and gives NO gradient to the DOB
    (there is no d_t path) -> g must explain all CV movement (unbiased gain).
  Stage 2 (DOB id):   g FROZEN, DOB trainable + ACTIVE ->
    recon backward trains the DOB observer (A,K) and gives NO gradient to g
    (frozen) -> the observer is identified on the fixed plant (identifiable).
  Stage 3 (actor):    g + DOB both FROZEN, reward trainable ->
    recon backward gives NO gradient to g or DOB (the WM is static); the
    actor/critic train via imagination (covered by the existing rssm smoke).

Also checks: set_world_model_trainable partitions requires_grad exactly;
set_dob_active toggles d_t between zero (suppressed) and non-zero (active);
feat width stays core+n_cv across stages (no head-dim hiccup).

Run (CPU):
  CUDA_VISIBLE_DEVICES="" PYTHONPATH=$PWD DREAMER_COMPILE=0 \
  $PWD/../neural-apc-mbrl-env/bin/python tools/_smoke_curriculum.py
"""
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from training.train import TrainConfig, build_model, world_model_loss  # noqa: E402


def _mk(wm_type='rssm', cv_idx=(2,)):
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
    # curriculum REQUIRES the DOB on the whole run (constant feat width).
    cfg.disturbance_head_dim = 0           # DOB retires the P87 head
    cfg.cv_obs_indices = tuple(cv_idx)
    cfg.dob_enabled = True
    cfg.dob_reg_coef = 0.01
    cfg.curriculum_enabled = True
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


def _g_param(rssm):
    # a representative plant-``g`` parameter (decoder first weight).
    for n, p in rssm.named_parameters():
        if 'dob_log' not in n and p.dim() >= 1:
            return p
    raise RuntimeError('no g param found')


def _grad_sum(p):
    return 0.0 if p.grad is None else float(p.grad.abs().sum())


def _check(wm_type):
    print(f'\n=== {wm_type} curriculum ===')
    cfg, model, batch = _mk(wm_type)
    rssm = model.dynamics
    core = rssm.deter_dim + rssm.stoch_flat_dim
    n_cv = rssm.n_cv
    dob_decay, dob_gain = rssm.dob_log_decay, rssm.dob_log_gain
    g_ref = _g_param(rssm)
    rew_ref = next(model.reward.parameters())

    # ---- partition flags for each stage ----
    fz1 = model.set_world_model_trainable(g=True, dob=False, reward=True)
    model.set_dob_active(False)
    assert g_ref.requires_grad and not dob_decay.requires_grad \
        and not dob_gain.requires_grad and rew_ref.requires_grad, \
        'Stage1 partition wrong'
    assert not rssm.dob_active, 'Stage1 dob_active must be False'
    print(f'[smoke] OK  Stage1 partition g=train dob=frozen reward=train, '
          f'dob_active=False {fz1} [{wm_type}]')

    fz2 = model.set_world_model_trainable(g=False, dob=True, reward=True)
    model.set_dob_active(True)
    assert (not g_ref.requires_grad) and dob_decay.requires_grad \
        and dob_gain.requires_grad and rew_ref.requires_grad, \
        'Stage2 partition wrong'
    assert rssm.dob_active, 'Stage2 dob_active must be True'
    print(f'[smoke] OK  Stage2 partition g=FROZEN dob=train reward=train, '
          f'dob_active=True {fz2} [{wm_type}]')

    fz3 = model.set_world_model_trainable(g=False, dob=False, reward=True)
    assert (not g_ref.requires_grad) and (not dob_decay.requires_grad) \
        and (not dob_gain.requires_grad) and rew_ref.requires_grad, \
        'Stage3 partition wrong'
    print(f'[smoke] OK  Stage3 partition g=FROZEN dob=FROZEN reward=train '
          f'{fz3} [{wm_type}]')

    # ---- dob_active toggles d_t zero/non-zero; feat width constant ----
    model.set_dob_active(False)
    _f, _, _, _last, ds_off, *_ = rssm.rollout_observed(batch['obs'], batch['act'],
                                                    sample=True)
    assert ds_off is not None and float(ds_off.abs().sum()) == 0.0, \
        'suppressed d_t must be exactly zero'
    assert _f.shape[-1] == core + n_cv, 'feat width must stay core+n_cv'
    assert float(_f[..., core:].abs().sum()) == 0.0, 'suppressed d-tail !=0'
    model.set_dob_active(True)
    _f2, _, _, _l2, ds_on, *_ = rssm.rollout_observed(batch['obs'], batch['act'],
                                                  sample=True)
    assert ds_on is not None and float(ds_on.abs().sum()) > 0.0, \
        'active d_t must be non-zero'
    assert _f2.shape[-1] == core + n_cv, 'feat width must stay core+n_cv'
    print(f'[smoke] OK  dob_active toggles d_t 0<->nonzero; feat width '
          f'== core+n_cv == {core + n_cv} both [{wm_type}]')

    # ---- Stage 1 grad isolation: recon trains g, NOT the DOB ----
    model.set_world_model_trainable(g=True, dob=False, reward=True)
    model.set_dob_active(False)
    model.zero_grad(set_to_none=True)
    losses, _, _ = world_model_loss(model, batch, cfg)
    losses['wm_total'].backward()
    g_grad = _grad_sum(g_ref)
    dob_grad = _grad_sum(dob_decay) + _grad_sum(dob_gain)
    assert g_grad > 0.0, 'Stage1: g must get recon gradient'
    assert dob_grad == 0.0, 'Stage1: DOB must get NO gradient (suppressed+frozen)'
    print(f'[smoke] OK  Stage1: recon trains g (|g_grad|={g_grad:.4f}) and NOT '
          f'the DOB (|dob_grad|={dob_grad:.1f}) [{wm_type}]')

    # ---- Stage 2 grad isolation: recon trains the DOB, NOT g ----
    cfg, model, batch = _mk(wm_type)        # fresh model (clean grads)
    rssm = model.dynamics
    g_ref = _g_param(rssm)
    model.set_world_model_trainable(g=False, dob=True, reward=True)
    model.set_dob_active(True)
    model.zero_grad(set_to_none=True)
    losses, _, _ = world_model_loss(model, batch, cfg)
    losses['wm_total'].backward()
    g_grad = _grad_sum(g_ref)
    dob_grad = _grad_sum(rssm.dob_log_decay) + _grad_sum(rssm.dob_log_gain)
    assert g_grad == 0.0, 'Stage2: g is FROZEN -> must get NO gradient'
    assert dob_grad > 0.0, 'Stage2: the DOB observer must get recon gradient'
    assert float(losses['dob_reg']) > 0.0, 'Stage2: dob_reg must be active'
    print(f'[smoke] OK  Stage2: recon trains the DOB (|dob_grad|={dob_grad:.4f}) '
          f'and NOT g (|g_grad|={g_grad:.1f}) — observer identifiable on the '
          f'fixed plant [{wm_type}]')

    # ---- Stage 3 grad isolation: WM static (wm_total carries NO trainable
    # gradient).  The real P3 path trains the reward head via reward-MTP +
    # the actor/critic via imagination — NOT wm_total — so wm_total being
    # non-differentiable (all of g + DOB frozen) is exactly correct.  The
    # reward head must STAY trainable so the P3 reward-MTP keeps adapting. ----
    cfg, model, batch = _mk(wm_type)
    rssm = model.dynamics
    rew_ref = next(model.reward.parameters())
    model.set_world_model_trainable(g=False, dob=False, reward=True)
    model.set_dob_active(True)
    model.zero_grad(set_to_none=True)
    losses, _, _ = world_model_loss(model, batch, cfg)
    assert not losses['wm_total'].requires_grad, \
        'Stage3: g + DOB frozen -> wm_total must be non-differentiable (static)'
    assert rew_ref.requires_grad, \
        'Stage3: reward head must stay trainable (P3 reward-MTP)'
    print(f'[smoke] OK  Stage3: wm_total non-differentiable (WM+DOB static) + '
          f'reward head trainable -> actor/critic train on a fixed '
          f'WM+observer [{wm_type}]')


if __name__ == '__main__':
    for wm in ('rssm', 'tssm'):
        _check(wm)
    print('\n[smoke] ALL CURRICULUM CHECKS PASSED (both backbones)')
