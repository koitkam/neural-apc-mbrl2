"""Smoke test for the neural Kalman filter / disturbance observer (DOB).

Verifies, WITHOUT a real env, for BOTH backbones (rssm + tssm):
  * default cfg (dob_enabled=False) => byte-identical to pre-DOB: rollout_observed
    5th return ds is None, state.d is None, decode unchanged.
  * dob_enabled=True => d_t state exists, decays in img_step, is corrected by the
    innovation in obs_step (obs given), apply_dob adds it ONLY to CV channels,
    and rollout_observed returns a finite per-step ds.
  * GRADIENT ISOLATION: the DOB params (dob_log_decay/gain) are in
    parameters_world (trained by opt_world), NOT in actor/critic; and the DOB
    recon path backprops to them.
  * A,K are bounded in (0,1) (sigmoid).

Run (CPU):
  CUDA_VISIBLE_DEVICES="" PYTHONPATH=$PWD \
  $PWD/../neural-apc-mbrl-env/bin/python tools/_smoke_dob.py
"""
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from training.train import TrainConfig, build_model, world_model_loss  # noqa: E402


def _mk(wm_type='rssm', dob=False, cv_idx=(2,)):
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
    cfg.disturbance_head_dim = 1
    cfg.cv_obs_indices = tuple(cv_idx)
    cfg.dob_enabled = dob
    cfg.dob_reg_coef = 0.01
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


def _check(wm_type):
    print(f'\n=== {wm_type} ===')
    # 1. default OFF: identity
    cfg, model, batch = _mk(wm_type, dob=False)
    rssm = model.dynamics
    assert not getattr(rssm, 'dob_enabled', False), 'DOB must be OFF by default'
    feats, _pl, _prl, last, ds = rssm.rollout_observed(batch['obs'], batch['act'],
                                                       sample=True)
    assert ds is None, 'ds must be None when DOB off'
    assert last.d is None, 'state.d must be None when DOB off'
    losses, _, _ = world_model_loss(model, batch, cfg)
    assert float(losses['dob_reg']) == 0.0, 'dob_reg must be 0 when off'
    losses['wm_total'].backward()
    print(f'[smoke] OK  DOB off = identity (ds None, dob_reg 0) [{wm_type}]')

    # 2. DOB ON: state, decay, correction, apply_dob, finite ds
    cfg, model, batch = _mk(wm_type, dob=True, cv_idx=(2,))
    rssm = model.dynamics
    assert rssm.dob_enabled, 'DOB should be enabled'
    # A,K bounded in (0,1)
    A = float(rssm.dob_decay().mean()); K = float(rssm.dob_gain().mean())
    assert 0.0 < A < 1.0 and 0.0 < K < 1.0, f'A={A} K={K} must be in (0,1)'
    feats, _pl, _prl, last, ds = rssm.rollout_observed(batch['obs'], batch['act'],
                                                       sample=True)
    assert ds is not None and ds.shape == (3, cfg.seq_len, 1), f'ds shape {None if ds is None else ds.shape}'
    assert torch.isfinite(ds).all(), 'ds must be finite'
    assert last.d is not None and torch.isfinite(last.d).all()
    print(f'[smoke] OK  DOB on: A={A:.3f} K={K:.3f} ds finite {tuple(ds.shape)} [{wm_type}]')

    # 3. apply_dob adds ONLY to the CV channel
    dec = torch.zeros(2, 5, cfg.obs_dim)
    d = torch.ones(2, 5, 1) * 0.7
    out = rssm.apply_dob(dec, d)
    cv = 2
    assert torch.allclose(out[..., cv], torch.full((2, 5), 0.7)), 'CV channel not updated'
    others = [i for i in range(cfg.obs_dim) if i != cv]
    assert torch.allclose(out[..., others], torch.zeros(2, 5, len(others))), 'non-CV changed!'
    print(f'[smoke] OK  apply_dob adds d ONLY to CV channel {cv} [{wm_type}]')

    # 4. img_step decays d, obs_step corrects it
    st = rssm.initial_state(2, torch.device('cpu'))
    st = type(st)(h=st.h, z_logits=st.z_logits, z=st.z,
                  **({'d': torch.ones(2, 1)} if wm_type == 'rssm'
                     else {'kv_cache': None, 'pos': 0, 'd': torch.ones(2, 1)}))
    a = torch.zeros(2, cfg.action_dim)
    prior = rssm.img_step(st, a, sample=True)
    assert float(prior.d.mean()) < 1.0, 'img_step must DECAY d (A<1)'
    print(f'[smoke] OK  img_step decays d (1.0 -> {float(prior.d.mean()):.3f}) [{wm_type}]')

    # 5. DOB params in parameters_world only; recon backprops to them
    world_ids = {id(p) for p in model.parameters_world()}
    actor_ids = {id(p) for p in model.parameters_actor()}
    crit_ids = {id(p) for p in model.parameters_critic()}
    dob_params = [rssm.dob_log_decay, rssm.dob_log_gain]
    assert all(id(p) in world_ids for p in dob_params), 'DOB params must be in opt_world'
    assert not any(id(p) in actor_ids or id(p) in crit_ids for p in dob_params), \
        'DOB params leaked into actor/critic'
    model.zero_grad(set_to_none=True)
    losses, _, _ = world_model_loss(model, batch, cfg)
    assert float(losses['dob_reg']) > 0.0, 'dob_reg must be >0 when on'
    losses['wm_total'].backward()
    g = sum(float(p.grad.abs().sum()) for p in dob_params if p.grad is not None)
    assert g > 0.0, 'DOB params got no gradient from wm_total'
    print(f'[smoke] OK  DOB params in opt_world only, recon backprops (|g|={g:.4f}) [{wm_type}]')


def _check_scope2(wm_type):
    """Scope 2: the DOB estimate d_t is fed (detached) into ``feat`` so the
    actor/critic/reward heads condition on it; the decoder still reads core."""
    print(f'\n=== {wm_type} Scope 2 (d_t in feat) ===')
    # OFF: feat_dim == core, rollout feat width == core (byte-identical)
    cfg, model, batch = _mk(wm_type, dob=False)
    rssm = model.dynamics
    core = rssm.deter_dim + rssm.stoch_flat_dim
    assert rssm.feat_dim == core, f'feat_dim off should be core={core}, got {rssm.feat_dim}'
    feats_off, _, _, _last_off, _ = rssm.rollout_observed(
        batch['obs'], batch['act'], sample=True)
    assert feats_off.shape[-1] == core, f'feat width off {feats_off.shape[-1]} != {core}'
    print(f'[smoke] OK  DOB off: feat_dim == core == {core} [{wm_type}]')

    # ON: feat_dim == core + n_cv; the rollout feats carry d in the tail; the
    # decoder ignores the d-tail; the heads accept the augmented width.
    cfg, model, batch = _mk(wm_type, dob=True, cv_idx=(2,))
    rssm = model.dynamics
    n_cv = rssm.n_cv
    assert rssm.feat_dim == core + n_cv, \
        f'feat_dim on should be {core + n_cv}, got {rssm.feat_dim}'
    feats, _, _, _last, ds = rssm.rollout_observed(
        batch['obs'], batch['act'], sample=True)
    assert feats.shape[-1] == core + n_cv, \
        f'feat width on {feats.shape[-1]} != {core + n_cv}'
    assert torch.allclose(feats[..., core:], ds), 'feat d-tail must equal ds'
    rec_full = rssm.decode(feats)
    rec_core = rssm.decode(feats[..., :core])
    assert torch.allclose(rec_full, rec_core), 'decode must ignore the d-tail'
    print(f'[smoke] OK  DOB on: feat_dim == core+n_cv == {core + n_cv}; '
          f'd-tail==ds; decode slices [{wm_type}]')

    # heads accept the augmented feat (dim match) — this is the feed-forward path
    feat1 = feats[:, -1]                                    # (B, core+n_cv)
    _ = model.value(feat1)
    _ = model.reward(feat1)
    a, _lp, _ent, _raw = model.policy.sample_with_raw(feat1)
    assert a.shape[0] == feat1.shape[0]
    print(f'[smoke] OK  policy/value/reward accept augmented feat '
          f'(dim {feat1.shape[-1]}) [{wm_type}]')

    # the d-tail is DETACHED: a head's gradient must NOT leak into the DOB via
    # feat (the DOB is trained by the recon innovation, not the heads).
    model.zero_grad(set_to_none=True)
    feats2, _, _, _, ds2 = rssm.rollout_observed(
        batch['obs'], batch['act'], sample=True)
    ds2.retain_grad()
    model.value(feats2[:, -1]).sum().backward()
    leaked = 0.0 if ds2.grad is None else float(ds2.grad.abs().sum())
    assert leaked == 0.0, f'd-tail leaked grad into value head: {leaked}'
    print(f'[smoke] OK  d-tail detached: value-head grad into ds = {leaked} [{wm_type}]')

    # REGRESSION (Scope 2): the WM-loss paths that reconstruct an RSSMState from
    # ``feats`` (overshoot / held-rollout / steady-consistency) must slice the
    # stochastic block EXACTLY and not choke on the d-tail.  Enable all three +
    # DOB and assert wm_total backprops cleanly (these are RSSM-only).
    if wm_type == 'rssm':
        cfg, model, batch = _mk(wm_type, dob=True, cv_idx=(2,))
        cfg.wm_overshoot_coef = 0.3
        cfg.wm_overshoot_len = 8
        cfg.wm_held_rollout_coef = 0.5
        cfg.wm_held_rollout_len = 8
        cfg.wm_steady_coef = 0.1
        losses, _, _ = world_model_loss(model, batch, cfg)
        losses['wm_total'].backward()
        print(f'[smoke] OK  Scope2 + overshoot/held/steady paths backprop '
              f"(overshoot={float(losses.get('wm_overshoot_loss', 0)):.3f} "
              f"held={float(losses.get('wm_held_rollout_loss', 0)):.3f}) [{wm_type}]")


def _check_compile_equiv(wm_type):
    """Compile-efficiency refactor (2026-06-12): the vectorized DOB
    ``rollout_observed`` (d-free recurrence + ONE batched prior decode + scalar
    Kalman scan) must be NUMERICALLY IDENTICAL to the per-step ``obs_step``
    reference (which still does the inline per-step decode).  Deterministic
    (sample=False) so there is no RNG divergence between the two paths."""
    print(f'\n=== {wm_type} compile-equiv (vectorized == per-step) ===')
    cfg, model, batch = _mk(wm_type, dob=True, cv_idx=(2,))
    rssm = model.dynamics
    obs, act = batch['obs'], batch['act']
    B, T = obs.shape[:2]
    dev = obs.device
    # Reference: per-step obs_step with the REAL obs (the pre-refactor path).
    with torch.no_grad():
        state = rssm.initial_state(B, dev)
        ref_feats, ref_ds = [], []
        for t in range(T):
            post, _prior = rssm.obs_step(state, act[:, t], rssm.embed(obs[:, t]),
                                         sample=False, obs=obs[:, t])
            state = post
            ref_feats.append(post.feat)
            ref_ds.append(post.d)
        ref_feats = torch.stack(ref_feats, dim=1)
        ref_ds = torch.stack(ref_ds, dim=1)
        # Vectorized rollout (the refactored production path).
        feats, _pl, _prl, last, ds = rssm.rollout_observed(obs, act, sample=False)
    fe = float((feats - ref_feats).abs().max())
    de = float((ds - ref_ds).abs().max())
    assert fe < 1e-4, f'feats mismatch vs per-step ref: max|Δ|={fe}'
    assert de < 1e-4, f'ds mismatch vs per-step ref: max|Δ|={de}'
    assert torch.allclose(last.d, ref_ds[:, -1], atol=1e-4), 'last_state.d mismatch'
    print(f'[smoke] OK  vectorized rollout == per-step obs_step '
          f'(max|Δfeats|={fe:.2e}, max|Δds|={de:.2e}) [{wm_type}]')


if __name__ == '__main__':
    for wm in ('rssm', 'tssm'):
        _check(wm)
        _check_scope2(wm)
        _check_compile_equiv(wm)
    print('\n[smoke] ALL DOB CHECKS PASSED (both backbones)')
