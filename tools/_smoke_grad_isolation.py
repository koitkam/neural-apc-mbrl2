"""Full gradient-leakage / optimizer-coverage audit (2026-06-10).

Systematic static + dynamic check that gradients flow ONLY where intended in the
DreamerV4 trainer.  Catches the two failure classes we have actually hit:
  * optimizer-coverage bug (a supervised param in NO optimizer group -> grads
    computed then silently discarded; e.g. the disturbance head pre-5a31041),
  * cross-contamination (a loss term leaking grad into the wrong param group;
    e.g. an auxiliary head shaping the WM trunk, or actor/critic touching the WM).

Two parts:
  (A) PARTITION CHECK — every model parameter is assigned to exactly the right
      optimizer group {world, actor, critic}; flag orphans (requires_grad but in
      NO optimizer) and overlaps (in >1 optimizer), and confirm frozen modules
      (target_value, prior_policy) are excluded.
  (B) PER-TERM GRADIENT ISOLATION — backward each loss term in isolation and
      assert grad reaches ONLY the intended group(s).

Backbone-agnostic (runs rssm + sf_transformer).  CPU-safe; touches no GPU run.

Run:
  CUDA_VISIBLE_DEVICES="" PYTHONPATH=$PWD \
  ./../neural-apc-mbrl-env/bin/python tools/_smoke_grad_isolation.py
"""
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from training.train import (  # noqa: E402
    TrainConfig, build_model, world_model_loss, agent_finetune_loss,
    imagination_step, expert_bc_p3_loss, _disturbance_head_loss,
)


def _mk(wm_type='rssm', stop_grad=True, rel_weight=0.3):
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
    cfg.d_model = 64
    cfg.head_hidden = 64
    cfg.head_n_layers = 2
    cfg.mtp_length = 4
    cfg.horizon = 4
    cfg.seq_len = 16
    cfg.disturbance_head_dim = 1
    cfg.disturbance_head_stop_grad = stop_grad
    cfg.disturbance_loss_scale = 1.0
    cfg.disturbance_loss_rel_weight = rel_weight
    cfg.actor_loss_type = 'reinforce'
    model = build_model(cfg)
    # Perturb the zero-init disturbance head so its upstream gradient is not
    # masked by the zero last layer (mirrors a TRAINED head).
    if model.disturbance is not None:
        with torch.no_grad():
            for p in model.disturbance.parameters():
                p.add_(0.3 * torch.randn_like(p))
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


def _group_ids(model):
    """Return {param_id: group_name} and the named-param map for reporting."""
    groups = {
        'world': model.parameters_world(),
        'actor': model.parameters_actor(),
        'critic': model.parameters_critic(),
    }
    id2groups = {}
    for gname, plist in groups.items():
        for p in plist:
            id2groups.setdefault(id(p), []).append(gname)
    return groups, id2groups


def _which_groups_got_grad(model, id2groups):
    """After a backward, return the set of optimizer groups with nonzero grad,
    plus flags for frozen modules and orphaned (in-no-group) params."""
    got = set()
    frozen_hit = []
    orphan_hit = []
    name_by_id = {id(p): n for n, p in model.named_parameters()}
    for p in model.parameters():
        if p.grad is None:
            continue
        if float(p.grad.abs().sum()) == 0.0:
            continue
        grps = id2groups.get(id(p))
        if grps:
            got.update(grps)
        else:
            # got grad but in no optimizer group -> orphan OR a frozen module.
            nm = name_by_id.get(id(p), '?')
            if not p.requires_grad:
                frozen_hit.append(nm)
            else:
                orphan_hit.append(nm)
    return got, frozen_hit, orphan_hit


def _zero(model):
    for p in model.parameters():
        p.grad = None


# ---------------------------------------------------------------------------
# (A) Partition check
# ---------------------------------------------------------------------------

def audit_partition(wm_type):
    print(f'\n=== (A) PARTITION CHECK [{wm_type}] ===')
    cfg, model, _ = _mk(wm_type)
    groups, id2groups = _group_ids(model)

    # overlaps: any param in >1 optimizer group
    overlaps = {pid: g for pid, g in id2groups.items() if len(g) > 1}
    name_by_id = {id(p): n for n, p in model.named_parameters()}
    assert not overlaps, (
        'PARAM IN >1 OPTIMIZER GROUP: '
        + ', '.join(f'{name_by_id.get(pid, "?")}={g}' for pid, g in overlaps.items()))
    print(f'[ok] no param is in more than one optimizer group '
          f'(world={len(groups["world"])}, actor={len(groups["actor"])}, '
          f'critic={len(groups["critic"])})')

    # orphans: requires_grad=True but in NO optimizer group (the disturbance bug)
    orphans = []
    frozen_ok = []
    for n, p in model.named_parameters():
        if id(p) in id2groups:
            continue
        if p.requires_grad:
            orphans.append(n)
        else:
            frozen_ok.append(n)
    assert not orphans, (
        f'ORPHANED TRAINABLE PARAMS (requires_grad but in NO optimizer -> grads '
        f'silently discarded): {orphans}')
    print(f'[ok] no orphaned trainable params (every requires_grad param is in '
          f'an optimizer)')

    # frozen modules must be the ONLY out-of-optimizer params, and must be
    # exactly target_value + prior_policy (+ buffers).
    unexpected_frozen = [n for n in frozen_ok
                         if not (n.startswith('target_value')
                                 or n.startswith('prior_policy'))]
    assert not unexpected_frozen, (
        f'UNEXPECTED params outside every optimizer (not target_value/'
        f'prior_policy): {unexpected_frozen}')
    print(f'[ok] the only out-of-optimizer params are frozen target_value + '
          f'prior_policy ({len(frozen_ok)} tensors)')

    # disturbance head must be in WORLD (the 5a31041 fix) when present
    if model.disturbance is not None:
        dist_groups = {tuple(id2groups.get(id(p), [])) for p in model.disturbance.parameters()}
        assert dist_groups == {('world',)}, (
            f'disturbance head must be in WORLD only, got {dist_groups}')
        print('[ok] disturbance head is in the WORLD optimizer group (5a31041)')


# ---------------------------------------------------------------------------
# (B) Per-term gradient isolation
# ---------------------------------------------------------------------------

def audit_isolation(wm_type):
    print(f'\n=== (B) PER-TERM GRADIENT ISOLATION [{wm_type}] ===')
    cfg, model, batch = _mk(wm_type, stop_grad=True)
    _groups, id2groups = _group_ids(model)

    def check(name, loss, expect, forbid):
        _zero(model)
        loss.backward(retain_graph=False)
        got, frozen_hit, orphan_hit = _which_groups_got_grad(model, id2groups)
        assert not frozen_hit, f'{name}: grad reached FROZEN modules {set(frozen_hit)}'
        assert not orphan_hit, f'{name}: grad reached ORPHAN params {set(orphan_hit)}'
        for g in expect:
            assert g in got, f'{name}: expected grad in {g}, got {sorted(got)}'
        for g in forbid:
            assert g not in got, f'{name}: LEAK — grad reached {g} (forbidden); got {sorted(got)}'
        print(f'[ok] {name:28} -> grad in {sorted(got)} (expect {expect}, forbid {forbid})')

    # 1. wm_total (recon/kl/overshoot/held + isolated disturbance probe) -> world only
    wm_losses, _, agent_hid = world_model_loss(model, batch, cfg)
    check('wm_total (stop_grad probe)', wm_losses['wm_total'],
          expect=['world'], forbid=['actor', 'critic'])

    # 2. reward MTP (the P3 WM-update agent term) -> world only (reward head + WM)
    cfg2, model2, batch2 = _mk(wm_type, stop_grad=True)
    _g2, id2g2 = _group_ids(model2)
    wl2, _, ah2 = world_model_loss(model2, batch2, cfg2)
    ag2 = agent_finetune_loss(model2, batch2, ah2, cfg2)

    def check2(name, loss, expect, forbid):
        _zero(model2)
        loss.backward(retain_graph=True)
        got, fr, orph = _which_groups_got_grad(model2, id2g2)
        assert not fr, f'{name}: grad reached FROZEN {set(fr)}'
        assert not orph, f'{name}: grad reached ORPHAN {set(orph)}'
        for g in expect:
            assert g in got, f'{name}: expected {g}, got {sorted(got)}'
        for g in forbid:
            assert g not in got, f'{name}: LEAK to {g}; got {sorted(got)}'
        print(f'[ok] {name:28} -> grad in {sorted(got)} (expect {expect}, forbid {forbid})')

    check2('reward_mtp_total', ag2['reward_mtp_total'],
           expect=['world'], forbid=['actor', 'critic'])
    # 3. agent_total (P1/P2 path = bc_scale*bc + reward_scale*reward_mtp) ->
    #    legitimately hits BOTH actor (BC) and world (reward head + WM), but
    #    NEVER critic.  (The dict's 'bc_loss' is .detach()'d for logging, so the
    #    live BC gradient lives only inside agent_total.)
    cfg2.bc_scale = 0.15          # ensure the BC term is live in agent_total
    ag2b = agent_finetune_loss(model2, batch2, ah2, cfg2)
    check2('agent_total (P1/P2 BC+rmtp)', ag2b['agent_total'],
           expect=['actor', 'world'], forbid=['critic'])

    # 4. imagination actor_loss -> actor only; critic_loss -> critic only
    cfg3, model3, batch3 = _mk(wm_type, stop_grad=True)
    _g3, id2g3 = _group_ids(model3)
    ac = imagination_step(model3, batch3, cfg3)

    def check3(name, loss, expect, forbid):
        _zero(model3)
        loss.backward(retain_graph=True)
        got, fr, orph = _which_groups_got_grad(model3, id2g3)
        assert not fr, f'{name}: grad reached FROZEN {set(fr)}'
        assert not orph, f'{name}: grad reached ORPHAN {set(orph)}'
        for g in expect:
            assert g in got, f'{name}: expected {g}, got {sorted(got)}'
        for g in forbid:
            assert g not in got, f'{name}: LEAK to {g}; got {sorted(got)}'
        print(f'[ok] {name:28} -> grad in {sorted(got)} (expect {expect}, forbid {forbid})')

    check3('imag actor_loss', ac['actor_loss'],
           expect=['actor'], forbid=['world', 'critic'])
    check3('imag critic_loss', ac['critic_loss'],
           expect=['critic'], forbid=['world', 'actor'])

    # 5. expert_bc_p3 anchor (detaches agent_hid) -> actor only
    cfg4, model4, batch4 = _mk(wm_type, stop_grad=True)
    _g4, id2g4 = _group_ids(model4)
    _wl4, _, ah4 = world_model_loss(model4, batch4, cfg4)
    bc_p3, _ = expert_bc_p3_loss(model4, batch4, ah4)
    _zero(model4)
    bc_p3.backward()
    got, fr, orph = _which_groups_got_grad(model4, id2g4)
    assert not fr and not orph, f'expert_bc_p3: frozen={fr} orphan={orph}'
    assert got == {'actor'}, f'expert_bc_p3 must hit actor ONLY (agent_hid detached), got {sorted(got)}'
    print(f'[ok] {"expert_bc_p3 (detached hid)":28} -> grad in {sorted(got)} (expect [actor])')


# ---------------------------------------------------------------------------
# (C) disturbance shaping regime: trunk grad bounded + sim-agnostic
# ---------------------------------------------------------------------------

def audit_disturbance_regimes(wm_type):
    print(f'\n=== (C) DISTURBANCE REGIMES [{wm_type}] ===')
    # stop_grad: head trains, trunk gets ZERO grad from the disturbance term
    cfg, model, batch = _mk(wm_type, stop_grad=True)
    feat = _feat(model, cfg, batch)
    _zero(model)
    term, _, _ = _disturbance_head_loss(model, feat, batch['dist'], torch.tensor(0.2), cfg)
    term.backward()
    trunk = _trunk_grad(model, wm_type)
    head = sum(float(p.grad.abs().sum()) for p in model.disturbance.parameters() if p.grad is not None)
    assert head > 0 and trunk == 0.0, f'stop_grad: head={head} trunk={trunk} (trunk must be 0)'
    print(f'[ok] stop_grad=True: head trains (|g|={head:.2f}), trunk grad == 0 (isolated)')

    # shaping adaptive: trunk grad > 0 but term == rel * recon_term (bounded)
    cfg, model, batch = _mk(wm_type, stop_grad=False, rel_weight=0.3)
    cfg.recon_scale = 0.1
    feat = _feat(model, cfg, batch)
    recon = torch.tensor(0.25)
    _zero(model)
    term, _, _ = _disturbance_head_loss(model, feat, batch['dist'], recon, cfg)
    trunk_term = float(term)
    expected = 0.3 * float(0.1 * recon)
    assert abs(trunk_term - expected) < 1e-5, f'adaptive term {trunk_term} != 0.3*recon_term {expected}'
    term.backward()
    trunk = _trunk_grad(model, wm_type)
    assert trunk > 0, 'shaping must reach the trunk'
    print(f'[ok] stop_grad=False: term==0.3*recon_term ({trunk_term:.5f}), trunk grad>0 ({trunk:.2f}) — bounded')


def _feat(model, cfg, batch):
    B, T = batch['obs'].shape[:2]
    if cfg.world_model_type == 'rssm':
        f, *_ = model.dynamics.rollout_observed(batch['obs'], batch['act'], sample=True)
        return f
    z = model.tokenizer.encode(batch['obs'])
    tau = torch.full((B, T), 0.75)
    d = torch.full((B, T), 0.25)
    return model.dynamics(z, tau, d, batch['act'])['agent_hid']


def _trunk_grad(model, wm_type):
    g = sum(float(p.grad.abs().sum()) for p in model.dynamics.parameters() if p.grad is not None)
    if wm_type != 'rssm' and getattr(model, 'tokenizer', None) is not None:
        g += sum(float(p.grad.abs().sum()) for p in model.tokenizer.parameters() if p.grad is not None)
    return g


if __name__ == '__main__':
    for wm in ('rssm', 'sf_transformer'):
        audit_partition(wm)
        audit_isolation(wm)
        audit_disturbance_regimes(wm)
    print('\n[grad-audit] ALL GRADIENT-ISOLATION CHECKS PASSED (both backbones)')
