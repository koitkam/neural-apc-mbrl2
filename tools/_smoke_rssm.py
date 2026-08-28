"""Standalone smoke test for the RSSM world-model port (P68).

Exercises every RSSM execution path that training touches WITHOUT a real
env:
  * RSSMDynamics shape contracts (rollout_observed / obs_step / img_step /
    decode) + rssm_kl_loss.
  * V4 heads on RSSM feat (reward MTP, value, policy).
  * train.world_model_loss   (P1/P2 WM loss)
  * train.agent_finetune_loss (P2 BC + reward MTP)
  * train._realsim_actor_critic_step    (P3 actor/critic on real-sim rollouts)

All outputs must be finite.  Run:
  CUDA_VISIBLE_DEVICES="" PYTHONPATH=$PWD \
  CONTROL_SETUP_JSON=$PWD/simulation/test_sim/control_setup.json \
  ~/neural-APC-mbrl2-env/bin/python tools/_smoke_rssm.py
"""
import torch

from training.train import (
    TrainConfig, build_model, world_model_loss,
                            agent_finetune_loss, _realsim_actor_critic_step,
                            expert_bc_p3_loss, _adaptive_return_cap,
                            _steady_held_mask, _critic_anchor_lambda,
                            _critic_anchor_coef, _force_p1_cap_at,
                            _skip_storm_continue_p1,
                            _skip_storm_should_continue_p1,
                            _wm_fidelity_es_suppressed_frozen_g,
                            _resolve_aux_tbptt_steps, _buffer_lap_iters,
                            _resolve_inject_cadence, _cfg_from_env,
                            collect_episode, collect_prbs_episode,
                            _build_dv_prbs_schedule,
                            _isolated_hold_action, _hold_other_action_dims,
                            _isolation_settle_counts,
                            _isolation_buf_capacity, _isolation_sample_seq_len,
                            _wm_train_seq_len, wm_train_seq_len_for_plant,
                            _dv_isolation_delta, _isolation_dcv_scales,
                            _stash_isolation_dcv_scales,
                            _isolation_dcv_scale_payload, _isolation_edge_du,
                            _scale_isolation_level, _gain_col_rms,
                            _dynamics_g_trainable,
                            _maybe_clean_steady_seed,
                            _recon_still_healthy, _skip_storm_restore_ckpt,
                            _actor_experiment_valid,
                            _should_skip_invalid_p3,
                            _should_warm_restore_wm_best,
                            _should_restore_last_ok_at_p1_freeze,
                            _wm_recon_scalar,
                            _auto_if_unset, _write_resolved_run_plan,
                            _resolve_compile_mode,
                            _clone_module_state, _refresh_module_state,
                            _p1_need_agent_finetune)


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
    assert int(getattr(model, 'n_critics', 1)) == int(getattr(cfg, 'n_critics', 1))
    assert len(model.values) == model.n_critics
    diag = _realsim_actor_critic_step(model, batch, cfg)
    _finite('_realsim_actor_critic_step', diag)
    # P26 RCA / P27: freeze return_scale — second call must not move ret_scale.
    s0 = float(model.ret_scale.reshape(-1)[0])
    _ = _realsim_actor_critic_step(model, batch, cfg, freeze_return_scale=True)
    s1 = float(model.ret_scale.reshape(-1)[0])
    assert abs(s1 - s0) < 1e-8, f'return_scale moved under freeze: {s0} -> {s1}'
    print(f'[smoke] OK  n_critics={model.n_critics} return_scale freeze ({s0:.4f})')

    # P28 skip-storm: capping P1 must close the extension budget so the
    # next P1→P2 gate cannot re-open full-BPTT gain-match.
    _p1, _ext, _cap = _force_p1_cap_at(12345)
    assert (_p1, _ext, _cap) == (12345, 0, 0), (_p1, _ext, _cap)
    print('[smoke] OK  P1 skip-storm cap closes extension budget')
    # P32: first storm keeps original P1 AND the quality-gate
    # extension (P26/P31 needed it past ~iter 75). Storm 2 still
    # ``_force_p1_cap_at`` (closes extension so a cap-now cannot
    # re-open the next-iter gate — P28).
    _p1c, _extc, _capc = _skip_storm_continue_p1(753960, 12000, 175924)
    assert (_p1c, _extc, _capc) == (753960, 12000, 175924), (
        _p1c, _extc, _capc)
    assert int(TrainConfig().skip_storm_p1_cap_after) == 2
    assert _skip_storm_should_continue_p1(1, 2) is True
    assert _skip_storm_should_continue_p1(2, 2) is False
    assert _skip_storm_should_continue_p1(1, 1) is False
    assert _wm_fidelity_es_suppressed_frozen_g(False) is True
    assert _wm_fidelity_es_suppressed_frozen_g(True) is False
    print('[smoke] OK  P1 skip-storm continue first / cap second; frozen-g ES')
    assert _p1_need_agent_finetune(0.0, False, 0, 100) is False
    assert _p1_need_agent_finetune(0.0, True, 98, 100) is False
    assert _p1_need_agent_finetune(0.0, True, 99, 100) is True
    assert _p1_need_agent_finetune(1.0, False, 0, 100) is True
    print('[smoke] OK  P1 MTP skip when reward_scale_loss_p1=0 (log last only)')

    class _Snap(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Parameter(torch.arange(4.0))

    _m = _Snap()
    _d0 = _clone_module_state(_m, torch.device('cpu'))
    _m.w.data.add_(1.0)
    _d1 = _refresh_module_state(_d0, _m, torch.device('cpu'))
    assert _d1 is _d0
    assert torch.equal(_d1['w'], _m.w.detach())
    assert _d1['w'].data_ptr() != _m.w.data_ptr()
    print('[smoke] OK  last-ok snapshot refresh copy_ (same storage, not aliased)')
    _tc = TrainConfig()
    assert float(_tc.gain_match_huber_beta) == 1.0
    assert int(_tc.aux_tbptt_steps) == 16
    assert not hasattr(_tc, 'gain_match_relative')
    print('[smoke] OK  gain-match defaults (abs Huber only, beta=1, TBPTT=16)')

    # Isolation TBPTT is sim-adaptive: 16 of K≈55 (test_sim) scales as K/3.5.
    _tb = TrainConfig()
    _tb.wm_input_isolation_len = 55
    assert _resolve_aux_tbptt_steps(_tb) == 16, _tb.aux_tbptt_steps
    _tb_fast = TrainConfig()
    _tb_fast.wm_input_isolation_len = 15
    assert _resolve_aux_tbptt_steps(_tb_fast) == 8, _tb_fast.aux_tbptt_steps
    _tb_slow = TrainConfig()
    _tb_slow.horizon = 120
    _tb_slow.wm_input_isolation_len = 120
    assert _resolve_aux_tbptt_steps(_tb_slow) == 34, _tb_slow.aux_tbptt_steps
    _tb_set = TrainConfig()
    _tb_set.aux_tbptt_steps = 2
    _tb_set.wm_input_isolation_len = 55
    assert _resolve_aux_tbptt_steps(_tb_set) == 2
    _tb_ex = TrainConfig()
    _tb_ex.aux_tbptt_steps = 16
    _tb_ex.wm_input_isolation_len = 120
    _tb_ex._explicit_fields = {'aux_tbptt_steps'}
    assert _resolve_aux_tbptt_steps(_tb_ex) == 16
    print('[smoke] OK  aux_tbptt_steps sim-adaptive (16-of-55; explicit wins)')

    # P28 follow-up 3: skip-storm restores last healthy P1 step, not wm_best.
    import tempfile
    from pathlib import Path as _P
    with tempfile.TemporaryDirectory() as _td:
        _tmp = _P(_td)
        _last = _tmp / 'wm_last_ok.pt'
        _best = _tmp / 'wm_best.pt'
        _last.write_bytes(b'ok')
        _best.write_bytes(b'best')
        _p, _src = _skip_storm_restore_ckpt(_last, _best)
        assert _src == 'wm_last_ok' and _p == _last, (_src, _p)
        _last.unlink()
        _p, _src = _skip_storm_restore_ckpt(_last, _best)
        assert _src == 'wm_best' and _p == _best, (_src, _p)
        _best.unlink()
        _p, _src = _skip_storm_restore_ckpt(_last, _best)
        assert _src == 'none' and _p is None, (_src, _p)
    assert _recon_still_healthy(0.004, None)
    assert _recon_still_healthy(0.004, 0.0039, 5.0)
    assert not _recon_still_healthy(0.50, 0.0039, 5.0)
    assert not _recon_still_healthy(float('nan'), 0.01)
    assert _actor_experiment_valid(skip_storm_source='wm_last_ok',
                                   gain_not_ready_capped=False)
    assert not _actor_experiment_valid(skip_storm_source='wm_best',
                                       gain_not_ready_capped=False)
    assert not _actor_experiment_valid(skip_storm_source=None,
                                       gain_not_ready_capped=True)
    assert bool(TrainConfig().skip_invalid_p3) is True
    assert _should_skip_invalid_p3(actor_valid=False, skip_enabled=True)
    assert not _should_skip_invalid_p3(actor_valid=True, skip_enabled=True)
    assert not _should_skip_invalid_p3(actor_valid=False, skip_enabled=False)
    print('[smoke] OK  skip_invalid_p3 default-on (GAIN_NOT_READY skips P3)')
    assert float(TrainConfig().skip_storm_last_ok_recon_ratio) == 5.0
    assert bool(TrainConfig().wm_best_restore_at_p2) is False
    assert bool(TrainConfig().wm_best_restore_at_p3) is False
    assert int(TrainConfig().wm_best_restore_min_gap) == 10
    # Helper still restores when explicitly enabled (opt-in).
    assert _should_warm_restore_wm_best(
        restore_enabled=True, skip_storm_recovered=False,
        wm_best_iter=20, total_iters=80, min_gap=10, wm_best_exists=True)
    # Skip-storm last-ok must not be overwritten by the fidelity peak.
    assert not _should_warm_restore_wm_best(
        restore_enabled=True, skip_storm_recovered=True,
        wm_best_iter=20, total_iters=50, min_gap=10, wm_best_exists=True)
    assert not _should_warm_restore_wm_best(
        restore_enabled=True, skip_storm_recovered=False,
        wm_best_iter=75, total_iters=80, min_gap=10, wm_best_exists=True)
    assert not _should_warm_restore_wm_best(
        restore_enabled=False, skip_storm_recovered=False,
        wm_best_iter=20, total_iters=80, min_gap=10, wm_best_exists=True)
    assert not _should_warm_restore_wm_best(
        restore_enabled=True, skip_storm_recovered=False,
        wm_best_iter=20, total_iters=80, min_gap=10, wm_best_exists=True,
        last_ok_on_model=True)
    print('[smoke] OK  skip-storm last-ok restore prefers late-P1 observer')
    print('[smoke] OK  skip-storm blocks P1→P2 wm_best overwrite')
    # P31: quality-gate CAPPED while recon is detonated must restore last-ok
    # (skip-storm needs >5 skips; a single huge-grad step does not trip it).
    assert _should_restore_last_ok_at_p1_freeze(
        recon=0.5669, recon_best=0.0033, ratio=5.0, has_last_ok=True)
    assert _should_restore_last_ok_at_p1_freeze(
        recon=0.71, recon_best=0.0035, ratio=5.0, has_last_ok=True)
    assert not _should_restore_last_ok_at_p1_freeze(
        recon=0.004, recon_best=0.0035, ratio=5.0, has_last_ok=True)
    assert not _should_restore_last_ok_at_p1_freeze(
        recon=0.71, recon_best=0.0035, ratio=5.0, has_last_ok=False)
    _nan_r = _wm_recon_scalar(None)
    assert _nan_r != _nan_r
    assert abs(_wm_recon_scalar({'recon_loss': 0.0035}) - 0.0035) < 1e-12
    print('[smoke] OK  P1 detonated-freeze last-ok restore (P31 RCA)')

    # P28 follow-up 5: P1 re-inject EVERY is f(buffer lap).  test_sim
    # (ep_len=1220, 400k cap, 5 eps/iter) stays 20/10.
    _inj = TrainConfig()
    assert int(_inj.const_action_inject_every) == 20
    assert int(_inj.dv_prbs_inject_every) == 10
    assert int(_inj.wm_probe_every_iters) == 10
    assert float(_inj.gain_ready_lo) == 0.80
    assert bool(_inj.p1_gain_gate) is True
    _inj.episode_length = 1220
    _inj.buffer_capacity_steps = 400_000
    _inj.ep_per_iter = 5
    assert abs(_buffer_lap_iters(_inj) - 65.57377) < 0.01
    _got = _resolve_inject_cadence(_inj, n_mv=1, n_dv=1)
    assert _got['const_action_inject_every'] == 20, _got
    assert _got['step_test_inject_every'] == 20, _got
    assert _got['dv_prbs_inject_every'] == 10, _got
    assert _got['expert_inject_every'] == 20, _got
    assert _got['const_action_inject_n'] == 5, _got
    assert _got['step_test_inject_n'] == 2, _got
    assert _got['dv_prbs_inject_n'] == 2, _got
    assert _got['expert_inject_n'] == 3, _got
    _slow = TrainConfig()
    _slow.episode_length = 4000
    _slow.buffer_capacity_steps = 400_000
    _slow.ep_per_iter = 5
    _sg = _resolve_inject_cadence(_slow, n_mv=1, n_dv=1)
    assert _sg['const_action_inject_every'] == 6, _sg   # 20-iter lap × 0.30
    assert _sg['dv_prbs_inject_every'] == 5, _sg        # floor 5
    _fast = TrainConfig()
    _fast.episode_length = 500
    _fast.buffer_capacity_steps = 400_000
    _fast.ep_per_iter = 5
    _fg = _resolve_inject_cadence(_fast, n_mv=1, n_dv=1)
    assert _fg['const_action_inject_every'] == 48, _fg  # 160-iter lap × 0.30
    assert _fg['dv_prbs_inject_every'] == 10, _fg       # capped at warmup/4
    _off = TrainConfig()
    _off.episode_length = 1220
    _off.const_action_inject_every = 0
    _off.step_test_inject_n = 0
    _og = _resolve_inject_cadence(_off, n_mv=1, n_dv=1)
    assert _og['const_action_inject_every'] == 0
    assert _og['step_test_inject_n'] == 0
    _ex = TrainConfig()
    _ex.episode_length = 4000
    _ex.const_action_inject_every = 20
    _ex._explicit_fields = {'const_action_inject_every'}
    assert _resolve_inject_cadence(_ex, n_mv=1, n_dv=1)['const_action_inject_every'] == 20
    # P28 follow-up 6: inject N is f(n_mv, n_dv).  distillation 4 MV + 1 DV.
    _mimo = TrainConfig()
    _mimo.episode_length = 1220
    _mimo.buffer_capacity_steps = 400_000
    _mimo.ep_per_iter = 5
    _mg = _resolve_inject_cadence(_mimo, n_mv=4, n_dv=1)
    assert _mg['const_action_inject_n'] == 5, _mg      # max(5, 5)
    assert _mg['step_test_inject_n'] == 5, _mg         # max(2, 5)
    assert _mg['dv_prbs_inject_n'] == 4, _mg           # max(2, 4)
    assert _mg['expert_inject_n'] == 4, _mg            # max(3, 4)
    _exn = TrainConfig()
    _exn.episode_length = 1220
    _exn.step_test_inject_n = 2
    _exn._explicit_fields = {'step_test_inject_n'}
    assert _resolve_inject_cadence(_exn, n_mv=4, n_dv=1)['step_test_inject_n'] == 2
    print('[smoke] OK  inject cadence sim-adaptive (test_sim 20/10 n 5/2/2/3; 0=off)')

    # P28 follow-up 7: isolation settle is per-input; isolation_buf cap
    # holds every channel's settle episodes.  test_sim 1+1 stays 24+24 / 48.
    import numpy as _np
    _tgt = _np.array([[0.4, -0.2, 0.9], [0.1, 0.3, -0.5]], dtype='float32')
    _held = _hold_other_action_dims(_tgt, isolate_dim=1, hold_level=0.25)
    assert _np.allclose(_held[:, 1], _tgt[:, 1])
    assert _np.allclose(_held[:, 0], 0.25) and _np.allclose(_held[:, 2], 0.25)
    _passthru = _hold_other_action_dims(_tgt, None)
    assert _passthru is _tgt
    assert _isolation_settle_counts(1, 1, 24) == (24, 24)
    assert _isolation_settle_counts(4, 1, 24) == (96, 24)
    assert _isolation_buf_capacity(n_mv=1, n_dv=1, n_settle_per=24) == 48
    assert _isolation_buf_capacity(n_mv=4, n_dv=1, n_settle_per=24) == 120
    assert _isolation_settle_counts(2, 0, 0) == (0, 0)
    print('[smoke] OK  isolation settle per-input (test_sim 24+24/cap 48; '
          'distillation 96+24/cap 120; cap settle-only)')

    # P28 follow-up 8: long-hold isolation settle zeros sim noise (P89 gate).
    class _DummyIsoEnv:
        def __init__(self):
            self.action_dim = 2
            self.obs_dim = 3
            self.rng = _np.random.default_rng(0)
            self._schedule = ['x']
            self._hidden_disturbance = 'ou'
            self.noise_scale = 1.0
            self.acts = []
        def reset(self, exploration=True):
            return _np.zeros((1, self.obs_dim), dtype='float32')
        def step(self, a):
            self.acts.append(_np.asarray(a, dtype='float32').copy())
            return (_np.zeros((1, self.obs_dim), dtype='float32'),
                    0.0, False, {})
        def set_sim_noise_scale(self, s):
            self.noise_scale = float(s)
    _iso_cfg = TrainConfig()
    _iso_cfg.episode_length = 12
    _iso_cfg.horizon = 4
    _iso_cfg.wm_input_isolation_len = 4
    _iso_cfg.prbs_seed_segment_steps = 8
    _iso_cfg.clean_steady_seeds = True
    _e = _DummyIsoEnv()
    collect_prbs_episode(_e, _iso_cfg, long_hold=True, isolate_dim=0)
    assert _e.noise_scale == 0.0, _e.noise_scale
    assert _e._schedule == [] and _e._hidden_disturbance is None
    _e2 = _DummyIsoEnv()
    collect_prbs_episode(_e2, _iso_cfg, long_hold=False)
    assert _e2.noise_scale == 1.0, _e2.noise_scale
    _e3 = _DummyIsoEnv()
    _iso_cfg.clean_steady_seeds = False
    _maybe_clean_steady_seed(_e3, _iso_cfg)
    assert _e3.noise_scale == 1.0
    print('[smoke] OK  isolation settle clean_steady_seeds (long_hold zeros noise)')

    # P28 follow-up 9: whole-episode hold at isolated_level, others at 0,
    # no action dither.  ``_st_levels`` must land on the isolated dim.
    _a = _isolated_hold_action(2, isolate_dim=0, hold_level=0.0,
                               isolated_level=0.4)
    assert _np.allclose(_a, [0.4, 0.0]), _a
    _a1 = _isolated_hold_action(1, isolate_dim=0, hold_level=0.25,
                                isolated_level=-0.3)
    assert _np.allclose(_a1, [-0.3]), _a1
    _iso_cfg.clean_steady_seeds = True
    _e4 = _DummyIsoEnv()
    ep = collect_prbs_episode(
        _e4, _iso_cfg, action_std=0.05, long_hold=True,
        isolate_dim=0, hold_level=0.0, isolated_level=0.4)
    acts = _np.stack(ep['act'])
    assert acts.shape == (12, 2), acts.shape
    assert _np.allclose(acts[:, 0], 0.4) and _np.allclose(acts[:, 1], 0.0), acts[:3]
    assert _np.unique(_np.round(acts[:, 0], 6)).size == 1
    class _DvSim:
        cv_indices = [0]
        dv_indices = [1]
        dv_normalization_ranges = [[0.0, 10.0]]
        state_variables = ['cv', 'dv']
    class _DvEnv:
        def __init__(self):
            self.sim = _DvSim()
            self.rng = _np.random.default_rng(0)
    _iso_cfg.dv_prbs_op_frac = 0.8
    _iso_cfg.episode_length = 40
    sched = _build_dv_prbs_schedule(
        _DvEnv(), _iso_cfg, long_hold=True, isolate_dv_idx=0,
        isolated_level=0.5)
    assert len(sched) == 1, sched
    assert sched[0]['start'] == 0 and sched[0]['source'] == 'dv_isolation_settle'
    # Follow-up 10: isolated_level × span/2 (MV-action units).  span=10
    # → delta = 0.5 * 5 = 2.5.  Extra dv_prbs_op_frac must NOT apply.
    assert abs(float(sched[0]['delta']) - 2.5) < 1e-6, sched[0]
    assert abs(_dv_isolation_delta(0.5, 10.0) - 2.5) < 1e-9
    assert abs(_dv_isolation_delta(-0.6, 10.0) + 3.0) < 1e-9
    assert _dv_isolation_delta(0.0, 10.0) == 0.0
    sched0 = _build_dv_prbs_schedule(
        _DvEnv(), _iso_cfg, long_hold=True, isolate_dv_idx=0,
        isolated_level=0.0)
    assert sched0 == []
    sched_prbs = _build_dv_prbs_schedule(_DvEnv(), _iso_cfg, long_hold=False)
    assert len(sched_prbs) > 1
    assert all(e['source'] == 'dv_prbs_seed' for e in sched_prbs)
    # Isolation windows must fit K (test_sim seq_len≥H → unchanged;
    # slow plant seq_len < H → grow to K+1).
    _sl = TrainConfig()
    _sl.seq_len = 64
    _sl.wm_input_isolation_len = 55
    _sl.horizon = 55
    _sl.episode_length = 1220
    assert _isolation_sample_seq_len(_sl) == 64, _isolation_sample_seq_len(_sl)
    _sl.wm_input_isolation_len = 120
    _sl.horizon = 120
    assert _isolation_sample_seq_len(_sl) == 121, _isolation_sample_seq_len(_sl)
    _sl.episode_length = 80
    assert _isolation_sample_seq_len(_sl) == 80, _isolation_sample_seq_len(_sl)
    # Main P1/P2 WM batch must also fit K (follow-up 13).  Isolation-only
    # growth left overshoot/gain-match truncating at seq_len-1.
    _sl.seq_len = 64
    _sl.horizon = 55
    _sl.wm_overshoot_len = 55
    _sl.gain_match_len = 55
    _sl.episode_length = 1220
    assert _wm_train_seq_len(_sl) == 64, _wm_train_seq_len(_sl)
    assert wm_train_seq_len_for_plant(64, 55, 1220) == 64
    _sl.horizon = 120
    _sl.wm_overshoot_len = 120
    _sl.gain_match_len = 120
    assert _wm_train_seq_len(_sl) == 121, _wm_train_seq_len(_sl)
    assert wm_train_seq_len_for_plant(64, 120, 1220) == 121
    assert wm_train_seq_len_for_plant(64, 120, 80) == 80
    print('[smoke] OK  isolation settle whole-episode hold (isolated_level, '
          'action_std=0, DV MV-action-isomorphic, sample len ≥ K+1)')
    print('[smoke] OK  P1/P2 wm_train_T = max(seq_len, K+1) (test_sim unchanged)')

    # Frozen-g skip: isolation extra unroll (follow-up 10) AND the in-graph
    # g-only aux (overshoot / held-rollout / full-BPTT gain-match,
    # follow-up 11) are dead when DOB curriculum freezes the plant model
    # (P2).  Isolation is a separate extra unroll; the aux trio is ~73% of
    # each WM step.  Restore g=True so later smokes stay live.
    assert _dynamics_g_trainable(model)
    _gate_over = float(cfg.wm_overshoot_gate_recon)
    _gate_held = float(cfg.wm_held_rollout_gate_recon)
    cfg.wm_overshoot_gate_recon = 0.0
    cfg.wm_held_rollout_gate_recon = 0.0
    cfg.wm_overshoot_coef = 0.3
    cfg.wm_overshoot_len = min(8, T - 1)
    cfg.wm_held_rollout_coef = 0.5
    cfg.wm_held_rollout_len = 8
    cfg.gain_match_coef = 1.0
    cfg.gain_match_len = 4
    cfg.gain_match_mv_target = ((0.5,) * max(1, action_dim),)
    losses_g, _, _ = world_model_loss(model, batch, cfg)
    if wm_type == 'rssm':
        assert float(losses_g['wm_overshoot_loss']) > 0.0, losses_g['wm_overshoot_loss']
    model.set_world_model_trainable(g=False, dob=True, reward=True)
    assert not _dynamics_g_trainable(model)
    losses_fz, _, _ = world_model_loss(model, batch, cfg)
    assert float(losses_fz['wm_overshoot_loss']) == 0.0, losses_fz['wm_overshoot_loss']
    assert float(losses_fz['wm_held_rollout_loss']) == 0.0, losses_fz['wm_held_rollout_loss']
    assert float(losses_fz['gain_match_loss']) == 0.0, losses_fz['gain_match_loss']
    assert 'gain_match_n' not in losses_fz
    model.set_world_model_trainable(g=True, dob=False, reward=True)
    assert _dynamics_g_trainable(model)
    cfg.wm_overshoot_gate_recon = _gate_over
    cfg.wm_held_rollout_gate_recon = _gate_held
    cfg.gain_match_coef = 0.0
    cfg.gain_match_mv_target = ()
    print('[smoke] OK  isolation skip when g frozen (_dynamics_g_trainable)')
    print('[smoke] OK  g-only aux (overshoot/held/gain-match) skip when g frozen')

    # P1/P2 random collect is numpy-only (no RSSM).  P3 on-policy still
    # streams obs_step.
    if wm_type == 'rssm':
        class _DummyCollectEnv:
            def __init__(self):
                self.action_dim = action_dim
                self.obs_dim = obs_dim
                self.rng = __import__('numpy').random.default_rng(0)
            def reset(self, exploration=True):
                return __import__('numpy').zeros((1, self.obs_dim),
                                                 dtype='float32')
            def step(self, a):
                return (__import__('numpy').zeros((1, self.obs_dim),
                                                  dtype='float32'),
                        0.0, False, {})
        _n_obs = {'n': 0}
        _orig_obs = model.dynamics.obs_step

        def _count_obs(*a, **k):
            _n_obs['n'] += 1
            return _orig_obs(*a, **k)

        model.dynamics.obs_step = _count_obs
        _ccfg = TrainConfig()
        _ccfg.episode_length = 8
        _ccfg.lookback = 4
        _ccfg.k_max = 4
        _ccfg.tau_ctx = 0.1
        _ccfg.world_model_type = 'rssm'
        try:
            collect_episode(_DummyCollectEnv(), model, torch.device('cpu'),
                            _ccfg, random_action=True)
            assert _n_obs['n'] == 0, _n_obs['n']
            collect_episode(_DummyCollectEnv(), model, torch.device('cpu'),
                            _ccfg, random_action=False)
            assert _n_obs['n'] == 8, _n_obs['n']
        finally:
            model.dynamics.obs_step = _orig_obs
        print('[smoke] OK  random collect skips RSSM obs_step; on-policy streams')

    # (mbrl2 p04) critic_batch split (Fix 2) + MC-grounding (Fix 1): pass a
    # DISTINCT replay critic_batch; the critic loss must stay finite and
    # backprop through the diverse-state path, and the MC-grounding term
    # (critic_mc_grounding_coef > 0) must be exercised without NaN.
    assert float(cfg.critic_mc_grounding_coef) > 0.0, 'MC-grounding coef off'
    critic_batch = {
        'obs': torch.randn(B, T, obs_dim),
        'act': torch.rand(B, T, action_dim) * 2 - 1,
        'rew': torch.randn(B, T),
        'cont': torch.ones(B, T),
    }
    diagC = _realsim_actor_critic_step(model, batch, cfg,
                                       critic_batch=critic_batch)
    _finite('_realsim_actor_critic_step[critic_split+MC]', diagC)
    (diagC['actor_loss'] + diagC['critic_loss']).backward()
    print('[smoke] OK  _realsim_actor_critic_step critic_batch split + '
          'MC-grounding backward()')

    # (B) imagination with the long-horizon anchor ENGAGED — exercise the
    # ``lam_anchor`` recursion + raised coef and confirm the critic loss is
    # finite and backprops through the engaged path.
    cfg.critic_anchor_lambda = 0.97
    cfg.critic_anchor_coef_long = 1.0
    diagB = _realsim_actor_critic_step(model, batch, cfg)
    _finite('_realsim_actor_critic_step[anchorB]', diagB)
    diagB['critic_loss'].backward()
    print('[smoke] OK  _realsim_actor_critic_step critic_loss.backward() with anchor B engaged')
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


def _test_cfg_from_env_whitelist() -> None:
    """train.py CLI must honor ENV_OVERRIDES (P28 follow-up 6)."""
    import os
    keys = {
        'DREAMER_AUX_TBPTT_STEPS': '9',
        'DREAMER_SKIP_STORM_RECOVER_P1': '0',
        'DREAMER_ES_GRADSKIP_MAX': '11',
        'DREAMER_N_CRITICS': '3',
        'DREAMER_STEP_TEST_INJECT_N': '7',
        'DREAMER_WM_ISOLATION_DCV_MATCH': '0',
    }
    prev = {k: os.environ.get(k) for k in keys}
    try:
        os.environ.update(keys)
        cfg = _cfg_from_env()
        assert int(cfg.aux_tbptt_steps) == 9, cfg.aux_tbptt_steps
        assert cfg.skip_storm_recover_p1 is False
        assert int(cfg.early_stop_grad_skip_max) == 11
        assert int(cfg.n_critics) == 3
        assert int(cfg.step_test_inject_n) == 7
        assert cfg.wm_isolation_dcv_match is False
        explicit = getattr(cfg, '_explicit_fields', set()) or set()
        assert 'aux_tbptt_steps' in explicit
        assert 'step_test_inject_n' in explicit
        print('[smoke] OK  _cfg_from_env applies ENV_OVERRIDES (aux TBPTT / skip-storm / N)')
    finally:
        for k, old in prev.items():
            if old is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = old


def _test_store_aux_feats_identity() -> None:
    """Isolation encode may drop logit stacks; feats must match the full pass."""
    from models.dreamer_v4_rssm import RSSMConfig, RSSMDynamics, _dob_scan_mix_budget_bytes
    torch.manual_seed(0)
    cfg = RSSMConfig(obs_dim=6, action_dim=2, deter_dim=16,
                     n_categoricals=4, n_classes=4, embed_dim=16,
                     hidden_dim=16, latent_type='deterministic',
                     cont_gain_dim=2, dob_enabled=True, cv_indices=(0,))
    m = RSSMDynamics(cfg).eval()
    B, T = 2, 8
    obs = torch.randn(B, T, 6)
    act = torch.rand(B, T, 2) * 2 - 1
    with torch.no_grad():
        f_full, post, prior, *_ = m.rollout_observed(obs, act, sample=False)
        f_iso, post2, prior2, *_ = m.rollout_observed(
            obs, act, sample=False, store_aux=False)
    assert post is not None and prior is not None
    assert post2 is None and prior2 is None
    err = float((f_full - f_iso).abs().max())
    assert err < 1e-6, f'store_aux=False feats drifted (max_err={err})'
    bud = _dob_scan_mix_budget_bytes()
    assert 4 * 1024 * 1024 <= bud <= 64 * 1024 * 1024
    print(f'[smoke] OK  store_aux=False feats identity (max_err={err:.2e}); '
          f'kalman mix budget={bud}')


def _test_isolation_dcv_scales() -> None:
    """|ΔCV| excitation: Δu ∝ 1/|G| floored at op-band (not a loss reweight)."""
    import numpy as _np
    cfg = TrainConfig()
    cfg.wm_isolation_dcv_match = True
    cfg.gain_match_mv_target = ((-2.80807126652211,),)
    cfg.gain_match_dv_target = ((0.48662864649935383,),)
    mv_sc, dv_sc = _isolation_dcv_scales(cfg, 1, 1, 0.6)
    # P38 match-at-g_min was 0.289; floor 1.0 keeps MV at op-band.
    assert abs(mv_sc[0] - 1.0) < 1e-6, mv_sc
    assert abs(dv_sc[0] - 1.0 / 0.6) < 1e-6, dv_sc
    assert _scale_isolation_level(0.6, dv_sc[0]) == 1.0
    assert abs(_scale_isolation_level(0.6, mv_sc[0]) - 0.6) < 1e-9
    assert abs(_isolation_edge_du(dv_sc[0], 0.6) - 1.0) < 1e-9
    assert abs(_isolation_edge_du(mv_sc[0], 0.6) - 0.6) < 1e-9
    from pathlib import Path as _P
    import training.train as _tr
    _src = _P(_tr.__file__).read_text()
    assert "log_label='pre-iso targets'" in _src
    assert 'keeping pre-iso targets' in _src
    assert 'Isolation settle already tried this (for |ΔCV| excitation scales); skip a second print' not in _src
    cfg.gain_match_mv_target = ((1.0,),)
    cfg.gain_match_dv_target = ((1.0,),)
    assert _isolation_dcv_scales(cfg, 1, 1, 0.6) == ([1.0], [1.0])
    cfg.wm_isolation_dcv_match = False
    cfg.gain_match_mv_target = ((-2.8,),)
    cfg.gain_match_dv_target = ((0.49,),)
    assert _isolation_dcv_scales(cfg, 1, 1, 0.6) == ([1.0], [1.0])
    _stash_isolation_dcv_scales(cfg, [0.289], [1.67], 0.6)
    pay = _isolation_dcv_scale_payload(cfg)
    assert pay['on'] is False and pay['mv'] == [0.289] and pay['dv'] == [1.67]
    assert pay['min_scale'] == 1.0
    assert pay['op_band'] == 0.6
    assert abs(pay['edge_du_mv'][0] - 0.6 * 0.289) < 1e-9
    assert abs(pay['edge_du_dv'][0] - 1.0) < 1e-9
    assert 'isolation_dcv_scales' in _src
    assert '_stash_isolation_dcv_scales' in _src
    assert '_isolation_edge_du' in _src
    assert "'p1_last_ok_iter'" in _src
    assert "row['wm_isolation_loss'] = row['wm_input_isolation_loss']" in _src
    assert 'np.clip(g_min / (g * a0), 1.0, smax)' in _src
    assert _isolation_dcv_scales(cfg, 1, 0, 0.6) == ([1.0], [])
    assert _gain_col_rms(((-2.8, 0.0),))[0] > 1.0
    class _DvSim:
        cv_indices = [0]
        dv_indices = [1]
        dv_normalization_ranges = [[0.0, 10.0]]
        state_variables = ['cv', 'dv']
    class _DvEnv:
        def __init__(self):
            self.sim = _DvSim()
            self.rng = _np.random.default_rng(0)
    iso_cfg = TrainConfig()
    iso_cfg.episode_length = 40
    iso_cfg.dv_prbs_op_frac = 0.8
    scaled = _scale_isolation_level(0.5, 1.0 / 0.6)
    sched = _build_dv_prbs_schedule(
        _DvEnv(), iso_cfg, long_hold=True, isolate_dv_idx=0,
        isolated_level=scaled)
    assert abs(float(sched[0]['delta']) - _dv_isolation_delta(scaled, 10.0)) < 1e-9
    print('[smoke] OK  isolation |ΔCV| dcv_match scales (floor 1.0; DV cube)')


def _test_envfree_observer_recipe() -> None:
    """Env-free TrainConfig must already be the P26 observer / P28 actor stack."""
    c = TrainConfig()
    assert c.rssm_latent_type == 'deterministic', c.rssm_latent_type
    assert c.wm_best_restore_at_p2 is False
    assert int(c.n_critics) == 2
    assert c.return_scale_freeze_after_warmup is True
    assert c.dob_enabled is True
    assert not hasattr(c, 'gain_match_relative')
    from workflow._plant_prepare import ENV_OVERRIDES
    assert 'DREAMER_GAIN_MATCH_RELATIVE' not in ENV_OVERRIDES
    assert 'DREAMER_WM_ISOLATION_VAR_NORM' not in ENV_OVERRIDES
    assert not hasattr(c, 'wm_isolation_var_norm')
    assert c.wm_isolation_dcv_match is True
    assert 'DREAMER_WM_ISOLATION_DCV_MATCH' in ENV_OVERRIDES
    assert c.cont_gain_deterministic_roll is True
    assert _resolve_compile_mode(c) == '', _resolve_compile_mode(c)
    print('[smoke] OK  env-free TrainConfig = P26 observer / P28 actor recipe')


def _test_auto_if_unset_honours_explicit() -> None:
    cfg = TrainConfig()
    assert _auto_if_unset(cfg, 'dob_ground_coef', 2.0) is True
    assert float(cfg.dob_ground_coef) == 2.0
    cfg2 = TrainConfig()
    cfg2.dob_ground_coef = 0.0
    cfg2._explicit_fields = {'dob_ground_coef'}
    assert _auto_if_unset(cfg2, 'dob_ground_coef', 2.0) is False
    assert float(cfg2.dob_ground_coef) == 0.0
    print('[smoke] OK  _auto_if_unset skips explicit 0 (A/B disable)')


def _test_write_resolved_run_plan(tmp_path: str) -> None:
    import json
    from pathlib import Path
    cfg = TrainConfig()
    cfg.out_dir = tmp_path
    cfg.rssm_latent_type = 'deterministic'
    cfg.gain_match_coef = 1.0
    cfg.dob_ground_coef = 2.0
    _stash_isolation_dcv_scales(cfg, [0.289], [1.67], 0.6)
    plan_path = Path(tmp_path) / 'run_plan.json'
    plan_path.write_text(json.dumps({'config': {'rssm_latent_type': 'categorical',
                                                'gain_match_coef': 0.0}}))
    _write_resolved_run_plan(cfg)
    plan = json.loads(plan_path.read_text())
    assert plan['config']['rssm_latent_type'] == 'deterministic'
    assert float(plan['config']['gain_match_coef']) == 1.0
    assert float(plan['config']['dob_ground_coef']) == 2.0
    dcv = plan['isolation_dcv_scales']
    assert dcv['on'] is True
    assert dcv['min_scale'] == 1.0
    assert dcv['mv'] == [0.289] and dcv['dv'] == [1.67]
    assert abs(dcv['edge_du_mv'][0] - 0.6 * 0.289) < 1e-9
    assert abs(dcv['edge_du_dv'][0] - 1.0) < 1e-9
    print('[smoke] OK  _write_resolved_run_plan rewrites run_plan.json config')


def _test_dv_gain_gate(tmp_path: str) -> None:
    """MV-only wm_gain_pass must not hide a biased DV (P29 ×0.56)."""
    import json
    from pathlib import Path
    from evaluation.validate import _dv_gain_gate_from_json, _ss_gain_rel_errs
    p = Path(tmp_path) / 'wm_dv_transfer_matrix.json'
    p.write_text(json.dumps({
        'pairs': {
            'CV0<-DV0': {
                'real_ss_gain': 0.18,
                'wm_ss_gain': 0.101,
                'ss_gain_abs_err': 0.079,
                'ss_gain_ratio_wm_over_real': 0.5616,
            }
        }
    }))
    g = _dv_gain_gate_from_json(p)
    assert g is not None
    assert g['wm_dv_gain_pass'] is True   # rel_err 0.44 < 1.0 (loose 2×)
    assert g['wm_dv_gain_healthy'] is False  # 0.44 > 0.35
    assert abs(g['wm_dv_ss_ratio_worst'] - 0.5616) < 1e-6
    rel = _ss_gain_rel_errs({'a': {'real_ss_gain': 0.32, 'ss_gain_abs_err': 0.031}})
    assert abs(rel[0] - 0.031 / 0.32) < 1e-9
    print('[smoke] OK  DV gain gate fields (MV-only wm_gain_pass unchanged)')


def _test_stage1_dob_ground_skip() -> None:
    """Stage-1 ``d_t≡0``: skip apply_dob clone and ground/reg (no grad)."""
    torch.manual_seed(0)
    cfg = TrainConfig()
    cfg.obs_dim, cfg.action_dim = 6, 1
    cfg.lookback, cfg.seq_len, cfg.horizon = 8, 16, 4
    cfg.mtp_length = 4
    cfg.dob_enabled = True
    cfg.dob_ground_coef = 2.0
    cfg.cv_obs_indices = (0,)
    cfg.compile_mode = 'off'
    cfg.wm_overshoot_coef = 0.0
    cfg.wm_held_rollout_coef = 0.0
    cfg.gain_match_coef = 0.0
    cfg.wm_input_isolation_coef = 0.0
    model = build_model(cfg)
    model.set_dob_active(False)
    B, T = 2, cfg.seq_len
    batch = {
        'obs': torch.randn(B, T, cfg.obs_dim),
        'act': torch.rand(B, T, cfg.action_dim) * 2 - 1,
        'rew': torch.randn(B, T),
        'cont': torch.ones(B, T),
        'expert': torch.zeros(B, T),
        'dist': torch.randn(B, T, 1) * 5,
    }
    losses, _, _ = world_model_loss(model, batch, cfg)
    assert float(losses['dob_ground']) == 0.0, float(losses['dob_ground'])
    recon = torch.randn(B, T, cfg.obs_dim)
    d0 = torch.zeros(B, T, model.dynamics.n_cv)
    out = model.dynamics.apply_dob(recon, d0)
    assert out is recon, 'Stage-1 apply_dob must be identity'
    print('[smoke] OK  Stage-1 dob_ground skipped; apply_dob identity')


if __name__ == '__main__':
    import os
    import sys
    sims = [
        ('test_sim', 'simulation/test_sim/control_setup.json'),
        ('nonlinear_sim', 'simulation/nonlinear_sim/control_setup.json'),
        ('generic', 'simulation/generic/control_setup.json'),
        ('distillation', 'simulation/distillation/control_setup.json'),
        ('softsensor_lab', 'simulation/softsensor_lab/control_setup.json'),
    ]
    only = sys.argv[1] if len(sys.argv) > 1 else None
    _test_cfg_from_env_whitelist()
    _test_store_aux_feats_identity()
    _test_stage1_dob_ground_skip()
    _test_envfree_observer_recipe()
    _test_auto_if_unset_honours_explicit()
    _test_isolation_dcv_scales()
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        _test_write_resolved_run_plan(td)
        _test_dv_gain_gate(td)
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
    print(f'\n[smoke] DONE: {ran} simulator config(s) passed')
