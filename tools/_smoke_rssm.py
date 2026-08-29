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
                            _steady_held_mask, _force_p1_cap_at,
                            _skip_storm_continue_p1,
                            _skip_storm_should_continue_p1,
                            _wm_fidelity_es_suppressed_frozen_g,
                            _p1_fidelity_local_plateau,
                            _resolve_aux_tbptt_steps, _buffer_lap_iters,
                            _resolve_inject_cadence, _cfg_from_env,
                            _resolve_baseline_seed_op_band, _cfg_or_env,
                            _cfg_or_env_float, _resolve_policy_sigma_bounds,
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
                            _should_lock_last_ok,
                            _wm_recon_scalar,
                            _persist_last_ok_ckpt,
                            _auto_if_unset, _isolation_teacher_on,
                            _promote_isolation_aux, _write_resolved_run_plan,
                            _resolve_compile_mode,
                            _clone_module_state, _refresh_module_state,
                            _p1_need_agent_finetune,
                            _smooth_l1_gain_match, _gain_match_fd_held,
                            _gain_match_state_from_feat,
                            _gain_match_held_settle, _auto_gain_match_settle_len,
                            _gain_match_pred_over_tgt,
                            _gain_match_rest_window, _held_rollout_win,
                            collect_rest_lookback,
                            _wm_gain_match_loss, _require_realsim_actor,
                            _adv_action_corr, _format_gain_probe_line,
                            _isolation_seq_is_mv, _snr_build_report,
                            _snr_moving_average,
                            _as_hold_action, _per_mv_hold_rows,
                            _step_test_mv_index, _sample_step_settle_params)


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

    # ---- P2 agent finetune (BC + reward MTP) ----
    _, _, agent_hid2 = world_model_loss(model, batch, cfg)
    af = agent_finetune_loss(model, batch, agent_hid2, cfg)
    _finite('agent_finetune_loss', af)
    af['agent_total'].backward()
    print('[smoke] OK  agent_total.backward()')

    # ---- P3 real-sim actor-critic ----
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
    assert _tc.gain_match_huber_per_input is True
    assert int(_tc.gain_match_settle_len) == -1
    assert _tc.gain_match_rest_ic is True
    assert _tc.p3_reset_log_std is False
    assert int(_tc.aux_tbptt_steps) == 16
    assert not hasattr(_tc, 'gain_match_relative')
    print('[smoke] OK  gain-match defaults (abs Huber, per-input β, rest-IC, TBPTT=16)')

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
    with tempfile.TemporaryDirectory() as _td:
        _path = _P(_td) / 'wm_last_ok.pt'
        _sd = {'w': torch.tensor([1.0, 2.0])}
        assert _persist_last_ok_ckpt(
            _path, _sd, TrainConfig(), {'n': 1}, 66)
        _blob = torch.load(_path, map_location='cpu', weights_only=False)
        assert int(_blob['iter']) == 66
        assert torch.equal(_blob['model']['w'], _sd['w'])
        assert not _persist_last_ok_ckpt(
            _path, None, TrainConfig(), None, 0)
    print('[smoke] OK  persist last-ok ckpt (lock/freeze/storm)')
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

    # P40: silent extra-P1 spike then recovered recon overwrote last-ok.
    assert float(TrainConfig().skip_storm_last_ok_lock_ratio) == 20.0
    assert _should_lock_last_ok(
        recon=0.4816, recon_best=0.0015, lock_ratio=20.0,
        has_last_ok=True, skip_storm_restored=False, already_locked=False)
    # Wrap ~6.7× must not lock (post-wrap snapshots may resume).
    assert not _should_lock_last_ok(
        recon=0.02, recon_best=0.003, lock_ratio=20.0,
        has_last_ok=True, skip_storm_restored=False, already_locked=False)
    assert _should_lock_last_ok(
        recon=0.0068, recon_best=0.0015, lock_ratio=20.0,
        has_last_ok=True, skip_storm_restored=False, already_locked=True)
    assert not _should_lock_last_ok(
        recon=0.4657, recon_best=0.0021, lock_ratio=20.0,
        has_last_ok=True, skip_storm_restored=True, already_locked=True)
    # P41 live (original P1): silent spike 0.0887 vs best 0.0021 (42×)
    # locks; recovered 0.0098 is <5× (overwrite hole) and <20× (must
    # not *start* a lock). Missing recon must not lock.
    assert _should_lock_last_ok(
        recon=0.0887, recon_best=0.0021, lock_ratio=20.0,
        has_last_ok=True, skip_storm_restored=False, already_locked=False)
    assert _recon_still_healthy(0.0098, 0.0021, 5.0)
    assert not _should_lock_last_ok(
        recon=0.0098, recon_best=0.0021, lock_ratio=20.0,
        has_last_ok=True, skip_storm_restored=False, already_locked=False)
    assert not _should_lock_last_ok(
        recon=float('nan'), recon_best=0.0021, lock_ratio=20.0,
        has_last_ok=True, skip_storm_restored=False, already_locked=False)
    # Freeze recon healthy — restore because locked (P40 CAPPED 0.0045).
    assert _should_restore_last_ok_at_p1_freeze(
        recon=0.0045, recon_best=0.0015, ratio=5.0,
        has_last_ok=True, last_ok_locked=True)
    assert not _should_restore_last_ok_at_p1_freeze(
        recon=0.0045, recon_best=0.0015, ratio=5.0,
        has_last_ok=True, last_ok_locked=False)
    print('[smoke] OK  last-ok lock after silent recon spike (P40 RCA)')

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

    # P1/P2 random collect is numpy-only (no RSSM).  P3 on-policy streams
    # stream_serve_step (DV + Kalman when DOB is live).
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
        _n_post = {'n': 0}
        _orig_obs = model.dynamics.obs_step
        _orig_post = model.dynamics._posterior_step

        def _count_obs(*a, **k):
            _n_obs['n'] += 1
            return _orig_obs(*a, **k)

        def _count_post(*a, **k):
            _n_post['n'] += 1
            return _orig_post(*a, **k)

        model.dynamics.obs_step = _count_obs
        model.dynamics._posterior_step = _count_post
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
            assert _n_post['n'] == 0, _n_post['n']
            collect_episode(_DummyCollectEnv(), model, torch.device('cpu'),
                            _ccfg, random_action=False)
            assert _n_obs['n'] == 0, _n_obs['n']
            assert _n_post['n'] == 8, _n_post['n']
        finally:
            model.dynamics.obs_step = _orig_obs
            model.dynamics._posterior_step = _orig_post
        print('[smoke] OK  random collect skips RSSM; on-policy streams stream_serve_step')

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
        'DREAMER_SKIP_STORM_LAST_OK_LOCK_RATIO': '40',
        'DREAMER_ES_GRADSKIP_MAX': '11',
        'DREAMER_N_CRITICS': '3',
        'DREAMER_STEP_TEST_INJECT_N': '7',
        'DREAMER_WM_ISOLATION_DCV_MATCH': '0',
        'DREAMER_GAIN_MATCH_HUBER_PER_INPUT': '0',
        'DREAMER_GAIN_MATCH_SETTLE_LEN': '55',
        'DREAMER_GAIN_MATCH_REST_IC': '1',
        'DREAMER_P3_RESET_LOG_STD': '1',
        'DREAMER_BASELINE_SEED_OP_BAND': '0.4',
        'DREAMER_CONST_ACTION_OP_BAND': '0.55',
        'DREAMER_PRBS_SEED_OP_BAND': '0.8',
        'DREAMER_REWARD_RAW_CLIP_MIN': '-30',
        'DREAMER_REWARD_CAL_PCT': '90',
        'DREAMER_DIAG_PERHEAD_GRADS_EVERY': '10',
        'DREAMER_RUN_WM_DIAGNOSTIC': '0',
        'DREAMER_SEED_TARGET_CV_FRAC': '0.25',
        'DREAMER_PMPO_ENTROPY_ETA_V3': '2e-4',
        'DREAMER_PRBS_SEG_MIN': '6',
        'DREAMER_TRAIN_STEPS_PER_ITER': '50',
        'DREAMER_P3_TRAIN_STEPS_PER_ITER': '4',
        'DREAMER_OBJ_REWARD_SCALE': 'off',
        'DREAMER_ATTN_IMPL': 'manual',
        'DREAMER_SIGMA_MIN_RATIO': '1.4',
        'DREAMER_WM_TF_LEVELS': '7',
        'DREAMER_WM_TF_SPAN': '0.5',
        'DREAMER_VAL_WM_TRANSFER': '0',
        'DREAMER_HORIZON_SETTLE_NTAU': '3.5',
        'DREAMER_HORIZON_MAX': '80',
        'DREAMER_INIT_RANDOMIZATION': '0',
        'DREAMER_INIT_RANDOMIZATION_FRAC': '0.4',
        'DREAMER_WM_OVERHEAD': '1.5',
        'DREAMER_TARGET_UTIL': '0.65',
        'DREAMER_MAX_BS': '64',
        'DREAMER_BATCH_SIZE': '32',
        'DREAMER_DERIVED_OBSERVABLES': '0',
        'DREAMER_DERIVED_OBS_WINDOW': '16',
        'DREAMER_PROCESS_NOISE_AMP_RAMP': '0.1:0.5',
        'DREAMER_HIDDEN_DIST_SETTLE_NTAU': '3.0',
        'DREAMER_HIDDEN_DISTURBANCE': '0',
        'DREAMER_HIDDEN_DIST_P_REVERT': '0.4',
        'DREAMER_HIDDEN_DIST_SHAPE_WEIGHTS': '0.4,0.4,0.2',
    }
    prev = {k: os.environ.get(k) for k in keys}
    try:
        os.environ.update(keys)
        cfg = _cfg_from_env()
        assert int(cfg.aux_tbptt_steps) == 9, cfg.aux_tbptt_steps
        assert cfg.skip_storm_recover_p1 is False
        assert abs(float(cfg.skip_storm_last_ok_lock_ratio) - 40.0) < 1e-12
        assert int(cfg.early_stop_grad_skip_max) == 11
        assert int(cfg.n_critics) == 3
        assert int(cfg.step_test_inject_n) == 7
        assert cfg.wm_isolation_dcv_match is False
        assert cfg.gain_match_huber_per_input is False
        assert int(cfg.gain_match_settle_len) == 55
        assert cfg.gain_match_rest_ic is True
        assert cfg.p3_reset_log_std is True
        assert abs(float(cfg.baseline_seed_op_band) - 0.4) < 1e-12
        assert abs(float(cfg.constant_action_seed_op_band) - 0.55) < 1e-12
        assert abs(float(cfg.prbs_seed_op_band) - 0.8) < 1e-12
        assert abs(float(cfg.reward_raw_clip_min) + 30.0) < 1e-12
        assert abs(float(cfg.reward_cal_pct) - 90.0) < 1e-12
        assert int(cfg.diag_perhead_grads_every) == 10
        assert cfg.run_wm_diagnostic is False
        assert abs(float(cfg.seed_target_cv_frac) - 0.25) < 1e-12
        assert abs(float(cfg.pmpo_entropy_eta_v3) - 2e-4) < 1e-12
        assert int(cfg.prbs_seg_min) == 6
        assert int(cfg.train_steps_per_iter) == 50
        assert int(cfg.phase3_train_steps_per_iter) == 4
        assert cfg.obj_reward_scale == 'off'
        assert cfg.attn_impl == 'manual'
        assert abs(float(cfg.sigma_min_ratio) - 1.4) < 1e-12
        assert int(cfg.wm_tf_levels) == 7
        assert abs(float(cfg.wm_tf_span) - 0.5) < 1e-12
        assert cfg.val_wm_transfer is False
        assert abs(float(cfg.horizon_settle_n_tau) - 3.5) < 1e-12
        assert int(cfg.horizon_max) == 80
        assert cfg.init_randomization is False
        assert abs(float(cfg.init_randomization_frac) - 0.4) < 1e-12
        assert abs(float(cfg.wm_overhead) - 1.5) < 1e-12
        assert abs(float(cfg.gpu_target_util) - 0.65) < 1e-12
        assert int(cfg.gpu_max_bs) == 64
        assert int(cfg.batch_size) == 32
        assert cfg.derived_observables is False
        assert int(cfg.derived_observables_window) == 16
        assert cfg.process_noise_amp_ramp == '0.1:0.5'
        assert abs(float(cfg.hidden_dist_settle_n_tau) - 3.0) < 1e-12
        assert cfg.hidden_disturbance is False
        assert abs(float(cfg.hidden_dist_p_revert) - 0.4) < 1e-12
        assert cfg.hidden_dist_shape_weights == '0.4,0.4,0.2'
        explicit = getattr(cfg, '_explicit_fields', set()) or set()
        assert 'aux_tbptt_steps' in explicit
        assert 'step_test_inject_n' in explicit
        assert 'baseline_seed_op_band' in explicit
        assert 'constant_action_seed_op_band' in explicit
        assert 'prbs_seed_op_band' in explicit
        assert 'reward_raw_clip_min' in explicit
        assert 'reward_cal_pct' in explicit
        assert 'diag_perhead_grads_every' in explicit
        assert 'run_wm_diagnostic' in explicit
        assert 'seed_target_cv_frac' in explicit
        assert 'pmpo_entropy_eta_v3' in explicit
        assert 'prbs_seg_min' in explicit
        assert 'train_steps_per_iter' in explicit
        assert 'phase3_train_steps_per_iter' in explicit
        assert 'obj_reward_scale' in explicit
        assert 'attn_impl' in explicit
        assert 'sigma_min_ratio' in explicit
        assert 'wm_tf_levels' in explicit
        assert 'wm_tf_span' in explicit
        assert 'val_wm_transfer' in explicit
        assert 'horizon_settle_n_tau' in explicit
        assert 'horizon_max' in explicit
        assert 'init_randomization' in explicit
        assert 'init_randomization_frac' in explicit
        assert 'wm_overhead' in explicit
        assert 'gpu_target_util' in explicit
        assert 'gpu_max_bs' in explicit
        assert 'batch_size' in explicit
        assert 'derived_observables' in explicit
        assert 'derived_observables_window' in explicit
        assert 'process_noise_amp_ramp' in explicit
        assert 'hidden_dist_settle_n_tau' in explicit
        assert 'hidden_disturbance' in explicit
        assert 'hidden_dist_p_revert' in explicit
        assert 'hidden_dist_shape_weights' in explicit
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
    with torch.no_grad():
        f_last, _, _, st_last, ds_last, *_ = m.rollout_observed(
            obs, act, sample=False, store_aux=False, last_only=True)
        _, _, _, st_full, ds_full, *_ = m.rollout_observed(
            obs, act, sample=False, store_aux=False)
    assert f_last.shape[1] == 1, f_last.shape
    last_err = float((f_last[:, 0] - f_full[:, -1]).abs().max())
    assert last_err < 1e-6, f'last_only feats != stack[:, -1] (max_err={last_err})'
    h_err = float((st_last.h - st_full.h).abs().max())
    z_err = float((st_last.z - st_full.z).abs().max())
    assert h_err < 1e-6 and z_err < 1e-6, (h_err, z_err)
    if st_last.c_mean is not None:
        c_err = float((st_last.c_mean - st_full.c_mean).abs().max())
        assert c_err < 1e-6, c_err
    if ds_full is not None:
        ds_err = float((ds_last[:, 0] - ds_full[:, -1]).abs().max())
        assert ds_err < 1e-6, ds_err
    m.train()
    m.zero_grad(set_to_none=True)
    f_b, *_ = m.rollout_observed(
        obs, act, sample=False, store_aux=False, last_only=True)
    f_b.sum().backward()
    gru_g = sum(float(p.grad.abs().sum()) for p in m.gru.parameters()
                if p.grad is not None)
    assert gru_g > 0.0, 'last_only observed encode lost GRU gradient'
    m.zero_grad(set_to_none=True)
    _, _, _, st_nf, *_ = m.rollout_observed(
        obs, act, sample=False, store_aux=False, last_only=True,
        return_feats=False)
    nf_h = float((st_nf.h.detach() - st_full.h).abs().max())
    nf_z = float((st_nf.z.detach() - st_full.z).abs().max())
    assert nf_h < 1e-6 and nf_z < 1e-6, (nf_h, nf_z)
    if st_nf.c_mean is not None:
        nf_c = float((st_nf.c_mean.detach() - st_full.c_mean).abs().max())
        assert nf_c < 1e-6, nf_c
    st_nf.h.sum().backward()
    gru_g_nf = sum(float(p.grad.abs().sum()) for p in m.gru.parameters()
                   if p.grad is not None)
    assert gru_g_nf > 0.0, 'return_feats=False last_only lost GRU gradient'
    # Stage-1 rest-IC path (dob_active=False): _posterior_step must match
    # the full obs_step last state (prior heads unused).
    m.dob_active = False
    with torch.no_grad():
        _, _, _, st_full_s1, *_ = m.rollout_observed(
            obs, act, sample=False, store_aux=False)
        _, _, _, st_lo_s1, *_ = m.rollout_observed(
            obs, act, sample=False, store_aux=False, last_only=True,
            return_feats=False)
    s1_h = float((st_lo_s1.h - st_full_s1.h).abs().max())
    s1_z = float((st_lo_s1.z - st_full_s1.z).abs().max())
    assert s1_h < 1e-6 and s1_z < 1e-6, (s1_h, s1_z)
    if st_lo_s1.c_mean is not None:
        s1_c = float((st_lo_s1.c_mean - st_full_s1.c_mean).abs().max())
        assert s1_c < 1e-6, s1_c
    m.zero_grad(set_to_none=True)
    _, _, _, st_s1_b, *_ = m.rollout_observed(
        obs, act, sample=False, store_aux=False, last_only=True,
        return_feats=False)
    st_s1_b.h.sum().backward()
    gru_g_s1 = sum(float(p.grad.abs().sum()) for p in m.gru.parameters()
                   if p.grad is not None)
    assert gru_g_s1 > 0.0, 'Stage-1 last_only posterior-step lost GRU gradient'
    prior_g_s1 = sum(float(p.grad.abs().sum()) for p in m.prior_net.parameters()
                     if p.grad is not None)
    post_g_s1 = sum(float(p.grad.abs().sum()) for p in m.post_net.parameters()
                    if p.grad is not None)
    assert prior_g_s1 == 0.0, f'Stage-1 rest-IC still used prior_net (|g|={prior_g_s1})'
    assert post_g_s1 > 0.0, 'Stage-1 last_only lost post_net gradient'
    if m.cont_dim > 0:
        cprior_g = sum(float(p.grad.abs().sum())
                       for p in m.cont_prior_net.parameters()
                       if p.grad is not None)
        cpost_g = sum(float(p.grad.abs().sum())
                      for p in m.cont_post_net.parameters()
                      if p.grad is not None)
        assert cprior_g == 0.0, f'Stage-1 rest-IC still used cont_prior (|g|={cprior_g})'
        assert cpost_g > 0.0, 'Stage-1 last_only lost cont_post_net gradient'
    m.dob_active = True
    bud = _dob_scan_mix_budget_bytes()
    assert 4 * 1024 * 1024 <= bud <= 64 * 1024 * 1024
    print(f'[smoke] OK  store_aux=False feats identity (max_err={err:.2e}); '
          f'observed last_only ≡ stack[:, -1] (feat={last_err:.2e} '
          f'h={h_err:.2e} z={z_err:.2e}); gru |g|={gru_g:.3f}; '
          f'return_feats=False h={nf_h:.2e} z={nf_z:.2e} gru |g|={gru_g_nf:.3f}; '
          f'Stage-1 last_only ≡ full h={s1_h:.2e} gru |g|={gru_g_s1:.3f}; '
          f'kalman mix budget={bud}')


def _test_img_rollout_last_only() -> None:
    """Gain-match last-step Huber: last_only ≡ stack[:, -1]; GRU still gets grad.

    ``out='h'`` / ``out='obs'`` are identity vs slicing/decoding the feat
    stack (overshoot/held skip the unused F-stack).
    """
    from models.dreamer_v4_rssm import RSSMConfig, RSSMDynamics
    torch.manual_seed(0)
    cfg = RSSMConfig(obs_dim=6, action_dim=2, deter_dim=16,
                     n_categoricals=4, n_classes=4, embed_dim=16,
                     hidden_dim=16, latent_type='deterministic',
                     cont_gain_dim=2)
    m = RSSMDynamics(cfg)
    B, K = 3, 5
    h0 = torch.randn(B, cfg.deter_dim, requires_grad=True)
    z0 = torch.zeros(B, cfg.n_categoricals, cfg.n_classes)
    z0[..., 0] = 1.0
    acts = torch.rand(B, K, cfg.action_dim) * 2 - 1
    roll = m.img_rollout(h0, z0, acts, sample=False)
    last = m.img_rollout(h0, z0, acts, sample=False, last_only=True)
    assert last.shape == roll[:, -1].shape, (last.shape, roll[:, -1].shape)
    err = float((last - roll[:, -1]).detach().abs().max())
    assert err < 1e-6, f'last_only != stack[:, -1] (max_err={err})'
    h_roll = m.img_rollout(h0, z0, acts, sample=False, out='h')
    h_err = float((h_roll - roll[..., :cfg.deter_dim]).detach().abs().max())
    assert h_err < 1e-6, f"out='h' != feat[..., :deter] (max_err={h_err})"
    obs_roll = m.img_rollout(h0, z0, acts, sample=False, out='obs')
    obs_err = float((obs_roll - m.decode(roll)).detach().abs().max())
    assert obs_err < 1e-5, f"out='obs' != decode(feat) (max_err={obs_err})"
    last_h = m.img_rollout(h0, z0, acts, sample=False, last_only=True, out='h')
    last_h_err = float((last_h - h_roll[:, -1]).detach().abs().max())
    assert last_h_err < 1e-6, f"last_only out='h' != stack[:, -1] (max_err={last_h_err})"
    last_obs = m.img_rollout(h0, z0, acts, sample=False, last_only=True, out='obs')
    last_obs_err = float((last_obs - obs_roll[:, -1]).detach().abs().max())
    assert last_obs_err < 1e-5, f"last_only out='obs' != stack[:, -1] (max_err={last_obs_err})"
    m.zero_grad(set_to_none=True)
    last.sum().backward()
    gru_g = sum(float(p.grad.abs().sum()) for p in m.gru.parameters()
                if p.grad is not None)
    assert gru_g > 0.0, 'last_only decode/feat lost GRU gradient'
    print(f'[smoke] OK  img_rollout last_only ≡ stack[:, -1] '
          f'(max_err={err:.2e}); out=h/obs identity '
          f'(h={h_err:.2e} obs={obs_err:.2e} last_h={last_h_err:.2e} '
          f'last_obs={last_obs_err:.2e}); gru |g|={gru_g:.3f}')


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
    _stash_isolation_dcv_scales(cfg, mv_sc, dv_sc, 0.6)
    pay_floor = _isolation_dcv_scale_payload(cfg)
    assert pay_floor['equalize_possible'] is False
    assert abs(pay_floor['g_ratio'] - (2.80807126652211 / 0.48662864649935383)) < 1e-9
    assert abs(pay_floor['smax'] - 1.0 / 0.6) < 1e-9
    cfg.wm_isolation_dcv_min_scale = 0.0
    mv0, dv0 = _isolation_dcv_scales(cfg, 1, 1, 0.6)
    assert abs(mv0[0] - 0.289) < 0.01, mv0
    assert abs(dv0[0] - 1.0 / 0.6) < 1e-6, dv0
    _stash_isolation_dcv_scales(cfg, mv0, dv0, 0.6)
    pay0 = _isolation_dcv_scale_payload(cfg)
    assert pay0['equalize_possible'] is True
    cfg.wm_isolation_dcv_min_scale = 1.0
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
    assert 'equalize_possible' not in pay
    assert abs(pay['edge_du_mv'][0] - 0.6 * 0.289) < 1e-9
    assert abs(pay['edge_du_dv'][0] - 1.0) < 1e-9
    assert 'isolation_dcv_scales' in _src
    assert '_stash_isolation_dcv_scales' in _src
    assert '_isolation_edge_du' in _src
    assert '_record_isolation_dcv_span' in _src
    assert "'p1_last_ok_iter'" in _src
    assert "'p1_last_ok_locked'" in _src
    assert "'p1_recon_best'" in _src
    assert '_persist_last_ok_ckpt' in _src
    assert "wrote {p1_last_ok_ckpt_path.name}" in _src
    assert 'gain_match_mv_loss' in _src
    assert 'gain_match_dv_loss' in _src
    assert 'gain_match_mv_ratio' in _src
    assert 'gain_match_dv_ratio' in _src
    assert '_gain_match_pred_over_tgt' in _src
    assert '_should_lock_last_ok' in _src
    assert "lock={float(getattr(cfg, 'skip_storm_last_ok_lock_ratio'" in _src
    assert "huber_per_in={bool(getattr(cfg, 'gain_match_huber_per_input'" in _src
    assert "gmatch_settle={int(getattr(cfg, 'gain_match_settle_len'" in _src
    assert "gmatch_rest={bool(getattr(cfg, 'gain_match_rest_ic'" in _src
    assert "p3_sigreset={bool(getattr(cfg, 'p3_reset_log_std'" in _src
    assert '_gain_match_held_settle' in _src
    assert '_gain_match_rest_window' in _src
    assert '_gain_match_rest_ic_state' in _src
    assert '_cache_gain_match_rest_ic' in _src
    assert 'reset_policy_exploration' in _src
    assert 'reset_policy_exploration(opt_actor)' in _src
    assert 'stream_serve_step' in _src
    assert '[p3] on-policy collect streams measured DV + Kalman' in _src
    assert "setdefault('jemb_loss'" in _src
    assert '_require_realsim_actor' in _src
    assert 'store_aux=False, last_only=True' in _src
    assert 'return_feats=False' in _src
    assert '_posterior_step' in _src
    assert 'refusing PRBS-posterior fallback' in _src
    assert 'collect_rest_lookback' in _src
    assert '_gain_match_state_from_feat' in _src
    assert '_auto_gain_match_settle_len' in _src
    assert '_adv_action_corr' in _src
    assert '[p1→p2] recon' in _src
    assert '_smooth_l1_gain_match' in _src
    assert '_wm_recon_scalar(wm_losses)' in _src
    assert 'Lock is recon-only (not skip-free)' in _src
    assert "row.setdefault('wm_isolation_loss'" in _src
    assert "out='obs'" in _src
    assert "out='h'" in _src
    assert 'last_only=True' in _src
    assert "last_only=True, out='obs'" in _src
    assert '_p1_fidelity_local_plateau' in _src
    assert 'np.clip(g_min / (g * a0), floor, smax)' in _src
    assert 'wm_isolation_dcv_min_scale' in _src
    assert 'clamp_min(1.0)).detach()' in _src
    assert 'bool(is_mv_bm.any())' not in _src
    assert 'wm_isolation_mv_traj' in _src
    assert 'summary_scope' in _src
    assert 'diag_perhead_grads_every' in _src
    assert 'diag_latent_stability_every' in _src
    assert "getattr(cfg, 'diag_perhead_grads_every'" in _src
    assert "getattr(cfg, 'diag_latent_stability_every'" in _src
    assert "getattr(cfg, 'reward_cal_mode'" in _src
    assert "getattr(cfg, 'run_wm_diagnostic'" in _src
    assert '_cfg_or_env_float' in _src
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
    print('[smoke] OK  isolation |ΔCV| dcv_match scales (floor 1.0; span audit)')


def _test_mimo_hold_rows() -> None:
    """test_sim n_mv=1 stays scalar; MIMO gets independent per-MV holds."""
    import numpy as _np
    rng = _np.random.default_rng(0)
    levels = _np.linspace(-0.6, 0.6, 8, dtype='float32')
    assert _per_mv_hold_rows(levels, 1, 1, rng) is None
    assert _per_mv_hold_rows(levels, 0, 1, rng) is None
    r1 = _np.random.default_rng(7)
    r2 = _np.random.default_rng(7)
    assert _per_mv_hold_rows(levels, 1, 1, r1) is None
    assert float(r1.random()) == float(r2.random())
    rows = _per_mv_hold_rows(levels, 4, 4, rng)
    assert rows is not None and rows.shape == (8, 4)
    for j in range(4):
        assert _np.allclose(_np.sort(rows[:, j]), _np.sort(levels))
    # Not the all-MVs-equal diagonal.
    assert not _np.allclose(rows[:, 0], rows[:, 1])
    a = _as_hold_action(0.5, 3)
    assert a.shape == (3,) and _np.allclose(a, 0.5)
    b = _as_hold_action(_np.array([0.1, -0.2, 0.3], dtype='float32'), 3)
    assert _np.allclose(b, [0.1, -0.2, 0.3])
    cfg = TrainConfig()
    cfg.episode_length = 20
    u1, sw = _sample_step_settle_params(_np.random.default_rng(1), cfg, 0.2)
    assert isinstance(u1, float) and 1 <= sw <= 19
    u1v, swv = _sample_step_settle_params(
        _np.random.default_rng(1), cfg,
        _np.array([0.2, -0.1], dtype='float32'))
    assert getattr(u1v, 'shape', ()) == (2,) and 1 <= swv <= 19
    assert _step_test_mv_index(1, 3) == 0
    assert _step_test_mv_index(4, 2) == 2
    assert _step_test_mv_index(4, -1) == 0
    # MIMO step-test: vector hold + one-MV step; other MVs stay put.
    cur = _as_hold_action(_np.array([0.1, -0.2, 0.3], dtype='float32'), 3)
    j = _step_test_mv_index(cur.size, 1)
    cur[j] = 0.5
    assert _np.allclose(cur, [0.1, 0.5, 0.3])
    print('[smoke] OK  MIMO const-action hold rows (n_mv=1 identity)')


def _test_isolation_seq_is_mv() -> None:
    B, T, A = 4, 8, 1
    act = torch.zeros(B, T, A)
    act[:2] = 0.6
    m = _isolation_seq_is_mv(act)
    assert bool(m[0]) and bool(m[1]) and (not bool(m[2])) and (not bool(m[3]))
    err = torch.tensor([1.0, 3.0, 5.0, 7.0])
    mv_w = m.to(dtype=err.dtype)
    dv_w = 1.0 - mv_w
    mv_m = (err * mv_w).sum() / mv_w.sum().clamp_min(1.0)
    dv_m = (err * dv_w).sum() / dv_w.sum().clamp_min(1.0)
    assert abs(float(mv_m) - 2.0) < 1e-6
    assert abs(float(dv_m) - 6.0) < 1e-6
    print('[smoke] OK  isolation seq MV mask (action energy)')


def _test_snr_measured_scope() -> None:
    import numpy as _np
    T, C = 80, 4
    rng = _np.random.default_rng(0)
    arr = _np.zeros((T, C), dtype='float64')
    arr[:, 0] = _np.linspace(0.0, 1.0, T) + 0.01 * rng.normal(size=T)
    arr[:, 1] = 0.05 * rng.normal(size=T)
    meta = [
        {'name': 'CV', 'kind': 'state', 'role': 'cv'},
        {'name': 'DV', 'kind': 'state', 'role': 'dv'},
        {'name': 'CV0_tgt', 'kind': 'aug_bounds', 'role': 'cv_tgt'},
        {'name': 'CV0_tgt_on', 'kind': 'aug_bounds', 'role': 'cv_tgt'},
    ]
    # cumsum MA ≡ per-channel np.convolve valid
    trend, detail = _snr_moving_average(arr, 8)
    conv = _np.convolve(arr[:, 0], _np.ones(8) / 8.0, mode='valid')
    assert _np.allclose(trend[:, 0], conv)
    assert detail.shape == trend.shape
    r = _snr_build_report(arr, 8, 50.0, 4, meta, [0, 1])
    assert r['summary_scope'] == 'measured_cv_dv'
    assert r['per_channel'][2]['constant'] is True
    assert r['per_channel'][3]['constant'] is True
    assert r['snr_db_min'] > -50.0, r['snr_db_min']
    assert r['constant_n'] == 2
    print('[smoke] OK  SNR summary excludes constant aug channels')


def _test_envfree_observer_recipe() -> None:
    """Env-free TrainConfig must already be the P26 observer / P28 actor stack."""
    c = TrainConfig()
    assert c.rssm_latent_type == 'deterministic', c.rssm_latent_type
    assert c.wm_best_restore_at_p2 is False
    assert int(c.n_critics) == 2
    assert c.return_scale_freeze_after_warmup is True
    assert c.dob_enabled is True
    assert c.gain_match_huber_per_input is True
    assert float(c.gain_match_huber_beta) == 1.0
    assert int(c.gain_match_settle_len) == -1
    assert c.gain_match_rest_ic is True
    assert c.p3_reset_log_std is False
    assert abs(float(c.baseline_seed_op_band) - 0.6) < 1e-12
    assert abs(float(c.constant_action_seed_op_band) - 0.6) < 1e-12
    assert abs(float(c.prbs_seed_op_band) - 0.95) < 1e-12
    assert abs(float(c.reward_raw_clip_min) + 1e6) < 1e-6
    assert abs(float(c.reward_raw_clip_max) - 1e18) < 1e6
    assert abs(float(c.reward_cal_pct) - 95.0) < 1e-12
    assert abs(float(c.reward_cal_target_sym_mag) - 6.0) < 1e-12
    assert abs(float(c.seed_target_cv_frac) - 0.20) < 1e-12
    assert abs(float(c.seed_sigma_cap) - 0.30) < 1e-12
    assert abs(float(c.pmpo_entropy_eta_v3) - 3e-4) < 1e-12
    assert abs(float(c.pmpo_entropy_sigma_ref) - 1.0) < 1e-12
    assert int(c.prbs_seg_min) == 8
    assert int(c.prbs_seg_min_floor) == 2
    assert int(c.train_steps_per_iter) == 100
    assert int(c.phase3_train_steps_per_iter) == 8
    assert int(c.diag_perhead_grads_every) == 0
    assert int(c.diag_latent_stability_every) == 0
    assert c.diag_disable_reward_mtp_in_p1 is False
    assert c.run_wm_diagnostic is True
    assert int(c.wm_diag_n_starts) == 8
    assert int(c.wm_diag_horizon) == 0
    assert c.actor_train_source == 'realsim'
    assert not hasattr(c, 'gain_match_relative')
    from workflow._plant_prepare import ENV_OVERRIDES
    assert 'DREAMER_GAIN_MATCH_SETTLE_LEN' in ENV_OVERRIDES
    assert 'DREAMER_GAIN_MATCH_REST_IC' in ENV_OVERRIDES
    assert 'DREAMER_P3_RESET_LOG_STD' in ENV_OVERRIDES
    assert 'DREAMER_BASELINE_SEED_OP_BAND' in ENV_OVERRIDES
    assert 'DREAMER_CONST_ACTION_OP_BAND' in ENV_OVERRIDES
    assert 'DREAMER_PRBS_SEED_OP_BAND' in ENV_OVERRIDES
    assert 'DREAMER_REWARD_RAW_CLIP_MIN' in ENV_OVERRIDES
    assert 'DREAMER_REWARD_RAW_CLIP_MAX' in ENV_OVERRIDES
    assert 'DREAMER_REWARD_CAL_MODE' in ENV_OVERRIDES
    assert 'DREAMER_REWARD_CAL_TARGET' in ENV_OVERRIDES
    assert 'DREAMER_REWARD_CAL_PCT' in ENV_OVERRIDES
    assert 'DREAMER_REWARD_CAL_PCT_VAL' in ENV_OVERRIDES
    assert 'DREAMER_REWARD_CAL_TARGET_SYM_MAG' in ENV_OVERRIDES
    assert 'DREAMER_DIAG_PERHEAD_GRADS_EVERY' in ENV_OVERRIDES
    assert 'DREAMER_DIAG_LATENT_STABILITY_EVERY' in ENV_OVERRIDES
    assert 'DREAMER_DIAG_DISABLE_REWARD_MTP_IN_P1' in ENV_OVERRIDES
    assert 'DREAMER_DIAG_REWARD_MTP_STOP_GRAD_IN_P1' in ENV_OVERRIDES
    assert 'DREAMER_RUN_WM_DIAGNOSTIC' in ENV_OVERRIDES
    assert 'DREAMER_WM_DIAG_N_STARTS' in ENV_OVERRIDES
    assert 'DREAMER_WM_DIAG_HORIZON' in ENV_OVERRIDES
    assert 'DREAMER_SEED_TARGET_CV_FRAC' in ENV_OVERRIDES
    assert 'DREAMER_SEED_SIGMA_CAP' in ENV_OVERRIDES
    assert 'DREAMER_PMPO_ENTROPY_ETA_V3' in ENV_OVERRIDES
    assert 'DREAMER_PMPO_ENTROPY_SIGMA_REF' in ENV_OVERRIDES
    assert 'DREAMER_PRBS_SEG_MIN' in ENV_OVERRIDES
    assert 'DREAMER_PRBS_SEG_MIN_FLOOR' in ENV_OVERRIDES
    assert 'DREAMER_TRAIN_STEPS_PER_ITER' in ENV_OVERRIDES
    assert 'DREAMER_P3_TRAIN_STEPS_PER_ITER' in ENV_OVERRIDES
    assert 'DREAMER_OBJ_REWARD_SCALE' in ENV_OVERRIDES
    assert 'DREAMER_ATTN_IMPL' in ENV_OVERRIDES
    assert 'DREAMER_FAST_ATTN' in ENV_OVERRIDES
    assert 'DREAMER_SIGMA_MIN_RATIO' in ENV_OVERRIDES
    assert 'DREAMER_WM_TF_LEVELS' in ENV_OVERRIDES
    assert 'DREAMER_WM_TF_SPAN' in ENV_OVERRIDES
    assert 'DREAMER_WM_TF_STEP_FRAC' in ENV_OVERRIDES
    assert 'DREAMER_WM_TF_HORIZON' in ENV_OVERRIDES
    assert 'DREAMER_WM_TF_SETTLE' in ENV_OVERRIDES
    assert 'DREAMER_VAL_WM_TRANSFER' in ENV_OVERRIDES
    assert 'DREAMER_VAL_WM_POSTPRIOR' in ENV_OVERRIDES
    assert 'DREAMER_VAL_WM_DISTPRED' in ENV_OVERRIDES
    assert int(c.wm_tf_levels) == 5
    assert abs(float(c.wm_tf_span) - 0.6) < 1e-12
    assert abs(float(c.wm_tf_step_frac) - 0.4) < 1e-12
    assert int(c.wm_tf_horizon) == 0
    assert int(c.wm_tf_settle) == 0
    assert c.val_wm_transfer is True
    assert c.val_wm_postprior is True
    assert c.val_wm_distpred is True
    assert abs(float(c.horizon_settle_n_tau) - 4.0) < 1e-12
    assert int(c.horizon_max) == 120
    assert c.init_randomization is True
    assert abs(float(c.init_randomization_frac) - 0.6) < 1e-12
    assert abs(float(c.wm_overhead) - 1.30) < 1e-12
    assert abs(float(c.gpu_target_util) - 0.80) < 1e-12
    assert int(c.gpu_max_bs) == 512
    assert 'DREAMER_HORIZON_SETTLE_NTAU' in ENV_OVERRIDES
    assert 'DREAMER_HORIZON_MAX' in ENV_OVERRIDES
    assert 'DREAMER_INIT_RANDOMIZATION' in ENV_OVERRIDES
    assert 'DREAMER_INIT_RANDOMIZATION_FRAC' in ENV_OVERRIDES
    assert 'DREAMER_WM_OVERHEAD' in ENV_OVERRIDES
    assert 'DREAMER_TARGET_UTIL' in ENV_OVERRIDES
    assert 'DREAMER_MAX_BS' in ENV_OVERRIDES
    assert 'DREAMER_BATCH_SIZE' in ENV_OVERRIDES
    assert c.obj_reward_scale == 'auto'
    assert c.attn_impl == 'auto'
    assert abs(float(c.sigma_min_ratio) - 1.2) < 1e-12
    assert 'DREAMER_GAIN_MATCH_RELATIVE' not in ENV_OVERRIDES
    assert 'DREAMER_GAIN_MATCH_HUBER_PER_INPUT' in ENV_OVERRIDES
    assert 'DREAMER_WM_ISOLATION_VAR_NORM' not in ENV_OVERRIDES
    assert not hasattr(c, 'wm_isolation_var_norm')
    assert c.wm_isolation_dcv_match is True
    assert float(c.wm_isolation_dcv_min_scale) == 1.0
    assert 'DREAMER_WM_ISOLATION_DCV_MATCH' in ENV_OVERRIDES
    assert 'DREAMER_WM_ISOLATION_DCV_MIN_SCALE' in ENV_OVERRIDES
    assert 'DREAMER_SKIP_STORM_LAST_OK_LOCK_RATIO' in ENV_OVERRIDES
    assert not hasattr(c, 'rssm_imag_latent_mode')
    assert 'DREAMER_RSSM_IMAG_LATENT_MODE' not in ENV_OVERRIDES
    assert 'DREAMER_CRITIC_IMAG_LOSS_COEF' not in ENV_OVERRIDES
    assert 'DREAMER_CRITIC_REPLAY_ANCHOR_COEF' not in ENV_OVERRIDES
    assert 'DREAMER_CRITIC_ANCHOR_LAMBDA' not in ENV_OVERRIDES
    assert 'DREAMER_CRITIC_ANCHOR_COEF_LONG' not in ENV_OVERRIDES
    assert 'DREAMER_CRITIC_MC_GROUNDING_COEF' in ENV_OVERRIDES
    assert not hasattr(c, 'critic_imag_loss_coef')
    assert not hasattr(c, 'critic_replay_anchor_coef')
    assert not hasattr(c, 'critic_anchor_lambda')
    assert not hasattr(c, 'critic_anchor_coef_long')
    assert not hasattr(c, 'critic_mc_tail_bootstrap')
    assert float(c.critic_mc_grounding_coef) == 2.0
    assert c.cont_gain_deterministic_roll is True
    assert _resolve_compile_mode(c) == '', _resolve_compile_mode(c)
    assert float(c.wm_input_isolation_coef) == 0.0
    assert float(c.wm_ss_match_coef) == 0.0
    assert int(c.wm_isolation_settle_episodes) == 0
    assert _isolation_teacher_on(c) is False
    print('[smoke] OK  env-free TrainConfig = gain-match observer / P28 actor')


def _test_require_realsim_actor() -> None:
    """Imagination actor_train_source is a false A/B (p01 off-policy chatter)."""
    ok = TrainConfig()
    _require_realsim_actor(ok)
    bad = TrainConfig()
    bad.actor_train_source = 'imagination'
    try:
        _require_realsim_actor(bad)
    except RuntimeError as exc:
        assert 'imagination' in str(exc).lower()
        assert '_realsim_actor_critic_step' in str(exc)
    else:
        raise AssertionError('imagination actor_train_source was allowed')
    print('[smoke] OK  non-realsim actor_train_source is refused')


def _test_gain_match_per_input_huber() -> None:
    """P43: per-input β = |tgt| saturates L1 at ±1/N, not P27 1/|tgt|."""
    import torch.nn.functional as F
    pred = torch.tensor([0.0, 0.0], requires_grad=True)
    tgt = torch.tensor([2.62, 0.51])
    loss = _smooth_l1_gain_match(pred, tgt, per_input=True)
    loss.backward()
    n = float(pred.numel())
    assert torch.allclose(
        pred.grad, torch.full_like(pred, -1.0 / n), atol=1e-5), pred.grad
    rel = torch.tensor([-1.0 / (n * 2.62), -1.0 / (n * 0.51)])
    assert float((pred.grad - rel).abs().min()) > 0.1, (
        'per-input Huber looks like relative Huber (P27)')
    a = torch.tensor([0.3, -0.1])
    b = torch.tensor([2.62, 0.51])
    got = _smooth_l1_gain_match(a, b, beta=1.0, per_input=False)
    ref = F.smooth_l1_loss(a, b, beta=1.0)
    assert torch.allclose(got, ref)
    print('[smoke] OK  per-input Huber β=|tgt| (L1 sat ±1/N, not 1/|tgt|)')


def _test_gain_match_pred_over_tgt() -> None:
    """jsonl G_pred/G_tgt: 1.0 on match, 0.75 on P43-class DV miss, sign-safe."""
    mv = torch.tensor([[[-2.624]]])   # (n_in, Bm, n_cv)
    dv = torch.tensor([[[0.51 * 0.75]]])
    assert abs(float(_gain_match_pred_over_tgt(mv, ((-2.624,),))) - 1.0) < 1e-6
    assert abs(float(_gain_match_pred_over_tgt(dv, ((0.51,),))) - 0.75) < 1e-6
    z = _gain_match_pred_over_tgt(mv, ())
    assert float(z) == 0.0
    print('[smoke] OK  gain-match pred/tgt ratio (P43 Huber-blind miss)')


def _test_gain_match_fd_held() -> None:
    """Broadcast FD held actions ≡ clone-loop stack (MIMO + SISO)."""
    torch.manual_seed(0)
    Bm, n_mv, n_dv, step = 4, 2, 1, 1.0
    a_base = torch.randn(Bm, n_mv)
    dv0 = torch.randn(Bm, n_dv)
    got_a, got_dv = _gain_match_fd_held(a_base, dv0, n_mv, n_dv, step)
    a_list = [a_base]
    dv_list = [dv0]
    for j in range(n_mv):
        a_j = a_base.clone()
        a_j[:, j] = a_j[:, j] + step
        a_list.append(a_j)
        dv_list.append(dv0)
    for j in range(n_dv):
        a_list.append(a_base)
        dv_j = dv0.clone()
        dv_j[:, j] = dv_j[:, j] + step
        dv_list.append(dv_j)
    ref_a = torch.stack(a_list, dim=0)
    ref_dv = torch.stack(dv_list, dim=0)
    assert torch.allclose(got_a, ref_a), (got_a - ref_a).abs().max()
    assert torch.allclose(got_dv, ref_dv), (got_dv - ref_dv).abs().max()
    a1, dv1 = _gain_match_fd_held(a_base[:, :1], None, 1, 0, step)
    assert dv1 is None and a1.shape == (2, Bm, 1)
    assert torch.allclose(a1[0], a_base[:, :1])
    assert torch.allclose(a1[1, :, 0], a_base[:, 0] + step)
    print('[smoke] OK  gain-match FD held stack (broadcast ≡ clone-loop)')


def _test_gain_match_held_settle() -> None:
    """P44: held settle unpacks last feat; <2 is identity; grad reaches GRU."""
    from models.dreamer_v4_rssm import RSSMConfig, RSSMDynamics
    torch.manual_seed(0)
    cfg = RSSMConfig(obs_dim=6, action_dim=2, deter_dim=16,
                     n_categoricals=4, n_classes=4, embed_dim=16,
                     hidden_dim=16, latent_type='deterministic',
                     cont_gain_dim=2)
    m = RSSMDynamics(cfg)
    B, S = 3, 4
    h0 = torch.randn(B, cfg.deter_dim, requires_grad=True)
    z0 = torch.zeros(B, cfg.n_categoricals, cfg.n_classes)
    z0[..., 0] = 1.0
    c0 = torch.randn(B, cfg.cont_gain_dim)
    a_base = torch.rand(B, cfg.action_dim) * 2 - 1
    hs, zs, cs = _gain_match_held_settle(m, h0, z0, c0, a_base, None, 0)
    assert hs is h0 and zs is z0 and cs is c0
    hs, zs, cs = _gain_match_held_settle(m, h0, z0, c0, a_base, None, -1)
    assert hs is h0 and zs is z0 and cs is c0
    a_seq = a_base.unsqueeze(1).expand(B, S, cfg.action_dim).contiguous()
    last = m.img_rollout(h0, z0, a_seq, sample=False, c0=c0,
                         last_only=True, out='feat')
    hu, zu, cu = _gain_match_state_from_feat(m, last)
    hs, zs, cs = _gain_match_held_settle(m, h0, z0, c0, a_base, None, S)
    assert float((hs - hu).detach().abs().max()) < 1e-6
    assert float((zs - zu).detach().abs().max()) < 1e-6
    assert float((cs - cu).detach().abs().max()) < 1e-6
    m.zero_grad(set_to_none=True)
    hs.sum().backward()
    gru_g = sum(float(p.grad.abs().sum()) for p in m.gru.parameters()
                if p.grad is not None)
    assert gru_g > 0.0, 'held settle last feat lost GRU gradient'
    c0_auto = TrainConfig()
    c0_auto.gain_match_settle_len = 0
    c0_auto.horizon = 55
    assert _auto_gain_match_settle_len(c0_auto) == 55
    assert int(c0_auto.gain_match_settle_len) == 55
    c_off = TrainConfig()
    c_off.gain_match_settle_len = -1
    c_off.horizon = 55
    assert _auto_gain_match_settle_len(c_off) == -1
    c_set = TrainConfig()
    c_set.gain_match_settle_len = 12
    assert _auto_gain_match_settle_len(c_set) == 12
    from evaluation.wm_transfer_matrix import wm_tf_horizon, wm_tf_roll_len
    assert wm_tf_horizon(55) == 220
    assert wm_tf_horizon(15) == 80
    assert wm_tf_horizon(30) == 120
    _cfg_h = TrainConfig()
    _cfg_h.horizon = 55
    assert wm_tf_roll_len(_cfg_h, 0) == 220
    assert wm_tf_roll_len(_cfg_h, 100) == 100
    print(f'[smoke] OK  gain-match held settle unpack identity; gru |g|={gru_g:.3f}')


def _test_gain_match_rest_window() -> None:
    """P45 collect window is max(H, lookback), not TM 4H."""
    from evaluation.wm_transfer_matrix import wm_tf_horizon
    c = TrainConfig()
    c.horizon = 55
    c.lookback = 128
    s, L = _gain_match_rest_window(c)
    assert (s, L) == (128, 128), (s, L)
    assert s != wm_tf_horizon(55)
    c.lookback = 32
    assert _gain_match_rest_window(c) == (55, 32)
    c.horizon = 15
    c.lookback = 8
    assert _gain_match_rest_window(c) == (15, 8)
    print('[smoke] OK  rest-ic window = max(H, lookback) not wm_tf_horizon')


def _test_held_rollout_win_fits_k() -> None:
    """win=8 is identity at test_sim K=55; clamp (K-1)/4 so fast plants are not 0."""
    assert _held_rollout_win(55, 8) == 8
    assert _held_rollout_win(55, 0) == 8
    assert _held_rollout_win(15, 8) == 3
    assert _held_rollout_win(15, 0) == 2
    assert _held_rollout_win(32, 8) == 7
    # Two windows of win plus settle_frac=0.5 must fit in K (same test as loss).
    for k, w_req in ((55, 8), (15, 8), (15, 0), (32, 8)):
        w = _held_rollout_win(k, w_req)
        s = int(0.5 * k)
        s = max(w, min(s, k - 2 * w))
        assert s >= w and (k - w) > (s + w), (k, w_req, w, s)
    print('[smoke] OK  held-rollout win clamps to (K-1)/4 (test_sim 8)')


def _test_collect_rest_lookback_tm_pairing() -> None:
    """Rest collect records obs AFTER the hold step (TM ``_settle_capture``)."""
    import numpy as _np

    class _Fake:
        action_dim = 1
        t = 0
        _schedule: list = []
        _hidden_disturbance = None
        sim = type('S', (), {})()

        def reset(self, exploration=False):
            self.t = 0
            return _np.zeros((2, 3), dtype='float32')

        def step(self, _a):
            self.t += 1
            o = _np.full((2, 3), float(self.t), dtype='float32')
            return o, 0.0, False, {}

        def set_sim_noise_scale(self, _x):
            pass

    cfg = TrainConfig()
    o, a = collect_rest_lookback(
        _Fake(), cfg, 0.0, settle=5, lookback=3)
    assert o.shape == (3, 3) and a.shape == (3, 1), (o.shape, a.shape)
    # After-step marks 1..5; last L=3 → 3,4,5.  Before-step would be 2,3,4
    # (and would include the reset frame when S==L).
    assert _np.allclose(o[:, 0], [3.0, 4.0, 5.0]), o[:, 0]
    assert _np.allclose(a, 0.0)
    print('[smoke] OK  rest-ic collect is TM post-step pairing')


def _test_gain_match_rest_ic() -> None:
    """Rest cache is the FD IC: changing rest tensors moves the loss."""
    torch.manual_seed(0)
    cfg = TrainConfig()
    cfg.obs_dim = 6
    cfg.action_dim = 1
    cfg.lookback = 8
    cfg.world_model_type = 'rssm'
    cfg.rssm_deter_dim = 32
    cfg.rssm_n_categoricals = 4
    cfg.rssm_n_classes = 4
    cfg.rssm_embed_dim = 16
    cfg.rssm_hidden_dim = 16
    cfg.d_model = 32
    cfg.head_hidden = 32
    cfg.head_n_layers = 1
    cfg.mtp_length = 2
    cfg.horizon = 4
    cfg.seq_len = 16
    cfg.dv_dim = 1
    cfg.dv_indices = (3,)
    cfg.cv_obs_indices = (0,)
    cfg.dob_enabled = False
    cfg.cont_latent_enabled = True
    cfg.cont_gain_dim = 2
    cfg.cont_dist_dim = 0
    cfg.gain_match_coef = 1.0
    cfg.gain_match_len = 4
    cfg.gain_match_settle_len = 8
    cfg.gain_match_rest_ic = True
    cfg.gain_match_mv_target = ((-1.0,),)
    cfg.gain_match_dv_target = ((0.5,),)
    cfg.gain_match_huber_per_input = False
    model = build_model(cfg)
    B, T, O, A = 2, 16, 6, 1
    obs = torch.randn(B, T, O)
    act = torch.rand(B, T, A) * 2 - 1
    with torch.no_grad():
        feats, *_ = model.dynamics.rollout_observed(obs, act, sample=False)
    N, L = 3, 8
    rest_o = torch.randn(N, L, O)
    rest_a = torch.rand(N, L, A) * 2 - 1
    cfg._gain_match_rest_obs = rest_o.numpy()
    cfg._gain_match_rest_act = rest_a.numpy()
    model.zero_grad(set_to_none=True)
    gm1, _ = _wm_gain_match_loss(model, feats.detach(), obs, act, cfg)
    assert torch.isfinite(gm1).all() and float(gm1) > 0.0, float(gm1)
    gm1.backward()
    cont_g = sum(float(p.grad.abs().sum())
                 for n, p in model.dynamics.named_parameters()
                 if p.grad is not None and 'cont' in n)
    gru_g = sum(float(p.grad.abs().sum())
                for n, p in model.dynamics.named_parameters()
                if p.grad is not None and 'gru' in n)
    assert cont_g > 0.0, 'rest-ic FD did not reach cont-gain'
    assert gru_g > 0.0, 'rest-ic encode/FD did not reach GRU'
    cfg._gain_match_rest_obs = (rest_o + 1.5).numpy()
    cfg._gain_match_rest_dev = None
    model.zero_grad(set_to_none=True)
    gm2, _ = _wm_gain_match_loss(model, feats.detach(), obs, act, cfg)
    assert abs(float(gm1) - float(gm2)) > 1e-6, (
        f'rest IC unused (gm1={float(gm1):.6f} gm2={float(gm2):.6f})')
    cfg._gain_match_rest_obs = None
    cfg._gain_match_rest_act = None
    cfg._gain_match_rest_dev = None
    gm3, _ = _wm_gain_match_loss(model, feats.detach(), obs, act, cfg)
    assert torch.isfinite(gm3).all() and float(gm3) > 0.0
    print(f'[smoke] OK  rest-ic encode is the FD IC '
          f'(Δloss={abs(float(gm1) - float(gm2)):.4g})')


def _test_p3_reset_log_std() -> None:
    """P46: zeroing log_std residual restores σ=init, keeps μ, clears Adam rows."""
    from models.dreamer_v4 import ContinuousPolicyHead, PolicyHead
    torch.manual_seed(0)
    pol = ContinuousPolicyHead(
        in_dim=8, hidden_dim=16, action_dim=2, n_layers=2, mtp_length=1,
        init_log_std=-1.5, log_std_min=-2.3, log_std_max=0.0)
    z = torch.zeros(5, 8)
    last = pol.head.net[-1]
    n = int(pol.mtp_length) * int(pol.action_dim)
    idx = torch.arange(n, device=last.weight.device) * 2 + 1
    opt = torch.optim.AdamW(pol.parameters(), lr=1e-3)
    mu_g, ls_g = pol.dist_params(z)
    loss = (ls_g ** 2).mean() + (mu_g ** 2).mean()
    loss.backward()
    opt.step()
    st_w = opt.state[last.weight]
    assert float(st_w['exp_avg'][idx].abs().max()) > 0.0
    mu_idx = torch.arange(n, device=last.weight.device) * 2
    mu_mom0 = float(st_w['exp_avg'][mu_idx].abs().max())
    assert mu_mom0 > 0.0
    with torch.no_grad():
        last.weight.index_fill_(0, idx, -8.0)
        last.bias.index_fill_(0, idx, -8.0)
    mu0, ls0 = pol.dist_params(z)
    assert float(ls0.max()) <= pol.log_std_min + 1e-4, float(ls0.max())
    mu_saved = mu0.detach().clone()
    pol.reset_log_std(opt)
    mu1, ls1 = pol.dist_params(z)
    assert torch.allclose(mu1, mu_saved, atol=1e-6)
    assert torch.allclose(ls1, torch.full_like(ls1, -1.5), atol=1e-5)
    assert float(st_w['exp_avg'][idx].abs().max()) == 0.0
    # μ rows keep their Adam moments
    assert abs(float(st_w['exp_avg'][mu_idx].abs().max()) - mu_mom0) < 1e-12
    disc = PolicyHead(8, 16, 2, n_action_bins=5, n_layers=1, mtp_length=1)
    disc.reset_log_std()
    print('[smoke] OK  P3 reset_log_std restores σ=init, keeps μ, zeros Adam log_std rows')


def _test_resolve_baseline_seed_op_band() -> None:
    """Env-free min(0.6, PRBS); explicit override is not PRBS-capped."""
    c = TrainConfig()
    assert abs(_resolve_baseline_seed_op_band(c, 0.95) - 0.6) < 1e-12
    assert abs(_resolve_baseline_seed_op_band(c, 0.4) - 0.4) < 1e-12
    c.baseline_seed_op_band = 0.8
    c._explicit_fields = {'baseline_seed_op_band'}  # type: ignore[attr-defined]
    assert abs(_resolve_baseline_seed_op_band(c, 0.4) - 0.8) < 1e-12
    print('[smoke] OK  baseline_seed_op_band env-free min(0.6, PRBS); explicit wins')


def _test_cfg_or_env_float_identity() -> None:
    """Reward clip/cal identity: TrainConfig default, explicit, leftover env."""
    import os
    c = TrainConfig()
    v, user = _cfg_or_env_float(c, 'reward_raw_clip_min',
                                'DREAMER_REWARD_RAW_CLIP_MIN', -1e6)
    assert abs(v + 1e6) < 1e-6 and user is False
    c.reward_raw_clip_min = -30.0
    c._explicit_fields = {'reward_raw_clip_min'}  # type: ignore[attr-defined]
    v, user = _cfg_or_env_float(c, 'reward_raw_clip_min',
                                'DREAMER_REWARD_RAW_CLIP_MIN', -1e6)
    assert abs(v + 30.0) < 1e-12 and user is True
    prev = os.environ.get('DREAMER_REWARD_RAW_CLIP_MAX')
    try:
        os.environ['DREAMER_REWARD_RAW_CLIP_MAX'] = '100.0'
        c2 = TrainConfig()
        v, user = _cfg_or_env_float(c2, 'reward_raw_clip_max',
                                    'DREAMER_REWARD_RAW_CLIP_MAX', 1e18)
        assert abs(v - 100.0) < 1e-12 and user is True
    finally:
        if prev is None:
            os.environ.pop('DREAMER_REWARD_RAW_CLIP_MAX', None)
        else:
            os.environ['DREAMER_REWARD_RAW_CLIP_MAX'] = prev
    print('[smoke] OK  reward clip/cal cfg-or-env identity (default / explicit / leftover)')


def _test_auto_tune_formula_input_cfg_or_env() -> None:
    """Seed/PRBS/η formula inputs: TrainConfig default, explicit, leftover env."""
    import os
    c = TrainConfig()
    v, user = _cfg_or_env(c, 'seed_target_cv_frac', 'SEED_TARGET_CV_FRAC', 0.20, float)
    assert abs(v - 0.20) < 1e-12 and user is False
    v, user = _cfg_or_env(c, 'prbs_seg_min', 'PRBS_SEG_MIN', 8, int)
    assert v == 8 and user is False
    c.seed_target_cv_frac = 0.15
    c._explicit_fields = {'seed_target_cv_frac'}  # type: ignore[attr-defined]
    v, user = _cfg_or_env(c, 'seed_target_cv_frac', 'SEED_TARGET_CV_FRAC', 0.20, float)
    assert abs(v - 0.15) < 1e-12 and user is True
    prev = os.environ.get('SEED_SIGMA_CAP')
    try:
        os.environ['SEED_SIGMA_CAP'] = '0.22'
        c2 = TrainConfig()
        v, user = _cfg_or_env(c2, 'seed_sigma_cap', 'SEED_SIGMA_CAP', 0.30, float)
        assert abs(v - 0.22) < 1e-12 and user is True
    finally:
        if prev is None:
            os.environ.pop('SEED_SIGMA_CAP', None)
        else:
            os.environ['SEED_SIGMA_CAP'] = prev
    print('[smoke] OK  auto-tune formula-input cfg-or-env (default / explicit / leftover)')


def _test_policy_sigma_bounds_honours_cfg() -> None:
    """p10 σ_min_ratio=1.2 must not be floored at leftover 1.3."""
    import os
    c = TrainConfig()
    assert abs(float(c.sigma_min_ratio) - 1.2) < 1e-12
    b = _resolve_policy_sigma_bounds(c, 0.219)
    assert abs(b['sigma_min_ratio'] - 1.2) < 1e-12
    assert abs(b['target_sigma_min'] - 0.219 / 1.2) < 1e-9
    c.sigma_min_ratio = 1.2
    c._explicit_fields = {'sigma_min_ratio'}  # type: ignore[attr-defined]
    prev = os.environ.get('SIGMA_MIN_RATIO_OF_MAX')
    try:
        os.environ['SIGMA_MIN_RATIO_OF_MAX'] = '2.5'
        b_exp = _resolve_policy_sigma_bounds(c, 0.219)
        assert abs(b_exp['sigma_min_ratio'] - 1.2) < 1e-12
        c2 = TrainConfig()
        b_left = _resolve_policy_sigma_bounds(c2, 0.219)
        assert abs(b_left['sigma_min_ratio'] - 2.5) < 1e-12
    finally:
        if prev is None:
            os.environ.pop('SIGMA_MIN_RATIO_OF_MAX', None)
        else:
            os.environ['SIGMA_MIN_RATIO_OF_MAX'] = prev
    v, user = _cfg_or_env(TrainConfig(), 'obj_reward_scale',
                          'OBJ_REWARD_SCALE', 'auto', str)
    assert v == 'auto' and user is False
    print('[smoke] OK  policy σ bounds honour TrainConfig 1.2 (leftover still wins)')


def _test_attention_auto_ignores_leftover_fast_attn() -> None:
    """``attn_impl='auto'`` must not re-read leftover ``DREAMER_FAST_ATTN``.

    Whitelist maps FAST_ATTN then ATTN_IMPL onto ``cfg.attn_impl``.
    Constructor ``auto`` is device-only (SDPA on CUDA, manual on CPU).
    Smoke uses ``CUDA_VISIBLE_DEVICES=""`` so leftover FAST_ATTN=1
    used to force SDPA on CPU.
    """
    import os
    from models.dreamer_v4 import CausalAttention
    prev = os.environ.get('DREAMER_FAST_ATTN')
    try:
        os.environ['DREAMER_FAST_ATTN'] = '1'
        blk = CausalAttention(8, 2, attn_impl='auto')
        expected = 'sdpa' if torch.cuda.is_available() else 'manual'
        assert blk.attn_impl == expected, (blk.attn_impl, expected)
    finally:
        if prev is None:
            os.environ.pop('DREAMER_FAST_ATTN', None)
        else:
            os.environ['DREAMER_FAST_ATTN'] = prev
    print('[smoke] OK  CausalAttention auto ignores leftover FAST_ATTN')


def _test_wm_tf_knobs_cfg_or_env() -> None:
    """Eval TM knobs: TrainConfig default, explicit, leftover env."""
    import os
    from evaluation.wm_transfer_matrix import resolve_wm_tf_knobs, val_diag_enabled
    c = TrainConfig()
    k = resolve_wm_tf_knobs(c)
    assert k['n_levels'] == 5
    assert abs(k['span'] - 0.6) < 1e-12
    assert abs(k['step_frac'] - 0.4) < 1e-12
    assert k['horizon'] == 0 and k['settle'] == 0
    assert val_diag_enabled(c, 'val_wm_transfer', 'DREAMER_VAL_WM_TRANSFER') is True
    c.wm_tf_levels = 7
    c._explicit_fields = {'wm_tf_levels'}  # type: ignore[attr-defined]
    assert resolve_wm_tf_knobs(c)['n_levels'] == 7
    prev_span = os.environ.get('DREAMER_WM_TF_SPAN')
    prev_val = os.environ.get('DREAMER_VAL_WM_TRANSFER')
    try:
        os.environ['DREAMER_WM_TF_SPAN'] = '0.5'
        c2 = TrainConfig()
        assert abs(resolve_wm_tf_knobs(c2)['span'] - 0.5) < 1e-12
        os.environ['DREAMER_VAL_WM_TRANSFER'] = '0'
        assert val_diag_enabled(
            TrainConfig(), 'val_wm_transfer', 'DREAMER_VAL_WM_TRANSFER') is False
        c3 = TrainConfig()
        c3.val_wm_transfer = True
        c3._explicit_fields = {'val_wm_transfer'}  # type: ignore[attr-defined]
        assert val_diag_enabled(
            c3, 'val_wm_transfer', 'DREAMER_VAL_WM_TRANSFER') is True
    finally:
        if prev_span is None:
            os.environ.pop('DREAMER_WM_TF_SPAN', None)
        else:
            os.environ['DREAMER_WM_TF_SPAN'] = prev_span
        if prev_val is None:
            os.environ.pop('DREAMER_VAL_WM_TRANSFER', None)
        else:
            os.environ['DREAMER_VAL_WM_TRANSFER'] = prev_val
    print('[smoke] OK  WM TF knobs cfg-or-env identity (default / explicit / leftover)')


def _test_horizon_ic_overhead_cfg_or_env() -> None:
    """Horizon formula + IC DR + WM overhead: TrainConfig default, leftover env."""
    import os
    from utils.auto_episode_length import derive_horizon
    from utils.initial_conditions import _enabled, _frac
    c = TrainConfig()
    assert abs(float(c.horizon_settle_n_tau) - 4.0) < 1e-12
    assert int(c.horizon_max) == 120
    assert c.init_randomization is True
    assert abs(float(c.init_randomization_frac) - 0.6) < 1e-12
    assert abs(float(c.wm_overhead) - 1.30) < 1e-12
    assert abs(float(c.gpu_target_util) - 0.80) < 1e-12
    assert int(c.gpu_max_bs) == 512
    assert _enabled() is True
    assert abs(_frac() - 0.6) < 1e-12
    h, src = derive_horizon(tau=55.0, dead_time=8.0, sample_rate=4)
    assert h == 57, (h, src)
    prev_max = os.environ.get('DREAMER_HORIZON_MAX')
    prev_n = os.environ.get('DREAMER_HORIZON_SETTLE_NTAU')
    prev_ic = os.environ.get('DREAMER_INIT_RANDOMIZATION')
    prev_frac = os.environ.get('DREAMER_INIT_RANDOMIZATION_FRAC')
    try:
        os.environ['DREAMER_HORIZON_MAX'] = '40'
        h2, _ = derive_horizon(tau=55.0, dead_time=8.0, sample_rate=4)
        assert h2 == 40, h2
        os.environ.pop('DREAMER_HORIZON_MAX', None)
        os.environ['DREAMER_HORIZON_SETTLE_NTAU'] = '2.0'
        h3, src3 = derive_horizon(tau=55.0, dead_time=8.0, sample_rate=4)
        assert h3 == 30, (h3, src3)
        os.environ['DREAMER_INIT_RANDOMIZATION'] = '0'
        os.environ['DREAMER_INIT_RANDOMIZATION_FRAC'] = '0.4'
        assert _enabled() is False
        assert abs(_frac() - 0.4) < 1e-12
    finally:
        for key, prev in (
            ('DREAMER_HORIZON_MAX', prev_max),
            ('DREAMER_HORIZON_SETTLE_NTAU', prev_n),
            ('DREAMER_INIT_RANDOMIZATION', prev_ic),
            ('DREAMER_INIT_RANDOMIZATION_FRAC', prev_frac),
        ):
            if prev is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = prev
    print('[smoke] OK  horizon/IC/overhead cfg-or-env identity')


def _test_derived_observables_cfg() -> None:
    """Derived-obs block: TrainConfig default ON, explicit, leftover env."""
    import os
    from utils.derived_observations import (
        derived_observables_enabled, derived_observables_window)
    c = TrainConfig()
    assert c.derived_observables is True
    assert int(c.derived_observables_window) == 0
    assert derived_observables_enabled(c) is True
    auto = derived_observables_window(cfg=c, tau=53.0, sample_rate=4.0)
    assert auto == 26, auto  # round(2*53/4)=26 (ties-to-even), clamp [8,128]
    c.derived_observables = False
    c._explicit_fields = {'derived_observables'}  # type: ignore[attr-defined]
    prev = os.environ.get('DREAMER_DERIVED_OBSERVABLES')
    prev_w = os.environ.get('DREAMER_DERIVED_OBS_WINDOW')
    try:
        os.environ['DREAMER_DERIVED_OBSERVABLES'] = '1'
        assert derived_observables_enabled(c) is False  # explicit beats leftover
        os.environ.pop('DREAMER_DERIVED_OBSERVABLES', None)
        c2 = TrainConfig()
        os.environ['DREAMER_DERIVED_OBSERVABLES'] = '0'
        assert derived_observables_enabled(c2) is False  # leftover env
        os.environ.pop('DREAMER_DERIVED_OBSERVABLES', None)
        os.environ['DREAMER_DERIVED_OBS_WINDOW'] = '16'
        assert derived_observables_window(cfg=TrainConfig(), tau=53.0,
                                          sample_rate=4.0) == 16
        c3 = TrainConfig()
        c3.derived_observables_window = 0
        c3._explicit_fields = {'derived_observables_window'}  # type: ignore
        assert derived_observables_window(cfg=c3, tau=53.0,
                                          sample_rate=4.0) == 26
    finally:
        if prev is None:
            os.environ.pop('DREAMER_DERIVED_OBSERVABLES', None)
        else:
            os.environ['DREAMER_DERIVED_OBSERVABLES'] = prev
        if prev_w is None:
            os.environ.pop('DREAMER_DERIVED_OBS_WINDOW', None)
        else:
            os.environ['DREAMER_DERIVED_OBS_WINDOW'] = prev_w
    print('[smoke] OK  derived_observables TrainConfig + leftover env identity')


def _test_noise_hidden_cfg() -> None:
    """Process-noise ramp + hidden-load: TrainConfig default, leftover, explicit."""
    import os
    from utils.hidden_disturbance import (
        get_phase_disturbance_prob, hidden_disturbance_enabled)
    from utils.noise_config import noise_curriculum_scale
    keys = (
        'DREAMER_PROCESS_NOISE_AMP_RAMP',
        'DREAMER_DISTURBANCE_PROB_WM',
        'DREAMER_HIDDEN_DISTURBANCE',
    )
    prev = {k: os.environ.get(k) for k in keys}
    try:
        for k in keys:
            os.environ.pop(k, None)
        c = TrainConfig()
        assert c.process_noise_amp_ramp == '0.0:0.4'
        assert c.hidden_disturbance is True
        assert abs(float(c.hidden_dist_p_revert) - 0.7) < 1e-12
        assert c.hidden_dist_shape_weights == '0.5,0.3,0.2'
        s0 = noise_curriculum_scale(0.0, phase=1)
        s_cfg = noise_curriculum_scale(0.0, phase=1, cfg=c)
        assert s0 == 0.0 and s_cfg == 0.0
        assert noise_curriculum_scale(0.4, phase=1, cfg=c) == 1.0
        assert noise_curriculum_scale(0.0, phase=3, cfg=c) == 1.0
        assert get_phase_disturbance_prob(phase=1) == get_phase_disturbance_prob(
            phase=1, cfg=c)
        assert hidden_disturbance_enabled(cfg=c) is True
        os.environ['DREAMER_PROCESS_NOISE_AMP_RAMP'] = '0.5:0.4'
        assert abs(noise_curriculum_scale(0.0, phase=1, cfg=c) - 0.5) < 1e-12
        os.environ['DREAMER_PROCESS_NOISE_AMP_RAMP'] = ''
        assert noise_curriculum_scale(0.0, phase=1, cfg=c) == 1.0
        os.environ.pop('DREAMER_PROCESS_NOISE_AMP_RAMP', None)
        c_ex = TrainConfig()
        c_ex.process_noise_amp_ramp = '0.2:0.4'
        c_ex._explicit_fields = {'process_noise_amp_ramp'}  # type: ignore
        os.environ['DREAMER_PROCESS_NOISE_AMP_RAMP'] = '0.9:0.4'
        assert abs(noise_curriculum_scale(0.0, phase=1, cfg=c_ex) - 0.2) < 1e-12
        os.environ.pop('DREAMER_PROCESS_NOISE_AMP_RAMP', None)
        os.environ['DREAMER_DISTURBANCE_PROB_WM'] = '0.25'
        assert abs(get_phase_disturbance_prob(phase=1, cfg=c) - 0.25) < 1e-12
        os.environ.pop('DREAMER_DISTURBANCE_PROB_WM', None)
        os.environ['DREAMER_HIDDEN_DISTURBANCE'] = '0'
        assert hidden_disturbance_enabled(cfg=c) is False
        c_on = TrainConfig()
        c_on.hidden_disturbance = True
        c_on._explicit_fields = {'hidden_disturbance'}  # type: ignore
        assert hidden_disturbance_enabled(cfg=c_on) is True  # explicit beats leftover
    finally:
        for k, old in prev.items():
            if old is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = old
    print('[smoke] OK  process-noise / hidden-load TrainConfig + leftover env identity')


def _test_gpu_calib_cfg() -> None:
    """GPU-calib budget + batch pin: TrainConfig default, leftover OBJ, DREAMER wins."""
    import os
    from workflow._plant_prepare import ENV_OVERRIDES, explicit_batch_size
    c = TrainConfig()
    assert abs(float(c.gpu_target_util) - 0.80) < 1e-12
    assert int(c.gpu_max_bs) == 512
    assert 'DREAMER_TARGET_UTIL' in ENV_OVERRIDES
    assert 'DREAMER_MAX_BS' in ENV_OVERRIDES
    assert 'DREAMER_BATCH_SIZE' in ENV_OVERRIDES
    keys = ('DREAMER_BATCH_SIZE', 'OBJ_BATCH_SIZE')
    prev = {k: os.environ.get(k) for k in keys}
    try:
        for k in keys:
            os.environ.pop(k, None)
        assert explicit_batch_size() is None
        os.environ['OBJ_BATCH_SIZE'] = '24'
        assert explicit_batch_size() == 24
        os.environ['DREAMER_BATCH_SIZE'] = '48'
        assert explicit_batch_size() == 48
    finally:
        for k, old in prev.items():
            if old is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = old
    print('[smoke] OK  GPU-calib TrainConfig + DREAMER_BATCH_SIZE beats leftover OBJ')


def _test_adv_action_corr_vectorized() -> None:
    """P3 diag: batched |corr(adv, a_i)| ≡ the old per-channel loop."""
    torch.manual_seed(0)
    adv = torch.randn(8, 16)
    act = torch.randn(8 * 16, 3)
    got = _adv_action_corr(adv, act)
    adv_c = (adv.float() - adv.float().mean()).reshape(-1)
    refs = []
    for i in range(act.shape[-1]):
        a = act[:, i].float()
        a_c = a - a.mean()
        den = (adv_c.norm() * a_c.norm()).clamp_min(1e-8)
        refs.append(((adv_c * a_c).sum() / den).abs())
    ref = torch.stack(refs).mean()
    assert torch.allclose(got, ref, atol=1e-6), (got, ref)
    z = _adv_action_corr(adv, act[:, :0])
    assert float(z) == 0.0
    print('[smoke] OK  vectorized adv-action corr (P3 diag identity)')


def _test_format_gain_probe_line() -> None:
    """Gate line must print ss AND @H (already in the probe; was dropped)."""
    line = _format_gain_probe_line({
        'gain_ready': False,
        'r_min': 0.75, 'r_max': 1.04,
        'worst_ratio': 0.75, 'worst_input': 'DV CV0<-DV0',
        'band': [0.8, 1.3], 'noise_worst': 0.5, 'sign_flips': 0,
        'n_checks': 2, 'unbiased': False, 'not_noisy': True,
        'atH_min': 0.78, 'atH_max': 1.01,
        'ss_pairs': [('MV CV0<-MV0', 1.04), ('DV CV0<-DV0', 0.75)],
        'ath_pairs': [('MV CV0<-MV0', 1.01), ('DV CV0<-DV0', 0.78)],
    })
    assert 'DCgain_ratio[0.75,1.04]' in line
    assert '@H[0.78,1.01]' in line
    assert 'MV CV0<-MV0=1.04/@H=1.01' in line
    assert 'DV CV0<-DV0=0.75/@H=0.78' in line
    print('[smoke] OK  gain-probe line prints ss and @H')


def _test_p1_fidelity_local_plateau() -> None:
    """P40: warmup spike must not block the recent-floor gate."""
    # Too few probes.
    ok, flat, rmax, band = _p1_fidelity_local_plateau(
        [(10, 6.541)], n_probes=3, plateau_frac=0.05, ema_min=1.5)
    assert ok is False and flat is False

    # P40-like: all-time 6.541 at iter 10; late-P1 recovered ~5.2.
    hist = [(10, 6.541), (20, 6.06), (30, 5.50), (40, 5.20),
            (50, 4.80), (60, 4.40), (70, 5.00), (80, 5.20)]
    ok, _flat, rmax, _band = _p1_fidelity_local_plateau(
        hist, n_probes=3, plateau_frac=0.05, ema_min=1.5)
    assert ok is True, (ok, rmax)
    assert rmax == 5.20
    # Gate uses recent_ok, so original-budget P1 would run the GAIN probe
    # instead of EXTEND-for-not_plateaued.

    low = [(10, 1.2), (20, 1.1), (30, 1.0)]
    ok2, _, r2, _ = _p1_fidelity_local_plateau(
        low, n_probes=3, plateau_frac=0.05, ema_min=1.5)
    assert ok2 is False and r2 == 1.2
    print('[smoke] OK  P1 fidelity gate is recent-floor (not warmup spike)')


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


def _test_promote_isolation_aux() -> None:
    """Env-free isolation stays off; opt-in fills len/settle."""
    c = TrainConfig()
    c.horizon = 55
    assert _promote_isolation_aux(c) is False
    assert float(c.wm_input_isolation_coef) == 0.0
    assert float(c.wm_ss_match_coef) == 0.0
    assert int(c.wm_input_isolation_len) == 0
    assert int(c.wm_isolation_settle_episodes) == 0
    c2 = TrainConfig()
    c2.horizon = 55
    c2.wm_input_isolation_coef = 1.0
    assert _promote_isolation_aux(c2) is True
    assert int(c2.wm_input_isolation_len) == 55
    assert int(c2.wm_isolation_settle_episodes) == 24
    c3 = TrainConfig()
    c3.wm_ss_match_coef = 3.0
    assert _promote_isolation_aux(c3) is True
    assert abs(float(c3.wm_ss_match_settle_var) - 0.05) < 1e-12
    assert int(c3.wm_isolation_settle_episodes) == 24
    print('[smoke] OK  isolation aux off env-free; opt-in fills len/settle')


def _test_write_resolved_run_plan(tmp_path: str) -> None:
    import contextlib
    import io
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
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        _write_resolved_run_plan(cfg)
    banner = buf.getvalue()
    assert 'lock=20' in banner, banner
    assert 'iso_dcv=off' in banner, banner
    assert 'huber_per_in=True' in banner, banner
    assert 'gmatch_settle=-1' in banner, banner
    assert 'gmatch_rest=True' in banner, banner
    assert 'p3_sigreset=False' in banner, banner
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
    from evaluation.validate import (
        _dv_gain_gate_from_json, _ss_gain_rel_errs, _merge_observer_gain_gate,
        _gain_status,
    )
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
    mv = {
        'wm_gain_pass': True,
        'wm_gain_healthy': True,
    }
    merged = _merge_observer_gain_gate(dict(mv), g)
    assert merged['wm_gain_pass'] is True
    assert merged['wm_observer_gain_pass'] is True
    assert merged['wm_observer_gain_healthy'] is False
    assert _gain_status(True, True) == 'HEALTHY'
    assert _gain_status(False, True) == 'PASS'
    assert _gain_status(False, False) == 'FAIL'
    no_dv = _merge_observer_gain_gate(dict(mv), None)
    assert no_dv['wm_observer_gain_healthy'] is True
    print('[smoke] OK  DV gain gate + observer-wide AND (MV-only wm_gain_pass kept)')


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


def _test_stream_serve_matches_rollout() -> None:
    """P3 serve feat matches training re-encode (measured DV + Kalman d_t)."""
    from models.dreamer_v4_rssm import stream_serve_step
    torch.manual_seed(0)
    cfg = TrainConfig()
    cfg.obs_dim, cfg.action_dim = 6, 1
    cfg.lookback, cfg.seq_len, cfg.horizon = 8, 8, 4
    cfg.mtp_length = 4
    cfg.rssm_deter_dim = 32
    cfg.rssm_n_categoricals = 4
    cfg.rssm_n_classes = 4
    cfg.rssm_embed_dim = 16
    cfg.rssm_hidden_dim = 16
    cfg.head_hidden = 16
    cfg.dob_enabled = True
    cfg.cv_obs_indices = (0,)
    cfg.dv_as_input = True
    cfg.dv_indices = (3,)
    cfg.dv_dim = 1
    cfg.dv_feedforward = True
    cfg.compile_mode = 'off'
    cfg.wm_overshoot_coef = 0.0
    cfg.wm_held_rollout_coef = 0.0
    cfg.gain_match_coef = 0.0
    cfg.wm_input_isolation_coef = 0.0
    model = build_model(cfg)
    model.set_dob_active(True)
    rssm = model.dynamics
    assert rssm.dob_enabled and rssm.dv_dim == 1, (rssm.dob_enabled, rssm.dv_dim)
    B, T = 2, 8
    obs = torch.randn(B, T, cfg.obs_dim)
    act = torch.rand(B, T, cfg.action_dim) * 2 - 1
    feats, *_ = rssm.rollout_observed(obs, act, sample=False, store_aux=False)
    state = rssm.initial_state(B, obs.device)
    streamed = []
    for t in range(T):
        state = stream_serve_step(
            rssm, state, act[:, t], obs[:, t], sample=False)
        streamed.append(state.feat)
    streamed = torch.stack(streamed, dim=1)
    assert streamed.shape == feats.shape, (streamed.shape, feats.shape)
    # GRU does not take d; Kalman on feat must match batched dob_kalman_scan.
    if not torch.allclose(streamed, feats, atol=1e-5, rtol=1e-4):
        err = (streamed - feats).abs().max().item()
        raise AssertionError(f'serve vs rollout_observed max|Δ|={err:.4e}')
    # Collect with DOB+DV must stream obs_step (Kalman needs the prior decode).
    class _Dummy:
        def __init__(self):
            self.action_dim = cfg.action_dim
            self.obs_dim = cfg.obs_dim
            self.rng = __import__('numpy').random.default_rng(0)
        def reset(self, exploration=True):
            return __import__('numpy').zeros((1, self.obs_dim), dtype='float32')
        def step(self, a):
            o = __import__('numpy').zeros((1, self.obs_dim), dtype='float32')
            o[0, 3] = 0.4
            return o, 0.0, False, {}
    _n_obs = {'n': 0}
    _orig = rssm.obs_step

    def _count(*a, **k):
        _n_obs['n'] += 1
        return _orig(*a, **k)

    rssm.obs_step = _count
    _ccfg = TrainConfig()
    _ccfg.episode_length = 5
    _ccfg.lookback = 4
    _ccfg.k_max = 4
    _ccfg.tau_ctx = 0.1
    _ccfg.world_model_type = 'rssm'
    try:
        collect_episode(_Dummy(), model, torch.device('cpu'),
                        _ccfg, random_action=False)
        assert _n_obs['n'] == 5, _n_obs['n']
    finally:
        rssm.obs_step = _orig
    print('[smoke] OK  stream_serve_step ≡ rollout_observed (DV+Kalman); '
          'P3 collect uses obs_step')


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
    _test_img_rollout_last_only()
    _test_stage1_dob_ground_skip()
    _test_stream_serve_matches_rollout()
    _test_envfree_observer_recipe()
    _test_require_realsim_actor()
    _test_gain_match_per_input_huber()
    _test_gain_match_pred_over_tgt()
    _test_gain_match_fd_held()
    _test_gain_match_held_settle()
    _test_gain_match_rest_window()
    _test_held_rollout_win_fits_k()
    _test_collect_rest_lookback_tm_pairing()
    _test_gain_match_rest_ic()
    _test_p3_reset_log_std()
    _test_resolve_baseline_seed_op_band()
    _test_cfg_or_env_float_identity()
    _test_auto_tune_formula_input_cfg_or_env()
    _test_policy_sigma_bounds_honours_cfg()
    _test_attention_auto_ignores_leftover_fast_attn()
    _test_wm_tf_knobs_cfg_or_env()
    _test_horizon_ic_overhead_cfg_or_env()
    _test_derived_observables_cfg()
    _test_noise_hidden_cfg()
    _test_gpu_calib_cfg()
    _test_adv_action_corr_vectorized()
    _test_format_gain_probe_line()
    _test_p1_fidelity_local_plateau()
    _test_auto_if_unset_honours_explicit()
    _test_promote_isolation_aux()
    _test_isolation_dcv_scales()
    _test_mimo_hold_rows()
    _test_isolation_seq_is_mv()
    _test_snr_measured_scope()
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
