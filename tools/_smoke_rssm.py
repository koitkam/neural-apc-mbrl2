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
    TrainConfig, APCEnv, build_model, world_model_loss,
                            agent_finetune_loss, _realsim_actor_critic_step,
                            expert_bc_p3_loss, _adaptive_return_cap,
                            _steady_held_mask, _force_p1_cap_at,
                            _skip_storm_continue_p1,
                            _skip_storm_should_continue_p1,
                            _wm_fidelity_es_suppressed_frozen_g,
                            _p1_fidelity_local_plateau,
                            _resolve_aux_tbptt_steps, _resolve_gain_match_step,
                            _buffer_lap_iters,
                            _resolve_inject_cadence, _cfg_from_env,
                            _CLI_ONLY_ENV, _recon_channel_weights,
                            _cached_arange_1k, _cached_strided_arange,
                            _cached_time_gather_idx, _overshoot_tail_wk,
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
                            _should_probe_gain_on_last_ok,
                            _wm_recon_scalar,
                            _persist_last_ok_ckpt,
                            _auto_if_unset, _isolation_teacher_on,
                            _promote_isolation_aux, _write_resolved_run_plan,
                            _resolve_compile_mode,
                            _clone_module_state, _refresh_module_state,
                            _load_module_state,
                            _p1_need_agent_finetune,
                            _wm_need_logged_aux,
                            _wm_need_dist_target, _p1_wm_h2d_keys,
                            _replay_h2d_keys,
                            _smooth_l1_gain_match, _gain_match_fd_held,
                            _gain_match_fd_action_seq,
                            _gain_match_realized_du, _cube_step_held,
                            _cube_plus_would_clip, _gain_match_clip_frac_t,
                            _gain_match_state_from_feat,
                            _gain_match_held_settle, _auto_gain_match_settle_len,
                            _gain_match_pred_over_tgt, _gain_match_tgt_tensor,
                            _gain_match_rest_window, _gain_match_rest_ic_state,
                            _held_rollout_win,
                            _wm_held_rollout_stationarity_loss,
                            _rest_ic_can_cuda_graph,
                            _RestICGraphModule, _rest_ic_last_tensors,
                            _rest_ic_note_capture_miss,
                            _warmup_rest_ic_cuda_graph,
                            _release_rest_ic_cuda_graph,
                            _arm_rest_ic_stream_mismatch_warn,
                            _amp_parent_autocast_on,
                            _rssm_param_grad_snapshot,
                            _rssm_param_grad_restore,
                            collect_rest_lookback,
                            _wm_gain_match_loss, _require_realsim_actor,
                            _adv_action_corr, _row_adv_action_corr,
                            _save_training_diagnostics_plot,
                            _p3_logp_clip_bound,
                            _gaussian_entropy_nats, _entropy_collapse_threshold,
                            _p3_mu_ratio_surrogate, _p3_frozen_unfreeze_policy,
                            _format_gain_probe_line,
                            _isolation_seq_is_mv, _snr_build_report,
                            _snr_moving_average,
                            _as_hold_action, _per_mv_hold_rows,
                            _step_test_mv_index, _sample_step_settle_params,
                            _runtime_setpoint_config_from_cfg)


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
    assert 'critic_mc_loss' in diag, sorted(diag)
    assert 'critic_pred_target_r' in diag, sorted(diag)
    assert 'critic_target_v_r' in diag, sorted(diag)
    assert 'actor_pos_adv_frac' in diag, sorted(diag)
    assert 'adv_action_corr' in diag, sorted(diag)
    assert 'pmpo_pos_frac' not in diag, sorted(diag)
    assert 'imag_adv_action_corr' not in diag, sorted(diag)
    assert 'imagined_return_mean' not in diag, sorted(diag)
    assert torch.isfinite(diag['critic_mc_loss']).all()
    assert float(diag['critic_mc_loss']) >= 0.0
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
    assert _wm_need_logged_aux(False, 0, 100) is False
    assert _wm_need_logged_aux(True, 98, 100) is False
    assert _wm_need_logged_aux(True, 99, 100) is True
    print('[smoke] OK  P1 MTP skip when reward_scale_loss_p1=0 (log last only)')
    assert _p1_wm_h2d_keys(False) == ('obs', 'act')
    assert _p1_wm_h2d_keys(True) == ('obs', 'act', 'dist')
    print('[smoke] OK  P1 H2D keys skip dist when WM does not read it')

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
    assert abs(float(_tc.gain_match_step) - 0.0) < 1e-12
    assert _tc.gain_match_clip_realized is True
    assert _tc.gain_match_rest_ic is True
    assert int(_tc.gain_match_rest_ic_len) == 0
    assert _tc.gain_match_rest_ic_cuda_graph is True
    assert _tc.p3_reset_log_std is False
    assert _tc.bc_mean_only is True
    assert _tc.p3_stop_grad_log_std is True
    assert abs(float(_tc.p3_logp_clip) - 8.0) < 1e-12
    assert abs(float(_tc.p3_mu_ratio_clip) - 0.2) < 1e-12
    assert int(_tc.p3_mu_ratio_refresh_iters) == 0
    assert abs(float(_tc.early_stop_entropy_collapse_floor_frac) - 0.25) < 1e-12
    assert int(_tc.aux_tbptt_steps) == 16
    assert not hasattr(_tc, 'gain_match_relative')
    assert not hasattr(_tc, 'actor_kl_coef')
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

    _gs = TrainConfig()
    assert abs(_resolve_gain_match_step(_gs) - 0.4) < 1e-12, _gs.gain_match_step
    assert abs(float(_gs.gain_match_step) - 0.4) < 1e-12
    _gs1 = TrainConfig()
    _gs1.gain_match_step = 1.0
    assert abs(_resolve_gain_match_step(_gs1) - 1.0) < 1e-12
    _gs_tf = TrainConfig()
    _gs_tf.wm_tf_step_frac = 0.25
    assert abs(_resolve_gain_match_step(_gs_tf) - 0.25) < 1e-12
    _src = open('training/train.py').read()
    assert 'gain_match_step: float = 0.0' in _src
    assert 'def _resolve_gain_match_step' in _src
    assert 'gain_match_clip_realized: bool = True' in _src
    print('[smoke] OK  gain_match_step sentinel 0 auto=wm_tf_step_frac; explicit 1.0 kept')

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
    # P40 recovered extra-P1 basin stays locked (freeze must restore 83).
    assert _should_lock_last_ok(
        recon=0.0068, recon_best=0.0015, lock_ratio=20.0,
        has_last_ok=True, skip_storm_restored=False, already_locked=True,
        extra_p1=True)
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
    # P48: original-P1 wrap 43× locks; recovered 2.8× unlocks so last-ok
    # can advance (freeze must not restore the wrap-era snapshot).
    assert _should_lock_last_ok(
        recon=0.1447, recon_best=0.0033, lock_ratio=20.0,
        has_last_ok=True, skip_storm_restored=False, already_locked=False,
        extra_p1=False)
    assert not _should_lock_last_ok(
        recon=0.0093, recon_best=0.0033, lock_ratio=20.0,
        has_last_ok=True, skip_storm_restored=False, already_locked=True,
        extra_p1=False)
    # Still in the wrap (recon remains >20×) → stay locked.
    assert _should_lock_last_ok(
        recon=0.1447, recon_best=0.0033, lock_ratio=20.0,
        has_last_ok=True, skip_storm_restored=False, already_locked=True,
        extra_p1=False)
    # Freeze recon healthy — restore because locked (P40 CAPPED 0.0045).
    assert _should_restore_last_ok_at_p1_freeze(
        recon=0.0045, recon_best=0.0015, ratio=5.0,
        has_last_ok=True, last_ok_locked=True)
    assert not _should_restore_last_ok_at_p1_freeze(
        recon=0.0045, recon_best=0.0015, ratio=5.0,
        has_last_ok=True, last_ok_locked=False)
    # P50: wrap-adjacent recon 19× best (under 20× lock) must probe
    # last-ok, not live wrap-damaged g. Healthy recon stays live.
    assert _should_probe_gain_on_last_ok(
        recon=0.0283, recon_best=0.0015, ratio=5.0, has_last_ok=True)
    assert _should_probe_gain_on_last_ok(
        recon=0.1069, recon_best=0.0015, ratio=5.0, has_last_ok=True)
    assert not _should_probe_gain_on_last_ok(
        recon=0.0041, recon_best=0.0015, ratio=5.0, has_last_ok=True)
    assert not _should_probe_gain_on_last_ok(
        recon=0.0283, recon_best=0.0015, ratio=5.0, has_last_ok=False)
    print('[smoke] OK  last-ok lock after silent recon spike (P40 RCA)')
    print('[smoke] OK  P1 gain-probe uses last-ok when recon detonated (P50)')

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
    _n_roll = {'n': 0}
    _orig_roll = model.dynamics.rollout_observed

    def _count_roll(*a, **k):
        _n_roll['n'] += 1
        return _orig_roll(*a, **k)

    model.dynamics.rollout_observed = _count_roll
    try:
        diagC = _realsim_actor_critic_step(model, batch, cfg,
                                           critic_batch=critic_batch)
        assert _n_roll['n'] == 1, (
            f'fused observer encode expected 1 rollout, got {_n_roll["n"]}')
    finally:
        model.dynamics.rollout_observed = _orig_roll
    _finite('_realsim_actor_critic_step[critic_split+MC]', diagC)
    assert float(diagC['critic_mc_loss']) >= 0.0
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
        'DREAMER_GAIN_MATCH_CLIP_REALIZED': '0',
        'DREAMER_GAIN_MATCH_SETTLE_LEN': '55',
        'DREAMER_GAIN_MATCH_REST_IC': '1',
        'DREAMER_GAIN_MATCH_REST_IC_CUDA_GRAPH': '0',
        'DREAMER_GAIN_MATCH_REST_IC_LEN': '16',
        'DREAMER_P3_RESET_LOG_STD': '1',
        'DREAMER_P3_STOP_GRAD_LOG_STD': '0',
        'DREAMER_P3_LOGP_CLIP': '0',
        'DREAMER_P3_MU_RATIO_CLIP': '0',
        'DREAMER_P3_MU_RATIO_REFRESH': '0',
        'DREAMER_ES_ENT_FLOOR_FRAC': '0.1',
        'DREAMER_BC_MEAN_ONLY': '0',
        'DREAMER_BASELINE_SEED_OP_BAND': '0.4',
        'DREAMER_CONST_ACTION_OP_BAND': '0.55',
        'DREAMER_PRBS_SEED_OP_BAND': '0.8',
        'DREAMER_REWARD_RAW_CLIP_MIN': '-30',
        'DREAMER_REWARD_CAL_PCT': '90',
        'DREAMER_DIAG_PERHEAD_GRADS_EVERY': '10',
        'DREAMER_RUN_WM_DIAGNOSTIC': '0',
        'DREAMER_WM_DIAG_DEVICE': 'cpu',
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
        'DREAMER_EPISODE_SETTLE_MULTIPLE': '10',
        'DREAMER_EPISODE_MIN_LENGTH': '700',
        'DREAMER_EPISODE_MAX_LENGTH': '2000',
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
        'DREAMER_SIM_NOISE_ADAPTIVE': '0',
        'DREAMER_SIM_OU_SIGMA_FRAC': '0.01',
        'DREAMER_SIM_OU_GAIN_CV': '0.2',
        'DREAMER_SIM_OU_GAIN_DV': '0.4',
        'DREAMER_SIM_MEAS_NOISE_CV_FRAC': '0.006',
        'DREAMER_SIM_MEAS_NOISE_DV_FRAC': '0.012',
        'DREAMER_SIM_NOISE_ENABLED': '0',
        'DREAMER_SIM_NOISE_SEED': '7',
        'DREAMER_SIM_NOISE_JITTER_PCT': '0.15',
        'DREAMER_SIM_DOMAIN_RANDOMIZATION': '0',
        'DREAMER_SIM_DOMAIN_RANDOMIZATION_SEED': '11',
        'DREAMER_SIM_PARAM_RANDOMIZATION_PCT': '0.22',
        'DREAMER_DISTURBANCE_AUTHORITY_FRAC': '0.50',
        'DREAMER_DISTURBANCE_RECOVERY_FRAC': '0.15',
        'DREAMER_DISTURBANCE_SETTLE_STEPS': '40',
        'DREAMER_DISTURBANCE_QUIET_FRAC': '0.20',
        'DREAMER_OBJECTIVE_INTEGRAL_COEF': '0.08',
        'DREAMER_OBJECTIVE_PENALTY_CLIP': '40',
        'DREAMER_OBJ_AUTO_MV_OVER_CV_RATIO': '3.0',
        'DREAMER_OBJ_USE_NORMALIZED': '0',
        'DREAMER_RUNTIME_SETPOINT_BOUNDS_JITTER_FRAC': '0.22',
        'DREAMER_RUNTIME_SETPOINT_TARGET_JITTER_FRAC': '0.31',
        'DREAMER_RUNTIME_SETPOINT_BOUNDS_CHANGES_MAX': '3',
        'DREAMER_RUNTIME_SETPOINT_RAMP_DURATION_FRAC': '0.12',
        'DREAMER_STEP_SEED_DELTA_MIN': '0.25',
        'DREAMER_STEP_SEED_DELTA_MAX': '0.50',
        'DREAMER_STEP_SEED_PREFIX_FRAC_MIN': '0.08',
        'DREAMER_STEP_SEED_PREFIX_FRAC_MAX': '0.18',
        'DREAMER_SHAPING_SAFE_MARGIN_FRAC': '0.30',
        'DREAMER_PRBS_SEED_SEGMENT_STEPS': '40',
        'DREAMER_PRBS_SEED_SEGMENT_STEPS_MIN': '6',
        'DREAMER_WM_SS_MATCH_WINDOW_FRAC': '0.40',
        'DREAMER_EXPERT_MOVE_FRAC': '0.22',
        'DREAMER_BASELINE_SEED_EPS': '12',
        'DREAMER_EXPLORATION_SEED_EPS': '4',
        'DREAMER_DV_PRBS_SEEDS': '18',
        'DREAMER_DV_PRBS_OP_FRAC': '0.7',
        'DREAMER_GRAD_CLIP': '80',
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
        assert cfg.gain_match_clip_realized is False
        assert int(cfg.gain_match_settle_len) == 55
        assert cfg.gain_match_rest_ic is True
        assert cfg.gain_match_rest_ic_cuda_graph is False
        assert int(cfg.gain_match_rest_ic_len) == 16
        assert cfg.p3_reset_log_std is True
        assert cfg.p3_stop_grad_log_std is False
        assert abs(float(cfg.p3_logp_clip)) < 1e-12
        assert abs(float(cfg.p3_mu_ratio_clip)) < 1e-12
        assert int(cfg.p3_mu_ratio_refresh_iters) == 0
        assert abs(float(cfg.early_stop_entropy_collapse_floor_frac) - 0.1) < 1e-12
        assert cfg.bc_mean_only is False
        assert abs(float(cfg.baseline_seed_op_band) - 0.4) < 1e-12
        assert abs(float(cfg.constant_action_seed_op_band) - 0.55) < 1e-12
        assert abs(float(cfg.prbs_seed_op_band) - 0.8) < 1e-12
        assert abs(float(cfg.reward_raw_clip_min) + 30.0) < 1e-12
        assert abs(float(cfg.reward_cal_pct) - 90.0) < 1e-12
        assert int(cfg.diag_perhead_grads_every) == 10
        assert cfg.run_wm_diagnostic is False
        assert cfg.wm_diag_device == 'cpu'
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
        assert abs(float(cfg.episode_settle_multiple) - 10.0) < 1e-12
        assert int(cfg.episode_min_length) == 700
        assert int(cfg.episode_max_length) == 2000
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
        assert cfg.sim_noise_adaptive is False
        assert abs(float(cfg.sim_ou_sigma_frac) - 0.01) < 1e-12
        assert abs(float(cfg.sim_ou_gain_cv) - 0.2) < 1e-12
        assert abs(float(cfg.sim_ou_gain_dv) - 0.4) < 1e-12
        assert abs(float(cfg.sim_meas_noise_cv_frac) - 0.006) < 1e-12
        assert abs(float(cfg.sim_meas_noise_dv_frac) - 0.012) < 1e-12
        assert cfg.sim_noise_enabled is False
        assert cfg.sim_noise_seed == '7'
        assert abs(float(cfg.sim_noise_jitter_pct) - 0.15) < 1e-12
        assert cfg.sim_domain_randomization is False
        assert cfg.sim_domain_randomization_seed == '11'
        assert abs(float(cfg.sim_param_randomization_pct) - 0.22) < 1e-12
        assert abs(float(cfg.disturbance_authority_frac) - 0.50) < 1e-12
        assert abs(float(cfg.disturbance_recovery_frac) - 0.15) < 1e-12
        assert int(cfg.disturbance_settle_steps) == 40
        assert abs(float(cfg.disturbance_quiet_frac) - 0.20) < 1e-12
        assert abs(float(cfg.objective_integral_coef) - 0.08) < 1e-12
        assert abs(float(cfg.objective_penalty_clip) - 40.0) < 1e-12
        assert abs(float(cfg.obj_auto_mv_over_cv_ratio) - 3.0) < 1e-12
        assert cfg.objective_use_normalized is False
        assert abs(float(cfg.runtime_setpoint_bounds_jitter_frac) - 0.22) < 1e-12
        assert abs(float(cfg.runtime_setpoint_target_jitter_frac) - 0.31) < 1e-12
        assert int(cfg.runtime_setpoint_bounds_changes_max) == 3
        assert abs(float(cfg.runtime_setpoint_ramp_duration_frac) - 0.12) < 1e-12
        assert abs(float(cfg.step_seed_delta_min) - 0.25) < 1e-12
        assert abs(float(cfg.step_seed_delta_max) - 0.50) < 1e-12
        assert abs(float(cfg.step_seed_prefix_frac_min) - 0.08) < 1e-12
        assert abs(float(cfg.step_seed_prefix_frac_max) - 0.18) < 1e-12
        assert abs(float(cfg.shaping_safe_margin_frac) - 0.30) < 1e-12
        assert int(cfg.prbs_seed_segment_steps) == 40
        assert int(cfg.prbs_seed_segment_steps_min) == 6
        assert abs(float(cfg.wm_ss_match_window_frac) - 0.40) < 1e-12
        assert abs(float(cfg.expert_move_frac) - 0.22) < 1e-12
        assert int(cfg.baseline_seed_episodes) == 12
        assert int(cfg.exploration_seed_episodes) == 4
        assert int(cfg.dv_prbs_seed_episodes) == 18
        assert abs(float(cfg.dv_prbs_op_frac) - 0.7) < 1e-12
        assert abs(float(cfg.grad_clip) - 80.0) < 1e-12
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
        assert 'episode_settle_multiple' in explicit
        assert 'episode_min_length' in explicit
        assert 'episode_max_length' in explicit
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
        assert 'sim_noise_adaptive' in explicit
        assert 'sim_ou_sigma_frac' in explicit
        assert 'sim_ou_gain_cv' in explicit
        assert 'sim_ou_gain_dv' in explicit
        assert 'sim_meas_noise_cv_frac' in explicit
        assert 'sim_meas_noise_dv_frac' in explicit
        assert 'sim_noise_enabled' in explicit
        assert 'sim_noise_seed' in explicit
        assert 'sim_noise_jitter_pct' in explicit
        assert 'sim_domain_randomization' in explicit
        assert 'sim_domain_randomization_seed' in explicit
        assert 'sim_param_randomization_pct' in explicit
        assert 'disturbance_authority_frac' in explicit
        assert 'disturbance_recovery_frac' in explicit
        assert 'disturbance_settle_steps' in explicit
        assert 'disturbance_quiet_frac' in explicit
        assert 'objective_integral_coef' in explicit
        assert 'objective_penalty_clip' in explicit
        assert 'obj_auto_mv_over_cv_ratio' in explicit
        assert 'objective_use_normalized' in explicit
        assert 'runtime_setpoint_bounds_jitter_frac' in explicit
        assert 'runtime_setpoint_target_jitter_frac' in explicit
        assert 'runtime_setpoint_bounds_changes_max' in explicit
        assert 'runtime_setpoint_ramp_duration_frac' in explicit
        assert 'step_seed_delta_min' in explicit
        assert 'step_seed_prefix_frac_max' in explicit
        assert 'shaping_safe_margin_frac' in explicit
        assert 'prbs_seed_segment_steps' in explicit
        assert 'wm_ss_match_window_frac' in explicit
        assert 'expert_move_frac' in explicit
        assert 'wm_diag_device' in explicit
        assert 'baseline_seed_episodes' in explicit
        assert 'exploration_seed_episodes' in explicit
        assert 'dv_prbs_seed_episodes' in explicit
        assert 'gain_match_rest_ic_cuda_graph' in explicit
        assert 'gain_match_rest_ic_len' in explicit
        assert 'gain_match_clip_realized' in explicit
        print('[smoke] OK  _cfg_from_env applies ENV_OVERRIDES (aux TBPTT / skip-storm / N)')
    finally:
        for k, old in prev.items():
            if old is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = old


def _test_recon_channel_weights_cache() -> None:
    """Mean-1 recon weights cache on cfg (identity vs rebuild)."""
    from types import SimpleNamespace
    from pathlib import Path
    n_ch, cv_idx = 4, (1,)
    cfg = SimpleNamespace(
        wm_recon_cv_weight=6.0,
        wm_recon_dv_weight=1.0,
        cv_obs_indices=cv_idx,
        dv_indices=(),
        _recon_ch_w=None,
    )
    w1 = _recon_channel_weights(cfg, n_ch, 'cpu', torch.float32)
    w2 = _recon_channel_weights(cfg, n_ch, 'cpu', torch.float32)
    assert w1 is not None and w1 is w2, 'recon channel weights must cache'
    assert abs(float(w1.mean().item()) - 1.0) < 1e-6
    raw = torch.ones(n_ch, dtype=torch.float32)
    raw[cv_idx[0]] = 6.0
    expected = raw * (float(raw.numel()) / raw.sum().clamp_min(1e-8))
    assert torch.allclose(w1.cpu(), expected)
    cfg._recon_ch_w = None
    w3 = _recon_channel_weights(cfg, n_ch, 'cpu', torch.float32)
    assert torch.allclose(w1, w3)
    cfg_off = SimpleNamespace(
        wm_recon_cv_weight=1.0, wm_recon_dv_weight=1.0,
        cv_obs_indices=(), dv_indices=(), _recon_ch_w=None,
    )
    assert _recon_channel_weights(cfg_off, n_ch, 'cpu', torch.float32) is None
    src = Path(__file__).resolve().parents[1].joinpath('training', 'train.py')
    text = src.read_text()
    assert "getattr(cfg, 'wm_overshoot_tail_power', 2.0)" in text
    assert 'def _cached_arange_1k' in text
    assert 'k_off = torch.arange(1, K + 1' not in text
    cfg_idx = TrainConfig()
    k1 = _cached_arange_1k(cfg_idx, 8, torch.device('cpu'))
    k2 = _cached_arange_1k(cfg_idx, 8, torch.device('cpu'))
    assert k1 is k2
    assert torch.equal(k1, torch.arange(1, 9))
    k3 = _cached_arange_1k(cfg_idx, 5, torch.device('cpu'))
    assert k3 is not k1 and k3.numel() == 5
    s1 = _cached_strided_arange(cfg_idx, 73, 3, torch.device('cpu'))
    s2 = _cached_strided_arange(cfg_idx, 73, 3, torch.device('cpu'))
    assert s1 is s2
    assert torch.equal(s1, torch.arange(0, 73, 3))
    st, idx = _cached_time_gather_idx(cfg_idx, 73, 3, 8, torch.device('cpu'))
    st2, idx2 = _cached_time_gather_idx(cfg_idx, 73, 3, 8, torch.device('cpu'))
    assert st is st2 and idx is idx2
    k_off = torch.arange(1, 9)
    expect = s1.view(-1, 1) + k_off.view(1, -1)
    assert torch.equal(idx, expect)
    st3, idx3 = _cached_time_gather_idx(cfg_idx, 73, 3, 5, torch.device('cpu'))
    assert idx3 is not idx and idx3.shape[-1] == 5
    ov = text[text.index('def _wm_latent_overshoot_loss'):
              text.index('def _cached_arange_1k')]
    assert 'obs_win = obs[:, idx]' in ov
    assert ov.count('obs[:, idx]') == 1
    iso = text[text.index('def _wm_input_isolation_loss'):
               text.index('def _auto_gain_match_settle_len')]
    assert 'obs_win = obs[:, idx]' in iso
    assert 'obs[:, idx].index_select' not in iso
    cfg_idx.wm_overshoot_tail_power = 2.0
    wk1 = _overshoot_tail_wk(cfg_idx, 8, torch.device('cpu'), torch.float32)
    wk2 = _overshoot_tail_wk(cfg_idx, 8, torch.device('cpu'), torch.float32)
    assert wk1 is wk2
    k_idx = torch.arange(1, 9, dtype=torch.float32)
    expected_wk = (k_idx / 8.0) ** 2.0
    assert torch.allclose(wk1, expected_wk)
    cfg_idx.wm_overshoot_tail_power = 0.0
    cfg_idx._overshoot_wk = None
    wk0 = _overshoot_tail_wk(cfg_idx, 8, torch.device('cpu'), torch.float32)
    assert torch.allclose(wk0, torch.ones(8))
    print('[smoke] OK  recon channel weights cache identity; tail_power fallback 2.0; overshoot arange/wk/gather-idx cache')


def _test_cli_only_env_disjoint() -> None:
    """CLI leftovers must not double-setattr ENV_OVERRIDES DREAMER keys."""
    from workflow._plant_prepare import ENV_OVERRIDES
    overlap = {k for k, _, _ in _CLI_ONLY_ENV if k in ENV_OVERRIDES}
    assert not overlap, overlap
    kept = {k for k, _, _ in _CLI_ONLY_ENV}
    assert kept == {
        'AGENT_TOTAL_STEPS', 'SIM_EPISODE_LENGTH', 'SIM_SAMPLE_RATE',
        'CONTROLLER_OUT_DIR',
    }
    print('[smoke] OK  _CLI_ONLY_ENV disjoint from ENV_OVERRIDES')


def _test_batch_np_to_device_identity() -> None:
    """Pinned replay H2D reuse must copy values (CPU always; CUDA if visible)."""
    import numpy as np
    from training.train import (
        _batch_np_to_device, TrainConfig)

    c = TrainConfig()
    assert not hasattr(c, 'p3_prior_refresh_iters')
    assert not hasattr(c, 'joint_prior_refresh_iters')

    batch = {
        'obs': np.linspace(0, 1, 2 * 4 * 3, dtype='float32').reshape(2, 4, 3),
        'act': np.linspace(-1, 1, 2 * 4 * 1, dtype='float32').reshape(2, 4, 1),
        'rew': np.linspace(-2, 0, 2 * 4, dtype='float32').reshape(2, 4),
    }
    cpu = torch.device('cpu')
    out = _batch_np_to_device(batch, cpu)
    for k, v in batch.items():
        np.testing.assert_allclose(out[k].numpy(), v, rtol=0, atol=0)
    out2 = _batch_np_to_device(
        {k: np.zeros_like(v) for k, v in batch.items()}, cpu)
    np.testing.assert_allclose(out['obs'].numpy(), batch['obs'], rtol=0, atol=0)
    np.testing.assert_allclose(out2['obs'].numpy(), 0.0, rtol=0, atol=0)

    zeros = {k: np.zeros_like(v) for k, v in batch.items()}
    replay = _batch_np_to_device(batch, cpu, slot='replay')
    critic = _batch_np_to_device(zeros, cpu, slot='critic')
    np.testing.assert_allclose(replay['obs'].numpy(), batch['obs'], rtol=0, atol=0)
    np.testing.assert_allclose(critic['obs'].numpy(), 0.0, rtol=0, atol=0)

    if torch.cuda.is_available():
        _batch_np_to_device._host = {}  # type: ignore[attr-defined]
        _batch_np_to_device._gpu = {}  # type: ignore[attr-defined]
        dev = torch.device('cuda')
        g_rep = _batch_np_to_device(batch, dev, slot='replay')
        g_crit = _batch_np_to_device(zeros, dev, slot='critic')
        for k, v in batch.items():
            np.testing.assert_allclose(
                g_rep[k].detach().cpu().numpy(), v, rtol=0, atol=1e-6)
            np.testing.assert_allclose(
                g_crit[k].detach().cpu().numpy(), 0.0, rtol=0, atol=1e-6)
        g_rep2 = _batch_np_to_device(zeros, dev, slot='replay')
        np.testing.assert_allclose(
            g_rep2['obs'].detach().cpu().numpy(), 0.0, rtol=0, atol=1e-6)
        np.testing.assert_allclose(
            g_crit['obs'].detach().cpu().numpy(), 0.0, rtol=0, atol=1e-6)
        _batch_np_to_device._host = {}  # type: ignore[attr-defined]
        _batch_np_to_device._gpu = {}  # type: ignore[attr-defined]
    slim = _batch_np_to_device(batch, cpu, keys=('obs', 'act'))
    assert set(slim.keys()) == {'obs', 'act'}
    np.testing.assert_allclose(slim['obs'].numpy(), batch['obs'], rtol=0, atol=0)
    np.testing.assert_allclose(slim['act'].numpy(), batch['act'], rtol=0, atol=0)
    assert 'rew' not in slim
    print('[smoke] OK  replay H2D identity + PMPO prior-refresh REMOVED')


def _test_time_unbind_and_p1_h2d_keys() -> None:
    """Unbind dim=1 ≡ ``x[:, t]``; Stage-1 zeros cache; P1 skip unused dist."""
    from models.dreamer_v4_rssm import _time_unbind, cached_zeros_btd

    assert _time_unbind(None) is None
    x = torch.arange(2 * 5 * 3, dtype=torch.float32).reshape(2, 5, 3)
    parts = _time_unbind(x)
    assert parts is not None and len(parts) == 5
    for t in range(5):
        assert torch.equal(parts[t], x[:, t])

    class _Mod:
        pass

    mod = _Mod()
    z1 = cached_zeros_btd(mod, 2, 4, 3, torch.float32, torch.device('cpu'))
    z2 = cached_zeros_btd(mod, 2, 4, 3, torch.float32, torch.device('cpu'))
    assert z1 is z2
    assert float(z1.abs().sum()) == 0.0
    z3 = cached_zeros_btd(mod, 2, 5, 3, torch.float32, torch.device('cpu'))
    assert z3 is not z1
    assert z3.shape == (2, 5, 3)
    assert z1 is cached_zeros_btd(mod, 2, 4, 3, torch.float32, torch.device('cpu'))
    assert float(z1.abs().sum()) == 0.0

    class _M:
        pass

    class _D:
        pass

    m = _M()
    m.dynamics = _D()
    m.disturbance = None
    m.dynamics.dob_enabled = True
    m.dynamics.dob_active = False
    m.dynamics.cont_dist_dim = 0
    c = TrainConfig()
    c.dob_ground_coef = 2.0
    c.dist_match_coef = 0.0
    c.disturbance_loss_scale = 1.0
    assert _wm_need_dist_target(m, c) is False
    assert _p1_wm_h2d_keys(_wm_need_dist_target(m, c)) == ('obs', 'act')
    assert _replay_h2d_keys(False, True) == ('obs', 'act', 'rew', 'expert')
    assert _replay_h2d_keys(False, True, False) == ('obs', 'act', 'rew')
    assert _replay_h2d_keys(True, True) == (
        'obs', 'act', 'dist', 'rew', 'expert')
    m.dynamics.dob_active = True
    assert _wm_need_dist_target(m, c) is True
    assert 'dist' in _p1_wm_h2d_keys(True)
    m.dynamics.dob_active = False
    c.dist_match_coef = 0.6
    m.dynamics.cont_dist_dim = 1
    assert _wm_need_dist_target(m, c) is True
    c.dist_match_coef = 0.0
    m.dynamics.cont_dist_dim = 0
    m.disturbance = object()
    assert _wm_need_dist_target(m, c) is True
    print('[smoke] OK  time-unbind identity + Stage-1 zeros cache + P1 dist H2D')


def _test_lambda_returns_scan() -> None:
    """Weighted reverse-cumsum ≡ sequential TD-λ (incl. MC / λ=0 / T=1)."""
    from training.train import _lambda_discount_weights, _lambda_returns

    def _ref(rew, v, gamma, lam, cap=None):
        if cap is not None:
            v = v.clamp(-cap, cap)
        out = torch.zeros_like(v)
        out[:, -1] = v[:, -1]
        for t in reversed(range(int(v.shape[1]) - 1)):
            boot = (1.0 - lam) * v[:, t + 1] + lam * out[:, t + 1]
            out[:, t] = rew[:, t] + gamma * boot
        out = out.detach()
        if cap is not None:
            out = out.clamp(-cap, cap)
        return out

    torch.manual_seed(0)
    for T, gamma, lam, cap in (
            (8, 0.99, 0.90, 50.0),
            (1, 0.99, 0.90, 50.0),
            (16, 0.983, 0.90, None),
            (8, 0.99, 1.0, 50.0),
            (8, 0.99, 0.0, None),
    ):
        rew = torch.randn(4, T)
        v = torch.randn(4, T)
        got = _lambda_returns(rew, v, gamma, lam, cap)
        ref = _ref(rew, v, gamma, lam, cap)
        assert torch.allclose(got, ref, atol=1e-5, rtol=1e-5), (
            f'λ-scan mismatch T={T} lam={lam} max='
            f'{float((got - ref).abs().max())}')
        got2 = _lambda_returns(rew, v, gamma, lam, cap)
        assert torch.allclose(got, got2, atol=0.0, rtol=0.0)
    w1 = _lambda_discount_weights(8, 0.99 * 0.90, torch.device('cpu'),
                                  torch.float32)
    w2 = _lambda_discount_weights(8, 0.99 * 0.90, torch.device('cpu'),
                                  torch.float32)
    assert w1 is w2
    w8 = _lambda_discount_weights(16, 0.99 * 0.90, torch.device('cpu'),
                                  torch.float32)
    assert w8 is not w1
    print('[smoke] OK  TD-λ reverse-cumsum ≡ sequential recurrence')


def _test_buffer_sample_keys() -> None:
    """Subset sample skips leftover channels; RNG identity vs full sample."""
    import numpy as np
    from training.train import TrajectoryBuffer

    T, D, A, N = 16, 3, 1, 5
    buf = TrajectoryBuffer(N, T, D, A, n_dist=1)
    for i in range(N):
        obs = np.full((T, D), float(i), dtype='float32')
        act = np.full((T, A), float(i) + 0.5, dtype='float32')
        rew = np.arange(T, dtype='float32') + i
        cont = np.ones(T, dtype='float32')
        expert = (np.arange(T) == i).astype('float32')
        dist = np.full((T, 1), float(i) + 0.25, dtype='float32')
        buf.add_episode(obs, act, rew, cont, expert=expert, dist=dist)
    B, S = 4, 6
    full = buf.sample(B, S, np.random.default_rng(3))
    slim = buf.sample(B, S, np.random.default_rng(3),
                      keys=('obs', 'act', 'rew'))
    assert set(slim) == {'obs', 'act', 'rew'}
    for k in slim:
        assert np.array_equal(slim[k], full[k])
    assert 'cont' not in slim and 'expert' not in slim and 'dist' not in slim
    buf2 = TrajectoryBuffer(2, T, D, A, n_dist=1)
    buf2.add_episode(obs, act, rew, cont, expert=None, dist=dist)
    assert np.all(buf2.expert[0] == 0.0)
    expert_row = (np.arange(T) == 1).astype('float32')
    buf2.add_episode(obs, act, rew, cont, expert=expert_row, dist=dist)
    assert np.array_equal(buf2.expert[1], expert_row)
    print('[smoke] OK  buffer sample keys subset identity')


def _test_buffer_clear() -> None:
    """P1→P2 flush drops filled/write; leftover slots are not sampled."""
    import numpy as np
    from training.train import TrajectoryBuffer

    T, D, A = 8, 3, 1
    buf = TrajectoryBuffer(4, T, D, A, n_dist=1)
    for i in range(3):
        obs = np.full((T, D), float(i), dtype='float32')
        act = np.full((T, A), float(i), dtype='float32')
        rew = np.zeros(T, dtype='float32')
        cont = np.ones(T, dtype='float32')
        dist = np.zeros((T, 1), dtype='float32')
        buf.add_episode(obs, act, rew, cont, dist=dist)
    assert buf.filled == 3 and buf.write == 3
    buf.clear()
    assert buf.filled == 0 and buf.write == 0
    try:
        buf.sample(2, 4, np.random.default_rng(0))
        raise AssertionError('sample on empty buffer must raise')
    except ValueError:
        pass
    obs = np.ones((T, D), dtype='float32')
    act = np.ones((T, A), dtype='float32')
    rew = np.ones(T, dtype='float32')
    cont = np.ones(T, dtype='float32')
    dist = np.full((T, 1), 7.0, dtype='float32')
    buf.add_episode(obs, act, rew, cont, dist=dist)
    assert buf.filled == 1 and buf.write == 1
    got = buf.sample(8, 4, np.random.default_rng(1))
    assert got['obs'].shape[0] == 8
    assert np.allclose(got['dist'], 7.0)
    print('[smoke] OK  buffer clear drops P1 rows; refill samples new')


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
    from models.dreamer_v4_rssm import _append_decode_core, _stack_decode_core
    dec_in = (m.deter_dim + m.stoch_flat_dim + m.cont_dim + m._dv_feed_dim)
    emb = m.embed(obs)
    dvs = (obs.index_select(-1, m.dv_index_t) if m.dv_dim > 0 else None)
    st = m.initial_state(B, obs.device)
    feat_l, hh, zz, cc, ddv = [], [], [], [], []
    for t in range(T):
        dv_t = None if dvs is None else dvs[:, t]
        post, _ = m.obs_step(
            st, act[:, t], emb[:, t], dv=dv_t, sample=False, obs=None)
        feat_l.append(post.feat[..., :dec_in])
        _append_decode_core(hh, zz, cc, ddv, post)
        st = post
    stack_err = float(
        (_stack_decode_core(hh, zz, cc, ddv) - torch.stack(feat_l, 1)
         ).detach().abs().max())
    assert stack_err < 1e-6, f'_stack_decode_core != feat[..., :dec_in] ({stack_err})'
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
          f'stack-core ≡ feat slice ({stack_err:.2e}); '
          f'kalman mix budget={bud}')


def _test_img_rollout_last_only() -> None:
    """Gain-match last-step Huber: last_only ≡ stack[:, -1]; GRU still gets grad.

    ``out='obs'`` is identity vs decoding the feat stack (overshoot / P62
    held).  ``out='h'`` was P61 held and is removed (isolation TBPTT
    keeps default feat and slices ``h`` for ``keep_c``).
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
    try:
        m.img_rollout(h0, z0, acts, sample=False, out='h')
        raise AssertionError("out='h' should be removed")
    except ValueError as exc:
        assert 'out' in str(exc)
    obs_roll = m.img_rollout(h0, z0, acts, sample=False, out='obs')
    obs_err = float((obs_roll - m.decode(roll)).detach().abs().max())
    assert obs_err < 1e-5, f"out='obs' != decode(feat) (max_err={obs_err})"
    last_obs = m.img_rollout(h0, z0, acts, sample=False, last_only=True, out='obs')
    last_obs_err = float((last_obs - obs_roll[:, -1]).detach().abs().max())
    assert last_obs_err < 1e-5, f"last_only out='obs' != stack[:, -1] (max_err={last_obs_err})"
    store = getattr(m, '_img_zlogits_zeros', None)
    assert isinstance(store, dict) and store, 'img_rollout z_logits zeros not cached'
    roll2 = m.img_rollout(h0, z0, acts, sample=False)
    cache_err = float((roll2 - roll).detach().abs().max())
    assert cache_err < 1e-6, f'cached z_logits zeros changed rollout (max_err={cache_err})'
    m.zero_grad(set_to_none=True)
    last.sum().backward()
    gru_g = sum(float(p.grad.abs().sum()) for p in m.gru.parameters()
                if p.grad is not None)
    assert gru_g > 0.0, 'last_only decode/feat lost GRU gradient'
    print(f'[smoke] OK  img_rollout last_only ≡ stack[:, -1] '
          f'(max_err={err:.2e}); out=obs identity '
          f'(obs={obs_err:.2e} last_obs={last_obs_err:.2e}); gru |g|={gru_g:.3f}; '
          f'z_logits cache identity max_err={cache_err:.2e}')


def _test_img_step_det_roll_skips_sample() -> None:
    """Gain det-roll: sample=True prior c is the mean; skip discarded randn."""
    from models.dreamer_v4_rssm import (
        RSSMConfig, RSSMDynamics, cached_zeros_bd)
    torch.manual_seed(0)
    cfg = RSSMConfig(obs_dim=6, action_dim=2, deter_dim=16,
                     n_categoricals=4, n_classes=4, embed_dim=16,
                     hidden_dim=16, latent_type='deterministic',
                     cont_gain_dim=2, latent_noise=0.0)
    m = RSSMDynamics(cfg)
    assert m.cont_gain_deterministic_roll is True
    assert int(m.cont_gain_dim) >= int(m.cont_dim)
    B = 4
    state = m.initial_state(B, torch.device('cpu'))
    a = torch.zeros(B, cfg.action_dim)
    samples = []
    orig = m.cont_prior_net.forward

    def _spy(x, sample=True):
        samples.append(bool(sample))
        return orig(x, sample=sample)

    m.cont_prior_net.forward = _spy
    s_true = m.img_step(state, a, sample=True)
    s_false = m.img_step(state, a, sample=False)
    assert samples == [False, False], samples
    assert s_true.c is not None and s_true.c_mean is not None
    assert torch.allclose(s_true.c, s_true.c_mean)
    assert torch.allclose(s_true.c, s_false.c)
    assert torch.allclose(s_true.z, s_false.z)
    assert torch.allclose(s_true.h, s_false.h)
    z1 = cached_zeros_bd(m, B, m.cont_dim, a.dtype, a.device)
    z2 = cached_zeros_bd(m, B, m.cont_dim, a.dtype, a.device)
    assert z1 is z2
    print('[smoke] OK  img_step det-roll skips discarded prior-c sample')


def _test_initial_state_zeros_cache() -> None:
    """``initial_state`` reuses zero/one-hot ICs; GRU does not write them."""
    from models.dreamer_v4_rssm import (
        RSSMConfig, RSSMDynamics, cached_onehot_z)
    torch.manual_seed(0)
    cfg = RSSMConfig(obs_dim=6, action_dim=2, deter_dim=16,
                     n_categoricals=4, n_classes=4, embed_dim=16,
                     hidden_dim=16, latent_type='deterministic',
                     cont_gain_dim=2, dob_enabled=True, cv_indices=(0,))
    m = RSSMDynamics(cfg)
    B = 4
    device = torch.device('cpu')
    s1 = m.initial_state(B, device)
    s2 = m.initial_state(B, device)
    assert s1.h is s2.h
    assert s1.z is s2.z
    assert s1.z_logits is s2.z_logits
    assert s1.d is s2.d
    assert s1.c is s2.c
    assert s1.c_mean is s2.c_mean and s1.c_mean is not s1.c
    assert s1.c_std is not s1.c
    assert float(s1.h.abs().sum()) == 0.0
    assert float(s1.z[..., 0].min()) == 1.0
    assert float(s1.z[..., 1:].abs().sum()) == 0.0
    z_oh = cached_onehot_z(
        m, B, m.n_categoricals, m.n_classes, s1.z.dtype, device)
    assert z_oh is s1.z
    s8 = m.initial_state(8, device)
    assert s8.h is not s1.h
    assert s8.h.shape[0] == 8
    assert m.initial_state(B, device).h is s1.h
    h_before = s1.h.clone()
    z_before = s1.z.clone()
    a = torch.zeros(B, cfg.action_dim)
    _ = m.img_step(s1, a, sample=False)
    assert torch.equal(s1.h, h_before)
    assert torch.equal(s1.z, z_before)
    obs = torch.randn(B, 6, cfg.obs_dim)
    act = torch.zeros(B, 6, cfg.action_dim)
    f1, *_ = m.rollout_observed(obs, act, sample=False, store_aux=False)
    f2, *_ = m.rollout_observed(obs, act, sample=False, store_aux=False)
    assert torch.allclose(f1, f2)
    print('[smoke] OK  initial_state zero/one-hot cache (RSSM)')


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
    assert '_gain_match_tgt_tensor' in _src
    assert '_should_lock_last_ok' in _src
    assert '_should_probe_gain_on_last_ok' in _src
    assert '_probe_observer_gain_ready_maybe_last_ok' in _src
    assert '[gain-ready-probe] last-ok iter' in _src
    assert 'extra_p1=int(p1_ext_steps) > 0' in _src
    assert 'unlocked after wrap recovery' in _src
    assert "lock={float(getattr(cfg, 'skip_storm_last_ok_lock_ratio'" in _src
    assert "huber_per_in={bool(getattr(cfg, 'gain_match_huber_per_input'" in _src
    assert "gmatch_settle={int(getattr(cfg, 'gain_match_settle_len'" in _src
    assert "gmatch_step={float(getattr(cfg, 'gain_match_step'" in _src
    assert "gmatch_clip={bool(getattr(cfg, 'gain_match_clip_realized'" in _src
    assert '_cube_step_held' in _src
    assert '_gain_match_realized_du' in _src
    assert 'gain_match_du_frac' in _src
    assert 'wm_gain_match_du_frac' in _src
    assert 'gain_match_clip_frac' in _src
    assert 'wm_gain_match_clip_frac' in _src
    assert 'def _cube_plus_would_clip' in _src
    assert 'def _gain_match_clip_frac_t' in _src
    assert "gmatch_rest={bool(getattr(cfg, 'gain_match_rest_ic'" in _src
    assert 'gmatch_rest_L=' in _src
    assert "gmatch_rest_cg={bool(getattr(cfg, 'gain_match_rest_ic_cuda_graph'" in _src
    assert "p3_sigreset={bool(getattr(cfg, 'p3_reset_log_std'" in _src
    assert "bc_mean={bool(getattr(cfg, 'bc_mean_only'" in _src
    assert "p3_sglogstd={bool(getattr(cfg, 'p3_stop_grad_log_std'" in _src
    assert "p3_logpclip={float(getattr(cfg, 'p3_logp_clip'" in _src
    assert "p3_muratio={float(getattr(cfg, 'p3_mu_ratio_clip'" in _src
    assert "p3_murefresh={int(getattr(cfg, 'p3_mu_ratio_refresh_iters'" in _src
    assert '_p3_copy_policy_snapshot' in _src
    assert '_p3_load_policy_snapshot' in _src
    assert '_release_rest_ic_cuda_graph' in _src
    assert "'actor_pos_adv_frac'" in _src
    assert "'adv_action_corr'" in _src
    assert "'n_grad_skip_iter'" in _src
    assert 'logp {_logp:.2f}' in _src
    assert 'clip {_clip:.2f}' in _src
    assert 'def _row_adv_action_corr(' in _src
    assert 'plt.subplots(3, 3' in _src
    assert '_row_adv_action_corr(row)' in _src
    assert 'out_path = Path(out_path)' in _src
    assert "es_ent_floor={float(getattr(cfg, 'early_stop_entropy_collapse_floor_frac'" in _src
    assert '_p3_logp_clip_bound' in _src
    assert '_entropy_collapse_threshold' in _src
    assert '_p3_mu_ratio_surrogate' in _src
    assert '_p3_frozen_unfreeze_policy' in _src
    assert '_maybe_snapshot_prior_policy' not in _src
    assert '_actor_uses_prior_policy' not in _src
    assert 'p3_prior_refresh_iters: int' not in _src
    assert 'joint_prior_refresh_iters: int' not in _src
    assert 'actor_kl_coef: float' not in _src
    assert 'pmpo_alpha: float' not in _src
    assert 'pmpo_beta: float' not in _src
    assert "'critic_mc_loss'" in _src
    assert 'def _pearson_r(' in _src
    assert 'reuse pinned host + GPU dest' in _src
    assert "slot: str = 'replay'" in _src
    assert "slot='iso'" in _src
    assert "slot='critic'" in _src
    assert "SIM_IDENTIFIED_TAU_DOMINANT', '50'" not in _src
    assert 'lb // 4' in _src
    assert '_gain_match_held_settle' in _src
    assert '_gain_match_rest_window' in _src
    assert 'gain_match_rest_ic_len: int = 0' in _src
    assert '_gain_match_rest_ic_state' in _src
    assert '_rest_ic_can_cuda_graph' in _src
    assert 'make_graphed_callables' in _src
    assert 'cache_enabled=False' in _src
    assert 'enabled=False' in _src
    assert 'class _RestICGraphModule' in _src
    assert 'wrapper.train(bool(rssm.training))' in _src
    assert 'ck_scale > 0.0' in _src
    _ric = _src[_src.index('class _RestICGraphModule'):
                _src.index('def _rest_ic_note_capture_miss')]
    assert 'def parameters(self' in _ric
    assert 'def named_parameters' in _ric
    assert 'def buffers(self' in _ric
    assert 'def named_buffers' in _ric
    assert 'return self._rssm.parameters' not in _ric
    assert '_rest_ic_note_capture_miss' in _src
    assert 'warmup retry after empty_cache' in _src
    assert 'allow_unused_input=True' in _src
    assert "o_s = obs.detach().requires_grad_(True)" not in _src
    assert 'def _fn(o, a):' not in _src
    assert '_warmup_rest_ic_cuda_graph' in _src
    assert '_amp_parent_autocast_on' in _src
    assert 'def _suppress_accumulate_grad_stream_warn' in _src
    assert 'def _arm_rest_ic_stream_mismatch_warn' in _src
    assert 'set_warn_on_accumulate_grad_stream_mismatch' in _src
    _cap = _src[_src.index('def _capture_rest_ic_cuda_graph'):
                _src.index('\ndef _rest_ic_encode_hzc')]
    assert 'with _suppress_accumulate_grad_stream_warn()' in _cap
    assert (_cap.find('with _suppress_accumulate_grad_stream_warn()')
            < _cap.find('.backward()'))
    assert '_arm_rest_ic_stream_mismatch_warn(True)' in _cap
    # Arm after the suppress context (canary `del`) so restore cannot
    # clobber the live-WM flag.
    assert _cap.find('del h, z, c') < _cap.find(
        '_arm_rest_ic_stream_mismatch_warn(True)')
    _rel = _src[_src.index('def _release_rest_ic_cuda_graph'):
                _src.index('def _release_rest_ic_after_g_freeze')]
    assert '_arm_rest_ic_stream_mismatch_warn(False)' in _rel
    _rssm_src = _P(_tr.__file__).resolve().parents[1].joinpath(
        'models/dreamer_v4_rssm.py').read_text()
    _app = _rssm_src[_rssm_src.index('def _append_decode_core'):
                     _rssm_src.index('def _stack_decode_core')]
    assert 'z_l.append(st.z)' in _app
    assert 'st.stoch_flat' not in _app
    _stk = _rssm_src[_rssm_src.index('def _stack_decode_core'):
                     _rssm_src.index('def cached_zeros_btd')]
    assert '.flatten(start_dim=-2)' in _stk
    _cz = _rssm_src[_rssm_src.index('def cached_zeros_btd'):
                    _rssm_src.index('class RSSMConfig')]
    assert 'isinstance(store, dict)' in _cz
    assert 'def cached_zeros_bd' in _cz
    assert 'def cached_onehot_z' in _cz
    assert 'def _prior_c_from_net' in _cz
    assert 'sample=not take_mean' in _cz
    assert 'c0 if c0 is not None else cached_zeros_bd' in _rssm_src
    assert '_img_zlogits_zeros' in _rssm_src
    assert 'def _cached_arange_1k' in _src
    assert 'def _cached_strided_arange' in _src
    assert 'def _cached_time_gather_idx' in _src
    assert 'def _lambda_discount_weights' in _src
    assert 'expo = torch.arange(t_len, device=v.device, dtype=v.dtype)' not in _src
    assert "attr='_gmatch_starts'" in _src
    assert 'starts = torch.arange(0, n_valid, stride, device=obs.device)' not in _src
    assert 'def _overshoot_tail_wk' in _src
    assert 'k_off = torch.arange(1, K + 1' not in _src
    assert 'obs_win = obs[:, idx]' in _src
    assert '_gain_match_fd_action_seq' in _src
    assert 'def _wm_need_logged_aux' in _src
    assert '_wm_need_enc_diag' in _src
    assert 'def _wm_need_dist_target' in _src
    assert 'def _p1_wm_h2d_keys' in _src
    assert 'def _replay_h2d_keys' in _src
    assert '_h2d_keys = _p1_wm_h2d_keys' in _src
    assert "_h2d_keys = ('obs', 'act', 'dist')" not in _src
    assert '_h2d_keys = None' not in _src
    assert '_replay_h2d_keys(False, True)' in _src
    assert '_replay_h2d_keys(False, True, False)' in _src
    assert 'self.expert[i] = 0.0' in _src
    assert "else np.zeros(self.T, dtype='float32')" not in _src
    assert 'keys=_h2d_keys' in _src
    assert "keys=('obs', 'act')" in _src
    assert '_cache_gain_match_rest_ic' in _src
    assert 'reset_policy_exploration' in _src
    assert 'reset_policy_exploration(opt_actor)' in _src
    assert 'stream_serve_step' in _src
    assert 'get_collect_serve_cuda_graph' in _src
    assert '_rssm_prev_a.copy_' in _src
    assert '[p3] on-policy collect streams measured DV + Kalman' in _src
    assert "setdefault('jemb_loss'" in _src
    assert '_require_realsim_actor' in _src
    assert 'DREAMER_ACTOR_LOSS would be a false A/B' in _src
    assert '_rssm_param_grad_snapshot' in _src
    assert '_rssm_param_grad_restore' in _src
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
    assert 'held_cv=True' in _src
    assert 'p1amp=' in _src
    assert 'p2amp=' in _src
    assert 'p3amp=' in _src
    assert 'wm_held_cv_drift' in _src
    assert "'wm_held_ol_ratio'" not in _src
    assert "'pmpo_pos_frac': pos_adv_frac" not in _src
    assert "'imag_adv_action_corr': adv_action_corr" not in _src
    assert "'imagined_return_mean': target_returns" not in _src
    assert not (__import__('pathlib').Path(__file__).resolve().parent
                / 'run_nl_then_p09.sh').exists()
    assert 'cv_index_t' in _src
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
    assert c.gain_match_clip_realized is True
    assert c.gain_match_rest_ic is True
    assert int(c.gain_match_rest_ic_len) == 0
    assert c.gain_match_rest_ic_cuda_graph is True
    assert c.p3_reset_log_std is False
    assert not hasattr(c, 'actor_kl_coef')
    assert not hasattr(c, 'pmpo_alpha')
    assert not hasattr(c, 'pmpo_beta')
    assert not hasattr(c, 'p3_prior_refresh_iters')
    assert not hasattr(c, 'joint_prior_refresh_iters')
    assert c.wm_diag_device == 'cuda'
    assert c.bc_mean_only is True
    assert c.p3_stop_grad_log_std is True
    assert abs(float(c.p3_logp_clip) - 8.0) < 1e-12
    assert abs(float(c.p3_mu_ratio_clip) - 0.2) < 1e-12
    assert int(c.p3_mu_ratio_refresh_iters) == 0
    assert abs(float(c.early_stop_entropy_collapse_floor_frac) - 0.25) < 1e-12
    assert abs(float(c.obj_auto_mv_over_cv_ratio) - 2.0) < 1e-12
    assert abs(float(c.runtime_setpoint_bounds_jitter_frac) - 0.15) < 1e-12
    assert int(c.runtime_setpoint_bounds_changes_min) == 1
    assert int(c.runtime_setpoint_bounds_changes_max) == 2
    assert abs(float(c.runtime_setpoint_ramp_duration_frac) - 0.10) < 1e-12
    assert int(c.runtime_setpoint_n_magnitude_strata) == 3
    assert c.objective_use_normalized is True
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
    assert 'DREAMER_WM_HELD_ROLLOUT_SETTLE_FRAC' not in ENV_OVERRIDES
    assert 'DREAMER_ACT_HIST_REQUIRED' not in ENV_OVERRIDES
    _dv4 = open(__import__('models.dreamer_v4', fromlist=['dummy']).__file__).read()
    assert 'DREAMER_ACT_HIST_REQUIRED' not in _dv4
    assert 'action_history is None' in _dv4
    assert 'DREAMER_GAIN_MATCH_SETTLE_LEN' in ENV_OVERRIDES
    assert 'DREAMER_GAIN_MATCH_REST_IC' in ENV_OVERRIDES
    assert 'DREAMER_GAIN_MATCH_REST_IC_LEN' in ENV_OVERRIDES
    assert 'DREAMER_P3_RESET_LOG_STD' in ENV_OVERRIDES
    assert 'DREAMER_P3_STOP_GRAD_LOG_STD' in ENV_OVERRIDES
    assert 'DREAMER_P3_LOGP_CLIP' in ENV_OVERRIDES
    assert 'DREAMER_P3_MU_RATIO_CLIP' in ENV_OVERRIDES
    assert 'DREAMER_P3_MU_RATIO_REFRESH' in ENV_OVERRIDES
    assert 'DREAMER_ES_ENT_FLOOR_FRAC' in ENV_OVERRIDES
    assert 'DREAMER_BC_MEAN_ONLY' in ENV_OVERRIDES
    assert 'DREAMER_OBJ_AUTO_MV_OVER_CV_RATIO' in ENV_OVERRIDES
    assert 'DREAMER_RUNTIME_SETPOINT_BOUNDS_JITTER_FRAC' in ENV_OVERRIDES
    assert 'DREAMER_RUNTIME_SETPOINT_BOUNDS_CHANGES_MAX' in ENV_OVERRIDES
    assert 'DREAMER_RUNTIME_SETPOINT_RAMP_DURATION_FRAC' in ENV_OVERRIDES
    assert 'DREAMER_OBJ_USE_NORMALIZED' in ENV_OVERRIDES
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
    assert 'DREAMER_WM_DIAG_DEVICE' in ENV_OVERRIDES
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
    assert c.hidden_dist_spread is True
    assert 'DREAMER_HIDDEN_DIST_SPREAD' in ENV_OVERRIDES
    assert abs(float(c.horizon_settle_n_tau) - 4.0) < 1e-12
    assert int(c.horizon_max) == 120
    assert abs(float(c.episode_settle_multiple) - 20.0) < 1e-12
    assert int(c.episode_min_length) == 500
    assert int(c.episode_max_length) == 4000
    assert c.init_randomization is True
    assert abs(float(c.init_randomization_frac) - 0.6) < 1e-12
    assert abs(float(c.wm_overhead) - 1.30) < 1e-12
    assert abs(float(c.gpu_target_util) - 0.80) < 1e-12
    assert int(c.gpu_max_bs) == 512
    assert 'DREAMER_HORIZON_SETTLE_NTAU' in ENV_OVERRIDES
    assert 'DREAMER_HORIZON_MAX' in ENV_OVERRIDES
    assert 'DREAMER_EPISODE_SETTLE_MULTIPLE' in ENV_OVERRIDES
    assert 'DREAMER_EPISODE_MIN_LENGTH' in ENV_OVERRIDES
    assert 'DREAMER_EPISODE_MAX_LENGTH' in ENV_OVERRIDES
    assert 'DREAMER_INIT_RANDOMIZATION' in ENV_OVERRIDES
    assert 'DREAMER_INIT_RANDOMIZATION_FRAC' in ENV_OVERRIDES
    assert 'DREAMER_WM_OVERHEAD' in ENV_OVERRIDES
    assert 'DREAMER_TARGET_UTIL' in ENV_OVERRIDES
    assert 'DREAMER_MAX_BS' in ENV_OVERRIDES
    assert 'DREAMER_BATCH_SIZE' in ENV_OVERRIDES
    assert c.sim_noise_adaptive is True
    assert abs(float(c.sim_ou_sigma_frac) - 0.008) < 1e-12
    assert abs(float(c.sim_ou_gain_cv) - 0.15) < 1e-12
    assert abs(float(c.sim_ou_gain_dv) - 0.60) < 1e-12
    assert abs(float(c.sim_meas_noise_cv_frac) - 0.005) < 1e-12
    assert abs(float(c.sim_meas_noise_dv_frac) - 0.010) < 1e-12
    assert 'DREAMER_SIM_NOISE_ADAPTIVE' in ENV_OVERRIDES
    assert 'DREAMER_SIM_OU_SIGMA_FRAC' in ENV_OVERRIDES
    assert 'DREAMER_SIM_OU_GAIN_CV' in ENV_OVERRIDES
    assert 'DREAMER_SIM_OU_GAIN_DV' in ENV_OVERRIDES
    assert 'DREAMER_SIM_MEAS_NOISE_CV_FRAC' in ENV_OVERRIDES
    assert 'DREAMER_SIM_MEAS_NOISE_DV_FRAC' in ENV_OVERRIDES
    assert c.sim_noise_enabled is True
    assert c.sim_noise_seed == ''
    assert abs(float(c.sim_noise_jitter_pct) - 0.20) < 1e-12
    assert c.sim_domain_randomization is True
    assert c.sim_domain_randomization_seed == ''
    assert abs(float(c.sim_param_randomization_pct) + 1.0) < 1e-12
    assert 'DREAMER_SIM_NOISE_ENABLED' in ENV_OVERRIDES
    assert 'DREAMER_SIM_NOISE_SEED' in ENV_OVERRIDES
    assert 'DREAMER_SIM_NOISE_JITTER_PCT' in ENV_OVERRIDES
    assert 'DREAMER_SIM_DOMAIN_RANDOMIZATION' in ENV_OVERRIDES
    assert 'DREAMER_SIM_DOMAIN_RANDOMIZATION_SEED' in ENV_OVERRIDES
    assert 'DREAMER_SIM_PARAM_RANDOMIZATION_PCT' in ENV_OVERRIDES
    assert abs(float(c.disturbance_authority_frac) - 0.65) < 1e-12
    assert abs(float(c.disturbance_recovery_frac) - 0.20) < 1e-12
    assert int(c.disturbance_settle_steps) == 0
    assert abs(float(c.disturbance_quiet_frac) - 0.12) < 1e-12
    assert abs(float(c.identified_tau_dominant)) < 1e-12
    assert abs(float(c.identified_dead_time)) < 1e-12
    assert abs(float(c.objective_integral_coef) - 0.05) < 1e-12
    assert abs(float(c.objective_integral_windup) - 5.0) < 1e-12
    assert abs(float(c.objective_integral_leak) - 0.98) < 1e-12
    assert c.obj_auto_integral_soft_compensate is True
    assert abs(float(c.obj_auto_cv_over_econ_ratio)) < 1e-12
    assert abs(float(c.objective_penalty_clip) + 1.0) < 1e-12
    assert c.objective_penalty_sat_mode == 'tanh'
    assert c.objective_violation_rate_coef == 'auto'
    assert 'DREAMER_OBJECTIVE_INTEGRAL_COEF' in ENV_OVERRIDES
    assert 'DREAMER_OBJECTIVE_PENALTY_CLIP' in ENV_OVERRIDES
    assert 'DREAMER_OBJ_AUTO_INTEGRAL_SOFT_COMPENSATE' in ENV_OVERRIDES
    assert 'DREAMER_DISTURBANCE_AUTHORITY_FRAC' in ENV_OVERRIDES
    assert 'DREAMER_DISTURBANCE_RECOVERY_FRAC' in ENV_OVERRIDES
    assert 'DREAMER_DISTURBANCE_SETTLE_STEPS' in ENV_OVERRIDES
    assert 'DREAMER_DISTURBANCE_QUIET_FRAC' in ENV_OVERRIDES
    assert abs(float(c.expert_move_frac) - 0.30) < 1e-12
    assert abs(float(c.expert_backoff_frac) - 0.12) < 1e-12
    assert not hasattr(c, 'pmpo_alpha')
    assert not hasattr(c, 'pmpo_beta')
    assert c.wm_diag_device == 'cuda'
    assert int(c.dv_prbs_seed_episodes) == 24
    assert 'DREAMER_EXPERT_MOVE_FRAC' in ENV_OVERRIDES
    assert 'DREAMER_PMPO_ALPHA' not in ENV_OVERRIDES
    assert 'DREAMER_PMPO_BETA' not in ENV_OVERRIDES
    assert 'DREAMER_P3_PRIOR_REFRESH_ITERS' not in ENV_OVERRIDES
    assert 'DREAMER_JOINT_PRIOR_REFRESH_ITERS' not in ENV_OVERRIDES
    assert 'DREAMER_WM_DIAG_DEVICE' in ENV_OVERRIDES
    assert 'DREAMER_BASELINE_SEED_EPS' in ENV_OVERRIDES
    assert 'DREAMER_EXPLORATION_SEED_EPS' in ENV_OVERRIDES
    assert 'DREAMER_DV_PRBS_SEEDS' in ENV_OVERRIDES
    assert 'DREAMER_GRAD_CLIP' in ENV_OVERRIDES
    assert 'DREAMER_POLICY_INIT_LOG_STD' in ENV_OVERRIDES
    assert c.obj_reward_scale == 'auto'
    assert c.attn_impl == 'auto'
    assert abs(float(c.sigma_min_ratio) - 1.2) < 1e-12
    assert 'DREAMER_GAIN_MATCH_RELATIVE' not in ENV_OVERRIDES
    assert 'DREAMER_ACTOR_KL_COEF' not in ENV_OVERRIDES
    assert not hasattr(c, 'actor_kl_coef')
    assert 'DREAMER_GAIN_MATCH_HUBER_PER_INPUT' in ENV_OVERRIDES
    assert 'DREAMER_GAIN_MATCH_CLIP_REALIZED' in ENV_OVERRIDES
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


def _test_identified_tau_cfg() -> None:
    """Plant τ/θ on TrainConfig; APCEnv cache; leftover env only when unset."""
    import os
    c = TrainConfig()
    c.identified_tau_dominant = 53.0
    c.identified_dead_time = 8.0
    env = APCEnv.__new__(APCEnv)
    env.cfg = c
    env.sim = None
    prev_tau = os.environ.get('IDENTIFIED_TAU_DOMINANT')
    prev_dead = os.environ.get('IDENTIFIED_DEAD_TIME')
    try:
        os.environ['IDENTIFIED_TAU_DOMINANT'] = '99'
        os.environ['IDENTIFIED_DEAD_TIME'] = '1'
        tau, dead = env._resolve_plant_timing()
        assert abs(tau - 53.0) < 1e-12, tau
        assert abs(dead - 8.0) < 1e-12, dead
        tau2, dead2 = env._resolve_plant_timing()
        assert (tau2, dead2) == (tau, dead)
        c0 = TrainConfig()
        env0 = APCEnv.__new__(APCEnv)
        env0.cfg = c0
        env0.sim = None
        tau0, dead0 = env0._resolve_plant_timing()
        assert abs(tau0 - 99.0) < 1e-12, tau0
        assert abs(dead0 - 1.0) < 1e-12, dead0
    finally:
        if prev_tau is None:
            os.environ.pop('IDENTIFIED_TAU_DOMINANT', None)
        else:
            os.environ['IDENTIFIED_TAU_DOMINANT'] = prev_tau
        if prev_dead is None:
            os.environ.pop('IDENTIFIED_DEAD_TIME', None)
        else:
            os.environ['IDENTIFIED_DEAD_TIME'] = prev_dead
    print('[smoke] OK  identified tau/dead_time TrainConfig beats leftover env')


def _test_objective_runtime_cfg() -> None:
    """Reward-engine leftovers: TrainConfig default, leftover OBJECTIVE_*, DREAMER wins."""
    import os
    from utils.objective_runtime import (
        resolve_integral_config, resolve_integral_leak,
        _maybe_auto_weights, _AUTO_W_CACHE,
    )
    from workflow._plant_prepare import ENV_OVERRIDES

    c = TrainConfig()
    assert abs(float(c.objective_integral_coef) - 0.05) < 1e-12
    assert abs(float(c.objective_integral_leak) - 0.98) < 1e-12
    assert c.obj_auto_integral_soft_compensate is True
    assert abs(float(c.objective_penalty_clip) + 1.0) < 1e-12
    assert 'DREAMER_OBJECTIVE_INTEGRAL_COEF' in ENV_OVERRIDES
    keys = (
        'DREAMER_OBJECTIVE_INTEGRAL_COEF', 'OBJECTIVE_INTEGRAL_COEF',
        'DREAMER_OBJECTIVE_INTEGRAL_LEAK', 'OBJECTIVE_INTEGRAL_LEAK',
        'DREAMER_OBJ_AUTO_INTEGRAL_SOFT_COMPENSATE',
        'OBJ_AUTO_INTEGRAL_SOFT_COMPENSATE',
        'DREAMER_OBJ_AUTO_VIOLATION_MARGIN', 'OBJ_AUTO_VIOLATION_MARGIN',
        'DREAMER_OBJ_AUTO_CV_OVER_ECON_RATIO', 'OBJ_AUTO_CV_OVER_ECON_RATIO',
        'DREAMER_OBJECTIVE_PENALTY_CLIP', 'OBJECTIVE_PENALTY_CLIP',
        'IDENTIFIED_TAU_DOMINANT', 'IDENTIFIED_DEAD_TIME',
    )
    prev = {k: os.environ.get(k) for k in keys}
    try:
        for k in keys:
            os.environ.pop(k, None)
        on, coef, windup = resolve_integral_config()
        assert on is True
        assert abs(coef - 0.05) < 1e-12, coef
        assert abs(windup - 5.0) < 1e-12
        assert abs(resolve_integral_leak() - 0.98) < 1e-12
        on_c, coef_c, _ = resolve_integral_config(cfg=c)
        assert abs(coef_c - 0.05) < 1e-12
        os.environ['OBJECTIVE_INTEGRAL_COEF'] = '0.11'
        _, coef_l, _ = resolve_integral_config(cfg=c)
        assert abs(coef_l - 0.11) < 1e-12, coef_l
        os.environ['DREAMER_OBJECTIVE_INTEGRAL_COEF'] = '0.07'
        _, coef_d, _ = resolve_integral_config(cfg=c)
        assert abs(coef_d - 0.07) < 1e-12, coef_d
        c_ex = TrainConfig()
        c_ex.objective_integral_coef = 0.03
        c_ex._explicit_fields = {'objective_integral_coef'}  # type: ignore
        _, coef_e, _ = resolve_integral_config(cfg=c_ex)
        assert abs(coef_e - 0.03) < 1e-12, coef_e
        os.environ.pop('DREAMER_OBJECTIVE_INTEGRAL_COEF', None)
        os.environ.pop('OBJECTIVE_INTEGRAL_COEF', None)
        os.environ['OBJECTIVE_INTEGRAL_LEAK'] = '0.90'
        assert abs(resolve_integral_leak() - 0.90) < 1e-12
        os.environ['DREAMER_OBJECTIVE_INTEGRAL_LEAK'] = '0.85'
        assert abs(resolve_integral_leak(cfg=c) - 0.85) < 1e-12
        # Ratio sentinel 0 follows leftover margin (historical get(..., str(margin))).
        os.environ.pop('DREAMER_OBJECTIVE_INTEGRAL_LEAK', None)
        os.environ.pop('OBJECTIVE_INTEGRAL_LEAK', None)
        os.environ['OBJ_AUTO_VIOLATION_MARGIN'] = '4.0'
        os.environ['OBJ_AUTO_CV_OVER_ECON_RATIO'] = '1.0'
        os.environ['OBJ_AUTO_INTEGRAL_SOFT_COMPENSATE'] = '1'
        os.environ['OBJ_AUTO_INTEGRAL_DEADTIME_K'] = '0'
        _, coef_b, _ = resolve_integral_config()
        assert abs(coef_b - 0.20) < 1e-9, coef_b  # 0.05 * min(4/1, 10)
        # Cache: second derive with same empty obj_w + bounds is the same object.
        _AUTO_W_CACHE.clear()
        ow = {}
        a = _maybe_auto_weights(
            ow, n_mv=1, n_cv=1, spec={},
            mv_bounds=[[20.0, 80.0]], cv_bounds=[[78.5, 85.5]],
            mv_norm_ranges=[[20.0, 80.0]], cv_norm_ranges=[[78.5, 85.5]])
        b = _maybe_auto_weights(
            ow, n_mv=1, n_cv=1, spec={},
            mv_bounds=[[20.0, 80.0]], cv_bounds=[[78.5, 85.5]],
            mv_norm_ranges=[[20.0, 80.0]], cv_norm_ranges=[[78.5, 85.5]])
        assert a is b
        assert a.get('mv_violation_weights')
    finally:
        for k, old in prev.items():
            if old is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = old
        _AUTO_W_CACHE.clear()
    print('[smoke] OK  objective leftover TrainConfig + DREAMER beats OBJECTIVE_*')


def _test_auto_weights_cfg() -> None:
    """Auto-weights leftover OBJ_AUTO_*: TrainConfig default, leftover, DREAMER, explicit."""
    import os
    from utils.auto_weights import derive_auto_weights
    from utils.runtime_setpoints import RuntimeSetpointConfig
    from workflow._plant_prepare import ENV_OVERRIDES

    c = TrainConfig()
    assert abs(float(c.obj_auto_mv_over_cv_ratio) - 2.0) < 1e-12
    assert abs(float(c.obj_auto_typical_cv_violation) - 0.10) < 1e-12
    assert abs(float(c.obj_auto_differentiable_depth) - 0.20) < 1e-12
    assert abs(float(c.runtime_setpoint_bounds_jitter_frac) - 0.15) < 1e-12
    assert abs(float(c.runtime_setpoint_target_jitter_frac) - 0.20) < 1e-12
    assert c.objective_use_normalized is True
    # APCEnv path uses dataclass jitter / schedule, not τ-derived auto_derive.
    rs = RuntimeSetpointConfig()
    assert abs(float(rs.bounds_jitter_fraction) - 0.15) < 1e-12
    assert abs(float(rs.target_jitter_fraction) - 0.20) < 1e-12
    assert rs.bounds_changes_per_episode == (1, 2)
    assert 'DREAMER_OBJ_AUTO_MV_OVER_CV_RATIO' in ENV_OVERRIDES
    assert 'DREAMER_RUNTIME_SETPOINT_BOUNDS_JITTER_FRAC' in ENV_OVERRIDES
    assert 'DREAMER_RUNTIME_SETPOINT_BOUNDS_CHANGES_MAX' in ENV_OVERRIDES
    assert 'DREAMER_OBJ_USE_NORMALIZED' in ENV_OVERRIDES
    kw = dict(
        spec={}, n_mv=1, n_cv=1,
        mv_bounds=[[20.0, 80.0]], cv_bounds=[[78.5, 85.5]],
        mv_norm_ranges=[[20.0, 80.0]], cv_norm_ranges=[[78.5, 85.5]])
    keys = (
        'DREAMER_OBJ_AUTO_MV_OVER_CV_RATIO', 'OBJ_AUTO_MV_OVER_CV_RATIO',
        'DREAMER_OBJ_USE_NORMALIZED', 'OBJ_USE_NORMALIZED',
        'RUNTIME_SETPOINT_BOUNDS_JITTER_FRACTION',
    )
    prev = {k: os.environ.get(k) for k in keys}
    try:
        for k in keys:
            os.environ.pop(k, None)
        w0 = derive_auto_weights(**kw)
        w1 = derive_auto_weights(cfg=c, **kw)
        assert w0['mv_violation_weights'] == w1['mv_violation_weights']
        os.environ['OBJ_AUTO_MV_OVER_CV_RATIO'] = '4.0'
        w_l = derive_auto_weights(cfg=c, **kw)
        assert float(w_l['mv_violation_weights'][0]) > float(w0['mv_violation_weights'][0])
        os.environ['DREAMER_OBJ_AUTO_MV_OVER_CV_RATIO'] = '2.0'
        w_d = derive_auto_weights(cfg=c, **kw)
        assert abs(float(w_d['mv_violation_weights'][0])
                   - float(w0['mv_violation_weights'][0])) < 1e-6
        c_ex = TrainConfig()
        c_ex.obj_auto_mv_over_cv_ratio = 1.0
        c_ex._explicit_fields = {'obj_auto_mv_over_cv_ratio'}  # type: ignore
        w_e = derive_auto_weights(cfg=c_ex, **kw)
        assert float(w_e['mv_violation_weights'][0]) < float(w0['mv_violation_weights'][0])
    finally:
        for k, old in prev.items():
            if old is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = old
    print('[smoke] OK  auto-weights TrainConfig + leftover OBJ_AUTO_* identity')


def _test_runtime_setpoint_schedule_cfg() -> None:
    """Operator-limit schedule: TrainConfig identity, DREAMER A/B, auto_derive jitter."""
    import os
    from utils.runtime_setpoints import RuntimeSetpointConfig
    from workflow._plant_prepare import ENV_OVERRIDES

    c = TrainConfig()
    rs = _runtime_setpoint_config_from_cfg(c)
    assert rs.bounds_changes_per_episode == (1, 2)
    assert rs.target_changes_per_episode == (1, 2)
    assert abs(float(rs.ramp_duration_fraction) - 0.10) < 1e-12
    assert abs(float(rs.curriculum_warmup_fraction) - 0.10) < 1e-12
    assert int(rs.n_magnitude_strata) == 3
    assert abs(float(rs.target_inside_margin_frac) - 0.05) < 1e-12
    assert abs(float(rs.bounds_jitter_fraction) - 0.15) < 1e-12
    assert abs(float(rs.target_jitter_fraction) - 0.20) < 1e-12
    assert 'DREAMER_RUNTIME_SETPOINT_BOUNDS_CHANGES_MAX' in ENV_OVERRIDES
    assert 'DREAMER_RUNTIME_SETPOINT_RAMP_DURATION_FRAC' in ENV_OVERRIDES
    keys = (
        'DREAMER_RUNTIME_SETPOINT_BOUNDS_CHANGES_MAX',
        'DREAMER_RUNTIME_SETPOINT_RAMP_DURATION_FRAC',
        'DREAMER_RUNTIME_SETPOINT_BOUNDS_JITTER_FRAC',
        'RUNTIME_SETPOINT_BOUNDS_JITTER_FRACTION',
        'RUNTIME_SETPOINT_TARGET_JITTER_FRACTION',
    )
    prev = {k: os.environ.get(k) for k in keys}
    try:
        for k in keys:
            os.environ.pop(k, None)
        ad = RuntimeSetpointConfig.auto_derive(
            targets_enabled=False, episode_length=1220,
            tau_dominant=53.0, dead_time=8.0, dt=1.0)
        assert abs(float(ad.bounds_jitter_fraction) - 0.15) < 1e-12
        assert abs(float(ad.target_jitter_fraction) - 0.20) < 1e-12
        os.environ['RUNTIME_SETPOINT_BOUNDS_JITTER_FRACTION'] = '0.28'
        ad2 = RuntimeSetpointConfig.auto_derive(
            targets_enabled=False, episode_length=1220)
        assert abs(float(ad2.bounds_jitter_fraction) - 0.28) < 1e-12
        os.environ['DREAMER_RUNTIME_SETPOINT_BOUNDS_JITTER_FRAC'] = '0.18'
        ad3 = RuntimeSetpointConfig.auto_derive(
            targets_enabled=False, episode_length=1220)
        assert abs(float(ad3.bounds_jitter_fraction) - 0.18) < 1e-12
        c_ex = TrainConfig()
        c_ex.runtime_setpoint_bounds_changes_max = 4
        c_ex._explicit_fields = {'runtime_setpoint_bounds_changes_max'}  # type: ignore[attr-defined]
        rs_ex = _runtime_setpoint_config_from_cfg(c_ex)
        assert rs_ex.bounds_changes_per_episode == (1, 4)
    finally:
        for k, old in prev.items():
            if old is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = old
    print('[smoke] OK  runtime-setpoint schedule TrainConfig + auto_derive jitter 0.15/0.20')


def _test_step_seed_shaping_prbs_seg_cfg() -> None:
    """Step-seed / shaping-safe / PRBS-seg / ss-window: TrainConfig identity + whitelist."""
    from workflow._plant_prepare import ENV_OVERRIDES

    c = TrainConfig()
    assert abs(float(c.step_seed_delta_min) - 0.20) < 1e-12
    assert abs(float(c.step_seed_delta_max) - 0.60) < 1e-12
    assert abs(float(c.step_seed_prefix_frac_min) - 0.05) < 1e-12
    assert abs(float(c.step_seed_prefix_frac_max) - 0.20) < 1e-12
    assert abs(float(c.shaping_safe_margin_frac) - 0.25) < 1e-12
    assert int(c.prbs_seed_segment_steps) == 0
    assert int(c.prbs_seed_segment_steps_min) == 0
    assert abs(float(c.wm_ss_match_window_frac) - 0.34) < 1e-12
    for k in (
        'DREAMER_STEP_SEED_DELTA_MIN',
        'DREAMER_STEP_SEED_DELTA_MAX',
        'DREAMER_STEP_SEED_PREFIX_FRAC_MIN',
        'DREAMER_STEP_SEED_PREFIX_FRAC_MAX',
        'DREAMER_SHAPING_SAFE_MARGIN_FRAC',
        'DREAMER_PRBS_SEED_SEGMENT_STEPS',
        'DREAMER_PRBS_SEED_SEGMENT_STEPS_MIN',
        'DREAMER_WM_SS_MATCH_WINDOW_FRAC',
    ):
        assert k in ENV_OVERRIDES, k
    print('[smoke] OK  step-seed / shaping-safe / PRBS-seg / ss-window TrainConfig + ENV_OVERRIDES')


def _test_expert_move_law_cfg() -> None:
    """Expert move-law leftovers: TrainConfig default, leftover env, explicit wins."""
    import os
    import numpy as np
    from utils.apc_expert import GainScheduleExpert
    from workflow._plant_prepare import ENV_OVERRIDES

    c = TrainConfig()
    assert abs(float(c.expert_move_frac) - 0.30) < 1e-12
    assert abs(float(c.expert_backoff_frac) - 0.12) < 1e-12
    assert abs(float(c.expert_econ_scale) - 1.0) < 1e-12
    assert int(c.expert_opt_iters) == 40
    assert 'DREAMER_EXPERT_MOVE_FRAC' in ENV_OVERRIDES
    assert 'DREAMER_EXPERT_OPT_ITERS' in ENV_OVERRIDES
    mv = np.array([[0.0, 1.0]])
    cv = np.array([[0.0, 1.0]])
    G = np.array([[-0.28]])
    kw = dict(
        mv_bounds=mv, cv_bounds=cv,
        cv_priority_weight=np.ones(1), cv_side_lo=np.ones(1),
        cv_side_hi=np.ones(1), mv_econ_sign=np.ones(1),
        anchors=[(np.array([0.5]), G)],
    )
    keys = ('DREAMER_EXPERT_MOVE_FRAC', 'DREAMER_EXPERT_BACKOFF_FRAC')
    prev = {k: os.environ.get(k) for k in keys}
    try:
        for k in keys:
            os.environ.pop(k, None)
        e0 = GainScheduleExpert(cfg=c, **kw)
        assert abs(e0.move_frac - 0.30) < 1e-12
        assert abs(e0.backoff_frac - 0.12) < 1e-12
        os.environ['DREAMER_EXPERT_MOVE_FRAC'] = '0.22'
        e1 = GainScheduleExpert(cfg=c, **kw)
        assert abs(e1.move_frac - 0.22) < 1e-12
        c_ex = TrainConfig()
        c_ex.expert_move_frac = 0.11
        c_ex._explicit_fields = {'expert_move_frac'}  # type: ignore[attr-defined]
        e2 = GainScheduleExpert(cfg=c_ex, **kw)
        assert abs(e2.move_frac - 0.11) < 1e-12
    finally:
        for k, old in prev.items():
            if old is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = old
    print('[smoke] OK  expert move-law TrainConfig + leftover env identity')


def _test_pin_eval_modules() -> None:
    """Launch pin binds validate/TM/training_disturbance; second call is a no-op."""
    import sys
    from workflow._plant_prepare import pin_eval_modules_at_launch
    pin_eval_modules_at_launch()
    assert 'evaluation.validate' in sys.modules
    assert 'evaluation.wm_transfer_matrix' in sys.modules
    assert 'utils.training_disturbance' in sys.modules
    pin_eval_modules_at_launch()
    print('[smoke] OK  pin eval modules at launch (P47/P48/P49 race)')


def _test_control_quality_gates() -> None:
    """Empty scripted pairs must not 0-vs-0 pass (P49 false all_pass)."""
    from evaluation.validate import control_quality_gates
    empty = control_quality_gates([])
    assert empty['beats_baseline_pass'] is False
    assert empty['n_scripted_pairs'] == 0
    assert empty.get('control_gate_skipped') == 'no_scripted_disturbance_pairs'
    seeded = control_quality_gates(
        [],
        seed_metrics=[{'kpi_mv_reversal_rate': 0.1, 'kpi_economic_score': -700.0}],
    )
    assert seeded['beats_baseline_pass'] is False
    assert abs(float(seeded['agent_economic_score']) + 700.0) < 1e-9
    assert seeded['smooth_pass'] is True
    paired = control_quality_gates([{
        'episode_metrics_agent': {'mv_reversal_rate': 0.1, 'economic_score': -50.0},
        'episode_metrics_baseline': {'economic_score': -90.0},
    }])
    assert paired['beats_baseline_pass'] is True
    assert abs(float(paired['agent_economic_score']) + 50.0) < 1e-9
    worse = control_quality_gates([{
        'episode_metrics_agent': {'mv_reversal_rate': 0.1, 'economic_score': -200.0},
        'episode_metrics_baseline': {'economic_score': -90.0},
    }])
    assert worse['beats_baseline_pass'] is False
    print('[smoke] OK  control_quality_gates empty records do not 0-vs-0 pass')


def _test_require_realsim_actor() -> None:
    """Imagination actor_train_source is a false A/B (p01 off-policy chatter).

    ``DREAMER_ACTOR_LOSS=pmpo`` is the same class: P3 always inlines
    REINFORCE + μ-ratio (``pmpo_loss`` REMOVED; never had a P3 call site).
    """
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
    pmpo = TrainConfig()
    pmpo.actor_loss_type = 'pmpo'
    try:
        _require_realsim_actor(pmpo)
    except RuntimeError as exc:
        assert 'pmpo' in str(exc).lower()
        assert 'false a/b' in str(exc).lower()
    else:
        raise AssertionError('actor_loss_type=pmpo was allowed')
    import models.dreamer_v4 as _dv4
    assert not hasattr(_dv4, 'pmpo_loss'), 'dead PMPO actor loss still exported'
    assert not hasattr(_dv4, 'reinforce_actor_loss'), (
        'dead imagination REINFORCE actor loss still exported')
    from models.dreamer_v4 import ContinuousPolicyHead, PolicyHead
    assert not hasattr(PolicyHead, 'kl_to'), 'dead PMPO PolicyHead.kl_to'
    assert not hasattr(ContinuousPolicyHead, 'kl_to'), (
        'dead PMPO ContinuousPolicyHead.kl_to')
    print('[smoke] OK  non-realsim actor_train_source / pmpo actor_loss refused')


def _test_rssm_param_grad_snapshot() -> None:
    """CUDA-graph canary must restore in-flight grads (identity when None)."""
    import torch.nn as nn
    m = nn.Linear(2, 2, bias=False)
    snap0 = _rssm_param_grad_snapshot(m)
    assert all(g is None for _, g in snap0)
    (m(torch.ones(1, 2)).sum()).backward()
    g0 = m.weight.grad.detach().clone()
    snap = _rssm_param_grad_snapshot(m)
    m.zero_grad(set_to_none=True)
    assert m.weight.grad is None
    _rssm_param_grad_restore(snap)
    assert torch.equal(m.weight.grad, g0)
    _rssm_param_grad_restore(snap0)
    assert m.weight.grad is None
    print('[smoke] OK  rest-IC CUDA-graph canary grad snapshot restore')


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
    owner = TrainConfig()
    r1 = _gain_match_pred_over_tgt(mv, ((-2.624,),), owner)
    r2 = _gain_match_pred_over_tgt(mv, ((-2.624,),), owner)
    assert abs(float(r1) - 1.0) < 1e-6 and abs(float(r2) - 1.0) < 1e-6
    t1 = _gain_match_tgt_tensor(mv, ((-2.624,),), owner)
    t2 = _gain_match_tgt_tensor(mv, ((-2.624,),), owner)
    assert t1 is t2
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


def _test_gain_match_clip_realized() -> None:
    """P61: interior rest+step is identity; rail reverses; Huber uses |Δu|."""
    torch.manual_seed(0)
    Bm, n_mv, n_dv, step = 4, 1, 1, 0.4
    a_int = torch.linspace(-0.5, 0.5, Bm).unsqueeze(-1)
    dv_int = torch.linspace(-0.3, 0.3, Bm).unsqueeze(-1)
    raw_a, raw_dv = _gain_match_fd_held(
        a_int, dv_int, n_mv, n_dv, step, clip_realized=False)
    clip_a, clip_dv = _gain_match_fd_held(
        a_int, dv_int, n_mv, n_dv, step, clip_realized=True)
    assert torch.allclose(raw_a, clip_a), (raw_a - clip_a).abs().max()
    assert torch.allclose(raw_dv, clip_dv), (raw_dv - clip_dv).abs().max()
    du = _gain_match_realized_du(clip_a, clip_dv, n_mv, n_dv)
    assert du.shape == (n_mv + n_dv, Bm), du.shape
    assert torch.allclose(du, torch.full_like(du, step)), du
    # MIMO: cat (n_mv,Bm)+(n_dv,Bm), not stack (only works n_mv==n_dv).
    a2 = torch.linspace(-0.4, 0.4, Bm).unsqueeze(-1).expand(Bm, 2).contiguous()
    dv1 = torch.linspace(-0.2, 0.2, Bm).unsqueeze(-1)
    held_m, held_d = _gain_match_fd_held(
        a2, dv1, 2, 1, step, clip_realized=True)
    du_m = _gain_match_realized_du(held_m, held_d, 2, 1)
    assert du_m.shape == (3, Bm), du_m.shape
    assert torch.allclose(du_m, torch.full_like(du_m, step)), du_m
    a_rail = torch.full((Bm, 1), 0.9)
    dv_rail = torch.full((Bm, 1), 0.85)
    held_a, held_dv = _gain_match_fd_held(
        a_rail, dv_rail, n_mv, n_dv, step, clip_realized=True)
    # +0.4 from 0.9 would clip to 1.0 (Δ=0.1); reverse → 0.5 (Δ=-0.4).
    assert torch.allclose(held_a[1, :, 0], torch.full((Bm,), 0.5))
    assert torch.allclose(held_dv[2, :, 0], torch.full((Bm,), 0.45))
    du_r = _gain_match_realized_du(held_a, held_dv, n_mv, n_dv)
    assert torch.allclose(du_r.abs(), torch.full_like(du_r, step)), du_r
    assert float((du_r < 0).all()) == 1.0
    raw_rail_a, _ = _gain_match_fd_held(
        a_rail, dv_rail, n_mv, n_dv, step, clip_realized=False)
    assert float(raw_rail_a[1, :, 0].max()) > 1.0
    rows = _cube_step_held(a_rail, 1, step)
    assert rows.shape == (1, Bm, 1)
    assert torch.allclose(rows[0, :, 0], torch.full((Bm,), 0.5))
    # Reverse predicate ≡ held reverse; du_frac cannot see this.
    mask_int = _cube_plus_would_clip(a_int, 1, step)
    assert not bool(mask_int.any())
    mask_rail = _cube_plus_would_clip(a_rail, 1, step)
    assert bool(mask_rail.all())
    cf_int = _gain_match_clip_frac_t(
        a_int, dv_int, n_mv, n_dv, step, clip_realized=True)
    cf_off = _gain_match_clip_frac_t(
        a_rail, dv_rail, n_mv, n_dv, step, clip_realized=False)
    cf_rail = _gain_match_clip_frac_t(
        a_rail, dv_rail, n_mv, n_dv, step, clip_realized=True)
    assert float(cf_int) == 0.0
    assert float(cf_off) == 0.0
    assert abs(float(cf_rail) - 1.0) < 1e-6
    print('[smoke] OK  gain-match clip+realized (interior identity; rail reverse)')


def _test_gain_match_fd_action_seq() -> None:
    """Rest-IC FD a_seq cache is identity with the uncached expand."""
    torch.manual_seed(0)
    Bm, n_mv, n_dv, step, K = 4, 2, 1, 1.0, 5
    a_base = torch.randn(Bm, n_mv)
    dv0 = torch.randn(Bm, n_dv)
    a_held, dv_held = _gain_match_fd_held(a_base, dv0, n_mv, n_dv, step)
    n_rolls = int(a_held.shape[0])
    ref_a = (a_held.unsqueeze(2).expand(n_rolls, Bm, K, n_mv)
             .reshape(n_rolls * Bm, K, n_mv).contiguous())
    ref_dv = (dv_held.unsqueeze(2).expand(n_rolls, Bm, K, n_dv)
              .reshape(n_rolls * Bm, K, n_dv).contiguous())

    class _Own:
        pass

    own = _Own()
    key = ('rest', n_mv, n_dv, step, K, Bm)
    a1, d1, du1, cf1 = _gain_match_fd_action_seq(
        a_base, dv0, n_mv, n_dv, step, K, Bm,
        cache_owner=own, cache_key=key)
    a2, d2, du2, cf2 = _gain_match_fd_action_seq(
        a_base, dv0, n_mv, n_dv, step, K, Bm,
        cache_owner=own, cache_key=key)
    assert a1 is a2 and d1 is d2 and du1 is du2 and cf1 is cf2
    assert torch.allclose(a1, ref_a)
    assert torch.allclose(d1, ref_dv)
    du_ref = _gain_match_realized_du(a_held, dv_held, n_mv, n_dv)
    assert torch.allclose(du1, du_ref)
    a3, d3, du3, cf3 = _gain_match_fd_action_seq(
        a_base, dv0, n_mv, n_dv, step, K, Bm)
    a4, d4, du4, cf4 = _gain_match_fd_action_seq(
        a_base, dv0, n_mv, n_dv, step, K, Bm)
    assert a3 is not a4
    assert torch.allclose(a3, ref_a) and torch.allclose(d3, ref_dv)
    assert torch.allclose(du3, du_ref) and torch.allclose(du4, du_ref)
    assert float(cf1) == 0.0 and float(cf3) == 0.0
    print('[smoke] OK  gain-match FD action seq cache (rest-IC identity)')


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
    """Collect settle is max(H, lookback); default L=lookback (P57 REVERT)."""
    from evaluation.wm_transfer_matrix import wm_tf_horizon
    c = TrainConfig()
    c.horizon = 55
    c.lookback = 128
    # tau=0 → legacy L=lookback (no fake 50 s).
    s, L = _gain_match_rest_window(c)
    assert (s, L) == (128, 128), (s, L)
    assert s != wm_tf_horizon(55)
    c.lookback = 32
    assert _gain_match_rest_window(c) == (55, 32)
    c.horizon = 15
    c.lookback = 8
    assert _gain_match_rest_window(c) == (15, 8)
    # test_sim: τ=53, sr=4, K=55 → L=max(55, round(2*53/4)=27)=55.
    c.horizon = 55
    c.lookback = 128
    c.gain_match_len = 55
    c.sample_rate = 4
    c.identified_tau_dominant = 53.0
    assert int(c.gain_match_rest_ic_len) == 0
    assert _gain_match_rest_window(c) == (128, 128), _gain_match_rest_window(c)
    c.gain_match_rest_ic_len = -1
    assert _gain_match_rest_window(c) == (128, 55), _gain_match_rest_window(c)
    c.gain_match_rest_ic_len = 0
    assert _gain_match_rest_window(c) == (128, 128)
    c.gain_match_rest_ic_len = 16
    assert _gain_match_rest_window(c) == (128, 16)
    print('[smoke] OK  rest-ic settle=max(H,lookback); default L=lookback; -1 A/B max(K,2τ/sr)')


def _test_held_rollout_win_fits_k() -> None:
    """win=8 is identity at test_sim K=55; clamp (K-1)/4 so fast plants are not 0."""
    assert _held_rollout_win(55, 8) == 8
    assert _held_rollout_win(55, 0) == 8
    assert _held_rollout_win(15, 8) == 3
    assert _held_rollout_win(15, 0) == 2
    assert _held_rollout_win(32, 8) == 7
    # Two windows [s, s+win) and [K-win, K) must not overlap.
    for k, w_req in ((55, 8), (15, 8), (15, 0), (32, 8)):
        w = _held_rollout_win(k, w_req)
        cap = max(1, (k - 1) // 4)
        assert 1 <= w <= min(k, cap), (k, w_req, w, cap)
        s = k // 2
        s = max(w, min(s, k - 2 * w))
        assert s >= w and k - w > s + w, (k, w, s)
    assert not hasattr(TrainConfig(), 'wm_held_rollout_settle_frac')
    print('[smoke] OK  held-rollout win clamps to (K-1)/4 (test_sim 8)')


def _test_held_rollout_cv_space() -> None:
    """P62: held is decode-CV late−early, not GRU h / P63 FO magnitude."""
    import inspect
    from models.dreamer_v4_rssm import RSSMConfig, RSSMDynamics

    src = inspect.getsource(_wm_held_rollout_stationarity_loss)
    assert "out='obs'" in src, 'held must roll decode, not out=h'
    assert 'cv_index_t' in src
    assert "out='h'" not in src
    assert 'K // 2' in src
    assert '_held_ol_fo_scale' not in src
    assert 'delta_1.detach()' not in src
    assert 'wm_held_ol_ratio' not in src

    class _Wrap:
        world_model_type = 'rssm'

        def __init__(self, dyn):
            self.dynamics = dyn

    torch.manual_seed(0)
    rcfg = RSSMConfig(obs_dim=4, action_dim=1, deter_dim=16,
                      n_categoricals=4, n_classes=4, embed_dim=16,
                      hidden_dim=16, latent_type='deterministic',
                      cont_gain_dim=2, cv_indices=(0,))
    dyn = RSSMDynamics(rcfg)
    model = _Wrap(dyn)
    B, T = 2, 16
    obs = torch.randn(B, T, rcfg.obs_dim)
    act = torch.rand(B, T, rcfg.action_dim) * 2 - 1
    feat_dim = dyn.deter_dim + dyn.stoch_flat_dim + dyn.cont_dim
    feats = torch.randn(B, T, feat_dim)
    cfg = TrainConfig()
    cfg.wm_held_rollout_coef = 0.5
    cfg.wm_held_rollout_len = 8
    cfg.wm_held_rollout_gate_recon = 0.0
    cfg.wm_held_rollout_max_starts = 2
    torch.manual_seed(0)
    hd0, n0, d0 = _wm_held_rollout_stationarity_loss(model, feats, obs, act, cfg)
    assert torch.isfinite(hd0).all() and float(hd0) >= 0.0 and n0 > 0
    assert float(d0['wm_held_rollout_scale']) >= 1e-3
    assert float(d0['wm_held_cv_drift']) >= 0.0
    assert 'wm_held_ol_ratio' not in d0
    assert float(hd0) < 50.0, f'held detonated at init ({float(hd0):.3f})'
    snap = {n: p.detach().clone() for n, p in dyn.decoder.named_parameters()}
    with torch.no_grad():
        for p in dyn.decoder.parameters():
            p.add_(1.0)
    torch.manual_seed(0)
    hd1, _, d1 = _wm_held_rollout_stationarity_loss(model, feats, obs, act, cfg)
    with torch.no_grad():
        for n, p in dyn.decoder.named_parameters():
            p.copy_(snap[n])
    assert abs(float(hd0) - float(hd1)) > 1e-8, (
        f'held ignored decoder ({float(hd0):.5f} vs {float(hd1):.5f})')
    assert abs(float(d0['wm_held_cv_drift']) - float(d1['wm_held_cv_drift'])) > 1e-8
    print(f'[smoke] OK  held decode-CV late−early '
          f'(loss {float(hd0):.4f}→{float(hd1):.4f} on decoder noise; '
          f'drift {float(d0["wm_held_cv_drift"]):.4f}→'
          f'{float(d1["wm_held_cv_drift"]):.4f})')


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
    st1 = _gain_match_rest_ic_state(
        model.dynamics, cfg, obs.device, obs.dtype)
    st2 = _gain_match_rest_ic_state(
        model.dynamics, cfg, obs.device, obs.dtype)
    assert st1 is not None and st2 is not None
    assert st1[3] is st2[3] and st1[4] is st2[4]
    cfg._gain_match_rest_obs = (rest_o + 1.5).numpy()
    cfg._gain_match_rest_dev = None
    # Leave `_gain_match_rest_adv` stale: new device tensors must bust it.
    st3 = _gain_match_rest_ic_state(
        model.dynamics, cfg, obs.device, obs.dtype)
    assert st3 is not None and st3[3] is not st1[3]
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
    wrap = _RestICGraphModule(model.dynamics)
    wrap.train(bool(model.dynamics.training))
    assert wrap.training == model.dynamics.training
    rp = list(model.dynamics.parameters())
    wp = list(wrap.parameters())
    assert rp and len(wp) == len(rp)
    assert all(a is b for a, b in zip(wp, rp))
    rnp = list(model.dynamics.named_parameters())
    wnp = list(wrap.named_parameters())
    assert [n for n, _ in wnp] == [n for n, _ in rnp]
    assert all(a is b for (_, a), (_, b) in zip(wnp, rnp))
    rnb = list(model.dynamics.named_buffers())
    wnb = list(wrap.named_buffers())
    assert [n for n, _ in wnb] == [n for n, _ in rnb]
    assert all(not b.requires_grad for _, b in wnb)
    rb = list(model.dynamics.buffers())
    wb = list(wrap.buffers())
    assert len(wb) == len(rb) and all(a is b for a, b in zip(wb, rb))
    assert wrap is not model.dynamics
    assert model.dynamics not in list(wrap.children())
    # P56: capture autograd.grad inputs = sample_args (no grad) + params.
    o_s, a_s = rest_o.detach(), rest_a.detach()
    surface = (o_s, a_s) + tuple(wrap.parameters())
    req = tuple(t for t in surface if t.requires_grad)
    assert req, 'P56: capture autograd.grad inputs would be empty'
    assert any('gru' in n for n, p in wrap.named_parameters() if p.requires_grad)
    h, z, c = wrap(rest_o, rest_a)
    h2, z2, c2 = _rest_ic_last_tensors(model.dynamics, rest_o, rest_a)
    assert torch.allclose(h, h2) and torch.allclose(z, z2)
    assert torch.allclose(c, c2)
    print('[smoke] OK  rest-ic graph module shares RSSM params/buffers (no re-parent)')
    # CUDA graph is skipped on CPU (and while a GPU job occupies the A10).
    assert not _rest_ic_can_cuda_graph(model.dynamics, rest_o, cfg)
    _warmup_rest_ic_cuda_graph(model.dynamics, cfg, torch.device('cpu'))
    assert not hasattr(model.dynamics, '_rest_ic_cg')
    assert _amp_parent_autocast_on('cpu') is False
    print('[smoke] OK  rest-ic CUDA graph skipped on CPU')
    r_miss = type('R', (), {})()
    _rest_ic_note_capture_miss(r_miss, False)
    assert not hasattr(r_miss, '_rest_ic_cg_fail')
    _rest_ic_note_capture_miss(r_miss, True)
    assert r_miss._rest_ic_cg_fail is True
    print('[smoke] OK  rest-ic VRAM skip does not pin eager-fail')
    r = type('R', (), {})()
    r._rest_ic_cg = ('k', None)
    r._rest_ic_cg_fail = True
    r._gain_match_fd_seq = ('seq', None, None)
    assert _release_rest_ic_cuda_graph(r) is True
    assert not hasattr(r, '_rest_ic_cg')
    assert not hasattr(r, '_rest_ic_cg_fail')
    assert not hasattr(r, '_gain_match_fd_seq')
    assert _release_rest_ic_cuda_graph(r) is False
    assert _release_rest_ic_cuda_graph(None) is False
    print('[smoke] OK  rest-ic CUDA graph release is identity on CPU')
    _arm_rest_ic_stream_mismatch_warn(True)
    _arm_rest_ic_stream_mismatch_warn(True)
    _arm_rest_ic_stream_mismatch_warn(False)
    _arm_rest_ic_stream_mismatch_warn(False)
    print('[smoke] OK  rest-ic AccumulateGrad warn arm/disarm is idempotent')


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


def _test_bc_mean_only() -> None:
    """P50: MSE-on-μ BC has zero log_std grad; NLL does not."""
    from models.dreamer_v4 import ContinuousPolicyHead
    torch.manual_seed(0)
    kwargs = dict(
        in_dim=8, hidden_dim=16, action_dim=2, n_layers=2, mtp_length=2,
        init_log_std=-1.5, log_std_min=-2.3, log_std_max=0.0)
    feat = torch.randn(6, 8)
    act = torch.tanh(torch.randn(6, 2, 2))
    pol_mse = ContinuousPolicyHead(**kwargs)
    last = pol_mse.head.net[-1]
    n = int(pol_mse.mtp_length) * int(pol_mse.action_dim)
    idx = torch.arange(n, device=last.weight.device) * 2 + 1
    mu_idx = torch.arange(n, device=last.weight.device) * 2
    det = pol_mse.mean_of_mtp(feat, L=2)
    mse = ((det - act) ** 2).mean()
    mse.backward()
    assert float(last.weight.grad[idx].abs().max()) < 1e-8, (
        float(last.weight.grad[idx].abs().max()))
    assert float(last.weight.grad[mu_idx].abs().max()) > 1e-6
    pol_nll = ContinuousPolicyHead(**kwargs)
    last_n = pol_nll.head.net[-1]
    nll = -pol_nll.log_prob_of_mtp(feat, act).mean()
    nll.backward()
    assert float(last_n.weight.grad[idx].abs().max()) > 1e-4, (
        float(last_n.weight.grad[idx].abs().max()))
    print('[smoke] OK  bc_mean_only MSE has zero log_std grad (NLL does not)')


def _test_p3_stop_grad_log_std() -> None:
    """P51: REINFORCE + η with stop_grad_log_std has zero log_std grad."""
    from models.dreamer_v4 import ContinuousPolicyHead
    torch.manual_seed(0)
    kwargs = dict(
        in_dim=8, hidden_dim=16, action_dim=2, n_layers=2, mtp_length=1,
        init_log_std=-1.5, log_std_min=-2.3, log_std_max=0.0)
    feat = torch.randn(6, 8)
    act = torch.tanh(torch.randn(6, 2))
    adv = torch.randn(6)
    n = 2
    idx = torch.arange(n) * 2 + 1
    mu_idx = torch.arange(n) * 2
    pol = ContinuousPolicyHead(**kwargs)
    last = pol.head.net[-1]
    logp = pol.log_prob_of(feat, act, stop_grad_log_std=True)
    ent = pol.entropy(feat, stop_grad_log_std=True)
    loss = -(adv * logp).mean() - 1e-4 * ent.mean()
    loss.backward()
    assert float(last.weight.grad[idx].abs().max()) < 1e-8, (
        float(last.weight.grad[idx].abs().max()))
    assert float(last.weight.grad[mu_idx].abs().max()) > 1e-6
    pol2 = ContinuousPolicyHead(**kwargs)
    last2 = pol2.head.net[-1]
    logp2 = pol2.log_prob_of(feat, act)
    ent2 = pol2.entropy(feat)
    loss2 = -(adv * logp2).mean() - 1e-4 * ent2.mean()
    loss2.backward()
    assert float(last2.weight.grad[idx].abs().max()) > 1e-4, (
        float(last2.weight.grad[idx].abs().max()))
    pol3 = ContinuousPolicyHead(**kwargs)
    logp3, ent3 = pol3.log_prob_and_entropy(
        feat, act, stop_grad_log_std=True)
    logp4 = pol3.log_prob_of(feat, act, stop_grad_log_std=True)
    ent4 = pol3.entropy(feat, stop_grad_log_std=True)
    assert torch.allclose(logp3, logp4, atol=1e-6, rtol=1e-6)
    assert torch.allclose(ent3, ent4, atol=1e-6, rtol=1e-6)
    print('[smoke] OK  p3_stop_grad_log_std zeros log_std REINFORCE grad')


def _test_p3_shared_ac_forwards() -> None:
    """P58 GPU-occupied: one critic MLP pass ≡ CE+min_v+MC; fused encode."""
    torch.manual_seed(0)
    cfg = TrainConfig()
    cfg.obs_dim = 4
    cfg.action_dim = 1
    cfg.lookback = 8
    cfg.world_model_type = 'rssm'
    cfg.n_critics = 2
    cfg.rssm_deter_dim = 32
    cfg.rssm_n_categoricals = 4
    cfg.rssm_n_classes = 4
    cfg.rssm_embed_dim = 16
    cfg.rssm_hidden_dim = 16
    cfg.head_hidden = 16
    cfg.head_n_layers = 2
    cfg.mtp_length = 1
    cfg.horizon = 4
    cfg.seq_len = 8
    cfg.d_model = 32
    cfg.rssm_latent_type = 'deterministic'
    cfg.dob_enabled = False
    cfg.dv_as_input = False
    model = build_model(cfg)
    feat = torch.randn(10, model.dynamics.feat_dim)
    ret = torch.randn(10)
    rmc = torch.randn(10)
    ce_ref = model.critic_ensemble_ce(feat, ret)
    v_ref = model.critic_min_v(feat, target=False)
    mc_ref = model.critic_ensemble_ce(feat, rmc)
    ce, v, mc = model.critic_online_ce_and_min_v(feat, ret, rmc)
    assert mc is not None
    assert torch.allclose(ce, ce_ref, atol=1e-5, rtol=1e-5), float(
        (ce - ce_ref).abs().max())
    assert torch.allclose(v, v_ref, atol=1e-5, rtol=1e-5), float(
        (v - v_ref).abs().max())
    assert torch.allclose(mc, mc_ref, atol=1e-5, rtol=1e-5), float(
        (mc - mc_ref).abs().max())
    print('[smoke] OK  critic_online_ce_and_min_v ≡ CE + min_v + MC')


def _test_p3_logp_clip() -> None:
    """P52: clamp REINFORCE logp zeros μ-grad on railed (u,μ); keeps in-support."""
    from models.dreamer_v4 import ContinuousPolicyHead
    torch.manual_seed(0)
    assert abs(_p3_logp_clip_bound(8.0, 1) - 8.0) < 1e-12
    assert abs(_p3_logp_clip_bound(8.0, 2) - 16.0) < 1e-12
    assert abs(_p3_logp_clip_bound(0.0, 4)) < 1e-12
    assert abs(_p3_logp_clip_bound(8.0, 0) - 8.0) < 1e-12
    kwargs = dict(
        in_dim=8, hidden_dim=16, action_dim=1, n_layers=2, mtp_length=1,
        init_log_std=-1.5, log_std_min=-2.3, log_std_max=0.0)
    feat = torch.zeros(4, 8)
    act = torch.ones(4, 1) * 0.999
    adv = torch.ones(4) * 8.0

    def _railed_policy():
        pol = ContinuousPolicyHead(**kwargs)
        with torch.no_grad():
            last = pol.head.net[-1]
            last.bias[0] = -6.0
            last.bias[1] = 0.0
        return pol

    pol_raw = _railed_policy()
    logp = pol_raw.log_prob_of(feat, act, stop_grad_log_std=True)
    assert float(logp.abs().mean().detach()) > 16.0, float(logp.abs().mean().detach())
    (-(adv * logp).mean()).backward()
    g_raw = float(pol_raw.head.net[-1].weight.grad[0].abs().max())
    pol_c = _railed_policy()
    logp_c = pol_c.log_prob_of(feat, act, stop_grad_log_std=True)
    bound = _p3_logp_clip_bound(8.0, 1)
    (-(adv * logp_c.clamp(-bound, bound)).mean()).backward()
    g_c = float(pol_c.head.net[-1].weight.grad[0].abs().max())
    assert g_c < 1e-8, g_c
    assert g_raw > 1e-3, g_raw
    pol_h = ContinuousPolicyHead(**kwargs)
    feat_h = torch.randn(6, 8)
    with torch.no_grad():
        mu_h, _ = pol_h.dist_params(feat_h)
        act_h = torch.tanh(mu_h + 0.25)
    adv_h = torch.ones(6)
    logp_h = pol_h.log_prob_of(feat_h, act_h, stop_grad_log_std=True)
    assert float(logp_h.abs().max().detach()) < bound, float(logp_h.abs().max().detach())
    (-(adv_h * logp_h.clamp(-bound, bound)).mean()).backward()
    assert float(pol_h.head.net[-1].weight.grad[0].abs().max()) > 1e-6
    print('[smoke] OK  p3_logp_clip zeros μ grad on railed logp; keeps in-support')


def _test_p3_mu_ratio_clip() -> None:
    """P53: PPO ratio clip vs frozen snapshot bounds μ-grad; ε=0 is REINFORCE."""
    from models.dreamer_v4 import ContinuousPolicyHead
    torch.manual_seed(0)
    assert abs(float(TrainConfig().p3_mu_ratio_clip) - 0.2) < 1e-12
    kwargs = dict(
        in_dim=8, hidden_dim=16, action_dim=1, n_layers=2, mtp_length=1,
        init_log_std=-1.5, log_std_min=-2.3, log_std_max=0.0)
    feat = torch.randn(8, 8)
    pol = ContinuousPolicyHead(**kwargs)
    with torch.no_grad():
        mu, _ = pol.dist_params(feat)
        act = torch.tanh(mu + 0.25)
    adv = torch.ones(8)
    logp = pol.log_prob_of(feat, act, stop_grad_log_std=True)
    surr0, ratio0, clip0 = _p3_mu_ratio_surrogate(logp, logp.detach(), adv, 0.0)
    assert torch.allclose(surr0, adv * logp)
    assert abs(float(ratio0.mean()) - 1.0) < 1e-12
    assert abs(float(clip0.mean())) < 1e-12
    (-surr0.mean()).backward(retain_graph=True)
    g0 = pol.head.net[-1].weight.grad[0].detach().clone()
    pol.zero_grad(set_to_none=True)
    (-(adv * logp).mean()).backward()
    g_rf = pol.head.net[-1].weight.grad[0].detach().clone()
    assert torch.allclose(g0, g_rf, atol=1e-6)

    pol2 = ContinuousPolicyHead(**kwargs)
    with torch.no_grad():
        mu2, _ = pol2.dist_params(feat)
        act2 = torch.tanh(mu2 + 0.25)
    logp2 = pol2.log_prob_of(feat, act2, stop_grad_log_std=True)
    surr_in, ratio_in, clip_in = _p3_mu_ratio_surrogate(
        logp2, logp2.detach(), adv, 0.2)
    assert float((ratio_in - 1.0).abs().max()) < 1e-5
    assert abs(float(clip_in.mean())) < 1e-12
    (-surr_in.mean()).backward()
    assert float(pol2.head.net[-1].weight.grad[0].abs().max()) > 1e-6

    torch.manual_seed(1)
    pol_old = ContinuousPolicyHead(**kwargs)
    torch.manual_seed(1)
    pol_new = ContinuousPolicyHead(**kwargs)
    torch.manual_seed(1)
    pol_raw = ContinuousPolicyHead(**kwargs)
    with torch.no_grad():
        pol_new.head.net[-1].bias[0] = 0.4
        pol_raw.head.net[-1].bias[0] = 0.4
        mu_n, _ = pol_new.dist_params(feat)
        act_n = torch.tanh(mu_n + 0.3)
    adv_n = torch.ones(feat.shape[0]) * 8.0
    logp_n = pol_new.log_prob_of(feat, act_n, stop_grad_log_std=True)
    with torch.no_grad():
        logp_o = pol_old.log_prob_of(feat, act_n, stop_grad_log_std=True)
    surr_w, ratio_w, clip_w = _p3_mu_ratio_surrogate(
        logp_n, logp_o, adv_n, 0.2)
    assert float(ratio_w.mean()) > 1.2, float(ratio_w.mean())
    assert float(clip_w.mean()) > 0.5, float(clip_w.mean())
    (-surr_w.mean()).backward()
    g_clip = float(pol_new.head.net[-1].weight.grad[0].abs().max())
    logp_r = pol_raw.log_prob_of(feat, act_n.detach(), stop_grad_log_std=True)
    ratio_u = torch.exp((logp_r - logp_o.detach()).clamp(-20.0, 20.0))
    (-(ratio_u * adv_n).mean()).backward()
    g_raw = float(pol_raw.head.net[-1].weight.grad[0].abs().max())
    assert g_clip < 1e-6, g_clip
    assert g_raw > 1e-4, g_raw

    class _Hold(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.policy = ContinuousPolicyHead(**kwargs)

    m = _Hold()
    snap = _p3_frozen_unfreeze_policy(m)
    assert _p3_frozen_unfreeze_policy(m) is snap
    snap_ids = {id(p) for p in snap.parameters()}
    live_ids = {id(p) for p in m.parameters()}
    assert snap_ids.isdisjoint(live_ids)
    print('[smoke] OK  p3_mu_ratio_clip bounds μ-grad vs frozen snapshot; ε=0 identity')


def _test_p3_mu_ratio_refresh() -> None:
    """P55: recopy snapshot every N P3 iters; 0 stays freeze-forever."""
    from models.dreamer_v4 import ContinuousPolicyHead
    torch.manual_seed(0)
    assert int(TrainConfig().p3_mu_ratio_refresh_iters) == 0
    kwargs = dict(
        in_dim=8, hidden_dim=16, action_dim=1, n_layers=2, mtp_length=1,
        init_log_std=-1.5, log_std_min=-2.3, log_std_max=0.0)

    class _Hold(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.policy = ContinuousPolicyHead(**kwargs)

    m0 = _Hold()
    snap0 = _p3_frozen_unfreeze_policy(m0)
    with torch.no_grad():
        m0.policy.head.net[-1].bias[0] = 0.5
    assert _p3_frozen_unfreeze_policy(m0) is snap0

    cfg0 = TrainConfig()
    cfg0.p3_mu_ratio_refresh_iters = 0
    cfg0.phase3_train_steps_per_iter = 2
    m1 = _Hold()
    snap_a = _p3_frozen_unfreeze_policy(m1, cfg0)
    with torch.no_grad():
        m1.policy.head.net[-1].bias[0] = 0.5
    assert _p3_frozen_unfreeze_policy(m1, cfg0) is snap_a
    assert _p3_frozen_unfreeze_policy(m1, cfg0) is snap_a

    cfg = TrainConfig()
    cfg.p3_mu_ratio_refresh_iters = 1
    cfg.phase3_train_steps_per_iter = 2
    m = _Hold()
    s1 = _p3_frozen_unfreeze_policy(m, cfg)
    s2 = _p3_frozen_unfreeze_policy(m, cfg)
    assert s2 is s1
    with torch.no_grad():
        m.policy.head.net[-1].bias[0] = 0.4
    s3 = _p3_frozen_unfreeze_policy(m, cfg)
    assert s3 is s1
    live_ids = {id(p) for p in m.parameters()}
    assert {id(p) for p in s3.parameters()}.isdisjoint(live_ids)
    with torch.no_grad():
        b_live = float(m.policy.head.net[-1].bias[0])
        b_snap = float(s3.head.net[-1].bias[0])
    assert abs(b_live - b_snap) < 1e-6
    print('[smoke] OK  p3_mu_ratio_refresh recopies each P3-iter epoch; 0 freeze')


def _test_entropy_collapse_threshold() -> None:
    """P53: 0.20-nat floor margin false-trips open-σ; band frac does not."""
    import math
    unit_g = 0.5 * math.log(2.0 * math.pi * math.e)
    assert abs(_gaussian_entropy_nats(0.0, 1) - unit_g) < 1e-12
    assert abs(_gaussian_entropy_nats(-1.5, 2)
               - 2.0 * (-1.5 + unit_g)) < 1e-12

    cfg = TrainConfig()
    cfg.policy_type = 'continuous'
    # P53 auto-tune σ bounds (test_sim).
    cfg.policy_log_std_min = -1.7020867503855468
    cfg.policy_log_std_max = -1.5197651935915923
    ref = unit_g
    thr = _entropy_collapse_threshold(cfg, 1, ref)
    assert thr is not None
    h_open = _gaussian_entropy_nats(cfg.policy_log_std_max, 1)
    h_floor = _gaussian_entropy_nats(cfg.policy_log_std_min, 1)
    gap = h_open - h_floor
    expect = h_floor + 0.25 * gap
    assert abs(float(thr) - expect) < 1e-9, (thr, expect)
    # Open-σ / H(σ_init) must sit ABOVE the trip (P53 −0.101 vs old −0.083).
    assert h_open > float(thr)
    assert h_floor < float(thr)
    old_nats_thr = h_floor + 0.20
    assert h_open < old_nats_thr  # documents the P53 false trip

    cfg.early_stop_entropy_collapse_floor_frac = 0.0
    thr_legacy = _entropy_collapse_threshold(cfg, 1, ref)
    assert abs(float(thr_legacy) - 0.20 * ref) < 1e-12

    cfg.early_stop_entropy_collapse_floor_frac = 0.25
    cfg.policy_log_std_min = cfg.policy_log_std_max
    assert _entropy_collapse_threshold(cfg, 1, ref) is None

    cfg.policy_type = 'discrete'
    cfg.policy_log_std_min = -1.7
    cfg.policy_log_std_max = -1.5
    assert abs(float(_entropy_collapse_threshold(cfg, 1, math.log(21)))
               - 0.20 * math.log(21)) < 1e-12
    print('[smoke] OK  entropy-collapse thr is σ-band frac; open-σ does not trip')


def _test_id_tau_no_plant_sentinel() -> None:
    """Missing SysID keys must not invent τ=50 s / θ=5 s."""
    from pathlib import Path
    from utils.noise_config import _theta_from_tau, build_noise_config
    root = Path(__file__).resolve().parents[1]
    pp = (root / 'workflow' / '_plant_prepare.py').read_text()
    nc = (root / 'utils' / 'noise_config.py').read_text()
    assert "dyn.get('tau_dominant', 50.0)" not in pp
    assert "dyn.get('dead_time', 5.0)" not in pp
    assert "tau_dominant_identified', 50.0" not in nc
    assert "dead_time_identified', 5.0" not in nc
    assert abs(_theta_from_tau(0.0, 4) - 0.02) < 1e-12
    tau, sr = 53.0, 4
    expected = max(float(sr) / (0.10 * max(1.0, tau)), 0.02)
    assert abs(_theta_from_tau(tau, sr) - expected) < 1e-12
    baked = build_noise_config(
        state_variables=['CV', 'x', 'y', 'DV'],
        cv_indices=[0], dv_indices=[3], mv_indices=[1],
        cv_normalization_ranges=[[68.0, 96.0]],
        dv_normalization_ranges=[[60.0, 140.0]],
        sample_rate=4, noise_stdv=0.03,
        dynamics_json={},
    )
    thetas = [float(row['theta']) for row in baked['ou_noise']]
    assert thetas and all(abs(t - 0.02) < 1e-9 for t in thetas), thetas
    ident = build_noise_config(
        state_variables=['CV', 'x', 'y', 'DV'],
        cv_indices=[0], dv_indices=[3], mv_indices=[1],
        cv_normalization_ranges=[[68.0, 96.0]],
        dv_normalization_ranges=[[60.0, 140.0]],
        sample_rate=4, noise_stdv=0.03,
        dynamics_json={'tau_dominant_identified': 53.0,
                       'dead_time_identified': 8.0},
    )
    ident_th = [float(row['theta']) for row in ident['ou_noise']]
    assert ident_th and all(abs(t - expected) < 5e-4 for t in ident_th), ident_th
    probe_src = (root / 'tools' / 'wm_posterior_prior_probe.py').read_text()
    assert "IDENTIFIED_TAU_DOMINANT', '53'" not in probe_src
    assert "IDENTIFIED_DEAD_TIME', '8'" not in probe_src
    assert "REPO / 'simulation' / 'test_sim'" not in probe_src
    assert 'def _wire_run_artifacts' in probe_src
    print('[smoke] OK  missing SysID τ does not invent 50 s; identified τ identity')


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
    """Horizon / episode formula + IC DR: TrainConfig default, leftover env."""
    import os
    from utils.auto_episode_length import (
        derive_episode_length, derive_horizon, episode_formula_knobs,
        horizon_formula_knobs)
    from utils.initial_conditions import _enabled, _frac, ic_randomization_knobs
    keys = (
        'DREAMER_HORIZON_MAX', 'DREAMER_HORIZON_SETTLE_NTAU',
        'DREAMER_EPISODE_SETTLE_MULTIPLE', 'DREAMER_EPISODE_MIN_LENGTH',
        'DREAMER_EPISODE_MAX_LENGTH', 'SIM_EPISODE_LENGTH',
        'IDENTIFIED_TAU_DOMINANT', 'IDENTIFIED_DEAD_TIME',
        'DREAMER_INIT_RANDOMIZATION', 'DREAMER_INIT_RANDOMIZATION_FRAC',
    )
    prev = {k: os.environ.get(k) for k in keys}
    try:
        for k in keys:
            os.environ.pop(k, None)
        c = TrainConfig()
        assert abs(float(c.horizon_settle_n_tau) - 4.0) < 1e-12
        assert int(c.horizon_max) == 120
        assert abs(float(c.episode_settle_multiple) - 20.0) < 1e-12
        assert int(c.episode_min_length) == 500
        assert int(c.episode_max_length) == 4000
        assert c.init_randomization is True
        assert abs(float(c.init_randomization_frac) - 0.6) < 1e-12
        assert abs(float(c.wm_overhead) - 1.30) < 1e-12
        assert abs(float(c.gpu_target_util) - 0.80) < 1e-12
        assert int(c.gpu_max_bs) == 512
        n_tau, hmax = horizon_formula_knobs()
        assert abs(n_tau - 4.0) < 1e-12
        assert hmax == 120
        k_ep, emin, emax = episode_formula_knobs()
        assert abs(k_ep - 20.0) < 1e-12
        assert emin == 500
        assert emax == 4000
        en, fr = ic_randomization_knobs()
        assert en is True
        assert abs(fr - 0.6) < 1e-12
        assert _enabled() is True
        assert abs(_frac() - 0.6) < 1e-12
        h, src = derive_horizon(tau=55.0, dead_time=8.0, sample_rate=4)
        assert h == 57, (h, src)
        os.environ['DREAMER_HORIZON_MAX'] = '40'
        h2, _ = derive_horizon(tau=55.0, dead_time=8.0, sample_rate=4)
        assert h2 == 40, h2
        assert horizon_formula_knobs()[1] == 40
        os.environ.pop('DREAMER_HORIZON_MAX', None)
        os.environ['DREAMER_HORIZON_SETTLE_NTAU'] = '2.0'
        h3, src3 = derive_horizon(tau=55.0, dead_time=8.0, sample_rate=4)
        assert h3 == 30, (h3, src3)
        assert abs(horizon_formula_knobs()[0] - 2.0) < 1e-12
        os.environ['DREAMER_INIT_RANDOMIZATION'] = '0'
        os.environ['DREAMER_INIT_RANDOMIZATION_FRAC'] = '0.4'
        assert _enabled() is False
        assert abs(_frac() - 0.4) < 1e-12
        os.environ['IDENTIFIED_TAU_DOMINANT'] = '53'
        os.environ['IDENTIFIED_DEAD_TIME'] = '8'
        L, lsrc = derive_episode_length()
        assert L == 1220, (L, lsrc)
        assert '20x_tau_plus_dt' in lsrc
        os.environ['DREAMER_EPISODE_SETTLE_MULTIPLE'] = '10'
        L2, lsrc2 = derive_episode_length()
        assert L2 == 610, (L2, lsrc2)
        assert abs(episode_formula_knobs()[0] - 10.0) < 1e-12
        # MIN floors: 10*(53+8)=610 < 800 → leftover MIN binds.
        os.environ['DREAMER_EPISODE_MIN_LENGTH'] = '800'
        L2b, _ = derive_episode_length()
        assert L2b == 800, L2b
        os.environ.pop('DREAMER_EPISODE_MIN_LENGTH', None)
        os.environ.pop('DREAMER_EPISODE_SETTLE_MULTIPLE', None)
        # MAX caps: 20*(53+8)=1220 > 800 → leftover MAX binds.
        os.environ['DREAMER_EPISODE_MAX_LENGTH'] = '800'
        L3, _ = derive_episode_length()
        assert L3 == 800, L3
        os.environ.pop('DREAMER_EPISODE_MAX_LENGTH', None)
        os.environ['SIM_EPISODE_LENGTH'] = '900'
        L4, lsrc4 = derive_episode_length()
        assert L4 == 900 and lsrc4 == 'env', (L4, lsrc4)
    finally:
        for k, old in prev.items():
            if old is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = old
    print('[smoke] OK  horizon/episode/IC/overhead cfg-or-env identity')


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
        get_phase_disturbance_prob, hidden_disturbance_enabled,
        curriculum_amp_scale)
    from utils.noise_config import noise_curriculum_scale
    keys = (
        'DREAMER_PROCESS_NOISE_AMP_RAMP',
        'DREAMER_DISTURBANCE_PROB_WM',
        'DREAMER_HIDDEN_DISTURBANCE',
        'DREAMER_HIDDEN_DIST_SPREAD',
    )
    prev = {k: os.environ.get(k) for k in keys}
    try:
        for k in keys:
            os.environ.pop(k, None)
        c = TrainConfig()
        assert c.process_noise_amp_ramp == '0.0:0.4'
        assert c.hidden_disturbance is True
        assert abs(float(c.hidden_ou_amp_max_scale) - 0.2) < 1e-12
        assert abs(float(c.hidden_ou_amp_max_scale_p3) - 1.0) < 1e-12
        assert abs(curriculum_amp_scale(1.0, phase=1, cfg=c) - 0.2) < 1e-12
        assert abs(curriculum_amp_scale(1.0, phase=2, cfg=c) - 1.0) < 1e-12
        assert abs(curriculum_amp_scale(1.0, phase=3, cfg=c) - 1.0) < 1e-12
        assert abs(curriculum_amp_scale(1.0, phase=None, cfg=c) - 0.2) < 1e-12
        assert abs(float(c.hidden_dist_p_revert) - 0.7) < 1e-12
        assert c.hidden_dist_shape_weights == '0.5,0.3,0.2'
        assert c.hidden_dist_spread is True
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
        from utils.hidden_disturbance import (
            _knob_bool, force_val_hidden_dist_spread)
        os.environ['DREAMER_HIDDEN_DIST_SPREAD'] = '0'
        c_spread = TrainConfig()
        assert _knob_bool(c_spread, 'hidden_dist_spread',
                          'DREAMER_HIDDEN_DIST_SPREAD', True) is False
        with force_val_hidden_dist_spread(c_spread):
            assert c_spread.hidden_dist_spread is True
            assert _knob_bool(c_spread, 'hidden_dist_spread',
                              'DREAMER_HIDDEN_DIST_SPREAD', True) is True
        assert _knob_bool(c_spread, 'hidden_dist_spread',
                          'DREAMER_HIDDEN_DIST_SPREAD', True) is False
        from pathlib import Path
        _root = Path(__file__).resolve().parents[1]
        src_val = (_root / 'evaluation' / 'validate.py').read_text()
        src_dp = (_root / 'evaluation' / 'wm_disturbance_prediction.py').read_text()
        assert "os.environ['DREAMER_HIDDEN_DIST_SPREAD']" not in src_val
        assert "os.environ['DREAMER_HIDDEN_DIST_SPREAD']" not in src_dp
        assert 'force_val_hidden_dist_spread' in src_val
        assert 'force_val_hidden_dist_spread' in src_dp
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
    from workflow._plant_prepare import (
        ENV_OVERRIDES, explicit_batch_size, gpu_probe_knobs,
    )
    c = TrainConfig()
    assert abs(float(c.gpu_target_util) - 0.80) < 1e-12
    assert int(c.gpu_max_bs) == 512
    assert abs(float(c.wm_overhead) - 1.30) < 1e-12
    assert 'DREAMER_TARGET_UTIL' in ENV_OVERRIDES
    assert 'DREAMER_MAX_BS' in ENV_OVERRIDES
    assert 'DREAMER_BATCH_SIZE' in ENV_OVERRIDES
    assert 'DREAMER_WM_OVERHEAD' in ENV_OVERRIDES
    keys = (
        'DREAMER_BATCH_SIZE', 'OBJ_BATCH_SIZE',
        'DREAMER_WM_OVERHEAD', 'DREAMER_TARGET_UTIL', 'DREAMER_MAX_BS',
    )
    prev = {k: os.environ.get(k) for k in keys}
    try:
        for k in keys:
            os.environ.pop(k, None)
        assert explicit_batch_size() is None
        oh, util, cap = gpu_probe_knobs()
        assert abs(oh - 1.30) < 1e-12
        assert abs(util - 0.80) < 1e-12
        assert cap == 512
        os.environ['OBJ_BATCH_SIZE'] = '24'
        assert explicit_batch_size() == 24
        os.environ['DREAMER_BATCH_SIZE'] = '48'
        assert explicit_batch_size() == 48
        os.environ['DREAMER_WM_OVERHEAD'] = '1.5'
        os.environ['DREAMER_TARGET_UTIL'] = '0.65'
        os.environ['DREAMER_MAX_BS'] = '64'
        oh, util, cap = gpu_probe_knobs()
        assert abs(oh - 1.5) < 1e-12
        assert abs(util - 0.65) < 1e-12
        assert cap == 64
    finally:
        for k, old in prev.items():
            if old is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = old
    print('[smoke] OK  GPU-calib TrainConfig + gpu_probe_knobs leftover env')


def _test_sim_snr_cfg() -> None:
    """Plant SNR: TrainConfig default, leftover SIM_*, DREAMER beats leftover."""
    import os
    from utils.noise_config import build_noise_config, resolve_sim_snr_knobs
    c = TrainConfig()
    assert c.sim_noise_adaptive is True
    assert abs(float(c.sim_ou_gain_cv) - 0.15) < 1e-12
    assert abs(float(c.sim_ou_gain_dv) - 0.60) < 1e-12
    keys = (
        'DREAMER_SIM_NOISE_ADAPTIVE', 'SIM_NOISE_ADAPTIVE',
        'DREAMER_SIM_OU_GAIN_CV', 'SIM_OU_GAIN_CV',
        'DREAMER_SIM_OU_GAIN_DV', 'SIM_OU_GAIN_DV',
        'DREAMER_SIM_OU_SIGMA_FRAC', 'SIM_OU_SIGMA_FRAC',
        'DREAMER_SIM_MEAS_NOISE_CV_FRAC', 'SIM_MEAS_NOISE_CV_FRAC',
        'DREAMER_SIM_MEAS_NOISE_DV_FRAC', 'SIM_MEAS_NOISE_DV_FRAC',
    )
    prev = {k: os.environ.get(k) for k in keys}
    try:
        for k in keys:
            os.environ.pop(k, None)
        kn = resolve_sim_snr_knobs()
        kn_cfg = resolve_sim_snr_knobs(c)
        assert kn == kn_cfg
        assert kn['adaptive'] is True
        assert abs(float(kn['ou_gain_cv']) - 0.15) < 1e-12
        baked = build_noise_config(
            state_variables=['CV', 'x', 'y', 'DV'],
            cv_indices=[0], dv_indices=[3], mv_indices=[1],
            cv_normalization_ranges=[[68.0, 96.0]],
            dv_normalization_ranges=[[60.0, 140.0]],
            sample_rate=4, noise_stdv=0.03,
        )
        ou = {row['channel_type']: row for row in baked['ou_noise']}
        assert abs(float(ou['cv']['gain']) - 0.225) < 1e-9, ou['cv']
        assert abs(float(ou['dv']['gain']) - 0.9) < 1e-9, ou['dv']
        os.environ['SIM_OU_GAIN_CV'] = '0.25'
        assert abs(float(resolve_sim_snr_knobs()['ou_gain_cv']) - 0.25) < 1e-12
        os.environ['DREAMER_SIM_OU_GAIN_CV'] = '0.35'
        assert abs(float(resolve_sim_snr_knobs()['ou_gain_cv']) - 0.35) < 1e-12
        c_ex = TrainConfig()
        c_ex.sim_ou_gain_cv = 0.11
        c_ex._explicit_fields = {'sim_ou_gain_cv'}  # type: ignore
        assert abs(float(resolve_sim_snr_knobs(c_ex)['ou_gain_cv']) - 0.11) < 1e-12
        os.environ.pop('DREAMER_SIM_OU_GAIN_CV', None)
        os.environ.pop('SIM_OU_GAIN_CV', None)
        os.environ['SIM_NOISE_ADAPTIVE'] = '0'
        assert resolve_sim_snr_knobs()['adaptive'] is False
        os.environ['DREAMER_SIM_NOISE_ADAPTIVE'] = '1'
        assert resolve_sim_snr_knobs()['adaptive'] is True
    finally:
        for k, old in prev.items():
            if old is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = old
    print('[smoke] OK  plant-SNR TrainConfig + DREAMER_SIM_* beats leftover SIM_*')


def _test_agent_disturbance_cfg() -> None:
    """Operator-event schedule: TrainConfig default, leftover AGENT_*, DREAMER wins."""
    import os
    from utils.training_disturbance import (
        build_training_disturbance_schedule, get_authority_target_frac,
        clamp_event_to_authority_budget)
    from workflow._plant_prepare import ENV_OVERRIDES

    c = TrainConfig()
    assert abs(float(c.disturbance_authority_frac) - 0.65) < 1e-12
    assert abs(float(c.disturbance_recovery_frac) - 0.20) < 1e-12
    assert int(c.disturbance_settle_steps) == 0
    assert abs(float(c.disturbance_quiet_frac) - 0.12) < 1e-12
    assert 'DREAMER_DISTURBANCE_AUTHORITY_FRAC' in ENV_OVERRIDES
    keys = (
        'DREAMER_DISTURBANCE_AUTHORITY_FRAC', 'AGENT_DISTURBANCE_AUTHORITY_FRAC',
        'DREAMER_DISTURBANCE_RECOVERY_FRAC', 'AGENT_DISTURBANCE_RECOVERY_FRAC',
        'DREAMER_DISTURBANCE_QUIET_FRAC', 'AGENT_DISTURBANCE_QUIET_FRAC',
        'DREAMER_DISTURBANCE_SETTLE_STEPS', 'AGENT_DISTURBANCE_SETTLE_STEPS',
    )
    prev = {k: os.environ.get(k) for k in keys}
    try:
        for k in keys:
            os.environ.pop(k, None)
        assert abs(get_authority_target_frac() - 0.65) < 1e-12
        assert abs(get_authority_target_frac(cfg=c) - 0.65) < 1e-12
        os.environ['AGENT_DISTURBANCE_AUTHORITY_FRAC'] = '0.40'
        assert abs(get_authority_target_frac(cfg=c) - 0.40) < 1e-12
        os.environ['DREAMER_DISTURBANCE_AUTHORITY_FRAC'] = '0.55'
        assert abs(get_authority_target_frac(cfg=c) - 0.55) < 1e-12
        c_ex = TrainConfig()
        c_ex.disturbance_authority_frac = 0.70
        c_ex._explicit_fields = {'disturbance_authority_frac'}  # type: ignore
        assert abs(get_authority_target_frac(cfg=c_ex) - 0.70) < 1e-12
        os.environ.pop('DREAMER_DISTURBANCE_AUTHORITY_FRAC', None)
        os.environ.pop('AGENT_DISTURBANCE_AUTHORITY_FRAC', None)
        os.environ['AGENT_DISTURBANCE_RECOVERY_FRAC'] = '0.50'
        d0, _ = clamp_event_to_authority_budget(
            proposed_delta=1.0, cv_impact_per_unit=1.0,
            cumulative_cv_impact=10.0, mv_authority_cv=1.0,
            target_frac=0.65)
        assert abs(d0 + 0.50) < 1e-12
        c_rec = TrainConfig()
        c_rec.disturbance_recovery_frac = 0.10
        c_rec._explicit_fields = {'disturbance_recovery_frac'}  # type: ignore
        d1, _ = clamp_event_to_authority_budget(
            proposed_delta=1.0, cv_impact_per_unit=1.0,
            cumulative_cv_impact=10.0, mv_authority_cv=1.0,
            target_frac=0.65, cfg=c_rec)
        assert abs(d1 + 0.10) < 1e-12
        os.environ.pop('AGENT_DISTURBANCE_RECOVERY_FRAC', None)
        # Quiet-frac clips to 0.5 (HEAD identity). Stub uniform=0 so
        # the gate is deterministic (seed-0 Generator is not).
        class _QuietRng:
            def uniform(self, *a, **k):
                return 0.0
            def integers(self, *a, **k):
                raise AssertionError('quiet path must not draw integers')
        c_q = TrainConfig()
        c_q.disturbance_quiet_frac = 1.0
        c_q._explicit_fields = {'disturbance_quiet_frac'}  # type: ignore
        assert build_training_disturbance_schedule(1220, _QuietRng(), cfg=c_q) == []
        import utils.training_disturbance as td
        a = td._load_identifier_context()
        b = td._load_identifier_context()
        assert a is b, 'identifier JSON must be process-cached'
        assert not hasattr(td, 'MVSaturationMonitor')
        assert not hasattr(td, 'DisturbanceIntensityController')
        assert not hasattr(td, 'disturbance_curriculum_enabled')
        assert not hasattr(td, 'apply_episode_init_offsets')
    finally:
        for k, old in prev.items():
            if old is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = old
    print('[smoke] OK  operator-event TrainConfig + leftover AGENT_* identity')


def _test_sim_runtime_cfg() -> None:
    """Wrapper seed/jitter/enable/DR: TrainConfig default, leftover SIM_*, DREAMER wins."""
    import os
    from utils.noise_config import build_noise_config, resolve_sim_runtime_knobs
    from utils.sim_noise import DomainRandomizer, SimNoiseWrapper

    c = TrainConfig()
    assert c.sim_noise_enabled is True
    assert abs(float(c.sim_noise_jitter_pct) - 0.20) < 1e-12
    assert c.sim_domain_randomization is True
    assert abs(float(c.sim_param_randomization_pct) + 1.0) < 1e-12
    keys = (
        'DREAMER_SIM_NOISE_ENABLED', 'SIM_NOISE_ENABLED',
        'DREAMER_SIM_NOISE_SEED', 'SIM_NOISE_SEED',
        'DREAMER_SIM_NOISE_JITTER_PCT', 'SIM_NOISE_JITTER_PCT',
        'SIM_NOISE_AMPLITUDE_JITTER_PCT',
        'DREAMER_SIM_DOMAIN_RANDOMIZATION', 'SIM_DOMAIN_RANDOMIZATION',
        'DREAMER_DOMAIN_RANDOMIZATION',
        'DREAMER_SIM_DOMAIN_RANDOMIZATION_SEED',
        'SIM_DOMAIN_RANDOMIZATION_SEED',
        'DREAMER_SIM_PARAM_RANDOMIZATION_PCT',
        'SIM_PARAM_RANDOMIZATION_PCT',
    )
    prev = {k: os.environ.get(k) for k in keys}

    class _Bare:
        noise_config = {
            'ou_noise': [{'index': 0, 'sigma': 0.01, 'gain': 1.0,
                          'bounds': (-1.0, 1.0)}],
            'measurement_noise': [],
        }
        _randomizer = None

        def step(self, action):
            return [0.0], False

        def reset(self):
            return [0.0]

    try:
        for k in keys:
            os.environ.pop(k, None)
        kn = resolve_sim_runtime_knobs()
        kn_cfg = resolve_sim_runtime_knobs(c)
        assert kn == kn_cfg
        assert kn['noise_enabled'] is True
        assert abs(float(kn['jitter_pct']) - 0.20) < 1e-12
        assert kn['domain_randomization'] is True
        assert kn['noise_seed'] == ''
        assert kn['param_randomization_pct'] is None
        wrap = SimNoiseWrapper(_Bare())
        assert wrap._has_noise is True
        assert abs(float(wrap._noise_jitter_pct) - 0.20) < 1e-9
        dr = DomainRandomizer()
        assert dr.enabled is True
        baked = build_noise_config(
            state_variables=['CV', 'x', 'y', 'DV'],
            cv_indices=[0], dv_indices=[3], mv_indices=[1],
            cv_normalization_ranges=[[68.0, 96.0]],
            dv_normalization_ranges=[[60.0, 140.0]],
            sample_rate=4, noise_stdv=0.03,
        )
        assert baked['domain_randomization']['enabled'] is True
        os.environ['SIM_NOISE_AMPLITUDE_JITTER_PCT'] = '0'
        assert abs(float(resolve_sim_runtime_knobs()['jitter_pct'])) < 1e-12
        os.environ['SIM_NOISE_JITTER_PCT'] = '0.10'
        assert abs(float(resolve_sim_runtime_knobs()['jitter_pct']) - 0.10) < 1e-12
        os.environ['DREAMER_SIM_NOISE_JITTER_PCT'] = '0.05'
        assert abs(float(resolve_sim_runtime_knobs()['jitter_pct']) - 0.05) < 1e-12
        os.environ['SIM_NOISE_ENABLED'] = '0'
        assert resolve_sim_runtime_knobs()['noise_enabled'] is False
        wrap_off = SimNoiseWrapper(_Bare())
        assert wrap_off._has_noise is False
        os.environ['DREAMER_SIM_NOISE_ENABLED'] = '1'
        assert resolve_sim_runtime_knobs()['noise_enabled'] is True
        os.environ['SIM_DOMAIN_RANDOMIZATION'] = '0'
        assert resolve_sim_runtime_knobs()['domain_randomization'] is False
        os.environ['DREAMER_SIM_DOMAIN_RANDOMIZATION'] = '1'
        assert resolve_sim_runtime_knobs()['domain_randomization'] is True
        os.environ['SIM_PARAM_RANDOMIZATION_PCT'] = '0.25'
        assert abs(float(resolve_sim_runtime_knobs()['param_randomization_pct'])
                   - 0.25) < 1e-12
        baked_ov = build_noise_config(
            state_variables=['CV', 'x', 'y', 'DV'],
            cv_indices=[0], dv_indices=[3], mv_indices=[1],
            cv_normalization_ranges=[[68.0, 96.0]],
            dv_normalization_ranges=[[60.0, 140.0]],
            sample_rate=4, noise_stdv=0.03,
        )
        assert abs(float(baked_ov['domain_randomization']
                         ['param_randomization_pct']) - 0.25) < 1e-12
        os.environ['DREAMER_SIM_PARAM_RANDOMIZATION_PCT'] = '0.12'
        assert abs(float(resolve_sim_runtime_knobs()['param_randomization_pct'])
                   - 0.12) < 1e-12
        c_ex = TrainConfig()
        c_ex.sim_noise_jitter_pct = 0.33
        c_ex._explicit_fields = {'sim_noise_jitter_pct'}  # type: ignore
        assert abs(float(resolve_sim_runtime_knobs(c_ex)['jitter_pct']) - 0.33) < 1e-12
        wrap.apply_runtime_knobs(c_ex)
        assert abs(float(wrap._noise_jitter_pct) - 0.33) < 1e-12
        c_off = TrainConfig()
        c_off.sim_noise_enabled = False
        c_off._explicit_fields = {'sim_noise_enabled'}  # type: ignore
        wrap.apply_runtime_knobs(c_off)
        assert wrap._has_noise is False
        wrap.apply_runtime_knobs(TrainConfig())
        assert wrap._has_noise is True
    finally:
        for k, old in prev.items():
            if old is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = old
    print('[smoke] OK  wrapper jitter/enable/DR TrainConfig + dead AMPLITUDE name')


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
    assert _row_adv_action_corr({'adv_action_corr': 0.4}) == 0.4
    assert _row_adv_action_corr({'imag_adv_action_corr': 0.3}) == 0.3
    assert _row_adv_action_corr(
        {'adv_action_corr': 0.4, 'imag_adv_action_corr': 0.1}) == 0.4
    assert _row_adv_action_corr({}) is None
    print('[smoke] OK  _row_adv_action_corr prefers canonical jsonl key')


def _test_training_diagnostics_cascade_axes() -> None:
    """P55 GPU-occupied: plot/csv/parser surface real-sim cascade axes."""
    import json
    import os
    import tempfile
    from evaluation.diagnostics import _parse_train_log

    rows = [
        {'iter': 1, 'phase': 1, 'env_steps': 100,
         'ema_return': -200.0, 'recon_loss': 0.01},
        {'iter': 147, 'phase': 3, 'env_steps': 1000,
         'ema_return': -170.0, 'entropy_mean': -0.101,
         'actor_logp_std': 0.67, 'actor_ratio_clip_frac': 0.47,
         'actor_ratio_mean': 1.02, 'critic_rew_to_tgt_var': 0.06,
         'agent_minus_expert_return': 19.0, 'actor_pos_adv_frac': 0.55,
         'pmpo_pos_frac': 0.55, 'adv_action_corr': 0.12,
         'imag_adv_action_corr': 0.12, 'return_scale': 1.91,
         'realsim_return_mean': -7.3, 'realsim_reward_mean': -0.11},
        {'iter': 175, 'phase': 3, 'env_steps': 2000,
         'ema_return': -418.0, 'entropy_mean': -0.136,
         'actor_logp_std': 18.8, 'actor_ratio_clip_frac': 0.46,
         'actor_ratio_mean': 0.98, 'critic_rew_to_tgt_var': 0.0001,
         'agent_minus_expert_return': -1237.0, 'actor_pos_adv_frac': 0.62,
         'pmpo_pos_frac': 0.62, 'adv_action_corr': 0.08,
         'imag_adv_action_corr': 0.12, 'return_scale': 1.91},
    ]
    with tempfile.TemporaryDirectory() as td:
        logp = os.path.join(td, 'train_log.jsonl')
        with open(logp, 'w') as fh:
            for r in rows:
                fh.write(json.dumps(r) + '\n')
        out = os.path.join(td, 'training_diagnostics.png')
        _save_training_diagnostics_plot(logp, out)
        assert os.path.isfile(out)
        csvp = os.path.join(td, 'training_diagnostics.csv')
        assert os.path.isfile(csvp)
        header = open(csvp).readline()
        for col in ('actor_logp_std', 'actor_ratio_clip_frac',
                    'critic_rew_to_tgt_var', 'agent_minus_expert_return',
                    'actor_pos_adv_frac', 'adv_action_corr',
                    'n_grad_skip', 'n_grad_skip_iter',
                    'realsim_return_mean', 'realsim_reward_mean'):
            assert col in header, header
        _, summary = _parse_train_log(logp)
        p3 = summary['p3']
        assert 'actor_pos_adv_frac' in p3
        assert 'actor_logp_std' in p3
        assert 'realsim_return_mean' in p3
        assert abs(float(p3['actor_logp_std']['last']) - 18.8) < 1e-9
        assert abs(float(p3['realsim_return_mean']['first']) + 7.3) < 1e-9
        flags = ' '.join(summary['flags'])
        assert 'imagined returns' not in flags
        # Pre-P65 jsonl still surfaces the imagination alias.
        oldp = os.path.join(td, 'old_train_log.jsonl')
        with open(oldp, 'w') as fh:
            fh.write(json.dumps({
                'iter': 10, 'phase': 3, 'env_steps': 500,
                'imagined_return_mean': -8.1, 'imagined_reward_mean': -0.2,
            }) + '\n')
        _, old_sum = _parse_train_log(oldp)
        assert 'imagined_return_mean' in old_sum['p3']
        assert 'realsim_return_mean' not in old_sum['p3']
    print('[smoke] OK  training_diagnostics cascade axes + old-log leftover names')


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
    line_lo = _format_gain_probe_line({
        'gain_ready': True,
        'r_min': 0.91, 'r_max': 1.02,
        'worst_ratio': 0.91, 'worst_input': 'DV CV0<-DV0',
        'band': [0.8, 1.3], 'noise_worst': 0.4, 'sign_flips': 0,
        'n_checks': 2, 'unbiased': True, 'not_noisy': True,
        'atH_min': 0.90, 'atH_max': 1.00,
        'ss_pairs': [('MV CV0<-MV0', 1.02), ('DV CV0<-DV0', 0.91)],
        'ath_pairs': [('MV CV0<-MV0', 1.00), ('DV CV0<-DV0', 0.90)],
        'probed_last_ok': True, 'last_ok_iter': 81,
    })
    assert 'last_ok_iter=81' in line_lo
    print('[smoke] OK  gain-probe line prints ss and @H')


def _test_load_module_state_roundtrip() -> None:
    """Last-ok gain-probe must restore live weights after the swap."""
    m = torch.nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        m.weight.fill_(1.0)
    live = _clone_module_state(m, torch.device('cpu'))
    last = {k: torch.full_like(v, 2.0) for k, v in live.items()}
    _load_module_state(m, last, torch.device('cpu'))
    assert float(m.weight[0, 0]) == 2.0
    _load_module_state(m, live, torch.device('cpu'))
    assert float(m.weight[0, 0]) == 1.0
    print('[smoke] OK  last-ok probe swap restores live weights')


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
    assert 'gmatch_step=0.4' in banner, banner
    assert 'gmatch_clip=True' in banner, banner
    assert 'gmatch_rest=True' in banner, banner
    assert 'gmatch_rest_L=' in banner, banner
    assert 'gmatch_rest_cg=True' in banner, banner
    assert 'p3_sigreset=False' in banner, banner
    assert 'p3_sglogstd=True' in banner, banner
    assert 'p3_logpclip=8' in banner, banner
    assert 'p3_muratio=0.2' in banner, banner
    assert 'p3_murefresh=0' in banner, banner
    assert 'es_ent_floor=0.25' in banner, banner
    plan = json.loads(plan_path.read_text())
    assert plan['config']['rssm_latent_type'] == 'deterministic'
    assert float(plan['config']['gain_match_coef']) == 1.0
    assert abs(float(plan['config']['gain_match_step']) - 0.4) < 1e-12
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


def _test_collect_serve_cuda_graph_cpu() -> None:
    """GPU-occupied identity: collect graph is CUDA-only; CPU stays eager."""
    from models.dreamer_v4_rssm import get_collect_serve_cuda_graph
    from models import dreamer_v4_rssm as _rssm_mod
    import training.train as _train_mod
    torch.manual_seed(0)
    cfg = TrainConfig()
    cfg.obs_dim, cfg.action_dim = 4, 1
    cfg.lookback, cfg.seq_len, cfg.horizon = 8, 8, 4
    cfg.mtp_length = 1
    cfg.rssm_deter_dim = 16
    cfg.rssm_n_categoricals = 4
    cfg.rssm_n_classes = 4
    cfg.rssm_embed_dim = 8
    cfg.rssm_hidden_dim = 8
    cfg.head_hidden = 8
    cfg.compile_mode = 'off'
    cfg.dob_enabled = False
    cfg.dv_as_input = False
    model = build_model(cfg)
    st = model.dynamics.initial_state(1, torch.device('cpu'))
    assert get_collect_serve_cuda_graph(
        model.dynamics, st, torch.device('cpu'), cfg.obs_dim,
        cfg.action_dim) is None
    from models.dreamer_v4_rssm import (
        RSSMConfig as _RSSMConfig, RSSMDynamics as _RSSMDynamics,
        _copy_rssm_state, stream_serve_step)
    cfg_c = _RSSMConfig(obs_dim=6, action_dim=2, deter_dim=16,
                        n_categoricals=4, n_classes=4, embed_dim=16,
                        hidden_dim=16, latent_type='deterministic',
                        cont_gain_dim=2, dob_enabled=True, cv_indices=(0,))
    m_c = _RSSMDynamics(cfg_c).eval()
    st_c = m_c.initial_state(1, torch.device('cpu'))
    assert st_c.c is not None and st_c.c_mean is not None and st_c.c_std is not None
    with torch.no_grad():
        st_n = stream_serve_step(
            m_c, st_c, torch.zeros(1, 2), torch.zeros(1, 6), sample=False)
    _copy_rssm_state(st_c, st_n)
    _rsrc = open(_rssm_mod.__file__).read()
    assert 'def get_collect_serve_cuda_graph' in _rsrc
    assert 'class CollectServeCudaGraph' in _rsrc
    _tsrc = open(_train_mod.__file__).read()
    assert 'get_collect_serve_cuda_graph' in _tsrc
    assert '_rssm_prev_a.copy_' in _tsrc
    assert 'def warmup_collect_serve_cuda_graph' in _rsrc
    assert '_warmup_p3_collect_serve_graph' in _tsrc
    import evaluation.validate as _val_mod
    _vsrc = open(_val_mod.__file__).read()
    assert 'get_collect_serve_cuda_graph' in _vsrc
    assert '_prev.copy_' in _vsrc
    assert '_rssm_prev_a = action_t.detach().float()' not in _vsrc
    print('[smoke] OK  collect serve CUDA graph skipped on CPU; eager copy_')


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
    _test_recon_channel_weights_cache()
    _test_cli_only_env_disjoint()
    _test_batch_np_to_device_identity()
    _test_time_unbind_and_p1_h2d_keys()
    _test_lambda_returns_scan()
    _test_buffer_sample_keys()
    _test_buffer_clear()
    _test_store_aux_feats_identity()
    _test_img_rollout_last_only()
    _test_img_step_det_roll_skips_sample()
    _test_initial_state_zeros_cache()
    _test_stage1_dob_ground_skip()
    _test_stream_serve_matches_rollout()
    _test_collect_serve_cuda_graph_cpu()
    _test_envfree_observer_recipe()
    _test_identified_tau_cfg()
    _test_objective_runtime_cfg()
    _test_auto_weights_cfg()
    _test_runtime_setpoint_schedule_cfg()
    _test_step_seed_shaping_prbs_seg_cfg()
    _test_expert_move_law_cfg()
    _test_pin_eval_modules()
    _test_control_quality_gates()
    _test_require_realsim_actor()
    _test_rssm_param_grad_snapshot()
    _test_gain_match_per_input_huber()
    _test_gain_match_pred_over_tgt()
    _test_gain_match_fd_held()
    _test_gain_match_clip_realized()
    _test_gain_match_fd_action_seq()
    _test_gain_match_held_settle()
    _test_gain_match_rest_window()
    _test_held_rollout_win_fits_k()
    _test_held_rollout_cv_space()
    _test_collect_rest_lookback_tm_pairing()
    _test_gain_match_rest_ic()
    _test_p3_reset_log_std()
    _test_bc_mean_only()
    _test_p3_stop_grad_log_std()
    _test_p3_shared_ac_forwards()
    _test_p3_logp_clip()
    _test_p3_mu_ratio_clip()
    _test_p3_mu_ratio_refresh()
    _test_entropy_collapse_threshold()
    _test_id_tau_no_plant_sentinel()
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
    _test_sim_snr_cfg()
    _test_agent_disturbance_cfg()
    _test_sim_runtime_cfg()
    _test_adv_action_corr_vectorized()
    _test_training_diagnostics_cascade_axes()
    _test_format_gain_probe_line()
    _test_load_module_state_roundtrip()
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
