"""End-to-end (in-loop) smoke for the 2026-06-09 WM-fix levers.

Runs the REAL ``train()`` loop on test_sim for a tiny budget (CPU, joint mode)
with the new knobs turned ON, then asserts that each new code path actually
EXECUTED in the loop (not just the unit-level helpers in _smoke_wm_fixes.py):

  * #5 freeze-after-pretrain  -> a row with ``wm_frozen == True`` + the
                                 "[joint] WM CORE FROZEN" banner.
  * #4 BC expert tracking      -> at least one row carries a non-null
                                 ``expert_det_return`` + ``agent_minus_expert_return``.
  * #6 CV-weighted recon       -> ``recon_loss`` stays finite throughout with
                                 the lever engaged.
  (#7 WM-only excitation partition was removed 2026-06-12 — see TrainConfig.)
  * the run completes without raising.

This is intentionally heavier than the unit smoke; keep the budget tiny.

Run (CPU, do not disturb a live GPU run):
  CUDA_VISIBLE_DEVICES="" PYTHONPATH=$PWD \
  $PWD/../neural-apc-mbrl-env/bin/python tools/_smoke_wm_fixes_e2e.py
"""
import json
import os
import tempfile


def _setup_env():
    os.environ['CONTROL_SETUP_JSON'] = 'simulation/test_sim/control_setup.json'
    os.environ['CONTROL_OBJECTIVE_JSON'] = \
        'simulation/test_sim/control_objective.json'
    os.environ['SEED'] = '0'
    # test_sim identified timing (the disturbance tau-fix reads these).
    os.environ.setdefault('IDENTIFIED_TAU_DOMINANT', '53')
    os.environ.setdefault('IDENTIFIED_DEAD_TIME', '8')
    # Feed the static expert real identified gains (else it is skipped =>
    # no BC tracking).  Reuse p106's plant-id artefact (test_sim is stationary).
    _dyn = ('output/test_sim/run_20260609_p106_mccritic/plant_id/'
            'dynamics_identification.json')
    if os.path.isfile(_dyn):
        os.environ['DREAMER_DYNAMICS_ID_JSON'] = _dyn


def _mk_cfg(out_dir):
    from training.train import TrainConfig
    cfg = TrainConfig()
    # --- tiny model + budget ---
    cfg.out_dir = out_dir
    cfg.episode_length = 120
    cfg.sample_rate = 4
    cfg.seq_len = 16
    cfg.lookback = 8
    cfg.horizon = 8
    cfg.d_model = 64
    cfg.head_hidden = 64
    cfg.rssm_deter_dim = 64
    cfg.rssm_n_categoricals = 8
    cfg.rssm_n_classes = 8
    cfg.rssm_embed_dim = 32
    cfg.rssm_hidden_dim = 32
    cfg.mtp_length = 4
    cfg.batch_size = 8
    cfg.world_model_type = 'rssm'
    cfg.compile_mode = 'none'
    cfg.train_mode = 'joint'
    cfg.train_steps_per_iter = 2
    cfg.phase3_train_steps_per_iter = 2
    cfg.phase3_eval_every_iters = 2
    cfg.phase3_collect_every_iters = 1
    cfg.phase3_pruner_warmup_iters = 0
    cfg.early_stop_enable = False
    # --- small seed buffer (mark explicit so auto-tune leaves them) ---
    cfg.baseline_seed_episodes = 2
    cfg.random_seed_episodes = 1
    cfg.exploration_seed_episodes = 2      # PRBS -> fills excitation buffer
    cfg.constant_action_seed_episodes = 2  # const/step -> fills exc buffer
    cfg.step_test_seed_episodes = 0        # drop the 20-episode floor
    cfg.step_test_episodes_per_channel = 1  # step-test -> fills exc buffer
    cfg.expert_type = 'static'
    cfg.expert_seed_episodes = 2           # expert active -> BC tracking on
    # prefill ~ (2+1+2+2+2+2) eps * 120 = ~1320 steps; leave room for ~16 iters.
    cfg.total_steps = 3600
    cfg._explicit_fields = {
        'baseline_seed_episodes', 'random_seed_episodes',
        'exploration_seed_episodes', 'gamma', 'policy_init_log_std',
        'policy_log_std_max', 'policy_log_std_min', 'pmpo_entropy_coef',
    }
    # --- THE FOUR NEW LEVERS, all ON ---
    cfg.wm_freeze_after_iters = 4          # #5
    cfg.wm_recon_cv_weight = 4.0           # #6
    cfg.wm_excitation_buffer_frac = 0.5    # #7
    cfg.bc_track_expert_every = 1          # #4
    cfg.expert_bc_p3_floor = 0.0           # #4 full-release (floor confound off)
    return cfg


def main():
    _setup_env()
    out_dir = tempfile.mkdtemp(prefix='wmfix_e2e_')
    from training.train import train
    cfg = _mk_cfg(out_dir)
    print(f'[e2e] running tiny joint train() on test_sim -> {out_dir}')
    train(cfg)

    log_path = os.path.join(out_dir, 'train_log.jsonl')
    assert os.path.exists(log_path), f'no train_log.jsonl at {log_path}'
    rows = [json.loads(l) for l in open(log_path) if l.strip()]
    assert rows, 'train_log.jsonl is empty'
    print(f'[e2e] {len(rows)} log rows')

    # #6 recon finite throughout (lever engaged).
    recons = [r.get('recon_loss') for r in rows if r.get('recon_loss') is not None]
    assert recons and all(v == v and abs(v) < 1e6 for v in recons), \
        f'recon_loss not finite with CV-weight on: {recons[:5]}'
    print(f'[e2e] OK  #6 recon_loss finite throughout '
          f'(n={len(recons)}, last={recons[-1]:.4f})')

    # #5 freeze fired.
    froze = [r for r in rows if r.get('wm_frozen') is True]
    assert froze, 'WM never froze (wm_frozen never True) — freeze hook not wired'
    print(f'[e2e] OK  #5 WM froze in-loop (wm_frozen=True in {len(froze)} rows)')

    # #4 BC expert tracking logged.
    exp_rows = [r for r in rows if r.get('expert_det_return') is not None]
    assert exp_rows, 'expert_det_return never logged — BC tracking not wired'
    amx = [r.get('agent_minus_expert_return') for r in exp_rows
           if r.get('agent_minus_expert_return') is not None]
    assert amx, 'agent_minus_expert_return never logged'
    print(f'[e2e] OK  #4 BC expert tracking logged '
          f'(n={len(exp_rows)}, last expert_det={exp_rows[-1]["expert_det_return"]:.2f}, '
          f'gap={amx[-1]:.2f})')

    print('\n[e2e] ALL IN-LOOP WM-FIX PATHS EXECUTED — END-TO-END SMOKE PASSED')


if __name__ == '__main__':
    main()
