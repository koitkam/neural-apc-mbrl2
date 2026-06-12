"""End-to-end (in-loop) smoke for the staged clean->disturbance curriculum.

Runs the REAL ``train()`` loop on test_sim for a tiny PHASED budget (CPU) with
``curriculum_enabled=True`` + ``dob_enabled=True`` and asserts the in-loop stage
machinery actually executes across all three phases (not just the unit-level
freeze helpers in _smoke_curriculum.py):

  * the run reaches all THREE stages (P1, P2, P3 appear in train_log rows).
  * the "[curriculum] >>> STAGE k" transition banner fires for k=1,2,3 (the
    per-iter latch applied the freeze + dob_active for each stage).
  * recon_loss stays finite throughout (no blow-up at a freeze boundary).
  * ``wm_frozen`` becomes True by Stage 3 (the P3 _wm_frozen_now latch).
  * the run completes without raising (phase transitions don't hiccup).

Run (CPU, do not disturb a live GPU run):
  CUDA_VISIBLE_DEVICES="" PYTHONPATH=$PWD DREAMER_COMPILE=0 \
  $PWD/../neural-apc-mbrl-env/bin/python tools/_smoke_curriculum_e2e.py
"""
import io
import json
import os
import contextlib
import tempfile


def _setup_env():
    os.environ['CONTROL_SETUP_JSON'] = 'simulation/test_sim/control_setup.json'
    os.environ['CONTROL_OBJECTIVE_JSON'] = \
        'simulation/test_sim/control_objective.json'
    os.environ['SEED'] = '0'
    os.environ.setdefault('IDENTIFIED_TAU_DOMINANT', '53')
    os.environ.setdefault('IDENTIFIED_DEAD_TIME', '8')
    _dyn = ('output/test_sim/run_20260609_p106_mccritic/plant_id/'
            'dynamics_identification.json')
    if os.path.isfile(_dyn):
        os.environ['DREAMER_DYNAMICS_ID_JSON'] = _dyn


def _mk_cfg(out_dir):
    from training.train import TrainConfig
    cfg = TrainConfig()
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
    # PHASED mode — the curriculum IS the phased schedule.
    cfg.train_mode = 'phased'
    cfg.train_steps_per_iter = 2
    cfg.phase3_train_steps_per_iter = 2
    cfg.phase3_eval_every_iters = 2
    cfg.phase3_collect_every_iters = 1
    cfg.phase3_pruner_warmup_iters = 0
    cfg.p3_critic_warmup_iters = 1
    cfg.early_stop_enable = False
    # Even-ish phase split so all three stages get iters in a tiny budget.
    cfg.phase1_frac = 0.4
    cfg.phase2_frac = 0.3
    cfg.phase3_frac = 0.3
    cfg.phase3_min_frac = 0.2
    # Disable the adaptive phase-gate extensions so the tiny budget actually
    # advances P1->P2->P3 (the gates can otherwise hold P1 the whole run).
    cfg.p1_gate_max_extension = 0.0
    cfg.p2_gate_max_extension = 0.0
    # small seed buffer
    cfg.baseline_seed_episodes = 2
    cfg.random_seed_episodes = 1
    cfg.exploration_seed_episodes = 2
    cfg.constant_action_seed_episodes = 2
    cfg.step_test_seed_episodes = 0
    cfg.step_test_episodes_per_channel = 1
    cfg.expert_type = 'static'
    cfg.expert_seed_episodes = 2
    cfg.total_steps = 6000                 # ~prefill 1320 + room for P1/P2/P3
    cfg._explicit_fields = {
        'baseline_seed_episodes', 'random_seed_episodes',
        'exploration_seed_episodes', 'phase1_frac', 'phase2_frac',
        'phase3_frac', 'gamma', 'policy_init_log_std',
        'policy_log_std_max', 'policy_log_std_min', 'pmpo_entropy_coef',
    }
    # --- THE CURRICULUM (requires the DOB) ---
    cfg.dob_enabled = True
    cfg.disturbance_head_dim = 0           # DOB retires the P87 head
    cfg.wm_recon_cv_weight = 4.0
    cfg.curriculum_enabled = True
    cfg.curriculum_stage2_disturbance_prob = 1.0
    cfg.curriculum_stage3_disturbance_prob = 0.85
    return cfg


def main():
    _setup_env()
    out_dir = tempfile.mkdtemp(prefix='curriculum_e2e_')
    from training.train import train
    cfg = _mk_cfg(out_dir)
    print(f'[e2e] running tiny PHASED curriculum train() on test_sim -> {out_dir}')
    # Capture stdout so we can assert on the [curriculum] STAGE banners.
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        train(cfg)
    out = buf.getvalue()
    print(out[-2000:])   # tail of the captured training log for context

    # ENABLED banner.
    assert '[curriculum] ENABLED' in out, \
        'curriculum never enabled (precondition gate misfired)'
    print('[e2e] OK  curriculum ENABLED banner present')

    # Stage transition banners for all three stages.
    for k in (1, 2, 3):
        assert f'>>> STAGE {k}' in out, \
            f'STAGE {k} transition banner missing — stage latch not wired'
    print('[e2e] OK  all three STAGE transition banners fired (1,2,3)')

    # Stage descriptors carry the right semantics.
    assert 'DOB suppressed' in out, 'Stage 1 descriptor wrong'
    assert 'g FROZEN' in out and 'observer' in out, 'Stage 2 descriptor wrong'
    assert 'FROZEN WM+DOB' in out, 'Stage 3 descriptor wrong'
    print('[e2e] OK  stage descriptors correct (S1 suppress / S2 freeze-g+id '
          'observer / S3 frozen WM+DOB)')

    log_path = os.path.join(out_dir, 'train_log.jsonl')
    rows = [json.loads(l) for l in open(log_path) if l.strip()]
    assert rows, 'train_log.jsonl empty'
    phases = sorted({int(r.get('phase', 0)) for r in rows})
    assert phases == [1, 2, 3], f'did not pass through all phases: {phases}'
    print(f'[e2e] OK  reached all three phases {phases} ({len(rows)} rows)')

    recons = [r.get('recon_loss') for r in rows
              if r.get('recon_loss') is not None]
    assert recons and all(v == v and abs(v) < 1e6 for v in recons), \
        f'recon_loss not finite across the curriculum: {recons[:5]}'
    print(f'[e2e] OK  recon_loss finite through all stages '
          f'(n={len(recons)}, last={recons[-1]:.4f})')

    froze = [r for r in rows if r.get('wm_frozen') is True]
    assert froze, 'wm_frozen never True — Stage 3 _wm_frozen_now latch not set'
    print(f'[e2e] OK  WM frozen by Stage 3 (wm_frozen=True in {len(froze)} rows)')

    print('\n[e2e] ALL IN-LOOP CURRICULUM PATHS EXECUTED — '
          'END-TO-END SMOKE PASSED')


if __name__ == '__main__':
    main()
