"""Bayesian Optimization driver for DreamerV4.

One axis (the only knob we tune):

  - model_size_preset : {S, M, L} — coordinated triples
                       ``(d_model, n_layers, n_heads, z_dim, n_register)``.

Lookback is pinned to ``seq_len`` (unified 2026-05-24): the world-model's
training context length and the deployment encoder/dynamics context length
are the same number, so the trained transformer attends over exactly the
same number of positions at inference as it did during training.  The
plant-identifier output is retained as a diagnostic to flag plants where
``derive_seq_len`` would under-size the WM memory horizon.
Horizon is pinned to the DreamerV3/V4 paper default ``H = 15`` (same
value ``workflow/single_run.py`` uses) — the plant-derived horizon
formula and the ``HORIZON_BAND`` BO axis were removed 2026-05-20 with
the short-budget knob cleanup; at the 1M-step default budget the
critic settles at H=15 on any plant.

All axes are seeded from plant identification (`dynamics_identifier` +
`lookback_identifier`).  No per-knob NN dimension search; no
auto-tuning band-aids.

For each trial we:
  1. set the model_size preset,
  2. train ``trial_total_steps`` with `training/train.train()`,
  3. score by the final EMA return on the policy buffer.

After the study, we re-train the best config for the full
``final_total_steps`` and export a single integrated ONNX artifact.
"""

from __future__ import annotations

import atexit
import json
import math
import os
import shutil
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import optuna

from training.train import TrainConfig, train as run_training
from inference.export_onnx import export_dreamer_v4_onnx
from models.dreamer_v4 import DreamerV4, DreamerV4Config
import torch


# ---------------------------------------------------------------------------
# Tee stdout/stderr to <out_dir>/workflow.log
# ---------------------------------------------------------------------------

class _Tee:
    """Duplicate writes to multiple streams.  Used to mirror stdout/stderr
    into ``workflow.log`` so the run directory is self-contained."""
    def __init__(self, *streams):
        self._streams = streams
    def write(self, s):
        for st in self._streams:
            try:
                st.write(s)
            except Exception:
                pass
    def flush(self):
        for st in self._streams:
            try:
                st.flush()
            except Exception:
                pass
    def isatty(self):
        return False


def _install_workflow_log(out_dir: Path) -> None:
    """Mirror stdout + stderr into ``<out_dir>/workflow.log``.

    Idempotent: subsequent calls overwrite the previous Tee target so the
    log always tracks the current run directory.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / 'workflow.log'
    f = open(log_path, 'a', buffering=1)  # line-buffered
    sys.stdout = _Tee(sys.__stdout__, f)
    sys.stderr = _Tee(sys.__stderr__, f)


# ---------------------------------------------------------------------------
# Model size presets (paper §C scales)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Model size presets — V4 transformer dims (paper-adapted to vector APC obs)
# ---------------------------------------------------------------------------
# Mapping from the plant-derived complexity score → S/M/L label
# (see ``utils.plant_init.derive_model_size``) → transformer dimensions.
# Used as the BO seed and as the discrete model-size axis for BO.
#
# Note: V4 paper (arXiv:2509.24527) targets billion-parameter video models;
# we scale these way down for low-D APC vector observations on a single GPU.
# Dimensions: ``d_model``, ``n_layers``, ``n_heads``, ``z_dim`` (tokenizer
# bottleneck), ``n_register``, plus tokenizer + head hidden width.

MODEL_SIZE_PRESETS: Dict[str, Dict[str, int]] = {
    'S': {'d_model': 128, 'n_layers': 4, 'n_heads': 4,
          'z_dim': 16, 'n_register': 2,
          'tok_hidden': 128, 'head_hidden': 128},
    'M': {'d_model': 256, 'n_layers': 6, 'n_heads': 8,
          'z_dim': 24, 'n_register': 4,
          'tok_hidden': 256, 'head_hidden': 256},
    'L': {'d_model': 384, 'n_layers': 8, 'n_heads': 8,
          'z_dim': 32, 'n_register': 4,
          'tok_hidden': 384, 'head_hidden': 384},
}

HORIZON_BAND = (1.0,)
# Horizon BO axis removed 2026-05-20: ``H`` is pinned to the V3/V4
# paper default of 15 (parity with ``workflow/single_run.py``).  This
# single-element tuple is retained for backward compatibility with any
# code that still references the symbol; ``run_trial`` no longer
# suggests over it.


# ---------------------------------------------------------------------------
# Plant initialization
# ---------------------------------------------------------------------------

# Plant-prep helpers live in ``workflow/_plant_prepare.py`` (shared with
# ``workflow/single_run.py``).  Re-exported here under their historical
# names so any external caller importing them from ``workflow.bo_runner``
# keeps working.
from workflow._plant_prepare import (  # noqa: E402
    identify_dynamics as identify_dynamics_from_plant,
    identify_lookback,
    build_noise_config as _build_noise_config,
    apply_dreamer_env_overrides,
)


def initialize_from_plant(out_dir: Path) -> Dict:
    """Backward-compat wrapper: dynamics + lookback identification using
    the env-provided ``SIM_SAMPLE_RATE`` (defaults to 5).

    New callers should use ``identify_dynamics_from_plant`` then
    ``identify_lookback`` after the sample_rate is derived so the
    lookback scan range reflects the actual agent timestep.
    """
    plant = identify_dynamics_from_plant(out_dir)
    sr = int(os.environ.get('SIM_SAMPLE_RATE', '5'))
    lb = identify_lookback(out_dir, tau=plant['tau'],
                           dead_time=plant['dead_time'], sample_rate=sr,
                           dynamics_raw=plant['dynamics_raw'],
                           tau_fast=plant.get('tau_fast'),
                           dead_time_fast=plant.get('dead_time_fast'))
    plant.update(lb)
    return plant


def horizon_init(tau: float, dead_time: float, sample_rate: int) -> int:
    """Imagination horizon (agent steps), adaptive to identified dynamics.

    Parity with ``workflow/single_run.py`` (2026-06-05): delegates to
    ``utils.auto_episode_length.derive_horizon``, which sizes the horizon to
    the identified 2%% settling time ``dead_time + 4*tau`` (converted to agent
    steps via ``sample_rate``) so the actor/critic credit the full settling
    response of the slowest loop.  Floored at the DreamerV3/V4 paper default
    (15) and capped by ``DREAMER_HORIZON_MAX``; the settle multiple is tunable
    via ``DREAMER_HORIZON_SETTLE_NTAU``.  An explicit ``DREAMER_HORIZON`` still
    hard-overrides downstream via ``apply_dreamer_env_overrides``.

    Returns the paper floor (15) when no usable dynamics are available.
    Signature retained so BO trial code that multiplies the base by
    ``horizon_mult ∈ HORIZON_BAND`` keeps working.
    """
    from utils.auto_episode_length import derive_horizon
    h, _src = derive_horizon(tau=tau, dead_time=dead_time,
                             sample_rate=max(1, int(sample_rate or 1)))
    return int(h)


def lookback_grid(plant_lookback: int) -> List[int]:
    """3-point grid around the plant-identified lookback."""
    a = max(8, int(round(0.75 * plant_lookback)))
    b = max(a + 1, int(plant_lookback))
    c = max(b + 1, int(round(1.5 * plant_lookback)))
    return sorted({a, b, c})


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Async per-trial validation
# ---------------------------------------------------------------------------
#
# After each BO trial finishes training we kick off ``evaluation.validate``
# in a *CPU-only* background subprocess.  This lets the next trial start
# training on the GPU immediately while validation data (plots, summary
# json, scripted-disturbance episode) is generated and written into the
# trial folder in parallel.
#
# Knobs (env vars):
#   BO_TRIAL_VALIDATION          : '0' to disable.       Default: enabled.
#   BO_TRIAL_VALIDATION_SEEDS    : seeds per trial.       Default: 2.
#   BO_TRIAL_VALIDATION_EPISODES : episodes per seed.     Default: 2.
#   BO_TRIAL_VALIDATION_MAXJOBS  : max in-flight jobs.    Default: 2.
#
# CPU-only is enforced by clearing CUDA_VISIBLE_DEVICES in the child env;
# this is critical because the next trial is GPU-bound and we don't want
# validation contending for VRAM.

_REPO_ROOT = Path(__file__).resolve().parent.parent
_pending_validations: List[Tuple[subprocess.Popen, Path]] = []


def _validation_enabled() -> bool:
    return os.environ.get('BO_TRIAL_VALIDATION', '1').strip().lower() not in (
        '0', 'false', 'no', '')


def _drain_validations(timeout: float | None = None,
                        max_in_flight: int = 0) -> None:
    """Reap finished validation subprocesses.

    ``max_in_flight``: if > 0, block until at most this many remain
    running.  ``timeout``: hard deadline in seconds (None = no deadline).
    """
    deadline = (time.monotonic() + timeout) if timeout is not None else None
    while _pending_validations:
        # Reap any that have finished.
        for i in range(len(_pending_validations) - 1, -1, -1):
            proc, tdir = _pending_validations[i]
            rc = proc.poll()
            if rc is not None:
                if rc == 0:
                    print(f'[trial-val] {tdir.name}: ok', flush=True)
                else:
                    print(f'[trial-val] {tdir.name}: failed rc={rc} '
                          f'(see {tdir}/validation.log)', flush=True)
                _pending_validations.pop(i)
        if len(_pending_validations) <= max_in_flight:
            return
        if deadline is not None and time.monotonic() >= deadline:
            print(f'[trial-val] drain timeout: '
                  f'{len(_pending_validations)} still running', flush=True)
            return
        time.sleep(1.0)


def _launch_trial_validation(trial_dir: Path) -> None:
    """Spawn a CPU-only ``evaluation.validate`` subprocess for ``trial_dir``."""
    if not _validation_enabled():
        return
    final_pt = trial_dir / 'final.pt'
    if not final_pt.exists():
        print(f'[trial-val] skip {trial_dir.name}: no final.pt', flush=True)
        return

    seeds = int(os.environ.get('BO_TRIAL_VALIDATION_SEEDS', '2') or 2)
    episodes = int(os.environ.get('BO_TRIAL_VALIDATION_EPISODES', '2') or 2)
    max_jobs = max(1, int(os.environ.get('BO_TRIAL_VALIDATION_MAXJOBS', '2') or 2))

    # Throttle: keep at most ``max_jobs - 1`` in flight before launching one
    # more, so the queue cannot grow unbounded if validation is slower than
    # training (shouldn't happen, but be defensive).
    _drain_validations(max_in_flight=max_jobs - 1)

    env = os.environ.copy()
    # Force CPU: validation must NOT compete with the next trial for VRAM.
    env['CUDA_VISIBLE_DEVICES'] = ''
    # Validation doesn't need compile / SDPA tweaks.
    env.pop('DREAMER_COMPILE', None)
    env.pop('DREAMER_FAST_ATTN', None)

    log_path = trial_dir / 'validation.log'
    log_f = open(log_path, 'w', buffering=1)
    cmd = [sys.executable, '-m', 'evaluation.validate',
           '--controller-dir', str(trial_dir),
           '--episodes', str(episodes),
           '--seeds', str(seeds)]
    try:
        proc = subprocess.Popen(cmd, env=env, cwd=str(_REPO_ROOT),
                                 stdout=log_f, stderr=subprocess.STDOUT)
    except Exception as e:
        print(f'[trial-val] spawn failed for {trial_dir.name}: {e!r}',
              flush=True)
        log_f.close()
        return
    _pending_validations.append((proc, trial_dir))
    print(f'[trial-val] launched (cpu) for {trial_dir.name} '
          f'pid={proc.pid} seeds={seeds} episodes={episodes} '
          f'-> {log_path.name}', flush=True)


def _atexit_cleanup() -> None:
    for proc, tdir in list(_pending_validations):
        if proc.poll() is None:
            try:
                proc.terminate()
            except Exception:
                pass


atexit.register(_atexit_cleanup)


# ---------------------------------------------------------------------------
# Trial runner
# ---------------------------------------------------------------------------

def make_trial_config(base: TrainConfig, *, lookback: int,
                      model_size: str, horizon: int,
                      total_steps: int, out_dir: Path,
                      batch_size: int | None = None) -> TrainConfig:
    preset = MODEL_SIZE_PRESETS[model_size]
    overrides: Dict[str, object] = {
        'lookback': int(lookback),
        'horizon': int(horizon),
        'total_steps': int(total_steps),
        'out_dir': str(out_dir),
        **preset,
    }
    if batch_size is not None:
        overrides['batch_size'] = int(batch_size)
    cfg = TrainConfig(**{**asdict(base), **overrides})
    return cfg


def run_trial(trial: optuna.Trial, base: TrainConfig, plant: Dict,
              study_dir: Path, trial_steps: int) -> float:
    from tools.gpu_calibrate import pick_batch_size_for_plant
    # ``lookback`` is unified with ``seq_len`` (2026-05-24): there is one
    # history-window length, used identically by the training transformer
    # context (over seq_len positions) and the deployment encoder/dynamics.
    # The plant-identifier output is retained only as a diagnostic.  BO
    # therefore has no separate ``lookback`` axis to optimize.
    lookback = int(plant['lookback'])
    model_size = trial.suggest_categorical('model_size', list(MODEL_SIZE_PRESETS))
    # Imagination horizon: adaptive to the identified settling time (parity
    # with workflow/single_run.py — dead_time + 4*tau in agent steps),
    # floored at the paper default 15.  γ auto-adapts to this horizon inside
    # train() (P48); an explicit DREAMER_HORIZON still hard-overrides via
    # apply_dreamer_env_overrides below.
    horizon = horizon_init(float(plant.get('tau', 0.0)),
                           float(plant.get('dead_time', 0.0)),
                           int(base.sample_rate))
    H_init = horizon

    # Empirical per-trial batch sizing (cached on (model_size, seq, lb,
    # horizon) so repeated trials with same shape skip the ~10 s probe).
    bs_env = os.environ.get('OBJ_BATCH_SIZE', '').strip()
    if bs_env:
        try:
            bs = max(1, int(bs_env))
            bs_info = {'batch_size': bs, 'source': 'env_override'}
        except Exception:
            bs_info = pick_batch_size_for_plant(
                model_size=model_size, seq_len=int(base.seq_len),
                lookback=lookback, horizon=horizon, k_max=int(base.k_max),
                sample_rate=int(base.sample_rate),
                episode_length=int(base.episode_length))
            bs = int(bs_info['batch_size'])
    else:
        bs_info = pick_batch_size_for_plant(
            model_size=model_size, seq_len=int(base.seq_len),
            lookback=lookback, horizon=horizon, k_max=int(base.k_max),
            sample_rate=int(base.sample_rate),
            episode_length=int(base.episode_length))
        bs = int(bs_info['batch_size'])

    trial_dir = study_dir / f'trial_{trial.number:04d}'
    trial_dir.mkdir(parents=True, exist_ok=True)
    print(f"[trial {trial.number}] lookback={lookback} model={model_size} "
          f"horizon={horizon} batch={bs} ({bs_info['source']}; "
          f"per_batch≈{bs_info.get('per_batch_mb',0):.0f}MB)", flush=True)
    cfg = make_trial_config(base, lookback=lookback, model_size=model_size,
                            horizon=horizon, total_steps=trial_steps,
                            out_dir=trial_dir, batch_size=bs)
    # Apply DREAMER_* env-var overrides per-trial (parity with
    # ``workflow/single_run.py``).  Without this BO trials silently
    # ignore the user's paper-strip-back / A-B overrides
    # (DREAMER_GAE_LAMBDA, DREAMER_PHASE*_FRAC, DREAMER_LR_*,
    # DREAMER_HORIZON, σ-max, ...).
    apply_dreamer_env_overrides(cfg)

    # Pruning hook: report the running EMA return after each log iter so the
    # MedianPruner can stop visibly-bad trials early.  We track the last
    # ``step`` value we reported because in P3 we collect a fresh
    # on-policy episode only every K iters (``phase3_collect_every_iters``)
    # while we still log every iter — this used to spam Optuna with
    # duplicate-step warnings (``UserWarning: The reported value is
    # ignored because this `step` X is already reported``).
    last_reported_step = {'v': -1}

    def _on_iter(it: int, steps: int, ema: float) -> bool:
        try:
            if int(steps) > last_reported_step['v']:
                trial.report(float(ema), step=int(steps))
                last_reported_step['v'] = int(steps)
        except Exception:
            pass
        return bool(trial.should_prune())

    try:
        summary = run_training(cfg, on_iter_end=_on_iter)
    except optuna.TrialPruned:
        with open(trial_dir / 'trial_summary.json', 'w') as f:
            json.dump({'params': trial.params, 'horizon_concrete': horizon,
                       'H_init': H_init, 'pruned': True}, f, indent=2)
        raise
    except torch.cuda.OutOfMemoryError as e:
        # Don't kill the whole BO study on a single OOM trial.  Free GPU
        # memory, mark this trial as failed in Optuna (worst score), and
        # let the next trial run with a clean slate.
        try:
            import gc
            gc.collect()
            torch._dynamo.reset()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        except Exception:
            pass
        print(f'[trial {trial.number}] OOM: {e!r}; marking failed',
              flush=True)
        with open(trial_dir / 'trial_summary.json', 'w') as f:
            json.dump({'params': trial.params, 'horizon_concrete': horizon,
                       'H_init': H_init, 'oom': True,
                       'error': str(e)[:500]}, f, indent=2)
        return float('-inf')

    # Score by the rolling-window mean of the last 10 *policy* episode
    # returns, not the single final EMA value.  This dampens the
    # seed-variance that previously caused identical configs to disagree
    # by hundreds of points (e.g. trial #1 +60.85 vs trial #5 −127.50 in
    # the bo_20260429_184201 run).  Falls back to final_ema_return when
    # the window is empty (very short trials).
    score = summary.get('final_return_window_mean')
    if score is None or not np.isfinite(score):
        score = summary.get('final_ema_return')
    if score is None or not np.isfinite(score):
        score = float('-inf')
    with open(trial_dir / 'trial_summary.json', 'w') as f:
        json.dump({'params': trial.params, 'horizon_concrete': horizon,
                   'H_init': H_init, 'summary': summary,
                   'score': float(score),
                   'score_metric': 'final_return_window_mean'}, f, indent=2)
    # Spawn CPU-only validation for this trial; it runs in parallel with the
    # next trial's GPU training.  Drained before final retrain.
    _launch_trial_validation(trial_dir)

    # Inter-trial GPU cleanup: torch.compile / inductor cache stale kernels
    # for previous trials' shapes (different model_size / horizon / batch),
    # which accumulate across trials and eventually OOM.  Reset dynamo state
    # and free CUDA memory to give the next trial a clean slate.
    try:
        import gc
        gc.collect()
        torch._dynamo.reset()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception as e:
        print(f'[trial-gc] cleanup skipped: {e!r}', flush=True)
    return float(score)


# ---------------------------------------------------------------------------
# Final retrain + ONNX export
# ---------------------------------------------------------------------------

def train_final_and_export(base: TrainConfig, plant: Dict, best_params: Dict,
                           out_dir: Path, total_steps: int) -> Dict:
    from tools.gpu_calibrate import pick_batch_size_for_plant
    # ``lookback`` is no longer a BO axis (see ``run_trial``); read it from
    # the plant identification, falling back to ``best_params`` for
    # backward-compat with older study summaries that still recorded it.
    lookback = int(best_params.get('lookback', plant['lookback']))
    model_size = str(best_params['model_size'])
    # Imagination horizon: adaptive to the identified settling time (parity
    # with workflow/single_run.py and run_trial).  γ auto-adapts to this
    # horizon inside train() (P48); an explicit DREAMER_HORIZON still
    # hard-overrides via apply_dreamer_env_overrides.
    horizon = horizon_init(float(plant.get('tau', 0.0)),
                           float(plant.get('dead_time', 0.0)),
                           int(base.sample_rate))
    H_init = horizon

    bs_env = os.environ.get('OBJ_BATCH_SIZE', '').strip()
    if bs_env:
        try:
            bs = max(1, int(bs_env))
        except Exception:
            bs = int(pick_batch_size_for_plant(
                model_size=model_size, seq_len=int(base.seq_len),
                lookback=lookback, horizon=horizon, k_max=int(base.k_max),
                sample_rate=int(base.sample_rate),
                episode_length=int(base.episode_length))['batch_size'])
    else:
        bs = int(pick_batch_size_for_plant(
            model_size=model_size, seq_len=int(base.seq_len),
            lookback=lookback, horizon=horizon, k_max=int(base.k_max),
            sample_rate=int(base.sample_rate),
            episode_length=int(base.episode_length))['batch_size'])
    print(f'[final] model={model_size} lookback={lookback} horizon={horizon} '
          f'batch={bs}', flush=True)

    final_dir = out_dir / 'final'
    final_dir.mkdir(parents=True, exist_ok=True)
    cfg = make_trial_config(base, lookback=lookback, model_size=model_size,
                            horizon=horizon, total_steps=total_steps,
                            out_dir=final_dir, batch_size=bs)
    # Parity with single_run.py / run_trial: honour DREAMER_* env vars
    # on the final retrain too.
    apply_dreamer_env_overrides(cfg)
    summary = run_training(cfg)

    # Reload model from final.pt and export ONNX.
    ckpt = torch.load(final_dir / 'final.pt', map_location='cpu', weights_only=False)
    cfg_loaded = TrainConfig(**{k: v for k, v in ckpt['cfg'].items()
                                 if k in {f for f in TrainConfig.__dataclass_fields__}})
    model_cfg = DreamerV4Config(
        obs_dim=cfg_loaded.obs_dim, action_dim=cfg_loaded.action_dim,
        lookback=cfg_loaded.lookback,
        tok_hidden=cfg_loaded.tok_hidden, z_dim=cfg_loaded.z_dim,
        mae_p_max=cfg_loaded.mae_p_max,
        d_model=cfg_loaded.d_model, n_layers=cfg_loaded.n_layers,
        n_heads=cfg_loaded.n_heads, ff_mult=cfg_loaded.ff_mult,
        n_register=cfg_loaded.n_register,
        k_max=cfg_loaded.k_max, tau_n_bins=cfg_loaded.tau_n_bins,
        soft_cap=cfg_loaded.soft_cap,
        n_action_bins=cfg_loaded.n_action_bins,
        head_hidden=cfg_loaded.head_hidden,
        head_n_layers=cfg_loaded.head_n_layers,
        mtp_length=max(1, int(getattr(cfg_loaded, 'mtp_length', 1))),
        policy_type=str(getattr(cfg_loaded, 'policy_type', 'continuous')),
        policy_init_log_std=float(
            getattr(cfg_loaded, 'policy_init_log_std', -0.5)),
        policy_log_std_min=float(
            getattr(cfg_loaded, 'policy_log_std_min', -2.3)),
        policy_log_std_max=float(
            getattr(cfg_loaded, 'policy_log_std_max', 0.0)),
        world_model_type=str(
            getattr(cfg_loaded, 'world_model_type', 'sf_transformer')),
        rssm_deter_dim=int(getattr(cfg_loaded, 'rssm_deter_dim', 512)),
        rssm_n_categoricals=int(
            getattr(cfg_loaded, 'rssm_n_categoricals', 32)),
        rssm_n_classes=int(getattr(cfg_loaded, 'rssm_n_classes', 32)),
        rssm_embed_dim=int(getattr(cfg_loaded, 'rssm_embed_dim', 256)),
        rssm_hidden_dim=int(getattr(cfg_loaded, 'rssm_hidden_dim', 256)),
        rssm_unimix=float(getattr(cfg_loaded, 'rssm_unimix', 0.01)),
        disturbance_head_dim=int(getattr(cfg_loaded, 'disturbance_head_dim', 0) or 0),
        disturbance_head_hidden=int(getattr(cfg_loaded, 'disturbance_head_hidden', 0) or 0),
        disturbance_head_layers=int(getattr(cfg_loaded, 'disturbance_head_layers', 2) or 2),
        dv_dim=int(getattr(cfg_loaded, 'dv_dim', 0) or 0),
        dv_indices=tuple(getattr(cfg_loaded, 'dv_indices', ()) or ()),
        dv_feedforward=bool(getattr(cfg_loaded, 'dv_feedforward', True)),
        dob_enabled=bool(getattr(cfg_loaded, 'dob_enabled', False)),
        cv_obs_indices=tuple(getattr(cfg_loaded, 'cv_obs_indices', ()) or ()),
        dob_decay_init=float(getattr(cfg_loaded, 'dob_decay_init', 3.0)),
        dob_gain_init=float(getattr(cfg_loaded, 'dob_gain_init', -2.2)),
        cont_gain_dim=int(getattr(cfg_loaded, 'cont_gain_dim', 0) or 0),
        cont_dist_dim=int(getattr(cfg_loaded, 'cont_dist_dim', 0) or 0),
        cont_min_std=float(getattr(cfg_loaded, 'cont_min_std', 0.1)),
        cont_max_std=float(getattr(cfg_loaded, 'cont_max_std', 2.0)),
        attn_impl='manual',  # ONNX export: manual path is safer than SDPA
    )
    model = DreamerV4(model_cfg)
    sd = ckpt['model']
    if any('._orig_mod.' in k for k in sd):
        sd = {k.replace('._orig_mod.', '.'): v for k, v in sd.items()}
    model.load_state_dict(sd)
    onnx_path = out_dir / 'dreamer_v4.onnx'
    export_dreamer_v4_onnx(model, onnx_path)

    # Auto-validation on held-out seeds (paper-faithful "test set" gate).
    val_summary: Dict = {}
    try:
        from evaluation.validate import run_validation
        print('[final] phase 4: validation on held-out seeds', flush=True)
        val_summary = run_validation(controller_dir=final_dir,
                                     episodes=3, seeds=3, ckpt='best.pt')
        print(f"[final] validation cum_raw_reward "
              f"mean={val_summary.get('cum_raw_reward_mean', float('nan')):.2f} "
              f"std={val_summary.get('cum_raw_reward_std', float('nan')):.2f} "
              f"-> {final_dir}/validation/", flush=True)
    except Exception as e:
        print(f'[final] validation skipped: {e!r}', flush=True)

    return {
        'final_summary': summary,
        'best_params': best_params,
        'horizon_concrete': horizon,
        'H_init': H_init,
        'onnx_path': str(onnx_path),
        'final_ckpt': str(final_dir / 'final.pt'),
        'validation': val_summary,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_bo(out_dir: str | Path, n_trials: int = 8,
           trial_steps: int = 0, final_steps: int = 0,
           study_name: str = 'dreamer_v4_bo',
           init_from_ckpt: str = '') -> Dict:
    """Run BO.  ``trial_steps`` / ``final_steps`` ≤ 0 → plant-tied auto.

    ``init_from_ckpt``: optional path to a previous run's ``best.pt`` /
    ``final.pt`` to warm-start *every* trial's model weights from.
    Optimisers / replay buffer / counters still start fresh per trial;
    only the network parameters are loaded (``strict=False``).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _install_workflow_log(out_dir)
    print(f'[runner] workflow log: {out_dir}/workflow.log', flush=True)

    base = TrainConfig()
    if init_from_ckpt:
        ckpt_path = str(init_from_ckpt).strip()
        if not Path(ckpt_path).exists():
            raise FileNotFoundError(
                f'--init-from-ckpt path does not exist: {ckpt_path!r}')
        base.init_from_ckpt = ckpt_path
        print(f'[BO] init_from_ckpt={ckpt_path} '
              f'(applied to every trial + final retrain)', flush=True)
    # Sample-rate env override is supported here for sims that hard-code their
    # scan rate.  Episode length is auto-derived from identification (or from
    # SIM_EPISODE_LENGTH env via derive_episode_length).
    sr_env = os.environ.get('SIM_SAMPLE_RATE', '').strip()

    print('[BO] Phase 1a: dynamics identification', flush=True)
    plant = identify_dynamics_from_plant(out_dir / 'plant_id')

    # Plant-aware noise config — see ``workflow/_plant_prepare.build_noise_config``.
    # Builds dynamics-derived OU + measurement noise and exports
    # ``SIM_NOISE_CONFIG_JSON`` so every downstream subprocess (training,
    # validation) loads the same noise profile via ``SimNoiseWrapper``.
    # Without this step the wrapper falls back to an empty config and the
    # agent sees zero process / measurement / DV-drift noise — making
    # "the plant" deterministic except for discrete disturbance steps.
    _build_noise_config(out_dir,
                         dynamics_raw=plant.get('dynamics_raw') or {},
                         sample_rate=base.sample_rate,
                         log_prefix='[BO]')

    # Plant-tied derivations (sample_rate from fastest dynamics, model_size
    # from complexity, seq_len ≥ settling time).  Env-supplied values take
    # precedence so this stays simulator-agnostic.
    from utils.sim_factory import create_sim, resolve_sim_metadata
    from utils.plant_init import derive_all, derive_step_budgets
    from tools.gpu_calibrate import pick_batch_size_for_plant
    sr_override = int(sr_env) if sr_env else 0
    tmp_sim = create_sim(episode_length=10,
                         sample_rate=max(1, sr_override or base.sample_rate))
    sim_meta = resolve_sim_metadata(tmp_sim)
    derived = derive_all(plant.get('dynamics_raw') or {}, sim_meta,
                         sample_rate_override=sr_override)
    sr = derived['sample_rate']
    base.sample_rate = sr
    base.seq_len = derived['seq_len']
    base.k_max = derived['k_max']
    os.environ['SIM_SAMPLE_RATE'] = str(sr)

    # Phase 1b: Lookback identification (DIAGNOSTIC ONLY) — historically
    # this set the runtime history-window independently from seq_len; as
    # of 2026-05-24 the two are unified so train/deploy context lengths
    # match exactly.  We still run the identifier to populate the report
    # and to flag plants where the identified memory horizon exceeds the
    # derived seq_len (i.e. ``derive_seq_len`` would be undersizing).
    print('[BO] Phase 1b: lookback identification (diagnostic)', flush=True)
    lb_info = identify_lookback(out_dir / 'plant_id',
                                 tau=plant['tau'],
                                 dead_time=plant['dead_time'],
                                 sample_rate=sr,
                                 dynamics_raw=plant.get('dynamics_raw') or {},
                                 tau_fast=plant.get('tau_fast'),
                                 dead_time_fast=plant.get('dead_time_fast'))
    plant.update(lb_info)
    identified_lookback = int(plant['lookback'])
    # Unified history-window length: train context = deploy context.
    plant['identified_lookback'] = identified_lookback
    plant['lookback'] = int(base.seq_len)
    if identified_lookback > int(base.seq_len):
        print(f"[BO] WARNING identified_lookback={identified_lookback} > "
              f"seq_len={base.seq_len}; ``derive_seq_len`` may be "
              f"undersizing the WM context.", flush=True)
    print(f"[BO] lookback={plant['lookback']} (=seq_len; "
          f"identified={identified_lookback}, sr={sr})", flush=True)

    # Episode length: env override > auto-derived from settling time > paper.
    from utils.auto_episode_length import derive_episode_length
    ep_len, ep_source = derive_episode_length()
    base.episode_length = int(ep_len)
    os.environ['SIM_EPISODE_LENGTH'] = str(base.episode_length)
    print(f"[BO] episode_length={base.episode_length} ({ep_source})", flush=True)

    derived_model_size = derived['model_size']
    print(f"[BO] derived: sample_rate={sr} ({derived['sample_rate_source']}) "
          f"seq_len={base.seq_len} k_max={base.k_max} "
          f"model_size_seed={derived_model_size} "
          f"complexity={derived['complexity_score']:.2f}", flush=True)

    # Adaptive batch size SEED (per-trial batch is re-derived in run_trial
    # from the trial's actual model_size + horizon via the empirical
    # probe).  This block only records the seed-config batch in
    # run_plan.json.  Size the probe at the adaptive horizon (parity with
    # run_trial / single_run) so the recorded seed batch is realistic.
    seed_horizon = horizon_init(float(plant.get('tau', 0.0)),
                                float(plant.get('dead_time', 0.0)),
                                int(base.sample_rate))
    bs_env = os.environ.get('OBJ_BATCH_SIZE', '').strip()
    if bs_env:
        try:
            base.batch_size = max(1, int(bs_env))
            bs_info = {'batch_size': base.batch_size, 'source': 'env_override'}
        except Exception:
            bs_info = pick_batch_size_for_plant(
                model_size=derived_model_size,
                seq_len=int(base.seq_len), lookback=int(plant['lookback']),
                horizon=int(seed_horizon), k_max=int(base.k_max),
                sample_rate=int(base.sample_rate),
                episode_length=int(base.episode_length))
            base.batch_size = int(bs_info['batch_size'])
    else:
        bs_info = pick_batch_size_for_plant(
            model_size=derived_model_size,
            seq_len=int(base.seq_len), lookback=int(plant['lookback']),
            horizon=int(seed_horizon), k_max=int(base.k_max),
            sample_rate=int(base.sample_rate),
            episode_length=int(base.episode_length))
        base.batch_size = int(bs_info['batch_size'])
    print(f"[BO] batch_size_seed={base.batch_size} ({bs_info['source']}; "
          f"per_batch≈{bs_info.get('per_batch_mb',0):.0f}MB, "
          f"gpu={bs_info.get('gpu_total_gb',0):.1f}GB; "
          f"per-trial batch re-derived from trial config)", flush=True)

    # Plant-tied step budgets (trial_steps / final_steps).  CLI / caller
    # values > 0 take precedence so power users can still override.
    budgets = derive_step_budgets(
        episode_length=base.episode_length,
        complexity_score=derived['complexity_score'],
    )
    if int(trial_steps) > 0:
        budgets['trial_steps'] = int(trial_steps)
        budgets['source'] = 'override'
    if int(final_steps) > 0:
        budgets['final_steps'] = int(final_steps)
        budgets['source'] = 'override'
    trial_steps = int(budgets['trial_steps'])
    final_steps = int(budgets['final_steps'])
    print(f"[BO] step budgets: trial={trial_steps} final={final_steps} "
          f"({budgets['source']}; trial_eps_target={budgets['trial_episodes_target']}, "
          f"final_eps_target={budgets['final_episodes_target']})", flush=True)

    with open(out_dir / 'plant_id.json', 'w') as f:
        json.dump({**plant, 'derived': derived, 'step_budgets': budgets,
                   'batch_size': bs_info},
                  f, indent=2, default=str)

    # Workflow-level run_plan.json — single source of truth for everything
    # downstream (trials, validation, ONNX export) reads from.
    plan = {
        'mode': 'bo',
        'simulation_dir': os.environ.get('SIMULATION_DIR'),
        'simulation_name': Path(os.environ.get('SIMULATION_DIR', 'unknown')).name,
        'out_dir': str(out_dir),
        'sample_rate': sr,
        'sample_rate_source': derived['sample_rate_source'],
        # Deployment manifest (canonical runtime-API names).  Phase 1
        # unification (2026-05-24): ``lookback == seq_len`` so train and
        # deploy contexts are bit-identical.
        'sample_rate_seconds': sr,
        'history_window_samples': plant['lookback'],
        'tau': plant['tau'], 'dead_time': plant['dead_time'],
        'tau_fast': plant['tau_fast'], 'dead_time_fast': plant['dead_time_fast'],
        'lookback': plant['lookback'],
        'identified_lookback': plant.get('identified_lookback',
                                          plant['lookback']),
        'episode_length': base.episode_length,
        'episode_length_source': ep_source,
        'seq_len': base.seq_len,
        'k_max': base.k_max,
        'model_size_seed': derived_model_size,
        'model_size_source': 'auto:complexity',
        'complexity_score': derived['complexity_score'],
        'complexity_inputs': derived['inputs'],
        'batch_size': base.batch_size,
        'batch_size_source': bs_info['source'],
        'trial_steps': trial_steps, 'final_steps': final_steps,
        'step_budget_source': budgets['source'],
        'n_trials': int(n_trials),
        'seed': int(os.environ.get('SEED', '0')),
    }
    with open(out_dir / 'run_plan.json', 'w') as f:
        json.dump(plan, f, indent=2, default=str)
    print(f"[BO] plant: tau={plant['tau']:.2f}  dead_time={plant['dead_time']:.2f}  "
          f"lookback={plant['lookback']}  H=15 (paper default)", flush=True)

    print('[BO] Phase 2: Optuna search', flush=True)
    # Persistent SQLite study so BO can resume after a crash / SIGINT.
    # ``load_if_exists=True`` reattaches to an existing study at the same
    # path with all completed/pruned trial history (TPE prior +
    # MedianPruner statistics intact). To start a clean study, delete
    # ``out_dir/study.db`` or pass a fresh ``--out`` directory.
    study_db = out_dir / 'study.db'
    storage_url = f'sqlite:///{study_db}'
    study = optuna.create_study(
        study_name=study_name, direction='maximize',
        storage=storage_url, load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=int(os.environ.get('SEED', '0'))),
        # Prune trials whose intermediate EMA return is below the median of
        # completed trials at the same step, after the first 3 trials have
        # finished and ≥ 5 reports have come in.
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=3, n_warmup_steps=5, interval_steps=1),
    )
    study_dir = out_dir / 'trials'
    study_dir.mkdir(parents=True, exist_ok=True)

    # Resume bookkeeping: count COMPLETE/PRUNED trials and detect any
    # in-flight RUNNING/FAIL state from a previous (crashed) process so
    # we don't double-count the budget.
    import optuna.trial as _otrial
    prior_done = sum(1 for t in study.trials
                     if t.state in (_otrial.TrialState.COMPLETE,
                                    _otrial.TrialState.PRUNED))
    prior_failed = sum(1 for t in study.trials
                       if t.state == _otrial.TrialState.FAIL)
    prior_running = [t for t in study.trials
                     if t.state == _otrial.TrialState.RUNNING]
    if prior_done or prior_failed or prior_running:
        print(f'[BO] resuming study {study_name!r} from {study_db.name}: '
              f'{prior_done} done, {prior_failed} failed, '
              f'{len(prior_running)} stale-running '
              f'({len(study.trials)} total in DB)', flush=True)
        # Mark stale RUNNING trials (left over from a crashed process)
        # as FAIL so Optuna's running-trial counter is consistent and
        # the next attempt may reuse those param combinations.
        for t in prior_running:
            try:
                study._storage.set_trial_state_values(
                    t._trial_id, state=_otrial.TrialState.FAIL)
                print(f'[BO]   marked stale-running trial #{t.number} as FAIL',
                      flush=True)
            except Exception as e:
                print(f'[BO]   could not mark trial #{t.number} FAIL: {e!r}',
                      flush=True)
    remaining = max(0, int(n_trials) - prior_done)
    if prior_done == 0:
        # Fresh study: seed the first trial at the plant-derived
        # configuration (model_size from complexity, horizon at the
        # centre of the plant band). Optuna explores the rest from
        # there. Skip on resume to avoid duplicating an already-run
        # trial.
        study.enqueue_trial({
            'model_size': derived_model_size,
            'horizon_mult': 1.0,
        })
    if remaining > 0:
        print(f'[BO] running {remaining} more trial(s) '
              f'(target n_trials={n_trials}, already done={prior_done})',
              flush=True)
        study.optimize(lambda t: run_trial(t, base, plant, study_dir, trial_steps),
                       n_trials=remaining, show_progress_bar=False)
    else:
        print(f'[BO] target n_trials={n_trials} already reached '
              f'({prior_done} done); skipping search', flush=True)

    best = study.best_trial
    print(f"[BO] best trial #{best.number}: score={best.value:.2f}  params={best.params}",
          flush=True)
    with open(out_dir / 'study_summary.json', 'w') as f:
        json.dump({'best_trial': best.number, 'best_score': float(best.value),
                   'best_params': best.params,
                   'all_trials': [{'number': t.number, 'value': t.value,
                                   'params': t.params, 'state': str(t.state)}
                                  for t in study.trials]}, f, indent=2)

    print('[BO] Phase 3: retrain best config and export ONNX', flush=True)
    # Make sure all per-trial validations have completed before we hand
    # the GPU back to the final retrain.  CPU-only, so this should be a
    # no-op in steady state, but a safety net is cheap.
    print(f'[BO] draining {len(_pending_validations)} async trial '
          f'validations…', flush=True)
    _drain_validations(max_in_flight=0)
    final = train_final_and_export(base, plant, best.params, out_dir, final_steps)
    with open(out_dir / 'workflow_summary.json', 'w') as f:
        json.dump({'plant': plant, 'study_best': best.params,
                   'study_best_score': float(best.value),
                   'final': final}, f, indent=2)
    print(f"[BO] done. ONNX -> {final['onnx_path']}", flush=True)
    return final


if __name__ == '__main__':
    import argparse
    import time as _time
    p = argparse.ArgumentParser(
        description='Full DreamerV4 workflow: plant identification → '
                    'Optuna BO over (lookback, model_size, horizon) → '
                    'final retrain → ONNX export.')
    p.add_argument('--simulation-dir', '-s', default=None,
                   help='Simulation directory (e.g. simulation/test_sim). '
                        'Required unless --out is given (legacy mode where '
                        'CONTROL_SETUP_JSON / SIMULATION_DIR are set externally).')
    p.add_argument('--out', '-o', default=None,
                   help='Workflow output directory. Default: '
                        '<repo>/output/<sim>/bo_<timestamp>/')
    p.add_argument('--n_trials', type=int, default=8)
    p.add_argument('--trial_steps', type=int, default=1_000_000,
                   help='per-trial training steps (default 1,000,000 — '
                        'parity with workflow/single_run.py; 0 = '
                        'plant-tied auto)')
    p.add_argument('--final_steps', type=int, default=1_000_000,
                   help='final retrain steps (default 1,000,000 — '
                        'parity with workflow/single_run.py; 0 = '
                        'plant-tied auto)')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--init-from-ckpt', type=str, default='',
                   help="Optional path to a previous run's best.pt / "
                        'final.pt to warm-start every trial (and the '
                        'final retrain) from. Model weights only; '
                        'optimisers/buffer/counters start fresh.')
    args = p.parse_args()

    repo = Path(__file__).resolve().parent.parent
    if args.simulation_dir is not None:
        from workflow.single_run import _resolve_sim_dir
        sim_dir = _resolve_sim_dir(args.simulation_dir)
        setup_path = sim_dir / 'control_setup.json'
        obj_path = sim_dir / 'control_objective.json'
        if not setup_path.exists():
            raise FileNotFoundError(f'Missing {setup_path}')
        if not obj_path.exists():
            raise FileNotFoundError(f'Missing {obj_path}')
        os.environ['CONTROL_SETUP_JSON'] = str(setup_path)
        os.environ['CONTROL_OBJECTIVE_JSON'] = str(obj_path)
        os.environ['SIMULATION_DIR'] = str(sim_dir)
        os.environ.setdefault('SEED', str(args.seed))
        sim_name = sim_dir.name
    else:
        sim_name = Path(os.environ.get('SIMULATION_DIR', 'unknown')).name

    if args.out:
        out_dir = args.out
    else:
        ts = _time.strftime('%Y%m%d_%H%M%S')
        out_dir = str(repo / 'output' / sim_name / f'bo_{ts}')

    print(f'[runner] sim={sim_name}  out_dir={out_dir}', flush=True)
    run_bo(out_dir, n_trials=args.n_trials,
           trial_steps=args.trial_steps, final_steps=args.final_steps,
           init_from_ckpt=args.init_from_ckpt)
