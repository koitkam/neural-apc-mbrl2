"""WM posterior-vs-prior lag probe — localize WHERE the world model loses the
steady-state gain.

On a clean held-MV step test (real sim), decompose the WM into three nested
predictions of the SAME real trajectory and compare their CV step-response
gain to the real plant:

  1. POSTERIOR reconstruction  (0-step, teacher-forced): decode(post.feat).
     The posterior SEES the real obs each step, so this tests the
     encoder+decoder+latent capacity ONLY — the "can the posterior even
     represent the gain?" check.  If THIS is already attenuated, the bottleneck
     is upstream of the prior and free_bits CANNOT help.
  2. PRIOR 1-step prediction  (teacher-forced h, prior z): decode(prior.feat).
     Same deterministic h as the posterior (obs_step shares h); the ONLY
     difference is z came from the prior (no obs) vs the posterior (sees obs).
     gap(posterior, prior_1step) == exactly what the free_bits KL floor
     controls — the prior's inability to reproduce the posterior latent.
  3. PRIOR open-loop rollout  (N-step imagination): the transfer-matrix WM
     curve.  gap(prior_1step, open_loop) == compounding/contraction over the
     horizon.

Also reports the literal posterior↔prior latent KL per step (kl_dyn_raw) vs the
model's free_bits floor — i.e. whether the prior is pinned at the floor.

RSSM/TSSM only (needs obs_step/decode).  CPU-safe; does not touch a GPU run.

Usage:
  CUDA_VISIBLE_DEVICES="" PYTHONPATH=$PWD \
  ./../neural-apc-mbrl-env/bin/python tools/wm_posterior_prior_probe.py \
      --run-dir output/test_sim/run_20260608_p102_joint_distfix --ckpt best.pt
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from tools.wm_steady_state_diagnostic import (  # noqa: E402
    _find_ckpt, _load_model, _pick_device, _imagine_open_loop_rssm,
    _is_rssm_model,
)
from evaluation.wm_transfer_matrix import _settle_capture, _real_step_rollout  # noqa: E402


def _cv_gain(curve_cv: np.ndarray, baseline: float, tail_frac: float = 0.25) -> float:
    n = len(curve_cv)
    tail = curve_cv[-max(1, int(tail_frac * n)):]
    return float(tail.mean() - baseline)


@torch.no_grad()
def _teacher_forced_post_prior(model, lookback_obs, lookback_act,
                               step_obs, step_act, device):
    """Warm-start the posterior over the lookback, then teacher-force the
    posterior through the real STEP trajectory, decoding BOTH the posterior
    feature (0-step recon) and the prior feature (1-step prediction) each step.

    Returns (post_decode (H,O), prior_decode (H,O), kl_per_step (H,))."""
    rssm = model.dynamics
    obs_dim = rssm.obs_dim
    _dv_on = int(getattr(rssm, 'dv_dim', 0) or 0) > 0
    state = rssm.initial_state(1, device)

    def _emb_dv(o_np):
        o = torch.from_numpy(np.asarray(o_np, 'float32')).to(device).unsqueeze(0)
        emb = rssm.embed(o)
        dv = o.index_select(-1, rssm.dv_index_t) if _dv_on else None
        return emb, dv

    for l in range(lookback_obs.shape[0]):
        a = torch.from_numpy(lookback_act[l]).to(device).unsqueeze(0)
        emb, dv = _emb_dv(lookback_obs[l])
        post, _ = rssm.obs_step(state, a, emb, dv=dv, sample=True)
        state = post

    H = step_obs.shape[0]
    post_dec = np.zeros((H, obs_dim), dtype='float32')
    prior_dec = np.zeros((H, obs_dim), dtype='float32')
    kl = np.zeros(H, dtype='float32')
    a_step = torch.from_numpy(np.asarray(step_act, 'float32')).to(device).unsqueeze(0)
    for t in range(H):
        emb, dv = _emb_dv(step_obs[t])
        post, prior = rssm.obs_step(state, a_step, emb, dv=dv, sample=True)
        post_dec[t] = rssm.decode(post.feat).squeeze(0).float().cpu().numpy()[:obs_dim]
        prior_dec[t] = rssm.decode(prior.feat).squeeze(0).float().cpu().numpy()[:obs_dim]
        # KL(post || prior) over the K categorical groups (kl_dyn_raw analogue).
        p = F.softmax(post.z_logits, dim=-1)
        lp = F.log_softmax(post.z_logits, dim=-1)
        lq = F.log_softmax(prior.z_logits, dim=-1)
        kl[t] = float((p * (lp - lq)).sum(dim=-1).sum(dim=-1).mean())
        state = post
    return post_dec, prior_dec, kl


def probe(run_dir: Path, ckpt_name: str, levels=(0.0, 0.3, -0.3),
          step_frac=0.4, horizon=220, settle=220):
    device, _ = _pick_device()
    ckpt = _find_ckpt(run_dir, ckpt_name)
    model, cfg, on = _load_model(ckpt, device)
    model.eval()
    if not _is_rssm_model(model):
        print('[probe] requires an RSSM/TSSM checkpoint (needs obs_step/decode).')
        return
    from training.train import APCEnv
    env = APCEnv(cfg, np.random.default_rng(20260609))
    if on is not None and on.get('var') is not None:
        var = np.asarray(on.get('var'), 'float32')
        env.set_obs_norm_stats(mean=np.asarray(on.get('mean')), var=var,
                               count=float(on.get('count', 1.0)), learn=False)
    free_bits = float(getattr(cfg, 'rssm_free_bits', 1.0))
    cv_idx = int(env.cv_indices[0])
    L = min(int(getattr(cfg, 'lookback', 64)), settle)
    n_mv = int(env.action_dim)

    rows = []
    kl_all = []
    for lev in levels:
        for d in (+abs(step_frac), -abs(step_frac)):
            base = np.zeros(n_mv, dtype='float32'); base[0] = float(lev)
            lb_obs, lb_act, _, settled = _settle_capture(env, base, settle, L)
            stepped = base.copy(); stepped[0] = float(np.clip(lev + d, -1, 1))
            real_obs, _ = _real_step_rollout(env, stepped, horizon)
            post_dec, prior_dec, kl = _teacher_forced_post_prior(
                model, lb_obs, lb_act, real_obs, stepped, device)
            ol = _imagine_open_loop_rssm(model, lb_obs, lb_act,
                                         np.tile(stepped, (horizon, 1)),
                                         horizon, device)
            b = float(settled[cv_idx])
            g_real = _cv_gain(real_obs[:, cv_idx], b)
            if abs(g_real) < 1e-6:
                continue
            g_post = _cv_gain(post_dec[:, cv_idx], b)
            g_prior = _cv_gain(prior_dec[:, cv_idx], b)
            g_ol = _cv_gain(ol[:, cv_idx], b)
            rows.append((g_real, g_post, g_prior, g_ol))
            kl_all.append(float(kl.mean()))

    if not rows:
        print('[probe] no usable step responses (gain ~0).')
        return
    arr = np.asarray(rows)  # (N, 4): real, post, prior1, openloop
    gr, gp, gpr, gol = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]
    def _ratio(num, den):
        return float(np.mean(num / den))
    kl_mean = float(np.mean(kl_all))

    print(f'\n=== WM posterior-vs-prior lag probe: {run_dir.name} ({ckpt_name}) ===')
    print(f'  free_bits floor = {free_bits:.3f} nats/step | '
          f'observed KL(post||prior) on step-test = {kl_mean:.3f} nats/step '
          f'({"PINNED at floor" if abs(kl_mean - free_bits) < 0.08 else "above floor"})')
    print(f'  CV step-response gain ratios vs REAL (n={len(rows)} steps):')
    print(f'    posterior recon (0-step)   : {_ratio(gp, gr):.3f}   '
          f'<- encoder/decoder/latent capacity (free_bits CANNOT help if <~0.9)')
    print(f'    prior 1-step prediction    : {_ratio(gpr, gr):.3f}   '
          f'<- prior vs posterior (free_bits territory)')
    print(f'    prior open-loop (N-step)   : {_ratio(gol, gr):.3f}   '
          f'<- compounding/contraction over horizon')
    print('  Decomposition (where the gain is lost, multiplicative):')
    print(f'    real -> posterior   : x{_ratio(gp, gr):.3f}  (autoencoder)')
    print(f'    posterior -> 1-step : x{_ratio(gpr, gp):.3f}  (prior latent gap = free_bits)')
    print(f'    1-step -> open-loop : x{_ratio(gol, gpr):.3f}  (compounding)')
    # Verdict
    r_post = _ratio(gp, gr)
    if r_post < 0.9:
        print('  VERDICT: the POSTERIOR already loses the gain -> bottleneck is '
              'encoder/decoder/latent (or data). free_bits will NOT fix it.')
    elif _ratio(gpr, gp) < 0.9:
        print('  VERDICT: posterior is faithful but the PRIOR lags it -> '
              'free_bits (tighter prior->posterior KL) is the right lever.')
    else:
        print('  VERDICT: posterior + 1-step prior are faithful; the loss is in '
              'COMPOUNDING -> contraction/horizon issue, not free_bits.')


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run-dir', required=True)
    ap.add_argument('--ckpt', default='best.pt')
    ap.add_argument('--horizon', type=int, default=220)
    ap.add_argument('--settle', type=int, default=220)
    args = ap.parse_args()
    rd = Path(args.run_dir)
    if not rd.is_absolute():
        rd = REPO / rd
    sim = REPO / 'simulation' / 'test_sim'
    os.environ.setdefault('CONTROL_SETUP_JSON', str(sim / 'control_setup.json'))
    os.environ.setdefault('CONTROL_OBJECTIVE_JSON', str(sim / 'control_objective.json'))
    os.environ.setdefault('SIMULATION_DIR', str(sim))
    os.environ.setdefault('IDENTIFIED_TAU_DOMINANT', '53')
    os.environ.setdefault('IDENTIFIED_DEAD_TIME', '8')
    probe(rd, args.ckpt, horizon=args.horizon, settle=args.settle)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
