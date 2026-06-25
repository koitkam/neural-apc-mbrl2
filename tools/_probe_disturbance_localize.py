"""Decisive probe: WHERE is the unmeasured-disturbance information?

Rolls one forced-disturbance episode with the TRAINED deterministic policy
(closed-loop, controlled CV — the deployment condition) and taps, per step:
  - the disturbance HEAD output (reproduces the validation det_r),
  - the cont DISTURBANCE channel c_dist (the structural target),
  - a FRESH in-sample OLS probe on the full posterior latent [h,z,c]
    (overfit UPPER BOUND: does the latent encode the disturbance AT ALL?),
  - the controlled CV observation (is the disturbance visible in the CV?),
  - the MV action (does the controller's rejection action reflect it?),
  - the 1-step PRIOR CV innovation = CV_obs - decode(prior).CV
    (the DOB residual: the disturbance SHOULD live here).
Scores each vs the true hidden disturbance, raw + detrended (high-pass).

Verdict logic:
  innovation high + c_dist/head low  => info is in the residual, the latent
    does NOT extract it -> STRUCTURAL (supervise c_dist / re-enable DOB).
  full-latent probe high + head low  => head undertrained (less likely).
  everything low                     => deep observability problem.
"""
from __future__ import annotations
import os, sys, json
import numpy as np
import torch

RUN = sys.argv[1] if len(sys.argv) > 1 else \
    'output/test_sim/run_20260623_p138_rscalecap'
CKPT = sys.argv[2] if len(sys.argv) > 2 else 'best.pt'

sys.path.insert(0, os.getcwd())
from tools.wm_steady_state_diagnostic import _load_model
from training.train import TrainConfig, APCEnv

device = torch.device('cpu')
model, cfg, obs_norm = _load_model(os.path.join(RUN, CKPT), device)
model.eval()
rssm = model.dynamics

deter = int(getattr(rssm, 'deter_dim', 0))
stoch = int(getattr(rssm, 'stoch_flat_dim', 0))
cont = int(getattr(rssm, 'cont_dim', 0) or 0)
cont_gain = int(getattr(rssm, 'cont_gain_dim', 0) or 0)
cont_dist = int(getattr(rssm, 'cont_dist_dim', 0) or 0)
dv_feed = int(getattr(rssm, '_dv_feed_dim', 0) or 0)
print(f'[layout] deter={deter} stoch={stoch} cont={cont} '
      f'(gain={cont_gain} dist={cont_dist}) dv_feed={dv_feed}')
core = deter + stoch + cont
cdist_lo = deter + stoch + cont_gain
cdist_hi = deter + stoch + cont_gain + cont_dist

import numpy as _np
rng = _np.random.default_rng(7)
env = APCEnv(cfg, rng)
if obs_norm is not None:
    try:
        env.set_obs_norm_stats(mean=_np.asarray(obs_norm.get('mean')),
                               var=_np.asarray(obs_norm.get('var')),
                               count=float(obs_norm.get('count', 1.0)))
    except Exception as e:
        print('obs_norm restore skipped', e)

os.environ['DREAMER_HIDDEN_DIST_SPREAD'] = '1'
env._hidden_disturbance_force = True
env._current_phase = 3
env._training_progress = 1.0
env._disturbance_prob_override = 1.0

T = int(cfg.episode_length)
n_cv = len(env.cv_indices)
cv_idx = list(env.cv_indices)
mv_idx = list(getattr(env, 'mv_indices', []) or [])
head = getattr(model, 'disturbance', None)

obs_window = env.reset(exploration=False)
state = rssm.initial_state(1, device)
prev_a = torch.zeros(1, int(env.action_dim), device=device)

feats, cdists, heads, trues, cvs, mvs, innov = [], [], [], [], [], [], []
with torch.no_grad():
    for t in range(T):
        o = torch.from_numpy(obs_window[-1]).to(device).unsqueeze(0).float()
        emb = rssm.embed(o)
        # prior (img_step) BEFORE seeing obs -> for the DOB-style innovation
        cv_prior = None
        try:
            prior = rssm.img_step(state, prev_a, sample=False)
            dec = rssm.decode(prior.feat)
            cvit = getattr(rssm, 'cv_index_t', None)
            if cvit is not None:
                cv_prior = dec.index_select(-1, cvit).float().squeeze(0).cpu().numpy()
        except Exception as _e:
            cv_prior = None
        post, _ = rssm.obs_step(state, prev_a, emb, sample=True, obs=o)
        feat = post.feat
        fh = feat.clone()
        if dv_feed > 0 and feat.shape[-1] >= core + dv_feed:
            fh[..., core:core + dv_feed] = 0.0
        hd_out = head(fh).float().squeeze(0).cpu().numpy() if head is not None \
            else _np.zeros(n_cv)
        fnp = feat.float().squeeze(0).cpu().numpy()
        feats.append(fnp)
        cdists.append(fnp[cdist_lo:cdist_hi] if cont_dist > 0 else _np.zeros(1))
        heads.append(hd_out[:n_cv])
        ow = obs_window[-1]
        cvs.append(_np.asarray([ow[ci] for ci in cv_idx], dtype='float32'))
        mvs.append(_np.asarray([ow[mi] for mi in mv_idx], dtype='float32') if mv_idx else _np.zeros(1))
        # innovation = normalized CV obs - prior-decoded CV (DOB residual)
        cv_obs_norm = _np.asarray([ow[ci] for ci in cv_idx], dtype='float32')
        if cv_prior is not None:
            innov.append(cv_obs_norm[:len(cv_prior)] - cv_prior[:len(cv_obs_norm)])
        else:
            innov.append(_np.zeros(n_cv))
        action_t, _, _ = model.policy(feat, deterministic=True)
        a_np = action_t.float().squeeze(0).cpu().numpy().astype('float32')
        prev_a = torch.from_numpy(a_np).to(device).unsqueeze(0)
        state = post
        obs_window, _r, done, info = env.step(a_np)
        hd = info.get('hidden_disturbance')
        trues.append(_np.asarray(hd, dtype='float32').reshape(-1)[:n_cv]
                     if hd is not None else _np.zeros(n_cv))
        if done:
            break

feats = _np.asarray(feats); cdists = _np.asarray(cdists)
heads = _np.asarray(heads); trues = _np.asarray(trues)
cvs = _np.asarray(cvs); mvs = _np.asarray(mvs); innov = _np.asarray(innov)
n = len(trues)
w0 = max(int(cfg.lookback), int(0.1 * n))
sl = slice(w0, n)


def ma(x, w):
    if w <= 1 or w >= len(x):
        return _np.full_like(x, float(x.mean()))
    pad = w // 2
    xp = _np.pad(x, pad, mode='edge')
    k = _np.ones(w) / w
    return _np.convolve(xp, k, mode='same')[pad:pad + len(x)]


def hp(x, w):
    return x - ma(x, w)


def corr(a, b):
    a = _np.asarray(a, float); b = _np.asarray(b, float)
    if a.std() < 1e-9 or b.std() < 1e-9:
        return float('nan')
    return float(_np.corrcoef(a, b)[0, 1])


dw = 4 * int(getattr(cfg, 'horizon', 55))
true0 = trues[sl, 0]
print(f'\n[probe] n={n} scored={n-w0} detrend_window={dw} '
      f'true_std={true0.std():.3f} true_mean={true0.mean():.3f}')


def score(name, sig):
    sig = _np.asarray(sig, float)
    raw = corr(sig, true0)
    det = corr(hp(sig, dw), hp(true0, dw))
    print(f'  {name:34s} raw_r={raw:+.3f}  det_r={det:+.3f}  std={sig.std():.3f}')
    return det


print('--- single-signal correlations vs true disturbance (CV0) ---')
score('HEAD output (val estimator)', heads[sl, 0])
if cont_dist > 0:
    for j in range(cont_dist):
        score(f'c_dist[{j}] (cont disturbance chan)', cdists[sl, j])
score('CV obs (controlled)', cvs[sl, 0])
if mvs.shape[1] > 0:
    score('MV obs (rejection action)', mvs[sl, 0])
score('prior-CV innovation (DOB residual)', innov[sl, 0])

# HELD-OUT ridge probe on the FULL posterior latent [h,z,c] -> true.
# Train on the first 60%, test on the last 40% (honest: overfit will NOT
# generalize if the latent has no real disturbance signal).
X = feats[sl]; y = true0
ntr = int(0.6 * len(y))
Xtr, Xte = X[:ntr], X[ntr:]; ytr, yte = y[:ntr], y[ntr:]
mu = Xtr.mean(0); ym = ytr.mean()
Xtrc = Xtr - mu; Xtec = Xte - mu; ytrc = ytr - ym
lam = 1e1 * (Xtrc.T @ Xtrc).diagonal().mean() + 1e-6
w = _np.linalg.solve(Xtrc.T @ Xtrc + lam * _np.eye(Xtrc.shape[1]), Xtrc.T @ ytrc)
yhat_te = Xtec @ w + ym
print('--- HELD-OUT ridge probe (train 60% / test 40%, honest) ---')
print(f'  full latent [h,z,c] -> true (TEST): raw_r={corr(yhat_te,yte):+.3f} '
      f'det_r={corr(hp(yhat_te,dw),hp(yte,dw)):+.3f}')
yhat_in = Xtrc @ w + ym
print(f'  full latent [h,z,c] -> true (TRAIN): raw_r={corr(yhat_in,ytr):+.3f}')

# Held-out probe on c_dist only (same split)
if cont_dist > 0:
    C = cdists[sl]
    Ctr, Cte = C[:ntr], C[ntr:]
    cmu = Ctr.mean(0)
    Ctrc = Ctr - cmu; Ctec = Cte - cmu
    lam2 = 1e-2 * (Ctrc.T @ Ctrc).diagonal().mean() + 1e-9
    w2 = _np.linalg.solve(Ctrc.T @ Ctrc + lam2 * _np.eye(Ctrc.shape[1]), Ctrc.T @ ytrc)
    yh2 = Ctec @ w2 + ym
    print(f'  c_dist block      -> true (TEST): raw_r={corr(yh2,yte):+.3f} '
          f'det_r={corr(hp(yh2,dw),hp(yte,dw)):+.3f}')
