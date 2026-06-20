"""Decisive probe: is the WM open-loop MV->CV gain contraction from categorical
SAMPLING (Jensen/EIV over the free-running rollout) or the learned prior?

Rolls the prior to steady-state under an MV step with sample=True vs sample=False
(expected latent = categorical probabilities) and reports the steady-state CV
response of each.  If sample=False >> sample=True (closer to the real gain), the
contraction is sampling-driven; if they match, it is the learned prior / weak
supervision.
"""
import sys, json
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[0] if (Path(__file__).name != '<stdin>') else '.'))
RUN = sys.argv[1]
from tools.wm_steady_state_diagnostic import _load_model, _quiet_env
from evaluation.wm_transfer_matrix import _settle_capture
from training.train import APCEnv

device = torch.device('cpu')
model, cfg, obs_norm = _load_model(Path(RUN) / 'final.pt', device)
model.eval()
rssm = model.dynamics
L = int(cfg.lookback)
cv_idx = int(getattr(cfg, 'cv_indices', [0])[0]) if getattr(cfg, 'cv_indices', None) else 0
# action MV step size in normalized action space
delta = 0.5

rng = np.random.default_rng(0)
env = APCEnv(cfg, rng)
_quiet_env(env)
if obs_norm is not None:
    env.set_obs_norm_stats(mean=np.asarray(obs_norm['mean']), var=np.asarray(obs_norm['var']),
                           count=float(obs_norm.get('count', 1.0)), learn=False)

# settle at mid-action, capture lookback
base_action = np.zeros(env.action_dim, dtype='float32')
lb_obs, lb_act, base_ctrl, settled = _settle_capture(env, base_action, settle_steps=200, L=L)
print(f"settled CV={settled[cv_idx]:.4f}  L={lb_obs.shape[0]} cv_idx={cv_idx} dv_dim={getattr(rssm,'dv_dim',0)}")

K = 220
_dv_on = int(getattr(rssm, 'dv_dim', 0) or 0) > 0

@torch.no_grad()
def rollout(action_vec, sample):
    state = rssm.initial_state(1, device)
    for l in range(L):
        o = torch.from_numpy(lb_obs[l]).to(device).unsqueeze(0)
        a = torch.from_numpy(lb_act[l]).to(device).unsqueeze(0)
        emb = rssm.embed(o)
        dv = o.index_select(-1, rssm.dv_index_t) if _dv_on else None
        post, _ = rssm.obs_step(state, a, emb, dv=dv, sample=sample)
        state = post
    dv_hold = (torch.from_numpy(lb_obs[L-1]).to(device).unsqueeze(0).index_select(-1, rssm.dv_index_t)
               if _dv_on else None)
    a_t = torch.from_numpy(action_vec).to(device).unsqueeze(0)
    cv = np.zeros(K, dtype='float32')
    for kk in range(K):
        state = rssm.img_step(state, a_t, dv=dv_hold, sample=sample)
        obs_hat = rssm.decode(state.feat).squeeze(0).float().cpu().numpy()
        cv[kk] = obs_hat[cv_idx]
    return cv

step_action = base_action.copy(); step_action[0] += delta
for sample in (True, False):
    base_cv = rollout(base_action, sample)
    step_cv = rollout(step_action, sample)
    resp = (step_cv[-20:].mean() - base_cv[-20:].mean())
    print(f"sample={sample!s:5}  ss CV response to +{delta} MV = {resp:+.4f}  (per-unit gain={resp/delta:+.4f})")

# real env response for calibration
def real_resp():
    env2 = APCEnv(cfg, np.random.default_rng(1)); _quiet_env(env2)
    if obs_norm is not None:
        env2.set_obs_norm_stats(mean=np.asarray(obs_norm['mean']), var=np.asarray(obs_norm['var']),
                                count=float(obs_norm.get('count',1.0)), learn=False)
    _settle_capture(env2, base_action, 200, L)
    b = np.array([env2.step(base_action)[0][-1][cv_idx] for _ in range(K)])
    env3 = APCEnv(cfg, np.random.default_rng(1)); _quiet_env(env3)
    if obs_norm is not None:
        env3.set_obs_norm_stats(mean=np.asarray(obs_norm['mean']), var=np.asarray(obs_norm['var']),
                                count=float(obs_norm.get('count',1.0)), learn=False)
    _settle_capture(env3, base_action, 200, L)
    s = np.array([env3.step(step_action)[0][-1][cv_idx] for _ in range(K)])
    return s[-20:].mean() - b[-20:].mean()
rr = real_resp()
print(f"REAL    ss CV response to +{delta} MV = {rr:+.4f}  (per-unit gain={rr/delta:+.4f})")
