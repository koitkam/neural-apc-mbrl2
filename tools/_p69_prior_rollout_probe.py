"""P69 offline probe: why does the RSSM prior rollout not settle?

Isolates the imagination-divergence root cause by rolling the PRIOR
(``img_step``) forward under a HELD constant action and comparing:
  * sample=True  (stochastic categorical, what training+diagnostic use)
  * sample=False (deterministic / argmax-ish prior mode)

Also logs per-step prior categorical entropy, deter-state delta, and
decoded-obs tail std so we can attribute non-convergence to:
  (A) open-loop categorical sampling variance (re-injected each step), or
  (B) the deterministic prior map itself not contracting.
"""
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from tools.wm_steady_state_diagnostic import _load_model  # noqa: E402


def _prior_entropy(z_logits: torch.Tensor) -> float:
    # mean per-categorical entropy (nats) over the K groups
    p = F.softmax(z_logits, dim=-1)
    logp = F.log_softmax(z_logits, dim=-1)
    ent = -(p * logp).sum(-1)        # (..., K)
    return float(ent.mean().item())


@torch.no_grad()
def roll(rssm, device, *, sample: bool, n_warm=12, n_steps=200,
         action_dim, obs_dim, held_action_val=0.0, warm_obs_val=0.0):
    state = rssm.initial_state(1, device)
    a_held = torch.full((1, action_dim), float(held_action_val),
                        device=device)
    o_warm = torch.full((1, obs_dim), float(warm_obs_val), device=device)
    # warm-start posterior under held obs+action (settled context)
    for _ in range(n_warm):
        emb = rssm.embed(o_warm)
        post, _ = rssm.obs_step(state, a_held, emb, sample=True)
        state = post
    decoded = np.zeros((n_steps, obs_dim), dtype='float32')
    ent = np.zeros(n_steps, dtype='float32')
    h_delta = np.zeros(n_steps, dtype='float32')
    prev_h = state.h.clone()
    for k in range(n_steps):
        state = rssm.img_step(state, a_held, sample=sample)
        decoded[k] = rssm.decode(state.feat).squeeze(0).cpu().numpy()[:obs_dim]
        ent[k] = _prior_entropy(state.z_logits)
        h_delta[k] = float((state.h - prev_h).abs().max().item())
        prev_h = state.h.clone()
    return decoded, ent, h_delta


def tail_stats(traj, tail_frac=0.2):
    T = traj.shape[0]
    tail = traj[-max(2, int(T * tail_frac)):]
    return dict(tail_std_max=float(tail.std(0).max()),
                tail_std_mean=float(tail.std(0).mean()),
                tail_drift=float(np.abs(tail[tail.shape[0] // 2:].mean(0)
                                        - tail[:tail.shape[0] // 2].mean(0)).max()))


def main():
    ckpt = REPO / 'output/test_sim/run_20260530_p69_rssm_freebits0/final.pt'
    device = torch.device('cpu')
    model, cfg, _ = _load_model(ckpt, device)
    model.eval()
    rssm = model.dynamics
    A = rssm.action_dim
    D = rssm.obs_dim
    print(f'loaded {ckpt.name}  action_dim={A} obs_dim={D} '
          f'unimix={float(getattr(rssm.prior_net, "unimix", float("nan")))}')

    for held in (0.0, 0.5, -0.5):
        print(f'\n===== held_action={held} =====')
        for sample in (True, False):
            dec, ent, hd = roll(rssm, device, sample=sample,
                                 action_dim=A, obs_dim=D,
                                 held_action_val=held)
            ts = tail_stats(dec)
            print(f'  sample={str(sample):5s}  '
                  f'tail_std_max={ts["tail_std_max"]:.4f}  '
                  f'tail_drift={ts["tail_drift"]:.4f}  '
                  f'prior_ent[tail]={ent[-40:].mean():.3f}nats  '
                  f'h_delta[tail]={hd[-40:].mean():.4f}  '
                  f'converged(<0.05)={ts["tail_std_max"] < 0.05}')


if __name__ == '__main__':
    main()
