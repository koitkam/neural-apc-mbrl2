"""Smoke test for the P89 wm_best-score convergence fix + steady consolidation.

Verifies, WITHOUT a real env/sim (a minimal fake env + the real RSSM model):
  * ``_probe_wm_held_convergence`` runs for the RSSM backbone and returns a
    valid ``wm_converge_frac`` in [0,1] + finite ``tail_drift_mean``.
  * it is a graceful no-op (``None``) for the SF backbone.
  * the convergence-aware wm_best score lets a CONVERGED ckpt beat an equally-
    correlated DRIFTING one, while a degenerate flat WM (best_h==0) gets NO
    convergence credit (guard).
  * consolidation: the 1-step ``_rssm_steady_consistency`` FIRES when the
    multi-step held-rollout is OFF and is SKIPPED when it is ON (no double-up),
    for the RSSM backbone.

Run (CPU; does not disturb a live GPU run):
  CUDA_VISIBLE_DEVICES="" PYTHONPATH=/home/koitkam/neural-apc-dreamerV4 \
  DREAMER_COMPILE=0 \
  /home/koitkam/neural-apc-dreamerV4-env/bin/python tools/_smoke_wm_best_score.py
"""
import numpy as np
import torch

from tools._smoke_overshoot_critic import _mk
from training.train import _probe_wm_held_convergence, world_model_loss


class _FakeEnv:
    """Minimal env exposing only what the probes touch."""

    def __init__(self, cfg, obs_dim, action_dim):
        self.cfg = cfg
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self._rng = np.random.default_rng(0)

    def reset(self, exploration=False):
        return self._rng.normal(size=(2, self.obs_dim)).astype('float32')

    def step(self, a):
        ow = (self._rng.normal(size=(2, self.obs_dim)) * 0.5).astype('float32')
        return ow, 0.0, False, {}


def _score(r_vals, best_h, H, conv_frac, w=1.0):
    """Replicates the wm_best score formula (train.py ~L6693)."""
    s = sum(max(0.0, r) for r in r_vals) + 0.5 * best_h / max(1, H)
    if w > 0 and conv_frac is not None and best_h > 0:
        s += w * conv_frac
    return s


def main():
    dev = torch.device('cpu')
    cfg, model, _ = _mk(wm_type='rssm')
    cfg.horizon = 12          # conv probe needs H >= 8
    cfg.lookback = 8
    env = _FakeEnv(cfg, cfg.obs_dim, cfg.action_dim)

    # ---- convergence probe: valid fields (RSSM) ----
    conv = _probe_wm_held_convergence(model, env, dev, cfg)
    assert conv is not None, 'conv probe returned None for RSSM'
    assert 0.0 <= conv['wm_converge_frac'] <= 1.0, conv
    assert np.isfinite(conv['tail_drift_mean']), conv
    print(f"[smoke] OK conv probe: frac={conv['wm_converge_frac']:.2f} "
          f"drift={conv['tail_drift_mean']:.3f} n={conv['n_starts']}")

    # ---- SF backbone -> None (graceful no-op) ----
    cfg_sf, model_sf, _ = _mk(wm_type='sf_transformer')
    cfg_sf.horizon = 12
    cfg_sf.lookback = 8
    assert _probe_wm_held_convergence(model_sf, env, dev, cfg_sf) is None
    print("[smoke] OK conv probe SF -> None (graceful)")

    # ---- score: converged beats equally-correlated drifting; flat guard ----
    s_drift = _score([0.6, 0.5, 0.4, 0.3], best_h=12, H=12, conv_frac=0.0)
    s_conv = _score([0.6, 0.5, 0.4, 0.3], best_h=12, H=12, conv_frac=1.0)
    assert s_conv > s_drift, (s_conv, s_drift)
    s_flat = _score([0.0, 0.0, 0.0, 0.0], best_h=0, H=12, conv_frac=1.0)
    assert s_flat == 0.0, s_flat
    print(f"[smoke] OK score: drift={s_drift:.2f} < conv={s_conv:.2f}; "
          f"flat-guard={s_flat:.2f}")

    # ---- consolidation gate (RSSM) ----
    B, T = 2, 16
    obs = torch.zeros(B, T, cfg.obs_dim) + torch.randn(B, 1, cfg.obs_dim) * 0.05
    act = torch.zeros(B, T, cfg.action_dim) + 0.3
    b2 = {'obs': obs, 'act': act, 'rew': torch.randn(B, T),
          'cont': torch.ones(B, T), 'expert': torch.zeros(B, T)}
    cfg.wm_steady_consistency_coef = 0.5
    cfg.wm_held_rollout_coef = 0.0
    f0 = float(world_model_loss(model, b2, cfg)[0]['wm_steady_held_frac'])
    cfg.wm_held_rollout_coef = 0.5
    f1 = float(world_model_loss(model, b2, cfg)[0]['wm_steady_held_frac'])
    assert f0 > 0.0, f'1-step steady should FIRE when held-rollout off ({f0})'
    assert f1 == 0.0, f'1-step steady should be SKIPPED when held-rollout on ({f1})'
    print(f"[smoke] OK consolidation: held_off frac={f0:.2f} (ran) -> "
          f"held_on frac={f1:.2f} (skipped)")

    print("\n[smoke] ALL P89 wm_best-score + consolidation checks PASSED")


if __name__ == '__main__':
    main()
