"""Single-graph deterministic ONNX export for Dreamer 4.

Reference: arXiv:2509.24527.

The exported graph implements the V4 streaming inference path:

    inputs : obs_window      (1, history_window_samples, obs_dim)
             prev_actions    (1, history_window_samples, action_dim)

    outputs: action          (1, action_dim)   in [-1, 1]

``history_window_samples`` is the training/inference context length
(unified 2026-05-24: ``lookback == seq_len`` so the exported graph uses
the same number of context positions the world-model was trained on).
The deployment runtime must supply observation frames spaced by
``sample_rate_seconds`` (the agent's control interval) — these two
fields are written to ``run_plan.json`` for unambiguous runtime
configuration.

Per-step computation:
  1. Encode every observation in the history window through the
     causal tokenizer  →  z_ctx of shape (1, history_window_samples, z_dim).
  2. Run the dynamics transformer with τ = 1 − τ_ctx, d = 1/k_max
     (clean past) over the (z_ctx, prev_actions) sequence.
  3. Read the agent-register hidden state at the latest time slot.
  4. argmax over the policy logits per action dim → bin centre.

This is the **full-recompute** inference path (no KV cache) that we
selected for ONNX-friendliness — the wrapper module does not maintain
any persistent state between calls. The deployment runtime is responsible
for sliding the (obs_window, prev_actions) buffers between calls.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.dreamer_v4 import DreamerV4


class DeterministicController(nn.Module):
    """ONNX-friendly wrapper around a trained DreamerV4 model."""

    def __init__(self, model: DreamerV4):
        super().__init__()
        self.tokenizer = model.tokenizer
        self.dynamics = model.dynamics
        self.policy = model.policy
        self.cfg = model.cfg
        self.lookback = model.cfg.lookback
        self.k_max = model.cfg.k_max
        # τ_ctx default — must land past tokens at (k_max-1)/k_max
        # which is the MAX trained τ in the sample_tau_d grid.
        # τ=0.9 (the historical default) is OOD for k_max=4 or 8.
        self.tau_ctx = 1.0 / float(model.cfg.k_max)

    def forward(self, obs_window: torch.Tensor, prev_actions: torch.Tensor
                ) -> torch.Tensor:
        if self.tokenizer is not None:
            # SF-transformer backbone: tokenize + the tau/d-conditioned dynamics.
            B = obs_window.shape[0]
            L = self.lookback
            z_ctx = self.tokenizer.encode(obs_window)         # (B, L, z_dim)
            tau = torch.full((B, L), 1.0 - self.tau_ctx,
                              device=obs_window.device, dtype=z_ctx.dtype)
            d = torch.full((B, L), 1.0 / self.k_max,
                            device=obs_window.device, dtype=z_ctx.dtype)
            out = self.dynamics(z_ctx, tau, d, prev_actions)
            agent_hid = out['agent_hid'][:, -1]               # (B, d_model)
        else:
            # RSSM / TSSM backbone (no tokenizer): roll the posterior over the
            # window DETERMINISTICALLY (sample=False ⇒ categorical argmax + the
            # continuous-latent MEAN — no RNG, ONNX-safe + cont-latent-safe) and
            # read the last feature.  Backbone-agnostic (both expose
            # rollout_observed → (feats, ...)).  Fixes the ONNX export for the
            # RSSM/TSSM production backbones (the wrapper was SF-only).
            feats = self.dynamics.rollout_observed(
                obs_window, prev_actions, sample=False,
                store_aux=False)[0]                           # (B, L, F)
            agent_hid = feats[:, -1]                           # (B, F)
        # Deterministic action — works for both PolicyHead (argmax bin)
        # and ContinuousPolicyHead (tanh(mu)).
        action, _, _ = self.policy(agent_hid, deterministic=True)
        return action


def export_dreamer_v4_onnx(model: DreamerV4, out_path: str | Path,
                            opset: int = 18) -> str:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    model = model.eval()
    wrapper = DeterministicController(model).eval()

    cfg = model.cfg
    obs_window = torch.zeros(1, cfg.lookback, cfg.obs_dim)
    prev_actions = torch.zeros(1, cfg.lookback, cfg.action_dim)

    torch.onnx.export(
        wrapper,
        (obs_window, prev_actions),
        str(out_path),
        input_names=['obs_window', 'prev_actions'],
        output_names=['action'],
        opset_version=opset,
        do_constant_folding=True,
        dynamic_axes=None,           # fixed batch=1, fixed lookback
        dynamo=False,                # legacy TorchScript exporter: traces the
                                     # RSSM/TSSM rollout loop + needs no
                                     # ``onnxscript`` (torch>=2.9 defaults the
                                     # dynamo exporter, which 500s here on the
                                     # data-dependent latent control flow).
    )
    return str(out_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import argparse, json
    from dataclasses import fields
    from training.train import TrainConfig
    from models.dreamer_v4 import dreamer_v4_config_from_train

    p = argparse.ArgumentParser()
    p.add_argument('ckpt', help='final.pt or ckpt_iter_*.pt produced by '
                                'training/train.py')
    p.add_argument('--out', default=None,
                   help='output ONNX path (default: <ckpt_dir>/dreamer_v4.onnx)')
    args = p.parse_args()

    state = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    cfg_dict = state['cfg']
    cfg = TrainConfig(**{k: v for k, v in cfg_dict.items()
                          if k in {f.name for f in fields(TrainConfig)}})
    model = DreamerV4(
        dreamer_v4_config_from_train(cfg, attn_impl='manual'))
    model.load_state_dict(state['model'])

    out = args.out or str(Path(args.ckpt).with_name('dreamer_v4.onnx'))
    export_dreamer_v4_onnx(model, out)
    print(json.dumps({'onnx': out, 'inputs': {
        'obs_window':   [1, cfg.lookback, cfg.obs_dim],
        'prev_actions': [1, cfg.lookback, cfg.action_dim],
    }, 'history_window_samples': int(cfg.lookback),
        'sample_rate_seconds': int(getattr(cfg, 'sample_rate', 1)),
        'outputs': ['action']}, indent=2))
