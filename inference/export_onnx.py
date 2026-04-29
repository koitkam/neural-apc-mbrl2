"""Single-graph deterministic ONNX export for DreamerV4.

The exported graph takes the previous controller state ``(prev_h, prev_z,
prev_action)`` plus the current observation window and returns the next
state ``(next_h, next_z)`` together with the deterministic action:

    inputs : prev_h            (1, deter_dim)
             prev_z            (1, n_categoricals * n_classes)
             prev_action       (1, action_dim)
             obs_window        (1, lookback, obs_dim)

    outputs: next_h            (1, deter_dim)
             next_z            (1, n_categoricals * n_classes)
             action            (1, action_dim)        in [-1, 1]

Deterministic mode used at inference:
- ``next_z`` = one-hot of argmax over the posterior logits (per categorical).
- ``action``  = bin centre at argmax of the actor logits (per action dim).

This single integrated graph is the only artifact we ship — there is no
separate observer or actor checkpoint at deployment time.  The lookback,
state dim and action dim are baked in at export.
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
        self.rssm = model.rssm
        self.actor = model.actor
        self.n_categoricals = model.cfg.rssm.n_categoricals
        self.n_classes = model.cfg.rssm.n_classes
        self.action_dim = model.cfg.rssm.action_dim
        self.n_action_bins = model.cfg.n_action_bins
        # Register actor bin centres as buffer (already on Actor) — accessed
        # through ``self.actor.bin_centres``.

    def forward(self, prev_h: torch.Tensor, prev_z: torch.Tensor,
                prev_action: torch.Tensor, obs_window: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Encoder -> embedding
        B = obs_window.shape[0]
        flat = obs_window.reshape(B, -1)
        e = self.rssm.encoder(flat)
        # GRU advance
        gru_in = torch.cat([prev_z, prev_action], dim=-1)
        h = self.rssm.gru(gru_in, prev_h)
        # Posterior logits q(z | h, e); take argmax per categorical (deterministic)
        post_logits = self.rssm.posterior_head(torch.cat([h, e], dim=-1))
        post_logits = post_logits.view(B, self.n_categoricals, self.n_classes)
        idx = post_logits.argmax(dim=-1)
        z_onehot = F.one_hot(idx, num_classes=self.n_classes).to(post_logits.dtype)
        z = z_onehot.view(B, self.n_categoricals * self.n_classes)
        # Actor — argmax bin per action dim
        latent = torch.cat([h, z], dim=-1)
        action_logits = self.actor.head(latent).view(B, self.action_dim, self.n_action_bins)
        a_idx = action_logits.argmax(dim=-1)
        action = self.actor.bin_centres[a_idx]      # (B, action_dim) in [-1, 1]
        return h, z, action


def export_dreamer_v4_onnx(model: DreamerV4, out_path: str | Path,
                           opset: int = 18) -> str:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    model = model.eval()
    wrapper = DeterministicController(model).eval()

    cfg = model.cfg.rssm
    deter = cfg.deter_dim
    stoch = cfg.n_categoricals * cfg.n_classes
    action_dim = cfg.action_dim
    lookback = cfg.lookback
    obs_dim = cfg.obs_dim

    prev_h = torch.zeros(1, deter)
    prev_z = torch.zeros(1, stoch)
    prev_action = torch.zeros(1, action_dim)
    obs_window = torch.zeros(1, lookback, obs_dim)

    torch.onnx.export(
        wrapper,
        (prev_h, prev_z, prev_action, obs_window),
        str(out_path),
        input_names=['prev_h', 'prev_z', 'prev_action', 'obs_window'],
        output_names=['next_h', 'next_z', 'action'],
        opset_version=opset,
        do_constant_folding=True,
        dynamic_axes=None,            # fixed batch=1 per the spec
    )
    return str(out_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import argparse, json
    from dataclasses import fields
    from training.train import TrainConfig
    from models.dreamer_v4 import DreamerV4Config, RSSMConfig

    p = argparse.ArgumentParser()
    p.add_argument('ckpt', help='path to a final.pt or ckpt_iter_*.pt produced by training/train.py')
    p.add_argument('--out', default=None, help='output ONNX path (default: <ckpt_dir>/dreamer_v4.onnx)')
    args = p.parse_args()

    state = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    cfg_dict = state['cfg']
    cfg = TrainConfig(**{k: v for k, v in cfg_dict.items()
                         if k in {f.name for f in fields(TrainConfig)}})
    rssm_cfg = RSSMConfig(
        obs_dim=cfg.obs_dim, action_dim=cfg.action_dim, lookback=cfg.lookback,
        deter_dim=cfg.deter_dim, embed_dim=cfg.embed_dim, hidden_dim=cfg.hidden_dim,
        n_categoricals=cfg.n_categoricals, n_classes=cfg.n_classes,
        free_nats=cfg.free_nats,
    )
    model_cfg = DreamerV4Config(rssm=rssm_cfg, n_action_bins=cfg.n_action_bins,
                                actor_hidden=cfg.hidden_dim,
                                critic_hidden=cfg.hidden_dim)
    model = DreamerV4(model_cfg)
    model.load_state_dict(state['model'])

    out = args.out or str(Path(args.ckpt).with_name('dreamer_v4.onnx'))
    export_dreamer_v4_onnx(model, out)
    print(json.dumps({'onnx': out, 'inputs': {
        'prev_h': [1, cfg.deter_dim],
        'prev_z': [1, cfg.n_categoricals * cfg.n_classes],
        'prev_action': [1, cfg.action_dim],
        'obs_window': [1, cfg.lookback, cfg.obs_dim],
    }, 'outputs': ['next_h', 'next_z', 'action']}, indent=2))
