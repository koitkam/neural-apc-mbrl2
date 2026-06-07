"""Transformer state-space world-model backbone (TSSM) — DESIGN SCAFFOLD (WIP).

neural-apc-mbrl, 2026-06-06.  This is the "RSSM-training-structure + transformer
dynamics-core" backbone the user asked for: keep the *entire* proven phased/joint
training pipeline (clean steady-state seeds, noise + DR curriculum, realistic
hidden disturbances, overshoot + held-rollout losses, critic warmup, joint mode)
and swap ONLY the recurrent dynamics core (GRU + 32x32 categorical RSSM) for a
**causal transformer sequence model** that performs IN-CONTEXT SYSTEM
IDENTIFICATION over the lookback window.

This is deliberately a SCAFFOLD: the shared, low-risk pieces (config, state,
encoder/decoder, initial_state) are implemented; the three transition methods
(``rollout_observed`` / ``obs_step`` / ``img_step``) carry the full design spec
and raise ``NotImplementedError`` so selecting this backbone fails LOUDLY rather
than misbehaving silently.  It is NOT yet wired into ``build_model`` dispatch, so
the default RSSM path is untouched.  See "Wiring plan" + "Open design decisions"
below.

WHY a transformer core (vs the current SF/flow transformer):
  The existing ``world_model_type='sf_transformer'`` is a shortcut-forcing/flow
  model with its OWN training machinery (MAE tokenizer, shortcut-forcing loss) —
  it does NOT reuse the RSSM training structure, and it has NO recurrent fixed
  point (steady-state convergence 0% by construction).  This TSSM instead
  implements the SAME interface as ``RSSMDynamics`` so it is a drop-in core for
  the RSSM pipeline.  Motivation (P90 RCA): under narrow domain randomization the
  RSSM's single fixed recurrent state averages domains into a FUZZY fixed point
  (gain 0.354, 3x too small).  A transformer attends over the full lookback and
  can INFER the plant's gain/tau/dead-time from the recent (obs, action) history,
  conditioning its prediction on the identified domain -> a SHARP per-domain
  fixed point.  This is the principled route to TRUE wide-DR generalization that
  a fixed-state RSSM cannot reach (the agent then trains in imagination that
  contains the right per-domain dynamics).

INTERFACE CONTRACT (must match models.dreamer_v4_rssm.RSSMDynamics exactly so the
existing dispatch in train.py / world_model_loss / _imagination_step_rssm / the
overshoot + held-rollout losses / the WM probes all work unchanged):
  attributes : deter_dim, n_categoricals, n_classes, obs_dim, prior_net,
               post_net, pre_gru-equivalent, encoder, decoder
  state      : object with .h (..., deter_dim), .z_logits (..., K, C),
               .z (..., K, C one-hot ST), .feat ([h, z_flat]), .stoch_flat
  methods    : embed(obs)->(...,embed_dim); initial_state(B,device)->State;
               img_step(prev, prev_action, sample)->State;
               obs_step(prev, prev_action, embed, sample)->(post, prior);
               rollout_observed(obs, act, sample)->(feats, post_logits,
                                                     prior_logits, last_state);
               decode(feat)->obs
  conventions: act[:, t] drives the transition INTO obs[:, t] (contemporaneous-
               action: feat[t] has seen a_t).  ``feat = [h, z_flat]`` with
               feat_dim = deter_dim + n_categoricals*n_classes so the V4
               reward/value/policy heads (built on feat_dim) are reused unchanged.

CORE ARCHITECTURE (the transition methods to implement):
  Token at step t = proj([ z_{t-1}_flat ; a_t ]) + (optional obs-embed for the
  posterior) + positional/time encoding.  A causal Transformer encoder over the
  running token sequence produces a per-step hidden ``g_t``.  Define the
  RSSM-compatible deterministic state ``h_t := g_t`` (so deter_dim = d_model).
  Prior head: ``prior_net(h_t) -> (K, C) categorical logits`` (reuse the RSSM
  ``_CategoricalLatent`` head verbatim).  Posterior head: ``post_net([h_t,
  embed_t]) -> (K, C)``.  ``z_t`` = straight-through one-hot sample (sample=True
  for training prior grad — same ST requirement the overshoot/held-rollout losses
  rely on).  decode(feat) reconstructs obs.

  img_step (imagination, the PERF-CRITICAL path): advance ONE step under a held/
  given action with NO obs.  Naive = re-run the transformer over the whole token
  history each step (O(T^2) over H imagination steps).  CORRECT + FAST = maintain
  a KV-CACHE in the State object: each img_step appends one token, attends it
  against cached keys/values -> O(T) per step.  *** This KV-cache is the main
  reason this is a scaffold, not a finished impl — it must be implemented +
  numerically validated (img_step result == teacher-forced rollout on the same
  actions) before any training run. ***

WIRING PLAN (when implemented):
  1. models/dreamer_v4.py DreamerV4.__init__: add ``elif world_model_type ==
     'tssm': self.dynamics = TransformerSSMDynamics(tssm_cfg)`` alongside the
     RSSM branch; ensure parameters_world() includes it (it already globs
     self.dynamics.parameters()).
  2. training/train.py world_model_loss / imagination_step dispatch: the RSSM
     branch checks ``world_model_type == 'rssm'``; widen to
     ``in ('rssm', 'tssm')`` since the interface is identical (feat=[h,z_flat],
     rollout_observed/img_step/decode).  Verify _wm_latent_overshoot_loss and
     _wm_held_rollout_stationarity_loss (which read rssm.deter_dim /
     n_categoricals / img_step) work unchanged — they will, by contract.
  3. ENV_OVERRIDES: DREAMER_WORLD_MODEL_TYPE already exists; add TSSM dims
     (DREAMER_TSSM_{D_MODEL,N_LAYERS,N_HEADS}).
  4. compile + the inference/export_onnx path (RSSM ONNX is not implemented;
     TSSM ONNX is a separate task).

OPEN DESIGN DECISIONS (resolve before implementing):
  - deter_dim == d_model couples the feature size to the transformer width; the
    V4 heads are built on feat_dim = deter_dim + K*C, so picking d_model sets the
    head input size (fine, but note it for model-size BO).
  - KL free-bits: reuse rssm_kl_loss verbatim (post vs prior logits) — no change.
  - Positional encoding over the lookback: absolute vs rotary; rotary preferred
    for length generalization (imagination H may exceed training seq_len).
  - Imagination determinism: RSSM uses sample=False (mode) under
    DREAMER_RSSM_IMAG_LATENT_MODE; keep the same flag semantics.
  - Numerical-equivalence test (MUST pass): img_step rolled K steps under the
    SAME actions as a teacher-forced rollout must match within tol (proves the
    KV-cache path == the full-attention path).

Until the transition methods are implemented, ``build_model`` must NOT dispatch
to this class; selecting world_model_type='tssm' should raise the clear error
below.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn

# Reuse the proven RSSM building blocks so the categorical latent + KL are
# bit-for-bit identical to the default backbone (only the dynamics core changes).
from models.dreamer_v4_rssm import _CategoricalLatent, RSSMState


@dataclass
class TransformerSSMConfig:
    """Config for the transformer dynamics core.  ``deter_dim`` == d_model."""
    obs_dim: int
    action_dim: int
    deter_dim: int = 512          # = transformer d_model (h_t := g_t)
    n_categoricals: int = 32      # match RSSM paper default
    n_classes: int = 32
    embed_dim: int = 256
    n_layers: int = 4
    n_heads: int = 8
    ffn_mult: int = 4
    dropout: float = 0.0
    unimix: float = 0.01          # match RSSM categorical mixing
    max_seq_len: int = 256        # >= lookback + imagination horizon


class TransformerSSMDynamics(nn.Module):
    """Causal-transformer dynamics core implementing the RSSMDynamics interface.

    SCAFFOLD: shared pieces are real; the three transition methods raise
    NotImplementedError with the design spec in the module docstring.
    """

    def __init__(self, cfg: TransformerSSMConfig):
        super().__init__()
        self.cfg = cfg
        self.obs_dim = int(cfg.obs_dim)
        self.action_dim = int(cfg.action_dim)
        self.deter_dim = int(cfg.deter_dim)
        self.n_categoricals = int(cfg.n_categoricals)
        self.n_classes = int(cfg.n_classes)
        self.stoch_flat_dim = self.n_categoricals * self.n_classes
        self.feat_dim = self.deter_dim + self.stoch_flat_dim

        # ----- shared, low-risk pieces (real implementations) -----
        self.encoder = nn.Sequential(
            nn.Linear(self.obs_dim, cfg.embed_dim), nn.SiLU(),
            nn.Linear(cfg.embed_dim, cfg.embed_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.feat_dim, cfg.embed_dim), nn.SiLU(),
            nn.Linear(cfg.embed_dim, self.obs_dim),
        )
        # Token projection: [z_{t-1}_flat ; a_t] -> d_model.
        self.token_proj = nn.Linear(self.stoch_flat_dim + self.action_dim,
                                    self.deter_dim)
        # Causal transformer encoder (the dynamics core).
        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.deter_dim, nhead=cfg.n_heads,
            dim_feedforward=self.deter_dim * cfg.ffn_mult,
            dropout=cfg.dropout, batch_first=True, activation='gelu',
            norm_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer,
                                                 num_layers=cfg.n_layers)
        # Categorical latent heads (reuse RSSM block: prior from h, post from
        # [h, embed]).  prior_net is read by the smoke tests + the overshoot /
        # held-rollout losses, so the attribute name MUST match the RSSM.
        self.prior_net = _CategoricalLatent(
            self.deter_dim, self.n_categoricals, self.n_classes,
            unimix=cfg.unimix)
        self.post_net = _CategoricalLatent(
            self.deter_dim + cfg.embed_dim, self.n_categoricals,
            self.n_classes, unimix=cfg.unimix)

    # ----- shared pieces (real) -----
    def embed(self, obs: torch.Tensor) -> torch.Tensor:
        return self.encoder(obs)

    def decode(self, feat: torch.Tensor) -> torch.Tensor:
        return self.decoder(feat)

    def initial_state(self, batch_size: int,
                      device: torch.device) -> RSSMState:
        h = torch.zeros(batch_size, self.deter_dim, device=device)
        z_logits = torch.zeros(batch_size, self.n_categoricals,
                               self.n_classes, device=device)
        z = torch.zeros_like(z_logits)
        z[..., 0] = 1.0
        return RSSMState(h=h, z_logits=z_logits, z=z)

    # ----- transitions (TO IMPLEMENT — see module docstring "CORE ARCHITECTURE") -----
    def img_step(self, prev: RSSMState, prev_action: torch.Tensor,
                 sample: bool = True) -> RSSMState:
        raise NotImplementedError(
            "TransformerSSMDynamics.img_step: implement the KV-cached single-step "
            "prior advance (see models/transformer_ssm.py docstring). Must pass "
            "the numerical-equivalence test vs rollout_observed before use.")

    def obs_step(self, prev: RSSMState, prev_action: torch.Tensor,
                 embed: torch.Tensor, sample: bool = True
                 ) -> Tuple[RSSMState, RSSMState]:
        raise NotImplementedError(
            "TransformerSSMDynamics.obs_step: implement posterior+prior step "
            "(see module docstring).")

    def rollout_observed(self, obs: torch.Tensor, act: torch.Tensor,
                         sample: bool = True
                         ) -> Tuple[torch.Tensor, torch.Tensor,
                                    torch.Tensor, RSSMState]:
        raise NotImplementedError(
            "TransformerSSMDynamics.rollout_observed: implement teacher-forced "
            "causal-transformer rollout (see module docstring). Returns "
            "(feats, post_logits, prior_logits, last_state).")
