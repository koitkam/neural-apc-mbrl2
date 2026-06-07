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

Until the transition methods' KV-cache + dispatch wiring are done, ``build_model``
must NOT dispatch to this class.  STATUS: transitions IMPLEMENTED (naive windowed
recompute, CPU-tested via tools/_smoke_tssm.py); KV-cache + consumer-compat +
dispatch remain (see top-of-file STATUS).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn

# Reuse the proven RSSM building blocks so the categorical latent + KL are
# bit-for-bit identical to the default backbone (only the dynamics core changes).
from models.dreamer_v4_rssm import _CategoricalLatent


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
    max_seq_len: int = 256        # context window cap (>= lookback + horizon)


@dataclass
class TSSMState:
    """Duck-compatible with RSSMState (.h, .z_logits, .z, .feat, .stoch_flat)
    PLUS the transformer's token ``window`` (B, t, d_model) needed to continue
    the causal rollout.  ``window=None`` => no history (feat-only reconstruction
    by a Markovian consumer; the next step starts a fresh single-token context).
    """
    h: torch.Tensor             # (..., deter_dim) transformer output at step t
    z_logits: torch.Tensor      # (..., n_categoricals, n_classes)
    z: torch.Tensor             # (..., n_categoricals, n_classes) one-hot (ST)
    window: Optional[torch.Tensor] = None   # (..., t, deter_dim) token history

    @property
    def stoch_flat(self) -> torch.Tensor:
        return self.z.flatten(start_dim=-2)

    @property
    def feat(self) -> torch.Tensor:
        return torch.cat([self.h, self.stoch_flat], dim=-1)


def _sinusoidal_pos(n: int, d: int, device, dtype) -> torch.Tensor:
    """(n, d) sinusoidal positional encoding (Vaswani et al.)."""
    pos = torch.arange(n, device=device, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(torch.arange(0, d, 2, device=device, dtype=torch.float32)
                    * (-math.log(10000.0) / d))
    pe = torch.zeros(n, d, device=device, dtype=torch.float32)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe.to(dtype)


class TransformerSSMDynamics(nn.Module):
    """Causal-transformer dynamics core implementing the RSSMDynamics interface.

    Naive (recompute) transitions — correct + CPU-tested.  KV-cache is a future
    pure-speed optimization gated by the equivalence test (see module docstring).
    """

    def __init__(self, cfg: TransformerSSMConfig):
        super().__init__()
        self.cfg = cfg
        self.obs_dim = int(cfg.obs_dim)
        self.action_dim = int(cfg.action_dim)
        self.deter_dim = int(cfg.deter_dim)
        self.n_categoricals = int(cfg.n_categoricals)
        self.n_classes = int(cfg.n_classes)
        self.embed_dim = int(cfg.embed_dim)
        self.unimix = float(cfg.unimix)
        self.max_seq_len = int(cfg.max_seq_len)
        self.stoch_flat_dim = self.n_categoricals * self.n_classes

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

    @property
    def feat_dim(self) -> int:
        return self.deter_dim + self.stoch_flat_dim

    # ----- shared pieces (real) -----
    def embed(self, obs: torch.Tensor) -> torch.Tensor:
        return self.encoder(obs)

    def decode(self, feat: torch.Tensor) -> torch.Tensor:
        return self.decoder(feat)

    def initial_state(self, batch_size: int,
                      device: torch.device) -> TSSMState:
        h = torch.zeros(batch_size, self.deter_dim, device=device)
        z_logits = torch.zeros(batch_size, self.n_categoricals,
                               self.n_classes, device=device)
        z = torch.zeros_like(z_logits)
        z[..., 0] = 1.0
        return TSSMState(h=h, z_logits=z_logits, z=z, window=None)

    # ----- internal: token build + causal-window encode -----
    def _build_token(self, z: torch.Tensor,
                     action: torch.Tensor) -> torch.Tensor:
        """token = proj([z_flat ; action]) -> (B, d_model)."""
        z_flat = z.flatten(start_dim=-2)
        return self.token_proj(torch.cat([z_flat, action], dim=-1))

    def _encode_window(self, window: torch.Tensor) -> torch.Tensor:
        """Causal transformer over (B, S, d_model); returns (B, S, d_model)."""
        S = window.shape[1]
        pe = _sinusoidal_pos(S, self.deter_dim, window.device, window.dtype)
        x = window + pe.unsqueeze(0)
        mask = torch.triu(
            torch.ones(S, S, device=window.device, dtype=torch.bool),
            diagonal=1)  # True = disallow attending to the future
        return self.transformer(x, mask=mask)

    def _append(self, window: Optional[torch.Tensor],
                token: torch.Tensor) -> torch.Tensor:
        """Append one (B, d_model) token; cap the window at max_seq_len."""
        tok = token.unsqueeze(1)                         # (B, 1, d)
        win = tok if window is None else torch.cat([window, tok], dim=1)
        if win.shape[1] > self.max_seq_len:
            win = win[:, -self.max_seq_len:]
        return win

    # ----- transitions -----
    def img_step(self, prev: TSSMState, prev_action: torch.Tensor,
                 sample: bool = True) -> TSSMState:
        """Imagined (prior-only) step: build the token from (prev.z, action),
        append to the window, recompute the causal transformer, read the prior
        off the last position.  ``window=None`` (feat-only reconstruction by a
        Markovian consumer) starts a fresh single-token context."""
        token = self._build_token(prev.z, prev_action)
        window = self._append(getattr(prev, 'window', None), token)
        h = self._encode_window(window)[:, -1]           # (B, d_model)
        z_logits, z = self.prior_net(h, sample=sample)
        return TSSMState(h=h, z_logits=z_logits, z=z, window=window)

    def obs_step(self, prev: TSSMState, prev_action: torch.Tensor,
                 embed: torch.Tensor, sample: bool = True
                 ) -> Tuple[TSSMState, TSSMState]:
        """Observation step -> (posterior, prior).  Prior is needed for KL; both
        share ``h``; the posterior conditions on the obs embedding and is the z
        carried forward."""
        prior = self.img_step(prev, prev_action, sample=sample)
        post_in = torch.cat([prior.h, embed], dim=-1)
        post_logits, post_z = self.post_net(post_in, sample=sample)
        post = TSSMState(h=prior.h, z_logits=post_logits, z=post_z,
                         window=prior.window)
        return post, prior

    def rollout_observed(self, obs: torch.Tensor, act: torch.Tensor,
                         sample: bool = True
                         ) -> Tuple[torch.Tensor, torch.Tensor,
                                    torch.Tensor, TSSMState]:
        """Teacher-forced posterior rollout over (B, T, *).  ``act[:, t]`` drives
        the transition INTO ``obs[:, t]`` (contemporaneous-action convention, as
        in RSSMDynamics).  Returns (feats, post_logits, prior_logits,
        last_state) with shapes (B, T, F), (B, T, K, C), (B, T, K, C)."""
        B, T = obs.shape[:2]
        device = obs.device
        embeds = self.embed(obs)                         # (B, T, embed_dim)
        state = self.initial_state(B, device)
        feats_l, post_l, prior_l = [], [], []
        for t in range(T):
            post, prior = self.obs_step(state, act[:, t], embeds[:, t],
                                        sample=sample)
            state = post
            feats_l.append(post.feat)
            post_l.append(post.z_logits)
            prior_l.append(prior.z_logits)
        feats = torch.stack(feats_l, dim=1)
        post_logits = torch.stack(post_l, dim=1)
        prior_logits = torch.stack(prior_l, dim=1)
        return feats, post_logits, prior_logits, state
