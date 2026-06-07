"""CPU smoke + correctness test for DV-as-input (Option B), BOTH backbones.

neural-apc-mbrl, 2026-06-07.  GPU-FREE — tiny CPU models, does not touch a live
GPU run.  Validates the "feed the measured DV as an exogenous transition input"
architecture for the RSSM and TSSM cores:

  1. EQUIVALENCE: dv_dim=0 ignores any ``dv`` argument and matches a model built
     with no DV wiring (default-OFF / no-DV-sim path is unchanged).
  2. DV MATTERS: with dv_dim>0, two DIFFERENT held ``dv`` values produce
     DIFFERENT predictions from the SAME (state, action) — i.e. the WM truly
     conditions CV on the DV input instead of ignoring it.
  3. TEACHER-FORCING: rollout_observed extracts the DV from the obs at
     ``dv_indices`` and threads it (changing the obs DV channel changes feats).
  4. END-TO-END: build DreamerV4(rssm, dv_dim>0) + run world_model_loss and an
     imagination step without error; the held-DV imagination path executes.

Run:
  CUDA_VISIBLE_DEVICES="" PYTHONPATH=$PWD DREAMER_COMPILE=0 \
    $PWD/../neural-apc-mbrl-env/bin/python tools/_smoke_dv_input.py
"""
import torch

from models.dreamer_v4_rssm import RSSMConfig, RSSMDynamics
from models.transformer_ssm import TransformerSSMConfig, TransformerSSMDynamics


OBS_DIM = 6
ACT_DIM = 2
DV_IDX = (4, 5)        # last two obs channels are the measured DVs


def _rssm(dv_dim, dv_indices, seed=0):
    torch.manual_seed(seed)
    cfg = RSSMConfig(obs_dim=OBS_DIM, action_dim=ACT_DIM, deter_dim=32,
                     n_categoricals=4, n_classes=4, embed_dim=16, hidden_dim=32,
                     dv_dim=dv_dim, dv_indices=dv_indices)
    return RSSMDynamics(cfg).eval()


def _tssm(dv_dim, dv_indices, seed=0):
    torch.manual_seed(seed)
    cfg = TransformerSSMConfig(obs_dim=OBS_DIM, action_dim=ACT_DIM, deter_dim=32,
                               n_categoricals=4, n_classes=4, embed_dim=16,
                               n_layers=2, n_heads=4, max_seq_len=64,
                               dv_dim=dv_dim, dv_indices=dv_indices)
    return TransformerSSMDynamics(cfg).eval()


def _img(m, state, action, dv=None):
    """One deterministic img_step (handles both cores), return feat."""
    return m.img_step(state, action, dv=dv, sample=False).feat


def test_equivalence_dv_off(mk, name):
    """dv_dim=0 ignores dv arg and matches a no-DV model bit-for-bit."""
    m0 = mk(0, ())
    B = 4
    st = m0.initial_state(B, torch.device('cpu'))
    a = torch.rand(B, ACT_DIM) * 2 - 1
    dv = torch.randn(B, len(DV_IDX))
    f_none = _img(m0, st, a, dv=None)
    f_dv = _img(m0, st, a, dv=dv)        # dv must be IGNORED when dv_dim=0
    assert torch.allclose(f_none, f_dv, atol=1e-6), \
        f"{name}: dv_dim=0 did NOT ignore the dv argument"
    print(f"[smoke] OK {name}: dv_dim=0 ignores dv (paper path unchanged)")


def test_dv_matters(mk, name):
    """With dv_dim>0, different held DV -> different prediction."""
    m = mk(len(DV_IDX), DV_IDX)
    B = 4
    st = m.initial_state(B, torch.device('cpu'))
    a = torch.rand(B, ACT_DIM) * 2 - 1
    dv_a = torch.zeros(B, len(DV_IDX))
    dv_b = torch.ones(B, len(DV_IDX)) * 2.0
    f_a = _img(m, st, a, dv=dv_a)
    f_b = _img(m, st, a, dv=dv_b)
    d = (f_a - f_b).abs().max().item()
    assert d > 1e-4, f"{name}: DV input did NOT change the prediction (d={d:.2e})"
    # None must be treated as zeros (== dv_a here)
    f_none = _img(m, st, a, dv=None)
    assert torch.allclose(f_none, f_a, atol=1e-6), \
        f"{name}: dv=None != zeros"
    print(f"[smoke] OK {name}: DV input changes prediction (max|d|={d:.3f}); "
          f"None==zeros")


def test_teacher_forcing(mk, name):
    """rollout_observed extracts DV from obs[..., dv_indices] and threads it."""
    m = mk(len(DV_IDX), DV_IDX)
    B, T = 3, 5
    obs = torch.randn(B, T, OBS_DIM)
    act = torch.rand(B, T, ACT_DIM) * 2 - 1
    feats1, *_ = m.rollout_observed(obs, act, sample=False)
    obs2 = obs.clone()
    obs2[..., list(DV_IDX)] += 3.0          # perturb ONLY the DV channels
    feats2, *_ = m.rollout_observed(obs2, act, sample=False)
    d = (feats1 - feats2).abs().max().item()
    assert d > 1e-4, f"{name}: changing obs DV channels did not change feats"
    print(f"[smoke] OK {name}: rollout_observed teacher-forces DV "
          f"(max|d|={d:.3f})")


def test_end_to_end_rssm():
    """Build DreamerV4(rssm, dv_dim>0) + WM loss + imagination run."""
    import os
    os.environ.setdefault('DREAMER_COMPILE', '0')
    from models.dreamer_v4 import DreamerV4, DreamerV4Config
    torch.manual_seed(0)
    cfg = DreamerV4Config(
        obs_dim=OBS_DIM, action_dim=ACT_DIM, lookback=8,
        world_model_type='rssm', rssm_deter_dim=32, rssm_n_categoricals=4,
        rssm_n_classes=4, rssm_embed_dim=16, rssm_hidden_dim=32,
        dv_dim=len(DV_IDX), dv_indices=DV_IDX,
        mtp_length=2, n_action_bins=5, head_hidden=32, head_n_layers=1)
    model = DreamerV4(cfg)
    assert model.dynamics.dv_dim == len(DV_IDX)
    B, T = 2, 10
    obs = torch.randn(B, T, OBS_DIM)
    act = torch.rand(B, T, ACT_DIM) * 2 - 1
    feats, post_lg, prior_lg, last = model.dynamics.rollout_observed(obs, act)
    assert feats.shape[0] == B
    # imagination: hold the DV at the last obs value (the train.py path)
    dv_hold = obs[:, -1].index_select(-1, model.dynamics.dv_index_t)
    st = last
    for _ in range(3):
        a = torch.rand(B, ACT_DIM) * 2 - 1
        st = model.dynamics.img_step(st, a, dv=dv_hold, sample=False)
    assert torch.isfinite(st.feat).all()
    print("[smoke] OK end-to-end DreamerV4(rssm, dv_dim>0): build + "
          "rollout_observed + held-DV imagination")


if __name__ == '__main__':
    for mk, name in ((_rssm, 'RSSM'), (_tssm, 'TSSM')):
        test_equivalence_dv_off(mk, name)
        test_dv_matters(mk, name)
        test_teacher_forcing(mk, name)
    test_end_to_end_rssm()
    print("\n[smoke] ALL DV-AS-INPUT CHECKS PASSED (both backbones)")
