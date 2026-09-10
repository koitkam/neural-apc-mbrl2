"""CPU proof for GOAL_PLAN #4: P116-class CV hunting vs the MV-relu analogue.

Val ``cv_reversal_rate`` is ``_sign_reversal_rate`` (nonzero-Δ sign flips /
steps, deadband = 1e-3·bound width). P116 FAIL is rev **0.421** with d2
**0.034 PASS** — frequent small hunting near the economic limit, not large
bang-bang. The MV term ``relu(-du_t·du_prev)`` is amplitude-weighted; the
same formula on dCV is nearly silent on that hunting, so copying it would
not close R2.

#4 must use a sticky last-nonzero sign surrogate of ``_sign_reversal_rate``
with sticky threshold ``tanh(1)`` (same as val ``|d|>deadband``), not
``relu(-dCV_t·dCV_prev)`` and not ``|s|>=0.5`` (sub-deadband holds). Do not
land on the P123 pid.

Run:
  CUDA_VISIBLE_DEVICES="" PYTHONPATH=$PWD \\
    ~/neural-APC-mbrl2-env/bin/python tools/_test_cv_reversal_surrogate.py
"""
from __future__ import annotations

import numpy as np

from evaluation.residual_board import CV_D2_RMS_MAX, CV_REVERSAL_MAX
from evaluation.validate import _sign_reversal_rate

# tanh(1) ⇔ |dCV| > deadband, matching val dropping |d|<=deadband zeros.
_STICKY_THRESH = float(np.tanh(1.0))


def _d2_rms_normed(col: np.ndarray, rng: float) -> float:
    if col.size < 3:
        return 0.0
    return float(np.sqrt(np.mean((np.diff(col, n=2) / max(1e-9, rng)) ** 2)))


def _amp_relu_mean(col: np.ndarray, rng: float) -> float:
    """MV-style analogue on CV: mean relu(-d_t·d_prev) / rng²."""
    d = np.diff(np.asarray(col, dtype='float64'))
    if d.size < 2:
        return 0.0
    prod = -d[1:] * d[:-1]
    return float(np.mean(np.maximum(0.0, prod)) / max(1e-9, rng * rng))


def _sticky_tanh_flip_mean(col: np.ndarray, rng: float) -> float:
    """GOAL_PLAN #4 surrogate: mean relu(-s_t · sign_sticky).

    ``s = tanh(dCV / deadband)``; ``s_sticky`` is the last ``|s|>=tanh(1)``
    sign. Holds (sub-deadband) do not update sticky; an opposite move after a
    hold still counts, matching val's nonzero-Δ flip rate.
    """
    d = np.diff(np.asarray(col, dtype='float64'))
    if d.size == 0:
        return 0.0
    deadband = 1e-3 * max(float(rng), 1e-9)
    s = np.tanh(d / deadband)
    sticky = 0.0
    pens = np.empty(s.shape, dtype='float64')
    for i, st in enumerate(s):
        if abs(sticky) >= _STICKY_THRESH:
            pens[i] = max(0.0, -float(st) * float(np.sign(sticky)))
        else:
            pens[i] = 0.0
        if abs(st) >= _STICKY_THRESH:
            sticky = float(np.sign(st))
    return float(np.mean(pens))


def _p116_class_hunting(n: int = 400, rng: float = 7.0) -> np.ndarray:
    """Small limit-hunting: 3-step ramps of 0.03 (≫ deadband 0.007, ≪ band)."""
    lo, hi = 78.5, 78.5 + rng
    y = np.empty(n, dtype='float64')
    y[0] = hi - 0.25
    sign = 1.0
    steps_in_leg = 0
    for t in range(1, n):
        y[t] = y[t - 1] + sign * 0.03
        steps_in_leg += 1
        if steps_in_leg >= 3:
            sign = -sign
            steps_in_leg = 0
        y[t] = float(np.clip(y[t], lo + 0.05, hi - 0.05))
    return y


def _monotonic_ride_to_hi(n: int = 400, rng: float = 7.0) -> np.ndarray:
    """Slow ride toward the hi limit — one sign, should not look like hunting."""
    lo, hi = 78.5, 78.5 + rng
    y = np.linspace(lo + 0.4 * rng, hi - 0.05, n, dtype='float64')
    return y


def main() -> None:
    rng = 7.0  # test_sim CONTROL_TEMP 78.5–85.5
    hunt = _p116_class_hunting(rng=rng)
    rev = _sign_reversal_rate(hunt, rng)
    d2 = _d2_rms_normed(hunt, rng)
    amp = _amp_relu_mean(hunt, rng)
    sticky = _sticky_tanh_flip_mean(hunt, rng)
    print(f'[hunt] rev={rev:.3f} d2={d2:.4f} amp_relu/rng²={amp:.6f} '
          f'sticky={sticky:.3f} thresh=tanh(1)={_STICKY_THRESH:.3f}')
    assert d2 <= CV_D2_RMS_MAX, f'fixture d2 {d2} should PASS like P116'
    assert rev > CV_REVERSAL_MAX, f'fixture rev {rev} should FAIL like P116'
    # Bang-bang of 0.5·rng has amp relu ~0.25; hunting must be ≪ that.
    bang = np.array([82.0 + (0.5 * rng if i % 2 else 0.0) for i in range(400)])
    bang_amp = _amp_relu_mean(bang, rng)
    print(f'[bang] amp_relu/rng²={bang_amp:.4f}')
    assert amp < 0.05 * bang_amp
    # Sticky tanh must fire on hunting at val-rate order, not vanish like amp.
    assert sticky > CV_REVERSAL_MAX
    assert abs(sticky - rev) < 0.05, (sticky, rev)
    ride = _monotonic_ride_to_hi(rng=rng)
    ride_rev = _sign_reversal_rate(ride, rng)
    ride_sticky = _sticky_tanh_flip_mean(ride, rng)
    print(f'[ride] rev={ride_rev:.3f} sticky={ride_sticky:.3f}')
    assert ride_rev == 0.0
    assert ride_sticky == 0.0
    print('[ok] amplitude-weighted CV relu is silent on P116-class hunting; '
          '#4 sticky tanh(1) matches _sign_reversal_rate and is 0 on a '
          'monotonic ride — not relu(-dCV·dCV_prev)')


if __name__ == '__main__':
    main()
