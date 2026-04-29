"""Utilities for time-window subsampling used by latent observer pipelines.

The workflow can keep a long physical lookback horizon while reducing model input
length by sampling every Nth point. Example: horizon=50 and sample_rate=5 yields
10 sampled points.
"""

import math
import json
import numpy as np


def normalize_sample_rate(sample_rate: int) -> int:
    try:
        sr = int(sample_rate)
    except Exception:
        sr = 1
    return max(1, sr)


def effective_lookback(lookback_horizon: int, sample_rate: int) -> int:
    horizon = max(1, int(lookback_horizon))
    sr = normalize_sample_rate(sample_rate)
    return max(1, int(math.ceil(horizon / float(sr))))


def init_raw_history(state: np.ndarray, lookback_horizon: int) -> np.ndarray:
    horizon = max(1, int(lookback_horizon))
    return np.repeat(np.asarray(state, dtype="float32")[None, :], horizon, axis=0)


def update_raw_history(raw_hist: np.ndarray, new_state: np.ndarray) -> None:
    raw_hist[:-1] = raw_hist[1:]
    raw_hist[-1] = np.asarray(new_state, dtype="float32")


def sample_history_window(raw_hist: np.ndarray, sample_rate: int, output_len: int = None) -> np.ndarray:
    # Sample from most recent backward with step=sample_rate, then restore time order.
    sr = normalize_sample_rate(sample_rate)
    sampled = raw_hist[::-1][::sr][::-1]

    if output_len is not None:
        out_n = max(1, int(output_len))
        if sampled.shape[0] > out_n:
            sampled = sampled[-out_n:]
        elif sampled.shape[0] < out_n:
            pad = np.repeat(sampled[:1], out_n - sampled.shape[0], axis=0)
            sampled = np.concatenate([pad, sampled], axis=0)

    return np.asarray(sampled, dtype="float32")


def parse_feature_lookback_vector(raw_value, state_dim: int, default_lookback: int) -> np.ndarray:
    default_val = max(1, int(default_lookback))
    n = max(1, int(state_dim))

    value = raw_value
    if isinstance(raw_value, str):
        if raw_value.strip() == "":
            value = None
        else:
            try:
                value = json.loads(raw_value)
            except Exception:
                value = None

    if value is None:
        return np.full((n,), default_val, dtype=np.int32)

    if isinstance(value, dict):
        out = np.full((n,), default_val, dtype=np.int32)
        for k, v in value.items():
            try:
                idx = int(k)
                if 0 <= idx < n:
                    out[idx] = max(1, int(v))
            except Exception:
                continue
        return out

    if isinstance(value, (list, tuple)):
        arr = np.full((n,), default_val, dtype=np.int32)
        for i, v in enumerate(value[:n]):
            try:
                arr[i] = max(1, int(v))
            except Exception:
                arr[i] = default_val
        return arr

    return np.full((n,), default_val, dtype=np.int32)


def parse_feature_sample_rate_vector(raw_value, state_dim: int, default_sample_rate: int) -> np.ndarray:
    default_val = normalize_sample_rate(default_sample_rate)
    n = max(1, int(state_dim))

    value = raw_value
    if isinstance(raw_value, str):
        if raw_value.strip() == "":
            value = None
        else:
            try:
                value = json.loads(raw_value)
            except Exception:
                value = None

    if value is None:
        return np.full((n,), default_val, dtype=np.int32)

    if isinstance(value, dict):
        out = np.full((n,), default_val, dtype=np.int32)
        for k, v in value.items():
            try:
                idx = int(k)
                if 0 <= idx < n:
                    out[idx] = normalize_sample_rate(v)
            except Exception:
                continue
        return out

    if isinstance(value, (list, tuple)):
        arr = np.full((n,), default_val, dtype=np.int32)
        for i, v in enumerate(value[:n]):
            try:
                arr[i] = normalize_sample_rate(v)
            except Exception:
                arr[i] = default_val
        return arr

    return np.full((n,), default_val, dtype=np.int32)


def sample_history_window_feature_scan(
    raw_hist: np.ndarray,
    base_sample_rate: int,
    output_len: int = None,
    feature_sample_rates: np.ndarray = None,
) -> np.ndarray:
    base = sample_history_window(raw_hist, sample_rate=base_sample_rate, output_len=output_len)
    if feature_sample_rates is None:
        return base

    w = np.asarray(base, dtype="float32")
    if w.ndim != 2:
        return w

    t_len, f_dim = w.shape
    fsr = np.asarray(feature_sample_rates, dtype=np.int32)
    if fsr.shape[0] != f_dim:
        return w

    out = w.copy()
    for j in range(f_dim):
        sr_j = normalize_sample_rate(int(fsr[j]))
        col = np.asarray(raw_hist[:, j:j + 1], dtype="float32")
        sampled_col = sample_history_window(col, sample_rate=sr_j, output_len=t_len)[:, 0]
        out[:, j] = sampled_col
    return out


def mask_window_by_feature_lookback(window: np.ndarray, feature_lookbacks: np.ndarray) -> np.ndarray:
    w = np.asarray(window, dtype="float32")
    if w.ndim != 2:
        return w

    t_len, f_dim = w.shape
    lb = np.asarray(feature_lookbacks, dtype=np.int32)
    if lb.shape[0] != f_dim:
        return w

    out = w.copy()
    for j in range(f_dim):
        keep = int(max(1, min(t_len, lb[j])))
        cutoff = t_len - keep
        if cutoff > 0:
            out[:cutoff, j] = out[cutoff, j]
    return out
