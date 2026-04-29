"""State normalization helpers for simulator I/O and model-facing tensors.

These helpers enforce a single normalized representation for observer/agent
inputs while still allowing objective calculations in either normalized or
absolute engineering units.
"""

from typing import List, Optional

import numpy as np


def _safe_float(v, default=0.0) -> float:
    try:
        return float(v)
    except Exception:
        return float(default)


def _coerce_ranges(raw) -> List[List[float]]:
    out = []
    src = raw if isinstance(raw, list) else []
    for item in src:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            out.append([])
            continue
        lo = _safe_float(item[0], 0.0)
        hi = _safe_float(item[1], lo + 1.0)
        if hi <= lo:
            hi = lo + 1.0
        out.append([lo, hi])
    return out


def _index_map(indices, ranges, state_dim: int) -> List[Optional[List[float]]]:
    out: List[Optional[List[float]]] = [None for _ in range(state_dim)]
    idxs = indices if isinstance(indices, list) else []
    for pos, idx in enumerate(idxs):
        try:
            i = int(idx)
        except Exception:
            continue
        if not (0 <= i < state_dim):
            continue
        if 0 <= pos < len(ranges) and len(ranges[pos]) >= 2:
            out[i] = [float(ranges[pos][0]), float(ranges[pos][1])]
    return out


def _merge_ranges(preferred: List[Optional[List[float]]], fallback: List[Optional[List[float]]]) -> List[Optional[List[float]]]:
    out: List[Optional[List[float]]] = []
    n = max(len(preferred), len(fallback))
    for i in range(n):
        p = preferred[i] if i < len(preferred) else None
        f = fallback[i] if i < len(fallback) else None
        out.append(p if p is not None else f)
    return out


def _derive_sv_ranges_from_names(names: List[str], by_index: List[Optional[List[float]]]) -> None:
    if not names:
        return
    name_to_index = {str(n): i for i, n in enumerate(names)}
    for i, name in enumerate(names):
        if by_index[i] is not None:
            continue
        s = str(name)
        if '_SV_' not in s:
            continue
        pv_name = s.replace('_SV_', '_PV_')
        j = name_to_index.get(pv_name)
        if j is None:
            continue
        if 0 <= int(j) < len(by_index) and by_index[int(j)] is not None:
            by_index[i] = list(by_index[int(j)])


def resolve_state_normalization_ranges(sim) -> List[List[float]]:
    cached = getattr(sim, '_resolved_state_normalization_ranges', None)
    if isinstance(cached, list) and cached:
        return cached

    state_dim = int(len(getattr(sim, 'state_variables', [])) or 0)
    if state_dim <= 0:
        state_dim = int(getattr(sim, 'state_dim', 0) or 0)
    if state_dim <= 0:
        raise ValueError('Cannot resolve state normalization ranges: missing simulator state dimension.')

    explicit = _coerce_ranges(getattr(sim, 'state_normalization_ranges', []))
    explicit_by_idx: List[Optional[List[float]]] = [None for _ in range(state_dim)]
    for i in range(min(state_dim, len(explicit))):
        if len(explicit[i]) >= 2:
            explicit_by_idx[i] = [float(explicit[i][0]), float(explicit[i][1])]

    mv_map = _index_map(
        list(getattr(sim, 'mv_indices', [])),
        _coerce_ranges(getattr(sim, 'mv_normalization_ranges', [])),
        state_dim,
    )
    cv_map = _index_map(
        list(getattr(sim, 'cv_indices', [])),
        _coerce_ranges(getattr(sim, 'cv_normalization_ranges', [])),
        state_dim,
    )
    dv_map = _index_map(
        list(getattr(sim, 'dv_indices', [])),
        _coerce_ranges(getattr(sim, 'dv_normalization_ranges', [])),
        state_dim,
    )

    merged = _merge_ranges(explicit_by_idx, _merge_ranges(mv_map, _merge_ranges(cv_map, dv_map)))
    names = [str(x) for x in list(getattr(sim, 'state_variables', []))]
    _derive_sv_ranges_from_names(names, merged)

    unresolved = [i for i, rng in enumerate(merged) if rng is None]
    if unresolved:
        raise ValueError(
            'State normalization ranges are incomplete for indices: '
            + ', '.join(str(i) for i in unresolved)
            + '. Provide io.state_normalization_ranges in control_setup.json for every state variable.'
        )

    resolved = [[float(r[0]), float(r[1])] for r in merged if r is not None]
    setattr(sim, '_resolved_state_normalization_ranges', resolved)
    return resolved


def _range_for_index(sim, state_index: int) -> Optional[List[float]]:
    idx = int(state_index)
    ranges = resolve_state_normalization_ranges(sim)
    if 0 <= idx < len(ranges):
        return [float(ranges[idx][0]), float(ranges[idx][1])]
    return None


def normalize_value(value: float, lo: float, hi: float) -> float:
    span = max(1e-6, float(hi) - float(lo))
    return (float(value) - float(lo)) / span


def denormalize_value(value: float, lo: float, hi: float) -> float:
    return float(lo) + float(value) * max(1e-6, float(hi) - float(lo))


def normalize_state_vector(state, sim) -> np.ndarray:
    x = np.asarray(state, dtype='float32').reshape(-1)
    ranges = resolve_state_normalization_ranges(sim)
    if len(x) != len(ranges):
        raise ValueError(f'State dimension mismatch: got {len(x)}, expected {len(ranges)}.')

    out = np.empty_like(x)
    for i in range(len(x)):
        lo, hi = ranges[i]
        out[i] = float(normalize_value(float(x[i]), lo, hi))
    return out.astype('float32')


def normalize_state_window(window, sim) -> np.ndarray:
    w = np.asarray(window, dtype='float32')
    if w.ndim != 2:
        return w.astype('float32')
    ranges = np.asarray(resolve_state_normalization_ranges(sim), dtype='float32')
    if w.shape[1] != ranges.shape[0]:
        raise ValueError(f'State window dimension mismatch: got {w.shape[1]}, expected {ranges.shape[0]}.')
    lo = ranges[:, 0][None, :]
    span = np.maximum(1e-6, ranges[:, 1] - ranges[:, 0])[None, :]
    return ((w - lo) / span).astype('float32')


def to_symmetric(x: np.ndarray) -> np.ndarray:
    """Map a [0, 1]-normalised tensor into [-1, 1]."""
    return np.asarray(x, dtype='float32') * 2.0 - 1.0


def from_symmetric(x: np.ndarray) -> np.ndarray:
    """Inverse of :func:`to_symmetric`."""
    return (np.asarray(x, dtype='float32') + 1.0) * 0.5


def normalize_state_vector_symmetric(state, sim) -> np.ndarray:
    """Return a state vector in [-1, 1] based on the simulator's ranges."""
    return to_symmetric(normalize_state_vector(state, sim))


def normalize_state_window_symmetric(window, sim) -> np.ndarray:
    """Return a state window in [-1, 1] (one timestep per row)."""
    return to_symmetric(normalize_state_window(window, sim))


def state_value_in_mode(state, sim, state_index: int, use_normalized: bool) -> float:
    idx = int(state_index)
    raw_value = float(np.asarray(state, dtype='float32').reshape(-1)[idx])
    rng = _range_for_index(sim, idx)
    if rng is None:
        return raw_value
    lo, hi = float(rng[0]), float(rng[1])
    is_norm = bool(getattr(sim, 'state_is_normalized', False))
    if use_normalized:
        if is_norm:
            return raw_value
        return normalize_value(raw_value, lo, hi)
    if is_norm:
        return denormalize_value(raw_value, lo, hi)
    return raw_value
