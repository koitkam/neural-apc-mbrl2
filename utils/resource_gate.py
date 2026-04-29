"""Resource-aware gating for parallel training workloads.

Two user-facing knobs, nothing else:

* ``APC_CPU_MAX_PERCENT`` (1..100, default 100) -- max CPU load we are
  willing to occupy, as a percentage of machine cores.  Also gates RAM
  use at the same percentage, clamped at 95 % for OOM safety.
* ``APC_GPU_MAX_PERCENT`` (1..100, default 100) -- max GPU memory we are
  willing to occupy, as a percentage of total VRAM on the most-empty GPU.

Both knobs are read from a live config file (default
``utils/apc_resource.env`` alongside this module, overridable via
``APC_RESOURCE_CONFIG_FILE``) on every gate check, and fall back to
matching environment variables, and finally to 100/100.  Editing the
file takes effect at the next worker spawn -- no workflow restart
required.

Public API:

* :func:`acquire` blocks until CPU/GPU/RAM headroom exists.
* :func:`recommended_parallelism` returns a worker-count cap derived
  purely from the machine (cpu_count and GPU memory budget).

RAM is clamped to a hard 95 % ceiling regardless of the CPU knob so the
Linux OOM killer never gets to kill a running training.
"""
from __future__ import annotations

import os
import subprocess
import threading
import time
from typing import List, Optional


# -- Live config file -----------------------------------------------------

# Default: the tracked file alongside this module (utils/apc_resource.env).
# Override with the ``APC_RESOURCE_CONFIG_FILE`` env var if you want a
# different path (e.g. an untracked per-host override).
_LIVE_CONFIG_DEFAULT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    'apc_resource.env')

# Internal, not operator-facing. Hard ceiling on RAM so the Linux OOM
# killer never triggers even when the user asks for 100 % CPU use.
_RAM_HARD_CEILING = 0.95

# Per-worker rough footprints for recommended_parallelism.  The gate is
# the real limiter; these just prevent launching absurd subprocess counts.
_CORES_PER_WORKER = 2
_GPU_MB_PER_WORKER = 3500
_PARALLEL_HARD_CAP = 16


def _live_config_path() -> str:
    return os.environ.get('APC_RESOURCE_CONFIG_FILE') or _LIVE_CONFIG_DEFAULT


def _read_live_overrides() -> dict:
    """Parse the live KEY=VALUE config file. Returns {} if missing."""
    path = _live_config_path()
    out: dict = {}
    try:
        with open(path, 'r', encoding='utf-8') as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith('#') or '=' not in line:
                    continue
                k, _, v = line.partition('=')
                out[k.strip()] = v.strip().strip('"').strip("'")
    except FileNotFoundError:
        pass
    except Exception:
        pass
    return out


def _resolve_percent(key: str, default_pct: float = 100.0) -> float:
    """Effective percentage (1..100) for *key*.  Live file > env > default."""
    live = _read_live_overrides().get(key)
    raw = live if live is not None else os.environ.get(key)
    if raw is None:
        return float(default_pct)
    try:
        return max(1.0, min(100.0, float(raw)))
    except Exception:
        return float(default_pct)


# -- Machine measurements -------------------------------------------------

def gpu_free_mb() -> List[int]:
    """Free MB per GPU. Empty list if nvidia-smi unavailable."""
    try:
        out = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=memory.free',
             '--format=csv,noheader,nounits'],
            stderr=subprocess.DEVNULL, text=True, timeout=5,
        )
        return [int(x.strip()) for x in out.strip().splitlines() if x.strip()]
    except Exception:
        return []


def gpu_total_mb() -> List[int]:
    """Total MB per GPU. Empty list if nvidia-smi unavailable."""
    try:
        out = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=memory.total',
             '--format=csv,noheader,nounits'],
            stderr=subprocess.DEVNULL, text=True, timeout=5,
        )
        return [int(x.strip()) for x in out.strip().splitlines() if x.strip()]
    except Exception:
        return []


def gpu_best_free_ratio() -> float:
    """max(free/total) across GPUs (most-empty GPU). 1.0 if no GPU info."""
    free = gpu_free_mb()
    total = gpu_total_mb()
    if not free or not total or len(free) != len(total):
        return 1.0
    ratios = [float(f) / float(t) for f, t in zip(free, total) if t > 0]
    return max(ratios) if ratios else 1.0


def cpu_load_ratio() -> float:
    """1-min load average divided by cpu_count. 0.0 on failure."""
    try:
        load1, _, _ = os.getloadavg()
        cpus = max(1, os.cpu_count() or 1)
        return float(load1) / float(cpus)
    except Exception:
        return 0.0


def ram_used_ratio() -> float:
    """Used-RAM fraction (1 - MemAvailable/MemTotal). 0.0 on failure."""
    try:
        info: dict = {}
        with open('/proc/meminfo', 'r', encoding='utf-8') as fh:
            for line in fh:
                key, _, rest = line.partition(':')
                if not rest:
                    continue
                tokens = rest.strip().split()
                if not tokens:
                    continue
                try:
                    info[key.strip()] = int(tokens[0])
                except ValueError:
                    continue
        total = info.get('MemTotal') or 0
        avail = info.get('MemAvailable', total)
        if total <= 0:
            return 0.0
        return max(0.0, 1.0 - float(avail) / float(total))
    except Exception:
        return 0.0


# -- Gate -----------------------------------------------------------------

def _matching_external_pids() -> List[int]:
    """Return PIDs of running processes that match ``APC_WAIT_FOR_PATTERNS``.

    The env var is a comma-separated list of substrings.  A process
    matches when any pattern appears in its full command line.  Used to
    block the workflow on external workloads (e.g. a sister project's
    training run sharing the box) without requiring a tmux/job
    coordinator.  Returns the empty list if the env var is unset or
    matches nothing.
    """
    raw = os.environ.get('APC_WAIT_FOR_PATTERNS', '').strip()
    if not raw:
        return []
    patterns = [p.strip() for p in raw.split(',') if p.strip()]
    if not patterns:
        return []
    try:
        out = subprocess.check_output(
            ['ps', '-eo', 'pid=,args='], stderr=subprocess.DEVNULL,
            text=True, timeout=5,
        )
    except Exception:
        return []
    own = os.getpid()
    hits: List[int] = []
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            pid_str, _, args = line.partition(' ')
            pid = int(pid_str)
        except ValueError:
            continue
        if pid == own:
            continue
        if any(p in args for p in patterns):
            hits.append(pid)
    return hits


def resources_ok() -> bool:
    """True when current CPU/GPU/RAM use is below the live thresholds
    AND no process matches ``APC_WAIT_FOR_PATTERNS``."""
    if _matching_external_pids():
        return False
    cpu_cap = _resolve_percent('APC_CPU_MAX_PERCENT') / 100.0
    gpu_pct = _resolve_percent('APC_GPU_MAX_PERCENT')
    min_gpu_free_ratio = max(0.0, 1.0 - (gpu_pct / 100.0))
    ram_cap = min(cpu_cap, _RAM_HARD_CEILING)

    gpus = gpu_free_mb()
    gpu_ok = True if not gpus else (gpu_best_free_ratio() >= min_gpu_free_ratio)
    cpu_ok = cpu_load_ratio() < cpu_cap
    ram_ok = ram_used_ratio() < ram_cap
    return gpu_ok and cpu_ok and ram_ok


def acquire(poll_sec: float = 5.0,
            timeout_sec: Optional[float] = None,
            label: str = '') -> None:
    """Block until host has spare capacity.

    Thresholds are re-read on every poll from the live config file, so
    retuning takes effect at the next spawn without workflow restart.
    """
    start = time.monotonic()
    waited = False
    while True:
        if resources_ok():
            if waited:
                print(f'[RESOURCE_GATE] {label or "task"}: capacity available '
                      f'after {time.monotonic() - start:.1f}s')
            return
        if timeout_sec is not None and (time.monotonic() - start) > float(timeout_sec):
            print(f'[RESOURCE_GATE] {label or "task"}: timeout after '
                  f'{timeout_sec:.0f}s, proceeding anyway')
            return
        if not waited:
            cpu_cap = _resolve_percent('APC_CPU_MAX_PERCENT') / 100.0
            gpu_cap_pct = _resolve_percent('APC_GPU_MAX_PERCENT')
            ram_cap = min(cpu_cap, _RAM_HARD_CEILING)
            ext_pids = _matching_external_pids()
            ext_msg = (f' external_pids={ext_pids}' if ext_pids else '')
            print(
                f'[RESOURCE_GATE] {label or "task"}: waiting '
                f'(cpu_load={cpu_load_ratio():.2f}/cap {cpu_cap:.2f} '
                f'gpu_free={gpu_best_free_ratio():.2f}/'
                f'need {1.0 - gpu_cap_pct / 100.0:.2f} '
                f'ram_used={ram_used_ratio():.2f}/cap {ram_cap:.2f}'
                f'{ext_msg})'
            )
            waited = True
        time.sleep(max(0.5, float(poll_sec)))


# -- Worker-count recommendation ------------------------------------------

def recommended_parallelism() -> int:
    """Max concurrent worker slots to open.

    Auto-derived purely from the machine (cpu_count and GPU memory
    budget).  The real throttle is :func:`acquire`, so this number is
    intentionally generous -- it only prevents launching an absurd
    subprocess count.
    """
    cpu_budget = max(1, (os.cpu_count() or 1) // _CORES_PER_WORKER)
    gpus = gpu_free_mb()
    if gpus:
        gpu_budget = sum(max(0, free // _GPU_MB_PER_WORKER) for free in gpus)
        budget = max(gpu_budget or 1, cpu_budget)
    else:
        budget = cpu_budget
    return max(1, min(_PARALLEL_HARD_CAP, budget))


# -- One-shot announcer ---------------------------------------------------

_ANNOUNCE_LOCK = threading.Lock()
_ANNOUNCED: set = set()


def announce_once(label: str, n_workers: int) -> None:
    """Print the chosen parallelism exactly once per (label, n) pair."""
    key = f'{label}:{n_workers}'
    with _ANNOUNCE_LOCK:
        if key in _ANNOUNCED:
            return
        _ANNOUNCED.add(key)
    print(f'[PARALLEL] {label}: max_workers={n_workers}')
