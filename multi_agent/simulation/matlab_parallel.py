"""MATLAB subprocess concurrency controls for external-MC evaluation.

Paper protocol is unchanged (still 3-seed AND, n_runs=100). These helpers only
overlap independent MATLAB batch processes so wall-clock time drops.
"""

from __future__ import annotations

import logging
import os
import threading
from contextlib import contextmanager
from typing import Iterator, Optional

logger = logging.getLogger(__name__)

_LOCK = threading.Lock()
_SEMAPHORE: Optional[threading.BoundedSemaphore] = None
_SEMAPHORE_SIZE = 0


def resolve_matlab_max_parallel(default: int = 3) -> int:
    """Return max concurrent MATLAB/Octave batch processes.

    Priority: ``GSM_MATLAB_MAX_PARALLEL`` env → ``simulation.matlab_max_parallel``
    config → ``default``. Values ``<= 0`` mean serial (1).
    """
    env = os.environ.get("GSM_MATLAB_MAX_PARALLEL", "").strip()
    if env:
        try:
            return max(1, int(env))
        except ValueError:
            pass
    try:
        from multi_agent.config_loader import get_config

        cfg_val = getattr(get_config().simulation, "matlab_max_parallel", None)
        if cfg_val is not None:
            return max(1, int(cfg_val))
    except Exception:
        pass
    return max(1, int(default))


def resolve_mc_and_workers(n_seeds: int, default: int = 3) -> int:
    """How many AND seeds to evaluate concurrently (capped by seed count)."""
    env = os.environ.get("GSM_MC_AND_WORKERS", "").strip()
    workers = None
    if env:
        try:
            workers = int(env)
        except ValueError:
            workers = None
    if workers is None:
        try:
            from multi_agent.config_loader import get_config

            cfg_val = getattr(get_config().simulation, "mc_and_workers", None)
            if cfg_val is not None:
                workers = int(cfg_val)
        except Exception:
            workers = None
    if workers is None:
        workers = int(default)
    if workers <= 1:
        return 1
    return max(1, min(int(n_seeds), int(workers)))


def _get_semaphore() -> threading.BoundedSemaphore:
    global _SEMAPHORE, _SEMAPHORE_SIZE
    size = resolve_matlab_max_parallel()
    with _LOCK:
        if _SEMAPHORE is None or _SEMAPHORE_SIZE != size:
            _SEMAPHORE = threading.BoundedSemaphore(size)
            _SEMAPHORE_SIZE = size
            logger.info("[matlab_parallel] max concurrent MATLAB batches = %d", size)
        return _SEMAPHORE


@contextmanager
def matlab_process_slot() -> Iterator[None]:
    """Acquire a global slot before launching a MATLAB/Octave batch process."""
    sem = _get_semaphore()
    sem.acquire()
    try:
        yield
    finally:
        sem.release()
