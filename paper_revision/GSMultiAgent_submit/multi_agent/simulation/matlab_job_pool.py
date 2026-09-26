"""Persistent MATLAB batch workers for external-MC jobs (file queue).

Avoids paying MATLAB cold-start on every ``run_mc_with_external_data`` call.
Falls back to one-shot ``matlab -batch`` when the pool is disabled or unhealthy.

Enable via ``simulation.matlab_job_pool: true`` or ``GSM_MATLAB_JOB_POOL=1``.
"""

from __future__ import annotations

import atexit
import json
import logging
import os
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
WORKER_M_DIR = ROOT / "knowledge_base" / "matlab" / "runtime"
DEFAULT_JOB_ROOT = ROOT / "guidance_output" / "matlab_job_pool"

_POOL_LOCK = threading.Lock()
_POOL: Optional["MatlabJobPool"] = None


def job_pool_enabled(default: bool = True) -> bool:
    env = os.environ.get("GSM_MATLAB_JOB_POOL", "").strip().lower()
    if env in ("0", "false", "no", "off"):
        return False
    if env in ("1", "true", "yes", "on"):
        return True
    try:
        from multi_agent.config_loader import get_config

        return bool(getattr(get_config().simulation, "matlab_job_pool", default))
    except Exception:
        return bool(default)


def resolve_pool_workers(default: int = 3) -> int:
    env = os.environ.get("GSM_MATLAB_POOL_WORKERS", "").strip()
    if env:
        try:
            return max(1, int(env))
        except ValueError:
            pass
    try:
        from multi_agent.config_loader import get_config
        from multi_agent.simulation.matlab_parallel import resolve_matlab_max_parallel

        cfg = int(getattr(get_config().simulation, "matlab_pool_workers", 0) or 0)
        if cfg > 0:
            return max(1, cfg)
        return resolve_matlab_max_parallel(default)
    except Exception:
        return max(1, int(default))


class MatlabJobPool:
    """Own N long-lived ``matlab -batch gsm_extmc_worker(...)`` processes."""

    def __init__(
        self,
        *,
        n_workers: int,
        matlab_path: str,
        job_root: Path,
    ) -> None:
        self.n_workers = max(1, int(n_workers))
        self.matlab_path = matlab_path
        self.job_root = Path(job_root)
        self.job_root.mkdir(parents=True, exist_ok=True)
        self._procs: List[subprocess.Popen] = []
        self._rr = 0
        self._submit_lock = threading.Lock()
        self._started = False

    @property
    def started(self) -> bool:
        return self._started and any(p.poll() is None for p in self._procs)

    def start(self) -> bool:
        if self._started and self.started:
            return True
        self.stop()
        worker_m = WORKER_M_DIR / "gsm_extmc_worker.m"
        if not worker_m.is_file():
            logger.warning("[matlab_job_pool] missing worker script %s", worker_m)
            return False
        worker_dir = str(WORKER_M_DIR.resolve()).replace("\\", "/")
        ok_any = False
        for i in range(self.n_workers):
            wdir = self.job_root / f"w{i}"
            for sub in ("pending", "running", "done"):
                (wdir / sub).mkdir(parents=True, exist_ok=True)
            wdir_s = str(wdir.resolve()).replace("\\", "/")
            eval_str = (
                "set(0,'DefaultFigureVisible','off'); "
                "try, maxNumCompThreads(2); end; "
                "try, delete(gcp('nocreate')); end; "
                f"addpath('{worker_dir}'); "
                f"gsm_extmc_worker('{wdir_s}');"
            )
            cmd = [
                self.matlab_path,
                "-nosplash",
                "-nodesktop",
                "-nodisplay",
                "-batch",
                eval_str,
            ]
            try:
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    cwd=str(ROOT),
                )
                self._procs.append(proc)
                ok_any = True
                logger.info(
                    "[matlab_job_pool] started worker %d pid=%s dir=%s",
                    i,
                    proc.pid,
                    wdir,
                )
            except Exception as exc:
                logger.warning("[matlab_job_pool] failed to start worker %d: %s", i, exc)
        self._started = ok_any
        if ok_any:
            # Give MATLAB a moment to enter the poll loop.
            time.sleep(2.0)
        return ok_any

    def stop(self) -> None:
        for proc in self._procs:
            try:
                if proc.poll() is None:
                    proc.kill()
                    proc.wait(timeout=5)
            except Exception:
                pass
        self._procs = []
        self._started = False

    def _pick_worker_dir(self) -> Path:
        alive = [i for i, p in enumerate(self._procs) if p.poll() is None]
        if not alive:
            raise RuntimeError("no alive MATLAB workers")
        with self._submit_lock:
            self._rr = (self._rr + 1) % len(alive)
            idx = alive[self._rr % len(alive)]
        return self.job_root / f"w{idx}"

    def run_eval(self, eval_str: str, *, timeout_sec: float) -> Dict[str, Any]:
        """Submit an eval string and wait for stdout / ok flag."""
        if not self.started:
            if not self.start():
                raise RuntimeError("MATLAB job pool failed to start")
        wdir = self._pick_worker_dir()
        job_id = uuid.uuid4().hex
        req_path = wdir / "pending" / f"{job_id}.req.json"
        res_path = wdir / "done" / f"{job_id}.res.json"
        payload = {"job_id": job_id, "eval_str": eval_str}
        tmp = req_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        tmp.replace(req_path)
        deadline = time.perf_counter() + float(timeout_sec)
        while time.perf_counter() < deadline:
            if res_path.is_file():
                try:
                    data = json.loads(res_path.read_text(encoding="utf-8"))
                except Exception:
                    time.sleep(0.1)
                    continue
                try:
                    res_path.unlink(missing_ok=True)
                except Exception:
                    pass
                return {
                    "ok": bool(data.get("ok", False)),
                    "stdout": str(data.get("stdout") or ""),
                    "job_id": job_id,
                }
            # If all workers died, fail fast.
            if not any(p.poll() is None for p in self._procs):
                raise RuntimeError("all MATLAB workers exited")
            time.sleep(0.2)
        try:
            req_path.unlink(missing_ok=True)
        except Exception:
            pass
        raise TimeoutError(f"MATLAB job timed out after {timeout_sec}s (job_id={job_id})")


def get_job_pool() -> Optional[MatlabJobPool]:
    """Return process-wide pool, starting it on demand when enabled."""
    global _POOL
    if not job_pool_enabled():
        return None
    with _POOL_LOCK:
        if _POOL is not None and _POOL.started:
            return _POOL
        try:
            from multi_agent.config_loader import get_config
            from multi_agent.simulation.engine_resolver import resolve_engine_config

            sim = get_config().simulation
            eng = resolve_engine_config(
                engine=sim.engine,
                octave_path=sim.octave_path,
                matlab_path=sim.matlab_path,
            )
            matlab_path = eng.matlab_path or sim.matlab_path or "matlab"
        except Exception:
            matlab_path = os.environ.get("MATLAB_PATH", "matlab")
        n_workers = resolve_pool_workers()
        env_root = os.environ.get("GSM_MATLAB_JOB_POOL_DIR", "").strip()
        job_root = Path(env_root) if env_root else DEFAULT_JOB_ROOT
        pool = MatlabJobPool(
            n_workers=n_workers,
            matlab_path=str(matlab_path),
            job_root=job_root,
        )
        if pool.start():
            _POOL = pool
            return _POOL
        return None


def shutdown_job_pool() -> None:
    global _POOL
    with _POOL_LOCK:
        if _POOL is not None:
            _POOL.stop()
            _POOL = None


atexit.register(shutdown_job_pool)
