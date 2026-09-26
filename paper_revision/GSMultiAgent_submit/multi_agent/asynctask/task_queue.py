"""
File-backed async task queue for long-running CLI / PPO jobs.

Provides task_id + status polling without requiring Celery/Redis.
Designed so a future HTTP layer or Redis backend can replace the store.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

ENV_TASK_ID = "GSMULTI_TASK_ID"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class TaskProgress:
    phase: str = "queued"
    episode: int = 0
    max_episodes: int = 0
    mean_reward_last10: float = 0.0
    mean_peak_ny_max_g: float = 0.0
    message: str = ""


@dataclass
class AsyncTaskRecord:
    task_id: str
    status: str = "queued"
    created_at: str = field(default_factory=_utc_now)
    updated_at: str = field(default_factory=_utc_now)
    prompt_preview: str = ""
    progress: TaskProgress = field(default_factory=TaskProgress)
    result_summary: Dict[str, Any] = field(default_factory=dict)
    error: str = ""
    log_path: str = ""
    pid: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d


class AsyncTaskStore:
    """JSON file store under guidance_output/async_tasks/{task_id}.json"""

    DEFAULT_DIR = "./guidance_output/async_tasks"

    def __init__(self, base_dir: str = DEFAULT_DIR) -> None:
        self.base_dir = os.path.abspath(base_dir)
        os.makedirs(self.base_dir, exist_ok=True)

    def _path(self, task_id: str) -> str:
        return os.path.join(self.base_dir, f"{task_id}.json")

    def save(self, record: AsyncTaskRecord) -> None:
        record.updated_at = _utc_now()
        path = self._path(record.task_id)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(record.to_dict(), fh, ensure_ascii=False, indent=2)
        os.replace(tmp, path)

    def load(self, task_id: str) -> Optional[AsyncTaskRecord]:
        path = self._path(task_id)
        if not os.path.isfile(path):
            return None
        with open(path, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
        prog = raw.pop("progress", {}) or {}
        record = AsyncTaskRecord(
            task_id=raw.get("task_id", task_id),
            status=raw.get("status", "unknown"),
            created_at=raw.get("created_at", ""),
            updated_at=raw.get("updated_at", ""),
            prompt_preview=raw.get("prompt_preview", ""),
            progress=TaskProgress(**{k: prog.get(k, v) for k, v in asdict(TaskProgress()).items()}),
            result_summary=raw.get("result_summary") or {},
            error=raw.get("error", ""),
            log_path=raw.get("log_path", ""),
            pid=raw.get("pid"),
        )
        return record

    def list_tasks(self, limit: int = 20) -> List[AsyncTaskRecord]:
        files = sorted(
            Path(self.base_dir).glob("*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        out: List[AsyncTaskRecord] = []
        for p in files[:limit]:
            rec = self.load(p.stem)
            if rec:
                out.append(rec)
        return out


class AsyncTaskQueue:
    """Submit background CLI runs and track PPO progress via shared JSON store."""

    def __init__(self, store: Optional[AsyncTaskStore] = None) -> None:
        self.store = store or AsyncTaskStore()
        self._lock = threading.Lock()

    def create_task(self, prompt_preview: str = "") -> AsyncTaskRecord:
        task_id = uuid.uuid4().hex
        record = AsyncTaskRecord(
            task_id=task_id,
            status="queued",
            prompt_preview=(prompt_preview or "")[:200],
        )
        self.store.save(record)
        return record

    def get_status(self, task_id: str) -> Optional[AsyncTaskRecord]:
        return self.store.load(task_id)

    def update_progress(
        self,
        task_id: str,
        *,
        status: Optional[str] = None,
        phase: Optional[str] = None,
        episode: Optional[int] = None,
        max_episodes: Optional[int] = None,
        mean_reward_last10: Optional[float] = None,
        mean_peak_ny_max_g: Optional[float] = None,
        message: Optional[str] = None,
        result_summary: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
    ) -> None:
        with self._lock:
            record = self.store.load(task_id)
            if record is None:
                return
            if status is not None:
                record.status = status
            if phase is not None:
                record.progress.phase = phase
            if episode is not None:
                record.progress.episode = int(episode)
            if max_episodes is not None:
                record.progress.max_episodes = int(max_episodes)
            if mean_reward_last10 is not None:
                record.progress.mean_reward_last10 = float(mean_reward_last10)
            if mean_peak_ny_max_g is not None:
                record.progress.mean_peak_ny_max_g = float(mean_peak_ny_max_g)
            if message is not None:
                record.progress.message = message
            if result_summary is not None:
                record.result_summary = result_summary
            if error is not None:
                record.error = error
            self.store.save(record)

    def submit_cli_run(
        self,
        cli_argv: List[str],
        *,
        prompt_preview: str = "",
        env: Optional[Dict[str, str]] = None,
    ) -> AsyncTaskRecord:
        """Spawn detached subprocess running the same CLI with GSMULTI_TASK_ID set."""
        record = self.create_task(prompt_preview=prompt_preview)
        run_env = os.environ.copy()
        if env:
            run_env.update(env)
        run_env[ENV_TASK_ID] = record.task_id
        run_env["GSMULTI_ASYNC_CHILD"] = "1"

        creationflags = 0
        if sys.platform == "win32":
            creationflags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS

        proc = subprocess.Popen(
            cli_argv,
            env=run_env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=creationflags,
            close_fds=(sys.platform != "win32"),
        )
        record.status = "running"
        record.progress.phase = "cli_main"
        record.pid = proc.pid
        self.store.save(record)
        logger.info("[AsyncTask] submitted %s pid=%s", record.task_id, proc.pid)
        return record


# Module-level singleton for progress callbacks from RL optimizer
_global_queue: Optional[AsyncTaskQueue] = None


def get_task_queue() -> AsyncTaskQueue:
    global _global_queue
    if _global_queue is None:
        _global_queue = AsyncTaskQueue()
    return _global_queue


def current_async_task_id() -> Optional[str]:
    return os.environ.get(ENV_TASK_ID)


def make_progress_callback(task_id: str) -> Callable[..., None]:
    """Build callback for MatlabRLOptimizer.progress_callback."""

    def _cb(
        episode: int,
        max_episodes: int,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        extra = extra or {}
        get_task_queue().update_progress(
            task_id,
            status="running",
            phase="ppo",
            episode=episode,
            max_episodes=max_episodes,
            mean_reward_last10=float(extra.get("mean_reward_last10", 0.0)),
            mean_peak_ny_max_g=float(extra.get("mean_peak_ny_max_g", 0.0)),
            message=str(extra.get("message", "")),
        )

    return _cb
