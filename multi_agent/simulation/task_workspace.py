"""
Per-task isolated workspace for MATLAB simulation scripts.

Each optimization iteration (or concurrent job) gets a UUID sandbox directory
under ``guidance_output/tasks/{uuid}/`` so temp ``rltmp_*.m`` files and
compiled caches never collide across workers.
"""

from __future__ import annotations

import logging
import os
import shutil
import uuid
from pathlib import Path
from typing import Optional, Union

logger = logging.getLogger(__name__)

PathLike = Union[str, os.PathLike]


class TaskWorkspace:
    """UUID sandbox for one simulation / RL task."""

    DEFAULT_BASE = "./guidance_output/tasks"

    def __init__(
        self,
        base_dir: PathLike = DEFAULT_BASE,
        task_id: Optional[str] = None,
        *,
        auto_cleanup: bool = True,
    ) -> None:
        self.base_dir = os.path.abspath(str(base_dir))
        self.task_id = task_id or uuid.uuid4().hex
        self.path = os.path.join(self.base_dir, self.task_id)
        self.auto_cleanup = auto_cleanup
        self._prepared_scripts: list[str] = []

    def __enter__(self) -> "TaskWorkspace":
        os.makedirs(self.path, exist_ok=True)
        logger.debug("[TaskWorkspace] created %s", self.path)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.auto_cleanup:
            self.cleanup()

    def prepare_script(self, src_path: PathLike) -> str:
        """
        Copy *src_path* into this workspace and return the isolated absolute path.

        If *src_path* is already inside this workspace, returns it unchanged.
        """
        src = os.path.abspath(str(src_path))
        if not os.path.isfile(src):
            raise FileNotFoundError(f"Script not found: {src}")

        if os.path.commonpath([src, self.path]) == self.path:
            return src

        dest = os.path.join(self.path, os.path.basename(src))
        shutil.copy2(src, dest)
        self._prepared_scripts.append(dest)
        logger.debug("[TaskWorkspace] prepared script %s → %s", src, dest)
        return dest

    def cleanup(self) -> None:
        """Remove the entire sandbox directory (best-effort)."""
        if os.path.isdir(self.path):
            try:
                shutil.rmtree(self.path, ignore_errors=True)
                logger.debug("[TaskWorkspace] cleaned up %s", self.path)
            except Exception as exc:
                logger.warning("[TaskWorkspace] cleanup failed for %s: %s", self.path, exc)

    async def verify_physics_bounds(
        self,
        script_path: PathLike,
        *,
        ny_limit_g: float = 20.0,
        engine: str = "matlab",
        octave_path: str = "octave",
        matlab_path: str = "matlab",
        timeout_sec: float = 120.0,
    ) -> tuple:
        """
        Run gf() extreme-stub probe on an isolated copy of *script_path*.

        Returns (ok, message).
        """
        from multi_agent.simulation.physics_bounds import verify_physics_bounds as _verify

        isolated = self.prepare_script(script_path)
        return await _verify(
            isolated,
            ny_limit_g=ny_limit_g,
            work_dir=self.path,
            engine=engine,
            octave_path=octave_path,
            matlab_path=matlab_path,
            timeout_sec=timeout_sec,
        )

    @classmethod
    def sweep_stale(cls, base_dir: PathLike = DEFAULT_BASE, max_age_hours: float = 24.0) -> int:
        """Delete task workspaces older than *max_age_hours*. Returns count removed."""
        import time

        base = Path(base_dir)
        if not base.is_dir():
            return 0
        cutoff = time.time() - max_age_hours * 3600.0
        removed = 0
        for child in base.iterdir():
            if not child.is_dir():
                continue
            try:
                if child.stat().st_mtime < cutoff:
                    shutil.rmtree(child, ignore_errors=True)
                    removed += 1
            except OSError:
                pass
        return removed
