"""Append-only JSONL logger for RL / local-search episodes."""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Dict, Optional


class EpisodeJsonlLogger:
    """Write one JSON object per line under ``logs/rl_episodes_<run_id>.jsonl``."""

    def __init__(self, log_dir: str = "logs", run_id: Optional[str] = None, path: Optional[str] = None):
        self.log_dir = log_dir
        self.run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = path or os.path.join(log_dir, f"rl_episodes_{self.run_id}.jsonl")
        self._fh = None
        self._opened = False

    def open(self, append_only: bool = False) -> str:
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        if append_only and os.path.isfile(self.path):
            self._fh = open(self.path, "a", encoding="utf-8")
            self._opened = True
            return self.path
        self._fh = open(self.path, "a", encoding="utf-8")
        if not self._opened:
            self.write_meta({"event": "run_start", "run_id": self.run_id})
            self._opened = True
        return self.path

    def write_meta(self, record: Dict[str, Any]) -> None:
        self._write(record)

    def write_episode(
        self,
        *,
        optimizer: str,
        episode: int,
        params: Dict[str, float],
        metrics: Dict[str, Any],
        reward: float,
        fitness: float,
        constrained: bool,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        rec: Dict[str, Any] = {
            "event": "episode",
            "optimizer": optimizer,
            "episode": episode,
            "params": params,
            "metrics": metrics,
            "reward": reward,
            "fitness": fitness,
            "constraints_met": constrained,
        }
        if extra:
            rec.update(extra)
        self._write(rec)

    def _write(self, record: Dict[str, Any]) -> None:
        if self._fh is None:
            self.open()
        record.setdefault("ts", datetime.now().isoformat(timespec="seconds"))
        self._fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._fh.flush()

    def write_rolling_stats(self, episode: int, stats: Dict[str, Any]) -> None:
        rec = {"event": "rolling_stats", "episode": episode, **stats}
        self._write(rec)

    def close(self) -> None:
        if self._fh is not None:
            self.write_meta({"event": "run_end", "run_id": self.run_id})
            self._fh.close()
            self._fh = None
