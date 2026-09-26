"""Wall-clock budget for full cli_agent runs (30-minute SLA)."""

from __future__ import annotations

import time
from typing import Optional

_budget: Optional["RunTimeBudget"] = None


class RunTimeBudget:
    """Monotonic deadline shared across Layer2/3."""

    def __init__(self, deadline_mono: Optional[float], *, label: str = "run"):
        self._deadline = deadline_mono
        self.label = label
        self._started = time.monotonic()

    @classmethod
    def from_seconds(cls, seconds: Optional[float], *, label: str = "run") -> "RunTimeBudget":
        if seconds is None or float(seconds) <= 0:
            return cls(None, label=label)
        return cls(time.monotonic() + float(seconds), label=label)

    @property
    def enabled(self) -> bool:
        return self._deadline is not None

    def remaining_sec(self) -> Optional[float]:
        if self._deadline is None:
            return None
        return max(0.0, self._deadline - time.monotonic())

    def elapsed_sec(self) -> float:
        return time.monotonic() - self._started

    def expired(self) -> bool:
        rem = self.remaining_sec()
        return rem is not None and rem <= 0.0

    def should_stop(self, reserve_sec: float = 90.0) -> bool:
        """True when less than *reserve_sec* remains (save time for report)."""
        rem = self.remaining_sec()
        if rem is None:
            return False
        return rem <= max(0.0, float(reserve_sec))

    def allowed_expert_rounds(self, default: int) -> int:
        rem = self.remaining_sec()
        if rem is None:
            return max(1, int(default))
        if rem < 360:
            return 1
        if rem < 540:
            return min(int(default), 2)
        if rem < 900:
            return min(int(default), 2)
        return max(1, int(default))

    def allowed_cls_steps(self, default: int, *, sec_per_step: float = 32.0) -> int:
        rem = self.remaining_sec()
        if rem is None:
            return max(1, int(default))
        usable = rem - 90.0
        if usable <= 0:
            return 1
        est = int(usable / max(8.0, sec_per_step))
        return max(8, min(int(default), est))

    def allowed_ppo_episodes(self, default: int) -> int:
        rem = self.remaining_sec()
        if rem is None:
            return max(1, int(default))
        if rem < 600:
            return 0
        est = int((rem - 180.0) / 45.0)
        return max(0, min(int(default), est))


def init_run_budget(seconds: Optional[float], *, label: str = "run") -> RunTimeBudget:
    global _budget
    _budget = RunTimeBudget.from_seconds(seconds, label=label)
    return _budget


def get_run_budget() -> Optional[RunTimeBudget]:
    return _budget


def clear_run_budget() -> None:
    global _budget
    _budget = None
