"""Rolling-window statistics for RL episode streams."""

from __future__ import annotations

import logging
from collections import deque
from typing import Any, Deque, Dict, List, Optional

from multi_agent.rl.metric_utils import get_peak_ny, get_sep

logger = logging.getLogger(__name__)


class RollingEpisodeStats:
    """Track sliding mean / diverge rate over the last *window* episodes."""

    def __init__(self, window: int = 20):
        self.window = max(1, int(window))
        self._rewards: Deque[float] = deque(maxlen=self.window)
        self._seps: Deque[float] = deque(maxlen=self.window)
        self._peak_ny_mean: Deque[float] = deque(maxlen=self.window)
        self._hit_rates: Deque[float] = deque(maxlen=self.window)
        self._diverge_flags: Deque[bool] = deque(maxlen=self.window)
        self._total_episodes = 0

    def update(
        self,
        *,
        reward: float,
        metrics: Dict[str, float],
        diverge_reward_threshold: float = -9.5,
    ) -> Dict[str, float]:
        self._total_episodes += 1
        sep = get_sep(metrics, float("nan"))
        pny_avg = get_peak_ny(metrics)
        hit = float(metrics.get("hit_rate", 0.0))

        self._rewards.append(float(reward))
        if sep == sep:  # not NaN
            self._seps.append(float(sep))
        if pny_avg > 0:
            self._peak_ny_mean.append(float(pny_avg))
        self._hit_rates.append(hit)
        self._diverge_flags.append(reward <= diverge_reward_threshold)

        return self.snapshot()

    def snapshot(self) -> Dict[str, float]:
        def _mean(vals: Deque[float]) -> float:
            return float(sum(vals) / len(vals)) if vals else float("nan")

        diverge_pct = (
            100.0 * sum(1 for f in self._diverge_flags if f) / len(self._diverge_flags)
            if self._diverge_flags
            else 0.0
        )
        mean_pny = _mean(self._peak_ny_mean)
        return {
            "window": float(self.window),
            "count_in_window": float(len(self._rewards)),
            "total_episodes": float(self._total_episodes),
            "mean_reward": _mean(self._rewards),
            "mean_sep_m": _mean(self._seps),
            "mean_peak_ny_g": mean_pny,
            "mean_peak_ny_max_g": mean_pny,  # backward-compat alias
            "mean_hit_rate_pct": _mean(self._hit_rates),
            "diverge_pct": diverge_pct,
        }

    def log_summary(self, episode: int, max_episodes: int) -> None:
        s = self.snapshot()
        from multi_agent.logging.log_verbosity import is_verbose, should_log_rl_episode

        msg = (
            f"  [RollingStats @ Ep {episode}/{max_episodes}] "
            f"window={int(s['window'])} | "
            f"mean_reward={s['mean_reward']:.3f} | "
            f"mean_SEP={s['mean_sep_m']:.2f}m | "
            f"mean_PeakNy_avg={s['mean_peak_ny_g']:.2f}g | "
            f"mean_hit={s['mean_hit_rate_pct']:.1f}% | "
            f"diverge={s['diverge_pct']:.0f}%"
        )
        ep0 = max(0, int(episode) - 1)
        if is_verbose() or should_log_rl_episode(ep0, max_episodes):
            logger.info(msg)
        else:
            logger.debug(msg)

    def to_record(self, episode: int) -> Dict[str, Any]:
        out = self.snapshot()
        out["episode"] = float(episode)
        return out


def analyze_jsonl_episodes(
    path: str,
    window: int = 20,
    diverge_reward_threshold: float = -9.5,
) -> List[Dict[str, Any]]:
    """Read JSONL and emit per-episode rolling snapshots (for offline reports)."""
    import json

    roller = RollingEpisodeStats(window=window)
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("event") != "episode":
                continue
            metrics = rec.get("metrics") or {}
            snap = roller.update(
                reward=float(rec.get("reward", 0.0)),
                metrics=metrics,
                diverge_reward_threshold=diverge_reward_threshold,
            )
            rows.append(
                {
                    "episode": rec.get("episode"),
                    "optimizer": rec.get("optimizer"),
                    "reward": rec.get("reward"),
                    **snap,
                }
            )
    return rows
