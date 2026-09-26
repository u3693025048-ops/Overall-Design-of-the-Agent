#!/usr/bin/env python3
"""Unit tests for rolling-window RL episode statistics."""

import json
import os
import tempfile
import unittest

from multi_agent.rl.rolling_stats import RollingEpisodeStats, analyze_jsonl_episodes


def _episode_line(ep: int, reward: float, sep: float, peak_ny: float, hit: float) -> str:
    return json.dumps({
        "event": "episode",
        "optimizer": "PPO",
        "episode": ep,
        "reward": reward,
        "metrics": {
            "SEP": sep,
            "peak_ny": peak_ny,
            "hit_rate": hit,
        },
    })


class TestRollingEpisodeStats(unittest.TestCase):
    def test_empty_snapshot(self):
        roller = RollingEpisodeStats(window=5)
        snap = roller.snapshot()
        self.assertEqual(snap["count_in_window"], 0.0)
        self.assertEqual(snap["diverge_pct"], 0.0)

    def test_update_mean_and_diverge(self):
        roller = RollingEpisodeStats(window=3)
        roller.update(
            reward=1.0,
            metrics={"SEP": 4.0, "peak_ny": 15.0, "hit_rate": 90.0},
            diverge_reward_threshold=-9.5,
        )
        roller.update(
            reward=-10.0,
            metrics={"SEP": 8.0, "peak_ny": 22.0, "hit_rate": 70.0},
            diverge_reward_threshold=-9.5,
        )
        snap = roller.snapshot()
        self.assertEqual(snap["count_in_window"], 2.0)
        self.assertAlmostEqual(snap["mean_reward"], -4.5)
        self.assertAlmostEqual(snap["mean_sep_m"], 6.0)
        self.assertAlmostEqual(snap["mean_peak_ny_g"], 18.5)
        self.assertAlmostEqual(snap["mean_peak_ny_max_g"], 18.5)
        self.assertAlmostEqual(snap["mean_hit_rate_pct"], 80.0)
        self.assertAlmostEqual(snap["diverge_pct"], 50.0)

    def test_window_rollover(self):
        roller = RollingEpisodeStats(window=2)
        for i in range(4):
            roller.update(
                reward=float(i),
                metrics={"SEP": float(i), "peak_ny": 10.0, "hit_rate": 50.0},
            )
        snap = roller.snapshot()
        self.assertEqual(snap["count_in_window"], 2.0)
        self.assertEqual(snap["total_episodes"], 4.0)
        self.assertAlmostEqual(snap["mean_reward"], 2.5)  # ep 2 and 3

    def test_to_record_includes_episode(self):
        roller = RollingEpisodeStats(window=5)
        roller.update(reward=0.5, metrics={"SEP": 3.0, "hit_rate": 85.0})
        rec = roller.to_record(episode=7)
        self.assertEqual(rec["episode"], 7.0)
        self.assertIn("mean_reward", rec)

    def test_log_summary_does_not_raise(self):
        import logging

        roller = RollingEpisodeStats(window=2)
        roller.update(reward=0.0, metrics={"SEP": 5.0, "hit_rate": 80.0})
        with self.assertLogs(logging.getLogger("multi_agent.rl.rolling_stats"), level="INFO") as cm:
            roller.log_summary(1, 50)
        self.assertTrue(any("[RollingStats @ Ep 1/50]" in m for m in cm.output))


class TestAnalyzeJsonlEpisodes(unittest.TestCase):
    def test_reads_episode_lines(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
        ) as fh:
            fh.write(_episode_line(1, 1.0, 5.0, 12.0, 88.0) + "\n")
            fh.write(_episode_line(2, -10.0, 9.0, 21.0, 72.0) + "\n")
            fh.write(json.dumps({"event": "rolling_stats", "episode": 2}) + "\n")
            path = fh.name
        try:
            rows = analyze_jsonl_episodes(path, window=2)
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["episode"], 1)
            self.assertAlmostEqual(rows[1]["mean_reward"], -4.5)
            self.assertAlmostEqual(rows[1]["diverge_pct"], 50.0)
        finally:
            os.unlink(path)

    def test_empty_file_returns_empty(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
        ) as fh:
            fh.write(json.dumps({"event": "run_start"}) + "\n")
            path = fh.name
        try:
            self.assertEqual(analyze_jsonl_episodes(path), [])
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main()
