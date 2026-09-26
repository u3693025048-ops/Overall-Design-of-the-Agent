#!/usr/bin/env python3
"""Unit tests for Layer 2 script gate helpers."""

import unittest

from multi_agent.integration.layer2_script_gate import (
    metrics_usable,
    stdout_smoke_ok,
)


class TestLayer2ScriptGate(unittest.TestCase):
    def test_metrics_usable_hit_rate(self):
        self.assertTrue(metrics_usable({"hit_rate": 55.0}))

    def test_metrics_usable_sep(self):
        self.assertTrue(metrics_usable({"SEP": 6.3}))

    def test_metrics_usable_rejects_sentinel_defaults(self):
        raw = """
  [Run] Hit  Miss(m)  PeakNy(g)  PM(°)   BW(rad/s)  GM(dB)
  Mean pitch PM: 30.5 deg, BW: 21.12 rad/s, GM: 6.09 dB
Case        Hit %   MeanMiss MeanNy  MeanPM MeanBW MeanGM
"""
        from multi_agent.rl.matlab_rl_optimizer import parse_sim_stdout

        m = parse_sim_stdout(raw)
        self.assertFalse(metrics_usable(m, raw))

    def test_metrics_usable_empty(self):
        self.assertFalse(metrics_usable({}))
        self.assertFalse(metrics_usable(None))

    def test_stdout_smoke_ok_summary(self):
        raw = "命中率(miss<10m): 55%  SEP: 6.30m\n"
        ok, _ = stdout_smoke_ok(raw, {})
        self.assertTrue(ok)

    def test_stdout_smoke_all_err(self):
        raw = "[ 1] ERR: endfunction\n[ 2] ERR: endfunction\n"
        ok, reason = stdout_smoke_ok(raw, {"hit_rate": 0.0})
        self.assertFalse(ok)
        self.assertIn("ERR", reason)

    def test_stdout_smoke_rejects_sentinel_only(self):
        raw = """
  Mean pitch PM: 30.5 deg, BW: 21.12 rad/s, GM: 6.09 dB
Case        Hit %   MeanMiss MeanNy  MeanPM MeanBW MeanGM
"""
        from multi_agent.rl.matlab_rl_optimizer import parse_sim_stdout

        m = parse_sim_stdout(raw)
        ok, reason = stdout_smoke_ok(raw, m)
        self.assertFalse(ok)
        self.assertIn("占位", reason)


if __name__ == "__main__":
    unittest.main()
