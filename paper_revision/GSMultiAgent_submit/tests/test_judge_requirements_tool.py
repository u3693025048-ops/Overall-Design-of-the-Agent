#!/usr/bin/env python3
"""Tests for judge_requirements PeakNy mean-limit checks."""

import unittest

from multi_agent.tools.judge_requirements_tool import (
    JudgeRequirementsTool,
    _resolve_requirements,
    _rule_check,
)
from multi_agent.tools.simulation_tool import (
    _aggregate_mc_seed_metrics,
    _inject_matlab_rng_seed,
)


class _FakeRunSim:
    name = "run_simulation"

    def __init__(self, metrics):
        self.last_metrics = metrics


class TestJudgePeakNyLimit(unittest.TestCase):
    def test_mean_only_can_pass_peak_limit(self):
        metrics = {
            "hit_rate": 100.0,
            "SEP": 4.0,
            "peak_ny": 19.49,
            "pitch_PM": 61.0,
            "pitch_BW": 70.0,
        }
        reqs = {"peak_ny_max": 20.0, "hit_rate_min": 92.0, "sep_max": 7.0}
        ok, reasons = _rule_check(metrics, reqs)
        self.assertTrue(ok)
        self.assertTrue(any("PeakNy(平均值)" in r and "[OK]" in r for r in reasons))

    def test_max_over_limit_ignored_when_mean_ok(self):
        metrics = {
            "hit_rate": 100.0,
            "SEP": 4.0,
            "peak_ny": 19.49,
            "peak_ny_max": 25.4,
            "pitch_PM": 61.0,
            "pitch_BW": 70.0,
        }
        reqs = {
            "peak_ny_max": 20.0,
            "peak_ny_mean_max": 20.0,
            "hit_rate_min": 92.0,
            "sep_max": 7.0,
        }
        ok, reasons = _rule_check(metrics, reqs)
        self.assertTrue(ok)
        self.assertFalse(any("PeakNy(最大值)" in r for r in reasons))

    def test_peak_ny_pass_v20_cls(self):
        metrics = {
            "hit_rate": 100.0,
            "SEP": 4.98,
            "peak_ny": 17.86,
            "peak_ny_max": 20.35,
            "pitch_PM": 52.3,
            "pitch_BW": 76.5,
        }
        reqs = {
            "peak_ny_max": 20.0,
            "peak_ny_mean_max": 20.0,
            "hit_rate_min": 92.0,
            "sep_max": 7.0,
            "pm_min": 45.0,
            "pm_max": 70.0,
            "bw_min": 20.0,
            "bw_max": 85.0,
        }
        ok, reasons = _rule_check(metrics, reqs)
        self.assertTrue(ok)
        self.assertTrue(any("PeakNy(平均值)" in r and "[OK]" in r for r in reasons))

    def test_merge_trusted_sim_overrides_llm_handfill(self):
        judge = JudgeRequirementsTool()
        trusted = {
            "hit_rate": 100.0,
            "SEP": 4.0,
            "peak_ny": 19.0,
            "peak_ny_max": 24.0,
            "pitch_PM": 60.0,
            "pitch_BW": 50.0,
        }
        judge.set_run_simulation_tool(_FakeRunSim(trusted))
        llm_fill = {
            "hit_rate": 100.0,
            "SEP": 4.0,
            "peak_ny": 19.0,
            "peak_ny_max": 0.0,
            "pitch_PM": 60.0,
            "pitch_BW": 50.0,
        }
        merged, overridden = judge._merge_trusted_sim_metrics(llm_fill)
        self.assertTrue(overridden)
        self.assertAlmostEqual(merged["peak_ny_max"], 24.0)


class TestMultiSeedMcValidation(unittest.TestCase):
    def setUp(self):
        self.reqs = {
            "hit_rate_min": 92.0,
            "sep_max": 7.0,
            "peak_ny_max": 20.0,
            "peak_ny_mean_max": 20.0,
            "pm_min": 45.0,
            "pm_max": 70.0,
            "bw_min": 20.0,
            "bw_max": 85.0,
        }

    def _seed_metric(self, seed, sep=5.0, hit=100.0, peak=18.0, pm=55.0, bw=60.0):
        return {
            "mc_seed": seed,
            "hit_rate": hit,
            "SEP": sep,
            "peak_ny": peak,
            "pitch_PM": pm,
            "pitch_BW": bw,
        }

    def test_all_seed_metrics_must_pass(self):
        metrics = {
            "mc_seed_results": [
                self._seed_metric(42),
                self._seed_metric(2025, sep=8.0),
                self._seed_metric(3407),
            ]
        }
        ok, reasons = _rule_check(metrics, self.reqs)
        self.assertFalse(ok)
        self.assertTrue(any("MC seed 2025: FAIL" in r for r in reasons))

    def test_all_seed_metrics_pass(self):
        metrics = {
            "mc_seed_results": [
                self._seed_metric(42),
                self._seed_metric(2025, sep=6.5),
                self._seed_metric(3407, peak=19.5),
            ]
        }
        ok, _ = _rule_check(metrics, self.reqs)
        self.assertTrue(ok)

    def test_aggregate_uses_worst_seed_values(self):
        agg = _aggregate_mc_seed_metrics([
            self._seed_metric(42, sep=4.0, hit=100.0, peak=17.0, pm=50.0, bw=40.0),
            self._seed_metric(2025, sep=6.0, hit=96.0, peak=19.0, pm=48.0, bw=82.0),
            self._seed_metric(3407, sep=5.0, hit=98.0, peak=18.0, pm=69.0, bw=25.0),
        ])
        self.assertEqual(agg["hit_rate"], 96.0)
        self.assertEqual(agg["SEP"], 6.0)
        self.assertEqual(agg["peak_ny"], 19.0)
        self.assertEqual(agg["pitch_PM_min"], 48.0)
        self.assertEqual(agg["pitch_PM_max"], 69.0)
        self.assertEqual(agg["pitch_BW_min"], 25.0)
        self.assertEqual(agg["pitch_BW_max"], 82.0)

    def test_rng_seed_injection_replaces_existing_rng(self):
        content = "clc; close all; rng(42);\nRUN_CASE='T';\n"
        patched = _inject_matlab_rng_seed(content, 2025)
        self.assertIn("rng(2025, 'twister');", patched)
        self.assertNotIn("rng(42);", patched)


if __name__ == "__main__":
    unittest.main()
