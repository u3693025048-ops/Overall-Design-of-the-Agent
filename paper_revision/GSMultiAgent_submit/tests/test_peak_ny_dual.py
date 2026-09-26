#!/usr/bin/env python3
"""Tests for PeakNy mean-limit requirement helpers."""

import unittest

from multi_agent.rl.metric_utils import (
    check_peak_ny,
    default_peak_ny_limit,
    normalize_peak_ny_requirements,
    peak_ny_satisfied,
)


class TestPeakNyLimit(unittest.TestCase):
    def test_normalize_single_limit(self):
        reqs = normalize_peak_ny_requirements({"peak_ny_max": 20.0})
        self.assertEqual(reqs["peak_ny_mean_max"], 20.0)
        self.assertEqual(reqs["peak_ny_max"], 20.0)
        self.assertNotIn("peak_ny_hard_max", reqs)

    def test_strips_legacy_hard_max_from_reqs(self):
        reqs = normalize_peak_ny_requirements(
            {"peak_ny_max": 20.0, "peak_ny_hard_max": 21.0}
        )
        self.assertNotIn("peak_ny_hard_max", reqs)

    def test_satisfied_mean_under_limit(self):
        metrics = {"peak_ny": 18.88, "peak_ny_max": 20.87}
        self.assertTrue(peak_ny_satisfied(metrics, {"peak_ny_max": 20.0}))

    def test_satisfied_v20_cls7(self):
        metrics = {"peak_ny": 17.86, "peak_ny_max": 20.35}
        self.assertTrue(peak_ny_satisfied(metrics, {"peak_ny_max": 20.0}))

    def test_fails_when_mean_over_limit(self):
        metrics = {"peak_ny": 20.91, "peak_ny_max": 20.35}
        ok, reasons = check_peak_ny(metrics, {"peak_ny_max": 20.0})
        self.assertFalse(ok)
        self.assertTrue(any("平均值" in r and "[NG]" in r for r in reasons))

    def test_max_over_limit_ignored_when_mean_ok(self):
        metrics = {"peak_ny": 17.0, "peak_ny_max": 22.29}
        self.assertTrue(peak_ny_satisfied(metrics, {"peak_ny_max": 20.0}))

    def test_max_over_21_ignored_when_mean_ok(self):
        metrics = {"peak_ny": 18.88, "peak_ny_max": 21.5}
        self.assertTrue(peak_ny_satisfied(metrics, {"peak_ny_max": 20.0}))

    def test_default_limit_from_config(self):
        self.assertEqual(default_peak_ny_limit(), 20.0)


if __name__ == "__main__":
    unittest.main()
