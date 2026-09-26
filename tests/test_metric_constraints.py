#!/usr/bin/env python3
"""Tests for multi-criteria constraint selection."""

import unittest

from multi_agent.rl.metric_constraints import (
    constrained_rank_key,
    constraints_satisfied,
    eligible_partial_bw_band,
    is_better_constrained,
    is_better_partial_progress,
    metric_satisfaction_flags,
    should_adopt_cls_result,
    violates_locked_metrics,
)
from multi_agent.rl.metric_utils import get_peak_ny_max


class TestMetricConstraints(unittest.TestCase):
    def test_constraints_satisfied_all_ok(self):
        metrics = {
            "hit_rate": 100.0,
            "SEP": 4.5,
            "peak_ny": 18.0,
            "peak_ny_max": 19.0,
            "pitch_PM": 60.0,
            "pitch_BW": 40.0,
        }
        reqs = {
            "hit_rate_min": 92.0,
            "sep_max": 7.0,
            "peak_ny_max": 20.0,
            "pm_min": 45.0,
            "pm_max": 70.0,
            "bw_min": 20.0,
            "bw_max": 85.0,
        }
        self.assertTrue(constraints_satisfied(metrics, reqs))

    def test_constraints_pass_when_max_over_but_mean_ok(self):
        metrics = {
            "hit_rate": 100.0,
            "SEP": 4.5,
            "peak_ny": 17.0,
            "peak_ny_max": 25.0,
            "pitch_PM": 60.0,
            "pitch_BW": 40.0,
        }
        reqs = {"hit_rate_min": 92.0, "sep_max": 7.0, "peak_ny_max": 20.0}
        self.assertTrue(constraints_satisfied(metrics, reqs))

    def test_peak_ny_pass_mean_only(self):
        """v20 CLS-7 style: mean<=20 satisfies even if max>21."""
        metrics = {
            "hit_rate": 100.0,
            "SEP": 4.98,
            "peak_ny": 17.86,
            "peak_ny_max": 20.35,
            "pitch_PM": 52.3,
            "pitch_BW": 76.5,
        }
        reqs = {
            "hit_rate_min": 92.0,
            "sep_max": 7.0,
            "peak_ny_max": 20.0,
            "peak_ny_mean_max": 20.0,
            "pm_min": 45.0,
            "pm_max": 70.0,
            "bw_min": 20.0,
            "bw_max": 85.0,
        }
        self.assertTrue(constraints_satisfied(metrics, reqs))

    def test_get_peak_ny_max_prefers_max_key(self):
        m = {"peak_ny": 17.0, "peak_ny_max": 27.37}
        self.assertAlmostEqual(get_peak_ny_max(m), 27.37)

    def test_is_better_constrained_prefers_lower_peak_ny(self):
        a = {"peak_ny_max": 18.0, "peak_ny": 18.0, "SEP": 5.0, "miss_distance": 5.0}
        b = {"peak_ny_max": 24.0, "peak_ny": 24.0, "SEP": 4.0, "miss_distance": 4.0}
        self.assertTrue(is_better_constrained(a, 0.8, b, 0.9))

    def test_constrained_rank_key_order(self):
        low = constrained_rank_key({"peak_ny_max": 15.0, "peak_ny": 15.0, "SEP": 5.0}, 0.5)
        high = constrained_rank_key({"peak_ny_max": 30.0, "peak_ny": 30.0, "SEP": 4.0}, 0.9)
        self.assertLess(low, high)

    def test_should_adopt_cls_iter1_mean_beats_max(self):
        """CLS SEP/mean win even when peak_ny_max is slightly higher than PPO."""
        prior = {
            "hit_rate": 96.0,
            "SEP": 11.32,
            "peak_ny": 21.55,
            "peak_ny_max": 25.44,
            "pitch_PM": 69.2,
            "pitch_BW": 88.62,
        }
        cls = {
            "hit_rate": 100.0,
            "SEP": 4.98,
            "peak_ny": 20.65,
            "peak_ny_max": 26.30,
            "pitch_PM": 54.7,
            "pitch_BW": 84.2,
        }
        reqs = {
            "hit_rate_min": 92.0,
            "sep_max": 7.0,
            "peak_ny_max": 20.0,
            "pm_min": 45.0,
            "pm_max": 70.0,
            "bw_min": 20.0,
            "bw_max": 85.0,
        }
        self.assertTrue(
            should_adopt_cls_result(prior, 0.5, cls, 0.6, reqs=reqs, task_prompt="PeakNy")
        )

    def test_partial_bw_band_rejects_low_basin(self):
        reqs = {"bw_min": 20.0, "bw_max": 85.0}
        low = {"pitch_BW": 5.8}
        flags = metric_satisfaction_flags(
            {
                "hit_rate": 100.0,
                "SEP": 4.6,
                "peak_ny": 19.9,
                "pitch_PM": 81.6,
                "pitch_BW": 5.8,
            },
            {
                **reqs,
                "hit_rate_min": 92.0,
                "sep_max": 7.0,
                "peak_ny_max": 20.0,
                "pm_min": 45.0,
                "pm_max": 70.0,
            },
        )
        self.assertFalse(eligible_partial_bw_band(low, flags, reqs))

    def test_partial_bw_band_allows_four_of_five_high_bw(self):
        reqs = {
            "hit_rate_min": 92.0,
            "sep_max": 7.0,
            "peak_ny_max": 20.0,
            "pm_min": 45.0,
            "pm_max": 70.0,
            "bw_min": 20.0,
            "bw_max": 85.0,
        }
        metrics = {
            "hit_rate": 100.0,
            "SEP": 4.44,
            "peak_ny": 19.85,
            "pitch_PM": 69.7,
            "pitch_BW": 90.8,
        }
        flags = metric_satisfaction_flags(metrics, reqs)
        self.assertFalse(flags["bw"])
        self.assertTrue(eligible_partial_bw_band(metrics, flags, reqs))

    def test_partial_bw_band_allows_in_band(self):
        reqs = {"bw_min": 20.0, "bw_max": 85.0}
        metrics = {"pitch_BW": 50.0}
        flags = {"bw": True}
        self.assertTrue(eligible_partial_bw_band(metrics, flags, reqs))

    def test_metric_flags_and_no_regression(self):
        reqs = {
            "hit_rate_min": 92.0,
            "sep_max": 7.0,
            "peak_ny_max": 20.0,
            "pm_min": 45.0,
            "pm_max": 70.0,
            "bw_min": 20.0,
            "bw_max": 85.0,
        }
        good = {
            "hit_rate": 100.0,
            "SEP": 6.0,
            "peak_ny": 19.0,
            "pitch_PM": 55.0,
            "pitch_BW": 40.0,
        }
        flags = metric_satisfaction_flags(good, reqs)
        self.assertTrue(all(flags.values()))
        bad_hit = dict(good, hit_rate=80.0)
        self.assertTrue(
            violates_locked_metrics(
                metric_satisfaction_flags(bad_hit, reqs),
                frozenset({"hit", "sep", "peak"}),
            )
        )
        self.assertTrue(
            is_better_partial_progress(
                good,
                flags,
                0.9,
                bad_hit,
                metric_satisfaction_flags(bad_hit, reqs),
                0.95,
            )
        )


class TestReliableSimMetrics(unittest.TestCase):
    def test_fallback_prior_yields_cls_adoption(self):
        from multi_agent.rl.matlab_rl_optimizer import is_reliable_sim_metrics

        prior = {
            "hit_rate": 96.0,
            "SEP": 4.5,
            "peak_ny": 7.48,
            "pitch_PM": 60.0,
            "pitch_BW": 40.0,
            "_sim_source": "fallback",
        }
        cls = {
            "hit_rate": 100.0,
            "SEP": 4.98,
            "peak_ny": 20.65,
            "pitch_PM": 54.7,
            "pitch_BW": 84.2,
            "_sim_source": "matlab",
        }
        reqs = {
            "hit_rate_min": 92.0,
            "sep_max": 7.0,
            "peak_ny_max": 20.0,
            "pm_min": 45.0,
            "pm_max": 70.0,
            "bw_min": 20.0,
            "bw_max": 85.0,
        }
        self.assertFalse(is_reliable_sim_metrics(prior))
        self.assertTrue(is_reliable_sim_metrics(cls))
        self.assertTrue(
            should_adopt_cls_result(prior, 0.9, cls, 0.6, reqs=reqs, task_prompt="PeakNy")
        )

    def test_unreliable_cls_not_adopted(self):
        from multi_agent.rl.matlab_rl_optimizer import is_reliable_sim_metrics

        prior = {
            "hit_rate": 96.0,
            "SEP": 4.5,
            "peak_ny": 18.0,
            "pitch_PM": 60.0,
            "pitch_BW": 40.0,
            "_sim_source": "matlab",
        }
        cls = {"_sim_source": "unavailable", "_parse_incomplete": True}
        self.assertFalse(is_reliable_sim_metrics(cls))
        self.assertFalse(should_adopt_cls_result(prior, 0.5, cls, 0.6))


if __name__ == "__main__":
    unittest.main()
