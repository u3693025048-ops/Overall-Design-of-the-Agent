#!/usr/bin/env python3
"""Tests for run wall-clock budget."""

import unittest

from multi_agent.simulation.time_budget import RunTimeBudget, init_run_budget, clear_run_budget


class TestRunTimeBudget(unittest.TestCase):
    def tearDown(self):
        clear_run_budget()

    def test_allowed_cls_steps_scales_with_remaining(self):
        b = RunTimeBudget.from_seconds(600)
        self.assertGreaterEqual(b.allowed_cls_steps(25), 5)
        self.assertLessEqual(b.allowed_cls_steps(25), 25)

    def test_allowed_ppo_zero_when_low_time(self):
        b = RunTimeBudget.from_seconds(300)
        self.assertEqual(b.allowed_ppo_episodes(30), 0)

    def test_init_global_budget(self):
        b = init_run_budget(1800)
        self.assertTrue(b.enabled)
        self.assertGreater(b.remaining_sec() or 0, 1700)


if __name__ == "__main__":
    unittest.main()
