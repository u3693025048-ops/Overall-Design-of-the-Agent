"""Tests for multi_agent.logging.log_verbosity."""

import argparse
import unittest
from types import SimpleNamespace

from multi_agent.integration.design_path_policy import metrics_summary_line
from multi_agent.logging.log_verbosity import (
    get_verbosity,
    init_verbosity_from_config,
    set_verbosity,
    should_log_cls_step,
    should_log_expert_round,
    should_log_rl_episode,
)


class TestLogVerbosity(unittest.TestCase):
    def tearDown(self):
        import logging

        set_verbosity("normal")
        logging.getLogger().setLevel(logging.WARNING)
        for name in (
            "multi_agent.tools.simulation_tool",
            "multi_agent.integration",
            "multi_agent.simulation.optimization_workflow",
            "multi_agent.rl.matlab_rl_optimizer",
            "multi_agent.rl.constraint_local_search",
            "multi_agent.rl.rolling_stats",
            "multi_agent.integration.script_seed_policy",
        ):
            logging.getLogger(name).setLevel(logging.NOTSET)

    def test_cli_verbose_overrides_config(self):
        cfg = SimpleNamespace(logging=SimpleNamespace(verbosity="normal", rl_episode_interval=10))
        args = argparse.Namespace(verbose=True, quiet=False)
        level = init_verbosity_from_config(cfg, args)
        self.assertEqual(level, "verbose")
        self.assertEqual(get_verbosity(), "verbose")

    def test_cli_quiet_overrides_config(self):
        cfg = SimpleNamespace(logging=SimpleNamespace(verbosity="verbose", rl_episode_interval=10))
        args = argparse.Namespace(verbose=False, quiet=True)
        level = init_verbosity_from_config(cfg, args)
        self.assertEqual(level, "quiet")
        self.assertEqual(get_verbosity(), "quiet")

    def test_config_fallback_when_no_cli_flags(self):
        cfg = SimpleNamespace(logging=SimpleNamespace(verbosity="quiet", rl_episode_interval=10))
        args = argparse.Namespace(verbose=False, quiet=False)
        level = init_verbosity_from_config(cfg, args)
        self.assertEqual(level, "quiet")

    def test_rl_episode_interval_throttles(self):
        set_verbosity("normal")
        self.assertTrue(should_log_rl_episode(0, 50))
        self.assertFalse(should_log_rl_episode(1, 50))
        self.assertTrue(should_log_rl_episode(9, 50))
        self.assertTrue(should_log_rl_episode(49, 50, new_best=True))

    def test_expert_round_last_and_improved(self):
        set_verbosity("normal")
        self.assertFalse(should_log_expert_round(1, 5))
        self.assertTrue(should_log_expert_round(5, 5))
        self.assertTrue(should_log_expert_round(2, 5, improved=True))

    def test_cls_step_last_and_improved(self):
        set_verbosity("normal")
        self.assertFalse(should_log_cls_step(1, 50))
        self.assertTrue(should_log_cls_step(50, 50))
        self.assertTrue(should_log_cls_step(3, 50, improved=True))

    def test_metrics_summary_line_uses_peak_ny_mean_only(self):
        line = metrics_summary_line({
            "hit_rate": 100.0,
            "SEP": 5.5,
            "peak_ny": 17.2,
            "peak_ny_max": 27.4,
            "pitch_PM": 51.0,
            "pitch_BW": 80.0,
        })
        self.assertIn("PeakNy_avg=17.20g", line)
        self.assertNotIn("PeakNy_max", line)


if __name__ == "__main__":
    unittest.main()
