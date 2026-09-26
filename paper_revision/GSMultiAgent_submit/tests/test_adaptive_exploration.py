#!/usr/bin/env python3
"""Tests for adaptive exploration steepness controller."""

import unittest

from multi_agent.rl.adaptive_exploration import AdaptiveExplorationController


class TestAdaptiveExploration(unittest.TestCase):
    def test_steep_episode_shrinks_log_std(self):
        ctrl = AdaptiveExplorationController(
            steepness_threshold=2.0,
            log_std_min_scale=0.35,
            log_std_max_scale=1.0,
            ema_alpha=1.0,
        )
        p0 = {k: 40.0 for k in ("w1", "zeta1", "tao1", "w2", "zeta2", "tao2", "w3", "zeta3", "N_pn")}
        p1 = dict(p0)
        p1["w1"] = 41.0
        ctrl.update_episode(p0, p1, 0.5, 0.9, new_best=True)
        self.assertGreater(ctrl.state.steepness_ema, 2.0)
        self.assertEqual(ctrl.state.explore_mode, "fine")
        self.assertAlmostEqual(ctrl.state.log_std_scale, 0.35)

    def test_flat_episode_keeps_aggressive(self):
        ctrl = AdaptiveExplorationController(
            steepness_threshold=2.0,
            log_std_min_scale=0.35,
            log_std_max_scale=1.0,
            ema_alpha=1.0,
        )
        p0 = {k: 40.0 for k in ("w1", "zeta1", "tao1", "w2", "zeta2", "tao2", "w3", "zeta3", "N_pn")}
        p1 = dict(p0)
        p1["N_pn"] = 40.5
        ctrl.update_episode(p0, p1, 0.5, 0.51, new_best=False)
        self.assertLess(ctrl.state.steepness_ema, 0.7)
        self.assertEqual(ctrl.state.explore_mode, "aggressive")

    def test_neighborhood_probe_updates_ema(self):
        ctrl = AdaptiveExplorationController(steepness_threshold=2.0, ema_alpha=0.5)
        ctrl.update_episode({}, {}, 0.4, 0.4)
        ctrl.ingest_neighborhood_steepness(5.0, blend=1.0)
        self.assertEqual(ctrl.state.neighborhood_steepness, 5.0)
        self.assertEqual(ctrl.state.explore_mode, "fine")

    def test_explore_extra_keys(self):
        ctrl = AdaptiveExplorationController()
        extra = ctrl.explore_extra()
        self.assertIn("log_std_scale", extra)
        self.assertIn("explore_mode", extra)
        self.assertIn("steepness_ema", extra)


if __name__ == "__main__":
    unittest.main()
