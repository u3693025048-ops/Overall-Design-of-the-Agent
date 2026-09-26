#!/usr/bin/env python3
"""Unit tests for parameter firewall and task workspace."""

import os
import tempfile
import unittest

from multi_agent.security.parameter_firewall import (
    ParameterFirewallViolation,
    apply_rl_hyperparams_to_optimizer,
    clamp_tunable_params,
    sanitize_rl_hyperparams,
)
from multi_agent.simulation.task_workspace import TaskWorkspace


class TestClampTunableParams(unittest.TestCase):
    def test_clamps_out_of_range(self):
        out = clamp_tunable_params({"w1": 999.0, "N_pn": -1.0})
        self.assertEqual(out["w1"], 60.0)
        self.assertEqual(out["N_pn"], 3.0)

    def test_fills_missing_with_nominal(self):
        out = clamp_tunable_params({"w1": 40.0})
        self.assertEqual(out["w1"], 40.0)
        self.assertIn("zeta1", out)
        self.assertEqual(len(out), 10)
        self.assertIn("ny_lim", out)

    def test_forbidden_key_strict_raises(self):
        with self.assertRaises(ParameterFirewallViolation):
            clamp_tunable_params({"air_density": -5.0}, strict=True)


class TestSanitizeRLHyperparams(unittest.TestCase):
    def test_clamps_reward_weights(self):
        sanitized, warnings = sanitize_rl_hyperparams({
            "pm_penalty": 99.0,
            "hit_rate_weight": 0.01,
            "unknown_key": 1.0,
        })
        self.assertEqual(sanitized["pm_penalty"], 8.0)
        self.assertEqual(sanitized["hit_rate_weight"], 0.5)
        self.assertNotIn("unknown_key", sanitized)
        self.assertTrue(any("unknown" in w for w in warnings))

    def test_rejects_non_finite(self):
        sanitized, warnings = sanitize_rl_hyperparams({"lr_actor": float("nan")})
        self.assertEqual(sanitized, {})
        self.assertTrue(warnings)


class TestApplyRLHyperparams(unittest.TestCase):
    def test_applies_bounded_values(self):
        from multi_agent.rl.matlab_rl_optimizer import MatlabRLOptimizer

        opt = MatlabRLOptimizer(max_episodes=50, nmc_per_eval=10)
        applied = apply_rl_hyperparams_to_optimizer(opt, {
            "lr_actor": 0.5,
            "pm_penalty": 3.0,
            "max_episodes": 200,
        })
        self.assertEqual(applied["lr_actor"], 1e-2)
        self.assertEqual(applied["max_episodes"], 150)
        self.assertEqual(opt._rw["pm_penalty"], 3.0)


class TestTaskWorkspace(unittest.TestCase):
    def test_prepare_and_cleanup(self):
        with tempfile.TemporaryDirectory() as base:
            src = os.path.join(base, "demo.m")
            with open(src, "w", encoding="utf-8") as fh:
                fh.write("function demo(); end\n")

            ws_base = os.path.join(base, "tasks")
            with TaskWorkspace(base_dir=ws_base) as ws:
                isolated = ws.prepare_script(src)
                self.assertTrue(os.path.isfile(isolated))
                self.assertTrue(isolated.startswith(ws.path))
                self.assertNotEqual(isolated, src)

            self.assertFalse(os.path.isdir(ws.path))


if __name__ == "__main__":
    unittest.main()
