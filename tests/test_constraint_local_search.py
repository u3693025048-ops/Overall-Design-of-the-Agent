#!/usr/bin/env python3
"""Unit tests for ConstraintLocalSearch (mocked simulation)."""

import unittest
from unittest.mock import AsyncMock, MagicMock

from multi_agent.rl.constraint_local_search import ConstraintLocalSearch
from multi_agent.rl.matlab_rl_optimizer import MatlabRLOptimizer


class TestConstraintLocalSearch(unittest.IsolatedAsyncioTestCase):
    async def test_refine_picks_lower_peak_ny(self):
        backend = MatlabRLOptimizer()
        backend._compute_reward = MagicMock(return_value=0.5)
        backend._compute_pe_fitness = MagicMock(return_value=0.6)

        from multi_agent.rl.matlab_rl_optimizer import ALL_TUNABLE_PARAM_SPECS

        center = {k: float(v["nominal"]) for k, v in ALL_TUNABLE_PARAM_SPECS.items()}
        calls = {"n": 0}

        async def fake_sim(params, mission, script, nmc):
            calls["n"] += 1
            pny = 30.0 if calls["n"] == 1 else 19.0
            return {
                "hit_rate": 100.0,
                "SEP": 5.0,
                "miss_distance": 5.0,
                "peak_ny": pny,
                "peak_ny_max": pny,
                "pitch_PM": 55.0,
                "pitch_BW": 40.0,
            }

        backend._run_simulation_with_params = fake_sim
        backend.extract_params_from_script = MagicMock()

        cls = ConstraintLocalSearch(backend, max_iterations=3, step_scale=0.05, seed=42)
        reqs = {
            "hit_rate_min": 92.0,
            "sep_max": 7.0,
            "peak_ny_max": 20.0,
            "pm_min": 45.0,
            "pm_max": 70.0,
            "bw_min": 20.0,
            "bw_max": 85.0,
        }
        result = await cls.refine(
            center,
            script_path="dummy.m",
            mission_conditions={"T": [4]},
            nmc=5,
            metric_requirements=reqs,
        )
        self.assertEqual(result["selection_criteria"], "constrained")
        self.assertLessEqual(result["best_metrics"].get("peak_ny", 99), 20.0)
        self.assertEqual(result["total_iterations"], 1)
        self.assertEqual(calls["n"], 2)

    async def test_refine_stops_at_center_when_already_satisfied(self):
        backend = MatlabRLOptimizer()
        backend._compute_reward = MagicMock(return_value=0.8)
        backend._compute_pe_fitness = MagicMock(return_value=0.9)

        from multi_agent.rl.matlab_rl_optimizer import ALL_TUNABLE_PARAM_SPECS

        center = {k: float(v["nominal"]) for k, v in ALL_TUNABLE_PARAM_SPECS.items()}
        calls = {"n": 0}

        async def fake_sim(params, mission, script, nmc):
            calls["n"] += 1
            return {
                "hit_rate": 100.0,
                "SEP": 5.0,
                "miss_distance": 5.0,
                "peak_ny": 18.0,
                "peak_ny_max": 19.5,
                "pitch_PM": 55.0,
                "pitch_BW": 40.0,
            }

        backend._run_simulation_with_params = fake_sim
        backend.extract_params_from_script = MagicMock()

        cls = ConstraintLocalSearch(backend, max_iterations=50, step_scale=0.05, seed=42)
        reqs = {
            "hit_rate_min": 92.0,
            "sep_max": 7.0,
            "peak_ny_max": 20.0,
            "pm_min": 45.0,
            "pm_max": 70.0,
            "bw_min": 20.0,
            "bw_max": 85.0,
        }
        result = await cls.refine(
            center,
            script_path="dummy.m",
            mission_conditions={"T": [4]},
            nmc=5,
            metric_requirements=reqs,
        )
        self.assertEqual(result["selection_criteria"], "constrained")
        self.assertEqual(result["total_iterations"], 0)
        self.assertEqual(calls["n"], 1)

    async def test_refine_skips_borderline_when_five_metrics_required(self):
        backend = MatlabRLOptimizer()
        backend._compute_reward = MagicMock(return_value=0.5)
        backend._compute_pe_fitness = MagicMock(return_value=0.6)

        from multi_agent.rl.matlab_rl_optimizer import ALL_TUNABLE_PARAM_SPECS

        center = {k: float(v["nominal"]) for k, v in ALL_TUNABLE_PARAM_SPECS.items()}
        calls = {"n": 0}

        async def fake_sim(params, mission, script, nmc):
            calls["n"] += 1
            return {
                "hit_rate": 100.0,
                "SEP": 5.0,
                "miss_distance": 5.0,
                "peak_ny": 19.0,
                "peak_ny_max": 19.5,
                "pitch_PM": 55.0,
                "pitch_BW": 110.0,
            }

        backend._run_simulation_with_params = fake_sim
        backend.extract_params_from_script = MagicMock()

        cls = ConstraintLocalSearch(backend, max_iterations=2, step_scale=0.05, seed=42)
        task_prompt = (
            "命中率 >= 92% SEP <= 7 m PeakNy <= 20 g "
            "PM 在 45°~70° BW 在 20~85 rad/s"
        )
        result = await cls.refine(
            center,
            script_path="dummy.m",
            mission_conditions={"T": [4]},
            nmc=5,
            task_prompt=task_prompt,
        )
        self.assertNotEqual(result["selection_criteria"], "borderline")
        self.assertGreater(result["best_metrics"].get("pitch_BW", 0), 85.0)

    async def test_no_regression_gate_preserves_hit(self):
        backend = MatlabRLOptimizer()
        backend._compute_reward = MagicMock(return_value=0.5)
        backend._compute_pe_fitness = MagicMock(return_value=0.6)

        from multi_agent.rl.matlab_rl_optimizer import ALL_TUNABLE_PARAM_SPECS

        center = {k: float(v["nominal"]) for k, v in ALL_TUNABLE_PARAM_SPECS.items()}
        calls = {"n": 0}

        async def fake_sim(params, mission, script, nmc):
            calls["n"] += 1
            if calls["n"] == 1:
                return {
                    "hit_rate": 100.0,
                    "SEP": 8.0,
                    "miss_distance": 8.0,
                    "peak_ny": 22.0,
                    "peak_ny_max": 24.0,
                    "pitch_PM": 55.0,
                    "pitch_BW": 25.0,
                }
            if calls["n"] == 2:
                return {
                    "hit_rate": 80.0,
                    "SEP": 5.0,
                    "miss_distance": 5.0,
                    "peak_ny": 19.0,
                    "peak_ny_max": 20.0,
                    "pitch_PM": 55.0,
                    "pitch_BW": 40.0,
                }
            return {
                "hit_rate": 100.0,
                "SEP": 6.0,
                "miss_distance": 6.0,
                "peak_ny": 19.0,
                "peak_ny_max": 20.0,
                "pitch_PM": 55.0,
                "pitch_BW": 40.0,
            }

        backend._run_simulation_with_params = fake_sim
        backend.extract_params_from_script = MagicMock()

        cls = ConstraintLocalSearch(
            backend,
            max_iterations=3,
            step_scale=0.05,
            seed=42,
            no_regression_gate=True,
            directed_perturb=True,
        )
        reqs = {
            "hit_rate_min": 92.0,
            "sep_max": 7.0,
            "peak_ny_max": 20.0,
            "pm_min": 45.0,
            "pm_max": 70.0,
            "bw_min": 20.0,
            "bw_max": 85.0,
        }
        result = await cls.refine(
            center,
            script_path="dummy.m",
            mission_conditions={"T": [4]},
            nmc=5,
            metric_requirements=reqs,
        )
        self.assertGreaterEqual(result["best_metrics"].get("hit_rate", 0), 92.0)
        self.assertLessEqual(result["best_metrics"].get("peak_ny", 99), 20.0)


if __name__ == "__main__":
    unittest.main()
