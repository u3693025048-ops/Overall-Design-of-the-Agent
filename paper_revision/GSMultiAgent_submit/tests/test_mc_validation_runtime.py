#!/usr/bin/env python3
"""Runtime policy tests for final multi-seed MC validation."""

import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from multi_agent.config_loader import MCValidationConfig
from multi_agent.simulation.optimization_workflow import OptimizationWorkflow
from multi_agent.tools.simulation_tool import RunSimulationTool, _mc_validation_auto_runs


REQS = {
    "hit_rate_min": 92.0,
    "sep_max": 7.0,
    "peak_ny_max": 20.0,
    "peak_ny_mean_max": 20.0,
    "pm_min": 45.0,
    "pm_max": 70.0,
    "bw_min": 20.0,
    "bw_max": 85.0,
}


def _metric(seed, hit=100.0, sep=5.0, peak=18.0, pm=55.0, bw=60.0):
    return {
        "mc_seed": seed,
        "hit_rate": hit,
        "SEP": sep,
        "peak_ny": peak,
        "pitch_PM": pm,
        "pitch_BW": bw,
    }


class TestMcValidationConfig(unittest.TestCase):
    def test_mode_defaults_to_final(self):
        cfg = MCValidationConfig.from_dict(
            {"enabled": True, "seeds": "42,2025,3407"}
        )
        self.assertEqual(cfg.mode, "final")
        self.assertFalse(_mc_validation_auto_runs(cfg.mode))

    def test_all_mode_auto_runs(self):
        cfg = MCValidationConfig.from_dict({"enabled": True, "mode": "always"})
        self.assertEqual(cfg.mode, "all")
        self.assertTrue(_mc_validation_auto_runs(cfg.mode))


class TestFinalMcValidationWorkflow(unittest.IsolatedAsyncioTestCase):
    def _fake_cfg(self, mode="final", final_nmc=11):
        return SimpleNamespace(
            simulation=SimpleNamespace(
                mc_validation=SimpleNamespace(
                    enabled=True,
                    mode=mode,
                    seeds=[42, 2025, 3407],
                    final_nmc=final_nmc,
                ),
                engine="matlab",
                octave_path="octave",
                matlab_path="matlab",
                mc_data_path="",
            ),
            matlab_rl_optimizer=SimpleNamespace(nmc_per_eval=25),
        )

    async def test_run_simulation_final_mode_is_single_run_by_default(self):
        calls = []

        async def fake_run(self, script_path, dp_params, mission_conditions, nmc, random_seed=None):
            calls.append(random_seed)
            return {"status": "success", "_full_stdout": "", "stdout": ""}

        tool = RunSimulationTool()
        with patch("multi_agent.config_loader.get_config", return_value=self._fake_cfg()):
            with patch.object(RunSimulationTool, "_run_script_directly", fake_run):
                await tool.execute(script_path="dummy.m", nmc=5)
                await tool.execute(script_path="dummy.m", nmc=5, validate_final=True)

        self.assertEqual(calls, [None, 42, 2025, 3407])

    async def test_final_mc_failure_blocks_done(self):
        calls = []

        class FakeRunSimulationTool:
            def __init__(self):
                self.last_metrics = {}

            def set_simulator(self, simulator):
                self.simulator = simulator

            def set_parameter_experience(self, pe):
                self.parameter_experience = pe

            def set_mission_conditions(self, conditions):
                self.mission_conditions = conditions

            async def execute(self, **kwargs):
                calls.append(kwargs)
                seed_results = [
                    _metric(42),
                    _metric(2025, sep=8.0),
                    _metric(3407),
                ]
                self.last_metrics = {
                    "mc_seed_results": seed_results,
                    "mc_seed_count": 3,
                    "hit_rate": 100.0,
                    "SEP": 8.0,
                    "peak_ny": 18.0,
                    "pitch_PM": 55.0,
                    "pitch_BW": 60.0,
                }
                return {
                    "status": "success",
                    "mc_seed_results": seed_results,
                    "mc_seed_aggregate": self.last_metrics,
                }

        fake_cfg = self._fake_cfg()
        workflow = OptimizationWorkflow(
            judgment_agent=SimpleNamespace(_reqs=REQS),
            rl_optimizer=SimpleNamespace(simulator=None),
        )
        good_metrics = {
            "hit_rate": 100.0,
            "SEP": 5.0,
            "peak_ny": 18.0,
            "pitch_PM": 55.0,
            "pitch_BW": 60.0,
        }

        fd, script_path = tempfile.mkstemp(suffix=".m")
        os.close(fd)
        try:
            with patch("multi_agent.config_loader.get_config", return_value=fake_cfg):
                with patch(
                    "multi_agent.tools.simulation_tool.RunSimulationTool",
                    FakeRunSimulationTool,
                ):
                    final_metrics, ok, note = await workflow._maybe_validate_final_mc_seeds(
                        script_path=script_path,
                        params={},
                        metrics=good_metrics,
                        mission_conditions=None,
                        task_prompt="",
                        nmc=5,
                    )
        finally:
            os.unlink(script_path)

        self.assertFalse(ok)
        self.assertIn("Final MC validation failed", note)
        self.assertIn("mc_seed_results", final_metrics)
        self.assertEqual(calls[0]["mc_seeds"], [42, 2025, 3407])
        self.assertEqual(calls[0]["nmc"], 11)


if __name__ == "__main__":
    unittest.main()
