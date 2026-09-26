import json
import tempfile
import unittest
from pathlib import Path

from experiments import experiment_plan_runner as runner
from multi_agent.metric_contract import DEFAULT_METRIC_REQUIREMENTS


class ExperimentPlanRunnerTests(unittest.TestCase):
    def test_seed_manifest_uses_three_dev_and_three_test_seeds(self):
        manifest = runner.seed_manifest([42, 43])

        self.assertEqual(manifest["42"]["dev"], [4201, 4202, 4203])
        self.assertEqual(manifest["42"]["test"], [4204, 4205, 4206])
        self.assertEqual(manifest["43"]["dev"], [4301, 4302, 4303])
        self.assertEqual(manifest["43"]["test"], [4304, 4305, 4306])

    def test_protocol_records_metric_contract(self):
        protocol = runner.protocol_payload(
            scenarios=["T1"],
            master_seeds=[42],
            n_runs=100,
            iterations=5,
        )

        self.assertEqual(protocol["metric_requirements"], DEFAULT_METRIC_REQUIREMENTS)
        self.assertEqual(protocol["and_logic"], "all three dev seeds must pass; all three test seeds must pass")
        self.assertEqual(protocol["search_dim"], 10)

    def test_readme_ablation_configs_are_available(self):
        self.assertFalse(
            runner.MEMORY_CONFIGS["E1_no_memory"]["ablation"]["parameter_experience_reuse"]["enabled"]
        )
        self.assertFalse(runner.MEMORY_CONFIGS["E1_no_memory"]["ablation"]["memory_search"]["enabled"])
        self.assertTrue(runner.MEMORY_CONFIGS["E1_no_memory"]["ablation"]["reflection_agent"]["enabled"])
        self.assertFalse(runner.MEMORY_CONFIGS["E2_no_reflection"]["ablation"]["reflection_agent"]["enabled"])
        self.assertFalse(
            runner.OPTIMIZATION_CONFIGS["O1_no_optimization_agent"]["ablation"]["rl_optimization"]["enabled"]
        )
        self.assertFalse(
            runner.OPTIMIZATION_CONFIGS["O1_no_optimization_agent"]["matlab_rl_optimizer"][
                "constraint_local_search"
            ]["enabled"]
        )
        self.assertFalse(runner.CORRECTION_CONFIGS["C1_no_layer1_syntax"]["ablation"]["syntax_check"]["enabled"])
        self.assertFalse(
            runner.CORRECTION_CONFIGS["C2_no_layers23_compat_semantic"]["ablation"][
                "guidance_compat_verification"
            ]["enabled"]
        )

    def test_make_plan_writes_manifest_and_order_commands(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "plan"
            runner.main(
                [
                    "make-plan",
                    "--sections",
                    "optimizer",
                    "order",
                    "--scenarios",
                    "T1",
                    "--master-seeds",
                    "42",
                    "--iterations",
                    "5",
                    "--n-runs",
                    "100",
                    "--output-dir",
                    str(out_dir),
                ]
            )

            manifest_path = out_dir / "manifest.json"
            commands_path = out_dir / "commands.ps1"
            execute_path = out_dir / "commands_execute.ps1"
            self.assertTrue(manifest_path.exists())
            self.assertTrue(commands_path.exists())
            self.assertTrue(execute_path.exists())
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            names = [item["name"] for item in payload["commands"]]
            self.assertIn("optimizer_baselines_43", names)
            self.assertIn("order_ablation_forward", names)
            self.assertIn("order_ablation_reverse", names)
            self.assertIn("order_ablation_shuffled", names)
            self.assertEqual(payload["protocol"]["metric_requirements"], DEFAULT_METRIC_REQUIREMENTS)
            self.assertIn("--execute", execute_path.read_text(encoding="utf-8"))

    def test_gsm_suite_plan_writes_protocol_without_execution(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_csv = Path(tmp) / "memory.csv"
            runner.main(
                [
                    "run-gsm-suite",
                    "--suite",
                    "memory",
                    "--configs",
                    "E1_no_memory",
                    "--scenarios",
                    "T1",
                    "--master-seeds",
                    "42",
                    "--output-csv",
                    str(output_csv),
                ]
            )

            protocol_path = output_csv.with_suffix(".protocol.json")
            self.assertTrue(protocol_path.exists())
            payload = json.loads(protocol_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["suite"], "memory")
            self.assertEqual(len(payload["jobs"]), 1)
            self.assertFalse(payload["jobs"][0]["execute"])


if __name__ == "__main__":
    unittest.main()
