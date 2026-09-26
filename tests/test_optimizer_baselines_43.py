#!/usr/bin/env python3
"""Tests for Section 4.3 optimizer baseline protocol helpers."""

import math
import unittest
from pathlib import Path

from experiments import optimizer_baselines_43 as ob43
from multi_agent.config_loader import RLRewardConfig
from multi_agent.memory.parameter_experience import ParameterExperience
from multi_agent.metric_contract import DEFAULT_METRIC_REQUIREMENTS
from multi_agent.tools.simulation_tool import check_constraints


class TestOptimizerBaselines43Protocol(unittest.TestCase):
    def test_search_space_is_10d(self):
        self.assertEqual(len(ob43.PARAM_NAMES), 10)
        self.assertIn("ny_lim", ob43.PARAM_NAMES)
        ob43.assert_protocol_dimension(10)

    def test_cma_popsize_matches_protocol(self):
        dim = len(ob43.PARAM_NAMES)
        popsize = int(4 + math.floor(3.0 * math.log(dim)))
        self.assertEqual(popsize, 10)

    def test_equal_mc_budget_matches_gsm_ceiling(self):
        from experiments.equal_mc_budget import (
            DEFAULT_MAX_CANDIDATES,
            DEFAULT_MAX_MC_CALLS,
            GSM_MAX_EPISODES,
            GSM_OUTER_ITERATIONS,
            max_mc_calls,
            remaining_candidates,
            split_agent_budget,
        )

        self.assertEqual(ob43.DEFAULT_ITERATIONS, DEFAULT_MAX_CANDIDATES)
        self.assertEqual(DEFAULT_MAX_CANDIDATES, GSM_OUTER_ITERATIONS * GSM_MAX_EPISODES)
        self.assertEqual(DEFAULT_MAX_CANDIDATES, 75)
        self.assertEqual(max_mc_calls(n_dev_seeds=3), 225)
        self.assertEqual(DEFAULT_MAX_MC_CALLS, 225)
        self.assertEqual(remaining_candidates(0, 15, 3), 5)
        self.assertEqual(remaining_candidates(12, 15, 3), 1)
        self.assertEqual(remaining_candidates(15, 15, 3), 0)
        self.assertEqual(split_agent_budget(75), (5, 15))
        self.assertEqual(split_agent_budget(5), (5, 1))

    def test_cma_es_stops_at_shared_candidate_ceiling(self):
        """CMA-ES must not spend λ evaluations per 'iteration'."""
        calls = {"n": 0}

        class FakeEval:
            def evaluate(self, params, scenario, seeds, split):
                calls["n"] += 1
                n_seeds = len(seeds)
                return ob43.CandidateEvaluation(
                    split=split,
                    seeds=list(seeds),
                    seed_results=[],
                    all_pass=False,
                    score=1.0,
                    metrics={"hit_rate": 0.0, "SEP": 99.0, "peak_nz": 99.0, "PM": 0.0, "BW": 0.0},
                    mc_calls=n_seeds,
                    sim_time_sec=0.0,
                    error="",
                )

        result = ob43.run_cma_es(
            FakeEval(),
            scenario="T1",
            master_seed=42,
            dev_seeds=[4201, 4202, 4203],
            test_seeds=[4204, 4205, 4206],
            iterations=5,
        )
        self.assertLessEqual(result.mc_calls_dev, 15)
        self.assertEqual(result.mc_calls_dev, 15)
        self.assertEqual(calls["n"], 6)  # 5 dev candidates + 1 test eval
        self.assertLessEqual(result.iterations_used, 5)

    def test_random_search_respects_same_ceiling(self):
        class FakeEval:
            def evaluate(self, params, scenario, seeds, split):
                return ob43.CandidateEvaluation(
                    split=split,
                    seeds=list(seeds),
                    seed_results=[],
                    all_pass=False,
                    score=1.0,
                    metrics={},
                    mc_calls=len(seeds),
                    sim_time_sec=0.0,
                    error="",
                )

        result = ob43.run_random_search(
            FakeEval(),
            scenario="T1",
            master_seed=42,
            dev_seeds=[4201, 4202, 4203],
            test_seeds=[4204, 4205, 4206],
            iterations=5,
        )
        self.assertEqual(result.mc_calls_dev, 15)
        self.assertEqual(result.iterations_used, 5)


class TestPPOProbabilityRatio(unittest.TestCase):
    def test_ratio_is_exp_of_log_difference(self):
        from multi_agent.rl.matlab_rl_optimizer import ppo_probability_ratio

        self.assertAlmostEqual(ppo_probability_ratio(0.0, 0.0), 1.0)
        self.assertAlmostEqual(ppo_probability_ratio(math.log(2.0), 0.0), 2.0)
        self.assertAlmostEqual(ppo_probability_ratio(0.0, math.log(2.0)), 0.5)
        huge = ppo_probability_ratio(100.0, 0.0)
        self.assertLess(huge, math.exp(10.0) + 1e-6)
        self.assertGreater(huge, math.exp(10.0) - 1e-6)

    def test_default_n_runs_matches_requirement(self):
        self.assertEqual(ob43.DEFAULT_N_RUNS, 100)

    def test_metric_contract_matches_current_requirement(self):
        self.assertEqual(DEFAULT_METRIC_REQUIREMENTS["hit_rate_min"], 92.0)
        self.assertEqual(DEFAULT_METRIC_REQUIREMENTS["sep_max"], 7.0)
        self.assertEqual(DEFAULT_METRIC_REQUIREMENTS["peak_ny_max"], 20.0)
        self.assertEqual(DEFAULT_METRIC_REQUIREMENTS["pm_min"], 45.0)
        self.assertEqual(DEFAULT_METRIC_REQUIREMENTS["pm_max"], 70.0)
        self.assertEqual(DEFAULT_METRIC_REQUIREMENTS["bw_min"], 20.0)
        self.assertEqual(DEFAULT_METRIC_REQUIREMENTS["bw_max"], 85.0)

    def test_check_constraints_uses_current_requirement(self):
        feasible, violations = check_constraints(92.0, 7.0, 20.0, 45.0, 85.0)
        self.assertTrue(feasible)
        self.assertTrue(all(v == 0.0 for v in violations.values()))

        feasible, violations = check_constraints(91.0, 7.1, 20.5, 44.0, 85.5)
        self.assertFalse(feasible)
        self.assertGreater(violations["hit_rate"], 0.0)
        self.assertGreater(violations["SEP"], 0.0)
        self.assertGreater(violations["peak_nz"], 0.0)
        self.assertGreater(violations["PM_low"], 0.0)
        self.assertGreater(violations["BW_high"], 0.0)

    def test_reward_config_defaults_use_current_pm_bw_range(self):
        cfg = RLRewardConfig()
        self.assertEqual(cfg.pm_min, 45.0)
        self.assertEqual(cfg.pm_max, 70.0)
        self.assertEqual(cfg.bw_min, 20.0)
        self.assertEqual(cfg.bw_max, 85.0)

    def test_parameter_experience_fitness_uses_current_requirement(self):
        good = ParameterExperience.compute_fitness_from_objectives({
            "hit_rate": 92.0,
            "SEP": 7.0,
            "peak_ny": 20.0,
            "pitch_PM": 45.0,
            "pitch_BW": 85.0,
        })
        bad = ParameterExperience.compute_fitness_from_objectives({
            "hit_rate": 91.0,
            "SEP": 8.0,
            "peak_ny": 21.0,
            "pitch_PM": 44.0,
            "pitch_BW": 86.0,
        })
        self.assertGreater(good, bad)

    def test_seed_derivation_matches_requirement(self):
        self.assertEqual(ob43.dev_seeds_for_master(42), [4201, 4202, 4203])
        self.assertEqual(ob43.test_seeds_for_master(42), [4204, 4205, 4206])
        for master in (42, 43, 44):
            dev, test = ob43.resolve_split_seeds(master)
            self.assertEqual(dev, [master * 100 + o for o in (1, 2, 3)])
            self.assertEqual(test, [master * 100 + o for o in (4, 5, 6)])

    def test_normalized_round_trip(self):
        x = ob43.params_to_normalized(ob43.DEFAULT_PARAMS)
        self.assertEqual(x.shape[0], 10)
        params = ob43.normalized_to_params(x)
        for key in ob43.PARAM_NAMES:
            self.assertAlmostEqual(params[key], ob43.DEFAULT_PARAMS[key], places=6)

    def test_external_mc_path_layout(self):
        path = ob43.mc_data_path(Path("data/mc"), "T4", 4201, "dev")
        self.assertEqual(path.as_posix(), "data/mc/T4/seed_4201_dev.mat")

    def test_seed_override_keeps_legacy_layout_available(self):
        dev, test = ob43.resolve_split_seeds(
            42,
            dev_override=[42, 43, 44],
            test_override=[45, 46, 47],
        )
        self.assertEqual(dev, [42, 43, 44])
        self.assertEqual(test, [45, 46, 47])

    def test_and_aggregation_fails_on_one_bad_seed(self):
        good = {
            "success": True,
            "feasible": True,
            "hit_rate": 100.0,
            "SEP": 4.0,
            "peak_nz": 18.0,
            "PM": 55.0,
            "BW": 40.0,
            "constraint_violations": {},
        }
        bad = {
            **good,
            "feasible": False,
            "SEP": 12.0,
            "constraint_violations": {"SEP": 2.0},
        }
        all_pass, metrics, score, error = ob43.aggregate_seed_results([good, bad, good])
        self.assertFalse(all_pass)
        self.assertEqual(metrics["SEP"], 12.0)
        self.assertGreater(score, 0.0)
        self.assertEqual(error, "")

    def test_sah_first_success_matches_final_dev(self):
        """Table 5 GSM/SAH: first_success = final_dev_feasible."""
        common = dict(
            scenario="T1",
            master_seed=42,
            search_dim=10,
            iterations_budget=5,
            iterations_used=5,
            mc_calls_dev=12,
            mc_calls_test=3,
            wall_clock_sec=1.0,
            sim_time_sec=1.0,
            first_success_iter=None,
            test_all_pass=False,
            best_score=1.0,
            best_params={},
            best_dev={},
            test_eval={},
        )
        sah = ob43.BaselineRunResult(
            method="single_agent_hermes",
            dev_all_pass=True,
            first_success_definition="final_dev_feasible",
            **common,
        )
        row = ob43.run_to_row(sah, Path("t.json"))
        self.assertTrue(row["first_success"])

        random = ob43.BaselineRunResult(
            method="random_search",
            dev_all_pass=True,
            **common,
        )
        row_r = ob43.run_to_row(random, Path("t.json"))
        self.assertFalse(row_r["first_success"])


if __name__ == "__main__":
    unittest.main()
