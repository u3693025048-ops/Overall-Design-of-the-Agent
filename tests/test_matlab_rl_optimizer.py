#!/usr/bin/env python3
"""Unit tests for MatlabRLOptimizer — aligned with docsNew/PPO_intr.html."""

import math
import unittest

import numpy as np

from multi_agent.rl.matlab_rl_optimizer import (
    ALL_TUNABLE_PARAM_SPECS,
    AUTOPILOT_PARAM_SPECS,
    GUIDANCE_PARAM_SPECS,
    HARD_REWARD_PENALTY,
    MatlabRLOptimizer,
    get_peak_ny,
    get_sep,
    metrics_look_like_parse_defaults,
    parse_sim_stdout,
    patch_ny_lim_in_script,
    stdout_has_mc_evidence,
    validate_rl_script_content,
)


class TestMetricHelpers(unittest.TestCase):
    def test_get_peak_ny_prefers_peak_ny(self):
        self.assertEqual(get_peak_ny({"peak_ny": 12.0, "peak_n": 99.0}), 12.0)

    def test_get_peak_ny_fallback_peak_n(self):
        self.assertEqual(get_peak_ny({"peak_n": 8.5}), 8.5)

    def test_get_sep_prefers_sep(self):
        self.assertEqual(get_sep({"SEP": 3.2, "miss_distance": 99.0}), 3.2)

    def test_get_sep_fallback_miss_distance(self):
        self.assertEqual(get_sep({"miss_distance": 7.0}), 7.0)


class TestParamSpecsDocAlignment(unittest.TestCase):
    """Verify AUTOPILOT_PARAM_SPECS match PPO_intr.html §2.2 / §3.2."""

    def test_w1_range(self):
        s = AUTOPILOT_PARAM_SPECS["w1"]
        self.assertEqual(s["min"], 20.0)
        self.assertEqual(s["max"], 60.0)

    def test_n_pn_range(self):
        s = GUIDANCE_PARAM_SPECS["N_pn"]
        self.assertEqual(s["min"], 3.0)
        self.assertEqual(s["max"], 6.0)

    def test_ten_tunable_params(self):
        self.assertEqual(len(ALL_TUNABLE_PARAM_SPECS), 10)
        self.assertIn("ny_lim", ALL_TUNABLE_PARAM_SPECS)


class TestPatchNyLim(unittest.TestCase):
    _SNIPPET = (
        "N1=ayb/g0;\n"
        "ny_lim=20;\n"
        "N1=ny_lim*tanh(N1/ny_lim);\n"
        "N1=max(-20,min(20,N1));\n"
    )

    def test_lowers_ny_lim_and_hard_clip(self):
        out = patch_ny_lim_in_script(self._SNIPPET, 18.5)
        self.assertIn("ny_lim=18.5", out)
        self.assertIn("max(-ny_lim,min(ny_lim,N1)", out)
        self.assertNotIn("max(-20,min(20,N1)", out)


class TestActionMapping(unittest.TestCase):
    def setUp(self):
        self.opt = MatlabRLOptimizer()

    def test_action_zero_maps_to_mid(self):
        action = np.zeros(self.opt._action_dim, dtype=np.float32)
        params = self.opt._action_to_params(action)
        for k, spec in ALL_TUNABLE_PARAM_SPECS.items():
            mid = (spec["min"] + spec["max"]) / 2.0
            self.assertAlmostEqual(params[k], mid, places=5)

    def test_action_plus_one_maps_to_max(self):
        action = np.ones(self.opt._action_dim, dtype=np.float32)
        params = self.opt._action_to_params(action)
        for k, spec in ALL_TUNABLE_PARAM_SPECS.items():
            self.assertAlmostEqual(params[k], spec["max"], places=5)

    def test_state_uses_current_params_not_only_nominal(self):
        self.opt._init_networks()
        custom = {k: spec["max"] for k, spec in ALL_TUNABLE_PARAM_SPECS.items()}
        self.opt._current_params = custom
        state = self.opt._build_state(
            self.opt._current_params,
            self.opt._base_phys,
            self.opt._last_metrics,
        )
        # w1 at max → normalized +1
        self.assertAlmostEqual(float(state[0]), 1.0, places=4)


class TestRewardFunction(unittest.TestCase):
    def setUp(self):
        self.opt = MatlabRLOptimizer(peak_n_penalty=2.5)

    def test_hard_truncation_nan_sep(self):
        r = self.opt._compute_reward({"SEP": float("nan"), "peak_ny": 5.0})
        self.assertEqual(r, HARD_REWARD_PENALTY)

    def test_hard_truncation_peak_ny_over_50(self):
        r = self.opt._compute_reward({"SEP": 100.0, "peak_ny": 55.0})
        self.assertEqual(r, HARD_REWARD_PENALTY)

    def test_phase_a_no_pm_bw_bonus(self):
        """SEP > 500 m → Phase A: no PM/BW terms (PPO_intr §4.1)."""
        r_good_pm = self.opt._compute_reward({
            "SEP": 800.0, "peak_ny": 10.0,
            "pitch_PM": 60.0, "pitch_BW": 50.0, "hit_rate": 100.0,
        })
        r_bad_pm = self.opt._compute_reward({
            "SEP": 800.0, "peak_ny": 10.0,
            "pitch_PM": 10.0, "pitch_BW": 5.0, "hit_rate": 0.0,
        })
        self.assertAlmostEqual(r_good_pm, r_bad_pm, places=5)

    def test_phase_b_pm_affects_reward(self):
        r_good = self.opt._compute_reward({
            "SEP": 50.0, "peak_ny": 10.0,
            "pitch_PM": 55.0, "pitch_BW": 40.0, "hit_rate": 90.0,
        })
        r_bad = self.opt._compute_reward({
            "SEP": 50.0, "peak_ny": 10.0,
            "pitch_PM": 10.0, "pitch_BW": 5.0, "hit_rate": 90.0,
        })
        self.assertGreater(r_good, r_bad)

    def test_fallback_sep_only(self):
        r = self.opt._compute_reward({
            "_sim_source": "fallback",
            "SEP": 200.0,
            "pitch_PM": 70.0,
            "pitch_BW": 80.0,
        })
        self.assertAlmostEqual(r, -0.2, places=5)

    def test_peak_n_penalty_synced_to_reward_weights(self):
        self.assertEqual(self.opt._rw["peak_ny_penalty"], 2.5)
        self.assertEqual(self.opt._peak_n_penalty, 2.5)

    def test_reward_defaults_use_current_metric_ranges(self):
        self.assertEqual(self.opt._rw["pm_min"], 45.0)
        self.assertEqual(self.opt._rw["pm_max"], 70.0)
        self.assertEqual(self.opt._rw["bw_min"], 20.0)
        self.assertEqual(self.opt._rw["bw_max"], 85.0)

    def test_pe_fitness_prefers_current_requirement_band(self):
        good = self.opt._compute_pe_fitness({
            "hit_rate": 92.0,
            "SEP": 7.0,
            "peak_ny": 20.0,
            "pitch_PM": 45.0,
            "pitch_BW": 85.0,
        })
        bad = self.opt._compute_pe_fitness({
            "hit_rate": 91.0,
            "SEP": 8.0,
            "peak_ny": 21.0,
            "pitch_PM": 44.0,
            "pitch_BW": 86.0,
        })
        self.assertGreater(good, bad)


class TestRequirementsAndAnomaly(unittest.TestCase):
    def test_requirements_met_uses_sep_alias(self):
        ok = MatlabRLOptimizer._requirements_met(
            {"SEP": 5.0, "hit_rate": 90.0},
            {"missmean": 10.0},
        )
        self.assertTrue(ok)

    def test_requirements_met_peak_ny_alias(self):
        ok = MatlabRLOptimizer._requirements_met(
            {"peak_ny": 15.0},
            {"peak_n": 20.0},
        )
        self.assertTrue(ok)

    def test_is_anomalous_nan_sep(self):
        self.assertTrue(MatlabRLOptimizer._is_anomalous({"SEP": float("nan")}))


class TestPPOConfig(unittest.TestCase):
    def test_contextual_bandit_gamma_zero(self):
        opt = MatlabRLOptimizer(gamma=0.995)
        self.assertEqual(opt.gamma, 0.0)

    def test_default_batch_size_64(self):
        opt = MatlabRLOptimizer()
        self.assertEqual(opt.episodes_per_update, 64)


class TestDocChecklist(unittest.TestCase):
    """Automated checklist vs PPO_intr.html key facts."""

    DOC_EXPECTATIONS = {
        "action_dim": 10,
        "state_dim": 17,
        "hard_reward_penalty": -10.0,
        "gamma": 0.0,
        "episodes_per_update": 64,
        "sep_survival_threshold": 500.0,
        "w1_max": 60.0,
        "n_pn_max": 6.0,
    }

    def test_doc_checklist(self):
        opt = MatlabRLOptimizer()
        opt._init_networks()
        dummy_state = opt._build_state(
            opt._base_auto, opt._base_phys, opt._last_metrics
        )
        self.assertEqual(len(dummy_state), self.DOC_EXPECTATIONS["state_dim"])
        self.assertEqual(opt._action_dim, self.DOC_EXPECTATIONS["action_dim"])
        self.assertEqual(HARD_REWARD_PENALTY, self.DOC_EXPECTATIONS["hard_reward_penalty"])
        self.assertEqual(opt.gamma, self.DOC_EXPECTATIONS["gamma"])
        self.assertEqual(opt.episodes_per_update, self.DOC_EXPECTATIONS["episodes_per_update"])
        self.assertEqual(
            opt._rw["sep_survival_threshold"],
            self.DOC_EXPECTATIONS["sep_survival_threshold"],
        )
        self.assertEqual(
            AUTOPILOT_PARAM_SPECS["w1"]["max"],
            self.DOC_EXPECTATIONS["w1_max"],
        )
        self.assertEqual(
            GUIDANCE_PARAM_SPECS["N_pn"]["max"],
            self.DOC_EXPECTATIONS["n_pn_max"],
        )


class TestValidateRlScriptContent(unittest.TestCase):
    _MIN_OK = (
        "%% RL_PARAMS_BEGIN\nrl_w1=1;\n%% RL_PARAMS_END\n"
        + "function res=sim_s(p)\nres=struct('miss',1);\nend\n" * 20
        + "function gf()\nend\nfunction cg()\nend\n"
        + "print_summary(res,KILL_R,'T4');\n"
        + "fprintf('  [%%3d] HIT\\n', 1);\n"
    )

    def test_rejects_endfunction(self):
        bad = self._MIN_OK + "\nendfunction\n"
        ok, reason = validate_rl_script_content(bad)
        self.assertFalse(ok)
        self.assertIn("endfunction", reason)

    def test_rejects_placeholder_sim_s(self):
        bad = (
            "%% RL_PARAMS_BEGIN\nrl_w1=1;\n%% RL_PARAMS_END\n"
            "Placeholder simulation function\n"
        )
        ok, _ = validate_rl_script_content(bad)
        self.assertFalse(ok)

    def test_accepts_rl_params_block(self):
        ok, reason = validate_rl_script_content(self._MIN_OK)
        self.assertTrue(ok, reason)


class TestParseSimStdout(unittest.TestCase):
    _STD_MC = """
  [Run] Hit  Miss(m)  PeakNy(g)  PM(°)   BW(rad/s)  GM(dB)
  [  1] HIT    5.20    10.50    55.0    40.00     8.00
  [  2] MISS  12.30    11.20    52.0    38.00     7.50
──────────────────────────────────
【汇总】T4:高频较大幅度机动
  命中率(miss<10m): 50.0%  SEP: 12.30m
  峰值法向过载: 10.80g  max=11.20g
  俯仰PM: 53.5°  BW: 39.00 rad/s  GM: 7.80 dB
"""

    _LOG06050926_HEADER_ONLY = """
  [Run] Hit  Miss(m)  PeakNy(g)  PM(°)   BW(rad/s)  GM(dB)
  Mean pitch PM: 30.5 deg, BW: 21.12 rad/s, GM: 6.09 dB
Case        Hit %   MeanMiss MeanNy  MeanPM MeanBW MeanGM
"""

    _ENGLISH_CASE_TABLE = """
  [Run] Hit  Miss(m)  PeakNy(g)  PM(°)   BW(rad/s)  GM(dB)
  Mean pitch PM: 45.2 deg, BW: 21.12 rad/s, GM: 6.09 dB
Case        Hit %   MeanMiss MeanNy  MeanPM MeanBW MeanGM
  T4:high maneuver     55.0%    6.30    10.20    45.2    21.12     6.09
"""

    def test_chinese_print_summary(self):
        m = parse_sim_stdout(self._STD_MC)
        self.assertAlmostEqual(m["hit_rate"], 50.0, places=1)
        self.assertAlmostEqual(m["SEP"], 12.30, places=2)
        self.assertAlmostEqual(m.get("peak_ny_max", 0.0), 11.20, places=2)
        self.assertFalse(m.get("_parse_incomplete"))

    def test_log06050926_header_only_marked_incomplete(self):
        m = parse_sim_stdout(self._LOG06050926_HEADER_ONLY)
        self.assertTrue(m.get("_parse_incomplete"))
        self.assertAlmostEqual(m["hit_rate"], 0.0)
        self.assertAlmostEqual(m["SEP"], 50.0)

    def test_english_case_table(self):
        m = parse_sim_stdout(self._ENGLISH_CASE_TABLE)
        self.assertAlmostEqual(m["hit_rate"], 55.0, places=1)
        self.assertAlmostEqual(m["SEP"], 6.30, places=2)
        self.assertAlmostEqual(m["pitch_PM"], 45.2, places=1)
        self.assertFalse(m.get("_parse_incomplete"))

    def test_stdout_has_mc_evidence_per_run(self):
        self.assertTrue(stdout_has_mc_evidence(self._STD_MC))

    def test_stdout_has_mc_evidence_english_table(self):
        self.assertTrue(stdout_has_mc_evidence(self._ENGLISH_CASE_TABLE))

    def test_metrics_look_like_defaults_without_evidence(self):
        m = parse_sim_stdout(self._LOG06050926_HEADER_ONLY)
        self.assertTrue(
            metrics_look_like_parse_defaults(m, self._LOG06050926_HEADER_ONLY)
        )


class TestSelectReturnBestAndReflection(unittest.IsolatedAsyncioTestCase):
    """Regression: log06121335 Ep25 — params/metrics must not be swapped for reflect."""

    def test_select_return_best_order_is_params_then_metrics(self):
        opt = MatlabRLOptimizer(early_stop_use_multi_criteria_best=False)
        opt._best_params = {"w1": 45.0, "zeta1": 0.6, "N_pn": 4.0}
        opt._best_metrics = {
            "hit_rate": 96.0,
            "SEP": 12.41,
            "peak_ny_max": 53.44,
            "pitch_PM": 69.2,
            "pitch_BW": 88.62,
        }
        params, metrics, sel = opt._select_return_best()
        self.assertEqual(sel, "reward")
        self.assertIn("w1", params)
        self.assertIn("hit_rate", metrics)
        self.assertNotIn("hit_rate", params)
        self.assertNotIn("w1", metrics)

    def test_select_return_best_prefers_constrained_when_enabled(self):
        opt = MatlabRLOptimizer(early_stop_use_multi_criteria_best=True)
        opt._best_params = {"w1": 99.0}
        opt._best_metrics = {"hit_rate": 50.0, "SEP": 99.0}
        opt._best_constrained_params = {"w1": 48.0, "N_pn": 3.8}
        opt._best_constrained_metrics = {
            "hit_rate": 100.0,
            "SEP": 4.95,
            "peak_ny_max": 23.08,
            "pitch_PM": 67.0,
            "pitch_BW": 90.0,
        }
        params, metrics, sel = opt._select_return_best()
        self.assertEqual(sel, "constrained")
        self.assertAlmostEqual(metrics["SEP"], 4.95)
        self.assertAlmostEqual(params["N_pn"], 3.8)

    async def test_judgment_reflect_uses_sim_metrics_not_rl_params(self):
        from multi_agent.integration.judgment_agent import JudgmentAgent

        agent = JudgmentAgent()
        task = (
            "T4: RUN_CASE='T'; SUB_IDX=4; 命中率>=92%; SEP<=7m; "
            "PeakNy<=20g; PM在45-70度; BW在20-85rad/s"
        )
        opt = MatlabRLOptimizer(early_stop_use_multi_criteria_best=False)
        opt._best_params = {"w1": 45.0, "N_pn": 4.0}
        opt._best_metrics = {
            "hit_rate": 96.0,
            "SEP": 12.41,
            "peak_ny_max": 53.44,
            "pitch_PM": 69.2,
            "pitch_BW": 88.62,
        }
        _reflect_params, _reflect_metrics, _sel = opt._select_return_best()
        out = await agent.reflect(
            task,
            {
                "parameters": _reflect_params,
                "metrics": _reflect_metrics,
            },
        )
        self.assertTrue(out["needs_optimization"])
        suggestion = out["suggestion"]
        self.assertIn("12.41", suggestion)
        self.assertNotIn("999.00", suggestion)
        self.assertNotIn("命中率 0.0%", suggestion)

    async def test_params_as_metrics_shows_sentinel_regression(self):
        """Document the log06121335 failure mode when unpack order is wrong."""
        from multi_agent.integration.judgment_agent import JudgmentAgent

        agent = JudgmentAgent()
        task = "命中率>=92%; SEP<=7m; PM在45-70度; BW在20-85rad/s"
        out = await agent.reflect(
            task,
            {
                "parameters": {"w1": 45.0},
                "metrics": {"w1": 45.0, "N_pn": 4.0},  # swapped bug
            },
        )
        self.assertIn("999.00", out["suggestion"])
        self.assertIn("0.0%", out["suggestion"])


if __name__ == "__main__":
    unittest.main()
