#!/usr/bin/env python3
"""Tests for T4 low-risk policy module."""

import unittest

from multi_agent.integration.t4_low_risk import (
    GF_APN_T4_MINIMAL,
    apply_deterministic_t4_modify_law,
    apply_t4_deterministic_gf_if_enabled,
    should_skip_physics_bounds_probe,
    apply_reflection_t4_overrides,
    clamp_autopilot_params,
    enforce_expert_suggestion,
    is_t4_mission,
    layer3_max_episodes_cap,
    pre_cls_param_fixup,
    replace_gf_function,
    resolve_initial_auto_params,
    rule_seed_expert_params,
    should_block_tune_after_bad_modify,
    should_force_expert_after_modify,
    should_use_rule_first_expert_round,
    use_deterministic_apn,
)

_T4_MC = "RUN_CASE='T'; SUB_IDX=4;"
_T4_PROMPT = (
    "RUN_CASE='T'; SUB_IDX=4; "
    "命中率 >= 92%; SEP脱靶量 <= 7 m; PeakNy峰值法向过载 <= 20 g; "
    "PM相位裕度 在 45°~ 70° 范围内; BW带宽 在 20 ~ 85 rad/s 范围内"
)

_BASE_SCRIPT = """
function [N1,N2,G1,G2,gc]=gf(Tx,Ty,Tz,XK,TVx,TVy,TVz,Np)
N1=1; N2=1; G1=1; G2=1; gc=0;
end

function out=cg(x)
out=0;
end
"""


class TestT4LowRisk(unittest.TestCase):
    def test_is_t4_from_conditions(self):
        self.assertTrue(is_t4_mission(_T4_MC, _T4_PROMPT))
        self.assertFalse(is_t4_mission("RUN_CASE='T'; SUB_IDX=1;", ""))

    def test_is_t4_from_prompt(self):
        self.assertTrue(is_t4_mission("", "分析 T4 工况高频大幅机动"))

    def test_is_t4_from_prompt_sub_idx(self):
        self.assertTrue(
            is_t4_mission("", "RUN_CASE='T'; SUB_IDX=4; PeakNy<=20g"),
        )

    def test_use_deterministic_apn(self):
        self.assertTrue(use_deterministic_apn(_T4_MC, _T4_PROMPT))

    def test_clamp_autopilot_ceiling(self):
        out = clamp_autopilot_params({"w1": 60.0, "N_pn": 6.0})
        self.assertLessEqual(out["w1"], 45.0)
        self.assertLessEqual(out["N_pn"], 4.5)

    def test_resolve_initial_conservative_for_t4(self):
        out = resolve_initial_auto_params(
            {"w1": 50.0},
            task_prompt=_T4_PROMPT,
            mission_conditions=_T4_MC,
        )
        self.assertLessEqual(out["w1"], 45.0)
        self.assertAlmostEqual(out["N_pn"], 3.5, places=1)

    def test_replace_gf_function(self):
        patched = replace_gf_function(_BASE_SCRIPT, GF_APN_T4_MINIMAL)
        self.assertIn("T4 minimal APN", patched)
        self.assertIn("function out=cg", patched)
        self.assertNotIn("N1=1; N2=1", patched)

    def test_apply_deterministic_modify_law(self):
        patched = apply_deterministic_t4_modify_law(_BASE_SCRIPT)
        self.assertIn("tanh", patched)
        self.assertIn("ny_lim=20", patched)

    def test_apply_t4_deterministic_gf_if_enabled(self):
        out, applied = apply_t4_deterministic_gf_if_enabled(
            _BASE_SCRIPT, _T4_MC, _T4_PROMPT,
        )
        self.assertTrue(applied)
        self.assertIn("T4 minimal APN", out)

    def test_skip_physics_bounds_for_deterministic_apn(self):
        patched = apply_deterministic_t4_modify_law(_BASE_SCRIPT)
        self.assertTrue(
            should_skip_physics_bounds_probe(patched, _T4_MC, _T4_PROMPT)
        )
        self.assertFalse(
            should_skip_physics_bounds_probe(_BASE_SCRIPT, _T4_MC, _T4_PROMPT)
        )

    def test_block_tune_after_bad_modify_peak(self):
        hist = [{"task_mode": "MODIFY_LAW", "peak_ny": 61.0}]
        blocked, msg = should_block_tune_after_bad_modify(
            {"peak_ny": 61.0, "peak_ny_max": 80.0, "hit_rate": 100.0},
            _T4_PROMPT,
            hist,
        )
        self.assertTrue(blocked)
        self.assertIn("禁止 TUNE", msg)

    def test_no_block_tune_when_modify_ok(self):
        hist = [{"task_mode": "MODIFY_LAW"}]
        blocked, _ = should_block_tune_after_bad_modify(
            {"peak_ny": 22.0, "peak_ny_max": 55.0, "hit_rate": 95.0},
            _T4_PROMPT,
            hist,
        )
        self.assertFalse(blocked)

    def test_check_modify_law_closed_loop_gate_ignores_max(self):
        from multi_agent.integration.t4_low_risk import check_modify_law_closed_loop_gate
        ok, msg = check_modify_law_closed_loop_gate(
            {"peak_ny": 27.36, "peak_ny_max": 40.71, "hit_rate": 96.0},
            _T4_PROMPT,
        )
        self.assertTrue(ok, msg)
        bad, msg = check_modify_law_closed_loop_gate(
            {"peak_ny": 55.0, "peak_ny_max": 80.0, "hit_rate": 96.0},
            _T4_PROMPT,
        )
        self.assertFalse(bad)
        self.assertIn("PeakNy(均值)", msg)

    def test_force_expert_after_modify(self):
        hist = [{"task_mode": "MODIFY_LAW"}]
        self.assertTrue(should_force_expert_after_modify(hist, _T4_PROMPT))
        self.assertFalse(should_force_expert_after_modify([], _T4_PROMPT))

    def test_layer3_episode_cap(self):
        hist = [{"task_mode": "MODIFY_LAW"}]
        self.assertEqual(layer3_max_episodes_cap(hist, 100), 0)
        self.assertEqual(layer3_max_episodes_cap([], 100), 100)

    def test_should_allow_ppo_after_expert(self):
        from multi_agent.integration.t4_low_risk import should_allow_ppo_after_expert
        hist = [{"task_mode": "MODIFY_LAW"}]
        ok, _ = should_allow_ppo_after_expert(
            {"peak_ny": 24.0, "peak_ny_max": 40.0, "hit_rate": 96.0, "SEP": 11.32}, _T4_PROMPT, hist,
        )
        self.assertFalse(ok)
        ok_far, _ = should_allow_ppo_after_expert(
            {"peak_ny": 24.0, "peak_ny_max": 40.0, "hit_rate": 96.0, "SEP": 22.0}, _T4_PROMPT, hist,
        )
        self.assertFalse(ok_far)
        bad, msg = should_allow_ppo_after_expert(
            {"peak_ny": 55.0, "peak_ny_max": 80.0, "hit_rate": 96.0}, _T4_PROMPT, hist,
        )
        self.assertFalse(bad)
        self.assertIn("跳过 PPO", msg)

    def test_should_store_pe_memory(self):
        from multi_agent.integration.t4_low_risk import should_store_pe_memory
        good = {
            "hit_rate": 100.0,
            "SEP": 5.0,
            "peak_ny": 18.0,
            "peak_ny_max": 19.0,
            "pitch_PM": 60.0,
            "pitch_BW": 50.0,
        }
        ok, _ = should_store_pe_memory(good, _T4_PROMPT)
        self.assertTrue(ok)
        bad, _ = should_store_pe_memory({"peak_ny": 61.0, "peak_ny_max": 80.0, "hit_rate": 100.0}, _T4_PROMPT)
        self.assertFalse(bad)
        over_peak = {
            "hit_rate": 100.0,
            "SEP": 5.0,
            "peak_ny": 18.0,
            "peak_ny_max": 22.0,
            "pitch_PM": 60.0,
            "pitch_BW": 50.0,
        }
        ok, msg = should_store_pe_memory(over_peak, _T4_PROMPT)
        self.assertTrue(ok)
        # mean OK, max over task mean limit — no longer blocks PE when mean satisfies
        borderline_max = {
            "hit_rate": 100.0,
            "SEP": 5.0,
            "peak_ny": 18.0,
            "peak_ny_max": 21.5,
            "pitch_PM": 60.0,
            "pitch_BW": 50.0,
        }
        ok2, _ = should_store_pe_memory(borderline_max, _T4_PROMPT)
        self.assertTrue(ok2)

    def test_reflection_override_bad_modify(self):
        hist = [{"task_mode": "MODIFY_LAW"}]
        out = apply_reflection_t4_overrides(
            {"next_action": "tune_params", "suggestion": "继续调 w"},
            {"peak_ny": 55.0, "peak_ny_max": 80.0, "hit_rate": 100.0},
            _T4_PROMPT,
            hist,
        )
        self.assertEqual(out["next_action"], "modify_law")

    def test_rule_seed_r1_low_bw(self):
        metrics = {"pitch_BW": 11.0, "pitch_PM": 72.0}
        seed = rule_seed_expert_params(
            metrics,
            1,
            {},
            task_prompt=_T4_PROMPT,
            mission_conditions=_T4_MC,
        )
        self.assertIsNotNone(seed)
        self.assertLessEqual(seed["w1"], 40.0)
        self.assertGreaterEqual(seed["w1"], 36.0)

    def test_should_use_rule_first_r1(self):
        metrics = {"pitch_BW": 11.0, "pitch_PM": 72.0}
        self.assertTrue(
            should_use_rule_first_expert_round(
                1, metrics, task_prompt=_T4_PROMPT, mission_conditions=_T4_MC,
            )
        )
        self.assertFalse(
            should_use_rule_first_expert_round(
                2, metrics, task_prompt=_T4_PROMPT, mission_conditions=_T4_MC,
            )
        )

    def test_enforce_expert_rejects_max_w_trap(self):
        metrics = {"pitch_BW": 11.0, "pitch_PM": 72.0}
        out = enforce_expert_suggestion(
            {"w1": 44.0, "zeta1": 0.65, "tao1": 0.12, "w2": 35, "zeta2": 0.7,
             "tao2": 0.1, "w3": 25, "zeta3": 0.9, "N_pn": 4.0},
            metrics,
            round_i=1,
            task_prompt=_T4_PROMPT,
            mission_conditions=_T4_MC,
        )
        self.assertLessEqual(out["w1"], 40.0)

    def test_pre_cls_fixup_high_bw(self):
        params = {"w1": 44.0, "zeta1": 0.65, "N_pn": 4.0}
        metrics = {"pitch_BW": 103.0, "pitch_PM": 55.0}
        fixed = pre_cls_param_fixup(
            params,
            metrics,
            task_prompt=_T4_PROMPT,
            mission_conditions=_T4_MC,
        )
        self.assertLess(fixed["w1"], 44.0)
        self.assertGreaterEqual(fixed["w1"], 34.0)


if __name__ == "__main__":
    unittest.main()
