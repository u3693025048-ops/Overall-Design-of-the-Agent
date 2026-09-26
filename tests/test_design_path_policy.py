#!/usr/bin/env python3
"""Tests for T4 design-path policy."""

import unittest

from multi_agent.integration.design_path_policy import (
    apply_reflection_policy,
    gf_bodies_differ,
    normalize_planner_mode,
    resolve_next_action,
    resolve_planner_fallback_mode,
    resolve_task_mode,
    should_prefer_tune_over_modify,
    should_skip_layer3_param_search,
    should_use_modify_law,
    verify_modify_law_result,
)

_T4_PROMPT = (
    "RUN_CASE='T'; SUB_IDX=4; 命中率>=92%; SEP<=7m; PeakNy<=20g; "
    "PM 45-70; BW 20-85"
)

_GOOD_METRICS = {
    "hit_rate": 100.0,
    "SEP": 5.5,
    "miss_distance": 5.5,
    "pitch_PM": 55.0,
    "pitch_BW": 75.0,
    "peak_ny": 22.0,
    "peak_ny_max": 27.0,
}

_BASE_GF = """
function [N1,N2,G1,G2,gc]=gf(Tx,Ty,Tz,XK,TVx,TVy,TVz,Np)
N1=1; N2=1; G1=1; G2=1; gc=0;
end
"""

_MOD_GF = """
function [N1,N2,G1,G2,gc]=gf(Tx,Ty,Tz,XK,TVx,TVy,TVz,Np)
a_T_perp = 1; N1=2; N2=2; G1=1; G2=1; gc=0;
end
"""


class TestDesignPathPolicy(unittest.TestCase):
    def test_fallback_never_reuse_on_optimization_task(self):
        mode = resolve_planner_fallback_mode(task_prompt=_T4_PROMPT)
        self.assertNotEqual(mode, "REUSE_HISTORY")
        self.assertIn(mode, ("TUNE_PARAMS", "MODIFY_LAW"))

    def test_override_reuse_history(self):
        mode = normalize_planner_mode(
            "REUSE_HISTORY",
            task_prompt=_T4_PROMPT,
        )
        self.assertNotEqual(mode, "REUSE_HISTORY")

    def test_should_modify_after_tune_stalls(self):
        hist = [
            {"task_mode": "TUNE_PARAMS", "peak_ny": 22.0, "peak_ny_max": 27.0, "hit_rate": 100.0, "SEP": 5.0},
            {"task_mode": "TUNE_PARAMS", "peak_ny": 21.5, "peak_ny_max": 26.0, "hit_rate": 100.0, "SEP": 5.2},
        ]
        self.assertTrue(
            should_use_modify_law(
                task_prompt=_T4_PROMPT,
                metrics=_GOOD_METRICS,
                optimization_history=hist,
            )
        )

    def test_resolve_next_action_modify_law(self):
        hist = [
            {"task_mode": "TUNE_PARAMS", "peak_ny_max": 27.0},
            {"task_mode": "TUNE_PARAMS", "peak_ny_max": 26.0},
        ]
        action = resolve_next_action(
            reflection_next_action="tune_params",
            task_prompt=_T4_PROMPT,
            metrics=_GOOD_METRICS,
            optimization_history=hist,
        )
        self.assertEqual(action, "modify_law")

    def test_apply_reflection_policy_peak_ny_max(self):
        out = apply_reflection_policy(
            {"needs_optimization": False, "next_action": "done", "suggestion": ""},
            task_prompt=_T4_PROMPT,
            metrics=_GOOD_METRICS,
            optimization_history=[],
        )
        self.assertTrue(out["needs_optimization"])
        self.assertEqual(out["next_action"], "modify_law")

    def test_should_modify_on_layer2_4of5(self):
        self.assertTrue(
            should_use_modify_law(
                task_prompt=_T4_PROMPT,
                metrics=_GOOD_METRICS,
                optimization_history=[],
            )
        )

    def test_forced_tune_does_not_override_planner_modify_law(self):
        mode = resolve_task_mode(
            planner_mode="MODIFY_LAW",
            task_prompt=_T4_PROMPT,
            optimization_history=[],
            last_metrics=_GOOD_METRICS,
            forced_next_action="tune_params",
        )
        self.assertEqual(mode, "MODIFY_LAW")

    def test_forced_tune_does_not_override_4of5_modify_law_policy(self):
        mode = resolve_task_mode(
            planner_mode="TUNE_PARAMS",
            task_prompt=_T4_PROMPT,
            optimization_history=[],
            last_metrics=_GOOD_METRICS,
            forced_next_action="tune_params",
        )
        self.assertEqual(mode, "MODIFY_LAW")

    def test_skip_layer3_when_only_peak_fails(self):
        self.assertTrue(
            should_skip_layer3_param_search(
                _GOOD_METRICS, _T4_PROMPT, optimization_history=[],
            )
        )

    def test_no_skip_after_modify_law_near_miss(self):
        hist = [{"task_mode": "MODIFY_LAW", "peak_ny_max": 21.0}]
        metrics = dict(_GOOD_METRICS)
        metrics["peak_ny_max"] = 20.99
        metrics["peak_ny"] = 18.0
        self.assertFalse(
            should_skip_layer3_param_search(metrics, _T4_PROMPT, hist)
        )
        self.assertFalse(
            should_use_modify_law(
                task_prompt=_T4_PROMPT,
                metrics=metrics,
                optimization_history=hist,
            )
        )

    def test_fallback_modify_law_on_layer2_metrics(self):
        mode = resolve_planner_fallback_mode(
            task_prompt=_T4_PROMPT,
            last_metrics=_GOOD_METRICS,
        )
        self.assertEqual(mode, "MODIFY_LAW")

    def test_gf_verify(self):
        self.assertFalse(gf_bodies_differ(_BASE_GF, _BASE_GF))
        self.assertTrue(gf_bodies_differ(_BASE_GF, _MOD_GF))
        ok, msg = verify_modify_law_result(_MOD_GF, _BASE_GF)
        self.assertTrue(ok, msg)

    def test_prefer_tune_after_modify_catastrophic(self):
        hist = [{"task_mode": "MODIFY_LAW", "peak_ny_max": 61.0}]
        bad_metrics = {
            "hit_rate": 41.0,
            "SEP": 7.58,
            "pitch_PM": 62.7,
            "pitch_BW": 36.8,
            "peak_ny_max": 7.46,
        }
        self.assertTrue(
            should_prefer_tune_over_modify(
                task_prompt=_T4_PROMPT,
                metrics=bad_metrics,
                optimization_history=hist,
            )
        )
        mode = normalize_planner_mode(
            "TUNE_PARAMS",
            task_prompt=_T4_PROMPT,
            optimization_history=hist,
            last_metrics=bad_metrics,
            reflection_feedback="next_action=tune_params",
        )
        self.assertEqual(mode, "TUNE_PARAMS")

    def test_near_miss_peak_prefers_tune(self):
        metrics = dict(_GOOD_METRICS)
        metrics["peak_ny"] = 22.88
        metrics["peak_ny_max"] = 25.0
        hist = [{"task_mode": "MODIFY_LAW"}]
        self.assertTrue(
            should_prefer_tune_over_modify(
                task_prompt=_T4_PROMPT,
                metrics=metrics,
                optimization_history=hist,
            )
        )


if __name__ == "__main__":
    unittest.main()
