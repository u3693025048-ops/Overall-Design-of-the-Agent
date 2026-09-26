#!/usr/bin/env python3
"""Regression: all metrics must be parsed and satisfied (log06151449)."""

import unittest
from pathlib import Path

from multi_agent.integration.design_path_policy import (
    apply_reflection_policy,
    requirements_from_prompt,
)
from multi_agent.integration.judgment_agent import (
    _extract_numeric_requirements,
    _rule_based_check,
    all_requirements_satisfied,
    requirements_complete,
)
from multi_agent.rl.metric_constraints import constraints_satisfied


_PROMPT = Path("prompt.txt").read_text(encoding="utf-8")

# log06151449 CLS best — BW over limit, falsely marked done
_LOG_METRICS = {
    "hit_rate": 100.0,
    "SEP": 4.98,
    "peak_ny": 18.88,
    "peak_ny_max": 20.87,
    "pitch_PM": 67.8,
    "pitch_BW": 110.5,
}


class TestRequirementParsing(unittest.TestCase):
    def test_prompt_parses_pm_and_bw(self):
        reqs = _extract_numeric_requirements(_PROMPT)
        self.assertIn("pm_min", reqs)
        self.assertIn("pm_max", reqs)
        self.assertIn("bw_min", reqs)
        self.assertIn("bw_max", reqs)
        self.assertEqual(reqs["pm_min"], 45.0)
        self.assertEqual(reqs["pm_max"], 70.0)
        self.assertEqual(reqs["bw_min"], 20.0)
        self.assertEqual(reqs["bw_max"], 85.0)

    def test_requirements_complete_for_prompt(self):
        reqs = requirements_from_prompt(_PROMPT)
        self.assertTrue(requirements_complete(reqs, _PROMPT))

    def test_log06151449_bw_fail_not_satisfied(self):
        reqs = requirements_from_prompt(_PROMPT)
        self.assertFalse(all_requirements_satisfied(_LOG_METRICS, reqs, _PROMPT))
        self.assertFalse(constraints_satisfied(_LOG_METRICS, reqs, _PROMPT))
        ok, reasons = _rule_based_check(_LOG_METRICS, reqs)
        self.assertFalse(ok)
        ng = [r for r in reasons if "[NG]" in r]
        self.assertTrue(any("BW" in r for r in ng))
        self.assertFalse(any("最大值" in r and "[NG]" in r for r in reasons))

    def test_apply_reflection_policy_overrides_llm_false_done(self):
        """LLM needs_optimization=false cannot pass when BW exceeds limit."""
        out = apply_reflection_policy(
            {
                "needs_optimization": False,
                "next_action": "done",
                "suggestion": "BW超标但仍建议完成",
            },
            task_prompt=_PROMPT,
            metrics=_LOG_METRICS,
        )
        self.assertTrue(out["needs_optimization"])
        self.assertNotEqual(out["next_action"], "done")

    def test_task_fully_satisfied_log06151449(self):
        from multi_agent.integration.design_path_policy import task_fully_satisfied

        self.assertFalse(task_fully_satisfied(_LOG_METRICS, _PROMPT))
        good = dict(_LOG_METRICS)
        good["pitch_BW"] = 75.0
        good["peak_ny_max"] = 20.0
        good["peak_ny"] = 18.0
        self.assertTrue(task_fully_satisfied(good, _PROMPT))

    def test_incomplete_llm_reqs_do_not_drop_bw_max(self):
        """LLM reqs missing bw_max must not override regex pair (log06151449)."""
        from multi_agent.integration.judgment_agent import (
            merge_requirements,
            resolve_task_requirements,
        )

        regex_reqs = requirements_from_prompt(_PROMPT)
        llm_incomplete = {
            "hit_rate_min": 92.0,
            "sep_max": 7.0,
            "peak_ny_max": 20.0,
            "pm_min": 45.0,
            "pm_max": 70.0,
            "bw_min": 20.0,
        }
        merged = merge_requirements(regex_reqs, llm_incomplete)
        self.assertIn("bw_max", merged)
        self.assertEqual(merged["bw_max"], 85.0)
        self.assertFalse(all_requirements_satisfied(_LOG_METRICS, merged, _PROMPT))

        class _FakeJA:
            _reqs = llm_incomplete

        resolved = resolve_task_requirements(_PROMPT, _FakeJA())
        self.assertIn("bw_max", resolved)
        from multi_agent.integration.design_path_policy import task_fully_satisfied

        self.assertFalse(task_fully_satisfied(_LOG_METRICS, _PROMPT, resolved))

    def test_all_metrics_pass_when_in_range(self):
        good = dict(_LOG_METRICS)
        good["pitch_BW"] = 75.0
        good["peak_ny_max"] = 20.0
        reqs = requirements_from_prompt(_PROMPT)
        self.assertTrue(all_requirements_satisfied(good, reqs, _PROMPT))


if __name__ == "__main__":
    unittest.main()
