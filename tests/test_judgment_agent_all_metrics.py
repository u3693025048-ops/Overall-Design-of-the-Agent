#!/usr/bin/env python3
"""JudgmentAgent: all metrics must pass; no partial / optimization-only NG."""

import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from multi_agent.integration.judgment_agent import JudgmentAgent

_PROMPT = Path("prompt.txt").read_text(encoding="utf-8")

_BASE_OK = {
    "hit_rate": 100.0,
    "SEP": 4.98,
    "peak_ny": 18.88,
    "peak_ny_max": 20.87,
    "pitch_PM": 67.8,
    "pitch_BW": 75.0,
}


class TestJudgmentAgentAllMetrics(unittest.IsolatedAsyncioTestCase):
    async def _judge(self, metrics, prompt=_PROMPT):
        reflection = MagicMock()
        reflection.llm = None
        agent = JudgmentAgent(reflection_agent=reflection, task_prompt=prompt)
        return await agent.judge(metrics, task_prompt=prompt)

    async def test_all_ok_satisfied(self):
        result = await self._judge(dict(_BASE_OK))
        self.assertTrue(result.satisfied)
        self.assertEqual(result.next_action, "done")
        self.assertFalse(any("[NG]" in r for r in result.reasons))

    async def test_single_bw_ng_not_satisfied(self):
        m = dict(_BASE_OK)
        m["pitch_BW"] = 110.5
        result = await self._judge(m)
        self.assertFalse(result.satisfied)
        self.assertEqual(result.next_action, "tune_params")
        self.assertTrue(any("BW" in r and "[NG]" in r for r in result.reasons))
        self.assertTrue(any("[OK]" in r for r in result.reasons))

    async def test_single_sep_ng_not_satisfied(self):
        m = dict(_BASE_OK)
        m["SEP"] = 8.0
        result = await self._judge(m)
        self.assertFalse(result.satisfied)
        self.assertTrue(any("SEP" in r and "[NG]" in r for r in result.reasons))

    async def test_peak_mean_ng_not_satisfied(self):
        m = dict(_BASE_OK)
        m["peak_ny"] = 20.5
        result = await self._judge(m)
        self.assertFalse(result.satisfied)
        self.assertTrue(any("平均值" in r and "[NG]" in r for r in result.reasons))

    async def test_peak_max_over_limit_still_satisfied_if_mean_ok(self):
        m = dict(_BASE_OK)
        m["peak_ny_max"] = 21.5
        result = await self._judge(m)
        self.assertTrue(result.satisfied)
        self.assertFalse(any("最大值" in r and "[NG]" in r for r in result.reasons))

    async def test_llm_merge_incomplete_reqs_still_checks_bw(self):
        """Incomplete LLM reqs must not skip BW gate (log06151449)."""
        reflection = MagicMock()
        reflection.llm = MagicMock()
        reflection.llm.ainvoke = AsyncMock(
            return_value=MagicMock(
                content=(
                    '{"hit_rate_min":92,"sep_max":7,"peak_ny_max":20,'
                    '"pm_min":45,"pm_max":70,"bw_min":20}'
                )
            )
        )
        agent = JudgmentAgent(reflection_agent=reflection, task_prompt=_PROMPT)
        m = dict(_BASE_OK)
        m["pitch_BW"] = 110.5
        result = await agent.judge(m, task_prompt=_PROMPT)
        self.assertFalse(result.satisfied)
        self.assertTrue(any("BW" in r and "[NG]" in r for r in result.reasons))


if __name__ == "__main__":
    unittest.main()
