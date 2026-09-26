#!/usr/bin/env python3
"""Tests for ParameterExperience tool argument normalization."""

import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock

from multi_agent.tools.parameter_experience_tool import (
    ParameterExperienceBestTool,
    ParameterExperienceSearchTool,
    _coerce_pe_search_query,
)


class TestCoercePeSearchQuery(unittest.TestCase):
    def test_sub_query_and_mode_search(self):
        d = _coerce_pe_search_query(sub_query="T4 N_pn w1")
        self.assertEqual(d["task"], "T4 N_pn w1")


class TestParameterExperienceTools(unittest.IsolatedAsyncioTestCase):
    async def test_search_accepts_sub_query_and_mode(self):
        tool = ParameterExperienceSearchTool()
        pe = MagicMock()
        pe.retrieve_similar = AsyncMock(return_value=[])
        tool.set_parameter_experience(pe)

        out = await tool.execute(
            sub_query="APN saturation",
            mode="law",
            top_k=3,
        )

        self.assertEqual(out["status"], "success")
        self.assertEqual(out["query"]["search_mode"], "law")

    async def test_best_accepts_sub_query(self):
        tool = ParameterExperienceBestTool()
        pe = MagicMock()
        pe.retrieve_best = AsyncMock(return_value=[])
        tool.set_parameter_experience(pe)

        out = await tool.execute(sub_query="T4 hit rate bandwidth", top_k=2)

        self.assertEqual(out["status"], "success")
        self.assertIn("T4", out["task_context"]["task"])


if __name__ == "__main__":
    unittest.main()
