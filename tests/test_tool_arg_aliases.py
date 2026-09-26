#!/usr/bin/env python3
"""Hermes tool argument alias normalization across tools."""

import json
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from multi_agent.tools.judge_requirements_tool import JudgeRequirementsTool
from multi_agent.tools.memory_tools import AgentMemoryRememberTool
from multi_agent.tools.rag_tool import RAGAgentTool
from multi_agent.tools.syntax_check_tool import SyntaxCheckMATLABTool
from multi_agent.tools.tool_arg_aliases import (
    coerce_script_path,
    coerce_task_prompt,
)


class TestToolArgAliases(unittest.TestCase):
    def test_script_path_aliases(self):
        self.assertEqual(
            coerce_script_path(path="guidance_output/foo.m"),
            "guidance_output/foo.m",
        )

    def test_task_prompt_aliases(self):
        self.assertEqual(
            coerce_task_prompt(prompt="命中率>=92%"),
            "命中率>=92%",
        )


class TestToolExecuteAliases(unittest.IsolatedAsyncioTestCase):
    async def test_rag_query_keyword(self):
        kb = MagicMock()
        kb.retrieve = AsyncMock(return_value=[])
        tool = RAGAgentTool()
        tool.set_rag_kb(kb)
        with patch.object(tool, "_plan", AsyncMock(return_value=["T4"])):
            out = await tool.execute(keyword="T4 APN")
        self.assertEqual(out["status"], "success")

    async def test_judge_task_description_alias(self):
        judge = JudgeRequirementsTool()
        judge.set_task_prompt(
            "命中率>=92%; SEP<=7m; PeakNy<=20g; PM在45-70度; BW在20-85rad/s"
        )
        out = json.loads(
            await judge.execute(
                task_description=judge._task_prompt_cache,
                metrics={"hit_rate": 50.0, "SEP": 20.0, "pitch_PM": 50.0, "pitch_BW": 50.0},
            )
        )
        self.assertFalse(out["satisfied"])

    async def test_syntax_check_path_alias(self):
        tool = SyntaxCheckMATLABTool()
        with patch("os.path.isfile", return_value=False):
            out = json.loads(await tool.execute(path="missing.m"))
        self.assertFalse(out["valid"])
        self.assertIn("missing.m", out["fixed_path"])

    async def test_remember_memory_id_alias(self):
        tool = AgentMemoryRememberTool()
        mem = MagicMock()
        mem.remember.return_value = "ok"
        tool.set_memory(mem)
        out = json.loads(await tool.execute(memory_id="k1", value="v1"))
        self.assertEqual(out["status"], "success")
        mem.remember.assert_called_once_with("k1", "v1")


if __name__ == "__main__":
    unittest.main()
