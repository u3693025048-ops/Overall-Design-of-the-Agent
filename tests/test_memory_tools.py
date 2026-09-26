#!/usr/bin/env python3
"""Tests for agent memory tool argument aliases."""

import json
import unittest
from unittest.mock import MagicMock

from multi_agent.tools.memory_tools import AgentMemoryRecallTool


class TestAgentMemoryRecall(unittest.IsolatedAsyncioTestCase):
    async def test_query_alias(self):
        tool = AgentMemoryRecallTool()
        mem = MagicMock()
        mem.recall.return_value = "stored-value"
        tool.set_memory(mem)

        out = json.loads(await tool.execute(query="T4_MODIFY_LAW_result"))

        self.assertEqual(out["status"], "success")
        mem.recall.assert_called_once_with("T4_MODIFY_LAW_result")


if __name__ == "__main__":
    unittest.main()
