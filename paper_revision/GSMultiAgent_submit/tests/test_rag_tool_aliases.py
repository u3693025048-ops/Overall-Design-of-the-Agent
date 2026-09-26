#!/usr/bin/env python3
"""RAG tool alias argument normalization."""

import unittest
from unittest.mock import AsyncMock, MagicMock

from multi_agent.tools.rag_tool import (
    RAGExpandTool,
    RAGRetrievalTool,
    _coerce_expand_doc_id,
    _coerce_retrieve_query,
)


class TestRAGToolAliases(unittest.TestCase):
    def test_coerce_retrieve_query_keyword(self):
        self.assertEqual(
            _coerce_retrieve_query(keyword="T4 机动"),
            "T4 机动",
        )

    def test_coerce_expand_doc_id_keyword(self):
        self.assertEqual(
            _coerce_expand_doc_id(keyword="file:kb/doc.md"),
            "file:kb/doc.md",
        )


class TestRAGExpandKeyword(unittest.IsolatedAsyncioTestCase):
    async def test_expand_accepts_keyword_without_type_error(self):
        kb = MagicMock()
        kb.get_document = AsyncMock(return_value={"content": "hello", "metadata": {}})
        tool = RAGExpandTool()
        tool.set_rag_kb(kb)

        out = await tool.execute(keyword="file:test.md")
        self.assertEqual(out["status"], "success")
        self.assertEqual(out["content"], "hello")
        kb.get_document.assert_awaited_with("file:test.md")

    async def test_expand_keyword_falls_back_to_retrieve(self):
        kb = MagicMock()
        kb.get_document = AsyncMock(side_effect=[None, {"content": "body", "metadata": {}}])
        kb.retrieve = AsyncMock(return_value=[{"doc_id": "file:found.md", "score": 0.9}])
        tool = RAGExpandTool()
        tool.set_rag_kb(kb)

        out = await tool.execute(keyword="monte_carlo_single")
        self.assertEqual(out["status"], "success")
        self.assertEqual(out["doc_id"], "file:found.md")
        kb.retrieve.assert_awaited_once()


class TestRAGRetrieveKeyword(unittest.IsolatedAsyncioTestCase):
    async def test_retrieve_accepts_keyword(self):
        kb = MagicMock()
        kb.retrieve = AsyncMock(return_value=[])
        tool = RAGRetrievalTool()
        tool.set_rag_kb(kb)

        out = await tool.execute(keyword="APN 制导")
        self.assertEqual(out["status"], "success")
        kb.retrieve.assert_awaited_once_with(query="APN 制导", top_k=5)


if __name__ == "__main__":
    unittest.main()
