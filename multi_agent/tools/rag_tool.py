#!/usr/bin/env python3
"""
RAG Tools for Hermes Agent
Provides knowledge retrieval and indexing capabilities
"""

import asyncio
import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple, Type
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ToolConfig:
    name: str
    description: str
    input_schema: Dict[str, Any]


class RAGToolMixin:
    """Mixin class to add RAG capabilities to tools"""

    def __init__(self, *args, **kwargs):
        self.rag_kb = None
        super().__init__(*args, **kwargs)

    def set_rag_kb(self, rag_kb) -> None:
        """Set the RAG knowledge base"""
        self.rag_kb = rag_kb


# Content length limits for each detail level
_DETAIL_CHARS: Dict[str, int] = {
    "summary": 200,   # title + first sentence — for fast triage
    "snippet": 600,   # paragraph-level — enough to judge relevance
    "full":    2000,  # full chunk (current default)
}


from multi_agent.tools.tool_arg_aliases import (
    coerce_rag_query as _coerce_retrieve_query,
    coerce_script_path as _coerce_script_path,
    coerce_task_prompt as _coerce_task_prompt,
)


def _coerce_expand_doc_id(
    doc_id: Any = None,
    *,
    keyword: Any = None,
    query: Any = None,
    document_id: Any = None,
    file_id: Any = None,
    path: Any = None,
    id: Any = None,
    **_: Any,
) -> str:
    """Normalize LLM/Hermes alias args for rag_expand."""
    for raw in (doc_id, document_id, file_id, path, keyword, query, id):
        if raw is None:
            continue
        text = str(raw).strip()
        if text:
            return text
    return ""


class RAGRetrievalTool(RAGToolMixin):
    """Tool for retrieving knowledge from RAG knowledge base (progressive disclosure)."""

    name = "rag_retrieve"
    description = (
        "从知识库检索相关文档。支持渐进式披露：\n"
        "  detail_level='summary'  — 仅返回前 200 字摘要，用于快速过滤候选结果；\n"
        "  detail_level='snippet'  — 返回前 600 字片段，足以判断相关性（默认）；\n"
        "  detail_level='full'     — 返回前 2000 字完整内容。\n"
        "返回结果包含 doc_id，可传给 rag_expand 进一步获取指定文档的全文。"
    )

    input_schema = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "搜索查询，用于检索相关知识",
            },
            "top_k": {
                "type": "integer",
                "description": "最多返回结果数",
                "default": 5,
            },
            "detail_level": {
                "type": "string",
                "enum": ["summary", "snippet", "full"],
                "description": "内容披露层级：summary=200字摘要 / snippet=600字片段 / full=2000字全文",
                "default": "snippet",
            },
        },
        "required": ["query"],
    }

    async def execute(
        self,
        query: str = "",
        top_k: int = 5,
        detail_level: str = "snippet",
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Execute RAG retrieval with progressive disclosure."""
        query = _coerce_retrieve_query(query, **kwargs)
        if not query:
            return {"status": "error", "message": "query (or keyword/sub_query) is required"}
        if not self.rag_kb:
            return {"status": "error", "message": "RAG knowledge base not initialized"}

        max_chars = _DETAIL_CHARS.get(detail_level, _DETAIL_CHARS["snippet"])

        try:
            results = await self.rag_kb.retrieve(query=query, top_k=top_k)

            return {
                "status": "success",
                "query": query,
                "detail_level": detail_level,
                "results_count": len(results),
                "hint": (
                    "若需某条文档完整内容，请用 rag_expand(doc_id=...) 展开。"
                    if detail_level != "full" else ""
                ),
                "results": [
                    {
                        "doc_id":   r.get("doc_id", ""),
                        "score":    round(r.get("score", 0.0), 4),
                        "metadata": r.get("metadata", {}),
                        "content":  r.get("content", "")[:max_chars],
                        "truncated": len(r.get("content", "")) > max_chars,
                    }
                    for r in results
                ],
            }
        except Exception as e:
            logger.error(f"RAG retrieval failed: {e}")
            return {"status": "error", "message": str(e)}


class RAGExpandTool(RAGToolMixin):
    """Expand a specific document to its full content by doc_id."""

    name = "rag_expand"
    description = (
        "按 doc_id 获取知识库中指定文档的完整内容。\n"
        "配合 rag_retrieve 使用：先用摘要/片段模式筛选出感兴趣的 doc_id，\n"
        "再调用本工具展开全文，避免一次性加载所有文档的全量内容。"
    )

    input_schema = {
        "type": "object",
        "properties": {
            "doc_id": {
                "type": "string",
                "description": "rag_retrieve 返回结果中的 doc_id（别名: document_id, file_id, path, keyword, query）",
            },
            "max_chars": {
                "type": "integer",
                "description": "最多返回字符数，默认 6000。配合 start_char 可分页读取大文件。",
                "default": 6000,
            },
            "start_char": {
                "type": "integer",
                "description": "从第几个字符开始返回，默认 0（文件开头）。用于分页读取：第二页传 start_char=6000，以此类推。",
                "default": 0,
            },
        },
        "required": ["doc_id"],
    }

    async def execute(
        self,
        doc_id: str = "",
        max_chars: int = 6000,
        start_char: int = 0,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Fetch full (or paginated) content of a single document by ID.

        Parameters
        ----------
        doc_id     : document identifier from rag_retrieve results.
        max_chars  : maximum characters to return in this call (default 6000).
        start_char : character offset to start from (default 0).  Use together
                     with max_chars to page through large documents:
                     page 1 → start_char=0, page 2 → start_char=6000, …

        Hermes/LLM 可能误传 ``keyword`` / ``query``；若值不是已有 doc_id，
        会按检索词取 top-1 文档再展开。
        """
        if not self.rag_kb:
            return {"status": "error", "message": "RAG knowledge base not initialized"}

        resolved = _coerce_expand_doc_id(doc_id, **kwargs)
        if not resolved:
            return {
                "status": "error",
                "message": "doc_id is required (aliases: document_id, file_id, path, keyword, query)",
            }

        try:
            doc = await self.rag_kb.get_document(resolved)
            if not doc and resolved and not resolved.startswith("file:"):
                # LLM passed a search term instead of doc_id — retrieve then expand.
                results = await self.rag_kb.retrieve(query=resolved, top_k=1)
                if results:
                    resolved = str(results[0].get("doc_id") or resolved)
                    doc = await self.rag_kb.get_document(resolved)
            if not doc:
                return {
                    "status": "error",
                    "message": f"doc_id '{resolved}' not found in knowledge base",
                }
            content    = doc.get("content", "")
            total_chars = len(content)
            start      = max(0, int(start_char))
            end        = start + max(1, int(max_chars))
            slice_     = content[start:end]
            return {
                "status":      "success",
                "doc_id":      resolved,
                "metadata":    doc.get("metadata", {}),
                "content":     slice_,
                "start_char":  start,
                "end_char":    start + len(slice_),
                "total_chars": total_chars,
                "truncated":   (start + len(slice_)) < total_chars,
                "has_more":    (start + len(slice_)) < total_chars,
            }
        except Exception as e:
            logger.error(f"RAG expand failed for doc_id={resolved}: {e}")
            return {"status": "error", "message": str(e)}


class RAGIndexTool(RAGToolMixin):
    """Tool for indexing documents into RAG knowledge base"""

    name = "rag_index"
    description = """
    Index new documents into the knowledge base.
    Use this when you need to add new knowledge or update existing documents.
    """

    input_schema = {
        "type": "object",
        "properties": {
            "documents": {
                "type": "array",
                "description": "List of documents to index",
                "items": {
                    "type": "object",
                    "properties": {"content": {"type": "string"}, "metadata": {"type": "object"}},
                },
            }
        },
        "required": ["documents"],
    }

    async def execute(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Execute RAG indexing"""
        if not self.rag_kb:
            return {"status": "error", "message": "RAG knowledge base not initialized"}

        try:
            count = await self.rag_kb.index_documents(documents=documents)

            return {"status": "success", "indexed_count": count}
        except Exception as e:
            logger.error(f"RAG indexing failed: {e}")
            return {"status": "error", "message": str(e)}


class RAGAgentTool(RAGToolMixin):
    """
    Agentic RAG tool — multi-hop, self-refining retrieval.

    Internal loop (transparent to the caller):
      Hop 1: LLM decomposes the query into N focused sub-queries → retrieves each
      Assess: LLM scores each result for relevance (0-1); keeps score ≥ threshold
      Gap:    LLM identifies what knowledge is still missing
      Hop 2…max_hops: new sub-queries for the gaps → retrieve → assess → merge
      Synthesise: deduplicate by doc_id, sort by final score, apply detail_level

    Returns a single consolidated results list with a coverage_notes field
    explaining what was searched and what gaps (if any) remain.
    """

    name = "rag_query"
    description = (
        "Agentic RAG 多跳检索：自动将问题分解为多个子查询，\n"
        "多轮检索+LLM相关性评估+缺口检测，最终返回去重合并后的高质量结果。\n"
        "适用于复杂/多方面问题（如同时需要制导律原理+参数范围+仿真工况）。\n"
        "参数：\n"
        "  query        — 原始问题（自然语言，可以很复杂）\n"
        "  max_hops     — 最大检索轮次（默认 2，每轮消耗 1 次 LLM 调用）\n"
        "  top_k        — 每个子查询返回的候选数（默认 4）\n"
        "  detail_level — summary / snippet(默认) / full\n"
        "  rel_threshold — 相关性分数阈值，低于此值的结果被过滤（默认 0.3）"
    )
    input_schema = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "原始检索问题（可复杂，会被自动分解）",
            },
            "max_hops": {
                "type": "integer",
                "description": "最大检索跳数（默认 2）",
                "default": 2,
            },
            "top_k": {
                "type": "integer",
                "description": "每子查询返回候选数（默认 4）",
                "default": 4,
            },
            "detail_level": {
                "type": "string",
                "enum": ["summary", "snippet", "full"],
                "description": "内容披露层级（默认 snippet）",
                "default": "snippet",
            },
            "rel_threshold": {
                "type": "number",
                "description": "相关性阈值，低于此分数的结果被丢弃（默认 0.3）",
                "default": 0.3,
            },
        },
        "required": ["query"],
    }

    # ── LLM helpers ───────────────────────────────────────────────────────────

    @staticmethod
    async def _llm(prompt: str, max_tokens: int = 512) -> str:
        """Single async LLM call; returns empty string on failure."""
        try:
            from multi_agent.config_loader import get_config
            import openai as _openai

            cfg = get_config().llm
            client = _openai.AsyncOpenAI(
                api_key=cfg.api_key or "sk-dummy",
                base_url=cfg.base_url or "https://api.openai.com/v1",
                timeout=30,
                max_retries=3,
            )
            resp = await client.chat.completions.create(
                model=cfg.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=max_tokens,
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as exc:
            logger.warning("[RAGAgent] LLM call failed: %s", exc)
            return ""

    @staticmethod
    def _parse_json_list(text: str) -> List[str]:
        """Extract a JSON string array from LLM output; fallback to line split."""
        text = text.strip()
        m = re.search(r"\[.*?\]", text, re.DOTALL)
        if m:
            try:
                items = json.loads(m.group())
                return [str(i).strip() for i in items if str(i).strip()]
            except json.JSONDecodeError:
                pass
        # fallback: one item per non-empty line
        return [ln.lstrip("-•123456789. ").strip() for ln in text.splitlines() if ln.strip()]

    # ── Plan: decompose query → sub-queries ───────────────────────────────────

    async def _plan(self, query: str) -> List[str]:
        prompt = (
            "你是制导系统知识库检索规划器。\n"
            "将以下问题分解为 2-4 个精准的英文/中文检索子查询，每个子查询应聚焦于问题的一个具体方面。\n"
            "仅输出 JSON 字符串数组，例如：[\"子查询1\", \"子查询2\"]。\n\n"
            f"原始问题：{query}"
        )
        raw = await self._llm(prompt, max_tokens=256)
        sub_queries = self._parse_json_list(raw)
        if not sub_queries:
            sub_queries = [query]  # fallback: use original query
        logger.info("[RAGAgent] Plan: %d sub-queries: %s", len(sub_queries), sub_queries)
        return sub_queries[:5]  # cap at 5 sub-queries

    # ── Assess: score each result for relevance ───────────────────────────────

    async def _assess(
        self, query: str, candidates: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Ask LLM to score each candidate 0.0-1.0 for relevance to query.
        Returns candidates with a new 'rel_score' field.
        """
        if not candidates:
            return []

        items_text = "\n".join(
            f"[{i}] {c.get('content', '')[:300]}"
            for i, c in enumerate(candidates)
        )
        prompt = (
            "对以下检索结果与原始问题的相关性打分（0.0=完全无关，1.0=高度相关）。\n"
            "只输出 JSON 数字数组，长度与结果数相同，例如：[0.9, 0.3, 0.7]。\n\n"
            f"原始问题：{query}\n\n"
            f"检索结果：\n{items_text}"
        )
        raw = await self._llm(prompt, max_tokens=128)
        scores: List[float] = []
        m = re.search(r"\[.*?\]", raw, re.DOTALL)
        if m:
            try:
                scores = [float(x) for x in json.loads(m.group())]
            except Exception:
                pass
        # Pad with 0.5 if parse failed or length mismatch
        while len(scores) < len(candidates):
            scores.append(0.5)

        for i, c in enumerate(candidates):
            c["rel_score"] = round(scores[i], 3)
        return candidates

    # ── Gap: identify missing aspects ─────────────────────────────────────────

    async def _gap(
        self, original_query: str, found_summaries: List[str]
    ) -> List[str]:
        """
        Ask LLM: given what we found so far, what aspects of the query
        are still not covered?  Returns follow-up sub-queries (empty = done).
        """
        found_block = "\n".join(f"- {s}" for s in found_summaries[:8])
        prompt = (
            "原始问题如下，以及已检索到的文档摘要。\n"
            "判断问题中哪些方面尚未被覆盖，并列出 0-3 条补充检索子查询。\n"
            "若已充分覆盖，输出空数组 []。\n"
            "只输出 JSON 字符串数组。\n\n"
            f"原始问题：{original_query}\n\n"
            f"已找到的内容摘要：\n{found_block}"
        )
        raw = await self._llm(prompt, max_tokens=200)
        gaps = self._parse_json_list(raw)
        # Treat "[]" or empty as no gaps
        if raw.strip() in ("[]", "[ ]", ""):
            return []
        logger.info("[RAGAgent] Gap queries: %s", gaps[:3])
        return gaps[:3]

    # ── Main execute ──────────────────────────────────────────────────────────

    async def execute(
        self,
        query: str = "",
        max_hops: int = 2,
        top_k: int = 4,
        detail_level: str = "snippet",
        rel_threshold: float = 0.3,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        query = _coerce_retrieve_query(query, **kwargs)
        if not query:
            return {"status": "error", "message": "query (or keyword/sub_query) is required"}
        if not self.rag_kb:
            return {"status": "error", "message": "RAG knowledge base not initialized"}

        max_chars = _DETAIL_CHARS.get(detail_level, _DETAIL_CHARS["snippet"])
        seen_doc_ids: set = set()
        all_results: List[Dict[str, Any]] = []
        all_sub_queries: List[str] = []
        hops_done = 0

        # ── Hop loop ─────────────────────────────────────────────────────────
        pending_queries: List[str] = []
        try:
            # Hop 1: plan sub-queries
            pending_queries = await self._plan(query)
        except Exception:
            pending_queries = [query]

        for hop in range(1, max_hops + 1):
            if not pending_queries:
                break
            hops_done = hop
            all_sub_queries.extend(pending_queries)

            # Retrieve for all pending sub-queries in parallel
            retrieve_tasks = [
                self.rag_kb.retrieve(query=sq, top_k=top_k)
                for sq in pending_queries
            ]
            try:
                hop_raw: List[List[Dict]] = await asyncio.gather(*retrieve_tasks)
            except Exception as exc:
                logger.warning("[RAGAgent] Retrieval error on hop %d: %s", hop, exc)
                break

            # Flatten + deduplicate
            hop_candidates: List[Dict[str, Any]] = []
            for batch in hop_raw:
                for r in batch:
                    did = r.get("doc_id", "")
                    if did and did not in seen_doc_ids:
                        seen_doc_ids.add(did)
                        hop_candidates.append(r)

            # Assess relevance
            hop_candidates = await self._assess(query, hop_candidates)

            # Filter by threshold
            hop_kept = [c for c in hop_candidates if c.get("rel_score", 0) >= rel_threshold]
            logger.info(
                "[RAGAgent] Hop %d: %d candidates → %d kept (threshold=%.2f)",
                hop, len(hop_candidates), len(hop_kept), rel_threshold,
            )
            all_results.extend(hop_kept)

            # Gap detection (only if more hops remain)
            if hop < max_hops:
                found_summaries = [r.get("content", "")[:200] for r in hop_kept]
                pending_queries = await self._gap(query, found_summaries)
            else:
                pending_queries = []

        # ── Synthesise ────────────────────────────────────────────────────────
        # Sort by combined vector score + rel_score, then apply detail_level cap
        all_results.sort(
            key=lambda r: r.get("rel_score", 0) * 0.6 + r.get("score", 0) * 0.4,
            reverse=True,
        )

        output_results = [
            {
                "doc_id":    r.get("doc_id", ""),
                "score":     round(r.get("score", 0.0), 4),
                "rel_score": round(r.get("rel_score", 0.0), 3),
                "metadata":  r.get("metadata", {}),
                "content":   r.get("content", "")[:max_chars],
                "truncated": len(r.get("content", "")) > max_chars,
            }
            for r in all_results
        ]

        return {
            "status":          "success",
            "query":           query,
            "hops":            hops_done,
            "sub_queries":     all_sub_queries,
            "results_count":   len(output_results),
            "detail_level":    detail_level,
            "hint": (
                "若需某条文档全文，用 rag_expand(doc_id=...) 展开。"
                if detail_level != "full" else ""
            ),
            "results":         output_results,
        }


class SimilaritySearchTool(RAGToolMixin):
    """Tool for similarity search in knowledge base"""

    name = "rag_similarity_search"
    description = """
    Perform similarity search with a minimum threshold.
    Use this when you need to find highly similar documents.
    """

    input_schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "The search query"},
            "threshold": {
                "type": "number",
                "description": "Minimum similarity threshold (0-1)",
                "default": 0.7,
            },
            "top_k": {"type": "integer", "description": "Maximum number of results", "default": 10},
        },
        "required": ["query"],
    }

    async def execute(self, query: str, threshold: float = 0.7, top_k: int = 10) -> Dict[str, Any]:
        """Execute similarity search"""
        if not self.rag_kb:
            return {"status": "error", "message": "RAG knowledge base not initialized"}

        try:
            results = await self.rag_kb.similarity_search(
                query=query, threshold=threshold, top_k=top_k
            )

            return {
                "status": "success",
                "query": query,
                "threshold": threshold,
                "results_count": len(results),
                "results": results,
            }
        except Exception as e:
            logger.error(f"Similarity search failed: {e}")
            return {"status": "error", "message": str(e)}
