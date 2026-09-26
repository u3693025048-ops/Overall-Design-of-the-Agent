#!/usr/bin/env python3
"""
ParameterExperience (Dynamic Memory Buffer) Tools for Hermes Agent
Provides experience storage and retrieval capabilities
"""

import json
import logging
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


def _coerce_pe_search_query(
    query: Any = None,
    *,
    sub_query: Any = None,
    keywords: Any = None,
) -> Dict[str, Any]:
    """Normalize LLM/Hermes variants (sub_query, keywords, str query) to a dict."""
    raw = query
    if raw is None and sub_query is not None:
        raw = sub_query
    if raw is None and keywords is not None:
        raw = keywords
    if raw is None:
        return {}
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return {}
        return {"task": text, "description": text, "keywords": text}
    if isinstance(raw, dict):
        return dict(raw)
    if isinstance(raw, (list, tuple)):
        text = " ".join(str(x).strip() for x in raw if str(x).strip())
        if not text:
            return {}
        return {"task": text, "description": text, "keywords": text}
    text = str(raw).strip()
    return {"task": text, "description": text} if text else {}


class ParameterExperienceToolMixin:
    """Mixin class to add ParameterExperience capabilities to tools"""

    def __init__(self, *args, **kwargs):
        self.parameter_experience = None
        super().__init__(*args, **kwargs)

    def set_parameter_experience(self, parameter_experience) -> None:
        """Set the ParameterExperience memory buffer"""
        self.parameter_experience = parameter_experience


class ParameterExperienceSearchTool(ParameterExperienceToolMixin):
    """Tool for retrieving similar experiences from ParameterExperience"""

    name = "parameter_experience_search"
    description = """
    Search for similar experiences in the dynamic memory buffer.
    Use this when you need to find historical cases with similar task context.
    Returns best matching experiences with fitness scores.
    """

    input_schema = {
        "type": "object",
        "properties": {
            "query": {
                "description": (
                    "Task context to search for (object preferred). "
                    "May also be a plain string, e.g. 'T4 N_pn w1'."
                ),
            },
            "sub_query": {
                "type": "string",
                "description": "Alias for query when passing a short text search string.",
            },
            "keywords": {
                "type": "string",
                "description": "Alias for query (keyword-style search string).",
            },
            "top_k": {
                "type": "integer",
                "description": "Number of similar experiences to retrieve",
                "default": 5,
            },
            "memory_type": {
                "type": "string",
                "description": "Type of memory: short_term, long_term, or all",
                "enum": ["short_term", "long_term", "all"],
                "default": "all",
            },
        },
    }

    async def execute(
        self,
        query: Any = None,
        top_k: int = 5,
        memory_type: str = "all",
        sub_query: Any = None,
        keywords: Any = None,
        mode: Any = None,
        **_: Any,
    ) -> Dict[str, Any]:
        """Execute ParameterExperience search"""
        if not self.parameter_experience:
            return {"status": "error", "message": "ParameterExperience not initialized"}

        query_dict = _coerce_pe_search_query(
            query, sub_query=sub_query, keywords=keywords,
        )
        if mode and isinstance(mode, str) and mode.strip():
            query_dict.setdefault("search_mode", mode.strip())
        if not query_dict:
            return {
                "status": "error",
                "message": "query (or sub_query / keywords) is required",
            }

        try:
            from ..memory.parameter_experience import MemoryType

            mem_type = None
            if memory_type == "short_term":
                mem_type = MemoryType.SHORT_TERM
            elif memory_type == "long_term":
                mem_type = MemoryType.LONG_TERM

            results = await self.parameter_experience.retrieve_similar(
                query=query_dict, top_k=top_k, memory_type=mem_type
            )

            return {
                "status": "success",
                "query": query_dict,
                "results_count": len(results),
                "experiences": results,
            }
        except Exception as e:
            logger.error(f"ParameterExperience search failed: {e}")
            return {"status": "error", "message": str(e)}


class ParameterExperienceStoreTool(ParameterExperienceToolMixin):
    """Tool for storing experiences in ParameterExperience"""

    name = "parameter_experience_store"
    description = """
    Store a new experience in the dynamic memory buffer.
    Use this after successful optimization to save best parameters and results.
    """

    input_schema = {
        "type": "object",
        "properties": {
            "task_context": {"type": "object", "description": "Task context information"},
            "parameters": {"type": "object", "description": "Optimization parameters"},
            "objectives": {"type": "object", "description": "Objective values achieved"},
            "fitness": {"type": "number", "description": "Overall fitness score"},
            "memory_type": {
                "type": "string",
                "description": "short_term or long_term memory",
                "enum": ["short_term", "long_term"],
                "default": "short_term",
            },
        },
        "required": ["task_context", "parameters", "fitness"],
    }

    async def execute(
        self,
        task_context: Dict[str, Any],
        parameters: Dict[str, float],
        objectives: Dict[str, float],
        fitness: float,
        memory_type: str = "short_term",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Execute ParameterExperience store"""
        if not self.parameter_experience:
            return {"status": "error", "message": "ParameterExperience not initialized"}

        try:
            from ..memory.parameter_experience import MemoryType

            mem_type = MemoryType.LONG_TERM if memory_type == "long_term" else MemoryType.SHORT_TERM

            memory_id = await self.parameter_experience.store(
                task_context=task_context,
                parameters=parameters,
                objectives=objectives,
                fitness=fitness,
                memory_type=mem_type,
                metadata=metadata,
            )

            return {"status": "success", "memory_id": memory_id}
        except Exception as e:
            logger.error(f"ParameterExperience store failed: {e}")
            return {"status": "error", "message": str(e)}


class ParameterExperienceBestTool(ParameterExperienceToolMixin):
    """Tool for retrieving best performing experiences"""

    name = "parameter_experience_best"
    description = """
    Retrieve the best performing experiences for a given task context.
    Optionally filter by param_ranges so only experiences whose parameters
    fall within the RAG-inferred reasonable ranges are returned.
    Use this to leverage proven best solutions for similar tasks.
    """

    input_schema = {
        "type": "object",
        "properties": {
            "task_context": {
                "description": (
                    "Task context to match against experiences (object preferred). "
                    "Can include: mission_type, guidance_law, autopilot_params, etc."
                ),
            },
            "sub_query": {
                "type": "string",
                "description": "Short text alias — converted to task_context.task / keywords.",
            },
            "query": {
                "type": "string",
                "description": "Alias for sub_query (keyword search string).",
            },
            "top_k": {
                "type": "integer",
                "description": "Number of best experiences to retrieve (default: 5)",
                "default": 5,
            },
            "param_ranges": {
                "type": "object",
                "description": (
                    "Optional parameter range constraints inferred from RAG knowledge. "
                    "Format: {\"w1\": [min, max], \"zeta1\": [min, max], ...}. "
                    "Only experiences whose parameters all fall within these ranges are returned. "
                    "If no experience satisfies the ranges, falls back to unrestricted retrieval."
                ),
            },
        },
        "required": [],
    }

    @staticmethod
    def _within_ranges(params: Dict[str, Any], ranges: Dict[str, Any]) -> bool:
        """Return True if every param specified in ranges falls within [min, max]."""
        for key, bounds in ranges.items():
            if not isinstance(bounds, (list, tuple)) or len(bounds) < 2:
                continue
            val = params.get(key)
            if val is None:
                continue
            try:
                lo, hi = float(bounds[0]), float(bounds[1])
                if not (lo <= float(val) <= hi):
                    return False
            except (TypeError, ValueError):
                continue
        return True

    async def execute(
        self,
        task_context: Optional[Any] = None,
        top_k: int = 5,
        param_ranges: Optional[Dict[str, Any]] = None,
        sub_query: Any = None,
        query: Any = None,
        **_: Any,
    ) -> Dict[str, Any]:
        """Execute ParameterExperience best retrieval with optional range filtering."""
        if not self.parameter_experience:
            return {"status": "error", "message": "ParameterExperience not initialized"}

        if task_context is None:
            task_context = {}
        elif isinstance(task_context, str):
            task_context = _coerce_pe_search_query(task_context)
        elif not isinstance(task_context, dict):
            task_context = _coerce_pe_search_query(task_context)

        if sub_query or query:
            _extra = _coerce_pe_search_query(sub_query=sub_query or query)
            task_context = {**_extra, **task_context}

        try:
            fetch_k = top_k * 4 if param_ranges else top_k
            results = await self.parameter_experience.retrieve_best(
                task_context=task_context, top_k=fetch_k
            )

            filtered = results
            range_applied = False
            if param_ranges:
                in_range = [
                    r for r in results
                    if self._within_ranges(r.get("parameters", {}), param_ranges)
                ]
                if in_range:
                    filtered = in_range[:top_k]
                    range_applied = True
                else:
                    filtered = results[:top_k]

            return {
                "status": "success",
                "task_context": task_context,
                "results_count": len(filtered),
                "param_ranges_applied": range_applied,
                "best_experiences": filtered,
            }
        except Exception as e:
            logger.error(f"ParameterExperience best retrieval failed: {e}")
            return {"status": "error", "message": str(e)}


class ParameterExperienceStatsTool(ParameterExperienceToolMixin):
    """Tool for getting ParameterExperience statistics"""

    name = "parameter_experience_stats"
    description = """
    Get statistics about the dynamic memory buffer.
    Use this to understand memory usage and performance.
    """

    input_schema = {"type": "object", "properties": {}}

    async def execute(self) -> Dict[str, Any]:
        """Execute ParameterExperience stats"""
        if not self.parameter_experience:
            return {"status": "error", "message": "ParameterExperience not initialized"}

        try:
            stats = await self.parameter_experience.get_statistics()

            return {"status": "success", "statistics": stats}
        except Exception as e:
            logger.error(f"ParameterExperience stats failed: {e}")
            return {"status": "error", "message": str(e)}
