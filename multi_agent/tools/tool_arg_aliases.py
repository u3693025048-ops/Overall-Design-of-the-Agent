"""Normalize Hermes/LLM tool argument aliases to canonical parameter names."""

from __future__ import annotations

from typing import Any, Optional


def _first_str(*values: Any) -> str:
    for raw in values:
        if raw is None:
            continue
        text = str(raw).strip()
        if text:
            return text
    return ""


def coerce_script_path(
    script_path: Any = None,
    *,
    path: Any = None,
    file_path: Any = None,
    filename: Any = None,
    file: Any = None,
    script: Any = None,
    **_: Any,
) -> str:
    return _first_str(script_path, path, file_path, filename, file, script)


def coerce_task_prompt(
    task_prompt: Any = None,
    *,
    prompt: Any = None,
    task_description: Any = None,
    task: Any = None,
    description: Any = None,
    **_: Any,
) -> str:
    return _first_str(task_prompt, prompt, task_description, task, description)


def coerce_rag_query(
    query: Any = None,
    *,
    keyword: Any = None,
    sub_query: Any = None,
    q: Any = None,
    search: Any = None,
    **_: Any,
) -> str:
    return _first_str(query, sub_query, keyword, q, search)
