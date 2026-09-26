"""
LLM utilities for handling model-specific constraints.

Some models (e.g., DeepSeek reasoning models, Qwen models, Kimi K3) have strict
temperature constraints and only accept temperature=1.0.

DeepSeek-V4 (Pro/Flash) enables thinking mode by default. The CoT lands in
``reasoning_content`` and can exhaust ``max_tokens`` before any visible
``content`` is written — Expert Tuning then logs "empty LLM response".
"""

from __future__ import annotations

import os
import random
import re
from typing import Any, Dict


def get_safe_temperature(model_name: str, desired_temperature: float = 0.2) -> float:
    """
    Get a safe temperature value for the given model.

    Some models only allow temperature=1.0. This function detects such models
    and returns the appropriate temperature value.

    Args:
        model_name: The model identifier (e.g., "deepseek-v4-pro", "qwen3.7-max", "moonshot-v1-auto")
        desired_temperature: The desired temperature if the model supports it

    Returns:
        float: 1.0 for constrained models, desired_temperature otherwise
    """
    if not model_name:
        return desired_temperature

    model_lower = str(model_name).lower()

    # Models that only allow temperature=1.0
    constrained_models = [
        'deepseek-reasoner',
        'deepseek-r1',
        'qwen-plus',
        'qwen-max',
        'qwen-turbo',
        'qwen3',  # Covers qwen3.7-max, qwen3.8-max, etc.
        'qwen2.5-plus',
        'qwen2.5-max',
        'qwen2.5-turbo',
        'moonshot',  # Covers moonshot-v1-auto, moonshot-v1-8k, etc. (Kimi K3)
        'kimi',
    ]

    for pattern in constrained_models:
        if pattern in model_lower:
            return 1.0

    return desired_temperature


def uses_deepseek_thinking(model_name: str) -> bool:
    """True for DeepSeek-V4 / reasoner models that default to thinking mode."""
    name = str(model_name or "").lower()
    return any(
        key in name
        for key in ("deepseek-v4", "deepseek-reasoner", "deepseek-r1")
    )


def thinking_extra_body(
    model_name: str,
    *,
    enabled: bool = False,
) -> Dict[str, Any]:
    """OpenAI-SDK ``extra_body`` to toggle DeepSeek thinking mode.

    Expert / JSON-only calls should pass ``enabled=False`` so the model writes
    ``content`` instead of burning the token budget on ``reasoning_content``.
    """
    if not uses_deepseek_thinking(model_name):
        return {}
    return {"thinking": {"type": "enabled" if enabled else "disabled"}}


def _first_choice_message(resp: Any) -> Any:
    choices = getattr(resp, "choices", None) or []
    if not choices:
        return None
    return getattr(choices[0], "message", None)


def extract_chat_message_text(resp: Any) -> str:
    """Visible assistant text from a chat.completions response.

    Prefer ``message.content``. If that is empty (typical when thinking ate
    the token budget), fall back to a JSON object embedded in
    ``reasoning_content`` so a truncated thought still yields parameters.
    """
    msg = _first_choice_message(resp)
    if msg is None:
        return ""
    content = getattr(msg, "content", None)
    if isinstance(content, str) and content.strip():
        return content.strip()
    reasoning = getattr(msg, "reasoning_content", None)
    if not reasoning:
        extra = getattr(msg, "model_extra", None) or {}
        if isinstance(extra, dict):
            reasoning = extra.get("reasoning_content") or extra.get("reasoning")
    if not isinstance(reasoning, str) or not reasoning.strip():
        return ""
    text = reasoning.strip()
    match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if match:
        return match.group(0)
    return ""


def describe_chat_choice(resp: Any) -> str:
    """Compact diagnostic for empty-content debugging (finish_reason, usage)."""
    choices = getattr(resp, "choices", None) or []
    choice = choices[0] if choices else None
    msg = getattr(choice, "message", None) if choice is not None else None
    content = (getattr(msg, "content", None) or "") if msg is not None else ""
    reasoning = getattr(msg, "reasoning_content", None) if msg is not None else None
    if not reasoning and msg is not None:
        extra = getattr(msg, "model_extra", None) or {}
        if isinstance(extra, dict):
            reasoning = extra.get("reasoning_content") or extra.get("reasoning")
    usage = getattr(resp, "usage", None)
    usage_bits = []
    if usage is not None:
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            val = getattr(usage, key, None)
            if val is not None:
                usage_bits.append(f"{key}={val}")
        details = getattr(usage, "completion_tokens_details", None)
        if details is not None:
            rt = getattr(details, "reasoning_tokens", None)
            if rt is not None:
                usage_bits.append(f"reasoning_tokens={rt}")
    finish = getattr(choice, "finish_reason", None) if choice is not None else None
    return (
        f"finish_reason={finish!r} content_len={len(str(content))} "
        f"reasoning_len={len(str(reasoning or ''))} "
        + (" ".join(usage_bits) if usage_bits else "usage=?")
    )


def exponential_backoff_seconds(
    attempt: int,
    *,
    base_delay: float = 1.0,
    max_delay: float = 32.0,
    jitter_ratio: float = 0.20,
) -> float:
    """Exponential backoff with symmetric jitter (default ±20%).

    ``attempt`` is 1-based (after the 1st failure, wait before 2nd try).
    Delay = min(base * 2**(attempt-1), max_delay) * (1 ± jitter_ratio).

    Env overrides (optional):
      GSM_LLM_BACKOFF_BASE, GSM_LLM_BACKOFF_MAX, GSM_LLM_BACKOFF_JITTER
    """
    try:
        base_delay = float(os.environ.get("GSM_LLM_BACKOFF_BASE", base_delay))
    except (TypeError, ValueError):
        pass
    try:
        max_delay = float(os.environ.get("GSM_LLM_BACKOFF_MAX", max_delay))
    except (TypeError, ValueError):
        pass
    try:
        jitter_ratio = float(os.environ.get("GSM_LLM_BACKOFF_JITTER", jitter_ratio))
    except (TypeError, ValueError):
        pass

    attempt = max(1, int(attempt))
    delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
    if jitter_ratio > 0:
        delay *= 1.0 + random.uniform(-jitter_ratio, jitter_ratio)
    return max(0.0, delay)


def llm_max_retries(default: int = 5) -> int:
    """Cap per-request LLM retries (finite; never infinite). Env: GSM_LLM_MAX_RETRIES."""
    try:
        n = int(os.environ.get("GSM_LLM_MAX_RETRIES", default))
    except (TypeError, ValueError):
        n = default
    return max(1, min(n, 8))
