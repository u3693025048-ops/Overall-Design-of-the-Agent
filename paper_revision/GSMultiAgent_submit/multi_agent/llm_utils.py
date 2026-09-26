"""
LLM utilities for handling model-specific constraints.

Some models (e.g., DeepSeek reasoning models, Qwen models, Kimi K3) have strict
temperature constraints and only accept temperature=1.0.
"""

from __future__ import annotations

import os
import random


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
