#!/usr/bin/env python3
"""Fairness ablation for automated DeepSeek-V4 feedback loops."""

from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Any, Dict, Mapping, Tuple

from experiments.baselines import (
    DEFAULT_MATLAB_SCRIPT,
    DEFAULT_PARAMS,
    PARAM_BOUNDS,
    PARAM_NAMES,
    _baseline_payload,
    _clip_params,
    _evaluate,
    _fitness_from_result,
)

logger = logging.getLogger(__name__)


def _parse_params(text: str) -> Dict[str, float]:
    """Parse a JSON parameter block from LLM output."""
    for candidate in re.findall(r"\{[\s\S]*?\}", text or ""):
        try:
            raw = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        parsed = {k: float(v) for k, v in raw.items() if k in PARAM_BOUNDS}
        if parsed:
            return _clip_params({**DEFAULT_PARAMS, **parsed})
    return {}


def _deepseek_feedback_params(
    scenario: str,
    previous_params: Mapping[str, float],
    previous_result: Mapping[str, Any],
) -> Tuple[Dict[str, float], float]:
    """Call DeepSeek-V4 for the next automated feedback proposal."""
    if os.getenv("GSM_BASELINE_SKIP_LLM", "").strip() == "1":
        return _clip_params(previous_params), 0.0

    started = time.perf_counter()
    try:
        from openai import OpenAI
        from multi_agent.config_loader import get_config

        cfg = get_config().llm
        api_key = cfg.api_key or os.getenv("OPENAI_API_KEY") or ""
        if not api_key or api_key == "sk-dummy":
            return _clip_params(previous_params), 0.0
        client = OpenAI(api_key=api_key, base_url=cfg.base_url or "https://api.openai.com/v1")
        prompt = (
            "You are simulating the Windsurf DeepSeek-V4 workflow with automatic MATLAB feedback. "
            "Return only JSON parameter values, no markdown. "
            f"Scenario={scenario}; keys={PARAM_NAMES}; bounds={json.dumps(PARAM_BOUNDS)}; "
            f"previous_params={json.dumps(dict(previous_params))}; "
            f"previous_result={json.dumps(dict(previous_result), default=str)}."
        )
        response = client.chat.completions.create(
            model=cfg.model,
            messages=[
                {"role": "system", "content": "Tune missile guidance/autopilot code parameters."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            timeout=60.0,
        )
        params = _parse_params(response.choices[0].message.content or "")
        inference = time.perf_counter() - started
        return (params or _clip_params(previous_params)), inference
    except Exception as exc:
        logger.warning("DeepSeek feedback call failed: %s", exc)
        return _clip_params(previous_params), 0.0


def deepseek_v4_auto_feedback(scenario, dev_data_path, test_data_path, seed):
    """Run a DeepSeek-V4 auto-feedback fairness ablation.

    This simulates the Windsurf loop without manual file transfer latency:
    LLM proposes a code/parameter update, Python pushes it to MATLAB, parsed
    metrics are fed back to the next LLM round, and the best design is tested.

    Args:
        scenario: Scenario key.
        dev_data_path: Development MC data path.
        test_data_path: Test MC data path.
        seed: Master seed for traceability.

    Returns:
        Baseline-shaped result dictionary with timing and convergence fields.
    """
    started = time.perf_counter()
    max_rounds = int(os.getenv("GSM_SUPP_MAX_ROUNDS", "5"))
    params = DEFAULT_PARAMS.copy()
    best_params = params.copy()
    best_dev = _evaluate(params, dev_data_path)
    best_fit = _fitness_from_result(best_dev)
    convergence_log = [(0, best_fit)]
    inference_sec = 0.0
    sim_sec = float(best_dev.get("sim_time_sec", 0.0) or 0.0)
    mc_calls = 1

    for round_idx in range(1, max_rounds + 1):
        params, llm_time = _deepseek_feedback_params(scenario, params, best_dev)
        inference_sec += llm_time
        dev_result = _evaluate(params, dev_data_path)
        sim_sec += float(dev_result.get("sim_time_sec", 0.0) or 0.0)
        mc_calls += 1
        fitness = _fitness_from_result(dev_result)
        convergence_log.append((round_idx, fitness))
        if fitness < best_fit:
            best_fit = fitness
            best_dev = dev_result
            best_params = params
        if bool(best_dev.get("feasible", False)):
            break

    test_result = _evaluate(best_params, test_data_path)
    sim_sec += float(test_result.get("sim_time_sec", 0.0) or 0.0)
    mc_calls += 1
    wall = time.perf_counter() - started
    return _baseline_payload(
        best_params=best_params,
        dev_result=best_dev,
        test_result=test_result,
        mc_calls_used=mc_calls,
        wall_clock_sec=wall,
        inference_sec=inference_sec,
        orchestration_sec=max(0.0, wall - inference_sec - sim_sec),
        iterations_used=len(convergence_log) - 1,
        design_path="DEEPSEEK_AUTO_FEEDBACK",
        feasibility_rate=1.0 if bool(best_dev.get("feasible", False)) else 0.0,
        convergence_log=convergence_log,
    )


__all__ = ["deepseek_v4_auto_feedback"]
