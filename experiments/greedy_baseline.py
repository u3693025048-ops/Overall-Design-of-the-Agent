#!/usr/bin/env python3
"""Greedy feasibility-first baseline for Reflection Agent ablation."""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, Mapping, Tuple

from experiments.baselines import (
    DEFAULT_PARAMS,
    PARAM_BOUNDS,
    PARAM_NAMES,
    _baseline_payload,
    _clip_params,
    _evaluate,
    _fitness_from_result,
)

logger = logging.getLogger(__name__)

CONSTRAINT_PRIORITY = ["PM", "BW", "peak_nz", "SEP", "hit_rate"]


def _constraint_violation(result: Mapping[str, Any], metric: str) -> float:
    """Return signed violation magnitude for one priority metric."""
    violations = result.get("constraint_violations") or {}
    if metric == "PM":
        return max(float(violations.get("PM_low", 0.0)), float(violations.get("PM_high", 0.0)))
    if metric == "BW":
        return max(float(violations.get("BW_low", 0.0)), float(violations.get("BW_high", 0.0)))
    if metric == "peak_nz":
        return float(violations.get("peak_nz", 0.0))
    if metric == "SEP":
        return float(violations.get("SEP", 0.0))
    if metric == "hit_rate":
        return float(violations.get("hit_rate", 0.0))
    return 0.0


def _worst_priority_metric(result: Mapping[str, Any]) -> str:
    """Select the first violated metric by fixed reflection-ablation priority."""
    for metric in CONSTRAINT_PRIORITY:
        if _constraint_violation(result, metric) > 0.0:
            return metric
    return CONSTRAINT_PRIORITY[-1]


def _objective_for_metric(result: Mapping[str, Any], metric: str) -> float:
    """Objective minimized during finite-difference greedy correction."""
    primary = _constraint_violation(result, metric)
    return primary * 1000.0 + _fitness_from_result(result)


def _finite_difference_step(
    params: Mapping[str, float],
    metric: str,
    dev_data_path: str,
) -> Tuple[Dict[str, float], Dict[str, Any], int, float]:
    """Try plus/minus finite-difference steps for every parameter."""
    best_params = _clip_params(params)
    best_result = _evaluate(best_params, dev_data_path)
    best_obj = _objective_for_metric(best_result, metric)
    calls = 1
    sim_sec = float(best_result.get("sim_time_sec", 0.0) or 0.0)

    max_params_raw = os.getenv("GSM_SUPP_MAX_PARAMS", "").strip()
    max_params = int(max_params_raw) if max_params_raw.isdigit() else len(PARAM_NAMES)
    for name in PARAM_NAMES[:max_params]:
        lo, hi = PARAM_BOUNDS[name]
        step = 0.05 * (hi - lo)
        for direction in (-1.0, 1.0):
            candidate = dict(best_params)
            candidate[name] = min(hi, max(lo, candidate[name] + direction * step))
            result = _evaluate(candidate, dev_data_path)
            calls += 1
            sim_sec += float(result.get("sim_time_sec", 0.0) or 0.0)
            obj = _objective_for_metric(result, metric)
            if obj < best_obj:
                best_obj = obj
                best_params = candidate
                best_result = result
    return best_params, best_result, calls, sim_sec


def greedy_feasibility_first(scenario, dev_data_path, test_data_path, seed):
    """Run a simple greedy feasibility-first correction strategy.

    The policy uses a fixed constraint priority ``PM > BW > peak_nz > SEP >
    hit_rate``. At each round it estimates finite-difference improvements with
    a step equal to 5% of each parameter range, accepts the best improving
    parameter move, and stops after five rounds or once feasible.

    Args:
        scenario: Scenario key.
        dev_data_path: Development MC data path.
        test_data_path: Test MC data path.
        seed: Master seed for traceability.

    Returns:
        Baseline-shaped result dictionary.
    """
    started = time.perf_counter()
    max_rounds = int(os.getenv("GSM_SUPP_MAX_ROUNDS", "5"))
    params = DEFAULT_PARAMS.copy()
    dev_result = _evaluate(params, dev_data_path)
    sim_sec = float(dev_result.get("sim_time_sec", 0.0) or 0.0)
    mc_calls = 1
    convergence_log = [(0, _fitness_from_result(dev_result))]
    best_params = params.copy()
    best_dev = dev_result
    best_fit = _fitness_from_result(dev_result)

    for round_idx in range(1, max_rounds + 1):
        if bool(dev_result.get("feasible", False)):
            break
        metric = _worst_priority_metric(dev_result)
        params, dev_result, calls, sec = _finite_difference_step(params, metric, dev_data_path)
        mc_calls += calls
        sim_sec += sec
        fit = _fitness_from_result(dev_result)
        convergence_log.append((round_idx, fit))
        if fit < best_fit:
            best_fit = fit
            best_params = _clip_params(params)
            best_dev = dev_result

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
        inference_sec=0.0,
        orchestration_sec=max(0.0, wall - sim_sec),
        iterations_used=len(convergence_log) - 1,
        design_path="GREEDY_FEASIBILITY_FIRST",
        feasibility_rate=1.0 if bool(best_dev.get("feasible", False)) else 0.0,
        convergence_log=convergence_log,
    )


__all__ = ["greedy_feasibility_first"]
