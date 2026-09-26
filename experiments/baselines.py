#!/usr/bin/env python3
"""Baseline optimizers for paper-aligned external-MC experiments."""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from multi_agent.rl.matlab_rl_optimizer import ALL_TUNABLE_PARAM_SPECS
from multi_agent.tools.simulation_tool import (
    RunSimulationTool,
    check_constraints,
    run_mc_with_external_data,
)
from multi_agent.metric_contract import DEFAULT_METRIC_REQUIREMENTS, PEAK_NY_MAX
from experiments.equal_mc_budget import DEFAULT_MAX_CANDIDATES

logger = logging.getLogger(__name__)

PARAM_BOUNDS: Dict[str, Tuple[float, float]] = {
    key: (float(spec["min"]), float(spec["max"]))
    for key, spec in ALL_TUNABLE_PARAM_SPECS.items()
}
PARAM_NAMES: List[str] = list(PARAM_BOUNDS.keys())
DEFAULT_PARAMS: Dict[str, float] = {
    key: float(spec["nominal"]) for key, spec in ALL_TUNABLE_PARAM_SPECS.items()
}
DEFAULT_EQUAL_BUDGET = DEFAULT_MAX_CANDIDATES
DEFAULT_MATLAB_SCRIPT = RunSimulationTool.KB_TEMPLATE_PATH


def _ensure_input_files(*paths: str) -> None:
    """Raise ``FileNotFoundError`` for missing experiment input files."""
    for path in paths:
        if not Path(path).exists():
            raise FileNotFoundError(path)


def _clip_params(params: Mapping[str, float]) -> Dict[str, float]:
    """Clip params to the source-code search bounds."""
    clipped: Dict[str, float] = {}
    for name, (lo, hi) in PARAM_BOUNDS.items():
        raw = params.get(name, DEFAULT_PARAMS[name])
        clipped[name] = float(np.clip(float(raw), lo, hi))
    return clipped


def _array_to_params(values: Sequence[float], names: Sequence[str] = PARAM_NAMES) -> Dict[str, float]:
    """Convert a flat array to a clipped parameter dictionary."""
    return _clip_params({name: float(value) for name, value in zip(names, values)})


def _params_to_array(params: Mapping[str, float], names: Sequence[str] = PARAM_NAMES) -> np.ndarray:
    """Convert a parameter dictionary to a flat array ordered by ``names``."""
    clipped = _clip_params(params)
    return np.asarray([clipped[name] for name in names], dtype=float)


def _violation_sum(result: Mapping[str, Any]) -> float:
    """Return aggregate constraint violation for result ranking."""
    violations = result.get("constraint_violations") or {}
    return float(sum(float(v) for v in violations.values()))


def _fitness_from_result(result: Mapping[str, Any]) -> float:
    """Fitness used by random search and CMA-ES minimizers."""
    if bool(result.get("feasible", False)):
        return -(float(result.get("hit_rate", 0.0)) / 100.0 + 0.01)
    return _violation_sum(result)


def _empty_result(error: str) -> Dict[str, Any]:
    """Build a result-shaped failure payload."""
    feasible, violations = check_constraints(0.0, 1e9, 1e9, 0.0, 0.0)
    return {
        "hit_rate": 0.0,
        "SEP": float("inf"),
        "peak_nz": float("inf"),
        "PM": 0.0,
        "BW": 0.0,
        "miss_distances": [],
        "nz_peaks": [],
        "feasible": feasible,
        "constraint_violations": violations,
        "sim_time_sec": 0.0,
        "success": False,
        "error": error,
        "error_msg": error,
    }


def _baseline_payload(
    *,
    best_params: Mapping[str, float],
    dev_result: Mapping[str, Any],
    test_result: Mapping[str, Any],
    mc_calls_used: int,
    wall_clock_sec: float,
    inference_sec: float,
    orchestration_sec: float,
    iterations_used: int,
    design_path: str,
    feasibility_rate: float,
    convergence_log: Iterable[Tuple[int, float]],
) -> Dict[str, Any]:
    """Normalize baseline return fields."""
    return {
        "best_params": dict(best_params),
        "dev_result": dict(dev_result),
        "test_result": dict(test_result),
        "feasible": bool(test_result.get("feasible", False)),
        "mc_calls_used": int(mc_calls_used),
        "wall_clock_sec": float(wall_clock_sec),
        "inference_sec": float(inference_sec),
        "orchestration_sec": float(orchestration_sec),
        "iterations_used": int(iterations_used),
        "design_path": design_path,
        "feasibility_rate": float(feasibility_rate),
        "convergence_log": list(convergence_log),
    }


def _evaluate(
    params: Mapping[str, float],
    mc_data_path: str,
    matlab_script_path: str = DEFAULT_MATLAB_SCRIPT,
    n_runs: int = 100,
) -> Dict[str, Any]:
    """Run one external-MC evaluation with clipped parameters."""
    return run_mc_with_external_data(
        matlab_script_path,
        mc_data_path,
        n_runs=n_runs,
        kill_radius=10.0,
        dp_params=_clip_params(params),
    )


def random_search(
    scenario: str,
    dev_data_path: str,
    test_data_path: str,
    budget: int = DEFAULT_EQUAL_BUDGET,
    seed: int = 42,
) -> Dict[str, Any]:
    """Optimize tunable parameters with uniform random search.

    Args:
        scenario: Scenario key, e.g. ``"T1"``.
        dev_data_path: Development MC ``.mat`` file.
        test_data_path: Test MC ``.mat`` file.
        budget: Maximum number of development MC calls.
        seed: Explicit random seed.

    Returns:
        Baseline result dictionary with best params, dev/test metrics, timings,
        and convergence log.
    """
    _ensure_input_files(DEFAULT_MATLAB_SCRIPT, dev_data_path, test_data_path)
    logger.info("[RandomSearch] scenario=%s seed=%s budget=%d", scenario, seed, budget)
    rng = np.random.default_rng(seed)
    started = time.perf_counter()
    best_params = DEFAULT_PARAMS.copy()
    best_dev = _empty_result("not evaluated")
    best_rank: Tuple[int, float, float, float] = (1, float("inf"), 0.0, float("inf"))
    feasible_count = 0
    convergence_log: List[Tuple[int, float]] = []
    sim_sec = 0.0

    lows = np.asarray([PARAM_BOUNDS[name][0] for name in PARAM_NAMES], dtype=float)
    highs = np.asarray([PARAM_BOUNDS[name][1] for name in PARAM_NAMES], dtype=float)
    for eval_num in range(1, int(budget) + 1):
        values = rng.uniform(lows, highs)
        params = _array_to_params(values)
        result = _evaluate(params, dev_data_path)
        sim_sec += float(result.get("sim_time_sec", 0.0) or 0.0)
        feasible_count += int(bool(result.get("feasible", False)))
        fitness = _fitness_from_result(result)
        convergence_log.append((eval_num, fitness))
        rank = (
            0 if bool(result.get("feasible", False)) else 1,
            _violation_sum(result),
            -float(result.get("hit_rate", 0.0)),
            float(result.get("SEP", float("inf"))),
        )
        if rank < best_rank:
            best_rank = rank
            best_params = params
            best_dev = result

    test_result = _evaluate(best_params, test_data_path)
    sim_sec += float(test_result.get("sim_time_sec", 0.0) or 0.0)
    wall = time.perf_counter() - started
    orchestration = max(0.0, wall - sim_sec)
    return _baseline_payload(
        best_params=best_params,
        dev_result=best_dev,
        test_result=test_result,
        mc_calls_used=int(budget) + 1,
        wall_clock_sec=wall,
        inference_sec=0.0,
        orchestration_sec=orchestration,
        iterations_used=0,
        design_path="N/A",
        feasibility_rate=feasible_count / max(int(budget), 1),
        convergence_log=convergence_log,
    )


def _fitness(
    params_array: Sequence[float],
    dev_data_path: str,
    param_names: Sequence[str],
) -> float:
    """CMA-ES objective function.

    Args:
        params_array: Candidate parameter vector.
        dev_data_path: Development MC dataset path.
        param_names: Names matching vector order.

    Returns:
        Negative score for feasible candidates, positive violation sum otherwise.
    """
    params = _array_to_params(params_array, param_names)
    result = _evaluate(params, dev_data_path)
    return _fitness_from_result(result)


def cma_es_optimize(
    scenario: str,
    dev_data_path: str,
    test_data_path: str,
    initial_params: dict,
    budget: int = DEFAULT_EQUAL_BUDGET,
    seed: int = 42,
) -> Dict[str, Any]:
    """Optimize tunable parameters with CMA-ES.

    Args:
        scenario: Scenario key, e.g. ``"T1"``.
        dev_data_path: Development MC ``.mat`` file.
        test_data_path: Test MC ``.mat`` file.
        initial_params: Initial parameter dictionary.
        budget: Maximum number of development MC calls.
        seed: Explicit random seed passed to CMA-ES.

    Returns:
        Baseline result dictionary.
    """
    _ensure_input_files(DEFAULT_MATLAB_SCRIPT, dev_data_path, test_data_path)
    logger.info("[CMA-ES] scenario=%s seed=%s budget=%d", scenario, seed, budget)
    started = time.perf_counter()
    try:
        import cma
    except ImportError as exc:
        raise ImportError("cma package is required for cma_es_optimize") from exc

    names = PARAM_NAMES
    lows = [PARAM_BOUNDS[name][0] for name in names]
    highs = [PARAM_BOUNDS[name][1] for name in names]
    x0 = _params_to_array(initial_params or DEFAULT_PARAMS, names)
    sigma0 = 0.4
    popsize = 4 + 3 * int(math.floor(math.log(len(names))))
    max_generations = max(1, int(math.ceil(int(budget) / max(popsize, 1))))
    opts = {
        "bounds": [lows, highs],
        "CMA_stds": [max((hi - lo) / 2.0, 1e-6) for lo, hi in zip(lows, highs)],
        "popsize": popsize,
        "seed": int(seed),
        "maxiter": max_generations,
        "verbose": -9,
    }
    es = cma.CMAEvolutionStrategy(x0.tolist(), sigma0, opts)

    best_params = _array_to_params(x0, names)
    best_dev = _evaluate(best_params, dev_data_path)
    best_fitness = _fitness_from_result(best_dev)
    convergence_log: List[Tuple[int, float]] = [(1, best_fitness)]
    feasible_count = int(bool(best_dev.get("feasible", False)))
    sim_sec = float(best_dev.get("sim_time_sec", 0.0) or 0.0)
    eval_count = 1

    while not es.stop() and eval_count < int(budget):
        solutions = es.ask()
        remaining = int(budget) - eval_count
        batch = solutions[: max(0, remaining)]
        if not batch:
            break
        fitnesses: List[float] = []
        for sol in batch:
            params = _array_to_params(sol, names)
            result = _evaluate(params, dev_data_path)
            sim_sec += float(result.get("sim_time_sec", 0.0) or 0.0)
            feasible_count += int(bool(result.get("feasible", False)))
            fit = _fitness_from_result(result)
            fitnesses.append(fit)
            eval_count += 1
            convergence_log.append((eval_count, fit))
            if fit < best_fitness:
                best_fitness = fit
                best_params = params
                best_dev = result
        if len(batch) == popsize:
            es.tell(solutions, fitnesses)

    test_result = _evaluate(best_params, test_data_path)
    sim_sec += float(test_result.get("sim_time_sec", 0.0) or 0.0)
    wall = time.perf_counter() - started
    orchestration = max(0.0, wall - sim_sec)
    return _baseline_payload(
        best_params=best_params,
        dev_result=best_dev,
        test_result=test_result,
        mc_calls_used=eval_count + 1,
        wall_clock_sec=wall,
        inference_sec=0.0,
        orchestration_sec=orchestration,
        iterations_used=0,
        design_path="N/A",
        feasibility_rate=feasible_count / max(eval_count, 1),
        convergence_log=convergence_log,
    )


def _parse_llm_params(text: str) -> Dict[str, float]:
    """Extract a parameter JSON object from LLM text."""
    candidates = re.findall(r"\{[\s\S]*?\}", text or "")
    for candidate in candidates:
        try:
            raw = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        params = {k: float(v) for k, v in raw.items() if k in PARAM_BOUNDS}
        if params:
            return _clip_params({**DEFAULT_PARAMS, **params})
    return {}


def _deepseek_v4_initial_params(scenario: str) -> Tuple[Dict[str, float], float]:
    """Request a one-shot DeepSeek-V4 initial parameter suggestion.

    Args:
        scenario: Scenario key.

    Returns:
        Tuple of ``(params, inference_seconds)``. Returns nominal params with
        zero inference time if the API cannot be called.
    """
    if os.getenv("GSM_BASELINE_SKIP_LLM", "").strip() == "1":
        logger.info("[DirectPPO] GSM_BASELINE_SKIP_LLM=1; using nominal initial params")
        return DEFAULT_PARAMS.copy(), 0.0

    started = time.perf_counter()
    try:
        from openai import OpenAI
        from multi_agent.config_loader import get_config

        cfg = get_config().llm
        api_key = cfg.api_key or os.getenv("OPENAI_API_KEY") or ""
        if not api_key or api_key == "sk-dummy":
            logger.info("[DirectPPO] LLM key missing; using nominal initial params")
            return DEFAULT_PARAMS.copy(), 0.0
        client = OpenAI(api_key=api_key, base_url=cfg.base_url or "https://api.openai.com/v1")
        prompt = (
            "Return only JSON for GSMultiAgent initial autopilot parameters. "
            f"Scenario={scenario}. Keys must be exactly: {', '.join(PARAM_NAMES)}. "
            f"Bounds: {json.dumps(PARAM_BOUNDS)}. "
            "No markdown, no explanation."
        )
        resp = client.chat.completions.create(
            model=cfg.model,
            messages=[
                {"role": "system", "content": "You tune missile guidance autopilot parameters."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            timeout=60.0,
        )
        content = resp.choices[0].message.content or ""
        parsed = _parse_llm_params(content)
        inference = time.perf_counter() - started
        if parsed:
            return parsed, inference
        logger.warning("[DirectPPO] LLM response did not contain params; using nominal")
        return DEFAULT_PARAMS.copy(), inference
    except Exception as exc:
        logger.warning("[DirectPPO] LLM initial suggestion failed: %s", exc)
        return DEFAULT_PARAMS.copy(), 0.0


async def _run_direct_ppo_async(
    *,
    scenario: str,
    dev_data_path: str,
    max_iterations: int,
    initial_params: Dict[str, float],
) -> Tuple[Dict[str, Any], float, int]:
    """Run the existing MatlabRLOptimizer without multi-agent orchestration."""
    from multi_agent.config_loader import get_config
    from multi_agent.rl.matlab_rl_optimizer import MatlabRLOptimizer
    from multi_agent.simulation import GuidanceSimulator

    cfg = get_config()
    rl_cfg = cfg.matlab_rl_optimizer
    previous_mc_path = str(getattr(cfg.simulation, "mc_data_path", "") or "")
    cfg.simulation.mc_data_path = dev_data_path
    os.environ["GSM_MC_DATA_PATH"] = dev_data_path
    try:
        simulator = GuidanceSimulator(
            output_dir="./guidance_output",
            engine=cfg.simulation.engine,
            octave_path=cfg.simulation.octave_path,
            matlab_path=cfg.simulation.matlab_path,
        )
        optimizer = MatlabRLOptimizer(
            simulator=simulator,
            parameter_experience=None,
            reflection_agent=None,
            hidden_dim=rl_cfg.hidden_dim,
            lr_actor=rl_cfg.lr_actor,
            lr_critic=rl_cfg.lr_critic,
            gamma=rl_cfg.gamma,
            clip_ratio=rl_cfg.clip_ratio,
            max_episodes=max_iterations,
            episodes_per_update=rl_cfg.episodes_per_update,
            nmc_per_eval=100,
            peak_n_max=getattr(rl_cfg, "peak_n_max", PEAK_NY_MAX),
            peak_n_penalty=rl_cfg.peak_n_penalty,
            entropy_coef=getattr(rl_cfg, "entropy_coef", 0.05),
            pitch_log_std=rl_cfg.exploration.pitch_log_std,
            other_log_std=rl_cfg.exploration.other_log_std,
            warm_start_std_scale=rl_cfg.exploration.warm_start_std_scale,
            baseline_reward_threshold=rl_cfg.exploration.baseline_reward_threshold,
            diverge_penalty_episodes=rl_cfg.exploration.diverge_penalty_episodes,
            diverge_log_std_delta=rl_cfg.exploration.diverge_log_std_delta,
            diverge_reward_threshold=rl_cfg.exploration.diverge_reward_threshold,
            reflect_every=rl_cfg.early_stop.reflect_every,
            early_stop_enabled=rl_cfg.early_stop.enabled,
            early_stop_min_episodes=rl_cfg.early_stop.min_episodes,
            early_stop_patience=rl_cfg.early_stop.patience,
            early_stop_plateau_delta=rl_cfg.early_stop.reward_plateau_delta,
            early_stop_use_rule_first=rl_cfg.early_stop.use_rule_first,
            early_stop_use_multi_criteria_best=rl_cfg.early_stop.use_multi_criteria_best,
            early_stop_peak_only_patience=rl_cfg.early_stop.peak_only_patience,
            fre_config=cfg.fre,
            reward_weights=rl_cfg.reward_weights.to_dict(),
            checkpoint_enabled=False,
            allow_internal_fallback=getattr(rl_cfg, "allow_internal_fallback", False),
        )
        optimizer.extract_params_from_script(DEFAULT_MATLAB_SCRIPT)
        result = await optimizer.optimize(
            script_path=DEFAULT_MATLAB_SCRIPT,
            mission_conditions={"T": [max(0, int(scenario[1]) - 1)]} if scenario in {"T1", "T2", "T3", "T4"} else {"T": [0]},
            max_episodes=max_iterations,
            nmc=100,
            task_context={"baseline": "direct_ppo", "scenario": scenario},
            task_prompt=f"Direct PPO baseline for {scenario}",
            reflection_agent=None,
            initial_auto_params=initial_params,
            metric_requirements=dict(DEFAULT_METRIC_REQUIREMENTS),
            warm_start_source="deepseek_v4",
        )
        sim_sec = float(getattr(optimizer, "_external_mc_sim_time_sec", 0.0))
        mc_calls = int(getattr(optimizer, "_external_mc_calls", 0))
        return result, sim_sec, mc_calls
    finally:
        cfg.simulation.mc_data_path = previous_mc_path
        if previous_mc_path:
            os.environ["GSM_MC_DATA_PATH"] = previous_mc_path
        else:
            os.environ.pop("GSM_MC_DATA_PATH", None)


def direct_ppo_optimize(
    scenario: str,
    dev_data_path: str,
    test_data_path: str,
    max_iterations: int = DEFAULT_EQUAL_BUDGET,
    seed: int = 42,
) -> Dict[str, Any]:
    """Run Direct PPO without Hermes or memory reuse.

    Args:
        scenario: Scenario key, e.g. ``"T1"``.
        dev_data_path: Development MC ``.mat`` file.
        test_data_path: Test MC ``.mat`` file.
        max_iterations: Maximum PPO episodes.
        seed: Explicit seed. Existing PPO core is additionally seeded via
            ``np.random.seed`` for deterministic network initialization.

    Returns:
        Baseline result dictionary with a unified schema.
    """
    _ensure_input_files(DEFAULT_MATLAB_SCRIPT, dev_data_path, test_data_path)
    logger.info(
        "[DirectPPO] scenario=%s seed=%s max_iterations=%d",
        scenario,
        seed,
        max_iterations,
    )
    np.random.default_rng(seed)
    np.random.seed(seed)
    started = time.perf_counter()
    initial_params, inference_sec = _deepseek_v4_initial_params(scenario)
    ppo_result, ppo_sim_sec, ppo_mc_calls = asyncio.run(
        _run_direct_ppo_async(
            scenario=scenario,
            dev_data_path=dev_data_path,
            max_iterations=max_iterations,
            initial_params=initial_params,
        )
    )
    best_params = _clip_params(ppo_result.get("best_params") or initial_params)
    dev_result = _evaluate(best_params, dev_data_path)
    test_result = _evaluate(best_params, test_data_path)
    sim_sec = (
        ppo_sim_sec
        + float(dev_result.get("sim_time_sec", 0.0) or 0.0)
        + float(test_result.get("sim_time_sec", 0.0) or 0.0)
    )
    wall = time.perf_counter() - started
    orchestration = max(0.0, wall - inference_sec - sim_sec)
    history = ppo_result.get("history") or []
    convergence_log = [
        (int(item.get("episode", idx + 1)), float(item.get("reward", 0.0)))
        for idx, item in enumerate(history)
        if isinstance(item, Mapping)
    ]
    return _baseline_payload(
        best_params=best_params,
        dev_result=dev_result,
        test_result=test_result,
        mc_calls_used=ppo_mc_calls + 2,
        wall_clock_sec=wall,
        inference_sec=inference_sec,
        orchestration_sec=orchestration,
        iterations_used=int(ppo_result.get("total_episodes", max_iterations)),
        design_path=str(ppo_result.get("best_selection", "N/A")),
        feasibility_rate=1.0 if bool(dev_result.get("feasible", False)) else 0.0,
        convergence_log=convergence_log,
    )


__all__ = [
    "PARAM_BOUNDS",
    "PARAM_NAMES",
    "DEFAULT_PARAMS",
    "random_search",
    "cma_es_optimize",
    "direct_ppo_optimize",
    "_fitness",
]
