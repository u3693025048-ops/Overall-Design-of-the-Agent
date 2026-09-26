#!/usr/bin/env python3
"""Section 4.3 optimizer baselines.

This module is intentionally standalone: it does not call Hermes, RAG,
reflection, memory search, design-path selection, or ParameterExperience.  It
only runs raw optimizer loops over the same tunable parameter vector used by the
GSMultiAgent PPO optimizer.

Protocol implemented here:
  - methods: random_search, cma_es, direct_ppo
  - scenarios: T1..T4
  - master seeds: 42 by default (speed profile; pass 42 43 44 for full paper replicates)
  - one baseline run per (method, scenario, master_seed); MC splits via resolve_split_seeds
  - search space: ALL_TUNABLE_PARAM_SPECS (currently 10-D)
  - per-candidate dev evaluation: master_seed * 100 + {1,2,3}, AND pass logic
  - final test evaluation: master_seed * 100 + {4,5,6}, AND pass logic
  - equal Monte Carlo budget: all methods share the same maximum number of
    development MC calls (Reviewer #1 R2).  Default is
    workflow.max_iterations × max_episodes × n_dev_seeds = 5 × 15 × 3 = 225.
    ``--iterations`` / ``--max-candidates`` count *candidate evaluations*,
    never CMA-ES generations.
  - default MC trajectories per call: 100
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from multi_agent.rl.matlab_rl_optimizer import ALL_TUNABLE_PARAM_SPECS
from multi_agent.tools.simulation_tool import RunSimulationTool, check_constraints, run_mc_with_external_data
from experiments.equal_mc_budget import (
    DEFAULT_MAX_CANDIDATES,
    GSM_OUTER_ITERATIONS,
    max_mc_calls as equal_max_mc_calls,
    remaining_candidates,
    split_agent_budget,
)
from experiments.single_agent_hermes.agent import run_single_agent_hermes as _run_sah


METHODS = ("random_search", "cma_es", "direct_ppo", "single_agent_hermes")
SCENARIOS = ("T1", "T2", "T3", "T4")
MASTER_SEEDS = (42,)
DEV_OFFSETS = (1, 2, 3)
TEST_OFFSETS = (4, 5, 6)
DEFAULT_ITERATIONS = DEFAULT_MAX_CANDIDATES  # candidate evaluations, not CMA generations
DEFAULT_N_RUNS = 100
DEFAULT_OUTPUT_DIR = Path("results") / "optimizer_baselines_43"
DEFAULT_SCRIPT = RunSimulationTool.KB_TEMPLATE_PATH

PARAM_NAMES: List[str] = list(ALL_TUNABLE_PARAM_SPECS.keys())
PARAM_BOUNDS: Dict[str, Tuple[float, float]] = {
    name: (float(spec["min"]), float(spec["max"]))
    for name, spec in ALL_TUNABLE_PARAM_SPECS.items()
}
DEFAULT_PARAMS: Dict[str, float] = {
    name: float(spec["nominal"])
    for name, spec in ALL_TUNABLE_PARAM_SPECS.items()
}


def assert_protocol_dimension(expected_dim: int = 10) -> None:
    """Fail fast if the code/search-space drifted away from the paper setup."""
    actual = len(PARAM_NAMES)
    if actual != int(expected_dim):
        raise RuntimeError(
            f"Section 4.3 expects {expected_dim} tunable parameters, got {actual}: "
            f"{PARAM_NAMES}"
        )


def scenario_sub_idx(scenario: str) -> int:
    """Return MATLAB T-subcase index for T1..T4."""
    if scenario not in SCENARIOS:
        raise ValueError(f"Unsupported scenario '{scenario}', expected one of {SCENARIOS}")
    return int(scenario[1])


def mc_data_path(mc_root: Path, scenario: str, seed: int, split: str) -> Path:
    """Return the expected external-MC data path."""
    return mc_root / scenario / f"seed_{int(seed)}_{split}.mat"


def dev_seeds_for_master(master_seed: int) -> List[int]:
    """Return paper-protocol dev seeds: master*100 + {1,2,3}.

    Example: master 42 → [4201, 4202, 4203] → seed_4201_dev.mat …
    """
    master = int(master_seed)
    return [master * 100 + offset for offset in (1, 2, 3)]


def test_seeds_for_master(master_seed: int) -> List[int]:
    """Return paper-protocol test seeds: master*100 + {4,5,6}.

    Example: master 42 → [4204, 4205, 4206] → seed_4204_test.mat …
    """
    master = int(master_seed)
    return [master * 100 + offset for offset in (4, 5, 6)]


def resolve_split_seeds(
    master_seed: int,
    dev_override: Optional[Sequence[int]] = None,
    test_override: Optional[Sequence[int]] = None,
) -> Tuple[List[int], List[int]]:
    """Resolve dev/test MC seeds for a master seed (3+3 AND protocol)."""
    dev = list(dev_override) if dev_override is not None else dev_seeds_for_master(master_seed)
    test = list(test_override) if test_override is not None else test_seeds_for_master(master_seed)
    return [int(v) for v in dev], [int(v) for v in test]


def clip_params(params: Mapping[str, float]) -> Dict[str, float]:
    """Clip/fill a parameter dictionary to the canonical 10-D search space."""
    out: Dict[str, float] = {}
    for name in PARAM_NAMES:
        lo, hi = PARAM_BOUNDS[name]
        value = float(params.get(name, DEFAULT_PARAMS[name]))
        out[name] = float(np.clip(value, lo, hi))
    return out


def normalized_to_params(x: Sequence[float]) -> Dict[str, float]:
    """Map normalized [0, 1]^d coordinates to physical parameter values."""
    arr = np.asarray(x, dtype=float)
    if arr.shape[0] != len(PARAM_NAMES):
        raise ValueError(f"Expected vector length {len(PARAM_NAMES)}, got {arr.shape[0]}")
    params: Dict[str, float] = {}
    for idx, name in enumerate(PARAM_NAMES):
        lo, hi = PARAM_BOUNDS[name]
        z = float(np.clip(arr[idx], 0.0, 1.0))
        params[name] = lo + z * (hi - lo)
    return params


def params_to_normalized(params: Mapping[str, float]) -> np.ndarray:
    """Map physical parameters to normalized [0, 1]^d coordinates."""
    clipped = clip_params(params)
    values: List[float] = []
    for name in PARAM_NAMES:
        lo, hi = PARAM_BOUNDS[name]
        values.append((clipped[name] - lo) / max(hi - lo, 1e-12))
    return np.asarray(values, dtype=float)


def action_to_params(action: Sequence[float]) -> Dict[str, float]:
    """Map PPO action in [-1, 1]^d to physical parameters."""
    action_arr = np.asarray(action, dtype=float)
    z = 0.5 * (np.clip(action_arr, -1.0, 1.0) + 1.0)
    return normalized_to_params(z)


def params_to_action(params: Mapping[str, float]) -> np.ndarray:
    """Map physical parameters to PPO action coordinates [-1, 1]^d."""
    return 2.0 * params_to_normalized(params) - 1.0


def _float(result: Mapping[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(result.get(key, default))
    except (TypeError, ValueError):
        return default


@dataclass
class CandidateEvaluation:
    """Evaluation of one parameter vector over multiple MC seeds."""

    split: str
    seeds: List[int]
    seed_results: List[Dict[str, Any]]
    all_pass: bool
    score: float
    metrics: Dict[str, float]
    mc_calls: int
    sim_time_sec: float
    error: str = ""

    def to_json(self) -> Dict[str, Any]:
        return asdict(self)


def aggregate_seed_results(seed_results: Sequence[Mapping[str, Any]]) -> Tuple[bool, Dict[str, float], float, str]:
    """Aggregate seed-level results with conservative/worst-case metrics."""
    if not seed_results:
        return False, {}, float("inf"), "no seed results"

    errors = [
        str(item.get("error") or item.get("error_msg") or "")
        for item in seed_results
        if not bool(item.get("success", False))
    ]
    hit_vals = [_float(item, "hit_rate") for item in seed_results]
    sep_vals = [_float(item, "SEP", float("inf")) for item in seed_results]
    peak_vals = [
        _float(item, "peak_nz", _float(item, "peak_ny", _float(item, "peak_n", float("inf"))))
        for item in seed_results
    ]
    pm_vals = [_float(item, "PM", _float(item, "pitch_PM")) for item in seed_results]
    bw_vals = [_float(item, "BW", _float(item, "pitch_BW")) for item in seed_results]

    total_violation = 0.0
    all_pass = True
    for item in seed_results:
        feasible = bool(item.get("feasible", False)) and bool(item.get("success", False))
        all_pass = all_pass and feasible
        violations = item.get("constraint_violations") or {}
        if not violations:
            feasible2, violations = check_constraints(
                _float(item, "hit_rate"),
                _float(item, "SEP", float("inf")),
                _float(item, "peak_nz", _float(item, "peak_ny", float("inf"))),
                _float(item, "PM", _float(item, "pitch_PM")),
                _float(item, "BW", _float(item, "pitch_BW")),
            )
            all_pass = all_pass and feasible2
        total_violation += sum(float(v) for v in violations.values())

    metrics = {
        "hit_rate": min(hit_vals) if hit_vals else 0.0,
        "SEP": max(sep_vals) if sep_vals else float("inf"),
        "peak_nz": max(peak_vals) if peak_vals else float("inf"),
        "PM_min": min(pm_vals) if pm_vals else 0.0,
        "PM_max": max(pm_vals) if pm_vals else 0.0,
        "BW_min": min(bw_vals) if bw_vals else 0.0,
        "BW_max": max(bw_vals) if bw_vals else 0.0,
        "PM": min(pm_vals) if pm_vals else 0.0,
        "BW": min(bw_vals) if bw_vals else 0.0,
        "violation_sum": float(total_violation),
    }
    if all_pass:
        score = -1.0 - metrics["hit_rate"] / 100.0 + 0.001 * metrics["SEP"]
    else:
        score = total_violation + 1e6 * len(errors)
    return all_pass, metrics, float(score), "; ".join(e for e in errors if e)


class ExternalMCEvaluator:
    """External-MC evaluator shared by all baselines."""

    def __init__(
        self,
        *,
        mc_root: Path,
        matlab_script: Path,
        n_runs: int,
        strict_paths: bool = True,
    ) -> None:
        self.mc_root = Path(mc_root)
        self.matlab_script = Path(matlab_script)
        self.n_runs = int(n_runs)
        self.strict_paths = bool(strict_paths)

    def evaluate(
        self,
        *,
        params: Mapping[str, float],
        scenario: str,
        seeds: Sequence[int],
        split: str,
    ) -> CandidateEvaluation:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        from multi_agent.simulation.matlab_parallel import resolve_mc_and_workers

        seed_list = [int(s) for s in seeds]
        workers = resolve_mc_and_workers(len(seed_list))

        def _eval_one(seed: int) -> Dict[str, Any]:
            data_path = mc_data_path(self.mc_root, scenario, seed, split)
            if not data_path.exists():
                msg = f"missing MC data: {data_path}"
                if self.strict_paths:
                    raise FileNotFoundError(msg)
                return {
                    "success": False,
                    "error": msg,
                    "error_msg": msg,
                    "feasible": False,
                    "constraint_violations": {"missing_data": 1e6},
                    "sim_time_sec": 0.0,
                    "mc_seed": seed,
                    "split": split,
                }
            n_runs = int(self.n_runs)
            try:
                from multi_agent.config_loader import get_config

                sim = get_config().simulation
                if bool(getattr(sim, "mc_tiered_eval", False)):
                    if split == "dev":
                        n_runs = int(getattr(sim, "mc_search_n_runs", n_runs) or n_runs)
                    else:
                        n_runs = int(getattr(sim, "mc_final_n_runs", n_runs) or n_runs)
            except Exception:
                pass
            result = run_mc_with_external_data(
                str(self.matlab_script),
                str(data_path),
                n_runs=n_runs,
                kill_radius=10.0,
                dp_params=clip_params(params),
            )
            return dict(result, mc_seed=seed, split=split)

        if workers <= 1 or len(seed_list) <= 1:
            seed_results = [_eval_one(seed) for seed in seed_list]
        else:
            by_seed: Dict[int, Dict[str, Any]] = {}
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {pool.submit(_eval_one, seed): seed for seed in seed_list}
                for fut in as_completed(futures):
                    seed = futures[fut]
                    by_seed[seed] = fut.result()
            seed_results = [by_seed[s] for s in seed_list]

        sim_time = sum(_float(result, "sim_time_sec") for result in seed_results)
        all_pass, metrics, score, agg_error = aggregate_seed_results(seed_results)
        error = agg_error
        return CandidateEvaluation(
            split=split,
            seeds=seed_list,
            seed_results=seed_results,
            all_pass=all_pass,
            score=score,
            metrics=metrics,
            mc_calls=len(seed_list),
            sim_time_sec=sim_time,
            error=error,
        )


@dataclass
class IterationRecord:
    iteration: int
    candidate_index: int
    params: Dict[str, float]
    dev: Dict[str, Any]
    score: float
    is_best: bool


@dataclass
class BaselineRunResult:
    method: str
    scenario: str
    master_seed: int
    search_dim: int
    iterations_budget: int
    iterations_used: int
    mc_calls_dev: int
    mc_calls_test: int
    wall_clock_sec: float
    sim_time_sec: float
    first_success_iter: Optional[int]
    dev_all_pass: bool
    test_all_pass: bool
    best_score: float
    best_params: Dict[str, float]
    best_dev: Dict[str, Any]
    test_eval: Dict[str, Any]
    history: List[Dict[str, Any]] = field(default_factory=list)
    error: str = ""
    design_path: str = "N/A"
    first_success_definition: str = ""
    n_runs_search: int = 0
    n_runs_test: int = 0

    @property
    def time_per_call_min(self) -> float:
        total_calls = self.mc_calls_dev + self.mc_calls_test
        if total_calls <= 0:
            return float("nan")
        return (self.wall_clock_sec / 60.0) / total_calls

    def to_json(self) -> Dict[str, Any]:
        out = asdict(self)
        out["time_per_call_min"] = self.time_per_call_min
        return out


def _better(a_score: float, b_score: float) -> bool:
    return float(a_score) < float(b_score)


def run_random_search(
    evaluator: ExternalMCEvaluator,
    *,
    scenario: str,
    master_seed: int,
    dev_seeds: Sequence[int],
    test_seeds: Sequence[int],
    iterations: int,
) -> BaselineRunResult:
    """Uniform random search over [0, 1]^10."""
    rng = np.random.default_rng(int(master_seed))
    started = time.perf_counter()
    best_params = dict(DEFAULT_PARAMS)
    best_dev: Optional[CandidateEvaluation] = None
    best_score = float("inf")
    first_success: Optional[int] = None
    history: List[Dict[str, Any]] = []
    mc_calls_dev = 0
    sim_time = 0.0

    n_seeds = max(1, len(dev_seeds))
    max_mc = equal_max_mc_calls(n_dev_seeds=n_seeds, max_candidates=int(iterations))
    iteration = 0
    for iteration in range(1, int(iterations) + 1):
        if remaining_candidates(mc_calls_dev, max_mc, n_seeds) <= 0:
            break
        x = rng.uniform(0.0, 1.0, size=len(PARAM_NAMES))
        params = normalized_to_params(x)
        dev = evaluator.evaluate(params=params, scenario=scenario, seeds=dev_seeds, split="dev")
        mc_calls_dev += dev.mc_calls
        sim_time += dev.sim_time_sec
        if dev.all_pass and first_success is None:
            first_success = iteration
        is_best = best_dev is None or _better(dev.score, best_score)
        if is_best:
            best_params = params
            best_dev = dev
            best_score = dev.score
        history.append(asdict(IterationRecord(
            iteration=iteration,
            candidate_index=1,
            params=params,
            dev=dev.to_json(),
            score=dev.score,
            is_best=is_best,
        )))

    assert best_dev is not None
    test_eval = evaluator.evaluate(params=best_params, scenario=scenario, seeds=test_seeds, split="test")
    sim_time += test_eval.sim_time_sec
    return BaselineRunResult(
        method="random_search",
        scenario=scenario,
        master_seed=int(master_seed),
        search_dim=len(PARAM_NAMES),
        iterations_budget=int(iterations),
        iterations_used=int(iteration),
        mc_calls_dev=mc_calls_dev,
        mc_calls_test=test_eval.mc_calls,
        wall_clock_sec=time.perf_counter() - started,
        sim_time_sec=sim_time,
        first_success_iter=first_success,
        dev_all_pass=best_dev.all_pass,
        test_all_pass=test_eval.all_pass,
        best_score=best_score,
        best_params=best_params,
        best_dev=best_dev.to_json(),
        test_eval=test_eval.to_json(),
        history=history,
    )


class NumpyCMAES:
    """Small full-covariance CMA-ES implementation in normalized coordinates."""

    def __init__(self, x0: np.ndarray, sigma0: float, popsize: int, seed: int) -> None:
        self.n = int(x0.shape[0])
        self.mean = np.asarray(x0, dtype=float).copy()
        self.sigma = float(sigma0)
        self.popsize = int(popsize)
        self.mu = self.popsize // 2
        weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights = weights / np.sum(weights)
        self.mueff = float(1.0 / np.sum(self.weights ** 2))
        n = self.n
        self.cc = (4.0 + self.mueff / n) / (n + 4.0 + 2.0 * self.mueff / n)
        self.cs = (self.mueff + 2.0) / (n + self.mueff + 5.0)
        self.c1 = 2.0 / ((n + 1.3) ** 2 + self.mueff)
        self.cmu = min(
            1.0 - self.c1,
            2.0 * (self.mueff - 2.0 + 1.0 / self.mueff) / ((n + 2.0) ** 2 + self.mueff),
        )
        self.damps = (
            1.0
            + 2.0 * max(0.0, math.sqrt((self.mueff - 1.0) / (n + 1.0)) - 1.0)
            + self.cs
        )
        self.pc = np.zeros(n)
        self.ps = np.zeros(n)
        self.cov = np.eye(n)
        self.bmat = np.eye(n)
        self.dvec = np.ones(n)
        self.invsqrt = np.eye(n)
        self.chi_n = math.sqrt(n) * (1.0 - 1.0 / (4.0 * n) + 1.0 / (21.0 * n * n))
        self.rng = np.random.default_rng(int(seed))
        self.generation = 0

    def ask(self) -> Tuple[np.ndarray, np.ndarray]:
        z = self.rng.normal(size=(self.popsize, self.n))
        y = z @ (self.bmat * self.dvec).T
        x = np.clip(self.mean + self.sigma * y, 0.0, 1.0)
        return x, z

    def tell(self, xs: np.ndarray, fitnesses: Sequence[float]) -> None:
        order = np.argsort(np.asarray(fitnesses, dtype=float))
        selected = xs[order[: self.mu]]
        old_mean = self.mean.copy()
        self.mean = np.sum(selected * self.weights[:, None], axis=0)
        y_k = (selected - old_mean) / max(self.sigma, 1e-12)
        y_w = np.sum(y_k * self.weights[:, None], axis=0)
        z_w = self.invsqrt @ y_w
        self.ps = (
            (1.0 - self.cs) * self.ps
            + math.sqrt(self.cs * (2.0 - self.cs) * self.mueff) * z_w
        )
        ps_norm = np.linalg.norm(self.ps)
        hsig = float(
            ps_norm
            / math.sqrt(1.0 - (1.0 - self.cs) ** (2.0 * (self.generation + 1)))
            / self.chi_n
            < (1.4 + 2.0 / (self.n + 1.0))
        )
        self.pc = (
            (1.0 - self.cc) * self.pc
            + hsig * math.sqrt(self.cc * (2.0 - self.cc) * self.mueff) * y_w
        )
        rank_mu = np.zeros_like(self.cov)
        for weight, y in zip(self.weights, y_k):
            rank_mu += weight * np.outer(y, y)
        self.cov = (
            (1.0 - self.c1 - self.cmu) * self.cov
            + self.c1
            * (
                np.outer(self.pc, self.pc)
                + (1.0 - hsig) * self.cc * (2.0 - self.cc) * self.cov
            )
            + self.cmu * rank_mu
        )
        self.sigma *= math.exp((self.cs / self.damps) * (ps_norm / self.chi_n - 1.0))
        self._update_eigensystem()
        self.generation += 1

    def _update_eigensystem(self) -> None:
        self.cov = np.triu(self.cov) + np.triu(self.cov, 1).T
        vals, vecs = np.linalg.eigh(self.cov)
        vals = np.maximum(vals, 1e-12)
        self.dvec = np.sqrt(vals)
        self.bmat = vecs
        self.invsqrt = vecs @ np.diag(1.0 / self.dvec) @ vecs.T


def run_cma_es(
    evaluator: ExternalMCEvaluator,
    *,
    scenario: str,
    master_seed: int,
    dev_seeds: Sequence[int],
    test_seeds: Sequence[int],
    iterations: int,
    sigma0: float = 0.25,
) -> BaselineRunResult:
    """CMA-ES with λ = 4 + floor(3 ln d), σ₀ = 0.25.

    ``iterations`` is the shared *candidate-evaluation* ceiling, not the
    number of CMA generations.  Search stops when development MC calls
    reach ``iterations × len(dev_seeds)``.
    """
    try:
        if hasattr(evaluator, "mc_root"):
            from multi_agent.simulation.pipeline_warmup import warmup_scenario

            warmup_scenario(scenario, master_seed=int(master_seed), include_rag=False)
    except Exception:
        pass
    dim = len(PARAM_NAMES)
    popsize = int(4 + math.floor(3.0 * math.log(dim)))
    cma = NumpyCMAES(params_to_normalized(DEFAULT_PARAMS), sigma0, popsize, int(master_seed))
    started = time.perf_counter()
    best_params = dict(DEFAULT_PARAMS)
    best_dev: Optional[CandidateEvaluation] = None
    best_score = float("inf")
    first_success: Optional[int] = None
    history: List[Dict[str, Any]] = []
    mc_calls_dev = 0
    sim_time = 0.0

    n_seeds = max(1, len(dev_seeds))
    max_mc = equal_max_mc_calls(n_dev_seeds=n_seeds, max_candidates=int(iterations))
    generation = 0
    while remaining_candidates(mc_calls_dev, max_mc, n_seeds) > 0:
        generation += 1
        n_ask = remaining_candidates(mc_calls_dev, max_mc, n_seeds)
        n_ask = min(cma.popsize, n_ask)
        if n_ask <= 0:
            break
        xs_full, _ = cma.ask()
        xs = xs_full[:n_ask]
        fitnesses: List[float] = [0.0] * len(xs)
        cand_workers = 1
        try:
            from multi_agent.config_loader import get_config
            from multi_agent.simulation.matlab_parallel import resolve_matlab_max_parallel

            cand_workers = max(
                1,
                min(
                    len(xs),
                    int(getattr(get_config().simulation, "mc_candidate_workers", 3) or 3),
                    resolve_matlab_max_parallel(),
                ),
            )
        except Exception:
            cand_workers = 1

        def _eval_candidate(idx_x: tuple[int, Any]) -> tuple[int, Dict[str, float], CandidateEvaluation]:
            idx, x = idx_x
            params = normalized_to_params(x)
            dev = evaluator.evaluate(
                params=params, scenario=scenario, seeds=dev_seeds, split="dev"
            )
            return idx, params, dev

        indexed = list(enumerate(xs, start=1))
        results: List[tuple[int, Dict[str, float], CandidateEvaluation]]
        if cand_workers <= 1 or len(indexed) <= 1:
            results = [_eval_candidate(item) for item in indexed]
        else:
            from concurrent.futures import ThreadPoolExecutor, as_completed

            results = []
            with ThreadPoolExecutor(max_workers=cand_workers) as pool:
                futs = [pool.submit(_eval_candidate, item) for item in indexed]
                for fut in as_completed(futs):
                    results.append(fut.result())
            results.sort(key=lambda t: t[0])

        for idx, params, dev in results:
            fitnesses[idx - 1] = dev.score
            mc_calls_dev += dev.mc_calls
            sim_time += dev.sim_time_sec
            if dev.all_pass and first_success is None:
                first_success = generation
            is_best = best_dev is None or _better(dev.score, best_score)
            if is_best:
                best_params = params
                best_dev = dev
                best_score = dev.score
            history.append(asdict(IterationRecord(
                iteration=generation,
                candidate_index=idx,
                params=params,
                dev=dev.to_json(),
                score=dev.score,
                is_best=is_best,
            )))
        # Truncated final generation is evaluated but does not update CMA.
        if n_ask == cma.popsize:
            cma.tell(xs, fitnesses)

    assert best_dev is not None
    test_eval = evaluator.evaluate(params=best_params, scenario=scenario, seeds=test_seeds, split="test")
    sim_time += test_eval.sim_time_sec
    return BaselineRunResult(
        method="cma_es",
        scenario=scenario,
        master_seed=int(master_seed),
        search_dim=dim,
        iterations_budget=int(iterations),
        iterations_used=generation,
        mc_calls_dev=mc_calls_dev,
        mc_calls_test=test_eval.mc_calls,
        wall_clock_sec=time.perf_counter() - started,
        sim_time_sec=sim_time,
        first_success_iter=first_success,
        dev_all_pass=best_dev.all_pass,
        test_all_pass=test_eval.all_pass,
        best_score=best_score,
        best_params=best_params,
        best_dev=best_dev.to_json(),
        test_eval=test_eval.to_json(),
        history=history,
    )


def run_direct_ppo(
    evaluator: ExternalMCEvaluator,
    *,
    scenario: str,
    master_seed: int,
    dev_seeds: Sequence[int],
    test_seeds: Sequence[int],
    iterations: int,
    lr: float = 3e-2,
    clip_ratio: float = 0.2,
    entropy_coef: float = 0.01,
    ppo_epochs: int = 8,
) -> BaselineRunResult:
    """Direct PPO baseline with no agentic infrastructure."""
    try:
        import torch
    except ImportError as exc:
        raise ImportError("direct_ppo requires torch; install requirements.txt") from exc

    torch.manual_seed(int(master_seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(master_seed))
    rng = np.random.default_rng(int(master_seed))
    del rng  # documents optimizer seed reset; torch controls PPO sampling below

    # Prefer CUDA on GPU hosts (e.g. Ubuntu RTX). Policy is tiny; wall-clock
    # remains dominated by MATLAB MC evaluations, but updates run on GPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"[direct_ppo] device={device}"
        + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""),
        flush=True,
    )
    action0 = params_to_action(DEFAULT_PARAMS)
    mean = torch.nn.Parameter(torch.tensor(action0, dtype=torch.float32, device=device))
    log_std = torch.nn.Parameter(
        torch.full((len(PARAM_NAMES),), math.log(0.35), dtype=torch.float32, device=device)
    )
    value = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float32, device=device))
    optim = torch.optim.Adam([mean, log_std, value], lr=float(lr))

    started = time.perf_counter()
    best_params = dict(DEFAULT_PARAMS)
    best_dev: Optional[CandidateEvaluation] = None
    best_score = float("inf")
    first_success: Optional[int] = None
    history: List[Dict[str, Any]] = []
    mc_calls_dev = 0
    sim_time = 0.0
    reward_baseline = 0.0
    n_seeds = max(1, len(dev_seeds))
    max_mc = equal_max_mc_calls(n_dev_seeds=n_seeds, max_candidates=int(iterations))
    iteration = 0

    for iteration in range(1, int(iterations) + 1):
        if remaining_candidates(mc_calls_dev, max_mc, n_seeds) <= 0:
            break
        with torch.no_grad():
            std = torch.exp(log_std).clamp(1e-3, 2.0)
            dist = torch.distributions.Normal(mean, std)
            raw_action = dist.sample()
            old_log_prob = dist.log_prob(raw_action).sum()
            action = torch.clamp(raw_action, -1.0, 1.0).cpu().numpy()

        params = action_to_params(action)
        dev = evaluator.evaluate(params=params, scenario=scenario, seeds=dev_seeds, split="dev")
        mc_calls_dev += dev.mc_calls
        sim_time += dev.sim_time_sec
        reward = float(-dev.score)
        reward_baseline = 0.8 * reward_baseline + 0.2 * reward
        if dev.all_pass and first_success is None:
            first_success = iteration
        is_best = best_dev is None or _better(dev.score, best_score)
        if is_best:
            best_params = params
            best_dev = dev
            best_score = dev.score

        reward_t = torch.tensor(reward, dtype=torch.float32, device=device)
        old_lp = old_log_prob.detach()
        old_action = raw_action.detach()
        old_value = value.detach()
        advantage = (reward_t - old_value).detach()
        for _ in range(int(ppo_epochs)):
            std = torch.exp(log_std).clamp(1e-3, 2.0)
            dist = torch.distributions.Normal(mean, std)
            log_prob = dist.log_prob(old_action).sum()
            ratio = torch.exp(log_prob - old_lp)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * advantage
            actor_loss = -torch.min(surr1, surr2)
            value_loss = 0.5 * (value - reward_t).pow(2)
            entropy = dist.entropy().sum()
            loss = actor_loss + value_loss - float(entropy_coef) * entropy
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([mean, log_std, value], 1.0)
            optim.step()
            with torch.no_grad():
                mean.clamp_(-1.0, 1.0)
                log_std.clamp_(math.log(0.03), math.log(1.0))

        history.append(asdict(IterationRecord(
            iteration=iteration,
            candidate_index=1,
            params=params,
            dev=dev.to_json(),
            score=dev.score,
            is_best=is_best,
        )))
        history[-1]["reward"] = reward
        history[-1]["reward_ema"] = reward_baseline

    assert best_dev is not None
    test_eval = evaluator.evaluate(params=best_params, scenario=scenario, seeds=test_seeds, split="test")
    sim_time += test_eval.sim_time_sec
    return BaselineRunResult(
        method="direct_ppo",
        scenario=scenario,
        master_seed=int(master_seed),
        search_dim=len(PARAM_NAMES),
        iterations_budget=int(iterations),
        iterations_used=int(iteration),
        mc_calls_dev=mc_calls_dev,
        mc_calls_test=test_eval.mc_calls,
        wall_clock_sec=time.perf_counter() - started,
        sim_time_sec=sim_time,
        first_success_iter=first_success,
        dev_all_pass=best_dev.all_pass,
        test_all_pass=test_eval.all_pass,
        best_score=best_score,
        best_params=best_params,
        best_dev=best_dev.to_json(),
        test_eval=test_eval.to_json(),
        history=history,
    )


def run_one(
    method: str,
    evaluator: ExternalMCEvaluator,
    *,
    scenario: str,
    master_seed: int,
    dev_seeds: Sequence[int],
    test_seeds: Sequence[int],
    iterations: int,
) -> BaselineRunResult:
    """Dispatch a single method/scenario/master-seed run."""
    if method == "random_search":
        return run_random_search(
            evaluator,
            scenario=scenario,
            master_seed=master_seed,
            dev_seeds=dev_seeds,
            test_seeds=test_seeds,
            iterations=iterations,
        )
    if method == "cma_es":
        return run_cma_es(
            evaluator,
            scenario=scenario,
            master_seed=master_seed,
            dev_seeds=dev_seeds,
            test_seeds=test_seeds,
            iterations=iterations,
        )
    if method == "direct_ppo":
        return run_direct_ppo(
            evaluator,
            scenario=scenario,
            master_seed=master_seed,
            dev_seeds=dev_seeds,
            test_seeds=test_seeds,
            iterations=iterations,
        )
    if method == "single_agent_hermes":
        n_outer, n_ep = split_agent_budget(int(iterations), GSM_OUTER_ITERATIONS)
        pe_raw = os.environ.get("GSM_SAH_PE_DIR", "").strip()
        result_dict = _run_sah(
            evaluator_mc_root=evaluator.mc_root,
            matlab_script=evaluator.matlab_script,
            scenario=scenario,
            master_seed=master_seed,
            dev_seeds=dev_seeds,
            test_seeds=test_seeds,
            n_outer_iterations=n_outer,
            n_episodes_per_outer=n_ep,
            n_runs=evaluator.n_runs,
            pe_dir=Path(pe_raw) if pe_raw else None,
        )
        return BaselineRunResult(**{
            k: result_dict[k]
            for k in BaselineRunResult.__dataclass_fields__
            if k in result_dict
        })
    raise ValueError(f"Unknown method: {method}")


RUN_CSV_COLUMNS = [
    "method",
    "scenario",
    "master_seed",
    "search_dim",
    "first_success_iter",
    "first_success",
    "dev_all_pass",
    "test_all_pass",
    "total_time_min",
    "iterations_used",
    "design_path",
    "mc_calls_dev",
    "mc_calls_test",
    "time_per_call_min",
    "best_score",
    "dev_hit_rate",
    "dev_SEP",
    "dev_peak_nz",
    "dev_PM_min",
    "dev_PM_max",
    "dev_BW_min",
    "dev_BW_max",
    "test_hit_rate",
    "test_SEP",
    "test_peak_nz",
    "test_PM_min",
    "test_PM_max",
    "test_BW_min",
    "test_BW_max",
    "json_path",
    "error",
]


def _metric_first(metrics: Dict[str, Any], *keys: str, default: Any = "") -> Any:
    for key in keys:
        value = metrics.get(key)
        if value not in (None, ""):
            return value
    return default


def run_to_row(result: BaselineRunResult, json_path: Path) -> Dict[str, Any]:
    """Flatten a run result into a CSV row."""
    dev_metrics = (result.best_dev.get("metrics") or {}) if result.best_dev else {}
    test_metrics = (result.test_eval.get("metrics") or {}) if result.test_eval else {}
    return {
        "method": result.method,
        "scenario": result.scenario,
        "master_seed": result.master_seed,
        "search_dim": result.search_dim,
        "first_success_iter": result.first_success_iter or "",
        # GSM Table 5 / SAH: first_success = final development feasibility.
        # Random / CMA / PPO keep the first-candidate-all_pass definition.
        "first_success": (
            bool(result.dev_all_pass)
            if result.method == "single_agent_hermes"
            else result.first_success_iter is not None
        ),
        "dev_all_pass": result.dev_all_pass,
        "test_all_pass": result.test_all_pass,
        "total_time_min": result.wall_clock_sec / 60.0,
        "iterations_used": result.iterations_used,
        "design_path": getattr(result, "design_path", "") or "N/A",
        "mc_calls_dev": result.mc_calls_dev,
        "mc_calls_test": result.mc_calls_test,
        "time_per_call_min": result.time_per_call_min,
        "best_score": result.best_score,
        "dev_hit_rate": _metric_first(dev_metrics, "hit_rate"),
        "dev_SEP": _metric_first(dev_metrics, "SEP"),
        "dev_peak_nz": _metric_first(dev_metrics, "peak_nz"),
        "dev_PM_min": _metric_first(dev_metrics, "PM_min", "PM"),
        "dev_PM_max": _metric_first(dev_metrics, "PM_max", "PM"),
        "dev_BW_min": _metric_first(dev_metrics, "BW_min", "BW"),
        "dev_BW_max": _metric_first(dev_metrics, "BW_max", "BW"),
        "test_hit_rate": _metric_first(test_metrics, "hit_rate"),
        "test_SEP": _metric_first(test_metrics, "SEP"),
        "test_peak_nz": _metric_first(test_metrics, "peak_nz"),
        "test_PM_min": _metric_first(test_metrics, "PM_min", "PM"),
        "test_PM_max": _metric_first(test_metrics, "PM_max", "PM"),
        "test_BW_min": _metric_first(test_metrics, "BW_min", "BW"),
        "test_BW_max": _metric_first(test_metrics, "BW_max", "BW"),
        "json_path": str(json_path.as_posix()),
        "error": result.error,
    }


def _mean_sd(values: Sequence[float]) -> Tuple[float, float]:
    arr = np.asarray([float(v) for v in values], dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan")
    return float(np.mean(arr)), float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0


def write_summary(rows: Sequence[Mapping[str, Any]], summary_csv: Path) -> None:
    """Write Table-7-style mean/SD rows grouped by scenario and method."""
    grouped: Dict[Tuple[str, str], List[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((str(row["scenario"]), str(row["method"])), []).append(row)

    columns = [
        "scenario",
        "method",
        "first_success",
        "total_time_min_mean",
        "total_time_min_sd",
        "iterations_mean",
        "iterations_sd",
        "mc_calls_mean",
        "mc_calls_sd",
        "time_per_call_min_mean",
        "time_per_call_min_sd",
        "test_pass",
    ]
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    with summary_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        for scenario in SCENARIOS:
            for method in METHODS:
                items = grouped.get((scenario, method), [])
                if not items:
                    continue
                time_mean, time_sd = _mean_sd([float(r["total_time_min"]) for r in items])
                iter_mean, iter_sd = _mean_sd([float(r["iterations_used"]) for r in items])
                calls_mean, calls_sd = _mean_sd([float(r["mc_calls_dev"]) for r in items])
                tpc_mean, tpc_sd = _mean_sd([float(r["time_per_call_min"]) for r in items])
                first_count = sum(str(r["first_success"]).lower() == "true" for r in items)
                test_count = sum(str(r["test_all_pass"]).lower() == "true" for r in items)
                writer.writerow({
                    "scenario": scenario,
                    "method": method,
                    "first_success": f"{first_count}/{len(items)}",
                    "total_time_min_mean": time_mean,
                    "total_time_min_sd": time_sd,
                    "iterations_mean": iter_mean,
                    "iterations_sd": iter_sd,
                    "mc_calls_mean": calls_mean,
                    "mc_calls_sd": calls_sd,
                    "time_per_call_min_mean": tpc_mean,
                    "time_per_call_min_sd": tpc_sd,
                    "test_pass": f"{test_count}/{len(items)}",
                })


def write_protocol(output_dir: Path, args: argparse.Namespace) -> None:
    """Record the exact protocol used for a run."""
    dev_override = getattr(args, "dev_seeds", None)
    test_override = getattr(args, "test_seeds", None)
    payload = {
        "methods": list(args.methods),
        "scenarios": list(args.scenarios),
        "master_seeds": list(args.master_seeds),
        "seed_protocol": "explicit_override" if dev_override is not None or test_override is not None else "derived_from_master",
        "dev_seed_rule": f"master_seed * 100 + {list(DEV_OFFSETS)}",
        "test_seed_rule": f"master_seed * 100 + {list(TEST_OFFSETS)}",
        "dev_seeds_override": list(dev_override) if dev_override is not None else None,
        "test_seeds_override": list(test_override) if test_override is not None else None,
        "iterations": int(args.iterations),
        "max_candidates": int(args.iterations),
        "max_mc_calls_dev": equal_max_mc_calls(
            n_dev_seeds=len(DEV_OFFSETS),
            max_candidates=int(args.iterations),
        ),
        "budget_unit": "candidate_evaluations",
        "cma_es_generation_note": (
            "CMA-ES stops at the shared candidate ceiling; a truncated final "
            "generation is evaluated but does not update the covariance."
        ),
        "n_runs": int(args.n_runs),
        "search_dim": len(PARAM_NAMES),
        "param_names": PARAM_NAMES,
        "param_bounds": PARAM_BOUNDS,
        "default_params": DEFAULT_PARAMS,
        "matlab_script": str(Path(args.matlab_script).as_posix()),
        "mc_root": str(Path(args.mc_root).as_posix()),
    }
    try:
        from multi_agent.config_loader import get_config

        sim = get_config().simulation
        payload["mc_tiered_eval"] = bool(getattr(sim, "mc_tiered_eval", False))
        payload["mc_search_n_runs"] = int(getattr(sim, "mc_search_n_runs", 10) or 10)
        payload["mc_final_n_runs"] = int(getattr(sim, "mc_final_n_runs", 100) or 100)
    except Exception:
        payload["mc_tiered_eval"] = None
    payload["sah_bootstrap_fill"] = False
    payload["first_success_definition_gsm_sah"] = "final_dev_feasible"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "protocol.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def parse_int_list(values: Optional[Iterable[str]], default: Sequence[int]) -> List[int]:
    if not values:
        return [int(v) for v in default]
    out: List[int] = []
    for value in values:
        for part in str(value).replace(",", " ").split():
            if part.strip():
                out.append(int(part))
    return out


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Section 4.3 raw optimizer baselines."
    )
    parser.add_argument("--methods", nargs="+", default=list(METHODS), choices=METHODS)
    parser.add_argument("--scenarios", nargs="+", default=list(SCENARIOS), choices=SCENARIOS)
    parser.add_argument("--master-seeds", nargs="*", default=[str(s) for s in MASTER_SEEDS])
    parser.add_argument(
        "--dev-seeds",
        nargs="*",
        default=None,
        help="Override dev MC seeds for every master seed. Default: master*100 + 1,2,3.",
    )
    parser.add_argument(
        "--test-seeds",
        nargs="*",
        default=None,
        help="Override test MC seeds for every master seed. Default: master*100 + 4,5,6.",
    )
    parser.add_argument(
        "--iterations",
        "--max-candidates",
        dest="iterations",
        type=int,
        default=DEFAULT_ITERATIONS,
        help=(
            "Shared candidate-evaluation ceiling for every method "
            f"(default {DEFAULT_ITERATIONS} = "
            f"{GSM_OUTER_ITERATIONS} outer × 15 PPO episodes). "
            "Each candidate costs len(dev_seeds) Monte Carlo calls. "
            "This is NOT the number of CMA-ES generations."
        ),
    )
    parser.add_argument("--n-runs", type=int, default=DEFAULT_N_RUNS)
    parser.add_argument("--mc-root", default="data/mc")
    parser.add_argument("--matlab-script", default=DEFAULT_SCRIPT)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--expected-dim", type=int, default=10)
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="Write error JSON rows and continue after a failed run.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    args.master_seeds = parse_int_list(args.master_seeds, MASTER_SEEDS)
    args.dev_seeds = parse_int_list(args.dev_seeds, []) if args.dev_seeds is not None else None
    args.test_seeds = parse_int_list(args.test_seeds, []) if args.test_seeds is not None else None
    if args.dev_seeds is not None and not args.dev_seeds:
        parser.error("--dev-seeds requires at least one seed when specified")
    if args.test_seeds is not None and not args.test_seeds:
        parser.error("--test-seeds requires at least one seed when specified")
    assert_protocol_dimension(args.expected_dim)

    output_dir = Path(args.output_dir)
    runs_dir = output_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    write_protocol(output_dir, args)

    evaluator = ExternalMCEvaluator(
        mc_root=Path(args.mc_root),
        matlab_script=Path(args.matlab_script),
        n_runs=int(args.n_runs),
    )
    run_csv = output_dir / "runs.csv"
    summary_csv = output_dir / "summary_table7.csv"

    # Preserve prior runs so --master-seeds 44 backfills do not wipe seed43 rows.
    rows_by_key: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    if run_csv.is_file():
        with run_csv.open("r", encoding="utf-8-sig", newline="") as fh:
            for row in csv.DictReader(fh):
                key = (
                    str(row.get("method", "")),
                    str(row.get("scenario", "")),
                    str(row.get("master_seed", "")),
                )
                rows_by_key[key] = dict(row)

    for scenario in args.scenarios:
        for method in args.methods:
            # One run per (method, scenario, master_seed).
            # Default MC splits follow paper protocol via resolve_split_seeds:
            #   master M → dev=[M*100+1..3], test=[M*100+4..6]
            # Optional --dev-seeds / --test-seeds override those lists for every master.
            for master_seed in args.master_seeds:
                master_seed = int(master_seed)
                dev_seeds, test_seeds = resolve_split_seeds(
                    master_seed,
                    args.dev_seeds,
                    args.test_seeds,
                )

                run_name = f"{scenario}_{method}_seed{master_seed}"
                json_path = runs_dir / f"{run_name}.json"
                try:
                    print(
                        f"[4.3] {run_name}: start "
                        f"dev_seeds={dev_seeds} test_seeds={test_seeds}",
                        flush=True,
                    )
                    result = run_one(
                        method,
                        evaluator,
                        scenario=scenario,
                        master_seed=int(master_seed),
                        dev_seeds=dev_seeds,
                        test_seeds=test_seeds,
                        iterations=int(args.iterations),
                    )
                except Exception as exc:
                    if not args.keep_going:
                        raise
                    result = BaselineRunResult(
                        method=method,
                        scenario=scenario,
                        master_seed=int(master_seed),
                        search_dim=len(PARAM_NAMES),
                        iterations_budget=int(args.iterations),
                        iterations_used=0,
                        mc_calls_dev=0,
                        mc_calls_test=0,
                        wall_clock_sec=0.0,
                        sim_time_sec=0.0,
                        first_success_iter=None,
                        dev_all_pass=False,
                        test_all_pass=False,
                        best_score=float("inf"),
                        best_params=dict(DEFAULT_PARAMS),
                        best_dev={},
                        test_eval={},
                        history=[],
                        error=f"{type(exc).__name__}: {exc}",
                    )
                json_path.write_text(
                    json.dumps(
                        {
                            **result.to_json(),
                            "dev_seeds": dev_seeds,
                            "test_seeds": test_seeds,
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                row = run_to_row(result, json_path)
                key = (str(row["method"]), str(row["scenario"]), str(row["master_seed"]))
                rows_by_key[key] = row
                rows = list(rows_by_key.values())
                with run_csv.open("w", newline="", encoding="utf-8") as fh:
                    writer = csv.DictWriter(fh, fieldnames=RUN_CSV_COLUMNS)
                    writer.writeheader()
                    writer.writerows(rows)
                write_summary(rows, summary_csv)
                print(
                    f"[4.3] {run_name}: dev_pass={result.dev_all_pass} "
                    f"test_pass={result.test_all_pass} calls={result.mc_calls_dev}",
                    flush=True,
                )

    print(f"Wrote: {run_csv}")
    print(f"Wrote: {summary_csv}")


if __name__ == "__main__":
    main()
