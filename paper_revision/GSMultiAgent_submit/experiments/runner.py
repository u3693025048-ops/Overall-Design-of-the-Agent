#!/usr/bin/env python3
"""Batch runner for GSMultiAgent external-MC experiments."""

from __future__ import annotations

import csv
import asyncio
import logging
import os
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from tqdm import tqdm

from experiments.baselines import (
    DEFAULT_EQUAL_BUDGET,
    DEFAULT_PARAMS,
    cma_es_optimize,
    direct_ppo_optimize,
    random_search,
)
from multi_agent.metric_contract import (
    BW_MAX,
    BW_MIN,
    HIT_RATE_MIN,
    PEAK_NY_MAX,
    PM_MAX,
    PM_MIN,
    SEP_MAX,
)
from multi_agent.tools.simulation_tool import check_constraints, run_mc_with_external_data

logger = logging.getLogger(__name__)

MASTER_SEEDS = list(range(42, 45))  # 3 dev/test seed pairs (42, 43, 44)
SCENARIOS = ["T1", "T2", "T3", "T4"]
CSV_COLUMNS = [
    "method",
    "scenario",
    "seed",
    "error",
    "error_msg",
    "first_success",
    "design_path",
    "mc_calls_dev",
    "mc_calls_test",
    "wall_clock_sec",
    "inference_sec",
    "orchestration_sec",
    "sim_sec",
    "iterations_used",
    "hit_rate_dev",
    "SEP_dev",
    "peak_nz_dev",
    "PM_dev",
    "BW_dev",
    "feasible_dev",
    "hit_rate_test",
    "SEP_test",
    "peak_nz_test",
    "PM_test",
    "BW_test",
    "feasible_test",
]


def _mc_paths(scenario: str, seed: int) -> Tuple[str, str]:
    """Legacy single-file paths (first dev/test seed) for backward compatibility."""
    from experiments.optimizer_baselines_43 import (
        dev_seeds_for_master,
        mc_data_path,
        test_seeds_for_master,
    )

    scenario_dir = Path("data/mc")
    dev_seed = dev_seeds_for_master(int(seed))[0]
    test_seed = test_seeds_for_master(int(seed))[0]
    return (
        str(mc_data_path(scenario_dir, scenario, dev_seed, "dev")),
        str(mc_data_path(scenario_dir, scenario, test_seed, "test")),
    )


def _protocol_seeds(master_seed: int) -> Tuple[List[int], List[int]]:
    """Paper §4.1: master*100+{1,2,3} / {4,5,6}."""
    from experiments.optimizer_baselines_43 import resolve_split_seeds

    return resolve_split_seeds(int(master_seed))


def _metric(result: Mapping[str, Any], key: str, default: float = 0.0) -> float:
    """Read numeric metric value from a result dictionary."""
    try:
        return float(result.get(key, default))
    except (TypeError, ValueError):
        return default


def _result_from_metrics(metrics: Mapping[str, Any]) -> Dict[str, Any]:
    """Convert GSM internal metric names into external-MC result shape."""
    hit_rate = _metric(metrics, "hit_rate")
    sep = _metric(metrics, "SEP", _metric(metrics, "miss_distance", 1e9))
    peak_nz = _metric(metrics, "peak_nz", _metric(metrics, "peak_ny", _metric(metrics, "peak_n", 1e9)))
    pm = _metric(metrics, "PM", _metric(metrics, "pitch_PM"))
    bw = _metric(metrics, "BW", _metric(metrics, "pitch_BW"))
    feasible, violations = check_constraints(hit_rate, sep, peak_nz, pm, bw)
    return {
        "hit_rate": hit_rate,
        "SEP": sep,
        "peak_nz": peak_nz,
        "PM": pm,
        "BW": bw,
        "miss_distances": [],
        "nz_peaks": [],
        "feasible": feasible,
        "constraint_violations": violations,
        "sim_time_sec": _metric(metrics, "_sim_time_sec"),
        "success": True,
        "error": None,
        "error_msg": None,
    }


def _empty_result(error_msg: str = "") -> Dict[str, Any]:
    """Return an empty metric result with violations populated."""
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
        "error": error_msg or None,
        "error_msg": error_msg or None,
    }


def _row_from_baseline(
    method: str,
    scenario: str,
    seed: int,
    result: Mapping[str, Any],
    *,
    mc_calls_dev: Optional[int] = None,
    mc_calls_test: int = 1,
) -> Dict[str, Any]:
    """Flatten a baseline result to one CSV row."""
    dev = result.get("dev_result") or {}
    test = result.get("test_result") or {}
    total_calls = int(result.get("mc_calls_used", 0) or 0)
    if mc_calls_dev is None:
        mc_calls_dev = max(0, total_calls - mc_calls_test)
    wall = _metric(result, "wall_clock_sec")
    inference = _metric(result, "inference_sec")
    orchestration = _metric(result, "orchestration_sec")
    sim_sec = max(0.0, wall - inference - orchestration)
    error_msg = str(dev.get("error") or test.get("error") or "")
    return {
        "method": method,
        "scenario": scenario,
        "seed": int(seed),
        "error": bool(error_msg and not test.get("success", False)),
        "error_msg": error_msg,
        # Table 5 / SAH: first_success = final development feasibility.
        "first_success": bool(dev.get("success", False) and dev.get("feasible", False)),
        "design_path": result.get("design_path", "N/A"),
        "mc_calls_dev": int(mc_calls_dev),
        "mc_calls_test": int(mc_calls_test),
        "wall_clock_sec": wall,
        "inference_sec": inference,
        "orchestration_sec": orchestration,
        "sim_sec": sim_sec,
        "iterations_used": int(result.get("iterations_used", 0) or 0),
        "hit_rate_dev": _metric(dev, "hit_rate"),
        "SEP_dev": _metric(dev, "SEP", float("inf")),
        "peak_nz_dev": _metric(dev, "peak_nz", float("inf")),
        "PM_dev": _metric(dev, "PM"),
        "BW_dev": _metric(dev, "BW"),
        "feasible_dev": bool(dev.get("feasible", False)),
        "hit_rate_test": _metric(test, "hit_rate"),
        "SEP_test": _metric(test, "SEP", float("inf")),
        "peak_nz_test": _metric(test, "peak_nz", float("inf")),
        "PM_test": _metric(test, "PM"),
        "BW_test": _metric(test, "BW"),
        "feasible_test": bool(test.get("feasible", False)),
    }


def _apply_override_value(target: Any, key: str, value: Any) -> None:
    """Apply one nested override to a dataclass-like config object."""
    if not hasattr(target, key):
        return
    current = getattr(target, key)
    if isinstance(value, Mapping) and "enabled" in value:
        enabled = bool(value["enabled"])
        if hasattr(current, "enabled"):
            setattr(current, "enabled", enabled)
        else:
            setattr(target, key, enabled)
        for sub_key, sub_value in value.items():
            if sub_key == "enabled" or not hasattr(current, sub_key):
                continue
            _apply_override_value(current, sub_key, sub_value)
        return
    if isinstance(value, Mapping) and current is not None and not isinstance(
        current, (bool, int, float, str, list, tuple, dict)
    ):
        for sub_key, sub_value in value.items():
            _apply_override_value(current, sub_key, sub_value)
        return
    setattr(target, key, value)


def _apply_runtime_config_overrides(config: Mapping[str, Any]) -> None:
    """Apply ablation/runtime overrides to the in-process config object."""
    try:
        from multi_agent.config_loader import get_config

        cfg = get_config()
        for section_name in ("ablation", "workflow", "matlab_rl_optimizer", "simulation"):
            section = config.get(section_name) or {}
            target = getattr(cfg, section_name, None)
            if target is None:
                continue
            for key, value in section.items():
                _apply_override_value(target, key, value)
    except Exception as exc:
        logger.warning("Could not apply runtime config overrides: %s", exc)


def _default_prompt(scenario: str) -> str:
    """Build a compact GSMultiAgent prompt for a paper scenario."""
    """
    return (
        f"针对论文 Table 3 的 {scenario} 目标机动场景，使用外部 Monte Carlo "
        "dev 数据评估并优化 10 维制导/驾驶仪参数，要求命中率>=92%, "
        "SEP<=7m, PeakNy<=20g, PM在[45,70]度, BW在[20,85]rad/s。"
    )
    """
    return (
        f"Paper scenario {scenario}: use the external Monte Carlo dev set to optimize "
        "the 10-D guidance/autopilot parameter vector. A candidate passes only when "
        "all three dev seeds satisfy every hard constraint: "
        f"hit_rate>={HIT_RATE_MIN:.0f}%, SEP<={SEP_MAX:.0f}m, "
        f"PeakNy<={PEAK_NY_MAX:.0f}g, PM in [{PM_MIN:.0f},{PM_MAX:.0f}] deg, "
        f"BW in [{BW_MIN:.0f},{BW_MAX:.0f}] rad/s."
    )


def _run_gsmultiagent(
    scenario: str,
    dev_data_path: str,
    test_data_path: str,
    seed: int,
    config: Mapping[str, Any],
) -> Dict[str, Any]:
    """Run GSMultiAgent for one master seed using paper 3-dev/3-test AND protocol."""
    from experiments.optimizer_baselines_43 import resolve_split_seeds
    from multi_agent.tools.simulation_tool import run_mc_and_across_seeds

    started = time.perf_counter()
    dev_seeds, test_seeds = resolve_split_seeds(int(seed))
    mc_root = str(config.get("mc_root", "data/mc"))

    if config.get("dry_run") or config.get("gsm_dry_run"):
        dev_result = run_mc_and_across_seeds(
            "knowledge_base/matlab/guidance/monte_carlo_single.m",
            mc_root=mc_root,
            scenario=scenario,
            seeds=dev_seeds,
            split="dev",
            n_runs=int(config.get("n_runs", 1)),
            dp_params=DEFAULT_PARAMS,
        )
        test_result = run_mc_and_across_seeds(
            "knowledge_base/matlab/guidance/monte_carlo_single.m",
            mc_root=mc_root,
            scenario=scenario,
            seeds=test_seeds,
            split="test",
            n_runs=int(config.get("n_runs", 1)),
            dp_params=DEFAULT_PARAMS,
        )
        wall = time.perf_counter() - started
        sim_sec = _metric(dev_result, "sim_time_sec") + _metric(test_result, "sim_time_sec")
        return {
            "best_params": DEFAULT_PARAMS.copy(),
            "dev_result": dev_result,
            "test_result": test_result,
            "feasible": bool(test_result.get("feasible", False)),
            "mc_calls_used": int(dev_result.get("mc_calls", len(dev_seeds)))
            + int(test_result.get("mc_calls", len(test_seeds))),
            "wall_clock_sec": wall,
            "inference_sec": 0.0,
            "orchestration_sec": max(0.0, wall - sim_sec),
            "iterations_used": 0,
            "design_path": "DRY_RUN",
            "feasibility_rate": 0.0,
            "convergence_log": [],
        }

    # Reload YAML defaults before applying per-run ablation overrides so flags
    # from a previous config (e.g. C1→C2) cannot leak through the singleton.
    from multi_agent.config_loader import get_config, reload_config

    reload_config()
    _apply_runtime_config_overrides(config)

    cfg = get_config()
    cfg.simulation.mc_root = mc_root
    cfg.simulation.mc_scenario = scenario
    cfg.simulation.mc_seeds = list(dev_seeds)
    cfg.simulation.mc_data_path = ""
    os.environ["GSM_MC_DATA_PATH"] = ""
    if config.get("skip_llm_for_smoke"):
        os.environ["GSM_BASELINE_SKIP_LLM"] = "1"

    try:
        from multi_agent.simulation.pipeline_warmup import (
            start_warmup_async,
            warmup_scenario,
        )

        warmup_scenario(scenario, master_seed=int(seed), mc_root=mc_root)
        next_sc = os.environ.get("GSM_WARMUP_NEXT_SCENARIO", "").strip()
        next_seed = os.environ.get("GSM_WARMUP_NEXT_SEED", "").strip()
        if next_sc:
            start_warmup_async(
                next_sc,
                master_seed=int(next_seed or seed),
                mc_root=mc_root,
            )
    except Exception as exc:
        print(f"  [pipeline_warmup] skipped: {exc}")

    import cli_agent
    from multi_agent.run_file_manager import hermes_rl_best_path

    snapshot_path = hermes_rl_best_path()
    # Ensure we only accept a snapshot produced by this run (isolated PE workspace).
    try:
        if snapshot_path.exists():
            snapshot_path.unlink()
    except OSError:
        pass

    prompt = str(config.get("prompt") or config.get("prompt_template", "")).format(
        scenario=scenario,
        seed=seed,
        dev_data_path=dev_data_path,
        test_data_path=test_data_path,
    ) if (config.get("prompt") or config.get("prompt_template")) else _default_prompt(scenario)
    args = SimpleNamespace(
        prompt=prompt,
        file=None,
        skip_preflight=True,
        preflight_only=False,
        preflight_strict=False,
        verbose=bool(config.get("verbose", False)),
        quiet=bool(config.get("quiet", True)),
        resume_checkpoint=config.get("resume_checkpoint"),
        mc_data_path="",
        async_run=False,
        task_status=None,
        serve_tasks=False,
        serve_port=8765,
    )
    return_code = asyncio.run(cli_agent.main(args))
    wall = time.perf_counter() - started

    best_params = DEFAULT_PARAMS.copy()
    dev_result = _empty_result("GSM run completed without fresh metrics snapshot")
    design_path = "GSMultiAgent"
    iterations_used = 0
    if snapshot_path.exists():
        try:
            import json

            # Prefer fresh mtime; still accept if file exists post-run (writeback path).
            snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
            mtime_ok = snapshot_path.stat().st_mtime >= (started - 1.0)
            has_metrics = bool(snapshot.get("best_metrics"))
            if mtime_ok or has_metrics:
                best_params = snapshot.get("best_params") or best_params
                if has_metrics:
                    dev_result = _result_from_metrics(snapshot.get("best_metrics") or {})
                iterations_used = int(
                    snapshot.get("outer_iterations")
                    or snapshot.get("total_episodes")
                    or 0
                )
                design_path = str(
                    snapshot.get("task_mode")
                    or snapshot.get("best_selection")
                    or snapshot.get("status")
                    or design_path
                )
            else:
                dev_result = _empty_result("GSM snapshot exists but looks stale")
        except Exception as exc:
            dev_result = _empty_result(f"could not parse GSM snapshot: {exc}")

    test_n_runs = 100
    try:
        from multi_agent.config_loader import get_config

        _sim = get_config().simulation
        test_n_runs = int(getattr(_sim, "mc_final_n_runs", 100) or 100)
    except Exception:
        test_n_runs = 100

    test_result = run_mc_and_across_seeds(
        "knowledge_base/matlab/guidance/monte_carlo_single.m",
        mc_root=mc_root,
        scenario=scenario,
        seeds=test_seeds,
        split="test",
        n_runs=test_n_runs,
        dp_params=best_params,
        early_abort=False,
    )
    wall = time.perf_counter() - started
    sim_sec = _metric(test_result, "sim_time_sec") + _metric(dev_result, "sim_time_sec")
    error_msg = "" if return_code in (None, 0) else f"cli_agent returned {return_code}"
    if test_result.get("error"):
        error_msg = test_result.get("error") or error_msg
    return {
        "best_params": best_params,
        "dev_result": dev_result,
        "test_result": test_result,
        "feasible": bool(test_result.get("feasible", False)),
        "mc_calls_used": max(1, iterations_used) * len(dev_seeds)
        + int(test_result.get("mc_calls", len(test_seeds))),
        "wall_clock_sec": wall,
        "inference_sec": 0.0,
        "orchestration_sec": max(0.0, wall - sim_sec),
        "iterations_used": iterations_used,
        "design_path": design_path,
        "feasibility_rate": 1.0 if bool(dev_result.get("feasible", False)) else 0.0,
        "convergence_log": [],
        "error_msg": error_msg,
    }


def run_single_experiment(
    method: str,
    scenario: str,
    seed: int,
    config: dict,
) -> dict:
    """Run one method x scenario x seed experiment.

    Args:
        method: Method name, e.g. ``"GSMultiAgent"``, ``"random_search"``,
            ``"cma_es"``, or ``"direct_ppo"``.
        scenario: Scenario key.
        seed: Master seed.
        config: Ablation/runtime configuration overrides.

    Returns:
        CSV row dictionary. If execution crashes, returns a row with
        ``error=True`` and a traceback excerpt.
    """
    config = dict(config or {})
    dev_data_path, test_data_path = _mc_paths(scenario, seed)
    try:
        if method in {"random_search", "RandomSearch", "RS"}:
            budget = int(config.get("budget", config.get("random_search_budget", DEFAULT_EQUAL_BUDGET)))
            result = random_search(scenario, dev_data_path, test_data_path, budget=budget, seed=seed)
            return _row_from_baseline(method, scenario, seed, result, mc_calls_dev=budget)
        if method in {"cma_es", "CMA-ES", "CMAES"}:
            budget = int(config.get("budget", config.get("cma_budget", DEFAULT_EQUAL_BUDGET)))
            result = cma_es_optimize(
                scenario,
                dev_data_path,
                test_data_path,
                config.get("initial_params", DEFAULT_PARAMS),
                budget=budget,
                seed=seed,
            )
            dev_calls = max(1, int(result.get("mc_calls_used", 1)) - 1)
            return _row_from_baseline(method, scenario, seed, result, mc_calls_dev=dev_calls)
        if method in {"direct_ppo", "DirectPPO", "PPO"}:
            max_iterations = int(config.get("max_iterations", config.get("ppo_max_iterations", DEFAULT_EQUAL_BUDGET)))
            result = direct_ppo_optimize(
                scenario,
                dev_data_path,
                test_data_path,
                max_iterations=max_iterations,
                seed=seed,
            )
            dev_calls = max(0, int(result.get("mc_calls_used", 1)) - 1)
            return _row_from_baseline(method, scenario, seed, result, mc_calls_dev=dev_calls)
        if method in {"GSMultiAgent", "gsmultiagent", "gsm"}:
            result = _run_gsmultiagent(scenario, dev_data_path, test_data_path, seed, config)
            _, test_seeds = _protocol_seeds(seed)
            dev_calls = max(0, int(result.get("mc_calls_used", 1)) - len(test_seeds))
            return _row_from_baseline(
                method,
                scenario,
                seed,
                result,
                mc_calls_dev=dev_calls,
                mc_calls_test=len(test_seeds),
            )
        raise ValueError(f"Unknown method: {method}")
    except Exception as exc:
        tb = traceback.format_exc()
        logger.error("Experiment crashed: %s %s %s\n%s", method, scenario, seed, tb)
        row = _row_from_baseline(
            method,
            scenario,
            seed,
            {
                "dev_result": _empty_result(str(exc)),
                "test_result": _empty_result(str(exc)),
                "mc_calls_used": 0,
                "wall_clock_sec": 0.0,
                "inference_sec": 0.0,
                "orchestration_sec": 0.0,
                "iterations_used": 0,
                "design_path": "N/A",
            },
            mc_calls_dev=0,
            mc_calls_test=0,
        )
        row["error"] = True
        row["error_msg"] = tb[-2000:]
        row["traceback"] = tb
        return row


def _already_done(output_csv: str, method: str, scenario: str, seed: int) -> bool:
    """Return True when a successful row already exists in ``output_csv``."""
    if not Path(output_csv).exists():
        return False
    with open(output_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if (
                row.get("method") == method
                and row.get("scenario") == scenario
                and int(row.get("seed", -1)) == int(seed)
                and row.get("error") != "True"
            ):
                # Incomplete / placeholder rows must not block reruns.
                msg = str(row.get("error_msg") or "")
                if "without fresh metrics" in msg:
                    continue
                try:
                    iters = int(float(row.get("iterations_used") or 0))
                except (TypeError, ValueError):
                    iters = 0
                first_ok = str(row.get("first_success") or "").lower() in {"true", "1", "yes"}
                dev_ok = str(row.get("feasible_dev") or "").lower() in {"true", "1", "yes"}
                if iters <= 0 and not first_ok and not dev_ok:
                    continue
                return True
    return False


def _append_row(output_csv: str, row: Mapping[str, Any]) -> None:
    """Append one CSV row, writing the header if needed.

    Treat empty (truncated) files as header-less so force-rerun after ``: > file``
    or zero-byte wipe still produces a valid CSV.
    """
    path = Path(output_csv)
    path.parent.mkdir(parents=True, exist_ok=True)
    need_header = (not path.exists()) or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        if need_header:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in CSV_COLUMNS})


def run_all_experiments(
    methods: list[str],
    scenarios: list[str],
    seeds: list[int],
    config_overrides: dict = None,
    output_csv: str = "results/main_results.csv",
    max_workers: int = 4,
) -> None:
    """Run all method x scenario x seed combinations with resumable CSV output.

    Args:
        methods: Methods to run.
        scenarios: Scenario keys to run.
        seeds: Master seeds to run.
        config_overrides: Runtime/ablation overrides passed to each task.
        output_csv: Destination CSV path.
        max_workers: Process pool parallelism.
    """
    config = dict(config_overrides or {})
    tasks = [
        (method, scenario, seed)
        for method in methods
        for scenario in scenarios
        for seed in seeds
        if not _already_done(output_csv, method, scenario, seed)
    ]
    if not tasks:
        logger.info("All requested experiments already complete: %s", output_csv)
        return

    if max_workers <= 1:
        for method, scenario, seed in tqdm(tasks, desc="Experiments"):
            row = run_single_experiment(method, scenario, seed, config)
            _append_row(output_csv, row)
        return

    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(run_single_experiment, method, scenario, seed, config): (
                method,
                scenario,
                seed,
            )
            for method, scenario, seed in tasks
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Experiments"):
            row = future.result()
            _append_row(output_csv, row)


def run_ablation_memory(seeds: Sequence[int], output_csv: str) -> None:
    """Run memory ablations E0-E3 sequentially per seed over T1 through T4.

    Protocol: within each (seed, config) group, T1→T4 accumulate PE/memory.
    At the start of the next group (new seed or new config), PE + Hermes memory
    are cleared and prior scripts/reports are archived under DATA/run_archives.
    """
    from datetime import datetime

    from multi_agent.run_file_manager import (
        archive_scripts_and_reports,
        begin_experiment_group,
        write_group_manifest,
    )

    configs = {
        "E0_full": {
            "ablation": {
                "parameter_experience_reuse": {"enabled": True},
                "memory_search": {"enabled": True},
                "reflection_agent": {"enabled": True},
            }
        },
        "E1_no_memory": {
            "ablation": {
                "parameter_experience_reuse": {"enabled": False},
                "memory_search": {"enabled": False},
                "reflection_agent": {"enabled": True},
            }
        },
        "E2_no_reflection": {
            "ablation": {
                "parameter_experience_reuse": {"enabled": True},
                "memory_search": {"enabled": True},
                "reflection_agent": {"enabled": False},
            }
        },
        "E3_no_memory_no_reflection": {
            "ablation": {
                "parameter_experience_reuse": {"enabled": False},
                "memory_search": {"enabled": False},
                "reflection_agent": {"enabled": False},
            }
        },
    }
    for seed in seeds:
        for method, cfg in configs.items():
            group_dir = None
            for scenario in SCENARIOS:
                if _already_done(output_csv, method, scenario, int(seed)):
                    continue
                if group_dir is None:
                    _, group_dir = begin_experiment_group(
                        f"memory_{method}_seed{int(seed)}"
                    )
                    logger.info(
                        "Cleared PE/memory for %s seed=%s → %s",
                        method,
                        seed,
                        group_dir,
                    )
                started = datetime.now()
                row = run_single_experiment("GSMultiAgent", scenario, int(seed), cfg)
                row["method"] = method
                _append_row(output_csv, row)
                archive_scripts_and_reports(
                    group_dir,
                    scenario,
                    only_newer_than=datetime.fromtimestamp(started.timestamp() - 1.0),
                )
            if group_dir is not None:
                write_group_manifest(group_dir, {"status": "completed"})


def run_ablation_optimization(seeds: Sequence[int], output_csv: str) -> None:
    """Run optimization ablations for GSMultiAgent.

    Clears PE/memory at the start of each (config, seed) group so O0/O1 are
    independent cold starts, matching the paper protocol.
    """
    from datetime import datetime

    from multi_agent.run_file_manager import (
        archive_scripts_and_reports,
        begin_experiment_group,
        write_group_manifest,
    )

    configs = {
        "O0_full": {},
        "O1_no_optimization_agent": {
            "ablation": {"rl_optimization": {"enabled": False}},
            "matlab_rl_optimizer": {
                "constraint_local_search": {"enabled": False},
            },
        },
    }
    for method, cfg in configs.items():
        for seed in seeds:
            group_dir = None
            for scenario in SCENARIOS:
                if _already_done(output_csv, method, scenario, int(seed)):
                    continue
                if group_dir is None:
                    _, group_dir = begin_experiment_group(
                        f"optimization_{method}_seed{int(seed)}"
                    )
                started = datetime.now()
                row = run_single_experiment("GSMultiAgent", scenario, int(seed), cfg)
                row["method"] = method
                _append_row(output_csv, row)
                archive_scripts_and_reports(
                    group_dir,
                    scenario,
                    only_newer_than=datetime.fromtimestamp(started.timestamp() - 1.0),
                )
            if group_dir is not None:
                write_group_manifest(group_dir, {"status": "completed"})


def run_ablation_correction(
    seeds: Sequence[int],
    output_csv: str,
    *,
    n_independent_runs: int = 5,
) -> None:
    """Run correction-pipeline ablation on T4 MODIFY_LAW.

    Paper §4.3.3: five independent runs per configuration, clearing experience
    before each run. ``seeds`` may be empty; when empty, synthetic run ids
    1..n_independent_runs are used as CSV seed column.
    """
    from datetime import datetime

    from multi_agent.run_file_manager import (
        archive_scripts_and_reports,
        begin_experiment_group,
        write_group_manifest,
    )

    configs = {
        "C0_full_pipeline": {},
        "C1_no_layer1_syntax": {
            "ablation": {
                "syntax_check": {"enabled": False},
                "guidance_compat_verification": {"enabled": True},
            }
        },
        "C2_no_layers23_compat_semantic": {
            "ablation": {
                "syntax_check": {"enabled": True},
                "guidance_compat_verification": {"enabled": False},
                "reflection_agent": {"enabled": False},
            }
        },
    }
    run_ids = [int(s) for s in seeds] if seeds else list(range(1, int(n_independent_runs) + 1))
    # Prefer exactly 5 independent clears when caller passes master seeds; expand
    # short lists up to n_independent_runs by repeating with offset tags.
    while len(run_ids) < int(n_independent_runs):
        run_ids.append(9000 + len(run_ids))
    run_ids = run_ids[: int(n_independent_runs)]

    for method, cfg in configs.items():
        for run_id in run_ids:
            if _already_done(output_csv, method, "T4", int(run_id)):
                continue
            _, group_dir = begin_experiment_group(f"correction_{method}_run{int(run_id)}")
            cfg_run = dict(cfg)
            cfg_run["prompt_template"] = (
                "T4 MODIFY_LAW correction-pipeline ablation, seed={seed}. "
                "Use external MC data and repair the guidance law if needed. "
                "Hard constraints: hit_rate>=92%, SEP<=7m, PeakNy<=20g, "
                "PM in [45,70] deg, BW in [20,85] rad/s."
            )
            # Map synthetic run ids back to a real master seed for MC data.
            master = int(run_id) if int(run_id) in MASTER_SEEDS else MASTER_SEEDS[0]
            started = datetime.now()
            row = run_single_experiment("GSMultiAgent", "T4", master, cfg_run)
            row["method"] = method
            row["seed"] = int(run_id)
            _append_row(output_csv, row)
            archive_scripts_and_reports(
                group_dir,
                "T4",
                only_newer_than=datetime.fromtimestamp(started.timestamp() - 1.0),
            )
            write_group_manifest(group_dir, {"status": "completed", "run_id": int(run_id)})


__all__ = [
    "MASTER_SEEDS",
    "SCENARIOS",
    "CSV_COLUMNS",
    "run_single_experiment",
    "run_all_experiments",
    "run_ablation_memory",
    "run_ablation_optimization",
    "run_ablation_correction",
]
