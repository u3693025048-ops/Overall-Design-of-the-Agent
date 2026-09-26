#!/usr/bin/env python3
"""Paper experiment orchestration entry point.

The heavy GSMultiAgent runs still go through ``experiments.runner`` and
``cli_agent.py``.  This module provides a small, reproducible front door that
matches the README protocol: it can generate command manifests, launch the
Section 4.3 optimizer baselines, generate MC data, and dispatch the GSM
ablation suites with named configs.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.equal_mc_budget import DEFAULT_MAX_CANDIDATES, GSM_MAX_EPISODES, max_mc_calls
from multi_agent.metric_contract import DEFAULT_METRIC_REQUIREMENTS

DEFAULT_SCENARIOS = ("T1", "T2", "T3", "T4")
DEFAULT_MASTER_SEEDS = (42,)
DEV_OFFSETS = (1, 2, 3)
TEST_OFFSETS = (4, 5, 6)
DEFAULT_OUTPUT_ROOT = Path("results") / "experiment_plan"

MEMORY_CONFIGS: Dict[str, Dict[str, Any]] = {
    "E0_full": {
        "ablation": {
            "parameter_experience_reuse": {"enabled": True},
            "memory_search": {"enabled": True},
            "reflection_agent": {"enabled": True},
        }
    },
    # Model-substitution study (paper §4.3.4 / R15–R16): same ablation flags as E0_full,
    # distinct config names → isolated PE workspaces + CSV method labels.
    "E0_full_qwen": {
        "ablation": {
            "parameter_experience_reuse": {"enabled": True},
            "memory_search": {"enabled": True},
            "reflection_agent": {"enabled": True},
        }
    },
    "E0_full_kimi": {
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

ORDER_CONFIGS: Dict[str, Dict[str, Any]] = {"E0_full": MEMORY_CONFIGS["E0_full"]}
TASK_ORDERS: Dict[str, Sequence[str]] = {
    "forward": ("T1", "T2", "T3", "T4"),
    "reverse": ("T4", "T3", "T2", "T1"),
    "shuffled": ("T3", "T1", "T4", "T2"),
}

OPTIMIZATION_CONFIGS: Dict[str, Dict[str, Any]] = {
    "O0_full": {},
    "O1_no_optimization_agent": {
        "ablation": {
            "rl_optimization": {"enabled": False},
        },
        "matlab_rl_optimizer": {
            "constraint_local_search": {"enabled": False},
        },
    },
}

CORRECTION_CONFIGS: Dict[str, Dict[str, Any]] = {
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

SUITE_CONFIGS: Dict[str, Mapping[str, Dict[str, Any]]] = {
    "memory": MEMORY_CONFIGS,
    "order": ORDER_CONFIGS,
    "optimization": OPTIMIZATION_CONFIGS,
    "correction": CORRECTION_CONFIGS,
}


@dataclass
class PlannedCommand:
    """One executable command emitted by ``make-plan``."""

    section: str
    name: str
    command: List[str]
    output: str
    notes: str = ""


@dataclass
class GsmJob:
    """One GSMMultiAgent ablation job."""

    suite: str
    config_name: str
    scenario: str
    master_seed: int
    output_csv: str
    order_name: str = ""
    execute: bool = False


def _split_values(values: Optional[Iterable[str]], default: Sequence[str]) -> List[str]:
    if not values:
        return list(default)
    out: List[str] = []
    for value in values:
        for part in str(value).replace(",", " ").split():
            if part.strip():
                out.append(part.strip())
    return out


def _split_ints(values: Optional[Iterable[str]], default: Sequence[int]) -> List[int]:
    return [int(v) for v in _split_values(values, [str(v) for v in default])]


def seed_manifest(master_seeds: Sequence[int]) -> Dict[str, Dict[str, List[int]]]:
    """Return the README dev/test split seeds for each master seed."""
    payload: Dict[str, Dict[str, List[int]]] = {}
    for master_seed in master_seeds:
        master = int(master_seed)
        payload[str(master)] = {
            "dev": [master * 100 + offset for offset in DEV_OFFSETS],
            "test": [master * 100 + offset for offset in TEST_OFFSETS],
        }
    return payload


def protocol_payload(
    *,
    scenarios: Sequence[str],
    master_seeds: Sequence[int],
    n_runs: int,
    iterations: int,
) -> Dict[str, Any]:
    """Common protocol block recorded by plan and suite commands."""
    try:
        from multi_agent.simulation.mc_seed_bootstrap import bootstrap_fill_enabled

        bootstrap = bootstrap_fill_enabled(default=True)
    except Exception:
        bootstrap = True
    return {
        "scenarios": list(scenarios),
        "master_seeds": [int(seed) for seed in master_seeds],
        "seed_protocol": "master_seed * 100 + offsets",
        "dev_offsets": list(DEV_OFFSETS),
        "test_offsets": list(TEST_OFFSETS),
        "seed_manifest": seed_manifest(master_seeds),
        "and_logic": (
            "bootstrap_fill: MATLAB on first listed seed only; "
            "remaining seeds filled by resampling first-seed trajectories"
            if bootstrap
            else "all three dev seeds must pass; all three test seeds must pass"
        ),
        "mc_and_bootstrap_fill": bool(bootstrap),
        "n_runs_per_mc_call": int(n_runs),
        "outer_iterations": int(iterations),
        "iterations": int(iterations),
        "max_candidates": int(iterations) * int(GSM_MAX_EPISODES),
        "max_mc_calls_dev": max_mc_calls(
            n_dev_seeds=len(DEV_OFFSETS),
            max_candidates=int(iterations) * int(GSM_MAX_EPISODES),
        ),
        "equal_mc_budget": (
            "All methods share the same maximum number of development Monte Carlo "
            f"calls (outer_iterations × max_episodes × n_dev_seeds = "
            f"{int(iterations)} × {int(GSM_MAX_EPISODES)} × {len(DEV_OFFSETS)}). "
            "CMA-ES generations are not a separate budget unit."
        ),
        "metric_requirements": dict(DEFAULT_METRIC_REQUIREMENTS),
        "search_dim": 10,
        "search_space_source": "multi_agent.rl.matlab_rl_optimizer.ALL_TUNABLE_PARAM_SPECS",
    }


def _script_path(relative: str) -> str:
    return str((REPO_ROOT / relative).resolve())


def _python_command(relative_script: str, *args: str) -> List[str]:
    return [sys.executable, _script_path(relative_script), *args]


def _ps_quote(arg: str) -> str:
    text = str(arg)
    if not text:
        return "''"
    if any(ch.isspace() for ch in text) or any(ch in text for ch in "'`$;&()[]{}"):
        return "'" + text.replace("'", "''") + "'"
    return text


def command_to_powershell(command: Sequence[str]) -> str:
    """Format a command list for a PowerShell script."""
    if not command:
        return ""
    return "& " + " ".join(_ps_quote(part) for part in command)


def command_with_execute(command: Sequence[str]) -> List[str]:
    """Return command with ``--execute`` added for GSM suite runs."""
    out = list(command)
    if "run-gsm-suite" in out and "--execute" not in out:
        out.append("--execute")
    return out


def build_plan(
    *,
    sections: Sequence[str],
    scenarios: Sequence[str],
    master_seeds: Sequence[int],
    n_runs: int,
    iterations: int,
    output_root: Path,
) -> List[PlannedCommand]:
    """Build README-aligned commands without executing them."""
    selected = set(sections)
    commands: List[PlannedCommand] = []
    scenario_args = [str(s) for s in scenarios]
    seed_args = [str(seed) for seed in master_seeds]

    if "mc" in selected:
        commands.append(
            PlannedCommand(
                section="mc",
                name="generate_mc_data",
                command=_python_command(
                    "experiments/mc_data_generator.py",
                    "--output-root",
                    "data/mc",
                    "--n-runs",
                    str(n_runs),
                    "--no-broadside",
                ),
                output="data/mc/seed_map.json",
                notes="Generate 3 dev and 3 test MC files for each master seed.",
            )
        )

    if "optimizer" in selected:
        commands.append(
            PlannedCommand(
                section="optimizer",
                name="optimizer_baselines_43",
                command=_python_command(
                    "experiments/optimizer_baselines_43.py",
                    "--methods",
                    "random_search",
                    "cma_es",
                    "direct_ppo",
                    "--scenarios",
                    *scenario_args,
                    "--master-seeds",
                    *seed_args,
                    "--iterations",
                    str(int(iterations) * int(GSM_MAX_EPISODES)),
                    "--n-runs",
                    str(n_runs),
                    "--output-dir",
                    str(output_root / "optimizer_baselines_43"),
                ),
                output=str(output_root / "optimizer_baselines_43"),
                notes=(
                    "Raw optimizer loops only; no Hermes, memory, RAG, or reflection. "
                    f"Shared candidate ceiling = outer_iterations × {GSM_MAX_EPISODES} "
                    f"= {int(iterations) * int(GSM_MAX_EPISODES)} "
                    f"({int(iterations) * int(GSM_MAX_EPISODES) * len(DEV_OFFSETS)} MC calls)."
                ),
            )
        )

    for suite in ("memory", "optimization", "correction"):
        if suite not in selected:
            continue
        suite_output = output_root / f"ablation_{suite}.csv"
        suite_args = [
            "run-gsm-suite",
            "--suite",
            suite,
            "--scenarios",
            *scenario_args,
            "--master-seeds",
            *seed_args,
            "--iterations",
            str(iterations),
            "--n-runs",
            str(n_runs),
            "--output-csv",
            str(suite_output),
        ]
        if suite == "correction":
            suite_args = [
                "run-gsm-suite",
                "--suite",
                suite,
                "--scenarios",
                "T4",
                "--master-seeds",
                *seed_args,
                "--iterations",
                str(iterations),
                "--n-runs",
                str(n_runs),
                "--output-csv",
                str(suite_output),
            ]
        commands.append(
            PlannedCommand(
                section=suite,
                name=f"{suite}_ablation",
                command=_python_command("experiments/experiment_plan_runner.py", *suite_args),
                output=str(suite_output),
                notes="Add --execute to run; without it this command writes the GSM suite protocol only.",
            )
        )
    if "order" in selected:
        for order_name in TASK_ORDERS:
            suite_output = output_root / f"ablation_order_{order_name}.csv"
            commands.append(
                PlannedCommand(
                    section="order",
                    name=f"order_ablation_{order_name}",
                    command=_python_command(
                        "experiments/experiment_plan_runner.py",
                        "run-gsm-suite",
                        "--suite",
                        "order",
                        "--order",
                        order_name,
                        "--scenarios",
                        *scenario_args,
                        "--master-seeds",
                        *seed_args,
                        "--iterations",
                        str(iterations),
                        "--n-runs",
                        str(n_runs),
                        "--output-csv",
                        str(suite_output),
                    ),
                    output=str(suite_output),
                    notes="Add --execute to run this task-order robustness suite.",
                )
            )
    return commands


def write_plan(
    commands: Sequence[PlannedCommand],
    output_dir: Path,
    protocol: Mapping[str, Any],
) -> None:
    """Write manifest JSON, CSV, and PowerShell command file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "protocol": dict(protocol),
        "commands": [asdict(command) for command in commands],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with (output_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["section", "name", "output", "notes", "command"])
        writer.writeheader()
        for item in commands:
            writer.writerow(
                {
                    "section": item.section,
                    "name": item.name,
                    "output": item.output,
                    "notes": item.notes,
                    "command": " ".join(item.command),
                }
            )
    lines = [
        "# Generated by experiments/experiment_plan_runner.py",
        "$ErrorActionPreference = 'Stop'",
        "",
    ]
    for item in commands:
        lines.extend([f"# [{item.section}] {item.name}", command_to_powershell(item.command), ""])
    (output_dir / "commands.ps1").write_text("\n".join(lines), encoding="utf-8")
    exec_lines = [
        "# Generated by experiments/experiment_plan_runner.py",
        "# This version appends --execute to GSM ablation suite commands.",
        "$ErrorActionPreference = 'Stop'",
        "",
    ]
    for item in commands:
        exec_lines.extend([
            f"# [{item.section}] {item.name}",
            command_to_powershell(command_with_execute(item.command)),
            "",
        ])
    (output_dir / "commands_execute.ps1").write_text("\n".join(exec_lines), encoding="utf-8")


def _configs_for_suite(suite: str, requested: Sequence[str]) -> List[str]:
    config_map = SUITE_CONFIGS[suite]
    if requested:
        unknown = sorted(set(requested) - set(config_map))
        if unknown:
            raise ValueError(f"Unknown config(s) for suite {suite}: {unknown}")
        return list(requested)
    return list(config_map.keys())


def build_gsm_jobs(
    *,
    suite: str,
    configs: Sequence[str],
    scenarios: Sequence[str],
    master_seeds: Sequence[int],
    output_csv: str,
    execute: bool,
    order_name: str = "forward",
) -> List[GsmJob]:
    """Expand a GSM ablation suite into sequential jobs."""
    if suite not in SUITE_CONFIGS:
        raise ValueError(f"Unknown suite: {suite}")
    config_names = _configs_for_suite(suite, configs)
    jobs: List[GsmJob] = []
    if suite == "order":
        orders = [order_name] if order_name else list(TASK_ORDERS)
        for order in orders:
            ordered_scenarios = [s for s in TASK_ORDERS[order] if s in set(scenarios)]
            for master_seed in master_seeds:
                for scenario in ordered_scenarios:
                    jobs.append(
                        GsmJob(
                            suite=suite,
                            config_name="E0_full",
                            scenario=scenario,
                            master_seed=int(master_seed),
                            output_csv=output_csv,
                            order_name=order,
                            execute=execute,
                        )
                    )
        return jobs

    run_scenarios = ["T4"] if suite == "correction" else list(scenarios)
    for config_name in config_names:
        for master_seed in master_seeds:
            for scenario in run_scenarios:
                jobs.append(
                    GsmJob(
                        suite=suite,
                        config_name=config_name,
                        scenario=scenario,
                        master_seed=int(master_seed),
                        output_csv=output_csv,
                        execute=execute,
                    )
                )
    return jobs


def _suite_config(suite: str, config_name: str, args: argparse.Namespace) -> Dict[str, Any]:
    config = json.loads(json.dumps(SUITE_CONFIGS[suite][config_name]))
    config.setdefault("workflow", {})["max_iterations"] = int(args.iterations)
    config["n_runs"] = int(args.n_runs)
    if getattr(args, "gsm_dry_run", False):
        config["dry_run"] = True
    if suite == "correction":
        config["prompt_template"] = (
            "T4 MODIFY_LAW correction-pipeline ablation, seed={seed}. "
            "Use external MC data and repair the guidance law if needed. "
            "Requirements: hit_rate>=92%, SEP<=7m, PeakNy<=20g, PM in [45,70] deg, BW in [20,85] rad/s."
        )
        # Correction ablation compares syntax / compat / reflection layers.
        # T4 MODIFY_LAW fast path bypasses Hermes + those tools → C0≡C1≡C2.
        config.setdefault("workflow", {})["modify_law_t4_fast_path"] = False
    return config


def write_gsm_protocol(
    *,
    output_csv: Path,
    suite: str,
    config_names: Sequence[str],
    scenarios: Sequence[str],
    master_seeds: Sequence[int],
    n_runs: int,
    iterations: int,
    jobs: Sequence[GsmJob],
) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    payload = protocol_payload(
        scenarios=scenarios,
        master_seeds=master_seeds,
        n_runs=n_runs,
        iterations=iterations,
    )
    payload.update(
        {
            "suite": suite,
            "configs": {name: SUITE_CONFIGS[suite][name] for name in config_names},
            "jobs": [asdict(job) for job in jobs],
            "csv": str(output_csv),
        }
    )
    protocol_path = output_csv.with_suffix(".protocol.json")
    protocol_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def run_mc_generator(args: argparse.Namespace) -> None:
    from experiments.mc_data_generator import generate_all_mc_data

    generate_all_mc_data(
        output_root=args.output_root,
        n_runs=int(args.n_runs),
        include_broadside=not bool(args.no_broadside),
    )


def run_optimizer_baselines(args: argparse.Namespace) -> None:
    from experiments.optimizer_baselines_43 import main as optimizer_main

    forwarded = [
        "--methods",
        *args.methods,
        "--scenarios",
        *args.scenarios,
        "--master-seeds",
        *[str(seed) for seed in args.master_seeds],
        "--iterations",
        str(int(args.iterations)),
        "--n-runs",
        str(args.n_runs),
        "--output-dir",
        args.output_dir,
    ]
    if args.keep_going:
        forwarded.append("--keep-going")
    optimizer_main(forwarded)


def run_gsm_suite(args: argparse.Namespace) -> None:
    scenarios = list(args.scenarios)
    master_seeds = [int(seed) for seed in args.master_seeds]
    config_names = _configs_for_suite(args.suite, args.configs)
    jobs = build_gsm_jobs(
        suite=args.suite,
        configs=config_names,
        scenarios=scenarios,
        master_seeds=master_seeds,
        output_csv=args.output_csv,
        execute=bool(args.execute),
        order_name=args.order,
    )
    output_csv = Path(args.output_csv)
    write_gsm_protocol(
        output_csv=output_csv,
        suite=args.suite,
        config_names=config_names,
        scenarios=scenarios,
        master_seeds=master_seeds,
        n_runs=int(args.n_runs),
        iterations=int(args.iterations),
        jobs=jobs,
    )
    if not args.execute:
        print(f"Planned {len(jobs)} GSM jobs. Add --execute to run them.")
        print(f"Wrote: {output_csv.with_suffix('.protocol.json')}")
        return

    from datetime import datetime
    import os
    from collections import OrderedDict
    from concurrent.futures import as_completed

    from experiments.runner import _already_done, _append_row, run_single_experiment
    from multi_agent.run_file_manager import (
        archive_scripts_and_reports,
        begin_experiment_group,
        write_group_manifest,
    )

    # Group jobs by (config, order, seed). Within a group T1→T4 must stay serial
    # (PE accumulates). Independent groups may run in parallel with isolated PE.
    groups: "OrderedDict[tuple, List[GsmJob]]" = OrderedDict()
    for job in jobs:
        method_name = (
            job.config_name if not job.order_name else f"{job.config_name}_{job.order_name}"
        )
        if not args.force_rerun and _already_done(
            str(output_csv), method_name, job.scenario, job.master_seed
        ):
            continue
        key = (job.config_name, job.order_name or "", int(job.master_seed))
        groups.setdefault(key, []).append(job)

    if not groups:
        print("[gsm-suite] nothing to run (all rows already in CSV)")
        return

    # Independent (config,seed) groups: default serial in-process (PE via env is
    # process-global). Set GSM_GSM_GROUP_WORKERS>1 to overlap groups with threads
    # ONLY when each group uses distinct PE workspaces AND callers accept that
    # Hermes/PE constructors read env at start of each group (thread-safe if
    # groups don't interleave PE ops — we run one group fully per thread).
    # For true isolation under concurrency, prefer workers=1 or separate OS
    # processes launched by run_all_paper_experiments.
    try:
        from multi_agent.config_loader import get_config

        group_workers = int(
            os.environ.get("GSM_GSM_GROUP_WORKERS")
            or getattr(get_config().simulation, "gsm_group_workers", 1)
            or 1
        )
    except Exception:
        group_workers = 1
    group_workers = max(1, min(group_workers, len(groups)))

    def _run_one_group(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Run one (config, seed) group with isolated PE workspace paths."""
        # Bind env for this thread for the duration of the group. Groups are
        # scheduled without nested PE access across threads when workers>1 only
        # if each payload uses a unique workspace (guaranteed below).
        prev = {
            k: os.environ.get(k)
            for k in (
                "GSM_PE_JSON",
                "GSM_PE_BASE",
                "GSM_HERMES_JSON",
                "GSM_HERMES_RL_BEST",
                "GSM_MATLAB_SCRIPTS_DIR",
                "GSM_MATLAB_MAX_PARALLEL",
                "GSM_MATLAB_POOL_WORKERS",
            )
        }
        try:
            os.environ["GSM_PE_JSON"] = payload["pe_json"]
            os.environ["GSM_PE_BASE"] = payload["pe_base"]
            os.environ["GSM_HERMES_JSON"] = payload["hermes_json"]
            os.environ["GSM_HERMES_RL_BEST"] = payload["hermes_rl"]
            os.environ["GSM_MATLAB_SCRIPTS_DIR"] = payload["scripts_dir"]
            os.environ["GSM_MATLAB_MAX_PARALLEL"] = str(
                max(1, int(payload.get("matlab_slots", 3)))
            )
            os.environ["GSM_MATLAB_POOL_WORKERS"] = str(
                max(1, int(payload.get("matlab_slots", 3)))
            )

            method_name = payload["method_name"]
            tag = payload["tag"]
            _, group_dir = begin_experiment_group(tag)
            print(f"[gsm-suite] Cleared PE/memory for {tag} → {group_dir}", flush=True)
            rows: List[Dict[str, Any]] = []
            job_list = list(payload["jobs"])
            try:
                from multi_agent.simulation.pipeline_warmup import (
                    prefetch_scenario_sequence,
                    start_warmup_async,
                    wait_warmup,
                )

                scen_seq = [j["scenario"] for j in job_list]
                ms = int(job_list[0]["master_seed"]) if job_list else 42
                prefetch_scenario_sequence(scen_seq, master_seed=ms)
            except Exception as exc:
                print(f"[gsm-suite] pipeline warmup skip: {exc}", flush=True)
                start_warmup_async = None  # type: ignore
                wait_warmup = None  # type: ignore

            csv_lock = payload.get("csv_lock")
            out_csv = str(payload.get("output_csv") or "")

            for idx, job_d in enumerate(job_list):
                if wait_warmup is not None:
                    wait_warmup(
                        job_d["scenario"],
                        master_seed=int(job_d["master_seed"]),
                        timeout_sec=180.0,
                    )
                if start_warmup_async is not None and idx + 1 < len(job_list):
                    nxt = job_list[idx + 1]
                    os.environ["GSM_WARMUP_NEXT_SCENARIO"] = str(nxt["scenario"])
                    os.environ["GSM_WARMUP_NEXT_SEED"] = str(nxt["master_seed"])
                    start_warmup_async(
                        nxt["scenario"],
                        master_seed=int(nxt["master_seed"]),
                    )
                else:
                    os.environ.pop("GSM_WARMUP_NEXT_SCENARIO", None)
                    os.environ.pop("GSM_WARMUP_NEXT_SEED", None)

                started = datetime.now()
                config = _suite_config(job_d["suite"], job_d["config_name"], args)
                row = run_single_experiment(
                    "GSMultiAgent",
                    job_d["scenario"],
                    int(job_d["master_seed"]),
                    config,
                )
                row["method"] = method_name
                rows.append(row)
                # Per-scenario durable writeback so a mid-group kill keeps prior rows.
                if out_csv:
                    if csv_lock is not None:
                        with csv_lock:
                            _append_row(out_csv, row)
                    else:
                        _append_row(out_csv, row)
                    print(
                        f"[gsm-suite] appended {method_name}/{job_d['scenario']}/"
                        f"seed{job_d['master_seed']} → {out_csv}",
                        flush=True,
                    )
                archive_scripts_and_reports(
                    group_dir,
                    job_d["scenario"],
                    only_newer_than=datetime.fromtimestamp(started.timestamp() - 1.0),
                )
            write_group_manifest(group_dir, {"status": "completed"})
            return rows
        finally:
            for k, v in prev.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    payloads: List[Dict[str, Any]] = []
    work_root = REPO_ROOT / "guidance_output" / "pe_workspaces"
    work_root.mkdir(parents=True, exist_ok=True)
    try:
        from multi_agent.simulation.matlab_parallel import resolve_matlab_max_parallel

        matlab_slots = resolve_matlab_max_parallel(default=2)
    except Exception:
        matlab_slots = max(1, 2 // max(1, group_workers))
    matlab_slots = max(1, int(matlab_slots) // max(1, group_workers))
    import threading

    csv_lock = threading.Lock()
    for key, group_jobs in groups.items():
        config_name, order_name, master_seed = key
        method_name = config_name if not order_name else f"{config_name}_{order_name}"
        tag = f"{group_jobs[0].suite}_{method_name}_seed{master_seed}"
        ws = work_root / tag
        ws.mkdir(parents=True, exist_ok=True)
        scripts_dir = ws / "scripts"
        scripts_dir.mkdir(parents=True, exist_ok=True)
        payloads.append(
            {
                "tag": tag,
                "method_name": method_name,
                "pe_json": str(ws / "parameter_experience_memory.json"),
                "pe_base": str(ws / "parameter_experience_base"),
                "hermes_json": str(ws / "hermes_agent_memory.json"),
                "hermes_rl": str(ws / "hermes_rl_best.json"),
                "scripts_dir": str(scripts_dir),
                "matlab_slots": matlab_slots,
                "output_csv": str(output_csv),
                "csv_lock": csv_lock,
                "jobs": [
                    {
                        "suite": j.suite,
                        "config_name": j.config_name,
                        "scenario": j.scenario,
                        "master_seed": int(j.master_seed),
                    }
                    for j in group_jobs
                ],
            }
        )

    print(
        f"[gsm-suite] launching {len(payloads)} groups with workers={group_workers} "
        f"(per-group matlab_slots={matlab_slots}; per-scenario CSV append)",
        flush=True,
    )

    # Serialize group execution by default. Parallel groups share process-global
    # os.environ and would race PE paths; use run_all multi-process for seed
    # parallelism instead.
    # Rows are appended inside _run_one_group after each scenario.
    if group_workers <= 1 or len(payloads) == 1:
        for payload in payloads:
            _run_one_group(payload)
    else:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=group_workers) as pool:
            futs = [pool.submit(_run_one_group, p) for p in payloads]
            for fut in as_completed(futs):
                fut.result()

def make_plan(args: argparse.Namespace) -> None:
    scenarios = list(args.scenarios)
    master_seeds = [int(seed) for seed in args.master_seeds]
    output_dir = Path(args.output_dir)
    commands = build_plan(
        sections=args.sections,
        scenarios=scenarios,
        master_seeds=master_seeds,
        n_runs=int(args.n_runs),
        iterations=int(args.iterations),
        output_root=output_dir,
    )
    protocol = protocol_payload(
        scenarios=scenarios,
        master_seeds=master_seeds,
        n_runs=int(args.n_runs),
        iterations=int(args.iterations),
    )
    write_plan(commands, output_dir, protocol)
    print(f"Wrote: {output_dir / 'manifest.json'}")
    print(f"Wrote: {output_dir / 'commands.ps1'}")
    print(f"Wrote: {output_dir / 'commands_execute.ps1'}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run or plan GSMMultiAgent paper experiments.")
    sub = parser.add_subparsers(dest="command", required=True)

    plan = sub.add_parser("make-plan", help="Write manifest.json and commands.ps1.")
    plan.add_argument(
        "--sections",
        nargs="+",
        default=["mc", "optimizer", "memory", "order", "optimization", "correction"],
        choices=["mc", "optimizer", "memory", "order", "optimization", "correction"],
    )
    plan.add_argument("--scenarios", nargs="+", default=list(DEFAULT_SCENARIOS))
    plan.add_argument("--master-seeds", nargs="*", default=[str(s) for s in DEFAULT_MASTER_SEEDS])
    plan.add_argument("--iterations", type=int, default=5)
    plan.add_argument("--n-runs", type=int, default=100)
    plan.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_ROOT))
    plan.set_defaults(func=make_plan)

    mc = sub.add_parser("generate-mc", help="Generate external MC dev/test .mat files.")
    mc.add_argument("--output-root", default="data/mc")
    mc.add_argument("--n-runs", type=int, default=100)
    mc.add_argument("--no-broadside", action="store_true")
    mc.set_defaults(func=run_mc_generator)

    opt = sub.add_parser("run-optimizer-baselines", help="Execute Section 4.3 optimizer baselines.")
    opt.add_argument("--methods", nargs="+", default=["random_search", "cma_es", "direct_ppo"])
    opt.add_argument("--scenarios", nargs="+", default=list(DEFAULT_SCENARIOS))
    opt.add_argument("--master-seeds", nargs="*", default=[str(s) for s in DEFAULT_MASTER_SEEDS])
    opt.add_argument(
        "--iterations",
        "--max-candidates",
        dest="iterations",
        type=int,
        default=DEFAULT_MAX_CANDIDATES,
        help=(
            "Shared candidate-evaluation ceiling for every baseline method "
            f"(default {DEFAULT_MAX_CANDIDATES} = 5 outer × 15 PPO episodes). "
            "Not CMA-ES generations."
        ),
    )
    opt.add_argument("--n-runs", type=int, default=100)
    opt.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_ROOT / "optimizer_baselines_43"))
    opt.add_argument("--keep-going", action="store_true")
    opt.set_defaults(func=run_optimizer_baselines)

    gsm = sub.add_parser("run-gsm-suite", help="Plan or execute GSMMultiAgent ablation suites.")
    gsm.add_argument("--suite", required=True, choices=sorted(SUITE_CONFIGS))
    gsm.add_argument("--configs", nargs="*", default=[])
    gsm.add_argument("--scenarios", nargs="+", default=list(DEFAULT_SCENARIOS))
    gsm.add_argument("--master-seeds", nargs="*", default=[str(s) for s in DEFAULT_MASTER_SEEDS])
    gsm.add_argument("--order", default="forward", choices=sorted(TASK_ORDERS))
    gsm.add_argument("--iterations", type=int, default=5)
    gsm.add_argument("--n-runs", type=int, default=100)
    gsm.add_argument("--output-csv", default=str(DEFAULT_OUTPUT_ROOT / "ablation.csv"))
    gsm.add_argument("--execute", action="store_true", help="Actually run jobs; default only writes protocol.")
    gsm.add_argument("--gsm-dry-run", action="store_true", help="Use runner dry_run mode when executing.")
    gsm.add_argument("--force-rerun", action="store_true")
    gsm.set_defaults(func=run_gsm_suite)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if hasattr(args, "master_seeds"):
        args.master_seeds = _split_ints(args.master_seeds, DEFAULT_MASTER_SEEDS)
    args.func(args)


if __name__ == "__main__":
    main()
