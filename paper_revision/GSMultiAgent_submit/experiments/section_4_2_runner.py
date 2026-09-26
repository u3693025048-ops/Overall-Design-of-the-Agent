"""
Section 4.2.1 and 4.2.2 Experiment Runner

Section 4.2.1: Pluggable Optimizer Backend Ablation
- Compare hybrid_ppo, direct_ppo, cma_es, random_search on GSMultiAgent
- Keep cascade/memory/harness intact, only swap optimizer backend

Section 4.2.2: Cascade Ablation (Single-Agent Hermes vs GSMultiAgent)
- Compare full GSMultiAgent vs HermesAgentSingle
- Same tools/PE/RAG/optimizer, but no sub-agent cascade in HermesAgentSingle

Both experiments output CSV format compatible with runs.csv structure.
"""

import asyncio
import csv
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from multi_agent.config_loader import load_config

logger = logging.getLogger(__name__)


# ============================================================================
# Section 4.2.1: Optimizer Backend Ablation
# ============================================================================

async def run_optimizer_ablation(
    scenarios: Optional[List[str]] = None,
    seeds: Optional[List[int]] = None,
    backends: Optional[List[str]] = None,
    output_dir: Optional[str] = None,
) -> Path:
    """
    Run optimizer backend ablation experiment (Section 4.2.1).

    Compares different optimizer backends on GSMultiAgent while keeping
    cascade/memory/harness intact.

    Args:
        scenarios: List of scenario names (e.g., ['T1', 'T2']). Default: all T1-T4.
        seeds: List of random seeds. Default: [42, 43, 44].
        backends: List of optimizer backend names. Default: all 4 backends.
        output_dir: Output directory path. Default: results/optimizer_ablation_{timestamp}/

    Returns:
        Path to output CSV file
    """
    from cli_agent import run_baseline_scenario

    # Defaults
    if scenarios is None:
        scenarios = ["T1", "T2", "T3", "T4"]
    if seeds is None:
        seeds = [42, 43, 44]
    if backends is None:
        backends = ["hybrid_ppo", "direct_ppo", "cma_es", "random_search"]

    # Output directory
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"results/optimizer_ablation_{timestamp}"
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    csv_path = output_path / "runs.csv"

    logger.info("=" * 80)
    logger.info("Section 4.2.1: Optimizer Backend Ablation")
    logger.info("=" * 80)
    logger.info(f"Scenarios: {scenarios}")
    logger.info(f"Seeds: {seeds}")
    logger.info(f"Backends: {backends}")
    logger.info(f"Output: {csv_path}")
    logger.info("")

    # CSV header
    fieldnames = [
        "scenario", "seed", "backend", "status",
        "final_reward", "final_score", "converged_at_episode",
        "max_reward", "max_score",
        "all_pass_rate", "llm_calls", "elapsed_sec",
    ]

    rows = []
    total_runs = len(scenarios) * len(seeds) * len(backends)
    current_run = 0

    for scenario in scenarios:
        for seed in seeds:
            for backend in backends:
                current_run += 1
                logger.info("-" * 80)
                logger.info(f"[{current_run}/{total_runs}] scenario={scenario} seed={seed} backend={backend}")
                logger.info("-" * 80)

                try:
                    # Load config and override optimizer backend
                    cfg = load_config()
                    cfg.matlab_rl_optimizer.optimizer_backend = backend
                    cfg.seed = seed

                    # Run baseline scenario (GSMultiAgent with specified backend)
                    result = await run_baseline_scenario(
                        scenario_name=scenario,
                        config=cfg,
                    )

                    # Extract metrics
                    row = {
                        "scenario": scenario,
                        "seed": seed,
                        "backend": backend,
                        "status": result.status,
                        "final_reward": result.final_reward if result.final_reward is not None else "",
                        "final_score": result.final_score if result.final_score is not None else "",
                        "converged_at_episode": result.converged_at_episode if result.converged_at_episode is not None else "",
                        "max_reward": result.max_reward if result.max_reward is not None else "",
                        "max_score": result.max_score if result.max_score is not None else "",
                        "all_pass_rate": result.all_pass_rate if result.all_pass_rate is not None else "",
                        "llm_calls": result.llm_calls if result.llm_calls is not None else "",
                        "elapsed_sec": result.elapsed_sec if result.elapsed_sec is not None else "",
                    }

                    logger.info(f"✓ {scenario}/{seed}/{backend}: {result.status}")
                    if result.final_reward is not None:
                        logger.info(f"  final_reward={result.final_reward:.4f} final_score={result.final_score:.4f}")

                except Exception as e:
                    logger.error(f"✗ {scenario}/{seed}/{backend} failed: {e}", exc_info=True)
                    row = {
                        "scenario": scenario,
                        "seed": seed,
                        "backend": backend,
                        "status": "error",
                        "final_reward": "",
                        "final_score": "",
                        "converged_at_episode": "",
                        "max_reward": "",
                        "max_score": "",
                        "all_pass_rate": "",
                        "llm_calls": "",
                        "elapsed_sec": "",
                    }

                rows.append(row)

                # Write incrementally
                with open(csv_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(rows)

    logger.info("=" * 80)
    logger.info(f"Optimizer ablation complete: {csv_path}")
    logger.info("=" * 80)

    return csv_path


# ============================================================================
# Section 4.2.2: Cascade Ablation (GSMultiAgent vs HermesAgentSingle)
# ============================================================================

async def run_cascade_ablation(
    scenarios: Optional[List[str]] = None,
    seeds: Optional[List[int]] = None,
    output_dir: Optional[str] = None,
) -> Path:
    """
    Run cascade ablation experiment (Section 4.2.2).

    Compares GSMultiAgent (full cascade) vs HermesAgentSingle (no cascade).
    Both use same tools/PE/RAG/optimizer.

    Args:
        scenarios: List of scenario names (e.g., ['T1', 'T2']). Default: all T1-T4.
        seeds: List of random seeds. Default: [42, 43, 44].
        output_dir: Output directory path. Default: results/cascade_ablation_{timestamp}/

    Returns:
        Path to output CSV file
    """
    from cli_agent import run_baseline_scenario
    from experiments.hermes_agent_single.agent import HermesAgentSingle

    # Defaults
    if scenarios is None:
        scenarios = ["T1", "T2", "T3", "T4"]
    if seeds is None:
        seeds = [42, 43, 44]

    # Output directory
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"results/cascade_ablation_{timestamp}"
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    csv_path = output_path / "runs.csv"

    logger.info("=" * 80)
    logger.info("Section 4.2.2: Cascade Ablation (GSMultiAgent vs HermesAgentSingle)")
    logger.info("=" * 80)
    logger.info(f"Scenarios: {scenarios}")
    logger.info(f"Seeds: {seeds}")
    logger.info(f"Output: {csv_path}")
    logger.info("")

    # CSV header
    fieldnames = [
        "scenario", "seed", "method", "status",
        "final_reward", "final_score", "converged_at_episode",
        "max_reward", "max_score",
        "all_pass_rate", "llm_calls", "elapsed_sec",
    ]

    rows = []
    methods = ["GSMultiAgent", "HermesAgentSingle"]
    total_runs = len(scenarios) * len(seeds) * len(methods)
    current_run = 0

    for scenario in scenarios:
        for seed in seeds:
            for method in methods:
                current_run += 1
                logger.info("-" * 80)
                logger.info(f"[{current_run}/{total_runs}] scenario={scenario} seed={seed} method={method}")
                logger.info("-" * 80)

                try:
                    # Load config with seed
                    cfg = load_config()
                    cfg.seed = seed

                    if method == "GSMultiAgent":
                        # Full cascade
                        result = await run_baseline_scenario(
                            scenario_name=scenario,
                            config=cfg,
                        )
                    else:
                        # Single-agent Hermes (no cascade)
                        agent = HermesAgentSingle(
                            scenario_name=scenario,
                            config=cfg,
                        )
                        result = await agent.run_async()

                    # Extract metrics
                    row = {
                        "scenario": scenario,
                        "seed": seed,
                        "method": method,
                        "status": result.status,
                        "final_reward": result.final_reward if result.final_reward is not None else "",
                        "final_score": result.final_score if result.final_score is not None else "",
                        "converged_at_episode": result.converged_at_episode if result.converged_at_episode is not None else "",
                        "max_reward": result.max_reward if result.max_reward is not None else "",
                        "max_score": result.max_score if result.max_score is not None else "",
                        "all_pass_rate": result.all_pass_rate if result.all_pass_rate is not None else "",
                        "llm_calls": result.llm_calls if result.llm_calls is not None else "",
                        "elapsed_sec": result.elapsed_sec if result.elapsed_sec is not None else "",
                    }

                    logger.info(f"✓ {scenario}/{seed}/{method}: {result.status}")
                    if result.final_reward is not None:
                        logger.info(f"  final_reward={result.final_reward:.4f} final_score={result.final_score:.4f}")
                    if result.llm_calls is not None:
                        logger.info(f"  llm_calls={result.llm_calls}")

                except Exception as e:
                    logger.error(f"✗ {scenario}/{seed}/{method} failed: {e}", exc_info=True)
                    row = {
                        "scenario": scenario,
                        "seed": seed,
                        "method": method,
                        "status": "error",
                        "final_reward": "",
                        "final_score": "",
                        "converged_at_episode": "",
                        "max_reward": "",
                        "max_score": "",
                        "all_pass_rate": "",
                        "llm_calls": "",
                        "elapsed_sec": "",
                    }

                rows.append(row)

                # Write incrementally
                with open(csv_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(rows)

    logger.info("=" * 80)
    logger.info(f"Cascade ablation complete: {csv_path}")
    logger.info("=" * 80)

    return csv_path


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Run both experiments from command line."""
    import argparse

    parser = argparse.ArgumentParser(description="Run Section 4.2.1 and 4.2.2 experiments")
    parser.add_argument(
        "--experiment",
        choices=["optimizer", "cascade", "both"],
        default="both",
        help="Which experiment to run (default: both)",
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=None,
        help="Scenario names (default: T1 T2 T3 T4)",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=None,
        help="Random seeds (default: 42 43 44)",
    )
    parser.add_argument(
        "--backends",
        nargs="+",
        default=None,
        help="Optimizer backends for 4.2.1 (default: all 4)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: auto-generated timestamp)",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    if args.experiment in ["optimizer", "both"]:
        logger.info("Starting Section 4.2.1: Optimizer Backend Ablation")
        output_path = asyncio.run(run_optimizer_ablation(
            scenarios=args.scenarios,
            seeds=args.seeds,
            backends=args.backends,
            output_dir=args.output_dir,
        ))
        logger.info(f"Section 4.2.1 results: {output_path}")

    if args.experiment in ["cascade", "both"]:
        logger.info("Starting Section 4.2.2: Cascade Ablation")
        output_path = asyncio.run(run_cascade_ablation(
            scenarios=args.scenarios,
            seeds=args.seeds,
            output_dir=args.output_dir,
        ))
        logger.info(f"Section 4.2.2 results: {output_path}")


if __name__ == "__main__":
    main()
