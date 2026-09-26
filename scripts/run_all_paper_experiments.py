#!/usr/bin/env python3
"""Run the full paper experiment stack in README recommended order.

Order (experiments/README_experiment_plan_full.md §12):
  1) Main / memory ablation (E0→E3, seed 42 by default, T1–T4)
  2) Optimizer baselines (random_search, cma_es, direct_ppo)
  3) Task-order robustness (forward / reverse / shuffled)
  4) Optimization-agent ablation
  5) Correction-pipeline ablation (T4)

Speed profile defaults: single master seed 42; pass 42 43 44 for full paper replicates.
Each step is resumable via existing CSV / result folders.
UTF-8 logging avoids PowerShell Tee-Object mojibake.

Usage:
  python scripts/run_all_paper_experiments.py
  python scripts/run_all_paper_experiments.py --from-step memory
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
OUT = ROOT / "results" / "experiment_plan"
LOG_DIR = ROOT / "logs"


STEPS: list[tuple[str, list[str]]] = [
    (
        "memory",
        [
            PY,
            str(ROOT / "scripts" / "run_parallel_master_seeds.py"),
            "--suite",
            "memory",
            "--configs",
            "E0_full",
            "E1_no_memory",
            "E2_no_reflection",
            "E3_no_memory_no_reflection",
            "--master-seeds",
            "42",
            "--scenarios",
            "T1",
            "T2",
            "T3",
            "T4",
            "--iterations",
            "5",
            "--n-runs",
            "100",
            "--output-csv",
            str(OUT / "ablation_memory.csv"),
            "--matlab-slots-per-seed",
            "2",
        ],
    ),
    (
        "optimizer",
        [
            PY,
            str(ROOT / "experiments" / "optimizer_baselines_43.py"),
            "--methods",
            "random_search",
            "cma_es",
            "direct_ppo",
            "--scenarios",
            "T1",
            "T2",
            "T3",
            "T4",
            "--master-seeds",
            "42",
            "--iterations",
            "5",
            "--n-runs",
            "100",
            "--output-dir",
            str(OUT / "optimizer_baselines_43"),
            "--keep-going",
        ],
    ),
    # order_forward omitted: identical to memory/E0_full T1→T4 baseline order
    (
        "order_reverse",
        [
            PY,
            str(ROOT / "experiments" / "experiment_plan_runner.py"),
            "run-gsm-suite",
            "--suite",
            "order",
            "--order",
            "reverse",
            "--scenarios",
            "T1",
            "T2",
            "T3",
            "T4",
            "--master-seeds",
            "42",
            "--iterations",
            "5",
            "--n-runs",
            "100",
            "--output-csv",
            str(OUT / "ablation_order_reverse.csv"),
            "--execute",
        ],
    ),
    (
        "order_shuffled",
        [
            PY,
            str(ROOT / "experiments" / "experiment_plan_runner.py"),
            "run-gsm-suite",
            "--suite",
            "order",
            "--order",
            "shuffled",
            "--scenarios",
            "T1",
            "T2",
            "T3",
            "T4",
            "--master-seeds",
            "42",
            "--iterations",
            "5",
            "--n-runs",
            "100",
            "--output-csv",
            str(OUT / "ablation_order_shuffled.csv"),
            "--execute",
        ],
    ),
    (
        "optimization",
        [
            PY,
            str(ROOT / "experiments" / "experiment_plan_runner.py"),
            "run-gsm-suite",
            "--suite",
            "optimization",
            "--scenarios",
            "T1",
            "T2",
            "T3",
            "T4",
            "--master-seeds",
            "42",
            "--iterations",
            "5",
            "--n-runs",
            "100",
            "--output-csv",
            str(OUT / "ablation_optimization.csv"),
            "--execute",
        ],
    ),
    (
        "correction",
        [
            PY,
            str(ROOT / "experiments" / "experiment_plan_runner.py"),
            "run-gsm-suite",
            "--suite",
            "correction",
            "--scenarios",
            "T4",
            "--master-seeds",
            "42",
            "--iterations",
            "5",
            "--n-runs",
            "100",
            "--output-csv",
            str(OUT / "ablation_correction.csv"),
            "--execute",
        ],
    ),
]


def _run_step(name: str, cmd: list[str], log_path: Path) -> int:
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = datetime.now().isoformat(timespec="seconds")
    print(f"\n===== [{name}] START {started} =====", flush=True)
    print(f"===== log: {log_path}", flush=True)
    print(f"===== cmd: {' '.join(cmd)}", flush=True)
    with log_path.open("a", encoding="utf-8", errors="replace", newline="\n") as fh:
        fh.write(f"\n# ===== [{name}] START {started} =====\n")
        fh.write(f"# cmd: {' '.join(cmd)}\n\n")
        fh.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            fh.write(line)
            fh.flush()
        code = int(proc.wait())
        ended = datetime.now().isoformat(timespec="seconds")
        fh.write(f"\n# ===== [{name}] END exit={code} {ended} =====\n")
    print(f"===== [{name}] END exit={code} =====", flush=True)
    return code


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Run full paper experiment stack")
    p.add_argument(
        "--from-step",
        default=STEPS[0][0],
        choices=[s[0] for s in STEPS],
        help="Resume from this step (inclusive)",
    )
    p.add_argument(
        "--only-step",
        default="",
        choices=[""] + [s[0] for s in STEPS],
        help="Run only one step",
    )
    p.add_argument(
        "--keep-going",
        action="store_true",
        default=True,
        help="Continue after a failed step (default: true)",
    )
    p.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop the stack when a step fails",
    )
    args = p.parse_args(argv)

    OUT.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    master_log = LOG_DIR / f"all_paper_experiments_{ts}.txt"
    status_path = LOG_DIR / f"all_paper_experiments_{ts}.status.txt"

    start_idx = 0
    if args.only_step:
        steps = [s for s in STEPS if s[0] == args.only_step]
    else:
        for i, (name, _) in enumerate(STEPS):
            if name == args.from_step:
                start_idx = i
                break
        steps = STEPS[start_idx:]

    keep_going = not bool(args.stop_on_error)
    results: list[tuple[str, int]] = []
    with master_log.open("w", encoding="utf-8", newline="\n") as mh:
        mh.write(f"# all paper experiments started {datetime.now().isoformat(timespec='seconds')}\n")
        mh.write(f"# steps: {[n for n, _ in steps]}\n")
        mh.write(f"# master_log: {master_log}\n\n")

    print(f"[run_all] master_log={master_log}", flush=True)
    print(f"[run_all] steps={[n for n, _ in steps]}", flush=True)

    for name, cmd in steps:
        step_log = LOG_DIR / f"all_paper_{name}_{ts}.txt"
        code = _run_step(name, cmd, step_log)
        results.append((name, code))
        with status_path.open("a", encoding="utf-8", newline="\n") as sf:
            sf.write(f"{datetime.now().isoformat(timespec='seconds')}\t{name}\texit={code}\n")
        with master_log.open("a", encoding="utf-8", newline="\n") as mh:
            mh.write(f"{name}\texit={code}\tlog={step_log}\n")
        if code != 0 and not keep_going:
            print(f"[run_all] stopping after failed step {name}", flush=True)
            break

    print("\n[run_all] summary:", flush=True)
    for name, code in results:
        print(f"  {name}: exit={code}", flush=True)
    failed = [n for n, c in results if c != 0]
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
