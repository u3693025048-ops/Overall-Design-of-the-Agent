#!/usr/bin/env python3
"""Launch one GSM suite process per master seed (isolated PE / Hermes).

Example (E0 main / memory suite):
  python scripts/run_parallel_master_seeds.py \\
    --suite memory --configs E0_full \\
    --master-seeds 42 --scenarios T1 T2 T3 T4 \\
    --output-csv results/experiment_plan/ablation_memory.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


def _merge_csvs(partials: List[Path], output: Path) -> int:
    rows = []
    fieldnames = None
    for p in partials:
        if not p.is_file() or p.stat().st_size == 0:
            continue
        with p.open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            if reader.fieldnames:
                fieldnames = list(reader.fieldnames)
            for row in reader:
                rows.append(row)
    if not fieldnames:
        return 0
    output.parent.mkdir(parents=True, exist_ok=True)
    # de-dupe by method/scenario/seed keeping last
    keyed = {}
    for row in rows:
        key = (row.get("method"), row.get("scenario"), row.get("seed"))
        keyed[key] = row
    with output.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in keyed.values():
            writer.writerow(row)
    return len(keyed)


def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Parallel master-seed GSM launcher")
    p.add_argument("--suite", default="memory")
    p.add_argument("--configs", nargs="+", default=["E0_full"])
    p.add_argument("--master-seeds", nargs="+", type=int, default=[42])
    p.add_argument("--scenarios", nargs="+", default=["T1", "T2", "T3", "T4"])
    p.add_argument("--iterations", type=int, default=5)
    p.add_argument("--n-runs", type=int, default=100)
    p.add_argument(
        "--output-csv",
        default=str(ROOT / "results" / "experiment_plan" / "ablation_memory.csv"),
    )
    p.add_argument("--order", default="")
    p.add_argument(
        "--matlab-slots-per-seed",
        type=int,
        default=2,
        help="GSM_MATLAB_MAX_PARALLEL / pool workers for each seed process",
    )
    p.add_argument(
        "--force-rerun",
        action="store_true",
        help="Pass --force-rerun to experiment_plan_runner (ignore existing CSV rows)",
    )
    args = p.parse_args(argv)

    out_csv = Path(args.output_csv)
    if not out_csv.is_absolute():
        out_csv = ROOT / out_csv
    log_dir = ROOT / "logs" / "parallel_seeds"
    pe_root = ROOT / "guidance_output" / "pe_workspaces"
    log_dir.mkdir(parents=True, exist_ok=True)
    pe_root.mkdir(parents=True, exist_ok=True)

    procs = []
    partials = []
    for seed in args.master_seeds:
        ws = pe_root / f"parallel_seed{seed}"
        ws.mkdir(parents=True, exist_ok=True)
        scripts = ws / "scripts"
        scripts.mkdir(parents=True, exist_ok=True)
        partial = out_csv.with_name(f"{out_csv.stem}_seed{seed}{out_csv.suffix}")
        partials.append(partial)
        log_path = log_dir / f"seed{seed}_{int(time.time())}.txt"
        cmd = [
            PY,
            "-u",
            str(ROOT / "experiments" / "experiment_plan_runner.py"),
            "run-gsm-suite",
            "--suite",
            args.suite,
            "--configs",
            *args.configs,
            "--master-seeds",
            str(seed),
            "--scenarios",
            *args.scenarios,
            "--iterations",
            str(args.iterations),
            "--n-runs",
            str(args.n_runs),
            "--output-csv",
            str(partial),
            "--execute",
        ]
        if args.force_rerun:
            cmd.append("--force-rerun")
        if args.order:
            cmd.extend(["--order", args.order])

        env = dict(os.environ)
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUTF8"] = "1"
        env["GSM_PE_JSON"] = str(ws / "parameter_experience_memory.json")
        env["GSM_PE_BASE"] = str(ws / "parameter_experience_base")
        env["GSM_HERMES_JSON"] = str(ws / "hermes_agent_memory.json")
        env["GSM_HERMES_RL_BEST"] = str(ws / "hermes_rl_best.json")
        env["GSM_MATLAB_SCRIPTS_DIR"] = str(scripts)
        env["GSM_MATLAB_MAX_PARALLEL"] = str(max(1, int(args.matlab_slots_per_seed)))
        env["GSM_MATLAB_POOL_WORKERS"] = str(max(1, int(args.matlab_slots_per_seed)))
        env["GSM_MC_AND_WORKERS"] = str(min(3, max(1, int(args.matlab_slots_per_seed))))
        # Isolate persistent MATLAB workers per master seed (avoid shared queue races).
        env["GSM_MATLAB_JOB_POOL_DIR"] = str(ws / "matlab_job_pool")

        log_fh = log_path.open("w", encoding="utf-8")
        log_fh.write(f"# seed={seed}\n# cmd={' '.join(cmd)}\n\n")
        log_fh.flush()
        print(f"[parallel-seeds] start seed={seed} log={log_path}")
        proc = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            env=env,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
        )
        procs.append((seed, proc, log_fh, log_path))

    exit_codes = []
    for seed, proc, log_fh, log_path in procs:
        code = proc.wait()
        log_fh.close()
        exit_codes.append(code)
        print(f"[parallel-seeds] seed={seed} exit={code} log={log_path}")

    n = _merge_csvs(partials, out_csv)
    print(f"[parallel-seeds] merged {n} rows → {out_csv}")
    return 0 if all(c == 0 for c in exit_codes) else 1


if __name__ == "__main__":
    raise SystemExit(main())
