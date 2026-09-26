#!/usr/bin/env python3
"""UTF-8 safe launcher for long experiment runs (avoids PowerShell Tee-Object mojibake).

Example:
  python scripts/run_e0_seed42.py
  python scripts/run_utf8.py -- experiments/experiment_plan_runner.py run-gsm-suite ...
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Run a command with UTF-8 stdout/stderr log")
    p.add_argument(
        "--log",
        type=str,
        default="",
        help="Log file path (default: logs/run_YYYYMMDD_HHMMSS.txt)",
    )
    p.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="Command after -- , e.g. -- python experiments/...",
    )
    args = p.parse_args(argv)
    cmd = list(args.command)
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]
    if not cmd:
        # Default: E0 seed42 T1-T4
        cmd = [
            sys.executable,
            str(ROOT / "experiments" / "experiment_plan_runner.py"),
            "run-gsm-suite",
            "--suite",
            "memory",
            "--configs",
            "E0_full",
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
            "--execute",
            "--output-csv",
            str(ROOT / "results" / "experiment_plan" / "e0_seed42_t14.csv"),
        ]

    logs = ROOT / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    log_path = Path(args.log) if args.log else logs / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    if not log_path.is_absolute():
        log_path = ROOT / log_path

    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"

    print(f"[run_utf8] log={log_path}")
    print(f"[run_utf8] cmd={' '.join(cmd)}")

    with log_path.open("w", encoding="utf-8", errors="replace", newline="\n") as fh:
        fh.write(f"# started {datetime.now().isoformat(timespec='seconds')}\n")
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
        code = proc.wait()
        fh.write(f"\n# ended exit={code} {datetime.now().isoformat(timespec='seconds')}\n")
    return int(code)


if __name__ == "__main__":
    raise SystemExit(main())
