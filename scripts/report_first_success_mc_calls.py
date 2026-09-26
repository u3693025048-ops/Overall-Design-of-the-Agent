#!/usr/bin/env python3
"""从基线 JSON 的 history 累加「第一次可行解」的开发集 MC 调用次数。

用法（仓库根目录）：

    python scripts/report_first_success_mc_calls.py

默认读 results/experiment_plan/optimizer_equal_mc_75/runs/
写出 first_success_mc_calls.csv（不改原 runs.csv）。
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNS = ROOT / "results" / "experiment_plan" / "optimizer_equal_mc_75" / "runs"
METHODS = ("random_search", "cma_es", "direct_ppo")


def first_success_mc_calls(payload: dict) -> tuple[int | None, int | None]:
    """返回 (第几个候选, 累计 MC 次数)。找不到则为 (None, None)。"""
    cum = 0
    for k, item in enumerate(payload.get("history") or [], 1):
        dev = item.get("dev") or {}
        cum += int(dev.get("mc_calls") or 3)
        if dev.get("all_pass"):
            return k, cum
    return None, None


def main() -> int:
    try:
        import sys

        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS)
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=None,
        help="默认写到 runs 目录的上一级 first_success_mc_calls.csv",
    )
    args = parser.parse_args()
    runs_dir = args.runs_dir
    if not runs_dir.is_dir():
        raise SystemExit(f"找不到目录: {runs_dir}")

    rows = []
    for path in sorted(runs_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        method = str(payload.get("method") or "")
        if method not in METHODS:
            continue
        cand, mc_first = first_success_mc_calls(payload)
        rows.append(
            {
                "method": method,
                "scenario": payload.get("scenario"),
                "master_seed": payload.get("master_seed"),
                "first_success_iter": payload.get("first_success_iter"),
                "first_feasible_candidate": cand if cand is not None else "",
                "mc_calls_first_success": mc_first if mc_first is not None else "",
                "mc_calls_dev": payload.get("mc_calls_dev"),
                "dev_all_pass": payload.get("dev_all_pass"),
                "json_name": path.name,
            }
        )

    print(
        f"{'method':<16} {'sc':<3} {'seed':<5} "
        f"{'iter':>6} {'cand':>5} {'mc_first':>8} {'mc_cap':>7} final_dev"
    )
    for row in rows:
        print(
            f"{row['method']:<16} {row['scenario']:<3} {row['master_seed']:<5} "
            f"{str(row['first_success_iter']):>6} {str(row['first_feasible_candidate']):>5} "
            f"{str(row['mc_calls_first_success']):>8} {str(row['mc_calls_dev']):>7} "
            f"{row['dev_all_pass']}"
        )

    by_method: dict[str, list] = defaultdict(list)
    for row in rows:
        val = row["mc_calls_first_success"]
        by_method[row["method"]].append(int(val) if val != "" else None)

    print()
    for method in METHODS:
        vals = by_method.get(method) or []
        found = [v for v in vals if v is not None]
        mean = (sum(found) / len(found)) if found else float("nan")
        print(
            f"{method}: 找到 {len(found)}/{len(vals)}  "
            f"mc_first={found}  均值={mean:.1f}"
        )

    out_csv = args.out_csv or (runs_dir.parent / "first_success_mc_calls.csv")
    fieldnames = list(rows[0].keys()) if rows else []
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
