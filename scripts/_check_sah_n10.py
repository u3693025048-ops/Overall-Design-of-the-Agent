#!/usr/bin/env python3
"""One-shot status of the Ubuntu SAH n10 12-cell run."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import paramiko

ROOT = Path(__file__).resolve().parents[1]
HOST_CANDIDATES = (
    os.environ.get("GSM_SSH_HOST", "").strip(),
    "10.13.11.221",
    "192.168.111.8",
)
USER = os.environ.get("GSM_SSH_USER", "admcai")
REMOTE = os.environ.get("GSM_REMOTE_DIR", "/home/admcai/workspace/GSMultiAgent13-V0.3")
OUT = f"{REMOTE}/results/experiment_plan/optimizer_equal_mc_75_sah_n10"
LOG_GLOB = f"{REMOTE}/logs/ubuntu_sah_n10_*.txt"

PASS = os.environ.get("GSM_SSH_PASS", "")
if not PASS:
    pf = ROOT / "scripts" / ".gsm_ssh_pass"
    if pf.is_file():
        PASS = pf.read_text(encoding="utf-8").strip()
if not PASS:
    raise SystemExit("GSM_SSH_PASS or scripts/.gsm_ssh_pass required")

CMD = f"""
echo '=== TIME ==='
date
echo '=== PROCESS ==='
ps -ef | grep -E 'optimizer_baselines_43|gsm_sah_n10' | grep -v grep || echo IDLE
echo '=== DONE_JSON ==='
ls -1 {OUT}/runs/*.json 2>/dev/null | wc -l
ls -1 {OUT}/runs/*.json 2>/dev/null || true
echo '=== RUNS_CSV ==='
if [ -f {OUT}/runs.csv ]; then
  python3 - <<'PY' || true
import csv
from pathlib import Path
p = Path("{OUT}/runs.csv")
rows = list(csv.DictReader(p.open(encoding="utf-8")))
print("rows", len(rows))
for r in rows:
    print(r.get("scenario"), r.get("master_seed"), "dev", r.get("dev_all_pass"), "test", r.get("test_all_pass"), "first", r.get("first_success"), "calls", r.get("mc_calls_dev"), "min", r.get("total_time_min"))
PY
else
  echo NO_CSV
fi
echo '=== LOG ==='
LOG=$(ls -t {LOG_GLOB} 2>/dev/null | head -1)
echo log=$LOG
if [ -n "$LOG" ]; then
  grep -E '\\[start\\]|\\[cell-done\\]|\\[cell-fail\\]|\\[all-done\\]|\\[4\\.3\\]|MC protocol|Outer ' "$LOG" | tail -n 40
  echo '--- tail ---'
  tail -n 15 "$LOG"
fi
"""


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    last_err = None
    for host in HOST_CANDIDATES:
        if not host:
            continue
        try:
            client.connect(
                host,
                username=USER,
                password=PASS,
                timeout=20,
                allow_agent=False,
                look_for_keys=False,
            )
            print(f"[ssh] {USER}@{host}", flush=True)
            break
        except Exception as exc:
            last_err = exc
            client = paramiko.SSHClient()
            client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    else:
        print(f"SSH_FAIL {last_err}", flush=True)
        return 2
    _, o, e = client.exec_command(CMD, timeout=60)
    print(o.read().decode("utf-8", "replace"), end="")
    err = e.read().decode("utf-8", "replace").strip()
    if err:
        print("STDERR:", err[:800], flush=True)
    client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
