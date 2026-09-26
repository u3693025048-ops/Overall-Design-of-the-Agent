#!/usr/bin/env python3
"""Deploy 12-cell SAH (search n_runs=10) to Ubuntu; new output dir, no overwrite."""
from __future__ import annotations

import os
import sys
import time
from datetime import datetime
from pathlib import Path

import paramiko

ROOT = Path(__file__).resolve().parents[1]
HOST = os.environ.get("GSM_SSH_HOST", "")
HOST_CANDIDATES = (
    HOST,
    "10.13.11.221",
    "192.168.111.8",
)
USER = os.environ.get("GSM_SSH_USER", "admcai")
REMOTE = os.environ.get("GSM_REMOTE_DIR", "/home/admcai/workspace/GSMultiAgent13-V0.3")
PY = "/home/admcai/miniconda3/envs/toddler_ai/bin/python"
MATLAB = "/usr/local/Polyspace/R2020b/bin/matlab"
OUT_SAH = "results/experiment_plan/optimizer_equal_mc_75_sah_n10"
PE_DIR = "parameter_experience_base/single_agent_hermes_equal_mc_n10"
OLD_SAH = "results/experiment_plan/optimizer_equal_mc_75_sah"

CELLS = tuple(
    (scenario, seed)
    for scenario in ("T1", "T2", "T3", "T4")
    for seed in (42, 43, 44)
)

SYNC_FILES = (
    "experiments/single_agent_hermes/agent.py",
    "experiments/optimizer_baselines_43.py",
    "experiments/equal_mc_budget.py",
    "multi_agent/tools/simulation_tool.py",
)

PASS = os.environ.get("GSM_SSH_PASS", "")
if not PASS:
    pf = ROOT / "scripts" / ".gsm_ssh_pass"
    if pf.is_file():
        PASS = pf.read_text(encoding="utf-8").strip()
if not PASS:
    raise SystemExit("GSM_SSH_PASS or scripts/.gsm_ssh_pass required")


def run(client: paramiko.SSHClient, cmd: str, timeout: int = 90) -> tuple[int, str]:
    print(f"==== {cmd[:240]}", flush=True)
    _, o, e = client.exec_command(cmd, timeout=timeout)
    code = o.channel.recv_exit_status()
    out = o.read().decode("utf-8", "replace")
    err = e.read().decode("utf-8", "replace")
    if out.strip():
        print(out.rstrip(), flush=True)
    if err.strip():
        print("STDERR:", err[:1500].rstrip(), flush=True)
    print(f"[exit={code}]", flush=True)
    return code, out


def upload(sftp: paramiko.SFTPClient, rel: str) -> None:
    src = ROOT / rel
    dst = f"{REMOTE}/{rel.replace(chr(92), '/')}"
    sftp.put(str(src), dst)
    print(f"[upload] {rel} -> {dst}", flush=True)


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    last_err: Exception | None = None
    connected_host = ""
    for host in HOST_CANDIDATES:
        host = (host or "").strip()
        if not host:
            continue
        try:
            print(f"[ssh] trying {USER}@{host}", flush=True)
            client.connect(
                host,
                username=USER,
                password=PASS,
                timeout=20,
                allow_agent=False,
                look_for_keys=False,
            )
            connected_host = host
            break
        except Exception as exc:
            last_err = exc
            print(f"[ssh] {host} failed: {type(exc).__name__}: {exc}", flush=True)
            client = paramiko.SSHClient()
            client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    if not connected_host:
        raise SystemExit(f"SSH failed: {last_err}")
    print(f"[ssh] connected {USER}@{connected_host}", flush=True)

    _, ps = run(
        client,
        "ps -ef | grep -E 'optimizer_baselines_43|ubuntu_experiment_queue|gsm_equal_mc|gsm_sah_' "
        "| grep -v grep || echo IDLE",
    )
    if "optimizer_baselines_43" in ps and "IDLE" not in ps:
        print("Ubuntu already has optimizer_baselines_43 running; abort.", flush=True)
        client.close()
        return 2

    sftp = client.open_sftp()
    for rel in SYNC_FILES:
        upload(sftp, rel)
    sftp.close()

    code, _ = run(
        client,
        f"cd {REMOTE} && {PY} - <<'PY'\n"
        "from pathlib import Path\n"
        "from experiments.single_agent_hermes.agent import SingleAgentHermes, FIRST_SUCCESS_DEFINITION\n"
        "from multi_agent.tools.simulation_tool import run_mc_and_across_seeds\n"
        "import inspect\n"
        "assert FIRST_SUCCESS_DEFINITION == 'final_dev_feasible'\n"
        "agent = SingleAgentHermes(mc_root=Path('.'), matlab_script=Path('x.m'), n_runs=100, use_llm=False)\n"
        "assert agent._n_runs_for_split('dev') == 10\n"
        "assert agent._n_runs_for_split('test') == 100\n"
        "assert 'bootstrap_fill' in inspect.signature(run_mc_and_across_seeds).parameters\n"
        "print('remote_sah_ok', FIRST_SUCCESS_DEFINITION, agent._n_runs_for_split('dev'), agent._n_runs_for_split('test'))\n"
        "PY",
        timeout=40,
    )
    if code != 0:
        print("Remote SAH protocol check failed; abort.", flush=True)
        client.close()
        return 3

    code, _ = run(
        client,
        f"cd {REMOTE} && {PY} - <<'PY'\n"
        "from multi_agent.llm_failover import endpoints_from_config, probe_endpoint\n"
        "ep = endpoints_from_config()[0]\n"
        "ok, detail = probe_endpoint(ep)\n"
        "host = (ep.base_url or '').split('//')[-1].split('/')[0]\n"
        "print('ubuntu_llm', ep.model, host, 'OK' if ok else 'FAIL', (detail or '')[:120])\n"
        "if not ok:\n"
        "    raise SystemExit(4)\n"
        "PY",
        timeout=40,
    )
    if code != 0:
        print("Ubuntu LLM probe failed; abort.", flush=True)
        client.close()
        return 4

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log = f"{REMOTE}/logs/ubuntu_sah_n10_{stamp}.txt"
    remote_sh = f"/tmp/gsm_sah_n10_{stamp}.sh"
    pe_root = f"{REMOTE}/{PE_DIR}"

    run(
        client,
        f"test -d {REMOTE}/{OLD_SAH} && echo KEEP_OLD_SAH_OK || echo OLD_SAH_MISSING; "
        f"mkdir -p {REMOTE}/logs {REMOTE}/{OUT_SAH}/runs {pe_root}; "
        f"test ! -e {REMOTE}/{OUT_SAH}/runs.csv && echo NEW_DIR_EMPTY || echo NEW_DIR_HAS_CSV",
    )

    cells = "\n".join(
        f"export GSM_SAH_PE_DIR={pe_root}\n"
        f"{PY} -u {REMOTE}/experiments/optimizer_baselines_43.py \\\n"
        f"  --methods single_agent_hermes \\\n"
        f"  --scenarios {scenario} \\\n"
        f"  --master-seeds {seed} \\\n"
        f"  --max-candidates 75 --n-runs 100 \\\n"
        f"  --output-dir {REMOTE}/{OUT_SAH} --keep-going \\\n"
        f"  >> {log} 2>&1 || echo \"[cell-fail] {scenario} seed{seed} $(date)\" | tee -a {log}\n"
        f'echo "[cell-done] {scenario} seed{seed} $(date)" | tee -a {log}\n'
        for scenario, seed in CELLS
    )
    script = f"""#!/bin/bash
set -euo pipefail
cd {REMOTE}
export PATH=/home/admcai/miniconda3/envs/toddler_ai/bin:$PATH
export PYTHONIOENCODING=utf-8
export PYTHONUTF8=1
export MATLAB_PATH={MATLAB}
export MATLAB_TIMEOUT_SEC=1200
export GSM_SAH_PE_DIR={pe_root}
mkdir -p {OUT_SAH}/runs logs {PE_DIR}
echo "[start] $(date) SAH n_runs=10 12 cells out={OUT_SAH}" | tee -a {log}
{cells}
echo "[all-done] $(date)" | tee -a {log}
"""
    sftp = client.open_sftp()
    with sftp.file(remote_sh, "w") as fh:
        fh.write(script)
    sftp.chmod(remote_sh, 0o755)
    sftp.close()

    run(
        client,
        f"nohup bash {remote_sh} > /tmp/gsm_sah_n10_nohup_{stamp}.out 2>&1 & echo LAUNCH_PID:$!",
    )
    time.sleep(8)
    run(
        client,
        "ps -ef | grep -E 'optimizer_baselines_43|gsm_sah_n10' | grep -v grep | head -20; "
        f"echo '--- log ---'; tail -n 50 {log} 2>/dev/null || true; "
        f"echo '--- old dir untouched ---'; ls {REMOTE}/{OLD_SAH}/runs.csv >/dev/null && echo OLD_SAH_CSV_OK",
    )
    print(f"Remote log: {log}", flush=True)
    print(f"SAH out:    {REMOTE}/{OUT_SAH}", flush=True)
    print(f"Old SAH:    {REMOTE}/{OLD_SAH} (not overwritten)", flush=True)
    client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
