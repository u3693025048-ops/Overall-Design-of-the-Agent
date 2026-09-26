#!/usr/bin/env python3
"""Pull Ubuntu experiment CSVs, optimizer dirs, and queue logs to local disk.

Usage:
  $env:GSM_SSH_PASS='...'
  python -u scripts/pull_ubuntu_artifacts.py

Writes into the local repo mirroring remote relative paths, plus a manifest under
results/experiment_plan/ubuntu_pull_<stamp>/.
"""

from __future__ import annotations

import json
import os
import stat
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import paramiko

ROOT = Path(__file__).resolve().parents[1]
HOST = os.environ.get("GSM_SSH_HOST", "10.13.11.221")
USER = os.environ.get("GSM_SSH_USER", "admcai")
PASS = os.environ.get("GSM_SSH_PASS", "")
REMOTE = os.environ.get("GSM_REMOTE_DIR", "/home/admcai/workspace/GSMultiAgent13-V0.3")

if not PASS:
    for cand in (ROOT / "scripts" / ".gsm_ssh_pass", ROOT / ".gsm_ssh_pass"):
        if cand.is_file():
            PASS = cand.read_text(encoding="utf-8").strip()
            break


def _connect() -> paramiko.SSHClient:
    if not PASS:
        raise SystemExit("GSM_SSH_PASS (or scripts/.gsm_ssh_pass) required")
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        HOST,
        username=USER,
        password=PASS,
        timeout=30,
        allow_agent=False,
        look_for_keys=False,
    )
    return client


def _run(client: paramiko.SSHClient, cmd: str, timeout: int = 120) -> Tuple[int, str]:
    _, o, e = client.exec_command(cmd, timeout=timeout)
    code = o.channel.recv_exit_status()
    out = o.read().decode("utf-8", "replace")
    err = e.read().decode("utf-8", "replace")
    if code != 0 and err.strip():
        print(f"[warn exit={code}] {cmd}\n{err[:500]}", flush=True)
    return code, out


def _remote_list(client: paramiko.SSHClient) -> Dict[str, List[str]]:
    """Ask remote to list artifact relative paths."""
    py = r"""
import json
from pathlib import Path
ROOT = Path(%r)
out = {"csv": [], "opt_files": [], "logs": [], "status": []}

plan = ROOT / "results" / "experiment_plan"
if plan.is_dir():
    for p in sorted(plan.glob("ablation_*ubuntu*.csv")):
        out["csv"].append(str(p.relative_to(ROOT)).replace("\\\\", "/"))
    for p in sorted(plan.glob("optimizer_baselines_*ubuntu*")):
        if p.is_dir():
            for f in sorted(p.rglob("*")):
                if f.is_file():
                    out["opt_files"].append(str(f.relative_to(ROOT)).replace("\\\\", "/"))
    st = plan / "ubuntu_queue_status.json"
    if st.is_file():
        out["status"].append(str(st.relative_to(ROOT)).replace("\\\\", "/"))

logs = ROOT / "logs"
if logs.is_dir():
    for pat in ("ubuntu_queue_*.txt", "watchdog_ubuntu_*.txt", "ubuntu_experiment_queue*.txt"):
        for p in sorted(logs.glob(pat)):
            out["logs"].append(str(p.relative_to(ROOT)).replace("\\\\", "/"))
    ps = logs / "parallel_seeds"
    if ps.is_dir():
        for p in sorted(ps.glob("seed43_*.txt")) + sorted(ps.glob("seed44_*.txt")):
            out["logs"].append(str(p.relative_to(ROOT)).replace("\\\\", "/"))

print(json.dumps(out))
""" % REMOTE
    # write temp script remotely to avoid quoting hell
    sftp = client.open_sftp()
    remote_tmp = "/tmp/gsm_list_pull_artifacts.py"
    with sftp.file(remote_tmp, "w") as f:
        f.write(py)
    sftp.close()
    code, out = _run(client, f"/usr/bin/python3 {remote_tmp}", timeout=60)
    if code != 0 or not out.strip():
        raise RuntimeError(f"remote list failed: exit={code} out={out[:500]}")
    return json.loads(out.strip().splitlines()[-1])


def _sftp_get(sftp: paramiko.SFTPClient, remote_abs: str, local: Path) -> Dict[str, Any]:
    local.parent.mkdir(parents=True, exist_ok=True)
    tmp = local.with_suffix(local.suffix + ".part")
    if tmp.exists():
        tmp.unlink()
    sftp.get(remote_abs, str(tmp))
    if local.exists():
        local.unlink()
    tmp.replace(local)
    st = local.stat()
    return {"rel": str(local.relative_to(ROOT)).replace("\\", "/"), "bytes": st.st_size}


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    manifest_dir = ROOT / "results" / "experiment_plan" / f"ubuntu_pull_{stamp}"
    manifest_dir.mkdir(parents=True, exist_ok=True)

    print(f"Connecting {USER}@{HOST} ...", flush=True)
    client = _connect()
    print("Listing remote artifacts ...", flush=True)
    listing = _remote_list(client)
    print(
        json.dumps({k: len(v) for k, v in listing.items()}, ensure_ascii=False),
        flush=True,
    )

    all_rels: List[str] = []
    for key in ("csv", "opt_files", "logs", "status"):
        all_rels.extend(listing.get(key) or [])
    # unique preserve order
    seen = set()
    rels = []
    for r in all_rels:
        r = r.replace("\\", "/")
        if r not in seen:
            seen.add(r)
            rels.append(r)

    sftp = client.open_sftp()
    pulled: List[Dict[str, Any]] = []
    failed: List[Dict[str, str]] = []
    for i, rel in enumerate(rels, 1):
        remote_abs = f"{REMOTE}/{rel}"
        local = ROOT / rel
        try:
            info = _sftp_get(sftp, remote_abs, local)
            pulled.append(info)
            print(f"[{i}/{len(rels)}] OK {info['rel']} ({info['bytes']} B)", flush=True)
        except Exception as exc:
            failed.append({"rel": rel, "error": str(exc)})
            print(f"[{i}/{len(rels)}] FAIL {rel}: {exc}", flush=True)

    sftp.close()
    client.close()

    # Also mirror into latest collected_bundle if present
    bundles = sorted((ROOT / "results" / "experiment_plan").glob("collected_bundle_*"))
    mirrored = 0
    if bundles:
        bundle = bundles[-1]
        mirror_root = bundle / "ubuntu_pulled"
        for info in pulled:
            src = ROOT / info["rel"]
            dst = mirror_root / info["rel"]
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(src.read_bytes())
            mirrored += 1
        print(f"Mirrored {mirrored} files into {mirror_root}", flush=True)

    manifest = {
        "pulled_at": datetime.now().isoformat(sep=" ", timespec="seconds"),
        "host": HOST,
        "remote": REMOTE,
        "counts": {k: len(v) for k, v in listing.items()},
        "pulled_n": len(pulled),
        "failed_n": len(failed),
        "pulled": pulled,
        "failed": failed,
        "listing": listing,
    }
    (manifest_dir / "pull_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    # human summary
    lines = [
        f"# Ubuntu pull {stamp}",
        "",
        f"- Host: `{HOST}`",
        f"- Pulled: **{len(pulled)}** files",
        f"- Failed: **{len(failed)}**",
        "",
        "## Counts",
        f"- CSV: {len(listing.get('csv') or [])}",
        f"- Optimizer files: {len(listing.get('opt_files') or [])}",
        f"- Logs: {len(listing.get('logs') or [])}",
        f"- Status: {len(listing.get('status') or [])}",
        "",
    ]
    if failed:
        lines.append("## Failed")
        for f in failed:
            lines.append(f"- `{f['rel']}`: {f['error']}")
        lines.append("")
    (manifest_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")

    print(json.dumps({k: manifest[k] for k in ("pulled_n", "failed_n", "counts")}, indent=2))
    print(f"Manifest: {manifest_dir}")
    return 0 if not failed else 2


if __name__ == "__main__":
    raise SystemExit(main())
