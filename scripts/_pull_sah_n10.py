#!/usr/bin/env python3
"""Pull SAH n10 results from Ubuntu. Does not touch optimizer_equal_mc_75_sah."""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime
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
REL_DIR = "results/experiment_plan/optimizer_equal_mc_75_sah_n10"
OLD_SAH = ROOT / "results" / "experiment_plan" / "optimizer_equal_mc_75_sah"

PASS = os.environ.get("GSM_SSH_PASS", "")
if not PASS:
    pf = ROOT / "scripts" / ".gsm_ssh_pass"
    if pf.is_file():
        PASS = pf.read_text(encoding="utf-8").strip()
if not PASS:
    raise SystemExit("GSM_SSH_PASS or scripts/.gsm_ssh_pass required")


def _connect() -> tuple[paramiko.SSHClient, str]:
    last = None
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
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
            return client, host
        except Exception as exc:
            last = exc
            client = paramiko.SSHClient()
            client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    raise SystemExit(f"SSH_FAIL {last}")


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    if not OLD_SAH.is_dir():
        print(f"[warn] old SAH dir missing: {OLD_SAH}", flush=True)

    client, host = _connect()
    cmd = f"""
python3 - <<'PY'
from pathlib import Path
root = Path({REMOTE!r})
rels = []
n10 = root / {REL_DIR!r}
if not n10.is_dir():
    raise SystemExit('missing ' + str(n10))
for f in sorted(n10.rglob('*')):
    if f.is_file():
        rels.append(str(f.relative_to(root)).replace('\\\\', '/'))
logs = root / 'logs'
for p in sorted(logs.glob('ubuntu_sah_n10_*.txt')):
    rels.append(str(p.relative_to(root)).replace('\\\\', '/'))
print('\\n'.join(rels))
PY
"""
    _, stdout, stderr = client.exec_command(cmd, timeout=60)
    code = stdout.channel.recv_exit_status()
    text = stdout.read().decode("utf-8", "replace")
    err = stderr.read().decode("utf-8", "replace")
    if code != 0:
        print(text)
        print(err[:500], flush=True)
        return 2
    rels = [ln.strip().replace("\\", "/") for ln in text.splitlines() if ln.strip()]
    # never pull the old 100-traj directory
    rels = [r for r in rels if "optimizer_equal_mc_75_sah/" not in r + "/"]
    print(f"remote files: {len(rels)}", flush=True)

    sftp = client.open_sftp()
    pulled = []
    for i, rel in enumerate(rels, 1):
        local = ROOT / rel
        local.parent.mkdir(parents=True, exist_ok=True)
        tmp = local.with_suffix(local.suffix + ".part")
        if tmp.exists():
            tmp.unlink()
        sftp.get(f"{REMOTE}/{rel}", str(tmp))
        tmp.replace(local)
        nbytes = local.stat().st_size
        pulled.append({"rel": rel, "bytes": nbytes})
        print(f"[{i}/{len(rels)}] {rel} ({nbytes} B)", flush=True)
    sftp.close()
    client.close()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    manifest_dir = ROOT / "results" / "experiment_plan" / f"ubuntu_pull_sah_n10_{stamp}"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "pulled_at": datetime.now().isoformat(sep=" ", timespec="seconds"),
        "host": host,
        "remote": REMOTE,
        "pulled_n": len(pulled),
        "skipped_old_sah": str(OLD_SAH),
        "pulled": pulled,
    }
    (manifest_dir / "pull_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"pulled_n": len(pulled), "manifest": str(manifest_dir)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
