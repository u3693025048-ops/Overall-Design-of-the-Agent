#!/usr/bin/env python3
"""Unit tests for experiment watchdog helpers (no long sleep / no process kill)."""

from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_MOD_PATH = ROOT / "scripts" / "watchdog_experiment.py"
_spec = importlib.util.spec_from_file_location("watchdog_experiment", _MOD_PATH)
wd = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
sys.modules["watchdog_experiment"] = wd
_spec.loader.exec_module(wd)


def test_match_any():
    assert wd._match_any(
        "python experiments/experiment_plan_runner.py run-gsm-suite",
        ["experiment_plan_runner"],
    )
    assert not wd._match_any("notepad.exe", ["experiment_plan_runner"])


def test_log_fingerprint(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("hello", encoding="utf-8")
    size, mtime = wd.log_progress_fingerprint(f)
    assert size == 5
    assert mtime > 0


def test_looks_successful(tmp_path):
    log = tmp_path / "seed42.txt"
    log.write_text("noise\n[parallel-seeds] merged 4 rows\n", encoding="utf-8")
    assert wd.looks_successful(log, "") is True
    bad = tmp_path / "bad.txt"
    bad.write_text("still running Layer 3\n", encoding="utf-8")
    assert wd.looks_successful(bad, "") is False
    csv = tmp_path / "out.csv"
    csv.write_text("a,b\n1,2\n", encoding="utf-8")
    assert wd.looks_successful(None, str(csv)) is True


def test_parse_restart_cmd():
    cmd = wd.parse_restart_cmd(
        'python scripts/run_utf8.py --log "logs/a.txt" -- python x.py'
    )
    assert cmd[0] == "python"
    assert "run_utf8.py" in cmd[1]


def test_progress_fingerprint_prefers_newer(tmp_path, monkeypatch):
    monkeypatch.setattr(wd, "ROOT", tmp_path)
    a = tmp_path / "a.txt"
    bdir = tmp_path / "logs" / "parallel_seeds"
    bdir.mkdir(parents=True)
    b = bdir / "seed43_1.txt"
    a.write_text("aaaa", encoding="utf-8")
    time.sleep(0.05)
    b.write_text("bbbbbb", encoding="utf-8")
    size, mtime, path = wd.progress_fingerprint(str(a), "logs/parallel_seeds/seed*.txt")
    assert path == b
    assert size == 6
    assert mtime > 0


def test_list_processes_nonempty_on_posix():
    procs = wd._list_processes()
    assert isinstance(procs, list)
    # At least the current interpreter should appear somehow on a live OS.
    assert any(int(p.get("pid") or 0) > 0 for p in procs)