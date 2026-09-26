#!/usr/bin/env python3
"""Tests for bootstrap export / remote fill helpers."""

from __future__ import annotations

from pathlib import Path

from multi_agent.simulation.bootstrap_export import (
    process_export_file,
    write_bootstrap_export,
)


def test_write_and_process_export(tmp_path, monkeypatch):
    monkeypatch.setenv("GSM_BOOTSTRAP_EXPORT", "1")
    monkeypatch.setenv("GSM_BOOTSTRAP_EXPORT_DIR", str(tmp_path / "export"))

    base = {
        "success": True,
        "feasible": True,
        "hit_rate": 100.0,
        "SEP": 3.0,
        "peak_nz": 5.0,
        "PM": 55.0,
        "BW": 40.0,
        "miss_distances": [2.0, 3.0, 4.0, 2.5],
        "nz_peaks": [4.0, 5.0, 4.5],
        "constraint_violations": {},
        "sim_time_sec": 1.0,
        "mc_seed": 4201,
    }
    path = write_bootstrap_export(
        scenario="T1",
        split="dev",
        source_seed=4201,
        target_seeds=[4202, 4203],
        n_runs=4,
        kill_radius=10.0,
        base_result=base,
    )
    assert path is not None and path.is_file()

    out_dir = tmp_path / "filled"
    filled = process_export_file(path, out_dir=out_dir)
    assert filled.is_file()
    text = filled.read_text(encoding="utf-8")
    assert "gsm_bootstrap_filled_v1" in text
    assert "4202" in text
    assert (out_dir / "filled_summary.csv").is_file()
