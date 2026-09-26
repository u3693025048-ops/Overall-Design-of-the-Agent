#!/usr/bin/env python3
"""Pipeline warmup tests (no PE writes, MC prefault only)."""

from __future__ import annotations

from pathlib import Path

from multi_agent.simulation import pipeline_warmup as pw


def test_warmup_mc_files_prefault(tmp_path, monkeypatch):
    monkeypatch.setenv("GSM_PIPELINE_WARMUP", "1")
    root = tmp_path / "mc"
    scen = root / "T2"
    scen.mkdir(parents=True)
    for seed in (4201, 4202, 4203):
        (scen / f"seed_{seed}_dev.mat").write_bytes(b"0" * 4096)
        (scen / f"seed_{seed}_test.mat").write_bytes(b"1" * 2048)

    report = pw._warmup_mc_files("T2", 42, root)
    assert report["files_found"] >= 6
    assert report["bytes_read"] >= 6 * 2048


def test_warmup_scenario_disabled(monkeypatch):
    monkeypatch.setenv("GSM_PIPELINE_WARMUP", "0")
    report = pw.warmup_scenario("T1", master_seed=42, include_matlab=False, include_rag=False)
    assert report.get("enabled") is False


def test_warmup_does_not_create_pe(tmp_path, monkeypatch):
    monkeypatch.setenv("GSM_PIPELINE_WARMUP", "1")
    monkeypatch.setenv("GSM_MATLAB_JOB_POOL", "0")  # skip MATLAB in unit test
    pe_json = tmp_path / "parameter_experience_memory.json"
    pe_base = tmp_path / "parameter_experience_base"
    monkeypatch.setenv("GSM_PE_JSON", str(pe_json))
    monkeypatch.setenv("GSM_PE_BASE", str(pe_base))
    before = list(tmp_path.rglob("*"))
    pw.warmup_scenario("T1", master_seed=42, include_matlab=False, include_rag=False)
    assert not pe_json.exists()
    # Should not create PE base either
    assert not pe_base.exists() or not any(pe_base.rglob("*"))


def test_async_warmup_wait(monkeypatch):
    monkeypatch.setenv("GSM_PIPELINE_WARMUP", "1")
    monkeypatch.setenv("GSM_MATLAB_JOB_POOL", "0")
    fut = pw.start_warmup_async("T3", master_seed=42)
    assert fut is not None
    report = pw.wait_warmup("T3", master_seed=42, timeout_sec=60)
    assert report is not None
    assert report.get("scenario") == "T3"
