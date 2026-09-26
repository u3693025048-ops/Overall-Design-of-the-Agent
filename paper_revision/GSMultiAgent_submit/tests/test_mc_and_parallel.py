#!/usr/bin/env python3
"""Unit tests for parallel 3-seed AND evaluation (no MATLAB required)."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List
from unittest import mock

from multi_agent.simulation.matlab_parallel import (
    resolve_mc_and_workers,
    resolve_matlab_max_parallel,
)
from multi_agent.tools.simulation_tool import run_mc_and_across_seeds


def _fake_result(seed: int, delay: float = 0.05) -> Dict[str, Any]:
    time.sleep(delay)
    return {
        "hit_rate": 95.0,
        "SEP": 3.0,
        "peak_nz": 5.0,
        "PM": 55.0,
        "BW": 40.0,
        "feasible": True,
        "constraint_violations": {},
        "sim_time_sec": delay,
        "success": True,
        "error": None,
        "error_msg": None,
    }


def test_resolve_workers_env(monkeypatch):
    monkeypatch.setenv("GSM_MC_AND_WORKERS", "2")
    monkeypatch.setenv("GSM_MATLAB_MAX_PARALLEL", "4")
    assert resolve_mc_and_workers(3) == 2
    assert resolve_matlab_max_parallel() == 4


def test_and_parallel_preserves_order_and_aggregates(monkeypatch, tmp_path):
    monkeypatch.setenv("GSM_MC_AND_WORKERS", "3")
    monkeypatch.setenv("GSM_MATLAB_MAX_PARALLEL", "3")
    monkeypatch.setenv("GSM_MC_AND_BOOTSTRAP", "0")
    monkeypatch.setenv("GSM_MC_EVAL_CACHE", "0")

    calls: List[int] = []

    def fake_run(script, data_path, **kwargs):
        # data_path ends with seed_XXXX_dev.mat
        name = str(data_path).replace("\\", "/").split("/")[-1]
        seed = int(name.split("_")[1])
        calls.append(seed)
        return _fake_result(seed, delay=0.08)

    script = tmp_path / "dummy.m"
    script.write_text("% dummy", encoding="utf-8")
    root = tmp_path / "mc"
    for seed in (4201, 4202, 4203):
        d = root / "T1"
        d.mkdir(parents=True, exist_ok=True)
        (d / f"seed_{seed}_dev.mat").write_bytes(b"x")

    with mock.patch(
        "multi_agent.tools.simulation_tool.run_mc_with_external_data",
        side_effect=fake_run,
    ):
        wall0 = time.perf_counter()
        out = run_mc_and_across_seeds(
            str(script),
            mc_root=str(root),
            scenario="T1",
            seeds=[4201, 4202, 4203],
            split="dev",
            n_runs=1,
        )
        wall = time.perf_counter() - wall0

    assert out["success"] is True
    assert out["feasible"] is True
    assert out["mc_and_workers"] == 3
    assert [r["mc_seed"] for r in out["seed_results"]] == [4201, 4202, 4203]
    # Parallel should finish well under serial 3*0.08
    assert wall < 0.20
    assert set(calls) == {4201, 4202, 4203}


def test_and_serial_when_workers_one(monkeypatch, tmp_path):
    monkeypatch.setenv("GSM_MC_AND_WORKERS", "1")
    monkeypatch.setenv("GSM_MC_AND_BOOTSTRAP", "0")
    monkeypatch.setenv("GSM_MC_EVAL_CACHE", "0")

    def fake_run(script, data_path, **kwargs):
        name = str(data_path).replace("\\", "/").split("/")[-1]
        seed = int(name.split("_")[1])
        return _fake_result(seed, delay=0.02)

    script = tmp_path / "dummy.m"
    script.write_text("% dummy", encoding="utf-8")
    root = tmp_path / "mc"
    for seed in (1, 2):
        d = root / "T1"
        d.mkdir(parents=True, exist_ok=True)
        (d / f"seed_{seed}_dev.mat").write_bytes(b"x")

    with mock.patch(
        "multi_agent.tools.simulation_tool.run_mc_with_external_data",
        side_effect=fake_run,
    ):
        out = run_mc_and_across_seeds(
            str(script),
            mc_root=str(root),
            scenario="T1",
            seeds=[1, 2],
            split="dev",
            n_runs=1,
        )
    assert out["mc_and_workers"] == 1
    assert out["feasible"] is True
