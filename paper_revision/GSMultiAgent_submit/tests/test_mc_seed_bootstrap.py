#!/usr/bin/env python3
"""Tests for single-seed MATLAB + bootstrap fill of remaining AND seeds."""

from __future__ import annotations

from typing import Any, Dict, List
from unittest import mock

from multi_agent.simulation.mc_seed_bootstrap import bootstrap_seed_result
from multi_agent.tools.simulation_tool import run_mc_and_across_seeds


def _fake_result(seed: int) -> Dict[str, Any]:
    misses = [2.0 + 0.1 * i for i in range(20)]
    peaks = [4.0 + 0.05 * i for i in range(20)]
    return {
        "hit_rate": 100.0,
        "SEP": 3.0,
        "peak_nz": 5.0,
        "PM": 55.0,
        "BW": 40.0,
        "feasible": True,
        "constraint_violations": {},
        "sim_time_sec": 0.01,
        "success": True,
        "error": None,
        "error_msg": None,
        "miss_distances": misses,
        "nz_peaks": peaks,
    }


def test_bootstrap_seed_result_deterministic():
    base = _fake_result(4201)
    a = bootstrap_seed_result(
        base, target_seed=4202, source_seed=4201, split="dev", n_runs=20
    )
    b = bootstrap_seed_result(
        base, target_seed=4202, source_seed=4201, split="dev", n_runs=20
    )
    assert a["synthetic"] is True
    assert a["mc_seed"] == 4202
    assert a["bootstrap_from"] == 4201
    assert a["success"] is True
    assert a["miss_distances"] == b["miss_distances"]
    assert a["hit_rate"] == b["hit_rate"]


def test_and_bootstrap_calls_matlab_once(monkeypatch, tmp_path):
    monkeypatch.setenv("GSM_MC_AND_BOOTSTRAP", "1")
    monkeypatch.setenv("GSM_MC_EVAL_CACHE", "0")
    calls: List[int] = []

    def fake_run(script, data_path, **kwargs):
        name = str(data_path).replace("\\", "/").split("/")[-1]
        seed = int(name.split("_")[1])
        calls.append(seed)
        return _fake_result(seed)

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
        out = run_mc_and_across_seeds(
            str(script),
            mc_root=str(root),
            scenario="T1",
            seeds=[4201, 4202, 4203],
            split="dev",
            n_runs=20,
        )

    assert calls == [4201]
    assert out["bootstrap_fill"] is True
    assert out["mc_calls"] == 1
    assert [r["mc_seed"] for r in out["seed_results"]] == [4201, 4202, 4203]
    assert out["seed_results"][0]["synthetic"] is False
    assert out["seed_results"][1]["synthetic"] is True
    assert out["seed_results"][2]["synthetic"] is True
    assert out["success"] is True


def test_and_full_when_bootstrap_disabled(monkeypatch, tmp_path):
    monkeypatch.setenv("GSM_MC_AND_BOOTSTRAP", "0")
    monkeypatch.setenv("GSM_MC_AND_WORKERS", "1")
    monkeypatch.setenv("GSM_MC_EVAL_CACHE", "0")
    calls: List[int] = []

    def fake_run(script, data_path, **kwargs):
        name = str(data_path).replace("\\", "/").split("/")[-1]
        seed = int(name.split("_")[1])
        calls.append(seed)
        return _fake_result(seed)

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
        out = run_mc_and_across_seeds(
            str(script),
            mc_root=str(root),
            scenario="T1",
            seeds=[4201, 4202, 4203],
            split="dev",
            n_runs=20,
        )

    assert set(calls) == {4201, 4202, 4203}
    assert out.get("bootstrap_fill") is False
    assert out["mc_calls"] == 3


def test_bootstrap_fill_false_overrides_env(monkeypatch, tmp_path):
    monkeypatch.setenv("GSM_MC_AND_BOOTSTRAP", "1")
    monkeypatch.setenv("GSM_MC_AND_WORKERS", "1")
    monkeypatch.setenv("GSM_MC_EVAL_CACHE", "0")
    calls: List[int] = []

    def fake_run(script, data_path, **kwargs):
        name = str(data_path).replace("\\", "/").split("/")[-1]
        seed = int(name.split("_")[1])
        calls.append(seed)
        return _fake_result(seed)

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
        out = run_mc_and_across_seeds(
            str(script),
            mc_root=str(root),
            scenario="T1",
            seeds=[4201, 4202, 4203],
            split="dev",
            n_runs=20,
            bootstrap_fill=False,
        )

    assert set(calls) == {4201, 4202, 4203}
    assert out["bootstrap_fill"] is False
    assert out["mc_calls"] == 3
