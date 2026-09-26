#!/usr/bin/env python3
"""Tests for MC cache, tiered-eval config, and external-MC parfor patch."""

from __future__ import annotations

from pathlib import Path
from unittest import mock

from multi_agent.config_loader import get_config, reload_config
from multi_agent.simulation.mc_eval_cache import make_cache_key
from multi_agent.tools.simulation_tool import (
    _patch_external_mc_script,
    run_mc_and_across_seeds,
)


def test_cache_key_stable():
    a = make_cache_key(
        matlab_script_path="a.m",
        mc_root="data/mc",
        scenario="T1",
        seeds=[1, 2],
        split="dev",
        n_runs=25,
        kill_radius=10.0,
        dp_params={"w1": 40.0, "N_pn": 4.0},
    )
    b = make_cache_key(
        matlab_script_path="a.m",
        mc_root="data/mc",
        scenario="T1",
        seeds=[1, 2],
        split="dev",
        n_runs=25,
        kill_radius=10.0,
        dp_params={"N_pn": 4.0, "w1": 40.0},
    )
    assert a == b


def test_tiered_config_defaults():
    reload_config()
    sim = get_config().simulation
    assert bool(sim.mc_tiered_eval) is True
    assert int(sim.mc_search_n_runs) == 10
    assert int(sim.mc_final_n_runs) == 100


def test_external_mc_loop_serial_by_default(monkeypatch):
    monkeypatch.setenv("GSM_MC_USE_PARFOR", "0")
    src = Path("knowledge_base/matlab/guidance/monte_carlo_single.m")
    content = src.read_text(encoding="utf-8", errors="ignore")
    patched, _ = _patch_external_mc_script(
        content, Path("data/mc/T1/seed_4201_dev.mat"), 10, 10.0
    )
    assert "for k=1:N_MC" in patched
    assert "parfor k=1:N_MC" not in patched
    assert "miss_v=nan(N_MC,1)" in patched


def test_external_mc_loop_parfor_when_enabled(monkeypatch):
    monkeypatch.setenv("GSM_MC_USE_PARFOR", "1")
    src = Path("knowledge_base/matlab/guidance/monte_carlo_single.m")
    content = src.read_text(encoding="utf-8", errors="ignore")
    patched, _ = _patch_external_mc_script(
        content, Path("data/mc/T1/seed_4201_dev.mat"), 10, 10.0
    )
    assert "parfor k=1:N_MC" in patched
    assert "miss_v=nan(N_MC,1)" in patched


def test_and_uses_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("GSM_MC_EVAL_CACHE", "1")
    monkeypatch.setenv("GSM_MC_AND_WORKERS", "1")
    monkeypatch.setenv("GSM_MC_AND_EARLY_ABORT", "0")
    monkeypatch.setenv("GSM_MC_AND_BOOTSTRAP", "0")

    calls = {"n": 0}

    def fake_run(script, data_path, **kwargs):
        calls["n"] += 1
        return {
            "hit_rate": 95.0,
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
        }

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
        a = run_mc_and_across_seeds(
            str(script),
            mc_root=str(root),
            scenario="T1",
            seeds=[1, 2],
            split="dev",
            n_runs=5,
            dp_params={"w1": 1.0},
            early_abort=False,
        )
        b = run_mc_and_across_seeds(
            str(script),
            mc_root=str(root),
            scenario="T1",
            seeds=[1, 2],
            split="dev",
            n_runs=5,
            dp_params={"w1": 1.0},
            early_abort=False,
        )
    assert a["feasible"] and b.get("from_cache") is True
    assert calls["n"] == 2  # only first AND evaluated both seeds
