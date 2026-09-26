#!/usr/bin/env python3
"""SAH must share GSM's external-MC path: tiered 10/100, no bootstrap fill."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List
from unittest import mock

from multi_agent.config_loader import reload_config
from experiments.single_agent_hermes.agent import (
    FIRST_SUCCESS_DEFINITION,
    SingleAgentHermes,
)


def test_tiered_n_runs_matches_gsm():
    reload_config()
    agent = SingleAgentHermes(
        mc_root=Path("."),
        matlab_script=Path("dummy.m"),
        n_runs=100,
        use_llm=False,
    )
    assert agent._n_runs_for_split("dev") == 10
    assert agent._n_runs_for_split("test") == 100
    assert FIRST_SUCCESS_DEFINITION == "final_dev_feasible"


def test_eval_seeds_uses_and_path_without_bootstrap(tmp_path):
    captured: List[Dict[str, Any]] = []

    def fake_and(*args, **kwargs):
        captured.append(dict(kwargs))
        seeds = list(kwargs.get("seeds") or [])
        return {
            "mc_calls": len(seeds),
            "bootstrap_fill": kwargs.get("bootstrap_fill"),
            "seed_results": [
                {
                    "success": True,
                    "feasible": True,
                    "sim_time_sec": 0.0,
                    "mc_seed": seed,
                    "split": kwargs.get("split"),
                }
                for seed in seeds
            ],
        }

    agent = SingleAgentHermes(
        mc_root=tmp_path,
        matlab_script=tmp_path / "dummy.m",
        n_runs=100,
        use_llm=False,
    )
    with mock.patch(
        "experiments.single_agent_hermes.agent.run_mc_and_across_seeds",
        side_effect=fake_and,
    ):
        agent._eval_seeds({"w1": 40.0}, "T1", [4201, 4202, 4203], "dev")
        agent._eval_seeds({"w1": 40.0}, "T1", [4204, 4205, 4206], "test")

    assert len(captured) == 2
    assert captured[0]["n_runs"] == 10
    assert captured[0]["bootstrap_fill"] is False
    assert captured[0]["early_abort"] is False
    assert captured[1]["n_runs"] == 100
    assert captured[1]["bootstrap_fill"] is False
    assert agent._last_mc_calls == 3
