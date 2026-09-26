#!/usr/bin/env python3
"""Tests for MATLAB memory-guard helpers."""

from __future__ import annotations

from multi_agent.simulation.memory_guard import (
    looks_like_oom,
    matlab_eval_preamble,
    mc_use_parfor,
)


def test_looks_like_oom():
    assert looks_like_oom("Out of memory. Type HELP MEMORY for more info.")
    assert looks_like_oom("R6025 - pure virtual function call")
    assert not looks_like_oom("hit_rate=100 SEP=3.0")


def test_preamble_limits_threads_and_parpool():
    pre = matlab_eval_preamble()
    assert "maxNumCompThreads" in pre
    assert "delete(gcp" in pre
    assert "DefaultFigureVisible" in pre


def test_mc_use_parfor_env(monkeypatch):
    monkeypatch.setenv("GSM_MC_USE_PARFOR", "0")
    assert mc_use_parfor(default=True) is False
    monkeypatch.setenv("GSM_MC_USE_PARFOR", "1")
    assert mc_use_parfor(default=False) is True
