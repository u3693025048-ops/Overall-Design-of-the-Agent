#!/usr/bin/env python3
"""Offline smoke test (no MATLAB / no LLM). Run from repo root:

    python scripts/smoke_offline.py
"""

from __future__ import annotations

import importlib
import py_compile
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

T4_MC = "RUN_CASE='T'; SUB_IDX=4;"


def main() -> int:
    errors: list[str] = []
    checks: list[str] = []

    from multi_agent.config_loader import get_config

    cfg = get_config()
    sim = cfg.simulation
    if cfg.workflow.max_iterations < 1:
        errors.append(f"workflow.max_iterations={cfg.workflow.max_iterations}, want >=1")
    else:
        checks.append(f"workflow.max_iterations={cfg.workflow.max_iterations}")
    if int(cfg.workflow.max_iterations) != 5:
        errors.append(f"workflow.max_iterations={cfg.workflow.max_iterations}, paper wants 5")
    else:
        checks.append("workflow.max_iterations=5 (paper)")
    backend = str(getattr(cfg.matlab_rl_optimizer, "optimizer_backend", ""))
    if backend != "hybrid_ppo":
        errors.append(f"optimizer_backend={backend!r}, want hybrid_ppo")
    else:
        checks.append("optimizer_backend=hybrid_ppo")
    nmc = int(getattr(cfg.matlab_rl_optimizer, "nmc_per_eval", 0) or 0)
    if nmc < 100:
        errors.append(f"nmc_per_eval={nmc}, paper wants 100")
    else:
        checks.append(f"nmc_per_eval={nmc}")
    compat = cfg.ablation.guidance_compat_verification
    compat_on = bool(getattr(compat, "enabled", compat))
    if not compat_on:
        errors.append("guidance_compat_verification disabled")
    else:
        checks.append("guidance_compat_verification enabled")
    if sim.matlab_timeout_sec < 120:
        errors.append(f"matlab_timeout_sec={sim.matlab_timeout_sec}, want >=120")
    else:
        checks.append(f"matlab_timeout_sec={sim.matlab_timeout_sec}")
    if not sim.physics_bounds_skip_deterministic_t4_apn:
        errors.append("physics_bounds_skip_deterministic_t4_apn=false")
    else:
        checks.append("physics_bounds_skip_deterministic_t4_apn=true")

    from experiments.optimizer_baselines_43 import (
        DEFAULT_N_RUNS,
        dev_seeds_for_master,
        test_seeds_for_master,
    )

    if DEFAULT_N_RUNS != 100:
        errors.append(f"DEFAULT_N_RUNS={DEFAULT_N_RUNS}, want 100")
    else:
        checks.append("DEFAULT_N_RUNS=100")
    if dev_seeds_for_master(42) != [4201, 4202, 4203]:
        errors.append(f"dev_seeds_for_master(42)={dev_seeds_for_master(42)}")
    else:
        checks.append("dev seeds 42→4201..4203")
    if test_seeds_for_master(42) != [4204, 4205, 4206]:
        errors.append(f"test_seeds_for_master(42)={test_seeds_for_master(42)}")
    else:
        checks.append("test seeds 42→4204..4206")
    for sid in (4201, 4202, 4203, 4204, 4205, 4206):
        split = "dev" if sid % 10 <= 3 else "test"
        p = ROOT / "data" / "mc" / "T1" / f"seed_{sid}_{split}.mat"
        if not p.is_file():
            errors.append(f"missing MC data {p.relative_to(ROOT)}")
    else:
        checks.append("T1 protocol MC mats present (4201-4206)")

    from multi_agent.tools.simulation_tool import run_mc_and_across_seeds

    checks.append("run_mc_and_across_seeds importable")
    _ = run_mc_and_across_seeds  # silence lint

    from multi_agent.simulation.sim_timeout import (
        get_matlab_timeout_sec,
        get_physics_bounds_timeout_sec,
    )

    t1, t25 = get_matlab_timeout_sec(1), get_matlab_timeout_sec(25)
    if t25 <= t1 or t1 < 120:
        errors.append(f"timeout scale bad: nmc1={t1} nmc25={t25}")
    else:
        checks.append(f"timeout nmc1={t1}s nmc25={t25}s pb={get_physics_bounds_timeout_sec()}s")

    from multi_agent.integration.t4_low_risk import (
        apply_t4_deterministic_gf_if_enabled,
        should_skip_physics_bounds_probe,
    )

    template = (ROOT / "knowledge_base/matlab/guidance/monte_carlo_single.m").read_text(
        encoding="utf-8"
    )
    script, applied = apply_t4_deterministic_gf_if_enabled(template, T4_MC, "T4 smoke")
    if not applied:
        errors.append("apply_t4_deterministic_gf_if_enabled did not apply")
    elif "T4 minimal APN" not in script or "tanh" not in script:
        errors.append("minimal APN markers missing in script")
    else:
        checks.append("T4 deterministic APN injection")
    if not should_skip_physics_bounds_probe(script, T4_MC, "T4"):
        errors.append("should_skip_physics_bounds_probe=false for det APN")
    else:
        checks.append("skip physics bounds for deterministic APN")

    importlib.import_module("multi_agent.async.task_queue")
    checks.append("importlib multi_agent.async.task_queue")

    for rel in (
        "multi_agent/simulation/sim_timeout.py",
        "multi_agent/tools/simulation_tool.py",
        "multi_agent/integration/layer2_script_gate.py",
        "multi_agent/integration/model_generation_agent.py",
        "multi_agent/integration/t4_low_risk.py",
        "multi_agent/rl/matlab_rl_optimizer.py",
        "multi_agent/rl/metric_utils.py",
        "multi_agent/rl/metric_constraints.py",
        "multi_agent/integration/judgment_agent.py",
        "multi_agent/tools/judge_requirements_tool.py",
        "cli_agent.py",
    ):
        py_compile.compile(str(ROOT / rel), doraise=True)
    checks.append("py_compile key modules")

    # ── PeakNy gate (mean ≤ 20g) ───────────────────────────────────────────
    from multi_agent.tools.judge_requirements_tool import _rule_check
    from multi_agent.rl.metric_utils import (
        check_peak_ny,
        default_peak_ny_limit,
        normalize_peak_ny_requirements,
        peak_ny_satisfied,
    )

    mean_lim = default_peak_ny_limit()
    if mean_lim != 20.0:
        errors.append(f"PeakNy limit expected 20, got {mean_lim}")
    else:
        checks.append(f"PeakNy mean limit={mean_lim}g")

    v20_cls7 = {
        "hit_rate": 100.0,
        "SEP": 4.98,
        "peak_ny": 17.86,
        "peak_ny_max": 20.35,
        "pitch_PM": 52.3,
        "pitch_BW": 76.5,
    }
    reqs = normalize_peak_ny_requirements({"peak_ny_max": 20.0})
    ok_peak, _ = check_peak_ny(v20_cls7, reqs)
    if not ok_peak:
        errors.append("v20 CLS-7 metrics should pass PeakNy mean gate")
    else:
        checks.append("v20 CLS-7 PeakNy mean pass")

    ok_full, reasons = _rule_check(
        v20_cls7,
        {
            **reqs,
            "hit_rate_min": 92.0,
            "sep_max": 7.0,
            "pm_min": 45.0,
            "pm_max": 70.0,
            "bw_min": 20.0,
            "bw_max": 85.0,
        },
    )
    if not ok_full:
        errors.append(f"Layer2 judge should pass v20 CLS-7: {reasons}")
    else:
        checks.append("Layer2 judge v20 CLS-7 full 5-metric pass")

    old_fail = {
        "hit_rate": 100.0,
        "SEP": 5.71,
        "peak_ny": 20.91,
        "peak_ny_max": 22.29,
        "pitch_PM": 52.3,
        "pitch_BW": 71.6,
    }
    if peak_ny_satisfied(old_fail, reqs):
        errors.append("Hermes v20 baseline (mean=20.91) must NOT pass PeakNy")
    else:
        checks.append("Hermes v20 baseline correctly rejected (mean over 20g)")

    high_max_ok_mean = {
        "peak_ny": 19.0,
        "peak_ny_max": 25.0,
    }
    if not peak_ny_satisfied(high_max_ok_mean, reqs):
        errors.append("high peak_ny_max must not fail when mean is under limit")
    else:
        checks.append("peak_ny_max over limit ignored for gating")

    rw = cfg.matlab_rl_optimizer.reward_weights
    if float(getattr(rw, "peak_ny_mean_max", 0)) != 20.0:
        errors.append("reward_weights.peak_ny_mean_max != 20")
    else:
        checks.append("config reward_weights peak_ny_mean_max")

    print("=== Offline smoke (no MATLAB) ===")
    for c in checks:
        print(f"  [OK] {c}")
    if errors:
        print("=== FAILURES ===")
        for e in errors:
            print(f"  [FAIL] {e}")
        return 1
    print("=== ALL PASSED ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
