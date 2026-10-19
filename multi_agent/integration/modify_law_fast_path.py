"""Direct MODIFY_LAW pipeline for T4 — skip Hermes orchestration/delegate."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional

from multi_agent.integration.t4_low_risk import is_t4_mission

logger = logging.getLogger(__name__)


def should_use_modify_law_t4_fast_path(
    *,
    task_mode: str,
    iteration: int,
    mission_conditions_str: str,
    task_prompt: str,
    enabled: bool = True,
) -> bool:
    # Correction-pipeline ablations must exercise Hermes tool gating; the fast
    # path hard-codes generate→syntax→sim→judge and collapses C0/C1/C2.
    if not enabled:
        return False
    try:
        from multi_agent.config_loader import get_config

        ab = get_config().ablation
        if (not ab.syntax_check) or (not ab.guidance_compat_verification) or (
            not ab.reflection_agent
        ):
            return False
    except Exception:
        pass
    return (
        str(task_mode).upper() == "MODIFY_LAW"
        and int(iteration) == 1
        and is_t4_mission(mission_conditions_str, task_prompt)
    )


async def run_modify_law_t4_fast_path(
    *,
    mission_conditions_str: str,
    task_prompt: str,
    autopilot_params: Optional[Dict[str, float]] = None,
    simulator: Any = None,
    nmc: int = 25,
) -> Dict[str, Any]:
    """generate_matlab → static syntax_check → run_simulation → judge_requirements."""
    from multi_agent.tools.simulation_tool import GenerateMATLABTool, RunSimulationTool
    from multi_agent.tools.syntax_check_tool import SyntaxCheckMATLABTool
    from multi_agent.tools.judge_requirements_tool import JudgeRequirementsTool
    # P2-5 兵部：工具返回类型统一层（此处四个工具形态并不一致）
    from multi_agent.tools import tool_result as _tr

    gen = GenerateMATLABTool()
    syn = SyntaxCheckMATLABTool()
    run = RunSimulationTool()
    judge = JudgeRequirementsTool()
    if simulator is not None:
        run.set_simulator(simulator)

    ap = dict(autopilot_params or {})
    gen_result = await gen.execute(
        mode="MODIFY_LAW",
        script_name="guidance_T4_APN",
        task_description=(task_prompt or "")[:2000],
        mission_conditions=mission_conditions_str,
        autopilot_params=ap,
        nmc=nmc,
    )
    gen_result = _tr.as_dict(gen_result)
    if gen_result.get("status") != "success":
        return {
            "ok": False,
            "script_path": "",
            "metrics": {},
            "satisfied": False,
            "summary": f"generate_matlab failed: {gen_result.get('message', gen_result)}",
        }

    script_path = gen_result.get("script_path") or ""
    syn_raw = await syn.execute(
        script_path=script_path,
        auto_fix=True,
        llm_fix=False,
        max_fix_attempts=0,
    )
    syn_result = _tr.as_dict(syn_raw, default={"valid": False})
    script_path = syn_result.get("fixed_path") or script_path
    if not syn_result.get("valid"):
        return {
            "ok": False,
            "script_path": script_path,
            "metrics": {},
            "satisfied": False,
            "summary": f"syntax_check failed: {syn_result.get('issues', [])[:2]}",
        }

    sim_result = await run.execute(
        script_path=script_path,
        mission_conditions=mission_conditions_str,
        nmc=nmc,
    )
    sim_result = _tr.as_dict(sim_result)
    if sim_result.get("status") != "success":
        return {
            "ok": False,
            "script_path": script_path,
            "metrics": {},
            "satisfied": False,
            "summary": f"run_simulation failed: {sim_result.get('message', sim_result)}",
        }

    metrics = dict(getattr(run, "last_metrics", None) or sim_result.get("metrics") or {})
    judge_raw = await judge.execute(
        metrics=metrics,
        script_path=script_path,
        task_prompt=task_prompt,
    )
    judge_result = _tr.as_dict(judge_raw, default={})
    satisfied = bool(judge_result.get("satisfied"))

    logger.info(
        "[ModifyLawFastPath] script=%s satisfied=%s metrics_keys=%s",
        script_path,
        satisfied,
        list(metrics.keys()),
    )
    return {
        "ok": True,
        "script_path": script_path,
        "metrics": metrics,
        "satisfied": satisfied,
        "judge": judge_result,
        "summary": (
            f"T4 MODIFY_LAW fast path: hit={metrics.get('hit_rate', 0):.1f}% "
            f"SEP={metrics.get('SEP', metrics.get('miss_distance', 99)):.2f}m "
            f"satisfied={satisfied}"
        ),
    }
