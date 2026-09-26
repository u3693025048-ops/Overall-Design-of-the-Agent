"""
Layer 2 → Layer 3 script gate.

Ensures a Hermes (or fallback) .m script is RL/MATLAB-ready before Layer 3:
  1. validate_rl_script_content (placeholder / endfunction / RL_PARAMS)
  2. regenerate from monte_carlo_single.m when invalid (optional)
  3. syntax_check_matlab (static + auto-fix)
  4. run_simulation smoke test (nmc=5 by default)
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
from typing import Any, Dict, Optional, Tuple

from multi_agent.rl.matlab_rl_optimizer import (
    metrics_look_like_parse_defaults,
    stdout_has_mc_evidence,
)

logger = logging.getLogger(__name__)


def _read_script(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        return fh.read()


def metrics_usable(metrics: Optional[Dict[str, Any]], stdout: str = "") -> bool:
    """True when parsed simulation metrics look like a real MC run."""
    if not metrics:
        return False
    if metrics.get("_parse_incomplete"):
        return False
    if metrics_look_like_parse_defaults(metrics, stdout):
        return False
    try:
        hr = float(metrics.get("hit_rate") or 0.0)
    except (TypeError, ValueError):
        hr = 0.0
    if hr > 0.0:
        return True
    for key in ("SEP", "miss_distance"):
        if key not in metrics:
            continue
        try:
            sep = float(metrics[key])
        except (TypeError, ValueError):
            continue
        if not math.isnan(sep) and not math.isinf(sep) and sep < 1e5:
            return True
    return False


def stdout_smoke_ok(stdout: str, metrics: Dict[str, Any]) -> Tuple[bool, str]:
    """Check smoke-run stdout is not all ERR with zero usable metrics."""
    if not stdout:
        return False, "仿真无 stdout 输出"
    err_n = len(re.findall(r"\[\s*\d+\]\s*ERR:", stdout))
    hit_n = len(re.findall(r"\[\s*\d+\]\s*(?:HIT|MISS)\b", stdout, re.I))
    if hit_n > 0 or re.search(r"命中率\s*\(\s*miss\s*<", stdout, re.I):
        return True, "ok"
    if re.search(r"SEP:\s*\d", stdout, re.I):
        return True, "ok"
    if re.search(r"Hit\s+rate\s*\(\s*miss\s*<", stdout, re.I):
        return True, "ok"
    if err_n > 0 and hit_n == 0:
        return False, f"烟雾测试 {err_n} 次 ERR、无 HIT/MISS 行"
    if metrics_usable(metrics, stdout):
        return True, "ok"
    if stdout_has_mc_evidence(stdout):
        return True, "ok"
    if metrics.get("_parse_incomplete") or metrics_look_like_parse_defaults(metrics, stdout):
        return False, "指标仍为解析器占位默认值（hit=0% SEP=50m），stdout 缺少 MC 结果"
    return False, "烟雾测试未能解析有效指标"


async def ensure_script_ready_for_layer3(
    *,
    script_path: Optional[str],
    mission_conditions_str: str,
    task_prompt: str,
    existing_metrics: Optional[Dict[str, Any]] = None,
    simulator: Any = None,
    smoke_nmc: int = 5,
    enabled: bool = True,
    task_mode: str = "TUNE_PARAMS",
) -> Tuple[bool, str, Dict[str, Any], str]:
    """
    Returns (ok, script_path, metrics, message).

    When ``enabled`` is False, returns the input script unchanged (always ok).
    """
    from multi_agent.config_loader import get_config
    from multi_agent.rl.matlab_rl_optimizer import (
        parse_sim_stdout,
        validate_rl_script_content,
    )
    from multi_agent.tools.simulation_tool import (
        GenerateMATLABTool,
        RunSimulationTool,
    )
    from multi_agent.tools.syntax_check_tool import SyntaxCheckMATLABTool

    cfg = get_config()
    if not enabled:
        sp = script_path or RunSimulationTool.KB_TEMPLATE_PATH
        return True, sp, dict(existing_metrics or {}), "gate disabled"

    smoke_nmc = int(smoke_nmc or getattr(cfg.simulation, "layer2_smoke_nmc", 5))
    smoke_nmc = max(1, min(smoke_nmc, 20))

    gen_tool = GenerateMATLABTool()
    run_tool = RunSimulationTool()
    if simulator is not None:
        run_tool.set_simulator(simulator)
    syntax_tool = SyntaxCheckMATLABTool()

    path = script_path or ""
    content = ""
    if path and os.path.isfile(path):
        content = _read_script(path)

    rl_ok, rl_reason = validate_rl_script_content(content) if content else (False, "脚本不存在")
    need_regen = not path or not os.path.isfile(path) or not rl_ok

    if need_regen:
        logger.info(
            "[Layer2Gate] Regenerating from KB template (reason=%s)", rl_reason
        )
        gen_result = await gen_tool.execute(
            script_name="guidance_gate",
            task_description=(task_prompt or "")[:2000],
            mode="TUNE_PARAMS",
            mission_conditions=mission_conditions_str,
            nmc=smoke_nmc,
            force_kb_template=True,
        )
        if gen_result.get("status") != "success":
            msg = gen_result.get("message", "generate_matlab 失败")
            return False, path, {}, f"KB 模板 generate_matlab 失败: {msg}"
        path = gen_result.get("script_path", "")
        if not path or not os.path.isfile(path):
            return False, "", {}, "generate_matlab 未返回有效 script_path"
        content = _read_script(path)
        rl_ok, rl_reason = validate_rl_script_content(content)
        if not rl_ok:
            return False, path, {}, f"KB 生成后仍不可用: {rl_reason}"

    # syntax_check (static + endfunction fix; no LLM loop for advisory-only issues)
    syntax_raw = await syntax_tool.execute(
        script_path=path, auto_fix=True, llm_fix=False, max_fix_attempts=0,
    )
    try:
        syntax_result = json.loads(syntax_raw)
    except json.JSONDecodeError:
        syntax_result = {"valid": False, "issues": [syntax_raw[:200]]}
    fixed_path = syntax_result.get("fixed_path") or path
    if fixed_path and os.path.isfile(fixed_path):
        path = fixed_path

    content = _read_script(path)
    rl_ok, rl_reason = validate_rl_script_content(content)
    if not rl_ok:
        issues = syntax_result.get("issues") or []
        detail = "; ".join(str(i) for i in issues[:3])
        return False, path, {}, f"syntax_check 后仍不可用: {rl_reason}. {detail}"

    # MODIFY_LAW: gf() physics bounds probe before smoke / PPO
    if (
        str(task_mode).upper() == "MODIFY_LAW"
        and getattr(cfg.simulation, "physics_bounds_enabled", True)
    ):
        from multi_agent.integration.t4_low_risk import should_skip_physics_bounds_probe
        from multi_agent.simulation.sim_timeout import get_physics_bounds_timeout_sec
        from multi_agent.simulation.task_workspace import TaskWorkspace
        from multi_agent.simulation.engine_resolver import resolve_engine_config

        _skip_pb = should_skip_physics_bounds_probe(
            content, mission_conditions_str, task_prompt,
        )
        if _skip_pb:
            logger.info(
                "[Layer2Gate] Skip physics bounds (deterministic T4 minimal APN)"
            )
        else:
            _eng_cfg = resolve_engine_config(
                engine=cfg.simulation.engine,
                octave_path=cfg.simulation.octave_path,
                matlab_path=cfg.simulation.matlab_path,
            )
            _engine = _eng_cfg.engine
            if _engine in ("matlab_engine", "python", "auto"):
                _engine = "matlab"
            _ny_lim = float(getattr(cfg.simulation, "physics_bounds_ny_limit_g", 20.0))
            with TaskWorkspace(auto_cleanup=True) as _ws:
                _pb_ok, _pb_msg = await _ws.verify_physics_bounds(
                    path,
                    ny_limit_g=_ny_lim,
                    engine=_engine,
                    octave_path=_eng_cfg.octave_path,
                    matlab_path=_eng_cfg.matlab_path,
                    timeout_sec=get_physics_bounds_timeout_sec(),
                )
            if not _pb_ok:
                return False, path, {}, f"物理限幅探测未通过: {_pb_msg}"

    metrics = dict(existing_metrics or {})
    run_needed = not metrics_usable(metrics, "")

    async def _run_smoke(target_path: str) -> Tuple[bool, str, Dict[str, Any], str]:
        logger.info("[Layer2Gate] Smoke run_simulation nmc=%d → %s", smoke_nmc, target_path)
        sim_result = await run_tool.execute(
            script_path=target_path,
            mission_conditions=mission_conditions_str,
            nmc=smoke_nmc,
        )
        if sim_result.get("status") != "success":
            err = sim_result.get("message") or sim_result.get("stderr") or "unknown"
            stderr = (sim_result.get("stderr") or sim_result.get("message") or "")[:2000]
            if stderr:
                await syntax_tool.execute(
                    script_path=target_path,
                    error_message=stderr,
                    auto_fix=True,
                )
                sim_result = await run_tool.execute(
                    script_path=target_path,
                    mission_conditions=mission_conditions_str,
                    nmc=smoke_nmc,
                )
            if sim_result.get("status") != "success":
                err = sim_result.get("message") or sim_result.get("stderr") or err
                return False, target_path, {}, f"run_simulation 烟雾测试失败: {err[:300]}"
        stdout = sim_result.get("_full_stdout") or sim_result.get("stdout") or ""
        parsed = parse_sim_stdout(stdout)
        ok_smoke, smoke_reason = stdout_smoke_ok(stdout, parsed)
        if not ok_smoke:
            return False, target_path, parsed, f"烟雾测试指标无效: {smoke_reason}"
        return True, target_path, parsed, "ok"

    if run_needed:
        smoke_ok, path, metrics, smoke_msg = await _run_smoke(path)
        if not smoke_ok and not need_regen:
            logger.warning(
                "[Layer2Gate] Existing script failed smoke (%s) — regenerating from KB",
                smoke_msg,
            )
            gen_result = await gen_tool.execute(
                script_name="guidance_gate",
                task_description=(task_prompt or "")[:2000],
                mode="TUNE_PARAMS",
                mission_conditions=mission_conditions_str,
                nmc=smoke_nmc,
                force_kb_template=True,
            )
            if gen_result.get("status") != "success":
                return False, path, metrics, smoke_msg
            new_path = gen_result.get("script_path", "")
            if not new_path or not os.path.isfile(new_path):
                return False, path, metrics, smoke_msg
            path = new_path
            content = _read_script(path)
            rl_ok, rl_reason = validate_rl_script_content(content)
            if not rl_ok:
                return False, path, metrics, f"KB 重生后仍不可用: {rl_reason}"
            smoke_ok, path, metrics, smoke_msg = await _run_smoke(path)
        if not smoke_ok:
            return False, path, metrics, smoke_msg

    # T4 MODIFY_LAW: closed-loop peak gate (Nmc=modify_gate_nmc)
    if str(task_mode).upper() == "MODIFY_LAW":
        try:
            from multi_agent.integration.t4_low_risk import (
                check_modify_law_closed_loop_gate,
                is_t4_mission,
            )
            if is_t4_mission(task_prompt=task_prompt):
                gate_nmc = int(getattr(cfg.t4_low_risk, "modify_gate_nmc", 10))
                gate_nmc = max(
                    gate_nmc,
                    int(getattr(cfg.simulation, "modify_gate_nmc", 10)),
                )
                gate_nmc = max(5, min(gate_nmc, 25))
                logger.info("[Layer2Gate] T4 MODIFY closed-loop gate nmc=%d", gate_nmc)
                sim_result = await run_tool.execute(
                    script_path=path,
                    mission_conditions=mission_conditions_str,
                    nmc=gate_nmc,
                )
                if sim_result.get("status") != "success":
                    err = sim_result.get("message") or sim_result.get("stderr") or "unknown"
                    return False, path, metrics, f"MODIFY 闭环快扫失败: {err[:300]}"
                stdout = sim_result.get("_full_stdout") or sim_result.get("stdout") or ""
                gate_metrics = parse_sim_stdout(stdout)
                ok_smoke, smoke_reason = stdout_smoke_ok(stdout, gate_metrics)
                if not ok_smoke:
                    return False, path, gate_metrics, f"MODIFY 闭环快扫无效: {smoke_reason}"
                metrics = gate_metrics
                _pass, _msg = check_modify_law_closed_loop_gate(gate_metrics, task_prompt)
                if not _pass:
                    return False, path, gate_metrics, _msg
        except Exception as _gate_exc:
            logger.warning("[Layer2Gate] MODIFY closed-loop gate skipped: %s", _gate_exc)

    return True, path, metrics, "Layer2 gate passed (syntax + smoke)"


__all__ = [
    "ensure_script_ready_for_layer3",
    "metrics_usable",
    "stdout_smoke_ok",
]
