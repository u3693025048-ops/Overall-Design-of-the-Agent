#!/usr/bin/env python3
"""
JudgeRequirementsTool - Layer 2 Hermes tool.

Called immediately after run_simulation completes.
Evaluates whether the simulation metrics satisfy the task requirements
extracted from the user natural language prompt.

Decision logic (two-stage):
  Stage 1 - Rule-based fast check (hit_rate/SEP/peak_ny/PM/BW thresholds).
  Stage 2 - LLM judgment via ReflectionAgent (when Stage 1 passes or is close).

Returns JSON: {satisfied, reason, metrics_summary, next_step}
  next_step: "done" -> Layer3 reflection | "rl_optimize" -> Layer3 RL
"""

import json
import logging
import re
from typing import Any, Dict, Optional

from multi_agent.integration.judgment_agent import (
    _extract_numeric_requirements,
    all_requirements_satisfied,
    requirement_score,
    requirements_complete,
)
from multi_agent.tools.tool_arg_aliases import (
    coerce_script_path as _coerce_script_path,
    coerce_task_prompt as _coerce_task_prompt,
)

logger = logging.getLogger(__name__)


def _parse_requirements_from_prompt(prompt: str) -> Dict[str, Any]:
    """Extract numeric requirements from natural language prompt."""
    return _extract_numeric_requirements(prompt)


def _requirement_score(reqs: Dict[str, Any]) -> int:
    """Higher = more complete task requirements."""
    return requirement_score(reqs)


def _resolve_requirements(task_prompt: str, cache: str = "") -> Dict[str, Any]:
    """Prefer the cached user prompt when it carries fuller numeric requirements."""
    cached = _parse_requirements_from_prompt(cache or "")
    called = _parse_requirements_from_prompt(task_prompt or "")
    if _requirement_score(cached) >= max(_requirement_score(called), 3):
        merged = dict(cached)
        for k, v in called.items():
            if k not in merged:
                merged[k] = v
        return merged
    return {**cached, **called}


def _hard_constraint_failed(reasons: list) -> bool:
    """True when rule check failed on safety-critical metrics (never LLM-overridden)."""
    for r in reasons:
        if "[NG]" not in r:
            continue
        # Hard constraints: PeakNy, hit_rate, SEP, PM, BW (all critical)
        if any(k in r for k in ("PeakNy", "命中率", "SEP", "hit_rate", "PM", "BW")):
            return True
    return False


def _extract_mc_seed_metrics(metrics: Dict[str, Any]) -> list:
    """Return per-seed metric dicts if present in simulation output."""
    raw = (
        metrics.get("mc_seed_results")
        or metrics.get("seed_metrics")
        or metrics.get("mc_results_by_seed")
    )
    if not isinstance(raw, list):
        return []
    return [item for item in raw if isinstance(item, dict)]


def _rule_check(metrics: Dict[str, float], reqs: Dict[str, Any]) -> tuple:
    """Returns (satisfied: bool, reasons: list[str])."""
    seed_metrics = _extract_mc_seed_metrics(metrics)
    if seed_metrics:
        all_ok = True
        all_reasons = []
        for idx, seed_item in enumerate(seed_metrics, start=1):
            single = dict(seed_item)
            single.pop("mc_seed_results", None)
            ok, reasons = _rule_check(single, reqs)
            all_ok = all_ok and ok
            seed = single.get("mc_seed", single.get("random_seed", idx))
            status = "PASS" if ok else "FAIL"
            all_reasons.append(
                f"MC seed {seed}: {status} ({'; '.join(reasons)})"
            )
        return all_ok, all_reasons

    reasons = []
    satisfied = True

    if "hit_rate_min" in reqs:
        hr = metrics.get("hit_rate", 0.0)
        ok = hr >= reqs["hit_rate_min"]
        satisfied = satisfied and ok
        reasons.append(
            f"命中率 {hr:.1f}% {'>=' if ok else '<'} 要求 {reqs['hit_rate_min']:.1f}%"
            + (" [OK]" if ok else " [NG]")
        )

    if "sep_max" in reqs:
        sep = metrics.get("SEP", metrics.get("miss_distance", 999.0))
        ok = sep <= reqs["sep_max"]
        satisfied = satisfied and ok
        reasons.append(
            f"SEP {sep:.2f}m {'<=' if ok else '>'} 要求 {reqs['sep_max']:.2f}m"
            + (" [OK]" if ok else " [NG]")
        )

    if "peak_ny_max" in reqs or "peak_ny_mean_max" in reqs:
        from multi_agent.rl.metric_utils import check_peak_ny, normalize_peak_ny_requirements

        ok_peak, peak_reasons = check_peak_ny(
            metrics, normalize_peak_ny_requirements(reqs)
        )
        satisfied = satisfied and ok_peak
        reasons.extend(peak_reasons)

    if "pm_min" in reqs:
        pm = metrics.get("pitch_PM", 0.0)
        pm_min_seen = metrics.get("pitch_PM_min", pm)
        pm_max_seen = metrics.get("pitch_PM_max", pm)
        if "pm_max" in reqs:
            ok = reqs["pm_min"] <= pm_min_seen and pm_max_seen <= reqs["pm_max"]
            satisfied = satisfied and ok
            reasons.append(
                f"PM {pm:.1f}deg 在 [{reqs['pm_min']:.1f}, {reqs['pm_max']:.1f}]"
                + (" [OK]" if ok else " [NG]")
            )
        else:
            ok = pm_min_seen >= reqs["pm_min"]
            satisfied = satisfied and ok
            reasons.append(
                f"PM {pm:.1f}deg {'>=' if ok else '<'} 要求 {reqs['pm_min']:.1f}deg"
                + (" [OK]" if ok else " [NG]")
            )

    if "bw_min" in reqs and "bw_max" in reqs:
        bw = metrics.get("pitch_BW", 0.0)
        bw_min_seen = metrics.get("pitch_BW_min", bw)
        bw_max_seen = metrics.get("pitch_BW_max", bw)
        ok = reqs["bw_min"] <= bw_min_seen and bw_max_seen <= reqs["bw_max"]
        satisfied = satisfied and ok
        reasons.append(
            f"BW {bw:.1f} rad/s 在 [{reqs['bw_min']},{reqs['bw_max']}]"
            + (" [OK]" if ok else " [NG]")
        )

    return satisfied, reasons


class JudgeRequirementsTool:

    name = "judge_requirements"

    description = (
        "判断仿真结果是否满足任务指标要求。"
        "输入任务描述和仿真指标（hit_rate/SEP/peak_ny/pitch_PM/pitch_BW），"
        "返回 {satisfied, reason, next_step}。"
        "next_step='done' 表示可跳过 RL 直接进入反思智能体；"
        "'rl_optimize' 表示需要 RL 参数优化。"
    )

    input_schema = {
        "type": "object",
        "properties": {
            "task_prompt": {
                "type": "string",
                "description": "用户任务描述（含指标要求）",
            },
            "metrics": {
                "type": "object",
                "description": (
                    "仿真指标字典。支持的键名（任一格式均可）：\n"
                    "  hit_rate 或 hit_rate_pct (命中率%)\n"
                    "  SEP 或 SEP_m (脱靶量m)\n"
                    "  peak_ny 或 PeakNy_g (峰值过载g)\n"
                    "  pitch_PM 或 PM_deg (相位裕度°)\n"
                    "  pitch_BW 或 BW_rads (带宽rad/s)"
                ),
            },
            "script_path": {
                "type": "string",
                "description": "（可选）仿真脚本路径，用于标识本次结果",
            },
        },
        "required": ["task_prompt", "metrics"],
    }

    def __init__(self):
        self._reflection_agent = None
        self._run_simulation_tool = None
        self._task_prompt_cache: str = ""
        self._last_result: Optional[Dict[str, Any]] = None

    def set_reflection_agent(self, agent: Any) -> None:
        self._reflection_agent = agent

    def set_run_simulation_tool(self, tool: Any) -> None:
        """Optional trusted metrics source (run_simulation.last_metrics)."""
        self._run_simulation_tool = tool

    def _merge_trusted_sim_metrics(
        self, metrics: Dict[str, float]
    ) -> tuple[Dict[str, float], bool]:
        """Prefer parsed simulation metrics over LLM hand-filled values."""
        tool = self._run_simulation_tool
        if tool is None:
            return metrics, False
        trusted = getattr(tool, "last_metrics", None) or {}
        if not trusted:
            return metrics, False

        merged = dict(metrics)
        overridden = False
        _trusted_keys = (
            "hit_rate", "SEP", "miss_distance",
            "peak_ny", "peak_n", "peak_ny_max", "peak_n_max",
            "pitch_PM", "pitch_PM_min", "pitch_PM_max",
            "pitch_BW", "pitch_BW_min", "pitch_BW_max",
        )
        for key in _trusted_keys:
            if key not in trusted:
                continue
            try:
                val = float(trusted[key])
            except (TypeError, ValueError):
                continue
            if val == 0.0 and key in ("peak_ny_max", "peak_n_max"):
                continue
            if merged.get(key) != val:
                merged[key] = val
                overridden = True
        seed_metrics = getattr(tool, "last_seed_metrics", None) or trusted.get(
            "mc_seed_results"
        )
        if seed_metrics:
            merged["mc_seed_results"] = seed_metrics
            overridden = True
        if "SEP" not in merged and "miss_distance" in merged:
            merged["SEP"] = merged["miss_distance"]
        from multi_agent.rl.metric_utils import normalize_peak_ny_aliases
        merged = normalize_peak_ny_aliases(merged)
        return merged, overridden

    def set_task_prompt(self, prompt: str) -> None:
        self._task_prompt_cache = prompt

    async def execute(
        self,
        task_prompt: str = "",
        metrics: Optional[Dict[str, Any]] = None,
        script_path: str = "",
        **kwargs: Any,
    ) -> str:
        task_prompt = _coerce_task_prompt(task_prompt, **kwargs) or self._task_prompt_cache
        if kwargs.get("path") and not script_path:
            script_path = _coerce_script_path(script_path, **kwargs)
        if metrics is None:
            metrics = {}
        raw_metrics = dict(metrics)
        raw_seed_metrics = _extract_mc_seed_metrics(raw_metrics)

        m: Dict[str, float] = {}
        for k, v in metrics.items():
            try:
                m[k] = float(v)
            except (TypeError, ValueError):
                pass
        if raw_seed_metrics:
            m["mc_seed_results"] = raw_seed_metrics
        # Normalise key aliases that Hermes agent may send.
        # Hermes often attaches unit suffixes (_pct, _m, _g, _deg, _rads)
        # or uses short forms (PM, BW, PeakNy).  Map them all to canonical
        # keys expected by _rule_check: hit_rate, SEP, peak_ny, pitch_PM, pitch_BW.
        _key_aliases = {
            # short forms
            "PM":           "pitch_PM",
            "BW":           "pitch_BW",
            "GM":           "pitch_GM",
            "PeakNy":       "peak_ny",
            "PeakNy_max":   "peak_ny_max",
            "PeakN":        "peak_n",
            "PeakN_max":    "peak_ny_max",
            # unit-suffixed forms (Hermes LLM generated)
            "hit_rate_pct": "hit_rate",
            "SEP_m":        "SEP",
            "PeakNy_g":     "peak_ny",
            "PeakNy_mean_g": "peak_ny",
            "peak_ny_mean_g": "peak_ny",
            "PeakNy_max_g": "peak_ny_max",
            "peak_ny_max_g": "peak_ny_max",
            "peak_n_max_g": "peak_ny_max",
            "PM_deg":       "pitch_PM",
            "BW_rads":      "pitch_BW",
            "BW_rad_s":     "pitch_BW",
            "PM_degree":    "pitch_PM",
            "miss_distance_m": "miss_distance",
            "peak_ny_g":    "peak_ny",
            "peak_n_g":     "peak_n",
        }
        for _short, _full in _key_aliases.items():
            if _short in m and _full not in m:
                m[_full] = m[_short]
        from multi_agent.rl.metric_utils import normalize_peak_ny_aliases
        m = normalize_peak_ny_aliases(m)
        m, _trusted_override = self._merge_trusted_sim_metrics(m)
        if _trusted_override:
            logger.info("[JudgeRequirements] metrics overridden from run_simulation.last_metrics")

        reqs = _resolve_requirements(task_prompt, self._task_prompt_cache)
        if not reqs and self._task_prompt_cache and task_prompt != self._task_prompt_cache:
            reqs = _parse_requirements_from_prompt(self._task_prompt_cache)

        if reqs:
            satisfied, reasons = _rule_check(m, reqs)
            reason_str = "; ".join(reasons) if reasons else "规则检查通过"
            if not requirements_complete(reqs, task_prompt or self._task_prompt_cache):
                satisfied = False
                reason_str += "; 任务要求解析不完整，禁止 satisfied 放行"
            elif _requirement_score(reqs) < 3:
                satisfied = False
                reason_str += "; 任务要求解析不完整，禁止 satisfied 放行"
        else:
            satisfied = False
            reason_str = "未检测到明确数值要求，将转入 RL 优化"

        if self._reflection_agent is not None and reqs and task_prompt:
            try:
                ref = await self._reflection_agent.reflect(task_prompt, {"metrics": m})
                llm_satisfied = not ref.get("needs_optimization", True)
                llm_suggestion = ref.get("suggestion", "")
                if llm_satisfied and not satisfied:
                    reason_str += (
                        "; LLM 建议通过但规则未满足，忽略 LLM 放行"
                    )
                elif not llm_satisfied and satisfied:
                    satisfied = False
                    reason_str += f"; LLM判断需改进: {llm_suggestion[:200]}"
                elif llm_satisfied:
                    # LLM cannot override — all metrics must pass rule check
                    rule_ok, _ = _rule_check(m, reqs)
                    if not rule_ok or not all_requirements_satisfied(
                        m, reqs, task_prompt or self._task_prompt_cache
                    ):
                        satisfied = False
                        reason_str += "; LLM 建议通过但规则未全部满足，忽略 LLM 放行"
            except Exception as exc:
                logger.warning(f"JudgeRequirements LLM stage failed: {exc}")

        summary = {
            "命中率(%)": m.get("hit_rate", 0.0),
            "SEP(m)": m.get("SEP", m.get("miss_distance", 0.0)),
            "PeakNy_mean(g)": m.get("peak_ny", m.get("peak_n", 0.0)),
            "PM(deg)": m.get("pitch_PM", 0.0),
            "BW(r/s)": m.get("pitch_BW", 0.0),
        }

        seed_metrics = _extract_mc_seed_metrics(m)
        if seed_metrics:
            summary["MC_seed_count"] = len(seed_metrics)
            summary["MC_seed_all_pass"] = bool(satisfied)

        result = {
            "satisfied": satisfied,
            "reason": reason_str,
            "metrics_summary": summary,
            "next_step": "done" if satisfied else "rl_optimize",
            "requirements_found": reqs,
            "metrics_trusted_from_sim": _trusted_override,
        }
        self._last_result = result
        logger.info(f"[JudgeRequirements] satisfied={satisfied} | {reason_str[:120]}")
        return json.dumps(result, ensure_ascii=False)
