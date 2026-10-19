#!/usr/bin/env python3
"""
JudgmentAgent — Layer 3 per-episode judgment component.

Called by MatlabRLOptimizer after each episode when a new best is found.
Provides lightweight real-time evaluation of whether RL optimization can
stop early (requirements already met).

Judgment contract (all hard constraints, no partial pass):
  - Every metric stated in the task prompt must pass rule checks.
  - ``satisfied=True`` ONLY when hit/SEP/PeakNy(mean)/PM/BW are ALL [OK].
  - Any single [NG] → ``satisfied=False``; failures are not split into
    "blocking" vs "optimization-only" categories inside this agent.

Pipeline inside ``judge()``:
  Level 0 - LLM requirement extraction (once, merged onto regex base).
  Level 1 - Numeric rule check: AND of all metric gates.
  (Post-run LLM reflection lives in ReflectionAgent / OptimWorkflow.)

Interface contract with MatlabRLOptimizer:
  result = await judgment_agent.judge(metrics, task_prompt)
  result.satisfied  -> bool: trigger RL early exit?
  result.suggestion -> str: all [NG] reasons joined when not satisfied
"""

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class JudgmentResult:
    satisfied: bool
    suggestion: str = ""
    reasons: List[str] = field(default_factory=list)
    metrics_snapshot: Dict[str, float] = field(default_factory=dict)
    next_action: str = "tune_params"  # "done" | "tune_params" | "modify_law"


def _extract_numeric_requirements(prompt: str) -> Dict[str, Any]:
    """Parse numeric performance requirements from task prompt (regex fallback)."""
    reqs: Dict[str, Any] = {}

    # hit_rate >= X%
    for pat in [
        r"命中率\s*[>=>=]+\s*(\d+(?:\.\d+)?)\s*%",
        r"hit[_\s]*rate\s*[>=>=]+\s*(\d+(?:\.\d+)?)",
    ]:
        m = re.search(pat, prompt, re.IGNORECASE)
        if m:
            reqs["hit_rate_min"] = float(m.group(1))
            break

    # SEP / miss <= X m
    for pat in [
        r"(?:脱靶量|SEP|miss)\s*[<=<=]+\s*(\d+(?:\.\d+)?)\s*m",
        r"miss\s*distance\s*[<=<=]+\s*(\d+(?:\.\d+)?)",
    ]:
        m = re.search(pat, prompt, re.IGNORECASE)
        if m:
            reqs["sep_max"] = float(m.group(1))
            break

    # PeakNy <= X g
    m = re.search(
        r"(?:峰值法向过载|法向过载|PeakNy|peak_ny)\s*[<=<=]+\s*(\d+(?:\.\d+)?)\s*g",
        prompt, re.IGNORECASE,
    )
    if m:
        reqs["peak_ny_max"] = float(m.group(1))

    # PM range: [X, Y] or >= X  (allow "PM相位裕度 在 ..." style labels)
    m = re.search(
        r"PM[^\d]*?(?:在|in|∈)\s*\[?\s*(\d+(?:\.\d+)?)\s*°?\s*[,~到\-]\s*(\d+(?:\.\d+)?)",
        prompt, re.IGNORECASE,
    )
    if m:
        reqs["pm_min"] = float(m.group(1))
        reqs["pm_max"] = float(m.group(2))
    else:
        for pat in [
            r"PM[^\d]*?[>=>=]+\s*(\d+(?:\.\d+)?)\s*°?",
            r"相位裕度\s*[>=>=]+\s*(\d+(?:\.\d+)?)",
        ]:
            m = re.search(pat, prompt, re.IGNORECASE)
            if m:
                reqs["pm_min"] = float(m.group(1))
                break

    # BW in range [X, Y]  (allow "BW带宽 在 ..." style labels)
    m = re.search(
        r"BW[^\d]*?(?:在|in)\s*\[?\s*(\d+(?:\.\d+)?)\s*[,~到\-]\s*(\d+(?:\.\d+)?)",
        prompt, re.IGNORECASE,
    )
    if m:
        reqs["bw_min"] = float(m.group(1))
        reqs["bw_max"] = float(m.group(2))

    from multi_agent.rl.metric_utils import normalize_peak_ny_requirements

    return normalize_peak_ny_requirements(reqs)


_RANGE_PAIRS = (("pm_min", "pm_max"), ("bw_min", "bw_max"))


def merge_requirements(
    base: Dict[str, Any],
    overlay: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Merge overlay onto regex-parsed *base* without dropping range upper bounds.

    LLM extraction may return ``bw_min`` without ``bw_max``; the prompt regex
    pair must be preserved or BW checks are silently skipped (log06151449).
    """
    from multi_agent.rl.metric_utils import normalize_peak_ny_requirements

    out = dict(base or {})
    for k, v in (overlay or {}).items():
        out[k] = v
    for lo, hi in _RANGE_PAIRS:
        if lo in base and hi in base and (lo not in overlay or hi not in overlay):
            out[lo] = base[lo]
            out[hi] = base[hi]
    return normalize_peak_ny_requirements(out)


def resolve_task_requirements(
    task_prompt: str,
    judgment_agent: Optional[Any] = None,
) -> Dict[str, Any]:
    """Authoritative requirements for gating — regex base + optional JA overlay."""
    base = _extract_numeric_requirements(task_prompt) if task_prompt else {}
    ja_reqs: Dict[str, Any] = {}
    if judgment_agent is not None and getattr(judgment_agent, "_reqs", None):
        ja_reqs = dict(judgment_agent._reqs)
    return merge_requirements(base, ja_reqs)


def requirement_score(reqs: Dict[str, Any]) -> int:
    """Higher = more complete task requirements (max 7 keys incl. range bounds)."""
    keys = (
        "hit_rate_min",
        "sep_max",
        "peak_ny_max",
        "pm_min",
        "pm_max",
        "bw_min",
        "bw_max",
    )
    return sum(1 for k in keys if k in reqs)


def requirements_complete(reqs: Dict[str, Any], task_prompt: str) -> bool:
    """True when every metric type mentioned in *task_prompt* was parsed."""
    if not reqs or not task_prompt:
        return False

    expected: List[str] = []
    if re.search(r"命中率|hit[_\s]*rate", task_prompt, re.IGNORECASE):
        expected.append("hit_rate_min")
    if re.search(r"SEP|脱靶|miss", task_prompt, re.IGNORECASE):
        expected.append("sep_max")
    if re.search(r"PeakNy|peak_ny|峰值法向过载|法向过载", task_prompt, re.IGNORECASE):
        expected.append("peak_ny_max")
    if re.search(r"\bPM\b|相位裕度", task_prompt, re.IGNORECASE):
        expected.append("pm_min")
    if re.search(r"\bBW\b|带宽", task_prompt, re.IGNORECASE):
        expected.append("bw_min")

    if not expected:
        return requirement_score(reqs) >= 3
    return all(k in reqs for k in expected)


def _extract_mc_seed_metrics(metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
    raw = (
        metrics.get("mc_seed_results")
        or metrics.get("seed_metrics")
        or metrics.get("mc_results_by_seed")
    )
    if not isinstance(raw, list):
        return []
    return [item for item in raw if isinstance(item, dict)]


def all_requirements_satisfied(
    metrics: Dict[str, float],
    reqs: Dict[str, Any],
    task_prompt: str = "",
) -> bool:
    """All parsed requirements met; incomplete parsing never counts as satisfied.

    Completion requires every stated metric to pass its hard bound (hit, SEP,
    PeakNy mean limit, PM range, BW range).
    """
    if not reqs:
        return False
    if task_prompt and not requirements_complete(reqs, task_prompt):
        return False
    ok, _ = _rule_based_check(metrics, reqs)
    return ok


def _rule_based_check(
    metrics: Dict[str, float],
    reqs: Dict[str, Any],
) -> tuple:
    """Fast numeric check. Returns (satisfied: bool, reasons: list[str])."""
    if not reqs:
        return False, ["未检测到数值要求，保守返回不满足"]

    seed_metrics = _extract_mc_seed_metrics(metrics)
    if seed_metrics:
        all_ok = True
        all_reasons: List[str] = []
        for idx, seed_item in enumerate(seed_metrics, start=1):
            single = dict(seed_item)
            single.pop("mc_seed_results", None)
            ok, reasons = _rule_based_check(single, reqs)
            all_ok = all_ok and ok
            seed = single.get("mc_seed", single.get("random_seed", idx))
            status = "PASS" if ok else "FAIL"
            all_reasons.append(
                f"MC seed {seed}: {status} ({'; '.join(reasons)})"
            )
        return all_ok, all_reasons

    satisfied = True
    reasons: List[str] = []

    if "hit_rate_min" in reqs:
        hr = metrics.get("hit_rate", 0.0)
        ok = hr >= reqs["hit_rate_min"]
        satisfied = satisfied and ok
        reasons.append(
            f"命中率 {hr:.1f}% {'>=' if ok else '<'} {reqs['hit_rate_min']:.1f}%"
            + (" [OK]" if ok else " [NG]")
        )

    if "sep_max" in reqs:
        sep = metrics.get("SEP", metrics.get("miss_distance", 999.0))
        ok = sep <= reqs["sep_max"]
        satisfied = satisfied and ok
        reasons.append(
            f"SEP {sep:.2f}m {'<=' if ok else '>'} {reqs['sep_max']:.2f}m"
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
                f"PM {pm:.1f}deg {'>=' if ok else '<'} {reqs['pm_min']:.1f}deg"
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


class JudgmentAgent:
    """
    Layer 3 per-episode judgment agent.

    Usage in RL optimizer:
        judgment = JudgmentAgent(reflection_agent=..., task_prompt=...)
        result = await judgment.judge(metrics)
        if result.satisfied:
            break  # early exit from RL loop
    """

    def __init__(
        self,
        reflection_agent: Optional[Any] = None,
        task_prompt: str = "",
    ):
        self._reflection_agent = reflection_agent
        self._task_prompt = task_prompt
        self._reqs: Dict[str, Any] = {}
        self._reqs_from_llm: bool = False
        if task_prompt:
            self._reqs = _extract_numeric_requirements(task_prompt)
            logger.info(f"[JudgmentAgent] Regex requirements: {self._reqs}")

    def update_task_prompt(self, prompt: str) -> None:
        self._task_prompt = prompt
        self._reqs = _extract_numeric_requirements(prompt)
        self._reqs_from_llm = False

    async def _extract_requirements_llm(self) -> Dict[str, Any]:
        """Use the reflection agent's LLM to extract numeric requirements.

        Returns a dict with the same keys as _extract_numeric_requirements().
        Falls back to empty dict on any failure.
        """
        llm = getattr(self._reflection_agent, "llm", None)
        if llm is None or not self._task_prompt:
            return {}

        extract_prompt = (
            "从以下任务描述中提取数值性能要求，以JSON格式返回。\n"
            f"任务描述：{self._task_prompt}\n\n"
            "返回格式（只包含任务中明确提及的指标，未提及的不要出现）：\n"
            "{\n"
            '  "hit_rate_min": 92.0,   // 命中率下限 (%)\n'
            '  "sep_max": 7.0,         // SEP/脱靶量上限 (m)\n'
            '  "peak_ny_max": 20.0,    // 峰值法向过载上限 (g)\n'
            '  "pm_min": 45.0,         // 相位裕度下限 (deg)\n'
            '  "pm_max": 70.0,         // 相位裕度上限 (deg，若有区间要求则填写)\n'
            '  "bw_min": 20.0,         // 带宽下限 (rad/s)\n'
            '  "bw_max": 85.0          // 带宽上限 (rad/s)\n'
            "}\n"
            "只返回JSON，不要任何说明文字。"
        )

        try:
            from langchain_core.messages import HumanMessage
            response = await llm.ainvoke([HumanMessage(content=extract_prompt)])
            # P2-1 户部：记一次真实用量（langchain AIMessage 走 usage_metadata；
            # 取不到时只计 n_unobserved，不填零）
            from multi_agent import cost_meter as _cost
            _cost.record(response, label="judgment_agent.condition_extract")
            content = response.content
            content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
            content = re.sub(r"```(?:json)?", "", content).strip().strip("`").strip()
            parsed = json.loads(content)
            result = {
                k: float(v)
                for k, v in parsed.items()
                if isinstance(v, (int, float)) and v is not None
            }
            logger.info(f"[JudgmentAgent] LLM-extracted requirements: {result}")
            return result
        except Exception as exc:
            logger.warning(f"[JudgmentAgent] LLM requirement extraction failed: {exc}")
            return {}

    async def judge(
        self,
        metrics: Dict[str, float],
        task_prompt: Optional[str] = None,
    ) -> JudgmentResult:
        """Evaluate metrics against task requirements.

        All stated metrics must pass (hard AND gate). There is no partial
        satisfaction and no subset of [NG] items treated as optional.

        Level 0: LLM requirement extraction (once), merged onto regex base.
        Level 1: numeric rule check on every metric; any [NG] → not satisfied.
        """
        if task_prompt:
            self._task_prompt = task_prompt
            self._reqs = _extract_numeric_requirements(task_prompt)
            self._reqs_from_llm = False

        # Level 0: LLM extraction (one-time, merged onto regex — never drop bw_max/pm_max)
        if not self._reqs_from_llm and self._task_prompt:
            llm_reqs = await self._extract_requirements_llm()
            if llm_reqs:
                self._reqs = merge_requirements(
                    _extract_numeric_requirements(self._task_prompt),
                    llm_reqs,
                )
            self._reqs_from_llm = True

        # Level 1: Rule check — all stated metrics must pass
        rule_ok, reasons = _rule_based_check(metrics, self._reqs)
        if self._task_prompt:
            from multi_agent.integration.judgment_agent import requirements_complete

            if not requirements_complete(self._reqs, self._task_prompt):
                rule_ok = False
                reasons.append("任务要求解析不完整 [NG]")

        if not rule_ok:
            # Rules say not satisfied - skip LLM to save cost
            suggestion = "; ".join(r for r in reasons if "[NG]" in r)
            return JudgmentResult(
                satisfied=False,
                suggestion=suggestion,
                reasons=reasons,
                metrics_snapshot=dict(metrics),
                next_action="tune_params",
            )

        # Level 1 passed — report satisfied.
        # NOTE: No Level 2 LLM call here.  The authoritative LLM judgment is
        # performed once by ReflectionAgent *after* the full RL run completes
        # (OptimizationWorkflow.run → self.reflection_agent.reflect).
        # Adding a second LLM call inside the RL loop is redundant cost with
        # no additional safety net benefit.
        return JudgmentResult(
            satisfied=True,
            suggestion="规则检查通过",
            reasons=reasons,
            metrics_snapshot=dict(metrics),
            next_action="done",
        )

    async def reflect(
        self,
        task_prompt: str,
        input_data: Dict[str, Any],
        **kwargs,
    ) -> Dict[str, Any]:
        """Adapter so JudgmentAgent can be used as a drop-in replacement for
        ReflectionAgent inside MatlabRLOptimizer.

        RL optimizer calls active_reflector.reflect(task_prompt, data) per
        episode. This adapter forwards to judge() and converts the result
        to the dict format expected by the optimizer.
        """
        metrics = input_data.get("metrics", input_data)
        if not isinstance(metrics, dict):
            metrics = {}
        result = await self.judge(metrics, task_prompt=task_prompt)
        return {
            "needs_optimization": not result.satisfied,
            "suggestion":         result.suggestion,
            "next_action":        result.next_action,
            "reasons":            result.reasons,
            "satisfied":          result.satisfied,
        }
