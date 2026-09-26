"""
Design-path policy for T4 multi-criteria optimization.

Encodes when to use TUNE_PARAMS vs MODIFY_LAW, PeakNy mean gating, and
MODIFY_LAW gf() change verification.
"""

from __future__ import annotations

import hashlib
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

from multi_agent.integration.judgment_agent import (
    _extract_numeric_requirements,
    all_requirements_satisfied,
    requirement_score,
    requirements_complete,
)
from multi_agent.rl.metric_constraints import constraints_satisfied
from multi_agent.rl.metric_utils import get_peak_ny, get_sep
from multi_agent.metric_contract import PEAK_NY_MAX

# Consecutive TUNE_PARAMS iterations with PeakNy mean over limit → MODIFY_LAW
TUNE_STALL_THRESHOLD = 2

# After MODIFY_LAW, PeakNy mean within this margin (g) above limit → tune_params not modify_law
PEAK_NY_NEAR_MISS_G = 2.5


def peak_near_miss_margin_g(default: float = 8.0) -> float:
    """Configurable near-miss margin (g above peak limit) for tune vs modify."""
    try:
        from multi_agent.integration.t4_low_risk import get_peak_near_miss_margin_g
        return get_peak_near_miss_margin_g(default)
    except Exception:
        return default

_OPTIM_TASK_MARKERS = (
    "命中率", "SEP", "PeakNy", "peak_ny", "PM", "BW", "脱靶", "过载",
)

_APN_MARKERS = (
    "a_t_perp", "a_T_perp", "TVx", "TVy", "TVz", "tanh", "APN",
    "增广", "目标加速度", "前馈", "T_go", "Tgo", "schedul", "LPF", "低通",
)


def is_optimization_task(task_prompt: str) -> bool:
    if not task_prompt:
        return False
    t = task_prompt
    return any(m in t for m in _OPTIM_TASK_MARKERS)


def requirements_from_prompt(task_prompt: str) -> Dict[str, Any]:
    if not task_prompt:
        return {}
    return _extract_numeric_requirements(task_prompt)


def task_fully_satisfied(
    metrics: Optional[Dict[str, Any]],
    task_prompt: str,
    reqs: Optional[Dict[str, Any]] = None,
) -> bool:
    """True only when every parsed task metric (hit/SEP/PeakNy/PM/BW) passes."""
    r = reqs if reqs is not None else requirements_from_prompt(task_prompt)
    if not r or not metrics:
        return False
    return all_requirements_satisfied(metrics, r, task_prompt)


def peak_ny_limit(reqs: Optional[Dict[str, Any]], default: float = PEAK_NY_MAX) -> float:
    if not reqs:
        return default
    try:
        return float(reqs.get("peak_ny_max", default))
    except (TypeError, ValueError):
        return default


def peak_ny_over_limit(metrics: Optional[Dict[str, Any]], limit: float) -> bool:
    if not metrics:
        return False
    from multi_agent.rl.metric_utils import (
        normalize_peak_ny_requirements,
        peak_ny_satisfied,
    )

    return not peak_ny_satisfied(
        metrics, normalize_peak_ny_requirements({"peak_ny_max": limit})
    )


def non_peak_constraints_met(
    metrics: Optional[Dict[str, Any]],
    reqs: Optional[Dict[str, Any]],
) -> bool:
    """Hit/SEP/PM/BW satisfied; PeakNy may still fail."""
    if not metrics or not reqs:
        return False
    subset = dict(reqs)
    for k in ("peak_ny_max", "peak_ny_mean_max"):
        subset.pop(k, None)
    if not subset:
        return False
    ok, _ = _rule_check_subset(metrics, subset)
    return ok


def _rule_check_subset(metrics: Dict[str, Any], reqs: Dict[str, Any]):
    from multi_agent.integration.judgment_agent import _rule_based_check

    return _rule_based_check(metrics, reqs)


def count_tune_peakny_stalls(
    optimization_history: Optional[List[Dict[str, Any]]],
    peak_limit: float,
) -> int:
    """Count trailing TUNE rounds where PeakNy mean still fails."""
    if not optimization_history:
        return 0
    from multi_agent.rl.metric_utils import (
        normalize_peak_ny_requirements,
        peak_ny_satisfied,
    )

    reqs = normalize_peak_ny_requirements({"peak_ny_max": peak_limit})
    stalls = 0
    for row in reversed(optimization_history):
        mode = str(row.get("task_mode") or row.get("mode") or "").upper()
        if mode == "MODIFY_LAW":
            break
        if mode not in ("TUNE_PARAMS", "REUSE_HISTORY", ""):
            continue
        if not peak_ny_satisfied(row, reqs):
            stalls += 1
        else:
            break
    return stalls


def peak_ny_near_miss(
    metrics: Optional[Dict[str, Any]],
    limit: float,
    margin_g: Optional[float] = None,
) -> bool:
    """True when PeakNy mean fails but is close enough to keep tuning."""
    if margin_g is None:
        margin_g = peak_near_miss_margin_g()
    from multi_agent.rl.metric_utils import (
        get_peak_ny,
        get_peak_ny_limit,
        normalize_peak_ny_requirements,
        peak_ny_satisfied,
    )

    reqs = normalize_peak_ny_requirements({"peak_ny_max": limit})
    if peak_ny_satisfied(metrics or {}, reqs):
        return False
    mean_max = get_peak_ny_limit(reqs)
    pny_avg = get_peak_ny(metrics or {})
    return mean_max < pny_avg <= mean_max + margin_g


def should_prefer_tune_over_modify(
    *,
    task_prompt: str = "",
    metrics: Optional[Dict[str, Any]] = None,
    optimization_history: Optional[List[Dict[str, Any]]] = None,
    reflection_feedback: str = "",
) -> bool:
    """
    True when policy must NOT override Planner/Reflection TUNE → MODIFY.

    Covers: post-MODIFY multi-metric failure, near-miss peak band, explicit tune.
    """
    reqs = requirements_from_prompt(task_prompt)
    limit = peak_ny_limit(reqs)
    margin = peak_near_miss_margin_g()

    last_mode = ""
    if optimization_history:
        last_mode = str(optimization_history[-1].get("task_mode", "")).upper()
        if last_mode == "MODIFY_LAW" and metrics and not non_peak_constraints_met(metrics, reqs):
            return True

    if last_mode == "MODIFY_LAW" and metrics and peak_ny_near_miss(metrics, limit, margin):
        return True

    fb = (reflection_feedback or "").lower()
    if "tune_params" in fb or "tune_params" in fb.replace(" ", ""):
        if metrics and not non_peak_constraints_met(metrics, reqs):
            return True
        if metrics and peak_ny_over_limit(metrics, limit):
            return True

    return False


def should_t4_first_iter_modify_law(
    task_prompt: str,
    optimization_history: Optional[List[Dict[str, Any]]] = None,
) -> bool:
    """T4 with no prior history → skip wasteful first TUNE round."""
    if optimization_history:
        return False
    try:
        from multi_agent.integration.t4_low_risk import is_t4_mission, t4_first_iter_mode
        if not is_t4_mission(task_prompt=task_prompt):
            return False
        return t4_first_iter_mode() == "modify_law"
    except Exception:
        return False


def should_skip_layer3_param_search(
    metrics: Optional[Dict[str, Any]],
    task_prompt: str,
    optimization_history: Optional[List[Dict[str, Any]]] = None,
) -> bool:
    """
    Skip Expert/PPO when Layer2 metrics are 4/5: non-peak OK but PeakNy mean over limit.

    After a MODIFY_LAW round with a near-miss (e.g. 20.99g vs 20g), allow param tuning.
    """
    reqs = requirements_from_prompt(task_prompt)
    limit = peak_ny_limit(reqs)
    if not non_peak_constraints_met(metrics, reqs):
        return False
    if not peak_ny_over_limit(metrics, limit):
        return False
    if optimization_history:
        last_mode = str(optimization_history[-1].get("task_mode", "")).upper()
        if last_mode == "MODIFY_LAW" and peak_ny_near_miss(metrics, limit):
            return False
    return True


def should_use_modify_law(
    *,
    task_prompt: str = "",
    metrics: Optional[Dict[str, Any]] = None,
    optimization_history: Optional[List[Dict[str, Any]]] = None,
    reflection_feedback: str = "",
    planner_mode: Optional[str] = None,
) -> bool:
    """True when structural guidance-law change is the recommended path."""
    if planner_mode and str(planner_mode).upper() == "MODIFY_LAW":
        return True

    fb = (reflection_feedback or "").lower()
    if any(k in fb for k in ("modify_law", "修改制导律", "apn", "增广比例导引", "改 gf")):
        return True

    reqs = requirements_from_prompt(task_prompt)
    limit = peak_ny_limit(reqs)
    if not peak_ny_over_limit(metrics, limit):
        return False

    if not non_peak_constraints_met(metrics, reqs):
        return False

    # Layer2 / first Layer3 entry: hit/SEP/PM/BW OK, only PeakNy mean fails → MODIFY_LAW
    if not optimization_history:
        return True

    if optimization_history:
        last_mode = str(optimization_history[-1].get("task_mode", "")).upper()
        if last_mode == "MODIFY_LAW":
            if metrics and not non_peak_constraints_met(metrics, reqs):
                return False
            if peak_ny_near_miss(metrics, limit):
                return False

    stalls = count_tune_peakny_stalls(optimization_history, limit)
    if stalls >= TUNE_STALL_THRESHOLD and non_peak_constraints_met(metrics, reqs):
        return True

    return False


def resolve_planner_fallback_mode(
    *,
    task_prompt: str,
    reflection_feedback: str = "",
    optimization_history: Optional[List[Dict[str, Any]]] = None,
    last_metrics: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Safe fallback when Planner JSON is missing.

    Never returns REUSE_HISTORY for full optimization tasks (T4 5/5).
    """
    if should_use_modify_law(
        task_prompt=task_prompt,
        metrics=last_metrics,
        optimization_history=optimization_history,
        reflection_feedback=reflection_feedback,
    ):
        return "MODIFY_LAW"

    if is_optimization_task(task_prompt):
        return "TUNE_PARAMS"

    from multi_agent.integration.task_planner import _heuristic_mode

    hm = _heuristic_mode(task_prompt)
    if hm:
        return hm
    return "TUNE_PARAMS"


def normalize_planner_mode(
    llm_mode: str,
    *,
    task_prompt: str,
    reflection_feedback: str = "",
    optimization_history: Optional[List[Dict[str, Any]]] = None,
    last_metrics: Optional[Dict[str, Any]] = None,
) -> str:
    """Override unsafe LLM modes (e.g. REUSE_HISTORY on optimization tasks)."""
    mode = (llm_mode or "TUNE_PARAMS").upper()
    if mode == "REUSE_HISTORY":
        try:
            from multi_agent.integration.t4_low_risk import honor_reuse_history
            if honor_reuse_history():
                logger.info(
                    "[Planner] honor_reuse_history: keeping REUSE_HISTORY "
                    "(skip T4 first-iter MODIFY override)"
                )
                return "REUSE_HISTORY"
        except Exception:
            pass
    if mode == "REUSE_HISTORY" and is_optimization_task(task_prompt):
        mode = resolve_planner_fallback_mode(
            task_prompt=task_prompt,
            reflection_feedback=reflection_feedback,
            optimization_history=optimization_history,
            last_metrics=last_metrics,
        )

    if should_t4_first_iter_modify_law(task_prompt, optimization_history):
        return "MODIFY_LAW"

    if should_prefer_tune_over_modify(
        task_prompt=task_prompt,
        metrics=last_metrics,
        optimization_history=optimization_history,
        reflection_feedback=reflection_feedback,
    ):
        if mode == "TUNE_PARAMS":
            return "TUNE_PARAMS"

    if should_use_modify_law(
        task_prompt=task_prompt,
        metrics=last_metrics,
        optimization_history=optimization_history,
        reflection_feedback=reflection_feedback,
        planner_mode=mode,
    ):
        if should_prefer_tune_over_modify(
            task_prompt=task_prompt,
            metrics=last_metrics,
            optimization_history=optimization_history,
            reflection_feedback=reflection_feedback,
        ):
            return "TUNE_PARAMS" if llm_mode.upper() == "TUNE_PARAMS" else mode
        return "MODIFY_LAW"
    return mode


def resolve_task_mode(
    *,
    planner_mode: Optional[str],
    task_prompt: str,
    optimization_history: Optional[List[Dict[str, Any]]] = None,
    last_metrics: Optional[Dict[str, Any]] = None,
    reflection_feedback: str = "",
    forced_next_action: Optional[str] = None,
) -> str:
    """Final Layer2 mode before Hermes generate_matlab."""
    if forced_next_action == "modify_law":
        return "MODIFY_LAW"
    if forced_next_action == "tune_params":
        if should_use_modify_law(
            task_prompt=task_prompt,
            metrics=last_metrics,
            optimization_history=optimization_history,
            reflection_feedback=reflection_feedback,
            planner_mode=planner_mode,
        ) and not should_prefer_tune_over_modify(
            task_prompt=task_prompt,
            metrics=last_metrics,
            optimization_history=optimization_history,
            reflection_feedback=reflection_feedback,
        ):
            return "MODIFY_LAW"
        return "TUNE_PARAMS"

    return normalize_planner_mode(
        planner_mode or "TUNE_PARAMS",
        task_prompt=task_prompt,
        reflection_feedback=reflection_feedback,
        optimization_history=optimization_history,
        last_metrics=last_metrics,
    )


def resolve_next_action(
    *,
    reflection_next_action: str,
    task_prompt: str,
    metrics: Optional[Dict[str, Any]],
    optimization_history: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """Merge reflection LLM output with rule-based PeakNy mean policy."""
    action = (reflection_next_action or "").strip().lower()
    reqs = requirements_from_prompt(task_prompt)
    limit = peak_ny_limit(reqs)

    if metrics and all_requirements_satisfied(metrics, reqs, task_prompt):
        return "done"

    if should_use_modify_law(
        task_prompt=task_prompt,
        metrics=metrics,
        optimization_history=optimization_history,
    ):
        return "modify_law"

    if action in ("done", "modify_law", "tune_params"):
        if action == "done" and metrics and not all_requirements_satisfied(metrics, reqs, task_prompt):
            pny = get_peak_ny(metrics)
            if pny > limit:
                return "modify_law"
            return "tune_params"
        return action

    return "tune_params"


def apply_reflection_policy(
    parsed: Dict[str, Any],
    *,
    task_prompt: str,
    metrics: Optional[Dict[str, Any]],
    optimization_history: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Rule overlay on reflection JSON — PeakNy mean is authoritative for peak gate."""
    out = dict(parsed)
    reqs = requirements_from_prompt(task_prompt)
    limit = peak_ny_limit(reqs)
    pny_avg = get_peak_ny(metrics or {})

    if metrics and all_requirements_satisfied(metrics, reqs, task_prompt):
        out["needs_optimization"] = False
        out["next_action"] = "done"
        return out

    out["needs_optimization"] = True
    next_action = resolve_next_action(
        reflection_next_action=str(out.get("next_action", "")),
        task_prompt=task_prompt,
        metrics=metrics,
        optimization_history=optimization_history,
    )
    out["next_action"] = next_action

    from multi_agent.integration.t4_low_risk import apply_reflection_t4_overrides
    out = apply_reflection_t4_overrides(
        out,
        metrics=metrics,
        task_prompt=task_prompt,
        optimization_history=optimization_history,
    )
    next_action = str(out.get("next_action", next_action))

    if pny_avg > limit and next_action == "modify_law":
        note = (
            f"PeakNy(均值)={pny_avg:.2f}g 超过要求 {limit:.0f}g；"
            f"命中率/SEP/PM/BW 已接近达标，"
            f"应 MODIFY_LAW 改 gf()（APN/N 调度/限幅），勿继续纯 TUNE_PARAMS。"
        )
        existing = str(out.get("suggestion") or "")
        if "PeakNy(均值)" not in existing:
            out["suggestion"] = f"{existing}\n{note}".strip()

    return out


def extract_gf_body(content: str) -> str:
    """Extract normalized gf() function body from MATLAB source."""
    if not content:
        return ""
    pat = re.compile(
        r"^[ \t]*function\s+.{0,80}?=\s*gf\s*\(.*?\n(.*?)(?=^[ \t]*function\s|\Z)",
        re.MULTILINE | re.DOTALL | re.IGNORECASE,
    )
    m = pat.search(content)
    if not m:
        return ""
    body = m.group(0)
    body = re.sub(r"%[^\n]*", "", body)
    body = re.sub(r"\s+", " ", body).strip().lower()
    return body


def gf_body_hash(content: str) -> str:
    body = extract_gf_body(content)
    if not body:
        return ""
    return hashlib.sha256(body.encode("utf-8")).hexdigest()[:16]


def gf_bodies_differ(baseline: str, modified: str) -> bool:
    h0, h1 = gf_body_hash(baseline), gf_body_hash(modified)
    if not h0 or not h1:
        return True
    return h0 != h1


def gf_structural_features(content: str) -> Tuple[bool, List[str]]:
    """Return (has_apn_like_features, matched markers)."""
    text = (content or "").lower()
    found = [m for m in _APN_MARKERS if m.lower() in text]
    return bool(found), found


def verify_modify_law_result(
    new_content: str,
    baseline_content: str,
    *,
    require_structural: bool = False,
) -> Tuple[bool, str]:
    """
    Verify MODIFY_LAW actually changed gf().

    require_structural: when True (retry path), also require APN-like markers.
    """
    if not gf_bodies_differ(baseline_content, new_content):
        return False, "MODIFY_LAW 后 gf() 与种子脚本完全一致，疑似未改制导律"

    if require_structural:
        ok, markers = gf_structural_features(new_content)
        if not ok:
            return False, (
                "MODIFY_LAW 后 gf() 已变化但未检测到 APN/N 调度/限幅等结构特征"
            )
        return True, f"gf() 已修改，检测到特征: {', '.join(markers[:5])}"

    return True, "gf() 已相对种子脚本发生变化"


def metrics_summary_line(metrics: Dict[str, Any]) -> str:
    """Human-readable one-liner (PeakNy gate uses MC mean only)."""
    pny_avg = get_peak_ny(metrics)
    sep = get_sep(metrics, float("nan"))
    return (
        f"hit={metrics.get('hit_rate', 0):.1f}% "
        f"SEP={sep:.2f}m "
        f"PeakNy_avg={pny_avg:.2f}g "
        f"PM={metrics.get('pitch_PM', 0):.1f}° "
        f"BW={metrics.get('pitch_BW', 0):.1f}r/s"
    )


__all__ = [
    "PEAK_NY_NEAR_MISS_G",
    "TUNE_STALL_THRESHOLD",
    "apply_reflection_policy",
    "task_fully_satisfied",
    "count_tune_peakny_stalls",
    "extract_gf_body",
    "gf_bodies_differ",
    "gf_structural_features",
    "is_optimization_task",
    "metrics_summary_line",
    "normalize_planner_mode",
    "peak_ny_limit",
    "peak_ny_near_miss",
    "peak_ny_over_limit",
    "requirements_from_prompt",
    "should_skip_layer3_param_search",
    "resolve_next_action",
    "resolve_planner_fallback_mode",
    "resolve_task_mode",
    "should_use_modify_law",
    "verify_modify_law_result",
]
