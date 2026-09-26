"""Multi-criteria metric constraints for RL / local search selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, FrozenSet, Optional, Tuple

from multi_agent.integration.judgment_agent import _rule_based_check
from multi_agent.metric_contract import HIT_RATE_MIN, SEP_MAX
from multi_agent.rl.metric_utils import (
    get_peak_ny,
    get_peak_ny_limit,
    get_peak_ny_max,
    get_sep,
)


@dataclass(frozen=True)
class MetricLockMargins:
    """Slack used before locking a metric during CLS (reduces MC flip-flop)."""

    sep_m: float = 0.4
    peak_ny_g: float = 0.35
    hit_pct: float = 1.0
    pm_deg: float = 1.5
    bw_rad_s: float = 2.0


# Tunable params most influential per metric family (CLS directed perturb).
METRIC_PARAM_GROUPS: Dict[str, Tuple[str, ...]] = {
    "hit": ("N_pn", "w2", "zeta2"),
    "sep": ("w2", "zeta2", "tao2", "N_pn"),
    "peak": ("N_pn", "ny_lim", "w3", "zeta3"),
    "pm": ("w1", "zeta1", "tao1"),
    "bw": ("w1", "zeta1", "tao1", "w2"),
}


def metric_satisfaction_flags(
    metrics: Optional[Dict[str, float]],
    reqs: Dict[str, Any],
) -> Dict[str, bool]:
    """Per-metric pass map: hit / sep / peak / pm / bw."""
    m = metrics or {}
    seed_metrics = m.get("mc_seed_results") if isinstance(m, dict) else None
    if isinstance(seed_metrics, list) and seed_metrics:
        combined: Dict[str, bool] = {}
        for item in seed_metrics:
            if not isinstance(item, dict):
                continue
            flags_i = metric_satisfaction_flags(item, reqs)
            for key, ok in flags_i.items():
                combined[key] = combined.get(key, True) and ok
        if combined:
            return combined

    flags: Dict[str, bool] = {}
    if "hit_rate_min" in reqs:
        try:
            flags["hit"] = float(m.get("hit_rate") or 0.0) >= float(reqs["hit_rate_min"])
        except (TypeError, ValueError):
            flags["hit"] = False
    if "sep_max" in reqs:
        flags["sep"] = get_sep(m, 1e6) <= float(reqs["sep_max"])
    if "peak_ny_max" in reqs or "peak_ny_mean_max" in reqs:
        from multi_agent.rl.metric_utils import check_peak_ny, normalize_peak_ny_requirements

        ok_peak, _ = check_peak_ny(m, normalize_peak_ny_requirements(reqs))
        flags["peak"] = ok_peak
    if "pm_min" in reqs:
        try:
            pm = float(m.get("pitch_PM") or 0.0)
        except (TypeError, ValueError):
            pm = 0.0
        try:
            pm_lo_seen = float(m.get("pitch_PM_min", pm) or 0.0)
            pm_hi_seen = float(m.get("pitch_PM_max", pm) or 0.0)
        except (TypeError, ValueError):
            pm_lo_seen = pm
            pm_hi_seen = pm
        lo = float(reqs["pm_min"])
        if "pm_max" in reqs:
            hi = float(reqs["pm_max"])
            flags["pm"] = lo <= pm_lo_seen and pm_hi_seen <= hi
        else:
            flags["pm"] = pm_lo_seen >= lo
    if "bw_min" in reqs and "bw_max" in reqs:
        try:
            bw = float(m.get("pitch_BW") or 0.0)
        except (TypeError, ValueError):
            bw = 0.0
        try:
            bw_lo_seen = float(m.get("pitch_BW_min", bw) or 0.0)
            bw_hi_seen = float(m.get("pitch_BW_max", bw) or 0.0)
        except (TypeError, ValueError):
            bw_lo_seen = bw
            bw_hi_seen = bw
        flags["bw"] = (
            float(reqs["bw_min"]) <= bw_lo_seen
            and bw_hi_seen <= float(reqs["bw_max"])
        )
    return flags


def count_satisfied_flags(flags: Dict[str, bool]) -> int:
    return sum(1 for v in flags.values() if v)


def flags_imply_lock(
    flags: Dict[str, bool],
    metrics: Dict[str, float],
    reqs: Dict[str, Any],
    margins: MetricLockMargins,
) -> Dict[str, bool]:
    """Metrics that pass with margin — safe to lock against regression."""
    locked: Dict[str, bool] = {}
    if flags.get("hit") and "hit_rate_min" in reqs:
        try:
            hit = float(metrics.get("hit_rate") or 0.0)
        except (TypeError, ValueError):
            hit = 0.0
        locked["hit"] = hit >= float(reqs["hit_rate_min"]) + margins.hit_pct
    if flags.get("sep") and "sep_max" in reqs:
        locked["sep"] = get_sep(metrics, 1e6) <= float(reqs["sep_max"]) - margins.sep_m
    if flags.get("peak"):
        limit = get_peak_ny_limit(reqs)
        locked["peak"] = get_peak_ny(metrics) <= limit - margins.peak_ny_g
    if flags.get("pm") and "pm_min" in reqs and "pm_max" in reqs:
        try:
            pm = float(metrics.get("pitch_PM") or 0.0)
        except (TypeError, ValueError):
            pm = 0.0
        try:
            pm_lo_seen = float(metrics.get("pitch_PM_min", pm) or 0.0)
            pm_hi_seen = float(metrics.get("pitch_PM_max", pm) or 0.0)
        except (TypeError, ValueError):
            pm_lo_seen = pm
            pm_hi_seen = pm
        lo, hi = float(reqs["pm_min"]), float(reqs["pm_max"])
        locked["pm"] = (
            (lo + margins.pm_deg) <= pm_lo_seen
            and pm_hi_seen <= (hi - margins.pm_deg)
        )
    if flags.get("bw") and "bw_min" in reqs and "bw_max" in reqs:
        try:
            bw = float(metrics.get("pitch_BW") or 0.0)
        except (TypeError, ValueError):
            bw = 0.0
        try:
            bw_lo_seen = float(metrics.get("pitch_BW_min", bw) or 0.0)
            bw_hi_seen = float(metrics.get("pitch_BW_max", bw) or 0.0)
        except (TypeError, ValueError):
            bw_lo_seen = bw
            bw_hi_seen = bw
        lo, hi = float(reqs["bw_min"]), float(reqs["bw_max"])
        locked["bw"] = (
            (lo + margins.bw_rad_s) <= bw_lo_seen
            and bw_hi_seen <= (hi - margins.bw_rad_s)
        )
    return locked


def violates_locked_metrics(
    candidate_flags: Dict[str, bool],
    locked: FrozenSet[str],
) -> bool:
    """True if candidate breaks any previously locked metric."""
    for key in locked:
        if not candidate_flags.get(key, False):
            return True
    return False


def metric_overshoot(
    metric_key: str,
    metrics: Dict[str, float],
    reqs: Dict[str, Any],
) -> float:
    """Distance past requirement for one metric family (0 = satisfied)."""
    if metric_key == "hit":
        try:
            hit = float(metrics.get("hit_rate") or 0.0)
        except (TypeError, ValueError):
            hit = 0.0
        return max(0.0, float(reqs.get("hit_rate_min", 0.0)) - hit)
    if metric_key == "sep":
        return max(0.0, get_sep(metrics, 1e6) - float(reqs.get("sep_max", 999.0)))
    if metric_key == "peak":
        return max(0.0, get_peak_ny(metrics) - get_peak_ny_limit(reqs))
    if metric_key == "pm":
        try:
            pm = float(metrics.get("pitch_PM") or 0.0)
        except (TypeError, ValueError):
            pm = 0.0
        try:
            pm_lo_seen = float(metrics.get("pitch_PM_min", pm) or 0.0)
            pm_hi_seen = float(metrics.get("pitch_PM_max", pm) or 0.0)
        except (TypeError, ValueError):
            pm_lo_seen = pm
            pm_hi_seen = pm
        lo = float(reqs.get("pm_min", 0.0))
        hi = float(reqs.get("pm_max", 999.0))
        return max(0.0, lo - pm_lo_seen, pm_hi_seen - hi)
    if metric_key == "bw":
        try:
            bw = float(metrics.get("pitch_BW") or 0.0)
        except (TypeError, ValueError):
            bw = 0.0
        try:
            bw_lo_seen = float(metrics.get("pitch_BW_min", bw) or 0.0)
            bw_hi_seen = float(metrics.get("pitch_BW_max", bw) or 0.0)
        except (TypeError, ValueError):
            bw_lo_seen = bw
            bw_hi_seen = bw
        lo = float(reqs.get("bw_min", 0.0))
        hi = float(reqs.get("bw_max", 999.0))
        return max(0.0, lo - bw_lo_seen, bw_hi_seen - hi)
    return 0.0


def eligible_partial_bw_band(
    metrics: Dict[str, float],
    flags: Dict[str, bool],
    reqs: Dict[str, Any],
    *,
    focus_metric_keys: Optional[Tuple[str, ...]] = None,
    enabled: bool = True,
) -> bool:
    """Partial ★ candidates must keep BW in [bw_min, bw_max], with exceptions.

    - Rejects low-BW basin (e.g. BW<20 after CLS overshoot).
    - Allows BW above max when BW is the sole failing metric (4/5 plateau).
    - BW-only finish pass may start outside the band.
    """
    if not enabled:
        return True
    if focus_metric_keys == ("bw",):
        return True
    if "bw_min" not in reqs or "bw_max" not in reqs:
        return True
    try:
        bw = float(metrics.get("pitch_BW") or 0.0)
    except (TypeError, ValueError):
        return False
    try:
        bw_lo_seen = float(metrics.get("pitch_BW_min", bw) or 0.0)
        bw_hi_seen = float(metrics.get("pitch_BW_max", bw) or 0.0)
    except (TypeError, ValueError):
        return False
    lo = float(reqs["bw_min"])
    hi = float(reqs["bw_max"])
    if bw_lo_seen < lo:
        return False
    if bw_hi_seen <= hi:
        return True
    failed = failed_metric_keys(flags)
    return len(failed) == 1 and failed[0] == "bw"


def partial_rank_key(
    metrics: Dict[str, float],
    flags: Dict[str, bool],
    reqs: Dict[str, Any],
    fitness: float = 0.0,
) -> Tuple[float, ...]:
    """Rank partial candidates — when one metric fails, prioritize fixing it."""
    failed = failed_metric_keys(flags)
    if len(failed) == 1:
        return (
            metric_overshoot(failed[0], metrics, reqs),
            get_sep(metrics, 1e6),
            -float(fitness),
        )
    mean_max = get_peak_ny_limit(reqs)
    pny_mean = get_peak_ny(metrics)
    if pny_mean <= 0:
        pny_mean = 1e6
    mean_over = max(0.0, pny_mean - mean_max)
    sep = get_sep(metrics, 1e6)
    bw_over = metric_overshoot("bw", metrics, reqs) if "bw" in flags else 0.0
    return (mean_over, bw_over, sep, -float(fitness))


def is_better_partial_progress(
    metrics_a: Dict[str, float],
    flags_a: Dict[str, bool],
    fitness_a: float,
    metrics_b: Optional[Dict[str, float]],
    flags_b: Optional[Dict[str, bool]],
    fitness_b: float,
    reqs: Optional[Dict[str, Any]] = None,
) -> bool:
    """Prefer more satisfied metrics, then partial rank (failed-metric aware)."""
    if metrics_b is None or flags_b is None:
        return True
    ca, cb = count_satisfied_flags(flags_a), count_satisfied_flags(flags_b)
    if ca != cb:
        return ca > cb
    if reqs:
        return partial_rank_key(metrics_a, flags_a, reqs, fitness_a) < partial_rank_key(
            metrics_b, flags_b, reqs, fitness_b
        )
    return is_better_constrained(metrics_a, fitness_a, metrics_b, fitness_b)


def failed_metric_keys(flags: Dict[str, bool]) -> Tuple[str, ...]:
    return tuple(k for k, ok in flags.items() if not ok)


def params_for_failed_metrics(failed: Tuple[str, ...]) -> Tuple[str, ...]:
    out: list[str] = []
    seen: set[str] = set()
    for mk in failed:
        for p in METRIC_PARAM_GROUPS.get(mk, ()):
            if p not in seen:
                seen.add(p)
                out.append(p)
    return tuple(out) if out else tuple(
        p for group in METRIC_PARAM_GROUPS.values() for p in group
    )


def constraints_satisfied(
    metrics: Dict[str, float],
    reqs: Dict[str, Any],
    task_prompt: str = "",
) -> bool:
    """True when all parsed task requirements are met."""
    from multi_agent.integration.judgment_agent import all_requirements_satisfied

    return all_requirements_satisfied(metrics, reqs, task_prompt)


def constrained_rank_key(
    metrics: Dict[str, float],
    fitness: float = 0.0,
) -> Tuple[float, float, float, float]:
    """Sort key for constrained candidates — lower is better.

    Priority: mean overshoot → SEP → fitness (negated).
    """
    mean_max = get_peak_ny_limit({})
    pny_mean = get_peak_ny(metrics)
    if pny_mean <= 0:
        pny_mean = 1e6
    mean_over = max(0.0, pny_mean - mean_max)
    sep = get_sep(metrics, 1e6)
    return (mean_over, sep, -float(fitness))


def is_better_constrained(
    metrics_a: Dict[str, float],
    fitness_a: float,
    metrics_b: Optional[Dict[str, float]],
    fitness_b: float,
) -> bool:
    """Return True if *a* strictly dominates *b* under constrained rank."""
    if metrics_b is None:
        return True
    return constrained_rank_key(metrics_a, fitness_a) < constrained_rank_key(
        metrics_b, fitness_b
    )


def is_borderline_hit_sep_ok(
    metrics: Dict[str, float],
    *,
    hit_min_pct: float = HIT_RATE_MIN,
    sep_max_m: float = SEP_MAX,
) -> bool:
    """True when hit/SEP meet requirements (peak/PM/BW may still fail)."""
    try:
        hit = float(metrics.get("hit_rate") or 0.0)
    except (TypeError, ValueError):
        hit = 0.0
    sep = get_sep(metrics, 1e6)
    if hit < hit_min_pct:
        return False
    if sep > sep_max_m:
        return False
    return True


def should_adopt_cls_result(
    prior_metrics: Optional[Dict[str, float]],
    prior_fitness: float,
    cls_metrics: Optional[Dict[str, float]],
    cls_fitness: float,
    *,
    reqs: Optional[Dict[str, Any]] = None,
    task_prompt: str = "",
) -> bool:
    """True when CLS output should replace Expert/PPO best (mean-based rank, not peak_max)."""
    from multi_agent.rl.matlab_rl_optimizer import is_reliable_sim_metrics

    if not cls_metrics:
        return False
    cls_ok = is_reliable_sim_metrics(cls_metrics)
    prior_ok = is_reliable_sim_metrics(prior_metrics) if prior_metrics else False
    if not cls_ok:
        return False
    if not prior_metrics or not prior_ok:
        return True
    if reqs and constraints_satisfied(cls_metrics, reqs, task_prompt):
        return True
    if reqs and constraints_satisfied(prior_metrics, reqs, task_prompt):
        return False
    return is_better_constrained(cls_metrics, cls_fitness, prior_metrics, prior_fitness)


def is_better_borderline_peak(
    metrics_a: Dict[str, float],
    metrics_b: Optional[Dict[str, float]],
) -> bool:
    """Lower mean peak then SEP wins among borderline hit/SEP-ok candidates."""
    if metrics_b is None:
        return True
    mean_a = get_peak_ny(metrics_a)
    mean_b = get_peak_ny(metrics_b)
    if mean_a != mean_b:
        return mean_a < mean_b
    return get_sep(metrics_a, 1e6) < get_sep(metrics_b, 1e6)


def resolve_requirements(
    metric_requirements: Optional[Dict[str, Any]],
    task_prompt: Optional[str],
    judgment_agent: Any = None,
) -> Dict[str, Any]:
    """Merge explicit requirements with JudgmentAgent / prompt parsing."""
    from multi_agent.rl.metric_utils import normalize_peak_ny_requirements

    from multi_agent.integration.judgment_agent import (
        merge_requirements,
        resolve_task_requirements,
    )

    if metric_requirements:
        base = (
            resolve_task_requirements(task_prompt or "", judgment_agent)
            if task_prompt
            else {}
        )
        return merge_requirements(base, metric_requirements)
    if task_prompt:
        return resolve_task_requirements(task_prompt, judgment_agent)
    if judgment_agent is not None and getattr(judgment_agent, "_reqs", None):
        return normalize_peak_ny_requirements(dict(judgment_agent._reqs))
    return normalize_peak_ny_requirements({})
