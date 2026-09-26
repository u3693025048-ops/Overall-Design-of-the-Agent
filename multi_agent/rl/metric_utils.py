"""Shared metric read helpers for RL optimizers."""

from __future__ import annotations

import math
from typing import Any, Dict, List, MutableMapping, Optional, Tuple

from multi_agent.metric_contract import PEAK_NY_MAX

DEFAULT_PEAK_NY_MAX = PEAK_NY_MAX


def get_peak_ny(metrics: Dict[str, float]) -> float:
    """Unified peak overload read — accepts ``peak_ny`` or legacy ``peak_n``."""
    raw = metrics.get("peak_ny", metrics.get("peak_n", 0.0))
    try:
        v = float(raw)
    except (TypeError, ValueError):
        return 0.0
    if math.isnan(v) or math.isinf(v):
        return 0.0
    return v


def get_peak_ny_max(metrics: Dict[str, float]) -> float:
    """Peak overload worst-case read — prefers MC ``max=XXg`` over mean."""
    for key in ("peak_ny_max", "peak_n_max"):
        raw = metrics.get(key)
        if raw is None:
            continue
        try:
            v = float(raw)
        except (TypeError, ValueError):
            continue
        if not math.isnan(v) and not math.isinf(v) and v > 0.0:
            return v
    return get_peak_ny(metrics)


def get_sep(metrics: Dict[str, float], default: float = float("nan")) -> float:
    """Unified SEP read — accepts ``SEP`` or legacy ``miss_distance``."""
    raw = metrics.get("SEP", metrics.get("miss_distance", default))
    try:
        v = float(raw)
    except (TypeError, ValueError):
        return default
    return v


def normalize_peak_ny_aliases(metrics: MutableMapping[str, Any]) -> Dict[str, Any]:
    """Ensure peak_ny / peak_ny_max / peak_n aliases are present and consistent."""
    if not metrics:
        return {}
    m = dict(metrics)
    pny = get_peak_ny(m)
    pny_max = get_peak_ny_max(m)
    if pny_max <= 0 and pny > 0:
        m["peak_ny_max"] = pny
        m["peak_n_max"] = pny
        pny_max = pny
    if pny <= 0 and pny_max > 0:
        m["peak_ny"] = pny_max
        m["peak_n"] = pny_max
    elif pny > 0:
        m.setdefault("peak_ny", pny)
        m.setdefault("peak_n", pny)
    if pny_max > 0:
        m.setdefault("peak_ny_max", pny_max)
        m.setdefault("peak_n_max", pny_max)
    if "SEP" not in m and "miss_distance" in m:
        m["SEP"] = m["miss_distance"]
    return m


def default_peak_ny_limit() -> float:
    """Configured PeakNy mean upper bound (g)."""
    try:
        from multi_agent.config_loader import get_config

        cfg = get_config().matlab_rl_optimizer
        return float(
            getattr(cfg, "peak_ny_mean_max", None)
            or getattr(cfg, "peak_n_max", DEFAULT_PEAK_NY_MAX)
        )
    except Exception:
        return DEFAULT_PEAK_NY_MAX


def normalize_peak_ny_requirements(reqs: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure PeakNy limit keys: ``peak_ny_max`` / ``peak_ny_mean_max`` (same value).

    Task prompt ``PeakNy <= 20 g`` maps to the MC *mean* overload only.
    ``peak_ny_max`` in metrics remains informational and is not gated separately.
    """
    out = dict(reqs or {})
    cfg_limit = default_peak_ny_limit()
    if "peak_ny_mean_max" not in out:
        if "peak_ny_max" in out:
            out["peak_ny_mean_max"] = float(out["peak_ny_max"])
        else:
            out["peak_ny_mean_max"] = cfg_limit
    if "peak_ny_max" not in out:
        out["peak_ny_max"] = float(out["peak_ny_mean_max"])
    out.pop("peak_ny_hard_max", None)
    return out


def get_peak_ny_limit(reqs: Optional[Dict[str, Any]] = None) -> float:
    norm = normalize_peak_ny_requirements(reqs or {})
    return float(norm["peak_ny_mean_max"])


def peak_ny_satisfied(
    metrics: Dict[str, float],
    reqs: Optional[Dict[str, Any]] = None,
) -> bool:
    ok, _ = check_peak_ny(metrics, reqs)
    return ok


def check_peak_ny(
    metrics: Dict[str, float],
    reqs: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, List[str]]:
    """PeakNy gate: MC mean (``peak_ny``) <= ``peak_ny_max`` / ``peak_ny_mean_max``."""
    limit = get_peak_ny_limit(reqs)
    pny_avg = get_peak_ny(metrics)
    if pny_avg <= 0.0:
        return False, ["PeakNy 数据不可用（跳过约束）"]

    ok = pny_avg <= limit
    reason = (
        f"PeakNy(平均值) {pny_avg:.2f}g {'<=' if ok else '>'} {limit:.2f}g"
        + (" [OK]" if ok else " [NG]")
    )
    return ok, [reason]
