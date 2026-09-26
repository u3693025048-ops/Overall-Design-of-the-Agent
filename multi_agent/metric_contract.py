"""Canonical metric requirements used across simulation, reward, and memory.

The current experiment protocol uses five hard constraints:
hit_rate >= 92%, SEP <= 7 m, PeakNy <= 20 g, PM in [45, 70] deg,
and BW in [20, 85] rad/s.
"""

from __future__ import annotations

from typing import Dict, Tuple


HIT_RATE_MIN = 92.0
SEP_MAX = 7.0
PEAK_NY_MAX = 20.0
PM_MIN = 45.0
PM_MAX = 70.0
BW_MIN = 20.0
BW_MAX = 85.0


DEFAULT_METRIC_REQUIREMENTS: Dict[str, float] = {
    "hit_rate_min": HIT_RATE_MIN,
    "sep_max": SEP_MAX,
    "peak_ny_max": PEAK_NY_MAX,
    "peak_ny_mean_max": PEAK_NY_MAX,
    "pm_min": PM_MIN,
    "pm_max": PM_MAX,
    "bw_min": BW_MIN,
    "bw_max": BW_MAX,
}


def constraint_violations(
    hit_rate: float,
    sep: float,
    peak_ny: float,
    pm: float,
    bw: float,
) -> Tuple[bool, Dict[str, float]]:
    """Return hard-constraint feasibility and non-negative violations."""
    violations = {
        "hit_rate": max(0.0, HIT_RATE_MIN - float(hit_rate)),
        "SEP": max(0.0, float(sep) - SEP_MAX),
        "peak_nz": max(0.0, float(peak_ny) - PEAK_NY_MAX),
        "PM_low": max(0.0, PM_MIN - float(pm)),
        "PM_high": max(0.0, float(pm) - PM_MAX),
        "BW_low": max(0.0, BW_MIN - float(bw)),
        "BW_high": max(0.0, float(bw) - BW_MAX),
    }
    return all(v <= 0.0 for v in violations.values()), violations
