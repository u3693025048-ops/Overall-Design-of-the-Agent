"""Bootstrap synthetic MC seed results from a single real seed evaluation.

Used when ``mc_and_bootstrap_fill`` is enabled: only the first listed seed is
run in MATLAB; remaining seeds are filled by resampling trajectory vectors
(and small seed-dependent metric noise when vectors are absent) so AND
aggregates and per-seed records stay structurally complete.
"""

from __future__ import annotations

import math
import os
import random
from typing import Any, Dict, List, Optional, Sequence


def bootstrap_fill_enabled(default: bool = True) -> bool:
    """Resolve bootstrap-fill switch from env then config."""
    env = os.environ.get("GSM_MC_AND_BOOTSTRAP", "").strip().lower()
    if env in ("0", "false", "no", "off"):
        return False
    if env in ("1", "true", "yes", "on"):
        return True
    try:
        from multi_agent.config_loader import get_config

        return bool(
            getattr(get_config().simulation, "mc_and_bootstrap_fill", default)
        )
    except Exception:
        return bool(default)


def _as_float_list(values: Any) -> List[float]:
    if not values:
        return []
    out: List[float] = []
    for item in values:
        try:
            out.append(float(item))
        except (TypeError, ValueError):
            continue
    return out


def _median(vals: Sequence[float]) -> float:
    ordered = sorted(float(v) for v in vals)
    if not ordered:
        return float("inf")
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[mid])
    return 0.5 * (float(ordered[mid - 1]) + float(ordered[mid]))


def _resample(vals: Sequence[float], n: int, rng: random.Random) -> List[float]:
    if not vals:
        return []
    return [float(rng.choice(list(vals))) for _ in range(max(1, int(n)))]


def _noisy_scalar(value: float, rng: random.Random, *, rel_std: float = 0.03) -> float:
    base = float(value)
    if not math.isfinite(base):
        return base
    # Keep sign/scale; clamp relative jitter.
    jitter = rng.gauss(0.0, abs(base) * rel_std + 1e-9)
    return float(base + jitter)


def bootstrap_seed_result(
    base: Dict[str, Any],
    *,
    target_seed: int,
    source_seed: int,
    split: str,
    kill_radius: float = 10.0,
    n_runs: Optional[int] = None,
) -> Dict[str, Any]:
    """Build a synthetic per-seed MC record from ``base`` via bootstrap sampling."""
    from multi_agent.metric_contract import constraint_violations

    if not bool(base.get("success", False)):
        err = str(base.get("error") or base.get("error_msg") or "source seed failed")
        return {
            "success": False,
            "feasible": False,
            "hit_rate": 0.0,
            "SEP": 1e9,
            "peak_nz": 1e9,
            "peak_ny": 1e9,
            "PM": 0.0,
            "BW": 0.0,
            "miss_distances": [],
            "nz_peaks": [],
            "constraint_violations": {"bootstrap_source_failed": 1.0},
            "sim_time_sec": 0.0,
            "error": err,
            "error_msg": err,
            "mc_seed": int(target_seed),
            "split": split,
            "synthetic": True,
            "bootstrap_from": int(source_seed),
            "bootstrap_method": "copy_failure",
        }

    rng = random.Random((int(source_seed) * 1_000_003) ^ (int(target_seed) * 97_351))
    misses = _as_float_list(base.get("miss_distances"))
    peaks = _as_float_list(base.get("nz_peaks"))
    n = int(n_runs) if n_runs and int(n_runs) > 0 else (len(misses) or 1)

    if misses:
        sampled_miss = _resample(misses, n, rng)
        hit_rate = (
            sum(v < float(kill_radius) for v in sampled_miss) / len(sampled_miss) * 100.0
        )
        sep = _median(sampled_miss)
    else:
        sampled_miss = []
        hit_rate = _noisy_scalar(float(base.get("hit_rate", 0.0)), rng, rel_std=0.01)
        hit_rate = max(0.0, min(100.0, hit_rate))
        sep = max(0.0, _noisy_scalar(float(base.get("SEP", 1e9)), rng, rel_std=0.04))

    if peaks:
        sampled_peaks = _resample(peaks, len(peaks) if len(peaks) >= n else n, rng)
        peak_nz = sum(sampled_peaks) / len(sampled_peaks)
    else:
        sampled_peaks = []
        peak_nz = max(
            0.0,
            _noisy_scalar(
                float(base.get("peak_nz", base.get("peak_ny", 0.0))), rng, rel_std=0.04
            ),
        )

    pm = _noisy_scalar(float(base.get("PM", base.get("pitch_PM", 0.0))), rng, rel_std=0.02)
    bw = max(
        0.0,
        _noisy_scalar(float(base.get("BW", base.get("pitch_BW", 0.0))), rng, rel_std=0.03),
    )
    feasible, violations = constraint_violations(hit_rate, sep, peak_nz, pm, bw)

    return {
        "success": True,
        "feasible": bool(feasible),
        "hit_rate": float(hit_rate),
        "SEP": float(sep),
        "peak_nz": float(peak_nz),
        "peak_ny": float(peak_nz),
        "PM": float(pm),
        "BW": float(bw),
        "miss_distances": sampled_miss,
        "nz_peaks": sampled_peaks,
        "constraint_violations": violations,
        "sim_time_sec": 0.0,
        "error": None,
        "error_msg": None,
        "mc_seed": int(target_seed),
        "split": split,
        "synthetic": True,
        "bootstrap_from": int(source_seed),
        "bootstrap_method": "resample_vectors" if misses else "noisy_scalars",
    }
