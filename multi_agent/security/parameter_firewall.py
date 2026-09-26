"""
Pydantic-style business firewall for LLM-suggested tunable params and RL hyperparams.

All paths that accept Reflection / Expert / external JSON suggestions should
pass through these helpers before touching the simulator or PPO optimizer.
"""

from __future__ import annotations

import logging
import math
import re
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

from multi_agent.rl.matlab_rl_optimizer import ALL_TUNABLE_PARAM_SPECS

if TYPE_CHECKING:
    from multi_agent.rl.matlab_rl_optimizer import MatlabRLOptimizer

logger = logging.getLogger(__name__)


class ParameterFirewallViolation(ValueError):
    """Raised when an LLM suggestion contains disallowed or poison values."""


_FORBIDDEN_PARAM_KEYS = frozenset({
    "rho", "air_density", "density", "g", "gravity", "mass", "m", "thrust", "P",
    "temperature", "pressure", "altitude",
})

_POISON_VALUE_PATTERNS = (
    re.compile(r"-\s*\d+\.?\d*\s*(?:kg/m|m/s|Pa|N/)", re.I),
    re.compile(r"(?:density|rho|空气密度).{0,20}(?:<|=\s*)-\s*\d", re.I),
)


RL_HP_BOUNDS: Dict[str, Tuple[float, float]] = {
    "lr_actor": (1e-5, 1e-2),
    "lr_critic": (1e-5, 3e-2),
    "clip_ratio": (0.05, 0.4),
    "gamma": (0.90, 0.999),
    "max_episodes": (10, 150),
    "nmc_per_eval": (5, 50),
    "hit_rate_weight": (0.5, 10.0),
    "sep_low_bonus": (0.0, 8.0),
    "sep_mid_weight": (0.0, 5.0),
    "pm_bonus": (0.0, 4.0),
    "pm_penalty": (0.0, 8.0),
    "bw_bonus": (0.0, 4.0),
    "bw_penalty": (0.0, 8.0),
    "peak_ny_penalty": (0.0, 8.0),
}

RL_HP_TO_RW: Dict[str, str] = {
    "hit_rate_weight": "hit_rate",
    "sep_low_bonus": "sep_low_bonus",
    "sep_mid_weight": "sep_mid_weight",
    "pm_bonus": "pm_bonus",
    "pm_penalty": "pm_penalty",
    "bw_bonus": "bw_bonus",
    "bw_penalty": "bw_penalty",
    "peak_ny_penalty": "peak_ny_penalty",
}


def _scan_poison_keys(raw: Dict[str, Any]) -> List[str]:
    violations: List[str] = []
    for key in raw:
        if str(key).lower() in _FORBIDDEN_PARAM_KEYS:
            violations.append(f"forbidden key: {key}")
    return violations


def _scan_poison_values(raw: Dict[str, Any]) -> List[str]:
    violations: List[str] = []
    for key, val in raw.items():
        if isinstance(val, str):
            for pat in _POISON_VALUE_PATTERNS:
                if pat.search(val):
                    violations.append(f"suspicious value for {key}: {val!r}")
                    break
        elif isinstance(val, (int, float)) and math.isfinite(float(val)):
            if float(val) < 0 and str(key).lower() in ("rho", "air_density", "density", "mass", "m", "P", "thrust"):
                violations.append(f"negative physical quantity {key}={val}")
    return violations


def _coerce_finite_float(val: Any) -> Optional[float]:
    try:
        fv = float(val)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(fv):
        return None
    return fv


def clamp_tunable_params(
    raw: Dict[str, Any],
    *,
    strict: bool = False,
    fill_missing: bool = True,
) -> Dict[str, float]:
    """
    Validate and clamp LLM-suggested autopilot/guidance params.

    Returns a full 9-dim dict when *fill_missing* is True (Expert path).
    Raises ParameterFirewallViolation in *strict* mode on poison keys/values.
    """
    if not isinstance(raw, dict):
        raw = {}

    poison = _scan_poison_keys(raw) + _scan_poison_values(raw)
    if poison and strict:
        raise ParameterFirewallViolation("; ".join(poison))
    for msg in poison:
        logger.warning("[ParameterFirewall] %s — ignored", msg)

    out: Dict[str, float] = {}
    for key, spec in ALL_TUNABLE_PARAM_SPECS.items():
        if key not in raw:
            if fill_missing:
                out[key] = float(spec["nominal"])
            continue
        fv = _coerce_finite_float(raw[key])
        if fv is None:
            if strict:
                raise ParameterFirewallViolation(f"non-numeric tunable param {key}={raw[key]!r}")
            logger.warning("[ParameterFirewall] bad value for %s=%r — using nominal", key, raw[key])
            fv = float(spec["nominal"])
        if fv < spec["min"] or fv > spec["max"]:
            logger.debug(
                "[ParameterFirewall] clamped %s: %.4f → [%.4f, %.4f]",
                key, fv, spec["min"], spec["max"],
            )
        out[key] = max(spec["min"], min(spec["max"], fv))
    return out


def sanitize_rl_hyperparams(raw: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    """
    Validate RL hyperparameter suggestions from ReflectionAgent.

    Returns (sanitized_dict, warning_messages). Unknown keys are dropped.
    """
    warnings: List[str] = []
    if not isinstance(raw, dict):
        return {}, ["rl_hyperparams is not a dict — ignored"]

    warnings.extend(_scan_poison_keys(raw))
    warnings.extend(_scan_poison_values(raw))

    sanitized: Dict[str, Any] = {}
    for key, val in raw.items():
        if key not in RL_HP_BOUNDS:
            warnings.append(f"unknown rl_hyperparam key dropped: {key}")
            continue
        lo, hi = RL_HP_BOUNDS[key]
        if key in ("max_episodes", "nmc_per_eval"):
            fv = _coerce_finite_float(val)
            if fv is None:
                warnings.append(f"invalid {key}={val!r} — skipped")
                continue
            sanitized[key] = int(max(lo, min(hi, int(fv))))
        else:
            fv = _coerce_finite_float(val)
            if fv is None:
                warnings.append(f"invalid {key}={val!r} — skipped")
                continue
            sanitized[key] = max(lo, min(hi, fv))
    return sanitized, warnings


def apply_rl_hyperparams_to_optimizer(
    optimizer: "MatlabRLOptimizer",
    raw: Dict[str, Any],
) -> Dict[str, Any]:
    """Sanitize and apply Reflection-suggested RL hyperparams to an optimizer."""
    sanitized, warnings = sanitize_rl_hyperparams(raw)
    for w in warnings:
        print(f"  [ParameterFirewall] ⚠ {w}")
        logger.warning("[ParameterFirewall] %s", w)

    for key in ("lr_actor", "lr_critic", "clip_ratio", "gamma"):
        if key in sanitized:
            setattr(optimizer, key, sanitized[key])

    for ext_key, int_key in RL_HP_TO_RW.items():
        if ext_key in sanitized:
            optimizer._rw[int_key] = sanitized[ext_key]

    if "max_episodes" in sanitized:
        optimizer.max_episodes = sanitized["max_episodes"]
    if "nmc_per_eval" in sanitized:
        optimizer.nmc_per_eval = sanitized["nmc_per_eval"]

    if sanitized:
        print(f"  [ParameterFirewall] Applied RL hyperparams: {sanitized}")
    return sanitized
