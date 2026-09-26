#!/usr/bin/env python3
"""
MATLAB Guidance System RL Optimizer

Dynamically extracts parameters from generated MATLAB scripts (based on the
MC_gongkuang_simulation_robust_all.m template), defines state/action spaces,
implements a guidance-specific reward function, runs PPO-style optimization,
and writes best parameters back to ParameterExperience.
"""

import re
import os
import asyncio
import copy
import json
import locale
import logging
import subprocess
import time
import math
import uuid
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np

from multi_agent.simulation.guidance_simulator import (
    OCTAVE_BATCH_FLAGS,
    DEFAULT_SUBPROCESS_TIMEOUT_SEC,
    build_octave_eval_string,
)
from multi_agent.rl.feasible_region_explorer import (
    FeasibleRegionExplorer,
    FeasibleRegionRewardModifier,
)
from multi_agent.rl.adaptive_exploration import AdaptiveExplorationController
from multi_agent.rl.metric_constraints import (
    constraints_satisfied,
    is_better_borderline_peak,
    is_better_constrained,
    is_borderline_hit_sep_ok,
    resolve_requirements,
)
from multi_agent.rl.episode_log import EpisodeJsonlLogger
from multi_agent.rl.ppo_checkpoint import PPOCheckpointManager
from multi_agent.rl.rolling_stats import RollingEpisodeStats
from multi_agent.config_loader import FREConfig, FREPresetConfig
from multi_agent.logging.log_verbosity import is_verbose, should_log_rl_episode
from multi_agent.metric_contract import (
    BW_MAX,
    BW_MIN,
    HIT_RATE_MIN,
    PEAK_NY_MAX,
    PM_MAX,
    PM_MIN,
    SEP_MAX,
)
# AMRO (AdaptiveActionModifier) removed: action tampering broke PPO importance sampling.

logger = logging.getLogger(__name__)

PPO_RATIO_LOG_CLIP = 10.0


def ppo_probability_ratio(
    new_logprob: float,
    old_logprob: float,
    log_clip: float = PPO_RATIO_LOG_CLIP,
) -> float:
    """PPO importance-sampling ratio from log-probability difference.

    Matches Eq. (30) and the implementation:

        r_t(θ) = exp( clip( log π_θ(a_t|s_t) − log π_{θ_old}(a_t|s_t), −L, L ) )

    The clip is only a numerical-stability bound on the log-difference
    (prevents overflow of ``exp``); the PPO surrogate then clips ``r_t``
    itself to ``[1−ε, 1+ε]``.
    """
    delta = float(new_logprob) - float(old_logprob)
    bound = abs(float(log_clip))
    return math.exp(min(max(delta, -bound), bound))


# ─────────────────────────────────────────────────────────────────────────────
# Steep Gradient Detector (for T4 narrow feasible region)
# ─────────────────────────────────────────────────────────────────────────────

class SteepGradientDetector:
    """检测参数空间中的梯度陡峭区域"""
    
    def __init__(self, window_size=5, threshold=2.0):
        """
        Args:
            window_size: 用于计算梯度的历史窗口大小
            threshold: 梯度阈值（奖励变化/参数变化）
        """
        self.window_size = window_size
        self.threshold = threshold
        self.reward_history = []
        self.param_history = []
    
    def is_steep_gradient(self):
        """检测是否在梯度陡峭区域"""
        if len(self.reward_history) < self.window_size:
            return False
        
        recent_rewards = self.reward_history[-self.window_size:]
        recent_params = self.param_history[-self.window_size:]
        
        # 计算奖励变化率
        reward_changes = []
        for i in range(1, len(recent_rewards)):
            change = abs(recent_rewards[i] - recent_rewards[i-1])
            reward_changes.append(change)
        
        if not reward_changes:
            return False
        
        avg_reward_change = sum(reward_changes) / len(reward_changes)
        
        # 计算参数变化率
        param_changes = []
        for i in range(1, len(recent_params)):
            change = 0.0
            for k in recent_params[i]:
                change += abs(recent_params[i][k] - recent_params[i-1][k])
            param_changes.append(change)
        
        if not param_changes:
            return False
        
        avg_param_change = sum(param_changes) / len(param_changes)
        
        # 梯度 = 奖励变化 / 参数变化
        if avg_param_change > 1e-6:
            gradient = avg_reward_change / avg_param_change
            return gradient > self.threshold
        
        return False
    
    def update(self, reward, params):
        """更新历史记录"""
        self.reward_history.append(reward)
        self.param_history.append(dict(params))
        
        # 保持窗口大小
        if len(self.reward_history) > 2 * self.window_size:
            self.reward_history.pop(0)
            self.param_history.pop(0)


# ─────────────────────────────────────────────────────────────────────────────
# Parameter specifications (derived from monte_carlo_single.m template)
# ─────────────────────────────────────────────────────────────────────────────

# Autopilot params: match base=struct(...,'w1',...) in monte_carlo_single.m
# RL tunes all 8 autopilot params across pitch/yaw/roll channels.
AUTOPILOT_PARAM_SPECS: Dict[str, Dict[str, float]] = {
    # Pitch channel — w* bandwidth capped at 60 to avoid matrix singularity / divergence
    "w1":    {"nominal": 40.0,  "min": 20.0,  "max": 60.0,  "scale": 40.0},
    "zeta1": {"nominal": 0.75,  "min": 0.35,  "max": 1.20,  "scale": 0.75},
    "tao1":  {"nominal": 0.15,  "min": 0.05,  "max": 0.35,  "scale": 0.15},
    # Yaw channel
    "w2":    {"nominal": 35.0,  "min": 20.0,  "max": 60.0,  "scale": 35.0},
    "zeta2": {"nominal": 0.75,  "min": 0.35,  "max": 1.20,  "scale": 0.75},
    "tao2":  {"nominal": 0.15,  "min": 0.05,  "max": 0.35,  "scale": 0.15},
    # Roll channel
    "w3":    {"nominal": 40.0,  "min": 20.0,  "max": 60.0,  "scale": 40.0},
    "zeta3": {"nominal": 0.70,  "min": 0.35,  "max": 1.20,  "scale": 0.70},
}

# Guidance law params: rl_N_pn (navigation ratio)  [sw_dist removed — unused in gf()]
# ny_lim: tanh saturation limit in gf() (g); autopilot overshoot needs headroom below 20g task limit
GUIDANCE_PARAM_SPECS: Dict[str, Dict[str, float]] = {
    "N_pn":   {"nominal": 4.0,  "min": 3.0,  "max": 6.0,  "scale": 4.0},
    "ny_lim": {"nominal": 20.0, "min": 16.0, "max": 20.0, "scale": 20.0},
}

# Merged: RL tunes 10 params (8 autopilot + N_pn + ny_lim)
ALL_TUNABLE_PARAM_SPECS: Dict[str, Dict[str, float]] = {
    **AUTOPILOT_PARAM_SPECS,
    **GUIDANCE_PARAM_SPECS,
}

# RL_PARAMS block variable name mapping (Python key → MATLAB variable name)
RL_PARAM_TO_MATLAB: Dict[str, str] = {
    "w1":    "rl_w1",
    "zeta1": "rl_zeta1",
    "tao1":  "rl_tao1",
    "w2":    "rl_w2",
    "zeta2": "rl_zeta2",
    "tao2":  "rl_tao2",
    "w3":    "rl_w3",
    "zeta3": "rl_zeta3",
    "N_pn":   "rl_N_pn",
    "ny_lim": "rl_ny_lim",
}


def patch_ny_lim_in_script(content: str, ny_lim: float) -> str:
    """Patch gf() ny_lim assignment and hard-coded ±20g clips to match RL value."""
    val = float(ny_lim)
    val_str = f"{val:.4f}".rstrip("0").rstrip(".")
    out = re.sub(
        r"(\bny_lim\s*=\s*)[-+]?\d+(?:\.\d*)?(?:[eE][-+]?\d+)?",
        rf"\g<1>{val_str}",
        content,
    )
    out = re.sub(
        r"max\s*\(\s*-20\s*,\s*min\s*\(\s*20\s*,\s*(N[12])\s*\)",
        r"max(-ny_lim,min(ny_lim,\1)",
        out,
    )
    return out

PHYSICAL_PARAM_SPECS: Dict[str, Dict[str, float]] = {
    "m":  {"nominal": 231.0,  "scale": 231.0},
    "P":  {"nominal": 5000.0, "scale": 5000.0},
}

# Metrics output by monte_carlo_single: hit_rate, SEP, peak_ny, pitch_PM, pitch_BW
METRIC_SPECS: Dict[str, Dict[str, float]] = {
    "hit_rate":  {"scale": 100.0, "target": 100.0},
    "SEP":       {"scale":  20.0, "target":   0.0},
    "peak_ny":   {"scale":  20.0, "target":   8.0},
    "pitch_PM":  {"scale":  90.0, "target":  45.0},
    "pitch_BW":  {"scale":  50.0, "target":  20.0},
}

# Backward-compat aliases used by external code (e.g. reflection_agent, cli_agent)
BCKCOMPAT_METRIC_ALIASES: Dict[str, str] = {
    "miss_distance": "SEP",
    "pitch_GM":      "pitch_GM",   # GM still parsed but not in reward
    "peak_n":        "peak_ny",
}

# Reward returned on hard truncation (dynamics divergence / invalid SEP).
HARD_REWARD_PENALTY: float = -10.0


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
    """Safety PeakNy — prefers MC max=XXg over per-run mean."""
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


def _tag_sim_metrics(metrics: Dict[str, float], source: str) -> Dict[str, float]:
    """Attach simulation provenance and keep peak_n / peak_ny aliases in sync."""
    out = dict(metrics)
    out["_sim_source"] = source
    pny = get_peak_ny(out)
    pny_max = get_peak_ny_max(out)
    out["peak_ny"] = pny
    out["peak_n"] = pny
    if pny_max > 0:
        out["peak_ny_max"] = pny_max
        out["peak_n_max"] = pny_max
    return out

# ── Patterns for monte_carlo_single.m print_summary / print_table output ────
# print_summary format:
#   命中率(miss<10m): 85.0%  SEP: 7.23m
#   峰值法向过载: 12.45±3.21g  max=18.67g
#   俯仰PM: 42.3±5.1°  BW: 18.5±2.3 rad/s  GM: 12.1±1.8 dB
# print_table format:
#   子工况名称   95.0%  2.34   8.12   45.2   22.5   18.3
_SIM_PARSE_PATTERNS_MC = [
    # 命中率行
    (r"命中率\(miss<\d+(?:\.\d+)?m\):\s*(\d+(?:\.\d+)?)%",      "hit_rate"),
    (r"SEP:\s*(\d+(?:\.\d+)?)m",                                  "SEP"),
    # 峰值法向过载行 — 取均值（第一个数字）; ± NOT used (GBK encoding issue on Windows)
    (r"峰值法向过载:\s*(\d+(?:\.\d+)?)",                           "peak_ny"),
    # 俯仰PM行
    (r"俯仰PM:\s*(\d+(?:\.\d+)?)",                                "pitch_PM"),
    (r"BW:\s*(\d+(?:\.\d+)?)",                                    "pitch_BW"),
    (r"GM:\s*(\d+(?:\.\d+)?)",                                    "pitch_GM"),
    # 汇总表格行：名称  命中率%  SEP  PeakNy  PM  BW  GM
    # 匹配形如 "某工况名  88.0%  3.45  10.2  42.1  18.5  11.3"
    (r"(?:T|G|AP|R)\d[\u4e00-\u9fffA-Za-z\u00b7:：×\d]*\s+([\d.]+)%\s+([\d.]+)",
     "_table_row"),
]

# Legacy patterns for backward compatibility with old template output
_SIM_PARSE_PATTERNS_LEGACY = [
    (r"Hit\s*=\s*([-+]?\d+\.?\d*)\s*%",                     "hit_rate"),
    (r"HitRate\s*[:=]?\s*([-+]?\d+\.?\d*)\s*%",             "hit_rate"),
    (r"\bMiss\s*=\s*([-+]?\d+\.?\d*)",                       "SEP"),
    (r"miss[_\s]*dist[ance]*\s*[:=]\s*([-+]?\d+\.?\d*)",     "SEP"),
    (r"\bPM\s*=\s*([-+]?\d+\.?\d*)",                         "pitch_PM"),
    (r"\bGM\s*=\s*([-+]?\d+\.?\d*)",                         "pitch_GM"),
    (r"\bBW\s*=\s*([-+]?\d+\.?\d*)",                         "pitch_BW"),
    (r"\bPeakN\s*=\s*([-+]?\d+\.?\d*)",                      "peak_ny"),
    (r"peak[_\s]*n\s*[:=]\s*([-+]?\d+\.?\d*)",               "peak_ny"),
]

# Sentinel defaults used when regex finds no MC evidence (see log06050926).
_PARSE_SENTINEL_HIT = 0.0
_PARSE_SENTINEL_SEP = 50.0
_PARSE_SENTINEL_PM = 30.0
_PARSE_SENTINEL_PEAK_NY = 0.0

_CASE_TABLE_SKIP_PREFIXES = (
    "Case ", "Hit %", "MeanMiss", "[Run]", "════", "──", "╔", "╠", "╚", "【",
    "子工况", "Mean pitch", "命中率", "峰值法向", "俯仰PM",
)


def stdout_has_mc_evidence(raw: str) -> bool:
    """True when stdout contains parseable Monte-Carlo result lines."""
    if not raw or not str(raw).strip():
        return False
    text = str(raw)
    if re.search(r"\[\s*\d+\]\s*(?:HIT|MISS)\b", text, re.I):
        return True
    if re.search(r"命中率\s*\(\s*miss\s*<", text, re.I):
        return True
    if re.search(r"SEP:\s*\d+(?:\.\d+)?m", text, re.I):
        return True
    if re.search(r"Hit\s+rate\s*\(\s*miss\s*<", text, re.I):
        return True
    if re.search(r"Mean\s+miss(?:\s+distance|\s*\(SEP\))?\s*:\s*\d", text, re.I):
        return True
    for line in text.splitlines():
        s = line.strip()
        if not s or any(s.startswith(p) for p in _CASE_TABLE_SKIP_PREFIXES):
            continue
        if re.match(
            r"^.+\s+\d+(?:\.\d+)?\s*%\s+\d+(?:\.\d+)?\s+-?\d",
            s,
        ):
            return True
    return False


def metrics_look_like_parse_defaults(
    metrics: Dict[str, Any],
    raw: str = "",
) -> bool:
    """True when hit/SEP are still parser sentinels and stdout lacks MC evidence."""
    if not metrics:
        return not stdout_has_mc_evidence(raw) if raw else True
    try:
        hr = float(metrics.get("hit_rate", _PARSE_SENTINEL_HIT))
        sep = float(metrics.get("SEP", metrics.get("miss_distance", _PARSE_SENTINEL_SEP)))
    except (TypeError, ValueError):
        return True
    at_sentinel = (
        abs(hr - _PARSE_SENTINEL_HIT) < 0.01
        and abs(sep - _PARSE_SENTINEL_SEP) < 0.01
    )
    if not at_sentinel:
        return False
    return not stdout_has_mc_evidence(raw)


_UNRELIABLE_SIM_SOURCES = frozenset({"fallback", "unavailable", "timeout"})


def is_reliable_sim_metrics(
    metrics: Optional[Dict[str, Any]],
    raw: str = "",
) -> bool:
    """True when metrics come from a real MATLAB/MC run (not fallback/timeout stubs)."""
    if not metrics:
        return False
    src = str(metrics.get("_sim_source", "")).lower()
    if src in _UNRELIABLE_SIM_SOURCES:
        return False
    if metrics.get("_parse_incomplete"):
        return False
    if metrics_look_like_parse_defaults(metrics, raw):
        return False
    return True


def _unreliable_sim_stub(source: str = "unavailable") -> Dict[str, float]:
    """Tagged empty metrics that must not enter Expert/CLS/PPO best selection."""
    stub = _tag_sim_metrics({}, source)
    stub["_parse_incomplete"] = True
    return stub


def _parse_generic_case_table_lines(
    raw: str,
    hr_vals: List[float],
    sep_vals: List[float],
    ny_vals: List[float],
    pm_vals: List[float],
    bw_vals: List[float],
    gm_vals: List[float],
) -> None:
    """Parse English/Hermes case summary rows (Hit % / MeanMiss columns)."""
    for line in raw.splitlines():
        s = line.strip()
        if not s or any(s.startswith(p) for p in _CASE_TABLE_SKIP_PREFIXES):
            continue
        m = re.match(
            r"^(?P<name>.+?)\s+"
            r"(?P<hr>\d+(?:\.\d+)?)\s*%\s+"
            r"(?P<sep>\d+(?:\.\d+)?)\s+"
            r"(?P<ny>-?\d+(?:\.\d+)?)\s+"
            r"(?P<pm>-?\d+(?:\.\d+)?)\s+"
            r"(?P<bw>-?\d+(?:\.\d+)?)\s+"
            r"(?P<gm>-?\d+(?:\.\d+)?|Inf)\s*$",
            s,
        )
        if not m:
            continue
        try:
            hr_vals.append(float(m.group("hr")))
            sep_vals.append(float(m.group("sep")))
            ny_vals.append(float(m.group("ny")))
            pm_vals.append(float(m.group("pm")))
            bw_vals.append(float(m.group("bw")))
            gm_s = m.group("gm")
            if gm_s != "Inf":
                gm_vals.append(float(gm_s))
        except ValueError:
            pass


def _aggregate_parsed_metrics(raw: Any) -> Dict[str, float]:
    """Parse monte_carlo_single.m stdout into a metrics dict (regex only).

    Aggregation across multiple subcases/categories:
      - hit_rate : min  (worst-case hit rate)
      - SEP      : max  (worst-case miss distance)
      - peak_ny  : max  (worst-case peak normal load)
      - pitch_PM : min  (worst-case phase margin)
      - pitch_BW : mean (average bandwidth)
      - pitch_GM : min  (worst-case gain margin)

    Also populates legacy alias keys for backward compatibility:
      miss_distance = SEP,  peak_n = peak_ny
    """
    defaults: Dict[str, float] = {
        "hit_rate": 0.0, "SEP": 50.0, "peak_ny": 0.0,
        "pitch_PM": 30.0, "pitch_BW": 15.0, "pitch_GM": 6.0,
        # legacy aliases
        "miss_distance": 50.0, "peak_n": 0.0,
    }
    if isinstance(raw, dict):
        inner = raw.get("metrics", raw)
        for k in ("hit_rate", "SEP", "peak_ny", "pitch_PM", "pitch_BW", "pitch_GM"):
            v = inner.get(k)
            if v is not None:
                defaults[k] = float(v)
        # Legacy aliases in dict input
        for alias, target in (("miss_distance", "SEP"), ("peak_n", "peak_ny"),
                              ("pitch_BW", "pitch_BW")):
            v = inner.get(alias)
            if v is not None:
                defaults[target] = float(v)
        if "success" in inner:
            defaults["hit_rate"] = 100.0 if inner["success"] else 0.0
        defaults["miss_distance"] = defaults["SEP"]
        defaults["peak_n"] = defaults["peak_ny"]
        return defaults
    if not isinstance(raw, str) or not raw.strip():
        defaults["miss_distance"] = defaults["SEP"]
        defaults["peak_n"] = defaults["peak_ny"]
        return defaults

    # Collect all per-subcase values for aggregation
    hr_vals: List[float] = []
    sep_vals: List[float] = []
    ny_vals: List[float] = []
    pm_vals: List[float] = []
    bw_vals: List[float] = []
    gm_vals: List[float] = []

    # ── Parse per-run lines (encoding-immune, pure ASCII) ────────────
    # Format: [  N] HIT/MISS  miss_val  peakny_val  pm_val  bw_val  gm_val
    # These lines are always present even if the simulation crashes before
    # producing the summary section, making this the most robust parser.
    _hit_count = 0
    _run_count = 0
    _run_miss_vals: List[float] = []
    _run_ny_vals: List[float] = []
    _run_pm_vals: List[float] = []
    _run_bw_vals: List[float] = []
    _run_gm_vals: List[float] = []
    for m in re.finditer(
        r"\[\s*\d+\]\s*(HIT|MISS)\s+"
        r"([\d.]+)\s+"          # miss distance (always >= 0)
        r"(-?[\d.]+)\s+"        # peak Ny (can be negative in edge cases)
        r"(-?[\d.]+)\s+"        # PM (can be negative for unstable systems)
        r"(-?[\d.]+)\s+"        # BW (can be negative in edge cases)
        r"(-?[\d.]+|Inf|NaN)",   # GM
        raw,
    ):
        _run_count += 1
        if m.group(1) == "HIT":
            _hit_count += 1
        try:
            _run_miss_vals.append(float(m.group(2)))
            _run_ny_vals.append(float(m.group(3)))
            _v_pm = float(m.group(4))
            if not (math.isnan(_v_pm) or math.isinf(_v_pm)):
                _run_pm_vals.append(_v_pm)
            _v_bw = float(m.group(5))
            if not (math.isnan(_v_bw) or math.isinf(_v_bw)):
                _run_bw_vals.append(_v_bw)
            _v_gm = m.group(6)
            if _v_gm not in ("Inf", "NaN"):
                _run_gm_vals.append(float(_v_gm))
        except ValueError:
            pass
    # Also count ERR lines — if all runs produce ERR, hit_rate = 0%
    _err_count = len(re.findall(r"\[\s*\d+\]\s*ERR:", raw))
    if _run_count > 0:
        # Compute aggregated metrics from per-run data
        _run_hr = _hit_count / _run_count * 100.0
        hr_vals.append(_run_hr)
        if _run_miss_vals:
            sep_vals.append(max(_run_miss_vals))
        if _run_ny_vals:
            ny_vals.append(max(_run_ny_vals))
        if _run_pm_vals:
            pm_vals.append(min(_run_pm_vals))
        if _run_bw_vals:
            bw_vals.append(sum(_run_bw_vals) / len(_run_bw_vals))
        if _run_gm_vals:
            gm_vals.append(min(_run_gm_vals))
        logger.debug(
            f"[parse] per-run lines: {_run_count} runs (+{_err_count} ERR), "
            f"hit={_run_hr:.0f}%, SEP={max(_run_miss_vals) if _run_miss_vals else '?'}, "
            f"PM={min(_run_pm_vals) if _run_pm_vals else '?'}, "
            f"BW={sum(_run_bw_vals)/len(_run_bw_vals) if _run_bw_vals else '?'}"
        )
    elif _err_count > 0:
        # All runs crashed — signal 0% hit rate so RL knows params are bad
        hr_vals.append(0.0)
        logger.info(
            f"[parse] ALL {_err_count} runs produced ERR (no HIT/MISS). "
            f"Parameters likely caused sim crash."
        )

    # Parse 【汇总】block lines
    for m in re.finditer(r"命中率\(miss<\d+(?:\.\d+)?m\):\s*(\d+(?:\.\d+)?)%", raw):
        hr_vals.append(float(m.group(1)))
    for m in re.finditer(r"SEP:\s*(\d+(?:\.\d+)?)m", raw):
        sep_vals.append(float(m.group(1)))
    # NOTE: \u00b1 (±) is NOT used below.
    # MATLAB on Windows outputs ± as GBK single-byte \xb1; when the subprocess
    # stdout is decoded as UTF-8 (or with errors='replace') the character becomes
    # a replacement/garbage byte, so patterns anchored on \u00b1 never match.
    # Instead we match the number immediately after the label and stop greedily.
    for m in re.finditer(r"峰值法向过载:\s*(\d+(?:\.\d+)?)", raw):
        ny_vals.append(float(m.group(1)))
    ny_max_vals: List[float] = []
    for m in re.finditer(r"max=(\d+(?:\.\d+)?)\s*g", raw, re.IGNORECASE):
        ny_max_vals.append(float(m.group(1)))
    for m in re.finditer(r"俯仰PM:\s*(\d+(?:\.\d+)?)", raw):
        pm_vals.append(float(m.group(1)))
    for m in re.finditer(r"BW:\s*(\d+(?:\.\d+)?)", raw):
        bw_vals.append(float(m.group(1)))
    for m in re.finditer(r"GM:\s*(\d+(?:\.\d+)?)", raw):
        gm_vals.append(float(m.group(1)))

    # Parse summary table rows — always run (not just as fallback).
    # Row format: <名称>  <命中率>%  <SEP>  <PeakNy>  <PM>  <BW>  <GM>
    # Table is the canonical source for PeakNy/PM/BW/GM because it has no ±
    # character, so it is immune to the GBK/UTF-8 encoding issue that causes
    # those metrics to be missing from print_summary parsing.
    # Guards: only append to a list if it is still empty, to avoid double-counting
    # when print_summary already provided the value.
    #
    # NOTE: The name portion may contain \ufffd replacement characters when
    # GBK bytes are partially invalid, so we use \S+ (any non-whitespace)
    # instead of the restrictive [\u4e00-\u9fff] character class.  The %
    # after hit_rate is optional because some print_table variants omit it.
    for m in re.finditer(
        r"(?:T|G|AP|R)\d\S*"
        r"\s+(\d+(?:\.\d+)?)%?\s+"   # group 1: hit_rate (% optional)
        r"(\d+(?:\.\d+)?)\s+"        # group 2: SEP (always >= 0)
        r"(-?\d+(?:\.\d+)?)\s+"      # group 3: PeakNy
        r"(-?\d+(?:\.\d+)?)\s+"      # group 4: PM (can be negative)
        r"(-?\d+(?:\.\d+)?)\s+"      # group 5: BW
        r"(-?\d+(?:\.\d+)?|Inf)",    # group 6: GM
        raw,
    ):
        try:
            _hr = float(m.group(1))
            _sep = float(m.group(2))
            _ny = float(m.group(3))
            _pm = float(m.group(4))
            _bw = float(m.group(5))
            _gm_s = m.group(6)
            # Append to lists (aggregation handles multiple subcases)
            hr_vals.append(_hr)
            sep_vals.append(_sep)
            ny_vals.append(_ny)
            pm_vals.append(_pm)
            bw_vals.append(_bw)
            if _gm_s != "Inf":
                gm_vals.append(float(_gm_s))
        except ValueError:
            pass

    # English Hermes / legacy stdout: "Mean pitch PM: 30.5 deg, BW: 21.12 rad/s, GM: 6.09 dB"
    for m in re.finditer(
        r"Mean\s+pitch\s+PM:\s*(-?[\d.]+)\s*deg,\s*BW:\s*([\d.]+)\s*rad/s,\s*GM:\s*([\d.]+)\s*dB",
        raw,
        re.I,
    ):
        try:
            pm_vals.append(float(m.group(1)))
            bw_vals.append(float(m.group(2)))
            gm_vals.append(float(m.group(3)))
        except ValueError:
            pass

    # English summary lines
    for m in re.finditer(
        r"Hit\s+rate\s*\(\s*miss\s*<\s*\d+(?:\.\d+)?m\s*\):\s*(\d+(?:\.\d+)?)\s*%",
        raw,
        re.I,
    ):
        hr_vals.append(float(m.group(1)))
    for m in re.finditer(
        r"Mean\s+miss(?:\s+distance|\s*\(SEP\))?\s*:\s*(\d+(?:\.\d+)?)\s*m",
        raw,
        re.I,
    ):
        sep_vals.append(float(m.group(1)))

    # Generic case table (English "Case Hit % MeanMiss ..." data rows)
    if not hr_vals or not sep_vals:
        _parse_generic_case_table_lines(
            raw, hr_vals, sep_vals, ny_vals, pm_vals, bw_vals, gm_vals,
        )

    # Legacy patterns: always run for metrics not yet found.
    # hit_rate/SEP use the primary guard; PM/BW/PeakNy/GM run unconditionally
    # so that mixed-format output (standard hit_rate + legacy PM=75.9) is handled.
    _legacy_need_hr  = not hr_vals
    _legacy_need_sep = not sep_vals
    for pat, key in _SIM_PARSE_PATTERNS_LEGACY:
        # Skip hit_rate / SEP if already found to avoid double-counting
        if key == "hit_rate"  and not _legacy_need_hr:  continue
        if key == "SEP"       and not _legacy_need_sep: continue
        # For PM/BW/PeakNy always try (override only when list is still empty)
        if key == "pitch_PM"  and pm_vals: continue
        if key == "pitch_BW"  and bw_vals: continue
        if key == "pitch_GM"  and gm_vals: continue
        if key == "peak_ny"   and ny_vals: continue
        if key == "peak_n"    and ny_vals: continue
        matches = re.findall(pat, raw, re.IGNORECASE)
        if matches:
            try:
                val = float(matches[-1])
                if key == "hit_rate":   hr_vals.append(val)
                elif key == "SEP":      sep_vals.append(val)
                elif key in ("peak_ny", "peak_n"): ny_vals.append(val)
                elif key == "pitch_PM": pm_vals.append(val)
                elif key == "pitch_BW": bw_vals.append(val)
                elif key == "pitch_GM": gm_vals.append(val)
            except ValueError:
                pass

    # Aggregate: worst-case where applicable
    if hr_vals:  defaults["hit_rate"] = min(hr_vals)
    if sep_vals: defaults["SEP"]      = max(sep_vals)
    if ny_vals:
        defaults["peak_ny"] = float(sum(ny_vals) / len(ny_vals))
        defaults["peak_ny_mean"] = defaults["peak_ny"]
    if ny_max_vals:
        defaults["peak_ny_max"] = max(ny_max_vals)
        defaults["peak_n_max"] = defaults["peak_ny_max"]
    elif ny_vals:
        defaults["peak_ny_max"] = max(ny_vals)
        defaults["peak_n_max"] = defaults["peak_ny_max"]
    if pm_vals:  defaults["pitch_PM"] = min(pm_vals)
    if bw_vals:  defaults["pitch_BW"] = float(sum(bw_vals) / len(bw_vals))
    if gm_vals:  defaults["pitch_GM"] = min(gm_vals)

    defaults["miss_distance"] = defaults["SEP"]
    defaults["peak_n"]        = defaults["peak_ny"]
    return defaults


def _parse_metrics_with_llm(raw: str) -> Dict[str, float]:
    """Use LLM to extract simulation metrics from raw MATLAB stdout.

    Called when regex parsing leaves key metrics at their default values.
    Uses the same LLM config as the rest of the system (sync openai client).
    Returns a partial dict — only keys with successfully parsed values.
    """
    try:
        from multi_agent.config_loader import get_config
        import openai as _openai

        cfg = get_config().llm
        client = _openai.OpenAI(
            api_key=cfg.api_key or "sk-dummy",
            base_url=cfg.base_url or "https://api.openai.com/v1",
            timeout=30,
            max_retries=3,
        )

        # Trim stdout to last 3000 chars (summary section is always near the end)
        snippet = raw[-3000:] if len(raw) > 3000 else raw

        prompt = (
            "Below is the stdout of a MATLAB Monte-Carlo simulation for a missile "
            "guidance system. Extract ONLY these numeric metrics and return them as "
            "a single-line JSON object with these exact keys: "
            "hit_rate (percentage 0-100), SEP (metres), peak_ny (mean g), "
            "peak_ny_max (worst-case max g), "
            "pitch_PM (degrees), pitch_BW (rad/s), pitch_GM (dB). "
            "If a value is Inf or cannot be determined, omit that key. "
            "Return ONLY the JSON object, no other text.\n\n"
            f"STDOUT:\n{snippet}"
        )

        from multi_agent.llm_utils import get_safe_temperature

        resp = client.chat.completions.create(
            model=cfg.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=get_safe_temperature(cfg.model, 0.0),
            max_tokens=128,
        )
        content = resp.choices[0].message.content
        if not content:
            logger.warning("[LLM metric parse] LLM returned None content")
            return {}
        
        content = content.strip()
        logger.debug(f"[LLM metric parse] Raw response (first 200 chars): {content[:200]}")
        
        # Strip markdown fences if present
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        content = content.strip()
        if not content:
            logger.warning("[LLM metric parse] LLM returned empty content after stripping")
            return {}
        
        parsed = json.loads(content)
        result: Dict[str, float] = {}
        for k, v in parsed.items():
            try:
                result[k] = float(v)
            except (TypeError, ValueError):
                pass
        logger.info(f"[LLM metric parse] extracted: {result}")
        return result
    except json.JSONDecodeError as exc:
        logger.warning(f"[LLM metric parse] JSON decode failed: {exc}")
        return {}
    except Exception as exc:
        logger.warning(f"[LLM metric parse] failed: {exc}")
        return {}


def parse_sim_stdout(raw: Any) -> Dict[str, float]:
    """Parse MATLAB stdout with regex; fall back to LLM for missing metrics."""
    result = _aggregate_parsed_metrics(raw if isinstance(raw, str) else str(raw))

    # Identify which key metrics are still at their hard-coded defaults
    _pm_default = 30.0
    _bw_default = 15.0
    _ny_default = 0.0
    _raw_str = str(raw)
    # If MATLAB explicitly printed NaN for PM/BW (cg() failed), record as nan
    # and do NOT trigger the LLM fallback for those metrics.
    _pm_nan = bool(re.search(r'俯仰PM[^\n]{0,40}NaN|NaN[^\n]{0,40}俯仰PM', _raw_str))
    _bw_nan = bool(re.search(r'\bBW[^\n]{0,30}NaN|NaN[^\n]{0,30}BW\b', _raw_str))
    if _pm_nan and abs(result.get("pitch_PM", _pm_default) - _pm_default) < 0.01:
        result["pitch_PM"] = float("nan")
    if _bw_nan and abs(result.get("pitch_BW", _bw_default) - _bw_default) < 0.01:
        result["pitch_BW"] = float("nan")
    _needs_llm = (
        (abs(result.get("pitch_PM", _pm_default) - _pm_default) < 0.01 and not _pm_nan) or
        (abs(result.get("pitch_BW", _bw_default) - _bw_default) < 0.01 and not _bw_nan) or
        result.get("peak_ny", _ny_default) == _ny_default
    )
    if _needs_llm and raw and len(_raw_str) > 50:
        logger.debug("[parse_sim_stdout] Regex incomplete — calling LLM fallback")
        llm_vals = _parse_metrics_with_llm(str(raw))
        for k, v in llm_vals.items():
            _key_map = {
                "hit_rate": "hit_rate", "SEP": "SEP", "peak_ny": "peak_ny",
                "peak_ny_max": "peak_ny_max",
                "pitch_PM": "pitch_PM", "pitch_BW": "pitch_BW", "pitch_GM": "pitch_GM",
            }
            if k in _key_map:
                result[_key_map[k]] = v
        # Sync aliases
        result["miss_distance"] = result.get("SEP", result.get("miss_distance", 0.0))
        result["peak_n"] = result.get("peak_ny", result.get("peak_n", 0.0))
        if result.get("peak_ny_max", 0.0) <= 0 and result.get("peak_ny", 0.0) > 0:
            result["peak_ny_max"] = result["peak_ny"]
        if result.get("peak_ny", 0.0) <= 0 and result.get("peak_ny_max", 0.0) > 0:
            result["peak_ny"] = result["peak_ny_max"]
        result["peak_n_max"] = result.get("peak_ny_max", result.get("peak_n_max", 0.0))

    _incomplete = metrics_look_like_parse_defaults(result, _raw_str)
    result["_parse_incomplete"] = _incomplete
    if _incomplete:
        logger.warning(
            "[parse_sim_stdout] hit_rate/SEP at sentinel defaults (0%% / 50m) "
            "with no MC evidence in stdout — metrics unreliable for RL"
        )

    return result

# Mission categories as defined in monte_carlo_single.m
# RUN_CASE: 'T'|'G'|'AP'|'R'|'ALL'; SUB_IDX: 0=全部, 1..N=指定子工况
CATEGORY_DEFS: Dict[str, Dict] = {
    "T":  {"name": "目标机动 (Target Maneuver)",      "max_sub": 4,
           "subcases": {0:"T1:匀速直飞", 1:"T2:低频小幅度机动",
                        2:"T3:中频中等幅度机动", 3:"T4:高频大幅度机动"}},
    "G":  {"name": "交战几何 (Engagement Geometry)",  "max_sub": 5,
           "subcases": {0:"G1:标称几何", 1:"G2:近距小偏置",
                        2:"G3:近距大偏置", 3:"G4:远距小偏置", 4:"G5:远距大偏置"}},
    "AP": {"name": "驾驶仪退化 (Autopilot Degradation)","max_sub": 5,
           "subcases": {0:"AP1:全标称驾驶仪", 1:"AP2:通道时间常数单独退化",
                        2:"AP3:驾驶仪轻度退化+tao轻度退化",
                        3:"AP4:驾驶仪中度退化+tao中度退化",
                        4:"AP5:驾驶仪重度退化+tao重度退化"}},
    "R":  {"name": "综合鲁棒 (Comprehensive Robustness)","max_sub": 3,
           "subcases": {0:"R1:全标称基准", 1:"R2:多参数综合摄动",
                        2:"R3:大初始姿态偏差"}},
}


# ─────────────────────────────────────────────────────────────────────────────
# Dynamic MATLAB Parameter Extractor
# ─────────────────────────────────────────────────────────────────────────────

def extract_matlab_params(script_content: str) -> Dict[str, float]:
    """
    Dynamically extract all numeric parameter assignments from a MATLAB script.
    Handles scalar assignments and struct.field assignments.
    All values returned as float.
    """
    params: Dict[str, float] = {}

    scalar_pat = re.compile(
        r"(?<![.\w])([a-zA-Z_]\w*)\s*=\s*([-+]?\d+(?:\.\d*)?(?:[eE][-+]?\d+)?)\s*;",
        re.MULTILINE,
    )
    struct_pat = re.compile(
        r"([a-zA-Z_]\w*)\.([a-zA-Z_]\w*)\s*=\s*([-+]?\d+(?:\.\d*)?(?:[eE][-+]?\d+)?)\s*;",
        re.MULTILINE,
    )

    for m in scalar_pat.finditer(script_content):
        try:
            params[m.group(1)] = float(m.group(2))
        except ValueError:
            pass

    for m in struct_pat.finditer(script_content):
        key = f"{m.group(1)}_{m.group(2)}"
        try:
            params[key] = float(m.group(3))
        except ValueError:
            pass

    return params


def extract_autopilot_params(script_content: str) -> Dict[str, float]:
    """Extract tunable parameters from RL_PARAMS block (rl_* variables) in
    monte_carlo_single-style scripts.  Falls back to scanning all scalar
    assignments if the block is absent (legacy dp.* templates)."""
    result: Dict[str, float] = {}

    # Primary: parse RL_PARAMS_BEGIN ... RL_PARAMS_END block
    rl_block_match = re.search(
        r"%%.*?RL_PARAMS_BEGIN.*?\n(.*?)%%.*?RL_PARAMS_END",
        script_content, re.DOTALL
    )
    if rl_block_match:
        block = rl_block_match.group(1)
        for py_key, ml_var in RL_PARAM_TO_MATLAB.items():
            m = re.search(
                rf"{re.escape(ml_var)}\s*=\s*([-+]?\d+(?:\.\d*)?(?:[eE][-+]?\d+)?)\s*;",
                block,
            )
            if m:
                try:
                    result[py_key] = float(m.group(1))
                except ValueError:
                    pass

    # Fallback: scan full script for dp.* or bare scalar assignments
    all_params = extract_matlab_params(script_content)
    for key, spec in ALL_TUNABLE_PARAM_SPECS.items():
        if key in result:
            continue
        dp_key = f"dp_{key}"
        if dp_key in all_params:
            result[key] = float(all_params[dp_key])
        elif key in all_params:
            result[key] = float(all_params[key])
        else:
            result[key] = float(spec["nominal"])

    if "ny_lim" not in result:
        m_ny = re.search(
            r"\bny_lim\s*=\s*([-+]?\d+(?:\.\d*)?(?:[eE][-+]?\d+)?)",
            script_content,
        )
        if m_ny:
            try:
                result["ny_lim"] = float(m_ny.group(1))
            except ValueError:
                pass
    return result


def extract_physical_params(script_content: str) -> Dict[str, float]:
    """Extract physical parameters (m, P) from script."""
    all_params = extract_matlab_params(script_content)
    result: Dict[str, float] = {}
    for key, spec in PHYSICAL_PARAM_SPECS.items():
        result[key] = float(all_params.get(key, spec["nominal"]))
    return result


def parse_mission_conditions(prompt: str) -> Dict[str, List[int]]:
    """
    Parse mission conditions from a prompt string for monte_carlo_single.m.

    Supported patterns
    ------------------
    * Chinese keywords: 目标机动/T, 交战几何/G, 驾驶仪/AP, 鲁棒/R、全工况
    * Explicit: ``RUN_CASE='T'; SUB_IDX=0;``
    * Legacy: ``run_T=true; sub_T=[0,1];``

    Returns
    -------
    Dict ``{category: [sub_indices]}``.  Keys are 'T','G','AP','R'.
    Empty dict means ALL categories (maps to RUN_CASE='ALL').
    If nothing is detected, defaults to ALL categories.

    Sub-indices follow CATEGORY_DEFS numbering (0-based in MATLAB
    for T0-T3, G0-G4, AP0-AP3, R0-R2).
    """
    conditions: Dict[str, List[int]] = {}

    # ── Explicit RUN_CASE='X' format ──────────────────────────────────────
    rc_match = re.search(
        r"RUN_CASE\s*=\s*['\"]([A-Za-z]+)['\"]\s*;?", prompt, re.IGNORECASE
    )
    si_match = re.search(r"SUB_IDX\s*=\s*(\d+)", prompt, re.IGNORECASE)
    if rc_match:
        rc_val = rc_match.group(1).upper()
        si_val = int(si_match.group(1)) if si_match else 0
        if rc_val == "ALL":
            return {}
        elif rc_val in CATEGORY_DEFS:
            max_s = CATEGORY_DEFS[rc_val]["max_sub"]
            conditions[rc_val] = [si_val] if si_val > 0 else list(range(0, max_s))
        return conditions

    # ── Compound-token parsing: T1/T2/G1/AP3/R2 style (highest priority) ──
    # \b(\d+)\b misses these because T and 1 share no word-boundary.
    compound_pat = re.compile(
        r"(?<![A-Za-z])(T|G|AP|R)(\d+)(?!\d)", re.IGNORECASE
    )
    for m in compound_pat.finditer(prompt):
        cat = m.group(1).upper()
        idx = int(m.group(2))
        if cat in CATEGORY_DEFS and 1 <= idx <= CATEGORY_DEFS[cat]["max_sub"]:
            conditions.setdefault(cat, [])
            if idx not in conditions[cat]:
                conditions[cat].append(idx)
    for cat in list(conditions):
        conditions[cat] = sorted(conditions[cat])
    if conditions:
        return conditions

    # ── Chinese keyword detection ────────────────────────────────────────
    kw_map = [
        ("T",  ["目标机动", "横向机动", "target maneuver", r"\bT\b"]),
        ("G",  ["交战几何", "几何工况", "engagement geometry", r"\bG\b"]),
        ("AP", ["驾驶仪", "autopilot", r"\bAP\b"]),
        ("R",  ["综合鲁棒", "鲁棒性", "robustness", r"\bR\b"]),
    ]
    all_kws = ["全工况", "所有工况", "all cases", r"\ball\b"]
    if any(re.search(kw if kw.startswith(r"\b") else re.escape(kw), prompt, re.IGNORECASE)
           for kw in all_kws):
        return {}  # ALL
    for cat, kws in kw_map:
        if any(re.search(kw, prompt, re.IGNORECASE) for kw in kws):
            max_s = CATEGORY_DEFS[cat]["max_sub"]
            nums = [int(x) for x in re.findall(r"\b(\d+)\b", prompt)
                    if 1 <= int(x) <= max_s]
            conditions[cat] = sorted(set(nums)) if nums else list(range(0, max_s))

    # ── Legacy run_X = true/false format ──────────────────────────────────
    for cat in CATEGORY_DEFS:
        max_sub = CATEGORY_DEFS[cat]["max_sub"]
        run_pat = re.search(
            rf"run_{cat}\s*=\s*(true|false|\[\s*\]|1|0)\b", prompt, re.IGNORECASE
        )
        if run_pat:
            val = run_pat.group(1).lower().replace(" ", "")
            if val in ("true", "1", "[]"):
                sub_pat = re.search(rf"sub_{cat}\s*=\s*\[(.*?)\]", prompt, re.IGNORECASE)
                if sub_pat:
                    content_norm = sub_pat.group(1).strip().replace("\uff0c", ",")
                    nums = [int(x) for x in re.findall(r"\d+", content_norm)
                            if 0 <= int(x) <= max_sub]
                    conditions[cat] = sorted(set(nums)) if nums else list(range(0, max_sub))
                else:
                    conditions[cat] = list(range(0, max_sub))

    if not conditions:
        conditions["T"] = list(range(0, CATEGORY_DEFS["T"]["max_sub"]))

    return conditions


def build_matlab_conditions_str(conditions: Dict[str, List[int]]) -> str:
    """
    Convert a parsed conditions dict to MATLAB patching string for
    monte_carlo_single.m.

    Format: ``RUN_CASE='T';SUB_IDX=0;N_MC=50;``

    - Empty dict  → ``RUN_CASE='ALL';SUB_IDX=0;``
    - Single cat  → ``RUN_CASE='T';SUB_IDX=0;``  (SUB_IDX=0 = all subcases)
    - Specific sub→ ``RUN_CASE='T';SUB_IDX=1;``  (first listed subcase)
    - Multi cat   → ``RUN_CASE='ALL';SUB_IDX=0;`` (run all, filter by category)
    """
    # Only categories with at least one subcase entry are "active".
    # Entries like 'G': [] mean "not requested" and must be ignored.
    active = {k: v for k, v in (conditions or {}).items() if v}
    if not active or len(active) >= len(CATEGORY_DEFS):
        return "RUN_CASE='ALL';SUB_IDX=0;"
    if len(active) == 1:
        cat = list(active.keys())[0]
        subs = active[cat]
        sub_idx = subs[0] if subs else 0
        return f"RUN_CASE='{cat}';SUB_IDX={sub_idx};"
    # Multiple active categories → 'ALL' (RUN_CASE can't express selective cats natively)
    return "RUN_CASE='ALL';SUB_IDX=0;"


class _NumpyNet:
    """Two-layer tanh network with Adam optimizer."""

    def __init__(self, in_dim: int, hid: int, out_dim: int, lr: float):
        self.lr = lr
        scale1 = math.sqrt(2.0 / in_dim)
        scale2 = math.sqrt(2.0 / hid)
        self.W1 = np.random.randn(in_dim, hid).astype(np.float32) * scale1
        self.b1 = np.zeros(hid, dtype=np.float32)
        self.W2 = np.random.randn(hid, out_dim).astype(np.float32) * scale2
        self.b2 = np.zeros(out_dim, dtype=np.float32)
        params = [self.W1, self.b1, self.W2, self.b2]
        self._m = [np.zeros_like(p) for p in params]
        self._v = [np.zeros_like(p) for p in params]
        self._t = 0

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        h = np.tanh(x @ self.W1 + self.b1)
        out = h @ self.W2 + self.b2
        return out, h

    def adam_step(self, grads: List[np.ndarray],
                  beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8) -> None:
        self._t += 1
        params = [self.W1, self.b1, self.W2, self.b2]
        for i, (p, g) in enumerate(zip(params, grads)):
            g = np.clip(g, -0.5, 0.5)
            self._m[i] = beta1 * self._m[i] + (1 - beta1) * g
            self._v[i] = beta2 * self._v[i] + (1 - beta2) * g ** 2
            mh = self._m[i] / (1 - beta1 ** self._t)
            vh = self._v[i] / (1 - beta2 ** self._t)
            p -= self.lr * mh / (np.sqrt(vh) + eps)


class ActorNet(_NumpyNet):
    """Gaussian policy: outputs action mean; fixed log_std per dimension."""

    def __init__(self, state_dim: int, action_dim: int, hid: int = 64, lr: float = 3e-4):
        super().__init__(state_dim, hid, action_dim, lr)
        self.action_dim = action_dim
        self.log_std = np.full(action_dim, -0.5, dtype=np.float32)
        self._m_ls = np.zeros(action_dim, dtype=np.float32)
        self._v_ls = np.zeros(action_dim, dtype=np.float32)

    def sample(self, state: np.ndarray, explore: bool = True
               ) -> Tuple[np.ndarray, float]:
        mean, _ = self.forward(state)
        if explore:
            std = np.exp(np.clip(self.log_std, -4.0, 2.0))
            noise = np.random.randn(self.action_dim).astype(np.float32)
            action = mean + std * noise
            lp = -0.5 * float(np.sum(noise ** 2 + 2 * self.log_std + math.log(2 * math.pi)))
        else:
            action = mean
            lp = 0.0
        return action, lp


class CriticNet(_NumpyNet):
    """State-value network V(s)."""

    def __init__(self, state_dim: int, hid: int = 64, lr: float = 1e-3):
        super().__init__(state_dim, hid, 1, lr)

    def value(self, state: np.ndarray) -> float:
        v, _ = self.forward(state)
        return float(v[0])


# ─────────────────────────────────────────────────────────────────────────────
# MATLAB script helpers
# ─────────────────────────────────────────────────────────────────────────────

def _extract_matlab_function_name(content: str) -> Optional[str]:
    """
    If *content* is a MATLAB function file (first non-blank, non-comment line
    starts with 'function'), return the function name; otherwise return None.
    """
    for line in content.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("%"):
            continue
        m = re.match(r"^function\s+(?:[^=\s]+\s*=\s*)?([a-zA-Z_]\w*)", stripped)
        if m:
            return m.group(1)
        break  # First real line is not a function declaration → script
    return None


def _looks_like_matlab_script(content: str) -> Tuple[bool, str]:
    """
    Heuristically decide whether *content* is a real MATLAB/Octave script
    rather than natural-language prose, markdown documentation, or LLM
    chatter that accidentally landed in a ``.m`` file.

    Returns ``(is_valid, reason)``.  ``reason`` is a short diagnostic
    string suitable for logging.

    A script is considered valid if it satisfies BOTH:
      • Contains at least one structural MATLAB token (``function``,
        ``clc;``, ``clear``, ``end``, ``for``/``if``/``while``).
      • Contains at least one MATLAB-style assignment with a trailing
        semicolon (``... = ...;``).

    And does NOT exhibit ANY of these red flags:
      • A heavy fraction of lines look like natural-language sentences
        (start with capital letter, end with period, no semicolons,
        no ``%`` comment marker, no MATLAB punctuation).
      • Markdown headings (``# `` at line start) or fenced code markers
        (triple-backtick blocks).
    """
    if not content or not content.strip():
        return False, "empty content"

    lines = content.splitlines()

    # Markdown / LLM-output red flags
    if any(ln.lstrip().startswith("```") for ln in lines):
        return False, "contains fenced code markers (```)"
    md_headings = sum(
        1 for ln in lines if re.match(r"^\s{0,3}#{1,6}\s+\S", ln)
    )
    if md_headings >= 2:
        return False, f"contains {md_headings} markdown headings"

    # Structural tokens — at least one must be present
    structural_re = re.compile(
        r"(^\s*function\s+\w|"
        r"^\s*clc\s*;|^\s*clear\s+|^\s*close\s+all\s*;|"
        r"^\s*end\s*$|"
        r"^\s*(for|if|while|switch)\s+\w)",
        re.MULTILINE,
    )
    has_structure = bool(structural_re.search(content))

    # MATLAB-style assignment with trailing semicolon
    assignment_re = re.compile(
        r"^\s*[a-zA-Z_]\w*(?:\.[a-zA-Z_]\w*)?\s*=\s*[^=;]+;",
        re.MULTILINE,
    )
    n_assignments = len(assignment_re.findall(content))

    # Natural-language line heuristic: starts with capital letter, ends with
    # a period or Chinese full stop, has no MATLAB punctuation (=, ;, %, end,
    # etc.) on the same line.
    nat_lang_re = re.compile(
        r"^\s*[A-Z\u4e00-\u9fff][^=;%]{20,}[\.\u3002]\s*$"
    )
    n_nat_lang = sum(1 for ln in lines if nat_lang_re.match(ln))
    nat_ratio = n_nat_lang / max(1, len(lines))

    if not has_structure:
        return False, "no MATLAB structural tokens (function/clc/clear/end/…)"
    if n_assignments < 3:
        return False, (
            f"only {n_assignments} MATLAB-style assignments (… = …;) found "
            f"— expected at least 3 for a real simulation script"
        )
    if nat_ratio > 0.30:
        return False, (
            f"{nat_ratio:.0%} of lines look like natural-language prose "
            f"({n_nat_lang}/{len(lines)})"
        )

    # Minimum line count: a real monte_carlo_single-style script is hundreds of
    # lines.  Fewer than 80 almost certainly means LLM output was truncated or
    # only the header / RL_PARAMS block was emitted.
    if len(lines) < 80:
        return False, (
            f"script too short ({len(lines)} lines < 80 minimum) "
            f"— likely LLM-truncated or incomplete template"
        )

    # Key body functions: at least one of these must be present for the script
    # to be capable of running a meaningful MC simulation.
    _body_markers = [
        "run_category_mc", "run_case_mc", "for.*N_MC", "parfor.*N_MC",
        r"function\s+gf\b", r"function\s+guidance_local\b",
        r"function\s+cg\b", "mont.*carlo",
    ]
    _body_re = re.compile("|".join(_body_markers), re.IGNORECASE | re.MULTILINE)
    if not _body_re.search(content):
        return False, (
            "missing MC-body markers (run_category_mc / gf / guidance_local / cg) "
            "— script appears to be header-only"
        )

    return True, (
        f"valid (structural tokens present, {n_assignments} assignments, "
        f"natural-language ratio {nat_ratio:.0%}, {len(lines)} lines)"
    )


def validate_rl_script_content(content: str) -> Tuple[bool, str]:
    """Return (ok, reason).  When ok is False, RL must not use MATLAB on this script."""
    if re.search(r"\bendfunction\b", content, re.IGNORECASE):
        return False, "含 Octave 语法 endfunction（MATLAB 应使用 end）"
    if (
        "Placeholder simulation function" in content
        or "sim_s(p) %#ok<INUSD>" in content
    ):
        return False, (
            "sim_s 为 PLACEHOLDER 占位实现，RL 参数无法改变仿真结果"
        )
    if not re.search(r"RL_PARAMS_BEGIN", content):
        if len(re.findall(r"\bdp\.\w+\s*=", content)) < 3:
            return False, "缺少 RL_PARAMS_BEGIN 且 dp.* 参数不足，RL 无法 patch"
    _has_parseable_output = (
        "print_summary" in content
        or re.search(r"命中率\s*\(\s*miss\s*<", content, re.I)
        or re.search(r"fprintf\s*\(\s*['\"]\s*\[\s*%3d\]", content)
        or re.search(r"\[\s*%3d\].*(?:HIT|MISS)", content, re.I)
        or re.search(r"Mean\s+pitch\s+PM:", content, re.I)
    )
    if not _has_parseable_output:
        return False, (
            "脚本缺少 RL 解析器可识别的仿真输出"
            "（print_summary / [N] HIT-MISS / Mean pitch PM）"
        )
    return True, "ok"


# ─────────────────────────────────────────────────────────────────────────────
# Episode result dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EpisodeResult:
    params: Dict[str, float]
    metrics: Dict[str, float]
    reward: float
    state: np.ndarray
    action: np.ndarray
    log_prob: float
    value: float


# ─────────────────────────────────────────────────────────────────────────────
# MatlabRLOptimizer
# ─────────────────────────────────────────────────────────────────────────────

class MatlabRLOptimizer:
    """
    Contextual-bandit optimizer for guidance/autopilot parameters (PPO-style updates).

    Each episode is an independent pull: sample action → absolute params → single MC
    evaluation → reward.  There is **no** inter-episode state transition, so training
    uses gamma=0 (G_t = R_t) and Advantage = R_t − V(s_t).

    Workflow
    --------
    1. Extract tunable params from the MATLAB script RL_PARAMS block.
    2. Build state from **last-executed** params + physical constants + last metrics.
    3. Sample action → map to bounded absolute params → run MC simulation → reward.
    4. PPO clipped policy update every ``episodes_per_update`` independent episodes.
    5. Persist best params to ParameterExperience.
    """

    def __init__(
        self,
        simulator=None,
        parameter_experience=None,
        reflection_agent=None,
        hidden_dim: int = 64,
        lr_actor: float = 1e-4,
        lr_critic: float = 5e-4,
        gamma: float = 0.0,
        clip_ratio: float = 0.15,
        max_episodes: int = 50,
        episodes_per_update: int = 64,
        nmc_per_eval: int = 10,
        reflect_every: int = 5,
        peak_n_max: float = PEAK_NY_MAX,
        peak_n_penalty: float = 1.0,
        reward_weights: Optional[Dict[str, float]] = None,
        entropy_coef: float = 0.05,
        pitch_log_std: float = -0.7,
        other_log_std: float = -1.6,
        warm_start_std_scale: float = 0.3,
        baseline_reward_threshold: float = 0.5,
        diverge_penalty_episodes: int = 3,
        diverge_log_std_delta: float = -0.5,
        diverge_reward_threshold: float = -9.5,
        early_stop_enabled: bool = True,
        early_stop_min_episodes: int = 25,
        early_stop_patience: int = 15,
        early_stop_plateau_delta: float = 0.05,
        early_stop_use_rule_first: bool = True,
        early_stop_use_multi_criteria_best: bool = True,
        early_stop_peak_only_patience: int = 25,
        fre_config: Optional[FREConfig] = None,
        checkpoint_enabled: bool = True,
        checkpoint_dir: str = "./parameter_experience_base/checkpoints/ppo",
        checkpoint_save_every: int = 10,
        checkpoint_keep_last_n: int = 5,
        rolling_window: int = 20,
        stats_log_every: int = 1,
        jsonl_log_dir: str = "logs",
        amro_enabled: bool = False,
        amro_gradient_threshold: float = 0.5,
        amro_forbidden_zone_radius: float = 0.15,
        amro_history_window: int = 10,
        adaptive_explore_enabled: bool = True,
        adaptive_steepness_threshold: float = 2.0,
        adaptive_log_std_min_scale: float = 0.35,
        adaptive_log_std_max_scale: float = 1.0,
        adaptive_ema_alpha: float = 0.3,
        progress_callback: Optional[Callable[[int, int, Dict[str, Any]], None]] = None,
        allow_internal_fallback: bool = False,
    ):
        self.simulator = simulator
        self.parameter_experience = parameter_experience
        self.reflection_agent = reflection_agent
        self.hidden_dim = hidden_dim
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        # Contextual bandit: episodes are independent; gamma must stay 0.
        self.gamma = 0.0
        if gamma != 0.0:
            logger.warning(
                f"[RL] gamma={gamma} ignored — contextual bandit mode forces gamma=0.0"
            )
        self.clip_ratio = clip_ratio
        self.max_episodes = max_episodes
        self.episodes_per_update = max(1, int(episodes_per_update))
        self.nmc_per_eval = nmc_per_eval
        self.entropy_coef = float(entropy_coef)
        # Cadence at which the reflection agent is consulted during RL.
        # 1 = every episode (default); 2 = every other; etc. Only invoked when
        # there is a NEW best on that episode, so cost stays bounded.
        self.reflect_every = max(1, int(reflect_every))
        self._pitch_log_std = float(pitch_log_std)
        self._other_log_std = float(other_log_std)
        self._warm_start_std_scale = float(warm_start_std_scale)
        self._baseline_reward_threshold = float(baseline_reward_threshold)
        self._diverge_penalty_episodes = max(1, int(diverge_penalty_episodes))
        self._diverge_log_std_delta = float(diverge_log_std_delta)
        self._diverge_reward_threshold = float(diverge_reward_threshold)
        self._early_stop_enabled = bool(early_stop_enabled)
        self._early_stop_min_episodes = max(1, int(early_stop_min_episodes))
        self._early_stop_patience = max(1, int(early_stop_patience))
        self._early_stop_plateau_delta = float(early_stop_plateau_delta)
        self._early_stop_use_rule_first = bool(early_stop_use_rule_first)
        self._early_stop_use_multi_criteria_best = bool(early_stop_use_multi_criteria_best)
        self._early_stop_peak_only_patience = max(1, int(early_stop_peak_only_patience))
        self._fre_config = fre_config or FREConfig()
        self._consecutive_diverge = 0
        self._last_constrained_improve_ep = 0
        self._last_peak_improve_ep = 0
        self._checkpoint_enabled = bool(checkpoint_enabled)
        self._checkpoint_dir = str(checkpoint_dir)
        self._checkpoint_save_every = max(1, int(checkpoint_save_every))
        self._checkpoint_keep_last_n = max(1, int(checkpoint_keep_last_n))
        self._rolling_window = max(1, int(rolling_window))
        self._stats_log_every = max(1, int(stats_log_every))
        self._jsonl_log_dir = str(jsonl_log_dir)
        self._checkpoint_mgr: Optional[PPOCheckpointManager] = None
        self._episode_logger: Optional[EpisodeJsonlLogger] = None
        self._rolling_stats: Optional[RollingEpisodeStats] = None
        self._adaptive_explore_enabled = bool(adaptive_explore_enabled)
        self._adaptive_steepness_threshold = float(adaptive_steepness_threshold)
        self._adaptive_log_std_min = float(adaptive_log_std_min_scale)
        self._adaptive_log_std_max = float(adaptive_log_std_max_scale)
        self._adaptive_ema_alpha = float(adaptive_ema_alpha)
        self.progress_callback = progress_callback
        self._adaptive_ctrl: Optional[AdaptiveExplorationController] = None
        self._ep_prev_params: Dict[str, float] = {}
        self._ep_prev_reward: float = 0.0
        # peak_n_penalty applied via _rw["peak_ny_penalty"] after merge below
        # Reward weights: merged from defaults + caller-supplied overrides
        _rw_defaults: Dict[str, float] = {
            # hit_rate: maximize (0~100%)
            "hit_rate": 5.0,
            # SEP: minimize (m)
            "sep_low_bonus": 3.0,   "sep_low_threshold": 2.0,
            "sep_mid_weight": 1.0,  "sep_mid_threshold": 5.0,
            # PeakNy: soft constraint on MC mean only
            "peak_ny_max": PEAK_NY_MAX,    "peak_ny_mean_max": PEAK_NY_MAX,
            "peak_ny_penalty": 1.0,
            # PM: target range [pm_min, pm_max] degrees
            "pm_bonus": 0.8,        "pm_min": PM_MIN,  "pm_max": PM_MAX,
            "pm_penalty": 3.0,
            # BW: target range [bw_min, bw_max] rad/s
            "bw_bonus": 0.4,        "bw_min": BW_MIN,  "bw_max": BW_MAX,
            "bw_penalty": 1.2,
            # Phase A threshold (SEP > this → no PM/BW terms)
            "sep_survival_threshold": 500.0,
            "peak_ny_progressive_low": 15.0,
            "hard_truncation_peak_g": 35.0,
            "baseline_reset_peak_g": 40.0,
        }
        if reward_weights:
            _rw_defaults.update(reward_weights)
        # Top-level peak_n_max / peak_n_penalty (config.yaml) override reward_weights
        # so there is a single authoritative PeakNy soft-constraint source.
        _rw_defaults["peak_ny_max"] = float(peak_n_max)
        _rw_defaults["peak_ny_mean_max"] = float(_rw_defaults.get("peak_ny_mean_max", peak_n_max))
        _rw_defaults["peak_ny_penalty"] = float(peak_n_penalty)
        self._rw: Dict[str, float] = _rw_defaults
        self._peak_n_max = float(self._rw["peak_ny_mean_max"])
        self._peak_n_penalty = float(self._rw["peak_ny_penalty"])

        self._actor: Optional[ActorNet] = None
        self._critic: Optional[CriticNet] = None
        self._action_keys: List[str] = list(ALL_TUNABLE_PARAM_SPECS.keys())
        self._state_dim: int = 0
        self._action_dim: int = len(self._action_keys)

        self._best_reward: float = -float("inf")
        self._best_params: Dict[str, float] = {}
        self._best_metrics: Dict[str, float] = {}
        self._best_miss_ever: float = float("inf")    # min miss across ALL episodes
        self._best_peak_n_ever: float = float("inf")  # min PeakN across ALL episodes
        self._best_constrained_params: Dict[str, float] = {}
        self._best_constrained_metrics: Dict[str, float] = {}
        self._best_constrained_fitness: float = -1.0
        self._best_borderline_params: Dict[str, float] = {}
        self._best_borderline_metrics: Dict[str, float] = {}
        self._history: List[Dict] = []

        # Stores the stderr of the last Octave/MATLAB subprocess call so that
        # _run_matlab_simulation can inspect it for syntax-error patterns and
        # decide whether to trigger a script repair pass.
        self._last_exec_stderr: str = ""
        # Optional callable: async fn(script_content, error_msg) -> fixed_content
        # Injected by cli_agent when a ModelGenerationAgent is available.
        self.script_repair_fn = None
        self.allow_internal_fallback = bool(allow_internal_fallback)

        self._base_auto: Dict[str, float] = {k: float(v["nominal"]) for k, v in ALL_TUNABLE_PARAM_SPECS.items()}
        self._base_phys: Dict[str, float] = {k: float(v["nominal"]) for k, v in PHYSICAL_PARAM_SPECS.items()}
        self._last_metrics: Dict[str, float] = {k: 0.0 for k in METRIC_SPECS}
        # Last-executed tunable params (feeds s_auto in _build_state).
        self._current_params: Dict[str, float] = copy.deepcopy(self._base_auto)

        if amro_enabled:
            logger.warning(
                "[RL] amro_enabled=True is deprecated and ignored — AMRO action "
                "tampering breaks PPO importance sampling."
            )
        self._amro_enabled = False
        self._adaptive_explorer = None
        self._action_modifier = None
        self._fre = FeasibleRegionExplorer()
        self._fre_reward_modifier = FeasibleRegionRewardModifier(self._fre)
        self._apply_fre_preset(self._fre_config.default)

    def _apply_fre_preset(self, preset: FREPresetConfig) -> None:
        self._fre = FeasibleRegionExplorer(
            initial_sep_threshold=preset.initial_sep,
            initial_hit_rate_threshold=preset.initial_hit_rate,
            target_sep_threshold=preset.target_sep,
            target_hit_rate_threshold=preset.target_hit_rate,
            phase_episodes=preset.phase_episodes,
            num_phases=preset.num_phases,
        )
        self._fre_reward_modifier = FeasibleRegionRewardModifier(self._fre)
        logger.info(
            f"[FRE] preset SEP {preset.initial_sep:.0f}→{preset.target_sep:.0f}m | "
            f"HitRate {preset.initial_hit_rate:.0f}→{preset.target_hit_rate:.0f}% | "
            f"phases={preset.num_phases} x {preset.phase_episodes} ep"
        )

    @staticmethod
    def _resolve_sub_idx(
        mission_conditions: Optional[Dict[str, Any]],
        task_prompt: Optional[str] = None,
    ) -> int:
        if mission_conditions:
            for subs in mission_conditions.values():
                if isinstance(subs, (list, tuple)) and subs:
                    try:
                        return int(subs[0])
                    except (TypeError, ValueError):
                        pass
                if isinstance(subs, int):
                    return int(subs)
        if task_prompt:
            m = re.search(r"SUB_IDX\s*=\s*(\d+)", task_prompt, re.I)
            if m:
                return int(m.group(1))
            m = re.search(r"T\s*4|T4", task_prompt, re.I)
            if m and "T4" in task_prompt.upper().replace(" ", ""):
                return 4
        return 0

    def _configure_fre(
        self,
        mission_conditions: Optional[Dict[str, Any]],
        task_prompt: Optional[str],
        metric_requirements: Optional[Dict[str, Any]] = None,
    ) -> None:
        sub_idx = self._resolve_sub_idx(mission_conditions, task_prompt)
        preset = self._fre_config.preset_for_sub_idx(sub_idx)
        reqs = resolve_requirements(metric_requirements, task_prompt, None)
        if reqs.get("sep_max") is not None:
            preset = FREPresetConfig(
                initial_sep=preset.initial_sep,
                initial_hit_rate=preset.initial_hit_rate,
                target_sep=float(reqs["sep_max"]),
                target_hit_rate=float(reqs.get("hit_rate_min", preset.target_hit_rate)),
                phase_episodes=preset.phase_episodes,
                num_phases=preset.num_phases,
            )
        self._apply_fre_preset(preset)

    def _shrink_log_std_for_warmstart(self) -> None:
        if self._actor is None or self._warm_start_std_scale >= 1.0:
            return
        delta = math.log(max(self._warm_start_std_scale, 0.05))
        for i in range(self._action_dim):
            self._actor.log_std[i] = float(
                np.clip(self._actor.log_std[i] + delta, -4.0, 2.0)
            )
        logger.info(
            f"[RL] Warm-start exploration shrink ×{self._warm_start_std_scale:.2f} "
            f"(pitch_std≈{np.exp(self._actor.log_std[0]):.2f})"
        )

    def _apply_diverge_exploration_penalty(self) -> None:
        if self._actor is None:
            return
        for i in range(self._action_dim):
            self._actor.log_std[i] = float(
                np.clip(
                    self._actor.log_std[i] + self._diverge_log_std_delta,
                    -4.0,
                    2.0,
                )
            )
        logger.info(
            f"[RL] {self._consecutive_diverge} consecutive divergent episodes — "
            f"reducing log_std by {self._diverge_log_std_delta:.2f}"
        )

    def _update_constrained_best(
        self,
        params: Dict[str, float],
        metrics: Dict[str, float],
        ep: int,
    ) -> bool:
        if not is_reliable_sim_metrics(metrics):
            return False
        fitness = self._compute_pe_fitness(metrics)
        improved = is_better_constrained(
            metrics,
            fitness,
            self._best_constrained_metrics or None,
            self._best_constrained_fitness,
        )
        if improved:
            self._best_constrained_params = copy.deepcopy(params)
            self._best_constrained_metrics = copy.deepcopy(metrics)
            self._best_constrained_fitness = fitness
            self._last_constrained_improve_ep = ep + 1
        return improved

    def _select_return_best(self) -> Tuple[Dict[str, float], Dict[str, float], str]:
        if (
            self._early_stop_use_multi_criteria_best
            and self._best_constrained_metrics
        ):
            return (
                copy.deepcopy(self._best_constrained_params),
                copy.deepcopy(self._best_constrained_metrics),
                "constrained",
            )
        if (
            self._early_stop_use_multi_criteria_best
            and self._best_borderline_metrics
            and is_borderline_hit_sep_ok(self._best_borderline_metrics)
            and is_better_borderline_peak(
                self._best_borderline_metrics,
                self._best_metrics or None,
            )
        ):
            return (
                copy.deepcopy(self._best_borderline_params),
                copy.deepcopy(self._best_borderline_metrics),
                "borderline",
            )
        return (
            copy.deepcopy(self._best_params),
            copy.deepcopy(self._best_metrics),
            "reward",
        )

    def _update_borderline_best(
        self,
        params: Dict[str, float],
        metrics: Dict[str, float],
    ) -> None:
        if not is_reliable_sim_metrics(metrics):
            return
        if not is_borderline_hit_sep_ok(metrics):
            return
        if is_better_borderline_peak(metrics, self._best_borderline_metrics or None):
            self._best_borderline_params = copy.deepcopy(params)
            self._best_borderline_metrics = copy.deepcopy(metrics)

    def _init_training_loggers(self, resume_run_id: Optional[str] = None, append_only: bool = False) -> None:
        run_id = resume_run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self._episode_logger = EpisodeJsonlLogger(
            log_dir=self._jsonl_log_dir,
            run_id=run_id,
        )
        path = self._episode_logger.open(append_only=append_only)
        logger.info(f"[RL] JSONL episode log → {path}")
        self._checkpoint_mgr = PPOCheckpointManager(
            checkpoint_dir=self._checkpoint_dir,
            save_every_episodes=self._checkpoint_save_every,
            keep_last_n=self._checkpoint_keep_last_n,
            enabled=self._checkpoint_enabled,
        )
        self._checkpoint_mgr.set_run_id(run_id)
        self._rolling_stats = RollingEpisodeStats(window=self._rolling_window)

    def _log_episode_training(
        self,
        *,
        ep: int,
        max_ep: int,
        new_params: Dict[str, float],
        metrics: Dict[str, float],
        reward: float,
        constrained: bool,
    ) -> None:
        if self._rolling_stats is not None:
            snap = self._rolling_stats.update(
                reward=reward,
                metrics=metrics,
                diverge_reward_threshold=self._diverge_reward_threshold,
            )
            if self._episode_logger is not None:
                self._episode_logger.write_episode(
                    optimizer="PPO",
                    episode=ep + 1,
                    params=new_params,
                    metrics=metrics,
                    reward=reward,
                    fitness=self._compute_pe_fitness(metrics),
                    constrained=constrained,
                    extra={
                        "peak_ny": get_peak_ny(metrics),
                        "peak_ny_max": metrics.get("peak_ny_max", metrics.get("peak_n_max")),
                        "SEP": get_sep(metrics, float("nan")),
                    },
                )
            if (ep + 1) % self._stats_log_every == 0:
                self._rolling_stats.log_summary(ep + 1, max_ep)
                if self._episode_logger is not None:
                    self._episode_logger.write_rolling_stats(ep + 1, snap)
            if self.progress_callback is not None:
                try:
                    self.progress_callback(
                        ep + 1,
                        max_ep,
                        {
                            "mean_reward_last10": snap.get("mean_reward", reward),
                            "mean_peak_ny_max_g": snap.get("mean_peak_ny_max_g", 0.0),
                            "reward": reward,
                            "constrained": constrained,
                        },
                    )
                except Exception as _pcb_exc:
                    logger.debug("[RL] progress_callback failed: %s", _pcb_exc)

    def _maybe_save_checkpoint(
        self,
        *,
        ep: int,
        max_ep: int,
        script_path: Optional[str],
        nmc: int,
        force: bool = False,
    ) -> None:
        if self._checkpoint_mgr is None or not self._checkpoint_enabled:
            return
        ep_num = ep + 1
        if not force and (ep_num % self._checkpoint_save_every != 0):
            return
        jsonl_path = self._episode_logger.path if self._episode_logger else ""
        self._checkpoint_mgr.save(
            self,
            episode=ep_num,
            max_episodes=max_ep,
            script_path=script_path,
            nmc=nmc,
            extra={"jsonl_path": jsonl_path},
        )

    # ── state / action helpers ────────────────────────────────────────────────

    def _build_state(
        self,
        auto: Dict[str, float],
        phys: Dict[str, float],
        metrics: Dict[str, float],
    ) -> np.ndarray:
        import math as _math
        parts: List[float] = []
        for k in self._action_keys:
            spec = ALL_TUNABLE_PARAM_SPECS[k]
            mid = (spec["min"] + spec["max"]) / 2.0
            half = (spec["max"] - spec["min"]) / 2.0
            raw = float(auto.get(k, spec["nominal"]))
            val = spec["nominal"] if (_math.isnan(raw) or _math.isinf(raw)) else raw
            parts.append((val - mid) / half)
        for k, spec in PHYSICAL_PARAM_SPECS.items():
            raw = float(phys.get(k, spec["nominal"]))
            val = spec["nominal"] if (_math.isnan(raw) or _math.isinf(raw)) else raw
            parts.append(val / spec["scale"])
        for k, spec in METRIC_SPECS.items():
            raw = float(metrics.get(k, 0.0))
            val = 0.0 if (_math.isnan(raw) or _math.isinf(raw)) else raw
            parts.append(val / max(spec["scale"], 1e-6))
        return np.array(parts, dtype=np.float32)

    def _action_to_params(self, action: np.ndarray) -> Dict[str, float]:
        """Map raw Actor action to bounded absolute parameter values (no AMRO)."""
        result: Dict[str, float] = {}
        for i, k in enumerate(self._action_keys):
            spec = ALL_TUNABLE_PARAM_SPECS[k]
            mid = (spec["min"] + spec["max"]) / 2.0
            half = (spec["max"] - spec["min"]) / 2.0
            val = float(np.clip(mid + float(action[i]) * half, spec["min"], spec["max"]))
            result[k] = val
        return result

    # ── lightweight requirements check (no LLM) ──────────────────────────────

    @staticmethod
    def _requirements_met(metrics: Dict[str, float], reqs: Dict[str, Any]) -> bool:
        """Return True iff every non-None entry in *reqs* is satisfied by *metrics*.

        Keys: ``hitrate`` (>=), ``missmean`` (<=), ``peak_n`` (<=, only when >0),
        ``pm`` (>=).
        """
        seed_metrics = metrics.get("mc_seed_results") if isinstance(metrics, dict) else None
        if isinstance(seed_metrics, list) and seed_metrics:
            return all(
                MatlabRLOptimizer._requirements_met(dict(item), reqs)
                for item in seed_metrics
                if isinstance(item, dict)
            )
        if reqs.get("hitrate")  is not None and metrics.get("hit_rate",      0.0)           < reqs["hitrate"]:  return False
        if reqs.get("missmean") is not None:
            miss = get_sep(metrics, float("inf"))
            if math.isnan(miss) or math.isinf(miss) or miss > reqs["missmean"]:
                return False
        if reqs.get("peak_n")  is not None:
            v = get_peak_ny(metrics)
            if v > 0.0 and v > reqs["peak_n"]: return False
        if reqs.get("pm")      is not None and metrics.get("pitch_PM_min", metrics.get("pitch_PM", 0.0)) < reqs["pm"]: return False
        return True

    # ── reward ───────────────────────────────────────────────────────────────

    @staticmethod
    def _is_anomalous(metrics: Dict[str, float]) -> bool:
        """Return True when metrics contain NaN/inf SEP (simulation diverged)."""
        sep = metrics.get("SEP", metrics.get("miss_distance", 0.0))
        try:
            sep_f = float(sep)
        except (TypeError, ValueError):
            return True
        return math.isnan(sep_f) or math.isinf(sep_f)

    def _compute_reward(self, metrics: Dict[str, float]) -> float:
        """Tiered reward for monte_carlo_single.m metrics.

        Hard truncation on dynamics divergence (PeakNy mean > 50g or SEP NaN/Inf).
        Phase A (SEP > sep_survival_threshold): SEP + PeakNy only — no PM/BW.
        Phase B: full fine-tuning terms; PM/BW scaled down when SEP is still large.
        """
        rw = self._rw

        # Analytical fallback must not reward fake PM/BW from w1 scaling.
        if metrics.get("_sim_source") == "fallback":
            _sep_fb = metrics.get("SEP", metrics.get("miss_distance", rw["sep_mid_threshold"]))
            try:
                _sep_fb = float(_sep_fb)
            except (TypeError, ValueError):
                _sep_fb = rw["sep_mid_threshold"]
            if math.isnan(_sep_fb) or math.isinf(_sep_fb):
                _sep_fb = rw["sep_mid_threshold"]
            _r = -float(np.clip(_sep_fb / 1000.0, 0.0, 10.0))
            logger.debug("[Reward] fallback sim → SEP-only reward=%.3f", _r)
            return float(np.clip(_r, -10.0, 10.0))

        # ── Hard truncation (must run BEFORE sentinel substitution) ──────────
        _sep_raw = metrics.get("SEP", metrics.get("miss_distance", float("nan")))
        _pn_raw  = get_peak_ny(metrics)
        try:
            _sep_check = float(_sep_raw)
        except (TypeError, ValueError):
            _sep_check = float("nan")
        try:
            _pn_check = float(_pn_raw)
        except (TypeError, ValueError):
            _pn_check = float("nan")

        if (
            (not math.isnan(_pn_check) and _pn_check > float(self._rw.get("hard_truncation_peak_g", 50.0)))
            or math.isnan(_sep_check)
            or math.isinf(_sep_check)
            or math.isnan(_pn_check)
        ):
            logger.debug(
                "[Reward] Hard truncation: peak_ny=%s SEP=%s → %.1f",
                _pn_raw, _sep_raw, HARD_REWARD_PENALTY,
            )
            return HARD_REWARD_PENALTY

        hit_rate = float(metrics.get("hit_rate", 0.0))
        sep      = _sep_check
        pny_mean = get_peak_ny(metrics)
        peak_ny  = pny_mean
        pitch_pm = float(metrics.get("pitch_PM", 0.0))
        pitch_bw = float(metrics.get("pitch_BW", 0.0))

        if math.isnan(hit_rate) or math.isinf(hit_rate):  hit_rate = 0.0
        if math.isnan(pny_mean) or math.isinf(pny_mean):  pny_mean = 0.0
        if math.isnan(peak_ny)  or math.isinf(peak_ny):   peak_ny  = 0.0
        if math.isnan(pitch_pm) or math.isinf(pitch_pm):  pitch_pm = 0.0
        if math.isnan(pitch_bw) or math.isinf(pitch_bw):  pitch_bw = 0.0

        reward = 0.0
        _survival_thr = float(rw.get("sep_survival_threshold", 500.0))
        survival_phase = sep > _survival_thr
        mean_limit = float(rw.get("peak_ny_mean_max", rw.get("peak_ny_max", PEAK_NY_MAX)))

        # ── SEP absolute priority: scaled negative term (capped) ─────────────
        r_sep = -float(np.clip(sep / 1000.0, 0.0, 10.0))
        reward += r_sep

        # ── PeakNy soft penalty (MC mean only) ──────────────────────────────
        def _apply_peak_penalty(val: float, limit: float) -> None:
            nonlocal reward
            if val <= 0.0 or limit <= 0.0 or val <= limit:
                return
            overshoot = (val - limit) / limit
            if overshoot < 0.05:
                penalty_scale = 2.0
            elif overshoot < 0.2:
                penalty_scale = 1.0
            elif overshoot < 0.5:
                penalty_scale = 1.5
            else:
                penalty_scale = 2.5
            reward -= rw["peak_ny_penalty"] * penalty_scale * min(overshoot, 2.0)
            overshoot_g = val - limit
            if 0.0 < overshoot_g <= 2.0:
                reward -= rw["peak_ny_penalty"] * 2.5 * overshoot_g

        _apply_peak_penalty(pny_mean, mean_limit)

        if survival_phase:
            # Phase A — survival: no hit_rate / PM / BW; force approach to target first
            return float(np.clip(reward, -10.0, 10.0))

        # ── Phase B — fine-tuning (SEP <= sep_survival_threshold) ────────────

        # Hit rate
        reward += rw["hit_rate"] * (hit_rate / 100.0)

        # Low-SEP bonus (only meaningful when already in range)
        _sl = rw["sep_low_threshold"]
        _sm = rw["sep_mid_threshold"]
        if sep < _sl:
            reward += rw["sep_low_bonus"]
        elif sep < _sm:
            reward += rw["sep_mid_weight"] * (1.0 - (sep - _sl) / (_sm - _sl))

        # PeakNy progressive guide (fine-tuning only, toward mean limit)
        _pn_progressive_low  = rw.get("peak_ny_progressive_low", 15.0)
        _pn_progressive_high = mean_limit
        if pny_mean > 0.0 and _pn_progressive_low <= pny_mean <= _pn_progressive_high:
            reward += 0.5 * (_pn_progressive_high - pny_mean) / (_pn_progressive_high - _pn_progressive_low)
        elif pny_mean > 0.0 and pny_mean > _pn_progressive_high:
            reward += 0.05 * max(-1.0, (_pn_progressive_high - pny_mean) / _pn_progressive_high)

        # PM/BW terms fade in as SEP improves (prevents hacking at large SEP).
        _fine_scale = max(0.0, 1.0 - sep / max(_survival_thr, 1.0))

        # PM (target range)
        _pm_min = rw["pm_min"]
        _pm_max = rw["pm_max"]
        if _pm_min <= pitch_pm <= _pm_max:
            reward += rw["pm_bonus"] * _fine_scale
        elif pitch_pm < 0:
            reward -= rw["pm_penalty"] * _fine_scale * (3.0 + abs(pitch_pm) / 15.0)
        else:
            dist_pm = max(_pm_min - pitch_pm, pitch_pm - _pm_max, 0.0)
            reward -= rw["pm_penalty"] * _fine_scale * min(5.0, dist_pm / max(_pm_min, 1.0))

        # BW (target range)
        _bw_min = rw["bw_min"]
        _bw_max = rw["bw_max"]
        if _bw_min <= pitch_bw <= _bw_max:
            reward += rw["bw_bonus"] * _fine_scale
        else:
            dist_bw = max(_bw_min - pitch_bw, pitch_bw - _bw_max, 0.0)
            reward -= rw["bw_penalty"] * _fine_scale * min(5.0, dist_bw / max(_bw_max, 1.0))

        return float(np.clip(reward, -10.0, 10.0))

    # ── network init ──────────────────────────────────────────────────────────

    # Pitch-channel params that dominate PM & BW — get higher initial
    # exploration noise so RL discovers the PM/BW sweet spot faster.
    _PITCH_KEYS = {"w1", "zeta1", "tao1"}

    def _init_networks(self) -> None:
        dummy = self._build_state(self._base_auto, self._base_phys, self._last_metrics)
        self._state_dim = len(dummy)
        self._actor  = ActorNet(self._state_dim, self._action_dim, self.hidden_dim, self.lr_actor)
        self._critic = CriticNet(self._state_dim, self.hidden_dim, self.lr_critic)

        # Per-dimension exploration: pitch params (w1/zeta1/tao1) get 2× more
        # exploration noise because PM & BW are almost entirely determined by
        # these three.  Other params start with lower noise to avoid wasteful
        # exploration in dimensions that barely affect stability margins.
        _pitch_log_std = self._pitch_log_std
        _other_log_std = self._other_log_std
        for i, k in enumerate(self._action_keys):
            self._actor.log_std[i] = (
                _pitch_log_std if k in self._PITCH_KEYS else _other_log_std
            )
        logger.info(
            f"RL networks initialised: state_dim={self._state_dim}, "
            f"action_dim={self._action_dim}, "
            f"mode=contextual_bandit(gamma=0), "
            f"batch={self.episodes_per_update}, entropy_coef={self.entropy_coef}, "
            f"pitch_std={np.exp(_pitch_log_std):.2f}, "
            f"other_std={np.exp(_other_log_std):.2f}"
        )

    # ── parameter extraction from script ─────────────────────────────────────

    def extract_params_from_script(self, script_path: str) -> Dict[str, float]:
        """Load a MATLAB script and extract parameters dynamically."""
        try:
            with open(script_path, "r", encoding="utf-8", errors="ignore") as fh:
                content = fh.read()
        except FileNotFoundError:
            logger.warning(f"MATLAB script not found: {script_path}")
            return {}
        self._base_auto = extract_autopilot_params(content)
        self._base_phys = extract_physical_params(content)
        combined = {f"auto_{k}": v for k, v in self._base_auto.items()}
        combined.update({f"phys_{k}": v for k, v in self._base_phys.items()})
        logger.info(f"Extracted {len(combined)} params from {script_path}")
        return combined

    # ── simulation helpers ────────────────────────────────────────────────────

    async def _run_simulation_with_params(
        self,
        auto_params: Dict[str, float],
        mission_conditions: Optional[Dict[str, List[int]]],
        script_path: Optional[str],
        nmc: int,
    ) -> Dict[str, float]:
        try:
            from multi_agent.config_loader import get_config as _get_config

            _sim_cfg = _get_config().simulation
            _mc_data_path = str(getattr(_sim_cfg, "mc_data_path", "") or "")
            _mc_root = str(getattr(_sim_cfg, "mc_root", "") or "")
            _mc_scenario = str(getattr(_sim_cfg, "mc_scenario", "") or "")
            _mc_seeds = list(getattr(_sim_cfg, "mc_seeds", []) or [])
        except Exception:
            _mc_data_path = ""
            _mc_root = ""
            _mc_scenario = ""
            _mc_seeds = []
        if _mc_seeds and _mc_root and _mc_scenario:
            _external_script = script_path or (
                "./knowledge_base/matlab/guidance/monte_carlo_single.m"
            )
            try:
                from multi_agent.tools.simulation_tool import run_mc_and_across_seeds

                _n_runs = int(nmc or getattr(self, "nmc_per_eval", 100) or 100)
                try:
                    import os as _os_nruns

                    _env_n = int(_os_nruns.environ.get("GSM_MC_SEARCH_N_RUNS", "0") or 0)
                    if _env_n > 0:
                        _n_runs = _env_n
                    elif bool(getattr(_sim_cfg, "mc_tiered_eval", False)):
                        _n_runs = int(
                            getattr(_sim_cfg, "mc_search_n_runs", _n_runs) or _n_runs
                        )
                except Exception:
                    pass

                _ext = run_mc_and_across_seeds(
                    _external_script,
                    mc_root=_mc_root,
                    scenario=_mc_scenario,
                    seeds=[int(s) for s in _mc_seeds],
                    split="dev",
                    n_runs=_n_runs,
                    kill_radius=10.0,
                    dp_params=auto_params,
                )
                _n_calls = int(_ext.get("mc_calls", len(_mc_seeds)) or len(_mc_seeds))
                self._external_mc_calls = int(getattr(self, "_external_mc_calls", 0)) + _n_calls
                self._external_mc_sim_time_sec = float(
                    getattr(self, "_external_mc_sim_time_sec", 0.0)
                ) + float(_ext.get("sim_time_sec", 0.0) or 0.0)
                if _ext.get("success") or _ext.get("feasible"):
                    return _tag_sim_metrics(
                        {
                            "hit_rate": float(_ext.get("hit_rate", 0.0)),
                            "SEP": float(_ext.get("SEP", 0.0)),
                            "miss_distance": float(_ext.get("SEP", 0.0)),
                            "peak_ny": float(_ext.get("peak_nz", _ext.get("peak_ny", 0.0))),
                            "pitch_PM": float(_ext.get("PM", 0.0)),
                            "pitch_BW": float(_ext.get("BW", 0.0)),
                            "_sim_time_sec": float(_ext.get("sim_time_sec", 0.0)),
                            "feasible": bool(_ext.get("feasible", False)),
                            "mc_seed_results": _ext.get("seed_results") or [],
                        },
                        "external_mc_and",
                    )
                logger.warning(
                    "External MC AND failed: %s",
                    _ext.get("error") or _ext.get("error_msg"),
                )
                return _unreliable_sim_stub("external_mc_and_error")
            except (asyncio.CancelledError, KeyboardInterrupt):
                raise
            except Exception as exc:
                logger.warning("External MC AND branch failed: %s", exc)
                return _unreliable_sim_stub("external_mc_and_error")
        if _mc_data_path:
            _external_script = script_path or (
                "./knowledge_base/matlab/guidance/monte_carlo_single.m"
            )
            try:
                from multi_agent.tools.simulation_tool import run_mc_with_external_data

                _ext = run_mc_with_external_data(
                    _external_script,
                    _mc_data_path,
                    n_runs=100,
                    kill_radius=10.0,
                    dp_params=auto_params,
                )
                self._external_mc_calls = int(getattr(self, "_external_mc_calls", 0)) + 1
                self._external_mc_sim_time_sec = float(
                    getattr(self, "_external_mc_sim_time_sec", 0.0)
                ) + float(_ext.get("sim_time_sec", 0.0) or 0.0)
                if _ext.get("success"):
                    return _tag_sim_metrics(
                        {
                            "hit_rate": float(_ext.get("hit_rate", 0.0)),
                            "SEP": float(_ext.get("SEP", 0.0)),
                            "miss_distance": float(_ext.get("SEP", 0.0)),
                            "peak_ny": float(_ext.get("peak_nz", 0.0)),
                            "pitch_PM": float(_ext.get("PM", 0.0)),
                            "pitch_BW": float(_ext.get("BW", 0.0)),
                            "_sim_time_sec": float(_ext.get("sim_time_sec", 0.0)),
                        },
                        "external_mc",
                    )
                logger.warning(
                    "External MC simulation failed: %s",
                    _ext.get("error") or _ext.get("error_msg"),
                )
                return _unreliable_sim_stub("external_mc_error")
            except (asyncio.CancelledError, KeyboardInterrupt):
                raise
            except Exception as exc:
                logger.warning("External MC simulation branch failed: %s", exc)
                return _unreliable_sim_stub("external_mc_error")

        if script_path and os.path.exists(script_path) and self.simulator:
            try:
                return await self._run_matlab_simulation(
                    auto_params, script_path, mission_conditions, nmc
                )
            except (asyncio.CancelledError, KeyboardInterrupt):
                raise
            except Exception as exc:
                logger.warning(f"MATLAB run failed ({exc}), falling back to internal sim")

        if self.simulator:
            try:
                return await self._run_internal_simulation(auto_params, mission_conditions)
            except (asyncio.CancelledError, KeyboardInterrupt):
                raise
            except Exception as exc:
                logger.warning(f"Internal sim failed: {exc}")

        # Last-resort analytical fallback — *still depends on auto_params* so the
        # RL gradient is non-degenerate even when no simulator is available.
        return await self._run_internal_simulation(auto_params, mission_conditions)

    @staticmethod
    def _cleanup_stale_temp_files(directory: str) -> None:
        """Remove stale RL temp files left over from a crashed previous run.

        Sweeps *directory* for files matching any of the three RL temp-file
        naming conventions (legacy ``_rl_tmp_*``, new ``rltmp_*``, function-
        style ``*_rltmp_*``).  Best-effort: silently ignores any errors so
        cleanup never blocks an RL run.
        """
        if not directory or not os.path.isdir(directory):
            return
        removed: List[str] = []
        try:
            for fname in os.listdir(directory):
                if not fname.endswith(".m"):
                    continue
                if (
                    fname.startswith("_rl_tmp")     # legacy timestamp pattern
                    or fname.startswith("rltmp_")   # new uuid pattern (plain script)
                    or "_rltmp_" in fname           # function-file pattern
                ):
                    try:
                        os.remove(os.path.join(directory, fname))
                        removed.append(fname)
                    except OSError:
                        pass
        except OSError:
            return
        if removed:
            logger.info(
                f"  [RL] Cleaned up {len(removed)} stale temp file(s) "
                f"in '{directory}': {removed[:5]}"
                f"{' …' if len(removed) > 5 else ''}"
            )

    # Compiled once at class load.  Matches a full template line that calls
    # one of the plot_* helper functions, optionally guarded by `if run_X,`.
    # Used by ``_strip_plot_calls_for_rl`` to neutralise plotting in RL.
    #
    # IMPORTANT: indent capture uses ``[ \t]*`` (horizontal whitespace only),
    # not ``\s*`` — the latter is greedy across ``\n``, so on a blank line
    # preceding ``if run_A, plot_*(…)`` the engine would erroneously anchor
    # at the blank line and consume the line break, mis-targeting the
    # substitution.
    _PLOT_CALL_RE = re.compile(
        r"^([ \t]*)(?:if\s+\w+\s*,[ \t]*)?"
        r"plot_(?:category_results|category_robust|multi_point_results|"
        r"param_scan_results|cross_category_summary|cross_category_robust)"
        r"\s*\(",
        re.MULTILINE,
    )

    # Matches lines (with optional LHS assignment) that call expensive
    # pre-analysis routines which run unconditionally before the A-F MC
    # loops and are not needed by the RL inner loop.
    # Template lines stripped:
    #   robNom      = compute_robust_metrics(960, 7000, struct(), struct());
    #   print_robust_one('Nominal ...', robNom);
    #   multiPtReport = multi_point_trajectory_analysis(nom);
    #   scanReport  = param_perturbation_scan();
    # None of these outputs feed into run_category_mc(); they are only
    # consumed by print/plot helpers that are themselves already stripped.
    _HEAVY_ANALYSIS_RE = re.compile(
        r"^([ \t]*)(?:[\w]+\s*=\s*)?"
        r"(?:"
        r"compute_robust_metrics\s*\(|"
        r"print_robust_one\s*\(|"
        r"multi_point_trajectory_analysis\s*\(|"
        r"param_perturbation_scan\s*\("
        r")",
        re.MULTILINE,
    )

    @classmethod
    def _strip_heavy_analysis_for_rl(cls, content: str) -> str:
        """Comment out heavyweight pre-analysis calls that run unconditionally
        before the A-F MC loops.  Safe to skip: their return values are only
        used by print/plot helpers (also stripped).  Typical saving: 10-16
        minutes per RL episode on a cold MATLAB in-process session."""
        spans = list(cls._HEAVY_ANALYSIS_RE.finditer(content))
        if not spans:
            return content
        new_content = content
        for m in reversed(spans):
            ls = m.start()
            le = new_content.find("\n", ls)
            if le == -1:
                le = len(new_content)
            full = new_content[ls:le]
            indent = m.group(1)
            stripped = full[len(indent):].rstrip("\r")
            new_content = (
                new_content[:ls]
                + f"{indent}% [RL skip] {stripped}"
                + new_content[le:]
            )
        logger.debug(
            f"_strip_heavy_analysis_for_rl: skipped {len(spans)} pre-analysis call(s)"
        )
        return new_content

    @classmethod
    def _strip_plot_calls_for_rl(cls, content: str) -> str:
        """Comment out top-level ``plot_*`` calls in *content* so the RL
        inner loop skips Octave figure rendering.

        The KB template ``MC_gongkuang_simulation_robust_all.m`` calls 6
        plotting helper functions after the simulation completes, each of
        which renders multiple ``figure()``s.  In a tight RL loop where we
        only need the metrics that ``print_category_summary`` already
        ``fprintf``ed (Hit, Miss, Ju, PM, GM, BW, TermAngErr), the entire
        plot phase is wasted work — typically 30+ figures per episode at
        ~200-500 ms each on Octave's Qt offscreen toolkit.

        This routine textually rewrites every line that begins (after
        optional whitespace and an optional ``if run_X,`` guard) with one
        of the six plot helper names so the line becomes an Octave comment
        starting with ``% [RL stripped]``.  ``print_category_summary``
        calls are left intact so RL metric parsing still works.

        Idempotent: a second pass adds no further changes because already
        commented lines no longer match the regex (which anchors on
        ``plot_<name>\\s*(``).
        """
        n_before = len(cls._PLOT_CALL_RE.findall(content))
        if not n_before:
            return content

        def _comment_full_line(match: "re.Match[str]") -> str:
            line_start = match.start()
            # Find the end of this line (handles \n / \r\n / EOF transparently)
            line_end = content.find("\n", line_start)
            if line_end == -1:
                line_end = len(content)
            full_line = content[line_start:line_end]
            indent = match.group(1)
            stripped = full_line[len(indent):].rstrip("\r")
            return f"{indent}% [RL stripped] {stripped}"

        # Iterative substitution because we need access to the *full* line,
        # not just the regex match span.  Walk backwards so earlier
        # substitutions don't shift later match offsets.
        spans = [m for m in cls._PLOT_CALL_RE.finditer(content)]
        new_content = content
        for m in reversed(spans):
            line_start = m.start()
            line_end   = new_content.find("\n", line_start)
            if line_end == -1:
                line_end = len(new_content)
            full_line  = new_content[line_start:line_end]
            indent     = m.group(1)
            stripped   = full_line[len(indent):].rstrip("\r")
            replacement = f"{indent}% [RL stripped] {stripped}"
            new_content = (
                new_content[:line_start] + replacement + new_content[line_end:]
            )

        logger.debug(
            f"_strip_plot_calls_for_rl: commented out {n_before} plot_* call line(s)"
        )
        return new_content

    async def _run_matlab_simulation(
        self,
        auto_params: Dict[str, float],
        script_path: str,
        mission_conditions: Optional[Dict[str, List[int]]],
        nmc: int,
    ) -> Dict[str, float]:
        with open(script_path, "r", encoding="utf-8", errors="ignore") as fh:
            content = fh.read()

        # ── Defensive content validation ────────────────────────────────────
        # If `script_path` was set incorrectly (e.g. a stale natural-language
        # ".m" file produced by a buggy upstream code path landed in the
        # matlab_scripts dir), bail out *before* writing it into a temp file
        # under our control.  This prevents the kind of bug where a temp
        # file like `_rl_tmp_<digits>.m` would be created containing
        # natural-language prose copied verbatim from a bogus source.
        is_valid, reason = _looks_like_matlab_script(content)
        if not is_valid:
            logger.error(
                f"  [RL] Refusing to run on '{script_path}': "
                f"content does not look like a MATLAB script ({reason}). "
                f"This usually means the file is LLM-generated prose, a "
                f"markdown doc, or otherwise corrupted. Falling back to "
                f"the internal Python simulator for this episode."
            )
            return await self._run_internal_simulation(auto_params, mission_conditions)

        # ── Reject scripts that cannot drive real MATLAB MC simulation ─────
        _rl_ok, _rl_reason = validate_rl_script_content(content)
        if not _rl_ok:
            logger.error(
                f"  [RL] Refusing to run on '{os.path.basename(script_path)}': "
                f"{_rl_reason}. Falling back to internal Python simulator."
            )
            return await self._run_internal_simulation(auto_params, mission_conditions)
        # ────────────────────────────────────────────────────────────────────

        # ── Strategy A: patch RL_PARAMS_BEGIN block (monte_carlo_single style) ──
        rl_block_pat = re.compile(
            r"(%%.*?RL_PARAMS_BEGIN.*?\n)(.*?)(%%.*?RL_PARAMS_END)",
            re.DOTALL,
        )
        rl_block_match = rl_block_pat.search(content)
        if rl_block_match:
            block = rl_block_match.group(2)
            new_block = block
            patched_rl = []
            for py_key, ml_var in RL_PARAM_TO_MATLAB.items():
                v = auto_params.get(py_key)
                if v is None:
                    continue
                new_block, n = re.subn(
                    rf"({re.escape(ml_var)}\s*=\s*)[-+]?\d+(?:\.\d*)?(?:[eE][-+]?\d+)?(\s*;)",
                    rf"\g<1>{v:.6f}\2",
                    new_block,
                )
                if n:
                    patched_rl.append(f"{ml_var}={v:.4f}")
                else:
                    # Variable missing from block — append it
                    new_block += f"{ml_var} = {v:.6f};\n"
                    patched_rl.append(f"+{ml_var}={v:.4f}")
            content = rl_block_pat.sub(
                lambda m: m.group(1) + new_block + m.group(3), content, count=1
            )
            logger.debug(f"  [RL] RL_PARAMS block patched: {patched_rl}")
        else:
            # ── Strategy B: legacy dp.* patching for old-style templates ────
            _unpatched: Dict[str, float] = {}
            for k, v in auto_params.items():
                new_content = re.sub(
                    rf"(dp\.{k}\s*=\s*)[-+]?\d+(?:\.\d*)?(?:[eE][-+]?\d+)?\s*;",
                    f"dp.{k} = {v:.6f};",
                    content,
                )
                if new_content == content:
                    _unpatched[k] = v
                else:
                    content = new_content
            if _unpatched:
                _struct_pat = re.compile(r"(dp\s*=\s*struct\s*\([^)]*\)\s*;)", re.DOTALL)
                _inject = "\n".join(f"dp.{k} = {v:.6f};" for k, v in _unpatched.items())
                content, _n = _struct_pat.subn(r"\1\n" + _inject, content, count=1)
                if not _n:
                    content = _inject + "\n" + content
                    logger.warning(f"  [RL] No dp=struct; prepended {len(_unpatched)} fields")

        _ny_lim_val = auto_params.get("ny_lim")
        if _ny_lim_val is not None and re.search(r"\bny_lim\s*=", content):
            content = patch_ny_lim_in_script(content, float(_ny_lim_val))

        # ── Patch N_MC (monte_carlo_single) or Nmc (legacy) ─────────────────
        content = re.sub(r"\bN_MC\s*=\s*\d+\s*;", f"N_MC = {nmc};", content)
        content = re.sub(r"\bNmc\s*=\s*\d+\s*;",  f"Nmc = {nmc};",  content)

        # ── Patch RUN_CASE / SUB_IDX (monte_carlo_single) ──────────────────
        if mission_conditions is not None:
            cond_str = build_matlab_conditions_str(mission_conditions)
            rc_m = re.search(r"RUN_CASE='([A-Za-z]+)';SUB_IDX=(\d+);", cond_str)
            if rc_m:
                rc_val, si_val = rc_m.group(1), rc_m.group(2)
                content = re.sub(
                    r"RUN_CASE\s*=\s*'[A-Za-z]+'",
                    f"RUN_CASE='{rc_val}'", content
                )
                content = re.sub(r"SUB_IDX\s*=\s*\d+", f"SUB_IDX={si_val}", content)
                if not getattr(self, "_rl_mission_logged", False):
                    logger.info(
                        f"  [RL] Mission: RUN_CASE='{rc_val}' SUB_IDX={si_val} nmc={nmc}"
                    )
                    self._rl_mission_logged = True
                else:
                    logger.debug(
                        f"  [RL] Mission patched: RUN_CASE='{rc_val}' SUB_IDX={si_val}"
                    )

        # ── Strip plot_results call (monte_carlo_single has one call) ────────
        content = re.sub(
            r"^([ \t]*)plot_results\s*\([^;]*\)\s*;?\s*$",
            r"\1% [RL stripped] plot_results",
            content, flags=re.MULTILINE,
        )
        # Legacy: strip old-style plot_* helpers
        content = self._strip_plot_calls_for_rl(content)

        # ── Strip clc / close all (interfere with engine stdout capture) ────
        # In MATLAB Engine API on Windows, 'clc' can disrupt fprintf redirection
        # to the Python StringIO buffer, causing empty stdout.  Safe to remove
        # for batch/headless execution.
        content = re.sub(r"\bclc\b\s*;?", "", content)
        content = re.sub(r"\bclose\s+all\b\s*;?", "", content)

        tmp_dir = os.path.dirname(os.path.abspath(script_path))

        # If the template is a *function* file, the temp file must be named after
        # the function so MATLAB/Octave can resolve it by name.  We append a
        # short uuid hex so concurrent RL workers don't clobber each other.
        unique_tag = uuid.uuid4().hex[:8]
        func_name  = _extract_matlab_function_name(content)
        if func_name:
            new_func = f"{func_name}_rltmp_{unique_tag}"
            tmp_path = os.path.join(tmp_dir, f"{new_func}.m")
            # Rename the function declaration inside to match
            content = re.sub(
                r"^function\s+" + re.escape(func_name),
                f"function {new_func}",
                content, count=1, flags=re.MULTILINE,
            )
            call_name = new_func
        else:
            # Plain script: use a basename that is itself a valid MATLAB/Octave
            # identifier (letter-prefix + hex chars, no leading underscore, no
            # all-digit suffix) so MATLAB's run() / Octave's source() never
            # mis-parse the filename as numeric or invalid identifier.
            tmp_path  = os.path.join(tmp_dir, f"rltmp_{unique_tag}.m")
            call_name = None

        # Syntax-error keywords used to distinguish parse/compile errors from
        # runtime errors (timeout, no output) so we can trigger a repair pass.
        _SYNTAX_ERR_PATTERNS = (
            "无效表达式", "invalid expression",
            "parse error", "syntax error",
            "unbalanced", "unexpected",
            "不匹配的分隔符", "mismatched",
            "此类型的变量不支持", "undefined function",
            # Octave: illegal use of reserved keyword (e.g. spurious function-level 'end')
            "非法使用保留关键字", "illegal use of reserved keyword",
            "非法使用", "reserved keyword",
        )

        def _is_syntax_error(stderr_text: str) -> bool:
            low = stderr_text.lower()
            return any(p.lower() in low for p in _SYNTAX_ERR_PATTERNS)

        try:
            with open(tmp_path, "w", encoding="utf-8") as fh:
                fh.write(content)

            # ── Diagnostic: verify the patched RL_PARAMS actually appear in the file ──
            _diag_block = re.search(r"RL_PARAMS_BEGIN.*?RL_PARAMS_END", content, re.DOTALL)
            if _diag_block:
                _diag_w1_m = re.search(r"rl_w1\s*=\s*([\d.]+)", _diag_block.group())
                logger.debug(
                    f"  [RL DIAG] tmp_path={os.path.basename(tmp_path)} "
                    f"call_name={call_name} "
                    f"rl_w1_in_file={_diag_w1_m.group(1) if _diag_w1_m else 'NOT_FOUND'}"
                )

            # ── First attempt ────────────────────────────────────────────────
            self._last_exec_stderr = ""
            stdout_text = await self._exec_matlab_subprocess(tmp_path, call_name)

            if not stdout_text:
                stderr_snap = self._last_exec_stderr
                # ── Retry path 1: re-sanitise the temp file content ──────────
                # Applies _sanitize_matlab_guidance_script to catch issues the
                # pre-write pass may have missed (e.g., from RL patching).
                if _is_syntax_error(stderr_snap):
                    logger.info(
                        "  [RL] Syntax error detected – re-sanitising script and retrying."
                    )
                    try:
                        from multi_agent.tools.simulation_tool import _sanitize_matlab_guidance_script as _san
                        fixed_content, _san_notes = _san(content)
                        if fixed_content != content:
                            content = fixed_content
                            with open(tmp_path, "w", encoding="utf-8") as fh:
                                fh.write(content)
                            stdout_text = await self._exec_matlab_subprocess(tmp_path, call_name)
                    except Exception as _san_exc:
                        logger.debug(f"  [RL] Re-sanitise failed: {_san_exc}")

                # ── Retry path 2: LLM repair via injected script_repair_fn ───
                if not stdout_text and _is_syntax_error(stderr_snap) and self.script_repair_fn:
                    logger.info(
                        "  [RL] Calling script repair agent to fix syntax error…"
                    )
                    try:
                        repaired = await self.script_repair_fn(
                            content, stderr_snap[:1500]
                        )
                        if repaired and repaired != content:
                            content = repaired
                            with open(tmp_path, "w", encoding="utf-8") as fh:
                                fh.write(content)
                            # Also update the source script file so future runs benefit
                            with open(script_path, "w", encoding="utf-8") as fh:
                                fh.write(content)
                            logger.info(
                                "  [RL] Script repaired and written back to: "
                                f"{os.path.basename(script_path)}"
                            )
                            stdout_text = await self._exec_matlab_subprocess(tmp_path, call_name)
                    except Exception as _rep_exc:
                        logger.warning(f"  [RL] Script repair failed: {_rep_exc}")

                if not stdout_text and not _is_syntax_error(stderr_snap):
                    logger.info(
                        "[RL] MATLAB produced no stdout (timeout/empty) — retry once"
                    )
                    stdout_text = await self._exec_matlab_subprocess(tmp_path, call_name)

                if not stdout_text:
                    if self.allow_internal_fallback:
                        logger.warning(
                            "MATLAB subprocess produced no output; using internal fallback"
                        )
                        _fb_result = await self._run_internal_simulation(
                            auto_params, mission_conditions
                        )
                        logger.debug(
                            f"  [RL DIAG] internal_fallback metrics: "
                            f"BW={_fb_result.get('pitch_BW',0):.1f} "
                            f"PM={_fb_result.get('pitch_PM',0):.1f}"
                        )
                        return _tag_sim_metrics(_fb_result, "fallback")
                    logger.warning(
                        "MATLAB subprocess produced no output; "
                        "skipping internal fallback (isolated)"
                    )
                    return _unreliable_sim_stub("unavailable")

            # ── Diagnostic: log stdout snippet to verify parsing input ──
            _stdout_len = len(stdout_text) if stdout_text else 0
            if _stdout_len > 0:
                _diag_lines = [l for l in stdout_text.splitlines() if 'rl_w1' in l.lower() or 'PM' in l or 'BW' in l or '命中率' in l or 'HIT' in l or 'MISS' in l]
                logger.debug(
                    f"  [RL DIAG] stdout_len={_stdout_len} "
                    f"metric_lines(first3)={_diag_lines[:3]}"
                )
            _parsed = self._parse_sim_output(stdout_text)
            # If all key metrics are at defaults, dump raw stdout for debugging
            if (abs(_parsed.get("pitch_PM", 30.0) - 30.0) < 0.01
                    and abs(_parsed.get("pitch_BW", 15.0) - 15.0) < 0.01
                    and _parsed.get("hit_rate", 0.0) == 0.0
                    and _stdout_len > 0):
                logger.warning(
                    f"  [RL DIAG] Parse produced ALL defaults from {_stdout_len} chars. "
                    f"Raw stdout (first 800 chars):\n{stdout_text[:800]}"
                )
            return _tag_sim_metrics(_parsed, "matlab")
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    async def _exec_matlab_subprocess(self, script_path: str, call_name: Optional[str] = None) -> str:
        """
        Invoke Octave/MATLAB on *script_path* using the simulator executor's
        configured engine and capture stdout.  Returns empty string on failure.

        *call_name*: if the script is a function file, pass the function name
        so the subprocess calls it explicitly instead of using run().
        """
        executor = getattr(self.simulator, "executor", None)
        if executor is None:
            logger.warning("RL optimizer: simulator has no executor; using internal fallback")
            return ""

        engine    = getattr(executor, "engine",       "octave")
        oct_path  = getattr(executor, "octave_path",  "octave")
        mat_path  = getattr(executor, "matlab_path",  "matlab")

        # ── One-time engine availability diagnostic ──────────────────────
        if not getattr(self, "_engine_diag_done", False):
            self._engine_diag_done = True
            _be = getattr(executor, "matlab_engine_backend", None)
            _be_status = "started" if (_be and _be.started) else ("exists-not-started" if _be else "None")
            import shutil
            _mat_on_path = shutil.which(mat_path)
            _oct_on_path = shutil.which(oct_path)
            logger.info(
                f"  [RL] Engine diagnostic: engine={engine}  "
                f"matlab_engine_backend={_be_status}  "
                f"matlab_path={mat_path!r}→{_mat_on_path}  "
                f"octave_path={oct_path!r}→{_oct_on_path}"
            )

        script_dir  = os.path.dirname(os.path.abspath(script_path)).replace("\\", "/")
        script_abs  = os.path.abspath(script_path).replace("\\", "/")

        # ── Branch 0: in-process MATLAB Engine API for Python ──────────
        # Reuses the executor's warm matlab.engine session so we don't pay
        # the (5–15 s on Windows) MATLAB launch cost on every RL episode.
        # When the backend can't start we silently fall through to the
        # subprocess matlab branch below.
        #
        # SKIP on Python 3.13+: MATLAB Engine API (R2025a) only supports
        # Python 3.9-3.12.  On 3.13 the C extension returns garbage stdout
        # or crashes silently.  Go directly to subprocess which works fine.
        import sys as _sys
        _skip_engine = _sys.version_info >= (3, 13)
        if _skip_engine and engine == "matlab_engine":
            if not getattr(self, "_engine_skip_warned", False):
                self._engine_skip_warned = True
                logger.info(
                    "  [RL] Skipping matlab_engine (Python %d.%d unsupported); "
                    "using subprocess 'matlab -batch' instead.",
                    _sys.version_info.major, _sys.version_info.minor,
                )
            engine = "matlab"

        if engine == "matlab_engine":
            backend = getattr(executor, "matlab_engine_backend", None)
            if backend is not None and backend.started:
                # run_script() is synchronous — wrap in executor to avoid
                # blocking the asyncio event loop during MATLAB execution.
                _loop = asyncio.get_event_loop()
                _tsc  = float(DEFAULT_SUBPROCESS_TIMEOUT_SEC)
                import time as _time_mod
                _t0_eng = _time_mod.perf_counter()
                stdout, stderr, ok = await _loop.run_in_executor(
                    None,
                    lambda: backend.run_script(
                        script_abs,
                        call_name=call_name,
                        timeout_sec=_tsc,
                    ),
                )
                _dt_eng = _time_mod.perf_counter() - _t0_eng
                if ok:
                    if stdout.strip():
                        logger.info(
                            f"  [RL] matlab.engine OK in {_dt_eng:.1f}s  "
                            f"stdout={len(stdout)}chars  call={call_name}"
                        )
                        return stdout
                    # Empty stdout: MATLAB Engine API sometimes fails to
                    # capture fprintf() inside function calls on Windows.
                    # Fall through to subprocess MATLAB which reliably pipes
                    # fprintf to stdout.
                    logger.warning(
                        f"matlab.engine produced no stdout on "
                        f"'{os.path.basename(script_path)}'; "
                        f"falling through to MATLAB subprocess."
                    )
                    engine = "matlab"
                elif not backend.started:
                    # Engine crashed during run_script
                    logger.warning(
                        f"matlab.engine crashed on '{os.path.basename(script_path)}'; "
                        f"falling through to subprocess MATLAB for this episode."
                    )
                    engine = "matlab"
                else:
                    # Normal MATLAB script-level error (not a crash)
                    logger.warning(
                        f"matlab.engine script error on '{os.path.basename(script_path)}':"
                        f"\n{(stderr or '')[:2000]}"
                    )
                    return ""
            if engine == "matlab_engine":  # backend unavailable from the start
                logger.info(
                    "matlab.engine backend not started; falling back to "
                    "MATLAB subprocess for this RL call."
                )
                engine = "matlab"

        if engine == "octave":
            # source() executes the file in the current scope without
            # requiring the basename to be a valid identifier (whereas run()
            # does internal name parsing that can choke on unusual names).
            # build_octave_eval_string injects OCTAVE_BATCH_PREFIX, which
            # silences warnings, disables the pager, hides figures (no
            # rendering overhead), and pre-loads the control package.
            eval_str = build_octave_eval_string(script_abs, call_name)
            cmd = [oct_path, *OCTAVE_BATCH_FLAGS, "--eval", eval_str]
        elif engine == "matlab":
            # MATLAB -batch auto-exits after script completes (no 'exit'
            # needed).  -nodisplay is Linux-only; on Windows use
            # set(0,'DefaultFigureVisible','off') instead.
            if call_name:
                eval_str = (
                    f"set(0,'DefaultFigureVisible','off'); "
                    f"cd('{script_dir}'); addpath('{script_dir}'); "
                    f"{call_name}"
                )
            else:
                eval_str = (
                    f"set(0,'DefaultFigureVisible','off'); "
                    f"run('{script_abs}')"
                )
            cmd = [mat_path, "-nosplash", "-nodesktop", "-batch", eval_str]
        else:
            logger.warning(f"Unsupported engine '{engine}' for RL optimizer")
            return ""

        # Log the exact command at debug level for reproducibility
        logger.debug(f"RL subprocess: {' '.join(repr(c) for c in cmd)}")

        # Use Popen + async poll instead of blocking subprocess.run() so that
        # KeyboardInterrupt / CancelledError can be delivered between polls.
        # Python 3.13 on Windows has a ProactorEventLoop bug with async
        # subprocess pipes, so we stay with synchronous Popen.
        _poll_interval = 2.0  # seconds between interrupt-check polls
        from multi_agent.simulation.sim_timeout import get_matlab_timeout_sec

        _timeout = get_matlab_timeout_sec(self.nmc_per_eval)

        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            import time as _time_sub
            _t0_sub = _time_sub.monotonic()
            while proc.poll() is None:
                elapsed = _time_sub.monotonic() - _t0_sub
                if elapsed > _timeout:
                    proc.kill()
                    proc.wait(timeout=5)
                    raise subprocess.TimeoutExpired(cmd, _timeout)
                await asyncio.sleep(_poll_interval)  # yields to event loop → Ctrl+C works
            _stdout_bytes = proc.stdout.read() if proc.stdout else b""
            _stderr_bytes = proc.stderr.read() if proc.stderr else b""

            class _SubResult:
                def __init__(self):
                    self.returncode = proc.returncode
                    self.stdout = _stdout_bytes
                    self.stderr = _stderr_bytes

            _result = _SubResult()
            _enc = locale.getpreferredencoding(False) or "utf-8"
            _stdout = _result.stdout.decode(_enc, errors="replace")
            _stderr = _result.stderr.decode(_enc, errors="replace")
            stderr_excerpt = _stderr.strip()
            self._last_exec_stderr = _stderr   # always store for retry logic
            if _result.returncode != 0:
                logger.warning(
                    f"{engine} returned code {_result.returncode} on "
                    f"'{os.path.basename(script_path)}'.\n"
                    f"  cmd: {' '.join(cmd)}\n"
                    f"  stderr (up to 2000 chars):\n{stderr_excerpt[:2000]}"
                )
                # If MATLAB crashed AFTER printing some per-run lines
                # (e.g. error in plot_results), return the partial stdout
                # so the parser can still extract metrics from completed runs.
                if _stdout.strip() and re.search(r"\[\s*\d+\]\s*(?:HIT|MISS)", _stdout):
                    logger.info(
                        f"  [RL] {engine} failed but stdout has {len(_stdout)} chars "
                        f"with per-run data; returning partial output for parsing."
                    )
                    return _stdout
                return ""
            if not _stdout.strip():
                logger.warning(
                    f"{engine} produced no stdout (returncode=0) on "
                    f"'{os.path.basename(script_path)}'. "
                    f"This usually means the script ran but printed nothing "
                    f"(missing fprintf? early early-exit?).\n"
                    f"  stderr (up to 2000 chars):\n{stderr_excerpt[:2000]}"
                )
                return ""
            logger.debug(f"  [RL] {engine} subprocess OK  stdout={len(_stdout)}chars")
            return _stdout
        except FileNotFoundError:
            logger.warning(f"{engine} executable not found; will use internal fallback")
            return ""
        except subprocess.TimeoutExpired:
            logger.warning(
                f"{engine} subprocess timed out after "
                f"{_timeout:.0f}s "
                f"(set MATLAB_TIMEOUT_SEC env var to override)"
            )
            return ""
        except (asyncio.CancelledError, KeyboardInterrupt):
            # Kill the MATLAB process on interrupt to avoid orphaned processes
            try:
                proc.kill()
                proc.wait(timeout=5)
            except Exception:
                pass
            raise
        except Exception as exc:
            import traceback as _tb
            # Ensure subprocess is cleaned up on any error
            try:
                proc.kill()
                proc.wait(timeout=5)
            except Exception:
                pass
            logger.warning(
                f"{engine} subprocess failed: {type(exc).__name__}: {exc}\n"
                f"  traceback: {_tb.format_exc()[-500:]}"
            )
            return ""

    async def _run_internal_simulation(
        self,
        auto_params: Dict[str, float],
        mission_conditions: Optional[Dict[str, List[int]]],
    ) -> Dict[str, float]:
        """
        Lightweight Python fallback used when MATLAB/Octave is unavailable.

        Builds analytical metrics that vary smoothly with 9 tunable
        parameters (8 autopilot + N_pn guidance) so the RL signal is
        non-degenerate even without a full simulator.
        """
        # 1. Pull all 9 tunable params from ALL_TUNABLE_PARAM_SPECS
        w1    = float(auto_params.get("w1",    ALL_TUNABLE_PARAM_SPECS["w1"]["nominal"]))
        zeta1 = float(auto_params.get("zeta1", ALL_TUNABLE_PARAM_SPECS["zeta1"]["nominal"]))
        tao1  = float(auto_params.get("tao1",  ALL_TUNABLE_PARAM_SPECS["tao1"]["nominal"]))
        w2    = float(auto_params.get("w2",    ALL_TUNABLE_PARAM_SPECS["w2"]["nominal"]))
        zeta2 = float(auto_params.get("zeta2", ALL_TUNABLE_PARAM_SPECS["zeta2"]["nominal"]))
        tao2  = float(auto_params.get("tao2",  ALL_TUNABLE_PARAM_SPECS["tao2"]["nominal"]))
        w3    = float(auto_params.get("w3",    ALL_TUNABLE_PARAM_SPECS["w3"]["nominal"]))
        zeta3 = float(auto_params.get("zeta3", ALL_TUNABLE_PARAM_SPECS["zeta3"]["nominal"]))
        N_pn  = float(auto_params.get("N_pn",  ALL_TUNABLE_PARAM_SPECS["N_pn"]["nominal"]))

        # 2. Try the high-level Python simulator first (varies w/ nav & damping)
        nav_coeff = N_pn
        damping   = zeta1
        try:
            study = await self.simulator.parameter_study(
                param_grid={"navigation_coefficient": [nav_coeff],
                            "damping_ratio":          [damping]},
                duration=100.0, dt=0.01,
            )
        except Exception as exc:
            logger.debug(f"parameter_study failed: {exc}")
            study = []

        # 3. Use sep_mid_threshold as analytical baseline for fallback
        rw   = self._rw
        _mm  = rw["sep_mid_threshold"]   # e.g. 5.0 m

        if study:
            import math as _im
            m = study[0].get("metrics", {}) or {}
            base_miss = float(m.get("SEP", m.get("miss_distance", _mm)))
            if _im.isnan(base_miss) or _im.isinf(base_miss):
                logger.debug("[InternalSim] parameter_study returned NaN SEP; using mid-range fallback")
                base_miss = _mm
            base_hit  = 100.0 if m.get("success", True) else 50.0
        else:
            base_miss, base_hit = _mm, 80.0

        # Distance-from-nominal penalties (drives RL toward ALL_TUNABLE_PARAM_SPECS
        # nominal values when no real simulator is available)
        def _dev(name: str, val: float) -> float:
            spec = ALL_TUNABLE_PARAM_SPECS[name]
            half = (spec["max"] - spec["min"]) / 2.0
            return abs(val - spec["nominal"]) / max(half, 1e-6)

        dev_total = (_dev("w1", w1) + _dev("zeta1", zeta1) + _dev("tao1", tao1)
                   + _dev("w2", w2) + _dev("zeta2", zeta2) + _dev("tao2", tao2)
                   + _dev("w3", w3) + _dev("zeta3", zeta3)
                   + _dev("N_pn", N_pn)) / 9.0

        sep      = base_miss * (1.0 + 1.5 * dev_total)
        hit_rate = max(0.0, base_hit * (1.0 - 0.5 * dev_total))

        # Stability margins: PM & BW are PRIMARILY determined by pitch
        # channel (w1, zeta1, tao1).  Yaw/roll have negligible effect.
        rw        = self._rw
        _pm_mid   = (rw["pm_min"] + rw["pm_max"]) / 2.0
        _pm_range = abs(rw["pm_max"] - rw["pm_min"]) / 2.0
        # PM ↑ with zeta1, ↓ with tao1; w1 has weak inverse effect
        pitch_PM  = max(5.0, _pm_mid
                        + _pm_range * (zeta1 - 0.7) * 2.5
                        - 25.0 * (tao1 - 0.15)
                        - 0.15 * (w1 - 40.0))

        # BW is dominated by w1 (pitch natural frequency)
        pitch_BW = max(2.0, 0.8 * w1 + 0.05 * w2)
        # Rough overload proxy so fallback is not treated as zero-g
        peak_ny_est = max(0.0, 0.12 * w1 + 0.05 * w2)

        return _tag_sim_metrics({
            "hit_rate":    float(hit_rate),
            "SEP":         float(sep),
            "miss_distance": float(sep),
            "peak_ny":     float(peak_ny_est),
            "pitch_PM":    float(pitch_PM),
            "pitch_BW":    float(pitch_BW),
            "pitch_GM":    8.0,
        }, "fallback")

    def _parse_sim_output(self, raw: Any) -> Dict[str, float]:
        """Parse rich MATLAB stdout into the metric dict.

        Delegates to the module-level :func:`parse_sim_stdout` so both
        the RL optimizer and external callers (e.g. cli_agent) share
        identical parsing logic.
        """
        return parse_sim_stdout(raw)

    # ── PPO update (contextual bandit: gamma=0, no GAE) ─────────────────────

    def _flush_ppo_batch_if_ready(
        self,
        batch: List[EpisodeResult],
        old_lps: List[float],
    ) -> None:
        """Run PPO update when batch is full; clear batch buffers."""
        if len(batch) >= self.episodes_per_update:
            loss = self._ppo_update(batch, old_lps)
            logger.info(
                f"  [Update] batch={len(batch)} actor_loss={loss.get('actor_loss', 0):.4f}  "
                f"critic_loss={loss.get('critic_loss', 0):.4f}  "
                f"entropy={loss.get('entropy', 0):.4f}"
            )
            batch.clear()
            old_lps.clear()

    def _ppo_update(
        self, episodes: List[EpisodeResult], old_log_probs: List[float]
    ) -> Dict[str, float]:
        if not episodes or self._actor is None or self._critic is None:
            return {}
        n = len(episodes)
        rewards = np.array([ep.reward for ep in episodes], dtype=np.float32)
        values  = np.array([ep.value for ep in episodes], dtype=np.float32)
        old_lp  = np.array(old_log_probs, dtype=np.float32)

        # Contextual bandit: each episode is independent → G_t = R_t (gamma forced 0).
        returns = rewards.copy()
        adv = returns - values
        adv_std = float(adv.std())
        if adv_std > 1e-6:
            adv = (adv - adv.mean()) / (adv_std + 1e-8)

        log_std = self._actor.log_std
        std = np.exp(np.clip(log_std, -4.0, 2.0))
        entropy = float(np.sum(log_std + 0.5 * (1.0 + math.log(2.0 * math.pi))))

        dW1_a = np.zeros_like(self._actor.W1)
        db1_a = np.zeros_like(self._actor.b1)
        dW2_a = np.zeros_like(self._actor.W2)
        db2_a = np.zeros_like(self._actor.b2)

        dW1_c = np.zeros_like(self._critic.W1)
        db1_c = np.zeros_like(self._critic.b1)
        dW2_c = np.zeros_like(self._critic.W2)
        db2_c = np.zeros_like(self._critic.b2)

        a_loss_total, c_loss_total = 0.0, 0.0

        for i, ep in enumerate(episodes):
            s = ep.state
            a = ep.action

            mean_out, h_a = self._actor.forward(s)
            noise = (a - mean_out) / (std + 1e-8)
            new_lp = float(-0.5 * np.sum(noise ** 2 + 2 * log_std + math.log(2 * math.pi)))

            ratio = ppo_probability_ratio(new_lp, float(old_lp[i]))
            clip_r = float(np.clip(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio))
            obj1  = ratio * adv[i]
            obj2  = clip_r * adv[i]
            a_loss = -min(obj1, obj2) - self.entropy_coef * entropy
            a_loss_total += a_loss

            d_mean = (-(a - mean_out) / (std ** 2 + 1e-8) * clip_r * adv[i])
            dW2_a += np.outer(h_a, d_mean)
            db2_a += d_mean
            dh_a  = d_mean @ self._actor.W2.T
            dpre  = dh_a * (1.0 - h_a ** 2)
            dW1_a += np.outer(s, dpre)
            db1_a += dpre

            v_pred, h_c = self._critic.forward(s)
            v_err = float(v_pred[0]) - returns[i]
            c_loss_total += v_err ** 2
            dv   = np.array([2.0 * v_err], dtype=np.float32)
            dW2_c += np.outer(h_c, dv)
            db2_c += dv
            dhc  = dv @ self._critic.W2.T
            dprc = dhc * (1.0 - h_c ** 2)
            dW1_c += np.outer(s, dprc)
            db1_c += dprc

        inv_n = 1.0 / n
        self._actor.adam_step([dW1_a * inv_n, db1_a * inv_n, dW2_a * inv_n, db2_a * inv_n])
        self._critic.adam_step([dW1_c * inv_n, db1_c * inv_n, dW2_c * inv_n, db2_c * inv_n])

        # Entropy bonus — one update per batch (not per sample).
        d_log_std = self.entropy_coef * np.ones(self._action_dim, dtype=np.float32)
        self._actor.log_std = np.clip(
            self._actor.log_std + self.lr_actor * d_log_std,
            -4.0, 2.0,
        )

        logger.debug(
            f"[PPO/bandit] batch={n} gamma=0 adv_mean={float(adv.mean()):.4f} "
            f"entropy={entropy:.4f} ent_coef={self.entropy_coef}"
        )
        return {
            "actor_loss":  float(a_loss_total / n),
            "critic_loss": float(c_loss_total / n),
            "entropy":     entropy,
        }

    # ── main optimization loop ────────────────────────────────────────────────

    async def optimize(
        self,
        script_path: Optional[str] = None,
        mission_conditions: Optional[Dict[str, List[int]]] = None,
        max_episodes: Optional[int] = None,
        nmc: Optional[int] = None,
        task_context: Optional[Dict[str, Any]] = None,
        miss_threshold: Optional[float] = None,
        task_prompt: Optional[str] = None,
        reflection_agent: Optional[Any] = None,
        reflect_every: Optional[int] = None,
        initial_auto_params: Optional[Dict[str, float]] = None,
        metric_requirements: Optional[Dict[str, Any]] = None,
        optimization_history: Optional[List[Dict[str, Any]]] = None,
        resume_checkpoint: Optional[str] = None,
        warm_start_source: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run the RL optimization loop. Early-exit is **driven solely by the
        reflection agent**: on every new best, the agent is consulted with
        the user's task prompt and the current best metrics; when it reports
        ``needs_optimization=False`` the loop exits with status
        ``"task_requirements_met"``.

        ``miss_threshold`` is no longer used as a decision criterion; if
        supplied it is forwarded to the reflection agent as additional
        numeric context only (the LLM can also read this directly from the
        prompt, so it is purely informational).

        Best parameters are persisted to ParameterExperience before returning,
        regardless of which exit path was taken.

        Returns
        -------
        dict with keys: status, best_params, best_metrics, best_reward,
        baseline_reward, total_episodes, history, reflection (if used).
        ``status`` is ``"task_requirements_met"`` on reflection-driven exit,
        otherwise ``"success"``.
        """
        max_ep = max_episodes or self.max_episodes
        n_mc   = nmc or self.nmc_per_eval

        self._configure_fre(mission_conditions, task_prompt, metric_requirements)
        _task_reqs = resolve_requirements(
            metric_requirements, task_prompt,
            reflection_agent or self.reflection_agent,
        )

        active_reflector = reflection_agent or self.reflection_agent
        cadence = max(1, int(reflect_every) if reflect_every is not None
                      else self.reflect_every)
        reflection_enabled = bool(active_reflector and task_prompt)

        # ── Defensive cleanup of stale RL temp files ──────────────────────
        # If a previous RL run crashed before its `finally: os.remove(...)`
        # cleanup, a temp file may linger.  Sweep the script_path's directory
        # so stale files cannot be picked up as the "latest" user script.
        if script_path:
            self._cleanup_stale_temp_files(os.path.dirname(os.path.abspath(script_path)))
        # ──────────────────────────────────────────────────────────────────

        if script_path and os.path.exists(script_path):
            self.extract_params_from_script(script_path)

        _ws_src = (warm_start_source or "unspecified").strip().lower()
        _ws_labels = {
            "pe": "PE",
            "expert": "Expert",
            "hermes": "Hermes/Layer2",
            "script": "MATLAB script",
            "default": "default nominal",
            "checkpoint": "PPO checkpoint",
        }
        _ws_label = _ws_labels.get(_ws_src, warm_start_source or "unspecified")

        # Override _base_auto with warm-start params when provided.
        if initial_auto_params:
            for k, v in initial_auto_params.items():
                if k in ALL_TUNABLE_PARAM_SPECS:
                    spec = ALL_TUNABLE_PARAM_SPECS[k]
                    self._base_auto[k] = float(
                        np.clip(v, spec["min"], spec["max"])
                    )
            logger.info(
                f"[RL] Baseline uses warm-start params "
                f"(source={_ws_label}, {len(initial_auto_params)} values)"
            )

        self._init_networks()

        resume_meta: Optional[Dict[str, Any]] = None
        start_ep = 0
        baseline_reward = 0.0
        baseline_metrics: Dict[str, float] = {}

        ckpt_path = PPOCheckpointManager.resolve_checkpoint_path(resume_checkpoint)
        if ckpt_path:
            self._checkpoint_mgr = PPOCheckpointManager(
                checkpoint_dir=self._checkpoint_dir,
                save_every_episodes=self._checkpoint_save_every,
                keep_last_n=self._checkpoint_keep_last_n,
                enabled=self._checkpoint_enabled,
            )
            resume_meta = self._checkpoint_mgr.load(self, ckpt_path)
            start_ep = int(resume_meta.get("episode_completed", 0))
            baseline_reward = float(self._best_reward)
            baseline_metrics = copy.deepcopy(self._best_metrics)
            self._init_training_loggers(
                resume_run_id=str(resume_meta.get("run_id", "")),
                append_only=True,
            )
            logger.info(
                f"[RL] Resuming from checkpoint ep {start_ep} → "
                f"will run episodes {start_ep + 1}..{max_ep}"
            )
        else:
            self._init_training_loggers()

        logger.info(
            f"[RL] Contextual-bandit PPO | batch={self.episodes_per_update} | "
            f"nmc={n_mc} | gamma=0 | entropy_coef={self.entropy_coef}"
        )

        if not resume_meta:
            if initial_auto_params:
                logger.info(f"Evaluating baseline (warm-start source={_ws_label})…")
            else:
                logger.info("Evaluating baseline (script/default params)…")
            baseline_metrics = await self._run_simulation_with_params(
                self._base_auto, mission_conditions, script_path, n_mc
            )
            _bl_peak = get_peak_ny(baseline_metrics)
            _reset_peak = float(self._rw.get("baseline_reset_peak_g", 40.0))
            if task_prompt and _bl_peak > _reset_peak:
                try:
                    from multi_agent.integration.t4_low_risk import resolve_initial_auto_params
                    _conservative = resolve_initial_auto_params(
                        self._base_auto,
                        task_prompt=task_prompt,
                    )
                    _retry_metrics = await self._run_simulation_with_params(
                        _conservative, mission_conditions, script_path, n_mc
                    )
                    if get_peak_ny(_retry_metrics) < _bl_peak:
                        logger.info(
                            "[RL] Baseline peak mean %.1fg > %.1fg — reset to conservative params (peak→%.1fg)",
                            _bl_peak,
                            _reset_peak,
                            get_peak_ny(_retry_metrics),
                        )
                        self._base_auto = copy.deepcopy(_conservative)
                        baseline_metrics = _retry_metrics
                except Exception as _br_exc:
                    logger.debug("[RL] Baseline conservative reset skipped: %s", _br_exc)
            baseline_reward = self._compute_reward(baseline_metrics)
            logger.info(f"Baseline reward={baseline_reward:.3f}  metrics={baseline_metrics}")

            if initial_auto_params or baseline_reward >= self._baseline_reward_threshold:
                self._shrink_log_std_for_warmstart()

            self._best_constrained_params = {}
            self._best_constrained_metrics = {}
            self._best_constrained_fitness = -1.0
            self._best_borderline_params = {}
            self._best_borderline_metrics = {}
            self._last_constrained_improve_ep = 0
            self._last_peak_improve_ep = 0
            self._consecutive_diverge = 0
            self._update_constrained_best(self._base_auto, baseline_metrics, -1)
            self._update_borderline_best(self._base_auto, baseline_metrics)

            self._best_reward    = baseline_reward
            self._best_params    = copy.deepcopy(self._base_auto)
            self._best_metrics   = copy.deepcopy(baseline_metrics)
            import math as _bm
            _bl_miss = float(baseline_metrics.get("miss_distance", float("inf")))
            self._best_miss_ever   = _bl_miss if not _bm.isnan(_bl_miss) else float("inf")
            _bl_pkn  = get_peak_ny(baseline_metrics)
            self._best_peak_n_ever = _bl_pkn if _bl_pkn > 0 and not _bm.isnan(_bl_pkn) else float("inf")
            _bl_sep = float(baseline_metrics.get("SEP", baseline_metrics.get("miss_distance", 0.0)))
            if _bm.isnan(_bl_sep) or _bm.isinf(_bl_sep):
                logger.warning(
                    "[RL] Baseline SEP=NaN — external MATLAB and internal fallback both failed. "
                    "Using zeroed default metrics for RL state initialisation to avoid NaN propagation."
                )
                self._last_metrics = {k: 0.0 for k in self._last_metrics}
            else:
                self._last_metrics = baseline_metrics

            self._current_params = copy.deepcopy(self._base_auto)
        else:
            import math as _bm
            if resume_meta.get("max_episodes"):
                max_ep = max(max_ep, int(resume_meta["max_episodes"]))

        batch: List[EpisodeResult] = []
        old_lps: List[float] = []
        final_status = "success"
        episodes_run = 0
        last_reflection: Optional[Dict[str, Any]] = None
        _interrupted = False
        
        # ── Initialize steep gradient detector for T4 narrow feasible region ──
        steep_detector = SteepGradientDetector(window_size=5, threshold=2.0)
        if self._adaptive_explore_enabled:
            self._adaptive_ctrl = AdaptiveExplorationController(
                steepness_threshold=self._adaptive_steepness_threshold,
                log_std_min_scale=self._adaptive_log_std_min,
                log_std_max_scale=self._adaptive_log_std_max,
                ema_alpha=self._adaptive_ema_alpha,
                low_reward_threshold=self._diverge_reward_threshold,
            )
        else:
            self._adaptive_ctrl = None
        self._ep_prev_params = copy.deepcopy(self._current_params)
        self._ep_prev_reward = 0.0

        for ep in range(start_ep, max_ep):
          try:
            episodes_run = ep + 1
            # Episode exploration decay
            if self._actor is not None and max_ep > 0:
                # ── Adaptive exploration decay based on gradient steepness ──
                is_steep = steep_detector.is_steep_gradient()
                
                if is_steep:
                    # Steep gradient: reduce exploration (conservative)
                    _decay = max(0.05, 1.0 - ep / float(max_ep)) * 0.5
                    logger.info(
                        f"  [Ep {ep+1}] Steep gradient detected, reducing exploration "
                        f"(decay={_decay:.3f})"
                    )
                else:
                    # Normal gradient: standard exploration decay
                    _decay = max(0.15, 1.0 - ep / float(max_ep))

                _adapt_scale = 1.0
                if self._adaptive_ctrl is not None:
                    _adapt_scale = float(self._adaptive_ctrl.state.log_std_scale)
                    if steep_detector.is_steep_gradient():
                        self._adaptive_ctrl.ingest_neighborhood_steepness(
                            steep_detector.threshold * 1.5, blend=0.5,
                        )
                        _adapt_scale = min(
                            _adapt_scale,
                            float(self._adaptive_ctrl.state.log_std_scale),
                        )

                for i, k in enumerate(self._action_keys):
                    _base_ls = (
                        self._pitch_log_std if k in self._PITCH_KEYS else self._other_log_std
                    )
                    self._actor.log_std[i] = float(
                        np.clip(
                            _base_ls + math.log(max(_decay * _adapt_scale, 0.05)),
                            -4.0,
                            2.0,
                        )
                    )
            state      = self._build_state(self._current_params, self._base_phys, self._last_metrics)
            action, lp = self._actor.sample(state, explore=True)
            value      = self._critic.value(state)

            new_params = self._action_to_params(action)

            metrics = await self._run_simulation_with_params(
                new_params, mission_conditions, script_path, n_mc
            )

            # ── Staleness guard: identical metrics with different params ──────
            if ep > 0 and self._last_metrics:
                _key_metrics = ("pitch_PM", "pitch_BW", "SEP", "hit_rate", "peak_ny")
                _same = all(
                    abs(metrics.get(k, 0) - self._last_metrics.get(k, -1)) < 1e-6
                    for k in _key_metrics
                    if isinstance(metrics.get(k, 0), (int, float)) and isinstance(self._last_metrics.get(k, 0), (int, float))
                )
                if _same:
                    self._stale_count = getattr(self, "_stale_count", 0) + 1
                    if self._stale_count >= 3:
                        logger.warning(
                            f"  [Ep {ep+1}] ⚠️ STALE METRICS detected ({self._stale_count} consecutive identical results). "
                            f"Disabling matlab_engine for remaining episodes; forcing subprocess."
                        )
                        # Force the executor to use subprocess instead of engine
                        _exec = getattr(self.simulator, "executor", None)
                        if _exec and getattr(_exec, "engine", "") == "matlab_engine":
                            _exec.engine = "matlab"
                            logger.info("  [RL] Switched executor engine → 'matlab' (subprocess)")
                        self._stale_count = 0  # reset after switching
                else:
                    self._stale_count = 0

            reward = self._compute_reward(metrics)
            _anomalous = self._is_anomalous(metrics) or not is_reliable_sim_metrics(metrics)

            if reward <= self._diverge_reward_threshold:
                self._consecutive_diverge += 1
                if self._consecutive_diverge >= self._diverge_penalty_episodes:
                    self._apply_diverge_exploration_penalty()
                    self._consecutive_diverge = 0
                if self._adaptive_ctrl is not None:
                    self._adaptive_ctrl.on_low_reward()
            else:
                self._consecutive_diverge = 0

            if self._adaptive_ctrl is not None:
                _was_best = reward > self._best_reward
                self._adaptive_ctrl.update_episode(
                    self._ep_prev_params,
                    new_params,
                    self._ep_prev_reward,
                    reward,
                    new_best=_was_best,
                )
                self._ep_prev_params = copy.deepcopy(new_params)
                self._ep_prev_reward = float(reward)
            
            # ── Feasible Region Explorer: modify reward and track phase progress ──
            reward_with_fre_bonus = self._fre_reward_modifier.modify_reward(reward, metrics)
            self._fre.update_episode(metrics, reward)
            
            # Log FRE phase status periodically
            if (ep + 1) % 5 == 0:
                self._fre.log_phase_status()
            
            # Check if should advance to next phase
            if self._fre.should_advance_phase():
                self._fre.advance_phase()

            batch.append(EpisodeResult(
                params=copy.deepcopy(new_params), metrics=copy.deepcopy(metrics),
                reward=reward_with_fre_bonus, state=state, action=action, log_prob=lp, value=value,
            ))
            old_lps.append(lp)

            if _anomalous:
                _sep_raw = metrics.get("SEP", metrics.get("miss_distance", "?"))
                logger.warning(
                    f"  [Ep {ep+1}/{max_ep}] 数据异常 — SEP={_sep_raw} hit={metrics.get('hit_rate',0):.1f}% "
                    f"(仿真发散或解析失败); 跳过 best-update; reward={reward:.3f}"
                )
                self._log_episode_training(
                    ep=ep,
                    max_ep=max_ep,
                    new_params=new_params,
                    metrics=metrics,
                    reward=reward,
                    constrained=False,
                )
                self._maybe_save_checkpoint(
                    ep=ep, max_ep=max_ep, script_path=script_path, nmc=n_mc,
                )
                self._flush_ppo_batch_if_ready(batch, old_lps)
                continue

            self._update_constrained_best(new_params, metrics, ep)
            self._update_borderline_best(new_params, metrics)

            new_best = reward > self._best_reward
            if new_best:
                self._best_reward  = reward
                self._best_params  = copy.deepcopy(new_params)
                self._best_metrics = copy.deepcopy(metrics)
                if is_verbose():
                    logger.info(
                        f"  [Ep {ep+1}] New best reward={reward:.3f}  params={new_params}"
                    )
                # Store every reward-best to SHORT_TERM for intra-session tracking
                if self.parameter_experience:
                    try:
                        from ..memory.parameter_experience import MemoryType as _MT
                        _ep_fitness = self._compute_pe_fitness(metrics)
                        await self.parameter_experience.store(
                            task_context={"task": "rl_episode_best", "episode": ep + 1,
                                          **(task_context or {})},
                            parameters={k: float(v) for k, v in new_params.items()},
                            objectives={k: float(v) for k, v in metrics.items()
                                        if isinstance(v, (int, float))},
                            fitness=_ep_fitness,
                            metadata={"episode": ep + 1, "reward": float(reward)},
                            memory_type=_MT.SHORT_TERM,
                        )
                    except Exception as _exc:
                        logger.debug(f"  [RL] Episode best SHORT_TERM store failed: {_exc}")
                # ── Lightweight requirements check (no LLM) ───────────────────────
                if metric_requirements and self._requirements_met(self._best_metrics, metric_requirements):
                    logger.info(
                        f"  [RL] All metric requirements met at episode {ep+1}; early exit."
                    )
                    final_status = "requirements_met"
                    break
                if (
                    self._early_stop_enabled
                    and self._early_stop_use_rule_first
                    and _task_reqs
                    and (ep + 1) >= self._early_stop_min_episodes
                    and self._best_constrained_metrics
                    and constraints_satisfied(self._best_constrained_metrics, _task_reqs)
                ):
                    logger.info(
                        f"  [RL] Multi-criteria requirements met at episode {ep+1}; early exit."
                    )
                    final_status = "requirements_met"
                    break
            # Track best miss and peak_n independently of reward
            cur_miss   = float(metrics.get("miss_distance", float("inf")))
            cur_peak_n = get_peak_ny(metrics)
            if cur_miss < self._best_miss_ever:
                self._best_miss_ever = cur_miss
            if cur_peak_n > 0 and cur_peak_n < self._best_peak_n_ever:
                self._best_peak_n_ever = cur_peak_n
                self._last_peak_improve_ep = ep + 1

            self._last_metrics = metrics
            self._current_params = copy.deepcopy(new_params)
            self._history.append({
                "episode": ep + 1, "reward": reward,
                "metrics": metrics, "params": new_params,
            })
            
            # ── Update steep gradient detector ──
            steep_detector.update(reward, new_params)

            best_reward_miss = float(self._best_metrics.get("miss_distance", 99.0))
            _pny = get_peak_ny(metrics)
            _peak_n_str = f"{_pny:.2f}" if _pny > 0 else "N/A"
            _sep = float(metrics.get("SEP", metrics.get("miss_distance", 99)))
            _hit = float(metrics.get("hit_rate", 0))
            _ep_line_full = (
                f"  [Ep {ep+1}/{max_ep}] reward={reward:.3f}  "
                f"hit={_hit:.1f}%  "
                f"SEP={_sep:.2f}m  "
                f"best_reward_SEP={best_reward_miss:.2f}m  "
                f"best_SEP_ever={self._best_miss_ever:.2f}m  "
                f"PeakNy_avg={_peak_n_str}g  "
                f"PM={metrics.get('pitch_PM',0):.1f}°(best={self._best_metrics.get('pitch_PM',0):.1f}°)  "
                f"BW={metrics.get('pitch_BW',0):.1f}r/s(best={self._best_metrics.get('pitch_BW',0):.1f}r/s)"
            )
            _ep_line_compact = (
                f"  [Ep {ep+1}/{max_ep}] reward={reward:.2f} "
                f"hit={_hit:.0f}% SEP={_sep:.2f}m PeakNy_avg={_peak_n_str}g"
            )
            if should_log_rl_episode(ep, max_ep, new_best=new_best):
                if is_verbose():
                    logger.info(_ep_line_full)
                else:
                    logger.info(_ep_line_compact + (" ★" if new_best else ""))
            else:
                logger.debug(_ep_line_full)

            _constrained_ok = bool(
                self._best_constrained_metrics
                and constraints_satisfied(self._best_constrained_metrics, _task_reqs)
            ) if _task_reqs else False
            self._log_episode_training(
                ep=ep,
                max_ep=max_ep,
                new_params=new_params,
                metrics=metrics,
                reward=reward,
                constrained=_constrained_ok,
            )
            self._maybe_save_checkpoint(
                ep=ep, max_ep=max_ep, script_path=script_path, nmc=n_mc,
            )

            # ── Patience early stop (multi-criteria plateau) ─────────────────
            _peak_limit = float(self._rw.get("peak_ny_mean_max", self._rw.get("peak_ny_max", PEAK_NY_MAX)))
            _near_miss = (
                self._best_peak_n_ever < float("inf")
                and _peak_limit < self._best_peak_n_ever <= _peak_limit + 8.0
            )
            _patience = (
                self._early_stop_peak_only_patience
                if _near_miss
                else self._early_stop_patience
            )
            _last_improve = (
                self._last_peak_improve_ep
                if _near_miss and not self._best_constrained_metrics
                else self._last_constrained_improve_ep
            )
            if (
                self._early_stop_enabled
                and (ep + 1) >= self._early_stop_min_episodes
                and (
                    self._best_constrained_metrics
                    or _near_miss
                )
                and (ep + 1 - _last_improve) >= _patience
            ):
                logger.info(
                    f"  [RL] Patience early stop at episode {ep+1} "
                    f"(no {'peak' if _near_miss else 'constrained'} improvement for {_patience} ep)"
                )
                final_status = "success"
                break

            # ── Reflection-based task-requirement check ─────────────────────
            _reflect_params, _reflect_metrics, _sel = self._select_return_best()
            if (reflection_enabled
                    and ((ep + 1) % cadence == 0)
                    and (ep + 1) >= self._early_stop_min_episodes):
                try:
                    reflection_input = {
                        "parameters":          _reflect_params,
                        "metrics":             _reflect_metrics,
                        "best_reward_metrics": self._best_metrics,
                        "best_constrained_metrics": self._best_constrained_metrics,
                        "best_selection":      _sel,
                        "best_miss_ever":      self._best_miss_ever,
                        "best_peak_n_ever":    self._best_peak_n_ever,
                        "episode":             ep + 1,
                        "new_best":            new_best,
                        "best_reward":         float(self._best_reward),
                    }
                    if miss_threshold is not None:
                        reflection_input["miss_threshold_hint"] = float(miss_threshold)
                    reflection = await active_reflector.reflect(
                        task_prompt, reflection_input,
                        optimization_history=optimization_history,
                    )
                    last_reflection = reflection
                    needs_more = bool(reflection.get("needs_optimization", True))
                    suggestion = reflection.get("suggestion", "")
                    logger.info(
                        f"  [RL Reflection @ Ep {ep+1}] needs_optimization="
                        f"{needs_more} | {suggestion[:120]}"
                    )
                    if not needs_more:
                        logger.info(
                            f"  [RL] Task requirements met per reflection at "
                            f"episode {ep+1}; early exit."
                        )
                        final_status = "task_requirements_met"
                        break
                    _MODIFY_SIGNALS = [
                        "制导律", "算法结构", "modify_law", "修改制导律", "新的制导",
                        "重新设计", "滑模", "预测制导", "guidance law", "algorithm",
                    ]
                    if any(sig in suggestion.lower() for sig in _MODIFY_SIGNALS):
                        logger.info(
                            f"  [RL] Reflection suggests design-path change at "
                            f"episode {ep+1}; breaking to let Hermes re-plan. "
                            f"Hint: {suggestion[:200]}"
                        )
                        final_status = "needs_design_path_change"
                        break
                except Exception as ref_exc:
                    logger.warning(f"  [RL] Reflection call failed: {ref_exc}")
            # ──────────────────────────────────────────────────────────────

            self._flush_ppo_batch_if_ready(batch, old_lps)

            if _interrupted:
                break
          except (KeyboardInterrupt, asyncio.CancelledError):
            _interrupted = True
            logger.warning(
                f"  [RL] ⚠️ Interrupted at episode {ep+1}/{max_ep}. "
                f"Saving best results (reward={self._best_reward:.3f})..."
            )
            print(
                f"\n  [RL] ⚠️ 优化被中断 (episode {ep+1}/{max_ep}). "
                f"保存当前最优结果..."
            )
            final_status = "interrupted"
            self._maybe_save_checkpoint(
                ep=ep, max_ep=max_ep, script_path=script_path, nmc=n_mc, force=True,
            )
            break

        if batch:
            self._ppo_update(batch, old_lps)

        self._maybe_save_checkpoint(
            ep=max(episodes_run - 1, 0),
            max_ep=max_ep,
            script_path=script_path,
            nmc=n_mc,
            force=True,
        )
        if self._episode_logger is not None:
            self._episode_logger.close()

        ret_params, ret_metrics, selection = self._select_return_best()
        self._best_params = ret_params
        self._best_metrics = ret_metrics

        if self.parameter_experience and self._best_params:
            await self._store_best_params(task_context or {}, final_status=final_status)

        # Calculate total MC calls: baseline + episodes_run * nmc + any extra calls
        total_mc_calls = n_mc  # baseline evaluation
        total_mc_calls += episodes_run * n_mc  # each episode
        if hasattr(self, "_external_mc_calls"):
            total_mc_calls += int(self._external_mc_calls)  # external MC data evaluations

        return {
            "status":          final_status,
            "best_params":     ret_params,
            "best_metrics":    ret_metrics,
            "best_reward":     self._best_reward,
            "best_selection":  selection,
            "best_constrained_metrics": copy.deepcopy(self._best_constrained_metrics),
            "baseline_reward": baseline_reward,
            "total_episodes":  episodes_run,
            "total_mc_calls":  total_mc_calls,
            "history":         self._history[-10:],
            "reflection":      last_reflection,
            "amro_stats":      None,
            "jsonl_path":      self._episode_logger.path if self._episode_logger else "",
            "checkpoint_dir":  self._checkpoint_dir,
            "resumed_from":    resume_meta.get("_json_path", "") if resume_meta else "",
        }

    def _compute_pe_fitness(self, metrics: Dict[str, float]) -> float:
        """
        五指标综合适应度，用于 ParameterExperience 存储与检索排序。
        权重: hit_rate(↑)×0.35 + miss_distance(↓)×0.25 + pitch_PM(↑)×0.20
              + pitch_BW(目标区间)×0.10 + peak_n(↓)×0.10
        返回值在 [0, 1] 之间，越大越好。
        BW score: 在 [bw_min, bw_max] 内得 1.0，线性衰减到边界外 ±50% 处得 0。
        peak_n 无有效值（≤0）时按 0.5 中性处理。
        """
        import math as _m
        hit    = float(metrics.get("hit_rate",      0.0))
        miss   = get_sep(metrics, 100.0)
        if math.isnan(miss) or math.isinf(miss):
            miss = 100.0
        pm     = float(metrics.get("pitch_PM",       0.0))
        bw     = float(metrics.get("pitch_BW",       0.0))
        peak_n = get_peak_ny(metrics)
        pn_ref = max(1.0, float(self._peak_n_max))

        # Sanitize NaN/inf → neutral/worst-case values
        if _m.isnan(hit)    or _m.isinf(hit):    hit    = 0.0
        if _m.isnan(miss)   or _m.isinf(miss):   miss   = 100.0
        if _m.isnan(pm)     or _m.isinf(pm):     pm     = 0.0
        if _m.isnan(bw)     or _m.isinf(bw):     bw     = 0.0
        if _m.isnan(peak_n) or _m.isinf(peak_n): peak_n = 0.0

        rw = self._rw
        hit_min = HIT_RATE_MIN
        sep_max = SEP_MAX

        hit_score = max(0.0, min(1.0, (hit - hit_min) / max(100.0 - hit_min, 1.0)))
        miss_score = max(0.0, min(1.0, 1.0 - miss / max(sep_max, 1.0)))
        if peak_n <= 0.0:
            peak_n_score = 0.5
        elif peak_n <= pn_ref:
            peak_n_score = 1.0
        else:
            peak_n_score = max(0.0, 1.0 - (peak_n - pn_ref) / pn_ref)

        # PM score: 1.0 inside [pm_min, pm_max], linear decay outside
        pm_min   = rw.get("pm_min", PM_MIN)
        pm_max   = rw.get("pm_max", PM_MAX)
        pm_range = max(pm_max - pm_min, 1.0)
        if pm_min <= pm <= pm_max:
            pm_score = 1.0
        elif pm < pm_min:
            pm_score = max(0.0, 1.0 - (pm_min - pm) / (0.5 * pm_range))
        else:
            pm_score = max(0.0, 1.0 - (pm - pm_max) / (0.5 * pm_range))

        # BW score: 1.0 inside [bw_min, bw_max], linear decay outside
        bw_min   = rw.get("bw_min", BW_MIN)
        bw_max   = rw.get("bw_max", BW_MAX)
        bw_range = max(bw_max - bw_min, 1.0)
        if bw_min <= bw <= bw_max:
            bw_score = 1.0
        elif bw < bw_min:
            bw_score = max(0.0, 1.0 - (bw_min - bw) / (0.5 * bw_range))
        else:
            bw_score = max(0.0, 1.0 - (bw - bw_max) / (0.5 * bw_range))

        return max(0.0, min(1.0,
            0.35 * hit_score + 0.25 * miss_score + 0.20 * pm_score
            + 0.10 * bw_score + 0.10 * peak_n_score
        ))

    async def _store_best_params(self, task_context: Dict[str, Any], final_status: str = "success") -> None:
        """Persist best params to ParameterExperience.

        Memory layer strategy:
        - ``task_requirements_met`` → LONG_TERM (reflection-validated, cross-session)
        - ``success``               → SHORT_TERM (completed, pending auto_promote)
        """
        from ..memory.parameter_experience import MemoryType

        combined = {f"dp_{k}": float(v) for k, v in self._best_params.items()}
        combined.update({f"phys_{k}": float(v) for k, v in self._base_phys.items()})

        objectives = {
            "hit_rate":       float(self._best_metrics.get("hit_rate",       0.0)),
            "SEP":            get_sep(self._best_metrics, 100.0),
            "miss_distance":  get_sep(self._best_metrics, 100.0),
            "control_energy": float(self._best_metrics.get("control_energy", 50.0)),
            "pitch_PM":       float(self._best_metrics.get("pitch_PM",       0.0)),
            "pitch_BW":       float(self._best_metrics.get("pitch_BW",       0.0)),
            "pitch_GM":       float(self._best_metrics.get("pitch_GM",       0.0)),
            "peak_ny":        get_peak_ny(self._best_metrics),
            "peak_n":         get_peak_ny(self._best_metrics),
            "peak_ny_max":    self._best_metrics.get(
                "peak_ny_max", self._best_metrics.get("peak_n_max", get_peak_ny(self._best_metrics)),
            ),
        }

        fitness = self._compute_pe_fitness(self._best_metrics)

        ctx = {"task": "guidance_rl_optimization", "algorithm": "PPO",
               "best_reward": float(self._best_reward), **task_context}

        # Split best_params into guidance vs autopilot groups for metadata
        guidance_best  = {k: float(v) for k, v in self._best_params.items() if k in GUIDANCE_PARAM_SPECS}
        autopilot_best = {k: float(v) for k, v in self._best_params.items() if k in AUTOPILOT_PARAM_SPECS}

        target_type = (
            MemoryType.LONG_TERM
            if final_status == "task_requirements_met"
            else MemoryType.SHORT_TERM
        )
        mem_id = await self.parameter_experience.store(
            task_context=ctx,
            parameters=combined,
            objectives=objectives,
            fitness=fitness,
            metadata={
                "optimizer":       "MatlabRLOptimizer",
                "episode_count":   len(self._history),
                "timestamp":       time.time(),
                "final_status":    final_status,
                "guidance_params": guidance_best,
                "autopilot_params": autopilot_best,
            },
            memory_type=target_type,
        )
        logger.info(
            f"Best params stored → memory_id={mem_id}, fitness={fitness:.4f}, "
            f"layer={'LONG_TERM' if target_type == MemoryType.LONG_TERM else 'SHORT_TERM'} | "
            f"guidance={guidance_best} | autopilot_w1={autopilot_best.get('w1')}"
        )

    def get_statistics(self) -> Dict[str, Any]:
        if not self._history:
            return {"status": "not_started"}
        rewards = [ep["reward"] for ep in self._history]
        return {
            "total_episodes":     len(self._history),
            "best_reward":        self._best_reward,
            "best_params":        self._best_params,
            "best_metrics":       self._best_metrics,
            "mean_reward_last10": float(np.mean(rewards[-10:])),
            "reward_trend":       rewards[-20:],
        }
