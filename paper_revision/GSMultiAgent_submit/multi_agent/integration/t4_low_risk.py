"""
T4 low-risk optimization policy — deterministic minimal APN + conservative tuning.

Goals:
  - MODIFY_LAW via fixed gf() template (skip LLM creative rewrite on T4)
  - Conservative autopilot seeds (w≤45, N_pn≤4.5)
  - Block tune after catastrophic MODIFY (peak too high / hit too low)
  - First post-MODIFY tune uses Expert, not PPO
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from multi_agent.metric_contract import (
    BW_MAX,
    BW_MIN,
    HIT_RATE_MIN,
    PEAK_NY_MAX,
    PM_MAX,
    SEP_MAX,
)

logger = logging.getLogger(__name__)

# Conservative autopilot defaults for T4 after MODIFY_LAW
T4_CONSERVATIVE_AUTOPILOT: Dict[str, float] = {
    "w1": 35.0,
    "zeta1": 0.65,
    "tao1": 0.12,
    "w2": 32.0,
    "zeta2": 0.65,
    "tao2": 0.12,
    "w3": 35.0,
    "zeta3": 0.65,
    "N_pn": 3.5,
}

T4_PARAM_CEILINGS: Dict[str, float] = {
    "w1": 45.0,
    "w2": 45.0,
    "w3": 45.0,
    "N_pn": 4.5,
}

_GF_BLOCK_RE = re.compile(
    r"(?m)^function\s+(?:\[[^\]]+\]|[^\n=]+=\s*)?gf\s*\([^\n]*\n[\s\S]*?(?=^function\b|\Z)",
    re.IGNORECASE,
)

# Deterministic minimal APN for T4 — PN + scaled a_T_perp + LPF + output clamp
GF_APN_T4_MINIMAL = r"""function [N1,N2,G1,G2,gc]=gf(Tx,Ty,Tz,XK,TVx,TVy,TVz,Np)
% T4 minimal APN (deterministic low-risk template) — output must stay clamped
g0=9.8; V=XK(1);up=XK(2);ga=XK(3);ph=XK(4);th=XK(5);pv=XK(6);
x=XK(10);y=XK(11);z=XK(12);
Vax=V*cos(th)*cos(pv); Vay=V*sin(th); Vaz=-V*cos(th)*sin(pv);
R=max(norm([x-Tx,y-Ty,z-Tz]),0.1);
Vrx=TVx-Vax; Vry=TVy-Vay; Vrz=TVz-Vaz;
qy=((Tz-z)*Vrx-(Tx-x)*Vrz)/R^2; qz=((Tx-x)*Vry-(Ty-y)*Vrx)/R^2;
Vc=max(-(Vrx*(Tx-x)+Vry*(Ty-y)+Vrz*(Tz-z))/R, 0.1);
persistent TVx_p TVy_p TVz_p qy_f qz_f
if isempty(TVx_p), TVx_p=TVx; TVy_p=TVy; TVz_p=TVz; qy_f=qy; qz_f=qz; end
dt=0.002;
a_Tx=(TVx-TVx_p)/dt; a_Ty=(TVy-TVy_p)/dt; a_Tz=(TVz-TVz_p)/dt;
TVx_p=TVx; TVy_p=TVy; TVz_p=TVz;
alpha=dt/0.04;
qy_f=qy_f+alpha*(qy-qy_f); qz_f=qz_f+alpha*(qz-qz_f);
qy=qy_f; qz=qz_f;
t_go=R/Vc;
N_eff=Np;
ayb_pn=cos(up)*cos(ga)*N_eff*Vc*qz+(sin(up)*sin(ph)*cos(ga)+cos(ph)*sin(ga))*(-N_eff*Vc*qy);
azb_pn=-cos(up)*sin(ga)*N_eff*Vc*qz+(cos(ph)*cos(ga)-sin(up)*sin(ph)*sin(ga))*(-N_eff*Vc*qy);
eR=[Tx-x;Ty-y;Tz-z]/R;
a_T=[a_Tx;a_Ty;a_Tz];
a_T_perp=a_T-dot(a_T,eR)*eR;
comp_scale=0.45;
if t_go<0.15, comp_scale=0; end
A=cos(up)*cos(ga); B=sin(up)*sin(ph)*cos(ga)+cos(ph)*sin(ga);
C=-cos(up)*sin(ga); D=cos(ph)*cos(ga)-sin(up)*sin(ph)*sin(ga);
comp=(N_eff/2)*comp_scale*a_T_perp;
ayb=ayb_pn+A*comp(2)+B*comp(3);
azb=azb_pn+C*comp(2)+D*comp(3);
N1=ayb/g0; N2=azb/g0;
ny_lim=20;
N1=ny_lim*tanh(N1/ny_lim);
N2=ny_lim*tanh(N2/ny_lim);
N1=max(-ny_lim,min(ny_lim,N1));
N2=max(-ny_lim,min(ny_lim,N2));
G1=cos(up)*cos(ga); G2=-cos(up)*sin(ga); gc=0;
end
"""

PLANNER_T4_LOW_RISK_BLOCK = """
【T4 低风险策略 — 高于反思 LLM 的硬约束】
- T4(SUB_IDX=4) 且 Layer2 为 4/5（仅 PeakNy 均值超标）→ mode 必须 MODIFY_LAW，使用系统内置克制版 APN 模板（非 LLM 自由发挥）
- MODIFY_LAW 后首轮 Layer3 仅用 Expert 小步调参，禁止 PPO 100 ep；初值 w1/w2/w3∈[30,40]、N_pn∈[3,4.5]
- 禁止初值 w>45 或 N_pn>4.5
- MODIFY 后若 PeakNy(均值)>灾难阈值或命中率<85%% → 禁止 TUNE_PARAMS，必须重新 MODIFY_LAW
- 判定过载：PeakNy MC 平均值 ≤ peak_ny_mean_max（默认 20g）；peak_ny_max 仅日志参考
"""


def _get_t4_cfg():
    try:
        from multi_agent.config_loader import get_config
        return getattr(get_config(), "t4_low_risk", None)
    except Exception:
        return None


def get_peak_near_miss_margin_g(default: float = 8.0) -> float:
    cfg = _get_t4_cfg()
    if cfg is None:
        return default
    return float(getattr(cfg, "peak_near_miss_g", default))


def t4_first_iter_mode(default: str = "modify_law") -> str:
    cfg = _get_t4_cfg()
    if cfg is None:
        return default
    return str(getattr(cfg, "first_iter_mode", default)).lower()


def post_modify_expert_max_rounds(
    optimization_history: Optional[List[Dict[str, Any]]],
    default: int = 5,
) -> int:
    cfg = _get_t4_cfg()
    if cfg is None or last_mode_from_history(optimization_history) != "MODIFY_LAW":
        return default
    return int(getattr(cfg, "post_modify_expert_rounds", 3))


def should_allow_ppo_after_expert(
    expert_metrics: Optional[Dict[str, Any]],
    task_prompt: str = "",
    optimization_history: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[bool, str]:
    """After Expert on T4, skip PPO when baseline is usable or catastrophic."""
    cfg = _get_t4_cfg()
    if cfg is None or not getattr(cfg, "enabled", True):
        return True, ""
    if not is_t4_mission(task_prompt=task_prompt):
        return True, ""

    from multi_agent.rl.metric_utils import get_peak_ny

    pny = get_peak_ny(expert_metrics or {})
    hit = float((expert_metrics or {}).get("hit_rate") or 0.0)
    peak_max = float(getattr(cfg, "post_modify_ppo_peak_max_g", 40.0))
    hit_min = float(getattr(cfg, "post_modify_ppo_hit_min_pct", 80.0))
    sep_max_skip = float(getattr(cfg, "skip_ppo_expert_sep_max_m", 15.0))
    hit_min_skip = float(getattr(cfg, "skip_ppo_expert_hit_min_pct", 90.0))

    try:
        sep = float(
            (expert_metrics or {}).get("SEP", (expert_metrics or {}).get("miss_distance", 999.0))
        )
    except (TypeError, ValueError):
        sep = 999.0

    if getattr(cfg, "force_skip_ppo_after_expert", True):
        if pny <= peak_max and hit >= hit_min:
            return False, (
                f"T4 快路径：Expert 未灾难性失效 (hit={hit:.1f}% PeakNy={pny:.1f}g)，"
                f"跳过 PPO，直接 CLS"
            )

    if pny > peak_max:
        return False, (
            f"T4 Expert 后 PeakNy(均值)={pny:.1f}g>{peak_max:.0f}g，"
            f"跳过 PPO fallback"
        )
    if hit > 0 and hit < hit_min:
        return False, (
            f"T4 Expert 后命中率 {hit:.1f}%<{hit_min:.0f}%，"
            f"跳过 PPO fallback"
        )
    if sep <= sep_max_skip and hit >= hit_min_skip:
        return False, (
            f"T4 Expert 基线 SEP={sep:.2f}m≤{sep_max_skip:.0f}m 且 "
            f"hit={hit:.1f}%≥{hit_min_skip:.0f}%，跳过 PPO，直接 CLS"
        )
    # Post-MODIFY: PPO rarely helps when Expert already has reasonable hit/SEP
    if (
        last_mode_from_history(optimization_history) == "MODIFY_LAW"
        and sep <= sep_max_skip + 3.0
        and hit >= hit_min_skip - 2.0
    ):
        return False, (
            f"T4 post-MODIFY Expert SEP={sep:.2f}m / hit={hit:.1f}% 接近达标，"
            f"跳过 PPO，直接 CLS"
        )
    return True, ""


def should_store_pe_memory(
    metrics: Optional[Dict[str, Any]],
    task_prompt: str = "",
) -> Tuple[bool, str]:
    """Filter catastrophic PE entries from polluting Planner retrieval."""
    cfg = _get_t4_cfg()
    if cfg is None or not getattr(cfg, "enabled", True):
        return True, ""
    if not is_t4_mission(task_prompt=task_prompt):
        return True, ""

    from multi_agent.integration.design_path_policy import task_fully_satisfied

    if task_prompt and not task_fully_satisfied(metrics, task_prompt):
        return False, "五项指标未全部达标，禁止写入 PE"

    from multi_agent.rl.metric_utils import get_peak_ny

    pny = get_peak_ny(metrics or {})
    hit = float((metrics or {}).get("hit_rate") or 0.0)
    peak_max = float(getattr(cfg, "pe_store_peak_max_g", 25.0))
    hit_min = float(getattr(cfg, "pe_store_hit_min_pct", HIT_RATE_MIN))

    if pny > peak_max:
        return False, f"PeakNy(均值)={pny:.1f}g>{peak_max:.0f}g"
    if hit > 0 and hit < hit_min:
        return False, f"hit={hit:.1f}%<{hit_min:.0f}%"
    return True, ""


def should_gate_near_miss_tune(
    metrics: Optional[Dict[str, Any]],
    gate_message: str = "",
    task_prompt: str = "",
    task_mode: str = "",
) -> Tuple[bool, str]:
    """
    Gate blocked MODIFY but metrics are T4 near-miss → same-iteration Layer3 TUNE.

    Conditions: peak_ny mean in (task_limit, modify_gate_peak], hit/SEP OK.
    """
    cfg = _get_t4_cfg()
    if cfg is None or not getattr(cfg, "enabled", True):
        return False, ""
    if not getattr(cfg, "gate_near_miss_tune_enabled", True):
        return False, ""
    if str(task_mode).upper() != "MODIFY_LAW":
        return False, ""
    if not is_t4_mission(task_prompt=task_prompt):
        return False, ""
    if not metrics:
        return False, ""

    from multi_agent.integration.design_path_policy import (
        peak_ny_limit,
        requirements_from_prompt,
    )
    from multi_agent.rl.metric_utils import get_peak_ny

    reqs = requirements_from_prompt(task_prompt)
    limit = peak_ny_limit(reqs, default=float(getattr(cfg, "ny_limit_g", PEAK_NY_MAX)))
    gate_peak = float(getattr(cfg, "modify_gate_peak_g", 30.0))
    pny_mean = get_peak_ny(metrics)

    if pny_mean <= limit:
        return False, ""
    if pny_mean > gate_peak:
        return False, (
            f"PeakNy(均值)={pny_mean:.1f}g>{gate_peak:.0f}g，超过 Gate 上限，需继续 MODIFY"
        )

    hit_min = float(reqs.get("hit_rate_min", HIT_RATE_MIN))
    sep_max = float(reqs.get("sep_max", SEP_MAX))
    hit = float(metrics.get("hit_rate") or 0.0)
    try:
        sep = float(metrics.get("SEP", metrics.get("miss_distance", 999.0)))
    except (TypeError, ValueError):
        sep = 999.0

    if hit > 0 and hit < hit_min:
        return False, "命中率未达标，不转 TUNE"
    if sep > sep_max:
        return False, "SEP 未达标，不转 TUNE"

    if gate_message and "命中率" in gate_message and "[NG]" in gate_message:
        return False, ""

    return True, (
        f"T4 near-miss: PeakNy(均值)={pny_mean:.1f}g∈({limit:.0f},{gate_peak:.0f}]g "
        f"hit={hit:.1f}% SEP={sep:.2f}m → 同轮 Layer3 TUNE"
    )


def check_modify_law_closed_loop_gate(
    metrics: Optional[Dict[str, Any]],
    task_prompt: str = "",
) -> Tuple[bool, str]:
    """MODIFY_LAW closed-loop quick gate — reject catastrophic scripts (mean only)."""
    cfg = _get_t4_cfg()
    if cfg is None or not getattr(cfg, "enabled", True):
        return True, ""
    if not is_t4_mission(task_prompt=task_prompt):
        return True, ""

    from multi_agent.rl.metric_utils import get_peak_ny

    pny = get_peak_ny(metrics or {})
    hit = float((metrics or {}).get("hit_rate") or 0.0)
    peak_fail = float(getattr(cfg, "modify_fail_peak_g", 50.0))
    hit_min = float(getattr(cfg, "modify_fail_hit_rate_min", 85.0))

    if pny > peak_fail:
        return False, (
            f"MODIFY 闭环快扫 PeakNy(均值)={pny:.1f}g>{peak_fail:.0f}g，"
            f"拒绝进入 Layer3"
        )
    if hit > 0 and hit < hit_min:
        return False, (
            f"MODIFY 闭环快扫命中率 {hit:.1f}%<{hit_min:.0f}%，拒绝进入 Layer3"
        )
    return True, ""


def borderline_nmc_revalidate_threshold(task_prompt: str = "") -> int:
    cfg = _get_t4_cfg()
    if cfg is None or not is_t4_mission(task_prompt=task_prompt):
        return 0
    return int(getattr(cfg, "borderline_nmc_revalidate", 0))


def is_t4_mission(
    mission_conditions: Optional[str] = None,
    task_prompt: Optional[str] = None,
) -> bool:
    """True when RUN_CASE T + SUB_IDX 4, or prompt mentions T4 / 工况4."""
    mc = mission_conditions or ""
    tp = task_prompt or ""
    combined = f"{mc}\n{tp}"
    if re.search(r"RUN_CASE\s*=\s*['\"]T['\"]", combined, re.I):
        m = re.search(r"SUB_IDX\s*=\s*(\d+)", combined)
        if m and int(m.group(1)) == 4:
            return True
        if re.search(r"SUB_IDX\s*=\s*0", combined) and re.search(
            r"\bT4\b|工况\s*4|子工况\s*4", combined, re.I
        ):
            return True
    if re.search(r"\bT4\b|工况\s*4|T\s*类.*4|高频大幅", tp, re.I):
        return True
    return False


def use_deterministic_apn(
    mission_conditions: Optional[str] = None,
    task_prompt: Optional[str] = None,
) -> bool:
    cfg = _get_t4_cfg()
    if cfg is None or not getattr(cfg, "enabled", True):
        return False
    if not getattr(cfg, "deterministic_apn", True):
        return False
    return is_t4_mission(mission_conditions, task_prompt)


def conservative_autopilot_dict(cfg: Any = None) -> Dict[str, float]:
    if cfg is None:
        cfg = _get_t4_cfg()
    base = dict(T4_CONSERVATIVE_AUTOPILOT)
    if cfg is not None:
        overrides = getattr(cfg, "conservative_autopilot", None) or {}
        for k, v in overrides.items():
            try:
                base[k] = float(v)
            except (TypeError, ValueError):
                pass
    return clamp_autopilot_params(base, cfg)


def clamp_autopilot_params(
    params: Optional[Dict[str, float]] = None,
    cfg: Any = None,
) -> Dict[str, float]:
    """Merge with conservative defaults and enforce T4 ceilings."""
    out = conservative_autopilot_dict(cfg) if not params else dict(params)
    if cfg is None:
        cfg = _get_t4_cfg()
    ceilings = dict(T4_PARAM_CEILINGS)
    if cfg is not None:
        for k in ("max_w", "max_n_pn"):
            v = getattr(cfg, k, None)
            if v is None:
                continue
            if k == "max_w":
                for wk in ("w1", "w2", "w3"):
                    ceilings[wk] = float(v)
            elif k == "max_n_pn":
                ceilings["N_pn"] = float(v)
    for key, ceil in ceilings.items():
        if key in out:
            try:
                out[key] = min(float(out[key]), float(ceil))
            except (TypeError, ValueError):
                out[key] = T4_CONSERVATIVE_AUTOPILOT.get(key, out[key])
    floors = {"w1": 20.0, "w2": 20.0, "w3": 20.0, "N_pn": 2.5}
    if cfg is not None:
        min_w1 = getattr(cfg, "min_w1", None)
        if min_w1 is not None:
            floors["w1"] = max(float(floors["w1"]), float(min_w1))
    for key, floor in floors.items():
        if key in out:
            out[key] = max(float(out[key]), floor)
    return out


def maybe_clamp_t4_params(
    params: Optional[Dict[str, float]],
    *,
    task_prompt: Optional[str] = None,
    mission_conditions: Optional[Any] = None,
) -> Dict[str, float]:
    """Apply T4 w/N_pn ceilings and w1 floor when mission is T4."""
    base = dict(params or {})
    mc = mission_conditions if isinstance(mission_conditions, str) else str(mission_conditions or "")
    if not is_t4_mission(mc, task_prompt):
        return base
    return clamp_autopilot_params(base)


def replace_gf_function(script: str, new_gf: str) -> str:
    m = _GF_BLOCK_RE.search(script or "")
    if not m:
        logger.warning("[T4LowRisk] gf() not found — script unchanged")
        return script
    tail = script[m.end():]
    if tail.startswith("\n"):
        tail = tail[1:]
    return script[: m.start()] + new_gf.strip() + "\n\n" + tail


def apply_deterministic_t4_modify_law(
    script: str,
    ny_limit_g: float = PEAK_NY_MAX,
) -> str:
    """Replace gf() with minimal APN template (optionally patch ny_lim in source)."""
    gf_body = GF_APN_T4_MINIMAL
    if abs(ny_limit_g - PEAK_NY_MAX) > 0.01:
        gf_body = gf_body.replace("ny_lim=20", f"ny_lim={ny_limit_g:.1f}")
    out = replace_gf_function(script, gf_body)
    logger.info("[T4LowRisk] Applied deterministic minimal APN gf() for T4")
    return out


DETERMINISTIC_APN_MARKER = "T4 minimal APN"


def script_has_deterministic_t4_apn(script_text: str) -> bool:
    return DETERMINISTIC_APN_MARKER in (script_text or "")


def apply_t4_deterministic_gf_if_enabled(
    script: str,
    mission_conditions: Optional[str] = None,
    task_prompt: Optional[str] = None,
) -> Tuple[str, bool]:
    """Inject minimal APN gf() on T4 when ``deterministic_apn`` is enabled."""
    if not use_deterministic_apn(mission_conditions, task_prompt):
        return script, False
    try:
        from multi_agent.config_loader import get_config

        _ny_lim = float(getattr(get_config().t4_low_risk, "ny_limit_g", PEAK_NY_MAX))
    except Exception:
        _ny_lim = PEAK_NY_MAX
    return apply_deterministic_t4_modify_law(script, ny_limit_g=_ny_lim), True


def should_skip_physics_bounds_probe(
    script_text: str,
    mission_conditions: Optional[str] = None,
    task_prompt: Optional[str] = None,
) -> bool:
    """Skip gf() probe when built-in deterministic T4 APN is present (already clamped)."""
    if not script_has_deterministic_t4_apn(script_text):
        return False
    if not is_t4_mission(mission_conditions, task_prompt):
        return False
    try:
        from multi_agent.config_loader import get_config

        cfg = get_config().simulation
        if not getattr(cfg, "physics_bounds_skip_deterministic_t4_apn", True):
            return False
    except Exception:
        pass
    return True


def last_mode_from_history(
    optimization_history: Optional[List[Dict[str, Any]]],
) -> str:
    if not optimization_history:
        return ""
    return str(optimization_history[-1].get("task_mode") or optimization_history[-1].get("mode") or "").upper()


def should_block_tune_after_bad_modify(
    metrics: Optional[Dict[str, Any]],
    task_prompt: str,
    optimization_history: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[bool, str]:
    """After MODIFY_LAW, block TUNE if results are catastrophically bad."""
    cfg = _get_t4_cfg()
    if cfg is None or not getattr(cfg, "enabled", True):
        return False, ""
    if last_mode_from_history(optimization_history) != "MODIFY_LAW":
        return False, ""
    if not is_t4_mission(task_prompt=task_prompt):
        return False, ""

    from multi_agent.integration.design_path_policy import peak_ny_limit, requirements_from_prompt
    from multi_agent.rl.metric_utils import get_peak_ny

    reqs = requirements_from_prompt(task_prompt)
    limit = peak_ny_limit(reqs)
    pny = get_peak_ny(metrics or {})
    hit = float((metrics or {}).get("hit_rate") or 0.0)
    peak_fail = float(getattr(cfg, "modify_fail_peak_g", 50.0))
    hit_min = float(getattr(cfg, "modify_fail_hit_rate_min", 85.0))

    if pny > max(limit, peak_fail):
        return True, (
            f"MODIFY_LAW 后 PeakNy(均值)={pny:.1f}g 仍>{peak_fail:.0f}g，"
            f"禁止 TUNE_PARAMS，需重新 MODIFY（检查 gf 限幅/前馈增益）"
        )
    if hit > 0 and hit < hit_min:
        return True, (
            f"MODIFY_LAW 后命中率 {hit:.1f}%<{hit_min:.0f}%，"
            f"禁止 TUNE_PARAMS，需重新 MODIFY_LAW"
        )
    return False, ""


def _metric_bw_pm(metrics: Optional[Dict[str, Any]]) -> Tuple[float, float]:
    m = metrics or {}
    try:
        bw = float(m.get("pitch_BW") or 0.0)
    except (TypeError, ValueError):
        bw = 0.0
    try:
        pm = float(m.get("pitch_PM") or 0.0)
    except (TypeError, ValueError):
        pm = 0.0
    return bw, pm


def build_expert_playbook_block(
    metrics: Dict[str, Any],
    round_i: int,
    *,
    task_prompt: str = "",
    mission_conditions: Optional[Any] = None,
) -> str:
    """Deterministic Expert prompt appendix — reduces DeepSeek variance on T4."""
    if not is_t4_mission(mission_conditions, task_prompt):
        return ""
    bw, pm = _metric_bw_pm(metrics)
    cfg = _get_t4_cfg()
    max_w = float(getattr(cfg, "max_w", 44.0) or 44.0) if cfg else 44.0
    r1_cap = float(getattr(cfg, "expert_r1_w1_cap", 40.0) or 40.0) if cfg else 40.0
    lines = [
        "\n【T4 调参强制策略 — 必须遵守】\n",
        f"- 当前 BW={bw:.1f}rad/s PM={pm:.1f}° 第{round_i}轮\n",
    ]
    if round_i == 1 and bw < BW_MIN:
        lines += [
            "- R1 且 BW<20：w1 必须在 [36, 40] 内，禁止取 max_w 上限\n",
            f"- PM>70 时禁止增大 w1；优先 zeta1∈[0.65,0.75] tao1≤0.12\n",
            f"- 推荐初值：w1={min(38.0, r1_cap):.0f} zeta1=0.70 tao1=0.10 "
            "w2=35 zeta2=0.70 w3=25 zeta3=0.90 N_pn=4.0\n",
        ]
    if bw > BW_MAX:
        lines += [
            "- BW>85：必须降低 w1（至少减 4），禁止 w1≥"
            f"{min(r1_cap, max_w - 2):.0f}\n",
        ]
    if pm > PM_MAX:
        lines += [
            "- PM>70：禁止增大 w1；可增大 zeta1 或减小 tao1 以降低 PM\n",
        ]
    lines.append("- 只返回 JSON，数值保留 3 位小数，analysis 不超过 80 字\n")
    return "".join(lines)


def should_use_rule_first_expert_round(
    round_i: int,
    metrics: Dict[str, Any],
    *,
    task_prompt: str = "",
    mission_conditions: Optional[Any] = None,
    cfg: Any = None,
) -> bool:
    """True when R1 should skip LLM and use deterministic seed (FastPath low-BW)."""
    if round_i != 1:
        return False
    if not is_t4_mission(mission_conditions, task_prompt):
        return False
    if cfg is None:
        cfg = _get_t4_cfg()
    if cfg is None or not getattr(cfg, "enabled", True):
        return False
    if not getattr(cfg, "expert_rule_first_r1", True):
        return False
    bw, _pm = _metric_bw_pm(metrics)
    return bw < BW_MIN


def rule_seed_expert_params(
    metrics: Dict[str, Any],
    round_i: int,
    best_params: Dict[str, float],
    *,
    task_prompt: str = "",
    mission_conditions: Optional[Any] = None,
) -> Optional[Dict[str, float]]:
    """Deterministic Expert parameter seed for T4 (None if not applicable)."""
    if not is_t4_mission(mission_conditions, task_prompt):
        return None
    cfg = _get_t4_cfg()
    if cfg is None or not getattr(cfg, "enabled", True):
        return None

    bw, pm = _metric_bw_pm(metrics)
    max_w = float(getattr(cfg, "max_w", 44.0) or 44.0)
    min_w1 = float(getattr(cfg, "min_w1", 34.0) or 34.0)
    r1_cap = float(getattr(cfg, "expert_r1_w1_cap", 40.0) or 40.0)
    cons = conservative_autopilot_dict(cfg)
    ny_lim = float(getattr(cfg, "ny_limit_g", PEAK_NY_MAX) or PEAK_NY_MAX)

    if round_i == 1 and bw < BW_MIN:
        seed = {
            **cons,
            "w1": min(38.0, r1_cap, max_w - 4.0),
            "zeta1": 0.70,
            "tao1": 0.10,
            "w2": 35.0,
            "zeta2": 0.70,
            "tao2": 0.10,
            "w3": 25.0,
            "zeta3": 0.90,
            "N_pn": 4.0,
            "ny_lim": ny_lim,
        }
        return clamp_autopilot_params(seed, cfg)

    if round_i >= 2 and (bw > BW_MAX or pm > PM_MAX):
        base = dict(best_params or cons)
        try:
            w1 = float(base.get("w1", cons["w1"]))
        except (TypeError, ValueError):
            w1 = float(cons["w1"])
        if bw > BW_MAX:
            base["w1"] = max(min_w1, w1 - 4.0)
        if pm > PM_MAX:
            try:
                z1 = float(base.get("zeta1", cons["zeta1"]))
            except (TypeError, ValueError):
                z1 = float(cons["zeta1"])
            base["zeta1"] = min(0.85, z1 + 0.05)
        return clamp_autopilot_params(base, cfg)

    return None


def enforce_expert_suggestion(
    suggested: Dict[str, float],
    metrics: Dict[str, Any],
    *,
    round_i: int,
    task_prompt: str = "",
    mission_conditions: Optional[Any] = None,
) -> Dict[str, float]:
    """Post-LLM hard rules: block max_w trap and high-BW / high-PM violations."""
    out = dict(suggested or {})
    if not is_t4_mission(mission_conditions, task_prompt):
        return maybe_clamp_t4_params(
            out, task_prompt=task_prompt, mission_conditions=mission_conditions,
        )

    cfg = _get_t4_cfg()
    max_w = float(getattr(cfg, "max_w", 44.0) or 44.0) if cfg else 44.0
    min_w1 = float(getattr(cfg, "min_w1", 34.0) or 34.0) if cfg else 34.0
    r1_cap = float(getattr(cfg, "expert_r1_w1_cap", 40.0) or 40.0) if cfg else 40.0
    bw, pm = _metric_bw_pm(metrics)

    try:
        w1 = float(out.get("w1", 35.0))
    except (TypeError, ValueError):
        w1 = 35.0

    if round_i == 1 and bw < BW_MIN:
        if w1 >= max_w - 0.5 or w1 > r1_cap:
            w1 = min(38.0, r1_cap, max_w - 4.0)
            out["w1"] = w1

    if bw > BW_MAX or pm > PM_MAX:
        cap = r1_cap if bw > BW_MAX else max(r1_cap, min(42.0, max_w - 2.0))
        if w1 > cap:
            out["w1"] = max(min_w1, cap)
        if bw > BW_MAX and float(out.get("w1", w1)) > min_w1 + 2:
            out["w1"] = max(min_w1, min(float(out["w1"]), cap))

    return maybe_clamp_t4_params(
        out, task_prompt=task_prompt, mission_conditions=mission_conditions,
    )


def pre_cls_param_fixup(
    params: Dict[str, float],
    metrics: Dict[str, Any],
    *,
    task_prompt: str = "",
    mission_conditions: Optional[Any] = None,
) -> Dict[str, float]:
    """Nudge Expert best away from high-BW / high-PM basin before CLS."""
    if not is_t4_mission(mission_conditions, task_prompt):
        return dict(params or {})
    cfg = _get_t4_cfg()
    if cfg is None or not getattr(cfg, "enabled", True):
        return dict(params or {})
    if not getattr(cfg, "pre_cls_bw_pm_fixup", True):
        return dict(params or {})

    bw, pm = _metric_bw_pm(metrics)
    if bw <= BW_MAX and pm <= PM_MAX:
        return dict(params or {})

    out = dict(params or {})
    min_w1 = float(getattr(cfg, "min_w1", 34.0) or 34.0)
    drop = float(getattr(cfg, "pre_cls_w1_drop", 6.0) or 6.0)
    try:
        w1 = float(out.get("w1", 38.0))
    except (TypeError, ValueError):
        w1 = 38.0

    if bw > BW_MAX:
        out["w1"] = max(min_w1, w1 - drop)
    if pm > PM_MAX:
        try:
            z1 = float(out.get("zeta1", 0.70))
        except (TypeError, ValueError):
            z1 = 0.70
        out["zeta1"] = min(0.88, z1 + 0.08)
        if bw > 70.0:
            out["w1"] = max(min_w1, float(out.get("w1", w1)) - 2.0)

    fixed = maybe_clamp_t4_params(
        out, task_prompt=task_prompt, mission_conditions=mission_conditions,
    )
    logger.info(
        "[T4LowRisk] pre-CLS fixup BW=%.1f PM=%.1f → w1=%.1f zeta1=%.2f",
        bw,
        pm,
        float(fixed.get("w1", 0)),
        float(fixed.get("zeta1", 0)),
    )
    return fixed


def expert_llm_temperature(attempt: int, cfg: Any = None) -> float:
    """Expert LLM temperature — default 0 for reproducibility."""
    if cfg is None:
        cfg = _get_t4_cfg()
    base = float(getattr(cfg, "expert_temperature", 0.0) or 0.0) if cfg else 0.0
    return min(0.2, base + 0.1 * max(0, attempt - 1))


def should_force_expert_after_modify(
    optimization_history: Optional[List[Dict[str, Any]]],
    task_prompt: str = "",
) -> bool:
    """First tune after MODIFY_LAW on T4 → Expert only (no PPO)."""
    cfg = _get_t4_cfg()
    if cfg is None or not getattr(cfg, "enabled", True):
        return False
    if last_mode_from_history(optimization_history) != "MODIFY_LAW":
        return False
    if not is_t4_mission(task_prompt=task_prompt):
        return False
    strat = str(getattr(cfg, "post_modify_first_tune_strategy", "expert")).lower()
    return strat in ("expert", "skip_ppo")


def resolve_initial_auto_params(
    params: Optional[Dict[str, float]],
    task_prompt: str = "",
    mission_conditions: Optional[str] = None,
    optimization_history: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, float]:
    """Apply conservative seeds for T4 / post-MODIFY iterations."""
    cfg = _get_t4_cfg()
    if cfg is None or not getattr(cfg, "enabled", True):
        return dict(params or {})

    use_conservative = is_t4_mission(mission_conditions, task_prompt)
    if last_mode_from_history(optimization_history) == "MODIFY_LAW":
        use_conservative = True
    if not use_conservative:
        return clamp_autopilot_params(params, cfg) if params else dict(params or {})

    merged = conservative_autopilot_dict(cfg)
    if params:
        merged.update({k: float(v) for k, v in params.items() if isinstance(v, (int, float))})
    return clamp_autopilot_params(merged, cfg)


def layer3_max_episodes_cap(
    optimization_history: Optional[List[Dict[str, Any]]],
    default_max: int,
    metrics: Optional[Dict[str, Any]] = None,
    task_prompt: str = "",
) -> int:
    cfg = _get_t4_cfg()
    if cfg is None or not getattr(cfg, "enabled", True):
        return default_max

    cap = default_max
    if last_mode_from_history(optimization_history) == "MODIFY_LAW":
        cap = min(cap, int(getattr(cfg, "post_modify_ppo_max_episodes", 40)))
    elif metrics and is_t4_mission(task_prompt=task_prompt):
        from multi_agent.integration.design_path_policy import (
            peak_ny_limit,
            peak_ny_near_miss,
            requirements_from_prompt,
        )
        reqs = requirements_from_prompt(task_prompt)
        limit = peak_ny_limit(reqs)
        margin = get_peak_near_miss_margin_g()
        if peak_ny_near_miss(metrics, limit, margin):
            cap = max(
                cap,
                min(
                    int(getattr(cfg, "near_miss_ppo_max_episodes", 60)),
                    default_max,
                ),
            )
    if last_mode_from_history(optimization_history) == "MODIFY_LAW":
        legacy = int(getattr(cfg, "post_modify_max_episodes", 50))
        cap = min(cap, legacy)
    return cap


def hermes_modify_law_prompt_suffix(
    mission_conditions: Optional[str] = None,
    task_prompt: Optional[str] = None,
) -> str:
    if not use_deterministic_apn(mission_conditions, task_prompt):
        return ""
    return (
        "\n[T4 低风险 MODIFY_LAW]\n"
        "系统将使用内置克制版 APN gf() 模板（含低通、0.45 前馈、20g 输出限幅）。\n"
        "请勿请求 LLM 重写为激进 APN；初值 w1/w2/w3≤40、N_pn≤4.0。\n"
    )


def apply_reflection_t4_overrides(
    parsed: Dict[str, Any],
    metrics: Optional[Dict[str, Any]],
    task_prompt: str,
    optimization_history: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Override reflection when it contradicts T4 low-risk rules."""
    cfg = _get_t4_cfg()
    if cfg is None or not getattr(cfg, "enabled", True):
        return parsed
    if not is_t4_mission(task_prompt=task_prompt):
        return parsed

    out = dict(parsed)
    blocked, reason = should_block_tune_after_bad_modify(
        metrics, task_prompt, optimization_history,
    )
    if blocked:
        out["next_action"] = "modify_law"
        out["needs_optimization"] = True
        prev = str(out.get("suggestion") or "")
        if reason not in prev:
            out["suggestion"] = f"{prev}\n[T4低风险] {reason}".strip()
        return out

    from multi_agent.integration.design_path_policy import (
        non_peak_constraints_met,
        peak_ny_limit,
        peak_ny_over_limit,
        requirements_from_prompt,
    )

    reqs = requirements_from_prompt(task_prompt)
    limit = peak_ny_limit(reqs)
    if (
        metrics
        and non_peak_constraints_met(metrics, reqs)
        and peak_ny_over_limit(metrics, limit)
        and str(out.get("next_action", "")).lower() == "tune_params"
        and not optimization_history
    ):
        out["next_action"] = "modify_law"
        note = "T4 4/5 仅 peak 超标 — 低风险策略要求 MODIFY_LAW 而非继续 TUNE"
        if note not in str(out.get("suggestion") or ""):
            out["suggestion"] = f"{out.get('suggestion', '')}\n{note}".strip()

    return out
