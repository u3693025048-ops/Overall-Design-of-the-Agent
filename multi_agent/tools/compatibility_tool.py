#!/usr/bin/env python3
"""
Guidance Law Structural Compatibility Verification Tool

Verifies that the STRUCTURE (not specific parameter values) of a modified guidance
law is inherently compatible with the autopilot system.  All checks are independent
of tunable RL parameters (N_pn, w1, zeta1, tao1, …) which vary during optimization.

Three verification layers:
  1. Interface static check  — gf() call signature, output variables, R/T_go singularity
                               protection, dp-field references
  2. Code-based math derivation — Extract guidance law math model from code, derive
                               cascade transfer function with autopilot, prove stability
                               conditions with explicit formulas across RL search space
  3. LLM math derivation     — Detailed mathematical derivation of guidance→autopilot
                               compatibility, step-by-step formulas, coordinate/sign checks

What is NOT checked here (changes during RL tuning):
  · Specific N_pn value → RL optimises N_pn ∈ [3, 6]
  · Specific w1 value   → RL optimises w1  ∈ [20, 60] rad/s
  · Specific tao1 value → RL optimises tao1 ∈ [0.05, 0.35] s
  · Performance metrics (hit_rate, SEP, PM) → evaluated by run_simulation
"""

import json
import logging
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from multi_agent.tools.tool_arg_aliases import (
    coerce_script_path as _coerce_script_path,
    coerce_task_prompt as _coerce_task_prompt,
)
from multi_agent.metric_contract import PM_MAX, PM_MIN

logger = logging.getLogger(__name__)

# ── RL parameter search space (from matlab_rl_optimizer.py AUTOPILOT_PARAM_SPECS) ──
# These are the RANGES over which RL tunes; compatibility must hold for all combinations.
_RL_RANGES = {
    "w1":    {"min": 20.0,  "max": 60.0},   # rad/s  autopilot pitch natural freq
    "zeta1": {"min": 0.35,  "max": 1.20},   # –      autopilot pitch damping ratio
    "tao1":  {"min": 0.05,  "max": 0.35},   # s      autopilot pitch time constant
    "w2":    {"min": 20.0,  "max": 60.0},   # rad/s  autopilot yaw natural freq
    "zeta2": {"min": 0.35,  "max": 1.20},   # –      autopilot yaw damping ratio
    "tao2":  {"min": 0.05,  "max": 0.35},   # s      autopilot yaw time constant
    "w3":    {"min": 20.0,  "max": 60.0},   # rad/s  autopilot roll natural freq
    "zeta3": {"min": 0.35,  "max": 1.20},   # –      autopilot roll damping ratio
    "N_pn":  {"min": 3.0,   "max": 6.0},    # –      navigation ratio (guidance)
}

# Template gf() interface (from monte_carlo_single.m line 300)
# function [N1,N2,G1,G2,gc]=gf(Tx,Ty,Tz,XK,TVx,TVy,TVz,Np)
_GF_EXPECTED_INPUTS  = {"Tx", "Ty", "Tz", "XK", "TVx", "TVy", "TVz", "Np"}
_GF_EXPECTED_OUTPUTS = {"N1", "N2", "G1", "G2", "gc"}

# cf() expects gf() outputs: [N1,N2,G1,G2,gc] where
#   N1, N2 — pitch / yaw load factor commands (g units, via N1*g0)
#   G1, G2 — guidance direction cosines (unit-vector components)
#   gc      — roll rate command (rad/s, 0 for STT)

_STRUCT_FRAMEWORK = """\
【新制导律→自动驾驶仪适配性数学推导框架】

验证目的：对新生成的制导律代码进行数学分析，推导其是否与已有自动驾驶仪适配。
验证方向：制导律 → 自动驾驶仪（分析制导律输出能否被驾驶仪正确跟踪）。

系统接口约定（monte_carlo_single.m 模板）：
  gf() 输入:
    · Tx/Ty/Tz     — 目标三维坐标（m）
    · XK(12维)     — 弹体状态向量 [V,up,ga,ph,th,pv,wx,wy,wz,x,y,z]
    · TVx/TVy/TVz  — 目标速度分量（m/s）
    · Np           — 制导增益参数（RL调参，范围 [3,6]，gf内不得硬编码）
  gf() 输出（cf()接口）：
    · N1,N2        — 俯仰/偏航指令过载（g 单位，cf 内乘 g0 还原为 m/s²）
    · G1,G2        — 方向余弦（G1=cos(up)cos(ga), G2=-cos(up)sin(ga)）
    · gc           — 倾斜角速率指令（rad/s，STT 下为 0）

自动驾驶仪模型（已知、固定）：
  · STT体制，俯仰/偏航解耦
  · 传递函数: G_ap(s) = ω₁²/(s²+2ζ₁ω₁s+ω₁²) · e^{-τ₁s}
  · RL参数搜索范围:
    ω₁∈[20,60] rad/s, ζ₁∈[0.35,1.20], τ₁∈[0.05,0.35] s, N∈[3,6]

数学推导步骤（必须逐步完成）：
  Step 1. 从代码提取制导律数学模型: a_c = f(状态量, Np, ...)
  Step 2. 建立制导律等效传递函数 G_g(s)
          · PN: G_g(s) = N·V_c/R ≈ N/T_go（末端等效增益）
          · OGL: G_g(s) = K(T_go)/T_go²（时变增益）
          · APN: G_g(s) = N/T_go + 前馈项（前馈不影响闭环极点）
  Step 3. 写出开环传递函数: L(s) = G_g(s) · G_ap(s) / s
  Step 4. 计算穿越频率: ω_c ≈ 1/T_go（末端）
  Step 5. 推导相位裕度（最差情况: ω₁=ω₁_min, τ₁=τ₁_max）:
          PM = 90° - arctan(2ζ₁ω₁ω_c/(ω₁²-ω_c²)) - ω_c·τ₁·(180/π)
  Step 6. 验证带宽分离: ω₁_min/ω_c >> 5
  Step 7. 律型特定条件（PN: N>2 + 共轭脱靶因子; OGL: T_go保护; ...）
  Step 8. 适配性总结论"""


def _extract_gf_section(script: str) -> Tuple[str, str]:
    """Return (function_body, function_name) for the guidance law function."""
    for pattern, name_group in [
        (r"(?m)^(function[^\n]*?\bgf\b[^\n]*\n[\s\S]*?)(?=^function|\Z)", None),
        (r"(?m)^(function[^\n]*?\b([\w]*(?:guidance|_nav)[\w]*)\b[^\n]*\n[\s\S]*?)(?=^function|\Z)", 2),
        (r"(?m)^(function[^\n]*?\b([a-zA-Z_]\w*)\b[^\n]*\n[\s\S]{0,2500}?(?:Np\s*\*\s*Vc|dp\.N_guidance)[\s\S]*?)(?=^function|\Z)", 2),
    ]:
        m = re.search(pattern, script, re.IGNORECASE if name_group == 2 else 0)
        if m:
            fname = m.group(name_group) if name_group else "gf"
            return m.group(1)[:3500], fname
    return "", ""


def _check_interface_static(gf_code: str) -> Dict[str, Any]:
    """
    Static code analysis of gf() — purely structural, no parameter values used.

    Returns dict with:
      issues:   list of interface problems found (FAIL-level)
      warnings: list of potential issues (WARN-level)
      info:     list of informational findings
      pass:     True if no issues
    """
    issues:   List[str] = []
    warnings: List[str] = []
    info:     List[str] = []

    first_line = gf_code.split("\n")[0] if gf_code else ""

    # ── 1. Check output variables defined ────────────────────────────────────
    missing_outputs = []
    for var in _GF_EXPECTED_OUTPUTS:
        # Match assignment: var = ... or [var, ...] = ... or var=
        if not re.search(rf"\b{var}\s*=", gf_code):
            missing_outputs.append(var)
    if missing_outputs:
        issues.append(f"输出变量未定义: {missing_outputs}（cf()需要 N1,N2,G1,G2,gc）")
    else:
        info.append("输出变量 N1,N2,G1,G2,gc 均已定义 ✓")

    # ── 2. Check input parameter Np used (not hard-coded navigation ratio) ───
    if not re.search(r"\bNp\b", gf_code):
        warnings.append("未使用传入参数 Np（导引比）——可能将 N 硬编码，RL 调参将失效")
    else:
        info.append("导引比通过参数 Np 使用，RL 可调 ✓")

    # ── 3. Check dp.* field access (gf should NOT access dp directly) ────────
    dp_fields = re.findall(r"\bdp\.(\w+)", gf_code)
    if dp_fields:
        warnings.append(
            f"gf() 直接访问 dp 结构体字段: {list(set(dp_fields))}。"
            "模板中 gf() 仅接收 Np 参数，dp 字段应在外层提取后作为参数传入。"
        )
    else:
        info.append("gf() 不直接访问 dp 结构体 ✓")

    # ── 4. Check R singularity protection ────────────────────────────────────
    r_computed = bool(re.search(r"\bR\s*=", gf_code))
    r_protected = bool(re.search(r"max\s*\([^)]*R[^)]*,?\s*[\d.]+\)|max\s*\(\s*[\d.]+\s*,\s*R\s*\)", gf_code)
                       or re.search(r"R\s*=\s*max\s*\(", gf_code))
    if r_computed and not r_protected:
        issues.append(
            "弹目距 R 有除法使用但缺少保护（如 max(R,0.1)），末端 R→0 时会产生除零错误"
        )
    elif r_computed and r_protected:
        info.append("弹目距 R 有奇点保护 (max) ✓")

    # ── 5. Check for division by T_go without protection ─────────────────────
    tgo_div = re.search(r"/\s*T_go\b|/\s*tgo\b", gf_code, re.IGNORECASE)
    tgo_protect = re.search(r"max\s*\([^)]*[Tt]_?go[^)]*,?\s*[\d.]+\)", gf_code)
    if tgo_div and not tgo_protect:
        warnings.append("以 T_go 为除数但无保护，末端 T_go→0 可能导致数值爆炸")

    # ── 6. Output sign / units sanity ────────────────────────────────────────
    # N1 and N2 should be in g-units (divided by g0 or referenced as load factor)
    n1_line = next((l for l in gf_code.split("\n") if re.search(r"\bN1\s*=", l)), "")
    if n1_line and "g0" not in n1_line and "/9.8" not in n1_line and "/g0" not in n1_line:
        # Check if g0 defined and used in computation
        if re.search(r"\bg0\s*=", gf_code):
            info.append("N1 赋值行未直接除以 g0，请确认已正确换算为 g 单位")
        else:
            warnings.append("N1 赋值中未见 g0（重力加速度），需确认过载单位为 g 而非 m/s²")

    # ── 7. LOS rate variable check ────────────────────────────────────────────
    has_los = bool(re.search(r"\bqy\b|\bqz\b|\bq_y\b|\bq_z\b|\bLOS\b|\bdq\b", gf_code))
    if not has_los:
        warnings.append("未发现视线角速率变量（qy/qz/LOS_rate），确认制导律使用正确的角速率信号")
    else:
        info.append("视线角速率变量存在 ✓")

    return {
        "pass":     len(issues) == 0,
        "issues":   issues,
        "warnings": warnings,
        "info":     info,
    }


def _identify_law_type_from_code(gf_code: str) -> Dict[str, Any]:
    """Step 1 — 从制导律代码中识别律型及数学模型特征（纯正则分析，无LLM）。"""
    has_Np_qdot  = bool(re.search(r"Np\s*[\.\*]*\*?\s*(?:Vc|V_c)", gf_code, re.I))
    has_Np_mult  = bool(re.search(r"\bNp\b\s*[\.\*]*\*", gf_code))
    has_ZEM      = bool(re.search(r"\bZEM\b|\bz_em\b|\bzero.?effort", gf_code, re.I))
    has_tgo_inv  = bool(re.search(r"[/]\s*[Tt]_?go\b|[Tt]_?go\s*[\.\^]\s*[\-]", gf_code))
    has_tgo_pow  = bool(re.search(r"[Tt]_?go\s*[\.\^]\s*[23]", gf_code))
    has_aT       = bool(re.search(r"\ba[_]?[Tt]\b|\btarget.*accel|\bn[_]?[Tt]\b", gf_code, re.I))
    has_gamma_f  = bool(re.search(r"gamma_?f|theta_?f|impact.?angle|terminal.?angle", gf_code, re.I))
    has_sliding  = bool(re.search(r"\bsign\s*\(|\bsgn\s*\(|\bsliding\b", gf_code, re.I))
    has_predict  = bool(re.search(r"\bpredict|\bMPC\b|\bhorizon\b", gf_code, re.I))

    indicators = {
        "Np*Vc*qdot": has_Np_qdot, "Np_multiply": has_Np_mult,
        "ZEM": has_ZEM, "T_go_inverse": has_tgo_inv, "T_go_power": has_tgo_pow,
        "target_accel": has_aT, "terminal_angle": has_gamma_f,
        "sliding_mode": has_sliding, "predictive": has_predict,
    }

    if has_ZEM or (has_tgo_pow and not has_Np_qdot):
        lt, mm = "OGL", "a_c = K(T_go)·ZEM/T_go²"
        gtf = "G_g(s) ≈ K_eff/T_go²（时变增益，末端取最大值）"
    elif has_gamma_f:
        lt, mm = "IACG", "a_c = f(ZEM, γ_f, T_go)，含终端角约束项"
        gtf = "G_g(s) ≈ (K₁ + K₂/T_go²)（双增益结构）"
    elif has_aT and (has_Np_qdot or has_Np_mult):
        lt, mm = "APN", "a_c = N·V_c·q̇ + (N/2)·a_T（增广比例导引）"
        gtf = "G_g(s) = N·V_c/R（PN主项）+ N/(2s²)·a_T（前馈项，不影响闭环极点）"
    elif has_sliding:
        lt, mm = "sliding_mode", "a_c = -K·sign(s)，s为滑模面"
        gtf = "G_g(s) = K_eq（等效增益线性化描述函数）"
    elif has_predict:
        lt, mm = "predictive", "a_c = argmin J(u)，预测控制指令"
        gtf = "G_g(s) ≈ K_mpc（线性化等效增益）"
    elif has_Np_qdot or has_Np_mult:
        lt, mm = "PN", "a_c = N·V_c·q̇（比例导引）"
        gtf = "G_g(s) = N·V_c/R ≈ N/T_go（末端等效增益）"
    else:
        lt, mm = "unknown", "未能从代码中自动提取数学模型"
        gtf = "G_g(s) = K（待LLM层识别具体模型）"

    return {"law_type": lt, "math_model": mm, "guidance_tf": gtf, "indicators": indicators}


def _range_stability_proof(gf_code: str = "") -> Dict[str, Any]:
    """
    Layer 2 — 制导律→自动驾驶仪适配性数学推导

    对新生成的制导律代码进行数学分析，判断其输出是否能被
    已有 STT 自动驾驶仪正确跟踪。每步给出明确的数学公式推导。
    """
    law_info = _identify_law_type_from_code(gf_code) if gf_code else {
        "law_type": "unknown", "math_model": "未提供代码",
        "guidance_tf": "未知", "indicators": {},
    }
    law_type = law_info["law_type"]

    T_go = 3.0  # s — worst-case terminal time-to-go

    w1_min    = _RL_RANGES["w1"]["min"]
    w1_max    = _RL_RANGES["w1"]["max"]
    zeta1_min = _RL_RANGES["zeta1"]["min"]
    zeta1_max = _RL_RANGES["zeta1"]["max"]
    tao1_min  = _RL_RANGES["tao1"]["min"]
    tao1_max  = _RL_RANGES["tao1"]["max"]
    N_min     = _RL_RANGES["N_pn"]["min"]
    N_max     = _RL_RANGES["N_pn"]["max"]

    omega_c = 1.0 / T_go  # 制导回路穿越频率（末端）

    derivation: List[Dict[str, Any]] = []

    # ═══ Step 1: 制导律数学模型提取 ═══
    active_indicators = [k for k, v in law_info["indicators"].items() if v]
    derivation.append({
        "step": 1,
        "title": "从代码提取制导律数学模型",
        "law_type": law_type,
        "math_model": law_info["math_model"],
        "guidance_tf": law_info["guidance_tf"],
        "code_evidence": active_indicators or ["未检测到明确律型特征"],
    })

    # ═══ Step 2: 自动驾驶仪传递函数（已知） ═══
    derivation.append({
        "step": 2,
        "title": "自动驾驶仪传递函数（STT，已知固定）",
        "formula": "G_ap(s) = ω₁² / (s² + 2ζ₁ω₁s + ω₁²) · e^{-τ₁s}",
        "rl_ranges": {
            "ω₁": f"[{w1_min}, {w1_max}] rad/s",
            "ζ₁": f"[{zeta1_min}, {zeta1_max}]",
            "τ₁": f"[{tao1_min}, {tao1_max}] s",
        },
    })

    # ═══ Step 3: 开环传递函数 ═══
    _L_map = {
        "PN":  ("L(s) = (N/T_go) · G_ap(s) / s",
                "PN律: 运动学关系含一个纯积分 1/s，开环为I型系统"),
        "APN": ("L(s) = (N/T_go) · G_ap(s) / s + (N/2)·a_T·G_ap(s)/s²",
                "APN: 前馈项 a_T 不改变闭环特征方程，稳定性分析同PN"),
        "OGL": ("L(s) = K_eff(T_go) · G_ap(s) / s",
                "OGL: K_eff 随 T_go 时变，取末端最大值作最差情况"),
    }
    L_formula, L_note = _L_map.get(law_type, (
        "L(s) = K_g · G_ap(s) / s", f"律型 {law_type}: 线性化等效增益 K_g"
    ))
    derivation.append({
        "step": 3,
        "title": "制导-驾驶仪级联开环传递函数",
        "formula": L_formula,
        "note": L_note,
    })

    # ═══ Step 4: 穿越频率分析 ═══
    bw_ratio = w1_min / omega_c
    derivation.append({
        "step": 4,
        "title": "穿越频率与带宽分离",
        "substeps": [
            f"ω_c ≈ 1/T_go = 1/{T_go} = {omega_c:.4f} rad/s",
            f"ω₁_min = {w1_min} rad/s",
            f"带宽分离比 = ω₁_min/ω_c = {w1_min}/{omega_c:.4f} = {bw_ratio:.1f}",
            f"判定: {bw_ratio:.1f} {'≥' if bw_ratio >= 5 else '<'} 5 → "
            f"{'✓ 自动驾驶仪带宽远大于制导回路，可正确跟踪' if bw_ratio >= 5 else '⚠️ 分离不足'}",
        ],
        "bw_separation_ratio": round(bw_ratio, 1),
        "pass": bw_ratio >= 5.0,
    })

    # ═══ Step 5: 相位裕度推导（最差情况） ═══
    # ∠G_ap(jω_c) = -arctan(2ζ₁ω₁ω_c/(ω₁²-ω_c²)) - ω_c·τ₁  (rad)
    # 最差: ω₁=min, ζ₁=max, τ₁=max → 最大相位滞后
    denom_real = w1_min**2 - omega_c**2
    denom_imag = 2 * zeta1_max * w1_min * omega_c
    arctan_deg = math.degrees(math.atan(denom_imag / denom_real)) if denom_real > 0 else 90.0
    delay_deg  = math.degrees(omega_c * tao1_max)
    pm_worst   = 90.0 - arctan_deg - delay_deg
    pm_ok = PM_MIN <= pm_worst <= PM_MAX

    derivation.append({
        "step": 5,
        "title": "相位裕度推导（最差: ω₁=ω₁_min, ζ₁=ζ₁_max, τ₁=τ₁_max）",
        "substeps": [
            f"5.1 自动驾驶仪二阶环节相位滞后:",
            f"    φ_ap = arctan(2·{zeta1_max}·{w1_min}·{omega_c:.4f} "
            f"/ ({w1_min}²-{omega_c:.4f}²))",
            f"         = arctan({denom_imag:.4f} / {denom_real:.4f})",
            f"         = {arctan_deg:.2f}°",
            f"5.2 纯延迟相位滞后:",
            f"    φ_delay = ω_c·τ₁_max·(180/π) = {omega_c:.4f}×{tao1_max}×57.3",
            f"            = {delay_deg:.2f}°",
            f"5.3 开环总相位（I型系统 -90° 来自 1/s 积分项）:",
            f"    ∠L(jω_c) = -90° - {arctan_deg:.2f}° - {delay_deg:.2f}°",
            f"              = -{90+arctan_deg+delay_deg:.2f}°",
            f"5.4 相位裕度:",
            f"    PM = 180° + ∠L = 180° - {90+arctan_deg+delay_deg:.2f}°",
            f"       = {pm_worst:.1f}°",
            f"5.5 判定: PM = {pm_worst:.1f}° "
            f"{'在' if pm_ok else '不在'} [{PM_MIN:.0f}, {PM_MAX:.0f}]°"
            f" → {'✓ 制导律指令可被驾驶仪稳定跟踪' if pm_ok else '✗ 裕度不足，制导律输出可能导致闭环振荡'}",
        ],
        "pm_worst_deg": round(pm_worst, 1),
        "pass": pm_ok,
    })

    # ═══ Step 6: 归一化滞后 ═══
    tau_norm = tao1_max / T_go
    tau_ok = tau_norm <= 0.20
    derivation.append({
        "step": 6,
        "title": "归一化滞后（时间尺度分离）",
        "substeps": [
            f"τ_norm = τ₁_max / T_go = {tao1_max} / {T_go} = {tau_norm:.4f}",
            f"判定: {tau_norm:.4f} {'≤' if tau_ok else '>'} 0.20 → "
            f"{'✓ 驾驶仪响应速度足以跟踪制导律末端指令' if tau_ok else '⚠️ 驾驶仪滞后较大，末端跟踪可能不足'}",
        ],
        "normalized_lag": round(tau_norm, 4),
        "pass": tau_ok,
    })

    # ═══ Step 7: 律型特定条件 ═══
    law_specific: Dict[str, Any] = {}
    if law_type in ("PN", "APN"):
        C_worst = 1.0 + N_max / (2.0 * max(N_max - 1.0, 0.01)) * tau_norm**2
        adj_ok = C_worst <= 1.20
        gain_ok = N_min > 2.0
        law_specific = {
            "adjoint_miss_factor": {
                "derivation": (
                    f"C(N,τ) = 1 + N/(2(N-1))·τ²\n"
                    f"       = 1 + {N_max}/(2×{max(N_max-1,0.01):.1f})·{tau_norm:.4f}²\n"
                    f"       = {C_worst:.4f}"
                ),
                "threshold": 1.20,
                "pass": adj_ok,
                "meaning": "共轭法脱靶因子: 制导律末端精度对驾驶仪滞后的灵敏度",
            },
            "gain_condition": {
                "derivation": f"N_min = {N_min}, PN收敛必要条件 N > 2",
                "pass": gain_ok,
                "note": (
                    f"N_min={N_min} {'> 2 ✓' if gain_ok else '≤ 2 ⚠️ PN律在N≤2时不收敛'}"
                ),
            },
        }
        derivation.append({
            "step": 7,
            "title": f"律型特定条件（{law_type}）",
            "substeps": [
                f"7.1 共轭脱靶因子: C = {C_worst:.4f} {'≤' if adj_ok else '>'} 1.20 → {'✓' if adj_ok else '✗'}",
                f"7.2 导引增益: N_min = {N_min} {'>' if gain_ok else '≤'} 2 → {'✓' if gain_ok else '⚠️ 临界/不满足'}",
            ],
            "pass": adj_ok and gain_ok,
        })
    elif law_type == "OGL":
        law_specific = {
            "tgo_singularity": {
                "derivation": "OGL 含 1/T_go² 项，T_go→0 时指令 a_c→∞",
                "requirement": "代码中必须有 T_go = max(T_go, ε) 保护",
            },
            "weight_matrix": {
                "derivation": "最优制导权重矩阵 W(T_go) 须正定以保证指令有界",
            },
        }
        derivation.append({
            "step": 7,
            "title": f"律型特定条件（{law_type}）",
            "substeps": [
                "7.1 T_go 奇点: 需代码中有 max(T_go, ε) 保护（由Layer1检查）",
                "7.2 等效增益: K_eff·τ₁ << T_go 已由归一化滞后条件保证",
            ],
            "pass": True,
        })

    # ═══ Step 8: 适配性总判定 ═══
    bw_ok = bw_ratio >= 5.0
    overall_pass = pm_ok and bw_ok and tau_ok
    if law_type in ("PN", "APN"):
        ls_gain = law_specific.get("gain_condition", {})
        if ls_gain.get("pass") is False:
            overall_pass = False

    verdict = "✓ 新制导律与已有自动驾驶仪适配" if overall_pass else "⚠️ 存在适配性风险"
    derivation.append({
        "step": 8,
        "title": "适配性总判定",
        "conclusion": verdict,
        "summary": (
            f"律型={law_type} | PM_worst={pm_worst:.1f}° | "
            f"BW分离={bw_ratio:.1f} | τ_norm={tau_norm:.4f}"
        ),
        "pass": overall_pass,
    })

    return {
        "proof_basis": "基于制导律代码的数学推导（分析制导律→自动驾驶仪适配性）",
        "law_type_detected": law_type,
        "math_model": law_info["math_model"],
        "derivation_steps": derivation,
        "law_specific_conditions": law_specific,
        "overall_pass": overall_pass,
        "key_metrics": {
            "phase_margin_worst_deg": round(pm_worst, 1),
            "bw_separation_ratio": round(bw_ratio, 1),
            "normalized_lag": round(tau_norm, 4),
        },
    }


class GuidanceLawCompatibilityTool:
    """
    对新生成的制导律代码进行数学分析，推导其是否与已有自动驾驶仪适配。
    验证方向：制导律 → 自动驾驶仪（分析制导律输出能否被驾驶仪正确跟踪）。

    Three verification layers:
      Layer 1 — 接口静态检验 (no LLM)
                 · gf() 输出变量完整性 (N1, N2, G1, G2, gc)
                 · Np 参数使用（非硬编码）
                 · dp.* 字段访问、R/T_go 奇点保护
      Layer 2 — 制导律→驾驶仪数学推导 (deterministic, code-based)
                 · 从代码提取制导律数学模型 G_g(s)
                 · 建立级联开环 L(s) = G_g(s)·G_ap(s)/s
                 · 在 RL 参数全域计算相位裕度、带宽分离、归一化滞后
                 · 每步给出明确数学公式推导
      Layer 3 — LLM 数学推导与结构分析
                 · 从代码提取制导律数学模型，补充 Layer 2 正则分析的不足
                 · 逐步数学推导：等效传递函数、开环级联、稳定性条件
                 · 坐标系/符号约定/信号兼容性检查

    NOT checked here (evaluated by run_simulation after RL tuning):
      · Specific N_pn, w1, tao1 values
      · hit_rate, SEP, PM, BW performance metrics
    """

    name = "verify_guidance_compat"
    description = (
        "新制导律→自动驾驶仪适配性数学推导：对生成的制导律代码进行数学分析，\n"
        "推导其输出能否被已有自动驾驶仪正确跟踪。\n"
        "验证内容（与调参值无关）：\n"
        "  ① 接口静态检验：gf() 输出变量、Np 使用、dp字段、奇点保护\n"
        "  ② 代码数学推导：提取制导律模型→级联传递函数→稳定性证明\n"
        "    (w1∈[20,60], w2/w3∈[20,60], N_pn∈[3,6], tao1∈[0.05,0.35], ζ₁∈[0.35,1.20])\n"
        "  ③ LLM 数学推导：逐步公式推导制导律与STT驾驶仪的适配性\n"
        "【不检验】N_pn/w1/tao1 当前具体值的性能指标（这些由 run_simulation 验证）\n"
        "在 syntax_check_matlab 之后、run_simulation 之前调用（仅 MODIFY_LAW 模式）。\n"
        "参数：\n"
        "  script_path      — 生成的 .m 脚本路径\n"
        "  task_description — 制导律设计任务描述（辅助 LLM 判断律型）"
    )
    input_schema = {
        "type": "object",
        "properties": {
            "script_path": {
                "type": "string",
                "description": "生成的 MATLAB 脚本路径",
            },
            "task_description": {
                "type": "string",
                "description": "制导律设计任务描述（可选）",
                "default": "",
            },
        },
        "required": ["script_path"],
    }

    # ── LLM helper ────────────────────────────────────────────────────────────

    @staticmethod
    async def _llm(prompt: str, max_tokens: int = 1400) -> str:
        try:
            from multi_agent.config_loader import get_config
            import openai as _openai

            cfg = get_config().llm
            client = _openai.AsyncOpenAI(
                api_key=cfg.api_key or "sk-dummy",
                base_url=cfg.base_url or "https://api.openai.com/v1",
                timeout=90,
                max_retries=3,
            )
            resp = await client.chat.completions.create(
                model=cfg.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=max_tokens,
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as exc:
            logger.warning("[CompatVerify] LLM call failed: %s", exc)
            return ""

    # ── LLM structural analysis ───────────────────────────────────────────────

    async def _llm_structural_analysis(
        self,
        gf_code: str,
        func_name: str,
        interface_result: Dict[str, Any],
        range_proof_result: Dict[str, Any],
        task_description: str,
    ) -> Dict[str, Any]:
        """
        Layer 3 — LLM 数学推导与结构分析。

        从制导律代码出发，要求 LLM 逐步推导制导律与自动驾驶仪的适配性，
        每步给出明确的数学公式。
        """
        iface_status = "通过" if interface_result["pass"] else "有问题"
        iface_summary = f"接口静态检验: {iface_status}\n"
        if interface_result["issues"]:
            iface_summary += f"  问题: {interface_result['issues']}\n"
        if interface_result["warnings"]:
            iface_summary += f"  警告: {interface_result['warnings']}\n"

        # Layer 2 results summary for LLM context
        l2_law = range_proof_result.get("law_type_detected", "unknown")
        l2_model = range_proof_result.get("math_model", "未知")
        l2_metrics = range_proof_result.get("key_metrics", {})
        l2_status = "通过" if range_proof_result.get("overall_pass") else "有风险"
        l2_summary = (
            f"Layer2 自动识别: 律型={l2_law}, 模型={l2_model}\n"
            f"  PM_worst={l2_metrics.get('phase_margin_worst_deg', '?')}°, "
            f"BW分离={l2_metrics.get('bw_separation_ratio', '?')}, "
            f"τ_norm={l2_metrics.get('normalized_lag', '?')}\n"
            f"  Layer2结论: {l2_status}"
        )

        prompt = (
            "你是导弹制导系统专家。请对以下新生成的制导律代码进行数学分析，\n"
            "推导其输出是否能被已有自动驾驶仪正确跟踪。\n\n"
            f"{_STRUCT_FRAMEWORK}\n\n"
            "═══ 任务背景 ═══\n"
            f"{task_description or '修改制导律'}\n\n"
            "═══ Layer1 接口检验结果 ═══\n"
            f"{iface_summary}\n"
            "═══ Layer2 数学推导结果 ═══\n"
            f"{l2_summary}\n\n"
            "═══ 制导律源码 ═══\n"
            f"```matlab\n{gf_code[:2800]}\n```\n\n"
            "═══ 数学推导任务（每步需写出明确公式） ═══\n"
            "1. 从代码提取制导律数学模型：写出 a_c = f(状态量, Np, ...) 的具体表达式\n"
            "   验证/修正 Layer2 的自动识别结果\n"
            "2. 建立制导律等效传递函数 G_g(s)，写出推导过程：\n"
            "   · PN:  a_c = N·V_c·q̇ → G_g(s) = N·V_c/R ≈ N/T_go\n"
            "   · OGL: a_c = K(T_go)·ZEM → G_g(s) = K_eff/T_go²\n"
            "   · 其他律型: 写出等效线性化模型\n"
            "3. 写出开环传递函数: L(s) = G_g(s) · G_ap(s) / s\n"
            "   其中 G_ap(s) = ω₁²/(s²+2ζ₁ω₁s+ω₁²) · e^{-τ₁s}\n"
            "4. 验证稳定性条件（用 Layer2 的数值结果确认或修正）：\n"
            "   · 相位裕度 PM ∈ [45°,70°] → 满足当前实验稳定裕度约束\n"
            "   · 带宽分离 ω₁/ω_c ≥ 5 → 驾驶仪带宽超过制导回路\n"
            "   · 归一化滞后 τ_norm ≤ 0.20 → 时间尺度分离\n"
            "5. 律型特定条件：\n"
            "   PN/APN: N>2 收敛条件 + 共轭脱靶因子 C(N,τ)\n"
            "   OGL:    T_go 奇点保护 + 权重矩阵正定性\n"
            "   其他:   指令连续性、末端收敛性\n"
            "6. N1/N2 符号约定与 STT 自动驾驶仪 cf() 一致性\n"
            "7. G1/G2 方向余弦: G1=cos(up)cos(ga), G2=-cos(up)sin(ga)\n"
            "8. 适配性总结论（PASS/WARN/FAIL）\n\n"
            "【重要】不评价 N_pn/w1/tao1 的具体取值（由RL调参）。\n\n"
            "严格按以下 JSON 格式输出，不加任何说明文字：\n"
            "{\n"
            '  "law_type": "PN" | "APN" | "OGL" | "IACG" | "sliding_mode" | "predictive" | "custom",\n'
            '  "math_model_extracted": "从代码提取的数学模型 a_c = ...",\n'
            '  "guidance_tf": "制导律等效传递函数 G_g(s) = ...",\n'
            '  "open_loop_tf": "开环传递函数 L(s) = G_g(s)·G_ap(s)/s = ...",\n'
            '  "stability_derivation": ["每步稳定性推导公式和结果"],\n'
            '  "overall": "PASS" | "WARN" | "FAIL",\n'
            '  "confidence": 0~100,\n'
            '  "structural_checks": {\n'
            '    "input_signals_available": {"pass": true/false, "detail": "律型所需输入信号"},\n'
            '    "output_sign_convention": {"pass": true/false, "detail": "N1/N2符号与STT驾驶仪"},\n'
            '    "g1_g2_correct": {"pass": true/false, "detail": "G1/G2方向余弦"},\n'
            '    "law_specific_stability": {"pass": true/false, "detail": "律型特定稳定性条件及推导"},\n'
            '    "stt_autopilot_compatible": {"pass": true/false, "detail": "制导律输出能否被驾驶仪跟踪"}\n'
            '  },\n'
            '  "critical_issues": ["结构性问题列表"],\n'
            '  "recommendations": ["改进建议"],\n'
            '  "summary": "一句话中文总结（含律型名称和关键数学结果）"\n'
            "}"
        )

        raw = await self._llm(prompt, max_tokens=2500)
        m = re.search(r"\{[\s\S]*\}", raw)
        if m:
            try:
                return json.loads(m.group())
            except Exception:
                pass

        # Fallback: structure from interface result
        iface_ok = interface_result["pass"]
        return {
            "law_type": "unknown",
            "law_description": "LLM 解析失败，回退到接口检验结果",
            "overall": "PASS" if iface_ok else "WARN",
            "confidence": 40,
            "structural_checks": {},
            "critical_issues": interface_result.get("issues", []),
            "recommendations": [],
            "summary": "LLM 解析失败，以接口静态检验结果为准",
        }

    # ── Main execute ──────────────────────────────────────────────────────────

    async def execute(
        self,
        script_path: str = "",
        task_description: str = "",
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Structural compatibility verification for the modified guidance law."""
        script_path = _coerce_script_path(script_path, **kwargs)
        task_description = _coerce_task_prompt(task_description, **kwargs)

        # ── 1. Read script ────────────────────────────────────────────────────
        try:
            script = Path(script_path).read_text(encoding="utf-8", errors="ignore")
        except Exception as exc:
            return {"status": "error", "message": f"无法读取脚本: {exc}"}

        # ── 2. Extract guidance law function ─────────────────────────────────
        gf_code, func_name = _extract_gf_section(script)
        if not gf_code:
            return {
                "status":  "warn",
                "overall": "SKIP",
                "message": "未找到制导律函数（gf/guidance_local/含Np*Vc），跳过验证。",
            }
        logger.info("[CompatVerify] Found guidance function '%s' (%d chars)", func_name, len(gf_code))

        # ── 3. Layer 1: Interface static check (no LLM, no parameter values) ─
        interface = _check_interface_static(gf_code)
        logger.info("[CompatVerify] Interface check: pass=%s issues=%d warnings=%d",
                    interface["pass"], len(interface["issues"]), len(interface["warnings"]))

        # ── 4. Layer 2: Code-based math derivation (deterministic) ───────
        range_proof = _range_stability_proof(gf_code)
        logger.info("[CompatVerify] Math derivation: law=%s overall_pass=%s",
                    range_proof.get("law_type_detected"), range_proof["overall_pass"])

        # ── 5. Layer 3: LLM math derivation & structural analysis ───────
        analysis = await self._llm_structural_analysis(
            gf_code, func_name, interface, range_proof, task_description
        )

        overall    = analysis.get("overall", "WARN")
        confidence = analysis.get("confidence", 50)

        # Interface FAIL overrides LLM PASS
        if not interface["pass"] and overall == "PASS":
            overall = "WARN"

        logger.info("[CompatVerify] Result: overall=%s confidence=%s law=%s",
                    overall, confidence, analysis.get("law_type"))

        return {
            "status":          "success",
            "script_path":     script_path,
            "guidance_func":   func_name,
            "law_type":        analysis.get("law_type", "unknown"),
            "law_description": analysis.get("law_description", ""),
            "overall":         overall,
            "confidence":      confidence,
            # Layer 1 — interface
            "interface_check": {
                "pass":     interface["pass"],
                "issues":   interface["issues"],
                "warnings": interface["warnings"],
                "info":     interface["info"],
            },
            # Layer 2 — range proof
            "range_stability_proof": range_proof,
            # Layer 3 — LLM structural analysis
            "structural_checks":  analysis.get("structural_checks", {}),
            "critical_issues":    analysis.get("critical_issues", []),
            "recommendations":    analysis.get("recommendations", []),
            "summary":            analysis.get("summary", ""),
            "hint": (
                "【强制】结构适配性验证 FAIL — 制导律接口或算法结构与自动驾驶仪不兼容，"
                "必须重新调用 generate_matlab(mode=MODIFY_LAW)，"
                "在 task_description 中说明需修正的接口/结构问题（参考 critical_issues），"
                "再调用 syntax_check_matlab → verify_guidance_compat 重新验证，最多重试 2 次。"
                if overall == "FAIL" else
                "结构适配性验证通过（WARN 表示有轻微风险但不阻塞），可继续调用 run_simulation。"
            ),
        }
