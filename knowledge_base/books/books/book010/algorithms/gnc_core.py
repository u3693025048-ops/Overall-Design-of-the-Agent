#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gnc_core.py — 制导基础工具层

提供全局通用的向量/角度运算、视线（LOS）运动学、碰撞三角求解、
剩余时间估计与零控脱靶量（ZEM）计算。纯标准库，无第三方依赖。

符号约定（全库统一，务必遵守）：
    r      = 目标位置 - 导弹位置      （相对位置矢量，指向目标）
    rdot   = 目标速度 - 导弹速度      （相对速度矢量）
    lam    = atan2(r_y, r_x)          （视线角）
    lamdot = (r_x * rdot_y - r_y * rdot_x) / |r|^2      （视线角速率）
    Vc     = -(r · rdot) / |r|        （接近速度，相互靠近时为正）
    n_hat  = (-sin(lam), cos(lam))    （视线逆时针 90°，指令加速度正方向）

    ⚠️ 经典坑：lamdot 写成 (r_y*rdot_x - r_x*rdot_y)/r^2 会符号反，
       导弹会被目标"推开"，表现为脱靶量发散。本模块自检含符号验证。

自检：python3 gnc_core.py
"""
from math import atan2, cos, hypot, sin, sqrt

__all__ = [
    "vec_sub", "vec_add", "vec_scale", "vec_norm", "vec_dot", "vec_cross",
    "wrap_pi", "deg", "rad",
    "los", "los_rate", "closing_velocity", "los_unit", "los_normal",
    "tgo_estimate", "zem", "collision_heading", "cpa_miss",
]


# ---------------------------------------------------------------- 向量运算
def vec_sub(a, b):
    """a - b（逐分量）"""
    return (a[0] - b[0], a[1] - b[1])


def vec_add(a, b):
    return (a[0] + b[0], a[1] + b[1])


def vec_scale(a, s):
    return (a[0] * s, a[1] * s)


def vec_norm(a):
    return sqrt(a[0] * a[0] + a[1] * a[1])


def vec_dot(a, b):
    return a[0] * b[0] + a[1] * b[1]


def vec_cross(a, b):
    """二维叉积的 z 分量：a × b"""
    return a[0] * b[1] - a[1] * b[0]


# ---------------------------------------------------------------- 角度工具
def wrap_pi(x):
    """把角度归一到 (-π, π]"""
    while x > 3.141592653589793:
        x -= 6.283185307179586
    while x <= -3.141592653589793:
        x += 6.283185307179586
    return x


def deg(x):
    """弧度 → 度"""
    return x * 57.29577951308232


def rad(x):
    """度 → 弧度"""
    return x * 0.017453292519943295


# ---------------------------------------------------------------- LOS 运动学
def los(t_pos, m_pos):
    """视线角 λ = atan2(r_y, r_x)"""
    rx = t_pos[0] - m_pos[0]
    ry = t_pos[1] - m_pos[1]
    return atan2(ry, rx)


def los_rate(r, rdot):
    """视线角速率 λ̇ = (r × ṙ)_z / |r|²

    Args:
        r: 相对位置 (rx, ry)，指向目标
        rdot: 相对速度 (v_rx, v_ry)
    """
    rr = r[0] * r[0] + r[1] * r[1]
    if rr <= 0.0:
        return 0.0
    return (r[0] * rdot[1] - r[1] * rdot[0]) / rr


def closing_velocity(r, rdot):
    """接近速度 Vc = -(r · ṙ)/|r|，相互靠近时为正"""
    rn = vec_norm(r)
    if rn <= 0.0:
        return 0.0
    return -vec_dot(r, rdot) / rn


def los_unit(r):
    """视线单位矢量 ê_r"""
    rn = vec_norm(r)
    return (r[0] / rn, r[1] / rn) if rn > 0 else (1.0, 0.0)


def los_normal(r):
    """视线法向单位矢量 n̂ = ê_r 逆时针 90°（指令加速度正方向）"""
    e = los_unit(r)
    return (-e[1], e[0])


# ---------------------------------------------------------------- 时间与 ZEM
def tgo_estimate(r, rdot, a_t=0.0, lo=0.1, hi=10.0, iters=3):
    """剩余飞行时间 t_go 估计（含目标机动修正的隐式迭代）。

    t_go = R/Vc + 0.5 * a_T * t_go² / Vc   （迭代 2~3 次即可）

    经验（book010 卡 B-02）：t_go 是全系统最容易出问题的地方，
    必须限幅 [0.1, 10] s 并加一阶低通（τ≈0.05 s），否则目标机动时
    估计值会跳变（实测 2 s → 4 s）导致指令剧烈震荡。
    """
    R = vec_norm(r)
    Vc = closing_velocity(r, rdot)
    if Vc <= 1e-6:
        return hi
    t = R / Vc
    for _ in range(iters):
        t = (R + 0.5 * a_t * t * t) / Vc
    return max(lo, min(hi, t))


def zem(r, rdot, t_go):
    """零控脱靶量 ZEM（视线法向分量）。

    ZEM = (r + ṙ·t_go) 在视线法向的投影 ≡ Vc·t_go²·λ̇ （线性化等价）

    含义：从现在起不再施加任何控制，t_go 秒后导弹与目标的最近距离。
    最优制导律（OGL）即 a = N·ZEM/t_go²，N=3 时退化为 PN。
    """
    n = los_normal(r)
    zr = (r[0] + rdot[0] * t_go, r[1] + rdot[1] * t_go)
    return vec_dot(zr, n)


def collision_heading(t0, v_t, v_m_speed):
    """解析求解碰撞三角：给定目标状态与导弹速率，求导弹初速使 λ̇ ≡ 0。

    由 V_T - V_M = -Vc·ê_r 且 |V_M| = Vm 得二次方程：
        Vc² + 2(V_T·ê_r)·Vc + (|V_T|² - Vm²) = 0

    ⚠️ 经验（book010 仿真教训）：初速若直接对准目标当前位置（纯追踪），
    理想工况也会有 0.5 m 级"假脱靶"，掩盖真实规律。必须落在碰撞三角上。

    Args:
        t0: 目标初始位置 (x, y)（导弹在原点）
        v_t: 目标速度 (vx, vy)
        v_m_speed: 导弹速率（标量）
    Returns:
        (vmx, vmy, Vc)：导弹初速分量与接近速度
    """
    R = hypot(t0[0], t0[1])
    if R <= 0.0:
        raise ValueError("目标与导弹初始重合")
    ex, ey = t0[0] / R, t0[1] / R          # ê_r（由导弹指向目标）
    dot = v_t[0] * ex + v_t[1] * ey
    disc = dot * dot - (v_t[0] ** 2 + v_t[1] ** 2 - v_m_speed ** 2)
    if disc < 0.0:
        raise ValueError(
            f"速度比不足（Vm={v_m_speed} 需 > {sqrt(v_t[0]**2 + v_t[1]**2 - disc):.1f}），"
            "无碰撞三角解")
    Vc = -dot + sqrt(disc)
    return v_t[0] + Vc * ex, v_t[1] + Vc * ey, Vc


def cpa_miss(r, rdot):
    """最近接近点（CPA）横向脱靶量解析估计。

    miss = |r × ṙ| / |ṙ|，即匀速假设下相对运动的最近距离。
    比逐点搜索 min|r| 平滑得多，适合作为仿真早停时的脱靶量。
    """
    vn = vec_norm(rdot)
    if vn <= 1e-9:
        return vec_norm(r)
    return abs(vec_cross(r, rdot)) / vn


# ---------------------------------------------------------------- 自检
def _self_test():
    print("=" * 66)
    print("  gnc_core 自检")
    print("=" * 66)

    # 1. λ̇ 符号验证：目标在 +y 方向远离（上偏），视线应逆时针转（λ̇ > 0）
    r = (1000.0, 0.0)
    rdot = (0.0, 100.0)          # 相对速度朝 +y
    ld = los_rate(r, rdot)
    assert ld > 0, f"λ̇ 符号错误：应为正，实得 {ld}"
    print(f"  [1] λ̇ 符号            : {ld:+.6f} rad/s  "
          f"（目标上偏 ⇒ 视线逆时针，应为正）  ✅")

    # 2. 接近速度：相对速度朝向导弹 ⇒ Vc > 0
    Vc = closing_velocity((1000.0, 0.0), (-300.0, 0.0))
    assert Vc > 0
    print(f"  [2] 接近速度 Vc        : {Vc:+.2f} m/s  （靠近为正）  ✅")

    # 3. 法向单位矢量正交性
    n = los_normal((1000.0, 0.0))
    assert abs(vec_dot(n, (1.0, 0.0))) < 1e-12
    print(f"  [3] n̂ 正交性           : n̂=({n[0]:+.3f},{n[1]:+.3f})  ✅")

    # 4. 碰撞三角：解出的导弹初速应使 λ̇ ≡ 0
    t0 = (5200.0, 500.0)
    v_t = (-210.0, 60.0)
    vmx, vmy, Vc0 = collision_heading(t0, v_t, 700.0)
    ld0 = los_rate(t0, (v_t[0] - vmx, v_t[1] - vmy))
    assert abs(ld0) < 1e-9, f"碰撞三角未使 λ̇ 归零：{ld0}"
    assert abs(hypot(vmx, vmy) - 700.0) < 1e-9
    print(f"  [4] 碰撞三角           : V_M=({vmx:.1f},{vmy:.1f}) "
          f"|V_M|={hypot(vmx, vmy):.3f}  Vc={Vc0:.1f}  λ̇={ld0:+.2e}  ✅")

    # 5. ZEM 两种算法一致性（几何投影 vs Vc·t_go²·λ̇ 线性化式）
    r5 = (3000.0, 250.0)
    rdot5 = (-600.0, -25.0)
    tgo = tgo_estimate(r5, rdot5)
    z_geo = zem(r5, rdot5, tgo)
    z_lin = closing_velocity(r5, rdot5) * tgo * tgo * los_rate(r5, rdot5)
    err = abs(z_geo - z_lin)
    assert err < 1e-6, f"ZEM 两算法不一致：{z_geo} vs {z_lin}"
    print(f"  [5] ZEM 一致性         : 几何 {z_geo:+.4f} m / 线性化 "
          f"{z_lin:+.4f} m  差 {err:.2e}  ✅")

    # 6. CPA 解析脱靶量 vs 数值搜索
    r6, rdot6 = (2000.0, 150.0), (-500.0, -40.0)
    best = min(vec_norm((r6[0] + rdot6[0] * t, r6[1] + rdot6[1] * t))
               for t in [i * 0.0005 for i in range(10000)])
    ana = cpa_miss(r6, rdot6)
    assert abs(ana - best) < 1e-3, f"CPA 解析解偏差过大：{ana} vs {best}"
    print(f"  [6] CPA 脱靶量         : 解析 {ana:.6f} m / 数值搜索 "
          f"{best:.6f} m  ✅")

    # 7. t_go 限幅
    assert tgo_estimate((1.0, 0.0), (-1e-9, 0.0)) == 10.0     # 上溢截断
    assert tgo_estimate((1e6, 0.0), (-1e6, 0.0)) <= 10.0
    print("  [7] t_go 限幅 [0.1,10] : ✅")

    # 8. wrap_pi
    for v in (7.0, -7.0, 0.0, 3.2):
        assert -3.1415927 <= wrap_pi(v) <= 3.1415927
    print("  [8] wrap_pi 归一化     : ✅")

    print("=" * 66)
    print("  gnc_core：8 项自检全部通过")
    print("=" * 66)


if __name__ == "__main__":
    _self_test()
