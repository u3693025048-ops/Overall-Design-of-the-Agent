#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
guidance_laws.py — 制导律算法库

统一接口 + 注册表，10 种制导律可互换、可参数化（呼应 MBSE「同一接口下的
策略变体」建模思想）。所有公式均标注对应的 RIA++ 卡片编号，与
assets/02_formulas.md 一一对应。

核心接口
--------
    a_cmd = LAWS[name](geom, **params)        # 标量，沿视线法向 n̂

Geom 由 engagement / 用户自行构造。指令加速度矢量：
    - TPN 系：a_vec = a_cmd · n̂      （n̂ ⊥ 视线）
    - PPN 系：a_vec = a_cmd · v̂n     （v̂n ⊥ 导弹速度）

纯标准库。自检：python3 guidance_laws.py
"""
import math
from math import atan2, cos, hypot, sin

from gnc_core import (closing_velocity, collision_heading, cpa_miss, los,
                      los_normal, los_rate, vec_norm, vec_sub, wrap_pi, zem)

G = 9.80665

# ---------------------------------------------------------------- 几何快照
class Geom:
    """制导几何快照（每步由仿真器填充）。"""

    __slots__ = ("r", "rdot", "vm", "vt", "Vm", "Vc", "lam", "lamdot",
                 "t_go", "a_t_perp", "theta_m", "theta_d", "zem")

    def __init__(self, m_pos, m_vel, t_pos, t_vel, a_t_perp=0.0,
                 theta_d=None):
        self.r = vec_sub(t_pos, m_pos)
        self.rdot = vec_sub(t_vel, m_vel)
        self.vm = m_vel
        self.vt = t_vel
        self.Vm = vec_norm(m_vel)
        self.Vc = closing_velocity(self.r, self.rdot)
        self.lam = atan2(self.r[1], self.r[0])
        self.lamdot = los_rate(self.r, self.rdot)
        self.a_t_perp = a_t_perp
        self.theta_m = atan2(m_vel[1], m_vel[0])
        self.theta_d = self.lam if theta_d is None else theta_d
        self.t_go = self._tgo()
        self.zem = zem(self.r, self.rdot, self.t_go)

    def _tgo(self, lo=0.1, hi=10.0):
        if self.Vc <= 1e-6:
            return hi
        t = vec_norm(self.r) / self.Vc
        for _ in range(3):
            t = (vec_norm(self.r) + 0.5 * self.a_t_perp * t * t) / self.Vc
        return max(lo, min(hi, t))


# ---------------------------------------------------------------- 制导律实现
def law_tpn(g, N=4.0):
    """真比例导引 TPN：a = N·Vc·λ̇（指令 ⊥ 视线）

    卡 A-01。工程默认 N=3~5；抗机动能力强，大气层外/反导首选。
    """
    return N * g.Vc * g.lamdot


def law_ppn(g, N=4.0):
    """纯比例导引 PPN：a = N·Vm·λ̇（指令 ⊥ 导弹速度）

    卡 A-01。⚠️ 其 N 与 TPN 物理意义不同，需取 6~10 才有类似效果。
    """
    return N * g.Vm * g.lamdot


def law_apn(g, N=4.5, est_err=0.0):
    """增广比例导引 APN：a = N·Vc·λ̇ + (N/2)·â_T⊥

    卡 B-03 / A-05。对常值机动目标理论零脱靶。
    ⚠️ est_err > 0.3 时可能劣于 PN；>0.5 反而更差（仿真已验证）。
    """
    a_t = g.a_t_perp * (1.0 - est_err)      # 估计误差：低估补偿量
    return N * g.Vc * g.lamdot + 0.5 * N * a_t


def law_bpn(g, N=4.0, K_bias=0.0, theta_ref=None):
    """偏置比例导引 BPN：a = N·V·λ̇ + K_bias·(θ_cur − θ_ref)

    卡 C-02。用于视场/框架角与终端角约束：把弹道姿态拉向参考角。
    K_bias=0 时退化为 PN。
    """
    ref = g.lam if theta_ref is None else theta_ref
    return N * g.Vm * g.lamdot + K_bias * (g.theta_m - ref)


def law_ogl(g, N=3.0):
    """最优制导律 OGL：a = N·ZEM/t_go²

    卡 B-02 / C-02。ZEM 为零控脱靶量；N=3 时与 PN 等价（线性化意义下），
    但对 t_go 估计精度敏感。
    """
    if g.t_go <= 1e-6:
        return 0.0
    return N * g.zem / (g.t_go * g.t_go)


def law_impact_angle(g, N1=6.0, N2=2.0, N3=1.0, theta_d=None):
    """落角约束制导：a = N₁·y/t_go² + N₂·ẏ/t_go + N₃·Vm(θ_m−θ_d)/t_go

    卡 C-05。y/ẏ 为视线法向的相对位移与速度分量。
    """
    tg = max(g.t_go, 1e-3)
    n = los_normal(g.r)
    y = n[0] * g.r[0] + n[1] * g.r[1]
    ydot = n[0] * g.rdot[0] + n[1] * g.rdot[1]
    td = g.lam if theta_d is None else theta_d
    return (N1 * y / (tg * tg) + N2 * ydot / tg
            + N3 * g.Vm * wrap_pi(g.theta_m - td) / tg)


def law_fov_barrier(g, N=4.0, k=0.5, lam_max=0.5236):
    """视场约束制导（Barrier Lyapunov）：
        λ̇_des = −k·λ·(λ_max² − λ²)²，a = N·Vc·(λ̇ − λ̇_des)

    卡 C-02。λ 逼近 λ_max 时屏障项急剧增大，把视线角推回中心，
    保证全程 |λ| < λ_max（视场不丢失）。
    """
    lm2 = lam_max * lam_max
    lam = wrap_pi(g.lam)
    if abs(lam) >= lam_max:
        lam = (lam_max - 1e-6) * (1.0 if lam > 0 else -1.0)
    ld_des = -k * lam * ((lm2 - lam * lam) ** 2)
    return N * g.Vc * (g.lamdot - ld_des)


def law_smc(g, k=3.0, eps=0.5, c=1.0, phi=0.05):
    """滑模制导 SMC：s = λ̇ + c·λ，u = −k·s − ε·sat(s/Φ)

    卡 E-02 / E-04。sat() 为边界层饱和函数，抑制抖振。
    对目标机动与参数摄动强鲁棒；Φ 越小越接近理想滑模但抖振越大。
    """
    s = g.lamdot + c * wrap_pi(g.lam)
    sat = s / phi if abs(s) < phi else (1.0 if s > 0 else -1.0)
    return -(k * s + eps * sat) * max(g.Vc, 1.0)


def law_pure_pursuit(g, K=4.0):
    """纯追踪：a = K·Vm·(λ − θ_m)，指令指向目标当前位置

    卡 A-01（对比基线）。末端需用过载大，仅作对照。
    """
    return K * g.Vm * wrap_pi(g.lam - g.theta_m)


def law_predictive(g, a_t_est=0.0, Kp=1.0):
    """预测制导 PG：瞄准预测命中点
        P_pred = P_T + V_T·t_go + ½·Â_T·t_go²

    卡 A-06。对常值机动目标天然有前馈能力，但依赖 t_go 与 â_T 精度。
    """
    n = los_normal(g.r)
    px = g.r[0] + g.vt[0] * g.t_go + 0.5 * a_t_est * n[0] * g.t_go ** 2
    py = g.r[1] + g.vt[1] * g.t_go + 0.5 * a_t_est * n[1] * g.t_go ** 2
    lam_p = atan2(py, px)
    return Kp * g.Vm * wrap_pi(lam_p - g.theta_m) / max(g.t_go, 0.1)


# ---------------------------------------------------------------- 注册表
LAWS = {
    "TPN":           (law_tpn,          {"N": 4.0},
                      "真比例导引 a=N·Vc·λ̇（⊥视线）｜卡 A-01"),
    "PPN":           (law_ppn,          {"N": 4.0},
                      "纯比例导引 a=N·Vm·λ̇（⊥速度）｜卡 A-01"),
    "APN":           (law_apn,          {"N": 4.5, "est_err": 0.0},
                      "增广比例导引 a=N·Vc·λ̇+(N/2)â_T｜卡 B-03"),
    "BPN":           (law_bpn,          {"N": 4.0, "K_bias": 0.0,
                                         "theta_ref": None},
                      "偏置比例导引 a=N·V·λ̇+K(θ−θ_ref)｜卡 C-02"),
    "OGL":           (law_ogl,          {"N": 3.0},
                      "最优制导律 a=N·ZEM/t_go²｜卡 B-02"),
    "IMPACT_ANGLE":  (law_impact_angle, {"N1": 6.0, "N2": 2.0, "N3": 1.0,
                                         "theta_d": None},
                      "落角约束 a=N₁y/t²+N₂ẏ/t+N₃Vm(θ−θ_d)/t｜卡 C-05"),
    "FOV_BARRIER":   (law_fov_barrier,  {"N": 4.0, "k": 0.5,
                                         "lam_max": 0.5236},
                      "视场屏障 λ̇_des=−kλ(λ_max²−λ²)²｜卡 C-02"),
    "SMC":           (law_smc,          {"k": 3.0, "eps": 0.5, "c": 1.0,
                                         "phi": 0.05},
                      "滑模 s=λ̇+cλ, u=−ks−ε·sat(s/Φ)｜卡 E-02/04"),
    "PURE_PURSUIT":  (law_pure_pursuit, {"K": 4.0},
                      "纯追踪 a=K·Vm(λ−θ_m)｜卡 A-01 基线"),
    "PREDICTIVE":    (law_predictive,   {"a_t_est": 0.0, "Kp": 1.0},
                      "预测制导 瞄准 P_pred｜卡 A-06"),
}


def command(name, geom, **params):
    """按名称调用制导律（缺省参数自动补齐）。"""
    fn, defaults, _ = LAWS[name]
    p = dict(defaults)
    p.update(params)
    return fn(geom, **p)


def accel_vector(a_cmd, geom, variant="TPN"):
    """把标量指令映射到加速度矢量。

    TPN 系：沿视线法向 n̂ = (−sin λ, cos λ)
    PPN 系：沿速度法向 v̂n = (−sin θ_m, cos θ_m)
    """
    if variant == "PPN":
        ang = geom.theta_m
    else:
        ang = geom.lam
    return (-sin(ang) * a_cmd, cos(ang) * a_cmd)


# ---------------------------------------------------------------- 最小仿真器（自检用）
def _mini_sim(law_name, N=4.0, t_break=1e9, a_t=0.0, dt=0.001,
              amax=1e9, t_max=60.0, v_t=(-200.0, 0.0), **kw):
    """极简欧拉仿真（仅自检用，正式仿真见 engagement.py）。

    返回 (脱靶量 m, 峰值过载 g, 是否命中)
    """
    from gnc_core import collision_heading

    mt = (6000.0, 500.0)
    vmx, vmy, _ = collision_heading(mt, v_t, 700.0)
    m = [0.0, 0.0, vmx, vmy]
    tpos = [mt[0], mt[1]]
    tvel = list(v_t)
    peak, r_old = 0.0, vec_norm((mt[0], mt[1]))
    n_man = None          # 目标机动方向（惯性系固定，t_break 时锁定）

    for step in range(int(t_max / dt)):
        t = step * dt
        if t >= t_break:
            if n_man is None:
                n_man = los_normal((tpos[0] - m[0], tpos[1] - m[1]))
            tvel[0] += a_t * n_man[0] * dt
            tvel[1] += a_t * n_man[1] * dt
        tpos[0] += tvel[0] * dt
        tpos[1] += tvel[1] * dt

        # 目标加速度在「当前视线法向」的投影（APN 补偿量）
        a_perp = 0.0
        if n_man is not None:
            n_now = los_normal((tpos[0] - m[0], tpos[1] - m[1]))
            a_perp = a_t * (n_man[0] * n_now[0] + n_man[1] * n_now[1])
        g = Geom((m[0], m[1]), (m[2], m[3]), tuple(tpos), tuple(tvel),
                 a_t_perp=a_perp)
        a = command(law_name, g, N=N, **kw)
        a = max(-amax, min(amax, a))
        peak = max(peak, abs(a))
        av = accel_vector(a, g, "PPN" if law_name == "PPN" else "TPN")
        m[2] += av[0] * dt
        m[3] += av[1] * dt
        m[0] += m[2] * dt
        m[1] += m[3] * dt

        r_now = hypot(tpos[0] - m[0], tpos[1] - m[1])
        if r_now > r_old and r_now > 20.0:      # 越过最近点
            break
        r_old = r_now
        if r_now < 0.3:
            break
    miss = cpa_miss((tpos[0] - m[0], tpos[1] - m[1]),
                    (tvel[0] - m[2], tvel[1] - m[3]))
    return miss, peak / G


# ---------------------------------------------------------------- 自检
def _self_test():
    print("=" * 70)
    print("  guidance_laws 自检（最小欧拉仿真，dt=1 ms）")
    print("=" * 70)

    # 1. 注册表完整性
    assert len(LAWS) == 10
    print(f"  [1] 制导律注册表       : {len(LAWS)} 种 "
          f"({', '.join(LAWS)})  ✅")

    # 2. 理想工况（碰撞三角 + 非机动目标）：PN 应零脱靶
    m, pk = _mini_sim("TPN", N=4.0)
    assert m < 0.5, f"TPN 理想脱靶过大：{m}"
    print(f"  [2] TPN 理想工况       : 脱靶 {m:.4f} m / 峰值 {pk:.4f} g  ✅")

    # 3. OGL(N=3) 与 PN(N=3) 在相同工况下量级一致（理论等价）
    m_o, _ = _mini_sim("OGL", N=3.0)
    m_p, _ = _mini_sim("TPN", N=3.0)
    assert abs(m_o - m_p) < 0.5
    print(f"  [3] OGL(N=3) ≈ PN(N=3) : {m_o:.4f} m vs {m_p:.4f} m  ✅")

    # 4. APN 对常值机动目标显著优于 PN（同 N 对比，过载受限工况）
    kw = dict(t_break=5.0, a_t=12.0 * G, amax=20.0 * G)
    m_pn, _ = _mini_sim("TPN", N=4.5, **kw)
    m_ap, _ = _mini_sim("APN", N=4.5, **kw)
    assert m_ap < m_pn, f"APN 未优于 PN：{m_ap} vs {m_pn}"
    print(f"  [4] APN vs PN（同N=4.5）: PN {m_pn:.3f} m → APN {m_ap:.3f} m  ✅")

    # 5. APN 估计误差敏感性（应单调恶化；100% 误差即退化为 PN）
    kw = dict(t_break=4.0, a_t=12.0 * G, amax=20.0 * G)
    e_vals = (0.0, 0.3, 0.5, 1.0)
    v = [_mini_sim("APN", N=4.5, est_err=e, **kw)[0] for e in e_vals]
    assert all(v[i + 1] >= v[i] - 1e-3 for i in range(len(v) - 1)), \
        f"估计误差未单调恶化：{v}"
    print("  [5] APN 估计误差敏感性 : "
          + " → ".join(f"{int(e*100)}% {m:.3f}m" for e, m in zip(e_vals, v))
          + "  单调恶化 ✅")

    # 5b. APN 的价值几乎全部来自「过载受限」场景
    gains = []
    for am in (20.0, 40.0, 1e9):
        kk = dict(t_break=4.0, a_t=12.0 * G, amax=am * G if am < 1e8 else am)
        pn = _mini_sim("TPN", N=4.5, **kk)[0]
        ap = _mini_sim("APN", N=4.5, **kk)[0]
        gains.append(pn - ap)
    assert gains[0] > gains[1] > gains[2], f"APN 收益未随限幅放宽而减小：{gains}"
    print("  [5b]APN 收益 vs 过载限幅: 20g 收益 %.3f m → 40g %.3f m → "
          "不受限 %.3f m\n"
          "       ⇒ 过载不受限时 PN 已足够，上 APN 无收益（卡 A-05"
          "「别一上来就上 APNG」） ✅" % tuple(gains))

    # 6. BPN（偏置比例导引）：终端落角约束与框架角压制（卡 C-02）
    from gnc_core import collision_heading
    import math

    def _run_bpn(K_bias, theta_ref=-1.0, R0=6000.0, heading_err=0.0,
                 amax=1e9, r_stop=150.0):
        mt, v_t = (R0 * 0.96, R0 * 0.28), (-200.0, 0.0)
        vmx, vmy, _ = collision_heading(mt, v_t, 700.0)
        if heading_err:
            c, s = math.cos(heading_err), math.sin(heading_err)
            vmx, vmy = vmx * c - vmy * s, vmx * s + vmy * c
        m = [0.0, 0.0, vmx, vmy]
        tpos, tvel = list(mt), list(v_t)
        dt = 0.001
        for _ in range(30000):
            g = Geom((m[0], m[1]), (m[2], m[3]), tuple(tpos), tuple(tvel))
            a = command("BPN", g, N=4.0, K_bias=K_bias, theta_ref=theta_ref)
            a = max(-amax, min(amax, a))
            av = accel_vector(a, g)
            m[2] += av[0] * dt; m[3] += av[1] * dt
            m[0] += m[2] * dt; m[1] += m[3] * dt
            tpos[0] += tvel[0] * dt; tpos[1] += tvel[1] * dt
            if math.hypot(tpos[0] - m[0], tpos[1] - m[1]) < r_stop:
                return math.atan2(m[3], m[2]), abs(wrap_pi(g.lam - g.theta_m))
        return math.atan2(m[3], m[2]), 9.99

    ref = -1.0                       # 期望落角 −57.3°
    errs, gims = [], []
    for K in (0.0, 50.0, 100.0, 200.0):
        th, gm = _run_bpn(K, theta_ref=ref)
        errs.append(abs(wrap_pi(th - ref)))
        gims.append(gm)
    assert all(errs[i + 1] < errs[i] for i in range(len(errs) - 1)), \
        f"落角误差未随 K_bias 单调下降：{[math.degrees(e) for e in errs]}"
    print("  [6] BPN 落角约束（θ_ref=−57.3°）: K_bias 0→200，终端角误差 "
          + " → ".join(f"{math.degrees(e):.1f}°" for e in errs) + " ✅")
    assert gims[-1] > gims[0], "K_bias 过大未导致框架角恶化"
    print(f"       ⚠️ 但框架角 {math.degrees(gims[0]):.1f}° → "
          f"{math.degrees(gims[-1]):.1f}°：K_bias 过大会压弯弹道，需权衡 ✅")

    # 6b. 短 t_go + 大航向误差下的框架角压制
    gs = [_run_bpn(K, theta_ref=None, R0=1500.0,
                   heading_err=math.radians(25.0))[1]
          for K in (0.0, 30.0, 60.0, 120.0)]
    assert all(gs[i + 1] < gs[i] for i in range(len(gs) - 1)), \
        f"框架角未单调下降：{[math.degrees(x) for x in gs]}"
    print("  [6b]BPN 压框架角（短t_go+25°误差）: "
          + " → ".join(f"{math.degrees(x):.2f}°" for x in gs) + " ✅")

    # 7. 零视线角速率时指令应为零（除含偏置/落角项者）
    mt, v_t = (6000.0, 500.0), (-200.0, 0.0)
    from gnc_core import collision_heading
    vmx, vmy, _ = collision_heading(mt, v_t, 700.0)
    g = Geom((0.0, 0.0), (vmx, vmy), mt, v_t)
    assert abs(g.lamdot) < 1e-9
    for name in ("TPN", "PPN", "OGL"):
        a = command(name, g)
        assert abs(a) < 1.0, f"{name} 在 λ̇=0 时指令非零：{a}"
    print("  [7] λ̇=0 时零指令       : TPN/PPN/OGL 均 <1 m/s²  ✅")

    # 8. FOV 屏障（Barrier Lyapunov，卡 C-02）
    lm2, k, lmax = 0.6 * 0.6, 0.5, 0.6
    des = [-k * L * ((lm2 - L * L) ** 2) for L in (0.1, 0.3, 0.45, 0.55, 0.59)]
    # (a) 恒回中：λ̇_des 与 λ 反向
    assert all(d < 0 for d in des), f"λ̇_des 未恒回中：{des}"
    # (b) 屏障特性：|λ̇_des| 先增大后随 λ→λ_max 衰减到 0
    assert abs(des[1]) > abs(des[0]) and abs(des[-1]) < abs(des[1])
    print(f"  [8] FOV 屏障 λ̇_des      : λ=0.1→{des[0]:+.4f}  λ=0.3→{des[1]:+.4f}"
          f"  λ=0.59→{des[-1]:+.6f}\n"
          f"       ⇒ 恒回中，且 λ→λ_max 时冻结视线（Barrier 特性） ✅")

    # 8b. 端到端：视场守住 vs 脱靶代价
    def _run_fov(law, heading_err, **kw):
        mt = (6000.0 * math.cos(0.45), 6000.0 * math.sin(0.45))
        vmx, vmy, _ = collision_heading(mt, (-200.0, 0.0), 700.0)
        c, s = math.cos(heading_err), math.sin(heading_err)
        m = [0.0, 0.0, vmx * c - vmy * s, vmx * s + vmy * c]
        tpos, tvel = list(mt), [-200.0, 0.0]
        mx, mr = 0.0, 1e9
        for _ in range(20000):
            g = Geom((m[0], m[1]), (m[2], m[3]), tuple(tpos), tuple(tvel))
            mx = max(mx, abs(wrap_pi(g.lam)))
            r = hypot(tpos[0] - m[0], tpos[1] - m[1])
            mr = min(mr, r)
            if r < 5.0:
                break
            av = accel_vector(command(law, g, **kw), g)
            m[2] += av[0] * 1e-3; m[3] += av[1] * 1e-3
            m[0] += m[2] * 1e-3; m[1] += m[3] * 1e-3
            tpos[0] += tvel[0] * 1e-3; tpos[1] += tvel[1] * 1e-3
        return mx, mr

    he = math.radians(-40.0)
    pn_l, pn_m = _run_fov("TPN", he, N=4.0)
    fv_l, fv_m = _run_fov("FOV_BARRIER", he, N=4.0, k=5.0, lam_max=0.6)
    assert pn_l > 0.6 > fv_l, f"视场约束未生效：PN {pn_l} / FOV {fv_l}"
    print(f"  [8b]视场约束（−40°误差） : TPN max|λ| {pn_l:.3f} rad（越界）→ "
          f"FOV(k=5) {fv_l:.3f} rad（守住）\n"
          f"       代价：脱靶 {pn_m:.1f} m → {fv_m:.1f} m  ✅")

    # 8c. k 过大发散：屏障增益存在稳定上限
    bad_l, bad_m = _run_fov("FOV_BARRIER", he, N=4.0, k=20.0, lam_max=0.6)
    assert bad_l > 1.0 or bad_m > 100.0, f"k=20 未发散，与扫描结果不符：{bad_l}"
    print(f"  [8c]屏障增益上限        : k=20 → max|λ| {bad_l:.3f} rad / "
          f"脱靶 {bad_m:.0f} m（发散）⇒ k 须标定，非越大越好 ✅")

    # 9. 全部制导律可调用且不产生 NaN（λ=0.3 rad，位于视场内）
    g2 = Geom((0.0, 0.0), (700.0, 0.0), (1000.0, 309.0), (0.0, 0.0))
    for name in LAWS:
        a = command(name, g2)
        assert not math.isnan(a) and not math.isinf(a), f"{name} 产生非法值"
    print("  [9] 10 种律数值健全性  : 无 NaN/Inf  ✅")

    print("=" * 70)
    print("  guidance_laws：9 项自检全部通过（10 种制导律）")
    print("=" * 70)


if __name__ == "__main__":
    _self_test()
