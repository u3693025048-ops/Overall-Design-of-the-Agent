#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
engagement.py — 平面交战仿真引擎

把 gnc_core（运动学）+ guidance_laws（制导律）+ seeker（导引头/噪声）
组装成完整闭环，支持目标机动、舵机延迟、过载限幅、视场约束与蒙特卡洛。

设计要点（均为 book010 仿真踩坑后固化的经验）
--------------------------------------------
1. **初速按碰撞三角解析设定**（collision_heading），否则理想工况也存在
   0.5 m 级「假脱靶」，会掩盖真实规律。
2. **目标机动必须进入目标运动学**（分段二次解析积分），
   只把机动喂给 APN 补偿项而不改变目标轨迹，是致命 bug。
3. **机动方向在惯性系锁定**（t_break 时刻的视线法向），
   若每步沿当前法向，目标会做螺旋机动，与 APN 的常值机动假设不符。
4. **框架角只在 r > 150 m 的制导段统计**，末端尖峰无物理意义。
5. **早停用 CPA 解析解**（|r×ṙ|/|ṙ|），比逐点搜索 min|r| 平滑。
6. 舵机一阶延迟用**解析离散解** a ← a_cmd + (a − a_cmd)·e^(−dt/τ)。

纯标准库。自检：python3 engagement.py
"""
import math
import random

import guidance_laws as gl
import seeker
from gnc_core import (collision_heading, cpa_miss, los, los_normal,
                      los_rate, vec_norm, vec_sub, wrap_pi)

G = 9.80665
GIMBAL_RMIN = 150.0          # 框架角统计的最近距离门限（m）

__all__ = ["G", "simulate", "monte_carlo", "default_scenario"]


def default_scenario():
    """book010 标定基准工况：末段突然机动（最坏工况，具有鉴别力）。"""
    return dict(R0=6000.0, H0=500.0, Vm=700.0, Vt=300.0,
                t_break=5.0, tgt_g=9.0, amax=25.0 * G)


def simulate(N=4.0, law="TPN", R0=6000.0, H0=500.0, Vm=700.0, Vt=300.0,
             t_break=1e9, tgt_g=0.0, amax=25.0 * G, delay=0.0,
             sig_lam=0.0, tau_f=0.05, rate_noise=0.0, noise_tau=0.02,
             fov_half=math.radians(30.0), heading_err=0.0,
             dt=0.002, t_max=40.0, r_stop=0.5, seed=None, record=False,
             **law_kw):
    """单次交战仿真。

    Args:
        N: 导航比（同时传给制导律）
        law: 制导律名称（见 guidance_laws.LAWS）
        R0, H0: 目标初始位置（导弹在原点）
        Vm, Vt: 导弹/目标速率（目标沿 −x 飞行）
        t_break: 目标开始机动的时刻（s）
        tgt_g: 目标机动过载（g）
        amax: 导弹过载限幅（m/s²）
        delay: 舵机一阶延迟时间常数（s）
        sig_lam: 测角噪声标准差（rad）— **推荐使用的噪声模型**
        tau_f: λ̇ 滤波微分时间常数（s）
        rate_noise: 速率噪声强度（rad/s）— 仅作对比，会得出相反结论
        heading_err: 初始航向误差（rad）
        dt: 积分步长（s）
        record: 是否记录轨迹（用于绘图）
    Returns:
        dict: miss / peak_g / gimbal_max / gimbal_term / lost / t_end /
              r_min / traj
    """
    rng = random.Random(seed if seed is not None else 0)
    T0 = (R0, H0)
    V_T = (-Vt, 0.0)
    aT = tgt_g * G

    # --- 初速：碰撞三角 + 可选的初始航向误差 ---
    vmx, vmy, _ = collision_heading(T0, V_T, Vm)
    if heading_err:
        c, s = math.cos(heading_err), math.sin(heading_err)
        vmx, vmy = vmx * c - vmy * s, vmx * s + vmy * c
    m = [0.0, 0.0, vmx, vmy]           # mx, my, vmx, vmy

    # --- 目标运动学：分段解析（机动前直飞，机动后等加速）---
    bx, by = T0[0] + V_T[0] * t_break, T0[1] + V_T[1] * t_break
    man_dir = None                      # 机动方向（t_break 时锁定，惯性系）

    def tgt_state(t):
        if t <= t_break:
            return (T0[0] + V_T[0] * t, T0[1] + V_T[1] * t,
                    V_T[0], V_T[1])
        s = t - t_break
        ex, ey = man_dir
        return (bx + V_T[0] * s + 0.5 * aT * ex * s * s,
                by + V_T[1] * s + 0.5 * aT * ey * s * s,
                V_T[0] + aT * ex * s, V_T[1] + aT * ey * s)

    sk = seeker.Seeker(sigma_lam=sig_lam, tau_f=tau_f,
                       rate_noise=rate_noise, noise_tau=noise_tau,
                       fov_half=fov_half, seed=seed)
    a_lag = 0.0                         # 舵机输出（一阶延迟状态）
    peak_g = 0.0
    gimbal_term = None
    r_min = 1e18
    traj = [] if record else None

    n_steps = int(t_max / dt)
    for step in range(n_steps):
        t = step * dt
        # 机动方向必须在 tgt_state 之前锁定：t_break 可能落在两个积分步之间
        # （蒙特卡洛随机采样时必然发生），否则会用到未初始化的 man_dir。
        if t >= t_break and man_dir is None:
            man_dir = los_normal((bx - m[0], by - m[1]))
        tx, ty, vtx, vty = tgt_state(t)
        r = (tx - m[0], ty - m[1])
        rn = vec_norm(r)
        r_min = min(r_min, rn)

        geom = gl.Geom((m[0], m[1]), (m[2], m[3]), (tx, ty), (vtx, vty))
        # 目标加速度在视线法向的投影（供 APN 等使用）
        if t >= t_break and man_dir is not None:
            n_now = los_normal(r)
            geom.a_t_perp = aT * (man_dir[0] * n_now[0]
                                  + man_dir[1] * n_now[1])

        lam_true = math.atan2(r[1], r[0])
        lam_m, ld_est = sk.measure(lam_true, geom.lamdot, dt)
        geom.lamdot = ld_est                      # 用含噪声的估计值

        # --- 制导指令 ---
        a_cmd = gl.command(law, geom, N=N, **law_kw) \
            if law != "PURE_PURSUIT" else gl.command(law, geom, K=N, **law_kw)
        a_cmd = max(-amax, min(amax, a_cmd))

        # --- 舵机一阶延迟（解析离散解）---
        if delay > 0:
            a_lag = a_cmd + (a_lag - a_cmd) * math.exp(-dt / delay)
        else:
            a_lag = a_cmd
        peak_g = max(peak_g, abs(a_lag))

        # --- 框架角 / 视场 ---
        theta_m = math.atan2(m[3], m[2])
        sk.update_gimbal(lam_true, theta_m, rn, GIMBAL_RMIN)
        if gimbal_term is None and rn <= GIMBAL_RMIN:
            gimbal_term = abs(wrap_pi(lam_true - theta_m))

        if record:
            traj.append((t, m[0], m[1], tx, ty, a_lag / G,
                         math.degrees(abs(wrap_pi(lam_true - theta_m)))))

        # --- 终止判据 ---
        rdot = (vtx - m[2], vty - m[3])
        if rn <= r_stop:
            miss = cpa_miss(r, rdot)
            break
        if step > 0 and (r[0] * rdot[0] + r[1] * rdot[1]) > 0:
            miss = cpa_miss(r, rdot)              # 越过最近点
            break

        # --- RK4 积分（步内加速度恒为 a_lag，解析精确）---
        ang = geom.lam if law != "PPN" else theta_m
        ax, ay = -math.sin(ang) * a_lag, math.cos(ang) * a_lag
        k1 = (m[2], m[3], ax, ay)
        s2 = [m[i] + 0.5 * dt * k1[i] for i in range(4)]
        k2 = (s2[2], s2[3], ax, ay)
        s3 = [m[i] + 0.5 * dt * k2[i] for i in range(4)]
        k3 = (s3[2], s3[3], ax, ay)
        s4 = [m[i] + dt * k3[i] for i in range(4)]
        k4 = (s4[2], s4[3], ax, ay)
        for i in range(4):
            m[i] += dt / 6.0 * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i])
    else:
        tx, ty, vtx, vty = tgt_state(n_steps * dt)
        miss = cpa_miss((tx - m[0], ty - m[1]), (vtx - m[2], vty - m[3]))

    return {
        "miss": miss,
        "peak_g": peak_g / G,
        "gimbal_max": math.degrees(sk.gimbal_max),
        "gimbal_term": (math.degrees(gimbal_term)
                        if gimbal_term is not None else None),
        "lost": sk.lost_lock,
        "t_end": step * dt,
        "r_min": r_min,
        "traj": traj,
    }


def monte_carlo(N=4.0, law="TPN", n_mc=40, g_lo=7.0, g_hi=11.0,
                tb_lo=4.6, tb_hi=5.3, seed0=0, **kw):
    """蒙特卡洛：目标机动幅值 + 机动起始时刻随机，返回脱靶量列表。"""
    rng = random.Random(20260901 + seed0)
    out = []
    for i in range(n_mc):
        p = dict(kw)
        p["tgt_g"] = rng.uniform(g_lo, g_hi) if g_hi > g_lo else g_lo
        p["t_break"] = rng.uniform(tb_lo, tb_hi) if tb_hi > tb_lo else tb_lo
        out.append(simulate(N=N, law=law, seed=seed0 + i, **p)["miss"])
    return out


# ---------------------------------------------------------------- 自检
def _self_test():
    print("=" * 72)
    print("  engagement 自检（基准工况：末段突然机动）")
    print("=" * 72)
    sc = default_scenario()

    # 1. 理想工况（碰撞三角、无机动/噪声/延迟）应零脱靶
    r = simulate(N=4.0, t_break=1e9, tgt_g=0.0, delay=0.0, dt=0.001)
    assert r["miss"] < 1e-6, f"理想工况脱靶过大：{r['miss']}"
    assert r["peak_g"] < 1e-6, f"理想工况不应需过载：{r['peak_g']}"
    print(f"  [1] 理想工况脱靶        : {r['miss']:.3e} m / "
          f"峰值 {r['peak_g']:.3e} g  ✅")

    # 2. 纯机动工况：脱靶随 N 单调下降
    ms = [monte_carlo(N=n, n_mc=20, g_lo=9.0, g_hi=9.0, tb_lo=5.0,
                      tb_hi=5.0, delay=0.0, dt=0.002, **sc)
          for n in (2.5, 4.0, 8.0)]
    rms = [math.sqrt(sum(x * x for x in v) / len(v)) for v in ms]
    assert rms[-1] < rms[0], f"机动工况未随 N 改善：{rms}"
    print("  [2] 机动工况 N 扫描     : N=2.5→8  RMS "
          + " → ".join(f"{v:.4f}" for v in rms) + " m  ✅")

    # 3. 延迟门槛效应
    ds = []
    for tau in (0.0, 0.05, 0.10, 0.20, 0.30):
        v = monte_carlo(N=4.0, n_mc=20, g_lo=9.0, g_hi=9.0, tb_lo=5.0,
                        tb_hi=5.0, delay=tau, dt=0.002, **sc)
        ds.append(math.sqrt(sum(x * x for x in v) / len(v)))
    assert ds[-1] > 10 * max(ds[0], 1e-6), f"延迟门槛不明显：{ds}"
    print("  [3] 舵机延迟门槛        : τ=0/50/100/200/300 ms → "
          + " → ".join(f"{v:.3f}" for v in ds) + " m  ✅")

    # 4. 角度噪声下「N 放大噪声」
    ns = []
    for n in (2.5, 5.0, 8.0):
        v = monte_carlo(N=n, n_mc=25, g_lo=0.0, g_hi=0.0, tb_lo=1e9,
                        tb_hi=1e9, sig_lam=1e-3, tau_f=0.05, delay=0.02,
                        dt=0.002, **sc)
        ns.append(math.sqrt(sum(x * x for x in v) / len(v)))
    assert ns[-1] > ns[0], f"角度噪声未复现 N 放大效应：{ns}"
    print("  [4] 角度噪声 N 扫描     : N=2.5→8  RMS "
          + " → ".join(f"{v:.3f}" for v in ns) + " m  单调↑ ✅")

    # 5. 过载限幅生效
    r = simulate(N=8.0, amax=10.0 * G, t_break=5.0, tgt_g=9.0,
                 delay=0.10, dt=0.002, R0=6000.0, H0=500.0,
                 Vm=700.0, Vt=300.0)
    assert r["peak_g"] <= 10.0 + 1e-6, f"限幅失效：{r['peak_g']}"
    print(f"  [5] 过载限幅            : 限幅 10g → 峰值 {r['peak_g']:.3f} g  ✅")

    # 6. APN 优于 PN（机动 + 过载受限）
    kw = dict(t_break=4.0, tgt_g=12.0, amax=20.0 * G, delay=0.05, dt=0.002,
              R0=6000.0, H0=500.0, Vm=700.0, Vt=300.0)
    pn = simulate(N=4.5, law="TPN", **kw)["miss"]
    ap = simulate(N=4.5, law="APN", **kw)["miss"]
    assert ap < pn, f"APN 未优于 PN：{ap} vs {pn}"
    print(f"  [6] APN vs PN           : PN {pn:.3f} m → APN {ap:.3f} m  ✅")

    # 7. 框架角合理性（非 175° 这类荒谬值）
    r = simulate(N=4.0, delay=0.05, dt=0.002, **sc)
    assert 0 <= r["gimbal_max"] < 90.0, f"框架角异常：{r['gimbal_max']}"
    assert r["gimbal_term"] is not None and r["gimbal_term"] < 90.0
    print(f"  [7] 框架角              : 最大 {r['gimbal_max']:.2f}° / "
          f"终端 {r['gimbal_term']:.2f}°  ✅")

    # 8. 轨迹记录可用于绘图
    r = simulate(N=4.0, delay=0.05, dt=0.002, record=True, **sc)
    assert len(r["traj"]) > 100 and len(r["traj"][0]) == 7
    print(f"  [8] 轨迹记录            : {len(r['traj'])} 个采样点 × 7 字段  ✅")

    print("=" * 72)
    print("  engagement：8 项自检全部通过")
    print("=" * 72)


if __name__ == "__main__":
    _self_test()
