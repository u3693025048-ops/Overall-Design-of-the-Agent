#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dynamics.py — 执行机构、限幅与数值积分

组件
----
    FirstOrderLag : 舵机一阶延迟（解析离散解，步长无关）
    Limiter       : 过载限幅（硬 / 软 tanh / 动态）
    rk4_step      : 经典四阶 Runge–Kutta（向量状态）
    integrate     : 定步长积分驱动器（带终止判据）

设计依据（book010 卡 A-04 / B-02 经验）
--------------------------------------
* 一阶延迟用**解析离散解** a ← a_cmd + (a − a_cmd)·e^(−dt/τ)，
  而非欧拉前向差分 —— 后者在大 dt/τ 时会失真甚至不稳定。
* 限幅保留 **20%~30% 过载裕度**："按 80% 可用过载设计，靶试时需求冲到 85%"。
* 软限幅 u = u_max·tanh(u/u_max) 优于硬限幅（避免指令突变、
  保护舵机）；动态限幅（按剩余时间/能量调整上限）工程上最推荐。

纯标准库。自检：python3 dynamics.py
"""
import math

__all__ = ["FirstOrderLag", "Limiter", "rk4_step", "integrate", "G"]

G = 9.80665


# ---------------------------------------------------------------- 一阶延迟
class FirstOrderLag:
    """一阶惯性环节（舵机 / 滤波器）：τ·ȧ + a = a_cmd

    解析离散解：a(t+dt) = a_cmd + (a(t) − a_cmd)·exp(−dt/τ)
    该解对任意 dt 都是精确的（假设 a_cmd 步内恒定），
    而前向差分 a += (a_cmd − a)·dt/τ 在 dt > 2τ 时会发散。
    """

    def __init__(self, tau, y0=0.0):
        self.tau = tau
        self.y = y0

    def reset(self, y0=0.0):
        self.y = y0
        return self.y

    def step(self, cmd, dt):
        if self.tau <= 0.0:
            self.y = cmd
        else:
            self.y = cmd + (self.y - cmd) * math.exp(-dt / self.tau)
        return self.y

    def step_euler(self, cmd, dt):
        """前向差分（仅作对比，证明解析解的优势）。"""
        if self.tau <= 0.0:
            self.y = cmd
        else:
            self.y += (cmd - self.y) * dt / self.tau
        return self.y


# ---------------------------------------------------------------- 限幅
class Limiter:
    """过载/指令限幅。

    mode:
        'hard'    : 直接截断（简单，但指令突变、易激发未建模动态）
        'soft'    : u = u_max·tanh(u/u_max)（连续可微，推荐）
        'dynamic' : 上限随时变化（如按 t_go 收紧），需每步传入 limit
    """

    def __init__(self, limit, mode="hard", margin=1.0):
        self.limit = limit
        self.mode = mode
        self.margin = margin            # 设计裕度系数：<1 表示降额使用

    def amax(self, dynamic_limit=None):
        lim = self.limit if dynamic_limit is None else dynamic_limit
        return lim * self.margin

    def __call__(self, u, dynamic_limit=None):
        am = self.amax(dynamic_limit)
        if self.mode == "hard":
            return max(-am, min(am, u))
        if self.mode == "soft":
            return am * math.tanh(u / am) if am > 0 else 0.0
        if self.mode == "dynamic":
            return max(-am, min(am, u))
        raise ValueError(f"未知限幅模式：{self.mode}")


# ---------------------------------------------------------------- 积分器
def rk4_step(f, t, y, dt):
    """经典 RK4 单步：y ← y + dt/6·(k1 + 2k2 + 2k3 + k4)

    Args:
        f: 导数函数 f(t, y) -> dy/dt（y 为序列）
        y: 状态序列
    Returns:
        新状态（与 y 同类型：list 或 tuple）
    """
    n = len(y)
    k1 = f(t, y)
    y2 = [y[i] + 0.5 * dt * k1[i] for i in range(n)]
    k2 = f(t + 0.5 * dt, y2)
    y3 = [y[i] + 0.5 * dt * k2[i] for i in range(n)]
    k3 = f(t + 0.5 * dt, y3)
    y4 = [y[i] + dt * k3[i] for i in range(n)]
    k4 = f(t + dt, y4)
    return [y[i] + dt / 6.0 * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i])
            for i in range(n)]


def integrate(f, y0, dt, t_max, stop=None):
    """定步长积分驱动器。

    Args:
        stop: 可选终止判据 stop(t, y) -> bool
    Returns:
        (t_end, y_end, history)  history 为 [(t, y...), ...]
    """
    y = list(y0)
    t = 0.0
    hist = [(t,) + tuple(y)]
    n = int(t_max / dt)
    for k in range(n):
        y = rk4_step(f, t, y, dt)
        t = (k + 1) * dt
        hist.append((t,) + tuple(y))
        if stop is not None and stop(t, y):
            break
    return t, y, hist


# ---------------------------------------------------------------- 自检
def _self_test():
    print("=" * 72)
    print("  dynamics 自检")
    print("=" * 72)

    # 1. 一阶延迟：解析解 vs 细分数值积分（应完全一致）
    tau, dt = 0.1, 0.02
    a = FirstOrderLag(tau)
    y = 0.0
    for _ in range(50):                      # 1 s
        y = a.step(1.0, dt)
    ref = FirstOrderLag(tau)                 # 细分 100 倍作为参考
    yr = 0.0
    for _ in range(5000):
        yr = ref.step(1.0, dt / 100.0)
    assert abs(y - yr) < 1e-12, f"解析解与细分不一致：{y} vs {yr}"
    print(f"  [1] 一阶延迟解析解      : dt/τ=0.2 时解析 {y:.12f} "
          f"= 细分参考 {yr:.12f}  ✅")

    # 2. 时间常数验证：t = τ 时应达 63.2%
    a = FirstOrderLag(0.1)
    for _ in range(10):                      # 10 × 0.01 s = 0.1 s
        v = a.step(1.0, 0.01)
    assert abs(v - 0.6321205588) < 1e-4, f"时间常数不符合一阶特性：{v}"
    print(f"  [2] 阶跃响应 τ 特性     : t=τ 时输出 {v:.6f} "
          f"(理论 1−e⁻¹=0.632121)  ✅")

    # 3. 大步长下解析解稳定，前向差分发散
    a1, a2 = FirstOrderLag(0.01), FirstOrderLag(0.01)
    v1 = v2 = 0.0
    for _ in range(200):                     # dt = 3τ = 0.03
        v1 = a1.step(1.0, 0.03)
        v2 = a2.step_euler(1.0, 0.03)
    assert abs(v1 - 1.0) < 1e-3, f"解析解在大步长下失真：{v1}"
    assert abs(v2) > 1e3, f"前向差分未发散，对比失效：{v2}"
    print(f"  [3] 大步长 dt=3τ        : 解析解 {v1:.6f}（收敛）vs "
          f"前向差分 {v2:.3e}（发散）✅")

    # 4. 三种限幅模式
    xs = [0.5, 1.0, 2.0, 10.0]
    for mode, lim in (("hard", 1.0), ("soft", 1.0)):
        L = Limiter(lim, mode=mode)
        outs = [L(x) for x in xs]
        assert all(abs(o) <= lim + 1e-12 for o in outs), f"{mode} 越限：{outs}"
        print(f"  [4] 限幅 {mode:5s}(u_max=1)   : "
              + " ".join(f"{o:.4f}" for o in outs) + "  ✅")
    L = Limiter(1.0, mode="soft")
    assert L(0.5) > 0.5 * 0.9 and L(0.5) < 0.5, "软限幅在小信号处应接近线性"
    print("       ⇒ 软限幅小信号接近线性、大信号平滑饱和 ✅")

    # 5. 降额裕度
    L = Limiter(10.0, mode="hard", margin=0.8)
    assert L(100.0) == 8.0, f"20% 裕度未生效：{L(100.0)}"
    print(f"  [5] 设计裕度 margin=0.8  : u=100 → {L(100.0):.1f}"
          f"（限幅 10 降额至 8，卡 A-04 建议留 20%~30%）✅")

    # 6. RK4 四阶精度：简谐运动，步长减半误差应降约 16 倍
    #    （积分时长取 2 s，避免长时间累积使大步长脱离渐近收敛区）
    def f(t, y):                             # y = [x, v], ẍ = −x
        return [y[1], -y[0]]

    errs = []
    for dt in (0.4, 0.2, 0.1):
        _, y, _ = integrate(f, [1.0, 0.0], dt, 2.0)
        errs.append(abs(y[0] - math.cos(2.0)))
    ratios = [errs[i] / errs[i + 1] for i in range(len(errs) - 1)]
    assert all(r > 14.0 for r in ratios), f"RK4 未达四阶精度：{ratios}"
    print("  [6] RK4 收敛阶          : 步长减半误差比 "
          + " / ".join(f"{r:.1f}" for r in ratios)
          + f"（理论 16，末值误差 {errs[-1]:.2e}）✅")

    # 7. 常加速度积分应解析精确
    def fc(t, y):
        return [y[1], 5.0]
    _, y7, _ = integrate(fc, [0.0, 0.0], 0.1, 2.0)
    assert abs(y7[0] - 0.5 * 5.0 * 4.0) < 1e-12, f"常加速度积分不精确：{y7[0]}"
    print(f"  [7] 常加速度积分        : x(2s) = {y7[0]:.12f} "
          f"(解析 10.0)  ✅")

    # 8. 终止判据
    def fs(t, y):
        return [1.0]
    t_end, y_end, hist = integrate(fs, [0.0], 0.1, 100.0,
                                   stop=lambda t, y: y[0] >= 2.0)
    assert 1.9 <= y_end[0] <= 2.2 and t_end < 100.0
    print(f"  [8] 终止判据            : 在 t={t_end:.2f}s 停止，"
          f"y={y_end[0]:.2f}（目标 2.0）✅")

    print("=" * 72)
    print("  dynamics：8 项自检全部通过")
    print("=" * 72)


if __name__ == "__main__":
    _self_test()
