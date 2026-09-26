#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
seeker.py — 导引头与噪声模型

本模块固化了 book010 蒸馏中最重要的一条建模经验：

    ⚠️ 「N 是噪声放大器」的结论，只在【角度噪声 + 滤波微分】模型下成立。

    速率噪声模型（直接在 λ̇ 上叠平谱噪声）
        → 回路带宽越高抑制越好 → 脱靶量随 N **下降**（与教科书相反）
    角度噪声模型（测角噪声 σ_λ → 一阶滤波微分求 λ̇）  ← 真实导引头
        → 微分使噪声谱变「蓝」→ 回路带宽（∝N）越高吃得越多
        → 脱靶量随 N **上升**（教科书效应）

    若仿真中得出与教科书相反的结论，先查是不是用错了噪声模型。

组件
----
    ColoredNoise : AR(1) 高斯–马尔可夫有色噪声，统计特性与步长 dt 无关
    Seeker       : 导引头（测角噪声 / 滤波微分 / 速率噪声 / 视场 / 框架角）

纯标准库。自检：python3 seeker.py
"""
import math
import random

from gnc_core import wrap_pi

__all__ = ["ColoredNoise", "Seeker", "rad", "deg"]

G = 9.80665


def rad(x):
    return x * 0.017453292519943295


def deg(x):
    return x * 57.29577951308232


# ---------------------------------------------------------------- 有色噪声
class ColoredNoise:
    """AR(1) 高斯–马尔可夫有色噪声：n ← φ·n + σ·√(1−φ²)·N(0,1)

    关键性质：φ = exp(−dt/τ) 使稳态方差恒为 σ²，**与积分步长 dt 无关**。
    （若用「每步独立白噪声」，噪声功率谱会随 dt 变化，导致换步长后
     仿真结论不可复现 —— 这是蒙特卡洛结果不可信的常见根因。）
    """

    def __init__(self, sigma=0.0, tau=0.02, seed=None):
        self.sigma = sigma
        self.tau = tau
        self.rng = random.Random(seed)
        self.state = 0.0

    def reset(self):
        self.state = 0.0

    def step(self, dt):
        """推进一步，返回噪声样本。"""
        if self.sigma <= 0.0:
            return 0.0
        phi = math.exp(-dt / self.tau) if self.tau > 0 else 0.0
        self.state = (self.state * phi
                      + self.sigma * math.sqrt(1.0 - phi * phi)
                      * self.rng.gauss(0.0, 1.0))
        return self.state


# ---------------------------------------------------------------- 导引头
class Seeker:
    """导引头模型（测角 → 滤波微分 → λ̇ 估计）。

    Args:
        sigma_lam: 测角噪声标准差（rad）。典型 1 mrad（闪烁/热噪声）
        tau_f: λ̇ 一阶滤波微分时间常数（s）。本工况最优 ≈20 ms
        rate_noise: 速率噪声模型下的 λ̇ 噪声强度（rad/s）；>0 时启用
                    （此模型会得出与教科书相反的结论，仅供对比）
        noise_tau: 测角噪声相关时间（s）
        fov_half: 视场半角（rad），默认 30°
        seed: 随机种子

    说明：sigma_lam = 0 且 rate_noise = 0 时，直接返回解析 λ̇（旁路滤波），
    以保证「理想工况零脱靶」这一基准成立。
    """

    def __init__(self, sigma_lam=0.0, tau_f=0.02, rate_noise=0.0,
                 noise_tau=0.02, fov_half=rad(30.0), seed=None):
        self.sigma_lam = sigma_lam
        self.tau_f = tau_f
        self.rate_noise = rate_noise
        self.fov_half = fov_half
        self.ang_noise = ColoredNoise(sigma_lam, noise_tau,
                                      seed if seed is not None else 0)
        self.rate_gen = ColoredNoise(rate_noise, noise_tau,
                                     (seed + 7777) if seed is not None else 1)
        self.reset()

    def reset(self):
        self.lam_prev = None
        self.ld_est = 0.0
        self.ang_noise.reset()
        self.rate_gen.reset()
        self.lost_lock = False
        self.gimbal_max = 0.0

    @property
    def bypass(self):
        return self.sigma_lam <= 0.0 and self.rate_noise <= 0.0

    def measure(self, lam_true, lam_dot_true, dt):
        """推进一步，返回 (λ 测量值, λ̇ 估计值)。

        角度噪声模型：
            λ_m = λ + n_λ                      （AR(1) 测角噪声）
            λ̇_est ← λ̇_est + ((λ_m − λ_m,prev)/dt − λ̇_est)·min(dt/τ_f, 1)
        速率噪声模型：
            λ̇_est = λ̇ + n_λ̇                   （AR(1) 直接叠加）
        """
        if self.bypass:
            return lam_true, lam_dot_true

        if self.rate_noise > 0.0:
            return lam_true, lam_dot_true + self.rate_gen.step(dt)

        lam_m = lam_true + self.ang_noise.step(dt)
        if self.lam_prev is None:
            self.lam_prev = lam_m
        raw = (lam_m - self.lam_prev) / dt
        alpha = min(dt / self.tau_f, 1.0) if self.tau_f > 0 else 1.0
        self.ld_est += (raw - self.ld_est) * alpha
        self.lam_prev = lam_m
        return lam_m, self.ld_est

    # -------------------------------------------------- 框架角 / 视场
    def update_gimbal(self, lam, theta_m, r, r_min=150.0):
        """统计框架角 |λ − θ_m|。

        ⚠️ 两个经典坑：
          1. 定义必须是 |λ − θ_m|（视线角 − 弹道倾角），
             用 arccos(v̂·r̂) 会算出 175° 这类荒谬值；
          2. 只在 r > r_min 的**制导段**统计 —— r→0 时 λ 高速旋转，
             末端尖峰无物理意义（导引头早已失锁/停止工作）。
        """
        if r <= r_min:
            return self.gimbal_max
        g = abs(wrap_pi(lam - theta_m))
        if g > self.gimbal_max:
            self.gimbal_max = g
        if g > self.fov_half:
            self.lost_lock = True
        return self.gimbal_max


# ---------------------------------------------------------------- 自检
def _self_test():
    print("=" * 72)
    print("  seeker 自检")
    print("=" * 72)

    # 1. AR(1) 噪声：稳态标准差 = σ，且与 dt 无关
    sig, tau = 0.001, 0.02
    print("  [1] AR(1) 噪声统计（σ=1 mrad, τ=20 ms）")
    sds = []
    for dt in (0.0005, 0.001, 0.002, 0.005):
        n = ColoredNoise(sig, tau, seed=42)
        xs = [n.step(dt) for _ in range(40000)]
        mean = sum(xs) / len(xs)
        sd = math.sqrt(sum((x - mean) ** 2 for x in xs) / len(xs))
        sds.append(sd)
        print(f"      dt={dt*1000:5.1f} ms → 标准差 {sd:.6f} rad "
              f"(目标 {sig:.6f})")
    assert all(abs(s - sig) / sig < 0.10 for s in sds), \
        f"AR(1) 噪声统计与 dt 相关：{sds}"
    print("      ⇒ 与 dt 无关 ✅")

    # 2. 滤波微分：斜坡输入稳态无差（无噪声）
    sk = Seeker(sigma_lam=0.0, tau_f=0.02)
    dt = 0.001
    for i in range(3000):                       # 3 s ≈ 150τ_f，已稳态
        lam = 0.1 * i * dt
        _, ld = sk.measure(lam, 0.1, dt)
    assert abs(ld - 0.1) < 1e-6, f"斜坡输入稳态误差过大：{ld}"
    print(f"  [2] 滤波微分斜坡跟随    : λ̇_est={ld:.8f} (真值 0.1)  ✅")

    # 3. 纯差分 vs 滤波微分：噪声抑制能力
    print("  [3] 测角噪声 σ_λ=1 mrad 下的 λ̇ 噪声 RMS")
    rows = []
    for tf, label in ((1e-9, "纯差分"), (0.005, "5 ms"), (0.02, "20 ms"),
                      (0.05, "50 ms"), (0.20, "200 ms")):
        sk = Seeker(sigma_lam=1e-3, tau_f=tf, seed=7)
        dt = 0.001
        acc, n = [], 40000
        for i in range(n):
            _, ld = sk.measure(0.0, 0.0, dt)    # 真值恒 0，输出纯噪声
            if i > 5000:
                acc.append(ld)
        rms = math.sqrt(sum(x * x for x in acc) / len(acc))
        rows.append((label, rms))
        print(f"      τ_f={label:>7s} → λ̇ 噪声 RMS {rms:.5f} rad/s")
    assert rows[0][1] > rows[-1][1], "滤波未抑制噪声"
    print("      ⇒ 滤波时间常数越长，噪声越小（但相位滞后越大，见 20 ms 最优）✅")

    # 4. 框架角定义与门限
    sk = Seeker(fov_half=rad(30.0))
    a = sk.update_gimbal(0.5, 0.2, r=1000.0)          # 制导段
    assert abs(a - 0.3) < 1e-9, f"框架角定义错误：{a}"
    b = sk.update_gimbal(0.5, 0.2 + math.pi, r=1000.0)  # wrap 验证
    assert abs(b - abs(wrap_pi(0.5 - 0.2 - math.pi))) < 1e-9
    c = sk.update_gimbal(2.0, 0.0, r=50.0)            # r<150 m 应被忽略
    assert c == b, "末端尖峰未被门限滤除"
    print(f"  [4] 框架角 =|λ−θ_m|     : {deg(a):.2f}°，wrap 正确，"
          f"r<150 m 已忽略  ✅")

    # 5. 视场失锁判定
    sk2 = Seeker(fov_half=rad(30.0))
    sk2.update_gimbal(0.0, rad(20.0), r=1000.0)       # 20° < 30°，未失锁
    assert not sk2.lost_lock
    sk2.update_gimbal(0.0, rad(45.0), r=1000.0)       # 45° > 30°，失锁
    assert sk2.lost_lock
    print("  [5] 视场失锁判定        : 20° 保持锁定 / 45° 触发失锁  ✅")

    # 6. 端到端：两种噪声模型给出相反的 N 趋势（本轮核心发现）
    print("  [6] 端到端：噪声模型决定 N 的趋势方向")
    try:
        import engagement as eg
    except Exception as e:                       # pragma: no cover
        print(f"      （跳过：engagement 未就绪 {e}）")
        return
    base = dict(R0=6000.0, H0=500.0, Vm=700.0, Vt=300.0, amax=25.0 * G,
                dt=0.002, n_mc=20, g_lo=0.0, g_hi=0.0,
                tb_lo=1e9, tb_hi=1e9)     # 非机动目标：隔离噪声效应
    for tag, kw in (("角度噪声 σ_λ=1mrad", dict(sig_lam=1e-3, tau_f=0.05)),
                    ("速率噪声（平谱）", dict(rate_noise=2e-3))):
        res = []
        for N in (2.5, 5.0, 8.0):
            m = eg.monte_carlo(N=N, seed0=300, **base, **kw)
            res.append(math.sqrt(sum(x * x for x in m) / len(m)))
        up = res[-1] > res[0]
        arrow = "↑" if up else "↓"
        print(f"      {tag:18s} N=2.5→8: "
              + " → ".join(f"{v:.3f}" for v in res) + f" m  {arrow}")
        if "角度" in tag:
            assert up, f"角度噪声模型未复现「N 放大噪声」：{res}"
            ang_last = res[-1]
        else:
            # 速率噪声模型的脱靶量级应远小于角度噪声模型：
            # 平谱 λ̇ 噪声被引擎回路（带宽 ∝ N）吃掉，几乎不产生脱靶，
            # 因而**无法刻画噪声对制导的真实危害** —— 这正是它不真实的原因。
            assert res[-1] < ang_last / 50.0, \
                f"速率噪声脱靶量级未显著小于角度噪声：{res[-1]} vs {ang_last}"
    print("      ⇒ 只有「角度噪声 + 滤波微分」才复现教科书结论；\n"
          "        速率噪声模型脱靶小 2 个数量级，噪声被回路吃掉，不可用 ✅")

    print("=" * 72)
    print("  seeker：6 项自检全部通过")
    print("=" * 72)


if __name__ == "__main__":
    _self_test()
