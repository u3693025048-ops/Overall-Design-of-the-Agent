# Python 制导算法库

> 与 book010 的 30 张 RIA++ 卡配套的可执行实现。
> **纯标准库**（无 numpy/scipy 依赖），Python 3.9+ 可直接运行。
> 每个模块自带自检，`python3 <模块>.py` 即可验证。

## 快速开始

```bash
cd algorithms
python3 run_all.py          # 一键自检（7 个模块）+ 综合工程演示
python3 gnc_core.py         # 单独跑某个模块的自检
```

```python
import engagement as eg

# 一次交战仿真：末段突然机动的最坏工况
r = eg.simulate(N=4.0, law="TPN", R0=6000, H0=500, Vm=700, Vt=300,
                t_break=5.0, tgt_g=9.0, amax=25*9.80665,
                delay=0.10, sig_lam=1e-3, tau_f=0.05)
print(r["miss"], r["peak_g"], r["gimbal_max"])   # 脱靶 / 峰值过载 / 最大框架角
```

## 模块清单

| 模块 | 职责 | 自检项 | 卡片出处 |
|---|---|---|---|
| `gnc_core.py` | 向量/角度/LOS 运动学、碰撞三角、$t_{go}$、ZEM、CPA | 8 | A-01, B-02 |
| `dynamics.py` | 一阶延迟（解析离散解）、限幅（硬/软/动态）、RK4 | 8 | A-04, B-02 |
| `guidance_laws.py` | **10 种制导律**统一接口 + 注册表 | 11 | A-01/05/06, B-02/03, C-02/05, E-02/04 |
| `seeker.py` | 有色噪声 AR(1)、角度/速率噪声模型、滤波微分、视场框架角 | 6 | A-01, C-02 |
| `engagement.py` | 闭环交战仿真（机动/延迟/噪声/限幅/视场） | 8 | A-02/04/05, D 域 |
| `metrics.py` | 分位数/CEP/工程评级/验收判据 | 8 | B-05 |
| `monte_carlo.py` | 分布建模、拉丁超立方、1D/2D 扫描 | 9 | B-05 |
| `run_all.py` | 一键自检 + 综合标定演示 | — | — |

## 制导律注册表

`guidance_laws.LAWS` 提供统一接口 `command(name, geom, **params) -> a_cmd`，
可直接互换、可参数化（便于 MBSE 中的「同一接口下的策略变体」建模）：

| 名称 | 公式 | 用途 | 卡片 |
|---|---|---|---|
| `TPN` | $a=N V_c\dot\lambda$（⊥视线） | 通用首选 | A-01 |
| `PPN` | $a=N V_m\dot\lambda$（⊥速度） | 大气层内，需 $N$=6~10 | A-01 |
| `APN` | $a=NV_c\dot\lambda+\frac{N}{2}\hat a_{T\perp}$ | 机动目标（过载受限时才有收益） | B-03, A-05 |
| `BPN` | $a=NV\dot\lambda+K_{bias}(\theta-\theta_{ref})$ | 落角/视场约束 | C-02 |
| `OGL` | $a=N\cdot \mathrm{ZEM}/t_{go}^2$ | 最优制导，$N$=3 等价 PN | B-02 |
| `IMPACT_ANGLE` | $a=N_1\frac{y}{t^2}+N_2\frac{\dot y}{t}+N_3\frac{V_m(\theta-\theta_d)}{t}$ | 终端落角控制 | C-05 |
| `FOV_BARRIER` | $\dot\lambda_{des}=-k\lambda(\lambda_{max}^2-\lambda^2)^2$ | 视场硬约束 | C-02 |
| `SMC` | $s=\dot\lambda+c\lambda,\ u=-ks-\varepsilon\,\mathrm{sat}(s/\Phi)$ | 强鲁棒 | E-02/04 |
| `PURE_PURSUIT` | $a=KV_m(\lambda-\theta_m)$ | 对照基线 | A-01 |
| `PREDICTIVE` | 瞄准 $P_{pred}=P_T+V_T t_{go}+\frac12\hat A_T t_{go}^2$ | 融合制导 | A-06 |

## 五条不可违背的建模约定

这些是从 book010 仿真踩坑中固化下来的，写在代码注释里，改动前请先读：

1. **$\dot\lambda$ 符号**：$\dot\lambda=(r_x\dot r_y-r_y\dot r_x)/|r|^2$，
   写反会让导弹被"推开"（`gnc_core.los_rate`）。
2. **初速按碰撞三角设定**：否则理想工况就有 0.5 m 级"假脱靶"
   （`gnc_core.collision_heading`）。
3. **目标机动必须进运动学**：只把机动喂给 APN 补偿项是致命 bug
   （`engagement.simulate` 内分段解析积分）。
4. **噪声用角度噪声模型**：测角噪声 + 滤波微分；速率噪声模型会得出
   与教科书相反的结论（`seeker.Seeker`）。
5. **框架角 $|\lambda-\theta_m|$ 且只在 $r>150$ m 统计**：
   用 $\arccos(\hat v\cdot\hat r)$ 会算出 175° 这类荒谬值。

## 自检覆盖

```text
gnc_core     8 项   λ̇ 符号 / 碰撞三角 λ̇≡0 / ZEM 双算法一致 / CPA 解析 vs 数值
dynamics     8 项   解析解 vs 细分 / τ 特性 / 大步长稳定性 / RK4 四阶（比值 16.5）
guidance    11 项   10 种律 / OGL≡PN / APN 收益边界 / BPN 落角 / FOV 屏障
seeker       6 项   AR(1) 与 dt 无关 / 滤波微分 / 两种噪声模型的相反结论
engagement   8 项   理想脱靶 2.4e-15 m / 延迟门槛 / 限幅 / 框架角合理性
metrics      8 项   分位数 / CEP / 评级阈值 / 验收倍率
monte_carlo  9 项   分布 / 拉丁超立方 / 1/√n 收敛 / 可复现性
```

## 已知数值结论（可复现）

| 现象 | 数值 | 说明 |
|---|---|---|
| 理想工况脱靶 | $2.4\times10^{-15}$ m | 碰撞三角 + 无扰动 |
| 舵机延迟门槛 | ≤50 ms 无碍；300 ms → 7.5 m | 判据 $N\tau/t_{go}\to1$ |
| $\lambdȧ$ 滤波最优 $\tau_f$ | ≈20 ms | 5 ms→3.45 m，200 ms→0.34 m |
| 纯差分 vs 滤波 | 噪声 RMS 0.312 vs 0.035 rad/s | **9 倍**，印证"$\lambdȧ$ 绝不能差分" |
| APN 收益 | 20 g 限幅下 3.0 m；不受限 ≈0 | 过载不受限时无需 APN |
| 角度噪声下 N 趋势 | 0.034 → 0.083 m（N=2.5→8） | 单调上升（教科书结论） |
| 速率噪声下脱靶 | 小 2 个数量级 | 噪声被回路吃掉，模型不真实 |
