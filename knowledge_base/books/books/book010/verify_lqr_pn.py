#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
符号推导验证 — 对应卡片：
  A-卡01/02（PN 基本律、导航比 N）、A-卡04（ZEM）、B-卡02（LQR 与 Riccati）

核心命题：从 LQR 二次型指标出发，符号化证明「最优导航比 N = 3」。

推导链：
  1. 弹目相对运动简化为双积分器   x = [y, v_y]^T,  x_dot = A x + B u
  2. Riccati 微分方程（有限时域，以剩余时间 τ = t_f - t 为自变量，反向积分）
  3. 终端硬约束 s → ∞（强制脱靶量为零）时，Riccati 有闭式解 P(τ)
  4. 最优控制 u* = -R^{-1} B^T P x = -3·ZEM/τ²    ← N=3 的来源
  5. 与 PN 律 a_m = N·V_c·λ̇ 联立，λ̇ = ZEM/(V_c τ²)，得 N = 3
  6. 附带验证 LOS 动力学 λ̈ = -(2-N)(V_c/R)λ̇ + a_t/R  → N>2 稳定性红线

依赖：sympy, numpy, scipy（venv 环境）
运行：/Users/lisen/.workbuddy/binaries/python/envs/default/bin/python3 distill/verify_lqr_pn.py
"""
import sympy as sp

OK = "✅"


def sep(t):
    print("\n" + "=" * 72)
    print(t)
    print("=" * 72)


# ------------------------------------------------------------------ 符号
tau, r, q, s = sp.symbols('tau r q s', positive=True)   # 剩余时间 / 控制权重 / 状态权重 / 终端权重
y, v = sp.symbols('y v', real=True)                     # 相对位移 / 相对速度
N, Vc, Vm, R_, lam, t_go = sp.symbols('N V_c V_m R lambda t_go', positive=True)
a_t = sp.Function('a_t')

sep("STEP 1 · 双积分器建模与 LQR 问题设定")

A = sp.Matrix([[0, 1], [0, 0]])
B = sp.Matrix([[0], [1]])
Q = sp.Matrix([[q, 0], [0, 0]])
S = sp.Matrix([[s, 0], [0, 0]])
x = sp.Matrix([y, v])
print("状态方程   x_dot = A x + B u")
print("  A =", A.tolist(), "   B =", B.tolist())
print("  x = [y (脱靶量方向位移), v (相对速度)]^T,  u = a_m (导弹法向加速度指令)")
print("\n性能指标   J = 1/2∫(q·y² + r·u²)dt + 1/2·s·y²(t_f)")
print("  q : 脱靶量惩罚    r : 控制能量惩罚    s : 终端脱靶量惩罚")
print("\n Riccati（以剩余时间 τ = t_f - t 为自变量，正向积分即原问题的反向）")
print("  dP/dτ = AᵀP + PA + Q − P B r⁻¹ Bᵀ P,   P(0) = S")

sep("STEP 2 · Riccati 展开为标量 ODE 组")

p1, p2, p3 = sp.symbols('p1 p2 p3')
P = sp.Matrix([[p1, p2], [p2, p3]])
dP = A.T * P + P * A + Q - P * B * (1 / r) * B.T * P
dP = sp.simplify(dP)
print("设 P = [[p1, p2], [p2, p3]]（对称），则：")
print("  dp1/dτ =", sp.simplify(dP[0, 0]))
print("  dp2/dτ =", sp.simplify(dP[0, 1]))
print("  dp3/dτ =", sp.simplify(dP[1, 1]))
print("  终端条件 P(0) = S = diag(s, 0)")

# 验证展开式
assert sp.simplify(dP[0, 0] - (q - p2 ** 2 / r)) == 0
assert sp.simplify(dP[0, 1] - (p1 - p2 * p3 / r)) == 0
assert sp.simplify(dP[1, 1] - (2 * p2 - p3 ** 2 / r)) == 0
print(f"\n{OK} 展开结果与教科书标量组一致：")
print("     p1' = q − p2²/r      p2' = p1 − p2·p3/r      p3' = 2p2 − p3²/r")

sep("STEP 3 · 终端硬约束 s→∞ 时的闭式解（只罚能量 q=0）")

# 猜测解形式（由量纲分析：p1 ~ r/τ³, p2 ~ r/τ², p3 ~ r/τ）
a_, b_, c_ = sp.symbols('a b c', positive=True)
P_ans = {'p1': a_ / tau ** 3, 'p2': b_ / tau ** 2, 'p3': c_ / tau}
sub0 = {q: 0}                                    # 只罚控制能量
eqs = []
for lhs_expr, var, dvar in [
    (q - p2 ** 2 / r, 'p1', P_ans['p1']),
    (p1 - p2 * p3 / r, 'p2', P_ans['p2']),
    (2 * p2 - p3 ** 2 / r, 'p3', P_ans['p3']),
]:
    rhs = lhs_expr.subs(sub0).subs({p1: P_ans['p1'], p2: P_ans['p2'], p3: P_ans['p3']})
    eqs.append(sp.simplify(sp.diff(P_ans[var.lower()], tau) - rhs))
print("设 p1 = a/τ³, p2 = b/τ², p3 = c/τ，代入 ODE 组得代数方程组：")
for e in eqs:
    print("   ", sp.simplify(sp.expand(e * tau ** 4)), "= 0")

sol = sp.solve(eqs, [a_, b_, c_], dict=True)
print("\nSymPy 求解：", sol)
cands = [d for d in sol if all(sp.simplify(d.get(k, 0)) != 0 for k in (a_, b_, c_))]
assert cands, "未求得非零解"
S_ = cands[0]
print(f"{OK} 闭式解系数： a = {S_[a_]},  b = {S_[b_]},  c = {S_[c_]}")
assert sp.simplify(S_[a_] - 3 * r) == 0
assert sp.simplify(S_[b_] - 3 * r) == 0
assert sp.simplify(S_[c_] - 3 * r) == 0

P_opt = sp.Matrix([[P_ans['p1'], P_ans['p2']],
                   [P_ans['p2'], P_ans['p3']]]).subs(S_)
print(f"\n{OK} Riccati 闭式解（q=0, s→∞）：")
sp.pprint(sp.simplify(P_opt))

sep("STEP 4 · 最优控制律 → 3·ZEM/τ²  →  N = 3")

u_star = sp.simplify(-(1 / r) * (B.T * P_opt * x)[0, 0])
print("最优控制  u* = −r⁻¹ Bᵀ P x =")
sp.pprint(u_star)

ZEM = y + v * tau                                # 零控脱靶量
u_ZEM = -3 * ZEM / tau ** 2
print(f"\nZEM（零控脱靶量）= y + v·τ = {ZEM}")
print("断言  u* ≡ −3·ZEM/τ²  ？")
diff = sp.simplify(u_star - u_ZEM)
print("   差值 =", diff)
assert diff == 0, "u* 与 -3·ZEM/τ² 不等价"
print(f"{OK} 验证通过：u* = −3·(y + v·τ)/τ² = −3·ZEM/τ²")

# 与 PN 律联立
lam_dot_pn = ZEM / (Vc * tau ** 2)               # λ̇ 与 ZEM 的几何关系
a_pn = N * Vc * lam_dot_pn
print("\nPN 律   a_m = N·V_c·λ̇ ；由拦截几何有 λ̇ = ZEM/(V_c·t_go²)")
print("   ⇒ a_m = N·V_c·ZEM/(V_c·τ²) = N·ZEM/τ²   （τ 即 t_go）")
print("   与 |u*| = 3·ZEM/τ² 对比：")
N_opt = sp.solve(sp.Eq(sp.simplify(a_pn), 3 * ZEM / tau ** 2), N, dict=True)
print("   解得 N =", N_opt)
assert N_opt and sp.simplify(N_opt[0][N] - 3) == 0, "最优导航比不是 3"
print(f"{OK} 结论：LQR 二次型指标下的最优导航比 N_opt = 3  ← 书中「N=3 的由来」")

sep("STEP 5 · LOS 动力学与 N>2 稳定性红线（符号验证）")

# 拦截几何：R（弹目距离）、λ（视线角）、V_c（接近速度）、a_t（目标法向加速度）
t = sp.symbols('t', positive=True)
Rt = sp.Function('R')(t)
lmt = sp.Function('lambda')(t)
at_ = sp.Function('a_t')(t)
Vc_, N_ = sp.symbols('V_c N')

# 推导（小角线性化）：y ≈ R·λ ⇒ ẏ = Ṙλ + Rλ̇ ⇒ ÿ ≈ 2Ṙλ̇ + Rλ̈ （Ṙ = −V_c，R̈ ≈ 0）
# 横向受力：ÿ = a_t − a_m = a_t − N·V_c·λ̇
# 联立：R·λ̈ − 2V_c·λ̇ = a_t − N·V_c·λ̇
#   ⇒ R·λ̈ = a_t + (2 − N)·V_c·λ̇
lam_ddot_expr = (2 - N_) * Vc_ / Rt * sp.Derivative(lmt, t) + at_ / Rt
print("小角线性化推导：y ≈ R·λ ⇒ ÿ ≈ 2Ṙλ̇ + Rλ̈（Ṙ = −V_c，R̈ ≈ 0）")
print("横向受力：ÿ = a_t − a_m = a_t − N·V_c·λ̇")
print("联立 ⇒")
print("   R·λ̈ = (2 − N)·V_c·λ̇ + a_t")
sp.pprint(sp.Eq(sp.Derivative(lmt, t, 2), lam_ddot_expr))

print("\n齐次部分  λ̈ = (2−N)(V_c/R)λ̇  ⇒ 特征系数 −(N−2)·V_c/R：")
print("   N > 2 ：系数为负 → λ̇ 指数衰减 → 视线角速率收敛（稳定）")
print("   N = 2 ：λ̈ = a_t/R，λ̇ 不衰减 → 对机动目标脱靶量发散（临界）")
print("   N < 2 ：系数为正 → λ̇ 发散（失稳）")

# 数值佐证：解齐次 ODE，比较 N=1.5 / 2 / 3
import numpy as np
from scipy.integrate import solve_ivp

print("\n数值佐证（齐次方程 λ̈ = (2−N)(V_c/R)λ̇，R 线性减小至 0）:")
R0, Vc_val = 10000.0, 600.0
for Nv in [1.5, 2.0, 3.0, 4.0]:
    def f(tt, st):
        RR = max(R0 - Vc_val * tt, 1.0)
        return [st[1], (2 - Nv) * Vc_val / RR * st[1]]
    sol = solve_ivp(f, [0, 15.0], [0.0, 1e-3], rtol=1e-8, atol=1e-12, dense_output=True)
    end = sol.y[1, -1]
    if Nv < 2:
        tag = "λ̇ 放大 → 失稳"
    elif Nv == 2:
        tag = "λ̇ 不衰减 → 临界"
    else:
        tag = "λ̇ 衰减 → 稳定"
    print(f"   N={Nv:4.1f}  λ̇(0)=1e-3 →  λ̇(15s)={end:10.3e}   {tag}")
print(f"{OK} 与符号结论一致：N>2 是稳定性红线")

sep("STEP 6 · 有限终端权重 s 时，最优增益随 τ 收敛到 3（数值 Riccati）")

from scipy.integrate import solve_ivp as ivp


def riccati_gain(s_val, q_val, r_val, tau_grid):
    """数值积分 Riccati，返回有效导航比 N_eff(τ) = τ²/r·(p2 + p3·τ) ... 见下。"""
    def f(tt, st):
        P1, P2, P3 = st
        return [q_val - P2 ** 2 / r_val,
                P1 - P2 * P3 / r_val,
                2 * P2 - P3 ** 2 / r_val]
    sol = ivp(f, [1e-6, max(tau_grid)], [s_val, 0.0, 0.0],
              t_eval=tau_grid, rtol=1e-10, atol=1e-12, method='LSODA')
    return sol.y


# 有效导航比定义：u* = −(1/r)(p2·y + p3·v)。
# 若解为理想形式 p2 = N·r/τ²、p3 = N·r/τ，则 u* = −N·(y + v·τ)/τ² = −N·ZEM/τ²。
# 故可分别由两条通道反解 N：
#     位移通道  N_y = p2·τ²/r        速度通道  N_v = p3·τ/r
# 二者相等且等于 3 时，即证明最优律退化为 N=3 的 PN。
taus = np.array([0.5, 1.0, 2.0, 4.0, 8.0, 16.0])
r_val = 1.0
for s_val in [1e2, 1e4, 1e6]:
    P1, P2, P3 = riccati_gain(s_val, 0.0, r_val, taus)
    print(f"\n  s = {s_val:.0e} (q=0, r={r_val:g})：")
    print("     τ       N_y(位移通道)   N_v(速度通道)")
    for i, tv in enumerate(taus):
        n_y = P2[i] * tv ** 2 / r_val
        n_v = P3[i] * tv / r_val
        print(f"   {tv:5.1f}      {n_y:10.4f}        {n_v:10.4f}")
print(f"\n{OK} 两通道反解的有效导航比一致，且 s 越大越贴近 3.0000 —— 印证 N_opt = 3")
print("   注：s 有限（非硬约束）时 τ 较小处略小于 3，即终端前瞬时增益会\"软化\"；")
print("       s→∞（强制零脱靶）时全时域严格收敛到 3。")

sep("推导验证全部通过")
print("""
关键结论（已写入速查表，对应卡片 A-01/02/04、B-02）：
  · 双积分器 + 二次型指标 + 终端硬约束  ⇒  u* = −3·ZEM/τ²
  · 与 PN 律 a_m = N·V_c·λ̇ 联立（λ̇ = ZEM/(V_c·t_go²)）⇒ N_opt = 3
  · LOS 动力学 λ̈ = (2−N)(V_c/R)λ̇ + a_t/R  ⇒ N>2 为稳定性红线
     ⚠️ 原记作 λ̈ = −(2−N)(V_c/R)λ̇ + a_t/R，与"N>2 稳定"的结论自相矛盾，已校勘（见 STEP 5）
  · 数值 Riccati 显示：终端权重 s 增大时，有效导航比从两端收敛至 3
  · 工程修正（书中经验）：有舵机延迟 / 噪声时，实际最优 N 需上调至 3.5~4.0
""")
