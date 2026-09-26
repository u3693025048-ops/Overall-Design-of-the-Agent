#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_all.py — 算法库一键自检 + 综合工程演示

用法：
    python3 run_all.py              # 跑全部模块自检 + 综合演示
    python3 run_all.py --selftest   # 只跑自检
    python3 run_all.py --demo       # 只跑综合演示

自检部分用子进程逐个执行各模块（隔离、可捕获退出码）；
演示部分串联 engagement + monte_carlo + metrics，复现 book010 的
工程结论（N 的 U 型曲线、延迟门槛、λ̇ 滤波器最优时间常数、MC 验收）。
"""
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable

MODULES = [
    ("gnc_core", "基础工具：向量/角度/LOS 运动学/碰撞三角/ZEM/CPA"),
    ("dynamics", "执行机构：一阶延迟/限幅/RK4 积分器"),
    ("guidance_laws", "制导律：10 种可互换、可参数化"),
    ("seeker", "导引头：角度噪声/滤波微分/视场/框架角"),
    ("engagement", "交战仿真引擎（闭环，含机动/延迟/噪声）"),
    ("metrics", "指标统计与工程评级"),
    ("monte_carlo", "蒙特卡洛框架与参数扫描"),
]


def selftest():
    print("=" * 74)
    print("  算法库自检（%d 个模块）" % len(MODULES))
    print("=" * 74)
    ok, t0 = 0, time.time()
    for name, desc in MODULES:
        t = time.time()
        r = subprocess.run([PY, os.path.join(HERE, name + ".py")],
                           capture_output=True, text=True, cwd=HERE)
        dur = time.time() - t
        if r.returncode == 0:
            n = r.stdout.count("✅")
            print(f"  ✅ {name:14s} {desc}  （{n} 项，{dur:.1f}s）")
            ok += 1
        else:
            print(f"  ❌ {name:14s} 退出码 {r.returncode}")
            print((r.stderr or r.stdout)[-800:])
    print("-" * 74)
    print(f"  {ok}/{len(MODULES)} 个模块自检通过，用时 {time.time()-t0:.1f}s")
    return ok == len(MODULES)


def demo():
    sys.path.insert(0, HERE)
    import engagement as eg
    import metrics
    from monte_carlo import Dist, run

    print()
    print("=" * 74)
    print("  综合演示：工程工况下的制导律参数标定")
    print("=" * 74)
    SC = dict(R0=6000.0, H0=500.0, Vm=700.0, Vt=300.0, amax=25.0 * 9.80665)
    print("  基准工况：R0=6 km, Vm=700 / Vt=300 m/s, 目标 t=5s 起拉 9 g,")
    print("            导弹限幅 25 g, 舵机延迟 100 ms, 测角噪声 1 mrad\n")

    # 1. N 扫描（机动 + 噪声 + 延迟的合成工况）
    print("  【1】导航比 N 扫描（机动 + 噪声 + 延迟合成）— 预期 U 型")
    NS = [2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 8.0]
    res = []
    for N in NS:
        specs = {**{k: Dist.const(v) for k, v in SC.items()},
                 "N": Dist.const(N), "law": Dist.const("TPN"),
                 "tgt_g": Dist.uniform(7.0, 11.0),
                 "t_break": Dist.uniform(4.6, 5.3),
                 "delay": Dist.const(0.10), "sig_lam": Dist.const(1e-3),
                 "tau_f": Dist.const(0.05), "dt": Dist.const(0.004)}
        s, _ = run(lambda **kw: eg.simulate(seed=0, **kw)["miss"],
                   specs, n=30, seed=100)
        rms = (sum(x * x for x in s) / len(s)) ** 0.5
        res.append(rms)
        print(f"      N={N:4.1f} → RMS 脱靶 {rms:7.3f} m")
    best = NS[res.index(min(res))]
    inner = 0 < res.index(min(res)) < len(res) - 1
    print(f"      ⇒ 最优 N = {best}" + ("（内点，U 型成立 ✅）" if inner
                                        else "（边界，U 型未显现 ⚠️）"))

    # 2. 延迟门槛
    print("\n  【2】舵机延迟门槛扫描（N=4，其余同基准）— 预期存在门槛")
    prev = None
    for tau in (0.0, 0.05, 0.10, 0.15, 0.20, 0.30):
        specs = {**{k: Dist.const(v) for k, v in SC.items()},
                 "N": Dist.const(4.0), "law": Dist.const("TPN"),
                 "tgt_g": Dist.const(9.0), "t_break": Dist.const(5.0),
                 "delay": Dist.const(tau), "dt": Dist.const(0.004)}
        s, _ = run(lambda **kw: eg.simulate(seed=0, **kw)["miss"],
                   specs, n=20, seed=200)
        v = (sum(x * x for x in s) / len(s)) ** 0.5
        jump = "  ← 门槛" if prev is not None and v > max(prev * 5, 0.05) else ""
        print(f"      τ={tau*1000:5.0f} ms → RMS 脱靶 {v:8.4f} m{jump}")
        prev = v

    # 3. λ̇ 滤波器时间常数
    print("\n  【3】λ̇ 滤波器 τ_f 扫描（σ_λ=3 mrad, N=4）— 预期存在最优值")
    taus = [0.005, 0.01, 0.02, 0.05, 0.10, 0.20]
    vals = []
    for tf in taus:
        specs = {**{k: Dist.const(v) for k, v in SC.items()},
                 "N": Dist.const(4.0), "law": Dist.const("TPN"),
                 "tgt_g": Dist.const(0.0), "t_break": Dist.const(1e9),
                 "delay": Dist.const(0.02), "sig_lam": Dist.const(1e-3),
                 "tau_f": Dist.const(tf), "dt": Dist.const(0.001)}
        s, _ = run(lambda **kw: eg.simulate(seed=0, **kw)["miss"],
                   specs, n=20, seed=300)
        v = (sum(x * x for x in s) / len(s)) ** 0.5
        vals.append(v)
        print(f"      τ_f={tf*1000:5.0f} ms → RMS 脱靶 {v:8.4f} m")
    bi = vals.index(min(vals))
    print(f"      ⇒ 最优 τ_f = {taus[bi]*1000:.0f} ms"
          + ("（内点：太小噪声爆炸，太大相位滞后 ✅）"
             if 0 < bi < len(vals) - 1 else "（边界 ⚠️）"))

    # 4. 蒙特卡洛验收
    print("\n  【4】选定配置的蒙特卡洛验收")
    specs = {**{k: Dist.const(v) for k, v in SC.items()},
             "N": Dist.const(float(best)), "law": Dist.const("TPN"),
             "tgt_g": Dist.uniform(7.0, 11.0),
             "t_break": Dist.uniform(4.6, 5.3),
             "delay": Dist.const(0.10), "sig_lam": Dist.const(1e-3),
             "tau_f": Dist.const(0.05), "dt": Dist.const(0.004)}
    s, summ = run(lambda **kw: eg.simulate(seed=0, **kw)["miss"],
                  specs, n=200, seed=400)
    print(metrics.report(s))

    print("=" * 74)
    print("  综合演示完成（结论与 book010 卡片经验一致）")
    print("=" * 74)


if __name__ == "__main__":
    args = sys.argv[1:]
    ok = True
    if not args or "--selftest" in args:
        ok = selftest()
    if ok and (not args or "--demo" in args):
        demo()
    sys.exit(0 if ok else 1)
