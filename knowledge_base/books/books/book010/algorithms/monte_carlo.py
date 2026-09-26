#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
monte_carlo.py — 蒙特卡洛试验框架

提供参数分布建模、批量试验、一维/二维参数扫描与统计分析。
与 engagement.monte_carlo（专用打靶）不同，这里是**通用框架**，
可对接任意 sim_fn(params) -> scalar。

核心能力
--------
    Dist        : 参数分布（const / uniform / normal / loguniform）
    run()       : 批量试验，返回样本与统计摘要
    sweep_1d()  : 一维扫描（如 N 扫描、延迟扫描、τ_f 扫描）
    sweep_2d()  : 二维扫描（如 N × 延迟交互面）

经验（book010 卡 B-05）
----------------------
* 样本量 ≥500~1000 次，且**必须覆盖边界工况**。
* 拉丁超立方/分层采样比纯随机收敛更快（方差更小）。
* 结果必须报告均值 + RMS + P95 + CEP，**只看均值会漏掉尾部风险**。

纯标准库。自检：python3 monte_carlo.py
"""
import math
import random

import metrics

__all__ = ["Dist", "run", "sweep_1d", "sweep_2d", "latin_hypercube"]


# ---------------------------------------------------------------- 分布
class Dist:
    """参数分布。

    kind:
        'const'      : 固定值（value）
        'uniform'    : 均匀分布 [lo, hi]
        'normal'     : 正态分布 N(mu, sigma)
        'loguniform' : 对数均匀分布（跨数量级参数，如噪声强度）
    """

    def __init__(self, kind="const", **kw):
        self.kind = kind
        self.kw = kw

    def sample(self, rng):
        k = self.kind
        if k == "const":
            return self.kw["value"]
        if k == "uniform":
            return rng.uniform(self.kw["lo"], self.kw["hi"])
        if k == "normal":
            return rng.gauss(self.kw.get("mu", 0.0), self.kw["sigma"])
        if k == "loguniform":
            lo = math.log10(self.kw["lo"])
            hi = math.log10(self.kw["hi"])
            return 10 ** rng.uniform(lo, hi)
        raise ValueError(f"未知分布：{k}")

    @staticmethod
    def const(v):
        return Dist("const", value=v)

    @staticmethod
    def uniform(lo, hi):
        return Dist("uniform", lo=lo, hi=hi)

    @staticmethod
    def normal(mu, sigma):
        return Dist("normal", mu=mu, sigma=sigma)


def latin_hypercube(n, dims, rng):
    """拉丁超立方采样：每维分成 n 个等概率区间，各取一次。

    比纯随机均匀收敛更快、方差更小（卡 B-05：PSO 初始化亦推荐）。
    Returns: list of list，shape (n, dims)，每维的值均匀分布在 [0,1]
    """
    cols = []
    for _ in range(dims):
        idx = list(range(n))
        rng.shuffle(idx)
        cols.append([(idx[i] + rng.random()) / n for i in range(n)])
    return [[cols[d][i] for d in range(dims)] for i in range(n)]


# ---------------------------------------------------------------- 试验
def run(sim_fn, specs, n=200, seed=0, stratified=True):
    """批量蒙特卡洛试验。

    Args:
        sim_fn: 函数 f(**params) -> scalar（通常为脱靶量）
        specs: {参数名: Dist}
        n: 样本数
        stratified: 是否用拉丁超立方（对随机参数）替代纯随机
    Returns:
        (samples, summary)  summary 来自 metrics.summarize
    """
    rng = random.Random(seed)
    keys = list(specs)
    rand_keys = [k for k in keys if specs[k].kind != "const"]

    if stratified and rand_keys and n > 1:
        lhs = latin_hypercube(n, len(rand_keys), rng)
    else:
        lhs = None

    samples = []
    for i in range(n):
        params = {}
        for j, k in enumerate(rand_keys):
            d = specs[k]
            if lhs is not None:
                u = lhs[i][j]
                if d.kind == "uniform":
                    params[k] = d.kw["lo"] + u * (d.kw["hi"] - d.kw["lo"])
                elif d.kind == "loguniform":
                    lo = math.log10(d.kw["lo"])
                    hi = math.log10(d.kw["hi"])
                    params[k] = 10 ** (lo + u * (hi - lo))
                else:
                    params[k] = d.sample(rng)
            else:
                params[k] = d.sample(rng)
        for k in keys:
            if k not in params:
                params[k] = specs[k].sample(rng)
        samples.append(sim_fn(**params))
    return samples, metrics.summarize(samples)


def sweep_1d(sim_fn, base, param, values, n=100, seed=0, stat="rms"):
    """一维参数扫描。返回 [(value, 统计量), ...]。"""
    out = []
    for v in values:
        b = dict(base)
        b[param] = v
        specs = {k: (x if isinstance(x, Dist) else Dist.const(x))
                 for k, x in b.items()}
        s, summ = run(sim_fn, specs, n=n, seed=seed)
        out.append((v, summ[stat]))
    return out


def sweep_2d(sim_fn, base, p1, v1, p2, v2, n=50, seed=0, stat="rms"):
    """二维参数扫描（交互面）。返回 {(a, b): 统计量}。"""
    out = {}
    for a in v1:
        for b in v2:
            bb = dict(base)
            bb[p1], bb[p2] = a, b
            specs = {k: (x if isinstance(x, Dist) else Dist.const(x))
                     for k, x in bb.items()}
            s, summ = run(sim_fn, specs, n=n, seed=seed)
            out[(a, b)] = summ[stat]
    return out


# ---------------------------------------------------------------- 自检
def _self_test():
    print("=" * 72)
    print("  monte_carlo 自检")
    print("=" * 72)
    rng = random.Random(1)

    # 1. 分布采样统计正确性
    d = Dist.uniform(0.0, 10.0)
    xs = [d.sample(rng) for _ in range(50000)]
    m = sum(xs) / len(xs)
    assert abs(m - 5.0) < 0.1, f"均匀分布均值错误：{m}"
    dn = Dist.normal(2.0, 0.5)
    ys = [dn.sample(rng) for _ in range(50000)]
    mn = sum(ys) / len(ys)
    sd = math.sqrt(sum((y - mn) ** 2 for y in ys) / len(ys))
    assert abs(mn - 2.0) < 0.02 and abs(sd - 0.5) < 0.02, f"{mn}/{sd}"
    print(f"  [1] 分布采样            : U(0,10) 均值 {m:.3f} / "
          f"N(2,0.5) 均值 {mn:.3f} 标准差 {sd:.3f}  ✅")

    # 2. 拉丁超立方：① 分层性质 ② 均值估计方差远小于纯随机
    bins = [0] * 20
    for x in [g[0] for g in latin_hypercube(20, 1, random.Random(5))]:
        bins[min(int(x * 20), 19)] += 1
    assert all(b == 1 for b in bins), f"LHS 分层不均衡：{bins}"
    print("  [2] LHS 分层性质        : n=20 时每区间恰好 1 个样本  ✅")

    def _var(xs):
        m = sum(xs) / len(xs)
        return sum((x - m) ** 2 for x in xs) / len(xs)

    N, REPS = 50, 200
    vl = [sum(g[0] for g in latin_hypercube(N, 1, random.Random(s))) / N
          for s in range(REPS)]
    vm = [sum(random.Random(s).random() for _ in range(N)) / N
          for s in range(REPS)]
    v_lhs, v_mc = _var(vl), _var(vm)
    assert v_lhs < v_mc / 100.0, f"LHS 未显著降低估计方差：{v_lhs} vs {v_mc}"
    print(f"  [2b]LHS vs 纯随机收敛   : 样本均值方差 {v_lhs:.3e} vs "
          f"{v_mc:.3e}（改善 {v_mc/v_lhs:.0f}×）✅")

    # 3. run()：固定参数应返回确定值，随机参数应产生散布
    def sim(a, b=0.0):
        return a + b
    s_const, sm = run(sim, {"a": Dist.const(3.0), "b": Dist.const(1.0)},
                      n=20, seed=0)
    assert set(s_const) == {4.0}, f"固定参数结果不一致：{set(s_const)}"
    s_rand, sm2 = run(sim, {"a": Dist.uniform(0.0, 1.0)}, n=500, seed=0)
    assert sm2["std"] > 0.2, f"随机参数未产生散布：{sm2['std']}"
    print(f"  [3] run() 参数处理      : 定值确定 / 随机散布 σ={sm2['std']:.3f} ✅")

    # 4. MC 收敛：样本量增大 → 均值误差下降（约 1/√n；关闭分层以观察本征收敛）
    def sim2(x):
        return x
    errs = []
    for n in (100, 400, 1600):
        s, _ = run(sim2, {"x": Dist.uniform(0.0, 1.0)}, n=n, seed=7,
                   stratified=False)
        errs.append(abs(sum(s) / n - 0.5))
    assert errs[0] > errs[-1], f"MC 未随样本量收敛：{errs}"
    print("  [4] MC 收敛性（1/√n）   : n=100/400/1600 均值误差 "
          + " → ".join(f"{e:.5f}" for e in errs) + "  ✅")

    # 5. sweep_1d：应能检出单调趋势
    def sim3(gain):
        return 1.0 / gain
    sw = sweep_1d(sim3, {"gain": 1.0}, "gain", [1.0, 2.0, 4.0, 8.0], n=5)
    vals = [v for _, v in sw]
    assert all(vals[i] > vals[i + 1] for i in range(len(vals) - 1))
    print("  [5] sweep_1d 趋势检出   : gain 1→8  RMS "
          + " → ".join(f"{v:.3f}" for v in vals) + "  ✅")

    # 6. sweep_2d：网格完整性
    sw2 = sweep_2d(sim, {"a": 1.0, "b": 2.0}, "a", [1.0, 2.0], "b",
                   [1.0, 2.0, 3.0], n=4)
    assert len(sw2) == 6 and sw2[(2.0, 3.0)] == 5.0
    print(f"  [6] sweep_2d 网格       : {len(sw2)} 个组合，取值正确  ✅")

    # 7. 统计量透传（P95/CEP 可用）
    s, sm3 = run(sim2, {"x": Dist.uniform(0.0, 1.0)}, n=2000, seed=3)
    assert abs(sm3["p95"] - 0.95) < 0.03 and abs(sm3["cep"] - 0.5) < 0.04
    print(f"  [7] 统计量透传          : P95 {sm3['p95']:.3f} / "
          f"CEP {sm3['cep']:.3f}（理论 0.95 / 0.5）✅")

    # 8. 可复现性（同种子同结果）
    a1, _ = run(sim2, {"x": Dist.uniform(0, 1)}, n=50, seed=99)
    a2, _ = run(sim2, {"x": Dist.uniform(0, 1)}, n=50, seed=99)
    assert a1 == a2, "同种子结果不可复现"
    print("  [8] 可复现性            : 同 seed 结果完全一致  ✅")

    print("=" * 72)
    print("  monte_carlo：8 项自检全部通过")
    print("=" * 72)


if __name__ == "__main__":
    _self_test()
