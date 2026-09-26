#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
metrics.py — 指标统计与工程评级

把 book010【A++ - 经验】中的工程验收阈值固化为可执行判据（卡 B-05）：

    脱靶量      <1 m 优秀 ｜ 1~3 m 良好 ｜ 3~5 m 及格 ｜ >5 m 不及格
    鲁棒性 σ/μ  <0.3 优秀 ｜ 0.3~0.5 良好 ｜ >0.5 需改进
    带干扰验收  带干扰脱靶均值 ≤ 理想值 2 倍可接受，>3 倍必须回炉
    样本量      ≥500~1000 次（少于此散布统计不可信）

纯标准库。自检：python3 metrics.py
"""
import math

__all__ = ["quantile", "summarize", "cep", "grade_miss",
           "grade_robustness", "acceptance", "report"]


def quantile(samples, q):
    """线性插值分位数。q ∈ [0, 1]。"""
    if not samples:
        raise ValueError("样本为空")
    s = sorted(samples)
    if len(s) == 1:
        return s[0]
    pos = q * (len(s) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def cep(samples):
    """圆概率误差 CEP：50% 样本落在其中的半径（= |脱靶量| 的中位数）。"""
    return quantile([abs(x) for x in samples], 0.5)


def summarize(samples):
    """脱靶量样本的完整统计摘要。"""
    n = len(samples)
    mean = sum(samples) / n
    var = sum((x - mean) ** 2 for x in samples) / max(n - 1, 1)
    std = math.sqrt(var)
    return {
        "n": n,
        "mean": mean,
        "std": std,
        "rms": math.sqrt(sum(x * x for x in samples) / n),
        "min": min(samples),
        "max": max(samples),
        "p50": quantile(samples, 0.50),
        "p95": quantile(samples, 0.95),
        "p99": quantile(samples, 0.99),
        "cep": cep(samples),
        "cv": (std / mean) if abs(mean) > 1e-12 else float("inf"),
        "p95_over_mean": (quantile(samples, 0.95) / mean
                          if abs(mean) > 1e-12 else float("inf")),
    }


def grade_miss(miss_mean):
    """脱靶量评级（卡 B-05）。返回 (等级, 说明)。"""
    if miss_mean < 1.0:
        return "优秀", "<1 m"
    if miss_mean < 3.0:
        return "良好", "1~3 m"
    if miss_mean < 5.0:
        return "及格", "3~5 m"
    return "不及格", ">5 m（需回炉）"


def grade_robustness(cv):
    """鲁棒性评级（σ/μ，卡 B-05）。"""
    if cv < 0.3:
        return "优秀", "σ/μ < 0.3"
    if cv <= 0.5:
        return "良好", "σ/μ 0.3~0.5"
    return "需改进", "σ/μ > 0.5"


def acceptance(ideal_mean, disturbed_mean):
    """带干扰验收判据（卡 B-05）。返回 (结论, 倍率)。

    ≤2 倍：可接受；2~3 倍：勉强；>3 倍：必须回炉重调。
    """
    if abs(ideal_mean) < 1e-12:
        return ("理想值过小，倍率判据不适用", float("inf"))
    k = disturbed_mean / ideal_mean
    if k <= 2.0:
        return "可接受", k
    if k <= 3.0:
        return "勉强（建议优化）", k
    return "不合格（必须回炉）", k


def report(samples, ideal_mean=None, title="脱靶量"):
    """生成一段可读的评估文本。"""
    s = summarize(samples)
    g, gd = grade_miss(abs(s["mean"]))
    r, rd = grade_robustness(s["cv"])
    L = [f"  {title}统计（n={s['n']}）",
         f"    均值 {s['mean']:.4f} m ｜ RMS {s['rms']:.4f} m ｜ "
         f"标准差 {s['std']:.4f} m",
         f"    CEP  {s['cep']:.4f} m ｜ P95 {s['p95']:.4f} m ｜ "
         f"P99 {s['p99']:.4f} m",
         f"    精度评级：{g}（{gd}）",
         f"    鲁棒性　：{r}（{rd}，σ/μ={s['cv']:.3f}）",
         f"    P95/均值：{s['p95_over_mean']:.2f}×"]
    if ideal_mean:
        c, k = acceptance(ideal_mean, s["mean"])
        L.append(f"    带干扰验收：{c}（{k:.2f}× 理想值 {ideal_mean:.4f} m）")
    if s["n"] < 500:
        L.append(f"    ⚠️ 样本量 {s['n']} < 500，散布统计仅供参考"
                 f"（卡 B-05 建议 ≥500~1000）")
    return "\n".join(L)


# ---------------------------------------------------------------- 自检
def _self_test():
    print("=" * 72)
    print("  metrics 自检")
    print("=" * 72)

    # 1. 分位数：均匀分布 U(0,10) 的解析分位数
    xs = [i * 0.01 for i in range(1001)]        # 0, 0.01, ..., 10
    for q, ref in ((0.5, 5.0), (0.95, 9.5), (0.0, 0.0), (1.0, 10.0)):
        v = quantile(xs, q)
        assert abs(v - ref) < 1e-6, f"q={q} 分位数错误：{v} vs {ref}"
    print("  [1] 分位数（U(0,10)）   : p50=5.0 / p95=9.5 / 端点正确  ✅")

    # 2. 均值、标准差、RMS
    s = summarize(xs)
    assert abs(s["mean"] - 5.0) < 1e-6, f"均值错误：{s['mean']}"
    assert abs(s["std"] - math.sqrt(100.0 / 12.0)) < 0.02, f"标准差错误：{s['std']}"
    assert abs(s["rms"] - math.sqrt(100.0 / 3.0)) < 0.02, f"RMS 错误：{s['rms']}"
    print(f"  [2] 均值/标准差/RMS     : {s['mean']:.4f} / {s['std']:.4f} / "
          f"{s['rms']:.4f}（理论 5 / 2.8868 / 5.7735）✅")

    # 3. CEP：对称分布 |x| 的中位数
    sym = [-1.0, -0.5, 0.0, 0.5, 1.0]          # |x| = [1, .5, 0, .5, 1]
    assert abs(cep(sym) - 0.5) < 1e-12, f"CEP 错误：{cep(sym)}"
    print(f"  [3] CEP（对称样本）     : {cep(sym):.4f}（|x| 中位数 0.5）✅")

    # 4. 评级阈值
    cases = [(0.5, "优秀"), (2.0, "良好"), (4.0, "及格"), (8.0, "不及格")]
    for v, exp in cases:
        g, _ = grade_miss(v)
        assert g == exp, f"脱靶 {v} m 评级错误：{g} vs {exp}"
    print("  [4] 脱靶评级阈值        : 0.5m 优秀 / 2m 良好 / 4m 及格 / "
          "8m 不及格  ✅")

    # 5. 鲁棒性评级
    for cv, exp in ((0.2, "优秀"), (0.4, "良好"), (0.8, "需改进")):
        r, _ = grade_robustness(cv)
        assert r == exp, f"σ/μ={cv} 评级错误：{r}"
    print("  [5] 鲁棒性评级          : 0.2 优秀 / 0.4 良好 / 0.8 需改进  ✅")

    # 6. 带干扰验收
    for k, exp in ((1.5, "可接受"), (2.5, "勉强"), (4.0, "不合格")):
        c, kk = acceptance(1.0, k)
        assert exp in c, f"倍率 {k} 判定错误：{c}"
    print("  [6] 带干扰验收          : 1.5× 可接受 / 2.5× 勉强 / "
          "4.0× 不合格  ✅")

    # 7. 报告生成（含小样本告警）
    txt = report([0.4, 0.6, 0.5, 0.7, 0.3], ideal_mean=0.4)
    assert "样本量 5 < 500" in txt and "优秀" in txt
    print("  [7] 评估报告            : 含精度/鲁棒性/验收/小样本告警  ✅")

    # 8. 空样本与单点样本的健壮性
    assert summarize([2.0])["mean"] == 2.0
    try:
        quantile([], 0.5)
        raise AssertionError("空样本未抛异常")
    except ValueError:
        pass
    print("  [8] 边界健壮性          : 单点可用 / 空样本抛 ValueError  ✅")

    print("=" * 72)
    print("  metrics：8 项自检全部通过")
    print("=" * 72)


if __name__ == "__main__":
    _self_test()
