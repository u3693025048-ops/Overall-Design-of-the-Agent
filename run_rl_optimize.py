#!/usr/bin/env python3
"""
直接用 PPO-RL 算法优化 MC_gongkuang_simulation_robust_all.m 的独立脚本。

功能
----
• 不依赖 Hermes Agent / LLM，直接使用 MatlabRLOptimizer
• 读取 config.yaml 中的 simulation / matlab_rl_optimizer 配置
• 支持命令行指定工况、轮次、Nmc、脚本路径
• 支持命令行输入指标要求（--hitrate / --missmean / --peak-n / --pm）
• Pre-RL 阶段先跑一次基线仿真输出 PeakN / PM / GM 等指标；基线满足全部要求则跳过 RL
• 优化结束后打印最优参数并将结果写入 ./rl_results/（不写入 ParameterExperience）

用法示例
--------
# 默认工况 A 全子工况，20 轮，Nmc=10：
python run_rl_optimize.py

# 指定工况 A 子工况 1,2 + B 子工况 1；50 轮；Nmc=20：
python run_rl_optimize.py --conditions "A:1,2;B:1" --episodes 50 --nmc 20

# 指定脚本（不用默认模板）：
python run_rl_optimize.py --script ./matlab_scripts/my_script.m

# 完整参数：
python run_rl_optimize.py --conditions "A:1,2,3;B:1" --episodes 30 --nmc 10
                          --script ./knowledge_base/matlab/robust_analysis/MC_gongkuang_simulation_robust_all.m
                          --hitrate 90 --missmean 5.0 --peak-n 20.0 --pm 30.0
"""

import argparse
import asyncio
import json
import logging
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# ── 保证项目根目录在 sys.path ─────────────────────────────────────────────────
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_rl_optimize")

# ── KB 模板路径 ───────────────────────────────────────────────────────────────
KB_TEMPLATE = str(
    _SCRIPT_DIR / "knowledge_base" / "matlab" / "robust_analysis"
    / "MC_gongkuang_simulation_robust_all.m"
)


# ─────────────────────────────────────────────────────────────────────────────
# 工况字符串解析
# ─────────────────────────────────────────────────────────────────────────────

def parse_conditions_arg(cond_str: Optional[str]) -> Dict[str, List[int]]:
    """
    解析命令行工况参数，返回 {category: [subcase, ...]} 字典。

    格式: "A:1,2,3;B:1;C"  （不指定子工况 → 全子工况）
    默认（不传）: {"A": [1,2,3,4,5,6]}
    """
    if not cond_str:
        return {"A": list(range(1, 7))}

    MAX_SUB = {"A": 6, "B": 6, "C": 4, "D": 4, "E": 3, "F": 3}
    result: Dict[str, List[int]] = {}
    for token in re.split(r"[;,\s]+", cond_str.strip()):
        if not token:
            continue
        m = re.match(r"([A-Fa-f])(?::(.+))?", token)
        if not m:
            continue
        cat = m.group(1).upper()
        subs_str = m.group(2)
        max_sub = MAX_SUB.get(cat, 6)
        if subs_str:
            subs = [int(x) for x in re.split(r"[,\s]+", subs_str) if x.isdigit()]
        else:
            subs = list(range(1, max_sub + 1))
        result[cat] = subs

    return result if result else {"A": list(range(1, 7))}


def build_conditions_str(conditions: Dict[str, List[int]]) -> str:
    """
    将工况字典转为 MATLAB patch 字符串，所有未提及的类别置为 false。
    e.g. {"A":[1,2]} → "run_A=true;sub_A=[1,2];run_B=false;...run_F=false;"
    """
    CATEGORY_DEFS = {
        "A": {"max_sub": 6},
        "B": {"max_sub": 6},
        "C": {"max_sub": 4},
        "D": {"max_sub": 4},
        "E": {"max_sub": 3},
        "F": {"max_sub": 3},
    }
    parts: List[str] = []
    for cat, info in CATEGORY_DEFS.items():
        if cat in conditions:
            subs = conditions[cat]
            max_sub = info["max_sub"]
            if sorted(subs) == list(range(1, max_sub + 1)):
                parts.append(f"run_{cat}=true;sub_{cat}=[];")
            else:
                sub_str = ",".join(str(s) for s in sorted(subs))
                parts.append(f"run_{cat}=true;sub_{cat}=[{sub_str}];")
        else:
            parts.append(f"run_{cat}=false;")
    return "".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# 主优化流程
# ─────────────────────────────────────────────────────────────────────────────

async def run_optimization(args):
    from multi_agent.config_loader import get_config
    from multi_agent.simulation.guidance_simulator import GuidanceSimulator
    from multi_agent.rl.matlab_rl_optimizer import (
        MatlabRLOptimizer,
        ALL_TUNABLE_PARAM_SPECS,
        build_matlab_conditions_str,
        parse_mission_conditions,
    )

    # ── 1. 加载配置 ───────────────────────────────────────────────────────────
    cfg = get_config()
    sim_cfg = cfg.simulation
    rl_cfg  = cfg.matlab_rl_optimizer

    max_episodes  = args.episodes   or rl_cfg.max_episodes
    nmc_per_eval  = args.nmc        or rl_cfg.nmc_per_eval
    peak_n_max    = (args.peak_n if args.peak_n is not None
                     else (args.peak_n_max if args.peak_n_max is not None
                           else rl_cfg.peak_n_max))
    miss_threshold = args.missmean if args.missmean is not None else args.miss_threshold
    req_hitrate   = args.hitrate
    req_missmean  = args.missmean
    req_peak_n    = args.peak_n
    req_pm        = args.pm

    _any_reqs     = any(v is not None for v in [req_hitrate, req_missmean, req_peak_n, req_pm])
    _explicit_eps = args.episodes is not None
    if _any_reqs and not _explicit_eps:
        _mode = 1  # 指标驱动：按轮次优化，满足指标则提前退出
    elif not _any_reqs:
        _mode = 2  # 轮次驱动：运行完整轮次，无提前退出
    else:
        _mode = 3  # 双约束：运行完整轮次，结束后评估指标要求

    _MODE_LABELS = {
        1: "模式1 [指标驱动] — 满足指标要求则提前退出",
        2: "模式2 [轮次驱动] — 运行完整轮次，无提前退出",
        3: "模式3 [双约束]   — 运行完整轮次后评估指标要求",
    }

    print("\n" + "="*60)
    print("  PPO-RL 优化器  —  MC_gongkuang_simulation_robust_all")
    print("="*60)
    print(f"  运行模式     : {_MODE_LABELS[_mode]}")
    print(f"  引擎         : {sim_cfg.engine}")
    print(f"  最大轮次     : {max_episodes}")
    print(f"  每轮 Nmc     : {nmc_per_eval}")
    print(f"  PeakN 上限   : {peak_n_max}")
    _reqs = []
    if req_hitrate  is not None: _reqs.append(f"hitrate≥{req_hitrate}%")
    if req_missmean is not None: _reqs.append(f"miss≤{req_missmean}m")
    if req_peak_n   is not None: _reqs.append(f"PeakN≤{req_peak_n}g")
    if req_pm       is not None: _reqs.append(f"PM≥{req_pm}°")
    if _reqs:
        print(f"  指标要求     : {', '.join(_reqs)}")
    if args.resume_checkpoint:
        print(f"  恢复检查点   : {args.resume_checkpoint}")

    # ── 2. 工况解析 ───────────────────────────────────────────────────────────
    mission_conditions = parse_conditions_arg(args.conditions)
    cond_str = build_conditions_str(mission_conditions)
    print(f"  工况         : {dict(mission_conditions)}")
    print(f"  Conditions   : {cond_str}")
    print("="*60 + "\n")

    # ── 3. 脚本路径 ───────────────────────────────────────────────────────────
    script_path = args.script or KB_TEMPLATE
    if not os.path.exists(script_path):
        print(f"[ERROR] 脚本未找到: {script_path}")
        print(f"        请确认路径或省略 --script 以使用 KB 模板。")
        return

    print(f"[Step 1] 使用脚本: {script_path}")

    # ── 4. 建立仿真器 & RL 优化器 ─────────────────────────────────────────────
    simulator = GuidanceSimulator(
        output_dir="./guidance_output",
        engine=sim_cfg.engine,
        octave_path=sim_cfg.octave_path,
        matlab_path=sim_cfg.matlab_path,
    )

    rl_optimizer = MatlabRLOptimizer(
        simulator=simulator,
        parameter_experience=None,
        max_episodes=max_episodes,
        nmc_per_eval=nmc_per_eval,
        episodes_per_update=rl_cfg.episodes_per_update,
        hidden_dim=rl_cfg.hidden_dim,
        lr_actor=rl_cfg.lr_actor,
        lr_critic=rl_cfg.lr_critic,
        gamma=rl_cfg.gamma,
        clip_ratio=rl_cfg.clip_ratio,
        peak_n_max=peak_n_max,
        peak_n_penalty=rl_cfg.peak_n_penalty,
        entropy_coef=getattr(rl_cfg, "entropy_coef", 0.05),
        pitch_log_std=rl_cfg.exploration.pitch_log_std,
        other_log_std=rl_cfg.exploration.other_log_std,
        warm_start_std_scale=rl_cfg.exploration.warm_start_std_scale,
        reflect_every=rl_cfg.early_stop.reflect_every,
        early_stop_enabled=rl_cfg.early_stop.enabled,
        early_stop_min_episodes=rl_cfg.early_stop.min_episodes,
        early_stop_patience=rl_cfg.early_stop.patience,
        early_stop_use_rule_first=rl_cfg.early_stop.use_rule_first,
        early_stop_use_multi_criteria_best=rl_cfg.early_stop.use_multi_criteria_best,
        fre_config=cfg.fre,
        reward_weights=rl_cfg.reward_weights.to_dict(),
        checkpoint_enabled=rl_cfg.checkpoint.enabled,
        checkpoint_dir=rl_cfg.checkpoint.dir,
        checkpoint_save_every=rl_cfg.checkpoint.save_every_episodes,
        checkpoint_keep_last_n=rl_cfg.checkpoint.keep_last_n,
        rolling_window=rl_cfg.checkpoint.rolling_window,
        stats_log_every=rl_cfg.checkpoint.stats_log_every,
        jsonl_log_dir=rl_cfg.checkpoint.jsonl_log_dir,
    )

    # ── 5. 确定运行脚本（不依赖 ParameterExperience）──────────────────────────
    _rl_script = script_path

    # ── 6. 提取基线参数 ────────────────────────────────────────────────────────
    print(f"\n[Step 3] 从脚本中提取基线参数...")
    rl_optimizer.extract_params_from_script(_rl_script)
    base = rl_optimizer._base_auto
    print(f"  N_pn={base.get('N_pn', 4):.2f}  "
          f"w1={base.get('w1', 40):.2f}  zeta1={base.get('zeta1', 0.75):.3f}")

    # ── 7. Pre-RL 基线仿真（含 PeakN）─────────────────────────────────────────
    print(f"\n[Pre-RL] 基线仿真 Nmc={nmc_per_eval}，评估初始参数性能...")
    pre_metrics: Dict = {}
    try:
        pre_metrics = await rl_optimizer._run_simulation_with_params(
            rl_optimizer._base_auto, mission_conditions, _rl_script, nmc_per_eval
        )
        print(f"  [Pre-RL] hit={pre_metrics.get('hit_rate',0):.1f}%  "
              f"miss={pre_metrics.get('miss_distance',99):.3f}m  "
              f"PeakN={pre_metrics.get('peak_n',0):.2f}g  "
              f"PM={pre_metrics.get('pitch_PM',0):.1f}°  "
              f"GM={pre_metrics.get('pitch_GM',0):.1f}dB  "
              f"BW={pre_metrics.get('pitch_BW',0):.1f}rad/s")
    except Exception as _e:
        print(f"  [Pre-RL] 基线仿真异常: {_e}")

    # ── 8. 快速判断是否已满足（可选指标要求）────────────────────────────────
    if _mode == 1 and pre_metrics:
        _ok = True
        _met: List[str] = []
        if req_hitrate is not None:
            _v = pre_metrics.get("hit_rate", 0.0)
            if _v >= req_hitrate:  _met.append(f"hit={_v:.1f}%≥{req_hitrate}%")
            else: _ok = False
        if req_missmean is not None:
            _v = pre_metrics.get("miss_distance", float("inf"))
            if _v <= req_missmean: _met.append(f"miss={_v:.2f}m≤{req_missmean}m")
            else: _ok = False
        if req_peak_n is not None:
            _v = pre_metrics.get("peak_ny", pre_metrics.get("peak_n", 0.0))
            if _v > 0 and _v > req_peak_n: _ok = False
            elif _v > 0:                   _met.append(f"PeakN={_v:.2f}g≤{req_peak_n}g")
        if req_pm is not None:
            _v = pre_metrics.get("pitch_PM", 0.0)
            if _v >= req_pm:       _met.append(f"PM={_v:.1f}°≥{req_pm}°")
            else: _ok = False
        if _ok:
            print(f"\n  [Pre-RL] ✓ 基线参数已满足全部指标要求: {', '.join(_met) or '无条件'}，跳过 RL。")
            _print_results(base, pre_metrics, total_episodes=0)
            return

    # ── 9. 运行 RL 优化 ────────────────────────────────────────────────────────
    _metric_reqs = ({"hitrate": req_hitrate, "missmean": req_missmean,
                     "peak_n": req_peak_n, "pm": req_pm}
                    if _mode == 1 else None)    # 仅模式1开启轮内提前退出
    if _mode == 1:
        print(f"\n[Step 4] 模式1: 指标驱动优化（最多 {max_episodes} 轮，满足全部指标要求则提前退出）...\n")
    elif _mode == 2:
        print(f"\n[Step 4] 模式2: 轮次驱动优化（无指标要求，运行 {max_episodes} 轮）...\n")
    else:
        print(f"\n[Step 4] 模式3: 运行完整 {max_episodes} 轮，结束后评估指标要求...\n")
    t0 = datetime.now()

    rl_result = await rl_optimizer.optimize(
        script_path=_rl_script,
        mission_conditions=mission_conditions,
        max_episodes=max_episodes,
        nmc=nmc_per_eval,
        task_context={"task": "guidance_rl_optimization"},
        miss_threshold=miss_threshold,
        task_prompt=None,
        reflection_agent=None,
        metric_requirements=_metric_reqs,
        resume_checkpoint=args.resume_checkpoint,
    )

    t1 = datetime.now()
    elapsed = (t1 - t0).total_seconds()

    # ── 10. 打印结果 ──────────────────────────────────────────────────────────
    best_params  = rl_result.get("best_params",  {})
    best_metrics = rl_result.get("best_metrics", {})
    total_ep     = rl_result.get("total_episodes", 0)
    status       = rl_result.get("status", "success")

    print(f"\n{'='*60}")
    print(f"  RL 优化完成  状态={status}  用时={elapsed:.1f}s  轮次={total_ep}")
    print(f"{'='*60}")
    _print_results(best_params, best_metrics, total_ep)

    if _any_reqs:
        from multi_agent.rl.matlab_rl_optimizer import MatlabRLOptimizer as _RLOpt
        _req_dict = {"hitrate": req_hitrate, "missmean": req_missmean,
                     "peak_n": req_peak_n, "pm": req_pm}
        _ok = _RLOpt._requirements_met(best_metrics, _req_dict)
        print(f"\n  [指标要求评估] {'✓ 全部满足' if _ok else '✕ 未全部满足'}")
        if req_hitrate  is not None:
            _v = best_metrics.get("hit_rate",    0.0)
            print(f"    {'✓' if _v >= req_hitrate  else '✗'} hitrate  = {_v:.1f}%  (要求≥{req_hitrate}%)")
        if req_missmean is not None:
            _v = best_metrics.get("miss_distance", 99.0)
            print(f"    {'✓' if _v <= req_missmean else '✗'} missmean = {_v:.3f}m  (要求≤{req_missmean}m)")
        if req_peak_n   is not None:
            _v = best_metrics.get("peak_n",       0.0)
            print(f"    {'✓' if _v <= req_peak_n   else '✗'} PeakN    = {_v:.2f}g  (要求≤{req_peak_n}g)")
        if req_pm       is not None:
            _v = best_metrics.get("pitch_PM",     0.0)
            print(f"    {'✓' if _v >= req_pm       else '✗'} PM       = {_v:.1f}°  (要求≥{req_pm}°)")

    # ── 11. 将最优结果写入 JSON ─────────────────────────────────────────────
    out_dir = Path("./rl_results")
    out_dir.mkdir(exist_ok=True)
    ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = out_dir / f"rl_best_{ts_str}.json"
    save_data = {
        "timestamp":    ts_str,
        "script":       _rl_script,
        "conditions":   mission_conditions,
        "requirements": {"hitrate": req_hitrate, "missmean": req_missmean,
                         "peak_n": req_peak_n, "pm": req_pm},
        "status":       status,
        "total_episodes": total_ep,
        "elapsed_sec":  elapsed,
        "best_params":  best_params,
        "best_metrics": best_metrics,
        "baseline_metrics": pre_metrics,
    }
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(save_data, f, ensure_ascii=False, indent=2)
    print(f"\n  最优结果已保存: {out_file}")


def _print_results(params: dict, metrics: dict, total_episodes: int):
    print(f"\n  ── 最优参数 (11维) ──")
    for k, v in params.items():
        print(f"    {k:20s} = {v:.6g}")
    print(f"\n  ── 最优指标 ──")
    print(f"    hit_rate      = {metrics.get('hit_rate',    0):.1f} %")
    print(f"    miss_distance = {metrics.get('miss_distance', 99):.3f} m")
    print(f"    PeakN         = {metrics.get('peak_n',       0):.2f} g")
    print(f"    pitch_PM      = {metrics.get('pitch_PM',     0):.1f} °")
    print(f"    pitch_GM      = {metrics.get('pitch_GM',     0):.1f} dB")
    print(f"    pitch_BW      = {metrics.get('pitch_BW',     0):.1f} rad/s")
    print(f"    ctrl_energy   = {metrics.get('control_energy', 0):.3f}")
    if total_episodes:
        print(f"\n  总 RL 轮次: {total_episodes}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI 入口
# ─────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="直接用 PPO-RL 优化 MC_gongkuang_simulation_robust_all.m",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--conditions", "-c",
        default=None,
        help='工况规格，格式 "A:1,2;B:1"（不指定子工况 → 全子工况）。默认: A 全子工况',
    )
    p.add_argument(
        "--episodes", "-e",
        type=int, default=None,
        help="最大 RL 轮次（覆盖 config.yaml）",
    )
    p.add_argument(
        "--nmc", "-n",
        type=int, default=None,
        help="每轮 Monte Carlo 仿真次数（覆盖 config.yaml）",
    )
    p.add_argument(
        "--script", "-s",
        default=None,
        help=f"MATLAB 脚本路径（默认使用 KB 模板: {KB_TEMPLATE}）",
    )
    p.add_argument(
        "--miss-threshold",
        type=float, default=None,
        help="[旧参数，建议改用 --missmean] 脱靶量阈值 (m)，传给 RL 作 hint。",
    )
    p.add_argument(
        "--peak-n-max",
        type=float, default=None,
        help="PeakN 奖励惩罚阈值 (g)（覆盖 config.yaml；未指定 --peak-n 时生效）",
    )
    p.add_argument(
        "--hitrate",
        type=float, default=None,
        metavar="PCT",
        help="命中率目标下限 (%%)，例: --hitrate 90。基线满足全部要求则跳过 RL。",
    )
    p.add_argument(
        "--missmean",
        type=float, default=None,
        metavar="M",
        help="脱靶量目标上限 (m)，例: --missmean 5.0。同时作为 RL hint。",
    )
    p.add_argument(
        "--peak-n",
        type=float, default=None,
        metavar="G",
        help="PeakN 目标上限 (g)，例: --peak-n 20.0。同时覆盖 RL 奖励惩罚阈值。",
    )
    p.add_argument(
        "--pm",
        type=float, default=None,
        metavar="DEG",
        help="相位裕度目标下限 (deg)，例: --pm 30。",
    )
    p.add_argument(
        "--resume-checkpoint",
        type=str,
        default=None,
        help="PPO checkpoint 路径（.json 文件、stem 或含 latest.json 的目录）",
    )
    return p


if __name__ == "__main__":
    from multi_agent.logging.run_logger import init_run_logging

    _log_path = init_run_logging(argv=sys.argv)
    print(f"[RunLog] 运行日志: {_log_path}")

    args = build_parser().parse_args()
    asyncio.run(run_optimization(args))
