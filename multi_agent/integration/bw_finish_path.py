"""Hot-start path: refine BW only when PE already holds a 4/5 solution."""

from __future__ import annotations

import glob
import logging
import os
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


def find_latest_t4_script(scripts_dir: str = "guidance_output/scripts") -> str:
    patterns = [
        os.path.join(scripts_dir, "guidance_T4_*.m"),
        os.path.join(scripts_dir, "guidance_T4_APN_*.m"),
    ]
    candidates: list[str] = []
    for pat in patterns:
        candidates.extend(glob.glob(pat))
    if not candidates:
        return ""
    return max(candidates, key=os.path.getmtime)


def _objectives_to_metrics(objectives: Dict[str, Any]) -> Dict[str, float]:
    m = dict(objectives or {})
    if "SEP" not in m and "miss_distance" in m:
        m["SEP"] = m["miss_distance"]
    if "peak_ny" not in m and "peak_n" in m:
        m["peak_ny"] = m["peak_n"]
    return m


async def load_bw_finish_seed(
    parameter_experience: Any,
    *,
    task_prompt: str = "",
    mission_conditions: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, str, Dict[str, float], Dict[str, float], str]:
    """
    Load best PE entry for BW-only refinement.

    Returns (ok, script_path, params, metrics, message).
    """
    from multi_agent.integration.design_path_policy import (
        requirements_from_prompt,
        task_fully_satisfied,
    )
    from multi_agent.rl.metric_constraints import (
        count_satisfied_flags,
        failed_metric_keys,
        metric_satisfaction_flags,
    )

    if parameter_experience is None:
        return False, "", {}, {}, "ParameterExperience 未初始化"

    ctx: Dict[str, Any] = {"task": "guidance_rl_optimization"}
    if mission_conditions:
        for case, subs in mission_conditions.items():
            if subs:
                ctx["RUN_CASE"] = str(case)
                ctx["SUB_IDX"] = int(subs[0])
                break
    else:
        ctx["RUN_CASE"] = "T"
        ctx["SUB_IDX"] = 4

    try:
        rows = await parameter_experience.retrieve_best(task_context=ctx, top_k=1)
    except Exception as exc:
        return False, "", {}, {}, f"PE retrieve_best 失败: {exc}"

    if not rows:
        return False, "", {}, {}, "PE 无可用记录，请先完成冷启动 4/5 运行"

    row = rows[0]
    params = dict(getattr(row, "parameters", None) or row.get("parameters", {}) or {})
    dp_params = {
        k[3:]: float(v)
        for k, v in params.items()
        if str(k).startswith("dp_")
    }
    objectives = dict(getattr(row, "objectives", None) or row.get("objectives", {}) or {})
    metrics = _objectives_to_metrics(objectives)

    script_path = ""
    meta = dict(getattr(row, "metadata", None) or row.get("metadata", {}) or {})
    for key in ("model_path", "script_path", "matlab_script"):
        cand = meta.get(key) or row.get(key) if isinstance(row, dict) else None
        if cand and os.path.isfile(str(cand)):
            script_path = str(cand)
            break
    if not script_path:
        script_path = find_latest_t4_script()
    if not script_path or not os.path.isfile(script_path):
        return False, "", {}, {}, "未找到 T4 制导脚本（PE 或 guidance_output/scripts）"

    reqs = requirements_from_prompt(task_prompt)
    if task_fully_satisfied(metrics, task_prompt, reqs):
        return True, script_path, dp_params, metrics, "五项已全部达标，无需 BW 收尾"

    flags = metric_satisfaction_flags(metrics, reqs)
    failed = failed_metric_keys(flags)
    n_ok = count_satisfied_flags(flags)
    if failed != ("bw",):
        return (
            False,
            script_path,
            dp_params,
            metrics,
            f"当前 PE 快照 {n_ok}/5 达标，失败项={failed}；BW 收尾仅支持「仅 BW 未达标」",
        )

    logger.info(
        "[BWFinish] Loaded PE seed | script=%s | BW=%.2f | params=%d",
        os.path.basename(script_path),
        float(metrics.get("pitch_BW", 0)),
        len(dp_params),
    )
    return True, script_path, dp_params, metrics, "PE 4/5 快照已加载，进入 BW 专扫"
