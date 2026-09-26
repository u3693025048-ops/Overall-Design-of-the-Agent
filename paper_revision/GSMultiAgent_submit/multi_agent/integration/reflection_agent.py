import asyncio
import json
import logging
from typing import Dict, Any, List, Optional

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

import os
import re as _re_module
import pathlib

from multi_agent.config_loader import get_config

logger = logging.getLogger(__name__)


def extract_guidance_functions(script_path: str, max_chars: int = 3000) -> str:
    """Extract guidance-law (gf) and autopilot (cg) function bodies from a .m file.

    Returns a compact string with the extracted source, truncated to *max_chars*.
    Returns empty string if the file cannot be read or no functions are found.
    """
    try:
        p = pathlib.Path(script_path)
        if not p.exists() or p.suffix != ".m":
            return ""
        content = p.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""

    snippets: list = []
    # Match MATLAB local/nested function definitions by function NAME.
    # Pattern targets: function [...]=gf(...), function g=cg(...),
    #   function ...=guidance_local(...), function ...=guidance(...)
    # The name must appear right before '(' to avoid matching comments.
    _TARGET_FUNCS = ("gf", "cg", "guidance_local", "guidance")
    func_pattern = _re_module.compile(
        r"(^[ \t]*function\s+.{0,60}?=\s*(\w+)\s*\(.*?$)"
        r"(.*?)"
        r"(?=^[ \t]*function\s|\Z)",
        _re_module.MULTILINE | _re_module.DOTALL,
    )
    for m in func_pattern.finditer(content):
        fname = m.group(2)
        if fname not in _TARGET_FUNCS:
            continue
        header = m.group(1).strip()
        body = m.group(3).rstrip()
        snippet = header + body
        # Remove very long comment blocks to save tokens
        snippet = _re_module.sub(r"(?m)^\s*%[^\n]{120,}$", "", snippet)
        snippets.append(snippet.strip())

    if not snippets:
        return ""

    result = "\n\n".join(snippets)
    if len(result) > max_chars:
        result = result[:max_chars] + "\n... (truncated)"
    return result


def extract_reasoning_content(response_msg: Any) -> Optional[str]:
    """
    Extract chain-of-thought / reasoning content from a LangChain response.

    Different providers return reasoning in different fields:
      • DeepSeek-R1                 → ``additional_kwargs.reasoning_content``
      • OpenRouter (Claude/DS-R1)   → ``additional_kwargs.reasoning``
      • OpenAI o1/o3                → embedded in ``response_metadata``
      • Anthropic w/ thinking       → ``additional_kwargs.thinking`` blocks

    Returns the reasoning string if found, else None.
    """
    if response_msg is None:
        return None

    addl = getattr(response_msg, "additional_kwargs", {}) or {}
    for key in ("reasoning_content", "reasoning", "thinking"):
        val = addl.get(key)
        if val:
            if isinstance(val, list):
                # Anthropic thinking blocks: [{"type":"thinking","thinking":"..."}]
                parts = [b.get("thinking", "") if isinstance(b, dict) else str(b) for b in val]
                joined = "\n".join(p for p in parts if p)
                if joined:
                    return joined
            elif isinstance(val, str) and val.strip():
                return val

    meta = getattr(response_msg, "response_metadata", {}) or {}
    for key in ("reasoning_content", "reasoning"):
        val = meta.get(key)
        if isinstance(val, str) and val.strip():
            return val

    # Some providers stash it under model_extra / extra
    for nest_key in ("model_extra", "extra", "raw"):
        nested = meta.get(nest_key)
        if isinstance(nested, dict):
            for key in ("reasoning_content", "reasoning"):
                val = nested.get(key)
                if isinstance(val, str) and val.strip():
                    return val

    return None


class ReflectionOutput(BaseModel):
    needs_optimization: bool = Field(description="是否需要进一步优化参数")
    suggestion: str = Field(description="给出的优化建议，如果不需要优化可以为空")
    next_action: str = Field(
        default="tune_params",
        description=(
            "下一轮设计路径建议。可选值："
            "'tune_params' — 当前制导律结构合理，仅需调整自驾仪/导引参数；"
            "'modify_law' — 当前制导律结构存在根本缺陷，需重新设计制导律；"
            "'done' — 所有指标已达标，无需继续优化。"
        ),
    )
    design_path_analysis: str = Field(
        default="",
        description=(
            "下一轮设计路径分析（2-4句话）。包含："
            "① 当前制导律的结构性问题诊断（若有）；"
            "② 参数调整无法解决的瓶颈指标（若有）；"
            "③ 建议的具体改进方向（如：增加目标加速度补偿项、修改饱和限幅策略、"
            "   切换为最优制导律等）；"
            "④ 若 next_action='tune_params'，说明重点调哪些参数及方向。"
        ),
    )
    rl_hyperparams: dict = Field(
        default_factory=dict,
        description=(
            "建议调整的 RL 超参数 dict，仅列出需要变更的项。"
            "可调字段: lr_actor, lr_critic, clip_ratio, gamma, max_episodes, nmc_per_eval, "
            "hit_rate_weight, sep_low_bonus, sep_mid_weight, "
            "pm_bonus, pm_penalty, bw_bonus, bw_penalty, peak_ny_penalty。"
            "不需要调整则返回空 {}。"
        ),
    )

class ReflectionAgent:
    """
    结果反思智能体，基于 LangChain 实现。
    用于判别仿真结果，并给出优化建议。
    """
    
    def __init__(self):
        cfg = get_config().llm
        # Enable reasoning capture for providers that need an explicit flag.
        # OpenRouter requires `include_reasoning: true`; OpenAI/Anthropic ignore unknown keys.
        # DeepSeek returns reasoning by default; passing it is harmless.
        extra_body = {"include_reasoning": True}

        # Some models (e.g., DeepSeek reasoning models, Qwen models) only allow temperature=1
        from multi_agent.llm_utils import get_safe_temperature
        temperature = get_safe_temperature(cfg.model, 0.2)

        self.llm = ChatOpenAI(
            model=cfg.model,
            openai_api_key=cfg.api_key or "sk-dummy",
            openai_api_base=cfg.base_url,
            temperature=temperature,
            extra_body=extra_body,
        )
        self.parser = JsonOutputParser(pydantic_object=ReflectionOutput)
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个制导系统仿真结果反思与判别专家。请根据用户任务描述中的具体要求，逐项对照指标判断是否全部满足，并给出下一轮设计路径分析和 RL 超参数调整建议。

【仿真结果字段说明】
  hit_rate / miss_distance(SEP) / pitch_PM / pitch_BW / peak_n / peak_ny_mean
  best_miss_ever / best_peak_n_ever / best_reward_metrics / best_constrained_metrics
  PeakNy 验收判据：MC 各轮峰值的**平均值** peak_n / peak_ny_mean ≤ 要求（默认 20g）。
  peak_ny_max 若存在仅作诊断参考，**不得**因 max 超标而判未达标。

【T 工况 MC 说明】
  在 RUN_CASE='T' 下，PM/BW 来自驾驶仪线性化设计点，同一组参数下 MC 各轮 PM/BW 相同（±0）是正常现象，不代表 MC 失效；Miss/PeakNy 仍会随目标机动扰动而变化。

【判断原则】
1. 以任务描述中明确提出的要求为准，全部满足才返回 needs_optimization=false。
2. PeakNy 约束以 MC **平均值** peak_n / peak_ny_mean 为准；peak_ny_max 超标但均值达标则 PeakNy 视为达标。
3. suggestion 中需逐项说明哪些指标已达标、哪些未达标及建议调整方向。
4. peak_n=0.0 且 best_peak_n_ever=inf 时视为数据不可用，不得仅因此阻塞退出。

【下一轮设计路径分析 — 必填（全局视角）】
next_action 和 design_path_analysis 是最关键的输出，决定下一轮迭代的方向。
你必须站在全局视角，综合分析 optimization_history 中每轮的 mode（设计路径）和指标变化趋势，而非仅看当前单轮结果。

【制导律源码分析 — 当提供了 gf_source 时必须阅读并分析】
如果下方 gf_source 非空，你必须：
  1. 阅读 gf() 函数源码，识别当前制导律算法类型（标准PN / APN / 滑模 / 最优等）
  2. 分析导引比取值方式（固定常数 / 动态计算）、是否有目标加速度补偿、是否有T_go增益调度
  3. 在 design_path_analysis 中说明当前算法的具体优缺点
  4. 若建议 modify_law，必须基于源码分析指出当前算法的具体缺陷
  5. 不要凭空推测算法类型，必须以源码为准

★ 设计路径连续性原则（最高优先级）：
- T4 低风险策略（见 t4_low_risk 配置）优先于下列连续性原则：4/5 仅 peak 超标 → modify_law；MODIFY 后 peak>30g 或 hit<85% → 禁止 tune_params
- 如果本轮使用 MODIFY_LAW（修改制导律）且指标有改善（命中率↑或SEP↓），
  说明制导律修改方向正确，下一轮应继续 next_action="modify_law" 迭代优化，
  而不是切回 tune_params。制导律优化通常需要多轮迭代才能收敛。
- 如果本轮 MODIFY_LAW 指标反而恶化（命中率↓或SEP↑），才考虑回退到
  tune_params 或换一个制导律修改方向。
- 如果连续多轮 TUNE_PARAMS 指标停滞，应建议切换到 modify_law。

路径判断规则：
- 命中率低（<80%）且多轮 TUNE_PARAMS 无改善 → 制导律结构问题 → next_action="modify_law"
- **命中率/SEP/PM/BW 已达标但 PeakNy 均值仍超标** → 参数调优触及天花板 → next_action="modify_law"
- MODIFY_LAW 后指标改善但未达标 → 继续 next_action="modify_law"
  design_path_analysis 需说明在当前制导律基础上进一步优化的方向
- SEP 持续偏大但命中率尚可 → 需增强末制导精度 → 可能是导引比/制导律问题
- PM/BW 超范围但其他指标正常 → 纯参数问题 → next_action="tune_params"
  design_path_analysis 需指出应增大/减小哪些参数
- 所有指标均达标 → next_action="done"
- 制导律结构性问题示例：
  · 标准 PN 对机动目标跟踪能力不足 → 建议改用 APN 或增加目标加速度估计
  · 饱和限幅参数不当导致制导指令截断 → 调整限幅策略
  · LOS 角速率估计精度不足 → 改进滤波方法
  · 末段制导增益不足 → 增加 T_go 相关的增益调度

★ optimization_history 中 mode 字段含义：
  TUNE_PARAMS  — 仅调整自驾仪/导引参数（w1/zeta1/tao1/N_pn）
  MODIFY_LAW   — 修改了制导律函数 gf() 的结构/算法
  REUSE_HISTORY — 复用历史模型

【RL 超参数动态调整原则】
根据 optimization_history（多轮优化历史）分析趋势，在 rl_hyperparams 中给出调整建议：

趋势分析规则（仅列出需要变更的字段，不变的字段不填）：
- 奖励连续多轮停滞（改善 < 5%）且未收敛 → 增大 lr_actor/lr_critic（×1.5~2），增大 max_episodes
- 奖励剧烈震荡（相邻轮差异 > 30%）→ 降低 lr_actor/lr_critic（×0.5），减小 clip_ratio（0.1~0.15）
- PM 多轮持续超出范围 → 增大 pm_penalty（当前值×1.5），减小 pm_bonus
- BW 多轮持续超出范围 → 增大 bw_penalty（当前值×1.5），减小 bw_bonus
- hit_rate 多轮 < 80% → 增大 hit_rate_weight（当前值×1.5~2）
- SEP 多轮 > 要求值×2 → 增大 sep_low_bonus / sep_mid_weight
- peak_n 多轮超标 → 增大 peak_ny_penalty（当前值×1.5）
- 首轮（无历史）→ rl_hyperparams 返回空 {{}}

可调字段及合理范围：
  lr_actor: 1e-5 ~ 1e-2        lr_critic: 1e-5 ~ 3e-2
  clip_ratio: 0.05 ~ 0.4       gamma: 0.90 ~ 0.999
  max_episodes: 10 ~ 150       nmc_per_eval: 5 ~ 50
  hit_rate_weight: 0.5 ~ 10    sep_low_bonus: 0 ~ 8    sep_mid_weight: 0 ~ 5
  pm_bonus: 0 ~ 4              pm_penalty: 0 ~ 8
  bw_bonus: 0 ~ 4              bw_penalty: 0 ~ 8
  peak_ny_penalty: 0 ~ 8

请只返回以下 JSON 格式，不要包含任何其他说明文字：
{format_instructions}
"""),
            ("user", """
当前的提示词/任务描述：
{prompt}

本轮仿真结果（当前最优）：
{best_result}

多轮优化历史（按迭代顺序，最新在后）：
{optimization_history}

当前制导律源码（gf/cg函数，来自仿真脚本）：
{gf_source}

请进行反思与判别，给出：
1. 指标达标情况逐项分析
2. 当前制导律算法结构分析（基于源码）
3. next_action（tune_params / modify_law / done）
4. design_path_analysis（下一轮设计路径分析，结合源码分析）
5. RL 超参数调整建议
""")
        ])

        self.chain = self.prompt | self.llm

    async def reflect(
        self,
        current_prompt: str,
        best_result: Dict[str, Any],
        optimization_history: Optional[List[Dict[str, Any]]] = None,
        script_path: str = "",
    ) -> Dict[str, Any]:
        """
        对仿真结果进行反思，返回判别结果 JSON 结构。
        optimization_history: 历次迭代的 {iteration, metrics, params, suggestion} 列表，
                              用于多轮趋势分析和 RL 超参数调整建议。
        """
        history_str = "（首轮，无历史记录）"
        if optimization_history:
            import json as _j
            rows = []
            for h in optimization_history[-8:]:
                _mode = h.get("task_mode", "")
                _script = h.get("script", "")
                _na = h.get("next_action", "")
                _dpa = h.get("design_path_analysis", "")
                rows.append(
                    f"  iter={h.get('iteration','')}  mode={_mode}  "
                    f"hit={h.get('hit_rate',0):.1f}%  SEP={h.get('SEP',h.get('miss_distance',99)):.2f}m  "
                    f"PM={h.get('pitch_PM',0):.1f}  BW={h.get('pitch_BW',0):.1f}  "
                    f"PeakNy_avg={h.get('peak_n', h.get('peak_ny', 0)):.2f}  "
                    f"reward={h.get('best_reward',0):.3f}"
                    + (f"  next_action={_na}" if _na else "")
                    + (f"  script={_script}" if _script else "")
                    + (f"\n    设计路径: {_dpa[:150]}" if _dpa else "")
                )
            history_str = "\n".join(rows)

        try:
            # Extract gf()/cg() source from current .m script
            gf_source = ""
            if script_path:
                gf_source = extract_guidance_functions(script_path)
                if gf_source:
                    logger.info(f"[Reflection] Extracted gf/cg source from {os.path.basename(script_path)} ({len(gf_source)} chars)")
                else:
                    logger.info(f"[Reflection] No gf/cg functions found in {os.path.basename(script_path)}")
            if not gf_source:
                gf_source = "（未提供源码，无法进行算法结构分析）"

            logger.info("正在调用 Reflection Agent 进行结果反思...")
            response_msg = await self.chain.ainvoke({
                "prompt": current_prompt,
                "best_result": json.dumps(best_result, ensure_ascii=False, indent=2),
                "optimization_history": history_str,
                "gf_source": gf_source,
                "format_instructions": self.parser.get_format_instructions()
            })

            # Surface reasoning content (chain-of-thought) for reasoning models
            # such as Claude-thinking, DeepSeek-R1, OpenAI o1/o3.
            reasoning = extract_reasoning_content(response_msg)
            if reasoning:
                # Limit log to avoid flooding; full reasoning still printed at INFO
                preview = reasoning[:1500] + ("…" if len(reasoning) > 1500 else "")
                logger.info(f"[Reasoning] (model chain-of-thought, {len(reasoning)} chars):\n{preview}")
            else:
                logger.debug(
                    "[Reasoning] No chain-of-thought found in response. "
                    "additional_kwargs keys: %s, response_metadata keys: %s",
                    list(getattr(response_msg, 'additional_kwargs', {}) or {}),
                    list(getattr(response_msg, 'response_metadata', {}) or {}),
                )

            content = response_msg.content

            # Remove <think>...</think> blocks often generated by reasoning models
            # (some return them inline in `content` instead of as a separate field)
            import re
            content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
            
            if content.startswith("```json"):
                content = content[7:-3].strip()
            elif content.startswith("```"):
                content = content[3:-3].strip()
                
            try:
                parsed_response = json.loads(content)
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse JSON natively, trying Langchain Parser. Raw: {content[:300]}")
                try:
                    parsed_response = self.parser.parse(response_msg.content)
                except Exception as _pe:
                    logger.warning(f"Langchain parser also failed ({_pe!r}); returning safe default.")
                    parsed_response = {
                        "needs_optimization": True,
                        "suggestion": f"JSON解析失败，原始内容: {content[:200]}",
                    }
            metrics = {}
            if isinstance(best_result, dict):
                metrics = best_result.get("metrics") or best_result
            from multi_agent.integration.design_path_policy import apply_reflection_policy

            return apply_reflection_policy(
                self._postprocess_reflection(parsed_response),
                task_prompt=current_prompt,
                metrics=metrics if isinstance(metrics, dict) else {},
                optimization_history=optimization_history,
            )
                
        except (KeyboardInterrupt, asyncio.CancelledError):
            raise  # always propagate Ctrl+C and task cancellation
        except Exception as e:
            import traceback as _tb
            logger.error(f"Reflection Agent 运行出错: {e!r}\n{_tb.format_exc()}")
            # 如果出错，保守返回需要继续优化，避免误判完成
            return {"needs_optimization": True, "suggestion": f"反思阶段出错: {e!r}"}

    @staticmethod
    def _postprocess_reflection(parsed: Any) -> Dict[str, Any]:
        """Sanitize LLM output through the parameter firewall before downstream use."""
        if not isinstance(parsed, dict):
            return {"needs_optimization": True, "suggestion": str(parsed)}

        from multi_agent.security.parameter_firewall import sanitize_rl_hyperparams

        raw_hp = parsed.get("rl_hyperparams") or {}
        sanitized, warnings = sanitize_rl_hyperparams(raw_hp if isinstance(raw_hp, dict) else {})
        parsed["rl_hyperparams"] = sanitized
        for w in warnings:
            logger.warning("[ParameterFirewall] Reflection %s", w)
        return parsed
