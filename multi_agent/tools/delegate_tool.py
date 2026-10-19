#!/usr/bin/env python3
"""
delegate_tool.py
Exposes a delegate_task tool that routes work to ONE of five specialized
sub-agents (each with a focused tool subset) OR to a full-toolset fallback.

agent_type choices:
  rag_agent      — knowledge retrieval     (rag_query, rag_retrieve, rag_expand)
  matlab_agent   — MATLAB generation       (generate_matlab, syntax_check_matlab, verify_guidance_compat)
  sim_agent      — simulation execution    (run_simulation, syntax_check_matlab)
  analysis_agent — requirements & params   (judge_requirements, parameter_experience_*)
  memory_agent   — persistent memory       (agent_memory_*)
  hermes         — full-toolset fallback   (legacy, all tools)

P1-4：全工具集回退**只有显式** `agent_type="hermes"` 才可达。具名专家执行失败
或名字未注册时一律拒绝（fail-closed），不再静默升级为全工具集 —— 否则一个只该
持 3 个只读工具的 `rag_agent` 会因一次异常而获得 `run_simulation` /
`agent_memory_forget` 的调用能力。

P2-4（礼部）：委托返回前经 `agent_protocol.refuse_if_violating()` 做**输出协议
校验**。默认 audit（只留痕、不改返回值），`ablation.protocol_strict=true` 时把
不合规回复换成结构化拒绝载荷。校验是旁路，异常一律吞掉、不影响委派主流程。
"""

import json
import logging
from typing import Any, Dict, Optional

from multi_agent.tools.tool_arg_aliases import coerce_task_prompt as _coerce_task_prompt

logger = logging.getLogger(__name__)

# Import catalogue for descriptions (no circular import — just constants)
try:
    from multi_agent.integration.sub_agent import SUB_AGENT_CONFIGS as _CONFIGS
    _AGENT_TYPE_CHOICES = list(_CONFIGS.keys()) + ["hermes"]
    _AGENT_TYPE_DESC = "\n".join(
        f"  {k}: {v['description']}" for k, v in _CONFIGS.items()
    ) + "\n  hermes: 全工具集回退（兼容旧调用）"
except ImportError:
    _AGENT_TYPE_CHOICES = ["hermes"]
    _AGENT_TYPE_DESC = "  hermes: 全工具集回退"


# ── P1-4：具名专家失败 / 未注册时，是否允许升级为"全工具集"回退 ────────────────
# 旧行为：任何 `agent_type` 走到最后都会落到 `self._sub_agent.run_with_tools(...)`
# （**全工具集**，绑 orchestrator 策略）。这意味着一个只该有 3 个只读工具的
# `rag_agent`，一旦执行报错或名字拼错，就**静默升级**成能调 `run_simulation`、
# `agent_memory_forget` 的全权子智能体 —— 与 `sub_agent.py:260-268`
# （差距映射 P1-4）是**同一类缺陷**（静默降级 / 越权暴露），只是一处由"配置写错"
# 触发、另一处由"执行异常"触发。
#
# False（默认）：只有**显式** `agent_type="hermes"` 才使用全工具集；具名专家失败
#   或未注册一律拒绝（fail-closed）。True：恢复旧的升级行为，仅供消融对照。
_ALLOW_SPECIALIST_ESCALATION = False

CODE_SPECIALIST_FAILED = "specialist_failed"
CODE_UNKNOWN_AGENT_TYPE = "unknown_agent_type"


def set_allow_specialist_escalation(enabled: bool) -> None:
    """开关"具名专家失败 → 全量工具集"的升级行为（由 `ablation.sub_agent_strict` 驱动）。"""
    global _ALLOW_SPECIALIST_ESCALATION
    _ALLOW_SPECIALIST_ESCALATION = bool(enabled)


def is_allow_specialist_escalation() -> bool:
    return _ALLOW_SPECIALIST_ESCALATION


class DelegateSubAgentTool:
    """
    Route a subtask to a specialized sub-agent by agent_type.

    Hermes (orchestrator) should call this instead of invoking tools directly,
    passing the appropriate agent_type so the right specialist handles the work.
    """

    name = "call_specialist"
    description = (
        "将子任务委托给专用子智能体执行。Hermes 作为纯编排器，通过此工具调用各领域专家子智能体。\n"
        "子智能体类型（agent_type）：\n"
        f"{_AGENT_TYPE_DESC}\n\n"
        "使用规则：\n"
        "  · 知识检索 → agent_type='rag_agent'\n"
        "  · MATLAB脚本生成与适配验证 → agent_type='matlab_agent'\n"
        "  · 仿真运行 → agent_type='sim_agent'\n"
        "  · 需求判断与参数推荐 → agent_type='analysis_agent'\n"
        "  · 记忆存取 → agent_type='memory_agent'\n"
        "  · 通用/不确定 → agent_type='hermes'（回退到全工具集）"
    )
    input_schema: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "agent_type": {
                "type": "string",
                "enum": _AGENT_TYPE_CHOICES,
                "description": "专用子智能体类型（见工具描述）",
                "default": "hermes",
            },
            "task": {
                "type": "string",
                "description": "子任务的完整描述，包含所有必要参数（文件路径、工况、指标等）",
            },
            "context": {
                "type": "string",
                "description": "可选背景信息（如上轮结果、工况参数等）",
                "default": "",
            },
        },
        "required": ["task"],
    }

    def __init__(self) -> None:
        self._sub_agent = None          # fallback: full HermesIntegration
        self._specialized: Dict[str, Any] = {}   # agent_type → SpecializedSubAgent
        self._mission_conditions: str = ""

    # ── Injection ──────────────────────────────────────────────────────────────

    def set_sub_agent(self, sub_agent) -> None:
        """Inject the full-toolset HermesIntegration for the 'hermes' fallback."""
        self._sub_agent = sub_agent

    def register_specialized_agents(self, agents: Dict[str, Any]) -> None:
        """Inject pre-built {agent_type: SpecializedSubAgent} mapping."""
        self._specialized.update(agents)
        logger.info(
            "[DelegateSubAgent] Specialized agents registered: %s",
            list(agents.keys()),
        )

    def update_mission_conditions(self, cond_str: str) -> None:
        self._mission_conditions = cond_str
        if self._sub_agent and hasattr(self._sub_agent, "update_mission_conditions"):
            self._sub_agent.update_mission_conditions(cond_str)

    # ── Execute ────────────────────────────────────────────────────────────────

    async def execute(
        self,
        task: str = "",
        agent_type: str = "hermes",
        context: str = "",
        **kwargs: Any,
    ) -> str:
        task = task or _coerce_task_prompt("", **kwargs)
        if not task:
            return json.dumps({"status": "error", "message": "task is required"})

        full_msg = f"[委托子任务]\n{task}"
        if context:
            full_msg = f"[背景信息]\n{context}\n\n{full_msg}"
        if self._mission_conditions:
            full_msg += (
                f"\n\nMISSION CONDITIONS (use exactly when calling generate_matlab): "
                f'"{self._mission_conditions}"'
            )

        logger.info(
            "[DelegateSubAgent] → %s | task: %s",
            agent_type, task[:100],
        )

        # ── Route to specialized sub-agent ────────────────────────────────────
        if agent_type != "hermes" and agent_type in self._specialized:
            agent = self._specialized[agent_type]
            try:
                result = await agent.run(task, context=context)

                # ── P2-4 礼部：子智能体间协议强制校验 ──────────────────────────
                # 差距原文：「智能体间协议仅"提示词约定 + 正则抽取"，无强制校验」。
                # 这里在**委派边界**对子智能体回复做输出契约校验：
                #   · 默认 audit —— 只记日志 + 留最近一次报告，**回复原样返回**
                #     （行为完全等价接入前）；
                #   · `ablation.protocol_strict=true` 时，不合规回复换成结构化
                #     拒绝载荷（含 raw_response，拒绝格式但不丢信息）。
                # ⚠️ 校验本身是旁路：任何异常都不得影响委派主流程（吞掉并 debug 记录）。
                try:
                    from multi_agent.integration import agent_protocol as _ap
                    _refusal = _ap.refuse_if_violating(agent_type, result)
                    if _refusal is not None:
                        logger.warning(
                            "[DelegateSubAgent] %s 协议不合规 → 按 protocol_strict 拒绝",
                            agent_type,
                        )
                        return _refusal
                except Exception as _pexc:  # pragma: no cover — 校验不可用即跳过
                    logger.debug(
                        "[DelegateSubAgent] 协议校验不可用，跳过（不改行为）：%s", _pexc
                    )

                logger.info("[DelegateSubAgent] %s done (%d chars)", agent_type, len(str(result)))
                return result
            except Exception as exc:
                logger.error("[DelegateSubAgent] %s failed: %s", agent_type, exc)
                # ── P1-4：具名专家失败**不得**静默升级为全量工具集 ──
                if not _ALLOW_SPECIALIST_ESCALATION:
                    return json.dumps({
                        "status": "error",
                        "code": CODE_SPECIALIST_FAILED,
                        "agent_type": agent_type,
                        "message": (
                            f"专用子智能体 {agent_type!r} 执行失败：{exc}。"
                            "已拒绝升级为全工具集回退（P1-4 fail-closed）——"
                            "请修正该专家的问题或显式使用 agent_type='hermes'。"
                        ),
                    }, ensure_ascii=False)
                logger.warning(
                    "[DelegateSubAgent] %s failed → 升级为全工具集回退"
                    "（allow_specialist_escalation=true，仅供消融对照）", agent_type,
                )

        # ── P1-4：具名但未注册 / 已被隔离 → 拒绝，不升级 ──────────────────────
        elif agent_type != "hermes" and not _ALLOW_SPECIALIST_ESCALATION:
            logger.error(
                "[DelegateSubAgent] 未知 agent_type=%r（未注册或已被启动期隔离），"
                "拒绝升级为全工具集回退（P1-4 fail-closed）。available=%s",
                agent_type, sorted(self._specialized.keys()),
            )
            return json.dumps({
                "status": "error",
                "code": CODE_UNKNOWN_AGENT_TYPE,
                "agent_type": agent_type,
                "available": sorted(self._specialized.keys()) + ["hermes"],
                "message": (
                    f"未知 agent_type={agent_type!r}（未注册或已被启动期隔离）。"
                    "已拒绝升级为全工具集回退（P1-4 fail-closed）。"
                ),
            }, ensure_ascii=False)

        # ── Fallback: full HermesIntegration（仅 agent_type='hermes' 或显式放开）──
        if self._sub_agent is None:
            return json.dumps({
                "status": "error",
                "message": (
                    f"agent_type='{agent_type}' not found and no fallback sub-agent configured. "
                    "Available: " + str(list(self._specialized.keys()))
                ),
            })

        try:
            # ── 全权回退也走显式策略（P1-2）─────────────────────────────────
            # 「全工具集回退」在设计上等价于编排者（orchestrator）全权，
            # 显式绑定 orchestrator 策略让审计链完整：即使这条路径，工具
            # 调用同样经过执行边界的权限闸门，而不是"无策略=不受限"。
            try:
                from multi_agent.security.tool_guard import (
                    orchestrator_policy, policy_scope,
                )
                from multi_agent.security.confirm_gate import (
                    orchestrator_policy as confirm_orch, policy_scope as confirm_scope,
                )
                with policy_scope(orchestrator_policy()), confirm_scope(confirm_orch()):
                    result = await self._sub_agent.run_with_tools(full_msg, verbose_thinking=False)
            except ImportError:  # pragma: no cover — 闸门不可用时保持旧行为
                result = await self._sub_agent.run_with_tools(full_msg, verbose_thinking=False)
            out = str(result) if result else "(sub-agent returned no output)"
            logger.info("[DelegateSubAgent] hermes fallback done (%d chars)", len(out))
            return out
        except Exception as exc:
            logger.error("[DelegateSubAgent] hermes fallback failed: %s", exc)
            return json.dumps({"status": "error", "message": str(exc)})
