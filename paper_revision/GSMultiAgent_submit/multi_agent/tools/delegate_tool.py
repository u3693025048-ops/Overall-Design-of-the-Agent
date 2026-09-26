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
                logger.info("[DelegateSubAgent] %s done (%d chars)", agent_type, len(str(result)))
                return result
            except Exception as exc:
                logger.error("[DelegateSubAgent] %s failed: %s — falling back to hermes", agent_type, exc)
                # fall through to hermes fallback

        # ── Fallback: full HermesIntegration ──────────────────────────────────
        if self._sub_agent is None:
            return json.dumps({
                "status": "error",
                "message": (
                    f"agent_type='{agent_type}' not found and no fallback sub-agent configured. "
                    "Available: " + str(list(self._specialized.keys()))
                ),
            })

        try:
            result = await self._sub_agent.run_with_tools(full_msg, verbose_thinking=False)
            out = str(result) if result else "(sub-agent returned no output)"
            logger.info("[DelegateSubAgent] hermes fallback done (%d chars)", len(out))
            return out
        except Exception as exc:
            logger.error("[DelegateSubAgent] hermes fallback failed: %s", exc)
            return json.dumps({"status": "error", "message": str(exc)})
