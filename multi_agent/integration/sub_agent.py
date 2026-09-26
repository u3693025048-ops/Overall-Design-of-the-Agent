#!/usr/bin/env python3
"""
sub_agent.py — Specialized Sub-Agent via Tool Filtering

Design rationale
────────────────
The Hermes registry is global and tool names are the unique key.
Re-registering the same tool name under a different toolset causes
handler collisions and `valid_tool_names` snapshot ordering issues.

Solution: SpecializedSubAgent does NOT touch the registry.
Instead it wraps the already-initialized `sub_hermes` HermesIntegration
and calls `run_with_tools(msg, tools=subset)`.

`run_with_tools` updates `self.agent.tools` (the formatted schema list
sent to the LLM in the prompt), so the LLM only *sees* the subset tools
and will not attempt to call anything outside it.  The registry dispatch
still works because the tool names in the subset are already registered
under `guidance_system` by the sub_hermes initialisation.

Message passing protocol
────────────────────────
  Hermes → call_specialist(agent_type, task, context) → str
                │
                ▼  DelegateSubAgentTool.execute()
  builds:  AgentMessage (structured JSON string)
                │
                ▼  SpecializedSubAgent.run()
  calls:   sub_hermes.run_with_tools(agent_message, tools=subset)
                │  LLM sees only subset tools in its system prompt
                │  LLM calls tools → registry dispatches to tool.execute()
                ▼
  returns: AgentResponse (structured JSON string)
                │
                ▼  DelegateSubAgentTool returns string to Hermes LLM

AgentMessage schema:
  {
    "agent_type": "matlab_agent",
    "system_context": "<specialized system prompt>",
    "task": "<full task description>",
    "context": "<optional background>",
    "available_tools": ["generate_matlab", "syntax_check_matlab", ...]
  }

The sub-agent's `final_response` (plain text) is returned as-is to Hermes.
"""

import json
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ── Sub-agent catalogue ────────────────────────────────────────────────────────
SUB_AGENT_CONFIGS: Dict[str, Dict[str, Any]] = {
    # ──────────────────────────────────────────────────────────────────────────
    # rag_agent
    # Output consumed by: matlab_agent (context), hermes (reasoning)
    # Key fields:  law_type, key_params, design_principles, sources
    # ──────────────────────────────────────────────────────────────────────────
    "rag_agent": {
        "tool_names": ["rag_query", "rag_retrieve", "rag_expand"],
        "description": "知识检索子智能体（多跳 RAG）",
        "system_context": (
            "【子智能体角色：知识检索专家】\n"
            "可用工具：rag_query（Agentic多跳检索）、rag_retrieve（单次检索）、rag_expand（获取全文）。\n"
            "若一次检索不足，分解为多子查询补充。\n\n"
            "【书籍知识库】除 conditions.md / expert design path.md / monte_carlo_single.m 外，\n"
            "还必须检索 knowledge_base/books/books/book010/ 下的蒸馏书籍知识库（《系统化飞控实战·导弹制导》）：\n"
            "  - 5 张域卡：domain_A_PN.md / domain_B_Optimal.md / domain_C_Terminal.md /\n"
            "    domain_D_SystemModeling.md / domain_E_Trajectory.md（RIA++ 卡：理论/公式/经验/关联）\n"
            "  - 数据资产：assets/00_INDEX.md（按问题检索路由）、01_theory.md、02_formulas.md（197 公式 F-xx-xx）、\n"
            "    03_heuristics.md（439 经验 Hxxx）、04_parameters.md（参数速查）、05_checklist.md（7 阶段清单）\n"
            "检索时把用户设计问题映射到对应知识域与资产，用中文关键词组合查询，\n"
            "例如「APN 目标加速度补偿 N 选取 稳定性红线 参数范围」。\n"
            "优先返回与制导律设计相关的：理论依据、公式编号、经验编号、参数推荐值与红线。\n\n"
            "【强制输出格式】完成检索后，必须在最后输出以下 JSON 块（```json ... ``` 包裹）：\n"
            "```json\n"
            "{\n"
            '  "law_type": "PN|APN|OGL|IACG|custom",\n'
            '  "key_params": {"N_pn": 4, "tao1": 0.20, "w1": 40},\n'
            '  "design_principles": "简要设计原则文字",\n'
            '  "stability_notes": "稳定性要点",\n'
            '  "sources": ["doc_id_1", "doc_id_2"]\n'
            "}\n"
            "```\n"
            "此 JSON 将被编排器提取后传递给 matlab_agent 的 context。"
        ),
    },
    # ──────────────────────────────────────────────────────────────────────────
    # matlab_agent
    # Input  from: rag_agent (context.law_type / key_params / design_principles)
    # Output to:   sim_agent (task 中包含 script_path)
    # Key fields:  script_path, compat_overall, key_params
    # ──────────────────────────────────────────────────────────────────────────
    "matlab_agent": {
        "tool_names": ["generate_matlab", "syntax_check_matlab", "verify_guidance_compat"],
        "description": "MATLAB制导律生成与验证子智能体",
        "system_context": (
            "【子智能体角色：MATLAB制导律设计专家】\n"
            "可用工具：generate_matlab、syntax_check_matlab、verify_guidance_compat。\n"
            "若 context 中有 rag_agent 检索结果（law_type / key_params / design_principles），\n"
            "  优先使用其中的参数和设计原则来生成脚本。\n\n"
            "【工作流程】\n"
            "  1. generate_matlab(task_description=<任务>, mission_conditions=<工况>)\n"
            "  2. syntax_check_matlab(script_path=<step1路径>)\n"
            "  3. verify_guidance_compat(script_path=<step2路径>, task_description=<任务>)\n"
            "  4. 若 overall=FAIL → 根据 recommendations 修正后重新执行 step1/2/3\n\n"
            "【强制输出格式】完成后，必须在最后输出以下 JSON 块（```json ... ``` 包裹）：\n"
            "```json\n"
            "{\n"
            '  "script_path": "/absolute/path/to/script.m",\n'
            '  "compat_overall": "PASS|WARN|FAIL",\n'
            '  "compat_score": 85,\n'
            '  "key_params": {"N_pn": 4, "tao1": 0.20, "w1": 40},\n'
            '  "stability_conclusion": "简要结论"\n'
            "}\n"
            "```\n"
            "⚠️ script_path 必须是绝对路径，sim_agent 将直接使用该字段运行仿真。"
        ),
    },
    # ──────────────────────────────────────────────────────────────────────────
    # sim_agent
    # Input  from: matlab_agent (task 中的 script_path)
    # Output to:   analysis_agent (context 中的仿真指标)
    # Key fields:  hit_rate, SEP, PeakNy, PM, BW, status
    # ──────────────────────────────────────────────────────────────────────────
    "sim_agent": {
        "tool_names": ["run_simulation", "syntax_check_matlab"],
        "description": "仿真运行子智能体",
        "system_context": (
            "【子智能体角色：仿真执行专家】\n"
            "可用工具：run_simulation、syntax_check_matlab。\n"
            "task 中必定包含 script_path 和 mission_conditions，直接提取后调用 run_simulation。\n"
            "若 run_simulation 返回 status='error'：调用 syntax_check_matlab 修复后重试（最多1次）。\n\n"
            "【强制输出格式】完成后，必须在最后输出以下 JSON 块（```json ... ``` 包裹）：\n"
            "```json\n"
            "{\n"
            '  "status": "success|error",\n'
            '  "script_path": "/path/used.m",\n'
            '  "hit_rate": 92.0,\n'
            '  "SEP": 5.2,\n'
            '  "PeakNy": 12.3,\n'
            '  "PM": 45.1,\n'
            '  "BW": 25.3,\n'
            '  "error_message": ""\n'
            "}\n"
            "```\n"
            "analysis_agent 将直接读取 hit_rate/SEP/PeakNy/PM/BW 判断需求符合性。"
        ),
    },
    # ──────────────────────────────────────────────────────────────────────────
    # analysis_agent
    # Input  from: sim_agent (context 中的 hit_rate/SEP/PM/BW)
    # Output to:   hermes (task 完成判断)
    # Key fields:  satisfied, layer3_path, recommended_params
    # ──────────────────────────────────────────────────────────────────────────
    "analysis_agent": {
        "tool_names": [
            "judge_requirements",
            "parameter_experience_best",
            "parameter_experience_search",
        ],
        "description": "需求分析与参数评估子智能体",
        "system_context": (
            "【子智能体角色：需求分析与参数经验专家】\n"
            "可用工具：judge_requirements、parameter_experience_best、parameter_experience_search。\n"
            "context 中包含仿真指标（hit_rate/SEP/PeakNy/PM/BW），直接提取后调用 judge_requirements。\n\n"
            "【强制输出格式】完成后，必须在最后输出以下 JSON 块（```json ... ``` 包裹）：\n"
            "```json\n"
            "{\n"
            '  "satisfied": true,\n'
            '  "layer3_path": "reflection|rl_optimize",\n'
            '  "metrics_vs_req": {"hit_rate": "92% >= 92% ✓", "PM": "45° within [45°,70°] ✓"},\n'
            '  "recommended_params": {"N_pn": 4, "tao1": 0.20},\n'
            '  "reason": "所有指标满足，进入反思智能体"\n'
            "}\n"
            "```\n"
            "hermes 将读取 satisfied 和 layer3_path 决定后续流程。"
        ),
    },
    # ──────────────────────────────────────────────────────────────────────────
    # memory_agent — no structured data passing requirement
    # ──────────────────────────────────────────────────────────────────────────
    "memory_agent": {
        "tool_names": [
            "agent_memory_remember",
            "agent_memory_recall",
            "agent_memory_list",
            "agent_memory_forget",
        ],
        "description": "持久记忆管理子智能体",
        "system_context": (
            "【子智能体角色：记忆管理专家】\n"
            "可用工具：agent_memory_remember、agent_memory_recall、agent_memory_list、agent_memory_forget。\n"
            "任务：根据指令存储、检索或删除持久化记忆条目。\n"
            "完成后返回操作结果（存储了哪个key / 检索到什么内容）。"
        ),
    },
}


class SpecializedSubAgent:
    """
    Routes a delegated task to the shared sub_hermes HermesIntegration with
    a filtered tool subset.  No registry re-registration — avoids all handler
    collision and valid_tool_names snapshot ordering issues.

    Message passing:
      Input  — structured prompt string:  system_context + task + context
      Output — sub-agent's final_response string (plain text or JSON)
      Both sides communicate via string; structure is enforced by the
      system_context instruction and the LLM's reasoning.
    """

    def __init__(
        self,
        agent_type: str,
        tool_names: List[str],
        system_context: str,
    ) -> None:
        self.agent_type = agent_type
        self.tool_names = tool_names          # names we want to expose
        self.system_context = system_context
        self._sub_hermes = None               # injected by build_specialized_agents()

    def set_sub_hermes(self, sub_hermes) -> None:
        """Inject the shared (already-initialized) sub_hermes."""
        self._sub_hermes = sub_hermes

    async def run(self, task: str, context: str = "") -> str:
        """
        Execute task via sub_hermes with only the configured tool subset visible
        to the LLM.

        Message construction:
          1. system_context (role + allowed tools + workflow)
          2. context        (background from orchestrator, if any)
          3. task           (the actual work to perform)
        """
        if self._sub_hermes is None:
            return json.dumps({
                "status": "error",
                "message": f"SpecializedSubAgent({self.agent_type}) not wired to sub_hermes",
            })

        # Build structured prompt
        parts = [self.system_context]
        if context:
            parts.append(f"\n[来自编排器的背景信息]\n{context}")
        parts.append(f"\n[任务]\n{task}")
        full_msg = "\n".join(parts)

        # Filter sub_hermes._tools to only the allowed subset
        all_tools = getattr(self._sub_hermes, "_tools", []) or []
        subset_tools = [t for t in all_tools if getattr(t, "name", "") in self.tool_names]

        if not subset_tools:
            logger.warning(
                "[SubAgent:%s] No tools matched from %d available. "
                "tool_names=%s  available=%s",
                self.agent_type, len(all_tools),
                self.tool_names, [t.name for t in all_tools],
            )
            # Fall back to unfiltered
            subset_tools = all_tools or None

        logger.info(
            "[SubAgent:%s] run — tools: %s | task: %.80s",
            self.agent_type,
            [t.name for t in (subset_tools or [])],
            task,
        )

        try:
            result = await self._sub_hermes.run_with_tools(
                full_msg,
                tools=subset_tools,
                verbose_thinking=False,
            )
            return str(result) if result else "(sub-agent returned no output)"
        except Exception as exc:
            logger.error("[SubAgent:%s] run failed: %s", self.agent_type, exc)
            return json.dumps({"status": "error", "message": str(exc)})


def build_specialized_agents(
    sub_hermes: Any,
) -> Dict[str, SpecializedSubAgent]:
    """
    Create all specialized sub-agents backed by the shared sub_hermes.
    No registry changes — just wraps sub_hermes with different tool filters.

    Args:
        sub_hermes: Already-initialized HermesIntegration (include_delegate=False).
    """
    agents: Dict[str, SpecializedSubAgent] = {}
    for agent_type, cfg in SUB_AGENT_CONFIGS.items():
        agent = SpecializedSubAgent(
            agent_type=agent_type,
            tool_names=cfg["tool_names"],
            system_context=cfg["system_context"],
        )
        agent.set_sub_hermes(sub_hermes)
        agents[agent_type] = agent
        logger.info(
            "[SubAgentFactory] %s → tools: %s",
            agent_type, cfg["tool_names"],
        )
    return agents
