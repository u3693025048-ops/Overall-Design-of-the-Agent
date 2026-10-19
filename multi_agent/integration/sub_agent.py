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

**P1-4（白名单失败即拒绝）**：若声明的工具名一个都匹配不上（拼写错误 /
未注册 / 清单为空），`run()` **不再**回退为全量工具集，而是返回结构化错误；
`build_specialized_agents()` 另在构建前做启动期断言（工具名须在
`tool_guard.TOOL_CATALOG` 中且与 `ROLE_TOOLS` 一致），有问题即 ERROR + 隔离。

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
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ── P2-4 礼部：子智能体间协议提示词改由集中注册表提供（唯一来源）──────────────
# 正文不再硬编码在本文件，而存于 `multi_agent.prompt_registry.INLINE_PROMPTS`
# 的 `prompt:sub_agent:<agent_type>` 下，与 `agent_protocol.AGENT_OUTPUT_SCHEMAS`
# 的输出契约同属一份台账（P2-4：提示词集中登记 + 版本化 + 协议强制校验）。
#
# ⚠️ 这里存的是 **prompt id**（以 `prompt:` 开头的字符串），**不是正文**：
#    正文在 `build_specialized_agents()` 构建时按 id 向注册表取值。
#    刻意惰性取值 —— 若在模块导入期取值，注册表任何一处异常都会让整个
#    `multi_agent` 包导入失败（连不相关的功能一起拖垮），代价过大。
#
# ⚠️ 取不到就抛错，**绝不静默返回空提示词** —— 空 system_context 会让子智能体
#    失去角色定义与工具约束（比直接崩溃更坏的失败形态）。
PROMPT_ID_PREFIX = "prompt:"


def resolve_system_context(cfg: Optional[Dict[str, Any]]) -> str:
    """把 ``system_context`` 解析成正文。

    ``"prompt:..."`` → 向 `prompt_registry` 取值（未登记即抛错）；
    其它字符串 → 视为内联正文原样返回（自定义 / 测试用）。
    """
    raw = str((cfg or {}).get("system_context", "") or "")
    if raw.startswith(PROMPT_ID_PREFIX):
        from multi_agent.prompt_registry import get_prompt
        return get_prompt(raw)
    return raw


# ── P1-4：白名单严格模式（失败即拒绝，绝不回退全量工具）────────────────────────
# 差距映射 P1-4：原实现在白名单**无匹配**时 `subset_tools = all_tools or None`，
# 把**全量工具**塞回给 LLM —— 典型的"配置写错就静默降级"，等于打穿工具隔离。
# （"凡白名单名与真实 Tool.name 一致时暂不触发"只是运气，不是防线。）
#
# 严格模式（默认）语义：
#   · 运行时：声明的工具**一个都匹配不上** → 返回结构化错误，不执行、不暴露
#     任何域外工具；
#   · 启动期：声明的工具名必须在权威目录 `tool_guard.TOOL_CATALOG` 中，且与该
#     角色矩阵 `ROLE_TOOLS[agent_type]` 一致 —— 任一不满足即报 ERROR 并**隔离**
#     该子智能体（不构建），绝不带着错误配置上线。
# 关闭（`ablation.sub_agent_strict=false`）可恢复接入前行为，仅供消融对照。
_STRICT_WHITELIST = True

# 失败即拒绝时返回给 LLM 的结构化错误码（soft-fail：返回错误串，不抛异常）
CODE_TOOLSET_UNAVAILABLE = "sub_agent_toolset_unavailable"


class SubAgentConfigError(RuntimeError):
    """子智能体白名单配置错误（工具名未知 / 与角色矩阵不一致）。"""


def set_strict_whitelist(enabled: bool) -> None:
    """开关白名单严格模式（由 `ablation.sub_agent_strict` 驱动）。"""
    global _STRICT_WHITELIST
    _STRICT_WHITELIST = bool(enabled)


def is_strict_whitelist() -> bool:
    """当前是否处于白名单严格模式。"""
    return _STRICT_WHITELIST


def known_tool_names() -> frozenset:
    """权威工具名集合 —— 取自 `tool_guard.TOOL_CATALOG`（零重依赖模块）。

    注意：权威目录是**工具的全体集合**，与 `sub_hermes._tools`（会被消融开关
    裁剪）不同。用后者做启动期校验会在消融跑次里误报，故此处只认目录。
    闸门不可用时返回空集，此时校验降级为"只比角色矩阵"。
    """
    try:
        from multi_agent.security.tool_guard import TOOL_CATALOG
        return frozenset(TOOL_CATALOG)
    except Exception:  # pragma: no cover — 闸门不可用时降级
        return frozenset()


def _role_matrix(agent_type: str) -> Optional[Tuple[str, ...]]:
    """角色矩阵中该 agent_type 的权威工具集（取自 `tool_guard.ROLE_TOOLS`）。"""
    try:
        from multi_agent.security.tool_guard import ROLE_TOOLS
        names = ROLE_TOOLS.get(agent_type)
        return tuple(names) if names else None
    except Exception:  # pragma: no cover
        return None


def validate_sub_agent_configs(
    configs: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """启动期校验：声明的工具名必须 ① 在权威目录中 ② 与角色矩阵一致。

    纯函数（无副作用），启动期与机检共用。返回::

        {
          "ok": bool,
          "known": [...],                 # 权威目录中的工具名（排序）
          "bad_agents": [...],            # 存在问题的 agent_type（排序）
          "problems": [ {agent_type, kind, detail}, ... ],
          "agents": { agent_type: {"declared": [...], "unknown": [...],
                                   "role_mismatch": [...]} },
        }

    ``kind`` ∈ {``no_tools``, ``unknown_tool``, ``role_mismatch``}。
    """
    cfgs = SUB_AGENT_CONFIGS if configs is None else configs
    known = known_tool_names()
    problems: List[Dict[str, Any]] = []
    per_agent: Dict[str, Any] = {}

    for agent_type, cfg in (cfgs or {}).items():
        declared = [str(n) for n in ((cfg or {}).get("tool_names") or []) if n]
        unknown = sorted({n for n in declared if known and n not in known})
        role_hint = _role_matrix(agent_type)
        mismatch = sorted(set(declared) ^ set(role_hint)) if role_hint else []
        per_agent[agent_type] = {
            "declared": sorted(declared),
            "unknown": unknown,
            "role_mismatch": mismatch,
        }
        if not declared:
            problems.append({
                "agent_type": agent_type, "kind": "no_tools",
                "detail": "tool_names 为空 —— 该子智能体将无工具可用",
            })
        for n in unknown:
            problems.append({
                "agent_type": agent_type, "kind": "unknown_tool",
                "detail": f"{n!r} 不在权威工具目录 TOOL_CATALOG 中（疑似拼写错误）",
            })
        if mismatch:
            problems.append({
                "agent_type": agent_type, "kind": "role_mismatch",
                "detail": f"与 ROLE_TOOLS[{agent_type!r}] 不一致：{mismatch}",
            })

    bad_agents = sorted({p["agent_type"] for p in problems})
    return {
        "ok": not problems,
        "known": sorted(known),
        "agents": per_agent,
        "problems": problems,
        "bad_agents": bad_agents,
    }


def assert_sub_agent_configs(
    configs: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """硬断言版（启动期断言）：配置有问题即抛 `SubAgentConfigError`。"""
    report = validate_sub_agent_configs(configs)
    if not report["ok"]:
        detail = "; ".join(
            f"{p['agent_type']}/{p['kind']}: {p['detail']}" for p in report["problems"]
        )
        raise SubAgentConfigError(
            f"子智能体白名单配置错误（{len(report['problems'])} 处）：{detail}"
        )
    return report


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
        "system_context": "prompt:sub_agent:rag_agent",
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
        "system_context": "prompt:sub_agent:matlab_agent",
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
        "system_context": "prompt:sub_agent:sim_agent",
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
        "system_context": "prompt:sub_agent:analysis_agent",
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
        "system_context": "prompt:sub_agent:memory_agent",
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
        self._policy = None                   # injected by set_policy() (P1-2)
        self._confirm_policy = None           # injected by set_confirm_policy() (P1-3)

    def set_sub_hermes(self, sub_hermes) -> None:
        """Inject the shared (already-initialized) sub_hermes."""
        self._sub_hermes = sub_hermes

    def set_policy(self, policy) -> None:
        """Inject the tool-call policy enforced at the execution boundary (P1-2).

        优先级：显式注入 > **吏部角色注册表**（P2-2，按 ``agent_type`` 授予计划级
        权限）> 内置 ``sub_agent_policy``（按 ``tool_guard.ROLE_TOOLS`` 推导）。
        """
        self._policy = policy

    def set_confirm_policy(self, policy) -> None:
        """Inject the confirmation policy enforced at the execution boundary (P1-3).

        优先级同上：显式注入 > 吏部角色注册表（P2-2，决定 ``deny_critical``）>
        内置 ``confirm_gate.sub_agent_policy``。
        """
        self._confirm_policy = policy

    def _effective_confirm_policy(self):
        """Current confirmation policy (injected, else role-scoped default).

        P2-2 吏部：先问角色注册表（它决定 ``deny_critical``），未登记再回落既有默认。
        """
        if self._confirm_policy is not None:
            return self._confirm_policy
        try:
            from multi_agent.security import role_registry as _rr
            _cp_reg = _rr.confirm_policy_for_role(
                self.agent_type, label=f"registry:{self.agent_type}",
            )
            if _cp_reg is not None:
                return _cp_reg
        except Exception:  # pragma: no cover — 注册表不可用则回落默认
            pass
        from multi_agent.security.confirm_gate import sub_agent_policy as _cp
        return _cp(self.agent_type)

    def _effective_policy(self):
        """Current tool-call policy (injected policy, else role-scoped default).

        P2-2 吏部：先问角色注册表（它授予**计划级权限**），未登记再回落既有默认。
        两条路径的 ``allowed_tools`` 都取**声明**的工具名，保证"能看见什么"与
        "许调什么"始终同源。
        """
        if self._policy is not None:
            return self._policy
        try:
            from multi_agent.security import role_registry as _rr
            _p_reg = _rr.tool_policy_for_role(
                self.agent_type,
                allowed_tools=frozenset(self.tool_names),
                label=f"registry:{self.agent_type}",
            )
            if _p_reg is not None:
                return _p_reg
        except Exception:  # pragma: no cover — 注册表不可用则回落默认
            pass
        from multi_agent.security.tool_guard import sub_agent_policy
        return sub_agent_policy(
            self.agent_type,
            allowed_tools=frozenset(self.tool_names),
        )

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
        available = {getattr(t, "name", "") for t in all_tools}
        declared = [n for n in (self.tool_names or []) if n]
        _declared_set = set(declared)
        subset_tools = [t for t in all_tools if getattr(t, "name", "") in _declared_set]
        missing = [n for n in declared if n not in available]

        # 部分未注册：用**已匹配子集**继续（子集 ⊆ 声明，永不放权），但告警留痕。
        # 注意这里绝不回退 `all_tools` —— 那正是 P1-4 点名要堵的静默降级。
        if missing and subset_tools:
            logger.warning(
                "[SubAgent:%s] 白名单部分未注册，按已匹配子集执行（不回退全量）："
                "declared=%s missing=%s",
                self.agent_type, declared, missing,
            )

        if not subset_tools:
            # ── P1-4：失败即拒绝 ──────────────────────────────────────────
            # 一个都匹配不上（拼写错误 / 未注册 / 清单为空）时，旧代码会
            # `subset_tools = all_tools or None` 把**全量工具**交给 LLM。
            # 严格模式下改为结构化拒绝，绝不接受这种降级。
            if is_strict_whitelist():
                logger.error(
                    "[SubAgent:%s] 白名单无法满足，拒绝执行（P1-4 fail-closed）。"
                    "declared=%s missing=%s available=%s",
                    self.agent_type, declared, missing,
                    sorted(n for n in available if n),
                )
                return json.dumps({
                    "status": "error",
                    "code": CODE_TOOLSET_UNAVAILABLE,
                    "agent_type": self.agent_type,
                    "declared_tools": declared,
                    "missing_tools": missing,
                    "message": (
                        "子智能体工具白名单未能解析出任何已注册工具，"
                        "已拒绝执行以避免越权暴露全量工具（P1-4 fail-closed）。"
                    ),
                }, ensure_ascii=False)

            logger.warning(
                "[SubAgent:%s] No tools matched from %d available. "
                "tool_names=%s  available=%s  "
                "(sub_agent_strict=false：回退全量工具，仅供消融对照)",
                self.agent_type, len(all_tools),
                self.tool_names, [t.name for t in all_tools],
            )
            # Fall back to unfiltered — 仅在严格模式关闭时可达
            subset_tools = all_tools or None

        # ── Bind the tool-call policy for the execution boundary (P1-2) ──
        # P1-4 之后本处只会拿到**声明的子集**（零匹配已在上面被拒），但闸门
        # 依旧保留：纵深防御 —— 即便 LLM 从历史消息 / 知识库正文得知域外工具名
        # 并试图调用，执行层照样拒绝。
        try:
            from multi_agent.security.tool_guard import policy_scope
        except Exception:  # pragma: no cover — guard 不可用时保持旧行为
            policy_scope = None
        # ── 绑定确认策略（P1-3）：写文件 / 跑仿真 / 删记忆需先确认 ──
        try:
            from multi_agent.security.confirm_gate import policy_scope as confirm_scope
        except Exception:  # pragma: no cover
            confirm_scope = None
        _policy = self._effective_policy()
        _cpol = self._effective_confirm_policy()

        logger.info(
            "[SubAgent:%s] run — tools: %s | role-policy: %s | task: %.80s",
            self.agent_type,
            [t.name for t in (subset_tools or [])],
            getattr(_policy, "label", None),
            task,
        )

        try:
            # 许可面（P1-2）在外，确认面（P1-3）在内 —— 与执行边界的判定顺序一致。
            if policy_scope is not None:
                with policy_scope(_policy):
                    if confirm_scope is not None:
                        with confirm_scope(_cpol):
                            result = await self._sub_hermes.run_with_tools(
                                full_msg,
                                tools=subset_tools,
                                verbose_thinking=False,
                            )
                    else:
                        result = await self._sub_hermes.run_with_tools(
                            full_msg,
                            tools=subset_tools,
                            verbose_thinking=False,
                        )
            elif confirm_scope is not None:
                with confirm_scope(_cpol):
                    result = await self._sub_hermes.run_with_tools(
                        full_msg,
                        tools=subset_tools,
                        verbose_thinking=False,
                    )
            else:
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
    strict: bool = False,
) -> Dict[str, SpecializedSubAgent]:
    """
    Create all specialized sub-agents backed by the shared sub_hermes.
    No registry changes — just wraps sub_hermes with different tool filters.

    P1-4 启动期校验
    ───────────────
    构建之前先跑 `validate_sub_agent_configs()`：声明的工具名必须都在权威目录
    中且与该角色矩阵一致。有问题时：

      · 始终以 ERROR 记录完整报告（**不静默**）；
      · 把有问题的子智能体**隔离**（不构建）—— 宁可该 agent_type 直接不可用，
        也不带着错误配置上线；
      · `strict=True` 时额外抛 `SubAgentConfigError`（供测试 / 严格启动使用）。

    之所以默认 `strict=False`：调用方（`HermesIntegration.initialize_with_tools`）
    对构建异常只记 warning，若在此直接抛出会导致**所有**子智能体都构建不出来，
    反而把每个具名委派都推给全量回退 —— 比隔离单个 agent 更糟。

    Args:
        sub_hermes: Already-initialized HermesIntegration (include_delegate=False).
        strict: 配置有问题时是否抛异常（默认否，只隔离 + ERROR）。
    """
    report = validate_sub_agent_configs()
    globals()["SUB_AGENT_CONFIG_REPORT"] = report
    if not report["ok"]:
        logger.error(
            "[SubAgentFactory] 子智能体白名单配置存在 %d 处问题，相关子智能体将被隔离：%s",
            len(report["problems"]),
            "; ".join(f"{p['agent_type']}/{p['kind']}: {p['detail']}" for p in report["problems"]),
        )
        if strict:
            raise SubAgentConfigError(
                f"子智能体白名单配置错误（{len(report['problems'])} 处）"
            )

    _bad = set(report["bad_agents"])
    agents: Dict[str, SpecializedSubAgent] = {}
    for agent_type, cfg in SUB_AGENT_CONFIGS.items():
        if agent_type in _bad:
            logger.error(
                "[SubAgentFactory] 隔离配置有误的子智能体 %s（fail-closed，不构建）",
                agent_type,
            )
            continue
        agent = SpecializedSubAgent(
            agent_type=agent_type,
            tool_names=cfg["tool_names"],
            system_context=resolve_system_context(cfg),
        )
        agent.set_sub_hermes(sub_hermes)
        agents[agent_type] = agent
        logger.info(
            "[SubAgentFactory] %s → tools: %s",
            agent_type, cfg["tool_names"],
        )
    return agents


# 最近一次启动期校验报告（供启动自检 / 诊断读取）；构建前为 None。
SUB_AGENT_CONFIG_REPORT: Optional[Dict[str, Any]] = None
