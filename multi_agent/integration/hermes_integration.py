#!/usr/bin/env python3
"""
Hermes Agent Integration Layer
"""

import asyncio
import os
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import warnings
warnings.filterwarnings("ignore", message="Could not import tool module")

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.propagate = False


HERMES_AVAILABLE = False
HermesIntegration = None


def load_env():
    """Load environment variables from .env file"""
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    os.environ.setdefault(key.strip(), value.strip())


load_env()

try:
    from run_agent import AIAgent
    from tools.registry import registry as _hermes_registry
    import inspect
    import json as _json

    HERMES_AVAILABLE = True

    # ── Monkey-patch model_tools._run_async for long-running tools ──────
    # The upstream Hermes ``model_tools._run_async`` has a hardcoded
    # ``future.result(timeout=300)`` in its thread-pool path (the path
    # taken whenever a tool handler is invoked from inside an async
    # context — which is *always* the case for our LLM-driven flow).
    # Our Octave-backed tools (``run_simulation``, ``matlab_rl_optimize``)
    # routinely run >300 s when executing the full KB Monte-Carlo +
    # robustness-analysis template, so the upstream timeout fires long
    # before our own 1200 s subprocess timeout has a chance to.
    #
    # We replace ``_run_async`` with a version that honours the env var
    # ``HERMES_TOOL_DISPATCH_TIMEOUT`` (default 1800 s — comfortably
    # above the 1200 s subprocess limit so the subprocess timeout fires
    # first with a meaningful message).  The patch is idempotent across
    # re-imports and falls through to the original implementation for
    # the no-running-loop / worker-thread paths (which have no timeout
    # issue and rely on persistent event loops).
    try:
        import model_tools as _model_tools_mod
        import asyncio as _asyncio_mt
        import concurrent.futures as _concurrent_futures_mt

        _ORIG_RUN_ASYNC = getattr(_model_tools_mod, "_run_async", None)

        def _patched_run_async(coro):
            """Drop-in replacement for ``model_tools._run_async`` with a
            configurable thread-pool timeout.

            Priority order:
              1. Env var HERMES_TOOL_DISPATCH_TIMEOUT (seconds; 0 = no timeout)
              2. config.yaml  workflow.hermes_tool_timeout
              3. Auto-computed: max_episodes * nmc_per_eval * 120 s (floor 3600 s)
            """
            _env_val = os.environ.get("HERMES_TOOL_DISPATCH_TIMEOUT", "")
            try:
                if _env_val.strip():
                    timeout_sec = float(_env_val)
                else:
                    try:
                        from multi_agent.config_loader import get_config
                        _cfg = get_config()
                        _wf_timeout = getattr(getattr(_cfg, "workflow", None),
                                              "hermes_tool_timeout", None)
                        if _wf_timeout is not None and float(_wf_timeout) >= 0:
                            timeout_sec = float(_wf_timeout)
                        else:
                            _rl = getattr(_cfg, "matlab_rl_optimizer", None)
                            _ep  = float(getattr(_rl, "max_episodes",  20))
                            _nmc = float(getattr(_rl, "nmc_per_eval",  10))
                            timeout_sec = max(3600.0, _ep * _nmc * 120.0)
                    except Exception:
                        timeout_sec = 3600.0
            except (TypeError, ValueError):
                timeout_sec = 3600.0
            # 0 means no timeout (wait forever)
            timeout_sec = None if timeout_sec == 0 else max(60.0, timeout_sec)

            try:
                loop = _asyncio_mt.get_running_loop()
            except RuntimeError:
                loop = None

            if loop and loop.is_running():
                # In an async context: spin a disposable thread (matches
                # upstream behaviour) but with our longer timeout.
                with _concurrent_futures_mt.ThreadPoolExecutor(max_workers=1) as pool:
                    future = pool.submit(_asyncio_mt.run, coro)
                    return future.result(timeout=timeout_sec)  # None = no timeout

            # Sync context: defer to the original implementation, which
            # uses persistent per-thread event loops and has no timeout.
            if _ORIG_RUN_ASYNC is not None:
                return _ORIG_RUN_ASYNC(coro)
            # Fallback if the upstream API ever changes
            return _asyncio_mt.run(coro)

        # Idempotency: tag the patched function so re-imports don't
        # double-wrap.  ``dispatch`` re-imports ``_run_async`` on every
        # call, so once we install the patch on the module attribute,
        # all subsequent dispatches pick it up automatically.
        if (
            _ORIG_RUN_ASYNC is not None
            and not getattr(_model_tools_mod._run_async, "_gs_patched", False)
        ):
            _patched_run_async._gs_patched = True
            _model_tools_mod._run_async = _patched_run_async
            logger.info(
                "Patched model_tools._run_async — tool dispatch timeout "
                f"= {os.environ.get('HERMES_TOOL_DISPATCH_TIMEOUT', '1800')}s "
                "(was hardcoded 300s upstream)"
            )
    except Exception as _patch_exc:  # pragma: no cover  (best-effort)
        logger.warning(
            f"Could not patch model_tools._run_async ({_patch_exc}); "
            f"long-running Octave tools may time out at the upstream 300 s limit."
        )

    # ── Hermes tool registration (official pattern) ──────────────────────
    # Hermes' run_agent dispatches LLM tool calls via
    # ``handle_function_call`` → ``tools.registry.dispatch``.  Tools must
    # therefore be registered through ``tools.registry.register(...)``.
    # The registry natively bridges async handlers (set ``is_async=True``)
    # so we don't need our own thread-pool / event-loop juggling.
    #
    # Tracks which tool *names* this process has registered so we can
    # deregister cleanly on re-initialisation (e.g. across CLI reruns in
    # the same Python process).
    _REGISTERED_TOOL_NAMES: List[str] = []

    def _make_async_handler(tool: Any):
        """Build the ``async def handler(args, **kw) -> str`` closure expected
        by ``tools.registry.register`` for *tool*.

        Calls ``tool.execute(**args)`` (async) and returns a JSON string,
        which is the contract enforced by ``registry.dispatch``.  All
        exceptions are caught and surfaced as a JSON error payload so the
        LLM sees a structured error rather than a crash.
        """
        async def _handler(args, **_kw):
            try:
                kw = dict(args) if isinstance(args, dict) else {}

                # ── 工具调用边界权限闸门（P1-2）────────────────────────────
                # `run_with_tools(tools=subset)` 只把子集 Schema 塞进
                # `agent.tools`，约束的是 LLM **看见** 什么；而
                # `registry.dispatch` 用的是 AIAgent 构造时快照的
                # `valid_tool_names`（**全量**）。此处才是真正的执行边界：
                # 越权调用（幻觉 / 提示注入）在此被拦下，绝不落到
                # `tool.execute()`。策略由调用方经
                # `security.tool_guard.policy_scope(...)` 绑定；未绑定则透传。
                try:
                    from multi_agent.security.tool_guard import guard_call
                    _g = guard_call(getattr(tool, "name", ""))
                    if not _g.allowed:
                        return _json.dumps({
                            "status":  "denied",
                            "tool":    getattr(tool, "name", ""),
                            "code":    _g.code,
                            "role":    _g.role,
                            "message": f"工具调用被权限闸门拒绝：{_g.reason}",
                        }, ensure_ascii=False)
                except Exception as _ge:  # pragma: no cover — 门禁接线异常不得阻断工具
                    logger.error("[ToolGuard] wiring error, call passed through: %s", _ge)

                # ── 高风险 / 不可逆操作确认闸门（P1-3）──────────────────────
                # 许可面（上一段）通过之后，还要过**确认面**：写文件 / 跑仿真 /
                # 删记忆等动作在获得人工确认前不得执行。未获确认时返回
                # `pending_confirmation`（含确定性确认号），人工经 web 端批准后
                # 以同参数重试即放行（一次性消耗）。策略由调用方经
                # `security.confirm_gate.policy_scope(...)` 绑定；未绑定则透传。
                try:
                    from multi_agent.security.confirm_gate import guard_confirm
                    _c = guard_confirm(getattr(tool, "name", ""), kw)
                    if not _c.allowed:
                        _is_pending = _c.code == "pending_confirmation"
                        return _json.dumps({
                            "status":          "pending_confirmation" if _is_pending else "denied",
                            "tool":            getattr(tool, "name", ""),
                            "code":            _c.code,
                            "risk":            _c.risk,
                            "role":            _c.role,
                            "confirmation_id": _c.confirmation_id,
                            "message":         (
                                f"高风险操作需人工确认（确认号 {_c.confirmation_id}），"
                                f"本次未执行：{_c.reason}"
                                if _is_pending else
                                f"高风险操作被确认闸门拒绝：{_c.reason}"
                            ),
                        }, ensure_ascii=False)
                except Exception as _ce:  # pragma: no cover — 门禁接线异常不得阻断工具
                    logger.error("[ConfirmGate] wiring error, call passed through: %s", _ce)

                exec_fn = tool.execute
                if inspect.iscoroutinefunction(exec_fn):
                    result = await exec_fn(**kw)
                else:
                    result = exec_fn(**kw)
                # ── P2-5 兵部：工具返回类型统一（Dict / JSON 字符串混用 → 强制 JSON）──
                # 差距原文：「返回类型 Dict 与 JSON 字符串混用，未强制 JSON」。
                # `registry.dispatch` 的契约是"JSON 字符串"，但旧实现只把 `str`
                # **原样返回** —— 于是工具的**裸文本**返回（如 call_specialist 透传的
                # 子智能体自由输出）不经任何 JSON 包装直接漏给 LLM，契约并未被强制。
                # 现统一走 `tool_result.to_wire()`：
                #   · 默认 audit —— 与旧行为**逐字等价**（str 原样 / 其余 dumps）；
                #   · `ablation.tool_result_strict=true` —— 偏离声明形态即结构化拒绝。
                # 该层是旁路：任何异常都不得影响工具主流程（吞掉并退回旧行为）。
                try:
                    from multi_agent.tools import tool_result as _tr
                    return _tr.to_wire(result, getattr(tool, "name", ""))
                except Exception as _tre:  # pragma: no cover — 归一不可用即退回旧行为
                    logger.debug("[ToolResult] normalize unavailable, legacy path: %s", _tre)
                if isinstance(result, str):
                    return result
                return _json.dumps(result, ensure_ascii=False, default=str)
            except (asyncio.CancelledError, KeyboardInterrupt):
                raise
            except TypeError as exc:
                # Most likely an argument-name mismatch from the LLM.
                logger.error(
                    f"Tool '{tool.name}' arg error: {exc}", exc_info=True
                )
                return _json.dumps({
                    "status":  "error",
                    "tool":    tool.name,
                    "message": f"argument error: {exc}",
                })
            except Exception as exc:
                logger.error(
                    f"Tool '{tool.name}' raised: {exc}", exc_info=True
                )
                return _json.dumps({
                    "status":  "error",
                    "tool":    tool.name,
                    "message": f"{type(exc).__name__}: {exc}",
                })

        return _handler

    def _register_tool_with_hermes(
        tool: Any,
        toolset: str = "guidance_system",
    ) -> bool:
        """Register *tool* with ``tools.registry`` using the official API.

        Idempotent — deregisters any previous binding under the same name
        first so re-running ``initialize_with_tools`` (or instantiating
        multiple ``HermesIntegration``s) doesn't pile up stale handlers.
        """
        name = getattr(tool, "name", None)
        if not name or not hasattr(tool, "execute"):
            logger.warning(
                f"Skipping tool registration: missing .name or .execute on {type(tool).__name__}"
            )
            return False

        schema = {
            "name":        name,
            "description": getattr(tool, "description", "") or "",
            "parameters":  getattr(tool, "input_schema", {"type": "object", "properties": {}}),
        }

        # Idempotent: drop any prior registration for this name.
        try:
            _hermes_registry.deregister(name)
        except Exception:  # pragma: no cover  (deregister API is permissive)
            pass

        try:
            _hermes_registry.register(
                name=name,
                toolset=toolset,
                schema=schema,
                handler=_make_async_handler(tool),
                is_async=True,
                description=schema["description"],
            )
        except Exception as exc:
            logger.error(f"Failed to register tool '{name}' with Hermes registry: {exc}")
            return False

        if name not in _REGISTERED_TOOL_NAMES:
            _REGISTERED_TOOL_NAMES.append(name)
        return True

    class HermesIntegration:
        """Hermes Agent wrapper"""

        def __init__(
            self,
            model: str = None,
            provider: str = None,
            api_key: str = None,
            max_history_turns: int = 3,
        ):
            from multi_agent.llm_failover import ensure_usable_endpoint

            # Resolve a usable endpoint (probe primary; auto-switch on balance fail).
            ep = ensure_usable_endpoint(probe=True)
            self.model = model or ep.model
            self.provider = provider or ep.provider
            self.api_key = api_key or ep.api_key
            self.base_url = ep.base_url
            self.agent: Optional[Any] = None
            self._initialized = False
            # Cross-iteration conversation history.
            # Stores non-system messages from previous run_with_tools() calls
            # and passes them as conversation_history= on the next call so
            # the LLM retains context across outer-loop iterations.
            self._prior_messages: List[Dict[str, Any]] = []
            # Each "turn" = 1 user msg + N assistant/tool msgs.  We keep
            # at most max_history_turns * 8 messages to bound context growth.
            self.max_history_turns: int = max(1, max_history_turns)
            # Optional reference to HermesAgentMemory for conversation sync.
            self._hermes_memory: Optional[Any] = None

        @staticmethod
        def _make_thinking_callbacks(verbose: bool = True):
            """Return a dict of callbacks that print the agent's reasoning in real time."""
            import sys

            def _thinking_cb(text: str) -> None:
                if text and verbose:
                    print(f"  [Thinking] {text[:200]}", flush=True)

            def _step_cb(step_info) -> None:
                if verbose:
                    msg = str(step_info)[:120] if step_info else ""
                    if msg:
                        print(f"  [Step] {msg}", flush=True)

            def _tool_start_cb(tool_name: str, *args) -> None:
                if verbose:
                    print(f"  [Tool →] {tool_name}", flush=True)

            def _tool_done_cb(tool_name: str, *args) -> None:
                if verbose:
                    print(f"  [Tool ✓] {tool_name}", flush=True)

            def _stream_cb(delta) -> None:
                if delta and verbose:
                    print(delta, end="", flush=True)

            return {
                "thinking_callback":    _thinking_cb,
                "step_callback":        _step_cb,
                "tool_start_callback":  _tool_start_cb,
                "tool_complete_callback": _tool_done_cb,
                "stream_delta_callback": _stream_cb,
            }

        async def initialize(self) -> bool:
            if not HERMES_AVAILABLE:
                return False
            try:
                from multi_agent.llm_failover import apply_endpoint, get_active_endpoint

                ep = get_active_endpoint()
                apply_endpoint(ep)
                self.model = self.model or ep.model
                self.provider = self.provider or ep.provider
                self.api_key = self.api_key or ep.api_key
                self.base_url = ep.base_url

                self.agent = AIAgent(
                    base_url=self.base_url,
                    api_key=self.api_key,
                    provider=self.provider,
                    model=self.model,
                    max_iterations=20,
                )
                self._initialized = True
                logger.info(f"Hermes initialized: {self.model} @ {self.base_url}")
                return True
            except Exception as e:
                logger.error(f"Failed to initialize Hermes: {e}")
                return False

        def reset_conversation(self) -> None:
            """Clear cross-iteration history (call at task start or when context is stale)."""
            self._prior_messages.clear()
            if self._hermes_memory is not None:
                self._hermes_memory.clear_conversation()
            logger.info("[Hermes] Conversation history cleared.")

        @property
        def conversation_history(self) -> List[Dict[str, Any]]:
            """Read-only view of the accumulated cross-iteration message history."""
            return list(self._prior_messages)

        async def run_with_tools(
            self,
            user_message: str,
            tools: List[Any] = None,
            verbose_thinking: bool = True,
        ) -> str:
            if tools is None:
                tools = []
            if not self._initialized:
                await self.initialize()
            if not self.agent:
                return "Hermes Agent not available"
            try:
                if tools:
                    formatted_tools = []
                    for tool in tools:
                        if hasattr(tool, "input_schema") and hasattr(tool, "name") and hasattr(tool, "description"):
                            formatted_tools.append({
                                "type": "function",
                                "function": {
                                    "name": tool.name,
                                    "description": tool.description,
                                    "parameters": tool.input_schema
                                }
                            })
                        else:
                            formatted_tools.append(tool)
                    self.agent.tools = formatted_tools

                # Attach real-time thinking callbacks
                cbs = self._make_thinking_callbacks(verbose=verbose_thinking)
                for attr, fn in cbs.items():
                    if hasattr(self.agent, attr):
                        setattr(self.agent, attr, fn)

                # Pass cross-iteration history so the LLM retains context.
                # Strip system messages — they are regenerated fresh each call
                # in the caller's full_msg / system_message argument.
                _prior = [m for m in self._prior_messages if m.get("role") != "system"]

                from multi_agent.llm_failover import (
                    is_balance_error,
                    note_failure_and_switch,
                    get_active_endpoint,
                    apply_endpoint,
                )

                async def _once():
                    import inspect
                    if inspect.iscoroutinefunction(self.agent.run_conversation):
                        return await self.agent.run_conversation(
                            user_message,
                            conversation_history=_prior if _prior else None,
                        )
                    return self.agent.run_conversation(
                        user_message,
                        conversation_history=_prior if _prior else None,
                    )

                result = await _once()
                # Hermes often returns balance errors as text instead of raising.
                _text_probe = ""
                if isinstance(result, dict):
                    _text_probe = str(result.get("final_response") or "")
                else:
                    _text_probe = str(result or "")
                if is_balance_error(_text_probe) or is_balance_error(result):
                    new_ep = note_failure_and_switch(_text_probe or result)
                    if new_ep is not None:
                        apply_endpoint(new_ep)
                        self.model = new_ep.model
                        self.provider = new_ep.provider
                        self.api_key = new_ep.api_key
                        self.base_url = new_ep.base_url
                        self._initialized = False
                        await self.initialize()
                        if self.agent:
                            result = await _once()

                # ── Update cross-iteration history ────────────────────────
                if isinstance(result, dict):
                    _new_msgs: List[Dict] = [
                        m for m in result.get("messages", [])
                        if m.get("role") != "system"
                    ]
                    # Sliding window: keep at most max_history_turns * 8 msgs.
                    _cap = self.max_history_turns * 8
                    self._prior_messages = _new_msgs[-_cap:] if len(_new_msgs) > _cap else _new_msgs
                    # Sync to HermesAgentMemory.conversation for external access.
                    if self._hermes_memory is not None:
                        self._hermes_memory.clear_conversation()
                        for _m in self._prior_messages:
                            self._hermes_memory.add_message(
                                _m.get("role", "user"), _m.get("content", "")
                            )
                    logger.debug(
                        "[Hermes] History updated: %d msgs retained (cap=%d)",
                        len(self._prior_messages), _cap,
                    )
                    return result.get("final_response", str(result))

                return result
            except (asyncio.CancelledError, KeyboardInterrupt):
                raise
            except Exception as e:
                from multi_agent.llm_failover import is_balance_error, note_failure_and_switch, apply_endpoint
                if is_balance_error(e):
                    new_ep = note_failure_and_switch(e)
                    if new_ep is not None:
                        try:
                            apply_endpoint(new_ep)
                            self.model = new_ep.model
                            self.provider = new_ep.provider
                            self.api_key = new_ep.api_key
                            self.base_url = new_ep.base_url
                            self._initialized = False
                            await self.initialize()
                            if self.agent:
                                return await self.run_with_tools(
                                    user_message,
                                    tools=tools,
                                    verbose_thinking=verbose_thinking,
                                )
                        except Exception as e2:
                            logger.error(f"Conversation failover failed: {e2}")
                logger.error(f"Conversation failed: {e}")
                return None

        async def run_with_context(
            self,
            user_message: str,
            context: str = "",
            verbose_thinking: bool = True,
        ) -> str:
            """
            Run Hermes with a pre-built context block prepended to the user message.
            Context may include RAG retrieval and ParameterExperience history.
            """
            full_message = user_message
            if context:
                full_message = (
                    f"=== 参考上下文（知识库与历史经验检索结果）===\n{context}\n"
                    f"{'='*60}\n\n{user_message}"
                )
            return await self.run_with_tools(
                full_message, verbose_thinking=verbose_thinking
            )

        async def generate_text(self, prompt: str) -> str:
            """Simple text generation using the agent"""
            return await self.run_with_tools(prompt, tools=[])

        def get_all_tools(self, include_delegate: bool = True) -> List[Any]:
            """Get all available tools from the tools directory.

            Layer 2 Hermes tool set (NO RL tool):
              RL optimization is exclusively handled by Layer 3 OptimizationWorkflow.
              1. rag_retrieve             — unified KB + PE search
              2. generate_matlab          — generate .m script (monte_carlo_single template)
              3. syntax_check_matlab      — validate/fix .m file
              4. run_simulation           — run monte_carlo_single simulation
              5. verify_guidance_compat   — verify modified guidance law is compatible with autopilot/CF
              6. judge_requirements       — check if sim metrics satisfy task requirements
              7. parameter_experience_*   — PE search/best
              8. agent_memory_remember    — persist info to long-term memory (JSON)
              9. agent_memory_recall      — retrieve from long-term memory
             10. agent_memory_list        — list all remembered keys
             11. agent_memory_forget      — delete a memory entry
            """
            from ..tools import (
                RAGRetrievalTool,
                RAGExpandTool,
                RAGAgentTool,
                ParameterExperienceSearchTool,
                ParameterExperienceBestTool,
                GenerateMATLABTool,
                GuidanceLawCompatibilityTool,
                RunSimulationTool,
                AgentMemoryRememberTool,
                AgentMemoryRecallTool,
                AgentMemoryListTool,
                AgentMemoryForgetTool,
                DelegateSubAgentTool,
                RequirementToIRTool,
                CadGeometryTool,
                AeroAnalysisTool,
                SimulateModelicaTool,
                RenderSSimTool,
            )
            from ..tools.syntax_check_tool import SyntaxCheckMATLABTool
            from ..tools.judge_requirements_tool import JudgeRequirementsTool

            tools = [
                RAGRetrievalTool(),
                RAGExpandTool(),
                RAGAgentTool(),
                ParameterExperienceSearchTool(),
                ParameterExperienceBestTool(),
                GenerateMATLABTool(),
                GuidanceLawCompatibilityTool(),
                SyntaxCheckMATLABTool(),
                RunSimulationTool(),
                JudgeRequirementsTool(),
                # Persistent memory tools (hermes-agent-demo pattern)
                AgentMemoryRememberTool(),
                AgentMemoryRecallTool(),
                AgentMemoryListTool(),
                AgentMemoryForgetTool(),
                # 阶段 2：需求 → 工具链 → SSim（由「需求执行」调度器在
                # 已批准的计划下逐任务调用；受 tool_guard + confirm_gate 管住）
                RequirementToIRTool(),
                CadGeometryTool(),
                AeroAnalysisTool(),
                SimulateModelicaTool(),
                RenderSSimTool(),
                # matlab_rl_optimize intentionally excluded:
                # RL is exclusively Layer 3 (OptimizationWorkflow), not Hermes.
            ]
            if include_delegate:
                tools.append(DelegateSubAgentTool())
            return tools

        def format_tools_for_agent(self, tools: List[Any]) -> List[Dict[str, Any]]:
            """Format tools to JSON schema format expected by the LLM client"""
            formatted_tools = []
            for tool in tools:
                if hasattr(tool, "input_schema") and hasattr(tool, "name") and hasattr(tool, "description"):
                    formatted_tools.append({
                        "type": "function",
                        "function": {
                            "name": tool.name,
                            "description": tool.description,
                            "parameters": tool.input_schema
                        }
                    })
                else:
                    formatted_tools.append(tool)
            return formatted_tools

        async def initialize_with_tools(
            self,
            rag_kb=None,
            parameter_experience=None,
            simulator=None,
            optimizer=None,
            orchestrator=None,
            reflection_agent=None,
            user_prompt: Optional[str] = None,
            ablation=None,
            hermes_agent_memory=None,
            include_delegate: bool = True,
        ) -> bool:
            """Initialize Hermes agent with all tools registered.

            ``user_prompt`` is forwarded to every tool exposing
            ``set_task_prompt`` (e.g. ``MatlabRLOptimizationTool``) so that
            per-episode reflection inside the RL loop can evaluate the user's
            full task requirements without the LLM having to forward the
            prompt as a tool argument.
            """
            if not HERMES_AVAILABLE:
                return False
            try:
                # ── 1. Build tool list and inject runtime dependencies ────
                # Done BEFORE the registry registration so each tool has
                # its rag_kb / parameter_experience / simulator / etc.
                # references in place when the LLM eventually calls it.
                tools = self.get_all_tools(include_delegate=include_delegate)

                # ── Ablation: remove tools for disabled components ────────
                # Tool names excluded per flag:
                #   parameter_experience_reuse           → "parameter_experience_*"
                #   guidance_compat_verification         → "verify_guidance_compat"
                #   syntax_check                         → "syntax_check_matlab"
                #   （P2-7：`hermes_rl_tool` 已删除 —— 它是死开关，全仓无读取点；
                #     matlab_rl_optimize 本就不在 Hermes 工具集里。）
                _pe_tool_names = {
                    "parameter_experience_search",
                    "parameter_experience_best",
                }
                if ablation is not None:
                    _removed: list = []
                    _memory_tool_names = {
                        "agent_memory_remember",
                        "agent_memory_recall",
                        "agent_memory_list",
                        "agent_memory_forget",
                        "session_search",
                    }
                    def _abl_keep(t) -> bool:
                        n = getattr(t, "name", "")
                        # matlab_rl_optimize is no longer in the Hermes tool set;
                        # RL is exclusively in Layer 3 OptimizationWorkflow.
                        if n in _pe_tool_names and not ablation.parameter_experience_reuse:
                            return False
                        if n in _memory_tool_names and not ablation.memory_search:
                            return False
                        if n == "verify_guidance_compat" and not ablation.guidance_compat_verification:
                            return False
                        if n == "syntax_check_matlab" and not ablation.syntax_check:
                            return False
                        return True
                    filtered = [t for t in tools if _abl_keep(t)]
                    _removed = [getattr(t, "name", "?") for t in tools if not _abl_keep(t)]
                    if _removed:
                        logger.info(
                            f"[Ablation] Excluded {len(_removed)} tools from Hermes: {sorted(_removed)}"
                        )
                    tools = filtered

                self._tools = tools
                self._user_prompt = user_prompt
                self._hermes_memory = hermes_agent_memory

                # ── Tool guard (P1-2): drive the execution-boundary gate ──
                # `ablation.tool_guard=False` disables the gate entirely
                # (behaviour identical to before P1-2).  Also surface any
                # registered tool missing from the catalogue at start-up:
                # under a strict policy such a tool would be denied, so it
                # must be caught during boot rather than at first call.
                try:
                    from multi_agent.security import tool_guard as _tg
                    _tg.set_enabled(
                        bool(getattr(ablation, "tool_guard", True))
                        if ablation is not None else True
                    )
                    _uncatalogued = sorted(
                        n for n in (getattr(t, "name", "") for t in tools)
                        if n and n not in _tg.TOOL_CATALOG
                    )
                    if _uncatalogued:
                        logger.warning(
                            "[ToolGuard] tools registered but absent from TOOL_CATALOG "
                            "(strict policies will deny them): %s", _uncatalogued,
                        )
                except Exception as _tge:  # pragma: no cover
                    logger.warning("[ToolGuard] init wiring failed: %s", _tge)

                # ── 确认闸门（P1-3）：装载模式 / 一次性授权 / 落盘路径 ──────
                # `ablation.confirm_gate.enabled=False` 或 `mode=off` → 透传。
                # `mode=enforce`（默认）下，未被 standing 覆盖的高风险调用
                # （尤其**不可逆**的 agent_memory_forget）一律先返回
                # pending_confirmation，不执行。
                try:
                    from multi_agent.security import confirm_gate as _cg
                    _cg_enabled = (
                        bool(getattr(ablation, "confirm_gate", True))
                        if ablation is not None else True
                    )
                    if not _cg_enabled:
                        _cg.set_mode(_cg.MODE_OFF)
                    else:
                        _cg.set_mode(str(getattr(ablation, "confirm_mode", "") or _cg.MODE_ENFORCE))
                        _cg.set_deny_critical(bool(getattr(ablation, "confirm_deny_critical", False)))
                        _standing = getattr(ablation, "confirm_standing", None)
                        if _standing is None:
                            _standing = list(_cg.DEFAULT_STANDING_TOOLS)
                        _cg.grant_standing(_standing, by="ablation.confirm_gate.standing")
                    try:
                        _cg.set_store_path(_cg.default_store_path())
                    except Exception:  # pragma: no cover — 落盘不可用不影响内存闸门
                        pass
                except Exception as _cge:  # pragma: no cover
                    logger.warning("[ConfirmGate] init wiring failed: %s", _cge)

                for tool in tools:
                    # RAG knowledge base (covers GenerateSysMLTool, GenerateMATLABTool, RAGRetrievalTool, etc.)
                    if rag_kb and hasattr(tool, "set_rag_kb"):
                        tool.set_rag_kb(rag_kb)
                    if parameter_experience and hasattr(tool, "set_parameter_experience"):
                        tool.set_parameter_experience(parameter_experience)
                    if simulator and hasattr(tool, "set_simulator"):
                        tool.set_simulator(simulator)
                    if optimizer and hasattr(tool, "set_optimizer"):
                        tool.set_optimizer(optimizer)
                    if orchestrator and hasattr(tool, "set_orchestrator"):
                        tool.set_orchestrator(orchestrator)
                    if reflection_agent and hasattr(tool, "set_reflection_agent"):
                        tool.set_reflection_agent(reflection_agent)
                    if user_prompt and hasattr(tool, "set_task_prompt"):
                        tool.set_task_prompt(user_prompt)
                    if hermes_agent_memory is not None and hasattr(tool, "set_memory"):
                        tool.set_memory(hermes_agent_memory)

                _run_sim_tool = next(
                    (t for t in tools if getattr(t, "name", "") == "run_simulation"),
                    None,
                )
                _judge_tool = next(
                    (t for t in tools if getattr(t, "name", "") == "judge_requirements"),
                    None,
                )
                if _run_sim_tool and _judge_tool and hasattr(
                    _judge_tool, "set_run_simulation_tool"
                ):
                    _judge_tool.set_run_simulation_tool(_run_sim_tool)

                # ── 1b. Wire sub-agent into DelegateSubAgentTool ──────────
                # Create a separate HermesIntegration instance for sub-tasks.
                # include_delegate=False prevents infinite recursion.
                if include_delegate:
                    _delegate_tool = next(
                        (t for t in tools if getattr(t, "name", "") == "call_specialist"),
                        None,
                    )
                    if _delegate_tool is not None:
                        try:
                            sub_hermes = HermesIntegration(
                                api_key=self.api_key,
                                provider=self.provider,
                                model=self.model,
                            )
                            await sub_hermes.initialize_with_tools(
                                rag_kb=rag_kb,
                                parameter_experience=parameter_experience,
                                simulator=simulator,
                                optimizer=optimizer,
                                orchestrator=orchestrator,
                                reflection_agent=reflection_agent,
                                user_prompt=user_prompt,
                                ablation=ablation,
                                hermes_agent_memory=hermes_agent_memory,
                                include_delegate=False,  # no nested delegation
                            )
                            _delegate_tool.set_sub_agent(sub_hermes)
                            logger.info("[DelegateSubAgent] Hermes fallback sub-agent wired.")

                            # ── Build specialized sub-agents ──────────────
                            # Each has only its domain tools; no call_specialist
                            # to prevent recursion.  They share the same LLM
                            # config as the parent but have unique toolsets.
                            try:
                                from multi_agent.integration.sub_agent import (
                                    build_specialized_agents, set_strict_whitelist,
                                )
                                from multi_agent.tools.delegate_tool import (
                                    set_allow_specialist_escalation,
                                )
                                # ── P1-4：白名单严格模式（失败即拒绝）─────────
                                # 同一个开关同时驱动两个模块：sub_agent 的"零匹配
                                # 拒绝"与 delegate 的"具名专家失败不升级全量"。
                                _strict = (
                                    bool(getattr(ablation, "sub_agent_strict", True))
                                    if ablation is not None else True
                                )
                                set_strict_whitelist(_strict)
                                set_allow_specialist_escalation(not _strict)
                                if not _strict:
                                    logger.warning(
                                        "[SubAgentFactory] sub_agent_strict=false："
                                        "已恢复「无匹配回退全量工具」（仅供消融对照）"
                                    )
                                # Collect all non-delegate tools (already dep-injected above)
                                _spec_agents = build_specialized_agents(
                                    sub_hermes=sub_hermes,
                                )
                                _delegate_tool.register_specialized_agents(_spec_agents)
                                logger.info(
                                    "[DelegateSubAgent] %d specialized agents ready: %s",
                                    len(_spec_agents), list(_spec_agents.keys()),
                                )
                            except Exception as _se:
                                logger.warning(
                                    "[DelegateSubAgent] Specialized agent build failed: %s", _se
                                )
                        except Exception as _de:
                            logger.warning(f"[DelegateSubAgent] Sub-agent init failed: {_de}")

                # ── 2. Register every tool with Hermes' tool registry ────
                # CRITICAL: this MUST happen BEFORE AIAgent() is instantiated.
                # AIAgent.__init__ snapshots ``valid_tool_names`` from the
                # registry once and never re-reads it; tools registered
                # after construction are silently filtered out by the
                # ``enabled_tools=valid_tool_names`` argument that
                # ``_invoke_tool`` passes into ``registry.dispatch``.
                _registered = []
                for tool in tools:
                    if _register_tool_with_hermes(tool, toolset="guidance_system"):
                        _registered.append(tool.name)
                logger.info(
                    f"Registered {len(_registered)} tools with Hermes registry "
                    f"(toolset='guidance_system'): {sorted(_registered)}"
                )

                # ── 3. NOW create the AIAgent ─────────────────────────────
                # ``enabled_toolsets=['guidance_system']`` filters out the
                # default 26 tools (browser_*, patch, process, …) so the
                # LLM only sees our domain tools.  Agent-level tools that
                # are dispatched in ``_invoke_tool`` BEFORE
                # ``handle_function_call`` (todo, clarify, the built-in delegate_task,
                # session_search, memory) are still available because
                # they bypass the registry filter entirely.
                # NOTE: our orchestration tool is named 'call_specialist'
                # (not 'delegate_task') to avoid colliding with that built-in.
                cbs = self._make_thinking_callbacks(verbose=True)
                from multi_agent.llm_failover import apply_endpoint, get_active_endpoint
                ep = get_active_endpoint()
                apply_endpoint(ep)
                self.model = self.model or ep.model
                self.provider = self.provider or ep.provider
                self.api_key = self.api_key or ep.api_key
                self.base_url = ep.base_url
                self.agent = AIAgent(
                    base_url=self.base_url,
                    api_key=self.api_key,
                    provider=self.provider,
                    model=self.model,
                    max_iterations=20,
                    enabled_toolsets=["guidance_system"],
                    thinking_callback=cbs["thinking_callback"],
                    step_callback=cbs["step_callback"],
                    tool_start_callback=cbs["tool_start_callback"],
                    tool_complete_callback=cbs["tool_complete_callback"],
                    stream_delta_callback=cbs["stream_delta_callback"],
                )

                # Sanity-check: every registered tool name must now appear
                # in the agent's ``valid_tool_names`` set.  Log a warning
                # if anything is missing — this means the registration
                # didn't reach the agent and the LLM won't be able to
                # call those tools.
                _agent_known = set(getattr(self.agent, "valid_tool_names", set()) or set())
                _missing = [n for n in _registered if n not in _agent_known]
                if _missing:
                    logger.warning(
                        f"Tools registered but NOT picked up by AIAgent "
                        f"(will be invisible to LLM): {_missing}"
                    )
                else:
                    logger.info(
                        f"AIAgent.valid_tool_names contains all "
                        f"{len(_registered)} registered tools."
                    )

                self._initialized = True
                logger.info(
                    f"Hermes initialized with {len(tools)} tools "
                    f"(task_prompt={'yes' if user_prompt else 'no'})"
                )
                return True
            except Exception as e:
                logger.error(f"Failed to initialize Hermes with tools: {e}")
                return False

        def update_user_prompt(self, user_prompt: Optional[str]) -> None:
            """Update the cached user prompt and re-inject into all tools that
            support ``set_task_prompt``. Useful when the prompt evolves between
            iterations (e.g. with reflection feedback appended)."""
            self._user_prompt = user_prompt
            for tool in getattr(self, "_tools", []) or []:
                if hasattr(tool, "set_task_prompt"):
                    tool.set_task_prompt(user_prompt)

        def update_mission_conditions(self, conditions_str: str) -> None:
            """Re-inject per-iteration mission conditions into all tools that
            support ``set_mission_conditions``.  Called by cli_agent after
            ``matlab_cond_str`` is resolved each iteration so that Hermes tool
            calls that omit the conditions argument still run only the
            user-specified categories rather than the template's all-enabled
            defaults (which cause timeouts)."""
            for tool in getattr(self, "_tools", []) or []:
                if hasattr(tool, "set_mission_conditions"):
                    tool.set_mission_conditions(conditions_str)

except ImportError as e:
    logger.warning(f"Hermes Agent not available: {e}")
    HermesIntegration = None
