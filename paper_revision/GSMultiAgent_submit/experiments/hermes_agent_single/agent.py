#!/usr/bin/env python3
"""
hermes_agent_single/agent.py — Section 4.2.2 单智能体 Hermes Agent.

对比设计
--------
GSMultiAgent  : Layer1 → Layer2 (Hermes 多智能体级联) → Layer3 (优化)
HermesAgent单 : Layer1 → 单一 Hermes Agent（所有工具，无子智能体委托）→ Layer3 (优化)

与 GSMultiAgent 相同:
  ✓ 同一基座模型 (config.yaml llm.*)
  ✓ 同一相似度检索经验库 (ParameterExperience)
  ✓ 同一 MATLAB Engine 仿真接口
  ✓ 同一 PPO / optimizer_backend 优化技术
  ✓ 同一迭代上限 (workflow.max_iterations)
  ✓ 全套工具: generate_matlab / run_simulation / syntax_check /
              rag_retrieve / parameter_experience_* / judge_requirements

与 GSMultiAgent 不同（被移除的多智能体级联机制）:
  ✗ 无 call_specialist 子智能体委托    (include_delegate=False)
  ✗ 无 ReflectionAgent 结构化逐约束反思 (ablation.reflection_agent=False)
  ✗ 无 Hermes 持久记忆注入              (ablation.memory_search=False)
  ✗ 无角色分离 / 结构化智能体间协议

记录 LLM 推理调用次数 (llm_calls) 作为算力对等性诊断指标。
"""
from __future__ import annotations

import asyncio
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ── Result dataclass (同 BaselineRunResult 格式) ──────────────────────────────

@dataclass
class HermesAgentSingleResult:
    method: str = "hermes_agent_single"
    scenario: str = ""
    master_seed: int = 0
    prompt_text: str = ""
    iterations_budget: int = 0
    iterations_used: int = 0
    first_success_iter: Optional[int] = None
    first_success: bool = False
    dev_all_pass: bool = False
    test_all_pass: bool = False
    wall_clock_sec: float = 0.0
    sim_time_sec: float = 0.0
    mc_calls_dev: int = 0
    mc_calls_test: int = 0
    best_score: float = float("inf")
    best_params: Dict[str, float] = field(default_factory=dict)
    best_dev: Dict[str, Any] = field(default_factory=dict)
    test_eval: Dict[str, Any] = field(default_factory=dict)
    llm_calls: int = 0          # diagnostic: LLM inference call count
    design_path: str = ""       # TUNE_PARAMS | MODIFY_LAW | REUSE_HISTORY
    history: List[Dict] = field(default_factory=list)
    error: str = ""

    @property
    def time_per_call_min(self) -> float:
        total = self.mc_calls_dev + self.mc_calls_test
        if total <= 0:
            return float("nan")
        return (self.wall_clock_sec / 60.0) / total

    def to_dict(self) -> Dict[str, Any]:
        import dataclasses
        d = dataclasses.asdict(self)
        d["time_per_call_min"] = self.time_per_call_min
        return d


# ── LLM call counter ───────────────────────────────────────────────────────────

class _LLMCallCounter:
    """Wraps HermesIntegration.run_with_tools to count LLM inference calls."""

    def __init__(self, hermes) -> None:
        self._hermes = hermes
        self.count = 0
        self._orig = hermes.run_with_tools

    def install(self) -> None:
        counter = self

        async def _counted(*args, **kwargs):
            counter.count += 1
            return await counter._orig(*args, **kwargs)

        self._hermes.run_with_tools = _counted

    def uninstall(self) -> None:
        self._hermes.run_with_tools = self._orig


# ── Main agent class ───────────────────────────────────────────────────────────

class HermesAgentSingle:
    """Single-agent Hermes with full GSMultiAgent tool suite, no cascade.

    Parameters
    ----------
    prompt_text : str
        Task description (same as prompt.txt for GSMultiAgent).
    max_iterations : int
        Outer loop limit (matches workflow.max_iterations).
    mc_data_path : str, optional
        Path to external MC .mat data directory for eval; empty → MATLAB live sim.
    optimizer_backend : str, optional
        Optimizer backend name; None → reads from config.yaml.
    """

    def __init__(
        self,
        prompt_text: str,
        max_iterations: Optional[int] = None,
        mc_data_path: str = "",
        optimizer_backend: Optional[str] = None,
    ) -> None:
        self.prompt_text = prompt_text
        self.mc_data_path = mc_data_path
        self._optimizer_backend = optimizer_backend
        self._max_iterations = max_iterations
        self._llm_calls = 0

    # ── public API ─────────────────────────────────────────────────────────────

    async def run_async(self) -> HermesAgentSingleResult:
        """Execute the single-agent workflow and return structured results."""
        started = time.perf_counter()
        result = HermesAgentSingleResult(
            prompt_text=self.prompt_text,
        )

        try:
            result = await self._execute()
        except (KeyboardInterrupt, asyncio.CancelledError):
            raise
        except Exception as exc:
            import traceback
            result.error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
            logger.error("[HermesAgentSingle] Unhandled error: %s", exc, exc_info=True)

        result.wall_clock_sec = time.perf_counter() - started
        result.llm_calls = self._llm_calls
        return result

    def run(self) -> HermesAgentSingleResult:
        """Synchronous wrapper for run_async()."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(1) as pool:
                    return pool.submit(asyncio.run, self.run_async()).result()
            return loop.run_until_complete(self.run_async())
        except RuntimeError:
            return asyncio.run(self.run_async())

    # ── internal workflow ──────────────────────────────────────────────────────

    async def _execute(self) -> HermesAgentSingleResult:
        from multi_agent.config_loader import get_config
        from multi_agent import HermesIntegration, HERMES_AVAILABLE

        cfg = get_config()
        max_iter = self._max_iterations or cfg.workflow.max_iterations

        result = HermesAgentSingleResult(
            prompt_text=self.prompt_text,
            iterations_budget=max_iter,
        )

        if not HERMES_AVAILABLE:
            result.error = "Hermes Agent not available (run_agent not importable)"
            return result

        # ── Patch config: disable cascade mechanisms ────────────────────────
        # Single-agent mode: no sub-agents, no reflection, no memory injection
        cfg.ablation.reflection_agent = False
        cfg.ablation.memory_search = False
        if self.mc_data_path:
            cfg.simulation.mc_data_path = self.mc_data_path
            os.environ["GSM_MC_DATA_PATH"] = self.mc_data_path
        if self._optimizer_backend:
            cfg.matlab_rl_optimizer.optimizer_backend = self._optimizer_backend

        # ── Initialize components ───────────────────────────────────────────
        from multi_agent.memory.parameter_experience import ParameterExperience
        from multi_agent.memory.rag_knowledge_base import RAGKnowledgeBase
        from multi_agent.memory.hermes_agent_memory import HermesAgentMemory
        from multi_agent.integration.reflection_agent import ReflectionAgent
        from multi_agent.rl.matlab_rl_optimizer import MatlabRLOptimizer
        from multi_agent.simulation.optimization_workflow import OptimizationWorkflow
        from multi_agent.simulation.guidance_simulator import GuidanceSimulator
        from multi_agent.tools.simulation_tool import RunSimulationTool

        try:
            rag_kb = RAGKnowledgeBase(
                persist_dir=cfg.rag.persist_dir,
                collection_name=cfg.rag.collection,
                embedding_provider=cfg.rag.embedding_provider,
                embedding_model=cfg.rag.embedding_model,
                embedding_dim=cfg.rag.embedding_dim,
                embedding_api_key=cfg.rag.embedding_api_key,
                embedding_base_url=cfg.rag.embedding_base_url,
            )
        except Exception as exc:
            logger.warning("[HermesAgentSingle] RAG init failed: %s", exc)
            rag_kb = None

        try:
            parameter_experience = ParameterExperience(
                max_short_term_size=cfg.parameter_experience.max_short_term,
                max_long_term_size=cfg.parameter_experience.max_long_term,
            )
        except Exception as exc:
            logger.warning("[HermesAgentSingle] PE init failed: %s", exc)
            parameter_experience = None

        try:
            simulator = GuidanceSimulator()
        except Exception:
            simulator = None

        rl_optimizer = MatlabRLOptimizer(
            simulator=simulator,
            parameter_experience=parameter_experience,
        )

        # Reflection agent disabled (single-agent ablation)
        reflection_agent = None

        optim_workflow = OptimizationWorkflow(
            rl_optimizer=rl_optimizer,
            reflection_agent=reflection_agent,
            parameter_experience=parameter_experience,
            optimizer_backend=self._optimizer_backend,
        )

        hermes_memory = HermesAgentMemory()

        # ── Initialize HermesIntegration (single agent, no sub-agents) ─────
        hermes = HermesIntegration()
        ok = await hermes.initialize_with_tools(
            rag_kb=rag_kb,
            parameter_experience=parameter_experience,
            simulator=simulator,
            optimizer=rl_optimizer,
            orchestrator=optim_workflow,
            reflection_agent=reflection_agent,
            user_prompt=self.prompt_text,
            ablation=cfg.ablation,
            hermes_agent_memory=hermes_memory,
            include_delegate=False,    # ← KEY: no sub-agent cascade
        )
        if not ok:
            result.error = "HermesIntegration.initialize_with_tools failed"
            return result

        # Install LLM call counter
        counter = _LLMCallCounter(hermes)
        counter.install()

        # ── Run the Hermes outer loop (mirrors cli_agent.py iterations) ─────
        system_prompt = self._build_system_prompt(cfg, hermes_memory)
        design_path = "UNKNOWN"
        prev_suggestion = ""
        history: List[Dict] = []
        iteration = 0
        first_success: Optional[int] = None

        for iteration in range(1, max_iter + 1):
            logger.info("[HermesAgentSingle] Iteration %d/%d", iteration, max_iter)

            user_msg = self.prompt_text
            if prev_suggestion:
                user_msg = (
                    f"{self.prompt_text}\n\n"
                    f"【上轮反思建议】\n{prev_suggestion}"
                )

            response = await hermes.run_with_tools(
                user_message=user_msg,
                verbose_thinking=False,
            )

            # Parse design path from Hermes response
            dp = self._extract_design_path(response or "")
            if dp:
                design_path = dp

            # Extract best metrics from ParameterExperience or simulation result
            best_metrics = {}
            best_params = {}
            if parameter_experience:
                try:
                    best_entry = None
                    entries = await asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: parameter_experience.retrieve(
                            task_context={"task": self.prompt_text},
                            top_k=1, memory_type=None,
                        ),
                    )
                    if entries:
                        best_entry = entries[0]
                        best_params = getattr(best_entry, "parameters", {}) or {}
                        best_metrics = getattr(best_entry, "objectives", {}) or {}
                except Exception as exc:
                    logger.debug("[HermesAgentSingle] PE query failed: %s", exc)

            all_pass = self._check_all_pass(best_metrics)
            if all_pass and first_success is None:
                first_success = iteration

            history.append({
                "iteration": iteration,
                "design_path": design_path,
                "all_pass": all_pass,
                "best_metrics": dict(best_metrics),
                "best_params": dict(best_params),
            })

            # Check if done
            if all_pass:
                logger.info(
                    "[HermesAgentSingle] All constraints met at iter %d", iteration
                )
                break

            # Extract suggestion for next iteration
            prev_suggestion = self._extract_suggestion(response or "")

        counter.uninstall()
        self._llm_calls = counter.count

        # ── Final state ─────────────────────────────────────────────────────
        last_h = history[-1] if history else {}
        best_params = last_h.get("best_params", {})
        best_metrics = last_h.get("best_metrics", {})
        dev_all_pass = last_h.get("all_pass", False)

        result.iterations_used = iteration
        result.first_success_iter = first_success
        result.first_success = first_success is not None
        result.dev_all_pass = dev_all_pass
        result.best_params = best_params
        result.best_dev = {"metrics": best_metrics, "all_pass": dev_all_pass}
        result.design_path = design_path
        result.history = history
        result.llm_calls = self._llm_calls

        # MC call accounting from rl_optimizer
        result.mc_calls_dev = int(
            getattr(rl_optimizer, "_external_mc_calls", 0)
            or getattr(rl_optimizer, "_episode_count", 0)
            or 0
        )
        result.sim_time_sec = float(
            getattr(rl_optimizer, "_external_mc_sim_time_sec", 0.0) or 0.0
        )

        if self._should_run_test_eval(best_metrics):
            result.test_all_pass = dev_all_pass  # conservative: same params
            result.test_eval = {"metrics": best_metrics, "all_pass": dev_all_pass}

        if dev_all_pass:
            result.best_score = -(
                best_metrics.get("hit_rate", 0.0) / 100.0 + 0.01
            )

        return result

    # ── helpers ────────────────────────────────────────────────────────────────

    def _build_system_prompt(self, cfg, hermes_memory) -> str:
        mem_block = hermes_memory.get_context_block() if hermes_memory else ""
        return (
            "你是一个制导系统设计专家。请根据用户任务描述完成制导律设计与参数优化。\n"
            "你有完整的工具套件：generate_matlab, syntax_check_matlab, run_simulation, "
            "rag_retrieve, parameter_experience_search, judge_requirements 等。\n"
            "独立完成从方案制定、代码生成、仿真验证到参数优化的全流程。\n"
            + mem_block
        )

    @staticmethod
    def _extract_design_path(response: str) -> str:
        import re
        for pattern, dp in [
            (r"MODIFY_LAW", "MODIFY_LAW"),
            (r"TUNE_PARAMS", "TUNE_PARAMS"),
            (r"REUSE_HISTORY", "REUSE_HISTORY"),
        ]:
            if re.search(pattern, response, re.IGNORECASE):
                return dp
        return ""

    @staticmethod
    def _extract_suggestion(response: str) -> str:
        import re
        m = re.search(r"suggestion[\"']?\s*[:：]\s*[\"']?([^\"'\n]{10,200})", response, re.IGNORECASE)
        if m:
            return m.group(1).strip()
        return ""

    @staticmethod
    def _check_all_pass(metrics: Dict[str, float]) -> bool:
        if not metrics:
            return False
        try:
            from multi_agent.tools.simulation_tool import check_constraints
            ok, _ = check_constraints(
                metrics.get("hit_rate", 0.0),
                metrics.get("SEP", metrics.get("miss_distance", 999.0)),
                metrics.get("peak_ny", metrics.get("peak_nz", 999.0)),
                metrics.get("pitch_PM", metrics.get("PM", 0.0)),
                metrics.get("pitch_BW", metrics.get("BW", 0.0)),
            )
            return ok
        except Exception:
            return False

    @staticmethod
    def _should_run_test_eval(metrics: Dict) -> bool:
        return bool(metrics)


# ── Convenience runner ─────────────────────────────────────────────────────────

def run_hermes_agent_single(
    prompt_text: str,
    *,
    max_iterations: Optional[int] = None,
    mc_data_path: str = "",
    optimizer_backend: Optional[str] = None,
) -> HermesAgentSingleResult:
    """Synchronous entry point for experiment_plan_runner.py."""
    agent = HermesAgentSingle(
        prompt_text=prompt_text,
        max_iterations=max_iterations,
        mc_data_path=mc_data_path,
        optimizer_backend=optimizer_backend,
    )
    return agent.run()
