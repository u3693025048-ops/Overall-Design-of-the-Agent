#!/usr/bin/env python3
"""
Subagent Integration for Hermes Agent
支持并行任务执行和结果聚合
"""

import asyncio
import logging
import uuid
from typing import Any, Callable, Dict, List, Optional
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


@dataclass
class SubagentConfig:
    """子Agent配置"""

    name: str = "subagent"
    max_iterations: int = 50
    tools: List[str] = field(default_factory=list)
    system_prompt: str = ""
    model: Optional[str] = None


class SubagentResult:
    """子Agent执行结果"""

    def __init__(
        self,
        agent_id: str,
        status: str,
        result: Any = None,
        error: Optional[str] = None,
    ):
        self.agent_id = agent_id
        self.status = status
        self.result = result
        self.error = error


class SubagentManager:
    """子Agent管理器"""

    def __init__(self, hermes_integration: Any):
        self.hermes = hermes_integration
        self._active_children: Dict[str, Any] = {}
        self._results: Dict[str, SubagentResult] = {}
        self._executor = ThreadPoolExecutor(max_workers=10)

    async def delegate(
        self,
        task: str,
        config: Optional[SubagentConfig] = None,
        parent_tools: Optional[List[Any]] = None,
    ) -> SubagentResult:
        """派生子Agent执行任务"""
        config = config or SubagentConfig(name="subagent")
        agent_id = str(uuid.uuid4())[:8]

        logger.info(f"[Subagent {agent_id}] Delegating task: {task[:50]}...")

        try:
            if not self.hermes._initialized:
                await self.hermes.initialize()

            subagent = self.hermes.agent.__class__(
                base_url=self.hermes.base_url,
                api_key=self.hermes.api_key,
                provider=self.hermes.provider,
                model=config.model or self.hermes.model,
                max_iterations=config.max_iterations,
                tools=parent_tools or self.hermes.agent.tools,
            )

            self._active_children[agent_id] = subagent

            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self._executor,
                self._run_sync,
                subagent,
                task,
            )

            self._results[agent_id] = SubagentResult(
                agent_id=agent_id,
                status="completed",
                result=result,
            )

            logger.info(f"[Subagent {agent_id}] Completed")
            return self._results[agent_id]

        except Exception as e:
            logger.error(f"[Subagent {agent_id}] Failed: {e}")
            self._results[agent_id] = SubagentResult(
                agent_id=agent_id,
                status="failed",
                error=str(e),
            )
            return self._results[agent_id]
        finally:
            self._active_children.pop(agent_id, None)

    def _run_sync(self, agent: Any, task: str) -> str:
        """同步运行Agent（在线程池中执行）"""
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(agent.run_conversation(task))
            loop.close()
            return result
        except Exception as e:
            return f"Error: {e}"

    async def delegate_parallel(
        self,
        tasks: List[str],
        configs: Optional[List[SubagentConfig]] = None,
        parent_tools: Optional[List[Any]] = None,
    ) -> List[SubagentResult]:
        """并行派发多个子Agent"""
        configs = configs or [SubagentConfig() for _ in tasks]

        logger.info(f"[SubagentManager] Parallel delegation of {len(tasks)} tasks")

        futures = []
        for task, config in zip(tasks, configs):
            future = await self.delegate(task, config, parent_tools)
            futures.append(future)

        results = await asyncio.gather(*futures, return_exceptions=True)

        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append(
                    SubagentResult(
                        agent_id=f"error_{i}",
                        status="failed",
                        error=str(result),
                    )
                )
            else:
                processed_results.append(result)

        return processed_results

    async def aggregate_results(
        self,
        results: List[SubagentResult],
        strategy: str = "best",
    ) -> Any:
        """聚合多个子Agent的结果"""
        if not results:
            return None

        successful = [r for r in results if r.status == "completed" and r.result]

        if not successful:
            return {"status": "failed", "error": "All subagents failed"}

        if strategy == "best":
            best = max(successful, key=lambda r: len(r.result) if r.result else 0)
            return best.result

        elif strategy == "all":
            return {
                "status": "success",
                "count": len(successful),
                "results": [r.result for r in successful],
            }

        elif strategy == "first":
            return successful[0].result

        return successful[0].result

    def interrupt_all(self):
        """中断所有子Agent"""
        for agent_id, agent in self._active_children.items():
            logger.info(f"[Subagent {agent_id}] Interrupting...")
            if hasattr(agent, "_interrupt_requested"):
                agent._interrupt_requested = True

    def get_active_count(self) -> int:
        """获取活跃子Agent数量"""
        return len(self._active_children)

    def get_results_count(self) -> int:
        """获取已完成的子Agent数量"""
        return len(self._results)

