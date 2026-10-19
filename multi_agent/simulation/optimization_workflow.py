#!/usr/bin/env python3
"""
OptimizationWorkflow — Layer 3 async workflow.

Entry point for the RL-based optimization pipeline.
Called by cli_agent.py when Hermes (Layer 2) has generated a script
and initial simulation results do NOT satisfy requirements.

Flow:
  1. RL Optimizer runs PPO optimization on the script.
     - After each episode that produces a new best → JudgmentAgent.judge()
     - Early exit if JudgmentAgent.satisfied == True
  2. ReflectionAgent provides final deep analysis.
  3a. If satisfied:
       - Write back params + model to ParameterExperience
       - Return status="done" + report data
  3b. If not satisfied:
       - Return status="needs_iteration" + suggestion + next_action
         (caller feeds suggestion back to Hermes as prev_suggestion)
"""

import asyncio
import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from multi_agent.integration.design_path_policy import metrics_summary_line
from multi_agent.logging.log_verbosity import (
    cprint,
    is_verbose,
    mprint,
    should_log_expert_round,
)
from multi_agent.metric_contract import BW_MAX, BW_MIN, HIT_RATE_MIN, PM_MAX, PM_MIN, SEP_MAX

logger = logging.getLogger(__name__)


def _expert_feasibility_score(m: Dict[str, float]) -> float:
    _hr = m.get("hit_rate", 0.0)
    _pm = m.get("pitch_PM", 0.0)
    _bw = m.get("pitch_BW", 0.0)
    _sep = m.get("SEP", m.get("miss_distance", 99.0))
    score = _hr / 100.0 * 3.0
    pm_mid = 0.5 * (PM_MIN + PM_MAX)
    pm_half = max(0.5 * (PM_MAX - PM_MIN), 1.0)
    if PM_MIN <= _pm <= PM_MAX:
        score += 2.0
    elif _pm < 0:
        score -= 5.0
    else:
        score -= min(2.0, abs(_pm - pm_mid) / pm_half)
    if BW_MIN <= _bw <= BW_MAX:
        score += 1.0
    if _sep < SEP_MAX:
        score += 1.0
    return score


class OptimizationWorkflow:
    """
    Async Layer 3 workflow that integrates:
      - MatlabRLOptimizer (RL optimization)
      - JudgmentAgent     (per-episode evaluation)
      - ReflectionAgent   (final analysis)
      - ParameterExperience (experience writeback)

    Section 4.2.1: the optimizer backend is pluggable via
    ``matlab_rl_optimizer.optimizer_backend`` in config.yaml.
    Supported values: hybrid_ppo (default) | direct_ppo | cma_es | random_search
    """

    def __init__(
        self,
        rl_optimizer=None,
        judgment_agent=None,
        reflection_agent=None,
        parameter_experience=None,
        output_dir: str = "./guidance_output",
        optimizer_backend: Optional[str] = None,
        kb_manager=None,
        emit_next_action_json: bool = True,
        run_id: Optional[str] = None,
        runs_root: Optional[str] = None,
    ):
        self.rl_optimizer = rl_optimizer
        self.judgment_agent = judgment_agent
        self.reflection_agent = reflection_agent
        self.parameter_experience = parameter_experience
        self.output_dir = output_dir
        # Explicit override; falls back to config if None
        self._optimizer_backend_name: Optional[str] = optimizer_backend
        self._backend = None   # lazy-loaded
        # Stage C4：可选接入四库 RAG，用于增强 next_action 决策并产出
        # 可机检的 next_action.json。为 None 时行为完全不变（纯策略决策）。
        self.kb_manager = kb_manager
        self.emit_next_action_json = emit_next_action_json
        # P0-4：C4 产物归跑次包（07_runs/{run_id}/）。run_id 缺省时在首次 C4 由任务
        # 生成并固化（见 run() 的 Stage C4 块）；runs_root 缺省为 deliverables/07_runs。
        self.run_id = run_id
        self.runs_root = runs_root

    def _get_optimizer_backend(self):
        """Lazy-load the optimizer backend from config or explicit override."""
        if self._backend is not None:
            return self._backend
        try:
            from multi_agent.rl.optimizer_registry import build_optimizer_backend
            name = self._optimizer_backend_name
            if not name:
                from multi_agent.config_loader import get_config
                name = getattr(
                    get_config().matlab_rl_optimizer,
                    "optimizer_backend", "hybrid_ppo",
                ) or "hybrid_ppo"
            self._backend = build_optimizer_backend(name)
            logger.info("[OptimWorkflow] Optimizer backend: %s", name)
        except Exception as exc:
            logger.warning(
                "[OptimWorkflow] Could not load optimizer backend (%s); "
                "falling back to hybrid_ppo", exc
            )
            from multi_agent.rl.optimizer_registry import HybridPPOBackend
            self._backend = HybridPPOBackend()
        return self._backend

    async def _run_optimizer_backend(self, **kwargs) -> dict:
        """Dispatch to the configured optimizer backend."""
        backend = self._get_optimizer_backend()
        return await backend.optimize(rl_optimizer=self.rl_optimizer, **kwargs)

    async def run(
        self,
        script_path: str,
        task_prompt: str,
        mission_conditions: Optional[Dict[str, List[int]]] = None,
        initial_metrics: Optional[Dict[str, float]] = None,
        directly_satisfied: bool = False,
        max_episodes: Optional[int] = None,
        nmc: Optional[int] = None,
        initial_auto_params: Optional[Dict[str, float]] = None,
        optimization_history: Optional[List[Dict[str, Any]]] = None,
        resume_checkpoint: Optional[str] = None,
        bw_finish_only: bool = False,
        warm_start_source: Optional[str] = None,
        level: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run the Layer 3 optimization workflow.

        Parameters
        ----------
        script_path : str
            Path to the generated .m script (monte_carlo_single style).
        task_prompt : str
            User's full task description (used by judgment + reflection).
        mission_conditions : dict, optional
            Parsed T/G/AP/R conditions dict.
        initial_metrics : dict, optional
            Metrics from Hermes Layer 2 run_simulation call.
        directly_satisfied : bool
            If True (judge_requirements already returned satisfied),
            skip RL and go directly to Reflection.
        max_episodes, nmc : optional overrides.

        Returns
        -------
        dict with keys:
            status         : "done" | "needs_iteration"
            best_params    : dict (tunable params)
            best_metrics   : dict (simulation metrics)
            suggestion     : str (only when needs_iteration)
            next_action    : "tune_params" | "modify_law" (only when needs_iteration)
            report_path    : str (only when done)
        """
        _t0 = datetime.now()
        from multi_agent.simulation.time_budget import get_run_budget

        _budget = get_run_budget()
        if _budget and _budget.expired():
            logger.warning("[OptimWorkflow] Wall-clock budget expired — returning best effort")
            return {
                "status": "needs_iteration",
                "best_params": dict(initial_auto_params or {}),
                "best_metrics": dict(initial_metrics or {}),
                "suggestion": "总运行时限已到，请使用当前最优结果或 golden 包",
                "next_action": "tune_params",
                "report_path": "",
            }

        # P2-1 户部：成本预算与墙钟预算**并列**在入口咨询（形态一致，不另造调度点）。
        # 未设限额 ⇒ should_stop() 恒 False ⇒ 与接入前逐字等价。
        from multi_agent import cost_meter as _costw

        if _costw.should_stop(reserve_ratio=0.10):
            _cw = _costw.budget_status() or {}
            logger.warning(
                "[OptimWorkflow] Cost budget — returning best effort (tokens=%s/%s)",
                _cw.get("used_tokens"),
                _cw.get("limit_tokens"),
            )
            return {
                "status": "needs_iteration",
                "best_params": dict(initial_auto_params or {}),
                "best_metrics": dict(initial_metrics or {}),
                "suggestion": "运行成本预算已到，请使用当前最优结果或 golden 包",
                "next_action": "tune_params",
                "report_path": "",
            }

        # P2-6 工部：运行指标在入口记一笔（与墙钟 / 成本预算**并列**在同一运行
        # 边界，不另造调度点）。未启用 ⇒ 全 no-op，等价接入前。
        from multi_agent import metrics as _metw

        _metw.counter("workflow_runs")
        _metw.gauge("workflow_max_episodes", int(max_episodes or 0))

        logger.info(
            f"[OptimWorkflow] Starting | directly_satisfied={directly_satisfied} | "
            f"script={os.path.basename(script_path)}"
        )
        if resume_checkpoint:
            logger.info(f"[OptimWorkflow] Resume checkpoint: {resume_checkpoint}")
            print(f"  [Layer 3] Resuming PPO from checkpoint: {resume_checkpoint}")

        if not directly_satisfied and script_path and os.path.isfile(script_path):
            from multi_agent.rl.matlab_rl_optimizer import validate_rl_script_content
            try:
                with open(script_path, "r", encoding="utf-8", errors="ignore") as _fh:
                    _script_body = _fh.read()
                _rl_ok, _rl_why = validate_rl_script_content(_script_body)
                if not _rl_ok:
                    logger.error("[OptimWorkflow] RL blocked: %s", _rl_why)
                    print(f"  [Layer 3] RL blocked — {_rl_why}")
                    return {
                        "status": "needs_iteration",
                        "best_params": dict(initial_auto_params or {}),
                        "best_metrics": dict(initial_metrics or {}),
                        "suggestion": (
                            f"MATLAB 脚本无法用于 RL 仿真: {_rl_why}。"
                            "请以 knowledge_base/matlab/guidance/monte_carlo_single.m "
                            "为模板重新 generate_matlab，保留完整 sim_s(p) 与 "
                            "RL_PARAMS_BEGIN/END 块。"
                        ),
                        "next_action": "modify_law",
                        "report_path": "",
                    }
            except OSError as _read_exc:
                logger.warning(
                    "[OptimWorkflow] Could not read script for RL validation: %s",
                    _read_exc,
                )

        rl_result: Dict[str, Any] = {}
        best_params: Dict[str, float] = {}
        best_metrics: Dict[str, float] = initial_metrics or {}

        if bw_finish_only:
            return await self._run_bw_finish_only(
                script_path=script_path,
                task_prompt=task_prompt,
                mission_conditions=mission_conditions,
                center_params=initial_auto_params or {},
                center_metrics=best_metrics,
                optimization_history=optimization_history,
                _t0=_t0,
            )

        from multi_agent.integration.t4_low_risk import (
            layer3_max_episodes_cap,
            post_modify_expert_max_rounds,
            resolve_initial_auto_params,
            should_allow_ppo_after_expert,
            should_block_tune_after_bad_modify,
        )

        _mc_str = ""
        if mission_conditions:
            try:
                from multi_agent.rl.matlab_rl_optimizer import build_matlab_conditions_str
                _mc_str = build_matlab_conditions_str(mission_conditions)
            except Exception:
                pass
        initial_auto_params = resolve_initial_auto_params(
            initial_auto_params,
            task_prompt=task_prompt,
            mission_conditions=_mc_str,
            optimization_history=optimization_history,
        )
        if max_episodes is not None:
            max_episodes = layer3_max_episodes_cap(
                optimization_history,
                int(max_episodes),
                metrics=best_metrics,
                task_prompt=task_prompt,
            )

        _blocked, _block_msg = should_block_tune_after_bad_modify(
            best_metrics, task_prompt, optimization_history,
        )
        if _blocked and not directly_satisfied:
            logger.warning("[OptimWorkflow] T4 low-risk block tune: %s", _block_msg)
            print(f"  [Layer 3] T4 低风险：禁止 TUNE — {_block_msg}")
            return {
                "status": "needs_iteration",
                "best_params": dict(initial_auto_params or {}),
                "best_metrics": dict(best_metrics or {}),
                "suggestion": _block_msg,
                "next_action": "modify_law",
                "report_path": "",
            }

        # ── Path A: directly satisfied from Layer 2 judge_requirements ───────
        if directly_satisfied and initial_metrics:
            from multi_agent.tools.judge_requirements_tool import (
                _resolve_requirements,
                _rule_check,
            )
            from multi_agent.integration.judgment_agent import all_requirements_satisfied

            _reqs = _resolve_requirements("", task_prompt)
            if _reqs:
                _ok, _reasons = _rule_check(initial_metrics, _reqs)
                if not _ok or not all_requirements_satisfied(initial_metrics, _reqs, task_prompt):
                    logger.warning(
                        "[OptimWorkflow] directly_satisfied overridden — metrics fail rule-check: %s",
                        "; ".join(_reasons[:5]),
                    )
                    directly_satisfied = False

        if directly_satisfied and not initial_metrics:
            logger.warning(
                "[OptimWorkflow] directly_satisfied ignored because metrics are empty"
            )
            directly_satisfied = False

        if directly_satisfied and initial_metrics:
            logger.info("[OptimWorkflow] Skipping RL (directly satisfied by Layer 2)")
            logger.info(f"[OptimWorkflow] Using Layer 2 metrics: {initial_metrics}")
            best_metrics = initial_metrics
            best_metrics, _final_ok, _final_note = await self._maybe_validate_final_mc_seeds(
                script_path=script_path,
                params=initial_auto_params or {},
                metrics=best_metrics,
                mission_conditions=mission_conditions,
                task_prompt=task_prompt,
                nmc=nmc,
            )
            if not _final_ok:
                directly_satisfied = False
                initial_metrics = best_metrics
                if _final_note:
                    logger.warning("[OptimWorkflow] %s", _final_note)
                    print(f"  [Layer 3] {_final_note}")
            
            # 直接返回，使用 Layer 2 的仿真结果
            logger.info("[OptimWorkflow] Returning with Layer 2 results (no re-simulation)")
            if _final_ok:
                logger.info("[OptimWorkflow] Returning with Layer 2 results")
                return {
                    "status": "done",
                    "best_params": initial_auto_params or {},
                    "best_metrics": best_metrics,
                    "report_path": "",
                }

        if not directly_satisfied:
            from multi_agent.integration.design_path_policy import (
                should_skip_layer3_param_search,
            )

            _skip_param_search = should_skip_layer3_param_search(
                best_metrics, task_prompt, optimization_history,
            )
            if _skip_param_search:
                print(
                    "  [Layer 3] Layer2 4/5 (only PeakNy mean over limit) — "
                    "skip Expert/PPO/CLS → modify_law"
                )
                logger.info(
                    "[OptimWorkflow] Skipping Expert/PPO: non-peak constraints met, "
                    "PeakNy mean over limit"
                )
            else:
                # ── Strategy Selection ─────────────────────────────────────────
                strategy = await self._select_strategy(
                    best_metrics, task_prompt, optimization_history,
                )
                if strategy == "skip":
                    print(
                        "  [Layer 3] Strategy: SKIP param search "
                        "(only PeakNy mean over limit)"
                    )
                else:
                    print(f"  [Layer 3] Strategy selected: {strategy.upper()}")
                    logger.info(f"[OptimWorkflow] Optimization strategy: {strategy}")

                if strategy != "skip":
                    if strategy == "expert":
                        if self.judgment_agent is not None:
                            self.judgment_agent.update_task_prompt(task_prompt)
                        _expert_nmc = nmc or 25
                        try:
                            from multi_agent.config_loader import get_config as _gc_expert
                            _rlc = _gc_expert().matlab_rl_optimizer
                            _fast = int(getattr(_rlc, "expert_nmc", 0) or 0)
                            if _fast > 0:
                                _expert_nmc = _fast
                        except Exception:
                            pass
                        _expert_rounds = post_modify_expert_max_rounds(
                            optimization_history, default=3,
                        )
                        if _budget:
                            _expert_rounds = _budget.allowed_expert_rounds(_expert_rounds)
                        expert_result = await self._run_expert_tuning(
                            script_path=script_path,
                            task_prompt=task_prompt,
                            mission_conditions=mission_conditions,
                            initial_metrics=best_metrics,
                            max_rounds=_expert_rounds,
                            nmc=_expert_nmc,
                            initial_params=initial_auto_params,
                            optimization_history=optimization_history,
                        )
                        best_params  = expert_result.get("best_params", best_params)
                        best_metrics = expert_result.get("best_metrics", best_metrics)
                        rl_result    = expert_result

                        print(
                            f"  [Layer 3/Expert] Done {expert_result.get('total_episodes',0)} rounds | "
                            f"status={expert_result.get('status')} | "
                            f"hit={best_metrics.get('hit_rate',0):.1f}% "
                            f"SEP={best_metrics.get('SEP',99):.2f}m"
                        )

                        if (
                            expert_result.get("status") != "task_requirements_met"
                            and self.rl_optimizer is not None
                        ):
                            _ppo_ok, _ppo_msg = should_allow_ppo_after_expert(
                                best_metrics, task_prompt, optimization_history,
                            )
                            if not _ppo_ok:
                                print(f"  [Layer 3] T4 低风险：{_ppo_msg}")
                                print("  [Layer 3] → 跳过 PPO，直接进入 CLS")
                                logger.info("[OptimWorkflow] %s", _ppo_msg)
                            else:
                                _ppo_max = int(max_episodes or 50)
                                if _budget:
                                    _ppo_max = _budget.allowed_ppo_episodes(_ppo_max)
                                if _ppo_max <= 0:
                                    print(
                                        "  [Layer 3] 剩余时限不足，跳过 PPO，直接进入 CLS"
                                    )
                                    logger.info(
                                        "[OptimWorkflow] PPO skipped — wall-clock budget"
                                    )
                                else:
                                    print("  [Layer 3] Expert tuning insufficient → falling back to PPO")
                                    logger.info("[OptimWorkflow] Expert insufficient → PPO fallback")
                                    if self.judgment_agent is not None:
                                        self.rl_optimizer.reflection_agent = self.judgment_agent
                                    try:
                                        ppo_result = await self._run_optimizer_backend(
                                            script_path=script_path,
                                            mission_conditions=mission_conditions,
                                            max_episodes=_ppo_max,
                                            nmc=nmc,
                                            task_prompt=task_prompt,
                                            reflection_agent=self.judgment_agent,
                                            initial_auto_params=best_params or initial_auto_params,
                                            warm_start_source="expert",
                                            optimization_history=optimization_history,
                                            resume_checkpoint=resume_checkpoint,
                                        )
                                        _ppo_peak = float(
                                            (ppo_result.get("best_metrics") or {}).get(
                                                "peak_ny_max",
                                                (ppo_result.get("best_metrics") or {}).get("peak_ny", 0),
                                            )
                                            or 0
                                        )
                                        _cur_peak = float(
                                            best_metrics.get(
                                                "peak_ny_max", best_metrics.get("peak_ny", 0),
                                            )
                                            or 0
                                        )
                                        if (
                                            ppo_result.get("best_metrics", {}).get("hit_rate", 0)
                                            >= best_metrics.get("hit_rate", 0)
                                            and (
                                                _ppo_peak <= 0
                                                or _cur_peak <= 0
                                                or _ppo_peak <= _cur_peak
                                                or ppo_result.get("best_metrics", {}).get("hit_rate", 0)
                                                > best_metrics.get("hit_rate", 0)
                                            )
                                        ):
                                            best_params = ppo_result.get("best_params", best_params)
                                            best_metrics = ppo_result.get("best_metrics", best_metrics)
                                            rl_result = ppo_result
                                        logger.info(
                                            f"[OptimWorkflow] PPO fallback done: status={ppo_result.get('status')} "
                                            f"episodes={ppo_result.get('total_episodes', 0)}"
                                        )
                                    except (KeyboardInterrupt, asyncio.CancelledError):
                                        logger.warning("[OptimWorkflow] PPO fallback interrupted — using expert results")
                                        print("  [Layer 3] PPO interrupted — using expert results")
                                    except Exception as exc:
                                        logger.error(f"[OptimWorkflow] PPO fallback failed: {exc}")

                    else:
                        if self.rl_optimizer is None:
                            logger.warning("[OptimWorkflow] No RL optimizer; using initial_metrics")
                        else:
                            if self.judgment_agent is not None:
                                self.judgment_agent.update_task_prompt(task_prompt)
                                self.rl_optimizer.reflection_agent = self.judgment_agent
                            try:
                                rl_result = await self._run_optimizer_backend(
                                    script_path=script_path,
                                    mission_conditions=mission_conditions,
                                    max_episodes=max_episodes,
                                    nmc=nmc,
                                    task_prompt=task_prompt,
                                    reflection_agent=self.judgment_agent,
                                    initial_auto_params=initial_auto_params,
                                    warm_start_source=warm_start_source or "hermes",
                                    optimization_history=optimization_history,
                                    resume_checkpoint=resume_checkpoint,
                                )
                                best_params  = rl_result.get("best_params", {})
                                best_metrics = rl_result.get("best_metrics", best_metrics)
                                logger.info(
                                    f"[OptimWorkflow] PPO done: status={rl_result.get('status')} "
                                    f"episodes={rl_result.get('total_episodes', 0)} "
                                    f"best_reward={rl_result.get('best_reward', 0):.3f}"
                                )
                            except (KeyboardInterrupt, asyncio.CancelledError):
                                logger.warning("[OptimWorkflow] PPO interrupted — using partial results")
                                print("  [Layer 3] PPO interrupted — using partial results")
                                best_params  = self.rl_optimizer._best_params or {}
                                best_metrics = self.rl_optimizer._best_metrics or best_metrics
                            except Exception as exc:
                                logger.error(f"[OptimWorkflow] PPO optimizer failed: {exc}")
                                best_metrics = initial_metrics or {}

        _rl_total_episodes = rl_result.get("total_episodes", 0) if rl_result else 0
        _rl_best_reward    = rl_result.get("best_reward",    0.0) if rl_result else 0.0
        _rl_baseline_reward = rl_result.get("baseline_reward", 0.0) if rl_result else 0.0

        # ── Constraint Local Search (CLS) after PPO ───────────────────────────
        if rl_result and self.rl_optimizer is not None:
            cls_out = await self._maybe_run_constraint_local_search(
                script_path=script_path,
                task_prompt=task_prompt,
                mission_conditions=mission_conditions,
                center_params=best_params or initial_auto_params,
                rl_result=rl_result,
            )
            if cls_out:
                from multi_agent.integration.design_path_policy import requirements_from_prompt
                from multi_agent.rl.metric_constraints import should_adopt_cls_result

                _reqs = requirements_from_prompt(task_prompt)
                _prior_fit = float(
                    rl_result.get("best_fitness")
                    or self.rl_optimizer._compute_pe_fitness(best_metrics or {})
                )
                _cls_fit = float(
                    cls_out.get("best_fitness")
                    or self.rl_optimizer._compute_pe_fitness(
                        cls_out.get("best_metrics") or {}
                    )
                )
                if should_adopt_cls_result(
                    best_metrics,
                    _prior_fit,
                    cls_out.get("best_metrics"),
                    _cls_fit,
                    reqs=_reqs,
                    task_prompt=task_prompt,
                ):
                    best_params = cls_out.get("best_params", best_params)
                    best_metrics = cls_out.get("best_metrics", best_metrics)
                else:
                    from multi_agent.rl.metric_utils import get_peak_ny, get_sep

                    _pm = cls_out.get("best_metrics") or {}
                    _bm = best_metrics or {}
                    logger.warning(
                        "[OptimWorkflow] CLS not adopted over prior best "
                        "(prior: SEP=%.2fm PeakNy_mean=%.2fg | CLS: SEP=%.2fm PeakNy_mean=%.2fg)",
                        get_sep(_bm, 0.0),
                        get_peak_ny(_bm),
                        get_sep(_pm, 0.0),
                        get_peak_ny(_pm),
                    )
                rl_result = {**rl_result, **cls_out, "cls_applied": True}

            from multi_agent.integration.design_path_policy import (
                requirements_from_prompt,
                task_fully_satisfied,
            )

            if not task_fully_satisfied(best_metrics, task_prompt):
                _finish_out = await self._maybe_run_single_metric_finish(
                    script_path=script_path,
                    task_prompt=task_prompt,
                    mission_conditions=mission_conditions,
                    center_params=best_params or initial_auto_params or {},
                    center_metrics=best_metrics,
                )
                if _finish_out and self.rl_optimizer is not None:
                    _reqs = requirements_from_prompt(task_prompt)
                    _prior_fit = float(
                        self.rl_optimizer._compute_pe_fitness(best_metrics or {})
                    )
                    _fin_fit = float(
                        _finish_out.get("best_fitness")
                        or self.rl_optimizer._compute_pe_fitness(
                            _finish_out.get("best_metrics") or {}
                        )
                    )
                    from multi_agent.rl.metric_constraints import should_adopt_cls_result

                    if should_adopt_cls_result(
                        best_metrics,
                        _prior_fit,
                        _finish_out.get("best_metrics"),
                        _fin_fit,
                        reqs=_reqs,
                        task_prompt=task_prompt,
                    ):
                        best_params = _finish_out.get("best_params", best_params)
                        best_metrics = _finish_out.get("best_metrics", best_metrics)
                        rl_result = {**rl_result, **_finish_out, "single_metric_finish": True}

        # ── Reflection Agent — always runs ────────────────────────────────────
        reflection_result: Dict[str, Any] = {}
        if self.reflection_agent is not None and task_prompt:
            try:
                reflection_result = await self.reflection_agent.reflect(
                    task_prompt,
                    {"parameters": best_params, "metrics": best_metrics},
                    optimization_history=optimization_history,
                    script_path=script_path,
                )
                _rl_hp = reflection_result.get("rl_hyperparams", {})
                logger.info(
                    f"[OptimWorkflow] Reflection: "
                    f"needs_optimization={reflection_result.get('needs_optimization')} | "
                    f"{str(reflection_result.get('suggestion', ''))[:150]}\n"
                    f"  rl_hyperparams={_rl_hp}"
                )
            except Exception as exc:
                logger.error(f"[OptimWorkflow] Reflection agent failed: {exc}")
                reflection_result = {"needs_optimization": True, "suggestion": str(exc)}

        needs_opt = reflection_result.get("needs_optimization", True)
        suggestion = reflection_result.get("suggestion", "")

        from multi_agent.integration.design_path_policy import (
            apply_reflection_policy,
            is_optimization_task,
            resolve_next_action,
            task_fully_satisfied,
        )

        reflection_result = apply_reflection_policy(
            reflection_result,
            task_prompt=task_prompt,
            metrics=best_metrics,
            optimization_history=optimization_history,
        )
        needs_opt = reflection_result.get("needs_optimization", True)
        suggestion = reflection_result.get("suggestion", "")

        next_action = resolve_next_action(
            reflection_next_action=str(reflection_result.get("next_action", "")),
            task_prompt=task_prompt,
            metrics=best_metrics,
            optimization_history=optimization_history,
        )

        # Append design path analysis to suggestion for downstream consumers
        _design_path = reflection_result.get("design_path_analysis", "").strip()
        if _design_path:
            suggestion = f"{suggestion}\n\n【设计路径分析】{_design_path}"
            logger.info(f"[OptimWorkflow] Design path: {_design_path[:200]}")

        best_metrics = await self._maybe_revalidate_borderline_nmc(
            script_path=script_path,
            params=best_params,
            metrics=best_metrics,
            mission_conditions=mission_conditions,
            task_prompt=task_prompt,
        )
        from multi_agent.integration.judgment_agent import resolve_task_requirements

        _reqs = resolve_task_requirements(task_prompt, self.judgment_agent)
        _all_ok = task_fully_satisfied(best_metrics, task_prompt, _reqs)
        if _all_ok:
            best_metrics, _all_ok, _final_note = await self._maybe_validate_final_mc_seeds(
                script_path=script_path,
                params=best_params,
                metrics=best_metrics,
                mission_conditions=mission_conditions,
                task_prompt=task_prompt,
                nmc=nmc,
            )
            if not _all_ok and _final_note:
                if _final_note not in suggestion:
                    suggestion = f"{suggestion}\n{_final_note}".strip()
                logger.warning("[OptimWorkflow] %s", _final_note)
        if _all_ok:
            needs_opt = False
            reflection_result["needs_optimization"] = False
            next_action = "done"
        else:
            needs_opt = True
            reflection_result["needs_optimization"] = True
            if next_action == "done":
                next_action = resolve_next_action(
                    reflection_next_action="tune_params",
                    task_prompt=task_prompt,
                    metrics=best_metrics,
                    optimization_history=optimization_history,
                )
            if is_optimization_task(task_prompt):
                from multi_agent.integration.judgment_agent import _rule_based_check

                _, _fail_reasons = _rule_based_check(best_metrics, _reqs)
                _ng = [r for r in _fail_reasons if "[NG]" in r]
                if _ng:
                    _gate_note = "五项指标未全部达标: " + "; ".join(_ng[:5])
                    if _gate_note not in suggestion:
                        suggestion = f"{suggestion}\n{_gate_note}".strip()
                    logger.info("[OptimWorkflow] Requirement gate: %s", _gate_note[:200])

        # ── Stage C4：RAG 增强 next_action 决策 + 产出 next_action.json ──────
        # P0-4：产物归跑次包 deliverables/07_runs/{run_id}/next_action.json，
        #       不再落 guidance_output/；同时回填 runs 的 run_dir / next_action* 列。
        _na_extra: Dict[str, Any] = {"next_action_json": None, "kb_context": []}
        # 工况：mission_conditions 的键即 RUN_CASE（T/G/AP/R）；多键或空 → ALL
        # （与 cli_agent 的 _rc_store 规则一致，不猜）。
        _na_case: Optional[str] = None
        if isinstance(mission_conditions, dict) and mission_conditions:
            _na_keys = sorted(str(k) for k in mission_conditions.keys())
            _na_case = _na_keys[0] if len(_na_keys) == 1 else "ALL"
        _na_level = level
        if self.kb_manager is not None and self.emit_next_action_json:
            try:
                from multi_agent.memory.kb_next_action import (
                    default_run_id, enrich_and_emit_next_action,
                )

                _na_iter = len(optimization_history or [])
                if not self.run_id:
                    self.run_id = default_run_id(task_prompt)
                _na_payload, _na_path = await enrich_and_emit_next_action(
                    self.kb_manager,
                    None,                      # P0-4：不再落 guidance_output/next_action/
                    run_id=self.run_id,
                    runs_root=self.runs_root,
                    task_prompt=task_prompt,
                    best_metrics=best_metrics,
                    suggestion=suggestion,
                    base_next_action=next_action,
                    iteration=_na_iter,
                    case=_na_case,
                    level=_na_level,
                )
                _na_extra["next_action_json"] = _na_path
                _na_extra["kb_context"] = _na_payload.get("kb_context", [])
                if _na_path:
                    logger.info(
                        "[OptimWorkflow] C4 next_action.json -> %s (run_id=%s, source=%s, hits=%s)",
                        _na_path, self.run_id, _na_payload.get("source"),
                        _na_payload.get("retrieved_count"),
                    )
            except Exception as exc:  # 绝不拖死主循环
                logger.warning("[OptimWorkflow] C4 next_action 增强失败（已跳过）: %s", exc)

        # ── Path A: satisfied → write back + report ───────────────────────────
        if not needs_opt:
            writeback_ok = await self._writeback_experience(
                script_path, best_params, best_metrics, task_prompt
            )
            report_path = await self._generate_report(
                script_path, best_params, best_metrics, task_prompt,
                rl_result=rl_result,
                reflection_result=reflection_result,
                elapsed=(datetime.now() - _t0).total_seconds(),
            )
            logger.info(
                f"[OptimWorkflow] DONE ✓ | "
                f"{metrics_summary_line(best_metrics)}"
            )
            return {
                "status":          "done",
                "best_params":     best_params,
                "best_metrics":    best_metrics,
                "suggestion":      suggestion,
                "next_action":     "done",
                "run_id":          self.run_id,
                "next_action_json": _na_extra["next_action_json"],
                "kb_context":       _na_extra["kb_context"],
                "report_path":     report_path,
                "writeback_ok":    writeback_ok,
                "rl_hyperparams":  reflection_result.get("rl_hyperparams", {}),
                "total_episodes":  _rl_total_episodes,
                "best_reward":     _rl_best_reward,
                "baseline_reward": _rl_baseline_reward,
                "elapsed_sec":     (datetime.now() - _t0).total_seconds(),
            }

        # ── Path B: not satisfied → return suggestion to Layer 1 ─────────────
        logger.info(
            f"[OptimWorkflow] NEEDS_ITERATION | next_action={next_action} | "
            f"{suggestion[:120]}"
        )
        return {
            "status":          "needs_iteration",
            "best_params":     best_params,
            "best_metrics":    best_metrics,
                "suggestion":      suggestion,
                "next_action":     next_action,
                "run_id":          self.run_id,
                "next_action_json": _na_extra["next_action_json"],
                "kb_context":       _na_extra["kb_context"],
                "report_path":     "",
            "rl_hyperparams":  reflection_result.get("rl_hyperparams", {}),
            "total_episodes":  _rl_total_episodes,
            "best_reward":     _rl_best_reward,
            "baseline_reward": _rl_baseline_reward,
            "elapsed_sec":     (datetime.now() - _t0).total_seconds(),
        }

    # ── Constraint Local Search ─────────────────────────────────────────────

    async def _run_cls_core(
        self,
        *,
        centre: Dict[str, float],
        script_path: str,
        task_prompt: str,
        mission_conditions: Optional[Dict[str, List[int]]],
        max_steps: int,
        reference_params: Optional[Dict[str, float]] = None,
        reference_metrics: Optional[Dict[str, float]] = None,
        focus_metric_keys: Optional[Tuple[str, ...]] = None,
        step_scale_override: Optional[float] = None,
        label: str = "CLS",
    ) -> Optional[Dict[str, Any]]:
        if self.rl_optimizer is None or not centre:
            return None
        try:
            from multi_agent.config_loader import get_config
            from multi_agent.rl.constraint_local_search import ConstraintLocalSearch
            from multi_agent.rl.metric_constraints import MetricLockMargins
            from multi_agent.simulation.time_budget import get_run_budget

            cfg = get_config().matlab_rl_optimizer.constraint_local_search
            if not cfg.enabled:
                return None
            _budget = get_run_budget()
            _cls_steps = max_steps
            if _budget:
                _cls_steps = _budget.allowed_cls_steps(_cls_steps)
            print(
                f"  [Layer 3/{label}] Starting local search ({_cls_steps} steps, "
                f"fast_nmc={cfg.nmc_fast}, validate_nmc={cfg.nmc_validate})…"
            )
            cls = ConstraintLocalSearch(
                self.rl_optimizer,
                max_iterations=_cls_steps,
                step_scale=cfg.step_scale,
                seed=cfg.seed,
                no_regression_gate=cfg.no_regression_gate,
                directed_perturb=cfg.directed_perturb,
                lock_margins=MetricLockMargins(
                    sep_m=cfg.lock_margin_sep_m,
                    peak_ny_g=cfg.lock_margin_peak_ny_g,
                    hit_pct=cfg.lock_margin_hit_pct,
                ),
            )
            return await cls.refine(
                centre,
                script_path=script_path,
                mission_conditions=mission_conditions,
                nmc=cfg.nmc_validate,
                nmc_fast=cfg.nmc_fast,
                nmc_validate=cfg.nmc_validate,
                task_prompt=task_prompt,
                judgment_agent=self.judgment_agent,
                reference_params=reference_params or centre,
                reference_metrics=reference_metrics,
                borderline_select=getattr(cfg, "borderline_select", False),
                borderline_hit_min_pct=getattr(cfg, "borderline_hit_min_pct", HIT_RATE_MIN),
                borderline_sep_max_m=getattr(cfg, "borderline_sep_max_m", SEP_MAX),
                prefer_reference_peak=getattr(cfg, "prefer_ppo_best_peak", True),
                focus_metric_keys=focus_metric_keys,
                step_scale_override=step_scale_override,
            )
        except Exception as exc:
            logger.warning("[OptimWorkflow] %s skipped/failed: %s", label, exc)
            return None

    async def _maybe_run_single_metric_finish(
        self,
        *,
        script_path: str,
        task_prompt: str,
        mission_conditions: Optional[Dict[str, List[int]]],
        center_params: Dict[str, float],
        center_metrics: Dict[str, float],
    ) -> Optional[Dict[str, Any]]:
        """Extra CLS pass when exactly one metric (usually BW) still fails."""
        try:
            from multi_agent.config_loader import get_config
            from multi_agent.integration.design_path_policy import requirements_from_prompt
            from multi_agent.rl.metric_constraints import (
                failed_metric_keys,
                metric_satisfaction_flags,
            )
            from multi_agent.simulation.time_budget import get_run_budget

            cfg = get_config().matlab_rl_optimizer.constraint_local_search
            if not getattr(cfg, "single_metric_finish", True):
                return None
            reqs = requirements_from_prompt(task_prompt)
            flags = metric_satisfaction_flags(center_metrics, reqs)
            failed = failed_metric_keys(flags)
            if len(failed) != 1:
                return None
            _budget = get_run_budget()
            if _budget and _budget.should_stop(reserve_sec=120.0):
                logger.info(
                    "[OptimWorkflow] Skip single-metric finish — wall-clock budget low"
                )
                return None
            # P2-1 户部：成本预算不足时同样跳过这笔可选的追加计算
            from multi_agent import cost_meter as _sm_cost
            if _sm_cost.should_stop(reserve_ratio=0.10):
                logger.info(
                    "[OptimWorkflow] Skip single-metric finish — cost budget low"
                )
                return None
            # P2-6 工部：本次追加计算确实要跑了 —— 记一笔（与上面两处预算检查并列）
            from multi_agent import metrics as _sm_met

            _sm_met.counter("single_metric_finish_runs")
            focus = tuple(failed)
            steps = int(getattr(cfg, "bw_finish_max_steps", 12))
            if focus != ("bw",):
                steps = max(8, steps // 2)
            scale = float(getattr(cfg, "bw_finish_step_scale", 0.14))
            print(
                f"  [Layer 3] Single-metric finish ({focus[0]} only, {steps} steps)…"
            )
            return await self._run_cls_core(
                centre=center_params,
                script_path=script_path,
                task_prompt=task_prompt,
                mission_conditions=mission_conditions,
                max_steps=steps,
                reference_params=center_params,
                reference_metrics=center_metrics,
                focus_metric_keys=focus,
                step_scale_override=scale,
                label=f"CLS-{focus[0]}",
            )
        except Exception as exc:
            logger.warning("[OptimWorkflow] single-metric finish failed: %s", exc)
            return None

    @staticmethod
    def _has_required_mc_seed_results(
        metrics: Optional[Dict[str, Any]],
        seeds: List[int],
    ) -> bool:
        raw = (metrics or {}).get("mc_seed_results")
        if not isinstance(raw, list) or not raw:
            return False
        if not seeds:
            return True
        seen = set()
        for item in raw:
            if not isinstance(item, dict):
                continue
            seed = item.get("mc_seed", item.get("random_seed"))
            try:
                seen.add(int(seed))
            except (TypeError, ValueError):
                continue
        return set(int(s) for s in seeds).issubset(seen)

    async def _maybe_validate_final_mc_seeds(
        self,
        *,
        script_path: str,
        params: Dict[str, float],
        metrics: Dict[str, Any],
        mission_conditions: Optional[Dict[str, List[int]]],
        task_prompt: str,
        nmc: Optional[int],
    ) -> Tuple[Dict[str, Any], bool, str]:
        """Run configured multi-seed MC only at the final acceptance gate."""
        try:
            from multi_agent.config_loader import get_config
            from multi_agent.integration.design_path_policy import task_fully_satisfied
            from multi_agent.integration.judgment_agent import (
                _rule_based_check,
                resolve_task_requirements,
            )
            from multi_agent.tools.simulation_tool import (
                RunSimulationTool,
                _coerce_mc_seed_list,
            )
        except Exception as exc:
            logger.warning("[OptimWorkflow] Final MC imports failed: %s", exc)
            return metrics, False, f"Final MC validation unavailable: {exc}"

        cfg_root = get_config()
        mc_cfg = getattr(cfg_root.simulation, "mc_validation", None)
        if mc_cfg is None or not bool(getattr(mc_cfg, "enabled", False)):
            return metrics, True, ""

        mode = str(getattr(mc_cfg, "mode", "final") or "final").strip().lower()
        if mode in ("off", "none", "disabled"):
            return metrics, True, ""
        if mode not in ("final", "all", "all_simulations", "always"):
            return metrics, True, ""

        seeds = _coerce_mc_seed_list(getattr(mc_cfg, "seeds", []))
        if not seeds:
            return metrics, True, ""

        reqs = resolve_task_requirements(task_prompt, self.judgment_agent)
        if self._has_required_mc_seed_results(metrics, seeds):
            ok = task_fully_satisfied(metrics, task_prompt, reqs)
            if ok:
                return metrics, True, ""
            _, reasons = _rule_based_check(metrics, reqs)
            return (
                metrics,
                False,
                "Final MC validation failed: " + "; ".join(reasons[:5]),
            )

        if not script_path or not os.path.isfile(script_path):
            return metrics, False, "Final MC validation skipped: script not found"

        final_nmc = int(getattr(mc_cfg, "final_nmc", 0) or 0)
        if final_nmc <= 0:
            final_nmc = int(nmc or cfg_root.matlab_rl_optimizer.nmc_per_eval)

        mission_str = ""
        try:
            if isinstance(mission_conditions, str):
                mission_str = mission_conditions
            elif mission_conditions:
                from multi_agent.rl.matlab_rl_optimizer import build_matlab_conditions_str

                mission_str = build_matlab_conditions_str(mission_conditions)
        except Exception as exc:
            logger.debug("[OptimWorkflow] Could not stringify mission conditions: %s", exc)
            mission_str = ""

        simulator = getattr(self.rl_optimizer, "simulator", None) if self.rl_optimizer else None
        if simulator is None:
            executor_proxy = type("_FinalMcExecutorProxy", (), {})()
            executor_proxy.engine = str(getattr(cfg_root.simulation, "engine", "octave"))
            executor_proxy.octave_path = str(getattr(cfg_root.simulation, "octave_path", "octave"))
            executor_proxy.matlab_path = str(getattr(cfg_root.simulation, "matlab_path", "matlab"))
            simulator = type("_FinalMcSimulatorProxy", (), {})()
            simulator.executor = executor_proxy

        run_tool = RunSimulationTool()
        run_tool.set_simulator(simulator)
        # P2-5 兵部：消费工具返回前统一归一（不再按工具逐个假定 dict / JSON 串）
        from multi_agent.tools import tool_result as _tr
        if self.parameter_experience is not None:
            run_tool.set_parameter_experience(self.parameter_experience)
        if mission_str:
            run_tool.set_mission_conditions(mission_str)

        print(
            f"  [Layer 3] Final MC validation: seeds={seeds}, Nmc={final_nmc}"
        )
        try:
            result = await run_tool.execute(
                script_path=script_path,
                dp_params=params or {},
                mission_conditions=mission_str,
                nmc=final_nmc,
                mc_seeds=seeds,
                validate_final=True,
            )
        except (asyncio.CancelledError, KeyboardInterrupt):
            raise
        except Exception as exc:
            logger.warning("[OptimWorkflow] Final MC validation failed: %s", exc)
            return metrics, False, f"Final MC validation failed: {exc}"

        result = _tr.as_dict(result)
        if result.get("status") != "success":
            msg = result.get("message", "unknown error")
            return metrics, False, f"Final MC validation failed: {msg}"

        final_metrics = dict(
            getattr(run_tool, "last_metrics", {})
            or result.get("mc_seed_aggregate")
            or {}
        )
        if not final_metrics:
            return metrics, False, "Final MC validation failed: no metrics parsed"
        if "mc_seed_results" not in final_metrics and result.get("mc_seed_results"):
            final_metrics["mc_seed_results"] = result.get("mc_seed_results")

        ok = task_fully_satisfied(final_metrics, task_prompt, reqs)
        if ok:
            logger.info(
                "[OptimWorkflow] Final MC validation passed | seeds=%s | %s",
                seeds,
                metrics_summary_line(final_metrics),
            )
            return final_metrics, True, ""

        _, reasons = _rule_based_check(final_metrics, reqs)
        note = "Final MC validation failed: " + "; ".join(reasons[:5])
        return final_metrics, False, note

    async def _run_bw_finish_only(
        self,
        *,
        script_path: str,
        task_prompt: str,
        mission_conditions: Optional[Dict[str, List[int]]],
        center_params: Dict[str, float],
        center_metrics: Dict[str, float],
        optimization_history: Optional[List[Dict[str, Any]]],
        _t0: datetime,
    ) -> Dict[str, Any]:
        """Hot-start: skip Expert/PPO, run BW-focused CLS only."""
        from multi_agent.config_loader import get_config
        from multi_agent.integration.design_path_policy import (
            apply_reflection_policy,
            resolve_next_action,
            task_fully_satisfied,
        )

        cfg = get_config().matlab_rl_optimizer.constraint_local_search
        steps = int(getattr(cfg, "bw_finish_max_steps", 12))
        scale = float(getattr(cfg, "bw_finish_step_scale", 0.14))
        print("  [Layer 3] BW finish mode — Expert/PPO skipped")
        cls_out = await self._run_cls_core(
            centre=center_params,
            script_path=script_path,
            task_prompt=task_prompt,
            mission_conditions=mission_conditions,
            max_steps=steps,
            reference_params=center_params,
            reference_metrics=center_metrics,
            focus_metric_keys=("bw",),
            step_scale_override=scale,
            label="BW-Finish",
        )
        best_params = dict(center_params)
        best_metrics = dict(center_metrics)
        if cls_out:
            best_params = cls_out.get("best_params", best_params)
            best_metrics = cls_out.get("best_metrics", best_metrics)
        reflection_result: Dict[str, Any] = {}
        if self.reflection_agent is not None:
            try:
                reflection_result = await self.reflection_agent.reflect(
                    task_prompt,
                    {"parameters": best_params, "metrics": best_metrics},
                    optimization_history=optimization_history,
                    script_path=script_path,
                )
            except Exception as exc:
                reflection_result = {"needs_optimization": True, "suggestion": str(exc)}
        reflection_result = apply_reflection_policy(
            reflection_result,
            task_prompt=task_prompt,
            metrics=best_metrics,
            optimization_history=optimization_history,
        )
        final_note = ""
        if task_fully_satisfied(best_metrics, task_prompt):
            best_metrics, final_ok, final_note = await self._maybe_validate_final_mc_seeds(
                script_path=script_path,
                params=best_params,
                metrics=best_metrics,
                mission_conditions=mission_conditions,
                task_prompt=task_prompt,
                nmc=None,
            )
            status = "done" if final_ok else "needs_iteration"
        else:
            status = "needs_iteration"
        suggestion = reflection_result.get("suggestion", "")
        if final_note:
            suggestion = f"{suggestion}\n{final_note}".strip()
        return {
            "status": status,
            "best_params": best_params,
            "best_metrics": best_metrics,
            "suggestion": suggestion,
            "next_action": resolve_next_action(
                reflection_next_action=str(reflection_result.get("next_action", "")),
                task_prompt=task_prompt,
                metrics=best_metrics,
                optimization_history=optimization_history,
            ),
            "report_path": "",
            "total_episodes": cls_out.get("total_iterations", 0) if cls_out else 0,
            "elapsed_sec": (datetime.now() - _t0).total_seconds(),
        }

    async def _maybe_run_constraint_local_search(
        self,
        *,
        script_path: str,
        task_prompt: str,
        mission_conditions: Optional[Dict[str, List[int]]],
        center_params: Optional[Dict[str, float]],
        rl_result: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Optional CLS refinement after PPO when enabled in config."""
        if rl_result.get("status") in ("task_requirements_met", "requirements_met"):
            return None
        centre = center_params or rl_result.get("best_params") or {}
        if not centre:
            return None
        _ppo_ref_metrics = rl_result.get("best_metrics") or {}
        _ppo_ref_params = rl_result.get("best_params") or centre
        try:
            from multi_agent.integration.t4_low_risk import pre_cls_param_fixup

            centre = pre_cls_param_fixup(
                centre,
                _ppo_ref_metrics,
                task_prompt=task_prompt,
                mission_conditions=mission_conditions,
            )
        except Exception as _pfx_exc:
            logger.debug("[OptimWorkflow] pre-CLS fixup skipped: %s", _pfx_exc)
        try:
            from multi_agent.config_loader import get_config

            cfg = get_config().matlab_rl_optimizer.constraint_local_search
            if not cfg.enabled:
                return None
            out = await self._run_cls_core(
                centre=centre,
                script_path=script_path,
                task_prompt=task_prompt,
                mission_conditions=mission_conditions,
                max_steps=cfg.max_steps,
                reference_params=_ppo_ref_params,
                reference_metrics=_ppo_ref_metrics,
                label="CLS",
            )
            if out:
                logger.info(
                    f"[OptimWorkflow] CLS done | selection={out.get('selection_criteria')} | "
                    f"iterations={out.get('total_iterations', 0)}"
                )
            return out
        except Exception as exc:
            logger.warning(f"[OptimWorkflow] CLS skipped/failed: {exc}")
            return None

    # ── Strategy selection ─────────────────────────────────────────────────

    async def _select_strategy(
        self,
        metrics: Dict[str, float],
        task_prompt: str,
        optimization_history: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """Decide whether to use 'ppo' or 'expert' tuning for this iteration.

        Reads ``matlab_rl_optimizer.layer3_strategy`` from config.yaml:
          'expert' → always LLM-guided agent tuning (fast, 3 rounds)
          'ppo'    → always PPO reinforcement learning
          'auto'   → heuristic + LLM decides based on current metrics (default)

        Falls back to 'ppo' on any error.
        """
        if self.rl_optimizer is None:
            return "expert"

        from multi_agent.integration.t4_low_risk import should_force_expert_after_modify
        if should_force_expert_after_modify(optimization_history, task_prompt):
            logger.info("[OptimWorkflow] T4 low-risk → expert (post-MODIFY_LAW)")
            return "expert"

        # ── Config-driven override ─────────────────────────────────────────
        try:
            from multi_agent.config_loader import get_config
            _cfg_strategy = get_config().matlab_rl_optimizer.layer3_strategy
            _cfg_strategy = str(_cfg_strategy).lower().strip()
            if _cfg_strategy in ("expert", "ppo"):
                logger.info(
                    f"[OptimWorkflow] Strategy from config: {_cfg_strategy}"
                )
                return _cfg_strategy
        except Exception:
            pass  # fall through to heuristic / LLM-based selection

        hit = metrics.get("hit_rate", 0.0)
        sep = metrics.get("SEP", metrics.get("miss_distance", 99.0))
        pm  = metrics.get("pitch_PM", 0.0)
        bw  = metrics.get("pitch_BW", 0.0)

        from multi_agent.integration.design_path_policy import (
            peak_ny_limit,
            requirements_from_prompt,
            should_skip_layer3_param_search,
        )
        from multi_agent.rl.metric_utils import get_peak_ny

        reqs = requirements_from_prompt(task_prompt)
        pny_limit = peak_ny_limit(reqs)
        pny_avg = get_peak_ny(metrics)
        hit_min = float(reqs.get("hit_rate_min", HIT_RATE_MIN))
        sep_max = float(reqs.get("sep_max", SEP_MAX))
        pm_min = float(reqs.get("pm_min", PM_MIN))
        pm_max = float(reqs.get("pm_max", PM_MAX))
        bw_min = float(reqs.get("bw_min", BW_MIN))
        bw_max = float(reqs.get("bw_max", BW_MAX))

        if should_skip_layer3_param_search(metrics, task_prompt, optimization_history):
            logger.info(
                "[OptimWorkflow] Heuristic → skip param search (only PeakNy mean over limit)"
            )
            return "skip"

        # ── Heuristic fast-path (avoid unnecessary LLM call) ──────────────
        _n_fail = 0
        if hit < hit_min:
            _n_fail += 1
        if sep > sep_max:
            _n_fail += 1
        if pny_avg > pny_limit:
            _n_fail += 1
        if pm < pm_min or pm > pm_max:
            _n_fail += 1
        if bw < bw_min or bw > bw_max:
            _n_fail += 1

        if _n_fail <= 1:
            logger.info(f"[OptimWorkflow] Heuristic → expert (only {_n_fail} failing metric(s))")
            return "expert"

        if _n_fail >= 4:
            logger.info(f"[OptimWorkflow] Heuristic → ppo ({_n_fail} failing metrics)")
            return "ppo"

        ny = pny_avg if pny_avg > 0 else metrics.get("peak_ny", metrics.get("peak_n", 0.0))
        try:
            from multi_agent.config_loader import get_config

            # Build history summary for LLM context
            _hist_summary = ""
            if optimization_history:
                _lines = []
                for h in optimization_history[-5:]:
                    _lines.append(
                        f"  iter={h.get('iteration','')} mode={h.get('task_mode','')} "
                        f"hit={h.get('hit_rate',0):.1f}% SEP={h.get('SEP',99):.2f}m"
                    )
                _hist_summary = f"\nRecent optimization history:\n" + "\n".join(_lines) + "\n"

            cfg    = get_config().llm
            from multi_agent.llm_failover import chat_completion_create_with_failover
            from multi_agent.llm_utils import extract_chat_message_text, get_safe_temperature
            prompt = (
                "You are a missile guidance optimization expert deciding the Layer 3 "
                "optimization strategy.\n\n"
                f"Current simulation metrics:\n"
                f"  hit_rate={hit:.1f}%  SEP={sep:.2f}m  "
                f"PeakNy_avg={ny:.2f}g (limit {pny_limit:.0f}g)  "
                f"PM={pm:.1f}°  BW={bw:.1f}rad/s\n"
                f"{_hist_summary}\n"
                f"Task requirements (extract from text below):\n{task_prompt[:600]}\n\n"
                "Strategy options:\n"
                "  'expert': LLM-guided expert analysis — 3 rounds of suggest→simulate→check. "
                "Fast (~3 simulations). PREFERRED when only 1-2 metrics need small adjustments, "
                "or when metrics are close to requirements (within ~10-20% of targets). "
                "Also preferred when previous iterations used MODIFY_LAW and improved metrics.\n"
                "  'ppo': PPO reinforcement learning — 10-50 episodes, systematic exploration. "
                "Use ONLY when ≥3 metrics are significantly off (>30% from target) AND "
                "expert tuning has already been tried without success.\n\n"
                "Return ONLY this JSON (no extra text): "
                "{\"strategy\": \"expert\" or \"ppo\", \"reason\": \"one sentence\"}"
            )
            resp = await chat_completion_create_with_failover(
                messages=[{"role": "user", "content": prompt}],
                temperature=get_safe_temperature(cfg.model, 0.1),
                max_tokens=256,
                thinking=False,
            )
            raw = extract_chat_message_text(resp)
            m   = re.search(r"\{[^}]+\}", raw, re.DOTALL)
            if m:
                data     = json.loads(m.group())
                strategy = str(data.get("strategy", "ppo")).lower().strip()
                reason   = data.get("reason", "")
                if strategy in ("expert", "ppo"):
                    logger.info(f"[OptimWorkflow] Strategy={strategy} — {reason}")
                    return strategy
        except (asyncio.CancelledError, KeyboardInterrupt):
            raise
        except Exception as exc:
            logger.warning(f"[OptimWorkflow] Strategy LLM failed ({exc}), defaulting to ppo")

        return "ppo"

    # ── Expert tuning loop ─────────────────────────────────────────────────

    async def _run_expert_tuning(
        self,
        script_path: str,
        task_prompt: str,
        mission_conditions: Optional[Dict[str, List[int]]],
        initial_metrics: Dict[str, float],
        max_rounds: int = 3,
        nmc: int = 30,
        initial_params: Optional[Dict[str, float]] = None,
        optimization_history: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """LLM-guided expert parameter tuning — up to *max_rounds* iterations.

        Each round:
          1. LLM analyses current metrics and suggests specific parameter values.
          2. Parameters are clamped to valid ranges and written into the script.
          3. Simulation is run; metrics are parsed.
          4. JudgmentAgent (or simple rule check) decides if requirements are met.

        Returns a dict with the same shape as ``rl_optimizer.optimize()`` so the
        caller can treat it uniformly:
          status  : "task_requirements_met" | "needs_iteration"
          best_params, best_metrics, total_episodes, best_reward, baseline_reward
        """
        from multi_agent.config_loader import get_config
        from multi_agent.rl.matlab_rl_optimizer import ALL_TUNABLE_PARAM_SPECS
        from multi_agent.security.parameter_firewall import clamp_tunable_params

        cfg = get_config().llm

        # ── Parameter metadata for LLM prompt (compact to avoid empty responses)
        param_effects = (
            "★ PM & BW 主要取决于俯仰通道(pitch):\n"
            "  w1↑ → BW↑, PM↓ (带宽增大但相位裕度降低)\n"
            "  zeta1↑ → PM↑ (阻尼增大→稳定性增强)\n"
            "  tao1↑ → PM↓↓ (时间常数增大→严重降低稳定性, tao1>0.3极易不稳定!)\n"
            "★ 安全区域: w1∈[30,50], zeta1∈[0.6,1.0], tao1∈[0.08,0.20]\n"
            "★ PM<0 表示系统不稳定! 必须: 降低tao1, 增大zeta1, 适当降低w1\n"
            "★ 偏航(w2,zeta2,tao2): 影响横向SEP. 滚转(w3,zeta3): 影响PeakNy\n"
            "★ N_pn(2-6): ↑→↑命中率,↓SEP,↑PeakNy. 推荐N_pn=3~5\n"
        )
        bounds_desc = " | ".join(
            f"{k}[{s['min']},{s['max']}]"
            for k, s in ALL_TUNABLE_PARAM_SPECS.items()
        )

        current_metrics: Dict[str, float] = dict(initial_metrics)
        best_metrics:    Dict[str, float] = dict(initial_metrics)
        best_params:     Dict[str, float] = dict(initial_params or {})
        history: List[Dict[str, Any]]     = []

        from multi_agent.simulation.time_budget import get_run_budget

        _ex_budget = get_run_budget()
        from multi_agent import cost_meter as _ex_cost
        from multi_agent import metrics as _ex_met

        for round_i in range(1, max_rounds + 1):
            if _ex_budget and _ex_budget.should_stop(reserve_sec=120.0):
                logger.warning(
                    "[ExpertTuning] Wall-clock budget — stopping before round %d",
                    round_i,
                )
                break
            # P2-1 户部：成本预算同样在每轮开头咨询（与墙钟预算并列）
            if _ex_cost.should_stop(reserve_ratio=0.10):
                logger.warning(
                    "[ExpertTuning] Cost budget — stopping before round %d", round_i
                )
                break
            # P2-6 工部：每轮 ExpertTuning 记一笔（与上面两处预算检查并列）
            _ex_met.counter("expert_tuning_rounds")
            # Brief pause between rounds to avoid API rate-limiting
            if round_i > 1:
                await asyncio.sleep(3)
            hit = current_metrics.get("hit_rate", 0.0)
            sep = current_metrics.get("SEP", current_metrics.get("miss_distance", 99.0))
            ny  = current_metrics.get("peak_ny", current_metrics.get("peak_n", 0.0))
            pm  = current_metrics.get("pitch_PM", 0.0)
            bw  = current_metrics.get("pitch_BW", 0.0)

            # Keep only last 3 rounds to limit prompt length
            _recent = history[-3:] if len(history) > 3 else history
            history_str = (
                "None"
                if not _recent else
                "\n".join(
                    f"  R{h['round']}: hit={h['metrics'].get('hit_rate',0):.0f}% "
                    f"SEP={h['metrics'].get('SEP',99):.1f} "
                    f"Ny={h['metrics'].get('peak_ny',0):.1f}g "
                    f"PM={h['metrics'].get('pitch_PM',0):.0f}° "
                    f"BW={h['metrics'].get('pitch_BW',0):.0f}"
                    for h in _recent
                )
            )

            # ── Build prev-iteration reflection context ──
            _prev_iter_ctx = ""
            if optimization_history:
                _last = optimization_history[-1]
                _prev_suggest = _last.get("suggestion", "")[:200]
                _prev_action  = _last.get("next_action", "")
                _prev_dpa     = _last.get("design_path_analysis", "")[:200]
                _prev_iter_ctx = (
                    f"\n[上轮反思建议 (next_action={_prev_action})]\n"
                    f"{_prev_suggest}\n"
                    + (f"[设计路径分析] {_prev_dpa}\n" if _prev_dpa else "")
                )

            # ── Build compact Chinese prompt (DeepSeek responds better) ──
            from multi_agent.integration.t4_low_risk import (
                build_expert_playbook_block,
                enforce_expert_suggestion,
                expert_llm_temperature,
                rule_seed_expert_params,
                should_use_rule_first_expert_round,
            )

            _t4_cfg = None
            try:
                from multi_agent.config_loader import get_config as _get_cfg

                _t4_cfg = getattr(_get_cfg(), "t4_low_risk", None)
            except Exception:
                pass

            suggested: Optional[Dict[str, float]] = None
            analysis = ""

            _rule_seed = rule_seed_expert_params(
                current_metrics,
                round_i,
                best_params,
                task_prompt=task_prompt,
                mission_conditions=mission_conditions,
            )
            if _rule_seed and should_use_rule_first_expert_round(
                round_i,
                current_metrics,
                task_prompt=task_prompt,
                mission_conditions=mission_conditions,
                cfg=_t4_cfg,
            ):
                suggested = _rule_seed
                analysis = "T4 rule-first R1 seed (BW<20 playbook, LLM skipped)"
                logger.info(
                    "[ExpertTuning] Round %d rule-first seed w1=%.1f (skip LLM)",
                    round_i,
                    float(suggested.get("w1", 0)),
                )
                mprint(f"  [Expert] Round {round_i}/{max_rounds}: rule-first R1 seed")

            raw = None
            if suggested is None:
                _playbook = build_expert_playbook_block(
                    current_metrics,
                    round_i,
                    task_prompt=task_prompt,
                    mission_conditions=mission_conditions,
                )
                prompt = (
                    f"导弹驾驶仪调参，第{round_i}轮。\n"
                    f"当前指标: 命中率={hit:.1f}% SEP={sep:.2f}m 峰值过载={ny:.1f}g PM={pm:.1f}° BW={bw:.1f}rad/s\n"
                    f"要求: 命中率>=92% SEP<=7m 峰值过载<=20g PM在45~70° BW在20~85\n"
                    f"{_prev_iter_ctx}"
                    f"{param_effects}"
                    f"参数范围: {bounds_desc}\n"
                    f"历史: {history_str}\n"
                    f"{_playbook}\n"
                    "请直接给出9个参数的数值，只返回JSON:\n"
                    '{"w1":数值,"zeta1":数值,"tao1":数值,'
                    '"w2":数值,"zeta2":数值,"tao2":数值,'
                    '"w3":数值,"zeta3":数值,"N_pn":数值,"analysis":"原因"}'
                )

                # ── LLM call (with retry + balance failover + jittered backoff)
                from multi_agent.llm_utils import (
                    describe_chat_choice,
                    exponential_backoff_seconds,
                    extract_chat_message_text,
                    get_safe_temperature,
                    llm_max_retries,
                )
                from multi_agent.llm_failover import (
                    chat_completion_create_with_failover,
                    is_balance_error,
                )

                _max_att = llm_max_retries(5)
                for _attempt in range(1, _max_att + 1):
                    try:
                        _temp = expert_llm_temperature(_attempt, _t4_cfg)
                        _safe_temp = get_safe_temperature(cfg.model, _temp)
                        # Outer loop owns the 1s→2s→4s…±20% budget; failover does 1 try.
                        resp = await chat_completion_create_with_failover(
                            messages=[{"role": "user", "content": prompt}],
                            temperature=_safe_temp,
                            max_tokens=4096,
                            timeout=120.0,
                            max_retries_per_endpoint=1,
                            thinking=False,
                        )
                        raw = extract_chat_message_text(resp)
                        if raw:
                            break
                        logger.warning(
                            "[ExpertTuning] Round %d attempt %d: empty LLM response (%s)",
                            round_i,
                            _attempt,
                            describe_chat_choice(resp),
                        )
                    except (asyncio.CancelledError, KeyboardInterrupt):
                        raise
                    except Exception as exc:
                        logger.error(
                            f"[ExpertTuning] Round {round_i} attempt {_attempt} LLM failed: {exc}"
                        )
                        mprint(
                            f"  [Expert] Round {round_i} attempt {_attempt} LLM failed: {exc}"
                        )
                        # Billing hard-fail: do not burn remaining attempts on sleep.
                        if is_balance_error(exc):
                            break
                    if _attempt < _max_att and not raw:
                        _delay = exponential_backoff_seconds(_attempt)
                        mprint(
                            f"  [Expert] backoff {_delay:.2f}s before attempt {_attempt + 1}/{_max_att}"
                        )
                        await asyncio.sleep(_delay)

            if suggested is None and not raw:
                if _rule_seed:
                    suggested = _rule_seed
                    analysis = "rule seed fallback (LLM empty)"
                    logger.warning(
                        "[ExpertTuning] Round %d: LLM empty → rule seed fallback",
                        round_i,
                    )
                    mprint(f"  [Expert] Round {round_i}: LLM empty → rule seed")
                else:
                    import random as _rnd

                    logger.warning(
                        f"[ExpertTuning] Round {round_i}: all attempts empty, using random perturbation"
                    )
                    mprint(
                        f"  [Expert] Round {round_i}: LLM empty → random pitch perturbation fallback"
                    )
                    _fb_raw: Dict[str, float] = {}
                    for k, spec in ALL_TUNABLE_PARAM_SPECS.items():
                        base_v = best_params.get(k, spec["nominal"])
                        half_range = (spec["max"] - spec["min"]) / 2.0
                        scale = 0.20 if k in ("w1", "zeta1", "tao1") else 0.05
                        delta = _rnd.uniform(-scale, scale) * half_range
                        _fb_raw[k] = base_v + delta
                    suggested = clamp_tunable_params(_fb_raw, fill_missing=True)
                    analysis = "random perturbation (LLM empty)"

            if suggested is None:
                # ── Parse suggested params ─────────────────────────────────
                try:
                    _clean = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`")

                    def _try_repair_json(s: str) -> Optional[str]:
                        if "{" not in s:
                            return None
                        frag = s[s.index("{"):]
                        frag = re.sub(r",\s*\"[^\"]*\"?\s*:\s*[\d.]*$", "", frag)
                        frag = re.sub(r",\s*\"[^\"]*\"?\s*$", "", frag)
                        frag = re.sub(r",\s*$", "", frag)
                        if not frag.endswith("}"):
                            frag += "}"
                        try:
                            json.loads(frag)
                            return frag
                        except Exception:
                            return None

                    m_json = re.search(r"\{[^{}]*\}", _clean, re.DOTALL)
                    if not m_json:
                        m_json = re.search(r"\{.*\}", _clean, re.DOTALL)
                    if not m_json:
                        _repaired = _try_repair_json(_clean)
                        if _repaired:
                            m_json = re.search(r"\{.*\}", _repaired, re.DOTALL)
                    if not m_json:
                        raise ValueError(f"No JSON in LLM response: {raw[:200]}")
                    try:
                        data = json.loads(m_json.group())
                    except json.JSONDecodeError:
                        _repaired2 = _try_repair_json(m_json.group())
                        if _repaired2:
                            data = json.loads(_repaired2)
                        else:
                            raise
                    analysis = str(data.pop("analysis", ""))
                    suggested = clamp_tunable_params(data, fill_missing=True)
                except Exception as exc:
                    if _rule_seed:
                        suggested = _rule_seed
                        analysis = "rule seed fallback (JSON parse failed)"
                        logger.warning(
                            "[ExpertTuning] Round %d JSON parse failed → rule seed: %s",
                            round_i,
                            exc,
                        )
                        mprint(f"  [Expert] Round {round_i} JSON failed → rule seed")
                    else:
                        logger.warning(
                            f"[ExpertTuning] Round {round_i} JSON parse failed: {exc} | raw={raw[:200]}"
                        )
                        mprint(f"  [Expert] Round {round_i} JSON parse failed: {exc}")
                        continue

            suggested = enforce_expert_suggestion(
                suggested,
                current_metrics,
                round_i=round_i,
                task_prompt=task_prompt,
                mission_conditions=mission_conditions,
            )

            logger.info(
                f"[ExpertTuning] Round {round_i} params: "
                + " ".join(f"{k}={v:.3f}" for k, v in suggested.items())
                + f" | {analysis}"
            )

            # ── Simulate ───────────────────────────────────────────────────
            if self.rl_optimizer is None:
                logger.error("[ExpertTuning] No rl_optimizer available for simulation")
                break
            try:
                sim_metrics = await self.rl_optimizer._run_simulation_with_params(
                    auto_params=suggested,
                    mission_conditions=mission_conditions,
                    script_path=script_path,
                    nmc=nmc,
                )
                current_metrics = dict(sim_metrics)
            except (asyncio.CancelledError, KeyboardInterrupt):
                raise
            except Exception as exc:
                logger.error(f"[ExpertTuning] Round {round_i} simulation failed: {exc}")
                mprint(f"  [Expert] Round {round_i} simulation failed: {exc}")
                continue

            from multi_agent.rl.matlab_rl_optimizer import is_reliable_sim_metrics

            if not is_reliable_sim_metrics(current_metrics):
                logger.warning(
                    "[ExpertTuning] Round %d unreliable sim (timeout/fallback) — skip best update",
                    round_i,
                )
                mprint(
                    f"  [Expert] Round {round_i}/{max_rounds}: "
                    "sim unreliable (timeout/fallback) — skipped"
                )
                history.append({
                    "round":    round_i,
                    "params":   {k: round(v, 4) for k, v in suggested.items()},
                    "metrics":  dict(current_metrics),
                    "analysis": analysis,
                    "unreliable": True,
                })
                continue

            _prior_score = (
                _expert_feasibility_score(best_metrics)
                if is_reliable_sim_metrics(best_metrics)
                else -1e9
            )
            _round_improved = _expert_feasibility_score(current_metrics) > _prior_score
            _summary = metrics_summary_line(current_metrics)
            if should_log_expert_round(round_i, max_rounds, improved=_round_improved):
                _line = f"  [Expert] Round {round_i}/{max_rounds}: {_summary}"
                if is_verbose():
                    cprint(_line + (f" | {analysis[:80]}" if analysis else ""))
                else:
                    cprint(_line + (" ★" if _round_improved else ""))
            else:
                logger.debug(f"[ExpertTuning] Round {round_i} → {_summary}")

            history.append({
                "round":    round_i,
                "params":   {k: round(v, 4) for k, v in suggested.items()},
                "metrics":  dict(current_metrics),
                "analysis": analysis,
            })

            # Update best using composite feasibility score (not just hit_rate)
            if _round_improved:
                best_metrics = dict(current_metrics)
                best_params  = dict(suggested)

            # Budget-aware early stop: leave wall-clock for CLS
            if _ex_budget and round_i >= 1:
                _rem = _ex_budget.remaining_sec()
                if _rem is not None and _rem < 780:
                    logger.info(
                        "[ExpertTuning] Wall-clock low (%.0fs) after round %d → CLS",
                        _rem,
                        round_i,
                    )
                    break

            # ── Check requirements via JudgmentAgent ───────────────────────
            if self.judgment_agent is not None:
                try:
                    judge = await self.judgment_agent.judge(
                        metrics=current_metrics,
                        task_prompt=task_prompt,
                    )
                    _satisfied = (
                        judge.satisfied if hasattr(judge, "satisfied")
                        else judge.get("satisfied", False) if isinstance(judge, dict)
                        else False
                    )
                    if _satisfied:
                        logger.info(
                            f"[ExpertTuning] ✓ Requirements met at round {round_i}!"
                        )
                        return {
                            "status":          "task_requirements_met",
                            "best_params":     best_params,
                            "best_metrics":    best_metrics,
                            "history":         history,
                            "strategy":        "expert",
                            "total_episodes":  round_i,
                            "best_reward":     0.0,
                            "baseline_reward": 0.0,
                        }
                except (asyncio.CancelledError, KeyboardInterrupt):
                    raise
                except Exception as exc:
                    logger.warning(f"[ExpertTuning] JudgmentAgent failed: {exc}")

        logger.info(
            f"[ExpertTuning] Finished {len(history)} rounds without meeting requirements. "
            f"Best: hit={best_metrics.get('hit_rate',0):.1f}% "
            f"SEP={best_metrics.get('SEP',99):.2f}m"
        )
        return {
            "status":          "needs_iteration",
            "best_params":     best_params,
            "best_metrics":    best_metrics,
            "history":         history,
            "strategy":        "expert",
            "total_episodes":  len(history),
            "best_reward":     0.0,
            "baseline_reward": 0.0,
        }

    async def _maybe_revalidate_borderline_nmc(
        self,
        *,
        script_path: str,
        params: Dict[str, float],
        metrics: Dict[str, float],
        mission_conditions: Optional[Dict[str, List[int]]],
        task_prompt: str,
    ) -> Dict[str, float]:
        """Re-run with larger Nmc when peak is borderline (20, 25]g and other metrics OK."""
        from multi_agent.integration.t4_low_risk import borderline_nmc_revalidate_threshold
        from multi_agent.integration.design_path_policy import (
            non_peak_constraints_met,
            peak_ny_limit,
            peak_ny_over_limit,
            requirements_from_prompt,
        )
        from multi_agent.rl.metric_utils import get_peak_ny, normalize_peak_ny_aliases

        nmc_big = borderline_nmc_revalidate_threshold(task_prompt)
        if nmc_big <= 0 or self.rl_optimizer is None:
            return metrics

        reqs = requirements_from_prompt(task_prompt)
        limit = peak_ny_limit(reqs)
        pny = get_peak_ny(metrics)
        if not (
            non_peak_constraints_met(metrics, reqs)
            and peak_ny_over_limit(metrics, limit)
            and pny <= limit + 5.0
        ):
            return metrics

        try:
            print(f"  [Layer 3] Borderline PeakNy mean {pny:.2f}g — Nmc={nmc_big} revalidation…")
            big_metrics = await self.rl_optimizer._run_simulation_with_params(
                params,
                mission_conditions,
                script_path,
                nmc_big,
            )
            big_metrics = normalize_peak_ny_aliases(big_metrics)
            logger.info(
                "[OptimWorkflow] Borderline revalidate Nmc=%d → PeakNy_avg=%.2fg hit=%.1f%%",
                nmc_big,
                get_peak_ny(big_metrics),
                big_metrics.get("hit_rate", 0),
            )
            return big_metrics
        except Exception as exc:
            logger.warning("[OptimWorkflow] Borderline Nmc revalidation failed: %s", exc)
            return metrics

    async def _writeback_experience(
        self,
        script_path: str,
        params: Dict[str, float],
        metrics: Dict[str, float],
        task_prompt: str,
    ) -> bool:
        """Write best params and model to ParameterExperience."""
        if self.parameter_experience is None:
            return False
        try:
            from multi_agent.integration.t4_low_risk import should_store_pe_memory
            _ok, _why = should_store_pe_memory(metrics, task_prompt)
            if not _ok:
                logger.info("[OptimWorkflow] PE writeback skipped: %s", _why)
                return False
            pe_params = {f"dp_{k}": v for k, v in params.items()}
            fitness = (
                metrics.get("hit_rate", 0.0) / 100.0
                - metrics.get("SEP", 50.0) / 50.0 * 0.3
            )
            await self.parameter_experience.store(
                task_context={"prompt": task_prompt[:200]},
                parameters=pe_params,
                objectives=metrics,
                fitness=fitness,
            )

            # Copy script to experience_base/models/
            if script_path and os.path.isfile(script_path):
                model_dir = Path(self.output_dir) / "parameter_experience_base" / "models"
                model_dir.mkdir(parents=True, exist_ok=True)
                import shutil, time as _time
                dest = model_dir / f"success_{int(_time.time())}_{os.path.basename(script_path)}"
                shutil.copy2(script_path, str(dest))
                logger.info(f"[OptimWorkflow] Model saved → {dest}")
            return True
        except Exception as exc:
            logger.error(f"[OptimWorkflow] Experience writeback failed: {exc}")
            return False

    async def _generate_report(
        self,
        script_path: str,
        params: Dict[str, float],
        metrics: Dict[str, float],
        task_prompt: str,
        rl_result: Dict[str, Any] = None,
        reflection_result: Dict[str, Any] = None,
        elapsed: float = 0.0,
    ) -> str:
        """Write a JSON+TXT report to output_dir and return the path."""
        try:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_dir = Path(self.output_dir) / "reports"
            report_dir.mkdir(parents=True, exist_ok=True)
            report_path = str(report_dir / f"report_{ts}.json")

            report = {
                "timestamp": ts,
                "task_prompt": task_prompt[:500],
                "script_path": script_path,
                "elapsed_sec": elapsed,
                "best_params": params,
                "best_metrics": metrics,
                "rl_summary": {
                    "status":        (rl_result or {}).get("status", ""),
                    "total_episodes":(rl_result or {}).get("total_episodes", 0),
                    "best_reward":   (rl_result or {}).get("best_reward", 0.0),
                },
                "reflection": {
                    "needs_optimization": (reflection_result or {}).get("needs_optimization"),
                    "suggestion": (reflection_result or {}).get("suggestion", "")[:1000],
                },
                "metrics_display": {
                    "命中率%":  f"{metrics.get('hit_rate', 0):.1f}",
                    "SEP(m)":   f"{metrics.get('SEP', metrics.get('miss_distance', 0)):.2f}",
                    "PeakNy(g)":f"{metrics.get('peak_ny', metrics.get('peak_n', 0)):.2f}",
                    "PM(°)":    f"{metrics.get('pitch_PM', 0):.1f}",
                    "BW(r/s)":  f"{metrics.get('pitch_BW', 0):.1f}",
                },
            }

            with open(report_path, "w", encoding="utf-8") as fh:
                json.dump(report, fh, ensure_ascii=False, indent=2)
            logger.info(f"[OptimWorkflow] Report saved → {report_path}")
            return report_path
        except Exception as exc:
            logger.error(f"[OptimWorkflow] Report generation failed: {exc}")
            return ""
