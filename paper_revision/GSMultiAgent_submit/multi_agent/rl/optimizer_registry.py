#!/usr/bin/env python3
"""
optimizer_registry.py — Section 4.2.1 pluggable optimizer backend factory.

Supported backends (config key: matlab_rl_optimizer.optimizer_backend):
    hybrid_ppo    — GSMultiAgent default: expert + PPO + CLS (no change)
    direct_ppo    — PPO-only, no expert LLM tuning, no CLS
    cma_es        — Full-covariance CMA-ES, λ = 4+floor(3·ln(d)), σ₀ = 0.25
    random_search — Uniform random, performance lower bound

All backends expose the same async interface so OptimizationWorkflow can swap
them transparently without touching any other component.  Every backend is
capped by the same ``max_episodes`` candidate-evaluation budget (not CMA-ES
generations).
"""
from __future__ import annotations

import asyncio
import logging
import math
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ── shared param-space helpers (reuse specs from the RL optimizer) ─────────────

def _get_param_specs() -> Dict[str, Any]:
    from multi_agent.rl.matlab_rl_optimizer import ALL_TUNABLE_PARAM_SPECS
    return dict(ALL_TUNABLE_PARAM_SPECS)

def _clip(params: Dict[str, float]) -> Dict[str, float]:
    specs = _get_param_specs()
    return {
        name: float(np.clip(float(params.get(name, specs[name]["nominal"])),
                            specs[name]["min"], specs[name]["max"]))
        for name in specs
    }

def _norm_to_params(x: np.ndarray) -> Dict[str, float]:
    """Map normalized [0,1]^d → physical params."""
    specs = _get_param_specs()
    names = list(specs)
    return {
        name: float(specs[name]["min"]) + float(x[i]) * (
            float(specs[name]["max"]) - float(specs[name]["min"])
        )
        for i, name in enumerate(names)
    }

def _params_to_norm(params: Dict[str, float]) -> np.ndarray:
    specs = _get_param_specs()
    names = list(specs)
    arr = np.zeros(len(names))
    for i, name in enumerate(names):
        lo, hi = float(specs[name]["min"]), float(specs[name]["max"])
        arr[i] = np.clip((float(params.get(name, (lo+hi)/2)) - lo) / max(hi - lo, 1e-12), 0.0, 1.0)
    return arr


# ── common evaluator wrapper ───────────────────────────────────────────────────

class _SimEvaluator:
    """Thin wrapper that calls _run_simulation_with_params from rl_optimizer."""

    def __init__(self, rl_optimizer, script_path: str,
                 mission_conditions: Optional[Dict], nmc: int) -> None:
        self._opt = rl_optimizer
        self._script = script_path
        self._mc = mission_conditions
        self._nmc = nmc
        self.call_count = 0
        self.sim_time_sec = 0.0

    async def __call__(self, params: Dict[str, float]) -> Dict[str, float]:
        self.call_count += 1
        metrics = await self._opt._run_simulation_with_params(
            auto_params=_clip(params),
            mission_conditions=self._mc,
            script_path=self._script,
            nmc=self._nmc,
        )
        self.sim_time_sec += float(metrics.get("_sim_time_sec", 0.0))
        return metrics

    @staticmethod
    def _feasible(m: Dict[str, float]) -> bool:
        from multi_agent.tools.simulation_tool import check_constraints
        ok, _ = check_constraints(
            m.get("hit_rate", 0.0),
            m.get("SEP", m.get("miss_distance", 999.0)),
            m.get("peak_ny", m.get("peak_nz", 999.0)),
            m.get("pitch_PM", 0.0),
            m.get("pitch_BW", 0.0),
        )
        return ok

    @staticmethod
    def _score(m: Dict[str, float], all_pass: bool) -> float:
        if all_pass:
            return -(m.get("hit_rate", 0.0) / 100.0 + 0.01)
        v = 0.0
        from multi_agent.tools.simulation_tool import check_constraints
        _, viols = check_constraints(
            m.get("hit_rate", 0.0),
            m.get("SEP", m.get("miss_distance", 999.0)),
            m.get("peak_ny", m.get("peak_nz", 999.0)),
            m.get("pitch_PM", 0.0),
            m.get("pitch_BW", 0.0),
        )
        v = sum(float(x) for x in viols.values())
        return v


# ── abstract base ──────────────────────────────────────────────────────────────

class OptimizerBackend(ABC):
    """Common interface for all Layer-3 optimizer backends."""

    @abstractmethod
    async def optimize(
        self,
        *,
        rl_optimizer,                          # MatlabRLOptimizer instance
        script_path: str,
        mission_conditions: Optional[Dict],
        max_episodes: int,
        nmc: int,
        task_prompt: str,
        initial_auto_params: Optional[Dict[str, float]] = None,
        warm_start_source: Optional[str] = None,
        optimization_history: Optional[List] = None,
        reflection_agent=None,
        resume_checkpoint: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Returns dict compatible with MatlabRLOptimizer.optimize():
            status, best_params, best_metrics, best_reward,
            total_episodes, history, mc_calls, sim_time_sec
        """


# ── hybrid_ppo (default — delegate entirely to MatlabRLOptimizer) ─────────────

class HybridPPOBackend(OptimizerBackend):
    """GSMultiAgent default: expert + PPO + CLS.  No change from baseline."""

    async def optimize(self, *, rl_optimizer, **kwargs) -> Dict[str, Any]:
        return await rl_optimizer.optimize(**kwargs)


# ── direct_ppo — PPO only, no expert tuning ───────────────────────────────────

class DirectPPOBackend(OptimizerBackend):
    """PPO-only.  Forces layer3_strategy='ppo', disables expert path."""

    async def optimize(self, *, rl_optimizer, **kwargs) -> Dict[str, Any]:
        orig = rl_optimizer.strategy
        try:
            rl_optimizer.strategy = "ppo"
            return await rl_optimizer.optimize(**kwargs)
        finally:
            rl_optimizer.strategy = orig


# ── CMA-ES backend ─────────────────────────────────────────────────────────────

class CMAESBackend(OptimizerBackend):
    """Full-covariance CMA-ES (μ/μ_w,λ-CMA-ES).
    λ = 4 + floor(3·ln(d)), σ₀ = 0.25, population evaluated in async batches.
    The evaluation budget is ``max_episodes`` candidate simulations — the same
    ceiling used by Direct PPO and Random Search — not ``max_episodes``
    generations.  A truncated final generation is evaluated but does not
    update the covariance.

    Supports adaptive sigma adjustment via reflection agent feedback:
    - If reflection suggests "exploration", increase sigma
    - If reflection suggests "exploitation", decrease sigma
    """

    def __init__(self, sigma0: float = 0.25, seed: int = 42) -> None:
        self.sigma0 = sigma0
        self.seed = seed

    def _parse_reflection_hint(self, reflection: Dict[str, Any]) -> Optional[str]:
        """Parse reflection suggestion for CMA-ES hyperparameter hints.

        Returns:
            'explore' - increase sigma for broader search
            'exploit' - decrease sigma for fine-tuning
            None - no adjustment
        """
        suggestion = str(reflection.get("suggestion", "")).lower()

        # Keywords suggesting need for exploration (increase sigma)
        explore_signals = [
            "探索", "扩大搜索", "增加搜索范围", "多样性", "跳出局部",
            "explore", "diversify", "broaden search", "escape local"
        ]

        # Keywords suggesting need for exploitation (decrease sigma)
        exploit_signals = [
            "精调", "微调", "收敛", "细化", "局部优化",
            "exploit", "fine-tune", "converge", "refine", "local search"
        ]

        if any(sig in suggestion for sig in explore_signals):
            return 'explore'
        elif any(sig in suggestion for sig in exploit_signals):
            return 'exploit'
        return None

    async def optimize(
        self,
        *,
        rl_optimizer,
        script_path: str,
        mission_conditions: Optional[Dict],
        max_episodes: int,
        nmc: int,
        task_prompt: str,
        initial_auto_params: Optional[Dict[str, float]] = None,
        reflection_agent=None,
        **kwargs,
    ) -> Dict[str, Any]:
        specs = _get_param_specs()
        d = len(specs)
        lam = 4 + int(math.floor(3.0 * math.log(d)))
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights ** 2)

        cc = (4.0 + mueff / d) / (d + 4.0 + 2.0 * mueff / d)
        cs = (mueff + 2.0) / (d + mueff + 5.0)
        c1 = 2.0 / ((d + 1.3) ** 2 + mueff)
        cmu = min(1.0 - c1,
                  2.0 * (mueff - 2.0 + 1.0 / mueff) / ((d + 2.0) ** 2 + mueff))
        damps = 1.0 + 2.0 * max(0.0, math.sqrt((mueff - 1.0) / (d + 1.0)) - 1.0) + cs
        chi_n = math.sqrt(d) * (1.0 - 1.0 / (4.0 * d) + 1.0 / (21.0 * d * d))

        rng = np.random.default_rng(self.seed)
        mean = _params_to_norm(initial_auto_params or {})
        sigma = self.sigma0
        pc = np.zeros(d)
        ps = np.zeros(d)
        cov = np.eye(d)
        bmat, dvec = np.eye(d), np.ones(d)
        invsqrt = np.eye(d)
        eigeneval = 0

        evaluator = _SimEvaluator(rl_optimizer, script_path, mission_conditions, nmc)
        best_score = float("inf")
        best_params = dict(initial_auto_params or {})
        best_metrics: Dict[str, float] = {}
        history: List[Dict] = []
        total_ep = 0
        first_success: Optional[int] = None
        last_reflection: Optional[Dict[str, Any]] = None
        reflect_every = 5  # Check every 5 generations

        generation = 0
        while total_ep < max_episodes:
            generation += 1
            # Sample λ offspring
            arz = rng.standard_normal((lam, d))
            ary = arz @ (bmat * dvec).T
            candidates_norm = np.clip(mean + sigma * ary, 0.0, 1.0)
            candidates = [_norm_to_params(candidates_norm[k]) for k in range(lam)]

            # Evaluate asynchronously
            tasks = [asyncio.ensure_future(evaluator(c)) for c in candidates]
            metrics_list = await asyncio.gather(*tasks, return_exceptions=True)

            fitnesses = []
            new_best = False
            for k, m_or_exc in enumerate(metrics_list):
                total_ep += 1
                if isinstance(m_or_exc, Exception):
                    fitnesses.append(float("inf"))
                    continue
                m = m_or_exc
                ok = evaluator._feasible(m)
                sc = evaluator._score(m, ok)
                fitnesses.append(sc)
                if ok and first_success is None:
                    first_success = generation
                if sc < best_score:
                    best_score = sc
                    best_params = _clip(candidates[k])
                    best_metrics = dict(m)
                    new_best = True
                history.append({"generation": generation, "candidate": k,
                                 "score": sc, "all_pass": ok, "metrics": m})
                if total_ep >= max_episodes:
                    break

            # Reflection check on new best or periodic check
            if reflection_agent and task_prompt and new_best and (generation % reflect_every == 0):
                try:
                    reflection = await reflection_agent.reflect(
                        task_prompt=task_prompt,
                        best_metrics=best_metrics,
                        best_params=best_params,
                    )
                    last_reflection = reflection
                    needs_opt = reflection.get("needs_optimization", True)
                    suggestion = reflection.get("suggestion", "")

                    # Adaptive sigma adjustment based on reflection hint
                    hint = self._parse_reflection_hint(reflection)
                    if hint == 'explore':
                        old_sigma = sigma
                        sigma = min(sigma * 1.5, 0.5)  # Increase sigma, cap at 0.5
                        logger.info(
                            "[CMA-ES] Reflection suggests exploration → "
                            f"sigma: {old_sigma:.4f} → {sigma:.4f}"
                        )
                    elif hint == 'exploit':
                        old_sigma = sigma
                        sigma = max(sigma * 0.7, 1e-4)  # Decrease sigma, floor at 1e-4
                        logger.info(
                            "[CMA-ES] Reflection suggests exploitation → "
                            f"sigma: {old_sigma:.4f} → {sigma:.4f}"
                        )

                    # Check for design-path change signals (same as PPO)
                    _MODIFY_SIGNALS = [
                        "制导律", "算法结构", "modify_law", "修改制导律", "新的制导",
                        "重新设计", "滑模", "预测制导", "guidance law", "algorithm",
                    ]
                    if any(sig in suggestion.lower() for sig in _MODIFY_SIGNALS):
                        logger.info(
                            f"[CMA-ES] Reflection suggests design-path change at "
                            f"generation {generation}; breaking to let Hermes re-plan. "
                            f"Hint: {suggestion[:200]}"
                        )
                        return {
                            "status": "needs_design_path_change",
                            "best_params": best_params,
                            "best_metrics": best_metrics,
                            "best_reward": float(-best_score),
                            "baseline_reward": 0.0,
                            "total_episodes": total_ep,
                            "history": history,
                            "reflection": last_reflection,
                            "mc_calls": evaluator.call_count,
                            "sim_time_sec": evaluator.sim_time_sec,
                            "first_success_generation": first_success,
                        }

                    if not needs_opt:
                        logger.info(
                            "[CMA-ES] Reflection early-exit at generation %d: task_requirements_met",
                            generation
                        )
                        return {
                            "status": "task_requirements_met",
                            "best_params": best_params,
                            "best_metrics": best_metrics,
                            "best_reward": float(-best_score),
                            "baseline_reward": 0.0,
                            "total_episodes": total_ep,
                            "history": history,
                            "reflection": last_reflection,
                            "mc_calls": evaluator.call_count,
                            "sim_time_sec": evaluator.sim_time_sec,
                            "first_success_generation": first_success,
                        }
                except Exception as exc:
                    logger.warning("[CMA-ES] Reflection failed: %s", exc)

            fitnesses = np.array(fitnesses, dtype=float)
            # Rank-based selection
            ranked_idx = np.argsort(fitnesses)[:mu]
            xold = mean.copy()
            mean = weights @ candidates_norm[ranked_idx]
            mean = np.clip(mean, 0.0, 1.0)

            # Cumulation
            ps = (1 - cs) * ps + math.sqrt(cs * (2 - cs) * mueff) * invsqrt @ (mean - xold) / sigma
            hsig = float(np.linalg.norm(ps) / math.sqrt(1 - (1 - cs) ** (2 * (total_ep / lam))) / chi_n < 1.4 + 2.0 / (d + 1.0))
            pc = (1 - cc) * pc + hsig * math.sqrt(cc * (2 - cc) * mueff) * (mean - xold) / sigma

            artmp = (1 / sigma) * (candidates_norm[ranked_idx] - xold)
            cov = ((1 - c1 - cmu) * cov
                   + c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * cov)
                   + cmu * artmp.T @ np.diag(weights) @ artmp)
            sigma *= math.exp((cs / damps) * (np.linalg.norm(ps) / chi_n - 1.0))
            sigma = float(np.clip(sigma, 1e-6, 1.0))

            # Eigendecomposition every ~d/10 evaluations
            if total_ep - eigeneval > lam / (c1 + cmu) / d / 10:
                eigeneval = total_ep
                cov = np.triu(cov) + np.triu(cov, 1).T
                dvec2, bmat = np.linalg.eigh(cov)
                dvec = np.sqrt(np.maximum(dvec2, 1e-20))
                invsqrt = bmat @ np.diag(1.0 / dvec) @ bmat.T

        logger.info("[CMA-ES] generations=%d  total_ep=%d  best_score=%.4f",
                    generation, total_ep, best_score)
        return {
            "status": "task_requirements_met" if best_score < 0 else "success",
            "best_params": best_params,
            "best_metrics": best_metrics,
            "best_reward": float(-best_score),
            "baseline_reward": 0.0,
            "total_episodes": total_ep,
            "history": history,
            "reflection": last_reflection,
            "mc_calls": evaluator.call_count,
            "sim_time_sec": evaluator.sim_time_sec,
            "first_success_generation": first_success,
        }


# ── Random Search backend ──────────────────────────────────────────────────────

class RandomSearchBackend(OptimizerBackend):
    """Uniform random search — performance lower bound for 4.2.1.

    Supports adaptive search space adjustment via reflection agent feedback:
    - If reflection suggests focusing on certain parameters, adjust search distribution
    - Can switch between uniform and focused sampling based on hints
    """

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed

    def _parse_reflection_hint(self, reflection: Dict[str, Any]) -> Dict[str, Any]:
        """Parse reflection suggestion for random search strategy hints.

        Returns:
            dict with 'mode': 'uniform' or 'focused', and optional 'focus_params'
        """
        suggestion = str(reflection.get("suggestion", "")).lower()

        # Keywords suggesting focused search around current best
        focus_signals = [
            "局部搜索", "当前参数附近", "微调", "精细搜索", "邻域",
            "local search", "near current", "fine-tune", "neighborhood"
        ]

        # Keywords suggesting broader uniform search
        uniform_signals = [
            "全局搜索", "重新探索", "扩大范围", "多样性",
            "global search", "re-explore", "broader", "diversity"
        ]

        if any(sig in suggestion for sig in focus_signals):
            return {'mode': 'focused', 'radius': 0.2}  # Search within 20% radius
        elif any(sig in suggestion for sig in uniform_signals):
            return {'mode': 'uniform'}
        return {'mode': 'uniform'}

    async def optimize(
        self,
        *,
        rl_optimizer,
        script_path: str,
        mission_conditions: Optional[Dict],
        max_episodes: int,
        nmc: int,
        task_prompt: str,
        initial_auto_params: Optional[Dict[str, float]] = None,
        reflection_agent=None,
        **kwargs,
    ) -> Dict[str, Any]:
        specs = _get_param_specs()
        d = len(specs)
        rng = np.random.default_rng(self.seed)
        evaluator = _SimEvaluator(rl_optimizer, script_path, mission_conditions, nmc)
        best_score = float("inf")
        best_params = dict(initial_auto_params or {})
        best_metrics: Dict[str, float] = {}
        history: List[Dict] = []
        first_success: Optional[int] = None
        last_reflection: Optional[Dict[str, Any]] = None
        reflect_every = 10  # Check every 10 episodes
        search_mode = 'uniform'  # 'uniform' or 'focused'
        focus_radius = 0.2  # Radius for focused search

        for ep in range(1, max_episodes + 1):
            # Adaptive sampling based on reflection hint
            if search_mode == 'focused' and best_params:
                # Sample around current best in normalized space
                best_norm = _params_to_norm(best_params)
                x = np.clip(
                    best_norm + rng.normal(0.0, focus_radius, size=d),
                    0.0, 1.0
                )
            else:
                # Uniform random sampling
                x = rng.uniform(0.0, 1.0, size=d)

            params = _clip(_norm_to_params(x))
            m = await evaluator(params)
            ok = evaluator._feasible(m)
            sc = evaluator._score(m, ok)
            new_best = False
            if ok and first_success is None:
                first_success = ep
            if sc < best_score:
                best_score = sc
                best_params = params
                best_metrics = dict(m)
                new_best = True
            history.append({"ep": ep, "score": sc, "all_pass": ok, "metrics": m})

            # Reflection check on new best or periodic check
            if reflection_agent and task_prompt and new_best and (ep % reflect_every == 0):
                try:
                    reflection = await reflection_agent.reflect(
                        task_prompt=task_prompt,
                        best_metrics=best_metrics,
                        best_params=best_params,
                    )
                    last_reflection = reflection
                    needs_opt = reflection.get("needs_optimization", True)
                    suggestion = reflection.get("suggestion", "")

                    # Adaptive search mode based on reflection hint
                    hint = self._parse_reflection_hint(reflection)
                    old_mode = search_mode
                    search_mode = hint.get('mode', 'uniform')
                    if 'radius' in hint:
                        focus_radius = hint['radius']
                    if old_mode != search_mode:
                        logger.info(
                            f"[RandomSearch] Reflection switches mode: {old_mode} → {search_mode}"
                            + (f" (radius={focus_radius:.2f})" if search_mode == 'focused' else "")
                        )

                    # Check for design-path change signals (same as PPO)
                    _MODIFY_SIGNALS = [
                        "制导律", "算法结构", "modify_law", "修改制导律", "新的制导",
                        "重新设计", "滑模", "预测制导", "guidance law", "algorithm",
                    ]
                    if any(sig in suggestion.lower() for sig in _MODIFY_SIGNALS):
                        logger.info(
                            f"[RandomSearch] Reflection suggests design-path change at "
                            f"episode {ep}; breaking to let Hermes re-plan. "
                            f"Hint: {suggestion[:200]}"
                        )
                        return {
                            "status": "needs_design_path_change",
                            "best_params": best_params,
                            "best_metrics": best_metrics,
                            "best_reward": float(-best_score),
                            "baseline_reward": 0.0,
                            "total_episodes": ep,
                            "history": history,
                            "reflection": last_reflection,
                            "mc_calls": evaluator.call_count,
                            "sim_time_sec": evaluator.sim_time_sec,
                            "first_success_ep": first_success,
                        }

                    if not needs_opt:
                        logger.info(
                            "[RandomSearch] Reflection early-exit at episode %d: task_requirements_met",
                            ep
                        )
                        return {
                            "status": "task_requirements_met",
                            "best_params": best_params,
                            "best_metrics": best_metrics,
                            "best_reward": float(-best_score),
                            "baseline_reward": 0.0,
                            "total_episodes": ep,
                            "history": history,
                            "reflection": last_reflection,
                            "mc_calls": evaluator.call_count,
                            "sim_time_sec": evaluator.sim_time_sec,
                            "first_success_ep": first_success,
                        }
                except Exception as exc:
                    logger.warning("[RandomSearch] Reflection failed: %s", exc)

        logger.info("[RandomSearch] total_ep=%d  best_score=%.4f", max_episodes, best_score)
        return {
            "status": "task_requirements_met" if best_score < 0 else "success",
            "best_params": best_params,
            "best_metrics": best_metrics,
            "best_reward": float(-best_score),
            "baseline_reward": 0.0,
            "total_episodes": max_episodes,
            "history": history,
            "reflection": last_reflection,
            "mc_calls": evaluator.call_count,
            "sim_time_sec": evaluator.sim_time_sec,
            "first_success_ep": first_success,
        }


# ── Registry / factory ─────────────────────────────────────────────────────────

_REGISTRY: Dict[str, type] = {
    "hybrid_ppo":    HybridPPOBackend,
    "direct_ppo":    DirectPPOBackend,
    "cma_es":        CMAESBackend,
    "random_search": RandomSearchBackend,
}


def build_optimizer_backend(name: str) -> OptimizerBackend:
    """Factory: return an OptimizerBackend instance for *name*.

    ``name`` maps to ``matlab_rl_optimizer.optimizer_backend`` in config.yaml.
    Raises ``KeyError`` on unknown backend name.
    """
    cls = _REGISTRY.get(str(name).strip().lower())
    if cls is None:
        raise KeyError(
            f"Unknown optimizer_backend '{name}'. "
            f"Valid options: {sorted(_REGISTRY)}"
        )
    return cls()


def get_backend_from_config() -> OptimizerBackend:
    """Read ``matlab_rl_optimizer.optimizer_backend`` from config and build."""
    try:
        from multi_agent.config_loader import get_config
        name = getattr(get_config().matlab_rl_optimizer, "optimizer_backend", "hybrid_ppo")
    except Exception:
        name = "hybrid_ppo"
    return build_optimizer_backend(name or "hybrid_ppo")
