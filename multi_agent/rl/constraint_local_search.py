"""Constraint-aware local search — refines Expert/PPO best in parameter neighbourhood."""

from __future__ import annotations

import copy
import logging
import math
import random
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

import numpy as np

from multi_agent.integration.design_path_policy import metrics_summary_line
from multi_agent.logging.log_verbosity import is_verbose, should_log_cls_step
from multi_agent.rl.episode_log import EpisodeJsonlLogger
from multi_agent.rl.matlab_rl_optimizer import (
    ALL_TUNABLE_PARAM_SPECS,
    MatlabRLOptimizer,
)
from multi_agent.rl.metric_utils import get_peak_ny
from multi_agent.rl.metric_constraints import (
    MetricLockMargins,
    constraints_satisfied,
    count_satisfied_flags,
    eligible_partial_bw_band,
    failed_metric_keys,
    flags_imply_lock,
    is_better_borderline_peak,
    is_better_constrained,
    is_better_partial_progress,
    is_borderline_hit_sep_ok,
    metric_satisfaction_flags,
    params_for_failed_metrics,
    resolve_requirements,
    violates_locked_metrics,
)

logger = logging.getLogger(__name__)


def _cls_skip_borderline_for_full_task(task_prompt: str, reqs: Dict[str, Any]) -> bool:
    """Skip hit/SEP-only borderline pick when prompt requires all five metrics."""
    if not task_prompt:
        return False
    try:
        from multi_agent.integration.design_path_policy import is_optimization_task
        from multi_agent.integration.judgment_agent import requirements_complete

        return is_optimization_task(task_prompt) and requirements_complete(reqs, task_prompt)
    except Exception:
        return False


class ConstraintLocalSearch:
    """Directed random walk with no-regression gate on satisfied metrics."""

    def __init__(
        self,
        rl_backend: MatlabRLOptimizer,
        max_iterations: int = 50,
        step_scale: float = 0.08,
        seed: Optional[int] = None,
        *,
        no_regression_gate: bool = True,
        directed_perturb: bool = True,
        lock_margins: Optional[MetricLockMargins] = None,
    ):
        self.rl_backend = rl_backend
        self.max_iterations = max(1, int(max_iterations))
        self.step_scale = float(step_scale)
        self._rng = random.Random(seed)
        self.no_regression_gate = bool(no_regression_gate)
        self.directed_perturb = bool(directed_perturb)
        self.lock_margins = lock_margins or MetricLockMargins()

    def _perturb_keys(
        self,
        centre: Dict[str, float],
        keys: Tuple[str, ...],
        scale: float,
    ) -> Dict[str, float]:
        out = copy.deepcopy(centre)
        for k in keys:
            spec = ALL_TUNABLE_PARAM_SPECS.get(k)
            if not spec:
                continue
            lo, hi = spec["min"], spec["max"]
            span = hi - lo
            delta = self._rng.uniform(-scale, scale) * span
            out[k] = float(np.clip(out.get(k, spec["nominal"]) + delta, lo, hi))
        return out

    def _perturb(self, centre: Dict[str, float], scale: Optional[float] = None) -> Dict[str, float]:
        sc = self.step_scale if scale is None else scale
        return self._perturb_keys(
            centre, tuple(ALL_TUNABLE_PARAM_SPECS.keys()), sc,
        )

    @staticmethod
    def _clamp_t4_candidate(
        params: Dict[str, float],
        *,
        task_prompt: Optional[str],
        mission_conditions: Optional[Any],
    ) -> Dict[str, float]:
        try:
            from multi_agent.integration.t4_low_risk import maybe_clamp_t4_params

            return maybe_clamp_t4_params(
                params, task_prompt=task_prompt, mission_conditions=mission_conditions,
            )
        except Exception:
            return params

    def _merge_locked(
        self,
        locked: Set[str],
        flags: Dict[str, bool],
        metrics: Dict[str, float],
        reqs: Dict[str, Any],
    ) -> None:
        for key in flags_imply_lock(flags, metrics, reqs, self.lock_margins):
            locked.add(key)

    async def refine(
        self,
        center_params: Dict[str, float],
        *,
        script_path: Optional[str],
        mission_conditions: Optional[Dict[str, List[int]]],
        nmc: int,
        task_prompt: Optional[str] = None,
        metric_requirements: Optional[Dict[str, Any]] = None,
        judgment_agent: Any = None,
        episode_logger: Optional[EpisodeJsonlLogger] = None,
        reference_params: Optional[Dict[str, float]] = None,
        reference_metrics: Optional[Dict[str, float]] = None,
        borderline_select: bool = True,
        borderline_hit_min_pct: float = 92.0,
        borderline_sep_max_m: float = 7.0,
        prefer_reference_peak: bool = True,
        nmc_fast: Optional[int] = None,
        nmc_validate: Optional[int] = None,
        focus_metric_keys: Optional[Tuple[str, ...]] = None,
        step_scale_override: Optional[float] = None,
    ) -> Dict[str, Any]:
        reqs = resolve_requirements(metric_requirements, task_prompt, judgment_agent)
        _partial_bw_gate = True
        try:
            from multi_agent.config_loader import get_config

            _partial_bw_gate = bool(
                getattr(
                    get_config().matlab_rl_optimizer.constraint_local_search,
                    "partial_bw_band_gate",
                    True,
                )
            )
        except Exception:
            pass
        centre = copy.deepcopy(center_params)
        eval_nmc = max(1, int(nmc_fast if nmc_fast is not None else nmc))
        validate_nmc = max(eval_nmc, int(nmc_validate if nmc_validate is not None else nmc))

        if script_path and self.rl_backend:
            self.rl_backend.extract_params_from_script(script_path)

        best_params = copy.deepcopy(centre)
        best_metrics: Dict[str, float] = {}
        best_fitness = -1.0
        best_partial_params: Optional[Dict[str, float]] = None
        best_partial_metrics: Optional[Dict[str, float]] = None
        best_partial_fitness = -1.0
        best_partial_count = -1

        best_constrained_params: Optional[Dict[str, float]] = None
        best_constrained_metrics: Optional[Dict[str, float]] = None
        best_constrained_fitness = -1.0
        best_borderline_params: Optional[Dict[str, float]] = None
        best_borderline_metrics: Optional[Dict[str, float]] = None

        locked_metrics: Set[str] = set()
        working_params = copy.deepcopy(centre)

        if (
            prefer_reference_peak
            and reference_metrics
            and is_borderline_hit_sep_ok(
                reference_metrics,
                hit_min_pct=borderline_hit_min_pct,
                sep_max_m=borderline_sep_max_m,
            )
        ):
            best_borderline_params = copy.deepcopy(reference_params or centre)
            best_borderline_metrics = copy.deepcopy(reference_metrics)
            if reference_params:
                best_params = copy.deepcopy(reference_params)
                best_metrics = copy.deepcopy(reference_metrics)
                best_fitness = self.rl_backend._compute_pe_fitness(reference_metrics)
                working_params = copy.deepcopy(reference_params)

        center_metrics = await self.rl_backend._run_simulation_with_params(
            centre, mission_conditions, script_path, eval_nmc,
        )
        center_reward = self.rl_backend._compute_reward(center_metrics)
        center_fitness = self.rl_backend._compute_pe_fitness(center_metrics)
        center_flags = metric_satisfaction_flags(center_metrics, reqs)

        if not best_metrics:
            best_metrics = copy.deepcopy(center_metrics)
            best_fitness = center_fitness
        elif get_peak_ny(center_metrics) < get_peak_ny(best_metrics):
            best_params = copy.deepcopy(centre)
            best_metrics = copy.deepcopy(center_metrics)
            best_fitness = center_fitness

        if (
            eligible_partial_bw_band(
                center_metrics,
                center_flags,
                reqs,
                focus_metric_keys=focus_metric_keys,
                enabled=_partial_bw_gate,
            )
            and is_better_partial_progress(
                center_metrics, center_flags, center_fitness,
                best_partial_metrics, metric_satisfaction_flags(best_partial_metrics or {}, reqs)
                if best_partial_metrics else None,
                best_partial_fitness,
                reqs=reqs,
            )
        ):
            best_partial_params = copy.deepcopy(centre)
            best_partial_metrics = copy.deepcopy(center_metrics)
            best_partial_fitness = center_fitness
            best_partial_count = count_satisfied_flags(center_flags)
            working_params = copy.deepcopy(centre)

        self._merge_locked(locked_metrics, center_flags, center_metrics, reqs)

        if constraints_satisfied(center_metrics, reqs):
            best_constrained_params = copy.deepcopy(centre)
            best_constrained_metrics = copy.deepcopy(center_metrics)
            best_constrained_fitness = center_fitness
        if borderline_select and is_borderline_hit_sep_ok(
            center_metrics,
            hit_min_pct=borderline_hit_min_pct,
            sep_max_m=borderline_sep_max_m,
        ):
            if is_better_borderline_peak(center_metrics, best_borderline_metrics):
                best_borderline_params = copy.deepcopy(centre)
                best_borderline_metrics = copy.deepcopy(center_metrics)

        if episode_logger:
            episode_logger.write_episode(
                optimizer="ConstraintLocalSearch",
                episode=0,
                params=centre,
                metrics=center_metrics,
                reward=center_reward,
                fitness=center_fitness,
                constrained=constraints_satisfied(center_metrics, reqs),
                extra={"phase": "center", "nmc": eval_nmc},
            )

        history: List[Dict[str, Any]] = []
        iterations_run = 0

        if focus_metric_keys:
            _cf = metric_satisfaction_flags(center_metrics, reqs)
            for _mk, _ok in _cf.items():
                if _ok and _mk not in focus_metric_keys:
                    locked_metrics.add(_mk)
            logger.info(
                "[CLS] Focus mode metrics=%s | locked=%s",
                focus_metric_keys,
                sorted(locked_metrics),
            )

        if constraints_satisfied(center_metrics, reqs):
            logger.info(
                "[CLS] All constraints satisfied at center — stopping early "
                f"(PeakNy_avg={get_peak_ny(center_metrics):.2f}g)"
            )
        else:
            from multi_agent.simulation.time_budget import get_run_budget

            _loop_budget = get_run_budget()
            for it in range(1, self.max_iterations + 1):
                if _loop_budget and _loop_budget.should_stop(reserve_sec=90.0):
                    logger.warning(
                        "[CLS] Wall-clock budget low — stopping at %d/%d",
                        it - 1,
                        self.max_iterations,
                    )
                    break
                iterations_run = it
                working_flags = metric_satisfaction_flags(
                    best_partial_metrics or center_metrics, reqs,
                )
                failed = failed_metric_keys(working_flags)
                if focus_metric_keys:
                    failed = focus_metric_keys
                n_locked = len(locked_metrics)
                scale = float(step_scale_override) if step_scale_override is not None else (
                    self.step_scale * max(0.35, 1.0 - 0.12 * n_locked)
                )

                if self.directed_perturb and failed:
                    pkeys = params_for_failed_metrics(failed)
                    candidate = self._perturb_keys(working_params, pkeys, scale)
                elif (
                    get_peak_ny(best_borderline_metrics or best_metrics or {}) > 0
                    and get_peak_ny(best_borderline_metrics or best_metrics or {})
                    <= float(getattr(self.rl_backend, "_peak_n_max", 20.0))
                ):
                    candidate = self._perturb_keys(
                        working_params, tuple(ALL_TUNABLE_PARAM_SPECS.keys()),
                        scale * 0.5,
                    )
                else:
                    candidate = self._perturb(working_params, scale)

                candidate = self._clamp_t4_candidate(
                    candidate, task_prompt=task_prompt, mission_conditions=mission_conditions,
                )

                metrics = await self.rl_backend._run_simulation_with_params(
                    candidate, mission_conditions, script_path, eval_nmc,
                )
                reward = self.rl_backend._compute_reward(metrics)
                fitness = self.rl_backend._compute_pe_fitness(metrics)
                cand_flags = metric_satisfaction_flags(metrics, reqs)
                constrained = constraints_satisfied(metrics, reqs)
                _improved = False
                _rejected_gate = False

                history.append(
                    {
                        "iteration": it,
                        "reward": reward,
                        "fitness": fitness,
                        "metrics": copy.deepcopy(metrics),
                        "params": copy.deepcopy(candidate),
                        "constraints_met": constrained,
                        "flags": cand_flags,
                        "locked": sorted(locked_metrics),
                    }
                )

                if episode_logger:
                    episode_logger.write_episode(
                        optimizer="ConstraintLocalSearch",
                        episode=it,
                        params=candidate,
                        metrics=metrics,
                        reward=reward,
                        fitness=fitness,
                        constrained=constrained,
                        extra={"nmc": eval_nmc, "locked": sorted(locked_metrics)},
                    )

                sep = metrics.get("SEP", metrics.get("miss_distance", 999.0))
                if isinstance(sep, float) and (math.isnan(sep) or math.isinf(sep)):
                    continue

                if (
                    self.no_regression_gate
                    and locked_metrics
                    and violates_locked_metrics(cand_flags, frozenset(locked_metrics))
                ):
                    _rejected_gate = True
                    logger.debug(
                        f"  [CLS {it}] no-regression reject locked={sorted(locked_metrics)} "
                        f"{metrics_summary_line(metrics)}"
                    )
                else:
                    if constrained and is_better_constrained(
                        metrics, fitness,
                        best_constrained_metrics,
                        best_constrained_fitness,
                    ):
                        best_constrained_params = copy.deepcopy(candidate)
                        best_constrained_metrics = copy.deepcopy(metrics)
                        best_constrained_fitness = fitness
                        working_params = copy.deepcopy(candidate)
                        _improved = True

                    if borderline_select and is_borderline_hit_sep_ok(
                        metrics,
                        hit_min_pct=borderline_hit_min_pct,
                        sep_max_m=borderline_sep_max_m,
                    ):
                        if is_better_borderline_peak(metrics, best_borderline_metrics):
                            best_borderline_params = copy.deepcopy(candidate)
                            best_borderline_metrics = copy.deepcopy(metrics)
                            _improved = True

                    partial_flags_b = (
                        metric_satisfaction_flags(best_partial_metrics, reqs)
                        if best_partial_metrics
                        else None
                    )
                    if (
                        eligible_partial_bw_band(
                            metrics,
                            cand_flags,
                            reqs,
                            focus_metric_keys=focus_metric_keys,
                            enabled=_partial_bw_gate,
                        )
                        and is_better_partial_progress(
                            metrics, cand_flags, fitness,
                            best_partial_metrics, partial_flags_b,
                            best_partial_fitness,
                            reqs=reqs,
                        )
                    ):
                        best_partial_params = copy.deepcopy(candidate)
                        best_partial_metrics = copy.deepcopy(metrics)
                        best_partial_fitness = fitness
                        best_partial_count = count_satisfied_flags(cand_flags)
                        working_params = copy.deepcopy(candidate)
                        _improved = True
                    elif (
                        _partial_bw_gate
                        and is_better_partial_progress(
                            metrics, cand_flags, fitness,
                            best_partial_metrics, partial_flags_b,
                            best_partial_fitness,
                            reqs=reqs,
                        )
                        and not eligible_partial_bw_band(
                            metrics,
                            cand_flags,
                            reqs,
                            focus_metric_keys=focus_metric_keys,
                            enabled=True,
                        )
                    ):
                        logger.debug(
                            "  [CLS %d] partial BW band reject %s",
                            it,
                            metrics_summary_line(metrics),
                        )

                    pny = get_peak_ny(metrics)
                    if (
                        not self.no_regression_gate
                        and (
                            fitness > best_fitness
                            or (
                                pny > 0
                                and get_peak_ny(best_metrics) > 0
                                and pny < get_peak_ny(best_metrics)
                                and metrics.get("hit_rate", 0)
                                >= best_metrics.get("hit_rate", 0) - 5
                            )
                        )
                    ):
                        best_params = copy.deepcopy(candidate)
                        best_metrics = copy.deepcopy(metrics)
                        best_fitness = fitness
                        _improved = True

                    self._merge_locked(locked_metrics, cand_flags, metrics, reqs)

                _summary = metrics_summary_line(metrics)
                if should_log_cls_step(it, self.max_iterations, improved=_improved):
                    _suffix = ""
                    if _rejected_gate:
                        _suffix = " [gate]"
                    elif _improved:
                        _suffix = " ★"
                    logger.info(
                        f"  [CLS {it}/{self.max_iterations}] {_summary}"
                        f"{_suffix} locked={sorted(locked_metrics)}"
                        if is_verbose()
                        else f"  [CLS {it}/{self.max_iterations}] {_summary}{_suffix}"
                    )
                else:
                    logger.debug(f"  [CLS {it}/{self.max_iterations}] {_summary}")

                if constrained:
                    logger.info(
                        f"[CLS] All constraints satisfied at step {it} — stopping early "
                        f"(PeakNy_avg={get_peak_ny(metrics):.2f}g)"
                    )
                    break

        selection = "partial"
        final_params = copy.deepcopy(
            best_partial_params or best_constrained_params or best_params,
        )
        final_metrics = copy.deepcopy(
            best_partial_metrics or best_constrained_metrics or best_metrics,
        )

        if best_constrained_params and best_constrained_metrics is not None:
            final_params = copy.deepcopy(best_constrained_params)
            final_metrics = copy.deepcopy(best_constrained_metrics)
            selection = "constrained"
        elif (
            best_partial_params
            and best_partial_metrics is not None
            and best_partial_count > 0
        ):
            final_params = copy.deepcopy(best_partial_params)
            final_metrics = copy.deepcopy(best_partial_metrics)
            selection = "partial"
        elif (
            borderline_select
            and best_borderline_params
            and best_borderline_metrics is not None
            and is_better_borderline_peak(best_borderline_metrics, final_metrics)
            and not _cls_skip_borderline_for_full_task(task_prompt or "", reqs)
        ):
            final_params = copy.deepcopy(best_borderline_params)
            final_metrics = copy.deepcopy(best_borderline_metrics)
            selection = "borderline"
        else:
            selection = "center" if final_params == centre else "best"

        if (
            validate_nmc > eval_nmc
            and final_params
            and not constraints_satisfied(final_metrics, reqs)
        ):
            validated = await self.rl_backend._run_simulation_with_params(
                final_params, mission_conditions, script_path, validate_nmc,
            )
            if constraints_satisfied(validated, reqs):
                final_metrics = validated
                selection = f"{selection}+validate"
            elif is_better_partial_progress(
                validated,
                metric_satisfaction_flags(validated, reqs),
                self.rl_backend._compute_pe_fitness(validated),
                final_metrics,
                metric_satisfaction_flags(final_metrics, reqs),
                best_partial_fitness,
                reqs=reqs,
            ):
                final_metrics = validated

        if (
            best_constrained_metrics
            and constraints_satisfied(final_metrics, reqs)
            and not constraints_satisfied(best_constrained_metrics, reqs)
        ):
            pass
        elif (
            best_constrained_params
            and best_constrained_metrics is not None
            and count_satisfied_flags(metric_satisfaction_flags(final_metrics, reqs))
            < count_satisfied_flags(
                metric_satisfaction_flags(best_constrained_metrics, reqs),
            )
        ):
            final_params = copy.deepcopy(best_constrained_params)
            final_metrics = copy.deepcopy(best_constrained_metrics)
            selection = "constrained"

        logger.info(
            f"[CLS] Done {iterations_run} iterations | selection={selection} | "
            f"satisfied={count_satisfied_flags(metric_satisfaction_flags(final_metrics, reqs))}/"
            f"{len(metric_satisfaction_flags(final_metrics, reqs))} | "
            f"PeakNy_avg={get_peak_ny(final_metrics):.2f}g "
            f"SEP={final_metrics.get('SEP', final_metrics.get('miss_distance', 0)):.2f}m"
        )

        return {
            "status": "success",
            "best_params": final_params,
            "best_metrics": final_metrics,
            "best_fitness": (
                best_constrained_fitness
                if selection.startswith("constrained")
                else best_partial_fitness
                if selection == "partial"
                else best_fitness
            ),
            "best_constrained_params": best_constrained_params or {},
            "best_constrained_metrics": best_constrained_metrics or {},
            "selection_criteria": selection,
            "total_iterations": iterations_run,
            "locked_metrics": sorted(locked_metrics),
            "history": history[-10:],
        }
