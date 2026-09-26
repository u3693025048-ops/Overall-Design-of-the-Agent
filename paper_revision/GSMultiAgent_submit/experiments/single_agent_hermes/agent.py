#!/usr/bin/env python3
"""
SingleAgentHermes — single-agent LLM-backed baseline for GSMultiAgent Section 4.3.

Components:
  - ParameterExperience   (经验库 warm-start + write-back)
  - MATLAB harness        (run_mc_and_across_seeds, search 10 / test 100)
  - PPO optimizer         (NumpyCMAES-style PPO, same as direct_ppo)
  - Correction pipeline   (parameter clipping, sanitizer in harness)
  - LLM advisor           (HermesIntegration or LangChain fallback)
    · Post-iteration reflection: analyse metrics, advise warm-start bias
      and RL hyper-parameter adjustments for the next outer loop
    · Early-stop signal when all constraints are met

LLM cascade (added back from ablation baseline):
  - _HermesLLMAdvisor     — wraps HermesIntegration (primary) or
                            LangChain ChatOpenAI (fallback)
  - Graceful no-op when neither backend is importable

Architecture
------------
Outer loop (n_outer_iterations):
    1. Query PE for best warm-start params for current scenario
    2. Run PPO episodes from warm-start (same NumpyCMAES PPO as direct_ppo)
    3. Write best_params + metrics back to PE
    4. [NEW] Call LLM advisor → parse advice → optionally apply param hints
             and RL adjustments, optionally early-stop
Final:
    Evaluate global best_params on test_seeds (AND over all seeds).
"""
from __future__ import annotations

import logging
import math
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from multi_agent.rl.matlab_rl_optimizer import ALL_TUNABLE_PARAM_SPECS
from multi_agent.tools.simulation_tool import check_constraints, run_mc_and_across_seeds

logger = logging.getLogger(__name__)


def _ensure_console_handler(log: logging.Logger) -> None:
    """Attach a stdout StreamHandler when no handler is configured.

    Called once at import time.  If the root logger or this logger already
    has handlers (e.g. the caller configured logging before importing this
    module) we leave them untouched to avoid duplicate output.
    """
    target = log
    while target:
        if target.handlers:
            return
        if target is target.root:
            break
        target = target.parent  # type: ignore[assignment]

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter("%(asctime)s [SAH] %(levelname)s %(message)s",
                          datefmt="%H:%M:%S")
    )
    log.setLevel(logging.INFO)
    log.addHandler(handler)
    log.propagate = False


_ensure_console_handler(logger)

# ── Parameter space (shared with optimizer_baselines_43) ─────────────────────
PARAM_NAMES: List[str] = list(ALL_TUNABLE_PARAM_SPECS.keys())
PARAM_BOUNDS: Dict[str, Tuple[float, float]] = {
    name: (float(spec["min"]), float(spec["max"]))
    for name, spec in ALL_TUNABLE_PARAM_SPECS.items()
}
DEFAULT_PARAMS: Dict[str, float] = {
    name: float(spec["nominal"])
    for name, spec in ALL_TUNABLE_PARAM_SPECS.items()
}

# ── Defaults aligned with GSMultiAgent config ────────────────────────────────
DEFAULT_OUTER_ITERATIONS = 5    # workflow.max_iterations
DEFAULT_EPISODES_PER_OUTER = 15 # matlab_rl_optimizer.max_episodes
DEFAULT_N_MC = 100              # CLI / test width; search uses mc_search_n_runs=10
DEFAULT_PE_DIR = Path("parameter_experience_base") / "single_agent_hermes"
FIRST_SUCCESS_DEFINITION = "final_dev_feasible"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _clip(params: Mapping[str, float]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for name in PARAM_NAMES:
        lo, hi = PARAM_BOUNDS[name]
        out[name] = float(np.clip(float(params.get(name, DEFAULT_PARAMS[name])), lo, hi))
    return out


def _float(result: Mapping[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(result.get(key, default))
    except (TypeError, ValueError):
        return default


def _fitness(result: Mapping[str, Any]) -> float:
    """Minimisation fitness: negative hit_rate when feasible, violation sum otherwise."""
    if bool(result.get("feasible", False)) and bool(result.get("success", False)):
        return -(float(result.get("hit_rate", 0.0)) / 100.0 + 0.01)
    violations = result.get("constraint_violations") or {}
    return float(sum(float(v) for v in violations.values()))


def _agg_seeds(seed_results: Sequence[Mapping[str, Any]]) -> Tuple[bool, float, Dict[str, float]]:
    """Aggregate seed results with conservative (worst-case) metrics."""
    all_pass = all(
        bool(r.get("feasible", False)) and bool(r.get("success", False))
        for r in seed_results
    )
    score = sum(_fitness(r) for r in seed_results) / max(len(seed_results), 1)
    metrics = {
        "hit_rate":  min(_float(r, "hit_rate")                              for r in seed_results),
        "SEP":       max(_float(r, "SEP", float("inf"))                     for r in seed_results),
        "peak_nz":   max(_float(r, "peak_nz", _float(r, "peak_ny", 1e9))   for r in seed_results),
        "PM":        min(_float(r, "PM",  _float(r, "pitch_PM"))            for r in seed_results),
        "BW":        min(_float(r, "BW",  _float(r, "pitch_BW"))            for r in seed_results),
    }
    return all_pass, score, metrics


def _mc_data_path(mc_root: Path, scenario: str, seed: int, split: str) -> Path:
    return mc_root / scenario / f"seed_{int(seed)}_{split}.mat"


# ── Tiny PPO — torch primary, numpy REINFORCE fallback ───────────────────────

def _run_ppo_episodes(
    evaluator_fn,           # callable(params) -> list[result_dict]  (over dev seeds)
    initial_params: Dict[str, float],
    master_seed: int,
    n_episodes: int,
    *,
    lr: float = 3e-4,
    clip_ratio: float = 0.2,
    entropy_coef: float = 0.01,
    ppo_epochs: int = 4,
) -> Tuple[Dict[str, float], float, List[Dict[str, Any]]]:
    """PPO episodes from initial_params.  torch when available, numpy fallback.

    Returns (best_params, best_score, history).
    No LLM calls.
    """
    def _norm(p: Dict[str, float]) -> np.ndarray:
        arr = []
        for name in PARAM_NAMES:
            lo, hi = PARAM_BOUNDS[name]
            arr.append((float(p.get(name, DEFAULT_PARAMS[name])) - lo) / max(hi - lo, 1e-12))
        return np.asarray(arr, dtype=float)

    def _denorm(x: np.ndarray) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for i, name in enumerate(PARAM_NAMES):
            lo, hi = PARAM_BOUNDS[name]
            z = float(np.clip(x[i], -1.0, 1.0))
            out[name] = lo + 0.5 * (z + 1.0) * (hi - lo)
        return out

    # ── try torch PPO ─────────────────────────────────────────────────────────
    try:
        import torch
        torch.manual_seed(int(master_seed))
        d   = len(PARAM_NAMES)
        x0  = 2.0 * _norm(initial_params) - 1.0
        mean    = torch.nn.Parameter(torch.tensor(x0, dtype=torch.float32))
        log_std = torch.nn.Parameter(torch.full((d,), math.log(0.35), dtype=torch.float32))
        value   = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        optim   = torch.optim.Adam([mean, log_std, value], lr=float(lr))

        best_params = dict(initial_params)
        best_score  = float("inf")
        history: List[Dict[str, Any]] = []

        for ep in range(1, int(n_episodes) + 1):
            with torch.no_grad():
                std  = torch.exp(log_std).clamp(1e-3, 2.0)
                dist = torch.distributions.Normal(mean, std)
                raw  = dist.sample()
                old_lp = dist.log_prob(raw).sum()
                action = torch.clamp(raw, -1.0, 1.0).cpu().numpy()

            params       = _clip(_denorm(action))
            seed_results = evaluator_fn(params)
            all_pass, score, metrics = _agg_seeds(seed_results)
            reward = float(-score)

            if score < best_score:
                best_score  = score
                best_params = params

            reward_t  = torch.tensor(reward, dtype=torch.float32)
            old_lp_d  = old_lp.detach()
            old_act_d = raw.detach()
            advantage = (reward_t - value.detach()).detach()
            for _ in range(int(ppo_epochs)):
                std  = torch.exp(log_std).clamp(1e-3, 2.0)
                dist = torch.distributions.Normal(mean, std)
                lp   = dist.log_prob(old_act_d).sum()
                ratio = torch.exp(lp - old_lp_d)
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * advantage
                loss  = (-torch.min(surr1, surr2)
                         + 0.5 * (value - reward_t).pow(2)
                         - float(entropy_coef) * dist.entropy().sum())
                optim.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_([mean, log_std, value], 1.0)
                optim.step()
                with torch.no_grad():
                    mean.clamp_(-1.0, 1.0)
                    log_std.clamp_(math.log(0.03), math.log(1.0))

            history.append({"ep": ep, "score": score, "all_pass": all_pass,
                             "reward": reward, "metrics": metrics, "backend": "torch"})
            _m = metrics
            logger.info(
                "[Ep %2d/%d] score=%.4f  reward=%+.4f  all_pass=%s  "
                "hit=%.1f%%  SEP=%.2fm  PM=%.1f  BW=%.1f  peak=%.1fg  [torch]",
                ep, n_episodes, score, reward, all_pass,
                _m.get("hit_rate", 0.0), _m.get("SEP", 0.0),
                _m.get("PM", 0.0), _m.get("BW", 0.0), _m.get("peak_nz", 0.0),
            )
        return best_params, best_score, history

    except ImportError:
        logger.info("[SAH] torch not found — using numpy REINFORCE fallback")

    # ── numpy REINFORCE fallback ──────────────────────────────────────────────
    rng = np.random.default_rng(int(master_seed))
    d   = len(PARAM_NAMES)
    x0  = 2.0 * _norm(initial_params) - 1.0       # [-1,1]^d

    # Adam state
    mean_np    = np.clip(x0, -1.0, 1.0).copy()
    log_std_np = np.full(d, math.log(0.35))
    m1, v1 = np.zeros(d), np.zeros(d)             # Adam moments for mean
    m2, v2 = np.zeros(d), np.zeros(d)             # Adam moments for log_std
    beta1, beta2, eps_adam, t = 0.9, 0.999, 1e-8, 0
    log_std_min, log_std_max = math.log(0.03), math.log(1.0)

    best_params = dict(initial_params)
    best_score  = float("inf")
    history: List[Dict[str, Any]] = []
    reward_baseline = 0.0

    for ep in range(1, int(n_episodes) + 1):
        std_np = np.clip(np.exp(log_std_np), 1e-3, 2.0)
        action = np.clip(rng.normal(mean_np, std_np), -1.0, 1.0)

        params       = _clip(_denorm(action))
        seed_results = evaluator_fn(params)
        all_pass, score, metrics = _agg_seeds(seed_results)
        reward = float(-score)
        reward_baseline = 0.8 * reward_baseline + 0.2 * reward
        advantage = reward - reward_baseline

        if score < best_score:
            best_score  = score
            best_params = params

        # REINFORCE gradient for mean and log_std
        z         = (action - mean_np) / (std_np + 1e-12)
        grad_mean = advantage * z / (std_np + 1e-12)
        grad_ls   = advantage * (z * z - 1.0) + float(entropy_coef)

        # Gradient clipping
        gm_norm = np.linalg.norm(grad_mean)
        if gm_norm > 1.0:
            grad_mean = grad_mean / gm_norm
        gl_norm = np.linalg.norm(grad_ls)
        if gl_norm > 1.0:
            grad_ls = grad_ls / gl_norm

        # Adam update (ascent → negate gradient)
        t += 1
        m1 = beta1 * m1 + (1 - beta1) * (-grad_mean)
        v1 = beta2 * v1 + (1 - beta2) * grad_mean ** 2
        m1h, v1h = m1 / (1 - beta1 ** t), v1 / (1 - beta2 ** t)
        mean_np -= lr * m1h / (np.sqrt(v1h) + eps_adam)
        mean_np  = np.clip(mean_np, -1.0, 1.0)

        m2 = beta1 * m2 + (1 - beta1) * (-grad_ls)
        v2 = beta2 * v2 + (1 - beta2) * grad_ls ** 2
        m2h, v2h = m2 / (1 - beta1 ** t), v2 / (1 - beta2 ** t)
        log_std_np -= lr * m2h / (np.sqrt(v2h) + eps_adam)
        log_std_np  = np.clip(log_std_np, log_std_min, log_std_max)

        history.append({"ep": ep, "score": score, "all_pass": all_pass,
                         "reward": reward, "metrics": metrics, "backend": "numpy"})
        _m = metrics
        logger.info(
            "[Ep %2d/%d] score=%.4f  reward=%+.4f  all_pass=%s  "
            "hit=%.1f%%  SEP=%.2fm  PM=%.1f  BW=%.1f  peak=%.1fg  [numpy]",
            ep, n_episodes, score, reward, all_pass,
            _m.get("hit_rate", 0.0), _m.get("SEP", 0.0),
            _m.get("PM", 0.0), _m.get("BW", 0.0), _m.get("peak_nz", 0.0),
        )

    return best_params, best_score, history


# ── ParameterExperience thin wrapper (no LLM) ────────────────────────────────

class _TinyPE:
    """Lightweight in-process PE: stores best params per scenario.

    Falls back to DEFAULT_PARAMS when the scenario has no history.
    Uses ParameterExperience if available, otherwise an in-memory dict.
    """

    def __init__(self, pe_dir: Path) -> None:
        self._pe_dir = pe_dir
        self._mem: Dict[str, Dict[str, float]] = {}   # in-process cache
        self._pe = None
        try:
            from multi_agent.memory.parameter_experience import ParameterExperience
            pe_dir.mkdir(parents=True, exist_ok=True)
            self._pe = ParameterExperience(
                max_short_term_size=200,
                max_long_term_size=500,
            )
            logger.info("[SAH] ParameterExperience initialised at %s", pe_dir)
        except Exception as exc:
            logger.warning("[SAH] ParameterExperience not available (%s); using in-memory dict", exc)

    def get_warm_start(self, scenario: str) -> Dict[str, float]:
        """Return best known params for scenario, or DEFAULT_PARAMS."""
        # In-process cache first (most recent outer-iteration result)
        if scenario in self._mem:
            return dict(self._mem[scenario])
        # Try ParameterExperience
        if self._pe is not None:
            try:
                results = self._pe.retrieve(
                    task_context={"scenario": scenario},
                    top_k=1,
                    memory_type=None,
                )
                if results:
                    entry = results[0]
                    params = entry.parameters if hasattr(entry, "parameters") else {}
                    if params:
                        clipped = _clip(params)
                        logger.info("[SAH] PE warm-start for %s: %s…", scenario,
                                    {k: round(v, 3) for k, v in list(clipped.items())[:3]})
                        return clipped
            except Exception as exc:
                logger.debug("[SAH] PE retrieval failed: %s", exc)
        return dict(DEFAULT_PARAMS)

    def store(self, scenario: str, params: Dict[str, float], metrics: Dict[str, float]) -> None:
        """Persist best params after an outer iteration."""
        self._mem[scenario] = dict(params)
        if self._pe is not None:
            try:
                import asyncio
                fitness = float(metrics.get("hit_rate", 0.0)) / 100.0
                coro = self._pe.store(
                    task_context={"scenario": scenario, "source": "single_agent_hermes"},
                    parameters=dict(params),
                    objectives=dict(metrics),
                    fitness=fitness,
                )
                # Run the coroutine: reuse running loop if inside async context,
                # otherwise create a new one.
                try:
                    loop = asyncio.get_running_loop()
                    loop.create_task(coro)
                except RuntimeError:
                    asyncio.run(coro)
            except Exception as exc:
                logger.debug("[SAH] PE store failed: %s", exc)


# ── LLM Advisor (HermesIntegration primary, LangChain fallback) ──────────────

@dataclass
class _LLMAdvice:
    """Structured advice produced by the LLM after each outer PPO iteration."""
    continue_optimization: bool = True
    suggestion: str = ""
    # delta_fraction ∈ [-0.15, 0.15] applied as fractional nudge to warm-start
    param_hints: Dict[str, float] = field(default_factory=dict)
    # subset of {lr, n_episodes, entropy_coef}
    rl_adjustments: Dict[str, Any] = field(default_factory=dict)
    raw_response: str = ""


class _HermesLLMAdvisor:
    """LLM 顾问，通过 Hermes Agent（AIAgent）向 PPO 外层循环注入大模型建议。

    后端优先级
    ----------
    1. ``run_agent.AIAgent``（Hermes Agent 原生接口，同步调用）
       - 无工具（enabled_toolsets 指向不存在的占位符）
       - 通过 ``system_message`` 参数注入 SYSTEM_PROMPT
       - ``run_conversation()`` 同步返回 ``{"final_response": str, ...}``
    2. ``LangChain ChatOpenAI``（与 ReflectionAgent 同配置，备用）
    3. 静默 no-op（两者均不可用时退化为纯 PPO）
    """

    # Toolset name that does not exist in the Hermes registry →
    # AIAgent loads 0 tools, giving us a pure text-generation endpoint.
    _ADVISORY_TOOLSET = "__sah_advisory__"

    _SYSTEM_PROMPT = (
        "你是制导系统参数优化专家。\n"
        "你将收到 SingleAgentHermes 某次外层 PPO 迭代的结果与历史记录。\n"
        "请分析最优指标，判断是否需要继续优化，并给出参数调整建议。\n\n"
        "请只返回如下 JSON，不包含任何其他文字：\n"
        '{"continue_optimization": true,\n'
        ' "suggestion": "简短中文说明，50字以内",\n'
        ' "param_hints": {},\n'
        ' "rl_adjustments": {}}\n\n'
        "param_hints 格式: {param_name: delta_fraction}，"
        "delta_fraction ∈ [-0.15, 0.15]，表示对 warm-start 的比例偏移量。\n"
        "rl_adjustments 可含: lr (float), n_episodes (int), entropy_coef (float)。\n"
        "若无需调整则填空 {}。"
    )

    def __init__(self) -> None:
        self._agent: Optional[Any] = None   # run_agent.AIAgent instance
        self._llm: Optional[Any] = None     # LangChain fallback
        self._available = False
        self._backend = "none"
        self._init_backend()

    def _init_backend(self) -> None:
        """Try AIAgent (Hermes); fall back to LangChain ChatOpenAI."""
        # ── Primary: Hermes Agent AIAgent ─────────────────────────────────────
        # run_agent.AIAgent is the public integration point of hermes-agent.
        # We create one instance with:
        #   • enabled_toolsets=[_ADVISORY_TOOLSET]  → 0 tools (pure text gen)
        #   • quiet_mode=True                        → no console output
        #   • skip_context_files=True                → skip SOUL.md/AGENTS.md
        #   • max_iterations=5                       → enough for one reply
        try:
            from run_agent import AIAgent
            from multi_agent.llm_failover import ensure_usable_endpoint

            ep = ensure_usable_endpoint(probe=True)
            if not ep.api_key:
                raise RuntimeError("no api_key configured")
            self._agent = AIAgent(
                base_url=ep.base_url,
                api_key=ep.api_key,
                provider=ep.provider,
                model=ep.model,
                max_iterations=5,
                enabled_toolsets=[self._ADVISORY_TOOLSET],
                quiet_mode=True,
                skip_context_files=True,
            )
            self._available = True
            self._backend = "hermes"
            logger.info(
                "[SAH-LLM] Backend: Hermes Agent AIAgent (%s @ %s)",
                ep.model,
                ep.base_url,
            )
            return
        except ImportError:
            logger.debug("[SAH-LLM] run_agent not importable; trying LangChain")
        except Exception as exc:
            logger.debug("[SAH-LLM] AIAgent init failed: %s", exc)

        # ── Fallback: LangChain ChatOpenAI (same as ReflectionAgent) ──────────
        try:
            from langchain_openai import ChatOpenAI
            from multi_agent.llm_failover import ensure_usable_endpoint

            ep = ensure_usable_endpoint(probe=True)
            if not ep.api_key:
                logger.warning("[SAH-LLM] No api_key in config; advisor disabled.")
                return
            self._llm = ChatOpenAI(
                model=ep.model,
                openai_api_key=ep.api_key,
                openai_api_base=ep.base_url,
                temperature=0.2,
                extra_body={"include_reasoning": True},
            )
            self._available = True
            self._backend = "langchain"
            logger.info("[SAH-LLM] Backend: LangChain ChatOpenAI (%s)", ep.model)
        except Exception as exc:
            logger.warning("[SAH-LLM] No LLM backend available: %s", exc)

    def _rebuild_hermes_agent(self) -> bool:
        """Rebuild AIAgent from the currently active (possibly switched) endpoint."""
        try:
            from run_agent import AIAgent
            from multi_agent.llm_failover import get_active_endpoint, apply_endpoint

            ep = get_active_endpoint()
            apply_endpoint(ep)
            if not ep.api_key:
                return False
            self._agent = AIAgent(
                base_url=ep.base_url,
                api_key=ep.api_key,
                provider=ep.provider,
                model=ep.model,
                max_iterations=5,
                enabled_toolsets=[self._ADVISORY_TOOLSET],
                quiet_mode=True,
                skip_context_files=True,
            )
            self._backend = "hermes"
            self._available = True
            logger.info(
                "[SAH-LLM] Rebuilt Hermes AIAgent after failover → %s",
                ep.label,
            )
            return True
        except Exception as exc:
            logger.warning("[SAH-LLM] Rebuild after failover failed: %s", exc)
            return False

    @property
    def available(self) -> bool:
        return self._available

    def advise(
        self,
        scenario: str,
        outer: int,
        n_outer: int,
        best_metrics: Dict[str, float],
        best_score: float,
        iter_history: List[Dict[str, Any]],
    ) -> "_LLMAdvice":
        """Call the LLM and return structured advice.  Fully synchronous."""
        if not self._available:
            return _LLMAdvice()

        import json as _json
        hist_rows: List[str] = []
        for h in iter_history[-10:]:
            hist_rows.append(
                f"  ep={h.get('ep','')}  score={h.get('score',0):.4f}"
                f"  all_pass={h.get('all_pass',False)}"
                f"  backend={h.get('backend','')}"
            )
        user_msg = (
            f"场景: {scenario}\n"
            f"外层迭代: {outer}/{n_outer}\n"
            f"当前最优指标: {_json.dumps(best_metrics, ensure_ascii=False)}\n"
            f"当前最优得分: {best_score:.6f}\n"
            f"最近 {len(hist_rows)} 个 episode 历史:\n"
            + ("\n".join(hist_rows) if hist_rows else "  （无历史）")
            + "\n\n请分析以上优化结果，给出 JSON 格式建议。"
        )

        from multi_agent.llm_failover import is_balance_error, note_failure_and_switch

        attempts = 0
        max_attempts = 3
        while attempts < max_attempts:
            attempts += 1
            raw = ""
            try:
                if self._agent is not None:
                    result = self._agent.run_conversation(
                        user_message=user_msg,
                        system_message=self._SYSTEM_PROMPT,
                    )
                    raw = (result or {}).get("final_response") or ""

                elif self._llm is not None:
                    import asyncio
                    from langchain_core.messages import HumanMessage, SystemMessage
                    msgs = [
                        SystemMessage(content=self._SYSTEM_PROMPT),
                        HumanMessage(content=user_msg),
                    ]
                    try:
                        asyncio.get_running_loop()
                        import concurrent.futures as _cf
                        with _cf.ThreadPoolExecutor(max_workers=1) as pool:
                            resp = pool.submit(
                                asyncio.run, self._llm.ainvoke(msgs)
                            ).result(timeout=90.0)
                    except RuntimeError:
                        resp = asyncio.run(self._llm.ainvoke(msgs))
                    raw = resp.content if hasattr(resp, "content") else str(resp)

            except Exception as exc:
                logger.warning(
                    "[SAH-LLM] Advisor call failed (%s): %s",
                    type(exc).__name__,
                    exc,
                )
                if is_balance_error(exc) and note_failure_and_switch(exc):
                    if self._rebuild_hermes_agent():
                        continue
                return _LLMAdvice()

            if is_balance_error(raw):
                logger.warning(
                    "[SAH-LLM] Balance/quota error in advisor response: %.200s",
                    raw,
                )
                if note_failure_and_switch(raw) and self._rebuild_hermes_agent():
                    continue
                return _LLMAdvice(suggestion=str(raw)[:200], raw_response=str(raw))

            return self._parse(str(raw))

        return _LLMAdvice()

    @staticmethod
    def _parse(raw: str) -> "_LLMAdvice":
        import json as _json
        import re as _re
        if not raw:
            return _LLMAdvice(raw_response="")
        # Strip markdown fences and <think> blocks from reasoning models
        clean = _re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
        clean = _re.sub(r"<think>.*?</think>", "", clean, flags=_re.DOTALL).strip()
        m = _re.search(r"\{.*\}", clean, _re.DOTALL)
        if not m:
            return _LLMAdvice(suggestion=raw[:200], raw_response=raw)
        try:
            data = _json.loads(m.group())
        except _json.JSONDecodeError as exc:
            logger.warning("[SAH-LLM] JSON parse failed: %s | raw: %.200s", exc, raw)
            return _LLMAdvice(suggestion=raw[:200], raw_response=raw)
        # Clamp param_hints to [-0.15, 0.15] for safety
        raw_hints = data.get("param_hints") or {}
        param_hints: Dict[str, float] = {}
        if isinstance(raw_hints, dict):
            for k, v in raw_hints.items():
                try:
                    param_hints[str(k)] = float(np.clip(float(v), -0.15, 0.15))
                except (TypeError, ValueError):
                    pass
        return _LLMAdvice(
            continue_optimization=bool(data.get("continue_optimization", True)),
            suggestion=str(data.get("suggestion", "")),
            param_hints=param_hints,
            rl_adjustments=dict(data.get("rl_adjustments") or {}),
            raw_response=raw,
        )


# ── Main class ────────────────────────────────────────────────────────────────

class SingleAgentHermes:
    """Single-agent Hermes baseline with LLM advisor.

    Parameters
    ----------
    mc_root : path-like
        Root directory containing ``<scenario>/seed_N_<split>.mat``.
    matlab_script : path-like
        Path to ``monte_carlo_single.m`` template.
    n_runs : int
        Requested trajectories per MATLAB call (test/final width).
        Search/dev uses ``mc_search_n_runs`` when ``mc_tiered_eval`` is on.
    pe_dir : path-like, optional
        Directory for ParameterExperience persistence.
        Defaults to ``parameter_experience_base/single_agent_hermes``.
    use_llm : bool, default True
        Whether to enable the _HermesLLMAdvisor.  When True, the LLM is
        called after each outer PPO iteration to analyse results and
        optionally adjust the next-iteration warm-start and RL
        hyper-parameters.  Falls back gracefully to no-op if neither
        HermesIntegration nor LangChain is available.
    """

    def __init__(
        self,
        mc_root: Path,
        matlab_script: Path,
        n_runs: int = DEFAULT_N_MC,
        pe_dir: Optional[Path] = None,
        use_llm: bool = True,
    ) -> None:
        self.mc_root       = Path(mc_root)
        self.matlab_script = Path(matlab_script)
        self.n_runs        = int(n_runs)
        self._last_mc_calls = 0
        self._pe           = _TinyPE(Path(pe_dir or DEFAULT_PE_DIR))
        self._llm_advisor: Optional[_HermesLLMAdvisor] = (
            _HermesLLMAdvisor() if use_llm else None
        )
        if self._llm_advisor is not None and self._llm_advisor.available:
            logger.info("[SAH] LLM advisor active (backend=%s)",
                        self._llm_advisor._backend)
        else:
            logger.info("[SAH] LLM advisor disabled or unavailable; "
                        "running as pure-PPO baseline.")

    # ── internal evaluator ────────────────────────────────────────────────────

    def _n_runs_for_split(self, split: str) -> int:
        """Match GSM: search/dev = 10, test/final = 100 when tiered eval is on."""
        n_runs = int(self.n_runs)
        try:
            from multi_agent.config_loader import get_config

            sim = get_config().simulation
            if bool(getattr(sim, "mc_tiered_eval", False)):
                if split == "dev":
                    n_runs = int(getattr(sim, "mc_search_n_runs", n_runs) or n_runs)
                else:
                    n_runs = int(getattr(sim, "mc_final_n_runs", n_runs) or n_runs)
        except Exception:
            pass
        return max(1, int(n_runs))

    def _eval_seeds(
        self,
        params: Dict[str, float],
        scenario: str,
        seeds: Sequence[int],
        split: str,
    ) -> List[Dict[str, Any]]:
        """Evaluate params on every seed via the shared external-MC AND path.

        Uses ``run_mc_and_across_seeds`` (same entry as GSMultiAgent) with
        tiered n_runs. Bootstrap fill is off so all listed seeds are real
        MATLAB calls — stricter than GSM's one-seed bootstrap.
        """
        seed_list = [int(s) for s in seeds]
        n_runs = self._n_runs_for_split(split)
        ext = run_mc_and_across_seeds(
            str(self.matlab_script),
            mc_root=str(self.mc_root),
            scenario=scenario,
            seeds=seed_list,
            split=split,
            n_runs=n_runs,
            kill_radius=10.0,
            dp_params=_clip(params),
            early_abort=False,
            bootstrap_fill=False,
        )
        results = list(ext.get("seed_results") or [])
        if not results:
            results = [
                {
                    "success": False,
                    "feasible": False,
                    "error": ext.get("error") or "empty seed_results",
                    "constraint_violations": {"missing_data": 1e6},
                    "sim_time_sec": 0.0,
                    "mc_seed": seed,
                    "split": split,
                }
                for seed in seed_list
            ]
        self._last_mc_calls = int(ext.get("mc_calls") or len(results) or 0)
        return results

    # ── public API ────────────────────────────────────────────────────────────

    def run(
        self,
        scenario: str,
        dev_seeds: Sequence[int],
        test_seeds: Sequence[int],
        master_seed: int = 42,
        n_outer_iterations: int = DEFAULT_OUTER_ITERATIONS,
        n_episodes_per_outer: int = DEFAULT_EPISODES_PER_OUTER,
    ) -> Dict[str, Any]:
        """Run the single-agent optimization loop.

        Returns a dict compatible with BaselineRunResult.to_json().
        """
        from experiments.equal_mc_budget import max_mc_calls, remaining_candidates

        started = time.perf_counter()
        best_params  = dict(DEFAULT_PARAMS)
        best_score   = float("inf")
        best_metrics: Dict[str, float] = {}
        first_success: Optional[int]   = None
        mc_calls_dev = 0
        sim_time     = 0.0
        history: List[Dict[str, Any]]  = []
        llm_advice_log: List[Dict[str, Any]] = []
        _n_episodes = int(n_episodes_per_outer)
        n_seeds = max(1, len(dev_seeds))
        max_mc = max_mc_calls(n_dev_seeds=n_seeds)
        n_runs_search = self._n_runs_for_split("dev")
        n_runs_test = self._n_runs_for_split("test")
        logger.info(
            "[SAH] MC protocol: search n_runs=%d  test n_runs=%d  "
            "bootstrap=off  cap=%d  first_success=%s",
            n_runs_search, n_runs_test, max_mc, FIRST_SUCCESS_DEFINITION,
        )

        def _eval_dev(params: Dict[str, float]) -> List[Dict[str, Any]]:
            nonlocal mc_calls_dev, sim_time
            rs = self._eval_seeds(params, scenario, dev_seeds, "dev")
            mc_calls_dev += int(self._last_mc_calls or len(rs))
            sim_time += sum(_float(r, "sim_time_sec") for r in rs)
            return rs

        iterations_used = 0
        for outer in range(1, int(n_outer_iterations) + 1):
            n_left = remaining_candidates(mc_calls_dev, max_mc, n_seeds)
            if n_left <= 0:
                logger.info("[SAH] MC budget exhausted at %d/%d; stop outer loop",
                            mc_calls_dev, max_mc)
                break
            iterations_used = outer
            warm_params = self._pe.get_warm_start(scenario)
            n_this = min(_n_episodes, n_left)

            # ── Outer-iteration header ────────────────────────────────────────
            _sep = "─" * 60
            logger.info(_sep)
            logger.info(
                "▶ Outer %d/%d  scenario=%s  episodes=%d  seed=%d  mc=%d/%d  n_runs=%d",
                outer, n_outer_iterations, scenario, n_this,
                master_seed + outer * 1000, mc_calls_dev, max_mc, n_runs_search,
            )
            _warm_preview = {k: round(v, 3)
                             for k, v in list(warm_params.items())[:4]}
            logger.info("  warm-start (PE): %s …", _warm_preview)

            iter_best_params, iter_best_score, iter_history = _run_ppo_episodes(
                evaluator_fn=_eval_dev,
                initial_params=warm_params,
                master_seed=master_seed + outer * 1000,   # deterministic per outer
                n_episodes=n_this,
            )

            history.extend(iter_history)
            if first_success is None:
                if any(h.get("all_pass") for h in iter_history):
                    first_success = outer

            # ── Post-PPO summary for this outer iteration ─────────────────────
            _iter_pass = sum(1 for h in iter_history if h.get("all_pass"))
            logger.info(
                "◀ Outer %d done  iter_best=%.4f  pass_eps=%d/%d",
                outer, iter_best_score, _iter_pass, len(iter_history),
            )

            if iter_best_score < best_score:
                best_score   = iter_best_score
                best_params  = iter_best_params
                if iter_history:
                    best_metrics = dict(iter_history[-1].get("metrics", {}))
                    for h in reversed(iter_history):
                        if h.get("all_pass"):
                            best_metrics = dict(h.get("metrics", {}))
                            break
                logger.info(
                    "  ★ New global best: score=%.4f  hit=%.1f%%  "
                    "SEP=%.2fm  PM=%.1f  BW=%.1f  peak=%.1fg",
                    best_score,
                    best_metrics.get("hit_rate", 0.0),
                    best_metrics.get("SEP", 0.0),
                    best_metrics.get("PM", 0.0),
                    best_metrics.get("BW", 0.0),
                    best_metrics.get("peak_nz", 0.0),
                )
            else:
                logger.info("  (no improvement over global best %.4f)", best_score)

            # Persist to PE after each outer iteration
            self._pe.store(scenario, iter_best_params, best_metrics)

            # ── LLM reflection ────────────────────────────────────────────────
            # Call the Hermes-backed advisor; skip on the final outer iteration
            # (nothing to apply) or when the advisor is disabled/unavailable.
            _advisor = self._llm_advisor
            if _advisor is not None and _advisor.available and outer < int(n_outer_iterations):
                logger.info("[LLM] Calling advisor (outer %d/%d)…",
                            outer, n_outer_iterations)
                advice = _advisor.advise(
                    scenario=scenario,
                    outer=outer,
                    n_outer=int(n_outer_iterations),
                    best_metrics=best_metrics,
                    best_score=best_score,
                    iter_history=iter_history,
                )
                if advice.suggestion:
                    logger.info("[LLM] 建议: %s", advice.suggestion)

                llm_advice_log.append({
                    "outer":          outer,
                    "continue":       advice.continue_optimization,
                    "suggestion":     advice.suggestion,
                    "param_hints":    advice.param_hints,
                    "rl_adjustments": advice.rl_adjustments,
                })

                # Apply param_hints: nudge best_params → becomes warm-start
                # for the next outer iteration via PE write-back above.
                if advice.param_hints:
                    nudged = dict(best_params)
                    for name, delta_frac in advice.param_hints.items():
                        if name in PARAM_BOUNDS:
                            lo, hi = PARAM_BOUNDS[name]
                            nudged[name] = float(np.clip(
                                nudged.get(name, DEFAULT_PARAMS[name])
                                + delta_frac * (hi - lo),
                                lo, hi,
                            ))
                    nudged = _clip(nudged)
                    self._pe.store(scenario, nudged, best_metrics)
                    logger.info("[LLM] Applied %d param hint(s) → PE warm-start"
                                " for outer %d",
                                len(advice.param_hints), outer + 1)

                # Apply RL adjustments for the next outer iteration
                _ra = advice.rl_adjustments
                if isinstance(_ra, dict) and _ra:
                    if "n_episodes" in _ra:
                        try:
                            _n_episodes = max(1, int(_ra["n_episodes"]))
                            logger.info("[LLM] n_episodes → %d (next outer)",
                                        _n_episodes)
                        except (TypeError, ValueError):
                            pass
                    for _k in ("lr", "entropy_coef"):
                        if _k in _ra:
                            logger.info("[LLM] Suggested %s → %s (recorded)",
                                        _k, _ra[_k])

                # Early stop if LLM signals all constraints met
                if not advice.continue_optimization:
                    logger.info("[LLM] Early stop signalled at outer=%d ✓",
                                outer)
                    break

        # ── Final test evaluation (100 trajectories; not counted in dev budget)
        test_results = self._eval_seeds(best_params, scenario, test_seeds, "test")
        sim_time += sum(_float(r, "sim_time_sec") for r in test_results)
        test_all_pass, test_score, test_metrics = _agg_seeds(test_results)

        # Dev pass: re-evaluate best params on dev (search width). Not added to
        # mc_calls_dev so the shared 225-call search ceiling stays comparable.
        final_dev_results = self._eval_seeds(best_params, scenario, dev_seeds, "dev")
        dev_all_pass, _, dev_metrics = _agg_seeds(final_dev_results)

        wall = time.perf_counter() - started
        return {
            "method":                "single_agent_hermes",
            "scenario":              scenario,
            "master_seed":           int(master_seed),
            "search_dim":            len(PARAM_NAMES),
            "iterations_budget":     int(n_outer_iterations),
            "iterations_used":       int(iterations_used),
            "design_path":           "TUNE_PARAMS",
            "mc_calls_dev":          int(mc_calls_dev),
            "mc_calls_test":         len(test_results),
            "wall_clock_sec":        wall,
            "sim_time_sec":          sim_time,
            "first_success_iter":    first_success,
            "first_success_definition": FIRST_SUCCESS_DEFINITION,
            "n_runs_search":         int(n_runs_search),
            "n_runs_test":           int(n_runs_test),
            "dev_all_pass":          bool(dev_all_pass),
            "test_all_pass":         bool(test_all_pass),
            "best_score":            float(best_score),
            "best_params":           dict(best_params),
            "best_dev":              {"metrics": dev_metrics, "all_pass": dev_all_pass},
            "test_eval":             {"metrics": test_metrics, "all_pass": test_all_pass,
                                      "seed_results": test_results},
            "history":               history,
            "llm_advice_log":        llm_advice_log,
            "llm_backend":           (self._llm_advisor._backend
                                      if self._llm_advisor is not None else "none"),
            "error":                 "",
        }


# ── Convenience function for optimizer_baselines_43 integration ───────────────

def run_single_agent_hermes(
    evaluator_mc_root: Path,
    matlab_script: Path,
    *,
    scenario: str,
    master_seed: int,
    dev_seeds: Sequence[int],
    test_seeds: Sequence[int],
    n_outer_iterations: int = DEFAULT_OUTER_ITERATIONS,
    n_episodes_per_outer: int = DEFAULT_EPISODES_PER_OUTER,
    n_runs: int = DEFAULT_N_MC,
    pe_dir: Optional[Path] = None,
    use_llm: bool = True,
) -> Dict[str, Any]:
    """Drop-in for optimizer_baselines_43.run_one routing."""
    agent = SingleAgentHermes(
        mc_root=evaluator_mc_root,
        matlab_script=matlab_script,
        n_runs=n_runs,
        pe_dir=pe_dir,
        use_llm=use_llm,
    )
    return agent.run(
        scenario=scenario,
        dev_seeds=dev_seeds,
        test_seeds=test_seeds,
        master_seed=master_seed,
        n_outer_iterations=n_outer_iterations,
        n_episodes_per_outer=n_episodes_per_outer,
    )

