#!/usr/bin/env python3

"""

Multi-Agent Guidance System - CLI End-to-End Execution

"""



import os

# Disable LangSmith tracing before any LangChain imports to avoid SSL
# connection warnings and timeout delays when the service is unreachable.
os.environ.setdefault("LANGCHAIN_TRACING_V2", "false")
os.environ.setdefault("LANGSMITH_TRACING", "false")

import asyncio
import sys

# Windows: ProactorEventLoop causes WinError 10038 with httpx streaming.
# Switch to SelectorEventLoop which is stable for HTTP client connections.
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import argparse

import math

import re

import json

import importlib

import logging


def _async_module(name: str):
    """Import submodules under multi_agent.asynctask (async is reserved in Py 3.14+)."""
    return importlib.import_module(f"multi_agent.asynctask.{name}")

from pathlib import Path

from datetime import datetime

from dotenv import load_dotenv



# Root logging is configured in __main__ via multi_agent.logging.run_logger.
# Keep asyncio noise filter only at import time.
class _SuppressEventLoopClosed(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return "Event loop is closed" not in record.getMessage()

logging.getLogger("asyncio").addFilter(_SuppressEventLoopClosed())



from multi_agent.tools.simulation_tool import (

    RunSimulationTool,

)

from multi_agent.rl.matlab_rl_optimizer import (

    parse_mission_conditions,

    build_matlab_conditions_str,

    extract_autopilot_params,

    extract_physical_params,

    parse_sim_stdout,

    AUTOPILOT_PARAM_SPECS,

    GUIDANCE_PARAM_SPECS,

    ALL_TUNABLE_PARAM_SPECS,

    CATEGORY_DEFS,

)

from multi_agent.security.parameter_firewall import (
    apply_rl_hyperparams_to_optimizer,
    sanitize_rl_hyperparams,
)
from multi_agent.simulation.task_workspace import TaskWorkspace




dotenv_path = Path(__file__).parent / ".env"

if dotenv_path.exists():

    load_dotenv(dotenv_path)



async def main(args) -> int:

    from multi_agent.config_loader import get_config
    from multi_agent.integration.design_path_policy import metrics_summary_line
    from multi_agent.logging.log_verbosity import (
        cprint,
        dprint,
        hermes_verbose_thinking,
        init_verbosity_from_config,
        is_quiet,
        is_verbose,
        mprint,
        task_summary_line,
        truncate_text,
    )

    cfg = get_config()
    if getattr(args, "mc_data_path", None):
        cfg.simulation.mc_data_path = str(args.mc_data_path)
        os.environ["GSM_MC_DATA_PATH"] = str(args.mc_data_path)
    _verbosity = init_verbosity_from_config(cfg, args)

    _run_start = datetime.now()

    if is_quiet():
        mprint(
            f"GSMultiAgent [{_verbosity}] {_run_start:%Y-%m-%d %H:%M:%S} | "
            f"{task_summary_line(args.prompt, 100)}"
        )
    else:
        mprint("=" * 80)
        mprint("Multi-Agent Guidance System - CLI Workflow")
        mprint("=" * 80)
        mprint(f"Start Time: {_run_start.strftime('%Y-%m-%d %H:%M:%S')}  verbosity={_verbosity}")
        if is_verbose():
            mprint(f"Task: {args.prompt}")
        else:
            mprint(f"Task: {task_summary_line(args.prompt, 320)}")
        mprint("=" * 80)



    # 1. Initialize components

    print("\n[Step 1] Initializing components...")

    

    from multi_agent import (

        HermesIntegration,

        HERMES_AVAILABLE,

        IntelligentTaskPlanner,

        ExecutionStrategy,

        RAGKnowledgeBase,

        ParameterExperience,

        MemoryType,

    )

    from multi_agent.memory.hermes_agent_memory import HermesAgentMemory

    from multi_agent.simulation import GuidanceSimulator

    from multi_agent.integration.reflection_agent import ReflectionAgent

    from multi_agent.integration.judgment_agent import JudgmentAgent

    from multi_agent.simulation.optimization_workflow import OptimizationWorkflow

    from multi_agent.config_loader import get_config

    from multi_agent.rl.matlab_rl_optimizer import MatlabRLOptimizer



    cfg          = get_config()

    ablation_cfg = cfg.ablation

    _abl_off = [k for k, v in {
        "guidance_compat_verification": ablation_cfg.guidance_compat_verification,
        "syntax_check":                 ablation_cfg.syntax_check,
        "rl_optimization":              ablation_cfg.rl_optimization,
        "parameter_experience_reuse":   ablation_cfg.parameter_experience_reuse,
        "reflection_agent":             ablation_cfg.reflection_agent,
        "memory_search":                ablation_cfg.memory_search,
    }.items() if not v]
    if _abl_off:
        print(f"  [Ablation Study] Components DISABLED: {', '.join(_abl_off)}")
    else:
        print("  [Ablation Study] All components enabled (full system).")

    rl_cfg  = cfg.matlab_rl_optimizer

    mo_cfg  = cfg.model_output

    wf_cfg  = cfg.workflow

    sim_cfg = cfg.simulation
    if getattr(sim_cfg, "mc_data_path", ""):
        print(f"  [External MC] Using dataset: {sim_cfg.mc_data_path}")



    MODEL_OUTPUT_DIR   = Path(mo_cfg.generated_dir)

    import os as _os
    PE_BASE_DIR = Path(_os.environ.get("GSM_PE_BASE", mo_cfg.experience_base_dir))
    PE_JSON_PATH = Path(
        _os.environ.get("GSM_PE_JSON", str(Path("./parameter_experience_memory.json")))
    )

    MATLAB_SCRIPTS_DIR = Path(
        _os.environ.get("GSM_MATLAB_SCRIPTS_DIR", mo_cfg.matlab_scripts_dir)
    )

    for _d in [MODEL_OUTPUT_DIR, PE_BASE_DIR / "models", PE_BASE_DIR / "params", MATLAB_SCRIPTS_DIR]:

        _d.mkdir(parents=True, exist_ok=True)

    _stale_ws = TaskWorkspace.sweep_stale()
    if _stale_ws:
        print(f"  [TaskWorkspace] swept {_stale_ws} stale sandbox(es)")



    rag = RAGKnowledgeBase()

    await rag.initialize()



    # Index only the knowledge_base into RAG (tagged source_tag=knowledge_base).
    # PE data is intentionally NOT indexed into RAG — it lives exclusively in
    # ParameterExperience memory and is accessed via parameter_experience_* tools
    # or memory_search(search_type="experience").  This keeps rag_retrieve results
    # clean of parameter JSON files regardless of parameter_experience_reuse flag.

    cprint("  Indexing knowledge_base + reports into RAG…")
    dprint("  Indexing knowledge_base into RAG (source_tag=knowledge_base)...")

    await rag.index_directory(
        "./knowledge_base",
        patterns=[".md", ".txt", ".m", ".xml"],
        source_tag="knowledge_base",
    )

    # Index simulation reports so Hermes can retrieve them in task planning
    import pathlib as _pl_rpt_init
    _rpt_dirs = [
        ("./guidance_output/reports", [".json"]),
        ("./guidance_output",         [".md"]),
    ]
    for _rpt_d, _rpt_pat in _rpt_dirs:
        _rpt_path = _pl_rpt_init.Path(_rpt_d)
        if _rpt_path.exists():
            _rpt_n = await rag.index_directory(
                str(_rpt_path), patterns=_rpt_pat, source_tag="reports",
            )
            if _rpt_n:
                dprint(f"  Indexed {_rpt_n} report file(s) from {_rpt_d} (source_tag=reports)")

    _purged = rag.purge_stale_pe_docs()
    if _purged:
        dprint(f"  [RAG Purge] Removed {_purged} stale PE/intermediate doc(s) from ChromaDB.")

    _tag_audit = rag.audit_tags()
    _tag_summary = " | ".join(f"{t}:{c}" for t, c in sorted(_tag_audit["by_tag"].items()))
    cprint(f"  RAG ready: total={_tag_audit['total']} | {_tag_summary}")
    if _tag_audit["non_kb_files"]:
        cprint(f"  [RAG] ⚠ {len(_tag_audit['non_kb_files'])} non-KB doc(s) in collection")
        for _fname, _ftag in _tag_audit["non_kb_files"][:10]:
            dprint(f"    {_ftag:22s}  {_fname}")
    else:
        dprint("  [RAG Tag Audit] ✓ All indexed documents are KB-tagged.")

    if not ablation_cfg.parameter_experience_reuse:

        print("  [Ablation] parameter_experience_reuse: DISABLED — PE tools/history skipped.")



    parameter_experience = ParameterExperience(

        experience_base_dir=str(PE_BASE_DIR),
        persist_path=str(PE_JSON_PATH),

    )

    # pe_for_tools: passed to Hermes tools and the RL optimizer.

    # None when parameter_experience_reuse is disabled so no historical

    # parameters influence the optimizer's initial conditions or Hermes.

    # The real `parameter_experience` object is still used in Step 4 writeback.

    pe_for_tools = parameter_experience if ablation_cfg.parameter_experience_reuse else None



    simulator = GuidanceSimulator(

        output_dir="./guidance_output",

        engine=sim_cfg.engine,

        octave_path=sim_cfg.octave_path,

        matlab_path=sim_cfg.matlab_path,

    )



    if ablation_cfg.reflection_agent:

        reflection_agent = ReflectionAgent()

    else:

        reflection_agent = None

        print("  [Ablation] reflection_agent: DISABLED.")



    # ── Parse mission conditions from prompt ─────────────────────────────────

    print("\n[Pre-Step] Parsing mission conditions from prompt…")

    dprint(f"  [DEBUG] args.prompt = {args.prompt!r}")
    mission_conditions = parse_mission_conditions(args.prompt)
    dprint(f"  [DEBUG] parse_mission_conditions → {mission_conditions}")

    if mission_conditions:

        for cat, subs in mission_conditions.items():

            sub_names = [CATEGORY_DEFS[cat]["subcases"].get(s, str(s)) for s in subs]

            dprint(f"  Category {cat} ({CATEGORY_DEFS[cat]['name']}): {sub_names}")

    else:

        cprint("  No explicit mission conditions found; defaulting to ALL categories.")

        mission_conditions = {}  # empty = RUN_CASE='ALL' in monte_carlo_single



    # ── Persistent memory (hermes-agent-demo pattern) ─────────────────────────
    # Loaded once at startup; all agent_memory_* tool calls read/write the
    # same JSON file so facts persist across CLI sessions.
    # Gated by memory_search: when disabled, pass None so agent_memory_*
    # tools have no backing store and cannot leak historical info.
    if ablation_cfg.memory_search:
        print("  Initializing Hermes persistent memory...")
        hermes_memory = HermesAgentMemory(
            memory_file=_os.environ.get("GSM_HERMES_JSON") or None
        )
        if hermes_memory:
            print(f"  Loaded {len(hermes_memory)} long-term memor"
                  f"{'y' if len(hermes_memory) == 1 else 'ies'} from "
                  f"{hermes_memory._path}")
    else:
        hermes_memory = None
        print("  [Ablation] Hermes persistent memory DISABLED (memory_search: false).")

    hermes = None

    hermes_available = False

    if HERMES_AVAILABLE:

        hermes = HermesIntegration()

        hermes_available = await hermes.initialize_with_tools(

            rag_kb=rag,

            parameter_experience=pe_for_tools,

            simulator=simulator,

            reflection_agent=reflection_agent,

            # Inject the user's full task prompt so the matlab_rl_optimize

            # tool can pass it to the reflection agent on every new RL best

            # — enables Hermes-orchestrated RL to also early-exit on

            # task_requirements_met, persist params, save the model, and

            # generate the report when the user's requirements are met.

            user_prompt=args.prompt,

            ablation=ablation_cfg,

            hermes_agent_memory=hermes_memory,

        )

        if hermes_available:

            print(f"  Hermes initialized with tools (task_prompt auto-injected).")

    

    # ── Layer 3 component creation ─────────────────────────────────────────────
    print("\n[Step 1b] Initializing Layer 3 components (RL optimizer, Judgment, Workflow)...")

    rl_optimizer = MatlabRLOptimizer(
        simulator=simulator,
        parameter_experience=pe_for_tools,
        reflection_agent=reflection_agent,
        hidden_dim=rl_cfg.hidden_dim,
        lr_actor=rl_cfg.lr_actor,
        lr_critic=rl_cfg.lr_critic,
        gamma=rl_cfg.gamma,
        clip_ratio=rl_cfg.clip_ratio,
        max_episodes=rl_cfg.max_episodes,
        episodes_per_update=rl_cfg.episodes_per_update,
        nmc_per_eval=rl_cfg.nmc_per_eval,
        peak_n_max=getattr(rl_cfg, "peak_n_max", 20.0),
        peak_n_penalty=getattr(rl_cfg, "peak_n_penalty", 1.0),
        entropy_coef=getattr(rl_cfg, "entropy_coef", 0.05),
        pitch_log_std=rl_cfg.exploration.pitch_log_std,
        other_log_std=rl_cfg.exploration.other_log_std,
        warm_start_std_scale=rl_cfg.exploration.warm_start_std_scale,
        baseline_reward_threshold=rl_cfg.exploration.baseline_reward_threshold,
        diverge_penalty_episodes=rl_cfg.exploration.diverge_penalty_episodes,
        diverge_log_std_delta=rl_cfg.exploration.diverge_log_std_delta,
        diverge_reward_threshold=rl_cfg.exploration.diverge_reward_threshold,
        adaptive_explore_enabled=rl_cfg.exploration.adaptive_enabled,
        adaptive_steepness_threshold=rl_cfg.exploration.adaptive_steepness_threshold,
        adaptive_log_std_min_scale=rl_cfg.exploration.adaptive_log_std_min_scale,
        adaptive_log_std_max_scale=rl_cfg.exploration.adaptive_log_std_max_scale,
        adaptive_ema_alpha=rl_cfg.exploration.adaptive_ema_alpha,
        reflect_every=rl_cfg.early_stop.reflect_every,
        early_stop_enabled=rl_cfg.early_stop.enabled,
        early_stop_min_episodes=rl_cfg.early_stop.min_episodes,
        early_stop_patience=rl_cfg.early_stop.patience,
        early_stop_plateau_delta=rl_cfg.early_stop.reward_plateau_delta,
        early_stop_use_rule_first=rl_cfg.early_stop.use_rule_first,
        early_stop_use_multi_criteria_best=rl_cfg.early_stop.use_multi_criteria_best,
        early_stop_peak_only_patience=getattr(
            rl_cfg.early_stop, "peak_only_patience", rl_cfg.early_stop.patience
        ),
        fre_config=get_config().fre,
        checkpoint_enabled=rl_cfg.checkpoint.enabled,
        checkpoint_dir=rl_cfg.checkpoint.dir,
        checkpoint_save_every=rl_cfg.checkpoint.save_every_episodes,
        checkpoint_keep_last_n=rl_cfg.checkpoint.keep_last_n,
        rolling_window=rl_cfg.checkpoint.rolling_window,
        stats_log_every=rl_cfg.checkpoint.stats_log_every,
        jsonl_log_dir=rl_cfg.checkpoint.jsonl_log_dir,
        reward_weights=(rl_cfg.reward_weights.to_dict()
                        if hasattr(rl_cfg.reward_weights, "to_dict")
                        else dict(getattr(rl_cfg, "reward_weights", {}))),
        allow_internal_fallback=getattr(rl_cfg, "allow_internal_fallback", False),
    )

    _task_queue = _async_module("task_queue")
    current_async_task_id = _task_queue.current_async_task_id
    make_progress_callback = _task_queue.make_progress_callback

    _async_task_id = current_async_task_id()
    if _async_task_id:
        get_task_queue = _task_queue.get_task_queue
        get_task_queue().update_progress(
            _async_task_id,
            status="running",
            phase="layer3_init",
            message="RL 优化器已初始化",
        )
        rl_optimizer.progress_callback = make_progress_callback(_async_task_id)

    judgment_agent = JudgmentAgent(
        reflection_agent=reflection_agent,
        task_prompt=args.prompt,
    ) if ablation_cfg.reflection_agent else None

    layer3_workflow = OptimizationWorkflow(
        rl_optimizer=rl_optimizer if ablation_cfg.rl_optimization else None,
        judgment_agent=judgment_agent,
        reflection_agent=reflection_agent,
        parameter_experience=parameter_experience,
        output_dir="./guidance_output",
    )
    print(f"  Layer 3 ready: rl={'enabled' if ablation_cfg.rl_optimization else 'disabled'} "
          f"judgment={'enabled' if judgment_agent else 'disabled'} "
          f"reflection={'enabled' if reflection_agent else 'disabled'}")

    current_prompt = args.prompt

    max_iterations = wf_cfg.max_iterations

    from multi_agent.simulation.time_budget import init_run_budget

    _bw_finish_mode = bool(getattr(args, "bw_finish", False))
    _bf_script = ""
    _bf_params: Dict[str, Any] = {}
    _bf_metrics: Dict[str, Any] = {}

    if _bw_finish_mode:
        max_iterations = 1
        _run_budget = init_run_budget(
            getattr(wf_cfg, "bw_finish_budget_sec", 900) or 900,
            label="bw_finish",
        )
        print(
            f"  [BW Finish] 热启动 BW 专扫 | 时限 "
            f"{getattr(wf_cfg, 'bw_finish_budget_sec', 900) // 60} 分钟"
        )
    else:
        _run_budget = init_run_budget(
            getattr(wf_cfg, "run_wall_clock_budget_sec", 0) or None,
            label="cli_agent",
        )
    if _run_budget.enabled and not _bw_finish_mode:
        print(
            f"  [Workflow] 总时限 {wf_cfg.run_wall_clock_budget_sec // 60} 分钟 "
            f"(reserve Layer3/report {getattr(wf_cfg, 'layer3_reserve_sec', 90):.0f}s)"
        )

    if _bw_finish_mode and ablation_cfg.parameter_experience_reuse:
        from multi_agent.integration.bw_finish_path import load_bw_finish_seed

        _ok_bf, _bf_script, _bf_params, _bf_metrics, _bf_msg = await load_bw_finish_seed(
            parameter_experience,
            task_prompt=args.prompt,
            mission_conditions={"T": [4]},
        )
        print(f"  [BW Finish] {_bf_msg}")
        if not _ok_bf:
            if "已全部达标" in _bf_msg:
                return 0
            return 1
        matlab_script_path = _bf_script
        _prev_rl_best_params = dict(_bf_params)
        hermes_sim_metrics = dict(_bf_metrics)

    iteration = 0

    # Clear any cross-iteration history carried over from a previous task
    # so the new task starts with a clean conversation slate.
    if hermes and hasattr(hermes, "reset_conversation"):
        hermes.reset_conversation()

    rl_result = None

    # Holds parsed MATLAB metrics from Hermes's last run_simulation call;
    # populated each loop iteration after hermes.run_with_tools() completes.
    hermes_sim_metrics: Dict[str, Any] = {}

    # Carries the reflection agent's optimization suggestion from the previous
    # iteration into the next Hermes system prompt (Step 0B), so Hermes can
    # factor the previous iteration's advice into its design path decision.
    prev_suggestion: str = ""

    hermes_generated_script: str = ""  # path written by generate_matlab tool

    matlab_script_path: str = ""       # script actually fed to RL
    gate_passed_script: str = ""       # last script that passed Layer2 Gate (iterative seed)

    hermes_modify_law_desc: str = ""   # Chinese task description for MODIFY_LAW

    from multi_agent.integration.t4_low_risk import (
        conservative_autopilot_dict,
        hermes_modify_law_prompt_suffix,
        is_t4_mission,
        resolve_initial_auto_params,
    )

    initial_auto_params = (
        conservative_autopilot_dict()
        if is_t4_mission(task_prompt=args.prompt)
        else {k: float(v["nominal"]) for k, v in ALL_TUNABLE_PARAM_SPECS.items()}
    )

    _prev_rl_best_params: dict = {}  # best params from the last RL run, injected into next iteration
    _optim_history: list = []        # per-iteration metrics summary for reflection trend analysis
    _rl_hp_override: dict = {}       # RL hyperparams suggested by reflection, applied next iteration

    # On startup, warm _prev_rl_best_params from PE so even the very first
    # Hermes call benefits from the best params found in a previous session.
    # Gated by ablation flag: skip when parameter_experience_reuse is disabled.
    if ablation_cfg.parameter_experience_reuse:
        try:
            _pe_startup = await parameter_experience.retrieve_best(
                task_context={"task": "guidance_rl_optimization"},
                top_k=1,
            )
            if _pe_startup:
                _raw = _pe_startup[0].get("parameters", {})
                # PE stores keys as "dp_w1", "dp_zeta1", etc. — strip the prefix.
                _prev_rl_best_params = {
                    k[3:] if k.startswith("dp_") else k: float(v)
                    for k, v in _raw.items()
                    if not k.startswith("phys_")
                }
                if _prev_rl_best_params:
                    print(f"  [Startup] Loaded previous RL best params from PE "
                          f"({len(_prev_rl_best_params)} values, "
                          f"fitness={_pe_startup[0].get('fitness', 0):.4f})")
        except Exception as _pe_startup_exc:
            print(f"  [Startup] PE warm-start failed ({_pe_startup_exc}); using defaults.")
    else:
        print("  [Startup] PE warm-start skipped (parameter_experience_reuse: DISABLED).")

    _pe_quality_info: dict = {}

    iteration_records: list = []



    # Parse user-specified miss distance hint from the prompt (informational only).

    # The reflection agent (not a hard threshold) decides when to stop the RL loop.

    miss_threshold = _parse_miss_threshold(args.prompt)

    if miss_threshold is not None:

        print(f"\n  [Miss Hint] Detected user requirement: miss distance ≤ {miss_threshold:.2f} m")

        print(f"  Note: only the reflection agent's evaluation of ALL stated")

        print(f"        requirements (PM/GM/hit rate/miss/etc.) will trigger early exit.")



    while iteration < max_iterations:

        iteration += 1

        suggestion = ""
        needs_optimization = False

        def _iter_record(outcome: str, force_no_rl: bool = False) -> dict:
            _rr = None if force_no_rl else rl_result
            _bm = (_rr.get("best_metrics", {}) if _rr else {})
            _bp = (_rr.get("best_params",  {}) if _rr else {})
            return {
                "iteration":        iteration,
                "task_mode":        task_mode,
                "plan_reason":      (task_plan.reason if task_plan and getattr(task_plan, "reason", None) else ""),
                "generated_script": (os.path.basename(hermes_generated_script) if hermes_generated_script else ""),
                "modify_law_desc":  hermes_modify_law_desc,
                "rl_status":        (_rr.get("status") if _rr else ""),
                "rl_episodes":      (int(_rr.get("total_episodes", 0)) if _rr else 0),
                "rl_best_reward":   (float(_rr.get("best_reward", 0.0)) if _rr else 0.0),
                "rl_miss":          (float(_bm.get("SEP", _bm.get("miss_distance", 0))) if _rr else None),
                "rl_hit":           (float(_bm.get("hit_rate", 0)) if _rr else None),
                "rl_pm":            (float(_bm.get("pitch_PM", 0)) if _rr else None),
                "rl_bw":            (float(_bm.get("pitch_BW", 0)) if _rr else None),
                "rl_ny":            (float(_bm.get("peak_ny", _bm.get("peak_n", 0))) if _rr else None),
                "best_metrics_iter":_bm,
                "best_params_iter": _bp,
                "suggestion":       suggestion,
                "needs_optimization": needs_optimization,
                "outcome":          outcome,
            }

        cprint(f"\n── Iter {iteration}/{max_iterations} ──")
        dprint(f"\n{'='*40}\nIteration {iteration}/{max_iterations}\n{'='*40}")

        from multi_agent.simulation.time_budget import get_run_budget

        _budget = get_run_budget()
        if _budget and _budget.expired():
            cprint(
                f"  [Workflow] 已达总时限 ({_budget.elapsed_sec() / 60:.1f} min)，停止迭代"
            )
            iteration_records.append(_iter_record("wall_clock_budget"))
            break
        if _budget and _budget.should_stop(
            reserve_sec=float(getattr(wf_cfg, "layer3_reserve_sec", 90.0)),
        ):
            cprint(
                f"  [Workflow] 剩余 {_budget.remaining_sec():.0f}s，不足以下一轮 Layer2+3，停止迭代"
            )
            iteration_records.append(_iter_record("wall_clock_budget"))
            break

        _last_hist_metrics = (
            (_optim_history[-1].get("metrics") or {})
            if _optim_history else {}
        )

        # 2. Task Planning & Parameter Extraction

        print("\n[Step 2] Task Planning & Parameter Extraction...")

        extracted_params = {}

        task_plan = None

        planner = None

        if hermes and hermes_available:

            planner = IntelligentTaskPlanner(

                hermes=hermes,

                rag_kb=rag,

                # Use pe_for_tools (None when parameter_experience_reuse=false)
                # so the planner skips PE retrieval when the ablation flag is off.
                parameter_experience=pe_for_tools,

            )

            # Determine which .m script to analyze for algorithm structure:
            # - First iteration: KB template (monte_carlo_single.m)
            # - Later iterations: previous round's generated .m file
            _planner_script = (
                gate_passed_script
                or matlab_script_path
                or hermes_generated_script
                or getattr(RunSimulationTool, "KB_TEMPLATE_PATH",
                           "./knowledge_base/matlab/guidance/monte_carlo_single.m")
            )

            task_plan = await planner.analyze_and_plan(
                current_prompt,
                reflection_feedback=prev_suggestion,
                prev_rl_best_params=_prev_rl_best_params or None,
                script_path=_planner_script,
                optimization_history=_optim_history or None,
                last_metrics=_last_hist_metrics or None,
            )

            # ── Step 0A result: 任务工况分析 (performed BEFORE mode selection) ──
            # The planner now derives mission conditions via LLM grounded in the
            # MC_gongkuang_simulation_robust_all reference, then selects the
            # design-path mode with that analysis injected into the prompt.
            if task_plan.mission_analysis:
                cprint(f"  工况: {truncate_text(task_plan.mission_analysis, 120)}")
                dprint(f"  工况分析 : {task_plan.mission_analysis[:300]}")
            if task_plan.mission_conditions:
                dprint(f"  工况映射 : {task_plan.mission_conditions}")
                # Planner's LLM-derived conditions override the upstream regex
                # parse so Step 2.5 (Hermes generate_matlab) and Step 3 (RL)
                # both run on the same set of categories the design-path mode
                # was chosen against.
                mission_conditions = dict(task_plan.mission_conditions)

            cprint(
                f"  Plan: {task_plan.strategy.value} | mode={task_plan.mode} | "
                f"subtasks={task_plan.subagent_count} | {truncate_text(task_plan.reason, 180)}"
            )
            dprint(f"  Reason : {task_plan.reason}")

            if task_plan.retrieval_context:

                _ctx_len = len(task_plan.retrieval_context)
                dprint(f"  Context ({_ctx_len} chars total, preview):\n{task_plan.retrieval_context[:600]}")

            await parameter_experience.store_event(
                "plan",
                f"iter={iteration} mode={task_plan.mode} reason={task_plan.reason[:150]}",
                metadata={"iteration": iteration, "mode": task_plan.mode},
            )



            # Dynamic parameter extraction: prefer gate-passed / Layer3 script
            if not matlab_script_path:
                matlab_script_path = (
                    gate_passed_script
                    or _find_latest_matlab_script("./matlab_scripts", "./guidance_output", ".")
                    or ""
                )

            if matlab_script_path:

                print(f"  Dynamically extracting params from MATLAB script: {matlab_script_path}")

                try:

                    with open(matlab_script_path, "r", encoding="utf-8", errors="ignore") as fh:

                        script_content = fh.read()

                    # extract_autopilot_params now returns all 11 tunable params

                    auto_params = extract_autopilot_params(script_content)

                    phys_params = extract_physical_params(script_content)

                    # Map to GuidanceParameters-compatible dict (all values as float)

                    N_g = float(auto_params.get("N_pn", auto_params.get("N_guidance",
                        GUIDANCE_PARAM_SPECS["N_pn"]["nominal"])))

                    extracted_params = {

                        # Guidance law params

                        "navigation_coefficient": N_g,

                        "guidance_N":             N_g,

                        "guidance_R_switch":      float(auto_params.get("R_switch", 7000.0)),

                        "guidance_gama_max_deg":  float(auto_params.get("gama_max_deg", 45.0)),

                        # Autopilot params (dp.* in MATLAB template)

                        "autopilot_w1":           float(auto_params.get("w1",    AUTOPILOT_PARAM_SPECS["w1"]["nominal"])),

                        "autopilot_zeta1":        float(auto_params.get("zeta1", AUTOPILOT_PARAM_SPECS["zeta1"]["nominal"])),

                        "autopilot_tao1":         float(auto_params.get("tao1",  AUTOPILOT_PARAM_SPECS["tao1"]["nominal"])),

                        "autopilot_w2":           float(auto_params.get("w2",    AUTOPILOT_PARAM_SPECS["w2"]["nominal"])),

                        "autopilot_zeta2":        float(auto_params.get("zeta2", AUTOPILOT_PARAM_SPECS["zeta2"]["nominal"])),

                        "autopilot_tao2":         float(auto_params.get("tao2",  AUTOPILOT_PARAM_SPECS["tao2"]["nominal"])),

                        "autopilot_w3":           float(auto_params.get("w3",    AUTOPILOT_PARAM_SPECS["w3"]["nominal"])),

                        "autopilot_zeta3":        float(auto_params.get("zeta3", AUTOPILOT_PARAM_SPECS["zeta3"]["nominal"])),

                    }

                    print(f"  Dynamically extracted {len(extracted_params)} params from script "

                          f"(guidance: N={N_g:.2f}, R_sw={extracted_params['guidance_R_switch']:.0f}m, "

                          f"gama_max={extracted_params['guidance_gama_max_deg']:.0f}deg).")

                except Exception as exc:

                    print(f"  Script param extraction failed ({exc}); will try LLM fallback.")



            # LLM fallback extraction if script not available

            if not extracted_params and hermes and hermes_available:

                engine = get_config().simulation.engine

                llm_client = hermes.agent if hasattr(hermes, "agent") else None

                if llm_client:

                    try:

                        ctx_hint = (task_plan.retrieval_context[:500] if task_plan and task_plan.retrieval_context else "")

                        prompt_extract = (

                            f"从以下提示词中提取制导系统的初始参数，返回JSON格式。\n"

                            f"任务：{current_prompt}\n"

                            f"{('参考上下文：' + ctx_hint) if ctx_hint else ''}\n"

                            f'返回格式示例：{{"navigation_coefficient":3.5,"damping_ratio":0.75}}'

                        )

                        import inspect

                        if hasattr(llm_client, "generate"):

                            response = await llm_client.generate(prompt=prompt_extract)

                        elif hasattr(llm_client, "run_conversation"):

                            if inspect.iscoroutinefunction(llm_client.run_conversation):

                                response = await llm_client.run_conversation(prompt_extract)

                            else:

                                response = llm_client.run_conversation(prompt_extract)

                            if isinstance(response, dict):

                                response = response.get("final_response", "")

                        if isinstance(response, str):

                            json_str = re.sub(r"```(?:json)?", "", response).strip().strip("`")

                            extracted_params = {k: float(v) for k, v in json.loads(json_str).items()

                                                if isinstance(v, (int, float))}

                            print(f"  LLM-extracted params: {extracted_params}")

                    except Exception as exc:

                        print(f"  LLM param extraction also failed: {exc}")

                    

        else:

            print("  Hermes not available. Skipping task planning.")



        # 2.5 Hermes content generation (MATLAB model creation / guidance law modification)

        from multi_agent.integration.design_path_policy import resolve_task_mode

        # When the previous iteration explicitly requested a guidance-law rewrite,
        # do NOT force tune_params — the modify_law intent must be honoured.
        _prev_next_action_for_forced = (
            str(_optim_history[-1].get("next_action", "")).lower()
            if _optim_history
            else ""
        )
        _forced_next = (
            "tune_params"
            if (
                iteration > 1
                and bool(getattr(wf_cfg, "force_tune_fast_path_iter2", True))
                and matlab_script_path
                and os.path.isfile(matlab_script_path)
                and _prev_next_action_for_forced != "modify_law"  # never override modify_law
            )
            else (
                _optim_history[-1].get("next_action")
                if _optim_history
                else None
            )
        )
        task_mode = resolve_task_mode(
            planner_mode=task_plan.mode if task_plan else None,
            task_prompt=args.prompt or current_prompt,
            optimization_history=_optim_history or None,
            last_metrics=_last_hist_metrics if _optim_history else None,
            reflection_feedback=prev_suggestion,
            forced_next_action=_forced_next if iteration > 1 else None,
        )
        if (
            iteration > 1
            and bool(getattr(wf_cfg, "force_tune_fast_path_iter2", True))
            and matlab_script_path
            and os.path.isfile(matlab_script_path)
            and task_mode == "TUNE_PARAMS"
        ):
            cprint(
                "  [DesignPath] Iter2 force_tune_fast_path → TUNE_PARAMS "
                "(skip Hermes MODIFY)"
            )
        if task_plan and task_mode != task_plan.mode:
            cprint(
                f"  [DesignPath] Planner mode {task_plan.mode} → enforced {task_mode} "
                f"(PeakNy mean / next_action policy)"
            )

        # Mission analysis already performed by the planner (Step 0A) so Hermes
        # only needs to verify it, not re-derive from scratch.
        planner_mission_analysis = (
            task_plan.mission_analysis if task_plan and task_plan.mission_analysis else ""
        )
        planner_conditions_md_ref = (
            task_plan.conditions_md_ref if task_plan and task_plan.conditions_md_ref else ""
        )

        # Build a fully-explicit MATLAB conditions string so every category is

        # given an unambiguous true/false flag (unmentioned ones → false).

        dprint(f"  [DEBUG] mission_conditions before build = {mission_conditions}")
        matlab_cond_str = build_matlab_conditions_str(mission_conditions)

        cprint(f"  Conditions: {matlab_cond_str}")

        # Push conditions into every Hermes tool that accepts them so that
        # tool calls which omit mission_conditions never fall back to the
        # template's all-categories-enabled defaults (→ timeout).
        if hermes and hermes_available and hasattr(hermes, "update_mission_conditions"):
            hermes.update_mission_conditions(matlab_cond_str)

        # Default values — overwritten inside the Hermes block when available
        hermes_sim_metrics: Dict[str, Any] = {}
        hermes_directly_satisfied: bool = False
        hermes_generated_script: str = ""

        _prev_next_action = (
            str(_optim_history[-1].get("next_action", "")).lower()
            if _optim_history
            else ""
        )
        from multi_agent.integration.modify_law_fast_path import (
            run_modify_law_t4_fast_path,
            should_use_modify_law_t4_fast_path,
        )
        _tune_only_skip = (
            bool(getattr(wf_cfg, "tune_only_skip_hermes", True))
            and iteration > 1
            and task_mode == "TUNE_PARAMS"
            and matlab_script_path
            and os.path.isfile(matlab_script_path)
            and (
                _prev_next_action == "tune_params"
                or bool(getattr(wf_cfg, "force_tune_fast_path_iter2", True))
            )
        )
        if _tune_only_skip:
            print(
                "\n[Step 2.5] tune_params — 跳过 Hermes 改律，复用上轮脚本直进 Layer3"
            )
            hermes_generated_script = matlab_script_path
            hermes_sim_metrics = dict(_optim_history[-1].get("metrics") or {})
            hermes_directly_satisfied = False
        elif should_use_modify_law_t4_fast_path(
            task_mode=task_mode,
            iteration=iteration,
            mission_conditions_str=matlab_cond_str,
            task_prompt=current_prompt,
            enabled=bool(getattr(wf_cfg, "modify_law_t4_fast_path", True)),
        ):
            print(
                "\n[Step 2.5] T4 MODIFY_LAW 快路径 "
                "(跳过 Hermes 编排/子 Agent，直接 generate→sim→judge)..."
            )
            _fp = await run_modify_law_t4_fast_path(
                mission_conditions_str=matlab_cond_str,
                task_prompt=current_prompt,
                autopilot_params=_prev_rl_best_params or None,
                simulator=simulator,
                nmc=int(rl_cfg.nmc_per_eval),
            )
            hermes_generated_script = _fp.get("script_path", "")
            hermes_sim_metrics = dict(_fp.get("metrics") or {})
            hermes_directly_satisfied = bool(_fp.get("satisfied"))
            if hermes_generated_script:
                cprint(
                    f"  [FastPath script] "
                    f"{os.path.basename(hermes_generated_script)}"
                )
            if hermes_sim_metrics:
                cprint(
                    f"  [FastPath sim metrics] "
                    f"{metrics_summary_line(hermes_sim_metrics)}"
                )
            await parameter_experience.store_event(
                "hermes_sim",
                f"iter={iteration} fast_path "
                f"{metrics_summary_line(hermes_sim_metrics)}",
                metadata={"iteration": iteration, "fast_path": True},
            )
        elif hermes and hermes_available and wf_cfg.hermes_execution:

            print(f"\n[Step 2.5] Hermes Content Generation (mode={task_mode})...")

            # ── Inject analysis context into generate_matlab tool ─────────
            # Ensures the LLM in _rewrite_guidance_law always sees the
            # accumulated suggestions regardless of what Hermes passes as
            # task_description.
            _gen_tool = next(
                (t for t in (getattr(hermes, "_tools", []) or [])
                 if getattr(t, "name", "") == "generate_matlab"),
                None,
            )
            if _gen_tool and hasattr(_gen_tool, "set_analysis_context"):
                from multi_agent.config_loader import get_config as _gc_seed
                from multi_agent.integration.script_seed_policy import pick_cli_seed_path

                _seed_path = pick_cli_seed_path(
                    task_mode=task_mode,
                    gate_passed_script=gate_passed_script,
                    layer3_script=matlab_script_path,
                    hermes_script=hermes_generated_script,
                    seed_policy=str(getattr(_gc_seed().simulation, "seed_policy", "iterative")),
                )
                if hasattr(_gen_tool, "set_iterative_seed"):
                    _gen_tool.set_iterative_seed(_seed_path)
                    if _seed_path:
                        print(
                            f"  [generate_matlab] Iterative seed: "
                            f"{os.path.basename(_seed_path)}"
                        )
                _ctx_parts = []
                if prev_suggestion:
                    _ctx_parts.append(f"[上轮反思建议]\n{prev_suggestion}")
                if _prev_rl_best_params:
                    import json as _json_ac
                    _ctx_parts.append(
                        f"[上轮最优参数]\n{_json_ac.dumps(_prev_rl_best_params, ensure_ascii=False)}"
                    )
                if task_mode == "MODIFY_LAW":
                    _t4_suffix = hermes_modify_law_prompt_suffix(
                        mission_conditions=matlab_cond_str,
                        task_prompt=current_prompt,
                    )
                    # Inject root-cause analysis: the 60g overload is NOT from guidance
                    # command saturation alone — it comes from autopilot transient overshoot
                    # at initial engagement when both channels simultaneously demand large
                    # maneuvers.  Command filtering (Iter2/3) didn't help because the actual
                    # body acceleration is aerodynamically driven.  The fix must address:
                    # (1) command RATE limiting (d/dt N1, N2) to prevent step transitions, OR
                    # (2) initial hold-off period (t < t_hold → zero command) to let autopilot
                    #     stabilise before guidance activates, OR
                    # (3) combined-channel vector limit already applied in sim_s() — gf() should
                    #     also respect base.ny_lim passed as Np-struct extension or persistent param.
                    _prev_peak = max(
                        (
                            float(h.get("peak_n") or h.get("metrics", {}).get("peak_ny", 0) or 0)
                            for h in (_optim_history or [])
                        ),
                        default=0.0,
                    )
                    _modify_rootcause = (
                        "\n[MODIFY_LAW 物理根因分析]\n"
                        f"历史最高实测 PeakNy ≈ {_prev_peak:.0f}g（要求 ≤20g）。"
                        "过载超标的根本原因是驾驶仪在初始交战时跟踪大幅值指令时产生的瞬态过冲，"
                        "而非制导指令限幅不足。\n"
                        "上轮 MODIFY_LAW（指令低通滤波+N调度）未降低过载，说明问题在时间域响应速率：\n"
                        "  ① 必须加入指令变化率限制 dN/dt_max（如 5~10g/s），阻止指令阶跃；\n"
                        "  ② 或加入初始制导冻结期（t<0.3s 输出0指令，等驾驶仪稳定）；\n"
                        "  ③ gf()内部需引用 base.ny_lim 做向量幅值限幅（已由sim_s层补充）。\n"
                        "本轮 gf() 修改的核心是'速率限制'而非'幅值限制'。\n"
                    )
                    _ctx_parts.append(
                        "[DesignPath] 本轮必须 MODIFY_LAW：修改 gf() 制导律"
                        "（APN/增广比例导引、N 调度、过载限幅/低通），"
                        "禁止仅调 autopilot 参数。"
                        + _modify_rootcause
                        + _t4_suffix
                    )
                _gen_tool.set_analysis_context("\n\n".join(_ctx_parts))
                if _ctx_parts:
                    print(f"  [generate_matlab] Injected {len(_ctx_parts)} analysis context block(s)")

            ctx = task_plan.retrieval_context if task_plan else ""

            # Build episodic session-history block for Step 0B context injection
            _episodic_ctx = ""
            _recent_evts = (
                parameter_experience.get_recent_events()
                if ablation_cfg.memory_search else ""
            )
            if _recent_evts:
                _episodic_ctx = (
                    "\n【本次会话历史事件（最近 3 类：plan / hermes_sim / reflection）】\n"
                    f"{_recent_evts}\n"
                )

            _step0b_text = (
                f"  【上一轮优化建议】{prev_suggestion}\n"
                f"  上述建议仅在明确指出需切换模式时才调整；否则继续按 mode={task_mode} 执行。\n"
                f"{_episodic_ctx}\n"
                if prev_suggestion else
                f"  按下方对应模式的指令执行工具调用。\n{_episodic_ctx}\n"
            )

            # ── Pre-compute RUN_CASE / SUB_IDX for pe_best task_context ─────
            _hmc_keys = sorted(mission_conditions.keys())
            _h_rc = _hmc_keys[0] if len(_hmc_keys) == 1 else "ALL"
            _h_si_list = mission_conditions.get(_h_rc, [0]) if _h_rc != "ALL" else [0]
            _h_si = _h_si_list[0] if len(_h_si_list) == 1 else 0

            _memory_ctx = hermes_memory.get_context_block() if hermes_memory else ""

            # ── Load monte mean results directly from file ─────────────────────
            # This is the authoritative baseline simulation data.  Injecting it
            # directly into the system prompt guarantees the Hermes agent always
            # sees it regardless of whether RAG retrieval happens to score it.
            _monte_mean_content = ""
            import pathlib as _pl_h
            _mm_candidates = [
                _pl_h.Path("knowledge_base/monte mean results.md"),
                _pl_h.Path("../knowledge_base/monte mean results.md"),
                _pl_h.Path(__file__).parent / "knowledge_base" / "monte mean results.md",
            ]
            for _mmp in _mm_candidates:
                try:
                    if _mmp.exists():
                        _monte_mean_content = _mmp.read_text(encoding="utf-8").strip()
                        break
                except Exception:
                    pass

            _monte_mean_block = (
                "\n[基线仿真结果 — monte mean results.md (全工况 N=100，最高事实优先级)]\n"
                f"{_monte_mean_content}\n"
                "  ► 在制定参数调整方案前，必须先查阅上表中对应子工况的实测指标，以数据事实为依据。\n\n"
            ) if _monte_mean_content else ""

            _iter_ctx_block = ""
            if prev_suggestion or _prev_rl_best_params:
                import json as _json_h
                _iter_hdr = (
                    f"\n{'═'*44}\n"
                    f"【迭代上下文 — 第 {iteration} 轮，优先级高于基线仿真结果和专家经验】\n"
                    f"{'═'*44}\n"
                    "⚠️ 以下来自上一轮的优化结论，是本轮分析的第一优先级依据，\n"
                    "   必须在参考 monte_mean_results / expert_design_path 之前先满足这些约束。\n"
                )
                _iter_suggest = (
                    f"\n  [上轮优化建议 — 最高优先级，本轮首先满足]\n  {prev_suggestion}\n"
                ) if prev_suggestion else ""
                _iter_params = (
                    f"\n  [上轮 RL 全局最优参数 — 直接用作 autopilot_params，不得重新推导]\n"
                    f"  {_json_h.dumps(_prev_rl_best_params, ensure_ascii=False)}\n"
                ) if _prev_rl_best_params else ""
                _iter_ctx_block = _iter_hdr + _iter_suggest + _iter_params + "\n"
            _prev_rl_block = _iter_ctx_block  # keep alias for backward compat

            _pe_on = ablation_cfg.parameter_experience_reuse
            _analysis_desc = "  analysis_agent → requirements judgment + parameter experience\n" if _pe_on else "  analysis_agent → requirements judgment\n"
            _direct_tools_line = (
                "Direct tools: rag_query, rag_retrieve, rag_expand, parameter_experience_best,\n"
                "  parameter_experience_search, generate_matlab, syntax_check_matlab,\n"
            ) if _pe_on else (
                "Direct tools: rag_query, rag_retrieve, rag_expand, generate_matlab, syntax_check_matlab,\n"
            )
            # REUSE_HISTORY PE-conditional fragments
            _rh_model_prio = (
                "    ① PE历史模型(.m) → ② guidance_output/scripts/ → ③ KB模板\n"
                if _pe_on else
                "    ① guidance_output/scripts/ → ② KB模板\n"
            )
            _rh_step2 = (
                f"  2. parameter_experience_best(task_context={{\"task\":\"guidance_rl_optimization\",\"RUN_CASE\":\"{_h_rc}\",\"SUB_IDX\":{_h_si}}},\n"
                "       param_ranges=<step1提取的范围字典>)\n"
                "     — 有匹配：使用返回的 parameters 字段；无匹配：可省略 dp_params（工具自动加载历史参数）\n"
            ) if _pe_on else (
                "  2. 跳过（参数经验复用已禁用），直接使用 rag_query 提取的范围推导初始值\n"
            )
            _rh_script_hint = (
                "     ⚠️ 不要传 script_path — 工具自动从PE历史模型/scripts文件夹/KB模板中选择最佳脚本。\n"
                if _pe_on else
                "     ⚠️ 不要传 script_path — 工具自动从scripts文件夹/KB模板中选择最佳脚本。\n"
            )
            # Memory/session guidance — gated by memory_search ablation flag
            _mem_on = ablation_cfg.memory_search
            _memory_guidance = (
                "MEMORY GUIDANCE: Use agent_memory_remember to persist important findings\n"
                "  (best params, design decisions, task requirements met/not met) so they\n"
                "  are available across future sessions.\n"
                "SESSION SEARCH GUIDANCE: At the very start of any task, call session_search\n"
                "  with the working condition (e.g. 'T1 匀速直飞 PM BW 调参') to retrieve\n"
                "  relevant past sessions.  If a matching session is found, extract the best\n"
                "  parameter combination and use it as the starting point instead of the\n"
                "  template default — this avoids re-deriving from scratch.\n"
            ) if _mem_on else (
                "MEMORY GUIDANCE: ⚠️ 持久记忆已禁用（memory_search 消融实验）。\n"
                "  禁止使用 agent_memory_recall / agent_memory_list / session_search。\n"
                "  禁止使用任何历史记忆中的参数或设计决策。\n"
                "  所有参数必须仅基于 RAG 知识库检索结果和当前工况分析从头推导。\n"
            )
            # TUNE_PARAMS PE-conditional fragment
            _tp_step2 = (
                f"  2. parameter_experience_best(task_context={{\"task\":\"guidance_rl_optimization\",\"RUN_CASE\":\"{_h_rc}\",\"SUB_IDX\":{_h_si}}},\n"
                "       param_ranges=<step1提取的范围字典>)\n"
                "     — 有匹配：使用返回的 parameters 字段作为初始参数\n"
                "     — 无匹配：根据工况与 rag_query 结果自行推导合理初始值\n"
            ) if _pe_on else (
                "  2. 跳过（参数经验复用已禁用），根据工况与 rag_query 结果自行推导合理初始值\n"
            )

            hermes_system = (

                "You are a missile guidance system orchestrator. Your PRIMARY tool is:\n"
                "  call_specialist(agent_type, task, context) — route work to a specialized sub-agent.\n"
                "Sub-agent types (agent_type parameter):\n"
                "  rag_agent      → knowledge retrieval (rag_query/rag_retrieve/rag_expand)\n"
                "  matlab_agent   → MATLAB generation + syntax check + compat proof\n"
                "  sim_agent      → simulation execution (run_simulation + syntax fix)\n"
                f"{_analysis_desc}"
                "  memory_agent   → persistent memory read/write\n"
                "  hermes         → full-toolset fallback (use when agent_type is unclear)\n"
                "You MAY also call tools directly when a single-step action is simpler than delegation.\n"
                f"{_direct_tools_line}"
                "  verify_guidance_compat, run_simulation, judge_requirements,\n"
                "  agent_memory_remember, agent_memory_recall, agent_memory_list, agent_memory_forget,\n"
                "  session_search.\n"
                "NOTE: RL parameter optimisation is handled automatically by Layer 3 —\n"
                "  do NOT attempt to call any rl_optimize tool; it does not exist here.\n"
                f"{_memory_guidance}"
                "DELEGATE GUIDANCE:\n"
                "  Preferred pattern — use call_specialist with agent_type for specialized work:\n"
                "    call_specialist(agent_type='rag_agent', task='检索制导律知识库：增强PN导引律设计原理')\n"
                "    call_specialist(agent_type='matlab_agent', task='生成APN制导律脚本，工况G3，N=4', context='<RAG结果>')\n"
                "    call_specialist(agent_type='sim_agent', task='运行仿真 script_path=<路径> mission_conditions=\"G:3\"')\n"
                "    call_specialist(agent_type='analysis_agent', task='判断是否满足PM≥30°、命中率≥90%', context='<仿真结果>')\n"
                "    call_specialist(agent_type='memory_agent', task='记住本轮最优参数：N=4 tao1=0.20 PM=45°')\n"
                "  When ≥2 sub-tasks are INDEPENDENT (no output dependency), delegate them in sequence\n"
                "  and synthesize results before the next dependent step.\n"
                "  For MODIFY_LAW, the preferred full-delegation flow is:\n"
                "    rag_agent → matlab_agent (includes compat proof) → sim_agent → analysis_agent\n\n"

                + _memory_ctx

                + f"Current task mode: {task_mode}\n\n"

                + _iter_ctx_block

                + _monte_mean_block

                + "════════════════════════════════════════\n"
                "【书籍知识库 RAG — 设计全程必须执行（book010《系统化飞控实战·导弹制导》）】\n"
                "════════════════════════════════════════\n"
                "系统内置一部蒸馏书籍知识库 knowledge_base/books/books/book010/，\n"
                "是《系统化飞控实战》010 本（20 门课 × 30 章 = 600 章）的 30 张 RIA++ 卡归档。\n"
                "设计任何制导律 / 整定参数前，必须先按下方路由表用 RAG 检索，\n"
                "把检索到的【理论 / 公式 / 经验 / 红线】落地到设计决策，禁止凭空拍参数。\n\n"
                "【5 大知识域 → 卡片文件】\n"
                "  A 比例导引 PN     → domain_A_PN.md           （含 APN / 预测制导融合）\n"
                "  B 最优制导        → domain_B_Optimal.md      （LQR/Riccati、N_opt=3 的由来）\n"
                "  C 终端制导        → domain_C_Terminal.md     （视场/落角/能量约束）\n"
                "  D 系统建模仿真    → domain_D_SystemModeling.md（稳定性/整定/蒙特卡洛/抗干扰）\n"
                "  E 飞行器轨迹控制  → domain_E_Trajectory.md   （规划/跟踪/辨识/鲁棒）\n\n"
                "【结构化数据资产 → 文件（可跨文档引用编号）】\n"
                "  assets/00_INDEX.md       资产总索引（按问题检索路由，先看它）\n"
                "  assets/01_theory.md      理论体系（30 卡【R】段）\n"
                "  assets/02_formulas.md    公式集 197 条（编号 F-xx-xx）\n"
                "  assets/03_heuristics.md  经验集 439 条（编号 H001…，注明出处卡号）\n"
                "  assets/04_parameters.md  工程参数速查表（9 大类：推荐值/调节方向/红线/出处）\n"
                "  assets/05_checklist.md   7 阶段设计与仿真检查清单（⚠️=作者踩过的坑）\n\n"
                "【按设计问题检索路由表】\n"
                "  选制导律类型 (PN/APN/OGL/滑模/预测)  → A-05/A-06 + B 域 + C 域\n"
                "  定导航比 N                         → A-02 + 04_parameters.md §1\n"
                "  稳定性红线 / 失稳机理              → A-02(N>2) + D-02(寄生回路)\n"
                "  脱靶量 / 蒙特卡洛验收              → A-04 + D-05 + B-05(样本量/分位数)\n"
                "  视线角速率符号 / 滤波              → A-03 + A-06(λ̇ 绝不差分)\n"
                "  驾驶仪参数整定 (w/ζ/τ)            → D-03 + 04_parameters.md\n"
                "  视场 / 框架角约束                  → C-01 / C-02\n"
                "  落角 / 攻角约束                    → C-03\n"
                "  抗干扰                             → D-06（全库经验最密）\n"
                "  交付前自检                         → 05_checklist.md（7 阶段）\n\n"
                "【必须检索并遵守的书籍红线（校勘注记 + 超书发现）】\n"
                "  1. N>2 是稳定性必要条件，N<2 发散（红线）；LQR 解析给出 N_opt=3，\n"
                "     但有延迟+噪声+机动时最优 N 呈 U 型，N≈6（不是 3）。\n"
                "  2. λ̇ 绝不能差分，必须滤波；λ̇ 滤波器存在最优时间常数 ≈20 ms。\n"
                "  3. 舵机延迟是头号杀手：τ≤50ms 无碍，τ=300ms 脱靶飙到 7.5 m。\n"
                "  4. 碰撞三角初始化：导弹初速必须落在碰撞三角上，否则\"零控\"不零。\n"
                "  5. 蒙特卡洛按分位数验收（P95），别只看均值（P95/均值≈3.4）。\n"
                "  6. APN 性能上限 = 目标加速度估计精度；误差>50% 反不如 PN，需置信度门限退化。\n"
                "  7. 限幅要\"双限\"（幅值 + 速率），留 20~30% 过载裕度。\n"
                "  8. 变导航比 / 多阶段切换不得阶跃，需 0.5~1.5 s 线性过渡区。\n\n"
                "【检索工具与查询模板】\n"
                "  多跳优先 rag_query(query=..., max_hops=2)；单点用 rag_retrieve；读全文用 rag_expand。\n"
                "    rag_query(query=\"增强比例导引 APN 目标加速度补偿 N 选取 稳定性 红线 参数范围\")\n"
                "    rag_retrieve(query=\"navigation ratio N selection delay noise maneuver optimum\", top_k=5)\n"
                "    rag_expand(doc_id=\"file:.../domain_A_PN.md\")   # 分页读卡片全文\n\n"
                "【落地要求】检索结果必须体现在：\n"
                "  ① 设计思路：law_type + 依据哪张卡 / 公式号 F-xx / 经验号 Hxxx\n"
                "  ② 参数初值：引用 04_parameters.md 的推荐值与红线\n"
                "  ③ 稳定性说明：引用 N>2、带宽分离、裕度红线\n"
                "  ④ 验收标准：引用 B-05/D-05 的样本量与分位数\n"
                "  ⑤ 交付前对照 05_checklist.md 逐项自检\n\n"

                + "════════════════════════════════════════\n"
                "【强制分析步骤 — 在任何工具调用前必须完成】\n"
                "════════════════════════════════════════\n\n"

                "Step 0A: 任务工况分析结果（已由任务规划器完成，参考 knowledge_base + monte_carlo_single.m）\n"
                + (
                    f"  [conditions.md 检索内容]:\n{planner_conditions_md_ref[:600]}\n"
                    if planner_conditions_md_ref else
                    "  工况体系 (T=目标机动 G=交战几何 AP=驾驶仪退化 R=综合鲁棒)\n"
                )
                + (
                    f"  规划器分析: {planner_mission_analysis}\n"
                    if planner_mission_analysis else
                    "  规划器未给出显式工况分析，请按下方 MISSION CONDITIONS 执行。\n"
                )
                + f"  规划器映射类别: {dict(mission_conditions)}\n"
                "  若你认为规划器结果与任务有出入，可调用 rag_retrieve\n"
                "  (query=\"conditions.md 工况分类 T G AP R 子工况\")\n"
                "  核对原始定义后修正；否则直接沿用上述映射。\n\n"

                f"Step 0B: 设计路径（已由任务规划器基于工况分析确定）\n"
                f"  ✅ 本轮执行模式：mode = {task_mode}（规划器综合 conditions.md 工况分析结果后确定，请严格执行）\n"
                f"{_step0b_text}"

                "MISSION CONDITIONS (use exactly as-is when calling generate_matlab):\n"

                f"  mission_conditions = \"{matlab_cond_str}\"\n"

                "  Sets RUN_CASE and SUB_IDX for monte_carlo_single.m.\n"

                "  Do NOT modify these values.\n"

                "\n"

                "════════════════════════════════════════\n"
                "【通用仿真流程规则（所有模式均适用）】\n"
                "════════════════════════════════════════\n"
                "规则1 — 生成脚本后必须先语法校正：\n"
                "  generate_matlab 返回脚本路径后，立即调用 syntax_check_matlab\n"
                "  (script_path=<返回的脚本路径>)，用校正后的路径再仿真。\n"
                "规则2 — 仿真出错时语法校正恢复：\n"
                "  run_simulation 返回 status='error' 时，调用 syntax_check_matlab\n"
                "  修正后再次调用 run_simulation（最多重试 1 次）。\n"
                "规则3 — 仿真成功后立即判断指标：\n"
                "  run_simulation 返回 status='success' 时，立即调用 judge_requirements\n"
                "  (task_prompt=<用户任务>, metrics=<仿真指标>)。\n"
                "规则4 — judge_requirements 结果决定 Layer3 路径：\n"
                "  satisfied=True  → 输出最终摘要并 STOP\n"
                "                    Layer3 直接进反思智能体（跳过 RL 优化）\n"
                "  satisfied=False → 输出最终摘要并 STOP\n"
                "                    Layer3 进入 RL 参数优化环节\n"
                "  无论哪种结果，你的任务到此结束，不要再调用任何工具。\n"
                "规则5 — MODIFY_LAW 适配性验证失败时必须重新设计制导律（强制）：\n"
                "  verify_guidance_compat 返回 overall='FAIL' 时：\n"
                "    ① 阅读返回的 critical_issues 和 recommendations 字段\n"
                "    ② 必须重新调用 generate_matlab(mode=MODIFY_LAW)，\n"
                "       在 task_description 中明确说明要修正的数学问题\n"
                "    ③ 重新调用 syntax_check_matlab → verify_guidance_compat\n"
                "    ④ 最多重试 2 次；2 次后仍 FAIL 则记录失败原因后 STOP\n"
                "  overall='WARN' 时：可直接继续调用 run_simulation（风险可接受）\n"
                "  overall='PASS' 时：直接继续调用 run_simulation\n"
                "\n"
                "\n"
                "  也可以直接调用工具（不委托子 Agent），两种方式等价。\n"
                "\n"
                "  Step 5 — 输出摘要并 STOP — MODIFY_LAW 摘要必须包含：\n"
                "       ① 制导律设计思路（law_type + 修改了什么、为什么）\n"
                "       ② 适配性验证结果（compat_overall + compat_score + 关键维度）\n"
                "       ③ 仿真指标（hit_rate/SEP/PeakNy/PM/BW）\n"
                "       ④ satisfied=True/False + layer3_path 说明\n"
                "\n"
                "REUSE_HISTORY:\n"
                "  核心思路：直接利用历史经验参数或历史模型文件，跳过脚本生成，直接仿真验证。\n"
                "  run_simulation 会自动按以下优先级查找最佳模型：\n"
                f"{_rh_model_prio}"
                "  同时自动加载该模型关联的历史最优参数。\n\n"
                "  1. rag_query(query=<工况描述>) — Agentic检索 conditions.md / expert design path.md\n"
                "     从返回内容中提取当前工况下各参数的合理范围，例如：\n"
                "     {\"w1\":[20,50], \"zeta1\":[0.5,0.9], \"N_pn\":[3,5], ...}\n"
                f"{_rh_step2}"
                f"  3. run_simulation(dp_params=<step2返回参数，有则传，无则省略>,\n"
                f"       mission_conditions=\"{matlab_cond_str}\", nmc={rl_cfg.nmc_per_eval})\n"
                f"{_rh_script_hint}"
                "     若 status='error'：调用 syntax_check_matlab → run_simulation\n"
                "     注意：不要调用 generate_matlab，直接复用历史模型。\n"
                "  4. judge_requirements(task_prompt=<用户任务>, metrics=<仿真指标>)\n"
                "  5. 输出摘要并 STOP（见规则4）\n"
                "\n"
                "TUNE_PARAMS:\n"
                "  1. rag_query(query=<任务描述>) — Agentic检索 conditions.md / expert design path.md\n"
                "     从返回内容中提取当前工况下各参数的合理范围，例如：\n"
                "     {\"w1\":[20,50], \"zeta1\":[0.5,0.9], \"N_pn\":[3,5], ...}\n"
                f"{_tp_step2}"
                "     ⚠️ 无论有无历史记录，都必须将最终参数 dict 赋给变量 INIT_PARAMS，\n"
                "        并在 step4 中以 autopilot_params=INIT_PARAMS 显式传入，\n"
                "        例如：autopilot_params={\"w1\":55,\"zeta1\":0.55,\"w2\":50,...}\n"
                "        不得省略 autopilot_params，否则脚本将使用模板默认值而非推导值。\n"
                f"  3. generate_matlab(task_description=<摘要>, mission_conditions=\"{matlab_cond_str}\",\n"
                "       autopilot_params=<INIT_PARAMS — 必填，见step2>)\n"
                "  4. syntax_check_matlab(script_path=<step3返回路径>) — 校正脚本\n"
                f"  5. run_simulation(script_path=<校正后路径>, mission_conditions=\"{matlab_cond_str}\", nmc={rl_cfg.nmc_per_eval})\n"
                "     若 status='error'：重新调用 syntax_check_matlab → run_simulation\n"
                "  6. judge_requirements(task_prompt=<用户任务>, metrics=<仿真指标>)\n"
                "  7. 输出摘要并 STOP（见规则4）\n"
                "\n"
                "════════════════════════════════════════\n"
                "【完成条件 — 立即输出摘要并停止的触发条件】\n"
                "════════════════════════════════════════\n"
                "满足以下任一条件时立即输出中文摘要并结束，不再调用任何工具：\n"
                "  A. judge_requirements 已返回结果（无论 satisfied 是 True 还是 False）\n"
                "  B. run_simulation 失败且已重试 1 次仍未成功\n"
                "  C. 任意工具连续失败超过 2 次\n"
                "摘要格式：仿真指标（命中率/SEP/PeakNy/PM/BW）+ 是否满足要求 + Layer3路径说明。"

            )

            full_msg = (

                f"System: {hermes_system}\n\n"

                + (f"[Reference Context]\n{ctx}\n\n" if ctx else "")

                + f"User: {current_prompt}"

            )

            # ── Determine subtask list and execution strategy ────────────────
            _hermes_strategy = (
                task_plan.strategy
                if task_plan and task_plan.subtasks and len(task_plan.subtasks) > 1
                else ExecutionStrategy.SINGLE
            )
            _hermes_subtasks = (
                task_plan.subtasks
                if _hermes_strategy != ExecutionStrategy.SINGLE
                else [current_prompt]
            )

            def _build_hermes_msg(subtask_desc: str, cond_str_override: str = "") -> str:
                """Build the full System+User message for one subtask.

                cond_str_override: if provided, replaces all occurrences of the
                global matlab_cond_str inside hermes_system so that per-subtask
                mission conditions are propagated into explicit tool-call examples.
                """
                sys_msg = hermes_system
                if cond_str_override and cond_str_override != matlab_cond_str:
                    sys_msg = sys_msg.replace(matlab_cond_str, cond_str_override)
                return (
                    f"System: {sys_msg}\n\n"
                    + (f"[Reference Context]\n{ctx}\n\n" if ctx else "")
                    + f"User: {subtask_desc}"
                )

            def _build_subtask_msg(subtask_desc: str, cond_str: str = "") -> str:
                """Build a ROLE-SCOPED System+User message for sequential subagents.

                Unlike _build_hermes_msg which injects the full hermes_system
                (causing subagents to execute the complete pipeline), this function
                generates a minimal system prompt describing ONLY what this specific
                subagent is allowed to do.  This is the root-cause fix for the
                sequential subtask boundary violation problem.
                """
                _d = subtask_desc.lower()
                _need_retrieve = any(k in _d for k in ("检索","查询","retrieve","经验","rag","知识库","历史参数","调优经验"))
                _need_generate = any(k in _d for k in ("生成","脚本","generate","matlab","语法","syntax"))
                _need_simulate = any(k in _d for k in ("仿真","运行","simulation","nmc"))
                _need_judge    = any(k in _d for k in ("判定","判断","judge","指标","requirement"))

                # Build role description and allowed tools list
                _role_parts = []
                _allowed_tools = []
                _forbidden = []

                if _need_retrieve:
                    _role_parts.append("从知识库和参数经验库检索相关信息")
                    _allowed_tools += ["rag_retrieve", "parameter_experience_best",
                                       "parameter_experience_search"]
                    _forbidden += ["generate_matlab", "syntax_check_matlab",
                                   "run_simulation", "judge_requirements"]
                if _need_generate:
                    _role_parts.append("生成 MATLAB 仿真脚本并完成语法校正")
                    _allowed_tools += ["generate_matlab", "syntax_check_matlab"]
                    if not _need_simulate:
                        _forbidden += ["run_simulation", "judge_requirements"]
                if _need_simulate:
                    _role_parts.append("运行 MATLAB 仿真并收集指标结果")
                    _allowed_tools += ["run_simulation"]
                    if not _need_judge:
                        _forbidden += ["judge_requirements"]
                if _need_judge:
                    _role_parts.append("判定仿真指标是否满足任务要求")
                    _allowed_tools += ["judge_requirements"]

                # Common utility tools always allowed
                _allowed_tools += ["agent_memory_remember", "agent_memory_recall",
                                    "agent_memory_list", "session_search"]

                _role_str = "、".join(_role_parts) if _role_parts else "执行分配的子任务"
                _allowed_str = "、".join(sorted(set(_allowed_tools)))
                _cond_str = cond_str or matlab_cond_str

                _sys = (
                    f"You are a specialized sub-agent responsible for: {_role_str}.\n"
                    f"Mission conditions: {_cond_str}\n\n"
                    f"ALLOWED TOOLS (use ONLY these): {_allowed_str}\n"
                )
                if _forbidden:
                    _sys += (
                        f"FORBIDDEN TOOLS (do NOT call these under any circumstances): "
                        f"{', '.join(sorted(set(_forbidden)))}\n"
                    )
                _sys += (
                    "\nCRITICAL RULES:\n"
                    "1. Complete ONLY the task assigned to you. Do NOT proceed to next steps.\n"
                    "2. Do NOT call any forbidden tools even if you think it would be helpful.\n"
                    "3. After completing your assigned task, output your findings and STOP.\n"
                    "4. Do NOT run the full retrieve→generate→simulate→judge pipeline yourself.\n"
                )
                return (
                    f"System: {_sys}\n\n"
                    + f"User: {subtask_desc}"
                )

            def _capture_intermediate_script() -> None:
                """After each Hermes call, pull last_script_path from generate_matlab tool."""
                nonlocal hermes_generated_script
                for _ht in (getattr(hermes, "_tools", []) or []):
                    if getattr(_ht, "name", "") == "generate_matlab":
                        _lsp = getattr(_ht, "last_script_path", "")
                        if _lsp and os.path.isfile(_lsp):
                            hermes_generated_script = _lsp

            print(
                f"  Running Hermes agent "
                f"[strategy={_hermes_strategy.value}, subtasks={len(_hermes_subtasks)}]..."
            )

            # Record hermes_rl_best.json mtime BEFORE Hermes runs so we can
            # detect a fresh write caused by an internal matlab_rl_optimize call.
            _hermes_snap_path = __import__("multi_agent.run_file_manager", fromlist=["hermes_rl_best_path"]).hermes_rl_best_path()
            _snap_mtime_before = (
                _hermes_snap_path.stat().st_mtime
                if _hermes_snap_path.exists() else 0.0
            )

            # ── Strategy dispatch ─────────────────────────────────────────────

            def _infer_subtask_tools(desc: str):
                """Return a filtered tool list for this subtask based on its description.

                Keyword rules (applied in order; first match wins):
                  retrieve-only  → rag_retrieve, parameter_experience_*
                  generate       → generate_matlab, syntax_check_matlab  (+ retrieve)
                  simulate       → run_simulation  (+ generate so it can fix script)
                  judge          → judge_requirements, run_simulation
                  (no match)     → all tools (no restriction)
                """
                _all = getattr(hermes, "_tools", []) or []
                if not _all:
                    return []   # empty = use all
                _by_name = {getattr(t, "name", ""): t for t in _all}

                _d = desc.lower()
                _retrieve_names = {
                    "rag_retrieve", "parameter_experience_best",
                    "parameter_experience_search", "agent_memory_recall",
                    "agent_memory_list",
                }
                _generate_names = {"generate_matlab", "syntax_check_matlab"}
                _simulate_names = {"run_simulation"}
                _judge_names    = {"judge_requirements"}

                # Determine which groups are needed
                _need_retrieve = any(kw in _d for kw in (
                    "检索", "查询", "retrieve", "经验", "rag", "知识库", "历史参数",
                    "调优经验", "parameter_experience",
                ))
                _need_generate = any(kw in _d for kw in (
                    "生成", "脚本", "generate", "matlab", "语法", "syntax",
                ))
                _need_simulate = any(kw in _d for kw in (
                    "仿真", "运行", "simulation", "run_simulation", "nmc",
                ))
                _need_judge = any(kw in _d for kw in (
                    "判定", "判断", "judge", "指标判", "requirement",
                ))

                # If nothing matched → no restriction
                if not any((_need_retrieve, _need_generate, _need_simulate, _need_judge)):
                    return []

                _wanted: set = set()
                if _need_retrieve: _wanted |= _retrieve_names
                if _need_generate: _wanted |= _generate_names | _retrieve_names
                if _need_simulate: _wanted |= _simulate_names | _generate_names
                if _need_judge:    _wanted |= _judge_names | _simulate_names

                # Always keep memory/utility tools
                _wanted |= {"agent_memory_remember", "session_search"}

                _filtered = [_by_name[n] for n in _wanted if n in _by_name]
                if not _filtered:
                    return []  # safety: if nothing matched, use all
                return _filtered

            def _extract_structured_blocks(text: str, max_text_chars: int = 2000) -> str:
                """Extract all ```json and ```matlab code blocks from text fully,
                plus a truncated excerpt of the surrounding narrative.
                Used to pass full structured output (design docs, pseudocode) between subtasks.

                Returns a condensed string:
                  - All JSON blocks (fully)
                  - All MATLAB/python code blocks (fully)
                  - Narrative: first 800 + last 600 chars
                """
                import re as _re
                code_blocks = _re.findall(
                    r"```(?:json|matlab|python|m)\s*\n([\s\S]*?)```",
                    text,
                    flags=_re.IGNORECASE,
                )
                # Also extract plain ``` blocks that look like code
                all_blocks = _re.findall(r"```[\w]*\s*\n([\s\S]*?)```", text, flags=_re.IGNORECASE)
                blocks_text = ""
                if all_blocks:
                    blocks_text = "\n\n[前序子任务结构化输出（代码/JSON块，完整保留）]\n"
                    for _b in all_blocks:
                        blocks_text += f"```\n{_b.strip()}\n```\n"

                # Narrative: strip code blocks, then take head+tail
                narrative = _re.sub(r"```[\s\S]*?```", "", text).strip()
                if len(narrative) > max_text_chars:
                    head = narrative[:800]
                    tail = narrative[-600:]
                    narrative = head + "\n...(省略中间部分)...\n" + tail

                return (narrative + blocks_text).strip()

            def _extract_params_from_text(text: str) -> dict:
                """Extract autopilot parameter values from free-text subagent output.

                Looks for patterns like:
                  w1=60, zeta1=0.65, N_pn=4.5
                  "w1": 60, "zeta1": 0.65
                  INIT_PARAMS = {w1: 55, ...}
                Returns a dict of recognised param names → float values.
                """
                _PARAM_NAMES = {
                    "w1","zeta1","tao1","w2","zeta2","tao2",
                    "w3","zeta3","tao3","N_pn","N_guidance",
                }
                result: dict = {}
                for _pn in _PARAM_NAMES:
                    # Match "w1": 60  or  w1=60  or  w1 = 60.5
                    _m = re.search(
                        rf"""["\']?{re.escape(_pn)}["\']?\s*[:=]\s*([-+]?\d+(?:\.\d+)?)""",
                        text,
                    )
                    if _m:
                        try:
                            result[_pn] = float(_m.group(1))
                        except ValueError:
                            pass
                return result

            if _hermes_strategy == ExecutionStrategy.SEQUENTIAL and len(_hermes_subtasks) > 1:
                hermes_response = None
                _prev_result: str = ""
                _seq_script_path: str = ""   # script generated by any subtask
                _seq_sim_done: bool = False   # whether run_simulation was called
                _seq_sim_summary: str = ""    # human-readable sim metric summary
                _seq_extracted_params: dict = {}  # params from retrieval step
                _llm_st_conds = (
                    task_plan.subtask_conditions
                    if task_plan and task_plan.subtask_conditions
                       and len(task_plan.subtask_conditions) == len(_hermes_subtasks)
                    else None
                )
                for _si, _st in enumerate(_hermes_subtasks, 1):
                    print(f"\n  [Sequential {_si}/{len(_hermes_subtasks)}] {_st[:120]}")
                    # Use LLM-derived per-subtask conditions (authoritative);
                    # fall back to regex re-parse of subtask description.
                    if _llm_st_conds:
                        _st_cond_str = _llm_st_conds[_si - 1]
                    else:
                        _st_cond = parse_mission_conditions(_st)
                        _st_cond_str = build_matlab_conditions_str(_st_cond)
                    if _st_cond_str != matlab_cond_str:
                        print(f"  [Sequential] Per-subtask conditions: {_st_cond_str}")
                        if hasattr(hermes, "update_mission_conditions"):
                            hermes.update_mission_conditions(_st_cond_str)
                    # Build context block: completed artifacts + prev result summary
                    _artifact_lines = []
                    if _seq_script_path:
                        _artifact_lines.append(f"- 已生成脚本: {_seq_script_path}")
                    if _seq_sim_done:
                        _artifact_lines.append(f"- 仿真已完成: {_seq_sim_summary}")
                    _artifact_ctx = ("\n".join(_artifact_lines) + "\n") if _artifact_lines else ""

                    # Structured parameter handoff block (retrieve → generate)
                    _param_handoff = ""
                    if _seq_extracted_params:
                        _param_handoff = (
                            f"[前序子任务提取的参数建议 — 直接用于 autopilot_params]\n"
                            f"{json.dumps(_seq_extracted_params, ensure_ascii=False)}\n"
                            f"请将上述 JSON 直接作为 generate_matlab 的 autopilot_params 参数传入，"
                            f"不要修改键名，不要重新推导参数。\n\n"
                        )

                    _st_with_ctx = _st
                    if _prev_result or _artifact_ctx or _param_handoff:
                        _st_with_ctx = (
                            f"[已完成工作记录]\n"
                            f"{_artifact_ctx}"
                            f"{_param_handoff}"
                            f"[前序子任务 {_si-1} 输出（含完整代码/JSON块）]\n"
                            f"{_extract_structured_blocks(_prev_result)}\n\n"
                            f"[当前子任务 {_si}/{len(_hermes_subtasks)}]\n{_st}\n\n"
                            f"⚠️ 严格限制：只执行本子任务指定的具体操作。\n"
                            f"上方[已完成工作记录]中列出的工具调用已完成，请勿重复执行。\n"
                            f"如已有脚本路径，直接使用，不要再次调用 generate_matlab。\n"
                            f"如仿真已完成，直接使用其结果，不要再次调用 run_simulation。"
                        )
                    _st_tools = _infer_subtask_tools(_st)
                    if _st_tools:
                        print(f"  [Sequential] Allowed tools: "
                              f"{[getattr(t,'name','?') for t in _st_tools]}")
                    # Use role-scoped system prompt (not full hermes_system)
                    # so the subagent only executes its assigned responsibility.
                    _st_resp = await hermes.run_with_tools(
                        _build_subtask_msg(_st_with_ctx, cond_str=_st_cond_str),
                        tools=_st_tools,
                        verbose_thinking=hermes_verbose_thinking(),
                    )
                    hermes_response = _st_resp
                    _prev_result = str(_st_resp) if _st_resp else ""
                    # Extract structured parameters from this subtask's output
                    # so they can be passed as autopilot_params to the next subtask
                    if _prev_result:
                        _extracted = _extract_params_from_text(_prev_result)
                        if _extracted:
                            _seq_extracted_params.update(_extracted)
                            print(f"  [Sequential] Extracted params for handoff: "
                                  f"{_seq_extracted_params}")
                    # Track completed artifacts from this subtask
                    for _ht2 in (getattr(hermes, "_tools", []) or []):
                        _ht2_name = getattr(_ht2, "name", "")
                        if _ht2_name == "generate_matlab":
                            _lsp2 = getattr(_ht2, "last_script_path", "")
                            if _lsp2:
                                _seq_script_path = _lsp2
                        elif _ht2_name == "run_simulation":
                            _last2 = getattr(_ht2, "last_stdout", "")
                            if _last2:
                                _seq_sim_done = True
                                _m2 = parse_sim_stdout(_last2)
                                _seq_sim_summary = (
                                    f"hit={_m2.get('hit_rate',0):.1f}% "
                                    f"SEP={_m2.get('SEP',99):.2f}m "
                                    f"PM={_m2.get('pitch_PM',0):.1f}° "
                                    f"BW={_m2.get('pitch_BW',0):.1f}r/s"
                                )
                    _capture_intermediate_script()

                    # ── Early-exit: stop if this subtask signals Layer 3 / task done ──
                    # Hermes writes these markers in its final summary when it determines
                    # that RL optimisation or termination is needed.  Continuing to the
                    # next subtask would overlap with Layer 3 and waste API calls.
                    # IMPORTANT: the trigger phrase appears at the END of a long response,
                    # so we check both the first 800 chars AND the last 600 chars.
                    _full_resp_str = str(_st_resp) if _st_resp else ""
                    _resp_lower = (
                        _full_resp_str[:800] + _full_resp_str[-600:]
                    ).lower()
                    _rl_trigger = any(kw in _resp_lower for kw in (
                        "next_step = rl_optimize",
                        "next_step=rl_optimize",
                        "→ rl_optimize",
                        "rl_optimize",
                        "rl 参数优化",
                        "layer 3",
                        "layer3",
                        "进入rl",
                        "进入 rl",
                        "satisfied=true",
                        "satisfied = true",
                        "任务结束",
                        "本轮任务结束",
                        "不再调用工具",
                        "摘要完毕",
                    ))
                    if _rl_trigger:
                        print(
                            f"  [Sequential] Subtask {_si}/{len(_hermes_subtasks)} "
                            "triggered early exit (RL/task-done signal detected). "
                            "Skipping remaining subtasks."
                        )
                        break

                # Restore global conditions after sequential subtasks finish
                if hasattr(hermes, "update_mission_conditions"):
                    hermes.update_mission_conditions(matlab_cond_str)

            elif _hermes_strategy == ExecutionStrategy.PARALLEL and len(_hermes_subtasks) > 1:
                print(f"  [Parallel] Launching {len(_hermes_subtasks)} subtasks concurrently...")
                _parallel_coros = [
                    hermes.run_with_tools(_build_hermes_msg(_st), verbose_thinking=False)
                    for _st in _hermes_subtasks
                ]
                _parallel_results = await asyncio.gather(*_parallel_coros, return_exceptions=True)
                # Surface any errors
                for _pi, _pr in enumerate(_parallel_results):
                    if isinstance(_pr, Exception):
                        print(f"  [Parallel subtask {_pi+1}] Error: {_pr}")
                # Use the last valid (non-exception) response
                hermes_response = next(
                    (r for r in reversed(_parallel_results) if isinstance(r, str) and r),
                    None,
                )
                _capture_intermediate_script()

            else:
                hermes_response = await hermes.run_with_tools(full_msg, verbose_thinking=hermes_verbose_thinking())

            # ── Capture Hermes simulation metrics for Step 3.5 ────────────────
            # Find the RunSimulationTool instance inside Hermes's registered
            # tools and read its last_stdout so Step 3.5 can reflect on real
            # simulation metrics rather than placeholder values when
            # rl_optimization is disabled.
            hermes_sim_metrics: Dict[str, Any] = {}
            hermes_directly_satisfied: bool = False  # from judge_requirements tool

            _hermes_tools = getattr(hermes, "_tools", []) or []

            hermes_generated_script = ""  # path from generate_matlab tool

            for _ht in _hermes_tools:

                _ht_name = getattr(_ht, "name", "")

                if _ht_name == "run_simulation":

                    # Prefer pre-parsed last_metrics (avoids re-parsing 252-char
                    # matlab.engine stdout that may lack full metric text).
                    _cached = getattr(_ht, "last_metrics", {})
                    if _cached and _cached.get("hit_rate", 0) > 0:
                        hermes_sim_metrics = _cached
                    else:
                        _last = getattr(_ht, "last_stdout", "")
                        if _last:
                            hermes_sim_metrics = parse_sim_stdout(_last)

                    cprint(f"  [Hermes sim metrics] {metrics_summary_line(hermes_sim_metrics)}")

                    await parameter_experience.store_event(
                        "hermes_sim",
                        f"iter={iteration} {metrics_summary_line(hermes_sim_metrics)}",
                        metadata={"iteration": iteration},
                    )

                elif _ht_name == "judge_requirements":
                    # Check if judge_requirements tool returned satisfied=True
                    _last_judge = getattr(_ht, "_last_result", None)
                    if isinstance(_last_judge, dict) and _last_judge.get("satisfied"):
                        hermes_directly_satisfied = True
                        cprint(f"  [Layer2] judge_requirements._last_result: satisfied=True → skipping RL")
                    elif isinstance(_last_judge, str):
                        try:
                            _lj_parsed = json.loads(_last_judge)
                            if _lj_parsed.get("satisfied"):
                                hermes_directly_satisfied = True
                                cprint(f"  [Layer2] judge_requirements (parsed): satisfied=True → skipping RL")
                        except Exception:
                            pass
                    # Fallback: parse from Hermes response text (weak signal only)
                    if (
                        not hermes_directly_satisfied
                        and hermes_response
                        and '"satisfied": true' in str(hermes_response).lower()
                    ):
                        hermes_directly_satisfied = True
                        dprint(
                            "  [Layer2] judge_requirements (response text): satisfied=True "
                            "(pending sim metrics rule-check)"
                        )

                elif _ht_name == "generate_matlab":

                    _lsp = getattr(_ht, "last_script_path", "")

                    if _lsp and os.path.isfile(_lsp):

                        hermes_generated_script = _lsp

                        cprint(f"  [Hermes generated script] {os.path.basename(_lsp)}")

                    _lmd = getattr(_ht, "last_modification_desc", "")

                    if _lmd:

                        hermes_modify_law_desc = _lmd

            # ── Detect Hermes-internal RL task_requirements_met ─────────────
            # If matlab_rl_optimize wrote a *new* snapshot with
            # status=task_requirements_met during this Hermes run, adopt it as
            # rl_result and break — Step 3 is redundant in this case.
            if _hermes_snap_path.exists():
                _snap_mtime_after = _hermes_snap_path.stat().st_mtime
                if _snap_mtime_after > _snap_mtime_before:
                    try:
                        with open(_hermes_snap_path, encoding="utf-8") as _sf:
                            _hermes_snap = json.load(_sf)
                        if _hermes_snap.get("status") == "task_requirements_met":
                            rl_result = _hermes_snap
                            _m = _hermes_snap.get("best_metrics", {})
                            print(
                                f"\n  [Hermes RL] ✓ task_requirements_met detected in "
                                f"hermes_rl_best.json — skipping Step 3.\n"
                                f"  hit={_m.get('hit_rate',0):.1f}%  "
                                f"miss={_m.get('miss_distance',99):.4f}m  "
                                f"PM={_m.get('pitch_PM',0):.1f}deg"
                            )
                            iteration_records.append(_iter_record("task_requirements_met"))
                            break
                    except Exception as _hsnap_exc:
                        print(f"  [Hermes RL] Snapshot read error ({_hsnap_exc}); continuing to Step 3.")

            # ─────────────────────────────────────────────────────────────────

            if hermes_response:

                print(f"  Hermes response length: {len(str(hermes_response))} chars")

                # Extract the largest MATLAB code block and save it

                matlab_blocks = re.findall(

                    r"```(?:matlab|octave|m)?\s*\n(.*?)```",

                    str(hermes_response), re.DOTALL

                )

                if matlab_blocks:

                    code = max(matlab_blocks, key=len).strip()

                    save_name = f"hermes_generated_{iteration}.m"

                    # Validate before writing to matlab_scripts/ — guidance-law
                    # modification explanations or prose often appear as matlab
                    # code blocks and must not pollute the script directory that
                    # _find_latest_matlab_script() scans for RL input.
                    from multi_agent.tools.simulation_tool import GenerateMATLABTool as _GMT

                    _is_sim_template = _GMT._is_real_matlab_template(code)

                    if _is_sim_template:

                        save_path = MATLAB_SCRIPTS_DIR / save_name

                        try:
                            from multi_agent.tools.simulation_tool import _sanitize_matlab_guidance_script as _san
                            code, _san_notes = _san(code)
                            if _san_notes:
                                print(f"  [Sanitise] Stripped {len(_san_notes)} annotation line(s); fixed dp/global issues.")
                        except Exception:
                            pass

                        with open(save_path, "w", encoding="utf-8") as _f:

                            _f.write(code)

                        print(f"  MATLAB model saved → {save_path}")

                        import shutil as _sh

                        _sh.copy2(str(save_path), str(MODEL_OUTPUT_DIR / save_name))

                        print(f"  Copy → {MODEL_OUTPUT_DIR / save_name}")

                    else:

                        # Not a valid simulation template (likely guidance-law
                        # modification text). Save to model_output/ only for
                        # traceability — do NOT write to matlab_scripts/.
                        save_path = MODEL_OUTPUT_DIR / save_name

                        with open(save_path, "w", encoding="utf-8") as _f:

                            _f.write(code)

                        print(f"  ⚠ Extracted code block is not a runnable simulation template "

                              f"(guidance-law explanation?); saved to model_output only → {save_path}")

                else:

                    print("  No MATLAB code block found in Hermes response.")

        # ── PE cached-params early exit (Change 1) ──────────────────────────────

        # Before running the expensive RL loop, ask the reflection agent whether

        # the best LONG_TERM experience already satisfies the user's requirements.

        # If yes: set rl_result from PE and break — Step 5 produces the report

        # directly from the cached satisfying parameters.

        if (ablation_cfg.reflection_agent

                and reflection_agent is not None

                and ablation_cfg.parameter_experience_reuse

                and parameter_experience is not None

                and not hermes_directly_satisfied):

            try:

                _pe_top = await parameter_experience.retrieve_best(

                    task_context={"mode": task_mode, "category": ",".join(sorted(mission_conditions.keys())), "subcases": str({k: sorted(v) for k, v in sorted(mission_conditions.items())}),

                                  "prompt": current_prompt},

                    top_k=5,

                )

                # Skip entries that have no usable auto_params (e.g. stored with
                # empty best_params from a previous buggy run).
                from multi_agent.rl.matlab_rl_optimizer import ALL_TUNABLE_PARAM_SPECS as _SPECS
                _pe_rec = _pe_objs = _pe_params = None
                for _candidate in (_pe_top or []):
                    _c_params = _candidate.get("parameters", {})
                    _c_auto = (
                        {k[len("dp_"):]: float(v) for k, v in _c_params.items() if k.startswith("dp_")}
                        or {k: float(v) for k, v in _c_params.items() if k in _SPECS}
                    )
                    if _c_auto:
                        _pe_rec    = _candidate
                        _pe_objs   = _candidate.get("objectives", {})
                        _pe_params = _c_params
                        break

                if _pe_rec is not None:

                    # ── Skip PE check if Hermes already has better metrics ──
                    # Hermes just simulated the CURRENT script under the CURRENT
                    # conditions.  If those metrics are already >= PE cached
                    # performance, re-simulating with old PE params is pointless
                    # (and confusing — two different results for two different
                    # parameter sets).
                    _hermes_hr = hermes_sim_metrics.get("hit_rate", 0.0)
                    _pe_hr     = (_pe_objs or {}).get("hit_rate", 0.0)
                    if hermes_sim_metrics and _hermes_hr >= _pe_hr and _hermes_hr > 0:
                        print(
                            f"  [Pre-RL] Hermes hit={_hermes_hr:.1f}% >= PE cached "
                            f"hit={_pe_hr:.1f}%. Skipping PE re-simulation "
                            f"(Hermes metrics are fresher)."
                        )
                        _pe_rec = None   # skip the PE branch entirely

                if _pe_rec is not None:

                    if _pe_objs:

                        # ── Re-simulate with PE params under CURRENT conditions ──
                        # Cached _pe_objs may come from a different sub-case (e.g.
                        # T3 stored, but current task is T4).  Always re-run a quick
                        # simulation so the reflection agent judges the *current*
                        # workload, not a stale result from a different run.
                        _pe_auto_params = {
                            k[len("dp_"):]: float(v)
                            for k, v in _pe_params.items()
                            if k.startswith("dp_")
                        }
                        if not _pe_auto_params:
                            # Older PE entries stored without dp_ prefix; handle both
                            _pe_auto_params = {
                                k: float(v)
                                for k, v in _pe_params.items()
                                if k in ALL_TUNABLE_PARAM_SPECS
                            }

                        _pe_script = (
                            hermes_generated_script
                            or _find_latest_matlab_script(
                                str(MATLAB_SCRIPTS_DIR), str(MODEL_OUTPUT_DIR), "."
                            )
                            or getattr(RunSimulationTool, "KB_TEMPLATE_PATH",
                                       "./knowledge_base/matlab/guidance/monte_carlo_single.m")
                        )

                        # Debug: show what we have for the condition check
                        _has_optimizer = layer3_workflow.rl_optimizer is not None
                        print(
                            f"  [Pre-RL] PE check — optimizer={'ok' if _has_optimizer else 'None'} "
                            f"auto_params={len(_pe_auto_params)} keys  "
                            f"script={'ok' if _pe_script else 'empty'}"
                        )

                        _fresh_sim_ok = False  # flag: did we get a fresh sim result?

                        if _has_optimizer and _pe_auto_params and _pe_script:
                            _quick_nmc = int(getattr(
                                layer3_workflow.rl_optimizer,
                                "nmc_per_eval",
                                rl_cfg.nmc_per_eval,
                            ))
                            print(
                                f"  [Pre-RL] Re-simulating PE params under current "
                                f"conditions (nmc={_quick_nmc})…"
                            )
                            try:
                                _fresh_metrics = await (
                                    layer3_workflow.rl_optimizer
                                    ._run_simulation_with_params(
                                        auto_params=_pe_auto_params,
                                        mission_conditions=mission_conditions,
                                        script_path=_pe_script,
                                        nmc=_quick_nmc,
                                    )
                                )
                                # Normalise key names so print + reflect see consistent keys
                                if "SEP" in _fresh_metrics and "miss_distance" not in _fresh_metrics:
                                    _fresh_metrics["miss_distance"] = _fresh_metrics["SEP"]
                                _fresh_sim_ok = True
                                print(
                                    f"  [Pre-RL] Fresh sim result: "
                                    f"hit={_fresh_metrics.get('hit_rate', 0):.1f}% "
                                    f"SEP={_fresh_metrics.get('miss_distance', 99):.3f}m "
                                    f"PM={_fresh_metrics.get('pitch_PM', 0):.1f}° "
                                    f"BW={_fresh_metrics.get('pitch_BW', 0):.1f}"
                                )
                            except (asyncio.CancelledError, KeyboardInterrupt):
                                raise
                            except Exception as _sim_exc:
                                print(
                                    f"  [Pre-RL] Fresh sim failed ({_sim_exc}); "
                                    f"cannot verify PE params — proceeding to RL."
                                )
                        else:
                            print(
                                f"  [Pre-RL] Cannot re-simulate PE params "
                                f"(missing optimizer/params/script) — proceeding to RL."
                            )

                        # Early-exit is ONLY allowed when a fresh simulation confirmed
                        # that the PE params satisfy the CURRENT task conditions.
                        if _fresh_sim_ok:
                            _pe_ref = await reflection_agent.reflect(

                                current_prompt,

                                {"parameters": _pe_params, "metrics": _fresh_metrics},

                            )

                            if not _pe_ref.get("needs_optimization", True):

                                print("  [Pre-RL] \u2713 PE params verified on current conditions "

                                      "— requirements met. Skipping Steps 3-4.")

                                _best_params_from_pe = {
                                    k.replace("dp_", ""): v
                                    for k, v in _pe_params.items()
                                    if k.startswith("dp_")
                                } or _pe_auto_params

                                rl_result = {

                                    "status":          "task_requirements_met",

                                    "best_params":     _best_params_from_pe,

                                    "best_metrics":    _fresh_metrics,

                                    "best_reward":     float(_pe_rec.get("fitness", 0.0)),

                                    "baseline_reward": 0.0,

                                    "total_episodes":  0,

                                    "source":          "pe_cache_verified",

                                }

                                break

                            else:

                                _hint = _pe_ref.get("suggestion", "")[:120]

                                print(f"  [Pre-RL] Fresh sim shows PE params do not satisfy "

                                      f"requirements. Proceeding to RL. "

                                      f"Hint: {_hint}")

            except Exception as _pe_exc:

                print(f"  [Pre-RL] PE early-exit check failed ({_pe_exc}); "

                      f"proceeding to RL.")

        # ────────────────────────────────────────────────────────────────────────



        # ── Ablation: optimization_workflow gate ────────────────────────────────

        # When disabled: Hermes content generation (Step 2.5) is the final

        # output; skip Steps 3, 3.5, 4 and jump directly to the report.

        if not ablation_cfg.optimization_workflow:

            print("  [Ablation] optimization_workflow: DISABLED — skipping Steps 3~4; continuing to next iteration.")

            iteration_records.append(_iter_record("ablation_optim_workflow_disabled", force_no_rl=True))

            continue

        # ── Authoritative rule-check on parsed sim metrics (always) ─────────
        # Never trust judge_requirements alone — Hermes may hand-fill metrics or
        # pass a truncated task_prompt (log06121158 false-positive regression).
        if hermes_sim_metrics and hermes_sim_metrics.get("hit_rate", 0) > 0:
            try:
                from multi_agent.tools.judge_requirements_tool import (
                    _resolve_requirements,
                    _rule_check,
                )
                _auth_reqs = _resolve_requirements(current_prompt, args.prompt or "")
                if _auth_reqs:
                    _auth_ok, _auth_reasons = _rule_check(hermes_sim_metrics, _auth_reqs)
                    if _auth_ok:
                        if not hermes_directly_satisfied:
                            hermes_directly_satisfied = True
                            cprint(
                                "  [Layer2→3] Rule-check on Hermes sim metrics → ALL PASS → skipping RL"
                            )
                        for _r in _auth_reasons:
                            dprint(f"    {_r}")
                    else:
                        if hermes_directly_satisfied:
                            mprint(
                                "  [Layer2→3] judge satisfied=True 但仿真指标复核未通过 → 强制进入 Layer3 RL"
                            )
                            for _r in _auth_reasons:
                                if "[NG]" in _r or "缺失" in _r:
                                    mprint(f"    {_r}")
                        hermes_directly_satisfied = False
                        _ng = [r for r in _auth_reasons if "[NG]" in r or "缺失" in r]
                        cprint(
                            f"  [Layer2→3] Rule-check: {len(_ng)} metric(s) not met → entering RL"
                        )
            except Exception as _fb_exc:
                logger.debug(f"Authoritative rule-check failed: {_fb_exc}")

        # ── Skip Layer 3 if Hermes already satisfied requirements ───────────────
        # When sequential/parallel Hermes already ran judge_requirements and
        # confirmed satisfied=True, running Layer 3 RL would be redundant work.
        if hermes_directly_satisfied:
            cprint(
                "\n  [Layer 3] Layer2 metrics satisfied requirements — "
                "skipping RL optimization → direct to Reflection Agent."
            )

        # ── Layer 2 script gate (KB regenerate + syntax + smoke) ─────────────
        # 优先使用：当前轮生成的脚本 → 上一轮的脚本 → 最新脚本 → KB模板
        _gate_script = (
            hermes_generated_script  # 当前轮 Hermes 生成的脚本
            or matlab_script_path  # 上一轮 Layer 3 使用的脚本
            or _find_latest_matlab_script(
                str(MATLAB_SCRIPTS_DIR), str(MODEL_OUTPUT_DIR), "."
            )
            or getattr(RunSimulationTool, "KB_TEMPLATE_PATH",
                       "./knowledge_base/matlab/guidance/monte_carlo_single.m")
        )
        _gate_metrics = dict(hermes_sim_metrics or {})
        _gate_enabled = bool(getattr(sim_cfg, "layer2_script_gate", True))
        _gate_smoke_nmc = int(getattr(sim_cfg, "layer2_smoke_nmc", 5))

        if (
            ablation_cfg.optimization_workflow
            and not hermes_directly_satisfied
            and _gate_enabled
        ):
            from multi_agent.integration.layer2_script_gate import (
                ensure_script_ready_for_layer3,
            )
            mprint(
                f"\n[Layer 2 Gate] Validating script before Layer 3 "
                f"(smoke nmc={_gate_smoke_nmc})…"
            )
            _g_ok, _g_script, _g_metrics, _g_msg = await ensure_script_ready_for_layer3(
                script_path=_gate_script,
                mission_conditions_str=matlab_cond_str,
                task_prompt=current_prompt,
                existing_metrics=_gate_metrics,
                simulator=simulator,
                smoke_nmc=_gate_smoke_nmc,
                enabled=True,
                task_mode=task_mode,
            )
            if _g_ok:
                _gate_script = _g_script
                gate_passed_script = _g_script
                # 只在 Hermes 执行时更新 hermes_generated_script，保留脚本路径
                if wf_cfg.hermes_execution:
                    hermes_generated_script = _g_script
                if _g_metrics:
                    _gate_metrics = _g_metrics
                    hermes_sim_metrics = _g_metrics
                mprint(f"  [Layer 2 Gate] ✓ {_g_msg}")
            else:
                mprint(f"  [Layer 2 Gate] ✗ BLOCKED Layer 3 — {_g_msg}")
                _gate_near_miss = False
                try:
                    from multi_agent.integration.t4_low_risk import (
                        should_gate_near_miss_tune,
                    )

                    _gate_near_miss, _nm_msg = should_gate_near_miss_tune(
                        _g_metrics or hermes_sim_metrics,
                        gate_message=_g_msg,
                        task_prompt=current_prompt,
                        task_mode=task_mode,
                    )
                    if _gate_near_miss:
                        cprint(f"  [Layer 2 Gate] Near-miss → {_nm_msg}")
                        task_mode = "TUNE_PARAMS"
                        if _g_metrics:
                            hermes_sim_metrics = _g_metrics
                        prev_suggestion = _nm_msg
                except Exception as _nm_exc:
                    logger.debug(f"Gate near-miss tune check failed: {_nm_exc}")

                if not _gate_near_miss:
                    iteration_records.append(_iter_record("layer2_gate_blocked"))
                    prev_suggestion = _g_msg
                    continue

        # ── LAYER 3: Optimization Workflow ────────────────────────────────────
        # Determine script path to feed into Layer 3

        _wf_script = _gate_script

        cprint(f"\n[Layer 3] Optimization Workflow | script={os.path.basename(_wf_script or 'None')}")
        dprint(f"  directly_satisfied={hermes_directly_satisfied}")

        # Apply reflection-suggested RL hyperparameter overrides for this iteration
        with TaskWorkspace() as _task_ws:
            if _wf_script and os.path.isfile(_wf_script):
                _wf_script = _task_ws.prepare_script(_wf_script)
                dprint(f"  [TaskWorkspace] isolated script → {_task_ws.path}")

            if _rl_hp_override and layer3_workflow.rl_optimizer is not None:
                apply_rl_hyperparams_to_optimizer(
                    layer3_workflow.rl_optimizer, _rl_hp_override
                )

            _wf_max_ep = getattr(layer3_workflow.rl_optimizer, "max_episodes", rl_cfg.max_episodes) if layer3_workflow.rl_optimizer else rl_cfg.max_episodes
            _wf_nmc    = getattr(layer3_workflow.rl_optimizer, "nmc_per_eval",  rl_cfg.nmc_per_eval)  if layer3_workflow.rl_optimizer else rl_cfg.nmc_per_eval

            _wf_init_params = _prev_rl_best_params or None
            _wf_warm_source = (
                "pe"
                if _prev_rl_best_params and ablation_cfg.parameter_experience_reuse
                else "default"
            )
            if hermes_generated_script and hermes_sim_metrics:
                try:
                    import re as _re_l3
                    from multi_agent.rl.matlab_rl_optimizer import ALL_TUNABLE_PARAM_SPECS as _L3_SPECS, RL_PARAM_TO_MATLAB
                    _script_txt = open(hermes_generated_script, "r", encoding="utf-8").read()
                    _hermes_params: dict = {}
                    _rev_map = {v: k for k, v in RL_PARAM_TO_MATLAB.items()}
                    for _rl_var, _py_key in _rev_map.items():
                        if _py_key in _L3_SPECS:
                            _m = _re_l3.search(rf"{_rl_var}\s*=\s*([\d.eE+\-]+)\s*;", _script_txt)
                            if _m:
                                _hermes_params[_py_key] = float(_m.group(1))
                    if _hermes_params:
                        _wf_init_params = _hermes_params
                        _wf_warm_source = "hermes"
                        dprint(f"  [Layer 3] Using Hermes script params as warm-start: {_hermes_params}")
                except Exception as _ext_exc:
                    dprint(f"  [Layer 3] Could not extract Hermes params ({_ext_exc}); using prev_rl_best")

            _mc_str_l3 = ""
            if mission_conditions:
                try:
                    _mc_str_l3 = build_matlab_conditions_str(mission_conditions)
                except Exception:
                    pass
            _wf_init_params = resolve_initial_auto_params(
                _wf_init_params,
                task_prompt=current_prompt,
                mission_conditions=_mc_str_l3,
                optimization_history=_optim_history or None,
            )
            if args.resume_checkpoint:
                _wf_warm_source = "checkpoint"

            try:
                print(f"  [DEBUG Layer3] mission_conditions = {mission_conditions}")
                workflow_result = await layer3_workflow.run(
                    script_path=_wf_script or "",
                    task_prompt=current_prompt,
                    mission_conditions=mission_conditions,
                    initial_metrics=hermes_sim_metrics,
                    directly_satisfied=hermes_directly_satisfied,
                    max_episodes=_wf_max_ep,
                    nmc=_wf_nmc,
                    initial_auto_params=_wf_init_params,
                    warm_start_source=_wf_warm_source,
                    optimization_history=_optim_history or None,
                    resume_checkpoint=args.resume_checkpoint,
                )
            except Exception as _wf_exc:
                print(f"  [Layer 3] Workflow error: {_wf_exc}")
                import traceback; traceback.print_exc()
                workflow_result = {
                    "status": "needs_iteration",
                    "best_params": {},
                    "best_metrics": hermes_sim_metrics,
                    "suggestion": f"工作流错误: {_wf_exc}",
                    "next_action": "tune_params",
                }

        _wf_status  = workflow_result.get("status", "needs_iteration")
        _wf_metrics = workflow_result.get("best_metrics", {})
        _wf_params  = workflow_result.get("best_params", {})
        suggestion  = workflow_result.get("suggestion", "")
        next_action = workflow_result.get("next_action", "tune_params")

        # 保存本轮使用的脚本路径，供下一轮使用
        if _wf_script and os.path.isfile(_wf_script):
            matlab_script_path = _wf_script
            logger.info(f"[Iteration {iteration}] Saved script path for next iteration: {os.path.basename(_wf_script)}")

        # Carry best params to next iteration so RL starts from a warm point.
        # Guard: if the autopilot became unstable (PM < 0), discard those params
        # to avoid poisoning the next iteration's warm-start with an unstable seed.
        _wf_pm = float(_wf_metrics.get("pitch_PM", 0.0) or 0.0)
        if _wf_params and _wf_pm >= 0:
            _prev_rl_best_params = dict(_wf_params)
            print(f"  [Layer 3] Best params carried to next iteration: "
                  f"{list(_wf_params.keys())} ({len(_wf_params)} values)")
        elif _wf_params and _wf_pm < 0:
            print(
                f"  [Layer 3] ⚠ PM={_wf_pm:.1f}° < 0 (unstable) — "
                "NOT carrying these params; next iteration will use previous stable seed."
            )

        # Record this iteration in optimization history for multi-round reflection analysis
        _optim_history.append({
            "iteration":   iteration,
            "task_mode":   task_mode,  # TUNE_PARAMS / MODIFY_LAW / REUSE_HISTORY
            "script":      os.path.basename(hermes_generated_script) if hermes_generated_script else "",
            "hit_rate":    _wf_metrics.get("hit_rate", 0),
            "SEP":         _wf_metrics.get("SEP", _wf_metrics.get("miss_distance", 99)),
            "miss_distance": _wf_metrics.get("miss_distance", 99),
            "pitch_PM":    _wf_metrics.get("pitch_PM", 0),
            "pitch_BW":    _wf_metrics.get("pitch_BW", 0),
            "peak_n":      _wf_metrics.get("peak_ny", _wf_metrics.get("peak_n", 0)),
            "peak_ny_max": _wf_metrics.get(
                "peak_ny_max",
                _wf_metrics.get("peak_ny_max", _wf_metrics.get("peak_ny", _wf_metrics.get("peak_n", 0))),
            ),
            "metrics":     dict(_wf_metrics),
            "best_reward": workflow_result.get("best_reward", 0),
            "suggestion":  suggestion,
            "next_action": next_action,
            "design_path_analysis": workflow_result.get("design_path_analysis",
                                        workflow_result.get("suggestion", "")).split("【设计路径分析】")[-1][:300] if suggestion else "",
            "best_params": {k: round(v, 4) for k, v in _wf_params.items()} if _wf_params else {},
        })

        # Extract RL hyperparameter suggestions from reflection for next iteration
        _rl_hp_override, _hp_fw_warn = sanitize_rl_hyperparams(
            workflow_result.get("rl_hyperparams", {}) or {}
        )
        for _w in _hp_fw_warn:
            print(f"  [ParameterFirewall] ⚠ {_w}")
        if _rl_hp_override:
            print(f"  [Reflection] RL hyperparameter suggestions for next iteration: {_rl_hp_override}")

        # Populate rl_result for backward-compatible report generation
        rl_result = {
            "status":          _wf_status,
            "best_params":     _wf_params,
            "best_metrics":    _wf_metrics,
            "best_reward":     workflow_result.get("best_reward", 0.0),
            "baseline_reward": workflow_result.get("baseline_reward", 0.0),
            "total_episodes":  workflow_result.get("total_episodes", 0),
        }

        mprint(f"\n  [Layer 3 Result] status={_wf_status}")
        mprint(f"  {metrics_summary_line(_wf_metrics)}")

        needs_optimization = (_wf_status != "done")
        prev_suggestion = suggestion

        await parameter_experience.store_event(
            "reflection",
            f"iter={iteration} needs_opt={needs_optimization} suggestion={suggestion[:150]}",
            metadata={"iteration": iteration, "needs_optimization": needs_optimization},
        )

        if _wf_status == "done":
            mprint("  ✓ Task requirements met!")
            if workflow_result.get("report_path"):
                mprint(f"  JSON Report: {workflow_result['report_path']}")
            iteration_records.append(_iter_record("done"))
            break

        else:
            if iteration < max_iterations:
                _m = _wf_metrics
                _rl_summary = f"[Layer3 Metrics] {metrics_summary_line(_m)}"
                current_prompt = (
                    f"{args.prompt}\n\n"
                    f"[Iteration {iteration} 优化建议 (next_action={next_action})]\n"
                    f"{suggestion}\n{_rl_summary}"
                )
                cprint(f"  Feeding back to Layer 2: next_action={next_action}")
                dprint(f"  Suggestion: {suggestion[:200]}")
                iteration_records.append(_iter_record("continue_optimize"))
                continue
            else:
                print("  Reached max iterations.")
                iteration_records.append(_iter_record("max_iterations_reached"))
                break

    # 4. ParameterExperience Writeback + model file saving

    print("\n[Step 4] ParameterExperience Writeback & Model File Saving...")

    last_mem_id = None

    _persistable_statuses = ("success", "task_requirements_met", "done", "needs_iteration")

    # If Hermes Agent satisfied requirements internally via matlab_rl_optimize tool,
    # rl_result is still None here.  Load the JSON snapshot it writes so Step 4
    # can persist the result just like a regular Step-3 RL run.
    if rl_result is None:

        _hermes_snap4 = __import__("multi_agent.run_file_manager", fromlist=["hermes_rl_best_path"]).hermes_rl_best_path()

        if _hermes_snap4.exists():

            try:

                with open(_hermes_snap4, encoding="utf-8") as _sf4:

                    rl_result = json.load(_sf4)

                print(f"  [Step 4] Loaded Hermes RL snapshot for writeback: "

                      f"status={rl_result.get('status')}  "

                      f"miss={rl_result.get('best_metrics', {}).get('miss_distance', '?')}")

            except Exception as _se4:

                print(f"  [Step 4] Failed to load Hermes RL snapshot ({_se4}); writeback skipped.")

    if not ablation_cfg.optimization_workflow:

        print("  [Ablation] optimization_workflow: DISABLED — Step 4 skipped.")

    elif rl_result and rl_result.get("status") in _persistable_statuses:

        m = rl_result["best_metrics"]

        _status_labels = {

            "task_requirements_met":  "task requirements met (reflection)",

            "success":                "RL completed",

        }

        _why = _status_labels.get(rl_result.get("status"), "RL completed")

        print(f"  RL best params persisted ({_why}): "

              f"miss={m.get('miss_distance', 99):.4f} m, "

              f"hit={m.get('hit_rate', 0):.1f}%, "

              f"PM={m.get('pitch_PM', 0):.1f} deg.")

        # Also explicitly store a grid-level summary so we know the ID

        rl_params  = {
            (f"dp_{k}" if not k.startswith("dp_") else k): float(v)
            for k, v in rl_result["best_params"].items()
        }

        def _safe_float(v):
            try:
                return float(v)
            except (TypeError, ValueError):
                s = str(v).strip().lower()
                s = s.replace("db", "").replace("deg", "").replace("rad/s", "").strip()
                if s in ("inf", "+inf"):
                    return float("inf")
                if s in ("-inf",):
                    return float("-inf")
                try:
                    return float(s)
                except ValueError:
                    return 0.0
        rl_objs    = {k: _safe_float(v) for k, v in m.items()}

        rl_fitness = max(0.0, 1.0/(1.0+m.get("miss_distance",100))) * ((m.get("hit_rate",0)/100)+0.01)

        _mem_type = (

            MemoryType.LONG_TERM

            if rl_result.get("status") in ("task_requirements_met", "done")

            else MemoryType.SHORT_TERM

        )

        _mc_keys = sorted(mission_conditions.keys())
        _rc_store = _mc_keys[0] if len(_mc_keys) == 1 else "ALL"
        _si_list_store = mission_conditions.get(_rc_store, [0]) if _rc_store != "ALL" else [0]
        _si_store = _si_list_store[0] if len(_si_list_store) == 1 else 0

        if not rl_params:
            print("  [PE Store] Skipping PE write — best_params is empty (nothing to persist).")
            last_mem_id = None
        else:
            last_mem_id = await parameter_experience.store(

                task_context={
                    "task":     "guidance_rl_optimization",
                    "mode":     task_mode,
                    "RUN_CASE": _rc_store,
                    "SUB_IDX":  _si_store,
                    "category": ",".join(_mc_keys),
                    "subcases":  str({k: sorted(v) for k, v in sorted(mission_conditions.items())}),
                    "prompt":   args.prompt,
                },

                parameters=rl_params,

                objectives=rl_objs,

                fitness=rl_fitness,

                metadata={"iteration": iteration, "total_episodes": rl_result.get("total_episodes", 0)},

                memory_type=_mem_type,

            )

        _layer_label = "LONG_TERM" if _mem_type == MemoryType.LONG_TERM else "SHORT_TERM"

        print(f"  Experience written (memory_id={last_mem_id}, layer={_layer_label}).")

        # Auto-promote any SHORT_TERM entries accumulated this session

        # that meet the quality bar into LONG_TERM for cross-session reuse.

        _promoted = await parameter_experience.auto_promote(min_fitness=0.4)

        if _promoted:

            print(f"  Auto-promoted {_promoted} short-term entries to LONG_TERM.")



        # Save successful MATLAB model file to PE base

        matlab_script = _find_latest_matlab_script(str(MATLAB_SCRIPTS_DIR), str(MODEL_OUTPUT_DIR), ".")

        if matlab_script and last_mem_id:

            saved_path = parameter_experience.save_successful_model(matlab_script, last_mem_id)

            if saved_path:

                print(f"  Successful MATLAB model saved to: {saved_path}")

            import shutil as _shutil

            try:

                _dst = MODEL_OUTPUT_DIR / Path(matlab_script).name

                if Path(matlab_script).resolve() != _dst.resolve():

                    _shutil.copy2(matlab_script, str(_dst))

                    print(f"  MATLAB model copied to: {_dst}")

                else:

                    print(f"  MATLAB model already in model_output: {_dst}")

            except Exception as _e:

                print(f"  Could not copy MATLAB model: {_e}")



        # Save successful SysML model files to PE base

        sysml_out_dir = Path(mo_cfg.sysml_output_dir)

        if sysml_out_dir.exists() and last_mem_id:

            sysml_files = sorted(sysml_out_dir.glob("*.xml"), key=lambda p: p.stat().st_mtime, reverse=True)

            for sysml_file in sysml_files[:3]:  # save latest 3 (BDD+Parametric+IBD)

                saved_xml = parameter_experience.save_successful_model(str(sysml_file), last_mem_id)

                if saved_xml:

                    print(f"  SysML model saved to: {saved_xml}")

    else:

        print("  RL optimisation did not produce a valid result; nothing written.")



    # 5. Generate Report

    print("\n[Step 5] Generating Report...")

    # If Step 3 was skipped (optimization_workflow / rl_optimization disabled),

    # rl_result is None.  Try to load the JSON snapshot that matlab_rl_optimize

    # writes when the per-episode reflection agent exits with task_requirements_met.

    # This makes the Hermes-orchestrated RL result available to the report.

    if rl_result is None:

        _hermes_snap = __import__("multi_agent.run_file_manager", fromlist=["hermes_rl_best_path"]).hermes_rl_best_path()

        if _hermes_snap.exists():

            try:

                import json as _json_s

                with open(_hermes_snap, encoding="utf-8") as _sf:

                    rl_result = _json_s.load(_sf)

                print(f"  Loaded Hermes RL snapshot for report: {_hermes_snap}")

                print(f"    status={rl_result.get('status')}  "

                      f"miss={rl_result.get('best_metrics', {}).get('miss_distance', '?')}")

            except Exception as _se:

                print(f"  Failed to load Hermes RL snapshot ({_se}); report will omit RL data.")

    _run_end = datetime.now()

    # Determine which script was actually used (for report)

    _design_script = hermes_generated_script or matlab_script_path or ""

    _plan_reason = (task_plan.reason if task_plan and hasattr(task_plan, 'reason') else "")

    # Persist RL/workflow snapshot for experiments/runner.py CSV writeback.
    if rl_result and isinstance(rl_result, dict) and rl_result.get("best_metrics"):
        try:
            from multi_agent.run_file_manager import hermes_rl_best_path as _hrl_best
            import json as _json_snap
            _snap_out = _hrl_best()
            _snap_out.parent.mkdir(parents=True, exist_ok=True)
            _payload = {
                "status": rl_result.get("status", "done"),
                "best_params": rl_result.get("best_params") or {},
                "best_metrics": rl_result.get("best_metrics") or {},
                "best_reward": rl_result.get("best_reward", 0.0),
                "total_episodes": rl_result.get("total_episodes", 0),
                "outer_iterations": int(iteration) if iteration else len(iteration_records or []),
                "task_mode": task_mode,
                "best_selection": task_mode
                or rl_result.get("best_selection")
                or rl_result.get("status")
                or "GSMultiAgent",
            }
            _snap_out.write_text(
                _json_snap.dumps(_payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"  [Snapshot] wrote {_snap_out}")
        except Exception as _snap_exc:
            print(f"  [Snapshot] write failed: {_snap_exc}")

    report = generate_report(args.prompt, initial_auto_params, rl_result,

                             run_start=_run_start, run_end=_run_end,

                             task_mode=task_mode,

                             design_script_path=_design_script,

                             task_plan_reason=_plan_reason,

                             modify_law_description=hermes_modify_law_desc,

                             pe_retrieval_info=_pe_quality_info,

                             iteration_records=iteration_records,

                             ablation_cfg=ablation_cfg)

    

    _report_ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    report_path = Path(f"./guidance_output/report_{_report_ts}.md")

    report_path.parent.mkdir(parents=True, exist_ok=True)

    with open(report_path, "w", encoding="utf-8") as f:

        f.write(report)

    # Also write a fixed-name symlink / copy for tooling that expects report.md

    _latest_path = Path("./guidance_output/report_latest.md")

    try:

        _latest_path.write_text(report, encoding="utf-8")

    except Exception:

        pass

    print(f"  Report saved to: {report_path}")

    print(f"  Latest link  →  {_latest_path}")

    print("\nWorkflow Complete!")



def _parse_miss_threshold(prompt: str):

    """

    Extract a miss-distance requirement from the task prompt.



    Recognised patterns (case-insensitive, Chinese or English):

      脱靶量 < 3 m    脱靶量≤5    脱靶量要求3m   脱靶量不超过4

      miss < 3 m      miss distance <= 2.5   miss_dist < 10

    Returns the numeric threshold in metres, or None if not found.

    """

    patterns = [

        r'脱靶量\s*[<\u2264=]{1,2}\s*([\d.]+)',

        r'脱靶量(?:要求|不超过|小于|限制)?\s*([\d.]+)\s*m?',

        r'miss(?:_?dist(?:ance)?)?\s*[<\u2264=]{1,2}\s*([\d.]+)',

        r'miss(?:_?dist(?:ance)?)?\s*(?:less than|no more than|under|within)\s*([\d.]+)',

    ]

    for pat in patterns:

        m = re.search(pat, prompt, re.IGNORECASE)

        if m:

            try:

                return float(m.group(1))

            except ValueError:

                pass

    return None





def _validate_matlab_script_file(path: str):

    """Return ``(is_valid, reason)`` for *path* using the same content

    heuristic the RL optimizer uses internally.



    Wraps :func:`multi_agent.rl.matlab_rl_optimizer._looks_like_matlab_script`

    with a defensive read so a missing/unreadable file is reported as

    invalid rather than crashing the caller.

    """

    try:

        from multi_agent.rl.matlab_rl_optimizer import _looks_like_matlab_script

    except Exception as exc:                                   # pragma: no cover

        return True, f"validator unavailable ({exc}); proceeding"

    try:

        with open(path, "r", encoding="utf-8", errors="ignore") as fh:

            content = fh.read()

    except OSError as exc:

        return False, f"cannot read file: {exc}"

    return _looks_like_matlab_script(content)





def _parse_model_conditions(fname: str) -> dict:
    """Extract {cat: [subcases]} from a MATLAB script filename.
    Matches patterns like: guidance_simulation_B2_..., *_C5_*, *_B_*
    """
    conds: dict = {}
    for m in re.finditer(r'_([A-F])(\d*)_', fname):
        cat = m.group(1).upper()
        sub = int(m.group(2)) if m.group(2) else 0
        if cat not in conds:
            conds[cat] = [sub] if sub else []
    return conds


def _condition_match_score(
    file_conds: dict,
    req_conds: dict,
) -> float:
    """Return [0,1] match score between a file's embedded conditions and
    the requested mission_conditions.
    1.0 = category + subcase exact match
    0.5 = category matches but subcase unknown / not embedded
    0.3 = same category, different subcase
    0.0 = different category
    """
    if not req_conds:
        return 0.5
    if not file_conds:
        return 0.0
    req_cats  = set(req_conds.keys())
    file_cats = set(file_conds.keys())
    cat_overlap = req_cats & file_cats
    if not cat_overlap:
        return 0.0
    scores = []
    for cat in cat_overlap:
        req_subs  = set(req_conds[cat])
        file_subs = set(file_conds[cat])
        if not req_subs or not file_subs:
            scores.append(0.5)
        elif req_subs & file_subs:
            scores.append(1.0)
        else:
            scores.append(0.3)
    return sum(scores) / len(scores)


def _find_latest_matlab_script(*search_dirs: str):

    """Return the path to the most recently modified *valid* .m file in any of the given dirs.

    Skips:
      - RL temp files (_rl_tmp*, rltmp_*, *_rltmp_*)
      - Placeholder scripts (contain stub sim_s that ignores parameters)
    """

    latest_path = None

    latest_mtime = 0.0

    for d in search_dirs:

        if not os.path.isdir(d):

            continue

        for fname in os.listdir(d):

            # Skip every RL temp filename pattern so a stale temp file from a

            # crashed RL run doesn't get picked up as the "latest" user script.

            if (

                fname.endswith(".m")

                and not fname.startswith("_rl_tmp")        # legacy pattern

                and not fname.startswith("rltmp_")          # new safe pattern

                and "_rltmp_" not in fname                  # function-file pattern

            ):

                fp = os.path.join(d, fname)

                try:

                    mt = os.path.getmtime(fp)

                    if mt > latest_mtime:

                        # Reject placeholder scripts with stub sim_s that
                        # generates random metrics without using params.
                        # These break RL because metrics don't respond to
                        # parameter changes.  Also reject tiny fragments
                        # (< 500 bytes) that aren't real simulation scripts.
                        if os.path.getsize(fp) < 500:
                            continue
                        _content = open(fp, "r", encoding="utf-8", errors="ignore").read(4000)
                        if "Placeholder simulation function" in _content:
                            continue
                        if "sim_s(p) %#ok<INUSD>" in _content:
                            continue
                        # Must contain at least a function header or key
                        # structural keyword to be a real MATLAB script
                        if not any(kw in _content for kw in ("function ", "for ", "while ", "N_MC")):
                            continue

                        latest_mtime = mt

                        latest_path = fp

                except OSError:

                    pass

    return latest_path





def _fmt(val):

    """Format number, showing nan for NaN values"""

    if isinstance(val, float) and math.isnan(val):

        return "nan"

    if isinstance(val, float):

        return f"{val:.4f}"

    return str(val)



def generate_report(

    user_input: str,

    init_auto_params: dict,

    rl_result,

    run_start=None,

    run_end=None,

    task_mode: str = "TUNE_PARAMS",

    design_script_path: str = "",

    task_plan_reason: str = "",

    modify_law_description: str = "",

    pe_retrieval_info: dict = None,

    iteration_records: list = None,

    ablation_cfg=None,

) -> str:

    """

    Generate a Markdown report for RL-based guidance parameter optimisation.



    Parameters

    ----------

    user_input       : original prompt string

    init_auto_params : autopilot design params before RL (dp.* dict, all float)

    rl_result        : dict returned by MatlabRLOptimizer.optimize(), or None

    """

    _rl = rl_result or {}
    best_params:  dict = _rl.get("best_params",  {}) or {}
    best_metrics: dict = _rl.get("best_metrics", {}) or {}
    history:      list = _rl.get("history",      []) or []

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    _start_str = run_start.strftime("%Y-%m-%d %H:%M:%S") if run_start else now

    _end_str   = run_end.strftime("%Y-%m-%d %H:%M:%S")   if run_end   else now

    _elapsed   = (run_end - run_start) if (run_start and run_end) else None

    if _elapsed is not None:

        _total_s = int(_elapsed.total_seconds())

        _elapsed_str = f"{_total_s // 3600:02d}:{(_total_s % 3600) // 60:02d}:{_total_s % 60:02d}"

    else:

        _elapsed_str = "N/A"

    _mode_label = {

        "TUNE_PARAMS":    "TUNE_PARAMS — 参数调优（PPO-RL 优化 9 维参数：驾驶仪 w/ζ/τ ×2通道 + w3/ζ3 + 制导律 N_pn）",

        "MODIFY_LAW":     "MODIFY_LAW — 修改制导律（LLM 重写 guidance_local 函数）",

        "REUSE_HISTORY":  "REUSE_HISTORY — 复用已有模型（直接使用知识库/历史参数）",

    }.get(task_mode, task_mode)

    lines = [

        "# 制导系统强化学习参数优化报告",

        "",

        f"**程序启动时间**: {_start_str}",

        f"**程序结束时间**: {_end_str}",

        f"**总耗时**      : {_elapsed_str}",

        "",

        "## 任务描述",

        f"用户输入: {user_input}",

        "",

        "## 设计路径",

        "",

        f"**执行模式**: {_mode_label}",

    ]

    if task_plan_reason:

        lines.append(f"")

        lines.append(f"**规划依据**: {task_plan_reason[:400]}")

    if design_script_path:

        lines.append(f"")

        lines.append(f"**使用脚本**: `{design_script_path}`")

    # ── MODIFY_LAW: show Chinese description of guidance law changes ────────

    if task_mode == "MODIFY_LAW":

        lines += ["", "### 制导律改动说明", ""]

        _desc = modify_law_description or task_plan_reason or ""

        if _desc:

            # Break into readable bullet points if there are Chinese sentence separators

            _desc_clean = _desc[:600].strip()

            lines.append(_desc_clean)

        else:

            lines.append("> 制导律已按任务要求重写（无详细描述可用）。")

        if design_script_path:

            lines.append(f"")

            lines.append(f"> 详细模型文件：`{os.path.basename(design_script_path)}`")

        lines.append("")

    # ── REUSE_HISTORY: show which model was reused ──────────────────────────

    elif task_mode == "REUSE_HISTORY":

        _script_name = os.path.basename(design_script_path) if design_script_path else "（知识库模板）"

        _src = "knowledge_base 模板" if "knowledge_base" in design_script_path else "parameter_experience / 历史最优"

        lines += [

            "",

            "### 复用模型信息",

            "",

            f"- **脚本文件**: {_script_name}",

            f"- **来源**: {_src}",

            "",

        ]

    else:

        lines.append("")

    lines.append("")

    # ── Ablation configuration ───────────────────────────────────────────────────────────────────

    if ablation_cfg is not None:

        _abl_fields = [

            ("rl_optimization",            "RL 优化 (Step 3 PPO)"),

            ("optimization_workflow",       "优化工作流 (Steps 3~4)"),

            ("hermes_rl_tool",              "Hermes RL 工具 (matlab_rl_optimize)"),

            ("parameter_experience_reuse",  "参数经验复用 (PE 热启动)"),

            ("reflection_agent",            "反思智能体 (Step 3.5)"),

            ("memory_search",               "Hermes 持久记忆 (agent_memory + session_search)"),

        ]

        _abl_rows = []

        _any_disabled = False

        for _field, _label in _abl_fields:

            _enabled = getattr(ablation_cfg, _field, True)

            _status = "✅ 开启" if _enabled else "❌ 关闭"

            if not _enabled:

                _any_disabled = True

            _abl_rows.append(f"| {_label} | {_status} |")

        _abl_note = (

            "> ⚠️ 本次运行存在已关闭的消融模块，下表带 ❌ 的模块在本次实验中被移除。"

            if _any_disabled else

            "> 所有消融模块均已开启，本次为完整流程运行。"

        )

        lines += [

            "## 消融实验配置",

            "",

            _abl_note,

            "",

            "| 模块 | 状态 |",

            "|------|------|",

        ] + _abl_rows + [""]

    # ── Multi-iteration design path ───────────────────────────────────────────────────────────────────

    if iteration_records:

        _outcome_labels = {

            "accepted":                         "满足要求 ✅",

            "task_requirements_met":             "提前满足 ✅",

            "continue_optimize":                "继续优化 🔄",

            "max_iterations_reached":            "达到上限 ⏹",

            "ablation_optim_workflow_disabled":  "消融逻辑截止",

            "ablation_reflection_disabled":      "反思关闭",

        }

        _n_iter = len(iteration_records)

        lines += [

            "## 多轮迭代设计路径",

            "",

        ]

        if _n_iter == 1:

            lines.append("> 本次运行仅执行了 **1** 轮迭代。")

        else:

            lines.append(f"> 共执行 **{_n_iter}** 轮迭代设计。")

        # ── Summary table ─────────────────────────────────────────────────

        lines += [

            "",

            "| 迭代 | 执行模式 | RL轮次 | 命中率(%) | SEP(m) | PeakNy(g) | PM(°) | BW(r/s) | 本轮结果 |",

            "|------|---------|--------|----------|--------|-----------|-------|---------|------|",

        ]

        for _rec in iteration_records:

            _ht  = _rec.get("rl_hit")
            _ms  = _rec.get("rl_miss")
            _pm_v = _rec.get("rl_pm")
            _bw_v = _rec.get("rl_bw")
            _ny_v = _rec.get("rl_ny")
            _ep   = _rec.get("rl_episodes", 0)

            lines.append(

                f"| {_rec['iteration']} "
                f"| {_rec.get('task_mode', '—')} "
                f"| {_ep if _ep else '—'} "
                f"| {f'{_ht:.1f}' if _ht is not None else '—'} "
                f"| {f'{_ms:.2f}' if _ms is not None else '—'} "
                f"| {f'{_ny_v:.1f}' if _ny_v else '—'} "
                f"| {f'{_pm_v:.1f}' if _pm_v is not None else '—'} "
                f"| {f'{_bw_v:.2f}' if _bw_v else '—'} "
                f"| {_outcome_labels.get(_rec.get('outcome', ''), _rec.get('outcome', '—'))} |"

            )

        lines.append("")

        # ── Per-iteration detail cards (always shown) ──────────────────

        _param_nom = {
            "w1": 40.0, "zeta1": 0.75, "tao1": 0.20,
            "w2": 35.0, "zeta2": 0.75, "tao2": 0.20,
            "w3": 40.0, "zeta3": 0.70, "N_pn": 4.0,
        }
        _param_label = {
            "w1": "w1 (俯仰带宽 r/s)", "zeta1": "ζ1 (俯仰阻尼)", "tao1": "τ1 (俯仰时间常数 s)",
            "w2": "w2 (偏航带宽 r/s)", "zeta2": "ζ2 (偏航阻尼)", "tao2": "τ2 (偏航时间常数 s)",
            "w3": "w3 (滚转带宽 r/s)", "zeta3": "ζ3 (滚转阻尼)",
            "N_pn": "N_pn (导引系数)",
        }

        lines += ["### 各轮迭代详情", ""]

        for _rec in iteration_records:

            _imode   = _rec.get("task_mode", "—")
            _iout    = _outcome_labels.get(_rec.get("outcome", ""), _rec.get("outcome", "—"))
            _ireason = _rec.get("plan_reason", "")
            _ibp     = _rec.get("best_params_iter", {})
            _ibm     = _rec.get("best_metrics_iter", {})
            _iml     = _rec.get("modify_law_desc", "")
            _isc     = _rec.get("generated_script", "")
            _isugg   = _rec.get("suggestion", "")
            _iep     = _rec.get("rl_episodes", 0)
            _irew    = _rec.get("rl_best_reward", 0.0)

            lines += [
                f"---",
                f"",
                f"#### 迭代 {_rec['iteration']}  `{_imode}` → {_iout}",
                f"",
            ]

            # ── 1. 设计路径选择与判断原因 ──────────────────────────────

            lines += ["##### ① 设计路径选择与判断原因", ""]
            _mode_cn = {
                "TUNE_PARAMS":   "**TUNE_PARAMS** — PPO-RL 优化 9 维驾驶仪/制导参数",
                "MODIFY_LAW":    "**MODIFY_LAW** — LLM 重写制导律 `gf()` 函数",
                "REUSE_HISTORY": "**REUSE_HISTORY** — 直接复用知识库/历史最优参数",
            }.get(_imode, f"**{_imode}**")
            lines.append(f"**执行模式**: {_mode_cn}")
            lines.append(f"")
            if _ireason:
                lines.append(f"**判断原因**:")
                lines.append(f"")
                lines.append(f"> {_ireason}")
                lines.append(f"")
            else:
                lines.append(f"> *(无规划依据记录)*")
                lines.append(f"")

            # ── 2. 模式专属说明 ────────────────────────────────────────

            if _imode == "REUSE_HISTORY":

                lines += ["##### ② 历史复用说明", ""]
                _h   = _ibm.get("hit_rate", _ibm.get("hitrate"))
                _sep = _ibm.get("SEP", _ibm.get("miss_distance"))
                _pm  = _ibm.get("pitch_PM")
                _bw  = _ibm.get("pitch_BW")
                _ny  = _ibm.get("peak_ny", _ibm.get("peak_n"))
                _parts = []
                if _h  is not None: _parts.append(f"命中率 {float(_h):.1f}%")
                if _sep is not None: _parts.append(f"SEP {float(_sep):.2f} m")
                if _ny  is not None: _parts.append(f"PeakNy {float(_ny):.1f} g")
                if _pm  is not None: _parts.append(f"PM {float(_pm):.1f}°")
                if _bw  is not None: _parts.append(f"BW {float(_bw):.2f} r/s")
                lines.append(f"历史最优指标: {' | '.join(_parts) if _parts else '（无）'}")
                lines.append("")
                if _ibp:
                    lines += [
                        "**复用参数** (来自 PE 历史记忆):",
                        "",
                        "| 参数 | 复用值 |",
                        "|------|-------|",
                    ]
                    for _pk, _pv in _ibp.items():
                        _plbl = _param_label.get(_pk, _pk)
                        lines.append(f"| {_plbl} | {float(_pv):.4g} |")
                    lines.append("")

            elif _imode == "MODIFY_LAW":

                lines += ["##### ② 制导律修改说明", ""]
                if _iml:
                    lines.append(f"> {_iml}")
                else:
                    lines.append(f"> *(无修改说明记录)*")
                lines.append("")
                if _isc:
                    lines.append(f"**生成脚本**: `{_isc}`")
                    lines.append("")

            else:  # TUNE_PARAMS

                lines += ["##### ② 参数调整说明", ""]
                if _ibp:
                    lines += [
                        "| 参数 | 标称初值 | 本轮优化值 | 变化量 | 变化方向 |",
                        "|------|---------|-----------|-------|--------|",
                    ]
                    for _pk, _plbl in _param_label.items():
                        _nom = _param_nom.get(_pk, 0.0)
                        _opt = float(_ibp.get(_pk, _nom))
                        _dlt = _opt - _nom
                        _dir = "↑" if _dlt > 1e-4 else ("↓" if _dlt < -1e-4 else "—")
                        lines.append(f"| {_plbl} | {_nom:.4g} | {_opt:.4g} | {_dlt:+.4g} | {_dir} |")
                    lines.append("")
                else:
                    lines.append(f"> *(参数未记录，可能复用了历史初始值)*")
                    lines.append("")
                if _isc:
                    lines.append(f"**脚本**: `{_isc}`")
                    lines.append("")

            # ── 3. RL 优化结果 ────────────────────────────────────────

            if _ibm:

                _r_hit = float(_ibm.get("hit_rate", 0))
                _r_sep = float(_ibm.get("SEP", _ibm.get("miss_distance", 99)))
                _r_ny  = float(_ibm.get("peak_ny", _ibm.get("peak_n", 0)))
                _r_pm  = float(_ibm.get("pitch_PM", 0))
                _r_bw  = float(_ibm.get("pitch_BW", 0))

                _ok = lambda v, lo, hi: "✅" if lo <= v <= hi else "⚠️"
                _ok_hit = "✅" if _r_hit >= 92.0 else "⚠️"
                _ok_sep = "✅" if _r_sep <= 7.0 else "⚠️"
                _ok_ny  = "✅" if _r_ny  <= 20.0 else "⚠️"
                _ok_pm  = _ok(_r_pm, 45.0, 70.0)
                _ok_bw  = _ok(_r_bw, 20.0, 85.0)

                lines += [
                    "##### ③ RL 优化结果",
                    "",
                    f"> RL 轮次: {_iep}　最优奖励: {_irew:.4f}",
                    "",
                    "| 指标 | 本轮结果 | 要求门限 | 评估 |",
                    "|------|---------|---------|------|",
                    f"| 命中率    | {_r_hit:.1f} %   | ≥ 92%         | {_ok_hit} |",
                    f"| SEP      | {_r_sep:.2f} m   | ≤ 7 m         | {_ok_sep} |",
                    f"| PeakNy   | {_r_ny:.1f} g    | ≤ 20 g        | {_ok_ny} |",
                    f"| PM 相位裕度 | {_r_pm:.1f} °  | [45°, 70°]   | {_ok_pm} |",
                    f"| BW 带宽   | {_r_bw:.2f} r/s | [20, 85] r/s | {_ok_bw} |",
                    "",
                ]

            # ── 4. 反思建议 ────────────────────────────────────────────

            if _isugg:

                lines += [
                    "##### ④ 反思建议（传入下一轮规划器）",
                    "",
                    f"> {_isugg}",
                    "",
                ]

        lines.append("")

    # ── PE retrieval quality score ───────────────────────────────────────────────────────────────────

    if pe_retrieval_info and pe_retrieval_info.get("retrieval_quality"):

        _rq   = pe_retrieval_info["retrieval_quality"]

        _mid  = pe_retrieval_info.get("memory_id", "")[:12]

        _mtyp = pe_retrieval_info.get("memory_type", "unknown")

        _mtyp_cn = "长期记忆 (LONG_TERM)" if _mtyp == "long_term" else "短期记忆 (SHORT_TERM)"

        _q    = float(_rq.get("quality_score",  0.0))

        _qs   = float(_rq.get("sim_score",      0.0))

        _qf   = float(_rq.get("fitness_score",  0.0))

        _qr   = float(_rq.get("reliability",    0.6))

        _grade = _rq.get("grade", "")

        _ac   = int(_rq.get("access_count", 0))

        _objs = pe_retrieval_info.get("objectives", {})

        lines += [

            "## PE 历史经验检索质量评分",

            "",

            f"> 检索来源: **{_mtyp_cn}** | memory_id: `{_mid}` | 历史引用次数: {_ac}",

            "",

            "| 检索质量维度 | 得分 | 权重 | 加权贡献 |",

            "|------------|:----:|:----:|:-------:|",

            f"| 上下文相似度 (similarity) | {_qs:.3f} | 0.40 | {0.4 * _qs:.3f} |",

            f"| 参数适应度   (fitness)    | {_qf:.3f} | 0.40 | {0.4 * _qf:.3f} |",

            f"| 可信度       (reliability)| {_qr:.3f} | 0.20 | {0.2 * _qr:.3f} |",

            f"| **综合检索质量 Q**         | — | — | **{_q:.4f}  [{_grade}]** |",

            "",

            "> 可信度: LONG_TERM = 1.0 (已经反思智能体验证)，SHORT_TERM = 0.6",

            "> Q ≥ 0.75 高可信 — 直接热启动 RL ｜ 0.5–0.75 中等 ｜ < 0.5 低可信",

            "",

        ]

        if _objs:

            _h  = _objs.get("hit_rate",      _objs.get("hitrate",      None))

            _ms = _objs.get("miss_distance", _objs.get("miss_dist",    None))

            _pm = _objs.get("pitch_PM",      _objs.get("pm",           None))

            _pn = _objs.get("peak_n",        _objs.get("peak_N",       None))

            _parts = []

            if _h  is not None: _parts.append(f"命中率 {float(_h):.1f}%")

            if _ms is not None: _parts.append(f"脱靶量 {float(_ms):.3f} m")

            if _pm is not None: _parts.append(f"相位裕度 {float(_pm):.1f}°")

            if _pn is not None and float(_pn) > 0: _parts.append(f"峰值过载 {float(_pn):.2f} g")

            if _parts:

                lines += [

                    "**历史最优指标** (from PE):",

                    "",

                    "  " + "  |  ".join(_parts),

                    "",

                ]

    # ── Initial autopilot params ─────────────────────────────────────────────

    lines += [

        "## 初始参数",

        "",

        "| 参数 | 初始値 | 说明 |",

        "|------|--------|------|",

        f"| w1    | {_fmt(float(init_auto_params.get('w1',    40.0)))} | 信仰漏道带宽 (rad/s) |",

        f"| ζ1    | {_fmt(float(init_auto_params.get('zeta1',  0.75)))} | 信仰漏道阻尼比 |",

        f"| τ1    | {_fmt(float(init_auto_params.get('tao1',   0.20)))} | 信仰漏道设计时间常数 (s) |",

        f"| w2    | {_fmt(float(init_auto_params.get('w2',    35.0)))} | 偏航漏道带宽 (rad/s) |",

        f"| ζ2    | {_fmt(float(init_auto_params.get('zeta2',  0.75)))} | 偏航漏道阻尼比 |",

        f"| τ2    | {_fmt(float(init_auto_params.get('tao2',   0.20)))} | 偏航漏道设计时间常数 (s) |",

        f"| w3    | {_fmt(float(init_auto_params.get('w3',    40.0)))} | 滚转漏道带宽 (rad/s) |",

        f"| ζ3    | {_fmt(float(init_auto_params.get('zeta3',  0.70)))} | 滚转漏道阻尼比 |",

        f"| N_pn  | {_fmt(float(init_auto_params.get('N_pn',   4.0)))} | 比例导引系数 |",

        "",

    ]

    # ... (rest of the code remains the same)

    _param_defaults = {
        "w1": 40.0, "zeta1": 0.75, "tao1": 0.20,
        "w2": 35.0, "zeta2": 0.75, "tao2": 0.20,
        "w3": 40.0, "zeta3": 0.70,
        "N_pn": 4.0,
    }

    param_labels = {

        "w1":    "w1 (信仰漏道带宽 rad/s)",

        "zeta1": "ζ1 (信仰阻尼比)",

        "tao1":  "τ1 (信仰设计时间常数 s)",

        "w2":    "w2 (偏航漏道带宽 rad/s)",

        "zeta2": "ζ2 (偏航阻尼比)",

        "tao2":  "τ2 (偏航设计时间常数 s)",

        "w3":    "w3 (滚转漏道带宽 rad/s)",

        "zeta3": "ζ3 (滚转阻尼比)",

        "N_pn":  "N_pn (比例导引系数)",

    }

    for k, label in param_labels.items():

        v_init = float(init_auto_params.get(k, _param_defaults.get(k, 0.0)))

        v_opt  = float(best_params.get(k, v_init))

        delta  = v_opt - v_init

        lines.append(f"| {label} | {_fmt(v_init)} | {_fmt(v_opt)} | {delta:+.4f} |")

    lines.append("")


    # ... (rest of the code remains the same)

    # ── Performance metrics ───────────────────────────────────────────────────

    _sep    = float(best_metrics.get('SEP',      best_metrics.get('miss_distance', 99.0)))
    _ny     = float(best_metrics.get('peak_ny',  best_metrics.get('peak_n',        0.0)))
    _pm     = float(best_metrics.get('pitch_PM', 0.0))
    _gm     = float(best_metrics.get('pitch_GM', 0.0))
    _bw     = float(best_metrics.get('pitch_BW', 0.0))
    _hit    = float(best_metrics.get('hit_rate', 0.0))

    _pm_ok  = "✅" if 45.0 <= _pm <= 70.0 else ("⚠️偏低" if _pm < 45.0 else "⚠️偏高")
    _bw_ok  = "✅" if 20.0 <= _bw <= 85.0 else ("⚠️偏低" if _bw < 20.0 else "⚠️偏高")
    _ny_ok  = "✅" if _ny <= 20.0 else "⚠️超限"
    _hit_ok = "✅" if _hit >= 92.0 else "⚠️偏低"
    _sep_ok = "✅" if _sep <= 7.0  else "⚠️偏大"

    lines += [

        "## 仿True真性能指标 (RL最优参数)",

        "",

        "| 指标 | 值 | 要求门限 | 评估 |",

        "|------|-----|---------|------|",

        f"| 命中率       | {_fmt(_hit)} % | ≥ 92%           | {_hit_ok} |",

        f"| SEP 脱靶量     | {_fmt(_sep)} m  | ≤ 7 m            | {_sep_ok} |",

        f"| PeakNy 峰値法向过载 | {_fmt(_ny)} g  | ≤ 20 g           | {_ny_ok} |",

        f"| PM 相位裕度  | {_fmt(_pm)} °  | [45°, 70°]      | {_pm_ok} |",

        f"| BW 带宽       | {_fmt(_bw)} r/s | [20, 85] r/s   | {_bw_ok} |",

    ]

    if _gm > 0:
        lines.append(f"| GM 幅値裕度  | {_fmt(_gm)} dB | —              | — |")

    lines.append("")

    # ── RL Optimization Statistics ───────────────────────────────────────────
    _total_episodes = _rl.get("total_episodes", 0)
    _total_mc_calls = _rl.get("total_mc_calls", 0)

    if _total_episodes > 0 or _total_mc_calls > 0:
        lines += [
            "## RL 优化统计",
            "",
            "| 统计项 | 数值 | 说明 |",
            "|--------|------|------|",
            f"| 总训练轮次 | {_total_episodes} | PPO 优化的 episode 总数 |",
        ]

        if _total_mc_calls > 0:
            lines.append(f"| Monte Carlo 调用次数 | {_total_mc_calls} | 包括基线评估 + 每轮训练的 MC 仿真 |")
            if _total_episodes > 0:
                _avg_mc_per_ep = _total_mc_calls / _total_episodes
                lines.append(f"| 平均每轮 MC 次数 | {_avg_mc_per_ep:.1f} | total_mc_calls / total_episodes |")

        lines.append("")

    # ... (rest of the code remains the same)

    # ── PE fitness score ─────────────────────────────────────────────────────

    try:
        from multi_agent.memory.parameter_experience import ParameterExperience as _PE
        # Pass config reward_weights ranges so fitness uses correct BW/PM bounds
        _rw_kw: Dict[str, float] = {}
        try:
            _cfg_rw = config.rl_optimization.get("reward_weights", {})
            for _k in ("bw_min", "bw_max", "pm_min", "pm_max", "peak_ny_max"):
                if _cfg_rw.get(_k) is not None:
                    _rw_kw[_k] = float(_cfg_rw[_k])
        except Exception:
            pass
        _pe_fitness = _PE.compute_fitness_from_objectives(best_metrics, **_rw_kw)
        _pe_subs = _PE.compute_fitness_from_objectives.__doc__  # reference only
    except Exception:
        _pe_fitness = 0.0

    # Decompose for display
    _pf_hit  = max(0.0, min(1.0, (_hit - 80.0) / 20.0))
    _pf_sep  = max(0.0, 1.0 - _sep / 10.0)
    _pf_ny   = 1.0 if (0 < _ny <= 20.0) else (max(0.0, 1.0 - (_ny - 20.0) / 20.0) if _ny > 0 else 0.5)
    _pf_pm   = 1.0 if (45.0 <= _pm <= 70.0) else max(0.0, 0.5 * (1.0 - max(45.0 - _pm, _pm - 70.0, 0.0) / 45.0))
    _pf_bw   = 1.0 if (20.0 <= _bw <= 85.0) else max(0.0, 0.5 * (1.0 - max(20.0 - _bw, _bw - 85.0, 0.0) / 20.0))

    lines += [

        "## 经验适应度分数 (fitness)",

        "",

        "> 存入 ParameterExperience 的综合适应度，影响下次同类任务检索时的参数推荐排序。",

        "",

        "| 指标 | 子得分 | 评分规则 | 权重 | 加权贡献 |",

        "|------|:-----:|------|:----:|:-------:|",

        f"| 命中率 (hit_rate ↑)       | {_pf_hit:.3f} | (hit-80)/20, 夹界[0,1] | 3/8 | {3/8 * _pf_hit:.3f} |",

        f"| SEP 脱靶量 (SEP ↓)          | {_pf_sep:.3f} | 1-SEP/10, 夹界[0,1]  | 2/8 | {2/8 * _pf_sep:.3f} |",

        f"| PeakNy (≤ 20 g 硬限)         | {_pf_ny:.3f}  | ≤限 1.0 超限按超出比例扣 | 1/8 | {1/8 * _pf_ny:.3f} |",

        f"| PM 相位裕度 ([45°,70°]范围) | {_pf_pm:.3f}  | 在范围 1.0 否则按距边界衰减 | 1/8 | {1/8 * _pf_pm:.3f} |",

        f"| BW 带宽 ([20,85] r/s 范围) | {_pf_bw:.3f}  | 在范围 1.0 否则按距边界衰减 | 1/8 | {1/8 * _pf_bw:.3f} |",

        f"| **综合适应度**                | — | (3×hit+2×SEP+ny+PM+BW)/8 | — | **{_pe_fitness:.4f}** |",

        "",

    ]


    # ... (rest of the code remains the same)

    # ── Episode reward history ────────────────────────────────────────────────

    if history:

        lines += [

            "## 近期奖励历史 (最后若干轮)",

            "",

            "| 轮次 | 奖励 | 命中率 | SEP (m) | PeakNy (g) | PM (°) | BW (r/s) |",

            "|------|------|--------|---------|-----------|-------|---------|",

        ]

        for ep in history:

            em = ep.get("metrics", {})

            _e_sep = float(em.get('SEP', em.get('miss_distance', 99.0)))
            _e_ny  = float(em.get('peak_ny', em.get('peak_n', 0.0)))
            _e_pm  = float(em.get('pitch_PM', 0.0))
            _e_bw  = float(em.get('pitch_BW', 0.0))

            lines.append(

                f"| {ep['episode']:4d} "

                f"| {_fmt(float(ep['reward']))} "

                f"| {_fmt(float(em.get('hit_rate', 0)))}% "

                f"| {_fmt(_e_sep)} "

                f"| {_fmt(_e_ny) if _e_ny > 0 else 'N/A'} "

                f"| {_fmt(_e_pm)} "

                f"| {_fmt(_e_bw) if _e_bw > 0 else 'N/A'} |"

            )

        lines.append("")



    return "\n".join(lines)



if __name__ == "__main__":

    import sys
    from multi_agent.logging.run_logger import init_run_logging

    _log_path = init_run_logging(argv=sys.argv)
    print(f"[RunLog] 运行日志: {_log_path}")

    parser = argparse.ArgumentParser(description="Multi-Agent Guidance CLI Agent")

    parser.add_argument("--skip-preflight", action="store_true",
                        help="跳过启动前环境/配置/解析器检查")
    parser.add_argument("--preflight-only", action="store_true",
                        help="仅运行启动前检查，不执行主流程")
    parser.add_argument("--preflight-strict", action="store_true",
                        help="启动前检查：警告也视为失败")
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="详细日志（等同 logging.verbosity=verbose）",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="精简日志（等同 logging.verbosity=quiet）",
    )
    parser.add_argument(
        "--resume-checkpoint",
        type=str,
        default=None,
        help="PPO checkpoint 路径（.json 文件、stem 或含 latest.json 的目录）",
    )
    parser.add_argument(
        "--mc-data-path",
        type=str,
        default=None,
        help="外部 Monte Carlo .mat 数据路径（实验复现实用）",
    )
    parser.add_argument(
        "--async-run",
        action="store_true",
        help="后台异步提交任务，立即返回 task_id（可通过 --task-status 轮询）",
    )
    parser.add_argument(
        "--task-status",
        type=str,
        default=None,
        help="查询异步任务状态（task_id）",
    )
    parser.add_argument(
        "--serve-tasks",
        action="store_true",
        help="启动 HTTP 任务状态服务（GET /api/tasks/{id}）",
    )
    parser.add_argument(
        "--serve-port",
        type=int,
        default=8765,
        help="--serve-tasks HTTP 端口（默认 8765）",
    )

    group = parser.add_mutually_exclusive_group(required=False)

    group.add_argument("--prompt", type=str, help="User task description as string")

    group.add_argument("--file", type=str, help="Path to a text file containing the user task description")

    args = parser.parse_args()

    if getattr(args, "verbose", False) and getattr(args, "quiet", False):
        print("Error: --verbose 与 --quiet 不能同时使用。")
        raise SystemExit(2)

    if args.serve_tasks:
        serve = _async_module("http_server").serve
        serve(port=args.serve_port)
        raise SystemExit(0)

    if args.task_status:
        get_task_queue = _async_module("task_queue").get_task_queue
        rec = get_task_queue().get_status(args.task_status)
        if rec is None:
            print(json.dumps({"error": "task not found", "task_id": args.task_status}, ensure_ascii=False))
            raise SystemExit(1)
        print(json.dumps(rec.to_dict(), ensure_ascii=False, indent=2))
        raise SystemExit(0)

    if args.async_run:
        if not getattr(args, "prompt", None) and not getattr(args, "file", None):
            print("Error: --async-run 需要 --prompt 或 --file。")
            raise SystemExit(1)

    if not getattr(args, "prompt", None) and not getattr(args, "file", None):
        print("Error: 需要 --prompt 或 --file。")
        raise SystemExit(2)

    if args.file:
        try:
            with open(args.file, "r", encoding="utf-8") as f:
                args.prompt = f.read().strip()
        except FileNotFoundError:
            print(f"Error: Prompt file '{args.file}' not found.")
            raise SystemExit(1)

    if not args.skip_preflight:
        from multi_agent.preflight import ensure_preflight_or_exit
        ensure_preflight_or_exit(
            strict=args.preflight_strict,
            preflight_only=args.preflight_only,
        )
    elif args.preflight_only:
        print("Error: --preflight-only 不能与 --skip-preflight 同时使用。")
        raise SystemExit(1)

    if args.async_run:
        import sys as _sys
        AsyncTaskQueue = _async_module("task_queue").AsyncTaskQueue

        _q = AsyncTaskQueue()
        _argv = [_sys.executable, str(Path(__file__).resolve()), "--skip-preflight"]
        if args.resume_checkpoint:
            _argv.extend(["--resume-checkpoint", args.resume_checkpoint])
        if args.mc_data_path:
            _argv.extend(["--mc-data-path", args.mc_data_path])
        _argv.extend(["--prompt", args.prompt])
        _rec = _q.submit_cli_run(_argv, prompt_preview=args.prompt or "")
        print(json.dumps({
            "task_id": _rec.task_id,
            "status": _rec.status,
            "poll": f"python cli_agent.py --task-status {_rec.task_id}",
            "http": f"GET /api/tasks/{_rec.task_id}",
        }, ensure_ascii=False, indent=2))
        raise SystemExit(0)

    # ── Human-in-the-loop restart loop ───────────────────────────────────────
    # Pressing Ctrl+C at ANY point (even deep inside an await) propagates
    # KeyboardInterrupt up through asyncio.run() to here.
    #
    # Input modes after interrupt:
    #   <修改指令>   → appended to the ORIGINAL prompt so the planner sees
    #                  the full original context + the human modification.
    #                  e.g. "仅将设计路径改为修改制导律，其他保持不变"
    #   !! <新指令>  → prefix with !! to fully REPLACE the original prompt.
    #   <直接回车>   → restore to the pure original prompt and restart.
    #   Ctrl+C again → exit the program.
    #
    # _original_prompt is fixed at startup; every modification is relative
    # to it so instructions don't accumulate across multiple interrupts.
    _SEP = "─" * 64
    _original_prompt = args.prompt          # never mutated after this line

    async def _run_with_task_lifecycle() -> None:
        _tq = _async_module("task_queue")
        current_async_task_id = _tq.current_async_task_id
        get_task_queue = _tq.get_task_queue
        _tid = current_async_task_id()
        _queue = get_task_queue()
        if _tid:
            _queue.update_progress(_tid, status="running", phase="cli_main", message="主流程启动")
        try:
            await main(args)
            if _tid:
                _queue.update_progress(
                    _tid,
                    status="completed",
                    phase="done",
                    message="主流程正常结束",
                )
        except Exception as _exc:
            if _tid:
                _queue.update_progress(
                    _tid,
                    status="failed",
                    phase="error",
                    error=f"{type(_exc).__name__}: {_exc}",
                )
            raise

    while True:
        try:
            asyncio.run(_run_with_task_lifecycle())
            break  # normal completion → exit program
        except KeyboardInterrupt:
            print(f"\n\n{_SEP}")
            print("  [人机中断] Ctrl+C — 当前执行已中止")
            print(_SEP)
            _preview = _original_prompt[:120]
            print(f"  原始任务: {_preview}{'...' if len(_original_prompt) > 120 else ''}")
            print(f"  {_SEP}")
            print("  继续执行（保持当前指令）               → 输入 c 或 continue")
            print("  手动晋升短期记忆到长期记忆             → 输入 p 或 promote")
            print("  输入修改指令（追加到原始任务上）        → 直接输入修改内容")
            print("  完全替换原始任务                       → 以 !! 开头输入新任务")
            print("  恢复原始任务重启                       → 直接回车")
            print("  退出程序                               → 再次 Ctrl+C")
            print(f"  {_SEP}")
            print("  >>> ", end="", flush=True)
            try:
                _new_input = input().strip()
            except (KeyboardInterrupt, EOFError):
                print("\n  [退出] 程序已终止。")
                break
            if _new_input.lower() in ("c", "continue", "继续"):
                # Resume: restart with args.prompt unchanged (may be a prior modification)
                print(f"\n  → 继续执行，重新启动...\n")
            elif _new_input.lower() in ("p", "promote", "晋升"):
                # Manual short-term → long-term promotion
                try:
                    from multi_agent.config_loader import get_config as _get_cfg
                    from multi_agent.memory.parameter_experience import ParameterExperience as _PECls
                    _pe_cfg = _get_cfg()
                    _pe_m = _PECls(
                        experience_base_dir=str(_pe_cfg.model_output.experience_base_dir),
                    )
                    _st_list = _pe_m.get_short_term_summary()
                    if not _st_list:
                        print(f"\n  [晋升] 短期记忆为空，无可晋升条目。")
                    else:
                        print(f"\n  短期记忆条目（共 {len(_st_list)} 条，按 fitness 降序）：")
                        print(f"  {'#':<3} {'memory_id':<14} {'fitness':>8} {'访问':>5} {'时间':>13}  目标值")
                        print(f"  {'-'*3} {'-'*14} {'-'*8} {'-'*5} {'-'*13}  {'-'*32}")
                        for _si, _se in enumerate(_st_list, 1):
                            _obj_s = "  ".join(
                                f"{k}={v}" for k, v in list(_se["objectives"].items())[:3]
                            )
                            print(f"  {_si:<3} {_se['memory_id']:<14} "
                                  f"{_se['fitness']:>8.4f} {_se['access_count']:>5} "
                                  f"{_se['timestamp_iso']:>13}  {_obj_s}")
                        print(f"\n  all→全部晋升 | N(数字)→晋升前N条 | <id前缀>→晋升指定条目 | 回车→跳过")
                        print(f"  >>> ", end="", flush=True)
                        try:
                            _pi = input().strip()
                        except (KeyboardInterrupt, EOFError):
                            _pi = ""
                        if _pi.lower() == "all":
                            _np = asyncio.run(_pe_m.promote_top_k(k=len(_st_list)))
                            print(f"\n  → 已晋升全部 {_np} 条到长期记忆。")
                        elif _pi.isdigit():
                            _np = asyncio.run(_pe_m.promote_top_k(k=int(_pi)))
                            print(f"\n  → 已晋升 top-{_np} 条到长期记忆。")
                        elif _pi:
                            _fid = next(
                                (_se["memory_id"] for _se in _st_list
                                 if _se["memory_id"].startswith(_pi)), _pi
                            )
                            _ok = asyncio.run(_pe_m.promote_to_long_term(_fid))
                            print(f"\n  → {'晋升成功' if _ok else '未找到该条目（短期记忆中无此 ID）'}。")
                        else:
                            print(f"\n  → 跳过晋升。")
                except Exception as _pe_exc:
                    print(f"\n  [晋升] 操作失败: {_pe_exc}")
                # After promote: ask what to do next before restarting
                print(f"\n  {_SEP}")
                print("  晋升完成。接下来（c 继续/修改指令/回车恢复原始/Ctrl+C 退出）：")
                print("  >>> ", end="", flush=True)
                try:
                    _post = input().strip()
                except (KeyboardInterrupt, EOFError):
                    print("\n  [退出] 程序已终止。")
                    break
                if _post.lower() in ("c", "continue", "继续"):
                    print(f"\n  → 继续执行，重新启动...\n")
                elif _post.startswith("!!"):
                    _original_prompt = _post[2:].strip()
                    args.prompt = _original_prompt
                    print(f"\n  → 原始任务已完全替换，重新启动执行...\n")
                elif _post:
                    args.prompt = (
                        f"{_original_prompt}\n\n"
                        f"[人工干预指令] 在上述任务基础上，应用如下调整（优先级高于原始指令中的相同设置）：\n"
                        f"{_post}"
                    )
                    print(f"\n  → 修改指令已追加到原始任务，重新启动执行...\n")
                else:
                    args.prompt = _original_prompt
                    print(f"\n  → 已恢复原始任务指令，重新启动执行...\n")
            elif _new_input.startswith("!!"):
                # Full replacement: update the stored original too
                _original_prompt = _new_input[2:].strip()
                args.prompt = _original_prompt
                print(f"\n  → 原始任务已完全替换，重新启动执行...\n")
            elif _new_input:
                # Partial modification: append to original (not accumulated)
                args.prompt = (
                    f"{_original_prompt}\n\n"
                    f"[人工干预指令] 在上述任务基础上，应用如下调整（优先级高于原始指令中的相同设置）：\n"
                    f"{_new_input}"
                )
                print(f"\n  → 修改指令已追加到原始任务，重新启动执行...\n")
            else:
                # Empty input: restore pure original
                args.prompt = _original_prompt
                print(f"\n  → 已恢复原始任务指令，重新启动执行...\n")

