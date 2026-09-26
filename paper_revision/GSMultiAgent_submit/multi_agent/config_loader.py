#!/usr/bin/env python3
"""
Configuration Loader
Loads settings from config.yaml (unified configuration system)
"""

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import yaml

from multi_agent.metric_contract import (
    BW_MAX,
    BW_MIN,
    HIT_RATE_MIN,
    PEAK_NY_MAX,
    PM_MAX,
    PM_MIN,
    SEP_MAX,
)


def get_env(key: str, default: Any = None) -> Any:
    """Get environment variable with optional default"""
    return os.environ.get(key, default)


@dataclass
class LLMConfig:
    """LLM Configuration"""
    provider: str = "openrouter"
    api_key: Optional[str] = None
    model: str = "anthropic/claude-sonnet-4.6"
    base_url: str = "https://openrouter.ai/api/v1"
    # Ordered external backups used by multi_agent.llm_failover on balance/quota errors.
    fallbacks: List[Dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_env(cls) -> "LLMConfig":
        provider = get_env("LLM_PROVIDER", "openrouter")
        if provider == "openai":
            base_url = get_env("OPENAI_BASE_URL", "https://api.openai.com/v1")
            model = get_env("OPENAI_MODEL", "gpt-4o")
        elif provider == "anthropic":
            base_url = "https://api.anthropic.com"
            model = get_env("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")
        elif provider == "custom":
            base_url = get_env("CUSTOM_BASE_URL", "https://api.custom.com/v1")
            model = get_env("CUSTOM_MODEL", "gpt-4o")
        else:
            base_url = "https://openrouter.ai/api/v1"
            model = get_env("OPENROUTER_MODEL", "anthropic/claude-sonnet-4.6")
        api_key = get_env("OPENROUTER_API_KEY") or get_env("OPENAI_API_KEY") or get_env("ANTHROPIC_API_KEY") or get_env("CUSTOM_API_KEY")
        return cls(provider=provider, api_key=api_key, model=model, base_url=base_url)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LLMConfig":
        env_config = cls.from_env()
        raw_fb = data.get("fallbacks") or []
        fallbacks: List[Dict[str, Any]] = (
            [dict(x) for x in raw_fb if isinstance(x, dict)] if isinstance(raw_fb, list) else []
        )
        return cls(
            provider=data.get("provider", env_config.provider),
            api_key=data.get("api_key", env_config.api_key),
            model=data.get("model", env_config.model),
            base_url=data.get("base_url", env_config.base_url),
            fallbacks=fallbacks,
        )


@dataclass
class RAGConfig:
    """RAG Knowledge Base Configuration"""
    persist_dir: str = "./chroma_db"
    collection: str = "knowledge_base"
    embedding_provider: str = "local"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 384
    embedding_api_key: Optional[str] = None
    embedding_base_url: Optional[str] = None

    @classmethod
    def from_env(cls) -> "RAGConfig":
        return cls(
            persist_dir=get_env("CHROMA_PERSIST_DIR", "./chroma_db"),
            collection=get_env("CHROMA_COLLECTION", "knowledge_base"),
            embedding_provider=get_env("EMBEDDING_PROVIDER", "local"),
            embedding_model=get_env("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
            embedding_dim=int(get_env("EMBEDDING_DIM", "384")),
            embedding_api_key=get_env("EMBEDDING_API_KEY", None),
            embedding_base_url=get_env("EMBEDDING_BASE_URL", None),
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RAGConfig":
        env_config = cls.from_env()
        return cls(
            persist_dir=data.get("persist_dir", env_config.persist_dir),
            collection=data.get("collection", env_config.collection),
            embedding_provider=data.get("embedding_provider", env_config.embedding_provider),
            embedding_model=data.get("embedding_model", env_config.embedding_model),
            embedding_dim=data.get("embedding_dim", env_config.embedding_dim),
            embedding_api_key=data.get("embedding_api_key", env_config.embedding_api_key),
            embedding_base_url=data.get("embedding_base_url", env_config.embedding_base_url),
        )


@dataclass
class AblationConfig:
    """Ablation study component switches.

    Each flag maps to a yaml block::

        ablation:
          rl_optimization:            { enabled: true }
          parameter_experience_reuse: { enabled: true }
          reflection_agent:           { enabled: true }
          hermes_rl_tool:             { enabled: true }

    ``hermes_rl_tool`` is a finer-grained switch than ``rl_optimization``:
    it only removes the ``matlab_rl_optimize`` tool from Hermes while
    keeping the direct Step-3 RL loop intact.
    """
    rl_optimization: bool = True
    optimization_workflow: bool = True
    parameter_experience_reuse: bool = True
    reflection_agent: bool = True
    hermes_rl_tool: bool = True
    memory_search: bool = True
    guidance_compat_verification: bool = True
    syntax_check: bool = True

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AblationConfig":
        def _flag(sub: Any, default: bool = True) -> bool:
            if isinstance(sub, dict):
                return bool(sub.get("enabled", default))
            if isinstance(sub, bool):
                return sub
            return default

        return cls(
            rl_optimization=_flag(data.get("rl_optimization", {})),
            optimization_workflow=_flag(data.get("optimization_workflow", {})),
            parameter_experience_reuse=_flag(data.get("parameter_experience_reuse", {})),
            reflection_agent=_flag(data.get("reflection_agent", {})),
            hermes_rl_tool=_flag(data.get("hermes_rl_tool", {})),
            memory_search=_flag(data.get("memory_search", {})),
            guidance_compat_verification=_flag(data.get("guidance_compat_verification", {})),
            syntax_check=_flag(data.get("syntax_check", {})),
        )


@dataclass
class ParameterExperienceConfig:
    """ParameterExperience Memory Configuration"""
    enabled: bool = True
    max_short_term: int = 100
    max_long_term: int = 1000
    similarity_threshold: float = 0.7
    decay_factor: float = 0.95

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ParameterExperienceConfig":
        return cls(
            enabled=data.get("enabled", True),
            max_short_term=data.get("max_short_term", 100),
            max_long_term=data.get("max_long_term", 1000),
            similarity_threshold=data.get("similarity_threshold", 0.7),
            decay_factor=data.get("decay_factor", 0.95),
        )


@dataclass
class GAConfig:
    """Genetic Algorithm Configuration"""
    population_size: int = 50
    crossover_rate: float = 0.8
    mutation_rate: float = 0.1
    max_generations: int = 100
    tournament_size: int = 3
    elite_size: int = 2

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GAConfig":
        return cls(
            population_size=data.get("population_size", 50),
            crossover_rate=data.get("crossover_rate", 0.8),
            mutation_rate=data.get("mutation_rate", 0.1),
            max_generations=data.get("max_generations", 100),
            tournament_size=data.get("tournament_size", 3),
            elite_size=data.get("elite_size", 2),
        )


@dataclass
class OptimizerRLConfig:
    """Optimizer Reinforcement Learning Configuration"""
    algorithm: str = "q_learning"
    learning_rate: float = 0.01
    discount_factor: float = 0.95
    epsilon: float = 0.1
    epsilon_decay: float = 0.995
    epsilon_min: float = 0.01
    batch_size: int = 32
    memory_size: int = 10000

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OptimizerRLConfig":
        return cls(
            algorithm=data.get("algorithm", "q_learning"),
            learning_rate=data.get("learning_rate", 0.01),
            discount_factor=data.get("discount_factor", 0.95),
            epsilon=data.get("epsilon", 0.1),
            epsilon_decay=data.get("epsilon_decay", 0.995),
            epsilon_min=data.get("epsilon_min", 0.01),
            batch_size=data.get("batch_size", 32),
            memory_size=data.get("memory_size", 10000),
        )


@dataclass
class OptimizerConfig:
    """Optimizer Configuration"""
    enabled: bool = True
    type: str = "ga"  # "ga" | "rl"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OptimizerConfig":
        return cls(
            enabled=data.get("enabled", True),
            type=data.get("type", "ga"),
        )


@dataclass
class LoggingConfig:
    verbosity: str = "normal"
    rl_episode_interval: int = 10

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LoggingConfig":
        return cls(
            verbosity=str(data.get("verbosity", "normal")).strip().lower(),
            rl_episode_interval=int(data.get("rl_episode_interval", 10)),
        )


@dataclass
class WorkflowConfig:
    """Workflow Configuration"""
    max_iterations: int = 4
    hermes_execution: bool = True
    hermes_tool_timeout: Optional[float] = 0.0
    run_wall_clock_budget_sec: int = 1800
    layer3_reserve_sec: float = 90.0
    tune_only_skip_hermes: bool = True
    modify_law_t4_fast_path: bool = True
    skip_planner_on_tune_only: bool = True
    force_tune_fast_path_iter2: bool = True
    require_all_pass_exit: bool = True
    bw_finish_budget_sec: int = 900

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorkflowConfig":
        _raw = data.get("hermes_tool_timeout", None)
        _timeout = float(_raw) if _raw is not None else 0.0
        return cls(
            max_iterations=data.get("max_iterations", 4),
            hermes_execution=data.get("hermes_execution", True),
            hermes_tool_timeout=_timeout,
            run_wall_clock_budget_sec=int(data.get("run_wall_clock_budget_sec", 1800)),
            layer3_reserve_sec=float(data.get("layer3_reserve_sec", 90.0)),
            tune_only_skip_hermes=bool(data.get("tune_only_skip_hermes", True)),
            modify_law_t4_fast_path=bool(data.get("modify_law_t4_fast_path", True)),
            skip_planner_on_tune_only=bool(data.get("skip_planner_on_tune_only", True)),
            force_tune_fast_path_iter2=bool(data.get("force_tune_fast_path_iter2", True)),
            require_all_pass_exit=bool(data.get("require_all_pass_exit", True)),
            bw_finish_budget_sec=int(data.get("bw_finish_budget_sec", 900)),
        )


@dataclass
class RLRewardConfig:
    """Reward function weights for MatlabRLOptimizer.
    Metrics: hit_rate (maximize), SEP (minimize), PeakNy (soft constraint),
    PM (range reward), BW (range reward).
    """
    hit_rate: float = 3.0
    # SEP (miss distance)
    sep_low_bonus: float = 2.0
    sep_low_threshold: float = 2.0
    sep_mid_weight: float = 1.0
    sep_mid_threshold: float = 5.0
    # PeakNy
    peak_ny_max: float = PEAK_NY_MAX
    peak_ny_mean_max: float = PEAK_NY_MAX
    peak_ny_penalty: float = 1.0
    peak_ny_progressive_low: float = 15.0
    hard_truncation_peak_g: float = 35.0
    baseline_reset_peak_g: float = 40.0
    # Phase A: SEP above this threshold → no PM/BW reward terms
    sep_survival_threshold: float = 500.0
    # PM range
    pm_bonus: float = 0.5
    pm_min: float = PM_MIN
    pm_max: float = PM_MAX
    pm_penalty: float = 0.3
    # BW range
    bw_bonus: float = 0.2
    bw_min: float = BW_MIN
    bw_max: float = BW_MAX
    bw_penalty: float = 0.1

    def to_dict(self) -> Dict[str, float]:
        """Convert to dict for MatlabRLOptimizer reward_weights."""
        import dataclasses
        return {k: float(v) for k, v in dataclasses.asdict(self).items()}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RLRewardConfig":
        return cls(
            hit_rate=float(data.get("hit_rate", 3.0)),
            sep_low_bonus=float(data.get("sep_low_bonus", 2.0)),
            sep_low_threshold=float(data.get("sep_low_threshold", 2.0)),
            sep_mid_weight=float(data.get("sep_mid_weight", 1.0)),
            sep_mid_threshold=float(data.get("sep_mid_threshold", 5.0)),
            peak_ny_max=float(data.get("peak_ny_max", PEAK_NY_MAX)),
            peak_ny_mean_max=float(
                data.get("peak_ny_mean_max", data.get("peak_ny_max", PEAK_NY_MAX))
            ),
            peak_ny_penalty=float(data.get("peak_ny_penalty", 1.0)),
            peak_ny_progressive_low=float(data.get("peak_ny_progressive_low", 15.0)),
            hard_truncation_peak_g=float(data.get("hard_truncation_peak_g", 35.0)),
            baseline_reset_peak_g=float(data.get("baseline_reset_peak_g", 40.0)),
            sep_survival_threshold=float(data.get("sep_survival_threshold", 500.0)),
            pm_bonus=float(data.get("pm_bonus", 0.5)),
            pm_min=float(data.get("pm_min", PM_MIN)),
            pm_max=float(data.get("pm_max", PM_MAX)),
            pm_penalty=float(data.get("pm_penalty", 0.3)),
            bw_bonus=float(data.get("bw_bonus", 0.2)),
            bw_min=float(data.get("bw_min", BW_MIN)),
            bw_max=float(data.get("bw_max", BW_MAX)),
            bw_penalty=float(data.get("bw_penalty", 0.1)),
        )


@dataclass
class RLExplorationConfig:
    pitch_log_std: float = -0.7
    other_log_std: float = -1.6
    warm_start_std_scale: float = 0.3
    baseline_reward_threshold: float = 0.5
    diverge_penalty_episodes: int = 3
    diverge_log_std_delta: float = -0.5
    diverge_reward_threshold: float = -9.5
    adaptive_enabled: bool = True
    adaptive_steepness_threshold: float = 2.0
    adaptive_log_std_min_scale: float = 0.35
    adaptive_log_std_max_scale: float = 1.0
    adaptive_ema_alpha: float = 0.3

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RLExplorationConfig":
        return cls(
            pitch_log_std=float(data.get("pitch_log_std", -0.7)),
            other_log_std=float(data.get("other_log_std", -1.6)),
            warm_start_std_scale=float(data.get("warm_start_std_scale", 0.3)),
            baseline_reward_threshold=float(data.get("baseline_reward_threshold", 0.5)),
            diverge_penalty_episodes=int(data.get("diverge_penalty_episodes", 3)),
            diverge_log_std_delta=float(data.get("diverge_log_std_delta", -0.5)),
            diverge_reward_threshold=float(data.get("diverge_reward_threshold", -9.5)),
            adaptive_enabled=bool(data.get("adaptive_enabled", True)),
            adaptive_steepness_threshold=float(data.get("adaptive_steepness_threshold", 2.0)),
            adaptive_log_std_min_scale=float(data.get("adaptive_log_std_min_scale", 0.35)),
            adaptive_log_std_max_scale=float(data.get("adaptive_log_std_max_scale", 1.0)),
            adaptive_ema_alpha=float(data.get("adaptive_ema_alpha", 0.3)),
        )


@dataclass
class RLEarlyStopConfig:
    enabled: bool = True
    min_episodes: int = 25
    patience: int = 15
    reward_plateau_delta: float = 0.05
    reflect_every: int = 5
    use_rule_first: bool = True
    use_multi_criteria_best: bool = True
    peak_only_patience: int = 25

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RLEarlyStopConfig":
        return cls(
            enabled=bool(data.get("enabled", True)),
            min_episodes=int(data.get("min_episodes", 25)),
            patience=int(data.get("patience", 15)),
            reward_plateau_delta=float(data.get("reward_plateau_delta", 0.05)),
            reflect_every=int(data.get("reflect_every", 5)),
            use_rule_first=bool(data.get("use_rule_first", True)),
            use_multi_criteria_best=bool(data.get("use_multi_criteria_best", True)),
            peak_only_patience=int(data.get("peak_only_patience", 25)),
        )


@dataclass
class RLConstraintLocalSearchConfig:
    enabled: bool = True
    max_steps: int = 12
    nmc_per_eval: int = 25
    nmc_fast: int = 5
    nmc_validate: int = 25
    step_scale: float = 0.08
    seed: Optional[int] = 42
    no_regression_gate: bool = True
    directed_perturb: bool = True
    borderline_select: bool = False
    borderline_hit_min_pct: float = HIT_RATE_MIN
    borderline_sep_max_m: float = SEP_MAX
    prefer_ppo_best_peak: bool = True
    lock_margin_sep_m: float = 0.4
    lock_margin_peak_ny_g: float = 0.35
    lock_margin_hit_pct: float = 1.0
    single_metric_finish: bool = True
    bw_finish_max_steps: int = 8
    bw_finish_step_scale: float = 0.14
    partial_bw_band_gate: bool = True

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RLConstraintLocalSearchConfig":
        seed_raw = data.get("seed", 42)
        seed = int(seed_raw) if seed_raw is not None and str(seed_raw).strip() != "" else None
        return cls(
            enabled=bool(data.get("enabled", True)),
            max_steps=int(data.get("max_steps", 12)),
            nmc_per_eval=int(data.get("nmc_per_eval", 25)),
            nmc_fast=int(data.get("nmc_fast", data.get("nmc_per_eval", 5))),
            nmc_validate=int(data.get("nmc_validate", data.get("nmc_per_eval", 25))),
            step_scale=float(data.get("step_scale", 0.08)),
            seed=seed,
            no_regression_gate=bool(data.get("no_regression_gate", True)),
            directed_perturb=bool(data.get("directed_perturb", True)),
            borderline_select=bool(data.get("borderline_select", False)),
            borderline_hit_min_pct=float(data.get("borderline_hit_min_pct", HIT_RATE_MIN)),
            borderline_sep_max_m=float(data.get("borderline_sep_max_m", SEP_MAX)),
            prefer_ppo_best_peak=bool(data.get("prefer_ppo_best_peak", True)),
            lock_margin_sep_m=float(data.get("lock_margin_sep_m", 0.4)),
            lock_margin_peak_ny_g=float(data.get("lock_margin_peak_ny_g", 0.35)),
            lock_margin_hit_pct=float(data.get("lock_margin_hit_pct", 1.0)),
            single_metric_finish=bool(data.get("single_metric_finish", True)),
            bw_finish_max_steps=int(data.get("bw_finish_max_steps", 8)),
            bw_finish_step_scale=float(data.get("bw_finish_step_scale", 0.14)),
            partial_bw_band_gate=bool(data.get("partial_bw_band_gate", True)),
        )


@dataclass
class FREPresetConfig:
    initial_sep: float = 100.0
    initial_hit_rate: float = 50.0
    target_sep: float = 7.0
    target_hit_rate: float = HIT_RATE_MIN
    phase_episodes: int = 20
    num_phases: int = 3

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FREPresetConfig":
        return cls(
            initial_sep=float(data.get("initial_sep", 100.0)),
            initial_hit_rate=float(data.get("initial_hit_rate", 50.0)),
            target_sep=float(data.get("target_sep", 7.0)),
            target_hit_rate=float(data.get("target_hit_rate", HIT_RATE_MIN)),
            phase_episodes=int(data.get("phase_episodes", 20)),
            num_phases=int(data.get("num_phases", 3)),
        )


@dataclass
class FREConfig:
    default: FREPresetConfig = field(default_factory=FREPresetConfig)
    t4: Optional[FREPresetConfig] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FREConfig":
        t4_raw = data.get("t4")
        return cls(
            default=FREPresetConfig.from_dict(data.get("default", {})),
            t4=FREPresetConfig.from_dict(t4_raw) if t4_raw else None,
        )

    def preset_for_sub_idx(self, sub_idx: int) -> FREPresetConfig:
        if sub_idx == 4 and self.t4 is not None:
            return self.t4
        return self.default


@dataclass
class RLCheckpointConfig:
    enabled: bool = True
    dir: str = "./parameter_experience_base/checkpoints/ppo"
    save_every_episodes: int = 10
    keep_last_n: int = 5
    rolling_window: int = 20
    stats_log_every: int = 1
    jsonl_log_dir: str = "logs"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RLCheckpointConfig":
        return cls(
            enabled=bool(data.get("enabled", True)),
            dir=str(data.get("dir", "./parameter_experience_base/checkpoints/ppo")),
            save_every_episodes=int(data.get("save_every_episodes", 10)),
            keep_last_n=int(data.get("keep_last_n", 5)),
            rolling_window=int(data.get("rolling_window", 20)),
            stats_log_every=int(data.get("stats_log_every", 1)),
            jsonl_log_dir=str(data.get("jsonl_log_dir", "logs")),
        )


@dataclass
class MatlabRLOptimizerConfig:
    """Contextual-bandit / PPO autopilot parameter optimizer"""
    layer3_strategy: str = "auto"  # 'expert' | 'ppo' | 'auto'
    # Section 4.2.1 pluggable optimizer backend:
    #   hybrid_ppo    — GSMultiAgent default (expert + PPO + CLS)
    #   direct_ppo    — PPO only, no expert LLM tuning
    #   cma_es        — CMA-ES, λ=4+floor(3·ln(d)), σ₀=0.25
    #   random_search — Uniform random, performance lower bound
    optimizer_backend: str = "hybrid_ppo"
    max_episodes: int = 20
    expert_nmc: int = 15
    nmc_per_eval: int = 10
    nmc_refine: int = 25
    episodes_per_update: int = 64
    hidden_dim: int = 64
    lr_actor: float = 3e-4
    lr_critic: float = 1e-3
    gamma: float = 0.0
    clip_ratio: float = 0.2
    entropy_coef: float = 0.05
    peak_n_max: float = 20.0
    peak_ny_mean_max: float = PEAK_NY_MAX
    peak_n_penalty: float = 1.0
    reward_weights: RLRewardConfig = field(default_factory=RLRewardConfig)
    exploration: RLExplorationConfig = field(default_factory=RLExplorationConfig)
    early_stop: RLEarlyStopConfig = field(default_factory=RLEarlyStopConfig)
    constraint_local_search: RLConstraintLocalSearchConfig = field(
        default_factory=RLConstraintLocalSearchConfig
    )
    checkpoint: RLCheckpointConfig = field(default_factory=RLCheckpointConfig)
    # AMRO deprecated (ignored at runtime; kept for config backward compatibility)
    amro_enabled: bool = False
    amro_gradient_threshold: float = 0.5
    amro_forbidden_zone_radius: float = 0.15
    amro_history_window: int = 10
    allow_internal_fallback: bool = False

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MatlabRLOptimizerConfig":
        return cls(
            layer3_strategy=str(data.get("layer3_strategy", "auto")),
            optimizer_backend=str(data.get("optimizer_backend", "hybrid_ppo")),
            max_episodes=data.get("max_episodes", 20),
            expert_nmc=int(data.get("expert_nmc", 15)),
            nmc_per_eval=data.get("nmc_per_eval", 10),
            nmc_refine=int(data.get("nmc_refine", 25)),
            episodes_per_update=data.get("episodes_per_update", 64),
            hidden_dim=data.get("hidden_dim", 64),
            lr_actor=float(data.get("lr_actor", 3e-4)),
            lr_critic=float(data.get("lr_critic", 1e-3)),
            gamma=float(data.get("gamma", 0.0)),
            clip_ratio=float(data.get("clip_ratio", 0.2)),
            entropy_coef=float(data.get("entropy_coef", 0.05)),
            peak_n_max=float(data.get("peak_n_max", data.get("peak_ny_mean_max", PEAK_NY_MAX))),
            peak_ny_mean_max=float(
                data.get("peak_ny_mean_max", data.get("peak_n_max", PEAK_NY_MAX))
            ),
            peak_n_penalty=float(data.get("peak_n_penalty", 1.0)),
            reward_weights=RLRewardConfig.from_dict(data.get("reward_weights", {})),
            exploration=RLExplorationConfig.from_dict(data.get("exploration", {})),
            early_stop=RLEarlyStopConfig.from_dict(data.get("early_stop", {})),
            constraint_local_search=RLConstraintLocalSearchConfig.from_dict(
                data.get("constraint_local_search", {})
            ),
            checkpoint=RLCheckpointConfig.from_dict(data.get("checkpoint", {})),
            amro_enabled=data.get("amro_enabled", False),
            amro_gradient_threshold=float(data.get("amro_gradient_threshold", 0.5)),
            amro_forbidden_zone_radius=float(data.get("amro_forbidden_zone_radius", 0.15)),
            amro_history_window=int(data.get("amro_history_window", 10)),
            allow_internal_fallback=bool(data.get("allow_internal_fallback", False)),
        )


@dataclass
class ModelOutputConfig:
    """Model file output paths"""
    generated_dir: str = "./guidance_output/models"
    success_dir: str = "./parameter_experience_base/models"
    matlab_scripts_dir: str = "./matlab_scripts"
    experience_base_dir: str = "./parameter_experience_base"
    sysml_output_dir: str = "./guidance_output/sysml"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelOutputConfig":
        return cls(
            generated_dir=data.get("generated_dir", "./guidance_output/models"),
            success_dir=data.get("success_dir", "./parameter_experience_base/models"),
            matlab_scripts_dir=data.get("matlab_scripts_dir", "./matlab_scripts"),
            experience_base_dir=data.get("experience_base_dir", "./parameter_experience_base"),
            sysml_output_dir=data.get("sysml_output_dir", "./guidance_output/sysml"),
        )


@dataclass
class MCValidationConfig:
    """Monte Carlo validation across multiple MATLAB RNG seeds."""
    enabled: bool = False
    seeds: List[int] = field(default_factory=lambda: [42, 2025, 3407])
    require_all_pass: bool = True
    mode: str = "final"
    final_nmc: int = 0

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MCValidationConfig":
        raw_seeds = data.get("seeds", [42, 2025, 3407])
        if isinstance(raw_seeds, str):
            seed_items = [s.strip() for s in raw_seeds.split(",")]
        elif isinstance(raw_seeds, (list, tuple)):
            seed_items = list(raw_seeds)
        else:
            seed_items = [raw_seeds]

        seeds: List[int] = []
        for item in seed_items:
            if item is None or str(item).strip() == "":
                continue
            try:
                seeds.append(int(item))
            except (TypeError, ValueError):
                continue
        if not seeds:
            seeds = [42, 2025, 3407]

        mode = str(data.get("mode", data.get("apply_to", "final"))).strip().lower()
        mode_aliases = {
            "final_only": "final",
            "final-validation": "final",
            "final_validation": "final",
            "all_simulations": "all",
            "always": "all",
        }
        mode = mode_aliases.get(mode, mode)
        if mode not in ("final", "all", "off"):
            mode = "final"

        return cls(
            enabled=bool(data.get("enabled", False)),
            seeds=seeds,
            require_all_pass=bool(data.get("require_all_pass", True)),
            mode=mode,
            final_nmc=int(data.get("final_nmc", 0) or 0),
        )


@dataclass
class SimulationConfig:
    engine: str = "auto"
    octave_path: str = "octave"
    matlab_path: str = "matlab"
    mc_data_path: str = ""
    # Paper §4.1 external-MC protocol: evaluate candidates on multiple .mat files
    # under mc_root/{scenario}/seed_{id}_{split}.mat with AND feasibility.
    mc_root: str = "./data/mc"
    mc_scenario: str = ""
    mc_seeds: List[int] = field(default_factory=list)
    prefer_kb_template: bool = False
    seed_policy: str = "iterative"
    layer2_script_gate: bool = True
    layer2_smoke_nmc: int = 5
    modify_gate_nmc: int = 10
    physics_bounds_enabled: bool = True
    physics_bounds_ny_limit_g: float = 20.0
    physics_bounds_timeout_sec: float = 120.0
    physics_bounds_skip_deterministic_t4_apn: bool = True
    matlab_timeout_sec: int = 900
    matlab_timeout_per_mc_sec: int = 22
    matlab_timeout_floor_sec: int = 360
    # Parallel external-MC: overlap independent MATLAB batches (protocol unchanged).
    # matlab_max_parallel caps total concurrent matlab -batch processes.
    # Keep this modest: each MATLAB can still use significant RAM.
    matlab_max_parallel: int = 2
    mc_and_workers: int = 1
    mc_candidate_workers: int = 2
    matlab_job_pool: bool = True
    matlab_pool_workers: int = 2  # 0 → use matlab_max_parallel
    # GSM suite: parallel independent (config, master_seed) groups
    gsm_group_workers: int = 1
    mc_tiered_eval: bool = True
    mc_search_n_runs: int = 10
    mc_final_n_runs: int = 100
    mc_eval_cache: bool = True
    mc_and_early_abort: bool = True
    # True: only first AND seed runs in MATLAB; remaining seeds bootstrap-filled.
    mc_and_bootstrap_fill: bool = True
    # Export real seed1 MC payloads for Ubuntu offline bootstrap/contract fill.
    bootstrap_export: bool = True
    bootstrap_export_dir: str = "./guidance_output/bootstrap_export"
    # False by default: nested parfor + outer pool caused OOM / R6025.
    mc_use_parfor: bool = False
    matlab_max_comp_threads: int = 2
    pipeline_warmup: bool = True
    mc_validation: MCValidationConfig = field(default_factory=MCValidationConfig)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SimulationConfig":
        raw_seeds = data.get("mc_seeds", []) or []
        if isinstance(raw_seeds, str):
            seed_items = [s.strip() for s in raw_seeds.replace(";", ",").split(",")]
        elif isinstance(raw_seeds, (list, tuple)):
            seed_items = list(raw_seeds)
        else:
            seed_items = [raw_seeds]
        mc_seeds: List[int] = []
        for item in seed_items:
            if item is None or str(item).strip() == "":
                continue
            try:
                mc_seeds.append(int(item))
            except (TypeError, ValueError):
                continue
        return cls(
            engine=data.get("engine", "auto"),
            octave_path=data.get("octave_path", "octave"),
            matlab_path=data.get("matlab_path", "matlab"),
            mc_data_path=str(
                data.get("mc_data_path", get_env("GSM_MC_DATA_PATH", "")) or ""
            ),
            mc_root=str(data.get("mc_root", "./data/mc") or "./data/mc"),
            mc_scenario=str(data.get("mc_scenario", "") or ""),
            mc_seeds=mc_seeds,
            prefer_kb_template=bool(data.get("prefer_kb_template", False)),
            seed_policy=str(data.get("seed_policy", "iterative")).strip().lower(),
            layer2_script_gate=bool(data.get("layer2_script_gate", True)),
            layer2_smoke_nmc=int(data.get("layer2_smoke_nmc", 5)),
            modify_gate_nmc=int(data.get("modify_gate_nmc", 10)),
            physics_bounds_enabled=bool(data.get("physics_bounds_enabled", True)),
            physics_bounds_ny_limit_g=float(data.get("physics_bounds_ny_limit_g", 20.0)),
            physics_bounds_timeout_sec=float(data.get("physics_bounds_timeout_sec", 120.0)),
            physics_bounds_skip_deterministic_t4_apn=bool(
                data.get("physics_bounds_skip_deterministic_t4_apn", True)
            ),
            matlab_timeout_sec=int(data.get("matlab_timeout_sec", 900)),
            matlab_timeout_per_mc_sec=int(data.get("matlab_timeout_per_mc_sec", 22)),
            matlab_timeout_floor_sec=int(data.get("matlab_timeout_floor_sec", 360)),
            matlab_max_parallel=int(data.get("matlab_max_parallel", 2) or 2),
            mc_and_workers=int(data.get("mc_and_workers", 1) or 1),
            mc_candidate_workers=int(data.get("mc_candidate_workers", 2) or 2),
            matlab_job_pool=bool(data.get("matlab_job_pool", True)),
            matlab_pool_workers=int(data.get("matlab_pool_workers", 2) or 2),
            gsm_group_workers=int(data.get("gsm_group_workers", 1) or 1),
            mc_tiered_eval=bool(data.get("mc_tiered_eval", True)),
            mc_search_n_runs=int(data.get("mc_search_n_runs", 10) or 10),
            mc_final_n_runs=int(data.get("mc_final_n_runs", 100) or 100),
            mc_eval_cache=bool(data.get("mc_eval_cache", True)),
            mc_and_early_abort=bool(data.get("mc_and_early_abort", True)),
            mc_and_bootstrap_fill=bool(data.get("mc_and_bootstrap_fill", True)),
            bootstrap_export=bool(data.get("bootstrap_export", True)),
            bootstrap_export_dir=str(
                data.get("bootstrap_export_dir", "./guidance_output/bootstrap_export")
                or "./guidance_output/bootstrap_export"
            ),
            mc_use_parfor=bool(data.get("mc_use_parfor", False)),
            matlab_max_comp_threads=int(data.get("matlab_max_comp_threads", 2) or 2),
            pipeline_warmup=bool(data.get("pipeline_warmup", True)),
            mc_validation=MCValidationConfig.from_dict(data.get("mc_validation", {})),
        )


@dataclass
class T4LowRiskConfig:
    enabled: bool = True
    deterministic_apn: bool = True
    deterministic_apn_kb_only: bool = True
    ny_limit_g: float = 20.0
    max_w: float = 45.0
    min_w1: float = 20.0
    max_n_pn: float = 4.5
    modify_fail_peak_g: float = 50.0
    modify_gate_peak_g: float = 30.0
    modify_fail_hit_rate_min: float = 85.0
    gate_near_miss_tune_enabled: bool = True
    post_modify_first_tune_strategy: str = "expert"
    post_modify_max_episodes: int = 50
    post_modify_ppo_max_episodes: int = 40
    post_modify_expert_rounds: int = 3
    post_modify_ppo_peak_max_g: float = 40.0
    post_modify_ppo_hit_min_pct: float = 80.0
    skip_ppo_expert_sep_max_m: float = 15.0
    skip_ppo_expert_hit_min_pct: float = 90.0
    force_skip_ppo_after_expert: bool = True
    peak_near_miss_g: float = 8.0
    near_miss_ppo_max_episodes: int = 60
    first_iter_mode: str = "modify_law"
    modify_gate_nmc: int = 10
    pe_store_peak_max_g: float = 25.0
    pe_store_hit_min_pct: float = 92.0
    borderline_nmc_revalidate: int = 50
    conservative_autopilot: Dict[str, float] = field(default_factory=dict)
    expert_temperature: float = 0.0
    expert_rule_first_r1: bool = True
    expert_r1_w1_cap: float = 40.0
    pre_cls_bw_pm_fixup: bool = True
    pre_cls_w1_drop: float = 6.0

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "T4LowRiskConfig":
        cap = data.get("conservative_autopilot") or {}
        modify_gate_peak = float(data.get("modify_gate_peak_g", 30.0))
        modify_fail_peak = float(data.get("modify_fail_peak_g", 50.0))
        modify_gate_hit = data.get(
            "modify_gate_hit_min_pct", data.get("modify_fail_hit_rate_min", 85.0)
        )
        return cls(
            enabled=bool(data.get("enabled", True)),
            deterministic_apn=bool(data.get("deterministic_apn", True)),
            deterministic_apn_kb_only=bool(data.get("deterministic_apn_kb_only", True)),
            ny_limit_g=float(data.get("ny_limit_g", 20.0)),
            max_w=float(data.get("max_w", 45.0)),
            min_w1=float(data.get("min_w1", 20.0)),
            max_n_pn=float(data.get("max_n_pn", 4.5)),
            modify_fail_peak_g=modify_fail_peak,
            modify_gate_peak_g=modify_gate_peak,
            modify_fail_hit_rate_min=float(modify_gate_hit),
            gate_near_miss_tune_enabled=bool(
                data.get("gate_near_miss_tune_enabled", True)
            ),
            post_modify_first_tune_strategy=str(
                data.get("post_modify_first_tune_strategy", "expert")
            ),
            post_modify_max_episodes=int(data.get("post_modify_max_episodes", 50)),
            post_modify_ppo_max_episodes=int(data.get("post_modify_ppo_max_episodes", 40)),
            post_modify_expert_rounds=int(data.get("post_modify_expert_rounds", 3)),
            post_modify_ppo_peak_max_g=float(data.get("post_modify_ppo_peak_max_g", 40.0)),
            post_modify_ppo_hit_min_pct=float(data.get("post_modify_ppo_hit_min_pct", 80.0)),
            skip_ppo_expert_sep_max_m=float(data.get("skip_ppo_expert_sep_max_m", 15.0)),
            skip_ppo_expert_hit_min_pct=float(data.get("skip_ppo_expert_hit_min_pct", 90.0)),
            force_skip_ppo_after_expert=bool(data.get("force_skip_ppo_after_expert", True)),
            peak_near_miss_g=float(data.get("peak_near_miss_g", 8.0)),
            near_miss_ppo_max_episodes=int(data.get("near_miss_ppo_max_episodes", 60)),
            first_iter_mode=str(data.get("first_iter_mode", "modify_law")),
            modify_gate_nmc=int(data.get("modify_gate_nmc", 10)),
            pe_store_peak_max_g=float(data.get("pe_store_peak_max_g", 25.0)),
            pe_store_hit_min_pct=float(data.get("pe_store_hit_min_pct", 92.0)),
            borderline_nmc_revalidate=int(data.get("borderline_nmc_revalidate", 50)),
            conservative_autopilot=dict(cap) if isinstance(cap, dict) else {},
            expert_temperature=float(data.get("expert_temperature", 0.0)),
            expert_rule_first_r1=bool(data.get("expert_rule_first_r1", True)),
            expert_r1_w1_cap=float(data.get("expert_r1_w1_cap", 40.0)),
            pre_cls_bw_pm_fixup=bool(data.get("pre_cls_bw_pm_fixup", True)),
            pre_cls_w1_drop=float(data.get("pre_cls_w1_drop", 6.0)),
        )


@dataclass
class AppConfig:
    """Application Configuration"""
    ablation: AblationConfig = field(default_factory=AblationConfig)
    parameter_experience: ParameterExperienceConfig = field(default_factory=ParameterExperienceConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    ga: GAConfig = field(default_factory=GAConfig)
    rl: OptimizerRLConfig = field(default_factory=OptimizerRLConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    rag: RAGConfig = field(default_factory=RAGConfig)
    workflow: WorkflowConfig = field(default_factory=WorkflowConfig)
    matlab_rl_optimizer: MatlabRLOptimizerConfig = field(default_factory=MatlabRLOptimizerConfig)
    fre: FREConfig = field(default_factory=FREConfig)
    model_output: ModelOutputConfig = field(default_factory=ModelOutputConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    t4_low_risk: T4LowRiskConfig = field(default_factory=T4LowRiskConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AppConfig":
        return cls(
            ablation=AblationConfig.from_dict(data.get("ablation", {})),
            parameter_experience=ParameterExperienceConfig.from_dict(data.get("parameter_experience", {})),
            optimizer=OptimizerConfig.from_dict(data.get("optimizer", {})),
            ga=GAConfig.from_dict(data.get("ga", {})),
            rl=OptimizerRLConfig.from_dict(data.get("rl", {})),
            llm=LLMConfig.from_dict(data.get("llm", {})),
            rag=RAGConfig.from_dict(data.get("rag", {})),
            workflow=WorkflowConfig.from_dict(data.get("workflow", {})),
            matlab_rl_optimizer=MatlabRLOptimizerConfig.from_dict(data.get("matlab_rl_optimizer", {})),
            fre=FREConfig.from_dict(data.get("fre", {})),
            model_output=ModelOutputConfig.from_dict(data.get("model_output", {})),
            simulation=SimulationConfig.from_dict(data.get("simulation", {})),
            t4_low_risk=T4LowRiskConfig.from_dict(data.get("t4_low_risk", {})),
            logging=LoggingConfig.from_dict(data.get("logging", {})),
        )


def load_config(config_path: str = None) -> AppConfig:
    """Load configuration from YAML file"""
    if config_path is None:
        parent_dir_config = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.yaml")
        if os.path.exists(parent_dir_config):
            config_path = parent_dir_config
        else:
            config_path = os.path.join(os.path.dirname(__file__), "config.yaml")

    if not os.path.exists(config_path):
        return AppConfig()

    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    return AppConfig.from_dict(data or {})


# Global config instance
_config: Optional[AppConfig] = None


def get_config() -> AppConfig:
    """Get global config instance (lazy loading)"""
    global _config
    if _config is None:
        _config = load_config()
    return _config


def reload_config(config_path: str = None) -> AppConfig:
    """Reload configuration from file"""
    global _config
    _config = load_config(config_path)
    return _config
