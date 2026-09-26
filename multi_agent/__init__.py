#!/usr/bin/env python3
"""
Multi-Agent Architecture
基于 Hermes Agent + RAG + ParameterExperience + RL + Simulation 的多智能体系统

核心设计理念：
- 使用 Hermes Agent 作为 LLM 编排引擎
- 我们的模块 (RAG/ParameterExperience/RL/Simulation) 作为 Hermes Tools 扩展
- 不重复实现 Agent 框架，专注垂直领域能力
"""

__version__ = "0.3.0"

import logging
import warnings
warnings.filterwarnings("ignore", message="Could not import tool module")
logging.getLogger("model_tools").setLevel(logging.ERROR)

# Memory
from .memory.parameter_experience import ParameterExperience, MemoryType
from .memory.rag_knowledge_base import RAGKnowledgeBase, EmbeddingConfig

# RL
from .rl.matlab_rl_optimizer import (
    MatlabRLOptimizer,
    parse_mission_conditions,
    extract_matlab_params,
    extract_autopilot_params,
    extract_physical_params,
    AUTOPILOT_PARAM_SPECS,
    CATEGORY_DEFS,
)

# Simulation
from .simulation import (
    GuidanceSimulator,
    GuidanceParameters,
    SimulationResult,
    SysMLModelGenerator,
    MATLABScriptGenerator,
    SimulationExecutor,
)

# Integration
from .integration import (
    HermesIntegration,
    HERMES_AVAILABLE,
    IntelligentTaskPlanner,
    TaskPlan,
    ExecutionStrategy,
)

# Configuration (unified from config_loader.py)
from .config_loader import (
    load_config,
    get_config,
    reload_config,
    AppConfig,
    LLMConfig,
    ParameterExperienceConfig,
    RAGConfig,
    GAConfig,
    OptimizerConfig,
    OptimizerRLConfig,
)

# Tools
from .tools import (
    RAGRetrievalTool,
    RAGIndexTool,
    ParameterExperienceSearchTool,
    ParameterExperienceStoreTool,
    GenerateMATLABTool,
    RunSimulationTool,
)

__all__ = [
    "__version__",
    # Memory
    "ParameterExperience",
    "MemoryType",
    "RAGKnowledgeBase",
    "EmbeddingConfig",
    # Simulation
    "GuidanceSimulator",
    "GuidanceParameters",
    "SimulationResult",
    "SysMLModelGenerator",
    "MATLABScriptGenerator",
    "SimulationExecutor",
    # RL
    "MatlabRLOptimizer",
    "parse_mission_conditions",
    "extract_matlab_params",
    "extract_autopilot_params",
    "extract_physical_params",
    "AUTOPILOT_PARAM_SPECS",
    "CATEGORY_DEFS",
    # Integration
    "HermesIntegration",
    "HERMES_AVAILABLE",
    "IntelligentTaskPlanner",
    "TaskPlan",
    "ExecutionStrategy",
    # Config
    "load_config",
    "get_config",
    "reload_config",
    "AppConfig",
    "LLMConfig",
    "ParameterExperienceConfig",
    "RAGConfig",
    "GAConfig",
    "OptimizerConfig",
    "OptimizerRLConfig",
    # Tools
    "RAGRetrievalTool",
    "RAGIndexTool",
    "ParameterExperienceSearchTool",
    "ParameterExperienceStoreTool",
    "GenerateMATLABTool",
    "RunSimulationTool",
]
