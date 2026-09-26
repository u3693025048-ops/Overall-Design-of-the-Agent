#!/usr/bin/env python3
"""
Tools for Hermes Agent
"""

from .rag_tool import RAGRetrievalTool, RAGIndexTool, RAGExpandTool, RAGAgentTool
from .parameter_experience_tool import ParameterExperienceSearchTool, ParameterExperienceStoreTool, ParameterExperienceBestTool, ParameterExperienceStatsTool
from .simulation_tool import (
    GenerateMATLABTool,
    RunSimulationTool,
)
from .syntax_check_tool import SyntaxCheckMATLABTool
from .judge_requirements_tool import JudgeRequirementsTool
from .compatibility_tool import GuidanceLawCompatibilityTool
from .memory_tools import (
    AgentMemoryRememberTool,
    AgentMemoryRecallTool,
    AgentMemoryListTool,
    AgentMemoryForgetTool,
)
from .delegate_tool import DelegateSubAgentTool

__all__ = [
    # RAG Tools
    "RAGRetrievalTool",
    "RAGIndexTool",
    "RAGExpandTool",
    "RAGAgentTool",
    # ParameterExperience Tools
    "ParameterExperienceSearchTool",
    "ParameterExperienceStoreTool",
    "ParameterExperienceBestTool",
    "ParameterExperienceStatsTool",
    # Simulation Tools
    "GenerateMATLABTool",
    "RunSimulationTool",
    # Layer 2 New Tools
    "SyntaxCheckMATLABTool",
    "JudgeRequirementsTool",
    "GuidanceLawCompatibilityTool",
    # Persistent Memory Tools
    "AgentMemoryRememberTool",
    "AgentMemoryRecallTool",
    "AgentMemoryListTool",
    "AgentMemoryForgetTool",
]
