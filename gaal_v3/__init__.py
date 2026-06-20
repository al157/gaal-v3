"""
GAAL v3 — LangGraph-inspired StateGraph Arena Framework.

GAAL (Goal-oriented Autonomous Arena Loop) v3 is a complete redesign
using LangGraph-inspired architecture with nodes+edges, proper checkpointing,
cost optimization, self-healing, and recursive self-evolution.

Version: 3.0.0
"""
__version__ = "3.0.0"
__all__ = [
    "StateGraph", "Node", "Edge", "ConditionalBranch", "CompiledGraph",
    "CheckpointStore", "ModelRouter", "CostTracker",
    "GAALOrchestrator", "ReportGenerator",
]
