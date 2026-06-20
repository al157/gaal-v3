"""
Cost-aware model router for GAAL v3.

Implements tiered model assignment with complexity detection:
- flash: Basic models for simple tasks (lowest cost)
- pro: Advanced models for moderate tasks
- ultra: Expert models for complex tasks (highest cost)

Mode-based assignment:
- lite: team_a=pro, team_b=flash
- hard: team_a=pro, team_b=flash
- super: team_a=ultra, team_b=pro

Also tracks estimated costs for reporting and optimization.
"""
from __future__ import annotations
import logging
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)

# ── Model Tier Definitions ──────────────────────────────────────────

MODEL_TIERS: Dict[str, Dict[str, Any]] = {
    "flash": {
        "cost": 1,
        "capability": "basic",
        "description": "轻量模型 — 简单任务、快速响应",
        "models": [
            "deepseek-v4-flash",
            "gpt-4o-mini",
            "claude-3-haiku",
            "gemini-1.5-flash",
        ],
    },
    "pro": {
        "cost": 3,
        "capability": "advanced",
        "description": "专业模型 — 中等复杂度、深度推理",
        "models": [
            "deepseek-v4-pro",
            "gpt-4o",
            "claude-3.5-sonnet",
            "gemini-1.5-pro",
        ],
    },
    "ultra": {
        "cost": 5,
        "capability": "expert",
        "description": "顶级模型 — 复杂推理、全网调研",
        "models": [
            "deepseek-v4-pro",
            "gpt-4-turbo",
            "claude-3-opus",
            "gemini-2.0-pro",
        ],
    },
}

# Complexity heuristic thresholds
COMPLEXITY_THRESHOLDS: Dict[str, Dict[str, Any]] = {
    "simple": {
        "max_tokens": 2000,
        "tier": "flash",
        "description": "简单任务 — 无需深度推理",
    },
    "moderate": {
        "max_tokens": 4000,
        "tier": "pro",
        "description": "中等任务 — 需要一定推理",
    },
    "complex": {
        "max_tokens": 8000,
        "tier": "ultra",
        "description": "复杂任务 — 深度推理与分析",
    },
}

# Mode -> team -> tier mapping
MODE_TIER_MAP: Dict[str, Dict[str, str]] = {
    "lite": {
        "team_a": "pro",
        "team_b": "flash",
    },
    "hard": {
        "team_a": "pro",
        "team_b": "flash",
    },
    "super": {
        "team_a": "ultra",
        "team_b": "pro",
    },
}

# Keywords for complexity detection
COMPLEXITY_KEYWORDS: Dict[str, List[str]] = {
    "simple": [
        "simple", "basic", "quick", "short", "trivial", "easy",
        "简单", "基础", "快速",
    ],
    "complex": [
        "complex", "advanced", "deep", "research", "analyze",
        "compare", "evaluate", "synthesize", "comprehensive",
        "复杂", "高级", "深入", "研究", "分析", "对比", "评估",
        "创新", "创新性", "差异化",
    ],
}


class CostTracker:
    """Track LLM API cost estimates across a session.

    Records each model call with its cost multiplier and estimated
    token usage. Provides aggregate cost summaries.

    Attributes:
        _calls: List of recorded call records.
    """

    def __init__(self) -> None:
        self._calls: List[Dict[str, Any]] = []

    def record_estimate(
        self,
        model: str,
        cost_multiplier: float,
        tokens: int = 1000,
    ) -> None:
        """Record an estimated cost for a model call.

        Args:
            model: Model name.
            cost_multiplier: Relative cost multiplier (1=flash, 3=pro, 5=ultra).
            tokens: Estimated token count.
        """
        self._calls.append({
            "model": model,
            "cost_multiplier": cost_multiplier,
            "estimated_tokens": tokens,
        })

    @property
    def total_cost(self) -> float:
        """Total estimated cost in arbitrary units."""
        return sum(
            c["cost_multiplier"] * (c["estimated_tokens"] / 1000.0)
            for c in self._calls
        )

    @property
    def total_calls(self) -> int:
        """Total number of recorded calls."""
        return len(self._calls)

    @property
    def models_used(self) -> Set[str]:
        """Set of unique models used."""
        return set(c["model"] for c in self._calls)

    def summary(self) -> Dict[str, Any]:
        """Get a summary of cost tracking data.

        Returns:
            Dict with total_calls, total_cost, models_used, and
            breakdown_by_model.
        """
        by_model: Dict[str, int] = {}
        for c in self._calls:
            by_model[c["model"]] = by_model.get(c["model"], 0) + 1

        return {
            "total_calls": self.total_calls,
            "total_cost": round(self.total_cost, 2),
            "models_used": sorted(self.models_used),
            "breakdown_by_model": by_model,
        }

    def reset(self) -> None:
        """Clear all recorded costs."""
        self._calls.clear()


class ModelRouter:
    """Cost-aware model router with tiered model assignment.

    Routes tasks to appropriate models based on task complexity and
    execution mode. Tracks estimated costs for reporting.

    The router uses heuristic keyword matching to estimate task
    complexity, then assigns a model tier based on the mode's
    team configuration.

    Attributes:
        config: Optional configuration dict.
        cost_tracker: CostTracker instance for tracking estimates.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}
        self.cost_tracker = CostTracker()

    def estimate_complexity(self, task: str) -> str:
        """Estimate task complexity using heuristic keyword matching.

        Checks for complexity indicators in the task string.
        Returns 'complex' if any complex keyword matches,
        'simple' if any simple keyword matches,
        'moderate' otherwise.

        Args:
            task: The task description string.

        Returns:
            Complexity level: 'simple', 'moderate', or 'complex'.
        """
        if not task:
            return "moderate"

        task_lower = task.lower()

        # Check for complex keywords first (higher priority)
        for keyword in COMPLEXITY_KEYWORDS["complex"]:
            if keyword in task_lower:
                return "complex"

        # Check for simple keywords
        for keyword in COMPLEXITY_KEYWORDS["simple"]:
            if keyword in task_lower:
                return "simple"

        return "moderate"

    def get_model_for_task(
        self,
        task: str,
        mode: str = "lite",
        team: str = "team_a",
    ) -> Dict[str, Any]:
        """Get the appropriate model configuration for a task.

        Uses mode -> team -> tier mapping to determine the model tier,
        then selects the first model in that tier (typically the most
        capable one).

        The result includes the selected model, tier info, complexity
        rating, max tokens, and cost multiplier.

        Args:
            task: Task description for complexity estimation.
            mode: Execution mode ('lite', 'hard', 'super').
            team: Team name ('team_a' or 'team_b').

        Returns:
            Dict with keys: model, tier, complexity, max_tokens,
            cost_multiplier, models (fallback list).
        """
        complexity = self.estimate_complexity(task)

        # Determine tier from mode+team
        mode_map = MODE_TIER_MAP.get(mode, MODE_TIER_MAP["lite"])
        tier_name = mode_map.get(team, "flash")

        # Fall back if tier not found
        tier = MODEL_TIERS.get(tier_name, MODEL_TIERS["flash"])

        # Complexity-based max tokens
        complexity_config = COMPLEXITY_THRESHOLDS.get(
            complexity, COMPLEXITY_THRESHOLDS["moderate"]
        )

        model_config = {
            "model": tier["models"][0] if tier["models"] else "deepseek-v4-flash",
            "tier": tier_name,
            "tier_description": tier.get("description", ""),
            "complexity": complexity,
            "max_tokens": complexity_config["max_tokens"],
            "cost_multiplier": tier["cost"],
            "models": tier["models"],
        }

        # Track cost
        self.cost_tracker.record_estimate(
            model=model_config["model"],
            cost_multiplier=model_config["cost_multiplier"],
            tokens=model_config["max_tokens"],
        )

        return model_config

    def get_judge_model(self, mode: str = "lite") -> Dict[str, Any]:
        """Get the recommended model for judging/evaluation.

        Judge always uses pro tier (or ultra in super mode).

        Args:
            mode: Execution mode.

        Returns:
            Model config for judging.
        """
        if mode == "super":
            return {
                "model": "deepseek-v4-pro",
                "tier": "ultra",
                "cost_multiplier": 5,
                "complexity": "complex",
            }
        return {
            "model": "deepseek-v4-pro",
            "tier": "pro",
            "cost_multiplier": 3,
            "complexity": "moderate",
        }

    def get_research_model(self) -> Dict[str, Any]:
        """Get the model for research tasks (super mode).

        Returns:
            Model config for research.
        """
        return {
            "model": "deepseek-v4-pro",
            "tier": "ultra",
            "capability": "research",
            "cost_multiplier": 5,
            "complexity": "complex",
            "max_tokens": 8000,
        }

    def get_team_tier(self, mode: str, team: str) -> str:
        """Get the model tier for a team in a given mode.

        Args:
            mode: Execution mode.
            team: Team name.

        Returns:
            Tier name ('flash', 'pro', or 'ultra').
        """
        mode_map = MODE_TIER_MAP.get(mode, MODE_TIER_MAP["lite"])
        return mode_map.get(team, "flash")

    @property
    def total_estimated_cost(self) -> float:
        """Get total estimated cost from the cost tracker."""
        return self.cost_tracker.total_cost

    def cost_summary(self) -> Dict[str, Any]:
        """Get a cost summary for reporting.

        Returns:
            Dict with cost tracking summary.
        """
        return self.cost_tracker.summary()

    def reset_costs(self) -> None:
        """Reset the cost tracker."""
        self.cost_tracker.reset()
