"""
Orchestrator agent for GAAL v3.

The OrchestratorAgent manages the entire arena loop:
1. Parses the goal and determines mode
2. Builds and configures TeamAgent instances (Team A + Team B)
3. Coordinates the propose -> evaluate -> eliminate -> deep_dive cycle
4. Manages JudgeAgent for final scoring
5. Handles self-evolution
6. Delegates all LLM work to sub-sessions for zero-leak isolation

Implements the CrewAI Hierarchical pattern: Orchestrator (manager)
delegates to TeamAgents (workers) and JudgeAgent (evaluator).
"""
from __future__ import annotations
import json
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

from .base import BaseAgent, AgentCapability, AgentContext

logger = logging.getLogger(__name__)


class OrchestratorAgent(BaseAgent):
    """Main orchestrator that manages the GAAL v3 arena loop.

    Orchestrator acts as the central manager in the CrewAI Hierarchical
    pattern, delegating proposal generation to TeamAgents and evaluation
    to JudgeAgent.

    The arena loop:
    1. Parse goal & determine mode
    2. Research (super mode only) — delegated
    3. Proposals from Team A + Team B (parallel, delegated)
    4. Aggregate & scorecard
    5. Eliminate weak proposals
    6. Deep dive comparison (head-to-head)
    7. Judge final scoring
    8. Evolve (self-improvement)
    9. Report generation

    Attributes:
        team_a_config: Configuration for Team A.
        team_b_config: Configuration for Team B.
        judge_config: Configuration for the judge.
        proposals: All proposals from both teams.
        eliminations: Elimination records.
        execution_history: Graph execution history.
        current_loop: Current arena loop iteration.
    """

    def __init__(
        self,
        name: str = "Orchestrator",
        context: Optional[AgentContext] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(name=name, context=context, config=config)
        self.capabilities = [
            AgentCapability(
                name="goal_parsing",
                description="解析目标并确定运行模式",
                complexity="simple",
            ),
            AgentCapability(
                name="arena_management",
                description="管理竞技场循环：提案→评估→淘汰→深度对比→评分",
                complexity="complex",
            ),
            AgentCapability(
                name="team_coordination",
                description="协调 Team A 和 Team B 的提案生成",
                complexity="moderate",
            ),
            AgentCapability(
                name="self_evolution",
                description="自我进化：修改配置/评分卡权重",
                complexity="complex",
            ),
        ]

        # Team configurations (from config dict)
        teams = self.config.get("teams", {})
        self.team_a_config = teams.get("team_a", {})
        self.team_b_config = teams.get("team_b", {})

        # Judge configuration
        self.judge_config = self.config.get("judge", {})

        # Arena state
        self.proposals: List[Dict[str, Any]] = []
        self.eliminations: List[Dict[str, Any]] = []
        self.execution_history: List[Dict[str, Any]] = []
        self.current_loop: int = 0
        self.final_scores: Dict[str, Any] = {}
        self.components: List[Dict[str, str]] = []
        self.suggestions: Dict[str, List[str]] = {}

    def parse_goal(self) -> Dict[str, Any]:
        """Parse the goal and determine execution mode.

        Analyzes the goal text to determine:
        - Is it simple or complex?
        - What mode is appropriate?
        - What dimensions to evaluate on?

        Returns:
            Dict with parsed goal info: mode, is_simple, dimensions.
        """
        goal = self.context.goal

        # Mode detection (explicit config override first)
        mode = self.config.get("gaal", {}).get("mode", "lite")

        # Heuristic: complex goals get auto-upgrade
        complexity_keywords = [
            "research", "analyze", "compare", "architect", "design",
            "框架", "架构", "研究", "对比", "分析", "设计",
        ]
        goal_lower = goal.lower()
        complexity_score = sum(1 for kw in complexity_keywords if kw in goal_lower)

        if complexity_score >= 3 and mode == "lite":
            mode = "hard"
        elif complexity_score >= 5:
            mode = "super"

        # Judge dimensions from config
        dimensions = self.judge_config.get("dimensions", [])

        return {
            "mode": mode,
            "is_simple": complexity_score < 2,
            "complexity_score": complexity_score,
            "dimensions": dimensions,
            "max_loops": self.config.get("gaal", {}).get("max_loops", 4),
            "max_proposals": self.config.get("gaal", {}).get("max_proposals_per_team", 2),
        }

    def determine_mode(self, parsed: Dict[str, Any]) -> str:
        """Determine the execution mode based on parsed goal.

        Args:
            parsed: Output from parse_goal().

        Returns:
            Mode string: 'lite', 'hard', or 'super'.
        """
        return parsed["mode"]

    def get_loop_params(self) -> Dict[str, int]:
        """Get loop parameters based on current mode.

        Returns:
            Dict with max_loops, max_proposals_per_team.
        """
        mode = self.context.mode
        params = {
            "lite": {"max_loops": 4, "max_proposals_per_team": 2},
            "hard": {"max_loops": 10, "max_proposals_per_team": 10},
            "super": {"max_loops": 20, "max_proposals_per_team": 10},
        }
        return params.get(mode, params["lite"])

    def execute(self) -> Dict[str, Any]:
        """Execute the orchestrator's main logic.

        This method sets up the arena state and returns configuration
        data. The actual graph execution is managed by the GAALOrchestrator
        in core/orchestrator.py.

        Returns:
            Dict with execution parameters.
        """
        self.start_timer()
        parsed = self.parse_goal()

        return {
            "parsed_goal": parsed,
            "team_a_config": self.team_a_config,
            "team_b_config": self.team_b_config,
            "judge_config": self.judge_config,
            "loop_params": self.get_loop_params(),
        }

    def summarize(self) -> Dict[str, Any]:
        """Generate a clean summary for the parent session.

        Zero-leak: Only includes high-level results, no tool internals.

        Returns:
            Clean summary dict.
        """
        return {
            "agent": self.name,
            "type": "OrchestratorAgent",
            "goal": self.context.goal,
            "mode": self.context.mode,
            "proposals_generated": len(self.proposals),
            "eliminations": len(self.eliminations),
            "loops_completed": self.current_loop,
            "final_scores": self.final_scores,
            "elapsed_seconds": round(self.elapsed_time, 2),
        }
