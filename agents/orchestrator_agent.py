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
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from .base import BaseAgent, AgentCapability, AgentContext

logger = logging.getLogger(__name__)


# ── Goal Analysis Helpers ────────────────────────────────────────────

# Technical terms that indicate a complex problem domain
TECH_DOMAIN_TERMS = [
    "分布式", "distributed", "微服务", "microservice", "容器", "container",
    "kubernetes", "k8s", "docker", "云原生", "cloud.?native",
    "serverless", "事件驱动", "event.?driven", "cqrs", "事件溯源",
    "event.?sourcing", "消息队列", "message.?queue", "kafka",
    "流处理", "stream.?process", "实时", "real.?time", "大数据",
    "big.?data", "机器学习", "machine.?learn", "深度学习", "deep.?learn",
    "区块链", "blockchain", "物联网", "iot", "边缘计算", "edge.?comput",
    "高并发", "high.?concurr", "高可用", "high.?availab", "高吞吐",
    "high.?throughput", "低延迟", "low.?latency", "秒杀", "flash.?sale",
    "推荐系统", "recommend", "搜索引擎", "search.?engine",
    "企业级", "enterprise", "saas", "paas", "iaas",
    "增量备份", "incremental.?backup", "加密存储", "encrypt",
    "定时调度", "schedule", "负载均衡", "load.?balanc",
]

# Action verbs that indicate the scope of work
SCOPE_VERBS = {
    "research": ["研究", "调研", "research", "investigat", "survey", "探索", "explore"],
    "design": ["设计", "design", "架构", "architect", "规划", "plan", "建模", "model"],
    "build": ["实现", "implement", "开发", "develop", "构建", "build", "创建", "create"],
    "analyze": ["分析", "analyze", "对比", "compare", "评估", "evaluat", "review"],
    "optimize": ["优化", "optimize", "重构", "refactor", "改造", "migrat"],
}

# Requirement delimiter patterns (Chinese + English)
REQUIREMENT_DELIMITERS = r"[、，,;\n\r]+"


def _count_tech_domain_terms(text: str) -> int:
    """Count technical domain terms in text."""
    text_lower = text.lower()
    count = 0
    for term in TECH_DOMAIN_TERMS:
        if re.search(term.lower(), text_lower):
            count += 1
    return count


def _count_action_verbs(text: str) -> Dict[str, int]:
    """Count action verbs by category."""
    text_lower = text.lower()
    counts = {}
    for category, verbs in SCOPE_VERBS.items():
        count = sum(1 for v in verbs if re.search(v.lower(), text_lower))
        if count > 0:
            counts[category] = count
    return counts


def _parse_requirements(goal: str) -> List[str]:
    """Parse individual requirements from a goal string.

    Handles both Chinese and English delimiters.
    """
    # Split by common delimiters
    parts = re.split(REQUIREMENT_DELIMITERS, goal)
    requirements = [p.strip() for p in parts if p.strip()]
    return requirements


def _detect_mode(goal: str, config: Dict[str, Any]) -> str:
    """Detect the appropriate execution mode based on goal analysis.

    Analyzes goal complexity using multiple signals:
    - Number of technical domain terms
    - Number of action verbs
    - Number of distinct requirements
    - Goal length
    - Presence of specific mode indicator keywords

    Args:
        goal: The goal text.
        config: Full config dict.

    Returns:
        Mode string: 'lite', 'hard', or 'super'.
    """
    # Check explicit config override first
    config_mode = config.get("gaal", {}).get("mode", "auto")
    if config_mode in ("lite", "hard", "super"):
        return config_mode

    goal_lower = goal.lower()

    # Quick check: empty/trivial goal
    if not goal.strip() or len(goal.strip()) < 5:
        return "lite"

    # Multi-signal complexity analysis
    tech_terms = _count_tech_domain_terms(goal)
    verb_counts = _count_action_verbs(goal)
    requirements = _parse_requirements(goal)
    req_count = len(requirements)
    goal_chars = len(goal)

    # Build complexity score (0-20 scale)
    complexity = 0.0

    # Signal 1: Technical term density (weight: 0-5)
    complexity += min(tech_terms * 1.0, 5.0)

    # Signal 2: Number of distinct requirements (weight: 0-5)
    complexity += min(req_count * 1.5, 5.0)

    # Signal 3: Goal length (weight: 0-4)
    if goal_chars > 100:
        complexity += 4.0
    elif goal_chars > 60:
        complexity += 2.5
    elif goal_chars > 30:
        complexity += 1.0

    # Signal 4: Action verb diversity (weight: 0-3)
    complexity += min(len(verb_counts) * 1.0, 3.0)

    # Signal 5: Multiple aspects detected (weight: 0-3)
    aspects_url = re.findall(r"支持|提供|实现|支持|含|包括|including|with|and", goal_lower)
    complexity += min(len(aspects_url) * 0.5, 3.0)

    # Mode determination based on complexity
    if complexity >= 10.0:
        return "super"
    elif complexity >= 5.0:
        return "hard"
    else:
        return "lite"


def _compute_simple_flag(goal: str) -> bool:
    """Determine if a goal is simple (trivial/one-shot).

    A goal is simple if:
    - Very short (< 10 chars)
    - No technical terms
    - No delimited requirements
    - Basic question/request format
    """
    goal = goal.strip()
    if len(goal) < 10:
        return True

    tech_terms = _count_tech_domain_terms(goal)
    if tech_terms == 0 and len(goal) < 30:
        return True

    # Has multiple requirements?
    requirements = _parse_requirements(goal)
    if len(requirements) <= 1 and tech_terms == 0:
        return True

    return False


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

        Analyzes the goal text using multi-signal complexity detection:
        - Technical term density
        - Action verb diversity
        - Requirement count (split by Chinese/English delimiters)
        - Goal length and structure

        Returns:
            Dict with parsed goal info: mode, is_simple, complexity details.
        """
        goal = self.context.goal

        # Multi-signal complexity analysis
        requirements = _parse_requirements(goal)
        tech_terms = _count_tech_domain_terms(goal)
        verb_counts = _count_action_verbs(goal)
        req_count = len(requirements)
        goal_chars = len(goal)

        # Mode detection using intelligent analysis
        mode = _detect_mode(goal, self.config)
        is_simple = _compute_simple_flag(goal)

        # Build complexity breakdown
        complexity_breakdown = {
            "technical_terms": tech_terms,
            "requirements_detected": req_count,
            "requirements_list": requirements[:5],  # Top 5 for context
            "goal_length": goal_chars,
            "action_verbs": verb_counts,
            "has_multiple_requirements": req_count > 1,
        }

        # Judge dimensions from config
        dimensions = self.judge_config.get("dimensions", [])

        logger.info(
            "parse_goal: mode=%s, is_simple=%s, tech_terms=%d, reqs=%d",
            mode, is_simple, tech_terms, req_count,
        )

        return {
            "mode": mode,
            "is_simple": is_simple,
            "complexity_score": min(tech_terms + req_count, 10),
            "complexity_breakdown": complexity_breakdown,
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
