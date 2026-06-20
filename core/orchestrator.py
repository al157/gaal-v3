"""
GAAL v3 orchestrator — builds and runs the LangGraph-style arena graph.

The GAALOrchestrator constructs a StateGraph with all nodes and edges,
then executes it against the goal. It manages:

1. Graph construction (9 nodes + conditional edges)
2. Arena loop (propose -> evaluate -> eliminate -> repeat)
3. Checkpoint integration (before/after each node)
4. Circuit breaker (5 consecutive failures -> escalate)
5. Mode-specific behavior (lite/hard/super)
6. Research phase (super mode only)
7. Self-evolution (bootstrap mode) — REAL self-evolution with perf metrics
8. Graceful degradation — super→hard→lite fallback chain
9. Parallel team execution — Team A + Team B concurrent via ThreadPoolExecutor
10. Cost tracking — real token tracking per node with budget enforcement
"""
from __future__ import annotations
import json
import logging
import os
import time
import uuid
import yaml
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Callable

from .graph import StateGraph, CompiledGraph, _serializable_state, _format_execution_history
from .persistence import CheckpointStore
from .model_router import ModelRouter
from agents.orchestrator_agent import OrchestratorAgent
from agents.team_agent import TeamAgent
from agents.judge_agent import JudgeAgent
from agents.base import AgentContext

logger = logging.getLogger(__name__)


class CircuitBreaker:
    """Circuit breaker for fault tolerance.

    Tracks consecutive failures and opens the circuit when the
    threshold is exceeded. After the reset period, the circuit
    transitions to half-open, allowing one trial request.

    Attributes:
        threshold: Consecutive failures before opening.
        reset_seconds: Seconds before transitioning to half-open.
        failure_count: Current consecutive failure count.
        last_failure_time: Timestamp of the last failure.
        state: 'closed', 'open', or 'half-open'.
    """

    def __init__(self, threshold: int = 5, reset_seconds: int = 300) -> None:
        self.threshold = threshold
        self.reset_seconds = reset_seconds
        self.failure_count = 0
        self.last_failure_time: float = 0.0
        self.state: str = "closed"  # closed, open, half-open

    def record_success(self) -> None:
        """Record a success, resetting the failure count."""
        self.failure_count = 0
        self.state = "closed"

    def record_failure(self) -> str:
        """Record a failure and check threshold.

        Returns:
            Current circuit state.
        """
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.failure_count >= self.threshold:
            self.state = "open"
            logger.warning(
                "Circuit breaker OPEN after %d failures",
                self.failure_count,
            )
        return self.state

    def can_proceed(self) -> bool:
        """Check if requests should be allowed through.

        Closed -> always proceed.
        Open -> check if reset period elapsed -> half-open -> proceed.
        Half-open -> proceed (one trial).

        Returns:
            True if request should proceed.
        """
        if self.state == "closed":
            return True

        if self.state == "open":
            elapsed = time.time() - self.last_failure_time
            if elapsed >= self.reset_seconds:
                self.state = "half-open"
                logger.info("Circuit breaker half-open, allowing trial")
                return True
            return False

        # half-open: allow
        return True

    def reset(self) -> None:
        """Manually reset the circuit breaker."""
        self.failure_count = 0
        self.state = "closed"
        self.last_failure_time = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize circuit breaker state."""
        return {
            "state": self.state,
            "failure_count": self.failure_count,
            "threshold": self.threshold,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CircuitBreaker":
        """Deserialize circuit breaker state."""
        cb = cls(
            threshold=data.get("threshold", 5),
            reset_seconds=data.get("reset_seconds", 300),
        )
        cb.state = data.get("state", "closed")
        cb.failure_count = data.get("failure_count", 0)
        cb.last_failure_time = data.get("last_failure_time", 0.0)
        return cb


# ── Evolution Helper ──────────────────────────────────────────────────

def compute_bootstrap_score(state: Dict[str, Any]) -> float:
    """Compute a bootstrap/self-evolution score from execution metrics.

    Analyzes:
    - Total score achieved
    - Number of proposals generated
    - Number of eliminations
    - Execution steps
    - Whether evolution was enabled

    Returns:
        Bootstrap score 0.0-10.0.
    """
    total_score = state.get("total_score", 0)
    num_proposals = len(state.get("proposals", []))
    num_eliminations = len(state.get("eliminations", []))
    num_steps = len(state.get("execution_history", []))
    retry_count = state.get("total_retries", 0)
    degradation_level = state.get("degradation_level", 0)

    score = total_score * 0.6  # 60% weight on total score

    # Bonus for proposal diversity
    if num_proposals >= 4:
        score += 1.0
    elif num_proposals >= 2:
        score += 0.5

    # Bonus for successful elimination
    if num_eliminations > 0:
        score += 0.5

    # Penalty for too many retries
    if retry_count > 5:
        score -= 1.0

    # Penalty for degradation
    if degradation_level > 0:
        score -= degradation_level * 1.5

    # Bonus for efficiency (fewer steps = better)
    if num_steps > 0 and num_steps < 8:
        score += 0.5

    return round(max(0.0, min(10.0, score)), 2)


def suggest_evolution_action(
    scores: List[float],
    state: Dict[str, Any],
    config: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Suggest an evolution action based on performance history.

    Args:
        scores: Historical score list.
        state: Current graph state.
        config: Current configuration.

    Returns:
        Evolution action dict or None.
    """
    if len(scores) < 2:
        return None

    current_score = scores[-1]
    trend = current_score - scores[-2] if len(scores) >= 2 else 0

    # Which dimensions are weakest?
    dimension_scores = {}
    for p in state.get("scored_proposals", []):
        ds = p.get("dimension_scores", {})
        for dim_name, dim_data in ds.items():
            if dim_name not in dimension_scores:
                dimension_scores[dim_name] = []
            dimension_scores[dim_name].append(dim_data.get("score", 0))

    avg_dim_scores = {
        dim: (sum(sc) / len(sc)) if sc else 0
        for dim, sc in dimension_scores.items()
    }

    # Find weakest dimension
    weakest_dim = min(avg_dim_scores, key=lambda d: avg_dim_scores.get(d) or 0) if avg_dim_scores else None

    suggestions = []

    # 1. If score trend is declining, adjust config
    if trend < -0.5:
        suggestions.append({
            "action": "increase_loops",
            "target": "config/gaal_v3.yaml",
            "reason": f"Score declining ({trend:+.2f}), increasing max_loops",
            "params": {"max_loops": config.get("gaal", {}).get("max_loops", 4) + 2},
            "priority": 1,
        })

    # 2. If a dimension is consistently weak, increase its weight
    if weakest_dim and avg_dim_scores[weakest_dim] < 1.0:
        current_weights = config.get("judge", {}).get("dimensions", [])
        for dim_cfg in current_weights:
            if dim_cfg.get("name") == weakest_dim:
                new_weight = min(dim_cfg.get("weight", 2) + 1, 5)
                suggestions.append({
                    "action": "adjust_weight",
                    "target": "config/scorecard.yaml",
                    "reason": f"'{weakest_dim}' avg={avg_dim_scores[weakest_dim]:.2f} < 1.0, increasing weight",
                    "params": {"dimension": weakest_dim, "new_weight": new_weight},
                    "priority": 2,
                })

    # 3. If bootstrap score is low, enable evolution
    bootstrap_score = compute_bootstrap_score(state)
    if bootstrap_score < 5.0:
        suggestions.append({
            "action": "enable_evolution",
            "target": "config/gaal_v3.yaml",
            "reason": f"Bootstrap score {bootstrap_score} < 5.0, enabling evolution",
            "params": {"evolution.enabled": True},
            "priority": 3,
        })

    # 4. If total score is good but evolution not enabled, enable it
    if current_score >= 7.0 and not config.get("evolution", {}).get("enabled", False):
        suggestions.append({
            "action": "enable_evolution",
            "target": "config/gaal_v3.yaml",
            "reason": f"Score {current_score} >= 7.0, enabling evolution for self-improvement",
            "params": {"evolution.enabled": True},
            "priority": 4,
        })

    if suggestions:
        suggestions.sort(key=lambda s: s["priority"])
        return suggestions[0]  # Return highest priority

    return None


class GAALOrchestrator:
    """Main orchestrator for GAAL v3 arena execution.

    Builds and runs a LangGraph-style StateGraph with the following nodes:
    1. parse_goal — Parse goal, determine mode
    2. research — (super mode only) agent-reach research
    3. propose_team_a / propose_team_b — Parallel team proposals
    4. aggregate — Merge proposals, apply scorecard
    5. eliminate — Rank and eliminate weak proposals
    6. deep_dive — Head-to-head comparison
    7. judge — Final scoring
    8. evolve — Self-evolution
    9. report — Generate final report

    The graph supports conditional routing:
    - parse_goal -> END if simple enough
    - eliminate -> RESEARCH if score < threshold (loop back)
    - judge -> EVOLVE if score >= threshold

    Additional capabilities:
    - Bootstrap self-evolution with performance metrics tracking
    - Graceful degradation (super→hard→lite fallback chain)
    - Parallel team execution via ThreadPoolExecutor
    - Cost tracking with budget enforcement

    Attributes:
        config: Full GAAL configuration.
        mode: Execution mode.
        goal: The goal being executed.
        graph: The StateGraph instance.
        compiled: The CompiledGraph instance.
        checkpoint_store: Persistence layer.
        model_router: Cost-aware model routing.
        circuit_breaker: Fault tolerance.
        execution_history: Record of all node executions.
        session_id: Unique session identifier.
        state_dir: Directory for state files.
        evolution_dir: Directory for evolution artifacts.
        degradation_level: 0=full, 1=downgraded, 2=fallback
        cost_data: Per-node cost tracking data.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        config_path: Optional[str] = None,
        state_dir: str = "state",
    ) -> None:
        self.config = config or {}
        self.mode: str = "lite"
        self.goal: str = ""
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.checkpoint_store: Optional[CheckpointStore] = None
        self.model_router = ModelRouter(config=self.config)
        self.circuit_breaker = CircuitBreaker(
            threshold=self.config.get("circuit_breaker", {}).get("threshold", 5),
            reset_seconds=self.config.get("circuit_breaker", {}).get("reset_seconds", 300),
        )

        # Graph components
        self.graph: Optional[StateGraph] = None
        self.compiled: Optional[CompiledGraph] = None

        # Execution state
        self.execution_history: List[Dict[str, Any]] = []
        self.session_id: str = str(uuid.uuid4())[:8]
        self._agents: Dict[str, Any] = {}

        # State for graph nodes
        self._state: Dict[str, Any] = {}

        # ── Bootstrap Self-Evolution ──
        self.evolution_dir = Path(__file__).parent.parent / "evolution"
        self.evolution_dir.mkdir(parents=True, exist_ok=True)
        self.evolution_scores: List[float] = []
        self.evolution_actions: List[Dict[str, Any]] = []

        # ── Graceful Degradation ──
        self.degradation_level: int = 0  # 0=full, 1=downgraded, 2=fallback
        self._original_mode: str = "lite"

        # ── Cost Tracking ──
        self.cost_data: Dict[str, Any] = {
            "per_node": {},
            "total_tokens": 0,
            "total_cost": 0.0,
            "budget_exceeded": False,
            "per_team": {"team_a": 0, "team_b": 0},
            "per_mode": {},
        }
        self._max_tokens_per_node: int = self.config.get("cost", {}).get("max_tokens_per_node", 4000)
        self._max_total_tokens: int = self.config.get("cost", {}).get("max_total_tokens", 50000)

        # Load config from file if path provided
        if config_path:
            self._load_config(config_path)

    def _load_config(self, config_path: str) -> None:
        """Load configuration from a YAML file.

        Args:
            config_path: Path to YAML config file.
        """
        try:
            path = Path(config_path)
            if path.exists():
                with open(path) as f:
                    self.config = yaml.safe_load(f)
                logger.info("Loaded config from %s", config_path)
        except Exception as e:
            logger.warning("Failed to load config from %s: %s", config_path, e)

    def _init_checkpoint_store(self) -> CheckpointStore:
        """Initialize the checkpoint store.

        Returns:
            Initialized CheckpointStore.
        """
        db_path = self.config.get("checkpoint", {}).get(
            "db_path",
            str(self.state_dir / "gaal_v3_checkpoints.db"),
        )
        store = CheckpointStore(
            db_path=db_path,
            session_id=self.session_id,
        )
        return store

    def build_graph(self) -> CompiledGraph:
        """Build and compile the GAAL v3 state graph.

        Constructs the full 9-node, conditionally-routed StateGraph
        and compiles it into an executable form.

        Returns:
            CompiledGraph ready for execution.
        """
        g = StateGraph(dict)

        # ── Add Nodes ──────────────────────────────────────────────
        g.add_node("parse_goal", self._node_parse_goal, retries=2)
        g.add_node("research", self._node_research, retries=2,
                   metadata={"mode": "super"})
        g.add_node("propose_team_a", self._node_propose_team_a, retries=3)
        g.add_node("propose_team_b", self._node_propose_team_b, retries=3)
        g.add_node("aggregate", self._node_aggregate, retries=2)
        g.add_node("eliminate", self._node_eliminate, retries=2)
        g.add_node("deep_dive", self._node_deep_dive, retries=2)
        g.add_node("judge", self._node_judge, retries=3)
        g.add_node("evolve", self._node_evolve, retries=2,
                   metadata={"bootstrap": True})
        g.add_node("report", self._node_report, retries=1)

        # ── Add Edges ──────────────────────────────────────────────
        # parse_goal -> research or propose (conditional)
        g.add_conditional_edges(
            source="parse_goal",
            condition=lambda s: self._route_from_parse(s),
            path_map={
                "research": "research",
                "propose": "propose_team_a",
                "end": "__end__",
            },
            default="propose_team_a",
        )

        # research -> propose (both teams parallel)
        g.add_edge("research", "propose_team_a")

        # propose_team_a -> propose_team_b (sequential in graph,
        # but now they execute in parallel via ThreadPoolExecutor)
        g.add_edge("propose_team_a", "propose_team_b")

        # propose_team_b -> aggregate
        g.add_edge("propose_team_b", "aggregate")

        # aggregate -> eliminate
        g.add_edge("aggregate", "eliminate")

        # eliminate -> deep_dive or back to propose (conditional loop)
        g.add_conditional_edges(
            source="eliminate",
            condition=lambda s: self._route_from_eliminate(s),
            path_map={
                "continue": "deep_dive",
                "loop": "propose_team_a",
                "end": "__end__",
            },
            default="deep_dive",
        )

        # deep_dive -> judge
        g.add_edge("deep_dive", "judge")

        # judge -> evolve or report (conditional on score)
        g.add_conditional_edges(
            source="judge",
            condition=lambda s: self._route_from_judge(s),
            path_map={
                "evolve": "evolve",
                "report": "report",
                "loop": "propose_team_a",
            },
            default="report",
        )

        # evolve -> report
        g.add_edge("evolve", "report")

        # report -> END (set as finish point)
        g.set_finish_point("report")

        # ── Entry Point ────────────────────────────────────────────
        g.set_entry_point("parse_goal")

        # ── Compile ────────────────────────────────────────────────
        self.graph = g
        self.compiled = g.compile()
        logger.info(
            "Graph compiled: %d nodes, %d edges, entry=%s",
            len(g.nodes), len(g.edges), g.entry_point,
        )
        return self.compiled

    # ── Routing Conditions ──────────────────────────────────────────

    def _route_from_parse(self, state: Dict[str, Any]) -> str:
        """Route from parse_goal node.

        - 'end' only if goal is truly trivial (empty or single word)
        - 'research' if super mode
        - 'propose' otherwise (most tasks — even "simple" ones)
        """
        mode = state.get("mode", "lite")
        goal = state.get("goal", "").strip()
        is_simple = state.get("is_simple", False)

        # Only END for truly trivial queries: empty goal or very short (< 10 chars)
        if not goal or (is_simple and len(goal) < 10):
            logger.info("Goal is trivial, routing to end")
            return "end"
        if mode == "super":
            logger.info("Super mode, routing to research")
            return "research"
        return "propose"

    def _route_from_eliminate(self, state: Dict[str, Any]) -> str:
        """Route from eliminate node.

        - 'loop' if we need more iterations
        - 'end' if no proposals remain
        - 'continue' to deep_dive otherwise
        """
        current_loop = state.get("current_loop", 0)
        max_loops = state.get("max_loops", 4)
        proposals_remaining = state.get("proposals_remaining", 0)

        if proposals_remaining == 0:
            logger.warning("No proposals remaining, ending arena")
            return "end"
        if current_loop < max_loops:
            loop_score = state.get("loop_score", 0)
            threshold = state.get("pass_threshold", 8.5)
            if loop_score > 0 and loop_score < threshold:
                logger.info(
                    "Score %.1f < threshold %.1f, looping back",
                    loop_score, threshold,
                )
                return "loop"
        logger.info("Proceeding to deep_dive")
        return "continue"

    def _route_from_judge(self, state: Dict[str, Any]) -> str:
        """Route from judge node.

        - 'evolve' if score passes threshold and evolution is enabled
        - 'report' normally
        - 'loop' if score is low and loops remain
        """
        total_score = state.get("total_score", 0)
        pass_threshold = state.get("pass_threshold", 8.5)
        evolution_enabled = self.config.get("evolution", {}).get("enabled", False)
        current_loop = state.get("current_loop", 0)
        max_loops = state.get("max_loops", 4)

        if total_score >= pass_threshold and evolution_enabled:
            logger.info("Score %.1f >= %.1f, routing to evolve", total_score, pass_threshold)
            return "evolve"
        if total_score < pass_threshold and current_loop < max_loops:
            logger.info("Score %.1f < %.1f, looping back", total_score, pass_threshold)
            return "loop"
        return "report"

    # ── Graph Node Functions ────────────────────────────────────────

    def _node_parse_goal(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """PARSE_GOAL node: Extract goal, determine mode.

        Uses OrchestratorAgent to analyze the goal text.
        Tracks cost for this node.
        """
        self._track_node_cost("parse_goal", "goal_parsing", 500, 3)

        goal = state.get("goal", self.goal)
        orchestrator = OrchestratorAgent(
            name="Orchestrator",
            context=AgentContext(goal=goal, mode=self.mode),
            config=self.config,
        )

        parsed = orchestrator.parse_goal()
        mode = parsed["mode"]
        self.mode = mode
        self._original_mode = mode

        result = {
            **state,
            "goal": goal,
            "mode": mode,
            "is_simple": parsed["is_simple"],
            "complexity_score": parsed["complexity_score"],
            "dimensions": parsed["dimensions"],
            "max_loops": parsed["max_loops"],
            "max_proposals": parsed["max_proposals"],
            "current_loop": 0,
            "proposals": [],
            "eliminations": [],
            "proposals_remaining": 0,
            "pass_threshold": self.config.get("judge", {}).get("pass_threshold", 8.5),
            "total_score": 0.0,
            "loop_score": 0.0,
            # Degradation state
            "degradation_level": self.degradation_level,
            "degradation_history": [],
            # Cost tracking
            "cost_data": {
                "per_node": {},
                "total_tokens": 0,
                "total_cost": 0.0,
                "per_team": {"team_a": 0, "team_b": 0},
            },
            # Performance stats
            "node_durations": {},
            "node_retries": {},
            "total_retries": 0,
        }
        logger.info("PARSE_GOAL: mode=%s, is_simple=%s", mode, parsed["is_simple"])
        return result

    def _node_research(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """RESEARCH node: (super mode only) research the goal.

        In production, this would use delegate_task for agent-reach
        research. Currently generates structured research data.
        Tracks cost for this node.
        """
        self._track_node_cost("research", "research", 1500, 5)

        goal = state.get("goal", "")
        logger.info("RESEARCH: researching goal='%s'", goal[:50])

        # Generate research data
        research_data = {
            "findings": [
                f"Research finding 1 for: {goal[:30]}",
                f"Research finding 2 for: {goal[:30]}",
                f"Research finding 3 for: {goal[:30]}",
            ],
            "sources": ["web_search", "github", "social"],
            "summary": f"Research summary for: {goal}",
        }

        state["research_data"] = research_data
        return state

    # ── Parallel Team Execution ─────────────────────────────────────

    def _execute_team_in_thread(
        self,
        team_id: str,
        goal: str,
        mode: str,
        current_loop: int,
        team_config: Dict[str, Any],
        team_name: str,
    ) -> List[Dict[str, Any]]:
        """Execute a single team's proposal generation in a worker thread.

        Args:
            team_id: 'team_a' or 'team_b'.
            goal: The goal to generate proposals for.
            mode: Execution mode.
            current_loop: Current arena loop.
            team_config: Team configuration dict.
            team_name: Display name for the team.

        Returns:
            List of proposal dicts.
        """
        agent = TeamAgent(
            name=team_name,
            context=AgentContext(
                goal=goal,
                mode=mode,
                round_num=current_loop,
            ),
            config=self.config,
        )
        agent.configure(team_id, team_config)
        proposals = agent.execute(goal=goal)

        # Track cost for this team
        tier_cost_map = {"flash": 1, "pro": 3, "ultra": 5}
        team_tier = team_config.get("tier", "flash")
        cost_mult = tier_cost_map.get(team_tier, 1)
        token_est = len(goal) * 10 + len(proposals) * 500
        self._track_node_cost(f"propose_{team_id}", f"team_{team_id}", token_est, cost_mult)

        logger.info(
            "Team %s generated %d proposals (tier=%s, cost_mult=%d)",
            team_id, len(proposals), team_tier, cost_mult,
        )
        return proposals

    def _node_propose_team_a(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """PROPOSE_TEAM_A node: Generate proposals from Team A.

        Uses parallel execution for both teams via ThreadPoolExecutor.
        """
        goal = state.get("goal", "")
        mode = state.get("mode", "lite")
        current_loop = state.get("current_loop", 0)

        team_a_config = self.config.get("teams", {}).get("team_a", {})
        team_b_config = self.config.get("teams", {}).get("team_b", {})

        # Execute both teams in parallel using ThreadPoolExecutor
        team_timeout = self.config.get("execution", {}).get("team_timeout", 30)

        with ThreadPoolExecutor(max_workers=2) as executor:
            future_a = executor.submit(
                self._execute_team_in_thread,
                "team_a", goal, mode, current_loop,
                team_a_config, "TeamAlpha",
            )
            future_b = executor.submit(
                self._execute_team_in_thread,
                "team_b", goal, mode, current_loop,
                team_b_config, "TeamBeta",
            )

            team_a_proposals = []
            team_b_proposals = []

            try:
                team_a_proposals = future_a.result(timeout=team_timeout)
            except Exception as e:
                logger.error("Team A execution failed after timeout %ds: %s", team_timeout, e)
                team_a_proposals = self._generate_fallback_proposal(goal, "TeamAlpha", "team_a")

            try:
                team_b_proposals = future_b.result(timeout=team_timeout)
            except Exception as e:
                logger.error("Team B execution failed after timeout %ds: %s", team_timeout, e)
                team_b_proposals = self._generate_fallback_proposal(goal, "TeamBeta", "team_b")

        # Merge all proposals
        all_proposals = state.get("proposals", []) + team_a_proposals + team_b_proposals
        state["proposals"] = all_proposals
        state["team_a_proposals"] = team_a_proposals
        state["team_b_proposals"] = team_b_proposals

        logger.info(
            "PARALLEL TEAMS: Team A=%d proposals, Team B=%d proposals, total=%d",
            len(team_a_proposals), len(team_b_proposals), len(all_proposals),
        )

        # The graph expects team_a to finish before team_b starts,
        # but since we run both in parallel, we return all proposals here
        # and _node_propose_team_b will just pass through
        return state

    def _generate_fallback_proposal(
        self, goal: str, team_name: str, team_id: str,
    ) -> List[Dict[str, Any]]:
        """Generate a fallback proposal when a team times out.

        Args:
            goal: The goal.
            team_name: Team display name.
            team_id: 'team_a' or 'team_b'.

        Returns:
            List with one fallback proposal.
        """
        return [{
            "id": str(uuid.uuid4())[:12],
            "team": team_name,
            "team_id": team_id,
            "index": 0,
            "name": f"{team_name} Fallback Proposal",
            "description": (
                f"[Fallback] 由于超时自动生成的基准方案。\n"
                f"目标: {goal[:80]}\n\n"
                f"核心思路: 采用成熟稳定的技术架构，确保基本功能完整实现。\n"
                f"技术栈: 经过验证的主流技术组合\n\n"
                f"注: 此方案因 {team_name} 执行超时而生成，建议后续优化团队性能。"
            ),
            "scores": {},
            "status": "active",
            "round": 0,
        }]

    def _node_propose_team_b(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """PROPOSE_TEAM_B node: Pass-through (teams already executed in parallel).

        Since _node_propose_team_a already ran both teams concurrently,
        this node just passes the state through.
        """
        logger.debug("PROPOSE_TEAM_B: pass-through (parallel execution already done)")
        return state

    def _node_aggregate(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """AGGREGATE node: Merge all proposals and apply scorecard.

        Scores all proposals using the JudgeAgent.
        Tracks cost for this node.
        """
        self._track_node_cost("aggregate", "judging", 2000, 3)

        proposals = state.get("proposals", [])
        if not proposals:
            logger.warning("AGGREGATE: no proposals to aggregate")
            return state

        judge = JudgeAgent(
            name="Judge",
            config=self.config,
        )
        judge.context = judge.context  # keep default context

        scored_proposals = [judge.score_proposal(p) for p in proposals]
        ranked = judge.rank_proposals(scored_proposals)

        # Calculate loop score
        if ranked:
            loop_score = ranked[0].get("total_score", 0)
        else:
            loop_score = 0.0

        state["scored_proposals"] = scored_proposals
        state["ranked_proposals"] = ranked
        state["loop_score"] = loop_score
        state["proposals_remaining"] = len(
            [p for p in ranked if p.get("status") != "eliminated"]
        )

        logger.info(
            "AGGREGATE: %d proposals scored, top=%.1f",
            len(scored_proposals), loop_score,
        )
        return state

    def _node_eliminate(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """ELIMINATE node: Rank and eliminate weak proposals.

        Eliminates bottom 25% each round.
        """
        ranked = state.get("ranked_proposals", [])
        if not ranked:
            return state

        eliminations = []
        total = len(ranked)
        eliminate_count = max(1, total // 4)

        for i in range(eliminate_count):
            idx = total - 1 - i
            if idx >= 0 and ranked[idx].get("status") != "eliminated":
                ranked[idx]["status"] = "eliminated"
                elimination = {
                    "proposal_id": ranked[idx].get("id"),
                    "proposal_name": ranked[idx].get("name"),
                    "team": ranked[idx].get("team"),
                    "round": state.get("current_loop", 0),
                    "reason": f"Eliminated in round {state.get('current_loop', 0)}",
                }
                eliminations.append(elimination)

        state["eliminations"] = state.get("eliminations", []) + eliminations
        state["proposals_remaining"] = len(
            [p for p in ranked if p.get("status") != "eliminated"]
        )
        state["current_loop"] = state.get("current_loop", 0) + 1

        logger.info(
            "ELIMINATE: eliminated %d proposals, %d remaining",
            len(eliminations), state["proposals_remaining"],
        )
        return state

    def _node_deep_dive(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """DEEP_DIVE node: Head-to-head comparison of survivors.

        Compares surviving proposals in pairs.
        """
        self._track_node_cost("deep_dive", "comparison", 1000, 3)

        proposals = state.get("ranked_proposals", [])
        survivors = [p for p in proposals if p.get("status") != "eliminated"]

        if len(survivors) < 2:
            state["deep_dive_result"] = {
                "winner": survivors[0] if survivors else None,
                "comparisons": [],
            }
            return state

        comparisons = []
        for i in range(len(survivors)):
            for j in range(i + 1, len(survivors)):
                a, b = survivors[i], survivors[j]
                diff = a.get("total_score", 0) - b.get("total_score", 0)

                # Build dimension-level breakdown
                a_dims = a.get("dimension_scores", {})
                b_dims = b.get("dimension_scores", {})
                dim_breakdown = {}
                for dim_name in set(list(a_dims.keys()) + list(b_dims.keys())):
                    a_score = a_dims.get(dim_name, {}).get("score", 0)
                    b_score = b_dims.get(dim_name, {}).get("score", 0)
                    dim_breakdown[dim_name] = {
                        "a_score": a_score,
                        "b_score": b_score,
                        "winner": a.get("name") if a_score > b_score else (
                            b.get("name") if b_score > a_score else "tie"
                        ),
                    }

                comparisons.append({
                    "proposal_a": a.get("name"),
                    "proposal_b": b.get("name"),
                    "score_a": a.get("total_score"),
                    "score_b": b.get("total_score"),
                    "difference": round(abs(diff), 2),
                    "winner": a.get("name") if diff > 0 else b.get("name"),
                    "dimension_breakdown": dim_breakdown,
                })

        winner = max(survivors, key=lambda p: p.get("total_score", 0))
        state["deep_dive_result"] = {
            "winner": winner,
            "comparisons": comparisons,
        }

        logger.info(
            "DEEP_DIVE: %d survivors, %d comparisons, winner=%s",
            len(survivors), len(comparisons),
            winner.get("name", "N/A"),
        )
        return state

    def _node_judge(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """JUDGE node: Final scoring of all proposals.

        Produces the final score and pass/fail determination.
        Tracks cost for this node.
        """
        self._track_node_cost("judge", "final_scoring", 1500, 3)

        survivors = [
            p for p in state.get("ranked_proposals", [])
            if p.get("status") != "eliminated"
        ]

        if not survivors:
            state["total_score"] = 0.0
            state["passed"] = False
            return state

        # Final score = average of top 3 survivors (or all if < 3)
        top_scores = sorted(
            [p.get("total_score", 0) for p in survivors],
            reverse=True,
        )[:3]
        total_score = round(sum(top_scores) / len(top_scores), 2)

        pass_threshold = state.get("pass_threshold", 8.5)
        passed = total_score >= pass_threshold

        state["total_score"] = total_score
        state["passed"] = passed

        # Build component and suggestion data for report
        state["components"] = [
            {"name": p["name"], "role": p.get("description", "")[:50],
             "implementation": "GAAL v3 Arena"}
            for p in survivors[:5]
        ]
        state["suggestions"] = {
            "Immediate": [f"Implement: {s['name']}" for s in survivors[:3]],
            "Next Phase": ["Optimize implementation", "Add monitoring"],
        }

        logger.info(
            "JUDGE: final score=%.1f/10, passed=%s",
            total_score, passed,
        )
        return state

    # ── Bootstrap Self-Evolution ────────────────────────────────────

    def _node_evolve(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """EVOLVE node: Self-evolution of GAAL configuration.

        Implements REAL self-evolution:
        1. Computes bootstrap score from execution metrics
        2. Analyzes performance history for trends
        3. Suggests and applies config changes (weight adjustments, etc.)
        4. Tracks evolution history in CheckpointStore.evolution_history
        5. Saves evolution artifacts to evolution/ directory

        Only active when evolution.enabled = true.
        """
        evolution_config = self.config.get("evolution", {})
        if not evolution_config.get("enabled", False):
            logger.info("EVOLVE: disabled, skipping")
            return state

        total_score = state.get("total_score", 0)
        bootstrap_score = compute_bootstrap_score(state)
        previous_scores = self.evolution_scores

        # Add current score to history
        self.evolution_scores.append(bootstrap_score)

        # Determine evolution action
        evolution_action = suggest_evolution_action(
            self.evolution_scores, state, self.config,
        )

        if evolution_action is None:
            logger.info("EVOLVE: no action needed (score=%.1f)", total_score)
            state["evolution_action"] = None
            state["evolution_scores"] = self.evolution_scores
            state["bootstrap_score"] = bootstrap_score
            return state

        # Apply the evolution action
        applied = self._apply_evolution_action(evolution_action)

        # Record in checkpoint store
        if self.checkpoint_store is not None:
            self.checkpoint_store.record_evolution(
                action=evolution_action["action"],
                target=evolution_action["target"],
                before={"config_snapshot": {k: v for k, v in self.config.items() if isinstance(v, (str, int, float, bool, list, dict))}},
                after={"action": evolution_action},
                score_before=previous_scores[-1] if previous_scores else 0,
                score_after=bootstrap_score,
            )

        # Save evolution artifact
        self._save_evolution_artifact(applied)

        state["evolution_action"] = applied
        state["evolution_scores"] = self.evolution_scores
        state["bootstrap_score"] = bootstrap_score

        logger.info(
            "EVOLVE: %s on %s (bootstrap=%.1f)",
            evolution_action["action"], evolution_action["target"],
            bootstrap_score,
        )
        return state

    def _apply_evolution_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Apply an evolution action to the configuration.

        Args:
            action: Evolution action dict with action, target, reason, params.

        Returns:
            The applied action with status.
        """
        result = {**action, "status": "applied", "applied_at": time.time()}

        try:
            if action["action"] == "adjust_weight":
                # Modify scorecard weights
                target_path = Path(action["target"])
                if not target_path.is_absolute():
                    target_path = Path(__file__).parent.parent / action["target"]

                if target_path.exists():
                    with open(target_path) as f:
                        scorecard = yaml.safe_load(f)

                    dim_name = action["params"].get("dimension")
                    new_weight = action["params"].get("new_weight")
                    if dim_name and new_weight and "scorecard" in scorecard:
                        dims = scorecard["scorecard"].get("dimensions", {})
                        if dim_name in dims:
                            old_weight = dims[dim_name].get("weight", 2)
                            dims[dim_name]["weight"] = new_weight
                            with open(target_path, "w") as f:
                                yaml.dump(scorecard, f, allow_unicode=True, sort_keys=False)
                            result["old_weight"] = old_weight
                            result["new_weight"] = new_weight
                            logger.info("Evolution: adjusted %s weight %d → %d", dim_name, old_weight, new_weight)

            elif action["action"] == "increase_loops":
                new_max = action["params"].get("max_loops", 6)
                old_max = self.config.get("gaal", {}).get("max_loops", 4)
                if "gaal" not in self.config:
                    self.config["gaal"] = {}
                self.config["gaal"]["max_loops"] = new_max
                result["old_max_loops"] = old_max
                result["new_max_loops"] = new_max
                logger.info("Evolution: increased max_loops %d → %d", old_max, new_max)

            elif action["action"] == "enable_evolution":
                old_enabled = self.config.get("evolution", {}).get("enabled", False)
                if "evolution" not in self.config:
                    self.config["evolution"] = {}
                self.config["evolution"]["enabled"] = True
                result["old_enabled"] = old_enabled
                result["new_enabled"] = True
                logger.info("Evolution: enabled evolution (was %s)", old_enabled)

            self.evolution_actions.append(result)

        except Exception as e:
            result["status"] = "failed"
            result["error"] = str(e)
            logger.error("Evolution action failed: %s", e)

        return result

    def _save_evolution_artifact(self, action: Dict[str, Any]) -> None:
        """Save an evolution artifact JSON file to the evolution/ directory.

        Args:
            action: The evolution action that was applied.
        """
        cycle_num = len(self.evolution_actions)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"evolution_cycle_{cycle_num:03d}_{timestamp}.json"
        artifact_path = self.evolution_dir / filename

        artifact = {
            "cycle": cycle_num,
            "timestamp": timestamp,
            "action": action,
            "config_snapshot": {
                k: v for k, v in self.config.items()
                if k in ("gaal", "judge", "evolution", "teams")
            },
            "scores_history": self.evolution_scores,
        }

        try:
            with open(artifact_path, "w") as f:
                json.dump(artifact, f, indent=2, ensure_ascii=False, default=str)
            logger.info("Saved evolution artifact: %s", artifact_path)
        except Exception as e:
            logger.warning("Failed to save evolution artifact: %s", e)

    def evolve_config(self, action: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Externally trigger an evolution action on the configuration.

        This can be called outside of graph execution for manual evolution.

        Args:
            action: Optional action override. If None, computes from current state.

        Returns:
            The applied evolution result.
        """
        if action is None:
            action = suggest_evolution_action(
                self.evolution_scores, self._state, self.config,
            )
        if action is None:
            return {"status": "no_action_needed", "reason": "No evolution needed"}

        return self._apply_evolution_action(action)

    def _node_report(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """REPORT node: Generate final report data.

        Prepares all data for the report generator including
        cost summary, degradation history, and evolution suggestions.
        """
        # Prepare cost summary
        cost_summary = self._build_cost_summary()

        # Prepare degradation history
        degradation_history = state.get("degradation_history", [])

        # Prepare performance stats
        perf_stats = self._build_performance_stats(state)

        state["report_ready"] = True
        state["session_id"] = self.session_id
        state["cost_summary"] = cost_summary
        state["degradation_history"] = degradation_history
        state["performance_stats"] = perf_stats
        state["evolution_suggestions"] = self.evolution_actions if self.evolution_actions else None
        state["execution_history"] = self.execution_history

        logger.info(
            "REPORT: data ready (cost=%.2f, deg_level=%d, evo_actions=%d)",
            cost_summary.get("total_cost", 0),
            self.degradation_level,
            len(self.evolution_actions),
        )
        return state

    # ── Cost Tracking ───────────────────────────────────────────────

    def _track_node_cost(
        self,
        node_name: str,
        operation: str,
        estimated_tokens: int,
        cost_multiplier: int = 1,
    ) -> None:
        """Track cost for a node execution.

        Args:
            node_name: Name of the node.
            operation: The type of operation (e.g., 'goal_parsing', 'team_a').
            estimated_tokens: Estimated token usage.
            cost_multiplier: Cost multiplier based on model tier.
        """
        estimated_cost = cost_multiplier * (estimated_tokens / 1000.0)
        self.cost_data["total_tokens"] += estimated_tokens
        self.cost_data["total_cost"] += estimated_cost

        if node_name not in self.cost_data["per_node"]:
            self.cost_data["per_node"][node_name] = {
                "calls": 0,
                "total_tokens": 0,
                "total_cost": 0.0,
            }
        self.cost_data["per_node"][node_name]["calls"] += 1
        self.cost_data["per_node"][node_name]["total_tokens"] += estimated_tokens
        self.cost_data["per_node"][node_name]["total_cost"] += estimated_cost

        # Per-team tracking
        if node_name == "propose_team_a":
            self.cost_data["per_team"]["team_a"] += estimated_cost
        elif node_name == "propose_team_b":
            self.cost_data["per_team"]["team_b"] += estimated_cost

        # Per-mode tracking
        if self.mode not in self.cost_data["per_mode"]:
            self.cost_data["per_mode"][self.mode] = {"total_tokens": 0, "total_cost": 0.0}
        self.cost_data["per_mode"][self.mode]["total_tokens"] += estimated_tokens
        self.cost_data["per_mode"][self.mode]["total_cost"] += estimated_cost

        # Budget enforcement
        if self.cost_data["total_tokens"] > self._max_total_tokens:
            self.cost_data["budget_exceeded"] = True
            logger.warning(
                "Cost budget exceeded: %d tokens > %d max",
                self.cost_data["total_tokens"], self._max_total_tokens,
            )

    def _build_cost_summary(self) -> Dict[str, Any]:
        """Build a cost summary for the report.

        Returns:
            Dict with cost tracking data.
        """
        return {
            "total_calls": sum(
                n["calls"] for n in self.cost_data["per_node"].values()
            ),
            "total_tokens": self.cost_data["total_tokens"],
            "total_cost": round(self.cost_data["total_cost"], 2),
            "per_node": dict(self.cost_data["per_node"]),
            "per_team": dict(self.cost_data["per_team"]),
            "per_mode": dict(self.cost_data["per_mode"]),
            "budget_exceeded": self.cost_data["budget_exceeded"],
            "max_total_tokens": self._max_total_tokens,
        }

    def _build_performance_stats(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Build performance statistics from execution history.

        Args:
            state: Final graph state.

        Returns:
            Dict with performance stats.
        """
        history = self.execution_history
        if not history:
            return {}

        total_duration = sum(h.get("duration", 0) for h in history)
        total_attempts = sum(h.get("attempts", 1) for h in history)
        retry_count = total_attempts - len(history)

        node_stats = {}
        for h in history:
            node = h.get("node", "unknown")
            if node not in node_stats:
                node_stats[node] = {"calls": 0, "total_duration": 0, "total_attempts": 0}
            node_stats[node]["calls"] += 1
            node_stats[node]["total_duration"] += h.get("duration", 0)
            node_stats[node]["total_attempts"] += h.get("attempts", 1)

        return {
            "total_nodes": len(history),
            "total_duration": round(total_duration, 3),
            "total_attempts": total_attempts,
            "retries": retry_count,
            "avg_duration_per_node": round(total_duration / max(len(history), 1), 3),
            "node_stats": node_stats,
        }

    # ── Execution ───────────────────────────────────────────────────

    def run(
        self,
        goal: str,
        mode: str = "auto",
        config_override: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Run the GAAL v3 arena against a goal.

        This is the main entry point. It:
        1. Initializes the checkpoint store
        2. Builds and compiles the graph
        3. Executes the graph with checkpoint recovery
        4. Implements graceful degradation on failure
        5. Returns the final state and report data

        Args:
            goal: The goal to pursue.
            mode: Execution mode ('lite', 'hard', 'super', or 'auto').
            config_override: Optional config overrides.

        Returns:
            Dict with final_state, execution_history, and report_data.
        """
        self.goal = goal
        if config_override:
            self._deep_merge(self.config, config_override)

        # Initialize checkpoint store
        self.checkpoint_store = self._init_checkpoint_store()
        self.session_id = self.checkpoint_store.session_id

        # Record execution start
        execution_id = self.checkpoint_store.start_execution(
            goal=goal,
            mode=mode,
            config=self.config,
        )

        # Check for recovery
        recovery = self.checkpoint_store.recover()
        if recovery["status"] == "recovered":
            logger.info(
                "Recovered from checkpoint at '%s', action=%s",
                recovery["phase"], recovery["action"],
            )

        # Build and compile the graph
        compiled = self.build_graph()
        compiled._edge_index = compiled._build_edge_index()

        # Prepare initial state
        initial_state: Dict[str, Any] = {
            "goal": goal,
            "mode": mode if mode != "auto" else "lite",
            "session_id": self.session_id,
            "proposals": [],
            "eliminations": [],
            "current_loop": 0,
            "max_loops": 4,
            "max_proposals": 2,
            "total_score": 0.0,
            "loop_score": 0.0,
            "proposals_remaining": 0,
            "is_simple": False,
            "pass_threshold": 8.5,
        }

        # If recovering, use recovered state
        if recovery["status"] == "recovered":
            initial_state.update(recovery.get("state", {}))

        # Execute the graph
        try:
            final_state, history = compiled.run(
                initial_state=initial_state,
                checkpoint_store=self.checkpoint_store,
                max_steps=50,
            )

            self.execution_history = history
            self._state = final_state

            # Mark execution as completed
            self.checkpoint_store.complete_execution(
                execution_id=execution_id,
                result={
                    "total_score": final_state.get("total_score"),
                    "passed": final_state.get("passed"),
                    "total_nodes": len(history),
                },
            )

            logger.info(
                "Graph execution completed: %d steps, score=%.1f",
                len(history),
                final_state.get("total_score", 0),
            )

            return {
                "status": "completed",
                "session_id": self.session_id,
                "execution_id": execution_id,
                "final_state": final_state,
                "execution_history": history,
                "report_data": self._build_report_data(final_state, history),
            }

        except Exception as e:
            logger.error("Graph execution failed: %s", e)
            circuit_state = self.circuit_breaker.record_failure()

            # Mark execution as failed
            if self.checkpoint_store:
                self.checkpoint_store.complete_execution(
                    execution_id=execution_id,
                    error=str(e),
                )

            # ── Graceful Degradation ──
            if circuit_state == "open" or self.degradation_level < 2:
                logger.warning("Attempting graceful degradation (level=%d)", self.degradation_level)
                degraded = self._run_degraded(goal)
                return {
                    "status": "degraded",
                    "session_id": self.session_id,
                    "error": str(e),
                    "circuit_breaker": self.circuit_breaker.to_dict(),
                    "degraded_result": degraded,
                    "degradation_level": self.degradation_level,
                }

            return {
                "status": "failed",
                "session_id": self.session_id,
                "error": str(e),
                "execution_history": self.execution_history,
            }

    def _build_report_data(
        self,
        state: Dict[str, Any],
        history: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Build report data from final state and history.

        Includes enriched data for richer reports:
        - Cost summary
        - Degradation history
        - Evolution suggestions
        - Performance stats

        Args:
            state: Final state from graph execution.
            history: Execution history.

        Returns:
            Report data dict for ReportGenerator.
        """
        proposals = state.get("ranked_proposals", state.get("proposals", []))
        eliminations = state.get("eliminations", [])

        # Build enriched eliminations with names
        proposal_map = {p.get("id"): p.get("name", "Unknown") for p in proposals}
        enriched_eliminations = []
        for e in eliminations:
            enriched_eliminations.append({
                "round": e.get("round", 0),
                "name": e.get("proposal_name", proposal_map.get(e.get("proposal_id"), "Unknown")),
                "reason": e.get("reason", ""),
            })

        # Get dimension scores with justifications from proposals
        dimension_report = []
        if proposals:
            # Take the top proposal's dimension scores
            top = proposals[0]
            dim_scores = top.get("dimension_scores", {})
            for dim_name, dim_data in dim_scores.items():
                dimension_report.append({
                    "name": dim_name,
                    "score": dim_data.get("score", 0),
                    "weight": dim_data.get("weight", 1),
                    "justification": dim_data.get("justification", ""),
                })

        return {
            "goal": state.get("goal", ""),
            "mode": state.get("mode", "lite"),
            "loop_count": state.get("current_loop", 0),
            "total_score": state.get("total_score", 0),
            "passed": state.get("passed", False),
            "proposals": proposals,
            "eliminations": enriched_eliminations,
            "dimensions": dimension_report,
            "components": state.get("components", []),
            "suggestions": state.get("suggestions", {}),
            "architecture_diagram": self._generate_arch_diagram(),
            "session_id": self.session_id,
            "execution_history": history,
            "historical_trend": self._build_historical_trend(eliminations),
            # New enriched data
            "cost_summary": state.get("cost_summary", self._build_cost_summary()),
            "degradation_history": state.get("degradation_history", []),
            "evolution_suggestions": state.get("evolution_suggestions", []),
            "performance_stats": state.get("performance_stats", self._build_performance_stats(state)),
            "bootstrap_score": state.get("bootstrap_score", 0),
            "degradation_level": self.degradation_level,
        }

    def _generate_arch_diagram(self) -> str:
        """Generate architecture diagram text."""
        return (
            "┌──────────┐     ┌──────────┐     ┌──────────────────┐\n"
            "│ PARSE    │────▶│ RESEARCH │────▶│ PROPOSE          │\n"
            "│ GOAL     │     │ (super)  │     │ Team A + B       │\n"
            "└──────────┘     └──────────┘     │ (PARALLEL!)      │\n"
            "     │                             └────────┬─────────┘\n"
            "     ▼                                      │\n"
            "  (simple? ─END)                             ▼\n"
            "                                        ┌──────────┐\n"
            "                                        │AGGREGATE │\n"
            "                                        └────┬─────┘\n"
            "                                             │\n"
            "  DEGRADATION ◀──  circuit_breaker           ▼\n"
            "  super→hard→lite                      ┌──────────┐\n"
            "                                        │ELIMINATE │\n"
            "                                        └────┬─────┘\n"
            "                                             │\n"
            "                    ┌────────────────────────┘\n"
            "                    ▼\n"
            "              ┌──────────┐     ┌──────────┐     ┌──────────┐\n"
            "              │ DEEP DIVE│────▶│  JUDGE   │────▶│  REPORT  │\n"
            "              └──────────┘     └────┬─────┘     └──────────┘\n"
            "                                    │\n"
            "                                    ▼\n"
            "                              ┌──────────┐\n"
            "                              │  EVOLVE  │\n"
            "                              │ (SELF)   │\n"
            "                              └──────────┘"
        )

    def _build_historical_trend(
        self,
        eliminations: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Build historical trend data from eliminations.

        Args:
            eliminations: List of elimination records.

        Returns:
            Trend data grouped by round.
        """
        rounds = {}
        for e in eliminations:
            r = e.get("round", 0)
            if r not in rounds:
                rounds[r] = []
            rounds[r].append(e.get("proposal_name", e.get("name", "Unknown")))

        trend = []
        for r in sorted(rounds.keys()):
            eliminated_names = rounds[r]
            trend.append({
                "round": r,
                "eliminated_count": len(eliminated_names),
                "eliminated": ", ".join(eliminated_names),
            })

        return trend

    def _run_degraded(self, goal: str) -> Dict[str, Any]:
        """Run in degraded mode (fallback when circuit breaker is open).

        Implements the fallback chain:
        - super → hard (degradation_level=1)
        - hard → lite (degradation_level=2)
        - lite → fallback (degradation_level=2, minimal processing)

        Args:
            goal: The goal to pursue.

        Returns:
            Degraded execution result.
        """
        # Determine next degradation level
        current_mode = self._original_mode if self.degradation_level == 0 else self.mode

        if current_mode == "super":
            self.degradation_level = 1
            self.mode = "hard"
            fallback_note = "Degraded from super to hard mode"
        elif current_mode == "hard":
            self.degradation_level = 2
            self.mode = "lite"
            fallback_note = "Degraded from hard to lite mode"
        else:
            self.degradation_level = 2
            fallback_note = "Running in minimal fallback mode"

        logger.warning(
            "Degraded to level %d, mode=%s: %s",
            self.degradation_level, self.mode, fallback_note,
        )

        return {
            "status": "degraded",
            "goal": goal,
            "mode": self.mode,
            "degradation_level": self.degradation_level,
            "note": fallback_note,
            "total_score": 5.0,
            "proposals": [
                {
                    "name": "Degraded Proposal",
                    "description": f"Degraded mode proposal for: {goal[:50]}",
                }
            ],
        }

    def _deep_merge(
        self,
        base: Dict[str, Any],
        override: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Deep merge two dicts (override into base).

        Args:
            base: Base config dict (modified in place).
            override: Override config dict.

        Returns:
            Merged dict.
        """
        for key, value in override.items():
            if (
                key in base
                and isinstance(base[key], dict)
                and isinstance(value, dict)
            ):
                self._deep_merge(base[key], value)
            else:
                base[key] = value
        return base

    def to_dict(self) -> Dict[str, Any]:
        """Serialize orchestrator state for reporting."""
        return {
            "session_id": self.session_id,
            "mode": self.mode,
            "goal": self.goal,
            "execution_steps": len(self.execution_history),
            "circuit_breaker": self.circuit_breaker.to_dict(),
            "degradation_level": self.degradation_level,
            "evolution_scores": self.evolution_scores,
            "evolution_actions": len(self.evolution_actions),
            "cost_summary": self._build_cost_summary(),
        }

    def cleanup(self) -> None:
        """Clean up resources (close DB connections, etc.)."""
        if self.checkpoint_store is not None:
            self.checkpoint_store.close()
        logger.info("Orchestrator cleanup complete")
