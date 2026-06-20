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
7. Self-evolution (bootstrap mode)

The orchestrator delegates all LLM work to sub-sessions for zero-leak
isolation. The main session receives only clean summaries.
"""
from __future__ import annotations
import json
import logging
import time
import uuid
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

        # Load config from file if path provided
        if config_path:
            self._load_config(config_path)

    def _load_config(self, config_path: str) -> None:
        """Load configuration from a YAML file.

        Args:
            config_path: Path to YAML config file.
        """
        try:
            import yaml
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

        # propose_team_a -> propose_team_b (sequential
        # (in real deployment they'd be parallel via delegate_task)
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
        """
        goal = state.get("goal", self.goal)
        orchestrator = OrchestratorAgent(
            name="Orchestrator",
            context=AgentContext(goal=goal, mode=self.mode),
            config=self.config,
        )

        parsed = orchestrator.parse_goal()
        mode = parsed["mode"]
        self.mode = mode

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
        }
        logger.info("PARSE_GOAL: mode=%s, is_simple=%s", mode, parsed["is_simple"])
        return result

    def _node_research(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """RESEARCH node: (super mode only) research the goal.

        In production, this would use delegate_task for agent-reach
        research. Currently generates structured research data.
        """
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

    def _node_propose_team_a(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """PROPOSE_TEAM_A node: Generate proposals from Team A.

        Uses TeamAgent for Team A (higher-tier model).
        """
        goal = state.get("goal", "")
        mode = state.get("mode", "lite")
        max_proposals = state.get("max_proposals", 2)
        current_loop = state.get("current_loop", 0)

        team_config = self.config.get("teams", {}).get("team_a", {})
        agent = TeamAgent(name="TeamAlpha", config=self.config)
        agent.configure("team_a", team_config)

        proposals = []
        for i in range(max_proposals):
            proposal = agent.add_proposal(
                name=f"Team A - 方案 {current_loop * max_proposals + i + 1}",
                description=(
                    f"来自 Team A 的第 {i+1} 个方案 (Loop {current_loop + 1}). "
                    f"目标: {goal[:50]}"
                ),
            )
            proposals.append(proposal)

        state["proposals"] = state.get("proposals", []) + proposals
        state["team_a_proposals"] = proposals
        logger.info(
            "PROPOSE_TEAM_A: generated %d proposals",
            len(proposals),
        )
        return state

    def _node_propose_team_b(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """PROPOSE_TEAM_B node: Generate proposals from Team B.

        Uses TeamAgent for Team B (lower-tier model for diversity).
        """
        goal = state.get("goal", "")
        max_proposals = state.get("max_proposals", 2)
        current_loop = state.get("current_loop", 0)

        team_config = self.config.get("teams", {}).get("team_b", {})
        agent = TeamAgent(name="TeamBeta", config=self.config)
        agent.configure("team_b", team_config)

        proposals = []
        for i in range(max_proposals):
            offset = len(state.get("proposals", []))
            proposal = agent.add_proposal(
                name=f"Team B - 方案 {offset + i + 1}",
                description=(
                    f"来自 Team B 的第 {i+1} 个方案 (Loop {current_loop + 1}). "
                    f"目标: {goal[:50]}"
                ),
            )
            proposals.append(proposal)

        state["proposals"] = state.get("proposals", []) + proposals
        state["team_b_proposals"] = proposals
        logger.info(
            "PROPOSE_TEAM_B: generated %d proposals",
            len(proposals),
        )
        return state

    def _node_aggregate(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """AGGREGATE node: Merge all proposals and apply scorecard.

        Scores all proposals using the JudgeAgent.
        """
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
                comparisons.append({
                    "proposal_a": a.get("name"),
                    "proposal_b": b.get("name"),
                    "score_a": a.get("total_score"),
                    "score_b": b.get("total_score"),
                    "difference": round(abs(diff), 2),
                    "winner": a.get("name") if diff > 0 else b.get("name"),
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
        """
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

    def _node_evolve(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """EVOLVE node: Self-evolution of GAAL configuration.

        Can modify config YAML files and scorecard weights.
        Only active when evolution.enabled = true.
        """
        evolution_config = self.config.get("evolution", {})
        if not evolution_config.get("enabled", False):
            logger.info("EVOLVE: disabled, skipping")
            return state

        total_score = state.get("total_score", 0)
        previous_scores = state.get("evolution_scores", [])

        evolution_action = None
        if total_score < 7.0:
            # Low score: adjust scorecard weights
            evolution_action = {
                "action": "adjust_weights",
                "target": "config/scorecard.yaml",
                "reason": f"Score {total_score} < 7.0, adjusting weights",
            }
        elif len(previous_scores) >= 2:
            # Check trend
            recent = previous_scores[-2:]
            if recent[1] < recent[0]:
                evolution_action = {
                    "action": "modify_config",
                    "target": "config/gaal_v3.yaml",
                    "reason": "Declining score trend, adjusting config",
                }

        if evolution_action:
            state["evolution_action"] = evolution_action
            state["evolution_scores"] = previous_scores + [total_score]
            logger.info(
                "EVOLVE: %s on %s",
                evolution_action["action"], evolution_action["target"],
            )
        else:
            state["evolution_scores"] = previous_scores + [total_score]
            logger.info("EVOLVE: no action needed")

        return state

    def _node_report(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """REPORT node: Generate final report data.

        Prepares all data for the report generator.
        """
        state["report_ready"] = True
        state["session_id"] = self.session_id
        logger.info("REPORT: report data ready")
        return state

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
        4. Returns the final state and report data

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

            # Graceful degradation: fall back to simpler mode
            if circuit_state == "open":
                logger.warning("Circuit breaker open, attempting graceful degradation")
                degraded = self._run_degraded(goal)
                return {
                    "status": "degraded",
                    "session_id": self.session_id,
                    "error": str(e),
                    "circuit_breaker": self.circuit_breaker.to_dict(),
                    "degraded_result": degraded,
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

        return {
            "goal": state.get("goal", ""),
            "mode": state.get("mode", "lite"),
            "loop_count": state.get("current_loop", 0),
            "total_score": state.get("total_score", 0),
            "passed": state.get("passed", False),
            "proposals": proposals,
            "eliminations": enriched_eliminations,
            "dimensions": state.get("dimensions", []),
            "components": state.get("components", []),
            "suggestions": state.get("suggestions", {}),
            "architecture_diagram": self._generate_arch_diagram(),
            "session_id": self.session_id,
            "execution_history": history,
            "historical_trend": self._build_historical_trend(eliminations),
        }

    def _generate_arch_diagram(self) -> str:
        """Generate architecture diagram text."""
        return (
            "┌──────────┐     ┌──────────┐     ┌──────────┐\n"
            "│ PARSE    │────▶│ RESEARCH │────▶│ PROPOSE  │\n"
            "│ GOAL     │     │ (super)  │     │ Team A+B │\n"
            "└──────────┘     └──────────┘     └─────┬────┘\n"
            "     │                                   │\n"
            "     ▼                                   ▼\n"
            "  (simple? ─END)                    ┌──────────┐\n"
            "                                     │AGGREGATE │\n"
            "                                     └────┬─────┘\n"
            "                                          │\n"
            "                                          ▼\n"
            "┌──────────┐     ┌──────────┐     ┌──────────┐\n"
            "│  REPORT  │◀────│  JUDGE   │◀────│ DEEP DIVE│\n"
            "└──────────┘     └────┬─────┘     └──────────┘\n"
            "                      │\n"
            "                      ▼\n"
            "                 ┌──────────┐\n"
            "                 │  EVOLVE  │\n"
            "                 └──────────┘"
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

        Uses minimal processing: lite mode with single pass.

        Args:
            goal: The goal to pursue.

        Returns:
            Degraded execution result.
        """
        logger.info("Running degraded mode for: %s", goal[:50])
        return {
            "status": "degraded",
            "goal": goal,
            "mode": "lite",
            "note": "Circuit breaker was open, ran in degraded lite mode",
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
        }

    def cleanup(self) -> None:
        """Clean up resources (close DB connections, etc.)."""
        if self.checkpoint_store is not None:
            self.checkpoint_store.close()
        logger.info("Orchestrator cleanup complete")
