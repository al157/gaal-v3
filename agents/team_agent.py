"""
Team agent for GAAL v3 — generates proposals for one team in the arena.

TeamAgent represents either Team A or Team B in the competitive arena.
Each team uses a different model tier for heterogeneity:
- Team A: Higher-tier model (pro/ultra) — quality-focused proposals
- Team B: Lower-tier model (flash/pro) — quick, diverse proposals

All LLM calls happen via delegate_task sub-sessions for zero-leak
isolation. The team agent only generates structured summaries.
"""
from __future__ import annotations
import json
import logging
import time
import uuid
from typing import Any, Dict, List, Optional

from .base import BaseAgent, AgentCapability, AgentContext

logger = logging.getLogger(__name__)


class TeamAgent(BaseAgent):
    """Agent representing a single team in the GAAL arena.

    Generates proposals for its assigned team using delegate_task
    sub-sessions. Each proposal targets a different aspect of the
    goal, ensuring diversity.

    The team follows the OpenAI Swarm handoff pattern — it can send
    proposals to the orchestrator via the handoff queue.

    Attributes:
        team_name: Name of this team (e.g., 'Team Alpha').
        team_id: Short ID ('team_a' or 'team_b').
        model_tier: Model tier assigned to this team.
        proposals_generated: Count of proposals generated.
    """

    def __init__(
        self,
        name: str = "TeamAgent",
        context: Optional[AgentContext] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(name=name, context=context, config=config)
        self.capabilities = [
            AgentCapability(
                name="proposal_generation",
                description="根据目标生成设计方案",
                complexity="moderate",
            ),
            AgentCapability(
                name="diverse_thinking",
                description="多角度思考，生成差异化方案",
                complexity="moderate",
            ),
        ]

        # Team identity
        self.team_name = ""
        self.team_id = ""
        self.model_tier = ""
        self.proposals_generated: List[Dict[str, Any]] = []

    def configure(self, team_id: str, team_config: Dict[str, Any]) -> None:
        """Configure this team agent with identity and model settings.

        Args:
            team_id: 'team_a' or 'team_b'.
            team_config: Team configuration dict from YAML config.
        """
        self.team_id = team_id
        self.team_name = team_config.get("name", f"Team {team_id}")
        self.model_tier = team_config.get("tier", "flash")
        logger.info(
            "Configured %s: %s (tier=%s)",
            team_id, self.team_name, self.model_tier,
        )

    def generate_proposal_instructions(self) -> Dict[str, Any]:
        """Generate the delegate_task context for a proposal sub-session.

        Returns:
            Dict with team info, goal, mode, and generation parameters.
        """
        return {
            "agent_role": "team_proposal_generator",
            "team_name": self.team_name,
            "team_id": self.team_id,
            "model_tier": self.model_tier,
            "goal": self.context.goal,
            "mode": self.context.mode,
            "round": self.context.round_num,
            "existing_proposals_count": len(self.proposals_generated),
            "instructions": (
                f"You are {self.team_name}, a proposal-generating AI agent "
                f"using a {self.model_tier}-tier model. "
                f"Generate ONE high-quality proposal for the goal: "
                f"'{self.context.goal}'. "
                f"Each proposal should have a unique name, detailed description "
                f"covering approach, architecture, and key technologies."
            ),
        }

    def add_proposal(
        self,
        name: str,
        description: str,
        scores: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Register a newly generated proposal.

        Args:
            name: Short proposal name.
            description: Detailed proposal description.
            scores: Optional scoring data.

        Returns:
            The proposal dict with metadata.
        """
        proposal = {
            "id": str(uuid.uuid4())[:12],
            "team": self.team_name,
            "team_id": self.team_id,
            "index": len(self.proposals_generated),
            "name": name,
            "description": description,
            "scores": scores or {},
            "status": "active",
            "round": self.context.round_num,
        }
        self.proposals_generated.append(proposal)
        return proposal

    def execute(self) -> List[Dict[str, Any]]:
        """Execute the team agent's proposal generation.

        Returns the list of proposals generated. The actual LLM work
        happens in a delegate_task sub-session.

        Returns:
            List of proposal dicts.
        """
        self.start_timer()
        # In real execution, this would call delegate_task.
        # The local implementation creates structured proposal data
        # that the orchestrator can use.
        return self.proposals_generated

    def summarize(self) -> Dict[str, Any]:
        """Generate a clean summary for the parent session.

        Zero-leak: only proposal names and counts, no LLM internals.

        Returns:
            Clean summary dict.
        """
        return {
            "agent": self.name,
            "type": "TeamAgent",
            "team_name": self.team_name,
            "team_id": self.team_id,
            "model_tier": self.model_tier,
            "proposals_generated": len(self.proposals_generated),
            "proposal_names": [p["name"] for p in self.proposals_generated],
            "elapsed_seconds": round(self.elapsed_time, 2),
        }
