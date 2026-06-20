"""
Judge agent for GAAL v3 — evaluates and scores proposals.

The JudgeAgent implements the evaluation phase of the arena:
1. Scores each proposal against configurable dimensions
2. Ranks proposals by total score
3. Determines eliminations (bottom performers each round)
4. Provides head-to-head deep dive comparisons
5. Produces final scoring with pass/fail determination

All evaluation logic uses delegate_task sub-sessions for zero-leak
isolation. The judge only returns scores and rankings.
"""
from __future__ import annotations
import json
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

from .base import BaseAgent, AgentCapability, AgentContext

logger = logging.getLogger(__name__)


class JudgeAgent(BaseAgent):
    """Agent responsible for evaluating and scoring proposals.

    The Judge evaluates proposals across multiple dimensions,
    ranks them, eliminates weak ones, and produces final scores.

    Attributes:
        dimensions: Scoring dimensions (from config).
        pass_threshold: Minimum score to pass.
        scored_proposals: Proposals with their scores.
        scoring_history: Historical scoring data per round.
    """

    def __init__(
        self,
        name: str = "Judge",
        context: Optional[AgentContext] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(name=name, context=context, config=config)
        self.capabilities = [
            AgentCapability(
                name="proposal_scoring",
                description="按多个维度对方案评分",
                complexity="moderate",
            ),
            AgentCapability(
                name="ranking",
                description="对方案进行排名和淘汰",
                complexity="moderate",
            ),
            AgentCapability(
                name="deep_dive_comparison",
                description="头对头深度对比幸存方案",
                complexity="complex",
            ),
            AgentCapability(
                name="final_scoring",
                description="最终评分与通过/失败判定",
                complexity="moderate",
            ),
        ]

        # Judge configuration
        self.dimensions: List[Dict[str, Any]] = []
        self.pass_threshold: float = 8.5
        self._load_config()

        # Scoring state
        self.scored_proposals: List[Dict[str, Any]] = []
        self.scoring_history: List[Dict[str, Any]] = []
        self.final_scores: Dict[str, Any] = {}
        self.elimination_recommendations: List[Dict[str, Any]] = []

    def _load_config(self) -> None:
        """Load judge configuration from config."""
        judge_config = self.config.get("judge", {})
        self.dimensions = judge_config.get("dimensions", [])
        self.pass_threshold = judge_config.get("pass_threshold", 8.5)

    def score_proposal(
        self,
        proposal: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Score a single proposal across all dimensions.

        Each dimension gets a score of 0.0 to 2.0.
        Total score = sum(dimension_score * weight) / sum(weights) * 10

        Args:
            proposal: The proposal dict with name and description.

        Returns:
            Scored proposal dict with dimension scores and total.
        """
        if not self.dimensions:
            # Default dimensions if not configured
            dims = [
                {"name": "quality", "weight": 2, "description": "方案质量"},
                {"name": "feasibility", "weight": 2, "description": "可行性"},
            ]
        else:
            dims = self.dimensions

        dimension_scores = {}
        total_weighted = 0.0
        total_weight = 0.0

        for dim in dims:
            name = dim["name"]
            weight = dim.get("weight", 1)

            # Calculate score (in real execution via delegate_task)
            # Default: mid-range score with some variation based on proposal index
            score = self._calculate_score(
                proposal_name=proposal.get("name", ""),
                proposal_desc=proposal.get("description", ""),
                dimension_name=name,
            )
            dimension_scores[name] = {
                "score": score,
                "weight": weight,
                "note": f"评估完成: {name}",
            }
            total_weighted += score * weight
            total_weight += weight

        total_score = round(
            (total_weighted / max(total_weight, 1)) * 5.0,  # 0-2 -> 0-10
            2,
        )

        return {
            **proposal,
            "dimension_scores": dimension_scores,
            "total_score": total_score,
        }

    def _calculate_score(
        self,
        proposal_name: str,
        proposal_desc: str,
        dimension_name: str,
    ) -> float:
        """Calculate a dimension score using heuristics.

        In production, this would use delegate_task to an LLM.
        For now, returns a reasonable default.

        Args:
            proposal_name: Name of the proposal.
            proposal_desc: Description text.
            dimension_name: Dimension being scored.

        Returns:
            Score between 0.0 and 2.0.
        """
        # Heuristic: longer descriptions tend to score higher
        desc_len = len(proposal_desc or "")
        if desc_len > 500:
            base = 1.5
        elif desc_len > 200:
            base = 1.2
        else:
            base = 1.0

        # Small variation to differentiate proposals
        name_hash = sum(ord(c) for c in proposal_name) % 5
        variation = (name_hash - 2) * 0.1

        return round(max(0.0, min(2.0, base + variation)), 1)

    def rank_proposals(
        self,
        proposals: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Rank proposals by total score descending.

        Args:
            proposals: List of proposal dicts.

        Returns:
            Ranked proposals list (highest score first).
        """
        ranked = sorted(
            proposals,
            key=lambda p: p.get("total_score", 0),
            reverse=True,
        )
        for i, p in enumerate(ranked):
            p["rank"] = i + 1
        return ranked

    def determine_eliminations(
        self,
        ranked_proposals: List[Dict[str, Any]],
        proposals_per_team: int,
    ) -> List[Dict[str, Any]]:
        """Determine which proposals to eliminate.

        Eliminates the bottom 25% of proposals each round.

        Args:
            ranked_proposals: Proposals ranked by score.
            proposals_per_team: Number of proposals per team.

        Returns:
            List of elimination recommendations.
        """
        eliminations = []
        total = len(ranked_proposals)
        elimination_count = max(1, total // 4)

        for i in range(elimination_count):
            proposal = ranked_proposals[total - 1 - i]
            if proposal.get("status") != "eliminated":
                elimination = {
                    "proposal_id": proposal.get("id"),
                    "proposal_name": proposal.get("name"),
                    "team": proposal.get("team"),
                    "rank": proposal.get("rank", total),
                    "total_score": proposal.get("total_score", 0),
                    "reason": f"排名过低 (rank {proposal.get('rank', 'N/A')}/{total})",
                }
                eliminations.append(elimination)

        return eliminations

    def deep_dive_compare(
        self,
        proposals: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Perform head-to-head deep dive on survivors.

        Compares surviving proposals on detailed criteria.

        Args:
            proposals: Survivor proposals.

        Returns:
            Deep dive comparison results.
        """
        if len(proposals) < 2:
            return {
                "winner": proposals[0] if proposals else None,
                "comparisons": [],
            }

        comparisons = []
        for i in range(len(proposals)):
            for j in range(i + 1, len(proposals)):
                a, b = proposals[i], proposals[j]
                diff = a.get("total_score", 0) - b.get("total_score", 0)
                comparisons.append({
                    "proposal_a": a.get("name"),
                    "proposal_b": b.get("name"),
                    "score_a": a.get("total_score"),
                    "score_b": b.get("total_score"),
                    "difference": round(abs(diff), 2),
                    "winner": a.get("name") if diff > 0 else b.get("name"),
                })

        winner = max(proposals, key=lambda p: p.get("total_score", 0))
        return {"winner": winner, "comparisons": comparisons}

    def get_pass_fail(self, total_score: float) -> Dict[str, Any]:
        """Determine if the total score passes the threshold.

        Args:
            total_score: Final total score.

        Returns:
            Dict with passed (bool), threshold, and score.
        """
        return {
            "passed": total_score >= self.pass_threshold,
            "threshold": self.pass_threshold,
            "score": total_score,
        }

    def execute(self) -> Dict[str, Any]:
        """Execute the judge agent's evaluation logic.

        Returns scoring results, rankings, and pass/fail.

        Returns:
            Dict with scoring results.
        """
        self.start_timer()

        if not self.context.proposals:
            return {
                "status": "no_proposals",
                "scored": [],
                "ranked": [],
                "winner": None,
                "passed": False,
            }

        # Score all proposals
        self.scored_proposals = [
            self.score_proposal(p) for p in self.context.proposals
        ]

        # Rank them
        ranked = self.rank_proposals(self.scored_proposals)
        winner = ranked[0] if ranked else None

        return {
            "status": "completed",
            "scored": self.scored_proposals,
            "ranked": ranked,
            "winner": winner,
            "passed": winner.get("total_score", 0) >= self.pass_threshold
            if winner else False,
        }

    def summarize(self) -> Dict[str, Any]:
        """Generate a clean summary for the parent session.

        Zero-leak: only scores and rankings.

        Returns:
            Clean summary dict.
        """
        return {
            "agent": self.name,
            "type": "JudgeAgent",
            "proposals_scored": len(self.scored_proposals),
            "dimensions": [d["name"] for d in self.dimensions],
            "pass_threshold": self.pass_threshold,
            "top_proposal": (
                self.scored_proposals[0]["name"]
                if self.scored_proposals else None
            ),
            "top_score": (
                self.scored_proposals[0].get("total_score", 0)
                if self.scored_proposals else 0
            ),
            "eliminations_recommended": len(self.elimination_recommendations),
            "elapsed_seconds": round(self.elapsed_time, 2),
        }
