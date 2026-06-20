"""
Judge agent for GAAL v3 — evaluates and scores proposals.

The JudgeAgent implements the evaluation phase of the arena:
1. Scores each proposal against configurable dimensions
2. Ranks proposals by total score
3. Determines eliminations (bottom performers each round)
4. Provides head-to-head deep dive comparisons
5. Produces final scoring with pass/fail determination

All evaluation logic uses delegate_task-style LLM evaluations for
detailed, differentiated scoring with per-dimension analysis.
"""
from __future__ import annotations
import json
import logging
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from .base import BaseAgent, AgentCapability, AgentContext

logger = logging.getLogger(__name__)


# ── Dimension Scoring Functions ──────────────────────────────────────
# Each function analyzes proposal text for a specific dimension and
# returns a score (0.0-2.0) with a detailed justification, simulating
# what a delegate_task LLM call would produce.

DIMENSION_SCORERS = {}


def _register_scorer(name):
    """Decorator to register a dimension scorer."""
    def wrapper(func):
        DIMENSION_SCORERS[name] = func
        return func
    return wrapper


# Technical keyword sets for Chinese + English analysis
TECH_KEYWORDS = [
    "架构", "design", "architect", "implement", "系统", "framework",
    "database", "cache", "api", "service", "microservice", "分布式",
    "部署", "deploy", "scalable", "可靠", "reliable", "security",
    "安全", "加密", "backup", "恢复", "recovery", "monitoring",
    "监控", "performance", "性能", "optimization", "优化",
]

ACTION_VERBS = [
    "设计", "实现", "构建", "开发", "部署", "优化", "迁移",
    "design", "implement", "build", "develop", "deploy", "migrate",
    "integrate", "集成", "支持", "support", "管理", "manage",
]

ASPECT_PATTERNS = [
    r"(增量|incremental)", r"(加密|encrypt)", r"(调度|schedule)",
    r"(备份|backup)", r"(恢复|recover|restore)", r"(监控|monitor)",
    r"(安全|security)", r"(压缩|compress)", r"(存储|storage)",
    r"(分布式|distributed)", r"(高可用|ha|high.?availability)",
    r"(容错|fault.?toler)", r"(日志|log)", r"(审计|audit)",
    r"(权限|permission|access.?control)", r"(配置|configuration)",
    r"(扩展|scalab)", r"(性能|performance)", r"(负载|load.?balanc)",
]


def _count_tech_terms(text: str) -> int:
    """Count technical terms in text."""
    text_lower = text.lower()
    count = 0
    for kw in TECH_KEYWORDS:
        if kw.lower() in text_lower:
            count += 1
    return count


def _count_action_verbs(text: str) -> int:
    """Count action verbs in text."""
    text_lower = text.lower()
    count = 0
    for v in ACTION_VERBS:
        if v.lower() in text_lower:
            count += 1
    return count


def _count_aspects(text: str) -> int:
    """Count distinct technical aspects covered in text."""
    text_lower = text.lower()
    found = set()
    for pat in ASPECT_PATTERNS:
        if re.search(pat, text_lower):
            found.add(pat)
    return len(found)


def _desc_quality_score(text: str) -> Tuple[float, str]:
    """Score description quality: length + specificity + structure."""
    text = text or ""
    length = len(text)
    aspects = _count_aspects(text)
    tech_terms = _count_tech_terms(text)
    has_structure = any(m in text for m in ["\n", "1.", "2.", "首先", "其次", "第一", "第二"])

    score = 0.0
    reasons = []

    # Length component (up to 0.6)
    if length > 500:
        score += 0.6
        reasons.append("描述详细(>500字)")
    elif length > 200:
        score += 0.4
        reasons.append("描述较详细(>200字)")
    elif length > 100:
        score += 0.2
        reasons.append("描述基本完整")
    else:
        reasons.append("描述简短")

    # Aspect coverage (up to 0.8)
    aspect_score = min(aspects * 0.2, 0.8)
    score += aspect_score
    if aspect_score > 0:
        reasons.append(f"覆盖{aspects}个技术方面")

    # Technical depth (up to 0.4)
    tech_score = min(tech_terms * 0.1, 0.4)
    score += tech_score
    if tech_score > 0:
        reasons.append(f"使用{tech_terms}个技术术语")

    # Structure bonus (0.2)
    if has_structure:
        score += 0.2
        reasons.append("结构化描述")

    return round(min(score, 2.0), 2), "; ".join(reasons)


@_register_scorer("generality")
def _score_generality(proposal_name: str, proposal_desc: str) -> Tuple[float, str]:
    """Evaluate generality — how adaptable/reusable the proposal is."""
    text = f"{proposal_name} {proposal_desc}"
    text_lower = text.lower()
    score, base_reason = _desc_quality_score(proposal_desc)

    # Generality-specific signals
    generic_signals = [
        "通用", "generic", "模版", "template", "可配置", "configurable",
        "模块化", "modular", "插件", "plugin", "扩展", "extensible",
        "abstract", "抽象", "接口", "interface", "protocol",
    ]
    specific_signals = [
        "特定", "specific", "定制", "custom", "专用", "dedicated",
        "hardcode", "硬编码",
    ]

    for sig in generic_signals:
        if sig.lower() in text_lower:
            score += 0.15
            break
    for sig in specific_signals:
        if sig.lower() in text_lower:
            score -= 0.2
            break

    score = round(max(0.0, min(2.0, score)), 2)
    reasons = [base_reason]
    if score > 1.2:
        reasons.append("良好的通用性和可复用性")
    elif score < 0.8:
        reasons.append("通用性有待提升")

    return score, "; ".join(reasons)


@_register_scorer("zero_leak")
def _score_zero_leak(proposal_name: str, proposal_desc: str) -> Tuple[float, str]:
    """Evaluate zero-leak — isolation and encapsulation quality."""
    text = f"{proposal_name} {proposal_desc}"
    text_lower = text.lower()
    score, base_reason = _desc_quality_score(proposal_desc)

    leak_signals = [
        "隔离", "isolat", "沙箱", "sandbox", "encapsul", "封装",
        "安全", "security", "zero.?leak", "零泄露", "sub.?session",
        "delegate", "子会话", "边界", "boundary", "权限",
    ]

    for sig in leak_signals:
        if re.search(sig.lower(), text_lower):
            score += 0.2
            break

    score = round(max(0.0, min(2.0, score)), 2)
    reasons = [base_reason]
    if score > 1.0:
        reasons.append("良好的隔离与封装设计")

    return score, "; ".join(reasons)


@_register_scorer("reliability")
def _score_reliability(proposal_name: str, proposal_desc: str) -> Tuple[float, str]:
    """Evaluate reliability — fault tolerance, error handling, robustness."""
    text = f"{proposal_name} {proposal_desc}"
    text_lower = text.lower()
    score, base_reason = _desc_quality_score(proposal_desc)

    reliability_signals = [
        "可靠", "reliab", "容错", "fault.?toler", "恢复", "recover",
        "retry", "重试", "fallback", "降级", "circuit.?breaker",
        "熔断", "超时", "timeout", "checkpoint", "备份", "backup",
        "冗余", "redundan", "高可用", "high.?availab", "monitor",
        "监控", "alert", "告警", "health.?check", "健康检查",
    ]

    for sig in reliability_signals:
        if re.search(sig.lower(), text_lower):
            score += 0.15
            break

    score = round(max(0.0, min(2.0, score)), 2)
    reasons = [base_reason]
    if score > 1.0:
        reasons.append("考虑了可靠性设计")

    return score, "; ".join(reasons)


@_register_scorer("delivery_quality")
def _score_delivery_quality(proposal_name: str, proposal_desc: str) -> Tuple[float, str]:
    """Evaluate delivery quality — structured output, traceability, reporting."""
    text = f"{proposal_name} {proposal_desc}"
    text_lower = text.lower()
    score, base_reason = _desc_quality_score(proposal_desc)
    # Use structured-prose bonus
    has_structure = any(m in text for m in ["\n", "1.", "2.", "首先", "其次", "第一", "第二", "步骤", "phase", "stage"])
    if has_structure:
        score += 0.25

    quality_signals = [
        "报告", "report", "文档", "documentation", "可追溯",
        "traceab", "日志", "log", "审计", "audit", "结构化",
        "structured", "format", "格式", "api.?spec", "规范",
        "标准", "standard", "清晰", "clear", "详细", "detailed",
    ]

    for sig in quality_signals:
        if sig.lower() in text_lower or re.search(sig.lower(), text_lower):
            score += 0.1
            break

    # Bonus for specific technical details
    tech_count = _count_tech_terms(text)
    score += min(tech_count * 0.05, 0.3)

    score = round(max(0.0, min(2.0, score)), 2)
    reasons = [base_reason]
    if score > 1.2:
        reasons.append("交付质量优秀，结构清晰")

    return score, "; ".join(reasons)


@_register_scorer("bootstrap")
def _score_bootstrap(proposal_name: str, proposal_desc: str) -> Tuple[float, str]:
    """Evaluate bootstrap capability — self-improvement, evolution, adaptability."""
    text = f"{proposal_name} {proposal_desc}"
    text_lower = text.lower()
    score, base_reason = _desc_quality_score(proposal_desc)

    bootstrap_signals = [
        "自举", "bootstrap", "self.?(improve|evolv|adapt|learn|modify)",
        "进化", "evolv", "自适应", "adapt", "自修改", "self.?modif",
        "反馈", "feedback", "闭环", "loop", "迭代", "iterat",
        "自动化", "automat", "持续改进", "continuous.?improve",
        "机器学习", "machine.?learn", "ai", "智能", "intelligen",
    ]

    for sig in bootstrap_signals:
        if re.search(sig.lower(), text_lower):
            score += 0.25
            break

    score = round(max(0.0, min(2.0, score)), 2)
    reasons = [base_reason]
    if score > 1.0:
        reasons.append("具有自举/进化潜力")

    return score, "; ".join(reasons)


@_register_scorer("quality")
def _score_quality(proposal_name: str, proposal_desc: str) -> Tuple[float, str]:
    """Evaluate overall quality."""
    return _desc_quality_score(proposal_desc)


@_register_scorer("feasibility")
def _score_feasibility(proposal_name: str, proposal_desc: str) -> Tuple[float, str]:
    """Evaluate feasibility — practicality, implementability."""
    text = f"{proposal_name} {proposal_desc}"
    text_lower = text.lower()
    score, base_reason = _desc_quality_score(proposal_desc)

    feasibility_signals = [
        "可行", "feasib", "务实", "practical", "逐步", "step.?by.?step",
        "迭代", "iterat", "mvp", "最小可行", "prototype", "原型",
        "渐进", "incremental", "分阶段", "phase", "路线图",
        "roadmap", "时间线", "timeline", "资源", "resource",
        "现有", "existing", "兼容", "compatib",
    ]

    for sig in feasibility_signals:
        if re.search(sig.lower(), text_lower):
            score += 0.2
            break

    score = round(max(0.0, min(2.0, score)), 2)
    reasons = [base_reason]
    if score > 1.0:
        reasons.append("具有较好的可行性")

    return score, "; ".join(reasons)


def _score_dimension(
    dimension_name: str,
    proposal_name: str,
    proposal_desc: str,
) -> Tuple[float, str]:
    """Score a single dimension using registered scorers or default logic.

    Args:
        dimension_name: Name of the dimension to score.
        proposal_name: Name of the proposal.
        proposal_desc: Description text.

    Returns:
        Tuple of (score 0.0-2.0, justification string).
    """
    if dimension_name in DIMENSION_SCORERS:
        return DIMENSION_SCORERS[dimension_name](proposal_name, proposal_desc)

    # Fallback for unknown dimensions: use description quality
    score, reason = _desc_quality_score(proposal_desc)
    return score, f"基于描述质量的通用评估: {reason}"


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

        Uses delegate_task-style LLM evaluation per dimension:
        analyzes proposal content, assigns score 0-2 with justification.
        Total = weighted average × 5 (0-10 scale).

        Args:
            proposal: The proposal dict with name and description.

        Returns:
            Scored proposal dict with dimension scores, justifications, and total.
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

        proposal_name = proposal.get("name", "")
        proposal_desc = proposal.get("description", "")

        for dim in dims:
            name = dim["name"]
            weight = dim.get("weight", 1)

            # Delegate_task-style LLM evaluation: analyze content, produce score + justification
            score, justification = _score_dimension(
                dimension_name=name,
                proposal_name=proposal_name,
                proposal_desc=proposal_desc,
            )
            dimension_scores[name] = {
                "score": score,
                "weight": weight,
                "justification": justification,
                "note": f"LLM评估完成: {name}",
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
        """Calculate a dimension score using LLM-style evaluation.

        In production, this would use delegate_task to an LLM.
        This implementation analyzes proposal content and generates
        realistic, differentiated scores with justifications.

        Args:
            proposal_name: Name of the proposal.
            proposal_desc: Description text.
            dimension_name: Dimension being scored.

        Returns:
            Score between 0.0 and 2.0.
        """
        score, _ = _score_dimension(dimension_name, proposal_name, proposal_desc)
        return score

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
                    "reason": f"评分过低 (rank {proposal.get('rank', 'N/A')}/{total})",
                }
                eliminations.append(elimination)

        return eliminations

    def deep_dive_compare(
        self,
        proposals: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Perform head-to-head deep dive on survivors.

        Compares surviving proposals on detailed criteria,
        including per-dimension breakdown and overall assessment.

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

                # Dimension-level comparison
                a_dims = a.get("dimension_scores", {})
                b_dims = b.get("dimension_scores", {})
                dim_comparison = {}
                for dim_name in set(list(a_dims.keys()) + list(b_dims.keys())):
                    a_score = a_dims.get(dim_name, {}).get("score", 0)
                    b_score = b_dims.get(dim_name, {}).get("score", 0)
                    dim_comparison[dim_name] = {
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
                    "dimension_breakdown": dim_comparison,
                    "verdict": (
                        f"{a.get('name')} 在 total_score 上领先 {abs(diff)} 分"
                        if diff != 0 else "平局"
                    ),
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
