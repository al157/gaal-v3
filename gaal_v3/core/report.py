"""
Jinja2-based report generator for GAAL v3.

Generates structured arena reports with:
[T] Task Overview
[M] All Proposals
[V] Elimination History
[I] Final Deliverable / Architecture
[A] Achievement Scores
[W] Next Steps / Suggestions
[H] Historical Trend (optional)
[C] Cost Summary
[E] Evolution Suggestions
[P] Performance Stats

Uses the report_v3.md.j2 Jinja2 template.
"""
from __future__ import annotations
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    from jinja2 import Environment, FileSystemLoader, TemplateNotFound
except ImportError:
    Environment = None  # type: ignore
    FileSystemLoader = None  # type: ignore
    TemplateNotFound = type("TemplateNotFound", (Exception,), {})


class ReportGenerator:
    """Generates structured GAAL v3 arena reports using Jinja2 templates.

    Produces reports in the [T][M][V][I][A][W] format with optional
    [H] historical trend data, [C] cost summary, [E] evolution suggestions,
    and [P] performance stats.

    Attributes:
        template_dir: Directory containing the Jinja2 template.
        template_name: Name of the template file.
        config: GAAL configuration dict (for metadata).
    """

    def __init__(
        self,
        template_dir: Optional[str] = None,
        template_name: str = "report_v3.md.j2",
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.template_dir = Path(
            template_dir or Path(__file__).parent.parent / "templates"
        )
        self.template_name = template_name
        self.config = config or {}
        self.env: Optional[Environment] = None
        self._init_jinja()

    def _init_jinja(self) -> None:
        """Initialize Jinja2 environment.

        Raises:
            ImportError: If Jinja2 is not installed.
        """
        if Environment is None or FileSystemLoader is None:
            raise ImportError(
                "Jinja2 is required for report generation. "
                "Install with: pip install jinja2"
            )
        self.env = Environment(
            loader=FileSystemLoader(str(self.template_dir)),
            autoescape=False,
        )

    def generate_report(
        self,
        goal: str = "",
        mode: str = "lite",
        loop_count: int = 0,
        proposals: Optional[List[Dict[str, Any]]] = None,
        eliminations: Optional[List[Dict[str, Any]]] = None,
        dimensions: Optional[List[Dict[str, Any]]] = None,
        total_score: float = 0.0,
        components: Optional[List[Dict[str, str]]] = None,
        suggestions: Optional[Dict[str, List[str]]] = None,
        architecture_diagram: str = "",
        session_id: str = "",
        historical_trend: Optional[List[Dict[str, Any]]] = None,
        # New enriched data
        cost_summary: Optional[Dict[str, Any]] = None,
        degradation_history: Optional[List[Dict[str, Any]]] = None,
        evolution_suggestions: Optional[List[Dict[str, Any]]] = None,
        performance_stats: Optional[Dict[str, Any]] = None,
        bootstrap_score: float = 0.0,
        degradation_level: int = 0,
        **kwargs: Any,
    ) -> str:
        """Generate a full GAAL v3 arena report from template.

        Args:
            goal: The original goal.
            mode: Execution mode (lite/hard/super).
            loop_count: Number of arena loops executed.
            proposals: List of all proposal dicts.
            eliminations: List of elimination records.
            dimensions: Scoring dimensions with scores.
            total_score: Overall total score.
            components: Final component list.
            suggestions: Implementation suggestions by phase.
            architecture_diagram: ASCII architecture diagram.
            session_id: Unique session identifier.
            historical_trend: Optional trend data across rounds.
            cost_summary: Optional cost tracking summary.
            degradation_history: Optional degradation events.
            evolution_suggestions: Optional evolution action history.
            performance_stats: Optional performance statistics.
            bootstrap_score: Bootstrap/self-evolution score.
            degradation_level: Current degradation level.
            **kwargs: Additional template variables.

        Returns:
            Rendered report string in Markdown format.

        Raises:
            FileNotFoundError: If template file is not found.
        """
        if self.env is None:
            self._init_jinja()

        # Resolve template
        try:
            template = self.env.get_template(self.template_name)  # type: ignore
        except TemplateNotFound:
            # Fallback: try to find any .j2 file in the template dir
            template_files = list(self.template_dir.glob("*.md.j2"))
            if template_files:
                template = self.env.get_template(template_files[0].name)  # type: ignore
            else:
                raise FileNotFoundError(
                    f"No template found in {self.template_dir}. "
                    f"Expected {self.template_name}"
                )

        completion_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Build proposal map for enriched eliminations
        proposal_map: Dict[Any, str] = {}
        for p in (proposals or []):
            pid = p.get("id") or p.get("name", "Unknown")
            proposal_map[pid] = p.get("name", "Unknown")

        enriched_eliminations = []
        for e in (eliminations or []):
            enriched_eliminations.append({
                "round": e.get("round", 0),
                "name": e.get("name") or proposal_map.get(
                    e.get("proposal_id"), "Unknown"
                ),
                "reason": e.get("reason", ""),
            })

        # Default values for new fields
        if cost_summary is None:
            cost_summary = {}
        if degradation_history is None:
            degradation_history = []
        if evolution_suggestions is None:
            evolution_suggestions = []
        if performance_stats is None:
            performance_stats = {}

        # Render template
        report = template.render(
            goal=goal or "未设置",
            mode=mode or "lite",
            loop_count=loop_count,
            completion_time=completion_time,
            total_score=total_score,
            session_id=session_id or "N/A",
            proposals=proposals or [],
            eliminations=enriched_eliminations,
            dimensions=dimensions or [],
            components=components or [],
            suggestions=suggestions or {},
            architecture_diagram=architecture_diagram or "(未提供架构图)",
            historical_trend=historical_trend or [],
            # New enriched fields
            cost_summary=cost_summary,
            degradation_history=degradation_history,
            evolution_suggestions=evolution_suggestions,
            performance_stats=performance_stats,
            bootstrap_score=bootstrap_score,
            degradation_level=degradation_level,
            **kwargs,
        )

        return report

    def generate_comparison_table(
        self,
        all_rounds_data: List[Dict[str, Any]],
    ) -> str:
        """Generate a score trend / comparison table across multiple rounds.

        Args:
            all_rounds_data: List of dicts, each with round_number,
                           dimensions (list of {name, score}),
                           total_score, timestamp.

        Returns:
            Markdown table string of score trends.
        """
        if not all_rounds_data:
            return "（无历史数据 / No historical data）"

        # Collect all dimension names across rounds
        dim_names: List[str] = []
        for rd in all_rounds_data:
            for d in rd.get("dimensions", []):
                if d["name"] not in dim_names:
                    dim_names.append(d["name"])

        # Build table header
        header = "| 轮次 Round | 时间 Time | " + " | ".join(dim_names) + " | 总分 Total |"
        separator = "|:----------:|:---------:|" + "|".join([":---:"] * len(dim_names)) + "|:---------:|"
        lines = ["### 评分趋势对比 / Score Trend", "", header, separator]

        for rd in all_rounds_data:
            scores: Dict[str, str] = {}
            for d in rd.get("dimensions", []):
                scores[d["name"]] = str(d.get("score", "-"))
            row_scores = [scores.get(d, "-") for d in dim_names]
            lines.append(
                f"| {rd['round_number']} | {rd.get('timestamp', '')} | "
                f"{' | '.join(row_scores)} | {rd.get('total_score', '-')} |"
            )

        return "\n".join(lines)

    def generate_short_summary(
        self,
        goal: str,
        total_score: float,
        mode: str,
        loop_count: int,
        session_id: str,
    ) -> str:
        """Generate a concise one-line summary of the run.

        Args:
            goal: The goal.
            total_score: Final score.
            mode: Execution mode.
            loop_count: Number of loops.
            session_id: Session ID.

        Returns:
            Short summary string.
        """
        return (
            f"GAAL v3 {mode.upper()} | Goal: {goal[:60]} | "
            f"Score: {total_score}/10 | Loops: {loop_count} | "
            f"Session: {session_id}"
        )
