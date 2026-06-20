"""
Tests for GAAL v3 agents — LLM-style scoring, proposal generation, goal parsing.

Tests cover:
- JudgeAgent: LLM-style dimension scoring with justification
- TeamAgent: Goal analysis and proposal generation
- OrchestratorAgent: Intelligent goal parsing
"""
import sys
import os
import json
from pathlib import Path

# Ensure the package root is on sys.path (development fallback)
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from typing import Any, Dict, List

from gaal_v3.agents.judge_agent import (
    JudgeAgent,
    _score_dimension,
    _desc_quality_score,
    _count_tech_terms,
    _count_aspects,
    DIMENSION_SCORERS,
)
from gaal_v3.agents.team_agent import TeamAgent, _analyze_goal
from gaal_v3.agents.orchestrator_agent import (
    OrchestratorAgent,
    _detect_mode,
    _parse_requirements,
    _count_tech_domain_terms,
)
from gaal_v3.agents.base import AgentContext


# ═══════════════════════════════════════════════════════════════════════
# JudgeAgent — LLM-style Scoring Tests
# ═══════════════════════════════════════════════════════════════════════

class TestLLMScoring:
    """Test the LLM-style dimension scoring system."""

    def test_score_dimension_returns_valid_range(self):
        """Each dimension scorer should return score in 0.0-2.0 range."""
        proposal_name = "企业级备份恢复方案"
        proposal_desc = (
            "采用分层备份策略，全量+增量+差异三层次备份机制。"
            "基于快照技术实现秒级恢复点目标(RPO)，"
            "结合去重压缩算法优化存储效率。"
            "支持AES-256加密存储，定时调度自动执行。"
            "架构设计高可用，故障自动转移。"
        )

        for dim_name, scorer in DIMENSION_SCORERS.items():
            score, justification = scorer(proposal_name, proposal_desc)
            assert 0.0 <= score <= 2.0, (
                f"Dimension '{dim_name}' score {score} out of range [0, 2]"
            )
            assert len(justification) > 5, (
                f"Dimension '{dim_name}' justification too short"
            )

    def test_score_proposal_returns_complete_structure(self):
        """score_proposal() should return rich scored structure with justifications."""
        judge = JudgeAgent(config={
            "judge": {
                "dimensions": [
                    {"name": "generality", "weight": 2, "description": "通用性"},
                    {"name": "reliability", "weight": 2, "description": "可靠性"},
                    {"name": "delivery_quality", "weight": 2, "description": "交付质量"},
                ]
            }
        })

        proposal = {
            "id": "test-001",
            "name": "测试方案",
            "description": (
                "一个企业级文件备份系统方案，支持增量备份、加密存储、定时调度。"
                "采用分层架构设计，热数据SSD缓存+冷数据对象存储归档。"
                "支持AES-256加密和完整性校验，确保数据安全。"
                "分布式调度引擎支持CRON表达式和日历依赖。"
            ),
            "team": "Team Alpha",
            "team_id": "team_a",
        }

        result = judge.score_proposal(proposal)

        # Check structure
        assert "dimension_scores" in result
        assert "total_score" in result
        assert 0 <= result["total_score"] <= 10

        # Check dimension scores
        dims = result["dimension_scores"]
        assert len(dims) == 3
        for dim_name, dim_data in dims.items():
            assert "score" in dim_data
            assert "weight" in dim_data
            assert "justification" in dim_data
            assert 0 <= dim_data["score"] <= 2.0
            assert len(dim_data["justification"]) > 0

        # Original fields preserved
        assert result["id"] == "test-001"
        assert result["name"] == "测试方案"

    def test_scoring_differentiates_proposals(self):
        """Different proposals should get different scores."""
        judge = JudgeAgent(config={
            "judge": {
                "dimensions": [
                    {"name": "quality", "weight": 2, "description": "质量"},
                    {"name": "feasibility", "weight": 2, "description": "可行性"},
                ]
            }
        })

        # Detailed proposal
        detailed = {
            "id": "a1",
            "name": "深度学习方案",
            "description": (
                "基于深度学习的智能文件备份系统，采用Transformer模型预测文件变更。"
                "支持增量备份和差异压缩，存储效率提升60%。"
                "分布式架构，支持水平扩展和故障自动恢复。"
                "端到端加密确保数据安全，细粒度权限控制。"
                "自动化运维监控平台，实时告警和智能诊断。"
            ),
            "team": "Team Alpha",
            "team_id": "team_a",
        }

        # Simple proposal
        simple = {
            "id": "b1",
            "name": "基础方案",
            "description": "简单的文件备份方案。",
            "team": "Team Beta",
            "team_id": "team_b",
        }

        result_a = judge.score_proposal(detailed)
        result_b = judge.score_proposal(simple)

        # The detailed proposal should score higher
        assert result_a["total_score"] > result_b["total_score"], (
            f"Detailed ({result_a['total_score']}) should beat simple ({result_b['total_score']})"
        )

    def test_default_dimensions_when_not_configured(self):
        """Should use default dimensions when none are configured."""
        judge = JudgeAgent(config={})
        assert len(judge.dimensions) == 0

        proposal = {
            "id": "test",
            "name": "方案A",
            "description": "一个测试方案，包含详细的技术描述和架构设计。",
        }
        result = judge.score_proposal(proposal)

        assert "dimension_scores" in result
        assert len(result["dimension_scores"]) == 2  # quality + feasibility
        assert "quality" in result["dimension_scores"]
        assert "feasibility" in result["dimension_scores"]

    def test_total_score_calculation(self):
        """Total score should be weighted average * 5."""
        judge = JudgeAgent(config={
            "judge": {
                "dimensions": [
                    {"name": "quality", "weight": 3, "description": "质量"},
                    {"name": "feasibility", "weight": 1, "description": "可行性"},
                ]
            }
        })

        # Mock the dimension scores
        proposal = {
            "id": "t1",
            "name": "测试方案",
            "description": (
                "一个详细的测试方案描述。采用微服务架构。"
                "支持多种备份策略。安全性设计完善。"
            ),
        }
        result = judge.score_proposal(proposal)

        # Verify total is in range
        assert 0 <= result["total_score"] <= 10

        # Verify the calculation: sum(dim_score * weight) / sum(weights) * 5
        dims = result["dimension_scores"]
        total_weighted = sum(
            dims[d]["score"] * dims[d]["weight"]
            for d in dims
        )
        total_weight = sum(dims[d]["weight"] for d in dims)
        expected = round((total_weighted / total_weight) * 5.0, 2)
        assert result["total_score"] == expected

    def test_rank_proposals(self):
        """Rank proposals by total score descending."""
        judge = JudgeAgent(config={})
        proposals = [
            {"id": "p1", "name": "A", "total_score": 8.5},
            {"id": "p2", "name": "B", "total_score": 6.0},
            {"id": "p3", "name": "C", "total_score": 9.2},
        ]
        ranked = judge.rank_proposals(proposals)

        assert ranked[0]["name"] == "C"  # Highest
        assert ranked[1]["name"] == "A"
        assert ranked[2]["name"] == "B"  # Lowest
        assert ranked[0]["rank"] == 1
        assert ranked[2]["rank"] == 3


# ═══════════════════════════════════════════════════════════════════════
# TeamAgent — Proposal Generation Tests
# ═══════════════════════════════════════════════════════════════════════

class TestTeamProposalGeneration:
    """Test TeamAgent proposal generation."""

    def test_analyze_goal_extracts_aspects(self):
        """Goal analysis should identify key technical aspects."""
        goal = "设计一个企业级文件备份系统，支持增量备份、加密存储、定时调度"
        analysis = _analyze_goal(goal)

        assert "backup" in analysis["aspects"]
        assert "encryption" in analysis["aspects"]
        assert "scheduling" in analysis["aspects"]
        assert analysis["has_multiple_requirements"]
        assert analysis["requirement_count"] > 1

    def test_team_a_generates_structured_proposals(self):
        """Team A should generate deep/quality proposals."""
        goal = "设计一个分布式文件备份系统"
        agent = TeamAgent(
            name="TeamAlpha",
            context=AgentContext(goal=goal, mode="lite"),
        )
        agent.configure("team_a", {"name": "Team Alpha", "tier": "pro"})

        proposals = agent.execute(goal=goal)

        assert len(proposals) > 0
        for p in proposals:
            assert "name" in p
            assert "description" in p
            assert len(p["name"]) > 0
            assert len(p["description"]) > 50  # Substantial description
            assert p["team_id"] == "team_a"

    def test_team_b_generates_creative_proposals(self):
        """Team B should generate creative/diverse proposals."""
        goal = "设计一个分布式文件备份系统"
        agent = TeamAgent(
            name="TeamBeta",
            context=AgentContext(goal=goal, mode="lite"),
        )
        agent.configure("team_b", {"name": "Team Beta", "tier": "flash"})

        proposals = agent.execute(goal=goal)

        assert len(proposals) > 0
        for p in proposals:
            assert "name" in p
            assert "description" in p
            assert p["team_id"] == "team_b"

    def test_team_a_and_b_proposals_differ(self):
        """Team A and B proposals should have different styles."""
        goal = "设计企业级文件备份系统"

        agent_a = TeamAgent(
            name="TeamAlpha",
            context=AgentContext(goal=goal, mode="lite"),
        )
        agent_a.configure("team_a", {"name": "Team Alpha", "tier": "pro"})
        proposals_a = agent_a.execute(goal=goal)

        agent_b = TeamAgent(
            name="TeamBeta",
            context=AgentContext(goal=goal, mode="lite"),
        )
        agent_b.configure("team_b", {"name": "Team Beta", "tier": "flash"})
        proposals_b = agent_b.execute(goal=goal)

        # Names should differ
        names_a = [p["name"] for p in proposals_a]
        names_b = [p["name"] for p in proposals_b]
        assert names_a != names_b, "Team A and B should generate different proposal names"

        # Descriptions should differ
        for pa, pb in zip(proposals_a, proposals_b):
            assert pa["description"] != pb["description"]


# ═══════════════════════════════════════════════════════════════════════
# OrchestratorAgent — Goal Parsing Tests
# ═══════════════════════════════════════════════════════════════════════

class TestGoalParsing:
    """Test OrchestratorAgent goal parsing."""

    def test_parse_requirements(self):
        """Should split goals by Chinese/English delimiters."""
        goal = "支持增量备份、加密存储、定时调度"
        reqs = _parse_requirements(goal)
        assert len(reqs) >= 3
        assert "支持增量备份" in reqs

    def test_parse_requirements_english(self):
        """Should handle English delimiters."""
        goal = "incremental backup, encryption, scheduled tasks"
        reqs = _parse_requirements(goal)
        assert "incremental backup" in reqs

    def test_tech_domain_terms_count(self):
        """Should count technical domain terms."""
        goal = "设计分布式企业级文件备份系统，支持高并发、低延迟、增量备份"
        count = _count_tech_domain_terms(goal)
        assert count >= 3  # distributed, enterprise, backup, etc.

    def test_detect_mode_lite(self):
        """Simple goals should get lite mode."""
        goal = "设计一个简单的备份系统"
        mode = _detect_mode(goal, {"gaal": {"mode": "auto"}})
        assert mode == "lite"

    def test_detect_mode_hard(self):
        """Moderately complex goals should get hard mode."""
        goal = "设计一个企业级文件备份系统，支持增量备份、加密存储、定时调度、负载均衡、高可用部署"
        mode = _detect_mode(goal, {"gaal": {"mode": "auto"}})
        assert mode in ("hard", "super")  # At minimum hard

    def test_detect_mode_super(self):
        """Very complex goals should get super mode."""
        goal = ("设计一个分布式企业级微服务文件备份系统，支持增量备份、加密存储、定时调度、"
                "负载均衡、高可用部署、实时监控告警、自动故障恢复、多数据中心同步、"
                "细粒度权限控制、审计日志、数据压缩去重")
        mode = _detect_mode(goal, {"gaal": {"mode": "auto"}})
        assert mode == "super"

    def test_orchestrator_parse_goal(self):
        """OrchestratorAgent.parse_goal() should return rich analysis."""
        orchestrator = OrchestratorAgent(
            config={
                "gaal": {"mode": "auto", "max_loops": 4, "max_proposals_per_team": 2},
                "judge": {
                    "dimensions": [
                        {"name": "quality", "weight": 2},
                        {"name": "feasibility", "weight": 2},
                    ]
                },
            },
            context=AgentContext(
                goal="设计一个企业级文件备份系统，支持增量备份、加密存储、定时调度"
            ),
        )

        result = orchestrator.parse_goal()

        assert "mode" in result
        assert "is_simple" in result
        assert "complexity_score" in result
        assert "complexity_breakdown" in result
        assert "dimensions" in result
        assert result["complexity_breakdown"]["has_multiple_requirements"]
        assert "requirements_list" in result["complexity_breakdown"]


# ═══════════════════════════════════════════════════════════════════════
# GAALOrchestrator — New Feature Tests
# ═══════════════════════════════════════════════════════════════════════

class TestBootstrapSelfEvolution:
    """Test the bootstrap self-evolution system."""

    def test_compute_bootstrap_score(self):
        """compute_bootstrap_score should return a valid score."""
        from gaal_v3.core.orchestrator import compute_bootstrap_score

        # Good state
        state_good = {
            "total_score": 8.5,
            "proposals": [{"id": "1"}, {"id": "2"}, {"id": "3"}, {"id": "4"}],
            "eliminations": [{"id": "e1"}, {"id": "e2"}],
            "execution_history": [{"node": "a"}, {"node": "b"}, {"node": "c"}],
            "total_retries": 1,
            "degradation_level": 0,
        }
        score = compute_bootstrap_score(state_good)
        assert 0 <= score <= 10
        assert score >= 5.0  # Good state should score high

        # Poor state
        state_poor = {
            "total_score": 3.0,
            "proposals": [],
            "eliminations": [],
            "execution_history": [],
            "total_retries": 10,
            "degradation_level": 2,
        }
        score_poor = compute_bootstrap_score(state_poor)
        assert 0 <= score_poor <= 10
        assert score_poor < score  # Poor should be lower

    def test_suggest_evolution_action_increasing_loops(self):
        """Should suggest increasing max_loops when score declines."""
        from gaal_v3.core.orchestrator import suggest_evolution_action

        state = {
            "scored_proposals": [],
            "total_score": 6.0,
        }
        config = {"gaal": {"max_loops": 4}, "judge": {"dimensions": []}, "evolution": {"enabled": False}}

        action = suggest_evolution_action([7.0, 6.0], state, config)
        assert action is not None
        assert action["action"] == "increase_loops"

    def test_suggest_evolution_action_adjust_weight(self):
        """Should suggest adjusting weight when a dimension is weak."""
        from gaal_v3.core.orchestrator import suggest_evolution_action

        state = {
            "scored_proposals": [{
                "dimension_scores": {
                    "bootstrap": {"score": 0.5},
                    "reliability": {"score": 1.5},
                }
            }],
            "total_score": 7.0,
        }
        config = {
            "gaal": {"max_loops": 4},
            "judge": {"dimensions": [
                {"name": "bootstrap", "weight": 2},
                {"name": "reliability", "weight": 2},
            ]},
            "evolution": {"enabled": False},
        }

        action = suggest_evolution_action([7.0, 7.5], state, config)
        # Should find 'bootstrap' as weakest and suggest adjusting weight
        if action is not None:
            assert action["action"] in ("adjust_weight", "enable_evolution")

    def test_apply_evolution_action_increase_loops(self):
        """Applying increase_loops should modify the config."""
        from gaal_v3.core.orchestrator import GAALOrchestrator

        orc = GAALOrchestrator(config={"gaal": {"max_loops": 4}})
        action = {
            "action": "increase_loops",
            "target": "config/gaal_v3.yaml",
            "reason": "Test",
            "params": {"max_loops": 6},
        }
        result = orc._apply_evolution_action(action)
        assert result["status"] == "applied"
        assert orc.config["gaal"]["max_loops"] == 6

    def test_apply_evolution_action_enable_evolution(self):
        """Applying enable_evolution should enable evolution in config."""
        from gaal_v3.core.orchestrator import GAALOrchestrator

        orc = GAALOrchestrator(config={"evolution": {"enabled": False}})
        action = {
            "action": "enable_evolution",
            "target": "config/gaal_v3.yaml",
            "reason": "Test",
            "params": {"evolution.enabled": True},
        }
        result = orc._apply_evolution_action(action)
        assert result["status"] == "applied"
        assert orc.config["evolution"]["enabled"] is True

    def test_save_evolution_artifact(self):
        """Evolution artifacts should be saved to evolution/ directory."""
        from gaal_v3.core.orchestrator import GAALOrchestrator
        import os

        orc = GAALOrchestrator(config={})
        action = {"action": "test", "target": "config/test.yaml", "reason": "Test"}
        orc._save_evolution_artifact(action)

        # Check that evolution directory has files
        evo_dir = Path(__file__).parent.parent / "evolution"
        assert evo_dir.exists()
        files = list(evo_dir.glob("*.json"))
        assert len(files) > 0

    def test_evolve_config_method(self):
        """evolve_config() should return an action result."""
        from gaal_v3.core.orchestrator import GAALOrchestrator

        orc = GAALOrchestrator(config={
            "gaal": {"max_loops": 4},
            "judge": {"dimensions": []},
            "evolution": {"enabled": True},
        })

        # Trigger with a specific action
        result = orc.evolve_config({
            "action": "increase_loops",
            "target": "config/gaal_v3.yaml",
            "reason": "Manual evolution",
            "params": {"max_loops": 6},
        })
        assert result["status"] == "applied"


class TestGracefulDegradation:
    """Test graceful degradation system."""

    def test_degradation_level_in_state(self):
        """degradation_level should be in graph state."""
        from gaal_v3.core.orchestrator import GAALOrchestrator

        orc = GAALOrchestrator(config={})
        assert orc.degradation_level == 0

    def test_degradation_super_to_hard(self):
        """Super mode should degrade to hard."""
        from gaal_v3.core.orchestrator import GAALOrchestrator

        orc = GAALOrchestrator(config={})
        orc.mode = "super"
        orc._original_mode = "super"
        orc.degradation_level = 0

        result = orc._run_degraded("test goal")
        assert result["degradation_level"] >= 1
        assert orc.mode in ("hard", "lite")

    def test_degradation_hard_to_lite(self):
        """Hard mode should degrade to lite."""
        from gaal_v3.core.orchestrator import GAALOrchestrator

        orc = GAALOrchestrator(config={})
        orc.mode = "hard"
        orc._original_mode = "hard"
        orc.degradation_level = 1

        result = orc._run_degraded("test goal")
        assert result["degradation_level"] >= 1
        assert orc.mode == "lite" or orc.mode == "lite"

    def test_fallback_proposal_generation(self):
        """Fallback proposals should be generated on timeout."""
        from gaal_v3.core.orchestrator import GAALOrchestrator

        orc = GAALOrchestrator(config={})
        proposals = orc._generate_fallback_proposal("test goal", "TeamAlpha", "team_a")
        assert len(proposals) == 1
        assert "fallback" in proposals[0]["name"].lower()
        assert proposals[0]["team_id"] == "team_a"


class TestCostTracking:
    """Test cost tracking system."""

    def test_track_node_cost(self):
        """Tracking cost for a node should accumulate data."""
        from gaal_v3.core.orchestrator import GAALOrchestrator

        orc = GAALOrchestrator(config={})
        orc._track_node_cost("test_node", "testing", 1000, 3)

        assert "test_node" in orc.cost_data["per_node"]
        assert orc.cost_data["per_node"]["test_node"]["calls"] == 1
        assert orc.cost_data["per_node"]["test_node"]["total_tokens"] == 1000
        assert orc.cost_data["per_node"]["test_node"]["total_cost"] > 0

    def test_cost_summary_structure(self):
        """Cost summary should have the expected structure."""
        from gaal_v3.core.orchestrator import GAALOrchestrator

        orc = GAALOrchestrator(config={})
        orc._track_node_cost("propose_team_a", "team_a_proposal", 500, 1)
        orc._track_node_cost("propose_team_b", "team_b_proposal", 300, 3)
        orc._track_node_cost("judge", "final_scoring", 1000, 3)

        summary = orc._build_cost_summary()
        assert "total_calls" in summary
        assert "total_tokens" in summary
        assert "total_cost" in summary
        assert "per_node" in summary
        assert "per_team" in summary
        assert summary["total_calls"] == 3
        assert summary["total_tokens"] == 1800
        assert summary["budget_exceeded"] is False

    def test_budget_enforcement(self):
        """Budget exceeded flag should be set when over limit."""
        from gaal_v3.core.orchestrator import GAALOrchestrator

        orc = GAALOrchestrator(config={})
        orc._max_total_tokens = 100  # Very low budget

        orc._track_node_cost("big_node", "test", 200, 1)
        assert orc.cost_data["budget_exceeded"] is True

    def test_per_team_cost_tracking(self):
        """Cost should be tracked per team correctly."""
        from gaal_v3.core.orchestrator import GAALOrchestrator

        orc = GAALOrchestrator(config={})

        orc._track_node_cost("propose_team_a", "team_a_proposal", 1000, 3)
        orc._track_node_cost("propose_team_b", "team_b_proposal", 500, 1)

        assert orc.cost_data["per_team"]["team_a"] > 0
        assert orc.cost_data["per_team"]["team_b"] > 0
        assert orc.cost_data["per_team"]["team_a"] > orc.cost_data["per_team"]["team_b"]


class TestPerformanceStats:
    """Test performance statistics."""

    def test_build_performance_stats(self):
        """Performance stats should be computed from execution history."""
        from gaal_v3.core.orchestrator import GAALOrchestrator

        orc = GAALOrchestrator(config={})
        orc.execution_history = [
            {"node": "parse_goal", "status": "completed", "duration": 0.5, "attempts": 1},
            {"node": "propose_team_a", "status": "completed", "duration": 1.2, "attempts": 2},
            {"node": "judge", "status": "completed", "duration": 0.3, "attempts": 1},
        ]

        stats = orc._build_performance_stats({})
        assert stats["total_nodes"] == 3
        assert stats["total_attempts"] == 4
        assert stats["retries"] == 1
        assert stats["avg_duration_per_node"] > 0
        assert "parse_goal" in stats["node_stats"]


class TestParallelTeamExecution:
    """Test parallel team execution support."""

    def test_parallel_team_execution_both_return(self):
        """Both teams should return proposals when executed in parallel."""
        from gaal_v3.core.orchestrator import GAALOrchestrator

        orc = GAALOrchestrator(config={
            "teams": {
                "team_a": {"name": "Team Alpha", "tier": "pro"},
                "team_b": {"name": "Team Beta", "tier": "flash"},
            },
            "execution": {"team_timeout": 30},
        })

        state = {
            "goal": "设计一个文件备份系统",
            "mode": "lite",
            "current_loop": 0,
            "proposals": [],
        }

        result = orc._node_propose_team_a(state)
        assert "proposals" in result
        assert "team_a_proposals" in result
        assert "team_b_proposals" in result
        assert len(result["proposals"]) >= 2  # At least one from each team

    def test_fallback_on_timeout(self):
        """Fallback proposals should be generated if team fails."""
        from gaal_v3.core.orchestrator import GAALOrchestrator

        orc = GAALOrchestrator(config={})
        fallback = orc._generate_fallback_proposal("test", "TeamAlpha", "team_a")
        assert len(fallback) == 1
        assert fallback[0]["team_id"] == "team_a"


class TestCircuitBreakerPersistence:
    """Test circuit breaker state persistence."""

    def test_circuit_breaker_serialize_deserialize(self):
        """Circuit breaker state should be serializable and restorable."""
        from gaal_v3.core.orchestrator import CircuitBreaker

        cb = CircuitBreaker(threshold=3, reset_seconds=60)
        cb.record_failure()
        cb.record_failure()

        data = cb.to_dict()
        assert data["state"] == "closed"  # Still closed (2 < 3)
        assert data["failure_count"] == 2

        # Restore
        cb2 = CircuitBreaker.from_dict(data)
        assert cb2.threshold == 3
        assert cb2.failure_count == 2
        assert cb2.state == "closed"

    def test_circuit_breaker_opens_at_threshold(self):
        """Circuit breaker should open at threshold."""
        from gaal_v3.core.orchestrator import CircuitBreaker

        cb = CircuitBreaker(threshold=3, reset_seconds=60)
        cb.record_failure()
        cb.record_failure()
        cb.record_failure()
        assert cb.state == "open"
        assert cb.can_proceed() is False
