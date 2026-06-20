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

# Ensure the package root is on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from typing import Any, Dict, List

from agents.judge_agent import (
    JudgeAgent,
    _score_dimension,
    _desc_quality_score,
    _count_tech_terms,
    _count_aspects,
    DIMENSION_SCORERS,
)
from agents.team_agent import TeamAgent, _analyze_goal
from agents.orchestrator_agent import (
    OrchestratorAgent,
    _detect_mode,
    _parse_requirements,
    _count_tech_domain_terms,
)
from agents.base import AgentContext


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
