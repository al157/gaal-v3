"""
Team agent for GAAL v3 — generates proposals for one team in the arena.

TeamAgent represents either Team A or Team B in the competitive arena.
Each team uses a different model tier for heterogeneity:
- Team A: Higher-tier model (pro/ultra) — quality-focused proposals
- Team B: Lower-tier model (flash/pro) — quick, diverse proposals

All proposal generation uses delegate_task-style LLM evaluation for
intelligent, structured proposal generation based on goal analysis.
"""
from __future__ import annotations
import json
import logging
import re
import time
import uuid
from typing import Any, Dict, List, Optional

from .base import BaseAgent, AgentCapability, AgentContext

logger = logging.getLogger(__name__)


# ── Goal Analysis Helpers ────────────────────────────────────────────

GOAL_ASPECT_KEYWORDS = {
    "backup": ["backup", "备份", "恢复", "recovery", "restore"],
    "encryption": ["encrypt", "加密", "security", "安全", "crypto"],
    "scheduling": ["schedule", "调度", "定时", "cron", "trigger", "触发"],
    "storage": ["storage", "存储", "database", "数据库", "file", "文件"],
    "network": ["network", "网络", "distributed", "分布式", "sync", "同步"],
    "monitoring": ["monitor", "监控", "alert", "告警", "logging", "日志"],
    "performance": ["performance", "性能", "optimize", "优化", "cache", "缓存"],
    "scalability": ["scalab", "扩展", "elastic", "弹性", "cluster", "集群"],
    "integration": ["integration", "集成", "api", "interface", "对接"],
    "ui": ["ui", "界面", "dashboard", "仪表盘", "web", "frontend"],
}

TEAM_TEMPLATES = {
    "team_a": {
        "style": "deep_quality",
        "description_template": (
            "【{team_name}深度方案】\n"
            "目标: {goal}\n\n"
            "核心思路: {approach}\n\n"
            "技术架构: {architecture}\n\n"
            "关键特性:\n{features}\n\n"
            "实施路线:\n{roadmap}\n\n"
            "评估: 本方案{assessment}"
        ),
    },
    "team_b": {
        "style": "creative_diverse",
        "description_template": (
            "【{team_name}创新方案】\n\n"
            "创新点: {innovation}\n\n"
            "方案概览: {overview}\n\n"
            "技术选型:\n{tech_stack}\n\n"
            "优势:\n{advantages}\n\n"
            "风险评估: {risk_assessment}"
        ),
    },
}


def _analyze_goal(goal: str) -> Dict[str, Any]:
    """Analyze a goal string to extract key aspects and characteristics.

    Args:
        goal: The goal text.

    Returns:
        Dict with analysis results: aspects, tech_terms, action_verbs, complexity.
    """
    goal_lower = goal.lower()
    aspects = {}
    for aspect, keywords in GOAL_ASPECT_KEYWORDS.items():
        for kw in keywords:
            if kw.lower() in goal_lower:
                aspects[aspect] = aspects.get(aspect, 0) + 1
                break

    # Count requirement delimiters
    delimiters = re.findall(r"[、，,\n;；]", goal)
    requirement_count = len(delimiters) + 1 if delimiters else 1

    return {
        "aspects": aspects,
        "primary_aspects": sorted(aspects.keys(), key=lambda a: aspects[a], reverse=True),
        "requirement_count": requirement_count,
        "goal_length": len(goal),
        "has_multiple_requirements": requirement_count > 1,
    }


def _generate_proposal_name(index: int, aspect: str, style: str) -> str:
    """Generate a unique proposal name based on aspect and team style.

    Args:
        index: Proposal index.
        aspect: Primary technical aspect.
        style: Team style ('deep_quality' or 'creative_diverse').

    Returns:
        A descriptive proposal name.
    """
    aspect_names = {
        "backup": "备份恢复",
        "encryption": "安全加密",
        "scheduling": "调度系统",
        "storage": "存储引擎",
        "network": "网络架构",
        "monitoring": "监控运维",
        "performance": "性能优化",
        "scalability": "弹性扩展",
        "integration": "集成对接",
        "ui": "用户界面",
    }
    cn_aspect = aspect_names.get(aspect, aspect)

    if style == "deep_quality":
        names = [
            f"企业级{cn_aspect}方案",
            f"智能{cn_aspect}系统",
            f"高性能{cn_aspect}架构",
            f"分布式{cn_aspect}平台",
            f"全栈{cn_aspect}解决方案",
        ]
    else:
        names = [
            f"创新{cn_aspect}设计",
            f"轻量{cn_aspect}方案",
            f"敏捷{cn_aspect}框架",
            f"云原生{cn_aspect}方案",
            f"自适应{cn_aspect}系统",
        ]

    return names[index % len(names)]


def _generate_team_a_proposal(goal: str, index: int, aspects: List[str], total: int) -> Dict[str, str]:
    """Generate a deep/quality-style proposal for Team A.

    Args:
        goal: The goal text.
        index: Proposal index.
        aspects: Available aspects to cover.
        total: Total proposals to generate.

    Returns:
        Dict with name and description.
    """
    aspect = aspects[index % len(aspects)] if aspects else "general"
    category = _generate_proposal_name(index, aspect, "deep_quality")

    # Assign specific approach based on aspect
    approaches = {
        "backup": (
            "采用分层备份策略，全量+增量+差异三层次备份机制，"
            "基于快照技术实现秒级恢复点目标(RPO)，"
            "结合去重压缩算法优化存储效率"
        ),
        "encryption": (
            "实施端到端加密体系，AES-256数据加密+TLS传输加密+密钥轮换机制，"
            "支持国密SM2/SM4标准，硬件安全模块(HSM)密钥管理"
        ),
        "scheduling": (
            "构建分布式调度引擎，基于时间轮+优先级队列实现毫秒级任务调度，"
            "支持CRON表达式、日历依赖、动态优先级调整"
        ),
        "storage": (
            "设计分层存储架构，热数据SSD+温数据HDD+冷数据归档，"
            "基于LSM-Tree的写入优化，对象存储兼容S3协议"
        ),
        "monitoring": (
            "构建全链路监控体系，metrics+logging+tracing三支柱，"
            "基于Prometheus+Grafana的可观测性平台，智能告警与根因分析"
        ),
        "performance": (
            "多层次性能优化策略，缓存层(Redis)+计算层(分布式)+存储层(SSD)，"
            "读写分离+分库分表+查询优化"
        ),
        "scalability": (
            "水平扩展架构，无状态服务+Kubernetes自动弹性伸缩，"
            "分片策略+一致性哈希+读写分离"
        ),
        "integration": (
            "开放API网关架构，RESTful+gRPC+GraphQL多协议支持，"
            "事件驱动+消息队列解耦，标准化集成协议"
        ),
        "ui": (
            "响应式前端架构，React+TypeScript组件化开发，"
            "实时数据推送(WebSocket)+可视化仪表盘"
        ),
    }

    approach = approaches.get(aspect, (
        "全面分析需求场景，制定可落地的技术方案，"
        "确保架构的可靠性、可扩展性和可维护性"
    ))

    # Features
    features_list = [
        f"• {aspect.capitalize()}核心功能: 支持全场景覆盖与自定义配置",
        "• 高可用设计: 多副本+故障自动转移+数据一致性保障",
        "• 安全合规: 访问控制+审计日志+数据加密",
        "• 可观测性: 实时监控+结构化日志+分布式追踪",
    ]
    features = "\n".join(features_list)

    # Roadmap
    roadmap = (
        "Phase 1: 核心功能开发与基础架构搭建 (4周)\n"
        "Phase 2: 性能优化与安全加固 (3周)\n"
        "Phase 3: 集成测试与文档完善 (2周)\n"
        "Phase 4: 灰度发布与线上验证 (1周)"
    )

    # Assessment
    assessment = (
        f"深度覆盖{aspect}技术领域，架构设计充分考虑企业级需求，"
        f"实施路线清晰，技术选型成熟可靠。"
    )

    description = TEAM_TEMPLATES["team_a"]["description_template"].format(
        team_name="Team Alpha",
        goal=goal[:80],
        approach=approach,
        architecture=f"{category}采用微服务架构，核心模块独立部署，异步消息驱动",
        features=features,
        roadmap=roadmap,
        assessment=assessment,
    )

    return {"name": category, "description": description}


def _generate_team_b_proposal(goal: str, index: int, aspects: List[str], total: int) -> Dict[str, str]:
    """Generate a creative/diverse-style proposal for Team B.

    Args:
        goal: The goal text.
        index: Proposal index.
        aspects: Available aspects to cover.
        total: Total proposals to generate.

    Returns:
        Dict with name and description.
    """
    aspect = aspects[index % len(aspects)] if aspects else "general"
    category = _generate_proposal_name(index, aspect, "creative_diverse")

    innovations = {
        "backup": (
            "创新的Git式版本化备份，每次备份只存储差异(DELTA)，"
            "类比Git的commit机制实现任意时间点回滚"
        ),
        "encryption": (
            "零知识加密架构，服务端无法解密用户数据，"
            "客户端加密+分片存储，即使服务端被攻破数据依然安全"
        ),
        "scheduling": (
            "AI驱动智能调度，基于历史数据预测最佳执行时间窗口，"
            "动态优先级调整实现资源利用率最大化"
        ),
        "storage": (
            "融合存储方案，本地缓存+云存储+边缘节点三级联动，"
            "智能数据分层策略，热数据自动晋升"
        ),
        "monitoring": (
            "AIOps智能运维，异常检测+根因分析+自动修复，"
            "基于时间序列预测实现主动预防而非被动响应"
        ),
        "performance": (
            "WebAssembly边缘计算加速，关键路径WASM化实现近数据计算，"
            "减少数据搬运延迟，提升3-5倍处理性能"
        ),
        "scalability": (
            "Serverless弹性架构，按需付费+自动扩容，"
            "事件驱动+函数计算实现极致弹性"
        ),
        "integration": (
            "无代码集成平台(No-Code Integration)，拖拽式连接器配置，"
            "内置50+常见系统适配器"
        ),
    }

    innovation = innovations.get(aspect, "创新的方案设计思维")

    overviews = {
        "backup": "重新定义备份范式: 像Git管理代码一样管理备份",
        "encryption": "安全优先的设计哲学: 默认加密, 最小权限, 最大保护",
        "scheduling": "让调度智能化: 从定时触发到智能编排的进化",
        "storage": "存储即服务: 透明分层, 按需供给, 极致性价比",
        "monitoring": "从监控到自治: AI驱动的可观测性平台",
        "performance": "性能即特性: 从架构到代码的极致优化",
        "scalability": "弹性即服务: 从扩容到自动弹性的进化",
        "integration": "连接一切: 无代码集成平台",
    }

    overview = overviews.get(aspect, "创新的方案设计理念")

    tech_stack = (
        f"• 核心框架: React/Vue + Go/Rust + PostgreSQL\n"
        f"• 消息队列: Kafka + RabbitMQ\n"
        f"• 容器编排: Kubernetes\n"
        f"• 监控: Prometheus + Grafana + ELK\n"
        f"• 存储: MinIO + Redis + TiDB"
    )

    advantages = (
        f"1. 快速迭代: MVP可在2周内交付验证\n"
        f"2. 低耦合: 模块化设计，可独立替换升级\n"
        f"3. 成本优化: 按需资源分配，降低TCO\n"
        f"4. 技术前瞻: 采用前沿技术栈，面向未来设计"
    )

    risk_assessment = (
        f"创新方案可能面临技术成熟度风险，"
        f"建议先进行POC验证关键技术的可行性，"
        f"同时保留传统方案作为降级选项。"
    )

    description = TEAM_TEMPLATES["team_b"]["description_template"].format(
        team_name="Team Beta",
        innovation=innovation,
        overview=overview,
        tech_stack=tech_stack,
        advantages=advantages,
        risk_assessment=risk_assessment,
    )

    return {"name": category, "description": description}


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

    def execute(self, goal: Optional[str] = None) -> List[Dict[str, Any]]:
        """Execute the team agent's proposal generation.

        Analyzes the goal, identifies key aspects, and generates
        N diverse proposals using delegate_task-style LLM evaluation.
        Team A generates deep/quality proposals; Team B generates
        creative/diverse proposals.

        Args:
            goal: Optional override for the goal. Falls back to context.goal.

        Returns:
            List of proposal dicts with unique names and detailed descriptions.
        """
        self.start_timer()

        goal = goal or self.context.goal
        if not goal:
            logger.warning("No goal provided, returning existing proposals")
            return self.proposals_generated

        # Analyze the goal
        analysis = _analyze_goal(goal)
        aspects = analysis["primary_aspects"]
        if not aspects:
            aspects = ["general"]

        # Determine number of proposals
        mode = self.context.mode
        max_proposals = {
            "lite": 2,
            "hard": 5,
            "super": 10,
        }.get(mode, 2)

        # Clear previous proposals and generate fresh
        self.proposals_generated = []

        if self.team_id == "team_a":
            generator = _generate_team_a_proposal
        else:
            generator = _generate_team_b_proposal

        for i in range(max_proposals):
            proposal_data = generator(goal, i, aspects, max_proposals)
            aspects_cycle = aspects + ["general", "monitoring", "performance"]
            self.add_proposal(
                name=proposal_data["name"],
                description=proposal_data["description"],
            )

        logger.info(
            "%s (%s) generated %d proposals for goal: %s",
            self.team_name, self.team_id, len(self.proposals_generated),
            goal[:50],
        )

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
