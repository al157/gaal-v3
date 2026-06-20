# GAAL v3 — LangGraph-inspired StateGraph Arena Framework

**GAAL (Goal-oriented Autonomous Arena Loop)** v3 是竞技场框架的完全重写版，采用 LangGraph 风格的 StateGraph 架构，支持模式驱动、成本优化、自愈和自我进化。

GAAL v3 is a complete rewrite of the arena framework using LangGraph-inspired StateGraph architecture, with mode-driven execution, cost optimization, self-healing, and recursive self-evolution.

## Architecture / 架构

```
         ┌──────────────────────────────────────────────────┐
         │                                                  │
         ▼                                                  │
    ┌─────────┐    ┌──────────┐    ┌──────────┐    ┌──────┐│
    │PARSE_GOAL│───▶│ RESEARCH │───▶│ PROPOSE  │───▶│ EVAL ││
    └─────────┘    └──────────┘    └──────────┘    └──────┘│
         │                            │               │     │
         │                            │               │     │
         ▼                            ▼               ▼     │
    Condition:                  Team A + B        Score ≥    │
    is_simple? → END            parallel          threshold? │
    else → RESEARCH             delegate_task     YES→EVOLVE │
                                                  NO→loop ───┘
```

### Nodes / 节点
| Node | Description |
|------|-------------|
| `parse_goal` | 解析目标，确定模式 |
| `research` | (super mode) 全网调研 |
| `propose_team_a` | Team A 提案生成 |
| `propose_team_b` | Team B 提案生成 |
| `aggregate` | 合并所有方案，应用评分卡 |
| `eliminate` | 排名并淘汰弱方案 |
| `deep_dive` | 幸存方案头对头对比 |
| `judge` | 最终评分 |
| `evolve` | 自进化（修改配置/评分卡） |
| `report` | 生成最终报告 |

### Agents / 代理
| Agent | Pattern | Role |
|-------|---------|------|
| **OrchestratorAgent** | CrewAI Hierarchical | 管理者，协调整个竞技场 |
| **TeamAgent** | OpenAI Swarm | 工作节点，生成方案（两队异构模型） |
| **JudgeAgent** | — | 评估者，评分和排名 |

## Modes / 模式

| Mode | Loops | Proposals/Team | Research | Team A Tier | Team B Tier |
|------|-------|----------------|----------|-------------|-------------|
| **lite** | 4 | 2 | No | pro | flash |
| **hard** | 10 | 10 | No | pro | flash |
| **super** | 20 | 10 | Yes | ultra | pro |

## Key Features (v3.2 Enhancement)

### LLM-Style Scoring (JudgeAgent)
Replaced heuristic description-length + name-hash scoring with proper delegate_task-style LLM evaluations:
- **Per-dimension content analysis**: Each dimension (generality, reliability, delivery quality, bootstrap, etc.) gets a detailed analysis of the proposal description
- **Realistic 0-2 scoring per dimension** with written justification explaining the score
- **Weighted total** = sum(dimension\_score × weight) / sum(weights) × 5 (0-10 scale)
- **Differentiated scores**: Detailed proposals with technical depth score higher than vague ones
- **8 built-in dimension scorers**: generality, zero\_leak, reliability, delivery\_quality, bootstrap, quality, feasibility, plus extensible fallback for custom dimensions

### Intelligent Proposal Generation (TeamAgent)
Replaced empty/generic proposals with goal-driven intelligent generation:
- **Goal analysis**: Extracts key technical aspects (backup, encryption, scheduling, etc.) from the goal
- **Team A (Team Alpha)**: Deep/quality proposals with technical architecture, roadmap, and assessment
- **Team B (Team Beta)**: Creative/diverse proposals with innovation points, tech stack, and risk assessment
- **Unique naming**: Aspect-aware proposal naming based on the technical domain
- **Structured description format**: Each proposal includes approach, architecture, features, and roadmap

### Intelligent Goal Analysis (OrchestratorAgent)
Replaced basic keyword matching with multi-signal complexity detection:
- **Technical term analysis**: Detects 30+ domain-specific technical terms (分布式, microservice, kubernetes, etc.)
- **Action verb categorization**: Classifies verbs into research/design/build/analyze/optimize categories
- **Requirement parsing**: Splits goals by Chinese/English delimiters (、，,\n)
- **Multi-signal mode detection**: Combines term density, requirement count, goal length, and verb diversity for accurate mode tiering (lite/hard/super)
- **Comprehensive breakdown**: Returns detailed complexity analysis including requirement list and verb counts

## Directory Structure / 目录结构

```
~/.hermes/gaal-v3/
├── core/
│   ├── __init__.py       # Package init, version
│   ├── graph.py          # LangGraph-like StateGraph (no external deps)
│   ├── persistence.py    # SQLite checkpoint store (thread-safe, WAL)
│   ├── model_router.py   # Cost-aware model routing
│   ├── orchestrator.py   # GAAL v3 orchestrator (builds + runs graph)
│   └── report.py         # Jinja2 report generator
├── agents/
│   ├── __init__.py
│   ├── base.py           # BaseAgent abstract class
│   ├── orchestrator_agent.py
│   ├── team_agent.py
│   └── judge_agent.py
├── config/
│   ├── gaal_v3.yaml      # Lite mode config
│   ├── gaal_v3_hard.yaml # Hard mode config
│   ├── gaal_v3_super.yaml# Super mode config
│   └── scorecard.yaml    # Scoring dimensions & weights
├── templates/
│   └── report_v3.md.j2   # Jinja2 report template
├── state/                # DB files (gitignored)
├── evolution/            # Self-evolution artifacts
├── tests/
│   ├── test_graph.py     # 30+ tests for graph engine
│   ├── test_persistence.py # 30+ tests for persistence
│   └── test_agents.py    # 15+ tests for agents (LLM scoring, proposals, parsing)
├── setup.py / pyproject.toml
└── README.md
```

## Key Design Decisions / 关键设计决策

1. **No external langgraph dependency** — 纯 stdlib 实现 StateGraph 模式
2. **Delegate_task isolation** — 所有 LLM 调用在子会话中执行，零泄漏
3. **LLM-style evaluations** — 评分、提案生成、目标分析均模拟 delegate_task 输出
4. **SQLite with WAL + checkpointing** — 线程安全，崩溃可恢复
5. **Config-driven** — 所有参数在 YAML 中，无硬编码
6. **Circuit breaker** — 5 次连续失败后自动熔断
7. **Graceful degradation** — 熔断后降级到 lite 模式
8. **Retry with exponential backoff** — 自动重试（2^n * 5s）

## Usage / 使用

```python
from core.orchestrator import GAALOrchestrator

# Initialize orchestrator with config
orchestrator = GAALOrchestrator(
    config_path="config/gaal_v3.yaml",
    state_dir="state",
)

# Run the arena
result = orchestrator.run(
    goal="Design a microservice architecture for an e-commerce platform",
    mode="lite",  # auto, lite, hard, super
)

# Access results
print(f"Score: {result['final_state']['total_score']}/10")
print(f"Passed: {result['final_state']['passed']}")
print(f"Steps: {len(result['execution_history'])}")
```

### CLI Usage

```bash
# Run with a goal
cd ~/.hermes/gaal_v3
python run.py "设计一个企业级文件备份系统，支持增量备份、加密存储、定时调度" lite 2

# Run with defaults
python run.py "设计一个简单的文件备份系统"
```

## Testing / 测试

```bash
cd ~/.hermes/gaal_v3
python -m pytest tests/ -v
```

## Report Format / 报告格式

Reports follow the [T][M][V][I][A][W] format:

- **[T]** Task Overview — 目标、模式、轮次、评分
- **[M]** All Proposals — 全部方案记录
- **[V]** Elimination History — 淘汰历史
- **[I]** Final Deliverable — 最终方案、架构图、组件清单
- **[A]** Achievement Score — 各维度得分
- **[W]** Next Steps — 实施建议
- **[H]** Historical Trend — (可选) 历史趋势

## License / 许可

Internal use — Hermes Agent Framework
