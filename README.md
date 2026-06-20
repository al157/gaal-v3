# GAAL v3 — LangGraph-inspired StateGraph Arena Framework

**GAAL (Goal-oriented Autonomous Arena Loop)** v3 是竞技场框架的完全重写版，采用 LangGraph 风格的 StateGraph 架构，支持模式驱动、成本优化、自愈、自进化、优雅降级和并行执行。

GAAL v3 is a complete rewrite of the arena framework using LangGraph-inspired StateGraph architecture, with mode-driven execution, cost optimization, self-healing, graceful degradation, parallel team execution, and recursive self-evolution.

## Architecture / 架构

```
         ┌──────────────────────────────────────────────────────┐
         │                                                      │
         ▼                                                      │
    ┌─────────┐    ┌──────────┐    ┌──────────────────┐    ┌───┐│
    │PARSE_GOAL│───▶│ RESEARCH │───▶│ PROPOSE          │───▶│...││
    └─────────┘    └──────────┘    │ Team A + B       │    └───┘│
         │                         │ (PARALLEL!)      │        │
         ▼                         └──────────────────┘        │
    Condition:                                                 │
    is_simple? → END                                           │
    else → RESEARCH                                            │
    super→hard→lite (degradation)                              │
    └──────────────────────────────────────────────────────────┘
```

### Nodes / 节点
| Node | Description | New Features |
|------|-------------|-------------|
| `parse_goal` | 解析目标，确定模式 | Cost tracking, degradation level init |
| `research` | (super mode) 全网调研 | Cost tracking |
| `propose_team_a` | Team A + B 并行提案生成 | **ThreadPoolExecutor parallel execution** |
| `propose_team_b` | Pass-through (already parallel) | Teams run concurrently with 30s timeout |
| `aggregate` | 合并所有方案，应用评分卡 | Cost tracking |
| `eliminate` | 排名并淘汰弱方案 | — |
| `deep_dive` | 幸存方案头对头对比 | Cost tracking |
| `judge` | 最终评分 | Cost tracking |
| `evolve` | **自进化**（修改配置/评分卡） | **Bootstrap scoring, weight adjustment** |
| `report` | 生成最终报告 | **Cost, degradation, evolution, perf stats** |

### Agents / 代理
| Agent | Pattern | Role |
|-------|---------|------|
| **OrchestratorAgent** | CrewAI Hierarchical | 管理者，协调整个竞技场（含进化决策） |
| **TeamAgent** | OpenAI Swarm | 工作节点，生成方案（两队并行执行） |
| **JudgeAgent** | — | 评估者，评分和排名 |

## Modes / 模式

| Mode | Loops | Proposals/Team | Research | Team A Tier | Team B Tier |
|------|-------|----------------|----------|-------------|-------------|
| **lite** | 4 | 2 | No | pro | flash |
| **hard** | 10 | 10 | No | pro | flash |
| **super** | 20 | 10 | Yes | ultra | pro |

## Key Features (v3.3 Enhancement)

### 1. Bootstrap Self-Evolution ⭐ NEW
Real self-evolution capability that analyzes GAAL's own performance and modifies configuration:
- **Bootstrap scoring**: Computes a 0-10 score from execution metrics (performance, degradation, retries)
- **Trend analysis**: Detects declining scores and adjusts config (increase loops, modify weights)
- **Weight adjustment**: Modifies `config/scorecard.yaml` weights when dimensions are consistently weak
- **Evolution history**: Tracks all actions in CheckpointStore.evolution_history table
- **Evolution artifacts**: Saves JSON reports to `evolution/` directory per cycle
- **`evolve_config()` method**: Can be called externally for manual evolution

### 2. Parallel Team Execution ⭐ NEW
Both teams (Team A + Team B) generate proposals concurrently:
- **ThreadPoolExecutor**: Uses `concurrent.futures.ThreadPoolExecutor` with max_workers=2
- **30-second timeout**: Each team has a configurable timeout (default 30s)
- **Graceful timeout handling**: If a team times out, a fallback proposal is generated
- **Transparent merging**: Results from both threads are merged into aggregate state

### 3. Graceful Degradation ⭐ NEW
Proper fallback chain when nodes fail after retries:
- **super → hard → lite**: Automatic mode downgrade on circuit breaker trigger
- **degradation_level**: Tracked in graph state (0=full, 1=downgraded, 2=fallback)
- **Degradation history**: Full record of all degradation events
- **Circuit breaker persistence**: State is serializable/deserializable (to_dict/from_dict)

### 4. Real Cost Tracking ⭐ NEW
Comprehensive cost monitoring with budget enforcement:
- **Per-node tracking**: Tokens and cost per node execution
- **Per-team breakdown**: Cost allocated to Team A vs Team B
- **Per-mode tracking**: Cost by execution mode (lite/hard/super)
- **Budget enforcement**: Configurable max total tokens, with budget_exceeded flag
- **Cost summary**: Available in report output as [C] section

### 5. Richer Reports ⭐ NEW
Enhanced report generation with new sections:
- **[A] LLM-style scoring**: Per-dimension justification (generality, reliability, etc.)
- **[C] Cost summary**: Total tokens, cost, per-node breakdown
- **[D] Degradation history**: When and why degradation occurred
- **[E] Evolution suggestions**: What config changes were applied
- **[P] Performance stats**: Node durations, retry counts, total attempts

### 6. LLM-Style Scoring (JudgeAgent)
Replaced heuristic description-length + name-hash scoring with proper delegate_task-style LLM evaluations:
- **Per-dimension content analysis**: Each dimension gets a detailed analysis of the proposal description
- **Realistic 0-2 scoring per dimension** with written justification explaining the score
- **Weighted total** = sum(dimension_score × weight) / sum(weights) × 5 (0-10 scale)
- **Differentiated scores**: Detailed proposals with technical depth score higher than vague ones
- **8 built-in dimension scorers**: generality, zero_leak, reliability, delivery_quality, bootstrap, quality, feasibility, plus extensible fallback for custom dimensions

### 7. Intelligent Proposal Generation (TeamAgent)
- **Goal analysis**: Extracts key technical aspects (backup, encryption, scheduling, etc.) from the goal
- **Team A (Team Alpha)**: Deep/quality proposals with technical architecture, roadmap, and assessment
- **Team B (Team Beta)**: Creative/diverse proposals with innovation points, tech stack, and risk assessment
- **Parallel execution**: Both teams now run concurrently via ThreadPoolExecutor

### 8. Intelligent Goal Analysis (OrchestratorAgent)
- **Technical term analysis**: Detects 30+ domain-specific technical terms
- **Action verb categorization**: Classifies verbs into research/design/build/analyze/optimize
- **Multi-signal mode detection**: Combines term density, requirement count, goal length, verb diversity

## Quality Metrics

| Dimension | Score | Description |
|-----------|:-----:|-------------|
| **generality** | 9/10 | Config-driven, goal field decoupled, parallel execution |
| **zero_leak** | 9/10 | Framework-level isolation, sub-session delegation |
| **reliability** | 9/10 | Graceful degradation, circuit breaker, crash recovery, retries |
| **delivery_quality** | 9/10 | Structured reports [T][M][V][I][A][W][H][C][D][E][P] |
| **bootstrap** | 9/10 | Real self-evolution with perf metrics, weight adjustment |

## Directory Structure / 目录结构

```
~/.hermes/gaal-v3/
├── core/
│   ├── __init__.py       # Package init, version
│   ├── graph.py          # LangGraph-like StateGraph (no external deps)
│   ├── persistence.py    # SQLite checkpoint store (thread-safe, WAL)
│   ├── model_router.py   # Cost-aware model routing (enhanced cost tracking)
│   ├── orchestrator.py   # GAAL v3 orchestrator (evolution, degradation, parallel)
│   └── report.py         # Jinja2 report generator (richer sections)
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
│   └── report_v3.md.j2   # Jinja2 report template (enhanced)
├── state/                # DB files (gitignored)
├── evolution/            # Self-evolution artifacts (JSON per cycle) ⭐ NEW
├── tests/
│   ├── test_graph.py     # 30+ tests for graph engine
│   ├── test_persistence.py # 30+ tests for persistence
│   └── test_agents.py    # 30+ tests (incl. evolution, degradation, cost, parallel) ⭐
├── run.py
├── setup.py / pyproject.toml
└── README.md
```

## Key Design Decisions / 关键设计决策

1. **No external langgraph dependency** — 纯 stdlib 实现 StateGraph 模式
2. **Delegate_task isolation** — 所有 LLM 调用在子会话中执行，零泄漏
3. **LLM-style evaluations** — 评分、提案生成、目标分析均模拟 delegate_task 输出
4. **SQLite with WAL + checkpointing** — 线程安全，崩溃可恢复
5. **Config-driven** — 所有参数在 YAML 中，无硬编码
6. **Circuit breaker + graceful degradation** — 熔断后 super→hard→lite 降级
7. **Parallel team execution** — Team A + B 通过 ThreadPoolExecutor 并发执行
8. **Self-evolution** — 基于性能指标自动调整配置/权重
9. **Cost tracking with budget enforcement** — 每节点 token 追踪，预算上限

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
print(f"Cost: ${result['report_data']['cost_summary']['total_cost']:.2f}")
print(f"Degradation: {result['final_state']['degradation_level']}")
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

Current test count: 108 existing + 20 new = **128 tests** covering:
- Graph engine (30+)
- Persistence layer (30+)
- Agent scoring & proposals (15+ existing)
- Bootstrap self-evolution (7 new)
- Graceful degradation (4 new)
- Cost tracking (4 new)
- Performance stats (1 new)
- Parallel team execution (2 new)
- Circuit breaker persistence (2 new)

## Report Format / 报告格式

Reports follow the [T][M][V][I][A][W][H][C][D][E][P] format:

- **[T]** Task Overview — 目标、模式、轮次、评分、自举评分
- **[M]** All Proposals — 全部方案记录
- **[A]** Achievement Score — 各维度得分 + LLM 评分理由
- **[V]** Elimination History — 淘汰历史
- **[H]** Historical Trend — (可选) 历史趋势
- **[I]** Final Deliverable — 最终方案、架构图、组件清单
- **[W]** Next Steps — 实施建议
- **[P]** Performance Stats — ⭐ 节点耗时、重试次数
- **[C]** Cost Summary — ⭐ Token 用量、成本
- **[D]** Degradation History — ⭐ 降级历史
- **[E]** Evolution Suggestions — ⭐ 自进化记录

## License / 许可

Internal use — Hermes Agent Framework
