"""
Base agent class for GAAL v3.

Defines the abstract BaseAgent that all GAAL agents extend:
- OrchestratorAgent: Manages the arena loop, delegates to teams
- TeamAgent: Generates proposals for a specific team
- JudgeAgent: Evaluates and scores proposals

All agent work happens via delegate_task sub-sessions to maintain
zero-leak isolation — the main session only receives clean summaries.
"""
from __future__ import annotations
import abc
import json
import logging
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, TypeVar, Generic

logger = logging.getLogger(__name__)


@dataclass
class AgentCapability:
    """Represents a capability or skill an agent can perform.

    Attributes:
        name: Capability name.
        description: What this capability does.
        required_tools: Toolsets needed (e.g., ['terminal', 'web']).
        complexity: Estimated complexity ('simple', 'moderate', 'complex').
    """
    name: str
    description: str
    required_tools: List[str] = field(default_factory=list)
    complexity: str = "moderate"


@dataclass
class AgentMessage:
    """A message from one agent to another.

    Supports structured agent handoff (OpenAI Swarm pattern).

    Attributes:
        sender: Agent name.
        recipient: Target agent name.
        content: Message content (text or structured dict).
        msg_type: Type ('proposal', 'score', 'instruction', 'handoff').
        metadata: Additional context.
    """
    sender: str
    recipient: str
    content: Any
    msg_type: str = "message"
    metadata: Dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])


@dataclass
class AgentContext:
    """Context passed between agents during execution.

    Attributes:
        goal: The overall goal being pursued.
        mode: Execution mode ('lite', 'hard', 'super').
        session_id: Unique session identifier.
        round_num: Current arena round.
        model_config: Model configuration for this agent.
        proposals: Proposals generated so far.
        scores: Current score state.
        config: Full GAAL configuration.
    """
    goal: str
    mode: str = "lite"
    session_id: str = ""
    round_num: int = 0
    model_config: Dict[str, Any] = field(default_factory=dict)
    proposals: List[Dict[str, Any]] = field(default_factory=list)
    scores: Dict[str, Any] = field(default_factory=dict)
    config: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict."""
        return asdict(self)


class BaseAgent(abc.ABC):
    """Abstract base class for all GAAL v3 agents.

    Provides the common interface for:
    - Initialization with context and config
    - Execution with retry and timeout
    - Structured agent handoff (OpenAI Swarm pattern)
    - Clean summary generation (zero-leak)

    Subclasses must implement:
    - execute() — The main agent logic
    - summarize() — Generate a clean summary for the parent session

    Attributes:
        name: Agent name.
        context: Current AgentContext.
        capabilities: List of agent capabilities.
    """

    def __init__(
        self,
        name: str,
        context: Optional[AgentContext] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.name = name
        self.context = context or AgentContext(goal="")
        self.config = config or {}
        self.capabilities: List[AgentCapability] = []
        self._start_time: float = 0.0
        self._handoff_queue: List[AgentMessage] = []

    @abc.abstractmethod
    def execute(self) -> Any:
        """Execute the agent's primary logic.

        This is where the agent does its main work. All LLM calls
        should happen via delegate_task sub-sessions for isolation.

        Returns:
            Agent-specific result (proposal list, score, etc.).
        """
        ...

    @abc.abstractmethod
    def summarize(self) -> Dict[str, Any]:
        """Generate a clean summary for the parent session.

        The summary should contain only what the parent needs to know,
        with no intermediate tool output or process internals.

        Returns:
            Clean summary dict with keys relevant to this agent type.
        """
        ...

    def handoff(self, message: AgentMessage) -> None:
        """Send a structured handoff message to another agent.

        Implements the OpenAI Swarm lightweight agent handoff pattern.
        Messages are queued and processed by the orchestrator.

        Args:
            message: AgentMessage to send.
        """
        self._handoff_queue.append(message)
        logger.debug(
            "Agent '%s' handoff -> '%s': %s",
            message.sender, message.recipient, message.msg_type,
        )

    def get_handoff_messages(self) -> List[AgentMessage]:
        """Retrieve and clear the handoff queue.

        Returns:
            List of pending handoff messages.
        """
        messages = list(self._handoff_queue)
        self._handoff_queue.clear()
        return messages

    def build_delegate_task_context(self) -> str:
        """Build the context string for a delegate_task sub-session.

        This context is passed to the subagent, telling it what to do
        and providing the necessary background information. The subagent
        returns only a clean summary.

        Returns:
            Context string for delegate_task.
        """
        return json.dumps({
            "agent": self.name,
            "goal": self.context.goal,
            "mode": self.context.mode,
            "round": self.context.round_num,
            "model_config": self.context.model_config,
            "proposals_so_far": len(self.context.proposals),
        }, ensure_ascii=False)

    def start_timer(self) -> None:
        """Start the execution timer."""
        self._start_time = time.time()

    @property
    def elapsed_time(self) -> float:
        """Get elapsed execution time in seconds."""
        if self._start_time == 0:
            return 0.0
        return time.time() - self._start_time

    def to_dict(self) -> Dict[str, Any]:
        """Serialize agent to dict for reporting."""
        return {
            "name": self.name,
            "type": type(self).__name__,
            "capabilities": [
                {"name": c.name, "complexity": c.complexity}
                for c in self.capabilities
            ],
        }
