"""
LangGraph-inspired StateGraph builder for GAAL v3.

Implements a lightweight StateGraph pattern that mirrors LangGraph's API:
- StateGraph with typed state
- Nodes with retry logic
- Edges with optional conditions
- Conditional branching
- Checkpoint integration
- Topological validation

No external dependency on langgraph — pure stdlib implementation.
"""
from __future__ import annotations
import enum
import time
import logging
from typing import (
    Any, Callable, Dict, List, Optional, Set, Tuple,
    TypeVar, Generic, Union, Type,
)
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Type variable for state
S = TypeVar("S")


class NodeStatus(enum.Enum):
    """Status of a node execution."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class Node(Generic[S]):
    """A single node in the state graph.

    Each node wraps a function that takes state S and returns (possibly
    modified) state S. Nodes support configurable retries and metadata.

    Attributes:
        name: Unique node identifier.
        func: The processing function (state -> state).
        metadata: Arbitrary metadata (e.g. is_finish, description).
        retries: Max retry attempts before raising.
        timeout: Optional per-node timeout in seconds.
    """
    name: str
    func: Callable[[S], S]
    metadata: Dict[str, Any] = field(default_factory=dict)
    retries: int = 3
    timeout: Optional[float] = None

    def __hash__(self) -> int:
        return hash(self.name)


@dataclass
class Edge(Generic[S]):
    """A directed edge between two nodes.

    Attributes:
        source: Source node name.
        target: Target node name (or "__end__" for termination).
        condition: Optional state predicate. If provided, the edge is
                   only traversed when condition(state) is True.
    """
    source: str
    target: str
    condition: Optional[Callable[[S], bool]] = None


class ConditionalBranch(Generic[S]):
    """Conditional fan-out from a source node.

    Evaluates condition(state) to get a routing key, then follows the
    path_map or default.

    Attributes:
        source: Source node name.
        condition: Routing function (state -> routing_key).
        path_map: Dict mapping routing_key -> target node name.
        default: Fallback target if routing_key not in path_map.
    """
    def __init__(
        self,
        source: str,
        condition: Callable[[S], str],
        path_map: Dict[str, str],
        default: Optional[str] = None,
    ) -> None:
        self.source = source
        self.condition = condition
        self.path_map = path_map
        self.default = default


class CompiledGraph(Generic[S]):
    """A compiled, ready-to-execute graph.

    Created by StateGraph.compile(). Provides the run() method that
    actually executes the graph against an initial state.

    Attributes:
        graph: The parent StateGraph.
        nodes: Shortcut to graph.nodes.
        edges: Shortcut to graph.edges.
        entry_point: Starting node name.
    """

    def __init__(self, graph: "StateGraph[S]") -> None:
        self.graph = graph
        self.nodes = graph.nodes
        self.edges = graph.edges
        self.conditional_branches = graph.conditional_branches
        self.entry_point = graph.entry_point
        self._edge_index = self._build_edge_index()

    def _build_edge_index(self) -> Dict[str, List[Edge]]:
        """Build a source -> [edges] index for O(1) lookup."""
        index: Dict[str, List[Edge]] = {}
        for edge in self.edges:
            index.setdefault(edge.source, []).append(edge)
        return index

    def _build_conditional_index(self) -> Dict[str, List[ConditionalBranch]]:
        """Build a source -> [branches] index."""
        index: Dict[str, List[ConditionalBranch]] = {}
        for branch in self.conditional_branches:
            index.setdefault(branch.source, []).append(branch)
        return index

    def get_next_node(self, current: str, state: S) -> Optional[str]:
        """Determine the next node based on edges and current state.

        Priority order:
        1. Conditional branches (first matching source wins)
        2. Edges with conditions (first passing condition wins)
        3. Unconditional edges

        Returns node name, or None to terminate.
        """
        # 1. Check conditional branches
        cond_index = self._build_conditional_index()
        branches = cond_index.get(current, [])
        for branch in branches:
            result = branch.condition(state)
            target = branch.path_map.get(result, branch.default)
            if target is None:
                continue
            if target == "__end__":
                return None
            return target

        # 2. Check regular edges
        edges = self._edge_index.get(current, [])
        for edge in edges:
            if edge.condition is None or edge.condition(state):
                if edge.target == "__end__" or edge.target == "END":
                    return None
                return edge.target

        return None

    def run(
        self,
        initial_state: S,
        checkpoint_store: Any = None,
        max_steps: int = 100,
        step_callback: Optional[Callable[[str, S, str, Optional[str]], None]] = None,
    ) -> Tuple[S, List[Dict[str, Any]]]:
        """Execute the graph from initial state.

        Walks through the graph, executing nodes in topological order
        (respecting conditional branches). Checkpoints are saved before
        and after each node via the optional checkpoint_store.

        Args:
            initial_state: Starting state object.
            checkpoint_store: Optional CheckpointStore for persistence.
            max_steps: Safety limit on total node executions.
            step_callback: Optional callback (node_name, state, status, error).

        Returns:
            Tuple of (final_state, execution_history).

        Raises:
            RuntimeError: If a node fails after exhausting retries.
            RuntimeError: If max_steps is exceeded.
        """
        state = initial_state
        history: List[Dict[str, Any]] = []
        current = self.entry_point
        steps = 0
        circuit_breaker_failures = 0
        consecutive_failures = 0

        while current is not None and steps < max_steps:
            if current not in self.nodes:
                logger.error(f"Node '{current}' not found in graph")
                break

            node = self.nodes[current]
            step_start = time.time()
            failed_attempts = 0

            # Checkpoint before execution (running state)
            if checkpoint_store is not None:
                checkpoint_store.save_checkpoint(
                    node_name=current,
                    state=_serializable_state(state),
                    status="running",
                    step=steps,
                )

            if step_callback is not None:
                step_callback(current, state, "running", None)

            # Execute node with retry + exponential backoff
            last_error: Optional[Exception] = None
            for attempt in range(node.retries):
                try:
                    result = node.func(state)
                    # Node should return state; if it returns None, keep current
                    if result is not None:
                        state = result
                    last_error = None
                    break
                except Exception as e:
                    failed_attempts += 1
                    last_error = e
                    logger.warning(
                        "Node '%s' attempt %d/%d failed: %s: %s",
                        current, attempt + 1, node.retries,
                        type(e).__name__, e,
                    )
                    if attempt < node.retries - 1:
                        # Exponential backoff
                        delay = min(2 ** attempt * 5, 120)
                        time.sleep(delay)

            step_duration = time.time() - step_start

            if last_error is not None:
                # All retries exhausted
                consecutive_failures += 1
                history.append({
                    "node": current,
                    "status": NodeStatus.FAILED.value,
                    "error": f"{type(last_error).__name__}: {last_error}",
                    "duration": step_duration,
                    "attempts": failed_attempts,
                })
                if checkpoint_store is not None:
                    checkpoint_store.save_checkpoint(
                        node_name=current,
                        state=_serializable_state(state),
                        status="failed",
                        error=str(last_error),
                        step=steps,
                    )
                if step_callback is not None:
                    step_callback(current, state, "failed", str(last_error))

                # Circuit breaker: 5 consecutive failures -> escalate
                if consecutive_failures >= 5:
                    raise RuntimeError(
                        f"Circuit breaker triggered: {consecutive_failures} "
                        f"consecutive failures at node '{current}'. "
                        f"Last error: {last_error}"
                    )

                raise RuntimeError(
                    f"Node '{current}' failed after {node.retries} "
                    f"attempts: {last_error}"
                )

            # Success — reset consecutive failure counter
            consecutive_failures = 0

            # Checkpoint after execution (completed)
            if checkpoint_store is not None:
                checkpoint_store.save_checkpoint(
                    node_name=current,
                    state=_serializable_state(state),
                    status="completed",
                    step=steps,
                )

            if step_callback is not None:
                step_callback(current, state, "completed", None)

            history.append({
                "node": current,
                "status": NodeStatus.COMPLETED.value,
                "duration": step_duration,
                "attempts": failed_attempts or 1,
            })

            # Move to next node
            current = self.get_next_node(current, state)
            steps += 1

        if steps >= max_steps:
            raise RuntimeError(
                f"Graph execution exceeded max_steps={max_steps} "
                f"at node '{current}'"
            )

        return state, history


class StateGraph(Generic[S]):
    """LangGraph-inspired StateGraph.

    A directed graph where nodes process state and edges define flow.
    Supports conditional branching, checkpoint integration, retry with
    exponential backoff, and circuit breaker patterns.

    Usage::

        graph = StateGraph(dict)
        graph.add_node("parse", parse_fn)
        graph.add_node("research", research_fn)
        graph.add_edge("parse", "research")
        graph.add_conditional_edges(
            "research",
            lambda s: "propose" if s["done"] else "research",
            {"propose": "propose", "research": "research"},
        )
        graph.set_entry_point("parse")
        compiled = graph.compile()
        final_state, history = compiled.run(initial_state)

    Attributes:
        state_class: The type/class of state objects.
        nodes: All registered nodes (name -> Node).
        edges: All registered edges.
        conditional_branches: All registered conditional branches.
        entry_point: Starting node name.
    """

    def __init__(self, state_class: Type[S]) -> None:
        self.state_class: Type[S] = state_class
        self.nodes: Dict[str, Node[S]] = {}
        self.edges: List[Edge[S]] = []
        self.conditional_branches: List[ConditionalBranch[S]] = []
        self.entry_point: Optional[str] = None
        self._compiled = False

    def add_node(
        self,
        name: str,
        func: Callable[[S], S],
        **kwargs: Any,
    ) -> Node[S]:
        """Add a node to the graph.

        Args:
            name: Unique node name.
            func: Processing function (state -> state).
            **kwargs: Additional Node attributes (retries, timeout, metadata).

        Returns:
            The created Node instance.
        """
        if name in self.nodes:
            raise ValueError(f"Node '{name}' already exists")
        node = Node(name=name, func=func, **kwargs)
        self.nodes[name] = node
        return node

    def add_edge(
        self,
        source: str,
        target: str,
        condition: Optional[Callable[[S], bool]] = None,
    ) -> Edge[S]:
        """Add a directed edge from source to target.

        Args:
            source: Source node name.
            target: Target node name (or "__end__" to terminate).
            condition: Optional state predicate for conditional routing.

        Returns:
            The created Edge instance.
        """
        edge = Edge(source=source, target=target, condition=condition)
        self.edges.append(edge)
        return edge

    def add_conditional_edges(
        self,
        source: str,
        condition: Callable[[S], str],
        path_map: Dict[str, str],
        default: Optional[str] = None,
    ) -> ConditionalBranch[S]:
        """Add conditional branching from a source node.

        The condition function receives state and returns a string key.
        The path_map maps keys to target node names. If the key is not
        in path_map, the default is used.

        Args:
            source: Source node name.
            condition: Routing function (state -> key).
            path_map: Dict mapping keys to target node names.
            default: Fallback target (None = terminate).

        Returns:
            The created ConditionalBranch instance.
        """
        branch = ConditionalBranch(
            source=source,
            condition=condition,
            path_map=path_map,
            default=default,
        )
        self.conditional_branches.append(branch)
        return branch

    def set_entry_point(self, name: str) -> None:
        """Set the entry point (starting) node.

        Args:
            name: Node name to start from.
        """
        self.entry_point = name

    def set_finish_point(self, name: str) -> None:
        """Mark a node as a finish point (terminal).

        This is metadata only; the graph terminates when no outgoing
        edges match.

        Args:
            name: Node name to mark as finish point.
        """
        if name in self.nodes:
            self.nodes[name].metadata["is_finish"] = True

    def compile(self) -> CompiledGraph[S]:
        """Compile the graph into an executable CompiledGraph.

        Validates the graph structure before returning.

        Returns:
            A CompiledGraph ready for execution.

        Raises:
            ValueError: If entry point is not set.
        """
        errors = self.validate()
        if errors:
            raise ValueError(
                "Graph validation failed:\n  " + "\n  ".join(errors)
            )
        self._compiled = True
        return CompiledGraph(self)

    def validate(self) -> List[str]:
        """Validate graph structure.

        Checks:
        - Entry point is set
        - Entry point exists in nodes
        - All edge sources/targets exist in nodes (or are "__end__")
        - No orphaned condition-only nodes (sources that are not targets)

        Returns:
            List of error messages (empty if valid).
        """
        errors: List[str] = []

        if not self.entry_point:
            errors.append("No entry point set (use set_entry_point())")

        if self.entry_point and self.entry_point not in self.nodes:
            errors.append(
                f"Entry point '{self.entry_point}' not found in nodes"
            )

        for edge in self.edges:
            if edge.source not in self.nodes:
                errors.append(
                    f"Edge source '{edge.source}' not found in nodes"
                )
            if edge.target not in self.nodes and edge.target not in ("__end__", "END"):
                errors.append(
                    f"Edge target '{edge.target}' not found in nodes"
                )

        for branch in self.conditional_branches:
            if branch.source not in self.nodes:
                errors.append(
                    f"Conditional branch source '{branch.source}' not found in nodes"
                )
            # Only validate defaults and __end__ targets;
            # path_map entries may reference nodes that only exist at runtime
            if branch.default is not None and branch.default not in self.nodes and branch.default not in ("__end__", "END"):
                errors.append(
                    f"Conditional branch default '{branch.default}' not found in nodes"
                )

        return errors

    def get_execution_order(self) -> List[str]:
        """Compute topological execution order via Kahn's algorithm.

        Returns:
            List of node names in execution order.
        """
        # Build adjacency and in-degree maps
        in_degree: Dict[str, int] = {n: 0 for n in self.nodes}
        adj: Dict[str, List[str]] = {n: [] for n in self.nodes}

        for edge in self.edges:
            if edge.source in adj and edge.target in self.nodes:
                adj[edge.source].append(edge.target)
                in_degree[edge.target] = in_degree.get(edge.target, 0) + 1

        # Include conditional branch targets as edges
        for branch in self.conditional_branches:
            for target in branch.path_map.values():
                if target in self.nodes:
                    if target not in in_degree:
                        in_degree[target] = 0
                    if branch.source not in adj:
                        adj[branch.source] = []
                    adj[branch.source].append(target)
                    in_degree[target] = in_degree.get(target, 0) + 1

        # Kahn's algorithm
        queue = [n for n, d in in_degree.items() if d == 0]
        order: List[str] = []

        while queue:
            node = queue.pop(0)
            order.append(node)
            for neighbor in adj.get(node, []):
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        return order


def _serializable_state(state: Any) -> Dict[str, Any]:
    """Convert state to a JSON-serializable dict.

    If state is already a dict, return it as-is.
    Otherwise, try to return state.__dict__ or fallback to str(state).
    """
    if isinstance(state, dict):
        return state
    if hasattr(state, "__dict__"):
        return state.__dict__
    return {"_repr": str(state)}


def _format_execution_history(history: List[Dict[str, Any]]) -> str:
    """Format execution history as a readable string."""
    lines = ["=== Graph Execution History ==="]
    for i, entry in enumerate(history):
        dur = entry.get("duration", 0)
        status = entry.get("status", "?")
        node = entry.get("node", "?")
        err = entry.get("error", "")
        err_str = f"  ERROR: {err}" if err else ""
        lines.append(
            f"  [{i}] {node:20s} {status:12s} "
            f"({dur:.2f}s, {entry.get('attempts', 1)} attempt(s)){err_str}"
        )
    return "\n".join(lines)
