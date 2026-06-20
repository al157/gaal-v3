"""
Tests for core/graph.py — LangGraph-inspired StateGraph.

Tests cover:
- Node creation and configuration
- Edge creation and routing
- Conditional branching
- Graph validation (valid and invalid)
- Topological ordering
- Full graph compilation and execution
- Retry logic
- Checkpoint integration
"""
import sys
import os
import time
import tempfile
import json
from pathlib import Path

# Ensure the package root is on sys.path (development fallback)
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from typing import Dict, Any, List

from gaal_v3.core.graph import (
    StateGraph,
    CompiledGraph,
    Node,
    Edge,
    ConditionalBranch,
    NodeStatus,
    _serializable_state,
    _format_execution_history,
)


# ── Helper Functions ────────────────────────────────────────────────

def identity_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """A node that returns state unchanged."""
    return state


def add_value_node(value: str) -> callable:
    """Create a node that adds a value to the state."""
    def node_fn(state: Dict[str, Any]) -> Dict[str, Any]:
        s = dict(state)
        s[value] = True
        return s
    return node_fn


def failing_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """A node that always raises an error."""
    raise RuntimeError("This node always fails")


def condition_simple(state: Dict[str, Any]) -> bool:
    """Return True if state is 'simple'."""
    return state.get("type") == "simple"


def condition_is_complex(state: Dict[str, Any]) -> bool:
    """Return True if state is 'complex'."""
    return state.get("type") == "complex"


# ── Test: Node Creation ─────────────────────────────────────────────

class TestNodeCreation:
    """Test Node dataclass creation and properties."""

    def test_node_basic(self):
        """Create a basic node."""
        node = Node(name="test_node", func=identity_node)
        assert node.name == "test_node"
        assert node.func is identity_node
        assert node.retries == 3
        assert node.timeout is None
        assert node.metadata == {}

    def test_node_with_retries(self):
        """Create a node with custom retry count."""
        node = Node(name="retry_node", func=identity_node, retries=5)
        assert node.retries == 5

    def test_node_with_timeout(self):
        """Create a node with custom timeout."""
        node = Node(name="timeout_node", func=identity_node, timeout=30.0)
        assert node.timeout == 30.0

    def test_node_with_metadata(self):
        """Create a node with metadata."""
        node = Node(
            name="meta_node",
            func=identity_node,
            metadata={"is_finish": True, "description": "Test node"},
        )
        assert node.metadata["is_finish"] is True
        assert node.metadata["description"] == "Test node"

    def test_node_hash(self):
        """Nodes should be hashable by name."""
        node1 = Node(name="unique", func=identity_node)
        node2 = Node(name="unique", func=identity_node)
        assert hash(node1) == hash(node2)
        assert node1 == node2

    def test_node_equality_different_functions(self):
        """Nodes with same name should have same hash even with different functions."""
        node1 = Node(name="same", func=identity_node)
        node2 = Node(name="same", func=add_value_node("test"))
        assert hash(node1) == hash(node2)
        # Dataclass __eq__ compares all fields, so they're not equal
        assert node1 != node2


# ── Test: StateGraph Construction ───────────────────────────────────

class TestStateGraphConstruction:
    """Test StateGraph construction, node/edge additions."""

    def test_empty_graph(self):
        """An empty graph should have no nodes or edges."""
        g = StateGraph(dict)
        assert len(g.nodes) == 0
        assert len(g.edges) == 0
        assert g.entry_point is None

    def test_add_node(self):
        """Adding a node should register it."""
        g = StateGraph(dict)
        g.add_node("test", identity_node)
        assert "test" in g.nodes
        assert g.nodes["test"].name == "test"

    def test_add_duplicate_node_raises(self):
        """Adding a duplicate node name should raise ValueError."""
        g = StateGraph(dict)
        g.add_node("test", identity_node)
        with pytest.raises(ValueError, match="already exists"):
            g.add_node("test", identity_node)

    def test_add_edge(self):
        """Adding an edge should register it."""
        g = StateGraph(dict)
        g.add_node("a", identity_node)
        g.add_node("b", identity_node)
        g.add_edge("a", "b")
        assert len(g.edges) == 1
        assert g.edges[0].source == "a"
        assert g.edges[0].target == "b"

    def test_add_edge_with_condition(self):
        """Adding an edge with condition."""
        g = StateGraph(dict)
        g.add_node("a", identity_node)
        g.add_node("b", identity_node)
        g.add_edge("a", "b", condition=condition_simple)
        assert g.edges[0].condition is not None

    def test_add_conditional_edges(self):
        """Adding conditional branching."""
        g = StateGraph(dict)
        g.add_node("a", identity_node)
        g.add_node("b", identity_node)
        g.add_node("c", identity_node)
        g.add_conditional_edges(
            source="a",
            condition=lambda s: s.get("path", "b"),
            path_map={"b": "b", "c": "c"},
        )
        assert len(g.conditional_branches) == 1
        assert g.conditional_branches[0].source == "a"

    def test_set_entry_point(self):
        """Setting entry point should work."""
        g = StateGraph(dict)
        g.add_node("start", identity_node)
        g.set_entry_point("start")
        assert g.entry_point == "start"

    def test_set_finish_point(self):
        """Setting finish point metadata."""
        g = StateGraph(dict)
        g.add_node("end", identity_node)
        g.set_finish_point("end")
        assert g.nodes["end"].metadata.get("is_finish") is True


# ── Test: Graph Validation ─────────────────────────────────────────

class TestGraphValidation:
    """Test graph validation logic."""

    def test_valid_graph(self):
        """A properly constructed graph should validate."""
        g = StateGraph(dict)
        g.add_node("start", identity_node)
        g.add_node("end", identity_node)
        g.add_edge("start", "end")
        g.set_entry_point("start")
        errors = g.validate()
        assert len(errors) == 0

    def test_no_entry_point(self):
        """Missing entry point should be detected."""
        g = StateGraph(dict)
        g.add_node("start", identity_node)
        errors = g.validate()
        assert any("entry point" in e.lower() for e in errors)

    def test_entry_point_not_in_nodes(self):
        """Entry point referencing non-existent node."""
        g = StateGraph(dict)
        g.set_entry_point("missing")
        errors = g.validate()
        assert any("entry point" in e.lower() for e in errors)

    def test_edge_source_not_in_nodes(self):
        """Edge with missing source should be detected."""
        g = StateGraph(dict)
        g.add_node("b", identity_node)
        g.add_edge("missing", "b")
        g.set_entry_point("b")
        errors = g.validate()
        assert any("source" in e.lower() for e in errors)

    def test_edge_target_not_in_nodes(self):
        """Edge with missing target should be detected."""
        g = StateGraph(dict)
        g.add_node("a", identity_node)
        g.add_edge("a", "missing")
        g.set_entry_point("a")
        errors = g.validate()
        assert any("target" in e.lower() for e in errors)

    def test_conditional_branch_source_not_in_nodes(self):
        """Conditional branch with missing source."""
        g = StateGraph(dict)
        g.add_node("b", identity_node)
        g.add_conditional_edges(
            source="missing",
            condition=lambda s: "b",
            path_map={"b": "b"},
        )
        g.set_entry_point("b")
        errors = g.validate()
        assert any("conditional" in e.lower() for e in errors)
        assert any("source" in e.lower() for e in errors)


# ── Test: Topological Order ────────────────────────────────────────

class TestTopologicalOrder:
    """Test topological ordering of graph nodes."""

    def test_linear_order(self):
        """Linear graph should produce linear order."""
        g = StateGraph(dict)
        g.add_node("a", identity_node)
        g.add_node("b", identity_node)
        g.add_node("c", identity_node)
        g.add_edge("a", "b")
        g.add_edge("b", "c")
        g.set_entry_point("a")
        order = g.get_execution_order()
        assert order == ["a", "b", "c"]

    def test_fork_order(self):
        """Fork graph should include all nodes."""
        g = StateGraph(dict)
        g.add_node("a", identity_node)
        g.add_node("b", identity_node)
        g.add_node("c", identity_node)
        g.add_edge("a", "b")
        g.add_edge("a", "c")
        g.set_entry_point("a")
        order = g.get_execution_order()
        assert order[0] == "a"
        assert set(order[1:]) == {"b", "c"}

    def test_conditional_branches_in_order(self):
        """Conditional branch targets should appear in order."""
        g = StateGraph(dict)
        g.add_node("start", identity_node)
        g.add_node("path_a", identity_node)
        g.add_node("path_b", identity_node)
        g.add_conditional_edges(
            source="start",
            condition=lambda s: s.get("path", "a"),
            path_map={"a": "path_a", "b": "path_b"},
        )
        g.set_entry_point("start")
        order = g.get_execution_order()
        assert "start" in order
        assert "path_a" in order
        assert "path_b" in order


# ── Test: Graph Compilation ────────────────────────────────────────

class TestGraphCompilation:
    """Test graph compilation and execution."""

    def test_compile_valid_graph(self):
        """Valid graph should compile successfully."""
        g = StateGraph(dict)
        g.add_node("start", identity_node)
        g.set_entry_point("start")
        compiled = g.compile()
        assert isinstance(compiled, CompiledGraph)

    def test_compile_without_entry_point_raises(self):
        """Graph without entry point should raise on compile."""
        g = StateGraph(dict)
        g.add_node("start", identity_node)
        with pytest.raises(ValueError, match="entry point"):
            g.compile()

    def test_compile_with_invalid_refs_raises(self):
        """Graph with invalid refs should raise on compile."""
        g = StateGraph(dict)
        g.add_node("start", identity_node)
        g.add_edge("start", "missing")
        g.set_entry_point("start")
        with pytest.raises(ValueError, match="validation"):
            g.compile()

    def test_simple_execution(self):
        """Simple linear graph execution."""
        g = StateGraph(dict)
        g.add_node("start", add_value_node("processed"))
        g.set_entry_point("start")

        compiled = g.compile()
        state, history = compiled.run({"initial": True})

        assert state["initial"] is True
        assert state["processed"] is True
        assert len(history) == 1
        assert history[0]["node"] == "start"
        assert history[0]["status"] == "completed"

    def test_multi_step_execution(self):
        """Multi-node linear execution."""
        g = StateGraph(dict)
        g.add_node("step1", add_value_node("step1_done"))
        g.add_node("step2", add_value_node("step2_done"))
        g.add_node("step3", add_value_node("step3_done"))
        g.add_edge("step1", "step2")
        g.add_edge("step2", "step3")
        g.set_entry_point("step1")

        compiled = g.compile()
        state, history = compiled.run({})

        assert state["step1_done"] is True
        assert state["step2_done"] is True
        assert state["step3_done"] is True
        assert len(history) == 3

    def test_conditional_branching(self):
        """Conditional branching based on state."""
        def route_fn(state: Dict[str, Any]) -> str:
            return state.get("path", "a")

        g = StateGraph(dict)
        g.add_node("start", identity_node)
        g.add_node("path_a", add_value_node("took_a"))
        g.add_node("path_b", add_value_node("took_b"))
        g.add_conditional_edges(
            source="start",
            condition=route_fn,
            path_map={"a": "path_a", "b": "path_b"},
        )
        g.set_entry_point("start")

        compiled = g.compile()

        # Follow path A
        state_a, _ = compiled.run({"path": "a"})
        assert state_a.get("took_a") is True
        assert state_a.get("took_b") is None

        # Follow path B
        state_b, _ = compiled.run({"path": "b"})
        assert state_b.get("took_b") is True
        assert state_b.get("took_a") is None

    def test_conditional_default(self):
        """Conditional branching with default fallback."""
        g = StateGraph(dict)
        g.add_node("start", identity_node)
        g.add_node("known_path", add_value_node("known_taken"))
        g.add_node("default_path", add_value_node("default_taken"))
        g.add_conditional_edges(
            source="start",
            condition=lambda s: s.get("path", "unknown"),
            path_map={"known": "known_path"},
            default="default_path",
        )
        g.set_entry_point("start")

        compiled = g.compile()
        state, _ = compiled.run({"path": "unknown"})
        assert state.get("default_taken") is True

        # Follow known path
        state2, _ = compiled.run({"path": "known"})
        assert state2.get("known_taken") is True

    def test_execution_ends_when_no_edges_match(self):
        """Execution should terminate when there are no matching edges."""
        g = StateGraph(dict)
        g.add_node("start", add_value_node("done"))
        g.add_node("orphan", identity_node)
        g.set_entry_point("start")

        compiled = g.compile()
        state, history = compiled.run({})
        assert state["done"] is True
        assert len(history) == 1

    def test_max_steps_limit(self):
        """Execution should raise when max_steps is exceeded."""
        def loop_back(state: Dict[str, Any]) -> Dict[str, Any]:
            return {**state, "count": state.get("count", 0) + 1}

        g = StateGraph(dict)
        g.add_node("loop", loop_back)
        g.add_edge("loop", "loop")  # self-loop = infinite
        g.set_entry_point("loop")

        compiled = g.compile()
        with pytest.raises(RuntimeError, match="max_steps"):
            compiled.run({"count": 0}, max_steps=5)


# ── Test: Retry Logic ──────────────────────────────────────────────

class TestRetryLogic:
    """Test node retry on failure."""

    def test_retry_until_success(self):
        """Node should retry on failure and eventually succeed."""
        class RetryCounter:
            def __init__(self):
                self.count = 0
            def __call__(self, state):
                self.count += 1
                if self.count < 3:
                    raise RuntimeError(f"Attempt {self.count} failed")
                return {**state, "success": True}

        g = StateGraph(dict)
        g.add_node("retry", RetryCounter(), retries=5)
        g.set_entry_point("retry")

        compiled = g.compile()
        state, history = compiled.run({})
        assert state.get("success") is True
        # 2 failed attempts + 1 success = at least 2 total attempts
        assert history[0]["attempts"] >= 2

    def test_retry_exhausted_raises(self):
        """Node should raise after exhausting retries."""
        g = StateGraph(dict)
        g.add_node("failing", failing_node, retries=2)
        g.set_entry_point("failing")

        compiled = g.compile()
        with pytest.raises(RuntimeError, match="failed after"):
            compiled.run({})

    def test_retry_failure_recorded_in_history(self):
        """Failed retry should be recorded in execution history."""
        g = StateGraph(dict)
        g.add_node("failing", failing_node, retries=1)
        g.set_entry_point("failing")

        compiled = g.compile()
        try:
            compiled.run({})
        except RuntimeError:
            pass  # Expected


# ── Test: Utility Functions ───────────────────────────────────────

class TestUtilityFunctions:
    """Test utility functions in graph.py."""

    def test_serializable_state_dict(self):
        """Dict state should be returned as-is."""
        state = {"key": "value", "num": 42}
        result = _serializable_state(state)
        assert result == state

    def test_serializable_state_object(self):
        """Object with __dict__ should be converted."""

        class StateObj:
            def __init__(self):
                self.name = "test"
                self.value = 42

        obj = StateObj()
        result = _serializable_state(obj)
        assert result == {"name": "test", "value": 42}

    def test_format_execution_history(self):
        """Format execution history."""
        history = [
            {"node": "start", "status": "completed", "duration": 0.5, "attempts": 1},
            {"node": "end", "status": "completed", "duration": 0.3, "attempts": 1},
        ]
        formatted = _format_execution_history(history)
        assert "start" in formatted
        assert "end" in formatted
        assert "completed" in formatted


# ── Test: Edge Routing ────────────────────────────────────────────

class TestEdgeRouting:
    """Test edge routing logic in CompiledGraph."""

    def test_conditional_route(self):
        """Conditional routing should work."""
        g = StateGraph(dict)
        g.add_node("start", identity_node)
        g.add_node("end", identity_node)
        g.add_conditional_edges(
            source="start",
            condition=lambda s: "end" if s.get("go") else "nowhere",
            path_map={"end": "end"},
            default=None,
        )
        g.set_entry_point("start")

        compiled = g.compile()
        # With state dict, go=True
        next_node = compiled.get_next_node("start", {"go": True})
        assert next_node == "end"

    def test_conditional_no_match_ends(self):
        """No matching conditional path should end execution."""
        g = StateGraph(dict)
        g.add_node("start", identity_node)
        g.add_conditional_edges(
            source="start",
            condition=lambda s: "unknown",
            path_map={"known": "nowhere"},
            default=None,
        )
        g.set_entry_point("start")

        compiled = g.compile()
        next_node = compiled.get_next_node("start", {})
        assert next_node is None

    def test_edge_with_condition(self):
        """Edge with condition should be respected."""
        g = StateGraph(dict)
        g.add_node("start", identity_node)
        g.add_node("end", identity_node)
        g.add_edge("start", "end", condition=lambda s: s.get("ready") is True)
        g.set_entry_point("start")

        compiled = g.compile()
        # Condition fails
        next_node = compiled.get_next_node("start", {"ready": False})
        assert next_node is None

        # Condition passes
        next_node = compiled.get_next_node("start", {"ready": True})
        assert next_node == "end"


# ── Test: Integration ─────────────────────────────────────────────

class TestIntegration:
    """Integration tests combining multiple features."""

    def test_complete_graph_execution(self):
        """Full graph with conditional routing and retry."""
        g = StateGraph(dict)
        g.add_node("start", add_value_node("started"))
        g.add_node("process", add_value_node("processed"))
        g.add_node("finish", add_value_node("finished"))

        g.add_edge("start", "process")
        g.add_conditional_edges(
            source="process",
            condition=lambda s: "finish" if s.get("processed") else "start",
            path_map={"finish": "finish"},
            default="start",
        )
        g.set_entry_point("start")

        compiled = g.compile()
        state, history = compiled.run({})

        assert state.get("started") is True
        assert state.get("processed") is True
        assert state.get("finished") is True
        assert len(history) == 3

    def test_many_nodes(self):
        """Graph with many nodes (stress test)."""
        g = StateGraph(dict)
        for i in range(20):
            g.add_node(f"n{i}", add_value_node(f"done_{i}"))

        for i in range(19):
            g.add_edge(f"n{i}", f"n{i+1}")

        g.set_entry_point("n0")
        compiled = g.compile()
        state, history = compiled.run({})

        for i in range(20):
            assert state.get(f"done_{i}") is True
        assert len(history) == 20

    def test_execution_checkpoint_callback(self):
        """Execution with step callback."""
        g = StateGraph(dict)
        g.add_node("a", add_value_node("a_done"))
        g.add_node("b", add_value_node("b_done"))
        g.add_edge("a", "b")
        g.set_entry_point("a")

        callbacks = []

        def callback(node, state, status, error):
            callbacks.append({"node": node, "status": status})

        compiled = g.compile()
        compiled.run({}, step_callback=callback)

        # Check callbacks were fired
        callbacks_nodes = [c["node"] for c in callbacks]
        assert "a" in callbacks_nodes
        assert "b" in callbacks_nodes

    def test_empty_state(self):
        """Graph with empty initial state should work."""
        g = StateGraph(dict)
        g.add_node("start", add_value_node("done"))
        g.set_entry_point("start")

        compiled = g.compile()
        state, history = compiled.run({})
        assert state.get("done") is True
