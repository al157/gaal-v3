"""
SQLite checkpoint store with thread-safe WAL mode for GAAL v3.

Implements LangGraph-style checkpointing with:
- Thread-safe SQLite with WAL journal mode
- Checkpoint per node transition
- Proposal/elimination/evolution tracking
- Crash recovery by replaying last consistent state
- Proper connection management with thread-local connections

Unlike GAAL v2's naive WAL-based state machine, this implements proper
graph-oriented checkpointing with full node state capture, execution
session management, and evolution history tracking.
"""
from __future__ import annotations
import json
import logging
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    import sqlite3
except ImportError:
    sqlite3 = None  # type: ignore


class CheckpointStore:
    """Thread-safe SQLite checkpoint store with WAL mode.

    Provides checkpoint persistence for LangGraph-style state graphs.
    Each node transition is checkpointed with full state capture, enabling
    crash recovery and execution replay.

    Thread safety is achieved via:
    - Thread-local connections (no sharing)
    - A threading.Lock for DDL and write transactions

    Attributes:
        db_path: Path to the SQLite database file.
        session_id: Unique identifier for the current execution session.
    """

    def __init__(
        self,
        db_path: str = "state/gaal_v3_checkpoints.db",
        session_id: Optional[str] = None,
        auto_init: bool = True,
    ) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.session_id = session_id or str(uuid.uuid4())[:8]
        self._local = threading.local()
        self._lock = threading.Lock()
        self._closed = False
        if auto_init:
            self.init_db()

    def _get_conn(self) -> "sqlite3.Connection":
        """Get thread-local database connection.

        Each thread gets its own connection to avoid sharing issues.
        Connections use WAL mode and a 5-second busy timeout.
        """
        if not hasattr(self._local, "conn") or self._local.conn is None:
            if sqlite3 is None:
                raise ImportError("sqlite3 is required for CheckpointStore")
            conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=5000")
            conn.execute("PRAGMA foreign_keys=ON")
            conn.row_factory = sqlite3.Row
            self._local.conn = conn
        return self._local.conn

    def init_db(self) -> None:
        """Initialize database schema.

        Creates tables for checkpoints, proposals, eliminations,
        graph executions, and evolution history. Schema is idempotent
        (CREATE TABLE IF NOT EXISTS).
        """
        with self._lock:
            conn = self._get_conn()

            # Use individual execute() calls for better compatibility
            statements = [
                "CREATE TABLE IF NOT EXISTS checkpoints ("
                "id INTEGER PRIMARY KEY AUTOINCREMENT, "
                "session_id TEXT NOT NULL, "
                "node_name TEXT NOT NULL, "
                "status TEXT NOT NULL DEFAULT 'running', "
                "state_json TEXT NOT NULL DEFAULT '{}', "
                "error TEXT, "
                "step INTEGER NOT NULL DEFAULT 0, "
                "created_at REAL NOT NULL DEFAULT (julianday('now')), "
                "UNIQUE(session_id, node_name, step)"
                ")",

                "CREATE TABLE IF NOT EXISTS proposals ("
                "id INTEGER PRIMARY KEY AUTOINCREMENT, "
                "session_id TEXT NOT NULL, "
                "team TEXT NOT NULL, "
                "round INTEGER NOT NULL, "
                "\"index\" INTEGER NOT NULL, "
                "name TEXT NOT NULL, "
                "description TEXT, "
                "scores_json TEXT DEFAULT '{}', "
                "score REAL DEFAULT 0.0, "
                "status TEXT DEFAULT 'active', "
                "created_at REAL DEFAULT (julianday('now'))"
                ")",

                "CREATE TABLE IF NOT EXISTS eliminations ("
                "id INTEGER PRIMARY KEY AUTOINCREMENT, "
                "session_id TEXT NOT NULL, "
                "proposal_id INTEGER, "
                "round INTEGER NOT NULL, "
                "reason TEXT, "
                "scores_json TEXT DEFAULT '{}', "
                "created_at REAL DEFAULT (julianday('now'))"
                ")",

                "CREATE TABLE IF NOT EXISTS graph_executions ("
                "id INTEGER PRIMARY KEY AUTOINCREMENT, "
                "session_id TEXT NOT NULL, "
                "goal TEXT, "
                "mode TEXT, "
                "status TEXT DEFAULT 'running', "
                "config_json TEXT DEFAULT '{}', "
                "result_json TEXT DEFAULT '{}', "
                "started_at REAL, "
                "completed_at REAL, "
                "error TEXT"
                ")",

                "CREATE TABLE IF NOT EXISTS evolution_history ("
                "id INTEGER PRIMARY KEY AUTOINCREMENT, "
                "session_id TEXT NOT NULL, "
                "action TEXT NOT NULL, "
                "target TEXT NOT NULL, "
                "before_json TEXT, "
                "after_json TEXT, "
                "score_before REAL, "
                "score_after REAL, "
                "created_at REAL DEFAULT (julianday('now'))"
                ")",

                "CREATE INDEX IF NOT EXISTS idx_checkpoints_session ON checkpoints(session_id, step)",
                "CREATE INDEX IF NOT EXISTS idx_proposals_session ON proposals(session_id)",
                "CREATE INDEX IF NOT EXISTS idx_eliminations_session ON eliminations(session_id)",
                "CREATE INDEX IF NOT EXISTS idx_executions_session ON graph_executions(session_id)",
                "CREATE INDEX IF NOT EXISTS idx_evolution_session ON evolution_history(session_id)",
            ]

            for sql in statements:
                conn.execute(sql)
            conn.commit()

    # ── Checkpoint Operations ──────────────────────────────────────

    def save_checkpoint(
        self,
        node_name: str,
        state: Dict[str, Any],
        status: str = "running",
        error: Optional[str] = None,
        step: int = 0,
    ) -> int:
        """Save a checkpoint for a node execution.

        Args:
            node_name: Name of the node being checkpointed.
            state: Serialized state dict.
            status: One of 'running', 'completed', 'failed'.
            error: Error message if failed.
            step: Execution step counter.

        Returns:
            Checkpoint ID.
        """
        with self._lock:
            conn = self._get_conn()
            now = time.time()
            state_json = json.dumps(state, default=str, ensure_ascii=False)
            conn.execute(
                """INSERT OR REPLACE INTO checkpoints
                   (session_id, node_name, status, state_json, error, step, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (self.session_id, node_name, status, state_json, error, step, now),
            )
            conn.commit()
            cursor = conn.execute("SELECT last_insert_rowid()")
            return cursor.fetchone()[0]

    def get_checkpoint(self, checkpoint_id: int) -> Optional[Dict[str, Any]]:
        """Get a specific checkpoint by ID."""
        conn = self._get_conn()
        cur = conn.execute(
            "SELECT * FROM checkpoints WHERE id = ?",
            (checkpoint_id,),
        )
        row = cur.fetchone()
        return dict(row) if row else None

    def get_last_checkpoint(self) -> Optional[Dict[str, Any]]:
        """Get the most recent checkpoint for this session.

        Returns:
            Checkpoint dict with keys: node_name, status, state, error, step.
            None if no checkpoints exist.
        """
        conn = self._get_conn()
        cur = conn.execute(
            """SELECT node_name, status, state_json, error, step
               FROM checkpoints
               WHERE session_id = ?
               ORDER BY id DESC LIMIT 1""",
            (self.session_id,),
        )
        row = cur.fetchone()
        if row is None:
            return None
        return {
            "node_name": row["node_name"],
            "status": row["status"],
            "state": json.loads(row["state_json"]),
            "error": row["error"],
            "step": row["step"],
        }

    def get_execution_chain(self) -> List[Dict[str, Any]]:
        """Get the full execution chain for this session, ordered by step.

        Returns:
            List of checkpoint dicts in execution order.
        """
        conn = self._get_conn()
        cur = conn.execute(
            """SELECT node_name, status, state_json, error, step
               FROM checkpoints
               WHERE session_id = ?
               ORDER BY step ASC, id ASC""",
            (self.session_id,),
        )
        return [
            {
                "node_name": r["node_name"],
                "status": r["status"],
                "state": json.loads(r["state_json"]),
                "error": r["error"],
                "step": r["step"],
            }
            for r in cur.fetchall()
        ]

    def get_checkpoints_by_node(self, node_name: str) -> List[Dict[str, Any]]:
        """Get all checkpoints for a specific node."""
        conn = self._get_conn()
        cur = conn.execute(
            """SELECT * FROM checkpoints
               WHERE session_id = ? AND node_name = ?
               ORDER BY step ASC""",
            (self.session_id, node_name),
        )
        return [dict(r) for r in cur.fetchall()]

    # ── Proposal Operations ────────────────────────────────────────

    def add_proposal(
        self,
        team: str,
        round_num: int,
        index: int,
        name: str,
        description: str,
        scores: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Register a proposal from a team.

        Args:
            team: Team name (e.g., 'team_a', 'team_b').
            round_num: Current arena round.
            index: Proposal index within the team.
            name: Short proposal name.
            description: Proposal description.
            scores: Optional initial scores dict.

        Returns:
            The proposal's auto-generated ID.
        """
        with self._lock:
            conn = self._get_conn()
            scores_json = json.dumps(scores or {}, default=str)
            cursor = conn.execute(
                """INSERT INTO proposals
                   (session_id, team, round, "index", name, description, scores_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (self.session_id, team, round_num, index, name, description, scores_json),
            )
            conn.commit()
            return cursor.lastrowid

    def update_proposal_score(
        self,
        proposal_id: int,
        score: float,
        scores: Dict[str, Any],
    ) -> None:
        """Update a proposal's score after evaluation."""
        with self._lock:
            conn = self._get_conn()
            conn.execute(
                """UPDATE proposals
                   SET score = ?, scores_json = ?
                   WHERE id = ?""",
                (score, json.dumps(scores, default=str), proposal_id),
            )
            conn.commit()

    def update_proposal_status(
        self,
        proposal_id: int,
        status: str,
    ) -> None:
        """Update a proposal's status (e.g., 'eliminated')."""
        with self._lock:
            conn = self._get_conn()
            conn.execute(
                "UPDATE proposals SET status = ? WHERE id = ?",
                (status, proposal_id),
            )
            conn.commit()

    def get_proposals(
        self,
        status: Optional[str] = None,
        team: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get proposals for the current session.

        Args:
            status: Optional filter ('active', 'eliminated').
            team: Optional filter by team name.

        Returns:
            List of proposal dicts.
        """
        conn = self._get_conn()
        query = "SELECT * FROM proposals WHERE session_id = ?"
        params: List[Any] = [self.session_id]

        if status is not None:
            query += " AND status = ?"
            params.append(status)
        if team is not None:
            query += " AND team = ?"
            params.append(team)

        query += ' ORDER BY round ASC, team ASC, "index" ASC'
        cur = conn.execute(query, params)
        return [dict(r) for r in cur.fetchall()]

    def count_proposals(self, status: Optional[str] = None) -> int:
        """Count proposals in the current session."""
        conn = self._get_conn()
        if status:
            cur = conn.execute(
                "SELECT COUNT(*) FROM proposals WHERE session_id = ? AND status = ?",
                (self.session_id, status),
            )
        else:
            cur = conn.execute(
                "SELECT COUNT(*) FROM proposals WHERE session_id = ?",
                (self.session_id,),
            )
        return cur.fetchone()[0]

    # ── Elimination Operations ─────────────────────────────────────

    def record_elimination(
        self,
        proposal_id: int,
        round_num: int,
        reason: str,
        scores: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Record a proposal elimination.

        Also updates the proposal's status to 'eliminated'.

        Args:
            proposal_id: The proposal being eliminated.
            round_num: Current elimination round.
            reason: Why it was eliminated.
            scores: Scores at time of elimination.

        Returns:
            Elimination record ID.
        """
        with self._lock:
            conn = self._get_conn()
            conn.execute(
                "UPDATE proposals SET status = 'eliminated' WHERE id = ?",
                (proposal_id,),
            )
            scores_json = json.dumps(scores or {}, default=str)
            cursor = conn.execute(
                """INSERT INTO eliminations
                   (session_id, proposal_id, round, reason, scores_json)
                   VALUES (?, ?, ?, ?, ?)""",
                (self.session_id, proposal_id, round_num, reason, scores_json),
            )
            conn.commit()
            return cursor.lastrowid

    def get_eliminations(
        self,
        round_num: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Get elimination records."""
        conn = self._get_conn()
        if round_num is not None:
            cur = conn.execute(
                """SELECT e.*, p.name as proposal_name, p.team
                   FROM eliminations e
                   LEFT JOIN proposals p ON e.proposal_id = p.id
                   WHERE e.session_id = ? AND e.round = ?
                   ORDER BY e.id ASC""",
                (self.session_id, round_num),
            )
        else:
            cur = conn.execute(
                """SELECT e.*, p.name as proposal_name, p.team
                   FROM eliminations e
                   LEFT JOIN proposals p ON e.proposal_id = p.id
                   WHERE e.session_id = ?
                   ORDER BY e.id ASC""",
                (self.session_id,),
            )
        return [dict(r) for r in cur.fetchall()]

    # ── Session / Execution Operations ─────────────────────────────

    def start_execution(
        self,
        goal: str,
        mode: str = "lite",
        config: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Record the start of a graph execution.

        Args:
            goal: The goal being executed.
            mode: Execution mode (lite/hard/super).
            config: Optional configuration snapshot.

        Returns:
            Execution record ID.
        """
        with self._lock:
            conn = self._get_conn()
            now = time.time()
            config_json = json.dumps(config or {}, default=str)
            cursor = conn.execute(
                """INSERT INTO graph_executions
                   (session_id, goal, mode, status, config_json, started_at)
                   VALUES (?, ?, ?, 'running', ?, ?)""",
                (self.session_id, goal, mode, config_json, now),
            )
            conn.commit()
            return cursor.lastrowid

    def complete_execution(
        self,
        execution_id: int,
        result: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
    ) -> None:
        """Mark an execution as completed or failed."""
        with self._lock:
            conn = self._get_conn()
            now = time.time()
            status = "failed" if error else "completed"
            result_json = json.dumps(result or {}, default=str)
            conn.execute(
                """UPDATE graph_executions
                   SET status = ?, completed_at = ?, result_json = ?, error = ?
                   WHERE id = ?""",
                (status, now, result_json, error, execution_id),
            )
            conn.commit()

    def get_execution(self, execution_id: int) -> Optional[Dict[str, Any]]:
        """Get an execution record."""
        conn = self._get_conn()
        cur = conn.execute(
            "SELECT * FROM graph_executions WHERE id = ?",
            (execution_id,),
        )
        row = cur.fetchone()
        return dict(row) if row else None

    def get_latest_execution(self) -> Optional[Dict[str, Any]]:
        """Get the most recent execution record."""
        conn = self._get_conn()
        cur = conn.execute(
            "SELECT * FROM graph_executions ORDER BY id DESC LIMIT 1",
        )
        row = cur.fetchone()
        return dict(row) if row else None

    # ── Evolution Operations ───────────────────────────────────────

    def record_evolution(
        self,
        action: str,
        target: str,
        before: Any = None,
        after: Any = None,
        score_before: Optional[float] = None,
        score_after: Optional[float] = None,
    ) -> int:
        """Record a self-evolution action.

        Args:
            action: Type of change (e.g., 'modify_config', 'adjust_weight').
            target: What was changed (e.g., 'config/scorecard.yaml').
            before: Previous value.
            after: New value.
            score_before: Score before the change.
            score_after: Score after the change.

        Returns:
            Evolution record ID.
        """
        with self._lock:
            conn = self._get_conn()
            before_json = json.dumps(before, default=str) if before is not None else None
            after_json = json.dumps(after, default=str) if after is not None else None
            cursor = conn.execute(
                """INSERT INTO evolution_history
                   (session_id, action, target, before_json, after_json,
                    score_before, score_after)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (self.session_id, action, target, before_json, after_json,
                 score_before, score_after),
            )
            conn.commit()
            return cursor.lastrowid

    def get_evolution_history(self) -> List[Dict[str, Any]]:
        """Get all evolution records for this session."""
        conn = self._get_conn()
        cur = conn.execute(
            """SELECT * FROM evolution_history
               WHERE session_id = ?
               ORDER BY id ASC""",
            (self.session_id,),
        )
        return [dict(r) for r in cur.fetchall()]

    # ── Recovery ───────────────────────────────────────────────────

    def recover(self) -> Dict[str, Any]:
        """Recover from the last checkpoint.

        Analyzes the most recent checkpoint and determines the recovery
        action needed. If the last node was running, the caller should
        restart that node. If completed, the graph can continue from
        the next node.

        Returns:
            Recovery dict with keys:
                status: 'no_checkpoints', 'consistent', or 'recovered'.
                phase: Last node name.
                state: Last checkpointed state.
                action: 'continue', 'restart_node', or None.
        """
        last = self.get_last_checkpoint()
        if last is None:
            return {
                "status": "no_checkpoints",
                "phase": "init",
                "state": {},
                "action": None,
            }

        if last["status"] == "running":
            return {
                "status": "recovered",
                "phase": last["node_name"],
                "state": last["state"],
                "action": "restart_node",
            }

        return {
            "status": "consistent",
            "phase": last["node_name"],
            "state": last["state"],
            "action": "continue",
        }

    def recover_execution(self) -> Optional[Dict[str, Any]]:
        """Recover the latest incomplete execution.

        Returns the execution record if one is in 'running' state,
        along with the last checkpoint for resumption.

        Returns:
            Dict with keys: execution, last_checkpoint. None if no
            incomplete execution exists.
        """
        execution = self.get_latest_execution()
        if execution is None or execution["status"] != "running":
            return None

        chain = self.get_execution_chain()
        return {
            "execution": execution,
            "chain": chain,
            "last_checkpoint": self.get_last_checkpoint(),
        }

    # ── Lifecycle ──────────────────────────────────────────────────

    def clear_session(self) -> None:
        """Clear all data for the current session."""
        with self._lock:
            conn = self._get_conn()
            conn.execute("DELETE FROM checkpoints WHERE session_id = ?", (self.session_id,))
            conn.execute("DELETE FROM proposals WHERE session_id = ?", (self.session_id,))
            conn.execute("DELETE FROM eliminations WHERE session_id = ?", (self.session_id,))
            conn.execute("DELETE FROM graph_executions WHERE session_id = ?", (self.session_id,))
            conn.execute("DELETE FROM evolution_history WHERE session_id = ?", (self.session_id,))
            conn.commit()

    def close(self) -> None:
        """Close the thread-local database connection."""
        self._closed = True
        if hasattr(self._local, "conn") and self._local.conn is not None:
            try:
                self._local.conn.close()
            except Exception:
                pass
            self._local.conn = None

    def __enter__(self) -> "CheckpointStore":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
