from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any


class SQLiteStore:
    """Small explicit persistence layer; avoids checkpointer version coupling."""

    def __init__(self, path: str) -> None:
        self.path = path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        return connection

    def _init_schema(self) -> None:
        with self.connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS tasks (
                    task_id TEXT PRIMARY KEY, question TEXT NOT NULL, status TEXT NOT NULL,
                    state_json TEXT NOT NULL, report TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS sources (
                    task_id TEXT NOT NULL, source_id TEXT NOT NULL, title TEXT NOT NULL,
                    url TEXT NOT NULL, metadata_json TEXT NOT NULL,
                    PRIMARY KEY(task_id, source_id)
                );
                CREATE TABLE IF NOT EXISTS node_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, task_id TEXT NOT NULL,
                    node TEXT NOT NULL, status TEXT NOT NULL, detail TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS errors (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, task_id TEXT NOT NULL,
                    node TEXT NOT NULL, message TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
            """)

    def save_state(self, state: dict[str, Any], status: str = "running") -> None:
        task_id = state["task_id"]
        state_json = json.dumps(state, ensure_ascii=False, default=str)
        with self.connect() as db:
            db.execute("""INSERT INTO tasks(task_id, question, status, state_json, report)
                VALUES(?,?,?,?,?) ON CONFLICT(task_id) DO UPDATE SET status=excluded.status,
                state_json=excluded.state_json, report=excluded.report, updated_at=CURRENT_TIMESTAMP""",
                (task_id, state.get("question", ""), status, state_json, state.get("report", "")))
            for source in state.get("sources", []):
                db.execute("""INSERT OR REPLACE INTO sources(task_id, source_id, title, url, metadata_json)
                    VALUES(?,?,?,?,?)""", (task_id, source["source_id"], source["title"], str(source["url"]), json.dumps(source, ensure_ascii=False, default=str)))

    def record_node(self, task_id: str, node: str, status: str, detail: str = "") -> None:
        with self.connect() as db:
            db.execute("INSERT INTO node_runs(task_id,node,status,detail) VALUES(?,?,?,?)", (task_id, node, status, detail[:2000]))

    def record_error(self, task_id: str, node: str, message: str) -> None:
        with self.connect() as db:
            db.execute("INSERT INTO errors(task_id,node,message) VALUES(?,?,?)", (task_id, node, message[:4000]))

    def list_tasks(self, limit: int = 30) -> list[dict[str, Any]]:
        with self.connect() as db:
            rows = db.execute("SELECT task_id,question,status,created_at,updated_at FROM tasks ORDER BY updated_at DESC LIMIT ?", (limit,)).fetchall()
        return [dict(row) for row in rows]

    def load_state(self, task_id: str) -> dict[str, Any] | None:
        with self.connect() as db:
            row = db.execute("SELECT state_json FROM tasks WHERE task_id=?", (task_id,)).fetchone()
        return json.loads(row[0]) if row else None

    def task_status(self, task_id: str) -> str | None:
        with self.connect() as db:
            row = db.execute("SELECT status FROM tasks WHERE task_id=?", (task_id,)).fetchone()
        return str(row[0]) if row else None

    def latest_node_run(self, task_id: str) -> dict[str, Any] | None:
        with self.connect() as db:
            row = db.execute(
                "SELECT node,status,detail FROM node_runs WHERE task_id=? ORDER BY id DESC LIMIT 1",
                (task_id,),
            ).fetchone()
        return dict(row) if row else None
