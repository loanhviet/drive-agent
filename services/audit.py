"""Persistent and sanitized audit logging for tool calls."""

import json
import re
import sqlite3
from pathlib import Path
from threading import Lock
from typing import Any

from config import APP_DB_PATH

SENSITIVE_KEY = re.compile(
    r"(api[_-]?key|authorization|credential|password|private[_-]?key|secret|token)",
    re.IGNORECASE,
)


def sanitize(value: Any, max_string_length: int = 500) -> Any:
    if isinstance(value, dict):
        return {
            str(key): "[REDACTED]" if SENSITIVE_KEY.search(str(key)) else sanitize(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [sanitize(item) for item in value]
    if isinstance(value, str) and len(value) > max_string_length:
        return f"{value[:max_string_length]}… [truncated, {len(value)} chars]"
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


class AuditStore:
    def __init__(self, db_path: str = APP_DB_PATH):
        self.db_path = db_path
        self._lock = Lock()
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _init_db(self) -> None:
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS audit_logs (
                    audit_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    session_id TEXT,
                    tool TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    arguments TEXT NOT NULL,
                    steps TEXT NOT NULL,
                    status TEXT NOT NULL,
                    result TEXT,
                    error TEXT,
                    execution_time_ms REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_audit_user_time
                    ON audit_logs(user_id, timestamp DESC);
                CREATE INDEX IF NOT EXISTS idx_audit_session
                    ON audit_logs(session_id, timestamp DESC);
                """
            )

    def upsert(self, entry: dict[str, Any]) -> None:
        safe_entry = sanitize(entry)
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO audit_logs (
                    audit_id, timestamp, session_id, tool, user_id, role,
                    arguments, steps, status, result, error, execution_time_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(audit_id) DO UPDATE SET
                    user_id = excluded.user_id,
                    role = excluded.role,
                    steps = excluded.steps,
                    status = excluded.status,
                    result = excluded.result,
                    error = excluded.error,
                    execution_time_ms = excluded.execution_time_ms
                """,
                (
                    safe_entry["audit_id"],
                    safe_entry["timestamp"],
                    safe_entry.get("session_id"),
                    safe_entry["tool"],
                    safe_entry["user_id"],
                    safe_entry["role"],
                    json.dumps(safe_entry["arguments"], ensure_ascii=False),
                    json.dumps(safe_entry["steps"], ensure_ascii=False),
                    safe_entry["status"],
                    json.dumps(safe_entry.get("result"), ensure_ascii=False),
                    json.dumps(safe_entry.get("error"), ensure_ascii=False),
                    safe_entry["execution_time_ms"],
                ),
            )

    def list_logs(
        self,
        *,
        user_id: str | None = None,
        session_id: str | None = None,
        tool: str | None = None,
        status: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        for column, value in (
            ("user_id", user_id),
            ("session_id", session_id),
            ("tool", tool),
            ("status", status),
        ):
            if value is not None:
                clauses.append(f"{column} = ?")
                params.append(value)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(max(1, min(limit, 500)))
        with self._connect() as connection:
            rows = connection.execute(
                f"SELECT * FROM audit_logs{where} ORDER BY timestamp DESC LIMIT ?",  # noqa: S608
                params,
            ).fetchall()
        return [self._deserialize(row) for row in rows]

    @staticmethod
    def _deserialize(row: sqlite3.Row) -> dict[str, Any]:
        entry = dict(row)
        for field in ("arguments", "steps", "result", "error"):
            entry[field] = json.loads(entry[field]) if entry[field] else None
        return entry


_audit_store: AuditStore | None = None
_audit_lock = Lock()


def get_audit_store() -> AuditStore:
    global _audit_store
    if _audit_store is None:
        with _audit_lock:
            if _audit_store is None:
                _audit_store = AuditStore()
    return _audit_store
