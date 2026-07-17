"""SQLite persistence for user-visible chat history."""

import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

from config import APP_DB_PATH


class ChatStore:
    """Store completed user/assistant turns without tool payloads."""

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
                CREATE TABLE IF NOT EXISTS chat_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_chat_user_session_id
                    ON chat_messages(user_id, session_id, id);
                CREATE TABLE IF NOT EXISTS chat_sessions (
                    user_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (user_id, session_id)
                );
                CREATE INDEX IF NOT EXISTS idx_chat_sessions_user_updated
                    ON chat_sessions(user_id, updated_at DESC);
                """
            )
            self._migrate_legacy_session_schema(connection)

    @staticmethod
    def _migrate_legacy_session_schema(connection: sqlite3.Connection) -> None:
        """Make session IDs user-scoped without discarding existing conversations."""
        columns = connection.execute("PRAGMA table_info(chat_sessions)").fetchall()
        primary_key = [
            row["name"]
            for row in sorted(
                (row for row in columns if row["pk"]),
                key=lambda row: row["pk"],
            )
        ]
        if primary_key == ["user_id", "session_id"]:
            return

        connection.executescript(
            """
            DROP INDEX IF EXISTS idx_chat_sessions_user_updated;
            CREATE TABLE chat_sessions_v2 (
                user_id TEXT NOT NULL,
                session_id TEXT NOT NULL,
                title TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (user_id, session_id)
            );
            INSERT INTO chat_sessions_v2 (user_id, session_id, title, created_at, updated_at)
                SELECT user_id, session_id, title, created_at, updated_at
                FROM chat_sessions;
            DROP TABLE chat_sessions;
            ALTER TABLE chat_sessions_v2 RENAME TO chat_sessions;
            CREATE INDEX idx_chat_sessions_user_updated
                ON chat_sessions(user_id, updated_at DESC);
            """
        )

    def save_turn(self, *, user_id: str, session_id: str, user_message: str, assistant_message: str) -> None:
        timestamp = datetime.now(timezone.utc).isoformat()
        rows = (
            (user_id, session_id, "user", user_message, timestamp),
            (user_id, session_id, "assistant", assistant_message, timestamp),
        )
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO chat_sessions (session_id, user_id, title, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(user_id, session_id) DO UPDATE SET
                    title = CASE
                        WHEN chat_sessions.title = 'New chat' THEN excluded.title
                        ELSE chat_sessions.title
                    END,
                    updated_at = excluded.updated_at
                """,
                (session_id, user_id, self._title_from_message(user_message), timestamp, timestamp),
            )
            connection.executemany(
                """
                INSERT INTO chat_messages (user_id, session_id, role, content, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                rows,
            )

    def create_session(self, *, user_id: str) -> dict:
        timestamp = datetime.now(timezone.utc).isoformat()
        session_id = str(uuid.uuid4())
        entry = {
            "session_id": session_id,
            "title": "New chat",
            "created_at": timestamp,
            "updated_at": timestamp,
        }
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO chat_sessions (session_id, user_id, title, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (session_id, user_id, entry["title"], timestamp, timestamp),
            )
        return entry

    def list_sessions(self, *, user_id: str) -> list[dict]:
        """List stored sessions plus legacy message-only sessions."""
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT session_id, title, created_at, updated_at FROM chat_sessions
                WHERE user_id = ?
                UNION ALL
                SELECT m.session_id,
                       COALESCE(MIN(CASE WHEN m.role = 'user' THEN m.content END), 'Untitled chat'),
                       MIN(m.created_at), MAX(m.created_at)
                FROM chat_messages AS m
                WHERE m.user_id = ?
                  AND NOT EXISTS (
                      SELECT 1 FROM chat_sessions AS s
                      WHERE s.user_id = m.user_id AND s.session_id = m.session_id
                  )
                GROUP BY m.session_id
                ORDER BY updated_at DESC
                """,
                (user_id, user_id),
            ).fetchall()
        return [dict(row) for row in rows]

    def delete_session(self, *, user_id: str, session_id: str) -> bool:
        with self._lock, self._connect() as connection:
            owned = connection.execute(
                """
                SELECT 1 FROM chat_sessions WHERE user_id = ? AND session_id = ?
                UNION ALL
                SELECT 1 FROM chat_messages WHERE user_id = ? AND session_id = ? LIMIT 1
                """,
                (user_id, session_id, user_id, session_id),
            ).fetchone()
            if owned is None:
                return False
            connection.execute(
                "DELETE FROM chat_messages WHERE user_id = ? AND session_id = ?",
                (user_id, session_id),
            )
            connection.execute(
                "DELETE FROM chat_sessions WHERE user_id = ? AND session_id = ?",
                (user_id, session_id),
            )
        return True

    def list_messages(self, *, user_id: str, session_id: str, limit: int | None = None) -> list[dict]:
        params: list[object] = [user_id, session_id]
        query = (
            "SELECT id, role, content, created_at FROM chat_messages "
            "WHERE user_id = ? AND session_id = ? ORDER BY id ASC"
        )
        if limit is not None:
            query = (
                "SELECT id, role, content, created_at FROM ("
                "SELECT id, role, content, created_at FROM chat_messages "
                "WHERE user_id = ? AND session_id = ? ORDER BY id DESC LIMIT ?"
                ") ORDER BY id ASC"
            )
            params.append(max(1, limit))
        with self._connect() as connection:
            rows = connection.execute(query, params).fetchall()
        return [dict(row) for row in rows]

    @staticmethod
    def _title_from_message(message: str) -> str:
        normalized = " ".join(message.split())
        return normalized[:60] or "New chat"


_chat_store: ChatStore | None = None
_chat_lock = Lock()


def get_chat_store() -> ChatStore:
    global _chat_store
    if _chat_store is None:
        with _chat_lock:
            if _chat_store is None:
                _chat_store = ChatStore()
    return _chat_store
