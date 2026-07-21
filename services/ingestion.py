"""Durable SQLite state for shared Drive ingestion jobs and documents."""

import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

from config import APP_DB_PATH, DRIVE_CORPUS_ID
from services.database import connect_sqlite


ACTIVE_JOB_STATUSES = ("queued", "running")
TERMINAL_JOB_STATUSES = ("succeeded", "partial", "failed")
DOCUMENT_STATUSES = ("indexed", "stale", "failed", "unsupported", "removed")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class IngestionStore:
    def __init__(self, db_path: str = APP_DB_PATH):
        self.db_path = db_path
        self._lock = Lock()
        self._init_db()

    def _connect(self):
        return connect_sqlite(self.db_path)

    def _init_db(self) -> None:
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS drive_documents (
                    corpus_id TEXT NOT NULL,
                    file_id TEXT NOT NULL,
                    source_name TEXT NOT NULL,
                    mime_type TEXT NOT NULL,
                    drive_path TEXT NOT NULL DEFAULT '',
                    web_view_link TEXT NOT NULL DEFAULT '',
                    modified_time TEXT NOT NULL DEFAULT '',
                    source_fingerprint TEXT NOT NULL DEFAULT '',
                    active_revision_id TEXT,
                    status TEXT NOT NULL CHECK(
                        status IN ('indexed', 'stale', 'failed', 'unsupported', 'removed')
                    ),
                    error TEXT,
                    total_chars INTEGER NOT NULL DEFAULT 0,
                    page_count INTEGER NOT NULL DEFAULT 0,
                    chunk_count INTEGER NOT NULL DEFAULT 0,
                    last_seen_job_id TEXT,
                    indexed_at TEXT,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (corpus_id, file_id)
                );
                CREATE INDEX IF NOT EXISTS idx_drive_documents_status_name
                    ON drive_documents(corpus_id, status, source_name);

                CREATE TABLE IF NOT EXISTS ingestion_jobs (
                    job_id TEXT PRIMARY KEY,
                    corpus_id TEXT NOT NULL,
                    mode TEXT NOT NULL CHECK(mode IN ('incremental', 'full')),
                    trigger TEXT NOT NULL CHECK(trigger IN ('manual', 'scheduled')),
                    status TEXT NOT NULL CHECK(
                        status IN ('queued', 'running', 'succeeded', 'partial', 'failed')
                    ),
                    requested_by TEXT NOT NULL,
                    attempt_count INTEGER NOT NULL DEFAULT 0,
                    discovered_count INTEGER NOT NULL DEFAULT 0,
                    indexed_count INTEGER NOT NULL DEFAULT 0,
                    skipped_count INTEGER NOT NULL DEFAULT 0,
                    removed_count INTEGER NOT NULL DEFAULT 0,
                    failed_count INTEGER NOT NULL DEFAULT 0,
                    error TEXT,
                    created_at TEXT NOT NULL,
                    started_at TEXT,
                    completed_at TEXT,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_ingestion_jobs_corpus_created
                    ON ingestion_jobs(corpus_id, created_at DESC);

                CREATE TABLE IF NOT EXISTS ingestion_job_items (
                    job_id TEXT NOT NULL,
                    file_id TEXT NOT NULL,
                    source_name TEXT NOT NULL,
                    action TEXT NOT NULL CHECK(action IN ('index', 'skip', 'remove')),
                    status TEXT NOT NULL CHECK(
                        status IN ('queued', 'running', 'succeeded', 'skipped', 'failed')
                    ),
                    attempt_count INTEGER NOT NULL DEFAULT 0,
                    revision_id TEXT,
                    error TEXT,
                    started_at TEXT,
                    completed_at TEXT,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (job_id, file_id),
                    FOREIGN KEY (job_id) REFERENCES ingestion_jobs(job_id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_ingestion_items_status
                    ON ingestion_job_items(job_id, status);
                """
            )

    def enqueue_job(
        self,
        *,
        corpus_id: str = DRIVE_CORPUS_ID,
        mode: str = "incremental",
        trigger: str = "manual",
        requested_by: str = "system",
    ) -> dict:
        if mode not in {"incremental", "full"}:
            raise ValueError("mode must be incremental or full")
        if trigger not in {"manual", "scheduled"}:
            raise ValueError("trigger must be manual or scheduled")
        timestamp = _now()
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            active = connection.execute(
                """
                SELECT * FROM ingestion_jobs
                WHERE corpus_id = ? AND status IN ('queued', 'running')
                ORDER BY created_at ASC LIMIT 1
                """,
                (corpus_id,),
            ).fetchone()
            if active:
                result = dict(active)
                result["deduplicated"] = True
                return result

            job_id = str(uuid.uuid4())
            connection.execute(
                """
                INSERT INTO ingestion_jobs (
                    job_id, corpus_id, mode, trigger, status, requested_by,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, 'queued', ?, ?, ?)
                """,
                (job_id, corpus_id, mode, trigger, requested_by, timestamp, timestamp),
            )
            result = dict(
                connection.execute(
                    "SELECT * FROM ingestion_jobs WHERE job_id = ?", (job_id,)
                ).fetchone()
            )
            result["deduplicated"] = False
            return result

    def recover_incomplete_jobs(self) -> int:
        timestamp = _now()
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                UPDATE ingestion_job_items
                SET status = 'queued', updated_at = ?
                WHERE status = 'running'
                """,
                (timestamp,),
            )
            cursor = connection.execute(
                """
                UPDATE ingestion_jobs
                SET status = 'queued', started_at = NULL, updated_at = ?
                WHERE status = 'running'
                """,
                (timestamp,),
            )
            return cursor.rowcount

    def claim_next_job(self) -> dict | None:
        timestamp = _now()
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """
                SELECT * FROM ingestion_jobs
                WHERE status = 'queued'
                ORDER BY created_at ASC LIMIT 1
                """
            ).fetchone()
            if row is None:
                return None
            connection.execute(
                """
                UPDATE ingestion_jobs
                SET status = 'running', attempt_count = attempt_count + 1,
                    started_at = COALESCE(started_at, ?), updated_at = ?, error = NULL
                WHERE job_id = ? AND status = 'queued'
                """,
                (timestamp, timestamp, row["job_id"]),
            )
            return dict(
                connection.execute(
                    "SELECT * FROM ingestion_jobs WHERE job_id = ?", (row["job_id"],)
                ).fetchone()
            )

    def start_item(
        self,
        *,
        job_id: str,
        file_id: str,
        source_name: str,
        action: str,
        revision_id: str | None = None,
    ) -> dict:
        if action not in {"index", "skip", "remove"}:
            raise ValueError("Unsupported ingestion action")
        timestamp = _now()
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO ingestion_job_items (
                    job_id, file_id, source_name, action, status, attempt_count,
                    revision_id, started_at, updated_at
                ) VALUES (?, ?, ?, ?, 'running', 1, ?, ?, ?)
                ON CONFLICT(job_id, file_id) DO UPDATE SET
                    source_name = excluded.source_name,
                    action = excluded.action,
                    status = 'running',
                    attempt_count = ingestion_job_items.attempt_count + 1,
                    revision_id = excluded.revision_id,
                    error = NULL,
                    started_at = excluded.started_at,
                    completed_at = NULL,
                    updated_at = excluded.updated_at
                """,
                (
                    job_id,
                    file_id,
                    source_name,
                    action,
                    revision_id,
                    timestamp,
                    timestamp,
                ),
            )
            return dict(
                connection.execute(
                    """
                    SELECT * FROM ingestion_job_items
                    WHERE job_id = ? AND file_id = ?
                    """,
                    (job_id, file_id),
                ).fetchone()
            )

    def finish_item(
        self,
        *,
        job_id: str,
        file_id: str,
        status: str,
        error: str | None = None,
    ) -> None:
        if status not in {"succeeded", "skipped", "failed"}:
            raise ValueError("Unsupported terminal item status")
        timestamp = _now()
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                UPDATE ingestion_job_items
                SET status = ?, error = ?, completed_at = ?, updated_at = ?
                WHERE job_id = ? AND file_id = ?
                """,
                (status, error, timestamp, timestamp, job_id, file_id),
            )

    def complete_job(self, job_id: str, *, discovered_count: int) -> dict:
        timestamp = _now()
        with self._lock, self._connect() as connection:
            counts = {
                row["status"]: row["count"]
                for row in connection.execute(
                    """
                    SELECT status, COUNT(*) AS count
                    FROM ingestion_job_items WHERE job_id = ? GROUP BY status
                    """,
                    (job_id,),
                )
            }
            actions = {
                row["action"]: row["count"]
                for row in connection.execute(
                    """
                    SELECT action, COUNT(*) AS count
                    FROM ingestion_job_items
                    WHERE job_id = ? AND status IN ('succeeded', 'skipped')
                    GROUP BY action
                    """,
                    (job_id,),
                )
            }
            failed = counts.get("failed", 0)
            final_status = "partial" if failed else "succeeded"
            connection.execute(
                """
                UPDATE ingestion_jobs
                SET status = ?, discovered_count = ?, indexed_count = ?,
                    skipped_count = ?, removed_count = ?, failed_count = ?,
                    completed_at = ?, updated_at = ?
                WHERE job_id = ?
                """,
                (
                    final_status,
                    discovered_count,
                    actions.get("index", 0),
                    actions.get("skip", 0),
                    actions.get("remove", 0),
                    failed,
                    timestamp,
                    timestamp,
                    job_id,
                ),
            )
        return self.get_job(job_id)

    def requeue_job(self, job_id: str) -> None:
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                UPDATE ingestion_jobs
                SET status = 'queued', started_at = NULL, updated_at = ?
                WHERE job_id = ? AND status = 'running'
                """,
                (_now(), job_id),
            )

    def fail_job(self, job_id: str, error: str) -> dict:
        timestamp = _now()
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                UPDATE ingestion_jobs
                SET status = 'failed', error = ?, completed_at = ?, updated_at = ?
                WHERE job_id = ?
                """,
                (error, timestamp, timestamp, job_id),
            )
        return self.get_job(job_id)

    def get_job(self, job_id: str, *, include_items: bool = False) -> dict | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM ingestion_jobs WHERE job_id = ?", (job_id,)
            ).fetchone()
            if row is None:
                return None
            result = dict(row)
            if include_items:
                result["items"] = [
                    dict(item)
                    for item in connection.execute(
                        """
                        SELECT * FROM ingestion_job_items
                        WHERE job_id = ? ORDER BY source_name, file_id
                        """,
                        (job_id,),
                    )
                ]
            return result

    def sync_status(self, corpus_id: str = DRIVE_CORPUS_ID) -> dict:
        with self._connect() as connection:
            active = connection.execute(
                """
                SELECT * FROM ingestion_jobs
                WHERE corpus_id = ? AND status IN ('queued', 'running')
                ORDER BY created_at DESC LIMIT 1
                """,
                (corpus_id,),
            ).fetchone()
            latest = connection.execute(
                """
                SELECT * FROM ingestion_jobs
                WHERE corpus_id = ? ORDER BY created_at DESC LIMIT 1
                """,
                (corpus_id,),
            ).fetchone()
            last_success = connection.execute(
                """
                SELECT completed_at FROM ingestion_jobs
                WHERE corpus_id = ? AND status IN ('succeeded', 'partial')
                ORDER BY completed_at DESC LIMIT 1
                """,
                (corpus_id,),
            ).fetchone()
            document_counts = {
                row["status"]: row["count"]
                for row in connection.execute(
                    """
                    SELECT status, COUNT(*) AS count
                    FROM drive_documents WHERE corpus_id = ? GROUP BY status
                    """,
                    (corpus_id,),
                )
            }
        return {
            "corpus_id": corpus_id,
            "active_job": dict(active) if active else None,
            "latest_job": dict(latest) if latest else None,
            "last_success_at": last_success["completed_at"] if last_success else None,
            "documents": {
                status: document_counts.get(status, 0) for status in DOCUMENT_STATUSES
            },
        }

    def get_document(self, corpus_id: str, file_id: str) -> dict | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT * FROM drive_documents
                WHERE corpus_id = ? AND file_id = ?
                """,
                (corpus_id, file_id),
            ).fetchone()
        return dict(row) if row else None

    def record_document(
        self,
        *,
        corpus_id: str,
        file_id: str,
        source_name: str,
        mime_type: str,
        drive_path: str,
        web_view_link: str,
        modified_time: str,
        source_fingerprint: str,
        status: str,
        last_seen_job_id: str,
        active_revision_id: str | None = None,
        error: str | None = None,
        total_chars: int = 0,
        page_count: int = 0,
        chunk_count: int = 0,
        indexed_at: str | None = None,
    ) -> None:
        if status not in DOCUMENT_STATUSES:
            raise ValueError("Unsupported document status")
        timestamp = _now()
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO drive_documents (
                    corpus_id, file_id, source_name, mime_type, drive_path,
                    web_view_link, modified_time, source_fingerprint,
                    active_revision_id, status, error, total_chars,
                    page_count, chunk_count, last_seen_job_id, indexed_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(corpus_id, file_id) DO UPDATE SET
                    source_name = excluded.source_name,
                    mime_type = excluded.mime_type,
                    drive_path = excluded.drive_path,
                    web_view_link = excluded.web_view_link,
                    modified_time = excluded.modified_time,
                    source_fingerprint = excluded.source_fingerprint,
                    active_revision_id = CASE
                        WHEN excluded.status = 'stale' THEN drive_documents.active_revision_id
                        ELSE excluded.active_revision_id
                    END,
                    status = excluded.status,
                    error = excluded.error,
                    total_chars = excluded.total_chars,
                    page_count = excluded.page_count,
                    chunk_count = excluded.chunk_count,
                    last_seen_job_id = excluded.last_seen_job_id,
                    indexed_at = COALESCE(excluded.indexed_at, drive_documents.indexed_at),
                    updated_at = excluded.updated_at
                """,
                (
                    corpus_id,
                    file_id,
                    source_name,
                    mime_type,
                    drive_path,
                    web_view_link,
                    modified_time,
                    source_fingerprint,
                    active_revision_id,
                    status,
                    error,
                    total_chars,
                    page_count,
                    chunk_count,
                    last_seen_job_id,
                    indexed_at,
                    timestamp,
                ),
            )

    def mark_seen(self, corpus_id: str, file_id: str, job_id: str) -> None:
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                UPDATE drive_documents
                SET last_seen_job_id = ?, updated_at = ?
                WHERE corpus_id = ? AND file_id = ?
                """,
                (job_id, _now(), corpus_id, file_id),
            )

    def documents_missing_from_job(self, corpus_id: str, job_id: str) -> list[dict]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM drive_documents
                WHERE corpus_id = ?
                  AND status != 'removed'
                  AND COALESCE(last_seen_job_id, '') != ?
                ORDER BY source_name
                """,
                (corpus_id, job_id),
            ).fetchall()
        return [dict(row) for row in rows]

    def mark_removed(self, corpus_id: str, file_id: str, job_id: str) -> None:
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                UPDATE drive_documents
                SET status = 'removed', active_revision_id = NULL,
                    error = NULL, last_seen_job_id = ?, updated_at = ?
                WHERE corpus_id = ? AND file_id = ?
                """,
                (job_id, _now(), corpus_id, file_id),
            )

    def list_documents(
        self,
        *,
        corpus_id: str = DRIVE_CORPUS_ID,
        query: str = "",
        status: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> dict:
        limit = max(1, min(limit, 200))
        offset = max(0, offset)
        conditions = ["corpus_id = ?"]
        params: list[object] = [corpus_id]
        if query.strip():
            conditions.append("source_name LIKE ? ESCAPE '\\'")
            escaped = query.strip().replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            params.append(f"%{escaped}%")
        if status:
            if status not in DOCUMENT_STATUSES:
                raise ValueError("Unsupported document status")
            conditions.append("status = ?")
            params.append(status)
        where = " AND ".join(conditions)
        with self._connect() as connection:
            total = connection.execute(
                f"SELECT COUNT(*) AS count FROM drive_documents WHERE {where}",
                params,
            ).fetchone()["count"]
            rows = connection.execute(
                f"""
                SELECT * FROM drive_documents
                WHERE {where}
                ORDER BY source_name COLLATE NOCASE, file_id
                LIMIT ? OFFSET ?
                """,
                [*params, limit, offset],
            ).fetchall()
        return {"total": total, "documents": [dict(row) for row in rows]}


_ingestion_store: IngestionStore | None = None
_ingestion_lock = Lock()


def get_ingestion_store() -> IngestionStore:
    global _ingestion_store
    if _ingestion_store is None:
        with _ingestion_lock:
            if _ingestion_store is None:
                _ingestion_store = IngestionStore()
    return _ingestion_store


def set_ingestion_store_for_testing(store: IngestionStore | None) -> None:
    global _ingestion_store
    _ingestion_store = store
