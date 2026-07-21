"""Single-instance durable worker for the shared Drive knowledge base."""

import hashlib
import logging
import os
from datetime import datetime, timezone
from threading import Event, Lock, Thread
from typing import Callable

from config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DRIVE_CORPUS_ID,
    DRIVE_JOB_MAX_ATTEMPTS,
    DRIVE_PIPELINE_VERSION,
    DRIVE_SYNC_INTERVAL_SECONDS,
    DRIVE_WORKER_POLL_SECONDS,
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    EMBEDDING_PROVIDER,
    GOOGLE_DRIVE_FOLDER_ID,
)
from services import drive_service, embedding
from services.document_extraction import (
    UnsupportedDocumentError,
    extract_document,
    is_supported_document,
)
from services.drive_vectorstore import (
    DriveDocumentVectorStore,
    get_drive_document_store,
)
from services.ingestion import IngestionStore, get_ingestion_store


logger = logging.getLogger(__name__)
_RETRY_DELAYS = (0, 2, 10, 30)


def source_fingerprint(file_metadata: dict) -> str:
    source_revision = file_metadata.get("md5Checksum") or "|".join(
        str(file_metadata.get(key, ""))
        for key in ("id", "modifiedTime", "size", "mimeType")
    )
    pipeline = "|".join(
        (
            DRIVE_PIPELINE_VERSION,
            EMBEDDING_PROVIDER,
            EMBEDDING_MODEL,
            str(EMBEDDING_DIM),
            str(CHUNK_SIZE),
            str(CHUNK_OVERLAP),
        )
    )
    return hashlib.sha256(f"{source_revision}|{pipeline}".encode("utf-8")).hexdigest()


def _indexed_at() -> str:
    return datetime.now(timezone.utc).isoformat()


class DriveIngestionWorker:
    def __init__(
        self,
        *,
        store: IngestionStore | None = None,
        vector_store: DriveDocumentVectorStore | None = None,
        folder_id: str = GOOGLE_DRIVE_FOLDER_ID,
        corpus_id: str = DRIVE_CORPUS_ID,
        poll_seconds: float = DRIVE_WORKER_POLL_SECONDS,
        sync_interval_seconds: int = DRIVE_SYNC_INTERVAL_SECONDS,
        max_attempts: int = DRIVE_JOB_MAX_ATTEMPTS,
        discover: Callable[[str], list[dict]] | None = None,
        downloader: Callable[..., dict] | None = None,
    ):
        if poll_seconds <= 0 or sync_interval_seconds <= 0 or max_attempts < 1:
            raise ValueError("Worker timing and retry configuration must be positive")
        self.store = store or get_ingestion_store()
        self.vector_store = vector_store or get_drive_document_store()
        self.folder_id = folder_id
        self.corpus_id = corpus_id
        self.poll_seconds = poll_seconds
        self.sync_interval_seconds = sync_interval_seconds
        self.max_attempts = max_attempts
        self.discover = discover or drive_service.walk_files
        self.downloader = downloader or drive_service.download_file
        self._stop = Event()
        self._thread: Thread | None = None
        self._start_lock = Lock()

    @property
    def is_alive(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    def start(self) -> None:
        with self._start_lock:
            if self.is_alive:
                return
            self._stop.clear()
            self.store.recover_incomplete_jobs()
            self._thread = Thread(
                target=self._run,
                name="drive-ingestion-worker",
                daemon=True,
            )
            self._thread.start()

    def stop(self, timeout: float = 10.0) -> None:
        self._stop.set()
        thread = self._thread
        if thread:
            thread.join(timeout=max(0.0, timeout))

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self._enqueue_scheduled_if_due()
                processed = self.run_once()
            except Exception:
                logger.exception("Drive ingestion worker loop failed")
                processed = False
            if not processed:
                self._stop.wait(self.poll_seconds)

    def _enqueue_scheduled_if_due(self) -> None:
        if not self.folder_id:
            return
        status = self.store.sync_status(self.corpus_id)
        if status["active_job"]:
            return
        latest = status["latest_job"]
        if latest is None:
            self.store.enqueue_job(
                corpus_id=self.corpus_id,
                mode="incremental",
                trigger="scheduled",
                requested_by="system",
            )
            return
        reference = latest.get("completed_at") or latest.get("created_at")
        if not reference:
            return
        try:
            age = datetime.now(timezone.utc) - datetime.fromisoformat(reference)
        except ValueError:
            age_seconds = self.sync_interval_seconds
        else:
            age_seconds = age.total_seconds()
        if age_seconds >= self.sync_interval_seconds:
            self.store.enqueue_job(
                corpus_id=self.corpus_id,
                mode="incremental",
                trigger="scheduled",
                requested_by="system",
            )

    def run_once(self) -> bool:
        job = self.store.claim_next_job()
        if job is None:
            return False
        if not self.folder_id:
            self.store.fail_job(job["job_id"], "drive_folder_not_configured")
            return True
        self._process_job(job)
        return True

    def _process_job(self, job: dict) -> None:
        job_id = job["job_id"]
        try:
            files = self.discover(self.folder_id)
        except Exception as error:
            logger.exception("Drive discovery failed", extra={"job_id": job_id})
            self.store.fail_job(job_id, f"Drive discovery failed: {error}")
            return

        discovered_count = len(files)
        for file_metadata in files:
            if self._stop.is_set():
                self.store.requeue_job(job_id)
                return
            self._process_discovered_file(job, file_metadata)

        for document in self.store.documents_missing_from_job(self.corpus_id, job_id):
            if self._stop.is_set():
                self.store.requeue_job(job_id)
                return
            self._remove_document(job_id, document)

        self.store.complete_job(job_id, discovered_count=discovered_count)

    def _process_discovered_file(self, job: dict, file_metadata: dict) -> None:
        job_id = job["job_id"]
        file_id = str(file_metadata["id"])
        source_name = str(file_metadata.get("name", ""))
        fingerprint = source_fingerprint(file_metadata)
        existing = self.store.get_document(self.corpus_id, file_id)

        if not is_supported_document(file_metadata):
            self.store.start_item(
                job_id=job_id,
                file_id=file_id,
                source_name=source_name,
                action="skip",
            )
            try:
                if existing and existing.get("active_revision_id"):
                    self.vector_store.remove_file(self.corpus_id, file_id)
                self._record_file(
                    file_metadata,
                    fingerprint,
                    job_id,
                    status="unsupported",
                    error="unsupported_file_type",
                )
            except Exception as error:
                self.store.finish_item(
                    job_id=job_id,
                    file_id=file_id,
                    status="failed",
                    error=str(error)[:1000],
                )
            else:
                self.store.finish_item(job_id=job_id, file_id=file_id, status="skipped")
            return

        unchanged = (
            job["mode"] == "incremental"
            and existing is not None
            and existing.get("source_fingerprint") == fingerprint
            and existing.get("status") == "indexed"
        )
        if unchanged:
            self.store.start_item(
                job_id=job_id,
                file_id=file_id,
                source_name=source_name,
                action="skip",
            )
            self.store.mark_seen(self.corpus_id, file_id, job_id)
            self.store.finish_item(job_id=job_id, file_id=file_id, status="skipped")
            return

        revision_seed = f"{fingerprint}:{job_id}" if job["mode"] == "full" else fingerprint
        revision_id = hashlib.sha256(revision_seed.encode("utf-8")).hexdigest()[:32]
        self._index_with_retry(
            job_id=job_id,
            file_metadata=file_metadata,
            fingerprint=fingerprint,
            revision_id=revision_id,
            existing=existing,
        )

    def _index_with_retry(
        self,
        *,
        job_id: str,
        file_metadata: dict,
        fingerprint: str,
        revision_id: str,
        existing: dict | None,
    ) -> None:
        file_id = str(file_metadata["id"])
        source_name = str(file_metadata.get("name", ""))
        for attempt in range(self.max_attempts):
            delay = _RETRY_DELAYS[min(attempt, len(_RETRY_DELAYS) - 1)]
            if delay and self._stop.wait(delay):
                self.store.requeue_job(job_id)
                return
            self.store.start_item(
                job_id=job_id,
                file_id=file_id,
                source_name=source_name,
                action="index",
                revision_id=revision_id,
            )
            try:
                extraction = self._index_file(
                    file_metadata=file_metadata,
                    fingerprint=fingerprint,
                    revision_id=revision_id,
                )
            except UnsupportedDocumentError as error:
                status = "stale" if existing and existing.get("active_revision_id") else "unsupported"
                self._record_file(
                    file_metadata,
                    fingerprint,
                    job_id,
                    status=status,
                    error=str(error)[:1000],
                    existing=existing,
                )
                self.store.finish_item(
                    job_id=job_id,
                    file_id=file_id,
                    status="failed" if status == "stale" else "skipped",
                    error=str(error)[:1000] if status == "stale" else None,
                )
                return
            except Exception as error:
                if attempt + 1 < self.max_attempts:
                    logger.warning(
                        "Drive file indexing attempt failed",
                        extra={"job_id": job_id, "file_id": file_id, "attempt": attempt + 1},
                    )
                    continue
                status = "stale" if existing and existing.get("active_revision_id") else "failed"
                self._record_file(
                    file_metadata,
                    fingerprint,
                    job_id,
                    status=status,
                    error=str(error)[:1000],
                    existing=existing,
                )
                self.store.finish_item(
                    job_id=job_id,
                    file_id=file_id,
                    status="failed",
                    error=str(error)[:1000],
                )
                return

            self._record_file(
                file_metadata,
                fingerprint,
                job_id,
                status="indexed",
                active_revision_id=revision_id,
                total_chars=extraction.total_chars,
                page_count=extraction.page_count,
                chunk_count=len(extraction.chunks),
                indexed_at=_indexed_at(),
            )
            self.store.finish_item(job_id=job_id, file_id=file_id, status="succeeded")
            return

    def _index_file(self, *, file_metadata: dict, fingerprint: str, revision_id: str):
        download = self.downloader(str(file_metadata["id"]))
        temporary_path = download["temp_path"]
        try:
            extraction = extract_document(
                temporary_path,
                mime_type=str(file_metadata.get("mimeType", "")),
            )
            embedding_inputs = []
            for chunk in extraction.chunks:
                locator = (
                    f"Page: {chunk.page_number}"
                    if chunk.page_number is not None
                    else f"Section: {chunk.section or 'Document start'}"
                )
                embedding_inputs.append(
                    f"Source: {file_metadata.get('name', '')}\n{locator}\n{chunk.text}"
                )
            vectors = embedding.embed_texts(embedding_inputs)
            self.vector_store.stage_revision(
                corpus_id=self.corpus_id,
                file_metadata=file_metadata,
                revision_id=revision_id,
                source_fingerprint=fingerprint,
                pipeline_version=DRIVE_PIPELINE_VERSION,
                extraction=extraction,
                vectors=vectors,
            )
            self.vector_store.activate_revision(
                self.corpus_id,
                str(file_metadata["id"]),
                revision_id,
            )
            return extraction
        finally:
            try:
                os.unlink(temporary_path)
            except FileNotFoundError:
                pass

    def _record_file(
        self,
        file_metadata: dict,
        fingerprint: str,
        job_id: str,
        *,
        status: str,
        active_revision_id: str | None = None,
        error: str | None = None,
        total_chars: int = 0,
        page_count: int = 0,
        chunk_count: int = 0,
        indexed_at: str | None = None,
        existing: dict | None = None,
    ) -> None:
        existing = existing or {}
        self.store.record_document(
            corpus_id=self.corpus_id,
            file_id=str(file_metadata["id"]),
            source_name=str(file_metadata.get("name", "")),
            mime_type=str(file_metadata.get("mimeType", "")),
            drive_path=str(file_metadata.get("drive_path", "")),
            web_view_link=str(file_metadata.get("webViewLink", "")),
            modified_time=str(file_metadata.get("modifiedTime", "")),
            source_fingerprint=fingerprint,
            status=status,
            last_seen_job_id=job_id,
            active_revision_id=active_revision_id,
            error=error,
            total_chars=total_chars or int(existing.get("total_chars", 0)),
            page_count=page_count or int(existing.get("page_count", 0)),
            chunk_count=chunk_count or int(existing.get("chunk_count", 0)),
            indexed_at=indexed_at,
        )

    def _remove_document(self, job_id: str, document: dict) -> None:
        file_id = document["file_id"]
        self.store.start_item(
            job_id=job_id,
            file_id=file_id,
            source_name=document["source_name"],
            action="remove",
        )
        try:
            self.vector_store.remove_file(self.corpus_id, file_id)
            self.store.mark_removed(self.corpus_id, file_id, job_id)
        except Exception as error:
            self.store.finish_item(
                job_id=job_id,
                file_id=file_id,
                status="failed",
                error=str(error)[:1000],
            )
        else:
            self.store.finish_item(job_id=job_id, file_id=file_id, status="succeeded")


_worker: DriveIngestionWorker | None = None
_worker_lock = Lock()


def get_drive_ingestion_worker() -> DriveIngestionWorker:
    global _worker
    if _worker is None:
        with _worker_lock:
            if _worker is None:
                _worker = DriveIngestionWorker()
    return _worker


def set_drive_ingestion_worker_for_testing(worker: DriveIngestionWorker | None) -> None:
    global _worker
    _worker = worker
