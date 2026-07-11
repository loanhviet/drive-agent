"""Short-lived full document cache used before a user saves RAG memory."""

import time
import uuid
from dataclasses import dataclass
from threading import Lock
from typing import Any


class DocumentReferenceError(ValueError):
    """Raised when a cached document cannot be accessed."""


@dataclass(frozen=True)
class CachedDocument:
    document_ref: str
    user_id: str
    content: str
    metadata: dict[str, Any]
    created_at: float


class DocumentCache:
    def __init__(self, ttl_seconds: int = 30 * 60, clock=time.monotonic):
        self.ttl_seconds = ttl_seconds
        self.clock = clock
        self._documents: dict[str, CachedDocument] = {}
        self._lock = Lock()

    def put(self, user_id: str, content: str, metadata: dict[str, Any]) -> CachedDocument:
        self.cleanup_expired()
        document = CachedDocument(
            document_ref=str(uuid.uuid4()),
            user_id=user_id,
            content=content,
            metadata=dict(metadata),
            created_at=self.clock(),
        )
        with self._lock:
            self._documents[document.document_ref] = document
        return document

    def get(self, document_ref: str, user_id: str) -> CachedDocument:
        self.cleanup_expired()
        with self._lock:
            document = self._documents.get(document_ref)
        if document is None:
            raise DocumentReferenceError("Document reference was not found or has expired")
        if document.user_id != user_id:
            raise DocumentReferenceError("Document reference does not belong to this user")
        return document

    def discard(self, document_ref: str, user_id: str) -> None:
        document = self.get(document_ref, user_id)
        with self._lock:
            self._documents.pop(document.document_ref, None)

    def cleanup_expired(self) -> None:
        now = self.clock()
        with self._lock:
            expired = [
                reference
                for reference, document in self._documents.items()
                if now - document.created_at >= self.ttl_seconds
            ]
            for reference in expired:
                self._documents.pop(reference, None)


_document_cache = DocumentCache()


def get_document_cache() -> DocumentCache:
    return _document_cache
