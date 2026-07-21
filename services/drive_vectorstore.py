"""Qdrant storage for the shared, read-only Drive document corpus."""

import re
import unicodedata
import warnings
import uuid
from pathlib import Path
from threading import Lock
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    VectorParams,
)

from config import (
    DRIVE_CORPUS_ID,
    DRIVE_DOCUMENT_COLLECTION,
    DRIVE_QDRANT_PATH,
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    EMBEDDING_PROVIDER,
    MEMORY_SCORE_THRESHOLD,
    QDRANT_API_KEY,
    QDRANT_HOST,
    QDRANT_MODE,
    QDRANT_PATH,
    QDRANT_PORT,
    QDRANT_URL,
)
from services.document_extraction import ExtractionResult


class DriveDocumentStoreError(RuntimeError):
    pass


def resolved_drive_collection_name(
    base_name: str = DRIVE_DOCUMENT_COLLECTION,
    provider: str = EMBEDDING_PROVIDER,
    model: str = EMBEDDING_MODEL,
    dimension: int = EMBEDDING_DIM,
) -> str:
    suffix = re.sub(r"[^a-zA-Z0-9_]+", "_", f"{provider}_{model}_{dimension}").strip("_")
    return f"{base_name}_{suffix}"[:255]


def _condition(key: str, value: str | bool) -> FieldCondition:
    return FieldCondition(key=key, match=MatchValue(value=value))


def _normalized_tokens(value: str) -> set[str]:
    normalized = "".join(
        character
        for character in unicodedata.normalize("NFD", value.casefold())
        if unicodedata.category(character) != "Mn"
    )
    return set(re.findall(r"[a-z0-9]+", normalized))


class DriveDocumentVectorStore:
    def __init__(self, client: QdrantClient, collection_name: str, dimension: int):
        self.client = client
        self.collection_name = collection_name
        self.dimension = dimension
        self._collection_lock = Lock()

    def ensure_collection(self) -> None:
        with self._collection_lock:
            if not self.client.collection_exists(self.collection_name):
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=self.dimension, distance=Distance.COSINE),
                )
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message="Payload indexes have no effect in the local Qdrant.*",
                    )
                    for field_name, field_type in (
                        ("corpus_id", PayloadSchemaType.KEYWORD),
                        ("file_id", PayloadSchemaType.KEYWORD),
                        ("revision_id", PayloadSchemaType.KEYWORD),
                        ("is_active", PayloadSchemaType.BOOL),
                    ):
                        self.client.create_payload_index(
                            collection_name=self.collection_name,
                            field_name=field_name,
                            field_schema=field_type,
                            wait=True,
                        )
                return

            info = self.client.get_collection(self.collection_name)
            vector_config = info.config.params.vectors
            if not isinstance(vector_config, VectorParams):
                raise DriveDocumentStoreError("Named vectors are not supported by this collection")
            if vector_config.size != self.dimension or vector_config.distance != Distance.COSINE:
                raise DriveDocumentStoreError(
                    "Drive document collection is incompatible with embedding configuration"
                )

    def stage_revision(
        self,
        *,
        corpus_id: str,
        file_metadata: dict[str, Any],
        revision_id: str,
        source_fingerprint: str,
        pipeline_version: str,
        extraction: ExtractionResult,
        vectors: list[list[float]],
    ) -> None:
        if len(vectors) != len(extraction.chunks):
            raise DriveDocumentStoreError("Embedding count does not match extracted chunks")
        points: list[PointStruct] = []
        for chunk, vector in zip(extraction.chunks, vectors, strict=True):
            if len(vector) != self.dimension:
                raise DriveDocumentStoreError(
                    f"Vector dimension mismatch: expected {self.dimension}, got {len(vector)}"
                )
            point_id = str(
                uuid.uuid5(
                    uuid.NAMESPACE_URL,
                    f"{corpus_id}:{file_metadata['id']}:{revision_id}:{chunk.chunk_index}",
                )
            )
            points.append(
                PointStruct(
                    id=point_id,
                    vector=vector,
                    payload={
                        "corpus_id": corpus_id,
                        "file_id": file_metadata["id"],
                        "revision_id": revision_id,
                        "source_name": file_metadata.get("name", ""),
                        "mime_type": file_metadata.get("mimeType", ""),
                        "drive_path": file_metadata.get("drive_path", ""),
                        "web_view_link": file_metadata.get("webViewLink", ""),
                        "content_fingerprint": source_fingerprint,
                        "locator_type": chunk.locator_type,
                        "page_number": chunk.page_number,
                        "section": chunk.section,
                        "chunk_index": chunk.chunk_index,
                        "chunk_count": len(extraction.chunks),
                        "modified_time": file_metadata.get("modifiedTime", ""),
                        "pipeline_version": pipeline_version,
                        "is_active": False,
                        "text": chunk.text,
                    },
                )
            )
        self.ensure_collection()
        self.client.upsert(collection_name=self.collection_name, points=points, wait=True)

    def activate_revision(self, corpus_id: str, file_id: str, revision_id: str) -> None:
        self.ensure_collection()
        revision_filter = Filter(
            must=[
                _condition("corpus_id", corpus_id),
                _condition("file_id", file_id),
                _condition("revision_id", revision_id),
            ]
        )
        self.client.set_payload(
            collection_name=self.collection_name,
            payload={"is_active": True},
            points=revision_filter,
            wait=True,
        )
        old_revision_filter = Filter(
            must=[
                _condition("corpus_id", corpus_id),
                _condition("file_id", file_id),
            ],
            must_not=[_condition("revision_id", revision_id)],
        )
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=old_revision_filter,
            wait=True,
        )

    def remove_file(self, corpus_id: str, file_id: str) -> None:
        self.ensure_collection()
        file_filter = Filter(
            must=[
                _condition("corpus_id", corpus_id),
                _condition("file_id", file_id),
            ]
        )
        self.client.set_payload(
            collection_name=self.collection_name,
            payload={"is_active": False},
            points=file_filter,
            wait=True,
        )
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=file_filter,
            wait=True,
        )

    def search(
        self,
        query_vector: list[float],
        *,
        corpus_id: str = DRIVE_CORPUS_ID,
        top_k: int = 5,
        source_name: str | None = None,
        score_threshold: float | None = MEMORY_SCORE_THRESHOLD,
    ) -> list[dict[str, Any]]:
        if len(query_vector) != self.dimension:
            raise DriveDocumentStoreError(
                f"Vector dimension mismatch: expected {self.dimension}, got {len(query_vector)}"
            )
        self.ensure_collection()
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            query_filter=Filter(
                must=[
                    _condition("corpus_id", corpus_id),
                    _condition("is_active", True),
                ]
            ),
            limit=max(20, min(top_k * 4, 100)),
            with_payload=True,
            score_threshold=score_threshold,
        )
        source_tokens = _normalized_tokens(source_name or "")
        results: list[dict[str, Any]] = []
        seen: set[tuple[str, str, int | None, str | None]] = set()
        for point in response.points:
            payload = dict(point.payload or {})
            if source_tokens and not source_tokens.intersection(
                _normalized_tokens(str(payload.get("source_name", "")))
            ):
                continue
            key = (
                str(payload.get("file_id", "")),
                str(payload.get("revision_id", "")),
                payload.get("page_number"),
                payload.get("section"),
            )
            if key in seen:
                continue
            seen.add(key)
            text = str(payload.pop("text", ""))
            results.append({"text": text, "score": point.score, "metadata": payload})
            if len(results) >= top_k:
                break
        return results


_document_store: DriveDocumentVectorStore | None = None
_document_store_lock = Lock()


def _create_client() -> QdrantClient:
    if QDRANT_MODE == "local":
        path = Path(DRIVE_QDRANT_PATH)
        path.mkdir(parents=True, exist_ok=True)
        return QdrantClient(path=str(path))
    if QDRANT_MODE == "server":
        if QDRANT_URL:
            return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY or None)
        return QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, api_key=QDRANT_API_KEY or None)
    raise DriveDocumentStoreError("QDRANT_MODE must be local or server")


def get_drive_document_store() -> DriveDocumentVectorStore:
    global _document_store
    if _document_store is None:
        with _document_store_lock:
            if _document_store is None:
                _document_store = DriveDocumentVectorStore(
                    _create_client(),
                    resolved_drive_collection_name(),
                    EMBEDDING_DIM,
                )
    return _document_store


def set_drive_document_store_for_testing(
    store: DriveDocumentVectorStore | None,
) -> None:
    global _document_store
    _document_store = store
