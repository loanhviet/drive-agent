"""Persistent Qdrant vector storage for long-term agent memory."""

import re
import unicodedata
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchAny,
    MatchValue,
    PointStruct,
    VectorParams,
)

from config import (
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    EMBEDDING_PROVIDER,
    MEMORY_COLLECTION,
    QDRANT_API_KEY,
    QDRANT_HOST,
    QDRANT_MODE,
    QDRANT_PATH,
    QDRANT_PORT,
    QDRANT_URL,
)


class VectorStoreError(RuntimeError):
    """Raised for incompatible Qdrant configuration or data."""


_SOURCE_STOP_WORDS = {
    "csv",
    "docx",
    "document",
    "file",
    "html",
    "json",
    "md",
    "nguon",
    "pdf",
    "pptx",
    "source",
    "tai",
    "tep",
    "text",
    "txt",
    "lieu",
    "xlsx",
}


def _normalized_tokens(value: str) -> list[str]:
    normalized = "".join(
        character
        for character in unicodedata.normalize("NFD", value.casefold())
        if unicodedata.category(character) != "Mn"
    )
    return re.findall(r"[a-z0-9]+", normalized)


def _source_name_matches(query: str, source_name: str) -> bool:
    query_tokens = [
        token
        for token in _normalized_tokens(query)
        if token not in _SOURCE_STOP_WORDS and len(token) >= 2
    ]
    if not query_tokens:
        return True
    source_tokens = set(_normalized_tokens(source_name))
    return any(token in source_tokens for token in query_tokens)


def resolved_collection_name(
    base_name: str = MEMORY_COLLECTION,
    provider: str = EMBEDDING_PROVIDER,
    model: str = EMBEDDING_MODEL,
    dimension: int = EMBEDDING_DIM,
) -> str:
    suffix = re.sub(r"[^a-zA-Z0-9_]+", "_", f"{provider}_{model}_{dimension}").strip("_")
    return f"{base_name}_{suffix}"[:255]


class VectorStore:
    def __init__(
        self,
        client: QdrantClient,
        collection_name: str,
        dimension: int,
    ):
        self.client = client
        self.collection_name = collection_name
        self.dimension = dimension

    def ensure_collection(self) -> None:
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=self.dimension, distance=Distance.COSINE),
            )
            return

        info = self.client.get_collection(self.collection_name)
        vector_config = info.config.params.vectors
        if not isinstance(vector_config, VectorParams):
            raise VectorStoreError("Named vectors are not supported by this memory collection")
        if vector_config.size != self.dimension or vector_config.distance != Distance.COSINE:
            raise VectorStoreError(
                "Existing collection is incompatible with the configured embedding dimension or distance"
            )

    def save_memory(
        self,
        text: str,
        embedding: list[float],
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self.save_memories([(text, embedding, metadata)])[0]

    def save_memories(
        self,
        records: list[tuple[str, list[float], dict[str, Any] | None]],
    ) -> list[dict[str, Any]]:
        """Validate and upsert a group of memory chunks in one Qdrant request."""
        if not records:
            raise VectorStoreError("At least one memory record is required")

        points: list[PointStruct] = []
        saved: list[dict[str, Any]] = []
        for text, vector, metadata in records:
            if not text.strip():
                raise VectorStoreError("Memory text must not be empty")
            if len(vector) != self.dimension:
                raise VectorStoreError(
                    f"Vector dimension mismatch: expected {self.dimension}, got {len(vector)}"
                )
            payload = dict(metadata or {})
            payload.setdefault("memory_id", str(uuid.uuid4()))
            payload.setdefault("source_type", "fact")
            payload.setdefault("source_name", "")
            payload.setdefault("file_id", "")
            payload.setdefault("chunk_index", 0)
            payload.setdefault("created_at", datetime.now(timezone.utc).isoformat())
            payload["text"] = text
            point_id = str(
                uuid.uuid5(
                    uuid.NAMESPACE_URL,
                    f"{payload['memory_id']}:{payload['chunk_index']}",
                )
            )
            points.append(PointStruct(id=point_id, vector=vector, payload=payload))
            saved.append({"id": point_id, "metadata": payload})

        self.ensure_collection()
        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
            wait=True,
        )
        return saved

    def search_memory(
        self,
        query_vector: list[float],
        top_k: int = 5,
        user_id: str | None = None,
        memory_type: str = "all",
        source_name: str | None = None,
        score_threshold: float | None = None,
    ) -> list[dict[str, Any]]:
        if len(query_vector) != self.dimension:
            raise VectorStoreError(
                f"Vector dimension mismatch: expected {self.dimension}, got {len(query_vector)}"
            )
        allowed_memory_types = {"all", "fact", "document", "task"}
        if memory_type not in allowed_memory_types:
            raise VectorStoreError(
                f"memory_type must be one of: {', '.join(sorted(allowed_memory_types))}"
            )
        if score_threshold is not None and not -1.0 <= score_threshold <= 1.0:
            raise VectorStoreError("score_threshold must be between -1 and 1")

        self.ensure_collection()
        conditions: list[FieldCondition] = []
        if user_id:
            conditions.append(FieldCondition(key="user_id", match=MatchValue(value=user_id)))
        if memory_type != "all":
            source_types = {
                "fact": ["fact"],
                "document": ["document", "drive_file"],
                "task": ["task"],
            }[memory_type]
            match = (
                MatchValue(value=source_types[0])
                if len(source_types) == 1
                else MatchAny(any=source_types)
            )
            conditions.append(FieldCondition(key="source_type", match=match))
        query_filter = Filter(must=conditions) if conditions else None
        query_limit = max(1, min(max(top_k * 5, 20) if source_name else top_k, 100))
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            query_filter=query_filter,
            limit=query_limit,
            with_payload=True,
            score_threshold=score_threshold,
        )
        results = [
            {
                "text": point.payload.get("text", ""),
                "score": point.score,
                "metadata": {key: value for key, value in point.payload.items() if key != "text"},
            }
            for point in response.points
        ]
        if source_name and source_name.strip():
            results = [
                result
                for result in results
                if _source_name_matches(
                    source_name,
                    str(result["metadata"].get("source_name", "")),
                )
            ]
        return results[:top_k]

    def has_content_hash(self, user_id: str, content_hash: str) -> bool:
        """Return whether this user already stored the same source content."""
        self.ensure_collection()
        points, _ = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=Filter(
                must=[
                    FieldCondition(key="user_id", match=MatchValue(value=user_id)),
                    FieldCondition(key="content_hash", match=MatchValue(value=content_hash)),
                ]
            ),
            limit=1,
            with_payload=False,
            with_vectors=False,
        )
        return bool(points)

    def list_all_memories(self, limit: int = 100, user_id: str | None = None) -> list[dict[str, Any]]:
        self.ensure_collection()
        scroll_filter = None
        if user_id:
            scroll_filter = Filter(
                must=[FieldCondition(key="user_id", match=MatchValue(value=user_id))]
            )
        points, _ = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=scroll_filter,
            limit=max(1, min(limit, 500)),
            with_payload=True,
            with_vectors=False,
        )
        return [
            {
                "id": str(point.id),
                "text": point.payload.get("text", ""),
                "metadata": {key: value for key, value in point.payload.items() if key != "text"},
            }
            for point in points
        ]


_store: VectorStore | None = None


def _create_client() -> QdrantClient:
    if QDRANT_MODE == "local":
        path = Path(QDRANT_PATH)
        path.mkdir(parents=True, exist_ok=True)
        return QdrantClient(path=str(path))
    if QDRANT_MODE == "server":
        if QDRANT_URL:
            return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY or None)
        return QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, api_key=QDRANT_API_KEY or None)
    raise VectorStoreError("QDRANT_MODE must be 'local' or 'server'")


def get_vector_store() -> VectorStore:
    global _store
    if _store is None:
        _store = VectorStore(_create_client(), resolved_collection_name(), EMBEDDING_DIM)
    return _store


def set_vector_store_for_testing(store: VectorStore | None) -> None:
    global _store
    _store = store


def ensure_collection() -> None:
    get_vector_store().ensure_collection()


def save_memory(text: str, embedding: list[float], metadata: dict | None = None) -> dict[str, Any]:
    return get_vector_store().save_memory(text, embedding, metadata)


def save_memories(
    records: list[tuple[str, list[float], dict[str, Any] | None]],
) -> list[dict[str, Any]]:
    return get_vector_store().save_memories(records)


def search_memory(
    query_vector: list[float],
    top_k: int = 5,
    user_id: str | None = None,
    memory_type: str = "all",
    source_name: str | None = None,
    score_threshold: float | None = None,
) -> list[dict[str, Any]]:
    return get_vector_store().search_memory(
        query_vector,
        top_k=top_k,
        user_id=user_id,
        memory_type=memory_type,
        source_name=source_name,
        score_threshold=score_threshold,
    )


def has_content_hash(user_id: str, content_hash: str) -> bool:
    return get_vector_store().has_content_hash(user_id, content_hash)


def list_all_memories(limit: int = 100, user_id: str | None = None) -> list[dict[str, Any]]:
    return get_vector_store().list_all_memories(limit, user_id)
