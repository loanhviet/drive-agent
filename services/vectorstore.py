"""Persistent Qdrant vector storage for long-term agent memory."""

import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, FieldCondition, Filter, MatchValue, PointStruct, VectorParams

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
        if not text.strip():
            raise VectorStoreError("Memory text must not be empty")
        if len(embedding) != self.dimension:
            raise VectorStoreError(
                f"Vector dimension mismatch: expected {self.dimension}, got {len(embedding)}"
            )
        self.ensure_collection()
        payload = dict(metadata or {})
        payload.setdefault("memory_id", str(uuid.uuid4()))
        payload.setdefault("source_type", "fact")
        payload.setdefault("source_name", "")
        payload.setdefault("file_id", "")
        payload.setdefault("chunk_index", 0)
        payload.setdefault("created_at", datetime.now(timezone.utc).isoformat())
        payload["text"] = text
        point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{payload['memory_id']}:{payload['chunk_index']}"))
        self.client.upsert(
            collection_name=self.collection_name,
            points=[PointStruct(id=point_id, vector=embedding, payload=payload)],
            wait=True,
        )
        return {"id": point_id, "metadata": payload}

    def search_memory(
        self,
        query_vector: list[float],
        top_k: int = 5,
        user_id: str | None = None,
    ) -> list[dict[str, Any]]:
        if len(query_vector) != self.dimension:
            raise VectorStoreError(
                f"Vector dimension mismatch: expected {self.dimension}, got {len(query_vector)}"
            )
        self.ensure_collection()
        query_filter = None
        if user_id:
            query_filter = Filter(
                must=[FieldCondition(key="user_id", match=MatchValue(value=user_id))]
            )
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            query_filter=query_filter,
            limit=max(1, min(top_k, 100)),
            with_payload=True,
        )
        return [
            {
                "text": point.payload.get("text", ""),
                "score": point.score,
                "metadata": {key: value for key, value in point.payload.items() if key != "text"},
            }
            for point in response.points
        ]

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


def search_memory(
    query_vector: list[float], top_k: int = 5, user_id: str | None = None
) -> list[dict[str, Any]]:
    return get_vector_store().search_memory(query_vector, top_k, user_id)


def list_all_memories(limit: int = 100, user_id: str | None = None) -> list[dict[str, Any]]:
    return get_vector_store().list_all_memories(limit, user_id)
