"""
Vector Store Service - Qdrant for agent long-term memory.
"""

import uuid
from config import QDRANT_HOST, QDRANT_PORT, MEMORY_COLLECTION, EMBEDDING_DIM


def ensure_collection():
    """Ensure the Qdrant collection exists."""
    # TODO: Implement using QdrantClient
    # - Connect to Qdrant at QDRANT_HOST:QDRANT_PORT
    # - Check if MEMORY_COLLECTION exists
    # - If not, create it with EMBEDDING_DIM and cosine distance
    pass


def save_memory(text: str, embedding: list[float], metadata: dict = None):
    """Save a single memory entry to the vector store."""
    # TODO: Implement
    # - Call ensure_collection()
    # - Create a PointStruct with uuid, embedding, and payload (text + metadata)
    # - Upsert into MEMORY_COLLECTION
    pass


def search_memory(query_vector: list[float], top_k: int = 5) -> list[dict]:
    """Search memory by semantic similarity."""
    # TODO: Implement
    # - Call ensure_collection()
    # - Query Qdrant with query_vector, limit=top_k
    # - Return list of {"text": ..., "score": ..., "metadata": ...}
    pass


def list_all_memories(limit: int = 100) -> list[dict]:
    """List all stored memories."""
    # TODO: Implement
    # - Call ensure_collection()
    # - Scroll through MEMORY_COLLECTION
    # - Return list of {"id": ..., "text": ..., "metadata": ...}
    pass
