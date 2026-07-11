"""Long-term fact and document memory tools backed by Qdrant."""

import hashlib
import uuid
from datetime import datetime, timezone

from registry.context import get_current_actor
from registry.models import ToolDefinition
from config import MEMORY_SCORE_THRESHOLD
from services import embedding, vectorstore
from services.chunking import split_text
from services.documents import get_document_cache

MAX_TOP_K = 10


def _content_hash(content: str) -> str:
    normalized = " ".join(content.split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _resolve_content(
    *,
    content: str | None,
    document_ref: str | None,
    user_id: str,
) -> tuple[str, dict, bool]:
    if document_ref:
        document = get_document_cache().get(document_ref, user_id)
        return document.content, dict(document.metadata), True
    if content and content.strip():
        return content.strip(), {}, False
    raise ValueError("Provide non-empty content or a document_ref from read_file_tool")


def save_memory(
    content: str | None = None,
    category: str = "general",
    source_type: str = "fact",
    source_name: str = "",
    file_id: str = "",
    document_ref: str | None = None,
) -> dict:
    """Save a user fact or cached document as persistent semantic memory."""
    actor = get_current_actor()
    user_id = actor["user_id"]
    raw_content, document_metadata, is_document_ref = _resolve_content(
        content=content,
        document_ref=document_ref,
        user_id=user_id,
    )
    content_hash = _content_hash(raw_content)
    if vectorstore.has_content_hash(user_id, content_hash):
        if document_ref:
            get_document_cache().discard(document_ref, user_id)
        return {
            "status": "already_saved",
            "content_preview": raw_content[:160],
            "category": category,
            "chunks_saved": 0,
        }

    is_document = is_document_ref or source_type == "document" or len(raw_content) > 1200
    metadata = {
        "memory_id": str(uuid.uuid4()),
        "source_type": document_metadata.get(
            "source_type", "document" if is_document else source_type
        ),
        "source_name": document_metadata.get("file_name", source_name),
        "file_id": document_metadata.get("file_id", file_id),
        "category": category,
        "content_hash": content_hash,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "user_id": user_id,
    }
    chunks = split_text(raw_content) if is_document else [raw_content]
    if not chunks:
        raise ValueError("Memory content has no text to save")

    vectors = embedding.embed_texts(chunks)
    if len(vectors) != len(chunks):
        raise RuntimeError("Embedding provider returned incomplete document vectors")
    for index, (chunk, vector) in enumerate(zip(chunks, vectors, strict=True)):
        vectorstore.save_memory(chunk, vector, {**metadata, "chunk_index": index})

    if document_ref:
        get_document_cache().discard(document_ref, user_id)
    return {
        "status": "saved",
        "content_preview": raw_content[:160],
        "category": category,
        "chunks_saved": len(chunks),
        "memory_id": metadata["memory_id"],
    }


def search_memory(query: str, top_k: int = 5) -> dict:
    """Search only the authenticated user's long-term semantic memory."""
    if not query or not query.strip():
        raise ValueError("query must not be empty")
    if not isinstance(top_k, int) or isinstance(top_k, bool) or not 1 <= top_k <= MAX_TOP_K:
        raise ValueError(f"top_k must be an integer between 1 and {MAX_TOP_K}")

    actor = get_current_actor()
    memories = vectorstore.search_memory(
        embedding.embed_query(query.strip()),
        top_k=top_k,
        user_id=actor["user_id"],
    )
    memories = [memory for memory in memories if memory["score"] >= MEMORY_SCORE_THRESHOLD]
    return {
        "status": "found" if memories else "insufficient_data",
        "query": query,
        "results_count": len(memories),
        "memories": memories,
    }


save_memory_tool = ToolDefinition(
    name="save_memory",
    description=(
        "Save a user preference/fact or a file that was just read. "
        "For a fact, provide content. For file content, provide the document_ref from read_file_tool "
        "instead of copying the whole document."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "content": {"type": "string", "description": "Short fact or preference to remember."},
            "document_ref": {
                "type": "string",
                "description": "Reference returned by read_file_tool for a full document.",
            },
            "category": {"type": "string", "description": "Memory category."},
            "source_type": {"type": "string", "enum": ["fact", "document", "task"]},
            "source_name": {"type": "string", "description": "Optional source name."},
            "file_id": {"type": "string", "description": "Optional Google Drive file ID."},
        },
        "anyOf": [{"required": ["content"]}, {"required": ["document_ref"]}],
        "additionalProperties": False,
    },
    required_scopes=["memory:write"],
    handler=save_memory,
)


search_memory_tool = ToolDefinition(
    name="search_memory",
    description=(
        "Search the authenticated user's long-term memory for previously saved preferences, facts, "
        "or document knowledge. Use this before answering questions about prior interactions or saved files."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "query": {"type": "string", "minLength": 1, "description": "Semantic search query."},
            "top_k": {
                "type": "integer",
                "minimum": 1,
                "maximum": MAX_TOP_K,
                "description": "Number of results to return (default: 5).",
            },
        },
        "required": ["query"],
        "additionalProperties": False,
    },
    required_scopes=["memory:read"],
    handler=search_memory,
)


ALL_MEMORY_TOOLS = [save_memory_tool, search_memory_tool]
