"""Long-term fact and document memory tools backed by Qdrant."""

import hashlib
import uuid
from datetime import datetime, timezone

from registry.context import get_current_actor
from registry.models import ToolDefinition
from config import MEMORY_SCORE_THRESHOLD
from services import embedding, vectorstore
from services.chunking import DocumentChunk, chunk_document
from services.documents import get_document_cache

MAX_TOP_K = 10
PUBLIC_METADATA_KEYS = (
    "source_type",
    "source_name",
    "file_id",
    "category",
    "chunk_index",
    "chunk_count",
    "section",
)


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
    chunks = (
        chunk_document(raw_content)
        if is_document
        else [
            DocumentChunk(
                text=raw_content,
                chunk_index=0,
                start_char=0,
                end_char=len(raw_content),
                section="",
            )
        ]
    )
    if not chunks:
        raise ValueError("Memory content has no text to save")

    embedding_inputs = []
    for chunk in chunks:
        context = []
        if metadata["source_name"]:
            context.append(f"Source: {metadata['source_name']}")
        if chunk.section:
            context.append(f"Section: {chunk.section}")
        context.append(chunk.text)
        embedding_inputs.append("\n".join(context))

    vectors = embedding.embed_texts(embedding_inputs)
    if len(vectors) != len(chunks):
        raise RuntimeError("Embedding provider returned incomplete document vectors")
    vectorstore.save_memories(
        [
            (
                chunk.text,
                vector,
                {
                    **metadata,
                    "chunk_index": chunk.chunk_index,
                    "chunk_count": len(chunks),
                    "start_char": chunk.start_char,
                    "end_char": chunk.end_char,
                    "section": chunk.section,
                },
            )
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
    )

    if document_ref:
        get_document_cache().discard(document_ref, user_id)
    return {
        "status": "saved",
        "content_preview": raw_content[:160],
        "category": category,
        "chunks_saved": len(chunks),
        "memory_id": metadata["memory_id"],
    }


def search_memory(
    query: str,
    top_k: int = 5,
    memory_type: str = "all",
    source_name: str = "",
) -> dict:
    """Search only the authenticated user's long-term semantic memory."""
    if not query or not query.strip():
        raise ValueError("query must not be empty")
    if not isinstance(top_k, int) or isinstance(top_k, bool) or not 1 <= top_k <= MAX_TOP_K:
        raise ValueError(f"top_k must be an integer between 1 and {MAX_TOP_K}")
    allowed_memory_types = {"all", "fact", "document", "task"}
    if memory_type not in allowed_memory_types:
        raise ValueError(
            f"memory_type must be one of: {', '.join(sorted(allowed_memory_types))}"
        )

    actor = get_current_actor()
    memories = vectorstore.search_memory(
        embedding.embed_query(query.strip()),
        top_k=top_k,
        user_id=actor["user_id"],
        memory_type=memory_type,
        source_name=source_name.strip() or None,
        score_threshold=MEMORY_SCORE_THRESHOLD,
    )
    for memory in memories:
        metadata = {
            key: memory["metadata"].get(key)
            for key in PUBLIC_METADATA_KEYS
            if key in memory["metadata"]
        }
        memory["metadata"] = metadata
        memory["citation"] = {
            "source_name": metadata.get("source_name", ""),
            "file_id": metadata.get("file_id", ""),
            "section": metadata.get("section", ""),
            "chunk_index": metadata.get("chunk_index", 0),
        }
    return {
        "status": "found" if memories else "insufficient_data",
        "query": query,
        "memory_type": memory_type,
        "answer_policy": (
            "Use only claims explicitly present in the returned memory text. "
            "Do not add related background knowledge; omit any unsupported claim. "
            "Cite source_name and section/chunk_index without inventing a file URL."
        ),
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
        "or document knowledge. Use memory_type=fact for preferences and memory_type=document "
        "for saved files. Optionally filter by natural source-name keywords."
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
            "memory_type": {
                "type": "string",
                "enum": ["all", "fact", "document", "task"],
                "description": "Memory kind to search (default: all).",
            },
            "source_name": {
                "type": "string",
                "description": "Optional natural-language source name or distinctive keywords.",
            },
        },
        "required": ["query"],
        "additionalProperties": False,
    },
    required_scopes=["memory:read"],
    handler=search_memory,
)


ALL_MEMORY_TOOLS = [save_memory_tool, search_memory_tool]
