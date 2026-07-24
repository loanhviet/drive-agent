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
MAX_LIST_LIMIT = 50
MAX_LIST_SCAN = 500
MEMORY_PREVIEW_CHARS = 160
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
            "A found result may only be semantically related; if none explicitly answer, "
            "try another appropriate memory_type or report insufficient data. "
            "Cite source_name and section/chunk_index without inventing a file URL."
        ),
        "results_count": len(memories),
        "memories": memories,
    }


def _memory_type_for_source(source_type: str) -> str:
    if source_type in {"document", "drive_file"}:
        return "document"
    if source_type == "task":
        return "task"
    return "fact"


def list_saved_memories(
    memory_type: str = "all",
    limit: int = 20,
) -> dict:
    """List user-owned memories without running a semantic search."""
    allowed_memory_types = {"all", "fact", "document", "task"}
    if memory_type not in allowed_memory_types:
        raise ValueError(
            f"memory_type must be one of: {', '.join(sorted(allowed_memory_types))}"
        )
    if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= MAX_LIST_LIMIT:
        raise ValueError(f"limit must be an integer between 1 and {MAX_LIST_LIMIT}")

    actor = get_current_actor()
    chunks = vectorstore.list_all_memories(
        limit=MAX_LIST_SCAN,
        user_id=actor["user_id"],
    )
    grouped: dict[str, dict] = {}
    for chunk in chunks:
        metadata = chunk.get("metadata", {})
        normalized_type = _memory_type_for_source(
            str(metadata.get("source_type", "fact"))
        )
        if memory_type != "all" and normalized_type != memory_type:
            continue

        memory_id = str(metadata.get("memory_id") or chunk.get("id", ""))
        summary = grouped.setdefault(
            memory_id,
            {
                "memory_id": memory_id,
                "memory_type": normalized_type,
                "source_name": str(metadata.get("source_name", "")),
                "category": str(metadata.get("category", "")),
                "created_at": str(metadata.get("created_at", "")),
                "chunk_count": 1,
                "content_preview": "",
                "_preview_chunk_index": None,
            },
        )
        declared_count = metadata.get("chunk_count", 1)
        if isinstance(declared_count, int) and not isinstance(declared_count, bool):
            summary["chunk_count"] = max(summary["chunk_count"], declared_count)

        chunk_index = metadata.get("chunk_index", 0)
        if not isinstance(chunk_index, int) or isinstance(chunk_index, bool):
            chunk_index = 0
        preview_index = summary["_preview_chunk_index"]
        if preview_index is None or chunk_index < preview_index:
            summary["_preview_chunk_index"] = chunk_index
            normalized_text = " ".join(str(chunk.get("text", "")).split())
            summary["content_preview"] = normalized_text[:MEMORY_PREVIEW_CHARS]

    summaries = [
        {key: value for key, value in summary.items() if not key.startswith("_")}
        for summary in grouped.values()
    ]
    summaries.sort(key=lambda item: item["memory_id"])
    summaries.sort(key=lambda item: item["created_at"], reverse=True)
    limited = summaries[:limit]
    return {
        "status": "found" if limited else "empty",
        "memory_type": memory_type,
        "results_count": len(limited),
        "has_more": len(summaries) > len(limited),
        "memories": limited,
    }


def delete_memory(memory_id: str) -> dict:
    """Delete all chunks of a user-owned memory by memory_id."""
    if not memory_id or not str(memory_id).strip():
        raise ValueError("memory_id must not be empty")
    memory_id = str(memory_id).strip()

    actor = get_current_actor()
    points_deleted = vectorstore.delete_by_memory_id(actor["user_id"], memory_id)
    if points_deleted == 0:
        return {"status": "not_found", "memory_id": memory_id, "points_deleted": 0}
    return {
        "status": "deleted",
        "memory_id": memory_id,
        "points_deleted": points_deleted,
    }


def update_memory(
    memory_id: str,
    content: str,
    category: str | None = None,
) -> dict:
    """Replace a saved fact/task in place. Document memories must be deleted and re-saved."""
    if not memory_id or not str(memory_id).strip():
        raise ValueError("memory_id must not be empty")
    if not content or not str(content).strip():
        raise ValueError("content must not be empty")
    memory_id = str(memory_id).strip()
    raw_content = str(content).strip()

    actor = get_current_actor()
    user_id = actor["user_id"]
    points = vectorstore.list_memory_points(user_id, memory_id)
    if not points:
        return {"status": "not_found", "memory_id": memory_id}

    source_types = {
        str(point.get("metadata", {}).get("source_type", "fact")) for point in points
    }
    if source_types & {"document", "drive_file"}:
        return {
            "status": "unsupported_type",
            "memory_id": memory_id,
            "message": (
                "Document memories cannot be updated in place. "
                "Delete the memory and save the document again."
            ),
        }
    if not source_types.issubset({"fact", "task"}):
        return {
            "status": "unsupported_type",
            "memory_id": memory_id,
            "message": "Only fact and task memories can be updated.",
        }

    existing = points[0].get("metadata", {})
    source_type = str(existing.get("source_type", "fact"))
    if source_type not in {"fact", "task"}:
        source_type = "fact"
    resolved_category = (
        category.strip()
        if isinstance(category, str) and category.strip()
        else str(existing.get("category", "general") or "general")
    )
    content_hash = _content_hash(raw_content)
    now = datetime.now(timezone.utc).isoformat()
    created_at = str(existing.get("created_at") or now)

    vectorstore.delete_by_memory_id(user_id, memory_id)
    vectors = embedding.embed_texts([raw_content])
    if len(vectors) != 1:
        raise RuntimeError("Embedding provider returned incomplete fact vectors")
    vectorstore.save_memories(
        [
            (
                raw_content,
                vectors[0],
                {
                    "memory_id": memory_id,
                    "source_type": source_type,
                    "source_name": str(existing.get("source_name", "") or ""),
                    "file_id": str(existing.get("file_id", "") or ""),
                    "category": resolved_category,
                    "content_hash": content_hash,
                    "created_at": created_at,
                    "updated_at": now,
                    "user_id": user_id,
                    "chunk_index": 0,
                    "chunk_count": 1,
                    "start_char": 0,
                    "end_char": len(raw_content),
                    "section": "",
                },
            )
        ]
    )
    return {
        "status": "updated",
        "memory_id": memory_id,
        "content_preview": raw_content[:MEMORY_PREVIEW_CHARS],
        "category": resolved_category,
        "source_type": source_type,
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
        "for saved files. Use memory_type=all for identity/profile questions because answers may "
        "exist in either facts or saved CV/profile documents. Optionally filter by natural "
        "source-name keywords."
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


list_saved_memories_tool = ToolDefinition(
    name="list_saved_memories",
    description=(
        "List facts, tasks, and document sources saved in the authenticated user's memory. "
        "Use this when the user asks what is remembered or which documents are saved."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "memory_type": {
                "type": "string",
                "enum": ["all", "fact", "document", "task"],
                "description": "Optional memory kind to list.",
            },
            "limit": {
                "type": "integer",
                "minimum": 1,
                "maximum": MAX_LIST_LIMIT,
                "description": "Maximum number of memory summaries to return.",
            },
        },
        "required": [],
        "additionalProperties": False,
    },
    required_scopes=["memory:read"],
    handler=list_saved_memories,
)


delete_memory_tool = ToolDefinition(
    name="delete_memory",
    description=(
        "Permanently delete a saved memory (fact, task, or document) by memory_id. "
        "Use list_saved_memories first when the user wants to forget something but has not "
        "provided a memory_id. Deletes all chunks belonging to that memory."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "memory_id": {
                "type": "string",
                "minLength": 1,
                "description": "memory_id from list_saved_memories or a prior save_memory result.",
            },
        },
        "required": ["memory_id"],
        "additionalProperties": False,
    },
    required_scopes=["memory:write"],
    handler=delete_memory,
)


update_memory_tool = ToolDefinition(
    name="update_memory",
    description=(
        "Update a previously saved fact or task by memory_id, keeping the same memory_id. "
        "Use list_saved_memories first when the id is unknown. Document memories cannot be "
        "updated in place — delete them and save again."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "memory_id": {
                "type": "string",
                "minLength": 1,
                "description": "memory_id of the fact or task to update.",
            },
            "content": {
                "type": "string",
                "minLength": 1,
                "description": "Replacement fact or task text.",
            },
            "category": {
                "type": "string",
                "description": "Optional new category; defaults to the existing category.",
            },
        },
        "required": ["memory_id", "content"],
        "additionalProperties": False,
    },
    required_scopes=["memory:write"],
    handler=update_memory,
)


ALL_MEMORY_TOOLS = [
    save_memory_tool,
    search_memory_tool,
    list_saved_memories_tool,
    delete_memory_tool,
    update_memory_tool,
]
