"""
Memory Tools - Long-term memory using RAG (embed + vector search).
"""

from datetime import datetime
from registry.models import ToolDefinition
from services import embedding, vectorstore


# ============================================================
# Tool 1: SAVE MEMORY
# ============================================================

def save_memory(content: str, category: str = "general") -> dict:
    """Save information to long-term memory."""
    # TODO: Implement save_memory
    # - Embed content using embedding.embed_query(content)
    # - Create metadata with category and timestamp
    # - Save to vectorstore using vectorstore.save_memory(text, embedding, metadata)
    # - Return {"status": "saved", "content_preview": ..., "category": ...}
    pass


save_memory_tool = ToolDefinition(
    name="save_memory",
    description=(
        "Save important information to long-term memory for future reference. "
        "Use this to remember facts, user preferences, conversation insights, "
        "or any information that should persist across conversations. "
        "Provide a category to organize memories (e.g., 'user_preference', 'fact', 'task')."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "content": {
                "type": "string",
                "description": "The information to remember.",
            },
            "category": {
                "type": "string",
                "description": "Category for organizing the memory (e.g., 'user_preference', 'fact', 'task', 'general').",
            },
        },
        "required": ["content"],
    },
    required_scopes=["memory:write"],
    handler=save_memory,
)


# ============================================================
# Tool 2: SEARCH MEMORY
# ============================================================

def search_memory(query: str, top_k: int = 5) -> dict:
    """Search long-term memory for relevant information."""
    # TODO: Implement search_memory
    # - Embed query using embedding.embed_query(query)
    # - Search vectorstore using vectorstore.search_memory(query_vector, top_k)
    # - Return {"query": ..., "results_count": ..., "memories": ...}
    pass


search_memory_tool = ToolDefinition(
    name="search_memory",
    description=(
        "Search long-term memory for previously saved information. "
        "Uses semantic search to find the most relevant memories. "
        "Use this to recall facts, user preferences, or past interactions."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query to find relevant memories.",
            },
            "top_k": {
                "type": "integer",
                "description": "Number of results to return (default: 5).",
            },
        },
        "required": ["query"],
    },
    required_scopes=["memory:read"],
    handler=search_memory,
)


ALL_MEMORY_TOOLS = [save_memory_tool, search_memory_tool]
