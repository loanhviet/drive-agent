"""Tool for querying the shared, pre-indexed Drive document corpus."""

from config import DRIVE_CORPUS_ID, DRIVE_SEARCH_TOP_K
from registry.models import ToolDefinition
from services import embedding
from services.drive_vectorstore import get_drive_document_store


MAX_TOP_K = 10


def search_drive_knowledge(
    query: str,
    top_k: int = DRIVE_SEARCH_TOP_K,
    source_name: str = "",
) -> dict:
    if not query or not query.strip():
        raise ValueError("query must not be empty")
    if not isinstance(top_k, int) or isinstance(top_k, bool) or not 1 <= top_k <= MAX_TOP_K:
        raise ValueError(f"top_k must be an integer between 1 and {MAX_TOP_K}")

    matches = get_drive_document_store().search(
        embedding.embed_query(query.strip()),
        corpus_id=DRIVE_CORPUS_ID,
        top_k=top_k,
        source_name=source_name.strip() or None,
    )
    results = []
    for index, match in enumerate(matches, start=1):
        metadata = match["metadata"]
        citation_id = f"S{index}"
        text = match["text"]
        results.append(
            {
                "citation_id": citation_id,
                "text": text,
                "score": match["score"],
                "citation": {
                    "id": citation_id,
                    "file_id": metadata.get("file_id", ""),
                    "source_name": metadata.get("source_name", ""),
                    "locator_type": metadata.get("locator_type", "section"),
                    "page_number": metadata.get("page_number"),
                    "section": metadata.get("section"),
                    "snippet": text[:240],
                    "web_view_link": metadata.get("web_view_link", ""),
                },
            }
        )
    return {
        "status": "found" if results else "insufficient_data",
        "query": query,
        "answer_policy": (
            "Use only claims explicitly supported by result text. Cite claims with the exact "
            "citation_id in square brackets, for example [S1]. If results do not directly answer "
            "the question, report that the indexed Drive corpus has insufficient information."
        ),
        "results_count": len(results),
        "results": results,
    }


search_drive_knowledge_tool = ToolDefinition(
    name="search_drive_knowledge",
    description=(
        "Search documents already indexed from the shared Google Drive folder. "
        "Use this for questions about Drive documents without downloading or saving them first. "
        "Every supported claim must use the returned citation_id such as [S1]."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "minLength": 1,
                "description": "Question or semantic search query.",
            },
            "top_k": {
                "type": "integer",
                "minimum": 1,
                "maximum": MAX_TOP_K,
                "description": "Number of citable chunks to return.",
            },
            "source_name": {
                "type": "string",
                "description": "Optional natural-language source name filter.",
            },
        },
        "required": ["query"],
        "additionalProperties": False,
    },
    required_scopes=["drive:read"],
    handler=search_drive_knowledge,
)


ALL_DRIVE_KNOWLEDGE_TOOLS = [search_drive_knowledge_tool]
