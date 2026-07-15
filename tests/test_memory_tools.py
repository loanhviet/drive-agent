import pytest
from qdrant_client import QdrantClient

from registry.context import execution_context
from registry.registry import ToolRegistry
from services.documents import DocumentCache, DocumentReferenceError
from services.chunking import DocumentChunk
from services.embedding import set_embedding_provider_for_testing
from services.vectorstore import VectorStore, set_vector_store_for_testing
from tools import memory


class SemanticFakeProvider:
    name = "fake"
    model = "semantic-fake"
    dimension = 3

    def __init__(self):
        self.calls = []

    def embed(self, texts, task_type):
        self.calls.append((list(texts), task_type))
        vectors = []
        for text in texts:
            lower = text.lower()
            if "python" in lower or "ngôn ngữ" in lower:
                vectors.append([1.0, 0.0, 0.0])
            elif "quantum_middle_concept" in lower or "quantum" in lower:
                vectors.append([0.0, 1.0, 0.0])
            else:
                vectors.append([0.0, 0.0, 1.0])
        return vectors


@pytest.fixture
def memory_environment(monkeypatch, tmp_path):
    client = QdrantClient(path=str(tmp_path / "qdrant"))
    store = VectorStore(client, "memory_tools", dimension=3)
    provider = SemanticFakeProvider()
    documents = DocumentCache()
    set_vector_store_for_testing(store)
    set_embedding_provider_for_testing(provider)
    monkeypatch.setattr(memory, "get_document_cache", lambda: documents)
    yield {"store": store, "provider": provider, "documents": documents}
    client.close()
    set_vector_store_for_testing(None)
    set_embedding_provider_for_testing(None)


def actor(user_id="user-1", scopes=None):
    return {
        "user_id": user_id,
        "role": "admin",
        "scopes": scopes or ["memory:read", "memory:write"],
    }


def test_save_and_search_preference(memory_environment):
    with execution_context(actor()):
        saved = memory.save_memory(content="Tôi thích Python", category="user_preference")
        found = memory.search_memory("Tôi thích ngôn ngữ gì?")

    assert saved["status"] == "saved"
    assert saved["chunks_saved"] == 1
    assert found["status"] == "found"
    assert found["memories"][0]["text"] == "Tôi thích Python"
    assert found["memories"][0]["metadata"]["category"] == "user_preference"


def test_duplicate_fact_skips_embedding_and_upsert(memory_environment):
    provider = memory_environment["provider"]
    with execution_context(actor()):
        first = memory.save_memory(content="Tôi thích Python")
        second = memory.save_memory(content="Tôi thích Python")

    assert first["status"] == "saved"
    assert second["status"] == "already_saved"
    assert len(provider.calls) == 1


def test_document_memory_chunks_and_retrieves_middle_concept(monkeypatch, memory_environment):
    documents = memory_environment["documents"]
    document = documents.put(
        "user-1",
        "full document content",
        {"file_id": "drive-file", "file_name": "notes.txt", "source_type": "drive_file"},
    )
    monkeypatch.setattr(
        memory,
        "chunk_document",
        lambda _content: [
            DocumentChunk("introduction", 0, 0, 12, "Introduction"),
            DocumentChunk("QUANTUM_MIDDLE_CONCEPT details", 1, 13, 43, "Results"),
            DocumentChunk("conclusion", 2, 44, 54, "Conclusion"),
        ],
    )

    with execution_context(actor()):
        saved = memory.save_memory(document_ref=document.document_ref, category="document")
        found = memory.search_memory("What is the quantum middle concept?", top_k=3)

    assert saved["status"] == "saved"
    assert saved["chunks_saved"] == 3
    document_embedding_call = memory_environment["provider"].calls[0]
    assert document_embedding_call[1] == "RETRIEVAL_DOCUMENT"
    assert "Source: notes.txt\nSection: Results\nQUANTUM_MIDDLE_CONCEPT" in document_embedding_call[0][1]
    assert found["status"] == "found"
    assert found["memories"][0]["text"] == "QUANTUM_MIDDLE_CONCEPT details"
    assert found["memories"][0]["metadata"]["file_id"] == "drive-file"
    assert found["memories"][0]["metadata"]["chunk_count"] == 3
    assert found["memories"][0]["citation"] == {
        "source_name": "notes.txt",
        "file_id": "drive-file",
        "section": "Results",
        "chunk_index": 1,
    }
    with pytest.raises(DocumentReferenceError):
        documents.get(document.document_ref, "user-1")


def test_document_reference_cannot_be_saved_by_another_user(memory_environment):
    documents = memory_environment["documents"]
    document = documents.put("user-1", "private document", {})

    with execution_context(actor("user-2")):
        with pytest.raises(DocumentReferenceError, match="does not belong"):
            memory.save_memory(document_ref=document.document_ref)


def test_long_direct_content_is_stored_as_document(monkeypatch, memory_environment):
    monkeypatch.setattr(
        memory,
        "chunk_document",
        lambda _content: [
            DocumentChunk("part one", 0, 0, 8, ""),
            DocumentChunk("part two", 1, 9, 17, ""),
        ],
    )

    with execution_context(actor()):
        saved = memory.save_memory(content="x" * 1201)

    memories = memory_environment["store"].list_all_memories(user_id="user-1")
    assert saved["chunks_saved"] == 2
    assert {item["metadata"]["source_type"] for item in memories} == {"document"}


def test_search_reports_insufficient_data(memory_environment):
    with execution_context(actor()):
        result = memory.search_memory("Nothing has been saved")

    assert result == {
        "status": "insufficient_data",
        "query": "Nothing has been saved",
        "memory_type": "all",
        "answer_policy": (
            "Use only claims explicitly present in the returned memory text. "
            "Do not add related background knowledge; omit any unsupported claim. "
            "A found result may only be semantically related; if none explicitly answer, "
            "try another appropriate memory_type or report insufficient data. "
            "Cite source_name and section/chunk_index without inventing a file URL."
        ),
        "results_count": 0,
        "memories": [],
    }


def test_search_filters_memory_type_and_source(memory_environment):
    with execution_context(actor()):
        memory.save_memory(content="Tôi thích Python", category="user_preference")
        memory.save_memory(
            content="Python document details " * 80,
            source_type="document",
            source_name="python-notes.txt",
        )
        found = memory.search_memory(
            "Python",
            memory_type="document",
            source_name="python-notes.txt",
        )

    assert found["status"] == "found"
    assert found["memory_type"] == "document"
    assert {item["metadata"]["source_type"] for item in found["memories"]} == {"document"}
    assert {item["citation"]["source_name"] for item in found["memories"]} == {
        "python-notes.txt"
    }


@pytest.mark.parametrize("top_k", [0, 11, True])
def test_search_validates_top_k(memory_environment, top_k):
    with execution_context(actor()):
        with pytest.raises(ValueError, match="top_k"):
            memory.search_memory("Python", top_k=top_k)


def test_search_validates_memory_type(memory_environment):
    with execution_context(actor()):
        with pytest.raises(ValueError, match="memory_type"):
            memory.search_memory("Python", memory_type="unknown")


def test_registry_blocks_memory_write_for_read_only_user(memory_environment):
    registry = ToolRegistry(
        authenticator=lambda _token: actor(scopes=["memory:read"]),
        audit_store=None,
    )
    registry.register(memory.save_memory_tool)

    result = registry.call("save_memory", {"content": "Tôi thích Python"}, "token")

    assert result["ok"] is False
    assert result["error"]["code"] == "missing_scope"
