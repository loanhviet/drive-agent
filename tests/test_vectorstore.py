import pytest
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

import services.vectorstore as vectorstore
from services.vectorstore import VectorStore, VectorStoreError, resolved_collection_name


def make_store(tmp_path, collection="test_memory"):
    client = QdrantClient(path=str(tmp_path / "qdrant"))
    return VectorStore(client, collection, dimension=3)


def test_upsert_and_search_memory_with_user_filter(tmp_path):
    store = make_store(tmp_path)
    store.save_memory(
        "Python is the preferred language.",
        [1.0, 0.0, 0.0],
        {"user_id": "user-1", "source_type": "fact", "chunk_index": 0},
    )
    store.save_memory(
        "Java is stored for another user.",
        [0.0, 1.0, 0.0],
        {"user_id": "user-2", "source_type": "fact", "chunk_index": 0},
    )

    results = store.search_memory([1.0, 0.0, 0.0], user_id="user-1")

    assert len(results) == 1
    assert results[0]["text"] == "Python is the preferred language."
    assert results[0]["metadata"]["user_id"] == "user-1"
    assert results[0]["score"] > 0.99


def test_batch_upsert_writes_all_records_in_one_request(monkeypatch, tmp_path):
    store = make_store(tmp_path)
    store.ensure_collection()
    calls = []
    original_upsert = store.client.upsert

    def capture_upsert(**kwargs):
        calls.append(kwargs)
        return original_upsert(**kwargs)

    monkeypatch.setattr(store.client, "upsert", capture_upsert)

    saved = store.save_memories(
        [
            ("first", [1.0, 0.0, 0.0], {"memory_id": "memory", "chunk_index": 0}),
            ("second", [0.0, 1.0, 0.0], {"memory_id": "memory", "chunk_index": 1}),
        ]
    )

    assert len(saved) == 2
    assert len(calls) == 1
    assert len(calls[0]["points"]) == 2


def test_search_filters_memory_type_and_exact_source(tmp_path):
    store = make_store(tmp_path)
    store.save_memories(
        [
            (
                "Python preference",
                [1.0, 0.0, 0.0],
                {"user_id": "user-1", "source_type": "fact", "source_name": ""},
            ),
            (
                "Python document",
                [1.0, 0.0, 0.0],
                {
                    "user_id": "user-1",
                    "source_type": "drive_file",
                    "source_name": "notes.txt",
                },
            ),
            (
                "Other document",
                [1.0, 0.0, 0.0],
                {
                    "user_id": "user-1",
                    "source_type": "document",
                    "source_name": "other.txt",
                },
            ),
        ]
    )

    results = store.search_memory(
        [1.0, 0.0, 0.0],
        user_id="user-1",
        memory_type="document",
        source_name="tài liệu NOTES",
    )

    assert [item["text"] for item in results] == ["Python document"]


def test_batch_upsert_validates_every_record_before_writing(monkeypatch, tmp_path):
    store = make_store(tmp_path)
    upsert_called = False

    def capture_upsert(**_kwargs):
        nonlocal upsert_called
        upsert_called = True

    monkeypatch.setattr(store.client, "upsert", capture_upsert)

    with pytest.raises(VectorStoreError, match="expected 3, got 2"):
        store.save_memories(
            [
                ("valid", [1.0, 0.0, 0.0], None),
                ("invalid", [1.0, 0.0], None),
            ]
        )

    assert upsert_called is False


def test_data_persists_after_client_recreation(tmp_path):
    store = make_store(tmp_path, collection="persistent_memory")
    store.save_memory(
        "Persistent vector.",
        [0.0, 0.0, 1.0],
        {"user_id": "user-1", "memory_id": "memory-1", "chunk_index": 0},
    )
    store.client.close()

    recreated = make_store(tmp_path, collection="persistent_memory")
    results = recreated.search_memory([0.0, 0.0, 1.0], user_id="user-1")

    assert results[0]["text"] == "Persistent vector."
    recreated.client.close()


def test_collection_dimension_mismatch_is_clear(tmp_path):
    client = QdrantClient(path=str(tmp_path / "qdrant"))
    client.create_collection(
        collection_name="wrong_dimension",
        vectors_config=VectorParams(size=2, distance=Distance.COSINE),
    )
    store = VectorStore(client, "wrong_dimension", dimension=3)

    with pytest.raises(VectorStoreError, match="incompatible"):
        store.ensure_collection()


def test_vector_dimension_is_validated_before_upsert(tmp_path):
    store = make_store(tmp_path)

    with pytest.raises(VectorStoreError, match="expected 3, got 2"):
        store.save_memory("bad vector", [1.0, 0.0])


def test_query_dimension_and_empty_text_are_validated(tmp_path):
    store = make_store(tmp_path)

    with pytest.raises(VectorStoreError, match="Memory text must not be empty"):
        store.save_memory("   ", [1.0, 0.0, 0.0])
    with pytest.raises(VectorStoreError, match="expected 3, got 2"):
        store.search_memory([1.0, 0.0])


def test_list_all_memories_filters_by_user(tmp_path):
    store = make_store(tmp_path)
    store.save_memory("User one", [1.0, 0.0, 0.0], {"user_id": "user-1"})
    store.save_memory("User two", [0.0, 1.0, 0.0], {"user_id": "user-2"})

    memories = store.list_all_memories(user_id="user-1")

    assert len(memories) == 1
    assert memories[0]["text"] == "User one"


def test_module_helpers_use_configured_local_storage(monkeypatch, tmp_path):
    monkeypatch.setattr(vectorstore, "QDRANT_MODE", "local")
    monkeypatch.setattr(vectorstore, "QDRANT_PATH", str(tmp_path / "configured-qdrant"))
    monkeypatch.setattr(vectorstore, "EMBEDDING_DIM", 3)
    monkeypatch.setattr(vectorstore, "EMBEDDING_PROVIDER", "fake")
    monkeypatch.setattr(vectorstore, "EMBEDDING_MODEL", "fake-model")
    monkeypatch.setattr(vectorstore, "MEMORY_COLLECTION", "configured_memory")
    vectorstore.set_vector_store_for_testing(None)

    saved = vectorstore.save_memory(
        "Configured memory", [1.0, 0.0, 0.0], {"user_id": "user-1"}
    )
    found = vectorstore.search_memory([1.0, 0.0, 0.0], user_id="user-1")
    listed = vectorstore.list_all_memories(user_id="user-1")

    assert saved["id"]
    assert found[0]["text"] == "Configured memory"
    assert listed[0]["text"] == "Configured memory"
    vectorstore.get_vector_store().client.close()
    vectorstore.set_vector_store_for_testing(None)


def test_collection_name_separates_embedding_models():
    gemini = resolved_collection_name("memory", "gemini", "gemini-embedding-001", 768)
    openai = resolved_collection_name("memory", "openai", "text-embedding-3-small", 768)

    assert gemini != openai
    assert gemini.startswith("memory_")
