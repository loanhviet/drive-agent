import pytest

import services.embedding as embedding
from services.embedding import EmbeddingError


class FakeProvider:
    name = "fake"
    model = "fake-embedding"
    dimension = 3

    def __init__(self):
        self.calls = []

    def embed(self, texts, task_type):
        self.calls.append((list(texts), task_type))
        return [[float(index + 1), 0.0, 0.0] for index, _ in enumerate(texts)]


class WrongDimensionProvider(FakeProvider):
    def embed(self, texts, task_type):
        return [[1.0, 2.0] for _ in texts]


class FailingProvider(FakeProvider):
    def embed(self, texts, task_type):
        raise RuntimeError("provider unavailable")


@pytest.fixture(autouse=True)
def reset_embedding_provider():
    embedding.set_embedding_provider_for_testing(None)
    yield
    embedding.set_embedding_provider_for_testing(None)


def test_document_embeddings_are_batched():
    provider = FakeProvider()
    embedding.set_embedding_provider_for_testing(provider)

    vectors = embedding.embed_texts(["first", "second"])

    assert vectors == [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]
    assert provider.calls == [(["first", "second"], "RETRIEVAL_DOCUMENT")]


def test_document_embeddings_use_bounded_batches_in_order():
    provider = FakeProvider()
    embedding.set_embedding_provider_for_testing(provider)
    texts = [f"part-{index}" for index in range(34)]

    vectors = embedding.embed_texts(texts, batch_size=16)

    assert [len(call[0]) for call in provider.calls] == [16, 16, 2]
    assert [text for call in provider.calls for text in call[0]] == texts
    assert len(vectors) == len(texts)


@pytest.mark.parametrize("batch_size", [0, -1, True])
def test_document_embedding_validates_batch_size(batch_size):
    embedding.set_embedding_provider_for_testing(FakeProvider())

    with pytest.raises(EmbeddingError, match="batch_size"):
        embedding.embed_texts(["content"], batch_size=batch_size)


def test_query_embedding_uses_cache():
    provider = FakeProvider()
    embedding.set_embedding_provider_for_testing(provider)

    assert embedding.embed_query("python") == [1.0, 0.0, 0.0]
    assert embedding.embed_query("python") == [1.0, 0.0, 0.0]
    assert provider.calls == [(["python"], "RETRIEVAL_QUERY")]


@pytest.mark.parametrize("texts", [[], [""], ["  "]])
def test_empty_texts_are_rejected(texts):
    embedding.set_embedding_provider_for_testing(FakeProvider())

    with pytest.raises(EmbeddingError, match="must not be empty|required"):
        embedding.embed_texts(texts)


def test_dimension_mismatch_is_rejected():
    embedding.set_embedding_provider_for_testing(WrongDimensionProvider())

    with pytest.raises(EmbeddingError, match="dimension mismatch"):
        embedding.embed_query("python")


def test_provider_requires_api_key():
    with pytest.raises(EmbeddingError, match="GEMINI_API_KEY"):
        embedding.GeminiEmbeddingProvider("", "gemini-embedding-001", 768)
    with pytest.raises(EmbeddingError, match="OPENAI_API_KEY"):
        embedding.OpenAIEmbeddingProvider("", "text-embedding-3-small", 768)


def test_provider_errors_are_normalized():
    embedding.set_embedding_provider_for_testing(FailingProvider())

    with pytest.raises(EmbeddingError, match="fake embedding request failed: provider unavailable"):
        embedding.embed_query("python")


def test_unknown_configured_provider_is_rejected(monkeypatch):
    monkeypatch.setattr(embedding, "EMBEDDING_PROVIDER", "unsupported")
    embedding._configured_provider.cache_clear()

    with pytest.raises(EmbeddingError, match="Unsupported EMBEDDING_PROVIDER"):
        embedding.get_embedding_provider()


def test_gemini_provider_builds_current_sdk_config(monkeypatch):
    provider = embedding.GeminiEmbeddingProvider("fake-key", "gemini-embedding-001", 3)
    captured = {}

    def fake_embed_content(**kwargs):
        captured.update(kwargs)
        return type("Response", (), {"embeddings": [type("Vector", (), {"values": [1, 0, 0]})()]})()

    monkeypatch.setattr(provider.client.models, "embed_content", fake_embed_content)

    assert provider.embed(["python"], "RETRIEVAL_QUERY") == [[1, 0, 0]]
    assert captured["config"].task_type == "RETRIEVAL_QUERY"
    assert captured["config"].output_dimensionality == 3
