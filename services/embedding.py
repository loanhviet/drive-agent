"""Configurable Gemini/OpenAI embeddings with a test-friendly abstraction."""

from collections.abc import Sequence
from functools import lru_cache
from typing import Protocol

from config import (
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    EMBEDDING_PROVIDER,
    GEMINI_API_KEY,
    OPENAI_API_KEY,
)


class EmbeddingError(RuntimeError):
    """Raised for provider configuration or invalid embedding responses."""


class EmbeddingProvider(Protocol):
    name: str
    model: str
    dimension: int

    def embed(self, texts: Sequence[str], task_type: str) -> list[list[float]]: ...


class OpenAIEmbeddingProvider:
    name = "openai"

    def __init__(self, api_key: str, model: str, dimension: int):
        if not api_key:
            raise EmbeddingError("OPENAI_API_KEY is required for OpenAI embeddings")
        from openai import OpenAI

        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.dimension = dimension

    def embed(self, texts: Sequence[str], task_type: str) -> list[list[float]]:
        response = self.client.embeddings.create(
            input=list(texts),
            model=self.model,
            dimensions=self.dimension,
        )
        return [list(item.embedding) for item in response.data]


class GeminiEmbeddingProvider:
    name = "gemini"

    def __init__(self, api_key: str, model: str, dimension: int):
        if not api_key:
            raise EmbeddingError("GEMINI_API_KEY is required for Gemini embeddings")
        from google import genai

        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.dimension = dimension

    def embed(self, texts: Sequence[str], task_type: str) -> list[list[float]]:
        from google.genai import types

        config = types.EmbedContentConfig(
            taskType=task_type,
            outputDimensionality=self.dimension,
        )
        response = self.client.models.embed_content(
            model=self.model,
            contents=list(texts),
            config=config,
        )
        return [list(item.values) for item in response.embeddings]


_provider_override: EmbeddingProvider | None = None


@lru_cache(maxsize=1)
def _configured_provider() -> EmbeddingProvider:
    if EMBEDDING_PROVIDER == "gemini":
        return GeminiEmbeddingProvider(GEMINI_API_KEY, EMBEDDING_MODEL, EMBEDDING_DIM)
    if EMBEDDING_PROVIDER == "openai":
        return OpenAIEmbeddingProvider(OPENAI_API_KEY, EMBEDDING_MODEL, EMBEDDING_DIM)
    raise EmbeddingError(f"Unsupported EMBEDDING_PROVIDER: {EMBEDDING_PROVIDER}")


def get_embedding_provider() -> EmbeddingProvider:
    return _provider_override or _configured_provider()


def set_embedding_provider_for_testing(provider: EmbeddingProvider | None) -> None:
    """Override the provider without changing environment configuration."""
    global _provider_override
    _provider_override = provider
    _configured_provider.cache_clear()
    _embed_query_cached.cache_clear()


def _validate_inputs(texts: Sequence[str]) -> list[str]:
    normalized = [text.strip() for text in texts]
    if not normalized:
        raise EmbeddingError("At least one text is required for embedding")
    if any(not text for text in normalized):
        raise EmbeddingError("Texts to embed must not be empty")
    return normalized


def _embed(texts: Sequence[str], task_type: str) -> list[list[float]]:
    normalized = _validate_inputs(texts)
    provider = get_embedding_provider()
    try:
        vectors = provider.embed(normalized, task_type)
    except EmbeddingError:
        raise
    except Exception as error:
        raise EmbeddingError(f"{provider.name} embedding request failed: {error}") from error
    if len(vectors) != len(normalized):
        raise EmbeddingError("Embedding provider returned an unexpected number of vectors")
    for vector in vectors:
        if len(vector) != provider.dimension:
            raise EmbeddingError(
                f"Embedding dimension mismatch: expected {provider.dimension}, got {len(vector)}"
            )
    return vectors


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Batch embed documents for retrieval."""
    return _embed(texts, task_type="RETRIEVAL_DOCUMENT")


@lru_cache(maxsize=128)
def _embed_query_cached(query: str, provider_name: str, model: str, dimension: int) -> tuple[float, ...]:
    del provider_name, model, dimension
    return tuple(_embed([query], task_type="RETRIEVAL_QUERY")[0])


def embed_query(query: str) -> list[float]:
    """Embed one search query, reusing a small in-memory query cache."""
    provider = get_embedding_provider()
    return list(_embed_query_cached(query, provider.name, provider.model, provider.dimension))
