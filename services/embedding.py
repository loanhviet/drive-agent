"""
Embedding Service - Uses OpenAI text-embedding-3-small.
"""

from config import OPENAI_API_KEY, EMBEDDING_MODEL


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a list of texts into vectors."""
    # TODO: Implement using OpenAI embeddings API
    # - Create OpenAI client with OPENAI_API_KEY
    # - Call embeddings.create(input=texts, model=EMBEDDING_MODEL)
    # - Return list of embedding vectors
    pass


def embed_query(query: str) -> list[float]:
    """Embed a single query text."""
    # TODO: Implement using embed_texts
    pass
