"""Embedding providers, decoupled from the chat LLM.

The rest of the app only knows the `EmbeddingProvider` protocol. Which
implementation runs is decided by EMBEDDING_PROVIDER in .env.
"""
import logging
from functools import lru_cache
from typing import Protocol

from openai import OpenAI

from app.config import Settings, get_settings

log = logging.getLogger(__name__)


class EmbeddingProvider(Protocol):
    # Documents and queries are separate methods because some models embed
    # them differently (e.g. query instruction prefixes).
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class FastEmbedProvider:
    """Runs a small ONNX model inside the backend process. No server, no API key."""

    def __init__(self, model_name: str):
        # Imported here, not at module top: fastembed pulls in onnxruntime and
        # downloads the model on first use, which we don't want when the
        # openai_compatible provider is selected.
        from fastembed import TextEmbedding

        self._model = TextEmbedding(model_name=model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        # fastembed yields numpy arrays; convert to plain lists for JSON/PostgREST.
        return [vector.tolist() for vector in self._model.embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        return next(iter(self._model.query_embed(text))).tolist()


class OpenAICompatibleEmbeddingProvider:
    """Any server exposing POST /embeddings in the OpenAI format (Ollama, OpenAI, vLLM...)."""

    def __init__(self, base_url: str, api_key: str, model: str):
        # Some local servers ignore the key, but the SDK rejects an empty one.
        self._client = OpenAI(base_url=base_url, api_key=api_key or "not-needed")
        self._model = model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        response = self._client.embeddings.create(model=self._model, input=texts)
        # The API returns an `index` per item; sort to guarantee input order.
        return [item.embedding for item in sorted(response.data, key=lambda item: item.index)]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


def get_embedding_provider(settings: Settings) -> EmbeddingProvider:
    if settings.embedding_provider == "fastembed":
        return FastEmbedProvider(settings.embedding_model)
    return OpenAICompatibleEmbeddingProvider(
        settings.embedding_base_url, settings.embedding_api_key, settings.embedding_model
    )


@lru_cache
def get_embedder() -> EmbeddingProvider:
    # Cached: loading the ONNX model is slow, so do it once per process.
    return get_embedding_provider(get_settings())


def validate_embedding_dim(provider: EmbeddingProvider, expected_dim: int) -> None:
    """Fail fast at startup if the model's output size disagrees with vector(N) in schema.sql."""
    actual = len(provider.embed_query("dimension check"))
    if actual != expected_dim:
        raise RuntimeError(
            f"Embedding model produced {actual}-dim vectors but EMBEDDING_DIM={expected_dim}. "
            "Fix EMBEDDING_DIM and the vector(N) columns in supabase/schema.sql so they match."
        )
