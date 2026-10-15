"""RAG orchestration, implemented directly (no LangChain / LlamaIndex).

    ingest:    entry text -> chunks -> embeddings -> entry_chunks rows
    answer:    question -> embedding -> top-k similar chunks -> prompt -> LLM

Every function takes `user_id` + the per-request `db` client, so retrieval can
only ever see the calling user's chunks.
"""
from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import date
from typing import TYPE_CHECKING

from app.config import get_settings
from app.prompts import NOT_FOUND_MESSAGE, build_messages
from app.services import vector_store
from app.services.chunking import chunk_text, with_date_prefix
from app.services.embeddings import get_embedder
from app.services.vector_store import RetrievedChunk

if TYPE_CHECKING:
    from supabase import Client

    from app.services.llm import LLMClient

log = logging.getLogger(__name__)

SNIPPET_CHARS = 200


@dataclass(frozen=True)
class Source:
    entry_id: str
    date: date
    similarity: float
    snippet: str


# ---------------------------------------------------------------- ingestion --

def index_entry(db: Client, user_id: str, entry_id: str, entry_date: date, content: str) -> int:
    """(Re)build the chunks for one entry. Used for both create and update.

    Embedding happens BEFORE old chunks are deleted, so if the embedding model
    fails the entry keeps its previous, still-searchable chunks.
    """
    chunks = chunk_text(content)
    # Embed the date-prefixed text, but store the raw chunk text for display.
    vectors = get_embedder().embed_documents([with_date_prefix(c, entry_date) for c in chunks])

    # Update = delete old chunks, then insert new. Simple and always consistent.
    # (On entry DELETE nothing is needed: entry_chunks has ON DELETE CASCADE.)
    vector_store.delete_entry_chunks(db, user_id, entry_id)
    vector_store.insert_chunks(db, user_id, entry_id, entry_date, chunks, vectors)
    log.info("[INGEST] entry=%s chunks=%d dim=%d", entry_id, len(chunks), len(vectors[0]) if vectors else 0)
    return len(chunks)


# ---------------------------------------------------------------- retrieval --

def retrieve(db: Client, user_id: str, question: str) -> list[RetrievedChunk]:
    settings = get_settings()
    query_vector = get_embedder().embed_query(question)
    found = vector_store.search(db, user_id, query_vector, settings.rag_top_k, settings.rag_min_similarity)

    # A long entry can match with several chunks; keep only its best one.
    # Results arrive best-first, so the first time we see an entry is its best chunk.
    best_per_entry: dict[str, RetrievedChunk] = {}
    for chunk in found:
        best_per_entry.setdefault(chunk.entry_id, chunk)
    chunks = list(best_per_entry.values())

    summary = ", ".join(f"date={c.entry_date}, sim={c.similarity:.2f}" for c in chunks) or "none"
    log.info("[RAG] retrieved %d chunks: %s", len(chunks), summary)
    return chunks


def to_sources(chunks: list[RetrievedChunk]) -> list[Source]:
    return [Source(c.entry_id, c.entry_date, round(c.similarity, 4), c.content[:SNIPPET_CHARS]) for c in chunks]


# --------------------------------------------------------------- generation --

def _prepare(db: Client, user_id: str, question: str, llm: LLMClient) -> tuple[list[RetrievedChunk], list[dict] | None]:
    log.info('[RAG] user=%s question="%s"', user_id[:8], question)
    chunks = retrieve(db, user_id, question)
    if not chunks:
        # Nothing relevant: skip the LLM entirely. Saves cost and removes any
        # chance of the model inventing an answer from thin air.
        log.info("[RAG] no relevant entries - skipping LLM call")
        return chunks, None

    messages = build_messages(question, chunks, date.today())
    approx_tokens = sum(len(m["content"]) for m in messages) // 4  # rough: ~4 chars per token
    log.info("[RAG] prompt tokens~%d, provider=%s model=%s", approx_tokens, llm.provider, llm.model)
    return chunks, messages


def answer(db: Client, user_id: str, question: str, llm: LLMClient) -> tuple[str, list[Source]]:
    chunks, messages = _prepare(db, user_id, question, llm)
    if messages is None:
        return NOT_FOUND_MESSAGE, []
    started = time.perf_counter()
    text = llm.complete(messages)
    log.info("[RAG] answer generated in %.1fs", time.perf_counter() - started)
    return text, to_sources(chunks)


def answer_stream(db: Client, user_id: str, question: str, llm: LLMClient) -> tuple[list[Source], Iterator[str]]:
    """Returns sources immediately (retrieval is done) plus a lazy token iterator."""
    chunks, messages = _prepare(db, user_id, question, llm)
    if messages is None:
        return [], iter([NOT_FOUND_MESSAGE])

    def tokens() -> Iterator[str]:
        started = time.perf_counter()
        yield from llm.stream(messages)
        log.info("[RAG] answer generated in %.1fs", time.perf_counter() - started)

    return to_sources(chunks), tokens()
