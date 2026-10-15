"""Prompt templates and message building. Pure functions so they are unit-testable."""
from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.services.vector_store import RetrievedChunk

NOT_FOUND_MESSAGE = "I couldn't find that in your journal."

SYSTEM_PROMPT = """You are a private journal assistant. You answer questions about the user's own journal entries.

Today's date is {today}.

Rules:
1. Answer ONLY from the journal entries in the CONTEXT below. Never use outside knowledge and never guess.
2. If the context does not contain the answer, reply exactly: "{not_found}"
3. Cite the entry behind every fact using the format [Entry: YYYY-MM-DD (Weekday)], using the date shown in the context.
4. Resolve relative dates such as "yesterday" or "last Tuesday" using today's date above.
5. Be concise and write in plain language."""


def date_label(day: date) -> str:
    """Canonical 'YYYY-MM-DD (Weekday)' label, identical to the citation format."""
    return f"{day.isoformat()} ({day.strftime('%A')})"


def format_context(chunks: list[RetrievedChunk]) -> str:
    # Numbered blocks make it easy for the model (and a human reading logs) to refer to a source.
    return "\n\n".join(
        f"[{n}] Date: {date_label(chunk.entry_date)} | Similarity: {chunk.similarity:.2f}\n{chunk.content}"
        for n, chunk in enumerate(chunks, start=1)
    )


def build_messages(question: str, chunks: list[RetrievedChunk], today: date) -> list[dict]:
    system = SYSTEM_PROMPT.format(today=date_label(today), not_found=NOT_FOUND_MESSAGE)
    user = f"CONTEXT:\n{format_context(chunks)}\n\nQUESTION: {question}"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]
