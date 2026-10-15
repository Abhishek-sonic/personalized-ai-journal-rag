"""All reads/writes of `entry_chunks`. Isolation layer 2 (application layer).

Rule: every function takes `user_id` as a REQUIRED argument and filters on it
explicitly. There is deliberately no function here that can touch chunks
without a user id. (Layer 3, Postgres RLS, still backs this up if a bug slips in.)
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from supabase import Client


@dataclass(frozen=True)
class RetrievedChunk:
    entry_id: str
    entry_date: date
    content: str
    similarity: float


def delete_entry_chunks(db: Client, user_id: str, entry_id: str) -> None:
    # Used before re-indexing an edited entry. (Deleting the entry itself needs
    # no call here: entry_chunks.entry_id has ON DELETE CASCADE.)
    db.table("entry_chunks").delete().eq("user_id", user_id).eq("entry_id", entry_id).execute()


def insert_chunks(
    db: Client,
    user_id: str,
    entry_id: str,
    entry_date: date,
    chunks: list[str],
    embeddings: list[list[float]],
) -> None:
    rows = [
        {
            "user_id": user_id,  # RLS `with check` rejects the insert if this isn't the JWT's user
            "entry_id": entry_id,
            "chunk_index": index,
            "content": text,
            "entry_date": entry_date.isoformat(),
            "embedding": vector,
        }
        for index, (text, vector) in enumerate(zip(chunks, embeddings, strict=True))
    ]
    if rows:
        db.table("entry_chunks").insert(rows).execute()


def search(
    db: Client,
    user_id: str,
    query_embedding: list[float],
    top_k: int,
    min_similarity: float,
) -> list[RetrievedChunk]:
    """Vector search via the match_entry_chunks SQL function.

    The SQL function already restricts to auth.uid() (taken from the JWT inside
    Postgres, not from any argument). The extra `.eq("user_id", ...)` re-checks
    that on the application side too.
    """
    response = (
        db.rpc(
            "match_entry_chunks",
            {"query_embedding": query_embedding, "match_count": top_k, "min_similarity": min_similarity},
        )
        .eq("user_id", user_id)
        .execute()
    )
    return [
        RetrievedChunk(
            entry_id=row["entry_id"],
            entry_date=date.fromisoformat(row["entry_date"]),
            content=row["content"],
            similarity=row["similarity"],
        )
        for row in response.data
    ]
