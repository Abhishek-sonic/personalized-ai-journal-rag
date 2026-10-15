"""CRUD for journal entries.

Isolation here: the user id comes from the verified JWT (CurrentUser), every
query adds an explicit `.eq("user_id", user.id)`, and the per-request client
makes Postgres RLS enforce the same rule. Someone else's entry id therefore
behaves exactly like a nonexistent id: 404, never 403 (don't leak existence).
"""
import logging
from datetime import date, datetime, timezone
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Response, status
from supabase import Client

from app.auth import CurrentUser, get_current_user
from app.db import get_user_db
from app.schemas import EntryIn, EntryOut
from app.services import rag

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api/entries", tags=["entries"])

TABLE = "journal_entries"
NOT_FOUND = HTTPException(status.HTTP_404_NOT_FOUND, "Entry not found")


def _first_or_404(rows: list[dict]) -> dict:
    if not rows:
        raise NOT_FOUND
    return rows[0]


@router.get("", response_model=list[EntryOut])
def list_entries(user: CurrentUser = Depends(get_current_user), db: Client = Depends(get_user_db)):
    response = (
        db.table(TABLE)
        .select("*")
        .eq("user_id", user.id)
        .order("entry_date", desc=True)
        .order("created_at", desc=True)
        .execute()
    )
    return response.data


@router.post("", response_model=EntryOut, status_code=status.HTTP_201_CREATED)
def create_entry(body: EntryIn, user: CurrentUser = Depends(get_current_user), db: Client = Depends(get_user_db)):
    entry_date = body.entry_date or date.today()
    row = {"user_id": user.id, "content": body.content, "entry_date": entry_date.isoformat()}
    entry = _first_or_404(db.table(TABLE).insert(row).execute().data)

    try:
        rag.index_entry(db, user.id, entry["id"], entry_date, body.content)
    except Exception as exc:
        # An entry that was never indexed would be invisible to chat, so roll the insert back.
        log.exception("[INGEST] failed for entry=%s, rolling back", entry["id"])
        db.table(TABLE).delete().eq("id", entry["id"]).eq("user_id", user.id).execute()
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, "Could not index the entry for search; nothing was saved.") from exc
    return entry


@router.get("/{entry_id}", response_model=EntryOut)
def get_entry(entry_id: UUID, user: CurrentUser = Depends(get_current_user), db: Client = Depends(get_user_db)):
    response = db.table(TABLE).select("*").eq("id", str(entry_id)).eq("user_id", user.id).limit(1).execute()
    return _first_or_404(response.data)


@router.put("/{entry_id}", response_model=EntryOut)
def update_entry(
    entry_id: UUID,
    body: EntryIn,
    user: CurrentUser = Depends(get_current_user),
    db: Client = Depends(get_user_db),
):
    changes = {"content": body.content, "updated_at": datetime.now(timezone.utc).isoformat()}
    if body.entry_date:
        changes["entry_date"] = body.entry_date.isoformat()

    response = db.table(TABLE).update(changes).eq("id", str(entry_id)).eq("user_id", user.id).execute()
    entry = _first_or_404(response.data)  # empty result == not yours or doesn't exist

    try:
        # Re-embed so chat answers reflect the edit (old chunks are replaced).
        rag.index_entry(db, user.id, str(entry_id), date.fromisoformat(entry["entry_date"]), body.content)
    except Exception as exc:
        log.exception("[INGEST] re-index failed for entry=%s", entry_id)
        raise HTTPException(
            status.HTTP_502_BAD_GATEWAY, "Entry saved, but its search index is out of date. Save again to retry."
        ) from exc
    return entry


@router.delete("/{entry_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_entry(entry_id: UUID, user: CurrentUser = Depends(get_current_user), db: Client = Depends(get_user_db)):
    response = db.table(TABLE).delete().eq("id", str(entry_id)).eq("user_id", user.id).execute()
    _first_or_404(response.data)
    # No chunk cleanup needed: entry_chunks.entry_id is ON DELETE CASCADE, so
    # Postgres removes the entry's chunks in the same statement.
    return Response(status_code=status.HTTP_204_NO_CONTENT)
