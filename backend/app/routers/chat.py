"""RAG chat endpoint, with optional Server-Sent Events streaming."""
import json
import logging
from collections.abc import Iterator

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from supabase import Client

from app.auth import CurrentUser, get_current_user
from app.db import get_user_db
from app.schemas import ChatRequest, ChatResponse, SourceOut
from app.services import rag
from app.services.llm import LLMClient, LLMUnavailableError, get_llm

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api/chat", tags=["chat"])


def _sse(event: str, data: dict | str) -> str:
    """One SSE frame. json.dumps keeps newlines in tokens from breaking the protocol."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _source_payload(sources: list[rag.Source]) -> list[dict]:
    # mode="json" so UUID/date become strings that json.dumps can serialize for SSE.
    return [SourceOut(**vars(source)).model_dump(mode="json") for source in sources]


@router.post("", response_model=None)
def chat(
    body: ChatRequest,
    user: CurrentUser = Depends(get_current_user),
    db: Client = Depends(get_user_db),
    llm: LLMClient = Depends(get_llm),
):
    if body.stream:
        return _streaming_response(body, user, db, llm)
    try:
        answer, sources = rag.answer(db, user.id, body.question, llm)
    except LLMUnavailableError as exc:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, str(exc)) from exc
    return ChatResponse(answer=answer, sources=_source_payload(sources))


def _streaming_response(body: ChatRequest, user: CurrentUser, db: Client, llm: LLMClient) -> StreamingResponse:
    # Retrieval runs before the response starts, so a provider failure here is
    # still a clean 502 rather than an error buried inside a 200 stream.
    try:
        sources, tokens = rag.answer_stream(db, user.id, body.question, llm)
    except LLMUnavailableError as exc:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, str(exc)) from exc

    def frames() -> Iterator[str]:
        # Sources first: the UI can render citations while tokens are still arriving.
        yield _sse("sources", _source_payload(sources))
        try:
            for token in tokens:
                yield _sse("token", token)
        except LLMUnavailableError as exc:
            # Headers are already sent, so report mid-stream as an `error` event.
            log.warning("[CHAT] provider unavailable mid-stream: %s", exc)
            yield _sse("error", str(exc))
        yield _sse("done", "")

    return StreamingResponse(
        frames(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
