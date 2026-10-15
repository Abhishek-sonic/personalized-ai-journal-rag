"""Pydantic request/response models. FastAPI validates input (422) and shapes output with these.

Note what is NOT here: no request model has a `user_id` field. The caller's
identity only ever comes from the verified JWT (see auth.py).
"""
from datetime import date, datetime
from uuid import UUID

from pydantic import BaseModel, Field, field_validator


class Credentials(BaseModel):
    email: str = Field(min_length=3, max_length=254)
    password: str = Field(min_length=6, max_length=128)


class UserOut(BaseModel):
    id: str
    email: str | None = None


class AuthResponse(BaseModel):
    access_token: str
    refresh_token: str
    user: UserOut


class EntryIn(BaseModel):
    content: str = Field(min_length=1, max_length=20_000)
    entry_date: date | None = None  # defaults to today in the router

    @field_validator("content")
    @classmethod
    def _reject_blank(cls, value: str) -> str:
        # Strip at the boundary: a whitespace-only entry passes min_length but
        # produces zero chunks, which would save an entry that chat can never find.
        stripped = value.strip()
        if not stripped:
            raise ValueError("content cannot be blank")
        return stripped


class EntryOut(BaseModel):
    id: UUID
    content: str
    entry_date: date
    created_at: datetime
    updated_at: datetime


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2_000)
    stream: bool = False


class SourceOut(BaseModel):
    entry_id: UUID
    date: date
    similarity: float
    snippet: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceOut]
