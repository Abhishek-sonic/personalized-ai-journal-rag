"""Signup/login are thin proxies to Supabase Auth: we never see or store passwords."""
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import JSONResponse
from supabase import AuthApiError, Client

from app.db import get_anon_client
from app.schemas import AuthResponse, Credentials, UserOut

router = APIRouter(prefix="/api/auth", tags=["auth"])


def _to_response(session, user) -> AuthResponse:
    return AuthResponse(
        access_token=session.access_token,
        refresh_token=session.refresh_token,
        user=UserOut(id=str(user.id), email=user.email),
    )


@router.post("/signup", response_model=AuthResponse)
def signup(body: Credentials, client: Client = Depends(get_anon_client)):
    try:
        result = client.auth.sign_up({"email": body.email, "password": body.password})
    except AuthApiError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, exc.message) from exc

    if result.session is None:
        # Supabase "Confirm email" is on: the account exists but has no session yet.
        return JSONResponse(
            status_code=status.HTTP_202_ACCEPTED,
            content={"detail": "Account created. Confirm your email, then log in."},
        )
    return _to_response(result.session, result.user)


@router.post("/login", response_model=AuthResponse)
def login(body: Credentials, client: Client = Depends(get_anon_client)):
    try:
        result = client.auth.sign_in_with_password({"email": body.email, "password": body.password})
    except AuthApiError as exc:
        # One generic message for wrong email and wrong password: don't reveal which accounts exist.
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid email or password") from exc
    return _to_response(result.session, result.user)
