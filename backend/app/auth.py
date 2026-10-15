"""JWT verification: isolation layer 1 (API layer).

The ONLY place a user_id enters the system is the `sub` claim of a verified
Supabase JWT. Nothing downstream accepts a user id from a body, query string
or any header other than `Authorization: Bearer <token>`.
"""
import logging
from dataclasses import dataclass
from functools import lru_cache

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jwt import PyJWKClient

from app.config import Settings, get_settings

log = logging.getLogger(__name__)

# auto_error=False so a missing header becomes our own 401 (FastAPI would send 403).
bearer_scheme = HTTPBearer(auto_error=False)


@dataclass(frozen=True)
class CurrentUser:
    id: str  # from the JWT `sub` claim - the trusted identity
    email: str | None
    token: str  # kept so db.py can forward it to Postgres for RLS


@lru_cache
def _jwks_client(jwks_url: str, anon_key: str) -> PyJWKClient:
    # PyJWKClient caches the fetched keys, so we don't hit Supabase on every request.
    return PyJWKClient(jwks_url, headers={"apikey": anon_key})


def _decode_token(token: str, settings: Settings) -> dict:
    """Verify signature, expiry and audience. Raises jwt.PyJWTError on any problem."""
    common = {
        "audience": "authenticated",  # Supabase sets aud=authenticated for logged-in users
        "options": {"require": ["exp", "sub"]},
        "leeway": 10,  # tolerate small clock skew between Supabase and this machine
    }
    if settings.supabase_jwt_secret:
        # Legacy projects: symmetric secret shared with Supabase.
        return jwt.decode(token, settings.supabase_jwt_secret, algorithms=["HS256"], **common)

    # Newer projects: asymmetric signing keys published via JWKS. We only hold the
    # public key, so even a leaked backend could not mint valid tokens.
    signing_key = _jwks_client(settings.jwks_url, settings.supabase_anon_key).get_signing_key_from_jwt(token)
    return jwt.decode(token, signing_key.key, algorithms=["RS256", "ES256"], **common)


def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
    settings: Settings = Depends(get_settings),
) -> CurrentUser:
    if credentials is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Missing bearer token")
    try:
        claims = _decode_token(credentials.credentials, settings)
    except jwt.PyJWTError as exc:
        # Log the reason (never the token) so expired vs forged is debuggable.
        log.warning("[AUTH] rejected token: %s", exc)
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid or expired token") from exc
    return CurrentUser(id=claims["sub"], email=claims.get("email"), token=credentials.credentials)
