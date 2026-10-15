"""Supabase clients. Isolation layer 3 (database layer, Postgres RLS).

The backend NEVER uses the service-role key - there is deliberately no setting
for it. Data access uses the public anon key plus the caller's own access
token, so Postgres sees `auth.uid() = <that user>` and Row Level Security
filters every statement. Even a bug in our Python filters cannot leak another
user's rows.
"""
from fastapi import Depends
from supabase import Client, create_client
from supabase.client import ClientOptions

from app.auth import CurrentUser, get_current_user
from app.config import Settings, get_settings


def get_anon_client(settings: Settings = Depends(get_settings)) -> Client:
    """Un-authenticated client, used only for signup/login against Supabase Auth."""
    return create_client(settings.supabase_url, settings.supabase_anon_key)


def get_user_db(
    user: CurrentUser = Depends(get_current_user),
    settings: Settings = Depends(get_settings),
) -> Client:
    """A fresh client per request that acts AS the authenticated user.

    The token is injected as a default header at construction time. Note we do
    NOT use `client.postgrest.auth(token)`: in supabase-py 2.x the `.postgrest`
    property re-applies the client's own session token on access, which would
    silently put the anon key back on the request - Postgres would then see
    `auth.uid() = null` and RLS would return nothing.

    Built per request (never shared) so one user's token cannot leak into
    another user's request through a long-lived client.
    """
    return create_client(
        settings.supabase_url,
        settings.supabase_anon_key,
        options=ClientOptions(headers={"Authorization": f"Bearer {user.token}"}),
    )
