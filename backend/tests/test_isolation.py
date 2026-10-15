"""Proof of multi-tenant data isolation: user B can never reach user A's data.

This is an INTEGRATION test - it needs a real Supabase project (schema.sql
applied) and two confirmed test users, because the thing under test is partly
Postgres RLS. It skips itself when the env vars are missing, so `pytest` still
passes on a clean checkout.

Run with:
    TEST_USER_A_EMAIL=... TEST_USER_A_PASSWORD=... \
    TEST_USER_B_EMAIL=... TEST_USER_B_PASSWORD=... pytest tests/test_isolation.py -v
"""
import os
import uuid

import pytest
from fastapi.testclient import TestClient

REQUIRED_VARS = ["TEST_USER_A_EMAIL", "TEST_USER_A_PASSWORD", "TEST_USER_B_EMAIL", "TEST_USER_B_PASSWORD"]
pytestmark = pytest.mark.skipif(
    not all(os.getenv(name) for name in REQUIRED_VARS),
    reason=f"set {', '.join(REQUIRED_VARS)} in .env to run the isolation test",
)

SECRET_PHRASE = f"purple-aardvark-{uuid.uuid4().hex[:8]}"


@pytest.fixture(scope="module")
def client():
    # Imported inside the fixture, not at module level: importing app.main builds
    # Settings, which would fail collection on a checkout with no .env - even
    # though these tests are meant to skip in that case.
    from app.main import create_app

    with TestClient(create_app()) as test_client:
        yield test_client


def _login(client: TestClient, which: str) -> str:
    response = client.post(
        "/api/auth/login",
        json={"email": os.environ[f"TEST_USER_{which}_EMAIL"], "password": os.environ[f"TEST_USER_{which}_PASSWORD"]},
    )
    assert response.status_code == 200, f"login for user {which} failed: {response.text}"
    return response.json()["access_token"]


def _auth(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture(scope="module")
def token_a(client):
    return _login(client, "A")


@pytest.fixture(scope="module")
def token_b(client):
    return _login(client, "B")


@pytest.fixture(scope="module")
def entry_a(client, token_a):
    """An entry owned by A containing a phrase that exists nowhere else."""
    response = client.post(
        "/api/entries",
        json={"content": f"Today I ate a {SECRET_PHRASE} sandwich for breakfast.", "entry_date": "2026-10-13"},
        headers=_auth(token_a),
    )
    assert response.status_code == 201, response.text
    entry = response.json()
    yield entry
    client.delete(f"/api/entries/{entry['id']}", headers=_auth(token_a))


# --- A can see their own data (control case: proves the test can detect data) --

def test_a_sees_own_entry(client, token_a, entry_a):
    ids = [e["id"] for e in client.get("/api/entries", headers=_auth(token_a)).json()]
    assert entry_a["id"] in ids


def test_a_can_retrieve_own_secret_via_chat(client, token_a, entry_a):
    response = client.post(
        "/api/chat", json={"question": "What did I eat for breakfast?", "stream": False}, headers=_auth(token_a)
    )
    assert response.status_code == 200, response.text
    # A's own retrieval must surface the entry; without this the B tests below
    # could pass simply because retrieval is broken for everyone.
    assert entry_a["id"] in [s["entry_id"] for s in response.json()["sources"]]


# --- B is locked out of every access path --------------------------------------

def test_b_list_excludes_a_entry(client, token_b, entry_a):
    ids = [e["id"] for e in client.get("/api/entries", headers=_auth(token_b)).json()]
    assert entry_a["id"] not in ids


def test_b_get_a_entry_is_404(client, token_b, entry_a):
    # 404, not 403: a 403 would confirm the id exists.
    assert client.get(f"/api/entries/{entry_a['id']}", headers=_auth(token_b)).status_code == 404


def test_b_update_a_entry_is_404(client, token_b, entry_a):
    response = client.put(
        f"/api/entries/{entry_a['id']}", json={"content": "overwritten by B"}, headers=_auth(token_b)
    )
    assert response.status_code == 404


def test_b_delete_a_entry_is_404(client, token_b, entry_a):
    assert client.delete(f"/api/entries/{entry_a['id']}", headers=_auth(token_b)).status_code == 404


def test_a_entry_survives_b_attempts(client, token_a, entry_a):
    """B's update/delete attempts above must not have changed A's entry."""
    response = client.get(f"/api/entries/{entry_a['id']}", headers=_auth(token_a))
    assert response.status_code == 200
    assert SECRET_PHRASE in response.json()["content"]


def test_b_chat_cannot_retrieve_a_secret(client, token_b, entry_a):
    """The RAG path is isolated too: B's retrieval returns none of A's chunks."""
    response = client.post(
        "/api/chat", json={"question": f"Tell me about the {SECRET_PHRASE} sandwich", "stream": False}, headers=_auth(token_b)
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["sources"] == []
    # With no context the server short-circuits, so the phrase can't reach the LLM.
    assert SECRET_PHRASE not in payload["answer"]


# --- Token-level checks --------------------------------------------------------

def test_no_token_is_401(client):
    assert client.get("/api/entries").status_code == 401


def test_garbage_token_is_401(client):
    assert client.get("/api/entries", headers=_auth("not.a.real.jwt")).status_code == 401


def test_user_id_in_body_is_ignored(client, token_a, token_b):
    """A spoofed user_id in the body is ignored: ownership comes from the JWT only."""
    spoofed = {"content": "B's entry", "entry_date": "2026-10-13", "user_id": "00000000-0000-0000-0000-000000000000"}
    response = client.post("/api/entries", json=spoofed, headers=_auth(token_b))
    assert response.status_code == 201
    created = response.json()
    try:
        # Stored as B's (B sees it), and A does not - the spoofed id had no effect.
        assert created["id"] in [e["id"] for e in client.get("/api/entries", headers=_auth(token_b)).json()]
        assert created["id"] not in [e["id"] for e in client.get("/api/entries", headers=_auth(token_a)).json()]
    finally:
        client.delete(f"/api/entries/{created['id']}", headers=_auth(token_b))
