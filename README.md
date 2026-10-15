# Personalized AI Journal (RAG MVP)

A private journal where you write entries and then ask questions about them. Answers are generated
**only** from your own entries using Retrieval-Augmented Generation, with inline citations back to the
entry they came from.

The frontend is deliberately plain. The interesting part is the backend: a swappable LLM/embedding
layer, a hand-written RAG pipeline (no LangChain), and three independent layers of tenant isolation.

## Demo video

▶️ **[Watch the 1–3 minute demo](DEMO_VIDEO_LINK_HERE)** — login, adding a note, asking about it, and the
backend terminal showing `[INGEST]` and `[RAG]` retrieval logs.

## Architecture

```
┌──────────────────────────────┐
│ Browser (plain HTML/JS/CSS)  │   token in sessionStorage; reads SSE with fetch + ReadableStream
└──────────────┬───────────────┘
               │  JSON over HTTP, Authorization: Bearer <Supabase JWT>
┌──────────────▼───────────────┐
│ FastAPI (backend/app)        │
│                              │
│  auth.py    verify JWT ──────┼──────► Supabase Auth  (JWKS / HS256; signup + login proxy)
│  routers/   entries, chat    │
│  services/  rag.py           │
│    ├── chunking.py           │
│    ├── embeddings.py ────────┼──────► FastEmbed (in-process)  │ or any OpenAI-compatible
│    │                         │        BAAI/bge-small-en-v1.5  │ /embeddings endpoint
│    ├── vector_store.py ──────┼──────► Supabase Postgres + pgvector
│    │                         │        journal_entries, entry_chunks, match_entry_chunks()
│    │                         │        RLS: auth.uid() = user_id
│    └── llm.py ───────────────┼──────► Ollama (local) │ or OpenRouter (cloud)
│                              │        both speak the OpenAI /chat/completions spec
└──────────────────────────────┘
```

Request flow for a question: verify JWT → embed question → `match_entry_chunks` RPC (user-scoped by
`auth.uid()` inside Postgres) → de-duplicate by entry → build prompt → stream tokens from the LLM →
return the answer plus a `sources` array.

## Local setup

**1. Create a Supabase project** (free tier is fine) at <https://supabase.com>.

**2. Apply the schema.** Open the project's SQL editor, paste all of
[`supabase/schema.sql`](supabase/schema.sql) and run it. It is idempotent, so re-running is safe. It
creates both tables, the pgvector HNSW index, the RLS policies and the `match_entry_chunks` function.

**3. Disable email confirmation for testing.** Authentication → Sign In / Providers → Email → turn
**Confirm email** off. Otherwise signup returns `202` and you must click a link before logging in
(the UI tells you this).

**4. Configure the backend.**

```bash
cp .env.example .env
```

Fill in `SUPABASE_URL` and `SUPABASE_ANON_KEY` from Project Settings → API Keys. Leave
`SUPABASE_JWT_SECRET` empty unless your project still uses a legacy HS256 secret — with it empty the
backend verifies tokens against the project's JWKS endpoint instead.

**5. Install Python dependencies** (Python 3.11+):

```bash
python -m venv backend/.venv
backend/.venv/Scripts/python -m pip install -r backend/requirements.txt
```

On macOS/Linux the interpreter is `backend/.venv/bin/python`.

**6. Pick an LLM provider.** Local, private, free:

```bash
ollama pull llama3.2
```

Ollama must be running (`ollama serve`; the desktop app does this for you). Or skip Ollama entirely
and use the cloud: set `LLM_PROVIDER=openrouter` and `OPENROUTER_API_KEY=...` in `.env`.

Embeddings need no setup — the default `fastembed` provider downloads a ~45 MB ONNX model into a
local cache on first use and runs in-process.

**7. Run it:**

```bash
cd backend && .venv/Scripts/python -m uvicorn app.main:app --reload
```

Open <http://localhost:8000>. API docs are at `/docs`; `/api/health` shows the active providers.

**Switching LLM provider** is one line in `.env`, then restart the server:

```bash
LLM_PROVIDER=ollama       # local:  OLLAMA_BASE_URL + OLLAMA_MODEL
LLM_PROVIDER=openrouter   # cloud:  OPENROUTER_BASE_URL + OPENROUTER_API_KEY + OPENROUTER_MODEL
```

**What you will see in the terminal.** Saving a note logs `[INGEST] entry=... chunks=1 dim=384`. Asking a
question logs `[RAG] user=... question="..."`, `[RAG] retrieved N chunks: ...` with each chunk's date and
similarity, `[RAG] prompt tokens~...` with the active provider/model, and `[RAG] answer generated in ...s`.

## LLM abstraction

Everything the app needs from a chat model is "send messages, get text back, optionally streamed".
Both Ollama and OpenRouter implement the OpenAI `/chat/completions` spec, so the entire integration
is the official `openai` SDK pointed at a configurable `base_url`. No provider-specific SDKs, no
adapter per vendor, and no framework in between.

[`services/llm.py`](backend/app/services/llm.py) defines one `LLMClient` class with `complete()` and
`stream()`. `get_llm_client(settings)` is the factory: it reads `LLM_PROVIDER` and takes the matching
base URL, key and model from `Settings` — which exposes them as the computed properties `llm_base_url`,
`llm_api_key` and `llm_model`. Because that resolution lives in [`config.py`](backend/app/config.py),
the factory itself has only one provider-specific branch: the optional `HTTP-Referer`/`X-Title`
attribution headers OpenRouter uses. **Switching provider is a one-line `.env` change and a restart;
no code changes.** The active provider, base URL and model are logged at startup and exposed on
`/api/health` — never the API key.

Embeddings are deliberately decoupled from chat: a journal indexed with one embedding model cannot be
searched with another, so embeddings should not change just because you switched chat providers.
[`services/embeddings.py`](backend/app/services/embeddings.py) defines an `EmbeddingProvider`
`Protocol` (structural typing — an implementation just needs `embed_documents` and `embed_query`,
with no base class to inherit) and two implementations: `FastEmbedProvider`, which runs locally with
no API key, and `OpenAICompatibleEmbeddingProvider` for any `/embeddings` endpoint such as Ollama's
`nomic-embed-text`. `EMBEDDING_PROVIDER` selects one. At startup the app embeds a probe string and
fails fast unless the vector length equals `EMBEDDING_DIM`, because a mismatch with `vector(384)` in
the schema would otherwise surface much later as an opaque Postgres insert error.

## Multi-tenancy & data isolation

Three independent layers. Any one of them alone would be sufficient on a good day; together, a bug in
one does not become a data leak.

**Layer 1 — API: identity comes only from a verified JWT.**
[`auth.py`](backend/app/auth.py) verifies the token's signature, expiry and `authenticated` audience
with `PyJWT`, supporting both the asymmetric JWKS endpoint (default, keys cached) and a legacy HS256
secret. `user_id` is read from the `sub` claim and nothing else. No request body, query parameter or
header can supply a user id — `schemas.py` has no `user_id` field at all, so a spoofed one is simply
ignored (there is a test for exactly this).

**Layer 2 — Application: every query is explicitly scoped.**
Every function in [`vector_store.py`](backend/app/services/vector_store.py) and every query in
[`routers/entries.py`](backend/app/routers/entries.py) takes `user_id` as a required argument and adds
`.eq("user_id", user_id)`. There is deliberately no function that can read or write chunks without a
user id, so you cannot forget to scope a query — there is no unscoped version to call by accident.

**Layer 3 — Database: Postgres RLS enforces it independently.**
This is why **the service-role key is never used, and has no setting in `config.py`**. That key
bypasses RLS; using it would collapse layers 2 and 3 into "we hope the Python is right".
[`db.py`](backend/app/db.py) instead builds a **per-request** client from the public anon key plus the
caller's own access token, so every statement reaches Postgres as that user and the RLS policies
(`auth.uid() = user_id`) filter it. Clients are per-request, never shared, so one user's token cannot
leak into another's request.

The same applies to vector search. `match_entry_chunks` is declared `security invoker` and filters on
`user_id = auth.uid()` *inside the function body* — the caller cannot pass a user id, because it is
taken from the JWT inside Postgres. A malicious client cannot request someone else's chunks even if it
crafts the RPC call by hand.

> Implementation note worth knowing: `client.postgrest.auth(token)` does **not** work on supabase-py
> 2.x — accessing the `.postgrest` property re-applies the client's own session token, silently
> restoring the anon key. Postgres would then see `auth.uid() = null` and RLS would return nothing. The
> token is therefore injected as a default header at construction time via `ClientOptions`.

**How the test proves it.** [`tests/test_isolation.py`](backend/tests/test_isolation.py) logs in as two
real users. A writes an entry containing a unique random phrase. Then, as B, it asserts: the entry is
absent from `GET /api/entries`; fetching, updating and deleting it by id all return **404** (not 403 —
403 would confirm the id exists); and asking the chat about the secret phrase returns no sources and an
answer that does not contain the phrase. It also asserts A's entry is unchanged afterwards, that a
spoofed `user_id` in the body is ignored, and that missing/garbage tokens give 401. Critically it
includes a **control case** — A *can* retrieve their own secret through chat — so the B assertions
cannot pass merely because retrieval is broken for everyone.

Run it with two confirmed test accounts:

```bash
cd backend && .venv/Scripts/python -m pytest tests/ -v
```

The isolation test skips itself if `TEST_USER_A_EMAIL` and friends are not set in `.env`, so the
chunking and prompt unit tests still run on a clean checkout.

## RAG design

**Chunking** ([`chunking.py`](backend/app/services/chunking.py)) is a pure function, so it is unit
tested without any I/O. Journal entries are short, so an entry under ~1000 characters is kept as a
**single chunk**: a whole thought embeds into a more meaningful vector than three fragments of it, and
splitting hurts retrieval more than it helps. Longer entries split on paragraph boundaries first, then
sentences, then a hard character cut as a last resort, into ~800-character chunks with ~100 characters
of overlap so a sentence spanning a boundary is still reachable from both sides.

**Date prefixing.** Each chunk is embedded as
`"Journal entry from Tuesday, 2026-10-13:\n<text>"`. The weekday and date become part of the vector,
so "what did I eat on Tuesday?" matches semantically rather than needing a date filter. The raw text
is what gets stored and displayed; only the embedded copy carries the prefix.

**Retrieval** embeds the question with `embed_query`, calls the `match_entry_chunks` RPC with
`RAG_TOP_K` (5) and `RAG_MIN_SIMILARITY` (0.3), then **de-duplicates by `entry_id`**, keeping each
entry's best chunk — otherwise one long entry could crowd out every other entry in the context.

**Generation.** If retrieval returns nothing, the LLM is **never called**: the endpoint short-circuits
and returns "I couldn't find that in your journal." That saves a round trip and removes any chance of
the model answering from its own training data. Otherwise [`prompts.py`](backend/app/prompts.py)
builds a system prompt that restricts the model to the supplied context, requires that exact
"couldn't find" sentence when the answer isn't there, and requires citations as
`[Entry: YYYY-MM-DD (Weekday)]`. Today's date is injected so "yesterday" and "last Tuesday" resolve.
The response always includes a structured `sources` array (entry id, date, similarity, snippet) so the
UI can show citations independently of whatever the model chose to write.

**Observability.** Chat requests log the truncated user id and question, each retrieved chunk with its
similarity, the approximate prompt size with the active provider/model, and the generation time;
ingestion logs `[INGEST] entry=... chunks=... dim=...`.

## Stretch goals implemented

- **Token streaming** over Server-Sent Events (`sources` event → `token` events → `done`). The
  frontend uses `fetch` + `ReadableStream` rather than `EventSource`, because `EventSource` cannot
  send an `Authorization` header or use POST. Partial frames are buffered, so a token split across two
  TCP reads is not dropped.
- **Edit/delete stay in sync with the vector store.** Editing an entry re-embeds it (old chunks
  deleted, new ones inserted) so answers reflect the edit. Embedding happens *before* the old chunks
  are deleted, so a failure leaves the previous, still-searchable chunks intact. Deleting an entry
  needs no cleanup code at all: `entry_chunks.entry_id` is `ON DELETE CASCADE`, so Postgres removes the
  chunks in the same statement. Conversely, if indexing a *new* entry fails, the insert is rolled back
  rather than leaving an entry invisible to chat.
- **Source citations** in both the prompt and the structured `sources` array, rendered under each
  answer in the UI.
- **Clear provider errors**: an unreachable LLM is a `502` with an actionable message
  ("Cannot reach ollama at http://localhost:11434/v1. Is Ollama running?") instead of a stack trace,
  including mid-stream, where it arrives as an SSE `error` event because the headers have already been
  sent.

## Known limitations & what I'd do next

- **Vector-only retrieval.** Pure semantic search misses exact tokens — a proper noun or an order
  number. I'd add hybrid search: Postgres full-text search alongside the vector query, combined with
  Reciprocal Rank Fusion.
- **Dates are only semantic.** "What did I eat on Tuesday?" works because the date is in the embedded
  text, but "what did I do in the first week of March?" needs a real filter. I'd parse date ranges out
  of the question and pass them to `match_entry_chunks` as an optional `WHERE entry_date BETWEEN` clause.
- **A fixed similarity threshold is crude.** 0.3 is empirical: with bge-small, unrelated entries often
  still score ~0.5, so weak matches do reach the context (the prompt's grounding rules are what prevent
  bad answers). I'd calibrate against an eval set instead of a hand-picked constant.
- **No conversation memory.** Each question is independent, so "what about Wednesday?" won't work. I'd
  keep a short history and rewrite follow-ups into standalone queries before retrieval.
- **No refresh-token handling.** The access token is kept in `sessionStorage` and the UI returns to the
  login screen on any 401; the refresh token returned at login is unused. Rotating it in a
  `httpOnly` cookie would be the real fix, and would also get the access token out of JS reach.
- **No rate limiting or cost ceiling.** Every question triggers an embedding call plus LLM generation.
  I'd add per-user rate limits and a token budget.
- **No eval set.** Correctness is currently judged by reading answers. I'd build a fixture journal with
  known question/answer pairs and measure retrieval recall and citation accuracy on every change.
- **Re-embedding is all-or-nothing.** Editing one word re-embeds every chunk of the entry. Fine at
  journal scale; I'd hash chunks and only re-embed changed ones if entries got long.
