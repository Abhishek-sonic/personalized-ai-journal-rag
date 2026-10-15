-- AI Journal schema. Idempotent: safe to re-run in the Supabase SQL editor.
-- Run this once per Supabase project before starting the backend.

create extension if not exists vector;
create extension if not exists pgcrypto; -- gen_random_uuid()

-- ---------------------------------------------------------------------------
-- journal_entries: one row per journal entry, owned by the writing user.
-- ---------------------------------------------------------------------------
create table if not exists journal_entries (
    id uuid primary key default gen_random_uuid(),
    user_id uuid not null references auth.users(id) on delete cascade,
    content text not null,
    entry_date date not null default current_date,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

create index if not exists idx_journal_entries_user_id on journal_entries (user_id);

-- ---------------------------------------------------------------------------
-- entry_chunks: embedded slices of an entry, used for vector search.
-- vector(384) must match EMBEDDING_DIM in backend .env - the default
-- fastembed model (BAAI/bge-small-en-v1.5) produces 384-dim vectors.
-- If you switch embedding models/dims, alter this column and rebuild the
-- HNSW index below (pgvector cannot mix dimensions in one column).
-- ---------------------------------------------------------------------------
create table if not exists entry_chunks (
    id uuid primary key default gen_random_uuid(),
    entry_id uuid not null references journal_entries(id) on delete cascade,
    user_id uuid not null references auth.users(id) on delete cascade,
    chunk_index int not null,
    content text not null,
    entry_date date not null,
    embedding vector(384) not null
);

create index if not exists idx_entry_chunks_user_id on entry_chunks (user_id);
create index if not exists idx_entry_chunks_entry_id on entry_chunks (entry_id);

-- HNSW index for fast approximate cosine search, scoped later by user_id filter.
create index if not exists idx_entry_chunks_embedding_hnsw
    on entry_chunks using hnsw (embedding vector_cosine_ops);

-- ---------------------------------------------------------------------------
-- Row Level Security: the last line of defense. Even if application code
-- forgets a user_id filter, Postgres refuses to return or touch another
-- user's rows, because every query runs as that user's JWT identity
-- (auth.uid()), never as a superuser/service-role connection.
-- ---------------------------------------------------------------------------
alter table journal_entries enable row level security;
alter table entry_chunks enable row level security;

drop policy if exists journal_entries_select on journal_entries;
drop policy if exists journal_entries_insert on journal_entries;
drop policy if exists journal_entries_update on journal_entries;
drop policy if exists journal_entries_delete on journal_entries;

create policy journal_entries_select on journal_entries
    for select using (auth.uid() = user_id);
create policy journal_entries_insert on journal_entries
    for insert with check (auth.uid() = user_id);
create policy journal_entries_update on journal_entries
    for update using (auth.uid() = user_id) with check (auth.uid() = user_id);
create policy journal_entries_delete on journal_entries
    for delete using (auth.uid() = user_id);

drop policy if exists entry_chunks_select on entry_chunks;
drop policy if exists entry_chunks_insert on entry_chunks;
drop policy if exists entry_chunks_update on entry_chunks;
drop policy if exists entry_chunks_delete on entry_chunks;

create policy entry_chunks_select on entry_chunks
    for select using (auth.uid() = user_id);
create policy entry_chunks_insert on entry_chunks
    for insert with check (auth.uid() = user_id);
create policy entry_chunks_update on entry_chunks
    for update using (auth.uid() = user_id) with check (auth.uid() = user_id);
create policy entry_chunks_delete on entry_chunks
    for delete using (auth.uid() = user_id);

-- ---------------------------------------------------------------------------
-- match_entry_chunks: the only way the backend performs similarity search.
-- security invoker (the default, stated explicitly) means it runs as the
-- calling user, so RLS on entry_chunks still applies - but we also filter
-- by auth.uid() directly inside the function body. The caller cannot pass
-- a user_id; it is derived from the JWT inside Postgres, so a malicious
-- client can never request another user's chunks via this RPC.
-- ---------------------------------------------------------------------------
drop function if exists match_entry_chunks(vector(384), int, float);

create function match_entry_chunks(
    query_embedding vector(384),
    match_count int,
    min_similarity float
)
returns table (
    entry_id uuid,
    user_id uuid, -- returned only so the backend can re-apply its explicit .eq("user_id") filter (defense in depth)
    content text,
    entry_date date,
    similarity float
)
language sql
security invoker
as $$
    select
        entry_chunks.entry_id,
        entry_chunks.user_id,
        entry_chunks.content,
        entry_chunks.entry_date,
        1 - (entry_chunks.embedding <=> query_embedding) as similarity
    from entry_chunks
    where entry_chunks.user_id = auth.uid()
      and 1 - (entry_chunks.embedding <=> query_embedding) >= min_similarity
    order by entry_chunks.embedding <=> query_embedding
    limit match_count;
$$;
