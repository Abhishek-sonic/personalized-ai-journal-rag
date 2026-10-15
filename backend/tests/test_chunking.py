from datetime import date

from app.services.chunking import MAX_CHUNK_CHARS, chunk_text, with_date_prefix


def test_empty_text_gives_no_chunks():
    assert chunk_text("") == []
    assert chunk_text("   \n  ") == []


def test_short_entry_stays_one_chunk():
    assert chunk_text("Had oats for breakfast.") == ["Had oats for breakfast."]


def test_entry_at_exact_limit_stays_one_chunk():
    text = "a" * 1000
    assert chunk_text(text) == [text]


def test_long_entry_is_split_within_size_bound():
    paragraphs = [f"Paragraph {i}. " + "Today was a long and eventful day. " * 8 for i in range(10)]
    chunks = chunk_text("\n\n".join(paragraphs))
    assert len(chunks) > 1
    assert all(len(c) <= MAX_CHUNK_CHARS for c in chunks)


def test_no_content_is_lost_when_splitting():
    sentences = [f"Sentence number {i} is here." for i in range(80)]
    chunks = chunk_text(" ".join(sentences))
    joined = "\n".join(chunks)
    assert all(s in joined for s in sentences)


def test_consecutive_chunks_overlap():
    sentences = [f"Sentence number {i} is here." for i in range(80)]
    chunks = chunk_text(" ".join(sentences))
    for previous, following in zip(chunks, chunks[1:]):
        tail_word = previous.split()[-1]
        assert tail_word in following


def test_text_without_punctuation_is_hard_cut():
    chunks = chunk_text("x" * 3000)
    assert len(chunks) >= 3
    assert all(len(c) <= MAX_CHUNK_CHARS for c in chunks)


def test_date_prefix_includes_weekday_and_iso_date():
    # 2026-10-13 is a Tuesday.
    assert with_date_prefix("Ate eggs.", date(2026, 10, 13)) == "Journal entry from Tuesday, 2026-10-13:\nAte eggs."
