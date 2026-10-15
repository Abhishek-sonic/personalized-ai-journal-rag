from datetime import date
from types import SimpleNamespace

from app.prompts import NOT_FOUND_MESSAGE, build_messages, date_label, format_context


def _chunk(day: date, text: str, similarity: float):
    # Stand-in for RetrievedChunk: prompts only read these attributes.
    return SimpleNamespace(entry_date=day, content=text, similarity=similarity)


def test_date_label_has_weekday():
    assert date_label(date(2026, 10, 13)) == "2026-10-13 (Tuesday)"


def test_context_is_numbered_with_date_and_similarity():
    context = format_context(
        [_chunk(date(2026, 10, 13), "Ate eggs.", 0.8213), _chunk(date(2026, 10, 12), "Went running.", 0.5)]
    )
    assert "[1] Date: 2026-10-13 (Tuesday) | Similarity: 0.82\nAte eggs." in context
    assert "[2] Date: 2026-10-12 (Monday) | Similarity: 0.50\nWent running." in context


def test_messages_contain_rules_today_context_and_question():
    messages = build_messages("What did I eat?", [_chunk(date(2026, 10, 13), "Ate eggs.", 0.8)], date(2026, 10, 14))
    system, user = messages[0], messages[1]
    assert system["role"] == "system" and user["role"] == "user"
    assert "2026-10-14 (Wednesday)" in system["content"]
    assert NOT_FOUND_MESSAGE in system["content"]
    assert "[Entry: YYYY-MM-DD (Weekday)]" in system["content"]
    assert "Ate eggs." in user["content"]
    assert user["content"].endswith("QUESTION: What did I eat?")
