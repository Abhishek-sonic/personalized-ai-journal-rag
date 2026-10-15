"""Split journal text into chunks for embedding. Pure functions: no I/O, easy to unit-test.

Strategy: journal entries are usually short, so most stay as ONE chunk (keeps the
whole thought together, which retrieves better than fragments). Only long entries
are split - on paragraph boundaries first, then sentences, and finally a hard cut
as a last resort - with a small overlap so a sentence at a boundary isn't orphaned.
"""
import re
from datetime import date

SINGLE_CHUNK_MAX_CHARS = 1000
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

# Hard ceiling on one chunk, stated explicitly because it is not just CHUNK_SIZE:
# a continuation chunk starts with the overlap tail carried over from the
# previous chunk, then a newline, then up to one full CHUNK_SIZE unit.
MAX_CHUNK_CHARS = CHUNK_OVERLAP + 1 + CHUNK_SIZE

_PARAGRAPH_BREAK = re.compile(r"\n\s*\n")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def _split_into_units(text: str, chunk_size: int) -> list[str]:
    """Break text into pieces that each fit in chunk_size: paragraphs, else sentences, else hard cut."""
    units: list[str] = []
    for paragraph in _PARAGRAPH_BREAK.split(text):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if len(paragraph) <= chunk_size:
            units.append(paragraph)
            continue
        for sentence in _SENTENCE_END.split(paragraph):
            # A single "sentence" longer than chunk_size (e.g. no punctuation) is cut by length.
            for start in range(0, len(sentence), chunk_size):
                units.append(sentence[start : start + chunk_size])
    return units


def _overlap_tail(chunk: str, overlap: int) -> str:
    """Last `overlap` chars of a chunk, starting at a word boundary so we don't begin mid-word."""
    tail = chunk[-overlap:]
    first_space = tail.find(" ")
    return tail[first_space + 1 :] if first_space != -1 else tail


def chunk_text(
    text: str,
    single_chunk_max: int = SINGLE_CHUNK_MAX_CHARS,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """Return the chunks for one entry (see MAX_CHUNK_CHARS for the size ceiling)."""
    text = text.strip()
    if not text:
        return []
    if len(text) <= single_chunk_max:
        return [text]

    chunks: list[str] = []
    current = ""
    for unit in _split_into_units(text, chunk_size):
        if current and len(current) + 1 + len(unit) > chunk_size:
            chunks.append(current)
            current = _overlap_tail(current, overlap)
        current = f"{current}\n{unit}" if current else unit
    if current:
        chunks.append(current)
    return chunks


def with_date_prefix(chunk: str, entry_date: date) -> str:
    """Text that actually gets embedded.

    Prefixing the date lets a question like "what did I eat on Tuesday?" match
    semantically, since the weekday and date become part of the vector.
    """
    return f"Journal entry from {entry_date.strftime('%A')}, {entry_date.isoformat()}:\n{chunk}"
