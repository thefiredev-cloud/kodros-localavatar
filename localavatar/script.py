"""Turn a free-form script into speakable chunks for Kokoro.

Kokoro takes at most ~510 phoneme tokens per call, so long sentences are split at
clause punctuation, then at word boundaries. Very short sentences are merged with the
next one so the voice does not stop after every two words. Blank lines mark paragraph
breaks; those get a longer pause in the audio.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

MAX_CHARS = 220  # well under Kokoro's token limit for English
MIN_CHARS = 28   # merge anything shorter into its neighbour

_SENTENCE_END = re.compile(r"(?<=[.!?…])[\"')\]]*\s+(?=[\"'(\[]?[A-Z0-9])")
_CLAUSE = re.compile(r"(?<=[,;:—–])\s+")


@dataclass(frozen=True)
class Chunk:
    text: str
    paragraph_end: bool  # last chunk of a paragraph (longer pause after it)


def normalize(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t\u00a0]+", " ", text)
    text = re.sub(r"[“”]", '"', text)
    text = re.sub(r"[‘’]", "'", text)
    return text.strip()


def sentences(paragraph: str) -> list[str]:
    flat = " ".join(paragraph.split())
    return [s.strip() for s in _SENTENCE_END.split(flat) if s.strip()]


def _split_long(sentence: str, max_chars: int) -> list[str]:
    if len(sentence) <= max_chars:
        return [sentence]
    out: list[str] = []
    cur = ""
    for clause in _CLAUSE.split(sentence):
        candidate = f"{cur} {clause}".strip()
        if len(candidate) <= max_chars:
            cur = candidate
            continue
        if cur:
            out.append(cur)
        if len(clause) <= max_chars:
            cur = clause
            continue
        # A single clause is still too long: break at word boundaries.
        cur = ""
        for word in clause.split():
            candidate = f"{cur} {word}".strip()
            if len(candidate) > max_chars and cur:
                out.append(cur)
                cur = word
            else:
                cur = candidate
    if cur:
        out.append(cur)
    return out


def _merge_short(parts: list[str], min_chars: int, max_chars: int) -> list[str]:
    out: list[str] = []
    for p in parts:
        if out and (len(out[-1]) < min_chars or len(p) < min_chars) and len(out[-1]) + 1 + len(p) <= max_chars:
            out[-1] = f"{out[-1]} {p}"
        else:
            out.append(p)
    return out


def split_script(text: str, max_chars: int = MAX_CHARS, min_chars: int = MIN_CHARS) -> list[Chunk]:
    """Split ``text`` into ordered chunks. Every word of the input appears exactly once."""
    if max_chars < 20:
        raise ValueError("max_chars must be at least 20")
    chunks: list[Chunk] = []
    for para in re.split(r"\n\s*\n", normalize(text)):
        parts: list[str] = []
        for s in sentences(para):
            parts.extend(_split_long(s, max_chars))
        parts = _merge_short(parts, min_chars, max_chars)
        for i, p in enumerate(parts):
            chunks.append(Chunk(p, paragraph_end=i == len(parts) - 1))
    return chunks


def word_count(text: str) -> int:
    return len(text.split())
