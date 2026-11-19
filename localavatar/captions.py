"""Burned word-by-word captions (LocalClip style: up to three words, active word popped).

Word times come from the TTS itself, not from speech recognition: each chunk's start
and end are exact (we synthesized it), and words inside a chunk are spread by
character length with a small fixed cost per word and extra time after commas. That
is an approximation inside a chunk; chunk boundaries are exact.

Event and ASS layout follow the captions module of kodros-localclip.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .tts import Segment

BREAK_PUNCT = (".", "?", "!", ",", ";", ":")


@dataclass(frozen=True)
class Word:
    start: float
    end: float
    text: str


@dataclass(frozen=True)
class CaptionStyle:
    font: str = "Inter"
    size_ratio: float = 0.046
    primary: str = "&H00FFFFFF"
    highlight: str = "&H0000E6FF"  # warm yellow (&HAABBGGRR)
    outline: str = "&H00000000"
    outline_px: float = 6
    shadow_px: float = 2
    margin_ratio: float = 0.2
    pop: int = 112
    upper: bool = True


@dataclass(frozen=True)
class Event:
    start: float
    end: float
    text: str


def _weight(word: str) -> float:
    w = 1.5 + len(re.sub(r"\W", "", word))
    if word.endswith((",", ";", ":")):
        w += 3.0  # Kokoro pauses briefly at clause punctuation
    return w


def words_from_segments(segments: list[Segment]) -> list[Word]:
    out: list[Word] = []
    for seg in segments:
        tokens = seg.text.split()
        if not tokens:
            continue
        weights = [_weight(t) for t in tokens]
        total = sum(weights)
        span = seg.end - seg.start
        t = seg.start
        for tok, w in zip(tokens, weights):
            d = span * w / total
            out.append(Word(round(t, 3), round(t + d, 3), tok))
            t += d
    return out


def ass_time(t: float) -> str:
    cs = max(0, int(round(t * 100)))
    h, rem = divmod(cs, 360000)
    m, rem = divmod(rem, 6000)
    s, cs = divmod(rem, 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def clean_word(text: str, upper: bool) -> str:
    t = text.strip().replace("\\", "/").replace("{", "(").replace("}", ")")
    t = re.sub(r"[,;:.]+$", "", t)
    return t.upper() if upper else t


def chunk_words(words: list[Word], max_words: int = 3, max_chars: int = 18, max_gap: float = 0.45) -> list[list[Word]]:
    chunks: list[list[Word]] = []
    cur: list[Word] = []
    for w in words:
        if cur:
            joined = len(" ".join(x.text for x in cur + [w]))
            if len(cur) >= max_words or joined > max_chars or w.start - cur[-1].end > max_gap:
                chunks.append(cur)
                cur = []
        cur.append(w)
        if w.text.rstrip().endswith(BREAK_PUNCT):
            chunks.append(cur)
            cur = []
    if cur:
        chunks.append(cur)
    return chunks


def caption_events(words: list[Word], total: float, style: CaptionStyle = CaptionStyle(),
                   hold: float = 0.3, bridge: float = 0.25) -> list[Event]:
    chunks = chunk_words([w for w in words if w.text.strip()])
    events: list[Event] = []
    for ci, chunk in enumerate(chunks):
        next_start = chunks[ci + 1][0].start if ci + 1 < len(chunks) else total
        end_c = min(chunk[-1].end + hold, next_start)
        if next_start - end_c < bridge:
            end_c = next_start
        end_c = min(end_c, total)
        labels = [clean_word(w.text, style.upper) for w in chunk]
        for k, w in enumerate(chunk):
            start = chunk[0].start if k == 0 else w.start
            end = chunk[k + 1].start if k + 1 < len(chunk) else end_c
            if end - start < 0.01:
                continue
            parts = [
                f"{{\\c{style.highlight}\\fscx{style.pop}\\fscy{style.pop}}}{lab}{{\\r}}" if j == k else lab
                for j, lab in enumerate(labels)
            ]
            events.append(Event(round(start, 3), round(end, 3), " ".join(parts)))
    return events


def build_ass(events: list[Event], width: int, height: int, style: CaptionStyle = CaptionStyle()) -> str:
    size = round(height * style.size_ratio)
    margin_v = round(height * style.margin_ratio)
    margin_h = round(width * 0.06)
    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {width}
PlayResY: {height}
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Word,{style.font},{size},{style.primary},{style.primary},{style.outline},&H80000000,-1,0,0,0,100,100,0,0,1,{style.outline_px},{style.shadow_px},2,{margin_h},{margin_h},{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    lines = [f"Dialogue: 0,{ass_time(e.start)},{ass_time(e.end)},Word,,0,0,0,,{e.text}" for e in events]
    return header + "\n".join(lines) + "\n"
