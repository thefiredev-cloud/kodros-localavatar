"""Kokoro 82M ONNX narration, chunk by chunk, with exact per-chunk timing.

Uses the Kokoro model files in KOKORO_DIR. Runs on CPU on purpose: the
lip-sync model needs the GPU more. Measured 106-183 s for a 23 s script on a
heavily loaded desktop (load 35-70), so results are cached by text/voice/speed.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import wave
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

from .models import DATA, KOKORO_MODEL, KOKORO_VOICES
from .script import Chunk

RATE = 24000
CACHE = DATA / "cache" / "tts"
CACHE_VERSION = "kokoro-v1.0-onnx-cpu-2"  # bump when synthesis/pauses change
GAP_SENTENCE = 0.28
GAP_PARAGRAPH = 0.6
LEAD_IN = 0.25  # silence before the first word so the face settles
TAIL = 0.4


def lang_for(voice: str) -> str:
    """Phonemizer language for a Kokoro voice id: ``b?_*`` voices are British English."""
    return "en-gb" if voice.startswith("b") else "en-us"


@dataclass(frozen=True)
class Segment:
    start: float
    end: float
    text: str


def list_voices(voices_path: Path = KOKORO_VOICES) -> list[str]:
    with np.load(voices_path) as z:
        return sorted(z.files)


def synthesize(
    chunks: list[Chunk], voice: str, speed: float = 1.0,
    model_path: Path = KOKORO_MODEL, voices_path: Path = KOKORO_VOICES,
) -> tuple[np.ndarray, list[Segment]]:
    import onnxruntime as ort
    from kokoro_onnx import Kokoro

    ort.set_default_logger_severity(3)
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    k = Kokoro.from_session(session, str(voices_path))
    pieces: list[np.ndarray] = [np.zeros(int(RATE * LEAD_IN), dtype=np.float32)]
    t = LEAD_IN
    segments: list[Segment] = []
    for i, c in enumerate(chunks):
        samples, rate = k.create(c.text, voice=voice, speed=speed, lang=lang_for(voice))
        if rate != RATE:
            raise RuntimeError(f"unexpected Kokoro rate {rate}")
        audio = trim_silence(np.asarray(samples, dtype=np.float32))
        dur = len(audio) / RATE
        segments.append(Segment(round(t, 3), round(t + dur, 3), c.text))
        gap = TAIL if i == len(chunks) - 1 else (GAP_PARAGRAPH if c.paragraph_end else GAP_SENTENCE)
        pieces += [audio, np.zeros(int(RATE * gap), dtype=np.float32)]
        t += dur + int(RATE * gap) / RATE
    return np.concatenate(pieces), segments


def trim_silence(x: np.ndarray, thresh: float = 0.004, pad: int = int(RATE * 0.03)) -> np.ndarray:
    """Drop leading/trailing near-silence Kokoro sometimes adds, keeping a short pad."""
    idx = np.flatnonzero(np.abs(x) > thresh)
    if idx.size == 0:
        return x
    return x[max(0, idx[0] - pad): min(len(x), idx[-1] + pad + 1)]


def write_wav(path: Path, pcm: np.ndarray, rate: int = RATE) -> float:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes((np.clip(pcm, -1.0, 1.0) * 32767.0).astype("<i2").tobytes())
    return len(pcm) / rate


def segments_json(segments: list[Segment]) -> list[dict]:
    return [asdict(s) for s in segments]


def cache_key(text: str, voice: str, speed: float) -> str:
    raw = json.dumps([CACHE_VERSION, text, voice, round(speed, 3)], ensure_ascii=False)
    return hashlib.sha256(raw.encode()).hexdigest()[:24]


def load_cached(text: str, voice: str, speed: float, cache: Path = CACHE) -> tuple[Path, list[Segment]] | None:
    """Narration rendered earlier for the same text/voice/speed (re-renders skip TTS)."""
    base = cache / cache_key(text, voice, speed)
    wav, meta = base.with_suffix(".wav"), base.with_suffix(".json")
    if not (wav.is_file() and meta.is_file()):
        return None
    return wav, [Segment(**s) for s in json.loads(meta.read_text())]


def store_cached(text: str, voice: str, speed: float, wav: Path, segments: list[Segment], cache: Path = CACHE) -> None:
    cache.mkdir(parents=True, exist_ok=True)
    base = cache / cache_key(text, voice, speed)
    shutil.copyfile(wav, base.with_suffix(".wav"))
    base.with_suffix(".json").write_text(json.dumps(segments_json(segments)))
