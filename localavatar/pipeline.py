"""photo + script -> narrated talking-head MP4, plus a JSON manifest next to it."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from . import captions, gpu, render, timing, tts
from .animate import FPS, AnimateOptions, animate
from .models import DATA, SADTALKER_COMMIT
from .script import split_script, word_count

ENGINE = {
    "name": "SadTalker",
    "repo": "https://github.com/OpenTalker/SadTalker",
    "commit": SADTALKER_COMMIT,
    "weights": "SadTalker v0.0.2-rc release (safetensors) + GFPGAN v1.4 + facexlib detectors",
    "license": "Apache-2.0 (SadTalker code and released weights; GFPGAN Apache-2.0; facexlib MIT)",
}


@dataclass
class MakeRequest:
    photo: Path
    script_text: str
    out: Path
    voice: str = "af_heart"
    speed: float = 1.0
    aspect: str = "source"
    burn_captions: bool = False
    animate: AnimateOptions = field(default_factory=AnimateOptions)
    work_root: Path = DATA / "runs"
    audio: Path | None = None  # pre-recorded narration instead of TTS (no captions)


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def make(req: MakeRequest, log=print) -> dict:
    old = os.umask(0o077)
    try:
        return _make(req, log)
    finally:
        os.umask(old)


def _make(req: MakeRequest, log) -> dict:
    t_start = time.monotonic()
    stages: dict[str, float] = {}
    work = req.work_root / f"{req.out.stem}-{_utc()}"
    work.mkdir(parents=True, exist_ok=True)
    req.out.parent.mkdir(parents=True, exist_ok=True)

    reserved = gpu.gpu_reserved_running()
    if reserved and not req.animate.cpu:
        raise RuntimeError("a GPU-reserved process is running: refusing GPU work. Stop it or pass --cpu (very slow).")
    if not req.animate.cpu:
        cuda_err = gpu.cuda_context_error()
        if cuda_err:
            raise RuntimeError(cuda_err)

    chunks = []
    segments: list[tts.Segment] = []
    t = time.monotonic()
    if req.audio is not None:
        if req.burn_captions:
            raise ValueError("captions need a script; they are not available with --audio")
        wav = work / "narration.wav"
        # Normalize any input format to mono 24 kHz WAV for SadTalker.
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(req.audio),
                        "-ac", "1", "-ar", str(tts.RATE), "-c:a", "pcm_s16le", str(wav)], check=True)
        audio_sec = timing.audio_seconds(str(wav))
        stages["audio_import"] = round(time.monotonic() - t, 2)
        log(f"audio: {audio_sec:.2f}s from {req.audio}")
    else:
        chunks = split_script(req.script_text)
        if not chunks:
            raise ValueError("script is empty")
        log(f"script: {word_count(req.script_text)} words in {len(chunks)} chunks")
        wav = work / "narration.wav"
        cached = tts.load_cached(req.script_text, req.voice, req.speed)
        if cached:
            shutil.copyfile(cached[0], wav)
            segments = cached[1]
            audio_sec = timing.audio_seconds(str(wav))
            stages["tts_cache_hit"] = round(time.monotonic() - t, 2)
            log(f"tts: {audio_sec:.2f}s of audio from cache (Kokoro {req.voice})")
        else:
            pcm, segments = tts.synthesize(chunks, req.voice, req.speed)
            audio_sec = tts.write_wav(wav, pcm)
            tts.store_cached(req.script_text, req.voice, req.speed, wav, segments)
            stages["tts_kokoro_cpu"] = round(time.monotonic() - t, 2)
            log(f"tts: {audio_sec:.2f}s of audio in {stages['tts_kokoro_cpu']}s (Kokoro {req.voice}, CPU)")

    log(f"animate: SadTalker size {req.animate.size}, still={req.animate.still}, enhancer={req.animate.enhancer}")
    with gpu.GPUMonitor() as mon:
        raw, secs, engine_stats = animate(req.photo, wav, work, req.animate, work / "sadtalker.log")
    stages["animate_sadtalker"] = round(secs, 2)
    raw_info = timing.probe_streams(str(raw))
    log(f"animate: {raw_info['video']['sec']}s video at {raw_info['video']['width']}x{raw_info['video']['height']} in {secs:.1f}s")

    plan = timing.fit_plan(raw_info["video"]["sec"] or raw_info["format_sec"], audio_sec)
    ass_path = None
    out_w, out_h = render.output_size(req.aspect, raw_info["video"]["width"], raw_info["video"]["height"])
    if req.burn_captions:
        words = captions.words_from_segments(segments)
        events = captions.caption_events(words, plan.target)
        ass_path = work / "captions.ass"
        ass_path.write_text(captions.build_ass(events, out_w, out_h))

    t = time.monotonic()
    encoders = ("libx264",) if reserved else ("h264_nvenc", "libx264")
    enc = render.render(str(raw), str(wav), str(req.out), req.aspect,
                        raw_info["video"]["width"], raw_info["video"]["height"], plan,
                        str(ass_path) if ass_path else None, encoders)
    stages["render_ffmpeg"] = round(time.monotonic() - t, 2)

    final = timing.probe_streams(str(req.out))
    matched = timing.durations_match(final["video"]["sec"], final["audio"]["sec"], render.OUT_FPS)
    stages["total"] = round(time.monotonic() - t_start, 2)
    manifest = {
        "created_utc": _utc(),
        "output": str(req.out),
        "work_dir": str(work),
        "photo": str(req.photo),
        "voice": None if req.audio else req.voice,
        "speed": None if req.audio else req.speed,
        "audio_input": str(req.audio) if req.audio else None,
        "aspect": req.aspect,
        "size": [out_w, out_h],
        "captions": bool(ass_path),
        "caption_timing": "chunk bounds exact from TTS; words spread by length inside a chunk" if ass_path else None,
        "chunks": [asdict(c) for c in chunks],
        "segments": tts.segments_json(segments),
        "audio_sec": round(audio_sec, 3),
        "raw_video": raw_info,
        "fit": asdict(plan),
        "final": final,
        "durations_match": matched,
        "encoder": enc,
        "engine": ENGINE,
        "animate_options": asdict(req.animate),
        "sadtalker_fps": FPS,
        "runtime_sec": stages,
        "gpu": mon.summary(),
        "engine_stats": engine_stats,
    }
    manifest_path = req.out.with_suffix(".json")
    manifest_path.write_text(json.dumps(manifest, indent=2))
    if not matched:
        raise RuntimeError(f"duration mismatch: video {final['video']['sec']}s vs audio {final['audio']['sec']}s ({manifest_path})")
    log(f"done: {req.out} ({final['video']['sec']}s, {out_w}x{out_h}, {enc}) in {stages['total']}s")
    return manifest
