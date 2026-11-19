"""Audio/video duration matching.

SadTalker emits a whole number of frames at 25 fps, so its video can be a frame
short or long of the narration. The final render always pads (clones the last frame)
or trims the video so it ends with the audio, then checks the result with ffprobe.
"""

from __future__ import annotations

import json
import math
import subprocess
from dataclasses import dataclass


@dataclass(frozen=True)
class FitPlan:
    pad: float   # seconds of last-frame clone to add to the video
    trim: float  # seconds to cut from the end of the video
    target: float  # final duration (= audio duration)


def frames_for(audio_sec: float, fps: float) -> int:
    """Frames needed so the video covers the whole audio."""
    if audio_sec < 0 or fps <= 0:
        raise ValueError("audio_sec must be >= 0 and fps > 0")
    return math.ceil(round(audio_sec * fps, 6))


def fit_plan(video_sec: float, audio_sec: float) -> FitPlan:
    if video_sec < 0 or audio_sec <= 0:
        raise ValueError("durations must be positive")
    diff = audio_sec - video_sec
    return FitPlan(pad=round(max(diff, 0.0), 3), trim=round(max(-diff, 0.0), 3), target=round(audio_sec, 3))


def durations_match(video_sec: float, audio_sec: float, fps: float) -> bool:
    """True when the streams end within one frame (plus AAC padding slack) of each other."""
    return abs(video_sec - audio_sec) <= 1.0 / fps + 0.05


def probe_streams(path: str) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", path],
        check=True, capture_output=True, text=True,
    ).stdout
    info = json.loads(out)
    res: dict = {"format_sec": float(info["format"].get("duration", 0.0))}
    for s in info["streams"]:
        kind = s.get("codec_type")
        if kind in ("video", "audio") and kind not in res:
            dur = s.get("duration")
            entry = {"sec": float(dur) if dur else None, "codec": s.get("codec_name")}
            if kind == "video":
                num, den = (s.get("avg_frame_rate") or "0/1").split("/")
                entry.update(width=int(s["width"]), height=int(s["height"]),
                             fps=float(num) / float(den) if float(den) else 0.0,
                             frames=int(s["nb_frames"]) if s.get("nb_frames", "").isdigit() else None)
            res[kind] = entry
    return res


def audio_seconds(path: str) -> float:
    info = probe_streams(path)
    if "audio" not in info:
        raise ValueError(f"{path} has no audio stream")
    return info["audio"]["sec"] or info["format_sec"]
