#!/usr/bin/env python3
"""Rough lip-sync check: mouth-region frame-to-frame change during speech vs silence.

    ~/.venvs/localavatar/bin/python scripts/mouth_motion.py OUT.mp4 X Y [W H]

X Y W H is the mouth box in output pixels (find it from an extracted frame). Speech
spans come from the manifest next to the video. A working lip sync shows clearly
more motion during speech than in the pauses.
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np


def main() -> int:
    if len(sys.argv) not in (4, 6):
        print(__doc__)
        return 2
    video = Path(sys.argv[1])
    x, y = int(sys.argv[2]), int(sys.argv[3])
    w, h = (int(sys.argv[4]), int(sys.argv[5])) if len(sys.argv) == 6 else (120, 60)
    manifest = json.loads(video.with_suffix(".json").read_text())
    fps = manifest["final"]["video"]["fps"]
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(video), "-vf", f"crop={w}:{h}:{x}:{y},format=gray",
                          "-f", "rawvideo", "-"], check=True, capture_output=True).stdout
    frames = np.frombuffer(raw, np.uint8).reshape(-1, h, w).astype(np.float32)
    motion = np.abs(np.diff(frames, axis=0)).mean(axis=(1, 2))
    t = np.arange(1, len(frames)) / fps
    speech = np.zeros_like(t, dtype=bool)
    for s in manifest["segments"]:
        speech |= (t >= s["start"]) & (t <= s["end"])
    sp, si = motion[speech].mean(), motion[~speech].mean()
    print(f"frames={len(frames)} speech_frames={speech.sum()} silence_frames={(~speech).sum()}")
    print(f"mouth motion: speech={sp:.2f} silence={si:.2f} ratio={sp / max(si, 1e-6):.1f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
