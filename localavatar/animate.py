"""Talking-head animation with SadTalker (Apache-2.0), run as a child process.

SadTalker predicts 3DMM expression (lip shape, blinks) and head pose from the audio,
then renders the face with a face-vid2vid style generator. ``--preprocess full --still``
animates the face crop and pastes it back into the full photo, which keeps the original
framing and background. GFPGAN restores face detail on every frame.

The child is ``engine_run.py`` (staged model loading, lower peak VRAM) in the engine
venv, with cwd = the weights dir because SadTalker resolves ``gfpgan/weights`` relative
to the working directory.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from .models import CHECKPOINTS, VENDOR, WEIGHTS, missing

FPS = 25  # SadTalker renders at 25 fps
# SadTalker needs numpy<1.24 and torch 2.1 (cu121); it lives in its own venv.
ENGINE_PYTHON = os.environ.get("LOCALAVATAR_ENGINE_PYTHON",
                               str(Path.home() / ".venvs/localavatar-engine/bin/python"))
DRIVER = Path(__file__).with_name("engine_run.py")


@dataclass(frozen=True)
class AnimateOptions:
    size: int = 256            # face render size: 256 or 512
    still: bool = True         # keep head pose near the photo (best for full-photo paste-back)
    enhancer: str | None = "gfpgan"
    expression_scale: float = 1.0
    pose_style: int = 0
    batch_size: int = 1        # 2 OOMed on the shared 16 GB card with ~3 GB free
    cpu: bool = False


def build_command(photo: Path, audio: Path, result_dir: Path, opts: AnimateOptions,
                  python: str = ENGINE_PYTHON) -> list[str]:
    cmd = [
        python, str(DRIVER),
        "--vendor", str(VENDOR),
        "--driven_audio", str(audio),
        "--source_image", str(photo),
        "--result_dir", str(result_dir),
        "--checkpoint_dir", str(CHECKPOINTS),
        "--preprocess", "full",
        "--size", str(opts.size),
        "--batch_size", str(opts.batch_size),
        "--expression_scale", f"{opts.expression_scale:g}",
        "--pose_style", str(opts.pose_style),
    ]
    if opts.still:
        cmd.append("--still")
    if opts.enhancer:
        cmd += ["--enhancer", opts.enhancer]
    if opts.cpu:
        cmd.append("--cpu")
    return cmd


def parse_stats(log_text: str) -> dict:
    for line in reversed(log_text.splitlines()):
        if line.startswith("ENGINE_STATS "):
            return json.loads(line[len("ENGINE_STATS "):])
    return {}


def animate(photo: Path, audio: Path, work: Path, opts: AnimateOptions, log_path: Path) -> tuple[Path, float, dict]:
    """Run SadTalker. Returns (raw talking video, seconds, engine stats). Raises with the log tail on failure."""
    gaps = missing()
    if gaps:
        raise RuntimeError("SadTalker is not set up; run `localavatar setup`. Missing: " + ", ".join(gaps))
    result_dir = work / "sadtalker"
    result_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    t0 = time.monotonic()
    with open(log_path, "w") as log:
        r = subprocess.run(build_command(photo.resolve(), audio.resolve(), result_dir, opts),
                           cwd=WEIGHTS, env=env, stdout=log, stderr=subprocess.STDOUT)
    secs = time.monotonic() - t0
    text = log_path.read_text(errors="replace")
    video = result_dir / "talking.mp4"
    if r.returncode != 0 or not video.is_file():
        if "ENGINE_ERROR no face" in text:
            raise RuntimeError("no face found in the photo; use a clear, front-facing portrait")
        raise RuntimeError(f"SadTalker failed (exit {r.returncode}). Log tail:\n{text[-1500:]}")
    return video, secs, parse_stats(text)
