"""localavatar: local talking-head videos from one photo and a script.

    localavatar make --photo FACE.jpg --script script.txt --voice af_heart --out out.mp4
                     [--aspect source|9:16|1:1|16:9] [--captions] [--size 256|512] [--motion]
    localavatar make --photo FACE.jpg --audio my-recording.m4a --out out.mp4   # your own voice
    localavatar setup     # clone pinned SadTalker + download weights (~2.5 GB)
    localavatar voices    # list Kokoro voices
    localavatar doctor    # check weights, voices, GPU, GeForce NOW gate
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .render import ASPECTS

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
MAX_SCRIPT_CHARS = 6000  # ~6-7 minutes of speech


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="localavatar", description="Local talking-head videos (SadTalker + Kokoro).")
    sub = p.add_subparsers(dest="cmd", required=True)

    m = sub.add_parser("make", help="photo + script -> talking video")
    m.add_argument("--photo", required=True, type=Path, help="front-facing portrait (jpg/png/webp)")
    src = m.add_mutually_exclusive_group(required=True)
    src.add_argument("--script", type=Path, help="text file with the narration")
    src.add_argument("--text", help="narration given inline")
    src.add_argument("--audio", type=Path, help="pre-recorded narration (wav/mp3/m4a); skips TTS, no captions")
    m.add_argument("--voice", default="af_heart", help="Kokoro voice id (see `localavatar voices`)")
    m.add_argument("--speed", type=float, default=1.0, help="speech rate 0.5-2.0")
    m.add_argument("--out", required=True, type=Path, help="output .mp4")
    m.add_argument("--aspect", choices=list(ASPECTS), default="source", help="output frame (default: photo framing)")
    m.add_argument("--captions", action="store_true", help="burn word-by-word captions")
    m.add_argument("--size", type=int, choices=(256, 512), default=256, help="face render size")
    m.add_argument("--motion", action="store_true", help="let the head move (less stable paste-back)")
    m.add_argument("--pose-style", type=int, default=0, help="SadTalker pose style 0-45 (with --motion)")
    m.add_argument("--expression", type=float, default=1.0, help="mouth/expression strength 0.5-2.0")
    m.add_argument("--no-enhance", action="store_true", help="skip GFPGAN face restoration (faster)")
    m.add_argument("--cpu", action="store_true", help="run SadTalker on CPU (very slow)")
    m.add_argument("--work-dir", type=Path, help="where intermediates go (default ~/.local/share/localavatar/runs)")

    sub.add_parser("setup", help="clone SadTalker and download weights")
    sub.add_parser("voices", help="list Kokoro voices")
    w = sub.add_parser("serve", help="local web UI (upload photo, type script, download MP4)")
    w.add_argument("--port", type=int, default=8870)
    w.add_argument("--host", default="127.0.0.1", help="loopback only")
    sub.add_parser("doctor", help="check installation")
    return p


def validate_make(args: argparse.Namespace, voices: list[str] | None) -> list[str]:
    """Return human-readable problems with a parsed `make` command (empty list = OK)."""
    errs: list[str] = []
    if not args.photo.is_file():
        errs.append(f"--photo not found: {args.photo}")
    elif args.photo.suffix.lower() not in IMAGE_EXT:
        errs.append(f"--photo must be one of {', '.join(sorted(IMAGE_EXT))}")
    if args.script is not None:
        if not args.script.is_file():
            errs.append(f"--script not found: {args.script}")
        elif not args.script.read_text(errors="replace").strip():
            errs.append("--script is empty")
        elif len(args.script.read_text(errors="replace")) > MAX_SCRIPT_CHARS:
            errs.append(f"--script is longer than {MAX_SCRIPT_CHARS} characters; split it into several videos")
    elif args.audio is not None:
        if not args.audio.is_file():
            errs.append(f"--audio not found: {args.audio}")
        if args.captions:
            errs.append("--captions needs --script or --text (no transcript for --audio)")
    elif not (args.text or "").strip():
        errs.append("--text is empty")
    if args.out.suffix.lower() != ".mp4":
        errs.append("--out must end in .mp4")
    if not 0.5 <= args.speed <= 2.0:
        errs.append("--speed must be between 0.5 and 2.0")
    if not 0.5 <= args.expression <= 2.0:
        errs.append("--expression must be between 0.5 and 2.0")
    if not 0 <= args.pose_style <= 45:
        errs.append("--pose-style must be between 0 and 45")
    if voices is not None and args.voice not in voices:
        errs.append(f"unknown --voice {args.voice!r}; try one of: {', '.join(voices[:12])} ...")
    return errs


def _voices_or_none() -> list[str] | None:
    from .models import KOKORO_VOICES
    from .tts import list_voices

    return list_voices() if KOKORO_VOICES.is_file() else None


def cmd_make(args: argparse.Namespace) -> int:
    errs = validate_make(args, _voices_or_none())
    if errs:
        for e in errs:
            print(f"error: {e}", file=sys.stderr)
        return 2
    from .animate import AnimateOptions
    from .pipeline import MakeRequest, make

    text = args.script.read_text() if args.script else (args.text or "")
    req = MakeRequest(
        photo=args.photo, script_text=text, out=args.out, voice=args.voice, speed=args.speed,
        aspect=args.aspect, burn_captions=args.captions, audio=args.audio,
        animate=AnimateOptions(size=args.size, still=not args.motion, enhancer=None if args.no_enhance else "gfpgan",
                               expression_scale=args.expression, pose_style=args.pose_style, cpu=args.cpu),
    )
    if args.work_dir:
        req.work_root = args.work_dir
    try:
        make(req)
    except (RuntimeError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


def cmd_doctor() -> int:
    from . import gpu
    from .models import KOKORO_MODEL, KOKORO_VOICES, missing

    ok = True
    gaps = missing()
    print("sadtalker:", "ok" if not gaps else "missing " + ", ".join(gaps))
    ok &= not gaps
    for f in (KOKORO_MODEL, KOKORO_VOICES):
        print(f"kokoro: {'ok' if f.is_file() else 'missing'} {f}")
        ok &= f.is_file()
    print("gpu free MiB:", gpu.free_mib())
    gfn = gpu.geforce_now_running()
    print("geforce now running:", gfn)
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.cmd == "make":
        return cmd_make(args)
    if args.cmd == "setup":
        from .models import fetch

        fetch()
        return 0
    if args.cmd == "serve":
        from .webui import serve

        try:
            return serve(args.host, args.port)
        except (ValueError, OSError) as e:
            print(f"error: {e}", file=sys.stderr)
            return 1
    if args.cmd == "voices":
        from .tts import list_voices

        print("\n".join(list_voices()))
        return 0
    return cmd_doctor()


if __name__ == "__main__":
    raise SystemExit(main())
