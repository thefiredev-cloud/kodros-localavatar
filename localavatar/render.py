"""Final ffmpeg render: aspect layout, duration fit, burned captions, loudness, encode."""

from __future__ import annotations

import subprocess

from .timing import FitPlan

ASPECTS = {
    "source": None,
    "9:16": (1080, 1920),
    "1:1": (1080, 1080),
    "16:9": (1920, 1080),
}
OUT_FPS = 25

ENCODERS = {
    "h264_nvenc": ["-c:v", "h264_nvenc", "-preset", "p5", "-tune", "hq", "-rc", "vbr", "-cq", "20",
                   "-b:v", "0", "-profile:v", "high"],
    "libx264": ["-c:v", "libx264", "-preset", "medium", "-crf", "19", "-profile:v", "high"],
}


def filter_path(p: str) -> str:
    """Escape an unquoted filename through the option and filtergraph parsers."""
    for specials in ("\\':", "\\'[],;"):
        p = "".join("\\" + c if c in specials or c.isspace() else c for c in p)
    return p


def output_size(aspect: str, src_w: int, src_h: int) -> tuple[int, int]:
    if aspect not in ASPECTS:
        raise ValueError(f"unknown aspect {aspect!r}; choose from {', '.join(ASPECTS)}")
    size = ASPECTS[aspect]
    if size is None:
        return src_w - src_w % 2, src_h - src_h % 2
    return size


def build_filtergraph(aspect: str, src_w: int, src_h: int, plan: FitPlan, ass_path: str | None) -> tuple[str, int, int]:
    out_w, out_h = output_size(aspect, src_w, src_h)
    fit = f"fps={OUT_FPS},tpad=stop_mode=clone:stop_duration={plan.pad + 0.5:.3f},trim=duration={plan.target:.3f},setpts=PTS-STARTPTS"
    if ASPECTS[aspect] is None:
        v = f"[0:v]{fit},scale={out_w}:{out_h}:flags=lanczos,setsar=1[base]"
    else:
        # Foreground fits inside the frame; background is the same video, cover-scaled,
        # blurred and darkened, so the frame has no flat bars.
        fg_h_max = out_h if out_w >= out_h else round(out_h * 0.62)
        v = (
            f"[0:v]{fit},split=2[a][b];"
            f"[a]scale={out_w}:{out_h}:force_original_aspect_ratio=increase:flags=bicubic,crop={out_w}:{out_h},"
            f"gblur=sigma=40,eq=brightness=-0.18:saturation=0.75[bg];"
            f"[b]scale={out_w}:{fg_h_max}:force_original_aspect_ratio=decrease:flags=lanczos[fg];"
            f"[bg][fg]overlay=x=(W-w)/2:y={'(H-h)/2' if out_w >= out_h else 'round(H*0.12)'},setsar=1[base]"
        )
    if ass_path:
        v += f";[base]subtitles=filename={filter_path(ass_path)}[vout]"
    else:
        v += ";[base]null[vout]"
    a = (f"[1:a]aresample=48000,apad,atrim=duration={plan.target:.3f},"
         f"loudnorm=I=-16:TP=-1.5:LRA=11,aresample=48000[aout]")
    return f"{v};{a}", out_w, out_h


def render(raw_video: str, narration: str, out_path: str, aspect: str, src_w: int, src_h: int,
           plan: FitPlan, ass_path: str | None, encoders: tuple[str, ...] = ("h264_nvenc", "libx264")) -> str:
    graph, _, _ = build_filtergraph(aspect, src_w, src_h, plan, ass_path)
    notes = ""
    for enc in encoders:
        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
               "-i", raw_video, "-i", narration,
               "-filter_complex", graph, "-map", "[vout]", "-map", "[aout]",
               *ENCODERS[enc], "-pix_fmt", "yuv420p", "-r", str(OUT_FPS),
               "-c:a", "aac", "-b:a", "192k", "-t", f"{plan.target:.3f}",
               "-movflags", "+faststart", out_path]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode == 0:
            return enc
        notes += f"{enc} failed: {r.stderr.strip()[-400:]} "
    raise RuntimeError("all encoders failed: " + notes)
