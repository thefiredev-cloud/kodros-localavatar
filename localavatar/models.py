"""Engine and weight locations, plus a one-time fetch of the pinned SadTalker release.

Everything lives under ``~/.local/share/localavatar`` (mode 0700). Nothing here runs at
import time; ``localavatar setup`` calls :func:`fetch`.
"""

from __future__ import annotations

import os
import subprocess
import urllib.request
from dataclasses import dataclass
from pathlib import Path

DATA = Path(os.environ.get("LOCALAVATAR_HOME", Path.home() / ".local/share/localavatar"))
VENDOR = DATA / "vendor" / "SadTalker"
WEIGHTS = DATA / "models" / "sadtalker"
CHECKPOINTS = WEIGHTS / "checkpoints"

# SadTalker repo pinned to the commit that was evaluated (Apache-2.0, 2023-10-11).
SADTALKER_REPO = "https://github.com/OpenTalker/SadTalker.git"
SADTALKER_COMMIT = "cd4c0465ae0b54a6f85af57f5c65fec9fe23e7f8"

# Kokoro files are not fetched by `setup`. Put kokoro-v1.0.onnx and voices-v1.0.bin in KOKORO_DIR.
KOKORO_DIR = Path(os.environ.get("KOKORO_DIR", Path.home() / ".local/share/kokoro"))
KOKORO_MODEL = KOKORO_DIR / "kokoro-v1.0.onnx"
KOKORO_VOICES = KOKORO_DIR / "voices-v1.0.bin"

_REL = "https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/"
_FX = "https://github.com/xinntao/facexlib/releases/download/"


@dataclass(frozen=True)
class Weight:
    rel: str
    url: str
    min_bytes: int  # sanity floor: catches truncated or HTML error downloads


WEIGHT_FILES = (
    Weight("checkpoints/SadTalker_V0.0.2_256.safetensors", _REL + "SadTalker_V0.0.2_256.safetensors", 600_000_000),
    Weight("checkpoints/SadTalker_V0.0.2_512.safetensors", _REL + "SadTalker_V0.0.2_512.safetensors", 600_000_000),
    Weight("checkpoints/mapping_00109-model.pth.tar", _REL + "mapping_00109-model.pth.tar", 100_000_000),
    Weight("checkpoints/mapping_00229-model.pth.tar", _REL + "mapping_00229-model.pth.tar", 100_000_000),
    Weight("gfpgan/weights/alignment_WFLW_4HG.pth", _FX + "v0.1.0/alignment_WFLW_4HG.pth", 100_000_000),
    Weight("gfpgan/weights/detection_Resnet50_Final.pth", _FX + "v0.1.0/detection_Resnet50_Final.pth", 100_000_000),
    Weight("gfpgan/weights/parsing_parsenet.pth", _FX + "v0.2.2/parsing_parsenet.pth", 50_000_000),
    Weight(
        "gfpgan/weights/GFPGANv1.4.pth",
        "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth",
        300_000_000,
    ),
)


def missing() -> list[str]:
    out = [w.rel for w in WEIGHT_FILES if not (WEIGHTS / w.rel).is_file()]
    if not (VENDOR / "inference.py").is_file():
        out.insert(0, f"SadTalker source at {VENDOR}")
    return out


def _download(w: Weight) -> str:
    dest = WEIGHTS / w.rel
    if dest.is_file() and dest.stat().st_size >= w.min_bytes:
        return f"ok      {w.rel}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    urllib.request.urlretrieve(w.url, part)
    size = part.stat().st_size
    if size < w.min_bytes:
        part.unlink()
        raise RuntimeError(f"{w.rel}: got {size} bytes, expected at least {w.min_bytes}")
    part.rename(dest)
    return f"fetched {w.rel} ({size / 1e6:.0f} MB)"


def fetch(log=print) -> None:
    old = os.umask(0o077)
    try:
        DATA.mkdir(parents=True, exist_ok=True)
        if not (VENDOR / "inference.py").is_file():
            VENDOR.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(["git", "clone", "--depth", "1", SADTALKER_REPO, str(VENDOR)], check=True)
        head = subprocess.run(["git", "-C", str(VENDOR), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        log(f"sadtalker source {VENDOR} @ {head[:12]}" + ("" if head == SADTALKER_COMMIT else f" (evaluated {SADTALKER_COMMIT[:12]})"))
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(4) as ex:
            for line in ex.map(_download, WEIGHT_FILES):
                log(line)
    finally:
        os.umask(old)
