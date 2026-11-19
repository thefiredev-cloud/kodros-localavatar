"""Local web UI: upload a photo, type a script, pick voice and frame, download the MP4.

Standard library only. Binds loopback, rejects foreign Host/Origin headers, loads nothing
from the network. Videos render one at a time (one GPU), queued in submit order.
Jobs live in ``<LOCALAVATAR_HOME>/web/jobs/<id>/`` (mode 0700): photo, script, ``job.json``,
``video.mp4`` and the pipeline manifest ``video.json``.
"""

from __future__ import annotations

import json
import mimetypes
import os
import queue
import re
import secrets
import shutil
import subprocess
import threading
import time
from datetime import datetime, timezone
from email.parser import BytesParser
from email.policy import HTTP
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

from .models import DATA
from .render import ASPECTS

PORT = 8870
WEB_DIR = Path(__file__).with_name("web")
JOBS_DIR = DATA / "web" / "jobs"
PREVIEW_DIR = DATA / "cache" / "preview"
PREVIEW_TEXT = "Hello, this is how I sound. I can read your script for the video."
MAX_BODY = 25 * 1024 * 1024
MAX_PHOTO = 20 * 1024 * 1024
MIN_PHOTO_PX = 128
MAX_SCRIPT_CHARS = 6000  # same cap as the CLI
QUALITIES = ("fast", "best")  # fast = no GFPGAN restore, best = GFPGAN v1.4
JOB_ID = re.compile(r"^[0-9a-f]{12}$")
KEEP_LOG = 300

LANG_GROUPS = {"a": "US English", "b": "UK English"}
GENDERS = {"f": "female", "m": "male"}


# ---------------------------------------------------------------- voices


def voice_info(voice_id: str) -> dict | None:
    """Label an English Kokoro voice (``af_heart`` -> Heart, US English, female).

    Other-language voices are skipped: the TTS stage phonemizes English only.
    """
    m = re.fullmatch(r"([ab])([fm])_([a-z0-9]+)", voice_id)
    if not m:
        return None
    lang, gender, name = m.groups()
    return {"id": voice_id, "name": name.title(), "group": LANG_GROUPS[lang], "gender": GENDERS[gender],
            "label": f"{name.title()} ({GENDERS[gender]})"}


def english_voices(voice_ids: list[str]) -> list[dict]:
    out = [v for v in (voice_info(i) for i in voice_ids) if v]
    return sorted(out, key=lambda v: (v["group"], v["gender"] != "female", v["name"]))


# ---------------------------------------------------------------- uploads


def sniff_image(data: bytes) -> str | None:
    """Return the file extension for PNG/JPEG/WEBP/BMP bytes, else None."""
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"
    if data[:3] == b"\xff\xd8\xff":
        return ".jpg"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return ".webp"
    if data[:2] == b"BM":
        return ".bmp"
    return None


def image_size(data: bytes) -> tuple[int, int] | None:
    """Decoded (width, height) of image bytes via ffprobe, or None if ffprobe cannot read them."""
    try:
        r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                            "stream=width,height", "-of", "csv=p=0:s=x", "-i", "pipe:0"],
                           input=data, capture_output=True, timeout=20)
        w, h = r.stdout.decode().strip().split("x")
        return (int(w), int(h)) if int(w) > 0 and int(h) > 0 else None
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return None


def parse_multipart(content_type: str, body: bytes) -> tuple[dict[str, str], dict[str, tuple[str, bytes]]]:
    """Split a multipart/form-data body into text fields and (filename, bytes) files."""
    if not content_type.lower().startswith("multipart/form-data"):
        raise ValueError("expected multipart/form-data")
    msg = BytesParser(policy=HTTP).parsebytes(b"Content-Type: " + content_type.encode() + b"\r\n\r\n" + body)
    if not msg.is_multipart():
        raise ValueError("malformed multipart body")
    fields: dict[str, str] = {}
    files: dict[str, tuple[str, bytes]] = {}
    for part in msg.iter_parts():
        name = part.get_param("name", header="content-disposition")
        if not name:
            continue
        payload = part.get_payload(decode=True) or b""
        filename = part.get_filename()
        if filename is not None:
            files[name] = (filename, payload)
        else:
            fields[name] = payload.decode("utf-8", "replace")
    return fields, files


def _truthy(v: str | None) -> bool:
    return (v or "").strip().lower() in {"1", "true", "on", "yes"}


def parse_params(fields: dict[str, str], voices: list[str] | None) -> tuple[dict, list[str]]:
    """Validate form fields into job params. Returns (params, errors)."""
    errs: list[str] = []
    script = fields.get("script", "").replace("\r\n", "\n").strip()
    if not script:
        errs.append("script is empty")
    elif len(script) > MAX_SCRIPT_CHARS:
        errs.append(f"script is longer than {MAX_SCRIPT_CHARS} characters; split it into several videos")
    voice = fields.get("voice", "af_heart").strip() or "af_heart"
    if voices is not None and voice not in voices:
        errs.append(f"unknown voice {voice!r}")
    try:
        speed = float(fields.get("speed", "1.0") or 1.0)
    except ValueError:
        speed = -1.0
    if not 0.5 <= speed <= 2.0:
        errs.append("speed must be between 0.5 and 2.0")
    aspect = fields.get("aspect", "9:16") or "9:16"
    if aspect not in ASPECTS:
        errs.append(f"aspect must be one of {', '.join(ASPECTS)}")
    quality = fields.get("quality", "fast") or "fast"
    if quality not in QUALITIES:
        errs.append(f"quality must be one of {', '.join(QUALITIES)}")
    params = {"script": script, "voice": voice, "speed": speed, "aspect": aspect, "quality": quality,
              "captions": _truthy(fields.get("captions")), "motion": _truthy(fields.get("motion")),
              "cpu": _truthy(fields.get("cpu"))}
    return params, errs


# ---------------------------------------------------------------- jobs


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def default_runner(job_dir: Path, params: dict, photo: Path, log) -> dict:
    """Run the real pipeline for one job (imported lazily: pulls numpy/onnx)."""
    from .animate import AnimateOptions
    from .pipeline import MakeRequest, make

    req = MakeRequest(
        photo=photo, script_text=params["script"], out=job_dir / "video.mp4", voice=params["voice"],
        speed=params["speed"], aspect=params["aspect"], burn_captions=params["captions"],
        animate=AnimateOptions(still=not params["motion"], enhancer="gfpgan" if params["quality"] == "best" else None,
                               cpu=params["cpu"]),
        work_root=job_dir / "work",
    )
    return make(req, log=log)


class JobManager:
    """Queue of render jobs; one worker thread runs them in order."""

    def __init__(self, root: Path = JOBS_DIR, runner=default_runner, autostart: bool = True):
        self.root = root
        self.runner = runner
        self.lock = threading.RLock()
        self.q: queue.Queue[str] = queue.Queue()
        self.jobs: dict[str, dict] = {}
        old = os.umask(0o077)
        try:
            self.root.mkdir(parents=True, exist_ok=True)
        finally:
            os.umask(old)
        os.chmod(self.root, 0o700)
        self._load()
        self.worker = threading.Thread(target=self._work, name="render-worker", daemon=True)
        if autostart:
            self.worker.start()

    def _load(self) -> None:
        for p in sorted(self.root.glob("*/job.json")):
            try:
                job = json.loads(p.read_text())
            except (OSError, ValueError):
                continue
            if job.get("status") in ("queued", "running"):
                job["status"] = "error"
                job["error"] = "interrupted: the server restarted before this job finished"
                job["finished"] = _utc()
                self._save(job)
            self.jobs[job["id"]] = job

    def _save(self, job: dict) -> None:
        path = self.root / job["id"] / "job.json"
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(job))
        tmp.replace(path)

    def submit(self, params: dict, photo_bytes: bytes, ext: str) -> dict:
        jid = secrets.token_hex(6)
        old = os.umask(0o077)
        try:
            d = self.root / jid
            d.mkdir()
            (d / f"photo{ext}").write_bytes(photo_bytes)
        finally:
            os.umask(old)
        job = {"id": jid, "status": "queued", "created": _utc(), "started": None, "finished": None,
               "error": None, "log": [], "params": params, "photo": f"photo{ext}", "output": False,
               "runtime_sec": None, "video": None}
        with self.lock:
            self.jobs[jid] = job
            self._save(job)
        self.q.put(jid)
        return self.public(job)

    def log(self, jid: str, line: str) -> None:
        with self.lock:
            job = self.jobs[jid]
            job["log"] = (job["log"] + [line.rstrip()])[-KEEP_LOG:]
            self._save(job)

    def _work(self) -> None:
        while True:
            self.run_one(self.q.get())

    def run_one(self, jid: str) -> None:
        with self.lock:
            job = self.jobs.get(jid)
            if job is None or job["status"] != "queued":
                return  # cancelled or deleted while waiting
            job["status"] = "running"
            job["started"] = _utc()
            self._save(job)
        d = self.root / jid
        t0 = time.monotonic()
        status, error, manifest = "done", None, {}
        try:
            manifest = self.runner(d, job["params"], d / job["photo"], lambda line: self.log(jid, str(line))) or {}
            if not (d / "video.mp4").is_file():
                status, error = "error", "the renderer finished without writing a video"
            else:
                shutil.rmtree(d / "work", ignore_errors=True)  # raw frames are large; the manifest keeps the facts
        except Exception as e:  # noqa: BLE001 - any pipeline failure is shown to the user
            status, error = "error", f"{type(e).__name__}: {e}"[:2000]
        with self.lock:
            job.update(status=status, error=error, output=status == "done", video=manifest.get("final"),
                       size=manifest.get("size"), runtime_sec=round(time.monotonic() - t0, 1), finished=_utc())
            self._save(job)

    def cancel(self, jid: str) -> bool:
        with self.lock:
            job = self.jobs.get(jid)
            if job is None or job["status"] != "queued":
                return False
            job["status"] = "error"
            job["error"] = "cancelled"
            job["finished"] = _utc()
            self._save(job)
            return True

    def delete(self, jid: str) -> bool:
        with self.lock:
            job = self.jobs.get(jid)
            if job is None or job["status"] in ("queued", "running"):
                return False
            del self.jobs[jid]
        shutil.rmtree(self.root / jid, ignore_errors=True)
        return True

    @staticmethod
    def public(job: dict) -> dict:
        out = {k: v for k, v in job.items() if k != "photo"}
        return out

    def get(self, jid: str) -> dict | None:
        with self.lock:
            job = self.jobs.get(jid)
            return self.public(job) if job else None

    def recent(self, n: int = 20) -> list[dict]:
        with self.lock:
            jobs = sorted(self.jobs.values(), key=lambda j: j["created"], reverse=True)[:n]
            return [{"id": j["id"], "status": j["status"], "created": j["created"], "aspect": j["params"]["aspect"],
                     "voice": j["params"]["voice"], "snippet": j["params"]["script"][:70]} for j in jobs]

    def video_path(self, jid: str) -> Path | None:
        with self.lock:
            job = self.jobs.get(jid)
            if not job or job["status"] != "done":
                return None
        p = self.root / jid / "video.mp4"
        return p if p.is_file() else None

    def photo_path(self, jid: str) -> Path | None:
        with self.lock:
            job = self.jobs.get(jid)
            if not job:
                return None
        p = self.root / jid / job["photo"]
        return p if p.is_file() else None


# ---------------------------------------------------------------- previews


class VoicePreviews:
    """A short sample per voice/speed, synthesized on CPU once and cached on disk."""

    def __init__(self, cache: Path = PREVIEW_DIR, synth=None):
        self.cache = cache
        self.synth = synth
        self.lock = threading.Lock()

    def _default_synth(self, voice: str, speed: float, path: Path) -> None:
        from . import tts
        from .script import split_script

        pcm, _ = tts.synthesize(split_script(PREVIEW_TEXT), voice, speed)
        tts.write_wav(path, pcm)

    def get(self, voice: str, speed: float) -> Path:
        speed = round(speed, 2)
        old = os.umask(0o077)
        try:
            self.cache.mkdir(parents=True, exist_ok=True)
            path = self.cache / f"{voice}-{speed:g}.wav"
            with self.lock:
                if not path.is_file():
                    tmp = path.with_suffix(".part")
                    (self.synth or self._default_synth)(voice, speed, tmp)
                    tmp.replace(path)
        finally:
            os.umask(old)
        return path


# ---------------------------------------------------------------- http


def _voice_ids() -> list[str] | None:
    from .models import KOKORO_VOICES

    if not KOKORO_VOICES.is_file():
        return None
    from .tts import list_voices

    return list_voices()


class App:
    """Shared server state: job queue, previews, voice list."""

    def __init__(self, jobs: JobManager | None = None, previews: VoicePreviews | None = None, voice_ids=_voice_ids):
        self.jobs = jobs or JobManager()
        self.previews = previews or VoicePreviews()
        self._voice_ids = voice_ids
        self._voices: list[str] | None = None

    def voice_ids(self) -> list[str] | None:
        if self._voices is None:
            self._voices = self._voice_ids()
        return self._voices

    def english_ids(self) -> list[str] | None:
        ids = self.voice_ids()
        return None if ids is None else [v["id"] for v in english_voices(ids)]


def make_handler(app: App, port_getter):
    class Handler(BaseHTTPRequestHandler):
        server_version = "localavatar"
        protocol_version = "HTTP/1.0"  # one request per connection: no unread-body desync on early errors

        # -- plumbing
        def log_message(self, fmt, *args):  # quiet: no request logging of private content
            pass

        def _send(self, status: int, body: bytes, ctype: str, extra: dict | None = None) -> None:
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy",
                             "default-src 'none'; script-src 'self'; style-src 'self'; img-src 'self' blob:; "
                             "media-src 'self' blob:; connect-src 'self'; form-action 'self'; base-uri 'none'")
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)

        def _json(self, status: int, obj) -> None:
            self._send(status, json.dumps(obj).encode(), "application/json")

        def _error(self, status: int, msg: str, **extra) -> None:
            self._json(status, {"error": msg, **extra})

        def _host_ok(self) -> bool:
            port = port_getter()
            return self.headers.get("Host", "") in (f"127.0.0.1:{port}", f"localhost:{port}")

        def _origin_ok(self) -> bool:
            origin = self.headers.get("Origin")
            if origin is not None:
                host = urlsplit(origin).netloc
                if host not in (f"127.0.0.1:{port_getter()}", f"localhost:{port_getter()}"):
                    return False
            return self.headers.get("Sec-Fetch-Site", "same-origin") in ("same-origin", "none")

        def _drain(self) -> None:
            """Read and drop a declared request body so the client sees our error, not a reset."""
            try:
                n = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                return
            if 0 < n <= MAX_BODY:
                self.rfile.read(n)

        def _guard(self, write: bool) -> bool:
            if not self._host_ok():
                self._drain()
                self._error(HTTPStatus.FORBIDDEN, "bad Host header")
                return False
            if write and not self._origin_ok():
                self._drain()
                self._error(HTTPStatus.FORBIDDEN, "cross-origin request refused")
                return False
            return True

        def _body(self) -> bytes | None:
            try:
                n = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                n = -1
            if n < 0 or n > MAX_BODY:
                self._error(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, f"request body over {MAX_BODY // 1024 // 1024} MB")
                self.close_connection = True
                return None
            return self.rfile.read(n)

        def _file(self, path: Path, ctype: str, download_name: str | None = None) -> None:
            size = path.stat().st_size
            start, end, status = 0, size - 1, HTTPStatus.OK
            rng = self.headers.get("Range")
            if rng:
                m = re.fullmatch(r"bytes=(\d*)-(\d*)", rng.strip())
                if m and (m.group(1) or m.group(2)):
                    if m.group(1):
                        start = int(m.group(1))
                        end = int(m.group(2)) if m.group(2) else size - 1
                    else:
                        start = max(0, size - int(m.group(2)))
                    end = min(end, size - 1)
                    if start > end or start >= size:
                        self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                        self.send_header("Content-Range", f"bytes */{size}")
                        self.send_header("Content-Length", "0")
                        self.end_headers()
                        return
                    status = HTTPStatus.PARTIAL_CONTENT
            length = end - start + 1
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(length))
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            if status == HTTPStatus.PARTIAL_CONTENT:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            if download_name:
                self.send_header("Content-Disposition", f'attachment; filename="{download_name}"')
            self.end_headers()
            if self.command == "HEAD":
                return
            with open(path, "rb") as f:
                f.seek(start)
                left = length
                while left:
                    chunk = f.read(min(1 << 20, left))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    left -= len(chunk)

        # -- routes
        def do_HEAD(self):
            self.do_GET()

        def do_GET(self):
            if not self._guard(False):
                return
            path = urlsplit(self.path).path
            if path in ("/", "/index.html"):
                return self._static("index.html")
            if path in ("/app.js", "/app.css"):
                return self._static(path[1:])
            if path == "/api/voices":
                ids = app.voice_ids()
                if ids is None:
                    return self._error(HTTPStatus.SERVICE_UNAVAILABLE, "Kokoro voices file is missing; run `localavatar doctor`")
                return self._json(200, {"voices": english_voices(ids), "default": "af_heart"})
            if path == "/api/health":
                from . import gpu
                from .models import missing

                return self._json(200, {"missing": missing(), "geforce_now": gpu.geforce_now_running(),
                                        "gpu_free_mib": gpu.free_mib(), "cuda_error": gpu.cuda_context_error()})
            if path == "/api/jobs":
                return self._json(200, {"jobs": app.jobs.recent()})
            m = re.fullmatch(r"/api/jobs/([0-9a-f]{12})(?:/(video|download|photo))?", path)
            if m:
                jid, what = m.groups()
                if what is None:
                    job = app.jobs.get(jid)
                    return self._json(200, job) if job else self._error(HTTPStatus.NOT_FOUND, "no such job")
                if what == "photo":
                    p = app.jobs.photo_path(jid)
                    return self._file(p, mimetypes.guess_type(p.name)[0] or "application/octet-stream") if p \
                        else self._error(HTTPStatus.NOT_FOUND, "no such photo")
                p = app.jobs.video_path(jid)
                if not p:
                    return self._error(HTTPStatus.NOT_FOUND, "video not ready")
                return self._file(p, "video/mp4", f"avatar-{jid}.mp4" if what == "download" else None)
            self._error(HTTPStatus.NOT_FOUND, "not found")

        def _static(self, name: str) -> None:
            p = WEB_DIR / name
            ctype = {"index.html": "text/html; charset=utf-8", "app.js": "text/javascript; charset=utf-8",
                     "app.css": "text/css; charset=utf-8"}[name]
            self._send(200, p.read_bytes(), ctype)

        def do_POST(self):
            if not self._guard(True):
                return
            path = urlsplit(self.path).path
            if path == "/api/jobs":
                return self._create_job()
            if path == "/api/preview":
                return self._preview()
            m = re.fullmatch(r"/api/jobs/([0-9a-f]{12})/cancel", path)
            if m:
                return self._json(200, {"cancelled": True}) if app.jobs.cancel(m.group(1)) \
                    else self._error(HTTPStatus.CONFLICT, "only queued jobs can be cancelled")
            self._error(HTTPStatus.NOT_FOUND, "not found")

        def do_DELETE(self):
            if not self._guard(True):
                return
            m = re.fullmatch(r"/api/jobs/([0-9a-f]{12})", urlsplit(self.path).path)
            if m and app.jobs.delete(m.group(1)):
                return self._json(200, {"deleted": True})
            self._error(HTTPStatus.CONFLICT if m else HTTPStatus.NOT_FOUND, "cannot delete (running, queued or unknown job)")

        def _create_job(self) -> None:
            body = self._body()
            if body is None:
                return
            try:
                fields, files = parse_multipart(self.headers.get("Content-Type", ""), body)
            except ValueError as e:
                return self._error(HTTPStatus.BAD_REQUEST, str(e))
            params, errs = parse_params(fields, app.english_ids())
            photo = files.get("photo", ("", b""))[1]
            ext = None
            if not photo:
                errs.append("photo is missing")
            elif len(photo) > MAX_PHOTO:
                errs.append(f"photo is over {MAX_PHOTO // 1024 // 1024} MB")
            else:
                ext = sniff_image(photo)
                if ext is None:
                    errs.append("photo must be a PNG, JPEG, WEBP or BMP image")
                elif (size_err := photo_size_error(photo)):
                    errs.append(size_err)
            if errs:
                return self._json(400, {"errors": errs})
            self._json(201, app.jobs.submit(params, photo, ext))

        def _preview(self) -> None:
            body = self._body()
            if body is None:
                return
            try:
                req = json.loads(body or b"{}")
                voice = str(req.get("voice", "af_heart"))
                speed = float(req.get("speed", 1.0))
            except (ValueError, TypeError, AttributeError):
                return self._error(HTTPStatus.BAD_REQUEST, "bad JSON")
            ids = app.english_ids()
            if not 0.5 <= speed <= 2.0 or (ids is not None and voice not in ids) or voice_info(voice) is None:
                return self._error(HTTPStatus.BAD_REQUEST, "unknown voice or speed out of range")
            try:
                wav = app.previews.get(voice, speed)
            except Exception as e:  # noqa: BLE001
                return self._error(HTTPStatus.INTERNAL_SERVER_ERROR, f"preview failed: {e}")
            self._file(wav, "audio/wav")

    return Handler


def photo_size_error(data: bytes) -> str | None:
    size = image_size(data)
    if size is None:
        return "photo could not be decoded as an image"
    if min(size) < MIN_PHOTO_PX:
        return f"photo is {size[0]}x{size[1]}; use at least {MIN_PHOTO_PX}x{MIN_PHOTO_PX} px with a clear face"
    return None



def create_server(host: str = "127.0.0.1", port: int = PORT, app: App | None = None) -> ThreadingHTTPServer:
    if host not in ("127.0.0.1", "localhost"):
        raise ValueError("the web UI only binds to loopback")
    app = app or App()
    holder: dict = {}
    srv = ThreadingHTTPServer((host, port), make_handler(app, lambda: holder["srv"].server_address[1]))
    holder["srv"] = srv
    srv.daemon_threads = True
    srv.app = app  # type: ignore[attr-defined]
    return srv


def serve(host: str = "127.0.0.1", port: int = PORT) -> int:
    srv = create_server(host, port)
    print(f"localavatar web UI on http://127.0.0.1:{srv.server_address[1]}/  (Ctrl-C to stop)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
    return 0
