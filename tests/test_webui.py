"""Web UI tests: parsing, job queue and the real HTTP server with a fake render runner.

No SadTalker, Kokoro or GPU is used; ffprobe (already required by the pipeline) checks photo size.
"""

import http.client
import json
import struct
import threading
import time
import zlib
from pathlib import Path

import pytest

from localavatar import cli, tts, webui

PHOTO = Path(__file__).resolve().parents[1] / "fixtures" / "astronaut-public-domain.png"
VOICES = ["af_heart", "am_michael", "bf_emma", "bm_george", "ef_dora", "zf_xiaobei"]
BOUNDARY = "----testboundary"


def tiny_png(w: int, h: int) -> bytes:
    def chunk(tag: bytes, data: bytes) -> bytes:
        c = struct.pack(">I", len(data)) + tag + data
        return c + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    raw = b"".join(b"\x00" + b"\x80" * w for _ in range(h))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 0, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


def multipart(fields: dict, photo: bytes | None = None, filename: str = "me.png") -> tuple[bytes, str]:
    out = b""
    for k, v in fields.items():
        out += f'--{BOUNDARY}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode()
    if photo is not None:
        out += (f'--{BOUNDARY}\r\nContent-Disposition: form-data; name="photo"; filename="{filename}"\r\n'
                f"Content-Type: application/octet-stream\r\n\r\n").encode() + photo + b"\r\n"
    out += f"--{BOUNDARY}--\r\n".encode()
    return out, f"multipart/form-data; boundary={BOUNDARY}"


GOOD = {"script": "Hello there. This is a test.", "voice": "af_heart", "speed": "1.0", "aspect": "9:16",
        "quality": "fast", "captions": "1"}


# ---------------------------------------------------------------- pure helpers


def test_voice_labels_and_english_only():
    assert webui.voice_info("af_heart")["label"] == "Heart (female)"
    assert webui.voice_info("bm_george") == {"id": "bm_george", "name": "George", "group": "UK English",
                                             "gender": "male", "label": "George (male)"}
    assert webui.voice_info("ef_dora") is None and webui.voice_info("zf_xiaobei") is None
    voices = webui.english_voices(VOICES)
    assert [v["id"] for v in voices] == ["bf_emma", "bm_george", "af_heart", "am_michael"]
    assert [v["group"] for v in voices] == ["UK English", "UK English", "US English", "US English"]


def test_lang_for_voice():
    assert tts.lang_for("bf_emma") == "en-gb" and tts.lang_for("bm_george") == "en-gb"
    assert tts.lang_for("af_heart") == "en-us" and tts.lang_for("am_michael") == "en-us"


@pytest.mark.parametrize("data,ext", [
    (b"\x89PNG\r\n\x1a\n....", ".png"), (b"\xff\xd8\xff\xe0....", ".jpg"),
    (b"RIFF\x00\x00\x00\x00WEBPVP8 ", ".webp"), (b"BM....", ".bmp"), (b"GIF89a", None), (b"%PDF-1.7", None), (b"", None),
])
def test_sniff_image(data, ext):
    assert webui.sniff_image(data) == ext


def test_image_size_and_photo_errors():
    assert webui.image_size(PHOTO.read_bytes())[0] >= 128
    assert webui.photo_size_error(PHOTO.read_bytes()) is None
    assert "at least 128" in webui.photo_size_error(tiny_png(64, 64))
    assert "could not be decoded" in webui.photo_size_error(b"\x89PNG\r\n\x1a\nnot really")


def test_parse_multipart_fields_and_file():
    body, ctype = multipart({"script": "line one\r\nline two", "voice": "af_heart"}, b"\x89PNG\x00\xff\r\n\x00bin")
    fields, files = webui.parse_multipart(ctype, body)
    assert fields["voice"] == "af_heart" and fields["script"].startswith("line one")
    assert files["photo"] == ("me.png", b"\x89PNG\x00\xff\r\n\x00bin")


def test_parse_multipart_rejects_other_types():
    with pytest.raises(ValueError):
        webui.parse_multipart("application/json", b"{}")


def test_parse_params_defaults_and_validation():
    p, errs = webui.parse_params({"script": "  hi  "}, VOICES)
    assert errs == [] and p == {"script": "hi", "voice": "af_heart", "speed": 1.0, "aspect": "9:16",
                                "quality": "fast", "captions": False, "motion": False, "cpu": False}
    p, errs = webui.parse_params({"script": "x", "captions": "1", "motion": "on", "cpu": "1", "quality": "best",
                                  "aspect": "16:9"}, VOICES)
    assert errs == [] and p["captions"] and p["motion"] and p["cpu"] and p["quality"] == "best" and p["aspect"] == "16:9"
    _, errs = webui.parse_params({"script": " ", "voice": "nope", "speed": "9", "aspect": "4:3", "quality": "ultra"}, VOICES)
    joined = "\n".join(errs)
    for needle in ("script is empty", "unknown voice", "speed must be", "aspect must be", "quality must be"):
        assert needle in joined
    _, errs = webui.parse_params({"script": "x" * 6001}, VOICES)
    assert "longer than 6000" in errs[0]
    _, errs = webui.parse_params({"script": "x", "speed": "fast"}, VOICES)
    assert "speed must be" in errs[0]
    _, errs = webui.parse_params({"script": "x", "voice": "anything"}, None)  # no voice list: skip the check
    assert errs == []


# ---------------------------------------------------------------- job queue


def fake_runner(calls=None, fail=False):
    def run(job_dir, params, photo, log):
        if calls is not None:
            calls.append((job_dir.name, params, photo.name))
        log("tts: fake")
        if fail:
            raise RuntimeError("no face found in the photo")
        (job_dir / "video.mp4").write_bytes(b"\x00" * 4096)
        (job_dir / "work").mkdir()
        (job_dir / "work" / "frames.bin").write_bytes(b"1")
        return {"final": {"video": {"sec": 3.0}}, "size": [1080, 1920]}

    return run


def wait_for(fn, timeout=10.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        v = fn()
        if v:
            return v
        time.sleep(0.02)
    raise AssertionError("timed out")


PARAMS = {"script": "Hi there.", "voice": "af_heart", "speed": 1.0, "aspect": "9:16", "quality": "fast",
          "captions": True, "motion": False}


def test_default_runner_maps_form_params_to_make_request(tmp_path, monkeypatch):
    import localavatar.pipeline as pipeline

    seen = {}
    monkeypatch.setattr(pipeline, "make", lambda req, log: seen.update(req=req, log=log) or {"ok": True})
    params = {**PARAMS, "voice": "bf_emma", "speed": 1.1, "aspect": "16:9", "quality": "best", "motion": True, "cpu": True}
    out = webui.default_runner(tmp_path, params, tmp_path / "photo.png", print)
    req = seen["req"]
    assert out == {"ok": True} and req.out == tmp_path / "video.mp4" and req.work_root == tmp_path / "work"
    assert (req.voice, req.speed, req.aspect, req.burn_captions) == ("bf_emma", 1.1, "16:9", True)
    assert (req.animate.enhancer, req.animate.still, req.animate.cpu) == ("gfpgan", False, True)
    fast = webui.default_runner(tmp_path, {**PARAMS, "cpu": False}, tmp_path / "photo.png", print)
    assert seen["req"].animate.enhancer is None and seen["req"].animate.still is True and fast == {"ok": True}



def test_job_runs_and_cleans_work_dir(tmp_path):
    calls = []
    jm = webui.JobManager(tmp_path / "jobs", fake_runner(calls))
    job = jm.submit(PARAMS, PHOTO.read_bytes(), ".png")
    done = wait_for(lambda: (j := jm.get(job["id"]))["status"] == "done" and j)
    assert done["output"] is True and done["log"] == ["tts: fake"] and done["size"] == [1080, 1920]
    assert "photo" not in done and done["runtime_sec"] is not None
    assert calls[0][2] == "photo.png"
    d = tmp_path / "jobs" / job["id"]
    assert not (d / "work").exists() and (d / "video.mp4").is_file()
    assert oct((tmp_path / "jobs").stat().st_mode & 0o777) == "0o700"
    assert jm.video_path(job["id"]) == d / "video.mp4"


def test_job_error_is_reported(tmp_path):
    jm = webui.JobManager(tmp_path / "jobs", fake_runner(fail=True))
    job = jm.submit(PARAMS, PHOTO.read_bytes(), ".png")
    j = wait_for(lambda: (j := jm.get(job["id"]))["status"] == "error" and j)
    assert "no face found" in j["error"] and j["output"] is False
    assert jm.video_path(job["id"]) is None


def test_runner_without_video_is_an_error(tmp_path):
    jm = webui.JobManager(tmp_path / "jobs", lambda d, p, ph, log: {})
    job = jm.submit(PARAMS, b"\x89PNG", ".png")
    j = wait_for(lambda: (j := jm.get(job["id"]))["status"] == "error" and j)
    assert "without writing a video" in j["error"]


def test_jobs_run_one_at_a_time_in_order(tmp_path):
    order, active, peak = [], [0], [0]
    lock = threading.Lock()

    def run(job_dir, params, photo, log):
        with lock:
            active[0] += 1
            peak[0] = max(peak[0], active[0])
        order.append(params["script"])
        time.sleep(0.05)
        (job_dir / "video.mp4").write_bytes(b"v")
        with lock:
            active[0] -= 1
        return {}

    jm = webui.JobManager(tmp_path / "jobs", run)
    ids = [jm.submit({**PARAMS, "script": f"job {i}"}, b"\x89PNG", ".png")["id"] for i in range(3)]
    wait_for(lambda: all(jm.get(i)["status"] == "done" for i in ids))
    assert order == ["job 0", "job 1", "job 2"] and peak[0] == 1


def test_cancel_queued_and_delete(tmp_path):
    gate = threading.Event()

    def run(job_dir, params, photo, log):
        gate.wait(5)
        (job_dir / "video.mp4").write_bytes(b"v")
        return {}

    jm = webui.JobManager(tmp_path / "jobs", run)
    a = jm.submit(PARAMS, b"\x89PNG", ".png")["id"]
    wait_for(lambda: jm.get(a)["status"] == "running")
    b = jm.submit(PARAMS, b"\x89PNG", ".png")["id"]
    assert jm.cancel(b) is True and jm.get(b)["error"] == "cancelled"
    assert jm.cancel(a) is False and jm.delete(a) is False  # running jobs are untouchable
    gate.set()
    wait_for(lambda: jm.get(a)["status"] == "done")
    assert jm.get(b)["status"] == "error"  # worker skipped the cancelled job
    assert jm.delete(a) is True and jm.get(a) is None and not (tmp_path / "jobs" / a).exists()
    assert jm.delete("0" * 12) is False


def test_restart_marks_unfinished_jobs_interrupted(tmp_path):
    jm = webui.JobManager(tmp_path / "jobs", fake_runner(), autostart=False)
    jid = jm.submit(PARAMS, b"\x89PNG", ".png")["id"]
    again = webui.JobManager(tmp_path / "jobs", fake_runner(), autostart=False)
    j = again.get(jid)
    assert j["status"] == "error" and "restarted" in j["error"] and j["finished"]
    assert [x["id"] for x in again.recent()] == [jid]


# ---------------------------------------------------------------- http


@pytest.fixture
def server(tmp_path):
    def fake_synth(voice, speed, path):
        path.write_bytes(b"RIFFfake-" + voice.encode() + f"-{speed}".encode())

    app = webui.App(jobs=webui.JobManager(tmp_path / "jobs", fake_runner()),
                    previews=webui.VoicePreviews(tmp_path / "previews", fake_synth),
                    voice_ids=lambda: VOICES)
    srv = webui.create_server(port=0, app=app)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    srv.port = srv.server_address[1]
    yield srv
    srv.shutdown()
    srv.server_close()


def call(srv, method, path, body=None, headers=None, host=None):
    c = http.client.HTTPConnection("127.0.0.1", srv.port, timeout=10)
    h = {"Host": host or f"127.0.0.1:{srv.port}", **(headers or {})}
    c.request(method, path, body=body, headers=h)
    r = c.getresponse()
    data = r.read()
    out = (r.status, dict(r.getheaders()), data)
    c.close()
    return out


def post_job(srv, fields=GOOD, photo=None, headers=None):
    body, ctype = multipart(fields, PHOTO.read_bytes() if photo is None else photo)
    return call(srv, "POST", "/api/jobs", body, {"Content-Type": ctype, **(headers or {})})


def job_status(srv, jid):
    return json.loads(call(srv, "GET", f"/api/jobs/{jid}")[2])["status"]


def test_serves_static_ui_with_csp_and_no_external_urls(server):
    status, hdr, body = call(server, "GET", "/")
    assert status == 200 and b"LocalAvatar" in body and "text/html" in hdr["Content-Type"]
    assert "script-src 'self'" in hdr["Content-Security-Policy"] and hdr["Cache-Control"] == "no-store"
    assert b"http://" not in body and b"https://" not in body
    for name in ("/app.js", "/app.css"):
        s, _, b = call(server, "GET", name)
        assert s == 200 and b and b"http://" not in b and b"https://" not in b
    assert call(server, "GET", "/nope")[0] == 404


def test_foreign_host_and_origin_rejected(server):
    assert call(server, "GET", "/", host="evil.example")[0] == 403
    assert call(server, "GET", "/api/jobs", host=f"127.0.0.1:{server.port + 1}")[0] == 403
    assert call(server, "GET", "/", host=f"localhost:{server.port}")[0] == 200
    s, _, b = post_job(server, headers={"Origin": "http://evil.example"})
    assert s == 403 and b"cross-origin" in b
    assert post_job(server, headers={"Sec-Fetch-Site": "cross-site"})[0] == 403
    assert json.loads(call(server, "GET", "/api/jobs")[2])["jobs"] == []
    s, _, _ = post_job(server, headers={"Origin": f"http://127.0.0.1:{server.port}", "Sec-Fetch-Site": "same-origin"})
    assert s == 201


def test_create_server_refuses_non_loopback():
    with pytest.raises(ValueError):
        webui.create_server(host="0.0.0.0", port=0)


def test_voices_endpoint_lists_english_groups(server):
    s, _, b = call(server, "GET", "/api/voices")
    data = json.loads(b)
    assert s == 200 and data["default"] == "af_heart"
    assert {v["id"] for v in data["voices"]} == {"af_heart", "am_michael", "bf_emma", "bm_george"}


def test_end_to_end_job_video_range_and_download(server):
    s, _, b = post_job(server)
    job = json.loads(b)
    assert s == 201 and job["status"] in ("queued", "running", "done") and "photo" not in job
    jid = job["id"]
    wait_for(lambda: job_status(server, jid) == "done")
    s, hdr, body = call(server, "GET", f"/api/jobs/{jid}/video")
    assert s == 200 and hdr["Content-Type"] == "video/mp4" and len(body) == 4096 and hdr["Accept-Ranges"] == "bytes"
    s, hdr, body = call(server, "GET", f"/api/jobs/{jid}/video", headers={"Range": "bytes=100-199"})
    assert s == 206 and len(body) == 100 and hdr["Content-Range"] == "bytes 100-199/4096"
    s, hdr, body = call(server, "GET", f"/api/jobs/{jid}/video", headers={"Range": "bytes=-10"})
    assert s == 206 and len(body) == 10 and hdr["Content-Range"] == "bytes 4086-4095/4096"
    assert call(server, "GET", f"/api/jobs/{jid}/video", headers={"Range": "bytes=9999-"})[0] == 416
    s, hdr, _ = call(server, "GET", f"/api/jobs/{jid}/download")
    assert s == 200 and hdr["Content-Disposition"] == f'attachment; filename="avatar-{jid}.mp4"'
    s, hdr, body = call(server, "GET", f"/api/jobs/{jid}/photo")
    assert s == 200 and hdr["Content-Type"] == "image/png" and body == PHOTO.read_bytes()
    listing = json.loads(call(server, "GET", "/api/jobs")[2])["jobs"]
    assert listing[0]["id"] == jid and listing[0]["snippet"].startswith("Hello there")
    assert call(server, "DELETE", f"/api/jobs/{jid}")[0] == 200
    assert call(server, "GET", f"/api/jobs/{jid}")[0] == 404


def test_video_not_ready_and_unknown_job(server):
    assert call(server, "GET", "/api/jobs/" + "a" * 12 + "/video")[0] == 404
    assert call(server, "GET", "/api/jobs/not-an-id")[0] == 404


@pytest.mark.parametrize("fields,photo,needle", [
    ({**GOOD, "script": ""}, None, "script is empty"),
    ({**GOOD, "voice": "zf_xiaobei"}, None, "unknown voice"),
    ({**GOOD, "aspect": "4:3"}, None, "aspect must be"),
    (GOOD, b"", "photo is missing"),
    (GOOD, b"%PDF-1.7 not an image", "must be a PNG"),
    (GOOD, tiny_png(40, 40), "at least 128"),
])
def test_bad_submissions_are_400_and_not_queued(server, fields, photo, needle):
    s, _, b = post_job(server, fields, photo)
    assert s == 400 and needle in "\n".join(json.loads(b)["errors"])
    assert json.loads(call(server, "GET", "/api/jobs")[2])["jobs"] == []


def test_non_multipart_and_oversized_bodies(server, monkeypatch):
    s, _, b = call(server, "POST", "/api/jobs", b"{}", {"Content-Type": "application/json"})
    assert s == 400 and b"multipart" in b
    monkeypatch.setattr(webui, "MAX_BODY", 1000)
    s, _, _ = call(server, "POST", "/api/jobs", b"x" * 2000, {"Content-Type": "multipart/form-data; boundary=x"})
    assert s == 413


def test_preview_is_cached_and_validated(server):
    def preview(voice="af_heart", speed=1.0):
        return call(server, "POST", "/api/preview", json.dumps({"voice": voice, "speed": speed}).encode(),
                    {"Content-Type": "application/json"})

    s, hdr, body = preview()
    assert s == 200 and hdr["Content-Type"] == "audio/wav" and body.startswith(b"RIFFfake-af_heart-1.0")
    first = server.app.previews.cache / "af_heart-1.wav"
    mtime = first.stat().st_mtime_ns
    assert preview()[0] == 200 and first.stat().st_mtime_ns == mtime  # served from cache
    assert preview("ef_dora")[0] == 400 and preview("nope")[0] == 400 and preview(speed=5)[0] == 400
    assert call(server, "POST", "/api/preview", b"not json", {"Content-Type": "application/json"})[0] == 400


def test_cancel_endpoint_rejects_finished_job(server):
    jid = json.loads(post_job(server)[2])["id"]
    wait_for(lambda: job_status(server, jid) == "done")
    assert call(server, "POST", f"/api/jobs/{jid}/cancel")[0] == 409


def test_health_reports_gpu_state(server, monkeypatch):
    from localavatar import gpu

    monkeypatch.setattr(gpu, "cuda_context_error", lambda: "the GPU refused to create a CUDA context")
    monkeypatch.setattr(gpu, "geforce_now_running", lambda: False)
    monkeypatch.setattr(gpu, "free_mib", lambda: 6000)
    s, _, b = call(server, "GET", "/api/health")
    h = json.loads(b)
    assert s == 200 and h["cuda_error"].startswith("the GPU refused") and h["geforce_now"] is False
    assert h["gpu_free_mib"] == 6000 and isinstance(h["missing"], list)


def test_serve_subcommand_parses_and_rejects_non_loopback(capsys):
    a = cli.build_parser().parse_args(["serve"])
    assert (a.port, a.host) == (8870, "127.0.0.1")
    assert cli.main(["serve", "--host", "0.0.0.0", "--port", "0"]) == 1
    assert "loopback" in capsys.readouterr().err
