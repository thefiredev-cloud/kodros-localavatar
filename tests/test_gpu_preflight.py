"""CUDA context preflight: fail fast with a clear message when the GPU refuses new contexts."""

import subprocess
import sys
from types import SimpleNamespace

import pytest

from localavatar import gpu, pipeline
from localavatar.animate import AnimateOptions


def fake_run(stdout="", returncode=0):
    def run(cmd, **kw):
        assert cmd[0] == sys.executable and cmd[1] == "-c" and "cuCtxCreate_v2" in cmd[2]
        return SimpleNamespace(stdout=stdout, returncode=returncode)

    return run


def test_context_ok_or_no_gpu_is_none(monkeypatch):
    monkeypatch.setattr(subprocess, "run", fake_run("0\n"))
    assert gpu.cuda_context_error() is None
    monkeypatch.setattr(subprocess, "run", fake_run(""))  # no libcuda / no device: torch uses the CPU itself
    assert gpu.cuda_context_error() is None


def test_context_refused_names_the_driver_code_and_the_cpu_flag(monkeypatch):
    monkeypatch.setattr(subprocess, "run", fake_run("210\n"))
    msg = gpu.cuda_context_error()
    assert "driver error 210" in msg and "--cpu" in msg


def test_probe_crash_and_timeout_are_reported(monkeypatch):
    monkeypatch.setattr(subprocess, "run", fake_run("", returncode=-11))
    assert "crashed" in gpu.cuda_context_error()

    def hang(cmd, **kw):
        raise subprocess.TimeoutExpired(cmd, 1)

    monkeypatch.setattr(subprocess, "run", hang)
    assert "did not finish" in gpu.cuda_context_error()


def test_probe_script_really_runs_here():
    """The embedded probe must be valid Python and exit 0 on any host (GPU or not)."""
    r = subprocess.run([sys.executable, "-c", gpu._CUDA_PROBE], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0 and r.stdout.strip() in ("", "0", *[str(i) for i in range(1, 1000)])


def make_req(tmp_path, cpu):
    from localavatar.pipeline import MakeRequest

    photo = tmp_path / "p.png"
    photo.write_bytes(b"x")
    return MakeRequest(photo=photo, script_text="Hello there.", out=tmp_path / "o.mp4", work_root=tmp_path / "w",
                       animate=AnimateOptions(cpu=cpu))


def test_pipeline_refuses_gpu_run_before_any_work(tmp_path, monkeypatch):
    monkeypatch.setattr(gpu, "geforce_now_running", lambda: False)
    monkeypatch.setattr(gpu, "cuda_context_error", lambda: "the GPU refused to create a CUDA context")
    monkeypatch.setattr(pipeline.tts, "synthesize", lambda *a, **k: pytest.fail("TTS ran before the GPU check"))
    with pytest.raises(RuntimeError, match="refused to create a CUDA context"):
        pipeline.make(make_req(tmp_path, cpu=False))


def test_pipeline_skips_the_probe_for_cpu_runs(tmp_path, monkeypatch):
    class Reached(Exception):
        pass

    monkeypatch.setattr(gpu, "geforce_now_running", lambda: False)
    monkeypatch.setattr(gpu, "cuda_context_error", lambda: pytest.fail("probe must not run for --cpu"))
    monkeypatch.setattr(pipeline.tts, "load_cached", lambda *a, **k: (_ for _ in ()).throw(Reached()))
    with pytest.raises(Reached):
        pipeline.make(make_req(tmp_path, cpu=True))
