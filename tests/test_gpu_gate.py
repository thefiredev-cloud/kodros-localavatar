"""The optional GPU-busy gate: empty by default, driven by LOCALAVATAR_GPU_BLOCK."""

import subprocess
import sys

import pytest

from localavatar import gpu, pipeline

MARKER = "localavatar_gate_probe_marker"


def test_empty_by_default_and_proc_not_scanned(monkeypatch):
    monkeypatch.delenv("LOCALAVATAR_GPU_BLOCK", raising=False)

    def boom(_path):
        raise AssertionError("/proc must not be scanned when the list is empty")

    monkeypatch.setattr(gpu.os, "listdir", boom)
    assert gpu.reserved_names() == []
    assert gpu.gpu_reserved_running() is False


def test_names_are_split_trimmed_and_lowercased(monkeypatch):
    monkeypatch.setenv("LOCALAVATAR_GPU_BLOCK", " Blender, ComfyUI ,,")
    assert gpu.reserved_names() == ["blender", "comfyui"]


def test_blocks_only_while_a_listed_process_runs(monkeypatch):
    monkeypatch.setenv("LOCALAVATAR_GPU_BLOCK", MARKER)
    assert gpu.gpu_reserved_running() is False
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)", MARKER])
    try:
        assert gpu.gpu_reserved_running() is True
    finally:
        proc.kill()
        proc.wait()
    assert gpu.gpu_reserved_running() is False


def test_pipeline_refuses_gpu_work_while_a_reserved_process_runs(tmp_path, monkeypatch):
    from localavatar.animate import AnimateOptions
    from localavatar.pipeline import MakeRequest

    photo = tmp_path / "p.png"
    photo.write_bytes(b"x")
    req = MakeRequest(photo=photo, script_text="Hello there.", out=tmp_path / "o.mp4", work_root=tmp_path / "w",
                      animate=AnimateOptions(cpu=False))
    monkeypatch.setattr(gpu, "gpu_reserved_running", lambda: True)
    monkeypatch.setattr(pipeline.tts, "synthesize", lambda *a, **k: pytest.fail("TTS ran before the gate"))
    with pytest.raises(RuntimeError, match="GPU-reserved process is running"):
        pipeline.make(req)
