"""The SadTalker source must end up at the pinned commit, not at the default branch tip."""

import subprocess
from pathlib import Path

import pytest

from localavatar import models

GIT = ["git", "-c", "user.name=test", "-c", "user.email=test@example.invalid"]


def _git(cwd: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run([*GIT, "-C", str(cwd), *args], capture_output=True, text=True, check=True)


def _source_repo(root: Path) -> tuple[Path, str]:
    """A repo whose default branch is one commit ahead of the pinned commit."""
    src = root / "src"
    src.mkdir()
    _git(src, "init", "-q", "-b", "main")
    (src / "inference.py").write_text("pinned\n")
    _git(src, "add", "-A")
    _git(src, "commit", "-qm", "pinned")
    pinned = _git(src, "rev-parse", "HEAD").stdout.strip()
    (src / "inference.py").write_text("default branch\n")
    _git(src, "add", "-A")
    _git(src, "commit", "-qm", "later")
    # A local transport runs upload-pack in this repo, so this is the server-side switch.
    _git(src, "config", "uploadpack.allowAnySHA1InWant", "true")
    return src, pinned


def test_sync_source_checks_out_the_pinned_commit(tmp_path):
    src, pinned = _source_repo(tmp_path)
    dest = tmp_path / "vendor" / "SadTalker"

    got = models.sync_source(dest, str(src), pinned, log=lambda *_: None)

    assert got == pinned
    assert _git(dest, "rev-parse", "HEAD").stdout.strip() == pinned
    assert (dest / "inference.py").read_text() == "pinned\n"


def test_sync_source_moves_a_default_branch_checkout_to_the_pin(tmp_path):
    """An older release cloned the default branch; setup must move it to the pin."""
    src, pinned = _source_repo(tmp_path)
    dest = tmp_path / "vendor" / "SadTalker"
    dest.parent.mkdir(parents=True)
    _git(tmp_path, "clone", "-q", str(src), str(dest))
    assert (dest / "inference.py").read_text() == "default branch\n"

    models.sync_source(dest, str(src), pinned, log=lambda *_: None)

    assert _git(dest, "rev-parse", "HEAD").stdout.strip() == pinned
    assert (dest / "inference.py").read_text() == "pinned\n"


def test_sync_source_is_idempotent_and_offline_after_the_first_run(tmp_path):
    src, pinned = _source_repo(tmp_path)
    dest = tmp_path / "vendor" / "SadTalker"
    models.sync_source(dest, str(src), pinned, log=lambda *_: None)

    # The remote is gone: a second run must not need it.
    gone = tmp_path / "gone"
    src.rename(gone)
    assert models.sync_source(dest, str(gone), pinned, log=lambda *_: None) == pinned
    assert (dest / "inference.py").read_text() == "pinned\n"


def test_sync_source_reports_a_fetch_failure(tmp_path):
    with pytest.raises(RuntimeError, match="cannot fetch SadTalker"):
        models.sync_source(tmp_path / "vendor" / "SadTalker", str(tmp_path / "missing"), "0" * 40,
                           log=lambda *_: None)


def test_fetch_pins_the_commit_and_downloads_every_weight(tmp_path, monkeypatch):
    calls: dict = {}

    def fake_sync(dest, repo, commit, log):
        calls["sync"] = (dest, repo, commit)
        return commit

    def fake_download(w):
        calls.setdefault("weights", []).append(w.rel)
        return f"ok      {w.rel}"

    monkeypatch.setattr(models, "DATA", tmp_path)
    monkeypatch.setattr(models, "VENDOR", tmp_path / "vendor" / "SadTalker")
    monkeypatch.setattr(models, "sync_source", fake_sync)
    monkeypatch.setattr(models, "_download", fake_download)

    models.fetch(log=lambda *_: None)

    assert calls["sync"] == (models.VENDOR, models.SADTALKER_REPO, models.SADTALKER_COMMIT)
    assert calls["weights"] == [w.rel for w in models.WEIGHT_FILES]
