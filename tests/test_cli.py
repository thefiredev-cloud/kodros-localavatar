from pathlib import Path

import pytest

from localavatar import cli
from localavatar.animate import AnimateOptions, build_command

PHOTO = Path(__file__).resolve().parents[1] / "fixtures" / "astronaut-public-domain.png"
SCRIPT = Path(__file__).resolve().parents[1] / "fixtures" / "proof-script.txt"
VOICES = ["af_heart", "am_michael", "bf_emma"]


def parse(*argv):
    return cli.build_parser().parse_args(list(argv))


def base(tmp_path, *extra):
    return parse("make", "--photo", str(PHOTO), "--script", str(SCRIPT), "--out", str(tmp_path / "o.mp4"), *extra)


def test_defaults(tmp_path):
    a = base(tmp_path)
    assert (a.voice, a.aspect, a.size, a.captions, a.motion, a.speed) == ("af_heart", "source", 256, False, False, 1.0)
    assert cli.validate_make(a, VOICES) == []


def test_requires_photo_and_out():
    with pytest.raises(SystemExit):
        parse("make", "--script", str(SCRIPT), "--out", "o.mp4")
    with pytest.raises(SystemExit):
        parse("make", "--photo", str(PHOTO), "--script", str(SCRIPT))


def test_script_and_text_are_exclusive_and_one_required():
    with pytest.raises(SystemExit):
        parse("make", "--photo", str(PHOTO), "--script", str(SCRIPT), "--text", "hi", "--out", "o.mp4")
    with pytest.raises(SystemExit):
        parse("make", "--photo", str(PHOTO), "--out", "o.mp4")


def test_bad_choices_rejected_by_parser(tmp_path):
    with pytest.raises(SystemExit):
        base(tmp_path, "--aspect", "4:3")
    with pytest.raises(SystemExit):
        base(tmp_path, "--size", "384")


def test_validation_messages(tmp_path):
    a = parse("make", "--photo", str(tmp_path / "missing.jpg"), "--text", "  ", "--out", str(tmp_path / "o.mov"),
              "--voice", "zz_nobody", "--speed", "3", "--expression", "0.1", "--pose-style", "50")
    errs = cli.validate_make(a, VOICES)
    joined = "\n".join(errs)
    for needle in ("--photo not found", "--text is empty", "--out must end in .mp4", "--speed", "--expression",
                   "--pose-style", "unknown --voice 'zz_nobody'"):
        assert needle in joined, needle


def test_photo_must_be_image(tmp_path):
    a = parse("make", "--photo", str(SCRIPT), "--text", "hello there", "--out", str(tmp_path / "o.mp4"))
    assert any("--photo must be one of" in e for e in cli.validate_make(a, VOICES))


def test_empty_script_file(tmp_path):
    s = tmp_path / "s.txt"
    s.write_text("\n\n")
    a = parse("make", "--photo", str(PHOTO), "--script", str(s), "--out", str(tmp_path / "o.mp4"))
    assert cli.validate_make(a, VOICES) == ["--script is empty"]


def test_voice_check_skipped_without_voice_list(tmp_path):
    a = base(tmp_path, "--voice", "anything")
    assert cli.validate_make(a, None) == []


def test_audio_source_rules(tmp_path):
    wav = tmp_path / "me.wav"
    wav.write_bytes(b"RIFF")
    a = parse("make", "--photo", str(PHOTO), "--audio", str(wav), "--out", str(tmp_path / "o.mp4"))
    assert cli.validate_make(a, VOICES) == []
    a = parse("make", "--photo", str(PHOTO), "--audio", str(wav), "--out", str(tmp_path / "o.mp4"), "--captions")
    assert cli.validate_make(a, VOICES) == ["--captions needs --script or --text (no transcript for --audio)"]
    a = parse("make", "--photo", str(PHOTO), "--audio", str(tmp_path / "nope.wav"), "--out", str(tmp_path / "o.mp4"))
    assert cli.validate_make(a, VOICES) == [f"--audio not found: {tmp_path / 'nope.wav'}"]
    with pytest.raises(SystemExit):
        parse("make", "--photo", str(PHOTO), "--audio", str(wav), "--text", "hi", "--out", "o.mp4")


def test_main_returns_2_on_invalid_without_running_pipeline(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "_voices_or_none", lambda: VOICES)
    rc = cli.main(["make", "--photo", str(PHOTO), "--script", str(SCRIPT), "--out", str(tmp_path / "o.mp4"),
                   "--voice", "nope"])
    assert rc == 2
    assert "unknown --voice" in capsys.readouterr().err
    assert not (tmp_path / "o.mp4").exists()


def test_make_flags_reach_engine_command(tmp_path, monkeypatch):
    seen = {}

    def fake_make(req):
        seen["req"] = req

    monkeypatch.setattr(cli, "_voices_or_none", lambda: VOICES)
    import localavatar.pipeline as pipeline
    monkeypatch.setattr(pipeline, "make", fake_make)
    rc = cli.main(["make", "--photo", str(PHOTO), "--text", "Hello from a test.", "--out", str(tmp_path / "o.mp4"),
                   "--voice", "am_michael", "--aspect", "9:16", "--captions", "--size", "512", "--motion",
                   "--no-enhance", "--pose-style", "7"])
    assert rc == 0
    req = seen["req"]
    assert (req.voice, req.aspect, req.burn_captions, req.script_text) == ("am_michael", "9:16", True, "Hello from a test.")
    assert req.animate == AnimateOptions(size=512, still=False, enhancer=None, pose_style=7)
    cmd = build_command(PHOTO, tmp_path / "a.wav", tmp_path / "r", req.animate, python="py")
    assert cmd[:2][0] == "py" and "--still" not in cmd and "--enhancer" not in cmd
    assert cmd[cmd.index("--size") + 1] == "512" and cmd[cmd.index("--preprocess") + 1] == "full"


def test_default_engine_command_is_still_with_gfpgan(tmp_path):
    cmd = build_command(PHOTO, tmp_path / "a.wav", tmp_path / "r", AnimateOptions(), python="py")
    assert "--still" in cmd and cmd[cmd.index("--enhancer") + 1] == "gfpgan"


def test_engine_command_uses_staged_driver(tmp_path):
    from localavatar.animate import DRIVER
    from localavatar.models import VENDOR

    cmd = build_command(PHOTO, tmp_path / "a.wav", tmp_path / "r", AnimateOptions(), python="py")
    assert cmd[1] == str(DRIVER) and DRIVER.is_file()
    assert cmd[cmd.index("--vendor") + 1] == str(VENDOR)
    assert cmd[cmd.index("--batch_size") + 1] == "1"
