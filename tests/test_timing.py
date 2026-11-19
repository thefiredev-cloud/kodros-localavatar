import shutil
import subprocess
import wave

import numpy as np
import pytest

from localavatar import captions, render, timing
from localavatar.tts import RATE, Segment, write_wav

needs_ffmpeg = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg missing")


def test_frames_for_rounds_up():
    assert timing.frames_for(2.0, 25) == 50
    assert timing.frames_for(2.01, 25) == 51
    assert timing.frames_for(0.0, 25) == 0
    # float noise must not add a frame: 1.2 * 25 is 30.000000000000004
    assert timing.frames_for(1.2, 25) == 30


def test_frames_for_rejects_bad_input():
    with pytest.raises(ValueError):
        timing.frames_for(-1, 25)
    with pytest.raises(ValueError):
        timing.frames_for(1, 0)


def test_fit_plan_pads_short_video():
    p = timing.fit_plan(video_sec=10.0, audio_sec=10.37)
    assert (p.pad, p.trim, p.target) == (0.37, 0.0, 10.37)


def test_fit_plan_trims_long_video():
    p = timing.fit_plan(video_sec=10.4, audio_sec=10.0)
    assert (p.pad, p.trim, p.target) == (0.0, 0.4, 10.0)


def test_durations_match_tolerance():
    assert timing.durations_match(10.0, 10.04, 25)
    assert timing.durations_match(10.0, 10.08, 25)
    assert not timing.durations_match(10.0, 10.2, 25)


def test_words_cover_segment_exactly():
    segs = [Segment(0.25, 2.25, "Hello there, quiet world."), Segment(2.6, 3.6, "Bye.")]
    words = captions.words_from_segments(segs)
    assert [w.text for w in words] == ["Hello", "there,", "quiet", "world.", "Bye."]
    assert words[0].start == 0.25 and abs(words[3].end - 2.25) < 0.002
    assert all(a.end <= b.start + 1e-3 for a, b in zip(words, words[1:]))
    events = captions.caption_events(words, total=4.0)
    assert events and all(e.end <= 4.0 for e in events)
    assert all(a.end <= b.start + 1e-3 for a, b in zip(events, events[1:]))


def _make_inputs(tmp_path, video_sec: float, audio_sec: float):
    raw = tmp_path / "raw.mp4"
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi",
                    "-i", f"testsrc2=s=512x512:r=25:d={video_sec}", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    str(raw)], check=True)
    t = np.arange(int(RATE * audio_sec)) / RATE
    wav = tmp_path / "voice.wav"
    write_wav(wav, 0.3 * np.sin(2 * np.pi * 220 * t).astype(np.float32))
    with wave.open(str(wav)) as w:
        assert abs(w.getnframes() / w.getframerate() - audio_sec) < 1e-3
    return raw, wav


@needs_ffmpeg
@pytest.mark.parametrize("video_sec,audio_sec,aspect", [(2.0, 2.37, "9:16"), (3.0, 2.2, "source"), (2.0, 2.0, "1:1")])
def test_render_matches_audio_duration(tmp_path, video_sec, audio_sec, aspect):
    raw, wav = _make_inputs(tmp_path, video_sec, audio_sec)
    info = timing.probe_streams(str(raw))
    plan = timing.fit_plan(info["video"]["sec"], audio_sec)
    out_w, out_h = render.output_size(aspect, 512, 512)
    ass = tmp_path / "c.ass"
    words = captions.words_from_segments([Segment(0.1, audio_sec - 0.1, "Testing one two three.")])
    ass.write_text(captions.build_ass(captions.caption_events(words, plan.target), out_w, out_h))
    out = tmp_path / "out.mp4"
    enc = render.render(str(raw), str(wav), str(out), aspect, 512, 512, plan, str(ass), encoders=("libx264",))
    assert enc == "libx264"
    final = timing.probe_streams(str(out))
    assert (final["video"]["width"], final["video"]["height"]) == (out_w, out_h)
    assert timing.durations_match(final["video"]["sec"], final["audio"]["sec"], render.OUT_FPS)
    assert abs(final["video"]["sec"] - audio_sec) <= 1 / render.OUT_FPS + 1e-6
    assert timing.audio_seconds(str(wav)) == pytest.approx(audio_sec, abs=1e-3)


def test_output_size_rules():
    assert render.output_size("source", 513, 701) == (512, 700)
    assert render.output_size("9:16", 512, 512) == (1080, 1920)
    with pytest.raises(ValueError):
        render.output_size("4:3", 512, 512)


@needs_ffmpeg
@pytest.mark.parametrize("filename", ["captions.ass", "it's.ass", "colon:comma,[bracket];back\\slash.ass", " leading.ass "])
def test_render_decodes_captions_from_special_paths(tmp_path, filename):
    raw = tmp_path / "raw.mp4"
    subprocess.run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi",
        "-i", "color=c=black:s=320x240:r=25:d=1", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(raw),
    ], check=True)
    wav = tmp_path / "tone.wav"
    t = np.arange(RATE) / RATE
    write_wav(wav, (0.2 * np.sin(2 * np.pi * 440 * t)).astype(np.float32))
    style = captions.CaptionStyle(size_ratio=0.2, outline_px=2, shadow_px=0)
    ass = tmp_path / filename
    words = [captions.Word(0, 0.9, "TEST")]
    ass.write_text(captions.build_ass(captions.caption_events(words, 1, style), 320, 240, style))
    out = tmp_path / "captioned.mp4"
    enc = render.render(str(raw), str(wav), str(out), "source", 320, 240, timing.fit_plan(1, 1),
                        str(ass), encoders=("libx264",))
    assert enc == "libx264"
    final = timing.probe_streams(str(out))
    assert (final["video"]["width"], final["video"]["height"]) == (320, 240)
    assert timing.durations_match(final["video"]["sec"], final["audio"]["sec"], render.OUT_FPS)
    assert final["video"]["sec"] == pytest.approx(1, abs=1 / render.OUT_FPS)
    frame = subprocess.run([
        "ffmpeg", "-v", "error", "-ss", "0.5", "-i", str(out), "-frames:v", "1",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1",
    ], capture_output=True, check=True).stdout
    pixels = np.frombuffer(frame, np.uint8).reshape(240, 320, 3)
    assert np.count_nonzero(pixels.max(axis=2) > 100) > 100  # visible captions on black
    pcm = subprocess.run([
        "ffmpeg", "-v", "error", "-i", str(out), "-vn", "-f", "s16le", "pipe:1",
    ], capture_output=True, check=True).stdout
    assert np.abs(np.frombuffer(pcm, dtype=np.int16).astype(np.int32)).max() > 100
