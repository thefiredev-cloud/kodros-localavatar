import numpy as np

from localavatar import tts
from localavatar.animate import parse_stats
from localavatar.tts import Segment


def test_cache_roundtrip(tmp_path):
    wav = tmp_path / "n.wav"
    tts.write_wav(wav, np.zeros(tts.RATE, dtype=np.float32))
    segs = [Segment(0.25, 1.0, "Hello there.")]
    cache = tmp_path / "cache"
    assert tts.load_cached("Hello there.", "af_heart", 1.0, cache) is None
    tts.store_cached("Hello there.", "af_heart", 1.0, wav, segs, cache)
    hit = tts.load_cached("Hello there.", "af_heart", 1.0, cache)
    assert hit is not None and hit[1] == segs and hit[0].read_bytes() == wav.read_bytes()
    # any change in text, voice or speed misses
    assert tts.load_cached("Hello there!", "af_heart", 1.0, cache) is None
    assert tts.load_cached("Hello there.", "am_michael", 1.0, cache) is None
    assert tts.load_cached("Hello there.", "af_heart", 1.1, cache) is None


def test_trim_silence_keeps_speech():
    x = np.zeros(tts.RATE, dtype=np.float32)
    x[10000:12000] = 0.5
    y = tts.trim_silence(x)
    assert len(y) < len(x) and np.count_nonzero(y) == 2000


def test_parse_engine_stats():
    log = "noise\nENGINE_STATS {\"device\": \"cuda\", \"peak_alloc_mib\": {\"render\": 1800}}\n"
    assert parse_stats(log)["peak_alloc_mib"]["render"] == 1800
    assert parse_stats("nothing here") == {}
