from __future__ import annotations

import numpy as np

from dubl_engine.audio.dynamics import compress, voice_compress

from .synth_voice import SR, voice_track


def _rms_db(x):
    return 20 * np.log10(np.sqrt(np.mean(np.asarray(x, dtype=np.float64) ** 2)) + 1e-12)


def test_loud_parts_reduced_quiet_parts_untouched():
    x = voice_track([(0.2, 1.5)], 2.0, seed=1, level=0.1) + voice_track([(0.2, 1.5)], 2.0, seed=2, level=0.0) * 0
    loud = voice_track([(0.2, 1.5)], 2.0, seed=3, level=1.0)
    sig = np.concatenate([x, loud]).astype(np.float32)
    y, gr = compress(sig, SR, threshold_db=-22.0)
    assert gr > 3
    # тихая половина почти не тронута (колено мягкое), громкая заметно тише
    assert abs(_rms_db(y[:SR * 2]) - _rms_db(sig[:SR * 2])) < 1.5
    assert _rms_db(y[SR * 2:]) < _rms_db(sig[SR * 2:]) - 2.0
    assert len(y) == len(sig)


def test_voice_compress_threshold_follows_speech_level():
    base = voice_track([(0.3, 2.0), (2.6, 4.0)], 5.0, seed=5, level=1.0)
    a, _ = voice_compress(base, SR)
    b, _ = voice_compress(base * 0.05, SR)          # та же запись тише на 26 дБ
    assert abs((_rms_db(a) - _rms_db(base)) - (_rms_db(b) - _rms_db(base * 0.05))) < 1.0   # сжатие то же самое


def test_silence_and_empty_are_safe():
    z = np.zeros(SR, np.float32)
    y, gr = voice_compress(z, SR)
    assert gr == 0 and np.array_equal(y, z)
    assert compress(np.zeros(0, np.float32), SR, threshold_db=-20)[0].size == 0
