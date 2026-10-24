from __future__ import annotations

import numpy as np
import pytest

from dubl_engine.audio.fx import VOICE_FX, VOICE_FX_TITLES, apply_fx
from dubl_engine.audio.io import read_wav, write_wav
from dubl_engine.util import EngineError

from .synth_voice import SR, voice_track


def test_presets_listed():
    assert set(VOICE_FX) == {"none", "robot", "chipmunk", "giant", "villain", "radio", "phone", "ghost"}
    assert set(VOICE_FX_TITLES) == set(VOICE_FX)


@pytest.mark.parametrize("preset", sorted(VOICE_FX))
def test_fx_runs_and_keeps_length(tmp_path, preset):
    x = voice_track([(0.2, 1.4), (1.8, 2.6)], 3.0, seed=7)
    src = write_wav(tmp_path / "in.wav", x, SR)
    out = apply_fx(src, preset, tmp_path / "o" / f"{preset}.wav")
    y, sr = read_wav(out)
    assert sr == SR
    assert abs(len(y) - len(x)) / len(x) <= 0.01
    assert np.isfinite(y).all() and np.abs(y).max() > 0.01
    if preset == "none":
        assert np.allclose(x, y)


def _peak_hz(x):
    s = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    return float(np.fft.rfftfreq(len(x), 1 / SR)[int(np.argmax(s))])


def test_pitch_direction(tmp_path):
    t = np.arange(SR * 2) / SR
    x = (0.3 * np.sin(2 * np.pi * 400 * t)).astype(np.float32)
    src = write_wav(tmp_path / "in.wav", x, SR)
    hi, _ = read_wav(apply_fx(src, "chipmunk", tmp_path / "c.wav"))
    lo, _ = read_wav(apply_fx(src, "giant", tmp_path / "g.wav"))
    mid = slice(SR // 2, int(1.5 * SR))
    assert _peak_hz(hi[mid]) == pytest.approx(560, abs=15)
    assert _peak_hz(lo[mid]) == pytest.approx(300, abs=15)


def test_unknown_preset(tmp_path):
    src = write_wav(tmp_path / "in.wav", np.zeros(SR, dtype=np.float32), SR)
    with pytest.raises(EngineError):
        apply_fx(src, "nope", tmp_path / "o.wav")
