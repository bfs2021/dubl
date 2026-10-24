from __future__ import annotations

import numpy as np

from dubl_engine.audio.clean import clean_filter, clean_voice
from dubl_engine.audio.io import read_wav, write_wav

from .synth_voice import SR, voice_track


def _rms(x):
    return float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))


def test_clean_filter_string():
    f = clean_filter()
    assert f == "highpass=f=80,deesser=i=0.4"
    assert clean_filter(denoise=True, compress=True).startswith("highpass=f=80,afftdn=nf=-30:nr=12:tn=1,deesser=i=0.4,acompressor=threshold=0.08")
    assert clean_filter(denoise=False, deess=False, compress=False, highpass_hz=0) == "anull"


def _band_rms(x, lo, hi):
    sp = np.fft.rfft(x.astype(np.float64))
    f = np.fft.rfftfreq(len(x), 1 / SR)
    sp[(f < lo) | (f >= hi)] = 0
    return _rms(np.fft.irfft(sp, len(x)))


def test_clean_voice_removes_noise_and_rumble(tmp_path):
    rng = np.random.default_rng(0)
    voice = voice_track([(1.0, 2.5), (3.5, 4.5)], 5.0, seed=5)
    t = np.arange(len(voice)) / SR
    rumble = 0.05 * np.sin(2 * np.pi * 25 * t)
    hiss = 0.01 * rng.standard_normal(len(voice))
    noisy = (voice + rumble + hiss).astype(np.float32)
    src = write_wav(tmp_path / "in.wav", noisy, SR)
    out = clean_voice(src, tmp_path / "sub" / "out.wav", denoise=True, compress=True)   # старые режимы, включённые руками
    y, sr = read_wav(out)
    assert sr == SR and y.ndim == 1 and abs(len(y) - len(noisy)) <= 2
    pause = slice(int(0.3 * SR), int(0.9 * SR))
    # рокот 25 Гц ослаблен минимум втрое (9 дБ)
    assert _band_rms(y[pause], 10, 50) < 0.35 * _band_rms(noisy[pause], 10, 50)
    # шумодав реально работает: шипение в паузе тише, чем при denoise=False
    y2, _ = read_wav(clean_voice(src, tmp_path / "nodn.wav", denoise=False, compress=True))
    late = slice(int(2.7 * SR), int(3.4 * SR))
    assert _band_rms(y[late], 2000, 8000) < 0.8 * _band_rms(y2[late], 2000, 8000)
    # речь осталась слышной
    speech = slice(int(1.2 * SR), int(2.3 * SR))
    assert _band_rms(y[speech], 100, 4000) > 0.3 * _band_rms(noisy[speech], 100, 4000)
