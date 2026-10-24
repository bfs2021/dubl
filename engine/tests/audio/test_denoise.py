from __future__ import annotations

import numpy as np

from dubl_engine.audio.denoise import (DEPTH_MAX_DB, DEPTH_MIN_DB, auto_depth_db, denoise_voice, noise_profile,
                                       profile_from_list, profile_to_list, spectral_gate)
from dubl_engine.audio.levels import noise_floor_db

from .synth_voice import SR, voice_track


def _rms_db(x):
    return 20 * np.log10(np.sqrt(np.mean(np.asarray(x, dtype=np.float64) ** 2)) + 1e-12)


def _take(noise=0.006, hum=0.004, seconds=7.0, seed=0):
    rng = np.random.default_rng(seed)
    v = voice_track([(1.5, 3.2), (4.0, 6.0)], seconds, seed=3, level=0.3)
    t = np.arange(len(v)) / SR
    n = noise * rng.standard_normal(len(v)) + hum * np.sin(2 * np.pi * 100 * t) + 0.5 * hum * np.sin(2 * np.pi * 557 * t)
    return v, (v + n).astype(np.float32)


def test_gate_lowers_noise_in_pauses_and_keeps_speech():
    v, x = _take()
    y, info = denoise_voice(x, SR)
    assert info.applied and DEPTH_MIN_DB <= info.depth_db <= DEPTH_MAX_DB and info.profile_from == "take"
    pause = slice(int(0.2 * SR), int(1.2 * SR))
    late = slice(int(3.4 * SR), int(3.9 * SR))
    assert len(y) == len(x)
    for sl in (pause, late):
        assert _rms_db(y[sl]) < _rms_db(x[sl]) - 5.0           # шум в паузах заметно тише
    sp = slice(int(1.6 * SR), int(3.1 * SR))
    assert abs(_rms_db(y[sp]) - _rms_db(x[sp])) < 3.0          # речь на месте (синтетика с шумом в самих слогах: часть уходит)
    # речь не «съедена»: корреляция с исходной речью почти не упала
    c0 = np.corrcoef(x[sp], v[sp])[0, 1]
    c1 = np.corrcoef(y[sp], v[sp])[0, 1]
    assert c1 > c0 - 0.03


def test_gate_kills_hum_tones_in_pauses():
    _v, x = _take()
    y, _ = denoise_voice(x, SR)
    pause = slice(int(0.2 * SR), int(1.2 * SR))

    def tone(sig, hz):
        sp = np.abs(np.fft.rfft(sig[pause] * np.hanning(len(sig[pause]))))
        f = np.fft.rfftfreq(len(sig[pause]), 1 / SR)
        return 20 * np.log10(sp[(f > hz - 3) & (f < hz + 3)].max() + 1e-12)
    assert tone(y, 100) < tone(x, 100) - 9


def test_no_pumping_noise_floor_is_steady():
    """Шум в паузах не «дышит»: разброс громкости 20-мс кадров в паузе не больше, чем у исходного шума + 3 дБ."""
    _v, x = _take()
    y, _ = denoise_voice(x, SR)

    def wobble(sig):
        seg = sig[int(0.2 * SR):int(1.3 * SR)]
        n = int(0.02 * SR)
        fr = seg[: len(seg) // n * n].reshape(-1, n)
        return float(np.std(10 * np.log10(np.mean(fr ** 2, axis=1) + 1e-14)))
    assert wobble(y) < wobble(x) + 3.0


def test_clean_take_is_left_alone_and_profile_roundtrip():
    rng = np.random.default_rng(1)
    v = voice_track([(1.5, 3.0)], 5.0, seed=3, level=0.5)
    x = (v + 3e-5 * rng.standard_normal(len(v))).astype(np.float32)           # шум −90 дБ
    y, info = denoise_voice(x, SR)
    assert not info.applied and y is x
    y2, info2 = denoise_voice(x, SR, force=True)
    assert info2.applied
    prof = noise_profile(x, SR)
    back = profile_from_list(profile_to_list(prof))
    assert back is not None and len(back) == len(prof) and profile_from_list([1.0, 2.0]) is None


def test_uses_given_profile_when_take_has_no_silence():
    _v, x = _take()
    prof = noise_profile(x, SR)
    steady = voice_track([(0.0, 4.0)], 4.0, seed=2, level=0.3) + 0.006 * np.random.default_rng(2).standard_normal(4 * SR).astype(np.float32)
    y, info = denoise_voice(steady.astype(np.float32), SR)
    assert not info.applied and "тишины" in info.reason
    y2, info2 = denoise_voice(steady.astype(np.float32), SR, profile=prof, force=True)
    assert info2.applied and info2.profile_from == "given"


def test_depth_grows_with_noise():
    assert auto_depth_db(20) == DEPTH_MAX_DB and auto_depth_db(60) == DEPTH_MIN_DB
    assert auto_depth_db(30) > auto_depth_db(40)


def test_short_input_is_returned_as_is():
    x = np.zeros(1000, np.float32)
    assert spectral_gate(x, SR, np.ones(1025)) is x
    assert noise_floor_db(x, SR) is None
