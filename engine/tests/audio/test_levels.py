from __future__ import annotations

import numpy as np

from dubl_engine.audio.levels import (CLIP_WARN_PCT, clip_report, clip_warning, declip, measure_take, noise_floor_db,
                                      quiet_segment, snr_db)

from .synth_voice import SR, voice_track


def _noisy(noise=0.002, seconds=6.0, seed=1, level=1.0, speech=((1.5, 3.0), (3.8, 5.2))):
    rng = np.random.default_rng(seed)
    v = voice_track(list(speech), seconds, seed=4, level=level)
    return (v + noise * rng.standard_normal(len(v))).astype(np.float32)


def test_quiet_segment_is_preroll_and_noise_floor_matches():
    x = _noisy(noise=0.002)
    seg = quiet_segment(x, SR)
    assert seg is not None and seg[1] <= 1.55            # преролл до первого слова
    nf = noise_floor_db(x, SR)
    assert abs(nf - 20 * np.log10(0.002)) < 1.5, nf       # −54 дБ


def test_quiet_segment_none_when_no_silence_or_empty():
    assert quiet_segment(np.zeros(SR * 3, np.float32), SR) is None          # пустая запись
    steady = voice_track([(0.0, 3.0)], 3.0, seed=2)                         # речь без пауз
    assert quiet_segment(steady, SR) is None


def test_snr_reflects_noise_level():
    quiet, loud = snr_db(_noisy(0.0005), SR), snr_db(_noisy(0.01), SR)
    assert quiet > loud + 15
    assert 25 < quiet < 60


def test_clip_report_clean_and_clipped():
    t = np.arange(SR) / SR
    clean = (0.8 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)
    assert clip_report(clean).pct < 0.01
    hot = np.clip(1.6 * np.sin(2 * np.pi * 200 * t), -0.98, 0.98).astype(np.float32)
    rep = clip_report(hot)
    assert rep.pct > 5 and len(rep.runs) >= 100


def test_declip_restores_peaks_and_leaves_headroom():
    t = np.arange(SR * 2) / SR
    truth = (0.5 * (np.sin(2 * np.pi * 180 * t) + 0.5 * np.sin(2 * np.pi * 540 * t + 1))).astype(np.float32)
    truth *= 1.6 / np.abs(truth).max() * 0.75          # пик 1.2
    hot = np.clip(truth, -0.95, 0.95)
    y, rep = declip(hot)
    assert rep.pct > 1
    assert np.abs(y).max() <= 10 ** (-9 / 20) + 1e-3          # запас −9 дБ
    # форма волны ближе к настоящей (после подгонки по громкости), чем срезанная
    def err(a):
        k = float(np.dot(a, truth) / np.dot(a, a))
        return float(np.sqrt(np.mean((k * a - truth) ** 2)))
    assert err(y) < 0.6 * err(hot)
    assert clip_report(y).pct < 1.0                      # плоских полок почти не осталось


def test_declip_does_not_touch_clean_audio():
    x = _noisy()
    y, rep = declip(x)
    assert y is x and rep.pct == 0


def test_measure_take_and_warning():
    m = measure_take(_noisy(0.002), SR)
    assert m.clip_pct == 0 and m.noise_floor_db is not None and m.snr_db is not None and m.peak_db < 0
    assert clip_warning(m.clip_pct) is None
    hot = np.clip(_noisy(0.002) * 6, -0.99, 0.99)
    mh = measure_take(hot, SR)
    assert mh.clip_pct >= CLIP_WARN_PCT and "перегруз" in clip_warning(mh.clip_pct)
    assert measure_take(np.zeros(SR, np.float32), SR).noise_floor_db is None
