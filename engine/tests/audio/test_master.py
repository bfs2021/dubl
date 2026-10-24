from __future__ import annotations

import numpy as np
import pytest

from dubl_engine.audio.io import read_wav, write_wav
from dubl_engine.audio.master import master, measure

from .synth_voice import SR, voice_track


def _music_like(level: float, seconds=12.0, seed=0):
    sig = voice_track([(0.5, 5.0), (6.0, 11.5)], seconds, seed=seed, level=level)
    t = np.arange(len(sig)) / SR
    bed = 0.05 * level * np.sin(2 * np.pi * 220 * t)
    x = (sig + bed).astype(np.float32)
    return np.stack([x, 0.9 * x], axis=1)


def test_measure_sine_known_value():
    t = np.arange(SR * 5) / SR
    x = (0.1 * np.sin(2 * np.pi * 1000 * t)).astype(np.float32)
    m = measure(x, SR)
    # синус 1 кГц, амплитуда 0.1 (−20 dBFS): mono ≈ −23.0 LUFS
    assert m.lufs == pytest.approx(-23.0, abs=0.3)
    assert m.true_peak_db == pytest.approx(-20.0, abs=0.3)
    assert m.lra < 1.0


def test_measure_short_clip_does_not_crash():
    assert measure(np.zeros(1000, dtype=np.float32), SR).lufs <= -60


@pytest.mark.parametrize("level,target,tp", [(0.2, -14.0, -1.0), (0.9, -16.0, -1.5), (0.03, -14.0, -1.0)])
def test_master_hits_target(tmp_path, level, target, tp):
    src = write_wav(tmp_path / "in.wav", _music_like(level), SR)
    res = master(src, tmp_path / "o" / "out.wav", target_lufs=target, true_peak_db=tp)
    y, sr = read_wav(tmp_path / "o" / "out.wav")
    chk = measure(y, sr)
    assert abs(chk.lufs - target) <= 0.5, (chk, res)
    assert chk.true_peak_db <= tp + 0.2, (chk, res)
    assert abs(res.lufs - chk.lufs) < 0.1
    assert sr == SR and y.shape[1] == 2 and abs(len(y) - SR * 12) < SR * 0.05


def test_master_constant_gain_does_not_pump(tmp_path):
    """Постоянный гейн: тихий фон не поднимается сильнее речи (раньше loudnorm в динамическом режиме качал фон на +17 дБ)."""
    rng = np.random.default_rng(0)
    bed = (0.003 * rng.standard_normal(SR * 10)).astype(np.float32)            # фон ≈ −50 дБ
    speech = voice_track([(4.0, 6.0)], 10.0, seed=3, level=0.5)
    x = np.stack([bed + speech, bed + speech], axis=1)
    src = write_wav(tmp_path / "in.wav", x, SR)
    master(src, tmp_path / "o.wav", target_lufs=-14.0, true_peak_db=-1.0)
    y, _ = read_wav(tmp_path / "o.wav")
    g_bed = 20 * np.log10(np.sqrt(np.mean(y[:SR * 3] ** 2)) / np.sqrt(np.mean(x[:SR * 3] ** 2)))
    g_voice = 20 * np.log10(np.sqrt(np.mean(y[int(4.2 * SR):int(5.8 * SR)] ** 2)) / np.sqrt(np.mean(x[int(4.2 * SR):int(5.8 * SR)] ** 2)))
    assert abs(g_bed - g_voice) < 1.5, (g_bed, g_voice)


def test_resolve_target_lufs(tmp_path):
    from dubl_engine.audio.master import film_target_lufs, resolve_target_lufs
    x = voice_track([(0.5, 5.0)], 6.0, seed=1, level=0.3)
    p = write_wav(tmp_path / "o.wav", np.stack([x, x], axis=1), SR)
    film = film_target_lufs(p)
    assert film is not None and -30 < film < -10
    assert resolve_target_lufs("film", -14.0, p) == film
    assert resolve_target_lufs("online", -14.0, p) == -14.0
    assert resolve_target_lufs("film", -14.0, tmp_path / "нет.wav") == -14.0
    assert resolve_target_lufs("film", -14.0, None) == -14.0
