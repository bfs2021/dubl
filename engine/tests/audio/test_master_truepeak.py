"""Мастеринг: true peak держит цель, в том числе после AAC; проверка после кодирования в render."""
from __future__ import annotations

import subprocess

import numpy as np
import pytest

from dubl_engine.audio.io import read_wav, write_wav
from dubl_engine.audio.master import limit_true_peak, master, measure, true_peak_db
from dubl_engine.audio.render import aac_true_peak, master_for_aac

from .synth_voice import SR
from .test_master import _music_like


def _peaky(level: float, clicks: int, seconds=20.0, seed=3) -> np.ndarray:
    """Тихий материал с редкими короткими выбросами — тот случай, где loudnorm не удерживал пик."""
    rng = np.random.default_rng(seed)
    x = _music_like(level, seconds=seconds)
    n = int(0.002 * SR)
    for _ in range(clicks):
        i = int(rng.uniform(0.5, seconds - 1) * SR)
        x[i:i + n] += (0.9 * np.hanning(n) * np.sin(2 * np.pi * 3000 * np.arange(n) / SR))[:, None] * np.array([1.0, 0.8])
    return x.astype(np.float32)


def _through_aac(wav, tmp_path):
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", wav, "-c:a", "aac", "-b:a", "192k", tmp_path / "a.m4a"], check=True)
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", tmp_path / "a.m4a", "-c:a", "pcm_f32le", tmp_path / "back.wav"],
                   check=True)
    y, sr = read_wav(tmp_path / "back.wav")
    return measure(y, sr)


@pytest.mark.parametrize("name,x", [("музыка", None), ("пиковый", "peaky"), ("слишком громкий", "hot")])
def test_true_peak_target_holds_after_aac(tmp_path, name, x):
    """Цель −1.0 dBTP, −14 LUFS: в WAV пик не выше цели, после AAC — не выше −0.8, громкость ±0.5.
    Как в рендере: master_for_aac (master + проверка после кодирования; на «пиковом» AAC добавляет до 0.7 дБ — пересвод)."""
    sig = {None: lambda: _music_like(0.2), "peaky": lambda: _peaky(0.05, 12),
           "hot": lambda: np.clip(_music_like(0.9) * 3, -1.2, 1.2).astype(np.float32)}[x]()
    src = write_wav(tmp_path / "in.wav", sig, SR)
    res = master_for_aac(src, tmp_path / "out.wav", target_lufs=-14.0, true_peak_db=-1.0)
    y, sr = read_wav(tmp_path / "out.wav")
    chk = measure(y, sr)
    assert chk.true_peak_db <= -1.0 + 0.02, (name, chk)
    assert abs(chk.lufs + 14.0) <= 0.5, (name, chk)
    assert res.true_peak_db == pytest.approx(chk.true_peak_db, abs=0.05)
    aac = _through_aac(tmp_path / "out.wav", tmp_path)
    assert aac.true_peak_db <= -0.8, (name, aac)
    assert abs(aac.lufs + 14.0) <= 0.5, (name, aac)


def test_plain_master_holds_wav_peak(tmp_path):
    src = write_wav(tmp_path / "in.wav", _peaky(0.05, 12), SR)
    master(src, tmp_path / "o.wav", target_lufs=-14.0, true_peak_db=-1.0)
    y, sr = read_wav(tmp_path / "o.wav")
    assert measure(y, sr).true_peak_db <= -1.0 + 0.02


def test_master_for_aac_rechecks_after_encoding(tmp_path, monkeypatch):
    """Если после AAC пик выше цели + 0.2, пересводим с меньшей целью (проверяем на подмене измерения)."""
    src = write_wav(tmp_path / "in.wav", _peaky(0.05, 12), SR)
    calls = []
    real = master

    def spy(i, o, **kw):
        calls.append(kw["true_peak_db"])
        return real(i, o, **kw)

    monkeypatch.setattr("dubl_engine.audio.master.master", spy)
    readings = iter([-0.2, -1.0])        # первое «кодирование» дало −0.2 dBTP, второе — норма
    monkeypatch.setattr("dubl_engine.audio.render.aac_true_peak", lambda *_a, **_k: next(readings))
    master_for_aac(src, tmp_path / "o.wav", target_lufs=-14.0, true_peak_db=-1.0)
    assert len(calls) == 2 and calls[1] < calls[0] - 0.7, calls
    assert true_peak_db(read_wav(tmp_path / "o.wav")[0]) <= calls[1] + 0.02


def test_master_for_aac_real_encoding(tmp_path):
    src = write_wav(tmp_path / "in.wav", _peaky(0.05, 12), SR)
    master_for_aac(src, tmp_path / "o.wav", target_lufs=-14.0, true_peak_db=-1.0)
    assert aac_true_peak(tmp_path / "o.wav") <= -0.8


def test_limiter_is_transparent_below_ceiling_and_long_input_ok():
    x = (0.1 * np.sin(2 * np.pi * 440 * np.arange(SR * 2) / SR)).astype(np.float32)
    assert limit_true_peak(x, SR, -1.0) is x                    # тише потолка — не трогаем
    y = np.concatenate([x, np.full(10, 0.99, dtype=np.float32), x])
    z = limit_true_peak(np.stack([y, y], axis=1), SR, -3.0)
    assert true_peak_db(z) <= -3.0 + 0.05 and z.shape == (len(y), 2)
