from __future__ import annotations

import shutil
import subprocess
import sys

import numpy as np
import pytest

from dubl_engine.audio.io import read_wav, write_wav
from dubl_engine.audio.separate import separate, split_by_speaker, subtract_voices
from dubl_engine.models import SceneLine

SR = 48000


def _sl(a: float, b: float, spk: str | None) -> SceneLine:
    return SceneLine(line_id="l", speaker_id=spk, start_rel=a, end_rel=b, original="", text="")


def _voices(tmp_path, secs=6.0):
    x = (0.5 * np.sin(2 * np.pi * 300 * np.arange(int(secs * SR)) / SR)).astype("float32")
    return write_wav(tmp_path / "voices.wav", x, SR), x


def test_split_by_speaker_silence_outside_and_fades(tmp_path):
    p, x = _voices(tmp_path)
    out = split_by_speaker(p, [_sl(1, 2, "s1"), _sl(3, 4, "s2"), _sl(4.5, 5, "s1"), _sl(0, 0.2, None)],
                           tmp_path / "per")
    assert set(out) == {"s1", "s2", "unknown"}
    y1, sr = read_wav(out["s1"])
    assert sr == SR and len(y1) == len(x)
    t = np.arange(len(y1)) / SR
    # внутри реплики (без краёв) — как оригинал
    mid = (t > 1.05) & (t < 1.95)
    assert np.allclose(y1[mid], x[mid])
    # вне реплик (с запасом 0.1 с и фейдом) — тишина
    assert np.all(y1[(t > 2.2) & (t < 4.3)] == 0)
    assert np.all(y1[t < 0.8] == 0)
    # реплика 4.5–5 тоже есть, а чужая 3–4 нет
    assert np.abs(y1[(t > 4.6) & (t < 4.9)]).max() > 0.4
    assert np.all(y1[(t > 3.0) & (t < 4.0)] == 0)
    # запас по краям 0.1 с: звук есть чуть раньше начала реплики; фейд — плавный
    edge = (t >= 0.9 + 0.001) & (t < 0.9 + 0.03)
    assert np.abs(y1[edge]).max() < np.abs(x[edge]).max() + 1e-6
    assert np.abs(y1[(t > 0.95) & (t < 1.0)]).max() > 0.4
    # unknown — только его кусок
    yu, _ = read_wav(out["unknown"])
    assert np.abs(yu[: int(0.25 * SR)]).max() > 0.4 and np.all(yu[int(0.5 * SR):] == 0)


def test_split_by_speaker_clamps_to_edges_and_empty(tmp_path):
    p, x = _voices(tmp_path, 2.0)
    out = split_by_speaker(p, [_sl(-1, 0.5, "a"), _sl(1.8, 5, "a")], tmp_path / "per")
    y, _ = read_wav(out["a"])
    assert len(y) == len(x)
    assert split_by_speaker(p, [], tmp_path / "per2") == {}


def test_split_by_speaker_sums_back(tmp_path):
    p, x = _voices(tmp_path)
    out = split_by_speaker(p, [_sl(1, 2, "s1"), _sl(3, 4, "s2")], tmp_path / "per", fade=0.03)
    total = sum(read_wav(f)[0] for f in out.values())
    t = np.arange(len(x)) / SR
    inside = ((t > 1.1) & (t < 1.9)) | ((t > 3.1) & (t < 3.9))
    assert np.allclose(total[inside], x[inside])


def test_subtract_voices_removes_voice_keeps_music():
    n = SR * 3
    t = np.arange(n) / SR
    voice = (0.3 * np.sin(2 * np.pi * 500 * t) * (t > 1)).astype("float32")
    music = np.stack([0.1 * np.sin(2 * np.pi * 120 * t), 0.08 * np.sin(2 * np.pi * 180 * t)], 1).astype("float32")
    mix = music + 0.7 * voice[:, None]   # голос в миксе тише, чем в «нейросетевом» — уровень должен подобраться
    bg = subtract_voices(mix, voice)
    assert bg.shape == mix.shape and bg.dtype == np.float32
    sp = (t > 1.2) & (t < 2.8)
    resid = bg[sp] - music[sp]
    assert np.sqrt((resid ** 2).mean()) < 0.1 * np.sqrt((music[sp] ** 2).mean())
    # без «антиголоса»: остаток не громче исходного микса
    assert np.abs(bg).max() <= np.abs(mix).max() + 1e-6


def test_subtract_voices_silent_voice_returns_mix_and_no_negative_boost():
    mix = (0.2 * np.random.default_rng(0).standard_normal((SR, 2))).astype("float32")
    bg = subtract_voices(mix, np.zeros(SR, dtype="float32"))
    assert np.allclose(bg, mix, atol=1e-6)
    # голос, не имеющий отношения к миксу: усиление ≥ 0 и остаток не выше исходника
    v = (0.3 * np.random.default_rng(1).standard_normal(SR)).astype("float32")
    bg2 = subtract_voices(mix, v)
    assert float((bg2 ** 2).mean()) <= float((mix ** 2).mean()) * 1.05


@pytest.fixture
def shared_models(monkeypatch):
    import os

    from platformdirs import user_data_dir
    monkeypatch.setenv("DUBL_DATA_DIR", os.environ.get("DUBL_TEST_DATA_DIR") or user_data_dir("Dubl", "Dubl"))


@pytest.mark.slow
@pytest.mark.skipif(sys.platform != "darwin" or not shutil.which("say"), reason="нужна macOS `say`")
def test_separate_fast_speech_over_tones(tmp_path, shared_models):
    aiff, wav = tmp_path / "s.aiff", tmp_path / "s.wav"
    subprocess.run(["say", "-v", "Milena", "Привет, это проверка разделения голоса и фона. Сегодня хорошая погода.",
                    "-o", str(aiff)], check=True)
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(aiff), "-ar", "48000", "-ac", "1",
                    str(wav)], check=True)
    sp, _ = read_wav(wav)
    n = SR * 12
    t = np.arange(n) / SR
    music = sum(0.08 * np.sin(2 * np.pi * f * t) * (0.6 + 0.4 * np.sin(2 * np.pi * 0.5 * t + f))
                for f in (220, 277, 330, 440))
    v = np.zeros(n, dtype="float32")
    k = min(len(sp), n - SR)
    v[SR:SR + k] = sp[:k] * 0.8
    mix = np.stack([music, music * 0.9], 1).astype("float32") + v[:, None]
    src = write_wav(tmp_path / "mix.wav", mix, SR)
    progress: list[float] = []
    vp, bp = separate(src, tmp_path / "out", quality="fast", on_progress=progress.append)
    vx, sr1 = read_wav(vp)
    bx, sr2 = read_wav(bp)
    assert sr1 == sr2 == SR and vx.dtype == np.float32
    assert vx.shape == (n,) and bx.shape == (n, 2)
    speech = (t > 1.3) & (t < 1 + k / SR - 0.3)
    assert (vx[speech] ** 2).mean() > 2 * (bx[speech] ** 2).mean()   # в голосах речи больше, чем в фоне
    assert (bx[~speech & (t > 0.1)] ** 2).mean() > 0                  # фон остался
    assert progress and progress[-1] == 1.0
    # с центральным каналом: фон = микс − голоса
    cen = write_wav(tmp_path / "center.wav", v + 0.01 * np.random.default_rng(0).standard_normal(n).astype("float32"), SR)
    _vp2, bp2 = separate(src, tmp_path / "out2", center_wav=cen, quality="fast")
    bx2, _ = read_wav(bp2)
    assert bx2.shape == (n, 2)
    assert (bx2[speech] ** 2).mean() < (mix[speech] ** 2).mean() * 0.5
