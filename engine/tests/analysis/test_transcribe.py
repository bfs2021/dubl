from __future__ import annotations

import shutil
import subprocess
import sys

import pytest

import numpy as np

from dubl_engine.analysis import transcribe as T
from dubl_engine.analysis.transcribe import split_words, transcribe
from dubl_engine.devices import pick_device, whisper_runtime
from dubl_engine.models import Word


def _w(text: str, s: float, e: float) -> Word:
    return Word(start=s, end=e, text=text)


def test_short_segment_not_split():
    ws = [_w(" a", 0, 0.5), _w(" b", 2.0, 2.5), _w(" c", 4.0, 4.5)]
    assert split_words(ws) == [ws]


def test_long_segment_split_on_pauses():
    # 12 с, паузы 1.0 с после 3-го слова и 0.7 с после 6-го
    ws = [_w(f" w{i}", i * 1.0 + (1.0 if i >= 3 else 0) + (0.7 if i >= 6 else 0), i * 1.0 + 0.8 +
             (1.0 if i >= 3 else 0) + (0.7 if i >= 6 else 0)) for i in range(9)]
    parts = split_words(ws)
    assert len(parts) >= 2
    assert sum(len(p) for p in parts) == 9
    assert all(p[-1].end - p[0].start <= 8.0 for p in parts)
    # порядок сохранён
    assert [w.text for p in parts for w in p] == [w.text for w in ws]


def test_long_segment_without_pauses_stays_whole():
    ws = [_w(f" w{i}", i * 1.0, i * 1.0 + 0.95) for i in range(12)]  # пауз 0.05 < 0.6
    assert split_words(ws) == [ws]


def test_split_words_empty():
    assert split_words([]) == []


def test_pick_device_explicit_and_unavailable():
    assert pick_device("cpu") == "cpu"
    assert pick_device("auto") in ("cuda", "mps", "cpu")
    assert pick_device("nonsense") == "cpu"


def test_pick_device_prefers_cuda_then_mps(monkeypatch):
    import dubl_engine.devices as d
    monkeypatch.setattr(d, "cuda_available", lambda: True)
    monkeypatch.setattr(d, "mps_available", lambda: True)
    assert d.pick_device("auto") == "cuda"
    monkeypatch.setattr(d, "cuda_available", lambda: False)
    assert d.pick_device("auto") == "mps"
    monkeypatch.setattr(d, "mps_available", lambda: False)
    assert d.pick_device("auto") == "cpu"
    assert d.pick_device("cuda") == "cpu"  # запрошена недоступная


def test_device_candidates_fallback_to_cpu_and_bad_marks(monkeypatch):
    import dubl_engine.devices as d
    monkeypatch.setattr(d, "cuda_available", lambda: False)
    monkeypatch.setattr(d, "mps_available", lambda: True)
    monkeypatch.setattr(d, "_BAD", set())
    assert d.device_candidates("auto", task="t") == ["mps", "cpu"]
    d.mark_bad("t", "mps")
    assert d.device_candidates("auto", task="t") == ["cpu"]
    assert d.device_candidates("auto", task="other") == ["mps", "cpu"]


def test_whisper_runtime():
    assert whisper_runtime("cuda") == ("cuda", "float16")
    assert whisper_runtime("mps") == ("cpu", "int8")
    assert whisper_runtime("cpu") == ("cpu", "int8")


def _seg(s, e, text, prob=0.9, lp=-0.3, ns=0.0):
    n = max(1, len(text.split()))
    ws = [Word(start=s + i * (e - s) / n, end=s + (i + 1) * (e - s) / n, text=" " + w, prob=prob)
          for i, w in enumerate(text.split())]
    return T.Seg(s, e, " " + text, ws, avg_logprob=lp, no_speech_prob=ns, compression_ratio=1.1)


def test_real_speech_is_kept_music_hallucinations_dropped():
    env = T.Envelope(np.full(2000, -25.0))
    segs = [_seg(10, 12, "Look Celia we have to follow our passions"),
            _seg(12.5, 14, "You have your robotics"),
            _seg(300, 301, "Thank you.", prob=0.55, lp=-0.4),                  # одинокая «спасибо» посреди пустоты
            _seg(500, 503, "I'm sorry.", prob=0.16, lp=-1.2)]
    out = T.drop_hallucinations(segs, env)
    assert [s.text.strip() for s in out] == ["Look Celia we have to follow our passions", "You have your robotics"]


def test_low_confidence_word_inside_dialogue_is_kept():
    env = T.Envelope(np.full(2000, -25.0))
    segs = [_seg(10, 12, "Is that so"), _seg(12.2, 13, "Oh", prob=0.5, lp=-0.8), _seg(13.2, 15, "Fine then")]
    assert len(T.drop_hallucinations(segs, env)) == 3


def test_silence_kills_any_segment():
    db = np.full(2000, -25.0)
    db[100:140] = -90.0                                     # 5–7 с цифровой тишины
    segs = [_seg(5.2, 6.8, "Hello there my friend", prob=0.95, lp=-0.2)]
    assert T.drop_hallucinations(segs, T.Envelope(db)) == []


def test_looping_phrase_and_stock_phrase_dropped():
    env = T.Envelope(np.full(4000, -25.0))
    loop = [_seg(100 + i * 3, 102 + i * 3, "I'm going to go to the next video", prob=0.9, lp=-0.3) for i in range(6)]
    real = [_seg(50, 52, "Where are you going"), _seg(52.5, 54, "Home")]
    out = T.drop_hallucinations(sorted(real + loop, key=lambda s: s.start), env)
    assert [s.text.strip() for s in out] == ["Where are you going", "Home"]
    stock = [_seg(200, 201, "We'll be right back.", prob=0.8, lp=-0.3)]
    assert T.drop_hallucinations(stock, env) == []


def test_runaway_repeat_text_dropped():
    env = T.Envelope(np.full(2000, -25.0))
    bad = _seg(10, 12, "A" * 60)
    bad.compression_ratio = 52.0
    assert T.drop_hallucinations([bad], env) == []


def test_envelope_mean_db_and_quietest():
    db = np.full(400, -20.0)
    db[100:130] = -80.0
    env = T.Envelope(db)
    assert env.mean_db(0, 1) == pytest.approx(-20.0, abs=0.1)
    assert env.mean_db(5.0, 6.5) < -70
    assert 5.0 <= env.quietest(5.5, 3.0) <= 6.5


def test_chunk_bounds_split_on_quiet_point():
    assert T._chunk_bounds(200.0, None) == [(0.0, 200.0)]
    db = np.full(int(900 / T.ENV_HOP), -20.0)
    q = int(310 / T.ENV_HOP)
    db[q:q + 20] = -90.0                                      # тихое место на 310 с
    b = T._chunk_bounds(900.0, T.Envelope(db))
    assert b[0][0] == 0.0 and b[-1][1] == 900.0
    assert all(x[1] == y[0] for x, y in zip(b, b[1:]))
    assert abs(b[0][1] - 310.5) < 1.0


def test_pick_backend(monkeypatch):
    monkeypatch.setattr(T, "mlx_available", lambda: True)
    assert T.pick_backend("large-v3-turbo", "auto") == "mlx"
    assert T.pick_backend("large-v3-turbo", "mps") == "mlx"
    assert T.pick_backend("large-v3-turbo", "cpu") == "faster"          # явно просили процессор
    assert T.pick_backend("some-custom-model", "auto") == "faster"
    monkeypatch.setattr(T, "mlx_available", lambda: False)
    assert T.pick_backend("large-v3-turbo", "auto") == "faster"


def test_mlx_unavailable_off_apple_silicon(monkeypatch):
    monkeypatch.setattr(T.sys, "platform", "linux")
    assert T.mlx_available() is False


def test_transcribe_falls_back_to_faster_when_mlx_fails(monkeypatch, tmp_path):
    import soundfile as sf
    wav = tmp_path / "a.wav"
    sf.write(str(wav), np.zeros(16000, dtype="float32") + 0.01, 16000)
    monkeypatch.setattr(T, "pick_backend", lambda *a, **k: "mlx")

    def boom(*a, **k):
        raise RuntimeError("metal упал")
    monkeypatch.setattr(T, "_transcribe_mlx", boom)
    seen = {}

    def fake_faster(path, **k):
        seen["vad"] = k["vad"]
        return [_seg(0.1, 0.9, "hello world")], "en"
    monkeypatch.setattr(T, "_transcribe_faster", fake_faster)
    lines, lang = transcribe(wav, language="en", model="large-v3-turbo", vad=False)
    assert lang == "en" and seen["vad"] is False
    assert [l.text for l in lines] == ["hello world"] and lines[0].id == "l0001"


def test_fw_vad_params_are_soft():
    # штатный VAD (threshold 0.5) терял диалог поверх музыки; наш порог должен быть на порядки мягче
    assert T.FW_VAD_PARAMS["threshold"] <= 0.05
    assert T.FW_VAD_PARAMS["speech_pad_ms"] >= 400


@pytest.fixture
def shared_models(monkeypatch):
    """Для медленных тестов — общий каталог моделей (не качать нейросети заново в каждом тесте)."""
    import os

    from platformdirs import user_data_dir
    monkeypatch.setenv("DUBL_DATA_DIR", os.environ.get("DUBL_TEST_DATA_DIR") or user_data_dir("Dubl", "Dubl"))


@pytest.mark.slow
@pytest.mark.skipif(sys.platform != "darwin" or not shutil.which("say"), reason="нужна macOS `say`")
def test_whisper_tiny_real_speech(tmp_path, shared_models):
    aiff = tmp_path / "x.aiff"
    wav = tmp_path / "x.wav"
    subprocess.run(["say", "-v", "Milena", "Привет, это проверка", "-o", str(aiff)], check=True)
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(aiff), "-ar", "16000", "-ac", "1",
                    str(wav)], check=True)
    progress: list[float] = []
    lines, lang = transcribe(wav, model="tiny", on_progress=progress.append)
    assert lang == "ru"
    assert lines and lines[0].id == "l0001" and lines[0].source == "whisper"
    text = " ".join(l.text for l in lines).lower()
    assert "привет" in text
    assert all(l.words for l in lines)
    assert all(l.start <= l.end for l in lines)
    assert progress and progress[-1] == 1.0


def test_release_memory_drops_mlx_model_cache():
    """mlx-whisper держит модель в глобальном ModelHolder (1,6 ГБ для turbo) — release_memory её сбрасывает."""
    pytest.importorskip("mlx_whisper")
    from mlx_whisper.transcribe import ModelHolder
    ModelHolder.model, ModelHolder.model_path = object(), "x"
    T.release_memory()
    assert ModelHolder.model is None and ModelHolder.model_path is None
