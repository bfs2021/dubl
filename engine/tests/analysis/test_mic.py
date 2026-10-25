"""Проверка микрофона: уровни, шум, перегруз, советы; замер задержки петлёй; профиль «Мой микрофон»."""
from __future__ import annotations

import io

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from dubl_engine.analysis import mic
from dubl_engine.api import create_app
from dubl_engine.jobs import JobManager
from dubl_engine.store import ProjectStore

SR = 48000


def _rec(*, noise_db=-62.0, speech_db=-20.0, silence_s=2.0, speech_s=3.0, clip=False, seed=1) -> np.ndarray:
    """2 с тишины (шум noise_db RMS) и фраза: гласные с провалами, уровень речи speech_db (RMS по фразе)."""
    rng = np.random.default_rng(seed)
    n = int((silence_s + speech_s) * SR)
    t = np.arange(n) / SR
    x = 10 ** (noise_db / 20) * rng.standard_normal(n)
    sp = t >= silence_s
    tt = t[sp] - silence_s
    env = (np.sin(2 * np.pi * 2.5 * tt) > -0.6).astype(float) * (0.7 + 0.3 * np.sin(2 * np.pi * 0.7 * tt))
    voice = (np.sin(2 * np.pi * 180 * tt) + 0.5 * np.sin(2 * np.pi * 540 * tt)) * env
    on = env > 0.05
    voice *= 10 ** (speech_db / 20) / np.sqrt(np.mean(voice[on] ** 2))
    x[sp] += voice
    if clip:
        x = np.clip(x * 8, -1.0, 1.0)
    return np.clip(x, -1.0, 1.0).astype(np.float32)


def _wav(x, sr=SR) -> bytes:
    b = io.BytesIO()
    sf.write(b, x, sr, format="WAV")
    return b.getvalue()


# ───────────── уровни и советы ─────────────
def test_good_microphone():
    m = mic.analyze_mic(_rec(), SR, silence_s=2.0)
    assert m.verdict == "good" and m.advice.startswith("Всё хорошо")
    assert m.noise_floor_db == pytest.approx(-62.0, abs=1.5)
    assert m.speech_db == pytest.approx(-20.0, abs=1.5)
    assert -17 < m.peak_db < -12 and m.snr_db == pytest.approx(41, abs=2.5) and not m.clipping and m.speech_found


@pytest.mark.parametrize("kw,verdict,fragment", [
    (dict(speech_db=-42.0), "quiet", "подойди ближе"),
    (dict(speech_db=-11.0), "loud", "Громковато"),
    (dict(speech_db=-18.0, noise_db=-38.0), "noisy", "шумно"),
    (dict(speech_db=-6.0, clip=True), "loud", "рвётся"),
])
def test_verdicts_and_advice(kw, verdict, fragment):
    m = mic.analyze_mic(_rec(**kw), SR, silence_s=2.0)
    assert m.verdict == verdict and fragment in m.advice, m
    assert m.clipping is bool(kw.get("clip"))


def test_quiet_by_average_level_even_with_normal_peak():
    """Профиль с ПК пользователя: речь −35,4 дБ, пик −14,6, шум −67. Пик в норме, а голос тихий — так и говорим."""
    v, adv = mic.judge(-67.0, -35.4, -14.6, False, True)
    assert v == "quiet" and adv.startswith("Микрофон тихий") and "настройках звука" in adv
    assert mic.judge(-67.0, -24.0, -10.0, False, True)[0] == "good"


def test_silent_microphone():
    x = np.zeros(5 * SR, dtype=np.float32)
    m = mic.analyze_mic(x, SR, silence_s=2.0)
    assert m.verdict == "silent" and not m.speech_found and "Не слышу" in m.advice
    quiet = _rec(speech_db=-70.0, noise_db=-80.0)
    assert mic.analyze_mic(quiet, SR, silence_s=2.0).verdict == "silent"


def test_works_without_silence_hint():
    m = mic.analyze_mic(_rec(), SR)
    assert m.verdict == "good" and m.noise_floor_db < -55


def test_clipping_needs_several_samples_on_the_ceiling():
    x = _rec(speech_db=-12.0)
    x[int(3 * SR)] = 1.0                       # одинокий пик — это ещё не перегруз
    assert not mic.analyze_mic(x, SR, silence_s=2.0).clipping


def test_resampling_of_browser_wav_44100():
    x44 = _rec()[::1]
    import scipy.signal as sg
    y = sg.resample_poly(x44, 441, 480).astype(np.float32)
    got = mic.load_audio(_wav(y, 44100))
    assert abs(len(got) - len(x44)) < 20
    assert mic.analyze_mic(got, mic.SR, silence_s=2.0).verdict == "good"


def test_load_audio_garbage_is_human():
    from dubl_engine.util import EngineError
    with pytest.raises(EngineError, match="Не удалось прочитать запись"):
        mic.load_audio(b"not audio at all" * 50)


# ───────────── задержка петлёй ─────────────
def _looped(latency_s: float, *, gain=0.1, noise=0.003, total=5.0, play_at=0.3, seed=0, echo=False) -> np.ndarray:
    rng = np.random.default_rng(seed)
    rec = noise * rng.standard_normal(int(total * SR))
    sig = mic.latency_signal(SR)
    i = int((play_at + latency_s) * SR)
    rec[i:i + len(sig)] += gain * sig
    if echo:
        j = i + int(0.04 * SR)                       # комната: эхо через 40 мс
        rec[j:j + len(sig)] += 0.4 * gain * sig
    return rec.astype(np.float32)


def test_signal_description_matches_wav_and_is_deterministic():
    x = mic.latency_signal()
    assert len(x) == int(mic.SIGNAL_DURATION_S * SR) and np.max(np.abs(x)) == pytest.approx(0.7, abs=0.01)
    assert np.array_equal(x, mic.latency_signal())
    # тишина до первого бипа и три бипа с паузами
    assert np.max(np.abs(x[:int(0.49 * SR)])) < 1e-6
    for s in mic.SIGNAL_BEEPS_S:
        assert np.max(np.abs(x[int((s + 0.01) * SR):int((s + 0.07) * SR)])) > 0.2
    assert np.max(np.abs(x[int(0.62 * SR):int(0.88 * SR)])) < 1e-6
    data, sr = sf.read(io.BytesIO(mic.latency_signal_wav()))
    assert sr == SR and data.ndim == 1 and len(data) == len(x)


@pytest.mark.parametrize("lat_ms", [0, 12, 87, 143.5, 380, 900])
def test_latency_accuracy(lat_ms):
    m = mic.measure_latency(_looped(lat_ms / 1000), SR, play_offset_s=0.3)
    assert m.latency_ms == pytest.approx(lat_ms, abs=1.0) and m.confidence >= 0.9


def test_latency_survives_noise_hum_and_echo():
    rng = np.random.default_rng(5)
    rec = _looped(0.12, gain=0.05, noise=0.01, echo=True)
    t = np.arange(len(rec)) / SR
    rec = rec + 0.05 * np.sin(2 * np.pi * 50 * t).astype(np.float32)          # гул сети
    m = mic.measure_latency(rec, SR, play_offset_s=0.3)
    assert m.latency_ms == pytest.approx(120, abs=3.0), m
    assert rng is not None


def test_latency_not_found_cases():
    noise = (0.003 * np.random.default_rng(1).standard_normal(5 * SR)).astype(np.float32)
    m = mic.measure_latency(noise, SR)
    assert m.latency_ms is None and m.confidence < 0.5 and "Сними наушники" in m.message
    m = mic.measure_latency(np.zeros(5 * SR, dtype=np.float32), SR)
    assert m.latency_ms is None and "ничего не слышит" in m.message
    m = mic.measure_latency(noise[:SR], SR)
    assert m.latency_ms is None and "слишком короткая" in m.message
    # речь вместо сигнала не должна дать «уверенный» ответ
    speech = _rec()[:5 * SR]
    assert mic.measure_latency(speech, SR).latency_ms is None


# ───────────── роуты и профиль ─────────────
@pytest.fixture
def client(tmp_path):
    with TestClient(create_app(store=ProjectStore(tmp_path / "p"), jobs=JobManager())) as c:
        yield c


def test_routes_check_saves_profile(client, tmp_path):
    assert client.get("/mic/profile").json() is None
    assert client.get("/settings").json()["mic_profile"] is None
    r = client.post("/mic/check", files={"file": ("a.wav", _wav(_rec()))}, data={"silence_s": "2"})
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["verdict"] == "good" and j["advice"].startswith("Всё хорошо") and j["saved"] is True
    prof = client.get("/mic/profile").json()
    assert prof["noise_floor_db"] == j["noise_floor_db"] and prof["speech_db"] == j["speech_db"]
    assert prof["latency_ms"] is None and prof["checked_at"]
    assert client.get("/settings").json()["mic_profile"]["noise_floor_db"] == j["noise_floor_db"]
    assert (tmp_path / "data" / "settings.json").exists()                  # профиль пережил бы перезапуск
    assert "mic_profile" in (tmp_path / "data" / "settings.json").read_text(encoding="utf-8")


def test_check_without_speech_does_not_touch_profile(client):
    client.post("/mic/check", files={"file": ("a.wav", _wav(_rec()))}, data={"silence_s": "2"})
    before = client.get("/mic/profile").json()
    r = client.post("/mic/check", files={"file": ("a.wav", _wav(np.zeros(4 * SR, dtype=np.float32)))})
    assert r.json()["verdict"] == "silent" and r.json()["saved"] is False
    assert client.get("/mic/profile").json() == before
    r = client.post("/mic/check", files={"file": ("a.wav", _wav(_rec()))}, data={"save": "false"})
    assert r.json()["saved"] is False


def test_latency_route_saves_to_profile_and_latency_ms(client):
    client.post("/mic/check", files={"file": ("a.wav", _wav(_rec()))}, data={"silence_s": "2"})
    r = client.post("/mic/latency", files={"file": ("l.wav", _wav(_looped(0.087)))}, data={"play_offset_s": "0.3"})
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["ok"] and j["latency_ms"] == pytest.approx(87, abs=1.5) and j["confidence"] >= 0.9 and j["saved"]
    prof = client.get("/mic/profile").json()
    assert prof["latency_ms"] == pytest.approx(87, abs=1.5) and prof["noise_floor_db"] is not None     # прежние поля целы
    assert client.get("/settings").json()["latency_ms"] == pytest.approx(87, abs=1.5)                 # его читает подгонка
    bad = client.post("/mic/latency", files={"file": ("l.wav", _wav(np.zeros(4 * SR, dtype=np.float32)))}).json()
    assert bad["ok"] is False and bad["latency_ms"] is None and bad["message"] and bad["saved"] is False
    assert client.get("/mic/profile").json()["latency_ms"] == prof["latency_ms"]


def test_signal_routes(client):
    info = client.get("/mic/latency_signal").json()
    assert info["sample_rate"] == 48000 and info["beep_count"] == 3 and info["beep_starts_s"] == [0.5, 0.9, 1.4]
    assert info["duration_s"] == 2.0 and info["beep_s"] == 0.08 and info["min_record_s"] == 3.0
    assert info["freq_from_hz"] == 1500.0 and info["freq_to_hz"] == 3500.0
    r = client.get(info["wav_url"])
    assert r.status_code == 200 and r.headers["content-type"] == "audio/wav"
    data, sr = sf.read(io.BytesIO(r.content))
    assert sr == 48000 and len(data) == 96000


def test_delete_profile_keeps_latency_setting(client):
    client.post("/mic/latency", files={"file": ("l.wav", _wav(_looped(0.2)))}, data={"play_offset_s": "0.3"})
    assert client.delete("/mic/profile").json() == {"ok": True}
    assert client.get("/mic/profile").json() is None
    assert client.get("/settings").json()["latency_ms"] == pytest.approx(200, abs=1.5)


def test_bad_uploads_are_human(client):
    r = client.post("/mic/check", files={"file": ("a.wav", b"")})
    assert r.status_code == 400 and "Пустая запись" in r.json()["detail"]
    r = client.post("/mic/check", files={"file": ("a.wav", b"mp3? no" * 100)})
    assert r.status_code == 400 and "Не удалось прочитать запись" in r.json()["detail"]
    assert client.post("/mic/latency", files={"file": ("a.wav", b"")}).status_code == 400
    assert client.post("/mic/check", files={"file": ("a.wav", _wav(_rec()))}, data={"silence_s": "-1"}).status_code == 422


def test_token_is_required():
    with TestClient(create_app(store=ProjectStore("/tmp/dubl-mic-token"), jobs=JobManager(), token="t")) as c:
        assert c.get("/mic/profile").status_code == 401
        assert c.get("/mic/latency_signal.wav").status_code == 401
        assert c.get("/mic/profile", headers={"Authorization": "Bearer t"}).status_code == 200
