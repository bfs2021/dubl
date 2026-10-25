"""Полный цикл БЕЗ нейросетей: rich_mkv → разбор → сцена → голоса/фон → дубль → подгонка → превью → рендер 9:16.

Замоканы только whisper, диаризация и сама нейросеть разделения; всё остальное — настоящее
(ffmpeg, субтитры, подгонка, сведение, мастеринг, рендер, очередь задач, HTTP).
"""
from __future__ import annotations

import io
import os
import time
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from scipy.signal import stft

from dubl_engine.api import create_app
from dubl_engine.audio.io import read_wav, write_wav
from dubl_engine.jobs import JobManager
from dubl_engine.models import Line, Word
from dubl_engine.store import ProjectStore
from dubl_engine.util import ffprobe_json, run_ffmpeg

from ..audio.synth_voice import voice_track

SR = 48000
SCENE_START, SCENE_END = 1.0, 9.0         # сцена 8 с: реплики Барсика (2–4) и Милы (6–8) → rel 1–3 и 5–7
PREROLL = 1.0
DELAY = 0.25                              # дубль опоздал на 250 мс
DUB_HZ = 1000.0


def _fake_whisper(path, **_kw):
    """Что бы «услышал» whisper в rich_mkv: две фразы на 2–4 и 6–8 с."""
    def line(i, a, b, text):
        words = [Word(start=a, end=b, text=text)]
        return Line(id=f"l{i:04d}", start=a, end=b, text=text, words=words, source="whisper")
    return [line(1, 2.05, 3.95, "Who touched my sausage?"), line(2, 6.05, 7.95, "Not me.")], "en"


def _fake_diarize(path, lines, **_kw):
    raise AssertionError("диаризация не нужна: в ASS есть имена")


def _fake_net(kind, x, sr, device, on_progress):
    """Идеальная «нейросеть» по полному миксу rich_mkv: голос = полоса около 600 Гц, фон = остальное."""
    from scipy.signal import butter, sosfiltfilt
    sos = butter(4, [450, 800], btype="bandpass", fs=sr, output="sos")
    mono = (x if x.ndim == 1 else x.mean(axis=1)).astype(np.float64)
    voice = sosfiltfilt(sos, mono).astype(np.float32)
    st = x if x.ndim == 2 else np.stack([x, x], axis=1)
    other = (st - voice[:, None]).astype(np.float32)
    return voice, other


def _dub_wav_bytes() -> bytes:
    """Дубль первой реплики: «речь» 1000 Гц (слоги по 0.3 с с провалами 60 мс), запись с преролом 1 с,
    речь началась на 250 мс позже, чем в оригинале."""
    t = np.arange(int(2.0 * SR)) / SR
    gate = ((t % 0.36) < 0.30).astype(np.float32)
    k = int(0.005 * SR)
    gate = np.convolve(gate, np.ones(k) / k, mode="same").astype(np.float32)
    voice = (0.4 * gate * np.sin(2 * np.pi * DUB_HZ * t)).astype(np.float32)
    total = np.zeros(int((PREROLL + 8.0) * SR), dtype=np.float32)
    start = int((PREROLL + 1.0 + DELAY) * SR)       # сцена-время 1.0 + опоздание
    total[start:start + len(voice)] = voice
    total += 0.0003 * np.random.default_rng(1).standard_normal(len(total)).astype(np.float32)
    buf = io.BytesIO()
    import soundfile as sf
    sf.write(buf, total, SR, format="WAV", subtype="FLOAT")
    return buf.getvalue()


def _wait(client: TestClient, job_id: str, timeout: float = 240.0) -> dict:
    t0 = time.time()
    while time.time() - t0 < timeout:
        j = client.get(f"/jobs/{job_id}").json()
        if j["state"] in ("done", "failed", "cancelled"):
            return j
        time.sleep(0.15)
    raise AssertionError("задача не закончилась")


def _tone_onset(x: np.ndarray, sr: int, hz: float) -> tuple[float, float]:
    """(время начала, конец) звучания тона ~hz по STFT: уровень полосы выше половины максимума."""
    f, t, z = stft(x, fs=sr, nperseg=1024, noverlap=768)
    band = np.abs(z[(f > hz - 60) & (f < hz + 60)]).sum(axis=0)
    thr = band.max() * 0.25
    idx = np.flatnonzero(band > thr)
    return float(t[idx[0]]), float(t[idx[-1]])


@pytest.fixture
def env(monkeypatch, tmp_path):
    monkeypatch.setattr("dubl_engine.analysis.transcribe.transcribe", _fake_whisper)
    monkeypatch.setattr("dubl_engine.analysis.diarize.diarize", _fake_diarize)
    monkeypatch.setattr("dubl_engine.audio.separate._run_net", _fake_net)
    store = ProjectStore(tmp_path / "projects")
    app = create_app(store=store, jobs=JobManager())
    with TestClient(app) as client:
        yield client, store


def test_full_cycle(env, rich_mkv, tmp_path):
    client, store = env

    # 1. проект + разбор
    r = client.post("/projects", json={"path": str(rich_mkv)})
    assert r.status_code == 200, r.text
    pid = r.json()["id"]
    j = _wait(client, r.headers["x-job-id"])
    assert j["state"] == "done", j
    p = client.get(f"/projects/{pid}").json()
    an = p["analysis"]
    assert an["language"] == "en" and an["audio_stream_index"] == 1
    assert [s["name"] for s in an["speakers"]] == ["Barsik", "Mila"]
    assert len(an["lines"]) == 2
    assert an["lines"][0]["speaker_id"] == "s1" and an["lines"][1]["speaker_id"] == "s2"
    assert any(abs(c - 5.0) < 0.15 for c in an["shot_cuts"])
    assert (store.analysis_dir(pid) / "audio_16k.wav").is_file() and (store.analysis_dir(pid) / "center.wav").is_file()
    # тайминги реплик подогнаны к речи, а не уехали
    assert an["lines"][0]["start"] == pytest.approx(2.0, abs=0.2) and an["lines"][0]["end"] == pytest.approx(4.0, abs=0.2)

    # 2. сцена + подготовка
    r = client.post(f"/projects/{pid}/scenes", json={"start": SCENE_START, "end": SCENE_END, "title": "Колбаса"})
    assert r.status_code == 200, r.text
    scid = r.json()["id"]
    j = _wait(client, r.headers["x-job-id"])
    assert j["state"] == "done", j
    sd = store.scene_dir(pid, scid)
    for name in ("clip.mp4", "original.wav", "voices.wav", "background.wav", "voices/s1.wav", "voices/s2.wav"):
        assert (sd / name).is_file(), name
    sc = client.get(f"/projects/{pid}/scenes/{scid}").json()
    assert sc["stems_ready"] and [ln["line_id"] for ln in sc["lines"]] == ["l0001", "l0002"]
    l1 = sc["lines"][0]
    assert l1["start_rel"] == pytest.approx(1.0, abs=0.2) and l1["end_rel"] == pytest.approx(3.0, abs=0.2)
    voices, sr = read_wav(sd / "voices.wav")
    orig, sro = read_wav(sd / "original.wav")
    assert sr == sro == SR and len(voices) == len(orig) and voices.ndim == 1 and orig.ndim == 2
    assert len(orig) / SR == pytest.approx(SCENE_END - SCENE_START, abs=0.05)
    # голос Барсика звучит только на его реплике
    v1, _ = read_wav(sd / "voices/s1.wav")
    assert np.abs(v1[int(1.3 * SR):int(2.7 * SR)]).max() > 0.01 and np.abs(v1[int(5.3 * SR):int(6.7 * SR)]).max() < 1e-4

    # 3. озвучиваем Барсика: отдельно «кого озвучиваем» не выбирали — запись дубля сама отметит персонажа
    assert sc["dubbed_speakers"] == []

    # 4. дубль (опоздал на 250 мс) → автоподгонка
    r = client.post(f"/projects/{pid}/scenes/{scid}/takes",
                    files={"file": ("take.wav", _dub_wav_bytes(), "audio/wav")},
                    data={"line_id": "l0001", "start_rel": "0.0", "preroll": str(PREROLL)})
    assert r.status_code == 200, r.text
    take = r.json()
    assert (sd / take["file"]).is_file() and take["selected"] and take["speaker_id"] == "s1"
    assert take["offset_ms"] == pytest.approx(-DELAY * 1000, abs=60), take
    assert take["stretch"] == pytest.approx(1.0, abs=0.06) and take["fit_confidence"] > 0.3
    assert client.get(f"/projects/{pid}/scenes/{scid}").json()["dubbed_speakers"] == ["s1"]

    # 5. превью
    r = client.post(f"/projects/{pid}/scenes/{scid}/preview")
    assert r.status_code == 200, r.text
    j = _wait(client, r.json()["id"])
    assert j["state"] == "done", j
    prev = sd / "preview.mp4"
    assert prev.is_file() and j["result"]["url"] == f"/files/{pid}/scenes/{scid}/preview.mp4"
    assert client.get(j["result"]["url"]).status_code == 200
    # после сборки — проверка сведения: автопилот включён, голос не тонет → предупреждения нет
    chk = client.get(f"/projects/{pid}/scenes/{scid}").json()["mix_check"]
    assert chk is not None and chk["voice_over_bg_db"] is not None and chk["warning"] is None, chk

    # 6. рендер 9:16
    out = tmp_path / "готово: ролик 'один'.mp4"
    rs = {"aspect": "9:16", "height": 360, "target_lufs": -16.0, "subtitles": "burn", "output_path": str(out)}
    r = client.post(f"/projects/{pid}/scenes/{scid}/render", json=rs)
    assert r.status_code == 200, r.text
    j = _wait(client, r.json()["id"])
    assert j["state"] == "done", j
    if os.name == "nt":             # «:» Windows в имени не примет — движок заменяет его на «_»
        assert Path(j["result"]["path"]) == out.with_name("готово_ ролик 'один'.mp4")
        out = Path(j["result"]["path"])
    assert out.is_file()
    info = ffprobe_json(out, "-show_streams", "-show_format")
    v = next(s for s in info["streams"] if s["codec_type"] == "video")
    a = next(s for s in info["streams"] if s["codec_type"] == "audio")
    assert (v["width"], v["height"]) == (360, 640) and v["codec_name"] == "h264"
    assert float(info["format"]["duration"]) == pytest.approx(SCENE_END - SCENE_START, abs=0.25)
    assert a["codec_name"] == "aac"

    # 7. звук итога: где дубль, громкость, оригинал Милы на месте, оригинал Барсика пропал
    wav = tmp_path / "out.wav"
    run_ffmpeg(["-i", out, "-vn", "-ac", "1", "-ar", str(SR), "-c:a", "pcm_f32le", wav])
    y, _ = read_wav(wav)
    t0, t1 = _tone_onset(y[: int(4.0 * SR)], SR, DUB_HZ)
    # дубль встал на место (начало речи в оригинале на rel 1.0 с — по подогнанному сдвигу ±0.12 с)
    assert t0 == pytest.approx(1.0, abs=0.15), (t0, t1, take)
    # оригинальный голос Барсика (600 Гц) на 1–3 с заглушён, у Милы (5–7 с) остался
    f, tt, z = stft(y, fs=SR, nperseg=2048)
    b600 = np.abs(z[(f > 560) & (f < 640)]).sum(axis=0)
    in_a = b600[(tt > 1.3) & (tt < 2.7)].mean()
    in_b = b600[(tt > 5.3) & (tt < 6.7)].mean()
    assert in_b > 8 * in_a, (in_a, in_b)
    from dubl_engine.audio.master import measure
    assert measure(y, SR).lufs == pytest.approx(-16.0, abs=1.0)


def test_dub_position_formula_matches_between_fit_and_mix(tmp_path):
    """Формула позиции дубля одна на fit_take и mix_scene: то, что насчитала подгонка, возвращает речь точно на место."""
    from dubl_engine.audio.fit import fit_take
    from dubl_engine.audio.mix import SceneStems, mix_scene
    from dubl_engine.models import MixSettings, Scene, SceneLine, Take, TrackSettings

    dur = 8.0
    guide = voice_track([(2.0, 4.0)], dur, seed=3, f0=150)
    write_wav(tmp_path / "guide.wav", guide, SR)
    write_wav(tmp_path / "bg.wav", np.zeros((int(dur * SR), 2), dtype=np.float32), SR)
    for preroll, start_rel, delay in ((1.0, 0.0, 0.3), (0.5, 1.5, -0.2), (2.0, 0.0, 0.0)):
        # дубль: речь «guide» (та же) сдвинута на delay; в файле время t ↔ сцена start_rel + t − preroll
        take = np.zeros(int((dur - start_rel + preroll) * SR), dtype=np.float32)
        i = int((preroll + 2.0 - start_rel + delay) * SR)
        sp = voice_track([(0, 2.0)], 2.0, seed=3, f0=150)
        take[i:i + len(sp)] = sp[: len(take) - i]
        (tmp_path / "takes").mkdir(exist_ok=True)
        write_wav(tmp_path / "takes" / "t001.wav", take, SR)
        fit = fit_take(tmp_path / "takes" / "t001.wav", tmp_path / "guide.wav", line_start_rel=2.0, line_end_rel=4.0,
                       take_start_rel=start_rel, preroll=preroll, latency_ms=0.0)
        t = Take(id="t001", line_id="l1", speaker_id="s1", file="takes/t001.wav", duration=len(take) / SR,
                 preroll=preroll, start_rel=start_rel, offset_ms=fit.offset_ms, stretch=fit.stretch, selected=True)
        scene = Scene(id="sc001", title="t", start=0, end=dur, dubbed_speakers=["s1"], takes=[t],
                      lines=[SceneLine(line_id="l1", speaker_id="s1", start_rel=2.0, end_rel=4.0, original="a", text="b")],
                      mix=MixSettings(autopilot=False, tracks={"s1": TrackSettings(room="dry")}, keep_breaths=False,
                                      duck_background=False))
        stems = SceneStems(voices=tmp_path / "guide.wav", background=tmp_path / "bg.wav", per_speaker={},
                           takes_dir=tmp_path / "takes", scene_dir=tmp_path)
        y, _ = read_wav(mix_scene(scene, stems, tmp_path / f"mix_{preroll}.wav"))
        mono = y.mean(axis=1)
        # огибающая результата против огибающей оригинала: задержка по кросс-корреляции
        def env(x):
            k = int(0.02 * SR)
            return np.convolve(np.abs(x), np.ones(k) / k, mode="same")[:: 480]
        a, b = env(mono), env(guide)
        n = min(len(a), len(b))
        a, b = a[:n] - a[:n].mean(), b[:n] - b[:n].mean()
        cc = np.correlate(a, b, mode="full")
        lag = (np.argmax(cc) - (n - 1)) * 0.01
        assert abs(lag) <= 0.06, (preroll, start_rel, delay, lag, fit)
