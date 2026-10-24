"""Нейросетевой пакет: /health сообщает возможности, нехватка нейросетей — машинный признак в задаче, повторный разбор."""
from __future__ import annotations

import sys
import time

import pytest
from fastapi.testclient import TestClient

from dubl_engine.api import create_app
from dubl_engine.api.routes import health as health_mod
from dubl_engine.jobs import JobManager
from dubl_engine.models import MediaInfo
from dubl_engine.store import ProjectStore
from dubl_engine.util import MlMissingError


def _media():
    return MediaInfo(path="x", duration=10.0, container="mkv", width=640, height=360, fps=25, video_codec="h264")


@pytest.fixture
def client(tmp_path):
    store = ProjectStore(tmp_path / "projects")
    with TestClient(create_app(store=store, jobs=JobManager())) as c:
        c.store = store  # type: ignore[attr-defined]
        yield c


def _wait(client, jid, states=("failed", "done", "cancelled")):
    t0 = time.time()
    while time.time() - t0 < 15:
        j = client.get(f"/jobs/{jid}").json()
        if j["state"] in states:
            return j
        time.sleep(0.05)
    raise AssertionError(j)


def test_health_reports_capabilities(client):
    caps = client.get("/health").json()["capabilities"]
    assert {"ml", "asr", "separation", "diarization", "gigaam", "gigaam_downloaded", "gigaam_download_mb"} <= set(caps)
    assert caps["ml"] == (caps["asr"] and caps["separation"] and caps["diarization"])


def test_capabilities_without_ml_packages(monkeypatch):
    """Лёгкий движок: ни torch, ни whisper — ml=false по всем пунктам. Модули не импортируются, только ищутся."""
    for name in ("torch", "faster_whisper", "mlx_whisper", "demucs", "audio_separator", "speechbrain", "onnx_asr"):
        monkeypatch.setitem(sys.modules, name, None)       # find_spec на None-модуле бросает ValueError → считаем «нет»
    caps = health_mod.capabilities()
    assert (caps.ml, caps.asr, caps.separation, caps.diarization, caps.gigaam) == (False, False, False, False, False)
    assert caps.gigaam_downloaded is False and caps.gigaam_download_mb > 0


def test_capabilities_partial(monkeypatch):
    present = {"faster_whisper"}
    monkeypatch.setattr("importlib.util.find_spec", lambda m: object() if m in present else None)
    caps = health_mod.capabilities()
    assert caps.asr and not caps.separation and not caps.diarization and not caps.ml


def test_ml_missing_error_goes_to_job_error_code(client):
    def fn(ctx):
        with ctx.step("transcribe"):
            raise MlMissingError("Не установлен faster-whisper — распознавание речи недоступно.")
    job = client.app.state.jobs.submit("analyze", "p", [("transcribe", "Текст")], fn)
    j = _wait(client, job.id)
    assert j["state"] == "failed" and j["error_code"] == "ml_missing" and "faster-whisper" in j["error"]


def test_other_errors_have_no_code(client):
    from dubl_engine.util import EngineError

    def fn(ctx):
        raise EngineError("Файл не найден")
    job = client.app.state.jobs.submit("analyze", "p", [("a", "A")], fn)
    assert _wait(client, job.id)["error_code"] is None


def test_reanalyze_route(client, monkeypatch, tmp_path):
    src = tmp_path / "v.mp4"
    src.write_bytes(b"x")
    p = client.store.create(str(src), _media(), title="t")
    calls = []

    def fake(store, pid, ctx, *, settings, audio_stream_index=None, language=None):
        calls.append((pid, audio_stream_index, language))
        with ctx.step("probe"):
            pass
    monkeypatch.setattr("dubl_engine.api.routes.projects.analyze_project", fake)
    r = client.post(f"/projects/{p.id}/analyze", json={"language": "en"})
    assert r.status_code == 200 and r.json()["kind"] == "analyze" and r.json()["project_id"] == p.id
    assert _wait(client, r.json()["id"])["state"] == "done"
    assert calls == [(p.id, None, "en")]
    # тело необязательно
    assert client.post(f"/projects/{p.id}/analyze").status_code == 200


def test_reanalyze_errors(client, tmp_path):
    assert client.post("/projects/nope/analyze").status_code == 404
    p = client.store.create(str(tmp_path / "missing.mp4"), _media(), title="t")
    r = client.post(f"/projects/{p.id}/analyze")
    assert r.status_code == 400 and "Видео не найдено" in r.json()["detail"]


def test_transcribe_without_faster_whisper_raises_ml_missing(monkeypatch, tmp_path):
    from dubl_engine.analysis import transcribe as tr
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"RIFF")
    monkeypatch.setitem(sys.modules, "faster_whisper", None)
    monkeypatch.setattr(tr, "pick_backend", lambda *a, **k: "faster")
    with pytest.raises(MlMissingError):
        tr.transcribe(wav, language="en")
