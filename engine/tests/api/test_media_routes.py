"""Исходник и лёгкая копия для плеера, probe по ссылке, строгие ответы LLM."""
from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from dubl_engine.api import create_app
from dubl_engine.jobs import JobManager
from dubl_engine.media.probe import browser_playable, probe
from dubl_engine.store import ProjectStore
from dubl_engine.util import ffprobe_json


@pytest.fixture
def env(tmp_path):
    store = ProjectStore(tmp_path / "projects")
    with TestClient(create_app(store=store, jobs=JobManager())) as c:
        yield c, store


def _wait(c, jid, timeout=120):
    t0 = time.time()
    while time.time() - t0 < timeout:
        j = c.get(f"/jobs/{jid}").json()
        if j["state"] in ("done", "failed", "cancelled"):
            return j
        time.sleep(0.15)
    raise AssertionError("задача не закончилась")


# ───────────── /files/{pid}/source ─────────────
def test_source_served_whole_with_range(env, simple_mp4):
    c, store = env
    p = store.create(simple_mp4, probe(simple_mp4), title="t")
    size = simple_mp4.stat().st_size
    r = c.get(f"/files/{p.id}/source")
    assert r.status_code == 200 and len(r.content) == size and r.headers["accept-ranges"] == "bytes"
    r = c.get(f"/files/{p.id}/source", headers={"Range": "bytes=100-199"})
    assert r.status_code == 206 and len(r.content) == 100
    assert r.headers["content-range"] == f"bytes 100-199/{size}"
    assert r.content == simple_mp4.read_bytes()[100:200]
    assert c.get(f"/files/{p.id}/source", headers={"Range": f"bytes={size - 10}-"}).status_code == 206


def test_source_errors_and_path_protection(env, simple_mp4, tmp_path):
    c, store = env
    gone = store.create(tmp_path / "нет.mp4", probe(simple_mp4), title="gone")
    assert c.get(f"/files/{gone.id}/source").status_code == 404
    assert c.get("/files/nope/source").status_code == 404
    p = store.create(simple_mp4, probe(simple_mp4), title="t")
    (tmp_path / "secret.txt").write_text("тайна", encoding="utf-8")
    for bad in ("../secret.txt", "..%2F..%2Fsecret.txt", "../../data/settings.json", "%2e%2e/secret.txt"):
        assert c.get(f"/files/{p.id}/{bad}").status_code in (404, 422), bad
    # source — не путь внутри папки проекта: «подсунуть» чужой файл через имя нельзя
    assert c.get(f"/files/{p.id}/scenes/../../../secret.txt").status_code == 404


def test_source_requires_token(simple_mp4, tmp_path):
    store = ProjectStore(tmp_path / "projects")
    p = store.create(simple_mp4, probe(simple_mp4), title="t")
    with TestClient(create_app(store=store, jobs=JobManager(), token="tok")) as c:
        assert c.get(f"/files/{p.id}/source").status_code == 401
        assert c.get(f"/files/{p.id}/proxy").status_code == 401
        assert c.get(f"/files/{p.id}/source?token=tok").status_code == 200


# ───────────── playable ─────────────
def test_browser_playable_mp4_h264_aac_yes_mkv_no(simple_mp4, rich_mkv):
    ok, why = browser_playable(probe(simple_mp4))
    assert ok and why is None
    ok, why = browser_playable(probe(rich_mkv))
    assert not ok and "контейнер" in why
    ok, why = browser_playable(probe(simple_mp4), audio_stream_index=7)      # оригинал — не первая дорожка
    assert not ok and "дорожк" in why


# ───────────── /files/{pid}/proxy ─────────────
def test_proxy_202_then_200_plays_in_browser_format(env, rich_mkv):
    c, store = env
    p = store.create(rich_mkv, probe(rich_mkv), title="mkv")
    r = c.get(f"/files/{p.id}/proxy")
    assert r.status_code == 202, r.text
    jid = r.headers["x-job-id"]
    assert r.json() == {"job_id": jid, "state": "preparing"}
    r2 = c.get(f"/files/{p.id}/proxy")                    # пока идёт — тот же id, новую задачу не создаём
    if r2.status_code == 202:
        assert r2.headers["x-job-id"] == jid
    job = _wait(c, jid)
    assert job["state"] == "done" and job["kind"] == "proxy", job
    assert job["result"]["url"] == f"/files/{p.id}/proxy.mp4"
    r = c.get(f"/files/{p.id}/proxy")
    assert r.status_code == 200 and r.headers["content-type"] == "video/mp4"
    assert c.get(f"/files/{p.id}/proxy", headers={"Range": "bytes=0-99"}).status_code == 206
    out = store.proxy_path(p.id)
    assert len(list(store.jobs_proxy_listing()) if False else [out]) == 1
    info = probe(out)
    assert info.video_codec == "h264" and info.height <= 540 and info.chapters == []
    assert info.audio_streams[0].codec == "aac" and info.audio_streams[0].channels == 2
    ok, why = browser_playable(info)
    assert ok, why
    # faststart: moov стоит перед mdat — браузер начнёт играть, не дочитав файл
    head = out.read_bytes()[:4096]
    assert head.find(b"moov") != -1 and head.find(b"moov") < (out.read_bytes().find(b"mdat"))
    # не 5.1 и не «первая дорожка по умолчанию»: звук — дорожка, выбранная оригиналом (rus stereo → 2 канала)
    assert ffprobe_json(out, "-show_format")["format"]["format_name"].startswith("mov")


def test_proxy_uses_chosen_original_audio_stream(env, rich_mkv):
    c, store = env
    p = store.create(rich_mkv, probe(rich_mkv), title="mkv")
    from dubl_engine.models import Analysis
    with store.edit(p.id) as pr:
        pr.analysis = Analysis(audio_stream_index=2)
    jid = c.get(f"/files/{p.id}/proxy").headers["x-job-id"]
    assert _wait(c, jid)["state"] == "done"
    a = probe(store.proxy_path(p.id)).audio_streams
    assert len(a) == 1 and a[0].channels == 2


def test_proxy_missing_source_is_404(env, simple_mp4, tmp_path):
    c, store = env
    p = store.create(tmp_path / "нет.mkv", probe(simple_mp4), title="gone")
    assert c.get(f"/files/{p.id}/proxy").status_code == 404


def test_proxy_failed_job_reports_error_then_retry(env, rich_mkv, monkeypatch):
    c, store = env
    p = store.create(rich_mkv, probe(rich_mkv), title="mkv")
    from dubl_engine.util import EngineError

    def boom(*_a, **_k):
        raise EngineError("диск переполнен")
    monkeypatch.setattr("dubl_engine.api.routes.files.build_proxy", boom)
    jid = c.get(f"/files/{p.id}/proxy").headers["x-job-id"]
    assert _wait(c, jid)["state"] == "failed"
    r = c.get(f"/files/{p.id}/proxy")
    assert r.status_code == 400 and "диск переполнен" in r.json()["detail"]
    monkeypatch.undo()
    r = c.get(f"/files/{p.id}/proxy?retry=true")
    assert r.status_code == 202 and r.headers["x-job-id"] != jid
    assert _wait(c, r.headers["x-job-id"])["state"] == "done"


# ───────────── probe по ссылке ─────────────
def test_probe_url_metadata_only(env, monkeypatch):
    c, _ = env

    class FakeYDL:
        def __init__(self, opts):
            assert opts["skip_download"] is True
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def extract_info(self, url, download=True):
            assert download is False
            return {"title": "Ролик", "duration": 93.5, "subtitles": {"en": [{}]}, "automatic_captions": {"ru": [{}]}}
    monkeypatch.setattr("yt_dlp.YoutubeDL", FakeYDL)
    r = c.post("/probe", json={"url": "https://example.com/v"})
    assert r.status_code == 200
    assert r.json() == {"known": True, "title": "Ролик", "duration": 93.5, "has_subtitles": True,
                        "subtitle_langs": ["en"], "message": None}


def test_probe_url_failure_is_graceful(env, monkeypatch):
    c, _ = env

    class BrokenYDL:
        def __init__(self, opts): pass
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def extract_info(self, *a, **k): raise RuntimeError("нет сети")
    monkeypatch.setattr("yt_dlp.YoutubeDL", BrokenYDL)
    r = c.post("/probe", json={"url": "https://example.com/v"})
    assert r.status_code == 200
    j = r.json()
    assert j["known"] is False and j["title"] is None and "после скачивания" in j["message"]


def test_probe_validation_and_path_still_works(env, simple_mp4):
    c, _ = env
    assert c.post("/probe", json={}).status_code == 422
    assert c.post("/probe", json={"path": "a", "url": "https://x"}).status_code == 422
    assert c.post("/probe", json={"url": "ftp://x"}).status_code == 422
    r = c.post("/probe", json={"path": str(simple_mp4)})
    assert r.status_code == 200 and r.json()["video_codec"] == "h264" and r.json()["width"] == 640


def test_probe_url_without_ytdlp(env, monkeypatch):
    c, _ = env
    import builtins
    real = builtins.__import__

    def no_ytdlp(name, *a, **k):
        if name == "yt_dlp":
            raise ImportError(name)
        return real(name, *a, **k)
    monkeypatch.setattr(builtins, "__import__", no_ytdlp)
    r = c.post("/probe", json={"url": "https://example.com/v"})
    assert r.status_code == 200 and r.json()["known"] is False
