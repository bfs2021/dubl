"""HTTP-слой: схема OpenAPI, токен, коды ошибок, гонки, пути файлов, секреты."""
from __future__ import annotations

import io
import threading
import time

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from dubl_engine.api import create_app
from dubl_engine.audio.io import write_wav
from dubl_engine.jobs import JobManager
from dubl_engine.models import MediaInfo, Scene
from dubl_engine.store import ProjectStore

SR = 48000


def _media(dur=10.0):
    return MediaInfo(path="x", duration=dur, container="mkv", width=640, height=360, fps=25, video_codec="h264")


@pytest.fixture
def make(tmp_path):
    clients = []

    def _make(token=None):
        store = ProjectStore(tmp_path / "projects")
        c = TestClient(create_app(store=store, jobs=JobManager(), token=token))
        c.__enter__()
        clients.append(c)
        return c, store
    yield _make
    for c in clients:
        c.__exit__(None, None, None)


# ───────────── схема ─────────────
def test_every_route_has_typed_response_and_tag(tmp_path):
    app = create_app(store=ProjectStore(tmp_path / "p"), jobs=JobManager())
    spec = app.openapi()
    for path, ops in spec["paths"].items():
        for method, op in ops.items():
            assert op.get("tags"), (method, path)
            ok = next(c for c in op["responses"] if c.startswith("2"))
            content = op["responses"][ok].get("content", {})
            if path.startswith("/files") or path.endswith(".wav"):
                continue
            schema = content.get("application/json", {}).get("schema")
            assert schema and schema != {}, f"{method.upper()} {path}: нет типа ответа"
    # ошибки описаны одной моделью
    assert "ErrorResponse" in spec["components"]["schemas"]


def test_schema_has_no_untyped_dicts_in_responses(tmp_path):
    spec = create_app(store=ProjectStore(tmp_path / "p"), jobs=JobManager()).openapi()
    sch = spec["components"]["schemas"]
    for name in ("Health", "Ok", "TranslateResult", "FitLengthResult", "NameSpeakersResult", "DirectorTakeResult",
                 "DirectorCommandResult", "DirectorReport", "SnapResult", "CheckResult"):
        assert name in sch, name


def test_job_header_is_documented_and_exposed(make, simple_mp4):
    c, _ = make()
    spec = c.get("/openapi.json").json()
    assert "X-Job-Id" in spec["paths"]["/projects"]["post"]["responses"]["200"]["headers"]
    r = c.options("/projects", headers={"Origin": "tauri://localhost", "Access-Control-Request-Method": "POST"})
    assert r.status_code == 200
    r = c.get("/health", headers={"Origin": "tauri://localhost"})
    assert "x-job-id" in r.headers.get("access-control-expose-headers", "").lower()


# ───────────── токен ─────────────
def test_token_required_except_health(make):
    c, _ = make(token="s3cret-token")
    assert c.get("/health").status_code == 200
    assert c.get("/projects").status_code == 401
    assert c.get("/projects", headers={"Authorization": "Bearer wrong"}).status_code == 401
    assert c.get("/projects", headers={"Authorization": "Bearer s3cret-token"}).status_code == 200
    assert c.get("/projects?token=s3cret-token").status_code == 200
    assert c.get("/jobs").status_code == 401 and c.get("/settings").status_code == 401
    assert c.get("/files/x/y").status_code == 401 and c.post("/llm/translate", json={}).status_code == 401
    with pytest.raises(Exception), c.websocket_connect("/ws/jobs"):
        pass
    with c.websocket_connect("/ws/jobs?token=s3cret-token"):
        pass


# ───────────── ошибки ─────────────
def test_error_codes(make, tmp_path):
    c, store = make()
    assert c.get("/projects/nope").status_code == 404
    assert c.get("/projects/..%2Fx").status_code in (404, 422)
    assert c.get("/jobs/nope").status_code == 404
    assert c.post("/jobs/nope/cancel").status_code == 404
    r = c.post("/projects", json={})
    assert r.status_code == 422 and isinstance(r.json()["detail"], str)
    r = c.post("/projects", json={"path": str(tmp_path / "нет.mkv")})
    assert r.status_code == 400 and "не найден" in r.json()["detail"]
    r = c.post("/projects", json={"url": "ftp://x"})
    assert r.status_code == 422
    p = store.create("/x.mp4", _media(), title="t")
    assert c.post(f"/projects/{p.id}/snap", json={"a": 1, "mode": "dialogue"}).status_code == 400   # не разобран
    assert c.post(f"/projects/{p.id}/scenes", json={"start": 5, "end": 5}).status_code == 400
    assert c.post(f"/projects/{p.id}/scenes", json={"start": 1, "end": 99}).status_code == 400
    assert c.post(f"/projects/{p.id}/scenes", json={"start": 1, "end": "nan"}).status_code == 422
    assert c.get(f"/projects/{p.id}/scenes/sc999").status_code == 404
    assert c.patch(f"/projects/{p.id}/scenes/sc999", json={}).status_code == 404
    assert c.post("/llm/translate", json={"project_id": p.id, "scene_id": "x"}).status_code == 400  # нет модели


def test_scene_not_ready_is_409(make):
    c, store = make()
    p = store.create("/x.mp4", _media(), title="t")
    with store.edit(p.id) as pr:
        pr.scenes.append(Scene(id="sc001", title="s", start=0, end=5))
    wav = io.BytesIO()
    sf.write(wav, np.zeros(4800, dtype=np.float32), SR, format="WAV")
    r = c.post(f"/projects/{p.id}/scenes/sc001/takes", files={"file": ("a.wav", wav.getvalue())})
    assert r.status_code == 409
    assert c.post(f"/projects/{p.id}/scenes/sc001/preview").status_code == 409
    assert c.post(f"/projects/{p.id}/scenes/sc001/render", json={"output_path": "o.mp4"}).status_code == 409


def test_unhandled_error_is_500_without_internals(make, monkeypatch):
    c, store = make()
    monkeypatch.setattr(ProjectStore, "list", lambda self: (_ for _ in ()).throw(RuntimeError("пароль=sk-SECRET1234567890")))
    r = TestClient(c.app, raise_server_exceptions=False).get("/projects")
    assert r.status_code == 500 and "SECRET" not in r.text and "RuntimeError" in r.text


def test_key_not_echoed_in_validation_errors(make):
    c, _ = make()
    r = c.put("/settings/llm/x/key", json={"key": ""})
    assert r.status_code == 422
    r = c.put("/settings/llm/x/key", json={"key": 123456789})
    assert "123456789" not in r.text


# ───────────── файлы ─────────────
def test_files_path_traversal_blocked(make, tmp_path):
    c, store = make()
    p = store.create("/x.mp4", _media(), title="t")
    (tmp_path / "secret.txt").write_text("top secret", encoding="utf-8")
    (store.project_dir(p.id) / "ok.txt").write_text("fine", encoding="utf-8")
    assert c.get(f"/files/{p.id}/ok.txt").text == "fine"
    for bad in ("../../secret.txt", "..%2F..%2Fsecret.txt", "/etc/passwd", "%2e%2e/%2e%2e/secret.txt"):
        assert c.get(f"/files/{p.id}/{bad}").status_code in (404, 400), bad
    assert c.get(f"/files/{p.id}/project.json").status_code == 200       # свой проект — можно


def test_files_range_and_source(make, simple_mp4):
    c, store = make()
    p = store.create(str(simple_mp4), _media(), title="t")
    r = c.get(f"/files/{p.id}/source", headers={"Range": "bytes=0-99"})
    assert r.status_code == 206 and len(r.content) == 100


# ───────────── гонки ─────────────
def test_parallel_takes_get_distinct_ids(make):
    c, store = make()
    p = store.create("/x.mp4", _media(), title="t")
    sd = store.scene_dir(p.id, "sc001")
    write_wav(sd / "voices.wav", np.zeros(SR * 5, dtype=np.float32), SR)
    with store.edit(p.id) as pr:
        pr.scenes.append(Scene(id="sc001", title="s", start=0, end=5, stems_ready=True))
    buf = io.BytesIO()
    sf.write(buf, (0.1 * np.sin(np.arange(SR) * 0.1)).astype(np.float32), SR, format="WAV")
    out, errs = [], []

    def post():
        try:
            r = c.post(f"/projects/{p.id}/scenes/sc001/takes", files={"file": ("a.wav", buf.getvalue())})
            out.append(r)
        except Exception as e:  # noqa: BLE001
            errs.append(e)
    th = [threading.Thread(target=post) for _ in range(6)]
    [t.start() for t in th]
    [t.join() for t in th]
    assert not errs and all(r.status_code == 200 for r in out), [r.text for r in out]
    ids = [r.json()["id"] for r in out]
    assert len(set(ids)) == 6
    takes = store.get(p.id).scenes[0].takes
    assert sorted(t.id for t in takes) == sorted(ids)
    for t in takes:
        assert (store.scene_dir(p.id, "sc001") / t.file).stat().st_size > 1000


def test_delete_project_with_running_job_cancels_and_removes(make):
    c, store = make()
    p = store.create("/x.mp4", _media(), title="t")
    jm = c.app.state.jobs
    started = threading.Event()

    def fn(ctx):
        with ctx.step("s"):
            started.set()
            for _ in range(500):
                ctx.progress("s", 0.1)
                time.sleep(0.02)
    job = jm.submit("analyze", p.id, [("s", "Шаг")], fn)
    assert started.wait(5)
    assert c.delete(f"/projects/{p.id}").status_code == 200
    assert not store.exists(p.id)
    assert jm.get(job.id).state.value == "cancelled"


def test_ws_streams_job_updates(make):
    c, store = make()
    p = store.create("/x.mp4", _media(), title="t")
    jm = c.app.state.jobs
    with c.websocket_connect("/ws/jobs") as ws:
        jm.submit("render", p.id, [("a", "A")], lambda ctx: ctx.progress("a", 0.5), heavy=False)
        states = set()
        for _ in range(10):
            states.add(ws.receive_json()["state"])
            if "done" in states:
                break
        assert "done" in states
