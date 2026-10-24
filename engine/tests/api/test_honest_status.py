"""Честные статусы после остановки движка (разбор, подготовка сцены), атомарная запись analysis и удаление проекта."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from dubl_engine.api import create_app
from dubl_engine.audio.io import write_wav
from dubl_engine.jobs import JobManager
from dubl_engine.media.extract import cut_scene_video
from dubl_engine.media.probe import probe
from dubl_engine.models import Analysis, MediaInfo, Project, Scene
from dubl_engine.store import ProjectStore
from dubl_engine.store.atomic import atomic_write_text
from dubl_engine.util import data_dir


def _media(dur=10.0):
    return MediaInfo(path="x", duration=dur, container="mkv", width=640, height=360, fps=25, video_codec="h264")


def _make_scene_files(store, pid, scid, simple_mp4, *, complete=True):
    sd = store.scene_dir(pid, scid)
    sd.mkdir(parents=True, exist_ok=True)
    cut_scene_video(simple_mp4, 0.0, 3.0, sd / "clip.mp4", audio_stream_index=1)
    x = np.zeros(3 * 48000, dtype=np.float32)
    write_wav(sd / "original.wav", np.stack([x, x], axis=1), 48000)
    write_wav(sd / "voices.wav", x, 48000)
    if complete:
        write_wav(sd / "background.wav", np.stack([x, x], axis=1), 48000)
    (sd / "voices").mkdir(exist_ok=True)


# ───────────── модели: статус не врёт ─────────────
def test_models_states_follow_data():
    p = Project(id="a", title="t", source_path="x", media=_media())
    assert p.analysis_state == "none"
    p2 = Project(id="a", title="t", source_path="x", media=_media(), analysis=Analysis(audio_stream_index=1))
    assert p2.analysis_state == "done"
    p3 = Project(id="a", title="t", source_path="x", media=_media(), analysis_state="done")
    assert p3.analysis_state == "interrupted"            # «готов» без analysis — неправда
    sc = Scene(id="sc001", title="s", start=0, end=3, stems_ready=True)
    assert sc.prepare_state == "ready"
    assert Scene(id="sc001", title="s", start=0, end=3, prepare_state="ready").prepare_state == "interrupted"
    assert Scene(id="sc001", title="s", start=0, end=3).prepare_state == "interrupted"


def test_old_project_json_without_new_fields_is_read(tmp_path):
    st = ProjectStore(tmp_path)
    p = st.create("/x.mp4", _media(), title="t")
    f = st.project_file(p.id)
    data = json.loads(f.read_text(encoding="utf-8"))
    data.pop("analysis_state", None)
    f.write_text(json.dumps(data), encoding="utf-8")
    assert st.get(p.id).analysis_state == "none"


# ───────────── восстановление при старте ─────────────
def test_recover_marks_unfinished_analysis_and_scenes_interrupted(tmp_path, simple_mp4):
    st = ProjectStore(tmp_path / "projects")
    p = st.create(simple_mp4, probe(simple_mp4), title="t")
    with st.edit(p.id) as pr:
        pr.analysis_state = "running"                   # так проект выглядел, когда движок убили
        pr.scenes.append(Scene(id="sc001", title="готова", start=0, end=3, stems_ready=True))
        pr.scenes.append(Scene(id="sc002", title="файлов нет", start=0, end=3, stems_ready=True))
        pr.scenes.append(Scene(id="sc003", title="обрезан", start=0, end=3, stems_ready=True))
        pr.scenes.append(Scene(id="sc004", title="не готовилась", start=0, end=3, prepare_state="running"))
    _make_scene_files(st, p.id, "sc001", simple_mp4)
    _make_scene_files(st, p.id, "sc003", simple_mp4, complete=False)       # нет background.wav
    before = st.get(p.id).updated

    assert st.recover_interrupted() == [p.id]
    got = st.get(p.id)
    assert got.analysis is None and got.analysis_state == "interrupted"
    by = {s.id: s for s in got.scenes}
    assert by["sc001"].stems_ready and by["sc001"].prepare_state == "ready"
    for sid in ("sc002", "sc003", "sc004"):
        assert not by[sid].stems_ready and by[sid].prepare_state == "interrupted", sid
    assert got.updated == before                          # проект не «обновился» от проверки
    [sm] = st.list()
    assert not sm.has_analysis and sm.analysis_state == "interrupted"
    assert st.recover_interrupted() == []                 # второй запуск ничего не меняет


def test_recover_detects_truncated_audio(tmp_path, simple_mp4):
    st = ProjectStore(tmp_path / "projects")
    p = st.create(simple_mp4, probe(simple_mp4), title="t")
    with st.edit(p.id) as pr:
        pr.analysis = Analysis(audio_stream_index=1)
        pr.scenes.append(Scene(id="sc001", title="s", start=0, end=3, stems_ready=True))
    _make_scene_files(st, p.id, "sc001", simple_mp4)
    bg = st.scene_dir(p.id, "sc001") / "background.wav"
    bg.write_bytes(bg.read_bytes()[:20000])               # запись оборвалась: заголовок длинный, данных мало
    assert st.recover_interrupted() == [p.id]
    sc = st.get(p.id).scenes[0]
    assert not sc.stems_ready and sc.prepare_state == "interrupted"
    assert st.get(p.id).analysis_state == "done"          # разбор целый — его не трогаем


def test_recover_removes_half_deleted_project_folder(tmp_path):
    st = ProjectStore(tmp_path / "projects")
    trash = st.root / ".trash-abc-123456"
    (trash / "scenes").mkdir(parents=True)
    (trash / "project.json").write_text("{}", encoding="utf-8")
    st.recover_interrupted()
    assert not trash.exists()


def test_atomic_write_keeps_old_content_if_replace_fails(tmp_path, monkeypatch):
    f = tmp_path / "project.json"
    atomic_write_text(f, "старое")

    def boom(*_a, **_k):
        raise OSError("диск отвалился")
    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError):
        atomic_write_text(f, "новое, полупустое")
    assert f.read_text(encoding="utf-8") == "старое"
    assert [p.name for p in tmp_path.iterdir()] == ["project.json"]         # временных хвостов нет


def test_killed_engine_mid_analysis_leaves_no_partial_analysis(tmp_path, simple_mp4):
    """Настоящий процесс: разбор доходит до распознавания, процесс убивают SIGKILL. На диске — проект без analysis,
    после старта движка — «разбор прерван», а повторный разбор ставится в очередь."""
    data = tmp_path / "data"
    root = data / "projects"
    mark = tmp_path / "mark"
    script = textwrap.dedent(f"""
        import os, sys, time
        os.environ["DUBL_DATA_DIR"] = {str(data)!r}
        from pathlib import Path
        from dubl_engine.media.probe import probe
        from dubl_engine.pipeline.analyze import analyze_project
        from dubl_engine.settings import AppSettings
        from dubl_engine.store import ProjectStore
        import dubl_engine.analysis.transcribe as tr
        def hang(*a, **k):
            Path({str(mark)!r}).write_text("x"); time.sleep(600)
        tr.transcribe = hang
        class Ctx:
            job_id = "x"; cancelled = False
            def step(self, key):
                import contextlib
                return contextlib.nullcontext()
            def progress(self, *a): pass
            def skip(self, *a): pass
            def check(self): pass
            def note(self, *a): pass
        st = ProjectStore({str(root)!r})
        p = st.create({str(simple_mp4)!r}, probe({str(simple_mp4)!r}), title="kill")
        print(p.id, flush=True)
        analyze_project(st, p.id, Ctx(), settings=AppSettings())
    """)
    proc = subprocess.Popen([sys.executable, "-c", script], stdout=subprocess.PIPE, text=True, encoding="utf-8", errors="replace")
    try:
        pid = proc.stdout.readline().strip()
        deadline = time.monotonic() + 60
        while not mark.exists() and time.monotonic() < deadline:
            time.sleep(0.1)
        assert mark.exists(), "разбор не дошёл до распознавания"
    finally:
        proc.kill()                     # SIGKILL на Unix, TerminateProcess на Windows: движок убит без уборки
        proc.wait(10)

    st = ProjectStore(root)
    assert st.get(pid).analysis is None                    # частичный разбор не записан
    with TestClient(create_app(store=st, jobs=JobManager(), recover=True)) as c:
        [sm] = c.get("/projects").json()
        assert sm["has_analysis"] is False and sm["analysis_state"] == "interrupted"
        assert c.get(f"/projects/{pid}").json()["analysis_state"] == "interrupted"
        assert c.post(f"/projects/{pid}/analyze").status_code == 200        # перезапуск принимается


# ───────────── статус поверх очереди задач ─────────────
@pytest.fixture
def env(tmp_path):
    store = ProjectStore(tmp_path / "projects")
    jobs = JobManager()
    with TestClient(create_app(store=store, jobs=jobs)) as c:
        yield c, store, jobs


def test_analysis_state_running_then_interrupted_after_cancel_then_done(env):
    c, store, jobs = env
    p = store.create("/x.mp4", _media(), title="t")
    started, release = threading.Event(), threading.Event()

    def fn(ctx):
        with ctx.step("s"):
            started.set()
            while not release.is_set():
                ctx.progress("s", 0.1)
                time.sleep(0.02)
    job = jobs.submit("analyze", p.id, [("s", "Шаг")], fn)
    assert started.wait(5)
    assert c.get("/projects").json()[0]["analysis_state"] == "running"
    assert c.get(f"/projects/{p.id}").json()["analysis_state"] == "running"
    c.post(f"/jobs/{job.id}/cancel")
    deadline = time.monotonic() + 5
    while jobs.get(job.id).state.value not in ("cancelled", "done") and time.monotonic() < deadline:
        time.sleep(0.05)
    s = c.get("/projects").json()[0]
    assert s["analysis_state"] == "interrupted" and not s["has_analysis"]
    with store.edit(p.id) as pr:
        pr.analysis = Analysis(audio_stream_index=1)
    s = c.get("/projects").json()[0]
    assert s["analysis_state"] == "done" and s["has_analysis"]


def test_failed_analysis_is_not_marked_done(env):
    c, store, jobs = env
    p = store.create("/x.mp4", _media(), title="t")

    def fn(ctx):
        raise RuntimeError("упало")
    job = jobs.submit("analyze", p.id, [("s", "Шаг")], fn)
    deadline = time.monotonic() + 5
    while jobs.get(job.id).state.value not in ("failed",) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert c.get("/projects").json()[0]["analysis_state"] == "interrupted"


def test_scene_prepare_state_and_reprepare_route(env, simple_mp4):
    c, store, jobs = env
    p = store.create(simple_mp4, probe(simple_mp4), title="t")
    with store.edit(p.id) as pr:
        pr.scenes.append(Scene(id="sc001", title="s", start=0, end=3))
    sc = c.get(f"/projects/{p.id}/scenes/sc001").json()
    assert sc["stems_ready"] is False and sc["prepare_state"] == "interrupted"
    assert c.get(f"/projects/{p.id}").json()["scenes"][0]["prepare_state"] == "interrupted"
    # запись/рендер на неготовой сцене отказывают
    assert c.post(f"/projects/{p.id}/scenes/sc001/preview").status_code == 409
    gate = threading.Event()
    jobs.submit("prepare_scene", p.id, [("cut", "Вырезка")], lambda ctx: gate.wait(5), scene_id="sc001")
    assert c.get(f"/projects/{p.id}/scenes/sc001").json()["prepare_state"] == "running"
    assert c.post(f"/projects/{p.id}/scenes/sc001/prepare").status_code == 409        # уже готовится
    gate.set()
    jobs_done = time.monotonic() + 5
    while jobs.active() and time.monotonic() < jobs_done:
        time.sleep(0.05)
    r = c.post(f"/projects/{p.id}/scenes/sc001/prepare")
    assert r.status_code == 200 and r.json()["kind"] == "prepare_scene" and r.json()["scene_id"] == "sc001"
    assert c.post(f"/projects/{p.id}/scenes/nope/prepare").status_code == 404


def test_prepare_scene_drops_ready_flag_while_running_and_sets_it_at_the_end(tmp_path, simple_mp4, monkeypatch):
    from dubl_engine.jobs import NullContext
    from dubl_engine.pipeline import prepare_scene
    st = ProjectStore(tmp_path / "projects")
    p = st.create(simple_mp4, probe(simple_mp4), title="t")
    with st.edit(p.id) as pr:
        pr.scenes.append(Scene(id="sc001", title="s", start=0, end=3, stems_ready=True))
    seen = {}

    def fake_sep(wav, out_dir, **_kw):
        seen["during"] = (st.get(p.id).scenes[0].stems_ready, st.get(p.id).scenes[0].prepare_state)
        from dubl_engine.audio.io import read_wav
        x, sr = read_wav(wav)
        write_wav(out_dir / "voices.wav", x.mean(axis=1), sr)
        write_wav(out_dir / "background.wav", x, sr)
        return out_dir / "voices.wav", out_dir / "background.wav"
    monkeypatch.setattr("dubl_engine.audio.separate.separate", fake_sep)
    sc = prepare_scene(st, p.id, "sc001", NullContext())
    assert seen["during"] == (False, "running")           # пока пересобирается — «готово» не показываем
    assert sc.stems_ready and sc.prepare_state == "ready"
    assert st.get(p.id).scenes[0].stems_ready


def test_prepare_scene_that_dies_midway_stays_not_ready(tmp_path, simple_mp4, monkeypatch):
    from dubl_engine.jobs import NullContext
    from dubl_engine.pipeline import prepare_scene
    st = ProjectStore(tmp_path / "projects")
    p = st.create(simple_mp4, probe(simple_mp4), title="t")
    with st.edit(p.id) as pr:
        pr.scenes.append(Scene(id="sc001", title="s", start=0, end=3, stems_ready=True))

    def boom(*_a, **_k):
        raise RuntimeError("убили")
    monkeypatch.setattr("dubl_engine.audio.separate.separate", boom)
    with pytest.raises(RuntimeError):
        prepare_scene(st, p.id, "sc001", NullContext())
    sc = st.get(p.id).scenes[0]
    assert not sc.stems_ready and sc.prepare_state == "interrupted"        # упало — не «готовится» и не «готово»


# ───────────── удаление проекта ─────────────
def test_delete_project_with_user_file_never_touches_the_file(env, tmp_path, simple_mp4):
    c, store, _jobs = env
    mine = tmp_path / "мой фильм.mp4"
    mine.write_bytes(simple_mp4.read_bytes())
    p = store.create(mine, probe(mine), title="t")
    assert c.delete(f"/projects/{p.id}").status_code == 200
    assert mine.is_file() and not store.exists(p.id)
    assert not any(store.root.iterdir())                    # ни проекта, ни .trash-
    assert c.delete(f"/projects/{p.id}").status_code == 404
    assert c.get("/projects").json() == []


def test_delete_url_project_removes_own_download_only(env, simple_mp4):
    c, store, _jobs = env
    other = data_dir() / "downloads" / "other.mkv"
    other.parent.mkdir(parents=True, exist_ok=True)
    other.write_bytes(b"x" * 100)                           # чужой файл в downloads (другой проект)
    p = store.create("", _media(), title="по ссылке")
    own = store.downloads_root() / p.id
    own.mkdir(parents=True)
    dl = own / "abc.mkv"
    dl.write_bytes(simple_mp4.read_bytes())
    with store.edit(p.id) as pr:
        pr.source_path = str(dl)
    assert c.delete(f"/projects/{p.id}").status_code == 200
    assert not own.exists() and not store.exists(p.id)
    assert other.is_file()


def test_delete_legacy_flat_download_unless_shared(env, simple_mp4):
    """Старый порядок: downloads/<id ролика>.mkv. Стираем, если им не пользуется другой проект."""
    c, store, _jobs = env
    dl = data_dir() / "downloads" / "xyz.mkv"
    dl.parent.mkdir(parents=True, exist_ok=True)
    dl.write_bytes(b"v" * 10)
    a = store.create("", _media(), title="a")
    b = store.create("", _media(), title="b")
    for p in (a, b):
        with store.edit(p.id) as pr:
            pr.source_path = str(dl)
    assert c.delete(f"/projects/{a.id}").status_code == 200
    assert dl.is_file()                                     # b ещё использует
    assert c.delete(f"/projects/{b.id}").status_code == 200
    assert not dl.exists()


def test_delete_never_removes_file_outside_downloads_even_if_named_like_one(env, tmp_path):
    c, store, _jobs = env
    outside = tmp_path / "downloads" / "film.mkv"           # папка с тем же именем, но не в данных программы
    outside.parent.mkdir()
    outside.write_bytes(b"x")
    p = store.create("", _media(), title="t")
    with store.edit(p.id) as pr:
        pr.source_path = str(outside)
    assert c.delete(f"/projects/{p.id}").status_code == 200
    assert outside.is_file()


def test_delete_project_during_download_cancels_and_cleans_up(env, monkeypatch):
    c, store, jobs = env
    started = threading.Event()

    def fake_download(url, dest_dir, *, on_progress=None):
        Path(dest_dir).mkdir(parents=True, exist_ok=True)
        (Path(dest_dir) / "part.mkv.part").write_bytes(b"x" * 1000)
        started.set()
        for _ in range(500):
            on_progress(0.1)                                 # здесь срабатывает отмена задачи
            time.sleep(0.02)
        raise AssertionError("отмена не дошла")
    monkeypatch.setattr("dubl_engine.pipeline.download.download_video", fake_download)
    r = c.post("/projects", json={"url": "https://example.com/v"})
    assert r.status_code == 200, r.text
    pid, job_id = r.json()["id"], r.headers["x-job-id"]
    assert started.wait(10)
    assert c.delete(f"/projects/{pid}").status_code == 200
    assert jobs.get(job_id).state.value == "cancelled"
    assert not store.exists(pid)
    assert not (store.downloads_root() / pid).exists()
    assert not any(p.name.startswith(".trash-") for p in store.root.iterdir())


def test_delete_project_with_finished_job(env):
    c, store, jobs = env
    p = store.create("/x.mp4", _media(), title="t")
    job = jobs.submit("analyze", p.id, [("s", "Шаг")], lambda ctx: None)
    deadline = time.monotonic() + 5
    while jobs.get(job.id).state.value != "done" and time.monotonic() < deadline:
        time.sleep(0.05)
    assert c.delete(f"/projects/{p.id}").status_code == 200
    assert not store.exists(p.id)


def test_delete_project_cors_preflight_allowed(env):
    """Из окна приложения DELETE идёт с другого адреса (tauri://localhost): предварительный запрос должен пройти."""
    c, store, _ = env
    p = store.create("/x.mp4", _media(), title="t")
    r = c.options(f"/projects/{p.id}", headers={"Origin": "tauri://localhost", "Access-Control-Request-Method": "DELETE"})
    assert r.status_code == 200
    assert "DELETE" in r.headers["access-control-allow-methods"] or r.headers["access-control-allow-methods"] == "*"
    assert r.headers["access-control-allow-origin"] == "tauri://localhost"


# ───────────── «Персонажи 1» при «нашла 6 голосов»: числа в данных согласованы ─────────────
def test_speakers_list_matches_lines_exactly():
    """Каждый голос в Analysis.speakers — реально говорящий (есть реплики), счётчики сходятся с репликами. Фронт берёт
    число из одного и того же списка analysis.speakers, поэтому «Персонажи N» и «нашла N голосов» разойтись не могут."""
    from dubl_engine.analysis.diarize import _build_result
    from dubl_engine.models import Line
    lines = [Line(id=f"l{i}", start=i * 2.0, end=i * 2.0 + 1.5, text="t") for i in range(12)]
    labels = [0, 1, 2, 0, 1, 3, 4, 5, 0, 5, None, 1]
    new_lines, speakers, _ = _build_result(lines, labels, None)
    assert len(speakers) == 6
    for sp in speakers:
        mine = [ln for ln in new_lines if ln.speaker_id == sp.id]
        assert sp.line_count == len(mine) > 0
        assert sp.total_speech == pytest.approx(sum(ln.end - ln.start for ln in mine))
    assert {ln.speaker_id for ln in new_lines if ln.speaker_id} == {sp.id for sp in speakers}
