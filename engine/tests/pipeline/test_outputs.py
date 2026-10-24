"""Рендер не перезаписывает чужие файлы: « (2)», « (3)»; overwrite=True заменяет; итоговый путь возвращается."""
from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from dubl_engine.api import create_app
from dubl_engine.jobs import JobManager, NullContext
from dubl_engine.models import RenderSettings
from dubl_engine.pipeline.join import render_project
from dubl_engine.pipeline.outputs import numbered, release_output, reserve_output
from dubl_engine.pipeline.scene import render_scene

OLD = "старый ролик".encode()


def test_numbered_names(tmp_path):
    p = tmp_path / "Сцена.mp4"
    assert numbered(p, 1) == p
    assert numbered(p, 2).name == "Сцена (2).mp4" and numbered(p, 3).name == "Сцена (3).mp4"


def test_reserve_output_skips_existing_and_holds_the_name(tmp_path):
    p = tmp_path / "новая" / "Сцена.mp4"
    a = reserve_output(p)
    assert a == p and a.exists()                              # имя занято заглушкой: параллельная задача возьмёт следующее
    b = reserve_output(p)
    c = reserve_output(p)
    assert b.name == "Сцена (2).mp4" and c.name == "Сцена (3).mp4"
    assert reserve_output(p, overwrite=True) == p              # заменить — тот же путь
    release_output(c, False)
    assert not c.exists() and a.exists() and b.exists()        # убрали только свою пустую заглушку
    a.write_bytes(b"x")
    release_output(a, False)
    assert a.exists()                                          # чужой готовый файл не трогаем


def _rs(tmp_path, **kw):
    return RenderSettings(height=240, output_path=str(tmp_path / "Ролик.mp4"), **kw)


def test_render_scene_twice_makes_copy_and_overwrite_replaces(ready_project, tmp_path):
    store, pid, ids = ready_project
    first = render_scene(store, pid, ids[0], _rs(tmp_path), NullContext())
    assert first == tmp_path / "Ролик.mp4" and first.stat().st_size > 1000
    before = first.read_bytes()
    second = render_scene(store, pid, ids[0], _rs(tmp_path), NullContext())
    third = render_scene(store, pid, ids[1], _rs(tmp_path), NullContext())
    assert second.name == "Ролик (2).mp4" and third.name == "Ролик (3).mp4"
    assert second.stat().st_size > 1000 and third.stat().st_size > 1000
    assert first.read_bytes() == before                       # первый файл не тронут
    again = render_scene(store, pid, ids[1], _rs(tmp_path, overwrite=True), NullContext())
    assert again == first and first.read_bytes() != before     # заменили по просьбе
    assert not [p for p in tmp_path.iterdir() if p.name.startswith(".")]      # временных файлов не осталось


def test_failed_render_leaves_no_empty_file(ready_project, tmp_path, monkeypatch):
    store, pid, ids = ready_project
    (tmp_path / "Ролик.mp4").write_bytes(OLD)

    def boom(*a, **k):
        raise RuntimeError("сломалось")
    monkeypatch.setattr("dubl_engine.audio.render.render_video", boom)
    with pytest.raises(RuntimeError):
        render_scene(store, pid, ids[0], _rs(tmp_path), NullContext())
    assert sorted(p.name for p in tmp_path.iterdir() if p.is_file()) == ["Ролик.mp4"]       # заглушки «(2)» нет
    assert (tmp_path / "Ролик.mp4").read_bytes() == OLD
    with pytest.raises(RuntimeError):                         # и при overwrite старый ролик цел, пока новый не собран
        render_scene(store, pid, ids[0], _rs(tmp_path, overwrite=True), NullContext())
    assert (tmp_path / "Ролик.mp4").read_bytes() == OLD


def test_render_project_does_not_overwrite(ready_project, tmp_path):
    store, pid, ids = ready_project
    a = render_project(store, pid, ids, _rs(tmp_path), NullContext())
    b = render_project(store, pid, ids, _rs(tmp_path), NullContext())
    c = render_project(store, pid, [ids[0]], _rs(tmp_path, overwrite=True), NullContext())
    assert a.name == "Ролик.mp4" and b.name == "Ролик (2).mp4" and c == a
    assert b.stat().st_size > 1000


def _wait(c, jid, timeout=120):
    t0 = time.time()
    while time.time() - t0 < timeout:
        j = c.get(f"/jobs/{jid}").json()
        if j["state"] in ("done", "failed", "cancelled"):
            return j
        time.sleep(0.2)
    raise AssertionError("задача не закончилась")


def test_render_route_returns_final_path(ready_project, tmp_path):
    store, pid, ids = ready_project
    with TestClient(create_app(store=store, jobs=JobManager())) as c:
        body = {"height": 240, "output_path": "итог.mp4"}
        paths = []
        for i in range(2):
            job = c.post(f"/projects/{pid}/scenes/{ids[0]}/render", json=body).json()
            done = _wait(c, job["id"])
            assert done["state"] == "done", done
            paths.append(done["result"]["path"])
            assert c.get(done["result"]["url"]).status_code == 200
        assert paths[0].endswith("/итог.mp4") and paths[1].endswith("/итог (2).mp4")
        job = c.post(f"/projects/{pid}/render", json={"scene_ids": ids, "settings": {**body, "overwrite": True}}).json()
        done = _wait(c, job["id"])
        assert done["state"] == "done" and done["result"]["path"] == paths[0]
