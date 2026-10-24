from __future__ import annotations

import threading

import pytest

from dubl_engine.models import MediaInfo, Scene, Take
from dubl_engine.store import NotFoundError, ProjectStore


def _media():
    return MediaInfo(path="x", duration=10, container="mkv", width=640, height=360, fps=25, video_codec="h264")


def test_ids_are_validated(tmp_path):
    st = ProjectStore(tmp_path)
    for bad in ("../x", "a/b", "..", "", "a\\b", "x" * 200, ".hidden"):
        with pytest.raises(NotFoundError):
            st.project_dir(bad)
    p = st.create("/tmp/v.mp4", _media(), title="Привет, мир: «тест»")
    with pytest.raises(NotFoundError):
        st.scene_dir(p.id, "../../etc")
    with pytest.raises(NotFoundError):
        st.take_path(p.id, "sc001", "../t")


def test_title_with_cyrillic_and_list_uses_summary(tmp_path):
    st = ProjectStore(tmp_path)
    p = st.create("/tmp/v.mp4", _media(), title="Привет, мир")
    assert p.id.startswith("privet-mir-")
    (st.project_dir(p.id) / "project.json").write_text("{битый json", encoding="utf-8")     # list() его не читает
    [s] = st.list()
    assert s.id == p.id and s.title == "Привет, мир" and s.duration == 10


def test_list_falls_back_to_full_file_and_skips_broken(tmp_path):
    st = ProjectStore(tmp_path)
    a = st.create("/tmp/v.mp4", _media(), title="A")
    (st.project_dir(a.id) / "summary.json").unlink()
    b = st.create("/tmp/v.mp4", _media(), title="B")
    (st.project_dir(b.id) / "summary.json").unlink()
    (st.project_dir(b.id) / "project.json").write_text("{", encoding="utf-8")
    assert [s.id for s in st.list()] == [a.id]


def test_concurrent_edits_do_not_lose_updates(tmp_path):
    st = ProjectStore(tmp_path)
    p = st.create("/tmp/v.mp4", _media())
    with st.edit(p.id) as pr:
        pr.scenes.append(Scene(id="sc001", title="s", start=0, end=5))

    def add(i):
        with st.edit(p.id) as pr:
            sc = pr.scenes[0]
            sc.takes.append(Take(id=f"t{i:03d}", line_id=None, speaker_id=None, file="x", duration=1, preroll=0,
                                 start_rel=0))
    th = [threading.Thread(target=add, args=(i,)) for i in range(20)]
    [t.start() for t in th]
    [t.join() for t in th]
    assert len(st.get(p.id).scenes[0].takes) == 20


def test_failed_edit_does_not_write(tmp_path):
    st = ProjectStore(tmp_path)
    p = st.create("/tmp/v.mp4", _media())
    with pytest.raises(RuntimeError), st.edit(p.id) as pr:
        pr.title = "изменили"
        raise RuntimeError("упало")
    assert st.get(p.id).title == p.title
