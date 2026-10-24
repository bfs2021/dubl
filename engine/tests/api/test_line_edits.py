"""Правка реплик разбора: смена персонажа, склейка соседних реплик и её отмена, авто-склейка разрезанной фразы."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from dubl_engine.analysis.lines import join_close_lines
from dubl_engine.api import create_app
from dubl_engine.jobs import JobManager
from dubl_engine.models import Analysis, Line, MediaInfo, Scene, SceneLine, Speaker, Take, Word
from dubl_engine.store import ProjectStore


def _line(i, s, e, text, spk, source="whisper"):
    ws = text.split()
    step = (e - s) / max(1, len(ws))
    words = [Word(start=s + k * step, end=s + (k + 1) * step, text=" " + w) for k, w in enumerate(ws)]
    return Line(id=f"l{i:04d}", start=s, end=e, text=text, speaker_id=spk, words=words, source=source)


# как в проекте «Мегамозг»: одна фраза разрезана на l0004 + l0005, l0007 отдана не тому
LINES = [
    _line(3, 18.7, 21.32, "Этот город мал для двух суперзлодеев!", "s2"),
    _line(4, 21.7, 24.5, "О, ты злодей! Это правда!", "s1"),
    _line(5, 24.76, 26.5, "Но совсем не супер!", "s1"),
    _line(6, 26.86, 28.48, "Да? А в чем разница?", "s2"),
    _line(7, 33.62, 34.14, "Спецэффектах!", "s2"),
]


@pytest.fixture
def env(tmp_path):
    store = ProjectStore(tmp_path / "projects")
    media = MediaInfo(path="x", duration=40, container="mkv", width=640, height=360, fps=25, video_codec="h264")
    p = store.create("/x.mkv", media, title="Мегамозг")
    spk = [Speaker(id="s1", label="Мегамозг", name="Мегамозг", color="#4fb8a8"),
           Speaker(id="s2", label="Титан", name="Титан", color="#e57a5f")]
    sl = [SceneLine(line_id=ln.id, speaker_id=ln.speaker_id, start_rel=ln.start, end_rel=ln.end, original=ln.text,
                    text=ln.text) for ln in LINES]
    takes = [Take(id="t001", line_id="l0004", speaker_id="s1", file="takes/t001.wav", duration=3, preroll=0,
                  start_rel=20, selected=True),
             Take(id="t002", line_id="l0005", speaker_id="s1", file="takes/t002.wav", duration=2, preroll=0,
                  start_rel=23, selected=True, chosen_by_user=True),
             Take(id="t003", line_id="l0007", speaker_id="s2", file="takes/t003.wav", duration=1, preroll=0,
                  start_rel=32, selected=True)]
    with store.edit(p.id) as pr:
        pr.analysis = Analysis(audio_stream_index=1, speakers=spk, lines=[ln.model_copy(deep=True) for ln in LINES])
        pr.scenes.append(Scene(id="sc001", title="Бой", start=0, end=40, lines=sl, takes=takes,
                               dubbed_speakers=["s1", "s2"]))
    with TestClient(create_app(store=store, jobs=JobManager())) as c:
        yield c, p.id, store


def _scene(store, pid):
    return store.get(pid).scenes[0]


# ───────────── смена персонажа ─────────────
def test_change_line_speaker_moves_line_and_takes(env):
    c, pid, store = env
    r = c.patch(f"/projects/{pid}/lines/l0007/speaker", json={"speaker_id": "s1"})
    assert r.status_code == 200, r.text
    an = r.json()["analysis"]
    assert next(ln for ln in an["lines"] if ln["id"] == "l0007")["speaker_id"] == "s1"
    counts = {s["id"]: s["line_count"] for s in an["speakers"]}
    assert counts == {"s1": 3, "s2": 2}
    sc = _scene(store, pid)
    assert next(sl for sl in sc.lines if sl.line_id == "l0007").speaker_id == "s1"
    assert next(t for t in sc.takes if t.id == "t003").speaker_id == "s1"


def test_change_line_speaker_to_new_character(env):
    c, pid, store = env
    r = c.patch(f"/projects/{pid}/lines/l0006/speaker", json={"new_speaker_name": " Прислужник "})
    assert r.status_code == 200, r.text
    an = r.json()["analysis"]
    new = next(s for s in an["speakers"] if s["name"] == "Прислужник")
    assert new["id"] == "s3" and new["line_count"] == 1 and new["color"] not in ("#4fb8a8", "#e57a5f")
    assert "s3" in _scene(store, pid).dubbed_speakers          # озвучивали Титана — новую реплику тоже озвучиваем
    # то же имя второй раз — не плодим двойника
    r2 = c.patch(f"/projects/{pid}/lines/l0003/speaker", json={"new_speaker_name": "прислужник"})
    assert [s["id"] for s in r2.json()["analysis"]["speakers"]] == ["s1", "s2", "s3"]


def test_change_line_speaker_errors(env):
    c, pid, _ = env
    assert c.patch(f"/projects/{pid}/lines/l0099/speaker", json={"speaker_id": "s1"}).status_code == 404
    assert c.patch(f"/projects/{pid}/lines/l0007/speaker", json={"speaker_id": "s9"}).status_code == 404
    assert c.patch(f"/projects/{pid}/lines/l0007/speaker", json={}).status_code == 400


# ───────────── склейка ─────────────
def test_merge_lines_and_undo(env):
    c, pid, store = env
    r = c.post(f"/projects/{pid}/lines/merge", json={"line_ids": ["l0005", "l0004"]})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["merge_id"] and body["moved_takes"] == 1
    lines = body["analysis"]["lines"]
    assert [ln["id"] for ln in lines] == ["l0003", "l0004", "l0006", "l0007"]
    m = lines[1]
    assert m["text"] == "О, ты злодей! Это правда! Но совсем не супер!"
    assert (m["start"], m["end"], m["speaker_id"]) == (21.7, 26.5, "s1")
    assert len(m["words"]) == 9
    sc = _scene(store, pid)
    sl = next(x for x in sc.lines if x.line_id == "l0004")
    assert (sl.start_rel, sl.end_rel, sl.text) == (21.7, 26.5, "О, ты злодей! Это правда! Но совсем не супер!")
    assert {t.id: t.line_id for t in sc.takes}["t002"] == "l0004"
    assert sum(t.selected for t in sc.takes if t.line_id == "l0004") == 1        # одна ★ на реплику
    assert next(t for t in sc.takes if t.id == "t002").selected                  # ★ человека сохранилась

    u = c.post(f"/projects/{pid}/lines/merges/{body['merge_id']}/undo")
    assert u.status_code == 200, u.text
    assert [ln["id"] for ln in u.json()["analysis"]["lines"]] == ["l0003", "l0004", "l0005", "l0006", "l0007"]
    sc = _scene(store, pid)
    assert [x.line_id for x in sc.lines] == ["l0003", "l0004", "l0005", "l0006", "l0007"]
    assert next(x for x in sc.lines if x.line_id == "l0005").text == "Но совсем не супер!"
    t = {x.id: x for x in sc.takes}
    assert t["t002"].line_id == "l0005" and t["t002"].selected and t["t001"].selected
    assert c.post(f"/projects/{pid}/lines/merges/{body['merge_id']}/undo").status_code == 404


def test_merge_keeps_user_text(env):
    c, pid, store = env
    c.patch(f"/projects/{pid}/scenes/sc001", json={"lines": [{"line_id": "l0005", "text": "Но не супер!"}]})
    c.post(f"/projects/{pid}/lines/merge", json={"line_ids": ["l0004", "l0005"]})
    sl = next(x for x in _scene(store, pid).lines if x.line_id == "l0004")
    assert sl.text == "О, ты злодей! Это правда! Но не супер!" and sl.translation_source == "user"
    assert sl.original == "О, ты злодей! Это правда! Но совсем не супер!"


def test_merge_rejects_non_adjacent_and_single(env):
    c, pid, _ = env
    assert c.post(f"/projects/{pid}/lines/merge", json={"line_ids": ["l0004", "l0006"]}).status_code == 400
    assert c.post(f"/projects/{pid}/lines/merge", json={"line_ids": ["l0004"]}).status_code == 422
    assert c.post(f"/projects/{pid}/lines/merge", json={"line_ids": ["l0004", "nope"]}).status_code == 404


def test_undo_blocked_after_later_merge(env):
    c, pid, _ = env
    c.patch(f"/projects/{pid}/lines/l0003/speaker", json={"speaker_id": "s1"})
    m1 = c.post(f"/projects/{pid}/lines/merge", json={"line_ids": ["l0004", "l0005"]}).json()["merge_id"]
    m2 = c.post(f"/projects/{pid}/lines/merge", json={"line_ids": ["l0003", "l0004"]}).json()["merge_id"]
    assert c.post(f"/projects/{pid}/lines/merges/{m1}/undo").status_code == 409
    assert c.post(f"/projects/{pid}/lines/merges/{m2}/undo").status_code == 200
    assert c.post(f"/projects/{pid}/lines/merges/{m1}/undo").status_code == 200


# ───────────── склейка разных персонажей запрещена (слепой тест 7, баг 3) ─────────────
def test_merge_rejects_different_speakers(env):
    """Титан + Мегамозг: раньше склеенная реплика молча доставалась Титану, дубль Мегамозга переезжал к неозвучиваемому
    персонажу, и в ролике снова звучал оригинал. Теперь 400 и ничего не меняется."""
    c, pid, store = env
    before = store.get(pid)
    r = c.post(f"/projects/{pid}/lines/merge", json={"line_ids": ["l0003", "l0004"]})
    assert r.status_code == 400
    assert "разных персонажей" in r.json()["detail"] and "смени персонажа" in r.json()["detail"]
    after = store.get(pid)
    assert after.analysis.lines == before.analysis.lines and after.scenes == before.scenes
    assert after.analysis.merged_lines == {}


def test_merge_rejects_unknown_speaker_with_known(env):
    """Реплика без персонажа (None) и реплика персонажа — тоже разные."""
    c, pid, store = env
    with store.edit(pid) as p:
        next(ln for ln in p.analysis.lines if ln.id == "l0005").speaker_id = None
    assert c.post(f"/projects/{pid}/lines/merge", json={"line_ids": ["l0004", "l0005"]}).status_code == 400


# ───────────── «Разделить обратно» без merge_id ─────────────
def test_merge_records_map_and_split_restores(env):
    c, pid, store = env
    r = c.post(f"/projects/{pid}/lines/merge", json={"line_ids": ["l0004", "l0005"]}).json()
    assert r["analysis"]["merged_lines"] == {"l0004": r["merge_id"]}
    s = c.post(f"/projects/{pid}/lines/l0004/split")
    assert s.status_code == 200, s.text
    an = s.json()["analysis"]
    assert [ln["id"] for ln in an["lines"]] == ["l0003", "l0004", "l0005", "l0006", "l0007"]
    assert an["merged_lines"] == {}
    t = {x.id: x for x in _scene(store, pid).takes}
    assert t["t002"].line_id == "l0005" and t["t002"].speaker_id == "s1" and t["t002"].selected
    assert c.post(f"/projects/{pid}/lines/l0004/split").status_code == 404       # больше нечего разделять


def test_split_not_merged_line_is_404(env):
    c, pid, _ = env
    r = c.post(f"/projects/{pid}/lines/l0006/split")
    assert r.status_code == 404 and r.json()["detail"] == "Эта реплика не склеивалась."
    assert c.post(f"/projects/{pid}/lines/l0099/split").status_code == 404


def test_split_chain_of_two_merges_step_by_step(env):
    """Склеили l0004+l0005, потом l0003+(склеенная l0004): «Разделить» на l0003 отменяет вторую склейку, l0004 снова
    помнит первую — второй «Разделить» возвращает всё как было."""
    c, pid, store = env
    c.patch(f"/projects/{pid}/lines/l0003/speaker", json={"speaker_id": "s1"})
    m1 = c.post(f"/projects/{pid}/lines/merge", json={"line_ids": ["l0004", "l0005"]}).json()["merge_id"]
    r2 = c.post(f"/projects/{pid}/lines/merge", json={"line_ids": ["l0003", "l0004"]}).json()
    assert r2["analysis"]["merged_lines"] == {"l0003": r2["merge_id"]}            # поглощённая l0004 — не в карте
    assert [ln["id"] for ln in r2["analysis"]["lines"]] == ["l0003", "l0006", "l0007"]

    s1 = c.post(f"/projects/{pid}/lines/l0003/split").json()["analysis"]
    assert [ln["id"] for ln in s1["lines"]] == ["l0003", "l0004", "l0006", "l0007"]
    assert s1["merged_lines"] == {"l0004": m1}                                     # прежняя запись вернулась
    s2 = c.post(f"/projects/{pid}/lines/l0004/split").json()["analysis"]
    assert [ln["id"] for ln in s2["lines"]] == ["l0003", "l0004", "l0005", "l0006", "l0007"]
    assert s2["merged_lines"] == {}
    sc = _scene(store, pid)
    assert [x.line_id for x in sc.lines] == ["l0003", "l0004", "l0005", "l0006", "l0007"]
    assert {t.id: t.line_id for t in sc.takes} == {"t001": "l0004", "t002": "l0005", "t003": "l0007"}
    assert c.post(f"/projects/{pid}/lines/l0003/split").status_code == 404


def test_split_after_speaker_change_keeps_new_speaker(env):
    """Склеили реплики Мегамозга, потом отдали склеенную Титану — после «Разделить» обе части у Титана."""
    c, pid, store = env
    c.post(f"/projects/{pid}/lines/merge", json={"line_ids": ["l0004", "l0005"]})
    c.patch(f"/projects/{pid}/lines/l0004/speaker", json={"speaker_id": "s2"})
    an = c.post(f"/projects/{pid}/lines/l0004/split").json()["analysis"]
    assert {ln["id"]: ln["speaker_id"] for ln in an["lines"]}["l0005"] == "s2"
    assert {t.id: t.speaker_id for t in _scene(store, pid).takes}["t002"] == "s2"


def test_split_with_lost_snapshot_drops_map_entry(env):
    c, pid, store = env
    r = c.post(f"/projects/{pid}/lines/merge", json={"line_ids": ["l0004", "l0005"]}).json()
    (store.project_dir(pid) / "line_merges" / f"{r['merge_id']}.json").unlink()
    assert c.post(f"/projects/{pid}/lines/l0004/split").status_code == 404
    assert store.get(pid).analysis.merged_lines == {}


def test_old_snapshots_pruned_but_referenced_kept(env):
    """Снимков больше MAX_KEPT_EDITS — старые удаляются, но не тот, на который ссылается карта склеек."""
    import os
    import time

    from dubl_engine.pipeline import lines as L
    c, pid, store = env
    m = c.post(f"/projects/{pid}/lines/merge", json={"line_ids": ["l0004", "l0005"]}).json()["merge_id"]
    d = store.project_dir(pid) / "line_merges"
    old = time.time() - 1000
    os.utime(d / f"{m}.json", (old, old))                       # самый старый
    for i in range(L.MAX_KEPT_EDITS + 3):
        f = d / f"jdead{i:03d}.json"
        f.write_text("{}", encoding="utf-8")
        os.utime(f, (old + 1 + i, old + 1 + i))
    L._prune_snapshots(store, pid, store.get(pid).analysis.merged_lines)
    left = sorted(f.stem for f in d.glob("*.json"))
    assert m in left and len(left) == L.MAX_KEPT_EDITS
    assert c.post(f"/projects/{pid}/lines/l0004/split").status_code == 200


# ───────────── авто-склейка при разборе тоже разделяется ─────────────
def test_auto_join_at_analysis_can_be_split(tmp_path):
    """Как в analyze_project: join_close_groups → store_auto_merges → Analysis(merged_lines=…). Сцена создана после
    разбора (снимка сцены нет) — «Разделить» достаёт части из разбора; дубль остаётся у первой части."""
    from dubl_engine.analysis.lines import join_close_groups, merge_group
    from dubl_engine.pipeline.lines import split_line, store_auto_merges
    from dubl_engine.pipeline.scene import scene_lines_from_analysis

    store = ProjectStore(tmp_path / "projects")
    media = MediaInfo(path="x", duration=40, container="mkv", width=640, height=360, fps=25, video_codec="h264")
    p = store.create("/x.mkv", media, title="Мегамозг")
    spk = [Speaker(id="s1", label="Мегамозг", color="#4fb8a8"), Speaker(id="s2", label="Титан", color="#e57a5f")]
    stale = store.project_dir(p.id) / "line_merges"
    stale.mkdir(parents=True)
    (stale / "jold0001.json").write_text("{}", encoding="utf-8")         # от прошлого разбора — должен уйти

    groups = join_close_groups([ln.model_copy(deep=True) for ln in LINES])
    joined = [g for g in groups if len(g) > 1]
    mapping = store_auto_merges(store, p.id, joined)
    assert list(mapping) == ["l0004"] and not (stale / "jold0001.json").exists()
    with store.edit(p.id) as pr:
        pr.analysis = Analysis(audio_stream_index=1, speakers=spk, lines=[merge_group(g) for g in groups],
                               merged_lines=mapping)
        sc = Scene(id="sc001", title="Бой", start=0, end=40)
        sc.lines = scene_lines_from_analysis(pr, sc)
        sc.takes = [Take(id="t001", line_id="l0004", speaker_id="s1", file="takes/t001.wav", duration=4, preroll=0,
                         start_rel=21, selected=True)]
        pr.scenes.append(sc)
    assert [ln.id for ln in store.get(p.id).analysis.lines] == ["l0003", "l0004", "l0006", "l0007"]

    out = split_line(store, p.id, "l0004")
    assert [ln.id for ln in out.analysis.lines] == ["l0003", "l0004", "l0005", "l0006", "l0007"]
    assert out.analysis.merged_lines == {}
    sl = {x.line_id: x for x in out.scenes[0].lines}
    assert sl["l0004"].text == "О, ты злодей! Это правда!" and sl["l0005"].text == "Но совсем не супер!"
    assert out.scenes[0].takes[0].line_id == "l0004" and out.scenes[0].takes[0].selected


# ───────────── авто-склейка при разборе ─────────────
def test_auto_join_split_phrase_only():
    out = join_close_lines([ln.model_copy(deep=True) for ln in LINES])
    assert [ln.id for ln in out] == ["l0003", "l0004", "l0006", "l0007"]
    assert out[1].text == "О, ты злодей! Это правда! Но совсем не супер!"


def test_auto_join_leaves_subtitles_long_and_overlaps():
    a = _line(1, 0.0, 1.0, "раз", "s1", source="embedded")
    b = _line(2, 1.1, 2.0, "два", "s1", source="embedded")
    assert len(join_close_lines([a, b])) == 2                       # строки субтитров — как в файле
    c = _line(3, 0.0, 6.0, "длинная", "s1")
    d = _line(4, 6.1, 9.0, "ещё", "s1")
    assert len(join_close_lines([c, d])) == 2                       # вместе длиннее 8 с
    e = _line(5, 0.0, 1.0, "а", "s1").model_copy(update={"overlap": True})
    f = _line(6, 1.1, 2.0, "б", "s1")
    assert len(join_close_lines([e, f])) == 2
