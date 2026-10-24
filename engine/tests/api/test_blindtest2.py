"""Баги слепого теста №2 на уровне API: слияние/отмена голосов, переименование проекта, пустая запись, перевод ИИ vs правки."""
from __future__ import annotations

import io

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from dubl_engine.api import create_app
from dubl_engine.audio.io import write_wav
from dubl_engine.jobs import JobManager
from dubl_engine.models import (
    Analysis,
    Line,
    MediaInfo,
    MixSettings,
    Scene,
    SceneLine,
    Speaker,
    Take,
    TrackSettings,
)
from dubl_engine.store import ProjectStore

SR = 48000


@pytest.fixture
def env(tmp_path):
    store = ProjectStore(tmp_path / "projects")
    media = MediaInfo(path="x", duration=20, container="mkv", width=640, height=360, fps=25, video_codec="h264")
    p = store.create("/x.mp4", media, title="Фильм")
    spk = [Speaker(id=f"s{i}", label=f"Голос {i}", color="#fff") for i in (1, 2, 3)]
    lines = [Line(id=f"l{i}", start=i * 2, end=i * 2 + 1.5, text=f"t{i}", speaker_id=f"s{(i % 3) + 1}") for i in range(6)]
    for s in spk:
        mine = [ln for ln in lines if ln.speaker_id == s.id]
        s.line_count, s.total_speech = len(mine), 1.5 * len(mine)
    spk[1].name = "Том"
    spk[1].label = "Том"
    sl = [SceneLine(line_id=ln.id, speaker_id=ln.speaker_id, start_rel=ln.start, end_rel=ln.end, original=ln.text, text=ln.text)
          for ln in lines]
    take = Take(id="t001", line_id="l1", speaker_id="s2", file="takes/t001.wav", duration=1, preroll=0, start_rel=0)
    with store.edit(p.id) as pr:
        pr.analysis = Analysis(audio_stream_index=1, speakers=spk, lines=lines,
                               merge_suggestions=[("s1", "s2", 0.53), ("s2", "s3", 0.51), ("s1", "s3", 0.4)])
        pr.scenes.append(Scene(id="sc001", title="s", start=0, end=12, lines=sl, takes=[take], dubbed_speakers=["s2", "s3"],
                               mix=MixSettings(tracks={"s2": TrackSettings(gain_db=-3.0), "s3": TrackSettings(room="cave")}),
                               stems_ready=True))
    with TestClient(create_app(store=store, jobs=JobManager())) as c:
        yield c, p.id, store


# ───────────── переименование проекта ─────────────
def test_rename_project(env):
    c, pid, store = env
    r = c.patch(f"/projects/{pid}", json={"title": "  Новое имя  "})
    assert r.status_code == 200 and r.json()["title"] == "Новое имя" and r.json()["id"] == pid
    assert store.get(pid).title == "Новое имя"
    assert next(s for s in c.get("/projects").json() if s["id"] == pid)["title"] == "Новое имя"
    assert c.patch(f"/projects/{pid}", json={"title": "   "}).status_code == 400
    assert c.patch("/projects/nope", json={"title": "x"}).status_code == 404


# ───────────── слияние голосов и отмена ─────────────
def _state(project: dict):
    an = project["analysis"]
    return ([(s["id"], s["label"], s["name"], s["line_count"], s["total_speech"]) for s in an["speakers"]],
            [(ln["id"], ln["speaker_id"]) for ln in an["lines"]],
            [tuple(m) for m in an["merge_suggestions"]],
            [(sl["line_id"], sl["speaker_id"]) for sc in project["scenes"] for sl in sc["lines"]],
            [(t["id"], t["speaker_id"]) for sc in project["scenes"] for t in sc["takes"]],
            [sc["dubbed_speakers"] for sc in project["scenes"]],
            [sc["mix"]["tracks"] for sc in project["scenes"]])


def test_merge_returns_project_and_merge_id_and_undo_restores_everything(env):
    c, pid, _ = env
    before = c.get(f"/projects/{pid}").json()
    r = c.post(f"/projects/{pid}/speakers/merge", json={"a": "s1", "b": "s2"})
    assert r.status_code == 200
    body = r.json()
    assert body["merge_id"] and body["id"] == pid                         # проект + merge_id
    assert [s["id"] for s in body["analysis"]["speakers"]] == ["s1", "s3"]
    assert all(ln["speaker_id"] != "s2" for ln in body["analysis"]["lines"])
    assert body["scenes"][0]["takes"][0]["speaker_id"] == "s1" and "s2" not in body["scenes"][0]["mix"]["tracks"]
    assert body["analysis"]["speakers"][0]["line_count"] == 4              # 2 + 2

    u = c.post(f"/projects/{pid}/speakers/merges/{body['merge_id']}/undo")
    assert u.status_code == 200
    assert _state(u.json()) == _state(before)                              # голоса, реплики, дубли, имена — как было
    assert next(s for s in u.json()["analysis"]["speakers"] if s["id"] == "s2")["name"] == "Том"
    assert _state(c.get(f"/projects/{pid}").json()) == _state(before)      # и на диске
    # повторная отмена — 404
    assert c.post(f"/projects/{pid}/speakers/merges/{body['merge_id']}/undo").status_code == 404


def test_undo_unknown_merge_is_404(env):
    c, pid, _ = env
    assert c.post(f"/projects/{pid}/speakers/merges/nope/undo").status_code == 404
    assert c.post(f"/projects/{pid}/speakers/merges/..%2Fx/undo").status_code in (404, 422)


def test_merge_any_two_voices_and_chain_undo_in_reverse_order(env):
    c, pid, _ = env
    s0 = _state(c.get(f"/projects/{pid}").json())
    m1 = c.post(f"/projects/{pid}/speakers/merge", json={"a": "s3", "b": "s1"}).json()
    s1 = _state(m1)
    m2 = c.post(f"/projects/{pid}/speakers/merge", json={"a": "s2", "b": "s3"}).json()
    assert [s["id"] for s in m2["analysis"]["speakers"]] == ["s2"]
    # более раннее слияние при живом более позднем отменить нельзя: s3 уже влит в s2
    r = c.post(f"/projects/{pid}/speakers/merges/{m1['merge_id']}/undo")
    assert r.status_code == 409 and r.json()["detail"]
    assert _state(c.post(f"/projects/{pid}/speakers/merges/{m2['merge_id']}/undo").json()) == s1
    assert _state(c.post(f"/projects/{pid}/speakers/merges/{m1['merge_id']}/undo").json()) == s0


def test_merge_validation(env):
    c, pid, _ = env
    assert c.post(f"/projects/{pid}/speakers/merge", json={"a": "s1", "b": "s1"}).status_code == 400
    assert c.post(f"/projects/{pid}/speakers/merge", json={"a": "s1", "b": "zz"}).status_code == 404


def test_same_name_for_two_voices_does_not_merge(env):
    """Одно имя двум голосам — движок ничего не сливает (предлагает UI)."""
    c, pid, _ = env
    c.patch(f"/projects/{pid}/speakers/s1", json={"name": "Том"})
    p = c.patch(f"/projects/{pid}/speakers/s3", json={"name": "Том"}).json()
    assert [s["id"] for s in p["analysis"]["speakers"]] == ["s1", "s2", "s3"]
    assert [s["name"] for s in p["analysis"]["speakers"]] == ["Том", "Том", "Том"]


# ───────────── перевод: источник, перевод ИИ не перетирает пользователя ─────────────
def test_patch_scene_lines_marks_source(env):
    c, pid, _ = env
    url = f"/projects/{pid}/scenes/sc001"
    r = c.patch(url, json={"lines": [{"line_id": "l0", "text": "Мой текст"}]}).json()
    assert next(l for l in r["lines"] if l["line_id"] == "l0")["translation_source"] == "user"
    r = c.patch(url, json={"lines": [{"line_id": "l1", "text": "Из файла", "source": "file"}]}).json()
    assert next(l for l in r["lines"] if l["line_id"] == "l1")["translation_source"] == "file"
    r = c.patch(url, json={"lines": [{"line_id": "l0", "text": "t0"}]}).json()      # вернул оригинал — не перевод
    assert next(l for l in r["lines"] if l["line_id"] == "l0")["translation_source"] is None
    r = c.patch(url, json={"lines": [{"line_id": "l2", "text": "Тот же", "source": "llm"}]}).json()
    assert next(l for l in r["lines"] if l["line_id"] == "l2")["translation_source"] == "llm"


def test_llm_translate_does_not_overwrite_user_text_but_overwrites_file(env, monkeypatch):
    c, pid, store = env
    with store.edit(pid) as p:
        sl = {l.line_id: l for l in p.scenes[0].lines}
        sl["l0"].text, sl["l0"].translation_source = "Моё", "user"
        sl["l1"].text, sl["l1"].translation_source = "Из файла", "file"

    seen = []

    class Fake:
        id = "fake"

        async def complete(self, system, user, **kw):
            seen.append(user)
            import json
            ids = [int(x) for x in __import__("re").findall(r'"id":\s*(\d+)', user)]
            return json.dumps({"translations": [{"id": i, "text": f"ИИ{i}"} for i in ids]})
    monkeypatch.setattr("dubl_engine.api.routes.llm.provider_for", lambda task, settings: Fake())
    r = c.post("/llm/translate", json={"project_id": pid, "scene_id": "sc001"})
    assert r.status_code == 200, r.text
    by = {t["line_id"]: t for t in r.json()["translations"]}
    assert by["l0"] == {"line_id": "l0", "text": "Моё", "source": "user"}
    assert by["l1"]["source"] == "llm" and by["l1"]["text"].startswith("ИИ")
    assert "t0" not in "".join(seen)                                        # пользовательскую реплику ИИ не отправляли
    sc = store.get(pid).scenes[0]
    got = {l.line_id: (l.text, l.translation_source) for l in sc.lines}
    assert got["l0"] == ("Моё", "user") and got["l1"][1] == "llm" and got["l1"][0] != "Из файла"


# ───────────── пустая запись ─────────────
def _wav(x: np.ndarray) -> bytes:
    b = io.BytesIO()
    sf.write(b, x.astype(np.float32), SR, format="WAV")
    return b.getvalue()


def _prep_scene(store, pid):
    sd = store.scene_dir(pid, "sc001")
    sd.mkdir(parents=True, exist_ok=True)
    t = np.arange(SR * 12) / SR
    voice = (0.2 * np.sin(2 * np.pi * 300 * t) * (((t % 4) < 1.5))).astype(np.float32)
    write_wav(sd / "voices.wav", voice, SR)
    (sd / "voices").mkdir(exist_ok=True)


def _speech(level_db: float, dur=4.0, noise_db=None) -> np.ndarray:
    t = np.arange(int(SR * dur)) / SR
    env = ((t % 2) < 1.0).astype(np.float64)
    x = 10 ** (level_db / 20) * np.sin(2 * np.pi * 220 * t) * np.sqrt(2) * env
    if noise_db is not None:
        x = x + 10 ** (noise_db / 20) * np.random.default_rng(1).standard_normal(len(t))
    return x


def test_empty_take_is_saved_as_empty_and_never_starred(env):
    """Почти тишина не теряется молча и не попадает в ролик: дубль сохраняется с empty=true, без ★, без подгонки."""
    c, pid, store = env
    _prep_scene(store, pid)
    url = f"/projects/{pid}/scenes/sc001/takes"
    ids = []
    for name, x in {"цифровая тишина": np.zeros(SR * 2),
                    "шум −65 дБ": 10 ** (-65 / 20) * np.random.default_rng(2).standard_normal(SR * 3),
                    "речь −62 дБ RMS": _speech(-62, noise_db=-80)}.items():
        r = c.post(url, files={"file": (f"{name}.wav", _wav(x))}, data={"line_id": "l1", "start_rel": "0", "preroll": "0"})
        assert r.status_code == 200, (name, r.text)
        t = r.json()
        assert t["empty"] is True and t["selected"] is False and t["chosen_by_user"] is False, (name, t)
        assert t["fit_confidence"] is None and t["text_match"] is None and t["complete"] is None
        ids.append(t["id"])
    sc = store.get(pid).scenes[0]
    by = {t.id: t for t in sc.takes}
    assert by["t001"].selected is False or by["t001"].empty is False        # прежний дубль не тронут пустыми
    # пустой дубль ★ не взять и руками (409), а удалить — предупреждение уходит вместе с ним
    r = c.patch(f"/projects/{pid}/scenes/sc001/takes/{ids[0]}", json={"selected": True})
    assert r.status_code == 409 and "почти нет звука" in r.json()["detail"]
    assert c.delete(f"/projects/{pid}/scenes/sc001/takes/{ids[0]}").status_code == 200
    assert ids[0] not in [t.id for t in store.get(pid).scenes[0].takes]
    assert not store.take_path(pid, "sc001", ids[0]).exists()


def test_quiet_but_real_speech_passes(env):
    c, pid, store = env
    _prep_scene(store, pid)
    url = f"/projects/{pid}/scenes/sc001/takes"
    for level in (-30, -40, -46):                                           # тихий голос, дальний микрофон
        r = c.post(url, files={"file": ("a.wav", _wav(_speech(level, noise_db=-75)))},
                   data={"line_id": "l1", "start_rel": "0", "preroll": "0"})
        assert r.status_code == 200, (level, r.text)


def test_no_match_is_not_reported_as_in_lips():
    """Совпадение 0 (речь дубля не нашли в окне реплики) — не «попал в губы», а отдельная проблема no_match."""
    from dubl_engine.llm.director import rule_based_feedback, take_report
    line = SceneLine(line_id="l1", speaker_id=None, start_rel=0, end_rel=2, original="a", text="б")
    t = Take(id="t1", line_id="l1", speaker_id=None, file="x", duration=2, preroll=0, start_rel=0,
             offset_ms=0.0, stretch=1.0, fit_confidence=0.0, peak_db=-12, loudness_lufs=-20)
    rep = take_report(t, line)
    assert "no_match" in rep["problems"] and rep["ok"] is False
    assert rep["timing"] == "ok" and rep["fit_unsure"] is True
    assert "не нашёл твой голос" in rule_based_feedback(rep)
    good = take_report(t.model_copy(update={"fit_confidence": 0.8}), line)
    assert "no_match" not in good["problems"] and good["ok"] is True
