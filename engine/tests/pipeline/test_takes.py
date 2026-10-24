"""Дубли: пустые, из файла, разбор по тексту, автовыбор ★, хвост «давай ещё раз»."""
from __future__ import annotations

import io

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from dubl_engine.analysis import takecheck
from dubl_engine.api import create_app
from dubl_engine.audio.io import read_wav, write_wav
from dubl_engine.jobs import JobManager
from dubl_engine.models import Analysis, Line, MediaInfo, Scene, SceneLine, SceneTextMode, Speaker, Word
from dubl_engine.settings import AppSettings
from dubl_engine.store import ProjectStore

SR = 48000
TEXT = "Кто трогал мою колбасу пока я спал"
EN = "Who touched my sausage"


def _wav(x: np.ndarray) -> bytes:
    b = io.BytesIO()
    sf.write(b, x.astype(np.float32), SR, format="WAV")
    return b.getvalue()


def _voice(db: float = -20.0, dur: float = 4.0, parts=((0.2, 1.8),)) -> np.ndarray:
    t = np.arange(int(SR * dur)) / SR
    env = np.zeros_like(t)
    for a, b in parts:
        env[np.logical_and(t >= a, t < b)] = 1.0
    rng = np.random.default_rng(3)
    x = (10 ** (db / 20)) * np.sqrt(2) * np.sin(2 * np.pi * 220 * t) * (0.6 + 0.4 * np.sin(2 * np.pi * 3 * t)) * env
    return x + 10 ** (-70 / 20) * rng.standard_normal(len(t))


@pytest.fixture
def env(tmp_path):
    store = ProjectStore(tmp_path / "projects")
    media = MediaInfo(path="x", duration=20, container="mkv", width=640, height=360, fps=25, video_codec="h264")
    p = store.create("/x.mp4", media, title="Фильм")
    lines = [Line(id="l1", start=0.2, end=1.8, text=EN, speaker_id="s1"),
             Line(id="l2", start=2.2, end=3.8, text="Hello there", speaker_id="s1")]
    sl = [SceneLine(line_id="l1", speaker_id="s1", start_rel=0.2, end_rel=1.8, original=EN, text=TEXT),
          SceneLine(line_id="l2", speaker_id="s1", start_rel=2.2, end_rel=3.8, original="Hello there", text="Hello there")]
    sd = store.scene_dir(p.id, "sc001")
    (sd / "voices").mkdir(parents=True)
    write_wav(sd / "voices.wav", _voice(parts=((0.2, 1.8), (2.2, 3.8))), SR)
    with store.edit(p.id) as pr:
        pr.analysis = Analysis(language="en", audio_stream_index=1, lines=lines,
                               speakers=[Speaker(id="s1", label="Голос 1", color="#fff")])
        pr.scenes.append(Scene(id="sc001", title="s", start=0, end=4, lines=sl, stems_ready=True))
    with TestClient(create_app(store=store, jobs=JobManager())) as c:
        yield c, p.id, store


def _fake_asr(monkeypatch, result):
    """result — функция (wav_path) -> list[Word] | None, или готовый список."""
    monkeypatch.setattr(takecheck, "pick_model", lambda *a, **k: "small")
    monkeypatch.setattr(takecheck, "recognize_words",
                        lambda wav, **k: result(wav) if callable(result) else result)


def _post(c, pid, x, line_id="l1", **data):
    d = {"line_id": line_id, "start_rel": "0", "preroll": "0", **data}
    return c.post(f"/projects/{pid}/scenes/sc001/takes", files={"file": ("a.wav", _wav(x))}, data=d)


def _w(text, step=0.3, start=0.2, gaps=None):
    out, t = [], start
    for i, w in enumerate(text.split()):
        t += (gaps or {}).get(i, 0.0)
        out.append(Word(start=round(t, 3), end=round(t + step * 0.8, 3), text=w))
        t += step
    return out


# ───────────── пустой дубль ─────────────
def test_empty_take_has_no_star_and_never_displaces_a_good_one(env, monkeypatch):
    c, pid, store = env
    _fake_asr(monkeypatch, _w(TEXT))
    good = _post(c, pid, _voice()).json()
    assert good["empty"] is False and good["selected"] is True
    quiet = _post(c, pid, np.zeros(SR * 2)).json()
    assert quiet["empty"] is True and quiet["selected"] is False
    sc = store.get(pid).scenes[0]
    assert [(t.id, t.selected) for t in sc.takes] == [(good["id"], True), (quiet["id"], False)]


def test_only_empty_takes_means_no_star_and_deleting_takes_the_warning_with_it(env):
    c, pid, store = env
    t = _post(c, pid, np.zeros(SR * 2)).json()
    assert t["empty"] and not t["selected"]
    assert not any(x.selected for x in store.get(pid).scenes[0].takes)
    assert c.delete(f"/projects/{pid}/scenes/sc001/takes/{t['id']}").status_code == 200
    assert store.get(pid).scenes[0].takes == []
    assert not store.take_path(pid, "sc001", t["id"]).exists()


def test_empty_take_is_not_mixed(env):
    """Сведение берёт ★ или самый свежий — пустой дубль среди них оказаться не должен, пока есть живой."""
    from dubl_engine.audio.mix import _pick_takes
    c, pid, store = env
    good = _post(c, pid, _voice()).json()
    _post(c, pid, np.zeros(SR * 2))
    sc = store.get(pid).scenes[0]
    picks = _pick_takes(sc.model_copy(update={"dubbed_speakers": ["s1"]}))
    assert good["id"] in {t.id for t, _l, _s in picks}


# ───────────── дубль из файла ─────────────
def test_file_take_is_marked_and_fit_ignores_latency(env, monkeypatch):
    c, pid, store = env
    c.app.state.settings = AppSettings(latency_ms=300.0)
    seen = []
    import dubl_engine.audio.fit as fit

    real = fit.fit_take

    def spy(*a, **k):
        seen.append(k.get("latency_ms"))
        return real(*a, **k)
    monkeypatch.setattr(fit, "fit_take", spy)
    f = _post(c, pid, _voice(), source="file")
    m = _post(c, pid, _voice(), source="mic")
    assert f.status_code == 200 and f.json()["source"] == "file" and m.json()["source"] == "mic"
    assert seen == [0.0, 300.0]                               # задержка микрофона к файлу не относится
    assert f.json()["fit_confidence"] is not None             # на место реплики файл всё равно встаёт


def test_source_defaults_to_mic_and_rejects_garbage(env):
    c, pid, _ = env
    assert _post(c, pid, _voice()).json()["source"] == "mic"
    assert _post(c, pid, _voice(), source="telepathy").status_code == 422


# ───────────── разбор по тексту и автовыбор ★ ─────────────
def test_text_analysis_fields(env, monkeypatch):
    c, pid, store = env
    _fake_asr(monkeypatch, _w(TEXT))
    t = _post(c, pid, _voice()).json()
    assert t["text_match"] == 1.0 and t["complete"] is True and t["recognized_text"] == TEXT
    assert t["tail_cut_s"] is None
    _fake_asr(monkeypatch, _w("Кто трогал мою"))
    t = _post(c, pid, _voice()).json()
    assert t["complete"] is False and 0.3 < t["text_match"] < 0.5


def test_star_goes_to_the_most_complete_take_not_the_newest(env, monkeypatch):
    c, pid, store = env
    answers = iter([_w(TEXT), _w("Кто трогал мою колбасу"), _w("Кто")])
    _fake_asr(monkeypatch, lambda wav: next(answers))
    a, b, d = (_post(c, pid, _voice()).json() for _ in range(3))
    sc = store.get(pid).scenes[0]
    stars = {t.id: t.selected for t in sc.takes}
    assert stars == {a["id"]: True, b["id"]: False, d["id"]: False}        # старый, зато полный


def test_user_choice_beats_auto_pick(env, monkeypatch):
    c, pid, store = env
    answers = iter([_w(TEXT), _w("Кто трогал"), _w(TEXT)])
    _fake_asr(monkeypatch, lambda wav: next(answers))
    a = _post(c, pid, _voice()).json()
    b = _post(c, pid, _voice()).json()
    r = c.patch(f"/projects/{pid}/scenes/sc001/takes/{b['id']}", json={"selected": True})
    assert r.status_code == 200 and r.json()["chosen_by_user"] is True
    n = _post(c, pid, _voice()).json()                                       # новый полный дубль
    stars = {t.id: t.selected for t in store.get(pid).scenes[0].takes}
    assert stars == {a["id"]: False, b["id"]: True, n["id"]: False}          # выбор человека не сбит
    c.patch(f"/projects/{pid}/scenes/sc001/takes/{b['id']}", json={"selected": False})
    assert not any(t.selected for t in store.get(pid).scenes[0].takes)       # снял сам — так и остаётся


def test_deleting_starred_take_passes_star_to_best_remaining(env, monkeypatch):
    c, pid, store = env
    answers = iter([_w(TEXT), _w("Кто трогал мою колбасу"), _w("Кто")])
    _fake_asr(monkeypatch, lambda wav: next(answers))
    a, b, d = (_post(c, pid, _voice()).json() for _ in range(3))
    c.delete(f"/projects/{pid}/scenes/sc001/takes/{a['id']}")
    stars = {t.id: t.selected for t in store.get(pid).scenes[0].takes}
    assert stars == {b["id"]: True, d["id"]: False}


def test_without_asr_new_take_still_gets_star(env, monkeypatch):
    c, pid, _ = env
    monkeypatch.setattr(takecheck, "pick_model", lambda *a, **k: None)
    t = _post(c, pid, _voice()).json()
    assert t["selected"] is True and t["text_match"] is None and t["complete"] is None and t["recognized_text"] is None


def test_unknown_completeness_ranks_between_complete_and_incomplete():
    from datetime import datetime, timedelta

    from dubl_engine.models import Take
    from dubl_engine.pipeline.takes import pick_best
    now = datetime.now().astimezone()

    def mk(i, complete, match, age):
        return Take(id=f"t{i}", line_id="l", speaker_id=None, file="x", duration=1, preroll=0, start_rel=0,
                    complete=complete, text_match=match, created=now - timedelta(minutes=age))
    unknown, incomplete, complete = mk(1, None, None, 0), mk(2, False, 0.5, 1), mk(3, True, 0.9, 2)
    assert pick_best([unknown, incomplete, complete]).id == "t3"
    assert pick_best([unknown, incomplete]).id == "t1"
    assert pick_best([mk(4, False, 0.6, 5), mk(5, False, 0.4, 1)]).id == "t4"
    assert pick_best([mk(6, True, 0.9, 5), mk(7, True, 0.9, 1)]).id == "t7"        # при равенстве — свежий
    dead = mk(8, True, 1.0, 0)
    dead.empty = True
    assert pick_best([dead]) is None


def test_custom_text_is_compared_with_text(env, monkeypatch):
    c, pid, store = env
    with store.edit(pid) as p:
        sc = p.scenes[0]
        sc.text_mode = SceneTextMode.CUSTOM
        sc.lines[0].text, sc.lines[0].translation_source = "Моя собственная реплика про кота", "user"
    _fake_asr(monkeypatch, _w("Моя собственная реплика про кота"))
    t = _post(c, pid, _voice()).json()
    assert t["complete"] is True and t["text_match"] == 1.0
    _fake_asr(monkeypatch, _w(TEXT))                                         # сказал не своё — «неполный»
    assert _post(c, pid, _voice()).json()["complete"] is False


def test_line_in_other_language_is_not_scored(env, monkeypatch):
    c, pid, store = env
    called = []
    _fake_asr(monkeypatch, lambda wav: called.append(1) or _w("Hello there"))
    t = _post(c, pid, _voice(parts=((2.2, 3.8),)), line_id="l2").json()      # у l2 текст = оригинал, а язык перевода ru
    assert t["text_match"] is None and t["complete"] is None and t["recognized_text"] is None and not called
    assert t["selected"] is True                                             # без разбора ★ работает как раньше


def test_whole_scene_take_is_not_scored(env, monkeypatch):
    c, pid, _ = env
    _fake_asr(monkeypatch, _w(TEXT))
    r = c.post(f"/projects/{pid}/scenes/sc001/takes", files={"file": ("a.wav", _wav(_voice()))},
               data={"start_rel": "0", "preroll": "0"})
    t = r.json()
    assert r.status_code == 200 and t["line_id"] is None and t["text_match"] is None and t["selected"] is True


def test_asr_failure_does_not_lose_the_take(env, monkeypatch):
    c, pid, _ = env
    _fake_asr(monkeypatch, None)
    t = _post(c, pid, _voice()).json()
    assert t["complete"] is None and t["selected"] is True


# ───────────── хвост «давай ещё раз» ─────────────
def test_trim_and_untrim_tail(env, monkeypatch):
    c, pid, store = env
    ws = _w(TEXT + " давай ещё раз", gaps={7: 0.5}, step=0.2)
    _fake_asr(monkeypatch, ws)
    x = _voice(dur=4.0, parts=((0.2, 1.6), (2.4, 3.4)))
    t = _post(c, pid, x).json()
    assert t["complete"] is True and t["tail_text"] == "давай ещё раз" and t["tail_cut_s"] is not None
    cut = t["tail_cut_s"]
    base = f"/projects/{pid}/scenes/sc001/takes/{t['id']}"
    full_len = t["duration"]
    r = c.post(base + "/trim_tail")
    assert r.status_code == 200
    tr = r.json()
    assert tr["tail_trimmed"] is True and tr["duration"] == pytest.approx(cut, abs=0.01) and tr["duration"] < full_len
    y, sr = read_wav(store.take_path(pid, "sc001", t["id"]))
    assert len(y) / sr == pytest.approx(cut, abs=0.01) and abs(y[-1]) < 1e-3          # обрезка затухает, без щелчка
    assert c.post(base + "/trim_tail").status_code == 409                               # второй раз — нечего
    r = c.post(base + "/untrim_tail")
    assert r.status_code == 200 and r.json()["tail_trimmed"] is False
    assert r.json()["duration"] == pytest.approx(full_len, abs=0.01)
    assert c.post(base + "/untrim_tail").status_code == 409
    assert c.post(base.replace(t["id"], "t999") + "/trim_tail").status_code == 404
    c.post(base + "/trim_tail")
    assert c.delete(base).status_code == 200
    assert not list(store.takes_dir(pid, "sc001").glob(f"{t['id']}*"))                # полный файл не остаётся


def test_trim_without_tail_is_409(env, monkeypatch):
    c, pid, _ = env
    _fake_asr(monkeypatch, _w(TEXT))
    t = _post(c, pid, _voice()).json()
    assert c.post(f"/projects/{pid}/scenes/sc001/takes/{t['id']}/trim_tail").status_code == 409


def test_settings_put_cannot_overwrite_mic_profile(env):
    c, pid, _ = env
    s = c.get("/settings").json()
    s["mic_profile"] = {"noise_floor_db": -10.0}
    assert c.put("/settings", json=s).json()["mic_profile"] is None


def test_take_gets_noise_and_clip_measures(env, monkeypatch):
    """Замеры звука пишутся в дубль: шум в паузах, речь над шумом, перегруз (для предупреждений и шумодава)."""
    c, pid, _store = env
    _fake_asr(monkeypatch, None)
    clean = _post(c, pid, _voice(db=-20.0)).json()
    assert clean["noise_floor_db"] is not None and clean["noise_floor_db"] < -50
    assert clean["snr_db"] is not None and clean["snr_db"] > 25
    assert clean["clip_pct"] == 0
    loud = np.clip(_voice(db=0.0) * 3.0, -1.0, 1.0)              # срезанные пики
    assert _post(c, pid, loud).json()["clip_pct"] > 0.05


def test_delete_take_drops_stale_track_gain(env, monkeypatch):
    """Удалили единственный дубль персонажа — его ползунок не держит старое усиление автопилота."""
    c, pid, store = env
    _fake_asr(monkeypatch, None)
    tid = _post(c, pid, _voice()).json()["id"]
    with store.edit(pid) as p:
        p.scenes[0].mix.tracks["s1"] = p.scenes[0].mix.tracks.get("s1") or type(p.scenes[0].mix.background)()
        p.scenes[0].mix.tracks["s1"].gain_db = -7.8
    assert c.delete(f"/projects/{pid}/scenes/sc001/takes/{tid}").status_code < 300
    tr = store.get(pid).scenes[0].mix.tracks.get("s1")
    assert tr is None or tr.gain_db == 0.0
