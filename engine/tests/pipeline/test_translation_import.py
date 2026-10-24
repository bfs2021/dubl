"""Перевод из файла: русские субтитры (дорожка MKV или .srt рядом) → translation реплик → text сцены → субтитры дубля."""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from dubl_engine.analysis.subtitles import external_subtitle_files, match_translation, parse_subtitle_file
from dubl_engine.jobs import NullContext
from dubl_engine.media.probe import probe
from dubl_engine.models import Analysis, Line, Scene, SceneLine, SubtitleText
from dubl_engine.pipeline import analyze_project
from dubl_engine.pipeline.scene import scene_lines_from_analysis, subtitle_lines
from dubl_engine.settings import AppSettings
from dubl_engine.store import ProjectStore


def _ff(*args):
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *map(str, args)], check=True)


@pytest.fixture
def store(tmp_path):
    return ProjectStore(tmp_path / "projects")


@pytest.fixture
def no_speech(monkeypatch):
    monkeypatch.setattr("dubl_engine.analysis.transcribe.transcribe", lambda *a, **k: ([], "en"))

    def fake_diarize(wav, lines, **_kw):
        from dubl_engine.analysis.diarize import _build_result
        return _build_result(sorted(lines, key=lambda x: x.start), [0] * len(lines), None)
    monkeypatch.setattr("dubl_engine.analysis.diarize.diarize", fake_diarize)


@pytest.fixture(scope="module")
def two_subs_mkv(tmp_path_factory) -> Path:
    """Маленький MKV: звук eng, субтитры eng и rus (русские сдвинуты на 0,2 с и режут вторую реплику на две строки)."""
    d = tmp_path_factory.mktemp("twosubs")
    en = d / "en.srt"
    en.write_text("1\n00:00:01,000 --> 00:00:03,000\nHello there, how are you?\n\n"
                  "2\n00:00:05,000 --> 00:00:08,000\nI am fine. Thank you very much.\n", encoding="utf-8")
    ru = d / "ru.srt"
    ru.write_text("1\n00:00:01,200 --> 00:00:03,100\nПривет, как дела?\n\n"
                  "2\n00:00:05,100 --> 00:00:06,500\nУ меня всё хорошо.\n\n"
                  "3\n00:00:06,600 --> 00:00:08,200\nБольшое спасибо.\n", encoding="utf-8")
    out = d / "two.mkv"
    _ff("-f", "lavfi", "-i", "testsrc2=s=320x180:r=25:d=10", "-f", "lavfi", "-i", "sine=f=500:d=10",
        "-i", en, "-i", ru, "-map", "0:v", "-map", "1:a", "-map", "2:s", "-map", "3:s",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-c:s", "srt",
        "-metadata:s:a:0", "language=eng", "-metadata:s:s:0", "language=eng", "-metadata:s:s:1", "language=rus", out)
    return out


def test_match_translation_by_overlap():
    lines = [Line(id="l1", start=1.0, end=3.0, text="Hello"), Line(id="l2", start=5.0, end=8.0, text="Bye"),
             Line(id="l3", start=9.0, end=9.5, text="Silent")]
    tr = [Line(id="a", start=1.2, end=3.1, text="Привет"), Line(id="b", start=5.1, end=6.5, text="Пока."),
          Line(id="c", start=6.6, end=8.2, text="До встречи.")]
    out = match_translation(lines, tr)
    assert [(ln.translation, ln.translation_source) for ln in out] == \
        [("Привет", "file"), ("Пока. До встречи.", "file"), (None, None)]


def test_analyze_imports_russian_track_as_translation(store, two_subs_mkv, no_speech):
    p = store.create(two_subs_mkv, probe(two_subs_mkv), title="t")
    an = analyze_project(store, p.id, NullContext(), settings=AppSettings())
    assert [ln.text for ln in an.lines] == ["Hello there, how are you?", "I am fine. Thank you very much."]   # оригинал — eng
    assert [ln.translation for ln in an.lines] == ["Привет, как дела?", "У меня всё хорошо. Большое спасибо."]
    assert {ln.translation_source for ln in an.lines} == {"file"}


def test_translation_goes_into_scene_lines_with_source(store, two_subs_mkv, no_speech):
    p = store.create(two_subs_mkv, probe(two_subs_mkv), title="t")
    analyze_project(store, p.id, NullContext(), settings=AppSettings())
    project = store.get(p.id)
    sc = Scene(id="sc001", title="s", start=0.0, end=9.0)
    lines = scene_lines_from_analysis(project, sc)
    assert [(l.original, l.text, l.translation_source) for l in lines] == [
        ("Hello there, how are you?", "Привет, как дела?", "file"),
        ("I am fine. Thank you very much.", "У меня всё хорошо. Большое спасибо.", "file")]


def test_user_edit_and_llm_translation_survive_reprepare_but_file_is_refreshed():
    an = Analysis(audio_stream_index=0, lines=[
        Line(id="l1", start=1, end=3, text="One", translation="Один (файл)", translation_source="file"),
        Line(id="l2", start=4, end=6, text="Two", translation="Два (файл)", translation_source="file"),
        Line(id="l3", start=7, end=9, text="Three", translation="Три (файл)", translation_source="file")])

    class P:
        analysis = an
    old = Scene(id="sc001", title="s", start=0, end=10, lines=[
        SceneLine(line_id="l1", speaker_id=None, start_rel=1, end_rel=3, original="One", text="Раз!", translation_source="user"),
        SceneLine(line_id="l2", speaker_id=None, start_rel=4, end_rel=6, original="Two", text="Два (ИИ)", translation_source="llm"),
        SceneLine(line_id="l3", speaker_id=None, start_rel=7, end_rel=9, original="Three", text="Три (старый файл)", translation_source="file")])
    got = {l.line_id: (l.text, l.translation_source) for l in scene_lines_from_analysis(P, old)}
    assert got == {"l1": ("Раз!", "user"), "l2": ("Два (ИИ)", "llm"), "l3": ("Три (файл)", "file")}


def test_external_srt_next_to_video(store, simple_mp4, tmp_path, no_speech):
    """Русский .srt рядом с видео (фильм.ru.srt и фильм.srt с русским текстом) подхватывается как перевод."""
    video = tmp_path / "movie" / "кино.mp4"
    video.parent.mkdir()
    video.write_bytes(simple_mp4.read_bytes())
    (video.parent / "кино.en.srt").write_text("1\n00:00:02,000 --> 00:00:04,000\nWho touched my sausage?\n", encoding="utf-8")
    (video.parent / "кино.ru.srt").write_text("1\n00:00:02,100 --> 00:00:04,000\nКто трогал мою колбасу?\n", encoding="utf-8")
    assert [f.name for f in external_subtitle_files(video, "ru")] == ["кино.ru.srt"]
    (video.parent / "кино.ru.srt").rename(video.parent / "кино.srt")                       # без пометки — по тексту
    assert [f.name for f in external_subtitle_files(video, "ru")] == ["кино.srt"]
    assert [f.name for f in external_subtitle_files(video, "en")] == ["кино.en.srt"]

    p = store.create(video, probe(video), title="t")
    # whisper-заглушка находит одну реплику — перевод ложится на неё по времени
    import dubl_engine.analysis.transcribe as tr
    tr_lines = [Line(id="l0001", start=2.0, end=4.0, text="Who touched my sausage?")]
    import pytest as _pt
    mp = _pt.MonkeyPatch()
    mp.setattr(tr, "transcribe", lambda *a, **k: (tr_lines, "en"))
    try:
        an = analyze_project(store, p.id, NullContext(), settings=AppSettings())
    finally:
        mp.undo()
    assert [(ln.text, ln.translation, ln.translation_source) for ln in an.lines] == \
        [("Who touched my sausage?", "Кто трогал мою колбасу?", "file")]


def test_no_translation_track_is_fine(store, simple_mp4, no_speech):
    p = store.create(simple_mp4, probe(simple_mp4), title="t")
    an = analyze_project(store, p.id, NullContext(), settings=AppSettings())
    assert all(ln.translation is None and ln.translation_source is None for ln in an.lines)


def test_target_lang_setting_picks_other_language(store, two_subs_mkv, no_speech):
    """Язык перевода берётся из settings.target_lang: для «en» дорожка rus не нужна."""
    p = store.create(two_subs_mkv, probe(two_subs_mkv), title="t")
    an = analyze_project(store, p.id, NullContext(), settings=AppSettings(target_lang="de"))
    assert all(ln.translation is None for ln in an.lines)


# ───────────── субтитры дубля: перевод → русский из файла → оригинал ─────────────
def test_subtitle_lines_fall_back_to_file_translation_for_old_scenes(store, two_subs_mkv, no_speech):
    p = store.create(two_subs_mkv, probe(two_subs_mkv), title="t")
    analyze_project(store, p.id, NullContext(), settings=AppSettings())
    project = store.get(p.id)
    old_scene = Scene(id="sc001", title="s", start=0, end=9, lines=[       # сцена создана до импорта перевода
        SceneLine(line_id="l0001", speaker_id=None, start_rel=1, end_rel=3, original="Hello there, how are you?",
                  text="Hello there, how are you?"),
        SceneLine(line_id="l0002", speaker_id=None, start_rel=5, end_rel=8, original="I am fine.", text="Мой текст",
                  translation_source="user")])
    got = subtitle_lines(project, old_scene)
    assert got[0].text == "Привет, как дела?" and got[0].translation_source == "file"
    assert got[1].text == "Мой текст"
    from dubl_engine.audio.render import subtitle_items
    assert [t for _l, t in subtitle_items(got, SubtitleText.DUB)] == ["Привет, как дела?", "Мой текст"]
    assert [t for _l, t in subtitle_items(got, SubtitleText.ORIGINAL)] == ["Hello there, how are you?", "I am fine."]


def test_parse_keeps_working_on_translation_srt(two_subs_mkv, tmp_path):
    from dubl_engine.media.extract import extract_subtitles
    info = probe(two_subs_mkv)
    rus = next(s for s in info.subtitle_streams if s.language == "rus")
    lines = parse_subtitle_file(extract_subtitles(two_subs_mkv, rus.index, tmp_path / "ru"))
    assert [ln.text for ln in lines] == ["Привет, как дела?", "У меня всё хорошо.", "Большое спасибо."]
