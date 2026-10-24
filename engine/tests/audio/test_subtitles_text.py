"""Субтитры в ролике — текст дубля; нет перевода — оригинал (реплика не пропадает); по умолчанию выключены."""
from __future__ import annotations

import numpy as np

from dubl_engine.audio.io import write_wav
from dubl_engine.audio.render import render_video, subtitle_items, write_subtitles
from dubl_engine.models import RenderSettings, SceneLine, SubtitleMode, SubtitleText
from dubl_engine.util import ffprobe_json


def _lines():
    return [
        SceneLine(line_id="l1", speaker_id="s1", start_rel=0.5, end_rel=2.0, original="Who touched my sausage?",
                  text="Кто трогал мою колбасу?"),
        SceneLine(line_id="l2", speaker_id="s1", start_rel=3.0, end_rel=4.0, original="Not me.", text="Not me."),
        SceneLine(line_id="l3", speaker_id="s2", start_rel=5.0, end_rel=6.0, original="Hello", text="  "),
    ]


def test_defaults_are_off_and_dub():
    rs = RenderSettings(output_path="x.mp4")
    assert rs.subtitles == SubtitleMode.NONE and rs.subtitle_text == SubtitleText.DUB


def test_ass_has_dub_text_not_original(tmp_path):
    ass = write_subtitles(_lines(), tmp_path / "a.ass", fmt="ass").read_text(encoding="utf-8")
    assert "Кто трогал мою колбасу?" in ass
    assert "Who touched my sausage?" not in ass
    assert "Not me" in ass and "Hello" in ass               # перевода нет — в субтитрах оригинал, реплика не пропадает
    assert ass.count("Dialogue:") == 3


def test_srt_has_dub_text_and_original_on_request(tmp_path):
    srt = write_subtitles(_lines(), tmp_path / "a.srt", fmt="srt").read_text(encoding="utf-8")
    assert "Кто трогал" in srt and "sausage" not in srt
    orig = write_subtitles(_lines(), tmp_path / "o.ass", fmt="ass", text="original").read_text(encoding="utf-8")
    assert "Who touched my sausage?" in orig and "Not me." in orig and "Hello" in orig and "Кто трогал" not in orig
    assert orig.count("Dialogue:") == 3


def test_subtitle_items_order_and_untranslated_fall_back_to_original():
    items = subtitle_items(list(reversed(_lines())))
    assert [(l.line_id, t) for l, t in items] == [("l1", "Кто трогал мою колбасу?"), ("l2", "Not me."), ("l3", "Hello")]
    assert [l.line_id for l, _ in subtitle_items(_lines(), SubtitleText.ORIGINAL)] == ["l1", "l2", "l3"]


def test_burned_render_uses_dub_text(simple_mp4, tmp_path):
    """Сквозная проверка: _render берёт текст дубля. Подменяем write_subtitles-вывод и смотрим на файл."""
    import dubl_engine.audio.render as r
    seen = {}
    real = r.write_subtitles

    def spy(lines, out, **kw):
        res = real(lines, out, **kw)
        seen["ass"] = res.read_text(encoding="utf-8")
        seen["kw"] = kw
        return res
    wav = write_wav(tmp_path / "mix.wav", (0.1 * np.sin(2 * np.pi * 440 * np.arange(48000 * 10) / 48000)).astype(np.float32), 48000)
    ass = spy(_lines(), tmp_path / "s.ass", fmt="ass")
    out = tmp_path / "o.mp4"
    render_video(simple_mp4, wav, out, RenderSettings(height=360, subtitles=SubtitleMode.BURN, output_path=str(out)),
                 subtitles_ass=ass)
    assert out.exists() and "Кто трогал" in seen["ass"] and "sausage" not in seen["ass"]
    assert ffprobe_json(out, "-show_format")["format"]["duration"]


def test_scene_render_burn_gets_dub_text_end_to_end(ready_project, tmp_path, monkeypatch):
    """Рендер сцены с BURN: в ASS, который уходит в ffmpeg, — текст дубля; без флага «оригинал» английского нет."""
    from dubl_engine.jobs import NullContext
    from dubl_engine.pipeline.scene import render_scene
    store, pid, ids = ready_project
    with store.edit(pid) as p:
        p.scenes[0].lines = _lines()
    import dubl_engine.audio.render as r
    captured = []
    real = r.render_video

    def spy(clip, wav, out, rs, *, subtitles_ass=None, **kw):
        captured.append(subtitles_ass.read_text(encoding="utf-8"))
        return real(clip, wav, out, rs, subtitles_ass=subtitles_ass, **kw)
    monkeypatch.setattr(r, "render_video", spy)
    rs = RenderSettings(height=None, subtitles=SubtitleMode.BURN, output_path=str(tmp_path / "a.mp4"))
    render_scene(store, pid, ids[0], rs, NullContext())
    assert "Кто трогал мою колбасу?" in captured[0] and "sausage" not in captured[0]
    rs2 = rs.model_copy(update={"subtitle_text": SubtitleText.ORIGINAL, "output_path": str(tmp_path / "b.mp4")})
    render_scene(store, pid, ids[0], rs2, NullContext())
    assert "Who touched my sausage?" in captured[1]
