from __future__ import annotations

import pytest

from dubl_engine.analysis.palette import PALETTE
from dubl_engine.analysis.subtitles import parse_subtitle_file, speakers_from_names
from dubl_engine.media.extract import extract_subtitles
from dubl_engine.models import Line
from dubl_engine.util import EngineError


def test_palette():
    assert len(PALETTE) == 10 and len(set(PALETTE)) == 10
    assert PALETTE[:5] == ["#4fb8a8", "#e57a5f", "#6aa6e8", "#c58ad6", "#a8c95a"]

    def lum(h):
        r, g, b = (int(h[i:i + 2], 16) / 255 for i in (1, 3, 5))
        lin = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in (r, g, b)]
        return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]

    bg = lum("#0d0e0f")
    for c in PALETTE:
        assert len(c) == 7 and c.startswith("#")
        assert (lum(c) + 0.05) / (bg + 0.05) > 5   # читаемо на тёмном фоне


def test_parse_srt(tmp_path):
    f = tmp_path / "a.srt"
    f.write_text("1\n00:00:06,500 --> 00:00:08,000\n<i>Второй</i>\nкусок\n\n"
                 "2\n00:00:01,000 --> 00:00:02,250\nПервый\n\n"
                 "3\n00:00:03,000 --> 00:00:04,000\n\n\n"
                 "4\n01:02:03,004 --> 01:02:05,000\n{\\an8}Поздно\n", encoding="utf-8")
    lines = parse_subtitle_file(f)
    assert [ln.id for ln in lines] == ["l0001", "l0002", "l0003"]
    assert [ln.text for ln in lines] == ["Первый", "Второй кусок", "Поздно"]
    assert lines[0].start == 1.0 and lines[0].end == 2.25
    assert lines[2].start == pytest.approx(3723.004)
    assert all(ln.source == "embedded" and ln.speaker_id is None for ln in lines)


def test_parse_srt_crlf_bom_and_cp1251(tmp_path):
    f = tmp_path / "a.srt"
    f.write_bytes("\ufeff1\r\n00:00:01,000 --> 00:00:02,000\r\nПривет\r\n".encode("utf-8"))
    assert parse_subtitle_file(f)[0].text == "Привет"
    g = tmp_path / "b.srt"
    g.write_bytes("1\n00:00:01,000 --> 00:00:02,000\nПривет, мир\n".encode("cp1251"))
    assert parse_subtitle_file(g)[0].text == "Привет, мир"


ASS = (
    "[Script Info]\nScriptType: v4.00+\n\n[V4+ Styles]\nFormat: Name, Fontname\nStyle: Default,Arial\n\n"
    "[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    "Dialogue: 0,0:00:06.00,0:00:08.00,Default,Mila,0,0,0,,{\\i1}Not me.{\\i0} I was asleep.\n"
    "Dialogue: 0,0:00:02.00,0:00:04.50,Default,Barsik,0,0,0,,Who touched\\Nmy sausage?, really\n"
    "Dialogue: 0,0:00:09.00,0:00:10.00,Default,,0,0,0,,No name\\nhere\n"
    "Dialogue: 0,0:00:11.00,0:00:12.00,Default,Mila,0,0,0,,{\\pos(1,2)}\n"
    "Comment: 0,0:00:13.00,0:00:14.00,Default,Mila,0,0,0,,not a dialogue\n"
)


def test_parse_ass(tmp_path):
    f = tmp_path / "a.ass"
    f.write_text(ASS, encoding="utf-8")
    lines = parse_subtitle_file(f)
    assert [ln.text for ln in lines] == ["Who touched my sausage?, really", "Not me. I was asleep.", "No name here"]
    assert [ln.speaker_id for ln in lines] == ["name:Barsik", "name:Mila", None]
    assert [ln.id for ln in lines] == ["l0001", "l0002", "l0003"]
    assert lines[0].start == 2.0 and lines[0].end == 4.5
    assert all(ln.source == "embedded" for ln in lines)


def test_parse_ass_from_rich_mkv(rich_mkv, tmp_path):
    ass = extract_subtitles(rich_mkv, 3, tmp_path / "s.ass")
    lines = parse_subtitle_file(ass)
    assert len(lines) == 2
    assert lines[0].speaker_id == "name:Barsik" and lines[0].text == "Who touched my sausage?"
    assert lines[1].speaker_id == "name:Mila" and lines[1].text == "Not me. I was asleep."
    assert lines[0].start == pytest.approx(2.0) and lines[1].end == pytest.approx(8.0)


def test_parse_srt_from_rich_mkv(rich_mkv, tmp_path):
    srt = extract_subtitles(rich_mkv, 4, tmp_path / "s.srt")
    lines = parse_subtitle_file(srt)
    assert [ln.text for ln in lines] == ["Кто трогал мою колбасу?", "Не я. Я спал."]


def test_parse_empty_and_missing(tmp_path):
    f = tmp_path / "e.srt"
    f.write_text("", encoding="utf-8")
    assert parse_subtitle_file(f) == []
    with pytest.raises(EngineError, match="не найден"):
        parse_subtitle_file(tmp_path / "none.srt")


def _l(i, s, e, sid):
    return Line(id=f"l{i:04d}", start=s, end=e, text="x", speaker_id=sid, source="embedded")


def test_speakers_from_names():
    lines = [_l(1, 0, 2, "name:Barsik"), _l(2, 3, 4, "name:Mila"), _l(3, 5, 6, None),
             _l(4, 7, 8, "name:Barsik"), _l(5, 9, 10, "s9")]
    new, speakers = speakers_from_names(lines)
    assert [s.id for s in speakers] == ["s1", "s2"]
    assert [s.name for s in speakers] == ["Barsik", "Mila"]
    assert [s.label for s in speakers] == ["Barsik", "Mila"]
    assert [s.color for s in speakers] == PALETTE[:2]
    assert speakers[0].line_count == 2 and speakers[0].total_speech == pytest.approx(3.0)
    assert speakers[1].line_count == 1
    assert [ln.speaker_id for ln in new] == ["s1", "s2", None, "s1", "s9"]
    assert lines[0].speaker_id == "name:Barsik"   # исходные не тронуты


def test_speakers_from_names_none_and_many():
    lines = [_l(1, 0, 1, None)]
    assert speakers_from_names(lines) == (lines, [])
    many = [_l(i, i, i + 0.5, f"name:N{i}") for i in range(1, 13)]
    new, sp = speakers_from_names(many)
    assert len(sp) == 12 and sp[10].color == PALETTE[0]   # палитра зацикливается
    assert new[11].speaker_id == "s12"
