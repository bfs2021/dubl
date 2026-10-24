"""Субтитры «из жизни»: BOM, CRLF, cp1251, <i>, песни OP/ED, надписи {\\pos}, караоке, несколько стилей."""
from __future__ import annotations

from dubl_engine.analysis.subtitles import is_non_dialogue_style, parse_subtitle_file, speakers_from_names

ASS_HEAD = ("[Script Info]\nScriptType: v4.00+\n\n[V4+ Styles]\nFormat: Name, Fontname\nStyle: Default,Arial\n\n"
            "[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n")


def _ass(tmp_path, body: str, name="s.ass"):
    p = tmp_path / name
    p.write_text(ASS_HEAD + body, encoding="utf-8")
    return p


def test_srt_bom_crlf_italics(tmp_path):
    p = tmp_path / "a.srt"
    p.write_bytes("\ufeff1\r\n00:00:01,000 --> 00:00:02,500\r\n<i>Привет</i>,\r\n<font color=\"#fff\">мир</font>\r\n\r\n"
                  "2\r\n00:00:03,000 --> 00:00:04,000\r\n[music]\r\n\r\n3\r\n00:00:05,000 --> 00:00:06,000\r\n- Да.\r\n"
                  "- Нет.\r\n".encode("utf-8"))
    lines = parse_subtitle_file(p)
    assert [(l.start, l.end, l.text) for l in lines] == [(1.0, 2.5, "Привет, мир"), (5.0, 6.0, "- Да. - Нет.")]


def test_srt_cp1251(tmp_path):
    p = tmp_path / "a.srt"
    p.write_bytes("1\n00:00:01,000 --> 00:00:02,000\nКто трогал мою колбасу?\n".encode("cp1251"))
    assert parse_subtitle_file(p)[0].text == "Кто трогал мою колбасу?"


def test_srt_cp1252_western(tmp_path):
    p = tmp_path / "a.srt"
    p.write_bytes("1\n00:00:01,000 --> 00:00:02,000\nCafé crème, señor\n".encode("cp1252"))
    assert parse_subtitle_file(p)[0].text == "Café crème, señor"


def test_srt_utf16(tmp_path):
    p = tmp_path / "a.srt"
    p.write_bytes("1\n00:00:01,000 --> 00:00:02,000\nПривет\n".encode("utf-16"))
    assert parse_subtitle_file(p)[0].text == "Привет"


def test_srt_missing_blank_lines_and_junk(tmp_path):
    p = tmp_path / "a.srt"
    p.write_text("1\n00:00:01,000 --> 00:00:02,000\nОдин\n2\n00:00:03,000 --> 00:00:04,000\nДва\n\n\n\n"
                 "3\n00:00:05,000 --> 00:00:05,000\nНулевая\n\n4\n00:00:07,5 --> 00:00:08,25\nТри\n", encoding="utf-8")
    assert [l.text for l in parse_subtitle_file(p)] == ["Один", "Два", "Три"]
    assert parse_subtitle_file(p)[2].start == 7.5


def test_srt_empty_and_garbage(tmp_path):
    p = tmp_path / "a.srt"
    p.write_text("", encoding="utf-8")
    assert parse_subtitle_file(p) == []
    p.write_text("это не субтитры\nвообще", encoding="utf-8")
    assert parse_subtitle_file(p) == []


def test_ass_drops_songs_signs_karaoke_drawings(tmp_path):
    p = _ass(tmp_path,
             "Dialogue: 0,0:00:01.00,0:00:02.00,Default,Barsik,0,0,0,,Who touched it?\n"
             "Dialogue: 0,0:00:01.00,0:00:02.00,OP,,0,0,0,,{\\k20}Ka{\\k30}ra{\\k20}oke line\n"
             "Dialogue: 0,0:00:03.00,0:00:04.00,ED Romaji,,0,0,0,,Ending song text\n"
             "Dialogue: 0,0:00:03.00,0:00:04.00,Sign - door,,0,0,0,,EXIT\n"
             "Dialogue: 0,0:00:05.00,0:00:06.00,Default,,0,0,0,,{\\pos(100,50)}Надпись на стене\n"
             "Dialogue: 0,0:00:05.00,0:00:06.00,Default,,0,0,0,,{\\p1}m 0 0 l 10 10{\\p0}\n"
             "Dialogue: 0,0:00:05.00,0:00:06.00,Default,,0,0,0,,{\\move(1,2,3,4)}Бегущая\n"
             "Dialogue: 0,0:00:07.00,0:00:08.00,Default,Mila,0,0,0,,{\\an8}Not me.\\NI slept.\n"
             "Dialogue: 0,0:00:09.00,0:00:10.00,Song,,0,0,0,,la la la\n"
             "Dialogue: 0,0:00:11.00,0:00:12.00,Default,,0,0,0,,♪ sing ♪\n"
             "Dialogue: 1,0:00:07.00,0:00:08.00,Default,Mila,0,0,0,,{\\an8}Not me.\\NI slept.\n")
    lines = parse_subtitle_file(p)
    assert [l.text for l in lines] == ["Who touched it?", "Not me. I slept."]      # дубль слоя схлопнут
    assert [l.speaker_id for l in lines] == ["name:Barsik", "name:Mila"]


def test_ass_pos_in_dialogue_style_kept_and_names_case_merged(tmp_path):
    p = _ass(tmp_path,
             "Dialogue: 0,0:00:01.00,0:00:02.00,Main,BARSIK,0,0,0,,{\\pos(10,10)}Привет\n"
             "Dialogue: 0,0:00:03.00,0:00:04.00,Main,barsik ,0,0,0,,Пока\n"
             "Dialogue: 0,0:00:05.00,0:00:06.00,Main,-,0,0,0,,Без имени\n")
    lines = parse_subtitle_file(p)
    assert len(lines) == 3 and lines[2].speaker_id is None
    new, spk = speakers_from_names(lines)
    assert len(spk) == 1 and spk[0].line_count == 2


def test_ass_comma_in_text_and_missing_format_fields(tmp_path):
    p = _ass(tmp_path, "Dialogue: 0,0:00:01.00,0:00:02.00,Default,,0,0,0,,Да, но нет, и всё\n"
                       "Dialogue: 0,broken,0:00:02.00,Default,,0,0,0,,x\n")
    assert [l.text for l in parse_subtitle_file(p)] == ["Да, но нет, и всё"]


def test_ass_cp1251_and_crlf(tmp_path):
    p = tmp_path / "s.ass"
    p.write_bytes((ASS_HEAD + "Dialogue: 0,0:00:01.00,0:00:02.00,Default,Кот,0,0,0,,Привет\n").replace("\n", "\r\n")
                  .encode("cp1251"))
    lines = parse_subtitle_file(p)
    assert lines[0].text == "Привет" and lines[0].speaker_id == "name:Кот"


def test_style_classifier():
    for st in ("OP", "ED", "Sign", "Signs - Title", "Song", "OP1", "ED2 Kara", "Karaoke", "Опенинг"):
        assert is_non_dialogue_style(st), st
    for st in ("Default", "Main", "Top", "Italics", "Dialogue", "Opinion", "Edward", "Signal"):
        assert not is_non_dialogue_style(st), st
