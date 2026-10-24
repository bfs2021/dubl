from __future__ import annotations

import pytest

from dubl_engine.media.probe import pick_original_audio, probe
from dubl_engine.models import AudioStreamInfo, MediaInfo
from dubl_engine.util import EngineError


def test_probe_simple(simple_mp4):
    info = probe(simple_mp4)
    assert info.duration == pytest.approx(10.0, abs=0.2)
    assert (info.width, info.height) == (640, 360)
    assert info.fps == pytest.approx(25.0)
    assert info.video_codec == "h264"
    assert "mp4" in info.container
    assert len(info.audio_streams) == 1
    assert info.audio_streams[0].channels == 2
    assert info.audio_streams[0].sample_rate == 44100 or info.audio_streams[0].sample_rate > 0
    assert info.subtitle_streams == []
    assert info.chapters == []
    assert info.size_bytes > 0


def test_probe_rich(rich_mkv):
    info = probe(rich_mkv)
    assert "matroska" in info.container
    assert info.duration == pytest.approx(10.0, abs=0.2)
    assert [a.language for a in info.audio_streams] == ["eng", "rus"]
    a0, a1 = info.audio_streams
    assert a0.channels == 6 and a0.channel_layout and a0.channel_layout.startswith("5.1")
    assert a1.channels == 2
    assert a0.default is True
    assert a0.title == "English 5.1"
    assert [s.codec for s in info.subtitle_streams] == ["ass", "subrip"]
    assert all(s.is_text for s in info.subtitle_streams)
    assert [s.language for s in info.subtitle_streams] == ["eng", "rus"]
    assert len(info.chapters) == 2
    assert info.chapters[0].title == "Начало"
    assert info.chapters[1].start == pytest.approx(5.0)
    assert info.chapters[1].end == pytest.approx(10.0)


def test_probe_missing_file(tmp_path):
    with pytest.raises(EngineError, match="не найден"):
        probe(tmp_path / "nope.mkv")


def test_probe_not_media(tmp_path):
    f = tmp_path / "x.txt"
    f.write_text("hello", encoding="utf-8")
    with pytest.raises(EngineError):
        probe(f)


def _info(*streams: AudioStreamInfo) -> MediaInfo:
    return MediaInfo(path="x", duration=1, container="mkv", width=1, height=1, fps=25,
                     video_codec="h264", audio_streams=list(streams))


def _a(i, lang=None, default=False):
    return AudioStreamInfo(index=i, codec="aac", channels=2, sample_rate=48000, language=lang, default=default)


def test_pick_prefers_language():
    info = _info(_a(1, "rus", True), _a(2, "eng"), _a(3, "jpn"))
    assert pick_original_audio(info, "eng").index == 2
    assert pick_original_audio(info, "ENG").index == 2


def test_pick_default_when_no_prefer_or_not_found():
    info = _info(_a(1, "rus"), _a(2, "eng", True))
    assert pick_original_audio(info).index == 2
    assert pick_original_audio(info, "fra").index == 2


def test_pick_first_non_rus():
    info = _info(_a(1, "rus"), _a(2, "jpn"), _a(3, "eng"))
    assert pick_original_audio(info).index == 2


def test_pick_first_when_all_rus():
    info = _info(_a(4, "rus"), _a(5, "rus"))
    assert pick_original_audio(info).index == 4


def test_pick_no_audio():
    with pytest.raises(EngineError, match="нет звуковых"):
        pick_original_audio(_info())


def test_pick_rich(rich_mkv):
    assert pick_original_audio(probe(rich_mkv)).language == "eng"
