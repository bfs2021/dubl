from __future__ import annotations

from pathlib import Path

import pytest

from dubl_engine.media.extract import (
    cut_scene_video,
    extract_audio,
    extract_center_channel,
    extract_subtitles,
    has_front_center,
)
from dubl_engine.media.probe import probe
from dubl_engine.util import EngineError, ffprobe_json, run_ffmpeg


def _wav_info(path: Path) -> dict:
    s = ffprobe_json(path, "-show_streams")["streams"][0]
    return {"codec": s["codec_name"], "sr": int(s["sample_rate"]), "ch": int(s["channels"]),
            "dur": float(s["duration"])}


def _rms_db(path: Path, start: float, end: float) -> float:
    import numpy as np
    import soundfile as sf
    data, sr = sf.read(str(path), dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    seg = data[int(start * sr):int(end * sr)]
    return float(20 * np.log10(np.sqrt(np.mean(seg ** 2)) + 1e-9))


def test_extract_audio_basic(simple_mp4, tmp_path):
    info = probe(simple_mp4)
    out = extract_audio(simple_mp4, info.audio_streams[0].index, tmp_path / "sub" / "a.wav")
    assert out.exists()
    w = _wav_info(out)
    assert w["codec"] == "pcm_f32le" and w["sr"] == 48000 and w["ch"] == 2
    assert w["dur"] == pytest.approx(10.0, abs=0.2)


def test_extract_audio_mono_16k_range(simple_mp4, tmp_path):
    out = extract_audio(simple_mp4, 1, tmp_path / "a.wav", sample_rate=16000, mono=True, start=2.0, end=4.0)
    w = _wav_info(out)
    assert w["sr"] == 16000 and w["ch"] == 1
    assert w["dur"] == pytest.approx(2.0, abs=0.1)


def test_extract_audio_rich_stream_choice(rich_mkv, tmp_path):
    info = probe(rich_mkv)
    out = extract_audio(rich_mkv, info.audio_streams[0].index, tmp_path / "a51.wav")
    assert _wav_info(out)["ch"] == 6
    out2 = extract_audio(rich_mkv, info.audio_streams[1].index, tmp_path / "ast.wav", mono=True)
    assert _wav_info(out2)["ch"] == 1


def test_extract_audio_bad_stream(simple_mp4, tmp_path):
    with pytest.raises(EngineError, match="нет потока"):
        extract_audio(simple_mp4, 99, tmp_path / "a.wav")
    with pytest.raises(EngineError, match="не является"):
        extract_audio(simple_mp4, 0, tmp_path / "a.wav")   # видеопоток


def test_extract_audio_bad_range(simple_mp4, tmp_path):
    with pytest.raises(EngineError, match="позже"):
        extract_audio(simple_mp4, 1, tmp_path / "a.wav", start=5, end=2)


def test_center_channel_51(rich_mkv, tmp_path):
    info = probe(rich_mkv)
    out = extract_center_channel(rich_mkv, info.audio_streams[0].index, tmp_path / "c.wav")
    assert out is not None
    w = _wav_info(out)
    assert w["ch"] == 1 and w["sr"] == 48000
    # центр == третий канал полной дорожки (порядок FL FR FC LFE BL BR), а не микс и не другой канал
    import numpy as np
    import soundfile as sf
    full = extract_audio(rich_mkv, info.audio_streams[0].index, tmp_path / "full.wav")
    all_ch, _ = sf.read(str(full), dtype="float32")
    center, _ = sf.read(str(out), dtype="float32")
    n = min(len(all_ch), len(center))
    assert np.allclose(center[:n], all_ch[:n, 2], atol=1e-4)
    # диалог фикстуры лежит именно в FC (в conftest явная карта каналов join), а музыка — в остальных
    speech = center[int(2.5 * 48000):int(3.5 * 48000)]
    quiet = center[int(0.2 * 48000):int(1.5 * 48000)]
    assert np.sqrt(np.mean(speech ** 2)) > 0.01 and np.sqrt(np.mean(quiet ** 2)) < 1e-3


def test_center_channel_range(rich_mkv, tmp_path):
    out = extract_center_channel(rich_mkv, 1, tmp_path / "c.wav", sample_rate=16000, start=1.0, end=5.0)
    w = _wav_info(out)
    assert w["sr"] == 16000 and w["dur"] == pytest.approx(4.0, abs=0.1)


def test_center_channel_stereo_is_none(rich_mkv, simple_mp4, tmp_path):
    assert extract_center_channel(rich_mkv, 2, tmp_path / "c.wav") is None
    assert extract_center_channel(simple_mp4, 1, tmp_path / "c2.wav") is None
    assert not (tmp_path / "c.wav").exists()


@pytest.mark.parametrize("layout,expected", [
    ("5.1", True), ("5.1(side)", True), ("7.1", True), ("7.1(wide)", True), ("3.0", True),
    ("stereo", False), ("mono", False), ("quad", False), ("2.1", False), ("6.1(front)", False),
    ("FL+FR+FC", True), ("FL+FR", False), (None, False),
])
def test_has_front_center(layout, expected):
    assert has_front_center(layout) is expected


def test_has_front_center_without_layout_uses_channels():
    assert has_front_center(None, 6) is True
    assert has_front_center(None, 2) is False


def test_extract_subtitles_ass_and_srt(rich_mkv, tmp_path):
    ass = extract_subtitles(rich_mkv, 3, tmp_path / "s" / "subs")
    assert ass.suffix == ".ass" and ass.exists()
    assert "Barsik" in ass.read_text(encoding="utf-8")
    srt = extract_subtitles(rich_mkv, 4, tmp_path / "subs_ru.ass")   # суффикс подстраивается под кодек
    assert srt.suffix == ".srt"
    assert "Кто трогал мою колбасу?" in srt.read_text(encoding="utf-8")


def test_extract_subtitles_not_subtitle(rich_mkv, tmp_path):
    with pytest.raises(EngineError, match="не является"):
        extract_subtitles(rich_mkv, 1, tmp_path / "x.srt")


def test_extract_subtitles_image_codec(monkeypatch, tmp_path):
    from dubl_engine.media import extract as mod
    monkeypatch.setattr(mod, "ffprobe_json", lambda *a, **k: {"streams": [
        {"index": 2, "codec_type": "subtitle", "codec_name": "hdmv_pgs_subtitle"}]})
    with pytest.raises(EngineError, match="картинки"):
        mod.extract_subtitles("whatever.mkv", 2, tmp_path / "x.srt")


def test_cut_scene_video(rich_mkv, tmp_path):
    out = cut_scene_video(rich_mkv, 3.0, 7.0, tmp_path / "scene" / "clip.mp4", audio_stream_index=2)
    info = probe(out)
    assert info.duration == pytest.approx(4.0, abs=0.15)
    assert (info.width, info.height) == (640, 360)
    assert info.video_codec == "h264"
    assert len(info.audio_streams) == 1
    assert info.audio_streams[0].codec == "aac"
    assert info.audio_streams[0].channels == 2        # выбрана rus stereo, а не eng 5.1
    assert info.subtitle_streams == []


def test_cut_scene_video_keeps_selected_audio_51(rich_mkv, tmp_path):
    out = cut_scene_video(rich_mkv, 0.0, 2.0, tmp_path / "clip51.mp4", audio_stream_index=1)
    assert probe(out).audio_streams[0].channels == 6


def test_cut_scene_video_errors(rich_mkv, tmp_path):
    with pytest.raises(EngineError):
        cut_scene_video(rich_mkv, 5, 5, tmp_path / "a.mp4", audio_stream_index=1)
    with pytest.raises(EngineError):
        cut_scene_video(rich_mkv, 1, 2, tmp_path / "a.mp4", audio_stream_index=77)


def test_wav_is_readable_by_stdlib_wave_header(simple_mp4, tmp_path):
    out = extract_audio(simple_mp4, 1, tmp_path / "a.wav", mono=True, end=1.0)
    assert out.read_bytes()[:4] == b"RIFF"


# ───────────── главы и теги фильма не попадают в ролик ─────────────
def _chapters_and_tags(path):
    from dubl_engine.util import ffprobe_json
    d = ffprobe_json(path, "-show_chapters", "-show_format")
    return d.get("chapters") or [], (d.get("format") or {}).get("tags") or {}


def test_cut_scene_video_drops_chapters_and_global_tags(rich_mkv, tmp_path):
    src = tmp_path / "tagged.mkv"
    run_ffmpeg(["-i", rich_mkv, "-map", "0", "-c", "copy", "-metadata", "title=Фильм целиком", "-metadata",
                "comment=секрет исходника", src])
    chapters, tags = _chapters_and_tags(src)
    assert len(chapters) == 2 and tags.get("title") == "Фильм целиком"      # условие теста: у источника они есть
    out = cut_scene_video(src, 1.0, 9.0, tmp_path / "clip.mp4", audio_stream_index=2)
    chapters, tags = _chapters_and_tags(out)
    assert chapters == []
    assert "title" not in tags and "comment" not in tags


def test_render_video_drops_chapters_and_global_tags(rich_mkv, tmp_path):
    import numpy as np

    from dubl_engine.audio.io import write_wav
    from dubl_engine.audio.render import render_video
    from dubl_engine.models import RenderSettings

    src = tmp_path / "tagged.mkv"
    run_ffmpeg(["-i", rich_mkv, "-map", "0:v", "-map", "0:a:1", "-map_chapters", "0", "-c", "copy", "-metadata",
                "title=Фильм целиком", src])
    assert len(_chapters_and_tags(src)[0]) == 2
    wav = write_wav(tmp_path / "m.wav", np.zeros((48000 * 10, 2), dtype=np.float32), 48000)
    out = render_video(src, wav, tmp_path / "out.mp4", RenderSettings(output_path="x", height=None))
    chapters, tags = _chapters_and_tags(out)
    assert chapters == [] and "title" not in tags
