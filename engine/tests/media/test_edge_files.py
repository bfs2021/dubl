"""Файлы «из жизни»: нечётные размеры, анаморфный SAR, VFR, без звука, пути с кириллицей/пробелами/кавычками."""
from __future__ import annotations

import os
import subprocess

import numpy as np
import pytest

from dubl_engine.audio.io import write_wav
from dubl_engine.audio.render import compute_geometry, render_video, write_subtitles
from dubl_engine.media.extract import cut_scene_video, extract_audio
from dubl_engine.media.probe import main_video_index, probe
from dubl_engine.media.shots import detect_shot_cuts
from dubl_engine.models import RenderAspect, RenderSettings, SceneLine, SubtitleMode
from dubl_engine.util import EngineError, ffprobe_json


def _ff(*args):
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *map(str, args)], check=True)


def _mix_wav(tmp_path, seconds):
    return write_wav(tmp_path / "mix.wav", (0.1 * np.sin(2 * np.pi * 440 * np.arange(48000 * seconds) / 48000))
                     .astype(np.float32), 48000)


def _vstream(path):
    return next(s for s in ffprobe_json(path, "-show_streams")["streams"] if s["codec_type"] == "video")


def _vstream_by_index(path, idx):
    return next(s for s in ffprobe_json(path, "-show_streams")["streams"] if s["index"] == idx)


@pytest.fixture(scope="module")
def odd_mkv(tmp_path_factory):
    """640x358 → нечётная ширина 641x359 через crop нельзя в yuv420p; делаем yuv444, чтобы файл существовал."""
    d = tmp_path_factory.mktemp("odd")
    out = d / "odd.mkv"
    _ff("-f", "lavfi", "-i", "testsrc2=s=640x360:r=25:d=4", "-f", "lavfi", "-i", "sine=f=300:d=4",
        "-vf", "scale=641:359,format=yuv444p", "-c:v", "ffv1", "-c:a", "aac", out)
    return out


@pytest.fixture(scope="module")
def anamorphic_mp4(tmp_path_factory):
    d = tmp_path_factory.mktemp("ana")
    out = d / "ana.mp4"
    _ff("-f", "lavfi", "-i", "testsrc2=s=480x360:r=25:d=4", "-f", "lavfi", "-i", "sine=f=300:d=4",
        "-vf", "setsar=4/3", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", out)
    return out


@pytest.fixture(scope="module")
def silent_mp4(tmp_path_factory):
    out = tmp_path_factory.mktemp("sil") / "silent.mp4"
    _ff("-f", "lavfi", "-i", "testsrc2=s=320x240:r=25:d=3", "-c:v", "libx264", "-pix_fmt", "yuv420p", out)
    return out


@pytest.fixture(scope="module")
def vfr_mkv(tmp_path_factory):
    d = tmp_path_factory.mktemp("vfr")
    out = d / "vfr.mkv"
    _ff("-f", "lavfi", "-i", "testsrc2=s=320x240:r=30:d=6", "-f", "lavfi", "-i", "sine=f=300:d=6",
        "-vf", "select='not(mod(n,3))+not(mod(n,7))'", "-fps_mode", "vfr", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", out)
    return out


def test_odd_size_cut_scene_gives_even_h264(odd_mkv, tmp_path):
    info = probe(odd_mkv)
    assert (info.width, info.height) == (641, 359)
    clip = cut_scene_video(odd_mkv, 1.0, 3.0, tmp_path / "c.mp4", audio_stream_index=info.audio_streams[0].index)
    v = _vstream(clip)
    assert v["width"] % 2 == 0 and v["height"] % 2 == 0 and v["pix_fmt"] == "yuv420p"


def test_odd_size_render_9x16(odd_mkv, tmp_path):
    info = probe(odd_mkv)
    clip = cut_scene_video(odd_mkv, 0.0, 3.0, tmp_path / "c.mp4", audio_stream_index=info.audio_streams[0].index)
    st = RenderSettings(aspect=RenderAspect.V9_16, height=360, output_path=str(tmp_path / "o.mp4"))
    out = render_video(clip, _mix_wav(tmp_path, 3), tmp_path / "o.mp4", st)
    v = _vstream(out)
    assert (v["width"], v["height"]) == (360, 640)


def test_geometry_always_even():
    for w, h in ((641, 359), (1919, 1081), (3, 3), (1, 1000)):
        for asp in RenderAspect:
            for hh in (None, 721, 640):
                crop, (ow, oh) = compute_geometry(w, h, RenderSettings(aspect=asp, height=hh, output_path="x"))
                assert ow % 2 == 0 and oh % 2 == 0 and ow >= 2 and oh >= 2
                if crop:
                    assert crop[0] % 2 == 0 and crop[1] % 2 == 0 and crop[2] % 2 == 0 and crop[3] % 2 == 0
                    assert crop[2] + crop[0] <= w and crop[3] + crop[1] <= h


def test_anamorphic_sar_is_squared_on_render(anamorphic_mp4, tmp_path):
    info = probe(anamorphic_mp4)
    assert (info.width, info.height) == (480, 360)
    clip = cut_scene_video(anamorphic_mp4, 0.0, 3.0, tmp_path / "c.mp4",
                           audio_stream_index=info.audio_streams[0].index)
    st = RenderSettings(aspect=RenderAspect.ORIGINAL, height=None, output_path=str(tmp_path / "o.mp4"))
    out = render_video(clip, _mix_wav(tmp_path, 3), tmp_path / "o.mp4", st)
    v = _vstream(out)
    # 480x360 с пикселем 4:3 — это картинка 640x360 (16:9)
    assert (v["width"], v["height"]) == (640, 360)
    assert v.get("sample_aspect_ratio", "1:1") in ("1:1", "0:1")


def test_anamorphic_crop_9x16_uses_display_shape(anamorphic_mp4, tmp_path):
    info = probe(anamorphic_mp4)
    clip = cut_scene_video(anamorphic_mp4, 0.0, 3.0, tmp_path / "c.mp4",
                           audio_stream_index=info.audio_streams[0].index)
    st = RenderSettings(aspect=RenderAspect.V9_16, height=360, output_path=str(tmp_path / "o.mp4"))
    v = _vstream(render_video(clip, _mix_wav(tmp_path, 3), tmp_path / "o.mp4", st))
    assert (v["width"], v["height"]) == (360, 640)


def test_no_audio_file_probe_and_errors(silent_mp4, tmp_path):
    info = probe(silent_mp4)
    assert info.audio_streams == [] and info.width == 320
    from dubl_engine.media.probe import pick_original_audio
    with pytest.raises(EngineError, match="нет звуковых"):
        pick_original_audio(info)
    with pytest.raises(EngineError):
        extract_audio(silent_mp4, 0, tmp_path / "a.wav")            # 0 — видеопоток, не звук


def test_vfr_cut_and_shots_work(vfr_mkv, tmp_path):
    info = probe(vfr_mkv)
    assert info.duration > 3 and info.fps > 0
    clip = cut_scene_video(vfr_mkv, 1.0, 3.0, tmp_path / "c.mp4", audio_stream_index=info.audio_streams[0].index)
    assert float(ffprobe_json(clip, "-show_format")["format"]["duration"]) == pytest.approx(2.0, abs=0.35)
    assert isinstance(detect_shot_cuts(vfr_mkv), list)


def test_attached_cover_is_not_main_video(tmp_path):
    cover = tmp_path / "cover.jpg"
    _ff("-f", "lavfi", "-i", "color=c=green:s=100x100:d=1", "-frames:v", "1", cover)
    out = tmp_path / "with_cover.mp4"
    _ff("-f", "lavfi", "-i", "testsrc2=s=320x240:r=25:d=2", "-f", "lavfi", "-i", "sine=f=300:d=2", "-i", cover,
        "-map", "2:v", "-map", "0:v", "-map", "1:a", "-c:v:0", "mjpeg", "-c:v:1", "libx264", "-pix_fmt:v:1", "yuv420p",
        "-c:a", "aac", "-disposition:v:0", "attached_pic", out)
    info = probe(out)
    assert (info.width, info.height) == (320, 240)
    assert _vstream_by_index(out, main_video_index(out))["codec_name"] == "h264"
    clip = cut_scene_video(out, 0.0, 1.5, tmp_path / "c.mp4", audio_stream_index=info.audio_streams[0].index)
    assert _vstream(clip)["width"] == 320


NASTY = ["папка с пробелом", "it's: a [test], 100% ok;x=1", "тест'кавычка\"двойная", "a=b,c:d"]
if os.name == "nt":     # «:» и «"» Windows в именах не примет; остальное (кавычка ', скобки, %, ;, =, запятая) — оставляем
    NASTY = [n.replace(":", "_").replace('"', "_") for n in NASTY]


@pytest.mark.parametrize("dirname", NASTY)
def test_nasty_paths_through_whole_ffmpeg_chain(simple_mp4, tmp_path, dirname):
    d = tmp_path / dirname
    d.mkdir()
    src = d / (dirname + " видео.mp4")
    src.write_bytes(simple_mp4.read_bytes())
    info = probe(src)
    wav = extract_audio(src, info.audio_streams[0].index, d / "звук: 1.wav", start=1.0, end=3.0)
    assert wav.exists()
    clip = cut_scene_video(src, 1.0, 3.0, d / "клип 'один'.mp4", audio_stream_index=info.audio_streams[0].index)
    assert detect_shot_cuts(src) == []
    ass = write_subtitles([SceneLine(line_id="l", speaker_id=None, start_rel=0.2, end_rel=1.5, original="x",
                                     text="Привет, мир")], d / "суб: титры.ass", fmt="ass", play_res=(360, 640))
    out = d / "итог: [готово], 'ok'.mp4"
    st = RenderSettings(aspect=RenderAspect.V9_16, height=360, subtitles=SubtitleMode.BURN, output_path=str(out))
    render_video(clip, _mix_wav(d, 2), out, st, subtitles_ass=ass)
    assert (_vstream(out)["width"], _vstream(out)["height"]) == (360, 640)


def test_relative_name_with_colon_is_not_a_protocol(simple_mp4, tmp_path, monkeypatch):
    """«abc: x.mp4» без каталога ffmpeg принял бы за протокол «abc:» — движок всегда передаёт абсолютный путь."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "abc: x.mp4").write_bytes(simple_mp4.read_bytes())
    from pathlib import Path
    assert probe(Path("abc: x.mp4")).width == 640
    extract_audio(Path("abc: x.mp4"), 1, Path("out: y.wav"), start=0, end=1)
    assert (tmp_path / "out: y.wav").exists()


def test_log_filter_scrubs_token_in_urls():
    import logging

    from dubl_engine.main import ScrubTokenFilter
    rec = logging.LogRecord("uvicorn.access", logging.INFO, "", 0, '%s - "%s %s HTTP/%s" %d',
                            ("1.2.3.4:5", "GET", "/files/p/clip.mp4?token=SUPERSECRET123&x=1", "1.1", 200), None)
    assert ScrubTokenFilter().filter(rec)
    assert "SUPERSECRET123" not in rec.getMessage() and "token=***&x=1" in rec.getMessage()


def test_srt_blank_line_inside_text_does_not_break_block(tmp_path):
    out = write_subtitles([SceneLine(line_id="l", speaker_id=None, start_rel=0, end_rel=1, original="x",
                                     text="Раз\n\n\nДва")], tmp_path / "a.srt", fmt="srt").read_text(encoding="utf-8")
    assert "Раз\nДва" in out and "\n\n\n" not in out
