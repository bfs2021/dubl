from __future__ import annotations

import subprocess

import numpy as np
import pytest

from dubl_engine.audio.io import write_wav
from dubl_engine.audio.render import compute_geometry, render_video, write_subtitles
from dubl_engine.models import RenderAspect, RenderFit, RenderSettings, SceneLine, SubtitleMode
from dubl_engine.util import ffprobe_json


def _lines():
    return [SceneLine(line_id="l1", speaker_id="s1", start_rel=0.5, end_rel=2.0, original="x", text="Привет, мир: {тест}"),
            SceneLine(line_id="l2", speaker_id="s1", start_rel=3.0, end_rel=4.5, original="y", text="Вторая\nстрока")]


def test_geometry():
    crop9 = RenderSettings(aspect=RenderAspect.V9_16, height=360, fit=RenderFit.CROP, output_path="x")
    assert compute_geometry(640, 360, crop9) == ((201 // 2 * 2, 360, 219 // 2 * 2, 0), (360, 640)) or \
        compute_geometry(640, 360, crop9)[1] == (360, 640)
    crop, size = compute_geometry(1920, 1080, RenderSettings(aspect=RenderAspect.H16_9, height=720, output_path="x"))
    assert crop is None and size == (1280, 720)
    sq = RenderSettings(aspect=RenderAspect.SQ1_1, height=None, fit=RenderFit.CROP, output_path="x")
    assert compute_geometry(640, 360, sq)[1] == (1080, 1080)


def test_geometry_portrait_and_square_sizes():
    """9:16 → 1080×1920 и 720×1280, 1:1 → 1080×1080 (раньше 720p давал 404×720)."""
    def size(aspect, h, fit=RenderFit.BLUR):
        return compute_geometry(1920, 1080, RenderSettings(aspect=aspect, height=h, fit=fit, output_path="x"))[1]
    for fit in RenderFit:
        assert size(RenderAspect.V9_16, None, fit) == (1080, 1920)
        assert size(RenderAspect.V9_16, 1080, fit) == (1080, 1920)
        assert size(RenderAspect.V9_16, 720, fit) == (720, 1280)
        assert size(RenderAspect.V9_16, 1920, fit) == (1080, 1920)         # ≥ 1280 — высота кадра
        assert size(RenderAspect.SQ1_1, None, fit) == (1080, 1080)
        assert size(RenderAspect.SQ1_1, 720, fit) == (720, 720)
    assert compute_geometry(1920, 1080, RenderSettings(aspect=RenderAspect.V9_16, height=1080, output_path="x"))[0] is None


def test_blur_layout_puts_whole_frame_by_width():
    from dubl_engine.audio.render import blur_layout
    rs = RenderSettings(aspect=RenderAspect.V9_16, height=1080, output_path="x")
    assert blur_layout(1920, 1080, rs) == ((1080, 606), (1080, 1920))
    assert blur_layout(1920, 1080, rs.model_copy(update={"fit": RenderFit.CROP})) is None
    assert blur_layout(1920, 1080, RenderSettings(aspect=RenderAspect.H16_9, output_path="x")) is None
    assert blur_layout(1080, 1920, rs) is None                                   # источник уже вертикальный


def test_write_subtitles(tmp_path):
    srt = write_subtitles(_lines(), tmp_path / "a.srt", fmt="srt").read_text(encoding="utf-8")
    assert "1\n00:00:00,500 --> 00:00:02,000\nПривет" in srt and "00:00:03,000 --> 00:00:04,500" in srt
    ass = write_subtitles(_lines(), tmp_path / "a.ass", fmt="ass").read_text(encoding="utf-8")
    assert "Fontname" in ass and "Arial,54" in ass and "Dialogue: 0,0:00:00.50,0:00:02.00,Default" in ass
    assert "Вторая\\Nстрока" in ass and "{тест}" not in ass


def test_render_vertical_with_burned_subs_cyrillic_path(simple_mp4, tmp_path):
    wav = write_wav(tmp_path / "mix.wav", 0.1 * np.sin(2 * np.pi * 440 * np.arange(48000 * 10) / 48000).astype(np.float32), 48000)
    d = tmp_path / "папка с пробелом: тест"
    ass = write_subtitles(_lines(), d / "суб титры.ass", fmt="ass", play_res=(360, 640))
    out = d / "готовое видео.mp4"
    prog = []
    st = RenderSettings(aspect=RenderAspect.V9_16, height=360, subtitles=SubtitleMode.BURN, output_path=str(out))
    res = render_video(simple_mp4, wav, out, st, subtitles_ass=ass, on_progress=prog.append)
    assert res == out and out.exists()
    info = ffprobe_json(out, "-show_streams", "-show_format")
    v = next(s for s in info["streams"] if s["codec_type"] == "video")
    assert (v["width"], v["height"]) == (360, 640) and v["codec_name"] == "h264" and v["pix_fmt"] == "yuv420p"
    assert any(s["codec_type"] == "audio" and s["codec_name"] == "aac" for s in info["streams"])
    assert float(info["format"]["duration"]) == pytest.approx(10.0, abs=0.3)
    assert prog and prog[-1] == 1.0 and prog == sorted(prog)
