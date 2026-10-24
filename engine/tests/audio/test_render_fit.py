"""Вертикальный и квадратный форматы: размеры (ffprobe), размытые поля (blur) и кроп (crop)."""
from __future__ import annotations

import subprocess

import numpy as np
import pytest

from dubl_engine.audio.io import write_wav
from dubl_engine.audio.render import render_video, write_subtitles
from dubl_engine.models import RenderAspect, RenderFit, RenderSettings, SceneLine, SubtitleMode
from dubl_engine.util import ffprobe_json, find_binary


@pytest.fixture(scope="module")
def wav(tmp_path_factory):
    d = tmp_path_factory.mktemp("fitwav")
    return write_wav(d / "m.wav", (0.1 * np.sin(2 * np.pi * 440 * np.arange(48000 * 10) / 48000)).astype(np.float32), 48000)


def _size(path):
    v = next(s for s in ffprobe_json(path, "-show_streams")["streams"] if s["codec_type"] == "video")
    return v["width"], v["height"], v


def _gray(path, t=2.0) -> np.ndarray:
    """Кадр как массив яркостей (h, w)."""
    w, h, _ = _size(path)
    raw = subprocess.run([find_binary("ffmpeg"), "-v", "error", "-ss", str(t), "-i", str(path), "-frames:v", "1",
                          "-vf", "format=gray", "-f", "rawvideo", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.uint8).reshape(h, w)


def _render(simple_mp4, wav, tmp_path, name, **kw):
    out = tmp_path / name
    rs = RenderSettings(output_path=str(out), **kw)
    render_video(simple_mp4, wav, out, rs)
    return out


@pytest.mark.parametrize("aspect,height,expect", [
    (RenderAspect.V9_16, None, (1080, 1920)),
    (RenderAspect.V9_16, 1080, (1080, 1920)),
    (RenderAspect.V9_16, 720, (720, 1280)),
    (RenderAspect.SQ1_1, None, (1080, 1080)),
    (RenderAspect.SQ1_1, 720, (720, 720)),
])
@pytest.mark.parametrize("fit", [RenderFit.BLUR, RenderFit.CROP])
def test_sizes_are_right_and_even(simple_mp4, wav, tmp_path, aspect, height, expect, fit):
    out = _render(simple_mp4, wav, tmp_path, "o.mp4", aspect=aspect, height=height, fit=fit)
    w, h, v = _size(out)
    assert (w, h) == expect and w % 2 == 0 and h % 2 == 0
    assert v["pix_fmt"] == "yuv420p" and v.get("sample_aspect_ratio", "1:1") in ("1:1", "0:1")


def test_default_fit_is_blur():
    assert RenderSettings(output_path="x").fit == RenderFit.BLUR


def test_blur_vertical_has_blurred_copy_on_top_and_bottom(simple_mp4, wav, tmp_path):
    out = _render(simple_mp4, wav, tmp_path, "blur.mp4", aspect=RenderAspect.V9_16, height=720, fit=RenderFit.BLUR)
    g = _gray(out)
    assert g.shape == (1280, 720)
    top, bottom, mid = g[:200], g[-200:], g[500:780]
    assert top.mean() > 25 and bottom.mean() > 25                    # сверху и снизу не чёрные поля
    sharp = lambda a: np.abs(np.diff(a.astype(int), axis=1)).max()   # noqa: E731 — самый резкий перепад между соседними точками
    assert sharp(top) < 40 < sharp(mid)                              # и размытые: резких границ нет, в кадре — есть
    # кадр целиком по ширине: 640×360 → 720×405 по центру (поля по ~437 строк)
    assert g[437 + 20:437 + 385].std() > 20


def test_blur_square_and_crop_differ(simple_mp4, wav, tmp_path):
    blur = _gray(_render(simple_mp4, wav, tmp_path, "b.mp4", aspect=RenderAspect.SQ1_1, height=720, fit=RenderFit.BLUR))
    crop = _gray(_render(simple_mp4, wav, tmp_path, "c.mp4", aspect=RenderAspect.SQ1_1, height=720, fit=RenderFit.CROP))
    assert blur.shape == crop.shape == (720, 720)
    assert blur[:100].mean() > 25                                    # blur: сверху размытая копия, не чёрное
    assert np.abs(blur.astype(int) - crop.astype(int)).mean() > 5     # режимы дают разную картинку


def test_wide_source_not_cropped_in_blur_but_cropped_in_crop(tmp_path, wav):
    """Широкий 2.4:1: в crop теряются края (виден узкий центр), в blur — кадр целиком (левый край на месте)."""
    src = tmp_path / "wide.mp4"
    subprocess.run([find_binary("ffmpeg"), "-v", "error", "-y", "-f", "lavfi", "-i", "color=c=black:s=960x400:r=25:d=3",
                    "-vf", "drawbox=x=0:y=0:w=120:h=400:color=white:t=fill", "-pix_fmt", "yuv420p", str(src)], check=True)
    blur = _gray(_render(src, wav, tmp_path, "b.mp4", aspect=RenderAspect.V9_16, height=360, fit=RenderFit.BLUR), t=1.0)
    crop = _gray(_render(src, wav, tmp_path, "c.mp4", aspect=RenderAspect.V9_16, height=360, fit=RenderFit.CROP), t=1.0)
    assert blur.shape == crop.shape == (640, 360)
    mid = slice(280, 360)                                            # середина по высоте — здесь кадр
    assert blur[mid, :20].mean() > 150                               # blur: белая полоса слева видна (кадр целиком)
    assert crop[mid, :20].mean() < 30                                # crop: центр, левый край обрезан


def test_original_and_16x9_ignore_fit(simple_mp4, wav, tmp_path):
    a = _render(simple_mp4, wav, tmp_path, "a.mp4", aspect=RenderAspect.H16_9, height=360, fit=RenderFit.BLUR)
    b = _render(simple_mp4, wav, tmp_path, "b.mp4", aspect=RenderAspect.H16_9, height=360, fit=RenderFit.CROP)
    assert _size(a)[:2] == _size(b)[:2] == (640, 360)
    assert np.abs(_gray(a).astype(int) - _gray(b).astype(int)).mean() < 1
    o = _render(simple_mp4, wav, tmp_path, "o.mp4", aspect=RenderAspect.ORIGINAL, height=None)
    assert _size(o)[:2] == (640, 360)


def test_blur_with_burned_subtitles_and_progress(simple_mp4, wav, tmp_path):
    lines = [SceneLine(line_id="l", speaker_id=None, start_rel=1, end_rel=4, original="x", text="Привет, мир")]
    ass = write_subtitles(lines, tmp_path / "s.ass", fmt="ass", play_res=(720, 1280))
    out = tmp_path / "s.mp4"
    prog = []
    rs = RenderSettings(aspect=RenderAspect.V9_16, height=720, subtitles=SubtitleMode.BURN, output_path=str(out))
    render_video(simple_mp4, wav, out, rs, subtitles_ass=ass, on_progress=prog.append)
    assert _size(out)[:2] == (720, 1280) and prog[-1] == 1.0
    plain = _render(simple_mp4, wav, tmp_path, "p.mp4", aspect=RenderAspect.V9_16, height=720)
    # титры вшиты: внизу кадра картинка отличается от версии без субтитров
    assert np.abs(_gray(out, 2.0)[-260:-100].astype(int) - _gray(plain, 2.0)[-260:-100].astype(int)).mean() > 0.5


def test_blur_odd_and_anamorphic_sources_still_even(tmp_path, wav):
    src = tmp_path / "odd.mp4"
    subprocess.run([find_binary("ffmpeg"), "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=s=480x360:r=25:d=3",
                    "-vf", "setsar=4/3", "-pix_fmt", "yuv420p", str(src)], check=True)
    out = _render(src, wav, tmp_path, "o.mp4", aspect=RenderAspect.V9_16, height=360)
    assert _size(out)[:2] == (360, 640)
