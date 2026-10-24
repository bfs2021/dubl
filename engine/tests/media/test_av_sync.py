"""Длина и синхрон звука и картинки в сцене и в склейке сцен.

Баг слепого теста №3: в готовом ролике звук 10,458 с, видео 10,292 с. Причины: вырезка по точному времени давала видео,
которое «стартовало» с 0,04 с, а звук — с нуля; `-shortest` резал хвост кадров. Тесты на синтетическом видео, где
вспышка и щелчок в один момент, и сцена начинается вне начала GOP.
"""
from __future__ import annotations

import subprocess

import numpy as np
import pytest

from dubl_engine.audio.io import read_wav, write_wav
from dubl_engine.audio.render import (
    concat_videos,
    fit_length,
    join_audio,
    mux_video_audio,
    render_video,
    video_duration,
)
from dubl_engine.media.extract import cut_scene_video, extract_audio
from dubl_engine.models import RenderAspect, RenderSettings
from dubl_engine.util import ffprobe_json, find_binary, run_ffmpeg

TOL = 0.025          # 25 мс (меньше кадра при 24–30 fps)
FLASHES = (3.2, 6.5)     # вспышка и щелчок в эти моменты исходника
SR = 48000


def _ff(*args):
    subprocess.run([find_binary("ffmpeg"), "-hide_banner", "-loglevel", "error", "-y", *map(str, args)], check=True)


@pytest.fixture(scope="module", params=[("25", "mkv"), ("24000/1001", "mp4")], ids=["25fps-mkv", "23.976fps-mp4"])
def flash_src(request, tmp_path_factory):
    """10 с, длинный GOP (ключевые кадры 0 и 5 с), B-кадры. Белая вспышка на 2 кадра и щелчок 1 кГц в моменты FLASHES."""
    rate, ext = request.param
    out = tmp_path_factory.mktemp("flash") / f"flash.{ext}"
    en = "+".join(f"between(t,{t},{t + 0.08})" for t in FLASHES)
    clk = "+".join(f"between(t,{t},{t + 0.02})" for t in FLASHES)
    _ff("-f", "lavfi", "-i", f"color=c=black:s=320x240:r={rate}:d=10,drawbox=x=0:y=0:w=iw:h=ih:color=white:t=fill:enable='{en}'",
        "-f", "lavfi", "-i", f"aevalsrc='({clk})*0.9*sin(2*PI*1000*t)':s={SR}:d=10:c=stereo",
        "-c:v", "libx264", "-g", "125", "-keyint_min", "125", "-sc_threshold", "0", "-bf", "2", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k", out)
    return out


def _streams(path) -> dict:
    return {s["codec_type"]: s for s in ffprobe_json(path, "-show_streams")["streams"]}


def _end(stream) -> float:
    return float(stream.get("start_time") or 0) + float(stream["duration"])


def _flash_times(path) -> list[float]:
    """Времена начала вспышек (первый яркий кадр каждой серии) по pts кадров."""
    frames = ffprobe_json(path, "-show_frames", "-select_streams", "v", "-show_entries", "frame=pts_time,best_effort_timestamp_time")
    pts = sorted(float(f.get("pts_time") or f.get("best_effort_timestamp_time")) for f in frames["frames"])
    raw = subprocess.run([find_binary("ffmpeg"), "-v", "error", "-i", str(path), "-vf", "scale=8:8,format=gray",
                          "-fps_mode", "passthrough", "-f", "rawvideo", "-"], capture_output=True, check=True).stdout
    lum = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 64).mean(axis=1)
    assert len(lum) == len(pts)
    bright = [t for t, v in zip(pts, lum) if v > 128]
    groups: list[list[float]] = []
    for t in bright:
        if not groups or t - groups[-1][-1] > 0.3:
            groups.append([t])
        else:
            groups[-1].append(t)
    return [g[0] for g in groups]


def _click_times(path, tmp_path) -> list[float]:
    wav = tmp_path / "dec.wav"
    run_ffmpeg(["-i", path, "-vn", "-ac", "1", "-ar", str(SR), "-c:a", "pcm_f32le", wav])
    x, sr = read_wav(wav)
    x = np.asarray(x).reshape(len(x), -1).max(axis=1) if np.ndim(x) > 1 else np.asarray(x)
    hot = np.nonzero(np.abs(x) > 0.3)[0] / sr
    out: list[float] = []
    for t in hot:
        if not out or t - out[-1] > 0.3:
            out.append(float(t))
    return out


def _assert_sync(path, tmp_path, expected: list[float]) -> None:
    flashes, clicks = _flash_times(path), _click_times(path, tmp_path)
    assert len(flashes) == len(clicks) == len(expected), (flashes, clicks, expected)
    for f, c, e in zip(flashes, clicks, expected, strict=True):
        assert abs(f - c) <= TOL, f"вспышка {f:.3f} с, щелчок {c:.3f} с: разъехались (ждали {e:.3f})"
        assert abs(c - e) <= TOL, f"щелчок на {c:.3f} с, ждали {e:.3f} с"


def _scene_audio(src, a, b, tmp_path, name="mix.wav"):
    return extract_audio(src, 1, tmp_path / name, sample_rate=SR, stereo=True, start=a, end=b)


@pytest.mark.parametrize("start,end", [(1.37, 4.9), (2.123, 6.77), (5.31, 9.9)])
def test_clip_and_render_same_length_and_in_sync(flash_src, tmp_path, start, end):
    """Вырезка не по ключевому кадру и не по сетке кадров: длины потоков равны, вспышка и щелчок совпадают."""
    clip = cut_scene_video(flash_src, start, end, tmp_path / "clip.mp4", audio_stream_index=1)
    s = _streams(clip)
    assert abs(float(s["video"]["start_time"])) <= 0.001, "видео клипа должно начинаться с нуля"
    assert abs(_end(s["video"]) - (end - start)) <= TOL
    assert abs(_end(s["audio"]) - _end(s["video"])) <= TOL
    want = [t - start for t in FLASHES if start <= t - 0.0 < end - 0.1]
    _assert_sync(clip, tmp_path, want)

    mix = _scene_audio(flash_src, start, end, tmp_path)
    out = render_video(clip, mix, tmp_path / "out.mp4", RenderSettings(output_path=str(tmp_path / "out.mp4")))
    so = _streams(out)
    assert abs(_end(so["audio"]) - _end(so["video"])) <= TOL, (_end(so["audio"]), _end(so["video"]))
    assert abs(_end(so["video"]) - (end - start)) <= 0.05
    _assert_sync(out, tmp_path, want)


def test_render_other_aspects_same_length(flash_src, tmp_path):
    clip = cut_scene_video(flash_src, 1.37, 4.9, tmp_path / "clip.mp4", audio_stream_index=1)
    mix = _scene_audio(flash_src, 1.37, 4.9, tmp_path)
    for name, aspect in (("v", RenderAspect.V9_16), ("sq", RenderAspect.SQ1_1)):
        out = render_video(clip, mix, tmp_path / f"{name}.mp4", RenderSettings(aspect=aspect, height=360,
                                                                              output_path=str(tmp_path / f"{name}.mp4")))
        so = _streams(out)
        assert abs(_end(so["audio"]) - _end(so["video"])) <= TOL
        _assert_sync(out, tmp_path, [3.2 - 1.37])


def test_render_with_longer_or_shorter_audio_keeps_video_length(flash_src, tmp_path):
    """Звук длиннее/короче картинки: ролик всё равно ровно по длине видео (тишина в хвосте / обрезка)."""
    clip = cut_scene_video(flash_src, 1.37, 4.9, tmp_path / "clip.mp4", audio_stream_index=1)
    vlen = _end(_streams(clip)["video"])
    base = _scene_audio(flash_src, 1.37, 4.9, tmp_path)
    x, sr = read_wav(base)
    for name, n in (("long", len(x) + int(0.4 * sr)), ("short", len(x) - int(0.3 * sr))):
        wav = write_wav(tmp_path / f"{name}.wav", fit_length(x, n), sr)
        out = render_video(clip, wav, tmp_path / f"{name}.mp4", RenderSettings(output_path=str(tmp_path / f"{name}.mp4")))
        so = _streams(out)
        assert abs(_end(so["video"]) - vlen) <= TOL
        assert abs(_end(so["audio"]) - _end(so["video"])) <= TOL, name


def test_legacy_clip_starting_after_zero_is_aligned(flash_src, tmp_path):
    """Клипы, вырезанные старой версией (видео с 0,04 с, звук с нуля), при рендере выравниваются, а не рассинхронизируются."""
    start, end = 1.37, 4.9
    legacy = tmp_path / "legacy.mp4"
    run_ffmpeg(["-ss", f"{start:.3f}", "-to", f"{end:.3f}", "-i", flash_src, "-map", "0:v:0", "-map", "0:a:0",
                "-vf", "crop=trunc(iw/2)*2:trunc(ih/2)*2", "-c:v", "libx264", "-crf", "18", "-preset", "veryfast",
                "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", legacy])
    mix = _scene_audio(flash_src, start, end, tmp_path)
    out = render_video(legacy, mix, tmp_path / "out.mp4", RenderSettings(output_path=str(tmp_path / "out.mp4")))
    so = _streams(out)
    assert abs(_end(so["audio"]) - _end(so["video"])) <= TOL
    _assert_sync(out, tmp_path, [3.2 - start])


def test_joined_scenes_equal_length_and_in_sync(flash_src, tmp_path):
    """Склейка двух сцен: общая длина звука = длине видео, вспышки и щелчки совпадают в обеих сценах."""
    scenes = [(1.37, 4.2), (5.31, 8.0)]
    videos, parts = [], []
    for i, (a, b) in enumerate(scenes):
        clip = cut_scene_video(flash_src, a, b, tmp_path / f"c{i}.mp4", audio_stream_index=1)
        mix = _scene_audio(flash_src, a, b, tmp_path, f"m{i}.wav")
        v = render_video(clip, mix, tmp_path / f"s{i}.mp4", RenderSettings(output_path=str(tmp_path / f"s{i}.mp4")))
        videos.append(v)
        x, sr = read_wav(mix)
        parts.append(fit_length(x, round(video_duration(v) * sr)))
    joined_wav = write_wav(tmp_path / "j.wav", join_audio(parts, SR), SR)
    out = mux_video_audio(concat_videos(videos, tmp_path / "j.mp4"), joined_wav, tmp_path / "final.mp4")
    so = _streams(out)
    total = sum(video_duration(v) for v in videos)
    assert abs(_end(so["video"]) - total) <= TOL
    assert abs(_end(so["audio"]) - _end(so["video"])) <= TOL, (_end(so["audio"]), _end(so["video"]))
    first = video_duration(videos[0])
    _assert_sync(out, tmp_path, [3.2 - scenes[0][0], first + 6.5 - scenes[1][0]])
