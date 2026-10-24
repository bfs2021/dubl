"""Извлечение звука, центрального канала, субтитров и вырезка сцены."""
from __future__ import annotations

import re
import subprocess
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Literal

from ..util import _NO_WINDOW, EngineError, ffmpeg_failure, ffprobe_json, find_binary, run_ffmpeg
from .probe import TEXT_SUBTITLE_CODECS, main_video_index


def _stream_info(src: str | Path, stream_index: int, kind: str) -> dict:
    data = ffprobe_json(src, "-show_streams")
    for s in data.get("streams") or []:
        if s.get("index") == stream_index:
            if s.get("codec_type") != kind:
                names = {"audio": "звуковой", "subtitle": "субтитровой"}
                raise EngineError(f"Поток {stream_index} не является {names.get(kind, kind)} дорожкой.")
            return s
    raise EngineError(f"В файле нет потока с номером {stream_index}.")


def _range_args(start: float | None, end: float | None) -> list[str]:
    if start is not None and end is not None and end <= start:
        raise EngineError("Конец фрагмента должен быть позже начала.")
    args: list[str] = []
    if start is not None:
        args += ["-ss", f"{max(start, 0.0):.3f}"]
    if end is not None:
        args += ["-to", f"{end:.3f}"]
    return args


_CODECS = {"f32": "pcm_f32le", "s16": "pcm_s16le"}


def extract_audio(src: str | Path, stream_index: int, out_wav: str | Path, *, sample_rate: int = 48000,
                  mono: bool = False, stereo: bool = False, start: float | None = None, end: float | None = None,
                  sample_fmt: Literal["f32", "s16"] = "f32") -> Path:
    """Дорожка → WAV (по умолчанию float32; "s16" вдвое компактнее — для длинных файлов разбора).
    start/end — секунды исходника."""
    _stream_info(src, stream_index, "audio")
    out = Path(out_wav)
    out.parent.mkdir(parents=True, exist_ok=True)
    args: list[str | Path] = [*_range_args(start, end), "-i", Path(src), "-map", f"0:{stream_index}", "-vn",
                              "-ar", str(sample_rate)]
    if mono:
        args += ["-ac", "1"]
    elif stereo:
        # 5.1/7.1 → стерео стандартным даунмиксом ffmpeg (центр и тылы входят в L/R).
        # Иначе потом брались только FL/FR, и центральный канал с диалогом терялся.
        args += ["-ac", "2"]
    args += ["-c:a", _CODECS[sample_fmt], out]
    run_ffmpeg(args)
    return out


_NUMERIC_LAYOUT = re.compile(r"^(\d+)\.(\d+)(?:\.(\d+))?(?:\((.*)\))?$")


def has_front_center(channel_layout: str | None, channels: int = 0) -> bool:
    """Есть ли в раскладке канал FC (центр)."""
    if not channel_layout:
        return channels in (6, 8)   # раскладка не указана: 5.1 / 7.1 по числу каналов
    lay = channel_layout.strip()
    if lay in ("mono", "stereo", "downmix", "quad", "quad(side)"):
        return False
    if "FC" in lay.split("+"):
        return True
    if lay in ("hexagonal", "octagonal"):
        return True
    m = _NUMERIC_LAYOUT.match(lay)
    if m:
        front = int(m.group(1))
        variant = m.group(4) or ""
        # 6.0(front), 6.1(front): вместо центра — FLC/FRC
        return front >= 3 and variant != "front"
    return False


def extract_center_channel(src: str | Path, stream_index: int, out_wav: str | Path, *,
                           sample_rate: int = 48000, start: float | None = None,
                           end: float | None = None, sample_fmt: Literal["f32", "s16"] = "f32") -> Path | None:
    """Центральный канал (FC) в mono; None, если у дорожки нет центра."""
    s = _stream_info(src, stream_index, "audio")
    if not has_front_center(s.get("channel_layout"), int(s.get("channels") or 0)):
        return None
    out = Path(out_wav)
    out.parent.mkdir(parents=True, exist_ok=True)
    run_ffmpeg([*_range_args(start, end), "-i", Path(src), "-map", f"0:{stream_index}", "-vn",
                "-af", "pan=mono|c0=FC", "-ar", str(sample_rate), "-ac", "1", "-c:a", _CODECS[sample_fmt], out])
    return out


def extract_subtitles(src: str | Path, stream_index: int, out_path: str | Path) -> Path:
    """Текстовые субтитры → .ass (для ass/ssa) или .srt. Картинки (PGS, VobSub) не читаем."""
    s = _stream_info(src, stream_index, "subtitle")
    codec = s.get("codec_name", "")
    if codec not in TEXT_SUBTITLE_CODECS:
        raise EngineError(
            f"Эти субтитры — картинки ({codec}), прочитать текст нельзя. "
            "Выбери другие субтитры или распознай речь заново.")
    out = Path(out_path)
    if codec in ("ass", "ssa"):
        out = out.with_suffix(".ass")
        codec_args = ["-c:s", "copy"] if codec == "ass" else ["-c:s", "ass"]
    else:
        out = out.with_suffix(".srt")
        codec_args = ["-c:s", "copy"] if codec == "subrip" else ["-c:s", "srt"]
    out.parent.mkdir(parents=True, exist_ok=True)
    run_ffmpeg(["-i", Path(src), "-map", f"0:{stream_index}", *codec_args, out])
    return out


def _cfr_rate(stream: dict) -> str:
    """Постоянная частота кадров для вырезки: «24/1», «30000/1001». Берём r_frame_rate (ровная сетка), иначе средний."""
    for key in ("r_frame_rate", "avg_frame_rate"):
        raw = str(stream.get(key) or "")
        try:
            num, _, den = raw.partition("/")
            n, d = int(num), int(den or 1)
        except ValueError:
            continue
        if n > 0 and d > 0 and 1.0 <= n / d <= 120.0:
            return f"{n}/{d}"
    return "25/1"


def cut_scene_video(src: str | Path, start: float, end: float, out_mp4: str | Path, *,
                    audio_stream_index: int) -> Path:
    """Точная вырезка сцены: h264 crf 18 + aac, выбранная звуковая дорожка, разрешение прежнее.

    Длина и синхрон: видео приводится к постоянной частоте кадров и начинается ровно с 0 (первый кадр после точной
    перемотки повторяется до нуля, а не сдвигает всю картинку на кадр), оба потока обрезаются по `-t` — длина клипа
    равна длине сцены, звук и видео не расходятся (раньше видео «стартовало» с 0,04 с, а звук тянулся до 10,5 с)."""
    if end <= start:
        raise EngineError("Конец сцены должен быть позже начала.")
    _stream_info(src, audio_stream_index, "audio")
    out = Path(out_mp4)
    out.parent.mkdir(parents=True, exist_ok=True)
    vi = main_video_index(src)
    vs = next((s for s in ffprobe_json(src, "-show_streams", "-select_streams", "v").get("streams") or []
               if s.get("index") == vi), {})
    rate = _cfr_rate(vs)
    # crop до чётных сторон: libx264 + yuv420p не принимает нечётные размеры (иначе сцена из такого видео не вырежется)
    run_ffmpeg(["-ss", f"{max(start, 0.0):.3f}", "-i", Path(src), "-t", f"{end - max(start, 0.0):.3f}",
                "-map", f"0:{vi}", "-map", f"0:{audio_stream_index}", "-sn", "-dn",
                "-map_chapters", "-1", "-map_metadata", "-1",       # главы и теги фильма в клип не попадают
                "-vf", f"fps=fps={rate}:start_time=0,crop=trunc(iw/2)*2:trunc(ih/2)*2",
                "-c:v", "libx264", "-crf", "18", "-preset", "veryfast", "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-b:a", "192k",
                "-movflags", "+faststart", out])
    return out


def run_ffmpeg_progress(args: list[str | Path], duration: float | None,
                        on_progress: Callable[[float], None] | None = None) -> None:
    """ffmpeg с прогрессом 0..1 (по -progress). Исключение из on_progress (отмена задачи) убивает процесс."""
    cmd = [find_binary("ffmpeg"), "-hide_banner", "-loglevel", "error", "-nostdin", "-y", "-progress", "pipe:1",
           *(str(a if not isinstance(a, Path) else a.absolute()) for a in args)]
    p = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                         encoding="utf-8", errors="replace", creationflags=_NO_WINDOW)
    err: list[str] = []
    t = threading.Thread(target=lambda: err.append(p.stderr.read() if p.stderr else ""), daemon=True)
    t.start()
    try:
        assert p.stdout is not None
        for row in p.stdout:
            m = re.match(r"out_time_(?:us|ms)=(\d+)", row.strip())
            if m and duration and on_progress:
                on_progress(min(0.99, int(m.group(1)) / 1e6 / duration))
        rc = p.wait()
    except BaseException:
        p.kill()
        p.wait()
        raise
    t.join(timeout=5)
    if rc != 0:
        raise ffmpeg_failure(err[0] if err else "", rc)


def make_proxy(src: str | Path, out_mp4: str | Path, *, audio_stream_index: int | None = None, height: int = 540,
               on_progress: Callable[[float], None] | None = None) -> Path:
    """Лёгкая копия для плеера, который не тянет исходник (mkv, ac3/dts, hevc, 4K): h264 crf 26 + aac 128k stereo,
    высота не больше `height`, faststart, без глав и тегов. Пишет во временный файл и подменяет им результат."""
    src = Path(src)
    vi = main_video_index(src)
    if audio_stream_index is None:
        audio = [s for s in ffprobe_json(src, "-show_streams").get("streams", []) if s.get("codec_type") == "audio"]
        audio_stream_index = int(audio[0]["index"]) if audio else None
    else:
        _stream_info(src, audio_stream_index, "audio")
    try:
        duration = float(ffprobe_json(src, "-show_format").get("format", {}).get("duration") or 0) or None
    except (TypeError, ValueError):
        duration = None
    out = Path(out_mp4)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(f".{out.stem}.part{out.suffix}")
    maps: list[str | Path] = ["-map", f"0:{vi}"]
    codec: list[str | Path] = []
    if audio_stream_index is not None:
        maps += ["-map", f"0:{audio_stream_index}"]
        codec = ["-c:a", "aac", "-b:a", "128k", "-ac", "2"]
    else:
        maps += ["-an"]
    try:
        run_ffmpeg_progress(
            ["-i", src, *maps, "-sn", "-dn", "-map_chapters", "-1", "-map_metadata", "-1",
             "-vf", f"scale=-2:'min({height},ih)':flags=bicubic,setsar=1", "-c:v", "libx264", "-crf", "26",
             "-preset", "veryfast", "-pix_fmt", "yuv420p", *codec, "-movflags", "+faststart", "-f", "mp4", tmp],
            duration, on_progress)
        tmp.replace(out)
    finally:
        tmp.unlink(missing_ok=True)
    if on_progress:
        on_progress(1.0)
    return out
