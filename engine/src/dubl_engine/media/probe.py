"""Чтение метаданных файла: дорожки, субтитры, главы."""
from __future__ import annotations

from pathlib import Path

from ..models import AudioStreamInfo, Chapter, MediaInfo, SubtitleStreamInfo
from ..util import EngineError, ffprobe_json

TEXT_SUBTITLE_CODECS = {"subrip", "ass", "ssa", "webvtt", "mov_text", "text"}


def _to_float(v, default: float = 0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _to_int(v, default: int = 0) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def _fps(stream: dict) -> float:
    for key in ("avg_frame_rate", "r_frame_rate"):
        raw = stream.get(key) or ""
        try:
            num, _, den = raw.partition("/")
            d = float(den) if den else 1.0
            val = float(num) / d if d else 0.0
        except ValueError:
            continue
        if val > 0:
            return round(val, 3)
    return 0.0


def _lang(tags: dict) -> str | None:
    lang = (tags.get("language") or "").strip().lower()
    return lang if lang and lang != "und" else None


def main_video_index(path: str | Path) -> int:
    """Номер потока (index) настоящего видео: обложка-«видео» (attached_pic) не в счёт."""
    streams = ffprobe_json(Path(path), "-show_streams", "-select_streams", "v").get("streams") or []
    real = [s for s in streams if not (s.get("disposition") or {}).get("attached_pic")]
    pick = (real or streams)
    if not pick:
        raise EngineError("В файле нет видео.")
    return int(pick[0]["index"])


def probe(path: str | Path) -> MediaInfo:
    """ffprobe -show_format -show_streams -show_chapters → MediaInfo."""
    p = Path(path)
    if not p.exists():
        raise EngineError(f"Файл не найден: {p}")
    data = ffprobe_json(p, "-show_format", "-show_streams", "-show_chapters")
    fmt = data.get("format") or {}
    streams = data.get("streams") or []

    video = next(
        (s for s in streams
         if s.get("codec_type") == "video" and not (s.get("disposition") or {}).get("attached_pic")),
        None,
    ) or next((s for s in streams if s.get("codec_type") == "video"), None)

    audio: list[AudioStreamInfo] = []
    subs: list[SubtitleStreamInfo] = []
    for s in streams:
        tags = s.get("tags") or {}
        disp = s.get("disposition") or {}
        kind = s.get("codec_type")
        if kind == "audio":
            audio.append(AudioStreamInfo(
                index=s["index"],
                codec=s.get("codec_name", "unknown"),
                channels=_to_int(s.get("channels")),
                channel_layout=s.get("channel_layout") or None,
                sample_rate=_to_int(s.get("sample_rate")),
                language=_lang(tags),
                title=tags.get("title") or None,
                default=bool(disp.get("default")),
            ))
        elif kind == "subtitle":
            codec = s.get("codec_name", "unknown")
            subs.append(SubtitleStreamInfo(
                index=s["index"],
                codec=codec,
                language=_lang(tags),
                title=tags.get("title") or None,
                is_text=codec in TEXT_SUBTITLE_CODECS,
                default=bool(disp.get("default")),
                forced=bool(disp.get("forced")),
            ))

    duration = _to_float(fmt.get("duration"))
    if duration <= 0 and video:
        duration = _to_float(video.get("duration"))
    if duration <= 0:
        duration = max((_to_float(s.get("duration")) for s in streams), default=0.0)

    chapters = [
        Chapter(
            start=_to_float(c.get("start_time")),
            end=_to_float(c.get("end_time")),
            title=(c.get("tags") or {}).get("title") or None,
        )
        for c in data.get("chapters") or []
    ]

    return MediaInfo(
        path=str(p),
        duration=duration,
        container=fmt.get("format_name", ""),
        width=_to_int(video.get("width")) if video else 0,
        height=_to_int(video.get("height")) if video else 0,
        fps=_fps(video) if video else 0.0,
        video_codec=(video.get("codec_name", "") if video else ""),
        audio_streams=audio,
        subtitle_streams=subs,
        chapters=chapters,
        size_bytes=_to_int(fmt.get("size")),
    )


def pick_original_audio(info: MediaInfo, prefer_lang: str | None = None) -> AudioStreamInfo:
    """Какая дорожка — оригинал: заданный язык → default → первая не русская → первая."""
    tracks = info.audio_streams
    if not tracks:
        raise EngineError("В файле нет звуковых дорожек.")
    if prefer_lang:
        want = prefer_lang.lower()
        for t in tracks:
            if (t.language or "").lower() == want:
                return t
    for t in tracks:
        if t.default:
            return t
    for t in tracks:
        if (t.language or "").lower() != "rus":
            return t
    return tracks[0]


_WEB_VIDEO = {"h264", "vp8", "vp9", "av1"}
_WEB_AUDIO = {"aac", "mp3", "opus", "vorbis", "flac"}


def browser_playable(info: MediaInfo, audio_stream_index: int | None = None) -> tuple[bool, str | None]:
    """Проиграет ли исходник тег <video> в Chrome/Safari/WebView: (да/нет, причина «нет» по-русски).

    Нужен контейнер mp4/mov/webm, видео h264/vp8/vp9/av1, звук aac/mp3/opus/vorbis/flac. Браузер играет только первую
    звуковую дорожку: если оригиналом выбрана другая, нужна копия (proxy), где звук — выбранная дорожка."""
    names = set(info.container.lower().split(","))
    if "matroska" in names:                 # ffprobe зовёт mkv и webm одинаково («matroska,webm»): различаем по расширению
        names = {"webm"} if Path(info.path).suffix.lower() == ".webm" else {"matroska"}
    if not names & {"mov", "mp4", "webm"}:
        return False, f"контейнер «{info.container}» браузер не открывает"
    if info.video_codec.lower() not in _WEB_VIDEO:
        return False, f"видео {info.video_codec} браузер не проигрывает"
    first = info.audio_streams[0] if info.audio_streams else None
    if first is not None and first.codec.lower() not in _WEB_AUDIO:
        return False, f"звук {first.codec} браузер не проигрывает"
    if audio_stream_index is not None and first is not None and first.index != audio_stream_index:
        return False, "оригиналом выбрана не первая звуковая дорожка"
    return True, None
