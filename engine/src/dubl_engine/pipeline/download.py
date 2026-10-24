"""Скачивание видео по ссылке (yt-dlp)."""
from __future__ import annotations

import concurrent.futures
import logging
from collections.abc import Callable
from pathlib import Path

from ..util import EngineError, UnprocessableError, find_binary, redact

log = logging.getLogger(__name__)

PROBE_URL_TIMEOUT_S = 15.0
CANT_CHECK = "Не удалось проверить ссылку (возможно, нет интернета). Узнаем после скачивания: размер, дорожки и субтитры станут видны, когда видео будет на диске."

# Признаки в тексте ошибки yt-dlp (нижний регистр). Порядок важен: сначала «ссылка плохая», потом «нет сети».
_BAD_LINK = ("http error 404", "http error 410", "http error 403", "video unavailable", "video is unavailable",
             "this video is not available", "is not available", "unavailable",
             "private video", "has been removed", "does not exist", "account associated", "been terminated",
             "unsupported url", "not a valid url", "is not a valid", "no video formats", "no video could be found",
             "no video found", "incomplete youtube id", "sign in to confirm", "members-only", "age-restricted")
_NO_NETWORK = ("unable to download", "getaddrinfo", "name or service not known", "nodename nor servname",
               "temporary failure in name resolution", "network is unreachable", "timed out", "timeout",
               "connection refused", "connection reset", "connection aborted", "no route to host", "ssl:", "sslerror",
               "urlopen error",
               "remote end closed", "offline")


def classify_ytdlp_error(message: str) -> tuple[str, str]:
    """Текст ошибки yt-dlp → ('bad_link' | 'network' | 'other', понятное сообщение по-русски)."""
    low = message.lower()
    if "ffmpeg is not installed" in low or "ffmpeg not found" in low:
        return "other", "Не получилось скачать: не найден ffmpeg, без него не склеить картинку и звук. Переустанови Дубль."
    if "unsupported url" in low or "not a valid url" in low:
        return "bad_link", "По этой ссылке нет видео, которое можно скачать. Проверь адрес или вставь ссылку на сам ролик."
    if any(n in low for n in _BAD_LINK):
        if "private" in low or "sign in" in low or "members-only" in low or "age-restricted" in low:
            return "bad_link", "Это видео закрыто (приватное, для своих или с возрастным ограничением) — скачать не получится."
        return "bad_link", "Видео по этой ссылке недоступно: его нет, оно удалено или закрыто. Проверь ссылку."
    if any(n in low for n in _NO_NETWORK):
        return "network", CANT_CHECK
    return "other", "Не получилось открыть видео по этой ссылке. Проверь адрес или попробуй позже."


def download_video(url: str, dest_dir: str | Path, *,
                   on_progress: Callable[[float], None] | None = None) -> tuple[Path, str]:
    """→ (путь к файлу, название ролика)."""
    try:
        import yt_dlp
    except ImportError as e:  # pragma: no cover
        raise EngineError("В программе нет загрузчика видео — скачивание по ссылке недоступно. Переустанови Дубль.") from e
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)

    def hook(d: dict) -> None:
        if on_progress and d.get("status") == "downloading":
            total = d.get("total_bytes") or d.get("total_bytes_estimate") or 0
            if total:
                on_progress(min(float(d.get("downloaded_bytes", 0)) / total, 0.99))

    # yt-dlp ищет ffmpeg только в PATH, а в собранном приложении он лежит в DUBL_FFMPEG_DIR. Без этого ролики, где
    # видео и звук идут отдельными потоками (почти весь YouTube в HD), не склеиваются: «ffmpeg is not installed».
    ffmpeg_dir = str(Path(find_binary("ffmpeg")).parent)
    opts = {"outtmpl": str(dest / "%(id)s.%(ext)s"), "noplaylist": True, "quiet": True, "no_warnings": True,
            "progress_hooks": [hook], "merge_output_format": "mkv", "ffmpeg_location": ffmpeg_dir}
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=True)
            reqs = info.get("requested_downloads") or []
            path = Path(reqs[0]["filepath"]) if reqs and reqs[0].get("filepath") else Path(ydl.prepare_filename(info))
    except yt_dlp.utils.DownloadError as e:
        raw = redact(str(e)[-500:])
        log.warning("yt-dlp не скачал %s: %s", redact(url), raw)
        kind, text = classify_ytdlp_error(str(e))
        if kind == "network":
            text = "Не удалось скачать видео: нет связи с сайтом. Проверь интернет и попробуй ещё раз."
        raise EngineError(text, debug=raw) from None
    if on_progress:
        on_progress(1.0)
    return path, str(info.get("title") or path.stem)


def _extract_meta(url: str, timeout: float):
    import yt_dlp
    opts = {"quiet": True, "no_warnings": True, "noplaylist": True, "skip_download": True, "socket_timeout": timeout,
            "retries": 0, "extractor_retries": 0, "ignoreconfig": True}
    with yt_dlp.YoutubeDL(opts) as ydl:
        return ydl.extract_info(url, download=False)


def probe_url(url: str, *, timeout: float = PROBE_URL_TIMEOUT_S) -> dict:
    """Что известно о ролике по ссылке БЕЗ скачивания видео (yt-dlp, только метаданные, не дольше `timeout` секунд).

    → {"known", "title", "duration", "has_subtitles", "subtitle_langs", "message"}.
    * Ссылка рабочая → known=True.
    * Видео нет / оно закрыто / сайт не поддерживается / это не видео → UnprocessableError (API: 422, текст по-русски).
    * Нет yt-dlp, нет сети, таймаут, неожиданный сбой → known=False и message «не удалось проверить ссылку» (не ошибка)."""
    out: dict = {"known": False, "title": None, "duration": None, "has_subtitles": None, "subtitle_langs": [],
                 "message": CANT_CHECK}
    try:
        import yt_dlp  # noqa: F401
    except ImportError:
        return out
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="dubl-probe-url")
    try:
        info = pool.submit(_extract_meta, url, timeout).result(timeout=timeout + 1.0)
    except concurrent.futures.TimeoutError:
        log.info("probe_url: таймаут %s с для %s", timeout, redact(url))
        return out
    except Exception as e:  # noqa: BLE001 — разбираем по тексту; всё незнакомое = «не удалось проверить»
        kind, text = classify_ytdlp_error(str(e))
        log.info("probe_url: %s → %s: %s", redact(url), kind, redact(str(e)[-300:]))
        if kind == "bad_link":
            raise UnprocessableError(text, debug=redact(str(e)[-500:])) from None
        return out
    finally:
        pool.shutdown(wait=False)
    if not isinstance(info, dict):
        raise UnprocessableError("По этой ссылке нет видео, которое можно скачать.")
    if info.get("_type") == "playlist":
        entries = [e for e in (info.get("entries") or []) if isinstance(e, dict)]
        if not entries:
            raise UnprocessableError("По этой ссылке нет видео, которое можно скачать.")
        info = entries[0]
    manual = sorted((info.get("subtitles") or {}).keys())
    auto = sorted((info.get("automatic_captions") or {}).keys())
    dur = info.get("duration")
    out.update(known=True, title=str(info["title"]) if info.get("title") else None,
               duration=float(dur) if isinstance(dur, (int, float)) and dur > 0 else None,
               has_subtitles=bool(manual or auto), subtitle_langs=manual, message=None)
    if not manual and auto:
        out["message"] = "Есть только автоматические субтитры — для дубляжа лучше распознать речь."
    return out
