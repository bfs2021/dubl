"""Общие утилиты: пути, вызов ffmpeg/ffprobe, тестовые данные.

Все модули вызывают ffmpeg ТОЛЬКО через run_ffmpeg / ffprobe_json отсюда —
так проще поменять способ поставки бинарника (в сборке Tauri он лежит рядом с приложением).
"""
from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import threading
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path

from platformdirs import user_data_dir

log = logging.getLogger(__name__)


class EngineError(RuntimeError):
    """Ошибка с текстом для пользователя (по-русски).

    debug — необязательный сырой текст для отладки (хвост stderr ffmpeg и т. п.; секреты спрятаны). Пользователю
    его показывать не надо: API отдаёт его отдельным полем `debug`, задача — `Job.error_debug`."""

    def __init__(self, message: str = "", *, debug: str | None = None):
        super().__init__(message)
        self.debug = debug


class MlMissingError(EngineError):
    """Не установлены нейросети (torch, whisper, demucs…) — в сборке лёгкий движок. UI по `Job.error_code == "ml_missing"`
    предлагает «Скачать нейросети» и «Разобрать снова»."""
    code = "ml_missing"

    def __init__(self, message: str = "", *, debug: str | None = None):
        super().__init__(message or ML_MISSING_MSG, debug=debug)


ML_MISSING_MSG = "Не установлены нейросети — без них нельзя распознавать речь и отделять голоса. Скачай их один раз (~1,3 ГБ)."


class ConflictError(EngineError):
    """Действие сейчас невозможно (например, сцена ещё готовится). API отвечает 409."""


class UnprocessableError(EngineError):
    """Запрос понятен, но то, что в нём (например, ссылка), не подходит. API отвечает 422."""


_SECRET_RE = re.compile(
    r"(?i)((?:api[_-]?key|token|authorization|bearer|x-api-key|x-goog-api-key|password|secret)[\"']?\s*[:=]?\s*[\"']?)"
    r"([A-Za-z0-9._~+/=-]{8,})|\b(sk-[A-Za-z0-9_-]{10,}|hf_[A-Za-z0-9]{10,}|AIza[0-9A-Za-z_-]{20,})")


def redact(text: str) -> str:
    """Спрятать в тексте то, что похоже на ключ/токен (для логов и сообщений об ошибках)."""
    return _SECRET_RE.sub(lambda m: (m.group(1) + "***") if m.group(1) else "***", text)


# «Это не видео» (чужой формат: текст, картинка, мусор) и «видео повреждено» (формат узнали, но файл обрезан или битый) —
# разные беды с разным действием, поэтому и тексты разные.
NOT_VIDEO_MSG = "Это не видео: программа не узнала формат файла. Выбери видеофайл (mp4, mkv, mov, avi…)."
BROKEN_FILE_MSG = ("Видео повреждено или скачано не до конца — прочитать его не получилось. "
                   "Скачай файл ещё раз или выбери другой.")

# (признаки в stderr ffmpeg/ffprobe в нижнем регистре, понятный текст). Первое совпадение выигрывает.
_FFMPEG_RULES: list[tuple[tuple[str, ...], str]] = [
    (("no such file or directory",), "Файл не найден: его могли переместить или удалить."),
    (("permission denied",), "Нет доступа к файлу или папке. Проверь права или выбери другое место."),
    (("no space left on device",), "На диске не хватает места."),
    (("cannot allocate memory", "out of memory"), "Не хватило памяти, чтобы обработать видео."),
    # формат узнан, но файл обрезан или битый — раньше, чем «неизвестный формат»: у обрезанного mp4 в stderr есть оба признака
    (("moov atom not found", "truncated", "end of file", "invalid nal unit", "error while decoding stream",
      "could not find codec parameters", "partial file"), BROKEN_FILE_MSG),
    (("invalid data found when processing input", "does not contain any stream", "header missing", "unknown format",
      "ebml header parsing failed", "invalid png signature", "not a valid"), NOT_VIDEO_MSG),
    (("encoder not found", "unknown encoder", "no such filter", "unrecognized option"),
     "Программе не хватает кодека для этого файла. Обнови Дубль."),
]


def ffmpeg_failure(stderr: str | None, returncode: int | None = None, *, tool: str = "ffmpeg") -> EngineError:
    """Сырой stderr ffmpeg/ffprobe → EngineError с человеческим текстом. Сырое — в журнал и в `.debug`."""
    tail = (stderr or "").strip()[-1500:]
    raw = redact(tail) or f"код {returncode}"
    log.warning("%s завершился с ошибкой (код %s): %s", tool, returncode, raw)
    low = tail.lower()
    for needles, text in _FFMPEG_RULES:
        if any(n in low for n in needles):
            return EngineError(text, debug=raw)
    return EngineError("Не получилось обработать видео или звук. Подробности записаны в журнал программы.", debug=raw)


def data_dir() -> Path:
    """Корень данных: проекты, модели, кэш. Переопределяется DUBL_DATA_DIR (нужно тестам)."""
    d = Path(os.environ.get("DUBL_DATA_DIR") or user_data_dir("Dubl", "Dubl"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def find_binary(name: str) -> str:
    """ffmpeg / ffprobe: сначала DUBL_FFMPEG_DIR, потом PATH."""
    custom = os.environ.get("DUBL_FFMPEG_DIR")
    if custom:
        p = Path(custom) / (name + (".exe" if os.name == "nt" else ""))
        if p.exists():
            return str(p)
    found = shutil.which(name)
    if not found:
        raise EngineError("Не найдена программа, которая читает видео и звук (ffmpeg). Переустанови Дубль.",
                          debug=f"{name} не найден ни в DUBL_FFMPEG_DIR, ни в PATH")
    return found


# ───────────── отмена долгих внешних процессов ─────────────
# Задача (JobContext) на время работы ставит в своём потоке функцию проверки отмены; run_ffmpeg раз в 0.2 с
# её вызывает и, если она бросила исключение, убивает ffmpeg. Так «Отмена» работает и посреди долгого шага.
_cancel = threading.local()


@contextmanager
def cancel_check(fn: Callable[[], None] | None):
    """На время блока run_ffmpeg/run_process опрашивают fn(); она должна бросить исключение при отмене."""
    prev = getattr(_cancel, "fn", None)
    _cancel.fn = fn
    try:
        yield
    finally:
        _cancel.fn = prev


_NO_WINDOW = 0x08000000 if os.name == "nt" else 0       # CREATE_NO_WINDOW: без мигающей консоли в собранном приложении


def _abs_arg(a: str | Path) -> str:
    """Пути к файлам — только абсолютные: относительное имя вида «abc: x.mp4» ffmpeg принял бы за протокол."""
    if isinstance(a, Path):
        return str(a if a.is_absolute() else Path(os.path.abspath(a)))
    return str(a)


def run_process(cmd: list[str], *, timeout: float | None = None, cwd: str | Path | None = None
                ) -> subprocess.CompletedProcess[str]:
    """subprocess.run c текстовым выводом, таймаутом и отменой (см. cancel_check). Процесс не остаётся сиротой."""
    check = getattr(_cancel, "fn", None)
    p = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         text=True, encoding="utf-8", errors="replace", cwd=cwd, creationflags=_NO_WINDOW)
    waited = 0.0
    try:
        while True:
            try:
                out, err = p.communicate(timeout=0.2)
                break
            except subprocess.TimeoutExpired:
                waited += 0.2
                if check is not None:
                    check()
                if timeout is not None and waited >= timeout:
                    raise
    except BaseException:
        p.kill()
        try:
            p.communicate(timeout=5)
        except Exception:  # noqa: BLE001, S110 — процесс уже убит, дожидаться вывода необязательно
            pass
        raise
    return subprocess.CompletedProcess(cmd, p.returncode, out, err)


def run_ffmpeg(args: list[str | Path], *, timeout: float | None = None) -> subprocess.CompletedProcess[str]:
    """ffmpeg -hide_banner -loglevel error -y <args>. Бросает EngineError с хвостом stderr."""
    cmd = [find_binary("ffmpeg"), "-hide_banner", "-loglevel", "error", "-nostdin", "-y", *map(_abs_arg, args)]
    try:
        p = run_process(cmd, timeout=timeout)
    except subprocess.TimeoutExpired:
        raise EngineError("Обработка заняла слишком много времени и была остановлена.") from None
    if p.returncode != 0:
        raise ffmpeg_failure(p.stderr, p.returncode)
    return p


def ffprobe_json(path: str | Path, *extra: str) -> dict:
    cmd = [find_binary("ffprobe"), "-v", "error", "-print_format", "json", *extra, _abs_arg(Path(path))]
    try:
        p = run_process(cmd, timeout=120)
    except subprocess.TimeoutExpired:
        raise EngineError("Не удалось прочитать файл: он не ответил за 2 минуты. Проверь, что он не на сетевом диске и "
                          "не занят другой программой.") from None
    if p.returncode != 0:
        raise ffmpeg_failure(p.stderr, p.returncode, tool="ffprobe")
    return json.loads(p.stdout or "{}")
