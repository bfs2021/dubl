"""Имя файла для готового ролика: не затирать чужое без спроса.

Занять имя нужно атомарно (O_EXCL): две задачи с одинаковым названием сцены, запущенные подряд, не должны выбрать
один и тот же файл. Пока ролик собирается, на месте файла лежит пустая заглушка; готовый ролик заменяет её целиком.
"""
from __future__ import annotations

import contextlib
import os
from pathlib import Path

MAX_COPIES = 999


def numbered(path: Path, n: int) -> Path:
    """«Ролик.mp4» → «Ролик (2).mp4» (n=1 — само имя)."""
    return path if n <= 1 else path.with_name(f"{path.stem} ({n}){path.suffix}")


def reserve_output(path: Path, overwrite: bool = False) -> Path:
    """Итоговый путь ролика. overwrite=True — тот же путь (файл заменится). Иначе, если файл уже есть, берём
    « (2)», « (3)» … и сразу занимаем это имя пустой заглушкой."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if overwrite:
        return path
    for n in range(1, MAX_COPIES + 1):
        cand = numbered(path, n)
        try:
            fd = os.open(cand, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            continue
        os.close(fd)
        return cand
    raise FileExistsError(f"Слишком много копий с именем {path.name}")


def release_output(path: Path, overwrite: bool) -> None:
    """Сборка не удалась: убрать свою пустую заглушку (чужой готовый файл не трогаем)."""
    if overwrite:
        return
    with contextlib.suppress(OSError):
        if path.is_file() and path.stat().st_size == 0:
            path.unlink()
