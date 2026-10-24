"""Выбор устройства для нейросетей: cuda → mps → cpu.

torch импортируется только внутри функций, чтобы движок стартовал и без него.
"""
from __future__ import annotations

import logging

log = logging.getLogger(__name__)

DEVICES = ("cuda", "mps", "cpu")

# устройства, на которых нейросеть уже упала в этом процессе, — второй раз их не пробуем
_BAD: set[tuple[str, str]] = set()


def cuda_available() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001 — torch не установлен или сломан
        return False


def mps_available() -> bool:
    try:
        import torch
        return bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    except Exception:  # noqa: BLE001
        return False


def pick_device(requested: str = "auto") -> str:
    """'auto' → cuda, если есть; иначе mps (Apple Silicon); иначе cpu.

    Явное значение ('cuda', 'mps', 'cpu', 'cuda:1') возвращается как есть, если оно доступно;
    недоступное заменяется на cpu.
    """
    req = (requested or "auto").lower()
    if req == "auto":
        if cuda_available():
            return "cuda"
        if mps_available():
            return "mps"
        return "cpu"
    base = req.split(":")[0]
    if base == "cuda" and cuda_available():
        return req
    if base == "mps" and mps_available():
        return "mps"
    if base == "cpu":
        return "cpu"
    log.warning("Устройство %r недоступно, использую cpu", requested)
    return "cpu"


def device_candidates(requested: str = "auto", *, task: str = "") -> list[str]:
    """Список устройств в порядке попыток: выбранное, потом cpu. Упавшие ранее для `task` пропускаются."""
    first = pick_device(requested)
    out = [first] if (task, first) not in _BAD else []
    if "cpu" not in out:
        out.append("cpu")
    return out


def mark_bad(task: str, device: str) -> None:
    if device != "cpu":
        _BAD.add((task, device))


def whisper_runtime(device: str) -> tuple[str, str]:
    """(device, compute_type) для faster-whisper (CTranslate2): cuda float16, остальное cpu int8.

    CTranslate2 не умеет Apple MPS, поэтому на Mac распознавание идёт на процессоре (int8).
    """
    if device.split(":")[0] == "cuda":
        return "cuda", "float16"
    return "cpu", "int8"
