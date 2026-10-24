from __future__ import annotations

from functools import lru_cache

from fastapi import APIRouter

from ..deps import SettingsDep
from ..schemas import Capabilities, Health

router = APIRouter(tags=["system"])


def engine_version() -> str:
    try:
        from importlib.metadata import version
        return version("dubl-engine")
    except Exception:  # noqa: BLE001 — запуск из исходников без установки
        return "0.1.0"


@lru_cache(maxsize=8)
def _device(requested: str) -> str:
    try:
        from ...devices import pick_device
        return pick_device(requested)
    except Exception:  # noqa: BLE001 — torch не установлен или сломан: считаем процессор
        return "cpu"


def _has(*modules: str) -> bool:
    """Модули установлены? Без импорта (torch грузится секундами): только поиск в путях."""
    from importlib.util import find_spec
    try:
        return all(find_spec(m) is not None for m in modules)
    except (ImportError, ValueError):       # сломанный пакет
        return False


def capabilities() -> Capabilities:
    asr = _has("faster_whisper") or _has("mlx_whisper")
    separation = _has("torch") and (_has("demucs") or _has("audio_separator"))
    diarization = _has("torch", "speechbrain")
    from ...analysis import gigaam
    return Capabilities(ml=asr and separation and diarization, asr=asr, separation=separation, diarization=diarization,
                        gigaam=gigaam.installed(), gigaam_downloaded=gigaam.downloaded(),
                        gigaam_download_mb=gigaam.DOWNLOAD_MB)


@router.get("/health")
def health(settings: SettingsDep) -> Health:
    """Жив ли движок. Открытый роут (без токена): по нему UI находит движок. torch грузится один раз."""
    return Health(ok=True, version=engine_version(), device=_device(settings.processing_device),
                  capabilities=capabilities())
