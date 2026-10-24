"""Настройки приложения: <data_dir>/settings.json. Ключи моделей сюда НЕ пишутся — только в keyring.

Токен Hugging Face (для pyannote) лежит в keyring как провайдер 'huggingface' (llm.secrets).
"""
from __future__ import annotations

import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from .models import LLMSettings
from .store.atomic import atomic_write_text
from .util import data_dir

log = logging.getLogger(__name__)

HF_PROVIDER = "huggingface"


class MicProfile(BaseModel):
    """Профиль «Мой микрофон»: итог проверки микрофона (POST /mic/check, POST /mic/latency). Поля, которых ещё не
    мерили, — None. Меняется только через /mic/*; из тела PUT /settings движок его не берёт."""
    noise_floor_db: float | None = None     # уровень шума комнаты в тишине, дБFS (RMS); чем ниже, тем тише (−60 — тишина)
    speech_db: float | None = None          # средний уровень речи, дБFS (RMS по самим фразам); норма −26…−14
    peak_db: float | None = None            # самый громкий пик речи, дБFS
    latency_ms: float | None = None         # задержка «динамик → микрофон» по замеру петлёй, мс
    checked_at: datetime = Field(default_factory=lambda: datetime.now().astimezone())
    device: str | None = None               # какой микрофон проверяли (AppSettings.input_device на тот момент)


class AppSettings(BaseModel):
    llm: LLMSettings = LLMSettings()
    target_lang: str = Field("ru", min_length=2, max_length=16)   # язык перевода (на нём ищем субтитры файла как перевод)
    latency_ms: float = Field(0.0, ge=0.0, le=2000.0)   # задержка записи (звуковая карта), мс
    input_device: str | None = None                      # микрофон (выбирает UI; движок только хранит)
    # Профиль «Мой микрофон» (проверка микрофона); None — ещё не проверяли. Меняют только роуты /mic/*.
    mic_profile: MicProfile | None = None
    processing_device: str = "auto"                      # auto | cpu | cuda | mps
    whisper_model: str = "large-v3-turbo"
    # fast (demucs) на тестовом материале не хуже best, быстрее в 6 раз и легче по памяти (1,9 против 8,6–9,5 ГБ).
    # best (BS-RoFormer) — по выбору; для сцен длиннее BEST_MAX_SCENE_S pipeline.prepare_scene сам откатывается на fast.
    separation_quality: Literal["fast", "best"] = "fast"
    # Чем распознавать речь при разборе: auto — русский GigaAM (точнее границы реплик), остальное Whisper;
    # whisper — всегда Whisper; gigaam — GigaAM, но для нерусской речи всё равно Whisper (GigaAM знает только русский).
    asr_engine: Literal["auto", "whisper", "gigaam"] = "auto"

    @field_validator("processing_device")
    @classmethod
    def _device(cls, v: str) -> str:
        v = (v or "auto").strip().lower()
        if not re.fullmatch(r"auto|cpu|mps|cuda(:\d+)?", v):
            raise ValueError("Устройство: auto, cpu, mps или cuda")
        return v


def settings_path() -> Path:
    return data_dir() / "settings.json"


def load_settings(path: str | Path | None = None) -> AppSettings:
    p = Path(path) if path else settings_path()
    try:
        return AppSettings.model_validate_json(p.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return AppSettings()
    except Exception as e:  # noqa: BLE001 — битый файл настроек не должен ронять движок
        log.warning("Не удалось прочитать настройки (%s): %s — беру по умолчанию", p, e)
        return AppSettings()


def save_settings(s: AppSettings, path: str | Path | None = None) -> AppSettings:
    p = Path(path) if path else settings_path()
    # key_set / key_status / key_message / can_answer / state… / tasks — вычисляемые поля ответа API, в файл не пишем
    skip = {"llm": {"providers": {"__all__": {"key_set", "key_status", "key_message", "can_answer", "state",
                                                "state_message", "login_supported", "install_url"}},
                    "tasks": True}}
    atomic_write_text(p, s.model_dump_json(indent=1, exclude=skip))
    return s


# ───────────── токен Hugging Face (keyring) ─────────────
def get_hf_token() -> str | None:
    try:
        from .llm import secrets
        return secrets.get_key(HF_PROVIDER)
    except Exception as e:  # noqa: BLE001 — нет модуля/связки ключей: просто без токена
        log.debug("HF-токен недоступен: %s", e)
        return None


def set_hf_token(token: str) -> None:
    from .llm import secrets
    secrets.set_key(HF_PROVIDER, token)


def delete_hf_token() -> None:
    from .llm import secrets
    secrets.delete_key(HF_PROVIDER)
