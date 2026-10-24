"""Статус ключа модели: unknown / ok / invalid + русское объяснение.

Хранится в <data_dir>/key_status.json (без самих ключей), чтобы переживать перезапуск движка. Меняется:
проверкой при сохранении ключа, кнопкой «Проверить» и любым реальным запросом к ИИ (ответили — ok, ключ не подошёл — invalid).
"""
from __future__ import annotations

import json
import logging
import threading
from typing import Literal

from ..store.atomic import atomic_write_text
from ..util import data_dir

log = logging.getLogger(__name__)

KeyStatus = Literal["unknown", "ok", "invalid"]
KEY_INVALID_MSG = "Ключ не подходит — проверь, что скопировал его целиком"
KEY_NO_NETWORK_MSG = "Не удалось проверить ключ: нет связи с сервисом ИИ. Ключ сохранён, проверим при первом запросе."

_lock = threading.Lock()


def _path():
    return data_dir() / "key_status.json"


def _load() -> dict[str, dict]:
    try:
        data = json.loads(_path().read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def get_status(provider_id: str) -> tuple[KeyStatus, str | None]:
    with _lock:
        row = _load().get(provider_id) or {}
    status = row.get("status")
    if status not in ("ok", "invalid"):
        return "unknown", row.get("message") if isinstance(row.get("message"), str) else None
    return status, row.get("message") if isinstance(row.get("message"), str) else None


def set_status(provider_id: str, status: KeyStatus, message: str | None = None) -> None:
    with _lock:
        data = _load()
        row = {"status": status, "message": message}
        if status == "unknown" and message is None:
            if provider_id not in data:
                return
            data.pop(provider_id)
        elif data.get(provider_id) == row:
            return
        else:
            data[provider_id] = row
        try:
            atomic_write_text(_path(), json.dumps(data, ensure_ascii=False))
        except OSError as e:           # статус — удобство, а не данные: не ронять запрос ИИ из-за диска
            log.warning("Не удалось записать статус ключа: %s", e)


def clear(provider_id: str) -> None:
    set_status(provider_id, "unknown", None)
