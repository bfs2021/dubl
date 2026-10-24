"""Честный статус моделей для экрана: может ли каждая ответить сейчас и кто реально ответит на каждую задачу."""
from __future__ import annotations

from ..models import LLMSettings
from . import cli, keystatus, secrets
from .providers import needs_key
from .router import provider_health, task_routes


def fill_status(llm: LLMSettings, *, refresh: bool = False) -> LLMSettings:
    """Заполнить вычисляемые поля (key_*, can_answer, state…, tasks) у УЖЕ скопированных настроек. Возвращает их же."""
    health = {}
    for p in llm.providers:
        if not needs_key(p.kind):
            p.key_set, p.key_status, p.key_message = True, "unknown", None
        else:
            p.key_set = secrets.has_key(p.id)
            if p.key_set:
                p.key_status, p.key_message = keystatus.get_status(p.id)
            else:
                # ключа в связке нет, а в key_status.json остался «ok» от прошлой сессии — это и давало «3 модели → 2»
                keystatus.clear(p.id)
                p.key_status, p.key_message = "unknown", None
        h = provider_health(p, refresh=refresh)
        health[p.id] = h
        p.can_answer, p.state, p.state_message = h.can_answer, h.state, h.message          # type: ignore[assignment]
        p.login_supported, p.install_url = h.login_supported, h.install_url
        if not cli.is_cli_kind(p.kind):
            p.login_supported = False
    llm.tasks = task_routes(llm, health)
    return llm
