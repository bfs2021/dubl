"""Маршрутизация задач ИИ по настройкам пользователя — с запасной моделью и честным статусом.

· Каждой задаче в настройках назначена модель (`routing`). Если она не может ответить (нет программы, нет входа, нет ключа,
  ключ не подошёл, выключена) — берём следующую включённую рабочую.
· Если назначенная модель была рабочей, но запрос к ней упал (сеть, лимит, вход слетел) — один раз пробуем запасную
  и запоминаем, кто ответил.
· «ИИ не настроен» остаётся только тогда, когда рабочих моделей нет совсем (или задача намеренно выключена: «Не использовать»).
"""
from __future__ import annotations

import logging
import re
import threading
from dataclasses import dataclass

from ..models import LLMSettings, LLMTask, ProviderConfig, ProviderKind, TaskRoute
from ..util import EngineError
from . import cli, keystatus, secrets
from .providers import LLMProvider, make_provider, needs_key

log = logging.getLogger(__name__)

CHEAP_TASKS = (LLMTask.TRANSLATE, LLMTask.FIT_LENGTH, LLMTask.NAME_SPEAKERS)

_CHEAP = re.compile(r"flash|haiku|(?<!ge)mini|lite|nano|small|8b|instant|turbo", re.IGNORECASE)
_TOP = re.compile(r"opus|[-_ ]pro\b|reasoner|large|ultra|gpt-5(?!.*mini)|o1|o3", re.IGNORECASE)
_MID = re.compile(r"sonnet|chat|medium|4o\b|gpt-4", re.IGNORECASE)


def model_tier(cfg: ProviderConfig) -> int:
    """Грубая оценка «силы» модели по названию: 0 — дешёвая и быстрая, 1 — обычная, 2 — сильная, 3 — лучшая."""
    name = cfg.model or ""
    if _CHEAP.search(name):
        return 0
    if _TOP.search(name):
        return 3
    if _MID.search(name):
        return 2
    return 1


def default_routing(providers: list[ProviderConfig]) -> dict[LLMTask, str | None]:
    """Умолчания: дешёвая модель — переводу, укладке и именам; сильнейшая — режиссёру.

    Среди равных выигрывает тот, кто раньше в списке. Включённых нет → всё None.
    """
    usable = [p for p in providers if p.enabled]
    if not usable:
        return {t: None for t in LLMTask}
    cheap = min(usable, key=lambda p: model_tier(p))     # min/max берут первый из равных
    best = max(usable, key=lambda p: model_tier(p))
    routing: dict[LLMTask, str | None] = {t: cheap.id for t in CHEAP_TASKS}
    routing[LLMTask.DIRECTOR] = best.id
    return routing


# ─────────────────────────── может ли модель ответить ───────────────────────────
@dataclass(frozen=True)
class Health:
    id: str
    can_answer: bool
    state: str                 # ok | off | not_installed | not_logged_in | no_key | invalid_key | error
    message: str               # по-русски
    login_supported: bool = False
    install_url: str | None = None


def provider_health(cfg: ProviderConfig, *, refresh: bool = False) -> Health:
    """Честный ответ «может ли модель ответить прямо сейчас» и почему нет. Запросов к модели не делает."""
    spec = cli.spec_for(cfg.kind)
    if spec is not None:
        extra = {"login_supported": True, "install_url": spec.install_url}
        if not cfg.enabled:
            return Health(cfg.id, False, "off", "Выключено.", **extra)
        st = cli.cli_status(cfg.kind, refresh=refresh)
        if st.state == "ok":
            return Health(cfg.id, True, "ok", "Установлен, вход выполнен, отвечает.", **extra)
        if st.state == "not_installed":
            return Health(cfg.id, False, "not_installed", f"Не найден — {spec.install_hint[0].lower()}{spec.install_hint[1:]}", **extra)
        if st.state == "not_logged_in":
            return Health(cfg.id, False, "not_logged_in", "Установлен, но нужен вход.", **extra)
        return Health(cfg.id, False, "error", st.message, **extra)
    if not cfg.enabled:
        return Health(cfg.id, False, "off", "Выключено.")
    if needs_key(cfg.kind):
        try:
            present = secrets.has_key(cfg.id)
        except Exception:  # noqa: BLE001 — has_key не бросает, но связка ключей на чужой машине всякое умеет
            present = False
        if not present:
            return Health(cfg.id, False, "no_key", "Нужен ключ.")
        status, msg = keystatus.get_status(cfg.id)
        if status == "invalid":
            return Health(cfg.id, False, "invalid_key", msg or keystatus.KEY_INVALID_MSG)
    return Health(cfg.id, True, "ok", "Ключ сохранён, отвечает." if needs_key(cfg.kind) else "Готово.")


def settings_health(settings: LLMSettings, *, refresh: bool = False) -> dict[str, Health]:
    return {p.id: provider_health(p, refresh=refresh) for p in settings.providers}


# ─────────────────────────── кто отвечает ───────────────────────────
def _order(task: LLMTask, workable: list[ProviderConfig]) -> list[ProviderConfig]:
    """Запасные по порядку: дешёвым задачам — сначала простые модели, режиссёру — сильные. Среди равных — как в списке."""
    if task in CHEAP_TASKS:
        return sorted(workable, key=lambda p: model_tier(p))
    return sorted(workable, key=lambda p: -model_tier(p))


def assigned_id(task: LLMTask, settings: LLMSettings, health: dict[str, Health] | None = None) -> str | None:
    """Модель, назначенная задаче. Явное «Не использовать» (null) → None. Задачи нет в routing (человек ещё не выбирал) →
    программа выбирает сама среди рабочих моделей по умолчанию (default_routing)."""
    if task in settings.routing:
        return settings.routing[task]
    health = health if health is not None else settings_health(settings)
    workable = [p for p in settings.providers if p.enabled and health[p.id].can_answer]
    return default_routing(workable)[task]


def candidates(task: LLMTask, settings: LLMSettings, health: dict[str, Health] | None = None) -> list[ProviderConfig]:
    """Рабочие модели для задачи по порядку: назначенная (если рабочая), затем запасные.

    Пусто — если задача выключена («Не использовать») либо рабочих моделей нет.
    """
    health = health if health is not None else settings_health(settings)
    pid = assigned_id(task, settings, health)
    if not pid:
        return []
    workable = [p for p in settings.providers if p.enabled and health[p.id].can_answer]
    first = [p for p in workable if p.id == pid]
    rest = _order(task, [p for p in workable if p.id != pid])
    return first + rest


def _why_not(task: LLMTask, settings: LLMSettings, health: dict[str, Health]) -> str:
    pid = assigned_id(task, settings, health)
    cfg = next((p for p in settings.providers if p.id == pid), None)
    if cfg is None:
        return "Для этой задачи модель не выбрана."
    h = health[cfg.id]
    return f"«{cfg.title}»: {h.message[:1].lower() + h.message[1:]}"


def no_model_text(task: LLMTask, settings: LLMSettings) -> str:
    """Почему ИИ не ответит. Начинается с «ИИ для этой задачи не настроен» (по этой фразе экран ведёт в Настройки)."""
    health = settings_health(settings)
    pid = assigned_id(task, settings, health)
    if not pid:
        return "ИИ для этой задачи не настроен. Подключи модель в Настройках → Модели ИИ."
    others = [p for p in settings.providers if p.enabled and p.id != pid]
    tail = "Других рабочих моделей нет — подключи модель в Настройках → Модели ИИ." if not any(
        health[p.id].can_answer for p in others) else ""
    return f"ИИ для этой задачи не настроен: {_why_not(task, settings, health)} {tail}".strip()


def task_routes(settings: LLMSettings, health: dict[str, Health] | None = None) -> dict[LLMTask, TaskRoute]:
    """По каждой задаче: кто назначен, кто реально ответит (с запасной), кто запасной на случай сбоя."""
    health = health if health is not None else settings_health(settings)
    titles = {p.id: p.title for p in settings.providers}
    out: dict[LLMTask, TaskRoute] = {}
    for t in LLMTask:
        pid = assigned_id(t, settings, health)
        last = _last_used.get(t)
        route = TaskRoute(assigned=pid, last_used=last[0] if last else None, last_used_fallback=bool(last and last[1]))
        cand = candidates(t, settings, health)
        if not pid:
            route.message = "Не используется." if t in settings.routing else "Нет рабочей модели — подключи её в Настройках."
        elif not cand:
            route.message = no_model_text(t, settings).removeprefix("ИИ для этой задачи не настроен: ")
        else:
            route.answers, route.answers_title = cand[0].id, cand[0].title
            route.fallback = cand[0].id != pid
            route.fallback_id = cand[1].id if len(cand) > 1 else None
            if route.fallback:
                route.message = f"«{titles.get(pid, pid)}» сейчас не может ответить — отвечает «{cand[0].title}»."
        out[t] = route
    return out


# ─────────────────────────── запасная при сбое ───────────────────────────
_last_used: dict[LLMTask, tuple[str, bool]] = {}      # задача → (название ответившей модели, запасная ли)
_used_lock = threading.Lock()


def last_used(task: LLMTask) -> tuple[str, bool] | None:
    with _used_lock:
        return _last_used.get(task)


def _record_used(task: LLMTask, title: str, fallback: bool) -> None:
    with _used_lock:
        _last_used[task] = (title, fallback)


def reset_for_tests() -> None:
    with _used_lock:
        _last_used.clear()


class RoutedProvider:
    """Основная модель + одна запасная. Основная упала (EngineError) → один раз пробуем запасную.

    После удачного ответа запасной в этом же запросе (несколько вызовов подряд, например пачки перевода) основную
    повторно не дёргаем: она только что упала.
    """

    def __init__(self, task: LLMTask, chain: list[tuple[ProviderConfig, LLMProvider]], *, assigned_id: str | None = None) -> None:
        self.task = task
        self.chain = chain
        self.id = chain[0][0].id
        self.assigned_id = assigned_id or self.id
        self._pos = 0
        self.answered_by: str | None = None          # название модели, которая ответила последней
        self.answered_by_fallback = False

    async def complete(self, system: str, user: str, *, json_schema: dict | None = None,
                       max_tokens: int = 1024, temperature: float = 0.4) -> str:
        cfg, prov = self.chain[self._pos]
        try:
            text = await prov.complete(system, user, json_schema=json_schema, max_tokens=max_tokens, temperature=temperature)
        except EngineError as e:
            if self._pos + 1 >= len(self.chain):
                raise
            nxt_cfg, nxt = self.chain[self._pos + 1]
            log.warning("Модель «%s» не ответила (%s) — пробую запасную «%s»", cfg.id, type(e).__name__, nxt_cfg.id)
            try:
                text = await nxt.complete(system, user, json_schema=json_schema, max_tokens=max_tokens, temperature=temperature)
            except EngineError as e2:
                raise EngineError(f"{e} Запасная модель «{nxt_cfg.title}» тоже не ответила: {e2}", debug=getattr(e, "debug", None)) from None
            self._pos += 1
            cfg = nxt_cfg
        self.answered_by = cfg.title
        self.answered_by_fallback = cfg.id != self.assigned_id
        _record_used(self.task, cfg.title, self.answered_by_fallback)
        return text

    async def check(self) -> tuple[bool, str]:
        return await self.chain[0][1].check()


def provider_config_for(task: LLMTask, settings: LLMSettings) -> ProviderConfig | None:
    """Модель, назначенная задаче (включённая), даже если она сейчас не может ответить."""
    pid = settings.routing.get(task)
    if not pid:
        return None
    for p in settings.providers:
        if p.id == pid and p.enabled:
            return p
    return None


def provider_for(task: LLMTask, settings: LLMSettings) -> LLMProvider | None:
    """Модель для задачи или None, если отвечать некому (задача выключена или рабочих моделей нет совсем).

    Назначенная модель недоступна → берём следующую рабочую. Рабочих несколько → ответ можно получить и от запасной,
    если основная упадёт посреди запроса (RoutedProvider).
    """
    health = settings_health(settings)
    cand = candidates(task, settings, health)
    pid = assigned_id(task, settings, health)
    if not cand:
        # Последний шанс: ключ помечен «не подходит», но другой модели нет. Метка могла устареть (ключ исправили у
        # провайдера, случайный отказ) — один настоящий запрос её обновит. Нет программы или ключа — не пробуем.
        cfg = provider_config_for(task, settings)
        if cfg is None or provider_health(cfg).state != "invalid_key":
            return None
        cand = [cfg]
    if cand[0].id != pid:
        log.info("Модель «%s» для задачи %s недоступна — отвечает «%s»", pid, task.value, cand[0].id)
    chain = [(c, make_provider(c)) for c in cand[:2]]       # основная + одна запасная
    return RoutedProvider(task, chain, assigned_id=pid)


def who_answered(provider: object) -> tuple[str | None, bool]:
    """(название модели, ответившей на последний запрос; запасная ли) — для ответа API."""
    if isinstance(provider, RoutedProvider):
        return provider.answered_by, provider.answered_by_fallback
    return None, False


__all__ = ["CHEAP_TASKS", "Health", "ProviderKind", "RoutedProvider", "assigned_id", "candidates", "default_routing", "last_used",
           "model_tier", "no_model_text", "provider_config_for", "provider_for", "provider_health", "settings_health",
           "task_routes", "who_answered"]
