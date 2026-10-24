from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from fastapi.concurrency import run_in_threadpool

from ... import settings as app_settings
from ...store import NotFoundError
from ..deps import SettingsDep, require_token
from ..schemas import CheckResult, ErrorResponse, KeyBody, KeyResult, LoginResult, Ok

router = APIRouter(prefix="/settings", tags=["settings"], dependencies=[Depends(require_token)])


def with_key_state(settings: app_settings.AppSettings, *, refresh: bool = False) -> app_settings.AppSettings:
    """Копия настроек, где у каждой модели заполнено: есть ли ключ и подошёл ли он (key_*), может ли она ответить сейчас
    (can_answer / state / state_message) и кто реально ответит на каждую задачу (llm.tasks)."""
    from ...llm.status import fill_status

    out = settings.model_copy(deep=True)
    fill_status(out.llm, refresh=refresh)
    return out


@router.get("")
def get_settings(settings: SettingsDep, refresh: bool = False) -> app_settings.AppSettings:
    """Настройки без ключей: ключи лежат только в системной связке ключей. У каждой модели — key_set (ключ сохранён),
    key_status (unknown | ok | invalid) и key_message (по-русски, что не так)."""
    return with_key_state(settings, refresh=refresh)     # refresh=true — заново проверить установку и вход CLI-моделей


@router.put("")
def put_settings(body: app_settings.AppSettings, request: Request) -> app_settings.AppSettings:
    body.mic_profile = request.app.state.settings.mic_profile        # «Мой микрофон» меняют только роуты /mic/*
    request.app.state.settings = app_settings.save_settings(body)       # key_* из тела не сохраняются
    return with_key_state(request.app.state.settings)


@router.put("/llm/{provider_id}/key", responses={400: {"model": ErrorResponse}})
async def put_key(provider_id: str, body: KeyBody, settings: SettingsDep) -> KeyResult:
    """Сохранить ключ и сразу проверить его (один короткий запрос, до 10 с). Неверный ключ всё равно сохраняется,
    но key_status = invalid и key_message = «Ключ не подходит — проверь, что скопировал его целиком». Нет сети или
    сервис не ответил — unknown (ключ не считаем плохим)."""
    from ...llm import keystatus, secrets
    from ...llm.providers import verify_key
    await run_in_threadpool(secrets.set_key, provider_id, body.key.get_secret_value())
    keystatus.clear(provider_id)
    cfg = next((p for p in settings.llm.providers if p.id == provider_id), None)
    if cfg is None:                     # токен Hugging Face и другие ключи без модели: проверять нечем
        return KeyResult(key_status="unknown", key_message=None)
    status, message, _text = await verify_key(cfg)
    keystatus.set_status(provider_id, status, message)  # type: ignore[arg-type]
    return KeyResult(key_status=status, key_message=message)  # type: ignore[arg-type]


@router.delete("/llm/{provider_id}/key", responses={400: {"model": ErrorResponse}})
def delete_key(provider_id: str) -> Ok:
    from ...llm import keystatus, secrets
    secrets.delete_key(provider_id)
    keystatus.clear(provider_id)
    return Ok()


@router.post("/llm/{provider_id}/check", responses={404: {"model": ErrorResponse}})
async def check_provider(provider_id: str, settings: SettingsDep) -> CheckResult:
    """Проверить модель одним запросом. ok=false и key_status=invalid — ключ не подошёл; unknown — проверить не вышло."""
    from ...llm import keystatus
    from ...llm.providers import verify_key
    cfg = next((p for p in settings.llm.providers if p.id == provider_id), None)
    if cfg is None:
        raise NotFoundError("Такая модель не настроена.")
    status, message, text = await verify_key(cfg)
    if status != "unknown":
        keystatus.set_status(provider_id, status, message)  # type: ignore[arg-type]
    return CheckResult(ok=status == "ok", message=text, key_status=status, key_message=message)  # type: ignore[arg-type]


@router.post("/llm/{provider_id}/login", responses={400: {"model": ErrorResponse}, 404: {"model": ErrorResponse}})
async def login_provider(provider_id: str, settings: SettingsDep) -> LoginResult:
    """Кнопка «Войти» у модели-подписки (Claude Code, Codex): открыть системный терминал с командой входа.
    Команда берётся из фиксированного списка по виду модели — из запроса ничего не подставляется."""
    from ...llm import cli
    cfg = next((p for p in settings.llm.providers if p.id == provider_id), None)
    if cfg is None:
        raise NotFoundError("Такая модель не настроена.")
    message = await run_in_threadpool(cli.open_login_terminal, cfg.kind)
    return LoginResult(ok=True, message=message)
