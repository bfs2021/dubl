"""Зависимости API: хранилище, очередь задач, настройки, проверка токена."""
from __future__ import annotations

import hmac
from typing import Annotated

from fastapi import Depends, HTTPException, Request

from ..jobs import JobManager
from ..models import Project, Scene
from ..settings import AppSettings
from ..store import ProjectStore
from .live import live_project


def get_store(request: Request) -> ProjectStore:
    return request.app.state.store


def get_jobs(request: Request) -> JobManager:
    return request.app.state.jobs


def get_settings(request: Request) -> AppSettings:
    return request.app.state.settings


StoreDep = Annotated[ProjectStore, Depends(get_store)]
JobsDep = Annotated[JobManager, Depends(get_jobs)]
SettingsDep = Annotated[AppSettings, Depends(get_settings)]


def token_ok(expected: str | None, candidate: str | None) -> bool:
    """Сравнение за постоянное время; без токена в настройках пускаем всех (локальный режим)."""
    if not expected:
        return True
    return bool(candidate) and hmac.compare_digest(candidate.encode(), expected.encode())


def extract_token(headers, query_params) -> str | None:
    auth = headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return query_params.get("token")      # для <video src> и WebSocket, где заголовок не поставить


def require_token(request: Request) -> None:
    if not token_ok(request.app.state.token, extract_token(request.headers, request.query_params)):
        raise HTTPException(status_code=401, detail="Нужен токен доступа.")


def project_dep(pid: str, store: StoreDep, jobs: JobsDep) -> Project:
    return live_project(store.get(pid), jobs)


def scene_dep(scid: str, project: Annotated[Project, Depends(project_dep)], store: StoreDep) -> Scene:
    return store.find_scene(project, scid)


ProjectDep = Annotated[Project, Depends(project_dep)]
SceneDep = Annotated[Scene, Depends(scene_dep)]
