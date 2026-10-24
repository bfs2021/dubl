"""Отдача медиа: файлы проекта, исходное видео (с Range) и лёгкая копия для плеера."""
from __future__ import annotations

import threading
from pathlib import Path

from fastapi import APIRouter, Depends, Query
from fastapi.responses import FileResponse, JSONResponse

from ...pipeline import PROXY_STEPS, build_proxy, proxy_ready
from ...store import NotFoundError
from ...util import EngineError
from ..deps import JobsDep, StoreDep, require_token
from ..schemas import ErrorResponse, JobAccepted

router = APIRouter(tags=["files"], dependencies=[Depends(require_token)], responses={404: {"model": ErrorResponse}})

JOB_HEADER = {"X-Job-Id": {"description": "Id задачи, которая готовит файл (GET /jobs/{id} или WS /ws/jobs)",
                           "schema": {"type": "string"}}}
_proxy_lock = threading.Lock()


def _source_file(store, pid: str) -> Path:
    src = store.get(pid).source_path
    if not src or not Path(src).is_file():
        raise NotFoundError("Исходное видео не найдено — возможно, файл перенесён или ещё скачивается.")
    return Path(src)


@router.get("/files/{pid}/source", response_class=FileResponse)
def get_source(pid: str, store: StoreDep) -> FileResponse:
    """Исходное видео проекта целиком (лежит там, где его оставил пользователь). Range-запросы — для перемотки в плеере.
    Если браузер такой файл не проигрывает (mkv, ac3, hevc), плеер берёт /files/{pid}/proxy."""
    return FileResponse(_source_file(store, pid))


@router.get("/files/{pid}/proxy", response_class=FileResponse,
            responses={200: {"description": "mp4 540p h264+aac, faststart; Range поддерживается"},
                       202: {"model": JobAccepted, "headers": JOB_HEADER,
                             "description": "Копия готовится — повторите запрос позже"},
                       400: {"model": ErrorResponse}})
def get_proxy(pid: str, store: StoreDep, jobs: JobsDep, retry: bool = Query(False, description="Начать заново после ошибки")):
    """Лёгкая копия исходника для плеера (kind='proxy'). Нет копии — запускается задача и сразу 202 + X-Job-Id;
    пока она идёт, повторные запросы тоже получают 202 с тем же id. Готово — 200 и сам файл."""
    _source_file(store, pid)
    if proxy_ready(store, pid):
        return FileResponse(store.proxy_path(pid), media_type="video/mp4")
    with _proxy_lock:
        mine = [j for j in jobs.list() if j.kind == "proxy" and j.project_id == pid]     # свежие первыми
        running = next((j for j in mine if j.state.value in ("queued", "running")), None)
        if running is None:
            last = mine[0] if mine else None
            if last is not None and last.state.value == "failed" and not retry:
                raise EngineError(last.error or "Не удалось подготовить видео для просмотра.")
            running = jobs.submit("proxy", pid, PROXY_STEPS,
                                  lambda ctx: _proxy_result(store, pid, build_proxy(store, pid, ctx)))
    return JSONResponse(JobAccepted(job_id=running.id).model_dump(), status_code=202, headers={"X-Job-Id": running.id})


def _proxy_result(store, pid: str, path: Path):
    from ...pipeline import job_result
    return job_result(store, pid, path).model_dump()


@router.get("/files/{pid}/{path:path}", response_class=FileResponse)
def get_file(pid: str, path: str, store: StoreDep) -> FileResponse:
    """Медиа проекта (Range-запросы для видео поддерживает FileResponse). Только внутри папки проекта."""
    base = store.project_dir(pid).resolve()
    target = (base / path).resolve()
    if base not in target.parents:
        raise NotFoundError("Файл не найден.")
    if not target.is_file():
        raise NotFoundError("Файл не найден.")
    return FileResponse(target)
