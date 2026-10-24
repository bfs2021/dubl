"""FastAPI-приложение движка «Дубль»: сборка из роутеров (api/routes), ошибки, CORS, WebSocket прогресса."""
from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager

import anyio
import anyio.abc
from fastapi import FastAPI, Request, WebSocket
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.websockets import WebSocketDisconnect

from .. import settings as app_settings
from ..jobs import JobManager
from ..store import NotFoundError, ProjectStore
from ..util import ConflictError, EngineError, UnprocessableError, redact
from .deps import extract_token, token_ok
from .routes import files, health, llm, mic, projects, scenes
from .routes import jobs as jobs_routes
from .routes import settings as settings_routes
from .routes.health import engine_version

log = logging.getLogger(__name__)

CORS_ORIGINS = ["tauri://localhost", "http://tauri.localhost", "https://tauri.localhost",
                "http://localhost:1420", "http://127.0.0.1:1420"]


_HTTP_EN = {404: "Not Found", 405: "Method Not Allowed", 401: "Unauthorized", 403: "Forbidden", 400: "Bad Request",
            413: "Request Entity Too Large", 416: "Range Not Satisfiable"}
_HTTP_RU = {404: "Такого адреса в движке нет. Возможно, версии программы и движка не совпадают — обнови программу.",
            405: "Движок не принимает такой запрос по этому адресу.", 401: "Нужен токен доступа.",
            403: "Доступ запрещён.", 400: "Некорректный запрос.", 413: "Файл слишком большой.",
            416: "Запрошена неверная часть файла."}


def _msg_422(e: RequestValidationError) -> str:
    """Первая ошибка проверки — коротко и по-русски, без пути pydantic и без значений (там могут быть ключи)."""
    errs = e.errors()
    if not errs:
        return "Некорректный запрос."
    first = errs[0]
    where = ".".join(str(x) for x in first.get("loc", ()) if x not in ("body", "query", "path", "form"))
    text = str(first.get("msg", "")).removeprefix("Value error, ")
    return f"Некорректный запрос ({where}): {text}" if where else f"Некорректный запрос: {text}"


def create_app(store: ProjectStore | None = None, jobs: JobManager | None = None,
               token: str | None = None, *, recover: bool = False) -> FastAPI:
    """recover=True (так запускает main.run): при старте разбор и подготовка сцены, которые были в работе, когда движок
    остановили, помечаются прерванными, а stems_ready=true без целых файлов снимается."""
    store = store or ProjectStore()
    jobs = jobs or JobManager()

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        jobs.bind_loop(asyncio.get_running_loop())
        if recover:
            fixed = await asyncio.get_running_loop().run_in_executor(None, store.recover_interrupted)
            if fixed:
                log.info("Прерванные при остановке движка разбор или подготовка сцены: %s", ", ".join(fixed))
        yield
        jobs.shutdown()

    app = FastAPI(title="Дубль — движок", version=engine_version(), lifespan=lifespan,
                  openapi_tags=[{"name": n} for n in ("system", "projects", "scenes", "files", "settings", "mic", "llm", "jobs")])
    app.state.store, app.state.jobs, app.state.token = store, jobs, token
    app.state.settings = app_settings.load_settings()
    app.add_middleware(CORSMiddleware, allow_origins=CORS_ORIGINS, allow_methods=["*"], allow_headers=["*"],
                       allow_credentials=False, expose_headers=["X-Job-Id"])

    # ───────────── ошибки: у всех один вид {"detail": "текст по-русски"} ─────────────
    @app.exception_handler(NotFoundError)
    async def _nf(_r: Request, e: NotFoundError):
        return JSONResponse({"detail": str(e)}, status_code=404)

    def _engine_body(e: EngineError) -> dict:
        body = {"detail": str(e)}
        if e.debug:
            body["debug"] = redact(e.debug)       # сырой текст — отдельным полем, для отладки
        return body

    @app.exception_handler(ConflictError)
    async def _conflict(_r: Request, e: ConflictError):
        return JSONResponse(_engine_body(e), status_code=409)

    @app.exception_handler(UnprocessableError)
    async def _unprocessable(_r: Request, e: UnprocessableError):
        return JSONResponse(_engine_body(e), status_code=422)

    @app.exception_handler(EngineError)
    async def _ee(_r: Request, e: EngineError):
        return JSONResponse(_engine_body(e), status_code=400)

    @app.exception_handler(StarletteHTTPException)
    async def _http(_r: Request, e: StarletteHTTPException):
        # свои тексты (например, «Нужен токен доступа.») оставляем, английские стандартные — переводим
        text = _HTTP_RU.get(e.status_code) if e.detail in (None, "", _HTTP_EN.get(e.status_code)) else str(e.detail)
        return JSONResponse({"detail": text or str(e.detail)}, status_code=e.status_code, headers=e.headers)

    @app.exception_handler(RequestValidationError)
    async def _422(_r: Request, e: RequestValidationError):
        return JSONResponse({"detail": _msg_422(e)}, status_code=422)

    @app.exception_handler(Exception)
    async def _500(_r: Request, e: Exception):
        log.error("Необработанная ошибка запроса: %s", redact(f"{type(e).__name__}: {e}"), exc_info=False)
        return JSONResponse({"detail": f"Внутренняя ошибка программы ({type(e).__name__})."}, status_code=500)

    for r in (health.router, projects.router, scenes.router, files.router, settings_routes.router, mic.router,
              llm.router, jobs_routes.router):
        app.include_router(r)

    # ───────────── WebSocket ─────────────
    @app.websocket("/ws/jobs")
    async def ws_jobs(ws: WebSocket):
        if not token_ok(token, extract_token(ws.headers, ws.query_params)):
            await ws.close(code=4401)
            return
        await ws.accept()
        q = jobs.subscribe()

        async def pump(tg: anyio.abc.TaskGroup) -> None:
            for j in jobs.active():
                await ws.send_text(j.model_dump_json())
            while True:
                await ws.send_text((await q.get()).model_dump_json())

        async def watch_disconnect(tg: anyio.abc.TaskGroup) -> None:
            while True:                      # клиент ничего не шлёт; receive() вернёт disconnect, когда он уйдёт
                msg = await ws.receive()
                if msg.get("type") == "websocket.disconnect":
                    tg.cancel_scope.cancel()
                    return

        try:
            async with anyio.create_task_group() as tg:
                tg.start_soon(pump, tg)
                tg.start_soon(watch_disconnect, tg)
        except (WebSocketDisconnect, RuntimeError, OSError):
            pass
        finally:
            jobs.unsubscribe(q)

    return app
