from __future__ import annotations

from typing import Annotated, Literal

from fastapi import APIRouter, Depends, File, Form, Response, UploadFile

from ...models import Job, RenderSettings, Scene, Take
from ...pipeline import (
    PREPARE_STEPS,
    RENDER_STEPS,
    add_take,
    job_result,
    prepare_scene,
    preview_scene,
    render_scene,
)
from ...pipeline.scene import MAX_SCENE_S, MIN_SCENE_S, delete_take, resolve_output
from ...pipeline.takes import set_user_choice, trim_tail, untrim_tail
from ...store import NotFoundError
from ...util import ConflictError, EngineError
from ..deps import JobsDep, ProjectDep, SceneDep, SettingsDep, StoreDep, require_token
from ..live import live_scene
from ..schemas import ErrorResponse, Ok, SceneCreate, ScenePatch, TakePatch

router = APIRouter(prefix="/projects/{pid}/scenes", tags=["scenes"], dependencies=[Depends(require_token)],
                   responses={404: {"model": ErrorResponse}})

MAX_UPLOAD = 1024 * 1024 * 1024      # 1 ГБ: часовая запись mono 48 кГц float32 — 690 МБ


@router.post("", responses={200: {"headers": {"X-Job-Id": {"description": "Id задачи подготовки сцены",
                                                           "schema": {"type": "string"}}}},
                           400: {"model": ErrorResponse}})
def create_scene(pid: str, body: SceneCreate, response: Response, store: StoreDep, jobs: JobsDep,
                 settings: SettingsDep) -> Scene:
    """Создать сцену и поставить в очередь вырезку и разделение голосов. Id задачи — в заголовке X-Job-Id."""
    with store.edit(pid) as p:
        dur = p.media.duration
        if body.end - body.start < MIN_SCENE_S:
            raise EngineError("Сцена слишком короткая.")
        if body.end - body.start > MAX_SCENE_S:
            raise EngineError(f"Сцена слишком длинная: за раз можно взять не больше {int(MAX_SCENE_S // 60)} минут.")
        if dur and body.end > dur + 0.01:
            raise EngineError("Сцена выходит за конец видео.")
        scid = store.next_scene_id(p)
        sc = Scene(id=scid, title=body.title.strip() or f"Сцена {len(p.scenes) + 1}", start=body.start, end=body.end)
        p.scenes.append(sc)
    job = jobs.submit("prepare_scene", pid, PREPARE_STEPS,
                      lambda ctx: prepare_scene(store, pid, scid, ctx, settings=settings).model_dump(mode="json"),
                      scene_id=scid)
    response.headers["X-Job-Id"] = job.id
    return live_scene(sc, jobs, pid)


@router.get("/{scid}")
def get_scene(scene: SceneDep) -> Scene:
    """Сцена. prepare_state: ready — голоса и фон готовы (stems_ready=true, все файлы на месте); running — готовится
    (или в очереди); interrupted — подготовка не дошла до конца (закрыли программу, отмена, ошибка) и не идёт:
    перезапуск — POST …/prepare."""
    return scene


@router.post("/{scid}/prepare", responses={200: {"headers": {"X-Job-Id": {"description": "Id задачи подготовки сцены",
                                                                         "schema": {"type": "string"}}}},
                                          400: {"model": ErrorResponse}, 409: {"model": ErrorResponse}})
def prepare(pid: str, scid: str, scene: SceneDep, response: Response, store: StoreDep, jobs: JobsDep,
            settings: SettingsDep) -> Job:
    """Подготовить сцену заново (вырезка, звук, голоса и фон): после прерванной подготовки или чтобы пересчитать.
    Пока идёт, stems_ready=false; готовые дубли и настройки сведения остаются."""
    if any(j.project_id == pid and j.kind == "prepare_scene" and j.scene_id == scid for j in jobs.active()):
        raise ConflictError("Эта сцена уже готовится.")
    job = jobs.submit("prepare_scene", pid, PREPARE_STEPS,
                      lambda ctx: prepare_scene(store, pid, scid, ctx, settings=settings).model_dump(mode="json"),
                      scene_id=scid)
    response.headers["X-Job-Id"] = job.id
    return job


@router.delete("/{scid}")
def delete_scene(pid: str, scid: str, store: StoreDep, jobs: JobsDep) -> Ok:
    store.find_scene(store.get(pid), scid)
    for j in jobs.active():
        if j.project_id == pid and (j.scene_id == scid or j.kind == "render_project"):   # склейка использует и эту сцену
            jobs.cancel(j.id)
    jobs.wait_project_idle(pid, scid)
    with store.edit(pid) as p:
        del p.scenes[store.scene_index(p, scid)]
    store.delete_scene_files(pid, scid)
    return Ok()


@router.patch("/{scid}")
def patch_scene(pid: str, scid: str, body: ScenePatch, store: StoreDep, jobs: JobsDep) -> Scene:
    with store.edit(pid) as p:
        sc = p.scenes[store.scene_index(p, scid)]
        if body.title is not None:
            sc.title = body.title
        if body.text_mode is not None:
            sc.text_mode = body.text_mode
        if body.dubbed_speakers is not None:
            sc.dubbed_speakers = list(dict.fromkeys(body.dubbed_speakers))
        if body.mix is not None:
            sc.mix = body.mix
        if body.lines:
            by_id = {ln.line_id: ln for ln in sc.lines}
            for lp in body.lines:
                if lp.line_id not in by_id:
                    raise NotFoundError(f"В сцене нет реплики {lp.line_id}.")
            for lp in body.lines:
                ln = by_id[lp.line_id]
                if lp.text == ln.text and lp.source is None:
                    continue
                ln.text = lp.text
                if lp.source is not None:
                    ln.translation_source = lp.source
                elif not lp.text.strip() or lp.text == ln.original:
                    ln.translation_source = None
                else:
                    ln.translation_source = "user"
        return live_scene(sc.model_copy(deep=True), jobs, pid)


# ───────────── дубли ─────────────
async def _read_upload(file: UploadFile) -> bytes:
    data = await file.read(MAX_UPLOAD + 1)
    if not data:
        raise EngineError("Пустая запись.")
    if len(data) > MAX_UPLOAD:
        raise EngineError("Запись слишком большая.")
    return data


@router.post("/{scid}/takes", responses={400: {"model": ErrorResponse}, 409: {"model": ErrorResponse}})
async def upload_take(pid: str, scid: str, scene: SceneDep, store: StoreDep, jobs: JobsDep, settings: SettingsDep,
                file: Annotated[UploadFile, File(description="WAV/любой звук, который понимает ffmpeg")],
                line_id: Annotated[str | None, Form()] = None,
                start_rel: Annotated[float, Form(ge=0.0, le=1e7, allow_inf_nan=False)] = 0.0,
                preroll: Annotated[float, Form(ge=0.0, le=600.0, allow_inf_nan=False)] = 0.0,
                source: Annotated[Literal["mic", "file"], Form(description="mic — запись с микрофона под видео; file — "
                                  "дубль загружен файлом (привязки ко времени нет)")] = "mic") -> Take:
    """Дубль: нормализация, замеры, автоподгонка под реплику, разбор по тексту (если есть нейросети), ★ по умолчанию
    самому полному дублю. Почти тишина (нет речи и пик ниже −40 дБFS, либо речь тише −50 дБFS по RMS) всё равно
    сохраняется, но с empty=true: ★ ему не ставится, в сведение он не идёт, предупреждение живёт и исчезает вместе с
    дублем. source=file — файл, а не запись: подгонка ставит его на место реплики, но «начал раньше/позже» тут не про него."""
    if not scene.stems_ready:
        raise ConflictError("Сцена ещё готовится — голоса и фон не разделены.")
    data = await _read_upload(file)
    # ffmpeg, подгонка и распознавание — в лёгком пуле очереди (не больше 2 одновременно), а не в потоках запроса без ограничения
    return await jobs.run_light(add_take, store, pid, scid, data, line_id or None, start_rel, preroll, settings, source)


@router.patch("/{scid}/takes/{tid}", responses={409: {"model": ErrorResponse}})
def patch_take(pid: str, scid: str, tid: str, body: TakePatch, store: StoreDep) -> Take:
    """Сдвиг, длина и ★. selected=true — выбор человека: автовыбор его потом не меняет. Пустой дубль выбрать нельзя (409)."""
    with store.edit(pid) as p:
        sc = p.scenes[store.scene_index(p, scid)]
        t = store.find_take(sc, tid)
        if body.offset_ms is not None:
            t.offset_ms = body.offset_ms
        if body.stretch is not None:
            t.stretch = body.stretch
        if body.selected is not None:
            set_user_choice(sc, t, body.selected)
        return t.model_copy(deep=True)


@router.post("/{scid}/takes/{tid}/trim_tail", responses={409: {"model": ErrorResponse}})
async def trim_take_tail(pid: str, scid: str, tid: str, store: StoreDep, jobs: JobsDep, settings: SettingsDep) -> Take:
    """Отрезать хвост-перезапуск («давай ещё раз») с места Take.tail_cut_s. Полный файл остаётся: untrim_tail вернёт его.
    409 — отрезать нечего."""
    return await jobs.run_light(trim_tail, store, pid, scid, tid, settings)


@router.post("/{scid}/takes/{tid}/untrim_tail", responses={409: {"model": ErrorResponse}})
async def untrim_take_tail(pid: str, scid: str, tid: str, store: StoreDep, jobs: JobsDep, settings: SettingsDep) -> Take:
    """Вернуть дубль целиком, вместе с хвостом (отмена trim_tail)."""
    return await jobs.run_light(untrim_tail, store, pid, scid, tid, settings)


@router.delete("/{scid}/takes/{tid}")
def remove_take(pid: str, scid: str, tid: str, store: StoreDep) -> Ok:
    delete_take(store, pid, scid, tid)
    return Ok()


# ───────────── превью и рендер ─────────────
@router.post("/{scid}/preview", responses={409: {"model": ErrorResponse}})
def preview(pid: str, scene: SceneDep, store: StoreDep, jobs: JobsDep) -> Job:
    if not scene.stems_ready:
        raise ConflictError("Сцена ещё готовится — голоса и фон не разделены.")
    return jobs.submit("preview", pid, RENDER_STEPS,
                       lambda ctx: job_result(store, pid, preview_scene(store, pid, scene.id, ctx)), scene_id=scene.id)


@router.post("/{scid}/render", responses={400: {"model": ErrorResponse}, 409: {"model": ErrorResponse}})
def render(pid: str, scene: SceneDep, project: ProjectDep, body: RenderSettings, store: StoreDep, jobs: JobsDep) -> Job:
    """Собрать ролик сцены. Файл с таким именем уже есть и body.overwrite=false → итог «<имя> (2).mp4», «(3)» …
    Итоговый путь — Job.result.path (и result.url, если файл внутри папки проекта)."""
    if not scene.stems_ready:
        raise ConflictError("Сцена ещё готовится — голоса и фон не разделены.")
    resolve_output(store, project, body)         # ошибку в пути показываем сразу, а не после минут работы
    return jobs.submit("render", pid, RENDER_STEPS,
                       lambda ctx: job_result(store, pid, render_scene(store, pid, scene.id, body, ctx)),
                       scene_id=scene.id)
