from __future__ import annotations

import shutil
from pathlib import Path

from fastapi import APIRouter, Depends, Response

from ...media.probe import probe as probe_media
from ...models import Job, MediaInfo, Project
from ...pipeline import ANALYZE_STEPS, analyze_project, job_result, project_render_steps, render_project
from ...pipeline.join import validate_scene_ids
from ...pipeline.scene import resolve_output
from ...store import NotFoundError, ProjectSummary
from ...util import ConflictError, EngineError
from ..deps import JobsDep, ProjectDep, SettingsDep, StoreDep, require_token
from ..live import live_project, live_summaries
from ..schemas import (
    AnalyzeRequest,
    ErrorResponse,
    LineSpeakerPatch,
    LinesMerge,
    MergedLinesProject,
    Ok,
    ProbeRequest,
    MergedProject,
    ProjectCreate,
    ProjectRename,
    RenderProjectRequest,
    SnapRequest,
    SnapResult,
    SpeakerMerge,
    SpeakerRename,
    UrlProbe,
    file_url_to_path,
)

router = APIRouter(tags=["projects"], dependencies=[Depends(require_token)],
                   responses={404: {"model": ErrorResponse}})


def clean_path(raw: str) -> Path:
    """Путь, как его вставляет человек: с кавычками и пробелами по краям, ~ в начале."""
    s = raw.strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in "\"'":
        s = s[1:-1].strip()
    if s.lower().startswith("file:"):                  # file:///Users/me/a%20b.mp4 → обычный путь
        s = file_url_to_path(s)
    return Path(s).expanduser()


def _existing_video(raw: str) -> tuple[Path, MediaInfo]:
    src = clean_path(raw)
    if not src.is_file():
        raise EngineError(f"Файл не найден: {src}")
    media = probe_media(src)
    if media.width <= 0 or media.height <= 0:
        raise EngineError("В файле нет видео — озвучивать нечего. Выбери видеофайл.")
    if not media.audio_streams:
        raise EngineError("В файле нет звука — озвучивать поверх нечего. Выбери файл со звуковой дорожкой.")
    return src, media


@router.post("/probe", responses={400: {"model": ErrorResponse}, 422: {"model": ErrorResponse}})
def probe(body: ProbeRequest) -> MediaInfo | UrlProbe:
    """Что известно до создания проекта. path → MediaInfo (дорожки, субтитры, главы). url → UrlProbe: название,
    длительность, есть ли субтитры — только метаданные (не дольше 15 с), видео не скачивается. Ссылка не ведёт на
    доступное видео → 422 с понятным текстом. Нет сети/таймаут → 200, known=false и message «не удалось проверить
    ссылку». file://… — то же, что path."""
    if body.url:
        from ...pipeline.download import probe_url
        return UrlProbe.model_validate(probe_url(body.url))
    return _existing_video(body.path or "")[1]


@router.get("/projects")
def list_projects(store: StoreDep, jobs: JobsDep) -> list[ProjectSummary]:
    """Сводки проектов. analysis_state: running — разбор идёт (или в очереди); interrupted — результата нет и разбор не
    идёт (закрыли программу, отмена, ошибка): перезапуск — POST /projects/{pid}/analyze; done — has_analysis=true."""
    return live_summaries(store.list(), jobs)


JOB_HEADER = {"X-Job-Id": {"description": "Id фоновой задачи (следить за ней: GET /jobs/{id} или WS /ws/jobs)",
                           "schema": {"type": "string"}}}


@router.post("/projects", responses={200: {"headers": JOB_HEADER}, 400: {"model": ErrorResponse}})
def create_project(body: ProjectCreate, response: Response, store: StoreDep, jobs: JobsDep,
                   settings: SettingsDep) -> Project:
    """Создать проект и сразу поставить разбор в очередь. Id задачи — в заголовке X-Job-Id."""
    if body.path:
        src, media = _existing_video(body.path)
        project = store.create(src, media, title=body.title)
        pid = project.id

        def job_fn(ctx):
            return analyze_project(store, pid, ctx, settings=settings,
                                   audio_stream_index=body.audio_stream_index, language=body.language)
        steps = list(ANALYZE_STEPS)
    else:
        from ...pipeline.download import download_video
        media = MediaInfo(path="", duration=0.0, container="", width=0, height=0, fps=0.0, video_codec="")
        project = store.create("", media, title=body.title or "Загрузка по ссылке")
        pid, url = project.id, body.url or ""
        steps = [("download", "Скачивание видео"), *ANALYZE_STEPS]

        def job_fn(ctx):
            with ctx.step("download"):
                path, title = download_video(url, store.downloads_root() / pid,
                                             on_progress=lambda v: ctx.progress("download", v))
            with store.edit(pid) as p:
                p.source_path = str(path)
                if not body.title:
                    p.title = title
            return analyze_project(store, pid, ctx, settings=settings,
                                   audio_stream_index=body.audio_stream_index, language=body.language)

    job = jobs.submit("analyze", pid, steps, job_fn)
    response.headers["X-Job-Id"] = job.id
    return live_project(project, jobs)


@router.post("/projects/{pid}/analyze", responses={400: {"model": ErrorResponse}, 409: {"model": ErrorResponse}})
def reanalyze_project(pid: str, project: ProjectDep, store: StoreDep, jobs: JobsDep, settings: SettingsDep,
                      body: AnalyzeRequest | None = None) -> Job:
    """Разобрать проект заново (после ошибки или когда докачали нейросети). Прежний разбор заменяется, сцены остаются."""
    if any(j.project_id == pid and j.kind == "analyze" for j in jobs.active()):
        raise ConflictError("Разбор этого проекта уже идёт.")
    if not project.source_path or not Path(project.source_path).is_file():
        raise EngineError("Видео не найдено на диске (или ещё не скачано по ссылке). Создай проект заново.")
    req = body or AnalyzeRequest()

    def job_fn(ctx):
        return analyze_project(store, pid, ctx, settings=settings,
                               audio_stream_index=req.audio_stream_index, language=req.language)
    return jobs.submit("analyze", pid, list(ANALYZE_STEPS), job_fn)


@router.get("/projects/{pid}")
def get_project(project: ProjectDep) -> Project:
    return project


@router.delete("/projects/{pid}")
def delete_project(pid: str, store: StoreDep, jobs: JobsDep) -> Ok:
    """Удалить проект: идущие задачи останавливаются, стираются папка проекта и видео, которое программа сама скачала
    по ссылке (<данные>/downloads/…). Файл, который пользователь выбрал на своём диске, не трогаем никогда."""
    store.project_dir(pid)               # проверка id
    if not store.exists(pid):
        raise NotFoundError("Проект не найден.")
    for j in jobs.active():
        if j.project_id == pid:
            jobs.cancel(j.id)
    idle = jobs.wait_project_idle(pid, timeout=30.0)        # задача успевает остановиться, а не пишет в удалённую папку
    store.delete(pid)
    if not idle:
        _sweep_later(store, jobs, pid)
    return Ok()


def _sweep_later(store, jobs, pid: str) -> None:
    """Задача не остановилась за 30 с и могла успеть создать папку заново — дочищаем, когда она закончится."""
    import threading

    def sweep() -> None:
        jobs.wait_project_idle(pid, timeout=600.0)
        try:
            if not store.exists(pid):
                shutil.rmtree(store.project_dir(pid), ignore_errors=True)
                shutil.rmtree(store.downloads_root() / pid, ignore_errors=True)
        except NotFoundError:
            pass
    threading.Thread(target=sweep, daemon=True, name="dubl-delete-sweep").start()


@router.patch("/projects/{pid}", responses={400: {"model": ErrorResponse}})
def rename_project(pid: str, body: ProjectRename, store: StoreDep, jobs: JobsDep) -> Project:
    """Переименовать проект (только название; папка и файлы остаются на месте)."""
    title = body.title.strip()
    if not title:
        raise EngineError("Название не может быть пустым.")
    with store.edit(pid) as p:
        p.title = title
    return live_project(store.get(pid), jobs)


@router.patch("/projects/{pid}/speakers/{sid}")
def rename_speaker(pid: str, sid: str, body: SpeakerRename, store: StoreDep, jobs: JobsDep) -> Project:
    with store.edit(pid) as p:
        sp = next((s for s in (p.analysis.speakers if p.analysis else []) if s.id == sid), None)
        if sp is None:
            raise NotFoundError("Такого говорящего нет.")
        name = (body.name or "").strip() or None
        sp.name = name
        if name:
            sp.label = name
    return live_project(store.get(pid), jobs)


@router.post("/projects/{pid}/speakers/merge", responses={400: {"model": ErrorResponse}})
def merge_speakers(pid: str, body: SpeakerMerge, store: StoreDep, jobs: JobsDep) -> MergedProject:
    """Влить голос b в голос a (любые два голоса проекта; кнопка «Объединить» и ручное слияние — один и тот же роут).
    Ответ — проект плюс merge_id: по нему слияние отменяется (POST …/speakers/merges/{merge_id}/undo)."""
    from ...pipeline.speakers import merge_speakers as do_merge
    project, merge_id = do_merge(store, pid, body.a, body.b)
    return MergedProject(**live_project(project, jobs).model_dump(), merge_id=merge_id)


@router.post("/projects/{pid}/speakers/merges/{merge_id}/undo",
             responses={404: {"model": ErrorResponse}, 409: {"model": ErrorResponse}})
def undo_speaker_merge(pid: str, merge_id: str, store: StoreDep, jobs: JobsDep) -> Project:
    """Отменить слияние: голос, его реплики, дубли и имя возвращаются как были. 404 — такого слияния нет или оно уже
    отменено; 409 — после него были другие слияния этих голосов, сначала отмени их."""
    from ...pipeline.speakers import undo_merge
    return live_project(undo_merge(store, pid, merge_id), jobs)


# ───────────── правка реплик разбора ─────────────
@router.patch("/projects/{pid}/lines/{line_id}/speaker", responses={400: {"model": ErrorResponse}, 409: {"model": ErrorResponse}})
def set_line_speaker(pid: str, line_id: str, body: LineSpeakerPatch, store: StoreDep, jobs: JobsDep) -> Project:
    """Сменить персонажа у реплики (в разборе и во всех сценах, вместе с её дублями). new_speaker_name — новый персонаж.
    409 — идёт разбор или подготовка сцены."""
    from ...pipeline.lines import set_line_speaker as do_set
    return live_project(do_set(store, pid, line_id, body.speaker_id, body.new_speaker_name, jobs=jobs), jobs)


@router.post("/projects/{pid}/lines/merge", responses={400: {"model": ErrorResponse}, 409: {"model": ErrorResponse}})
def merge_lines(pid: str, body: LinesMerge, store: StoreDep, jobs: JobsDep) -> MergedLinesProject:
    """Склеить соседние реплики ОДНОГО персонажа в одну. Ответ — проект, merge_id для отмены, moved_takes.
    400 — реплики не подряд, разных персонажей («сначала смени персонажа у одной из них»), или слишком длинно.
    В analysis.merged_lines появляется запись «склеенная реплика → merge_id» (для «Разделить»)."""
    from ...pipeline.lines import merge_lines as do_merge
    project, merge_id, moved = do_merge(store, pid, body.line_ids, jobs=jobs)
    return MergedLinesProject(**live_project(project, jobs).model_dump(), merge_id=merge_id, moved_takes=moved)


@router.post("/projects/{pid}/lines/merges/{merge_id}/undo",
             responses={404: {"model": ErrorResponse}, 409: {"model": ErrorResponse}})
def undo_merge_lines(pid: str, merge_id: str, store: StoreDep, jobs: JobsDep) -> Project:
    """Отменить склейку: реплики, их текст, персонаж, дубли и ★ — как были (карта analysis.merged_lines — тоже)."""
    from ...pipeline.lines import undo_merge_lines as do_undo
    return live_project(do_undo(store, pid, merge_id, jobs=jobs), jobs)


@router.post("/projects/{pid}/lines/{line_id}/split",
             responses={404: {"model": ErrorResponse}, 409: {"model": ErrorResponse}})
def split_line(pid: str, line_id: str, store: StoreDep, jobs: JobsDep) -> Project:
    """«Разделить обратно» склеенную реплику (ручная склейка или авто-склейка при разборе): отменяется последняя
    склейка, в которую она вошла (analysis.merged_lines[line_id]); цепочка склеек разбирается по шагу за вызов.
    404 — реплика не склеивалась (или отменять уже нечего); 409 — идёт разбор/подготовка сцены или склейку уже
    нельзя отменить."""
    from ...pipeline.lines import split_line as do_split
    return live_project(do_split(store, pid, line_id, jobs=jobs), jobs)


@router.post("/projects/{pid}/snap", responses={400: {"model": ErrorResponse}})
def snap(project: ProjectDep, body: SnapRequest) -> SnapResult:
    """Выделение на шкале: прилипание к репликам и склейкам, разговор целиком или кадр."""
    from ...analysis.snap import dialogue_around, shot_around, snap_selection
    an = project.analysis
    if an is None:
        raise EngineError("Проект ещё не разобран.")
    if body.mode == "dialogue":
        s, e = dialogue_around(body.a, an.lines)
    elif body.mode == "shot":
        s, e = shot_around(body.a, an.shot_cuts, project.media.duration)
    else:
        if body.b is None:
            raise EngineError("Для свободного выделения нужны оба края (a и b).")
        a, b = sorted((body.a, body.b))
        s, e = snap_selection(a, b, an.lines, an.shot_cuts)
    return SnapResult(start=s, end=e)


@router.post("/projects/{pid}/render", responses={400: {"model": ErrorResponse}, 409: {"model": ErrorResponse}})
def render_joined(pid: str, body: RenderProjectRequest, store: StoreDep, jobs: JobsDep) -> Job:
    """Склейка сцен в один ролик (kind='render_project'). Порядок scene_ids = порядок в ролике. Каждая сцена сводится и
    мастерится как обычно, видео склеиваются встык, на стыках звук 80 мс переходом, затем мастеринг всего ролика.
    Итог: job.result.path и job.result.url. Одна сцена тоже работает (то же, что POST …/scenes/{scid}/render).
    Файл с таким именем уже есть и settings.overwrite=false → «<имя> (2).mp4», «(3)» … (итоговый путь — в job.result.path)."""
    project = store.get(pid)
    validate_scene_ids(store, pid, body.scene_ids)
    resolve_output(store, project, body.settings)           # ошибку в пути показываем сразу
    ids = list(body.scene_ids)
    return jobs.submit("render_project", pid, project_render_steps(store, pid, ids),
                       lambda ctx: job_result(store, pid, render_project(store, pid, ids, body.settings, ctx)))
