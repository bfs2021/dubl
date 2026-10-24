from __future__ import annotations

from fastapi import APIRouter, Depends
from fastapi.concurrency import run_in_threadpool

from ...llm.router import no_model_text, provider_for, who_answered
from ...models import LLMTask, SceneLine
from ...util import EngineError
from ..deps import SettingsDep, StoreDep, require_token
from ..schemas import (
    DirectorCommandResult,
    DirectorReport,
    DirectorTakeResult,
    ErrorResponse,
    FitLengthResult,
    LineVariants,
    LLMRequest,
    NameSpeakersResult,
    TranslatedLine,
    TranslateResult,
)

router = APIRouter(prefix="/llm", tags=["llm"], dependencies=[Depends(require_token)],
                   responses={400: {"model": ErrorResponse}, 404: {"model": ErrorResponse}})


def _no_model(task: LLMTask, settings) -> EngineError:
    """Ошибка «отвечать некому» с причиной (какая модель выбрана и что с ней)."""
    return EngineError(no_model_text(task, settings.llm))


def _pick(lines: list[SceneLine], body: LLMRequest) -> list[SceneLine]:
    return [ln for ln in lines if not body.line_ids or ln.line_id in body.line_ids]


def _scene(store, body: LLMRequest):
    project = store.get(body.project_id)
    if not body.scene_id:
        raise EngineError("Нужна сцена (scene_id).")
    return project, store.find_scene(project, body.scene_id)


@router.post("/translate")
async def translate(body: LLMRequest, store: StoreDep, settings: SettingsDep) -> TranslateResult:
    """Перевести реплики сцены. Перевод человека (translation_source='user') не перетирается и ИИ не отправляется —
    в ответе такая реплика возвращается с её текстом и source='user'. Остальное (в том числе перевод из файла)
    перезаписывается и сразу сохраняется в сцену с translation_source='llm'.
    ai_model — кто ответил; ai_fallback — ответила запасная модель."""
    from ...llm.tasks import translate_lines
    provider = await run_in_threadpool(provider_for, LLMTask.TRANSLATE, settings.llm)
    if provider is None:
        raise _no_model(LLMTask.TRANSLATE, settings)
    _p, scene = _scene(store, body)
    lines = _pick(scene.lines, body)
    todo = [ln for ln in lines if ln.translation_source != "user"]
    texts = dict(zip((ln.line_id for ln in todo), await translate_lines(todo, body.target_lang, provider), strict=True))
    if texts:
        with store.edit(body.project_id) as p:
            sc = p.scenes[store.scene_index(p, scene.id)]
            for ln in sc.lines:
                if ln.line_id in texts and ln.translation_source != "user":
                    ln.text, ln.translation_source = texts[ln.line_id], "llm"
    model, fb = who_answered(provider)
    return TranslateResult(translations=[
        TranslatedLine(line_id=ln.line_id, text=texts.get(ln.line_id, ln.text),
                       source="llm" if ln.line_id in texts else ln.translation_source) for ln in lines],
        ai_model=model, ai_fallback=fb)


@router.post("/fit_length")
async def fit_length(body: LLMRequest, store: StoreDep, settings: SettingsDep) -> FitLengthResult:
    from ...llm.tasks import fit_line
    provider = await run_in_threadpool(provider_for, LLMTask.FIT_LENGTH, settings.llm)
    if provider is None:
        raise _no_model(LLMTask.FIT_LENGTH, settings)
    _p, scene = _scene(store, body)
    variants = [LineVariants(line_id=ln.line_id, texts=await fit_line(ln, provider, n=body.n)) for ln in _pick(scene.lines, body)]
    model, fb = who_answered(provider)
    return FitLengthResult(variants=variants, ai_model=model, ai_fallback=fb)


@router.post("/name_speakers")
async def name_speakers_route(body: LLMRequest, store: StoreDep, settings: SettingsDep) -> NameSpeakersResult:
    from ...llm.tasks import name_speakers
    provider = await run_in_threadpool(provider_for, LLMTask.NAME_SPEAKERS, settings.llm)
    if provider is None:
        raise _no_model(LLMTask.NAME_SPEAKERS, settings)
    p = store.get(body.project_id)
    if p.analysis is None:
        raise EngineError("Проект ещё не разобран.")
    names = await name_speakers(p.analysis.lines, p.analysis.speakers, provider)
    model, fb = who_answered(provider)
    return NameSpeakersResult(names=names, ai_model=model, ai_fallback=fb)


@router.post("/director")
async def director(body: LLMRequest, store: StoreDep, settings: SettingsDep) -> DirectorTakeResult:
    """Помощник-режиссёр разбирает дубль (нужен take_id): замеры + 1–2 фразы совета. Работает и без модели — по правилам.
    Просьбы словами («кот громче») — в POST /llm/director_command."""
    from ...llm.director import director_feedback, take_report
    if not body.take_id:
        raise EngineError("Для разбора нужен дубль (take_id). Просьбы словами — /llm/director_command.")
    provider = await run_in_threadpool(provider_for, LLMTask.DIRECTOR, settings.llm)
    _p, scene = _scene(store, body)
    t = store.find_take(scene, body.take_id)
    line = next((ln for ln in scene.lines if ln.line_id == t.line_id), None)
    if line is None:
        raise EngineError("У этого дубля нет реплики для сравнения.")
    report = take_report(t, line)
    feedback = await director_feedback(report, provider)
    model, fb = who_answered(provider)
    return DirectorTakeResult(report=DirectorReport.model_validate(report), feedback=feedback, ai_model=model, ai_fallback=fb)


@router.post("/director_command")
async def director_command_route(body: LLMRequest, store: StoreDep, settings: SettingsDep) -> DirectorCommandResult:
    """Просьба обычными словами → проверенные изменения сведения (каждое с note) и итоговые настройки mix."""
    from ...llm.director import apply_patches, director_command
    if not (body.text or "").strip():
        raise EngineError("Напиши, что сделать.")
    provider = await run_in_threadpool(provider_for, LLMTask.DIRECTOR, settings.llm)
    if provider is None:
        raise _no_model(LLMTask.DIRECTOR, settings)
    p, scene = _scene(store, body)
    patches = await director_command(body.text or "", scene, provider,
                                     speakers=p.analysis.speakers if p.analysis else None)
    model, fb = who_answered(provider)
    return DirectorCommandResult(patches=patches, mix=apply_patches(scene.mix, patches), ai_model=model, ai_fallback=fb)
