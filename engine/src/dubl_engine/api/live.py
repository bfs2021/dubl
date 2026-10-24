"""Честные статусы «разбор» и «подготовка сцены»: сохранённое на диске + что реально идёт в очереди задач.

На диске лежит только то, что точно известно (analysis записан целиком / stems_ready и файлы на месте). «Идёт сейчас»
знает только очередь задач, поэтому поверх сохранённого состояния накладываем её: нет результата и нет работающей
задачи — значит разбор (подготовка) прерван, как бы ни закончилась прошлая попытка (ошибка, отмена, закрытое окно).
"""
from __future__ import annotations

from ..jobs import JobManager
from ..models import AnalysisState, Project, Scene
from ..store import ProjectSummary


def _active(jobs: JobManager, kind: str, pid: str, scid: str | None = None) -> bool:
    return any(j.kind == kind and j.project_id == pid and (scid is None or j.scene_id == scid) for j in jobs.active())


def analysis_state(jobs: JobManager, pid: str, has_analysis: bool) -> AnalysisState:
    if has_analysis:
        return "done"
    return "running" if _active(jobs, "analyze", pid) else "interrupted"


def live_scene(scene: Scene, jobs: JobManager, pid: str) -> Scene:
    if scene.stems_ready:
        state = "ready"
    else:
        state = "running" if _active(jobs, "prepare_scene", pid, scene.id) else "interrupted"
    return scene.model_copy(update={"prepare_state": state}) if scene.prepare_state != state else scene


def live_project(project: Project, jobs: JobManager) -> Project:
    state = analysis_state(jobs, project.id, project.analysis is not None)
    scenes = [live_scene(sc, jobs, project.id) for sc in project.scenes]
    return project.model_copy(update={"analysis_state": state, "scenes": scenes})


def live_summaries(items: list[ProjectSummary], jobs: JobManager) -> list[ProjectSummary]:
    running = {j.project_id for j in jobs.active() if j.kind == "analyze"}
    out = []
    for s in items:
        state: AnalysisState = "done" if s.has_analysis else ("running" if s.id in running else "interrupted")
        out.append(s.model_copy(update={"analysis_state": state}) if s.analysis_state != state else s)
    return out
