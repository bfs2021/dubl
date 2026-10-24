"""Сценарии: разбор проекта, подготовка сцены, добавление дубля, рендер, склейка сцен, копия для плеера."""
from .analyze import ANALYZE_STEPS, analyze_project
from .join import project_render_steps, render_project, validate_scene_ids
from .proxy import PROXY_STEPS, build_proxy, proxy_ready
from .scene import (
           PREPARE_STEPS,
           RENDER_STEPS,
           add_take,
           job_result,
           prepare_scene,
           preview_scene,
           render_scene,
)

__all__ = ["ANALYZE_STEPS", "PREPARE_STEPS", "PROXY_STEPS", "RENDER_STEPS", "add_take", "analyze_project",
           "build_proxy", "job_result", "prepare_scene", "preview_scene", "project_render_steps", "proxy_ready",
           "render_project", "render_scene", "validate_scene_ids"]
