"""Лёгкая копия исходного видео (540p h264+aac) для плеера на экране «Разбор»."""
from __future__ import annotations

from pathlib import Path

from ..media.extract import make_proxy
from ..media.probe import browser_playable
from ..models import Project
from ..store import ProjectStore
from ..util import EngineError

PROXY_STEPS = [("proxy", "Лёгкая копия для плеера")]


def proxy_ready(store: ProjectStore, pid: str) -> bool:
    """Копия есть и не старше исходника (исходник могли подменить)."""
    p = store.proxy_path(pid)
    if not p.is_file() or p.stat().st_size == 0:
        return False
    try:
        src = Path(store.get(pid).source_path)
        return not src.is_file() or p.stat().st_mtime >= src.stat().st_mtime
    except OSError:
        return True


def build_proxy(store: ProjectStore, pid: str, ctx) -> Path:
    project = store.get(pid)
    if not project.source_path or not Path(project.source_path).is_file():
        raise EngineError("Исходное видео недоступно — копию для плеера сделать не из чего.")
    idx = project.analysis.audio_stream_index if project.analysis else None
    with ctx.step("proxy"):
        return make_proxy(project.source_path, store.proxy_path(pid), audio_stream_index=idx,
                          on_progress=lambda v: ctx.progress("proxy", v))


def playback_info(project: Project) -> tuple[bool, str | None]:
    idx = project.analysis.audio_stream_index if project.analysis else None
    if not project.source_path:
        return False, "видео ещё скачивается"
    return browser_playable(project.media, idx)
