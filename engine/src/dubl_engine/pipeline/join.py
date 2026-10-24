"""Склейка нескольких сцен в один ролик и лёгкая копия исходника для плеера."""
from __future__ import annotations

import logging
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from ..models import RenderSettings, SceneLine, SubtitleMode
from ..store import ProjectStore
from ..util import ConflictError, EngineError
from .outputs import release_output, reserve_output
from .scene import NOT_READY, _render, resolve_output, subtitle_lines

log = logging.getLogger(__name__)

PROJECT_RENDER_TAIL = [
    ("join", "Склейка сцен"),
    ("master", "Мастеринг ролика"),
    ("video", "Сборка ролика"),
]
MAX_JOIN_SCENES = 100


def project_render_steps(store: ProjectStore, pid: str, scene_ids: list[str]) -> list[tuple[str, str]]:
    project = store.get(pid)
    steps = []
    for i, scid in enumerate(scene_ids, 1):
        sc = store.find_scene(project, scid)
        steps.append((f"scene:{scid}", f"Сцена {i}: {sc.title}"))
    return [*steps, *PROJECT_RENDER_TAIL]


def validate_scene_ids(store: ProjectStore, pid: str, scene_ids: list[str]) -> None:
    """Сцены есть, готовы и не повторяются. Ошибки — сразу, а не после минут работы."""
    if not scene_ids:
        raise EngineError("Выбери хотя бы одну сцену.")
    if len(scene_ids) > MAX_JOIN_SCENES:
        raise EngineError(f"Сцен слишком много: за раз можно склеить не больше {MAX_JOIN_SCENES}.")
    if len(set(scene_ids)) != len(scene_ids):
        raise EngineError("Одна и та же сцена выбрана дважды.")
    project = store.get(pid)
    for scid in scene_ids:
        if not store.find_scene(project, scid).stems_ready:
            raise ConflictError(NOT_READY)


class _SceneCtx:
    """Контекст для _render одной сцены внутри общей задачи: три его шага (сведение, мастеринг, видео) складываются в
    прогресс одного шага `scene:<id>` общей задачи."""

    def __init__(self, parent, key: str):
        self._p, self._key, self._stage = parent, key, 0
        self.job_id = getattr(parent, "job_id", "null")

    @property
    def cancelled(self) -> bool:
        return bool(getattr(self._p, "cancelled", False))

    def check(self) -> None:
        self._p.check()

    @contextmanager
    def step(self, key: str):
        self._p.check()
        self._stage = {"mix": 0, "master": 1, "video": 2}.get(key, self._stage)
        self._p.progress(self._key, self._stage / 3)
        yield

    def progress(self, key: str, value: float) -> None:
        self._p.progress(self._key, (self._stage + min(max(value, 0.0), 1.0)) / 3)

    def skip(self, key: str) -> None:
        pass

    def set_result(self, result) -> None:
        pass

    def note(self, key: str, text: str) -> None:
        pass


def _merged_srt(items: list[tuple[float, list[SceneLine]]], out: Path, text: str = "dub") -> Path:
    from ..audio.render import write_subtitles

    lines = [ln.model_copy(update={"start_rel": ln.start_rel + off, "end_rel": ln.end_rel + off})
             for off, scene_lines in items for ln in scene_lines]
    return write_subtitles(lines, out, fmt="srt", text=text)


def render_project(store: ProjectStore, pid: str, scene_ids: list[str], rs: RenderSettings, ctx) -> Path:
    """Каждая сцена сводится и мастерится как обычно, затем:
    * видео склеиваются жёстко (у всех одинаковые размер и fps — это одна и та же дорожка исходника, один набор настроек);
    * звук сцен склеивается встык с переходом 80 мс на стыках (длина каждого куска = длине его видео, синхрон не уезжает);
    * весь ролик мастерится ещё раз под цель; true peak проверяется после AAC.
    Субтитры: burn — вшиты в каждую сцену; file — один .srt на весь ролик со сдвигами."""

    validate_scene_ids(store, pid, scene_ids)
    project = store.get(pid)
    out = reserve_output(resolve_output(store, project, rs), rs.overwrite)      # уже есть такой файл → « (2)»
    try:
        return _render_project(store, pid, scene_ids, rs, ctx, project, out)
    except BaseException:
        release_output(out, rs.overwrite)
        raise


def _render_project(store: ProjectStore, pid: str, scene_ids: list[str], rs: RenderSettings, ctx, project, out: Path
                    ) -> Path:
    from ..audio.io import read_wav, write_wav
    from ..audio.render import (
        SEAM_FADE_MS,
        concat_videos,
        fit_length,
        join_audio,
        master_for_aac,
        mux_video_audio,
        video_duration,
    )

    scene_rs = rs.model_copy(update={"export_stems": False})
    true_peak = min(store.find_scene(project, s).mix.true_peak_db for s in scene_ids)
    work = Path(tempfile.mkdtemp(prefix="work-join-", dir=store.project_dir(pid)))
    try:
        videos: list[Path] = []
        masters: list[Path] = []
        for i, scid in enumerate(scene_ids):
            key = f"scene:{scid}"
            with ctx.step(key):
                v = work / f"scene{i:03d}.mp4"
                m = work / f"scene{i:03d}.wav"
                _render(store, pid, scid, scene_rs, _SceneCtx(ctx, key), out=v, subtitle_file=False,
                        target_lufs=rs.target_lufs, keep_master=m)
                videos.append(v)
                masters.append(m)

        with ctx.step("join"):
            parts, durations = [], []
            sr = 48000
            for v, m in zip(videos, masters, strict=True):
                dur = video_duration(v)
                x, sr = read_wav(m)
                if x.ndim == 1:
                    x = np.stack([x, x], axis=1)
                parts.append(fit_length(x, round(dur * sr)))
                durations.append(dur)
            joined_wav = write_wav(work / "joined.wav", join_audio(parts, sr, seam_ms=SEAM_FADE_MS), sr)
            joined_mp4 = concat_videos(videos, work / "joined.mp4")

        with ctx.step("master"):
            from ..audio.master import film_target_lufs
            target = rs.target_lufs
            scenes = [store.find_scene(project, s) for s in scene_ids]
            if scenes and all(sc.mix.loudness_target == "film" for sc in scenes):
                # «как в фильме» для склейки: громкость оригиналов сцен (средняя), мастер только ловит пики
                vals = [film_target_lufs(store.scene_dir(pid, sc.id) / "original.wav") for sc in scenes]
                vals = [v for v in vals if v is not None]
                if vals:
                    target = float(np.mean(vals))
            final_wav = work / "final.wav"
            master_for_aac(joined_wav, final_wav, target_lufs=target, true_peak_db=true_peak)

        with ctx.step("video"):
            tmp = out.with_name(f".{out.stem}.join{out.suffix}")
            try:
                mux_video_audio(joined_mp4, final_wav, tmp)
                tmp.replace(out)
            finally:
                tmp.unlink(missing_ok=True)
            if rs.subtitles == SubtitleMode.FILE:
                offsets, t = [], 0.0
                fresh = store.get(pid)
                for scid, dur in zip(scene_ids, durations, strict=True):
                    offsets.append((t, subtitle_lines(fresh, store.find_scene(fresh, scid))))
                    t += dur
                _merged_srt(offsets, out.with_suffix(".srt"), rs.subtitle_text)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return out
