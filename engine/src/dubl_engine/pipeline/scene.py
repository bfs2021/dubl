"""prepare_scene, add_take, render_scene, preview_scene."""
from __future__ import annotations

import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Literal

import numpy as np

from ..models import (
    JobResult,
    RenderAspect,
    RenderSettings,
    Scene,
    SceneLine,
    SceneTextMode,
    SubtitleMode,
    Take,
)
from ..settings import AppSettings
from ..store import ProjectStore
from ..util import ConflictError, EngineError, ffprobe_json, run_ffmpeg
from .outputs import release_output, reserve_output, windows_safe_name
from .takes import analyze_take_text, apply_default_star, delete_take_files, empty_reason

log = logging.getLogger(__name__)

PREPARE_STEPS = [
    ("cut", "Вырезка сцены"),
    ("audio", "Звук сцены"),
    ("separate", "Голоса и фон"),
    ("speakers", "Голос каждого персонажа"),
]
RENDER_STEPS = [
    ("mix", "Сведение"),
    ("master", "Мастеринг"),
    ("video", "Сборка видео"),
]
MIN_SCENE_S = 0.3
MIN_LINE_S = 0.15
MAX_SCENE_S = 1200.0          # 20 минут: разделение голоса держит сцену в памяти целиком
RENDER_SUFFIXES = {".mp4", ".mkv", ".mov"}
# «Лучшее» разделение (BS-RoFormer) держит всю сцену в памяти: ≈ 8 ГБ база + ≈ 10 МБ на каждую секунду сцены.
BEST_MAX_SCENE_S = 60.0
BEST_BASE_GB = 8.0
BEST_MB_PER_S = 10.0
NOT_READY = "Сцена ещё готовится — голоса и фон не разделены."


def best_memory_gb(duration: float) -> float:
    """Грубая оценка памяти «лучшего» разделения для сцены такой длины, ГБ."""
    return BEST_BASE_GB + max(duration, 0.0) * BEST_MB_PER_S / 1024


def effective_quality(requested: str, duration: float) -> tuple[str, str | None]:
    """(качество, пояснение). «best» для сцены длиннее BEST_MAX_SCENE_S автоматически заменяется на «fast»."""
    if requested == "best" and duration > BEST_MAX_SCENE_S:
        return "fast", (f"Сцена {duration:.0f} с — для «лучшего» разделения нужно около {best_memory_gb(duration):.0f} ГБ "
                        f"памяти (≈ {BEST_MB_PER_S:.0f} МБ на секунду плюс {BEST_BASE_GB:.0f} ГБ). Взял быстрое: "
                        "оно ненамного хуже и в 6 раз быстрее. «Лучшее» работает для сцен до "
                        f"{BEST_MAX_SCENE_S:.0f} с.")
    return requested, None


def scene_lines_from_analysis(project, scene: Scene) -> list[SceneLine]:
    """Реплики разбора, пересекающие сцену, в координатах сцены. Уже отредактированный текст сохраняем."""
    if project.analysis is None:
        return list(scene.lines)
    old = {ln.line_id: ln for ln in scene.lines}
    dur = scene.end - scene.start
    out: list[SceneLine] = []
    for ln in project.analysis.lines:
        if ln.end <= scene.start or ln.start >= scene.end:
            continue
        s = max(ln.start, scene.start) - scene.start
        e = min(ln.end, scene.end) - scene.start
        if e - s < MIN_LINE_S:          # от реплики в сцену попал огрызок — озвучивать нечего
            continue
        prev = old.get(ln.id)
        fresh = ln.translation or ln.text                      # перевод из файла (если нашёлся) или оригинал
        fresh_src = "file" if ln.translation else None
        if prev is not None and scene.text_mode == SceneTextMode.CUSTOM:
            text, src = prev.text, prev.translation_source
        elif prev is None or prev.text == prev.original or prev.translation_source == "file":
            text, src = fresh, fresh_src                        # человек и ИИ текст не трогали — берём свежий
        else:
            text, src = prev.text, prev.translation_source      # правка человека или перевод ИИ сохраняем
        out.append(SceneLine(line_id=ln.id, speaker_id=ln.speaker_id, start_rel=round(s, 3),
                             end_rel=round(min(e, dur), 3), original=ln.text, text=text, translation_source=src))
    return out


def subtitle_lines(project, scene: Scene) -> list[SceneLine]:
    """Реплики сцены для субтитров. Реплике без своего перевода подставляем перевод из субтитров файла (по разбору
    проекта) — так работает и для сцен, созданных до того, как перевод из файла стал сохраняться в сцену."""
    if project.analysis is None:
        return list(scene.lines)
    tr = {ln.id: ln.translation for ln in project.analysis.lines if ln.translation}
    out = []
    for sl in scene.lines:
        if sl.translation_source is None and sl.text.strip() in ("", sl.original.strip()) and sl.line_id in tr:
            sl = sl.model_copy(update={"text": tr[sl.line_id], "translation_source": "file"})
        out.append(sl)
    return out


def stems_for(store: ProjectStore, pid: str, scene: Scene):
    from ..audio.mix import SceneStems
    sd = store.scene_dir(pid, scene.id)
    per = {p.stem: p for p in sorted((sd / "voices").glob("*.wav"))} if (sd / "voices").is_dir() else {}
    return SceneStems(voices=sd / "voices.wav", background=sd / "background.wav", per_speaker=per,
                      takes_dir=sd / "takes", scene_dir=sd)


def prepare_scene(store: ProjectStore, pid: str, scid: str, ctx, *, settings: AppSettings | None = None) -> Scene:
    from ..audio.separate import separate, split_by_speaker
    from ..media.extract import cut_scene_video, extract_audio, extract_center_channel

    settings = settings or AppSettings()
    project = store.get(pid)
    scene = store.find_scene(project, scid)
    sd = store.scene_dir(pid, scid)
    sd.mkdir(parents=True, exist_ok=True)
    src = Path(project.source_path)
    from .analyze import check_source
    check_source(src)
    idx = project.analysis.audio_stream_index if project.analysis else original_stream(project)

    # Пока файлы пересобираются, «готово» было бы неправдой (старый clip.mp4 + новый original.wav): сначала снимаем флаг.
    # Если программу закроют посреди подготовки, сцена останется «не готова» и будет перезапущена.
    with store.edit(pid) as p:
        sc0 = p.scenes[store.scene_index(p, scid)]
        sc0.stems_ready, sc0.prepare_state = False, "running"
    try:
        with ctx.step("cut"):
            cut_scene_video(src, scene.start, scene.end, sd / "clip.mp4", audio_stream_index=idx)
        with ctx.step("audio"):
            extract_audio(src, idx, sd / "original.wav", sample_rate=48000, stereo=True, start=scene.start, end=scene.end)
            center = None
            try:
                c = extract_center_channel(src, idx, sd / "center.wav", sample_rate=48000, start=scene.start, end=scene.end)
                center = Path(c) if c else None
            except EngineError as e:
                log.warning("Центральный канал не вытащился: %s", e)
        quality, why = effective_quality(settings.separation_quality, scene.end - scene.start)
        with ctx.step("separate"):
            if why:
                log.warning("%s", why)
                getattr(ctx, "note", lambda *_: None)("separate", why)
            voices, _bg = separate(sd / "original.wav", sd, center_wav=center, quality=quality,
                                   device=settings.processing_device, on_progress=lambda v: ctx.progress("separate", v))
        lines = scene_lines_from_analysis(project, scene)
        with ctx.step("speakers"):
            shutil.rmtree(sd / "voices", ignore_errors=True)          # при повторной подготовке не оставлять чужих персонажей
            (sd / "voices").mkdir(exist_ok=True)
            split_by_speaker(voices, lines, sd / "voices")
        with store.edit(pid) as p:
            sc = p.scenes[store.scene_index(p, scid)]
            if not store.scene_files_complete(pid, sc):         # «готово» только если все файлы на месте и целые
                raise EngineError("Подготовка сцены не завершилась: часть файлов не создалась. Попробуй ещё раз.")
            sc.lines = lines
            sc.stems_ready, sc.prepare_state = True, "ready"
            result = sc.model_copy(deep=True)
        return result
    except BaseException:                       # ошибка или отмена: не оставляем «готовится» — сцену можно запустить снова
        try:
            with store.edit(pid) as p:
                sc1 = p.scenes[store.scene_index(p, scid)]
                if not sc1.stems_ready:
                    sc1.prepare_state = "interrupted"
        except EngineError:
            pass                                 # проект или сцену уже удалили
        raise


def original_stream(project) -> int:
    from ..media.probe import pick_original_audio
    return pick_original_audio(project.media).index


def _normalize_wav(data: bytes, out: Path) -> None:
    """Любой присланный звук → WAV 48 кГц mono float32."""
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out.parent) as tmp:
        raw = Path(tmp) / "in.bin"
        raw.write_bytes(data)
        res = Path(tmp) / "out.wav"
        try:
            run_ffmpeg(["-i", raw, "-vn", "-ac", "1", "-ar", "48000", "-c:a", "pcm_f32le", res])
        except EngineError:
            raise EngineError("Не удалось прочитать записанный звук. Попробуй записать ещё раз.") from None
        shutil.move(str(res), out)


def _reserve_take(store: ProjectStore, pid: str, scid: str) -> tuple[str, Path]:
    """Выдать id дубля и занять файл — под замком, чтобы две одновременные записи не получили один id."""
    with store.lock(pid):
        scene = store.find_scene(store.get(pid), scid)
        tdir = store.takes_dir(pid, scid)
        tdir.mkdir(parents=True, exist_ok=True)
        used = {t.id for t in scene.takes} | {p.stem for p in tdir.glob("t*.wav")}
        n = 1
        while f"t{n:03d}" in used:
            n += 1
        take_id = f"t{n:03d}"
        path = store.take_path(pid, scid, take_id)
        path.touch()
        return take_id, path


def add_take(store: ProjectStore, pid: str, scid: str, wav_bytes: bytes, line_id: str | None,
             start_rel: float, preroll: float, settings: AppSettings | None = None, source: str = "mic") -> Take:
    """Сохранить дубль: нормализация → замеры → подгонка под реплику → разбор по тексту → ★.

    source: «mic» — записан с микрофона под видео; «file» — загружен файлом (привязки ко времени нет: задержку записи
    не учитываем, а советов про «начал раньше/позже» быть не должно). Почти пустая запись сохраняется с empty=True, без
    подгонки и разбора, и ★ не получает."""
    from ..audio.fit import fit_take
    from ..audio.io import read_wav, to_mono
    from ..audio.master import measure

    settings = settings or AppSettings()
    src: Literal["mic", "file"] = "file" if source == "file" else "mic"
    project = store.get(pid)
    scene = store.find_scene(project, scid)
    if not scene.stems_ready:
        raise ConflictError(NOT_READY)
    line = next((ln for ln in scene.lines if ln.line_id == line_id), None) if line_id else None
    if line_id and line is None:
        raise EngineError("В сцене нет такой реплики.")
    source_lang = project.analysis.language if project.analysis else None

    take_id, path = _reserve_take(store, pid, scid)
    try:
        # тяжёлое (ffmpeg, подгонка, распознавание) — вне замка проекта: пока идёт обработка, остальные запросы не стоят
        _normalize_wav(wav_bytes, path)
        x, sr = read_wav(path)
        x = to_mono(x)
        peak = float(20 * np.log10(max(float(np.max(np.abs(x))) if len(x) else 0.0, 1e-9)))
        why_empty = empty_reason(x, sr)
        if why_empty:
            log.info("Дубль %s почти пустой: %s", take_id, why_empty)
        try:
            lufs: float | None = None if why_empty else float(measure(x, sr).lufs)
        except Exception as e:  # noqa: BLE001 — слишком короткая запись и т. п.
            log.warning("Громкость дубля не измерилась: %s", e)
            lufs = None
        take = Take(id=take_id, line_id=line_id, speaker_id=line.speaker_id if line else None,
                    file=path.relative_to(store.scene_dir(pid, scid)).as_posix(), duration=round(len(x) / sr, 3),
                    preroll=preroll, start_rel=start_rel, peak_db=round(peak, 2), loudness_lufs=lufs,
                    source=src, empty=bool(why_empty))
        try:        # шум в паузах, речь над шумом, перегруз — для предупреждений и шумодава
            from ..audio.levels import measure_take
            m = measure_take(x, sr)
            take.noise_floor_db = None if m.noise_floor_db is None else round(m.noise_floor_db, 1)
            take.snr_db = None if m.snr_db is None else round(m.snr_db, 1)
            take.clip_pct = round(m.clip_pct, 3)
        except Exception as e:  # noqa: BLE001 — без замеров дубль всё равно сохраняем
            log.warning("Замеры дубля не удались: %s", e)

        if line is not None and not why_empty:
            sd = store.scene_dir(pid, scid)
            guide = sd / "voices" / f"{line.speaker_id}.wav" if line.speaker_id else sd / "voices.wav"
            if not guide.exists():
                guide = sd / "voices.wav"
            try:
                fit = fit_take(path, guide, line_start_rel=line.start_rel, line_end_rel=line.end_rel,
                               take_start_rel=start_rel, preroll=preroll,
                               latency_ms=settings.latency_ms if src == "mic" else 0.0)
                take.offset_ms, take.stretch, take.fit_confidence = fit.offset_ms, fit.stretch, fit.confidence
            except EngineError:
                raise
            except Exception as e:  # noqa: BLE001 — без подгонки дубль всё равно сохраняем
                log.warning("Автоподгонка не удалась: %s", e)
        if not why_empty:
            found = analyze_take_text(path, line, settings, source_lang)
            if found:
                for k, v in found.items():
                    setattr(take, k, v)
        with store.edit(pid) as p:
            sc = p.scenes[store.scene_index(p, scid)]
            sc.takes.append(take)
            sc.mix_check = None              # новый дубль — прежняя проверка сведения уже не про то, что звучит
            apply_default_star(sc, line_id)
            take = next(t for t in sc.takes if t.id == take_id).model_copy(deep=True)
            if take.speaker_id and not take.empty and take.speaker_id not in sc.dubbed_speakers:
                sc.dubbed_speakers.append(take.speaker_id)   # записал голос — значит, этого персонажа озвучиваем
    except BaseException:
        delete_take_files(store, pid, scid, take_id)
        raise
    return take


def delete_take(store: ProjectStore, pid: str, scid: str, tid: str) -> None:
    from ..audio.mix import drop_stale_tracks
    with store.edit(pid) as p:
        sc = p.scenes[store.scene_index(p, scid)]
        t = store.find_take(sc, tid)
        sc.takes = [x for x in sc.takes if x.id != tid]
        if t.selected:      # вместо удалённого ★ получает самый полный из оставшихся дублей той же реплики
            t.chosen_by_user = False
            apply_default_star(sc, t.line_id)
        sc.mix = drop_stale_tracks(sc)      # у «ещё не озвучен» не остаётся старого усиления (−7,8 дБ)
        sc.mix_check = None
    delete_take_files(store, pid, scid, tid)


def _render(store: ProjectStore, pid: str, scid: str, rs: RenderSettings, ctx, *, out: Path, subtitle_file: bool,
            target_lufs: float, keep_master: Path | None = None) -> Path:
    """Сведение → мастеринг → видео одной сцены. keep_master — куда сохранить мастер-WAV (для склейки сцен)."""
    from ..audio.mix import automix, export_stems, mix_check, mix_scene
    from ..audio.render import compute_geometry, master_for_aac, render_video, write_subtitles

    project = store.get(pid)
    scene = store.find_scene(project, scid)
    if not scene.stems_ready:
        raise ConflictError(NOT_READY)
    sd = store.scene_dir(pid, scid)
    stems = stems_for(store, pid, scene)
    out.parent.mkdir(parents=True, exist_ok=True)

    # промежуточные файлы — в своей папке на задачу: превью и рендер одной сцены могут идти одновременно
    work = Path(tempfile.mkdtemp(prefix="work-", dir=sd))
    try:
        with ctx.step("mix"):
            if scene.mix.autopilot:
                new_mix = automix(scene, stems)
                with store.edit(pid) as p:
                    p.scenes[store.scene_index(p, scid)].mix = new_mix
                scene = scene.model_copy(update={"mix": new_mix})
            rep: dict = {}
            mixed = mix_scene(scene, stems, work / "mix.wav", report=rep)
            if rs.export_stems:
                export_stems(scene, stems, out.with_name(out.stem + "_stems"))
        with ctx.step("master"):
            from ..audio.master import resolve_target_lufs
            target_lufs = resolve_target_lufs(scene.mix.loudness_target, target_lufs, sd / "original.wav")
            mastered = work / "master.wav"
            loud = master_for_aac(mixed, mastered, target_lufs=target_lufs, true_peak_db=scene.mix.true_peak_db)
            if keep_master is not None:
                keep_master.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(mastered, keep_master)
            # честная проверка: не тонет ли голос и не пришлось ли сильно поднимать ролик (UI покажет одну строку)
            gain = None if rep.get("mix_lufs") is None else float(loud.lufs) - float(rep["mix_lufs"])
            check = mix_check(scene.mix, rep.get("voice_over_bg_db"), gain)
            try:
                with store.edit(pid) as p:
                    p.scenes[store.scene_index(p, scid)].mix_check = check
            except EngineError:
                pass                                 # сцену удалили, пока шла сборка
        with ctx.step("video"):
            sub_lines = subtitle_lines(project, scene)
            ass = None
            if rs.subtitles in (SubtitleMode.BURN, SubtitleMode.FILE):
                if rs.subtitles == SubtitleMode.BURN:
                    vs = next((s for s in ffprobe_json(sd / "clip.mp4", "-show_streams").get("streams", [])
                               if s.get("codec_type") == "video"), None)
                    play_res = (1920, 1080)
                    if vs:        # размер ASS = размер итоговой картинки, иначе шрифт и поля поедут в вертикальном видео
                        _crop, play_res = compute_geometry(int(vs["width"]), int(vs["height"]), rs)
                    ass = write_subtitles(sub_lines, work / "subtitles.ass", fmt="ass", play_res=play_res,
                                          text=rs.subtitle_text)
                elif subtitle_file:
                    write_subtitles(sub_lines, out.with_suffix(".srt"), fmt="srt", text=rs.subtitle_text)
            # собираем во временный файл рядом и подменяем целиком: отмена или сбой не портят уже лежащий ролик
            part = out.with_name(f".{out.stem}.{getattr(ctx, 'job_id', 'job')}.part{out.suffix}")
            try:
                render_video(sd / "clip.mp4", mastered, part, rs, subtitles_ass=ass,
                             on_progress=lambda v: ctx.progress("video", v))
                part.replace(out)
            finally:
                part.unlink(missing_ok=True)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return out


def job_result(store: ProjectStore, pid: str, path: Path | str) -> JobResult:
    """Итог задачи с файлом: абсолютный путь и адрес для GET /files (None, если файл вне папки проекта)."""
    rel = store.rel_to_project(pid, path)
    return JobResult(path=str(path), url=f"/files/{pid}/{rel}" if rel else None)


def resolve_output(store: ProjectStore, project, rs: RenderSettings) -> Path:
    """Куда писать итог. Относительное имя — в renders/ проекта. Только видеоформаты; исходник не трогаем."""
    raw = (rs.output_path or "").strip()
    if not raw:
        raise EngineError("Не указан файл для результата.")
    out = Path(raw).expanduser()
    if not out.is_absolute():
        out = store.renders_dir(project.id) / out.name
    if os.name == "nt":
        out = out.with_name(windows_safe_name(out.name))
    if out.suffix.lower() not in RENDER_SUFFIXES:
        out = out.with_name(out.name + ".mp4")
    if project.source_path and out.resolve() == Path(project.source_path).resolve():
        raise EngineError("Нельзя сохранять результат поверх исходного видео.")
    return out


def render_scene(store: ProjectStore, pid: str, scid: str, rs: RenderSettings, ctx) -> Path:
    """Ролик сцены. Файл с таким именем уже есть и overwrite=False → «<имя> (2).mp4», «(3)» …; возвращает итоговый путь."""
    out = reserve_output(resolve_output(store, store.get(pid), rs), rs.overwrite)
    try:
        return _render(store, pid, scid, rs, ctx, out=out, subtitle_file=True, target_lufs=rs.target_lufs)
    except BaseException:
        release_output(out, rs.overwrite)
        raise


def preview_scene(store: ProjectStore, pid: str, scid: str, ctx) -> Path:
    scene = store.find_scene(store.get(pid), scid)
    out = store.scene_dir(pid, scid) / "preview.mp4"
    tmp = out.with_name(f".preview-{ctx.job_id}.mp4")           # готовый preview.mp4 подменяем только целиком
    rs = RenderSettings(aspect=RenderAspect.ORIGINAL, height=720, output_path=str(tmp))
    try:
        _render(store, pid, scid, rs, ctx, out=tmp, subtitle_file=False, target_lufs=scene.mix.target_lufs)
        tmp.replace(out)
    finally:
        tmp.unlink(missing_ok=True)
    return out
