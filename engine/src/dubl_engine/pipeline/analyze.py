"""analyze_project: probe → звук → встроенные субтитры → речь (язык по речи, Whisper/GigaAM) → кто говорит → склейки."""
from __future__ import annotations

import inspect
import logging
from pathlib import Path

from ..models import Analysis, AudioStreamInfo, Line, Speaker
from ..settings import AppSettings, get_hf_token
from ..store import ProjectStore
from ..util import EngineError

log = logging.getLogger(__name__)

ANALYZE_STEPS = [
    ("probe", "Чтение файла"),
    ("audio", "Звук из видео"),
    ("subtitles", "Субтитры из файла"),
    ("transcribe", "Текст и тайминги"),
    ("diarize", "Кто говорит"),
    ("shots", "Склейки кадров"),
]

_ISO1 = {
    "eng": "en", "rus": "ru", "jpn": "ja", "deu": "de", "ger": "de", "fra": "fr", "fre": "fr", "spa": "es",
    "ita": "it", "kor": "ko", "zho": "zh", "chi": "zh", "por": "pt", "ukr": "uk", "pol": "pl", "tur": "tr",
    "nld": "nl", "dut": "nl", "swe": "sv", "ces": "cs", "cze": "cs", "hin": "hi", "ara": "ar",
}


def iso1(code: str | None) -> str | None:
    if not code:
        return None
    c = code.lower()
    return _ISO1.get(c, c if len(c) == 2 else None)


def compat_kwargs(fn, **kw):
    """Оставить только те аргументы, которые функция принимает (запас на разные версии модулей)."""
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return kw
    if any(p.kind == p.VAR_KEYWORD for p in params.values()):
        return kw
    return {k: v for k, v in kw.items() if k in params}


def original_audio(project) -> AudioStreamInfo:
    from ..media.probe import pick_original_audio
    return pick_original_audio(project.media)


def check_source(src: Path) -> None:
    if not src.is_file():
        raise EngineError(f"Исходное видео не найдено: {src}. Верни файл на место — проект хранит только ссылку на него.")


def make_thumbnail(src: Path, duration: float, out: Path) -> bool:
    """Кадр-обложка проекта (jpg ~320 px). Не получилось — не беда."""
    from ..util import run_ffmpeg
    try:
        t = min(max(duration * 0.1, 0.0), max(duration - 1.0, 0.0)) if duration > 0 else 0.0
        run_ffmpeg(["-ss", f"{t:.2f}", "-i", Path(src), "-frames:v", "1", "-an", "-sn",
                    "-vf", "scale='min(320,iw)':-2", "-q:v", "4", out], timeout=120)
        return out.is_file()
    except EngineError as e:
        log.info("Обложку не сделали: %s", e)
        return False


def find_translation_track(src: Path, media, target: str, adir: Path, *, audio_lang: str | None = None) -> list[Line]:
    """Строки субтитров на языке перевода: из дорожек файла, иначе из файлов рядом (.ru.srt и т. п.).
    Если оригинал уже на этом языке — перевод не нужен, пустой список."""
    from ..analysis.subtitles import external_subtitle_files, parse_subtitle_file
    from ..media.extract import extract_subtitles

    if audio_lang and audio_lang == target:
        return []
    best: list[Line] = []
    for s in [s for s in media.subtitle_streams if s.is_text and not s.forced and iso1(s.language) == target][:4]:
        try:
            found = parse_subtitle_file(extract_subtitles(src, s.index, adir / f"tr_{s.index}"))
        except EngineError as e:
            log.warning("Субтитры перевода #%s не прочитались: %s", s.index, e)
            continue
        if len(found) > len(best):
            best = found
    if best:
        return best
    for f in external_subtitle_files(src, target)[:4]:
        try:
            found = parse_subtitle_file(f)
        except EngineError as e:
            log.warning("Субтитры рядом (%s) не прочитались: %s", f.name, e)
            continue
        if len(found) > len(best):
            best = found
    return best


def pick_original_subtitles(tracks: list[tuple[str | None, list[Line]]], language: str | None) -> list[Line]:
    """Субтитры оригинала: дорожка на языке речи, где больше всего строк; нет такой — дорожка без языка
    (если язык речи не известен — любая, кроме русской: русская обычно перевод). Чужой язык — не наш текст."""
    if language:
        tier = [t for t in tracks if t[0] == language] or [t for t in tracks if t[0] is None]
    else:
        tier = [t for t in tracks if t[0] != "ru"]
    best = max(tier, key=lambda t: len(t[1]), default=None)
    return best[1] if best else []


def analyze_project(store: ProjectStore, project_id: str, ctx, *, settings: AppSettings,
                    audio_stream_index: int | None = None, language: str | None = None) -> Analysis:
    from ..analysis.align import align_embedded_to_speech
    from ..analysis.diarize import diarize
    from ..analysis.speech import language_note
    from ..analysis.subtitles import match_translation, parse_subtitle_file, speakers_from_names
    from ..analysis.transcribe import transcribe
    from ..media.extract import extract_audio, extract_center_channel, extract_subtitles
    from ..media.probe import pick_original_audio, probe
    from ..media.shots import detect_shot_cuts

    project = store.get(project_id)
    src = Path(project.source_path)
    check_source(src)
    adir = store.analysis_dir(project_id)
    adir.mkdir(parents=True, exist_ok=True)

    # 1. probe
    with ctx.step("probe"):
        media = probe(src)
        if audio_stream_index is not None:
            audio = next((a for a in media.audio_streams if a.index == audio_stream_index), None)
            if audio is None:
                raise EngineError("В файле нет выбранной звуковой дорожки.")
        else:
            audio = pick_original_audio(media)
        with store.edit(project_id) as p:
            p.media = media
        make_thumbnail(src, media.duration, store.thumbnail_path(project_id))
        ctx.progress("probe", 1.0)

    # 2. звук: mono 16 кГц для распознавания (+ центр, если 5.1)
    with ctx.step("audio"):
        # 16 бит: вдвое меньше диска и памяти на длинном фильме (2 ч = 230 МБ); для распознавания этого достаточно
        wav16 = extract_audio(src, audio.index, adir / "audio_16k.wav", sample_rate=16000, mono=True,
                              sample_fmt="s16")
        ctx.progress("audio", 0.6)
        if audio.channels > 2:
            # центр целого фильма в 48 кГц float — 1.4 ГБ; сцены вырезают свой центр сами, здесь хватает 16 кГц
            extract_center_channel(src, audio.index, adir / "center.wav", sample_rate=16000, sample_fmt="s16")
        ctx.progress("audio", 1.0)

    # 3. встроенные текстовые субтитры. Язык оригинала, если человек его не выбрал, узнаем только по речи (шаг 4):
    # метке дорожки верить нельзя (русский дубляж бывает помечен «eng»), поэтому читаем все текстовые дорожки,
    # а нужную выбираем после распознавания.
    chosen_lang = iso1(language)                        # выбор человека — закон
    track_lang = iso1(audio.language)
    target = iso1(settings.target_lang) or "ru"
    sub_tracks: list[tuple[str | None, list[Line]]] = []
    with ctx.step("subtitles"):
        cands = [s for s in media.subtitle_streams if s.is_text and not s.forced]
        if chosen_lang:
            # сначала субтитры на языке оригинала, потом без языка; чужие — нет
            cands = [s for s in cands if iso1(s.language) == chosen_lang] or [s for s in cands if not s.language]
        for s in cands[:6]:                   # из нескольких дорожек потом берём ту, где больше диалога (не «надписи»/SDH)
            try:
                found = parse_subtitle_file(extract_subtitles(src, s.index, adir / f"subs_{s.index}"))
            except EngineError as e:
                log.warning("Субтитры #%s не прочитались: %s", s.index, e)
                continue
            if found:
                sub_tracks.append((iso1(s.language), found))
        # субтитры на языке перевода (в файле или рядом) — пригодятся как готовый перевод реплик
        translation_track = find_translation_track(src, media, target, adir, audio_lang=chosen_lang)
        ctx.progress("subtitles", 1.0)

    # 4. речь — всегда: язык по речи, текст и тайминги (Whisper или GigaAM, см. analysis/speech.py)
    asr: dict = {}
    with ctx.step("transcribe"):
        speech, language = transcribe(
            wav16, language=chosen_lang, model=settings.whisper_model,
            on_progress=lambda v: ctx.progress("transcribe", v),
            **compat_kwargs(transcribe, device=settings.processing_device, engine=settings.asr_engine,
                            track_language=track_lang, on_note=lambda t: ctx.note("transcribe", t), info=asr))
        language = iso1(language) or chosen_lang or track_lang
        embedded = pick_original_subtitles(sub_tracks, chosen_lang or language)
        if embedded:
            embedded = align_embedded_to_speech(embedded, speech)
        lines = embedded or speech
        source = "user" if chosen_lang else (asr.get("language_source") or ("speech" if language else None))
        note = language_note(language, track_lang, source)

    # 5. кто говорит
    speakers: list[Speaker] = []
    suggestions: list[tuple[str, str, float]] = []
    named = False
    if embedded and any((ln.speaker_id or "").startswith("name:") for ln in lines):
        lines, speakers = speakers_from_names(lines)
        named = bool(speakers)
    if named:
        ctx.skip("diarize")
    else:
        with ctx.step("diarize"):
            hf = get_hf_token()
            lines, speakers, suggestions = diarize(
                wav16, lines, hf_token=hf, on_progress=lambda v: ctx.progress("diarize", v))

    joined: list[list[Line]] = []
    if not named:
        # whisper режет одну фразу на две, когда в середине пауза: тот же голос, пауза короче трети секунды — склеиваем
        from ..analysis.lines import join_close_groups, merge_group
        joined = [g for g in join_close_groups(lines) if len(g) > 1]
        lines = [merge_group(g) for g in join_close_groups(lines)]
        for sp in speakers:
            mine = [ln for ln in lines if ln.speaker_id == sp.id]
            sp.line_count = len(mine)
            sp.total_speech = round(sum(max(ln.end - ln.start, 0.0) for ln in mine), 3)

    if translation_track and iso1(language) != target:      # речь уже на языке перевода — переводить нечего
        lines = match_translation(lines, translation_track)
        joined = [match_translation(g, translation_track) for g in joined]     # части — со своим переводом

    # 6. склейки
    with ctx.step("shots"):
        cuts = detect_shot_cuts(src)
        ctx.progress("shots", 1.0)

    # авто-склейки можно «Разделить» так же, как ручные: снимок каждой и карта merged_lines (старые снимки — в корзину)
    from .lines import store_auto_merges
    merged_lines = store_auto_merges(store, project_id, joined)
    analysis = Analysis(language=language, audio_stream_index=audio.index, lines=lines, speakers=speakers,
                        shot_cuts=cuts, chapters=media.chapters, merge_suggestions=suggestions,
                        merged_lines=merged_lines, asr_engine=asr.get("asr_engine"), language_source=source,
                        track_language=track_lang, language_note=note, asr_notes=list(asr.get("notes") or []))
    with store.edit(project_id) as p:
        p.analysis = analysis
    return analysis
