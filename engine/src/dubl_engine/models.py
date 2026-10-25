"""Общие модели данных «Дубль». Это контракт между модулями движка и фронтендом.

Правила:
- Время всегда в секундах (float) от начала исходного видео, если не сказано иное.
  Внутри сцены поля с суффиксом `_rel` — секунды от начала сцены.
- Громкость — в дБ (float). Громкость итога — LUFS.
- Модели неизменяемы по смыслу: модуль возвращает новые объекты, а не правит чужие.
- Не добавляй поля без обновления docs/ARCHITECTURE.md.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, model_validator


def _now() -> datetime:
    return datetime.now().astimezone()


AnalysisState = Literal["none", "running", "interrupted", "done"]


# ─────────────────────────── исходный файл ───────────────────────────
class AudioStreamInfo(BaseModel):
    index: int                      # индекс потока в контейнере (ffprobe stream.index)
    codec: str
    channels: int
    channel_layout: str | None = None   # "5.1", "stereo", ...
    sample_rate: int
    language: str | None = None     # ISO 639-2 из тегов ("eng", "jpn", "rus") или None
    title: str | None = None
    default: bool = False


class SubtitleStreamInfo(BaseModel):
    index: int
    codec: str                      # "subrip", "ass", "webvtt", "mov_text", "hdmv_pgs_subtitle", ...
    language: str | None = None
    title: str | None = None
    is_text: bool                   # False для картинок (PGS, VobSub) — их не читаем
    default: bool = False
    forced: bool = False


class Chapter(BaseModel):
    start: float
    end: float
    title: str | None = None


class MediaInfo(BaseModel):
    path: str
    duration: float
    container: str                  # format_name из ffprobe
    width: int
    height: int
    fps: float
    video_codec: str
    audio_streams: list[AudioStreamInfo] = []
    subtitle_streams: list[SubtitleStreamInfo] = []
    chapters: list[Chapter] = []
    size_bytes: int = 0


# ─────────────────────────── разбор ───────────────────────────
class Word(BaseModel):
    start: float
    end: float
    text: str
    prob: float | None = None


class Line(BaseModel):
    """Одна реплика исходного видео."""
    id: str                         # стабильный id: "l0001", "l0002", ...
    start: float
    end: float
    text: str                       # оригинальный текст
    speaker_id: str | None = None   # id из Speaker; None — не определён
    words: list[Word] = []
    source: Literal["embedded", "whisper"] = "whisper"
    overlap: bool = False           # в это время говорит ещё кто-то — чисто не разделить
    # перевод из файла: строки дорожки субтитров на языке перевода (по умолчанию «ru»), совпавшие с репликой по времени
    translation: str | None = None
    translation_source: Literal["file", "llm", "user"] | None = None


class Speaker(BaseModel):
    id: str                         # "s1", "s2", ...
    label: str                      # что видит пользователь; по умолчанию "Голос 1"
    name: str | None = None         # имя, если пользователь подписал или нашлось в субтитрах ASS
    color: str                      # hex "#4fb8a8" из фиксированной палитры
    total_speech: float = 0.0       # секунд речи
    line_count: int = 0


class Analysis(BaseModel):
    """Итог разбора всего видео. Делается один раз на проект."""
    language: str | None = None     # язык оригинала (ISO 639-1, "en")
    audio_stream_index: int         # какая дорожка взята как оригинал
    lines: list[Line] = []
    speakers: list[Speaker] = []
    shot_cuts: list[float] = []     # времена монтажных склеек
    chapters: list[Chapter] = []
    merge_suggestions: list[tuple[str, str, float]] = []  # (speaker_a, speaker_b, сходство 0..1)
    # Склеенные реплики: line_id склеенной реплики → merge_id последней склейки, в которую она вошла (ручной «Склеить»
    # или авто-склейки разрезанной фразы при разборе). По нему «Разделить» (POST …/lines/{line_id}/split) находит, что
    # отменить; снимок склейки — в <проект>/line_merges/<merge_id>.json. Нет записи — реплика не склеивалась.
    merged_lines: dict[str, str] = {}
    # Как распознавали речь (аддитивно, у старых разборов — None).
    asr_engine: Literal["whisper", "gigaam"] | None = None   # чем распознан текст (GigaAM — русский, с добором Whisper'ом)
    # откуда язык: выбрал человек / услышали в речи / по метке дорожки (речь неразборчива)
    language_source: Literal["user", "speech", "track"] | None = None
    track_language: str | None = None   # что написано в файле у дорожки (ISO 639-1), None — ничего
    # метка дорожки и язык речи разошлись — фраза для человека («В файле написано «английский», а слышно русский…»)
    language_note: str | None = None
    asr_notes: list[str] = []           # пояснения распознавания («GigaAM знает только русский — взял Whisper.»)


# ─────────────────────────── сцена ───────────────────────────
class SceneTextMode(str, Enum):
    AS_IS = "as_is"                 # озвучиваем оригинальный текст / его перевод
    CUSTOM = "custom"               # человек пишет свой текст


class MixMode(str, Enum):
    DUB = "dub"                     # голоса заменяются полностью
    VOICEOVER = "voiceover"         # оригинал тихо под новым голосом


class SceneLine(BaseModel):
    """Реплика внутри сцены — то, что озвучивает человек."""
    line_id: str                    # ссылка на Line.id
    speaker_id: str | None
    start_rel: float
    end_rel: float
    original: str
    text: str                       # текст для озвучки (как есть / перевод / свой)
    # откуда взят text, если он не оригинал: «file» — из субтитров файла, «llm» — перевёл ИИ, «user» — написал человек;
    # None — текст не менялся (или про него ничего не известно). Перевод ИИ не перетирает «user», «file» — можно.
    translation_source: Literal["file", "llm", "user"] | None = None


class Take(BaseModel):
    id: str                         # "t001"
    line_id: str | None             # None — дубль всей сцены целиком (потом режется по репликам)
    speaker_id: str | None
    file: str                       # путь относительно папки сцены, wav 48 кГц mono
    created: datetime = Field(default_factory=_now)
    duration: float
    preroll: float                  # секунд записи до момента старта видео (отрезается)
    start_rel: float                # с какого места сцены шло видео во время записи
    # заполняет автоподгонка:
    offset_ms: float = 0.0          # сдвиг: минус — раньше
    stretch: float = 1.0            # коэффициент длины (0.85..1.15)
    fit_confidence: float | None = None
    # замеры для помощника:
    peak_db: float | None = None
    loudness_lufs: float | None = None
    recognized_text: str | None = None
    selected: bool = False          # этот дубль идёт в сведение (★)
    # откуда дубль: «mic» — записан с микрофона под видео (есть привязка ко времени: offset_ms — это «начал раньше/позже»);
    # «file» — загружен файлом (привязки ко времени нет, offset_ms — просто на сколько подгонка сдвинула файл к реплике)
    source: Literal["mic", "file"] = "mic"
    # запись почти тишина (≈ −60 дБ и тише): дубль сохранён, но сам ★ не получает и в сведение не идёт
    empty: bool = False
    # разбор по тексту (whisper на языке перевода → сравнение с SceneLine.text); None — не считали или нельзя
    # (нет нейросетей, пустой дубль, текст реплики на другом языке, дубль всей сцены)
    text_match: float | None = None     # доля слов реплики, которые человек сказал, 0..1
    complete: bool | None = None        # сказана ли реплика целиком (text_match ≥ 0,85)
    tail_cut_s: float | None = None     # с какой секунды файла идёт хвост-перезапуск («давай ещё раз»); None — хвоста нет
    tail_text: str | None = None        # что сказано в хвосте — для подписи кнопки «Отрезать хвост»
    tail_trimmed: bool = False          # хвост уже отрезан (полный файл лежит рядом, POST …/untrim вернёт его)
    chosen_by_user: bool = False        # ★ поставил человек (PATCH selected=true): автовыбор его не меняет
    # звуковые замеры (audio.levels.measure_take; None — не мерили):
    noise_floor_db: float | None = None   # шум комнаты в паузах дубля, дБ ниже полной шкалы (RMS)
    snr_db: float | None = None           # речь над шумом, дБ; < 25 — шумно, < 35 — заметно
    clip_pct: float | None = None         # доля перегруженных (срезанных) отсчётов, %; > 0,05 — «записано с перегрузом»


class TrackSettings(BaseModel):
    """Настройки дорожки персонажа (или фона) в сведении."""
    gain_db: float = 0.0
    mute: bool = False
    solo: bool = False
    room: str = "auto"              # пресет комнаты (audio.rooms) или "auto"
    voice_fx: str = "none"          # пресет эффекта голоса (audio.fx)
    # заполняет автопилот: дубль записан слишком тихо, поднять до громкости фильма без шума нельзя — усиление упёрлось в потолок
    boost_capped: bool = False


class MixSettings(BaseModel):
    mode: MixMode = MixMode.DUB
    autopilot: bool = True          # всё ниже подбирается само, пока пользователь не тронул
    duck_background: bool = True
    background: TrackSettings = TrackSettings()
    original_voices_db: float = -14.0   # для VOICEOVER и неозвучиваемых персонажей — 0
    tracks: dict[str, TrackSettings] = {}   # speaker_id -> настройки
    target_lufs: float = -14.0
    true_peak_db: float = -1.0
    keep_breaths: bool = True       # подмешивать дыхание/шорохи из оригинала в паузах
    # громкость ролика: «online» — для YouTube/VK (target_lufs, −14), «film» — как в фильме (громкость оригинала сцены,
    # мастер только ловит пики лимитером). Тимур ещё не выбрал: пока online.
    loudness_target: Literal["online", "film"] = "online"
    denoise: bool = True            # шумодав дублей по снимку тишины (включается сам, если шум заметен)


class MixCheck(BaseModel):
    """Проверка сведения после последней сборки сцены (превью или рендер): не тонет ли голос и не пришлось ли сильно
    поднимать ролик. Считает движок (pipeline.scene._render); любая правка сведения или дублей её сбрасывает."""
    voice_over_bg_db: float | None = None   # насколько новые голоса громче фона в моменты речи, дБ; None — голосов нет
    master_gain_db: float | None = None     # на сколько мастеринг поднял сведение до цели, дБ (минус — сделал тише)
    # что посоветовать: «autopilot» — автопилот выключен, включи его; «record_louder» — дубли записаны тихо, перезапиши;
    # «buried» — голос тонет в фоне, убавь фон; None — всё в порядке
    warning: Literal["autopilot", "record_louder", "buried"] | None = None


class Scene(BaseModel):
    id: str                         # "sc001"
    title: str
    start: float                    # абсолютное время в исходнике
    end: float
    text_mode: SceneTextMode = SceneTextMode.AS_IS
    dubbed_speakers: list[str] = [] # кого озвучиваем; остальные остаются оригиналом
    lines: list[SceneLine] = []
    takes: list[Take] = []
    mix: MixSettings = MixSettings()
    mix_check: MixCheck | None = None   # итог проверки последней сборки; None — не собирали после правок
    stems_ready: bool = False       # голоса и фон разделены И все файлы сцены на месте
    # Честный статус подготовки: «ready» ⇔ stems_ready; «running» — идёт сейчас; «interrupted» — подготовка не дошла до
    # конца (закрыли программу, отмена, ошибка) и не идёт: сцену можно подготовить заново (POST …/scenes/{scid}/prepare).
    prepare_state: Literal["running", "interrupted", "ready"] = "interrupted"
    created: datetime = Field(default_factory=_now)

    @model_validator(mode="after")
    def sync_state(self):
        """prepare_state не врёт: ready ⇔ stems_ready."""
        if self.stems_ready:
            self.prepare_state = "ready"
        elif self.prepare_state == "ready":
            self.prepare_state = "interrupted"
        return self


# ─────────────────────────── проект ───────────────────────────
class Project(BaseModel):
    id: str
    title: str
    source_path: str                # исходник не копируем
    media: MediaInfo
    analysis: Analysis | None = None
    # Честный статус разбора: none — не запускали; running — идёт сейчас; interrupted — не дошёл до конца (закрыли программу,
    # отмена, ошибка) и не идёт: запусти заново через POST /projects/{pid}/analyze; done — analysis готов целиком.
    # analysis записывается одним атомарным сохранением в самом конце разбора, так что частичного разбора на диске нет.
    analysis_state: AnalysisState = "none"
    scenes: list[Scene] = []
    created: datetime = Field(default_factory=_now)
    updated: datetime = Field(default_factory=_now)

    @model_validator(mode="after")
    def sync_states(self):
        """Статусы не врут: analysis_state=done ⇔ есть analysis; ready у сцены ⇔ stems_ready."""
        if self.analysis is not None:
            self.analysis_state = "done"
        elif self.analysis_state == "done":
            self.analysis_state = "interrupted"
        for sc in self.scenes:
            sc.sync_state()
        return self


# ─────────────────────────── задачи ───────────────────────────
class JobState(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobStep(BaseModel):
    key: str                        # "probe", "transcribe", "diarize", ...
    title: str                      # по-русски, для интерфейса
    progress: float = 0.0           # 0..1
    state: JobState = JobState.QUEUED
    note: str | None = None         # пояснение для пользователя (например, «взял быстрое разделение: сцена длинная»)


class JobResult(BaseModel):
    """Итог задачи, если она что-то записала на диск (render, render_project, preview, proxy)."""
    path: str | None = None         # абсолютный путь файла
    url: str | None = None          # относительный адрес для GET (/files/<проект>/...); None — файл вне папки проекта


class Job(BaseModel):
    id: str
    kind: str                       # "analyze", "separate", "fit_take", "render", ...
    project_id: str
    scene_id: str | None = None
    state: JobState = JobState.QUEUED
    steps: list[JobStep] = []
    error: str | None = None        # человеческий текст ошибки (по-русски)
    error_code: str | None = None   # машинно-читаемая причина: "ml_missing" — не установлены нейросети (UI предлагает их скачать)
    error_debug: str | None = None  # сырые подробности для отладки (хвост stderr ffmpeg и т. п.); пользователю не показывать
    result: JobResult | None = None
    progress: float = 0.0           # общий прогресс задачи 0..1: по шагам с весами, только растёт
    eta_s: float | None = None      # сколько секунд осталось (по скорости с начала задачи); None — ещё рано оценивать
    created: datetime = Field(default_factory=_now)


# ─────────────────────────── рендер ───────────────────────────
class RenderAspect(str, Enum):
    ORIGINAL = "original"
    H16_9 = "16:9"
    V9_16 = "9:16"
    SQ1_1 = "1:1"


class SubtitleMode(str, Enum):
    NONE = "none"
    BURN = "burn"
    FILE = "file"


class SubtitleText(str, Enum):
    DUB = "dub"                     # текст дубля: перевод → русский текст из субтитров файла → оригинал (реплика не пропадает)
    ORIGINAL = "original"           # текст оригинала (как в исходном видео)


class RenderFit(str, Enum):
    BLUR = "blur"                   # вертикальный/квадрат: кадр целиком по ширине, сверху и снизу размытая копия кадра
    CROP = "crop"                   # кадр обрезается по центру до нужных пропорций


class RenderSettings(BaseModel):
    aspect: RenderAspect = RenderAspect.ORIGINAL
    # Что значит height. original / 16:9: высота кадра, None — как в исходнике. 9:16: качество по короткой стороне —
    # 720 → 720×1280, 1080 → 1080×1920, None → 1080×1920 (значения ≥ 1280 — это уже высота кадра: 1920 → 1080×1920).
    # 1:1: сторона квадрата, None → 1080×1080. Размеры всегда чётные.
    height: int | None = 1080
    target_lufs: float = -14.0
    subtitles: SubtitleMode = SubtitleMode.NONE
    subtitle_text: SubtitleText = SubtitleText.DUB     # какой текст в субтитрах: дубля (по умолчанию) или оригинала
    fit: RenderFit = RenderFit.BLUR   # как вписать кадр в 9:16 и 1:1: размытые поля (по умолчанию) или кроп; 16:9/original — не влияет
    export_stems: bool = False
    output_path: str
    # Файл с таким именем уже есть: False — не трогаем его, итог получает имя «<имя> (2).mp4», «(3)» и т. д.
    # True — заменить. Итоговый путь всегда в Job.result.path.
    overwrite: bool = False


# ─────────────────────────── LLM ───────────────────────────
class ProviderKind(str, Enum):
    CLAUDE_CLI = "claude_cli"       # локальный `claude -p`, подписка пользователя
    ANTHROPIC = "anthropic"         # API-ключ
    DEEPSEEK = "deepseek"
    GEMINI = "gemini"
    OPENAI_COMPAT = "openai_compat"
    CODEX_CLI = "codex_cli"         # локальный `codex exec`, подписка ChatGPT


class ProviderConfig(BaseModel):
    id: str                         # "deepseek", "my-openrouter", ...
    kind: ProviderKind
    title: str
    base_url: str | None = None
    model: str
    # ключ НЕ храним здесь — он в системной связке ключей (keyring), service="dubl", username=id
    enabled: bool = True
    # Только для ответов GET/PUT /settings: движок заполняет сам, из тела PUT /settings эти поля игнорируются и в
    # settings.json не пишутся. Меняются через PUT /settings/llm/{id}/key, POST …/check и реальные запросы ИИ.
    key_set: bool = False           # ключ лежит в связке ключей (у claude_cli ключа нет — там всегда True)
    key_status: Literal["unknown", "ok", "invalid"] = "unknown"   # unknown — ещё не проверяли или не было сети
    key_message: str | None = None  # по-русски, что не так с ключом; None — нечего сказать
    # Честный статус (тоже только в ответах GET/PUT /settings; в settings.json не пишется): может ли модель ответить
    # прямо сейчас. Для CLI-моделей (claude_cli, codex_cli) — установлен ли CLI и выполнен ли вход.
    # None — движок ещё не считал (старый ответ / демо): UI тогда судит по key_*.
    can_answer: bool | None = None
    state: Literal["ok", "off", "not_installed", "not_logged_in", "no_key", "invalid_key", "error"] | None = None
    state_message: str | None = None   # по-русски: «Установлен, вход выполнен, отвечает» / «Не найден — установи…»
    login_supported: bool | None = None  # у модели есть кнопка «Войти» (POST /settings/llm/{id}/login)
    install_url: str | None = None      # где установить CLI (только у CLI-моделей)

    @model_validator(mode="after")
    def _migrate_model_name(self):
        """Устаревшее имя модели DeepSeek (deepseek-chat) → актуальное deepseek-flash."""
        if self.kind == ProviderKind.DEEPSEEK and self.model.strip() == "deepseek-chat":
            self.model = "deepseek-flash"
        return self


class TaskRoute(BaseModel):
    """Кто на самом деле ответит на задачу (только в ответах GET/PUT /settings, в settings.json не пишется)."""
    assigned: str | None = None         # id модели из routing; None — не назначена или «не использовать»
    answers: str | None = None          # id модели, которая ответит с учётом запасной; None — отвечать некому
    answers_title: str | None = None
    fallback: bool = False              # отвечает не назначенная, а запасная
    fallback_id: str | None = None      # запасная на случай сбоя назначенной (если есть)
    last_used: str | None = None        # название модели, которая ответила в последний раз (за эту сессию движка)
    last_used_fallback: bool = False    # и это была запасная
    message: str | None = None          # по-русски: почему так / что делать


class LLMTask(str, Enum):
    TRANSLATE = "translate"         # перевести реплики
    FIT_LENGTH = "fit_length"       # уложить реплику под длину
    NAME_SPEAKERS = "name_speakers" # угадать имена по диалогам
    DIRECTOR = "director"           # помощник-режиссёр: советы и команды


class LLMSettings(BaseModel):
    providers: list[ProviderConfig] = []
    routing: dict[LLMTask, str | None] = {}   # задача -> provider id; None — не использовать; задачи нет в словаре — движок выберет рабочую сам
    # Только в ответах GET/PUT /settings (считает движок, в файл не пишется): кто реально ответит на каждую задачу
    tasks: dict[LLMTask, TaskRoute] | None = None
