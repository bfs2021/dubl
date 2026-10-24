"""Pydantic-модели запросов и ответов API (данные проекта — модели из models.py)."""
from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator, model_validator

from ..llm.director import MixPatch
from ..models import MixSettings, Project, RenderSettings, SceneTextMode

Seconds = Annotated[float, Field(ge=0.0, le=1e7, allow_inf_nan=False)]
Finite = Annotated[float, Field(allow_inf_nan=False)]


# ───────────── запросы ─────────────
def file_url_to_path(url: str) -> str:
    """file:///Users/a/b%20c.mp4 → /Users/a/b c.mp4 (file://localhost/…, file:///C:/x → C:/x)."""
    from urllib.parse import unquote, urlparse
    u = urlparse(url.strip())
    path = unquote(u.path)
    if u.netloc and u.netloc.lower() != "localhost":              # file://сервер/папка — сетевой путь
        path = f"//{u.netloc}{path}"
    elif len(path) > 2 and path[0] == "/" and path[2] == ":":      # /C:/x → C:/x
        path = path[1:]
    return path


def _split_file_url(m) -> None:
    """Ссылка вида file://… — это файл на диске: переносим в path. Остальное, кроме http(s), отклоняем."""
    if m.url and m.url.strip().lower().startswith("file:"):
        m.path, m.url = file_url_to_path(m.url), None


class ProbeRequest(BaseModel):
    """Что проверить: файл на диске (path) или ссылку (url) — ровно одно из двух."""
    path: str | None = Field(None, min_length=1, max_length=4096)
    url: str | None = Field(None, min_length=1, max_length=2048)

    @model_validator(mode="after")
    def _one_of(self):
        _split_file_url(self)
        if bool(self.path) == bool(self.url):
            raise ValueError("Нужен либо path (файл на диске), либо url (ссылка на видео) — что-то одно")
        if self.url and not self.url.lower().startswith(("http://", "https://")):
            raise ValueError("Ссылка должна начинаться с http:// или https:// (или передай путь к файлу)")
        return self


class AnalyzeRequest(BaseModel):
    """Повторный разбор: что угодно можно не передавать — разбор возьмёт те же значения по умолчанию, что при создании."""
    audio_stream_index: int | None = Field(None, ge=0)
    language: str | None = Field(None, max_length=8)


class RenderProjectRequest(BaseModel):
    """Склейка сцен в один ролик. Порядок scene_ids = порядок в ролике."""
    scene_ids: list[str] = Field(min_length=1, max_length=100)
    settings: RenderSettings


class ProjectCreate(BaseModel):
    path: str | None = Field(None, max_length=4096)
    url: str | None = Field(None, max_length=2048)
    title: str | None = Field(None, max_length=300)
    audio_stream_index: int | None = Field(None, ge=0)       # какую дорожку считать оригиналом; None — выбрать само
    language: str | None = Field(None, max_length=8)         # язык оригинала; None — определить

    @model_validator(mode="after")
    def _one_of(self):
        _split_file_url(self)
        if bool(self.path) == bool(self.url):
            raise ValueError("Нужен либо path (файл на диске), либо url (ссылка на видео) — что-то одно")
        if self.url and not self.url.lower().startswith(("http://", "https://")):
            raise ValueError("Ссылка должна начинаться с http:// или https:// (или передай путь к файлу)")
        return self


class ProjectRename(BaseModel):
    title: str = Field(max_length=300)


class MergedProject(Project):
    """Ответ слияния голосов: весь проект и merge_id для отмены."""
    merge_id: str


class SpeakerRename(BaseModel):
    name: str | None = Field(None, max_length=100)


class SpeakerMerge(BaseModel):
    a: str       # остаётся
    b: str       # вливается в a


class LineSpeakerPatch(BaseModel):
    """Сменить персонажа у реплики: speaker_id существующего или new_speaker_name — завести нового."""
    speaker_id: str | None = None
    new_speaker_name: str | None = Field(None, max_length=100)


class LinesMerge(BaseModel):
    line_ids: list[str] = Field(min_length=2, max_length=6)   # соседние реплики разбора одного персонажа


class MergedLinesProject(Project):
    """Ответ склейки реплик: проект, merge_id для отмены и сколько дублей перешло к склеенной реплике."""
    merge_id: str
    moved_takes: int = 0


class SnapRequest(BaseModel):
    a: Seconds
    b: Seconds | None = None
    mode: Literal["free", "dialogue", "shot"] = "free"


class SnapResult(BaseModel):
    start: float
    end: float


class SceneCreate(BaseModel):
    start: Seconds
    end: Seconds
    title: str = Field("", max_length=200)


class LineTextPatch(BaseModel):
    line_id: str
    text: str = Field(max_length=5000)
    # откуда текст: не указано — человек (если текст отличается от оригинала и от того, что уже стоит в реплике);
    # «llm» — UI сохраняет перевод ИИ, «file» — перевод из субтитров
    source: Literal["file", "llm", "user"] | None = None


class ScenePatch(BaseModel):
    title: str | None = Field(None, max_length=200)
    text_mode: SceneTextMode | None = None
    dubbed_speakers: list[str] | None = None
    lines: list[LineTextPatch] | None = None
    mix: MixSettings | None = None


class TakePatch(BaseModel):
    selected: bool | None = None
    offset_ms: float | None = Field(None, ge=-10000.0, le=10000.0, allow_inf_nan=False)
    stretch: float | None = Field(None, ge=0.5, le=2.0, allow_inf_nan=False)


class KeyBody(BaseModel):
    key: SecretStr = Field(min_length=1, max_length=2000)


class LLMRequest(BaseModel):
    project_id: str
    scene_id: str | None = None
    target_lang: str = Field("ru", max_length=16)
    line_ids: list[str] | None = None      # translate / fit_length: какие реплики (по умолчанию все)
    n: int = Field(3, ge=1, le=10)         # fit_length: сколько вариантов
    text: str | None = Field(None, max_length=2000)   # director: просьба обычными словами
    take_id: str | None = None             # director: разбор дубля

    @field_validator("target_lang")
    @classmethod
    def _lang(cls, v: str) -> str:
        return v.strip() or "ru"


# ───────────── ответы ─────────────
class Capabilities(BaseModel):
    """Что движок умеет прямо сейчас. Лёгкий движок из установщика — без нейросетей: ml=false."""
    ml: bool                        # есть всё для разбора и разделения голосов (asr + separation + diarization)
    asr: bool                       # распознавание речи (faster-whisper или mlx-whisper)
    separation: bool                # отделение голосов от музыки (torch + demucs / audio-separator)
    diarization: bool               # кто говорит (torch + speechbrain)
    gigaam: bool = False            # распознавание русской речи GigaAM установлено (onnx-asr + onnxruntime)
    gigaam_downloaded: bool = False  # модель GigaAM уже скачана (иначе скачается при первом разборе русской речи)
    gigaam_download_mb: int = 230   # сколько качать модель GigaAM


class Health(BaseModel):
    ok: bool
    version: str
    device: str
    capabilities: Capabilities


class Ok(BaseModel):
    ok: bool = True


class ErrorResponse(BaseModel):
    detail: str                     # понятный текст по-русски — его и показываем человеку
    debug: str | None = None        # сырые подробности для отладки (хвост ffmpeg и т. п.); есть не всегда, не показывать


class CheckResult(BaseModel):
    ok: bool
    message: str
    key_status: Literal["unknown", "ok", "invalid"] = "unknown"
    key_message: str | None = None


class LoginResult(BaseModel):
    """Итог POST /settings/llm/{id}/login: терминал с входом открыт (после входа статус перепроверь GET /settings?refresh=true)."""
    ok: bool
    message: str


class KeyResult(BaseModel):
    """Итог PUT /settings/llm/{id}/key: ключ сохранён и проверен."""
    key_status: Literal["unknown", "ok", "invalid"]
    key_message: str | None = None          # по-русски; None — всё хорошо


class UrlProbe(BaseModel):
    """Что известно о ролике по ссылке до скачивания. known=False — «узнаем после скачивания» (см. message)."""
    known: bool
    title: str | None = None
    duration: float | None = None           # секунды
    has_subtitles: bool | None = None       # None — неизвестно
    subtitle_langs: list[str] = []          # языки ручных субтитров
    message: str | None = None              # пояснение для пользователя


class TranslatedLine(BaseModel):
    line_id: str
    text: str
    source: Literal["file", "llm", "user"] | None = None   # «user» — перевод человека, ИИ его не трогал: text = его текст


class TranslateResult(BaseModel):
    translations: list[TranslatedLine]     # в порядке реплик сцены
    ai_model: str | None = None            # какая модель ответила (название); None — ИИ не участвовал
    ai_fallback: bool = False              # ответила запасная, а не назначенная в настройках


class LineVariants(BaseModel):
    line_id: str
    texts: list[str]                       # варианты текста под длину реплики, лучший первым


class FitLengthResult(BaseModel):
    variants: list[LineVariants]
    ai_model: str | None = None            # какая модель ответила (название); None — ИИ не участвовал
    ai_fallback: bool = False              # ответила запасная, а не назначенная в настройках


class NameSpeakersResult(BaseModel):
    names: dict[str, str]                 # speaker_id -> имя
    ai_model: str | None = None            # какая модель ответила (название); None — ИИ не участвовал
    ai_fallback: bool = False              # ответила запасная, а не назначенная в настройках


class DirectorReport(BaseModel):
    """Замеры дубля словами и числами (llm.director.take_report)."""
    model_config = ConfigDict(extra="ignore")

    line_id: str
    take_id: str
    offset_ms: float
    timing: Literal["ok", "late", "early"]
    timing_level: Literal["ok", "noticeable", "big"]
    stretch: float
    length: Literal["ok", "too_long", "too_short"]
    length_level: Literal["ok", "noticeable", "big"]
    fit_confidence: float | None = None
    fit_unsure: bool | None = None
    source: Literal["mic", "file"] = "mic"   # откуда дубль (file — загружен файлом, к видео не привязан)
    # попадание в губы проверено (дубль с микрофона и подгонка уверена); false — «попал в губы» не утверждаем
    lips_checked: bool = True
    text_match: Literal["ok", "partial", "mismatch", "silent"] | None = None
    text_ratio: float | None = None
    missing_words: list[str] = []
    peak_db: float | None = None
    loudness_lufs: float | None = None
    loudness: Literal["ok", "clipping", "too_quiet"]
    problems: list[str] = []
    ok: bool


class DirectorTakeResult(BaseModel):
    """POST /llm/director с take_id: разбор дубля."""
    report: DirectorReport                 # замеры
    feedback: str                          # 1–2 фразы человеческим языком (без модели — по правилам)
    ai_model: str | None = None            # какая модель ответила (название); None — ИИ не участвовал
    ai_fallback: bool = False              # ответила запасная, а не назначенная в настройках


class DirectorCommandResult(BaseModel):
    """POST /llm/director с text: что сделали по просьбе. Применять ничего не нужно — mix уже готов, его можно
    отправить в PATCH сцены как есть."""
    patches: list[MixPatch]                # каждое изменение с note; пусто — просьба не поняла
    mix: MixSettings                       # настройки сведения после изменений (autopilot выключен, если что-то поменялось)
    ai_model: str | None = None            # какая модель ответила (название); None — ИИ не участвовал
    ai_fallback: bool = False              # ответила запасная, а не назначенная в настройках


class JobAccepted(BaseModel):
    """Тело ответа 202: работа запущена, следите за задачей."""
    job_id: str
    state: Literal["preparing"] = "preparing"


# ───────────── проверка микрофона (POST /mic/check, /mic/latency) ─────────────
class MicCheckResult(BaseModel):
    """Итог проверки микрофона по записи «тишина, потом фраза». Уровни — дБ от полной шкалы (0 — потолок, −60 — тишина)."""
    noise_floor_db: float | None = None     # шум комнаты в тишине; ниже −45 — тихо, выше −45 — шумно
    speech_db: float | None = None          # средний уровень речи; хорошо: −26…−14
    peak_db: float | None = None            # пик речи; хорошо: −24…−6
    snr_db: float | None = None             # на сколько речь громче шума; меньше 25 — шумно
    clipping: bool = False                  # звук срезан (перегруз)
    speech_found: bool = True               # в записи нашлась речь
    verdict: Literal["good", "quiet", "loud", "noisy", "silent"]     # loud — и «слишком близко», и перегруз
    advice: str                             # совет по-русски, готов к показу: «Подойди ближе…», «Всё хорошо…»
    saved: bool = False                     # записано в профиль «Мой микрофон»


class LatencySignal(BaseModel):
    """Какой сигнал играть для замера задержки. Сам файл — GET /mic/latency_signal.wav (тот же сигнал, детерминированный)."""
    sample_rate: int
    duration_s: float                       # длина файла
    beep_count: int
    beep_starts_s: list[float]              # когда начинается каждый бип (от начала файла)
    beep_s: float                           # длина бипа
    freq_from_hz: float                     # тон бипа: растёт от freq_from_hz до freq_to_hz (чирик)
    freq_to_hz: float
    min_record_s: float                     # микрофон должен писать не меньше стольких секунд
    wav_url: str
    how_to: str


class LatencyResult(BaseModel):
    latency_ms: float | None = None         # None — сигнал не найден (см. message)
    confidence: float = 0.0                 # 0..1; результат принят от 0,5
    ok: bool = False
    message: str                            # по-русски, готово к показу
    saved: bool = False                     # записано в mic_profile.latency_ms и AppSettings.latency_ms
