# «Дубль» — архитектура v0.1

Десктопная программа для любительского дубляжа: грузишь фильм/серию целиком, нарезаешь моменты,
озвучиваешь персонажей своим голосом, автопилот подгоняет, сводит, мастерит и рендерит.

Границы продукта (что делаем и чего не будет): `ABOUT.md`.

## Состав

```
dubl/
  engine/            Python 3.11, FastAPI — вся обработка (локально или на другом компьютере)
    src/dubl_engine/
      models.py      КОНТРАКТ: все pydantic-модели. Менять только через этот документ.
      util.py        пути данных, run_ffmpeg / ffprobe_json, EngineError
      media/         чтение файла: probe, извлечение звука/субтитров, склейки, вырезка сцены
      analysis/      разбор: субтитры→реплики, whisper, кто говорит, привязка выделения
      audio/         разделение голос/фон, подгонка дубля, комнаты, эффекты, сведение, мастеринг, рендер
      llm/           провайдеры моделей, маршрутизация задач, промпты, помощник-режиссёр
      store/         хранение проектов на диске (JSON + файлы сцен)
      jobs/          очередь фоновых задач с прогрессом
      pipeline/      сценарии: analyze_project, prepare_scene, fit_take, render_scene
      api/           FastAPI-роуты, WebSocket прогресса
      main.py        запуск uvicorn
    tests/           pytest, синтетические медиа из conftest.py (генерирует ffmpeg)
  app/               Tauri 2 + React 18 + TypeScript + Vite — окно программы
  docs/
```

Платформы: macOS, Windows, Linux. Движок — отдельный процесс (в сборке — sidecar Tauri).
UI ходит к движку по HTTP `http://127.0.0.1:<port>` + WebSocket `/ws/jobs`. На удалённом компьютере — тот же API
за токеном (`Authorization: Bearer`), туннель/SSH на стороне пользователя.

## Общие правила кода

- Python 3.11, типы везде, pydantic v2. Строки для пользователя — по-русски. Код/имена — по-английски.
- ffmpeg вызываем только через `util.run_ffmpeg` / `util.ffprobe_json`.
- Ошибки, которые увидит пользователь, — `util.EngineError("понятный текст")`.
- Тяжёлые зависимости (torch, faster_whisper, demucs, pyannote) импортируются ВНУТРИ функций, не на уровне модуля:
  движок должен запускаться и проходить быстрые тесты без них.
- Аудио внутри движка: WAV, 48000 Гц, float32. Голоса/дубли — mono; фон и итог — stereo.
- Время — секунды float. Внутри сцены `_rel` = от начала сцены.
- Тесты: быстрые по умолчанию (`pytest`), с нейросетями — `@pytest.mark.slow`.
  Используй фикстуры `simple_mp4`, `rich_mkv` из `tests/conftest.py`; `tmp_path` для выходных файлов.
- Запуск тестов: `cd engine && .venv/bin/pytest -q`.
- Каждый модуль — чистые функции с явными входами/выходами. Состояние только в store.

## Хранилище на диске

```
<data_dir>/projects/<project_id>/
  project.json                 Project (models.Project)
  analysis/audio_16k.wav       дорожка оригинала, mono 16 кГц для распознавания
  analysis/center.wav          центральный канал (если был 5.1)
  scenes/<scene_id>/
    clip.mp4                   видео сцены (+ исходный звук), начинается с 0, длина = длине сцены
    original.wav               звук сцены, stereo 48 кГц
    voices.wav                 все голоса
    background.wav             музыка и шумы
    voices/<speaker_id>.wav    голос конкретного персонажа (тишина там, где он молчит)
    takes/<take_id>.wav        дубли
    preview.mp4                последняя сборка
  renders/<name>.mp4
<data_dir>/models/             скачанные нейросети
<data_dir>/settings.json       LLMSettings (без ключей), устройство, микрофон
```

## Интерфейсы модулей

Сигнатуры — обязательные. Внутренние помощники — на усмотрение автора модуля.

### media/

```python
# media/probe.py
def probe(path: str | Path) -> MediaInfo
    # ffprobe -show_format -show_streams -show_chapters. is_text = codec in {subrip, ass, ssa, webvtt, mov_text, text}.
def pick_original_audio(info: MediaInfo, prefer_lang: str | None = None) -> AudioStreamInfo
    # prefer_lang ("eng"); иначе default; иначе первый не-"rus"; иначе первый. Не бывает дорожек — EngineError.

# media/extract.py
def extract_audio(src, stream_index: int, out_wav, *, sample_rate=48000, mono=False,
                  start: float | None = None, end: float | None = None) -> Path
def extract_center_channel(src, stream_index: int, out_wav, *, sample_rate=48000,
                           start=None, end=None) -> Path | None
    # если layout содержит FC (5.1, 7.1, …) — канал FC в mono; иначе None
def extract_subtitles(src, stream_index: int, out_path) -> Path      # текстовые → .ass или .srt как есть
def cut_scene_video(src, start: float, end: float, out_mp4, *, audio_stream_index: int) -> Path
    # точная вырезка (перекодирование h264 crf 18, aac), без смены разрешения

# media/shots.py
def detect_shot_cuts(src, *, threshold: float = 0.3, start=None, end=None) -> list[float]
    # ffmpeg select='gt(scene,thr)' + showinfo; абсолютные секунды, отсортировано
```

### analysis/

```python
# analysis/subtitles.py
def parse_subtitle_file(path) -> list[Line]
    # SRT и ASS. Из ASS: поле Name → временный speaker name (кладём в Line.speaker_id как "name:<Name>"),
    # убрать теги {\...}, \N → пробел. source="embedded". id "l0001"…, сортировка по start.
def speakers_from_names(lines: list[Line]) -> tuple[list[Line], list[Speaker]]
    # "name:<X>" → Speaker(id="s1", label=X, name=X); палитра PALETTE по порядку появления

# analysis/transcribe.py
def transcribe(audio_16k_wav, *, language: str | None = None, model: str = "large-v3-turbo",
               on_progress: Callable[[float], None] | None = None, engine="whisper", track_language=None,
               on_note=None, info: dict | None = None) -> tuple[list[Line], str]
    # language=None → язык по речи (Whisper detect_language на 2–3 окнах по 30 с с речью; метка дорожки
    # track_language — только подсказка при низкой уверенности). engine: auto | whisper | gigaam → analysis/speech.py:
    # русский — GigaAM v3 (analysis/gigaam.py, onnx-asr, куски ≤ 20 с по тихим местам) + добор сомнительных реплик
    # Whisper'ом; остальное — Whisper (mlx на Mac arm64, иначе faster-whisper). info ← asr_engine, language_source, notes.

# analysis/align.py
def align_embedded_to_speech(lines: list[Line], speech: list[Line]) -> list[Line]
    # встроенные субтитры могут расходиться с речью: сдвинуть start/end каждой строки к ближайшим
    # словам whisper (если найдено пересечение по времени и похожему тексту), иначе оставить как есть

# analysis/diarize.py
def diarize(audio_16k_wav, lines: list[Line], *, hf_token: str | None,
            on_progress=None) -> tuple[list[Line], list[Speaker], list[tuple[str, str, float]]]
    # pyannote/speaker-diarization-3.1 если есть токен; иначе фолбэк: эмбеддинги + кластеризация.
    # Для каждой Line: speaker_id по максимальному перекрытию; overlap=True если двое одновременно.
    # merge_suggestions: пары спикеров со сходством > 0.8

# analysis/snap.py   — привязка выделения на шкале
def snap_selection(a: float, b: float, lines: list[Line], cuts: list[float], *,
                   tolerance: float = 0.6) -> tuple[float, float]
    # края, попавшие внутрь реплики, расширяются до её границ (+0.15 с запаса);
    # если рядом (≤ tolerance) есть склейка — край прилипает к ней; результат не режет слова
def dialogue_around(t: float, lines: list[Line], *, max_gap: float = 2.5) -> tuple[float, float]
    # клик по реплике → весь разговор: соседние реплики с паузой ≤ max_gap
def shot_around(t: float, cuts: list[float], duration: float) -> tuple[float, float]
    # от склейки до склейки

# analysis/palette.py
PALETTE: list[str]   # 10 цветов, хорошо различимых на тёмном фоне
```

### audio/

```python
# audio/io.py
def read_wav(path) -> tuple[np.ndarray, int]          # float32, shape (n,) или (n, ch)
def write_wav(path, data, sr) -> Path
def to_mono(x) -> np.ndarray

# audio/separate.py
def separate(scene_audio_wav, out_dir, *, center_wav: Path | None = None, quality: Literal["fast","best"]="best",
             device: str = "auto", on_progress=None) -> tuple[Path, Path]     # (voices.wav, background.wav)
    # center_wav (5.1) НЕ используется для разделения: нейросеть всегда по полному стерео-даунмиксу (диалог бывает и в L/R).
    # Модели: best = BS-RoFormer через audio-separator; fast = demucs htdemucs (--two-stems vocals).
    # device auto: cuda → mps (с fallback на cpu при ошибке) → cpu.
def split_by_speaker(voices_wav, lines_rel: list[SceneLine], out_dir, *, fade: float = 0.03) -> dict[str, Path]
    # голос каждого speaker_id: копия voices.wav с тишиной вне его реплик (+ короткие фейды)

# audio/fit.py       — автоподгонка дубля
def detect_speech(x, sr, *, min_gap=0.25) -> list[tuple[float, float]]    # интервалы речи по энергии
def fit_take(take_wav, guide_voice_wav, *, line_start_rel: float, line_end_rel: float,
             take_start_rel: float, preroll: float, latency_ms: float) -> FitResult
    # FitResult(offset_ms, stretch, confidence): совместить начало/конец речи дубля с речью оригинала
    # в окне реплики; кросс-корреляция огибающих для уточнения; stretch ∈ [0.85, 1.15], 1.0 если влезает.
def render_fitted(take_wav, fit: FitResult, out_wav) -> Path    # применить сдвиг и растяжение (atempo / rubberband)

# audio/rooms.py
ROOMS: dict[str, RoomPreset]     # "dry","room","kitchen","hall","street","cave","car","bathroom","phone","behind_door"
def apply_room(x, sr, preset: str) -> np.ndarray     # синтетическая IR (затухающий шум + ранние отражения) + EQ
def guess_room(background_wav) -> str    # по хвостам реверберации фона; простая эвристика, по умолчанию "room"

# audio/fx.py
VOICE_FX: dict[str, str]          # пресет → цепочка ffmpeg-фильтров: none, robot, chipmunk, giant, villain, radio, phone, ghost
def apply_fx(in_wav, preset: str, out_wav) -> Path

# audio/clean.py
def clean_voice(in_wav, out_wav, *, denoise=True, deess=True, compress=True, highpass_hz=80) -> Path

# audio/mix.py
def automix(scene: Scene, stems: SceneStems) -> MixSettings
    # автопилот: громкость дорожек персонажей к одному уровню речи (−20 LUFS кратко), фон −3..−8 дБ под речью,
    # комнату угадать по фону. Возвращает новые MixSettings (если autopilot=False — не трогать ручные поля).
def mix_scene(scene: Scene, stems: SceneStems, out_wav) -> Path
    # фон (+ducking sidechain) + неозвучиваемые оригинальные голоса + дубли (clean → fx → room → gain → позиция)
    # VOICEOVER: оригинальные голоса всех под original_voices_db
# SceneStems: dataclass с путями voices, background, per_speaker, takes_dir

# audio/master.py
def measure(x, sr) -> Loudness          # integrated LUFS, true peak dBTP, LRA (pyloudnorm)
def master(in_wav, out_wav, *, target_lufs=-14.0, true_peak_db=-1.0) -> Loudness
    # двухпроходный loudnorm ffmpeg (measure → linear=true) с запасом по пику, затем true-peak лимитер
    # (4× оверсэмплинг, look-ahead) — LUFS ±0.5 LU, пик не выше цели
def limit_true_peak(x, sr, ceiling_db) -> np.ndarray

# audio/render.py
def render_video(clip_mp4, mixed_wav, out_mp4, settings: RenderSettings, *, subtitles_ass: Path | None = None,
                 on_progress=None) -> Path
    # кроп/паддинг под aspect (9:16 — кроп по центру), scale до height, h264 + aac 192k, faststart, вшивка субтитров
def master_for_aac(in_wav, out_wav, *, target_lufs, true_peak_db) -> Loudness
    # master + проверка после AAC 192k: пик выше цели + 0.2 дБ → пересвод с меньшей целью (до 3 попыток)
    # render_video не копирует главы и теги исходника (-map_chapters -1 -map_metadata -1); так же cut_scene_video
def write_subtitles(lines: list[SceneLine], out_path, *, fmt: Literal["srt","ass"], play_res=..., text: 'dub'|'original' = 'dub') -> Path
    # dub — SceneLine.text (реплики без перевода/совпадающие с оригиналом пропускаются); original — SceneLine.original
```

### llm/

```python
# llm/providers.py
class LLMProvider(Protocol):
    id: str
    async def complete(self, system: str, user: str, *, json_schema: dict | None = None,
                       max_tokens: int = 1024, temperature: float = 0.4) -> str
    async def check(self) -> tuple[bool, str]       # (ok, сообщение) — тестовый запрос
def make_provider(cfg: ProviderConfig) -> LLMProvider
    # OPENAI_COMPAT / DEEPSEEK (base_url https://api.deepseek.com) — /chat/completions через httpx
    # ANTHROPIC — /v1/messages; GEMINI — generativelanguage v1beta generateContent
    # CLAUDE_CLI — subprocess `claude -p --output-format json --model <m>` (stdin = промпт), без ключа
# llm/secrets.py
def set_key(provider_id, key) / get_key(provider_id) -> str | None / delete_key(provider_id)   # keyring service="dubl"
# llm/router.py
def provider_for(task: LLMTask, settings: LLMSettings) -> LLMProvider | None
# llm/tasks.py
async def translate_lines(lines: list[SceneLine], target_lang: str, provider) -> list[str]
async def fit_line(line: SceneLine, provider, *, n: int = 3) -> list[str]
    # варианты текста, укладывающиеся в длину реплики; оценка длины — estimate_duration()
def estimate_duration(text: str, lang: str = "ru") -> float      # без LLM: слоги × темп, паузы на знаках
async def name_speakers(lines: list[Line], speakers: list[Speaker], provider) -> dict[str, str]
# llm/director.py  — помощник-режиссёр
def take_report(take: Take, line: SceneLine) -> dict              # замеры → факты (опоздание, длина, совпадение текста)
async def director_feedback(report: dict, provider) -> str        # 1–2 фразы человеческим языком
async def director_command(text: str, scene: Scene, provider) -> list[MixPatch]
    # «кот громче», «как в пещере» → список изменений MixSettings (строгая JSON-схема, валидация, без выдумок)
```

### api/ (FastAPI)

```
GET    /health                                   {"ok": true, "version", "device"}
GET    /projects                                 [ProjectSummary{…, has_analysis, analysis_state: none|running|interrupted|done}]
POST   /projects            {path | url}         → Project (сразу), Job analyze в фоне
GET    /projects/{pid}                           Project
DELETE /projects/{pid}                           останавливает задачи проекта, стирает папку проекта и видео, скачанное программой по ссылке
                                                 (<data>/downloads/<pid>/…); файл пользователя на его диске не трогается никогда
PATCH  /projects/{pid}  {title}                  переименовать проект (пусто → 400)
PATCH  /projects/{pid}/speakers/{sid}  {name}    переименовать; одно имя двум голосам ничего не сливает
POST   /projects/{pid}/speakers/merge  {a, b}    любые два голоса; b вливается в a → Project + merge_id (снимок в merges/<id>.json)
POST   /projects/{pid}/speakers/merges/{merge_id}/undo  → Project как до слияния; 404 — нет/уже отменено, 409 — есть более позднее слияние
POST   /projects/{pid}/snap   {a, b, mode: free|dialogue|shot}  → {start, end}
POST   /projects/{pid}/scenes {start, end, title}  → Scene, Job prepare_scene (вырезка + разделение)
GET    /projects/{pid}/scenes/{scid}             Scene{stems_ready, prepare_state: running|interrupted|ready}
POST   /projects/{pid}/scenes/{scid}/prepare    → Job prepare_scene: подготовить заново (после прерванной; 409, если уже идёт)
PATCH  /projects/{pid}/scenes/{scid}            {text_mode, dubbed_speakers, lines[]{line_id, text, source?}, mix}
                                                 source: file|llm|user → SceneLine.translation_source (не указан: «user», если текст не оригинал)
POST   /projects/{pid}/scenes/{scid}/takes      multipart wav + {line_id, start_rel, preroll, source: mic|file} → Take (+ автоподгонка,
                                                 разбор по тексту text_match/complete/tail_cut_s, ★ самому полному дублю);
                                                 запись почти тишина → 200, Take.empty=true, без ★ и без подгонки
PATCH  /projects/{pid}/scenes/{scid}/takes/{tid} {selected, offset_ms, stretch}; selected=true — выбор человека (chosen_by_user), пустой → 409
POST   /projects/{pid}/scenes/{scid}/takes/{tid}/trim_tail | untrim_tail   отрезать хвост «давай ещё раз» (Take.tail_cut_s) / вернуть
GET    /mic/profile | DELETE /mic/profile        профиль «Мой микрофон» (AppSettings.mic_profile)
POST   /mic/check      multipart wav (+silence_s, save) → MicCheckResult{noise_floor_db, speech_db, peak_db, clipping, verdict, advice}
GET    /mic/latency_signal | /mic/latency_signal.wav   описание и сам сигнал замера задержки (3 чирика, 48 кГц, 2 с)
POST   /mic/latency    multipart wav (+play_offset_s, save) → LatencyResult{latency_ms, confidence, ok, message}
POST   /projects/{pid}/scenes/{scid}/preview    → Job → preview.mp4
POST   /projects/{pid}/scenes/{scid}/render     RenderSettings → Job (kind='render'); result: {path, url}
POST   /projects/{pid}/render   {scene_ids[≥1, порядок = порядок в ролике], settings: RenderSettings} → Job (kind='render_project')
                                                 каждая сцена сводится и мастерится, видео склеиваются встык, звук — переход 80 мс на стыках,
                                                 мастеринг всего ролика; шаги: scene:<id>…, join, master, video; result: {path, url}
POST   /probe   {path | url}                     path → MediaInfo; url → UrlProbe {known, title, duration, has_subtitles, subtitle_langs, message};
                                                 file://… = path; ссылка не ведёт на видео → 422; нет сети → 200 known=false
                                                 (yt-dlp, только метаданные; не вышло — known=false, «узнаем после скачивания»)
GET    /files/{pid}/source                       исходное видео целиком (Range)
GET    /files/{pid}/proxy                        лёгкая копия 540p h264+aac faststart; нет копии → 202 + X-Job-Id (kind='proxy'),
                                                 тело {job_id, state}; готово → 200 + файл (Range); ?retry=true после ошибки
GET    /files/{pid}/...                          отдача медиа (Range-запросы для видео)
GET    /settings  PUT /settings                  LLMSettings и прочее (без ключей); у каждого провайдера key_set, key_status, key_message
PUT    /settings/llm/{provider_id}/key  {key}    ключ → keyring и сразу проверка (≤ 10 с) → {key_status, key_message}
DELETE /settings/llm/{provider_id}/key           удалить ключ, статус → unknown
POST   /settings/llm/{provider_id}/check         → {ok, message, key_status, key_message}
POST   /llm/translate        → {translations: [{line_id, text, source}]}  (перевод пользователя не перетирается; ИИ пишет в сцену с source=llm)
POST   /llm/fit_length       → {variants: [{line_id, texts[]}]}
POST   /llm/name_speakers    → {names: {speaker_id: имя}}
POST   /llm/director         (take_id)  → {report: DirectorReport, feedback: str}
POST   /llm/director_command (text)     → {patches: [MixPatch{target, field, value, note}], mix: MixSettings}
GET    /jobs/{id}   POST /jobs/{id}/cancel
WS     /ws/jobs                                  поток Job при каждом изменении
```

Запись микрофона делает UI (Web Audio / MediaRecorder в окне Tauri → WAV 48 кГц) и отправляет файл в движок.
Так движок может работать на другом компьютере.

### app/ (Tauri 2 + React + TS)

- Vite + React 18 + TypeScript strict, без UI-фреймворков «из коробки»; дизайн-токены из макета (тёмная тема,
  шрифты Onest + JetBrains Mono, акцент amber `#f0b44c`, запись `#ff4f3f`).
- `src/api/` — типизированный клиент (типы — зеркало models.py), `src/state/` — zustand, `src/screens/` — экраны
  как в макете: Projects, Analyze, Record, Mix, Render, Settings. `src/audio/recorder.ts` — запись в WAV.
- Tauri: запуск движка как sidecar, выбор файлов через диалог, права на микрофон.

## Пайплайны (pipeline/)

Умолчание разделения голоса — `separation_quality="fast"` (demucs). `best` (BS-RoFormer) для сцены длиннее 60 с
(оценка памяти ≈ 8 ГБ + 10 МБ на секунду) автоматически заменяется на `fast`; причина пишется в `JobStep.note` шага `separate`.
`pipeline/join.py` — `render_project` (склейка сцен), `pipeline/proxy.py` — `build_proxy` (копия для плеера).

1. **analyze_project**: probe → выбрать дорожку → audio_16k (+center) → встроенные текстовые субтитры
   (если есть: parse + speakers_from_names) → whisper (всегда, для слов/таймингов; если субтитры есть —
   align_embedded_to_speech) → diarize (если спикеров не дали ASS-имена) → shot cuts → Analysis.
   Перевод из файла: если есть дорожка субтитров на языке перевода (`AppSettings.target_lang`, по умолчанию `ru`) —
   в файле или рядом (`кино.ru.srt`, `кино.rus.ass`, `кино.srt` с кириллицей) — её строки по пересечению по времени
   ложатся в `Line.translation` (`translation_source="file"`); при создании сцены это `SceneLine.text`.
2. **prepare_scene**: (stems_ready=false) cut_scene_video → original.wav (+center) → separate → split_by_speaker → проверка файлов → stems_ready.
3. **add_take**: сохранить → measure → fit_take → (опц.) whisper дубля для recognized_text → Take.
4. **render_scene**: automix (если autopilot) → mix_scene → master → render_video.

## Честные статусы разбора и подготовки сцены
- `Project.analysis` записывается ОДНИМ атомарным сохранением (tmp + `os.replace`) в самом конце `analyze_project`; частичного
  разбора на диске нет. `analysis_state` (в `Project` и `ProjectSummary`): `done` ⇔ `analysis` есть (`has_analysis=true`);
  `running` — в очереди есть задача `analyze` этого проекта; `interrupted` — результата нет и задачи нет (закрыли программу,
  отмена, ошибка); `none` — только в файле проекта до первого старта движка. Поверх файла статус считает `api/live.py`
  по очереди задач, так что ответ API никогда не врёт.
- При старте движка (`main.run` → `create_app(recover=True)` → `ProjectStore.recover_interrupted`) проекты без `analysis`
  получают `interrupted` в файле и в `summary.json`; `stems_ready=true` без целых файлов сцены (`clip.mp4`, `original.wav`,
  `voices.wav`, `background.wav`, равные длины, клип открывается) сбрасывается; недоудалённые папки `.trash-*` стираются.
  Перезапуск разбора — `POST /projects/{pid}/analyze`, подготовки сцены — `POST …/scenes/{scid}/prepare`.
- `prepare_scene` в начале снимает `stems_ready` (`prepare_state=running`) и ставит `ready` только после проверки всех файлов.
- **Длина и синхрон ролика**: `cut_scene_video` приводит видео к постоянной частоте кадров (`fps=…:start_time=0`) и режет оба
  потока по `-t`, так что `clip.mp4` начинается с 0 и равен длине сцены; `render_video` и `mux_video_audio` не используют
  `-shortest`, а задают длину по концу картинки (`-t`), звук добивается тишиной (`apad`). Клипы старых сцен (видео с 0,04 с)
  выравниваются при рендере. Допуск проверок: звук и видео совпадают по длине ±25 мс, вспышка и щелчок — ±25 мс.

## Ошибки и прогресс задач
- Все ошибки API: `{"detail": "текст по-русски", "debug"?: "сырой текст ffmpeg/yt-dlp"}`. `debug` — только для отладки/лога.
  Сырой stderr ffmpeg/ffprobe превращает `util.ffmpeg_failure` (битый/не-видео файл → «Файл повреждён или это не видео…»).
  `EngineError(msg, debug=…)`; `UnprocessableError` → 422. Тот же текст в `Job.error`, сырое — в `Job.error_debug`.
- `Job.progress` (0..1) — общий, взвешенный по шагам (`jobs.manager.STEP_WEIGHTS`), монотонный. `Job.eta_s` — секунд осталось
  по скорости с начала задачи; `null`, пока прошло < 3 с или прогресс < 2%, и после завершения.
- `RenderSettings.subtitles` по умолчанию `none`; `RenderSettings.subtitle_text` = `dub` | `original` (по умолчанию `dub`).
- **Субтитры дубля** (`RenderSettings.subtitle_text`): `dub` (по умолчанию) — для каждой реплики текст перевода; нет перевода —
  русский текст из субтитров файла (по разбору проекта); нет и его — оригинал (реплика не пропадает). `original` — как есть.
- **Перевод**: `SceneLine.translation_source` = `file` | `llm` | `user` | null. ИИ-перевод и повторный разбор сцены не
  перетирают `user`; `file` перезаписывается и ИИ, и свежим файлом.
- **Ключи ИИ**: `ProviderConfig.key_set/key_status/key_message` — только в ответах `/settings`, в settings.json не пишутся;
  статус хранится в `<data_dir>/key_status.json` (`llm/keystatus.py`). Меняется при сохранении ключа, «Проверить» и реальных
  запросах (ответили → ok; 401/403/«API key not valid» → invalid; нет сети → без изменения). Ошибки провайдера — русский
  `detail` («Ключ ИИ не подходит — проверь, что скопировал его целиком.», «Лимит запросов исчерпан, попробуй позже.»,
  «Сервис ИИ сейчас недоступен.»), сырой ответ без ключа — в `debug`.
- **Пустая запись**: `pipeline.scene.check_take_not_empty` — речи нет и пик < −40 дБFS, либо RMS по найденной речи < −50 дБFS → 422.
  Совпадение подгонки 0 (`fit_confidence == 0`: речь дубля в окне реплики не найдена) — проблема `no_match`, а не «в губы».
- **Формат кадра** (`RenderSettings.aspect`, `height`, `fit`): 9:16 — `height` = короткая сторона (720 → 720×1280, 1080 и `null` →
  1080×1920; ≥ 1280 — высота кадра, 1920 → 1080×1920); 1:1 — сторона (`null` → 1080×1080); original/16:9 — высота кадра как раньше.
  `fit=blur` (по умолчанию): кадр целиком по ширине, сверху и снизу его же размытая увеличенная копия (ffmpeg split →
  scale/crop+boxblur → overlay); `fit=crop` — центр. Для original/16:9 `fit` не влияет. Размеры всегда чётные.
