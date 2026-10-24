# Бенчмарк распознавания речи (Whisper / GigaAM)

Сравнение распознавателей речи на коротких кусках: WER, потерянные реплики, выдуманные слова, ошибки начала и конца реплик.
В репозитории лежит эталон для **Tears of Steel** (английский, 3 куска, 54 реплики; тексты — официальные субтитры проекта
Blender, CC-BY 3.0). Итоги и выводы — `docs/QUALITY.md`, раздел «Распознавание русской речи: GigaAM v3».

Эталон для коммерческих фильмов (например, русский дубляж, на котором мерили GigaAM) сюда не кладём: это чужие тексты.
Свой эталон можно подключить с любого пути: `--ref путь/к/reference.json` или переменная `DUBL_BENCH_REF`; гипотезы лежат
в `--hyp-dir` (`DUBL_BENCH_HYP`), аудио для проверки «выдумок по всему файлу» — в `--audio-dir` (`DUBL_BENCH_AUDIO`).

Аудио в репозиторий не кладём. Нужны 16 кГц моно wav, например `audio/tos_full.wav`: `samples/fetch.py`, затем
`ffmpeg -i tos.mov -ac 1 -ar 16000 audio/tos_full.wav`.

Формат эталона (`reference.json`):

```json
{"sets": {"имя_набора": {"lang": "en", "tags": ["tos_full"],
   "pieces": [{"id": "p1", "start": 0, "end": 60, "title": "…",
     "nonverbal": [[12.0, 13.5]],
     "lines": [{"start": 0.0, "end": 2.8, "text": "Текст реплики", "conf": "high", "bound": "manual"}]}]}}}
```

`tags` — имена входных wav без `.wav` (по ним ищутся гипотезы `hyp_<модель>_<тег>.json`); `bound` = `energy` или `manual` —
границы, не зависящие от моделей: только они идут в метрики начала и конца. Остальные поля описаны в начале `score.py`.

| Файл | Что |
|---|---|
| `reference.json` | эталон: тексты реплик, границы (только `bound=energy/manual` — независимые от моделей — участвуют в метриках границ) |
| `score.py` | метрики: WER, CER, потерянные реплики, выдуманные слова, ошибки начала/конца, выдумки по всему файлу → `scores.md`, `scores.json` |
| `run_app.py` | **функция приложения** `transcribe()` (её же зовёт `analyze_project`) → `hyp/hyp_<имя>_<вход>.json` |
| `run_analyze.py` | полный `analyze_project` на видео (язык «Определить по речи») |
| `run_onnx.py`, `bench_common.py` | GigaAM через onnx-asr «как есть» (без добора), общая нарезка/формат гипотез |
| `hyp/` | гипотезы последнего прогона (`whisper_app` — Whisper как было, `app_auto` — как стало) |

Повтор (из `engine/`, нужна папка с моделями — та же, что у программы, или пустая: модели скачаются):

```bash
export DUBL_DATA_DIR=/tmp/dubl-data
B=tests/e2e/asr_bench
# русская речь: свой ролик, язык «Определить по речи»; --track — номер аудиодорожки
.venv/bin/python $B/run_app.py путь/к/ролику_16k.wav app_auto
.venv/bin/python $B/run_app.py путь/к/ролику_16k.wav app_whisper --engine whisper --language ru
.venv/bin/python $B/run_app.py $B/audio/tos_full.wav app_auto                     # английский: сам возьмёт Whisper
.venv/bin/python $B/score.py --out /tmp/scores.md
```

Пороги добора можно перебрать: `run_app.py … --doubt-p 0.75 --ctx 5` (по умолчанию 0,8 и 8 с), `--gaps` — добирать и
громкие места без слов GigaAM.
