# Дубль

Программа для любительского дубляжа. Устройство: `docs/ARCHITECTURE.md`, справка и границы продукта: `docs/ABOUT.md`. Как вносить правки — `CONTRIBUTING.md`.

## Правила для ИИ-ассистентов и людей
- Контракт данных — `engine/src/dubl_engine/models.py`. Не меняй его без явной причины; если не хватает поля — опиши это в описании правки.
- Работай в границах задачи: файлы вне неё только читай. Если без чужих файлов не обойтись, скажи об этом.
- Тяжёлые библиотеки (torch, faster_whisper, demucs, pyannote, audio_separator) импортируй внутри функций.
- ffmpeg — только через `dubl_engine.util.run_ffmpeg` / `ffprobe_json`.
- Python-зависимости ставь в `engine/.venv`: `uv pip install --python engine/.venv/bin/python <pkg>`; новые — добавь в `engine/pyproject.toml` (секция своего модуля, комментарием).
- Тесты: `cd engine && .venv/bin/pytest -q tests/<модуль>` — должны проходить. Медленные — `@pytest.mark.slow`.
- Не делай git commit, пока тебя об этом не попросили.
- Тексты для пользователя — по-русски, живо, без канцелярита.
- Справка `docs/ABOUT.md` показывается внутри программы. Пиши в ней только про то, что уже работает.

## Контракт API ↔ UI
- Типы фронтенда генерируются из OpenAPI движка: `cd app && pnpm gen:api` → `docs/openapi.json` и `app/src/api/schema.gen.ts`.
- `schema.gen.ts` руками не править. В коде UI бери типы так: `import type { components } from "./schema.gen"; type Scene = components["schemas"]["Scene"]` (удобные алиасы — в `src/api/types.ts`, который реэкспортирует из schema.gen).
- Поменял models.py или роуты — запусти `pnpm gen:api`. Проверка «типы не устарели»: `pnpm check:api`.
- У всех роутов FastAPI должен быть `response_model` или тип возврата — иначе в схеме будет `unknown`.

## Tauri
- См. `docs/TAURI.md`: минимальные capabilities, CSP, sidecar.
