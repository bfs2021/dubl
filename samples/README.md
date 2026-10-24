# Тестовые фильмы для «Дубля»

Настоящие фильмы нужны, чтобы мерить скорость и качество на живом материале (см. `docs/QUALITY.md`) и гонять e2e-тесты
(`engine/tests/e2e`). Сами файлы в git **не** лежат (`samples/.gitignore` пропускает только этот README и `fetch.py`).

```bash
python3 samples/fetch.py            # скачать всё (≈ 2,1 ГБ) и проверить размеры и длительности
python3 samples/fetch.py --check    # только проверка
python3 samples/fetch.py tos        # одна группа: tos | sintel | ed | cosmos
```

Скрипт докачивает оборванные файлы (HTTP Range), сверяет размер с сервером и длительность через ffprobe,
распаковывает Sintel из zip. Сервер `download.blender.org` отдаёт 403 на стандартный User-Agent Python — в скрипте это учтено.

Папки: `raw/` — скачанное, `work/` — промежуточное (релизный MKV, куски звука, результаты замеров), `out/` — готовые ролики-примеры.

## Что в `raw/`

| Файл | Что это | Зачем |
|---|---|---|
| `tears_of_steel_720p.mov` | Tears of Steel, 1280×534, 24 fps, 12:14, H.264 + MP3 стерео | основной тестовый фильм |
| `TOS-en.srt`, `TOS-ru.srt` | английские и русские субтитры (76 реплик, 332 слова) | эталон текста и таймингов, русский перевод для синтетических дублей |
| `Surround-TOS_DVDSURROUND-Dolby 5.1.ac3` | звук 5.1 (AC-3, 640 кбит/с), синхронен с фильмом | проверка центрального канала |
| `TOS_MUSIC+FX_NO_DIALOGUE.aif` | музыка и шумы **без диалога**, стерео 48 кГц | эталон для оценки разделения голоса и фона |
| `Sintel.2010.720p.mkv(.zip)` | Sintel, 1280×544, 14:48, AC-3 5.1 + 10 дорожек SRT, 8 глав | настоящий MKV-«релиз» |
| `sintel-m+e-st.flac` | музыка и шумы Sintel без диалога | второй эталон |
| `elephantsdream-720-h264-st-aac.mov` | Elephants Dream, 720×405, 10:58 | третий фильм (старая картинка, другой звук) |
| `ED-ME-5.1-DVD.ac3` | музыка и шумы Elephants Dream 5.1 (по названию — M+E) | запас |
| `cosmos_laundromat_1080p.mp4`, `cosmos_en.srt` | Cosmos Laundromat: First Cycle, 12:11, AAC стерео + английские субтитры (60 реплик) | диалог двух персонажей (овца и продавец) — проверка диаризации на третьем фильме |

Разметка «кто говорит» для проверки диаризации — `engine/tests/e2e/data/{tos,sintel,cosmos}_speakers.json` (см. `docs/QUALITY.md`).

Собранный из этого `work/tears_of_steel_release.mkv` — «как релиз»: видео + 5.1 ac3 (eng, по умолчанию) + стерео aac +
субтитры SRT eng и rus + 3 главы. Собирается функцией `helpers.build_release_mkv()` (e2e) или `bench.py prep`.

## Лицензии и атрибуция

Все фильмы — проекты Blender Foundation / Blender Institute, лицензия **Creative Commons Attribution 3.0 (CC-BY 3.0)**:
можно использовать, в том числе в коммерческих целях, с указанием авторов. Готовые ролики-примеры в `out/` — производные работы
(дубляж на русском, кроп 9:16) и тоже обязаны сохранять атрибуцию.

| Фильм | Атрибуция |
|---|---|
| **Tears of Steel** (2012) | © Blender Foundation, mango.blender.org — CC-BY 3.0. Режиссёр Ian Hubert, продюсер Ton Roosendaal. В ролях: Derek de Lint, Sergio Hasselbaink, Rogier Schippers, Vanja Rukavina, Denise Rebergen. Музыка и звук: Joram Letwory. |
| **Sintel** (2010) | © Blender Foundation, durian.blender.org — CC-BY 3.0. Режиссёр Colin Levy. Музыка Jan Morgenstern. |
| **Cosmos Laundromat: First Cycle** (2015) | © Blender Foundation, gooseberry.blender.org — CC-BY 4.0 (источник файла — archive.org/details/CosmosLaundromatFirstCycle). Режиссёр Mathieu Auvray. |
| **Elephants Dream** (2006) | © Blender Foundation / Netherlands Media Art Institute, orange.blender.org — CC-BY 2.5. Режиссёр Bassam Kurdali. |

Русские субтитры `TOS-ru.srt` — часть материалов проекта на download.blender.org (работа волонтёров-переводчиков).

Текст для подписи под роликом: «Tears of Steel © Blender Foundation, mango.blender.org, CC-BY 3.0. Русская озвучка — «Дубль» (синтез речи macOS, демонстрация)».
