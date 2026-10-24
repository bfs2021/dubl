# Сборка и упаковка «Дубль»

Коротко: установщик содержит оболочку (Tauri), **лёгкий движок** (Python без нейросетей) и ffmpeg. Нейросети (torch, whisper, demucs…) и их веса докачиваются при первом запуске кнопкой «Скачать нейросети».

## Схема

```
Установщик (dmg / msi / nsis / AppImage / deb)
├── Дубль            оболочка Tauri (Rust + WebView)
├── dubl-engine      sidecar-лаунчер, 320 КБ (Rust): находит engine/ и запускает его
└── resources/
    ├── engine/      движок: PyInstaller --onedir, без torch (≈120–150 МБ на диске)
    └── ffmpeg/      ffmpeg + ffprobe, статические (≈130 МБ на Mac)

<папка данных>/runtime/        появляется после «Скачать нейросети» (≈1,1 ГБ на Mac)
├── bin/uv                     uv (скачивается, sha256 проверяется)
├── python/                    python-build-standalone 3.11 от uv
└── ml/venv/                   torch + faster-whisper + demucs + … + колесо dubl-engine
```

### Почему так

| Вопрос | Решение | Причина |
|---|---|---|
| PyInstaller или python-build-standalone + venv | **Оба**: лёгкий движок — PyInstaller `--onedir`, ML — отдельный venv на python-build-standalone | PyInstaller с torch раздувает сборку, ломается на CUDA/MPS-библиотеках и не обновляется частями. Лёгкий движок (fastapi, numpy, scipy) собирается стабильно. venv на uv для torch — штатный способ, а CUDA/CPU выбирается на машине пользователя |
| Всё в установщике или докачка | **Докачка** | torch+зависимости ≈ 1 ГБ (Mac), ≈ 2–5 ГБ с CUDA. Установщик остаётся небольшим (dmg ≈ 116 МБ), пользователь без GPU не качает CUDA, версию torch можно менять без новой оболочки |
| `--onefile` или `--onedir` | `--onedir` | `--onefile` каждый запуск распаковывает себя во временную папку (заметная задержка на каждом старте) и чаще вызывает срабатывания антивирусов. Не замерялось — выбор по документации PyInstaller |
| Tauri ждёт один файл, а движок — папка | **Rust-лаунчер** `packaging/launcher` | `externalBin` принимает только один файл `dubl-engine-<triple>`. Лаунчер лежит рядом с оболочкой, ищет `engine/dubl-engine` в ресурсах и запускает. Unix: `exec` (тот же PID — `child.kill()` из Tauri убивает именно движок). Windows: дочерний процесс в Job Object с `KILL_ON_JOB_CLOSE` — вместе с оболочкой умирает и вся ветка процессов |
| Нейросетевой venv | Движок-лаунчер (`engine_entry.py`) сам определяет `runtime/ml` и подменяет себя на `ml/venv/python -m dubl_engine.main` | Для Tauri ничего не меняется: тот же sidecar, те же аргументы, та же строка `DUBL_ENGINE_READY port=…` |

Защита от «осиротевшего» движка: движок следит за PID родителя и завершается, если оболочку убили (`kill -9`). Отключить — `DUBL_PARENT_WATCH=0` (для запуска из терминала через `nohup`).

## Требования для сборки

- Python 3.11 и [uv](https://docs.astral.sh/uv/) (`UV=/путь/к/uv`, если не в PATH)
- Rust (`rustup`), Node 22 + pnpm 10
- Linux: `libwebkit2gtk-4.1-dev libappindicator3-dev librsvg2-dev patchelf libssl-dev libxdo-dev`
- Windows: Visual Studio Build Tools (MSVC) и WebView2 (в Windows 11 есть)

## Локальная сборка

Одна и та же последовательность на всех ОС (собирается под ту платформу, на которой запущена: кросс-сборки нет, у PyInstaller её нет вообще):

```bash
cd dubl
python packaging/prepare_bundle.py          # движок + лаунчер + ffmpeg → app/src-tauri/{binaries,resources}
cd app && pnpm install
pnpm tauri build \
  --config ../packaging/tauri-config/common.json \
  --config ../packaging/tauri-config/macos.json      # или windows.json / linux.json
```

На своём Маке после сборки — `bash packaging/install_mac.sh`: переносит `Дубль.app` из папки сборки в `/Applications` (закрыв запущенную копию). В папке сборки .app не остаётся, иначе Spotlight и Launchpad показывают два «Дубля». Установщик `.dmg` остаётся в `bundle/dmg`.

Что делает `prepare_bundle.py`: `build_engine.py` (колесо движка → PyInstaller → Rust-лаунчер `dubl-engine-<triple>`) + `fetch_ffmpeg.py`. Промежуточные файлы — в `packaging/build/` (в git не попадают). Сменили код движка — запустите заново.

`packaging/tauri-config/*.json` — добавки к `tauri.conf.json`: `common.json` подключает ресурсы `engine/` и `ffmpeg/`, остальные задают форматы установщика:

| ОС | Форматы | Файл |
|---|---|---|
| macOS | `.dmg` (+ `.app`) | `macos.json` |
| Windows | NSIS `.exe` (по умолчанию для пользователя, без админа) и `.msi` | `windows.json` |
| Linux | `.AppImage`, `.deb` | `linux.json` (имя продукта `Dubl` — кириллица в имени `.deb` ломает установку) |

Проверка собранного sidecar (запуск, `READY`, `/health`, ffmpeg из ресурсов, смерть движка после убитого родителя):

```bash
python packaging/smoke_sidecar.py                       # из packaging/build/stage
python packaging/smoke_sidecar.py --stage app/src-tauri # из того, что ляжет в установщик
```

### Если нужен только движок или только ffmpeg

```bash
python packaging/build_engine.py [--out DIR] [--skip-launcher] [--clean]
python packaging/fetch_ffmpeg.py [--target aarch64-apple-darwin|x86_64-pc-windows-msvc|…] [--out DIR]
```

## ffmpeg и лицензии

`fetch_ffmpeg.py` скачивает, сверяет контрольную сумму, распаковывает `ffmpeg`+`ffprobe`, запускает их и проверяет версию ≥ 6 и кодеки `libx264`, `aac`, `pcm_f32le` (нужны рендеру). Рядом кладёт `VERSION.json` и `NOTICE-ffmpeg.txt`.

| Платформа | Источник | Проверка |
|---|---|---|
| macOS arm64 / x64 | ffmpeg.martin-riedl.de (release) | sha256 с сайта |
| Windows x64 | gyan.dev, release-essentials | sha256 с сайта |
| Windows arm64 | BtbN/FFmpeg-Builds (GitHub) | sha256 из `checksums.sha256` |
| Linux x64 / arm64 | johnvansickle.com (release static) | md5 (целостность, не подлинность) |

Все эти сборки — **GPL** (есть libx264). Мы запускаем ffmpeg отдельным процессом, а не линкуем, так что лицензия приложения не заражается; но при поставке бинарников нужно положить текст лицензии и ссылку на исходники — это делает `NOTICE-ffmpeg.txt` (тот же текст и ссылки на исходники есть в THIRD-PARTY-NOTICES.md, который показывает окно «О программе»). Если понадобится LGPL-вариант (без x264) — BtbN выпускает `lgpl`-сборки, но тогда рендеру нужен другой видеокодек (`h264_videotoolbox` на Mac, `h264_mf` на Windows и т. п.), и это правка движка.

Версии не зафиксированы: берётся последний release. Для воспроизводимых релизов закрепите URL и sha256 в `fetch_ffmpeg.py`.

## Размеры (замерено на Mac arm64)

| Что | Размер |
|---|---|
| Лаунчер `dubl-engine-aarch64-apple-darwin` | 316 КБ |
| `resources/engine/` (PyInstaller, без ML) | 116 МБ на диске |
| `resources/ffmpeg/` (ffmpeg + ffprobe) | 132 МБ |
| `Дубль.app` | 273 МБ |
| `Дубль_0.1.0_aarch64.dmg` | 116 МБ |
| Нейросетевой пакет, `runtime/` целиком | ≈ 1,0 ГБ (`ml` 956 МБ + python 76 МБ + uv 40 МБ); кэш uv удаляется после успеха |
| Установка пакета | 63 с на домашнем интернете |
| Веса: whisper large-v3-turbo | ≈ 1,6 ГБ, скачиваются при первом разборе |
| Веса: demucs / BS-RoFormer / эмбеддинги голосов | десятки–сотни МБ, при первом использовании |

Старт движка: **0,15–0,4 с** до `DUBL_ENGINE_READY` (самый первый запуск свежескопированной папки — 2–4 с, дальше быстро). С ML-пакетом — 0,5–3 с (импорт torch лениво, при первом задании).

Windows и Linux не собирались и не замерялись (нет машин; в `fetch_ffmpeg.py` проверено только получение ссылок и контрольных сумм для всех шести платформ). Оценка: ffmpeg 100–200 МБ, движок 100–150 МБ; с CUDA нейросетевой пакет 3–5 ГБ.

## Что скачивается при первом запуске

Кнопка «Скачать нейросети» запускает `dubl-engine install-ml-pack --data-dir <папка данных> [--device auto|cpu|cuda]`:

1. uv (если его нет в PATH) — sha256 вшит в `ml_pack.py`;
2. Python 3.11 (python-build-standalone, через uv);
3. `torch==2.5.1` + `torchaudio`: Mac — обычные колёса (MPS работает, проверено: `/health` → `"device":"mps"`); Windows с NVIDIA — индекс `cu124`; Windows без видеокарты — CPU-колёса PyPI; Linux — CPU-колёса с индекса PyTorch (колёса PyPI тянут ≈ 2,5 ГБ CUDA), Linux с NVIDIA — `cu124`;
4. колесо `dubl_engine[ml]` (faster-whisper, demucs, speechbrain, audio-separator…), вшитое в сборку.

Вывод: строки `DUBL_ML_STAGE <этап> <i>/<n>`, в конце `DUBL_ML_DONE` (код 0) или `DUBL_ML_FAILED <причина>` (код 1). Статус: `dubl-engine ml-pack-status` (JSON). После установки движок нужно **перезапустить** — лаунчер подхватит `runtime/ml`. Если версия движка не совпадает с версией пакета (обновили программу), движок стартует в лёгком режиме, пока пакет не переустановят.

### Как это устроено в программе

- **Rust (`app/src-tauri/src/ml_pack.rs`)**, команды окна: `ml_pack_status` (JSON из `dubl-engine ml-pack-status --data-dir …`), `install_ml_pack` (sidecar с `install-ml-pack --data-dir … --device auto`, возвращается сразу), `cancel_ml_pack` (убивает установщик и его `uv`), `ml_pack_state` (идёт ли установка — на случай перезагрузки окна). Аргументы sidecar собирает Rust, из JS их задать нельзя, поэтому в `capabilities/main.json` для этого ничего не расширяется: только разрешения на сами команды (`allow-install-ml-pack` и т. д., они описаны в `build.rs`).
- **События**: окно получает `ml-pack` с нагрузкой `{kind: "stage"|"log"|"done"|"failed"|"cancelled", …}`. `stage` — из строк `DUBL_ML_STAGE <имя> i/n`; `log` — последняя осмысленная строка вывода uv («Downloading torch (60.9MiB)»); `failed` несёт сырую причину из `DUBL_ML_FAILED`.
- **Перезапуск движка**: после `DUBL_ML_DONE` и кода 0 Rust убивает старый (лёгкий) движок, запускает sidecar заново, ждёт `DUBL_ENGINE_READY` и публикует новый адрес тем же событием `engine-ready`; окно переподключается само. У каждого запуска движка свой номер, поэтому «смерть» старого движка не сбрасывает адрес нового.
- **Папка данных**: у движка и установщика одна — `DUBL_DATA_DIR` (если задана, для отладки и проверок), иначе папка данных приложения (`~/Library/Application Support/app.dubl.desktop` на Mac).
- **Движок сообщает, есть ли нейросети**: `GET /health` → `capabilities: {ml, asr, separation, diarization}` (`importlib.util.find_spec`, без импорта torch). Если разбор или разделение всё же упали из-за отсутствия библиотек, задача получает `error_code = "ml_missing"`. Повторный разбор — `POST /projects/{pid}/analyze` (необязательное тело `{audio_stream_index, language}`; прежний разбор заменяется, сцены остаются).
- **Интерфейс**: плашка «Чтобы распознавать речь и отделять голоса от музыки, нужно один раз скачать нейросети (~1,3 ГБ, несколько минут) · Скачать» — на главном экране и в Настройки → Обработка; во время установки — этап («Скачиваю установщик / Python / torch / библиотеки»), деталь, «Отмена»; потом «Готово, перезапускаю движок…» и плашка исчезает. «Начать разбор» без нейросетей сначала скачивает их и продолжает сам. Упавший разбор показывает «Попробовать снова» (а при `ml_missing` — «Скачать нейросети», затем «Разобрать снова»). Ошибки установки — по-человечески (нет интернета / мало места / прочее), сырой текст в «Подробностях». В браузере и при обработке «на своём сервере» кнопки нет — вместо неё подсказка с командой для терминала.

Проверка без `.app`: `DUBL_DATA_DIR=/tmp/пустая-папка` и отдельный `CARGO_TARGET_DIR` (чтобы не трогать собранное приложение и данные пользователя).

Диагностика: `dubl-engine self-check` — какой ffmpeg найден и откуда.

## Подпись

| ОС | Что нужно | Без подписи |
|---|---|---|
| macOS | Apple Developer Program, **$99/год**: сертификат «Developer ID Application» + нотаризация (Apple ID, пароль приложения, Team ID) | Gatekeeper: «приложение повреждено / от неустановленного разработчика», запуск только через «Открыть» в Системных настройках. Для arm64 ad-hoc подпись ставится автоматически |
| Windows | Сертификат подписи кода (OV ≈ $200–400/год; EV дороже, сразу снимает SmartScreen) или Azure Trusted Signing | SmartScreen: «Windows защитил ваш компьютер» → «Подробнее» → «Выполнить в любом случае». Репутация набирается со временем |
| Linux | Не нужна | — |

**macOS и вложенные бинарники.** Tauri подписывает `.app` и sidecar, но **не заходит в `resources/`**, а там сотни `.dylib/.so` и `ffmpeg`. Нотаризация отклонит приложение с неподписанным кодом внутри, поэтому перед `tauri build` запускается `packaging/sign_macos.py` (hardened runtime, `packaging/engine.entitlements`: unsigned executable memory + disable library validation — нужны Python и numpy). Проверено ad-hoc подписью: 127 файлов, движок после подписи стартует, `/health` отвечает. Настоящую Developer ID подпись и нотаризацию на этой машине проверить нельзя (нет сертификата).

Секреты GitHub для релиза: `APPLE_CERTIFICATE` (base64 .p12), `APPLE_CERTIFICATE_PASSWORD`, `APPLE_SIGNING_IDENTITY`, `APPLE_ID`, `APPLE_PASSWORD` (пароль приложения), `APPLE_TEAM_ID`. Без них `release.yml` соберёт неподписанные установщики.

Venv с ML-пакетом создаётся уже на машине пользователя и подписи не требует.

## CI

- `.github/workflows/ci.yml` (push в `main`, pull request) — на ubuntu / macos / windows: быстрые pytest движка (ffmpeg ставится тем же `fetch_ffmpeg.py`); фронт `pnpm install --frozen-lockfile`, `check:api` (кроме Windows), `build`, `test`; `cargo check` оболочки и сборка лаунчера; сборка sidecar + `smoke_sidecar.py`.
- `.github/workflows/release.yml` (тег `v*` или вручную) — `tauri-action` на macos-latest (arm64), windows-latest, ubuntu-22.04; собирает движок и ffmpeg, подписывает macOS-ресурсы (если есть секреты), кладёт установщики в черновик релиза.

Intel-Mac не собираем: для macOS x86_64 нет колёс `torch` 2.5.1, нейросети не встанут. Windows arm64 и Linux arm64 в CI не включены (`fetch_ffmpeg.py` их умеет).

> Статус: YAML проверен `actionlint 1.7.12` без замечаний. В облаке воркфлоу ещё не запускались — первый прогон наверняка выявит мелочи на Windows/Linux.

## Лицензии моделей и библиотек

Сводная таблица лицензий (модели, Python, npm, Rust) и отдельный раздел про ffmpeg — в [THIRD-PARTY-NOTICES.md](../THIRD-PARTY-NOTICES.md) в корне репозитория. Программа показывает тот же файл в «Настройки → О программе». Обновляйте его перед каждым выпуском (команды сборки списков — в самом файле).

## Известные ограничения

- Кросс-сборок нет: каждый установщик собирается на своей ОС (отсюда матрица в CI).
- `engine/pyproject.toml` закрепляет `torch==2.5.1` и `python>=3.11,<3.12` — не меняйте в одном месте: версия дублируется в `packaging/ml_pack.py` (`TORCH_VERSION`).
- Папка движка для macOS содержит симлинки (`libgfortran.5.dylib` → `numpy/.dylibs/…`); `build_engine.py` копирует их как симлинки, не разворачивайте через `cp -L`.
