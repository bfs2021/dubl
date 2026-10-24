# Сторонние компоненты

«Дубль» распространяется под GNU GPL v3 или новее (файл [LICENSE](LICENSE)). Здесь перечислено, что ещё входит в программу или скачивается ею, и на каких условиях. Этот же файл показывается в окне «О программе».

Лицензии собраны автоматически из метаданных пакетов на момент выпуска 0.1.0. Если вы заметили ошибку, откройте обращение: <https://github.com/pingvin1337/dubl/issues>.

## Что идёт в установщике, а что скачивается

- **В установщике:** оболочка (Tauri + интерфейс на React), лёгкий движок (Python, без нейросетей) и ffmpeg с ffprobe.
- **Скачивается на компьютер пользователя при первом запуске:** библиотеки нейросетей (torch и другие) и веса моделей. Мы их не распространяем и не вшиваем: программа качает их с серверов авторов (PyPI, Hugging Face и т. д.), поэтому условия авторов принимаются при скачивании.

## ffmpeg

В установщик кладутся готовые сборки ffmpeg и ffprobe, собранные сторонними авторами. Все они с флагом GPL (в них есть libx264). Лицензия: GNU GPL v2 или новее, часть кода под LGPL. «Дубль» запускает их отдельными процессами и с ними не линкуется.

Текст лицензии лежит в папке `ffmpeg/` рядом с программой в файле `NOTICE-ffmpeg.txt` (его создаёт `packaging/fetch_ffmpeg.py`). Исходный код проекта FFmpeg: <https://ffmpeg.org/download.html>. Исходники конкретных сборок:

| Платформа | Сборка | Исходники и скрипты сборки |
|---|---|---|
| macOS arm64 | ffmpeg.martin-riedl.de | https://ffmpeg.martin-riedl.de |
| Windows x64 | gyan.dev, release-essentials | https://www.gyan.dev/ffmpeg/builds/ |
| Windows arm64 | BtbN/FFmpeg-Builds | https://github.com/BtbN/FFmpeg-Builds |
| Linux x64 / arm64 | johnvansickle.com, release static | https://johnvansickle.com/ffmpeg/ |

Точные адреса для каждой платформы записаны в `packaging/fetch_ffmpeg.py`. Если хотите другую сборку, положите свой `ffmpeg` и `ffprobe` в PATH или в папку, на которую указывает переменная `DUBL_FFMPEG_DIR`.

## Модели и крупные библиотеки

Сведения о лицензиях весов взяты из `docs/BUILD.md` и страниц проектов на момент написания, карточки на Hugging Face заново не проверялись.

| Компонент | Лицензия | Примечание |
|---|---|---|
| Whisper (веса OpenAI, `large-v3-turbo`), faster-whisper | MIT | Скачиваются с Hugging Face при первом разборе |
| GigaAM v3 e2e-rnnt (Сбер, `ai-sage/GigaAM-v3`) и ONNX-экспорт `istupakov/gigaam-v3-onnx` (int8, ~216 МБ) | MIT (веса и экспорт) | Русская речь. Качается в папку данных при первом разборе русской речи |
| onnx-asr, onnxruntime | MIT | Запуск GigaAM; ставятся вместе с нейросетями |
| Demucs (код и веса `htdemucs`) | MIT | Репозиторий Meta заархивирован, лицензия не менялась |
| audio-separator | MIT | Запуск BS-RoFormer |
| BS-RoFormer `model_bs_roformer_ep_317_sdr_12.9755.ckpt` (viperx) | у файла отдельной лицензии нет; проект UVR заявляет MIT | Обучен на MUSDB18-HQ, данные которого выданы для некоммерческих исследований. Для бесплатной программы риск низкий; при коммерческом использовании нужна юридическая оценка. Веса мы не вшиваем: пользователь скачивает их сам |
| pyannote `speaker-diarization-3.1` (необязательно) | MIT, но модель «gated» на Hugging Face | Нужно принять условия на странице модели и ввести токен HF. Мы веса не распространяем. Без токена работает запасной вариант на эмбеддингах speechbrain |
| speechbrain (код) | Apache-2.0 | Определение, кто говорит. Лицензию весов `speechbrain/spkrec-ecapa-voxceleb` на Hugging Face отдельно не проверяли |
| torch, torchaudio | BSD-3-Clause | Скачиваются при установке нейросетей |
| mlx-whisper (код) и веса `mlx-community/whisper-*` | код MIT; веса — конвертация весов Whisper (MIT), их карточки отдельно не проверяли | Только Mac с Apple Silicon |
| yt-dlp | Unlicense | Скачивание видео по ссылке |
| Tauri, плагины Tauri | MIT или Apache-2.0 | Оболочка |
| React, React DOM, React Router, Zustand | MIT | Интерфейс |
| lucide-react | ISC | Значки |
| Шрифты Onest и JetBrains Mono (через Fontsource) | SIL OFL 1.1 | Шрифты вшиты в интерфейс |

## Совместимость с GPL-3.0

Проверка по метаданным пакетов показала два места, которые стоит держать в уме.

- **diffq** (CC BY-NC 4.0, «некоммерческая» лицензия). Его требует audio-separator на macOS и Linux; на Windows вместо него ставится `diffq-fixed`. Дубль diffq не вызывает напрямую и в установщик не кладёт: пакет скачивается пользователем вместе с нейросетями. Но лицензия не позволяет коммерческое использование, поэтому продавать сборку с этим пакетом нельзя.
- **sphn** (зависимость demucs): в метаданных PyPI лицензия не указана. Проект: <https://github.com/kyutai-labs/sphn>.

Остальное совместимо с GPL-3.0: MIT, BSD, Apache-2.0 (с GPL-3.0, но не с GPL-2.0-only), ISC, Unlicense, PSF, MPL-2.0 (файлы под MPL остаются под MPL), LGPL (lameenc, soxr: подключаются как библиотеки). Веса моделей с лицензиями MIT и Apache-2.0 не накладывают на программу ограничений. Данные для обучения BS-RoFormer (MUSDB18-HQ) см. выше.

## Python: зависимости движка и нейросетей

Список снят с окружения, в которое устанавливается `dubl-engine[ml]` (движок и нейросетевой пакет вместе).

| Пакет | Версия | Лицензия | Страница |
|---|---|---|---|
| absl-py | 2.5.1 | Apache-2.0 | https://github.com/abseil/abseil-py |
| annotated-doc | 0.0.5 | MIT | https://github.com/fastapi/annotated-doc |
| annotated-types | 0.8.0 | MIT | https://github.com/annotated-types/annotated-types |
| anyio | 4.15.1 | MIT | https://github.com/agronholm/anyio |
| audio-separator | 0.30.2 | MIT | https://github.com/karaokenerds/python-audio-separator |
| audioread | 3.1.0 | MIT | https://github.com/beetbox/audioread |
| av | 18.1.0 | BSD-3-Clause | https://github.com/PyAV-Org/PyAV |
| backports.tarfile | 1.2.0 | MIT | https://github.com/jaraco/backports.tarfile |
| beartype | 0.18.5 | MIT | https://beartype.readthedocs.io |
| certifi | 2026.10.19 | MPL-2.0 | https://github.com/certifi/python-certifi |
| cffi | 2.1.1 | MIT-0 | https://github.com/python-cffi/cffi |
| charset-normalizer | 3.5.2 | MIT |  |
| click | 8.5.0 | BSD-3-Clause | https://github.com/pallets/click/ |
| cloudpickle | 3.1.2 | BSD-3-Clause | https://github.com/cloudpipe/cloudpickle |
| ctranslate2 | 4.8.3 | MIT | https://opennmt.net |
| Cython | 3.3.0 | Apache-2.0 | https://cython.org/ |
| decorator | 5.3.1 | BSD-2-Clause |  |
| demucs | 4.1.0 | MIT | https://github.com/adefossez/demucs |
| diffq | 0.2.4 | Creative Commons Attribution-NonCommercial 4.0 International | https://github.com/facebookresearch/diffq |
| einops | 0.8.2 | MIT | https://github.com/arogozhnikov/einops |
| fastapi | 0.156.8 | MIT | https://github.com/fastapi/fastapi |
| faster-whisper | 1.2.1 | MIT | https://github.com/SYSTRAN/faster-whisper |
| filelock | 4.2.4 | MIT | https://github.com/tox-dev/py-filelock |
| flatbuffers | 25.12.19 | Apache-2.0 | https://google.github.io/flatbuffers/ |
| fsspec | 2026.9.0 | BSD-3-Clause | https://github.com/fsspec/filesystem_spec |
| h11 | 0.16.0 | MIT | https://github.com/python-hyper/h11 |
| hf-xet | 1.7.0 | Apache-2.0 | https://github.com/huggingface/xet-core |
| httpcore | 1.0.9 | BSD-3-Clause | https://www.encode.io/httpcore/ |
| httpcore2 | 2.16.0 | BSD-3-Clause | https://github.com/pydantic/httpx2 |
| httptools | 0.9.0 | MIT | https://github.com/MagicStack/httptools |
| httpx | 0.28.1 | BSD-3-Clause | https://github.com/encode/httpx |
| httpx2 | 2.16.0 | BSD-3-Clause | https://github.com/pydantic/httpx2 |
| huggingface_hub | 2.2.0 | Apache-2.0 | https://github.com/huggingface/huggingface_hub |
| HyperPyYAML | 1.2.3 | Apache-2.0 | https://github.com/speechbrain/HyperPyYAML |
| idna | 3.20 | BSD-3-Clause | https://github.com/kjd/idna |
| importlib_metadata | 9.0.1 | Apache-2.0 | https://github.com/python/importlib_metadata |
| iniconfig | 2.3.1 | MIT | https://github.com/pytest-dev/iniconfig |
| jaraco.classes | 3.4.0 | MIT | https://github.com/jaraco/jaraco.classes |
| jaraco.context | 6.1.2 | MIT | https://github.com/jaraco/jaraco.context |
| jaraco.functools | 4.6.0 | MIT | https://github.com/jaraco/jaraco.functools |
| Jinja2 | 3.1.6 | BSD | https://github.com/pallets/jinja/ |
| joblib | 1.6.0 | BSD-3-Clause | https://joblib.readthedocs.io |
| julius | 0.2.8 | MIT | https://github.com/adefossez/julius |
| keyring | 25.7.0 | MIT | https://github.com/jaraco/keyring |
| lameenc | 1.9.0 | LGPL-3.0-or-later | https://github.com/chrisstaite/lameenc |
| lazy-loader | 0.6 | BSD-3-Clause | https://scientific-python.org/specs/spec-0001/ |
| librosa | 0.11.0 | ISC | https://librosa.org |
| llvmlite | 0.50.0 | BSD-2-Clause AND Apache-2.0 WITH LLVM-exception | http://llvmlite.readthedocs.io |
| MarkupSafe | 3.0.4 | BSD-3-Clause | https://github.com/pallets/markupsafe/ |
| ml_collections | 1.1.0 | Apache-2.0 | https://github.com/google/ml_collections |
| ml_dtypes | 0.5.4 | Apache-2.0 | https://github.com/jax-ml/ml_dtypes |
| mlx | 0.32.4 | MIT | https://github.com/ml-explore/mlx |
| mlx-metal | 0.32.4 | MIT | https://github.com/ml-explore/mlx |
| mlx-whisper | 0.4.3 | MIT | https://github.com/ml-explore/mlx-examples |
| more-itertools | 11.2.1 | MIT | https://github.com/more-itertools/more-itertools |
| mpmath | 1.3.0 | BSD | http://mpmath.org/ |
| msgpack | 1.2.3 | Apache-2.0 | https://msgpack.org/ |
| narwhals | 2.27.1 | MIT | https://github.com/narwhals-dev/narwhals |
| networkx | 3.6.1 | BSD-3-Clause | https://networkx.org/ |
| numba | 0.68.0 | BSD | https://numba.pydata.org |
| numpy | 1.26.4 | BSD | https://numpy.org |
| onnx | 1.23.2 | Apache-2.0 | https://onnx.ai/ |
| onnx-asr | 0.12.0 | MIT | https://github.com/istupakov/onnx-asr |
| onnx2torch | 1.5.15 | Apache-2.0 | https://enot.ai |
| onnxruntime | 1.31.0 | MIT | https://onnxruntime.ai |
| opentelemetry-api | 1.45.1 | Apache-2.0 | https://github.com/open-telemetry/opentelemetry-python/tree/main/opentelemetry-api |
| packaging | 26.3 | Apache-2.0 OR BSD-2-Clause | https://github.com/pypa/packaging |
| pillow | 13.0.0 | MIT-CMU | https://python-pillow.org |
| platformdirs | 4.13.5 | MIT | https://github.com/tox-dev/platformdirs |
| pluggy | 1.7.0 | MIT | https://github.com/pytest-dev/pluggy |
| pooch | 1.9.0 | BSD-3-Clause | https://github.com/fatiando/pooch |
| protobuf | 7.36.2 | BSD-3-Clause | https://developers.google.com/protocol-buffers/ |
| pycparser | 3.11 | BSD-3-Clause | https://github.com/eliben/pycparser |
| pydantic | 2.14.1 | MIT | https://github.com/pydantic/pydantic |
| pydantic_core | 2.50.1 | MIT | https://github.com/pydantic/pydantic |
| pydub | 0.25.1 | MIT | http://pydub.com |
| pyloudnorm | 0.2.0 | MIT | https://github.com/csteinmetz1/pyloudnorm |
| pytest | 9.1.1 | MIT | https://docs.pytest.org/en/latest/ |
| pytest-asyncio | 1.4.0 | Apache-2.0 | https://github.com/pytest-dev/pytest-asyncio |
| python-dotenv | 1.2.4 | BSD-3-Clause | https://github.com/theskumar/python-dotenv |
| python-multipart | 0.0.32 | Apache-2.0 | https://github.com/Kludex/python-multipart |
| PyYAML | 6.0.3 | MIT | https://pyyaml.org/ |
| regex | 2026.9.29 | Apache-2.0 AND CNRI-Python | https://github.com/mrabarnett/mrab-regex |
| requests | 2.34.2 | Apache-2.0 | https://github.com/psf/requests |
| resampy | 0.4.3 | ISC | https://github.com/bmcfee/resampy |
| rotary-embedding-torch | 0.6.5 | MIT | https://github.com/lucidrains/rotary-embedding-torch |
| ruamel.yaml | 0.18.17 | MIT | https://sourceforge.net/p/ruamel-yaml/ |
| ruamel.yaml.clib | 0.2.15 | MIT | https://sourceforge.net/p/ruamel-yaml-clib/ |
| ruff | 0.17.1 | MIT | https://docs.astral.sh/ruff |
| safetensors | 0.8.0 | Apache-2.0 | https://github.com/huggingface/safetensors |
| samplerate | 0.1.0 | MIT | https://github.com/tuxu/python-samplerate |
| scikit-learn | 1.9.1 | BSD-3-Clause | https://scikit-learn.org |
| scipy | 1.17.1 | BSD | https://scipy.org/ |
| sentencepiece | 0.2.2 | Apache-2.0 | https://github.com/google/sentencepiece |
| six | 1.17.0 | MIT | https://github.com/benjaminp/six |
| soundfile | 0.14.0 | BSD-3-Clause | https://github.com/bastibe/python-soundfile |
| soxr | 1.1.0 | LGPL-2.1-or-later | https://github.com/dofuuz/python-soxr |
| speechbrain | 1.1.1 | Apache-2.0 | https://speechbrain.github.io/ |
| sphn | 0.2.1 | не указана в метаданных PyPI | https://github.com/kyutai-labs/sphn |
| starlette | 1.9.0 | BSD-3-Clause | https://github.com/Kludex/starlette |
| sympy | 1.13.1 | BSD | https://sympy.org |
| threadpoolctl | 3.7.0 | BSD-3-Clause | https://github.com/joblib/threadpoolctl |
| tiktoken | 0.14.0 | MIT | https://github.com/openai/tiktoken |
| tokenizers | 0.23.3 | Apache-2.0 | https://github.com/huggingface/tokenizers |
| torch | 2.5.1 | BSD-3-Clause | https://pytorch.org/ |
| torchaudio | 2.5.1 | BSD | https://github.com/pytorch/audio |
| torchvision | 0.20.1 | BSD | https://github.com/pytorch/vision |
| tqdm | 4.70.1 | MPL-2.0 AND MIT | https://tqdm.github.io |
| truststore | 0.10.5 | MIT | https://github.com/sethmlarson/truststore |
| typing-inspection | 0.4.4 | MIT | https://github.com/pydantic/typing-inspection |
| typing_extensions | 4.16.0 | PSF-2.0 | https://github.com/python/typing_extensions |
| urllib3 | 2.8.0 | MIT |  |
| uvicorn | 0.54.0 | BSD-3-Clause | https://uvicorn.dev/ |
| uvloop | 0.23.0 | MIT |  |
| watchfiles | 1.3.0 | MIT | https://github.com/samuelcolvin/watchfiles |
| websockets | 17.3 | BSD-3-Clause | https://github.com/python-websockets/websockets |
| yt-dlp | 2026.8.19 | Unlicense | https://github.com/yt-dlp/yt-dlp |
| zipp | 4.1.1 | MIT | https://github.com/jaraco/zipp |

## npm: интерфейс

`pnpm licenses list --prod` в папке `app/`.

| Пакет | Версия | Лицензия | Страница |
|---|---|---|---|
| @fontsource-variable/onest | 5.3.1 | OFL-1.1 | https://fontsource.org/fonts/onest |
| @fontsource/jetbrains-mono | 5.3.0 | OFL-1.1 | https://fontsource.org/fonts/jetbrains-mono |
| @remix-run/router | 1.23.4 | MIT | https://github.com/remix-run/react-router#readme |
| @tauri-apps/api | 2.12.4 | Apache-2.0 OR MIT | https://github.com/tauri-apps/tauri#readme |
| @tauri-apps/plugin-dialog | 2.8.3 | MIT OR Apache-2.0 | https://github.com/tauri-apps/plugins-workspace#readme |
| @tauri-apps/plugin-opener | 2.7.2 | MIT OR Apache-2.0 | https://github.com/tauri-apps/plugins-workspace#readme |
| @tauri-apps/plugin-shell | 2.5.0 | MIT OR Apache-2.0 | https://github.com/tauri-apps/plugins-workspace#readme |
| js-tokens | 4.0.0 | MIT | https://github.com/lydell/js-tokens#readme |
| loose-envify | 1.4.0 | MIT | https://github.com/zertosh/loose-envify |
| lucide-react | 1.59.0 | ISC | https://lucide.dev |
| react | 18.3.1 | MIT | https://reactjs.org/ |
| react-dom | 18.3.1 | MIT | https://reactjs.org/ |
| react-router | 6.30.6 | MIT | https://github.com/remix-run/react-router#readme |
| react-router-dom | 6.30.6 | MIT | https://github.com/remix-run/react-router#readme |
| scheduler | 0.23.2 | MIT | https://reactjs.org/ |
| zustand | 5.0.15 | MIT | https://github.com/pmndrs/zustand |

## Rust: оболочка

Прямые зависимости (`app/src-tauri/Cargo.toml`): tauri 2, tauri-plugin-shell 2, tauri-plugin-dialog 2, tauri-plugin-opener 2, serde 1, serde_json 1, tauri-build 2. Лаунчер движка (`packaging/launcher`) не использует сторонних крейтов.

Всего в `Cargo.lock` 469 крейтов с лицензиями (`cargo metadata`):

| Лицензия (как указано в крейте) | Крейтов |
|---|---|
| MIT OR Apache-2.0 | 210 |
| MIT | 114 |
| Apache-2.0 OR MIT | 49 |
| Unicode-3.0 | 18 |
| Zlib OR Apache-2.0 OR MIT | 17 |
| MIT/Apache-2.0 | 13 |
| Unlicense OR MIT | 9 |
| MPL-2.0 | 5 |
| Apache-2.0 WITH LLVM-exception OR Apache-2.0 OR MIT | 5 |
| Apache-2.0/MIT | 3 |
| BSD-3-Clause | 2 |
| Zlib | 2 |
| MIT OR Zlib OR Apache-2.0 | 2 |
| BSD-3-Clause OR MIT OR Apache-2.0 | 2 |
| MIT OR Apache-2.0 OR LGPL-2.1-or-later | 2 |
| MIT OR Apache-2.0 OR Zlib | 2 |
| Unlicense/MIT | 2 |
| Apache-2.0 | 2 |
| 0BSD OR MIT OR Apache-2.0 | 1 |
| BSD-3-Clause AND MIT | 1 |
| BSD-3-Clause/MIT | 1 |
| Apache-2.0 AND MIT | 1 |
| CC0-1.0 OR MIT-0 OR Apache-2.0 | 1 |
| (Apache-2.0 OR MIT) AND BSD-3-Clause | 1 |
| Apache-2.0 / MIT | 1 |
| ISC | 1 |
| Apache-2.0 WITH LLVM-exception | 1 |
| (MIT OR Apache-2.0) AND Unicode-3.0 | 1 |

Полный список: `cd app/src-tauri && cargo metadata --locked --format-version 1`.

## Как обновить этот файл

- Python: взять окружение с `dubl-engine[ml]` и вывести лицензии через `importlib.metadata` или `pip-licenses --with-urls --format=markdown`.
- npm: `cd app && pnpm licenses list --prod --json`.
- Rust: `cd app/src-tauri && cargo metadata --locked --format-version 1` (поле `license` у крейтов).
- ffmpeg: адреса сборок и `NOTICE-ffmpeg.txt` берутся из `packaging/fetch_ffmpeg.py`.
