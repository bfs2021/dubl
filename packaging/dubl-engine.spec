# -*- mode: python ; coding: utf-8 -*-
# PyInstaller: лёгкий движок «Дубль» (без torch) в режиме --onedir.
# Запускается через build_engine.py (он задаёт DUBL_WHEEL_DIR), руками:
#   pyinstaller --noconfirm --distpath build/dist --workpath build/work dubl-engine.spec
import os
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules, copy_metadata

HERE = Path(SPECPATH)  # noqa: F821  (задаётся PyInstaller)
wheel_dir = Path(os.environ.get("DUBL_WHEEL_DIR", HERE / "build" / "wheel"))

hidden = []
for pkg in ("dubl_engine", "uvicorn", "keyring", "yt_dlp", "websockets", "anyio"):
    hidden += collect_submodules(pkg)
hidden += ["ml_pack", "multipart", "python_multipart"]

datas = []
for dist in ("dubl-engine", "keyring", "fastapi", "uvicorn", "yt-dlp", "pydantic", "pyloudnorm"):
    try:
        datas += copy_metadata(dist)
    except Exception:  # noqa: BLE001  (метаданные необязательны)
        pass
datas += collect_data_files("certifi")
# колесо движка — для установки нейросетевого пакета (ml_pack.find_wheel ищет в <_MEIPASS>/ml)
datas += [(str(w), "ml") for w in sorted(wheel_dir.glob("dubl_engine-*.whl"))]

# Всё нейросетевое ставится отдельно (ml_pack.py) — в лёгкую сборку оно попасть не должно.
excludes = [
    "torch", "torchaudio", "torchvision", "faster_whisper", "ctranslate2", "demucs", "speechbrain",
    "audio_separator", "pyannote", "sklearn", "onnxruntime", "tensorflow", "numba", "llvmlite", "librosa",
    "matplotlib", "tkinter", "IPython", "pytest", "PIL", "pandas", "setuptools", "pip",
]

a = Analysis(  # noqa: F821
    [str(HERE / "engine_entry.py")],
    pathex=[str(HERE)],
    datas=datas,
    hiddenimports=hidden,
    excludes=excludes,
    noarchive=False,
)
pyz = PYZ(a.pure)  # noqa: F821
# UTF-8 mode (как PYTHONUTF8=1): на Windows open()/read_text() без encoding= и stdio иначе в cp1251/cp1252 —
# страховка поверх явных encoding="utf-8" в коде (их проверяет engine/tests/test_text_encoding.py).
options = [("X utf8", None, "OPTION")]
exe = EXE(  # noqa: F821
    pyz, a.scripts, options,
    exclude_binaries=True,
    name="dubl-engine",
    console=True,       # нужен stdout: Tauri читает DUBL_ENGINE_READY
    upx=False,
    strip=False,
)
coll = COLLECT(exe, a.binaries, a.datas, name="engine", upx=False, strip=False)  # noqa: F821
