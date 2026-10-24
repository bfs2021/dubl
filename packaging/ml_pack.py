"""Установка «нейросетевого пакета» (torch, whisper, demucs, …) рядом с лёгким движком.

Зачем: в установщик кладём лёгкий движок (~50 МБ), а torch & co (сотни МБ – несколько ГБ,
зависит от CUDA) докачиваются при первом запуске кнопкой «Скачать нейросети».

Как: uv скачивает python-build-standalone 3.11, создаёт venv в <data_dir>/runtime/ml/venv
и ставит туда колесо dubl-engine[ml], вшитое в сборку. Дальше лёгкий движок (engine_entry.py)
сам запускает движок из этого venv.

CLI (через sidecar):  dubl-engine install-ml-pack [--data-dir D] [--device auto|cpu|cuda]
Прогресс — строки `DUBL_ML_STAGE <stage> <i>/<n>` в stdout, остальное — вывод uv.
Итог: `DUBL_ML_DONE` (код 0) или `DUBL_ML_FAILED <причина>` (код 1).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path

TORCH_VERSION = "2.5.1"
UV_VERSION = "0.12.5"  # скачиваем, только если uv нет ни в PATH, ни в DUBL_UV
# SHA-256 архивов с https://github.com/astral-sh/uv/releases/download/<ver>/uv-<triple>.<ext>.sha256
UV_SHA256 = {
    "aarch64-apple-darwin": "5bb0e5fe008a773c3dbcb97ff79cd89e1241464fe9d2f986d52ad8f1b037bd62",
    "x86_64-apple-darwin": "b3b2137477cf96c9686ebfb71524614cec780c673fd73e59bce099aef02e70e8",
    "x86_64-pc-windows-msvc": "4c4d49d8738847d9b71ba319e49a5688c93eac0fe6204b1df24e98528dddf39a",
    "aarch64-pc-windows-msvc": "724279317fee6e5fa8ad1908e4eba2bbe764ef1ece5b3f4597927b62b1fe562a",
    "x86_64-unknown-linux-gnu": "68a509da24b06b4223a1c0175fb5eb5bc79342b76cbeff0cfe51ac3f5b17b6b2",
    "aarch64-unknown-linux-gnu": "9bf43b4d1a07665bf64d4c4e710930b382321a785e0eb10aac07f46471f86a31",
}
PY_VERSION = "3.11"
TORCH_INDEX = {"cpu": "https://download.pytorch.org/whl/cpu", "cuda": "https://download.pytorch.org/whl/cu124"}
STAGES = 4


def stage(i: int, name: str) -> None:
    print(f"DUBL_ML_STAGE {name} {i}/{STAGES}", flush=True)


def clean_env() -> dict[str, str]:
    """Окружение без хвостов PyInstaller (иначе дочерний Python/uv подхватит наши библиотеки)."""
    env = dict(os.environ)
    for k in ("PYTHONHOME", "PYTHONPATH", "_MEIPASS2", "_PYI_APPLICATION_HOME_DIR"):
        env.pop(k, None)
    for var in ("LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH", "DYLD_FALLBACK_LIBRARY_PATH"):
        orig = env.pop(var + "_ORIG", None)
        if orig is not None:
            env[var] = orig
        elif getattr(sys, "frozen", False):
            env.pop(var, None)
    return env


def gpu_env(data_dir: Path, env: dict[str, str]) -> None:
    """CUDA-библиотеки из колёс torch/nvidia-* лежат в site-packages; ctranslate2 (faster-whisper) сам их не ищет.
    Windows: добавляем в PATH; Linux: в LD_LIBRARY_PATH. На Mac и без CUDA ничего не делает."""
    venv = runtime_dir(data_dir) / "venv"
    sp = next(iter((venv / "Lib").glob("site-packages")), None) if os.name == "nt" else \
        next(iter((venv / "lib").glob("python3*/site-packages")), None)
    if sp is None or sys.platform == "darwin":
        return
    if os.name == "nt":
        dirs = [sp / "torch" / "lib", *sp.glob("nvidia/*/bin")]
        key = "PATH"
    else:
        dirs = [*sp.glob("nvidia/*/lib"), sp / "torch" / "lib"]
        key = "LD_LIBRARY_PATH"
    extra = [str(d) for d in dirs if d.is_dir()]
    if extra:
        env[key] = os.pathsep.join(extra + ([env[key]] if env.get(key) else []))


def runtime_dir(data_dir: Path) -> Path:
    return data_dir / "runtime" / "ml"


def venv_python(data_dir: Path) -> Path:
    v = runtime_dir(data_dir) / "venv"
    return v / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def marker_path(data_dir: Path) -> Path:
    return runtime_dir(data_dir) / "ml-pack.json"


def read_marker(data_dir: Path) -> dict | None:
    try:
        return json.loads(marker_path(data_dir).read_text("utf-8"))
    except (OSError, ValueError):
        return None


def detect_device() -> str:
    """cuda, если виден NVIDIA-драйвер (Windows/Linux); на Mac torch из PyPI уже умеет MPS."""
    if sys.platform == "darwin":
        return "cpu"  # обычные колёса PyPI; MPS включается сам
    return "cuda" if shutil.which("nvidia-smi") else "cpu"


def uv_triple() -> tuple[str, str]:
    m = platform.machine().lower()
    arch = "aarch64" if m in ("arm64", "aarch64") else "x86_64"
    if sys.platform == "darwin":
        return f"{arch}-apple-darwin", "tar.gz"
    if sys.platform == "win32":
        return f"{arch}-pc-windows-msvc", "zip"
    return f"{arch}-unknown-linux-gnu", "tar.gz"


def find_uv(data_dir: Path) -> str:
    for cand in (os.environ.get("DUBL_UV"), shutil.which("uv")):
        if cand and Path(cand).exists():
            return cand
    bin_dir = data_dir / "runtime" / "bin"
    exe = bin_dir / ("uv.exe" if os.name == "nt" else "uv")
    if exe.exists():
        return str(exe)
    triple, ext = uv_triple()
    url = f"https://github.com/astral-sh/uv/releases/download/{UV_VERSION}/uv-{triple}.{ext}"
    print(f"Скачиваю uv: {url}", flush=True)
    bin_dir.mkdir(parents=True, exist_ok=True)
    archive = bin_dir / f"uv.{ext}"
    urllib.request.urlretrieve(url, archive)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    if digest != UV_SHA256.get(triple):
        archive.unlink()
        raise RuntimeError(f"контрольная сумма uv не совпала ({digest[:12]}…) — загрузка повреждена или подменена")
    if ext == "zip":
        with zipfile.ZipFile(archive) as z:
            z.extractall(bin_dir)
    else:
        with tarfile.open(archive) as t:
            for m in t.getmembers():
                if m.isfile() and Path(m.name).name == "uv":
                    m.name = "uv"
                    t.extract(m, bin_dir)
    archive.unlink()
    if not exe.exists():  # zip кладёт в подпапку
        found = next(bin_dir.rglob(exe.name), None)
        if not found:
            raise RuntimeError("uv не найден в скачанном архиве")
        shutil.move(str(found), exe)
    exe.chmod(0o755)
    return str(exe)


def find_wheel() -> Path:
    env = os.environ.get("DUBL_ENGINE_WHEEL")
    if env:
        return Path(env)
    roots = [Path(getattr(sys, "_MEIPASS", "")) / "ml", Path(__file__).parent / "build" / "wheel"]
    for r in roots:
        wheels = sorted(r.glob("dubl_engine-*.whl")) if r.is_dir() else []
        if wheels:
            return wheels[-1]
    raise RuntimeError("В сборке нет колеса dubl_engine (ожидалось ml/dubl_engine-*.whl)")


def wheel_hash(wheel: Path) -> str:
    return hashlib.sha256(wheel.read_bytes()).hexdigest()


def sync_engine(data_dir: Path) -> bool:
    """Нейросети уже стоят, а код движка в сборке новее (другое колесо) — переставить в venv только сам движок.

    Версия пакета у нас не меняется от сборки к сборке, поэтому сравниваем хэш колеса. Без этого после обновления
    программы движок с нейросетями продолжал бы работать со старым кодом. Ставим без зависимостей (--no-deps):
    это секунды и без сети. True — переставили, False — и так свежий или переставить не вышло (тогда работаем как было).
    """
    marker = read_marker(data_dir)
    py = venv_python(data_dir)
    if not marker or not py.exists():
        return False
    try:
        wheel = find_wheel()
    except RuntimeError:
        return False
    h = wheel_hash(wheel)
    need_deps = marker.get("added_deps") != ADDED_ML_DEPS
    if marker.get("wheel_sha256") == h and not need_deps:
        return False
    env = clean_env()
    env["UV_CACHE_DIR"] = str(data_dir / "runtime" / "uv-cache")
    env.pop("VIRTUAL_ENV", None)
    try:
        uv = find_uv(data_dir)
        if need_deps and add_new_deps(uv, py, env):
            marker["added_deps"] = ADDED_ML_DEPS
            marker_path(data_dir).write_text(json.dumps(marker, ensure_ascii=False, indent=2), "utf-8")
        if marker.get("wheel_sha256") == h:
            return False
        cmd = [uv, "pip", "install", "--python", str(py), "--reinstall-package", "dubl-engine", "--no-deps", str(wheel)]
        p = subprocess.run(cmd, env=env, stdout=sys.stderr, stderr=sys.stderr)   # stdout занят строкой READY
        if p.returncode != 0:
            return False
    except Exception as e:  # noqa: BLE001
        print(f"Не удалось обновить код движка в окружении нейросетей: {e}", file=sys.stderr, flush=True)
        return False
    finally:
        shutil.rmtree(data_dir / "runtime" / "uv-cache", ignore_errors=True)
    marker["wheel_sha256"] = h
    marker["wheel"] = wheel.name
    marker_path(data_dir).write_text(json.dumps(marker, ensure_ascii=False, indent=2), "utf-8")
    print("Код движка в окружении нейросетей обновлён.", file=sys.stderr, flush=True)
    return True


# Зависимости, которые появились в extra «ml» после первых установок нейросетевого пакета. sync_engine ставит код движка
# без зависимостей (--no-deps), поэтому у тех, кто скачал нейросети раньше, их не было бы. Ставим их отдельно при запуске
# (onnx-asr — десятки КБ, onnxruntime обычно уже есть: его тянет faster-whisper), не трогая torch и остальное. Не вышло
# (нет сети) — не беда: движок работает и без них (GigaAM → Whisper с пояснением), попробуем при следующем запуске.
# Удалось — запоминаем в ml-pack.json («added_deps»). Держать в согласии с extra «ml» в engine/pyproject.toml.
ADDED_ML_DEPS = ["onnx-asr>=0.12", "onnxruntime>=1.18,!=1.24.1,!=1.25.*,!=1.26.0"]


def add_new_deps(uv: str, py: Path, env: dict[str, str], timeout: float = 180) -> bool:
    try:
        # уже стоят (поставил кто-то раньше) — без сети и uv
        chk = subprocess.run([str(py), "-c", "import onnx_asr, onnxruntime"], env=env, capture_output=True, timeout=60)
        if chk.returncode == 0:
            return True
    except Exception:  # noqa: BLE001, S110 — проверить не вышло: просто ставим
        pass
    try:
        p = subprocess.run([uv, "pip", "install", "--python", str(py), *ADDED_ML_DEPS], env=env,
                           stdout=sys.stderr, stderr=sys.stderr, timeout=timeout)
        return p.returncode == 0
    except Exception as e:  # noqa: BLE001
        print(f"Новые нейросетевые пакеты не поставились (поставим позже): {e}", file=sys.stderr, flush=True)
        return False


def run(cmd: list[str], env: dict[str, str]) -> None:
    print("$ " + " ".join(cmd), flush=True)
    p = subprocess.run(cmd, env=env)
    if p.returncode != 0:
        raise RuntimeError(f"команда завершилась с кодом {p.returncode}: {cmd[0]} {cmd[1] if len(cmd) > 1 else ''}")


def install(data_dir: Path, device: str = "auto") -> None:
    if device == "auto":
        device = detect_device()
    rt = runtime_dir(data_dir)
    rt.mkdir(parents=True, exist_ok=True)
    env = clean_env()
    env["UV_PYTHON_INSTALL_DIR"] = str(data_dir / "runtime" / "python")
    env["UV_CACHE_DIR"] = str(data_dir / "runtime" / "uv-cache")
    env.pop("VIRTUAL_ENV", None)

    stage(1, "uv")
    uv = find_uv(data_dir)
    wheel = find_wheel()

    stage(2, "python")
    venv = rt / "venv"
    if venv.exists():
        shutil.rmtree(venv)
    run([uv, "venv", "--python", PY_VERSION, "--managed-python", str(venv)], env)
    py = str(venv_python(data_dir))

    stage(3, "torch")
    torch_cmd = [uv, "pip", "install", "--python", py, f"torch=={TORCH_VERSION}", f"torchaudio=={TORCH_VERSION}"]
    if sys.platform != "darwin" and (device == "cuda" or sys.platform.startswith("linux")):
        # Linux: PyPI-колесо тянет CUDA (~2,5 ГБ) — для cpu берём лёгкое с индекса PyTorch.
        # Windows: PyPI-колесо = CPU; для cuda — индекс cu124.
        # --index (не --index-url): остальные пакеты по-прежнему ищутся на PyPI.
        torch_cmd += ["--index", TORCH_INDEX[device]]
    run(torch_cmd, env)

    stage(4, "engine")
    # Mac на Apple Silicon: распознавание речи на GPU через mlx-whisper (extra «mlx» с маркером платформы).
    # На других платформах extra пустой, но зависимость не ставим, чтобы не лезть в чужие колёса.
    mac_arm = sys.platform == "darwin" and platform.machine().lower() in ("arm64", "aarch64")
    run([uv, "pip", "install", "--python", py, f"{wheel}[ml,mlx]" if mac_arm else f"{wheel}[ml]"], env)

    # Кэш uv после успешной установки не нужен: он почти равен размеру самого окружения (~1 ГБ).
    # При ошибке кэш остаётся — повторная попытка не будет качать всё заново.
    shutil.rmtree(data_dir / "runtime" / "uv-cache", ignore_errors=True)

    version = wheel.name.split("-")[1]
    marker_path(data_dir).write_text(json.dumps({
        "engine_version": version, "device": device, "torch": TORCH_VERSION, "wheel": wheel.name,
        "wheel_sha256": wheel_hash(wheel), "added_deps": ADDED_ML_DEPS,
    }, ensure_ascii=False, indent=2), "utf-8")


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="dubl-engine install-ml-pack")
    ap.add_argument("--data-dir", default=os.environ.get("DUBL_DATA_DIR"))
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = ap.parse_args(argv)
    if args.data_dir:
        data_dir = Path(args.data_dir)
    else:
        from platformdirs import user_data_dir
        data_dir = Path(user_data_dir("Dubl", "Dubl"))
    try:
        install(data_dir, args.device)
    except Exception as e:  # noqa: BLE001
        print(f"DUBL_ML_FAILED {e}", flush=True)
        return 1
    print("DUBL_ML_DONE", flush=True)
    return 0


if __name__ == "__main__":
    for _s in (sys.stdout, sys.stderr):   # Windows: консоль/пайп в cp1252 — русский print иначе падает (UnicodeEncodeError)
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
        except Exception:  # noqa: BLE001
            pass
    sys.exit(main(sys.argv[1:]))
