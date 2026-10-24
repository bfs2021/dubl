#!/usr/bin/env python3
"""Сборка sidecar-а «Дубль»: колесо движка → PyInstaller (--onedir) → Rust-лаунчер → «ярус» для Tauri.

Результат (по умолчанию в packaging/build/stage):
  binaries/dubl-engine-<target-triple>[.exe]   лаунчер (то, что Tauri считает sidecar-ом)
  resources/engine/                            папка PyInstaller (dubl-engine + _internal/)
  resources/ffmpeg/                            ffmpeg/ffprobe (кладёт fetch_ffmpeg.py)

Использование:
  python packaging/build_engine.py                       # всё в packaging/build/stage
  python packaging/build_engine.py --out app/src-tauri   # сразу в проект Tauri
  python packaging/build_engine.py --skip-launcher       # без Rust (только PyInstaller)

Нужны: uv (PATH или $UV), Rust (cargo, rustc) для лаунчера. Python 3.11 uv скачает сам.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
BUILD = HERE / "build"
EXE = ".exe" if os.name == "nt" else ""
CARGO_ENV = dict(os.environ, PATH=os.pathsep.join([str(Path.home() / ".cargo" / "bin"), os.environ.get("PATH", "")]))


def sh(cmd: list, **kw) -> None:
    print("$", " ".join(map(str, cmd)), flush=True)
    subprocess.run([str(c) for c in cmd], check=True, **kw)


def find_uv() -> str:
    for c in (os.environ.get("UV"), shutil.which("uv"), str(Path.home() / ".local/bin/uv")):
        if c and Path(c).exists():
            return c
    sys.exit("Не найден uv (https://docs.astral.sh/uv/). Задайте путь в переменной UV.")


def host_triple() -> str:
    out = subprocess.run(["rustc", "--print", "host-tuple"], capture_output=True, text=True, encoding="utf-8", errors="replace", env=CARGO_ENV)
    if out.returncode == 0 and out.stdout.strip():
        return out.stdout.strip()
    out = subprocess.run(["rustc", "-vV"], capture_output=True, text=True, encoding="utf-8", errors="replace", env=CARGO_ENV)  # старый rustc
    for line in out.stdout.splitlines():
        if line.startswith("host:"):
            return line.split(":", 1)[1].strip()
    sys.exit("rustc не найден: поставьте Rust (https://rustup.rs)")


def venv_py(venv: Path) -> Path:
    return venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(BUILD / "stage"), help="куда класть binaries/ и resources/")
    ap.add_argument("--skip-launcher", action="store_true")
    ap.add_argument("--clean", action="store_true", help="удалить packaging/build перед сборкой")
    args = ap.parse_args()
    out = Path(args.out).resolve()
    t0 = time.time()
    if args.clean and BUILD.exists():
        shutil.rmtree(BUILD)
    BUILD.mkdir(exist_ok=True)
    uv = find_uv()

    # 1. колесо движка
    wheel_dir = BUILD / "wheel"
    shutil.rmtree(wheel_dir, ignore_errors=True)
    sh([uv, "build", "--wheel", "--out-dir", wheel_dir, ROOT / "engine"])
    wheel = next(wheel_dir.glob("dubl_engine-*.whl"))

    # 2. чистое окружение сборки: только лёгкие зависимости + PyInstaller
    venv = BUILD / "venv"
    if not venv_py(venv).exists():
        sh([uv, "venv", "--python", "3.11", venv])
    sh([uv, "pip", "install", "--python", venv_py(venv), "--reinstall-package", "dubl-engine", wheel, "pyinstaller>=6.10"])

    # 3. PyInstaller
    dist, work = BUILD / "dist", BUILD / "work"
    shutil.rmtree(dist, ignore_errors=True)
    sh([venv_py(venv), "-m", "PyInstaller", "--noconfirm", "--clean", "--distpath", dist, "--workpath", work,
        HERE / "dubl-engine.spec"], env=dict(os.environ, DUBL_WHEEL_DIR=str(wheel_dir)))
    engine_dir = dist / "engine"
    if not (engine_dir / f"dubl-engine{EXE}").exists():
        sys.exit("PyInstaller не создал dist/engine/dubl-engine")

    # 4. лаунчер (Rust) — единственный файл, который Tauri знает как sidecar
    triple = host_triple()
    stage_bin, stage_res = out / "binaries", out / "resources"
    stage_bin.mkdir(parents=True, exist_ok=True)
    if not args.skip_launcher:
        sh(["cargo", "build", "--release", "--manifest-path", HERE / "launcher" / "Cargo.toml"], env=CARGO_ENV)
        shutil.copy2(HERE / "launcher" / "target" / "release" / f"dubl-engine{EXE}",
                     stage_bin / f"dubl-engine-{triple}{EXE}")

    # 5. папка движка в ресурсы (symlinks сохраняем: на macOS/Linux в _internal их много)
    dest = stage_res / "engine"
    shutil.rmtree(dest, ignore_errors=True)
    stage_res.mkdir(parents=True, exist_ok=True)
    shutil.copytree(engine_dir, dest, symlinks=True)

    size = sum(f.stat().st_size for f in dest.rglob("*") if f.is_file())
    print(f"\nГотово за {time.time() - t0:.0f} с. Платформа: {triple}")
    print(f"  sidecar: {stage_bin / ('dubl-engine-' + triple + EXE)}")
    print(f"  engine/: {dest}  ({size / 1e6:.0f} МБ)")
    print(f"  ffmpeg:  python packaging/fetch_ffmpeg.py --out {stage_res / 'ffmpeg'}")


if __name__ == "__main__":
    for _s in (sys.stdout, sys.stderr):   # Windows: консоль/пайп в cp1252 — русский print иначе падает (UnicodeEncodeError)
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
        except Exception:  # noqa: BLE001
            pass
    main()
