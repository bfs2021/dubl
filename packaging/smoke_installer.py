#!/usr/bin/env python3
"""Дымовая проверка готовых установщиков: ставим/распаковываем в папку с кириллицей и пробелом и запускаем движок.

  python packaging/smoke_installer.py app/src-tauri/target/release/bundle

Linux: AppImage (--appimage-extract) и deb (dpkg-deb -x). Windows: NSIS (тихая установка /S /D=…) и MSI
(административная установка — распаковка без записи в систему). macOS: пропуск (dmg проверяется вручную).
Для каждого установщика — `smoke_sidecar.installed`: self-check (ffmpeg из ресурсов, UTF-8), READY, /health, /projects.
Выход 0 — всё хорошо.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from smoke_sidecar import installed  # noqa: E402

SIDECAR = "dubl-engine.exe" if os.name == "nt" else "dubl-engine"


def _run(cmd: list | str, **kw) -> None:
    """Строка — передаётся в CreateProcess как есть (NSIS и msiexec разбирают свою командную строку сами и не
    понимают аргумент, целиком взятый в кавычки, как сделал бы subprocess для пути с пробелом)."""
    print("$", cmd if isinstance(cmd, str) else " ".join(map(str, cmd)), flush=True)
    subprocess.run(cmd if isinstance(cmd, str) else [str(c) for c in cmd], check=True, **kw)


def _find_sidecar(root: Path) -> Path:
    """Лаунчер движка, а не сам движок: тот лежит в …/engine/dubl-engine."""
    hits = [p for p in root.rglob(SIDECAR) if p.is_file() and p.parent.name != "engine"]
    if not hits:
        raise SystemExit(f"в {root} нет {SIDECAR}")
    return sorted(hits, key=lambda p: len(p.parts))[0]


def check_appimage(f: Path, work: Path) -> None:
    f.chmod(0o755)
    _run([f, "--appimage-extract"], cwd=work, stdout=subprocess.DEVNULL)
    installed(_find_sidecar(work / "squashfs-root"))


def check_deb(f: Path, work: Path) -> None:
    _run(["dpkg-deb", "-x", f, work])
    installed(_find_sidecar(work))


def check_nsis(f: Path, work: Path) -> None:
    target = work / "Программа"
    # /D=… — последним аргументом и без кавычек, даже если в пути пробелы (так требует NSIS)
    _run(f'"{f}" /S /D={target}')
    try:
        installed(_find_sidecar(target))
    finally:
        un = target / "uninstall.exe"
        if un.exists():
            subprocess.run([str(un), "/S"], check=False)


def check_msi(f: Path, work: Path) -> None:
    target = work / "Распаковка"
    _run(f'msiexec /a "{f}" /qn TARGETDIR="{target}"')
    installed(_find_sidecar(target))


CHECKS = {".AppImage": check_appimage, ".deb": check_deb, ".exe": check_nsis, ".msi": check_msi}


def main() -> int:
    bundle = Path(sys.argv[1] if len(sys.argv) > 1 else "app/src-tauri/target/release/bundle").resolve()
    found = [p for p in sorted(bundle.rglob("*")) if p.is_file() and p.suffix in CHECKS
             and p.parent.name in ("appimage", "deb", "nsis", "msi")]
    if not found:
        if sys.platform == "darwin":
            print("macOS: установщики здесь не проверяем")
            return 0
        raise SystemExit(f"в {bundle} нет установщиков")
    for f in found:
        print(f"\n=== {f.name} ({f.stat().st_size / 1e6:.0f} МБ)", flush=True)
        work = Path(tempfile.mkdtemp(prefix="dubl-inst-")) / "Дубль тест"
        work.mkdir(parents=True)
        try:
            CHECKS[f.suffix](f, work)
        finally:
            shutil.rmtree(work.parent, ignore_errors=True)
    print(f"\nУстановщики проверены: {', '.join(f.name for f in found)}")
    return 0


if __name__ == "__main__":
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
        except Exception:  # noqa: BLE001
            pass
    sys.exit(main())
