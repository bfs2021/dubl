#!/usr/bin/env python3
"""Подпись всех Mach-O внутри resources/engine (и ffmpeg) под hardened runtime — нужна для нотаризации.

Tauri подписывает .app и externalBin (лаунчер), но НЕ заходит в папки из bundle.resources. Нотаризация отклонит
.app, если внутри есть неподписанный/без hardened runtime бинарник, поэтому подписываем до `tauri build`:

  python packaging/sign_macos.py --identity "Developer ID Application: Имя (TEAMID)" --root app/src-tauri/resources
  python packaging/sign_macos.py --identity -  --root packaging/build/stage/resources      # ad-hoc, для проверки

Идентичность берётся из --identity или APPLE_SIGNING_IDENTITY.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MACHO_MAGIC = {b"\xcf\xfa\xed\xfe", b"\xce\xfa\xed\xfe", b"\xca\xfe\xba\xbe", b"\xbe\xba\xfe\xca"}


def is_macho(p: Path) -> bool:
    if p.is_symlink() or not p.is_file():
        return False
    try:
        with open(p, "rb") as f:
            return f.read(4) in MACHO_MAGIC
    except OSError:
        return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--identity", default=os.environ.get("APPLE_SIGNING_IDENTITY"))
    ap.add_argument("--root", required=True, help="папка resources (содержит engine/ и ffmpeg/)")
    ap.add_argument("--entitlements", default=str(HERE / "engine.entitlements"))
    args = ap.parse_args()
    if sys.platform != "darwin":
        sys.exit("только macOS")
    if not args.identity:
        sys.exit("укажите --identity или APPLE_SIGNING_IDENTITY")
    root = Path(args.root).resolve()
    files = sorted(p for p in root.rglob("*") if is_macho(p))
    # сначала вложенное (глубже), потом исполняемые — порядок не критичен для плоских файлов, но так надёжнее
    files.sort(key=lambda p: -len(p.parts))
    print(f"Подписываю {len(files)} файлов в {root}")
    for p in files:
        cmd = ["codesign", "--force", "--options", "runtime", "--timestamp" if args.identity != "-" else "--timestamp=none",
               "--entitlements", args.entitlements, "--sign", args.identity, str(p)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            print(r.stderr, file=sys.stderr)
            sys.exit(f"не удалось подписать {p}")
    bad = [p for p in files if subprocess.run(["codesign", "--verify", "--strict", str(p)], capture_output=True).returncode]
    if bad:
        sys.exit("проверка подписи не прошла: " + ", ".join(map(str, bad[:5])))
    print("Готово, подписи проверены.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
