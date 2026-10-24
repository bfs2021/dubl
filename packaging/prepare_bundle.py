#!/usr/bin/env python3
"""Подготовка app/src-tauri к сборке бандла: кладёт sidecar, ресурсы движка и ffmpeg туда, где их ждёт Tauri.

  python packaging/prepare_bundle.py                 # собрать движок + скачать ffmpeg → app/src-tauri/{binaries,resources}
  python packaging/prepare_bundle.py --tauri-dir /tmp/copy/src-tauri
  python packaging/prepare_bundle.py --sign-macos --identity "Developer ID Application: …"   # подпись ресурсов до tauri build

Что нужно в tauri.conf.json (см. docs/BUILD.md):
  "bundle": { "externalBin": ["binaries/dubl-engine"],
              "resources": { "resources/engine/": "engine/", "resources/ffmpeg/": "ffmpeg/" } }
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def run(*cmd) -> None:
    print("$", " ".join(map(str, cmd)), flush=True)
    subprocess.run([str(c) for c in cmd], check=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tauri-dir", default=str(ROOT / "app" / "src-tauri"))
    ap.add_argument("--target", help="целевая тройка для ffmpeg (по умолчанию — эта машина)")
    ap.add_argument("--sign-macos", action="store_true")
    ap.add_argument("--identity")
    ap.add_argument("--skip-engine", action="store_true", help="не пересобирать движок (только ffmpeg/подпись)")
    args = ap.parse_args()
    out = Path(args.tauri_dir).resolve()
    if not args.skip_engine:
        run(sys.executable, HERE / "build_engine.py", "--out", out)
    ff = [sys.executable, HERE / "fetch_ffmpeg.py", "--out", out / "resources" / "ffmpeg"]
    if args.target:
        ff += ["--target", args.target]
    ff_dir = out / "resources" / "ffmpeg"
    have = (ff_dir / "VERSION.json").is_file() and any(ff_dir.glob("ffmpeg*")) and not args.target
    try:
        run(*ff)
    except subprocess.CalledProcessError:
        if not have:
            raise
        # сайт со сборками ffmpeg бывает недоступен — для локальной сборки годится уже скачанный
        print(f"Не удалось обновить ffmpeg, беру уже скачанный: {ff_dir}", flush=True)
    if args.sign_macos:
        if not args.identity:
            sys.exit("--sign-macos требует --identity")
        run(sys.executable, HERE / "sign_macos.py", "--identity", args.identity, "--root", out / "resources")
    print(f"\nГотово: {out}/binaries, {out}/resources. Дальше: cd app && pnpm tauri build")


if __name__ == "__main__":
    main()
