#!/usr/bin/env python3
"""Скачивает тестовые фильмы Blender Foundation (CC-BY) в samples/raw и проверяет целостность.

    python3 samples/fetch.py            # всё по списку (≈ 2,1 ГБ)
    python3 samples/fetch.py --check    # только проверить, что лежит на диске
    python3 samples/fetch.py tos        # только Tears of Steel (tos | sintel | ed | cosmos)

Докачка — через HTTP Range (если файл уже частично скачан). Размер сверяется с Content-Length сервера,
у аудио/видео — ещё ffprobe (длительность). Только стандартная библиотека.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

RAW = Path(__file__).resolve().parent / "raw"
BASE = "https://download.blender.org/"
ARCHIVE = "https://archive.org/download/CosmosLaundromatFirstCycle/"

# (группа, путь на сервере, имя файла на диске, ожидаемая длительность в секундах или None, допуск)
FILES = [
    ("tos", "demo/movies/ToS/tears_of_steel_720p.mov", "tears_of_steel_720p.mov", 734.17, 1.0),
    ("tos", "demo/movies/ToS/subtitles/TOS-en.srt", "TOS-en.srt", None, 0),
    ("tos", "demo/movies/ToS/subtitles/TOS-ru.srt", "TOS-ru.srt", None, 0),
    ("tos", "demo/movies/ToS/Surround-TOS_DVDSURROUND-Dolby%205.1.ac3", "Surround-TOS_DVDSURROUND-Dolby%205.1.ac3", 734.0, 1.0),
    ("tos", "demo/movies/ToS/TOS_MUSIC%2BFX_NO_DIALOGUE.aif", "TOS_MUSIC%2BFX_NO_DIALOGUE.aif", 734.0, 1.0),
    ("sintel", "durian/movies/Sintel.2010.720p.mkv.zip", "Sintel.2010.720p.mkv.zip", None, 0),
    ("sintel", "durian/movies/sintel-m%2Be-st.flac", "sintel-m%2Be-st.flac", 888.0, 1.0),
    ("ed", "ED/elephantsdream-720-h264-st-aac.mov", "elephantsdream-720-h264-st-aac.mov", 658.3, 1.0),
    ("ed", "ED/ED-ME-5.1-DVD.ac3", "ED-ME-5.1-DVD.ac3", 661.0, 1.0),
    # Cosmos Laundromat: First Cycle (CC-BY 4.0) — диалог двух персонажей (овца Franck и продавец Victor); 220 МБ, берём с archive.org
    ("cosmos", ARCHIVE + "Cosmos%20Laundromat%20-%20First%20Cycle%20%281080p%29.mp4", "cosmos_laundromat_1080p.mp4", 730.6, 1.0),
    ("cosmos", ARCHIVE + "CosmosLaundromat-FirstCycle1080p.en.srt", "cosmos_en.srt", None, 0),
]
UA = {"User-Agent": "curl/8.7"}   # сервер отдаёт 403 на python-urllib по умолчанию
UNZIP = {"Sintel.2010.720p.mkv.zip": ("Sintel.2010.720p.mkv", 888.03)}


def remote_size(url: str) -> int | None:
    try:
        req = urllib.request.Request(url, method="HEAD", headers=UA)
        with urllib.request.urlopen(req, timeout=30) as r:
            return int(r.headers.get("Content-Length") or 0) or None
    except Exception as e:  # noqa: BLE001
        print(f"   ! HEAD не удался: {e}")
        return None


def download(url: str, dest: Path, total: int | None) -> None:
    have = dest.stat().st_size if dest.exists() else 0
    if total and have == total:
        return
    if total and have > total:
        dest.unlink()
        have = 0
    req = urllib.request.Request(url, headers=UA)
    if have:
        req.add_header("Range", f"bytes={have}-")
    print(f"   качаю {'с ' + str(have) + ' байт ' if have else ''}…")
    with urllib.request.urlopen(req, timeout=60) as r:
        if have and r.status != 206:
            have = 0  # сервер не умеет докачку — начинаем сначала
        with open(dest, "ab" if have else "wb") as f:
            shutil.copyfileobj(r, f, 1 << 20)


def duration(path: Path) -> float | None:
    if not shutil.which("ffprobe"):
        return None
    p = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
                       capture_output=True, text=True)
    try:
        return float(p.stdout.strip())
    except ValueError:
        return None


def verify(name: str, path: Path, total: int | None, dur: float | None, tol: float) -> bool:
    if not path.exists():
        print(f"   ✗ {name}: нет файла")
        return False
    size = path.stat().st_size
    if total and size != total:
        print(f"   ✗ {name}: размер {size} ≠ {total} на сервере")
        return False
    if dur is not None:
        d = duration(path)
        if d is None or abs(d - dur) > tol:
            print(f"   ✗ {name}: длительность {d} вместо ≈{dur}")
            return False
    print(f"   ✓ {name}: {size / 1e6:.1f} МБ" + (f", {duration(path):.1f} с" if dur is not None else ""))
    return True


def main() -> int:
    args = sys.argv[1:]
    check_only = "--check" in args
    groups = [a for a in args if not a.startswith("--")]
    RAW.mkdir(parents=True, exist_ok=True)
    ok = True
    for group, rel, name, dur, tol in FILES:
        if groups and group not in groups:
            continue
        url = rel if rel.startswith("http") else BASE + rel
        dest = RAW / name
        print(f"{name}")
        total = remote_size(url)
        if not check_only:
            try:
                download(url, dest, total)
            except Exception as e:  # noqa: BLE001
                print(f"   ✗ ошибка скачивания: {e}")
        ok &= verify(name, dest, total, dur, tol)
        if name in UNZIP and dest.exists():
            inner, idur = UNZIP[name]
            target = RAW / inner
            if not target.exists() and not check_only:
                print(f"   распаковываю {inner}")
                with zipfile.ZipFile(dest) as z:
                    z.extract(inner, RAW)
            ok &= verify(inner, target, None, idur, 1.0)
    print("\nВсё на месте." if ok else "\nЕсть проблемы — см. ✗ выше.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
