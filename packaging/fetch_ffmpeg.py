#!/usr/bin/env python3
"""Скачать статические ffmpeg + ffprobe под нужную платформу и положить в папку ресурсов Tauri.

  python packaging/fetch_ffmpeg.py                      # для текущей машины → packaging/build/stage/resources/ffmpeg
  python packaging/fetch_ffmpeg.py --target x86_64-pc-windows-msvc --out app/src-tauri/resources/ffmpeg

Источники (все — GPL-сборки со статической линковкой, в них есть libx264 и aac, нужные рендеру):
  macOS arm64/x64   ffmpeg.martin-riedl.de   (release; sha256 лежит рядом с файлом)
  Windows x64       gyan.dev                 (release-essentials; sha256 рядом)
  Windows arm64     BtbN/FFmpeg-Builds       (GitHub, checksums.sha256)
  Linux x64/arm64   johnvansickle.com        (release static; рядом только md5 — проверяет целостность, не подлинность)

Что делает: скачивает (кэш в packaging/build/dl), сверяет контрольную сумму, распаковывает только ffmpeg и ffprobe,
для текущей ОС запускает их (`-version`) и проверяет версию >= 6 и наличие кодеков libx264 / aac / pcm_f32le,
пишет VERSION.json и NOTICE-ffmpeg.txt (лицензия GPL + где взять исходники).
Движок находит папку по переменной DUBL_FFMPEG_DIR (util.find_binary); её выставляет Rust-лаунчер.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
DL = HERE / "build" / "dl"
MIN_MAJOR = 6
REQUIRED_ENCODERS = ("libx264", "aac", "pcm_f32le")
UA = {"User-Agent": "dubl-packaging/1.0 (+https://github.com)"}

TARGETS = (
    "aarch64-apple-darwin", "x86_64-apple-darwin",
    "x86_64-pc-windows-msvc", "aarch64-pc-windows-msvc",
    "x86_64-unknown-linux-gnu", "aarch64-unknown-linux-gnu",
)


def host_target() -> str:
    m = platform.machine().lower()
    arch = "aarch64" if m in ("arm64", "aarch64") else "x86_64"
    if sys.platform == "darwin":
        return f"{arch}-apple-darwin"
    if sys.platform == "win32":
        return f"{arch}-pc-windows-msvc"
    return f"{arch}-unknown-linux-gnu"


def http_get(url: str, *, binary: bool = False, timeout: int = 60):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = r.read()
        return (data if binary else data.decode("utf-8", "replace")), r.geturl()


def download(url: str, dest: Path) -> Path:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  кэш: {dest.name}")
        return dest
    DL.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"  скачиваю {url}")
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        got = 0
        while chunk := r.read(1 << 20):
            f.write(chunk)
            got += len(chunk)
            if total and got % (20 << 20) < (1 << 20):
                print(f"    {got / 1e6:.0f}/{total / 1e6:.0f} МБ", flush=True)
    tmp.replace(dest)
    return dest


def digest(path: Path, algo: str) -> str:
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def first_hex(text: str, n: int) -> str:
    m = re.search(rf"\b[0-9a-fA-F]{{{n}}}\b", text)
    if not m:
        raise RuntimeError(f"не нашёл контрольную сумму в ответе: {text[:80]!r}")
    return m.group(0).lower()


def verify(path: Path, algo: str, expected: str) -> None:
    got = digest(path, algo)
    if got != expected.lower():
        path.unlink(missing_ok=True)
        raise RuntimeError(f"{path.name}: {algo} не совпал (ожидалось {expected[:12]}…, получено {got[:12]}…)")
    print(f"  {algo} ок: {got[:16]}…")


# ───────────── источники: каждая функция возвращает список файлов (url, имя, algo, sum) и «сорт» архива ─────────────

def plan_macos(target: str) -> dict:
    arch = "arm64" if target.startswith("aarch64") else "amd64"
    parts = []
    for name in ("ffmpeg", "ffprobe"):
        _, final = http_get(f"https://ffmpeg.martin-riedl.de/redirect/latest/macos/{arch}/release/{name}.zip", binary=True)
        sha, _ = http_get(final + ".sha256")
        parts.append({"url": final, "algo": "sha256", "sum": first_hex(sha, 64), "kind": "zip", "wants": [name]})
    return {"source": "ffmpeg.martin-riedl.de", "parts": parts,
            "source_code": "https://ffmpeg.martin-riedl.de/ (скрипты сборки) и https://ffmpeg.org/download.html#releases"}


def plan_windows(target: str) -> dict:
    if target.startswith("aarch64"):
        api = json.loads(http_get("https://api.github.com/repos/BtbN/FFmpeg-Builds/releases/latest")[0])
        assets = {a["name"]: a["browser_download_url"] for a in api["assets"]}
        name = next(n for n in assets if re.fullmatch(r"ffmpeg-n\d+\.\d+-latest-winarm64-gpl-\d+\.\d+\.zip", n))
        sums = http_get(assets["checksums.sha256"])[0]
        want = next(line.split()[0] for line in sums.splitlines() if line.strip().endswith(name))
        return {"source": "github.com/BtbN/FFmpeg-Builds",
                "parts": [{"url": assets[name], "algo": "sha256", "sum": want, "kind": "zip", "wants": ["ffmpeg", "ffprobe"]}],
                "source_code": "https://github.com/BtbN/FFmpeg-Builds"}
    base = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"
    sha, _ = http_get(base + ".sha256")
    return {"source": "gyan.dev (release-essentials)",
            "parts": [{"url": base, "algo": "sha256", "sum": first_hex(sha, 64), "kind": "zip", "wants": ["ffmpeg", "ffprobe"]}],
            "source_code": "https://www.gyan.dev/ffmpeg/builds/ и https://ffmpeg.org/download.html#releases"}


def plan_linux(target: str) -> dict:
    arch = "arm64" if target.startswith("aarch64") else "amd64"
    base = f"https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-{arch}-static.tar.xz"
    md5, _ = http_get(base + ".md5")
    return {"source": "johnvansickle.com (release static)",
            "parts": [{"url": base, "algo": "md5", "sum": first_hex(md5, 32), "kind": "tar", "wants": ["ffmpeg", "ffprobe"]}],
            "source_code": "https://johnvansickle.com/ffmpeg/ (в архиве — GPLv3.txt; исходники: https://ffmpeg.org/download.html#releases)"}


def make_plan(target: str) -> dict:
    if "apple-darwin" in target:
        return plan_macos(target)
    if "windows" in target:
        return plan_windows(target)
    return plan_linux(target)


def extract(archive: Path, kind: str, wants: list[str], exe_suffix: str, dest: Path) -> None:
    names = {w + exe_suffix: w for w in wants}
    found: set[str] = set()
    if kind == "zip":
        with zipfile.ZipFile(archive) as z:
            for info in z.infolist():
                base = Path(info.filename).name
                if base in names and not info.is_dir():
                    with z.open(info) as src, open(dest / base, "wb") as out:
                        shutil.copyfileobj(src, out)
                    found.add(base)
    else:
        with tarfile.open(archive) as t:
            for m in t.getmembers():
                base = Path(m.name).name
                if base in names and m.isfile():
                    f = t.extractfile(m)
                    assert f is not None
                    with open(dest / base, "wb") as out:
                        shutil.copyfileobj(f, out)
                    found.add(base)
    missing = set(names) - found
    if missing:
        raise RuntimeError(f"в архиве {archive.name} нет файлов: {', '.join(sorted(missing))}")


def probe(exe: Path) -> dict:
    out = subprocess.run([str(exe), "-hide_banner", "-version"], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60)
    if out.returncode != 0:
        raise RuntimeError(f"{exe.name} не запускается: {out.stderr.strip()[:300]}")
    first = out.stdout.splitlines()[0]
    m = re.search(r"version\s+(?:n)?(\d+)\.(\d+)(?:\.(\d+))?", first)
    if not m:
        raise RuntimeError(f"не разобрал версию: {first!r}")
    info = {"version_line": first, "major": int(m.group(1)), "version": ".".join(g for g in m.groups() if g)}
    info["gpl"] = "--enable-gpl" in out.stdout
    return info


def check_ffmpeg(ffmpeg: Path, ffprobe: Path) -> dict:
    ff = probe(ffmpeg)
    fp = probe(ffprobe)
    if ff["major"] < MIN_MAJOR or fp["major"] < MIN_MAJOR:
        raise RuntimeError(f"нужен ffmpeg >= {MIN_MAJOR}, получено {ff['version']}")
    enc = subprocess.run([str(ffmpeg), "-hide_banner", "-encoders"], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60).stdout
    missing = [e for e in REQUIRED_ENCODERS if not re.search(rf"\s{re.escape(e)}\s", enc)]
    if missing:
        raise RuntimeError(f"в этой сборке ffmpeg нет кодеков: {', '.join(missing)} (рендер не заработает)")
    return {"ffmpeg": ff["version"], "ffprobe": fp["version"], "gpl": ff["gpl"]}


NOTICE = """ffmpeg и ffprobe в этой папке — отдельные программы (проект FFmpeg, https://ffmpeg.org).
Сборка: {source}. Версия: {version}. Платформа: {target}.

Лицензия: GNU GPL v2 или новее (сборка включает libx264 и другие GPL-компоненты); часть кода — LGPL.
«Дубль» запускает ffmpeg/ffprobe как отдельные процессы и не линкуется с ними.
Текст лицензии: https://www.gnu.org/licenses/gpl-3.0.txt
Исходные коды и скрипты сборки: {source_code}
Исходники именно этой сборки можно взять у её автора по ссылке выше; если не получается — откройте обращение: https://github.com/pingvin1337/dubl/issues
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", default=host_target(), choices=TARGETS, help="целевая платформа (по умолчанию — эта машина)")
    ap.add_argument("--out", default=str(HERE / "build" / "stage" / "resources" / "ffmpeg"))
    ap.add_argument("--no-check", action="store_true", help="не запускать скачанный ffmpeg (кросс-загрузка)")
    args = ap.parse_args()

    target, out = args.target, Path(args.out).resolve()
    exe_suffix = ".exe" if "windows" in target else ""
    print(f"ffmpeg для {target} → {out}")
    plan = make_plan(target)
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)

    for part in plan["parts"]:
        # в имени кэша — контрольная сумма: «latest»-ссылки меняют содержимое
        arch_file = DL / (part["sum"][:12] + "-" + Path(part["url"]).name)
        download(part["url"], arch_file)
        verify(arch_file, part["algo"], part["sum"])
        part["sha"] = part["sum"]
        extract(arch_file, part["kind"], part["wants"], exe_suffix, out)

    ffmpeg, ffprobe = out / f"ffmpeg{exe_suffix}", out / f"ffprobe{exe_suffix}"
    for p in (ffmpeg, ffprobe):
        p.chmod(p.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    can_run = host_target() == target and not args.no_check
    info = check_ffmpeg(ffmpeg, ffprobe) if can_run else {"ffmpeg": "не проверялось (другая платформа)"}
    if can_run:
        print(f"  проверено: ffmpeg {info['ffmpeg']}, ffprobe {info['ffprobe']}, кодеки {', '.join(REQUIRED_ENCODERS)} на месте")
        if not info["gpl"]:
            print("  внимание: сборка без --enable-gpl, лицензионная заметка может быть неточной", file=sys.stderr)

    version = info.get("ffmpeg", "?")
    (out / "NOTICE-ffmpeg.txt").write_text(
        NOTICE.format(source=plan["source"], version=version, target=target, source_code=plan["source_code"]), "utf-8")
    (out / "VERSION.json").write_text(json.dumps({
        "target": target, "source": plan["source"], "version": version,
        "files": [{"url": p["url"], p["algo"]: p["sum"]} for p in plan["parts"]],
        "sizes_mb": {f.name: round(f.stat().st_size / 1e6, 1) for f in (ffmpeg, ffprobe)},
    }, ensure_ascii=False, indent=2), "utf-8")
    total = sum(f.stat().st_size for f in out.iterdir())
    print(f"Готово: {out} ({total / 1e6:.0f} МБ)")


if __name__ == "__main__":
    for _s in (sys.stdout, sys.stderr):   # Windows: консоль/пайп в cp1252 — русский print иначе падает (UnicodeEncodeError)
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
        except Exception:  # noqa: BLE001
            pass
    try:
        main()
    except Exception as e:  # noqa: BLE001
        print(f"Ошибка: {e}", file=sys.stderr)
        sys.exit(1)
