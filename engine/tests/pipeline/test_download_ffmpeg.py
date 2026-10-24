"""Скачивание по ссылке: yt-dlp должен получать путь к ffmpeg из DUBL_FFMPEG_DIR, а не искать его в PATH.

В собранном приложении ffmpeg лежит внутри Дубль.app и в PATH его нет — без ffmpeg_location ролики с раздельными
потоками видео и звука (почти весь YouTube в HD) падали с «ffmpeg is not installed».
"""
from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from dubl_engine.pipeline import download


def _fake_bin(d: Path, name: str) -> None:
    p = d / (name + (".exe" if os.name == "nt" else ""))
    p.write_text("#!/bin/sh\n")
    p.chmod(p.stat().st_mode | stat.S_IEXEC)


def test_ffmpeg_location_from_dubl_ffmpeg_dir(tmp_path, monkeypatch):
    yt_dlp = pytest.importorskip("yt_dlp")
    bindir = tmp_path / "bundle-ffmpeg"
    bindir.mkdir()
    _fake_bin(bindir, "ffmpeg")
    monkeypatch.setenv("DUBL_FFMPEG_DIR", str(bindir))
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))      # в PATH ffmpeg нет, как у запущенного .app

    seen: dict = {}

    class FakeYDL:
        def __init__(self, opts):
            seen.update(opts)

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def extract_info(self, url, download=True):
            f = tmp_path / "dl" / "abc.mkv"
            f.write_bytes(b"x")
            return {"title": "Ролик", "requested_downloads": [{"filepath": str(f)}]}

    monkeypatch.setattr(yt_dlp, "YoutubeDL", FakeYDL)
    path, title = download.download_video("https://example.com/v", tmp_path / "dl")
    assert seen["ffmpeg_location"] == str(bindir)
    assert path.name == "abc.mkv" and title == "Ролик"


def test_ffmpeg_missing_error_is_explained():
    kind, text = download.classify_ytdlp_error(
        "ERROR: You have requested merging of multiple formats but ffmpeg is not installed. Aborting")
    assert kind == "other"
    assert "ffmpeg" in text and "Переустанови" in text
