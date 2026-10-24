"""Общие фикстуры: синтетические видео, сгенерированные ffmpeg (без внешних файлов)."""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _isolated_data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("DUBL_DATA_DIR", str(tmp_path / "data"))


def _ff(*args):
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *map(str, args)], check=True)


@pytest.fixture(scope="session")
def media_dir(tmp_path_factory) -> Path:
    return tmp_path_factory.mktemp("media")


@pytest.fixture(scope="session")
def simple_mp4(media_dir) -> Path:
    """10 с, 640x360, 25 fps, стерео: тон 220 Гц (фон) + «речь» (бипы 600 Гц) на 2–4 с и 6–8 с."""
    out = media_dir / "simple.mp4"
    if not out.exists():
        _ff("-f", "lavfi", "-i", "testsrc2=s=640x360:r=25:d=10",
            "-f", "lavfi", "-i", "sine=f=220:d=10,volume=0.1",
            "-f", "lavfi", "-i", "sine=f=600:d=10,volume='between(t,2,4)+between(t,6,8)':eval=frame",
            "-filter_complex", "[1:a][2:a]amix=inputs=2:normalize=0,aformat=channel_layouts=stereo[a]",
            "-map", "0:v", "-map", "[a]", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", out)
    return out


@pytest.fixture(scope="session")
def rich_mkv(media_dir) -> Path:
    """MKV как у «релизов»: 2 аудио (eng 5.1 + rus stereo), субтитры ASS с именами и SRT, 2 главы, склейка на 5 с."""
    out = media_dir / "rich.mkv"
    if out.exists():
        return out
    ass = media_dir / "subs.ass"
    ass.write_text(
        "[Script Info]\nScriptType: v4.00+\nPlayResX: 640\nPlayResY: 360\n\n"
        "[V4+ Styles]\nFormat: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, "
        "Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding\n"
        "Style: Default,Arial,20,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,2,0,2,10,10,10,1\n\n"
        "[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
        "Dialogue: 0,0:00:02.00,0:00:04.00,Default,Barsik,0,0,0,,Who touched my sausage?\n"
        "Dialogue: 0,0:00:06.00,0:00:08.00,Default,Mila,0,0,0,,{\\i1}Not me.{\\i0} I was asleep.\n",
        encoding="utf-8")
    srt = media_dir / "subs.srt"
    srt.write_text("1\n00:00:02,000 --> 00:00:04,000\nКто трогал мою колбасу?\n\n"
                   "2\n00:00:06,000 --> 00:00:08,000\nНе я. Я спал.\n", encoding="utf-8")
    meta = media_dir / "chapters.txt"
    meta.write_text(";FFMETADATA1\n[CHAPTER]\nTIMEBASE=1/1000\nSTART=0\nEND=5000\ntitle=Начало\n"
                    "[CHAPTER]\nTIMEBASE=1/1000\nSTART=5000\nEND=10000\ntitle=Кухня\n", encoding="utf-8")
    # видео: 0–5 с красный, 5–10 с синий => одна склейка на 5.0
    _ff("-f", "lavfi", "-i", "color=c=red:s=640x360:r=25:d=5", "-f", "lavfi", "-i", "color=c=blue:s=640x360:r=25:d=5",
        "-f", "lavfi", "-i", "sine=f=600:d=10,volume='between(t,2,4)+between(t,6,8)':eval=frame",  # «диалог»
        "-f", "lavfi", "-i", "sine=f=150:d=10,volume=0.2",                                         # «музыка»
        "-i", ass, "-i", srt, "-i", meta,
        "-filter_complex",
        "[0:v][1:v]concat=n=2:v=1:a=0[v];"
        # 5.1: FL FR = музыка, FC = диалог, LFE = тихо, BL BR = музыка
        "[2:a]asplit=2[d1][d2];[3:a]asplit=6[m1][m2][m3][m4][m5][m6];[m3]volume=0.2[lfe];"
        "[m1][m2][d1][lfe][m4][m5]join=inputs=6:channel_layout=5.1:map=0.0-FL|1.0-FR|2.0-FC|3.0-LFE|4.0-BL|5.0-BR[a51];"
        "[d2][m6]amix=inputs=2:normalize=0,aformat=channel_layouts=stereo[ast]",
        "-map", "[v]", "-map", "[a51]", "-map", "[ast]", "-map", "4:s", "-map", "5:s", "-map_metadata", "6", "-map_chapters", "6",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-c:s:0", "ass", "-c:s:1", "srt",
        "-metadata:s:a:0", "language=eng", "-metadata:s:a:0", "title=English 5.1",
        "-metadata:s:a:1", "language=rus", "-metadata:s:a:1", "title=Русский дубляж",
        "-metadata:s:s:0", "language=eng", "-metadata:s:s:1", "language=rus",
        "-disposition:a:0", "default", out)
    return out


# ───────────── готовый проект со сценами (без нейросетей) ─────────────
@pytest.fixture
def ready_project(tmp_path, simple_mp4):
    """(store, pid, [scene_id, …]): проект из simple_mp4 и две подготовленные сцены (0–4 с и 5–9 с).
    Режим «закадровый» (оригинал слышен), голоса и фон записаны сразу («речь»: сцена 1 — тон 600 Гц, сцена 2 — 1000 Гц; фон 220 Гц), clip.mp4 — настоящая вырезка."""
    import numpy as np

    from dubl_engine.audio.io import write_wav
    from dubl_engine.media.extract import cut_scene_video
    from dubl_engine.media.probe import probe
    from dubl_engine.models import MixMode, MixSettings, Scene
    from dubl_engine.store import ProjectStore

    store = ProjectStore(tmp_path / "projects")
    project = store.create(simple_mp4, probe(simple_mp4), title="склейка")
    pid = project.id
    ids = []
    for n, (a, b) in enumerate([(0.0, 4.0), (5.0, 9.0)], 1):
        scid = f"sc{n:03d}"
        sd = store.scene_dir(pid, scid)
        sd.mkdir(parents=True, exist_ok=True)
        cut_scene_video(simple_mp4, a, b, sd / "clip.mp4", audio_stream_index=1)
        sr, dur = 48000, b - a
        t = np.arange(int(dur * sr)) / sr
        voice = (0.2 * np.sin(2 * np.pi * (600 if n == 1 else 1000) * t) * ((t % 2.0) < 1.0)).astype(np.float32)
        bg = (0.05 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
        write_wav(sd / "voices.wav", voice, sr)
        write_wav(sd / "background.wav", np.stack([bg, bg], axis=1), sr)
        write_wav(sd / "original.wav", np.stack([voice + bg, voice + bg], axis=1), sr)
        (sd / "voices").mkdir(exist_ok=True)
        with store.edit(pid) as p:
            p.scenes.append(Scene(id=scid, title=f"Сцена {n}", start=a, end=b, stems_ready=True,
                                  mix=MixSettings(mode=MixMode.VOICEOVER, autopilot=False, original_voices_db=0.0)))
        ids.append(scid)
    return store, pid, ids
