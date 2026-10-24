"""Полный analyze_project на видео (как кнопка «Начать разбор», язык «Определить по речи»): реплики, язык, пояснения.

  DUBL_DATA_DIR=<папка> python run_analyze.py <видео> [--language ru] [--engine auto|whisper|gigaam] [--out итог.json]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("video")
ap.add_argument("--language", default=None)
ap.add_argument("--engine", default="auto")
ap.add_argument("--out", default=None)
args = ap.parse_args()

from dubl_engine.jobs.manager import NullContext
from dubl_engine.media.probe import probe
from dubl_engine.pipeline.analyze import analyze_project
from dubl_engine.settings import AppSettings
from dubl_engine.store import ProjectStore
from dubl_engine.util import data_dir


class Ctx(NullContext):
    def note(self, key, text):
        print(f"[шаг {key}] {text}", flush=True)


store = ProjectStore(data_dir() / "projects")
src = Path(args.video)
p = store.create(src, probe(src), title=src.stem)
t = time.perf_counter()
an = analyze_project(store, p.id, Ctx(), settings=AppSettings(asr_engine=args.engine), language=args.language)
dt = round(time.perf_counter() - t, 1)
res = {"seconds": dt, "language": an.language, "language_source": an.language_source, "track_language": an.track_language,
       "language_note": an.language_note, "asr_engine": an.asr_engine, "asr_notes": an.asr_notes,
       "speakers": len(an.speakers), "lines": [(round(ln.start, 2), round(ln.end, 2), ln.speaker_id, ln.text) for ln in an.lines]}
print(json.dumps({k: v for k, v in res.items() if k != "lines"}, ensure_ascii=False, indent=1))
for s, e, sp, text in res["lines"]:
    print(f"{s:7.2f}–{e:6.2f} {sp}: {text}")
if args.out:
    Path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
