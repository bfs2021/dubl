"""Прогон ФУНКЦИИ ПРИЛОЖЕНИЯ (dubl_engine.analysis.transcribe.transcribe — её же зовёт analyze_project) на wav бенчмарка
→ гипотеза в формате hyp/ для score.py.

  DUBL_DATA_DIR=<папка с models/> python run_app.py <wav> <имя> [--engine auto|whisper|gigaam] [--language ru|en|auto]
                [--track eng] [--hyp-dir hyp] [--no-fill] [--gaps]

Язык по умолчанию — auto (определить по речи, как при «Определить сам»); --track — метка дорожки (подсказка).
Модели: Whisper — <DUBL_DATA_DIR>/models/mlx (или whisper), GigaAM — <DUBL_DATA_DIR>/models/gigaam (скачается сам, ~230 МБ).
"""
from __future__ import annotations

import argparse
import os
import json
import resource
import sys
import time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("wav")
ap.add_argument("name")
ap.add_argument("--engine", default="auto")
ap.add_argument("--language", default="auto")
ap.add_argument("--track", default=None)
ap.add_argument("--model", default="large-v3-turbo")
ap.add_argument("--hyp-dir", default=os.environ.get("DUBL_BENCH_HYP", str(Path(__file__).resolve().parent / "hyp")))
ap.add_argument("--no-fill", action="store_true", help="без добора Whisper'ом")
ap.add_argument("--gaps", action="store_true", help="добирать и громкие места без слов GigaAM")
ap.add_argument("--doubt-p", type=float, default=None, help="порог уверенности GigaAM для добора (FILL_DOUBT_P)")
ap.add_argument("--ctx", type=float, default=None, help="контекст окна добора, с (FILL_CTX_S)")
args = ap.parse_args()

import soundfile as sf

from dubl_engine.analysis import speech
from dubl_engine.analysis.transcribe import transcribe
from dubl_engine.pipeline.analyze import iso1

if args.no_fill:
    speech.FILL_DOUBT = False
if args.gaps:
    speech.FILL_GAPS = True
if args.doubt_p is not None:
    speech.FILL_DOUBT_P = args.doubt_p
if args.ctx is not None:
    speech.FILL_CTX_S = args.ctx
info: dict = {}
notes: list[str] = []
t = time.perf_counter()
lines, lang = transcribe(args.wav, language=None if args.language == "auto" else args.language, model=args.model,
                         engine=args.engine, track_language=iso1(args.track), on_note=notes.append, info=info)
dt = round(time.perf_counter() - t, 2)
segs = [{"start": ln.start, "end": ln.end, "text": ln.text,
         "words": [{"start": w.start, "end": w.end, "text": w.text.strip(), "prob": w.prob} for w in ln.words]} for ln in lines]
out = Path(args.hyp_dir) / f"hyp_{args.name}_{Path(args.wav).stem}.json"
out.parent.mkdir(parents=True, exist_ok=True)
rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
out.write_text(json.dumps({
    "model": args.name, "input": args.wav, "lang": lang, "device": "app", "load_s": None, "asr_s": dt,
    "audio_s": round(sf.info(args.wav).duration, 2), "info": info, "notes": notes,
    "chunking": "приложение: analysis/speech.recognize",
    "peak_rss_mb": round(rss / 1e6 if sys.platform == "darwin" else rss / 1e3, 1), "segments": segs},
    ensure_ascii=False, indent=1), encoding="utf-8")
print("saved", out, "segments", len(segs), "s", dt, "lang", lang, "info", json.dumps(info, ensure_ascii=False))
