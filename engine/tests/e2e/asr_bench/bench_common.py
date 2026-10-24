"""Общие функции бенчмарка ASR (engine/tests/e2e/asr_bench). Импортируется из run_*.py.

Формат гипотезы hyp_<model>_<input>.json:
{"model": str, "input": path, "lang": str|null, "device": str, "load_s": float, "asr_s": float, "audio_s": float,
 "chunking": str, "segments": [{"start","end","text","words":[{"start","end","text"}]}]}
"""
from __future__ import annotations

import json
import os
import re
import resource
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent
import os
HYP = Path(os.environ.get("DUBL_BENCH_HYP", ROOT / "hyp"))
SR = 16000


def load16k(path) -> np.ndarray:
    x, sr = sf.read(str(path), dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    if sr != SR:
        from math import gcd
        from scipy.signal import resample_poly
        g = gcd(sr, SR)
        x = resample_poly(x, SR // g, sr // g).astype(np.float32)
    return np.ascontiguousarray(x, dtype=np.float32)


def env_db(x: np.ndarray, hop: float = 0.05) -> np.ndarray:
    h = int(SR * hop)
    n = len(x) // h
    p = (x[: n * h].reshape(n, h) ** 2).mean(axis=1)
    return 10 * np.log10(p + 1e-12)


def energy_chunks(x: np.ndarray, max_s: float = 20.0, search_s: float = 6.0, win: float = 0.4) -> list[tuple[float, float]]:
    """Режет весь файл без потерь на куски ≤ max_s по самым тихим местам (окно win) в последних search_s секундах куска."""
    hop = 0.05
    db = env_db(x, hop)
    p = 10 ** (db / 10)
    w = max(1, int(win / hop))
    dur = len(x) / SR
    out, a = [], 0.0
    while dur - a > max_s:
        lo, hi = int((a + max_s - search_s) / hop), int((a + max_s) / hop) - w
        sums = np.convolve(p[lo:hi + w], np.ones(w), mode="valid")
        k = int(np.argmin(sums))
        cut = (lo + k + w / 2) * hop
        out.append((a, cut))
        a = cut
    out.append((a, dur))
    return out


def silero_chunks(x: np.ndarray, max_s: float = 20.0, pad: float = 0.2, threshold: float = 0.5,
                  min_silence_ms: int = 300) -> list[tuple[float, float]]:
    """silero-vad (штатные пороги) → интервалы речи с полями pad, склеенные в куски ≤ max_s. Всё вне речи выбрасывается."""
    import torch
    from silero_vad import get_speech_timestamps, load_silero_vad
    m = load_silero_vad()
    ts = get_speech_timestamps(torch.from_numpy(x), m, sampling_rate=SR, threshold=threshold,
                               min_silence_duration_ms=min_silence_ms, max_speech_duration_s=max_s - 2 * pad,
                               return_seconds=True)
    dur = len(x) / SR
    iv = [(max(0.0, t["start"] - pad), min(dur, t["end"] + pad)) for t in ts]
    out: list[list[float]] = []
    for a, b in iv:
        if out and b - out[-1][0] <= max_s and a - out[-1][1] < 1.0:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


_END = re.compile(r"[.!?…]$")


def words_to_lines(words: list[dict], gap: float = 0.5, max_len: float = 8.0) -> list[dict]:
    """Единая нарезка слов на реплики для моделей без своих сегментов: пауза ≥ gap, или конец предложения + пауза ≥ 0.15,
    или реплика длиннее max_len (режем по самой большой паузе)."""
    lines, cur = [], []
    for w in words:
        if cur:
            g = w["start"] - cur[-1]["end"]
            dash = w["text"].lstrip().startswith(("—", "–"))      # GigaAM e2e ставит тире на смене говорящего
            if g >= gap or dash or (_END.search(cur[-1]["text"].strip()) and g >= 0.15):
                lines.append(cur)
                cur = []
        cur.append(w)
    if cur:
        lines.append(cur)
    out = []

    def split(ws):
        if ws[-1]["end"] - ws[0]["start"] <= max_len or len(ws) < 2:
            return [ws]
        gi = max(range(1, len(ws)), key=lambda i: ws[i]["start"] - ws[i - 1]["end"])
        return split(ws[:gi]) + split(ws[gi:])

    for ln in lines:
        ln = [w for w in ln if w["text"].strip(" —–")] or ln
        if not any(w["text"].strip(" —–") for w in ln):
            continue
        for ws in split(ln):
            out.append({"start": ws[0]["start"], "end": ws[-1]["end"],
                        "text": " ".join(w["text"].strip() for w in ws).strip(), "words": ws})
    return out


def peak_rss_mb() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 1e6 if sys.platform == "darwin" else r / 1e3


def save(name: str, inp: str, data: dict) -> Path:
    HYP.mkdir(parents=True, exist_ok=True)
    tag = Path(inp).stem
    p = HYP / f"hyp_{name}_{tag}.json"
    data = {"model": name, "input": str(inp), "peak_rss_mb": round(peak_rss_mb(), 1), **data}
    p.write_text(json.dumps(data, ensure_ascii=False, indent=1))
    n = len(data.get("segments", []))
    print(f"saved {p}  segments={n}  asr_s={data.get('asr_s')}  audio_s={data.get('audio_s')}  rss={data['peak_rss_mb']}MB",
          flush=True)
    return p


class Timer:
    def __enter__(self):
        self.t = time.perf_counter()
        return self

    def __exit__(self, *a):
        self.s = round(time.perf_counter() - self.t, 2)
