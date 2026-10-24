"""Общие помощники для e2e-тестов и замеров: пути к сэмплам, WER, SDR, сборка релизного MKV, «дубли» через say."""
from __future__ import annotations

import difflib
import re
import subprocess
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DUBL = Path(__file__).resolve().parents[3]
SAMPLES = DUBL / "samples"
RAW = SAMPLES / "raw"
WORK = SAMPLES / "work"
OUT = SAMPLES / "out"

TOS_MOV = RAW / "tears_of_steel_720p.mov"
TOS_EN = RAW / "TOS-en.srt"
TOS_RU = RAW / "TOS-ru.srt"
TOS_51 = RAW / "Surround-TOS_DVDSURROUND-Dolby%205.1.ac3"
TOS_MFX = RAW / "TOS_MUSIC%2BFX_NO_DIALOGUE.aif"
SINTEL_MKV = RAW / "Sintel.2010.720p.mkv"
SINTEL_MFX = RAW / "sintel-m%2Be-st.flac"
ED_MOV = RAW / "elephantsdream-720-h264-st-aac.mov"
TOS_RELEASE = WORK / "tears_of_steel_release.mkv"

# Сцена-пример: «Память перезаписывается» (реплики 33–40: «...и, начали!», «Начата перезапись памяти!», «Ты придурок, Том!», «Ой, простите!», «Послушай, Селия…»)
DEMO_SCENE = (291.0, 330.0)

# Ручная разметка «кто говорит» в ToS по сюжету фильма (номер реплики в TOS-en.srt → персонаж).
# Только реплики, в которых уверены. Молодого и старого Тома играют разные актёры, но в симуляции их голоса
# по памяти сюжета не разделить, поэтому в эталоне это один «Том» — число найденных голосов сравниваем с 3–4 героями.
# Другие голоса (Барли, техник, учёные, Дженгис) в эталон не входят. Разметка вручную: точность метрик ±5 п.п.
GT_SPEAKERS: dict[int, str] = {
    **{i: "Селия" for i in (1, 4, 9, 35, 44, 55, 57, 59)},
    **{i: "Том" for i in (2, 3, 5, 6, 7, 8, 37, 38, 39, 40, 45, 46, 47, 56, 58, 60, 61, 71, 72)},
    **{i: "Капитан" for i in (25, 26, 27, 28, 29, 30, 31, 32, 33, 42, 43, 62, 65, 66, 75, 76)},
}


def need(*paths: Path) -> list[Path]:
    return [p for p in paths if not p.exists()]


def ff(*args) -> None:
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y", *map(str, args)], check=True)


# ───────────────────────── текст и WER ─────────────────────────
def norm_words(text: str) -> list[str]:
    t = unicodedata.normalize("NFKC", text).lower().replace("’", "'")
    t = re.sub(r"[^\w'\s]", " ", t)
    t = t.replace("_", " ")
    return [w.strip("'") for w in t.split() if w.strip("'")]


def edit_distance(a: list[str], b: list[str]) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def wer(ref: list[str], hyp: list[str]) -> float:
    return edit_distance(ref, hyp) / max(1, len(ref))


def clean_ref(text: str) -> str:
    """В TOS-en.srt встречается пометка «"Fourty years later"» — это титр на экране, не речь."""
    return re.sub(r'"Fourty years later"', " ", text)


@dataclass
class AlignStats:
    wer_total: float
    n_ref: int
    start_offsets: list[float]     # hyp − ref, сек, по репликам, где нашлось совпадение слова
    matched_lines: int
    total_lines: int


def compare_to_reference(ref_lines, hyp_lines) -> AlignStats:
    """WER по всему тексту + сдвиг начал реплик (по сопоставлению слов difflib)."""
    ref_words, ref_line_of = [], []
    for k, ln in enumerate(ref_lines):
        for w in norm_words(clean_ref(ln.text)):
            ref_words.append(w)
            ref_line_of.append(k)
    hyp_words, hyp_t = [], []
    for ln in hyp_lines:
        if ln.words:
            for w in ln.words:
                for x in norm_words(w.text):
                    hyp_words.append(x)
                    hyp_t.append(w.start)
        else:
            for x in norm_words(ln.text):
                hyp_words.append(x)
                hyp_t.append(ln.start)
    sm = difflib.SequenceMatcher(a=ref_words, b=hyp_words, autojunk=False)
    first_hit: dict[int, float] = {}
    for blk in sm.get_matching_blocks():
        for k in range(blk.size):
            line = ref_line_of[blk.a + k]
            if line not in first_hit:
                first_hit[line] = hyp_t[blk.b + k] - ref_lines[line].start
    return AlignStats(wer_total=wer(ref_words, hyp_words), n_ref=len(ref_words),
                      start_offsets=[first_hit[k] for k in sorted(first_hit)],
                      matched_lines=len(first_hit), total_lines=len(ref_lines))


# ───────────────────────── аудио-метрики ─────────────────────────
def align_lag(ref: np.ndarray, est: np.ndarray, sr: int, max_lag_s: float = 0.5) -> int:
    """Сдвиг (в отсчётах), на который надо сдвинуть est, чтобы совпасть с ref (по огибающим-мономиксам)."""
    from scipy.signal import fftconvolve
    a, b = ref.mean(axis=1) if ref.ndim == 2 else ref, est.mean(axis=1) if est.ndim == 2 else est
    n = min(len(a), len(b))
    c = fftconvolve(a[:n], b[:n][::-1], mode="full")
    mid = n - 1
    m = int(max_lag_s * sr)
    seg = c[mid - m: mid + m + 1]
    return int(np.argmax(seg) - m)


def shift(x: np.ndarray, k: int) -> np.ndarray:
    if k == 0:
        return x
    out = np.zeros_like(x)
    if k > 0:
        out[k:] = x[:-k]
    else:
        out[:k] = x[-k:]
    return out


def sdr(ref: np.ndarray, est: np.ndarray, *, scale_invariant: bool = False) -> float:
    n = min(len(ref), len(est))
    r, e = ref[:n].astype(np.float64).ravel(), est[:n].astype(np.float64).ravel()
    if scale_invariant:
        r = r * (np.dot(e, r) / (np.dot(r, r) + 1e-12))
    num = np.sum(r * r)
    den = np.sum((r - e) ** 2) + 1e-12
    return float(10 * np.log10(num / den + 1e-12))


def correlation(a: np.ndarray, b: np.ndarray) -> float:
    n = min(len(a), len(b))
    a, b = a[:n].astype(np.float64).ravel(), b[:n].astype(np.float64).ravel()
    a -= a.mean()
    b -= b.mean()
    return float(np.dot(a, b) / (np.sqrt(np.dot(a, a) * np.dot(b, b)) + 1e-12))


def rms_db(x: np.ndarray) -> float:
    return float(20 * np.log10(np.sqrt(np.mean(np.asarray(x, np.float64) ** 2)) + 1e-9))


# ───────────────────────── релизный MKV ─────────────────────────
def build_release_mkv(out: Path = TOS_RELEASE) -> Path:
    """MKV «как релиз»: видео + 5.1 (ac3, eng, default) + стерео aac + SRT eng/rus + 3 главы."""
    out.parent.mkdir(parents=True, exist_ok=True)
    meta = out.parent / "chapters.txt"
    meta.write_text(
        ";FFMETADATA1\n[CHAPTER]\nTIMEBASE=1/1000\nSTART=0\nEND=240000\ntitle=Пролог\n"
        "[CHAPTER]\nTIMEBASE=1/1000\nSTART=240000\nEND=480000\ntitle=Лаборатория\n"
        "[CHAPTER]\nTIMEBASE=1/1000\nSTART=480000\nEND=734000\ntitle=Финал\n", encoding="utf-8")
    ff("-i", TOS_MOV, "-i", TOS_51, "-i", TOS_EN, "-i", TOS_RU, "-i", meta,
       "-map", "0:v", "-map", "1:a", "-map", "0:a", "-map", "2:0", "-map", "3:0",
       "-map_metadata", "4", "-map_chapters", "4",
       "-c:v", "copy", "-c:a:0", "copy", "-c:a:1", "aac", "-b:a:1", "192k", "-c:s", "srt",
       "-metadata:s:a:0", "language=eng", "-metadata:s:a:0", "title=English 5.1",
       "-metadata:s:a:1", "language=eng", "-metadata:s:a:1", "title=English stereo",
       "-metadata:s:s:0", "language=eng", "-metadata:s:s:1", "language=rus",
       "-disposition:a:0", "default", "-disposition:a:1", "0", out)
    return out


# ───────────────────────── синтетический «дубль» ─────────────────────────
def say_take(text: str, out_wav: Path, *, voice: str = "Milena", lead_silence: float = 0.0,
             rate: int | None = None) -> Path:
    """Озвучка macOS `say` → wav 48 кГц mono float32 (+ тишина в начале)."""
    if sys.platform != "darwin":
        raise RuntimeError("say есть только на macOS")
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    # `say` читает «...» как «точка» — в субтитрах это просто пауза, так что убираем
    text = re.sub(r"\s*(\.{2,}|…)\s*", " ", text).strip(" -—") or text
    aiff = out_wav.with_suffix(".aiff")
    cmd = ["say", "-v", voice, "-o", str(aiff)]
    if rate:
        cmd += ["-r", str(rate)]
    subprocess.run([*cmd, text], check=True)
    af = f"adelay={int(lead_silence * 1000)}:all=1" if lead_silence > 0 else "anull"
    ff("-i", aiff, "-ac", "1", "-ar", "48000", "-af", af, "-c:a", "pcm_f32le", out_wav)
    aiff.unlink(missing_ok=True)
    return out_wav


def speech_onset(x: np.ndarray, sr: int, thr_db: float = -40.0) -> float:
    """Независимый от engine детектор начала речи: первое 10-мс окно выше (пик окна − |thr|)."""
    x = x.mean(axis=1) if x.ndim == 2 else x
    n = int(sr * 0.01)
    k = len(x) // n
    db = 20 * np.log10(np.sqrt(np.mean(x[: k * n].reshape(k, n) ** 2, axis=1)) + 1e-9)
    idx = np.where(db > db.max() + thr_db)[0]
    return float(idx[0] * 0.01) if len(idx) else 0.0


def speech_onset_sustained(x: np.ndarray, sr: int, thr_db: float = -30.0, frames: int = 4) -> float:
    """Начало речи, устойчивое к щелчкам: первое место, где `frames` 10-мс окон подряд выше (пик − |thr|).
    Нужен для голосов после разделения (demucs оставляет одиночные «щелчки» за 0,3 с до реплики)."""
    x = x.mean(axis=1) if x.ndim == 2 else x
    n = int(sr * 0.01)
    k = len(x) // n
    db = 20 * np.log10(np.sqrt(np.mean(x[: k * n].reshape(k, n) ** 2, axis=1)) + 1e-9)
    above = db > db.max() + thr_db
    for j in range(0, len(above) - frames + 1):
        if above[j:j + frames].all():
            return float(j * 0.01)
    return 0.0
