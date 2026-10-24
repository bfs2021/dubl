"""GigaAM v3 e2e-rnnt (Сбер, MIT) — распознавание русской речи через onnx-asr + onnxruntime (без torch, одинаково на всех ОС).

Почему он для русского (замеры — docs/QUALITY.md, «Распознавание русской речи»): текст на уровне Whisper, но границы реплик
в 2–3 раза точнее (Whisper ставит начало на 250–350 мс раньше голоса), без выдумок на музыке, с пунктуацией. Минусы:
знает только русский (на английском пишет транслит-мусор) и теряет короткие выкрики — их добирает Whisper (`asr_fill`).

Модель берёт ≤ 25 с за раз: длинный звук режем без потерь по самым тихим местам на куски ≤ 20 с (silero-vad не годится —
на мультфильме терял треть реплик под музыкой). Квантованная int8-сборка (~230 МБ) по тексту и границам не хуже fp32 (892 МБ).
CoreML-провайдер onnxruntime на энкодере GigaAM падает — провайдер всегда CPU.
"""
from __future__ import annotations

import logging
import re
from collections.abc import Callable
from pathlib import Path

import numpy as np

from ..models import Word
from ..util import data_dir

log = logging.getLogger(__name__)

MODEL_NAME = "gigaam-v3-e2e-rnnt"                  # имя модели в onnx-asr
REPO = "istupakov/gigaam-v3-onnx"                  # ONNX-экспорт ai-sage/GigaAM-v3 (MIT)
QUANT = "int8"
FILES = ("config.json", "v3_e2e_rnnt_encoder.int8.onnx", "v3_e2e_rnnt_decoder.int8.onnx", "v3_e2e_rnnt_joint.int8.onnx",
         "v3_e2e_rnnt_vocab.txt")
DOWNLOAD_MB = 230                                  # столько качается при первом разборе русского

CHUNK_MAX_S = 20.0          # куски ≤ 20 с (модель берёт ≤ 25 с)
CHUNK_SEARCH_S = 6.0        # …разрез ищем в последних 6 с куска
CHUNK_WIN_S = 0.4           # …по самому тихому окну 0,4 с
TOKEN_S = 0.04              # кадр энкодера: конец слова = последний токен + кадр
LINE_GAP = 0.5              # пауза между словами, после которой — новая реплика
SENT_GAP = 0.15             # …или конец предложения и пауза хотя бы такая
MAX_LINE = 8.0

_END = re.compile(r"[.!?…]$")


def model_dir() -> Path:
    return data_dir() / "models" / "gigaam" / f"{MODEL_NAME}-{QUANT}"


def installed() -> bool:
    """Пакеты для GigaAM стоят (без импорта: onnxruntime грузится заметно)."""
    from importlib.util import find_spec
    try:
        return find_spec("onnx_asr") is not None and find_spec("onnxruntime") is not None
    except (ImportError, ValueError):
        return False


def downloaded() -> bool:
    d = model_dir()
    return all((d / f).is_file() for f in FILES)


def ensure_model() -> Path:
    """Скачать модель в <данные>/models/gigaam (как Whisper — в <данные>/models/mlx). Уже есть — ничего не качает."""
    d = model_dir()
    if downloaded():
        return d
    from huggingface_hub import snapshot_download
    d.mkdir(parents=True, exist_ok=True)
    snapshot_download(REPO, local_dir=str(d), allow_patterns=[*FILES, "LICENSE.txt"])
    if not downloaded():
        raise RuntimeError("модель GigaAM скачалась не целиком")
    return d


def load_model():
    import onnx_asr
    return onnx_asr.load_model(MODEL_NAME, str(ensure_model()), quantization=QUANT,
                               providers=["CPUExecutionProvider"]).with_timestamps()


# ───────────────────────── нарезка и сборка слов (без нейросетей) ─────────────────────────
def energy_chunks(db: np.ndarray, duration: float, *, hop: float, max_s: float = CHUNK_MAX_S,
                  search_s: float = CHUNK_SEARCH_S, win: float = CHUNK_WIN_S) -> list[tuple[float, float]]:
    """Весь звук без потерь кусками ≤ max_s; разрез — посередине самого тихого окна `win` в последних `search_s` с куска.
    db — огибающая (дБ на кадр hop), как `transcribe.Envelope.db`."""
    p = 10 ** (np.asarray(db, dtype=np.float64) / 10)
    w = max(1, round(win / hop))
    out: list[tuple[float, float]] = []
    a = 0.0
    while duration - a > max_s:
        lo = int((a + max_s - search_s) / hop)
        hi = int((a + max_s) / hop) - w
        seg = p[lo:hi + w]
        if hi <= lo or len(seg) < w:
            cut = a + max_s
        else:
            k = int(np.argmin(np.convolve(seg, np.ones(w), mode="valid")))
            cut = (lo + k + w / 2) * hop
        cut = min(max(cut, a + 1.0), a + max_s)
        out.append((a, cut))
        a = cut
    out.append((a, duration))
    return out


def tokens_to_words(tokens: list[str], times: list[float], logprobs: list[float] | None, offset: float) -> list[Word]:
    """Токены onnx-asr → слова со временем и уверенностью (среднее по токенам, exp). У GigaAM пробел — отдельный токен."""
    words: list[Word] = []
    lps: list[list[float]] = []
    new = True
    for k, (tk, st) in enumerate(zip(tokens, times, strict=False)):
        nxt = times[k + 1] if k + 1 < len(times) else st + 0.3
        en = min(nxt, st + TOKEN_S)
        if not tk.strip(" ▁"):
            new = True
            continue
        lp = logprobs[k] if logprobs is not None and k < len(logprobs) else None
        if tk.startswith((" ", "▁")) or new or not words:
            words.append(Word(start=round(st + offset, 3), end=round(en + offset, 3), text=" " + tk.strip(" ▁")))
            lps.append([] if lp is None else [lp])
        else:
            words[-1].text += tk
            words[-1].end = round(en + offset, 3)
            if lp is not None:
                lps[-1].append(lp)
        new = False
    for w, ls in zip(words, lps, strict=True):
        w.prob = round(float(np.exp(np.mean(ls))), 3) if ls else None
    return words


def has_letters(t: str) -> bool:
    return bool(re.search(r"\w", t))


def group_words(words: list[Word], *, gap: float = LINE_GAP, max_len: float = MAX_LINE) -> list[list[Word]]:
    """Слова → реплики: пауза ≥ gap, тире (GigaAM ставит его на смене говорящего), конец предложения + пауза ≥ 0,15 с;
    длиннее max_len — режем по самой большой паузе."""
    groups: list[list[Word]] = []
    cur: list[Word] = []
    for w in words:
        if cur:
            g = w.start - cur[-1].end
            dash = w.text.lstrip().startswith(("—", "–"))
            if g >= gap or dash or (_END.search(cur[-1].text.strip()) and g >= SENT_GAP):
                groups.append(cur)
                cur = []
        cur.append(w)
    if cur:
        groups.append(cur)

    def split(ws: list[Word]) -> list[list[Word]]:
        if len(ws) < 2 or ws[-1].end - ws[0].start <= max_len:
            return [ws]
        gi = max(range(1, len(ws)), key=lambda i: ws[i].start - ws[i - 1].end)
        return split(ws[:gi]) + split(ws[gi:])

    out: list[list[Word]] = []
    for g in groups:
        g = [w for w in g if has_letters(w.text)]          # одиночные «—» без слов — не слова
        if g:
            out += split(g)
    return out


# ───────────────────────── распознавание ─────────────────────────
def recognize(read: Callable[[float, float], np.ndarray], chunks: list[tuple[float, float]], *,
              on_progress: Callable[[float], None] | None = None, model=None) -> list[Word]:
    """Слова всего звука. read(a, b) → моно float32 16 кГц куска [a, b)."""
    m = model or load_model()
    words: list[Word] = []
    total = chunks[-1][1] if chunks else 0.0
    for a, b in chunks:
        x = read(a, b)
        if len(x) >= 1600:                                  # < 0,1 с модели не нужно
            r = m.recognize(x, sample_rate=16000)
            words += tokens_to_words(list(r.tokens or []), list(r.timestamps or []), r.logprobs, a)
        if on_progress and total > 0:
            on_progress(min(1.0, b / total))
    return words
