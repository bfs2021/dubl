"""Замеры записи: шум комнаты, запас над шумом, перегруз — и починка перегруза (Clip Fix из Audacity).

Все функции работают с mono float32; частота любая (в движке — 48 кГц). Тяжёлых зависимостей нет (numpy + scipy).

Что меряем и зачем:
* `quiet_segment` / `noise_floor_db` — самый тихий ровный участок записи (чаще всего преролл до первого слова):
  по нему шумодав берёт «снимок тишины» (audio.denoise), по нему же считается шум для усиления дорожки;
* `snr_db` — насколько речь громче этого шума; от него зависит, насколько сильно чистим и сколько усиления допустимо;
* `clip_report` — доля срезанных отсчётов (плоские «полки» на верхушках волны);
* `declip` — Clip Fix: плоские участки восстанавливаются кубической кривой по наклону на соседних отсчётах.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

CLIP_THRESHOLD = 0.95          # доля от пика, выше которой отсчёт считается «на полке» (Audacity Clip Fix)
CLIP_MIN_RUN = 4               # подряд идущих плоских отсчётов, чтобы считать это срезом, а не вершиной волны
CLIP_FLAT = 2e-4               # «плоский» = соседние отсчёты различаются меньше этого (в долях полной шкалы)
CLIP_MIN_PEAK = 0.5            # тише −6 дБ перегруза в самом входе быть не может (тише пик — значит это не срез АЦП)
CLIP_WARN_PCT = 0.05           # от этой доли % срезанных отсчётов говорим «записано с перегрузом»
CLIPFIX_HEADROOM_DB = -9.0     # после восстановления пик опускаем до −9 дБ: запас под вернувшиеся вершины

SNR_NOISY_DB = 25.0            # речь над шумом меньше — запись шумная
SNR_OK_DB = 35.0               # меньше — шум заметен при подъёме


def _db(v: float) -> float:
    return float(20 * np.log10(max(float(v), 1e-9)))


def frame_power(x: np.ndarray, sr: int, win: float = 0.02) -> np.ndarray:
    """Средняя мощность по кадрам win секунд."""
    n = max(1, round(sr * win))
    k = len(x) // n
    if k == 0:
        return np.zeros(0)
    fr = np.asarray(x[: k * n], dtype=np.float64).reshape(k, n)
    return np.mean(fr * fr, axis=1)


# ─────────────────────────── тишина и шум ───────────────────────────
def quiet_segment(x: np.ndarray, sr: int, *, min_len: float = 0.25, max_len: float = 1.0,
                  min_gap_db: float = 14.0) -> tuple[float, float] | None:
    """Самый тихий ровный участок записи (начало, конец в секундах) или None.

    Участок должен быть минимум на `min_gap_db` тише громкой речи записи (иначе «тишины» в записи нет — всё время
    что-то звучит) и ровным: громкость кадров внутри почти не прыгает (не кусок слова и не щелчок мыши).
    Ищем окнами max_len → min_len, берём самое тихое.
    """
    x = np.asarray(x, dtype=np.float32)
    p = frame_power(x, sr)
    if len(p) < int(min_len / 0.02) + 1:
        return None
    db = 10 * np.log10(p + 1e-18)
    loud = float(np.percentile(db, 95))
    if loud < -75.0:                                  # пустая запись
        return None
    csum = np.concatenate([[0.0], np.cumsum(p)])
    for L in (max_len, 0.6, min_len):
        w = int(round(L / 0.02))
        if w > len(p) or w < 2:
            continue
        mean_p = (csum[w:] - csum[:-w]) / w
        order = np.argsort(mean_p)
        for i in order[:40]:
            seg = db[i: i + w]
            if 10 * np.log10(mean_p[i] + 1e-18) > loud - min_gap_db:
                break                                  # дальше только громче
            if float(np.std(seg)) > 4.5 or float(seg.max() - np.median(seg)) > 9.0:
                continue                               # не ровно: слово, щелчок, всплеск
            return (i * 0.02, (i + w) * 0.02)
    return None


def noise_floor_db(x: np.ndarray, sr: int) -> float | None:
    """Шум комнаты (RMS в самом тихом ровном участке), дБ ниже полной шкалы. None — тишины в записи нет."""
    seg = quiet_segment(x, sr)
    if seg is None:
        return None
    s = np.asarray(x[int(seg[0] * sr): int(seg[1] * sr)], dtype=np.float64)
    return _db(float(np.sqrt(np.mean(s * s))))


def speech_rms_db(x: np.ndarray, sr: int) -> float | None:
    """RMS по самим фразам (дБ). None — речи не нашли."""
    from .fit import detect_speech

    iv = detect_speech(x, sr, min_gap=0.15)
    if not iv:
        return None
    seg = np.concatenate([np.asarray(x[int(a * sr): int(b * sr)], dtype=np.float64) for a, b in iv])
    if len(seg) == 0:
        return None
    return _db(float(np.sqrt(np.mean(seg * seg))))


def snr_db(x: np.ndarray, sr: int) -> float | None:
    nf, sp = noise_floor_db(x, sr), speech_rms_db(x, sr)
    if nf is None or sp is None:
        return None
    return float(sp - nf)


# ─────────────────────────── перегруз ───────────────────────────
@dataclass
class ClipReport:
    pct: float                          # доля срезанных отсчётов, %
    runs: list[tuple[int, int]]         # [начало, конец) плоских участков
    peak: float


def clip_runs(x: np.ndarray) -> tuple[list[tuple[int, int]], float]:
    """Плоские участки на верхушках волны (срез). → ([начало, конец), …], пик."""
    x = np.asarray(x, dtype=np.float32)
    if x.size < CLIP_MIN_RUN + 2:
        return [], float(np.abs(x).max()) if x.size else 0.0
    a = np.abs(x)
    peak = float(a.max())
    if peak < CLIP_MIN_PEAK:
        return [], peak
    hi = a >= CLIP_THRESHOLD * peak
    same_sign = np.sign(x[1:]) == np.sign(x[:-1])
    flat = np.concatenate([[False], (np.abs(np.diff(x)) <= CLIP_FLAT) & same_sign])   # отсчёт повторяет предыдущий
    cand = hi & (flat | np.concatenate([flat[1:], [False]]))         # плоский с левой или с правой стороны
    idx = np.flatnonzero(np.diff(np.concatenate(([0], cand.astype(np.int8), [0]))))
    runs = [(int(s), int(e)) for s, e in zip(idx[0::2], idx[1::2]) if e - s >= CLIP_MIN_RUN]
    return runs, peak


def clip_report(x: np.ndarray) -> ClipReport:
    x = np.asarray(x, dtype=np.float32)
    runs, peak = clip_runs(x)
    n = sum(e - s for s, e in runs)
    return ClipReport(pct=round(100.0 * n / max(len(x), 1), 4), runs=runs, peak=peak)


def declip(x: np.ndarray, *, headroom_db: float = CLIPFIX_HEADROOM_DB) -> tuple[np.ndarray, ClipReport]:
    """Clip Fix: каждый срезанный участок заменяется кубической (Эрмита) кривой по наклону на соседних отсчётах;
    восстановленные вершины только выше среза. Если что-то починили — пик опускаем до `headroom_db`, чтобы
    вернувшиеся вершины не упёрлись в потолок. Без среза возвращает вход без изменений."""
    x = np.asarray(x, dtype=np.float32)
    rep = clip_report(x)
    if rep.pct < 1e-4 or not rep.runs:
        return x, rep
    y = x.astype(np.float64).copy()
    n = len(x)
    for s, e in rep.runs:
        i0, i1 = s - 1, e                                    # соседние отсчёты слева/справа, не срезанные
        if i0 < 1 or i1 > n - 2:
            continue
        sign = 1.0 if x[s] >= 0 else -1.0
        m0 = float(x[i0] - x[i0 - 1])                        # наклон слева (за отсчёт)
        m1 = float(x[i1 + 1] - x[i1])                        # наклон справа
        L = i1 - i0
        t = np.arange(1, L) / L
        h00, h10 = 2 * t**3 - 3 * t**2 + 1, t**3 - 2 * t**2 + t
        h01, h11 = -2 * t**3 + 3 * t**2, t**3 - t**2
        seg = h00 * x[i0] + h10 * L * m0 + h01 * x[i1] + h11 * L * m1
        cur = np.asarray(x[i0 + 1: i1], dtype=np.float64)
        up = np.maximum(sign * seg, sign * cur)               # только наружу, не вглубь
        y[i0 + 1: i1] = sign * up
    out = y.astype(np.float32)
    pk = float(np.abs(out).max())
    target = 10 ** (headroom_db / 20)
    if pk > target:
        out = (out * np.float32(target / pk)).astype(np.float32)
    return out, rep


# ─────────────────────────── общий замер дубля ───────────────────────────
@dataclass
class TakeMeasure:
    peak_db: float
    noise_floor_db: float | None
    snr_db: float | None
    clip_pct: float


def measure_take(x: np.ndarray, sr: int) -> TakeMeasure:
    """Замеры одного дубля для Take.noise_floor_db / snr_db / clip_pct (вызывает пайплайн при записи)."""
    x = np.asarray(x, dtype=np.float32)
    if x.ndim > 1:
        x = x.mean(axis=1).astype(np.float32)
    nf = noise_floor_db(x, sr)
    sp = speech_rms_db(x, sr)
    return TakeMeasure(
        peak_db=round(_db(float(np.abs(x).max()) if x.size else 0.0), 2),
        noise_floor_db=None if nf is None else round(nf, 1),
        snr_db=None if (nf is None or sp is None) else round(sp - nf, 1),
        clip_pct=clip_report(x).pct,
    )


def clip_warning(clip_pct: float | None) -> str | None:
    """Честная фраза для интерфейса и помощника; None — перегруза нет."""
    if clip_pct is None or clip_pct < CLIP_WARN_PCT:
        return None
    return "Записано с перегрузом — голос срезан. Отодвинься от микрофона или убавь его громкость."
