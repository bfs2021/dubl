"""Автоподгонка дубля: найти, где речь дубля, где речь оригинала, и подобрать сдвиг и растяжение.

Соглашения (на них опираются fit_take, render_fitted и audio.mix):

* Файл дубля — запись с момента `t = 0`; видео стартовало в `t = preroll`.
  Время сцены, которому «наивно» соответствует отсчёт `t` файла: `take_start_rel + (t − preroll)`.
* `t_ss` — момент начала речи в файле дубля (`find_take_speech`).
* `offset_ms` — на сколько миллисекунд надо сдвинуть НАЧАЛО РЕЧИ дубля относительно её наивного положения
  `take_start_rel + (t_ss − preroll)`, чтобы оно совпало с началом речи оригинала. Минус — раньше.
  Задержка звуковой карты (latency) и привычка опаздывать уже внутри offset: он считается по факту совпадения речи.
* `stretch` — коэффициент ДЛИНЫ (как в models.Take): <1 — дубль укорачивается, >1 — удлиняется.
  Растяжение идёт вокруг начала речи `t_ss`, поэтому начало речи не «плывёт» при смене stretch.
* Итоговая позиция отсчёта `t` файла в сцене:
  `take_start_rel + (t_ss − preroll) + offset_ms/1000 + (t − t_ss) * stretch`
  т.е. начало файла лежит в `take_start_rel − preroll + offset_ms/1000 + t_ss * (1 − stretch)`
  (при stretch = 1 это ровно `take_start_rel − preroll + offset_ms/1000`).
"""
from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..util import run_ffmpeg
from .io import read_wav, to_mono, write_wav

STRETCH_MIN = 0.85
STRETCH_MAX = 1.15
STRETCH_DEADZONE = 0.04   # отклонение длины меньше 4% — не растягиваем
XCORR_RANGE_S = 0.060   # уточнение огибающими: ±60 мс вокруг оценки по границам речи (±150 мс уводило на шумных голосах до 170 мс)
_HOP = 0.010


@dataclass
class FitResult:
    offset_ms: float
    stretch: float
    confidence: float


# ─────────────────────────── детектор речи ───────────────────────────
def _frame_db(x: np.ndarray, sr: int, win: float) -> np.ndarray:
    n = max(1, round(sr * win))
    k = len(x) // n
    if k == 0:
        return np.zeros(0)
    fr = x[: k * n].reshape(k, n).astype(np.float64)
    rms = np.sqrt(np.mean(fr * fr, axis=1))
    return 20 * np.log10(rms + 1e-9)


def detect_speech(x: np.ndarray, sr: int, *, min_gap: float = 0.25,
                  min_duration: float = 0.06) -> list[tuple[float, float]]:
    """Интервалы речи (секунды) по энергии: окна 20 мс, порог от шумового пола, паузы короче min_gap склеиваются."""
    x = to_mono(np.asarray(x, dtype=np.float32))
    win = 0.02
    db = _frame_db(x, sr, win)
    if len(db) == 0:
        return []
    peak = float(np.percentile(db, 98))
    floor = float(np.percentile(db, 10))
    if peak < -60 or peak - floor < 12:
        return []
    thr = max(floor + 10.0, peak - 30.0)
    active = db > thr
    # гистерезис: к активным кадрам прилипают соседние, чуть ниже порога (начала/концы слов)
    low = db > thr - 8.0
    for _ in range(3):
        grow = active.copy()
        grow[1:] |= active[:-1] & low[1:]
        grow[:-1] |= active[1:] & low[:-1]
        if (grow == active).all():
            break
        active = grow
    # в интервалы
    idx = np.flatnonzero(np.diff(np.concatenate(([0], active.astype(np.int8), [0]))))
    spans = [(int(a), int(b)) for a, b in zip(idx[0::2], idx[1::2])]
    merged: list[list[int]] = []
    gap_frames = round(min_gap / win)
    for a, b in spans:
        if merged and a - merged[-1][1] < gap_frames:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    out = [(a * win, b * win) for a, b in merged if (b - a) * win >= min_duration]
    return out


CLIP_TAIL_MIN = 0.12   # хвост чужого звука, заходящий в окно реплики меньше чем на столько секунд, речью этой реплики не считаем


def _clip_intervals(iv: list[tuple[float, float]], lo: float, hi: float) -> list[tuple[float, float]]:
    out = []
    for a, b in iv:
        a2, b2 = max(a, lo), min(b, hi)
        if b2 - a2 <= 0.03:
            continue
        # обрезок интервала, начавшегося до окна: это хвост предыдущей реплики/шума (после разделения голосов он бывает
        # и от демукса), а не начало этой реплики — иначе начало речи «съезжает» на левую границу окна
        if a < lo and b2 - a2 < CLIP_TAIL_MIN:
            continue
        out.append((a2, b2))
    return out


def find_take_speech(x: np.ndarray, sr: int, *, line_start_rel: float, line_end_rel: float,
                     take_start_rel: float, preroll: float, latency_ms: float = 0.0,
                     intervals: list[tuple[float, float]] | None = None,
                     orig_len: float | None = None) -> tuple[float, float] | None:
    """Начало и конец речи дубля ЭТОЙ реплики в секундах файла; None — речи нет."""
    if intervals is None:
        intervals = detect_speech(x, sr)
    dur = len(x) / sr
    shift = preroll + latency_ms / 1000.0 - take_start_rel
    t_lo = max(0.0, shift + line_start_rel - 1.5)
    t_hi = min(dur, shift + line_end_rel + 2.5)
    cand = [(a, b) for a, b in intervals if b > t_lo and a < t_hi]
    if not cand:
        return None
    first = cand[0]
    if orig_len is None:
        orig_len = max(0.3, line_end_rel - line_start_rel)
    bound = first[0] + max(orig_len * 1.4, orig_len + 1.0)
    group = [iv for iv in cand if iv[0] < bound]
    return group[0][0], group[-1][1]


@dataclass
class SpeechLocation:
    """Где речь в окне реплики: у оригинала (сцена, с) и у дубля (файл, с)."""
    guide_ok: bool          # речь оригинала нашли (иначе берём границы реплики)
    o_start: float
    o_end: float
    o_len: float
    t_ss: float             # начало речи в файле дубля
    t_se: float


def locate_speech(dub: np.ndarray, guide: np.ndarray, sr: int, *, line_start_rel: float, line_end_rel: float,
                  take_start_rel: float, preroll: float, latency_ms: float = 0.0) -> SpeechLocation | None:
    """Единая точка: где речь оригинала и где речь дубля. Её вызывают и fit_take (считает сдвиг), и mix_scene
    (ставит дубль) — иначе пивот t_ss у них мог бы разойтись, и дубль встал бы не туда, куда его подогнали."""
    margin = 0.2
    g_iv = _clip_intervals(detect_speech(guide, sr), line_start_rel - margin, line_end_rel + margin)
    guide_ok = bool(g_iv)
    if guide_ok:
        o_start, o_end = g_iv[0][0], g_iv[-1][1]
    else:
        o_start, o_end = line_start_rel, line_end_rel
    o_len = max(0.1, o_end - o_start)
    sp = find_take_speech(dub, sr, line_start_rel=line_start_rel, line_end_rel=line_end_rel,
                          take_start_rel=take_start_rel, preroll=preroll, latency_ms=latency_ms,
                          intervals=detect_speech(dub, sr), orig_len=o_len)
    if sp is None:
        return None
    return SpeechLocation(guide_ok, o_start, o_end, o_len, sp[0], sp[1])


# ─────────────────────────── огибающие для уточнения ───────────────────────────
def _env_series(x: np.ndarray, sr: int) -> tuple[np.ndarray, float]:
    """Лог-огибающая с шагом 10 мс (окно 20 мс). Возвращает (env_db, hop_s)."""
    hop = round(sr * _HOP)
    win = hop * 2
    n = (len(x) - win) // hop + 1
    if n <= 0:
        return np.zeros(0), _HOP
    # скользящая мощность через кумулятивную сумму
    c = np.concatenate(([0.0], np.cumsum(x.astype(np.float64) ** 2)))
    starts = np.arange(n) * hop
    p = (c[starts + win] - c[starts]) / win
    return 10 * np.log10(p + 1e-18), _HOP


def _norm_env(env_db: np.ndarray) -> np.ndarray:
    if len(env_db) == 0:
        return env_db
    peak = np.percentile(env_db, 98)
    return np.clip(env_db, peak - 40.0, None) - (peak - 40.0)


class _EnvMatcher:
    """Сравнение огибающей гида (сцена) и огибающей дубля (файл) при заданных offset/stretch."""

    def __init__(self, guide: np.ndarray, dub: np.ndarray, sr: int, *, g_lo: float, g_hi: float,
                 take_start_rel: float, preroll: float, t_ss: float):
        ge, _ = _env_series(guide, sr)
        de, _ = _env_series(dub, sr)
        self.ge, self.de = _norm_env(ge), _norm_env(de)
        self.sr = sr
        # сетка сцены
        self.T = np.arange(g_lo, g_hi, _HOP)
        gi = np.clip(((self.T + _HOP) / _HOP).astype(int), 0, max(len(self.ge) - 1, 0))  # центр окна ≈ +10 мс
        self.g = self.ge[gi] if len(self.ge) else np.zeros(len(self.T))
        self.tsr, self.preroll, self.t_ss = take_start_rel, preroll, t_ss
        self.de_t = (np.arange(len(self.de)) * _HOP) + _HOP  # центр окна дубля, сек файла

    def corr(self, offset_s: float, stretch: float) -> float:
        if len(self.g) < 8 or len(self.de) == 0:
            return 0.0
        t = self.t_ss + (self.T - self.tsr - (self.t_ss - self.preroll) - offset_s) / stretch
        d = np.interp(t, self.de_t, self.de, left=0.0, right=0.0)
        g = self.g - self.g.mean()
        d = d - d.mean()
        den = np.sqrt((g * g).sum() * (d * d).sum())
        return float((g * d).sum() / den) if den > 1e-9 else 0.0


# ─────────────────────────── подгонка ───────────────────────────
def fit_take(take_wav, guide_voice_wav, *, line_start_rel: float, line_end_rel: float,
             take_start_rel: float, preroll: float, latency_ms: float = 0.0) -> FitResult:
    """Совместить начало/конец речи дубля с речью оригинала в окне реплики. См. соглашения в шапке модуля."""
    dub, sr_d = read_wav(take_wav)
    guide, sr_g = read_wav(guide_voice_wav)
    dub, guide = to_mono(dub), to_mono(guide)
    if sr_d != sr_g:
        from math import gcd

        from scipy.signal import resample_poly
        g = gcd(sr_d, sr_g)
        dub = resample_poly(dub, sr_g // g, sr_d // g).astype(np.float32)
    sr = sr_g

    loc = locate_speech(dub, guide, sr, line_start_rel=line_start_rel, line_end_rel=line_end_rel,
                        take_start_rel=take_start_rel, preroll=preroll, latency_ms=latency_ms)
    if loc is None:
        return FitResult(offset_ms=0.0, stretch=1.0, confidence=0.0)
    guide_ok, o_start, o_end, o_len = loc.guide_ok, loc.o_start, loc.o_end, loc.o_len
    t_ss, t_se = loc.t_ss, loc.t_se
    d_len = max(0.1, t_se - t_ss)

    # растяжение
    raw = o_len / d_len
    clamped = raw < STRETCH_MIN or raw > STRETCH_MAX
    stretch = float(np.clip(raw, STRETCH_MIN, STRETCH_MAX))
    if abs(stretch - 1.0) < STRETCH_DEADZONE:
        stretch = 1.0

    def offset_for(s: float) -> float:
        return o_start - (take_start_rel + (t_ss - preroll))

    offset = offset_for(stretch)

    # уточнение огибающими
    g_lo = max(0.0, o_start - 0.3)
    g_hi = o_end + 0.3
    matcher = _EnvMatcher(guide, dub, sr, g_lo=g_lo, g_hi=g_hi, take_start_rel=take_start_rel,
                          preroll=preroll, t_ss=t_ss)
    lags = np.arange(-XCORR_RANGE_S, XCORR_RANGE_S + 1e-9, _HOP)

    def best_lag(s: float, base: float) -> tuple[float, float, float]:
        cs = np.array([matcher.corr(base + l, s) for l in lags])
        i = int(np.argmax(cs))
        lag = lags[i]
        if 0 < i < len(cs) - 1:  # парабола
            a, b, c = cs[i - 1], cs[i], cs[i + 1]
            den = a - 2 * b + c
            if abs(den) > 1e-9:
                lag += 0.5 * (a - c) / den * _HOP
        return float(lag), float(cs[i]), float(matcher.corr(base, s))

    corr0 = matcher.corr(offset, stretch)
    best_s, best_off, best_c = stretch, offset, corr0
    if guide_ok:
        cand_s = [stretch] if clamped else [stretch * (1 + d) for d in (-0.03, -0.02, -0.01, 0.0, 0.01, 0.02, 0.03)]
        for s in cand_s:
            s = float(np.clip(s, STRETCH_MIN, STRETCH_MAX))
            lag, c, _ = best_lag(s, offset_for(s))
            # небольшой штраф за уход от оценки по границам речи — чтобы шум не уводил растяжение
            c_adj = c - 0.02 * abs(s - stretch) / 0.03
            if c_adj > best_c + 0.02:
                best_s, best_off, best_c = s, offset_for(s) + lag, c
        if best_s == stretch and best_off == offset:
            lag, c, c0 = best_lag(stretch, offset)
            if c > c0 + 0.01:
                best_off, best_c = offset + lag, c
        stretch, offset = best_s, best_off
        if abs(stretch - 1.0) < STRETCH_DEADZONE and not clamped:
            # после уточнения вернулись в «мёртвую зону» — не трогаем длину, пересчитываем сдвиг
            stretch = 1.0
            lag, c, _ = best_lag(1.0, offset_for(1.0))
            offset, best_c = offset_for(1.0) + lag, max(c, best_c * 0.9)

    conf = float(np.clip(best_c, 0.0, 1.0))
    if clamped:
        conf *= 0.7
    if not guide_ok:
        conf *= 0.5
    return FitResult(offset_ms=float(offset * 1000.0), stretch=float(stretch), confidence=round(conf, 3))


# ─────────────────────────── применение ───────────────────────────
def time_stretch(x: np.ndarray, sr: int, stretch: float) -> np.ndarray:
    """Изменить длину в `stretch` раз без смены тона (ffmpeg atempo, он же WSOLA). stretch=1 — без изменений."""
    x = np.asarray(x, dtype=np.float32)
    if abs(stretch - 1.0) < 1e-4 or len(x) == 0:
        return x
    tempo = 1.0 / stretch
    if not 0.5 <= tempo <= 2.0:
        raise ValueError("stretch вне диапазона 0.5..2")
    with tempfile.TemporaryDirectory(prefix="dubl_fit_") as td:
        a, b = Path(td) / "in.wav", Path(td) / "out.wav"
        write_wav(a, x, sr)
        run_ffmpeg(["-i", a, "-af", f"atempo={tempo:.6f}", "-c:a", "pcm_f32le", b])
        y, _ = read_wav(b)
    return y


def retime(x: np.ndarray, sr: int, *, stretch: float, offset_ms: float, pivot_s: float) -> tuple[np.ndarray, float]:
    """Растянуть вокруг `pivot_s` (начало речи в файле) и вернуть (аудио, shift_s).

    shift_s — где начало результата лежит относительно «наивного» начала файла (take_start_rel − preroll).
    """
    y = time_stretch(x, sr, stretch)
    shift = pivot_s * (1.0 - stretch) + offset_ms / 1000.0
    return y, shift


def render_fitted(take_wav, fit: FitResult, out_wav, *, pivot_s: float | None = None) -> Path:
    """Применить подгонку к файлу дубля: растяжение вокруг начала речи + сдвиг (offset_ms).

    Результат — тот же файл, перекроенный так, будто его начало лежит на наивной позиции:
    сдвиг вперёд добавляет тишину в начало, сдвиг назад обрезает начало (там была тишина/преролл).
    `pivot_s` — начало речи в файле; по умолчанию находится detect_speech.
    """
    x, sr = read_wav(take_wav)
    mono = to_mono(x)
    if pivot_s is None:
        iv = detect_speech(mono, sr)
        pivot_s = iv[0][0] if iv else 0.0
    y, shift = retime(mono, sr, stretch=fit.stretch, offset_ms=fit.offset_ms, pivot_s=pivot_s)
    k = round(shift * sr)
    if k > 0:
        y = np.concatenate([np.zeros(k, dtype=np.float32), y])
    elif k < 0:
        y = y[-k:]
    return write_wav(out_wav, y, sr)
