"""Подгонка таймингов встроенных субтитров к реальной речи (whisper).

Главная идея — **одно монотонное выравнивание слов**, а не независимый поиск для каждой строки:
слова субтитров и слова речи сопоставляются динамическим программированием (как diff, только с учётом времени),
каждое слово речи достаётся максимум одной строке. Поэтому две соседние реплики не могут получить один и тот же кусок речи,
порядок сохраняется сам собой. После этого границы приводятся к правилам: без новых перекрытий, без перескоков,
ограниченный сдвиг, разумная длина. Если уверенности нет — остаётся тайминг из SRT.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher

import numpy as np

from ..models import Line

MAX_TIME_SHIFT = 0.79     # максимум сдвига, когда опираемся только на время
TEXT_WINDOW = 3.0         # где искать совпадающие по тексту слова (по времени)
MAX_TEXT_SHIFT = 3.0      # максимум сдвига границы, когда совпал почти весь текст строки
MID_TEXT_SHIFT = 1.5      # …когда совпала большая часть
WEAK_TEXT_SHIFT = 0.79    # …если совпала лишь часть слов
TEXT_THRESHOLD = 0.5      # доля (по весу) совпавших слов строки, ниже — тексты считаем непохожими
MIN_DURATION = 0.1
ORPHAN_GAP = 0.7          # пауза внутри совпавших слов строки, после которой крайнее слово считаем чужим
WORD_GUESS = 0.32         # сколько секунд на несовпавшее слово при достройке краёв строки
MAX_CELLS = 16_000_000    # предел размера матрицы выравнивания одного куска

_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)


def _norm(text: str) -> str:
    return " ".join(_PUNCT.sub(" ", text.lower().replace("ё", "е")).split())


def _script(text: str) -> str | None:
    cyr = sum(1 for ch in text if "\u0400" <= ch <= "\u04ff")
    lat = sum(1 for ch in text if ch.isalpha() and ch.isascii())
    other = sum(1 for ch in text if ch.isalpha() and not ch.isascii() and not "\u0400" <= ch <= "\u04ff")
    if cyr + lat + other == 0:
        return None
    return max((("cyr", cyr), ("lat", lat), ("other", other)), key=lambda x: x[1])[0]


def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def _bounds(group: list[Line]) -> tuple[float, float]:
    """Границы группы реплик речи по словам (если есть), иначе по репликам."""
    starts, ends = [], []
    for s in group:
        if s.words:
            starts.append(s.words[0].start)
            ends.append(s.words[-1].end)
        else:
            starts.append(s.start)
            ends.append(s.end)
    return min(starts), max(ends)


def _contiguous_groups(cands: list[Line], limit: int = 4):
    for i in range(len(cands)):
        for j in range(i, min(i + limit, len(cands))):
            yield cands[i:j + 1]


def _clamp_shift(new: float, old: float, limit: float = MAX_TIME_SHIFT) -> float:
    return min(max(new, old - limit), old + limit)


def _finish(line: Line, start: float, end: float) -> Line:
    start = max(start, 0.0)
    if end - start < MIN_DURATION:
        end = start + MIN_DURATION
    return line.model_copy(update={"start": round(start, 3), "end": round(end, 3)})


# ───────────────────────── слова речи и слова субтитров ─────────────────────────
@dataclass
class _Tok:
    text: str
    start: float
    end: float


def _line_tokens(x: Line) -> list[_Tok]:
    """Слова реплики речи с временем. Если слова whisper неполные (не покрывают текст) — размазываем текст по их границам."""
    text_toks = _norm(x.text).split()
    out: list[_Tok] = []
    for w in x.words:
        parts = _norm(w.text).split()
        if not parts:
            continue
        w0 = w.start
        expected = 0.08 * sum(len(p) for p in parts) + 0.15
        if w.end - w0 > max(0.6, 2.5 * expected):       # whisper растягивает слово на паузу: настоящая речь — у конца слова
            w0 = w.end - expected
        dur = max(w.end - w0, 0.0) / len(parts)
        for k, p in enumerate(parts):
            out.append(_Tok(p, w0 + k * dur, w0 + (k + 1) * dur))
    if out and (not text_toks or len(out) >= 0.7 * len(text_toks)):
        return out
    if not text_toks:
        return []
    a, b = (x.words[0].start, x.words[-1].end) if x.words else (x.start, x.end)
    if b <= a:
        b = a + MIN_DURATION
    step = (b - a) / len(text_toks)
    return [_Tok(t, a + i * step, a + (i + 1) * step) for i, t in enumerate(text_toks)]


def _token_weight(t: str) -> float:
    n = len(t)
    return 1.0 if n >= 4 else 0.7 if n == 3 else 0.4


def _similarity(a: str, b: str) -> float:
    if a == b:
        return 1.0
    if len(a) >= 3 and len(b) >= 3 and a[0] == b[0] and abs(len(a) - len(b)) <= 3:
        r = SequenceMatcher(None, a, b).ratio()
        if r >= 0.75:
            return 0.7 * r
    return 0.0


@dataclass
class _Match:
    start: float
    end: float
    token: int       # номер слова строки


def _dp_match(srt: list[tuple[str, float, float, float]], sp: list[_Tok]) -> list[tuple[int, int]]:
    """Монотонное выравнивание: srt = [(слово, время слова, начало строки, конец строки)], sp — слова речи.

    Возвращает пары (индекс слова srt, индекс слова речи). Матч допускается, если слово речи не дальше TEXT_WINDOW
    от интервала строки; цель — максимум суммарного веса совпадений с небольшим штрафом за расстояние по времени.
    """
    n, m = len(srt), len(sp)
    if not n or not m:
        return []
    sp_mid = np.array([(t.start + t.end) / 2 for t in sp])
    s = np.full((n, m), -1e9, dtype=np.float32)
    for i, (tok, t, l0, l1) in enumerate(srt):
        lo = int(np.searchsorted(sp_mid, l0 - TEXT_WINDOW, side="left"))
        hi = int(np.searchsorted(sp_mid, l1 + TEXT_WINDOW, side="right"))
        w = _token_weight(tok)
        for j in range(lo, hi):
            sim = _similarity(tok, sp[j].text)
            if sim <= 0:
                continue
            pen = 0.25 * min(1.0, abs(sp_mid[j] - t) / (2 * TEXT_WINDOW))
            v = w * sim - pen
            if v > 0.05:
                s[i, j] = v
    g = np.zeros((n + 1, m + 1), dtype=np.float32)
    for i in range(1, n + 1):
        tmp = np.maximum(g[i - 1, 1:], g[i - 1, :-1] + s[i - 1])
        g[i, 1:] = np.maximum.accumulate(np.maximum(tmp, 0.0))
    pairs: list[tuple[int, int]] = []
    i, j = n, m
    while i > 0 and j > 0:
        if s[i - 1, j - 1] > -1e8 and abs(g[i, j] - (g[i - 1, j - 1] + s[i - 1, j - 1])) < 1e-4 and g[i, j] > g[i - 1, j] + 1e-6 \
                and g[i, j] > g[i, j - 1] + 1e-6:
            pairs.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif g[i, j] <= g[i - 1, j] + 1e-6:
            i -= 1
        elif g[i, j] <= g[i, j - 1] + 1e-6:
            j -= 1
        else:  # g[i,j] получено диагональю (равенство с up/left выше проверили строго)
            pairs.append((i - 1, j - 1))
            i -= 1
            j -= 1
    pairs.reverse()
    return pairs


def _chunks(lines: list[Line], sp: list[_Tok]) -> list[tuple[int, int, int, int]]:
    """Режем задачу на независимые куски по паузам > TEXT_WINDOW, где нет ни субтитров, ни речи.

    Возвращает [(первая строка, последняя+1, первое слово речи, последнее+1)].
    """
    events: list[tuple[float, float, str, int]] = [(ln.start, ln.end, "l", i) for i, ln in enumerate(lines)]
    events += [(t.start, t.end, "s", j) for j, t in enumerate(sp)]
    events.sort()
    out: list[tuple[int, int, int, int]] = []
    cur_end = None
    l0, s0 = 0, 0
    nl = ns = 0
    for a, b, kind, idx in events:
        if cur_end is not None and a - cur_end > TEXT_WINDOW:
            out.append((l0, l0 + nl, s0, s0 + ns))
            l0, s0, nl, ns = l0 + nl, s0 + ns, 0, 0
        cur_end = b if cur_end is None else max(cur_end, b)
        if kind == "l":
            nl += 1
        else:
            ns += 1
    out.append((l0, l0 + nl, s0, s0 + ns))
    return out


def _split_big(lines: list[Line], sp: list[_Tok], l0: int, l1: int, s0: int, s1: int, tok_counts: list[int]
               ) -> list[tuple[int, int, int, int]]:
    """Слишком большой кусок (сплошная речь без пауз) делим по самой длинной паузе между строками в середине."""
    n_tok = sum(tok_counts[l0:l1])
    if n_tok * (s1 - s0) <= MAX_CELLS or l1 - l0 < 2:
        return [(l0, l1, s0, s1)]
    lo, hi = l0 + (l1 - l0) // 3, l0 + 2 * (l1 - l0) // 3 + 1
    best = max(range(max(lo, l0 + 1), max(hi, l0 + 2)), key=lambda k: lines[k].start - lines[k - 1].end)
    cut_t = (lines[best].start + lines[best - 1].end) / 2
    sc = s0
    while sc < s1 and (sp[sc].start + sp[sc].end) / 2 < cut_t:
        sc += 1
    return _split_big(lines, sp, l0, best, s0, sc, tok_counts) + _split_big(lines, sp, best, l1, sc, s1, tok_counts)


def _text_matches(lines: list[Line], speech: list[Line]) -> dict[int, tuple[list[_Match], float, int]]:
    """Для каждой строки субтитров → (совпавшие слова с временем речи, доля совпавшего веса, число слов строки)."""
    result: dict[int, tuple[list[_Match], float, int]] = {}
    line_toks = [_norm(ln.text).split() for ln in lines]
    scripts = [_script(ln.text) for ln in lines]
    for script in {s for s in scripts if s is not None}:
        idx = [i for i, s in enumerate(scripts) if s == script and line_toks[i]]
        sp_lines = sorted((x for x in speech if _script(x.text) == script), key=lambda x: (x.start, x.end))
        sp: list[_Tok] = []
        for x in sp_lines:
            sp.extend(_line_tokens(x))
        if not idx or not sp:
            continue
        sp.sort(key=lambda t: (t.start, t.end))
        sub = [lines[i] for i in idx]
        counts = [len(line_toks[i]) for i in idx]
        for a0, a1, b0, b1 in _chunks(sub, sp):
            for c0, c1, d0, d1 in _split_big(sub, sp, a0, a1, b0, b1, counts):
                srt: list[tuple[str, float, float, float]] = []
                owner: list[tuple[int, int]] = []     # (индекс в idx, номер слова)
                for k in range(c0, c1):
                    ln = sub[k]
                    toks = line_toks[idx[k]]
                    for q, tok in enumerate(toks):
                        t = ln.start + (q + 0.5) / len(toks) * (ln.end - ln.start)
                        srt.append((tok, t, ln.start, ln.end))
                        owner.append((k, q))
                for ti, tj in _dp_match(srt, sp[d0:d1]):
                    k, q = owner[ti]
                    result.setdefault(idx[k], ([], 0.0, len(line_toks[idx[k]])))[0].append(
                        _Match(sp[d0 + tj].start, sp[d0 + tj].end, q))
    for i, (ms, _, n) in list(result.items()):
        toks = line_toks[i]
        total = sum(_token_weight(t) for t in toks)
        got = sum(_token_weight(toks[m.token]) for m in ms)
        result[i] = (ms, got / total if total else 0.0, n)
    return result


# ───────────────────────── границы строки ─────────────────────────
@dataclass
class _Cand:
    start: float
    end: float
    conf: float          # 0 — тайминг SRT без изменений
    kind: str = "orig"   # orig | text | time


def _text_candidate(line: Line, ms: list[_Match], frac: float, n_tok: int) -> _Cand | None:
    if not ms or frac < TEXT_THRESHOLD or (len(ms) < 2 and n_tok > 1):
        return None
    ms = sorted(ms, key=lambda m: m.token)
    while len(ms) >= 3 and ms[1].start - ms[0].end > ORPHAN_GAP:     # слово далеко от остальных — это чужая фраза
        ms = ms[1:]
    while len(ms) >= 3 and ms[-1].start - ms[-2].end > ORPHAN_GAP:
        ms = ms[:-1]
    first, last = ms[0], ms[-1]
    start = first.start - first.token * WORD_GUESS
    end = last.end + (n_tok - 1 - last.token) * WORD_GUESS
    limit = MAX_TEXT_SHIFT if frac >= 0.9 else MID_TEXT_SHIFT if frac >= 0.75 else WEAK_TEXT_SHIFT
    # достроенные края не должны уходить за слова речи, поэтому подрезаем так, чтобы реплика не раздувалась
    start = max(start, first.start - 1.5)
    end = min(end, last.end + 1.5)
    start, end = _clamp_shift(start, line.start, limit), _clamp_shift(end, line.end, limit)
    if end <= start or end - start < 0.4 * (line.end - line.start):
        return None
    # разумная длина: не длиннее чем в 2 раза (+1 с) от исходной
    dur0 = max(line.end - line.start, MIN_DURATION)
    if end - start > 2 * dur0 + 1.0:
        mid_s, mid_e = first.start, last.end            # оставляем только то, что реально услышали
        start, end = _clamp_shift(mid_s, line.start, limit), _clamp_shift(mid_e, line.end, limit)
        if end <= start or end - start > 2 * dur0 + 1.0:
            return None
    return _Cand(start, end, 1.0 + frac, "text")


def _time_candidate(line: Line, speech: list[Line]) -> _Cand | None:
    """Только по времени: речь реально пересекается со строкой — подтягиваем края, но не дальше 1 с."""
    w0, w1 = line.start - 0.5, line.end + 0.5
    best_t: tuple[float, list[Line]] | None = None
    for group in _contiguous_groups([x for x in speech if _overlap(w0, w1, x.start, x.end) > 0]):
        a, b = _bounds(group)
        ov = _overlap(w0, w1, a, b)
        score = ov / max(min(line.end - line.start, b - a), 1e-6)
        if score >= 0.3 and (best_t is None or score > best_t[0] + 1e-9):
            best_t = (score, group)
    if best_t:
        a, b = _bounds(best_t[1])
        ns, ne = _clamp_shift(a, line.start), _clamp_shift(b, line.end)
        if ne > ns:
            return _Cand(ns, ne, 0.5, "time")
    return None


def _conflict(a: _Cand, b: _Cand, allowed: float) -> bool:
    return b.start < a.end - allowed - 1e-6 or b.start < a.start - 1e-6 or b.end < a.end - 1e-6


def _resolve(lines: list[Line], cands: list[_Cand]) -> list[_Cand]:
    """Приводит границы к правилам: порядок, отсутствие новых перекрытий.

    Перекрытие допустимо только такое, какое уже было в SRT. Конфликт решаем так: подрезаем границу менее уверенной
    стороны по соседу (если остаётся осмысленная длина), иначе возвращаем ей тайминг SRT. Откат только уменьшает число
    изменённых строк, поэтому цикл завершается.
    """
    orig = [_Cand(ln.start, ln.end, 0.0) for ln in lines]
    cur = list(cands)
    for _ in range(8 * len(cur) + 8):
        bad = next((i for i in range(1, len(cur))
                    if not (cur[i - 1].kind == "orig" and cur[i].kind == "orig")
                    and _conflict(cur[i - 1], cur[i], max(0.0, orig[i - 1].end - orig[i].start))), None)
        if bad is None:
            break
        a, b = cur[bad - 1], cur[bad]
        allowed = max(0.0, orig[bad - 1].end - orig[bad].start)
        if a.kind == "orig":
            weak = bad
        elif b.kind == "orig":
            weak = bad - 1
        else:
            weak = bad - 1 if a.conf <= b.conf else bad
        if weak == bad - 1:
            new_end = min(a.end, b.start + allowed)
            min_len = max(0.3, 0.4 * (orig[weak].end - orig[weak].start))
            if new_end - a.start >= min_len and b.end >= new_end:
                cur[weak] = _Cand(a.start, new_end, a.conf, a.kind)
            else:
                cur[weak] = orig[weak]
        else:
            new_start = max(b.start, a.end - allowed)
            min_len = max(0.3, 0.4 * (orig[weak].end - orig[weak].start))
            if b.end - new_start >= min_len and new_start >= a.start:
                cur[weak] = _Cand(new_start, b.end, b.conf, b.kind)
            else:
                cur[weak] = orig[weak]
    else:  # не сошлось — безопасный выход: всё, что ещё конфликтует, возвращаем к SRT
        for i in range(1, len(cur)):
            if _conflict(cur[i - 1], cur[i], max(0.0, orig[i - 1].end - orig[i].start)):
                cur[i - 1], cur[i] = orig[i - 1], orig[i]
    return cur


def align_embedded_to_speech(lines: list[Line], speech: list[Line]) -> list[Line]:
    """Сдвинуть start/end встроенных субтитров к словам whisper.

    - Слова субтитров и речи сопоставляются одним монотонным проходом (каждое слово речи — максимум одной строке).
      Если совпало достаточно слов (>= 0.5 по весу) — границы берутся по словам речи (сдвиг не больше 3 с, при слабом совпадении — 1,5 с).
    - Иначе, если речь пересекается по времени: подтягиваем края к речи, сдвиг не больше 1 с.
    - Иначе строка остаётся как есть. Порядок и число строк не меняются, текст/speaker_id тоже.
    - Новых перекрытий соседних строк нет; если уверенности нет — тайминг SRT.
    """
    if not lines or not speech:
        return [ln.model_copy() for ln in lines]
    sp = sorted(speech, key=lambda x: (x.start, x.end))
    order = sorted(range(len(lines)), key=lambda i: (lines[i].start, lines[i].end))
    sorted_lines = [lines[i] for i in order]
    tm = _text_matches(sorted_lines, sp)
    cands: list[_Cand] = []
    for k, ln in enumerate(sorted_lines):
        c = None
        if k in tm:
            ms, frac, n_tok = tm[k]
            c = _text_candidate(ln, ms, frac, n_tok)
        if c is None:
            c = _time_candidate(ln, sp)
        cands.append(c or _Cand(ln.start, ln.end, 0.0))
    cands = _resolve(sorted_lines, cands)
    out: list[Line | None] = [None] * len(lines)
    for pos, i in enumerate(order):
        c = cands[pos]
        out[i] = lines[i].model_copy() if c.kind == "orig" else _finish(lines[i], c.start, c.end)
    return out  # type: ignore[return-value]
