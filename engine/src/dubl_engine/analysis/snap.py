"""Привязка выделения на шкале: края к репликам и склейкам. Чистые функции."""
from __future__ import annotations

from bisect import bisect_right

from ..models import Line

PAD = 0.15  # запас вокруг реплики, с


def _span(line: Line) -> tuple[float, float]:
    """Границы реплики с учётом слов (слово не должно торчать за реплику)."""
    s, e = line.start, line.end
    if line.words:
        s = min(s, min(w.start for w in line.words))
        e = max(e, max(w.end for w in line.words))
    return s, e


def _spans(lines: list[Line]) -> list[tuple[float, float]]:
    return sorted(_span(ln) for ln in lines)


def _strictly_inside(t: float, spans: list[tuple[float, float]]) -> bool:
    return any(s < t < e for s, e in spans)


def _expand_left(a: float, spans: list[tuple[float, float]], pad: float) -> float:
    """Левый край внутри реплики → её начало минус запас (но не заходя на предыдущую реплику)."""
    inside = [(s, e) for s, e in spans if s <= a < e]
    if not inside:
        return a
    start = min(s for s, _ in inside)
    new = start - pad
    prev_ends = [e for s, e in spans if e <= start]
    if prev_ends:
        new = max(new, (max(prev_ends) + start) / 2) if max(prev_ends) > new else new
    return max(new, 0.0)


def _expand_right(b: float, spans: list[tuple[float, float]], pad: float) -> float:
    inside = [(s, e) for s, e in spans if s < b <= e]
    if not inside:
        return b
    end = max(e for _, e in inside)
    new = end + pad
    next_starts = [s for s, _ in spans if s >= end]
    if next_starts and min(next_starts) < new:
        new = (end + min(next_starts)) / 2
    return new


def _nearest_cut(t: float, cuts: list[float], tolerance: float, spans: list[tuple[float, float]]) -> float | None:
    best: float | None = None
    for c in cuts:
        if abs(c - t) <= tolerance and not _strictly_inside(c, spans) and (best is None or abs(c - t) < abs(best - t)):
            best = c
    return best


def snap_selection(a: float, b: float, lines: list[Line], cuts: list[float], *,
                   tolerance: float = 0.6) -> tuple[float, float]:
    """Подогнать выделение [a, b].

    1. Край внутри реплики уезжает на её границу (+0.15 с запаса, не заходя на соседнюю реплику).
    2. Край в пределах `tolerance` от склейки прилипает к ней — если склейка не режет реплику посередине.
    Если a > b, они меняются местами. Результат всегда start <= end.
    """
    if a > b:
        a, b = b, a
    spans = _spans(lines)
    na = _expand_left(a, spans, PAD)
    nb = _expand_right(b, spans, PAD)

    ca = _nearest_cut(na, cuts, tolerance, spans)
    cb = _nearest_cut(nb, cuts, tolerance, spans)
    sa = ca if ca is not None else na
    sb = cb if cb is not None else nb
    if sb <= sa:  # склейки схлопнули выделение — оставляем без прилипания
        sa, sb = na, nb
    return max(sa, 0.0), sb


def dialogue_around(t: float, lines: list[Line], *, max_gap: float = 2.5) -> tuple[float, float]:
    """Разговор вокруг момента t: цепочка реплик, где паузы между соседями <= max_gap.

    Если t не попадает в реплику, берётся ближайшая в пределах max_gap; иначе (t, t).
    """
    spans = _spans(lines)
    if not spans:
        return t, t
    idx = next((i for i, (s, e) in enumerate(spans) if s <= t <= e), None)
    if idx is None:
        dist = [min(abs(t - s), abs(t - e)) for s, e in spans]
        best = min(range(len(spans)), key=lambda i: dist[i])
        if dist[best] > max_gap:
            return t, t
        idx = best
    start, end = spans[idx]
    # влево
    for i in range(idx - 1, -1, -1):
        if start - spans[i][1] <= max_gap:
            start = min(start, spans[i][0])
        else:
            break
    # вправо
    for i in range(idx + 1, len(spans)):
        if spans[i][0] - end <= max_gap:
            end = max(end, spans[i][1])
        else:
            break
    return start, end


def shot_around(t: float, cuts: list[float], duration: float) -> tuple[float, float]:
    """Кадр от склейки до склейки, в котором лежит t."""
    if duration <= 0:
        return 0.0, 0.0
    t = min(max(t, 0.0), duration)
    bounds = sorted({c for c in cuts if 0.0 < c < duration})
    i = bisect_right(bounds, t)
    start = bounds[i - 1] if i > 0 else 0.0
    end = bounds[i] if i < len(bounds) else duration
    return start, end
