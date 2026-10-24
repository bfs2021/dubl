/**
 * Привязка выделения на шкале — копия engine/analysis/snap.py.
 * Нужна, чтобы выделение реагировало мгновенно, без запроса к движку.
 * Поведение должно совпадать 1:1 (тесты в snap.test.ts повторяют pytest движка).
 */
import type { Line } from "../api/types";

export const PAD = 0.15; // запас вокруг реплики, с

type Span = [number, number];

function spanOf(line: Line): Span {
  let s = line.start;
  let e = line.end;
  if (line.words.length) {
    s = Math.min(s, ...line.words.map((w) => w.start));
    e = Math.max(e, ...line.words.map((w) => w.end));
  }
  return [s, e];
}

const cache = new WeakMap<Line[], Span[]>();

/** Отсортированные границы реплик (с кэшем по массиву). */
export function spansOf(lines: Line[]): Span[] {
  let r = cache.get(lines);
  if (!r) {
    r = lines.map(spanOf).sort((x, y) => x[0] - y[0] || x[1] - y[1]);
    cache.set(lines, r);
  }
  return r;
}

function strictlyInside(t: number, spans: Span[]): boolean {
  return spans.some(([s, e]) => s < t && t < e);
}

function expandLeft(a: number, spans: Span[], pad: number): number {
  const inside = spans.filter(([s, e]) => s <= a && a < e);
  if (!inside.length) return a;
  const start = Math.min(...inside.map(([s]) => s));
  let nw = start - pad;
  const prevEnds = spans.filter(([, e]) => e <= start).map(([, e]) => e);
  if (prevEnds.length) {
    const pe = Math.max(...prevEnds);
    if (pe > nw) nw = Math.max(nw, (pe + start) / 2);
  }
  return Math.max(nw, 0);
}

function expandRight(b: number, spans: Span[], pad: number): number {
  const inside = spans.filter(([s, e]) => s < b && b <= e);
  if (!inside.length) return b;
  const end = Math.max(...inside.map(([, e]) => e));
  let nw = end + pad;
  const nextStarts = spans.filter(([s]) => s >= end).map(([s]) => s);
  if (nextStarts.length) {
    const ns = Math.min(...nextStarts);
    if (ns < nw) nw = (end + ns) / 2;
  }
  return nw;
}

function nearestCut(t: number, cuts: number[], tolerance: number, spans: Span[]): number | null {
  let best: number | null = null;
  for (const c of cuts) {
    if (Math.abs(c - t) <= tolerance && !strictlyInside(c, spans)) {
      if (best === null || Math.abs(c - t) < Math.abs(best - t)) best = c;
    }
  }
  return best;
}

/**
 * Подогнать выделение [a, b]:
 * 1. край внутри реплики уезжает на её границу (+0.15 с запаса, не заходя на соседнюю);
 * 2. край рядом (≤ tolerance) со склейкой прилипает к ней, если склейка не режет реплику.
 */
export function snapSelection(a: number, b: number, lines: Line[], cuts: number[], tolerance = 0.6): [number, number] {
  if (a > b) [a, b] = [b, a];
  const spans = spansOf(lines);
  const na = expandLeft(a, spans, PAD);
  const nb = expandRight(b, spans, PAD);
  const ca = nearestCut(na, cuts, tolerance, spans);
  const cb = nearestCut(nb, cuts, tolerance, spans);
  let sa = ca ?? na;
  let sb = cb ?? nb;
  if (sb <= sa) {
    sa = na;
    sb = nb;
  }
  return [Math.max(sa, 0), sb];
}

/** Разговор вокруг t: цепочка реплик, где паузы между соседями ≤ maxGap. */
export function dialogueAround(t: number, lines: Line[], maxGap = 2.5): [number, number] {
  const spans = spansOf(lines);
  if (!spans.length) return [t, t];
  let idx = spans.findIndex(([s, e]) => s <= t && t <= e);
  if (idx < 0) {
    let best = 0;
    let bestD = Infinity;
    spans.forEach(([s, e], i) => {
      const d = Math.min(Math.abs(t - s), Math.abs(t - e));
      if (d < bestD) {
        bestD = d;
        best = i;
      }
    });
    if (bestD > maxGap) return [t, t];
    idx = best;
  }
  let [start, end] = spans[idx];
  for (let i = idx - 1; i >= 0; i--) {
    if (start - spans[i][1] <= maxGap) start = Math.min(start, spans[i][0]);
    else break;
  }
  for (let i = idx + 1; i < spans.length; i++) {
    if (spans[i][0] - end <= maxGap) end = Math.max(end, spans[i][1]);
    else break;
  }
  return [start, end];
}

/** Кадр от склейки до склейки, в котором лежит t. */
export function shotAround(t: number, cuts: number[], duration: number): [number, number] {
  if (duration <= 0) return [0, 0];
  t = Math.min(Math.max(t, 0), duration);
  const bounds = [...new Set(cuts.filter((c) => c > 0 && c < duration))].sort((x, y) => x - y);
  // bisect_right
  let lo = 0;
  let hi = bounds.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (t < bounds[mid]) hi = mid;
    else lo = mid + 1;
  }
  const start = lo > 0 ? bounds[lo - 1] : 0;
  const end = lo < bounds.length ? bounds[lo] : duration;
  return [start, end];
}

/** Готовый результат для режима: диалог → привязка; сцена → привязка; вручную → привязка. */
export function selectByMode(
  mode: "dialogue" | "shot" | "free",
  t: number,
  a: number,
  b: number,
  lines: Line[],
  cuts: number[],
  duration: number,
): [number, number] {
  if (mode === "dialogue") {
    const [x, y] = dialogueAround(t, lines);
    return x === y ? [x, y] : snapSelection(x, y, lines, cuts);
  }
  if (mode === "shot") {
    const [x, y] = shotAround(t, cuts, duration);
    return [x, y];
  }
  return snapSelection(a, b, lines, cuts);
}
