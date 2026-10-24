/** Простые замеры дубля (в браузере, для мок-режима и индикаторов). Не LUFS — только оценка. */

export const toDb = (x: number) => (x > 1e-9 ? 20 * Math.log10(x) : -120);

export function peakDb(s: Float32Array): number {
  let p = 0;
  for (let i = 0; i < s.length; i++) {
    const a = Math.abs(s[i]);
    if (a > p) p = a;
  }
  return toDb(p);
}

export function rmsDb(s: Float32Array): number {
  if (!s.length) return -120;
  let acc = 0;
  for (let i = 0; i < s.length; i++) acc += s[i] * s[i];
  return toDb(Math.sqrt(acc / s.length));
}

/** Огибающая (RMS по окнам) → массив для волны. */
export function envelope(s: Float32Array, bins: number): number[] {
  const out: number[] = [];
  const step = Math.max(1, Math.floor(s.length / bins));
  for (let b = 0; b < bins; b++) {
    let m = 0;
    const from = b * step;
    for (let i = from; i < Math.min(s.length, from + step); i++) m = Math.max(m, Math.abs(s[i]));
    out.push(m);
  }
  return out;
}

/** Где в записи начинается и заканчивается речь (секунды). null — тишина. */
export function speechBounds(s: Float32Array, sr: number, thresholdDb = -42): [number, number] | null {
  const win = Math.max(1, Math.floor(sr * 0.02));
  const thr = Math.pow(10, thresholdDb / 20);
  let first = -1;
  let last = -1;
  for (let i = 0; i + win <= s.length; i += win) {
    let acc = 0;
    for (let j = i; j < i + win; j++) acc += s[j] * s[j];
    if (Math.sqrt(acc / win) > thr) {
      if (first < 0) first = i;
      last = i + win;
    }
  }
  return first < 0 ? null : [first / sr, last / sr];
}
