/** Работа с записью дубля: нарезка длинной записи на реплики и волна для списка. */
import { envelope } from "./analyze";

/**
 * Вырезать из записи кусок [from, to] (секунды сцены). `fileT0` — время сцены, которому соответствует
 * нулевой сэмпл записи (start_rel − preroll). Возвращает пустой массив, если кусок вне записи.
 */
export function sliceByScene(samples: Float32Array, sampleRate: number, fileT0: number, from: number, to: number): Float32Array {
  const a = Math.max(0, Math.floor((from - fileT0) * sampleRate));
  const b = Math.min(samples.length, Math.ceil((to - fileT0) * sampleRate));
  return b > a ? samples.slice(a, b) : new Float32Array(0);
}

export interface LineWindow { line_id: string; start_rel: number; end_rel: number }

/** Окно записи вокруг реплики: запас перед (0.5 с) и после (0.6 с), не заходя за начало сцены. */
export function takeWindow(l: LineWindow, sceneDur: number): [number, number] {
  return [Math.max(0, l.start_rel - 0.5), Math.min(sceneDur, l.end_rel + 0.6)];
}

export const peaksOf = (s: Float32Array, bins = 90) => {
  const e = envelope(s, bins);
  const mx = Math.max(1e-6, ...e);
  return e.map((v) => v / mx);
};
