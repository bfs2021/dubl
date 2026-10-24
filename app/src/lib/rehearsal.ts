/**
 * Репетиция реплики: сначала слушаем оригинал с видео, потом проходим тот же кусок с отсчётом и бегущей лентой — без записи.
 * Чистая часть: какой кусок играть.
 */
export interface RehearsalPlan {
  /** с какой секунды сцены */
  from: number;
  /** до какой секунды сцены */
  to: number;
  /** сколько секунд занимает проход (для подписи) */
  length: number;
}

export type RehearsalStep = "idle" | "listen" | "count" | "speak";

/** Кусок для репетиции: от «lead» секунд до реплики и до её конца + хвост, в пределах сцены. */
export function rehearsalPlan(line: { start_rel: number; end_rel: number }, sceneDur: number, leadS: number, tail = 0.5): RehearsalPlan {
  const from = Math.max(0, line.start_rel - Math.max(0, leadS));
  const to = Math.min(sceneDur, line.end_rel + tail);
  return { from, to, length: Math.max(0, to - from) };
}

export const REHEARSAL_TEXT: Record<Exclude<RehearsalStep, "idle">, string> = {
  listen: "Репетиция · слушай оригинал",
  count: "Репетиция · приготовься",
  speak: "Репетиция · говори вслед за лентой, запись не идёт",
};
