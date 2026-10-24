/**
 * «Выделить первый разговор» на экране «Разбор»: найти первый настоящий диалог, а не первую реплику.
 * Только фронтенд (кнопка-подсказка); привязка выделения к репликам — snap.ts (копия движка).
 */
import type { Line } from "../api/types";

/** Пауза между репликами, после которой разговор считаем законченным, с. */
export const TALK_GAP = 4;
/** Ответ другого персонажа может прозвучать и после паузы подольше (реплика «после паузы»), с. */
export const REPLY_GAP = 6;

export interface Talk { start: number; end: number; lines: Line[]; speakers: number }

/** Реплики подряд: пауза меньше maxGap, а если отвечает другой персонаж — меньше replyGap. */
export function talkGroups(lines: Line[], maxGap = TALK_GAP, replyGap = REPLY_GAP): Talk[] {
  const sorted = [...lines].sort((a, b) => a.start - b.start);
  const out: Talk[] = [];
  let cur: Line[] = [];
  let end = -Infinity;
  const flush = () => {
    if (!cur.length) return;
    out.push({ start: cur[0].start, end, lines: cur, speakers: new Set(cur.map((l) => l.speaker_id ?? "?")).size });
  };
  for (const l of sorted) {
    const prev = cur[cur.length - 1];
    const gap = l.start - end;
    const reply = !!prev && (prev.speaker_id ?? "?") !== (l.speaker_id ?? "?");
    if (prev && gap >= (reply ? replyGap : maxGap)) { flush(); cur = []; end = -Infinity; }
    cur.push(l);
    end = Math.max(end, l.end);
  }
  flush();
  return out;
}

/**
 * Первый разговор: первая группа, где говорят хотя бы двое и реплик не меньше трёх.
 * Нет такой — самая длинная группа, где говорят хотя бы двое (из равных — первая). Нет и её — первая группа.
 */
export function firstConversation(lines: Line[], maxGap = TALK_GAP, replyGap = REPLY_GAP): Talk | null {
  const groups = talkGroups(lines, maxGap, replyGap);
  if (!groups.length) return null;
  const real = groups.find((g) => g.speakers >= 2 && g.lines.length >= 3);
  if (real) return real;
  const two = groups.filter((g) => g.speakers >= 2);
  if (two.length) return two.reduce((best, g) => (g.lines.length > best.lines.length ? g : best));
  return groups[0];
}
