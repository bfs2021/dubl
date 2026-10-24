/**
 * Разбор дубля по тексту: что человек сказал против сценария (считает движок по Take.text_match / complete / tail_*).
 * Здесь только слова для интерфейса. Поля, которых у старого движка нет, читаем как «не знаем» (undefined/null).
 */
import type { Take } from "../api/types";

/** Поля разбора, которые добавляют агенты движка; если в схеме их ещё нет — читаем опционально. */
export interface TakeExtra {
  source?: "mic" | "file";
  empty?: boolean;
  text_match?: number | null;
  complete?: boolean | null;
  tail_cut_s?: number | null;
  tail_text?: string | null;
  chosen_by_user?: boolean;
  /** доля перегруженных отсчётов, %; > 0,05 — «записано с перегрузом» (считает движок) */
  clip_pct?: number | null;
}
export type TakeX = Take & TakeExtra;

export type ReviewKind = "full" | "missing" | "tail" | "partial" | "empty" | "unknown";

export interface TextReview {
  kind: ReviewKind;
  /** короткая строка под дублем */
  text: string;
  /** подсветить как замечание */
  warn: boolean;
}

const join = (ws: string[]) => ws.slice(0, 4).map((w) => `«${w}»`).join(", ");

/**
 * Главная строка разбора:
 *  «Всё сказал ✓» / «Пропущены слова: …» / «В конце повтор — отрезать?» / «Запись почти пустая».
 * missing — слова из сценария, которых нет в записи (из отчёта помощника; может быть пусто).
 */
export function textReview(t: TakeX, missing: string[] = []): TextReview {
  if (t.empty) return { kind: "empty", text: "Запись почти пустая — в ролик не пойдёт", warn: true };
  if (t.tail_cut_s != null) {
    return { kind: "tail", text: t.tail_text ? `В конце повтор («${t.tail_text.slice(0, 40)}») — отрезать?` : "В конце повтор — отрезать?", warn: true };
  }
  if (t.complete === true) return { kind: "full", text: "Всё сказал ✓", warn: false };
  if (t.complete === false) {
    if (missing.length) return { kind: "missing", text: `Пропущены слова: ${join(missing)}${missing.length > 4 ? "…" : ""}`, warn: true };
    const pct = t.text_match != null ? Math.round(t.text_match * 100) : null;
    return { kind: "partial", text: pct != null ? `Сказано примерно ${pct}% текста` : "Сказал не всё", warn: true };
  }
  return { kind: "unknown", text: "", warn: false };
}

/** Дубли реплики, из которых выбираем лучший: пустые не считаем. */
const usable = (ts: TakeX[]) => ts.filter((x) => !x.empty);

/**
 * Почему этот дубль со звёздочкой. Пустая строка — дубль не отмечен.
 *  - выбрал сам человек → «Выбран тобой»;
 *  - единственный годный → «Единственный годный дубль»;
 *  - самый полный → «Самый полный: сказано N% текста»;
 *  - иначе → «Последний записанный».
 */
export function starReason(t: TakeX, siblings: TakeX[]): string {
  if (!t.selected) return "";
  if (t.empty) return "";
  if (t.chosen_by_user) return "Выбран тобой";
  const ok = usable(siblings);
  if (ok.length <= 1) return "Единственный годный дубль";
  const scored = ok.filter((x) => x.text_match != null);
  if (scored.length >= 2) {
    const best = Math.max(...scored.map((x) => x.text_match as number));
    if ((t.text_match ?? -1) >= best - 1e-9) {
      const worse = scored.some((x) => (x.text_match as number) < best - 0.01);
      return worse ? `Самый полный: сказано ${Math.round(best * 100)}% текста` : "Все дубли одинаково полные — взят лучший по звуку";
    }
  }
  return "Взят последний записанный";
}

/** Порог перегруза движка: доля срезанных отсчётов выше 0,05 % (audio.levels). */
export const CLIP_PCT = 0.05;

/** Дубль с перегрузом? clip_pct считает движок; у старого движка поля нет — смотрим на пик. */
export function isClipped(t: TakeX): boolean {
  if (t.clip_pct != null) return t.clip_pct > CLIP_PCT;
  return t.peak_db != null && t.peak_db > -0.3;
}

export const CLIPPED_TEXT = "Записано с перегрузом, подальше от микрофона — перегруз починили, но лучше переписать.";
export const EMPTY_TEXT = "Запись почти пустая. Проверь, что выбран нужный микрофон, и запиши ещё раз.";
export const FILE_TEXT = "Загружен файлом: попадание в губы не проверяем.";

/** Откуда дубль. Старый движок поля не знает — считаем записью с микрофона. */
export const isFileTake = (t: TakeX) => t.source === "file";

/** Советы про губы и вступление не нужны ни для файла, ни для закадрового режима. */
export const TIMING_REMARK_RE = /губ|вступ|рано|поздн|раньше|позже|сдвин|опаздыва|отстаёшь|забегаешь|спешишь/i;

/**
 * Дубль в ролик не идёт: пустой (движок сам не ставит ему ★). Нужен, чтобы не предлагать «К рендеру»,
 * когда годных дублей нет.
 */
export const takeIsUsable = (t: TakeX) => !t.empty;

export interface RemarkLike { text: string; more?: string; severity: "ok" | "warn" }

/**
 * Совет под дублем. Про губы и вступление не говорим, если это закадровый режим или дубль из файла
 * (у файла нет привязки ко времени). undefined — совет прячем.
 */
export function visibleRemark<T extends RemarkLike>(raw: T | undefined, opts: { lips: boolean; file: boolean }): T | undefined {
  if (!raw) return raw;
  if ((!opts.lips || opts.file) && TIMING_REMARK_RE.test(raw.text)) return undefined;
  return raw;
}
