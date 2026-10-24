/**
 * Счётчик слогов и «бюджет» реплики: влезет ли текст в длину оригинала.
 * Русский — по гласным (сколько гласных, столько слогов). Английский — по группам гласных
 * с поправкой на немое «e» на конце («make», «time») и на «-ed» («asked»).
 * Чистые функции, без обращения к движку.
 */

const RU_VOWELS = new Set("аеёиоуыэюяАЕЁИОУЫЭЮЯ");
const WORD_RE = /[\p{L}'’]+/gu;
const DIGITS_RE = /\d+/g;

export type Lang = "ru" | "en";

const isCyr = (c: string) => /[а-яёА-ЯЁіїєІЇЄ]/.test(c);

/** Слоги русского слова: каждая гласная — слог. */
export function syllablesRu(word: string): number {
  let n = 0;
  for (const ch of word) if (RU_VOWELS.has(ch)) n++;
  return n;
}

/** Слоги английского слова: группы гласных (y считаем гласной) минус немое e и -ed. */
export function syllablesEn(word: string): number {
  const w = word.toLowerCase().replace(/['’]/g, "");
  if (!w) return 0;
  let n = (w.match(/[aeiouy]+/g) ?? []).length;
  if (n > 1) {
    // немое e: make, time, hope — но не «-le» после согласной (table, little) и не «-ee/-ie/-ye»
    if (w.endsWith("e") && !/(?:[^aeiouy]le|ee|ie|ye|ue|oe)$/.test(w)) n -= 1;
    // «asked», «played»: -ed не даёт слог, кроме «-ted», «-ded»
    else if (w.endsWith("ed") && !/(?:ted|ded)$/.test(w)) n -= 1;
    else if (/[^aeiouy]es$/.test(w) && !/(?:ses|zes|ces|ges|xes|ches|shes)$/.test(w)) n -= 1;
  }
  return Math.max(n, 1);
}

/** Какой язык у слова: кириллица — русский, остальное — английский. */
export function wordLang(word: string): Lang {
  for (const ch of word) if (isCyr(ch)) return "ru";
  return "en";
}

/** Преобладающий язык текста (по числу букв). Пустой — русский. */
export function detectLang(text: string): Lang {
  let cyr = 0;
  let lat = 0;
  for (const c of text) {
    if (isCyr(c)) cyr++;
    else if (/[a-zA-Z]/.test(c)) lat++;
  }
  return cyr >= lat ? "ru" : "en";
}

/** Цифры читаем словами: «15» — «пятнадцать»; на слух берём ~3 слога на число. */
const DIGIT_SYLLABLES = 3;

/**
 * Слоги текста. Если язык не задан, каждое слово считаем по своему алфавиту:
 * кириллица — по гласным, латиница — по английским правилам.
 */
export function countSyllables(text: string, lang?: Lang): number {
  let total = 0;
  for (const w of text.match(WORD_RE) ?? []) {
    const l = lang ?? wordLang(w);
    total += l === "ru" ? syllablesRu(w) : syllablesEn(w);
  }
  total += DIGIT_SYLLABLES * (text.match(DIGITS_RE)?.length ?? 0);
  return total;
}

// ───────────── бюджет реплики ─────────────

/** Комфортная скорость речи, слогов в секунду. Выше COMFORT — придётся торопиться, выше MAX — не влезет. */
export const RATE_COMFORT = 6.5;
export const RATE_MAX = 8.5;

export type BudgetLevel = "empty" | "ok" | "hurry" | "over";

export interface SyllableBudget {
  /** слогов в оригинале */
  origSyllables: number;
  /** слогов в твоём тексте */
  mySyllables: number;
  /** сколько секунд длится реплика в сцене */
  duration: number;
  /** слогов в секунду, если уложиться ровно в длину реплики */
  rate: number;
  /** сколько слогов влезет без спешки */
  maxComfortable: number;
  level: BudgetLevel;
  /** короткая подпись: «влезет», «придётся торопиться», «не влезет» */
  label: string;
  /** на сколько слогов надо сократить, чтобы влезло без спешки (0 — не надо) */
  trim: number;
}

export const BUDGET_LABELS: Record<BudgetLevel, string> = {
  empty: "текста нет",
  ok: "влезет",
  hurry: "придётся торопиться",
  over: "не влезет",
};

/**
 * Бюджет реплики: слоги оригинала и твоего текста против длины реплики.
 * Скорость считаем по твоему тексту: рассказать N слогов за `duration` секунд.
 */
export function syllableBudget(text: string, original: string, duration: number): SyllableBudget {
  const mine = countSyllables(text);
  const orig = countSyllables(original);
  const dur = Math.max(0.05, duration);
  const rate = mine / dur;
  const maxComfortable = Math.max(1, Math.floor(dur * RATE_COMFORT));
  const level: BudgetLevel = mine === 0 ? "empty" : rate <= RATE_COMFORT ? "ok" : rate <= RATE_MAX ? "hurry" : "over";
  return {
    origSyllables: orig, mySyllables: mine, duration, rate, maxComfortable, level,
    label: BUDGET_LABELS[level], trim: Math.max(0, mine - maxComfortable),
  };
}

/** «5,3 сл/с» — скорость речи для интерфейса. */
export const fmtRate = (r: number) => r.toFixed(1).replace(".", ",") + " сл/с";
