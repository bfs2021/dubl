/**
 * Оценка длины речи без ИИ — порт engine/llm/tasks.py (estimate_duration):
 * слоги × темп + паузы на знаках препинания. Нужна для полоски «влезает ли в губы».
 */
const VOWELS_RU = new Set("аеёиоуыэюяАЕЁИОУЫЭЮЯ");
const VOWELS_LAT = new Set("aeiouyAEIOUY");
const VOWELS_OTHER = new Set("àáâãäåæèéêëìíîïòóôõöøùúûüýœÀÁÂÃÄÅÆÈÉÊËÌÍÎÏÒÓÔÕÖØÙÚÛÜÝŒіїєІЇЄ");
const WORD_RE = /[\p{L}]+/gu;
const NUM_RE = /\d+/g;

const SYLLABLE_S: Record<string, number> = { ru: 0.2, en: 0.19 };
const PAUSE_SOFT = new Set(",;:—–");
const PAUSE_HARD = new Set(".!?…");

function enSyllables(word: string): number {
  const w = word.toLowerCase();
  let n = (w.match(/[aeiouy]+/g) ?? []).length;
  if (n > 1 && w.endsWith("e") && !/(le|ee|ie|ye)$/.test(w)) n -= 1;
  if (w.endsWith("ed") && n > 1 && !/(ted|ded)$/.test(w)) n -= 1;
  return Math.max(n, 1);
}

export function countSyllables(text: string, lang = "ru"): number {
  lang = (lang || "ru").toLowerCase().split("-")[0];
  let total = 0;
  for (const word of text.match(WORD_RE) ?? []) {
    if (lang === "en") total += enSyllables(word);
    else for (const ch of word) if (VOWELS_RU.has(ch) || VOWELS_LAT.has(ch) || VOWELS_OTHER.has(ch)) total += 1;
  }
  total += 3 * (text.match(NUM_RE)?.length ?? 0);
  return total;
}

export function guessLang(text: string): "ru" | "en" {
  let cyr = 0;
  let lat = 0;
  for (const c of text.toLowerCase()) {
    if ((c >= "а" && c <= "я") || "ёіїє".includes(c)) cyr++;
    else if (c >= "a" && c <= "z") lat++;
  }
  return cyr >= lat ? "ru" : "en";
}

export function estimateDuration(text: string, lang = "ru"): number {
  lang = (lang || "ru").toLowerCase().split("-")[0];
  const rate = SYLLABLE_S[lang] ?? SYLLABLE_S.ru;
  let t = countSyllables(text, lang) * rate;
  for (const ch of text.replace(/\.\.\./g, "…")) {
    if (PAUSE_HARD.has(ch)) t += 0.25;
    else if (PAUSE_SOFT.has(ch)) t += 0.15;
  }
  return Math.round(t * 1000) / 1000;
}

export type FitClass = "short" | "ok" | "warn" | "bad";
export interface FitInfo { cls: FitClass; label: string; ratio: number; est: number }

/** Влезает ли текст в длину реплики. */
export function fitInfo(text: string, duration: number, lang?: string): FitInfo {
  const est = estimateDuration(text, lang ?? guessLang(text));
  const ratio = duration > 0 ? est / duration : 2;
  if (ratio < 0.78) return { cls: "short", label: "коротко — можно тянуть", ratio, est };
  if (ratio <= 1.12) return { cls: "ok", label: "попадает в движения губ", ratio, est };
  if (ratio <= 1.3) return { cls: "warn", label: "длинновато", ratio, est };
  return { cls: "bad", label: "не влезет", ratio, est };
}
