/**
 * Данные демо-режима: мультсериал «Барсик и Мила» (выдуманный). Реплики — английский оригинал + русский перевод
 * (перевод отдаёт мок «ИИ» по кнопке «Перевести»). Всё детерминировано.
 */
import type { Analysis, Chapter, Line, Speaker, Word } from "./types";
import { estimateDuration } from "../lib/estimate";
import { rng } from "../lib/rng";

export const SPEAKER_DEFS: { id: string; name: string | null; label: string; color: string }[] = [
  { id: "s1", name: "Барсик", label: "Барсик", color: "#4fb8a8" },
  { id: "s2", name: "Мила", label: "Мила", color: "#e57a5f" },
  { id: "s3", name: "Гена", label: "Гена", color: "#6aa6e8" },
  { id: "s4", name: "Пипс", label: "Пипс", color: "#c58ad6" },
  { id: "s5", name: null, label: "Голос 5", color: "#a8c95a" },
  { id: "s6", name: null, label: "Голос 6", color: "#e9b34c" },
];

type Row = [spk: number, en: string, ru: string];
export interface Cluster { title: string; rows: Row[] }

export const CLUSTERS: Cluster[] = [
  { title: "Холодильник открыт", rows: [
    [1, "Barsik, you have mustard on your whiskers.", "Барсик, у тебя горчица на усах."],
    [0, "That's not mustard. That's... evidence.", "Это не горчица. Это... улика."],
    [1, "Evidence of what, exactly?", "Улика чего, интересно?"],
    [0, "Of a perfectly innocent sandwich.", "Абсолютно невинного бутерброда."],
    [2, "Who opened the fridge?!", "Кто открыл холодильник?!"],
    [3, "Fridge door open for forty-two seconds. Temperature rising.", "Дверца открыта сорок две секунды. Температура растёт."],
    [5, "Run!", "Бежим!"],
  ] },
  { title: "Блинчики", rows: [
    [1, "Good morning, everyone! Who wants pancakes?", "Всем доброе утро! Кто хочет блинчиков?"],
    [0, "Me. Obviously. Always me.", "Я. Очевидно. Всегда я."],
    [2, "Careful, the pan is hot.", "Осторожно, сковородка горячая."],
    [3, "Pancake count: eleven. Eating speed: alarming.", "Блинов: одиннадцать. Скорость поедания: тревожная."],
    [0, "I'm a growing cat.", "Я растущий кот."],
  ] },
  { title: "Письмо без марки", rows: [
    [2, "Look at this letter. It has no stamp.", "Посмотри на это письмо. На нём нет марки."],
    [1, "Open it! What does it say?", "Открой! Что там написано?"],
    [2, "Meet me at the old lighthouse at midnight.", "Приходи к старому маяку в полночь."],
    [0, "That's either a trap or a very romantic invitation.", "Это либо ловушка, либо очень романтичное приглашение."],
    [3, "Probability of trap: ninety-four percent.", "Вероятность ловушки: девяносто четыре процента."],
  ] },
  { title: "Голуби в парке", rows: [
    [1, "Barsik, stop chasing the pigeons!", "Барсик, перестань гоняться за голубями!"],
    [0, "They started it!", "Они первые начали!"],
    [4, "Young man, this is a public park.", "Молодой человек, это общественный парк."],
    [0, "Then the pigeons are public enemies.", "Значит, голуби — враги общества."],
    [1, "Sorry, ma'am. He's just... enthusiastic.", "Извините, мадам. Он просто... увлечённый."],
  ] },
  { title: "Гараж", rows: [
    [2, "Hand me the wrench. No, the other wrench.", "Подай ключ. Нет, другой ключ."],
    [3, "All wrenches are identical. Statistically.", "Все ключи одинаковые. Статистически."],
    [2, "Statistically, you're getting recycled.", "Статистически, тебя сдадут в переработку."],
    [1, "Boys, the engine is smoking.", "Мальчики, мотор дымится."],
    [5, "Everybody out!", "Все наружу!"],
  ] },
  { title: "Ночь на маяке", rows: [
    [0, "It's very dark in here.", "Здесь очень темно."],
    [1, "You're a cat. You see in the dark.", "Ты кот. Ты видишь в темноте."],
    [0, "I see dark. Lots of dark.", "Я вижу темноту. Очень много темноты."],
    [2, "Shh. Someone is coming up the stairs.", "Тс-с. Кто-то поднимается по лестнице."],
    [4, "Good evening. You're late.", "Добрый вечер. Вы опоздали."],
    [3, "Unexpected. Recalculating.", "Неожиданно. Пересчитываю."],
  ] },
  { title: "В магазине", rows: [
    [1, "We need milk, bread and — Barsik, put that back.", "Нам нужны молоко, хлеб и... Барсик, положи на место."],
    [0, "It jumped into the basket on its own.", "Оно само прыгнуло в корзину."],
    [2, "A whole smoked fish jumped by itself?", "Целая копчёная рыба прыгнула сама?"],
    [0, "It was a very athletic fish.", "Это была очень спортивная рыба."],
  ] },
  { title: "Дождь", rows: [
    [3, "Warning. Rain detected. I am not waterproof.", "Внимание. Обнаружен дождь. Я не водонепроницаем."],
    [1, "Come under the umbrella, Pips.", "Иди под зонт, Пипс."],
    [3, "Thank you. Your kindness is noted.", "Спасибо. Ваша доброта отмечена."],
    [0, "Nobody offers me an umbrella.", "Мне зонт никто не предлагает."],
    [1, "You have fur, Barsik.", "У тебя шерсть, Барсик."],
  ] },
  { title: "Чердак", rows: [
    [2, "What's in the box?", "Что в коробке?"],
    [1, "Old photographs. And... a key.", "Старые фотографии. И... ключ."],
    [0, "A key to what? The fridge?", "Ключ от чего? От холодильника?"],
    [2, "Everything is about the fridge with you.", "У тебя всё про холодильник."],
  ] },
  { title: "Звонок", rows: [
    [1, "Hello? Hello! Who is this?", "Алло? Алло! Кто это?"],
    [4, "I know what you did at the lighthouse.", "Я знаю, что вы сделали на маяке."],
    [1, "We didn't do anything!", "Мы ничего не делали!"],
    [0, "I did a little bit.", "Я чуть-чуть делал."],
  ] },
  { title: "Финал", rows: [
    [1, "We did it. The mystery is solved.", "Мы справились. Загадка разгадана."],
    [2, "And nobody got hurt. Except my pride.", "И никто не пострадал. Кроме моей гордости."],
    [0, "Can we go home now? I'm hungry.", "Мы можем уже домой? Я голоден."],
    [3, "Hunger level: critical. Pancakes recommended.", "Уровень голода: критический. Рекомендованы блины."],
    [1, "To the kitchen!", "На кухню!"],
  ] },
];

/** Русский перевод по английскому тексту (для кнопки «Перевести» в демо). */
export const RU_BY_EN: Record<string, string> = Object.fromEntries(CLUSTERS.flatMap((c) => c.rows.map((r) => [r[1], r[2]])));

export interface Plan { cluster: number; at: number }

export const FRIDGE_AT = 723.4; // 12:03.4

/** Реплики и склейки видео по плану: какой кусок диалога в какую секунду. */
export function buildAnalysis(duration: number, plan: Plan[], seed: number, audioStreamIndex = 1): Analysis {
  const r = rng(seed);
  const lines: Line[] = [];
  let n = 1;
  for (const p of plan) {
    const c = CLUSTERS[p.cluster];
    let cursor = p.at;
    c.rows.forEach(([spk, en], i) => {
      const dur = Math.max(1.0, estimateDuration(en, "en") + 0.25);
      const start = Math.round(cursor * 100) / 100;
      const end = Math.round((cursor + dur) * 100) / 100;
      const parts = en.split(/\s+/);
      const total = parts.reduce((a, w) => a + w.length + 1, 0);
      let wc = start + 0.05;
      const span = end - start - 0.1;
      const words: Word[] = parts.map((w) => {
        const d = ((w.length + 1) / total) * span;
        const word = { start: Math.round(wc * 100) / 100, end: Math.round((wc + d * 0.92) * 100) / 100, text: w, prob: 0.9 };
        wc += d;
        return word;
      });
      lines.push({
        id: `l${String(n++).padStart(4, "0")}`, start, end, text: en, speaker_id: SPEAKER_DEFS[spk].id,
        words, source: "embedded", overlap: p.cluster === 0 && i === 6,
      });
      cursor = end + 0.35 + r() * 0.55;
    });
  }
  lines.sort((a, b) => a.start - b.start);

  // склейки: каждые 2.5–11 с, внутри реплик почти не бывает; в «Холодильнике» — заданные
  const cuts: number[] = [];
  let t = 2 + r() * 3;
  while (t < duration - 1) {
    cuts.push(Math.round(t * 100) / 100);
    t += 2.5 + r() * 8.5;
  }
  const fridge = plan.find((p) => p.cluster === 0);
  const forced = fridge ? [fridge.at - 0.6, fridge.at + 4.9, fridge.at + 9.7, fridge.at + 14.8, fridge.at + 21.9] : [];
  const keep = cuts.filter((c) => !fridge || c < fridge.at - 3 || c > fridge.at + 25);
  const shot_cuts = [...keep, ...forced.map((x) => Math.round(x * 100) / 100)].sort((a, b) => a - b);

  const chapters: Chapter[] = [];
  if (duration > 600) {
    const names = ["Утро", "Письмо", "Парк", "Гараж", "Маяк", "Холодильник", "Дождь", "Финал"];
    const step = duration / names.length;
    names.forEach((title, i) => chapters.push({ start: Math.round(i * step), end: Math.round((i + 1) * step), title }));
  }

  const speakers: Speaker[] = SPEAKER_DEFS.map((d) => {
    const mine = lines.filter((l) => l.speaker_id === d.id);
    return {
      id: d.id, label: d.label, name: d.name, color: d.color, line_count: mine.length,
      total_speech: Math.round(mine.reduce((a, l) => a + (l.end - l.start), 0) * 10) / 10,
    };
  }).filter((s) => s.line_count > 0);

  const merge_suggestions: [string, string, number][] = speakers.some((s) => s.id === "s6") ? [["s1", "s6", 0.87]] : [];
  return { language: "en", audio_stream_index: audioStreamIndex, lines, speakers, shot_cuts, chapters, merge_suggestions, merged_lines: {}, asr_notes: [] };
}

export function planFor(kind: "series" | "short" | "next"): Plan[] {
  if (kind === "series") {
    const at = [14, 61, 118, 190, 262, 355, 441, 527, 640, FRIDGE_AT, 880, 1010, 1130, 1262, 1330];
    const order = [1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 10, 2, 3, 5, 10];
    // повторов текста стараемся не делать: порядок выше уникален, кроме хвоста
    return at.map((a, i) => ({ cluster: order[i], at: a }));
  }
  if (kind === "short") return [{ cluster: 6, at: 9 }, { cluster: 7, at: 72 }, { cluster: 3, at: 131 }, { cluster: 10, at: 196 }];
  return [{ cluster: 1, at: 11 }, { cluster: 8, at: 90 }, { cluster: 9, at: 260 }];
}
