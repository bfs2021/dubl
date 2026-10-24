import { describe, expect, it } from "vitest";
import { BUDGET_LABELS, countSyllables, detectLang, fmtRate, syllableBudget, syllablesEn, syllablesRu } from "./syllables";

describe("слоги: русский — по гласным", () => {
  it("считает гласные в слове", () => {
    expect(syllablesRu("молоко")).toBe(3);
    expect(syllablesRu("ёлка")).toBe(2);
    expect(syllablesRu("встреча")).toBe(2);
    expect(syllablesRu("в")).toBe(0);
  });
  it("в тексте суммирует слова и игнорирует знаки", () => {
    expect(countSyllables("Улыбаемся и машем, парни!", "ru")).toBe(5 + 1 + 2 + 2);
  });
});

describe("слоги: английский — группы гласных с поправкой на немое e", () => {
  it.each([
    ["the", 1], ["make", 1], ["time", 1], ["love", 1], ["table", 2], ["people", 2],
    ["asked", 1], ["wanted", 2], ["hello", 2], ["beautiful", 3], ["every", 3], ["makes", 1], ["boxes", 2], ["you", 1], ["my", 1],
  ])("%s → %i", (w, n) => {
    // every: e-v-e-r-y → 3 группы (e, e, y); словарь не нужен — приемлемая точность для бюджета
    expect(syllablesEn(w)).toBe(n);
  });
  it("слово без гласных считаем за один слог", () => {
    expect(syllablesEn("hmm")).toBe(1);
    expect(syllablesEn("")).toBe(0);
  });
  it("апостроф не мешает", () => {
    expect(syllablesEn("don't")).toBe(1);
  });
});

describe("слоги: язык определяется по алфавиту слова", () => {
  it("смешанный текст: русские слова по гласным, английские по правилам", () => {
    expect(countSyllables("Привет, hello")).toBe(2 + 2);
  });
  it("цифры считаем как три слога", () => {
    expect(countSyllables("Нас 15")).toBe(1 + 3);
  });
  it("detectLang", () => {
    expect(detectLang("Are you hurt?")).toBe("en");
    expect(detectLang("Тебе больно?")).toBe("ru");
    expect(detectLang("")).toBe("ru");
  });
});

describe("бюджет реплики", () => {
  it("«влезет»: мало слогов на длинную реплику", () => {
    const b = syllableBudget("Тебе больно?", "Are you hurt?", 2);
    expect(b.mySyllables).toBe(4);
    expect(b.origSyllables).toBe(3);
    expect(b.level).toBe("ok");
    expect(b.label).toBe("влезет");
    expect(b.trim).toBe(0);
  });
  it("«придётся торопиться»: от 6,5 до 8,5 слогов в секунду", () => {
    const b = syllableBudget("Шкипер может сказать им что топлива нет", "x", 1.5); // 12 слогов / 1,5 с = 8
    expect(b.rate).toBeCloseTo(8, 1);
    expect(b.level).toBe("hurry");
    expect(b.label).toBe("придётся торопиться");
    expect(b.trim).toBe(12 - Math.floor(1.5 * 6.5));
  });
  it("«не влезет»: быстрее 8,5 слогов в секунду", () => {
    const b = syllableBudget("Нет улыбаемся и машем парни улыбаемся и машем", "x", 1.5);
    expect(b.level).toBe("over");
    expect(b.label).toBe("не влезет");
    expect(b.trim).toBeGreaterThan(0);
  });
  it("пустой текст — отдельное состояние, без деления на ноль", () => {
    const b = syllableBudget("   ", "Hi", 0);
    expect(b.level).toBe("empty");
    expect(Number.isFinite(b.rate)).toBe(true);
    expect(BUDGET_LABELS.empty).toBe("текста нет");
  });
  it("fmtRate", () => {
    expect(fmtRate(5.25)).toBe("5,3 сл/с");
  });
});
