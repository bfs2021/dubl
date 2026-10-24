import { describe, expect, it } from "vitest";
import type { Take } from "../api/types";
import { CLIPPED_TEXT, TIMING_REMARK_RE, isClipped, isFileTake, starReason, textReview, type TakeX } from "./takeReview";

const take = (o: Partial<TakeX> = {}): TakeX => ({
  id: "t1", line_id: "l1", speaker_id: "s1", file: "takes/t1.wav", duration: 2, preroll: 1, start_rel: 0, offset_ms: 0, stretch: 1,
  selected: false, ...o,
} as Take as TakeX);

describe("разбор дубля по тексту", () => {
  it("полный дубль", () => {
    expect(textReview(take({ complete: true, text_match: 1 }))).toEqual({ kind: "full", text: "Всё сказал ✓", warn: false });
  });
  it("пропущены слова — перечисляем до четырёх", () => {
    const r = textReview(take({ complete: false, text_match: 0.5 }), ["топлива", "у", "нас", "не", "осталось"]);
    expect(r.kind).toBe("missing");
    expect(r.text).toBe("Пропущены слова: «топлива», «у», «нас», «не»…");
    expect(r.warn).toBe(true);
  });
  it("не полный, но слов нет — процент", () => {
    expect(textReview(take({ complete: false, text_match: 0.62 })).text).toBe("Сказано примерно 62% текста");
    expect(textReview(take({ complete: false })).text).toBe("Сказал не всё");
  });
  it("повтор в конце — предложение отрезать, с цитатой", () => {
    const r = textReview(take({ complete: true, tail_cut_s: 3.1, tail_text: "давай ещё раз" }));
    expect(r.kind).toBe("tail");
    expect(r.text).toBe("В конце повтор («давай ещё раз») — отрезать?");
  });
  it("пустой дубль важнее всего", () => {
    expect(textReview(take({ empty: true, complete: true })).kind).toBe("empty");
  });
  it("движок не считал разбор — ничего не показываем", () => {
    expect(textReview(take())).toEqual({ kind: "unknown", text: "", warn: false });
  });
});

describe("почему выбран ★", () => {
  const a = take({ id: "a", text_match: 0.5, selected: false });
  const b = take({ id: "b", text_match: 1, selected: true });
  it("самый полный", () => {
    expect(starReason(b, [a, b])).toBe("Самый полный: сказано 100% текста");
  });
  it("выбрал человек", () => {
    expect(starReason({ ...b, chosen_by_user: true }, [a, b])).toBe("Выбран тобой");
  });
  it("единственный годный: пустые не в счёт", () => {
    expect(starReason(b, [take({ id: "e", empty: true }), b])).toBe("Единственный годный дубль");
    expect(starReason(b, [b])).toBe("Единственный годный дубль");
  });
  it("все одинаково полные", () => {
    const c = take({ id: "c", text_match: 1, selected: true });
    expect(starReason(c, [take({ id: "d", text_match: 1 }), c])).toContain("одинаково полные");
  });
  it("нет разбора — последний записанный; без ★ — пусто; пустой — пусто", () => {
    const c = take({ id: "c", selected: true });
    expect(starReason(c, [take({ id: "d" }), c])).toBe("Взят последний записанный");
    expect(starReason(take(), [take()])).toBe("");
    expect(starReason(take({ selected: true, empty: true }), [])).toBe("");
  });
});

describe("перегруз, источник, советы про губы", () => {
  it("clip_pct из движка важнее пика; без поля — по пику", () => {
    expect(isClipped(take({ clip_pct: 0.4, peak_db: -12 }))).toBe(true);
    expect(isClipped(take({ clip_pct: 0, peak_db: 0 }))).toBe(false);
    expect(isClipped(take({ peak_db: -0.1 }))).toBe(true);
    expect(isClipped(take({ peak_db: -6 }))).toBe(false);
    expect(CLIPPED_TEXT).toMatch(/подальше от микрофона/);
  });
  it("файл от микрофона отличаем; старый движок — микрофон", () => {
    expect(isFileTake(take({ source: "file" }))).toBe(true);
    expect(isFileTake(take())).toBe(false);
  });
  it("фильтр советов про губы ловит «спешишь вперёд губ»", () => {
    expect(TIMING_REMARK_RE.test("Ты сильно спешишь вперёд губ — выжди паузу")).toBe(true);
    expect(TIMING_REMARK_RE.test("Тебя еле слышно — подойди ближе")).toBe(false);
  });
});

import { visibleRemark } from "./takeReview";
describe("совет под дублем", () => {
  const timing = { text: "Ты сильно спешишь вперёд губ — выжди паузу.", severity: "warn" as const };
  const loud = { text: "Тебя еле слышно — подойди ближе.", severity: "warn" as const };
  it("дубль из файла: про губы молчим, про громкость говорим", () => {
    expect(visibleRemark(timing, { lips: true, file: true })).toBeUndefined();
    expect(visibleRemark(loud, { lips: true, file: true })).toBe(loud);
  });
  it("закадровый режим: про губы молчим", () => {
    expect(visibleRemark(timing, { lips: false, file: false })).toBeUndefined();
  });
  it("запись с микрофона в дубляже: показываем всё", () => {
    expect(visibleRemark(timing, { lips: true, file: false })).toBe(timing);
    expect(visibleRemark(undefined, { lips: true, file: false })).toBeUndefined();
  });
});
