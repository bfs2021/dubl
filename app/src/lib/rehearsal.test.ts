import { describe, expect, it } from "vitest";
import { REHEARSAL_TEXT, rehearsalPlan } from "./rehearsal";

describe("репетиция: какой кусок играть", () => {
  it("берём запас до реплики и хвост после", () => {
    expect(rehearsalPlan({ start_rel: 5, end_rel: 8 }, 20, 2)).toEqual({ from: 3, to: 8.5, length: 5.5 });
  });
  it("не выходим за начало и конец сцены", () => {
    expect(rehearsalPlan({ start_rel: 0.5, end_rel: 9.9 }, 10, 4)).toEqual({ from: 0, to: 10, length: 10 });
  });
  it("отрицательный запас считаем нулём", () => {
    expect(rehearsalPlan({ start_rel: 2, end_rel: 3 }, 10, -1).from).toBe(2);
  });
  it("подписи по-русски и на «ты»", () => {
    expect(REHEARSAL_TEXT.speak).toMatch(/запись не идёт/);
  });
});
