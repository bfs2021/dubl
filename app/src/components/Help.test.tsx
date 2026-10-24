import { describe, expect, it } from "vitest";
import { placePopover } from "./Help";

const view = { w: 1200, h: 720 };
const size = { w: 280, h: 90 };

describe("подсказка «?» не вылезает за окно", () => {
  it("обычный случай: под значком по центру", () => {
    const p = placePopover({ left: 600, top: 300, bottom: 315, width: 15 }, size, view);
    expect(p.left).toBeCloseTo(600 + 7.5 - 140);
    expect(p.top).toBe(321);
    expect(p.above).toBe(false);
  });
  it("значок у правого края: сдвигаем влево", () => {
    const p = placePopover({ left: 1190, top: 300, bottom: 315, width: 15 }, size, view);
    expect(p.left + size.w).toBeLessThanOrEqual(view.w - 8);
  });
  it("значок у левого края: не уходим за 0", () => {
    expect(placePopover({ left: 2, top: 300, bottom: 315, width: 15 }, size, view).left).toBe(8);
  });
  it("внизу нет места: показываем над значком", () => {
    const p = placePopover({ left: 600, top: 680, bottom: 695, width: 15 }, size, view);
    expect(p.above).toBe(true);
    expect(p.top + size.h).toBeLessThanOrEqual(680);
  });
});
