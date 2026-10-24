import { describe, expect, it } from "vitest";
import { dialogueAround, shotAround, snapSelection, selectByMode } from "./snap";
import type { Line } from "../api/types";

const L = (id: string, start: number, end: number): Line => ({ id, start, end, text: id, speaker_id: "s1", words: [], source: "embedded", overlap: false });
const lines = [L("a", 10, 12), L("b", 12.5, 14), L("c", 15, 17), L("d", 30, 32)];

describe("snapSelection", () => {
  it("край внутри реплики уезжает на её границу с запасом", () => {
    const [a, b] = snapSelection(11, 13.2, lines, []);
    expect(a).toBeCloseTo(10 - 0.15, 2);
    expect(b).toBeCloseTo(14 + 0.15, 2);
  });
  it("край в паузе остаётся на месте", () => {
    expect(snapSelection(20, 25, lines, [])).toEqual([20, 25]);
  });
  it("прилипает к склейке рядом", () => {
    const [a, b] = snapSelection(20.2, 24.7, lines, [20, 25]);
    expect([a, b]).toEqual([20, 25]);
  });
  it("не прилипает к склейке, которая режет реплику", () => {
    const [a] = snapSelection(9.8, 13, lines, [11]);
    expect(a).not.toBe(11);
  });
  it("переставляет края, если перепутаны", () => {
    const [a, b] = snapSelection(25, 20, lines, []);
    expect(a).toBe(20);
    expect(b).toBe(25);
  });
  it("запас не заходит на соседнюю реплику", () => {
    const [a] = snapSelection(12.6, 13, lines, []);
    expect(a).toBeGreaterThanOrEqual(12.25 - 1e-6);
  });
  it("не уходит ниже нуля", () => {
    expect(snapSelection(0.05, 3, [L("x", 0, 2)], [])[0]).toBe(0);
  });
});

describe("dialogueAround", () => {
  it("клик по реплике — весь разговор с паузами ≤ 2.5 с", () => {
    expect(dialogueAround(11, lines)).toEqual([10, 17]);
  });
  it("далёкая реплика отдельно", () => {
    expect(dialogueAround(31, lines)).toEqual([30, 32]);
  });
  it("тишина далеко от реплик — пустой диапазон", () => {
    expect(dialogueAround(23, lines)).toEqual([23, 23]);
  });
});

describe("shotAround", () => {
  const cuts = [5, 20, 40];
  it("от склейки до склейки", () => {
    expect(shotAround(22, cuts, 60)).toEqual([20, 40]);
    expect(shotAround(1, cuts, 60)).toEqual([0, 5]);
    expect(shotAround(55, cuts, 60)).toEqual([40, 60]);
  });
});

describe("selectByMode", () => {
  it("режим «Сцена» отдаёт кадр целиком", () => {
    expect(selectByMode("shot", 22, 0, 0, lines, [5, 20, 40], 60)).toEqual([20, 40]);
  });
  it("режим «Диалог» привязывает края", () => {
    const [a, b] = selectByMode("dialogue", 11, 0, 0, lines, [], 60);
    expect(a).toBeLessThan(10);
    expect(b).toBeGreaterThan(17);
  });
});
