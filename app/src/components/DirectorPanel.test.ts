import { describe, expect, it } from "vitest";
import { directorExamples } from "./DirectorPanel";

describe("directorExamples", () => {
  it("строит примеры из имён персонажей сцены", () => {
    const ex = directorExamples(["Шкипер", "Ковальски"]);
    expect(ex).toContain("Шкипер громче");
    expect(ex.some((e) => e.startsWith("Ковальски"))).toBe(true);
    expect(ex.join(" ")).not.toMatch(/кот/);
  });
  it("без персонажей остаётся общий пример про фон", () => {
    expect(directorExamples([])).toEqual(["фон потише"]);
  });
});
