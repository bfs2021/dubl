import { describe, expect, it } from "vitest";
import { detailText, humanizeMlFailure, initialMl, mlFraction, reduceMl, stageTitle, type MlEvent } from "./mlpack";

const run = (events: MlEvent[]) => events.reduce(reduceMl, initialMl());

describe("события установщика нейросетей", () => {
  it("этапы ведут полосу вперёд, деталь сбрасывается на новом этапе", () => {
    let s = run([{ kind: "stage", stage: "uv", index: 1, total: 4 }]);
    expect(s).toMatchObject({ phase: "running", stage: "uv", index: 1, total: 4 });
    s = reduceMl(s, { kind: "log", detail: "Downloading torch (60.9MiB)" });
    expect(s.detail).toBe("Скачиваю torch (60,9 МБ)");
    const f1 = mlFraction(s);
    s = reduceMl(s, { kind: "stage", stage: "torch", index: 3, total: 4 });
    expect(s.detail).toBe("");
    expect(mlFraction(s)).toBeGreaterThan(f1);
    expect(mlFraction(s)).toBeLessThan(1);
  });

  it("done → перезапуск движка, installed=true; ошибка хранит сырую причину; отмена возвращает в покой", () => {
    const done = run([{ kind: "stage", stage: "engine", index: 4, total: 4 }, { kind: "done" }]);
    expect(done).toMatchObject({ phase: "restarting", installed: true });
    expect(mlFraction(done)).toBe(1);
    const bad = run([{ kind: "stage", stage: "torch", index: 3, total: 4 }, { kind: "failed", reason: "No space left on device" }]);
    expect(bad).toMatchObject({ phase: "failed", error: "No space left on device" });
    expect(run([{ kind: "stage", stage: "uv", index: 1, total: 4 }, { kind: "cancelled" }]).phase).toBe("idle");
  });

  it("лог до начала установки игнорируется", () => {
    expect(reduceMl(initialMl(), { kind: "log", detail: "что-то" }).detail).toBe("");
  });

  it("подписи этапов по-русски", () => {
    expect(stageTitle("uv")).toMatch(/установщик/);
    expect(stageTitle("python")).toMatch(/Python/);
    expect(stageTitle("torch")).toMatch(/torch/);
    expect(stageTitle("engine")).toMatch(/библиотеки/);
  });

  it("вывод uv превращается в короткую русскую деталь", () => {
    expect(detailText("Downloading cpython-3.11.16-macos-aarch64-none (download) (25.9MiB)")).toContain("Скачиваю cpython");
    expect(detailText("Downloaded torch")).toBe("Скачал torch");
    expect(detailText("Installed 10 packages in 144ms")).toBe("Поставил пакетов: 10");
    expect(detailText("x".repeat(200)).length).toBeLessThanOrEqual(90);
  });

  it("ошибки: нет сети, нет места, прочее", () => {
    expect(humanizeMlFailure("команда завершилась с кодом 2: uv pip — error sending request for url").title).toMatch(/интернет/);
    expect(humanizeMlFailure("<urlopen error [Errno 8] nodename nor servname provided>").title).toMatch(/интернет/);
    expect(humanizeMlFailure("failed to write: No space left on device (os error 28)").title).toMatch(/места/);
    expect(humanizeMlFailure("контрольная сумма uv не совпала").title).toMatch(/проверку/);
    expect(humanizeMlFailure("что-то странное").title).toMatch(/Не получилось/);
    expect(humanizeMlFailure(null).title).toMatch(/Не получилось/);
  });
});
