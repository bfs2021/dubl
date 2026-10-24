import { describe, expect, it } from "vitest";
import { humanizeError, isNoAiError } from "./errors";
import { remarkOf, takeCaption } from "./director";
import { aiReady, connectedCount, providerHealth } from "./ai";
import { etaText } from "../components/JobSteps";

describe("человеческие ошибки", () => {
  it("сырые ответы сервера и ffmpeg заменяются текстом с next step", () => {
    expect(humanizeError("Not Found")).toMatch(/Перезапусти/);
    expect(humanizeError("Ошибка движка (404)")).toMatch(/Перезапусти/);
    expect(humanizeError("[mov,mp4] moov atom not found")).toMatch(/Файл повреждён/);
    expect(humanizeError("/Users/x.mkv: No such file or directory")).toMatch(/Не нашёл файл/);
    expect(humanizeError("ffmpeg: Conversion failed!")).toMatch(/Повтори/);
    expect(humanizeError("Traceback (most recent call last): File x.py")).toMatch(/внутри программы/);
  });
  it("русский текст движка ({detail}) не трогаем — даже если в нём есть слово ffmpeg", () => {
    expect(humanizeError("Не получилось прочитать звук через ffmpeg: файл обрезан.")).toBe("Не получилось прочитать звук через ffmpeg: файл обрезан.");
    expect(humanizeError("В файле нет видео — озвучивать нечего.")).toBe("В файле нет видео — озвучивать нечего.");
  });
  it("«ИИ не настроен» узнаётся", () => {
    expect(isNoAiError("ИИ для этой задачи не настроен. Подключите модель в настройках.")).toBe(true);
    expect(isNoAiError("Файл не найден")).toBe(false);
  });
});

describe("подписи дубля", () => {
  const t = (offset_ms: number, stretch = 1, fit_confidence: number | null = 0.85) => ({ duration: 12.6, offset_ms, stretch, fit_confidence });
  it("вместо «+3049 мс · 85%» — слова", () => {
    const c = takeCaption(t(3049, 0.85));
    expect(c.timing).toBe("Начал на 3,0 с раньше оригинала");
    expect(c.match).toBe("Совпадение 85%");
    expect(c.length).toMatch(/Длиннее оригинала/);
    expect(takeCaption(t(-1200)).timing).toBe("Начал на 1,2 с позже оригинала");
    expect(takeCaption(t(40)).timing).toBe("Попал в движения губ");
  });
  it("совет помощника — одна главная фраза, остальное под «Ещё»", () => {
    const r = remarkOf({ report: { ok: false } as never, feedback: "Ты спешишь вперёд губ. Реплика длиннее оригинала — сократи текст." });
    expect(r.text).toBe("Ты спешишь вперёд губ.");
    expect(r.more).toMatch(/длиннее/);
    expect(remarkOf({ report: { ok: true } as never, feedback: "Хороший дубль." }).more).toBeUndefined();
  });
});

describe("ИИ подключён?", () => {
  const mk = (g: object) => ({
    llm: { providers: [{ id: "gemini", kind: "gemini", title: "G", model: "m", enabled: true, ...g }, { id: "claude_cli", kind: "claude_cli", title: "C", model: "m", enabled: true }],
      routing: { translate: "gemini", fit_length: "claude_cli", name_speakers: null } },
  }) as never;
  it("нужен маршрут, включённая модель и принятый ключ", () => {
    expect(aiReady(mk({ key_set: false, key_status: "unknown" }), "translate")).toBe(false);
    expect(aiReady(mk({ key_set: true, key_status: "ok" }), "translate")).toBe(true);
    expect(aiReady(mk({ key_set: true, key_status: "unknown" }), "translate")).toBe(true); // не проверен — не блокируем
    expect(aiReady(mk({ key_set: true, key_status: "invalid", key_message: "не подходит" }), "translate")).toBe(false);
    expect(aiReady(mk({ key_set: true, key_status: "ok" }), "fit_length")).toBe(true); // Claude CLI без ключа
    expect(aiReady(mk({ key_set: true, key_status: "ok" }), "name_speakers")).toBe(false);
    expect(aiReady(null, "translate")).toBe(false);
    expect(aiReady(null, "translate", true)).toBe(true); // демо
  });
  it("в шапке считаем только рабочие модели, «не подходит» не считается", () => {
    expect(connectedCount(mk({ key_set: true, key_status: "invalid" }), false)).toBe(1); // только Claude
    expect(connectedCount(mk({ key_set: true, key_status: "ok" }), false)).toBe(2);
    expect(providerHealth({ id: "g", kind: "gemini", title: "", model: "", enabled: true, key_set: true, key_status: "invalid" }, false)).toBe("invalid");
  });
});

describe("оценка времени (Job.eta_s от движка)", () => {
  it("null и мусор — молчим; иначе «осталось около…»", () => {
    expect(etaText(null)).toBeNull();
    expect(etaText(undefined)).toBeNull();
    expect(etaText(0)).toBeNull();
    expect(etaText(4)).toBe("почти готово");
    expect(etaText(40)).toBe("осталось около 40 с");
    expect(etaText(360)).toBe("осталось около 6 мин");
  });
});

import { findClick, levelStep } from "../audio/recorder";
describe("подписи дубля: противоречия", () => {
  it("совпадение 0% — не пишем «попал в губы»", () => {
    const c = takeCaption({ duration: 3, offset_ms: 0, stretch: 1, fit_confidence: 0 });
    expect(c.timing).not.toMatch(/Попал/);
    expect(c.match).toBeNull();
  });
  it("закадровый: про губы не говорим", () => {
    const c = takeCaption({ duration: 3, offset_ms: 900, stretch: 1, fit_confidence: 0.8 }, false);
    expect(c.timing).not.toMatch(/раньше|позже|Попал/);
  });
});

describe("шкала уровня и калибровка микрофона", () => {
  it("уровень быстро растёт, медленно падает; «слышу» зажигается от звука", () => {
    let r = { db: -120, peak: -120, heard: false };
    r = levelStep(r, -20, 33);
    expect(r.db).toBe(-20);
    expect(r.heard).toBe(true);
    const after = levelStep(r, -120, 100);
    expect(after.db).toBeLessThan(-20);
    expect(after.db).toBeGreaterThan(-30);
    expect(levelStep({ db: -120, peak: -120, heard: false }, -70, 33).heard).toBe(false);
  });
  it("findClick: находит щелчок и понятно ругается на тишину", () => {
    const sr = 48000;
    const a = new Float32Array(sr * 1.5);
    a.fill(0.0005);
    a[Math.floor(sr * 0.45)] = 0.5; // щелчок через 150 мс после старта окна (0.3 с)
    expect(findClick(a, sr)).toBe(150);
    expect(() => findClick(new Float32Array(sr).fill(0.0001), sr)).toThrow(/почти ничего не слышит/);
    expect(() => findClick(new Float32Array(sr).fill(0.005), sr)).toThrow(/Щелчок не услышан/);
  });
});
