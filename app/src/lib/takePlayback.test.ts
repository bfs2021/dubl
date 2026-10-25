import { describe, expect, it } from "vitest";
import { detectSpeech, findTakeSpeech, fmtShift, listenWindow, placedAudio, shiftLimit, clampShift, takePlacement, timeStretch } from "./takePlayback";

const SR = 48000;

/** Тот же сигнал, что в эталонном скрипте движка: «шум» + тон 220 Гц кусками. */
function synth(dur = 10, bursts: [number, number][] = [[0.5, 1.0], [2.9, 3.6], [3.9, 5.2], [9.0, 9.5]]) {
  const n = Math.floor(dur * SR);
  const x = new Float32Array(n);
  for (let i = 0; i < n; i++) x[i] = 0.001 * Math.sin(i * 12.9898) * Math.sin(i * 0.0137);
  for (const [a, b] of bursts) for (let i = Math.floor(a * SR); i < Math.floor(b * SR); i++) x[i] += 0.3 * Math.sin((2 * Math.PI * 220 * i) / SR);
  return x;
}

// Дубль из реального проекта: start_rel 13.382, preroll 1.0, offset 1358 мс, stretch 0.85, реплика 15.382–17.982.
const take = { start_rel: 13.382, preroll: 1.0, offset_ms: 1358, stretch: 0.85 };
const line = { start_rel: 15.382, end_rel: 17.982 };

describe("позиция дубля — как в сведении (audio/mix.py _process_take)", () => {
  it("детектор речи и поиск речи реплики совпадают с движком", () => {
    // движок (engine/.venv, audio.fit.detect_speech / find_take_speech) на этом сигнале даёт ровно это
    const x = synth();
    const iv = detectSpeech(x, SR);
    expect(iv.map(([a, b]) => [+a.toFixed(3), +b.toFixed(3)])).toEqual([[0.5, 1.0], [2.9, 3.6], [3.9, 5.2], [9.0, 9.5]]);
    const sp = findTakeSpeech(iv, x.length / SR, { lineStart: line.start_rel, lineEnd: line.end_rel, takeStart: take.start_rel, preroll: take.preroll });
    expect(sp![0]).toBeCloseTo(2.9, 9);
    expect(sp![1]).toBeCloseTo(5.2, 9);
  });

  it("start, обрезка и место в сцене — те же числа, что у движка", () => {
    // эталон движка: start 14.175, lo 106320, hi 236160, placed 16.39, речь с 16.64
    const p = takePlacement(take, [2.9, 5.2], SR, 10 * SR);
    expect(p.start).toBeCloseTo(14.175, 9);
    expect(p.lo).toBe(106320);
    expect(p.hi).toBe(236160);
    expect(p.placed).toBeCloseTo(16.39, 9);
    expect(p.speechStart).toBeCloseTo(16.64, 9);
    // начало речи от stretch не зависит: start_rel − preroll + pivot + offset
    expect(p.speechStart).toBeCloseTo(take.start_rel - take.preroll + 2.9 + take.offset_ms / 1000, 9);
  });

  it("при stretch = 1 — просто start_rel − preroll + offset", () => {
    const p = takePlacement({ ...take, stretch: 1 }, [2.9, 5.2], SR, 10 * SR);
    expect(p.start).toBeCloseTo(13.382 - 1 + 1.358, 9);
    expect(p.lo).toBe(Math.trunc((2.9 - 0.25) * SR));
    expect(p.hi).toBe(Math.trunc((5.2 + 0.5) * SR));
  });

  it("сдвиг на +50 мс двигает дубль ровно на 0,05 с", () => {
    const a = takePlacement(take, [2.9, 5.2], SR, 10 * SR);
    const b = takePlacement({ ...take, offset_ms: take.offset_ms + 50 }, [2.9, 5.2], SR, 10 * SR);
    expect(b.placed - a.placed).toBeCloseTo(0.05, 9);
  });

  it("окно прослушивания: от реплики − 0,5 с до конца + 0,5 с, захватывает и речь дубля", () => {
    const p = takePlacement(take, [2.9, 5.2], SR, 10 * SR);
    const [from, to] = listenWindow(line, p, 30);
    expect(from).toBeCloseTo(14.882, 9);
    expect(to).toBeCloseTo(Math.max(17.982, p.speechEnd) + 0.5, 9);
  });
});

describe("звук дубля в браузере", () => {
  it("растяжение меняет длину в stretch раз", () => {
    const x = synth(2, [[0.2, 1.8]]);
    expect(timeStretch(x, SR, 0.85).length).toBe(Math.round(x.length * 0.85));
    expect(timeStretch(x, SR, 1)).toBe(x);
  });
  it("кусок для сведения длиной hi − lo", () => {
    const x = synth();
    const p = takePlacement(take, [2.9, 5.2], SR, x.length);
    expect(placedAudio(x, SR, take.stretch, p).length).toBe(p.hi - p.lo);
    const p1 = takePlacement({ ...take, stretch: 1 }, [2.9, 5.2], SR, x.length);
    expect(placedAudio(x, SR, 1, p1).length).toBe(p1.hi - p1.lo);
  });
});

describe("подпись и пределы сдвига", () => {
  it("со знаком и запятой", () => {
    expect(fmtShift(300)).toBe("+0,30 с");
    expect(fmtShift(-50)).toBe("−0,05 с");
    expect(fmtShift(0)).toBe("0,00 с");
    expect(fmtShift(1358)).toBe("+1,36 с");
  });
  it("пределы как у ползунка Сведения", () => {
    expect(shiftLimit(0)).toBe(3000);
    expect(shiftLimit(3400)).toBe(3500);
    expect(clampShift(3050, 3000)).toBe(3000);
    expect(clampShift(-3050, 3000)).toBe(-3000);
  });
});
