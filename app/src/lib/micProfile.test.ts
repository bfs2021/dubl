import { describe, expect, it } from "vitest";
import { buildProfile, hasMicProfile, noiseFloor, profileSummary, speechStats, verdictOf, zoneOf } from "./micProfile";

const SR = 48000;
const tone = (db: number, sec: number) => {
  const a = Math.pow(10, db / 20);
  return Float32Array.from({ length: Math.round(SR * sec) }, (_, i) => a * Math.sin((2 * Math.PI * 220 * i) / SR));
};
const cat = (...xs: Float32Array[]) => { const o = new Float32Array(xs.reduce((n, x) => n + x.length, 0)); let p = 0; for (const x of xs) { o.set(x, p); p += x.length; } return o; };

describe("зоны шкалы", () => {
  it.each([[-80, "silent"], [-30, "low"], [-24, "good"], [-12, "good"], [-8, "good"], [-5, "hot"], [-1, "clip"]] as const)("%i дБ → %s", (db, z) => {
    expect(zoneOf(db)).toBe(z);
  });
});

describe("замеры", () => {
  it("шум комнаты — среднее по тишине", () => {
    expect(noiseFloor(tone(-60, 2))).toBeCloseTo(-63, 0);
  });
  it("громкость речи считается только по живым окнам, паузы не занижают среднее", () => {
    const x = cat(tone(-70, 1), tone(-18, 1), tone(-70, 1));
    const s = speechStats(x, SR, -70);
    expect(s.rms).toBeGreaterThan(-23);
    expect(s.rms).toBeLessThan(-19);
    expect(s.peak).toBeCloseTo(-18, 0);
    expect(s.activeS).toBeGreaterThan(0.9);
    expect(s.activeS).toBeLessThan(1.2);
  });
  it("шумная комната: порог не отсекает всю речь, запас над шумом не уходит в минус сто", () => {
    const x = cat(tone(-20, 1), tone(-8, 1));
    const s = speechStats(x, SR, -18);
    expect(s.rms).toBeGreaterThan(-30);
    expect(s.activeS).toBeGreaterThan(0.5);
    const p = buildProfile({ noiseDb: -18, speechPeak: s.peak, speechRms: -120, latencyMs: null, headphones: true, deviceId: null });
    expect(p.snr_db).toBe(0);
  });
  it("тишина → речи нет", () => {
    expect(speechStats(tone(-80, 1), SR, -80).activeS).toBe(0);
  });
});

describe("вердикт", () => {
  it.each([
    [-60, -12, -20, "good"], [-60, -35, -45, "quiet"], [-60, -1, -10, "loud"],
    [-40, -12, -20, "noisy"], [-60, -70, -80, "silent"],
  ] as const)("шум %i, пик %i, средняя %i → %s", (n, p, r, v) => {
    expect(verdictOf(n, p, r)).toBe(v);
  });
});

describe("профиль", () => {
  const p = buildProfile({ noiseDb: -58.04, speechPeak: -10.06, speechRms: -22.4, latencyMs: 99, headphones: false, deviceId: null, now: new Date("2026-10-22T10:00:00Z") });
  it("считает запас речи над шумом и округляет", () => {
    expect(p.snr_db).toBe(35.6);
    expect(p.noise_floor_db).toBe(-58);
    expect(p.verdict).toBe("good");
    expect(hasMicProfile(p)).toBe(true);
    expect(hasMicProfile(null)).toBe(false);
  });
  it("строка для главного экрана", () => {
    expect(profileSummary(p)).toBe("Микрофон в порядке · задержка 99 мс");
    expect(profileSummary({ ...p, latency_ms: null })).toBe("Микрофон в порядке");
  });
});
