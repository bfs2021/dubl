import { describe, expect, it } from "vitest";
import { decodeWav, encodeWav, wavBlob } from "./wav";
import { concatChunks, finishCapture } from "./recorder";
import { envelope, speechBounds } from "./analyze";
import { sliceByScene, takeWindow } from "./takes";

const sine = (n: number, f = 440, sr = 48000, amp = 0.5) => Float32Array.from({ length: n }, (_, i) => Math.sin((2 * Math.PI * f * i) / sr) * amp);

describe("WAV", () => {
  it("пишет корректный заголовок 48 кГц mono PCM16", () => {
    const buf = encodeWav(sine(480), 48000);
    const v = new DataView(buf);
    const tag = (o: number) => String.fromCharCode(...[0, 1, 2, 3].map((i) => v.getUint8(o + i)));
    expect(tag(0)).toBe("RIFF");
    expect(tag(8)).toBe("WAVE");
    expect(v.getUint16(20, true)).toBe(1);
    expect(v.getUint16(22, true)).toBe(1);
    expect(v.getUint32(24, true)).toBe(48000);
    expect(v.getUint16(34, true)).toBe(16);
    expect(v.getUint32(40, true)).toBe(960);
    expect(buf.byteLength).toBe(44 + 960);
  });
  it("кодирует и декодирует без потерь (в пределах PCM16)", () => {
    const x = sine(1000);
    const y = decodeWav(encodeWav(x, 48000));
    expect(y.sampleRate).toBe(48000);
    expect(y.samples.length).toBe(1000);
    for (let i = 0; i < 1000; i += 97) expect(y.samples[i]).toBeCloseTo(x[i], 3);
  });
  it("float32 сохраняет значения точно", () => {
    const x = sine(100);
    const y = decodeWav(encodeWav(x, 48000, 32));
    for (let i = 0; i < 100; i++) expect(y.samples[i]).toBeCloseTo(x[i], 6);
  });
  it("зажимает перегруз", () => {
    const y = decodeWav(encodeWav(Float32Array.from([2, -2]), 48000));
    expect(y.samples[0]).toBeCloseTo(1, 3);
    expect(y.samples[1]).toBeCloseTo(-1, 3);
  });
  it("отвергает не-WAV", () => {
    expect(() => decodeWav(new ArrayBuffer(64))).toThrow();
  });
  it("blob имеет тип audio/wav", () => {
    expect(wavBlob(sine(10)).type).toBe("audio/wav");
  });
});

describe("запись и замеры", () => {
  it("склеивает куски и считает длительность", () => {
    const r = finishCapture([sine(24000), sine(24000)], 48000);
    expect(r.duration).toBeCloseTo(1, 5);
    expect(r.peakDb).toBeCloseTo(-6, 0);
    expect(concatChunks([]).length).toBe(0);
  });
  it("находит границы речи в тишине", () => {
    const x = new Float32Array(48000 * 3);
    x.set(sine(48000), 48000);
    const b = speechBounds(x, 48000)!;
    expect(b[0]).toBeCloseTo(1, 1);
    expect(b[1]).toBeCloseTo(2, 1);
    expect(speechBounds(new Float32Array(4800), 48000)).toBeNull();
  });
  it("огибающая нужной длины", () => {
    expect(envelope(sine(48000), 50)).toHaveLength(50);
  });
  it("нарезает запись по времени сцены", () => {
    const x = new Float32Array(48000 * 4);
    const piece = sliceByScene(x, 48000, 9, 10, 12); // файл начинается на 9-й секунде сцены
    expect(piece.length).toBe(96000);
    expect(sliceByScene(x, 48000, 9, 20, 21).length).toBe(0);
    expect(takeWindow({ line_id: "l", start_rel: 0.2, end_rel: 2 }, 10)).toEqual([0, 2.6]);
  });
});
