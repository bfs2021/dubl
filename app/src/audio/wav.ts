/** WAV: кодер/декодер (PCM 16 бит или float32), mono/stereo. Без зависимостей. */

export function encodeWav(samples: Float32Array, sampleRate = 48000, bits: 16 | 32 = 16, channels = 1): ArrayBuffer {
  const bytesPer = bits / 8;
  const dataLen = samples.length * bytesPer;
  const buf = new ArrayBuffer(44 + dataLen);
  const v = new DataView(buf);
  const str = (o: number, s: string) => {
    for (let i = 0; i < s.length; i++) v.setUint8(o + i, s.charCodeAt(i));
  };
  str(0, "RIFF");
  v.setUint32(4, 36 + dataLen, true);
  str(8, "WAVE");
  str(12, "fmt ");
  v.setUint32(16, 16, true);
  v.setUint16(20, bits === 32 ? 3 : 1, true); // 3 = IEEE float
  v.setUint16(22, channels, true);
  v.setUint32(24, sampleRate, true);
  v.setUint32(28, sampleRate * channels * bytesPer, true);
  v.setUint16(32, channels * bytesPer, true);
  v.setUint16(34, bits, true);
  str(36, "data");
  v.setUint32(40, dataLen, true);
  let o = 44;
  if (bits === 32) {
    for (let i = 0; i < samples.length; i++, o += 4) v.setFloat32(o, samples[i], true);
  } else {
    for (let i = 0; i < samples.length; i++, o += 2) {
      const s = Math.max(-1, Math.min(1, samples[i]));
      v.setInt16(o, s < 0 ? s * 0x8000 : s * 0x7fff, true);
    }
  }
  return buf;
}

export const wavBlob = (samples: Float32Array, sampleRate = 48000, bits: 16 | 32 = 16) =>
  new Blob([encodeWav(samples, sampleRate, bits)], { type: "audio/wav" });

export interface DecodedWav { samples: Float32Array; sampleRate: number; channels: number }

/** Читает WAV (PCM16 / float32), каналы сводит в mono. */
export function decodeWav(buf: ArrayBuffer): DecodedWav {
  const v = new DataView(buf);
  const tag = (o: number) => String.fromCharCode(v.getUint8(o), v.getUint8(o + 1), v.getUint8(o + 2), v.getUint8(o + 3));
  if (tag(0) !== "RIFF" || tag(8) !== "WAVE") throw new Error("Это не WAV");
  let o = 12;
  let fmt = 1, channels = 1, sampleRate = 48000, bits = 16;
  let dataOff = -1, dataLen = 0;
  while (o + 8 <= buf.byteLength) {
    const id = tag(o);
    const len = v.getUint32(o + 4, true);
    if (id === "fmt ") {
      fmt = v.getUint16(o + 8, true);
      channels = v.getUint16(o + 10, true);
      sampleRate = v.getUint32(o + 12, true);
      bits = v.getUint16(o + 22, true);
    } else if (id === "data") {
      dataOff = o + 8;
      dataLen = Math.min(len, buf.byteLength - dataOff);
      break;
    }
    o += 8 + len + (len % 2);
  }
  if (dataOff < 0) throw new Error("В WAV нет данных");
  const bytes = bits / 8;
  const frames = Math.floor(dataLen / (bytes * channels));
  const out = new Float32Array(frames);
  for (let i = 0; i < frames; i++) {
    let acc = 0;
    for (let c = 0; c < channels; c++) {
      const p = dataOff + (i * channels + c) * bytes;
      acc += fmt === 3 ? v.getFloat32(p, true) : bits === 16 ? v.getInt16(p, true) / 0x8000 : 0;
    }
    out[i] = acc / channels;
  }
  return { samples: out, sampleRate, channels };
}
