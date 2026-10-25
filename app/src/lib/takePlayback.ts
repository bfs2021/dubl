/**
 * «Сдвиг» в карточке дубля: где сведение поставит дубль и как быстро послушать реплику с видео и фоном прямо в браузере,
 * без пересборки сцены на движке.
 *
 * Позиция дубля — ровно как в engine/src/dubl_engine/audio/mix.py (_process_take) и audio/fit.py:
 *   pivot  = начало речи в файле дубля (find_take_speech по detect_speech; при наличии голоса оригинала движок берёт
 *            locate_speech — тот же find_take_speech, только длину оригинала o_len меряет по его речи)
 *   start  = take.start_rel − take.preroll + offset_ms/1000 + pivot·(1 − stretch)     — где в сцене начало файла
 *   lo, hi = int((sp0·stretch − 0,25)·sr), int((sp1·stretch + 0,5)·sr)                   — оставляем только речь с запасом
 *   placed = start + lo/sr                                                               — где в сцене начало звучащего куска
 * Речь в сцене начинается в start + pivot·stretch = start_rel − preroll + pivot + offset_ms/1000 (от stretch не зависит).
 *
 * Упрощения браузера (честно): шумодав, чистка, эффекты, комната и громкость автопилота не повторяются — играет
 * сырой дубль; stretch ≠ 1 делаем своим WSOLA (как atempo в ffmpeg: длина меняется, тон — нет), он звучит чуть
 * грубее, но тайминг тот же. Фон — background.wav сцены без приглушения под голосом.
 */
import type { Player } from "./player";
import type { SceneLine, Take } from "../api/types";
import { fmtNum } from "./format";

// ───────────── сдвиг: шаг, пределы, подпись ─────────────
export const SHIFT_STEP_MS = 50;
/** Пределы — как у ползунка «Когда начинается» в Сведении (Mix.tsx offRange): ±3 с, шире, если значение уже за этим. */
export const shiftLimit = (v: number) => Math.max(3000, Math.ceil(Math.abs(v) / 500) * 500);
export const clampShift = (v: number, limit: number) => Math.round(Math.min(limit, Math.max(-limit, v)));
/** «+0,30 с», «−0,05 с», «0,00 с». */
export const fmtShift = (ms: number) => {
  const r = Math.round(ms / 10) / 100;
  return (r > 0 ? "+" : r < 0 ? "−" : "") + fmtNum(Math.abs(r), 2) + " с";
};

// ───────────── детектор речи: копия audio/fit.py ─────────────
/** round() Python: банковское округление половинок. */
const pyRound = (x: number) => {
  const f = Math.floor(x);
  const d = x - f;
  if (Math.abs(d - 0.5) < 1e-9) return f % 2 === 0 ? f : f + 1;
  return Math.round(x);
};

/** np.percentile, метод linear. */
function percentile(v: Float64Array, q: number): number {
  const s = Float64Array.from(v).sort();
  const pos = (s.length - 1) * (q / 100);
  const lo = Math.floor(pos);
  const hi = Math.min(s.length - 1, lo + 1);
  return s[lo] + (s[hi] - s[lo]) * (pos - lo);
}

function frameDb(x: Float32Array, sr: number, win: number): Float64Array {
  const n = Math.max(1, pyRound(sr * win));
  const k = Math.floor(x.length / n);
  const out = new Float64Array(k);
  for (let f = 0; f < k; f++) {
    let acc = 0;
    for (let i = f * n, e = i + n; i < e; i++) acc += x[i] * x[i];
    out[f] = 20 * Math.log10(Math.sqrt(acc / n) + 1e-9);
  }
  return out;
}

export type Span = [number, number];

/** detect_speech: окна 20 мс, порог от шумового пола, гистерезис, паузы короче min_gap склеиваются. */
export function detectSpeech(x: Float32Array, sr: number, minGap = 0.25, minDuration = 0.06): Span[] {
  const win = 0.02;
  const db = frameDb(x, sr, win);
  if (!db.length) return [];
  const peak = percentile(db, 98);
  const floor = percentile(db, 10);
  if (peak < -60 || peak - floor < 12) return [];
  const thr = Math.max(floor + 10, peak - 30);
  let active = Array.from(db, (d) => d > thr);
  const low = Array.from(db, (d) => d > thr - 8);
  for (let it = 0; it < 3; it++) {
    const grow = active.slice();
    for (let i = 1; i < grow.length; i++) grow[i] = grow[i] || (active[i - 1] && low[i]);
    for (let i = 0; i < grow.length - 1; i++) grow[i] = grow[i] || (active[i + 1] && low[i]);
    const same = grow.every((g, i) => g === active[i]);
    active = grow;
    if (same) break;
  }
  const spans: [number, number][] = [];
  for (let i = 0; i < active.length; i++) {
    if (active[i] && (i === 0 || !active[i - 1])) spans.push([i, i + 1]);
    else if (active[i]) spans[spans.length - 1][1] = i + 1;
  }
  const gap = pyRound(minGap / win);
  const merged: [number, number][] = [];
  for (const [a, b] of spans) {
    if (merged.length && a - merged[merged.length - 1][1] < gap) merged[merged.length - 1][1] = b;
    else merged.push([a, b]);
  }
  return merged.filter(([a, b]) => (b - a) * win >= minDuration).map(([a, b]) => [a * win, b * win]);
}

/** find_take_speech: начало и конец речи дубля этой реплики в секундах файла; null — речи нет. */
export function findTakeSpeech(intervals: Span[], fileDur: number, o: { lineStart: number; lineEnd: number; takeStart: number; preroll: number }): Span | null {
  const shift = o.preroll - o.takeStart;
  const tLo = Math.max(0, shift + o.lineStart - 1.5);
  const tHi = Math.min(fileDur, shift + o.lineEnd + 2.5);
  const cand = intervals.filter(([a, b]) => b > tLo && a < tHi);
  if (!cand.length) return null;
  const origLen = Math.max(0.3, o.lineEnd - o.lineStart);
  const bound = cand[0][0] + Math.max(origLen * 1.4, origLen + 1.0);
  const group = cand.filter(([a]) => a < bound);
  return [group[0][0], group[group.length - 1][1]];
}

// ───────────── позиция дубля: копия _process_take ─────────────
export interface Placement {
  /** где в сцене начало файла дубля (после растяжения) */
  start: number;
  /** кусок растянутого файла, который идёт в сведение, отсчёты [lo, hi) */
  lo: number;
  hi: number;
  /** где в сцене начинается звучащий кусок: start + lo/sr */
  placed: number;
  /** где в сцене начинается и кончается речь дубля */
  speechStart: number;
  speechEnd: number;
}

export function takePlacement(t: Pick<Take, "start_rel" | "preroll" | "offset_ms" | "stretch">, sp: Span | null, sr: number, fileLen: number): Placement {
  const stretch = t.stretch || 1;
  const pivot = sp ? sp[0] : 0;
  const start = t.start_rel - t.preroll + t.offset_ms / 1000 + pivot * (1 - stretch);
  const yLen = Math.abs(stretch - 1) > 1e-4 ? Math.round(fileLen * stretch) : fileLen;
  let lo = 0, hi = yLen;
  if (sp) {
    lo = Math.max(0, Math.trunc((sp[0] * stretch - 0.25) * sr));
    hi = Math.min(yLen, Math.trunc((sp[1] * stretch + 0.5) * sr));
  }
  return {
    start, lo, hi, placed: start + lo / sr,
    speechStart: start + pivot * stretch,
    speechEnd: start + (sp ? sp[1] : fileLen / sr) * stretch,
  };
}

/** Что играем: от начала реплики (или речи дубля, что раньше) − 0,5 с до конца + 0,5 с, в пределах сцены. */
export function listenWindow(line: Pick<SceneLine, "start_rel" | "end_rel">, p: Placement, sceneDur: number, pad = 0.5): Span {
  const from = Math.max(0, Math.min(line.start_rel, p.speechStart) - pad);
  const to = Math.min(sceneDur, Math.max(line.end_rel, p.speechEnd) + pad);
  return [from, Math.max(from + 0.2, to)];
}

// ───────────── растяжение без смены тона (WSOLA) ─────────────
/** Длина × stretch, тон тот же. Окна 40 мс, перекрытие 50%, подстройка фазы ±8 мс — как atempo, только попроще. */
export function timeStretch(x: Float32Array, sr: number, stretch: number): Float32Array {
  if (Math.abs(stretch - 1) < 1e-4 || x.length === 0) return x;
  const outLen = Math.round(x.length * stretch);
  const N = Math.max(4, Math.round(sr * 0.04) & ~1);
  const H = N / 2;
  const tol = Math.round(sr * 0.008);
  const w = new Float32Array(N);
  for (let i = 0; i < N; i++) w[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / N);
  const out = new Float32Array(outLen + N);
  const norm = new Float32Array(outLen + N);
  const maxPos = Math.max(0, x.length - N);
  let prev = 0;
  for (let k = 0; k * H < outLen; k++) {
    const nominal = Math.min(maxPos, Math.round((k * H) / stretch));
    let best = nominal;
    if (k > 0) {
      const ref = prev + H;           // как продолжился бы предыдущий кусок — с ним и совмещаем по фазе
      let bestC = -Infinity;
      for (let d = -tol; d <= tol; d += 2) {
        const p = nominal + d;
        if (p < 0 || p > maxPos) continue;
        let c = 0;
        for (let i = 0; i < H; i += 4) c += (x[ref + i] ?? 0) * x[p + i];
        if (c > bestC) { bestC = c; best = p; }
      }
    }
    const o = k * H;
    for (let i = 0; i < N; i++) { out[o + i] += (x[best + i] ?? 0) * w[i]; norm[o + i] += w[i]; }
    prev = best;
  }
  for (let i = 0; i < outLen; i++) if (norm[i] > 1e-3) out[i] /= norm[i];
  return out.slice(0, outLen);
}

const fade = (y: Float32Array, sr: number, ms = 25) => {
  const n = Math.min(Math.floor((sr * ms) / 1000), Math.floor(y.length / 2));
  for (let i = 0; i < n; i++) { const r = n > 1 ? i / (n - 1) : 1; y[i] *= r; y[y.length - 1 - i] *= r; }
  return y;
};

/** Звучащий кусок дубля, как его кладёт сведение: растянутый [lo, hi) и fade 25 мс. */
export function placedAudio(x: Float32Array, sr: number, stretch: number, p: Placement): Float32Array {
  const st = stretch || 1;
  if (Math.abs(st - 1) <= 1e-4) return fade(x.slice(p.lo, p.hi), sr);
  // растягиваем только нужный кусок (+ запас на окно) — так быстро и на длинных дублях
  const pad = Math.round(sr * 0.05);
  const a = Math.max(0, Math.floor(p.lo / st) - pad);
  const b = Math.min(x.length, Math.ceil(p.hi / st) + pad);
  const y = timeStretch(x.subarray(a, b), sr, st);
  const off = Math.round(a * st);
  return fade(y.slice(Math.max(0, p.lo - off), Math.max(0, p.hi - off)), sr);
}

// ───────────── чтение WAV кусками (фон длинный — берём только окно) ─────────────
interface WavHead { fmt: number; channels: number; sampleRate: number; bits: number; dataOff: number; dataLen: number }

function wavHead(buf: ArrayBuffer): WavHead {
  const v = new DataView(buf);
  const tag = (o: number) => String.fromCharCode(v.getUint8(o), v.getUint8(o + 1), v.getUint8(o + 2), v.getUint8(o + 3));
  if (buf.byteLength < 12 || tag(0) !== "RIFF" || tag(8) !== "WAVE") throw new Error("Это не WAV");
  let o = 12;
  const h: WavHead = { fmt: 1, channels: 1, sampleRate: 48000, bits: 16, dataOff: -1, dataLen: 0 };
  while (o + 8 <= buf.byteLength) {
    const id = tag(o);
    const len = v.getUint32(o + 4, true);
    if (id === "fmt ") {
      h.fmt = v.getUint16(o + 8, true);
      h.channels = v.getUint16(o + 10, true);
      h.sampleRate = v.getUint32(o + 12, true);
      h.bits = v.getUint16(o + 22, true);
      if (h.fmt === 0xfffe && len >= 26) h.fmt = v.getUint16(o + 32, true);   // WAVE_FORMAT_EXTENSIBLE: код формата в GUID
    } else if (id === "data") { h.dataOff = o + 8; h.dataLen = len; break; }
    o += 8 + len + (len % 2);
  }
  if (h.dataOff < 0) throw new Error("В WAV нет данных");
  return h;
}

function pcm(buf: ArrayBuffer, off: number, frames: number, h: WavHead): Float32Array[] {
  const v = new DataView(buf);
  const bytes = h.bits / 8;
  const out = Array.from({ length: h.channels }, () => new Float32Array(frames));
  for (let i = 0; i < frames; i++) {
    for (let c = 0; c < h.channels; c++) {
      const p = off + (i * h.channels + c) * bytes;
      if (p + bytes > buf.byteLength) continue;
      out[c][i] = h.fmt === 3 ? (h.bits === 64 ? v.getFloat64(p, true) : v.getFloat32(p, true))
        : h.bits === 16 ? v.getInt16(p, true) / 0x8000
        : h.bits === 24 ? (((v.getUint8(p + 2) << 24) | (v.getUint8(p + 1) << 16) | (v.getUint8(p) << 8)) >> 8) / 0x800000
        : h.bits === 32 ? v.getInt32(p, true) / 0x80000000 : 0;
    }
  }
  return out;
}

export interface WavPiece { channels: Float32Array[]; sampleRate: number; /** с какой секунды файла кусок */ start: number }

/** Кусок WAV [from, to) секунд: заголовок и нужные байты — Range-запросами (движок их умеет), целиком не качаем. */
export async function fetchWavPiece(url: string, from: number, to: number, signal?: AbortSignal): Promise<WavPiece> {
  const r0 = await fetch(url, { headers: { Range: "bytes=0-65535" }, signal });
  if (!r0.ok) throw new Error(`фон не скачался (${r0.status})`);
  const headBuf = await r0.arrayBuffer();
  const h = wavHead(headBuf);
  const fb = h.channels * (h.bits / 8);
  const total = Math.floor((r0.status === 200 ? headBuf.byteLength - h.dataOff : h.dataLen) / fb);
  const f0 = Math.max(0, Math.min(total, Math.floor(from * h.sampleRate)));
  const f1 = Math.max(f0, Math.min(total, Math.ceil(to * h.sampleRate)));
  const start = f0 / h.sampleRate;
  if (r0.status === 200) return { channels: pcm(headBuf, h.dataOff + f0 * fb, f1 - f0, h), sampleRate: h.sampleRate, start };
  if (f1 === f0) return { channels: Array.from({ length: h.channels }, () => new Float32Array(0)), sampleRate: h.sampleRate, start };
  const b0 = h.dataOff + f0 * fb;
  const r1 = await fetch(url, { headers: { Range: `bytes=${b0}-${h.dataOff + f1 * fb - 1}` }, signal });
  if (!r1.ok) throw new Error(`фон не скачался (${r1.status})`);
  const buf = await r1.arrayBuffer();
  // сервер мог проигнорировать Range и прислать файл целиком
  return { channels: pcm(buf, r1.status === 206 ? 0 : b0, f1 - f0, h), sampleRate: h.sampleRate, start };
}

const takeCache = new Map<string, { samples: Float32Array; sampleRate: number; speech: Span[] }>();

async function loadTake(url: string, signal?: AbortSignal) {
  const hit = takeCache.get(url);
  if (hit) return hit;
  const r = await fetch(url, { signal });
  if (!r.ok) throw new Error(`дубль не скачался (${r.status})`);
  const buf = await r.arrayBuffer();
  const h = wavHead(buf);
  const frames = Math.floor(Math.min(h.dataLen, buf.byteLength - h.dataOff) / (h.channels * (h.bits / 8)));
  const ch = pcm(buf, h.dataOff, frames, h);
  const samples = ch.length === 1 ? ch[0] : ch[0].map((_, i) => ch.reduce((a, c) => a + c[i], 0) / ch.length);
  const v = { samples, sampleRate: h.sampleRate, speech: detectSpeech(samples, h.sampleRate) };
  if (takeCache.size > 12) takeCache.delete(takeCache.keys().next().value as string);
  takeCache.set(url, v);
  return v;
}

// ───────────── проигрывание: видео без звука + фон + дубль ─────────────
export interface ListenArgs {
  player: Player;
  /** видео сцены — им управляет player; null — играем только звук (демо) */
  video: HTMLVideoElement | null;
  takeUrl: string;
  bgUrl: string | null;
  take: Take;
  line: SceneLine;
  sceneDur: number;
  /** громкость фона и голоса из сведения, дБ */
  bgGainDb?: number;
  voiceGainDb?: number;
  onEnd: () => void;
}

export interface Listening { stop: () => void; from: number; to: number; placement: Placement; bg: boolean }

let current: (() => void) | null = null;
/** Остановить то, что играет сейчас (одновременно играет только одна реплика). */
export const stopListening = () => current?.();

const waitEvent = (el: HTMLElement, ev: string, ms: number) => new Promise<void>((res) => {
  const t = setTimeout(done, ms);
  function done() { clearTimeout(t); el.removeEventListener(ev, done); res(); }
  el.addEventListener(ev, done, { once: true });
});

const db2gain = (db = 0) => Math.pow(10, db / 20);
const LEAD_S = 0.06;   // запас на запуск звука, чтобы фон и голос стартовали точно в рассчитанный момент

export async function listenTake(a: ListenArgs): Promise<Listening> {
  stopListening();
  const ac = new AbortController();
  let ended = false;
  let ctx: AudioContext | null = null;
  let off: (() => void) | null = null;
  const srcs: AudioBufferSourceNode[] = [];
  const wasMuted = a.video?.muted ?? false;
  const stop = () => {
    if (ended) return;
    ended = true;
    if (current === stop) current = null;
    ac.abort();
    off?.();
    srcs.forEach((s) => { try { s.stop(); } catch { /* не стартовал */ } });
    void ctx?.close().catch(() => {});
    a.player.pause();
    if (a.video) a.video.muted = wasMuted;
    a.onEnd();
  };
  current = stop;
  try {
    const tk = await loadTake(a.takeUrl, ac.signal);
    const sr = tk.sampleRate;
    const sp = findTakeSpeech(tk.speech, tk.samples.length / sr, { lineStart: a.line.start_rel, lineEnd: a.line.end_rel, takeStart: a.take.start_rel, preroll: a.take.preroll });
    const p = takePlacement(a.take, sp, sr, tk.samples.length);
    const [from, to] = listenWindow(a.line, p, a.sceneDur);
    const voice = placedAudio(tk.samples, sr, a.take.stretch, p);
    let bg: WavPiece | null = null;
    if (a.bgUrl) {
      try { bg = await fetchWavPiece(a.bgUrl, from, to + LEAD_S, ac.signal); } catch (e) {
        if ((e as DOMException).name === "AbortError") throw e;
        bg = null;   // нет фона — играем голос с видео
      }
    }
    if (ended) throw new DOMException("aborted", "AbortError");
    ctx = new AudioContext();
    await ctx.resume();
    const mk = (chs: Float32Array[], rate: number) => {
      const b = ctx!.createBuffer(chs.length, Math.max(1, chs[0].length), rate);
      chs.forEach((c, i) => b.copyToChannel(c as Float32Array<ArrayBuffer>, i));
      return b;
    };
    // видео: без звука, с начала окна
    a.player.pause();
    if (a.video) a.video.muted = true;
    a.player.seek(from);
    if (a.video && a.video.readyState >= 1) await waitEvent(a.video, "seeked", 1500);
    if (ended) throw new DOMException("aborted", "AbortError");
    a.player.play(to);
    if (a.video && a.video.paused) await waitEvent(a.video, "playing", 1500);
    if (ended) throw new DOMException("aborted", "AbortError");
    const sceneNow = a.video && !a.video.paused ? a.video.currentTime : from;
    const t0 = ctx.currentTime + LEAD_S;
    const sceneAt0 = sceneNow + LEAD_S;        // какая секунда сцены будет в момент t0
    const out = ctx.destination;
    if (bg && bg.channels[0].length) {
      const s = ctx.createBufferSource();
      s.buffer = mk(bg.channels, bg.sampleRate);
      const g = ctx.createGain(); g.gain.value = db2gain(a.bgGainDb);
      s.connect(g).connect(out);
      const into = sceneAt0 - bg.start;
      if (into < s.buffer.duration) { s.start(t0, Math.max(0, into)); srcs.push(s); }
    }
    if (voice.length) {
      const s = ctx.createBufferSource();
      s.buffer = mk([voice], sr);
      const g = ctx.createGain(); g.gain.value = db2gain(a.voiceGainDb);
      s.connect(g).connect(out);
      const d = p.placed - sceneAt0;
      if (d >= 0) { s.start(t0 + d); srcs.push(s); } else if (-d < s.buffer.duration) { s.start(t0, -d); srcs.push(s); }
    }
    // конец окна или пауза плеера (пробел, «Смотреть», уход с экрана) — тишина
    off = a.player.subscribe(() => { if (!a.player.playing) stop(); });
    return { stop, from, to, placement: p, bg: !!bg };
  } catch (e) {
    stop();
    throw e;
  }
}
