/**
 * Запись микрофона в браузере/окне Tauri: getUserMedia → AudioWorklet (или ScriptProcessor) → WAV 48 кГц mono.
 * Отсчёт — три сигнала через OscillatorNode. Предзапись (preroll) делает вызывающий:
 * startCapture() за `preroll` секунд до старта видео.
 */
import { encodeWav, wavBlob } from "./wav";
import { peakDb, rmsDb, toDb } from "./analyze";

export { encodeWav, wavBlob };
export const SAMPLE_RATE = 48000;

const WORKLET_SRC = `
class DublCapture extends AudioWorkletProcessor {
  process(inputs) {
    const ch = inputs[0] && inputs[0][0];
    if (ch && ch.length) this.port.postMessage(ch.slice(0));
    return true;
  }
}
registerProcessor('dubl-capture', DublCapture);
`;

export interface CaptureResult {
  samples: Float32Array;
  sampleRate: number;
  duration: number;
  peakDb: number;
  rmsDb: number;
}

export function concatChunks(chunks: Float32Array[]): Float32Array {
  let n = 0;
  for (const c of chunks) n += c.length;
  const out = new Float32Array(n);
  let o = 0;
  for (const c of chunks) {
    out.set(c, o);
    o += c.length;
  }
  return out;
}

export function finishCapture(chunks: Float32Array[], sampleRate: number): CaptureResult {
  const samples = concatChunks(chunks);
  return { samples, sampleRate, duration: samples.length / sampleRate, peakDb: peakDb(samples), rmsDb: rmsDb(samples) };
}

export async function listMicrophones(): Promise<MediaDeviceInfo[]> {
  if (!navigator.mediaDevices?.enumerateDevices) return [];
  const all = await navigator.mediaDevices.enumerateDevices();
  return all.filter((d) => d.kind === "audioinput");
}

export class MicRecorder {
  ctx: AudioContext | null = null;
  private stream: MediaStream | null = null;
  private source: MediaStreamAudioSourceNode | null = null;
  private node: AudioNode | null = null;
  private analyser: AnalyserNode | null = null;
  private buf: Float32Array<ArrayBuffer> | null = null;
  private chunks: Float32Array[] = [];
  private capturing = false;
  private onLevel?: (peakDb: number) => void;
  level = -120;

  get isOpen() {
    return !!this.stream;
  }

  async open(deviceId?: string | null, onLevel?: (db: number) => void): Promise<void> {
    if (this.stream) return;
    if (!navigator.mediaDevices?.getUserMedia) throw new Error("Браузер не умеет записывать звук");
    this.onLevel = onLevel;
    this.stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        deviceId: deviceId ? { exact: deviceId } : undefined,
        channelCount: 1,
        echoCancellation: false,
        noiseSuppression: false,
        autoGainControl: false,
      },
    });
    const Ctx = window.AudioContext ?? (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
    this.ctx = new Ctx({ sampleRate: SAMPLE_RATE, latencyHint: "interactive" });
    await this.ctx.resume();
    this.source = this.ctx.createMediaStreamSource(this.stream);
    // живой уровень для шкалы: AnalyserNode читаем по запросу (getLevel), он не зависит от записи
    this.analyser = this.ctx.createAnalyser();
    this.analyser.fftSize = 1024;
    this.buf = new Float32Array(this.analyser.fftSize);
    this.source.connect(this.analyser);
    const push = (data: Float32Array) => {
      let p = 0;
      for (let i = 0; i < data.length; i++) {
        const a = Math.abs(data[i]);
        if (a > p) p = a;
      }
      this.level = toDb(p);
      this.onLevel?.(this.level);
      if (this.capturing) this.chunks.push(data);
    };
    try {
      if (!this.ctx.audioWorklet) throw new Error("no worklet");
      const url = URL.createObjectURL(new Blob([WORKLET_SRC], { type: "application/javascript" }));
      await this.ctx.audioWorklet.addModule(url);
      URL.revokeObjectURL(url);
      const wn = new AudioWorkletNode(this.ctx, "dubl-capture", { numberOfInputs: 1, numberOfOutputs: 1, channelCount: 1 });
      wn.port.onmessage = (e: MessageEvent<Float32Array>) => push(e.data);
      this.source.connect(wn);
      // выход глушим: без подключения к destination некоторые движки не вызывают process()
      const mute = this.ctx.createGain();
      mute.gain.value = 0;
      wn.connect(mute).connect(this.ctx.destination);
      this.node = wn;
    } catch {
      const sp = this.ctx.createScriptProcessor(2048, 1, 1);
      sp.onaudioprocess = (e) => push(new Float32Array(e.inputBuffer.getChannelData(0)));
      this.source.connect(sp);
      const mute = this.ctx.createGain();
      mute.gain.value = 0;
      sp.connect(mute).connect(this.ctx.destination);
      this.node = sp;
    }
  }

  /** Текущий уровень (пик за последние ~20 мс) в дБ, −120 если микрофон закрыт. */
  getLevel(): number {
    if (!this.analyser || !this.buf) return -120;
    this.analyser.getFloatTimeDomainData(this.buf);
    let p = 0;
    for (let i = 0; i < this.buf.length; i++) {
      const a = Math.abs(this.buf[i]);
      if (a > p) p = a;
    }
    return toDb(p);
  }

  startCapture() {
    this.chunks = [];
    this.capturing = true;
  }

  stopCapture(): CaptureResult {
    this.capturing = false;
    const r = finishCapture(this.chunks, this.ctx?.sampleRate ?? SAMPLE_RATE);
    this.chunks = [];
    return r;
  }

  cancelCapture() {
    this.capturing = false;
    this.chunks = [];
  }

  get sampleRate() {
    return this.ctx?.sampleRate ?? SAMPLE_RATE;
  }

  async close() {
    this.capturing = false;
    this.node?.disconnect();
    this.analyser?.disconnect();
    this.source?.disconnect();
    this.stream?.getTracks().forEach((t) => t.stop());
    await this.ctx?.close().catch(() => {});
    this.ctx = null;
    this.stream = null;
    this.source = null;
    this.node = null;
    this.analyser = null;
    this.buf = null;
    this.level = -120;
  }
}

/** Сглаженный уровень для шкалы: быстро растёт, медленно падает; плюс «максимум» — чтобы короткий звук не пропадал. */
export interface LevelReading { db: number; peak: number; heard: boolean }
export function levelStep(prev: LevelReading, raw: number, dtMs: number): LevelReading {
  const fall = (dtMs / 1000) * 30; // 30 дБ в секунду
  const db = raw > prev.db ? raw : Math.max(raw, prev.db - fall);
  const peak = Math.max(prev.peak - (dtMs / 1000) * 6, raw);
  return { db, peak, heard: prev.heard || raw > -50 };
}

/**
 * Отсчёт: `beats` коротких сигналов с интервалом `interval` с (OscillatorNode).
 * Последний — выше тоном. Возвращает промис, который выполняется в момент последнего сигнала.
 */
export function playCountdown(ctx: AudioContext, onBeat: (n: number) => void, beats = 3, interval = 0.6, signal?: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    const t0 = ctx.currentTime + 0.05;
    for (let i = 0; i < beats; i++) {
      const osc = ctx.createOscillator();
      const g = ctx.createGain();
      osc.frequency.value = i === beats - 1 ? 1320 : 880;
      const t = t0 + i * interval;
      g.gain.setValueAtTime(0, t);
      g.gain.linearRampToValueAtTime(0.25, t + 0.01);
      g.gain.linearRampToValueAtTime(0, t + 0.11);
      osc.connect(g).connect(ctx.destination);
      osc.start(t);
      osc.stop(t + 0.14);
    }
    let n = 0;
    const iv = setInterval(() => {
      if (signal?.aborted) {
        clearInterval(iv);
        reject(new DOMException("aborted", "AbortError"));
        return;
      }
      n++;
      onBeat(n);
      if (n >= beats) {
        clearInterval(iv);
        resolve();
      }
    }, interval * 1000);
    // первый сигнал — сразу
    setTimeout(() => !signal?.aborted && onBeat(0), 0);
  });
}

export type CalibrationStep = "open" | "click" | "listen";
export const CALIBRATION_MS = 1500;

/**
 * Замер задержки на уже открытом микрофоне: играем щелчок через динамики и ищем его в записи.
 * Нужны колонки + микрофон (без наушников). Возвращает круговую задержку в мс.
 */
export async function measureLatency(rec: MicRecorder, onStep?: (s: CalibrationStep) => void): Promise<number> {
  const ctx = rec.ctx;
  if (!ctx) throw new Error("Микрофон не включён.");
  rec.startCapture();
  const t0 = ctx.currentTime + 0.3;
  // короткий бип 1,5 кГц: на динамиках ноутбука он слышнее и надёжнее находится, чем одиночный щелчок
  const n = Math.round(ctx.sampleRate * 0.05);
  const buf = ctx.createBuffer(1, n, ctx.sampleRate);
  const d = buf.getChannelData(0);
  for (let i = 0; i < n; i++) d[i] = 0.8 * Math.sin((2 * Math.PI * 1500 * i) / ctx.sampleRate) * (1 - i / n);
  const src = ctx.createBufferSource();
  src.buffer = buf;
  src.connect(ctx.destination);
  src.start(t0);
  onStep?.("click");
  await new Promise((r) => setTimeout(r, 400));
  onStep?.("listen");
  await new Promise((r) => setTimeout(r, CALIBRATION_MS - 400));
  const cap = rec.stopCapture();
  return findClick(cap.samples, cap.sampleRate);
}

/**
 * Замер задержки по сигналу движка (GET /mic/latency_signal.wav → POST /mic/latency): пишем микрофон,
 * через ~0,3 с играем эталонный сигнал, шлём запись. Нужны колонки без наушников. Возвращает мс и сообщение движка.
 */
export async function measureLatencyEngine(rec: MicRecorder, signal: ArrayBuffer, send: (wav: Blob, playOffsetS: number) => Promise<{ latency_ms?: number | null; ok: boolean; message: string }>, recordS = 3.2): Promise<number> {
  const ctx = rec.ctx;
  if (!ctx) throw new Error("Микрофон не включён.");
  if (!signal.byteLength) throw new Error("Нет эталонного сигнала.");
  const buf = await ctx.decodeAudioData(signal.slice(0));
  rec.startCapture();
  const lead = 0.3;
  const src = ctx.createBufferSource();
  src.buffer = buf;
  src.connect(ctx.destination);
  src.start(ctx.currentTime + lead);
  await new Promise((r) => setTimeout(r, recordS * 1000));
  const cap = rec.stopCapture();
  const res = await send(wavBlob(cap.samples, cap.sampleRate), lead);
  if (!res.ok || res.latency_ms == null) throw new Error(res.message);
  return Math.round(res.latency_ms);
}

/** Калибровка задержки в Настройках: открывает микрофон сама. onStep сообщает, что сейчас происходит. */
export async function calibrateLatency(deviceId?: string | null, onStep?: (s: CalibrationStep) => void): Promise<number> {
  const rec = new MicRecorder();
  onStep?.("open");
  try {
    await rec.open(deviceId);
  } catch (e) {
    throw new Error(micErrorText(e));
  }
  try {
    return await measureLatency(rec, onStep);
  } finally {
    await rec.close();
  }
}

/** Ищем щелчок: порог — 5× от шума в первые 0.2 с. Бросает понятную ошибку, если не нашли. */
export function findClick(samples: Float32Array, sampleRate: number): number {
  const n0 = Math.floor(sampleRate * 0.2);
  let noise = 1e-4;
  for (let i = 0; i < Math.min(n0, samples.length); i++) noise = Math.max(noise, Math.abs(samples[i]));
  const from = Math.floor(0.3 * sampleRate);
  for (let i = from; i < samples.length; i++) {
    if (Math.abs(samples[i]) > noise * 5 && Math.abs(samples[i]) > 0.01) {
      return Math.max(0, Math.round(((i - from) / sampleRate) * 1000));
    }
  }
  const peak = samples.reduce((m, v) => Math.max(m, Math.abs(v)), 0);
  throw new Error(peak < 0.003
    ? "Микрофон почти ничего не слышит. Проверь, что выбран нужный микрофон и он не выключен, потом повтори."
    : "Щелчок не услышан. Сними наушники, включи колонки погромче и повтори.");
}

/** Как включить микрофон для «Дубля» в системных настройках (когда доступ уже запретили). */
export const MIC_DENIED_HELP = "Доступ к микрофону запрещён. Открой «Системные настройки → Конфиденциальность и безопасность → Микрофон», включи «Дубль» и вернись сюда.";
/** Пока висит системное окно с вопросом — человек его ещё не нажал. */
export const MIC_ASKING_TEXT = "Разреши доступ к микрофону в окне macOS — оно появилось поверх программы.";

/** Что сказать человеку, если микрофон не открылся. */
export function micErrorText(e: unknown): string {
  const err = e as DOMException;
  if (err?.name === "NotAllowedError" || err?.name === "SecurityError") return MIC_DENIED_HELP;
  if (err?.name === "NotFoundError") return "Микрофон не найден. Подключи его и повтори.";
  if (err?.name === "NotReadableError") return "Микрофон занят другой программой. Закрой её и повтори.";
  return err?.message || "Микрофон не включился.";
}

export type MicPermission = "granted" | "denied" | "prompt" | "unknown";

/** Что система говорит про доступ к микрофону (Permissions API; где его нет — «unknown»). */
export async function micPermission(): Promise<MicPermission> {
  try {
    const st = await (navigator.permissions as Permissions | undefined)?.query({ name: "microphone" as PermissionName });
    return st ? (st.state as MicPermission) : "unknown";
  } catch {
    return "unknown";
  }
}

/**
 * Состояние микрофона для подписи на экране записи.
 *  - on — открыт; asking — ждём ответа на системный запрос; denied — доступ запретили;
 *  - off — просто выключен (можно включить кнопкой).
 */
export type MicUi = "on" | "asking" | "denied" | "off";

export function micUi(opts: { open: boolean; opening: boolean; permission: MicPermission; failed: boolean }): MicUi {
  if (opts.open) return "on";
  if (opts.permission === "denied" || (opts.failed && opts.permission !== "prompt")) return "denied";
  if (opts.opening || opts.permission === "prompt") return opts.opening ? "asking" : "off";
  return "off";
}

export const MIC_UI_TEXT: Record<MicUi, string> = {
  on: "Микрофон включён",
  asking: MIC_ASKING_TEXT,
  denied: MIC_DENIED_HELP,
  off: "Микрофон не включён",
};
