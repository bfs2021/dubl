/**
 * Профиль микрофона «Мой микрофон»: расчёты для мастера проверки (MicCheck).
 * Чистые функции — замеры по массиву отсчётов, зона и совет по уровню. Без обращения к микрофону.
 */
import { peakDb, rmsDb, toDb } from "../audio/analyze";

/** Что сохраняем после проверки. Живёт на этом устройстве (ui.mic_profile) и, если движок умеет, в настройках. */
export interface MicProfile {
  checked_at: string;
  device_id: string | null;
  device_label: string | null;
  /** шум комнаты (среднее по 2 секундам тишины), дБ относительно максимума */
  noise_floor_db: number;
  /** громкий пик фразы, дБ */
  speech_peak_db: number;
  /** средняя громкость речи, дБ */
  speech_rms_db: number;
  /** на сколько речь громче шума, дБ */
  snr_db: number;
  /** задержка записи, мс; null — не мерили (человек в наушниках) */
  latency_ms: number | null;
  headphones: boolean;
  verdict: MicVerdict;
}

export type MicVerdict = "good" | "quiet" | "loud" | "noisy" | "silent";

export type Zone = "silent" | "low" | "good" | "hot" | "clip";

/** Границы цветной зоны на шкале (пик, дБ от максимума). Зелёная зона: −24…−8. */
export const ZONE_GOOD_MIN = -24;
export const ZONE_GOOD_MAX = -8;
export const ZONE_HOT = -3;
const ZONE_HEARD = -45;

export function zoneOf(db: number): Zone {
  if (db <= ZONE_HEARD) return "silent";
  if (db < ZONE_GOOD_MIN) return "low";
  if (db <= ZONE_GOOD_MAX) return "good";
  if (db <= ZONE_HOT) return "hot";
  return "clip";
}

export const ZONE_TEXT: Record<Zone, string> = {
  silent: "Тихо. Скажи фразу вслух",
  low: "Тихо — говори громче или придвинься ближе",
  good: "Отлично — так и говори",
  hot: "Громковато — чуть отодвинься",
  clip: "Слишком громко: звук будет рваться. Отодвинься от микрофона",
};

/** Шум комнаты: среднее RMS по тишине. */
export function noiseFloor(samples: Float32Array): number {
  return rmsDb(samples);
}

/**
 * Замеры речи: пик и средняя громкость только по «живым» окнам (громче шума на 10 дБ),
 * чтобы паузы не занижали среднее.
 */
export function speechStats(samples: Float32Array, sampleRate: number, noiseDb: number): { peak: number; rms: number; activeS: number } {
  const win = Math.max(1, Math.floor(sampleRate * 0.03));
  // «живое» окно — заметно громче шума, но не выше пика речи: в шумной комнате порог шума+10 мог бы отсечь вообще всё
  const peak = peakDb(samples);
  const thr = Math.max(-50, Math.min(noiseDb + 10, peak - 12));
  let acc = 0;
  let n = 0;
  let active = 0;
  for (let i = 0; i + win <= samples.length; i += win) {
    let s = 0;
    for (let j = i; j < i + win; j++) s += samples[j] * samples[j];
    const r = toDb(Math.sqrt(s / win));
    if (r >= thr) { acc += s; n += win; active++; }
  }
  return { peak, rms: n ? toDb(Math.sqrt(acc / n)) : -120, activeS: (active * win) / sampleRate };
}

export function verdictOf(noiseDb: number, speechPeak: number, speechRms: number): MicVerdict {
  if (speechPeak < -50) return "silent";
  if (speechPeak > -3) return "loud";
  if (speechPeak < -26) return "quiet";
  if (noiseDb > -45 || speechRms - noiseDb < 25) return "noisy";
  return "good";
}

export const VERDICT_TEXT: Record<MicVerdict, { title: string; advice: string }> = {
  good: { title: "Микрофон в порядке", advice: "Уровень и тишина в комнате подходят. Можно записывать." },
  quiet: { title: "Тихо", advice: "Придвинься ближе к микрофону (на ширину ладони) или добавь усиление в системных настройках звука." },
  loud: { title: "Слишком громко", advice: "Звук рвётся на пиках. Отодвинься от микрофона или убавь усиление в системных настройках звука." },
  noisy: { title: "В комнате шумно", advice: "Фон слышен рядом с голосом. Закрой окно, выключи вентилятор и придвинься ближе к микрофону." },
  silent: { title: "Не слышу голос", advice: "Проверь, что выбран нужный микрофон и он не выключен. Потом повтори проверку." },
};

export function buildProfile(p: {
  noiseDb: number; speechPeak: number; speechRms: number; latencyMs: number | null; headphones: boolean;
  deviceId: string | null; deviceLabel?: string | null; now?: Date;
}): MicProfile {
  return {
    checked_at: (p.now ?? new Date()).toISOString(),
    device_id: p.deviceId, device_label: p.deviceLabel ?? null,
    noise_floor_db: Math.round(p.noiseDb * 10) / 10,
    speech_peak_db: Math.round(p.speechPeak * 10) / 10,
    speech_rms_db: Math.round(p.speechRms * 10) / 10,
    snr_db: Math.max(0, Math.round((p.speechRms - p.noiseDb) * 10) / 10),
    latency_ms: p.latencyMs, headphones: p.headphones,
    verdict: verdictOf(p.noiseDb, p.speechPeak, p.speechRms),
  };
}

/** Проверен ли микрофон (есть профиль, и он годится для записи). */
export const hasMicProfile = (p: MicProfile | null | undefined): p is MicProfile => !!p && typeof p.checked_at === "string";

/** Одна строка для главного экрана: «Мой микрофон: в порядке · задержка 99 мс». */
export function profileSummary(p: MicProfile): string {
  const lat = p.latency_ms != null ? ` · задержка ${Math.round(p.latency_ms)} мс` : "";
  return `${VERDICT_TEXT[p.verdict].title}${lat}`;
}
