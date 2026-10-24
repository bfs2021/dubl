/** Время «м:сс.д» для разбора и «00:сс.д» для сцены. */
export function fmt(s: number): string {
  s = Math.max(0, s);
  const m = Math.floor(s / 60);
  return m + ":" + (s - m * 60).toFixed(1).padStart(4, "0");
}

/** Часы при необходимости: 1:31:07 */
export function fmtLong(s: number): string {
  s = Math.max(0, Math.round(s));
  const h = Math.floor(s / 3600);
  const m = Math.floor((s - h * 3600) / 60);
  const ss = s % 60;
  const p = (n: number) => String(n).padStart(2, "0");
  return h ? `${h}:${p(m)}:${p(ss)}` : `${m}:${p(ss)}`;
}

/** Время внутри сцены — в том же виде, что и везде: «м:сс.д» (0:00.0, 0:10.5). */
export const fmtRel = fmt;

/** Дробное число для человека — по-русски, с запятой: 22,2. Таймкоды (0:33.1) сюда не относятся. */
export const fmtNum = (x: number, digits = 1) => x.toFixed(digits).replace(".", ",");

/** Секунды для человека: «22,2 с». */
export const fmtSec = (x: number, digits = 1) => fmtNum(x, digits) + " с";

export const fmtDb = (v: number, digits = 1) => (v > 0 ? "+" : v < 0 ? "−" : "") + fmtNum(Math.abs(v), digits) + " дБ";

const MONTHS = ["января", "февраля", "марта", "апреля", "мая", "июня", "июля", "августа", "сентября", "октября", "ноября", "декабря"];

/** Дата и время по-русски: «24 октября в 14:20» (год — только если не текущий). */
export function fmtDate(iso: string, now = new Date()): string {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return "";
  const p = (n: number) => String(n).padStart(2, "0");
  const year = d.getFullYear() !== now.getFullYear() ? ` ${d.getFullYear()}` : "";
  return `${d.getDate()} ${MONTHS[d.getMonth()]}${year} в ${p(d.getHours())}:${p(d.getMinutes())}`;
}

export const fmtMs = (v: number) => (v > 0 ? "+" : v < 0 ? "−" : "") + Math.abs(Math.round(v)) + " мс";

export function plural(n: number, one: string, few: string, many: string): string {
  const a = Math.abs(n) % 100;
  const b = a % 10;
  if (a > 10 && a < 20) return many;
  if (b > 1 && b < 5) return few;
  if (b === 1) return one;
  return many;
}

export const fmtBytes = (n: number) =>
  n >= 1e9 ? fmtNum(n / 1e9) + " ГБ" : n >= 1e6 ? fmtNum(n / 1e6) + " МБ" : Math.max(1, Math.round(n / 1e3)) + " КБ";

/** Подсказка про отмену: ⌘Z на Mac, Ctrl+Z в остальных системах. */
export function undoKey(): string {
  const p = typeof navigator !== "undefined" ? navigator.platform || navigator.userAgent : "";
  return /Mac|iPhone|iPad/i.test(p) ? "⌘Z" : "Ctrl+Z";
}

export const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));

export const dbToGain = (db: number) => Math.pow(10, db / 20);
export const gainToDb = (g: number) => (g > 1e-6 ? 20 * Math.log10(g) : -120);

/** Устройство расчёта человеческими словами (в движке — mps / cuda / cpu). */
export function deviceName(d: string | null | undefined): string {
  const x = (d ?? "").toLowerCase();
  if (x === "mps") return "видеокарта Mac";
  if (x.startsWith("cuda")) return "видеокарта NVIDIA";
  if (x === "cpu") return "процессор";
  return d || "—";
}

const NOISE_TOKEN = /^(\d{3,4}p|\d{3,4}i|\d{3,4}|\d+k|x26[45]|h\.?26[45]|avc|hevc|xvid|divx|aac\d?|ac3|eac3|dts|flac|mp3|opus|st|stereo|mono|bluray|blu-ray|bdrip|brrip|web-?dl|web-?rip|webrip|hdrip|dvdrip|dvd|hdtv|remux|proper|repack|uhd|hdr\d*|sdr|10bit|8bit|multi|5\.1|7\.1|2\.0|mkv|mp4|mov|avi|webm|m4v)$/i;

/**
 * Название проекта из имени файла: без расширения, точек и подчёркиваний, без «720p / x264 / WEB-DL» и скобок с мусором.
 * Год (1900–2099) остаётся. Если после чистки ничего не осталось — возвращаем имя без расширения как есть.
 */
export function cleanTitle(fileName: string): string {
  const base = fileName.split(/[\\/]/).pop() ?? fileName;
  const noExt = base.replace(/\.(mp4|mkv|mov|avi|webm|m4v|ts|flv|wmv)$/i, "");
  let s = noExt.replace(/\[[^\]]*\]|\([^)]*(?:\d{3,4}p|x26[45]|web|rip|bluray)[^)]*\)/gi, " ");
  // «a.b.c» и «a_b_c» — это пробелы; «a-b-c» без пробелов — тоже (но «Spider-Man» с одним дефисом оставляем)
  s = s.replace(/[._]+/g, " ");
  if (!/\s/.test(s.trim()) && (s.match(/-/g) ?? []).length >= 2) s = s.replace(/-/g, " ");
  const words = s.split(/\s+/).filter(Boolean);
  const kept = words.filter((w) => {
    if (/^(19|20)\d\d$/.test(w)) return true;
    return !NOISE_TOKEN.test(w);
  });
  // «- RuTracker», «-GROUP» в конце
  while (kept.length > 1 && /^-\w*$/.test(kept[kept.length - 1])) kept.pop();
  const out = kept.join(" ").replace(/\s+-\s*$/, "").replace(/\s{2,}/g, " ").trim();
  return out || noExt.trim() || fileName;
}
