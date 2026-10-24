/**
 * Правила помощника-режиссёра для демо-режима (в бою это делает ИИ по замерам, см. llm/director.py).
 * Помощник не слышит звук: он видит только цифры — сдвиг, длину, громкость, распознанный текст.
 */
import type { DirectorCommandResult, DirectorReport, DirectorTakeResult, MixPatch, MixSettings, Speaker, Take, TrackSettings } from "../api/types";
import { ROOMS, VOICE_FX, defaultTrack, fxTitle, roomTitle } from "../api/presets";
import { fmtNum, fmtSec } from "./format";

export type Severity = "ok" | "warn";

/** Что показать под дублем: главный совет (первая фраза) и «ещё» (остальное), чтобы не сыпать советами сразу. */
export function remarkOf(r: Pick<DirectorTakeResult, "feedback" | "report">): { text: string; more?: string; severity: Severity } {
  const warn = !r.report.ok;
  const full = r.feedback?.trim() || (warn ? "Есть замечания к дублю." : "Хороший дубль.");
  const parts = full.split(/(?<=[.!?…])\s+(?=[А-ЯЁA-Z«])/);
  return { text: parts[0], more: parts.length > 1 ? parts.slice(1).join(" ") : undefined, severity: warn ? "warn" : "ok" };
}

const sec = (ms: number) => fmtNum(Math.abs(ms) / 1000);

/**
 * Подпись к дублю человеческими словами (вместо «+3049 мс · 85%»). Знак — как в движке (models.Take):
 * offset_ms > 0 — дубль пришлось сдвинуть позже, значит человек начал РАНЬШЕ губ; < 0 — начал позже.
 */
export function takeCaption(t: Pick<Take, "duration" | "offset_ms" | "stretch" | "fit_confidence">, lips = true): { main: string; timing: string; length: string | null; match: string | null } {
  const main = fmtSec(t.duration);
  // закадровый режим: попадать в губы не нужно, говорим только про длину
  if (!lips) return { main, timing: "Закадровый: губы не важны", length: null, match: null };
  const off = t.offset_ms;
  // движок не нашёл, где человек говорит (совпадение ~0) — «попал в губы» писать нельзя
  const lost = t.fit_confidence != null && t.fit_confidence < 0.05;
  const timing = lost ? "Не нашёл, где ты говоришь" : Math.abs(off) < 150 ? "Попал в движения губ" : off > 0 ? `Начал на ${sec(off)} с раньше оригинала` : `Начал на ${sec(off)} с позже оригинала`;
  const dev = Math.round(Math.abs(t.stretch - 1) * 100);
  const length = dev < 3 ? null : t.stretch < 1 ? `Длиннее оригинала: сжал на ${dev}%` : `Короче оригинала: растянул на ${dev}%`;
  const match = t.fit_confidence == null || lost ? null : `Совпадение ${Math.round(t.fit_confidence * 100)}%`;
  return { main, timing, length, match };
}

/** Демо: замеры дубля в той же форме, что отдаёт движок (llm.director.take_report). */
export function takeReport(take: Pick<Take, "id" | "offset_ms" | "stretch" | "peak_db" | "fit_confidence" | "loudness_lufs"> & { source?: Take["source"] }, lineId: string, lineDuration: number | null): DirectorReport {
  const source = take.source ?? "mic";
  // как в движке: попадание в губы проверено только у дубля с микрофона и при уверенной подгонке
  const lips_checked = source === "mic" && (take.fit_confidence ?? 0) >= 0.4;
  const off = take.offset_ms;
  const timing = off <= -150 ? "late" : off >= 150 ? "early" : "ok";
  const length = lineDuration && take.stretch < 0.9 ? "too_long" : take.stretch > 1.1 ? "too_short" : "ok";
  const peak = take.peak_db ?? null;
  const loudness = peak !== null && peak > -1 ? "clipping" : peak !== null && peak < -24 ? "too_quiet" : "ok";
  const problems: string[] = [];
  if (timing !== "ok") problems.push(timing === "late" ? "вступил поздно" : "вступил рано");
  if (length === "too_long") problems.push("длиннее оригинала");
  if (loudness !== "ok") problems.push(loudness === "clipping" ? "звук рвётся" : "тихо");
  return {
    line_id: lineId, take_id: take.id, offset_ms: off, timing, timing_level: Math.abs(off) >= 300 ? "big" : timing === "ok" ? "ok" : "noticeable",
    stretch: take.stretch, length, length_level: length === "ok" ? "ok" : "noticeable", fit_confidence: take.fit_confidence ?? null,
    peak_db: peak, loudness_lufs: take.loudness_lufs ?? null, loudness, missing_words: [], problems, ok: problems.length === 0,
    source, lips_checked,
  };
}

/** Демо: слова помощника по замерам (в бою это делает ИИ). */
export function takeRemark(take: Pick<Take, "offset_ms" | "stretch" | "peak_db" | "duration" | "fit_confidence">, lineDuration: number | null, file = false): { text: string; severity: Severity } {
  const t = { ...take, peak_db: take.peak_db ?? null, fit_confidence: take.fit_confidence ?? null };
  // Знаки — как в движке (models.Take): offset_ms < 0 — человек говорил позже губ; stretch < 1 — дубль пришлось сжать (он длиннее оригинала).
  if (t.peak_db !== null && t.peak_db < -50) return { severity: "warn", text: "Почти ничего не слышно. Проверь микрофон и подойди ближе." };
  if (t.peak_db !== null && t.peak_db > -1) return { severity: "warn", text: "Звук рвётся на пиках. Говори чуть тише или отодвинься от микрофона." };
  // файл к губам не привязан: про вступление, длину и «попал в губы» не говорим — только про громкость
  if (file) {
    if (t.peak_db !== null && t.peak_db < -24) return { severity: "warn", text: "Тихая запись. Я подниму громкость, но может стать шумно." };
    return { severity: "ok", text: "Дубль из файла: громкость в порядке. Попадание в губы для файла не проверяем." };
  }
  if (t.fit_confidence !== null && t.fit_confidence < 0.3) return { severity: "warn", text: "Не нашёл, где ты говоришь. Попробуй ещё раз, чётче." };
  if (t.offset_ms <= -150) return { severity: "warn", text: "Вступил поздно — начни чуть раньше. Я подвинул, но лучше записать заново." };
  if (t.offset_ms >= 150) return { severity: "warn", text: "Вступил рано — подожди полсекунды после сигнала." };
  if (t.stretch < 0.9 && lineDuration) return { severity: "warn", text: "Длиннее оригинала. Сократи текст кнопкой «Подогнать под длину» или говори быстрее." };
  if (t.stretch > 1.1) return { severity: "ok", text: "Получилось короче оригинала — можно говорить медленнее, с паузой." };
  if (t.peak_db !== null && t.peak_db < -24) return { severity: "warn", text: "Тихо. Говори громче — я подниму, но будет шумно." };
  return { severity: "ok", text: "Хороший дубль: попал в движения губ и по громкости ровно." };
}

const STEMS = (s: string) => s.toLowerCase().replace(/[^a-zа-яё0-9]/g, "");
const stemOf = (s: string) => { const x = STEMS(s); return x.length > 4 ? x.slice(0, -2) : x; };

const ROOM_WORDS: [RegExp, string][] = [
  [/пещер/, "cave"], [/кухн/, "kitchen"], [/зал/, "hall"], [/улиц|на улице|снаружи/, "street"], [/ванн/, "bathroom"],
  [/машин|авто/, "car"], [/телефон/, "phone"], [/из-за двери|за дверью/, "behind_door"], [/комнат/, "room"],
  [/без эха|сухо|без реверб/, "dry"],
];
const FX_WORDS: [RegExp, string][] = [
  [/робот/, "robot"], [/бурундук|писклив|тонк/, "chipmunk"], [/гигант|басом|низк/, "giant"], [/злодей/, "villain"],
  [/рация|рации/, "radio"], [/призрак/, "ghost"],
];

/**
 * Демо: «кот громче», «как в пещере», «фон потише» → изменения настроек (значения абсолютные, как у движка).
 * Без выдумок: если не поняли — пустой список.
 */
export function parseCommand(text: string, speakers: Speaker[], mix: MixSettings): DirectorCommandResult {
  const t = text.toLowerCase();
  const patches: MixPatch[] = [];
  const louder = /громч|погромч|усил|выше/.test(t);
  const quieter = /тише|потише|тих|приглуш|убав|ниже/.test(t);
  const amount = Number(t.match(/на\s*(\d+(?:[.,]\d+)?)\s*(?:дб|db)/)?.[1]?.replace(",", ".")) || 3;
  const gain = (id: string) => (id === "background" ? mix.background : mix.tracks[id])?.gain_db ?? 0;

  // кому
  const names: { id: string; title: string }[] = speakers.map((s) => ({ id: s.id, title: s.name ?? s.label }));
  const aliases: Record<string, string[]> = { кот: ["барсик"], кошк: ["барсик"], робот: ["пипс", "робот"], дядя: ["гена"] };
  const hit = names.filter((n) => {
    const st = stemOf(n.title);
    if (st.length >= 3 && t.includes(st)) return true;
    return Object.entries(aliases).some(([w, al]) => t.includes(w) && al.some((a) => STEMS(n.title).includes(a)));
  });
  const isBg = /фон|музык|шум/.test(t);
  const isMaster = /всё|все голоса|общая|итог/.test(t);
  const d = louder ? amount : -amount;

  if ((louder || quieter) && (hit.length || isBg)) {
    if (isBg) patches.push({ target: "background", field: "gain_db", value: gain("background") + d, note: `Фон ${louder ? "громче" : "тише"} на ${amount} дБ` });
    for (const h of hit) patches.push({ target: h.id, field: "gain_db", value: gain(h.id) + d, note: `${h.title} ${louder ? "громче" : "тише"} на ${amount} дБ` });
  } else if ((louder || quieter) && isMaster) {
    patches.push({ target: "master", field: "target_lufs", value: mix.target_lufs + (louder ? 1 : -1), note: `Итог ${louder ? "громче" : "тише"} на 1 LU` });
  }

  const room = ROOM_WORDS.find(([re]) => re.test(t));
  if (room && /как|будто|в |на |из|сухо|без/.test(t)) {
    const targets = hit.length ? hit.map((h) => h.id) : speakers.map((s) => s.id);
    const title = ROOMS.find((r) => r.id === room[1])?.title ?? room[1];
    for (const id of targets) {
      const nm = names.find((n) => n.id === id)?.title ?? id;
      patches.push({ target: id, field: "room", value: room[1], note: `${nm}: пространство «${title}»` });
    }
  }
  const fx = FX_WORDS.find(([re]) => re.test(t));
  if (fx && hit.length) {
    const title = VOICE_FX.find((r) => r.id === fx[1])?.title ?? fx[1];
    for (const h of hit) patches.push({ target: h.id, field: "voice_fx", value: fx[1], note: `${h.title}: эффект «${title}»` });
  }
  if (/без эффект|обычн(ый|ым) голос/.test(t) && hit.length) {
    for (const h of hit) patches.push({ target: h.id, field: "voice_fx", value: "none", note: `${h.title}: без эффекта` });
  }
  return { patches, mix: applyPatches(mix, patches, defaultTrack), ai_fallback: false };
}

/** Применить изменения к настройкам (не меняя исходный объект). Как llm.director.apply_patches в движке. */
export function applyPatches(mix: MixSettings, patches: MixPatch[], defaults: () => TrackSettings = defaultTrack): MixSettings {
  const out = structuredClone(mix);
  for (const p of patches) {
    if (p.target === "master") {
      if (p.field === "target_lufs") out.target_lufs = Number(p.value);
      continue;
    }
    if (p.field === "duck_background") { out.duck_background = !!p.value; continue; }
    if (p.field === "mode") { out.mode = p.value === "voiceover" ? "voiceover" : "dub"; continue; }
    const tr: TrackSettings = p.target === "background" ? out.background : (out.tracks[p.target] ??= defaults());
    if (p.field === "gain_db") tr.gain_db = Math.round(Number(p.value) * 2) / 2;
    else if (p.field === "mute") tr.mute = !!p.value;
    else if (p.field === "room") tr.room = String(p.value);
    else if (p.field === "voice_fx") tr.voice_fx = String(p.value);
  }
  if (patches.length) out.autopilot = false; // ручная правка выключает автопилот (как в движке)
  return out;
}

/** Человеческое описание правки — запасной вариант, если `note` пустой. */
export function describePatch(p: Pick<MixPatch, "target" | "field" | "value">, speakers: Speaker[]): string {
  const who =
    p.target === "background" ? "Фон" : p.target === "master" ? "Итог"
      : (() => { const s = speakers.find((x) => x.id === p.target); return s ? (s.name ?? s.label) : p.target; })();
  switch (p.field) {
    case "gain_db": return `${who}: громкость ${Number(p.value)} дБ`;
    case "mute": return p.value ? `${who}: выключен` : `${who}: включён`;
    case "room": return `${who}: пространство «${roomTitle(String(p.value))}»`;
    case "voice_fx": return `${who}: эффект «${fxTitle(String(p.value))}»`;
    case "target_lufs": return `Итог: ${p.value} LUFS`;
    case "duck_background": return p.value ? "Фон приглушается под голосом" : "Фон не приглушается";
    case "mode": return p.value === "voiceover" ? "Режим: закадровый" : "Режим: дубляж";
  }
}
