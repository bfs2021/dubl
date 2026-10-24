import { describe, expect, it } from "vitest";
import { MockEngine, mockProbe, shortenVariants, synthVoice } from "./mock";
import { applyPatches, parseCommand, takeRemark, describePatch } from "../lib/director";
import { defaultMix, defaultTrack } from "./presets";
import { encodeWav } from "../audio/wav";
import { fitInfo, estimateDuration } from "../lib/estimate";

describe("mockProbe", () => {
  it("находит дорожки, субтитры и главы в «богатом» файле", () => {
    const m = mockProbe("Серия.mkv");
    expect(m.audio_streams.length).toBeGreaterThan(1);
    expect(m.audio_streams.some((a) => a.channel_layout === "5.1")).toBe(true);
    expect(m.subtitle_streams.some((s) => s.codec === "ass")).toBe(true);
    expect(m.subtitle_streams.some((s) => !s.is_text)).toBe(true);
    expect(m.chapters.length).toBeGreaterThan(0);
  });
  it("в простом клипе ничего нет", () => {
    const m = mockProbe("клип.mp4");
    expect(m.audio_streams).toHaveLength(1);
    expect(m.subtitle_streams).toHaveLength(0);
    expect(m.chapters).toHaveLength(0);
  });
});

describe("MockEngine", () => {
  it("создаёт проект из файла, анализ завершается по таймерам", async () => {
    const e = new MockEngine({ speed: 80, seed: false });
    const { project, job } = await e.createProject({ path: "/tmp/Новая серия.mkv" });
    expect(project.analysis).toBeNull();
    expect(job?.steps.length).toBe(6);
    await new Promise((r) => setTimeout(r, 1500));
    const p = await e.getProject(project.id);
    expect(p.analysis?.lines.length).toBeGreaterThan(10);
    e.dispose();
  });
  it("snap совпадает с lib/snap", async () => {
    const e = new MockEngine({ speed: 80 });
    const r = await e.snap("p-fridge", { a: 20, mode: "shot" });
    expect(r.end).toBeGreaterThan(r.start);
    e.dispose();
  });
  it("загруженный дубль получает замеры и становится выбранным", async () => {
    const e = new MockEngine({ speed: 80 });
    const x = synthVoice(2, 1, 5);
    const sc = (await e.getProject("p-fridge")).scenes[0];
    const line = sc.lines[2];
    const t = await e.uploadTake("p-fridge", "sc001", new Blob([encodeWav(x, 48000)], { type: "audio/wav" }), { line_id: line.line_id, start_rel: Math.max(0, line.start_rel - 2), preroll: 1 });
    expect(t.selected).toBe(true);
    expect(t.peak_db).toBeLessThan(0);
    expect(t.duration).toBeGreaterThan(3);
    e.dispose();
  });
  it("рендер: одна задача render_project, путь файла в result", async () => {
    const e = new MockEngine({ speed: 80 });
    const settings = { aspect: "16:9" as const, height: 1080, target_lufs: -14, subtitles: "none" as const, subtitle_text: "dub" as const, fit: "blur" as const, export_stems: false, output_path: "x.mp4" };
    const one = await e.render("p-park", { scene_ids: ["sc001"], settings });
    expect(one.kind).toBe("render_project");
    expect(one.result).toMatchObject({ path: "x.mp4" });
    expect(one.steps.some((s) => s.key === "concat")).toBe(false);
    await e.createScene("p-fridge", { start: 118, end: 140, title: "Вторая" });
    const two = await e.render("p-fridge", { scene_ids: ["sc001", "sc002"], settings });
    expect(two.steps.some((s) => s.key === "concat")).toBe(true);
    e.dispose();
  });
  it("задача отдаёт общий монотонный progress и eta_s", async () => {
    const e = new MockEngine({ speed: 12 });
    const job = await e.render("p-park", { scene_ids: ["sc001"], settings: { aspect: "16:9", height: 1080, target_lufs: -14, subtitles: "none", subtitle_text: "dub", fit: "blur", export_stems: false, output_path: "x.mp4" } });
    const seen: number[] = [];
    const off = e.subscribeJobs((j) => { if (j.id === job.id) seen.push(j.progress); });
    await new Promise((r) => setTimeout(r, 2500));
    off();
    expect(seen.length).toBeGreaterThan(1);
    expect(seen.every((v, i) => i === 0 || v >= seen[i - 1])).toBe(true);
    expect(seen.at(-1)).toBe(1);
    e.dispose();
  });
  it("движок отвечает в форме схемы: translations / variants / report+feedback / patches+mix", async () => {
    const e = new MockEngine({ speed: 80 });
    const sc = (await e.getProject("p-fridge")).scenes[0];
    const tr = await e.translate({ project_id: "p-fridge", scene_id: "sc001" });
    expect(tr.translations.map((t) => t.line_id)).toContain(sc.lines[0].line_id);
    const fit = await e.fitLength({ project_id: "p-fridge", scene_id: "sc001", line_ids: [sc.lines[5].line_id] });
    expect(fit.variants[0]).toMatchObject({ line_id: sc.lines[5].line_id });
    expect(fit.variants[0].texts.length).toBeGreaterThan(0);
    const d = await e.directorTake({ project_id: "p-fridge", scene_id: "sc001", take_id: sc.takes[0].id });
    expect(d.report.take_id).toBe(sc.takes[0].id);
    expect(typeof d.feedback).toBe("string");
    const c = await e.directorCommand({ project_id: "p-fridge", scene_id: "sc001", text: "кот громче" });
    expect(c.patches[0]).toMatchObject({ target: "s1", field: "gain_db", note: expect.stringContaining("громче") });
    expect(c.mix.autopilot).toBe(false);
    e.dispose();
  });
});

describe("помощник-режиссёр (rule-based)", () => {
  const speakers = [
    { id: "s1", label: "Барсик", name: "Барсик", color: "#fff", line_count: 1, total_speech: 1 },
    { id: "s2", label: "Мила", name: "Мила", color: "#fff", line_count: 1, total_speech: 1 },
  ];
  const mix0 = defaultMix();
  it("«кот громче» → Барсик +3 дБ (значение абсолютное)", () => {
    const r = parseCommand("кот громче", speakers, mix0);
    expect(r.patches).toEqual([expect.objectContaining({ target: "s1", field: "gain_db", value: 3 })]);
    expect(r.mix.tracks.s1.gain_db).toBe(3);
  });
  it("«как в пещере» → комната всем", () => {
    const r = parseCommand("как в пещере", speakers, mix0);
    expect(r.patches.map((p) => p.target).sort()).toEqual(["s1", "s2"]);
    expect(r.patches.every((p) => p.field === "room" && p.value === "cave")).toBe(true);
  });
  it("непонятное не выдумывает правок", () => {
    const r = parseCommand("спой песню", speakers, mix0);
    expect(r.patches).toHaveLength(0);
    expect(r.mix.autopilot).toBe(mix0.autopilot);
  });
  it("применяет правку, не меняя исходные настройки", () => {
    const m = defaultMix();
    const r = parseCommand("фон потише", speakers, m);
    const out = applyPatches(m, r.patches, defaultTrack);
    expect(out.background.gain_db).toBe(-3);
    expect(m.background.gain_db).toBe(0);
    expect(describePatch({ target: "s1", field: "room", value: "cave" }, speakers)).toContain("Пещера");
  });
  it("замечания по замерам", () => {
    expect(takeRemark({ offset_ms: -300, stretch: 1, peak_db: -8, duration: 2, fit_confidence: 0.9 }, 2).text).toMatch(/поздно/);
    expect(takeRemark({ offset_ms: 0, stretch: 0.8, peak_db: -8, duration: 2, fit_confidence: 0.9 }, 2).text).toMatch(/Длиннее/);
    expect(takeRemark({ offset_ms: 0, stretch: 1, peak_db: -3, duration: 2, fit_confidence: 0.9 }, 2).severity).toBe("ok");
    expect(takeRemark({ offset_ms: 0, stretch: 1, peak_db: -70, duration: 2, fit_confidence: null }, 2).severity).toBe("warn");
  });
});

describe("«Уложить по длине»", () => {
  it("сокращает до влезающего", () => {
    const text = "Это очень длинная реплика, которая просто совсем не влезает в отведённое время.";
    const v = shortenVariants(text, 2.0);
    expect(v.length).toBeGreaterThan(0);
    expect(estimateDuration(v[0], "ru")).toBeLessThan(estimateDuration(text, "ru"));
    expect(fitInfo(v[v.length - 1], 2.0).ratio).toBeLessThan(fitInfo(text, 2.0).ratio);
  });
});
