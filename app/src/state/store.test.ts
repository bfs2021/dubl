import { beforeEach, describe, expect, it } from "vitest";
import { freshStore, openFridge, wait } from "../test/helpers";
import { activeJob, jobProgress, useStore } from "./store";

describe("store + мок-движок", () => {
  beforeEach(async () => { await freshStore(); });

  it("подключается и грузит проекты, настройки", async () => {
    const s = useStore.getState();
    expect(s.demo).toBe(true);
    expect(s.projects.map((p) => p.id)).toContain("p-fridge");
    expect(s.settings?.llm.providers.length).toBeGreaterThan(0);
  });

  it("открывает проект с серией 22:41 и 5+ персонажами", async () => {
    await openFridge();
    const p = useStore.getState().project!;
    expect(p.media.duration).toBeCloseTo(1361, 0);
    expect(p.analysis!.speakers.length).toBeGreaterThanOrEqual(5);
    expect(p.analysis!.shot_cuts.length).toBeGreaterThan(20);
  });

  it("создаёт сцену и ведёт задачу подготовки по таймерам", async () => {
    await openFridge();
    const sc = await useStore.getState().createScene(118, 133, "Тест");
    expect(sc?.title).toBe("Тест");
    expect(sc!.lines.length).toBeGreaterThan(0);
    const job = activeJob(useStore.getState().jobs, "prepare_scene", "p-fridge", sc!.id)!;
    expect(job).toBeTruthy();
    await wait(1500);
    const j2 = activeJob(useStore.getState().jobs, "prepare_scene", "p-fridge", sc!.id)!;
    expect(j2.state).toBe("done");
    expect(jobProgress(j2)).toBe(1);
    await wait(100);
    expect(useStore.getState().project!.scenes.find((s) => s.id === sc!.id)!.stems_ready).toBe(true);
  });

  it("отклоняет некорректные границы сцены плашкой ошибки", async () => {
    await openFridge();
    const r = await useStore.getState().createScene(50, 40, "плохая");
    expect(r).toBeNull();
    expect(useStore.getState().toasts.some((t) => t.kind === "error")).toBe(true);
  });

  it("переключает «Свой текст» ↔ «Как есть» и возвращает текст", async () => {
    await openFridge();
    const st = useStore.getState();
    const lid = st.project!.scenes[0].lines[0].line_id;
    const before = useStore.getState().project!.scenes[0].lines[0].text;
    await st.setTextMode("custom");
    expect(useStore.getState().project!.scenes[0].text_mode).toBe("custom");
    await useStore.getState().setLineText(lid, "Мой свой текст");
    expect(useStore.getState().project!.scenes[0].lines[0].text).toBe("Мой свой текст");
    await useStore.getState().setTextMode("as_is");
    const sc = useStore.getState().project!.scenes[0];
    expect(sc.text_mode).toBe("as_is");
    expect(sc.lines[0].text).toBe(before);
  });

  it("перевод — только по кнопке: до неё текст оригинальный", async () => {
    await openFridge(null);
    const sc = await useStore.getState().createScene(118, 140, "Перевод");
    expect(sc!.lines.every((l) => l.text === l.original)).toBe(true);
    useStore.getState().setSceneId(sc!.id);
    await useStore.getState().translateScene();
    const after = useStore.getState().project!.scenes.find((s) => s.id === sc!.id)!;
    expect(after.lines.some((l) => l.text !== l.original)).toBe(true);
  });

  it("«Уложить по длине» отдаёт варианты не длиннее оригинала", async () => {
    await openFridge();
    const l = useStore.getState().project!.scenes[0].lines[5];
    const v = (await useStore.getState().fitLine(l.line_id)) ?? [];
    expect(v.length).toBeGreaterThan(0);
    v.forEach((t) => expect(t.length).toBeLessThanOrEqual(l.text.length));
  });

  it("выбор дубля снимает звезду с остальных дублей той же реплики", async () => {
    await openFridge();
    const sc = useStore.getState().project!.scenes[0];
    const first = sc.takes.filter((t) => t.line_id === sc.lines[0].line_id);
    expect(first.length).toBe(2);
    const notSel = first.find((t) => !t.selected)!;
    await useStore.getState().selectTake(notSel.id, true);
    const after = useStore.getState().project!.scenes[0].takes.filter((t) => t.line_id === sc.lines[0].line_id);
    expect(after.filter((t) => t.selected)).toHaveLength(1);
    expect(after.find((t) => t.id === notSel.id)!.selected).toBe(true);
  });

  it("ручная правка сведения сохраняется в сцене", async () => {
    await openFridge();
    const mix = useStore.getState().project!.scenes[0].mix;
    await useStore.getState().setMix({ ...mix, autopilot: false, target_lufs: -16 });
    const m = useStore.getState().project!.scenes[0].mix;
    expect(m.autopilot).toBe(false);
    expect(m.target_lufs).toBe(-16);
  });

  it("слияние голосов переносит реплики", async () => {
    await openFridge();
    await useStore.getState().mergeSpeakers("s1", "s6");
    const an = useStore.getState().project!.analysis!;
    expect(an.speakers.some((s) => s.id === "s6")).toBe(false);
    expect(an.lines.some((l) => l.speaker_id === "s6")).toBe(false);
  });

  it("настройки устройства живут в localStorage", () => {
    useStore.getState().setUi({ headphones: true, lead_s: 4 });
    expect(JSON.parse(localStorage.getItem("dubl.ui")!).lead_s).toBe(4);
  });

  it("тосты не дублируются: одинаковый показан один раз", () => {
    const s = useStore.getState();
    s.toast("Привет", "ok"); s.toast("Привет", "ok"); s.toast("Привет", "ok");
    expect(useStore.getState().toasts.filter((t) => t.text === "Привет")).toHaveLength(1);
  });

  it("ошибка сервера в тосте — по-человечески и с «Повторить»", async () => {
    const e = useStore.getState().engine!;
    e.listProjects = async () => { throw new Error("Not Found"); };
    await useStore.getState().loadProjects();
    const t = useStore.getState().toasts.at(-1)!;
    expect(t.text).not.toBe("Not Found");
    expect(t.text).toMatch(/Перезапусти/);
    expect(t.action?.label).toBe("Повторить");
  });

  it("о задачах, которые закончились до нас, не уведомляем; о своих — один раз", async () => {
    await openFridge();
    useStore.setState({ toasts: [] });
    const eng = useStore.getState().engine as unknown as { emit: (j: unknown) => void };
    const done = { id: "old1", kind: "prepare_scene", project_id: "p-fridge", scene_id: "sc001", state: "done", steps: [], created: "" };
    // задача пришла уже законченной (мы её не видели идущей) — тишина
    (eng as never as { listeners: Set<(j: unknown) => void> }).listeners.forEach((f) => f(done));
    await wait(30);
    expect(useStore.getState().toasts).toHaveLength(0);
    const run = { ...done, id: "new1", state: "running" };
    const fire = (j: unknown) => (eng as never as { listeners: Set<(j: unknown) => void> }).listeners.forEach((f) => f(j));
    fire(run); fire({ ...run, state: "done" }); fire({ ...run, state: "done" });
    await wait(60);
    expect(useStore.getState().toasts.filter((t) => /Голоса и фон готовы/.test(t.text))).toHaveLength(1);
  });

  it("«это разные» помним отдельно для каждого проекта", async () => {
    await openFridge();
    useStore.getState().dismissMerge("s1", "s2");
    expect(useStore.getState().ui.dismissed_merges).toEqual(["p-fridge|s1|s2"]);
  });
});
