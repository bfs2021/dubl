import { beforeEach, describe, expect, it, vi } from "vitest";
import { act, fireEvent, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { freshStore, openFridge, renderRoute } from "../test/helpers";
import { useStore } from "../state/store";
import { RenderScreen } from "./Render";
import { MixScreen } from "./Mix";
import { AnalyzeScreen } from "./Analyze";
import { ProjectsScreen, projectFacts } from "./Projects";
import { ConfirmHost } from "../components/ConfirmDialog";
import { noteGroups } from "../components/DirectorPanel";
import { dirOf, joinPath, defaultOutputPath } from "../lib/files";
import { fmtDb, fmtNum, fmtSec, fmtBytes, fmtDate } from "../lib/format";
import { firstConversation, talkGroups } from "../lib/conversation";
import { takeRemark } from "../lib/director";
import { MockEngine } from "../api/mock";
import { mergedLinesOf, type Line, type Take } from "../api/types";

beforeEach(async () => { await freshStore(); });

describe("числа по-русски", () => {
  it("fmtSec/fmtNum/fmtDb/fmtBytes — с запятой", () => {
    expect(fmtSec(22.19)).toBe("22,2 с");
    expect(fmtSec(0.5)).toBe("0,5 с");
    expect(fmtNum(4.56, 2)).toBe("4,56");
    expect(fmtDb(-7.8)).toBe("−7,8 дБ");
    expect(fmtBytes(30.3e6)).toBe("30,3 МБ");
  });
  it("дата создания проекта по-русски", () => {
    const now = new Date(2026, 9, 24, 18, 0);
    expect(fmtDate(new Date(2026, 9, 24, 14, 5).toISOString(), now)).toBe("24 октября в 14:05");
    expect(fmtDate(new Date(2025, 0, 3, 9, 30).toISOString(), now)).toBe("3 января 2025 в 09:30");
  });
  it("на «Записи» длина сцены и реплик — с запятой, таймкоды — с точкой", async () => {
    await openFridge();
    const { RecordScreen } = await import("./Record");
    renderRoute(<RecordScreen />);
    await screen.findByRole("button", { name: /Персонаж реплики 1/ });
    const bar = document.querySelector(".scenebar .mono")!.textContent!;
    expect(bar).toMatch(/\d+,\d с/);
    expect(bar).toMatch(/0:00\.0/);
    const lens = [...document.querySelectorAll(".rl-item .hd .mono.mute")].map((x) => x.textContent);
    expect(lens.length).toBeGreaterThan(0);
    for (const t of lens) expect(t).toMatch(/^\d+,\d с$/);
  });
});

describe("путь сохранения ролика", () => {
  it("dirOf/joinPath понимают / и \\ (Windows)", () => {
    expect(dirOf("/tmp/dubl/out/ролик.mp4")).toBe("/tmp/dubl/out");
    expect(dirOf("C:\\Users\\me\\Videos\\ролик.mp4")).toBe("C:\\Users\\me\\Videos");
    expect(dirOf("C:\\ролик.mp4")).toBe("C:\\");
    expect(dirOf("/ролик.mp4")).toBe("/");
    expect(dirOf("ролик.mp4")).toBe("");
    expect(joinPath("C:\\Users\\me\\Videos", "a.mp4")).toBe("C:\\Users\\me\\Videos\\a.mp4");
    expect(joinPath("/tmp/out", "a.mp4")).toBe("/tmp/out/a.mp4");
    expect(joinPath("/tmp/out/", "a.mp4")).toBe("/tmp/out/a.mp4");
  });
  it("по умолчанию — папка из прошлого раза + имя сцены", async () => {
    expect(await defaultOutputPath("Сцена: 1", "/tmp/out")).toBe("/tmp/out/Сцена 1.mp4");
    expect(await defaultOutputPath("Сцена", "D:\\Видео")).toBe("D:\\Видео\\Сцена.mp4");
    expect(await defaultOutputPath("Сцена", "")).toBe("~/Movies/Дубль/Сцена.mp4");
  });
  it("вписанный руками путь запоминается: при следующем входе на «Рендер» — та же папка", async () => {
    await openFridge();
    const first = renderRoute(<RenderScreen />);
    await screen.findByText(/Что собираем/);
    const path = screen.getByLabelText("Куда сохранить") as HTMLTextAreaElement;
    await waitFor(() => expect(path.value).toMatch(/\.mp4$/));
    fireEvent.change(path, { target: { value: "/tmp/dubl-blindtest7/out/мой ролик" } });
    fireEvent.blur(path);
    expect(path.value).toBe("/tmp/dubl-blindtest7/out/мой ролик.mp4");
    expect(useStore.getState().ui.last_output_dir).toBe("/tmp/dubl-blindtest7/out");
    expect(JSON.parse(localStorage.getItem("dubl.ui")!).last_output_dir).toBe("/tmp/dubl-blindtest7/out");
    first.unmount();
    renderRoute(<RenderScreen />);
    await screen.findByText(/Что собираем/);
    const again = screen.getByLabelText("Куда сохранить") as HTMLTextAreaElement;
    const title = useStore.getState().project!.scenes[0].title;
    await waitFor(() => expect(again.value).toBe(`/tmp/dubl-blindtest7/out/${title}.mp4`));
  });
  it("ручная правка не затирается сменой сцен в рамках экрана; «Собрать ролик» запоминает папку", async () => {
    await openFridge();
    const engine = useStore.getState().engine!;
    const spy = vi.spyOn(engine, "render");
    renderRoute(<RenderScreen />);
    await screen.findByText(/Что собираем/);
    const path = screen.getByLabelText("Куда сохранить") as HTMLTextAreaElement;
    await waitFor(() => expect(path.value).toMatch(/\.mp4$/));
    fireEvent.change(path, { target: { value: "C:\\Users\\me\\Videos\\итог.mp4" } });
    // снимаем и ставим сцену — путь остаётся тем, что вписали
    const box = screen.getByRole("checkbox", { name: useStore.getState().project!.scenes[0].title });
    await userEvent.click(box);
    await userEvent.click(box);
    expect(path.value).toBe("C:\\Users\\me\\Videos\\итог.mp4");
    await userEvent.click(screen.getByRole("button", { name: /Собрать ролик/ }));
    await waitFor(() => expect(spy).toHaveBeenCalled());
    expect(spy.mock.calls[0][1].settings.output_path).toBe("C:\\Users\\me\\Videos\\итог.mp4");
    expect(useStore.getState().ui.last_output_dir).toBe("C:\\Users\\me\\Videos");
  });
});

const L = (id: string, start: number, end: number, spk: string): Line => ({ id, start, end, text: id, speaker_id: spk, words: [], source: "asr", overlap: false } as unknown as Line);

describe("«Выделить первый разговор»", () => {
  // как в слепом тесте 7: одиночная фраза 0–2,9, 12 с музыки, потом диалог 13,5–35,7
  const lines = [
    L("a", 0, 2.9, "s2"),
    L("b", 14.6, 17.9, "s1"), L("c", 19.0, 21.4, "s2"), L("d", 21.8, 26.5, "s1"), L("e", 26.9, 28.4, "s2"), L("f", 33.6, 35.7, "s1"),
    L("g", 53.2, 54.8, "s3"), L("h", 55.2, 57.0, "s1"),
  ];
  it("пропускает одиночную первую фразу и берёт диалог целиком", () => {
    const t = firstConversation(lines)!;
    expect(t.start).toBe(14.6);
    expect(t.end).toBe(35.7);
    expect(t.lines.map((l) => l.id)).toEqual(["b", "c", "d", "e", "f"]);
    expect(t.speakers).toBe(2);
  });
  it("паузы < 4 с держат разговор вместе, ответ другого персонажа — и после паузы до 6 с; 12 с музыки — разрыв", () => {
    expect(talkGroups(lines).map((g) => g.lines.length)).toEqual([1, 5, 2]);
    // тот же персонаж после 5 с — уже новый кусок
    expect(talkGroups([L("x", 0, 1, "s1"), L("y", 6, 7, "s1")]).length).toBe(2);
  });
  it("нет группы из 3+ реплик — самая длинная, где говорят двое; нет и такой — первая", () => {
    expect(firstConversation([L("a", 0, 1, "s1"), L("b", 10, 11, "s1"), L("c", 11.5, 12, "s2")])!.lines.map((l) => l.id)).toEqual(["b", "c"]);
    expect(firstConversation([L("a", 0, 1, "s1"), L("b", 10, 11, "s1")])!.lines.map((l) => l.id)).toEqual(["a"]);
    expect(firstConversation([])).toBeNull();
  });
  it("на экране «Разбор» кнопка выделяет разговор, а не первую реплику", async () => {
    await useStore.getState().openProject("p-fridge");
    const st = useStore.getState();
    useStore.setState({ project: { ...st.project!, scenes: [], analysis: { ...st.project!.analysis!, lines } } });
    renderRoute(<AnalyzeScreen />);
    await userEvent.click(await screen.findByRole("button", { name: /Дальше: выбери момент/ }));
    const info = await screen.findByText(/реплик/, { selector: ".selinfo .mute" });
    expect(info).toHaveTextContent("5 реплик");
    expect(info).toHaveTextContent("2 голоса");
    expect(document.querySelector(".selinfo .mono")!.textContent).toMatch(/^0:14\.\d – 0:35\.\d$/);
  });
});

describe("Сведение: строка помощника и неозвученные персонажи", () => {
  const take = (id: string, line: string, spk: string, extra: Partial<Take> = {}) => ({ id, line_id: line, speaker_id: spk, selected: true, source: "file", ...extra } as unknown as Take);
  it("одинаковые замечания одного персонажа — одной строкой с номерами реплик", () => {
    const scene = { lines: [{ line_id: "l1" }, { line_id: "l2" }, { line_id: "l3" }, { line_id: "l4" }], mix: { mode: "dub" } } as never;
    const ok = { text: "Отличный дубль: попал в губы, текст и громкость на месте.", severity: "ok" as const };
    const g = noteGroups([take("t1", "l2", "s2"), take("t2", "l4", "s2")], scene, () => ok);
    expect(g).toHaveLength(1);
    expect(g[0].where).toBe("реплики 2 и 4");
    // дубль из файла: «попал в губы» не показываем
    expect(g[0].remark!.text).not.toMatch(/губ.*попал|попал.*губ/i);
    const two = noteGroups([take("t1", "l2", "s2", { source: "mic" }), take("t2", "l4", "s2", { source: "mic" })], scene, (t) => (t.id === "t1" ? ok : { text: "Тихо.", severity: "warn" }));
    expect(two).toHaveLength(2);
  });
  it("демо-помощник: для дубля из файла не пишет «попал в губы»", () => {
    const t = { offset_ms: 0, stretch: 1, peak_db: -6, duration: 2, fit_confidence: 0.9 };
    expect(takeRemark(t, 2).text).toMatch(/попал в движения губ/);
    expect(takeRemark(t, 2, true).text).not.toMatch(/попал/);
  });
  it("на экране — одна строка на персонажа, неозвучиваемый персонаж — строкой «<Имя> — оригинал»", async () => {
    await openFridge();
    const st = useStore.getState();
    const sc = st.project!.scenes[0];
    const mila = sc.takes.find((t) => t.selected)!;
    // две реплики одного персонажа с одинаковым замечанием; озвучиваем только его
    const other = sc.lines.find((l) => l.speaker_id === mila.speaker_id && l.line_id !== mila.line_id)!;
    const takes = [{ ...mila, id: "tA", source: "file" }, { ...mila, id: "tB", line_id: other.line_id, source: "file" }] as Take[];
    useStore.setState({
      project: { ...st.project!, scenes: [{ ...sc, takes, dubbed_speakers: [mila.speaker_id!] }] },
      remarks: { [`p-fridge/${sc.id}/tA`]: { text: "Отличный дубль.", severity: "ok" }, [`p-fridge/${sc.id}/tB`]: { text: "Отличный дубль.", severity: "ok" } },
    });
    renderRoute(<MixScreen />);
    await screen.findByRole("region", { name: "Помощник-режиссёр" });
    const notes = screen.getAllByTestId("director-note");
    expect(notes).toHaveLength(1);
    expect(notes[0]).toHaveTextContent(/реплики \d+ и \d+/);
    const others = [...new Set(sc.lines.map((l) => l.speaker_id))].filter((x) => x && x !== mila.speaker_id) as string[];
    for (const sid of others) {
      const row = screen.getByTestId(`track-${sid}`);
      const sp = useStore.getState().project!.analysis!.speakers.find((s) => s.id === sid)!;
      expect(row).toHaveTextContent(`${sp.name ?? sp.label} — оригинал`);
      // в дубляже громкость оригинального голоса движок берёт из mix.tracks — ползунок есть
      expect(within(row).getByRole("slider")).toBeInTheDocument();
    }
  });
});

describe("Проекты: тёзки и ссылка", () => {
  it("проект по ссылке сразу называется как ролик (название из проверки ссылки)", async () => {
    const engine = useStore.getState().engine!;
    vi.spyOn(engine, "probe").mockResolvedValueOnce({ known: true, title: "Три раунда боя Титана против Мегамозга 4K", duration: 601, has_subtitles: true, subtitle_langs: ["ru"] } as never);
    const spy = vi.spyOn(engine, "createProject");
    renderRoute(<ProjectsScreen />, "/", "/");
    await screen.findByText(/Перетащи сюда/);
    await userEvent.type(screen.getByLabelText("Ссылка или путь"), "https://youtu.be/xyz{enter}");
    await userEvent.click(await screen.findByRole("button", { name: /Начать разбор/ }));
    await waitFor(() => expect(spy).toHaveBeenCalled());
    expect(spy.mock.calls[0][0].title).toBe("Три раунда боя Титана против Мегамозга 4K");
  });
  it("на карточке — длина видео и дата создания; в окне удаления — длина, «создан …», сцены", async () => {
    const list = await useStore.getState().engine!.listProjects();
    const fridge = list.find((p) => p.id === "p-fridge")!;
    expect(projectFacts(fridge)).toMatch(/^Видео 22:41 · создан \d+ \S+( \d{4})? в \d\d:\d\d · 1 сцена$/);
    renderRoute(<><ProjectsScreen /><ConfirmHost /></>, "/", "/");
    const card = (await screen.findAllByTestId("project-card")).find((c) => c.textContent!.includes(fridge.title))!;
    expect(within(card).getByTestId("project-created")).toHaveTextContent(/^создан /);
    await userEvent.click(within(card).getByRole("button", { name: `Удалить проект ${fridge.title}` }));
    const dlg = await screen.findByRole("alertdialog");
    expect(dlg).toHaveTextContent("Видео 22:41");
    expect(dlg).toHaveTextContent(/создан /);
    expect(dlg).toHaveTextContent("1 сцена");
  });
});

describe("мок-движок: склейка, «Разделить обратно», merged_lines", () => {
  it("ведёт merged_lines, split возвращает как было, разных персонажей не склеивает", async () => {
    const e = new MockEngine({ speed: 60 });
    const p = await e.getProject("p-fridge");
    const [a, b] = p.scenes[0].lines;
    await expect(e.mergeLines(p.id, [a.line_id, b.line_id])).rejects.toThrow(/разных персонажей/);
    await e.setLineSpeaker(p.id, b.line_id, { speaker_id: a.speaker_id! });
    const r = await e.mergeLines(p.id, [a.line_id, b.line_id]);
    expect(mergedLinesOf(r.project.analysis)[a.line_id]).toBe(r.merge_id);
    await expect(e.splitLine(p.id, b.line_id)).rejects.toThrow(/не склеивалась/);
    const back = await e.splitLine(p.id, a.line_id);
    expect(back.scenes[0].lines.map((l) => l.line_id).slice(0, 2)).toEqual([a.line_id, b.line_id]);
    expect(mergedLinesOf(back.analysis)[a.line_id]).toBeUndefined();
    await act(async () => {});
  });
});
