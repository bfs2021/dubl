import { beforeEach, describe, expect, it, vi } from "vitest";
import { act, fireEvent, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { freshStore, openFridge, renderRoute } from "../test/helpers";
import { MERGE_TOAST_MS, useStore } from "../state/store";
import { RecordScreen } from "./Record";
import { menuPlace } from "../components/SpeakerMenu";
import { mergedLinesOf } from "../api/types";
import { LINES_OTHER_SPEAKER_TEXT } from "../api/mock";

beforeEach(async () => { await freshStore(); await openFridge(); });

const scene = () => useStore.getState().project!.scenes.find((s) => s.id === "sc001")!;
const item = (n: number) => document.querySelectorAll<HTMLElement>(".rl-item")[n - 1];
const menu = (n: number) => screen.getByRole("menu", { name: `Кто говорит реплику ${n}` });
const nameOf = (sid: string | null | undefined) => { const s = useStore.getState().project!.analysis!.speakers.find((x) => x.id === sid); return s?.name ?? s?.label ?? ""; };

/** Сделать вторую реплику репликой того же персонажа, что первая (чтобы их можно было склеить). */
async function sameSpeakerPair() {
  const [a, b] = scene().lines;
  await act(async () => { await useStore.getState().setLineSpeaker(b.line_id, { speaker_id: a.speaker_id! }); });
  return scene().lines.slice(0, 2);
}

describe("меню «Кто это говорит?»", () => {
  it("рисуется в body, поверх всего (вне карточки реплики), с ролями menu/menuitemradio", async () => {
    renderRoute(<RecordScreen />);
    await userEvent.click(await screen.findByRole("button", { name: /Персонаж реплики 1/ }));
    const m = menu(1);
    expect(m.parentElement).toBe(document.body);
    expect(m.closest(".rl-item")).toBeNull();
    expect(item(1).contains(m)).toBe(false);
    expect(m.style.position).toBe("fixed");
    const items = within(m).getAllByRole("menuitemradio");
    expect(items.length).toBe(useStore.getState().project!.analysis!.speakers.length);
    const cur = items.find((x) => x.getAttribute("aria-checked") === "true")!;
    expect(cur).toHaveTextContent(nameOf(scene().lines[0].speaker_id));
    expect(screen.getByRole("button", { name: /Персонаж реплики 1/ })).toHaveAttribute("aria-expanded", "true");
  });

  it("клик по пункту меняет персонажа — вместе с дублями; меню закрывается", async () => {
    renderRoute(<RecordScreen />);
    const first = scene().lines[0];
    const other = useStore.getState().project!.analysis!.speakers.find((s) => s.id !== first.speaker_id)!;
    await userEvent.click(await screen.findByRole("button", { name: /Персонаж реплики 1/ }));
    await userEvent.click(within(menu(1)).getByRole("menuitemradio", { name: new RegExp(other.name ?? other.label) }));
    await waitFor(() => expect(scene().lines[0].speaker_id).toBe(other.id));
    expect(scene().takes.filter((t) => t.line_id === first.line_id).every((t) => t.speaker_id === other.id)).toBe(true);
    expect(screen.queryByRole("menu")).toBeNull();
  });

  it("клавиатура: фокус на текущем, ↓/↑ по пунктам, Enter выбирает; ↑↓ не листают реплики", async () => {
    renderRoute(<RecordScreen />);
    const first = scene().lines[0];
    await userEvent.click(await screen.findByRole("button", { name: /Персонаж реплики 1/ }));
    const items = within(menu(1)).getAllByRole("menuitemradio");
    const all = [...items, within(menu(1)).getByRole("menuitem")]; // последним идёт «+ Новый персонаж…»
    const cur = items.findIndex((x) => x.getAttribute("aria-checked") === "true");
    await waitFor(() => expect(document.activeElement).toBe(items[cur]));
    await userEvent.keyboard("{ArrowDown}");
    expect(document.activeElement).toBe(all[(cur + 1) % all.length]);
    await userEvent.keyboard("{ArrowUp}");
    expect(document.activeElement).toBe(items[cur]);
    // к соседнему персонажу (не текущему) и Enter
    const down = cur + 1 < items.length;
    await userEvent.keyboard(down ? "{ArrowDown}" : "{ArrowUp}");
    const target = items[down ? cur + 1 : cur - 1];
    expect(document.activeElement).toBe(target);
    await userEvent.keyboard("{Enter}");
    await waitFor(() => expect(scene().lines[0].speaker_id).not.toBe(first.speaker_id));
    expect(screen.queryByRole("menu")).toBeNull();
    // выбранная реплика осталась первой: стрелки меню не долетели до списка реплик
    expect(item(1)).toHaveClass("sel");
  });

  it("Esc и клик мимо закрывают меню, ничего не меняя", async () => {
    renderRoute(<RecordScreen />);
    const before = scene().lines[0].speaker_id;
    const btn = await screen.findByRole("button", { name: /Персонаж реплики 1/ });
    await userEvent.click(btn);
    await userEvent.keyboard("{Escape}");
    expect(screen.queryByRole("menu")).toBeNull();
    expect(document.activeElement).toBe(btn);
    await userEvent.click(btn);
    expect(menu(1)).toBeInTheDocument();
    fireEvent.mouseDown(document.querySelector(".scenebar")!);
    expect(screen.queryByRole("menu")).toBeNull();
    await userEvent.click(btn);
    // прокрутка списка сдвинула кнопку — меню закрывается (в jsdom координаты подставляем сами)
    vi.spyOn(btn, "getBoundingClientRect").mockReturnValue({ top: -40, bottom: -20, left: 0, right: 50, width: 50, height: 20, x: 0, y: -40, toJSON: () => ({}) } as DOMRect);
    fireEvent.scroll(document.querySelector(".record .panel .pb")!);
    expect(screen.queryByRole("menu")).toBeNull();
    expect(scene().lines[0].speaker_id).toBe(before);
  });

  it("новый персонаж заводится прямо из меню", async () => {
    renderRoute(<RecordScreen />);
    await userEvent.click(await screen.findByRole("button", { name: /Персонаж реплики 2/ }));
    await userEvent.click(within(menu(2)).getByRole("menuitem", { name: "+ Новый персонаж…" }));
    await userEvent.type(screen.getByRole("textbox", { name: "Имя нового персонажа" }), "Прислужник");
    await userEvent.click(screen.getByRole("button", { name: "Добавить" }));
    await waitFor(() => {
      const sp = useStore.getState().project!.analysis!.speakers.find((s) => s.name === "Прислужник");
      expect(sp).toBeTruthy();
      expect(scene().lines[1].speaker_id).toBe(sp!.id);
    });
  });

  it("у нижнего края окна меню открывается вверх и не вылезает за края", () => {
    const view = { w: 1200, h: 800 };
    const size = { w: 220, h: 180 };
    expect(menuPlace({ top: 100, bottom: 120, left: 40 }, size, view)).toEqual({ top: 124, left: 40, up: false });
    const low = menuPlace({ top: 740, bottom: 760, left: 40 }, size, view);
    expect(low.up).toBe(true);
    expect(low.top + size.h).toBeLessThanOrEqual(740);
    expect(menuPlace({ top: 100, bottom: 120, left: 1150 }, size, view).left).toBe(1200 - 8 - 220);
  });
});

describe("склейка реплик", () => {
  it("соседи разных персонажей: ссылка неактивна, рядом — пояснение с именем; запроса нет", async () => {
    const spy = vi.spyOn(useStore.getState().engine!, "mergeLines");
    renderRoute(<RecordScreen />);
    await screen.findByRole("button", { name: /Персонаж реплики 1/ });
    const [a, b] = scene().lines;
    expect(a.speaker_id).not.toBe(b.speaker_id);
    await userEvent.click(item(1));
    const link = within(item(1)).getByRole("button", { name: "Склеить с репликой 2" });
    expect(link).toBeDisabled();
    const why = within(item(1)).getByTestId("merge-other");
    expect(why).toHaveTextContent(`Реплика 2 — у другого персонажа (${nameOf(b.speaker_id)}). Если это одна фраза, сначала смени персонажа у неё.`);
    await userEvent.click(link);
    expect(spy).not.toHaveBeenCalled();
  });

  it("движок всё же ответил 400 про разных персонажей — обычная плашка ошибки", async () => {
    const [a, b] = scene().lines;
    await act(async () => { await useStore.getState().mergeLines([a.line_id, b.line_id]); });
    const t = useStore.getState().toasts.find((x) => x.kind === "error");
    expect(t?.text).toBe(LINES_OTHER_SPEAKER_TEXT);
    expect(scene().lines[0].line_id).toBe(a.line_id);
  });

  it("один персонаж: склейка, плашка 12 с с «Отменить» и подсказкой про ⌘Z", async () => {
    renderRoute(<RecordScreen />);
    await screen.findByRole("button", { name: /Персонаж реплики 1/ });
    const [a, b] = await sameSpeakerPair();
    const before = scene().lines.map((l) => l.line_id);
    await userEvent.click(item(1));
    await userEvent.click(within(item(1)).getByRole("button", { name: "Склеить с репликой 2" }));
    await waitFor(() => expect(scene().lines.length).toBe(before.length - 1));
    const merged = scene().lines[0];
    expect(merged.line_id).toBe(a.line_id);
    expect(merged.end_rel).toBe(b.end_rel);
    expect(MERGE_TOAST_MS).toBe(12000);
    const toast = useStore.getState().toasts.find((t) => t.text.startsWith("Реплики склеены"));
    expect(toast?.text).toMatch(/(⌘Z|Ctrl\+Z) — отменить/);
    expect(toast?.action?.label).toBe("Отменить");
    toast!.action!.run();
    await waitFor(() => expect(scene().lines.map((l) => l.line_id)).toEqual(before));
  });

  it("у последней реплики склеивать не с чем", async () => {
    renderRoute(<RecordScreen />);
    await screen.findByRole("button", { name: /Персонаж реплики 1/ });
    const n = scene().lines.length;
    await userEvent.click(item(n));
    expect(within(item(n)).queryByRole("button", { name: /Склеить с репликой/ })).toBeNull();
  });
});

describe("отмена правок реплик", () => {
  it("«Разделить обратно» у склеенной реплики возвращает обе реплики", async () => {
    renderRoute(<RecordScreen />);
    await screen.findByRole("button", { name: /Персонаж реплики 1/ });
    const [a] = await sameSpeakerPair();
    const before = scene().lines.map((l) => l.line_id);
    await act(async () => { await useStore.getState().mergeLines(before.slice(0, 2)); });
    expect(mergedLinesOf(useStore.getState().project!.analysis)[a.line_id]).toBeTruthy();
    await userEvent.click(item(1));
    await userEvent.click(within(item(1)).getByRole("button", { name: /Разделить обратно/ }));
    await waitFor(() => expect(scene().lines.map((l) => l.line_id)).toEqual(before));
    expect(mergedLinesOf(useStore.getState().project!.analysis)[a.line_id]).toBeUndefined();
    // несклеенной реплике разделять нечего
    expect(within(item(1)).queryByRole("button", { name: /Разделить обратно/ })).toBeNull();
    // и ⌘Z эту склейку второй раз не отменяет
    expect(useStore.getState().lineUndo["p-fridge"].some((e) => e.kind === "merge")).toBe(false);
  });

  it("⌘Z отменяет склейку, потом смену персонажа — по очереди", async () => {
    renderRoute(<RecordScreen />);
    await screen.findByRole("button", { name: /Персонаж реплики 1/ });
    const orig = scene().lines.map((l) => [l.line_id, l.speaker_id]);
    await sameSpeakerPair();
    await act(async () => { await useStore.getState().mergeLines(scene().lines.slice(0, 2).map((l) => l.line_id)); });
    expect(scene().lines.length).toBe(orig.length - 1);
    fireEvent.keyDown(document.body, { code: "KeyZ", key: "z", metaKey: true });
    await waitFor(() => expect(scene().lines.length).toBe(orig.length));
    expect(useStore.getState().toasts.some((t) => t.text === "Отменено: склейка реплик.")).toBe(true);
    fireEvent.keyDown(document.body, { code: "KeyZ", key: "z", ctrlKey: true });
    await waitFor(() => expect(scene().lines.map((l) => [l.line_id, l.speaker_id])).toEqual(orig));
    expect(useStore.getState().toasts.some((t) => t.text === "Отменено: смена персонажа реплики.")).toBe(true);
    fireEvent.keyDown(document.body, { code: "KeyZ", key: "z", metaKey: true });
    await waitFor(() => expect(useStore.getState().toasts.some((t) => /Отменять нечего/.test(t.text))).toBe(true));
  });

  it("⌘Z в текстовом поле — обычная отмена ввода, правки реплик не трогает", async () => {
    renderRoute(<RecordScreen />);
    await screen.findByRole("button", { name: /Персонаж реплики 1/ });
    const [, b] = await sameSpeakerPair();
    await act(async () => { await useStore.getState().setTextMode("custom"); });
    const box = screen.getByRole("textbox", { name: "Текст реплики 1" });
    fireEvent.keyDown(box, { code: "KeyZ", key: "z", metaKey: true });
    await new Promise((r) => setTimeout(r, 50));
    expect(scene().lines[1].speaker_id).toBe(b.speaker_id);
    expect(useStore.getState().lineUndo["p-fridge"]).toHaveLength(1);
  });

  it("стек отмены — не больше 20 правок на проект", async () => {
    const st = useStore.getState();
    const [a, b] = scene().lines;
    for (let i = 0; i < 23; i++) {
      await act(async () => { await useStore.getState().setLineSpeaker(a.line_id, { speaker_id: (i % 2 ? a.speaker_id : b.speaker_id)! }); });
    }
    expect(st.lineUndo).toBeDefined();
    expect(useStore.getState().lineUndo["p-fridge"].length).toBe(20);
  });
});

describe("варианты ИИ «Выбери вариант — подставлю в текст»", () => {
  it("не появляются сами: только после «Подогнать под длину» и только у выбранной реплики", async () => {
    renderRoute(<RecordScreen />);
    await screen.findByRole("button", { name: /Персонаж реплики 1/ });
    expect(screen.queryByTestId("fit-variants")).toBeNull();
    await userEvent.click(item(2));
    expect(screen.queryByTestId("fit-variants")).toBeNull();
    await userEvent.click(within(item(2)).getByRole("button", { name: /Подогнать под длину/ }));
    const v = await screen.findByTestId("fit-variants", {}, { timeout: 4000 });
    expect(item(2).contains(v)).toBe(true);
    // ушли на другую реплику — варианты прячутся и сами больше не всплывают
    await userEvent.click(item(1));
    expect(screen.queryByTestId("fit-variants")).toBeNull();
    await userEvent.click(item(2));
    expect(screen.queryByTestId("fit-variants")).toBeNull();
  });
});
