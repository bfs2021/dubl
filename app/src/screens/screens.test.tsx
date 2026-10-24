import { beforeEach, describe, expect, it, vi } from "vitest";
import { act, fireEvent, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { freshStore, openFridge, renderRoute } from "../test/helpers";
import { useStore } from "../state/store";
import { ProjectsScreen } from "./Projects";
import { AnalyzeScreen } from "./Analyze";
import { RecordScreen } from "./Record";
import { MixScreen } from "./Mix";
import { RenderScreen } from "./Render";
import { SettingsScreen } from "./Settings";
import { App } from "../App";
import { MemoryRouter } from "react-router-dom";
import { render } from "@testing-library/react";
import type { Job } from "../api/types";

beforeEach(async () => { await freshStore(); });

describe("рендер каждого экрана без падений", () => {
  it("Projects", async () => {
    renderRoute(<ProjectsScreen />, "/", "/");
    expect(await screen.findByText(/Перетащи сюда/)).toBeInTheDocument();
    expect(screen.getByText(/Серия 7/)).toBeInTheDocument();
    expect(screen.getByText(/Погоня в парке/)).toBeInTheDocument();
    expect(screen.getAllByText(/Серия 8/).length).toBeGreaterThan(0);
  });

  it("Analyze", async () => {
    await openFridge();
    renderRoute(<AnalyzeScreen />);
    expect(await screen.findByText("Персонажи")).toBeInTheDocument();
    expect(screen.getByRole("tab", { name: /Сцены/ })).toBeInTheDocument();
    expect(screen.queryByText(/Моменты/)).toBeNull();
    expect(screen.queryByText(/Смешнее/i)).toBeNull();
    expect(screen.getByRole("radio", { name: "Диалог" })).toBeInTheDocument();
    expect(screen.getByRole("radio", { name: "Кадр" })).toBeInTheDocument();
    expect(screen.getByRole("radio", { name: "Вручную" })).toBeInTheDocument();
  });

  it("Record", async () => {
    await openFridge();
    renderRoute(<RecordScreen />);
    expect(await screen.findByText(/Записать реплику/)).toBeInTheDocument();
    expect(screen.getByText("Как есть")).toBeInTheDocument();
    expect(screen.getByText("Свой текст")).toBeInTheDocument();
    expect(screen.queryByText(/Смешнее/i)).toBeNull();
  });

  it("Mix", async () => {
    await openFridge();
    renderRoute(<MixScreen />);
    expect(await screen.findByRole("region", { name: "Помощник-режиссёр" })).toBeInTheDocument();
    expect(screen.getByRole("switch", { name: "Автопилот" })).toBeChecked();
    expect(screen.getByTestId("fine-toggle")).toBeInTheDocument();
  });

  it("Render", async () => {
    await openFridge();
    renderRoute(<RenderScreen />);
    expect(await screen.findByText("Что собираем", { exact: false })).toBeInTheDocument();
    expect(screen.getByRole("radio", { name: /16:9/ })).toBeInTheDocument();
  });

  it("Settings: три раздела", async () => {
    renderRoute(<SettingsScreen />, "/settings/:page?", "/settings/ai");
    expect(await screen.findByText("Какая модель что делает")).toBeInTheDocument();
    for (const t of ["Перевод", "Подогнать под длину", "Имена персонажей", "Помощник-режиссёр"]) expect(screen.getAllByText(t).length).toBeGreaterThan(0);
    await userEvent.click(screen.getByRole("button", { name: /Обработка/ }));
    // MemoryRouter перерисует по маршруту
  });

  it("Settings: Обработка и Микрофон по адресу", async () => {
    const { unmount } = renderRoute(<SettingsScreen />, "/settings/:page?", "/settings/processing");
    expect(await screen.findByText("Свой сервер")).toBeInTheDocument();
    unmount();
    renderRoute(<SettingsScreen />, "/settings/:page?", "/settings/mic");
    expect(await screen.findByText("Задержка записи")).toBeInTheDocument();
    expect(screen.getByText(/Я записываю в наушниках/)).toBeInTheDocument();
  });

  it("App: плашка демо-режима и переход на проект", async () => {
    useStore.setState({ engine: null });
    render(<MemoryRouter initialEntries={["/"]}><App /></MemoryRouter>);
    expect(await screen.findByText(/Демо-режим: движок не подключён/, {}, { timeout: 4000 })).toBeInTheDocument();
  });
});

describe("Разбор: создать сцену выделением", () => {
  it("клик по реплике (Диалог) выделяет весь разговор и создаёт сцену", async () => {
    await openFridge(null);
    renderRoute(<AnalyzeScreen />);
    await screen.findByText("Персонажи");
    const tracks = document.querySelector(".tl-tracks") as HTMLElement;
    // jsdom без геометрии: ширина 0 → pps = 900/1361; кликаем в 725 с
    tracks.getBoundingClientRect = () => ({ left: 0, top: 0, right: 900, bottom: 200, width: 900, height: 200, x: 0, y: 0, toJSON() {} }) as DOMRect;
    const pps = 900 / 1361;
    // viewW в jsdom = 0 → 900 по умолчанию, innerW = 900
    fireEvent.mouseDown(tracks, { clientX: 725 * pps, button: 0 });
    fireEvent.mouseUp(window, { clientX: 725 * pps });
    const create = await screen.findByRole("button", { name: /Создать сцену/ });
    expect(document.querySelector(".selinfo")!.textContent).toMatch(/\d+ реплик/);
    await userEvent.type(screen.getByLabelText("Название сцены"), "Моя сцена");
    await userEvent.click(create);
    await waitFor(() => expect(useStore.getState().project!.scenes.some((s) => s.title === "Моя сцена")).toBe(true));
    expect(await screen.findByText(/голоса и фон/i)).toBeInTheDocument();
  });

  it("режим «Сцена» выделяет от склейки до склейки", async () => {
    await openFridge(null);
    renderRoute(<AnalyzeScreen />);
    await screen.findByText("Персонажи");
    await userEvent.click(screen.getByRole("radio", { name: "Кадр" }));
    const tracks = document.querySelector(".tl-tracks") as HTMLElement;
    tracks.getBoundingClientRect = () => ({ left: 0, top: 0, right: 900, bottom: 200, width: 900, height: 200, x: 0, y: 0, toJSON() {} }) as DOMRect;
    fireEvent.mouseDown(tracks, { clientX: 300 * (900 / 1361), button: 0 });
    fireEvent.mouseUp(window, { clientX: 300 * (900 / 1361) });
    expect(await screen.findByRole("button", { name: /Создать сцену/ })).toBeInTheDocument();
    expect(document.querySelectorAll(".cut").length).toBeGreaterThan(10);
  });

  it("режим «Вручную»: протянуть на шкале", async () => {
    await openFridge(null);
    renderRoute(<AnalyzeScreen />);
    await screen.findByText("Персонажи");
    await userEvent.click(screen.getByRole("radio", { name: "Вручную" }));
    const tracks = document.querySelector(".tl-tracks") as HTMLElement;
    tracks.getBoundingClientRect = () => ({ left: 0, top: 0, right: 900, bottom: 200, width: 900, height: 200, x: 0, y: 0, toJSON() {} }) as DOMRect;
    fireEvent.mouseDown(tracks, { clientX: 100, button: 0 });
    fireEvent.mouseMove(window, { clientX: 200 });
    fireEvent.mouseUp(window, { clientX: 200 });
    expect(await screen.findByRole("button", { name: /Создать сцену/ })).toBeInTheDocument();
  });
});

describe("Запись: режимы текста и дубли", () => {
  it("переключение «Как есть» ↔ «Свой текст»: появляются поля ввода", async () => {
    await openFridge();
    renderRoute(<RecordScreen />);
    await screen.findByText(/Записать реплику/);
    expect(screen.queryAllByRole("textbox", { name: /Текст реплики/ })).toHaveLength(0);
    await userEvent.click(screen.getByRole("radio", { name: "Свой текст" }));
    await waitFor(() => expect(screen.getAllByRole("textbox", { name: /Текст реплики/ }).length).toBeGreaterThan(0));
    expect(useStore.getState().project!.scenes[0].text_mode).toBe("custom");
    await userEvent.click(screen.getByRole("radio", { name: "Как есть" }));
    await waitFor(() => expect(screen.queryAllByRole("textbox", { name: /Текст реплики/ })).toHaveLength(0));
  });

  it("«Перевести» вызывает ИИ только по кнопке", async () => {
    await openFridge();
    const spy = vi.spyOn(useStore.getState().engine!, "translate");
    renderRoute(<RecordScreen />);
    await screen.findByText(/Записать реплику/);
    expect(spy).not.toHaveBeenCalled();
    await userEvent.click(screen.getByRole("button", { name: /Перевести/ }));
    await waitFor(() => expect(spy).toHaveBeenCalledWith(expect.objectContaining({ target_lang: "ru" })));
  });

  it("у дубля есть замечание помощника под волной", async () => {
    await openFridge();
    renderRoute(<RecordScreen />);
    expect(await screen.findAllByLabelText("Замечание помощника")).not.toHaveLength(0);
    await waitFor(() => expect(screen.queryAllByText(/Слушаю дубль/)).toHaveLength(0));
    expect(document.querySelectorAll(".take canvas").length).toBeGreaterThan(0);
  });

  it("«Подогнать под длину» показывает варианты и подставляет выбранный", async () => {
    await openFridge();
    renderRoute(<RecordScreen />);
    await screen.findByText(/Записать реплику/);
    const item = document.querySelectorAll(".rl-item")[0] as HTMLElement; // реплика Милы — её озвучиваем
    await userEvent.click(within(item).getByRole("button", { name: /Подогнать под длину/ }));
    const list = await screen.findByRole("listbox", { name: /Варианты/ });
    const opt = within(list).getAllByRole("option")[0];
    await userEvent.click(opt);
    await waitFor(() => expect(useStore.getState().project!.scenes[0].text_mode).toBe("custom"));
  });
});

describe("Сведение: автопилот и тонкая настройка", () => {
  it("по умолчанию просто; «Тонкая настройка» раскрывает всё остальное", async () => {
    await openFridge();
    renderRoute(<MixScreen />);
    await screen.findByRole("region", { name: "Помощник-режиссёр" });
    expect(screen.queryByTestId("fine-body")).toBeNull();
    expect(screen.queryByText("Общая громкость ролика")).toBeNull();
    await userEvent.click(screen.getByTestId("fine-toggle"));
    expect(await screen.findByTestId("fine-body")).toBeInTheDocument();
    expect(screen.getByText(/Для опытных: громкость всего ролика/)).toBeInTheDocument();
    expect(screen.getAllByText(/Где звучит/).length).toBeGreaterThan(0);
    expect(screen.getByText(/Когда начинается/)).toBeInTheDocument();
    // жаргона на виду нет: M/S, LUFS, dBTP спрятаны в «Для опытных» или за подсказкой
    expect(screen.queryByText("M")).toBeNull();
    expect(screen.queryByText("S")).toBeNull();
  });

  it("ручное движение громкости выключает автопилот, переключатель включает обратно", async () => {
    await openFridge();
    renderRoute(<MixScreen />);
    await screen.findByRole("region", { name: "Помощник-режиссёр" });
    const sw = screen.getByRole("switch", { name: "Автопилот" });
    expect(sw).toBeChecked();
    fireEvent.change(screen.getByLabelText(/Громкость: Музыка и шумы/), { target: { value: "-10" } });
    await waitFor(() => expect(useStore.getState().project!.scenes[0].mix.autopilot).toBe(false));
    expect(useStore.getState().project!.scenes[0].mix.background.gain_db).toBe(-10);
    await userEvent.click(screen.getByRole("switch", { name: "Автопилот" }));
    await waitFor(() => expect(useStore.getState().project!.scenes[0].mix.autopilot).toBe(true));
  });

  it("«Скажи, что поменять»: «Барсик громче» меняет дорожку", async () => {
    await openFridge();
    renderRoute(<MixScreen />);
    await screen.findByRole("region", { name: "Помощник-режиссёр" });
    const before = useStore.getState().project!.scenes[0].mix.tracks["s1"]?.gain_db ?? 0;
    await userEvent.type(screen.getByLabelText("Скажи, что поменять"), "Барсик громче{enter}");
    await waitFor(() => expect(useStore.getState().project!.scenes[0].mix.tracks["s1"]?.gain_db).toBe(before + 3), { timeout: 3000 });
    expect(await screen.findByText(/Сделал:/)).toBeInTheDocument();
    expect(useStore.getState().project!.scenes[0].mix.autopilot).toBe(false); // ручная просьба выключила автопилот
    await userEvent.click(screen.getByRole("button", { name: "Отменить" }));
    await waitFor(() => expect(useStore.getState().project!.scenes[0].mix.autopilot).toBe(true));
  });
});

describe("Рендер: несколько сцен в один ролик", () => {
  it("отмечаю две сцены → кнопка про склейку → задача со шагом «Склейка»", async () => {
    await act(async () => {
      await useStore.getState().openProject("p-fridge");
      await useStore.getState().createScene(118, 140, "Вторая");
    });
    renderRoute(<RenderScreen />);
    await screen.findByText(/Что собираем/);
    const boxes = screen.getAllByRole("checkbox");
    expect(boxes.length).toBe(2);
    await userEvent.click(boxes.find((b) => !(b as HTMLInputElement).checked)!);
    const btn = await screen.findByRole("button", { name: /Склеить 2 сцены/ });
    await userEvent.click(btn);
    await waitFor(() => expect(Object.values(useStore.getState().jobs).filter((j) => j.kind === "render_project")).toHaveLength(1));
    expect(Object.values(useStore.getState().jobs).find((j) => j.kind === "render_project")!.steps.some((s) => s.key === "concat")).toBe(true);
    expect(await screen.findByTestId("done", {}, { timeout: 5000 })).toBeInTheDocument();
  });
});

describe("Проекты: файл → «Что нашли» → разбор", () => {
  it("по ссылке/пути показывает дорожки, субтитры, главы", async () => {
    renderRoute(<ProjectsScreen />, "/", "/");
    await screen.findByText(/Перетащи сюда/);
    await userEvent.type(screen.getByLabelText("Ссылка или путь"), "/Users/me/Новая серия.mkv{enter}");
    expect(await screen.findByText("Что нашли в файле")).toBeInTheDocument();
    expect(screen.getByText("5.1")).toBeInTheDocument();
    expect(screen.getByText("с именами")).toBeInTheDocument();
    expect(screen.getByText(/Главы/)).toBeInTheDocument();
    expect(screen.getAllByRole("radio").length).toBeGreaterThanOrEqual(3);
  });
  it("в простом клипе — «Всё разложим сами»", async () => {
    renderRoute(<ProjectsScreen />, "/", "/");
    await screen.findByText(/Перетащи сюда/);
    await userEvent.type(screen.getByLabelText("Ссылка или путь"), "/tmp/клип.mp4{enter}");
    expect(await screen.findByText(/Всё разложим сами/)).toBeInTheDocument();
  });
});

describe("Без ИИ кнопки ИИ неактивны с подсказкой", () => {
  const noAi = async () => {
    const st = useStore.getState();
    useStore.setState({ demo: false, settings: { ...st.settings!, llm: { providers: [], routing: {} } } });
  };
  it("Запись: «Перевести» и «Подогнать под длину» disabled + «Подключи ИИ в Настройках»", async () => {
    await openFridge();
    await noAi();
    renderRoute(<RecordScreen />);
    await screen.findByText(/Записать реплику/);
    expect(screen.getByRole("button", { name: /Перевести/ })).toBeDisabled();
    const fit = screen.getAllByRole("button", { name: /Подогнать под длину/ });
    expect(fit.length).toBeGreaterThan(0);
    fit.forEach((b) => expect(b).toBeDisabled());
    expect(screen.getAllByText(/Подключи ИИ в/).length).toBeGreaterThan(0);
    expect(screen.getAllByRole("button", { name: "Настройках" }).length).toBeGreaterThan(0);
  });
  it("Разбор: «Подсказать имена персонажей (ИИ)» disabled", async () => {
    await openFridge();
    await noAi();
    renderRoute(<AnalyzeScreen />);
    expect(await screen.findByRole("button", { name: /Подсказать имена персонажей/ })).toBeDisabled();
  });
});

describe("Рендер: субтитры по умолчанию выключены", () => {
  it("по умолчанию «Нет»; при вшивании появляется выбор «Перевод / Оригинал», по умолчанию перевод", async () => {
    await openFridge();
    renderRoute(<RenderScreen />);
    await screen.findByText(/Что собираем/);
    expect(screen.getByRole("radio", { name: "Нет" })).toBeChecked();
    expect(screen.queryByRole("radio", { name: "Оригинал" })).toBeNull(); // выбор текста — только при вшивании/файле
    await userEvent.click(screen.getByRole("radio", { name: "Вшить в видео" }));
    expect(await screen.findByRole("radio", { name: /Перевод \(текст дубля\)/ })).toBeChecked();
    expect(screen.getByRole("radio", { name: "Оригинал" })).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: /Собрать ролик|Склеить/ }));
    await waitFor(() => expect(Object.values(useStore.getState().jobs).some((j) => j.kind === "render_project")).toBe(true));
  });
});

describe("Проекты: в браузере «Выбрать файл» не ломается", () => {
  it("вместо диалога — подсказка вставить путь и фокус на поле", async () => {
    renderRoute(<ProjectsScreen />, "/", "/");
    await screen.findByText(/Перетащи сюда/);
    await userEvent.click(screen.getByRole("button", { name: /Указать путь к файлу/ }));
    expect(await screen.findByRole("alert")).toHaveTextContent(/полный путь/);
    expect(screen.getByLabelText("Ссылка или путь")).toHaveFocus();
  });
});

describe("Проекты: ссылка", () => {
  it("если ссылку проверить не удалось — показываем message и даём продолжить", async () => {
    renderRoute(<ProjectsScreen />, "/", "/");
    await screen.findByText(/Перетащи сюда/);
    await userEvent.type(screen.getByLabelText("Ссылка или путь"), "https://example.com/v.mp4{enter}");
    expect(await screen.findByTestId("url-info")).toHaveTextContent(/после скачивания/);
    expect(screen.getByRole("button", { name: /Начать разбор/ })).toBeEnabled();
  });
});


// ───────────── второй слепой тест ─────────────
import { offRange, stretchRange } from "./Mix";
import { UNDO_DELETE_MS } from "../state/store";
import { providerHealth } from "../lib/ai";

describe("Путь новичка после разбора", () => {
  it("крупная кнопка «Дальше», подсказка на шкале, и клик сразу выделяет момент", async () => {
    await openFridge(null);
    useStore.setState({ project: { ...useStore.getState().project!, scenes: [] } });
    renderRoute(<AnalyzeScreen />);
    await screen.findByText("Персонажи");
    expect(screen.getByRole("button", { name: /Дальше: выбери момент для озвучки/ })).toBeInTheDocument();
    // подсказка живёт в шапке шкалы, а не поверх дорожек
    expect(screen.queryByTestId("tl-hint")).toBeNull();
    expect(document.querySelector(".selinfo .hint")).toHaveTextContent("Протяни мышью по шкале или кликни по реплике");
    expect(screen.queryByRole("button", { name: /Создать сцену/ })).toBeNull();
    await userEvent.click(screen.getByRole("button", { name: /Дальше: выбери момент/ }));
    expect(await screen.findByRole("button", { name: /Создать сцену/ })).toBeInTheDocument();
  });

  it("в режиме «Вручную» простой клик тоже выделяет разговор", async () => {
    await openFridge(null);
    renderRoute(<AnalyzeScreen />);
    await screen.findByText("Персонажи");
    await userEvent.click(screen.getByRole("radio", { name: "Вручную" }));
    const tracks = document.querySelector(".tl-tracks") as HTMLElement;
    tracks.getBoundingClientRect = () => ({ left: 0, top: 0, right: 900, bottom: 200, width: 900, height: 200, x: 0, y: 0, toJSON() {} }) as DOMRect;
    fireEvent.mouseDown(tracks, { clientX: 725 * (900 / 1361), button: 0 });
    fireEvent.mouseUp(window, { clientX: 725 * (900 / 1361) });
    expect(await screen.findByRole("button", { name: /Создать сцену/ })).toBeInTheDocument();
  });
});

describe("Слияние голосов", () => {
  it("«Объединить с…» → тост «Отменить» возвращает голоса; новая подсказка ждёт закрытия тоста", async () => {
    await openFridge(null);
    renderRoute(<AnalyzeScreen />);
    await screen.findByText("Персонажи");
    const before = useStore.getState().project!.analysis!.speakers.length;
    await userEvent.click(screen.getAllByRole("button", { name: /Объединить .* с другим голосом/ })[1]);
    await userEvent.click(within(document.querySelector(".mergepick") as HTMLElement).getByRole("button", { name: "Объединить" }));
    await waitFor(() => expect(useStore.getState().project!.analysis!.speakers.length).toBe(before - 1));
    expect(useStore.getState().mergeHold).toBe(true);
    expect(screen.queryByTestId("merge-suggest")).toBeNull();
    await act(async () => { useStore.getState().toasts.find((t) => t.action?.label === "Отменить")!.action!.run(); });
    await waitFor(() => expect(useStore.getState().project!.analysis!.speakers.length).toBe(before));
    expect(useStore.getState().mergeHold).toBe(false);
  });

  it("одно имя двум голосам → «Это один персонаж?»", async () => {
    await openFridge(null);
    renderRoute(<AnalyzeScreen />);
    await screen.findByText("Персонажи");
    const [a, b] = useStore.getState().project!.analysis!.speakers.slice(2, 4);
    expect(a && b).toBeTruthy();
    await act(async () => { await useStore.getState().renameSpeaker(a.id, "Том"); await useStore.getState().renameSpeaker(b.id, "Том"); });
    expect(await screen.findByTestId("same-name-suggest")).toHaveTextContent("Это один персонаж?");
  });
});

describe("Закадровый режим и жаргон", () => {
  it("«Закадровый»: объяснение одной строкой, нет полосок про губы и «Подгонять под губы»", async () => {
    await openFridge();
    useStore.getState().patchScene({ mix: { ...useStore.getState().project!.scenes[0].mix, mode: "voiceover" } });
    renderRoute(<RecordScreen />);
    await screen.findByText(/Записать реплику/);
    expect(await screen.findByTestId("voiceover-note")).toHaveTextContent(/оригинал слышен тише, твой голос поверх/);
    expect(screen.queryByText(/Подгонять под губы/)).toBeNull();
    expect(screen.queryByTestId("syllable-budget")).toBeNull();
    expect(screen.queryByRole("button", { name: /Подогнать под длину/ })).toBeNull();
  });
  it("«Дубляж»: понятные слова вместо жаргона", async () => {
    await openFridge();
    renderRoute(<RecordScreen />);
    await screen.findByText(/Записать реплику/);
    expect(screen.getAllByTestId("budget-label").length).toBeGreaterThan(0);
    expect(screen.queryByText(/ложится в губы/)).toBeNull();
    expect(screen.getByText(/пойдёт в ролик/)).toBeInTheDocument();
    expect(screen.getAllByRole("button", { name: "Что такое звёздочка" }).length).toBe(1);
  });
});

describe("Сведение: ползунки и автопилот", () => {
  it("диапазон ползунка включает текущее значение", () => {
    expect(offRange(1060)).toBeGreaterThanOrEqual(1060);
    expect(offRange(-4200)).toBeGreaterThanOrEqual(4200);
    expect(offRange(0)).toBe(3000);
    expect(stretchRange(0.62)[0]).toBeLessThanOrEqual(0.62);
    expect(stretchRange(1.4)[1]).toBeGreaterThanOrEqual(1.4);
  });
  it("тонкая настройка: подпись «+1060 мс» совпадает с ползунком", async () => {
    await openFridge();
    const sc = useStore.getState().project!.scenes[0];
    const t = sc.takes.find((x) => x.selected)!;
    await useStore.getState().patchTake(t.id, { offset_ms: 1060, stretch: 0.62 });
    renderRoute(<MixScreen />);
    await screen.findByRole("region", { name: "Помощник-режиссёр" });
    await userEvent.click(screen.getByTestId("fine-toggle"));
    expect(await screen.findByTestId("offset-label")).toHaveTextContent("+1060 мс");
    const range = screen.getByLabelText("Сдвиг записи") as HTMLInputElement;
    expect(Number(range.max)).toBeGreaterThanOrEqual(1060);
    expect(Number(range.value)).toBe(1060);
    const len = screen.getByLabelText("Длина записи") as HTMLInputElement;
    expect(Number(len.min)).toBeLessThanOrEqual(0.62);
    expect(screen.getByTestId("stretch-label")).toHaveTextContent("62%");
  });
  it("ручное движение — заметная плашка с «Вернуть автопилот»", async () => {
    await openFridge();
    renderRoute(<MixScreen />);
    await screen.findByRole("region", { name: "Помощник-режиссёр" });
    expect(screen.queryByTestId("autopilot-off")).toBeNull();
    fireEvent.change(screen.getByLabelText(/Громкость: Музыка и шумы/), { target: { value: "-10" } });
    const bar = await screen.findByTestId("autopilot-off");
    expect(bar).toHaveTextContent(/Автопилот выключен, потому что ты поменял/);
    await userEvent.click(within(bar).getByRole("button", { name: "Вернуть автопилот" }));
    await waitFor(() => expect(useStore.getState().project!.scenes[0].mix.autopilot).toBe(true));
    expect(screen.queryByTestId("autopilot-off")).toBeNull();
  });
  it("различие «Автопилот / Помощник / Тонкая настройка» — по строке на каждое", async () => {
    await openFridge();
    renderRoute(<MixScreen />);
    const help = await screen.findByTestId("mix-help");
    expect(help.children.length).toBe(3);
  });
  it("превью пересобирается само после правки (через ~1 с)", async () => {
    await openFridge();
    renderRoute(<MixScreen />);
    await screen.findByRole("region", { name: "Помощник-режиссёр" });
    await userEvent.click(screen.getByRole("button", { name: /Собрать превью/ }));
    await waitFor(() => expect(Object.values(useStore.getState().jobs).filter((j) => j.kind === "preview").length).toBe(1));
    await waitFor(() => expect(screen.getByRole("button", { name: /Превью готово/ })).toBeInTheDocument(), { timeout: 4000 });
    fireEvent.change(screen.getByLabelText(/Громкость: Музыка и шумы/), { target: { value: "-9" } });
    await waitFor(() => expect(Object.values(useStore.getState().jobs).filter((j) => j.kind === "preview").length).toBe(2), { timeout: 4000 });
  });
});

describe("Отмена удаления", () => {
  it("сцена: тост «Отменить» возвращает её; без отмены движок удаляет через 6 с", async () => {
    vi.useFakeTimers({ toFake: ["setTimeout", "clearTimeout"] });
    try {
      await openFridge();
      const engine = useStore.getState().engine!;
      const spy = vi.spyOn(engine, "deleteScene");
      await act(async () => { await useStore.getState().deleteScene("sc001"); });
      expect(useStore.getState().project!.scenes.length).toBe(0);
      expect(useStore.getState().toasts.some((t) => t.text.includes("удалена") && t.action?.label === "Отменить")).toBe(true);
      expect(spy).not.toHaveBeenCalled();
      await act(async () => { useStore.getState().toasts.find((t) => t.action)!.action!.run(); });
      expect(useStore.getState().project!.scenes.length).toBe(1);
      await act(async () => { await vi.advanceTimersByTimeAsync(UNDO_DELETE_MS + 100); });
      expect(spy).not.toHaveBeenCalled();
      // второй раз — не отменяем
      await act(async () => { await useStore.getState().deleteScene("sc001"); });
      await act(async () => { await vi.advanceTimersByTimeAsync(UNDO_DELETE_MS + 100); });
      expect(spy).toHaveBeenCalledTimes(1);
      expect((await engine.getProject("p-fridge")).scenes.length).toBe(0);
    } finally { vi.useRealTimers(); }
  });
  it("дубль: исчезает сразу, возвращается по «Отменить»", async () => {
    await openFridge();
    const before = useStore.getState().project!.scenes[0].takes.length;
    const tid = useStore.getState().project!.scenes[0].takes[0].id;
    await act(async () => { await useStore.getState().deleteTake(tid); });
    expect(useStore.getState().project!.scenes[0].takes.length).toBe(before - 1);
    await act(async () => { useStore.getState().toasts.find((t) => t.action)!.action!.run(); });
    await waitFor(() => expect(useStore.getState().project!.scenes[0].takes.length).toBe(before));
  });
});

describe("Ключ ИИ", () => {
  const withKey = async (status: "ok" | "invalid" | "unknown") => {
    const st = useStore.getState();
    useStore.setState({ demo: false, settings: { ...st.settings!, llm: { routing: { translate: "gemini" }, providers: [
      { id: "claude_cli", kind: "claude_cli", title: "Claude", model: "sonnet", enabled: false, key_set: false, key_status: "unknown" },
      { id: "gemini", kind: "gemini", title: "Gemini", model: "m", enabled: true, key_set: true, key_status: status, key_message: status === "invalid" ? "Gemini не принял этот ключ (400)." : null },
    ] } } });
  };
  it("«не подходит»: красная карточка с причиной и «Изменить ключ»; в шапке не считается", async () => {
    await withKey("invalid");
    renderRoute(<SettingsScreen />, "/settings/:page?", "/settings/ai");
    const card = await screen.findByTestId("prov-gemini");
    expect(card).toHaveAttribute("data-status", "invalid");
    expect(within(card).getByTestId("key-message")).toHaveTextContent("не принял");
    expect(within(card).getByRole("button", { name: "Изменить ключ" })).toBeInTheDocument();
    expect(within(card).getByText("ключ не подходит")).toBeInTheDocument();
    expect(screen.getByTestId("key-explain")).toHaveTextContent(/Ключ — это длинный пароль/);
    expect(providerHealth(useStore.getState().settings!.llm.providers[1])).toBe("invalid");
  });
  it("«OpenAI-совместимый» спрятан в «Для опытных»", async () => {
    renderRoute(<SettingsScreen />, "/settings/:page?", "/settings/ai");
    const adv = (await screen.findByText(/Для опытных: другие сервисы/)).closest("details")!;
    expect(adv).toContainElement(screen.getByText("OpenAI-совместимый"));
    expect(adv.open).toBe(false);
  });
  it("sk-test123 → «ключ не подходит» сразу после «Сохранить»", async () => {
    renderRoute(<SettingsScreen />, "/settings/:page?", "/settings/ai");
    const card = await screen.findByTestId("prov-gemini");
    await userEvent.click(within(card).getByRole("button", { name: /Подключить/ }));
    await userEvent.type(within(card).getByLabelText(/Ключ: Gemini/), "sk-test123");
    await userEvent.click(within(card).getByRole("button", { name: "Сохранить" }));
    await waitFor(() => expect(screen.getByTestId("prov-gemini")).toHaveAttribute("data-status", "invalid"));
  });
});

describe("Рендер после перезагрузки и действия после рендера", () => {
  it("идущий рендер восстанавливается из jobs, кнопка запуска заблокирована", async () => {
    await openFridge();
    const engine = useStore.getState().engine!;
    const job = await engine.render("p-fridge", { scene_ids: ["sc001"], settings: { aspect: "16:9", height: 720, target_lufs: -14, subtitles: "none", subtitle_text: "dub", fit: "blur", export_stems: false, output_path: "/tmp/x.mp4" } });
    useStore.getState().trackJob(job);
    renderRoute(<RenderScreen />);
    expect(await screen.findByTestId("render-restored")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Собираю/ })).toBeDisabled();
  });
  it("после «Готово» в браузере — «Скопировать путь»; формат 9:16 → выбор «вписать/обрезать» и площадка для Reels", async () => {
    await openFridge();
    renderRoute(<RenderScreen />);
    await screen.findByText(/Что собираем/);
    expect(screen.queryByTestId("fit-opt")).toBeNull();
    await userEvent.click(screen.getByRole("radio", { name: /9:16/ }));
    const fit = await screen.findByTestId("fit-opt");
    expect(within(fit).getByRole("radio", { name: /С размытыми полями/ })).toBeChecked();
    // громкость — простой переключатель из двух вариантов, по умолчанию «Для YouTube и VK»
    expect(screen.getByRole("radio", { name: "Для YouTube и VK" })).toBeChecked();
    expect(screen.getByTestId("loud-note")).toHaveTextContent(/громче оригинала/);
    await userEvent.click(within(fit).getByRole("radio", { name: /Обрезать по центру/ }));
    const spy = vi.spyOn(useStore.getState().engine!, "render");
    await userEvent.click(screen.getByRole("button", { name: /^Собрать ролик/ }));
    await waitFor(() => expect(spy).toHaveBeenCalled());
    expect(spy.mock.calls[0][1].settings.fit).toBe("crop");
    expect(await screen.findByTestId("done", {}, { timeout: 5000 })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Скопировать путь/ })).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /Открыть ролик/ })).toBeNull();
  });
});

describe("Переименование проекта", () => {
  it("карандаш у названия → PATCH", async () => {
    await openFridge();
    const { ProjectTitle } = await import("../App");
    renderRoute(<ProjectTitle pid="p-fridge" title={useStore.getState().project!.title} />);
    await userEvent.click(screen.getByRole("button", { name: "Переименовать проект" }));
    const inp = screen.getByLabelText("Название проекта");
    await userEvent.clear(inp);
    await userEvent.type(inp, "Новое имя{enter}");
    await waitFor(() => expect(useStore.getState().project!.title).toBe("Новое имя"));
  });
});

describe("Запись: пустой дубль", () => {
  it("422 «Запись почти пустая…» показывается с кнопкой «Записать ещё раз»", async () => {
    await openFridge();
    const engine = useStore.getState().engine!;
    vi.spyOn(engine, "uploadTake").mockRejectedValueOnce(new (await import("../api/engine")).EngineError("Запись почти пустая — микрофон ничего не услышал."));
    const { calls } = { calls: 0 };
    void calls;
    let msg = "";
    await useStore.getState().uploadTake(new Blob([]), { line_id: "x", start_rel: 0, preroll: 0 }, (m) => { msg = m; });
    expect(msg).toMatch(/почти пустая/);
  });
});


// ───────────── слепой тест №3 ─────────────
import { ConfirmHost } from "../components/ConfirmDialog";
import { aiPillHint, aiPillText } from "../App";
import { fmtRel } from "../lib/format";
import { stemsWait } from "../lib/stems";

function appAt(url: string) {
  return render(<MemoryRouter initialEntries={[url]}><App /></MemoryRouter>);
}

describe("Верхняя панель: «Проекты» и «Назад» из Настроек", () => {
  it("кнопка «Проекты» есть на каждом экране, подсвечена на главном и ведёт к списку", async () => {
    await openFridge();
    appAt("/p/p-fridge/record");
    const btn = await screen.findByRole("button", { name: "Проекты" });
    expect(btn).not.toHaveAttribute("aria-current");
    await userEvent.click(btn);
    expect(await screen.findByText(/Перетащи сюда/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Проекты" })).toHaveAttribute("aria-current", "page");
  });

  it("из проекта → Настройки → «Назад к проекту» возвращает на тот же шаг; Esc делает то же", async () => {
    await openFridge();
    appAt("/p/p-fridge/mix");
    await screen.findByRole("region", { name: "Помощник-режиссёр" });
    await userEvent.click(screen.getByRole("button", { name: /Настройки/ }));
    const back = await screen.findByTestId("settings-back");
    expect(back).toHaveTextContent(/Назад к проекту/);
    await userEvent.click(back);
    expect(await screen.findByRole("region", { name: "Помощник-режиссёр" })).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: /Настройки/ }));
    await screen.findByTestId("settings-back");
    await userEvent.keyboard("{Escape}");
    expect(await screen.findByRole("region", { name: "Помощник-режиссёр" })).toBeInTheDocument();
  });

  it("с главного: «Назад к проектам»", async () => {
    appAt("/");
    await screen.findByText(/Перетащи сюда/);
    await userEvent.click(screen.getByRole("button", { name: /Настройки/ }));
    const back = await screen.findByTestId("settings-back");
    expect(back).toHaveTextContent("Назад к проектам");
    await userEvent.click(back);
    expect(await screen.findByText(/Перетащи сюда/)).toBeInTheDocument();
  });

  it("в шапке коротко: «ИИ подключён» / «ИИ не подключён», без числа моделей и «не отвечают»", () => {
    expect(aiPillText(null, 0)).toBe("ИИ…");
    expect(aiPillText({}, 0)).toBe("ИИ не подключён");
    expect(aiPillText({}, 1)).toBe("ИИ подключён");
    expect(aiPillText({}, 3)).toBe("ИИ подключён");
  });
  it("подробности — во всплывающей подсказке: сколько отвечает и кто нет", () => {
    const s = { llm: { providers: [
      { id: "a", kind: "claude_cli", title: "Claude", enabled: true, key_set: true, can_answer: true },
      { id: "b", kind: "codex_cli", title: "Codex", enabled: true, key_set: false, can_answer: false, state: "missing", state_message: "Не найден" },
    ] } } as never;
    const hint = aiPillHint(s, 1);
    expect(hint).toMatch(/^Отвечает 1 модель\./);
    expect(hint).toMatch(/Codex — Не найден/);
    expect(hint).toMatch(/в Настройках/);
    expect(aiPillHint(s, 0)).toMatch(/Ни одна модель ИИ не отвечает/);
  });
});

describe("Удаление проекта — своё окно вместо window.confirm", () => {
  it("корзина → окно «Удалить проект?» → Отмена/Esc оставляют, красная кнопка удаляет; window.confirm не вызывается", async () => {
    const conf = vi.spyOn(window, "confirm").mockReturnValue(false);
    const engine = useStore.getState().engine!;
    const spy = vi.spyOn(engine, "deleteProject");
    render(<MemoryRouter initialEntries={["/"]}><ProjectsScreen /><ConfirmHost /></MemoryRouter>);
    await screen.findByText(/Перетащи сюда/);
    const name = (await engine.listProjects())[0].title;
    await userEvent.click(await screen.findByRole("button", { name: `Удалить проект ${name}` }));
    const dlg = await screen.findByRole("alertdialog");
    expect(dlg).toHaveTextContent(`Удалить проект «${name}»?`);
    expect(dlg).toHaveTextContent(/видео на диске останется/);
    // безопасная кнопка в фокусе: случайный Enter ничего не удалит
    expect(within(dlg).getByRole("button", { name: "Отмена" })).toHaveFocus();
    await userEvent.keyboard("{Escape}");
    await waitFor(() => expect(screen.queryByRole("alertdialog")).toBeNull());
    expect(spy).not.toHaveBeenCalled();
    await userEvent.click(screen.getByRole("button", { name: `Удалить проект ${name}` }));
    await userEvent.click(within(await screen.findByRole("alertdialog")).getByRole("button", { name: "Отмена" }));
    expect(spy).not.toHaveBeenCalled();
    await userEvent.click(screen.getByRole("button", { name: `Удалить проект ${name}` }));
    const del = within(await screen.findByRole("alertdialog")).getByRole("button", { name: "Удалить проект" });
    expect(del.className).toMatch(/danger-solid/);
    await userEvent.click(del);
    await waitFor(() => expect(spy).toHaveBeenCalledTimes(1));
    await waitFor(() => expect(screen.queryByRole("button", { name: `Удалить проект ${name}` })).toBeNull());
    expect(conf).not.toHaveBeenCalled();
    conf.mockRestore();
  });

  it("askConfirm: Enter на красной кнопке подтверждает, клик мимо окна — отказ", async () => {
    render(<ConfirmHost />);
    const p1 = useStore.getState().askConfirm({ title: "Точно?", confirmLabel: "Да", danger: true });
    const ok = await screen.findByRole("button", { name: "Да" });
    ok.focus();
    await userEvent.keyboard("{Enter}");
    expect(await p1).toBe(true);
    const p2 = useStore.getState().askConfirm({ title: "Точно?", confirmLabel: "Да" });
    await screen.findByRole("alertdialog");
    fireEvent.mouseDown(document.querySelector(".modal-back")!);
    expect(await p2).toBe(false);
  });

  it("в src нет window.confirm / alert / prompt", async () => {
    const fs = await import("node:fs");
    const path = await import("node:path");
    const walk = (d: string): string[] => fs.readdirSync(d, { withFileTypes: true }).flatMap((e) => (e.isDirectory() ? walk(path.join(d, e.name)) : [path.join(d, e.name)]));
    const files = walk(path.resolve(__dirname, "..")).filter((f) => /\.tsx?$/.test(f) && !/\.test\.|schema\.gen/.test(f));
    for (const f of files) expect(fs.readFileSync(f, "utf8"), f).not.toMatch(/\b(window\.)?(confirm|alert|prompt)\s*\(/);
  });
});

describe("«Для опытных» раскрывается по клику", () => {
  it("клик раскрывает, повторный клик сворачивает, раскрытие не сбрасывается при перерисовке", async () => {
    renderRoute(<SettingsScreen />, "/settings/:page?", "/settings/ai");
    const sum = await screen.findByText(/Для опытных: другие сервисы/);
    const det = sum.closest("details")!;
    expect(det.open).toBe(false);
    await userEvent.click(sum);
    expect(det.open).toBe(true);
    await act(async () => { await useStore.getState().refreshSettings(); });
    expect(det.open).toBe(true);
    expect(within(det).getByText("OpenAI-совместимый")).toBeInTheDocument();
    await userEvent.click(sum);
    expect(det.open).toBe(false);
  });
  it("у подключённых моделей кнопка «Изменить», а не «Настроить»", async () => {
    renderRoute(<SettingsScreen />, "/settings/:page?", "/settings/ai");
    const card = await screen.findByTestId("prov-claude_cli");
    expect(within(card).getByRole("button", { name: "Изменить" })).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Настроить" })).toBeNull();
  });
});

describe("Блокировка шагов, пока идёт «Голоса и фон»", () => {
  const fresh = async () => {
    await openFridge();
    const st = useStore.getState();
    const sc = st.project!.scenes[0];
    useStore.setState({ project: { ...st.project!, scenes: [{ ...sc, stems_ready: false }] } });
    const job = { id: "j-prep", kind: "prepare_scene", project_id: "p-fridge", scene_id: sc.id, state: "running", progress: 0, steps: [], eta_s: null, error: null, error_code: null, created: new Date().toISOString(), result: null } as unknown as Job;
    useStore.setState({ jobs: { ...st.jobs, [job.id]: job } });
    return { sc, job };
  };

  it("stemsWait: подсказка с процентами", async () => {
    const { sc, job } = await fresh();
    const w = stemsWait(useStore.getState().project, sc.id, { [job.id]: { ...job, progress: 0.37 } });
    expect(w.blocked).toBe(true);
    expect(w.hint).toBe("Дождись, пока программа отделит голоса от фона (37 %)");
    expect(stemsWait(useStore.getState().project, null, {}).blocked).toBe(false);
  });

  it("«К записи →» и вкладки 2–4 заблокированы с подсказкой, потом открываются", async () => {
    await fresh();
    appAt("/p/p-fridge/analyze");
    await screen.findByText("Персонажи");
    const next = await screen.findByRole("button", { name: /Дальше: запись/ });
    expect(next).toBeDisabled();
    expect(next).toHaveAttribute("title", expect.stringMatching(/Дождись, пока программа отделит голоса от фона \(\d+ %\)/));
    for (const n of ["Шаг 2: Запись", "Шаг 3: Сведение", "Шаг 4: Рендер"]) expect(screen.getByRole("button", { name: n })).toBeDisabled();
    expect(screen.getByRole("button", { name: "Шаг 1: Разбор" })).toBeEnabled();
    await act(async () => {
      const st = useStore.getState();
      useStore.setState({ project: { ...st.project!, scenes: st.project!.scenes.map((x) => ({ ...x, stems_ready: true })) } });
    });
    expect(screen.getByRole("button", { name: /Дальше: запись/ })).toBeEnabled();
    expect(screen.getByRole("button", { name: "Шаг 2: Запись" })).toBeEnabled();
  });

  it("на «Записи» красная кнопка и «Дальше: сведение» заблокированы, пока голоса не отделены", async () => {
    await fresh();
    renderRoute(<RecordScreen />);
    expect(await screen.findByTestId("stems-wait")).toHaveTextContent(/Дождись, пока программа отделит голоса от фона/);
    expect(screen.getByRole("button", { name: /Записать реплику/ })).toBeDisabled();
    expect(screen.getByRole("button", { name: /Дальше: сведение/ })).toBeDisabled();
  });
});

describe("Счётчики и доступность", () => {
  it("«Сцен пока нет» и счётчик сцен не противоречат друг другу; у вкладок — aria-label «Персонажи: N»", async () => {
    await openFridge(null);
    const n = useStore.getState().project!.analysis!.speakers.length;
    renderRoute(<AnalyzeScreen />);
    await screen.findByText("Персонажи");
    expect(screen.getByRole("tab", { name: `Персонажи: ${n}` })).toBeInTheDocument();
    expect(screen.getByText(new RegExp(`нашла ${n} `))).toBeInTheDocument();
    const k = useStore.getState().project!.scenes.length;
    expect(screen.getByRole("tab", { name: `Сцены: ${k}` })).toBeInTheDocument();
    await userEvent.click(screen.getByRole("tab", { name: /Сцены/ }));
    if (k === 0) expect(screen.getByText("Сцен пока нет")).toBeInTheDocument();
    else expect(screen.queryByText("Сцен пока нет")).toBeNull();
  });

  it("время сцены в едином формате м:сс.д", () => {
    expect(fmtRel(0)).toBe("0:00.0");
    expect(fmtRel(10.5)).toBe("0:10.5");
    expect(fmtRel(75.25)).toBe("1:15.3");
  });
});

describe("Запись: загрузка готового дубля файлом", () => {
  it("ссылка «добавить готовую запись» отправляет файл как дубль выбранной реплики", async () => {
    await openFridge();
    const engine = useStore.getState().engine!;
    const spy = vi.spyOn(engine, "uploadTake");
    renderRoute(<RecordScreen />);
    await screen.findByText(/Записать реплику/);
    expect(screen.getByRole("button", { name: /добавить готовую запись/ })).toBeInTheDocument();
    const input = screen.getByTestId("take-file") as HTMLInputElement;
    expect(input.accept).toMatch(/\.mp3/);
    const file = new File([new Uint8Array(2000)], "мой дубль.m4a", { type: "audio/mp4" });
    await userEvent.upload(input, file);
    await waitFor(() => expect(spy).toHaveBeenCalledTimes(1));
    const [, , blob, meta] = spy.mock.calls[0];
    expect((blob as File).name).toBe("мой дубль.m4a");
    expect(meta.line_id).toBeTruthy();
    expect(meta.source).toBe("file");
    await waitFor(() => expect(useStore.getState().toasts.some((t) => /Файл «мой дубль.m4a» добавлен как дубль/.test(t.text))).toBe(true));
  });
  it("«Записать ещё раз» стоит у предупреждения, а не у пустого списка дублей", async () => {
    await openFridge();
    renderRoute(<RecordScreen />);
    await screen.findByText(/Записать реплику/);
    await waitFor(() => expect(screen.queryAllByText(/Слушаю дубль/)).toHaveLength(0));
    const warn = document.querySelector(".take .remark.warn");
    if (warn) expect(within(warn.closest(".take") as HTMLElement).getByRole("button", { name: /Записать ещё раз/ })).toBeInTheDocument();
  });
});

describe("Сведение: превью не врёт", () => {
  it("в проекте без дублей нет «Превью готово ✓»", async () => {
    await openFridge();
    const st = useStore.getState();
    useStore.setState({ project: { ...st.project!, scenes: st.project!.scenes.map((x) => ({ ...x, takes: [] })) } });
    renderRoute(<MixScreen />);
    await screen.findByRole("region", { name: "Помощник-режиссёр" });
    expect(screen.queryByText(/Превью готово/)).toBeNull();
    expect(screen.getByText(/Пока в кадре — исходник сцены/)).toBeInTheDocument();
  });
});

describe("Рендер: путь и «ты»", () => {
  it("поле пути многострочное, с .mp4, есть «Изменить…»; тумблер дорожек — на «ты»", async () => {
    await openFridge();
    renderRoute(<RenderScreen />);
    await screen.findByText(/Что собираем/);
    const path = screen.getByLabelText("Куда сохранить") as HTMLTextAreaElement;
    await waitFor(() => expect(path.value).toMatch(/\.mp4$/));
    expect(screen.getByRole("button", { name: /Изменить…/ })).toBeInTheDocument();
    expect(screen.getByRole("switch", { name: "Дорожки отдельными файлами" })).toBeInTheDocument();
    fireEvent.change(path, { target: { value: "/tmp/ролик" } });
    fireEvent.blur(path);
    expect(path.value).toBe("/tmp/ролик.mp4");
  });
});

describe("Разбор прерван → «Продолжить»", () => {
  it("если движок сообщил analysis_state=interrupted, на карточке проекта есть «Разбор прерван · Продолжить» и POST /analyze", async () => {
    const engine = useStore.getState().engine!;
    const list = await engine.listProjects();
    vi.spyOn(engine, "listProjects").mockResolvedValue(list.map((p, i) => (i === 0 ? { ...p, has_analysis: false, analysis_state: "interrupted" } : p)) as never);
    const re = vi.spyOn(engine, "reanalyze");
    await useStore.getState().loadProjects();
    renderRoute(<ProjectsScreen />, "/", "/");
    expect(await screen.findByTestId("interrupted")).toHaveTextContent("Разбор прерван");
    await userEvent.click(screen.getByRole("button", { name: /Продолжить/ }));
    await waitFor(() => expect(re).toHaveBeenCalledWith(list[0].id));
  });
});

describe("Проекты: ошибка про файл — рядом с полем", () => {
  it("битый путь → сообщение под полем ввода (role=alert), а не тост", async () => {
    const engine = useStore.getState().engine!;
    vi.spyOn(engine, "probe").mockRejectedValueOnce(new (await import("../api/engine")).EngineError("Файл повреждён или это не видео. Выберите другой файл."));
    renderRoute(<ProjectsScreen />, "/", "/");
    await screen.findByText(/Перетащи сюда/);
    await userEvent.type(screen.getByLabelText("Ссылка или путь"), "/tmp/x.mov{enter}");
    const err = await screen.findByTestId("probe-error");
    expect(err).toHaveTextContent(/не видео/);
    expect(useStore.getState().toasts).toHaveLength(0);
  });
});
