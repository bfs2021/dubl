import { beforeEach, describe, expect, it, vi } from "vitest";
import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { MockEngine } from "../api/mock";
import { defaultUi, useStore } from "../state/store";
import { SettingsScreen } from "./Settings";
import { ProjectsScreen } from "./Projects";
import { BUNDLED_VERSION, ISSUES_URL, REPO_URL, osName, reportText } from "../lib/about";
import { Markdown, splitSections, stripComments } from "../lib/markdown";
import * as ext from "../lib/external";
import pkg from "../../package.json";

async function open(health?: { version: string }) {
  localStorage.clear();
  const engine = new MockEngine({ speed: 60 });
  if (health) engine.health = async () => ({ ok: true, version: health.version, device: "mps", capabilities: { ml: true, asr: true, separation: true, diarization: true, gigaam: true, gigaam_downloaded: true, gigaam_download_mb: 230 } });
  useStore.setState({ ui: defaultUi(), toasts: [], remarks: {}, asIs: {}, project: null, sceneId: null });
  await useStore.getState().attachEngine(engine, !health, "");
  return render(
    <MemoryRouter initialEntries={["/settings/about"]}>
      <Routes><Route path="/settings/:page?" element={<SettingsScreen />} /></Routes>
    </MemoryRouter>,
  );
}

beforeEach(() => { vi.restoreAllMocks(); });

describe("О программе", () => {
  it("версия берётся из package.json; справка видна, заметки для разработчиков — нет", async () => {
    await open();
    expect(BUNDLED_VERSION).toBe(pkg.version);
    await waitFor(() => expect(screen.getByTestId("about-version")).toHaveTextContent(`версия ${pkg.version}`));
    const body = screen.getByTestId("help-body");
    expect(within(body).getByText("Как это работает")).toBeInTheDocument();
    expect(within(body).getByText("Частые вопросы")).toBeInTheDocument();
    expect(body.textContent).not.toMatch(/Пиши здесь только про то, что уже работает/);
    expect(body.textContent).not.toMatch(/<!--/);
    expect(screen.getByText(/GNU GPL v3/)).toBeInTheDocument();
  });

  it("версия движка из /health; если не совпадает с версией программы — показано", async () => {
    await open({ version: "9.9.9" });
    await waitFor(() => expect(screen.getByTestId("about-engine")).toHaveTextContent("9.9.9"));
    expect(screen.getByTestId("about-engine")).toHaveTextContent(/не совпадает/);
  });

  it("поиск фильтрует справку и оглавление", async () => {
    await open();
    const toc = screen.getByTestId("help-toc");
    expect(within(toc).getAllByRole("button").length).toBeGreaterThan(4);
    await userEvent.type(screen.getByTestId("help-search"), "SmartScreen-такого-слова-нет");
    expect(screen.getByTestId("help-empty")).toBeInTheDocument();
    await userEvent.clear(screen.getByTestId("help-search"));
    await userEvent.type(screen.getByTestId("help-search"), "связке ключей");
    const heads = within(screen.getByTestId("help-body")).getAllByRole("heading", { level: 2 }).map((h) => h.textContent);
    expect(heads).toContain("Частые вопросы");
    expect(heads).not.toContain("Словарь");
  });

  it("сведения для отчёта: версия, система, нейросети — и никаких путей и ключей", async () => {
    const copy = vi.spyOn(ext, "copyText").mockResolvedValue(true);
    await open({ version: pkg.version });
    await userEvent.click(screen.getByTestId("about-copy"));
    await waitFor(() => expect(copy).toHaveBeenCalled());
    const text = copy.mock.calls[0][0];
    expect(text).toContain(`Дубль ${pkg.version}`);
    expect(text).toContain("Нейросети: установлены");
    expect(text).not.toMatch(/\/Users\/|\\Users\\|C:\\|\/home\/|sk-|ключ:|\.mkv|\.mp4/);
    const demo = reportText("0.1.0", null, true, "macOS");
    expect(demo).toContain("демо");
    expect(osName("Mozilla/5.0 (Windows NT 10.0; Win64; x64)")).toBe("Windows");
    expect(osName("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)")).toBe("macOS");
  });

  it("ссылки открываются через openExternal на репозиторий проекта", async () => {
    const o = vi.spyOn(ext, "openExternal").mockResolvedValue();
    await open();
    await userEvent.click(screen.getByTestId("about-src"));
    await userEvent.click(screen.getByTestId("about-issues"));
    expect(o.mock.calls.map((c) => c[0])).toEqual([REPO_URL, ISSUES_URL]);
    expect(REPO_URL).toBe("https://github.com/pingvin1337/dubl");
  });

  it("сторонние компоненты: список из THIRD-PARTY-NOTICES.md открывается по кнопке", async () => {
    await open();
    expect(screen.queryByTestId("notices-body")).toBeNull();
    await userEvent.click(screen.getByTestId("notices-toggle"));
    const b = screen.getByTestId("notices-body");
    expect(b.textContent).toMatch(/ffmpeg/);
    expect(b.textContent).toMatch(/GigaAM/);
    expect(b.querySelectorAll("table").length).toBeGreaterThan(2);
  });

  it("с главного экрана есть ссылка «О программе»", async () => {
    localStorage.clear();
    useStore.setState({ ui: defaultUi(), toasts: [], remarks: {}, asIs: {}, project: null, sceneId: null });
    await useStore.getState().attachEngine(new MockEngine({ speed: 60 }), true, "");
    render(
      <MemoryRouter initialEntries={["/"]}>
        <Routes><Route path="/" element={<ProjectsScreen />} /><Route path="/settings/:page?" element={<div>настройки-страница</div>} /></Routes>
      </MemoryRouter>,
    );
    await userEvent.click(await screen.findByTestId("about-link"));
    expect(screen.getByText("настройки-страница")).toBeInTheDocument();
  });
});

describe("мини-markdown", () => {
  it("вырезает комментарии, не вставляет HTML, ссылки только http(s)", () => {
    expect(stripComments("a<!-- x\ny -->b")).toBe("ab");
    const { container } = render(<Markdown md={'<script>alert(1)</script> **жирно** [ок](https://a.b) [нет](javascript:alert(1)) [файл](X.md)'} />);
    expect(container.querySelector("script")).toBeNull();
    expect(container.textContent).toContain("<script>");
    expect(container.querySelectorAll("a")).toHaveLength(1);
    expect(container.querySelector("b")?.textContent).toBe("жирно");
  });
  it("режет по ## и не режет внутри кода", () => {
    const s = splitSections("# T\nintro\n## A\ntext\n```\n## не заголовок\n```\n## B\nx");
    expect(s.map((x) => x.title)).toEqual(["", "A", "B"]);
  });
  it("переводы строк Windows (\\r\\n) не вешают рендер: списки, заголовки и разделы те же", () => {
    // git на Windows по умолчанию отдаёт ABOUT.md с \r\n — раньше список с \r в конце строки зацикливал рендер до нехватки памяти
    const md = "## Раздел\r\nтекст\r\n\r\n- раз\r\n- два\r\n  - вложенный\r\n\r\n1. первый\r\n";
    const { container } = render(<Markdown md={md} />);
    expect(container.querySelectorAll("li")).toHaveLength(4);
    expect(container.textContent).not.toMatch(/\r/);
    expect(splitSections(md).map((x) => x.title)).toEqual(["Раздел"]);
  });
  it("вложенный пункт без родителя сразу после абзаца — не вечный цикл", () => {
    const { container } = render(<Markdown md={"абзац\n  - сирота\nещё"} />);
    expect(container.textContent).toContain("сирота");
  });
});
