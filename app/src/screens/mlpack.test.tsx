import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { act, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { renderRoute, wait } from "../test/helpers";
import { MockEngine } from "../api/mock";
import { defaultUi, useStore } from "../state/store";
import { initialMl, type MlEvent } from "../lib/mlpack";
import { ProjectsScreen } from "./Projects";
import { AnalyzeScreen } from "./Analyze";
import { SettingsScreen } from "./Settings";

// Rust-оболочка подменена: команды Tauri записываются, события установщика шлём вручную
const invoke = vi.fn(async (cmd: string, _args?: unknown) => (cmd === "ml_pack_state" ? { running: false, last: null } : undefined));
let emitMl: ((ev: MlEvent) => void) | null = null;
vi.mock("@tauri-apps/api/core", () => ({ invoke: (cmd: string, args?: unknown) => invoke(cmd, args) }));
vi.mock("@tauri-apps/api/event", () => ({
  listen: async (name: string, cb: (e: { payload: unknown }) => void) => {
    if (name === "ml-pack") emitMl = (ev) => cb({ payload: ev });
    return () => {};
  },
}));

vi.mock("@tauri-apps/api/webview", () => ({ getCurrentWebview: () => ({ onDragDropEvent: async () => () => {} }) }));

async function setup(opts: { ml: boolean; tauri?: boolean; demo?: boolean }) {
  localStorage.clear();
  if (opts.tauri) (window as unknown as { __TAURI_INTERNALS__: unknown }).__TAURI_INTERNALS__ = {};
  const engine = new MockEngine({ speed: 60, ml: opts.ml });
  useStore.setState({ ui: defaultUi(), toasts: [], remarks: {}, asIs: {}, project: null, sceneId: null, ml: initialMl() });
  await useStore.getState().attachEngine(engine, opts.demo ?? false, "");
  return engine;
}
const send = (ev: MlEvent) => act(() => { emitMl!(ev); });

// слушатель событий ставится один раз на всё приложение (так и в программе), поэтому между тестами не сбрасываем
beforeEach(() => { invoke.mockClear(); });
afterEach(() => { delete (window as unknown as { __TAURI_INTERNALS__?: unknown }).__TAURI_INTERNALS__; });

describe("нейросети: плашка и установка", () => {
  it("движок без нейросетей (ml=false) → плашка на главном; в браузере вместо кнопки — команда для терминала", async () => {
    await setup({ ml: false });
    renderRoute(<ProjectsScreen />, "/", "/");
    expect(await screen.findByTestId("ml-missing")).toHaveTextContent(/нужно один раз скачать нейросети/);
    expect(screen.queryByRole("button", { name: /Скачать нейросети/ })).toBeNull();
    expect(screen.getByTestId("ml-missing")).toHaveTextContent(/uv pip install/);
  });

  it("с нейросетями (ml=true) плашки нет", async () => {
    await setup({ ml: true });
    renderRoute(<ProjectsScreen />, "/", "/");
    await screen.findByText(/Перетащи сюда/);
    expect(screen.queryByTestId("ml-missing")).toBeNull();
  });

  it("в приложении: «Скачать» запускает установку, этапы по-русски, деталь, «Отмена», «Готово, перезапускаю…»", async () => {
    await setup({ ml: false, tauri: true });
    renderRoute(<ProjectsScreen />, "/", "/");
    await userEvent.click(await screen.findByRole("button", { name: /Скачать нейросети$/ }));
    expect(invoke).toHaveBeenCalledWith("install_ml_pack", undefined);
    await waitFor(() => expect(emitMl).toBeTruthy());

    await send({ kind: "stage", stage: "uv", index: 1, total: 4 });
    expect(await screen.findByTestId("ml-running")).toHaveTextContent(/Скачиваю установщик.*шаг 1 из 4/);
    await send({ kind: "stage", stage: "torch", index: 3, total: 4 });
    await send({ kind: "log", detail: "Downloading torch (60.9MiB)" });
    expect(screen.getByTestId("ml-running")).toHaveTextContent(/torch.*шаг 3 из 4/);
    expect(screen.getByTestId("ml-running")).toHaveTextContent("Скачиваю torch (60,9 МБ)");
    const bar = screen.getByRole("progressbar");
    expect(Number(bar.getAttribute("aria-valuenow"))).toBeGreaterThan(40);

    await userEvent.click(screen.getByRole("button", { name: "Отмена" }));
    expect(invoke).toHaveBeenCalledWith("cancel_ml_pack", undefined);
    await send({ kind: "cancelled" });
    expect(await screen.findByTestId("ml-missing")).toBeInTheDocument();

    await userEvent.click(screen.getByRole("button", { name: /Скачать нейросети$/ }));
    await send({ kind: "stage", stage: "engine", index: 4, total: 4 });
    await send({ kind: "done" });
    expect(await screen.findByTestId("ml-restarting")).toHaveTextContent(/Готово, перезапускаю движок/);
  });

  it("ошибка установки: человеческий текст, «Подробности» и «Повторить»", async () => {
    await setup({ ml: false, tauri: true });
    renderRoute(<ProjectsScreen />, "/", "/");
    await userEvent.click(await screen.findByRole("button", { name: /Скачать нейросети$/ }));
    await waitFor(() => expect(emitMl).toBeTruthy());
    await send({ kind: "stage", stage: "torch", index: 3, total: 4 });
    await send({ kind: "failed", reason: "команда завершилась с кодом 2: uv — error sending request" });
    const box = await screen.findByTestId("ml-failed");
    expect(box).toHaveTextContent(/Нет интернета/);
    expect(box).toHaveTextContent("Подробности");
    invoke.mockClear();
    await userEvent.click(screen.getByRole("button", { name: /Повторить/ }));
    expect(invoke).toHaveBeenCalledWith("install_ml_pack", undefined);
  });

  it("«Начать разбор» без нейросетей сначала предлагает скачать их", async () => {
    await setup({ ml: false, tauri: true });
    const create = vi.spyOn(useStore.getState(), "createProject");
    renderRoute(<ProjectsScreen />, "/", "/");
    await userEvent.type(await screen.findByLabelText("Ссылка или путь"), "/tmp/film.mkv");
    await userEvent.click(within(screen.getByLabelText("Ссылка или путь").closest("form") as HTMLElement).getByRole("button", { name: "Открыть" }));
    const begin = await screen.findByRole("button", { name: /Скачать нейросети и начать разбор/ });
    await userEvent.click(begin);
    expect(invoke).toHaveBeenCalledWith("install_ml_pack", undefined);
    expect(useStore.getState().projects.some((p) => p.title.includes("film"))).toBe(false);
    expect(create).not.toHaveBeenCalled();
  });

  it("Настройки → Обработка: блок «Нейросети» с кнопкой, а когда есть — «установлены»", async () => {
    await setup({ ml: false, tauri: true });
    const { unmount } = renderRoute(<SettingsScreen />, "/settings/:page?", "/settings/processing");
    expect(await screen.findByRole("button", { name: /Скачать нейросети$/ })).toBeInTheDocument();
    unmount();
    await setup({ ml: true });
    renderRoute(<SettingsScreen />, "/settings/:page?", "/settings/processing");
    expect(await screen.findByTestId("ml-ok")).toHaveTextContent(/Нейросети установлены/);
  });
});

describe("тупик на экране разбора", () => {
  async function failedAnalyze(ml: boolean, tauri: boolean) {
    const engine = await setup({ ml, tauri });
    const { project } = await engine.createProject({ path: "/tmp/film.mkv" });
    await useStore.getState().openProject(project.id);
    useStore.setState({ project: { ...useStore.getState().project!, analysis: null } });
    await waitFor(() => expect(Object.values(useStore.getState().jobs).some((j) => j.project_id === project.id && j.state === "failed")).toBe(true), { timeout: 5000 });
    return { engine, pid: project.id };
  }

  it("разбор упал без нейросетей: ошибка, error_code=ml_missing, кнопка «Скачать нейросети»; после установки — «Разобрать снова»", async () => {
    const { engine, pid } = await failedAnalyze(false, true);
    const job = Object.values(useStore.getState().jobs).find((j) => j.project_id === pid)!;
    expect(job.error_code).toBe("ml_missing");
    renderRoute(<AnalyzeScreen />, "/p/:pid/analyze", `/p/${pid}/analyze`);
    expect(await screen.findByText(/Не установлены нейросети/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Скачать нейросети$/ })).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /Попробовать снова|Разобрать снова/ })).toBeNull();
    expect(screen.getByRole("button", { name: "К проектам" })).toBeInTheDocument();

    // пакет поставлен, движок перезапущен (в тесте — просто с нейросетями) → «Разобрать снова»
    engine.setMl(true);
    useStore.setState({ health: await engine.health() });
    expect(await screen.findByRole("button", { name: "Разобрать снова" })).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: "Разобрать снова" }));
    await waitFor(() => {
      const jobs = Object.values(useStore.getState().jobs).filter((j) => j.project_id === pid && j.kind === "analyze");
      expect(jobs.length).toBe(2);
    });
    await waitFor(() => expect(Object.values(useStore.getState().jobs).filter((j) => j.project_id === pid).some((j) => j.state === "done")).toBe(true), { timeout: 8000 });
    await wait(50);
    await useStore.getState().refreshProject();
    expect(useStore.getState().project!.analysis).toBeTruthy();
  });

  it("любой другой упавший разбор: «Попробовать снова»", async () => {
    const engine = await setup({ ml: true });
    const { project } = await engine.createProject({ path: "/tmp/film.mkv" });
    useStore.getState().trackJob({
      id: "jx", kind: "analyze", project_id: project.id, scene_id: null, state: "failed", steps: [{ key: "transcribe", title: "Текст и тайминги", progress: 0.3, state: "failed" }],
      error: "Файл повреждён", error_debug: null, error_code: null, result: null, progress: 0.3, eta_s: null, created: new Date().toISOString(),
    });
    await useStore.getState().openProject(project.id);
    useStore.setState({ project: { ...useStore.getState().project!, analysis: null } });
    renderRoute(<AnalyzeScreen />, "/p/:pid/analyze", `/p/${project.id}/analyze`);
    expect(await screen.findByText("Файл повреждён")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Попробовать снова" })).toBeInTheDocument();
    expect(screen.queryByTestId("ml-missing")).toBeNull();
  });
});
