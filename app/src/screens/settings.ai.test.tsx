import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { freshStore, renderRoute } from "../test/helpers";
import { useStore } from "../state/store";
import { SettingsScreen } from "./Settings";

/** Как в собранном приложении: движок настоящий (kind=http) и сам говорит, кто может ответить. */
function withProviders(providers: object[], tasks: object = {}) {
  const st = useStore.getState();
  useStore.setState({ demo: false, settings: { ...st.settings!, llm: { routing: { translate: "claude_cli", fit_length: "claude_cli", name_speakers: "claude_cli", director: "claude_cli" }, providers, tasks } } as never });
}
const claude = (extra: object) => ({ id: "claude_cli", kind: "claude_cli", title: "Claude (через Claude Code)", model: "sonnet", enabled: true, key_set: true, key_status: "unknown", login_supported: true, ...extra });

let fetchMock: ReturnType<typeof vi.fn>;
beforeEach(async () => {
  await freshStore();
  fetchMock = vi.fn(async (_url: unknown, init?: RequestInit) => {
    if (init?.method === "POST") return new Response(JSON.stringify({ ok: true, message: "Открыл терминал: войди и вернись сюда." }), { status: 200 });
    return new Response(JSON.stringify(useStore.getState().settings), { status: 200 });     // GET /settings?refresh=true
  });
  vi.stubGlobal("fetch", fetchMock);
});
afterEach(() => vi.unstubAllGlobals());

describe("Настройки → Модели ИИ: подписки через программу", () => {
  it("установлен и вошёл: «установлен, вход выполнен, отвечает»", async () => {
    withProviders([claude({ can_answer: true, state: "ok", state_message: "Установлен, вход выполнен, отвечает." })]);
    renderRoute(<SettingsScreen />, "/settings/:page?", "/settings/ai");
    const card = await screen.findByTestId("prov-claude_cli");
    expect(card).toHaveAttribute("data-status", "ready");
    expect(within(card).getByText("установлен, вход выполнен, отвечает")).toBeInTheDocument();
    expect(within(card).queryByTestId(`login-claude_cli`)).toBeNull();
  });

  it("не найден: понятный текст и ссылка «Как установить»", async () => {
    withProviders([claude({ can_answer: false, state: "not_installed", state_message: "Не найден — установи Claude Code и войди в аккаунт Claude.", install_url: "https://claude.com/product/claude-code" })]);
    renderRoute(<SettingsScreen />, "/settings/:page?", "/settings/ai");
    const card = await screen.findByTestId("prov-claude_cli");
    expect(card).toHaveAttribute("data-status", "notinstalled");
    expect(within(card).getByTestId("cli-message")).toHaveTextContent(/Не найден на этом компьютере\. Установи Claude Code/);
    expect(within(card).getByRole("button", { name: /Как установить Claude Code/ })).toBeInTheDocument();
    expect(within(card).getByTestId("recheck-claude_cli")).toBeInTheDocument();
  });

  it("нужен вход: кнопка «Войти» просит движок открыть терминал; потом статус перепроверяется", async () => {
    withProviders([claude({ can_answer: false, state: "not_logged_in", state_message: "Установлен, но нужен вход." })]);
    const eng = useStore.getState().engine as unknown as { kind: string; base: string };
    Object.assign(eng, { kind: "http", base: "http://127.0.0.1:1" });      // иначе в демо вход не нужен
    renderRoute(<SettingsScreen />, "/settings/:page?", "/settings/ai");
    const card = await screen.findByTestId("prov-claude_cli");
    await userEvent.click(within(card).getByTestId("login-claude_cli"));
    await waitFor(() => expect(fetchMock.mock.calls.some(([u, i]) => String(u).endsWith("/settings/llm/claude_cli/login") && (i as RequestInit).method === "POST")).toBe(true));
    expect(await within(card).findByTestId("login-message")).toHaveTextContent("Открыл терминал");
  });

  it("в списке «Какая модель что делает» назначенная, но не отвечающая модель видна и подписана; запасная названа", async () => {
    withProviders(
      [claude({ can_answer: false, state: "not_installed", state_message: "Не найден" }),
        { id: "deepseek", kind: "deepseek", title: "DeepSeek", model: "deepseek-flash", enabled: true, key_set: true, key_status: "ok", can_answer: true, state: "ok" }],
      { fit_length: { assigned: "claude_cli", answers: "deepseek", answers_title: "DeepSeek", fallback: true } });
    renderRoute(<SettingsScreen />, "/settings/:page?", "/settings/ai");
    expect(await screen.findByTestId("route-note-fit_length")).toHaveTextContent("Сейчас отвечает «DeepSeek»");
    const sel = screen.getByLabelText("Подогнать под длину") as HTMLSelectElement;
    expect(within(sel).getByText(/Claude \(через Claude Code\) — не отвечает/)).toBeInTheDocument();
  });

  it("Codex живёт в «Для опытных» и показывает свой статус", async () => {
    withProviders([claude({ can_answer: true, state: "ok" }),
      { id: "codex_cli", kind: "codex_cli", title: "Codex (через ChatGPT)", model: "default", enabled: true, key_set: true, key_status: "unknown", can_answer: false, state: "not_installed", state_message: "Не найден — установи Codex CLI.", login_supported: true }]);
    renderRoute(<SettingsScreen />, "/settings/:page?", "/settings/ai");
    const card = await screen.findByTestId("prov-codex_cli");
    expect(card.closest("details")).not.toBeNull();
    expect(within(card).getByText("не найден")).toBeInTheDocument();
  });

  it("модели Claude: haiku, sonnet, opus", async () => {
    withProviders([claude({ can_answer: true, state: "ok" })]);
    renderRoute(<SettingsScreen />, "/settings/:page?", "/settings/ai");
    const card = await screen.findByTestId("prov-claude_cli");
    await userEvent.click(within(card).getByRole("button", { name: "Изменить" }));
    const sel = within(card).getByLabelText("Модель: Claude (через Claude Code)") as HTMLSelectElement;
    expect([...sel.options].map((o) => o.value)).toEqual(["haiku", "sonnet", "opus"]);
  });

  it("микрофон: кнопка «Проверить микрофон» открывает мастер", async () => {
    withProviders([claude({ can_answer: true, state: "ok" })]);
    renderRoute(<SettingsScreen />, "/settings/:page?", "/settings/mic");
    await userEvent.click(await screen.findByTestId("mic-check-open"));
    expect(await screen.findByRole("dialog")).toBeInTheDocument();
  });
});
