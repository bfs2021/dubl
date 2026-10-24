import { describe, expect, it } from "vitest";
import { aiPillTitle, aiReady, connectedCount, providerHealth, routeNote } from "./ai";
import { aiPillNote } from "../App";

const prov = (id: string, kind: string, extra: object = {}) => ({
  id, kind, title: id.toUpperCase(), model: "m", enabled: true, key_set: true, key_status: "ok", ...extra,
});
const settings = (providers: object[], tasks?: object, routing: object = { translate: "claude_cli", fit_length: "claude_cli" }) =>
  ({ llm: { providers, routing, tasks } }) as never;

const claudeMissing = prov("claude_cli", "claude_cli", { can_answer: false, state: "not_installed", state_message: "Не найден — установи Claude Code" });
const deepseek = prov("deepseek", "deepseek", { can_answer: true, state: "ok" });
const gemini = prov("gemini", "gemini", { key_set: false, key_status: "ok", can_answer: false, state: "no_key", state_message: "Нужен ключ." });

describe("честный статус ИИ", () => {
  it("шапка считает только модели, которые могут ответить (движок сказал can_answer)", () => {
    const s = settings([claudeMissing, deepseek, gemini]);
    expect(connectedCount(s, false)).toBe(1);
  });
  it("«ok» в key_status без ключа не считается: число не прыгает 3 → 2", () => {
    expect(providerHealth(gemini as never)).toBe("nokey");
  });
  it("причины разные для CLI", () => {
    expect(providerHealth(claudeMissing as never)).toBe("notinstalled");
    expect(providerHealth({ ...claudeMissing, state: "not_logged_in" } as never)).toBe("nologin");
    expect(providerHealth({ ...claudeMissing, can_answer: true, state: "ok" } as never)).toBe("ready");
    expect(providerHealth({ ...deepseek, enabled: false } as never)).toBe("off");
    expect(providerHealth({ ...deepseek, can_answer: false, state: "invalid_key" } as never)).toBe("invalid");
  });
  it("задача готова, если движок знает, кто ответит (в том числе запасная)", () => {
    const tasks = { fit_length: { assigned: "claude_cli", answers: "deepseek", answers_title: "DeepSeek", fallback: true },
      translate: { assigned: "claude_cli", answers: null } };
    const s = settings([claudeMissing, deepseek], tasks);
    expect(aiReady(s, "fit_length")).toBe(true);
    expect(aiReady(s, "translate")).toBe(false);
  });
  it("без поля tasks (старый движок) готовность по маршруту, как раньше", () => {
    const s = settings([prov("claude_cli", "claude_cli", { key_set: true })]);
    expect(aiReady(s, "fit_length")).toBe(true);
    expect(aiReady(s, "director")).toBe(false);
  });
  it("подсказка к плашке называет, кто не отвечает и почему; пометка «1 не отвечает»", () => {
    const s = settings([claudeMissing, deepseek]);
    expect(aiPillTitle(s)).toBe("Не отвечают: CLAUDE_CLI — Не найден — установи Claude Code");
    expect(aiPillNote(s as never)).toBe(" · 1 не отвечает");
    expect(aiPillNote(settings([deepseek]) as never)).toBe("");
  });
  it("запасная: «Сейчас отвечает …»", () => {
    const s = settings([claudeMissing, deepseek], { fit_length: { assigned: "claude_cli", answers: "deepseek", answers_title: "DeepSeek", fallback: true } });
    expect(routeNote(s, "fit_length")).toBe("Сейчас отвечает «DeepSeek» — выбранная модель не может.");
    expect(routeNote(s, "translate")).toBeNull();
  });
  it("никто не отвечает — причина из движка", () => {
    const s = settings([claudeMissing], { translate: { assigned: "claude_cli", answers: null, message: "«CLAUDE_CLI»: не найден" } });
    expect(routeNote(s, "translate")).toBe("«CLAUDE_CLI»: не найден");
  });
});
