import { useEffect } from "react";
import type { AppSettings, LLMTask, ProviderState } from "../api/types";
import type { Engine } from "../api/engine";
import { useStore } from "../state/store";

/**
 * Состояние модели по ответу движка (GET /settings): сам движок знает, сохранён ли ключ, подошёл ли он, а у моделей-подписок
 * (Claude Code, Codex) — установлена ли программа и выполнен ли вход.
 *  - ready — может ответить прямо сейчас;
 *  - unchecked — ключ сохранён, но ещё ни разу не проверялся (старый движок без can_answer);
 *  - invalid — модель ключ не приняла;
 *  - nokey — нужен ключ, его нет;
 *  - notinstalled — программа-подписка не найдена на компьютере;
 *  - nologin — программа есть, но нужен вход;
 *  - error — программа есть, но проверить её не вышло;
 *  - off — модель выключена.
 */
export type ProviderHealth = "ready" | "unchecked" | "invalid" | "nokey" | "notinstalled" | "nologin" | "error" | "off";

/** Это модель-подписка: работает через программу на компьютере, ключ не нужен. */
export const isCliKind = (kind: string | undefined) => kind === "claude_cli" || kind === "codex_cli";

export function providerHealth(p: ProviderState | undefined, demo = false): ProviderHealth {
  if (!p) return "nokey";
  if (!p.enabled) return "off";
  // новый движок сам говорит, может ли модель ответить, и почему нет
  if (typeof p.can_answer === "boolean" && p.state) {
    if (p.can_answer) return p.key_status === "ok" || isCliKind(p.kind) ? "ready" : "unchecked";
    switch (p.state) {
      case "not_installed": return "notinstalled";
      case "not_logged_in": return "nologin";
      case "no_key": return "nokey";
      case "invalid_key": return "invalid";
      case "off": return "off";
      default: return "error";
    }
  }
  // старый движок / демо: судим по ключу
  if (p.kind === "claude_cli") return "ready";
  if (demo && p.key_status === undefined) return "ready";
  if (p.key_set === false) return "nokey";
  if (p.key_status === "invalid") return "invalid";
  if (p.key_status === "ok") return "ready";
  // key_set === true или полей нет: ключ не проверен
  return "unchecked";
}

/** Может ли модель ответить (готова или ключ ещё не проверяли). */
export const canAnswer = (h: ProviderHealth) => h === "ready" || h === "unchecked";

/** Работает ли ИИ для задачи: движок знает, кто ответит с учётом запасной; без этого — по маршруту из настроек. */
export function aiReady(settings: AppSettings | null, task: LLMTask, demo = false): boolean {
  if (demo) return true;
  const route = settings?.llm.tasks?.[task];
  if (route) return !!route.answers;
  const id = settings?.llm.routing[task];
  if (!id) return false;
  return canAnswer(providerHealth(settings?.llm.providers.find((x) => x.id === id)));
}

export function useAiReady(task: LLMTask): boolean {
  const settings = useStore((s) => s.settings);
  const demo = useStore((s) => s.demo);
  return aiReady(settings, task, demo);
}

/** Сколько моделей реально могут ответить (для плашки сверху). Не включённые и «нужен вход / не найден / ключ не подходит» не считаем. */
export function connectedCount(settings: AppSettings | null, demo: boolean): number {
  return (settings?.llm.providers ?? []).filter((p) => canAnswer(providerHealth(p, demo))).length;
}

/** Подсказка к плашке: почему какие-то модели не считаются. Пусто, если все включённые модели работают. */
export function aiPillTitle(settings: AppSettings | null, demo = false): string {
  if (!settings) return "Модели ИИ";
  const bad = settings.llm.providers.filter((p) => p.enabled && !canAnswer(providerHealth(p, demo)) && p.state_message);
  if (!bad.length) return "Модели ИИ";
  return "Не отвечают: " + bad.map((p) => `${p.title} — ${p.state_message!.replace(/\.$/, "")}`).join("; ");
}

/** Кто ответит на задачу, если назначенная модель не может: «Сейчас отвечает Gemini». Пусто, если всё как назначено. */
export function routeNote(settings: AppSettings, task: LLMTask): string | null {
  const r = settings.llm.tasks?.[task];
  if (!r) return null;
  if (!r.assigned && !r.answers) return r.message ?? null;
  if (!r.answers) return r.message ?? "Отвечать некому — подключи модель.";
  if (r.fallback) return `Сейчас отвечает «${r.answers_title ?? r.answers}» — выбранная модель не может.`;
  return null;
}

// ─────────────────────────── вход и перепроверка ───────────────────────────
type HttpLike = Engine & { base: string; token?: string | null };

/** Адрес и заголовки для запросов, которых ещё нет в клиенте движка (вход, перепроверка статуса CLI). null — демо. */
function http(engine: Engine | null): HttpLike | null {
  return engine && engine.kind === "http" ? (engine as HttpLike) : null;
}

async function call(engine: HttpLike, method: string, path: string): Promise<Response> {
  const headers: Record<string, string> = {};
  const token = (engine as unknown as { token?: string | null }).token;   // поле приватное в HttpEngine, но заголовок нужен тот же
  if (token) headers.Authorization = `Bearer ${token}`;
  return fetch(engine.base + path, { method, headers });
}

/** Заново спросить движок про установку и вход CLI-моделей (GET /settings?refresh=true) и обновить настройки в памяти. */
export async function recheckAi(): Promise<void> {
  const eng = http(useStore.getState().engine);
  if (!eng) return;
  try {
    const res = await call(eng, "GET", "/settings?refresh=true");
    if (!res.ok) return;
    const fresh = (await res.json()) as AppSettings;
    if (fresh?.llm?.providers) useStore.setState({ settings: fresh });
  } catch { /* движок не ответил — оставляем прежнее */ }
}

/** Кнопка «Войти»: движок открывает системный терминал с командой входа выбранной модели. Возвращает текст для человека. */
export async function openLogin(providerId: string): Promise<{ ok: boolean; message: string }> {
  const eng = http(useStore.getState().engine);
  if (!eng) return { ok: true, message: "В демо-режиме вход не нужен." };
  try {
    const res = await call(eng, "POST", `/settings/llm/${encodeURIComponent(providerId)}/login`);
    const j = (await res.json().catch(() => ({}))) as { detail?: string; message?: string };
    if (!res.ok) return { ok: false, message: typeof j.detail === "string" ? j.detail : "Не получилось открыть терминал." };
    return { ok: true, message: j.message ?? "Открыл терминал: войди и вернись сюда." };
  } catch {
    return { ok: false, message: "Движок не отвечает. Проверь, что он запущен." };
  }
}

/** При возвращении в окно (после входа в терминале) статус моделей проверяется заново. Не чаще раза в 4 секунды. */
export function useRecheckOnFocus(enabled = true) {
  useEffect(() => {
    if (!enabled) return;
    let last = 0;
    const run = () => {
      const now = Date.now();
      if (now - last < 4000 || document.visibilityState === "hidden") return;
      last = now;
      void recheckAi();
    };
    window.addEventListener("focus", run);
    document.addEventListener("visibilitychange", run);
    return () => { window.removeEventListener("focus", run); document.removeEventListener("visibilitychange", run); };
  }, [enabled]);
}
