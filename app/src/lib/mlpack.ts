/**
 * «Скачать нейросети»: события установщика из Rust-оболочки (src-tauri/src/ml_pack.rs), русские подписи этапов и ошибок.
 * Сам запуск — команды Tauri `install_ml_pack` / `cancel_ml_pack` / `ml_pack_status`; аргументы sidecar собирает Rust.
 */

/** Событие `ml-pack` от оболочки. */
export type MlEvent =
  | { kind: "stage"; stage: string; index: number; total: number }
  | { kind: "log"; detail: string }
  | { kind: "done" }
  | { kind: "failed"; reason: string }
  | { kind: "cancelled" };

/** idle — ничего не идёт; running — качаем; restarting — пакет стоит, движок перезапускается; failed — не вышло. */
export type MlPhase = "idle" | "running" | "restarting" | "failed";

export interface MlProgress {
  phase: MlPhase;
  stage: string;
  index: number;
  total: number;
  /** последняя строка вывода uv, уже по-русски */
  detail: string;
  /** сырая причина сбоя (для «Подробности») */
  error: string | null;
  /** пакет только что установили в этой сессии (после этого на разборе предлагаем «Разобрать снова») */
  installed: boolean;
}

export const initialMl = (): MlProgress => ({ phase: "idle", stage: "", index: 0, total: 4, detail: "", error: null, installed: false });

export function reduceMl(s: MlProgress, ev: MlEvent): MlProgress {
  switch (ev.kind) {
    case "stage":
      return { ...s, phase: "running", stage: ev.stage, index: ev.index, total: ev.total || s.total, detail: "", error: null };
    case "log":
      return s.phase === "running" ? { ...s, detail: detailText(ev.detail) } : s;
    case "done":
      return { ...s, phase: "restarting", detail: "", error: null, installed: true };
    case "failed":
      return { ...s, phase: "failed", detail: "", error: ev.reason };
    case "cancelled":
      return { ...initialMl(), installed: s.installed };
  }
}

const STAGE_TITLES: Record<string, string> = {
  uv: "Скачиваю установщик (uv)",
  python: "Скачиваю Python",
  torch: "Скачиваю torch — это самый долгий этап",
  engine: "Ставлю библиотеки: речь, голоса, разделение звука",
};
export const stageTitle = (stage: string) => STAGE_TITLES[stage] ?? "Устанавливаю нейросети";

/** Доля полосы по этапам (внутри этапа точного прогресса нет — берём середину этапа). */
export function mlFraction(s: Pick<MlProgress, "phase" | "index" | "total">): number {
  if (s.phase === "restarting") return 1;
  if (s.phase !== "running" || s.index <= 0) return 0;
  return Math.min(0.97, (s.index - 0.5) / Math.max(1, s.total));
}

/** Строка вывода uv → короткая подсказка по-русски: «Downloading torch (60.9MiB)» → «Скачиваю torch (60,9 МБ)». */
export function detailText(line: string): string {
  const l = line.trim();
  const size = (s: string) => s.replace(/\((\d+(?:\.\d+)?)\s*MiB\)/, (_m, n) => `(${String(n).replace(".", ",")} МБ)`)
    .replace(/\((\d+(?:\.\d+)?)\s*KiB\)/, (_m, n) => `(${String(n).replace(".", ",")} КБ)`)
    .replace(/\((\d+(?:\.\d+)?)\s*GiB\)/, (_m, n) => `(${String(n).replace(".", ",")} ГБ)`);
  let m = l.match(/^Downloading\s+(.+)$/i);
  if (m) return size(`Скачиваю ${m[1]}`);
  m = l.match(/^Downloaded\s+(.+)$/i);
  if (m) return `Скачал ${m[1]}`;
  m = l.match(/^Building\s+(.+)$/i);
  if (m) return `Собираю ${m[1]}`;
  m = l.match(/^Prepared\s+(\d+)\s+packages?/i);
  if (m) return `Подготовил пакетов: ${m[1]}`;
  m = l.match(/^Installed\s+(\d+)\s+packages?/i);
  if (m) return `Поставил пакетов: ${m[1]}`;
  m = l.match(/^Resolved\s+(\d+)\s+packages?/i);
  if (m) return `Подобрал версии пакетов: ${m[1]}`;
  m = l.match(/^Скачиваю uv/);
  if (m) return "Скачиваю установщик (uv)";
  return l.length > 90 ? l.slice(0, 89) + "…" : l;
}

export interface MlFailure { title: string; /** что делать */ hint: string }

/** Сырая причина сбоя установщика → человеческий текст. Сырое остаётся в «Подробностях». */
export function humanizeMlFailure(reason: string | null | undefined): MlFailure {
  const r = reason ?? "";
  if (/No space left|os error 28|ENOSPC|Недостаточно места|not enough space|Disk quota/i.test(r)) {
    return { title: "Не хватило места на диске.", hint: "Освободи около 3 ГБ (после установки пакет занимает около 1,3 ГБ) и нажми «Повторить»." };
  }
  if (/Name or service not known|nodename nor servname|Temporary failure in name resolution|urlopen error|dns error|failed to lookup|error sending request|Connection (refused|reset|timed out|aborted)|timed out|Network is unreachable|No route to host|SSL|certificate|Could not resolve|connection error|не удалось подключиться/i.test(r)) {
    return { title: "Нет интернета или он оборвался.", hint: "Проверь подключение и нажми «Повторить» — уже скачанное повторно качать не придётся." };
  }
  if (/контрольная сумма/i.test(r)) {
    return { title: "Скачанный файл не прошёл проверку.", hint: "Он повредился по дороге или его подменили. Нажми «Повторить»; если не помогает — смени сеть (VPN, прокси)." };
  }
  if (/Permission denied|Operation not permitted|Отказано в доступе/i.test(r)) {
    return { title: "Нет доступа к папке данных программы.", hint: "Проверь права на папку «Дубль» в ~/Library/Application Support и повтори." };
  }
  if (/нет колеса|dubl_engine-\*\.whl|ожидалось ml/i.test(r)) {
    return { title: "В этой сборке нет набора библиотек для установки.", hint: "Скачай актуальную версию программы." };
  }
  return { title: "Не получилось установить нейросети.", hint: "Нажми «Повторить»; если снова не выйдет — открой «Подробности» и покажи их разработчику." };
}

// ───────────── Tauri ─────────────
export interface MlPackStatus {
  installed: boolean;
  engine_version: string;
  pack: { engine_version?: string; device?: string } | null;
  recommended_device: string;
}

export async function installMlPack(): Promise<void> {
  const { invoke } = await import("@tauri-apps/api/core");
  await invoke("install_ml_pack");
}

export async function cancelMlPack(): Promise<boolean> {
  const { invoke } = await import("@tauri-apps/api/core");
  return invoke<boolean>("cancel_ml_pack");
}

export async function mlPackStatus(): Promise<MlPackStatus | null> {
  try {
    const { invoke } = await import("@tauri-apps/api/core");
    return await invoke<MlPackStatus>("ml_pack_status");
  } catch {
    return null;
  }
}

export interface MlSnapshot { running: boolean; last: MlEvent | null }
export async function mlPackSnapshot(): Promise<MlSnapshot | null> {
  try {
    const { invoke } = await import("@tauri-apps/api/core");
    return await invoke<MlSnapshot>("ml_pack_state");
  } catch {
    return null;
  }
}

/** Подписка на события установки. Возвращает отписку. */
export async function listenMlPack(cb: (ev: MlEvent) => void): Promise<() => void> {
  const { listen } = await import("@tauri-apps/api/event");
  return listen<MlEvent>("ml-pack", (e) => cb(e.payload));
}

/** Команда для терминала, когда кнопки нет (браузер, разработка из исходников). */
export const ML_TERMINAL_HINT = 'cd engine && uv pip install -e ".[ml]"';
