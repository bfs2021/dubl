import type { Health } from "../api/types";
import { inTauri } from "../api/client";
import { deviceName } from "./format";

export const REPO_URL = "https://github.com/pingvin1337/dubl";
export const ISSUES_URL = `${REPO_URL}/issues/new/choose`;
export const RELEASES_URL = `${REPO_URL}/releases`;
export const LICENSE_NAME = "GPL-3.0-or-later";

/** Версия оболочки/интерфейса из app/package.json (vite define). В окне Tauri берём ту, что вшита в сам установщик. */
export const BUNDLED_VERSION: string = typeof __APP_VERSION__ === "string" ? __APP_VERSION__ : "?";

export async function appVersion(): Promise<string> {
  if (inTauri()) {
    try {
      const { getVersion } = await import("@tauri-apps/api/app");
      return await getVersion();
    } catch { /* нет права или старая оболочка — возьмём вшитую версию */ }
  }
  return BUNDLED_VERSION;
}

/** Название системы без версий и имён (для сообщения об ошибке). */
export function osName(ua: string = typeof navigator === "undefined" ? "" : navigator.userAgent): string {
  if (/Windows/i.test(ua)) return "Windows";
  if (/Mac OS X|Macintosh/i.test(ua)) return "macOS";
  if (/Linux|X11/i.test(ua)) return "Linux";
  return "неизвестная система";
}

const yn = (b: boolean | undefined) => (b ? "да" : "нет");

/** Короткие сведения для обращения на GitHub. Без ключей, путей, названий проектов и файлов. */
export function reportText(appV: string, health: Health | null, demo: boolean, os = osName()): string {
  const lines = [`Дубль ${appV}`, `Система: ${os}`];
  if (demo || !health) {
    lines.push("Движок: не подключён (демо-режим)");
  } else {
    lines.push(`Движок: ${health.version}${health.version !== appV ? " (отличается от версии программы)" : ""}`);
    lines.push(`Устройство обработки: ${deviceName(health.device)}`);
    const c = health.capabilities;
    lines.push(`Нейросети: ${c.ml ? "установлены" : "не установлены"} (речь: ${yn(c.asr)}, голоса и фон: ${yn(c.separation)}, кто говорит: ${yn(c.diarization)}, GigaAM: ${yn(c.gigaam)}${c.gigaam ? `, модель скачана: ${yn(c.gigaam_downloaded)}` : ""})`);
  }
  return lines.join("\n");
}
