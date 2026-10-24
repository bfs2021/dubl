import { inTauri } from "../api/client";

/** Открыть ссылку в обычном браузере (в окне программы переход по ссылке заменил бы сам интерфейс). */
export async function openExternal(url: string): Promise<void> {
  if (inTauri()) {
    try {
      const { open } = await import("@tauri-apps/plugin-shell");
      await open(url);
      return;
    } catch { /* нет права — упадём в window.open */ }
  }
  window.open(url, "_blank", "noopener,noreferrer");
}

/** Открыть готовый файл в программе по умолчанию (в окне Дубля — через плагин opener). */
export async function openFile(path: string): Promise<boolean> {
  if (!inTauri()) return false;
  try {
    const { openPath } = await import("@tauri-apps/plugin-opener");
    await openPath(path);
    return true;
  } catch {
    return false;
  }
}

/** Показать файл в Finder / Проводнике. */
export async function revealFile(path: string): Promise<boolean> {
  if (!inTauri()) return false;
  try {
    const { revealItemInDir } = await import("@tauri-apps/plugin-opener");
    await revealItemInDir(path);
    return true;
  } catch {
    return false;
  }
}

/** Скопировать текст в буфер обмена (в браузере, где открыть папку нельзя). */
export async function copyText(text: string): Promise<boolean> {
  try {
    await navigator.clipboard.writeText(text);
    return true;
  } catch {
    try {
      const ta = document.createElement("textarea");
      ta.value = text;
      document.body.appendChild(ta);
      ta.select();
      const ok = document.execCommand("copy");
      ta.remove();
      return ok;
    } catch {
      return false;
    }
  }
}
