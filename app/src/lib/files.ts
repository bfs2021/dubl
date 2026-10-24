import { inTauri } from "../api/client";

export interface Picked { path?: string; url?: string; name: string }

export const LANGS: Record<string, string> = {
  eng: "английский", jpn: "японский", rus: "русский", deu: "немецкий", ger: "немецкий", fra: "французский", fre: "французский",
  spa: "испанский", ita: "итальянский", kor: "корейский", zho: "китайский", chi: "китайский", por: "португальский",
  ukr: "украинский", pol: "польский", tur: "турецкий",
};
export const langName = (code: string | null | undefined) => (code ? LANGS[code.toLowerCase()] ?? code : "язык не указан");

/** Выбрать видео. В окне Tauri — системный диалог (получаем путь), в браузере — input[type=file] (только имя). */
export async function pickVideo(): Promise<Picked | null> {
  if (inTauri()) {
    const { open } = await import("@tauri-apps/plugin-dialog");
    const r = await open({ multiple: false, filters: [{ name: "Видео", extensions: ["mp4", "mkv", "mov", "avi", "webm", "m4v"] }] });
    if (!r || typeof r !== "string") return null;
    return { path: r, name: r.split(/[\\/]/).pop() ?? r };
  }
  return new Promise((resolve) => {
    const inp = document.createElement("input");
    inp.type = "file";
    inp.accept = "video/*,.mkv";
    inp.onchange = () => {
      const f = inp.files?.[0];
      resolve(f ? { path: (f as File & { path?: string }).path ?? f.name, name: f.name } : null);
    };
    inp.oncancel = () => resolve(null);
    inp.click();
  });
}

/** Имя файла ролика из названия сцены: без символов, которые нельзя в имени файла. */
export const safeFileName = (title: string) => title.replace(/[\\/:*?"<>|]+/g, "").trim() || "Ролик";

/** Папка из полного пути к файлу (понимает и «/», и «\» Windows). Пусто — папки в пути нет. */
export function dirOf(path: string): string {
  const p = path.trim();
  const i = Math.max(p.lastIndexOf("/"), p.lastIndexOf("\\"));
  if (i < 0) return "";
  if (i === 0) return p[0];                         // «/ролик.mp4» → «/»
  if (/^[A-Za-z]:[\\/]$/.test(p.slice(0, i + 1))) return p.slice(0, i + 1); // «C:\ролик.mp4» → «C:\»
  return p.slice(0, i);
}

/** Склеить папку и имя файла тем же разделителем, что в папке (Windows — «\»). */
export function joinPath(dir: string, name: string): string {
  const sep = dir.includes("\\") && !dir.includes("/") ? "\\" : "/";
  return /[\\/]$/.test(dir) ? dir + name : dir + sep + name;
}

/** Куда сохранять ролик по умолчанию: в папку, куда сохраняли в прошлый раз, иначе — «Видео/Дубль». */
export async function defaultOutputPath(title: string, lastDir?: string | null): Promise<string> {
  const safe = safeFileName(title);
  if (lastDir && lastDir.trim()) return joinPath(lastDir.trim(), safe + ".mp4");
  if (inTauri()) {
    try {
      const { videoDir, join } = await import("@tauri-apps/api/path");
      return await join(await videoDir(), "Дубль", safe + ".mp4");
    } catch { /* нет доступа к пути */ }
  }
  return `~/Movies/Дубль/${safe}.mp4`;
}

export async function pickSavePath(current: string): Promise<string | null> {
  if (!inTauri()) return null;
  const { save } = await import("@tauri-apps/plugin-dialog");
  return (await save({ defaultPath: current, filters: [{ name: "Видео MP4", extensions: ["mp4"] }] })) ?? null;
}
