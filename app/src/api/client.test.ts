import { afterEach, describe, expect, it, vi } from "vitest";
import { HttpEngine, connectEngine } from "./client";
import { MockEngine } from "./mock";

const json = (body: unknown, status = 200, headers: Record<string, string> = {}) =>
  new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json", ...headers } });

afterEach(() => vi.unstubAllGlobals());

describe("HttpEngine", () => {
  it("зовёт нужные адреса и ставит Bearer-токен", async () => {
    const f = vi.fn(async (..._a: unknown[]) => json({ ok: true, version: "1", device: "cpu" }));
    vi.stubGlobal("fetch", f);
    const e = new HttpEngine("http://127.0.0.1:8765/", "sekret");
    expect(await e.health()).toMatchObject({ ok: true });
    expect(f).toHaveBeenCalledWith("http://127.0.0.1:8765/health", expect.objectContaining({ method: "GET", headers: expect.objectContaining({ Authorization: "Bearer sekret" }) }));
  });
  it("показывает detail из ответа движка", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => json({ detail: "Проект ещё не разобран." }, 400)));
    await expect(new HttpEngine("http://x").snap("p", { a: 1, mode: "dialogue" })).rejects.toThrow("Проект ещё не разобран.");
  });
  it("понятная ошибка, если движок молчит", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => { throw new TypeError("fail"); }));
    await expect(new HttpEngine("http://x").listProjects()).rejects.toThrow(/не отвечает/);
  });
  it("перевод и подгонка отдают ответ движка как есть", async () => {
    const f = vi.fn(async (..._a: unknown[]) => json({ translations: [{ line_id: "l1", text: "привет" }], variants: [{ line_id: "l1", texts: ["да"] }] }));
    vi.stubGlobal("fetch", f);
    const e = new HttpEngine("http://x");
    expect(await e.translate({ project_id: "p", scene_id: "s" })).toEqual({ translations: [{ line_id: "l1", text: "привет" }], variants: expect.anything() });
    expect(f.mock.calls[0][0]).toBe("http://x/llm/translate");
    expect((await e.fitLength({ project_id: "p", scene_id: "s", line_ids: ["l1"] })).variants[0].texts).toEqual(["да"]);
  });
  it("помощник-режиссёр: разбор дубля — /llm/director, просьба — /llm/director_command, без переименований", async () => {
    const take = { report: { ok: false }, feedback: "Спешишь." };
    const cmd = { patches: [{ target: "s1", field: "gain_db", value: 3, note: "Барсик громче" }], mix: { autopilot: false } };
    const f = vi.fn(async (url: unknown) => json(String(url).endsWith("director_command") ? cmd : take));
    vi.stubGlobal("fetch", f);
    const e = new HttpEngine("http://x");
    expect(await e.directorTake({ project_id: "p", scene_id: "s", take_id: "t" })).toEqual(take);
    expect(await e.directorCommand({ project_id: "p", scene_id: "s", text: "кот громче" })).toEqual(cmd);
    expect(f.mock.calls.map((c) => c[0])).toEqual(["http://x/llm/director", "http://x/llm/director_command"]);
  });
  it("рендер и склейка — один запрос POST /projects/{pid}/render", async () => {
    const f = vi.fn(async (..._a: unknown[]) => json({ id: "j1", kind: "render_project", project_id: "p", state: "queued", steps: [], result: null }));
    vi.stubGlobal("fetch", f);
    const req = { scene_ids: ["a", "b"], settings: { aspect: "16:9" as const, height: 1080, target_lufs: -14, subtitles: "none" as const, subtitle_text: "dub" as const, fit: "blur" as const, export_stems: false, output_path: "/o/clip.mp4" } };
    const job = await new HttpEngine("http://x").render("p", req);
    expect(job.kind).toBe("render_project");
    expect(f).toHaveBeenCalledTimes(1);
    expect(f.mock.calls[0][0]).toBe("http://x/projects/p/render");
    expect(JSON.parse((f.mock.calls[0][1] as RequestInit).body as string)).toEqual(req);
  });
  it("исходное видео и копия для плеера — адреса /files/{pid}/source и /proxy", () => {
    const e = new HttpEngine("http://x", "t");
    expect(e.sourceVideoUrl("p")).toBe("http://x/files/p/source?token=t");
    expect(e.proxyVideo("p")!.url).toBe("http://x/files/p/proxy?token=t");
  });
  it("копия для плеера: 202 — готовится, 200/206 — готова, 404 — нет", async () => {
    const e = new HttpEngine("http://x");
    for (const [status, want] of [[202, "preparing"], [200, "ready"], [206, "ready"], [404, "missing"]] as const) {
      vi.stubGlobal("fetch", vi.fn(async () => new Response(status === 404 ? "{}" : "x", { status })));
      expect(await e.proxyVideo("p")!.status()).toBe(want);
    }
    vi.stubGlobal("fetch", vi.fn(async () => { throw new TypeError("fail"); }));
    expect(await e.proxyVideo("p")!.status()).toBe("preparing");
  });
  it("/probe: путь → {path}, ссылка → {url}; ответ known:false отдаётся как есть", async () => {
    const f = vi.fn(async (..._a: unknown[]) => json({ known: false, message: "Сети нет — узнаем после скачивания." }));
    vi.stubGlobal("fetch", f);
    const e = new HttpEngine("http://x");
    expect(await e.probe({ url: "https://y.tube/v" })).toMatchObject({ known: false });
    expect(JSON.parse((f.mock.calls[0][1] as RequestInit).body as string)).toEqual({ url: "https://y.tube/v" });
    await e.probe({ path: "file:///Users/me/a.mkv" });
    expect(JSON.parse((f.mock.calls[1][1] as RequestInit).body as string)).toEqual({ path: "file:///Users/me/a.mkv" });
  });
  it("ошибка движка: показываем detail, debug — никогда", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => json({ detail: "Ссылка недоступна.", debug: "yt_dlp.utils.DownloadError: secret trace" }, 422)));
    const err = await new HttpEngine("http://x").probe({ url: "https://bad" }).catch((e: Error) => e);
    expect((err as Error).message).toBe("Ссылка недоступна.");
    expect(String((err as Error).message)).not.toMatch(/yt_dlp|trace/);
  });
  it("в клиенте нет роутов, которых нет в движке", () => {
    const e = new HttpEngine("http://x") as unknown as Record<string, unknown>;
    expect(e.llm).toBeUndefined();
  });
});

describe("connectEngine", () => {
  it("без движка падает в демо", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => { throw new Error("down"); }));
    const r = await connectEngine({ mockFactory: async () => new MockEngine({ seed: false }) });
    expect(r.demo).toBe(true);
    expect(r.engine.kind).toBe("mock");
  });
  it("с живым /health берёт HttpEngine", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => json({ ok: true, version: "1", device: "cpu" })));
    const r = await connectEngine({ mockFactory: async () => new MockEngine({ seed: false }) });
    expect(r.demo).toBe(false);
    expect(r.engine.kind).toBe("http");
  });
  it("свой сервер: берёт адрес и токен из настроек", async () => {
    const f = vi.fn(async (..._a: unknown[]) => json({ ok: true, version: "1", device: "cuda" }));
    vi.stubGlobal("fetch", f);
    const r = await connectEngine({ mockFactory: async () => new MockEngine({ seed: false }), ui: { processing: "remote", remote_url: "http://srv:9000", remote_token: "t" } });
    expect(r.engine.base).toBe("http://srv:9000");
    expect(f.mock.calls[0][0]).toBe("http://srv:9000/health");
  });
});
