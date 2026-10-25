import { EngineError, type Engine } from "./engine";
import type {
  AppSettings, CheckResult, LatencyResult, MicCheckResult, CreateProjectOptions, DirectorCommandResult, DirectorTakeResult, FitLengthResult, Health, Job, KeyResult, LLMRequest, LinesMergeResult, MediaInfo, MergeResult, UrlProbe,
  NameSpeakersResult, Project, ProjectSummary, ProjectWithJob, ProxySource, ProxyState, RenderProjectRequest, Scene,
  ReanalyzeOptions, SceneCreate, SceneWithJob, ScenePatch, SnapRequest, SnapResult, Take, TakePatch, TakeUploadMeta, TranslateResult,
  UiSettings,
} from "./types";

/** Клиент к реальному движку: HTTP + WebSocket /ws/jobs. */
export class HttpEngine implements Engine {
  readonly kind = "http" as const;
  readonly base: string;
  private token: string | null;
  constructor(base: string, token: string | null = null) {
    this.base = base.replace(/\/+$/, "");
    this.token = token || null;
  }

  private headers(extra?: Record<string, string>): Record<string, string> {
    return { ...(this.token ? { Authorization: `Bearer ${this.token}` } : {}), ...extra };
  }

  private async raw(method: string, path: string, body?: unknown): Promise<Response> {
    let res: Response;
    const isForm = typeof FormData !== "undefined" && body instanceof FormData;
    try {
      res = await fetch(this.base + path, {
        method,
        headers: this.headers(body !== undefined && !isForm ? { "Content-Type": "application/json" } : undefined),
        body: body === undefined ? undefined : isForm ? (body as FormData) : JSON.stringify(body),
      });
    } catch {
      throw new EngineError("Движок не отвечает. Проверь, что он запущен.");
    }
    if (!res.ok) {
      let msg = `Ошибка движка (${res.status})`;
      try {
        const j = await res.json();
        if (typeof j.detail === "string") msg = j.detail;
        else if (Array.isArray(j.detail) && j.detail[0]?.msg) msg = String(j.detail[0].msg);
        else if (typeof j.message === "string") msg = j.message;
      } catch { /* не JSON */ }
      throw new EngineError(msg);
    }
    return res;
  }

  private async req<T>(method: string, path: string, body?: unknown): Promise<T> {
    const res = await this.raw(method, path, body);
    if (res.status === 204) return undefined as T;
    return (await res.json()) as T;
  }

  /** Найти задачу по проекту: заголовок X-Job-Id браузеру недоступен без expose_headers, берём из /jobs. */
  private async jobFor(res: Response, pid: string, kind: string, scid?: string): Promise<Job | null> {
    const id = res.headers.get("X-Job-Id");
    try {
      if (id) return await this.getJob(id);
      const all = await this.listJobs();
      const mine = all.filter((j) => j.project_id === pid && j.kind === kind && (!scid || j.scene_id === scid));
      return mine.length ? mine[mine.length - 1] : null;
    } catch {
      return null;
    }
  }

  health = () => this.req<Health>("GET", "/health");
  /** Что в файле (дорожки, субтитры, главы) до создания проекта. Ошибку (нет файла / не видео) показываем как есть. */
  probe = async (src: { path?: string; url?: string; name?: string }): Promise<MediaInfo | UrlProbe | null> =>
    src.path ? this.req<MediaInfo>("POST", "/probe", { path: src.path })
      : src.url ? this.req<UrlProbe>("POST", "/probe", { url: src.url }) // 422 — ссылка недоступна; known:false — проверить не удалось
        : null;
  listProjects = () => this.req<ProjectSummary[]>("GET", "/projects");

  async createProject(r: CreateProjectOptions): Promise<ProjectWithJob> {
    const res = await this.raw("POST", "/projects", {
      path: r.path ?? null, url: r.url ?? null, title: r.title ?? null,
      audio_stream_index: r.audio_stream_index ?? null, language: r.language ?? null,
    });
    const project = (await res.json()) as Project;
    return { project, job: await this.jobFor(res, project.id, "analyze") };
  }
  getProject = (id: string) => this.req<Project>("GET", `/projects/${id}`);
  deleteProject = async (id: string) => { await this.req("DELETE", `/projects/${id}`); };
  renameProject = (id: string, title: string) => this.req<Project>("PATCH", `/projects/${id}`, { title });
  reanalyze = (pid: string, opts?: ReanalyzeOptions) => this.req<Job>("POST", `/projects/${pid}/analyze`, opts);

  renameSpeaker = (pid: string, sid: string, name: string) => this.req<Project>("PATCH", `/projects/${pid}/speakers/${sid}`, { name });
  /** Движок отдаёт проект и merge_id: либо {project, merge_id}, либо проект с полем merge_id — понимаем оба вида. */
  mergeSpeakers = async (pid: string, a: string, b: string): Promise<MergeResult> => {
    const r = await this.req<Project & { merge_id?: string | null; project?: Project }>("POST", `/projects/${pid}/speakers/merge`, { a, b });
    return { project: r.project ?? r, merge_id: r.merge_id ?? null };
  };
  undoMerge = (pid: string, mergeId: string) => this.req<Project>("POST", `/projects/${pid}/speakers/merges/${encodeURIComponent(mergeId)}/undo`);
  setLineSpeaker = (pid: string, lineId: string, target: { speaker_id?: string; new_speaker_name?: string }) =>
    this.req<Project>("PATCH", `/projects/${pid}/lines/${encodeURIComponent(lineId)}/speaker`, target);
  mergeLines = async (pid: string, lineIds: string[]): Promise<LinesMergeResult> => {
    const r = await this.req<Project & { merge_id?: string | null; moved_takes?: number }>("POST", `/projects/${pid}/lines/merge`, { line_ids: lineIds });
    const { merge_id, moved_takes, ...project } = r;
    return { project: project as Project, merge_id: merge_id ?? null, moved_takes: moved_takes ?? 0 };
  };
  undoMergeLines = (pid: string, mergeId: string) => this.req<Project>("POST", `/projects/${pid}/lines/merges/${encodeURIComponent(mergeId)}/undo`);
  splitLine = (pid: string, lineId: string) => this.req<Project>("POST", `/projects/${pid}/lines/${encodeURIComponent(lineId)}/split`);
  snap = (pid: string, r: SnapRequest) => this.req<SnapResult>("POST", `/projects/${pid}/snap`, r);

  async createScene(pid: string, r: SceneCreate): Promise<SceneWithJob> {
    const res = await this.raw("POST", `/projects/${pid}/scenes`, r);
    const scene = (await res.json()) as Scene;
    return { scene, job: await this.jobFor(res, pid, "prepare_scene", scene.id) };
  }
  getScene = (pid: string, scid: string) => this.req<Scene>("GET", `/projects/${pid}/scenes/${scid}`);
  patchScene = (pid: string, scid: string, p: ScenePatch) => this.req<Scene>("PATCH", `/projects/${pid}/scenes/${scid}`, p);
  deleteScene = async (pid: string, scid: string) => { await this.req("DELETE", `/projects/${pid}/scenes/${scid}`); };
  prepareScene = (pid: string, scid: string) => this.req<Job>("POST", `/projects/${pid}/scenes/${scid}/prepare`);

  async uploadTake(pid: string, scid: string, wav: Blob, meta: TakeUploadMeta): Promise<Take> {
    const fd = new FormData();
    fd.append("file", wav, "take.wav");
    if (meta.line_id) fd.append("line_id", meta.line_id);
    fd.append("start_rel", String(meta.start_rel));
    fd.append("preroll", String(meta.preroll));
    if (meta.source) fd.append("source", meta.source);
    return this.req<Take>("POST", `/projects/${pid}/scenes/${scid}/takes`, fd);
  }
  patchTake = (pid: string, scid: string, tid: string, p: TakePatch) =>
    this.req<Take>("PATCH", `/projects/${pid}/scenes/${scid}/takes/${tid}`, p);
  // Роута DELETE для дублей в движке пока нет — при 404/405 пользователь увидит понятную ошибку.
  deleteTake = async (pid: string, scid: string, tid: string) => {
    try {
      await this.req("DELETE", `/projects/${pid}/scenes/${scid}/takes/${tid}`);
    } catch (e) {
      throw new EngineError(e instanceof EngineError && /404|405|Not Found|Method/i.test(e.message)
        ? "Движок пока не умеет удалять дубли." : (e as Error).message);
    }
  };

  untrimTakeTail = (pid: string, scid: string, tid: string) => this.req<Take>("POST", `/projects/${pid}/scenes/${scid}/takes/${tid}/untrim_tail`);
  micCheck = async (wav: Blob, silenceS: number): Promise<MicCheckResult> => {
    const fd = new FormData();
    fd.append("file", wav, "mic.wav");
    fd.append("silence_s", String(silenceS));
    fd.append("save", "true");
    return this.req<MicCheckResult>("POST", "/mic/check", fd);
  };
  micSignal = async (): Promise<ArrayBuffer> => (await this.raw("GET", "/mic/latency_signal.wav")).arrayBuffer();
  micLatency = async (wav: Blob, playOffsetS: number): Promise<LatencyResult> => {
    const fd = new FormData();
    fd.append("file", wav, "latency.wav");
    fd.append("play_offset_s", String(playOffsetS));
    fd.append("save", "true");
    return this.req<LatencyResult>("POST", "/mic/latency", fd);
  };

  trimTakeTail = async (pid: string, scid: string, tid: string): Promise<Take> => {
    try {
      return await this.req<Take>("POST", `/projects/${pid}/scenes/${scid}/takes/${tid}/trim_tail`);
    } catch (e) {
      throw new EngineError(e instanceof EngineError && /404|405|Not Found|Method/i.test(e.message)
        ? "Движок пока не умеет отрезать хвост. Запиши реплику заново." : (e as Error).message);
    }
  };

  preview = (pid: string, scid: string) => this.req<Job>("POST", `/projects/${pid}/scenes/${scid}/preview`);

  /** Рендер одной сцены или склейка нескольких: одна задача `render_project`, результат — `job.result.path` и `job.result.url`. */
  render = (pid: string, r: RenderProjectRequest) => this.req<Job>("POST", `/projects/${pid}/render`, r);

  private llmReq<T>(path: string, body: LLMRequest) { return this.req<T>("POST", `/llm/${path}`, body); }
  translate = (r: LLMRequest) => this.llmReq<TranslateResult>("translate", r);
  fitLength = (r: LLMRequest) => this.llmReq<FitLengthResult>("fit_length", r);
  nameSpeakers = (r: LLMRequest) => this.llmReq<NameSpeakersResult>("name_speakers", r);
  directorTake = (r: LLMRequest) => this.llmReq<DirectorTakeResult>("director", r);
  directorCommand = (r: LLMRequest) => this.llmReq<DirectorCommandResult>("director_command", r);

  getSettings = () => this.req<AppSettings>("GET", "/settings");
  putSettings = (s: AppSettings) => this.req<AppSettings>("PUT", "/settings", s);
  setKey = async (id: string, key: string): Promise<KeyResult> => {
    const r = await this.req<Partial<KeyResult> | null>("PUT", `/settings/llm/${id}/key`, { key });
    return { key_status: r?.key_status ?? "unknown", key_message: r?.key_message ?? null };
  };
  checkProvider = (id: string) => this.req<CheckResult>("POST", `/settings/llm/${id}/check`);

  listJobs = () => this.req<Job[]>("GET", "/jobs");
  getJob = (id: string) => this.req<Job>("GET", `/jobs/${id}`);
  cancelJob = async (id: string) => { await this.req("POST", `/jobs/${id}/cancel`); };

  subscribeJobs(cb: (job: Job) => void): () => void {
    let ws: WebSocket | null = null;
    let closed = false;
    let retry = 0;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const open = () => {
      if (closed) return;
      const url = this.base.replace(/^http/, "ws") + "/ws/jobs" + (this.token ? `?token=${encodeURIComponent(this.token)}` : "");
      ws = new WebSocket(url);
      ws.onmessage = (e) => {
        try { cb(JSON.parse(e.data as string) as Job); } catch { /* пропускаем мусор */ }
      };
      ws.onopen = () => { retry = 0; };
      ws.onclose = () => {
        if (closed) return;
        timer = setTimeout(open, Math.min(5000, 500 * 2 ** retry++));
      };
    };
    open();
    return () => { closed = true; clearTimeout(timer); ws?.close(); };
  }

  private file(pid: string, rel: string) {
    return `${this.base}/files/${pid}/${rel}${this.token ? `?token=${encodeURIComponent(this.token)}` : ""}`;
  }
  sourceVideoUrl = (pid: string) => this.file(pid, "source");
  /** Лёгкая копия: пока её готовят, движок отвечает 202 (+ X-Job-Id) и плеер показывает «Готовим видео…». */
  proxyVideo = (pid: string): ProxySource => ({
    url: this.file(pid, "proxy"),
    status: async (): Promise<ProxyState> => {
      try {
        const res = await fetch(this.file(pid, "proxy"), { method: "GET", headers: this.headers({ Range: "bytes=0-0" }) });
        void res.body?.cancel();
        if (res.status === 202) return "preparing";
        return res.ok ? "ready" : "missing";
      } catch {
        return "preparing"; // движок моргнул — спросим ещё раз
      }
    },
  });
  clipUrl = (pid: string, scid: string) => this.file(pid, `scenes/${scid}/clip.mp4`);
  previewUrl = (pid: string, scid: string) => this.file(pid, `scenes/${scid}/preview.mp4`);
  backgroundUrl = (pid: string, scid: string) => this.file(pid, `scenes/${scid}/background.wav`);
  takeUrl = (pid: string, scid: string, t: Take) => this.file(pid, `scenes/${scid}/${t.file}`);
  thumbnailUrl = (pid: string, rel: string | null | undefined) => (rel ? this.file(pid, rel) : null);
}

// ─────────────── выбор: движок или демо ───────────────
export type ConnectResult = { engine: Engine; demo: boolean; note?: string };

interface TauriWindow { __TAURI_INTERNALS__?: unknown }
export const inTauri = () => typeof window !== "undefined" && !!(window as unknown as TauriWindow).__TAURI_INTERNALS__;

export async function healthy(base: string, ms = 1200): Promise<boolean> {
  const ctl = new AbortController();
  const timer = setTimeout(() => ctl.abort(), ms);
  try {
    const r = await fetch(base.replace(/\/+$/, "") + "/health", { signal: ctl.signal });
    return r.ok && !!(await r.json()).ok;
  } catch {
    return false;
  } finally {
    clearTimeout(timer);
  }
}

/** Адрес движка от Tauri: команда engine_url, а если он ещё стартует — событие engine-ready. */
export async function tauriEngineUrl(timeoutMs = 25000): Promise<string | null> {
  const { invoke } = await import("@tauri-apps/api/core");
  const { listen } = await import("@tauri-apps/api/event");
  return new Promise<string | null>((resolve) => {
    let done = false;
    const unsub: (() => void)[] = [];
    const finish = (u: string | null) => {
      if (done) return;
      done = true;
      unsub.forEach((f) => f());
      resolve(u);
    };
    const timer = setTimeout(() => finish(null), timeoutMs);
    listen<string>("engine-ready", (e) => { clearTimeout(timer); finish(String(e.payload)); })
      .then((f) => { if (done) f(); else unsub.push(f); }).catch(() => {});
    listen("engine-failed", () => { clearTimeout(timer); finish(null); })
      .then((f) => { if (done) f(); else unsub.push(f); }).catch(() => {});
    invoke<string | null>("engine_url").then((u) => { if (u) { clearTimeout(timer); finish(u); } }).catch(() => {});
  });
}

export interface ConnectOptions {
  forceMock?: boolean;
  mockFactory: () => Promise<Engine>;
  ui?: Pick<UiSettings, "processing" | "remote_url" | "remote_token">;
}

export async function connectEngine(opts: ConnectOptions): Promise<ConnectResult> {
  const mock = async (note: string): Promise<ConnectResult> => ({ engine: await opts.mockFactory(), demo: true, note });
  if (opts.forceMock) return mock("");
  const ui = opts.ui;
  if (ui?.processing === "remote" && ui.remote_url) {
    if (await healthy(ui.remote_url, 3000)) return { engine: new HttpEngine(ui.remote_url, ui.remote_token), demo: false };
    return mock("Сервер не отвечает");
  }
  if (inTauri()) {
    const url = await tauriEngineUrl();
    if (url && (await healthy(url, 4000))) return { engine: new HttpEngine(url), demo: false };
    return mock("Движок не запустился");
  }
  const env = (import.meta.env.VITE_ENGINE_URL as string | undefined) || "http://127.0.0.1:8765";
  if (await healthy(env)) return { engine: new HttpEngine(env), demo: false };
  return mock("");
}
