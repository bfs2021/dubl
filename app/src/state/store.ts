import { create } from "zustand";
import { EngineError, type Engine } from "../api/engine";
import { HttpEngine, connectEngine, healthy, inTauri, tauriEngineUrl } from "../api/client";
import { cancelMlPack, initialMl, installMlPack, listenMlPack, mlPackSnapshot, reduceMl, type MlEvent, type MlProgress } from "../lib/mlpack";
import { remarkOf, type Severity } from "../lib/director";
import { humanizeError, isNoAiError } from "../lib/errors";
import { undoKey } from "../lib/format";
import type {
  AppSettings, CreateProjectOptions, Health, Job, KeyResult, MixSettings, Project, ProjectSummary, ReanalyzeOptions, RenderProjectRequest,
  Scene, ScenePatch, SceneTextMode, UiSettings,
} from "../api/types";
import { mergedLinesOf } from "../api/types";

// ───────────── настройки этого устройства ─────────────
const UI_KEY = "dubl.ui";

export const defaultUi = (): UiSettings => ({
  processing: "local", remote_url: "", remote_token: "", headphones: false, countdown: true, lead_s: 2, preroll_s: 1,
  original_level: 0.25, autofit: true, dismissed_merges: [], mic_profile: null,
});

export function loadUi(): UiSettings {
  try {
    const raw = localStorage.getItem(UI_KEY);
    return { ...defaultUi(), ...(raw ? JSON.parse(raw) : {}) };
  } catch {
    return defaultUi();
  }
}

function saveUi(u: UiSettings) {
  try { localStorage.setItem(UI_KEY, JSON.stringify(u)); } catch { /* приватный режим */ }
}

// ───────────── состояние ─────────────
export interface Remark { text: string; more?: string; severity: Severity; /** слова сценария, которых нет в записи */ missing?: string[] }
export interface ToastAction { label: string; run: () => void }
export interface Toast { id: number; text: string; kind: "info" | "error" | "ok"; action?: ToastAction; key?: string; onClose?: () => void }
/** Запрос подтверждения: окно рисует ConfirmHost (window.confirm в оболочке Tauri не работает). */
export interface ConfirmRequest { title: string; text?: string; confirmLabel: string; cancelLabel?: string; danger?: boolean; resolve: (ok: boolean) => void }
export type ConfirmOpts = Omit<ConfirmRequest, "resolve">;
export interface ToastOpts { key?: string; action?: ToastAction; /** сколько показывать, мс */ ms?: number; /** вызвать, когда плашка исчезла (любым способом) */ onClose?: () => void }

/** Сколько секунд даём передумать после удаления сцены или дубля. */
export const UNDO_DELETE_MS = 6000
/** Сколько висит плашка «Реплики склеены · Отменить». */
export const MERGE_TOAST_MS = 12000
/** Сколько правок реплик помним для ⌘Z (на проект). */
export const LINE_UNDO_MAX = 20

/** Правка реплики, которую можно отменить ⌘Z: склейка (по merge_id) или смена персонажа (вернуть прежнего). */
export type LineEdit =
  | { id: number; kind: "merge"; merge_id: string; line_id: string }
  | { id: number; kind: "speaker"; line_id: string; prev_speaker_id: string };

export interface AppState {
  engine: Engine | null;
  demo: boolean;
  demoNote: string;
  connecting: boolean;
  health: Health | null;
  ui: UiSettings;
  settings: AppSettings | null;
  projects: ProjectSummary[];
  jobs: Record<string, Job>;
  project: Project | null;
  sceneId: string | null;
  /** пока показана плашка «Голоса объединены · Отменить», новых подсказок про слияние не показываем */
  mergeHold: boolean;
  /** замечания помощника по дублям: ключ pid/scid/tid */
  remarks: Record<string, Remark>;
  /** тексты «Как есть» (оригинал или перевод) на случай возврата из «Своего текста» */
  asIs: Record<string, Record<string, string>>;
  toasts: Toast[];
  /** открытое окно подтверждения (одно за раз) */
  dialog: ConfirmRequest | null;
  /** куда вернёт «← Назад» из Настроек: адрес экрана, с которого пришли */
  backTo: string;
  /** установка нейросетей (Tauri): этап, деталь, ошибка */
  ml: MlProgress;

  // подключение
  connect: (opts?: { forceMock?: boolean; mockFactory?: () => Promise<Engine> }) => Promise<void>;
  attachEngine: (engine: Engine, demo: boolean, note?: string) => Promise<void>;
  // общее
  toast: (text: string, kind?: Toast["kind"], opts?: ToastOpts) => void;
  dismissToast: (id: number) => void;
  /** Спросить «точно?» своим окном. Резолвится true по кнопке подтверждения, false — по «Отмена», Esc и клику мимо. */
  askConfirm: (opts: ConfirmOpts) => Promise<boolean>;
  resolveConfirm: (ok: boolean) => void;
  setBackTo: (path: string) => void;
  setUi: (patch: Partial<UiSettings>) => void;
  // нейросети
  initMl: () => Promise<void>;
  startMlInstall: () => Promise<void>;
  cancelMlInstall: () => Promise<void>;
  /** в тестах и при ручном управлении: применить событие установщика */
  applyMlEvent: (ev: MlEvent) => void;
  // проекты
  loadProjects: () => Promise<void>;
  openProject: (id: string) => Promise<void>;
  refreshProject: () => Promise<void>;
  closeProject: () => void;
  createProject: (req: CreateProjectOptions) => Promise<Project | null>;
  deleteProject: (id: string) => Promise<void>;
  /** Разобрать проект заново (после ошибки или установки нейросетей). */
  reanalyze: (id: string, opts?: ReanalyzeOptions) => Promise<Job | null>;
  renameSpeaker: (sid: string, name: string) => Promise<void>;
  mergeSpeakers: (a: string, b: string) => Promise<void>;
  /** Сменить персонажа у реплики; newName — завести нового. */
  setLineSpeaker: (lineId: string, target: { speaker_id?: string; new_speaker_name?: string }) => Promise<void>;
  /** Склеить соседние реплики в одну (с отменой из плашки). */
  mergeLines: (lineIds: string[]) => Promise<void>;
  /** «Разделить обратно»: отменить последнюю склейку, в которую вошла реплика. */
  splitLine: (lineId: string) => Promise<void>;
  /** Последние правки реплик по проектам (для ⌘Z), новые — в конце. */
  lineUndo: Record<string, LineEdit[]>;
  /** Отменить правку реплик: последнюю (⌘Z) или конкретную (кнопка «Отменить» в плашке). false — отменять нечего. */
  undoLineEdit: (id?: number) => Promise<boolean>;
  renameProject: (id: string, title: string) => Promise<void>;
  guessNames: () => Promise<void>;
  dismissMerge: (a: string, b: string) => void;
  // сцены
  setSceneId: (id: string | null) => void;
  createScene: (start: number, end: number, title: string) => Promise<Scene | null>;
  deleteScene: (id: string) => Promise<void>;
  /** Подготовить сцену заново (после прерванной подготовки голосов и фона). */
  prepareScene: (id: string) => Promise<void>;
  patchScene: (patch: ScenePatch, sceneId?: string) => Promise<Scene | null>;
  setTextMode: (mode: SceneTextMode) => Promise<void>;
  setLineText: (lineId: string, text: string) => Promise<void>;
  translateScene: () => Promise<void>;
  /** null — не вышло (ошибку уже показали); [] — ИИ ответил, но вариантов нет */
  fitLine: (lineId: string) => Promise<string[] | null>;
  setMix: (mix: MixSettings) => Promise<void>;
  // дубли
  /** onError — показать ошибку самому (в экране записи), тост в этом случае не нужен */
  uploadTake: (wav: Blob, meta: { line_id: string | null; start_rel: number; preroll: number; source?: "mic" | "file" }, onError?: (message: string) => void) => Promise<string | null>;
  selectTake: (tid: string, selected: boolean) => Promise<void>;
  patchTake: (tid: string, patch: { offset_ms?: number; stretch?: number }) => Promise<void>;
  deleteTake: (tid: string) => Promise<void>;
  /** Отрезать повтор в конце дубля («давай ещё раз»). */
  trimTail: (tid: string) => Promise<void>;
  /** Вернуть отрезанный хвост. */
  untrimTail: (tid: string) => Promise<void>;
  loadRemark: (tid: string, force?: boolean) => Promise<void>;
  // задачи
  trackJob: (job: Job | null | undefined) => void;
  startPreview: () => Promise<Job | null>;
  startRender: (req: RenderProjectRequest) => Promise<Job | null>;
  cancelJob: (id: string) => Promise<void>;
  // настройки
  saveSettings: (s: AppSettings) => Promise<void>;
  setKey: (providerId: string, key: string) => Promise<KeyResult | null>;
  refreshSettings: () => Promise<void>;
}

let toastSeq = 1;
let editSeq = 1;
type DistributiveOmit<T, K extends PropertyKey> = T extends unknown ? Omit<T, K> : never;
let unsub: (() => void) | null = null;
/** задачи, которые мы видели идущими: об окончании сообщаем только о них (не о старых, пришедших при подключении) */
const watched = new Set<string>();
const notified = new Set<string>();
let lastRender: RenderProjectRequest | null = null;
let mlListening = false;
/** Удалено в интерфейсе, но ещё не в движке (6 секунд на «Отменить»): сцены `pid/scid`, дубли `pid/scid/tid`. */
const hidden = new Set<string>();
const hideDeleted = (p: Project): Project => {
  if (!hidden.size) return p;
  return {
    ...p,
    scenes: p.scenes.filter((sc) => !hidden.has(`${p.id}/${sc.id}`)).map((sc) => (sc.takes.some((t) => hidden.has(`${p.id}/${sc.id}/${t.id}`))
      ? { ...sc, takes: sc.takes.filter((t) => !hidden.has(`${p.id}/${sc.id}/${t.id}`)) } : sc)),
  };
};

/** Нейросети поставлены, Rust перезапускает движок: ждём новый адрес (engine-ready) и переподключаемся. */
async function reconnectAfterMl() {
  const set = useStore.setState;
  const url = await tauriEngineUrl(90000);
  if (url && (await healthy(url, 8000))) {
    await useStore.getState().attachEngine(new HttpEngine(url), false);
    set((s) => ({ ml: { ...s.ml, phase: "idle", installed: true } }));
    return;
  }
  set((s) => ({ ml: { ...s.ml, phase: "failed", error: "Движок не перезапустился после установки. Закрой и открой программу." } }));
}

const msgOf = (e: unknown) => (e instanceof EngineError || e instanceof Error ? e.message : "Что-то пошло не так");

export const useStore = create<AppState>()((set, get) => {
  /** Выполнить действие; ошибку показать плашкой (по-человечески, с «Повторить», если есть что повторять). */
  async function guard<T>(fn: () => Promise<T>, fallback: T, retry?: () => void): Promise<T> {
    try {
      return await fn();
    } catch (e) {
      const raw = msgOf(e);
      if (isNoAiError(raw)) {
        // статус ключей мог устареть — спросим у движка заново
        void get().refreshSettings();
        get().toast("ИИ не подключён. Подключи модель в Настройках — без неё эта кнопка не работает.", "error", {
          key: "no-ai", action: { label: "Настройки", run: () => { location.hash = "#/settings/ai"; } },
        });
      } else {
        get().toast(raw, "error", retry ? { action: { label: "Повторить", run: retry } } : undefined);
      }
      return fallback;
    }
  }
  const eng = (): Engine => {
    const e = get().engine;
    if (!e) throw new EngineError("Движок ещё не подключён.");
    return e;
  };
  const ctx = () => {
    const { engine, project, sceneId } = get();
    const scene = project?.scenes.find((s) => s.id === sceneId) ?? null;
    if (!engine || !project) throw new EngineError("Сначала открой проект.");
    return { engine, project, scene };
  };
  const replaceScene = (sc: Scene) =>
    set((s) => (s.project ? { project: hideDeleted({ ...s.project, scenes: s.project.scenes.map((x) => (x.id === sc.id ? sc : x)) }) } : {}));

  return {
    engine: null, demo: false, demoNote: "", connecting: true, health: null, ui: loadUi(), settings: null, projects: [], jobs: {},
    project: null, sceneId: null, remarks: {}, asIs: {}, toasts: [], dialog: null, backTo: "/", mergeHold: false, ml: initialMl(), lineUndo: {},

    // ───────────── нейросети ─────────────
    async initMl() {
      if (!inTauri() || mlListening) return;
      mlListening = true;
      try {
        await listenMlPack((ev) => get().applyMlEvent(ev));
        // окно могли перезагрузить посреди установки
        const snap = await mlPackSnapshot();
        if (snap?.running && snap.last) get().applyMlEvent(snap.last);
      } catch {
        mlListening = false;
      }
    },
    applyMlEvent(ev) {
      set((s) => ({ ml: reduceMl(s.ml, ev) }));
      if (ev.kind === "done") void reconnectAfterMl();
    },
    async startMlInstall() {
      if (!inTauri()) return;
      set((s) => ({ ml: { ...initialMl(), phase: "running", installed: s.ml.installed } }));
      try {
        await get().initMl();
        await installMlPack();
      } catch (e) {
        set((s) => ({ ml: { ...s.ml, phase: "failed", error: msgOf(e) } }));
      }
    },
    async cancelMlInstall() {
      try { await cancelMlPack(); } catch { /* уже закончилась */ }
    },

    // ───────────── подключение ─────────────
    async connect(opts = {}) {
      set({ connecting: true });
      const mockFactory = opts.mockFactory ?? (async () => new (await import("../api/mock")).MockEngine());
      const r = await connectEngine({ forceMock: opts.forceMock, mockFactory, ui: get().ui });
      await get().attachEngine(r.engine, r.demo, r.note);
    },

    async attachEngine(engine, demo, note = "") {
      unsub?.();
      set({ engine, demo, demoNote: note, connecting: false, projects: [], jobs: {}, project: null, sceneId: null, remarks: {}, mergeHold: false, lineUndo: {} });
      try { set({ health: await engine.health() }); } catch { /* демо без health */ }
      watched.clear();
      notified.clear();
      unsub = engine.subscribeJobs((job) => {
        set((s) => ({ jobs: { ...s.jobs, [job.id]: job } }));
        if (job.state === "queued" || job.state === "running") watched.add(job.id);
        else if (watched.has(job.id) && !notified.has(job.id)) { notified.add(job.id); void onJobEnd(job); }
      });
      try {
        const jobs = await engine.listJobs();
        set((s) => ({ jobs: { ...s.jobs, ...Object.fromEntries(jobs.map((j) => [j.id, j])) } }));
      } catch { /* нет списка задач */ }
      try { set({ settings: await engine.getSettings() }); } catch (e) { get().toast(msgOf(e), "error"); }
      await get().loadProjects();
    },

    toast(text, kind = "info", opts) {
      const shown = kind === "error" ? humanizeError(text) : text;
      const dup = get().toasts.find((t) => (opts?.key ? t.key === opts.key : t.text === shown && t.kind === kind));
      if (dup) return; // такой же показан прямо сейчас — не дублируем
      const id = toastSeq++;
      const all = [...get().toasts, { id, text: shown, kind, action: opts?.action, key: opts?.key, onClose: opts?.onClose }];
      const dropped = all.slice(0, -4);
      set({ toasts: all.slice(-4) });
      dropped.forEach((t) => t.onClose?.());
      setTimeout(() => get().dismissToast(id), opts?.ms ?? (opts?.action ? 10000 : kind === "error" ? 7000 : 3200));
    },
    dismissToast: (id) => {
      const t = get().toasts.find((x) => x.id === id);
      if (!t) return;
      set((s) => ({ toasts: s.toasts.filter((x) => x.id !== id) }));
      t.onClose?.();
    },
    askConfirm(opts) {
      return new Promise<boolean>((resolve) => {
        get().dialog?.resolve(false); // старое окно закрываем отказом — одновременно показываем одно
        set({ dialog: { ...opts, resolve } });
      });
    },
    resolveConfirm(ok) {
      const d = get().dialog;
      if (!d) return;
      set({ dialog: null });
      d.resolve(ok);
    },
    setBackTo: (path) => set({ backTo: path }),
    setUi(patch) {
      const ui = { ...get().ui, ...patch };
      saveUi(ui);
      set({ ui });
    },

    // ───────────── проекты ─────────────
    async loadProjects() {
      const e = get().engine;
      if (!e) return;
      await guard(async () => set({ projects: await e.listProjects() }), undefined, () => void get().loadProjects());
    },
    async openProject(id) {
      const e = eng();
      if (get().project?.id === id) return;
      set({ project: null, sceneId: null });
      const p = await guard(() => e.getProject(id), null, () => void get().openProject(id));
      if (!p) return;
      const shown = hideDeleted(p);
      set({ project: shown, sceneId: shown.scenes[0]?.id ?? null });
    },
    async refreshProject() {
      const { engine, project } = get();
      if (!engine || !project) return;
      const p = await guard(() => engine.getProject(project.id), null);
      if (!p) return;
      const shown = hideDeleted(p);
      set((s) => ({ project: shown, sceneId: shown.scenes.some((x) => x.id === s.sceneId) ? s.sceneId : shown.scenes[0]?.id ?? null }));
    },
    closeProject: () => set({ project: null, sceneId: null }),

    async createProject(req) {
      const e = eng();
      return guard(async () => {
        const { project, job } = await e.createProject(req);
        get().trackJob(job);
        await get().loadProjects();
        return project;
      }, null, () => void get().createProject(req));
    },
    async deleteProject(id) {
      const e = eng();
      await guard(async () => {
        await e.deleteProject(id);
        if (get().project?.id === id) get().closeProject();
        await get().loadProjects();
        get().toast("Проект удалён. Видео на диске не тронуто.");
      }, undefined);
    },
    async reanalyze(id, opts) {
      const e = eng();
      return guard(async () => {
        const job = await (opts ? e.reanalyze(id, opts) : e.reanalyze(id));
        get().trackJob(job);
        return job;
      }, null as Job | null, () => void get().reanalyze(id, opts));
    },
    async renameProject(id, title) {
      const e = eng();
      const t = title.trim();
      if (!t) return;
      await guard(async () => {
        const p = await e.renameProject(id, t);
        if (get().project?.id === id) set((s) => (s.project ? { project: { ...s.project, title: p.title } } : {}));
        await get().loadProjects();
      }, undefined);
    },
    async renameSpeaker(sid, name) {
      const { engine, project } = ctx();
      await guard(async () => set({ project: hideDeleted(await engine.renameSpeaker(project.id, sid, name)) }), undefined);
    },
    async mergeSpeakers(a, b) {
      const { engine, project } = ctx();
      await guard(async () => {
        const nameOf = (id: string) => { const sp = project.analysis?.speakers.find((x) => x.id === id); return sp ? (sp.name ?? sp.label) : id; };
        const r = await engine.mergeSpeakers(project.id, a, b);
        set({ project: hideDeleted(r.project), mergeHold: true });
        const release = () => set({ mergeHold: false });
        get().toast(`Голоса объединены: «${nameOf(b)}» теперь часть «${nameOf(a)}».`, "ok", {
          key: "merge", ms: 8000, onClose: release,
          action: r.merge_id ? {
            label: "Отменить",
            run: () => {
              release();
              void guard(async () => {
                set({ project: hideDeleted(await engine.undoMerge(project.id, r.merge_id!)) });
                get().toast("Слияние отменено.", "info");
              }, undefined);
            },
          } : undefined,
        });
      }, undefined);
    },
    async guessNames() {
      const { engine, project } = ctx();
      await guard(async () => {
        const { names } = await engine.nameSpeakers({ project_id: project.id });
        const ids = Object.keys(names);
        if (!ids.length) { get().toast("ИИ не нашёл подсказок для имён."); return; }
        for (const sid of ids) await get().renameSpeaker(sid, names[sid]);
        get().toast(`Подписал голосов: ${ids.length}. Проверь и поправь.`, "ok");
      }, undefined);
    },
    async setLineSpeaker(lineId, target) {
      const { engine, project } = ctx();
      const prev = project.analysis?.lines.find((l) => l.id === lineId)?.speaker_id
        ?? project.scenes.flatMap((sc) => sc.lines).find((l) => l.line_id === lineId)?.speaker_id ?? null;
      await guard(async () => {
        const p = await engine.setLineSpeaker(project.id, lineId, target);
        set({ project: hideDeleted(p) });
        const now = p.analysis?.lines.find((l) => l.id === lineId)?.speaker_id ?? null;
        const sp = p.analysis?.speakers.find((x) => x.id === now);
        // вернуть прежнего персонажа можно ⌘Z (нового, если его завели, не удаляем — достаточно вернуть реплику)
        const edit = prev && now !== prev ? pushEdit(project.id, { kind: "speaker", line_id: lineId, prev_speaker_id: prev }) : null;
        get().toast((sp ? `Реплика теперь у «${sp.name ?? sp.label}».` : "Персонаж реплики изменён.") + (edit ? ` ${undoKey()} — отменить.` : ""), "ok", {
          key: "line-speaker",
          action: edit ? { label: "Отменить", run: () => void get().undoLineEdit(edit.id) } : undefined,
        });
      }, undefined);
    },
    async mergeLines(lineIds) {
      const { engine, project } = ctx();
      await guard(async () => {
        const r = await engine.mergeLines(project.id, lineIds);
        set({ project: hideDeleted(r.project) });
        const edit = r.merge_id ? pushEdit(project.id, { kind: "merge", merge_id: r.merge_id, line_id: lineIds[0] }) : null;
        const takes = r.moved_takes ? ` Дубли второй части теперь у неё — лучше перезаписать реплику целиком.` : "";
        get().toast(`Реплики склеены в одну.${takes}${edit ? ` ${undoKey()} — отменить.` : ""}`, "ok", {
          key: "merge-lines", ms: MERGE_TOAST_MS,
          action: edit ? { label: "Отменить", run: () => void get().undoLineEdit(edit.id) } : undefined,
        });
      }, undefined);
    },
    async splitLine(lineId) {
      const { engine, project } = ctx();
      const mid = mergedLinesOf(project.analysis)[lineId];
      await guard(async () => {
        set({ project: hideDeleted(await engine.splitLine(project.id, lineId)) });
        // эта склейка отменена — из стека ⌘Z её убираем, плашку «Реплики склеены · Отменить» тоже
        if (mid) dropEdits(project.id, (e) => e.kind === "merge" && e.merge_id === mid);
        get().toasts.filter((t) => t.key === "merge-lines").forEach((t) => get().dismissToast(t.id));
        get().toast("Реплика разделена обратно.", "info", { key: "split-line" });
      }, undefined);
    },
    async undoLineEdit(id) {
      const { engine, project } = ctx();
      const stack = get().lineUndo[project.id] ?? [];
      const edit = id == null ? stack[stack.length - 1] : stack.find((e) => e.id === id);
      if (!edit) return false;
      dropEdits(project.id, (e) => e.id === edit.id);
      // плашка про эту правку и прошлое «Отменено: …» больше не нужны
      get().toasts.filter((t) => t.key === "line-undo" || t.key === (edit.kind === "merge" ? "merge-lines" : "line-speaker")).forEach((t) => get().dismissToast(t.id));
      return guard(async () => {
        if (edit.kind === "merge") {
          set({ project: hideDeleted(await engine.undoMergeLines(project.id, edit.merge_id)) });
          get().toast("Отменено: склейка реплик.", "info", { key: "line-undo" });
        } else {
          set({ project: hideDeleted(await engine.setLineSpeaker(project.id, edit.line_id, { speaker_id: edit.prev_speaker_id })) });
          get().toast("Отменено: смена персонажа реплики.", "info", { key: "line-undo" });
        }
        return true;
      }, false);
    },
    dismissMerge(a, b) {
      // в движке «это разные» не хранится — помним на этом устройстве, отдельно по проектам
      const pid = get().project?.id ?? "";
      get().setUi({ dismissed_merges: [...get().ui.dismissed_merges, `${pid}|${a}|${b}`] });
    },

    // ───────────── сцены ─────────────
    setSceneId: (id) => set({ sceneId: id }),
    async createScene(start, end, title) {
      const { engine, project } = ctx();
      return guard(async () => {
        const { scene, job } = await engine.createScene(project.id, { start, end, title });
        get().trackJob(job);
        set((s) => (s.project ? { project: { ...s.project, scenes: [...s.project.scenes, scene] }, sceneId: scene.id } : {}));
        get().toast(`Сцена «${scene.title}» создана. Готовлю голоса и фон…`, "ok", { key: `scene-created-${scene.id}` });
        return scene;
      }, null);
    },
    async deleteScene(id) {
      const { engine, project } = ctx();
      const sc = project.scenes.find((x) => x.id === id);
      if (!sc) return;
      deferDelete(`${project.id}/${id}`, `Сцена «${sc.title}» удалена.`, () => engine.deleteScene(project.id, id));
    },
    async prepareScene(id) {
      const { engine, project } = ctx();
      await guard(async () => {
        const job = await engine.prepareScene(project.id, id);
        get().trackJob(job);
        set((s) => (s.project ? { project: { ...s.project, scenes: s.project.scenes.map((x) => (x.id === id ? { ...x, stems_ready: false, prepare_state: "running" as const } : x)) } } : {}));
      }, undefined);
    },
    async patchScene(patch, sceneId) {
      const { engine, project, scene } = ctx();
      const id = sceneId ?? scene?.id;
      if (!id) return null;
      return guard(async () => {
        const sc = await engine.patchScene(project.id, id, patch);
        replaceScene(sc);
        return sc;
      }, null);
    },
    async setTextMode(mode) {
      const { scene } = ctx();
      if (!scene || scene.text_mode === mode) return;
      if (mode === "as_is") {
        const cache = get().asIs[`${get().project!.id}/${scene.id}`] ?? {};
        await get().patchScene({ text_mode: mode, lines: scene.lines.map((l) => ({ line_id: l.line_id, text: cache[l.line_id] ?? l.original })) });
      } else {
        // запоминаем текст «Как есть» (оригинал или перевод), чтобы вернуть при переключении назад
        const key = `${get().project!.id}/${scene.id}`;
        set((s) => ({ asIs: { ...s.asIs, [key]: { ...Object.fromEntries(scene.lines.map((l) => [l.line_id, l.text])), ...(s.asIs[key] ?? {}) } } }));
        await get().patchScene({ text_mode: mode });
      }
    },
    async setLineText(lineId, text) {
      await get().patchScene({ lines: [{ line_id: lineId, text }] });
    },
    async translateScene() {
      const { engine, project, scene } = ctx();
      if (!scene) return;
      await guard(async () => {
        const { translations } = await engine.translate({ project_id: project.id, scene_id: scene.id, target_lang: "ru" });
        const texts = Object.fromEntries(translations.map((t) => [t.line_id, t.text]));
        const key = `${project.id}/${scene.id}`;
        set((s) => ({ asIs: { ...s.asIs, [key]: { ...(s.asIs[key] ?? {}), ...texts } } }));
        await get().patchScene({ lines: Object.entries(texts).map(([line_id, text]) => ({ line_id, text })) });
        get().toast("Готово — проверь перевод.", "ok");
      }, undefined, () => void get().translateScene());
    },
    async fitLine(lineId) {
      const { engine, project, scene } = ctx();
      if (!scene) return [];
      return guard(async () => (await engine.fitLength({ project_id: project.id, scene_id: scene.id, line_ids: [lineId], n: 3 })).variants.find((v) => v.line_id === lineId)?.texts ?? [], null as string[] | null);
    },
    async setMix(mix) {
      await get().patchScene({ mix });
    },

    // ───────────── дубли ─────────────
    async uploadTake(wav, meta, onError) {
      const { engine, project, scene } = ctx();
      if (!scene) return null;
      if (onError) {
        try {
          const take = await engine.uploadTake(project.id, scene.id, wav, meta);
          replaceScene(await engine.getScene(project.id, scene.id));
          void get().loadRemark(take.id, true);
          return take.id;
        } catch (e) {
          onError(humanizeError(msgOf(e)));
          return null;
        }
      }
      return guard(async () => {
        const take = await engine.uploadTake(project.id, scene.id, wav, meta);
        const sc = await engine.getScene(project.id, scene.id);
        replaceScene(sc);
        void get().loadRemark(take.id, true);
        return take.id;
      }, null, () => void get().uploadTake(wav, meta));
    },
    async selectTake(tid, selected) {
      const { engine, project, scene } = ctx();
      if (!scene) return;
      await guard(async () => {
        await engine.patchTake(project.id, scene.id, tid, { selected });
        replaceScene(await engine.getScene(project.id, scene.id));
      }, undefined);
    },
    async patchTake(tid, patch) {
      const { engine, project, scene } = ctx();
      if (!scene) return;
      await guard(async () => {
        await engine.patchTake(project.id, scene.id, tid, patch);
        replaceScene(await engine.getScene(project.id, scene.id));
      }, undefined);
    },
    async deleteTake(tid) {
      const { engine, project, scene } = ctx();
      if (!scene) return;
      deferDelete(`${project.id}/${scene.id}/${tid}`, "Дубль удалён.", () => engine.deleteTake(project.id, scene.id, tid));
    },
    async trimTail(tid) {
      const { engine, project, scene } = ctx();
      if (!scene) return;
      await guard(async () => {
        await engine.trimTakeTail(project.id, scene.id, tid);
        replaceScene(await engine.getScene(project.id, scene.id));
        void get().loadRemark(tid, true);
        get().toast("Повтор в конце отрезан.", "ok", { key: `trim-${tid}`, action: { label: "Вернуть", run: () => void get().untrimTail(tid) } });
      }, undefined);
    },
    async untrimTail(tid) {
      const { engine, project, scene } = ctx();
      if (!scene) return;
      await guard(async () => {
        await engine.untrimTakeTail(project.id, scene.id, tid);
        replaceScene(await engine.getScene(project.id, scene.id));
        void get().loadRemark(tid, true);
      }, undefined);
    },
    async loadRemark(tid, force = false) {
      const { engine, project, scene } = ctx();
      if (!scene) return;
      const key = `${project.id}/${scene.id}/${tid}`;
      if (!force && get().remarks[key]) return;
      try {
        const r = await engine.directorTake({ project_id: project.id, scene_id: scene.id, take_id: tid });
        set((s) => ({ remarks: { ...s.remarks, [key]: { ...remarkOf(r), missing: r.report.missing_words ?? [] } } }));
      } catch { /* замечание — не критично */ }
    },

    // ───────────── задачи ─────────────
    trackJob(job) {
      if (!job) return;
      if (job.state === "queued" || job.state === "running") watched.add(job.id);
      set((s) => ({ jobs: { ...s.jobs, [job.id]: job } }));
    },
    async startPreview() {
      const { engine, project, scene } = ctx();
      if (!scene) return null;
      return guard(async () => {
        const job = await engine.preview(project.id, scene.id);
        get().trackJob(job);
        return job;
      }, null);
    },
    async startRender(req) {
      const { engine, project } = ctx();
      lastRender = req;
      return guard(async () => {
        const job = await engine.render(project.id, req);
        get().trackJob(job);
        return job;
      }, null as Job | null, () => void get().startRender(req));
    },
    async cancelJob(id) {
      const e = eng();
      await guard(() => e.cancelJob(id), undefined);
    },

    // ───────────── настройки ─────────────
    async saveSettings(s) {
      const e = eng();
      // key_set / key_status / key_message считает движок — обратно их не шлём
      const clean = { ...s, llm: { ...s.llm, providers: s.llm.providers.map(({ key_set: _a, key_status: _b, key_message: _c, ...p }) => p) } } as AppSettings;
      await guard(async () => { set({ settings: await e.putSettings(clean) }); }, undefined);
      await get().refreshSettings();
    },
    async setKey(providerId, key) {
      const e = eng();
      const r = await guard(() => e.setKey(providerId, key), null as KeyResult | null);
      await get().refreshSettings();
      return r;
    },
    async refreshSettings() {
      const e = get().engine;
      if (!e) return;
      try { set({ settings: await e.getSettings() }); } catch { /* оставляем прежние */ }
    },
  };

  /** Запомнить правку реплик для ⌘Z (не больше LINE_UNDO_MAX на проект). */
  function pushEdit(pid: string, e: DistributiveOmit<LineEdit, "id">): LineEdit {
    const edit = { ...e, id: editSeq++ } as LineEdit;
    set((s) => ({ lineUndo: { ...s.lineUndo, [pid]: [...(s.lineUndo[pid] ?? []), edit].slice(-LINE_UNDO_MAX) } }));
    return edit;
  }
  function dropEdits(pid: string, pred: (e: LineEdit) => boolean) {
    set((s) => ({ lineUndo: { ...s.lineUndo, [pid]: (s.lineUndo[pid] ?? []).filter((e) => !pred(e)) } }));
  }

  /** Убрать из списка сразу, а в движке удалить через 6 секунд — если не нажали «Отменить». */
  function deferDelete(key: string, text: string, commit: () => Promise<void>) {
    hidden.add(key);
    const pid = get().project?.id;
    set((s) => (s.project ? { project: hideDeleted(s.project) } : {}));
    const sceneKey = key.split("/").length === 2;
    if (sceneKey) set((s) => (s.project && !s.project.scenes.some((x) => x.id === s.sceneId) ? { sceneId: s.project.scenes[0]?.id ?? null } : {}));
    let undone = false;
    const timer = setTimeout(async () => {
      if (undone) return;
      try {
        await commit();
      } catch (e) {
        // не вышло — возвращаем в список и говорим об этом
        hidden.delete(key);
        get().toast(msgOf(e), "error");
      }
      hidden.delete(key);
      if (get().project?.id === pid) await get().refreshProject();
    }, UNDO_DELETE_MS);
    get().toast(text, "info", {
      key: `del-${key}`, ms: UNDO_DELETE_MS,
      action: {
        label: "Отменить",
        run: () => {
          undone = true;
          clearTimeout(timer);
          hidden.delete(key);
          if (get().project?.id === pid) void get().refreshProject();
        },
      },
    });
  }

  async function onJobEnd(job: Job) {
    const s = get();
    if (job.state === "failed") {
      const req = job.kind === "render_project" ? lastRender : null;
      s.toast(job.error || "Задача не удалась.", "error", { key: `job-${job.id}`, action: req ? { label: "Повторить", run: () => void get().startRender(req) } : undefined });
      return;
    }
    if (job.state !== "done") return;
    const title = s.project?.scenes.find((x) => x.id === job.scene_id)?.title;
    if (job.kind === "analyze") {
      await s.loadProjects();
      if (s.project?.id === job.project_id) await s.refreshProject();
      s.toast("Разбор готов — можно нарезать сцены.", "ok", { key: `job-${job.id}` });
    } else if (job.kind === "prepare_scene") {
      if (s.project?.id === job.project_id) await s.refreshProject();
      s.toast(title ? `Голоса и фон готовы: «${title}»` : "Голоса и фон готовы.", "ok", { key: `job-${job.id}` });
    } else if (job.kind === "render_project" || job.kind === "render") {
      s.toast("Ролик готов.", "ok", { key: `job-${job.id}` });
    }
  }
});

/** Выбранная сцена текущего проекта. */
export function useScene(): Scene | null {
  return useStore((s) => s.project?.scenes.find((x) => x.id === s.sceneId) ?? null);
}

/** Активная (идущая) задача проекта/сцены. */
export function activeJob(jobs: Record<string, Job>, kind: string, pid: string | undefined, scid?: string | null): Job | null {
  if (!pid) return null;
  const list = Object.values(jobs).filter((j) => j.kind === kind && j.project_id === pid && (!scid || j.scene_id === scid));
  return list.length ? list[list.length - 1] : null;
}

/** Общий прогресс задачи 0..1 — считает движок (монотонный). */
export function jobProgress(j: Job | null): number {
  if (!j) return 0;
  if (j.state === "done") return 1;
  return Math.min(1, Math.max(0, j.progress));
}
