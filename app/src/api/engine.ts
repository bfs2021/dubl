import type {
  AppSettings, CheckResult, LatencyResult, MicCheckResult, CreateProjectOptions, DirectorCommandResult, DirectorTakeResult, FitLengthResult, Health, Job,
  KeyResult, LLMRequest, LinesMergeResult, MediaInfo, MergeResult, UrlProbe, NameSpeakersResult, Project, ProjectSummary, ProjectWithJob, ProxySource, RenderProjectRequest, Scene,
  ReanalyzeOptions, SceneCreate, SceneWithJob, ScenePatch, SnapRequest, SnapResult, Take, TakePatch, TakeUploadMeta, TranslateResult,
} from "./types";

/** Всё, что UI умеет просить у движка. Реализации: HttpEngine (client.ts) и MockEngine (mock.ts). */
export interface Engine {
  readonly kind: "http" | "mock";
  readonly base: string;
  health(): Promise<Health>;

  /** Что найдено в файле (дорожки, субтитры, главы) до создания проекта. null — движок так не умеет. */
  probe(src: { path?: string; url?: string; name?: string }): Promise<MediaInfo | UrlProbe | null>;
  listProjects(): Promise<ProjectSummary[]>;
  createProject(req: CreateProjectOptions): Promise<ProjectWithJob>;
  getProject(id: string): Promise<Project>;
  deleteProject(id: string): Promise<void>;
  renameProject(id: string, title: string): Promise<Project>;
  /** Разобрать проект заново (после ошибки, после установки нейросетей или с другим языком): POST /projects/{pid}/analyze.
   *  opts.language — язык оригинала, выбранный человеком (ISO 639-1); не задан — определить по речи. */
  reanalyze(pid: string, opts?: ReanalyzeOptions): Promise<Job>;

  renameSpeaker(pid: string, sid: string, name: string): Promise<Project>;
  mergeSpeakers(pid: string, a: string, b: string): Promise<MergeResult>;
  /** Отменить слияние голосов по merge_id из ответа mergeSpeakers. */
  undoMerge(pid: string, mergeId: string): Promise<Project>;
  /** Сменить персонажа у реплики (вместе с её дублями, во всех сценах). newName — завести нового персонажа. */
  setLineSpeaker(pid: string, lineId: string, target: { speaker_id?: string; new_speaker_name?: string }): Promise<Project>;
  /** Склеить соседние реплики в одну (голос — у первой). */
  mergeLines(pid: string, lineIds: string[]): Promise<LinesMergeResult>;
  undoMergeLines(pid: string, mergeId: string): Promise<Project>;
  /** Разделить склеенную реплику обратно (отмена последней склейки, в которую она вошла). 404 — реплика не склеивалась. */
  splitLine(pid: string, lineId: string): Promise<Project>;
  snap(pid: string, req: SnapRequest): Promise<SnapResult>;

  createScene(pid: string, req: SceneCreate): Promise<SceneWithJob>;
  getScene(pid: string, scid: string): Promise<Scene>;
  patchScene(pid: string, scid: string, patch: ScenePatch): Promise<Scene>;
  deleteScene(pid: string, scid: string): Promise<void>;
  /** Подготовить сцену заново (голоса и фон) после прерванной подготовки: POST …/prepare. */
  prepareScene(pid: string, scid: string): Promise<Job>;

  uploadTake(pid: string, scid: string, wav: Blob, meta: TakeUploadMeta): Promise<Take>;
  patchTake(pid: string, scid: string, tid: string, patch: TakePatch): Promise<Take>;
  deleteTake(pid: string, scid: string, tid: string): Promise<void>;
  /** Отрезать повтор в конце дубля (Take.tail_cut_s): POST …/takes/{tid}/trim_tail. Старый движок ответит 404. */
  trimTakeTail(pid: string, scid: string, tid: string): Promise<Take>;
  /** Вернуть дубль целиком, вместе с хвостом (отмена trimTakeTail). */
  untrimTakeTail(pid: string, scid: string, tid: string): Promise<Take>;

  /** Проверка микрофона: запись «тишина, потом фраза» → уровни и совет (POST /mic/check). Старый движок ответит 404. */
  micCheck(wav: Blob, silenceS: number): Promise<MicCheckResult>;
  /** Эталонный сигнал для замера задержки (GET /mic/latency_signal.wav). */
  micSignal(): Promise<ArrayBuffer>;
  /** Замер задержки: запись, на которую попал сигнал из динамика (POST /mic/latency). */
  micLatency(wav: Blob, playOffsetS: number): Promise<LatencyResult>;

  preview(pid: string, scid: string): Promise<Job>;
  /** Рендер одной сцены или склейка нескольких (POST /projects/{pid}/render): одна задача, результат — job.result.path / url. */
  render(pid: string, req: RenderProjectRequest): Promise<Job>;

  // ИИ: ответы движка отдаются экранам как есть, без переименований
  translate(req: LLMRequest): Promise<TranslateResult>;
  fitLength(req: LLMRequest): Promise<FitLengthResult>;
  nameSpeakers(req: LLMRequest): Promise<NameSpeakersResult>;
  /** Разбор дубля (нужен take_id): замеры + совет. Работает и без модели. */
  directorTake(req: LLMRequest): Promise<DirectorTakeResult>;
  /** Просьба словами («кот громче»): patches + готовое mix, которое можно отправить в PATCH сцены как есть. */
  directorCommand(req: LLMRequest): Promise<DirectorCommandResult>;

  getSettings(): Promise<AppSettings>;
  putSettings(s: AppSettings): Promise<AppSettings>;
  /** Сохраняет ключ и сразу проверяет его: key_status ok / invalid (+ русское объяснение). */
  setKey(providerId: string, key: string): Promise<KeyResult>;
  checkProvider(providerId: string): Promise<CheckResult>;

  listJobs(): Promise<Job[]>;
  getJob(id: string): Promise<Job>;
  cancelJob(id: string): Promise<void>;
  /** Поток изменений задач (WS /ws/jobs). Возвращает функцию отписки. */
  subscribeJobs(cb: (job: Job) => void): () => void;

  /** Ссылки на медиа. null — видео нет (демо-режим рисует заглушку). */
  sourceVideoUrl(pid: string): string | null;
  /** Лёгкая mp4-копия для плеера, если браузер не играет исходник. null — копии нет (демо). */
  proxyVideo(pid: string): ProxySource | null;
  clipUrl(pid: string, scid: string): string | null;
  previewUrl(pid: string, scid: string): string | null;
  takeUrl(pid: string, scid: string, take: Take): string | null;
  thumbnailUrl(pid: string, rel: string | null | undefined): string | null;
}

export class EngineError extends Error {}
