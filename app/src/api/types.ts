/**
 * Типы UI. Всё, что есть в схеме движка, — тонкие алиасы на `schema.gen.ts` (генерируется `pnpm gen:api`,
 * руками не править). Раздел «Только UI» внизу — формы, которых в OpenAPI нет (ответы `/llm/*`, `/health`,
 * настройки, которые живут на этом устройстве). Они перечислены в отчёте как пожелания к контракту.
 */
import type { components } from "./schema.gen";

type S = components["schemas"];

// ───────────── исходный файл ─────────────
export type AudioStreamInfo = S["AudioStreamInfo"];
export type SubtitleStreamInfo = S["SubtitleStreamInfo"];
export type Chapter = S["Chapter"];
export type MediaInfo = S["MediaInfo"];

// ───────────── разбор ─────────────
export type Word = S["Word"];
export type Line = S["Line"];
export type Speaker = S["Speaker"];
export type Analysis = S["Analysis"];

// ───────────── сцена ─────────────
export type SceneTextMode = S["SceneTextMode"];
export type MixMode = S["MixMode"];
/** translation_source: откуда взят русский текст реплики (из субтитров файла, от ИИ или написан вручную). */
export type TranslationSource = NonNullable<S["SceneLine"]["translation_source"]>;
export type SceneLine = S["SceneLine"];
export type Take = S["Take"];
export type TrackSettings = S["TrackSettings"];
export type MixSettings = S["MixSettings"];
export type Scene = S["Scene"];

// ───────────── проект ─────────────
export type Project = S["Project"];
export type ProjectSummary = S["ProjectSummary"];

// ───────────── задачи ─────────────
export type JobState = S["JobState"];
export type JobStep = S["JobStep"];
export type Job = S["Job"];

// ───────────── рендер ─────────────
export type RenderAspect = S["RenderAspect"];
export type SubtitleMode = S["SubtitleMode"];
export type RenderFit = S["RenderFit"];
export type RenderSettings = S["RenderSettings"];
export type SubtitleText = S["SubtitleText"];
export type UrlProbe = S["UrlProbe"];

// ───────────── ИИ и настройки ─────────────
export type ProviderKind = S["ProviderKind"];
/** Статус ключа: unknown — ещё не проверяли, ok — модель ответила, invalid — модель ключ не приняла. */
export type KeyStatus = S["KeyResult"]["key_status"];
export type ProviderConfig = S["ProviderConfig"];
/** Провайдер в ответе GET /settings: движок считает key_set / key_status / key_message сам. */
export type ProviderState = S["ProviderConfig"];
/** То, что человек настраивает сам: без полей статуса ключа. */
export type ProviderDraft = Omit<S["ProviderConfig"], "key_set" | "key_status" | "key_message">;
export type KeyResult = S["KeyResult"];
export type LLMTask = S["LLMTask"];
export type LLMSettings = S["LLMSettings"];
export type AppSettings = S["AppSettings"];
export type CheckResult = S["CheckResult"];
export type EngineMicProfile = S["MicProfile"];
export type MicCheckResult = S["MicCheckResult"];
export type LatencyResult = S["LatencyResult"];

// ───────────── запросы (из схемы) ─────────────
export type SceneCreate = S["SceneCreate"];
export type ScenePatch = S["ScenePatch"];
export type TakePatch = S["TakePatch"];
export type SnapRequest = S["SnapRequest"];
export type SnapResult = S["SnapResult"];
export type SnapMode = SnapRequest["mode"];
export type LineTextPatch = S["LineTextPatch"];
export type Health = S["Health"];
export type CreateProjectOptions = S["ProjectCreate"];
/** Тело POST /projects/{pid}/analyze: язык и дорожка для нового разбора (не задано — определить самому). */
export type ReanalyzeOptions = S["AnalyzeRequest"];
export type FitLengthResult = S["FitLengthResult"];
export type TranslateResult = S["TranslateResult"];
/** Запрос к /llm/*: поля со значением по умолчанию (n, target_lang) можно не передавать. */
export type LLMRequest = Omit<S["LLMRequest"], "n" | "target_lang"> & Partial<Pick<S["LLMRequest"], "n" | "target_lang">>;
export type DirectorReport = S["DirectorReport"];
export type NameSpeakersResult = S["NameSpeakersResult"];

// ═══════════════ Ответы /llm/director и рендер склейки — прямо из схемы ═══════════════
export type MixPatch = S["MixPatch"];
export type DirectorTakeResult = S["DirectorTakeResult"];
export type DirectorCommandResult = S["DirectorCommandResult"];
/** overwrite — необязательное: без него движок не затирает чужой файл, а добавляет « (2)». */
export type RenderProjectRequest = Omit<S["RenderProjectRequest"], "settings"> & { settings: Omit<S["RenderSettings"], "overwrite"> & Partial<Pick<S["RenderSettings"], "overwrite">> };
export type JobResult = S["JobResult"];
export type TranslatedLine = S["TranslatedLine"];
export type LineVariants = S["LineVariants"];

// ═══════════════ Только UI (нет в OpenAPI) ═══════════════

/** Проект + задача разбора (id задачи движок отдаёт в заголовке X-Job-Id). */
export interface ProjectWithJob { project: Project; job: Job | null }
export interface SceneWithJob { scene: Scene; job: Job | null }

/** Поля multipart для POST …/takes (их описывает Body_upload_take…, но file там — строка). */
export interface TakeUploadMeta {
  line_id: string | null;
  start_rel: number;
  preroll: number;
  /** откуда дубль: запись с микрофона или загруженный файл (движок кладёт в Take.source) */
  source?: "mic" | "file";
}

/** Лёгкая копия исходного видео для плеера: GET /files/{pid}/proxy отвечает 202, пока её готовят. */
export type ProxyState = "ready" | "preparing" | "missing";
export interface ProxySource {
  url: string;
  status: () => Promise<ProxyState>;
}

/** Результат слияния голосов: проект и id, по которому слияние можно отменить. */
export interface MergeResult { project: Project; merge_id: string | null }
/** Результат склейки реплик: проект, id для отмены и сколько дублей перешло к склеенной реплике. */
export interface LinesMergeResult { project: Project; merge_id: string | null; moved_takes: number }

/**
 * Поля, которые движок добавляет аддитивно, а в schema.gen.ts их ещё нет (координатор перегенерирует типы).
 * merged_lines: id склеенной реплики → merge_id последней склейки, в которую она вошла («Разделить обратно»).
 */
export type AnalysisX = Analysis & { merged_lines?: Record<string, string> };
export const mergedLinesOf = (a: Analysis | null | undefined): Record<string, string> => (a as AnalysisX | null | undefined)?.merged_lines ?? {};
/** created — когда проект создан (если движок отдаёт его в сводке). */
export type ProjectSummaryX = ProjectSummary & { created?: string | null };

/** Хранится на этом устройстве (localStorage), на движок не уходит. */
export type ProcessingMode = "local" | "remote";
export interface UiSettings {
  processing: ProcessingMode;
  remote_url: string;
  remote_token: string;
  /** пользователь подтвердил, что записывает в наушниках */
  headphones: boolean;
  /** перед записью отсчёт «3 сигнала» */
  countdown: boolean;
  /** сколько секунд видео показывать до реплики */
  lead_s: number;
  /** сколько секунд микрофона писать до старта видео */
  preroll_s: number;
  /** громкость оригинала в наушниках, 0..1 */
  original_level: number;
  /** подгонять дубль к губам сам */
  autofit: boolean;
  /** пары голосов, про которые человек сказал «это разные»: `${pid}|${a}|${b}` */
  dismissed_merges: string[];
  /** профиль «Мой микрофон» после проверки (мастер MicCheck); null — ещё не проверяли */
  mic_profile?: import("../lib/micProfile").MicProfile | null;
  /** папка, куда человек в последний раз сохранил ролик (или выбрал в диалоге); пусто — папка по умолчанию */
  last_output_dir?: string;
}
