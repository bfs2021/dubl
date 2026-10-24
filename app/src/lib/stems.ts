import { activeJob, jobProgress, useStore } from "../state/store";
import type { Job, Project } from "../api/types";

export interface StemsWait {
  /** шаги 2–4 пока закрыты: голоса и фон выбранной сцены не готовы */
  blocked: boolean;
  /** готовность подготовки, % (null — задача не идёт) */
  pct: number | null;
  /** почему закрыто — для подсказки и тоста */
  hint: string;
}

/** Выбранная сцена ещё готовит «Голоса и фон»? Тогда запись, сведение и рендер открывать рано (движок ответил бы 409). */
export function stemsWait(project: Project | null, sceneId: string | null, jobs: Record<string, Job>): StemsWait {
  const sc = project?.scenes.find((s) => s.id === sceneId) ?? null;
  if (!project || !sc || sc.stems_ready) return { blocked: false, pct: null, hint: "" };
  // id сцен после удаления переиспользуются: среди старых задач ищем идущую, а не просто последнюю
  const all = Object.values(jobs).filter((j) => j.kind === "prepare_scene" && j.project_id === project.id && j.scene_id === sc.id);
  const job = all.find((j) => j.state === "running" || j.state === "queued") ?? activeJob(jobs, "prepare_scene", project.id, sc.id);
  if (job && (job.state === "running" || job.state === "queued")) {
    const pct = Math.round(jobProgress(job) * 100);
    return { blocked: true, pct, hint: `Дождись, пока программа отделит голоса от фона (${pct} %)` };
  }
  if (sc.prepare_state === "interrupted" || job?.state === "failed" || job?.state === "cancelled") {
    return { blocked: true, pct: null, hint: "Подготовка голосов и фона не дошла до конца — на вкладке «Сцены» нажми «Подготовить снова»" };
  }
  return { blocked: true, pct: null, hint: "Дождись, пока программа отделит голоса от фона" };
}

export function useStemsWait(): StemsWait {
  const project = useStore((s) => s.project);
  const sceneId = useStore((s) => s.sceneId);
  const jobs = useStore((s) => s.jobs);
  return stemsWait(project, sceneId, jobs);
}
