import { useRef } from "react";
import { Check } from "lucide-react";
import type { Job } from "../api/types";
import { jobProgress } from "../state/store";

/** «осталось около N мин» по оценке движка (Job.eta_s). null/мало — не показываем. */
export function etaText(etaS: number | null | undefined): string | null {
  if (etaS == null || !Number.isFinite(etaS) || etaS <= 0) return null;
  if (etaS < 10) return "почти готово";
  if (etaS < 90) return `осталось около ${Math.max(10, Math.round(etaS / 5) * 5)} с`;
  return `осталось около ${Math.round(etaS / 60)} мин`;
}

/**
 * Шаги фоновой задачи с прогрессом. Главная полоса — Job.progress (общий, монотонный), «осталось» — Job.eta_s.
 * Проценты внутри этапа не прыгают назад. Job.error показываем (он человеческий), Job.error_debug — никогда.
 */
export function JobSteps({ job, compact }: { job: Job; compact?: boolean }) {
  const peak = useRef<{ job: string; total: number; steps: Record<string, number> }>({ job: job.id, total: 0, steps: {} });
  if (peak.current.job !== job.id) peak.current = { job: job.id, total: 0, steps: {} };
  // общий прогресс считает движок и он только растёт; на всякий случай не даём полосе пятиться
  const fraction = job.state === "done" ? 1 : Math.max(peak.current.total, jobProgress(job));
  peak.current.total = fraction;
  const total = Math.round(fraction * 100);
  const eta = job.state === "running" ? etaText(job.eta_s) : null;
  return (
    <div className="steps" aria-label="Ход работы">
      {job.steps.map((s) => {
        const cls = s.state === "done" ? "done" : s.state === "running" ? "run" : s.state === "failed" ? "fail" : "wait";
        const pr = Math.max(peak.current.steps[s.key] ?? 0, s.progress);
        peak.current.steps[s.key] = s.state === "done" ? 0 : pr;
        return (
          <div key={s.key} className={"step " + cls} title={s.note ?? undefined}>
            <span className="ic">{s.state === "done" ? <Check size={9} strokeWidth={3} /> : s.state === "failed" ? "!" : ""}</span>
            <span>{s.title}</span>
            <span className="mono dim" style={{ fontSize: 11 }}>{s.state === "running" ? Math.round(pr * 100) + "%" : ""}</span>
          </div>
        );
      })}
      {!compact && (
        <>
          <div className="bar" role="progressbar" aria-valuenow={total} aria-valuemin={0} aria-valuemax={100} style={{ marginTop: 4 }}>
            <i style={{ width: total + "%" }} />
          </div>
          <div className="between note" style={{ fontSize: 11.5 }}><span>Всего {total}%</span><span>{eta ?? ""}</span></div>
        </>
      )}
    </div>
  );
}
