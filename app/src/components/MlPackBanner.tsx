import { useState } from "react";
import { Download, Loader2 } from "lucide-react";
import "../styles/mlpack.css";
import { inTauri } from "../api/client";
import { useStore } from "../state/store";
import { ML_TERMINAL_HINT, humanizeMlFailure, mlFraction, stageTitle } from "../lib/mlpack";

/** Нужно ли предлагать нейросети: движок сказал `capabilities.ml = false` (старый движок без поля — не гадаем). */
export function useMlMissing(): boolean {
  return useStore((s) => s.health?.capabilities?.ml === false);
}

/** Можно ли поставить нейросети кнопкой: только в приложении и только на этом компьютере (на своём сервере — руками). */
export function useCanInstallMl(): boolean {
  const demo = useStore((s) => s.demo);
  const processing = useStore((s) => s.ui.processing);
  return inTauri() && processing === "local" && !demo;
}

/**
 * Плашка «Чтобы распознавать речь … нужно один раз скачать нейросети · Скачать» с ходом установки.
 * Показывается, пока нейросетей нет или идёт установка; иначе ничего (в Настройках — `always`: там видно и «установлены»).
 */
export function MlPackBanner({ always = false, lead }: { always?: boolean; lead?: string }) {
  const ml = useStore((s) => s.ml);
  const missing = useMlMissing();
  const canInstall = useCanInstallMl();
  const start = useStore((s) => s.startMlInstall);
  const cancel = useStore((s) => s.cancelMlInstall);
  const processing = useStore((s) => s.ui.processing);
  const [cancelling, setCancelling] = useState(false);

  const busy = ml.phase === "running" || ml.phase === "restarting";
  if (!busy && ml.phase !== "failed" && !missing) {
    if (!always) return null;
    return (
      <div className="mlpack ok" data-testid="ml-ok" role="status">
        <div className="mlrow"><span className="mltitle">Нейросети установлены</span><span className="note">Речь, голоса и разделение звука работают на этом компьютере.</span></div>
      </div>
    );
  }

  if (ml.phase === "restarting") {
    return (
      <div className="mlpack" role="status" data-testid="ml-restarting">
        <div className="mlrow"><Loader2 size={15} className="spin" /><span className="mltitle">Готово, перезапускаю движок…</span></div>
        <div className="bar ok"><i style={{ width: "100%" }} /></div>
      </div>
    );
  }

  if (ml.phase === "running") {
    const pct = Math.round(mlFraction(ml) * 100);
    return (
      <div className="mlpack" role="status" data-testid="ml-running">
        <div className="mlrow">
          <div className="grow">
            <div className="mltitle">{stageTitle(ml.stage)} · шаг {ml.index} из {ml.total}</div>
            <div className="mldetail" title={ml.detail}>{ml.detail || "\u00a0"}</div>
          </div>
          <button className="btn sm" disabled={cancelling} onClick={async () => { setCancelling(true); await cancel(); setCancelling(false); }}>Отмена</button>
        </div>
        <div className="bar" role="progressbar" aria-valuenow={pct} aria-valuemin={0} aria-valuemax={100}><i style={{ width: pct + "%" }} /></div>
        <div className="note">Около 1,3 ГБ, несколько минут. Окно можно оставить открытым — разбор запустишь потом.</div>
      </div>
    );
  }

  if (ml.phase === "failed") {
    const f = humanizeMlFailure(ml.error);
    return (
      <div className="mlpack bad" role="alert" data-testid="ml-failed">
        <div className="mlrow">
          <div className="grow"><div className="mltitle">{f.title}</div><div className="note">{f.hint}</div></div>
          {canInstall && <button className="btn amber" onClick={() => void start()}><Download size={14} />Повторить</button>}
        </div>
        {ml.error && <details><summary>Подробности</summary><pre>{ml.error}</pre></details>}
      </div>
    );
  }

  // нейросетей нет, ничего не идёт
  const text = lead ?? "Чтобы распознавать речь и отделять голоса от музыки, нужно один раз скачать нейросети (~1,3 ГБ, несколько минут).";
  return (
    <div className="mlpack" data-testid="ml-missing">
      <div className="mlrow">
        <div className="grow">
          <div>{text}</div>
          {!canInstall && (
            <div className="note" style={{ marginTop: 4 }}>
              {processing === "remote"
                ? "Движок работает на другом компьютере — нейросети ставятся там."
                : <>Кнопка есть только в приложении «Дубль». Из терминала: <span className="mlcmd">{ML_TERMINAL_HINT}</span></>}
            </div>
          )}
        </div>
        {canInstall && <button className="btn amber" onClick={() => void start()} data-primary="ml"><Download size={14} />Скачать нейросети</button>}
      </div>
    </div>
  );
}
