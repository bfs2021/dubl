import { useEffect, useRef, useState } from "react";
import { useScene, useStore } from "../state/store";
import type { Player } from "../lib/player";
import type { Take } from "../api/types";
import { SHIFT_STEP_MS, clampShift, fmtShift, listenTake, shiftLimit, type Listening } from "../lib/takePlayback";

/** Пауза после клика: серия быстрых кликов даёт один запрос и одно проигрывание. */
export const SHIFT_SETTLE_MS = 450;

interface Props {
  take: Take;
  /** плеер экрана «Запись» — видео сцены */
  player: Player;
  /** идёт запись или репетиция — двигать и слушать нельзя */
  disabled?: boolean;
  settleMs?: number;
}

/** Строка «Сдвиг» в карточке дубля: ← раньше / позже → по 0,05 с и «▶ с видео» — реплика с видео и фоном, сразу. */
export function TakeShift({ take, player, disabled = false, settleMs = SHIFT_SETTLE_MS }: Props) {
  const engine = useStore((s) => s.engine)!;
  const project = useStore((s) => s.project)!;
  const scene = useScene()!;
  const patchTake = useStore((s) => s.patchTake);
  const toast = useStore((s) => s.toast);
  const [pending, setPending] = useState<number | null>(null);
  const [playing, setPlaying] = useState(false);
  const want = useRef<number | null>(null);
  const timer = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);
  const play = useRef<Listening | null>(null);
  const alive = useRef(true);

  const line = scene.lines.find((l) => l.line_id === take.line_id) ?? null;
  const lips = scene.mix.mode === "dub";
  const file = take.source === "file";
  const url = engine.takeUrl(project.id, scene.id, take);
  const value = pending ?? take.offset_ms;
  const limit = shiftLimit(take.offset_ms);

  useEffect(() => {
    alive.current = true;
    return () => { alive.current = false; clearTimeout(timer.current); play.current?.stop(); };
  }, []);
  useEffect(() => { if (disabled) { clearTimeout(timer.current); play.current?.stop(); } }, [disabled]);

  if (!lips || file || take.empty || !line) return null;

  const stopPlay = () => { play.current?.stop(); play.current = null; };

  const listen = async (offset: number) => {
    if (!url || !line) return;
    stopPlay();
    setPlaying(true);
    try {
      const p = await listenTake({
        player, video: player.media, takeUrl: url, bgUrl: engine.backgroundUrl(project.id, scene.id),
        take: { ...take, offset_ms: offset }, line, sceneDur: scene.end - scene.start,
        bgGainDb: scene.mix.background.gain_db, voiceGainDb: scene.mix.tracks[take.speaker_id ?? ""]?.gain_db ?? 0,
        onEnd: () => { if (alive.current) setPlaying(false); },
      });
      if (!alive.current) { p.stop(); return; }
      play.current = p;
    } catch (e) {
      if (alive.current) setPlaying(false);
      if ((e as DOMException).name !== "AbortError") toast("Не получилось проиграть реплику с видео: " + (e as Error).message, "error", { key: "take-shift-play" });
    }
  };

  const nudge = (d: -1 | 1) => {
    const next = clampShift((want.current ?? value) + d * SHIFT_STEP_MS, limit);
    want.current = next;
    setPending(next);
    stopPlay();
    clearTimeout(timer.current);
    timer.current = setTimeout(() => {
      const v = want.current;
      want.current = null;
      if (v === null) return;
      void listen(v);
      // после ответа движка показываем то, что лежит в проекте (не сохранилось — вернётся прежнее значение)
      void patchTake(take.id, { offset_ms: v }).finally(() => { if (alive.current && want.current === null) setPending(null); });
    }, settleMs);
  };

  return (
    <div className="takeshift" data-testid="take-shift" role="group" aria-label={`Сдвиг дубля: ${fmtShift(value)}`}>
      <span className="lbl">Сдвиг</span>
      <button type="button" className="btn sm" onClick={() => nudge(-1)} disabled={disabled || value <= -limit} title="Голос опаздывает за губами — подвинь раньше на 0,05 с">← раньше</button>
      <span className="val mono" data-testid="take-shift-value">{fmtShift(value)}</span>
      <button type="button" className="btn sm" onClick={() => nudge(1)} disabled={disabled || value >= limit} title="Голос спешит — подвинь позже на 0,05 с">позже →</button>
      <button type="button" className={"btn sm" + (playing ? "" : " amber")} disabled={disabled || !url} data-testid="take-shift-play"
        title="Послушать реплику с видео и фоном — как ляжет в ролик"
        onClick={() => { if (playing) { clearTimeout(timer.current); stopPlay(); setPlaying(false); } else void listen(want.current ?? value); }}>
        {playing ? "■ стоп" : "▶ с видео"}
      </button>
    </div>
  );
}
