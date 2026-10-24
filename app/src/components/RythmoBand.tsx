import { useEffect, useRef } from "react";
import type { SceneLine, Speaker } from "../api/types";
import type { Player } from "../lib/player";
import { fitInfo } from "../lib/estimate";

interface Props {
  lines: SceneLine[];
  speakers: Speaker[];
  player: Player;
  /** секунды сцены, соответствующие t=0 плеера */
  sceneStart: number;
  /** какую реплику подсветить */
  currentId?: string | null;
  /** пикселей на секунду */
  pps?: number;
  /** положение красной линии, доля ширины */
  markAt?: number;
}

/**
 * Бегущая строка: текст каждой реплики растянут ровно на её длину (слова разнесены по интервалу).
 * Лента едет влево, красная линия — «говори сейчас».
 */
export function RythmoBand({ lines, speakers, player, sceneStart, currentId, pps = 120, markAt = 0.28 }: Props) {
  const trackRef = useRef<HTMLDivElement>(null);
  const wrapRef = useRef<HTMLDivElement>(null);
  const colorOf = (id: string | null) => speakers.find((s) => s.id === id)?.color ?? "#85827d";

  useEffect(() => {
    const apply = () => {
      const w = wrapRef.current?.clientWidth ?? 600;
      const x = w * markAt - (player.t - sceneStart) * pps;
      if (trackRef.current) trackRef.current.style.transform = `translateX(${x.toFixed(1)}px)`;
    };
    apply();
    return player.subscribe(apply);
  }, [player, sceneStart, pps, markAt]);

  return (
    <div className="rythmo" ref={wrapRef} role="img" aria-label="Бегущий текст">
      <div className="track" ref={trackRef}>
        {lines.map((l) => {
          const dur = l.end_rel - l.start_rel;
          const c = colorOf(l.speaker_id);
          const fit = fitInfo(l.text, dur);
          const words = l.text.split(/\s+/).filter(Boolean);
          const w = Math.max(24, dur * pps);
          // шрифт подбираем так, чтобы текст не вылезал за длину реплики
          const fs = Math.max(11, Math.min(24, (w - 8) / Math.max(4, l.text.length * 0.58)));
          return (
            <div key={l.line_id} className={"ln" + (l.line_id === currentId ? " cur" : "")} data-fit={fit.cls}
              style={{ left: l.start_rel * pps, width: w, fontSize: fs, borderColor: fit.cls === "bad" ? "var(--bad)" : c, color: c }}>
              {words.map((w, i) => <span key={i}>{w}</span>)}
            </div>
          );
        })}
      </div>
      <div className="mark" />
      <div className="lbl">говори, когда слово проходит красную линию</div>
    </div>
  );
}
