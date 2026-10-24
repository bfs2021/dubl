import { useEffect, useRef } from "react";

interface Props {
  /** пики 0..1 */
  peaks: number[];
  color?: string;
  height?: number;
  /** доля воспроизведения 0..1 — подсвечивает пройденное */
  progress?: number;
  /** доля начала/конца речи, чтобы приглушить хвосты */
  className?: string;
}

/** Волна на canvas. Без данных рисует тихую линию. */
export function Waveform({ peaks, color = "#f0b44c", height = 30, progress = 0, className }: Props) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const c = ref.current;
    if (!c) return;
    const dpr = window.devicePixelRatio || 1;
    const w = c.clientWidth || 200;
    c.width = Math.floor(w * dpr);
    c.height = Math.floor(height * dpr);
    const g = c.getContext("2d");
    if (!g) return;
    g.scale(dpr, dpr);
    g.clearRect(0, 0, w, height);
    const n = peaks.length || 1;
    const bw = w / n;
    const mid = height / 2;
    for (let i = 0; i < n; i++) {
      const v = Math.max(0.03, Math.min(1, (peaks[i] ?? 0) * 1.6));
      g.fillStyle = i / n < progress ? color : color + "88";
      const h = v * (height - 4);
      g.fillRect(i * bw + 0.5, mid - h / 2, Math.max(1, bw - 1), h);
    }
  }, [peaks, color, height, progress]);
  return <canvas ref={ref} className={className} style={{ width: "100%", height, display: "block" }} aria-hidden="true" />;
}
