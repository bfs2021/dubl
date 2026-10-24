import { useEffect, useRef, useState, type ReactNode } from "react";
import { Pause, Play } from "lucide-react";
import { fmt } from "../lib/format";
import { Player, usePlayerTick } from "../lib/player";
import { rng } from "../lib/rng";

/** Лёгкая копия для плеера (GET /files/{pid}/proxy): пока движок её делает, отвечает 202. */
export interface ProxySource {
  url: string;
  status: () => Promise<"ready" | "preparing" | "missing">;
}

interface Props {
  player: Player;
  /** ссылка на видео; null — рисуем заглушку */
  src: string | null;
  /** что показать, если браузер не смог проиграть src (например, mkv в WebKit) */
  proxy?: ProxySource | null;
  /** склейки — в заглушке меняют «кадр» */
  cuts?: number[];
  /** подпись поверх видео */
  overlay?: ReactNode;
  /** показывать таймкод в углу; для сцены — время от её начала */
  timecode?: (t: number) => string;
  muted?: boolean;
  /** громкость видео 0..1 */
  volume?: number;
  className?: string;
  /** 16/9 по умолчанию; для 9:16 превью — "9 / 16" */
  aspect?: string;
}

const PALETTE = ["#17304a", "#3a2a40", "#2f4630", "#4a3622", "#223a44", "#40262c", "#2a2f4d", "#35412a"];

/** Заглушка настоящего движка (видео ещё готовится или не играет): тёмный кадр без надписи «ДЕМО-ВИДЕО». */
function drawPlain(c: HTMLCanvasElement) {
  const g = c.getContext("2d");
  if (!g) return;
  g.fillStyle = "#0b0c0d";
  g.fillRect(0, 0, c.width, c.height);
}

function drawStub(c: HTMLCanvasElement, t: number, cuts: number[]) {
  const g = c.getContext("2d");
  if (!g) return;
  const w = c.width, h = c.height;
  const shot = cuts.filter((x) => x <= t).length;
  const r = rng(shot * 97 + 13);
  const base = PALETTE[shot % PALETTE.length];
  const grad = g.createLinearGradient(0, 0, w, h);
  grad.addColorStop(0, base);
  grad.addColorStop(1, "#0b0c0d");
  g.fillStyle = grad;
  g.fillRect(0, 0, w, h);
  // пол и «персонажи» — просто фигуры, чтобы глаз видел смену кадра
  g.fillStyle = "rgba(0,0,0,.28)";
  g.fillRect(0, h * 0.72, w, h * 0.28);
  for (let i = 0; i < 3; i++) {
    const x = w * (0.2 + 0.3 * i) + (r() - 0.5) * w * 0.08;
    const rad = h * (0.09 + r() * 0.06);
    g.fillStyle = ["#4fb8a8", "#e57a5f", "#6aa6e8"][(i + shot) % 3] + "aa";
    g.beginPath();
    g.ellipse(x, h * 0.62, rad * 0.9, rad * 1.25, 0, 0, Math.PI * 2);
    g.fill();
    g.beginPath();
    g.arc(x, h * 0.62 - rad * 1.5, rad * 0.65, 0, Math.PI * 2);
    g.fill();
  }
  g.fillStyle = "rgba(236,233,227,.16)";
  g.font = `500 ${Math.round(h * 0.045)}px "JetBrains Mono", monospace`;
  g.textAlign = "center";
  g.fillText("ДЕМО-ВИДЕО · КАДР " + (shot + 1), w / 2, h * 0.14);
}

/** Видео из движка или заглушка с таймкодом (демо). Таймкод и подпись — поверх. */
export function VideoPlayer({ player, src, proxy = null, cuts = [], overlay, timecode, muted, volume, className, aspect = "16 / 9" }: Props) {
  usePlayerTick(player, 12);
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  /** ok — играем src; preparing — src не проигрался, ждём копию; proxy — играем копию; broken — играть нечем */
  const [stage, setStage] = useState<"ok" | "preparing" | "proxy" | "broken">("ok");
  const proxyUrl = proxy?.url ?? null;
  const proxyStatus = proxy?.status;
  const cur = stage === "proxy" ? proxyUrl : stage === "ok" ? src : null;
  const useVideo = !!cur;

  useEffect(() => { setStage("ok"); }, [src, proxyUrl]);

  const onError = () => {
    if (stage === "ok" && proxyStatus) setStage("preparing");
    else setStage("broken");
  };

  // ждём лёгкую копию: движок отвечает 202, пока делает её
  useEffect(() => {
    if (stage !== "preparing" || !proxyStatus) return;
    let dead = false;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const poll = async () => {
      let st: "ready" | "preparing" | "missing" = "missing";
      try { st = await proxyStatus(); } catch { st = "preparing"; }
      if (dead) return;
      if (st === "ready") setStage("proxy");
      else if (st === "missing") setStage("broken");
      else timer = setTimeout(poll, 1500);
    };
    void poll();
    return () => { dead = true; clearTimeout(timer); };
  }, [stage, proxyStatus]);

  useEffect(() => {
    player.attach(useVideo ? videoRef.current : null);
    return () => player.attach(null);
  }, [player, useVideo, cur]);
  useEffect(() => {
    if (useVideo || !canvasRef.current) return;
    if (src) drawPlain(canvasRef.current); else drawStub(canvasRef.current, player.t, cuts);
  });
  useEffect(() => {
    if (videoRef.current) { videoRef.current.muted = !!muted; videoRef.current.volume = volume ?? 1; }
  }, [muted, volume, useVideo]);

  return (
    <div className={"vwrap " + (className ?? "")}>
      <div className="video" style={{ aspectRatio: aspect, ["--ar" as string]: aspect.replace(" / ", "/") }}>
        {useVideo ? (
          <video ref={videoRef} src={cur!} preload="auto" playsInline onError={onError} onClick={() => player.toggle()}
            onLoadedData={(e) => {
              // WebKit (окно Tauri) рисует кадр только после перемотки: без неё до «Смотреть» стоит чёрный экран
              const v = e.currentTarget;
              if (v.paused && v.currentTime < 0.05) { try { v.currentTime = Math.min(0.05, (v.duration || 1) / 2); } catch { /* не готов */ } }
            }} />
        ) : (
          <canvas ref={canvasRef} width={640} height={360} aria-label="Заглушка вместо видео" />
        )}
        {stage === "preparing" && <div className="vwait" role="status">Готовим видео для просмотра…</div>}
        {stage === "broken" && src && <div className="vwait" role="status">Не получилось показать видео. Звук и разбор от этого не страдают.</div>}
        <div className="tc mono">{timecode ? timecode(player.t) : fmt(player.t)}</div>
        {overlay}
      </div>
    </div>
  );
}

interface TransportProps {
  player: Player;
  total?: number;
  format?: (t: number) => string;
  children?: ReactNode;
}

export function Transport({ player, total, format = fmt, children }: TransportProps) {
  usePlayerTick(player, 12);
  const dur = total ?? player.duration;
  return (
    <div className="transport">
      <button className="playbtn" aria-label={player.playing ? "Пауза" : "Воспроизвести"} onClick={() => player.toggle()}>
        {player.playing ? <Pause size={13} fill="currentColor" /> : <Play size={13} fill="currentColor" />}
      </button>
      <span className="mono">{format(player.t)} <span className="dim">/ {format(dur)}</span></span>
      <span className="grow" />
      {children}
    </div>
  );
}
