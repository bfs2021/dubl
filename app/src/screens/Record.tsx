import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { AlertTriangle, ArrowRight, Check, ChevronDown, Combine, Mic, Play, Repeat, RotateCw, Scissors, Square, Star, Trash2, Upload, Wand2, Languages } from "lucide-react";
import { useScene, useStore } from "../state/store";
import { Player, usePlayerTick } from "../lib/player";
import { VideoPlayer, Transport } from "../components/VideoPlayer";
import { RythmoBand } from "../components/RythmoBand";
import { Segmented } from "../components/Segmented";
import { Switch } from "../components/Switch";
import { Meter } from "../components/Meter";
import { Waveform } from "../components/Waveform";
import { useWavePeaks } from "../components/useWavePeaks";
import { EmptyState } from "../components/EmptyState";
import { MIC_DENIED_HELP, MIC_UI_TEXT, MicRecorder, levelStep, listMicrophones, micErrorText, micUi, playCountdown, wavBlob, type LevelReading, type MicPermission } from "../audio/recorder";
import { Help } from "../components/Help";
import { sliceByScene, takeWindow } from "../audio/takes";
import { RATE_COMFORT, RATE_MAX, countSyllables, fmtRate, syllableBudget, type BudgetLevel } from "../lib/syllables";
import { REHEARSAL_TEXT, rehearsalPlan, type RehearsalStep } from "../lib/rehearsal";
import { CLIPPED_TEXT, EMPTY_TEXT, isClipped, isFileTake, starReason, textReview, visibleRemark, type TakeX } from "../lib/takeReview";
import { MicCheck } from "../components/MicCheck";
import { profileSummary, hasMicProfile } from "../lib/micProfile";
import { fmtNum, fmtRel, fmtSec, plural } from "../lib/format";
import { takeCaption } from "../lib/director";
import { useAiReady } from "../lib/ai";
import { useStemsWait } from "../lib/stems";
import { AiHint, AI_OFF_TITLE } from "../components/AiHint";
import { SpeakerMenu } from "../components/SpeakerMenu";
import { mergedLinesOf, type SceneLine, type Speaker, type Take } from "../api/types";

type Phase = "idle" | "count" | "rec" | "save" | "rehearse";

const sleep = (ms: number, signal?: AbortSignal) =>
  new Promise<void>((res, rej) => {
    const t = setTimeout(res, ms);
    signal?.addEventListener("abort", () => { clearTimeout(t); rej(new DOMException("aborted", "AbortError")); });
  });

export function RecordScreen() {
  const nav = useNavigate();
  const project = useStore((s) => s.project);
  const scene = useScene();
  const setSceneId = useStore((s) => s.setSceneId);
  if (!project) return null;
  if (!scene) {
    return (
      <section className="screen center">
        <EmptyState title="Сначала нарежь сцену" text="Запись идёт по сценам: выдели на шкале момент, который хочешь озвучить.">
          <button className="btn primary lg" onClick={() => nav(`/p/${project.id}/analyze`)}>К разбору <ArrowRight size={16} /></button>
        </EmptyState>
      </section>
    );
  }
  return <RecordScene key={project.id + "/" + scene.id} onPickScene={setSceneId} />;
}

function RecordScene({ onPickScene }: { onPickScene: (id: string) => void }) {
  const nav = useNavigate();
  const engine = useStore((s) => s.engine)!;
  const project = useStore((s) => s.project)!;
  const scene = useScene()!;
  const ui = useStore((s) => s.ui);
  const setUi = useStore((s) => s.setUi);
  const patchScene = useStore((s) => s.patchScene);
  const setTextMode = useStore((s) => s.setTextMode);
  const setLineText = useStore((s) => s.setLineText);
  const translate = useStore((s) => s.translateScene);
  const fitLine = useStore((s) => s.fitLine);
  const setLineSpeaker = useStore((s) => s.setLineSpeaker);
  const mergeLines = useStore((s) => s.mergeLines);
  const splitLine = useStore((s) => s.splitLine);
  const undoLineEdit = useStore((s) => s.undoLineEdit);
  const uploadTake = useStore((s) => s.uploadTake);
  const patchTake = useStore((s) => s.patchTake);
  const toast = useStore((s) => s.toast);
  const settings = useStore((s) => s.settings);
  const aiTranslate = useAiReady("translate");
  const aiFit = useAiReady("fit_length");
  const wait = useStemsWait();
  const fileInput = useRef<HTMLInputElement>(null);

  const speakers: Speaker[] = project.analysis?.speakers ?? [];
  const merged = mergedLinesOf(project.analysis);
  const dur = scene.end - scene.start;
  const player = useMemo(() => new Player(dur, 0), [dur, scene.id]);
  useEffect(() => () => player.dispose(), [player]);
  usePlayerTick(player, 15);

  const [lineId, setLineId] = useState<string | null>(scene.lines[0]?.line_id ?? null);
  const [phase, setPhase] = useState<Phase>("idle");
  const [beat, setBeat] = useState(0);
  const [level, setLevel] = useState(-120);
  const [busyAI, setBusyAI] = useState(false);
  const [micOn, setMicOn] = useState(false);
  const [micOpening, setMicOpening] = useState(false);
  const [micErr, setMicErr] = useState<string | null>(null);
  const [perm, setPerm] = useState<MicPermission>("unknown");
  const [checkOpen, setCheckOpen] = useState(false);
  const [rstep, setRstep] = useState<RehearsalStep>("idle");
  const [heard, setHeard] = useState(false);
  const [empty, setEmpty] = useState<{ text: string; kind: "line" | "scene" } | null>(null);
  const [devices, setDevices] = useState<MediaDeviceInfo[]>([]);
  const mic = useRef<MicRecorder>(new MicRecorder());
  const abort = useRef<AbortController | null>(null);
  const finishNow = useRef<(() => void) | null>(null);
  const defaulted = useRef(false);
  const lips = scene.mix.mode === "dub";
  const autofit = ui.autofit && lips;

  const dubbed = useMemo(() => new Set(scene.dubbed_speakers), [scene.dubbed_speakers]);
  const lineById = (id: string | null) => scene.lines.find((l) => l.line_id === id) ?? null;
  const line = lineById(lineId);
  const spk = (id: string | null | undefined) => speakers.find((s) => s.id === id);
  const sceneSpeakerIds = useMemo(() => [...new Set(scene.lines.map((l) => l.speaker_id).filter(Boolean) as string[])], [scene.lines]);
  const isDub = (l: SceneLine) => !l.speaker_id || dubbed.has(l.speaker_id);
  const takesOf = (id: string | null) => scene.takes.filter((t) => t.line_id === id);
  const dubLines = scene.lines.filter(isDub);

  // по умолчанию озвучиваем всех, кто говорит в сцене (один раз)
  useEffect(() => {
    if (defaulted.current || scene.dubbed_speakers.length || !sceneSpeakerIds.length) return;
    defaulted.current = true;
    void patchScene({ dubbed_speakers: sceneSpeakerIds });
  }, [scene.dubbed_speakers.length, sceneSpeakerIds, patchScene]);

  // первая реплика, если выбранной нет
  useEffect(() => { if (!lineById(lineId) && scene.lines[0]) setLineId(scene.lines[0].line_id); /* eslint-disable-next-line */ }, [scene.lines]);

  // список микрофонов и разрешение
  useEffect(() => {
    let dead = false;
    let status: PermissionStatus | undefined;
    void (async () => {
      try {
        const st = await (navigator.permissions as Permissions | undefined)?.query({ name: "microphone" as PermissionName });
        if (!st || dead) return;
        status = st;
        setPerm(st.state as MicPermission);
        // человек нажал «Разрешить» в окне macOS — включаем сами
        st.onchange = () => { setPerm(st.state as MicPermission); if (st.state === "granted" && !mic.current.isOpen) void enableMic(); };
        if (st.state === "granted") await enableMic();
      } catch { /* нет Permissions API */ }
    })();
    return () => { dead = true; if (status) status.onchange = null; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
  useEffect(() => {
    const m = mic.current;
    return () => { abort.current?.abort(); void m.close(); };
  }, []);
  // живая шкала уровня: читаем AnalyserNode ~30 раз в секунду, пока микрофон включён (в том числе до начала записи)
  useEffect(() => {
    if (!micOn) return;
    let r: LevelReading = { db: -120, peak: -120, heard: false };
    let last = performance.now();
    const iv = setInterval(() => {
      const now = performance.now();
      r = levelStep(r, mic.current.getLevel(), now - last);
      last = now;
      setLevel(r.db);
      if (r.heard) setHeard(true);
    }, 33);
    return () => clearInterval(iv);
  }, [micOn]);

  const enableMic = useCallback(async () => {
    setMicOpening(true);
    setMicErr(null);
    try {
      await mic.current.open(settings?.input_device, setLevel);
      setMicOn(true);
      setDevices(await listMicrophones());
    } catch (e) {
      // «нет доступа» объясняем в панели справа (там сказано, где включить), остальное — плашкой
      const msg = micErrorText(e);
      setMicErr(msg);
      if (msg !== MIC_DENIED_HELP) toast(msg, "error");
    } finally {
      setMicOpening(false);
    }
  }, [settings?.input_device, toast]);

  const changeDevice = async (id: string) => {
    await mic.current.close();
    mic.current = new MicRecorder();
    setMicOn(false);
    setHeard(false);
    if (settings) await useStore.getState().saveSettings({ ...settings, input_device: id || null });
    setMicOpening(true);
    setMicErr(null);
    try {
      await mic.current.open(id || null, setLevel);
      setMicOn(true);
    } catch (e) { setMicErr(micErrorText(e)); } finally { setMicOpening(false); }
  };

  // ───────────── запись ─────────────
  const record = useCallback(async (kind: "line" | "scene") => {
    if (phase !== "idle") return;
    const targets = kind === "line" ? (line ? [line] : []) : dubLines;
    setEmpty(null);
    if (!targets.length) { toast("Нечего записывать: выбери реплику или персонажа.", "error"); return; }
    if (kind === "line" && !isDub(targets[0])) { toast("Этого персонажа не озвучиваем — он остаётся оригиналом.", "error"); return; }
    if (!mic.current.isOpen) await enableMic();
    if (!mic.current.isOpen) return;
    const m = mic.current;
    const ac = new AbortController();
    abort.current = ac;
    const first = targets[0];
    const last = targets[targets.length - 1];
    const startAt = kind === "line" ? Math.max(0, first.start_rel - ui.lead_s) : 0;
    const stopAt = Math.min(dur, last.end_rel + 0.6);
    const preroll = ui.preroll_s;
    let captured = false;
    try {
      player.pause();
      player.seek(startAt);
      if (ui.countdown && m.ctx) {
        setPhase("count");
        await playCountdown(m.ctx, setBeat, 3, 0.6, ac.signal);
      }
      m.startCapture();
      captured = true;
      setPhase("rec");
      await sleep(preroll * 1000, ac.signal);
      player.seek(startAt);
      const ended = new Promise<void>((res) => {
        const off = player.onEnd(() => { off(); res(); });
        finishNow.current = () => { off(); res(); };
      });
      player.play(stopAt);
      await ended;
      player.pause();
    } catch (e) {
      if ((e as DOMException).name !== "AbortError") toast((e as Error).message, "error");
      if (captured && abort.current === ac && finishNow.current === null) { /* прервано во время предзаписи */ }
    }
    finishNow.current = null;
    if (!captured) { m.cancelCapture(); setPhase("idle"); setBeat(0); return; }
    const cap = m.stopCapture();
    setPhase("save");
    try {
      // тишину не отсекаем сами: решает движок (422 «Запись почти пустая…»), его текст покажем с кнопкой «Записать ещё раз»
      if (kind === "line") {
        const id = await uploadTake(wavBlob(cap.samples, cap.sampleRate), { line_id: first.line_id, start_rel: startAt, preroll, source: "mic" }, (m) => setEmpty({ kind, text: m }));
        if (id && !autofit) await patchTake(id, { offset_ms: 0, stretch: 1 });
        if (id && cap.peakDb < -55) toast("Почти тишина — проверь микрофон.", "error");
      } else {
        const t0 = startAt - preroll;
        let saved = 0;
        for (const l of targets) {
          const [a, b] = takeWindow(l, dur);
          const piece = sliceByScene(cap.samples, cap.sampleRate, t0, a, b);
          if (piece.length < cap.sampleRate * 0.2) continue;
          const id = await uploadTake(wavBlob(piece, cap.sampleRate), { line_id: l.line_id, start_rel: a, preroll: 0, source: "mic" }, (m) => setEmpty((e) => e ?? { kind, text: m }));
          if (id) saved++;
          if (id && !autofit) await patchTake(id, { offset_ms: 0, stretch: 1 });
        }
        if (saved) toast("Сцена записана — послушай дубли справа.", "ok");
      }
    } finally {
      setPhase("idle");
      setBeat(0);
      abort.current = null;
    }
  }, [phase, line, dubLines, dur, ui, autofit, player, enableMic, uploadTake, patchTake, toast]); // eslint-disable-line react-hooks/exhaustive-deps

  /** Готовая запись из другой программы (wav/mp3/m4a): движок сам переведёт в нужный формат и подгонит так же, как запись с микрофона. */
  const loadTakeFile = async (f: File) => {
    if (!line || phase !== "idle") return;
    setEmpty(null);
    setPhase("save");
    try {
      // файл считаем записью одной реплики: речь ищем рядом с её началом (запас в секунду до неё)
      const startRel = Math.max(0, line.start_rel - 1);
      const id = await uploadTake(f, { line_id: line.line_id, start_rel: startRel, preroll: 0, source: "file" }, (m) => setEmpty({ kind: "line", text: m }));
      if (id && !autofit) await patchTake(id, { offset_ms: 0, stretch: 1 });
      if (id) toast(`Файл «${f.name}» добавлен как дубль.`, "ok");
    } finally {
      setPhase("idle");
    }
  };

  /** Играть кусок сцены от `from` до `to`; отмена останавливает видео. */
  const playRange = (from: number, to: number, signal: AbortSignal) => new Promise<void>((res, rej) => {
    const onAbort = () => { off(); player.pause(); rej(new DOMException("aborted", "AbortError")); };
    const off = player.onEnd(() => { off(); signal.removeEventListener("abort", onAbort); res(); });
    if (signal.aborted) { onAbort(); return; }
    signal.addEventListener("abort", onAbort, { once: true });
    player.seek(from);
    player.play(to);
  });

  /** Репетиция: оригинал → отсчёт → тот же кусок с бегущей лентой. Ничего не записывается. */
  const rehearse = useCallback(async () => {
    if (phase !== "idle" || !line) return;
    const plan = rehearsalPlan(line, dur, ui.lead_s);
    const ac = new AbortController();
    abort.current = ac;
    setEmpty(null);
    setPhase("rehearse");
    let own: AudioContext | null = null;
    try {
      player.pause();
      setRstep("listen");
      await playRange(plan.from, plan.to, ac.signal);
      if (ui.countdown) {
        setRstep("count");
        player.seek(plan.from);
        let ctx = mic.current.ctx;
        if (!ctx) { own = new AudioContext(); ctx = own; await ctx.resume(); }
        await playCountdown(ctx, setBeat, 3, 0.6, ac.signal);
      }
      setRstep("speak");
      await playRange(plan.from, plan.to, ac.signal);
    } catch (e) {
      if ((e as DOMException).name !== "AbortError") toast((e as Error).message, "error");
    } finally {
      player.pause();
      setRstep("idle");
      setPhase("idle");
      setBeat(0);
      abort.current = null;
      void own?.close();
    }
  }, [phase, line, dur, ui.lead_s, ui.countdown, player, toast]); // eslint-disable-line react-hooks/exhaustive-deps

  const stop = () => {
    if (phase === "rehearse") abort.current?.abort();
    else if (phase === "count") abort.current?.abort();
    else if (phase === "rec") { if (finishNow.current) finishNow.current(); else abort.current?.abort(); }
  };

  const watch = () => {
    if (phase !== "idle") return;
    if (player.playing) { player.pause(); return; }
    if (line) { player.seek(Math.max(0, line.start_rel - 1)); player.play(Math.min(dur, line.end_rel + 0.5)); }
    else player.play(dur);
  };

  const move = (d: 1 | -1) => {
    const i = scene.lines.findIndex((l) => l.line_id === lineId);
    const n = scene.lines[Math.min(scene.lines.length - 1, Math.max(0, i + d))];
    if (n) { setLineId(n.line_id); player.seek(Math.max(0, n.start_rel - 0.5)); }
  };

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const el = e.target as HTMLElement;
      if (/INPUT|TEXTAREA|SELECT/.test(el.tagName) || el.isContentEditable || el.closest?.("[role=menu]")) return;
      // ⌘Z / Ctrl+Z — отменить последнюю правку реплик (склейку или смену персонажа); во время записи — нет
      if ((e.metaKey || e.ctrlKey) && !e.shiftKey && !e.altKey && e.code === "KeyZ") {
        e.preventDefault();
        if (phase !== "idle") return;
        if (!useStore.getState().lineUndo[project.id]?.length) { toast("Отменять нечего: правок реплик не было.", "info", { key: "line-undo-none" }); return; }
        void undoLineEdit();
        return;
      }
      if (e.metaKey || e.ctrlKey || e.altKey) return;
      if (e.code === "KeyR") { e.preventDefault(); phase === "idle" ? void record("line") : stop(); }
      else if (e.code === "Space") { e.preventDefault(); watch(); }
      else if (e.code === "ArrowDown") { e.preventDefault(); move(1); }
      else if (e.code === "ArrowUp") { e.preventDefault(); move(-1); }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  });

  const selectLine = (id: string) => {
    if (phase !== "idle") return;
    setLineId(id);
    const l = lineById(id);
    if (l) player.seek(Math.max(0, l.start_rel - 0.5));
  };

  const doTranslate = async () => { setBusyAI(true); await translate(); setBusyAI(false); };

  // предупреждение «Запись почти пустая» живёт, пока на экране тот дубль, о котором оно: сменили реплику или удалили дубль — убираем
  useEffect(() => { setEmpty(null); }, [lineId, scene.takes.length]);

  const idx = scene.lines.findIndex((l) => l.line_id === lineId);
  const curLine = scene.lines.find((l) => l.start_rel <= player.t && player.t <= l.end_rel) ?? null;
  const recording = phase === "rec" || phase === "count" || phase === "rehearse";
  const totalTakes = scene.takes.filter((t) => !t.empty).length;
  const counting = phase === "count" || rstep === "count";
  const mui = micUi({ open: micOn, opening: micOpening, permission: perm, failed: micErr === MIC_DENIED_HELP });
  const profile = ui.mic_profile;
  const needTranslate = scene.text_mode === "as_is" && scene.lines.length > 0 && scene.lines.every((l) => l.text === l.original);

  const subOverlay = (curLine && phase !== "count" && rstep !== "count") ? (
    <div className="sub"><b style={{ color: spk(curLine.speaker_id)?.color }}>{spk(curLine.speaker_id)?.name ?? spk(curLine.speaker_id)?.label ?? ""}:</b> {curLine.original}</div>
  ) : null;

  const overlays = (
    <>
      {subOverlay}
      {phase === "rec" && <div className="recbadge on"><i />ЗАПИСЬ</div>}
      {phase === "rehearse" && rstep !== "idle" && <div className="rehbadge on" data-testid="rehearsal-badge">{REHEARSAL_TEXT[rstep]}</div>}
      {counting && (
        <div className="countdown on" aria-live="assertive">
          <span>{Math.max(1, 3 - beat)}</span>
          <div className="beeps">{[0, 1, 2].map((i) => <i key={i} className={i <= beat ? "on" : ""} />)}</div>
        </div>
      )}
    </>
  );

  return (
    <section className="screen record">
      <div className="scenebar">
        <div className="grow-0">
          {project.scenes.length > 1 ? (
            <select className="inp sm scenesel" value={scene.id} aria-label="Сцена" onChange={(e) => onPickScene(e.target.value)} disabled={recording}>
              {project.scenes.map((s) => <option key={s.id} value={s.id}>{s.title}</option>)}
            </select>
          ) : <div className="t">{scene.title}</div>}
          <div className="mono mute" style={{ fontSize: 11.5 }}>{fmtRel(0)} – {fmtRel(dur)} · {fmtSec(dur)} · {scene.lines.length} {plural(scene.lines.length, "реплика", "реплики", "реплик")}</div>
        </div>
        <div className="vsep" />
        <div className="row wrap grow" role="group" aria-label="Кого озвучиваем" style={{ gap: 6 }}>
          <span className="mute" style={{ fontSize: 12 }}>Озвучиваю:</span>
          {sceneSpeakerIds.map((id) => {
            const s = spk(id);
            const on = dubbed.has(id);
            return (
              <button key={id} className={"chip" + (on ? " on" : "")} aria-pressed={on} disabled={recording}
                onClick={() => void patchScene({ dubbed_speakers: on ? scene.dubbed_speakers.filter((x) => x !== id) : [...scene.dubbed_speakers, id] })}>
                <i className="dot" style={{ background: s?.color }} />{s?.name ?? s?.label ?? id}
              </button>
            );
          })}
        </div>
        <Segmented label="Режим озвучки" value={scene.mix.mode} onChange={(mode) => void patchScene({ mix: { ...scene.mix, mode } })} options={[
          { value: "dub", label: "Дубляж", title: "Голоса заменяются полностью, твоя речь попадает в движения губ" },
          { value: "voiceover", label: "Закадровый", title: "Оригинал слышен тише, твой голос поверх" },
        ]} />
        <Help label="Чем дубляж отличается от закадрового"><b>Дубляж</b> — твой голос заменяет оригинальный и должен совпасть с движениями губ. <b>Закадровый</b> — оригинал слышно тише, твой голос поверх, как в закадровом переводе; попадать в губы не нужно.</Help>
        <button className={"btn " + (totalTakes ? "primary" : "")} onClick={() => nav(`/p/${project.id}/mix`)} data-primary="next" disabled={wait.blocked}
          title={wait.blocked ? wait.hint : "Следующий шаг: программа выровняет громкость голосов и фона — послушаешь, как звучит ролик"}>Дальше: сведение <ArrowRight size={14} /></button>
      </div>

      <div className="panel">
        <div className="ph lineshead">
          <Segmented label="Какой текст" value={scene.text_mode} onChange={(m) => void setTextMode(m)} options={[
            { value: "as_is", label: "Как есть", title: "Оригинал или его перевод" },
            { value: "custom", label: "Свой текст", title: "Пишешь сам" },
          ]} />
          {scene.text_mode === "as_is" && (
            <span className="row" style={{ gap: 8 }}>
              {!aiTranslate && <AiHint />}
              <button className={"btn sm" + (needTranslate && aiTranslate ? " amber" : "")} onClick={doTranslate} disabled={!aiTranslate || busyAI || recording} title={aiTranslate ? undefined : AI_OFF_TITLE}><Languages size={12} />{busyAI ? "Перевожу…" : "Перевести"}</button>
            </span>
          )}
        </div>
        <div className="pb">
          {wait.blocked && <div className="banner" role="status" data-testid="stems-wait">{wait.hint}. Записывать можно будет сразу после.</div>}
          {!lips && (
            <div className="modenote" data-testid="voiceover-note">
              <b>Закадровый режим:</b> оригинал слышен тише, твой голос поверх — как в закадровом переводе. Попадать в движения губ не нужно.
            </div>
          )}
          <div className="note" style={{ marginBottom: 8 }}>
            {scene.text_mode === "as_is" ? "Оригинал или его перевод. ИИ переводит только когда нажмёшь «Перевести»." : lips ? "Пиши свой текст. ИИ не вмешивается — только по кнопке «Подогнать под длину»." : "Пиши свой текст. ИИ не вмешивается."}
          </div>
          {scene.lines.map((l, i) => {
            const nx = scene.lines[i + 1];
            const nsp = nx ? spk(nx.speaker_id) : undefined;
            return (
            <LineItem key={l.line_id} line={l} n={i + 1} sel={l.line_id === lineId} speaker={spk(l.speaker_id)} dubbed={isDub(l)} mode={scene.text_mode}
              takes={takesOf(l.line_id).length} disabled={recording} aiFit={aiFit} lips={lips} speakers={speakers}
              next={nx ? { n: i + 2, gap: nx.start_rel - l.end_rel, sameSpeaker: (nx.speaker_id ?? null) === (l.speaker_id ?? null), name: nsp?.name ?? nsp?.label ?? "другого персонажа" } : null}
              merged={!!merged[l.line_id]}
              onSplit={() => splitLine(l.line_id)}
              onSpeaker={(target) => setLineSpeaker(l.line_id, target)}
              onMergeNext={() => mergeLines([l.line_id, nx!.line_id])}
              onSelect={() => selectLine(l.line_id)}
              onText={async (t) => { await setLineText(l.line_id, t); }}
              onFit={async () => {
                const v = await fitLine(l.line_id);
                if (v && !v.length) toast("ИИ не придумал вариантов — попробуй сократить сам.", "error");
                return v ?? [];
              }}
              onApplyVariant={async (t) => {
                if (scene.text_mode === "as_is") {
                  await setTextMode("custom");
                  toast("Переключил на «Свой текст».");
                }
                await setLineText(l.line_id, t);
              }} />
            );
          })}
          {scene.lines.length === 0 && <EmptyState title="В сцене нет реплик" text="Выдели на шкале момент с речью." />}
        </div>
      </div>

      <div className="vcol">
        <VideoPlayer player={player} src={engine.clipUrl(project.id, scene.id)} timecode={fmtRel} volume={rstep === "listen" ? 1 : ui.original_level} overlay={overlays}
          cuts={(project.analysis?.shot_cuts ?? []).map((c) => c - scene.start)} />
        <RythmoBand lines={scene.lines} speakers={speakers} player={player} sceneStart={0} currentId={lineId} />
        <Transport player={player} format={(t) => fmtRel(t)}>
          <button className="btn" onClick={() => move(-1)} aria-label="Предыдущая реплика" disabled={recording}>←</button>
          <button className="btn" onClick={watch} disabled={phase !== "idle" && !player.playing}>{player.playing && phase === "idle" ? <><Square size={12} fill="currentColor" />Стоп</> : <><Play size={12} fill="currentColor" />Смотреть</>}</button>
          {phase === "rehearse" ? (
            <button className="btn" onClick={stop} data-primary="record"><Square size={12} fill="currentColor" />Стоп репетиции</button>
          ) : (
            <button className="btn" onClick={() => void rehearse()} disabled={phase !== "idle" || !line || !isDub(line) || wait.blocked}
              title="Сначала послушаешь оригинал, потом пройдёшь реплику с отсчётом и бегущим текстом — без записи" data-testid="rehearse"><Repeat size={13} />Репетиция</button>
          )}
          {phase === "rec" || phase === "count" ? (
            <button className="btn rec" onClick={stop} data-primary="record"><Square size={13} fill="currentColor" />Стоп</button>
          ) : (
            <button className="btn rec" onClick={() => void record("line")} disabled={phase === "save" || phase === "rehearse" || !line || wait.blocked} title={wait.blocked ? wait.hint : undefined} data-primary="record">
              <Mic size={14} />Записать реплику {idx >= 0 ? idx + 1 : ""}
            </button>
          )}
          <button className="btn" onClick={() => void record("scene")} disabled={phase !== "idle" || wait.blocked} title={wait.blocked ? wait.hint : "Одной записью всю сцену — потом нарежем на дубли"}>Вся сцена</button>
          <button className="btn" onClick={() => move(1)} aria-label="Следующая реплика" disabled={recording}>→</button>
          <span className="grow" />
          <span className="keys" title="R — запись, пробел — смотреть, ↑↓ — реплики"><span className="k">R</span> запись <span className="k">␣</span> смотреть <span className="k">↑↓</span> реплики</span>
        </Transport>
      </div>

      <div className="col right">
        <div className="panel takespanel">
          <div className="ph"><h2>Дубли{line ? ` · реплика ${idx + 1}` : ""}</h2><span className="mute starhint">★ — пойдёт в ролик<Help label="Что такое звёздочка">Если у реплики несколько дублей, в ролик попадает тот, что отмечен звёздочкой. Программа сама ставит её самому полному дублю; нажми ★ на другом, чтобы заменить.</Help></span></div>
          <div className="pb">
            {empty && (
              <div className="emptytake" role="alert" data-testid="empty-take">
                <span>{empty.text}</span>
                <button className="btn sm" onClick={() => { setEmpty(null); void record(empty.kind); }} disabled={phase !== "idle"}><Mic size={12} />Записать ещё раз</button>
              </div>
            )}
            {line && takesOf(line.line_id).length === 0 && !empty && (
              <div className="note">Пока нет дублей. Нажми красную кнопку или клавишу <span className="k">R</span> — после сигналов говори в такт бегущему тексту.</div>
            )}
            {line && [...takesOf(line.line_id)].reverse().map((t, i, arr) => (
              <TakeItem key={t.id} take={t} n={arr.length - i} speaker={spk(t.speaker_id)} lineDur={line.end_rel - line.start_rel} lips={lips} onRetake={() => void record("line")} canRetake={phase === "idle" && !wait.blocked} />
            ))}
            {!line && <div className="note">Выбери реплику слева.</div>}
            {line && isDub(line) && (
              <div className="uploadtake">
                <input ref={fileInput} type="file" hidden accept="audio/*,.wav,.mp3,.m4a,.aac,.ogg,.flac" data-testid="take-file"
                  onChange={(e) => { const f = e.target.files?.[0]; e.target.value = ""; if (f) void loadTakeFile(f); }} />
                <button type="button" className="link" onClick={() => fileInput.current?.click()} disabled={phase !== "idle" || wait.blocked}><Upload size={11} /> или добавить готовую запись</button>
              </div>
            )}
          </div>
        </div>
        <div className="panel">
          <div className="pb micpanel">
            {!ui.headphones && (
              <div className="phones">
                <span>Надень наушники — иначе микрофон запишет звук видео.</span>
                <button className="btn sm" onClick={() => setUi({ headphones: true })}><Check size={12} />Я в наушниках</button>
              </div>
            )}
            {!hasMicProfile(profile) && (
              <div className="micprompt" data-testid="mic-prompt">
                <span>Микрофон ещё не проверен. Это займёт 10 секунд.</span>
                <button className="btn sm amber" onClick={() => setCheckOpen(true)}><Mic size={12} />Проверить микрофон</button>
              </div>
            )}
            {hasMicProfile(profile) && (
              <div className="micprofile" data-testid="mic-profile">
                <span className="note"><Check size={11} style={{ verticalAlign: "-1px" }} /> Мой микрофон: {profileSummary(profile)}</span>
                <button className="link" onClick={() => setCheckOpen(true)}>Проверить снова</button>
              </div>
            )}
            <div className="between">
              {devices.length > 1 ? (
                <select className="inp sm" aria-label="Микрофон" value={settings?.input_device ?? ""} onChange={(e) => void changeDevice(e.target.value)} disabled={recording}>
                  <option value="">Микрофон по умолчанию</option>
                  {devices.map((d, i) => <option key={d.deviceId} value={d.deviceId}>{d.label || `Микрофон ${i + 1}`}</option>)}
                </select>
              ) : <span className={"micstate " + mui} data-testid="mic-state" role="status">{micErr && mui !== "denied" ? micErr : MIC_UI_TEXT[mui]}</span>}
            </div>
            {devices.length > 1 && mui !== "on" && <div className={"micstate " + mui} data-testid="mic-state" role="status">{micErr && mui !== "denied" ? micErr : MIC_UI_TEXT[mui]}</div>}
            {micOn ? (
              <div className="livemeter" data-testid="live-meter">
                <Meter db={level} />
                <div className="lbl"><span>{heard ? <span className="heard">Слышу тебя ✓</span> : "Скажи что-нибудь — шкала должна двигаться"}</span><span className="mono">{Math.max(-60, Math.round(level))} дБ</span></div>
              </div>
            ) : mui !== "asking" && <button className="btn sm" onClick={enableMic}><Mic size={12} />{mui === "denied" ? "Попробовать ещё раз" : "Включить микрофон — проверить уровень"}</button>}
            <div className="opt"><span className="l">Отсчёт<Help label="Что такое отсчёт">Три коротких сигнала перед записью, как «раз, два, три». Четвёртого нет — на его месте говори.</Help></span><Segmented label="Отсчёт" value={ui.countdown ? "on" : "off"} onChange={(v) => setUi({ countdown: v === "on" })} options={[{ value: "on", label: "3 сигнала" }, { value: "off", label: "нет" }]} /></div>
            <div className="opt"><span className="l">До реплики<Help label="Что такое «до реплики»">Сколько секунд видео показать перед тем, как нужно заговорить, — чтобы войти в сцену, а не с места в карьер.</Help></span>
              <select className="inp sm" style={{ width: 80 }} value={ui.lead_s} aria-label="До реплики" onChange={(e) => setUi({ lead_s: Number(e.target.value) })}>
                {[1, 2, 4].map((n) => <option key={n} value={n}>{n} с</option>)}
              </select></div>
            <div className="opt col2"><span className="l">Слышать оригинал в наушниках</span><input type="range" className="rng" min={0} max={1} step={0.05} value={ui.original_level} aria-label="Громкость оригинала" onChange={(e) => setUi({ original_level: Number(e.target.value) })} /></div>
            {lips && (
              <div className="opt top"><span className="l">Подгонять под губы сам
                <span className="optdesc">Программа сама сдвинет и чуть растянет голос под губы. Выключи — подгонишь вручную в «Сведении».</span></span>
                <Switch label="Подгонять под губы сам" checked={ui.autofit} onChange={(v) => setUi({ autofit: v })} /></div>
            )}
          </div>
        </div>
      </div>
      <MicCheck open={checkOpen} onClose={() => setCheckOpen(false)} />
    </section>
  );
}

// ───────────── реплика ─────────────
const BUDGET_CLS: Record<BudgetLevel, string> = { empty: "short", ok: "ok", hurry: "warn", over: "bad" };

function LineItem(props: {
  line: SceneLine; n: number; sel: boolean; speaker?: Speaker; dubbed: boolean; mode: "as_is" | "custom"; takes: number; disabled: boolean; aiFit: boolean; lips: boolean;
  speakers: Speaker[]; next: { n: number; gap: number; sameSpeaker: boolean; name: string } | null;
  /** реплика получилась склейкой — её можно разделить обратно */
  merged: boolean; onSplit: () => Promise<void>;
  onSpeaker: (target: { speaker_id?: string; new_speaker_name?: string }) => Promise<void>; onMergeNext: () => Promise<void>;
  onSelect: () => void; onText: (t: string) => Promise<void>; onFit: () => Promise<string[]>; onApplyVariant: (t: string) => Promise<void>;
}) {
  const { line, n, sel, speaker, dubbed, mode, takes, disabled, aiFit, lips } = props;
  const [text, setText] = useState(line.text);
  const [variants, setVariants] = useState<string[] | null>(null);
  const [busy, setBusy] = useState(false);
  const [who, setWho] = useState(false);
  const whoBtn = useRef<HTMLButtonElement>(null);
  useEffect(() => setText(line.text), [line.text]);
  // варианты ИИ показываем только у выбранной реплики и только после «Подогнать под длину»: ушли с реплики — убираем
  useEffect(() => { if (!sel) setVariants(null); }, [sel]);
  useEffect(() => { if (disabled) setWho(false); }, [disabled]);
  const dur = line.end_rel - line.start_rel;
  const b = syllableBudget(mode === "custom" ? text : line.text, line.original, dur);
  const fill = Math.min(100, (b.rate / RATE_MAX) * 100);
  const needFit = b.level === "hurry" || b.level === "over";

  return (
    <div className={"rl-item" + (sel ? " sel" : "") + (dubbed ? "" : " orig")} onClick={props.onSelect} data-line={line.line_id}>
      <div className="hd">
        <span className="mono dim">{n}</span>
        <i className="dot" style={{ background: speaker?.color ?? "#85827d" }} />
        <button ref={whoBtn} type="button" className="who whobtn" style={{ color: speaker?.color }} disabled={disabled} aria-expanded={who} aria-haspopup="menu"
          aria-label={`Персонаж реплики ${n}: ${speaker?.name ?? speaker?.label ?? "не определён"}. Сменить`} title="Не тот персонаж? Нажми и выбери другого"
          onClick={(e) => { e.stopPropagation(); setWho(!who); }}
          onKeyDown={(e) => { if (e.key === "ArrowDown" && !who) { e.preventDefault(); e.stopPropagation(); setWho(true); } }}>
          {speaker?.name ?? speaker?.label ?? "?"}<ChevronDown size={11} aria-hidden="true" />
        </button>
        <span className="mono mute">{fmtSec(dur)}</span>
        <span className="grow" />
        {!dubbed && <span className="tag">оригинал</span>}
        {line.translation_source === "file" && <span className="srcflag" title="Русский текст взят из субтитров твоего файла">перевод из файла</span>}
        {takes > 0 && <span className="tag ok">{takes} {plural(takes, "дубль", "дубля", "дублей")}</span>}
      </div>
      {who && (
        <SpeakerMenu anchor={whoBtn} label={`Кто говорит реплику ${n}`} speakers={props.speakers} current={line.speaker_id}
          onPick={(id) => void props.onSpeaker({ speaker_id: id })} onNew={(name) => void props.onSpeaker({ new_speaker_name: name })} onClose={() => setWho(false)} />
      )}
      {mode === "custom" && text !== line.original && <div className="o">{line.original}</div>}
      {mode === "custom" ? (
        <textarea className="txt" rows={2} value={text} aria-label={`Текст реплики ${n}`} disabled={disabled}
          onChange={(e) => setText(e.target.value)} onFocus={props.onSelect}
          onBlur={() => { if (text.trim() && text !== line.text) void props.onText(text.trim()); }} />
      ) : (
        <div className="txt static">{line.text}</div>
      )}
      {dubbed && lips && (
        <div className="budget" data-testid="syllable-budget" data-level={b.level}>
          <div className="fit">
            <div className="fitbar" title={`Спокойно говорится до ${RATE_COMFORT} слогов в секунду, быстрее ${RATE_MAX} — не успеть`}><i className={BUDGET_CLS[b.level]} style={{ width: fill + "%" }} /><b style={{ left: (RATE_COMFORT / RATE_MAX) * 100 + "%" }} /></div>
            <span className={"fitlbl " + BUDGET_CLS[b.level]} data-testid="budget-label">{b.label}</span>
          </div>
          <div className="budgetnums">
            <span title="Слогов в оригинале">оригинал <b className="mono">{b.origSyllables}</b></span>
            <span title="Слогов в твоём тексте">твой текст <b className="mono">{b.mySyllables}</b></span>
            <span title="Скорость речи, если уложиться в длину реплики" className={"rate " + BUDGET_CLS[b.level]}><b className="mono">{fmtRate(b.rate)}</b> за {fmtSec(dur)}</span>
          </div>
          {b.trim > 0 && <div className="note" data-testid="budget-trim">Чтобы не торопиться — убрать примерно {b.trim} {plural(b.trim, "слог", "слога", "слогов")}.</div>}
        </div>
      )}
      {dubbed && lips && (
        <div className="ai-row">
          <button className={"btn sm" + (needFit && aiFit ? " amber" : "")} disabled={!aiFit || busy || disabled} title={aiFit ? "ИИ предложит варианты текста, которые успеешь произнести за время реплики" : AI_OFF_TITLE} onClick={async (e) => { e.stopPropagation(); if (!sel) props.onSelect(); setBusy(true); setVariants(await props.onFit()); setBusy(false); }}>
            <Wand2 size={11} />{busy ? "Думаю…" : "Подогнать под длину (ИИ)"}
          </button>
          {!aiFit && <AiHint />}
        </div>
      )}
      {sel && (props.merged || props.next) && (
        <div className="mergenext">
          {props.merged && (
            <button type="button" className="link" disabled={disabled} title="Склеили зря? Вернём реплики, как были до склейки"
              onClick={(e) => { e.stopPropagation(); void props.onSplit(); }}>
              <Scissors size={11} /> Разделить обратно
            </button>
          )}
          {props.next && (props.next.sameSpeaker ? (
            <button type="button" className="link" disabled={disabled}
              title={props.next.gap > 1.5 ? `Между репликами пауза ${fmtSec(props.next.gap)} — точно одна фраза?` : "Разбор разрезал одну фразу на две? Склей их в одну реплику"}
              onClick={(e) => { e.stopPropagation(); void props.onMergeNext(); }}>
              <Combine size={11} /> Склеить с репликой {props.next.n}
            </button>
          ) : (
            // у соседа другой персонаж: склейка «съела» бы его дубль — объясняем, что сделать
            <span className="mergeoff" data-testid="merge-other">
              <button type="button" className="link" disabled aria-describedby={`mergewhy-${line.line_id}`}><Combine size={11} /> Склеить с репликой {props.next.n}</button>
              <span className="note" id={`mergewhy-${line.line_id}`}>Реплика {props.next.n} — у другого персонажа ({props.next.name}). Если это одна фраза, сначала смени персонажа у неё.</span>
            </span>
          ))}
        </div>
      )}
      {sel && variants && variants.length > 0 && (
        <div className="pop variants" data-testid="fit-variants" onClick={(e) => e.stopPropagation()} role="listbox" aria-label="Варианты по длине">
          <div className="hd2">Выбери вариант — подставлю в текст</div>
          {variants.map((v, i) => {
            const vb = syllableBudget(v, line.original, dur);
            return (
              <button key={i} role="option" aria-selected="false" onClick={() => { setVariants(null); void props.onApplyVariant(v); }}>
                <span>{v}</span><span className="d">{countSyllables(v)} сл · {vb.label}</span>
              </button>
            );
          })}
          <button className="ghostbtn" onClick={() => setVariants(null)}>Оставить как было</button>
        </div>
      )}
    </div>
  );
}

// ───────────── дубль ─────────────
function TakeItem({ take: t, n, speaker, lineDur, lips, onRetake, canRetake }: { take: Take; n: number; speaker?: Speaker; lineDur: number; lips: boolean; onRetake: () => void; canRetake: boolean }) {
  const take = t as TakeX;
  const engine = useStore((s) => s.engine)!;
  const project = useStore((s) => s.project)!;
  const scene = useScene()!;
  const selectTake = useStore((s) => s.selectTake);
  const deleteTake = useStore((s) => s.deleteTake);
  const trimTail = useStore((s) => s.trimTail);
  const untrimTail = useStore((s) => s.untrimTail);
  const rawRemark = useStore((s) => s.remarks[`${project.id}/${scene.id}/${take.id}`]);
  const file = isFileTake(take);
  // советы про губы не нужны ни в закадровом режиме, ни дублю из файла: у файла нет привязки ко времени
  const remark = visibleRemark(rawRemark, { lips, file });
  const loadRemark = useStore((s) => s.loadRemark);
  const url = engine.takeUrl(project.id, scene.id, take);
  const peaks = useWavePeaks(url, take.id);
  const audio = useRef<HTMLAudioElement | null>(null);
  const [playing, setPlaying] = useState(false);
  const [prog, setProg] = useState(0);
  const [more, setMore] = useState(false);
  const cap = takeCaption(take, lips && !file);
  const review = textReview(take, rawRemark?.missing ?? []);
  const siblings = scene.takes.filter((x) => x.line_id === take.line_id) as TakeX[];
  const why = starReason(take, siblings);
  const clipped = !take.empty && isClipped(take);

  useEffect(() => { void loadRemark(take.id); }, [take.id, loadRemark]);

  const toggle = () => {
    if (!url) return;
    if (!audio.current) {
      const a = new Audio(url);
      a.ontimeupdate = () => setProg(a.duration ? a.currentTime / a.duration : 0);
      a.onended = () => { setPlaying(false); setProg(0); };
      audio.current = a;
    }
    const a = audio.current;
    if (playing) { a.pause(); setPlaying(false); } else { void a.play(); setPlaying(true); }
  };
  useEffect(() => () => { audio.current?.pause(); }, []);

  return (
    <div className={"take" + (take.selected ? " best" : "") + (take.empty ? " emptyt" : "")} data-testid="take">
      <div className="row" style={{ gap: 8 }}>
        <button className="playbtn sm" onClick={toggle} aria-label={playing ? "Пауза" : "Послушать дубль " + n} disabled={!url}>{playing ? <Square size={9} fill="currentColor" /> : <Play size={10} fill="currentColor" />}</button>
        <div className="grow">
          <div className="t">Дубль {n}{file && <span className="tag srcfile" title="Загружен готовым файлом, а не записан с микрофона под видео">из файла</span>}</div>
          <div className="m" title={file ? undefined : `сдвиг ${Math.round(take.offset_ms)} мс · длина ×${fmtNum(take.stretch, 2)}`}>
            {cap.main}{lips && !file ? ` · ${cap.timing}` : ""}{!file && cap.match ? ` · ${cap.match}` : ""}
          </div>
        </div>
        <button className="star" aria-label={take.selected ? "Этот дубль идёт в ролик" : "Взять этот дубль в ролик"} aria-pressed={take.selected} disabled={!!take.empty}
          title={take.empty ? "Пустой дубль в ролик не берём" : undefined} onClick={() => void selectTake(take.id, !take.selected)}><Star size={16} fill={take.selected ? "currentColor" : "none"} /></button>
        <button className="btn ghost sm iconbtn" aria-label="Удалить дубль" onClick={() => void deleteTake(take.id)}><Trash2 size={13} /></button>
      </div>
      {why && <div className="starwhy" data-testid="star-why"><Star size={10} fill="currentColor" />{why}</div>}
      <Waveform peaks={peaks} color={speaker?.color ?? "#f0b44c"} progress={prog} height={30} />
      {take.empty && <div className="tnote bad" role="alert" data-testid="take-empty"><AlertTriangle size={11} /><span>{EMPTY_TEXT}</span></div>}
      {review.kind !== "unknown" && !take.empty && (
        <div className={"tnote " + (review.warn ? "warn" : "ok")} data-testid="take-review" data-kind={review.kind}>
          {review.warn ? <AlertTriangle size={11} /> : <Check size={11} />}<span>{review.text}</span>
          {review.kind === "tail" && <button className="btn sm" onClick={() => void trimTail(take.id)}><Scissors size={11} />Отрезать</button>}
        </div>
      )}
      {take.tail_trimmed && !take.empty && <div className="tnote" data-testid="take-trimmed"><Scissors size={11} /><span>Повтор в конце отрезан.</span><button className="btn sm" onClick={() => void untrimTail(take.id)}>Вернуть</button></div>}
      {clipped && <div className="tnote warn" data-testid="take-clipped"><AlertTriangle size={11} /><span>{CLIPPED_TEXT}</span></div>}
      {file && <div className="tnote" data-testid="take-from-file"><Upload size={11} /><span>Загружен файлом: попадание в губы не проверяем.</span></div>}
      {!take.empty && (remark || (!rawRemark && !file) || (!lips && rawRemark)) && (
        <div className={"remark " + (remark?.severity ?? "")} aria-label="Замечание помощника">
          <Wand2 size={11} />
          <span>{remark ? remark.text : rawRemark ? "Закадровый режим: попадать в губы не нужно." : "Слушаю дубль…"}{remark?.more && (
            <> <button className="link" onClick={() => setMore(!more)} aria-expanded={more}>{more ? "Скрыть" : "Ещё"}</button>{more && <> {remark.more}</>}</>
          )}</span>
        </div>
      )}
      {remark?.severity === "warn" && !take.empty && (
        <button className="btn sm retake" onClick={onRetake} disabled={!canRetake}><RotateCw size={11} />Записать ещё раз</button>
      )}
      <span className="sr">{fmtSec(lineDur)}</span>
    </div>
  );
}
