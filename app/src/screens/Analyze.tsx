import { useEffect, useMemo, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { ArrowRight, Play, RotateCw, Scissors, Sparkles, Trash2, X } from "lucide-react";
import { activeJob, useStore } from "../state/store";
import { Player, usePlayerTick } from "../lib/player";
import { VideoPlayer, Transport } from "../components/VideoPlayer";
import { Timeline, type PickMode, type Zoom } from "../components/Timeline";
import { Help } from "../components/Help";
import { snapSelection } from "../lib/snap";
import { firstConversation } from "../lib/conversation";
import { Segmented } from "../components/Segmented";
import { JobSteps } from "../components/JobSteps";
import { EmptyState } from "../components/EmptyState";
import { MlPackBanner, useMlMissing } from "../components/MlPackBanner";
import { fmt, fmtSec, plural } from "../lib/format";
import { useAiReady } from "../lib/ai";
import { useStemsWait } from "../lib/stems";
import { analysisStateOf } from "./Projects";
import { AiHint, AI_OFF_TITLE } from "../components/AiHint";
import type { Analysis, Job, Line, Speaker } from "../api/types";

type Tab = "chars" | "scenes" | "text";

export function AnalyzeScreen() {
  const nav = useNavigate();
  const engine = useStore((s) => s.engine);
  const project = useStore((s) => s.project);
  const jobs = useStore((s) => s.jobs);
  if (!project) return null;
  const job = activeJob(jobs, "analyze", project.id);
  const an = project.analysis;

  if (!an) {
    return (
      <section className="screen center">
        <div className="panel waitcard">
          <div className="ph"><h2>Разбираю видео</h2></div>
          <div className="pb col" style={{ gap: 14 }}>
            <div className="note">Один раз на проект: читаю файл, достаю текст и тайминги, различаю голоса, ищу склейки. Можно закрыть окно — разбор продолжится в движке.</div>
            {job ? <JobSteps job={job} /> : <div className="note">Разбор не идёт.</div>}
            <AnalyzeFailure pid={project.id} job={job} />
            <button className="btn" onClick={() => nav("/")}>К проектам</button>
          </div>
        </div>
      </section>
    );
  }
  return <AnalyzeReady key={project.id} engineKind={engine?.kind ?? "mock"} />;
}

/**
 * Что делать, когда разбор не идёт или упал. Нет нейросетей — та же кнопка «Скачать нейросети», что на главном, а после
 * установки «Разобрать снова». Любая другая ошибка — «Попробовать снова». Нет задачи (движок перезапускали) — «Разобрать».
 */
export function AnalyzeFailure({ pid, job }: { pid: string; job: Job | null }) {
  const reanalyze = useStore((s) => s.reanalyze);
  const summary = useStore((s) => s.projects.find((p) => p.id === pid));
  const interrupted = !job && !!summary && analysisStateOf(summary) === "interrupted";
  const ml = useStore((s) => s.ml);
  const mlMissing = useMlMissing();
  const [busy, setBusy] = useState(false);
  const failed = job?.state === "failed";
  if (job && !failed && job.state !== "cancelled") return null;       // разбор идёт или закончился
  const mlBusy = ml.phase === "running" || ml.phase === "restarting";
  const showBanner = mlMissing || mlBusy || ml.phase === "failed";
  const again = async () => { setBusy(true); await reanalyze(pid); setBusy(false); };
  const label = interrupted ? "Продолжить" : job?.error_code === "ml_missing" || ml.installed ? "Разобрать снова" : failed ? "Попробовать снова" : "Разобрать";
  return (
    <div className="col" style={{ gap: 10 }} data-testid="analyze-failure">
      {interrupted && <div className="tag amber" style={{ whiteSpace: "normal", height: "auto", padding: "6px 8px" }} data-testid="interrupted">Разбор прерван — программу закрыли, пока он шёл. Нажми «Продолжить».</div>}
      {failed && <div className="tag bad" style={{ whiteSpace: "normal", height: "auto", padding: "6px 8px" }}>{job?.error || "Разбор не удался"}</div>}
      {showBanner && <MlPackBanner lead="Чтобы распознавать речь и отделять голоса от музыки, нужно один раз скачать нейросети (~1,3 ГБ, несколько минут). Потом нажми «Разобрать снова»." />}
      {/* пока нейросетей нет, повторять разбор бессмысленно: он упадёт на том же месте */}
      {!mlMissing && !mlBusy && (
        <button className={"btn " + (failed || interrupted ? "amber" : "")} onClick={again} disabled={busy} data-primary="reanalyze"><RotateCw size={14} />{label}</button>
      )}
    </div>
  );
}

function AnalyzeReady({ engineKind }: { engineKind: string }) {
  const nav = useNavigate();
  const engine = useStore((s) => s.engine)!;
  const project = useStore((s) => s.project)!;
  const an = project.analysis!;
  const ui = useStore((s) => s.ui);
  const createScene = useStore((s) => s.createScene);
  const deleteScene = useStore((s) => s.deleteScene);
  const prepareScene = useStore((s) => s.prepareScene);
  const setSceneId = useStore((s) => s.setSceneId);
  const sceneId = useStore((s) => s.sceneId);
  const jobs = useStore((s) => s.jobs);
  const wait = useStemsWait();

  const duration = project.media.duration;
  const player = useMemo(() => new Player(duration, Math.min(an.lines[0]?.start ?? 0, duration)), [duration, an.lines]);
  useEffect(() => () => player.dispose(), [player]);
  usePlayerTick(player, 10);

  const [tab, setTab] = useState<Tab>("chars");
  const [mode, setMode] = useState<PickMode>("dialogue");
  const [zoom, setZoom] = useState<Zoom>("all");
  const [sel, setSel] = useState<[number, number] | null>(null);
  const [title, setTitle] = useState("");
  const [busy, setBusy] = useState(false);

  const lines = an.lines;
  const cur = useMemo(() => lines.find((l) => l.start <= player.t && player.t <= l.end) ?? null, [lines, player.t]);
  const spk = (id: string | null | undefined) => an.speakers.find((s) => s.id === id);

  const stepLine = (dir: 1 | -1) => {
    const t = player.t;
    const sorted = [...lines].sort((a, b) => a.start - b.start);
    const next = dir > 0 ? sorted.find((l) => l.start > t + 0.05) : [...sorted].reverse().find((l) => l.start < t - 0.3);
    if (next) player.seek(next.start);
  };

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const el = e.target as HTMLElement;
      if (/INPUT|TEXTAREA|SELECT/.test(el.tagName)) return;
      if (e.code === "Space") { e.preventDefault(); player.toggle(); }
      else if (e.code === "ArrowRight") stepLine(1);
      else if (e.code === "ArrowLeft") stepLine(-1);
      else if (e.code === "Escape") setSel(null);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [player, lines]);

  const pickRange = (r: [number, number]) => {
    setSel(r);
    player.pause();
    player.seek(r[0]);
  };

  const make = async () => {
    if (!sel || busy) return;
    setBusy(true);
    const first = lines.find((l) => l.end > sel[0] && l.start < sel[1]);
    const auto = title.trim() || (first ? first.text.split(/\s+/).slice(0, 4).join(" ").replace(/[.,!?…]+$/, "") : `Сцена ${project.scenes.length + 1}`);
    const sc = await createScene(sel[0], sel[1], auto);
    setBusy(false);
    if (sc) { setSel(null); setTitle(""); setTab("scenes"); }
  };

  /** «Дальше»: сразу выделяем первый разговор — остаётся нажать «Создать сцену» (или выбрать другой момент). */
  const suggestMoment = () => {
    setTab("scenes");
    // первый настоящий разговор (двое и больше, от трёх реплик, паузы меньше 4 с), а не одиночная первая фраза
    const talk = firstConversation(lines);
    if (!talk) return;
    const r = snapSelection(talk.start, talk.end, lines, an.shot_cuts);
    if (r[1] > r[0]) {
      pickRange(r);
      useStore.getState().toast("Выделил первый разговор. Подходит — жми «Создать сцену», нет — выбери другой момент на шкале.", "info", { key: "first-moment" });
    }
  };

  const inSel = (l: Line) => !!sel && l.end > sel[0] && l.start < sel[1];
  const selLines = sel ? lines.filter(inSel).length : 0;
  const selSpeakers = sel ? new Set(lines.filter(inSel).map((l) => l.speaker_id)).size : 0;
  const scenesCount = project.scenes.length;
  const hasScenes = scenesCount > 0;
  // сразу после «Создать сцену» движок ещё не вернул реплики — считаем их из разбора, чтобы не мелькало «0 реплик»
  const sceneLines = (sc: { start: number; end: number; lines: unknown[] }) =>
    sc.lines.length || lines.filter((l) => l.end > sc.start && l.start < sc.end).length;

  const overlay = cur ? (
    <div className="sub"><b style={{ color: spk(cur.speaker_id)?.color }}>{spk(cur.speaker_id)?.name ?? spk(cur.speaker_id)?.label ?? ""}:</b> {cur.text}</div>
  ) : null;

  return (
    <section className="screen analyze">
      <div className="vcol">
        <VideoPlayer player={player} src={engine.sourceVideoUrl(project.id)} proxy={engine.proxyVideo(project.id)} cuts={an.shot_cuts} overlay={overlay} />
        <Transport player={player}>
          <button className="btn ghost sm" onClick={() => stepLine(-1)}>← реплика</button>
          <button className="btn ghost sm" onClick={() => stepLine(1)}>реплика →</button>
          {engineKind === "mock" && <span className="tag" title="Видео в демо — заглушка">демо-кадры</span>}
        </Transport>
      </div>

      <div className="panel">
        {!hasScenes && !sel && (
          <div className="nextstep" data-testid="nextstep">
            <div className="tx"><b>Разбор готов</b><span className="note">Теперь выбери кусок видео, который будешь озвучивать.</span></div>
            <button className="btn amber lg" onClick={suggestMoment} data-primary="moment">Дальше: выбери момент для озвучки <ArrowRight size={15} /></button>
          </div>
        )}
        {an.language_note && <LanguageNote pid={project.id} note={an.language_note} lang={an.language ?? null} audioIdx={an.audio_stream_index} />}
        <div className="side-tabs" role="tablist">
          <button role="tab" aria-selected={tab === "chars"} className={tab === "chars" ? "on" : ""} aria-label={`Персонажи: ${an.speakers.length}`} onClick={() => setTab("chars")}>Персонажи<span className="c" aria-hidden="true">{an.speakers.length}</span></button>
          <button role="tab" aria-selected={tab === "scenes"} className={tab === "scenes" ? "on" : ""} aria-label={`Сцены: ${scenesCount}`} onClick={() => setTab("scenes")}>Сцены<span className="c" aria-hidden="true">{scenesCount}</span></button>
          <button role="tab" aria-selected={tab === "text"} className={tab === "text" ? "on" : ""} onClick={() => setTab("text")}>Текст</button>
        </div>
        {tab === "chars" && <CharsTab pid={project.id} speakers={an.speakers} suggestions={an.merge_suggestions} dismissed={ui.dismissed_merges} />}
        {tab === "scenes" && (
          <div className="pb" role="tabpanel">
            {scenesCount === 0 && (
              <EmptyState title="Сцен пока нет" text="Протяни мышью по шкале внизу или кликни по реплике — выделится кусок для озвучки. Потом нажми «Создать сцену». Сцен может быть сколько угодно.">
                <button className="btn amber" onClick={suggestMoment}>Выделить первый разговор</button>
              </EmptyState>
            )}
            {project.scenes.map((sc) => {
              const prep = activeJob(jobs, "prepare_scene", project.id, sc.id);
              const preparing = prep?.state === "running";
              return (
                <div key={sc.id} className={"scrow" + (sc.id === sceneId ? " on" : "")}>
                  <button className="scmain" onClick={() => { setSceneId(sc.id); pickRange([sc.start, sc.end]); }} aria-label={`Сцена ${sc.title}`}>
                    <span className="t ell">{sc.title}</span>
                    <span className="mono mute m">{fmt(sc.start)} – {fmt(sc.end)} · {Math.round(sc.end - sc.start)} с · {sceneLines(sc)} {plural(sceneLines(sc), "реплика", "реплики", "реплик")}</span>
                    {preparing && prep ? <JobSteps job={prep} compact /> : sc.stems_ready ? <span className="tag ok" style={{ width: "fit-content" }}>голоса и фон готовы</span> : sc.prepare_state === "interrupted" ? <span className="tag amber" style={{ width: "fit-content" }}>подготовка прервана</span> : null}
                  </button>
                  {!preparing && !sc.stems_ready && sc.prepare_state === "interrupted" && (
                    <button className="btn sm amber" onClick={() => void prepareScene(sc.id)}><RotateCw size={12} />Подготовить снова</button>
                  )}
                  <button className="btn ghost sm iconbtn" aria-label={`Удалить сцену ${sc.title}`} onClick={() => void deleteScene(sc.id)}><Trash2 size={14} /></button>
                </div>
              );
            })}
          </div>
        )}
        {tab === "text" && <TextTab lines={lines} speakers={an.speakers} playT={player.t} sel={sel} onSeek={(t) => player.seek(t)} how={asrSummary(an)} />}
      </div>

      <div className="timeline">
        <div className="ph tlhead">
          <div className="row">
            <h2>Вся серия</h2><span className="mono mute" style={{ fontSize: 11.5 }}>{fmt(duration).replace(/\.\d$/, "")}</span>
            <Help label="Как выделять момент">Клик по реплике выделяет весь разговор. «Кадр» — от одной смены плана до другой. «Вручную» — протяни мышью. Протянуть можно в любом режиме.</Help>
            <Segmented label="Как выделять" value={mode} onChange={(v) => { setMode(v); setSel(null); }} options={[
              { value: "dialogue", label: "Диалог", title: "Клик по реплике — выделится весь разговор" },
              { value: "shot", label: "Кадр", title: "Клик — от склейки до склейки" },
              { value: "free", label: "Вручную", title: "Протяни мышью по шкале — выделится ровно то, что захватил" },
            ]} />
            <Segmented label="Масштаб" value={zoom} onChange={setZoom} options={[{ value: "all", label: "Вся" }, { value: "5m", label: "5 мин" }, { value: "1m", label: "1 мин" }]} />
          </div>
          <div className="selinfo">
            {sel ? (
              <>
                <span className="mono">{fmt(sel[0])} – {fmt(sel[1])}</span>
                <span className="mute">{fmtSec(sel[1] - sel[0])} · {selLines} {plural(selLines, "реплика", "реплики", "реплик")} · {selSpeakers} {plural(selSpeakers, "голос", "голоса", "голосов")}</span>
                <button className="btn ghost sm" onClick={() => { player.seek(sel[0]); player.play(sel[1]); }}><Play size={12} fill="currentColor" />Смотреть</button>
                <input className="inp sm" style={{ width: 170 }} value={title} onChange={(e) => setTitle(e.target.value)} placeholder="Название сцены" aria-label="Название сцены" onKeyDown={(e) => { if (e.key === "Enter") void make(); }} />
                <button className="btn amber" onClick={make} disabled={busy} data-primary="scene"><Scissors size={14} />Создать сцену</button>
                <button className="btn ghost sm iconbtn" aria-label="Снять выделение" onClick={() => setSel(null)}><X size={14} /></button>
              </>
            ) : hasScenes ? (
              <>
                <span className={wait.blocked ? "stale" : "mute"} data-testid="next-hint">{wait.blocked ? wait.hint : "Можно выделить ещё момент — или переходи к записи"}</span>
                <button className="btn primary" onClick={() => nav(`/p/${project.id}/record`)} data-primary="next" disabled={wait.blocked} title={wait.blocked ? wait.hint : "Следующий шаг: записать свой голос"}>Дальше: запись <ArrowRight size={14} /></button>
              </>
            ) : (
              <span className="mute hint">
                {mode === "shot" ? "Кликни по шкале — выделится кадр между склейками, или протяни мышью" : "Протяни мышью по шкале или кликни по реплике"}
              </span>
            )}
          </div>
        </div>
        <Timeline duration={duration} lines={lines} speakers={an.speakers} cuts={an.shot_cuts} chapters={an.chapters.length ? an.chapters : project.media.chapters}
          scenes={project.scenes} activeSceneId={sceneId} mode={mode} selection={sel} onSelect={pickRange} playhead={player.t} onSeek={(t) => player.seek(t)}
          hint={null}
          zoom={zoom} onOpenScene={(id) => { setSceneId(id); const sc = project.scenes.find((s) => s.id === id); if (sc) pickRange([sc.start, sc.end]); setTab("scenes"); }} />
      </div>
    </section>
  );
}

function CharsTab({ pid, speakers, suggestions, dismissed }: { pid: string; speakers: Speaker[]; suggestions: [string, string, number][]; dismissed: string[] }) {
  const rename = useStore((s) => s.renameSpeaker);
  const guess = useStore((s) => s.guessNames);
  const merge = useStore((s) => s.mergeSpeakers);
  const dismiss = useStore((s) => s.dismissMerge);
  const hold = useStore((s) => s.mergeHold);
  const [busy, setBusy] = useState(false);
  const aiNames = useAiReady("name_speakers");
  const unnamed = speakers.filter((s) => !s.name).length;
  const isDismissed = (a: string, b: string) => dismissed.includes(`${pid}|${a}|${b}`) || dismissed.includes(`${pid}|${b}|${a}`);
  // подсказка только одна и только после закрытия плашки «Объединены · Отменить» — чтобы не слить лишнее
  const sugAll = hold ? undefined : suggestions.find(([a, b]) => !isDismissed(a, b) && speakers.some((s) => s.id === a) && speakers.some((s) => s.id === b));
  // двум голосам дали одно имя — вероятно, это один персонаж (это увереннее, чем сходство голосов — показываем первым)
  const sameName = hold ? null : (() => {
    const seen = new Map<string, Speaker>();
    for (const s of speakers) {
      const key = (s.name ?? "").trim().toLowerCase();
      if (!key) continue;
      const other = seen.get(key);
      if (other && !isDismissed(other.id, s.id)) return [other, s] as const;
      seen.set(key, s);
    }
    return null;
  })();
  const sug = sameName ? undefined : sugAll;
  const A = sug && speakers.find((s) => s.id === sug[0]);
  const B = sug && speakers.find((s) => s.id === sug[1]);
  return (
    <div className="pb" role="tabpanel">
      <div className="between" style={{ marginBottom: 6, alignItems: "flex-start" }}>
        <span className="note">Нейросеть нашла {speakers.length} {plural(speakers.length, "голос", "голоса", "голосов")}. {unnamed ? "Нажми на имя и впиши своё, или" : "Имена можно поправить: нажми на имя."}</span>
        {unnamed > 0 && (
          <span className="col" style={{ gap: 3, alignItems: "flex-end", flex: "none" }}>
            <button className="btn sm" disabled={!aiNames || busy} title={aiNames ? "ИИ предложит имена по тому, как к героям обращаются в репликах" : AI_OFF_TITLE} onClick={async () => { setBusy(true); await guess(); setBusy(false); }}><Sparkles size={12} />Подсказать имена персонажей (ИИ)</button>
            {!aiNames && <AiHint />}
          </span>
        )}
      </div>
      <div>
        {speakers.map((s) => <SpeakerRow key={s.id} s={s} others={speakers.filter((o) => o.id !== s.id)} onRename={(n) => void rename(s.id, n)} onMerge={(into) => void merge(into, s.id)} />)}
      </div>
      {sug && A && B && (
        <div className="suggest" data-testid="merge-suggest">
          <div style={{ fontSize: 12.5 }}>
            <b style={{ color: B.color }}>{B.name ?? B.label}</b> звучит как <b style={{ color: A.color }}>{A.name ?? A.label}</b> (совпадение {Math.round(sug[2] * 100)}%). Скорее всего, это один и тот же персонаж.
          </div>
          <div className="row"><button className="btn sm primary" onClick={() => void merge(A.id, B.id)}>Объединить</button><button className="btn sm ghost" onClick={() => dismiss(sug[0], sug[1])}>Это разные</button></div>
        </div>
      )}
      {sameName && (
        <div className="suggest" data-testid="same-name-suggest">
          <div style={{ fontSize: 12.5 }}>Двум голосам дано одно имя — «{sameName[0].name}». Это один персонаж?</div>
          <div className="row"><button className="btn sm primary" onClick={() => void merge(sameName[0].id, sameName[1].id)}>Объединить</button><button className="btn sm ghost" onClick={() => dismiss(sameName[0].id, sameName[1].id)}>Нет, разные</button></div>
        </div>
      )}
      <div className="note" style={{ marginTop: 10 }}>Если двое говорят одновременно, программа отметит это место значком ⚠ — чисто разделить их голоса нельзя.</div>
    </div>
  );
}

function SpeakerRow({ s, others, onRename, onMerge }: { s: Speaker; others: Speaker[]; onRename: (name: string) => void; onMerge: (into: string) => void }) {
  const [v, setV] = useState(s.name ?? "");
  const [pick, setPick] = useState(false);
  const [into, setInto] = useState("");
  const last = useRef(s.name ?? "");
  useEffect(() => { setV(s.name ?? ""); last.current = s.name ?? ""; }, [s.name]);
  const commit = () => { if (v.trim() !== last.current) { last.current = v.trim(); onRename(v.trim()); } };
  return (
    <div className="chr">
      <i className="dot" style={{ background: s.color }} />
      <div className="grow">
        <input className="nm" value={v} placeholder={s.label} aria-label={`Имя: ${s.label}`} title="Нажми и впиши имя" onChange={(e) => setV(e.target.value)} onBlur={commit}
          onKeyDown={(e) => { if (e.key === "Enter") (e.target as HTMLInputElement).blur(); }} />
        <div className="meta">{s.line_count} {plural(s.line_count, "реплика", "реплики", "реплик")} · {Math.round(s.total_speech)} с речи</div>
      </div>
      {others.length > 0 && (
        <button className="btn ghost sm" aria-expanded={pick} aria-label={`Объединить ${s.name ?? s.label} с другим голосом`} title="Это тот же человек, что другой голос? Объединить их в одного" onClick={() => { setPick(!pick); setInto(others[0].id); }}>Объединить с…</button>
      )}
      {pick && (
        <div className="mergepick">
          <select className="inp sm" aria-label={`С кем объединить ${s.name ?? s.label}`} value={into} onChange={(e) => setInto(e.target.value)}>
            {others.map((o) => <option key={o.id} value={o.id}>{o.name ?? o.label}</option>)}
          </select>
          <button className="btn sm primary" onClick={() => { setPick(false); onMerge(into); }}>Объединить</button>
          <button className="btn sm ghost" onClick={() => setPick(false)}>Отмена</button>
        </div>
      )}
    </div>
  );
}

const LANG_LABEL: Record<string, string> = { ru: "русский", en: "английский", ja: "японский", de: "немецкий", fr: "французский", es: "испанский", it: "итальянский", ko: "корейский", zh: "китайский", uk: "украинский" };
const REDO_LANGS: [string, string][] = [["ru", "Русский"], ["en", "Английский"], ["ja", "Японский"], ["de", "Немецкий"], ["fr", "Французский"], ["es", "Испанский"], ["uk", "Украинский"]];

/** Чем распознана речь и откуда язык — одной серой строкой (подсказка, не главное). */
export function asrSummary(an: Analysis): string | null {
  const parts: string[] = [];
  if (an.asr_engine) parts.push(`Текст распознал ${an.asr_engine === "gigaam" ? "GigaAM" : "Whisper"}`);
  if (an.language) {
    const name = LANG_LABEL[an.language] ?? an.language;
    const from = an.language_source === "user" ? "выбран вручную" : an.language_source === "track" ? "по метке в файле" : an.language_source === "speech" ? "по речи" : "";
    parts.push(`язык: ${name}${from ? ` (${from})` : ""}`);
  }
  const extra = (an.asr_notes ?? []).join(" ");
  if (!parts.length && !extra) return null;
  return [parts.join(" · "), extra].filter(Boolean).join(". ");
}

/** Метка дорожки и язык речи разошлись: одна фраза и «Разобрать заново» с выбранным языком. */
function LanguageNote({ pid, note, lang, audioIdx }: { pid: string; note: string; lang: string | null; audioIdx: number }) {
  const reanalyze = useStore((s) => s.reanalyze);
  const askConfirm = useStore((s) => s.askConfirm);
  const [open, setOpen] = useState(false);
  const [pick, setPick] = useState(REDO_LANGS.find(([v]) => v !== lang)?.[0] ?? "en");
  const [busy, setBusy] = useState(false);
  const redo = async () => {
    const ok = await askConfirm({ title: "Разобрать заново?", text: "Текст и голоса найдутся заново. Сцены останутся, но реплики в них могут поменяться.", confirmLabel: "Разобрать" });
    if (!ok) return;
    setBusy(true);
    await reanalyze(pid, { language: pick, audio_stream_index: audioIdx });
    setBusy(false);
  };
  return (
    <div className="note" style={{ padding: "8px 12px", borderBottom: "1px solid var(--line)" }} data-testid="language-note">
      {note}{" "}
      {!open ? <button className="btn ghost sm" onClick={() => setOpen(true)}>Выбрать язык</button> : (
        <span className="row" style={{ gap: 6, display: "inline-flex", marginTop: 4 }}>
          <select className="inp sm" aria-label="Язык оригинала" value={pick} onChange={(e) => setPick(e.target.value)}>
            {REDO_LANGS.map(([v, t]) => <option key={v} value={v}>{t}</option>)}
          </select>
          <button className="btn sm" onClick={redo} disabled={busy}><RotateCw size={12} />Разобрать заново</button>
        </span>
      )}
    </div>
  );
}

function TextTab({ lines, speakers, playT, sel, onSeek, how }: { lines: Line[]; speakers: Speaker[]; playT: number; sel: [number, number] | null; onSeek: (t: number) => void; how?: string | null }) {
  const [q, setQ] = useState("");
  const shown = useMemo(() => lines.filter((l) => !q.trim() || l.text.toLowerCase().includes(q.trim().toLowerCase())), [lines, q]);
  const curId = lines.find((l) => l.start <= playT && playT <= l.end)?.id;
  const firstSel = sel ? lines.find((l) => l.end > sel[0] && l.start < sel[1])?.id : null;
  const listRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!firstSel) return;
    listRef.current?.querySelector(`[data-line="${firstSel}"]`)?.scrollIntoView({ block: "center" });
  }, [firstSel]);
  return (
    <div className="pb" role="tabpanel" ref={listRef}>
      <input className="inp" placeholder="Поиск по репликам" aria-label="Поиск по репликам" value={q} onChange={(e) => setQ(e.target.value)} style={{ marginBottom: 8 }} />
      {how && <div className="note" style={{ marginBottom: 6 }} data-testid="asr-summary">{how}</div>}
      {shown.length === 0 && <div className="note">Ничего не нашлось.</div>}
      {shown.map((l) => {
        const sp = speakers.find((s) => s.id === l.speaker_id);
        const inside = !!sel && l.end > sel[0] && l.start < sel[1];
        return (
          <div key={l.id} data-line={l.id} className={"line" + (inside ? " in" : "") + (l.id === curId ? " cur" : "")} onClick={() => onSeek(l.start)}>
            <span className="tm">{fmt(l.start)}</span>
            <div><span className="who" style={{ color: sp?.color }}>{sp?.name ?? sp?.label ?? "?"}{l.overlap ? " ⚠" : ""}</span> {l.text}</div>
          </div>
        );
      })}
    </div>
  );
}

