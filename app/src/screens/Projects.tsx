import { useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { ArrowRight, FileVideo, Film, Link2, Plus, RotateCw, Trash2 } from "lucide-react";
import { activeJob, jobProgress, useStore } from "../state/store";
import { pickVideo, type Picked } from "../lib/files";
import { inTauri } from "../api/client";
import type { MediaInfo, UrlProbe } from "../api/types";
import { FoundInFile } from "../components/FoundInFile";
import { JobSteps } from "../components/JobSteps";
import { EmptyState } from "../components/EmptyState";
import { MlPackBanner, useCanInstallMl, useMlMissing } from "../components/MlPackBanner";
import { cleanTitle, fmtDate, fmtLong, plural } from "../lib/format";
import { MicCheck } from "../components/MicCheck";
import { hasMicProfile, profileSummary } from "../lib/micProfile";
import { Mic } from "lucide-react";
import { humanizeError } from "../lib/errors";
import type { ProjectSummary, ProjectSummaryX } from "../api/types";

/** Состояние разбора из сводки проекта (поле analysis_state добавляет движок; у старого движка его нет). */
export type AnalysisState = ProjectSummary["analysis_state"];
export const analysisStateOf = (p: ProjectSummary): AnalysisState | undefined => p.analysis_state;

const VIDEO_RE = /\.(mp4|mkv|mov|avi|webm|m4v)$/i;
const LANG_OPTS: [string, string][] = [["", "Определить по речи"], ["en", "Английский"], ["ja", "Японский"], ["ru", "Русский"], ["de", "Немецкий"], ["fr", "Французский"]];

function ago(iso: string): string {
  const m = Math.round((Date.now() - new Date(iso).getTime()) / 60000);
  if (m < 2) return "только что";
  if (m < 60) return `${m} мин назад`;
  const h = Math.round(m / 60);
  if (h < 24) return `${h} ${plural(h, "час", "часа", "часов")} назад`;
  const d = Math.round(h / 24);
  return `${d} ${plural(d, "день", "дня", "дней")} назад`;
}

/** Когда создан проект (если движок отдаёт created в сводке). */
const createdOf = (p: ProjectSummary) => (p as ProjectSummaryX).created ?? null;

/** Чем проект отличается от тёзки: длина видео, когда создан, сколько сцен. */
export function projectFacts(p: ProjectSummary): string {
  const parts = [`Видео ${fmtLong(p.duration)}`];
  const c = createdOf(p);
  if (c) parts.push(`создан ${fmtDate(c)}`);
  parts.push(p.scene_count ? `${p.scene_count} ${plural(p.scene_count, "сцена", "сцены", "сцен")}` : "сцен нет");
  return parts.join(" · ");
}

export function ProjectsScreen() {
  const nav = useNavigate();
  const engine = useStore((s) => s.engine);
  const projects = useStore((s) => s.projects);
  const jobs = useStore((s) => s.jobs);
  const ui = useStore((s) => s.ui);
  const settings = useStore((s) => s.settings);
  const createProject = useStore((s) => s.createProject);
  const deleteProject = useStore((s) => s.deleteProject);
  const loadProjects = useStore((s) => s.loadProjects);

  const [draft, setDraft] = useState<Picked | null>(null);
  const [media, setMedia] = useState<MediaInfo | null>(null);
  const [urlInfo, setUrlInfo] = useState<UrlProbe | null>(null);
  const [probing, setProbing] = useState(false);
  const [audioIdx, setAudioIdx] = useState<number | null>(null);
  const [lang, setLang] = useState("");
  const [link, setLink] = useState("");
  const [over, setOver] = useState(false);
  const [starting, setStarting] = useState(false);
  const [probeErr, setProbeErr] = useState<string | null>(null);
  const askConfirm = useStore((s) => s.askConfirm);
  const reanalyze = useStore((s) => s.reanalyze);
  const [needPath, setNeedPath] = useState(false);
  const [checkOpen, setCheckOpen] = useState(false);
  const mlMissing = useMlMissing();
  const canInstallMl = useCanInstallMl();
  const startMl = useStore((s) => s.startMlInstall);
  const mlPhase = useStore((s) => s.ml.phase);
  /** человек нажал «Начать разбор» без нейросетей: скачиваем и сами начинаем, как только они появятся */
  const [beginAfterMl, setBeginAfterMl] = useState(false);
  const draftSeq = useRef(0);
  const pathRef = useRef<HTMLInputElement>(null);

  useEffect(() => { void loadProjects(); }, [loadProjects, engine]);

  // пока что-то разбирается, список обновляем
  const running = projects.some((p) => activeJob(jobs, "analyze", p.id)?.state === "running");
  useEffect(() => {
    if (!running) return;
    const t = setInterval(() => void loadProjects(), 2500);
    return () => clearInterval(t);
  }, [running, loadProjects]);

  /** Удаление проекта — своим окном «Точно?»: window.confirm в окне Tauri не работает. */
  const removeProject = async (p: ProjectSummary) => {
    const ok = await askConfirm({
      title: `Удалить проект «${p.title}»?`,
      // имена бывают одинаковыми: в окне — длина видео, когда создан и что внутри, чтобы не удалить не тот
      text: `${projectFacts(p)}. Из списка пропадут разбор, сцены и дубли этого проекта. Само видео на диске останется как есть.`,
      confirmLabel: "Удалить проект", danger: true,
    });
    if (ok) await deleteProject(p.id);
  };

  const startDraft = async (p: Picked) => {
    if (!engine) return;
    const seq = ++draftSeq.current;
    setDraft(p);
    setMedia(null);
    setUrlInfo(null);
    setProbeErr(null);
    setProbing(true);
    try {
      const m = await engine.probe(p.url ? { url: p.url } : { path: p.path, name: p.name });
      if (seq !== draftSeq.current) return;
      if (m && "known" in m) {
        // ссылка: известно название и длина — или «узнаем после скачивания» (message), разбор всё равно можно начать
        setUrlInfo(m);
        if (m.title) setDraft({ ...p, name: m.title });
      } else {
        setMedia(m);
      }
      if (m && !("known" in m)) setAudioIdx((m.audio_streams.find((a) => a.default) ?? m.audio_streams.find((a) => a.language !== "rus") ?? m.audio_streams[0])?.index ?? null);
    } catch (e) {
      // ошибка про неподходящий файл — рядом с полем, где человек его указал, а не отдельной плашкой в углу
      if (seq === draftSeq.current) { setDraft(null); setProbeErr(humanizeError((e as Error).message)); setTimeout(() => pathRef.current?.focus(), 0); }
    } finally {
      if (seq === draftSeq.current) setProbing(false);
    }
  };

  /** В браузере путь к выбранному файлу не виден, поэтому диалог не открываем: говорим, что делать, и ставим курсор в поле пути. */
  const askForPath = () => {
    setNeedPath(true);
    setTimeout(() => pathRef.current?.focus(), 0);
  };

  const pick = async () => {
    if (!inTauri()) { askForPath(); return; }
    const p = await pickVideo();
    if (p) await startDraft(p);
  };

  const submitLink = async () => {
    const v = link.trim();
    if (!v) return;
    const isUrl = /^https?:\/\//i.test(v);
    const path = v.replace(/^["']|["']$/g, ""); // file:///… движок понимает сам
    setNeedPath(false);
    setProbeErr(null);
    await startDraft(isUrl ? { url: v, name: v } : { path, name: path.split(/[\\/]/).pop() ?? path });
    setLink("");
  };

  const begin = async () => {
    if (!draft || starting) return;
    if (mlMissing) {
      // сначала нейросети: без них разбор упал бы на шаге «Текст и тайминги»
      if (canInstallMl) { setBeginAfterMl(true); if (mlPhase !== "running" && mlPhase !== "restarting") void startMl(); }
      return;
    }
    setStarting(true);
    const p = await createProject({
      path: draft.path, url: draft.url, audio_stream_index: audioIdx, language: lang || null,
      // имя файла чистим: без расширения, точек, подчёркиваний и «720p / x264»; у ссылки — название ролика из проверки ссылки
      // (иначе, пока идёт скачивание, проект назывался бы «Загрузка по ссылке»)
      title: draft.path ? cleanTitle(draft.name) : urlInfo?.title?.trim() || undefined,
    });
    setStarting(false);
    if (p) {
      setDraft(null);
      setMedia(null);
      setUrlInfo(null);
      nav(`/p/${p.id}/analyze`);
    }
  };

  const onDrop = async (e: React.DragEvent) => {
    e.preventDefault();
    setOver(false);
    const f = e.dataTransfer.files?.[0];
    if (!f) return;
    if (!VIDEO_RE.test(f.name)) { setProbeErr("Это не похоже на видео. Подойдут MP4, MKV, MOV, AVI, WEBM."); return; }
    const path = (f as File & { path?: string }).path;
    if (!path) { askForPath(); return; } // браузер не отдаёт путь перетащенного файла
    await startDraft({ path, name: f.name });
  };

  // Tauri отдаёт пути перетащенных файлов отдельным событием
  useEffect(() => {
    if (!inTauri()) return;
    let off: (() => void) | undefined;
    void import("@tauri-apps/api/webview").then(async ({ getCurrentWebview }) => {
      off = await getCurrentWebview().onDragDropEvent((ev) => {
        if (ev.payload.type === "over") setOver(true);
        else if (ev.payload.type === "leave") setOver(false);
        else if (ev.payload.type === "drop") {
          setOver(false);
          const path = ev.payload.paths.find((p) => VIDEO_RE.test(p));
          if (path) void startDraft({ path, name: path.split(/[\\/]/).pop() ?? path });
        }
      });
    });
    return () => off?.();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [engine]);

  // нейросети встали, движок перезапущен — продолжаем с того места, где остановились
  useEffect(() => {
    if (beginAfterMl && !mlMissing && mlPhase === "idle" && draft && !starting) { setBeginAfterMl(false); void begin(); }
    if (beginAfterMl && mlPhase === "failed") setBeginAfterMl(false);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [beginAfterMl, mlMissing, mlPhase]);

  const latency = settings?.latency_ms ?? 0;
  const profile = ui.mic_profile;

  return (
    <section className="screen home">
      <div className="col">
        <MlPackBanner />
        <div className={"drop" + (over ? " over" : "") + (draft ? " compact" : "")} tabIndex={0} role="button" aria-label="Добавить видео" onClick={pick}
          onKeyDown={(e) => { if (e.key === "Enter" || e.key === " ") void pick(); }}
          onDragOver={(e) => { e.preventDefault(); setOver(true); }} onDragLeave={() => setOver(false)} onDrop={onDrop}>
          <h2>Новый проект</h2>
          <h1>Перетащи сюда серию, фильм или клип</h1>
          <p className="mute">Можно целиком — нужные моменты выберешь потом. Видео не копируем, держим ссылку на файл.</p>
          <div className="row wrap" style={{ marginTop: 6 }}>
            <button className="btn primary" onClick={(e) => { e.stopPropagation(); void pick(); }}><FileVideo size={15} />{inTauri() ? "Выбрать файл" : "Указать путь к файлу"}</button>
            {["MP4", "MKV", "MOV", "AVI", "WEBM"].map((f) => <span key={f} className="tag">{f}</span>)}
          </div>
        </div>

        {!draft && (
          <div className="panel"><div className="pb col" style={{ gap: 12 }}>
            <form className="field" onSubmit={(e) => { e.preventDefault(); void submitLink(); }}>
              <span>{needPath ? "Вставь полный путь к видео на этом компьютере" : "или путь к файлу / ссылка на видео"}</span>
              {needPath && (
                <div className="phones" role="alert">
                  В браузере нельзя выбрать файл кнопкой — он не сообщает, где лежит файл. Вставь сюда полный путь, например <span className="mono">/Users/имя/Movies/фильм.mkv</span>. В приложении «Дубль» файл выбирается обычным окном.
                </div>
              )}
              <div className="row">
                <input ref={pathRef} className={"inp" + (needPath || probeErr ? " attn" : "")} value={link} onChange={(e) => { setLink(e.target.value); setProbeErr(null); }} aria-invalid={!!probeErr} placeholder={needPath ? "/Users/имя/Movies/фильм.mkv" : "/путь/к/видео.mkv или https://…"} aria-label="Ссылка или путь" />
                <button className="btn" type="submit" disabled={!link.trim()}><Link2 size={14} />Открыть</button>
              </div>
              {probeErr && <div className="fielderr" role="alert" data-testid="probe-error">{probeErr}</div>}
            </form>
          </div></div>
        )}

        {draft && (
          <div className="panel intake" aria-label="Новый проект">
            <div className="ph"><div className="ell"><b>{draft.name}</b></div><button className="btn ghost sm" onClick={() => { setDraft(null); setMedia(null); setUrlInfo(null); }}>Отмена</button></div>
            <div className="pb col intakebody" style={{ gap: 14 }}>
              {probing && <div className="note">Читаю файл…</div>}
              {media && <FoundInFile media={media} audioIdx={audioIdx} onAudioIdx={setAudioIdx} />}
              {urlInfo && (
                <div className="note" data-testid="url-info">
                  {urlInfo.known
                    ? `Ролик найден${urlInfo.title ? `: «${urlInfo.title}»` : ""}${urlInfo.duration ? `, ${fmtLong(urlInfo.duration)}` : ""}${urlInfo.has_subtitles ? ". Есть субтитры — возьмём готовый текст." : "."}`
                    : (urlInfo.message || "Проверить ссылку сейчас не получилось — узнаем после скачивания.")}
                  {" "}Можно начинать разбор.
                </div>
              )}
              {!probing && !media && !urlInfo && (
                <div className="note">Дорожки, субтитры и главы прочитаем при загрузке; если их нет — всё разложим сами.</div>
              )}
              <label className="field">
                <span>Язык оригинала</span>
                <select className="inp" value={lang} onChange={(e) => setLang(e.target.value)}>
                  {LANG_OPTS.map(([v, t]) => <option key={v} value={v}>{t}</option>)}
                </select>
              </label>
            </div>
            {/* кнопка закреплена внизу: список дорожек может быть длинным — он прокручивается, а «Начать разбор» всегда на виду */}
            <div className="intakefoot">
              <button className="btn primary lg" onClick={begin} disabled={starting || probing || (mlMissing && !canInstallMl) || beginAfterMl}
                title={mlMissing && !canInstallMl ? "Сначала нужны нейросети — см. подсказку вверху" : undefined} data-primary="start">
                {beginAfterMl ? "Жду нейросети…" : mlMissing && canInstallMl ? "Скачать нейросети и начать разбор" : "Начать разбор"} <ArrowRight size={16} />
              </button>
              <div className="note">Голоса от музыки отделяются только в тех кусках, которые ты выберешь. Так серия обрабатывается минуты, а не часы.</div>
            </div>
          </div>
        )}

        {!draft && <div className="panel">
          <div className="ph"><h2>Перед первой записью</h2><button className="btn ghost sm" onClick={() => nav("/settings/mic")}>Микрофон <ArrowRight size={12} /></button></div>
          <div className="pb col" style={{ gap: 9 }}>
            <div className={"step " + (hasMicProfile(profile) ? "done" : "wait")} data-testid="mic-step">
              <span className="ic">{hasMicProfile(profile) ? "✓" : ""}</span>
              <div>{hasMicProfile(profile) ? `Мой микрофон: ${profileSummary(profile)}` : latency > 0 ? `Задержка записи измерена (${Math.round(latency)} мс), микрофон ещё не проверяли` : "Микрофон ещё не проверяли — это 10 секунд"}</div>
              <button className="btn sm" onClick={() => setCheckOpen(true)}><Mic size={12} />{hasMicProfile(profile) ? "Ещё раз" : "Проверить"}</button>
            </div>
            <div className={"step " + (ui.headphones ? "done" : "wait")}><span className="ic">{ui.headphones ? "✓" : ""}</span><div>Наушники — без них микрофон запишет звук фильма</div><span /></div>
          </div>
        </div>}
      </div>

      <div className="panel projpanel">
        <div className="ph"><h2>Проекты</h2><span className="mute" style={{ fontSize: 12 }}>{projects.length} {plural(projects.length, "проект", "проекта", "проектов")}</span></div>
        <div className="pb" role="list" aria-label="Список проектов">
          {projects.length === 0 && (
            <EmptyState title="Пока пусто" text="Перетащи видео в рамку слева или нажми «Выбрать файл». Программа прочитает файл и разберёт его: текст, голоса, склейки. Готовые проекты появятся здесь.">
              <Film size={34} className="dim" aria-hidden="true" />
            </EmptyState>
          )}
          {projects.length > 0 && (
            <div className="pgrid">
              {projects.map((p) => {
                const job = activeJob(jobs, "analyze", p.id);
                const busy = job?.state === "running";
                const st = analysisStateOf(p);
                const interrupted = !busy && st === "interrupted";
                const thumb = engine?.thumbnailUrl(p.id, p.thumbnail) ?? null;
                return (
                  <article key={p.id} className="pcard" role="listitem" data-testid="project-card">
                    <button className="pthumb" onClick={() => nav(`/p/${p.id}/analyze`)} aria-label={`Открыть проект ${p.title}`} tabIndex={-1}>
                      {thumb ? <img src={thumb} alt="" loading="lazy" /> : <Film size={28} className="dim" aria-hidden="true" />}
                    </button>
                    <div className="pinfo">
                      <div className="t ell" title={p.title}>{p.title}</div>
                      <div className="m">{fmtLong(p.duration)} · {p.scene_count} {plural(p.scene_count, "сцена", "сцены", "сцен")} · {ago(p.updated)}</div>
                      {createdOf(p) && <div className="m" data-testid="project-created">создан {fmtDate(createdOf(p)!)}</div>}
                      <div className="prog">
                        {busy && job ? <JobSteps job={job} compact /> : interrupted ? (
                          <span className="tag amber" data-testid="interrupted">Разбор прерван</span>
                        ) : p.has_analysis ? (
                          <span className="tag ok">разобран</span>
                        ) : job?.state === "failed" ? <span className="tag bad">разбор не удался</span> : <span className="tag">ждёт разбора</span>}
                        {busy && job && <div className="bar" style={{ marginTop: 6 }}><i style={{ width: Math.round(jobProgress(job) * 100) + "%" }} /></div>}
                      </div>
                    </div>
                    <div className="pact">
                      <button className="btn ghost sm iconbtn" aria-label={`Удалить проект ${p.title}`} title="Убрать из списка (видео на диске не трогаем)" onClick={() => void removeProject(p)}><Trash2 size={14} /></button>
                      {interrupted && <button className="btn amber" onClick={async () => { await reanalyze(p.id); nav(`/p/${p.id}/analyze`); }}><RotateCw size={13} />Продолжить</button>}
                      {!interrupted && <button className="btn primary" onClick={() => nav(`/p/${p.id}/analyze`)}>{busy ? "Смотреть" : "Открыть"}</button>}
                    </div>
                  </article>
                );
              })}
              <button className="pcard padd" onClick={pick} aria-label="Добавить ещё видео">
                <Plus size={22} /><b>Добавить видео</b><span className="note">Перетащи файл или нажми сюда</span>
              </button>
            </div>
          )}
          {projects.length <= 4 && (
            <div className="howto" data-testid="howto">
              <h2>Как это устроено</h2>
              <ol>
                <li><b>1 · Разбор</b><span>Программа читает видео: текст, голоса, склейки. Один раз на проект.</span></li>
                <li><b>2 · Запись</b><span>Выбираешь момент, пишешь свой голос под картинку — или загружаешь готовый файл.</span></li>
                <li><b>3 · Сведение</b><span>Громкость голосов и фона выравнивается сама. Можно подправить руками.</span></li>
                <li><b>4 · Готовый ролик</b><span>Собираем видео со звуком под YouTube, Reels или Telegram (в программе это шаг «Рендер»).</span></li>
              </ol>
            </div>
          )}
          <div className="aboutfoot"><button type="button" className="link" onClick={() => nav("/settings/about")} data-testid="about-link">О программе</button></div>
        </div>
      </div>
      <MicCheck open={checkOpen} onClose={() => setCheckOpen(false)} />
    </section>
  );
}
