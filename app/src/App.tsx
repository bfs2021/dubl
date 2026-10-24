import { useEffect, useRef, useState } from "react";
import { Navigate, Route, Routes, useLocation, useNavigate, useParams } from "react-router-dom";
import { Check, FolderOpen, Pencil, Settings as SettingsIcon } from "lucide-react";
import { useStore } from "./state/store";
import { ToastHost } from "./components/Toast";
import { ConfirmHost } from "./components/ConfirmDialog";
import { ProjectsScreen } from "./screens/Projects";
import { AnalyzeScreen } from "./screens/Analyze";
import { RecordScreen } from "./screens/Record";
import { MixScreen } from "./screens/Mix";
import { RenderScreen } from "./screens/Render";
import { SettingsScreen } from "./screens/Settings";
import { EmptyState } from "./components/EmptyState";
import { plural } from "./lib/format";
import { aiPillTitle, connectedCount, useRecheckOnFocus } from "./lib/ai";
import { useStemsWait } from "./lib/stems";
import type { AppSettings } from "./api/types";

const STEPS = [
  { key: "analyze", n: 1, title: "Разбор", hint: "Программа читает видео: текст, голоса, склейки" },
  { key: "record", n: 2, title: "Запись", hint: "Записать свой голос" },
  { key: "mix", n: 3, title: "Сведение", hint: "Выровнять громкость голосов и фона" },
  { key: "render", n: 4, title: "Рендер", hint: "Собрать готовый ролик" },
] as const;

/** Название проекта в шапке: карандаш рядом — переименовать (PATCH /projects/{id}). */
export function ProjectTitle({ pid, title }: { pid: string; title: string | undefined }) {
  const rename = useStore((s) => s.renameProject);
  const [edit, setEdit] = useState(false);
  const [v, setV] = useState(title ?? "");
  const ref = useRef<HTMLInputElement>(null);
  useEffect(() => { setV(title ?? ""); }, [title]);
  useEffect(() => { if (edit) ref.current?.select(); }, [edit]);
  const commit = async () => {
    setEdit(false);
    if (v.trim() && v.trim() !== title) await rename(pid, v.trim());
    else setV(title ?? "");
  };
  if (!edit) {
    return (
      <>
        <span className="ell crumbTitle" title={title}>{title ?? "…"}</span>
        <button className="btn ghost sm iconbtn renamebtn" aria-label="Переименовать проект" title="Переименовать проект" onClick={() => setEdit(true)} disabled={!title}><Pencil size={12} /></button>
      </>
    );
  }
  return (
    <span className="titleedit">
      <input ref={ref} className="inp" value={v} aria-label="Название проекта" onChange={(e) => setV(e.target.value)} onBlur={() => void commit()}
        onKeyDown={(e) => { if (e.key === "Enter") void commit(); else if (e.key === "Escape") { setV(title ?? ""); setEdit(false); } }} />
      <button className="btn ghost sm iconbtn" aria-label="Сохранить название" onMouseDown={(e) => e.preventDefault()} onClick={() => void commit()}><Check size={13} /></button>
    </span>
  );
}

/** Текст про ИИ в шапке. null — настройки ещё не пришли: показываем нейтральное «ИИ…», а не ложное «не подключён».
 * Сколько моделей и кто не отвечает — не пугаем любителя: подробности в подсказке и в Настройках. */
export function aiPillText(settings: unknown, connected: number): string {
  if (!settings) return "ИИ…";
  return connected ? "ИИ подключён" : "ИИ не подключён";
}

/** Подсказка к плашке ИИ: сколько моделей отвечает и кто нет (сама плашка — коротко). */
export function aiPillHint(settings: AppSettings | null, connected: number, demo = false): string {
  if (!settings) return "Модели ИИ";
  const head = connected ? `Отвечает ${connected} ${plural(connected, "модель", "модели", "моделей")}.` : "Ни одна модель ИИ не отвечает — подключи её в Настройках.";
  const bad = aiPillTitle(settings, demo);
  return bad === "Модели ИИ" ? head : `${head} ${bad}. Подробнее — в Настройках.`;
}

/** Сколько моделей включено, но не отвечает: на плашке показываем отдельной пометкой, чтобы число не казалось обманом. */
export function aiPillNote(settings: { llm: { providers: { enabled: boolean; can_answer?: boolean | null }[] } } | null): string {
  if (!settings) return "";
  const bad = settings.llm.providers.filter((p) => p.enabled && p.can_answer === false).length;
  return bad ? ` · ${bad} не ${bad === 1 ? "отвечает" : "отвечают"}` : "";
}

function TopBar() {
  const nav = useNavigate();
  const loc = useLocation();
  const project = useStore((s) => s.project);
  const settings = useStore((s) => s.settings);
  const ui = useStore((s) => s.ui);
  const demo = useStore((s) => s.demo);
  const setBackTo = useStore((s) => s.setBackTo);
  const wait = useStemsWait();
  const m = loc.pathname.match(/^\/p\/([^/]+)\/(\w+)/);
  const pid = m?.[1];
  const step = m?.[2];
  const connected = connectedCount(settings, demo);
  useRecheckOnFocus(!demo);       // вернулся в окно — заново узнаём, какие модели могут ответить (вошёл в Claude, поставил Codex…)
  const onProjects = loc.pathname === "/";
  const inSettings = loc.pathname.startsWith("/settings");

  // «← Назад» в Настройках ведёт на последний экран, который не был настройками (в проект — на тот же шаг)
  useEffect(() => { if (!loc.pathname.startsWith("/settings")) setBackTo(loc.pathname); }, [loc.pathname, setBackTo]);

  return (
    <header className="top">
      <button className="brand" onClick={() => nav("/")} aria-label="Дубль — к проектам"><b />Дубль</button>
      <nav className="crumb" aria-label="Где я">
        <button className={"navbtn" + (onProjects ? " on" : "")} onClick={() => nav("/")} aria-current={onProjects ? "page" : undefined} title="Все проекты">
          <FolderOpen size={14} />Проекты
        </button>
        {pid && (
          <>
            <span className="sep">/</span>
            <ProjectTitle pid={pid} title={project?.title} />
          </>
        )}
      </nav>
      {pid && (
        <nav className="tabs" aria-label="Этапы">
          {STEPS.map((s) => {
            // шаги 2–4 открываются, когда у выбранной сцены готовы голоса и фон (иначе движок ответит «сцена ещё готовится»)
            const locked = s.n > 1 && wait.blocked && step !== s.key;
            return (
              <button key={s.key} className={"tab" + (step === s.key ? " on" : "") + (locked ? " locked" : "")} onClick={() => nav(`/p/${pid}/${s.key}`)} aria-current={step === s.key ? "page" : undefined}
                title={locked ? wait.hint : s.hint} aria-label={`Шаг ${s.n}: ${s.title}`} disabled={locked}>
                <span className="n" aria-hidden="true">{s.n}</span>{s.title}
              </button>
            );
          })}
        </nav>
      )}
      <span className="grow" />
      <button className={"pill" + (loc.pathname === "/settings/processing" ? " on" : "")} onClick={() => nav("/settings/processing")} title="Где идёт обработка">
        <span className="dot" style={{ background: "var(--ok)" }} />{ui.processing === "remote" ? "Свой сервер" : "Этот компьютер"}
      </button>
      <button className={"pill" + (loc.pathname === "/settings/ai" ? " on" : "")} onClick={() => nav("/settings/ai")} title={aiPillHint(settings, connected, demo)} data-testid="ai-pill">
        <span className="dot" style={{ background: !settings ? "var(--dim)" : connected ? "var(--ok)" : "var(--dim)" }} />
        {aiPillText(settings, connected)}
      </button>
      <button className={"pill" + (inSettings ? " on" : "")} onClick={() => nav("/settings")} aria-current={inSettings ? "page" : undefined}><SettingsIcon size={13} />Настройки</button>
    </header>
  );
}

/** Открывает проект из адреса и показывает экран, когда он загружен. */
function ProjectGate({ children }: { children: (pid: string) => JSX.Element }) {
  const { pid = "" } = useParams();
  const engine = useStore((s) => s.engine);
  const project = useStore((s) => s.project);
  const openProject = useStore((s) => s.openProject);
  const nav = useNavigate();
  useEffect(() => { if (engine && pid) void openProject(pid); }, [engine, pid, openProject]);
  if (!engine) return <div className="screen center">Подключаюсь к движку…</div>;
  if (!project || project.id !== pid) {
    return (
      <div className="screen center">
        <EmptyState title="Открываю проект…"><button className="btn" onClick={() => nav("/")}>К проектам</button></EmptyState>
      </div>
    );
  }
  return children(pid);
}

export function App() {
  const connect = useStore((s) => s.connect);
  const connecting = useStore((s) => s.connecting);
  const demo = useStore((s) => s.demo);
  const note = useStore((s) => s.demoNote);
  const engine = useStore((s) => s.engine);
  const initMl = useStore((s) => s.initMl);

  useEffect(() => { void initMl(); }, [initMl]);
  useEffect(() => {
    if (!useStore.getState().engine) void connect({ forceMock: new URLSearchParams(location.search).has("demo") });
  }, [connect]);

  return (
    <div className="app">
      <TopBar />
      {demo && (
        <div className="demo" role="status">
          <b>Демо-режим: движок не подключён.</b>
          <span>Всё сохраняется только пока окно открыто{note ? ` · ${note}` : ""}.</span>
          <button className="link" onClick={() => void connect()}>Подключиться снова</button>
        </div>
      )}
      <main className="main">
        {connecting && !engine ? (
          <div className="screen center">Подключаюсь к движку…</div>
        ) : (
          <Routes>
            <Route path="/" element={<ProjectsScreen />} />
            <Route path="/p/:pid/analyze" element={<ProjectGate>{() => <AnalyzeScreen />}</ProjectGate>} />
            <Route path="/p/:pid/record" element={<ProjectGate>{() => <RecordScreen />}</ProjectGate>} />
            <Route path="/p/:pid/mix" element={<ProjectGate>{() => <MixScreen />}</ProjectGate>} />
            <Route path="/p/:pid/render" element={<ProjectGate>{() => <RenderScreen />}</ProjectGate>} />
            <Route path="/settings/:page?" element={<SettingsScreen />} />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        )}
      </main>
      <ToastHost />
      <ConfirmHost />
    </div>
  );
}
