import { useEffect, useRef, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { ArrowLeft, Bot, CheckCircle2, Cpu, Info, KeyRound, LogIn, Mic, Plus, RefreshCw } from "lucide-react";
import { useStore } from "../state/store";
import { Segmented } from "../components/Segmented";
import { Switch } from "../components/Switch";
import { Meter } from "../components/Meter";
import { LLM_TASKS } from "../api/presets";
import { CALIBRATION_MS, MicRecorder, calibrateLatency, levelStep, listMicrophones, micErrorText, type CalibrationStep, type LevelReading } from "../audio/recorder";
import { canAnswer, isCliKind, openLogin, providerHealth, recheckAi, routeNote, useRecheckOnFocus } from "../lib/ai";
import { MicCheck } from "../components/MicCheck";
import { Help } from "../components/Help";
import { Disclosure } from "../components/Disclosure";
import { deviceName } from "../lib/format";
import { MlPackBanner } from "../components/MlPackBanner";
import { openExternal } from "../lib/external";
import { AboutPage } from "./About";
import type { AppSettings, LLMTask, ProviderDraft, ProviderKind, ProviderState } from "../api/types";

type Page = "ai" | "processing" | "mic" | "about";
const PAGES: { id: Page; title: string; icon: typeof Bot }[] = [
  { id: "ai", title: "Модели ИИ", icon: Bot },
  { id: "processing", title: "Обработка", icon: Cpu },
  { id: "mic", title: "Микрофон", icon: Mic },
  { id: "about", title: "О программе", icon: Info },
];

export function SettingsScreen() {
  const nav = useNavigate();
  const { page } = useParams();
  const cur: Page = (PAGES.find((p) => p.id === page)?.id ?? "ai") as Page;
  const settings = useStore((s) => s.settings);
  const backTo = useStore((s) => s.backTo);
  const projectTitle = useStore((s) => s.project?.title);
  const hasConfirm = useStore((s) => !!s.dialog);
  const inProject = backTo.startsWith("/p/");
  const backLabel = inProject ? "Назад к проекту" : "Назад к проектам";
  const goBack = () => nav(backTo || "/");

  // Esc = «Назад» (если не набираем текст, не открыта подсказка «?» и не висит окно подтверждения)
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== "Escape" || e.defaultPrevented || hasConfirm) return;
      const el = e.target as HTMLElement | null;
      if (el && /INPUT|TEXTAREA|SELECT/.test(el.tagName)) { el.blur(); return; }
      if (document.querySelector(".helppop")) return;
      nav(backTo || "/");
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [nav, backTo, hasConfirm]);

  return (
    <section className="screen settings">
      <nav className="panel snav" aria-label="Разделы настроек">
        <button className="backbtn" onClick={goBack} data-testid="settings-back" title={`${backLabel}${inProject && projectTitle ? ` «${projectTitle}»` : ""} (Esc)`}>
          <ArrowLeft size={16} /><span className="bt"><b>{backLabel}</b>{inProject && projectTitle && <small className="ell">{projectTitle}</small>}</span>
        </button>
        <div className="snav-sep" />
        {PAGES.map((p) => (
          <button key={p.id} className={cur === p.id ? "on" : ""} aria-current={cur === p.id ? "page" : undefined} onClick={() => nav(`/settings/${p.id}`)}>
            <p.icon size={15} />{p.title}
          </button>
        ))}
      </nav>
      <div className="panel swrap">
        {cur === "about" && <AboutPage />}
        {cur !== "about" && !settings && <div className="note">Настройки загружаются…</div>}
        {settings && cur === "ai" && <AiPage settings={settings} />}
        {settings && cur === "processing" && <ProcessingPage settings={settings} />}
        {settings && cur === "mic" && <MicPage settings={settings} />}
      </div>
    </section>
  );
}

// ─────────────────────────── Модели ИИ ───────────────────────────
interface Preset {
  id: string; kind: ProviderKind; title: string; desc: string; model: string; base_url: string | null; needsKey: boolean;
  /** где взять ключ */
  keyUrl?: string; keyHow?: string; recommended?: string;
  /** прячем в «Для опытных» */
  advanced?: boolean;
  /** известные модели сервиса — выбор из списка вместо поля ввода */
  models?: { id: string; title: string }[];
}
const PRESETS: Preset[] = [
  { id: "claude_cli", kind: "claude_cli", title: "Claude (через Claude Code)", desc: "Работает по твоей подписке Claude, ключ не нужен. Нужна установленная программа Claude Code.", model: "sonnet", base_url: null, needsKey: false,
    keyUrl: "https://claude.com/product/claude-code", keyHow: "Как установить Claude Code",
    models: [
      { id: "haiku", title: "Haiku — самая быстрая, меньше тратит лимит подписки" },
      { id: "sonnet", title: "Sonnet — лучше переводит шутки, ответ за пару секунд (рекомендуем)" },
      { id: "opus", title: "Opus — самая умная, но медленнее и быстрее съедает лимит" },
    ] },
  { id: "deepseek", kind: "deepseek", title: "DeepSeek", desc: "Дёшево и неплохо переводит. Нужен ключ с platform.deepseek.com.", model: "deepseek-flash", base_url: "https://api.deepseek.com", needsKey: true,
    keyUrl: "https://platform.deepseek.com/api_keys", keyHow: "Где взять ключ DeepSeek",
    models: [
      { id: "deepseek-flash", title: "Flash — быстро и дёшево, для перевода хватает" },
      { id: "deepseek-v4-pro", title: "Pro — умнее, дороже: сложные шутки и игра слов" },
    ] },
  { id: "gemini", kind: "gemini", title: "Gemini", desc: "Google AI Studio, есть бесплатный лимит. Нужен ключ — берётся за минуту, карта не нужна.", model: "gemini-3.5-flash-lite", base_url: null, needsKey: true,
    keyUrl: "https://aistudio.google.com/apikey", keyHow: "Где взять ключ Gemini", recommended: "Рекомендуем, если ты новичок: бесплатно и быстро",
    models: [
      { id: "gemini-3.5-flash-lite", title: "3.5 Flash-Lite — быстро и дёшево, для перевода хватает" },
      { id: "gemini-3.6-flash", title: "3.6 Flash — умнее, для шуток и игры слов" },
      { id: "gemini-3.8-flash", title: "3.8 Flash — самая умная из быстрых" },
    ] },
  { id: "codex_cli", kind: "codex_cli", title: "Codex (через ChatGPT)", desc: "Работает по твоей подписке ChatGPT, ключ не нужен. Нужна установленная программа Codex CLI.", model: "default", base_url: null, needsKey: false, advanced: true,
    keyUrl: "https://developers.openai.com/codex/cli", keyHow: "Как установить Codex",
    models: [{ id: "default", title: "Как настроено в самом Codex" }] },
  { id: "openai", kind: "openai_compat", title: "OpenAI-совместимый", desc: "OpenAI, OpenRouter или свой сервер: любой сервис с совместимым адресом (…/chat/completions).", model: "gpt-4o-mini", base_url: "https://api.openai.com/v1", needsKey: true, advanced: true,
    keyUrl: "https://platform.openai.com/api-keys", keyHow: "Где взять ключ OpenAI" },
];

function AiPage({ settings }: { settings: AppSettings }) {
  const save = useStore((s) => s.saveSettings);
  const toast = useStore((s) => s.toast);
  const demo = useStore((s) => s.demo);
  const [open, setOpen] = useState<string | null>(null);
  // открыл настройки или вернулся в окно (после входа в терминале) — заново узнаём, что установлено и где выполнен вход
  useEffect(() => { void recheckAi(); }, []);
  useRecheckOnFocus();

  const cfgOf = (id: string) => settings.llm.providers.find((p) => p.id === id);
  const connectedIds = settings.llm.providers.filter((p) => canAnswer(providerHealth(p, demo))).map((p) => p.id);

  const upsert = async (cfg: ProviderDraft) => {
    const list: ProviderState[] = settings.llm.providers.some((p) => p.id === cfg.id)
      ? settings.llm.providers.map((p) => (p.id === cfg.id ? { ...p, ...cfg } : p))
      : [...settings.llm.providers, { ...cfg, key_set: false, key_status: "unknown" as const }];
    await save({ ...settings, llm: { ...settings.llm, providers: list } });
  };
  const setRoute = async (task: LLMTask, id: string) => {
    await save({ ...settings, llm: { ...settings.llm, routing: { ...settings.llm.routing, [task]: id || null } } });
  };
  const card = (pr: Preset) => (
    <ProviderCard key={pr.id} preset={pr} cfg={cfgOf(pr.id)} demo={demo} open={open === pr.id} onToggle={() => setOpen(open === pr.id ? null : pr.id)}
      onSave={async (c) => { await upsert(c); toast("Сохранено.", "ok"); }} />
  );
  const advanced = PRESETS.filter((p) => p.advanced);

  return (
    <div>
      <h1>Модели ИИ</h1>
      <p className="lead">ИИ работает только по твоей кнопке: перевести, подогнать реплику под длину, подсказать имена, помочь режиссёру. Подключи свою модель — мы свои ключи не раздаём, а твои хранятся только на этом устройстве.</p>
      <p className="note" style={{ marginBottom: 12 }} data-testid="key-explain"><b>Ключ</b> — это длинный пароль из букв и цифр. Его выдаёт сервис нейросети бесплатно или за копейки, а программа с ним ходит к модели от твоего имени. Выбери модель, нажми «Где взять ключ» и вставь его сюда.</p>

      {PRESETS.filter((p) => !p.advanced).map(card)}

      {advanced.length > 0 && (
        <Disclosure summary="Для опытных: другие сервисы" defaultOpen={advanced.some((p) => cfgOf(p.id) && providerHealth(cfgOf(p.id), demo) !== "nokey")} testId="adv-services">
          {advanced.map(card)}
        </Disclosure>
      )}

      <div className="sect"><h2>Какая модель что делает</h2></div>
      <table className="route">
        <tbody>
          {LLM_TASKS.map((t) => (
            <tr key={t.id}>
              <td><div style={{ fontWeight: 600 }}>{t.title}</div><div className="d">{t.hint}</div></td>
              <td>
                <select className="inp" aria-label={t.title} value={settings.llm.routing[t.id] ?? ""} onChange={(e) => void setRoute(t.id, e.target.value)}>
                  <option value="">Не использовать</option>
                  {settings.llm.providers.filter((p) => connectedIds.includes(p.id) || p.id === settings.llm.routing[t.id])
                    .map((p) => <option key={p.id} value={p.id}>{p.title}{connectedIds.includes(p.id) ? "" : " — не отвечает"}</option>)}
                </select>
                {routeNote(settings, t.id) && <div className="d routenote" data-testid={`route-note-${t.id}`} style={{ color: "var(--warn)", marginTop: 4 }}>{routeNote(settings, t.id)}</div>}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      {connectedIds.length === 0 && <p className="note" style={{ marginTop: 8 }} data-testid="no-ai-note">Пока ни одна модель не может ответить — программа работает без ИИ, текст можно писать самому.</p>}
    </div>
  );
}

const STATUS_TEXT: Record<string, string> = { ready: "подключено", unchecked: "ключ сохранён, не проверен", invalid: "ключ не подходит", nokey: "не подключено", off: "выключено",
  notinstalled: "не найден", nologin: "нужен вход", error: "не удалось проверить" };
/** Понятный статус модели-подписки: что с программой и входом. */
const CLI_STATUS_TEXT: Record<string, string> = { ready: "установлен, вход выполнен, отвечает", notinstalled: "не найден", nologin: "установлен, нужен вход", error: "не удалось проверить", off: "выключено" };

function ProviderCard({ preset, cfg, demo, open, onToggle, onSave }: {
  preset: Preset; cfg?: ProviderState; demo: boolean; open: boolean; onToggle: () => void; onSave: (c: ProviderDraft) => Promise<void>;
}) {
  const engine = useStore((s) => s.engine)!;
  const setKey = useStore((s) => s.setKey);
  const [model, setModel] = useState(cfg?.model ?? preset.model);
  const [url, setUrl] = useState(cfg?.base_url ?? preset.base_url ?? "");
  const [key, setKeyV] = useState("");
  const [check, setCheck] = useState<{ ok: boolean; message: string } | null>(null);
  const [busy, setBusy] = useState(false);
  const health = preset.needsKey || cfg ? providerHealth(cfg ?? undefined, demo) : "nokey";
  const keySaved = !!cfg?.key_set || (preset.needsKey === false);
  const invalid = health === "invalid";
  const ok = canAnswer(health);
  const cli = isCliKind(preset.kind);
  const cliProblem = cli && !!cfg && (health === "notinstalled" || health === "nologin" || health === "error");
  const [loginMsg, setLoginMsg] = useState<{ ok: boolean; message: string } | null>(null);
  const [rechecking, setRechecking] = useState(false);

  const login = async () => {
    setBusy(true);
    setLoginMsg(await openLogin(preset.id));
    setBusy(false);
  };
  const recheck = async () => {
    setRechecking(true);
    await recheckAi();
    setRechecking(false);
  };

  const build = (enabled: boolean): ProviderDraft => ({ id: preset.id, kind: preset.kind, title: preset.title, base_url: url.trim() || null, model: model.trim() || preset.model, enabled });

  /** Сохраняет ключ и модель; движок сразу проверяет ключ. Возвращает результат проверки. */
  const saveAll = async () => {
    let res: { ok: boolean; message: string } | null = null;
    await onSave(build(true));
    if (key.trim()) {
      const r = await setKey(preset.id, key.trim());
      if (!r) return null;
      setKeyV("");
      res = r.key_status === "invalid"
        ? { ok: false, message: r.key_message || "Модель не приняла этот ключ." }
        : { ok: true, message: r.key_status === "ok" ? "Ключ принят" : "Ключ сохранён" };
      setCheck(res);
    }
    return res;
  };
  const test = async () => {
    setBusy(true);
    try {
      const saved = await saveAll();
      if (saved && !saved.ok) { setBusy(false); return; }
      setCheck(await engine.checkProvider(preset.id));
    } catch (e) {
      setCheck({ ok: false, message: (e as Error).message });
    }
    setBusy(false);
  };
  const save = async () => { setBusy(true); await saveAll(); setBusy(false); };

  const statusCls = ok ? "ok" : invalid || cliProblem ? "bad" : "none";
  const statusText = cli && cfg ? CLI_STATUS_TEXT[health] ?? STATUS_TEXT[health] : STATUS_TEXT[health];

  return (
    <div className={"prov" + (open ? " open" : "") + (invalid || cliProblem ? " bad" : "")} data-status={health} data-testid={`prov-${preset.id}`}>
      <div>
        <div className="t">{preset.title}{preset.recommended && <span className="tag amber" style={{ marginLeft: 8 }}>рекомендуем</span>}</div>
        <div className="d">{preset.desc}</div>
        {preset.recommended && <div className="d" style={{ color: "var(--amber)" }}>{preset.recommended}</div>}
        {invalid && <div className="d" style={{ color: "var(--bad)", marginTop: 4 }} role="alert" data-testid="key-message">{cfg?.key_message || "Модель не приняла ключ. Проверь его или возьми новый."}</div>}
        {cliProblem && cfg?.state_message && <div className="d" style={{ color: "var(--bad)", marginTop: 4 }} role="alert" data-testid="cli-message">{health === "notinstalled" ? `Не найден на этом компьютере. ${cfg.state_message.replace(/^Не найден — /, "").replace(/^./, (c) => c.toUpperCase())}` : cfg.state_message}</div>}
        {preset.keyUrl && (!cli || !ok) && <button type="button" className="link keylink" onClick={() => void openExternal(preset.keyUrl!)}>{preset.keyHow} ↗</button>}
      </div>
      <div className={"status " + statusCls}><i className="dot" />{statusText}</div>
      {health === "nologin" && cfg ? (
        <button className="btn amber" onClick={() => void login()} disabled={busy} data-testid={`login-${preset.id}`}><LogIn size={13} />Войти</button>
      ) : health === "notinstalled" && cfg ? (
        <button className="btn" onClick={() => void recheck()} disabled={rechecking} data-testid={`recheck-${preset.id}`}><RefreshCw size={13} />{rechecking ? "Ищу…" : "Найти ещё раз"}</button>
      ) : (
        <button className={"btn" + (invalid ? " amber" : "")} onClick={onToggle} aria-expanded={open}>{open ? "Свернуть" : invalid ? "Изменить ключ" : ok ? "Изменить" : <><Plus size={13} />Подключить</>}</button>
      )}
      {loginMsg && <div className="d full" style={{ gridColumn: "1 / 4", color: loginMsg.ok ? "var(--mute)" : "var(--bad)" }} role="status" data-testid="login-message">{loginMsg.message}</div>}
      {open && (
        <div className="provform">
          {preset.models ? (
            <label className="field"><span>Модель</span>
              <select className="inp" value={preset.models.some((m) => m.id === model) ? model : preset.model} onChange={(e) => setModel(e.target.value)} aria-label={`Модель: ${preset.title}`}>
                {preset.models.map((m) => <option key={m.id} value={m.id}>{m.title}</option>)}
              </select>
            </label>
          ) : (
            <label className="field"><span>Модель</span><input className="inp" value={model} onChange={(e) => setModel(e.target.value)} aria-label={`Модель: ${preset.title}`} /></label>
          )}
          {(preset.kind === "openai_compat" || preset.kind === "deepseek") && (
            <label className="field"><span>Адрес сервиса</span><input className="inp" value={url} onChange={(e) => setUrl(e.target.value)} aria-label={`Адрес: ${preset.title}`} /></label>
          )}
          {preset.needsKey && (
            <label className="field full"><span><KeyRound size={11} style={{ verticalAlign: -1 }} /> Ключ {keySaved ? "(уже сохранён — впиши новый, чтобы заменить)" : ""}</span>
              <input className="inp" type="password" autoComplete="off" value={key} onChange={(e) => setKeyV(e.target.value)} placeholder={keySaved ? "••••••••••••" : "вставь ключ сюда"} aria-label={`Ключ: ${preset.title}`} />
              <span className="note">Хранится в защищённой связке ключей этого компьютера. На серверы разработчиков не уходит. После «Сохранить» программа сразу проверит ключ.</span></label>
          )}
          <div className="row full">
            <button className="btn primary" onClick={() => void save()} disabled={busy || (preset.needsKey && !keySaved && !key.trim())}>{busy ? "Проверяю…" : "Сохранить"}</button>
            <button className="btn" onClick={test} disabled={busy || (preset.needsKey && !keySaved && !key.trim())}>Проверить</button>
            {cli && cfg && <button className="btn" onClick={() => void login()} disabled={busy}><LogIn size={13} />Войти</button>}
            {cli && cfg && <button className="btn ghost" onClick={() => void recheck()} disabled={rechecking}><RefreshCw size={13} />Проверить статус</button>}
            {cfg && <button className="btn ghost" onClick={() => void onSave({ id: cfg.id, kind: cfg.kind, title: cfg.title, base_url: cfg.base_url, model: cfg.model, enabled: !cfg.enabled })}>{cfg.enabled ? "Выключить" : "Включить"}</button>}
            {check && <span className={"status " + (check.ok ? "ok" : "")} role="status" style={check.ok ? undefined : { color: "var(--bad)" }}>{check.ok && <CheckCircle2 size={13} />}{check.message}{demo && check.ok ? " (демо)" : ""}</span>}
          </div>
        </div>
      )}
    </div>
  );
}

// ─────────────────────────── Обработка ───────────────────────────
function ProcessingPage({ settings }: { settings: AppSettings }) {
  const ui = useStore((s) => s.ui);
  const setUi = useStore((s) => s.setUi);
  const save = useStore((s) => s.saveSettings);
  const connect = useStore((s) => s.connect);
  const health = useStore((s) => s.health);
  const demo = useStore((s) => s.demo);
  const toast = useStore((s) => s.toast);
  const [url, setUrl] = useState(ui.remote_url);
  const [token, setToken] = useState(ui.remote_token);

  const applyRemote = async () => {
    setUi({ processing: "remote", remote_url: url.trim(), remote_token: token.trim() });
    await connect();
    const st = useStore.getState();
    toast(st.demo ? "Сервер не отвечает — работаю в демо-режиме." : "Подключился к серверу.", st.demo ? "error" : "ok");
  };
  const useLocal = async () => {
    setUi({ processing: "local" });
    await connect();
  };

  return (
    <div>
      <h1>Обработка</h1>
      <p className="lead">Где считать: разбор, разделение голосов и рендер — тяжёлая работа. Можно на этом компьютере или на своём сервере с мощной видеокартой.</p>

      <div className={"radio" + (ui.processing === "local" ? " on" : "")} role="radio" aria-checked={ui.processing === "local"} tabIndex={0} onClick={() => ui.processing !== "local" && void useLocal()}
        onKeyDown={(e) => { if ((e.key === "Enter" || e.key === " ") && ui.processing !== "local") void useLocal(); }}>
        <span className="rb" /><div><div className="t">Этот компьютер</div>
          <div className="d">{demo ? "Движок не запущен — идёт демо." : `Движок работает рядом с программой${health ? ` · ${deviceName(health.device)} · версия ${health.version}` : ""}.`}</div></div>
      </div>
      <div className={"radio" + (ui.processing === "remote" ? " on" : "")} role="radio" aria-checked={ui.processing === "remote"} tabIndex={0}
        onClick={() => ui.processing !== "remote" && setUi({ processing: "remote" })} onKeyDown={(e) => { if ((e.key === "Enter" || e.key === " ") && ui.processing !== "remote") setUi({ processing: "remote" }); }}>
        <span className="rb" /><div><div className="t">Свой сервер</div><div className="d">Движок Дубля запущен на другом компьютере. Нужны адрес и токен.</div></div>
        {ui.processing === "remote" && (
          <div className="x" onClick={(e) => e.stopPropagation()} onKeyDown={(e) => e.stopPropagation()}>
            <input className="inp" placeholder="http://192.168.1.20:8765" aria-label="Адрес сервера" value={url} onChange={(e) => setUrl(e.target.value)} />
            <input className="inp" type="password" autoComplete="off" placeholder="Токен" aria-label="Токен сервера" value={token} onChange={(e) => setToken(e.target.value)} />
            <button className="btn primary" onClick={applyRemote} disabled={!url.trim()}>Подключить</button>
          </div>
        )}
      </div>

      <div className="sect"><h2>Нейросети</h2></div>
      <p className="note" style={{ marginBottom: 8 }}>Распознавание речи, отделение голосов от музыки и определение, кто говорит. В установщике их нет — докачиваются один раз.</p>
      {demo ? <div className="note">В демо-режиме нейросети не нужны.</div> : <MlPackBanner always lead="Чтобы распознавать речь и отделять голоса от музыки, нужно один раз скачать нейросети (~1,3 ГБ, несколько минут)." />}

      <div className="sect"><h2>Параметры расчёта</h2></div>
      <div className="mrow"><span>Устройство</span>
        <select className="inp" aria-label="Устройство" value={settings.processing_device} onChange={(e) => void save({ ...settings, processing_device: e.target.value })}>
          <option value="auto">Выбрать само</option><option value="cpu">Процессор</option><option value="mps">Видеокарта Mac (Apple)</option><option value="cuda">Видеокарта NVIDIA</option>
        </select><span /></div>
      <div className="mrow"><span>Распознавание речи</span>
        <select className="inp" aria-label="Распознавание речи" value={settings.asr_engine ?? "auto"} onChange={(e) => void save({ ...settings, asr_engine: e.target.value as AsrEngine })}>
          <option value="auto">Авто (рекомендуется)</option><option value="whisper">Whisper</option><option value="gigaam">GigaAM — только русский</option>
        </select><span /></div>
      <AsrExplain engine={settings.asr_engine ?? "auto"} />
      <div className="mrow"><span>Модель Whisper</span>
        <select className="inp" aria-label="Модель распознавания" value={settings.whisper_model} onChange={(e) => void save({ ...settings, whisper_model: e.target.value })}>
          <option value="large-v3-turbo">Точная (large-v3-turbo)</option><option value="small">Быстрая (small)</option>
        </select><span /></div>
      <div className="mrow"><span>Разделение голоса и фона</span>
        <Segmented label="Качество разделения" value={settings.separation_quality} onChange={(v) => void save({ ...settings, separation_quality: v })}
          options={[{ value: "best", label: "Лучшее" }, { value: "fast", label: "Быстрое" }]} /><span /></div>
    </div>
  );
}

type AsrEngine = AppSettings["asr_engine"];

/** Что выбрать любителю и скачана ли модель GigaAM. Смена действует на новый разбор. */
function AsrExplain({ engine }: { engine: AsrEngine }) {
  const caps = useStore((s) => s.health?.capabilities);
  const demo = useStore((s) => s.demo);
  const what = engine === "whisper"
    ? "Whisper распознаёт любой язык."
    : engine === "gigaam"
      ? "GigaAM слушает только русскую речь. Если речь на другом языке, программа всё равно возьмёт Whisper."
      : "Русскую речь распознаёт GigaAM, остальные языки — Whisper. Если не уверен, оставь так.";
  let model = "";
  if (!demo && caps && engine !== "whisper") {
    const mb = caps.gigaam_download_mb ?? 230;
    model = caps.ml === false
      ? ""
      : caps.gigaam === false
      ? " GigaAM пока не установлен — он поставится при следующем запуске программы с интернетом. До тех пор работает Whisper."
      : caps.gigaam_downloaded
        ? " Модель GigaAM скачана."
        : ` Модель GigaAM скачается при первом разборе русской речи, ~${mb} МБ.`;
  }
  return (
    <p className="note" style={{ margin: "6px 0 4px" }} data-testid="asr-explain">
      {what} GigaAM точнее ставит начало и конец русских реплик — дубль легче попадает в губы.{model} Смена действует на новый разбор: у готового проекта нажми «Разобрать снова».
    </p>
  );
}

// ─────────────────────────── Микрофон ───────────────────────────
function MicPage({ settings }: { settings: AppSettings }) {
  const ui = useStore((s) => s.ui);
  const setUi = useStore((s) => s.setUi);
  const save = useStore((s) => s.saveSettings);
  const toast = useStore((s) => s.toast);
  const [devices, setDevices] = useState<MediaDeviceInfo[]>([]);
  const [level, setLevel] = useState(-120);
  const [heard, setHeard] = useState(false);
  const [on, setOn] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [cal, setCal] = useState<{ step: CalibrationStep | "done"; t: number } | null>(null);
  const [calRes, setCalRes] = useState<{ ok: boolean; text: string } | null>(null);
  const [checkOpen, setCheckOpen] = useState(false);
  const rec = useRef(new MicRecorder());

  const start = async (id: string | null) => {
    await rec.current.close();
    rec.current = new MicRecorder();
    setErr(null);
    setHeard(false);
    try {
      await rec.current.open(id);
      setOn(true);
      setDevices(await listMicrophones());
    } catch (e) {
      setOn(false);
      setErr(micErrorText(e));
    }
  };
  useEffect(() => {
    const r = rec.current;
    void listMicrophones().then(setDevices);
    // разрешение уже дано — шкала живая сразу; иначе человек сам нажмёт «Включить микрофон» (системный запрос не выскакивает без спроса)
    let dead = false;
    void (async () => {
      try {
        const st = await (navigator.permissions as Permissions | undefined)?.query({ name: "microphone" as PermissionName });
        if (st?.state === "granted" && !dead) await start(settings.input_device ?? null);
      } catch { /* нет Permissions API — останется кнопка */ }
    })();
    return () => { dead = true; void r.close(); void rec.current.close(); };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
  // живая шкала: пока проверка включена, читаем уровень ~30 раз в секунду
  useEffect(() => {
    if (!on) { setLevel(-120); return; }
    let r: LevelReading = { db: -120, peak: -120, heard: false };
    let last = performance.now();
    const iv = setInterval(() => {
      const now = performance.now();
      r = levelStep(r, rec.current.getLevel(), now - last);
      last = now;
      setLevel(r.db);
      if (r.heard) setHeard(true);
    }, 33);
    return () => clearInterval(iv);
  }, [on]);

  const calibrate = async () => {
    setCalRes(null);
    setCal({ step: "open", t: 0 });
    await rec.current.close();
    setOn(false);
    const t0 = performance.now();
    const tick = setInterval(() => setCal((c) => (c && c.step !== "done" ? { ...c, t: Math.min(1, (performance.now() - t0) / CALIBRATION_MS) } : c)), 80);
    try {
      const ms = await calibrateLatency(settings.input_device, (step) => setCal({ step, t: Math.min(1, (performance.now() - t0) / CALIBRATION_MS) }));
      await save({ ...settings, latency_ms: ms });
      setCalRes({ ok: true, text: `Готово: задержка ${ms} мс. Сохранил — дубли будут сдвигаться на эту величину.` });
      toast(`Задержка: ${ms} мс. Сохранил.`, "ok");
    } catch (e) {
      setCalRes({ ok: false, text: (e as Error).message });
    }
    clearInterval(tick);
    setCal(null);
  };
  const calText = cal?.step === "open" ? "Включаю микрофон…" : cal?.step === "click" ? "Играю щелчок…" : cal?.step === "listen" ? "Слушаю, вернулся ли щелчок…" : "";

  return (
    <div>
      <h1>Микрофон</h1>
      <p className="lead">Один раз настрой — и записывай. Скажи что-нибудь: шкала должна двигаться и доходить до зелёной зоны, но не краснеть.</p>
      <div className="row" style={{ marginBottom: 10 }}>
        <button className="btn primary" data-testid="mic-check-open" onClick={async () => { await rec.current.close(); setOn(false); setCheckOpen(true); }}><Mic size={13} />Проверить микрофон</button>
        <span className="note">Десять секунд: тишина, фраза вслух, щелчок. Покажет, хорошо ли слышно и сколько шума.</span>
      </div>
      <MicCheck open={checkOpen} onClose={() => setCheckOpen(false)} />

      <div className="mrow"><span>Микрофон</span>
        <select className="inp" aria-label="Микрофон" value={settings.input_device ?? ""} onChange={(e) => { void save({ ...settings, input_device: e.target.value || null }); if (on) void start(e.target.value || null); }}>
          <option value="">По умолчанию</option>
          {devices.map((d, i) => <option key={d.deviceId || i} value={d.deviceId}>{d.label || `Микрофон ${i + 1}`}</option>)}
        </select><span /></div>
      <div className="mrow"><span>Уровень</span>
        <div className="livemeter" data-testid="mic-meter">
          {on ? (
            <>
              <Meter db={level} />
              <div className="lbl"><span>{heard ? <span className="heard">Слышу тебя ✓</span> : "Скажи что-нибудь — шкала должна двигаться"}</span><span className="mono">{Math.max(-60, Math.round(level))} дБ</span></div>
            </>
          ) : err ? <span className="calres bad" role="alert">{err}</span> : <span className="note">Микрофон выключен. Нажми «Включить микрофон» и скажи что-нибудь</span>}
        </div>
        <button className="btn" onClick={() => (on ? void rec.current.close().then(() => setOn(false)) : void start(settings.input_device ?? null))}>{on ? "Выключить" : <><Mic size={13} />Включить микрофон</>}</button></div>
      <div className="mrow"><span>Задержка записи<Help label="Что такое задержка">Между тем, как ты что-то сказал, и тем, как это записалось, проходит доля секунды — звуковая карта опаздывает. Программа измерит её и сдвинет твои дубли ровно на эту величину.</Help></span>
        <div>
          <span className="mono" data-testid="latency">{Math.round(settings.latency_ms)} мс</span>{" "}
          {cal ? (
            <div className="livemeter" style={{ marginTop: 6 }} data-testid="cal-progress"><div className="calbar"><i style={{ width: Math.round(cal.t * 100) + "%" }} /></div><div className="lbl"><span>{calText}</span></div></div>
          ) : calRes ? (
            <span className={"calres " + (calRes.ok ? "ok" : "bad")} role="status" data-testid="cal-result">{calRes.ok && <CheckCircle2 size={13} />}{calRes.text}</span>
          ) : settings.latency_ms > 0 ? <span className="note">— измерено ранее</span> : <span className="note">— ещё не измеряли</span>}
        </div>
        <button className="btn" onClick={calibrate} disabled={!!cal}>{cal ? "Измеряю…" : "Откалибровать"}</button></div>
      <p className="note" style={{ margin: "4px 0 10px" }}>Калибровка играет щелчок через колонки и слушает его микрофоном. Для неё наушники нужно снять на пару секунд — включи колонки погромче. Потом надень обратно: для самой записи наушники нужны.</p>
      <div className="mrow"><span>Я записываю в наушниках</span>
        <span className="note">Во время записи — в наушниках, иначе микрофон запишет звук видео вместе с твоим голосом.</span>
        <Switch label="Я записываю в наушниках" checked={ui.headphones} onChange={(v) => setUi({ headphones: v })} /></div>
      <div className="mrow"><span>Слышать оригинал в наушниках</span>
        <input type="range" className="rng" min={0} max={1} step={0.05} value={ui.original_level} aria-label="Громкость оригинала" onChange={(e) => setUi({ original_level: Number(e.target.value) })} /><span className="mono mute">{Math.round(ui.original_level * 100)}%</span></div>
    </div>
  );
}
