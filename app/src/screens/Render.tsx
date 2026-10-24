import { useEffect, useMemo, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { CheckCircle2, Copy, ExternalLink, FolderOpen, Film } from "lucide-react";
import { useStore } from "../state/store";
import { Player, usePlayerTick } from "../lib/player";
import { VideoPlayer, Transport } from "../components/VideoPlayer";
import { Segmented } from "../components/Segmented";
import { Switch } from "../components/Switch";
import { JobSteps } from "../components/JobSteps";
import { EmptyState } from "../components/EmptyState";
import { Help } from "../components/Help";
import { copyText, openFile, revealFile } from "../lib/external";
import { defaultOutputPath, dirOf, pickSavePath } from "../lib/files";
import { fmtRel, fmtSec, plural } from "../lib/format";
import { inTauri } from "../api/client";
import type { RenderAspect, RenderFit, SubtitleMode, SubtitleText } from "../api/types";
import type { TakeX } from "../lib/takeReview";

const FORMATS: { id: RenderAspect; title: string; desc: string; w: number; h: number }[] = [
  { id: "16:9", title: "16:9", desc: "YouTube, VK", w: 46, h: 26 },
  { id: "9:16", title: "9:16", desc: "Reels, Shorts, TikTok", w: 15, h: 26 },
  { id: "1:1", title: "1:1", desc: "Telegram", w: 26, h: 26 },
];

export function RenderScreen() {
  const project = useStore((s) => s.project);
  if (!project) return null;
  return <RenderReady key={project.id} />;
}

function RenderReady() {
  const nav = useNavigate();
  const engine = useStore((s) => s.engine)!;
  const project = useStore((s) => s.project)!;
  const sceneId = useStore((s) => s.sceneId);
  const jobs = useStore((s) => s.jobs);
  const startRender = useStore((s) => s.startRender);
  const toast = useStore((s) => s.toast);

  const scenes = project.scenes;
  const [picked, setPicked] = useState<string[]>(() => (sceneId ? [sceneId] : scenes[0] ? [scenes[0].id] : []));
  const [aspect, setAspect] = useState<RenderAspect>("16:9");
  /** film — как в фильме (не громче оригинала), web — для YouTube и VK (громко, −14 LUFS) */
  const patchScene = useStore((s) => s.patchScene);
  const [loud, setLoudState] = useState<"film" | "web">(() => {
    const sc = scenes.find((s) => s.id === (sceneId ?? scenes[0]?.id));
    return sc?.mix.loudness_target === "film" ? "film" : "web";
  });
  const [subs, setSubs] = useState<SubtitleMode>("none");
  const [subText, setSubText] = useState<SubtitleText>("dub");
  const [height, setHeight] = useState<number | null>(1080);
  const [stems, setStems] = useState(false);
  const [out, setOut] = useState("");
  const [jobId, setJobId] = useState<string | null>(null);
  const [started, setStarted] = useState(false);
  const [fit, setFit] = useState<RenderFit>("blur");
  const startedHere = useRef(false);

  const chosen = scenes.filter((s) => picked.includes(s.id));
  const total = chosen.reduce((a, s) => a + (s.end - s.start), 0);
  // «Как есть» без перевода = текст оригинала: вшивать его поверх русской озвучки не нужно
  const untranslated = chosen.filter((s) => s.text_mode === "as_is" && s.lines.length > 0 && s.lines.every((l) => l.text === l.original));
  const title = chosen.length === 1 ? chosen[0].title : chosen.length > 1 ? `${project.title} — нарезка` : project.title;
  const ui = useStore((s) => s.ui);
  const setUi = useStore((s) => s.setUi);
  /** человек поправил путь сам (руками или в диалоге) — смена сцен его уже не затирает */
  const touched = useRef(false);

  // путь по умолчанию: папка, куда сохраняли в прошлый раз, + имя сцены (а не каждый раз «Видео/Дубль»)
  useEffect(() => {
    if (touched.current) return;
    let dead = false;
    void defaultOutputPath(title, ui.last_output_dir).then((p) => { if (!dead && !touched.current) setOut(p); });
    return () => { dead = true; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [title]);
  /** Запомнить папку пути на этом устройстве: следующий ролик предложим сохранить туда же. */
  const rememberDir = (path: string) => {
    const dir = dirOf(path);
    if (dir && dir !== ui.last_output_dir) setUi({ last_output_dir: dir });
  };

  const player = useMemo(() => new Player(Math.max(1, total), 0), [total]);
  useEffect(() => () => player.dispose(), [player]);
  usePlayerTick(player, 12);

  // после перезагрузки окна идущий рендер берём у движка (GET /jobs уже загружен в store)
  const live = Object.values(jobs).filter((j) => j.kind === "render_project" && j.project_id === project.id && (j.state === "running" || j.state === "queued")).at(-1) ?? null;
  useEffect(() => {
    if (!jobId && live) { setJobId(live.id); setStarted(true); }
  }, [jobId, live]);
  const rjob = jobId ? jobs[jobId] ?? null : live;
  const running = !!live || rjob?.state === "running" || rjob?.state === "queued";
  const failed = rjob?.state === "failed" ? rjob : null;
  const done = started && rjob?.state === "done";
  const restored = !!live && live.id === jobId && !startedHere.current;
  const savedPath = typeof rjob?.result?.path === "string" ? rjob.result.path : out;

  if (!scenes.length) {
    return (
      <section className="screen center">
        <EmptyState title="Нечего рендерить" text="Сначала нарежь хотя бы одну сцену и озвучь её.">
          <button className="btn primary lg" onClick={() => nav(`/p/${project.id}/analyze`)}>К разбору</button>
        </EmptyState>
      </section>
    );
  }

  /** Выбор громкости запоминается в самой сцене (MixSettings.loudness_target), чтобы превью и ролик звучали одинаково. */
  const setLoud = (v: "film" | "web") => {
    setLoudState(v);
    for (const s of chosen) void patchScene({ mix: { ...s.mix, loudness_target: v === "film" ? "film" : "online" } }, s.id);
  };
  const toggle = (id: string) => setPicked((p) => (p.includes(id) ? p.filter((x) => x !== id) : [...p, id]));
  // «Как в фильме» движок решает сам по MixSettings.loudness_target (громкость оригинала сцены); target_lufs — запасной уровень
  const lufs = -14;
  const voiced = chosen.filter((s) => s.takes.some((t) => t.selected && !(t as TakeX).empty));
  const noVoice = chosen.length > 0 && voiced.length === 0;
  const shownPath = savedPath || out;
  const wanted = out.split(/[\\/]/).pop() ?? "";
  const gotName = shownPath.split(/[\\/]/).pop() ?? "";
  const renamed = done && !!wanted && !!gotName && wanted !== gotName;

  const go = async () => {
    if (!chosen.length || running || noVoice) return;
    const ordered = scenes.filter((s) => picked.includes(s.id)).sort((a, b) => a.start - b.start).map((s) => s.id);
    // у каждой выбранной сцены громкость должна совпадать с переключателем (сцену могли отметить уже после выбора)
    const want = loud === "film" ? "film" : "online";
    for (const s of chosen) if (s.mix.loudness_target !== want) await patchScene({ mix: { ...s.mix, loudness_target: want } }, s.id);
    const target = out || "~/Movies/Дубль/ролик.mp4";
    rememberDir(target);
    const job = await startRender({
      scene_ids: ordered,
      settings: { aspect, height, target_lufs: lufs, fit, subtitles: subs, subtitle_text: subText, export_stems: stems, output_path: target },
    });
    if (job) { startedHere.current = true; setJobId(job.id); setStarted(true); }
  };

  const chooseFolder = async () => {
    const p = await pickSavePath(out);
    if (p) { touched.current = true; setOut(p); rememberDir(p); }
    else if (!inTauri()) toast("Путь можно вписать в поле — выбор папки работает в программе.");
  };

  const first = chosen[0];
  const ar = aspect === "9:16" ? "9 / 16" : aspect === "1:1" ? "1 / 1" : "16 / 9";

  return (
    <section className="screen render">
      <div className="vcol">
        <VideoPlayer player={player} src={first ? engine.clipUrl(project.id, first.id) : null} aspect={ar} timecode={fmtRel} className="rvvideo"
          overlay={<div className="fmtbadge mono">{aspect}</div>} />
        <Transport player={player} format={fmtRel}>
          <span className="mute" style={{ fontSize: 12 }}>{chosen.length ? `Предпросмотр: ${first.title}` : "Выбери сцены"}</span>
        </Transport>
      </div>

      <div className="panel">
        <div className="ph"><h2>Готовый ролик<Help label="Что такое рендер">Рендер — это сборка: программа склеивает видео с твоим голосом и фоном в один файл, который можно выложить или отправить.</Help></h2><span className="mono mute" style={{ fontSize: 11.5 }}>{chosen.length} {plural(chosen.length, "сцена", "сцены", "сцен")} · {fmtSec(total)}</span></div>
        <div className="pb" style={{ padding: "0 14px" }}>
          <div className="ropt">
            <span className="lb">Что собираем {scenes.length > 1 ? "— отметь одну или несколько сцен, склеим в один ролик" : ""}</span>
            <div className="scenepick" role="group" aria-label="Сцены для ролика">
              {scenes.map((s) => (
                <label key={s.id} className={"scpick" + (picked.includes(s.id) ? " on" : "")}>
                  <input type="checkbox" checked={picked.includes(s.id)} onChange={() => toggle(s.id)} aria-label={s.title} disabled={running} />
                  <span className="ell grow">{s.title}</span>
                  <span className="mono mute">{fmtSec(s.end - s.start)}</span>
                  <span className={"tag " + (s.takes.some((t) => t.selected) ? "ok" : "")}>{s.takes.some((t) => t.selected) ? "озвучена" : "без дублей"}</span>
                </label>
              ))}
            </div>
          </div>
          <div className="ropt">
            <div className="field"><span>Куда сохранить ролик</span>
              {/* путь длинный: показываем целиком, с переносом, и всегда с .mp4 на конце */}
              <textarea className="inp pathinp mono" rows={2} value={out} spellCheck={false} aria-label="Куда сохранить"
                onChange={(e) => { touched.current = true; setOut(e.target.value.replace(/[\r\n]+/g, "")); }}
                onBlur={() => {
                  const v = out.trim();
                  const full = v && !/\.mp4$/i.test(v) ? v + ".mp4" : out;
                  setOut(full);
                  if (touched.current && v) rememberDir(full);
                }} />
              <div className="row">
                <button className="btn" onClick={chooseFolder}><FolderOpen size={14} />Изменить…</button>
                <span className="note">{inTauri() ? "Откроется окно выбора места и имени файла." : "В браузере путь вписывается в поле."}</span>
              </div>
            </div>

          </div>
          <div className="ropt">
            <span className="lb">Формат</span>
            <div className="cards" role="radiogroup" aria-label="Формат">
              {FORMATS.map((f) => (
                <button key={f.id} role="radio" aria-checked={aspect === f.id} className={"cardopt" + (aspect === f.id ? " on" : "")} onClick={() => setAspect(f.id)}>
                  <div className="fr" style={{ width: f.w, height: f.h }} /><span className="t">{f.title}</span><span className="d">{f.desc}</span>
                </button>
              ))}
            </div>
          </div>
          {aspect !== "16:9" && (
            <div className="ropt" data-testid="fit-opt">
              <span className="lb">Как вписать кадр {aspect === "9:16" ? "в вертикальное видео" : "в квадрат"}<Help label="Как вписать кадр">Исходное видео широкое. «С размытыми полями» — весь кадр виден, пустое место заполняет размытая копия. «Обрезать по центру» — берётся средняя часть, края пропадают.</Help></span>
              <div className="fitcards" role="radiogroup" aria-label="Как вписать кадр">
                {([["blur", "С размытыми полями", "виден весь кадр"], ["crop", "Обрезать по центру", "края пропадут"]] as const).map(([id, t, d]) => (
                  <button key={id} role="radio" aria-checked={fit === id} className={"fitcard" + (fit === id ? " on" : "")} onClick={() => setFit(id)}>
                    <span className={"sch" + (aspect === "1:1" ? " sq" : "")} aria-hidden="true">
                      {id === "blur" && <i className="bg" />}
                      <i className={"fg" + (id === "crop" ? " crop" : "")} />
                    </span>
                    <span><b style={{ display: "block", fontSize: 12.5 }}>{t}</b><span className="mute">{d}</span></span>
                  </button>
                ))}
              </div>
            </div>
          )}
          <div className="ropt">
            <span className="lb">Громкость<Help label="Что такое громкость ролика">Площадки вроде YouTube и VK сами подтягивают громкость до одного уровня, поэтому для них звук делаем заранее погромче («средняя громкость» −14 LUFS). Для фильма оставляем кинематографичную громкость: тише, с запасом на громкие места.</Help></span>
            <Segmented label="Громкость" value={loud} onChange={setLoud} options={[
              { value: "film", label: "Как в фильме", title: "Не громче оригинала: динамика кино сохраняется" },
              { value: "web", label: "Для YouTube и VK", title: "Погромче, как у роликов в ленте" },
            ]} />
            <span className="note" data-testid="loud-note">{loud === "film" ? "Звук останется таким же по громкости, как в оригинале, — но не тише телевизора: совсем тихие исходники подтянем. Превью на «Сведении» звучит так же." : "Ролик будет громче оригинала — так он не потеряется в ленте."}</span>
          </div>
          <div className="ropt">
            <span className="lb">Субтитры</span>
            <Segmented label="Субтитры" value={subs} onChange={setSubs} options={[{ value: "none", label: "Нет" }, { value: "burn", label: "Вшить в видео" }, { value: "file", label: "Отдельным файлом" }]} />
            {subs !== "none" && (
              <>
                <div className="between" style={{ marginTop: 8 }}>
                  <span className="note">Текст субтитров</span>
                  <Segmented label="Текст субтитров" value={subText} onChange={setSubText} options={[{ value: "dub", label: "Перевод (текст дубля)" }, { value: "original", label: "Оригинал" }]} />
                </div>
                {subText === "dub" && untranslated.length > 0 && (
                  <span className="note warnnote" data-testid="sub-warn">
                    {untranslated.length === 1 ? `В сцене «${untranslated[0].title}»` : `В ${untranslated.length} сценах`} перевод не написан. Для таких реплик возьмём русские субтитры из твоего файла, а если их нет — оригинальный текст. Или переведи на экране «Запись».
                  </span>
                )}
              </>
            )}
          </div>
          <div className="ropt">
            <span className="lb">Качество</span>
            <Segmented label="Качество" value={height === null ? "orig" : String(height)} onChange={(v) => setHeight(v === "orig" ? null : Number(v))}
              options={[{ value: "1080", label: "1080p" }, { value: "720", label: "720p" }, { value: "orig", label: "Как в оригинале" }]} />
          </div>
          <div className="ropt">
            <label className="between" style={{ fontSize: 12.5 }}>Дорожки отдельными файлами<Switch label="Дорожки отдельными файлами" checked={stems} onChange={setStems} /></label>
            <span className="note">Голоса, фон и каждый персонаж — отдельными WAV, если захочешь доделать в другой программе.</span>
          </div>
        </div>
        <div className="rfoot">
          {restored && (
            <div className="banner info" data-testid="render-restored">Ролик уже собирается — показываю прогресс. Новый запускать не нужно.</div>
          )}
          {started && rjob && !done && (
            <div className="col" style={{ gap: 10 }}>
              <JobSteps job={rjob} />
              {failed && <div className="tag bad" style={{ whiteSpace: "normal", height: "auto", padding: "5px 8px" }}>{failed.error || "Ролик собрать не удалось"}</div>}
            </div>
          )}
          {done && (
              <div className="donebox" data-testid="done">
                <div className="row" style={{ color: "var(--ok)", fontWeight: 600 }}><CheckCircle2 size={16} />Готово</div>
                <div className="mono pathshow" title={savedPath} data-testid="saved-path">{savedPath}</div>
                {renamed && <div className="renamed" role="status" data-testid="renamed">Файл «{wanted}» уже был, поэтому ролик сохранён как «{gotName}». Прежний файл не тронут.</div>}
                <div className="row wrap">
                  {inTauri() ? (
                    <>
                      <button className="btn primary" onClick={async () => { if (!(await openFile(savedPath))) toast("Не получилось открыть ролик. Найди его по пути выше.", "error"); }}><ExternalLink size={14} />Открыть ролик</button>
                      <button className="btn" onClick={async () => { if (!(await revealFile(savedPath))) toast("Не получилось открыть папку. Путь выше можно скопировать.", "error"); }}><FolderOpen size={14} />Показать в папке</button>
                    </>
                  ) : (
                    <button className="btn primary" onClick={async () => toast((await copyText(savedPath)) ? "Путь скопирован." : "Не получилось скопировать — выдели путь выше.", "ok")}><Copy size={14} />Скопировать путь</button>
                  )}
                </div>
                <div className="row wrap">
                  <button className="btn" onClick={() => { setStarted(false); setJobId(null); startedHere.current = false; }}>Ещё раз</button>
                  <button className="btn" onClick={() => nav(`/p/${project.id}/analyze`)}>Нарезать ещё сцену</button>
                  <button className="btn ghost" onClick={() => nav("/")}>К проектам</button>
                </div>
              </div>
          )}
          {!done && noVoice && (
            <div className="banner" role="status" data-testid="no-voice">
              <span>В {chosen.length === 1 ? "этой сцене" : "выбранных сценах"} нет дублей, поэтому собирать пока нечего. Запиши голос или отметь ★ нужный дубль.</span>
              <button className="btn sm amber" onClick={() => nav(`/p/${project.id}/record`)}>К записи</button>
            </div>
          )}
          {!done && (
              <button className="btn primary lg" style={{ width: "100%" }} onClick={go} disabled={!chosen.length || running || !out.trim() || noVoice} data-primary="render" title={running ? "Сборка уже идёт — дождись конца" : noVoice ? "В выбранных сценах нет дублей" : undefined}>
                <Film size={16} />{running ? "Собираю…" : chosen.length > 1 ? `Склеить ${chosen.length} ${plural(chosen.length, "сцену", "сцены", "сцен")} в ролик` : "Собрать ролик"}
              </button>
          )}
        </div>
      </div>
    </section>
  );
}
