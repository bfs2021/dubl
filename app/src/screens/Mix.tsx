import { useEffect, useMemo, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { ArrowRight, ChevronDown, ChevronRight, Sparkles, Volume2 } from "lucide-react";
import { activeJob, useScene, useStore } from "../state/store";
import { Player, usePlayerTick } from "../lib/player";
import { VideoPlayer, Transport } from "../components/VideoPlayer";
import { Switch } from "../components/Switch";
import { Fader } from "../components/Fader";
import { Meter } from "../components/Meter";
import { DirectorPanel } from "../components/DirectorPanel";
import { JobSteps } from "../components/JobSteps";
import { EmptyState } from "../components/EmptyState";
import { Help } from "../components/Help";
import { ROOMS, VOICE_FX, defaultTrack } from "../api/presets";
import { fmtDb, fmtMs, fmtRel } from "../lib/format";
import { takeIsUsable, type TakeX } from "../lib/takeReview";
import { mixWarning } from "../lib/mixCheck";
import type { MixSettings, Scene, Speaker, Take, TrackSettings } from "../api/types";

interface TrackRow { id: string; title: string; color: string; sub: string; settings: TrackSettings; kind: "background" | "speaker" | "original"; voiced?: boolean; /** можно двигать громкость */ adjustable: boolean }

export function MixScreen() {
  const nav = useNavigate();
  const project = useStore((s) => s.project);
  const scene = useScene();
  if (!project) return null;
  if (!scene) {
    return (
      <section className="screen center">
        <EmptyState title="Нет сцены для сведения" text="Сначала нарежь сцену и запиши хотя бы одну реплику.">
          <button className="btn primary lg" onClick={() => nav(`/p/${project.id}/analyze`)}>К разбору <ArrowRight size={16} /></button>
        </EmptyState>
      </section>
    );
  }
  return <MixScene key={project.id + "/" + scene.id} scene={scene} />;
}

/** Уровень в индикаторе — условный: зависит от громкости дорожки, пока играет сцена. */
function fakeLevel(gain: number, playing: boolean, t: number, seed: number): number {
  if (!playing) return -60;
  const wob = (Math.sin(t * 7 + seed) + Math.sin(t * 3.1 + seed * 2)) * 4;
  return Math.min(-3, -22 + gain + wob);
}

/** Диапазон ползунка сдвига: ±3 с, а если значение уже за этим — шире, чтобы подпись и ползунок совпадали. */
export const offRange = (v: number) => Math.max(3000, Math.ceil(Math.abs(v) / 500) * 500);
/** Диапазон ползунка длины: 70–130%, шире — если значение за пределами. */
export const stretchRange = (v: number): [number, number] => [Math.min(0.7, Math.floor(v * 20) / 20), Math.max(1.3, Math.ceil(v * 20) / 20)];

function MixScene({ scene }: { scene: Scene }) {
  const nav = useNavigate();
  const engine = useStore((s) => s.engine)!;
  const project = useStore((s) => s.project)!;
  const jobs = useStore((s) => s.jobs);
  const setMix = useStore((s) => s.setMix);
  const patchTake = useStore((s) => s.patchTake);
  const startPreview = useStore((s) => s.startPreview);
  const [fine, setFine] = useState(false);

  const dur = scene.end - scene.start;
  const player = useMemo(() => new Player(dur, 0), [dur, scene.id]);
  usePlayerTick(player, 15);
  const speakers: Speaker[] = project.analysis?.speakers ?? [];
  const mix = scene.mix;
  const prev = activeJob(jobs, "preview", project.id, scene.id);

  const rows: TrackRow[] = useMemo(() => {
    const inScene = [...new Set(scene.lines.map((l) => l.speaker_id).filter(Boolean) as string[])];
    const dubbedIds = scene.dubbed_speakers.length ? scene.dubbed_speakers : inScene;
    const dub = mix.mode === "dub";
    const nameOf = (id: string) => { const sp = speakers.find((s) => s.id === id); return { title: sp?.name ?? sp?.label ?? id, color: sp?.color ?? "#85827d" }; };
    const out: TrackRow[] = [];
    for (const id of dubbedIds) {
      const { title, color } = nameOf(id);
      const n = scene.takes.filter((t) => t.speaker_id === id && t.selected && takeIsUsable(t as TakeX)).length;
      // озвучиваем, но дублей нет — в ролике звучит оригинальный голос: так и пишем, чтобы персонаж не «пропал»
      if (n) out.push({ id, title, color, sub: `озвучено реплик: ${n}`, voiced: true, settings: mix.tracks[id] ?? defaultTrack(), kind: "speaker", adjustable: true });
      else out.push({ id, title: `${title} — оригинал`, color, sub: "дублей пока нет — звучит оригинальный голос", voiced: false, settings: mix.tracks[id] ?? defaultTrack(), kind: "original", adjustable: false });
    }
    // кого не озвучиваем — остаётся оригиналом; громкость его голоса движок берёт из mix.tracks (в дубляже)
    for (const id of inScene.filter((x) => !dubbedIds.includes(x))) {
      const { title, color } = nameOf(id);
      out.push({ id, title: `${title} — оригинал`, color, sub: dub ? "не озвучиваем — звучит оригинальный голос" : "не озвучиваем — громкость общая для оригинала, в «Тонкой настройке»",
        voiced: true, settings: mix.tracks[id] ?? defaultTrack(), kind: "original", adjustable: dub });
    }
    out.push({ id: "background", title: "Музыка и шумы", color: "#85827d", sub: "фон из оригинала", settings: mix.background, kind: "background", voiced: true, adjustable: true });
    return out;
  }, [scene, speakers, mix]);

  // «руками выключил» отличаем от «выключил переключателем», чтобы плашка не врала
  const [byHand, setByHand] = useState(false);
  const update = (next: MixSettings, hand = true) => {
    if (mix.autopilot && !next.autopilot) setByHand(hand);
    if (next.autopilot) setByHand(false);
    void setMix(next);
  };
  // автопилот выключает только громкость: комнату, эффект, «Выкл.» и «Только она» он сам не трогает (audio.mix.automix)
  const setTrack = (id: string, patch: Partial<TrackSettings>) => {
    const autopilot = mix.autopilot && patch.gain_db === undefined;
    if (id === "background") update({ ...mix, background: { ...mix.background, ...patch }, autopilot });
    else update({ ...mix, tracks: { ...mix.tracks, [id]: { ...(mix.tracks[id] ?? defaultTrack()), ...patch } }, autopilot });
  };
  const setAuto = (v: boolean) => update({ ...mix, autopilot: v }, false);
  const anyTakes = scene.takes.some((t) => t.selected && takeIsUsable(t as TakeX));
  const solo = rows.some((r) => r.settings.solo);
  const warn = anyTakes ? mixWarning(scene) : null;

  const overlay = null;

  // ───── превью пересобирается само (через ~1 с после правок); если не успело — видна метка «Превью устарело · Обновить» ─────
  const sig = useMemo(() => JSON.stringify([mix, scene.takes.filter((t) => t.selected).map((t) => [t.id, t.offset_ms, t.stretch])]), [mix, scene.takes]);
  const sigKey = `dubl.preview.${project.id}.${scene.id}`;
  const built = useRef<string | null>(null);
  if (built.current === null) { try { built.current = sessionStorage.getItem(sigKey) ?? ""; } catch { built.current = ""; } }
  const running = prev?.state === "running" || prev?.state === "queued";
  // пока собирается новое превью, продолжаем показывать прошлое (последнее готовое)
  const lastDone = Object.values(jobs).filter((j) => j.kind === "preview" && j.project_id === project.id && j.scene_id === scene.id && j.state === "done").at(-1) ?? null;
  const hasPreview = !!lastDone;
  const rebuild = () => {
    built.current = sig;
    try { sessionStorage.setItem(sigKey, sig); } catch { /* приватный режим */ }
    void startPreview();
  };
  const stale = anyTakes && !running && (!hasPreview || built.current !== sig);
  // зашли на сведение, голос записан, а превью ещё нет (например, программу перезапускали) — соберём само, один раз за вход
  const autoTried = useRef(false);
  useEffect(() => {
    if (autoTried.current || !anyTakes || running || hasPreview) return;
    // флаг ставим, когда сборка реально запущена (а не при планировании): при сбое повторять каждые 0,7 с не нужно
    const t = setTimeout(() => { autoTried.current = true; rebuild(); }, 700);
    return () => clearTimeout(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [anyTakes, running, hasPreview]);
  useEffect(() => {
    // авто-пересборка — только если превью уже собирали и настройки поменялись
    if (!anyTakes || running || !built.current || built.current === sig) return;
    const t = setTimeout(rebuild, 1000);
    return () => clearTimeout(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sig, running, anyTakes]);
  const previewSrc = lastDone ? engine.previewUrl(project.id, scene.id) : null;
  const vsrc = previewSrc ? previewSrc + (previewSrc.includes("?") ? "&" : "?") + "v=" + lastDone!.id : engine.clipUrl(project.id, scene.id);

  return (
    <section className="screen mix">
      <div className="vcol">
        <VideoPlayer player={player} src={vsrc}
          timecode={fmtRel} overlay={overlay} cuts={(project.analysis?.shot_cuts ?? []).map((c) => c - scene.start)} />
        <Transport player={player} format={fmtRel}>
          <button className={"btn sm" + (stale ? " amber" : "")} onClick={rebuild} disabled={running || !anyTakes} title="Собрать звук ролика с твоими записями, чтобы послушать результат">
            <Volume2 size={13} />{running ? "Собираю…" : !anyTakes ? "Превью" : stale ? (hasPreview ? "Обновить превью" : "Собрать превью") : "Превью готово ✓"}
          </button>
          {stale && hasPreview && <span className="stale" data-testid="stale">Превью устарело · обновится само через секунду</span>}
          {!hasPreview && anyTakes && !running && <span className="note" data-testid="preview-auto">Превью соберётся само. Можно и сразу нажать — послушаешь, как звучит ролик с твоим голосом.</span>}
          {!hasPreview && !anyTakes && <span className="note">Пока в кадре — исходник сцены. Когда запишешь голос, здесь появится превью со звуком.</span>}
        </Transport>
        {prev && prev.state === "running" && <div style={{ marginTop: 8 }}><JobSteps job={prev} /></div>}
        {!anyTakes && <div className="note" style={{ marginTop: 8 }}>Дублей пока нет — <button className="link" onClick={() => nav(`/p/${project.id}/record`)}>запиши голос</button> или <button className="link" onClick={() => nav(`/p/${project.id}/record`)}>загрузи файл дубля</button> — и здесь появится что сводить.</div>}
      </div>

      <div className="panel mixmain">
        <div className="ph">
          <div className="row" style={{ gap: 10 }}>
            <Switch label="Автопилот" checked={mix.autopilot} onChange={setAuto} />
            <div>
              <div style={{ fontWeight: 600 }}>Автопилот<Help label="Что такое автопилот">Программа сама выравнивает громкость голосов и фона, подбирает звучание комнаты и делает так, чтобы ролик не оглушал. Пока автопилот включён, ничего делать не нужно.</Help></div>
              <div className="note" style={{ lineHeight: 1.2 }}>{mix.autopilot ? "Сам выравнивает громкость голосов и подбирает звучание" : "Выключен — всё, что ты настроил, остаётся как есть"}</div>
            </div>
          </div>
          <div className="col" style={{ gap: 3, alignItems: "flex-end" }}>
            <button className={"btn " + (anyTakes ? "amber" : "")} onClick={() => nav(`/p/${project.id}/render`)} data-primary="next" disabled={!anyTakes}
              title={anyTakes ? "Следующий шаг: собрать готовый ролик" : "Пока нечего собирать: запиши голос или отметь ★ нужный дубль"}>Дальше: готовый ролик <ArrowRight size={14} /></button>
            {!anyTakes && <span className="note" data-testid="mix-next-why">Нет дублей — сначала запиши голос</span>}
          </div>
        </div>
        <div className="pb col" style={{ gap: 14 }}>
          {!mix.autopilot && (
            <div className="banner" role="status" data-testid="autopilot-off" data-warn={warn?.kind === "autopilot" ? "1" : undefined}>
              <span>{warn?.kind === "autopilot" ? warn.text : byHand ? "Автопилот выключен, потому что ты поменял громкость вручную. Всё остаётся как ты поставил." : "Автопилот выключен: громкость остаётся такой, как стоит сейчас."}</span>
              <button className="btn sm amber" onClick={() => setAuto(true)}>Вернуть автопилот</button>
            </div>
          )}
          {mix.autopilot && warn && (
            <div className="banner" role="status" data-testid="mix-warn" data-kind={warn.kind}>
              <span>{warn.text}</span>
              {warn.kind === "record_louder" && <button className="btn sm amber" onClick={() => nav(`/p/${project.id}/record`)}>К записи</button>}
            </div>
          )}
          <div className="modeinfo" data-testid="mix-help">
            <span><b>Автопилот</b> — программа сама выравнивает громкость и звучание. Ничего делать не нужно.</span>
            <span><b>Помощник-режиссёр</b> — скажи словами, что поменять («Шкипер громче», «фон потише»), и он подвинет ползунки за тебя.</span>
            <span><b>Тонкая настройка</b> — сдвиг, эхо и эффекты вручную. Если двинешь громкость, автопилот выключится.</span>
          </div>
          <DirectorPanel scene={scene} speakers={speakers} />

          <div>
            <h2 style={{ marginBottom: 8 }}>Дорожки</h2>
            <div className="trk-list">
              {rows.map((r, i) => (
                <div key={r.id} className={"trk" + (r.settings.mute || (solo && !r.settings.solo) ? " off" : "") + (r.kind === "original" ? " orig" : "")} data-testid={`track-${r.id}`}>
                  <i className="dot" style={{ background: r.color }} />
                  <div className="trk-name"><b className="ell">{r.title}</b><span className="mute ell">{r.sub}</span></div>
                  {r.adjustable ? (
                    <>
                      <Fader label={`Громкость: ${r.title}`} value={r.settings.gain_db} onChange={(v) => setTrack(r.id, { gain_db: v })} />
                      <span className="mono db">{fmtDb(r.settings.gain_db)}</span>
                    </>
                  ) : <><span /><span className="mono db">—</span></>}
                  <div style={{ width: 70 }}><Meter db={r.settings.mute ? -60 : fakeLevel(r.settings.gain_db, player.playing, player.t, i)} /></div>
                </div>
              ))}
            </div>
          </div>

          <div className="fine">
            <button className="fine-toggle" aria-expanded={fine} onClick={() => setFine(!fine)} data-testid="fine-toggle">
              {fine ? <ChevronDown size={15} /> : <ChevronRight size={15} />}
              <Sparkles size={13} />Тонкая настройка
              <span className="mute">сдвиг, эхо, эффекты — для тех, кому мало автопилота</span>
            </button>
            {fine && <FineTune scene={scene} rows={rows} mix={mix} setTrack={setTrack} update={update} onTake={(tid, p) => void patchTake(tid, p)} />}
          </div>
        </div>
      </div>
    </section>
  );
}

function FineTune({ scene, rows, mix, setTrack, update, onTake }: {
  scene: Scene; rows: TrackRow[]; mix: MixSettings;
  setTrack: (id: string, p: Partial<TrackSettings>) => void; update: (m: MixSettings, hand?: boolean) => void; onTake: (tid: string, p: { offset_ms?: number; stretch?: number }) => void;
}) {
  const [takeId, setTakeId] = useState<string | null>(null);
  const selTakes = scene.takes.filter((t) => t.selected);
  const take: Take | undefined = selTakes.find((t) => t.id === takeId) ?? selTakes[0];
  const line = take ? scene.lines.find((l) => l.line_id === take.line_id) : undefined;

  return (
    <div className="finebody" data-testid="fine-body">
      <div className="finecols">
        <div className="col" style={{ gap: 10 }}>
          <h2>Дорожки</h2>
          {rows.map((r) => (
            <div key={r.id} className="fine-trk">
              <div className="between"><span className="row"><i className="dot" style={{ background: r.color }} /><b>{r.title}</b></span>
                <span className="ms">
                  <button className={"m" + (r.settings.mute ? " on" : "")} aria-pressed={r.settings.mute} title="Выключить эту дорожку" onClick={() => setTrack(r.id, { mute: !r.settings.mute })}>Выкл.</button>
                  <button className={"s" + (r.settings.solo ? " on" : "")} aria-pressed={r.settings.solo} title="Слушать только эту дорожку, остальные приглушить" onClick={() => setTrack(r.id, { solo: !r.settings.solo })}>Только она</button>
                </span></div>
              {r.kind === "speaker" && (
                <div className="grid2">
                  <label className="field"><span>Где звучит<Help label="Что такое «Где звучит»">Эхо места: «Комната», «Пещера», «Из-за двери» и т. д. «Подобрать сам» берёт звучание из оригинала.</Help></span>
                    <select className="inp sm" value={r.settings.room} onChange={(e) => setTrack(r.id, { room: e.target.value })}>
                      {ROOMS.map((x) => <option key={x.id} value={x.id}>{x.title}</option>)}
                    </select></label>
                  <label className="field"><span>Эффект голоса</span>
                    <select className="inp sm" value={r.settings.voice_fx} onChange={(e) => setTrack(r.id, { voice_fx: e.target.value })}>
                      {VOICE_FX.map((x) => <option key={x.id} value={x.id}>{x.title}</option>)}
                    </select></label>
                </div>
              )}
            </div>
          ))}
        </div>

        <div className="col" style={{ gap: 10 }}>
          <h2>Запись голоса<Help label="Что такое запись голоса">Это твой дубль — одна записанная попытка реплики. Здесь можно точнее подвинуть её во времени и чуть растянуть.</Help></h2>
          {!take && <div className="note">Нет записей в ролике. Запиши голос или отметь ★ нужный дубль.</div>}
          {take && (
            <>
              <label className="field"><span>Реплика</span>
                <select className="inp sm" value={take.id} onChange={(e) => setTakeId(e.target.value)}>
                  {selTakes.map((t) => <option key={t.id} value={t.id}>{scene.lines.find((l) => l.line_id === t.line_id)?.text.slice(0, 40) ?? t.id}</option>)}
                </select></label>
              {line && <div className="note">{line.text}</div>}
              <div className="field"><div className="between"><span>Когда начинается<Help label="Что такое сдвиг">Сдвиг записи во времени. Плюс — позже, минус — раньше. Если голос опаздывает за губами — подвинь раньше.</Help></span><span className="mono" data-testid="offset-label">{fmtMs(take.offset_ms)}</span></div>
                <Fader label="Сдвиг записи" min={-offRange(take.offset_ms)} max={offRange(take.offset_ms)} step={10} value={take.offset_ms} onChange={(v) => onTake(take.id, { offset_ms: v })} /></div>
              <div className="field"><div className="between"><span>Скорость речи<Help label="Что такое длина">100% — как записал. Меньше 100% — запись сжата (быстрее), больше — растянута (медленнее).</Help></span><span className="mono" data-testid="stretch-label">{Math.round(take.stretch * 100)}% {Math.abs(take.stretch - 1) < 0.03 ? "· как записано" : ""}</span></div>
                <Fader label="Длина записи" min={stretchRange(take.stretch)[0]} max={stretchRange(take.stretch)[1]} step={0.01} neutral={1} value={take.stretch} onChange={(v) => onTake(take.id, { stretch: v })} /></div>
            </>
          )}
        </div>

        <div className="col" style={{ gap: 10 }}>
          <h2>Общее</h2>
          <div className="opt"><span className="l">Фон приглушается под голосом</span><Switch label="Фон приглушается под голосом" checked={mix.duck_background} onChange={(v) => update({ ...mix, duck_background: v, autopilot: false })} /></div>
          <div className="opt"><span className="l">Дыхание и шорохи из оригинала</span><Switch label="Дыхание и шорохи из оригинала" checked={mix.keep_breaths} onChange={(v) => update({ ...mix, keep_breaths: v })} /></div>
          <div className="field"><div className="between"><span>Громкость оригинала (у тех, кого не озвучиваем)</span><span className="mono">{fmtDb(mix.original_voices_db, 0)}</span></div>
            <Fader label="Оригинальные голоса" min={-40} max={0} step={1} value={mix.original_voices_db} neutral={-14} onChange={(v) => update({ ...mix, original_voices_db: v, autopilot: false })} /></div>
          <details className="adv">
            <summary>Для опытных: громкость всего ролика</summary>
            <div className="col" style={{ gap: 10 }}>
              <div className="field"><div className="between"><span>Общая громкость ролика<Help label="Что такое LUFS">Насколько громким получится весь ролик. Измеряется в LUFS: −14 — как на YouTube, −16 — Telegram, −23 — ТВ. Обычно это выбирается на экране «Рендер».</Help></span><span className="mono lufs">{mix.target_lufs} LUFS</span></div>
                <Fader label="Общая громкость ролика, LUFS" min={-30} max={-6} step={1} neutral={-14} value={mix.target_lufs} onChange={(v) => update({ ...mix, target_lufs: v, autopilot: false })} /></div>
              <div className="field"><div className="between"><span>Предел громкости<Help label="Что такое dBTP">Самый громкий момент, выше которого звук не поднимется (true peak, dBTP). −1 безопасно: не будет хрипа на пиках.</Help></span><span className="mono">{mix.true_peak_db} dBTP</span></div>
                <Fader label="Предел громкости, dBTP" min={-6} max={0} step={0.5} neutral={-1} value={mix.true_peak_db} onChange={(v) => update({ ...mix, true_peak_db: v, autopilot: false })} /></div>
            </div>
          </details>
          <div className="note">Ползунки громкости и «Фон приглушается» выключают автопилот, а комната, эффект и «Выкл.» — нет. Вернуть автопилот можно кнопкой сверху.</div>
        </div>
      </div>
    </div>
  );
}
