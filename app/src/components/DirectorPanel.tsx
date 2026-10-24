import { useEffect, useMemo, useState } from "react";
import { Clapperboard, Send } from "lucide-react";
import type { Scene, Speaker, Take } from "../api/types";
import { describePatch } from "../lib/director";
import { isFileTake, visibleRemark, type RemarkLike, type TakeX } from "../lib/takeReview";
import { useStore } from "../state/store";
import { useAiReady } from "../lib/ai";
import { AiHint, AI_OFF_TITLE } from "./AiHint";

interface Props {
  scene: Scene;
  speakers: Speaker[];
}

interface Msg { who: "me" | "ai"; text: string }

type RemarkView = RemarkLike & { more?: string };
export interface NoteGroup { speaker: string | null; remark: RemarkView | undefined; takes: Take[]; where: string }

/** Сгруппировать замечания по дублям: персонаж + одинаковый текст → одна строка с номерами реплик. */
export function noteGroups(takes: Take[], scene: Pick<Scene, "lines" | "mix">, remarkOf: (t: Take) => RemarkView | undefined): NoteGroup[] {
  const lips = scene.mix.mode === "dub";
  const out: NoteGroup[] = [];
  for (const t of takes) {
    const raw = remarkOf(t);
    const file = isFileTake(t as TakeX);
    let r = visibleRemark(raw, { lips, file });
    if (raw && !r) r = { ...raw, text: file ? "Дубль из файла: попадание в губы не проверяем." : "Закадровый режим: попадать в губы не нужно.", more: undefined };
    const g = out.find((x) => x.speaker === t.speaker_id && !!x.remark && !!r && x.remark.text === r.text && x.remark.severity === r.severity);
    if (g) g.takes.push(t);
    else out.push({ speaker: t.speaker_id, remark: r, takes: [t], where: "" });
  }
  const num = (t: Take) => scene.lines.findIndex((l) => l.line_id === t.line_id) + 1;
  for (const g of out) {
    const ns = g.takes.map(num).filter((n) => n > 0).sort((a, b) => a - b);
    g.where = !ns.length ? "" : ns.length === 1 ? `реплика ${ns[0]}` : `реплики ${ns.slice(0, -1).join(", ")} и ${ns[ns.length - 1]}`;
  }
  return out;
}

/** Примеры просьб из персонажей самой сцены: «Шкипер громче», а не абстрактный «кот». */
export function directorExamples(names: string[]): string[] {
  const [a, b] = names.filter(Boolean);
  const out: string[] = [];
  if (a) out.push(`${a} громче`);
  if (b) out.push(`${b} как по рации`);
  else if (a) out.push(`${a} как в пещере`);
  out.push("фон потише");
  return out;
}

/**
 * Помощник-режиссёр: замечания по дублям + поле «Скажи, что поменять».
 * Он не слышит звук — видит только замеры (сдвиг, длина, громкость) и пересказывает их простыми словами.
 */
export function DirectorPanel({ scene, speakers }: Props) {
  const engine = useStore((s) => s.engine);
  const project = useStore((s) => s.project);
  const remarks = useStore((s) => s.remarks);
  const loadRemark = useStore((s) => s.loadRemark);
  const setMix = useStore((s) => s.setMix);
  const toast = useStore((s) => s.toast);
  const [text, setText] = useState("");
  const [busy, setBusy] = useState(false);
  const [log, setLog] = useState<Msg[]>([]);
  const [undo, setUndo] = useState<Scene["mix"] | null>(null);
  const [openMore, setOpenMore] = useState<Record<string, boolean>>({});
  const aiOn = useAiReady("director");

  const selected = useMemo(() => scene.takes.filter((t) => t.selected && t.line_id), [scene.takes]);
  const nameOf = (id: string | null) => {
    const s = speakers.find((x) => x.id === id);
    return s ? (s.name ?? s.label) : "—";
  };
  const colorOf = (id: string | null) => speakers.find((x) => x.id === id)?.color ?? "#85827d";
  const examples = useMemo(() => {
    const ids = [...new Set(scene.lines.map((l) => l.speaker_id).filter((x): x is string => !!x))];
    return directorExamples(ids.map((id) => nameOf(id)).filter((n) => n !== "—"));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [scene.lines, speakers]);
  const hint = examples.slice(0, 2).map((e) => `«${e}»`).join(", ");

  /**
   * Одна строка на персонажа и одинаковое замечание: два «отличных дубля» Голоса 2 — это одна строка «реплики 2 и 4»,
   * а не две одинаковые подряд. Советы про губы для дубля из файла не показываем (он к губам не привязан).
   */
  const groups = useMemo(() => noteGroups(selected, scene, (t) => remarks[`${project?.id}/${scene.id}/${t.id}`]), [selected, scene, remarks, project?.id]);

  // подгружаем замечания для выбранных дублей
  useEffect(() => {
    selected.forEach((t) => { if (!remarks[`${project?.id}/${scene.id}/${t.id}`]) void loadRemark(t.id); });
  }, [selected, remarks, project?.id, scene.id, loadRemark]);

  const send = async (raw: string) => {
    const q = raw.trim();
    if (!q || !engine || !project || busy) return;
    setBusy(true);
    setLog((l) => [...l, { who: "me", text: q }]);
    setText("");
    try {
      const r = await engine.directorCommand({ project_id: project.id, scene_id: scene.id, text: q });
      if (r.patches.length) {
        // новые настройки приходят от движка готовыми — отправляем их в сцену как есть
        setUndo(scene.mix);
        await setMix(r.mix);
      }
      setLog((l) => [...l, { who: "ai", text: r.patches.length
        ? "Сделал: " + r.patches.map((p) => (p.note || describePatch(p, speakers)).toLowerCase()).join("; ") + "."
        : `Не понял, что поменять. Попробуй так: ${hint}.` }]);
    } catch (e) {
      toast((e as Error).message, "error");
    } finally {
      setBusy(false);
    }
  };

  const revert = async () => {
    if (!undo) return;
    await setMix(undo);
    setUndo(null);
    setLog((l) => [...l, { who: "ai", text: "Вернул как было." }]);
  };

  return (
    <section className="director" aria-label="Помощник-режиссёр">
      <div className="between">
        <h2 className="row"><Clapperboard size={13} /> Помощник-режиссёр</h2>
        {undo && <button className="btn ghost sm" onClick={revert}>Отменить</button>}
      </div>

      <div className="notes">
        {selected.length === 0 && <div className="note">Запиши пару реплик — я скажу, как получилось.</div>}
        {groups.map((g) => {
          const r = g.remark;
          const key = g.takes[0].id;
          return (
            <div key={key} className={"dnote " + (r?.severity ?? "")} data-testid="director-note">
              <i className="dot" style={{ background: r?.severity === "warn" ? "var(--warn)" : colorOf(g.speaker) }} aria-hidden="true" />
              <div>
                <b>{nameOf(g.speaker)}</b>{g.where && <span className="mute"> · {g.where}</span>}{" "}
                <span>{r ? r.text : "Слушаю дубль…"}{r?.more && (
                  <> <button className="link" onClick={() => setOpenMore((m) => ({ ...m, [key]: !m[key] }))} aria-expanded={!!openMore[key]}>{openMore[key] ? "Скрыть" : "Ещё"}</button>{openMore[key] && <> {r.more}</>}</>
                )}</span>
              </div>
            </div>
          );
        })}
      </div>

      {log.length > 0 && (
        <div className="dlog" aria-live="polite">
          {log.slice(-4).map((m, i) => <div key={i} className={"msg " + m.who}>{m.text}</div>)}
        </div>
      )}

      <form className="row" onSubmit={(e) => { e.preventDefault(); void send(text); }}>
        <input className="inp" value={text} onChange={(e) => setText(e.target.value)} placeholder={`Скажи, что поменять: ${hint}`}
          aria-label="Скажи, что поменять" disabled={busy || !aiOn} title={aiOn ? undefined : AI_OFF_TITLE} />
        <button className="btn amber iconbtn" type="submit" aria-label="Отправить" disabled={busy || !aiOn || !text.trim()} title={aiOn ? undefined : AI_OFF_TITLE}><Send size={14} /></button>
      </form>
      <div className="row wrap">
        {examples.map((e) => <button key={e} type="button" className="chipbtn" onClick={() => void send(e)} disabled={busy || !aiOn}>{e}</button>)}
        {!aiOn && <AiHint />}
      </div>
    </section>
  );
}
