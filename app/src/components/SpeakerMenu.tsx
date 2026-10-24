import { useEffect, useLayoutEffect, useRef, useState, type RefObject } from "react";
import { createPortal } from "react-dom";
import { Check } from "lucide-react";
import type { Speaker } from "../api/types";

/** Куда поставить меню у кнопки: под ней, а если внизу не помещается — над ней. Не вылезает за края окна. */
export function menuPlace(anchor: { top: number; bottom: number; left: number }, size: { w: number; h: number }, view: { w: number; h: number }, gap = 4, margin = 8) {
  const below = anchor.bottom + gap;
  const fitsBelow = below + size.h <= view.h - margin;
  const fitsAbove = anchor.top - gap - size.h >= margin;
  const up = !fitsBelow && fitsAbove;
  const top = up ? anchor.top - gap - size.h : Math.max(margin, Math.min(below, view.h - margin - size.h));
  const left = Math.max(margin, Math.min(anchor.left, view.w - margin - size.w));
  return { top, left, up };
}

interface Props {
  anchor: RefObject<HTMLElement>;
  /** подпись меню для экранного диктора */
  label: string;
  speakers: Speaker[];
  current: string | null | undefined;
  onPick: (speakerId: string) => void;
  onNew: (name: string) => void;
  onClose: () => void;
}

/**
 * Меню «Кто это говорит?» поверх всего окна: рисуется в document.body (position: fixed по координатам кнопки),
 * поэтому соседние карточки и прокрутка списка его не перекрывают. Закрывается по Esc, клику мимо, прокрутке.
 * Клавиатура: ↑↓ — по пунктам, Enter — выбрать, Home/End — первый/последний.
 */
export function SpeakerMenu({ anchor, label, speakers, current, onPick, onNew, onClose }: Props) {
  const ref = useRef<HTMLDivElement>(null);
  const [pos, setPos] = useState<{ top: number; left: number; up: boolean } | null>(null);
  const [newName, setNewName] = useState<string | null>(null);

  const openAt = useRef<{ top: number; left: number } | null>(null);
  const place = () => {
    const a = anchor.current?.getBoundingClientRect();
    const m = ref.current;
    if (!a || !m) return;
    openAt.current = { top: a.top, left: a.left };
    setPos(menuPlace(a, { w: m.offsetWidth, h: m.offsetHeight }, { w: window.innerWidth, h: window.innerHeight }));
  };
  useLayoutEffect(place, [newName]); // eslint-disable-line react-hooks/exhaustive-deps

  const items = () => [...(ref.current?.querySelectorAll<HTMLElement>("[data-mi]") ?? [])];
  // фокус — на текущего персонажа (или на первый пункт), когда меню уже встало на место
  const focused = useRef(false);
  useEffect(() => {
    if (!pos || focused.current) return;
    focused.current = true;
    const all = items();
    (all.find((x) => x.getAttribute("aria-checked") === "true") ?? all[0])?.focus({ preventScroll: true });
  }, [pos]); // eslint-disable-line react-hooks/exhaustive-deps

  const close = (refocus = false) => { onClose(); if (refocus) anchor.current?.focus(); };

  // клик мимо, прокрутка списка, смена размера окна — закрываем
  useEffect(() => {
    const down = (e: MouseEvent) => {
      const t = e.target as Node;
      if (ref.current?.contains(t) || anchor.current?.contains(t)) return;
      close();
    };
    // закрываем, только если прокрутка реально сдвинула кнопку (позднее событие scroll от прокрутки «до клика» меню не трогает)
    const scroll = (e: Event) => {
      if (ref.current?.contains(e.target as Node)) return;
      const a = anchor.current?.getBoundingClientRect();
      const at = openAt.current;
      if (!a || !at || Math.abs(a.top - at.top) > 2 || Math.abs(a.left - at.left) > 2) close();
    };
    const resize = () => close();
    document.addEventListener("mousedown", down, true);
    window.addEventListener("scroll", scroll, true);
    window.addEventListener("resize", resize);
    return () => {
      document.removeEventListener("mousedown", down, true);
      window.removeEventListener("scroll", scroll, true);
      window.removeEventListener("resize", resize);
    };
  }); // eslint-disable-line react-hooks/exhaustive-deps

  const onKey = (e: React.KeyboardEvent) => {
    // клавиши меню не должны долетать до экрана (↑↓ там листают реплики, R — запись)
    e.stopPropagation();
    if (e.key === "Escape") { e.preventDefault(); if (newName !== null) setNewName(null); else close(true); return; }
    if (e.key === "Tab") { close(); return; }
    if ((e.target as HTMLElement).tagName === "INPUT") return;
    const all = items();
    const i = all.indexOf(document.activeElement as HTMLElement);
    let next = -1;
    if (e.key === "ArrowDown") next = (i + 1) % all.length;
    else if (e.key === "ArrowUp") next = (i - 1 + all.length) % all.length;
    else if (e.key === "Home") next = 0;
    else if (e.key === "End") next = all.length - 1;
    if (next >= 0) { e.preventDefault(); all[next]?.focus({ preventScroll: true }); }
  };

  const submitNew = () => {
    const v = (newName ?? "").trim();
    if (!v) return;
    onNew(v);
    close();
  };

  return createPortal(
    <div ref={ref} className={"pop whopop" + (pos?.up ? " up" : "")} role="menu" aria-label={label} data-testid="speaker-menu"
      style={{ position: "fixed", top: pos?.top ?? 0, left: pos?.left ?? 0, opacity: pos ? 1 : 0 }}
      onKeyDown={onKey} onClick={(e) => e.stopPropagation()} onMouseDown={(e) => e.stopPropagation()}>
      <div className="hd2" aria-hidden="true">Кто это говорит?</div>
      {speakers.map((s) => {
        const on = s.id === current;
        return (
          <button key={s.id} type="button" role="menuitemradio" aria-checked={on} data-mi="" tabIndex={-1}
            onClick={() => { if (!on) onPick(s.id); close(true); }}>
            <span><i className="dot" style={{ background: s.color, marginRight: 7 }} />{s.name ?? s.label}</span>{on && <Check size={12} aria-hidden="true" />}
          </button>
        );
      })}
      {newName === null ? (
        <button type="button" role="menuitem" className="ghostbtn" data-mi="" tabIndex={-1} onClick={() => setNewName("")}>+ Новый персонаж…</button>
      ) : (
        <form className="row" style={{ gap: 6, padding: "2px 4px" }} onSubmit={(e) => { e.preventDefault(); submitNew(); }}>
          <input className="inp sm grow" autoFocus value={newName} maxLength={100} placeholder="Имя персонажа" aria-label="Имя нового персонажа" onChange={(e) => setNewName(e.target.value)} />
          <button type="submit" className="btn sm primary" disabled={!newName.trim()}>Добавить</button>
        </form>
      )}
    </div>,
    document.body,
  );
}
