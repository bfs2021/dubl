import { useEffect, useId, useLayoutEffect, useRef, useState, type ReactNode } from "react";
import { createPortal } from "react-dom";

const POP_W = 280;
const GAP = 6;
const EDGE = 8;

/** Где поставить подсказку: под значком, но так, чтобы не выйти за края окна; не хватает места снизу — над значком. */
export function placePopover(anchor: { left: number; top: number; bottom: number; width: number }, size: { w: number; h: number }, view: { w: number; h: number }): { left: number; top: number; above: boolean } {
  let left = anchor.left + anchor.width / 2 - size.w / 2;
  left = Math.max(EDGE, Math.min(left, view.w - size.w - EDGE));
  const below = anchor.bottom + GAP;
  const fitsBelow = below + size.h <= view.h - EDGE;
  const above = !fitsBelow && anchor.top - GAP - size.h >= EDGE;
  const top = above ? anchor.top - GAP - size.h : Math.max(EDGE, Math.min(below, view.h - size.h - EDGE));
  return { left, top, above };
}

/**
 * Маленький значок «?» с короткой подсказкой. Открывается по наведению, фокусу и клику (Esc закрывает).
 * Подсказка рисуется поверх всего окна (portal) и сама подбирает место: не обрезается краем панели с прокруткой.
 */
export function Help({ children, label = "Что это?" }: { children: ReactNode; label?: string }) {
  const [open, setOpen] = useState(false);
  const [pos, setPos] = useState<{ left: number; top: number } | null>(null);
  const id = useId();
  const ref = useRef<HTMLSpanElement>(null);
  const btn = useRef<HTMLButtonElement>(null);
  const pop = useRef<HTMLSpanElement>(null);
  useEffect(() => {
    if (!open) return;
    const off = (e: MouseEvent) => {
      const t = e.target as Node;
      if (!ref.current?.contains(t) && !pop.current?.contains(t)) setOpen(false);
    };
    const esc = (e: KeyboardEvent) => { if (e.key === "Escape") setOpen(false); };
    const close = () => setOpen(false);
    window.addEventListener("mousedown", off);
    window.addEventListener("keydown", esc);
    window.addEventListener("scroll", close, true);
    window.addEventListener("resize", close);
    return () => {
      window.removeEventListener("mousedown", off); window.removeEventListener("keydown", esc);
      window.removeEventListener("scroll", close, true); window.removeEventListener("resize", close);
    };
  }, [open]);
  useLayoutEffect(() => {
    if (!open || !btn.current) { setPos(null); return; }
    const r = btn.current.getBoundingClientRect();
    const el = pop.current;
    const size = { w: el?.offsetWidth || POP_W, h: el?.offsetHeight || 80 };
    const p = placePopover(r, size, { w: window.innerWidth, h: window.innerHeight });
    setPos({ left: p.left, top: p.top });
  }, [open, children]);
  return (
    <span className="help" ref={ref} onMouseEnter={() => setOpen(true)} onMouseLeave={() => setOpen(false)}>
      <button ref={btn} type="button" className="helpbtn" aria-label={label} aria-expanded={open} aria-describedby={open ? id : undefined}
        onClick={(e) => { e.stopPropagation(); setOpen((v) => !v); }} onFocus={() => setOpen(true)} onBlur={() => setOpen(false)}>?</button>
      {open && createPortal(
        <span id={id} ref={pop} role="tooltip" className="helppop" style={pos ? { left: pos.left, top: pos.top } : { left: -9999, top: 0 }}
          onMouseEnter={() => setOpen(true)} onMouseLeave={() => setOpen(false)}>{children}</span>,
        document.body,
      )}
    </span>
  );
}
