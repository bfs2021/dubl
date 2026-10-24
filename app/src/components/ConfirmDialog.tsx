import { useEffect, useId, useRef } from "react";
import { useStore } from "../state/store";

/**
 * Своё окно «Точно?» вместо window.confirm: в WebView Tauri 2 confirm/alert/prompt молча возвращают false.
 * Фокус — на кнопке подтверждения (у опасных действий — на «Отмена»), Esc и клик мимо = отмена, Tab не уходит за окно.
 */
export function ConfirmHost() {
  const dlg = useStore((s) => s.dialog);
  const resolve = useStore((s) => s.resolveConfirm);
  const id = useId();
  const box = useRef<HTMLDivElement>(null);
  const cancelRef = useRef<HTMLButtonElement>(null);
  const okRef = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    if (!dlg) return;
    const prev = document.activeElement as HTMLElement | null;
    // по умолчанию в фокусе безопасная кнопка: случайный Enter не удалит проект
    (dlg.danger ? cancelRef.current : okRef.current)?.focus();
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") { e.preventDefault(); e.stopPropagation(); resolve(false); return; }
      if (e.key !== "Tab") return;
      const items = [...(box.current?.querySelectorAll<HTMLElement>("button") ?? [])];
      if (!items.length) return;
      const first = items[0], last = items[items.length - 1];
      if (!box.current?.contains(document.activeElement)) { e.preventDefault(); first.focus(); }
      else if (e.shiftKey && document.activeElement === first) { e.preventDefault(); last.focus(); }
      else if (!e.shiftKey && document.activeElement === last) { e.preventDefault(); first.focus(); }
    };
    window.addEventListener("keydown", onKey, true);
    return () => { window.removeEventListener("keydown", onKey, true); prev?.focus?.(); };
  }, [dlg, resolve]);

  if (!dlg) return null;
  return (
    <div className="modal-back" onMouseDown={(e) => { if (e.target === e.currentTarget) resolve(false); }}>
      <div className="modal" role="alertdialog" aria-modal="true" aria-labelledby={id + "t"} aria-describedby={dlg.text ? id + "d" : undefined} ref={box}>
        <h3 id={id + "t"}>{dlg.title}</h3>
        {dlg.text && <p id={id + "d"} className="note">{dlg.text}</p>}
        <div className="modal-actions">
          <button className="btn" ref={cancelRef} onClick={() => resolve(false)}>{dlg.cancelLabel ?? "Отмена"}</button>
          <button className={"btn " + (dlg.danger ? "danger-solid" : "primary")} ref={okRef} onClick={() => resolve(true)}>{dlg.confirmLabel}</button>
        </div>
      </div>
    </div>
  );
}
