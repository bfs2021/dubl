import { X } from "lucide-react";
import { useLocation } from "react-router-dom";
import { useStore } from "../state/store";

export function ToastHost() {
  const toasts = useStore((s) => s.toasts);
  const dismiss = useStore((s) => s.dismissToast);
  // на «Рендере» внизу справа стоит главная кнопка — плашки поднимаем над ней
  const onRender = /\/render$/.test(useLocation().pathname);
  return (
    <div className={"toasts" + (onRender ? " raised" : "")} role="status" aria-live="polite">
      {toasts.map((t) => (
        <div key={t.id} className={"toast " + t.kind}>
          <span>{t.text}</span>
          {t.action && (
            <button className="btn sm toastact" onClick={() => { dismiss(t.id); t.action!.run(); }}>{t.action.label}</button>
          )}
          <button className="btn ghost sm iconbtn" style={{ color: "inherit" }} aria-label="Закрыть" onClick={() => dismiss(t.id)}><X size={14} /></button>
        </div>
      ))}
    </div>
  );
}
