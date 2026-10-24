import { useNavigate } from "react-router-dom";

/** Пояснение рядом с неактивной кнопкой ИИ: «Подключи ИИ в Настройках». */
export function AiHint({ className }: { className?: string }) {
  const nav = useNavigate();
  return (
    <span className={"aihint " + (className ?? "")}>
      Подключи ИИ в <button type="button" className="link" onClick={(e) => { e.stopPropagation(); nav("/settings/ai"); }}>Настройках</button>
    </span>
  );
}

export const AI_OFF_TITLE = "Подключи ИИ в Настройках";
