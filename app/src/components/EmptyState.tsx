import type { ReactNode } from "react";

export function EmptyState({ title, text, children }: { title: string; text?: string; children?: ReactNode }) {
  return (
    <div className="empty">
      <h3>{title}</h3>
      {text && <p className="note">{text}</p>}
      {children}
    </div>
  );
}
