import { useState, type ReactNode } from "react";

interface Props { summary: ReactNode; children: ReactNode; defaultOpen?: boolean; className?: string; testId?: string }

/**
 * Раскрывающийся блок «Для опытных». Состояние держим сами, а не доверяем <details>:
 * в WebView оболочки встроенное раскрытие терялось при перерисовке, если open задавался пропсом.
 */
export function Disclosure({ summary, children, defaultOpen = false, className = "adv", testId }: Props) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <details className={className} open={open} data-testid={testId}>
      <summary aria-expanded={open} onClick={(e) => { e.preventDefault(); setOpen((v) => !v); }}>{summary}</summary>
      {children}
    </details>
  );
}
