import type { ReactNode } from "react";

export interface SegOption<T extends string | number> {
  value: T;
  label: ReactNode;
  title?: string;
  disabled?: boolean;
}

interface Props<T extends string | number> {
  options: SegOption<T>[];
  value: T;
  onChange: (v: T) => void;
  label?: string;
  className?: string;
}

/** Переключатель из нескольких вариантов (радио-группа). */
export function Segmented<T extends string | number>({ options, value, onChange, label, className }: Props<T>) {
  return (
    <div className={"seg " + (className ?? "")} role="radiogroup" aria-label={label}>
      {options.map((o) => (
        <button key={String(o.value)} type="button" role="radio" aria-checked={o.value === value} className={o.value === value ? "on" : ""}
          title={o.title} disabled={o.disabled} onClick={() => onChange(o.value)}>
          {o.label}
        </button>
      ))}
    </div>
  );
}
