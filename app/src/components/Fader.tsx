interface Props {
  value: number;
  onChange: (v: number) => void;
  min?: number;
  max?: number;
  step?: number;
  label: string;
  vertical?: boolean;
  /** двойной щелчок возвращает сюда */
  neutral?: number;
  disabled?: boolean;
}

/** Ползунок (громкость, сдвиг…). Двойной щелчок — сброс. */
export function Fader({ value, onChange, min = -30, max = 12, step = 0.5, label, vertical, neutral = 0, disabled }: Props) {
  return (
    <input type="range" className={"fader" + (vertical ? " vert" : "")} min={min} max={max} step={step} value={value} aria-label={label}
      title={label + " (двойной щелчок — сбросить)"} disabled={disabled}
      onChange={(e) => onChange(Number(e.target.value))} onDoubleClick={() => onChange(neutral)} />
  );
}
