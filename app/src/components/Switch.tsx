interface Props {
  checked: boolean;
  onChange: (v: boolean) => void;
  label: string;
  disabled?: boolean;
}

/** Переключатель. `label` — для скринридеров и подсказки. */
export function Switch({ checked, onChange, label, disabled }: Props) {
  return (
    <span className="sw">
      <input type="checkbox" role="switch" aria-label={label} title={label} checked={checked} disabled={disabled}
        onChange={(e) => onChange(e.target.checked)} />
      <i />
    </span>
  );
}
