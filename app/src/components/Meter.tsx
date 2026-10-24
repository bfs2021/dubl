interface Props {
  /** уровень в дБ (−60…0) */
  db: number;
  vertical?: boolean;
  height?: number;
}

export const dbToPct = (db: number) => Math.max(0, Math.min(100, ((db + 60) / 60) * 100));

/** Индикатор уровня: зелёный → жёлтый → красный. */
export function Meter({ db, vertical, height }: Props) {
  const style = { ["--v" as string]: `${dbToPct(db)}%`, ...(height ? { height } : {}) } as React.CSSProperties;
  return (
    <div className={vertical ? "vmeter" : "meter"} style={style} role="meter" aria-valuemin={-60} aria-valuemax={0}
      aria-valuenow={Math.round(Math.max(-60, db))} aria-label="Уровень">
      <i />
    </div>
  );
}
