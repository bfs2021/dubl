interface Props {
  /** пик в дБ (−60…0) */
  db: number;
  /** удержанный максимум (метка), дБ */
  hold?: number;
  className?: string;
}

/** Границы цветных зон на шкале, % ширины: −60…−24 тихо, −24…−8 хорошо, −8…−3 громковато, выше — перегруз. */
const pct = (db: number) => Math.max(0, Math.min(100, ((db + 60) / 60) * 100));

/** Шкала уровня с цветной зоной «сюда попадай». Заливка — текущий уровень, чёрточка — недавний максимум. */
export function ZoneMeter({ db, hold, className }: Props) {
  return (
    <div className={"zonemeter " + (className ?? "")} role="meter" aria-label="Уровень голоса" aria-valuemin={-60} aria-valuemax={0} aria-valuenow={Math.round(Math.max(-60, db))}>
      <div className="zm-zones" aria-hidden="true">
        <i className="z-low" style={{ width: pct(-24) + "%" }} />
        <i className="z-good" style={{ width: pct(-8) - pct(-24) + "%" }} />
        <i className="z-hot" style={{ width: pct(-3) - pct(-8) + "%" }} />
        <i className="z-clip" style={{ width: 100 - pct(-3) + "%" }} />
      </div>
      <div className="zm-fill" style={{ width: pct(db) + "%" }} />
      {hold !== undefined && hold > -60 && <b className="zm-hold" style={{ left: pct(hold) + "%" }} />}
    </div>
  );
}
