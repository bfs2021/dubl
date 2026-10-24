import { Captions, ListMusic, Music2, Sparkles } from "lucide-react";
import type { MediaInfo } from "../api/types";
import { langName } from "../lib/files";
import { fmtBytes, fmtLong, plural } from "../lib/format";

interface Props {
  media: MediaInfo;
  audioIdx: number | null;
  onAudioIdx?: (idx: number) => void;
}

/** Что нашли в файле: звуковые дорожки (язык, 5.1), субтитры (язык, ASS с именами), главы. Выбор дорожки оригинала. */
export function FoundInFile({ media, audioIdx, onAudioIdx }: Props) {
  const nothing = media.audio_streams.length <= 1 && !media.subtitle_streams.length && !media.chapters.length;
  const layoutLabel = (a: MediaInfo["audio_streams"][number]) => (a.channel_layout === "5.1" || a.channels >= 6 ? "5.1" : a.channels === 1 ? "моно" : "стерео");
  const textSubs = media.subtitle_streams.filter((s) => s.is_text);
  const picSubs = media.subtitle_streams.length - textSubs.length;
  return (
    <div className="found" aria-label="Что нашли в файле">
      <div className="between">
        <h2>Что нашли в файле</h2>
        <span className="mono dim" style={{ fontSize: 11.5 }}>{fmtLong(media.duration)} · {media.width}×{media.height}{media.size_bytes ? ` · ${fmtBytes(media.size_bytes)}` : ""}</span>
      </div>

      {nothing && (
        <div className="foundrow">
          <Sparkles size={15} className="amberic" />
          <div><b>Всё разложим сами.</b> <span className="mute">В файле одна дорожка, без субтитров и глав — текст распознаем на слух, голоса различим сами, склейки найдём по картинке.</span></div>
        </div>
      )}

      <div className="foundrow">
        <Music2 size={15} className="mute" />
        <div className="grow">
          <div className="mute fl">Звук {media.audio_streams.length > 1 ? "— выбери дорожку оригинала" : ""}</div>
          {media.audio_streams.some((a) => a.language) && <div className="note" data-testid="lang-hint" style={{ marginBottom: 4 }}>Язык у дорожки — только подпись в файле, она бывает неверной. Если язык оригинала не выбран, программа определит его по речи.</div>}
          {media.audio_streams.map((a, i) => (
            <label key={a.index} className={"opt-row" + (audioIdx === a.index ? " on" : "")}>
              <input type="radio" name="audio-track" checked={audioIdx === a.index} disabled={!onAudioIdx || media.audio_streams.length < 2}
                onChange={() => onAudioIdx?.(a.index)} />
              <span className="grow ell">{a.title || (media.audio_streams.length > 1 ? `Дорожка ${i + 1}` : "Звуковая дорожка")}</span>
              {a.language && <span className="tag" title="Так подписана дорожка в файле. Подпись бывает неверной — язык программа услышит по речи.">в файле: {langName(a.language)}</span>}
              <span className={"tag" + (layoutLabel(a) === "5.1" ? " amber" : "")} title={layoutLabel(a) === "5.1" ? "Центральный канал — почти одни диалоги, это ускоряет разделение голосов" : undefined}>{layoutLabel(a)}</span>
            </label>
          ))}
        </div>
      </div>

      {media.subtitle_streams.length > 0 && (
        <div className="foundrow">
          <Captions size={15} className="mute" />
          <div className="grow">
            <div className="mute fl">Субтитры{textSubs.length ? " — возьмём готовый текст и тайминги" : ""}</div>
            {textSubs.map((s) => (
              <div key={s.index} className="opt-row static">
                <span className="grow ell">{s.title || langName(s.language)}</span>
                <span className="tag">{langName(s.language)}</span>
                <span className="tag">{s.codec === "ass" || s.codec === "ssa" ? "ASS" : s.codec === "subrip" ? "SRT" : s.codec}</span>
                {(s.codec === "ass" || s.codec === "ssa") && <span className="tag amber" title="В ASS обычно есть поле Name — имя персонажа">с именами</span>}
              </div>
            ))}
            {picSubs > 0 && <div className="note" style={{ marginTop: 4 }}>Ещё {picSubs} {plural(picSubs, "дорожка", "дорожки", "дорожек")} с картинками (Blu-ray) — текст из них пока не читаем.</div>}
          </div>
        </div>
      )}

      {media.chapters.length > 0 && (
        <div className="foundrow">
          <ListMusic size={15} className="mute" />
          <div className="grow">
            <div className="mute fl">Главы — готовые точки для нарезки</div>
            <div className="row wrap" style={{ gap: 5 }}>
              {media.chapters.slice(0, 8).map((c, i) => <span key={i} className="tag">{c.title ?? `Глава ${i + 1}`}</span>)}
              {media.chapters.length > 8 && <span className="tag">+{media.chapters.length - 8}</span>}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
