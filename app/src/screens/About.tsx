import { useEffect, useMemo, useState } from "react";
import { ClipboardCopy, ExternalLink, Search } from "lucide-react";
import aboutMd from "../../../docs/ABOUT.md?raw";
import noticesMd from "../../../THIRD-PARTY-NOTICES.md?raw";
import { useStore } from "../state/store";
import { Markdown, plainText, splitSections } from "../lib/markdown";
import { ISSUES_URL, LICENSE_NAME, RELEASES_URL, REPO_URL, appVersion, reportText } from "../lib/about";
import { copyText, openExternal } from "../lib/external";
import { deviceName } from "../lib/format";

/** Настройки → О программе: версия, лицензия, справка (docs/ABOUT.md) и сторонние компоненты (THIRD-PARTY-NOTICES.md). */
export function AboutPage() {
  const health = useStore((s) => s.health);
  const demo = useStore((s) => s.demo);
  const toast = useStore((s) => s.toast);
  const [appV, setAppV] = useState<string>("…");
  const [query, setQuery] = useState("");
  const [showNotices, setShowNotices] = useState(false);

  useEffect(() => { void appVersion().then(setAppV); }, []);

  const sections = useMemo(() => splitSections(aboutMd).map((s) => ({ ...s, text: plainText(s.title + " " + s.body).toLowerCase() })), []);
  const q = query.trim().toLowerCase();
  const found = q ? sections.filter((s) => s.text.includes(q)) : sections;
  const toc = found.filter((s) => s.title);

  const engineV = !demo && health ? health.version : null;
  const mismatch = !!engineV && appV !== "…" && engineV !== appV;

  const copyReport = async () => {
    const ok = await copyText(reportText(appV, health, demo));
    toast(ok ? "Сведения скопированы. Вставь их в сообщение об ошибке." : "Не удалось скопировать.", ok ? "ok" : "error");
  };
  const jump = (id: string) => document.getElementById(`about-${id}`)?.scrollIntoView({ behavior: "smooth", block: "start" });
  const link = (url: string, text: string, testId: string) => (
    <button type="button" className="btn" onClick={() => void openExternal(url)} data-testid={testId}><ExternalLink size={13} />{text}</button>
  );

  return (
    <div className="about" data-testid="about-page">
      <h1>О программе</h1>
      <div className="aboutcard panel-in">
        <div>
          <div className="aname">Дубль</div>
          <div className="mono" data-testid="about-version">версия {appV}</div>
          {engineV && (
            <div className="note" data-testid="about-engine">
              Движок: {engineV}{mismatch ? " — не совпадает с версией программы, так бывает после неполного обновления" : ""} · {deviceName(health?.device)}
            </div>
          )}
          {!engineV && <div className="note" data-testid="about-engine">Движок не подключён (демо).</div>}
        </div>
        <button className="btn primary" onClick={() => void copyReport()} data-testid="about-copy"><ClipboardCopy size={13} />Скопировать сведения для сообщения об ошибке</button>
      </div>

      <p className="lead" style={{ marginTop: 14 }}>
        Дубль — бесплатная программа для любительской озвучки с открытым исходным кодом. Лицензия — GNU GPL v3 или новее ({LICENSE_NAME}): программу можно
        использовать, менять и распространять, а изменённые версии тоже должны оставаться открытыми. Права на видео, которое ты озвучиваешь, остаются у его владельцев.
      </p>
      <div className="row" style={{ flexWrap: "wrap", gap: 8 }}>
        {link(REPO_URL, "Исходный код на GitHub", "about-src")}
        {link(ISSUES_URL, "Сообщить об ошибке", "about-issues")}
        {link(RELEASES_URL, "Что нового", "about-releases")}
      </div>

      <div className="sect"><h2>Справка</h2></div>
      <label className="aboutsearch">
        <Search size={14} />
        <input className="inp" type="search" placeholder="Поиск по справке" aria-label="Поиск по справке" value={query} onChange={(e) => setQuery(e.target.value)} data-testid="help-search" />
      </label>
      {toc.length > 0 && (
        <nav className="abouttoc" aria-label="Оглавление справки" data-testid="help-toc">
          {toc.map((s) => <button key={s.id} type="button" className="link" onClick={() => jump(s.id)}>{s.title}</button>)}
        </nav>
      )}
      {found.length === 0 && <p className="note" data-testid="help-empty">По запросу «{query.trim()}» в справке ничего нет.</p>}
      <div data-testid="help-body">
        {found.map((s) => (
          <section key={s.id} id={`about-${s.id}`} className="helpsec">
            {s.title && <h2 className="helph">{s.title}</h2>}
            <Markdown md={s.body} />
          </section>
        ))}
      </div>

      <div className="sect"><h2>Сторонние компоненты</h2></div>
      <p className="note" style={{ marginBottom: 8 }}>
        В программе работают чужие открытые библиотеки и модели: ffmpeg (GPL), Whisper, GigaAM, Demucs, yt-dlp, Tauri, React и другие. Лицензии — в списке ниже; он же лежит в файле THIRD-PARTY-NOTICES.md в репозитории.
      </p>
      <button className="btn" onClick={() => setShowNotices((v) => !v)} aria-expanded={showNotices} data-testid="notices-toggle">{showNotices ? "Скрыть список" : "Показать список лицензий"}</button>
      {showNotices && <div data-testid="notices-body" style={{ marginTop: 10 }}><Markdown md={noticesMd} /></div>}
    </div>
  );
}
