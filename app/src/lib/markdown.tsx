import type { ReactNode } from "react";
import { openExternal } from "./external";

/**
 * Мини-рендер markdown в React-элементы: заголовки, абзацы, списки, таблицы, цитаты, код, **жирный**, *курсив*, `код`, [ссылки](https://…).
 * HTML не вставляется никогда (текст всегда идёт как текст), так что чужая разметка в справке ничего не сломает.
 * Ссылки — только http(s), они открываются во внешнем браузере; остальные (относительные на другие файлы) показываются просто текстом.
 */

/** Убирает HTML-комментарии (в справке в них заметки для разработчиков: `<!-- dev … -->`). */
export function stripComments(md: string): string {
  return md.replace(/<!--[\s\S]*?-->/g, "");
}

/** Переводы строк к \n: на Windows git по умолчанию отдаёт файлы с \r\n, а `.` в регулярках не ест \r. */
const normalize = (md: string) => stripComments(md.replace(/\r\n?/g, "\n"));

export interface Section { id: string; title: string; body: string }

/** Режет документ по заголовкам второго уровня. Всё до первого «##» — вступление (title пустой). */
export function splitSections(md: string): Section[] {
  const out: Section[] = [];
  let cur: Section = { id: "intro", title: "", body: "" };
  let fence = false;
  for (const line of normalize(md).split("\n")) {
    if (/^```/.test(line)) fence = !fence;
    const m = !fence && line.match(/^##\s+(.+?)\s*$/);
    if (m) {
      out.push(cur);
      cur = { id: `sec-${out.length}`, title: m[1], body: "" };
    } else cur.body += line + "\n";
  }
  out.push(cur);
  return out.filter((s) => s.title || s.body.trim());
}

const INLINE = /(\*\*[^*]+\*\*|`[^`]+`|\[[^\]]+\]\([^)\s]+\)|\*[^*\s][^*]*\*)/g;

function inline(text: string, key: string): ReactNode[] {
  const out: ReactNode[] = [];
  text.split(INLINE).forEach((part, i) => {
    if (!part) return;
    const k = `${key}-${i}`;
    let m: RegExpMatchArray | null;
    if ((m = part.match(/^\*\*([^*]+)\*\*$/))) out.push(<b key={k}>{inline(m[1], k)}</b>);
    else if ((m = part.match(/^`([^`]+)`$/))) out.push(<code key={k} className="mono">{m[1]}</code>);
    else if ((m = part.match(/^\[([^\]]+)\]\(([^)\s]+)\)$/))) {
      const url = m[2];
      if (/^https?:\/\//i.test(url)) {
        out.push(<a key={k} href={url} className="link" onClick={(e) => { e.preventDefault(); void openExternal(url); }}>{m[1]}</a>);
      } else out.push(<span key={k}>{m[1]}</span>);
    } else if ((m = part.match(/^\*([^*\s][^*]*)\*$/))) out.push(<i key={k}>{m[1]}</i>);
    else out.push(part);
  });
  return out;
}

const cells = (line: string) => line.trim().replace(/^\||\|$/g, "").split("|").map((c) => c.trim());

export function Markdown({ md, className }: { md: string; className?: string }) {
  const lines = normalize(md).split("\n");
  const nodes: ReactNode[] = [];
  let i = 0;
  let n = 0;
  const key = () => `b${n++}`;
  const isBlockStart = (l: string) => /^(#{1,4}\s|\s*[-*]\s|\s*\d+\.\s|>|```|\||---+\s*$)/.test(l);

  while (i < lines.length) {
    const line = lines[i];
    if (!line.trim()) { i++; continue; }
    let m: RegExpMatchArray | null;

    if (/^```/.test(line)) {
      const buf: string[] = [];
      i++;
      while (i < lines.length && !/^```/.test(lines[i])) buf.push(lines[i++]);
      i++;
      nodes.push(<pre key={key()} className="mono md-pre">{buf.join("\n")}</pre>);
    } else if ((m = line.match(/^(#{1,4})\s+(.+?)\s*$/))) {
      const level = Math.min(m[1].length + 1, 5);          // # в справке — название документа; в странице уже есть своё h1
      const Tag = `h${level}` as "h2" | "h3" | "h4" | "h5";
      nodes.push(<Tag key={key()}>{inline(m[2], `h${n}`)}</Tag>);
      i++;
    } else if (/^---+\s*$/.test(line)) {
      nodes.push(<hr key={key()} />);
      i++;
    } else if (line.startsWith("|") && /^\s*\|?[\s:|-]+\|?\s*$/.test(lines[i + 1] ?? "") && (lines[i + 1] ?? "").includes("-")) {
      const head = cells(line);
      i += 2;
      const rows: string[][] = [];
      while (i < lines.length && lines[i].startsWith("|")) rows.push(cells(lines[i++]));
      const k = key();
      nodes.push(
        <div key={k} className="md-table"><table><thead><tr>{head.map((c, j) => <th key={j}>{inline(c, `${k}h${j}`)}</th>)}</tr></thead>
          <tbody>{rows.map((r, ri) => <tr key={ri}>{r.map((c, j) => <td key={j}>{inline(c, `${k}r${ri}c${j}`)}</td>)}</tr>)}</tbody></table></div>,
      );
    } else if (line.startsWith(">")) {
      const buf: string[] = [];
      while (i < lines.length && lines[i].startsWith(">")) buf.push(lines[i++].replace(/^>\s?/, ""));
      nodes.push(<blockquote key={key()}>{inline(buf.join(" "), `q${n}`)}</blockquote>);
    } else if (/^\s*([-*]|\d+\.)\s/.test(line)) {
      const ordered = /^\s*\d+\./.test(line);
      const items: { text: string; sub: string[] }[] = [];
      while (i < lines.length && lines[i].trim()) {
        const l = lines[i];
        const top = l.match(/^(?:[-*]|\d+\.)\s+(.*)$/);
        const nested = l.match(/^\s+(?:[-*]|\d+\.)\s+(.*)$/);
        if (top) items.push({ text: top[1], sub: [] });
        else if (nested && items.length) items[items.length - 1].sub.push(nested[1]);
        else if (items.length && /^\s+\S/.test(l)) items[items.length - 1].text += " " + l.trim();      // продолжение пункта
        else break;
        i++;
      }
      if (items.length === 0) { nodes.push(<p key={key()}>{inline(line.trim(), `p${n}`)}</p>); i++; continue; }   // страховка: цикл всегда двигается
      const k = key();
      const List = ordered ? "ol" : "ul";
      nodes.push(
        <List key={k}>{items.map((it, ii) => (
          <li key={ii}>{inline(it.text, `${k}i${ii}`)}
            {it.sub.length > 0 && <ul>{it.sub.map((s, si) => <li key={si}>{inline(s, `${k}i${ii}s${si}`)}</li>)}</ul>}
          </li>
        ))}</List>,
      );
    } else {
      const buf: string[] = [];
      while (i < lines.length && lines[i].trim() && (buf.length === 0 || !isBlockStart(lines[i]))) buf.push(lines[i++].trim());
      nodes.push(<p key={key()}>{inline(buf.join(" "), `p${n}`)}</p>);
    }
  }
  return <div className={"md" + (className ? " " + className : "")}>{nodes}</div>;
}

/** Только текст, без разметки: для поиска по справке. */
export function plainText(md: string): string {
  return normalize(md).replace(/\[([^\]]+)\]\([^)]*\)/g, "$1").replace(/[*`|>#]/g, " ").replace(/\s+/g, " ").trim();
}
