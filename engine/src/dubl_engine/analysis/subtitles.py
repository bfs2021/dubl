"""Субтитры → реплики (SRT и ASS). Свой разбор, без внешних библиотек.

Что делаем с «реальными» файлами:
* кодировки: UTF-8 (с BOM и без), UTF-16, cp1251 (старые русские), cp1252 (западные); CRLF/CR/LF;
* из ASS отбрасываем то, что не диалог: песни OP/ED/Song/Karaoke, надписи Sign/Title, рисунки {\\p1},
  движущийся текст {\\move}, позиционированные надписи {\\pos} (кроме стилей, похожих на диалог), караоке {\\k};
* теги {\\...}, HTML-теги (<i>, <font ...>), \\N → пробел, звуковые пометки [музыка] и строки с ♪ убираем;
* повторы (одинаковые время и текст — типично для слоёв ASS) схлопываем.
"""
from __future__ import annotations

import html
import logging
import re
from pathlib import Path

from ..models import Line, Speaker
from ..util import EngineError
from .palette import PALETTE

log = logging.getLogger(__name__)

NAME_PREFIX = "name:"

_TAG = re.compile(r"\{[^}]*\}")
_HTML_TAG = re.compile(r"</?[A-Za-z][^<>]{0,200}>")
_BREAK = re.compile(r"\\[Nn]|[\r\n]+")
_SDH = re.compile(r"\[[^\]]*\]")
_SPACES = re.compile(r"\s+")
_SRT_TIME = re.compile(
    r"(\d+):(\d{1,2}):(\d{1,2})[,.](\d{1,3})\s*-->\s*(\d+):(\d{1,2}):(\d{1,2})[,.](\d{1,3})")
_ASS_TIME = re.compile(r"^(\d+):(\d{1,2}):(\d{1,2})(?:[.,](\d{1,3}))?$")
_KARAOKE = re.compile(r"\\[kK][fo]?\d")
_POS_TAG = re.compile(r"\\(?:pos|move|org)\s*\(")
_DRAWING = re.compile(r"\\p[1-9]")
_MUSIC = re.compile(r"[♪♫♬]")

# Стили, которые не диалог (сравнение по «словам» названия стиля)
_NON_DIALOGUE_WORDS = {
    "op", "ed", "sign", "signs", "song", "songs", "karaoke", "kara", "credit", "credits", "lyric", "lyrics",
    "romaji", "kanji", "insert", "title", "titles", "eyecatch", "staff", "opening", "ending", "tl", "note", "notes",
    "надпись", "надписи", "песня", "песни", "титры", "опенинг", "эндинг", "караоке",
}
_NON_DIALOGUE_PREFIX = ("song", "kara", "lyric", "credit", "opening", "ending", "romaji", "eyecatch")
_NON_DIALOGUE_NUM = re.compile(r"^(op|ed|sign|song)\d+[a-z]?$")
# Стили, где {\pos} — просто положение обычной реплики на экране
_DIALOGUE_WORDS = {"default", "main", "dialog", "dialogue", "narrator", "narration", "thought", "thoughts", "italic",
                   "italics", "top", "overlap", "flashback", "phone", "tv", "radio", "диалог", "основной"}


def _read_text(path: Path) -> str:
    if not path.exists():
        raise EngineError(f"Файл субтитров не найден: {path}")
    raw = path.read_bytes()
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return raw.decode("utf-16", errors="replace")
    if raw[:3] == b"\xef\xbb\xbf":
        return raw[3:].decode("utf-8", errors="replace")
    if len(raw) >= 4 and raw[1:2] == b"\x00" and raw[3:4] == b"\x00":        # UTF-16 LE без BOM
        return raw.decode("utf-16-le", errors="replace")
    if len(raw) >= 4 and raw[0:1] == b"\x00" and raw[2:3] == b"\x00":        # UTF-16 BE без BOM
        return raw.decode("utf-16-be", errors="replace")
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        pass
    # однобайтовые: кириллица (cp1251) или Запад (cp1252). Смотрим, какая даёт больше букв нужного вида.
    cyr = raw.decode("cp1251", errors="replace")
    west = raw.decode("cp1252", errors="replace")

    def score(t: str) -> float:
        high = [c for c in t if ord(c) > 127]
        if not high:
            return 0.0
        good = sum(1 for c in high if c.isalpha() and (("а" <= c.lower() <= "я") or c in "ёЁ"
                                                       or "\u00c0" <= c <= "\u00ff"))
        return good / len(high) - 0.5 * t.count("\ufffd") / max(len(high), 1)

    # Байты E0–FF — «а–я» в cp1251 и «à–ÿ» в cp1252. Различаем по словам: в русском тексте кириллица стоит
    # в целых словах, а «Cafй crиme» (латиница с редкими кириллическими буквами внутри слов) — это cp1252.
    words = re.findall(r"[A-Za-z\u0400-\u04ff]+", cyr)
    cyr_words = [w for w in words if re.search(r"[\u0400-\u04ff]", w)]
    mixed = [w for w in cyr_words if re.search(r"[A-Za-z]", w)]
    if cyr_words and len(mixed) / len(cyr_words) > 0.5:
        return west
    return cyr if score(cyr) >= 0.5 else (west if score(west) > 0 else cyr)


def clean_text(text: str) -> str:
    """Убрать теги {\\...}, <i>, звуковые пометки [..], переносы строк → пробелы."""
    text = _TAG.sub("", text)
    text = _BREAK.sub(" ", text)
    text = text.replace("\\h", " ")
    text = _HTML_TAG.sub("", text)
    text = _SDH.sub("", text)
    text = html.unescape(text).replace("\u00a0", " ")
    return _SPACES.sub(" ", text).strip()


def _frac(s: str | None) -> float:
    return int(s.ljust(3, "0")[:3]) / 1000.0 if s else 0.0


def _parse_srt(text: str) -> list[tuple[float, float, str, str | None]]:
    """Построчный разбор: устойчив к пропущенным пустым строкам между репликами и к номерам без времени."""
    out: list[tuple[float, float, str, str | None]] = []
    rows = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    cur: list | None = None

    def flush() -> None:
        nonlocal cur
        if cur is not None:
            out.append((cur[0], cur[1], " ".join(cur[2]), None))
        cur = None

    for i, row in enumerate(rows):
        m = _SRT_TIME.search(row)
        if m:
            flush()
            g = m.groups()
            start = int(g[0]) * 3600 + int(g[1]) * 60 + int(g[2]) + _frac(g[3])
            end = int(g[4]) * 3600 + int(g[5]) * 60 + int(g[6]) + _frac(g[7])
            cur = [start, end, []]
            continue
        if not row.strip():
            flush()
            continue
        if cur is None:
            continue                                        # номер реплики или мусор до первого времени
        if row.strip().isdigit() and i + 1 < len(rows) and _SRT_TIME.search(rows[i + 1]):
            continue                                        # номер следующей реплики (пустую строку забыли)
        cur[2].append(row)
    flush()
    return out


def _ass_time(s: str) -> float:
    m = _ASS_TIME.match(s.strip())
    if not m:
        raise ValueError(s)
    return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + int(m.group(3)) + _frac(m.group(4))


def is_non_dialogue_style(style: str) -> bool:
    """Стиль песни/надписи/титров (OP, ED, Sign, Song, Karaoke…)."""
    words = [w for w in re.split(r"[^0-9a-zA-Zа-яА-ЯёЁ]+", style.lower()) if w]
    for w in words:
        if w in _NON_DIALOGUE_WORDS or _NON_DIALOGUE_NUM.match(w) or w.startswith(_NON_DIALOGUE_PREFIX):
            return True
    return False


def _is_dialogue_style(style: str) -> bool:
    words = [w for w in re.split(r"[^0-9a-zA-Zа-яА-ЯёЁ]+", style.lower()) if w]
    return any(w in _DIALOGUE_WORDS for w in words)


def _ass_is_dialogue(style: str, raw_text: str, has_name: bool = False) -> bool:
    if is_non_dialogue_style(style):
        return False
    if _DRAWING.search(raw_text):                             # векторный рисунок, а не текст
        return False
    if len(_KARAOKE.findall(raw_text)) >= 2:                  # караоке по слогам
        return False
    if re.search(r"\\move\s*\(", raw_text):
        return False
    if _POS_TAG.search(raw_text):          # {\pos}: надпись на экране, если это не реплика явного персонажа
        return _is_dialogue_style(style) and has_name
    return True


def _parse_ass(text: str) -> list[tuple[float, float, str, str | None]]:
    out: list[tuple[float, float, str, str | None]] = []
    in_events = False
    fmt: list[str] = []
    skipped = 0
    for raw in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        line = raw.strip()
        if line.startswith("["):
            in_events = line.lower() == "[events]"
            continue
        if not in_events:
            continue
        key, _, rest = line.partition(":")
        key = key.strip().lower()
        if key == "format":
            fmt = [f.strip().lower() for f in rest.split(",")]
        elif key == "dialogue" and fmt:
            parts = rest.lstrip().split(",", len(fmt) - 1)
            if len(parts) < len(fmt):
                continue
            row = dict(zip(fmt, parts, strict=False))
            try:
                start, end = _ass_time(row["start"]), _ass_time(row["end"])
            except (KeyError, ValueError):
                continue
            body = row.get("text", "")
            name = (row.get("name") or row.get("actor") or "").strip()
            if name in ("-", "?", "—"):
                name = ""
            if not _ass_is_dialogue((row.get("style") or "").strip().lstrip("*"), body, bool(name)):
                skipped += 1
                continue
            out.append((start, end, body, name or None))
    if skipped:
        log.info("ASS: пропущено %d строк, не относящихся к диалогу (песни, надписи, рисунки)", skipped)
    return out


def parse_subtitle_file(path: str | Path) -> list[Line]:
    """SRT/ASS → Line (source="embedded"). Имя из ASS кладём в speaker_id как "name:<Имя>"."""
    p = Path(path)
    text = _read_text(p).lstrip("\ufeff")
    is_ass = p.suffix.lower() in (".ass", ".ssa") or "[events]" in text.lower()
    raw = _parse_ass(text) if is_ass else _parse_srt(text)
    rows: list[tuple[float, float, str, str | None]] = []
    seen: set[tuple[float, float, str]] = set()
    canon: dict[str, str] = {}                  # "barsik"/"Barsik"/"BARSIK" — один персонаж
    for start, end, body, name in raw:
        if _MUSIC.search(body):                 # строки с ♪ — песни, не реплики
            continue
        cleaned = clean_text(body)
        if not cleaned or end <= start or start < 0:
            continue
        if not any(c.isalnum() for c in cleaned):
            continue
        key = (round(start, 3), round(end, 3), cleaned)
        if key in seen:
            continue
        seen.add(key)
        if name:
            name = canon.setdefault(_SPACES.sub(" ", name).casefold(), _SPACES.sub(" ", name))
        rows.append((start, end, cleaned, name))
    rows.sort(key=lambda r: (r[0], r[1]))
    return [
        Line(id=f"l{i:04d}", start=round(s, 3), end=round(e, 3), text=t,
             speaker_id=f"{NAME_PREFIX}{n}" if n else None, source="embedded")
        for i, (s, e, t, n) in enumerate(rows, 1)
    ]


def speakers_from_names(lines: list[Line]) -> tuple[list[Line], list[Speaker]]:
    """"name:<X>" → Speaker(s1..), цвета из PALETTE по порядку появления. Остальные speaker_id не трогаем."""
    ids: dict[str, str] = {}
    order: list[str] = []
    for ln in lines:
        sid = ln.speaker_id
        if sid and sid.startswith(NAME_PREFIX):
            name = sid[len(NAME_PREFIX):]
            if name not in ids:
                ids[name] = f"s{len(ids) + 1}"
                order.append(name)

    new_lines: list[Line] = []
    stats: dict[str, list[float]] = {n: [0.0, 0] for n in order}
    for ln in lines:
        sid = ln.speaker_id
        if sid and sid.startswith(NAME_PREFIX):
            name = sid[len(NAME_PREFIX):]
            stats[name][0] += max(ln.end - ln.start, 0.0)
            stats[name][1] += 1
            new_lines.append(ln.model_copy(update={"speaker_id": ids[name]}))
        else:
            new_lines.append(ln)

    speakers = [
        Speaker(id=ids[n], label=n, name=n, color=PALETTE[i % len(PALETTE)],
                total_speech=round(stats[n][0], 3), line_count=int(stats[n][1]))
        for i, n in enumerate(order)
    ]
    return new_lines, speakers


# ───────────── субтитры на языке перевода → перевод реплик ─────────────
_LANG_NAMES = {
    "ru": {"ru", "rus", "russian", "рус", "русский", "ru-ru"},
    "en": {"en", "eng", "english", "en-us", "en-gb"},
    "uk": {"uk", "ukr", "ukrainian"},
    "de": {"de", "deu", "ger", "german"},
    "fr": {"fr", "fra", "fre", "french"},
    "es": {"es", "spa", "spanish"},
    "it": {"it", "ita", "italian"},
    "ja": {"ja", "jpn", "japanese"},
}
SUBTITLE_SUFFIXES = (".srt", ".ass", ".ssa")


def _cyrillic_ratio(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    return sum(1 for c in letters if "\u0400" <= c <= "\u04ff") / len(letters)


def external_subtitle_files(video: str | Path, lang: str) -> list[Path]:
    """Субтитры рядом с видео на языке lang: «фильм.ru.srt», «фильм.rus.ass», «фильм.srt» (если текст на этом языке).

    Файл рядом должен начинаться с имени видео (без расширения). Язык берём из «слов» в имени (ru, rus, russian),
    а у файла без пометки — по тексту (для русского: большинство букв кириллица)."""
    v = Path(video)
    if not v.name or not v.parent.is_dir():
        return []
    want = _LANG_NAMES.get(lang, {lang})
    known = {w for names_ in _LANG_NAMES.values() for w in names_}
    out: list[Path] = []
    try:
        names = sorted(v.parent.iterdir())
    except OSError:
        return []
    for f in names:
        if f.suffix.lower() not in SUBTITLE_SUFFIXES or not f.is_file() or not f.name.startswith(v.stem):
            continue
        rest = f.name[len(v.stem):-len(f.suffix)]
        words = {w for w in re.split(r"[^0-9a-zA-Zа-яА-ЯёЁ-]+", rest.lower()) if w}
        if words & want:
            out.append(f)
        elif not words & known:                       # пометки языка нет — смотрим на текст
            try:
                sample = clean_text(_read_text(f)[:4000])
            except EngineError:
                continue
            if lang == "ru" and _cyrillic_ratio(sample) > 0.6:
                out.append(f)
    return out


def match_translation(lines: list[Line], tr: list[Line]) -> list[Line]:
    """Строки дорожки перевода `tr` → реплики `lines` по пересечению по времени.

    Каждая строка перевода достаётся той реплике, с которой пересекается дольше всего; реплика получает все свои
    строки подряд (по времени) через пробел в `translation`, `translation_source="file"`. Реплики без пары не меняются."""
    if not lines or not tr:
        return list(lines)
    owned: dict[int, list[Line]] = {}
    for t in tr:
        best, best_ov = None, 0.0
        for i, ln in enumerate(lines):
            ov = min(ln.end, t.end) - max(ln.start, t.start)
            if ov > best_ov:
                best, best_ov = i, ov
        if best is not None and best_ov >= 0.05:
            owned.setdefault(best, []).append(t)
    out: list[Line] = []
    for i, ln in enumerate(lines):
        mine = sorted(owned.get(i, []), key=lambda x: x.start)
        text = " ".join(x.text.strip() for x in mine if x.text.strip())
        if text:
            out.append(ln.model_copy(update={"translation": text, "translation_source": "file"}))
        else:
            out.append(ln)
    return out
