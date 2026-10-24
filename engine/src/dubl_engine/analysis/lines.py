"""Склейка реплик: общая для ручного «Склеить» и для авто-склейки после разбора."""
from __future__ import annotations

from ..models import Line

# Авто-склейка: whisper иногда режет одну фразу на две («О, ты злодей! Это правда!» + «Но совсем не супер!»).
# Склеиваем, только если это точно один человек на одном дыхании: тот же голос, пауза короче JOIN_GAP_S,
# вместе не длиннее JOIN_MAX_S (длинные реплики неудобно записывать), никто не перебивает.
JOIN_GAP_S = 0.35
JOIN_MAX_S = 8.0


def _join_text(a: str | None, b: str | None) -> str | None:
    a, b = (a or "").strip(), (b or "").strip()
    if not a and not b:
        return None
    return f"{a} {b}".strip()


def merge_two(a: Line, b: Line) -> Line:
    """Реплика b дописывается к a (b идёт позже). id, голос и источник — от a."""
    words = sorted([*a.words, *b.words], key=lambda w: w.start)
    tr = _join_text(a.translation, b.translation) if (a.translation or b.translation) else None
    src = a.translation_source or b.translation_source if tr else None
    return a.model_copy(update={
        "start": min(a.start, b.start), "end": max(a.end, b.end),
        "text": _join_text(a.text, b.text) or "", "words": words,
        "overlap": a.overlap or b.overlap, "translation": tr, "translation_source": src,
    })


def join_close_groups(lines: list[Line], *, gap: float = JOIN_GAP_S, max_len: float = JOIN_MAX_S) -> list[list[Line]]:
    """Реплики по времени, сгруппированные для авто-склейки: в группе больше одной — эти реплики склеиваются в одну
    (соседние одного голоса с паузой ≤ gap; только распознанные whisper: строки субтитров задал автор файла)."""
    groups: list[list[Line]] = []
    span: tuple[float, float] | None = None
    for ln in sorted(lines, key=lambda x: (x.start, x.end)):
        prev = groups[-1][-1] if groups else None
        if (prev is not None and span is not None and prev.source == "whisper" and ln.source == "whisper"
                and prev.speaker_id is not None and prev.speaker_id == ln.speaker_id
                and not any(x.overlap for x in groups[-1]) and not ln.overlap
                and 0 <= ln.start - span[1] <= gap and ln.end - span[0] <= max_len):
            groups[-1].append(ln)
            span = (min(span[0], ln.start), max(span[1], ln.end))
        else:
            groups.append([ln])
            span = (ln.start, ln.end)
    return groups


def merge_group(group: list[Line]) -> Line:
    out = group[0]
    for nxt in group[1:]:
        out = merge_two(out, nxt)
    return out


def join_close_lines(lines: list[Line], *, gap: float = JOIN_GAP_S, max_len: float = JOIN_MAX_S) -> list[Line]:
    """Соседние реплики одного голоса с паузой ≤ gap склеиваются (только распознанные whisper: строки субтитров
    задал автор файла — их не трогаем)."""
    return [merge_group(g) for g in join_close_groups(lines, gap=gap, max_len=max_len)]
