"""Распознавание речи целиком: язык по речи → выбор нейросети (Whisper / GigaAM) → добор пропусков.

Язык. Метке дорожки верить нельзя: русский дубляж «Мегамозга» лежит в файле с меткой `eng`, и Whisper с language=en
вместо расшифровки сочинял английский перевод. Поэтому язык (если человек его не выбрал) слушаем: Whisper
`detect_language` на 2–3 окнах по 30 с, где больше всего громких мест (`transcribe.speech_windows`), и складываем
вероятности (`transcribe.choose_language`). Метка дорожки — только подсказка, когда речь неразборчива.

Нейросеть (`pick_engine`): настройка asr_engine = auto | whisper | gigaam. auto — русский распознаёт GigaAM (точнее
границы реплик, без выдумок), остальные языки — Whisper. GigaAM нет/не скачался/упал — Whisper и пояснение в шаге.

Добор (`fill_with_whisper`): GigaAM теряет короткие выкрики и путает их («Тайм-аут!» → «Дай боу!»). Реплики GigaAM, в которых
он сам не уверен (средняя уверенность слов < FILL_DOUBT_P), переслушиваем Whisper'ом (окно с контекстом) и берём его слова,
только если Whisper уверен заметно больше. Места, где громко, а у GigaAM пусто, тоже можно переслушать (FILL_GAPS) —
по замерам это больше вредит, чем помогает (песня, крики, «Субтитры сделал…»), поэтому выключено. Цифры — docs/QUALITY.md.
"""
from __future__ import annotations

import itertools
import logging
import re
from collections.abc import Callable
from pathlib import Path

import numpy as np

from ..models import Line, Word
from ..util import MlMissingError
from . import gigaam
from .transcribe import (
    _STOCK_STRONG,
    ENV_HOP,
    Envelope,
    Seg,
    _norm_text,
    _read_chunk,
    _text_cr,
    caption_score,
    choose_language,
    detect_language_probs,
    drop_hallucinations,
    segs_to_lines,
    whisper_segments,
    whisper_window,
)

log = logging.getLogger(__name__)

ENGINES = ("auto", "whisper", "gigaam")

# добор Whisper'ом (подобрано на «Мегамозге», docs/QUALITY.md)
FILL_DOUBT = True           # переслушивать реплики GigaAM, в которых он не уверен
FILL_DOUBT_P = 0.8          # …средняя уверенность слов ниже этого
FILL_MARGIN = 0.05          # Whisper должен быть увереннее GigaAM хотя бы на столько
FILL_MIN_P = 0.6            # …и не меньше этого
FILL_GAPS = False           # переслушивать громкие места без слов GigaAM (по замерам — вред, см. выше)
GAP_MIN_S = 0.4             # громкое место без слов короче этого — не речь
GAP_MAX_S = 6.0             # …длиннее — музыка/шум, а не пропущенный выкрик
GAP_DB = 20.0               # «громко» — не тише уровня громких мест файла на столько дБ
FILL_CTX_S = 8.0            # контекст вокруг переслушиваемого места (Whisper с контекстом слышит лучше)
FILL_WIN_MAX_S = 30.0       # окно Whisper

LANG_NAMES = {"ru": "русский", "en": "английский", "ja": "японский", "de": "немецкий", "fr": "французский",
              "es": "испанский", "it": "итальянский", "ko": "корейский", "zh": "китайский", "pt": "португальский",
              "uk": "украинский", "pl": "польский", "tr": "турецкий", "nl": "нидерландский", "sv": "шведский",
              "cs": "чешский", "hi": "хинди", "ar": "арабский", "be": "белорусский", "kk": "казахский"}


def lang_name(code: str | None) -> str:
    return LANG_NAMES.get((code or "").lower(), code or "неизвестный")


def language_note(spoken: str | None, track: str | None, source: str | None) -> str | None:
    """Фраза для разбора, когда метка дорожки и язык речи разошлись (язык взят по речи)."""
    if source != "speech" or not spoken or not track or spoken == track:
        return None
    return (f"В файле написано «{lang_name(track)}», а слышно {lang_name(spoken)} — взял {lang_name(spoken)}. "
            "Не так? Выбери язык и разбери заново.")


def pick_engine(setting: str, language: str | None, *, gigaam_ok: bool) -> tuple[str, str | None]:
    """(нейросеть, пояснение для человека или None). GigaAM — только для русского и только если установлен."""
    setting = setting if setting in ENGINES else "auto"
    if setting == "whisper":
        return "whisper", None
    if language != "ru":
        if setting == "gigaam":
            return "whisper", "GigaAM знает только русский — взял Whisper."
        return "whisper", None
    if not gigaam_ok:
        if setting == "gigaam":
            return "whisper", "GigaAM не установлен — распознал Whisper. Обнови нейросети в настройках."
        return "whisper", None
    return "gigaam", None


# ───────────────────────── GigaAM ─────────────────────────
def words_to_lines(words: list[Word], *, start_id: int = 1) -> list[Line]:
    out = []
    for grp in gigaam.group_words(words):
        text = " ".join(w.text.strip() for w in grp).strip()
        if text:
            out.append(Line(id=f"l{len(out) + start_id:04d}", start=grp[0].start, end=max(grp[-1].end, grp[0].start),
                            text=text, words=grp, source="whisper"))
    return out


def drop_gigaam_noise(lines: list[Line], env: Envelope | None) -> list[Line]:
    """GigaAM почти не выдумывает, но пометки и «речь» в тишине убираем так же, как у Whisper."""
    keep = []
    for ln in lines:
        if caption_score(ln.text) >= 3:
            continue
        if env is not None and len(env.db):
            d = env.mean_db(ln.start, max(ln.end, ln.start + ENV_HOP))
            if d < -55 or d < env.ref - 40:
                log.info("GigaAM: реплика в тишине %.1f «%s» убрана", ln.start, ln.text)
                continue
        keep.append(ln)
    return keep


def gigaam_lines(path: Path, *, env: Envelope | None, on_progress=None, on_note=None) -> list[Line]:
    import soundfile as sf
    duration = sf.info(str(path)).duration
    if not gigaam.downloaded() and on_note:
        on_note(f"Скачиваю GigaAM — один раз, ~{gigaam.DOWNLOAD_MB} МБ.")
    model = gigaam.load_model()
    if env is not None and len(env.db):
        chunks = gigaam.energy_chunks(env.db, duration, hop=ENV_HOP)
    else:
        n = max(1, int(np.ceil(duration / gigaam.CHUNK_MAX_S)))
        edges = np.linspace(0.0, duration, n + 1)
        chunks = [(float(a), float(b)) for a, b in itertools.pairwise(edges)]
    words = gigaam.recognize(lambda a, b: _read_chunk(path, a, b), chunks, on_progress=on_progress, model=model)
    return drop_gigaam_noise(words_to_lines(words), env)


# ───────────────────────── добор Whisper'ом ─────────────────────────
def mean_prob(words: list[Word]) -> float | None:
    ps = [w.prob for w in words if w.prob is not None and gigaam.has_letters(w.text)]
    return float(np.mean(ps)) if ps else None


def not_speech(text: str) -> bool:
    """Текст Whisper'а для добора не похож на русскую реплику: не кириллица (песня по-английски), пометка из субтитров
    («[музыка]», «АПЛОДИСМЕНТЫ»), фраза из титров YouTube («Субтитры сделал…», «Продолжение следует…»)."""
    letters = [c for c in text if c.isalpha()]
    caps = len(letters) >= 6 and all(c.isupper() for c in letters)
    norm = _norm_text(text)
    return (cyrillic_share(text) <= 0.5 or caption_score(text) >= 3 or caps or any(p in norm for p in _STOCK_STRONG))


def cyrillic_share(t: str) -> float:
    letters = [c for c in t if c.isalpha()]
    return sum(("а" <= c.lower() <= "я") or c.lower() == "ё" for c in letters) / max(1, len(letters))


def find_gaps(env: Envelope, lines: list[Line], duration: float) -> list[tuple[float, float]]:
    """Громкие места ≥ GAP_MIN_S (и ≤ GAP_MAX_S), где у GigaAM нет ни одного слова (±0,3 с)."""
    if not len(env.db):
        return []
    loud = env.db > env.ref - GAP_DB
    covered = np.zeros_like(loud)
    for ln in lines:
        for w in ln.words or [Word(start=ln.start, end=ln.end, text=ln.text)]:
            covered[max(0, int((w.start - 0.3) / ENV_HOP)):int((w.end + 0.3) / ENV_HOP) + 1] = True
    cand = loud & ~covered
    out: list[tuple[float, float]] = []
    i, n = 0, len(cand)
    while i < n:
        if not cand[i]:
            i += 1
            continue
        j = i
        while j < n and (cand[j] or cand[j:j + 6].any()):        # провалы громкости короче 0,3 с не разрывают место
            j += 1
        a, b = i * ENV_HOP, min(duration, j * ENV_HOP)
        if GAP_MIN_S <= b - a <= GAP_MAX_S:
            out.append((a, b))
        i = j
    return out


def fill_windows(spans: list[tuple[float, float]], duration: float, *, ctx: float = FILL_CTX_S,
                 max_len: float = FILL_WIN_MAX_S) -> list[tuple[float, float]]:
    """Места для переслушивания → окна Whisper с контекстом ≤ max_len (соседние места — в одном окне)."""
    wins: list[list[float]] = []
    for a, b in sorted(spans):
        lo, hi = max(0.0, a - ctx), min(duration, b + ctx)
        if wins and hi - wins[-1][0] <= max_len and lo <= wins[-1][1]:
            wins[-1][1] = max(wins[-1][1], hi)
            continue
        if hi - lo > max_len:                                      # место само длиннее окна — режем контекст
            mid = (a + b) / 2
            lo, hi = max(0.0, mid - max_len / 2), min(duration, mid + max_len / 2)
        wins.append([lo, hi])
    return [(a, b) for a, b in wins]


def _clamp_words(words: list[Word], a: float, b: float) -> list[Word]:
    out = []
    for w in words:
        s = min(max(w.start, a), b)
        e = min(max(w.end, s), b)
        out.append(Word(start=round(s, 3), end=round(e, 3), text=w.text, prob=w.prob))
    return out


WHISPER_EARLY_S = 0.25      # Whisper ставит слова примерно на столько раньше голоса (замер на «Мегамозге»: −250…−350 мс)


def _distance(t: float, ln: Line) -> float:
    return 0.0 if ln.start <= t <= ln.end else min(abs(t - ln.start), abs(t - ln.end))


def assign_words(words: list[Word], lines: list[Line], *, max_dist: float = 0.5) -> list[list[Word]]:
    """Каждое слово Whisper'а — ближайшей реплике GigaAM (по середине слова, сдвинутой на WHISPER_EARLY_S позже).
    Дальше max_dist от всех реплик — ничьё. Так слова соседних реплик не попадают в чужую."""
    out: list[list[Word]] = [[] for _ in lines]
    for w in words:
        t = (w.start + w.end) / 2 + WHISPER_EARLY_S
        best, dist = None, max_dist
        for i, ln in enumerate(lines):
            d = _distance(t, ln)
            if d <= dist:
                best, dist = i, d
        if best is not None:
            out[best].append(w)
    return out


def merge_fill(lines: list[Line], heard: list[tuple[tuple[float, float], list[Seg]]], *,
               gaps: list[tuple[float, float]] | None = None, doubt_p: float = FILL_DOUBT_P, margin: float = FILL_MARGIN,
               min_p: float = FILL_MIN_P) -> tuple[list[Line], int, int]:
    """Слить реплики GigaAM с тем, что услышал Whisper на окнах добора: heard = [(окно, сегменты без выдумок)].
    → (реплики, сколько заменено, сколько добавлено). Без нейросетей — для тестов.

    Слова каждого окна раздаём ближайшим репликам (`assign_words`). Для сомнительной реплики GigaAM берём окно, где
    Whisper слышит её увереннее всего (реплика должна быть внутри окна, не у самого края), и заменяем текст, если
    Whisper уверен заметно больше и услышал что-то новое. Время слов зажимается в границы реплики GigaAM — их мы и
    хотели сохранить."""
    per_window: list[tuple[tuple[float, float], list[list[Word]]]] = []
    for win, segs in heard:
        ws = [w for s in segs for w in s.words if gigaam.has_letters(w.text)]
        per_window.append((win, assign_words(ws, lines)))
    out: list[Line] = []
    replaced = 0
    for i, ln in enumerate(lines):
        gp = mean_prob(ln.words)
        if gp is None or gp >= doubt_p:
            out.append(ln)
            continue
        best: tuple[float, list[Word]] | None = None
        for (lo, hi), assigned in per_window:
            if lo > ln.start - 1.0 or hi < ln.end + 0.5:          # реплика у края окна — Whisper слышит её плохо
                continue
            wp = mean_prob(assigned[i])
            if assigned[i] and wp is not None and (best is None or wp > best[0]):
                best = (wp, assigned[i])
        if best is None:
            out.append(ln)
            continue
        wp, inside = best
        wtext = "".join(w.text for w in inside).strip()
        n_giga = len([w for w in ln.words if gigaam.has_letters(w.text)])
        heard_w = _norm_text(wtext).split()
        prev_w = _norm_text(lines[i - 1].text).split() if i else []
        next_w = _norm_text(lines[i + 1].text).split() if i + 1 < len(lines) else []
        if (wp < max(min_p, gp + margin) or not_speech(wtext) or len(inside) > 2 * n_giga + 2
                or set(heard_w) <= set(_norm_text(ln.text).split())      # услышал то же или меньше — менять нечего
                or next_w[:len(heard_w)] == heard_w                     # это начало следующей реплики (Whisper раньше)
                or prev_w[-len(heard_w):] == heard_w):                  # …или хвост предыдущей
            out.append(ln)
            continue
        log.info("Добор: «%s» (%.2f) → «%s» (%.2f)", ln.text, gp, wtext, wp)
        # Whisper услышал начало, которого у GigaAM нет («гард!» → «Ангард!»): начало реплики — по его первому слову
        # (с поправкой на его раннее начало), но не раньше конца предыдущей реплики
        start = ln.start
        first = inside[0].start + WHISPER_EARLY_S
        if first < ln.start - 0.15 and _norm_text(inside[0].text) not in _norm_text(ln.text).split():
            start = max(first, (out[-1].end + 0.05) if out else 0.0, ln.start - 1.0)
        out.append(ln.model_copy(update={"text": wtext, "start": round(start, 3),
                                         "words": _clamp_words(inside, start, ln.end)}))
        replaced += 1
    mid = lambda w: (w.start + w.end) / 2
    windows = [w for w, _ in heard]
    added = 0
    for a, b in gaps or []:
        k = max(range(len(windows)), key=lambda j: min(a - windows[j][0], windows[j][1] - b), default=None)
        if k is None:
            continue
        for s in heard[k][1]:
            ws = [w for w in s.words if a - 0.35 <= mid(w) <= b + 0.15 and gigaam.has_letters(w.text)]
            if not ws or any(ln.start - 0.1 <= mid(w) <= ln.end + 0.1 for w in ws for ln in out):
                continue
            text = "".join(w.text for w in ws).strip()
            wp = mean_prob(ws)
            if (wp is None or wp < min_p or (s.avg_logprob is not None and s.avg_logprob < -0.7)
                    or not_speech(text) or _text_cr(text) > 2.4
                    or len(re.sub(r"\W", "", text)) < 2):
                continue
            log.info("Добор: в пропуске %.1f–%.1f «%s» (%.2f)", a, b, text, wp)
            out.append(Line(id="", start=ws[0].start, end=max(ws[-1].end, ws[0].start), text=text, words=ws,
                            source="whisper"))
            added += 1
    out.sort(key=lambda x: (x.start, x.end))
    out = [ln.model_copy(update={"id": f"l{i + 1:04d}"}) for i, ln in enumerate(out)]
    return out, replaced, added


def fill_with_whisper(path: Path, lines: list[Line], *, env: Envelope | None, language: str, model: str, device: str,
                      backend: str | None, on_progress=None) -> tuple[list[Line], int, int]:
    """Переслушать Whisper'ом сомнительные реплики GigaAM (и пропуски, если FILL_GAPS)."""
    import soundfile as sf
    duration = sf.info(str(path)).duration
    spans = []
    if FILL_DOUBT:
        spans += [(ln.start, ln.end) for ln in lines if (mean_prob(ln.words) or 1.0) < FILL_DOUBT_P]
    gaps = find_gaps(env, lines, duration) if (FILL_GAPS and env is not None) else []
    spans += gaps
    if not spans:
        return lines, 0, 0
    wins = fill_windows(spans, duration)
    heard: list[tuple[tuple[float, float], list[Seg]]] = []
    for k, (a, b) in enumerate(wins):
        part = whisper_window(_read_chunk(path, a, b), offset=a, language=language, model=model, device=device,
                              backend=backend)
        # повторы не считаем зацикливанием: в коротком окне «Тайм-аут! Тайм-аут! Тайм-аут!» — это крик, а не сбой
        heard.append(((a, b), drop_hallucinations(part, env, loops=False)))
        if on_progress:
            on_progress((k + 1) / len(wins))
    return merge_fill(lines, heard, gaps=gaps)


# ───────────────────────── всё вместе ─────────────────────────
def recognize(path: Path, *, env: Envelope | None, language: str | None, track_language: str | None, engine: str,
              model: str, device: str, backend: str | None, vad: bool | None,
              on_progress: Callable[[float], None] | None, on_note: Callable[[str], None] | None,
              info: dict) -> tuple[list[Line], str]:
    prog = on_progress or (lambda _v: None)
    notes: list[str] = []

    def note(t: str, *, keep: bool = True) -> None:
        """Пояснение в шаге задачи; keep — запомнить и в разборе (временные, вроде «Скачиваю…», не храним)."""
        if keep:
            notes.append(t)
        if on_note:
            on_note(t)

    # 1. язык
    lang, source = language, "user"
    if lang is None:
        try:
            probs = detect_language_probs(path, env=env, model=model, device=device, backend=backend)
            lang, source = choose_language(probs, track_language)
            info["language_probs"] = [dict(sorted(p.items(), key=lambda kv: -kv[1])[:3]) for p in probs]
        except MlMissingError:
            raise
        except Exception as e:  # noqa: BLE001 — не вышло послушать: метка дорожки или Whisper сам по первому окну
            log.warning("Язык по речи не определился: %s", e)
            lang, source = track_language, "track"
    info["language_source"] = source if lang else None
    prog(0.03)

    # 2. нейросеть
    eng, why = pick_engine(engine, lang, gigaam_ok=gigaam.installed())
    if why:
        note(why)
    lines: list[Line] | None = None
    if eng == "gigaam":
        try:
            lines = gigaam_lines(path, env=env, on_progress=lambda v: prog(0.03 + 0.72 * v),
                                 on_note=lambda t: note(t, keep=False))
            info["asr_engine"] = "gigaam"
        except Exception as e:  # noqa: BLE001 — не скачался/упал: Whisper
            log.warning("GigaAM не сработал: %s", e)
            note("GigaAM не сработал — распознал Whisper.")
            lines = None
        if lines is not None:
            try:
                lines, replaced, added = fill_with_whisper(
                    path, lines, env=env, language="ru", model=model, device=device, backend=backend,
                    on_progress=lambda v: prog(0.75 + 0.24 * v))
                info["filled"] = replaced + added
            except MlMissingError:
                info["filled"] = 0                                # Whisper нет — добор не обязателен
            except Exception as e:  # noqa: BLE001
                log.warning("Добор Whisper'ом не удался: %s", e)
                info["filled"] = 0
    if lines is None:
        segs, got = whisper_segments(path, language=lang, model=model, device=device, backend=backend, vad=vad,
                                     on_progress=lambda v: prog(0.03 + 0.96 * v), env=env)
        lines = segs_to_lines(segs)
        lang = lang or got
        info["asr_engine"] = "whisper"
        if language is None and info.get("language_source") is None and lang:
            info["language_source"] = "speech"
    info["notes"] = notes
    return lines, (lang or "")
