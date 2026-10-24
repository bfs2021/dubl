"""Разбор дубля по тексту: что человек сказал на самом деле и сказал ли он реплику целиком.

Whisper слушает дубль на языке перевода → слова сравниваются со SceneLine.text:
* text_match — какая доля слов реплики прозвучала (по порядку, с поправкой на окончания и мелкие ошибки распознавания);
* complete — реплика сказана целиком (text_match ≥ COMPLETE_MIN);
* хвост-перезапуск — после целой реплики человек говорит «давай ещё раз», «стоп», повторяет начало фразы: хвост можно
  отрезать (tail_cut_s — секунда файла, с которой он начинается).

Чистые функции (сравнение) не тянут нейросети; whisper подключается только в recognize_words, лениво.
Нейросети во время записи не скачиваем: нет модели на диске — разбор пропускается (все поля None).
"""
from __future__ import annotations

import logging
import re
import threading
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path

from ..models import Word

log = logging.getLogger(__name__)

COMPLETE_MIN = 0.85          # доля слов реплики, начиная с которой считаем её сказанной целиком
FUZZY_WORD = 0.78            # похожесть слов (SequenceMatcher), с которой «пошёл» и «пошла» считаются одним словом
RESTART_MIN_MATCH = 0.5      # хвост, где снова звучит больше половины реплики, — это перезапуск
MAX_TAIL_WORDS = 14          # в хвосте длиннее этого — уже не оговорка, а другой текст: ничего не предлагаем

# слова-«перезапуски»: человек сбился и просит себя же начать заново (нормализованные: е вместо ё, нижний регистр)
_RESTART_WORDS = {"заново", "сначала", "стоп", "снова", "ошибся", "ошиблась", "упс", "ой", "блин", "черт", "тьфу",
                  "подожди", "подождите", "секунду", "секундочку", "пардон", "опять"}
_RESTART_PHRASES = ("еще раз", "давай еще", "давайте еще", "нет нет", "не то", "не так", "попробуем еще", "как там",
                    "по новой", "с начала", "так стоп", "нет подожди")

_WORD = re.compile(r"[\w'’-]+", re.UNICODE)
_ASR_LOCK = threading.Lock()      # mlx-whisper держит модель в глобальной переменной: два разбора одновременно нельзя
KEEP_MODEL_S = 90.0               # после последнего разбора модель держим в памяти ещё столько: дубли идут пачкой
_idle_timer: threading.Timer | None = None


# ───────────────────────── слова ─────────────────────────
def norm_word(w: str) -> str:
    found = norm_words(w)
    return found[0] if found else ""


def norm_words(text: str) -> list[str]:
    return [x for x in (w.casefold().replace("ё", "е").strip("'’-_") for w in _WORD.findall(text or "")) if x]


@lru_cache(maxsize=20000)
def same_word(a: str, b: str) -> bool:
    if a == b:
        return True
    if a.isdigit() or b.isdigit():
        return False
    return min(len(a), len(b)) >= 3 and SequenceMatcher(None, a, b).ratio() >= FUZZY_WORD


# ───────────────────────── на каком языке текст ─────────────────────────
_CYR_LANGS = {"ru", "uk", "be", "bg", "sr", "mk", "kk", "ky", "mn", "tg"}


def _letters(text: str) -> tuple[int, int, int]:
    """(кириллица, латиница, прочие буквы)."""
    cyr = lat = oth = 0
    for ch in text:
        if not ch.isalpha():
            continue
        o = ord(ch)
        if 0x0400 <= o <= 0x052F:
            cyr += 1
        elif ch.isascii() or 0x00C0 <= o <= 0x024F:
            lat += 1
        else:
            oth += 1
    return cyr, lat, oth


def comparable(text: str, original: str, target_lang: str, source_lang: str | None = None) -> bool:
    """Можно ли сравнивать то, что услышит whisper на языке перевода, с этим текстом реплики.
    Нет — если текст реплики на другом языке (перевода нет, остался оригинал) или в нём почти нет слов."""
    t = (text or "").strip()
    if len(norm_words(t)) < 1:
        return False
    lang = (target_lang or "ru").split("-")[0].lower()
    cyr, lat, oth = _letters(t)
    total = cyr + lat + oth
    if total == 0:
        return False
    if lang in _CYR_LANGS:
        if cyr / total < 0.5:
            return False
    elif lang in ("ja", "zh", "ko", "th", "ar", "he", "hi", "fa"):
        if oth / total < 0.5:
            return False
    elif lat / total < 0.5:
        return False
    # язык оригинала известен и отличается от языка перевода, а текст — это он и есть: перевода нет
    return not (source_lang and source_lang.split("-")[0].lower() != lang and t == (original or "").strip())


# ───────────────────────── сравнение ─────────────────────────
@dataclass
class TextResult:
    recognized: str
    text_match: float
    complete: bool
    tail_cut_s: float | None = None
    tail_text: str | None = None
    notes: list[str] = field(default_factory=list)


def _lcs_table(want: list[str], got: list[str]) -> list[list[int]]:
    """dp[i][j] — длина общей подпоследовательности want[:i] и got[:j] (слова сравниваются «примерно»)."""
    n, m = len(want), len(got)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        wi = want[i - 1]
        row, prev = dp[i], dp[i - 1]
        for j in range(1, m + 1):
            if same_word(wi, got[j - 1]):
                row[j] = prev[j - 1] + 1
            else:
                row[j] = max(prev[j], row[j - 1])
    return dp


def match_ratio(want_text: str, got_text: str) -> float:
    """Доля слов реплики, которые прозвучали по порядку. 1.0 — все."""
    want, got = norm_words(want_text), norm_words(got_text)
    if not want:
        return 1.0
    if not got:
        return 0.0
    return _lcs_table(want, got)[len(want)][len(got)] / len(want)


def _looks_like_restart(tail: list[str], want: list[str]) -> bool:
    """Хвост — перезапуск: «давай ещё раз», «стоп», либо снова звучит начало (или большая часть) реплики."""
    joined = " ".join(tail)
    if any(p in joined for p in _RESTART_PHRASES):
        return True
    if any(w in _RESTART_WORDS for w in tail):
        return True
    head = want[:3]
    if len(head) >= 2 and len(tail) >= 2:
        hits = sum(1 for a, b in zip(head, tail, strict=False) if same_word(a, b))
        if hits >= min(2, len(head)):
            return True
    if len(tail) >= 3 and want:
        dp = _lcs_table(want, tail)
        if dp[len(want)][len(tail)] / len(want) >= RESTART_MIN_MATCH:
            return True
    return False


def analyze_text(want_text: str, words: list[Word]) -> TextResult:
    """Сравнить распознанные слова дубля с текстом реплики. words — со временем (секунды файла).
    Хвост-перезапуск ищем после того места, где реплика уже прозвучала целиком (или как можно полнее)."""
    want = norm_words(want_text)
    toks = [(norm_word(w.text), w) for w in words]
    toks = [(t, w) for t, w in toks if t]
    got = [t for t, _ in toks]
    recognized = " ".join(w.text.strip() for _, w in toks).strip()
    if not want:
        return TextResult(recognized, 1.0, True)
    if not got:
        return TextResult(recognized, 0.0, False)
    dp = _lcs_table(want, got)
    n, m = len(want), len(got)
    best = dp[n][m]
    ratio = best / n
    res = TextResult(recognized, round(ratio, 3), ratio >= COMPLETE_MIN)
    # где реплика закончилась: первое j, на котором достигнут максимум совпавших слов
    j_end = next(j for j in range(m + 1) if dp[n][j] == best)
    if best == 0 or j_end >= m:
        return res
    tail_toks = toks[j_end:]
    tail = got[j_end:]
    if not tail or len(tail) > MAX_TAIL_WORDS:
        return res
    if not _looks_like_restart(tail, want):
        return res
    prev_end = toks[j_end - 1][1].end
    start = tail_toks[0][1].start
    gap = max(0.0, start - prev_end)
    cut = prev_end + min(max(gap * 0.5, 0.04), 0.3)
    cut = min(cut, max(prev_end, start - 0.02))
    res.tail_cut_s = round(max(cut, 0.1), 3)
    res.tail_text = " ".join(w.text.strip() for _, w in tail_toks).strip()
    return res


# ───────────────────────── whisper ─────────────────────────
def model_on_disk(model: str, device: str = "auto") -> bool:
    """Модель уже скачана? Во время записи ничего не качаем: большой файл посреди работы — хуже, чем нет разбора."""
    from .transcribe import MLX_REPOS, mlx_dir, pick_backend, whisper_dir
    try:
        if pick_backend(model, device) == "mlx":
            d = mlx_dir() / MLX_REPOS[model].split("/")[1]
            return (d / "config.json").exists() and any(d.glob("weights.*"))
    except Exception as e:  # noqa: BLE001
        log.debug("mlx-модель не проверилась: %s", e)
        return False
    try:
        from faster_whisper.utils import download_model
        download_model(model, local_files_only=True, cache_dir=str(whisper_dir()))
        return True
    except Exception:  # noqa: BLE001 — нет пакета или модели на диске
        return False


def pick_model(whisper_model: str, device: str = "auto") -> str | None:
    """Какую модель взять для дубля: уже скачанную и самую лёгкую (small → base → та, что выбрана для разбора видео).
    None — подходящей на диске нет: дубль сохраняем без разбора по тексту."""
    for m in dict.fromkeys(("small", "base", whisper_model)):
        if m and model_on_disk(m, device):
            return m
    return None


def _release_later(delay: float) -> None:
    """Отдать память модели через delay секунд тишины (каждый новый разбор откладывает срок)."""
    global _idle_timer

    def go() -> None:
        from .transcribe import release_memory
        with _ASR_LOCK:
            release_memory()
    with _ASR_LOCK:
        if _idle_timer is not None:
            _idle_timer.cancel()
        _idle_timer = threading.Timer(delay, go)
        _idle_timer.daemon = True
        _idle_timer.start()


def recognize_words(wav: str | Path, *, language: str, model: str, device: str = "auto") -> list[Word] | None:
    """Слова дубля со временем. None — разбор недоступен (нет нейросетей или модели на диске, сбой): это не ошибка
    записи, дубль сохраняется без разбора."""
    try:
        from .transcribe import transcribe
        with _ASR_LOCK:
            lines, _lang = transcribe(wav, language=language, model=model, device=device, vad=False, keep_loaded=True)
    except Exception as e:  # noqa: BLE001 — разбор необязателен
        log.warning("Дубль не распознался: %s", e)
        _release_later(0.0)
        return None
    _release_later(KEEP_MODEL_S)
    return [w for ln in lines for w in (ln.words or [Word(start=ln.start, end=ln.end, text=ln.text)])]
