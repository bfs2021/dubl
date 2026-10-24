"""Распознавание речи со словами и таймингами.

Бэкенды:
- **mlx-whisper** — Mac на Apple Silicon (GPU): large-v3-turbo идёт ≈ 0,05 × реального времени и не теряет диалог поверх музыки;
- **faster-whisper** — всё остальное (CUDA float16 / CPU int8).

VAD (silero) на фильмах работает плохо: диалог под музыкой для него «не речь» — реплики пропадают (на Tears of Steel
из 76 реплик находилось 41). Поэтому mlx идёт без VAD, а у faster-whisper VAD «мягкий» и включается только там, где он
не вредит (см. `FW_VAD_PARAMS`, `docs/QUALITY.md`). Выдуманные на музыке фразы («Thank you.», «I'm sorry.») отсекает
`drop_hallucinations`: по вероятностям слов, avg_logprob, сжатию текста, скорости речи и энергии звука в этом месте.
"""
from __future__ import annotations

import itertools
import logging
import platform
import re
import sys
import zlib
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..devices import device_candidates, mark_bad, whisper_runtime
from ..models import Line, Word
from ..util import EngineError, MlMissingError, data_dir

log = logging.getLogger(__name__)

MAX_LINE = 8.0      # реплики длиннее режем
SPLIT_GAP = 0.6     # …по паузам между словами не короче этого

SR = 16000
ENV_HOP = 0.05                  # шаг огибающей энергии, с
MLX_CHUNK_S = 300.0             # mlx режем на куски ≈ 5 мин (прогресс и память), по самой тихой точке
MLX_CHUNK_SEARCH_S = 20.0       # …ищем тишину в ±20 с от ровной границы
MLX_REPOS = {
    "tiny": "mlx-community/whisper-tiny-mlx",
    "base": "mlx-community/whisper-base-mlx",
    "small": "mlx-community/whisper-small-mlx",
    "medium": "mlx-community/whisper-medium-mlx",
    "large-v3": "mlx-community/whisper-large-v3-mlx",
    "large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
}

# faster-whisper: VAD «мягкий» — порог почти нулевой (режет только настоящую тишину/цифровой ноль), большие поля,
# куски ≤ 12 с. Подобрано на Tears of Steel (см. docs/QUALITY.md): штатный VAD терял 35 реплик из 76, этот — 11.
FW_VAD_PARAMS = {"threshold": 0.003, "neg_threshold": 0.0015, "min_silence_duration_ms": 300, "speech_pad_ms": 500,
                 "max_speech_duration_s": 12, "min_speech_duration_ms": 100}

# фразы, которые whisper любит «слышать» в музыке и тишине
_STOCK = {"thank you", "thanks", "thanks for watching", "thank you for watching", "i'm sorry", "im sorry", "oh", "you",
          "bye", "the end", "okay", "so", "uh", "ugh", "hmm", "mm", "music", "subtitles by the amara.org community"}
# фразы из YouTube-субтитров, на которых учили whisper: в фильме их не говорят, а на музыке они «слышатся» с высокой уверенностью
_STOCK_STRONG = ("be right back", "next video", "subscribe", "for watching", "see you next", "see you in the next",
                 "amara.org", "like and share", "thanks for listening", "в следующем видео", "подписывайтесь", "продолжение следует",
                 "субтитры сделал", "субтитры создавал", "dimatorzok", "редактор субтитров")


def whisper_dir() -> Path:
    d = data_dir() / "models" / "whisper"
    d.mkdir(parents=True, exist_ok=True)
    return d


def mlx_dir() -> Path:
    d = data_dir() / "models" / "mlx"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ───────────────────────── нарезка слов ─────────────────────────
def split_words(words: list[Word], *, max_len: float = MAX_LINE, gap: float = SPLIT_GAP) -> list[list[Word]]:
    """Режет длинную реплику (> max_len) по паузам между словами > gap.

    Каждый раз режем по самой большой паузе, потом проверяем куски заново. Если пауз нет — оставляем как есть.
    """
    if not words:
        return []
    if words[-1].end - words[0].start <= max_len:
        return [words]
    best_i, best_gap = -1, gap
    for i in range(1, len(words)):
        g = words[i].start - words[i - 1].end
        if g > best_gap:
            best_i, best_gap = i, g
    if best_i < 0:
        return [words]
    return split_words(words[:best_i], max_len=max_len, gap=gap) + split_words(words[best_i:], max_len=max_len, gap=gap)


def _join(words: list[Word]) -> str:
    return "".join(w.text for w in words).strip()


def _segment_to_groups(seg_words: list[Word], seg_text: str, start: float, end: float) -> list[tuple[float, float, str, list[Word]]]:
    """Сегмент whisper → список (start, end, text, words) с учётом нарезки."""
    if not seg_words:
        t = seg_text.strip()
        return [(start, end, t, [])] if t else []
    out = []
    for grp in split_words(seg_words):
        text = _join(grp)
        if text:
            out.append((grp[0].start, grp[-1].end, text, grp))
    return out


# ───────────────────────── сегмент и фильтр галлюцинаций ─────────────────────────
@dataclass
class Seg:
    """Сегмент whisper в нейтральном виде (оба бэкенда приводятся к нему)."""
    start: float
    end: float
    text: str
    words: list[Word] = field(default_factory=list)
    avg_logprob: float | None = None
    no_speech_prob: float | None = None
    compression_ratio: float | None = None


class Envelope:
    """Огибающая громкости звука (дБ FS на кадр ENV_HOP) — для отсева «речи» в тишине и подбора точек разреза."""

    def __init__(self, db: np.ndarray) -> None:
        self.db = db
        self.ref = float(np.percentile(db, 90)) if len(db) else -30.0     # уровень «громкого места» файла

    @classmethod
    def from_file(cls, path) -> Envelope:
        import soundfile as sf
        info = sf.info(str(path))
        hop = max(1, int(info.samplerate * ENV_HOP))
        out: list[np.ndarray] = []
        block = hop * 1200                                            # ≈ 60 с за раз
        with sf.SoundFile(str(path)) as f:
            while True:
                x = f.read(block, dtype="float32", always_2d=True)
                if not len(x):
                    break
                x = x.mean(axis=1)
                n = len(x) // hop
                if n:
                    p = (x[:n * hop].reshape(n, hop) ** 2).mean(axis=1)
                    out.append(10 * np.log10(p + 1e-12))
                if len(x) % hop:                                       # хвост блока — отдельным кадром
                    out.append(10 * np.log10(np.array([(x[n * hop:] ** 2).mean()]) + 1e-12))
        return cls(np.concatenate(out) if out else np.zeros(0))

    def mean_db(self, start: float, end: float) -> float:
        a = int(max(0.0, start) / ENV_HOP)
        b = max(a + 1, int(end / ENV_HOP))
        seg = self.db[a:b]
        if not len(seg):
            return -120.0
        return float(10 * np.log10(np.mean(10 ** (seg / 10)) + 1e-12))

    def quietest(self, t: float, radius: float, win: float = 0.5) -> float:
        """Время середины самого тихого окна `win` секунд в [t − radius, t + radius]."""
        w = max(1, int(win / ENV_HOP))
        a, b = int(max(0.0, t - radius) / ENV_HOP), min(len(self.db), int((t + radius) / ENV_HOP))
        if b - a <= w:
            return t
        p = 10 ** (self.db[a:b] / 10)
        sums = np.convolve(p, np.ones(w), mode="valid")
        k = int(np.argmin(sums))
        return (a + k + w / 2) * ENV_HOP


def _words_prob(s: Seg) -> float | None:
    ps = [w.prob for w in s.words if w.prob is not None]
    return float(np.mean(ps)) if ps else None


def _norm_text(t: str) -> str:
    return " ".join(re.sub(r"[^\w\s']", " ", t.lower()).split())


def _text_cr(t: str) -> float:
    b = t.encode("utf-8")
    return len(b) / max(1, len(zlib.compress(b))) if len(b) > 40 else 1.0


# слова-пометки из субтитров, на которых учили whisper: на музыке и шумах он «пишет» их вместо речи
_CAPTION_WORDS = ("музык", "music", "смех", "смеёт", "смеет", "laugh", "аплодисмент", "applause", "шум", "noise",
                  "мелоди", "песн", "поёт", "поет", "song", "sing", "звук", "sound", "тишин", "silence", "крик", "плач",
                  "вздох", "sigh", "титры", "субтитр", "subtitle", "заставк", "intro")


def caption_score(text: str) -> float:
    """Похоже на пометку из субтитров, а не на речь: «[музыка]», «(смеётся)», «ДИНАМИЧНАЯ МУЗЫКА», «♪»."""
    t = text.strip()
    if not t:
        return 0.0
    low = t.lower()
    has_word = any(w in low for w in _CAPTION_WORDS)
    if "♪" in t or "♫" in t or (t[0] in "[(*" and t.rstrip(".!?")[-1:] in "])*"):
        return 4.0
    letters = [c for c in t if c.isalpha()]
    caps = len(letters) >= 6 and all(c.isupper() for c in letters) and len(t.split()) >= 2
    if caps and has_word:
        return 4.0
    return 1.5 if caps else 0.0


def hallucination_score(s: Seg, *, prev_end: float | None, next_start: float | None,
                        env: Envelope | None = None) -> float:
    """Чем больше, тем вероятнее, что сегмент выдуман на музыке/тишине. ≥ 3 — отбрасываем."""
    score = 0.0
    wp = _words_prob(s)
    if wp is not None:
        score += 2.0 if wp < 0.4 else 1.0 if wp < 0.6 else 0.5 if wp < 0.7 else 0.0
    if s.avg_logprob is not None:
        score += 1.5 if s.avg_logprob < -0.9 else 0.5 if s.avg_logprob < -0.6 else 0.0
    if s.no_speech_prob is not None and s.no_speech_prob > 0.6:
        score += 2.0 if (s.avg_logprob is None or s.avg_logprob < -0.5) else 0.5
    cr = _text_cr(s.text)
    if cr > 2.4 or (s.compression_ratio is not None and s.compression_ratio > 2.4 and len(s.text) > 40):
        score += 3.0                                    # зацикливание: «AHHHH…», повторы
    norm = _norm_text(s.text)
    score += caption_score(s.text)
    if norm in _STOCK:
        score += 1.0
    if any(p in norm for p in _STOCK_STRONG):
        score += 2.0
    n_words = max(1, len(norm.split()))
    if (s.end - s.start) / n_words > 1.0 and n_words <= 4:
        score += 1.0                                    # слова растянуты на секунды — так бывает на пустом месте
    near = 4.0
    isolated = ((prev_end is None or s.start - prev_end > near) and (next_start is None or next_start - s.end > near))
    if isolated:
        score += 1.0
    if env is not None and len(env.db):
        d = env.mean_db(s.start, s.end)
        if d < -55 or d < env.ref - 40:
            score += 4.0                                # тишина: речи тут быть не может
        elif d < env.ref - 25:
            score += 1.5
    return score


def _looped(segs: list[Seg], *, span: float = 90.0, times: int = 3) -> set[int]:
    """Индексы сегментов, чей текст (≥ 2 слов) повторился ≥ `times` раз за `span` секунд: так выглядит зацикливание whisper."""
    by_text: dict[str, list[int]] = {}
    for i, s in enumerate(segs):
        n = _norm_text(s.text)
        if len(n.split()) >= 2:
            by_text.setdefault(n, []).append(i)
    out: set[int] = set()
    for idxs in by_text.values():
        for k in range(len(idxs) - times + 1):
            if segs[idxs[k + times - 1]].start - segs[idxs[k]].start <= span:
                out.update(idxs[k:k + times])
    return out


def drop_hallucinations(segs: list[Seg], env: Envelope | None = None, *, threshold: float = 3.0,
                        loops: bool = True) -> list[Seg]:
    """Убирает выдуманные фразы. Решение принимается по сумме признаков, а не по одному (одно слово с низкой уверенностью
    в середине диалога — нормально; то же слово посреди пустоты — галлюцинация). loops=False — повтор фразы не признак
    (короткое окно добора, где «Тайм-аут! Тайм-аут! Тайм-аут!» — настоящий крик)."""
    keep: list[Seg] = []
    looped = _looped(segs) if loops else set()
    for i, s in enumerate(segs):
        prev_end = segs[i - 1].end if i else None
        next_start = segs[i + 1].start if i + 1 < len(segs) else None
        sc = hallucination_score(s, prev_end=prev_end, next_start=next_start, env=env) + (3.0 if i in looped else 0.0)
        if sc >= threshold:
            log.info("Отброшена выдуманная реплика %.1f–%.1f «%s» (счёт %.1f)", s.start, s.end, s.text.strip(), sc)
            continue
        keep.append(s)
    return keep


# ───────────────────────── бэкенды ─────────────────────────
def release_memory() -> None:
    """Отдаёт память после распознавания: кэш Metal (mlx), кэш MPS (torch) и мусор Python. Наши модели нигде не кэшируются, а mlx-whisper
    держит загруженную модель в глобальном `ModelHolder` до конца процесса — его обнуляем сами."""
    import gc
    _FW_CACHE.clear()
    if "mlx_whisper" in sys.modules:
        try:
            from mlx_whisper.transcribe import ModelHolder
            ModelHolder.model = None
            ModelHolder.model_path = None
        except Exception:  # noqa: BLE001, S110 — другая версия mlx-whisper
            pass
    gc.collect()
    try:
        import mlx.core as mx
        getattr(mx, "clear_cache", None) and mx.clear_cache()
    except Exception:  # noqa: BLE001, S110 — mlx нет
        pass
    try:
        import sys as _sys
        if "torch" in _sys.modules:
            import torch
            if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                torch.mps.empty_cache()
    except Exception:  # noqa: BLE001, S110
        pass


def mlx_available() -> bool:
    if sys.platform != "darwin" or platform.machine().lower() not in ("arm64", "aarch64"):
        return False
    try:
        import mlx_whisper  # noqa: F401
    except Exception:  # noqa: BLE001 — не установлен или сломан
        return False
    return True


def pick_backend(model: str, device: str = "auto") -> str:
    """'mlx' — Mac arm64, mlx-whisper установлен, модель есть в списке, устройство не cpu; иначе 'faster'."""
    if (device or "auto").lower().split(":")[0] == "cpu":
        return "faster"
    if model in MLX_REPOS and mlx_available():
        return "mlx"
    return "faster"


def _mlx_model_path(model: str) -> str:
    """Папка с моделью mlx в data_dir/models/mlx (скачивает при первом запуске); при сбое — имя репозитория (общий кэш HF)."""
    repo = MLX_REPOS[model]
    target = mlx_dir() / repo.split("/")[1]
    if (target / "config.json").exists() and any(target.glob("weights.*")):
        return str(target)
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(repo, local_dir=str(target))
        return str(target)
    except Exception as e:  # noqa: BLE001
        log.warning("Не удалось скачать %s в папку моделей (%s) — беру из общего кэша", repo, e)
        return repo


def _read_chunk(path: Path, a: float, b: float) -> np.ndarray:
    import soundfile as sf
    with sf.SoundFile(str(path)) as f:
        sr = f.samplerate
        f.seek(min(int(a * sr), f.frames))
        x = f.read(max(0, int((b - a) * sr)), dtype="float32", always_2d=True).mean(axis=1)
    if sr != SR:
        from math import gcd

        from scipy.signal import resample_poly
        g = gcd(sr, SR)
        x = resample_poly(x, SR // g, sr // g).astype(np.float32)
    return np.ascontiguousarray(x, dtype=np.float32)


def _chunk_bounds(duration: float, env: Envelope | None) -> list[tuple[float, float]]:
    if duration <= MLX_CHUNK_S * 1.3:
        return [(0.0, duration)]
    cuts = []
    t = MLX_CHUNK_S
    while t < duration - MLX_CHUNK_S * 0.3:
        cuts.append(env.quietest(t, MLX_CHUNK_SEARCH_S) if env is not None else t)
        t = cuts[-1] + MLX_CHUNK_S
    edges = [0.0, *cuts, duration]
    return list(itertools.pairwise(edges))


def _mlx_segments(x: np.ndarray, *, offset: float, language: str | None, repo_path: str) -> tuple[list[Seg], str]:
    """Кусок звука (16 кГц) → сегменты mlx-whisper со временем от начала файла."""
    import mlx_whisper
    r = mlx_whisper.transcribe(
        x, path_or_hf_repo=repo_path, language=language or None, word_timestamps=True,
        condition_on_previous_text=False, verbose=None,
        # свои пороги ниже (по сумме признаков): штатные отбрасывают вместе с мусором и реальные реплики
        no_speech_threshold=None, logprob_threshold=None, compression_ratio_threshold=None)
    segs = []
    for s in r["segments"]:
        words = [Word(start=float(w["start"]) + offset, end=float(w["end"]) + offset, text=w["word"],
                      prob=float(w["probability"]) if w.get("probability") is not None else None)
                 for w in s.get("words", [])]
        segs.append(Seg(float(s["start"]) + offset, float(s["end"]) + offset, s["text"], words, s.get("avg_logprob"),
                        s.get("no_speech_prob"), s.get("compression_ratio")))
    return segs, (r.get("language") or "")


def _transcribe_mlx(path: Path, *, language: str | None, model: str, on_progress, env: Envelope | None
                    ) -> tuple[list[Seg], str]:
    import soundfile as sf

    repo_path = _mlx_model_path(model)
    duration = sf.info(str(path)).duration
    segs: list[Seg] = []
    lang = language or ""
    for a, b in _chunk_bounds(duration, env):
        part, lg = _mlx_segments(_read_chunk(path, a, b), offset=a, language=lang or None, repo_path=repo_path)
        segs += part
        lang = lang or lg
        if on_progress and duration > 0:
            on_progress(min(0.99, b / duration))
    return segs, lang


# faster-whisper: загруженная модель на время одного разбора (определение языка, распознавание и добор — одна загрузка)
_FW_CACHE: dict[tuple[str, str], tuple[object, str]] = {}


def _faster_model(model: str, device: str):
    """(WhisperModel, устройство). Первое устройство, на котором модель загрузилась; упавшие помечаются."""
    try:
        from faster_whisper import WhisperModel
    except ImportError as e:  # pragma: no cover
        raise MlMissingError("Не установлены нейросети — без них нельзя распознавать речь. Скачай их один раз.") from e
    key = (model, device)
    if key in _FW_CACHE:
        return _FW_CACHE[key]
    last_err: Exception | None = None
    for cand in device_candidates(device, task="whisper"):
        dev, ctype = whisper_runtime(cand)
        try:
            wm = WhisperModel(model, device=dev, compute_type=ctype, download_root=str(whisper_dir()))
            _FW_CACHE[key] = (wm, dev)
            return wm, dev
        except Exception as e:  # noqa: BLE001
            log.warning("Whisper не запустился на %s: %s", dev, e)
            mark_bad("whisper", cand)
            last_err = e
    raise EngineError(f"Не удалось загрузить модель распознавания «{model}»: {last_err}")


def _faster_segments(audio, *, offset: float, language: str | None, model: str, device: str, vad: bool,
                     on_progress=None) -> tuple[list[Seg], str]:
    """Файл или кусок звука (numpy 16 кГц) → сегменты faster-whisper со временем от начала файла."""
    wm, _used = _faster_model(model, device)
    segments, info = wm.transcribe(
        audio, language=language, word_timestamps=True, vad_filter=vad,
        vad_parameters=FW_VAD_PARAMS if vad else None, beam_size=5, condition_on_previous_text=False,
        no_speech_threshold=None, log_prob_threshold=None, compression_ratio_threshold=None)
    total = float(getattr(info, "duration", 0.0) or 0.0)
    out: list[Seg] = []
    for seg in segments:
        words = [Word(start=float(w.start) + offset, end=float(w.end) + offset, text=w.word,
                      prob=float(w.probability) if w.probability is not None else None) for w in (seg.words or [])]
        out.append(Seg(float(seg.start) + offset, float(seg.end) + offset, seg.text, words, getattr(seg, "avg_logprob", None),
                       getattr(seg, "no_speech_prob", None), getattr(seg, "compression_ratio", None)))
        if on_progress and total > 0:
            on_progress(min(0.99, float(seg.end) / total))
    return out, (info.language or language or "")


def _transcribe_faster(path: Path, *, language: str | None, model: str, device: str, on_progress,
                       vad: bool | None) -> tuple[list[Seg], str]:
    _wm, used = _faster_model(model, device)
    # CUDA достаточно быстра без VAD (он режет диалог под музыкой); на CPU «мягкий» VAD экономит время на тишине.
    use_vad = (used != "cuda") if vad is None else vad
    return _faster_segments(str(path), offset=0.0, language=language, model=model, device=device, vad=use_vad,
                            on_progress=on_progress)


def segs_to_lines(segs: list[Seg], *, start_id: int = 1) -> list[Line]:
    lines: list[Line] = []
    for seg in segs:
        for s, e, text, grp in _segment_to_groups(seg.words, seg.text, seg.start, seg.end):
            lines.append(Line(id=f"l{len(lines) + start_id:04d}", start=s, end=max(e, s), text=text,
                              words=grp, source="whisper"))
    return lines


def whisper_segments(path: Path, *, language: str | None, model: str, device: str = "auto", backend: str | None = None,
                     vad: bool | None = None, on_progress=None, env: Envelope | None = None) -> tuple[list[Seg], str]:
    """Весь файл Whisper'ом (mlx на Mac arm64, иначе faster-whisper) → сегменты после фильтра выдумок и язык."""
    be = backend or pick_backend(model, device)
    segs: list[Seg] | None = None
    lang = ""
    if be == "mlx":
        try:
            segs, lang = _transcribe_mlx(path, language=language, model=model, on_progress=on_progress, env=env)
        except Exception as e:  # noqa: BLE001 — упал GPU/модель: пробуем faster-whisper
            log.warning("mlx-whisper не сработал (%s) — перехожу на faster-whisper", e)
    if segs is None:
        segs, lang = _transcribe_faster(path, language=language, model=model, device=device,
                                        on_progress=on_progress, vad=vad)
    return drop_hallucinations(segs, env), lang


def whisper_window(x: np.ndarray, *, offset: float, language: str | None, model: str, device: str = "auto",
                   backend: str | None = None) -> list[Seg]:
    """Короткий кусок звука Whisper'ом (для добора после GigaAM). Без фильтра — фильтрует вызывающий."""
    be = backend or pick_backend(model, device)
    if be == "mlx":
        try:
            return _mlx_segments(x, offset=offset, language=language, repo_path=_mlx_model_path(model))[0]
        except Exception as e:  # noqa: BLE001
            log.warning("mlx-whisper не сработал на куске (%s) — перехожу на faster-whisper", e)
    return _faster_segments(x, offset=offset, language=language, model=model, device=device, vad=False)[0]


# ───────────────────────── определение языка по речи ─────────────────────────
LID_WIN_S = 30.0            # окно Whisper
LID_WINDOWS = 3             # сколько окон слушаем


def speech_windows(env: Envelope | None, duration: float, *, n: int = LID_WINDOWS, win: float = LID_WIN_S
                   ) -> list[tuple[float, float]]:
    """До n непересекающихся окон по win секунд, где больше всего «громких» кадров (речь, а не тишина).
    Без огибающей — начало файла."""
    if duration <= 0:
        return []
    if duration <= win or env is None or not len(env.db):
        return [(0.0, min(duration, win))]
    loud = (env.db > env.ref - 20).astype(np.float64)
    w = int(win / ENV_HOP)
    step = max(1, int(5.0 / ENV_HOP))
    cands = []
    for i in range(0, max(1, len(loud) - w + 1), step):
        cands.append((float(loud[i:i + w].mean()), i * ENV_HOP))
    cands.sort(key=lambda c: -c[0])
    out: list[tuple[float, float]] = []
    for frac, a in cands:
        if frac <= 0.05:
            break
        if all(abs(a - b) >= win for b, _ in out):
            out.append((a, min(duration, a + win)))
        if len(out) >= n:
            break
    return sorted(out) or [(0.0, min(duration, win))]


def _lid_mlx(chunks: list[np.ndarray], model: str) -> list[dict[str, float]]:
    import mlx.core as mx
    from mlx_whisper.audio import N_FRAMES, log_mel_spectrogram, pad_or_trim
    from mlx_whisper.decoding import detect_language
    from mlx_whisper.transcribe import ModelHolder

    m = ModelHolder.get_model(_mlx_model_path(model), mx.float16)
    out = []
    for x in chunks:
        mel = log_mel_spectrogram(x, n_mels=m.dims.n_mels, padding=0)
        mel = pad_or_trim(mel, N_FRAMES, axis=-2).astype(mx.float16)
        _, probs = detect_language(m, mel)
        out.append({k: float(v) for k, v in probs.items()})
    return out


def _lid_faster(chunks: list[np.ndarray], model: str, device: str) -> list[dict[str, float]]:
    wm, _ = _faster_model(model, device)
    out = []
    for x in chunks:
        _lang, _p, allp = wm.detect_language(audio=x, language_detection_segments=1)
        out.append({k: float(v) for k, v in allp})
    return out


def detect_language_probs(path: Path, *, env: Envelope | None, model: str, device: str = "auto",
                          backend: str | None = None) -> list[dict[str, float]]:
    """Вероятности языков по нескольким окнам с речью (Whisper detect_language, ≈ 1 с на окно на Mac)."""
    import soundfile as sf
    duration = sf.info(str(path)).duration
    chunks = [_read_chunk(path, a, b) for a, b in speech_windows(env, duration)]
    chunks = [x for x in chunks if len(x) >= SR]
    if not chunks:
        return []
    be = backend or pick_backend(model, device)
    if be == "mlx":
        try:
            return _lid_mlx(chunks, model)
        except Exception as e:  # noqa: BLE001
            log.warning("mlx-whisper не определил язык (%s) — пробую faster-whisper", e)
    return _lid_faster(chunks, model, device)


LID_SURE = 0.6              # средняя вероятность лучшего языка, при которой метке дорожки уже не верим
LID_TRACK_PLAUSIBLE = 0.2   # …а при меньшей — берём метку, если речь хоть как-то на неё похожа


def choose_language(window_probs: list[dict[str, float]], track: str | None) -> tuple[str | None, str]:
    """(язык, откуда): по сумме вероятностей окон. Уверенно (≥ LID_SURE) — язык речи («speech»); неуверенно — метка дорожки
    («track»), если речь на неё хоть немного похожа; иначе всё равно язык речи. Окон нет — метка (или None)."""
    if not window_probs:
        return (track, "track") if track else (None, "speech")
    total: dict[str, float] = {}
    for probs in window_probs:
        for k, v in probs.items():
            total[k] = total.get(k, 0.0) + float(v)
    best = max(total, key=lambda k: total[k])
    n = len(window_probs)
    if total[best] / n >= LID_SURE or not track:
        return best, "speech"
    if total.get(track, 0.0) / n >= LID_TRACK_PLAUSIBLE:
        return track, "track"
    return best, "speech"


# ───────────────────────── главная функция ─────────────────────────
def transcribe(audio_16k_wav, *, language: str | None = None, model: str = "large-v3-turbo",
               on_progress: Callable[[float], None] | None = None,
               device: str = "auto", backend: str | None = None, vad: bool | None = None,
               keep_loaded: bool = False, engine: str = "whisper", track_language: str | None = None,
               on_note: Callable[[str], None] | None = None, info: dict | None = None) -> tuple[list[Line], str]:
    """Речь → реплики со словами → (реплики, язык ISO 639-1).

    language: язык, выбранный человеком, — закон. None — определить по речи (`choose_language`; метка дорожки
    `track_language` — только подсказка при низкой уверенности).
    engine: "whisper" | "gigaam" | "auto" (`pick_engine`): русский — GigaAM с добором пропусков Whisper'ом, остальное —
    Whisper. GigaAM не установлен/не скачался/упал — Whisper и пояснение через on_note.
    backend: None (выбор сам: Mac arm64 → mlx-whisper, иначе faster-whisper), "mlx" или "faster".
    vad: только для faster-whisper; None — «мягкий» VAD на CPU и без VAD на CUDA.
    keep_loaded: не отдавать память после распознавания (mlx держит модель в памяти) — для частых коротких вызовов;
    отдать её потом должен вызывающий (`release_memory`).
    info: сюда пишется, чем распознано и откуда язык (asr_engine, language_source, notes, filled…).
    Выдуманные фразы на музыке и в тишине отбрасываются (`drop_hallucinations`).
    """
    path = Path(audio_16k_wav)
    if not path.exists():
        raise EngineError(f"Нет файла со звуком: {path}")
    info = {} if info is None else info
    try:
        try:
            env: Envelope | None = Envelope.from_file(path)
        except Exception as e:  # noqa: BLE001 — без огибающей работаем, просто фильтр слабее
            log.warning("Не удалось измерить громкость звука (%s)", e)
            env = None
        if language is None or engine != "whisper":
            from .speech import recognize
            lines, lang = recognize(path, env=env, language=language, track_language=track_language, engine=engine,
                                    model=model, device=device, backend=backend, vad=vad, on_progress=on_progress,
                                    on_note=on_note, info=info)
        else:
            segs, lang = whisper_segments(path, language=language, model=model, device=device, backend=backend, vad=vad,
                                          on_progress=on_progress, env=env)
            lines = segs_to_lines(segs)
            info.update(asr_engine="whisper", language_source="user")
    except EngineError:
        raise
    except Exception as e:
        raise EngineError(f"Распознавание речи не удалось: {e}") from e
    finally:
        if not keep_loaded:
            release_memory()
    if on_progress:
        on_progress(1.0)
    return lines, (lang or language or "")
