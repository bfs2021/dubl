"""Определение говорящих.

Основной путь (без токена): эмбеддинг голоса каждой реплики (speechbrain ECAPA, бесплатный, без токена) →
вычитание среднего по фильму → агломеративная кластеризация (average linkage, косинус) → число голосов выбирается
по силуэту (silhouette) среди разбиений на 2…kmax групп; реплики, перекрывающиеся по времени, не склеиваются.
Результат детерминирован: никакой случайности нет, расстояния округляются до 1e-6, порядок реплик фиксирован.
Если есть токен Hugging Face и установлен pyannote.audio — pyannote/speaker-diarization-3.1.
"""
from __future__ import annotations

import logging
from collections.abc import Callable

import numpy as np

from ..models import Line, Speaker
from ..util import EngineError, MlMissingError, data_dir

log = logging.getLogger(__name__)

SR = 16000
MIN_EMBED = 0.5            # реплики короче в кластеризации не участвуют (их присоединяем к ближайшему)
MAX_EMBED = 10.0           # из длинной реплики берём не больше столько секунд (из середины)
CLUSTER_THRESHOLD = 0.98   # порог склейки по расстоянию — только если его задали явно (старый режим, бенчи); по умолчанию число голосов выбирает силуэт
SIL_MIN = 0.08             # лучший силуэт ниже этого → разбиения нет смысла делать, один голос (чистый шум в 192 измерениях даёт ≤ 0,055; у настоящих фильмов ≥ 0,11)
OVERLAP_MIN = 0.3          # реплики, пересекающиеся дольше стольких секунд, — разные люди: расстояние между ними максимальное
DIST_DECIMALS = 6          # округление расстояний: ничтожные различия float32 от числа потоков не должны менять кластеры
RAW_THRESHOLD = 0.85       # то же без центрирования — для совсем коротких списков реплик (< CENTER_MIN); один голос в ToS склеивается при 0,75–0,82
CENTER_MIN = 8             # меньше стольких реплик среднее по фильму оценить нельзя
MFCC_THRESHOLD = 0.35      # то же для запасных MFCC-эмбеддингов (после центрирования по измерениям)
MERGE_SIMILARITY = 0.8     # порог «это один человек?» для обычных (нецентрированных) центроидов
MERGE_SIMILARITY_CENTERED = 0.55  # то же в шкале 0..1 для центрированных центроидов: (1 + косинус) / 2, т. е. косинус ≥ 0,1, и только взаимно ближайшие пары
MAX_SPEAKERS = 16          # страховочный потолок, а не рабочее число: обычно число голосов решает порог
_FALLBACK_COLORS = ["#4fb8a8", "#e0a458", "#7aa2f7", "#e06c75", "#b48ead",
                    "#98c379", "#d19a66", "#56b6c2", "#c678dd", "#e5c07b"]


# ───────────────────────── чистая логика (без нейросетей) ─────────────────────────
def _normalize(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(n, 1e-9)


def _relabel(lab: np.ndarray) -> np.ndarray:
    remap: dict[int, int] = {}
    out = np.zeros(len(lab), dtype=int)
    for i, v in enumerate(lab):
        out[i] = remap.setdefault(int(v), len(remap))
    return out


def min_cluster_size(n: int) -> int:
    """Кластеры меньше этого размера — «мусор» (обрывки, шум, один и тот же человек в другой обстановке): присоединяем к соседям."""
    return 1 if n < 12 else 2 if n < 30 else 3


def absorb_small_clusters(emb: np.ndarray, labels: np.ndarray, min_size: int) -> np.ndarray:
    """Кластеры из < min_size реплик по очереди присоединяются к ближайшему (по косинусу центроидов) крупному кластеру."""
    labels = np.asarray(labels).copy()
    if min_size <= 1 or len(set(labels.tolist())) < 2:
        return labels
    x = _normalize(np.asarray(emb, dtype=np.float64))
    while True:
        ids, counts = np.unique(labels, return_counts=True)
        small = [int(c) for c, k in zip(ids, counts) if k < min_size]
        big = [int(c) for c, k in zip(ids, counts) if k >= min_size]
        if not small or not big:
            break
        cents = {int(c): _normalize(x[labels == c].mean(axis=0, keepdims=True))[0] for c in ids}
        c = min(small, key=lambda v: int((labels == v).sum()))      # самый маленький первым
        tgt = max(big, key=lambda b: float(cents[c] @ cents[b]))
        labels[labels == c] = tgt
    return _relabel(labels)


def max_clusters_for(n: int) -> int:
    """Сколько голосов вообще можно искать в n репликах: примерно одна группа на 9 реплик, от 6 до MAX_SPEAKERS.
    Без этого силуэт на длинных фильмах дробит одного человека (его голос меняется от сцены к сцене и от шума).
    Выбрано по сумме трёх фильмов (ToS, Sintel, Cosmos): kmax 7–9 даёт одинаково лучшие результаты на ToS, а на Sintel и Cosmos результат от kmax 4…10 не зависит."""
    return int(min(MAX_SPEAKERS, max(6, n // 9)))


def _distance_matrix(x: np.ndarray, spans: np.ndarray | None) -> np.ndarray:
    """Косинусное расстояние между строками (0..2); пересекающиеся по времени реплики — максимально далеко (2)."""
    xn = _normalize(x)
    d = np.clip(1.0 - xn @ xn.T, 0.0, 2.0)
    d = (d + d.T) / 2
    if spans is not None and len(spans) == len(x):
        st, en = spans[:, 0], spans[:, 1]
        ov = (np.minimum(en[:, None], en[None, :]) - np.maximum(st[:, None], st[None, :])) > OVERLAP_MIN
        d = np.where(ov, 2.0, d)
    np.fill_diagonal(d, 0.0)
    return np.round(d, DIST_DECIMALS)


def _auto_labels(x: np.ndarray, d: np.ndarray, kmax: int, min_size: int) -> np.ndarray:
    """Число групп выбирается по среднему силуэту: для k = 2…kmax режем дерево на k групп, мелкие присоединяем и берём лучшее.
    При равенстве — меньшее k. Если лучший силуэт < SIL_MIN, считаем, что говорящий один."""
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform
    from sklearn.metrics import silhouette_score

    n = len(x)
    z = linkage(squareform(d, checks=False), method="average")
    best: tuple[float, np.ndarray] | None = None
    seen: set[tuple[int, ...]] = set()
    for k in range(2, min(kmax, n - 1) + 1):
        lab = fcluster(z, t=k, criterion="maxclust")
        if min_size > 1:
            lab = absorb_small_clusters(x, lab, min_size)
        lab = _relabel(lab)
        key = tuple(lab.tolist())
        if lab.max() < 1 or key in seen:
            continue
        seen.add(key)
        sil = float(silhouette_score(d, lab, metric="precomputed"))
        if best is None or sil > best[0] + 1e-9:
            best = (sil, lab)
    if best is None or best[0] < SIL_MIN:
        return np.zeros(n, dtype=int)
    return best[1]


def enforce_no_overlap(x: np.ndarray, labels: np.ndarray, spans: np.ndarray) -> np.ndarray:
    """Реплики одной группы не должны звучать одновременно (> OVERLAP_MIN с): одну из пары переносим в ближайшую группу,
    где у неё нет такого конфликта, иначе — в новую (одновременно говорят разные люди). Порядок обхода фиксирован."""
    lab = np.asarray(labels).copy()
    n = len(lab)
    if n < 2:
        return lab
    st, en = spans[:, 0], spans[:, 1]
    ov = (np.minimum(en[:, None], en[None, :]) - np.maximum(st[:, None], st[None, :])) > OVERLAP_MIN
    np.fill_diagonal(ov, False)
    xn = _normalize(np.asarray(x, dtype=np.float64))
    for _ in range(n):                                   # каждый проход убирает хотя бы один конфликт
        conflict = np.argwhere(np.triu(ov & (lab[:, None] == lab[None, :]), k=1))
        if not len(conflict):
            break
        i, j = int(conflict[0][0]), int(conflict[0][1])
        cents = {int(c): _normalize(xn[lab == c].mean(axis=0, keepdims=True))[0] for c in np.unique(lab)}

        def fit(k: int) -> float:                        # насколько реплика k похожа на свою группу без неё самой
            rest = (lab == lab[k]) & (np.arange(n) != k)
            return float(xn[k] @ _normalize(xn[rest].mean(axis=0, keepdims=True))[0]) if rest.any() else 1.0
        mover = i if fit(i) <= fit(j) else j
        free = [c for c in cents if c != lab[mover] and not ov[mover, lab == c].any()]
        lab[mover] = max(free, key=lambda c: (float(xn[mover] @ cents[c]), -c)) if free else int(lab.max()) + 1
    return _relabel(lab)


def cluster_embeddings(emb: np.ndarray, *, threshold: float | None = None,
                       max_clusters: int = MAX_SPEAKERS, center: bool = True, min_size: int | None = None,
                       spans: np.ndarray | None = None) -> np.ndarray:
    """Агломеративная кластеризация (average linkage, косинусное расстояние). Метки 0..k-1, детерминированно.

    center=True: из эмбеддингов сначала вычитается среднее по фильму. ECAPA на речи поверх музыки и шумов
    даёт эмбеддинги с общей «фоновой» составляющей, и пороги в «сыром» пространстве ничего не различают.
    threshold=None (по умолчанию) и реплик ≥ CENTER_MIN: число голосов выбирает силуэт среди разбиений на 2…kmax групп
    (kmax = max_clusters_for(n), не больше max_clusters) — порог не нужен и не подгоняется под фильм.
    threshold задан (или реплик < CENTER_MIN, среднее оценить нельзя — тогда RAW_THRESHOLD): две группы склеиваются, пока среднее
    расстояние между ними < threshold.
    spans — (n, 2) начало/конец каждой реплики: пересекающиеся по времени реплики разные люди, склеиваться напрямую не будут.
    После этого кластеры меньше min_size присоединяются к ближайшим; если групп всё равно больше max_clusters — склеиваем.
    """
    emb = np.asarray(emb, dtype=np.float64)
    n = len(emb)
    if n == 0:
        return np.zeros(0, dtype=int)
    if n == 1:
        return np.zeros(1, dtype=int)
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform

    x = emb
    if center and n >= CENTER_MIN:
        x = x - x.mean(axis=0, keepdims=True)
    ms = min_cluster_size(n) if min_size is None else min_size
    sp = None if spans is None else np.asarray(spans, dtype=np.float64)
    d = _distance_matrix(x, sp)
    if threshold is None and n >= CENTER_MIN:
        lab = _relabel(_auto_labels(x, d, min(max_clusters, max_clusters_for(n)), ms))
        return enforce_no_overlap(x, lab, sp) if sp is not None and len(sp) == n else lab
    if threshold is None or (center and n < CENTER_MIN):    # мало реплик — среднего нет, берём «сырой» порог
        threshold = RAW_THRESHOLD if threshold is None else min(threshold, RAW_THRESHOLD)
    z = linkage(squareform(d, checks=False), method="average")
    lab = fcluster(z, t=threshold, criterion="distance")
    if ms > 1:
        lab = absorb_small_clusters(x, lab, ms)
    if len(set(lab.tolist())) > max_clusters:
        lab = fcluster(z, t=max_clusters, criterion="maxclust")
    lab = _relabel(lab)
    return enforce_no_overlap(x, lab, sp) if sp is not None and len(sp) == n else lab


def centroids(emb: np.ndarray, labels: np.ndarray) -> np.ndarray:
    emb = _normalize(np.asarray(emb, dtype=np.float64))
    k = int(labels.max()) + 1 if len(labels) else 0
    return _normalize(np.stack([emb[labels == c].mean(axis=0) for c in range(k)])) if k else np.zeros((0, emb.shape[1]))


def assign_to_nearest(emb: np.ndarray, cents: np.ndarray) -> np.ndarray:
    """Индекс ближайшего (по косинусу) центроида для каждого эмбеддинга."""
    if len(cents) == 0:
        return np.zeros(len(emb), dtype=int)
    return np.argmax(_normalize(np.asarray(emb, dtype=np.float64)) @ cents.T, axis=1)


def find_overlaps(lines: list[Line], speakers: list[str | None]) -> list[bool]:
    """True для реплик, интервал которых пересекается с репликой другого говорящего."""
    n = len(lines)
    order = sorted(range(n), key=lambda i: lines[i].start)
    res = [False] * n
    for a_pos, a in enumerate(order):
        for b in order[a_pos + 1:]:
            if lines[b].start >= lines[a].end:
                break
            if speakers[a] is not None and speakers[b] is not None and speakers[a] != speakers[b]:
                res[a] = res[b] = True
    return res


def merge_suggestions_from(cents: np.ndarray, ids: list[str], *, min_sim: float = MERGE_SIMILARITY,
                           mutual: bool = False) -> list[tuple[str, str, float]]:
    """Пары кластеров, которые, возможно, один человек. mutual=True: только взаимно ближайшие друг к другу пары —
    на разметке трёх фильмов такие верны заметно чаще, чем «просто похожие» (см. docs/QUALITY.md)."""
    out = []
    if len(cents) < 2:
        return out
    sim = _normalize(cents) @ _normalize(cents).T
    nn = np.argmax(np.where(np.eye(len(ids), dtype=bool), -9.0, sim), axis=1)
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            if sim[i, j] > min_sim and (not mutual or (nn[i] == j and nn[j] == i)):
                out.append((ids[i], ids[j], round(float(sim[i, j]), 3)))
    out.sort(key=lambda t: (-t[2], t[0], t[1]))
    return out


def _palette() -> list[str]:
    try:
        from .palette import PALETTE
        return list(PALETTE)
    except Exception:  # noqa: BLE001 — palette.py пишет другой модуль
        return _FALLBACK_COLORS


def _build_result(lines: list[Line], raw_labels: list[int | None], cents_by_label: dict[int, np.ndarray] | None,
                  overlap_extra: list[bool] | None = None, *, centered: bool = False
                  ) -> tuple[list[Line], list[Speaker], list[tuple[str, str, float]]]:
    """Из «сырых» меток (произвольные int) строим Speaker s1.. по убыванию объёма речи и размечаем Line."""
    total: dict[int, float] = {}
    count: dict[int, int] = {}
    for ln, lab in zip(lines, raw_labels):
        if lab is None:
            continue
        total[lab] = total.get(lab, 0.0) + max(0.0, ln.end - ln.start)
        count[lab] = count.get(lab, 0) + 1
    order = sorted(total, key=lambda c: (-total[c], c))
    sid = {lab: f"s{i + 1}" for i, lab in enumerate(order)}
    pal = _palette()
    speakers = [Speaker(id=sid[lab], label=f"Голос {i + 1}", color=pal[i % len(pal)],
                        total_speech=round(total[lab], 3), line_count=count[lab])
                for i, lab in enumerate(order)]
    spk_ids = [sid.get(lab) if lab is not None else None for lab in raw_labels]
    ov = find_overlaps(lines, spk_ids)
    if overlap_extra:
        ov = [a or b for a, b in zip(ov, overlap_extra)]
    new_lines = [ln.model_copy(update={"speaker_id": s, "overlap": o}) for ln, s, o in zip(lines, spk_ids, ov)]
    sugg: list[tuple[str, str, float]] = []
    if cents_by_label:
        labs = [lab for lab in order if lab in cents_by_label]
        if len(labs) > 1:
            cs = np.stack([cents_by_label[lab] for lab in labs])
            ids = [sid[lab] for lab in labs]
            if centered:      # центроиды после вычитания среднего: косинус ≥ 0 → «похожи», в шкалу 0..1 переводим (1 + cos) / 2
                sugg = [(a, b, round((1 + v) / 2, 3)) for a, b, v in
                        merge_suggestions_from(cs, ids, min_sim=2 * MERGE_SIMILARITY_CENTERED - 1, mutual=True)]
            else:
                sugg = merge_suggestions_from(cs, ids)
    return new_lines, speakers, sugg


# ───────────────────────── эмбеддинги ─────────────────────────
def _load_audio(path) -> np.ndarray:
    from ..audio.io import read_wav, to_mono
    x, sr = read_wav(path)
    x = to_mono(x)
    if sr != SR:
        from math import gcd

        from scipy.signal import resample_poly
        g = gcd(sr, SR)
        x = resample_poly(x, SR // g, sr // g).astype(np.float32)
    return x


class _LazyAudio:
    """Звук 16 кГц с диска кусками: фильм на 2 часа в float32 — это 460 МБ, а нам нужны только куски реплик."""

    def __init__(self, path) -> None:
        import soundfile as sf
        self._path = str(path)
        info = sf.info(self._path)
        self._sr, self._n = info.samplerate, info.frames
        if self._sr != SR:                       # нестандартный файл — один раз читаем целиком, как раньше
            self._whole: np.ndarray | None = _load_audio(path)
            self._n = len(self._whole)
        else:
            self._whole = None

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, sl: slice) -> np.ndarray:
        a, b, _ = sl.indices(self._n)
        if b <= a:
            return np.zeros(0, dtype=np.float32)
        if self._whole is not None:
            return self._whole[a:b]
        import soundfile as sf
        x, _sr = sf.read(self._path, start=a, stop=b, dtype="float32", always_2d=False)
        return x if x.ndim == 1 else x.mean(axis=1).astype(np.float32)


def _line_chunk(x, ln: Line) -> np.ndarray:
    s, e = ln.start, ln.end
    if e - s > MAX_EMBED:
        mid = (s + e) / 2
        s, e = mid - MAX_EMBED / 2, mid + MAX_EMBED / 2
    return x[max(0, int(s * SR)): max(0, int(e * SR))]


class _SpeechbrainEmbedder:
    name = "ecapa"

    def __init__(self) -> None:
        import torch
        try:
            from speechbrain.inference.speaker import EncoderClassifier
        except ImportError:  # старые версии speechbrain
            from speechbrain.pretrained import EncoderClassifier
        d = data_dir() / "models" / "speechbrain" / "spkrec-ecapa-voxceleb"
        d.mkdir(parents=True, exist_ok=True)
        self.torch = torch
        try:
            from speechbrain.utils.fetching import LocalStrategy
            extra = {"local_strategy": LocalStrategy.COPY}  # копии, не симлинки: надёжно на Windows
        except ImportError:
            extra = {}
        self.model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb", savedir=str(d), run_opts={"device": "cpu"}, **extra)

    def __call__(self, chunk: np.ndarray) -> np.ndarray:
        t = self.torch.from_numpy(np.ascontiguousarray(chunk, dtype=np.float32))[None, :]
        with self.torch.no_grad():
            e = self.model.encode_batch(t)
        return e.squeeze().cpu().numpy().astype(np.float64)


class _MfccEmbedder:
    """Запасной вариант: статистики MFCC (среднее и разброс). Хуже ECAPA, но не требует скачивания."""
    name = "mfcc"

    def __init__(self) -> None:
        import torch
        import torchaudio
        self.torch = torch
        self.mfcc = torchaudio.transforms.MFCC(sample_rate=SR, n_mfcc=30, melkwargs={"n_fft": 512, "n_mels": 64})

    def __call__(self, chunk: np.ndarray) -> np.ndarray:
        m = self.mfcc(self.torch.from_numpy(np.ascontiguousarray(chunk, dtype=np.float32))[None, :])[0].numpy()
        return np.concatenate([m.mean(axis=1), m.std(axis=1)]).astype(np.float64)


def _embed_lines(x, lines: list[Line], idx: list[int], embedder,
                 on_progress: Callable[[float], None] | None, lo: float, hi: float) -> np.ndarray:
    out = []
    for k, i in enumerate(idx):
        out.append(embedder(_line_chunk(x, lines[i])))
        if on_progress:
            on_progress(lo + (hi - lo) * (k + 1) / max(1, len(idx)))
    return np.stack(out) if out else np.zeros((0, 1))


# ───────────────────────── основной фолбэк ─────────────────────────
def _diarize_embeddings(audio_16k_wav, lines: list[Line], on_progress, cluster_threshold: float | None = None):
    x = _LazyAudio(audio_16k_wav)
    n = len(lines)
    # реплики, которые вообще можно разобрать (≥0.25 с)
    ok = [i for i in range(n) if lines[i].end - lines[i].start >= 0.25 and len(_line_chunk(x, lines[i])) >= SR // 4]
    long_idx = [i for i in ok if lines[i].end - lines[i].start >= MIN_EMBED]
    if not long_idx:  # нечего кластеризовать — все в одного говорящего
        return _build_result(lines, [0] * n if n else [], None)

    try:
        embedder = _SpeechbrainEmbedder()
    except Exception as e:  # noqa: BLE001
        log.warning("ECAPA недоступна (%s) — использую MFCC-статистики", e)
        embedder = _MfccEmbedder()

    if on_progress:
        on_progress(0.05)
    try:
        e_long = _embed_lines(x, lines, long_idx, embedder, on_progress, 0.05, 0.8)
    except Exception as e:
        if embedder.name == "mfcc":
            raise EngineError(f"Не удалось определить говорящих: {e}") from e
        log.warning("ECAPA упала (%s) — использую MFCC-статистики", e)
        embedder = _MfccEmbedder()
        e_long = _embed_lines(x, lines, long_idx, embedder, on_progress, 0.05, 0.8)

    centered = False
    if embedder.name == "mfcc":  # центрируем и нормируем по измерениям, чтобы косинус что-то значил
        mu, sd = e_long.mean(axis=0), e_long.std(axis=0) + 1e-6
        norm = lambda v: (v - mu) / sd
        e_long = norm(e_long)
        threshold = MFCC_THRESHOLD
    elif len(e_long) >= CENTER_MIN:   # ECAPA: вычитаем среднее по фильму (общую «фоновую» составляющую), см. cluster_embeddings
        mu = e_long.mean(axis=0)
        norm = lambda v: v - mu
        e_long = norm(e_long)
        threshold, centered = CLUSTER_THRESHOLD, True
    else:
        norm = lambda v: v
        threshold = RAW_THRESHOLD

    spans = np.array([[lines[i].start, lines[i].end] for i in long_idx])
    # ECAPA с ≥ CENTER_MIN реплик: число голосов выбирает силуэт; иначе (мало реплик, MFCC) — порог
    auto = embedder.name == "ecapa" and centered and cluster_threshold is None
    labels = cluster_embeddings(e_long, threshold=None if auto else (cluster_threshold or threshold),
                                center=False, spans=spans)
    cents = centroids(e_long, labels)

    raw: list[int | None] = [None] * n
    for i, lab in zip(long_idx, labels):
        raw[i] = int(lab)
    # короткие реплики — к ближайшему центроиду
    long_set = set(long_idx)
    short_idx = [i for i in ok if i not in long_set]
    if short_idx:
        e_short = norm(_embed_lines(x, lines, short_idx, embedder, None, 0, 0))
        for i, lab in zip(short_idx, assign_to_nearest(e_short, cents)):
            raw[i] = int(lab)
    # совсем крошечные (< 0.25 с) — как у ближайшей по времени размеченной
    labeled = [i for i in range(n) if raw[i] is not None]
    for i in range(n):
        if raw[i] is None:
            j = min(labeled, key=lambda k: abs((lines[k].start + lines[k].end) / 2 - (lines[i].start + lines[i].end) / 2))
            raw[i] = raw[j]
    if on_progress:
        on_progress(0.95)
    del embedder                       # модель ECAPA больше не нужна — не держим её до конца процесса
    import gc
    gc.collect()
    return _build_result(lines, raw, {c: cents[c] for c in range(len(cents))}, centered=centered)


# ───────────────────────── pyannote ─────────────────────────
def _diarize_pyannote(audio_16k_wav, lines: list[Line], hf_token: str, on_progress):
    import torch
    from pyannote.audio import Pipeline

    try:
        pipe = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=hf_token)
    except TypeError:
        pipe = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", token=hf_token)
    if pipe is None:
        raise EngineError("Hugging Face не дал доступ к модели, которая различает голоса: прими условия модели на её странице.")
    x = _load_audio(audio_16k_wav)
    if on_progress:
        on_progress(0.1)
    diar = pipe({"waveform": torch.from_numpy(x)[None, :], "sample_rate": SR})
    ann = getattr(diar, "speaker_diarization", diar)  # pyannote 4 отдаёт обёртку
    turns = [(t.start, t.end, str(lab)) for t, _, lab in ann.itertracks(yield_label=True)]
    if on_progress:
        on_progress(0.9)

    def overlap(a0, a1, b0, b1):
        return max(0.0, min(a1, b1) - max(a0, b0))

    raw: list[int | None] = []
    ids: dict[str, int] = {}
    extra: list[bool] = []
    for ln in lines:
        per: dict[str, float] = {}
        for s, e, lab in turns:
            o = overlap(ln.start, ln.end, s, e)
            if o > 0:
                per[lab] = per.get(lab, 0.0) + o
        if per:
            best = max(per, key=per.get)
            raw.append(ids.setdefault(best, len(ids)))
            dur = max(ln.end - ln.start, 1e-6)
            extra.append(any(v > 0.3 and v / dur > 0.15 for k, v in per.items() if k != best))
        else:
            raw.append(None)
            extra.append(False)
    # реплики вне речи pyannote — к ближайшей по времени размеченной
    labeled = [i for i, r in enumerate(raw) if r is not None]
    if labeled:
        for i, r in enumerate(raw):
            if r is None:
                j = min(labeled, key=lambda k: abs(lines[k].start - lines[i].start))
                raw[i] = raw[j]
    return _build_result(lines, raw, None, extra)


# ───────────────────────── публичная функция ─────────────────────────
def diarize(audio_16k_wav, lines: list[Line], *, hf_token: str | None,
            on_progress: Callable[[float], None] | None = None, cluster_threshold: float | None = None
            ) -> tuple[list[Line], list[Speaker], list[tuple[str, str, float]]]:
    """Расставляет speaker_id по репликам. Возвращает (новые реплики, говорящие, предложения «это один человек?»).
    cluster_threshold — для замеров: принудительно кластеризовать по порогу расстояния вместо выбора числа голосов по силуэту."""
    if not lines:
        return [], [], []
    lines = sorted(lines, key=lambda l: (l.start, l.end, l.id))
    if hf_token:
        try:
            import pyannote.audio  # noqa: F401
        except ImportError:
            log.info("pyannote.audio не установлен — определяю говорящих без токена")
        else:
            try:
                res = _diarize_pyannote(audio_16k_wav, lines, hf_token, on_progress)
                if on_progress:
                    on_progress(1.0)
                return res
            except Exception as e:  # noqa: BLE001
                log.warning("pyannote не сработал (%s) — перехожу на запасной способ", e)
    try:
        res = _diarize_embeddings(audio_16k_wav, lines, on_progress, cluster_threshold)
    except EngineError:
        raise
    except ImportError as e:        # лёгкий движок: torch/speechbrain не установлены
        raise MlMissingError(debug=f"{type(e).__name__}: {e}") from e
    except Exception as e:
        raise EngineError(f"Не удалось определить говорящих: {e}") from e
    if on_progress:
        on_progress(1.0)
    return res
