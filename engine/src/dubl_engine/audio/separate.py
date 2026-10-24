"""Разделение голосов и фона; нарезка голосов по персонажам.

best: BS-RoFormer (пакет audio-separator); fast: demucs htdemucs (только вокал).
Выход: voices.wav (mono) и background.wav (stereo), 48 кГц float32, длина ровно как у входа.
"""
from __future__ import annotations

import logging
import shutil
import tempfile
from collections.abc import Callable
from math import gcd
from pathlib import Path
from typing import Literal

import numpy as np

from ..devices import device_candidates, mark_bad
from ..models import SceneLine
from ..util import EngineError, MlMissingError, data_dir
from .io import SR, read_wav, to_mono, to_stereo, write_wav

log = logging.getLogger(__name__)

BEST_MODEL = "model_bs_roformer_ep_317_sdr_12.9755.ckpt"
DEMUCS_MODEL = "htdemucs"
DEMUCS_SR = 44100

Progress = Callable[[float], None] | None


def separator_dir() -> Path:
    d = data_dir() / "models" / "separator"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ───────────────────────── чистая обработка сигналов ─────────────────────────
def _fit_length(x: np.ndarray, n: int) -> np.ndarray:
    if len(x) >= n:
        return x[:n]
    pad = np.zeros((n - len(x),) + x.shape[1:], dtype=x.dtype)
    return np.concatenate([x, pad])


def _resample(x: np.ndarray, sr_from: int, sr_to: int) -> np.ndarray:
    """x: (n,) или (n, ch)."""
    if sr_from == sr_to:
        return x.astype(np.float32, copy=False)
    from scipy.signal import resample_poly
    g = gcd(sr_from, sr_to)
    return resample_poly(x, sr_to // g, sr_from // g, axis=0).astype(np.float32)


def _spectral_clamp(x: np.ndarray, r: np.ndarray, nfft: int) -> np.ndarray:
    """В каждом бине спектра остаток r не громче исходного x (нет «антиголоса»)."""
    from scipy.signal import istft, stft
    _, _, X = stft(x, nperseg=nfft)
    _, _, R = stft(r, nperseg=nfft)
    scale = np.minimum(1.0, np.abs(X) / np.maximum(np.abs(R), 1e-12))
    _, r2 = istft(R * scale, nperseg=nfft)
    return _fit_length(r2, len(r))


def subtract_voices(mix: np.ndarray, voices: np.ndarray, *, nfft: int = 2048, block_s: float = 30.0,
                    sr: int = SR) -> np.ndarray:
    """Фон = стерео-микс минус голоса, с подбором уровня и без «отрицательных» артефактов.

    mix: (n, 2) микс (в 5.1-сведении центр лежит в обоих каналах примерно с −3 дБ), voices: (n,) голос.
    1) Для каждого канала подбираем усиление g (МНК, g ∈ [0, 1.5]), с которым голос лежит в миксе.
    2) Вычитаем во времени, затем в спектре не даём остатку быть громче исходного микса ни в одном бине.
    Спектральный шаг идёт блоками по `block_s` с перекрытием и плавной склейкой — память не растёт с длиной сцены
    (20 минут целиком в float64 + STFT съели бы несколько гигабайт).
    """
    mix = to_stereo(mix).astype(np.float32)
    v = voices.astype(np.float32)
    n = len(mix)
    vv = float(np.dot(v.astype(np.float64), v.astype(np.float64)))
    out = np.zeros_like(mix)
    blk = max(int(block_s * sr), nfft * 8)
    ov = nfft * 4
    for ch in range(mix.shape[1]):
        x = mix[:, ch]
        g = float(np.dot(x.astype(np.float64), v.astype(np.float64)) / vv) if vv > 1e-12 else 0.0
        g = min(max(g, 0.0), 1.5)
        r = x - np.float32(g) * v
        if g <= 0 or n <= nfft:
            out[:, ch] = r
            continue
        acc = np.zeros(n, dtype=np.float32)
        wsum = np.zeros(n, dtype=np.float32)
        for a in range(0, n, blk):
            lo, hi = max(0, a - ov), min(n, a + blk + ov)
            seg = _spectral_clamp(x[lo:hi].astype(np.float64), r[lo:hi].astype(np.float64), nfft).astype(np.float32)
            w = np.ones(hi - lo, dtype=np.float32)
            if lo > 0:
                w[: 2 * ov] = np.linspace(0.0, 1.0, 2 * ov, dtype=np.float32)[: len(w)]
            if hi < n:
                w[-2 * ov:] = np.minimum(w[-2 * ov:], np.linspace(1.0, 0.0, 2 * ov, dtype=np.float32)[-len(w):])
            acc[lo:hi] += seg * w
            wsum[lo:hi] += w
        out[:, ch] = acc / np.maximum(wsum, 1e-6)
    return out


def span_mask(n: int, sr: int, spans, *, fade: float = 0.03, margin: float = 0.1) -> np.ndarray:
    """Маска 0..1 длиной n: единица внутри отрезков (с запасом `margin` по краям), плавные края `fade`."""
    mask = np.zeros(n, dtype=np.float32)
    f = max(1, round(fade * sr))
    for a, b in spans:
        i0 = max(0, round((a - margin) * sr))
        i1 = min(n, round((b + margin) * sr))
        if i1 <= i0:
            continue
        seg = np.ones(i1 - i0, dtype=np.float32)
        k = min(f, (i1 - i0) // 2)
        if k > 0:
            ramp = np.linspace(0.0, 1.0, k, dtype=np.float32)
            seg[:k] = ramp
            seg[-k:] = ramp[::-1]
        mask[i0:i1] = np.maximum(mask[i0:i1], seg)
    return mask


def voice_extents(voices: np.ndarray, sr: int, spans: list[tuple[float, float]], *, max_ext: float = 1.2,
                  gap: float = 0.2) -> list[tuple[float, float]]:
    """Расширить отрезки реплик до реальных границ речи в стеме голосов.

    Тайминги субтитров и распознавания часто начинаются/кончаются на 0.2–0.8 с раньше, чем сама фраза
    («Action!» звучит после конца строки «...and, action!»). Без расширения хвост оригинальной реплики остаётся
    в миксе под дублем. Отрезок растёт, пока голос звучит непрерывно (паузы до `gap`), не дальше `max_ext` секунд
    в каждую сторону и не заходя на соседние реплики. Порядок и число отрезков сохраняются."""
    if not spans:
        return []
    x = voices.mean(axis=1) if voices.ndim == 2 else voices
    hop = max(1, sr // 50)                                       # 20 мс
    k = len(x) // hop
    if k < 3:
        return list(spans)
    db = 20 * np.log10(np.sqrt(np.mean(x[: k * hop].astype(np.float64).reshape(k, hop) ** 2, axis=1)) + 1e-9)
    loud = db[db > -80.0]
    if not loud.size:
        return list(spans)
    active = db > max(np.percentile(loud, 98) - 30.0, -60.0)
    max_f, gap_f = int(max_ext * 50), max(1, int(gap * 50))
    order = sorted(range(len(spans)), key=lambda i: spans[i][0])
    out = list(spans)
    for pos, i in enumerate(order):
        a, b = spans[i]
        lo = spans[order[pos - 1]][1] if pos > 0 else 0.0          # не заходим на соседей
        hi = spans[order[pos + 1]][0] if pos + 1 < len(order) else k / 50
        fa, fb = int(a * 50), int(b * 50)
        # назад
        j, last, miss = fa - 1, fa, 0
        while j >= max(0, fa - max_f, int(lo * 50)):
            if active[j]:
                last, miss = j, 0
            else:
                miss += 1
                if miss > gap_f:
                    break
            j -= 1
        # вперёд
        j, end, miss = fb, fb, 0
        while j < min(k, fb + max_f, int(hi * 50)):
            if active[j]:
                end, miss = j + 1, 0
            else:
                miss += 1
                if miss > gap_f:
                    break
            j += 1
        out[i] = (min(a, max(lo, last / 50)), max(b, min(hi, end / 50)))
    # Соседи расширились навстречу друг другу и перекрылись (пауза между репликами короче двух хвостов): общий кусок
    # делим посередине самой длинной паузы в нём (паузы между слогами короче паузы между фразами), без пауз — по
    # самому тихому месту. Иначе сведение не знает, чей это голос, и хвост одного персонажа уезжает
    # к соседу (проект «Мегамозг»: хвост «…Мегамозгу?» звучал оригиналом под неозвученной репликой Титана).
    for pos in range(len(order) - 1):
        i, j = order[pos], order[pos + 1]
        (a0, b0), (a1, b1) = out[i], out[j]
        if b0 <= a1:
            continue
        lo_f = max(int(round(spans[i][1] * 50)), int(round(a1 * 50)))      # не заходим внутрь своих реплик
        hi_f = min(int(round(spans[j][0] * 50)), int(round(b0 * 50)))
        if hi_f > lo_f:
            quiet = ~active[lo_f:hi_f]
            best_len, best_mid, run = 0, None, 0
            for q, v in enumerate(quiet):
                run = run + 1 if v else 0
                if run > best_len:
                    best_len, best_mid = run, q - run / 2 + 0.5
            mid = best_mid if best_mid is not None else float(np.argmin(db[lo_f:hi_f]))
            cut = (lo_f + mid) / 50
        else:
            cut = (spans[i][1] + spans[j][0]) / 2
        out[i] = (a0, max(spans[i][1], min(b0, cut)))
        out[j] = (min(spans[j][0], max(a1, cut)), b1)
    return out


def split_by_speaker(voices_wav, lines_rel: list[SceneLine], out_dir, *, fade: float = 0.03,
                     margin: float = 0.1) -> dict[str, Path]:
    """Для каждого speaker_id — копия voices.wav с тишиной вне его реплик (запас `margin` по краям, фейды `fade`).

    Реплики без speaker_id попадают в ключ 'unknown'. Файл: <out_dir>/<speaker_id>.wav.
    Сведение раскладывает голоса заново по расширенным границам реплик (audio.mix._voice_layers);
    эти файлы нужны для замеров громкости и прослушивания персонажа.
    Всё, что нейросеть отнесла к голосам вне реплик (крики, пение, толпа), сюда не попадает —
    сведение возвращает это в фон (см. audio.mix).
    """
    x, sr = read_wav(voices_wav)
    n = len(x)
    groups: dict[str, list[tuple[float, float]]] = {}
    for ln in lines_rel:
        groups.setdefault(ln.speaker_id or "unknown", []).append((ln.start_rel, ln.end_rel))
    out: dict[str, Path] = {}
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for spk, spans in groups.items():
        mask = span_mask(n, sr, spans, fade=fade, margin=margin)
        y = x * (mask[:, None] if x.ndim == 2 else mask)
        out[spk] = write_wav(out_dir / f"{_safe(spk)}.wav", y, sr)
    return out


def _safe(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in name) or "unknown"


# ───────────────────────── нейросети ─────────────────────────
# Каждая возвращает (vocals, other) на 48 кГц: shape (n, 2) или (n,) — как получилось, длина = входу.
def _net_demucs(x: np.ndarray, sr: int, device: str, on_progress: Progress) -> tuple[np.ndarray, np.ndarray]:
    import torch
    from demucs.apply import apply_model
    from demucs.pretrained import get_model

    torch.hub.set_dir(str(data_dir() / "models" / "demucs"))
    model = get_model(DEMUCS_MODEL)
    model.eval()
    xs = to_stereo(x) if x.ndim == 2 else np.stack([x, x], axis=1)
    n = len(xs)
    wav = torch.from_numpy(_resample(xs, sr, DEMUCS_SR).T.copy())  # (2, n')
    ref = wav.mean(0)
    mean, std = ref.mean(), ref.std() + 1e-8
    wav = (wav - mean) / std

    def cb(d: dict) -> None:
        if on_progress and d.get("state") == "end":
            seg = float(getattr(model.models[0], "segment", 7.8)) if hasattr(model, "models") else 7.8
            done = d["segment_offset"] + seg * model.samplerate
            on_progress(min(0.95, done / max(1, wav.shape[-1])))

    try:
        model.to(device)
        with torch.no_grad():
            out = apply_model(model, wav[None], device=device, shifts=1, split=True, overlap=0.25,
                              progress=False, callback=cb)[0]
    finally:
        model.to("cpu")
    out = out * std + mean
    vi = model.sources.index("vocals")
    voc = out[vi]
    other = sum(out[i] for i in range(len(model.sources)) if i != vi)
    f = lambda t: _fit_length(_resample(t.cpu().numpy().T, DEMUCS_SR, sr), n)
    return f(voc), f(other)


def _net_roformer(x: np.ndarray, sr: int, device: str, on_progress: Progress) -> tuple[np.ndarray, np.ndarray]:
    import torch
    from audio_separator.separator import Separator

    n = len(x)
    work = Path(tempfile.mkdtemp(prefix="dubl-sep-"))
    try:
        src = write_wav(work / "in.wav", x if x.ndim == 2 else np.stack([x, x], axis=1), sr)
        sep = Separator(output_dir=str(work), model_file_dir=str(separator_dir()), output_format="WAV",
                        log_level=logging.WARNING, sample_rate=sr,
                        mdxc_params={"segment_size": 256, "override_model_segment_size": False,
                                     "batch_size": 1, "overlap": 4, "pitch_shift": 0})
        # audio-separator сам выбирает устройство; подменим на выбранное нами
        sep.torch_device = torch.device(device)
        sep.torch_device_mps = torch.device("mps") if device == "mps" else None
        if device != "mps":
            sep.torch_device_mps = None
        if on_progress:
            on_progress(0.1)
        sep.load_model(BEST_MODEL)
        if on_progress:
            on_progress(0.3)
        files = sep.separate(str(src), {"Vocals": "vocals", "Instrumental": "other"})
        paths = {Path(f).name: (Path(f) if Path(f).is_absolute() else work / f) for f in files}
        pv = next((p for k, p in paths.items() if k.startswith("vocals")), None)
        po = next((p for k, p in paths.items() if k.startswith("other")), None)
        if pv is None or po is None or not pv.exists() or not po.exists():
            raise EngineError(f"BS-RoFormer не вернул ожидаемые дорожки: {list(paths)}")
        v, vsr = read_wav(pv)
        o, osr = read_wav(po)
        return (_fit_length(_resample(v, vsr, sr), n), _fit_length(_resample(o, osr, sr), n))
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _run_net(kind: str, x: np.ndarray, sr: int, device: str, on_progress: Progress):
    fn = _net_roformer if kind == "best" else _net_demucs
    last: Exception | None = None
    for cand in device_candidates(device, task=kind):
        # demucs не работает на MPS (ограничение каналов), roformer на MPS бывает нестабилен — при ошибке уходим на cpu
        if kind == "fast" and cand == "mps":
            cand = "cpu"
        try:
            return fn(x, sr, cand, on_progress)
        except ImportError as e:    # лёгкий движок: torch/demucs не установлены — переход на другое устройство не поможет
            raise MlMissingError(debug=f"{type(e).__name__}: {e}") from e
        except Exception as e:  # noqa: BLE001
            last = e
            log.warning("Разделение (%s) на %s не удалось: %s", kind, cand, e)
            mark_bad(kind, cand)
            if cand == "cpu":
                break
    raise EngineError(f"Не удалось разделить голоса и фон ({kind}): {last}")


# ───────────────────────── публичная функция ─────────────────────────
def separate(scene_audio_wav, out_dir, *, center_wav: Path | None = None,
             quality: Literal["fast", "best"] = "best", device: str = "auto",
             on_progress: Callable[[float], None] | None = None) -> tuple[Path, Path]:
    """Разделяет звук сцены на голоса (mono) и фон (stereo). Возвращает (voices.wav, background.wav)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mix, sr = read_wav(scene_audio_wav)
    if len(mix) == 0:
        raise EngineError("В сцене нет звука.")
    mix = to_stereo(mix) if mix.ndim == 2 else np.stack([mix, mix], axis=1)
    if sr != SR:
        mix = _resample(mix, sr, SR)
    n = len(mix)

    center = None
    if center_wav:
        c, csr = read_wav(center_wav)
        c = to_mono(c)
        center = _fit_length(_resample(c, csr, SR) if csr != SR else c, n)

    if on_progress:
        on_progress(0.0)
    sub = (lambda p: on_progress(0.05 + 0.85 * p)) if on_progress else None
    # Нейросеть всегда работает по ПОЛНОМУ миксу. Центральный канал 5.1 не используется как источник голоса:
    # во многих сведениях (напр. Tears of Steel) диалог разведён и в L/R, и вычитание «центра» из микса
    # оставляло оригинальную речь в фоне — её было слышно под дубляжом. center_wav оставлен для совместимости.
    if center is not None:
        log.info("Есть центральный канал 5.1, но голос разделяется по полному миксу (диалог бывает и в L/R).")
    net_in = mix

    kind = quality
    voc = other = np.zeros(0, dtype=np.float32)
    if kind == "best":
        try:
            voc, other = _run_net("best", net_in, SR, device, sub)
        except EngineError as e:
            log.warning("Точная модель (BS-RoFormer) не сработала, перехожу на быструю (demucs): %s", e)
            kind = "fast"
    if kind == "fast":
        voc, other = _run_net("fast", net_in, SR, device, sub)

    voices = _fit_length(to_mono(voc), n)
    background = _fit_length(to_stereo(other), n)

    vp = write_wav(out_dir / "voices.wav", voices, SR)
    bp = write_wav(out_dir / "background.wav", background, SR)
    if on_progress:
        on_progress(1.0)
    return vp, bp
