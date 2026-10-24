"""Мастеринг: замер громкости и двухпроходная нормализация loudnorm."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

from ..util import EngineError, run_ffmpeg
from .io import read_wav, write_wav


@dataclass
class Loudness:
    lufs: float
    true_peak_db: float
    lra: float


def true_peak_db(x: np.ndarray) -> float:
    """True peak (dBTP): оверсэмплинг 4x."""
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0:
        return -120.0
    if x.ndim == 1:
        x = x[:, None]
    peak = 0.0
    for c in range(x.shape[1]):
        peak = max(peak, float(np.abs(resample_poly(x[:, c], 4, 1)).max()))
    return float(20 * np.log10(max(peak, 1e-9)))


_tp_of = true_peak_db        # у master() параметр с тем же именем


def _lra(x: np.ndarray, sr: int, meter) -> float:
    """Диапазон громкости (EBU 3342): short-term 3 с, шаг 1 с, гейты −70 / −20 LU, процентили 10–95."""
    win, hop = 3 * sr, sr
    vals = []
    for s in range(0, max(len(x) - win, 0) + 1, hop):
        seg = x[s: s + win]
        if len(seg) < win:
            break
        try:
            v = meter.integrated_loudness(seg)
        except (ValueError, FloatingPointError):      # окно тише порога pyloudnorm и т. п.
            continue
        if np.isfinite(v):
            vals.append(v)
    vals = np.array([v for v in vals if v > -70.0])
    if len(vals) < 2:
        return 0.0
    rel = 10 * np.log10(np.mean(10 ** (vals / 10))) - 20.0
    vals = vals[vals > rel]
    if len(vals) < 2:
        return 0.0
    return float(np.percentile(vals, 95) - np.percentile(vals, 10))


def measure(x: np.ndarray, sr: int) -> Loudness:
    """Integrated LUFS (BS.1770 через pyloudnorm), true peak, LRA."""
    import pyloudnorm as pyln

    x = np.asarray(x, dtype=np.float32)
    meter = pyln.Meter(sr)
    if len(x) < int(0.4 * sr):  # короче одного блока — pyloudnorm не умеет
        pad = np.zeros((int(0.4 * sr) - len(x),) + x.shape[1:], dtype=np.float32)
        x_l = np.concatenate([x, pad])
    else:
        x_l = x
    lufs = float(meter.integrated_loudness(x_l))
    if not np.isfinite(lufs):
        lufs = -70.0
    return Loudness(lufs=round(lufs, 2), true_peak_db=round(true_peak_db(x), 2), lra=round(_lra(x_l, sr, meter), 2))


def _loudnorm_stats(stderr: str) -> dict:
    m = re.findall(r"\{[^{}]*\}", stderr, flags=re.DOTALL)
    if not m:
        raise EngineError("Не удалось измерить громкость (loudnorm не вернул данные).")
    return json.loads(m[-1])


TP_HEADROOM_DB = 0.15      # запас под true peak после мастеринга: AAC добавляет ещё до ~0.3 дБ (проверяет render_video)
_CHUNK = 1 << 19
_PAD = 512


def _truepeak_envelope(x: np.ndarray, oversample: int = 4) -> np.ndarray:
    """Огибающая true peak на частоте файла: max по каналам |x| после оверсэмплинга ×4, свёрнутая обратно (по 4 под-отсчёта
    + соседи, чтобы пик между отсчётами попал в оба соседних). Считается кусками — длинная сцена не раздувает память."""
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        x = x[:, None]
    n = len(x)
    env = np.zeros(n, dtype=np.float32)
    for a in range(0, n, _CHUNK):
        lo, hi = max(a - _PAD, 0), min(a + _CHUNK + _PAD, n)
        seg = x[lo:hi]
        up = np.zeros((len(seg) * oversample,), dtype=np.float32)
        for c in range(seg.shape[1]):
            u = np.abs(resample_poly(seg[:, c], oversample, 1)).astype(np.float32)
            np.maximum(up, u[: len(up)], out=up)
        e = up.reshape(len(seg), oversample).max(axis=1)
        e = np.maximum(e, np.maximum(np.roll(e, 1), np.roll(e, -1)))
        env[a:min(a + _CHUNK, n)] = e[a - lo: a - lo + min(_CHUNK, n - a)]
    return env


def limit_true_peak(x: np.ndarray, sr: int, ceiling_db: float, *, lookahead_ms: float = 4.0,
                    release_ms: float = 60.0) -> np.ndarray:
    """True-peak лимитер: гейн по огибающей 4×-оверсэмплированного сигнала, окно заглядывания вперёд (минимум + скользящее
    среднее — гарантирует, что в каждой точке гейн не выше нужного), мягкий возврат. После обработки меряем true peak и,
    если он всё же выше потолка, снижаем потолок и повторяем (до 6 раз)."""
    from scipy.ndimage import minimum_filter1d, uniform_filter1d
    from scipy.signal import lfilter

    x = np.asarray(x, dtype=np.float32)
    if x.size == 0:
        return x
    la = max(int(lookahead_ms / 1000 * sr), 8)
    rel = float(np.exp(-1.0 / (release_ms / 1000 * sr)))
    env = _truepeak_envelope(x)
    ceil_db = ceiling_db
    y = x
    for _ in range(6):
        c = 10 ** (ceil_db / 20)
        need = np.minimum(1.0, c / np.maximum(env, 1e-9)).astype(np.float64)
        if need.min() >= 1.0:
            return x
        g = minimum_filter1d(need, size=2 * la + 1, mode="nearest")
        g = uniform_filter1d(g, size=2 * la + 1, mode="nearest")
        depth = 1.0 - g
        tail = lfilter([1.0 - rel], [1.0, -rel], depth)        # мягкий хвост после пика (релиз)
        g = 1.0 - np.maximum(depth, tail)
        y = (x * (g.astype(np.float32) if x.ndim == 1 else g.astype(np.float32)[:, None])).astype(np.float32)
        tp = true_peak_db(y)
        if tp <= ceiling_db + 0.02:
            return y
        ceil_db -= max(tp - ceiling_db, 0.05)
    return y


def _master_loudnorm(in_wav, out_wav, *, target_lufs: float, true_peak_db: float) -> Loudness:
    """Двухпроходный loudnorm (запас по пику) → доводка гейном → true-peak лимитер (4× оверсэмплинг).

    Итог: LUFS в пределах ±0.5 LU от цели (если пик позволяет), true peak не выше true_peak_db − запас
    (запас нужен AAC: после кодирования пик чуть вырастает; render_video перепроверяет)."""
    out = Path(out_wav)
    out.parent.mkdir(parents=True, exist_ok=True)
    x, sr = read_wav(in_wav)
    ceiling = true_peak_db - TP_HEADROOM_DB
    tp_ln = min(0.0, max(-9.0, ceiling - 0.5))        # loudnorm сам целится ниже: лимитеру остаётся мало работы
    base = f"I={target_lufs}:TP={tp_ln}:LRA=20"
    p1 = run_ffmpeg(["-loglevel", "info", "-i", in_wav, "-af", f"loudnorm={base}:print_format=json",
                     "-f", "null", "-"])
    st = _loudnorm_stats(p1.stderr)
    if not np.isfinite(float(st["input_i"])):          # тишина: нормализовать нечего
        write_wav(out, x, sr)
        return measure(x, sr)
    af = (f"loudnorm={base}:measured_I={st['input_i']}:measured_TP={st['input_tp']}:"
          f"measured_LRA={st['input_lra']}:measured_thresh={st['input_thresh']}:"
          f"offset={st['target_offset']}:linear=true:print_format=summary,aresample={sr}")
    run_ffmpeg(["-i", in_wav, "-af", af, "-c:a", "pcm_f32le", out])
    y, sr_y = read_wav(out)
    z = y
    for _ in range(4):
        res = measure(z, sr_y)
        if abs(res.lufs - target_lufs) > 0.3:          # loudnorm мог уйти в динамический режим / лимитер съел громкость
            z = (z * np.float32(10 ** ((target_lufs - res.lufs) / 20))).astype(np.float32)
        z = limit_true_peak(z, sr_y, ceiling)
        res = measure(z, sr_y)
        if abs(res.lufs - target_lufs) <= 0.3:
            break
    write_wav(out, z, sr_y)
    return measure(z, sr_y)


MAX_LIMIT_DB = 9.0         # лимитер ловит пики не глубже, чем на столько дБ; если для цели нужно больше — старый loudnorm


def master(in_wav, out_wav, *, target_lufs: float = -14.0, true_peak_db: float = -1.0) -> Loudness:
    """Мастеринг: ОДИН постоянный гейн до целевой громкости → true-peak лимитер (4× оверсэмплинг).

    Раньше был двухпроходный ffmpeg loudnorm. Но когда нужный гейн упирается в пик, loudnorm молча переходит в
    «динамический» режим и гоняет усиление по ходу ролика: тихие места (фон, гул, шум комнаты) он поднимал на +17 дБ,
    а в речи только на +9, — отсюда «качающийся» фон и жужжание, которое то нарастает, то пропадает (замеры на сцене
    «Улыбаемся и машем»). Постоянный гейн сохраняет соотношение громкости фона и голоса как в сведении.

    Итог: LUFS в пределах ±0.5 LU от цели (если пик позволяет), true peak не выше true_peak_db − запас
    (запас нужен AAC: после кодирования пик чуть вырастает; render_video перепроверяет)."""
    out = Path(out_wav)
    out.parent.mkdir(parents=True, exist_ok=True)
    x, sr = read_wav(in_wav)
    ceiling = true_peak_db - TP_HEADROOM_DB
    cur = measure(x, sr)
    if cur.lufs <= -69.0 or not np.isfinite(cur.lufs):          # тишина: нормализовать нечего
        write_wav(out, x, sr)
        return measure(x, sr)
    tp_in = _tp_of(x)
    gain_db = target_lufs - cur.lufs
    if gain_db > ceiling + MAX_LIMIT_DB - tp_in:                 # пики редкие и острые: постоянным гейном цель не взять
        return _master_loudnorm(in_wav, out_wav, target_lufs=target_lufs, true_peak_db=true_peak_db)
    z = x
    for _ in range(4):
        z = limit_true_peak((x * np.float32(10 ** (gain_db / 20))).astype(np.float32), sr, ceiling)
        res = measure(z, sr)
        if abs(res.lufs - target_lufs) <= 0.3:
            break
        gain_db += target_lufs - res.lufs                        # лимитер съел часть громкости — добавляем
    write_wav(out, z, sr)
    return measure(z, sr)


# ─────────────────────────── выбор громкости ролика ───────────────────────────
ONLINE_LUFS = -14.0            # «для YouTube/VK»
# «Как в фильме» = громкость оригинала, но не тише −24 LUFS (уровень ТВ-вещания, EBU R128 −23 / ATSC −24): у роликов
# с YouTube оригинал бывает −30 LUFS, и ролик выходил почти неслышным (на 15 дБ тише превью «для YouTube»).
FILM_LUFS_RANGE = (-24.0, -10.0)


def film_target_lufs(original_wav) -> float | None:
    """Громкость оригинала сцены (integrated LUFS) — цель режима «как в фильме». None — измерить не удалось."""
    try:
        x, sr = read_wav(original_wav)
        lufs = measure(x, sr).lufs
    except Exception:  # noqa: BLE001 — нет файла/слишком короткий и т. п.
        return None
    if not np.isfinite(lufs) or lufs <= -69.0:
        return None
    return float(np.clip(lufs, *FILM_LUFS_RANGE))


def resolve_target_lufs(loudness_target: str, requested_lufs: float, original_wav=None) -> float:
    """Цель мастеринга для выбора громкости ролика.

    · «online» (для YouTube/VK) — requested_lufs (обычно −14, можно подкрутить в настройках рендера);
    · «film» (как в фильме) — громкость оригинала сцены (original_wav); потом мастер лишь ловит пики лимитером.
      Нет оригинала — запасной вариант requested_lufs.
    Вызывать там, где пайплайн выбирает target_lufs для master_for_aac."""
    if loudness_target == "film" and original_wav is not None:
        t = film_target_lufs(original_wav)
        if t is not None:
            return t
    return float(requested_lufs)
