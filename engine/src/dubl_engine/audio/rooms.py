"""Комнаты: синтетическая импульсная характеристика (затухающий шум + ранние отражения) и EQ."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy import signal

from ..util import EngineError
from .io import read_wav, to_mono


@dataclass(frozen=True)
class RoomPreset:
    rt60: float                        # время реверберации, с (0 — без хвоста)
    predelay_ms: float = 0.0
    wet: float = 0.0                   # доля реверберации 0..1
    early_reflections: tuple[tuple[float, float], ...] = ()   # (мс, усиление)
    lowpass_hz: float | None = None    # EQ на весь сигнал (комната «съедает» верха)
    highpass_hz: float | None = None
    hf_damping: float = 0.5            # во сколько раз быстрее затухают верхи (rt60 верхов = rt60 * hf_damping)
    dry_gain_db: float = 0.0           # изменение уровня прямого звука
    title: str = ""


ROOMS: dict[str, RoomPreset] = {
    "dry": RoomPreset(rt60=0.0, wet=0.0),
    "room": RoomPreset(rt60=0.35, predelay_ms=8, wet=0.22,
                       early_reflections=((11, 0.35), (19, 0.28), (27, 0.2)), hf_damping=0.55),
    "kitchen": RoomPreset(rt60=0.55, predelay_ms=5, wet=0.3,
                          early_reflections=((7, 0.5), (13, 0.4), (21, 0.3), (31, 0.22)), hf_damping=0.85),
    "hall": RoomPreset(rt60=1.6, predelay_ms=25, wet=0.35,
                       early_reflections=((24, 0.3), (41, 0.25), (63, 0.2)), hf_damping=0.5),
    "street": RoomPreset(rt60=0.6, predelay_ms=40, wet=0.12,
                         early_reflections=((55, 0.25), (110, 0.15)), hf_damping=0.4, highpass_hz=100),
    "cave": RoomPreset(rt60=3.2, predelay_ms=45, wet=0.5,
                       early_reflections=((60, 0.3), (130, 0.25), (210, 0.2)), hf_damping=0.35),
    "car": RoomPreset(rt60=0.12, predelay_ms=1, wet=0.25,
                      early_reflections=((2, 0.5), (4, 0.4), (6, 0.3)), hf_damping=0.4, lowpass_hz=5000,
                      highpass_hz=90),
    "bathroom": RoomPreset(rt60=0.9, predelay_ms=6, wet=0.4,
                           early_reflections=((5, 0.55), (9, 0.45), (14, 0.4), (23, 0.3)), hf_damping=1.0),
    "phone": RoomPreset(rt60=0.0, wet=0.0, lowpass_hz=3400, highpass_hz=300),
    "behind_door": RoomPreset(rt60=0.3, predelay_ms=10, wet=0.25, early_reflections=((12, 0.3), (22, 0.2)),
                              lowpass_hz=1400, highpass_hz=120, dry_gain_db=-7.0),
}

ROOM_TITLES: dict[str, str] = {
    "dry": "Без комнаты",
    "room": "Комната",
    "kitchen": "Кухня",
    "hall": "Зал",
    "street": "Улица",
    "cave": "Пещера",
    "car": "Машина",
    "bathroom": "Ванная",
    "phone": "Телефон",
    "behind_door": "За дверью",
}

# пресеты, между которыми выбирает guess_room (по времени реверберации)
_GUESS_CANDIDATES = ("dry", "room", "kitchen", "bathroom", "hall", "cave")
_GUESS_RT60 = {"dry": 0.12, "room": 0.35, "kitchen": 0.55, "bathroom": 0.9, "hall": 1.6, "cave": 3.0}


def _lp(x: np.ndarray, sr: int, hz: float, order: int = 2) -> np.ndarray:
    sos = signal.butter(order, min(hz, sr * 0.45), btype="low", fs=sr, output="sos")
    return signal.sosfilt(sos, x)


@lru_cache(maxsize=64)
def make_ir(preset: str, sr: int) -> np.ndarray:
    """Нормированная (по энергии хвоста = 1) синтетическая ИХ без прямого звука. Детерминирована."""
    p = ROOMS[preset]
    if p.rt60 <= 0 and not p.early_reflections:
        return np.zeros(1)
    rng = np.random.default_rng(sum(map(ord, preset)))
    dur = max(p.rt60 * 1.1, 0.05) + p.predelay_ms / 1000
    n = int(dur * sr)
    t = np.arange(n) / sr - p.predelay_ms / 1000
    noise = rng.standard_normal(n)
    tt = np.clip(t, 0, None)
    lf = _lp(noise, sr, 1500)
    hf = noise - lf
    rt = max(p.rt60, 0.02)
    env_lf = np.exp(-6.9078 * tt / rt)
    env_hf = np.exp(-6.9078 * tt / (rt * max(p.hf_damping, 0.05)))
    tail = lf * env_lf + 0.7 * hf * env_hf
    tail[t < 0] = 0.0
    # плавное нарастание, чтобы не щёлкало
    tail *= np.clip(tt / 0.004, 0, 1)
    e = np.sqrt(np.sum(tail ** 2))
    tail = tail / e if e > 0 else tail
    ir = tail.copy()
    for ms, g in p.early_reflections:
        i = int((ms + 0.0) / 1000 * sr)
        if i < n:
            ir[i] += g * 0.5 * (1.0 if rng.random() > 0.3 else -1.0)
    return ir


def _eq(x: np.ndarray, sr: int, p: RoomPreset) -> np.ndarray:
    if p.highpass_hz:
        x = signal.sosfilt(signal.butter(2, p.highpass_hz, btype="high", fs=sr, output="sos"), x)
    if p.lowpass_hz:
        x = signal.sosfilt(signal.butter(2, min(p.lowpass_hz, sr * 0.45), btype="low", fs=sr, output="sos"), x)
    return x


def apply_room(x: np.ndarray, sr: int, preset: str) -> np.ndarray:
    """Положить голос в комнату. Длина результата = длине входа (хвост за краем обрезается —
    если он нужен, добавьте тишину в конец перед вызовом). Громкость (RMS) сохраняется."""
    if preset not in ROOMS:
        raise EngineError(f"Неизвестная комната: {preset}")
    x = np.asarray(x, dtype=np.float32)
    mono = x.ndim == 1
    if not mono:
        return np.stack([apply_room(x[:, c], sr, preset) for c in range(x.shape[1])], axis=1)
    if preset == "dry" or len(x) == 0:
        return x.copy()
    p = ROOMS[preset]
    xin = x.astype(np.float64)
    y = xin
    ir = make_ir(preset, sr)
    if len(ir) > 1 and p.wet > 0:
        wet_sig = signal.fftconvolve(xin, ir)[: len(xin)]
        y = (1 - p.wet) * xin + p.wet * wet_sig * 1.5
    y = _eq(y, sr, p)
    r_in = np.sqrt(np.mean(xin ** 2) + 1e-12)
    r_out = np.sqrt(np.mean(y ** 2) + 1e-12)
    y = y * (r_in / r_out) * 10 ** (p.dry_gain_db / 20)
    return y.astype(np.float32)


def estimate_rt60(x: np.ndarray, sr: int) -> float | None:
    """Грубая оценка RT60 по спадам огибающей: берём участки быстрого убывания энергии (после всплесков),
    считаем наклон и приводим к -60 дБ. None — нет подходящих спадов."""
    x = np.asarray(x, dtype=np.float32)
    if x.ndim > 1:
        x = to_mono(x)
    hop = int(sr * 0.01)
    k = len(x) // hop
    if k < 30:
        return None
    env = 10 * np.log10(np.mean(x[: k * hop].reshape(k, hop).astype(np.float64) ** 2, axis=1) + 1e-12)
    env = np.convolve(env, np.ones(3) / 3, mode="same")
    slopes: list[float] = []
    i = 2
    while i < k - 8:
        # вершина: заметно выше последующих кадров
        if env[i] > env[i - 1] - 0.1 and env[i] >= env[i + 1] and env[i] > env.max() - 40:
            j = i
            while j + 1 < k and env[j + 1] < env[j] + 0.5 and env[i] - env[j + 1] < 25:
                j += 1
            drop = env[i] - env[j]
            L = (j - i) * 0.01
            if drop >= 10 and L >= 0.08:
                # наклон по регрессии на участке
                seg = env[i + 1: j + 1]
                tt = (np.arange(len(seg)) + 1) * 0.01
                s = np.polyfit(tt, seg, 1)[0]
                if s < -3:
                    slopes.append(-60.0 / s)
            i = max(j, i + 1)
        else:
            i += 1
    if len(slopes) < 2:
        return None
    return float(np.median(slopes))


def guess_room(background_wav) -> str:
    """Угадать комнату по хвостам реверберации фона. Нет уверенных спадов — 'room'."""
    try:
        x, sr = read_wav(Path(background_wav))
    except (OSError, RuntimeError, ValueError):       # нет файла / не WAV — комната по умолчанию
        return "room"
    rt = estimate_rt60(to_mono(x), sr)
    if rt is None:
        return "room"
    return min(_GUESS_CANDIDATES, key=lambda k: abs(np.log(max(rt, 0.03)) - np.log(_GUESS_RT60[k])))
