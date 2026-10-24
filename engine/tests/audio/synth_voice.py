"""Синтетическая «речь» для тестов: слоговые всплески гармоник с шумом. Без нейросетей и файлов."""
from __future__ import annotations

import numpy as np

SR = 48000


def speech_burst(dur: float, *, seed: int = 0, f0: float = 140.0, sr: int = SR) -> np.ndarray:
    """Одна фраза длиной dur с.: слоги по 120–220 мс, между ними короткие провалы (40–80 мс)."""
    rng = np.random.default_rng(seed)
    n = int(dur * sr)
    out = np.zeros(n, dtype=np.float64)
    t = 0.0
    k = 0
    while t < dur - 0.05:
        L = float(rng.uniform(0.12, 0.22))
        L = min(L, dur - t)
        i0, i1 = int(t * sr), int((t + L) * sr)
        tt = np.arange(i1 - i0) / sr
        env = np.sin(np.pi * np.arange(i1 - i0) / max(i1 - i0, 1)) ** 2 * rng.uniform(0.5, 1.0)
        f = f0 * (1 + 0.1 * np.sin(2 * np.pi * 3 * tt + k))
        ph = 2 * np.pi * np.cumsum(f) / sr
        sig = sum(np.sin(h * ph) / h for h in range(1, 12))
        sig = sig + 0.15 * rng.standard_normal(len(tt))
        out[i0:i1] += sig * env
        t += L + float(rng.uniform(0.04, 0.08))
        k += 1
    out /= max(np.abs(out).max(), 1e-9)
    return (0.5 * out).astype(np.float32)


def voice_track(segments: list[tuple[float, float]], total: float, *, seed: int = 0, level: float = 1.0,
                f0: float = 140.0, sr: int = SR) -> np.ndarray:
    """Дорожка длиной total с «речью» в заданных интервалах и точной тишиной вне их."""
    x = np.zeros(int(total * sr), dtype=np.float32)
    for i, (a, b) in enumerate(segments):
        s = speech_burst(b - a, seed=seed + i, f0=f0, sr=sr)
        i0 = int(a * sr)
        x[i0:i0 + len(s)] += s[: len(x) - i0] * level
    return x


def stretch_signal(x: np.ndarray, factor: float) -> np.ndarray:
    """Грубое растяжение длины в factor раз (с изменением тона — для детекторов энергии неважно)."""
    n_out = int(len(x) * factor)
    return np.interp(np.arange(n_out) / factor, np.arange(len(x)), x).astype(np.float32)
