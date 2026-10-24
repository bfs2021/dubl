"""Динамика голоса: мягкий компрессор «Podcast/Radio» из Audacity (−15 дБ, 3:1, колено 24 дБ, 15/40 мс).

Порог задаём не в dBFS, а относительно громкости речи самого дубля (речь + 6 дБ = ровно та же точка, что −15 дБ у
Audacity при речи около −21 дБ): тихая запись с микрофона без усиления сжимается так же, как громкая. Makeup нет —
громкость потом ставит автопилот (по LUFS), поэтому компрессор не поднимает шум.
"""
from __future__ import annotations

import numpy as np

COMP_RATIO = 3.0
COMP_KNEE_DB = 24.0
COMP_ATTACK_MS = 15.0
COMP_RELEASE_MS = 40.0
COMP_THR_OVER_SPEECH_DB = 6.0        # порог = RMS речи + столько дБ
_HOP_MS = 1.0


def compress(x: np.ndarray, sr: int, *, threshold_db: float, ratio: float = COMP_RATIO, knee_db: float = COMP_KNEE_DB,
             attack_ms: float = COMP_ATTACK_MS, release_ms: float = COMP_RELEASE_MS) -> tuple[np.ndarray, float]:
    """Feed-forward компрессор с мягким коленом по пиковой огибающей. → (звук, максимальное сжатие в дБ)."""
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0:
        return x, 0.0
    hop = max(1, int(sr * _HOP_MS / 1000))
    k = len(x) // hop + 1
    pad = np.zeros(k * hop, dtype=np.float32)
    pad[: len(x)] = np.abs(x)
    env = pad.reshape(k, hop).max(axis=1).astype(np.float64)
    lvl = 20 * np.log10(env + 1e-9)
    over = lvl - threshold_db
    w = max(knee_db, 1e-6)
    slope = 1.0 - 1.0 / ratio
    gr = np.where(over <= -w / 2, 0.0, np.where(over >= w / 2, slope * over, slope * (over + w / 2) ** 2 / (2 * w)))
    # сглаживание: сжатие нарастает за attack, отпускается за release (в дБ, по кадрам 1 мс)
    ca = 1.0 - np.exp(-_HOP_MS / attack_ms)
    cr = 1.0 - np.exp(-_HOP_MS / release_ms)
    sm = np.empty_like(gr)
    cur = 0.0
    for i in range(k):
        c = ca if gr[i] > cur else cr
        cur += c * (gr[i] - cur)
        sm[i] = cur
    gain_db = -sm
    t = (np.arange(len(x)) / hop) - 0.5
    g = 10 ** (np.interp(t, np.arange(k), gain_db) / 20)
    return (x * g.astype(np.float32)).astype(np.float32), float(sm.max())


def voice_compress(x: np.ndarray, sr: int) -> tuple[np.ndarray, float]:
    """Компрессор речи с порогом относительно громкости самой речи (см. модуль). → (звук, макс. сжатие, дБ)."""
    from .levels import speech_rms_db

    sp = speech_rms_db(x, sr)
    if sp is None:
        return np.asarray(x, dtype=np.float32), 0.0
    return compress(x, sr, threshold_db=sp + COMP_THR_OVER_SPEECH_DB)
