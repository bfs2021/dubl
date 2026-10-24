"""Чтение/запись WAV. Внутри движка аудио — float32, 48 кГц."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

SR = 48000


def read_wav(path: str | Path) -> tuple[np.ndarray, int]:
    """float32, shape (n,) для mono или (n, ch)."""
    x, sr = sf.read(str(path), dtype="float32", always_2d=False)
    return x, sr


def write_wav(path: str | Path, data: np.ndarray, sr: int = SR, *, subtype: str = "FLOAT") -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(p), np.asarray(data, dtype=np.float32), sr, subtype=subtype)
    return p


def to_mono(x: np.ndarray) -> np.ndarray:
    return x if x.ndim == 1 else x.mean(axis=1).astype(np.float32)


def to_stereo(x: np.ndarray) -> np.ndarray:
    if x.ndim == 1:
        return np.stack([x, x], axis=1)
    if x.shape[1] == 2:
        return x
    return np.stack([x[:, 0], x[:, 1 % x.shape[1]]], axis=1)


def db_to_gain(db: float) -> float:
    return float(10 ** (db / 20))


def gain_to_db(g: float) -> float:
    return float(20 * np.log10(max(g, 1e-12)))
