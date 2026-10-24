"""Эффекты голоса (ffmpeg-цепочки). Без rubberband: тон меняем через asetrate + aresample + atempo."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..util import EngineError, run_ffmpeg
from .io import read_wav, write_wav

_SR = 48000


def _pitch(ratio: float) -> str:
    """Смена тона в ratio раз при той же длине."""
    return f"asetrate={int(_SR * ratio)},aresample={_SR},atempo={1 / ratio:.6f}"


VOICE_FX: dict[str, str] = {
    "none": "anull",
    "robot": ("highpass=f=150,aecho=0.8:0.88:6:0.6,"
              "acrusher=level_in=1:level_out=1:bits=9:mode=log:aa=1:mix=0.5,"
              "equalizer=f=1200:t=q:w=1.5:g=4"),
    "chipmunk": _pitch(1.4),
    "giant": _pitch(0.75) + ",lowpass=f=4500,equalizer=f=120:t=q:w=1:g=4",
    "villain": (_pitch(0.88) + ",aecho=0.8:0.6:45|80:0.22|0.12,"
                "equalizer=f=140:t=q:w=1:g=3,lowpass=f=7000"),
    "radio": ("highpass=f=350,lowpass=f=3200,"
              "acompressor=threshold=0.05:ratio=6:attack=5:release=60:makeup=3,"
              "equalizer=f=1800:t=q:w=1:g=4"),
    "phone": "highpass=f=400,lowpass=f=3000,acrusher=bits=12:mode=lin:mix=0.3,volume=1.5",
    "ghost": ("highpass=f=200,vibrato=f=5:d=0.25,aecho=0.8:0.9:70|140|220:0.5|0.35|0.2,"
              "lowpass=f=6000,tremolo=f=0.7:d=0.3"),
}

VOICE_FX_TITLES: dict[str, str] = {
    "none": "Без эффекта",
    "robot": "Робот",
    "chipmunk": "Бурундук",
    "giant": "Великан",
    "villain": "Злодей",
    "radio": "Рация",
    "phone": "Телефон",
    "ghost": "Призрак",
}


def apply_fx(in_wav, preset: str, out_wav) -> Path:
    """Применить эффект. Длина результата = длине входа (хвост эха обрезается, нехватка добивается тишиной)."""
    if preset not in VOICE_FX:
        raise EngineError(f"Неизвестный эффект голоса: {preset}")
    x, sr = read_wav(in_wav)
    n = len(x)
    if preset == "none" or n == 0:
        return write_wav(out_wav, x, sr)
    out = Path(out_wav)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.stem + ".fx_tmp.wav")
    try:
        af = f"aresample={_SR},{VOICE_FX[preset]}"
        run_ffmpeg(["-i", in_wav, "-af", af, "-ac", "1", "-c:a", "pcm_f32le", tmp])
        y, sr_y = read_wav(tmp)
    finally:
        tmp.unlink(missing_ok=True)
    n_out = round(n * sr_y / sr)
    y = y[:n_out] if len(y) >= n_out else np.concatenate([y, np.zeros(n_out - len(y), dtype=np.float32)])
    return write_wav(out, y, sr_y)
