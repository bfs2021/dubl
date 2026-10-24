"""Чистка голоса: срез низа → (де-эссер) (ffmpeg). Шумодав и выравнивание громкости — отдельно: audio.denoise, audio.dynamics.

Раньше здесь же стояли afftdn (шумодав) и acompressor с makeup +8 дБ. На записи Тимура (замеры в /tmp/dirty) afftdn
«булькал» — шум в паузах прыгал на ±15 дБ по ходу фразы, а компрессор с makeup поднимал шум микрофона на 7 дБ. Поэтому
по умолчанию они выключены; шумодав по снимку тишины — `denoise.denoise_voice`, выравнивание — `dynamics.voice_compress`.
"""
from __future__ import annotations

from pathlib import Path

from ..util import run_ffmpeg


def clean_filter(*, denoise: bool = False, deess: bool = True, compress: bool = False, highpass_hz: int = 80) -> str:
    """Цепочка фильтров ffmpeg. denoise/compress — старые режимы (afftdn, acompressor), оставлены для ручного включения."""
    chain: list[str] = []
    if highpass_hz and highpass_hz > 0:
        chain.append(f"highpass=f={highpass_hz}")
    if denoise:
        chain.append("afftdn=nf=-30:nr=12:tn=1")
    if deess:
        chain.append("deesser=i=0.4")
    if compress:
        chain.append("acompressor=threshold=0.08:ratio=3:attack=5:release=120:makeup=2.5")
    return ",".join(chain) or "anull"


def clean_voice(in_wav, out_wav, *, denoise: bool = False, deess: bool = True, compress: bool = False,
                highpass_hz: int = 80) -> Path:
    """Очистить голос. Длина не меняется; формат на выходе — float32 WAV, частота как у входа."""
    out = Path(out_wav)
    out.parent.mkdir(parents=True, exist_ok=True)
    af = clean_filter(denoise=denoise, deess=deess, compress=compress, highpass_hz=highpass_hz)
    run_ffmpeg(["-i", in_wav, "-af", af, "-c:a", "pcm_f32le", out])
    return out
