"""Проверка микрофона: уровень шума и речи, перегруз, совет — и замер задержки «динамик → микрофон» петлёй.

Всё считается на numpy/scipy, без нейросетей. Мастер проверки в интерфейсе делает три вещи:
1. 2 с тишины и фраза вслух одним куском → `analyze_mic` (POST /mic/check);
2. бипы из динамика, записанные микрофоном → `measure_latency` (POST /mic/latency), сигнал отдаёт `latency_signal`
   (GET /mic/latency_signal.wav);
3. итог лежит в AppSettings.mic_profile («Мой микрофон»).
"""
from __future__ import annotations

import io
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SR = 48000

# ───────────────────────── пороги ─────────────────────────
NOISY_FLOOR_DB = -45.0      # шум комнаты громче этого — «шумно»
NOISY_SNR_DB = 25.0         # речь громче шума меньше чем на столько — «шумно»
QUIET_PEAK_DB = -26.0       # пик речи тише — «подойди ближе»
LOUD_PEAK_DB = -6.0         # пик речи громче — «чуть дальше»
SILENT_PEAK_DB = -50.0      # пик тише — микрофон ничего не слышит
CLIP_LEVEL = 0.989          # отсчёт не тише этого (доля полной шкалы) — на потолке
CLIP_MIN_SAMPLES = 3        # …и таких отсчётов не меньше — звук срезан
MIN_SPEECH_S = 0.3          # речи меньше — считаем, что человек ничего не сказал

ADVICE = {
    "good": "Всё хорошо — можно записывать.",
    "silent": "Не слышу твой голос. Проверь, что выбран нужный микрофон и он не выключен, и скажи фразу ещё раз.",
    "quiet": "Тихо — подойди ближе к микрофону (на ширину ладони) или говори увереннее.",
    "loud": "Громковато — отодвинься от микрофона чуть дальше.",
    "clip": "Звук рвётся на громких местах. Отодвинься от микрофона или убавь его громкость в настройках звука.",
    "noisy": "В комнате шумно: фон слышен рядом с голосом. Закрой окно, выключи вентилятор — или подойди ближе к микрофону.",
}


def _db(v: float) -> float:
    return float(20 * np.log10(max(float(v), 1e-9)))


def _pdb(p: float) -> float:
    return float(10 * np.log10(max(float(p), 1e-18)))


def _frames_power(x: np.ndarray, sr: int, win: float = 0.02) -> np.ndarray:
    n = max(1, round(sr * win))
    k = len(x) // n
    if k == 0:
        return np.zeros(0)
    fr = np.asarray(x[: k * n], dtype=np.float64).reshape(k, n)
    return np.mean(fr * fr, axis=1)


# ───────────────────────── загрузка звука ─────────────────────────
def load_audio(data: bytes, sr: int = SR) -> np.ndarray:
    """Любой присланный звук (wav из браузера, webm, …) → mono float32 при sr. Не читается — EngineError по-русски."""
    from ..util import EngineError, run_ffmpeg
    try:
        with tempfile.TemporaryDirectory(prefix="dubl-mic-") as tmp:
            raw, out = Path(tmp) / "in.bin", Path(tmp) / "out.wav"
            raw.write_bytes(data)
            run_ffmpeg(["-i", raw, "-vn", "-ac", "1", "-ar", str(sr), "-c:a", "pcm_f32le", out])
            import soundfile as sf
            x, _ = sf.read(str(out), dtype="float32", always_2d=False)
    except EngineError:
        raise EngineError("Не удалось прочитать запись с микрофона. Попробуй записать ещё раз.") from None
    return np.asarray(x, dtype=np.float32)


# ───────────────────────── уровень, шум, перегруз ─────────────────────────
@dataclass
class MicMeasure:
    noise_floor_db: float | None     # шум комнаты в тишине (RMS), дБFS
    speech_db: float | None          # средний уровень речи (RMS по фразам), дБFS
    peak_db: float | None            # пик речи, дБFS
    snr_db: float | None
    clipping: bool
    speech_found: bool
    verdict: str                     # good | quiet | loud | noisy | silent
    advice: str


def noise_level(x: np.ndarray, sr: int) -> float | None:
    """Шум комнаты по куску, где речи быть не должно: берём самые тихие 80 % кадров (кашель и щелчок не в счёт)."""
    p = _frames_power(x, sr)
    if len(p) < 5:
        return None
    keep = np.sort(p)[: max(3, int(len(p) * 0.8))]
    return _pdb(float(np.mean(keep)))


def quietest_window_noise(x: np.ndarray, sr: int, win_s: float = 0.5) -> float | None:
    """Шум, когда границы тишины неизвестны: самое тихое окно win_s в записи."""
    p = _frames_power(x, sr)
    w = round(win_s / 0.02)
    if len(p) < w + 1:
        return None
    c = np.concatenate([[0.0], np.cumsum(p)])
    return _pdb(float(np.min((c[w:] - c[:-w]) / w)))


def judge(noise_db: float | None, speech_db: float | None, peak_db: float | None, clipping: bool,
          speech_found: bool) -> tuple[str, str]:
    """(вердикт, совет). Порядок: нет речи → перегруз → громко → тихо → шумно → хорошо.
    «Тихо» раньше «шумно»: шаг к микрофону лечит и то и другое."""
    if not speech_found or peak_db is None or peak_db < SILENT_PEAK_DB:
        return "silent", ADVICE["silent"]
    if clipping:
        return "loud", ADVICE["clip"]
    if peak_db > LOUD_PEAK_DB:
        return "loud", ADVICE["loud"]
    if peak_db < QUIET_PEAK_DB:
        return "quiet", ADVICE["quiet"]
    snr = None if (noise_db is None or speech_db is None) else speech_db - noise_db
    if (noise_db is not None and noise_db > NOISY_FLOOR_DB) or (snr is not None and snr < NOISY_SNR_DB):
        return "noisy", ADVICE["noisy"]
    return "good", ADVICE["good"]


def analyze_mic(x: np.ndarray, sr: int, *, silence_s: float | None = None) -> MicMeasure:
    """Запись «тишина, потом фраза».

    silence_s — сколько секунд от начала записи человек молчал (в мастере — 2 с). Тишина: от 0,1 с до silence_s − 0,05 с
    (края с щелчком кнопки отрезаем), речь — всё после. Не передано — речь ищем по энергии, а шум берём по самому тихому
    полусекундному окну."""
    from ..audio.fit import detect_speech

    x = np.asarray(x, dtype=np.float32)
    if x.ndim > 1:
        x = x.mean(axis=1).astype(np.float32)
    split = None
    if silence_s is not None and 0.4 <= silence_s <= len(x) / sr - 0.3:
        split = silence_s
    if split is not None:
        a, b = int(0.1 * sr), int(max(0.2, split - 0.05) * sr)
        noise = noise_level(x[a:b], sr)
        region = x[int(split * sr):]
    else:
        noise = quietest_window_noise(x, sr)
        region = x
    iv = detect_speech(region, sr, min_gap=0.15)
    found = sum(b - a for a, b in iv) >= MIN_SPEECH_S
    if not found:
        pk = _db(float(np.max(np.abs(region)))) if len(region) else None
        v, adv = judge(noise, None, pk, False, False)
        return MicMeasure(None if noise is None else round(noise, 1), None, None if pk is None else round(pk, 1),
                          None, False, False, v, adv)
    seg = np.concatenate([np.asarray(region[int(a * sr): int(b * sr)], dtype=np.float64) for a, b in iv])
    speech = _pdb(float(np.mean(seg * seg)))
    peak = _db(float(np.max(np.abs(seg))))
    clipping = int(np.sum(np.abs(seg) >= CLIP_LEVEL)) >= CLIP_MIN_SAMPLES
    snr = None if noise is None else speech - noise
    v, adv = judge(noise, speech, peak, clipping, True)
    return MicMeasure(None if noise is None else round(noise, 1), round(speech, 1), round(peak, 1),
                      None if snr is None else round(snr, 1), clipping, True, v, adv)


# ───────────────────────── замер задержки петлёй ─────────────────────────
# Сигнал, который играет интерфейс: тишина 0,5 с, потом три «чирика» (тон растёт от 1,5 до 3,5 кГц), затем тишина.
# Чирик, а не чистый тон: у него резкий пик корреляции (±0,3 мс), а колонки ноутбука его играют хорошо.
SIGNAL_SR = 48000
SIGNAL_LEAD_S = 0.5                    # тишина перед первым бипом
SIGNAL_BEEPS_S = (0.5, 0.9, 1.4)       # где начинается каждый бип (секунды от начала файла)
SIGNAL_BEEP_S = 0.08                   # длина одного бипа
SIGNAL_F0, SIGNAL_F1 = 1500.0, 3500.0  # тон бипа в начале и в конце, Гц
SIGNAL_DURATION_S = 2.0                # весь файл
SIGNAL_GAIN = 0.7                      # пик сигнала (доля полной шкалы)
MIN_RECORD_S = 3.0                     # писать микрофоном не меньше — сигнал 2 с и запас на задержку
MAX_LATENCY_MS = 1500.0
MIN_CONFIDENCE = 0.5                   # ниже — результату не верим и в профиль не пишем


def latency_signal(sr: int = SIGNAL_SR) -> np.ndarray:
    """Эталонный сигнал замера: mono float32, SIGNAL_DURATION_S секунд."""
    n = int(SIGNAL_DURATION_S * sr)
    y = np.zeros(n, dtype=np.float64)
    m = int(SIGNAL_BEEP_S * sr)
    t = np.arange(m) / sr
    k = (SIGNAL_F1 - SIGNAL_F0) / SIGNAL_BEEP_S
    beep = np.sin(2 * np.pi * (SIGNAL_F0 * t + 0.5 * k * t * t)) * np.hanning(m) ** 0.5
    for s in SIGNAL_BEEPS_S:
        i = int(s * sr)
        y[i:i + m] += beep
    return (SIGNAL_GAIN * y / np.max(np.abs(y))).astype(np.float32)


def latency_signal_wav(sr: int = SIGNAL_SR) -> bytes:
    import soundfile as sf
    buf = io.BytesIO()
    sf.write(buf, latency_signal(sr), sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


@dataclass
class LatencyMeasure:
    latency_ms: float | None
    confidence: float
    message: str


def measure_latency(rec: np.ndarray, sr: int, *, play_offset_s: float = 0.0) -> LatencyMeasure:
    """Найти эталонный сигнал в записи микрофона (кросс-корреляция) → задержка в мс.

    play_offset_s — в какую секунду записи интерфейс запустил воспроизведение. Задержка = (где сигнал нашёлся в записи)
    − play_offset_s. Уверенность 0..1: насколько сигнал похож на эталон и насколько главный пик выше остальных."""
    from scipy.signal import fftconvolve

    rec = np.asarray(rec, dtype=np.float64)
    if rec.ndim > 1:
        rec = rec.mean(axis=1)
    sig = latency_signal(sr).astype(np.float64)
    if len(rec) < len(sig) + int(0.05 * sr):
        return LatencyMeasure(None, 0.0, "Запись слишком короткая: нужно писать хотя бы 3 секунды, пока играют сигналы.")
    if _db(float(np.max(np.abs(rec)))) < -55.0:
        return LatencyMeasure(None, 0.0, "Микрофон почти ничего не слышит. Проверь, что выбран нужный микрофон и он "
                                         "не выключен, потом повтори.")
    # убираем гул и щелчки: полоса вокруг сигнала
    from scipy.signal import butter, sosfiltfilt
    sos = butter(4, [1000.0, 4500.0], btype="bandpass", fs=sr, output="sos")
    r = sosfiltfilt(sos, rec)
    s = sosfiltfilt(sos, sig)
    corr = fftconvolve(r, s[::-1], mode="valid")                    # corr[i] — сходство сигнала со звуком, начиная с i
    n = len(s)
    e = np.concatenate([[0.0], np.cumsum(r * r)])
    win_e = np.sqrt(np.maximum(e[n:] - e[:-n], 1e-12))
    ncc = np.abs(corr) / (np.sqrt(float(np.sum(s * s))) * win_e + 1e-12)
    i = int(np.argmax(ncc))
    peak = float(ncc[i])
    # вторая вершина вне ±60 мс: если главный пик не выделяется, это шум, а не наш сигнал
    guard = int(0.06 * sr)
    other = np.concatenate([ncc[:max(0, i - guard)], ncc[i + guard:]])
    second = float(np.max(other)) if len(other) else 0.0
    # уточнение до долей отсчёта по параболе
    off = 0.0
    if 0 < i < len(ncc) - 1:
        a, b, c = ncc[i - 1], ncc[i], ncc[i + 1]
        den = a - 2 * b + c
        if abs(den) > 1e-12:
            off = float(np.clip(0.5 * (a - c) / den, -0.5, 0.5))
    lat_ms = (i + off) / sr * 1000.0 - play_offset_s * 1000.0
    sharp = float(np.clip((peak / max(second, 1e-9) - 1.0) / 0.5, 0.0, 1.0))
    conf = float(np.clip(peak / 0.25, 0.0, 1.0)) * sharp
    conf = round(conf, 2)
    if conf < MIN_CONFIDENCE:
        return LatencyMeasure(None, conf, "Сигнал не расслышан. Сними наушники, включи колонки погромче и повтори в "
                                          "тишине.")
    if -5.0 <= lat_ms < 0:                                        # погрешность: сигнал пришёл «одновременно»
        lat_ms = 0.0
    if lat_ms < 0 or lat_ms > MAX_LATENCY_MS:
        return LatencyMeasure(None, conf, "Замер вышел странным. Повтори в тишине, не трогая звук во время сигналов.")
    return LatencyMeasure(round(lat_ms, 1), conf, f"Задержка записи — {round(lat_ms)} мс.")
