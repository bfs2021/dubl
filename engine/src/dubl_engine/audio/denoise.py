"""Шумодав дубля по снимку тишины: спектральный гейт в стиле Audacity Noise Reduction (без нейросетей).

Как работает (числа — из noisereductioneffect.cpp, см. design/research/audacity.md §2.1):
* профиль шума — средняя мощность по частотам в самом тихом ровном участке записи (`levels.quiet_segment`: чаще всего
  преролл до первого слова); можно передать готовый профиль (например, сохранённый «снимок комнаты» микрофона);
* окно STFT 2048, шаг 512 (Hann, 75 % перекрытия);
* полоса считается шумом, если во втором по величине из 5 соседних окон мощность не выше профиля + sensitivity дБ
  («второй по громкости» — защита от одиночных всплесков и «музыкального шума»);
* шумовые полосы ослабляются на `reduction_db` (6–12 дБ: умеренно, чтобы не было булькающего «музыкального шума»);
  узкие тональные пики профиля (сетевой фон 50/100 Гц, писк блока питания) гасятся глубже и без размазывания по частоте;
* сглаживание по частоте и по времени: открывается за 20 мс (attack), закрывается за 100 мс (release).

Что НЕ делает: не вырезает шум, который звучит одновременно с голосом (в полосах, где голос громче шума, пропускает всё).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .levels import noise_floor_db, quiet_segment, speech_rms_db

WIN = 2048
HOP = 512
SENSITIVITY_DB = 8.0          # порог «это шум»: профиль + столько дБ (у Audacity по умолчанию ≈11,4 дБ; 8 бережнее к согласным)
SMOOTH_BANDS = 3              # сглаживание коэффициента по частоте: ±полос (по 23 Гц на полосу при 48 кГц)
ATTACK_S = 0.020
RELEASE_S = 0.100
TONAL_EXTRA_DB = 6.0          # тональные пики профиля гасим глубже
TONAL_PROMINENCE_DB = 9.0     # пик профиля выше соседей на столько дБ — считаем тональным
MIN_PROFILE_FRAMES = 6

# автопилот: чистим, если речь громче шума меньше чем на столько дБ; чем шумнее запись, тем глубже (6…12 дБ)
AUTO_SNR_DB = 45.0
DEPTH_MIN_DB, DEPTH_MAX_DB = 6.0, 12.0


@dataclass
class DenoiseInfo:
    applied: bool
    reason: str = ""                  # почему не применили (для логов и тестов)
    depth_db: float = 0.0
    noise_floor_db: float | None = None
    snr_db: float | None = None
    profile_from: str = ""            # "take" — тишина самого дубля, "given" — переданный профиль


def _stft(x: np.ndarray, sr: int) -> np.ndarray:
    from scipy.signal import stft

    _f, _t, z = stft(x.astype(np.float32), fs=sr, window="hann", nperseg=WIN, noverlap=WIN - HOP,
                     boundary="zeros", padded=True)
    return z


def _istft(z: np.ndarray, sr: int, n: int) -> np.ndarray:
    from scipy.signal import istft

    _t, y = istft(z, fs=sr, window="hann", nperseg=WIN, noverlap=WIN - HOP, boundary=True)
    y = np.asarray(y, dtype=np.float32)
    return y[:n] if len(y) >= n else np.concatenate([y, np.zeros(n - len(y), dtype=np.float32)])


def noise_profile(x: np.ndarray, sr: int, seg: tuple[float, float] | None = None) -> np.ndarray | None:
    """Средняя мощность шума по частотам (длина WIN//2+1) из тихого участка. None — тишины в записи нет."""
    x = np.asarray(x, dtype=np.float32)
    seg = seg or quiet_segment(x, sr)
    if seg is None:
        return None
    s = x[int(seg[0] * sr): int(seg[1] * sr)]
    if len(s) < WIN + HOP * MIN_PROFILE_FRAMES:
        return None
    z = _stft(s, sr)[:, 1:-1]                             # края — с нулевым дополнением, в профиль не берём
    if z.shape[1] < MIN_PROFILE_FRAMES:
        return None
    return np.mean(np.abs(z) ** 2, axis=1).astype(np.float64)


def profile_to_list(profile: np.ndarray) -> list[float]:
    """Профиль шума → короткий список для settings.json (дБ по частотам, 1 знак)."""
    return [round(float(v), 1) for v in 10 * np.log10(np.asarray(profile, dtype=np.float64) + 1e-20)]


def profile_from_list(values: list[float]) -> np.ndarray | None:
    """Обратно: список дБ → профиль. None — размер не тот (профиль от другой версии)."""
    if len(values) != WIN // 2 + 1:
        return None
    return (10 ** (np.asarray(values, dtype=np.float64) / 10)).astype(np.float64)


def _tonal_bins(profile: np.ndarray) -> np.ndarray:
    from scipy.ndimage import median_filter

    pdb = 10 * np.log10(profile + 1e-20)
    prom = pdb - median_filter(pdb, size=41, mode="nearest")
    t = prom > TONAL_PROMINENCE_DB
    t[:2] = False                                          # постоянная составляющая и самый низ — не «писк»
    return t


def _second_greatest(p: np.ndarray) -> np.ndarray:
    """По времени: второе по величине из окон t−2…t+2 (форма (F, T)). Считаем кусками — длинные записи не раздувают память."""
    f, t = p.shape
    pad = np.pad(p, ((0, 0), (2, 2)), mode="edge")
    out = np.empty_like(p)
    step = 1500
    for a in range(0, t, step):
        b = min(a + step, t)
        st = np.stack([pad[:, a + k: b + k] for k in range(5)], axis=0)
        out[:, a:b] = np.partition(st, 3, axis=0)[3]
    return out


def spectral_gate(x: np.ndarray, sr: int, profile: np.ndarray, *, reduction_db: float = 9.0,
                  sensitivity_db: float = SENSITIVITY_DB, smooth_bands: int = SMOOTH_BANDS,
                  attack_s: float = ATTACK_S, release_s: float = RELEASE_S) -> np.ndarray:
    """Ослабить шум по профилю. Длина результата = длине входа."""
    from scipy.ndimage import uniform_filter1d

    x = np.asarray(x, dtype=np.float32)
    n = len(x)
    if n < WIN:
        return x
    z = _stft(x, sr)
    p = (np.abs(z) ** 2).astype(np.float32)
    thr = (profile * 10 ** (sensitivity_db / 10))[:, None].astype(np.float32)
    noise = _second_greatest(p) <= thr                                      # (F, T): True — здесь только шум

    tonal = _tonal_bins(profile)
    g_floor = 10 ** (-reduction_db / 20)
    g_tonal = 10 ** (-(reduction_db + TONAL_EXTRA_DB) / 20)
    target = np.where(noise, g_floor, 1.0).astype(np.float32)
    target[tonal, :] = np.where(noise[tonal, :], g_tonal, 1.0)
    if smooth_bands > 0:                                                    # сглаживаем логарифм коэффициента по частоте
        lg = np.log(target)
        sm = uniform_filter1d(lg, size=2 * smooth_bands + 1, axis=0, mode="nearest")
        sm[tonal, :] = lg[tonal, :]                                         # тональные пики не размазываем
        target = np.exp(np.minimum(sm, 0.0)).astype(np.float32)

    # упреждение: гейт открывается на attack раньше прихода звука — начало слова не «проглатывается»
    ahead = max(1, int(round(attack_s * sr / HOP)))
    if target.shape[1] > ahead:
        look = target.copy()
        for k in range(1, ahead + 1):
            look[:, :-k] = np.maximum(look[:, :-k], target[:, k:])
        target = look
    # по времени: открываемся быстро (attack), закрываемся медленно (release)
    dt = HOP / sr
    ca = 1.0 - np.exp(-dt / max(attack_s, 1e-4))
    cr = 1.0 - np.exp(-dt / max(release_s, 1e-4))
    g = np.empty_like(target)
    cur = target[:, 0].copy()
    for t in range(target.shape[1]):
        tg = target[:, t]
        c = np.where(tg > cur, ca, cr)
        cur = cur + c * (tg - cur)
        g[:, t] = cur
    return _istft(z * g, sr, n)


def auto_depth_db(snr_db: float | None) -> float:
    """Глубина шумодава автопилота: чем меньше речь над шумом, тем глубже (6…12 дБ)."""
    if snr_db is None:
        return DEPTH_MIN_DB + 3.0
    return float(np.clip(DEPTH_MIN_DB + 0.6 * (AUTO_SNR_DB - snr_db), DEPTH_MIN_DB, DEPTH_MAX_DB))


def denoise_voice(x: np.ndarray, sr: int, *, profile: np.ndarray | None = None, depth_db: float | None = None,
                  force: bool = False) -> tuple[np.ndarray, DenoiseInfo]:
    """Шумодав дубля по снимку тишины. → (звук, что сделали).

    Профиль: свой тихий участок записи; нет его — переданный `profile` (общий для микрофона/сцены); нет и его — не чистим.
    Без `force` чистит только если шум заметен (речь над шумом < AUTO_SNR_DB).
    """
    x = np.asarray(x, dtype=np.float32)
    seg = quiet_segment(x, sr)
    prof = noise_profile(x, sr, seg) if seg is not None else None
    src = "take"
    if prof is None:
        prof, src = profile, "given"
    nf = noise_floor_db(x, sr) if seg is not None else None
    sp = speech_rms_db(x, sr)
    snr = None if (nf is None or sp is None) else float(sp - nf)
    if prof is None:
        return x, DenoiseInfo(False, "нет участка тишины и нет готового профиля", noise_floor_db=nf, snr_db=snr)
    if not force and snr is not None and snr >= AUTO_SNR_DB:
        return x, DenoiseInfo(False, "шум и так тихий", noise_floor_db=nf, snr_db=snr, profile_from=src)
    depth = float(depth_db) if depth_db is not None else auto_depth_db(snr)
    y = spectral_gate(x, sr, prof, reduction_db=depth)
    return y, DenoiseInfo(True, "", depth_db=depth, noise_floor_db=nf, snr_db=snr, profile_from=src)
