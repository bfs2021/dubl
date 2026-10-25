"""Сведение сцены и автопилот.

Позиция дубля в сцене (допущение, общее с audio.fit):
    начало записанного файла лежит в сцене на
        take.start_rel − take.preroll + take.offset_ms/1000 + pivot·(1 − take.stretch)
    где pivot — начало речи в файле дубля (первая речь в окне реплики; её находит тот же детектор, что и в fit_take),
    а растяжение take.stretch делается вокруг pivot. При stretch = 1 это просто
        take.start_rel − take.preroll + take.offset_ms/1000.
    Latency звуковой карты отдельно не учитывается: он уже внутри offset_ms, потому что fit совмещает речь дубля
    с речью оригинала по факту.

Порядок обработки одного дубля: починка перегруза (declip) → шумодав по снимку тишины (denoise) → убрать «стук в
начале» (blip) → чистка (clean: срез низа, де-эссер) → выравнивание, если голос «прыгает» (dynamics) → эффект (fx) →
растяжение → обрезка по реплике → комната (room) → усиление дорожки → позиция. Комната идёт после растяжения (иначе хвост
реверберации растянулся бы). Шумодав стоит ДО всего, что меняет громкость: усиление и сжатие не должны поднимать шум.
"""
from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..models import MixCheck, MixMode, MixSettings, Scene, SceneLine, Take, TrackSettings
from ..util import EngineError
from .clean import clean_voice
from .denoise import denoise_voice
from .dynamics import voice_compress
from .fit import detect_speech, find_take_speech, locate_speech, time_stretch
from .fx import apply_fx
from .io import db_to_gain, read_wav, to_mono, to_stereo, write_wav
from .levels import CLIP_WARN_PCT, declip, noise_floor_db, quiet_segment
from .rooms import ROOMS, apply_room, guess_room

SPEECH_TARGET_LUFS = -20.0      # к этому уровню автопилот подтягивает речь каждого персонажа
BG_MARGIN_DB = 10.0             # речь выше фона в моменты речи (середина «8–12 дБ»)
DUCK_DB = 6.0                   # на сколько фон приседает под новыми голосами
BREATH_DB = -18.0
_NOROOM = {"dry"}
EMPTY_TAKE_PEAK_DB = -50.0      # пик тише — запись пустая (микрофон выключен): в сведение не идёт
MAX_BOOST_DB = 24.0             # потолок усиления дорожки автопилотом (шум к этому моменту уже вычищен шумодавом)
BOOST_CAP_TOL_DB = 0.5          # усиление меньше нужного больше чем на столько — честно говорим «упёрлись» (boost_capped)
NOISE_CEIL_DB = -56.0           # шум дубля после усиления не должен быть громче этого (RMS, дБ полной шкалы)
MIN_BOOST_DB = 6.0              # но хотя бы столько усиления тихому дублю даём всегда (с предупреждением)
SPREAD_COMPRESS_DB = 6.0        # голос «прыгает» по громкости больше чем на столько дБ (p95−p10) — выравниваем
BLIP_MAX_S = 0.25               # короткий стук/щелчок в самом начале записи (нажатие кнопки), отделённый паузой, вырезаем
BLIP_GAP_S = 0.8
TAKE_FADE_MS = 25.0
# проверка готового сведения (mix_check): голос тонет или ролик приходится сильно поднимать — честно предупреждаем
VOICE_BURIED_DB = 4.0           # новые голоса громче фона в моменты речи меньше чем на столько — голос тонет
MASTER_BOOST_WARN_DB = 10.0     # мастеринг поднимает сведение больше чем на столько — вместе с голосом вылезут шумы фона


@dataclass
class SceneStems:
    voices: Path
    background: Path
    per_speaker: dict[str, Path]
    takes_dir: Path
    scene_dir: Path
    # готовый профиль шума микрофона (denoise.noise_profile): берётся, если в самом дубле нет участка тишины
    noise_profile: "np.ndarray | None" = None


# ─────────────────────────── выбор дублей ───────────────────────────
def take_is_empty(t: Take) -> bool:
    """Запись пустая: пайплайн пометил (`empty`) или пик записи тише EMPTY_TAKE_PEAK_DB (≈ −60 дБ — шум АЦП, не голос)."""
    return bool(t.empty) or (t.peak_db is not None and t.peak_db < EMPTY_TAKE_PEAK_DB)


def _pick_takes(scene: Scene) -> list[tuple[Take, SceneLine | None, str]]:
    """Какие дубли идут в сведение: (дубль, реплика или None, ключ дорожки/персонаж).

    Для каждой реплики — выбранный (selected) дубль, иначе последний по created.
    Дубль всей сцены (line_id=None) берётся, если он выбран, либо если у персонажа нет дублей по репликам.
    Если выбран дубль всей сцены, дубли по репликам идут только выбранные (★).
    """
    lines = {l.line_id: l for l in scene.lines}
    dubbed = set(scene.dubbed_speakers)
    per_line: dict[str, list[Take]] = {}
    whole: dict[str, list[Take]] = {}
    for t in scene.takes:
        if take_is_empty(t):
            continue                                   # пустая запись (≈ −60 дБ) в сведение не идёт, ★ у неё нет смысла
        line = lines.get(t.line_id) if t.line_id else None
        if t.line_id and line is None:
            continue                                   # реплики уже нет в сцене
        spk = t.speaker_id or (line.speaker_id if line else None) or ""
        if spk and dubbed and spk not in dubbed:
            continue
        if not spk and dubbed and line is not None:
            continue
        key = spk
        if t.line_id:
            per_line.setdefault(f"{key}\x00{t.line_id}", []).append(t)
        else:
            whole.setdefault(key, []).append(t)

    def choose(ts: list[Take]) -> Take:
        sel = [t for t in ts if t.selected]
        pool = sel or ts
        return max(pool, key=lambda t: t.created)

    out: list[tuple[Take, SceneLine | None, str]] = []
    whole_chosen: dict[str, Take] = {}
    for spk, ts in whole.items():
        sel = [t for t in ts if t.selected]
        has_lines = any(k.startswith(spk + "\x00") for k in per_line)
        if sel or not has_lines:
            whole_chosen[spk] = choose(ts)
    for key, ts in per_line.items():
        spk, lid = key.split("\x00", 1)
        if spk in whole_chosen and whole_chosen[spk].selected:
            ts = [t for t in ts if t.selected]
            if not ts:
                continue
        out.append((choose(ts), lines[lid], spk))
    for spk, t in whole_chosen.items():
        out.append((t, None, spk))
    out.sort(key=lambda r: (r[1].start_rel if r[1] else -1.0))
    return out


# ─────────────────────────── вспомогательное ───────────────────────────
def _resample(x: np.ndarray, sr_from: int, sr_to: int) -> np.ndarray:
    if sr_from == sr_to:
        return x
    from math import gcd

    from scipy.signal import resample_poly
    g = gcd(sr_from, sr_to)
    return resample_poly(x, sr_to // g, sr_from // g, axis=0).astype(np.float32)


def _read_mono(path: Path, sr: int) -> np.ndarray:
    x, s = read_wav(path)
    return _resample(to_mono(x), s, sr)


def _fade(x: np.ndarray, sr: int, ms: float = 15.0) -> np.ndarray:
    n = min(int(sr * ms / 1000), len(x) // 2)
    if n > 0:
        r = np.linspace(0.0, 1.0, n, dtype=np.float32)
        x[:n] *= r
        x[-n:] *= r[::-1]
    return x


def _place(dst: np.ndarray, src: np.ndarray, start_s: float, sr: int) -> None:
    i = round(start_s * sr)
    lo = max(0, -i)
    hi = min(len(src), len(dst) - i)
    if hi > lo:
        dst[i + lo: i + hi] += src[lo:hi]


def _guide_for(stems: SceneStems, spk: str) -> Path | None:
    """Голос оригинала для сверки: дорожка персонажа, иначе общие голоса (как в pipeline.add_take)."""
    for cand in (stems.per_speaker.get(spk), stems.voices):
        if cand is not None and Path(cand).is_file():
            return Path(cand)
    return None


def _write_tmp(y: np.ndarray, sr: int, path: Path) -> Path:
    return write_wav(path, y, sr)


def _speech_spread_db(x: np.ndarray, sr: int, iv: list[tuple[float, float]], win: float = 0.4) -> float:
    """Насколько «прыгает» громкость речи: p95 − p10 по окнам 0,4 с внутри речи (дБ)."""
    vals = []
    n = int(win * sr)
    for a, b in iv:
        for s0 in np.arange(a, b - win, win / 2):
            seg = x[int(s0 * sr): int(s0 * sr) + n].astype(np.float64)
            if len(seg) == n:
                vals.append(20 * np.log10(np.sqrt(np.mean(seg * seg)) + 1e-9))
    if len(vals) < 4:
        return 0.0
    return float(np.percentile(vals, 95) - np.percentile(vals, 10))


def _blank_start_blip(x: np.ndarray, sr: int, iv: list[tuple[float, float]]) -> np.ndarray:
    """Стук/писк в самом начале записи (нажатие кнопки, хвост сигнала отсчёта из колонок), отделённый от речи паузой, —
    убираем из звука. Позицию речи (pivot) это не меняет: её считаем по исходной записи, как и fit_take."""
    if len(iv) < 2:
        return x
    (a, b), (a2, _b2) = iv[0], iv[1]
    if a < 0.3 and b - a <= BLIP_MAX_S and a2 - b >= BLIP_GAP_S:
        y = x.copy()
        y[: min(len(y), int((b + 0.05) * sr))] = 0.0      # сразу за стуком — пауза, ступеньки нет
        return y
    return x


def _process_take(take: Take, line: SceneLine | None, scene_dir: Path, sr: int, *, fx: str, room: str,
                  tmp: Path, guide: Path | None = None, denoise: bool = True,
                  noise_profile: np.ndarray | None = None, info: dict | None = None) -> tuple[np.ndarray, float] | None:
    """Дубль → (моно-аудио после clean/fx/stretch/room, позиция его начала в сцене в секундах).

    `info` (если передан) заполняется замерами: clip_pct (перегруз в записи), noise_db (шум после очистки),
    denoise_db (глубина шумодава или 0)."""
    path = scene_dir / take.file
    if not path.exists():
        raise EngineError(f"Не найден файл дубля {take.id}: {take.file}")
    raw = _read_mono(path, sr)
    if len(raw) < int(0.05 * sr) or float(np.abs(raw).max()) < 10 ** (EMPTY_TAKE_PEAK_DB / 20):
        return None                                      # пустая запись: в сведение не идёт
    info = info if info is not None else {}

    # 1. перегруз: чиним срезанные верхушки (до всего остального — шумодав и сжатие срез только подчеркнули бы)
    work, clip = declip(raw)
    info["clip_pct"] = clip.pct
    # 2. шумодав по снимку тишины самого дубля (нет тишины — по профилю микрофона, нет и его — не трогаем)
    iv = detect_speech(raw, sr)
    work = _blank_start_blip(work, sr, iv)
    info["denoise_db"] = 0.0
    if denoise:
        work, dn = denoise_voice(work, sr, profile=noise_profile)
        info["denoise_db"] = dn.depth_db if dn.applied else 0.0
    # 3. срез низа и де-эссер (ffmpeg). Старый afftdn и acompressor с makeup +8 дБ убраны: первый «булькал» (шум в паузах
    # прыгал на ±15 дБ), второй поднимал шум микрофона на 7 дБ.
    src = tmp / f"{take.id}_in.wav"
    write_wav(src, work, sr)
    cleaned = clean_voice(src, tmp / f"{take.id}_clean.wav")
    y = _read_mono(cleaned, sr)
    # 4. выравниваем, только если голос «прыгает» (шёпот → крик). Ровный дубль не трогаем.
    if _speech_spread_db(y, sr, iv) > SPREAD_COMPRESS_DB:
        y, _gr = voice_compress(y, sr)
    info["noise_db"] = noise_floor_db(y, sr)
    if fx and fx != "none":
        y = _read_mono(apply_fx(_write_tmp(y, sr, tmp / f"{take.id}_fxin.wav"), fx, tmp / f"{take.id}_fx.wav"), sr)

    # где речь в файле (по сырой записи, тем же детектором, что и в fit_take)
    iv = detect_speech(raw, sr)
    sp: tuple[float, float] | None
    if line is not None:
        loc = None
        if guide is not None:
            try:
                g, gsr = read_wav(guide)
                loc = locate_speech(raw, _resample(to_mono(g), gsr, sr), sr, line_start_rel=line.start_rel,
                                    line_end_rel=line.end_rel, take_start_rel=take.start_rel, preroll=take.preroll)
            except Exception:  # noqa: BLE001 — нет/битый гид: берём оценку по одной записи
                loc = None
        if loc is not None:
            sp = (loc.t_ss, loc.t_se)       # ровно тот же пивот, от которого fit_take считал offset_ms/stretch
        else:
            sp = find_take_speech(raw, sr, line_start_rel=line.start_rel, line_end_rel=line.end_rel,
                                  take_start_rel=take.start_rel, preroll=take.preroll, latency_ms=0.0, intervals=iv)
    else:
        sp = (iv[0][0], iv[-1][1]) if iv else None
    pivot = sp[0] if sp else 0.0

    stretch = float(take.stretch or 1.0)
    if abs(stretch - 1.0) > 1e-4:
        y = time_stretch(y, sr, stretch)
    start = take.start_rel - take.preroll + take.offset_ms / 1000.0 + pivot * (1.0 - stretch)

    if line is not None and sp is not None:      # дубль одной реплики: оставляем только речь с небольшим запасом
        lo = max(0, int((sp[0] * stretch - 0.25) * sr))
        hi = min(len(y), int((sp[1] * stretch + 0.5) * sr))
        y = _fade(y[lo:hi].copy(), sr, TAKE_FADE_MS)
        start += lo / sr

    room_name = room
    if room_name in ROOMS and room_name not in _NOROOM:
        tail = int(min(ROOMS[room_name].rt60 * 1.1 + 0.05, 3.5) * sr)
        y = apply_room(np.concatenate([y, np.zeros(tail, dtype=np.float32)]), sr, room_name)
    return y.astype(np.float32), float(start)


def _activity(v: np.ndarray, sr: int, *, lookahead: float = 0.03, attack: float = 0.03, release: float = 0.25,
              hold: float = 0.0) -> np.ndarray:
    """Огибающая «тут звучит речь» 0..1 на каждый семпл: гейт по RMS, атака, релиз, упреждение.
    `hold` — сколько держим «речь» после её конца: короткие паузы между словами не отпускают фон (иначе он качается)."""
    hop = max(1, sr // 100)
    k = len(v) // hop + 1
    pad = np.zeros(k * hop, dtype=np.float64)
    pad[: len(v)] = v
    rms = np.sqrt(np.mean(pad.reshape(k, hop) ** 2, axis=1))
    db = 20 * np.log10(rms + 1e-9)
    a = np.clip((db + 48.0) / 6.0, 0.0, 1.0)
    if hold > 0:
        from scipy.ndimage import maximum_filter1d
        h = max(1, int(round(hold / 0.01)))
        a = maximum_filter1d(a, size=h + 1, origin=(h // 2), mode="nearest")      # продлеваем вперёд на hold
    ca, cr = 1 - np.exp(-0.01 / max(attack, 1e-3)), 1 - np.exp(-0.01 / max(release, 1e-3))
    s = np.zeros(k)
    cur = 0.0
    for i in range(k):
        c = ca if a[i] > cur else cr
        cur += c * (a[i] - cur)
        s[i] = cur
    tt = (np.arange(len(v)) / sr + lookahead - 0.005) * 100.0
    return np.interp(tt, np.arange(k), s).astype(np.float32)


def _speech_lufs(x: np.ndarray, sr: int) -> float | None:
    """Громкость речи (LUFS) только по участкам, где есть речь. None — речи нет."""
    import pyloudnorm as pyln

    iv = detect_speech(x, sr, min_gap=0.15)
    if not iv:
        return None
    seg = np.concatenate([x[int(a * sr): int(b * sr)] for a, b in iv])
    if len(seg) < int(0.45 * sr):
        seg = np.concatenate([seg, np.zeros(int(0.45 * sr) - len(seg), dtype=np.float32)])
    v = float(pyln.Meter(sr).integrated_loudness(seg.astype(np.float32)))
    return v if np.isfinite(v) else None


def _bg_info(path: Path) -> tuple[int, int]:
    import soundfile as sf
    i = sf.info(str(path))
    return i.frames, i.samplerate


def _dub_tracks(scene: Scene, stems: SceneStems, mix: MixSettings, sr: int, n: int,
                report: dict[str, dict] | None = None) -> dict[str, np.ndarray]:
    """Новые голоса: speaker_id → моно-дорожка длиной n, без усиления дорожки (gain/mute не применены).

    `report` (если передан): speaker_id → {"noise_db": самый громкий шум комнаты среди дублей после очистки,
    "clip_pct": макс. перегруз в записи, "denoise_db": макс. глубина шумодава}."""
    picks = _pick_takes(scene)
    out: dict[str, np.ndarray] = {}
    if not picks:
        return out
    auto_room: str | None = None
    with tempfile.TemporaryDirectory(prefix="dubl_mix_") as td:
        tmp = Path(td)
        for take, line, spk in picks:
            ts = mix.tracks.get(spk, TrackSettings())
            room = ts.room
            if room == "auto":
                if auto_room is None:
                    auto_room = guess_room(stems.background)
                room = auto_room
            info: dict = {}
            res = _process_take(take, line, stems.scene_dir, sr, fx=ts.voice_fx, room=room, tmp=tmp,
                                guide=_guide_for(stems, spk), denoise=mix.denoise,
                                noise_profile=getattr(stems, "noise_profile", None), info=info)
            if res is None:
                continue
            if report is not None:
                r = report.setdefault(spk, {})
                for k in ("noise_db", "clip_pct", "denoise_db"):
                    v = info.get(k)
                    if v is not None:
                        r[k] = max(r.get(k, -200.0), v)
            y, start = res
            _place(out.setdefault(spk, np.zeros(n, dtype=np.float32)), y, start, sr)
    return out


def _track_audible(ts: TrackSettings, any_solo: bool) -> bool:
    return not ts.mute and (ts.solo or not any_solo)


def _any_solo(mix: MixSettings) -> bool:
    return mix.background.solo or any(t.solo for t in mix.tracks.values())


def _breaths(orig: np.ndarray, sr: int, occupied: np.ndarray) -> np.ndarray:
    """Тихие шорохи/дыхание из оригинала там, где персонаж молчит в новой версии и энергия оригинала низкая."""
    hop = sr // 50
    k = len(orig) // hop
    if k < 5:
        return np.zeros_like(orig)
    fr = orig[: k * hop].astype(np.float64).reshape(k, hop)
    db = 20 * np.log10(np.sqrt(np.mean(fr ** 2, axis=1)) + 1e-9)
    present = db > -75.0                                         # оригинал здесь вообще не тишина
    if not present.any():
        return np.zeros_like(orig)
    peak = np.percentile(db[present], 98)
    quiet = present & (db < peak - 24.0) & (db > peak - 60.0)
    occ = occupied[: k * hop].reshape(k, hop).max(axis=1) > 0.05
    m = (quiet & ~occ).astype(np.float32)
    m = np.convolve(m, np.ones(3, dtype=np.float32) / 3, mode="same")     # плавные края (~60 мс)
    mask = np.interp(np.arange(len(orig)) / hop - 0.5, np.arange(k), m).astype(np.float32)
    return orig * mask * np.float32(db_to_gain(BREATH_DB))


def _line_extents(scene: Scene, voices: np.ndarray, sr: int) -> dict[str, tuple[float, float]]:
    """line_id → отрезок реплики, расширенный до реальных границ речи в стеме голосов (см. separate.voice_extents)."""
    from .separate import voice_extents
    ext = voice_extents(voices, sr, [(l.start_rel, l.end_rel) for l in scene.lines], max_ext=2.0, gap=0.6)
    return {l.line_id: e for l, e in zip(scene.lines, ext)}


def _replaced_spans(scene: Scene, ext: dict[str, tuple[float, float]] | None = None) -> dict[str, list[tuple[float, float]]]:
    """Где голос оригинала заменён дублем: по персонажу — отрезки реплик, на которые есть дубль.

    Дубль всей сцены (line_id=None) заменяет все реплики своего персонажа. Реплики без дубля остаются оригиналом,
    даже если персонаж отмечен как озвучиваемый: «не записал — звучит как было», а не тишина.
    `ext` — расширенные границы реплик (хвосты фраз за пределами тайминга субтитров тоже заменяются).
    """
    ext = ext or {}
    span = lambda l: ext.get(l.line_id, (l.start_rel, l.end_rel))   # noqa: E731
    out: dict[str, list[tuple[float, float]]] = {}
    for _take, line, spk in _pick_takes(scene):
        if line is not None:
            out.setdefault(spk, []).append(span(line))
        else:
            out.setdefault(spk, []).extend(span(l) for l in scene.lines if (l.speaker_id or "") == spk)
    return out


def _voice_layers(scene: Scene, stems: SceneStems, sr: int, n: int):
    """Разложить голоса сцены заново из voices.wav по расширенным границам реплик.

    → (по персонажу: голос внутри его реплик; всё прочее из голосов — крики, смех, толпа — оно идёт в фон;
       расширенные границы реплик). None — если голоса по персонажам не разложены.
    Раскладка делается здесь, а не берётся из voices/<spk>.wav, чтобы сцены, подготовленные старой версией,
    сводились так же, как новые, и персонаж + «прочее» в сумме всегда давали ровно voices.wav."""
    from .separate import span_mask
    vp = Path(stems.voices)
    if not stems.per_speaker or not vp.exists():
        return None
    v, sv = read_wav(vp)
    v = to_stereo(_resample(v, sv, sr))
    m = min(n, len(v))
    v = v[:m]
    ext = _line_extents(scene, v, sr)
    groups: dict[str, list[tuple[float, float]]] = {}
    for l in scene.lines:
        groups.setdefault(l.speaker_id or "unknown", []).append(ext[l.line_id])
    per: dict[str, np.ndarray] = {}
    total = np.zeros(m, dtype=np.float32)
    for spk, spans in groups.items():
        mk = np.minimum(span_mask(m, sr, spans), 1.0 - total)          # пересечения — первому, без удвоения
        total += mk
        y = np.zeros((n, 2), dtype=np.float32)
        y[:m] = v * mk[:, None]
        per[spk] = y
    extra = np.zeros((n, 2), dtype=np.float32)
    extra[:m] = v * (1.0 - total)[:, None]
    return per, extra, ext


def _unassigned_voices(scene: Scene, stems: SceneStems, sr: int, n: int) -> np.ndarray | None:
    """То, что нейросеть отнесла к голосам, но что вне реплик (крики, смех, пение, толпа). Это часть фона сцены:
    без неё фон звучит «не так». None — если голоса по персонажам не разложены (тогда всё и так в voices.wav)."""
    lay = _voice_layers(scene, stems, sr, n)
    return None if lay is None else lay[1]


# ─────────────────────────── сведение ───────────────────────────
def _bed_vs_voice(bed: np.ndarray, voice_bus: np.ndarray, sr: int) -> float | None:
    """Насколько новые голоса громче всего остального (фон, оригиналы) в моменты, когда они звучат, дБ (LUFS − LUFS).
    None — новых голосов нет."""
    import pyloudnorm as pyln

    if np.abs(voice_bus).max() <= 1e-6:
        return None
    l_voice = _speech_lufs(voice_bus, sr)
    if l_voice is None:
        return None
    act = _activity(voice_bus, sr, lookahead=0.0) > 0.5
    ref = to_mono(bed)[act[: len(bed)]]
    if len(ref) < int(0.45 * sr):
        ref = np.concatenate([ref, np.zeros(int(0.45 * sr) - len(ref), dtype=np.float32)])
    l_bed = float(pyln.Meter(sr).integrated_loudness(ref.astype(np.float32)))
    if not np.isfinite(l_bed) or l_bed < -70.0:
        return 60.0                                           # фона в моменты речи нет — голос точно не тонет
    return float(l_voice - l_bed)


def _lufs(x: np.ndarray, sr: int) -> float | None:
    import pyloudnorm as pyln

    if len(x) < int(0.45 * sr):
        x = np.concatenate([x, np.zeros((int(0.45 * sr) - len(x),) + x.shape[1:], dtype=np.float32)])
    v = float(pyln.Meter(sr).integrated_loudness(np.asarray(x, dtype=np.float32)))
    return v if np.isfinite(v) and v > -70.0 else None


def mix_check(mix: MixSettings, voice_over_bg_db: float | None, master_gain_db: float | None) -> MixCheck:
    """Итог сведения для честного предупреждения. Голос тонет (меньше VOICE_BURIED_DB над фоном) или мастер поднимает
    ролик больше MASTER_BOOST_WARN_DB → совет: автопилот выключен — включи его; включён, но дубль не вытянуть без шума
    (boost_capped) — перезапиши громче; включён, а голос всё равно тонет — фон тише. Тихий исходник при включённом
    автопилоте и нормальном голосе — не повод ругаться: тут человеку делать нечего."""
    buried = voice_over_bg_db is not None and voice_over_bg_db < VOICE_BURIED_DB
    boosted = master_gain_db is not None and master_gain_db > MASTER_BOOST_WARN_DB
    warning = None
    if buried or boosted:
        if not mix.autopilot:
            warning = "autopilot"
        elif any(t.boost_capped for t in mix.tracks.values()):
            warning = "record_louder"
        elif buried:
            warning = "buried"
    r = lambda v: None if v is None else round(float(v), 1)   # noqa: E731
    return MixCheck(voice_over_bg_db=r(voice_over_bg_db), master_gain_db=r(master_gain_db), warning=warning)


def mix_scene(scene: Scene, stems: SceneStems, out_wav, report: dict | None = None) -> Path:
    """Свести сцену: фон (+ducking под новыми голосами) + оригинальные голоса + дубли. Выход — stereo float32 WAV
    длиной с background. Позиции дублей — см. docstring модуля.

    `report` (если передан) заполняется замерами для mix_check: voice_over_bg_db — новые голоса над всем остальным
    в моменты речи (дБ), mix_lufs — громкость сведения до мастеринга."""
    from .separate import span_mask
    mix = scene.mix
    bg, sr = read_wav(stems.background)
    bg = to_stereo(bg)
    n = len(bg)
    lay = _voice_layers(scene, stems, sr, n)
    any_solo = _any_solo(mix)
    replaced = _replaced_spans(scene, lay[2] if lay is not None else None)
    # Там, где реплику заменил дубль, оригинального голоса быть не должно ни у кого: границы реплик расширены по звуку,
    # и хвост озвученной фразы может оказаться в чужом слое (у соседа или в «прочем»). Исключение — собственные
    # (нерасширенные) реплики, которые остались оригиналом: если персонажи говорят встык, чужую речь не режем.
    rm_all = np.zeros(n, dtype=np.float32)
    if mix.mode != MixMode.VOICEOVER and replaced:
        rm_all = span_mask(n, sr, [s for spans in replaced.values() for s in spans])
        keep_core = [(l.start_rel, l.end_rel) for l in scene.lines
                     if not any(a <= l.start_rel and l.end_rel <= b for a, b in replaced.get(l.speaker_id or "", []))]
        if keep_core:
            rm_all *= 1.0 - span_mask(n, sr, keep_core, margin=0.0, fade=0.02)
    if lay is not None:
        bg = bg + lay[1] * (1.0 - rm_all)[:, None]

    dubs = _dub_tracks(scene, stems, mix, sr, n)
    voice_bus = np.zeros(n, dtype=np.float32)
    dub_mask: dict[str, np.ndarray] = {}
    for spk, y in dubs.items():
        ts = mix.tracks.get(spk, TrackSettings())
        if not _track_audible(ts, any_solo):
            continue
        g = y * np.float32(db_to_gain(ts.gain_db))
        voice_bus += g
        occ = np.zeros(n, dtype=np.float32)
        for a, b in detect_speech(y, sr, min_gap=0.3):          # где персонаж говорит в новой версии (+запас 100 мс)
            occ[max(0, int((a - 0.1) * sr)): int((b + 0.1) * sr)] = 1.0
        dub_mask[spk] = occ

    # фон
    out = np.zeros((n, 2), dtype=np.float32)
    if _track_audible(mix.background, any_solo):
        g_bg = np.float32(db_to_gain(mix.background.gain_db))
        if mix.duck_background and np.abs(voice_bus).max() > 1e-6:
            duck = 1.0 - _activity(voice_bus, sr, lookahead=0.08, attack=0.06, release=0.40, hold=0.30) * (1.0 - db_to_gain(-DUCK_DB))
            out += bg * (duck[:, None] * g_bg)
        else:
            out += bg * g_bg

    # оригинальные голоса
    dubbed = set(scene.dubbed_speakers)
    voiceover = mix.mode == MixMode.VOICEOVER
    orig_files = dict(stems.per_speaker)
    if voiceover and not orig_files and Path(stems.voices).exists():
        orig_files = {"": Path(stems.voices)}
    layers = lay[0] if lay is not None else {}
    for spk in list(dict.fromkeys(list(layers) + list(orig_files))):
        ts = mix.tracks.get(spk, TrackSettings())
        if not _track_audible(ts, any_solo):
            continue
        if voiceover:
            gain = db_to_gain(mix.original_voices_db)
        elif spk in dubbed:
            gain = 1.0                                          # DUB: оригинал остаётся там, где дубля нет
        else:
            gain = db_to_gain(ts.gain_db)
        if spk in layers:
            xo = layers[spk]
        elif lay is not None:
            continue                                            # у персонажа нет реплик в сцене — его голос уже в «прочем»
        else:
            xo, so = read_wav(orig_files[spk])
            xo = to_stereo(_resample(xo, so, sr))
        m = min(n, len(xo))
        if not voiceover and spk in dubbed:
            rm = span_mask(m, sr, replaced.get(spk, []))         # 1 — реплика заменена дублем
            rm = np.maximum(rm, rm_all[:m])
            out[:m] += xo[:m] * (1.0 - rm)[:, None]
            if mix.keep_breaths and spk in dub_mask and rm.any():
                b = _breaths(to_mono(xo[:m]), sr, dub_mask[spk][:m])
                out[:m] += (b * rm)[:, None]
        elif gain > 0.0:
            out[:m] += xo[:m] * (np.float32(gain) * (1.0 - rm_all[:m]))[:, None]

    if report is not None:
        report["voice_over_bg_db"] = _bed_vs_voice(out, voice_bus, sr)
    # новые голоса поверх (в центр)
    out += voice_bus[:, None]
    if report is not None:
        report["mix_lufs"] = _lufs(out, sr)
    return write_wav(out_wav, out, sr)


def drop_stale_tracks(scene: Scene) -> MixSettings:
    """Сбросить усиление у персонажей, у которых в сведении нет дубля (удалили, запись пустая): без звука и без чтения
    файлов. Нужно после удаления дубля: ползунок не должен хранить старое «−7,8 дБ» у «ещё не озвучен».
    Ручные поля (mute/solo/room/voice_fx) не трогаем. Возвращает новый MixSettings."""
    voiced = {spk for _t, _l, spk in _pick_takes(scene)}
    tracks = {}
    for spk, ts in scene.mix.tracks.items():
        tracks[spk] = ts if spk in voiced or (ts.gain_db == 0.0 and not ts.boost_capped) else ts.model_copy(
            update={"gain_db": 0.0, "boost_capped": False})
    return scene.mix.model_copy(update={"tracks": tracks})


# ─────────────────────────── автопилот ───────────────────────────
def automix(scene: Scene, stems: SceneStems) -> MixSettings:
    """Автопилот: сцена должна звучать как в фильме, только с новыми голосами.

    · дубль каждого персонажа — к громкости ЕГО оригинального голоса (нет оригинала — −20 LUFS);
    · оставшиеся оригинальные голоса и фон — 0 дБ, как в исходнике;
    · фон приглушается только если речь иначе тонет (меньше BG_MARGIN_DB над фоном); громче фон не делаем,
      «приседание» фона под голосом включаем только когда оно нужно;
    · комнату угадываем по фону.
    Ручные поля (mute/solo/voice_fx и явно выбранная комната) не трогаем.
    Если scene.mix.autopilot=False — вернуть scene.mix как есть."""
    mix = scene.mix
    if not mix.autopilot:
        return mix
    n, sr = _bg_info(stems.background)
    tracks = {k: v.model_copy() for k, v in mix.tracks.items()}

    # комната: угадываем один раз по фону и сохраняем конкретное имя там, где стояло "auto"
    room = guess_room(stems.background)
    dubbed = list(scene.dubbed_speakers)
    spk_in_takes = {spk for _, _, spk in _pick_takes(scene)}
    for spk in list(dict.fromkeys(dubbed + sorted(spk_in_takes))):
        ts = tracks.setdefault(spk, TrackSettings())
        if ts.room == "auto":
            ts.room = room

    # уровни новых голосов (измеряем то, что реально получится: clean → fx → stretch → room).
    # Цель — громкость ОРИГИНАЛЬНОГО голоса этого персонажа: сцена должна звучать как в фильме, а не «по стандарту».
    probe_mix = mix.model_copy(update={"tracks": tracks})
    rep: dict[str, dict] = {}
    dubs = _dub_tracks(scene, stems, probe_mix, sr, n, report=rep)
    orig_lufs: dict[str, float | None] = {}
    for spk, path in stems.per_speaker.items():
        orig_lufs[spk] = _speech_lufs(_read_mono(Path(path), sr), sr) if Path(path).exists() else None
    speech_bus = np.zeros(n, dtype=np.float32)
    for spk, y in dubs.items():
        lufs = _speech_lufs(y, sr)
        if lufs is None:
            continue
        target = orig_lufs.get(spk) or SPEECH_TARGET_LUFS
        ts = tracks.setdefault(spk, TrackSettings())
        need = float(target - lufs)
        # шум дубля (после очистки) не должен вылезти из-под фона вместе с усилением: потолок по шуму, но не ниже MIN_BOOST_DB
        noise = rep.get(spk, {}).get("noise_db")
        cap = MAX_BOOST_DB if noise is None else float(np.clip(NOISE_CEIL_DB - noise, MIN_BOOST_DB, MAX_BOOST_DB))
        g = float(np.clip(need, -30.0, cap))
        ts.boost_capped = need - g > BOOST_CAP_TOL_DB
        ts.gain_db = round(g, 2)
        speech_bus += y * np.float32(db_to_gain(ts.gain_db))

    # у персонажа нет дублей в сведении (удалили, пустая запись) — его дорожка не должна хранить старое усиление
    voiced = {spk for spk, y in dubs.items() if _speech_lufs(y, sr) is not None}
    for spk, ts in tracks.items():
        if spk not in voiced:
            ts.gain_db, ts.boost_capped = 0.0, False

    # оригинальные голоса, которые остаются (в DUB), — как в фильме: 0 дБ
    if mix.mode == MixMode.DUB:
        for spk in stems.per_speaker:
            if spk not in dubbed:
                tracks.setdefault(spk, TrackSettings()).gain_db = 0.0

    # фон: как в фильме (0 дБ). Приглушаем, только если речь тонет: тише фона + BG_MARGIN_DB. Громче не делаем.
    bg_ts = mix.background.model_copy()
    bg, _ = read_wav(stems.background)
    extra = _unassigned_voices(scene, stems, sr, n)
    if extra is not None and len(bg) == n:
        bg = to_stereo(bg) + extra                               # мерим тот фон, что реально звучит в сведении
    bgm = to_mono(bg)
    act = _activity(speech_bus, sr, lookahead=0.0) > 0.5 if np.abs(speech_bus).max() > 1e-6 else None
    if act is None or act.sum() < int(0.5 * sr):
        # новых голосов ещё нет — ориентируемся на речь оригинала
        voices = _read_mono(Path(stems.voices), sr) if Path(stems.voices).exists() else None
        act = (_activity(voices, sr, lookahead=0.0) > 0.5) if voices is not None else None
        if act is not None and len(act) != n:
            act = None
    ref = bgm[act[: len(bgm)]] if act is not None and act.sum() >= int(0.5 * sr) else bgm
    import pyloudnorm as pyln
    if len(ref) < int(0.45 * sr):
        ref = np.concatenate([ref, np.zeros(int(0.45 * sr) - len(ref), dtype=np.float32)])
    l_bg = float(pyln.Meter(sr).integrated_loudness(ref.astype(np.float32)))
    l_speech = _speech_lufs(speech_bus, sr) if np.abs(speech_bus).max() > 1e-6 else None
    if l_speech is None:
        known = [v for v in orig_lufs.values() if v is not None]
        l_speech = max(known) if known else SPEECH_TARGET_LUFS
    duck_bg = mix.duck_background
    if np.isfinite(l_bg) and l_bg > -70.0:
        margin = l_speech - l_bg                                 # насколько речь выше фона без всяких правок
        need = BG_MARGIN_DB - margin
        if need <= 0:
            bg_ts.gain_db, duck_bg = 0.0, False                  # и так разборчиво — фон как в фильме, без «приседаний»
        else:
            duck_bg = True                                       # речь тонет: фон приседает под голосом и, если мало, тише
            bg_ts.gain_db = round(float(np.clip(-(need - DUCK_DB), -18.0, 0.0)), 2)
    else:
        bg_ts.gain_db = 0.0
    upd: dict = {"tracks": tracks, "background": bg_ts, "duck_background": duck_bg}
    if mix.loudness_target == "film":            # «как в фильме»: целевая громкость = громкость оригинала сцены
        from .master import film_target_lufs
        t = film_target_lufs(Path(stems.scene_dir) / "original.wav")
        if t is not None:
            upd["target_lufs"] = round(t, 2)
    return mix.model_copy(update=upd)


# ─────────────────────────── экспорт дорожек ───────────────────────────
def export_stems(scene: Scene, stems: SceneStems, out_dir) -> dict[str, Path]:
    """RenderSettings.export_stems: отдельные WAV — фон (с усилением из сведения) и каждый новый голос.

    Файлы: background.wav (stereo), dub_<speaker_id>.wav (mono). Без мастеринга — это сырьё для дальнейшей работы.
    """
    mix = scene.mix
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bg, sr = read_wav(stems.background)
    bg = to_stereo(bg)
    files: dict[str, Path] = {}
    g_bg = np.float32(db_to_gain(mix.background.gain_db))
    files["background"] = write_wav(out_dir / "background.wav", bg * g_bg, sr)
    for spk, y in _dub_tracks(scene, stems, mix, sr, len(bg)).items():
        ts = mix.tracks.get(spk, TrackSettings())
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in spk) or "voice"
        files[f"dub_{spk}"] = write_wav(out_dir / f"dub_{safe}.wav", y * np.float32(db_to_gain(ts.gain_db)), sr)
    return files
