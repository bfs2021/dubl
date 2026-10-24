"""Что происходит с дублем после записи: пустой ли он, что в нём сказано, какой дубль получает ★, обрезка хвоста.

Правила ★ для одной реплики (дубли с одним line_id):
* дубль, который человек отметил сам (chosen_by_user), остаётся ★, пока человек его не снял;
* иначе ★ получает самый полный непустой дубль: сказан целиком > полнота неизвестна > сказан не целиком; при равенстве —
  больше совпавших слов, потом более свежий;
* пустой дубль (empty) ★ не получает никогда.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from ..analysis import takecheck
from ..models import Scene, SceneLine, Take
from ..settings import AppSettings
from ..store import ProjectStore
from ..util import ConflictError

log = logging.getLogger(__name__)

EMPTY_TAKE_MSG = "Запись почти пустая — проверь, что выбран нужный микрофон и он не выключен."
EMPTY_PEAK_DB = -40.0         # нет речи и пик тише — запись пустая (шум комнаты у микрофона обычно −50…−35, но без «всплесков»)
EMPTY_SPEECH_RMS_DB = -50.0   # речь найдена, но тише этого (RMS по самим фразам) — не слышно ничего
CUT_FADE_MS = 15.0            # затухание на месте обрезки: без щелчка


# ───────────────────────── пустая запись ─────────────────────────
def empty_reason(x: np.ndarray, sr: int) -> str | None:
    """Запись почти тишина → короткое объяснение для отладки; None — запись живая.
    Тихая, но настоящая речь (пик −35…−25 дБ, RMS фраз −45…−35) проходит."""
    from ..audio.fit import detect_speech

    peak_db = float(20 * np.log10(max(float(np.max(np.abs(x))) if len(x) else 0.0, 1e-9)))
    speech = detect_speech(x, sr)
    if not speech:
        return f"peak={peak_db:.1f} dBFS, речь не найдена" if peak_db < EMPTY_PEAK_DB else None
    chunks = [x[int(a * sr):int(b * sr)] for a, b in speech]
    sq = float(sum(float(np.sum(c.astype(np.float64) ** 2)) for c in chunks))
    n = sum(len(c) for c in chunks)
    rms_db = float(10 * np.log10(max(sq / max(n, 1), 1e-18)))
    if rms_db < EMPTY_SPEECH_RMS_DB:
        return f"peak={peak_db:.1f} dBFS, rms речи={rms_db:.1f} dBFS"
    return None


# ───────────────────────── разбор по тексту ─────────────────────────
def analyze_take_text(wav: Path, line: SceneLine | None, settings: AppSettings, source_lang: str | None
                      ) -> dict | None:
    """Распознать дубль и сравнить с текстом реплики → поля для Take (recognized_text, text_match, complete,
    tail_cut_s, tail_text). None — разбор невозможен: нет реплики, текст реплики не на языке перевода (перевода нет),
    нет нейросетей или модели на диске. Это не ошибка: дубль сохраняется без разбора."""
    if line is None or not (line.text or "").strip():
        return None
    lang = (settings.target_lang or "ru").split("-")[0].lower()
    if not takecheck.comparable(line.text, line.original, lang, source_lang):
        return None
    model = takecheck.pick_model(settings.whisper_model, settings.processing_device)
    if model is None:
        return None
    words = takecheck.recognize_words(wav, language=lang, model=model, device=settings.processing_device)
    if words is None:
        return None
    r = takecheck.analyze_text(line.text, words)
    return {"recognized_text": r.recognized, "text_match": r.text_match, "complete": r.complete,
            "tail_cut_s": r.tail_cut_s, "tail_text": r.tail_text}


# ───────────────────────── ★ ─────────────────────────
def _rank(t: Take) -> tuple:
    state = 1 if t.complete is None else (2 if t.complete else 0)
    return state, t.text_match if t.text_match is not None else 0.0, t.created


def pick_best(takes: list[Take]) -> Take | None:
    """Самый полный непустой дубль из группы; None — все пустые."""
    live = [t for t in takes if not t.empty]
    return max(live, key=_rank) if live else None


def apply_default_star(scene: Scene, line_id: str | None) -> None:
    """Расставить ★ среди дублей одной реплики (line_id=None — дубли всей сцены) по правилам выше. Правит сцену на месте."""
    group = [t for t in scene.takes if t.line_id == line_id]
    if not group:
        return
    chosen = next((t for t in group if t.selected and t.chosen_by_user and not t.empty), None)
    best = chosen or pick_best(group)
    for t in group:
        t.selected = t is best
        if t is not best:
            t.chosen_by_user = False


def set_user_choice(scene: Scene, take: Take, selected: bool) -> None:
    """Человек нажал ★ (или снял её). Пустой дубль в ролик взять нельзя."""
    if selected:
        if take.empty:
            raise ConflictError("В этом дубле почти нет звука — в ролик его взять нельзя. Запиши реплику ещё раз.")
        for o in scene.takes:
            if o.line_id == take.line_id and o.id != take.id:
                o.selected, o.chosen_by_user = False, False
        take.selected, take.chosen_by_user = True, True
    else:
        take.selected, take.chosen_by_user = False, False


# ───────────────────────── хвост-перезапуск ─────────────────────────
def _full_path(store: ProjectStore, pid: str, scid: str, tid: str) -> Path:
    p = store.take_path(pid, scid, tid)
    return p.with_name(f"{p.stem}.full.wav")


def _refit(store: ProjectStore, pid: str, scid: str, scene: Scene, take: Take, settings: AppSettings) -> None:
    """После обрезки хвоста конец речи другой — подгонку (сдвиг и длину) считаем заново."""
    line = next((ln for ln in scene.lines if ln.line_id == take.line_id), None)
    if line is None:
        return
    from ..audio.fit import fit_take
    sd = store.scene_dir(pid, scid)
    guide = sd / "voices" / f"{line.speaker_id}.wav" if line.speaker_id else sd / "voices.wav"
    if not guide.exists():
        guide = sd / "voices.wav"
    try:
        fit = fit_take(sd / take.file, guide, line_start_rel=line.start_rel, line_end_rel=line.end_rel,
                       take_start_rel=take.start_rel, preroll=take.preroll,
                       latency_ms=settings.latency_ms if take.source == "mic" else 0.0)
    except Exception as e:  # noqa: BLE001 — не вышло: оставляем прежнюю подгонку
        log.warning("Подгонка после обрезки не пересчиталась: %s", e)
        return
    take.offset_ms, take.stretch, take.fit_confidence = fit.offset_ms, fit.stretch, fit.confidence


def trim_tail(store: ProjectStore, pid: str, scid: str, tid: str, settings: AppSettings) -> Take:
    """Отрезать хвост-перезапуск («давай ещё раз») по Take.tail_cut_s. Полный файл остаётся рядом — untrim_tail вернёт."""
    from ..audio.io import read_wav, write_wav
    with store.edit(pid) as p:
        sc = p.scenes[store.scene_index(p, scid)]
        t = store.find_take(sc, tid)
        if t.tail_cut_s is None or t.tail_trimmed:
            raise ConflictError("В этом дубле нечего отрезать.")
        path = store.take_path(pid, scid, tid)
        full = _full_path(store, pid, scid, tid)
        if not full.exists():
            path.replace(full)
        x, sr = read_wav(full)
        n = max(int(t.tail_cut_s * sr), int(0.1 * sr))
        y = np.array(x[:n], dtype=np.float32)
        k = min(int(sr * CUT_FADE_MS / 1000), len(y) // 2)
        if k > 0:
            ramp = np.linspace(1.0, 0.0, k, dtype=np.float32)
            y[-k:] *= ramp if y.ndim == 1 else ramp[:, None]
        write_wav(path, y, sr)
        t.duration = round(len(y) / sr, 3)
        t.tail_trimmed = True
        _refit(store, pid, scid, sc, t, settings)
        return t.model_copy(deep=True)


def untrim_tail(store: ProjectStore, pid: str, scid: str, tid: str, settings: AppSettings) -> Take:
    """Вернуть полный дубль вместе с хвостом."""
    from ..audio.io import read_wav
    with store.edit(pid) as p:
        sc = p.scenes[store.scene_index(p, scid)]
        t = store.find_take(sc, tid)
        full = _full_path(store, pid, scid, tid)
        if not t.tail_trimmed or not full.exists():
            raise ConflictError("Хвост не отрезан — возвращать нечего.")
        path = store.take_path(pid, scid, tid)
        full.replace(path)
        x, sr = read_wav(path)
        t.duration = round(len(x) / sr, 3)
        t.tail_trimmed = False
        _refit(store, pid, scid, sc, t, settings)
        return t.model_copy(deep=True)


def delete_take_files(store: ProjectStore, pid: str, scid: str, tid: str) -> None:
    store.take_path(pid, scid, tid).unlink(missing_ok=True)
    _full_path(store, pid, scid, tid).unlink(missing_ok=True)
