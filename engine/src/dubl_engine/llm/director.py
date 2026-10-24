"""Помощник-режиссёр. Не слышит звук: работает по замерам дубля и выполняет просьбы словами через настройки сведения."""
from __future__ import annotations

import difflib
import logging
import math
import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from ..models import MixMode, MixSettings, Scene, SceneLine, Speaker, Take, TrackSettings
from ..util import EngineError
from . import prompts
from .providers import LLMProvider, extract_json

log = logging.getLogger(__name__)

# ─────────────────────────── допустимые значения ───────────────────────────
_ROOMS_FALLBACK = ["dry", "room", "kitchen", "hall", "street", "cave", "car", "bathroom", "phone", "behind_door"]
_FX_FALLBACK = ["none", "robot", "chipmunk", "giant", "villain", "radio", "phone", "ghost"]


def available_rooms() -> list[str]:
    """Пресеты комнат: из audio.rooms.ROOMS, если модуль есть, иначе локальный список с теми же ключами."""
    try:
        from ..audio.rooms import ROOMS  # type: ignore[import-not-found]
        keys = [str(k) for k in ROOMS]
        if keys:
            return keys
    except ImportError:  # модуль не импортируется — локальный список с теми же ключами
        log.debug("audio.rooms недоступен")
    return list(_ROOMS_FALLBACK)


def available_fx() -> list[str]:
    try:
        from ..audio.fx import VOICE_FX  # type: ignore[import-not-found]
        keys = [str(k) for k in VOICE_FX]
        if keys:
            return keys
    except ImportError:
        log.debug("audio.fx недоступен")
    return list(_FX_FALLBACK)


GAIN_MIN, GAIN_MAX = -30.0, 12.0
LUFS_MIN, LUFS_MAX = -30.0, -6.0

MixField = Literal["gain_db", "room", "voice_fx", "mute", "duck_background", "target_lufs", "mode"]


class MixPatch(BaseModel):
    """Одно изменение настроек сведения. value — всегда абсолютное (относительные просьбы уже пересчитаны)."""
    model_config = ConfigDict(json_schema_serialization_defaults_required=True)   # в схеме ответа note обязателен

    target: str                     # "background" | speaker_id | "master"
    field: MixField
    value: float | bool | str
    note: str = ""                  # что сделано, по-человечески: «Барсик: громкость +3 дБ» (заполняет director_command)


def _titles() -> tuple[dict[str, str], dict[str, str]]:
    try:
        from ..audio.fx import VOICE_FX_TITLES
        from ..audio.rooms import ROOM_TITLES
        return dict(ROOM_TITLES), dict(VOICE_FX_TITLES)
    except ImportError:
        return {}, {}


def describe_patch(patch: MixPatch, speakers: list[Speaker] | None = None) -> str:
    """Изменение словами для интерфейса: «Барсик: громкость +3 дБ», «Фон: выключен», «Итог: −16 LUFS»."""
    rooms, fx = _titles()
    by_id = {s.id: s for s in speakers or []}
    if patch.target == "background":
        who = "Фон"
    elif patch.target == "master":
        who = "Итог"
    else:
        sp = by_id.get(patch.target)
        who = (sp.name or sp.label) if sp else patch.target
    v = patch.value
    if patch.field == "gain_db":
        f = float(v) if not isinstance(v, str) else 0.0
        return f"{who}: громкость {f:+g} дБ"
    if patch.field == "mute":
        return f"{who}: выключен" if v else f"{who}: включён"
    if patch.field == "room":
        return f"{who}: пространство подбирается само" if v == "auto" else f"{who}: пространство «{rooms.get(str(v), v)}»"
    if patch.field == "voice_fx":
        return f"{who}: без эффекта" if v == "none" else f"{who}: эффект «{fx.get(str(v), v)}»"
    if patch.field == "target_lufs":
        return f"Итог: {float(v):g} LUFS"
    if patch.field == "duck_background":
        return "Фон приглушается под голосом" if v else "Фон не приглушается"
    if patch.field == "mode":
        return "Режим: закадровый голос" if v == "voiceover" else "Режим: дубляж"
    return f"{who}: {patch.field} = {v}"


# ─────────────────────────── применение патчей ───────────────────────────
def _as_bool(v: Any) -> bool | None:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)) and v in (0, 1):
        return bool(v)
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("true", "yes", "да", "1", "on", "вкл"):
            return True
        if s in ("false", "no", "нет", "0", "off", "выкл"):
            return False
    return None


def _as_float(v: Any) -> float | None:
    if isinstance(v, bool):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _normalize(target: str, field: str, value: Any, speaker_ids: set[str] | None) -> Any:
    """Проверить пару (цель, поле, значение). Вернуть очищенное значение или None, если патч недопустим."""
    is_speaker = target not in ("background", "master") and (speaker_ids is None or target in speaker_ids)
    if target == "master":
        allowed = {"duck_background", "target_lufs", "mode"}
    elif target == "background":
        allowed = {"gain_db", "mute"}
    elif is_speaker:
        allowed = {"gain_db", "mute", "room", "voice_fx"}
    else:
        return None
    if field not in allowed:
        return None
    if field == "gain_db":
        f = _as_float(value)
        return None if f is None else round(_clamp(f, GAIN_MIN, GAIN_MAX), 2)
    if field == "target_lufs":
        f = _as_float(value)
        return None if f is None else round(_clamp(f, LUFS_MIN, LUFS_MAX), 2)
    if field in ("mute", "duck_background"):
        return _as_bool(value)
    if field == "room":
        v = str(value).strip().lower()
        return v if v == "auto" or v in available_rooms() else None
    if field == "voice_fx":
        v = str(value).strip().lower()
        return v if v in available_fx() else None
    if field == "mode":
        v = str(value).strip().lower()
        return v if v in (MixMode.DUB.value, MixMode.VOICEOVER.value) else None
    return None


def apply_patches(mix: MixSettings, patches: list[MixPatch]) -> MixSettings:
    """Новые MixSettings с применёнными изменениями. Недопустимые патчи пропускаются.

    Если что-то изменилось, autopilot выключается: иначе автосведение затёрло бы просьбу пользователя.
    """
    out = mix.model_copy(deep=True)
    changed = False
    for p in patches:
        val = _normalize(p.target, p.field, p.value, None)
        if val is None:
            continue
        changed = True
        if p.target == "master":
            if p.field == "duck_background":
                out.duck_background = bool(val)
            elif p.field == "target_lufs":
                out.target_lufs = float(val)
            elif p.field == "mode":
                out.mode = MixMode(val)
            continue
        if p.target == "background":
            track = out.background
        else:
            track = out.tracks.get(p.target) or TrackSettings()
            out.tracks[p.target] = track
        setattr(track, p.field, val)
    if changed:
        out.autopilot = False
    return out


# ─────────────────────────── замеры дубля → факты ───────────────────────────
TIMING_OK_MS = 150.0           # до такого сдвига — «в губы»
TIMING_BIG_MS = 400.0
STRETCH_OK = 0.04
STRETCH_BIG = 0.10
CLIP_PEAK_DB = -1.0
NO_MATCH_CONF = 0.0        # подгонка не нашла речь дубля — сдвиг 0 ничего не значит
FIT_SURE = 0.4             # подгонка ниже этого не уверена — про губы молчим
QUIET_PEAK_DB = -24.0
QUIET_LUFS = -32.0
TEXT_GOOD = 0.9
TEXT_PARTIAL = 0.7

_WORD = re.compile(r"[\w'-]+", re.UNICODE)


def _words(text: str) -> list[str]:
    return [w.strip("'-") for w in _WORD.findall(text.casefold().replace("ё", "е"))]


def _level(x: float, ok: float, big: float) -> str:
    return "ok" if x <= ok else ("noticeable" if x <= big else "big")


def take_report(take: Take, line: SceneLine) -> dict:
    """Замеры дубля → факты для помощника. Знак offset_ms — как в Take: минус = подгонка сдвинула дубль раньше,
    значит человек говорил позже губ; stretch < 1 — дубль пришлось сжать (говорил дольше оригинала), > 1 — растянуть.
    """
    problems: list[str] = []
    report: dict[str, Any] = {"line_id": line.line_id, "take_id": take.id}

    from_file = getattr(take, "source", "mic") == "file"      # дубль из файла: губы тут ни при чём, про «спешишь» не говорим
    report["source"] = "file" if from_file else "mic"

    # время
    off = float(take.offset_ms)
    if off <= -TIMING_OK_MS:
        timing = "late"
    elif off >= TIMING_OK_MS:
        timing = "early"
    else:
        timing = "ok"
    report["offset_ms"] = round(off)
    report["timing"] = timing
    report["timing_level"] = _level(abs(off), TIMING_OK_MS, TIMING_BIG_MS)
    if timing != "ok" and not from_file:
        problems.append(timing)

    # длина
    st = float(take.stretch)
    dev = abs(st - 1.0)
    length = "ok" if dev <= STRETCH_OK else ("too_long" if st < 1 else "too_short")
    report["stretch"] = round(st, 3)
    report["length"] = length
    report["length_level"] = _level(dev, STRETCH_OK, STRETCH_BIG)
    if length != "ok" and not from_file:
        problems.append(length)
    if take.fit_confidence is not None:
        report["fit_confidence"] = round(take.fit_confidence, 2)
        report["fit_unsure"] = take.fit_confidence < FIT_SURE
        if take.fit_confidence <= NO_MATCH_CONF:        # речь дубля в окне реплики не нашли: «0%» — это не «в губы»
            problems.append("no_match")
            report["timing"], report["timing_level"] = "ok", "ok"
    # «попал в губы» можно говорить, только если это проверено: дубль с микрофона и подгонка уверена. Файл к видео не
    # привязан (offset — просто куда подгонка его сдвинула), а без уверенной подгонки сдвиг ничего не значит.
    report["lips_checked"] = (not from_file and take.fit_confidence is not None
                              and take.fit_confidence >= FIT_SURE)

    # текст
    report["text_match"] = None
    report["missing_words"] = []
    if take.recognized_text is not None and (line.text or "").strip():
        want, got = _words(line.text), _words(take.recognized_text)
        ratio = difflib.SequenceMatcher(None, " ".join(want), " ".join(got)).ratio() if want or got else 1.0
        sm = difflib.SequenceMatcher(None, want, got)
        missing: list[str] = []
        for op, a0, a1, _b0, _b1 in sm.get_opcodes():
            if op in ("delete", "replace"):
                missing += want[a0:a1]
        report["text_ratio"] = round(ratio, 2)
        report["missing_words"] = missing[:6]
        if not got:
            tm = "silent"
        elif ratio >= TEXT_GOOD:
            tm = "ok"
        elif ratio >= TEXT_PARTIAL:
            tm = "partial"
        else:
            tm = "mismatch"
        report["text_match"] = tm
        if tm != "ok":
            problems.append("text_" + tm)

    # громкость
    pk, lufs = take.peak_db, take.loudness_lufs
    report["peak_db"] = None if pk is None else round(pk, 1)
    report["loudness_lufs"] = None if lufs is None else round(lufs, 1)
    loud = "ok"
    if pk is not None and pk > CLIP_PEAK_DB:
        loud = "clipping"
    elif (pk is not None and pk < QUIET_PEAK_DB) or (lufs is not None and lufs < QUIET_LUFS):
        loud = "too_quiet"
    report["loudness"] = loud
    if loud != "ok":
        problems.append(loud)

    report["problems"] = problems
    report["ok"] = not problems
    return report


# ─────────────────────────── совет без ИИ ───────────────────────────
_ADVICE: dict[str, str] = {
    "no_match": "Я не нашёл твой голос на месте этой реплики — проверь микрофон и запиши ещё раз, начиная вместе с видео.",
    "clipping": "Звук перегружен — отодвинься от микрофона или говори чуть спокойнее.",
    "too_quiet": "Тебя еле слышно — подойди ближе к микрофону или говори увереннее.",
    "text_silent": "Я не разобрал слов в записи — проверь микрофон и запиши ещё раз.",
    "text_mismatch": "Текст заметно ушёл от сценария — перечитай реплику перед записью.",
    "text_partial": "Часть слов отличается от сценария — читай ближе к тексту.",
    "late": "Ты чуть отстаёшь от губ — начни фразу немного раньше.",
    "early": "Ты начинаешь чуть раньше губ — подожди мгновение перед началом.",
    "too_long": "Реплика выходит длиннее оригинала — говори чуть бодрее или сократи текст.",
    "too_short": "Получилось быстрее оригинала — можно говорить спокойнее, не тараторить.",
}
_ADVICE_BIG: dict[str, str] = {
    "late": "Ты сильно отстаёшь от губ — начинай фразу заметно раньше, прямо на вдохе.",
    "early": "Ты начинаешь заметно раньше губ — выжди паузу и говори, когда губы только двинутся.",
    "too_long": "Реплика заметно длиннее оригинала — сократи текст кнопкой «Уложить по длине» или говори быстрее.",
    "too_short": "Ты проговорил заметно быстрее оригинала — говори медленнее, с паузами.",
}
_PRAISE = "Отличный дубль: попал в губы, текст и громкость на месте. Можно брать."
_PRAISE_NO_TEXT = "Отличный дубль: попал в губы, громкость в порядке. Можно брать."
# дубль из файла: к видео не привязан, попадание в губы не проверялось
_PRAISE_FILE = "Дубль загружен файлом: текст и громкость на месте. Попадание в губы проверь на превью."
_PRAISE_FILE_NO_TEXT = "Дубль загружен файлом: громкость в порядке. Попадание в губы проверь на превью."
# с микрофона, но подгонка не уверена (или не считалась): про губы честно не знаем
_PRAISE_UNSURE = "Текст и громкость на месте. Попал ли в губы, я не уверен — проверь на превью."
_PRAISE_UNSURE_NO_TEXT = "Громкость в порядке. Попал ли в губы, я не уверен — проверь на превью."
_ADVICE_FILE: dict[str, str] = {
    "no_match": "Не нашёл в файле речь на месте этой реплики — проверь, тот ли это файл, и послушай на превью.",
    "text_silent": "Я не разобрал слов в файле — проверь, что в нём есть голос.",
}


def _praise(report: dict) -> str:
    """Похвала только тому, что проверено: губы — если lips_checked, текст — если его сравнивали."""
    text_ok = report.get("text_match") == "ok"
    if report.get("source") == "file":
        return _PRAISE_FILE if text_ok else _PRAISE_FILE_NO_TEXT
    if report.get("lips_checked") is False:
        return _PRAISE_UNSURE if text_ok else _PRAISE_UNSURE_NO_TEXT
    return _PRAISE if report.get("text_match", "ok") == "ok" else _PRAISE_NO_TEXT       # {} — старый отчёт без полей
_PRIORITY = ["no_match", "clipping", "text_silent", "text_mismatch", "text_partial", "too_quiet", "late", "early",
             "too_long", "too_short"]


def rule_based_feedback(report: dict) -> str:
    """Совет по фактам без ИИ: 1–2 короткие фразы на «ты», без цифр."""
    problems = [p for p in _PRIORITY if p in (report.get("problems") or [])]
    if not problems:
        return _praise(report)
    phrases: list[str] = []
    for p in problems[:2]:
        text = _ADVICE_FILE.get(p, _ADVICE[p]) if report.get("source") == "file" else _ADVICE[p]
        if p in _ADVICE_BIG and report.get("timing_level" if p in ("late", "early") else "length_level") == "big":
            text = _ADVICE_BIG[p]
        if p in ("text_partial", "text_mismatch") and report.get("missing_words"):
            miss = ", ".join(f"«{w}»" for w in report["missing_words"][:2])
            text = text.rstrip(".") + f" (не хватило: {miss})."
        phrases.append(text)
    return " ".join(phrases)


_DIGITS = re.compile(r"\d")


def _llm_view(report: dict) -> dict:
    """Факты для модели — только словами, без чисел (чтобы она не цитировала цифры). Попадание в губы не проверялось
    (дубль из файла или подгонка не уверена) — модель это знает (lips_checked=false), а для файла сдвиг и длину ей
    не даём вовсе: они не про то, как человек говорил."""
    keep = ("timing", "timing_level", "length", "length_level", "text_match", "missing_words", "loudness",
            "problems", "fit_unsure")
    view = {k: report[k] for k in keep if k in report}
    view["source"] = "file" if report.get("source") == "file" else "mic"
    view["lips_checked"] = bool(report.get("lips_checked", True))
    if view["source"] == "file":
        for k in ("timing", "timing_level", "length", "length_level", "fit_unsure"):
            view.pop(k, None)
    return view


_CLAIMS_LIPS = re.compile(r"попал|попада|в губы|синхрон", re.IGNORECASE)


async def director_feedback(report: dict, provider: LLMProvider | None) -> str:
    """1–2 фразы человеческим языком. Нет модели или она ответила плохо — совет по правилам."""
    if provider is None or report.get("ok") or not report.get("problems"):
        return rule_based_feedback(report)
    system, user = prompts.feedback_prompt(_llm_view(report))
    try:
        text = await provider.complete(system, user, max_tokens=200, temperature=0.5)
    except EngineError:
        return rule_based_feedback(report)
    text = " ".join((text or "").split()).strip().strip("«»\"")
    if not text or len(text) > 300 or _DIGITS.search(text):
        return rule_based_feedback(report)
    if not report.get("lips_checked", True) and _CLAIMS_LIPS.search(text):      # губы не проверяли — не хвалим за них
        return rule_based_feedback(report)
    return text


# ─────────────────────────── просьбы словами → изменения настроек ───────────────────────────
def _scene_speaker_ids(scene: Scene) -> list[str]:
    ids: list[str] = []
    for sid in [*scene.dubbed_speakers, *(ln.speaker_id for ln in scene.lines), *scene.mix.tracks]:
        if sid and sid not in ids:
            ids.append(sid)
    return ids


def _describe_targets(ids: list[str], speakers: list[Speaker] | None) -> tuple[str, dict[str, str]]:
    by_id = {s.id: s for s in speakers or []}
    lines = ["background — музыка и шумы (фон)", "master — вся сцена целиком (общая громкость, режим, приглушение фона)"]
    alias: dict[str, str] = {}
    for sid in ids:
        sp = by_id.get(sid)
        names = [n for n in ((sp.name if sp else None), (sp.label if sp else None)) if n]
        lines.append(f"{sid} — голос персонажа" + (f": {' / '.join(dict.fromkeys(names))}" if names else ""))
        for n in names:
            alias[n.casefold()] = sid
    return "\n".join(lines), alias


def _scene_characters(ids: list[str], speakers: list[Speaker] | None) -> list[tuple[str, str]]:
    """Персонажи сцены с понятными именами: [(id, имя)]. Безымянные («Голос 1» и т. п.) тоже годятся как имя для примера."""
    by_id = {s.id: s for s in speakers or []}
    out: list[tuple[str, str]] = []
    for sid in ids:
        sp = by_id.get(sid)
        name = ((sp.name or sp.label) if sp else None) or sid
        out.append((sid, name))
    return out


def command_examples_for(scene: Scene, speakers: list[Speaker] | None = None) -> list[str]:
    """Примеры просьб для подсказки в панели помощника — из персонажей самой сцены («Барсик громче»)."""
    chars = _scene_characters(_scene_speaker_ids(scene), speakers)
    out: list[str] = []
    if chars:
        out.append(f"{chars[0][1]} громче")
    if len(chars) > 1:
        out.append(f"{chars[1][1]} тише")
    out.append("фон потише")
    return out


def _describe_state(mix: MixSettings, ids: list[str]) -> str:
    def t(ts: TrackSettings) -> str:
        return f"gain_db={ts.gain_db:g}, mute={str(ts.mute).lower()}, room={ts.room}, voice_fx={ts.voice_fx}"
    rows = [f"background: {t(mix.background)}",
            (f"master: mode={mix.mode.value}, duck_background={str(mix.duck_background).lower()}, "
            f"target_lufs={mix.target_lufs:g}")]
    rows += [f"{sid}: {t(mix.tracks.get(sid, TrackSettings()))}" for sid in ids]
    return "\n".join(rows)


def _current(mix: MixSettings, target: str, field: str) -> float:
    ts = mix.background if target == "background" else mix.tracks.get(target, TrackSettings())
    return float(getattr(ts, field))


async def director_command(text: str, scene: Scene, provider: LLMProvider | None, *,
                           speakers: list[Speaker] | None = None) -> list[MixPatch]:
    """«кот громче», «как в пещере» → список проверенных изменений MixSettings.

    speakers — чтобы понимать персонажей по имени/метке (в Scene есть только их id).
    Недопустимое (неизвестные цели, поля, значения) отбрасывается; громкость зажимается в допустимый диапазон.
    """
    text = (text or "").strip()
    if not text:
        return []
    if provider is None:
        raise EngineError("Чтобы выполнять просьбы словами, подключи модель ИИ в настройках.")
    ids = _scene_speaker_ids(scene)
    targets, alias = _describe_targets(ids, speakers)
    rooms, fx = available_rooms(), available_fx()
    system, user = prompts.command_prompt(
        text, targets=targets, chars=_scene_characters(ids, speakers),
        rooms=", ".join(f"{r} ({prompts.ROOM_HINTS[r]})" if r in prompts.ROOM_HINTS else r for r in rooms) + ", auto (подобрать самому)",
        fx=", ".join(f"{f} ({prompts.FX_HINTS[f]})" if f in prompts.FX_HINTS else f for f in fx),
        state=_describe_state(scene.mix, ids))
    raw = await provider.complete(system, user, json_schema=prompts.COMMAND_SCHEMA, max_tokens=600, temperature=0.0)
    data = extract_json(raw)
    items = data.get("patches") if isinstance(data, dict) else data
    if not isinstance(items, list):
        raise EngineError("Помощник ответил в неожиданном формате. Попробуй сказать иначе.")

    valid_ids = set(ids)
    work = scene.mix.model_copy(deep=True)      # рабочая копия: относительные правки копятся друг за другом
    result: dict[tuple[str, str], MixPatch] = {}
    for it in items:
        if not isinstance(it, dict):
            continue
        target = str(it.get("target", "")).strip()
        if target not in ("background", "master") and target not in valid_ids:
            target = alias.get(target.casefold(), target)
        field, value = it.get("field"), it.get("value")
        if field == "gain_db" and it.get("relative") is True and target in ({"background"} | valid_ids):
            delta = _as_float(value)
            if delta is None:
                continue
            value = _current(work, target, "gain_db") + delta
        clean = _normalize(target, str(field), value, valid_ids)
        if clean is None:
            continue
        patch = MixPatch(target=target, field=field, value=clean)       # type: ignore[arg-type]
        patch.note = describe_patch(patch, speakers)
        work = apply_patches(work, [patch])
        result[(target, str(field))] = patch
    return list(result.values())
