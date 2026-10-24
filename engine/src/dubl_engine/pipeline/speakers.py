"""Слияние голосов со снимком для отмены."""
from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime

from ..models import Project
from ..store import NotFoundError, ProjectStore
from ..store.atomic import atomic_write_text
from ..util import ConflictError, EngineError

log = logging.getLogger(__name__)

MAX_KEPT_MERGES = 20


def _merges_dir(store: ProjectStore, pid: str):
    return store.project_dir(pid) / "merges"


def _recount(an, sid: str) -> None:
    sp = next(s for s in an.speakers if s.id == sid)
    mine = [ln for ln in an.lines if ln.speaker_id == sid]
    sp.line_count = len(mine)
    sp.total_speech = round(sum(max(ln.end - ln.start, 0.0) for ln in mine), 3)


def merge_speakers(store: ProjectStore, pid: str, a: str, b: str) -> tuple[Project, str]:
    """Голос b вливается в a. Возвращает (проект, merge_id). Снимок того, что поменялось, — в merges/<merge_id>.json."""
    if a == b:
        raise EngineError("Выбери двух разных говорящих.")
    with store.edit(pid) as p:
        an = p.analysis
        if an is None:
            raise EngineError("Проект ещё не разобран.")
        by_id = {s.id: s for s in an.speakers}
        if a not in by_id or b not in by_id:
            raise NotFoundError("Такого говорящего нет.")
        snap: dict = {
            "a": a, "b": b, "created": datetime.now().astimezone().isoformat(),
            "speaker_a": by_id[a].model_dump(mode="json"), "speaker_b": by_id[b].model_dump(mode="json"),
            "speaker_b_index": [s.id for s in an.speakers].index(b),
            "analysis_lines": [ln.id for ln in an.lines if ln.speaker_id == b],
            "suggestions": [list(m) for m in an.merge_suggestions],
            "scenes": {},
        }
        for ln in an.lines:
            if ln.speaker_id == b:
                ln.speaker_id = a
        an.speakers = [s for s in an.speakers if s.id != b]
        _recount(an, a)
        an.merge_suggestions = [m for m in an.merge_suggestions if b not in m[:2]]
        for sc in p.scenes:
            snap["scenes"][sc.id] = {
                "lines": [sl.line_id for sl in sc.lines if sl.speaker_id == b],
                "takes": [t.id for t in sc.takes if t.speaker_id == b],
                "dubbed": list(sc.dubbed_speakers),
                "track_b": sc.mix.tracks[b].model_dump(mode="json") if b in sc.mix.tracks else None,
            }
            for sl in sc.lines:
                if sl.speaker_id == b:
                    sl.speaker_id = a
            for t in sc.takes:
                if t.speaker_id == b:
                    t.speaker_id = a
            sc.dubbed_speakers = list(dict.fromkeys(a if s == b else s for s in sc.dubbed_speakers))
            sc.mix.tracks.pop(b, None)
        merge_id = f"m{uuid.uuid4().hex[:8]}"
        d = _merges_dir(store, pid)
        d.mkdir(parents=True, exist_ok=True)
        atomic_write_text(d / f"{merge_id}.json", json.dumps(snap, ensure_ascii=False))
        old = sorted(d.glob("m*.json"), key=lambda f: f.stat().st_mtime)
        for f in old[:-MAX_KEPT_MERGES]:
            f.unlink(missing_ok=True)
    return store.get(pid), merge_id


def undo_merge(store: ProjectStore, pid: str, merge_id: str) -> Project:
    """Откат слияния: голос возвращается с прежним именем, его реплики, дубли, настройки дорожки — на месте.
    Если после слияния голос a уже влили в третий (или b вернули) — сначала нужно отменить более позднее слияние (409)."""
    from ..models import Speaker, TrackSettings

    if not merge_id.isalnum():
        raise NotFoundError("Такого слияния нет или оно уже отменено.")
    path = _merges_dir(store, pid) / f"{merge_id}.json"
    with store.edit(pid) as p:
        try:
            snap = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            raise NotFoundError("Такого слияния нет или оно уже отменено.") from None
        an = p.analysis
        a, b = snap["a"], snap["b"]
        if an is None or a not in {s.id for s in an.speakers} or b in {s.id for s in an.speakers}:
            raise ConflictError("Это слияние уже нельзя отменить: сначала отмени более позднее слияние этих голосов.")
        ids = set(snap["analysis_lines"])
        for ln in an.lines:
            if ln.id in ids and ln.speaker_id == a:
                ln.speaker_id = b
        an.speakers.insert(min(int(snap["speaker_b_index"]), len(an.speakers)), Speaker.model_validate(snap["speaker_b"]))
        _recount(an, a)
        _recount(an, b)
        an.merge_suggestions = [(x[0], x[1], float(x[2])) for x in snap["suggestions"]]
        for sc in p.scenes:
            s = snap["scenes"].get(sc.id)
            if s is None:
                continue
            lines, takes = set(s["lines"]), set(s["takes"])
            for sl in sc.lines:
                if sl.line_id in lines and sl.speaker_id == a:
                    sl.speaker_id = b
            for t in sc.takes:
                if t.id in takes and t.speaker_id == a:
                    t.speaker_id = b
            sc.dubbed_speakers = list(s["dubbed"])
            if s["track_b"] is not None:
                sc.mix.tracks[b] = TrackSettings.model_validate(s["track_b"])
        path.unlink(missing_ok=True)
    return store.get(pid)
