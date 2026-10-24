"""Правка реплик разбора человеком: сменить персонажа у реплики, склеить соседние реплики в одну, разделить обратно.

Разбор (whisper + определение голосов) ошибается: режет одну фразу на две или отдаёт реплику не тому персонажу.
Правка идёт в разбор проекта (Analysis.lines) и сразу во все сцены, где есть эти реплики; у подготовленных сцен
дорожки персонажей (voices/<id>.wav) пересобираются — они нужны подгонке дублей и прослушиванию.

Каждая склейка (ручная и авто-склейка разрезанной фразы при разборе) оставляет снимок <проект>/line_merges/<id>.json:
реплики до склейки, их строки в сценах и дубли, прежние записи Analysis.merged_lines. Analysis.merged_lines[реплика] —
последняя склейка, в которую она вошла: по ней «Разделить» отменяет склейки по одной, от поздней к ранней.
"""
from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime
from typing import Literal, cast

from ..analysis.lines import merge_group, merge_two
from ..analysis.palette import PALETTE
from ..models import Line, Project, Scene, SceneLine, Speaker
from ..store import NotFoundError, ProjectStore
from ..store.atomic import atomic_write_text
from ..util import ConflictError, EngineError

log = logging.getLogger(__name__)

MAX_KEPT_EDITS = 20
MAX_MERGE = 6               # больше — это уже не «одна фраза разрезалась», а ошибка выделения
MAX_MERGED_S = 20.0
DIFFERENT_SPEAKERS = "Это реплики разных персонажей — сначала смени персонажа у одной из них."
NOT_MERGED = "Эта реплика не склеивалась."


def _edits_dir(store: ProjectStore, pid: str):
    return store.project_dir(pid) / "line_merges"


def _recount(an) -> None:
    for sp in an.speakers:
        mine = [ln for ln in an.lines if ln.speaker_id == sp.id]
        sp.line_count = len(mine)
        sp.total_speech = round(sum(max(ln.end - ln.start, 0.0) for ln in mine), 3)


def _resplit(store: ProjectStore, pid: str, scene: Scene) -> None:
    """Дорожки персонажей подготовленной сцены — по новым репликам. Нет voices.wav (сцена не готова) — нечего делать."""
    if not scene.stems_ready:
        return
    import shutil

    from ..audio.separate import split_by_speaker
    sd = store.scene_dir(pid, scene.id)
    if not (sd / "voices.wav").is_file():
        return
    tmp = sd / ".voices-new"
    shutil.rmtree(tmp, ignore_errors=True)
    split_by_speaker(sd / "voices.wav", scene.lines, tmp)
    shutil.rmtree(sd / "voices", ignore_errors=True)
    tmp.rename(sd / "voices")


def _new_speaker(an, name: str) -> Speaker:
    used = {s.id for s in an.speakers}
    n = 1
    while f"s{n}" in used:
        n += 1
    taken = {s.color for s in an.speakers}
    color = next((c for c in PALETTE if c not in taken), PALETTE[len(an.speakers) % len(PALETTE)])
    sp = Speaker(id=f"s{n}", label=name, name=name, color=color)
    an.speakers.append(sp)
    return sp


def _check_idle(jobs, pid: str) -> None:
    """Пока разбор или подготовка сцены идёт, они перезапишут реплики своими — правку потеряем."""
    if jobs is None:
        return
    busy = [j for j in jobs.active() if j.project_id == pid and j.kind in ("analyze", "prepare_scene", "separate")]
    if busy:
        raise ConflictError("Сейчас идёт разбор или подготовка сцены — подожди, пока закончится, и поправь реплику.")


# ─────────────────────────── персонаж реплики ───────────────────────────
def set_line_speaker(store: ProjectStore, pid: str, line_id: str, speaker_id: str | None = None,
                     new_speaker_name: str | None = None, *, jobs=None) -> Project:
    """Реплика (и её дубли) переходит к другому персонажу. new_speaker_name — завести нового и отдать реплику ему."""
    _check_idle(jobs, pid)
    name = (new_speaker_name or "").strip()
    if not name and not speaker_id:
        raise EngineError("Выбери персонажа.")
    touched: list[Scene] = []
    with store.edit(pid) as p:
        an = p.analysis
        if an is None:
            raise EngineError("Проект ещё не разобран.")
        ln = next((x for x in an.lines if x.id == line_id), None)
        if ln is None:
            raise NotFoundError("Такой реплики нет.")
        if name:
            same = next((s for s in an.speakers if (s.name or s.label).strip().lower() == name.lower()), None)
            sid = same.id if same else _new_speaker(an, name[:100]).id
        else:
            if speaker_id not in {s.id for s in an.speakers}:
                raise NotFoundError("Такого персонажа нет.")
            sid = speaker_id
        old = ln.speaker_id
        if old == sid:
            return p.model_copy(deep=True)
        ln.speaker_id = sid
        _recount(an)
        for sc in p.scenes:
            hit = False
            for sl in sc.lines:
                if sl.line_id == line_id:
                    sl.speaker_id, hit = sid, True
            if not hit:
                continue
            for t in sc.takes:
                if t.line_id == line_id:
                    t.speaker_id = sid
            # озвучивали прежнего персонажа — значит, и реплику нового хотят озвучить
            if (old in sc.dubbed_speakers or not old) and sid not in sc.dubbed_speakers:
                sc.dubbed_speakers.append(sid)
            touched.append(sc)
        for sc in touched:
            try:
                _resplit(store, pid, sc)
            except Exception as e:  # noqa: BLE001 — дорожки персонажей вторичны, реплику всё равно меняем
                log.warning("Дорожки персонажей сцены %s не пересобрались: %s", sc.id, e)
    return store.get(pid)


# ─────────────────────────── склейка ───────────────────────────
def _merged_scene_line(parts: list[SceneLine], merged_text: str) -> SceneLine:
    first = parts[0]
    untouched = all(sl.translation_source is None and sl.text == sl.original for sl in parts)
    if untouched:
        text, src = merged_text, None
    else:
        text = " ".join(sl.text.strip() for sl in parts if sl.text.strip())
        srcs = [sl.translation_source for sl in parts if sl.translation_source]
        src = cast(Literal["file", "llm", "user"] | None, "user" if "user" in srcs else (srcs[0] if srcs else None))
    return SceneLine(line_id=first.line_id, speaker_id=first.speaker_id,
                     start_rel=min(sl.start_rel for sl in parts), end_rel=max(sl.end_rel for sl in parts),
                     original=merged_text, text=text, translation_source=src)


def merge_lines(store: ProjectStore, pid: str, line_ids: list[str], *, jobs=None) -> tuple[Project, str, int]:
    """Склеить соседние реплики в одну. → (проект, merge_id для отмены, сколько дублей склеенных реплик перешло к новой).

    Склеиваются только реплики одного персонажа: иначе дубль второго молча переехал бы к первому (и в ролике снова
    зазвучал бы оригинал). Разные персонажи — 400 «сначала смени персонажа». Между репликами не должно быть чужих
    реплик. Дубли склеенных реплик переходят к новой; ★ остаётся одна — значит, реплику лучше перезаписать.
    Analysis.merged_lines[склеенная] = merge_id — по нему «Разделить» отменяет склейку, не зная merge_id."""
    _check_idle(jobs, pid)
    ids = list(dict.fromkeys(line_ids))
    if len(ids) < 2:
        raise EngineError("Чтобы склеить, нужны хотя бы две реплики.")
    if len(ids) > MAX_MERGE:
        raise EngineError(f"За раз можно склеить не больше {MAX_MERGE} реплик.")
    touched: list[Scene] = []
    with store.edit(pid) as p:
        an = p.analysis
        if an is None:
            raise EngineError("Проект ещё не разобран.")
        order = sorted(an.lines, key=lambda x: (x.start, x.end, x.id))
        pos = {ln.id: i for i, ln in enumerate(order)}
        if any(i not in pos for i in ids):
            raise NotFoundError("Такой реплики нет.")
        idx = sorted(pos[i] for i in ids)
        if idx[-1] - idx[0] != len(idx) - 1:
            raise EngineError("Склеить можно только реплики, которые идут подряд.")
        parts = order[idx[0]: idx[-1] + 1]
        if len({x.speaker_id for x in parts}) > 1:
            raise EngineError(DIFFERENT_SPEAKERS)
        if parts[-1].end - parts[0].start > MAX_MERGED_S:
            raise EngineError(f"Получится реплика длиннее {MAX_MERGED_S:.0f} с — такую трудно записать. Склей меньше.")
        merged = parts[0]
        for nxt in parts[1:]:
            merged = merge_two(merged, nxt)
        merged = merged.model_copy(update={"speaker_id": parts[0].speaker_id})
        part_ids = [x.id for x in parts]
        merge_id = _new_merge_id()
        snap: dict = {"created": datetime.now().astimezone().isoformat(), "keep": part_ids[0],
                      "lines": [x.model_dump(mode="json") for x in parts],
                      "merged": {"start": merged.start, "end": merged.end, "text": merged.text},
                      "scenes": {},
                      # прежние записи карты склеек: отмена вернёт их — цепочка склеек разбирается по шагу
                      "prev_merged": {i: an.merged_lines[i] for i in part_ids if i in an.merged_lines}}
        an.lines = sorted([merged if x.id == part_ids[0] else x for x in an.lines if x.id not in part_ids[1:]],
                          key=lambda x: (x.start, x.end, x.id))
        for i in part_ids:
            an.merged_lines.pop(i, None)
        an.merged_lines[part_ids[0]] = merge_id
        _recount(an)
        moved = 0
        for sc in p.scenes:
            mine = [sl for sl in sc.lines if sl.line_id in part_ids]
            if not mine:
                continue
            snap["scenes"][sc.id] = {
                "lines": [sl.model_dump(mode="json") for sl in mine],
                "takes": {t.id: {"line_id": t.line_id, "speaker_id": t.speaker_id, "selected": t.selected,
                                 "chosen_by_user": t.chosen_by_user} for t in sc.takes if t.line_id in part_ids},
                "dubbed": list(sc.dubbed_speakers),
            }
            new_sl = _merged_scene_line(sorted(mine, key=lambda s: s.start_rel), merged.text)
            new_sl.speaker_id = merged.speaker_id
            first_i = sc.lines.index(mine[0])
            rest = [sl for sl in sc.lines if sl.line_id not in part_ids]
            rest.insert(min(first_i, len(rest)), new_sl)
            sc.lines = sorted(rest, key=lambda s: (s.start_rel, s.end_rel))
            for t in sc.takes:
                if t.line_id in part_ids:
                    if t.line_id != part_ids[0]:
                        moved += 1
                    t.line_id, t.speaker_id = part_ids[0], merged.speaker_id
            _restar(sc, part_ids[0])
            touched.append(sc)
        d = _edits_dir(store, pid)
        d.mkdir(parents=True, exist_ok=True)
        atomic_write_text(d / f"{merge_id}.json", json.dumps(snap, ensure_ascii=False))
        _prune_snapshots(store, pid, an.merged_lines)
        for sc in touched:
            try:
                _resplit(store, pid, sc)
            except Exception as e:  # noqa: BLE001
                log.warning("Дорожки персонажей сцены %s не пересобрались: %s", sc.id, e)
    return store.get(pid), merge_id, moved


def _new_merge_id() -> str:
    return f"j{uuid.uuid4().hex[:8]}"


def _read_snapshot(store: ProjectStore, pid: str, merge_id: str) -> dict | None:
    if not merge_id.isalnum():
        return None
    try:
        return json.loads((_edits_dir(store, pid) / f"{merge_id}.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _prune_snapshots(store: ProjectStore, pid: str, merged_lines: dict[str, str]) -> None:
    """Старые снимки сверх MAX_KEPT_EDITS удаляем, но не те, на которые ссылается карта склеек (и, по цепочке, их
    прежние записи): без снимка «Разделить» не сработает."""
    d = _edits_dir(store, pid)
    files = sorted(d.glob("*.json"), key=lambda f: f.stat().st_mtime)
    if len(files) <= MAX_KEPT_EDITS:
        return
    keep: set[str] = set()
    todo = list(merged_lines.values())
    while todo:
        mid = todo.pop()
        if mid in keep:
            continue
        keep.add(mid)
        snap = _read_snapshot(store, pid, mid)
        if snap:
            todo += list((snap.get("prev_merged") or {}).values())
    excess = len(files) - MAX_KEPT_EDITS
    for f in files:
        if excess <= 0:
            break
        if f.stem not in keep:
            f.unlink(missing_ok=True)
            excess -= 1


def _restar(sc: Scene, line_id: str) -> None:
    from .takes import apply_default_star
    apply_default_star(sc, line_id)


def _split_into_scene(p: Project, sc: Scene, keep: str, part_ids: list[str]) -> bool:
    """Склеенная реплика есть в сцене, а снимка этой сцены нет (сцену создали после склейки, или склеил разбор):
    строки частей — из разбора, как при создании сцены. Текст, который человек правил, остаётся у первой части."""
    from .scene import scene_lines_from_analysis
    if not any(sl.line_id == keep for sl in sc.lines):
        return False
    fresh = [sl for sl in scene_lines_from_analysis(p, sc) if sl.line_id in part_ids]
    sc.lines = sorted([sl for sl in sc.lines if sl.line_id != keep] + fresh, key=lambda x: (x.start_rel, x.end_rel))
    return True


def undo_merge_lines(store: ProjectStore, pid: str, merge_id: str, *, jobs=None) -> Project:
    """Вернуть реплики, как были до склейки (текст, персонаж, дубли и ★; персонажа сменили после склейки — частям
    достаётся нынешний). Склеенную реплику с тех пор склеили ещё раз или поменяли границы — 409: сначала отмени более
    позднюю склейку. Карта Analysis.merged_lines возвращается к тому, что было до этой склейки."""
    _check_idle(jobs, pid)
    touched: list[Scene] = []
    with store.edit(pid) as p:
        snap = _read_snapshot(store, pid, merge_id)
        if snap is None:
            raise NotFoundError("Такой склейки нет или её уже отменили.")
        an = p.analysis
        keep = snap["keep"]
        cur = next((x for x in (an.lines if an else []) if x.id == keep), None)
        m = snap["merged"]
        if an is None or cur is None or abs(cur.start - m["start"]) > 1e-6 or abs(cur.end - m["end"]) > 1e-6:
            raise ConflictError("Эту склейку уже нельзя отменить: сначала отмени более позднюю склейку этих реплик.")
        sid = cur.speaker_id
        # склеивают одного персонажа; если его сменили после склейки — частям тоже нового. Старые снимки (до запрета
        # склейки разных персонажей) могли склеить разных — пока персонажа не меняли, каждой части — её прежнего.
        same = sid == snap["lines"][0].get("speaker_id")
        old = [Line.model_validate(x) for x in snap["lines"]]
        if not same:
            old = [x.model_copy(update={"speaker_id": sid}) for x in old]
        part_ids = [x.id for x in old]
        an.lines = sorted([x for x in an.lines if x.id != keep] + old, key=lambda x: (x.start, x.end, x.id))
        if an.merged_lines.get(keep) == merge_id:
            an.merged_lines.pop(keep)
        an.merged_lines.update(snap.get("prev_merged") or {})
        _recount(an)
        for sc in p.scenes:
            s = snap["scenes"].get(sc.id)
            if s is None:
                if _split_into_scene(p, sc, keep, part_ids):
                    for t in sc.takes:
                        if t.line_id == keep:
                            t.speaker_id = sid
                    for i in part_ids:
                        _restar(sc, i)
                    touched.append(sc)
                continue
            olds = [SceneLine.model_validate(x) for x in s["lines"]]
            if not same:
                olds = [x.model_copy(update={"speaker_id": sid}) for x in olds]
            sc.lines = sorted([sl for sl in sc.lines if sl.line_id != keep] + olds, key=lambda x: (x.start_rel, x.end_rel))
            for t in sc.takes:
                prev = s["takes"].get(t.id)
                if prev is not None:
                    t.line_id, t.speaker_id = prev["line_id"], (prev["speaker_id"] if same else sid)
                    t.selected, t.chosen_by_user = prev["selected"], prev["chosen_by_user"]
            # дубли, записанные уже после склейки, остаются у первой реплики; ★ расставляем заново по каждой реплике
            for sl in olds:
                _restar(sc, sl.line_id)
            touched.append(sc)
        (_edits_dir(store, pid) / f"{merge_id}.json").unlink(missing_ok=True)
        for sc in touched:
            try:
                _resplit(store, pid, sc)
            except Exception as e:  # noqa: BLE001
                log.warning("Дорожки персонажей сцены %s не пересобрались: %s", sc.id, e)
    return store.get(pid)


def split_line(store: ProjectStore, pid: str, line_id: str, *, jobs=None) -> Project:
    """«Разделить обратно»: отменить последнюю склейку, в которую вошла реплика (по Analysis.merged_lines).
    Реплика не склеивалась — 404; снимка склейки больше нет — запись убираем из карты и тоже 404."""
    _check_idle(jobs, pid)
    p = store.get(pid)
    merge_id = p.analysis.merged_lines.get(line_id) if p.analysis else None
    if merge_id is None:
        raise NotFoundError(NOT_MERGED)
    if _read_snapshot(store, pid, merge_id) is None:
        with store.edit(pid) as p2:
            if p2.analysis is not None and p2.analysis.merged_lines.get(line_id) == merge_id:
                p2.analysis.merged_lines.pop(line_id)
        raise NotFoundError(NOT_MERGED)
    return undo_merge_lines(store, pid, merge_id, jobs=jobs)


# ─────────────────────────── авто-склейка при разборе ───────────────────────────
def store_auto_merges(store: ProjectStore, pid: str, groups: list[list[Line]]) -> dict[str, str]:
    """Разбор начат заново: старые снимки склеек больше ни к чему (реплики другие) — стираем. Для каждой авто-склейки
    (группа из 2+ реплик, см. analysis.lines.join_close_groups) пишем такой же снимок, как у ручной (сцен при разборе
    нет). → карта для Analysis.merged_lines: id склеенной реплики → merge_id."""
    d = _edits_dir(store, pid)
    if d.is_dir():
        for f in d.glob("*.json"):
            f.unlink(missing_ok=True)
    out: dict[str, str] = {}
    created = datetime.now().astimezone().isoformat()
    for g in groups:
        if len(g) < 2:
            continue
        merged = merge_group(g)
        merge_id = _new_merge_id()
        snap = {"created": created, "keep": g[0].id, "auto": True,
                "lines": [x.model_dump(mode="json") for x in g],
                "merged": {"start": merged.start, "end": merged.end, "text": merged.text},
                "scenes": {}, "prev_merged": {}}
        d.mkdir(parents=True, exist_ok=True)
        atomic_write_text(d / f"{merge_id}.json", json.dumps(snap, ensure_ascii=False))
        out[g[0].id] = merge_id
    return out
