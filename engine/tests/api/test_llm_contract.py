"""Контракт ответов /llm/*: строгие модели в OpenAPI и реальные ответы с подставным провайдером."""
from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from dubl_engine.api import create_app
from dubl_engine.jobs import JobManager
from dubl_engine.models import (
    Analysis,
    MediaInfo,
    MixSettings,
    Scene,
    SceneLine,
    Speaker,
    Take,
    TrackSettings,
)
from dubl_engine.store import ProjectStore


class Fake:
    id = "fake"

    def __init__(self, *replies):
        self.replies = list(replies)

    async def complete(self, system, user, *, json_schema=None, max_tokens=1024, temperature=0.4):
        r = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        return r if isinstance(r, str) else json.dumps(r, ensure_ascii=False)

    async def check(self):
        return True, "ok"


@pytest.fixture
def env(tmp_path, monkeypatch):
    store = ProjectStore(tmp_path / "projects")
    media = MediaInfo(path="x", duration=10, container="mp4", width=640, height=360, fps=25, video_codec="h264")
    p = store.create("/x.mp4", media, title="t")
    lines = [SceneLine(line_id="l1", speaker_id="s1", start_rel=0, end_rel=3, original="Who touched my sausage?",
                       text="Кто трогал мою колбасу?"),
             SceneLine(line_id="l2", speaker_id="s2", start_rel=4, end_rel=6, original="Not me.", text="Не я.")]
    take = Take(id="t001", line_id="l1", speaker_id="s1", file="takes/t001.wav", duration=2.0, preroll=0.0, start_rel=0.0,
                offset_ms=-300, stretch=0.9, peak_db=-6.0, loudness_lufs=-20.0, selected=True)
    with store.edit(p.id) as pr:
        pr.analysis = Analysis(audio_stream_index=1, speakers=[Speaker(id="s1", label="Голос 1", name="Барсик", color="#fff"),
                                                                Speaker(id="s2", label="Голос 2", name="Мила", color="#000")])
        pr.scenes.append(Scene(id="sc001", title="s", start=0, end=8, lines=lines, takes=[take], dubbed_speakers=["s1"],
                               mix=MixSettings(tracks={"s1": TrackSettings(gain_db=-2.0)}), stems_ready=True))

    def use(*replies):
        fake = Fake(*replies)
        monkeypatch.setattr("dubl_engine.api.routes.llm.provider_for", lambda task, settings: fake)
    with TestClient(create_app(store=store, jobs=JobManager())) as c:
        yield c, p.id, use


def _ref(schema: dict) -> str | None:
    return schema.get("$ref", "").rsplit("/", 1)[-1] or None


def test_schemas_are_strict_in_openapi(tmp_path):
    spec = create_app(store=ProjectStore(tmp_path / "p"), jobs=JobManager()).openapi()
    sch = spec["components"]["schemas"]
    assert sch["TranslateResult"]["properties"]["translations"]["items"] == {"$ref": "#/components/schemas/TranslatedLine"}
    assert set(sch["TranslatedLine"]["required"]) == {"line_id", "text"}
    assert sch["FitLengthResult"]["properties"]["variants"]["items"] == {"$ref": "#/components/schemas/LineVariants"}
    assert set(sch["LineVariants"]["required"]) == {"line_id", "texts"}
    assert set(sch["DirectorTakeResult"]["required"]) == {"report", "feedback"}
    assert set(sch["DirectorCommandResult"]["required"]) == {"patches", "mix"}
    assert "note" in sch["MixPatch"]["required"]            # note есть всегда (в ответе), пустой — только у ручных патчей
    paths = spec["paths"]
    assert _ref(paths["/llm/director"]["post"]["responses"]["200"]["content"]["application/json"]["schema"]) == "DirectorTakeResult"
    assert _ref(paths["/llm/director_command"]["post"]["responses"]["200"]["content"]["application/json"]["schema"]) == "DirectorCommandResult"


def _walk_untyped(node, where, bad):
    """Ищем схемы без типа и объекты-«мешки» (dict[str, Any]): additionalProperties: true / {} или пустую схему."""
    if isinstance(node, dict):
        if node == {} or node.get("additionalProperties") in (True, {}):
            bad.append(where)
        for k, v in node.items():
            if k in ("default", "example", "examples"):          # значения по умолчанию ({} = пустой словарь) — не схемы
                continue
            _walk_untyped(v, f"{where}/{k}", bad)
    elif isinstance(node, list):
        for i, v in enumerate(node):
            _walk_untyped(v, f"{where}[{i}]", bad)


def test_no_untyped_dicts_anywhere_in_schema(tmp_path):
    """В схеме не должно быть dict[str, Any] / unknown ни в одном ответе (иначе в schema.gen.ts появится unknown)."""
    spec = create_app(store=ProjectStore(tmp_path / "p"), jobs=JobManager()).openapi()
    bad: list[str] = []
    _walk_untyped(spec["components"]["schemas"], "schemas", bad)
    for path, ops in spec["paths"].items():
        for method, op in ops.items():
            for code, resp in op["responses"].items():
                if path.startswith("/files"):
                    continue
                _walk_untyped(resp.get("content", {}), f"{method} {path} {code}", bad)
    # HTTPValidationError.detail.ctx / input — служебные поля FastAPI, их клиент не читает
    bad = [b for b in bad if "ValidationError" not in b]
    assert not bad, bad


def test_translate_response_shape(env):
    c, pid, use = env
    use({"translations": [{"id": 0, "text": "Кто трогал мою колбасу?"}, {"id": 1, "text": "Не я."}]})
    r = c.post("/llm/translate", json={"project_id": pid, "scene_id": "sc001"})
    assert r.status_code == 200, r.text
    assert r.json() == {"translations": [{"line_id": "l1", "text": "Кто трогал мою колбасу?", "source": "llm"},
                                          {"line_id": "l2", "text": "Не я.", "source": "llm"}],
                        "ai_model": None, "ai_fallback": False}


def test_fit_length_response_shape(env):
    c, pid, use = env
    use({"variants": ["Кто тронул колбасу?", "Кто взял колбасу?"]})
    r = c.post("/llm/fit_length", json={"project_id": pid, "scene_id": "sc001", "line_ids": ["l1"], "n": 2})
    assert r.status_code == 200, r.text
    j = r.json()
    assert list(j) == ["variants", "ai_model", "ai_fallback"] and len(j["variants"]) == 1
    assert j["variants"][0]["line_id"] == "l1" and all(isinstance(t, str) for t in j["variants"][0]["texts"])
    assert j["variants"][0]["texts"]


def test_director_take_returns_report_and_feedback(env):
    c, pid, use = env
    use("Чуть раньше начни фразу.")
    r = c.post("/llm/director", json={"project_id": pid, "scene_id": "sc001", "take_id": "t001"})
    assert r.status_code == 200, r.text
    j = r.json()
    assert set(j) == {"report", "feedback", "ai_model", "ai_fallback"}
    assert j["report"]["take_id"] == "t001" and j["report"]["timing"] == "late" and isinstance(j["feedback"], str) and j["feedback"]


def test_director_without_take_is_400_and_points_to_command(env):
    c, pid, use = env
    use("x")
    r = c.post("/llm/director", json={"project_id": pid, "scene_id": "sc001", "text": "кот громче"})
    assert r.status_code == 400 and "director_command" in r.json()["detail"]


def test_director_command_returns_patches_with_notes_and_mix(env):
    c, pid, use = env
    use({"patches": [{"target": "s1", "field": "gain_db", "value": 3, "relative": True},
                     {"target": "s1", "field": "room", "value": "cave"},
                     {"target": "master", "field": "target_lufs", "value": -16}]})
    r = c.post("/llm/director_command", json={"project_id": pid, "scene_id": "sc001", "text": "Барсик громче, как в пещере"})
    assert r.status_code == 200, r.text
    j = r.json()
    assert set(j) == {"patches", "mix", "ai_model", "ai_fallback"}
    notes = {(p["target"], p["field"]): p["note"] for p in j["patches"]}
    assert notes[("s1", "gain_db")] == "Барсик: громкость +1 дБ"
    assert notes[("s1", "room")] == "Барсик: пространство «Пещера»"
    assert notes[("master", "target_lufs")] == "Итог: -16 LUFS"
    assert all(p["note"] for p in j["patches"])
    assert j["mix"]["autopilot"] is False and j["mix"]["tracks"]["s1"]["gain_db"] == 1.0
    assert j["mix"]["tracks"]["s1"]["room"] == "cave" and j["mix"]["target_lufs"] == -16.0
    # mix годится для PATCH сцены как есть
    r2 = c.patch(f"/projects/{pid}/scenes/sc001", json={"mix": j["mix"]})
    assert r2.status_code == 200 and r2.json()["mix"]["tracks"]["s1"]["room"] == "cave"


def test_director_command_nothing_understood_keeps_mix(env):
    c, pid, use = env
    use({"patches": []})
    r = c.post("/llm/director_command", json={"project_id": pid, "scene_id": "sc001", "text": "сделай красиво"})
    assert r.status_code == 200 and r.json()["patches"] == []
    assert r.json()["mix"]["autopilot"] is True


def test_director_command_errors(env):
    c, pid, use = env
    use("x")
    assert c.post("/llm/director_command", json={"project_id": pid, "scene_id": "sc001"}).status_code == 400
    assert c.post("/llm/director_command", json={"project_id": pid, "text": "громче"}).status_code == 400   # нет сцены
