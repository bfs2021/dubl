"""Статус ключа ИИ в API: проверка при сохранении, key_set/key_status/key_message, русские ошибки. Без сети (мок httpx)."""
from __future__ import annotations

import httpx
import pytest
from fastapi.testclient import TestClient

from dubl_engine.api import create_app
from dubl_engine.jobs import JobManager
from dubl_engine.llm import keystatus, providers
from dubl_engine.llm.providers import make_provider
from dubl_engine.models import ProviderConfig
from dubl_engine.models import ProviderKind as K
from dubl_engine.store import ProjectStore
from dubl_engine.util import EngineError


def jr(data, status=200, headers=None):
    return httpx.Response(status, json=data, headers=headers)


GEMINI_BAD = {"error": {"code": 400, "message": "API key not valid. Please pass a valid API key.",
                        "status": "INVALID_ARGUMENT"}}
GEMINI_OK = {"candidates": [{"content": {"parts": [{"text": "ок"}]}, "finishReason": "STOP"}]}


@pytest.fixture
def api(tmp_path, monkeypatch):
    """(клиент, set_reply): провайдер-мок подставляется в make_provider; ответы — по очереди."""
    state = {"reply": jr(GEMINI_OK), "calls": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        state["calls"] += 1
        r = state["reply"]
        if isinstance(r, Exception):
            raise r
        return r

    real = providers.make_provider

    def mocked(cfg, *, api_key=None, transport=None):
        return real(cfg, api_key=api_key, transport=httpx.MockTransport(handler))
    from dubl_engine.llm import router
    monkeypatch.setattr(providers, "make_provider", mocked)       # проверка ключа (settings-роуты)
    monkeypatch.setattr(router, "make_provider", mocked)          # запросы ИИ (provider_for)
    store = ProjectStore(tmp_path / "projects")
    with TestClient(create_app(store=store, jobs=JobManager())) as c:
        cfgs = [ProviderConfig(id="gemini", kind=K.GEMINI, title="Gemini", model="gemini-2.5-flash").model_dump(mode="json"),
                ProviderConfig(id="cli", kind=K.CLAUDE_CLI, title="Claude", model="sonnet").model_dump(mode="json")]
        s = c.get("/settings").json()
        s["llm"]["providers"] = cfgs
        assert c.put("/settings", json=s).status_code == 200
        c.state = state  # type: ignore[attr-defined]
        yield c


def _gemini(settings):
    return next(p for p in settings["llm"]["providers"] if p["id"] == "gemini")


def test_settings_have_key_fields_before_and_after_key(api):
    g = _gemini(api.get("/settings").json())
    assert g["key_set"] is False and g["key_status"] == "unknown" and g["key_message"] is None
    cli = next(p for p in api.get("/settings").json()["llm"]["providers"] if p["id"] == "cli")
    assert cli["key_set"] is True                         # у claude_cli ключа нет — «ключ есть» всегда
    r = api.put("/settings/llm/gemini/key", json={"key": "AIza-good"})
    assert r.status_code == 200 and r.json() == {"key_status": "ok", "key_message": None}
    g = _gemini(api.get("/settings").json())
    assert g["key_set"] is True and g["key_status"] == "ok"


def test_put_key_checks_it_and_marks_invalid_but_keeps_key(api):
    api.state["reply"] = jr(GEMINI_BAD, 400)
    r = api.put("/settings/llm/gemini/key", json={"key": "sk-test123"})
    assert r.status_code == 200
    assert r.json() == {"key_status": "invalid",
                        "key_message": "Ключ не подходит — проверь, что скопировал его целиком"}
    g = _gemini(api.get("/settings").json())
    assert g["key_set"] is True and g["key_status"] == "invalid" and "Ключ не подходит" in g["key_message"]
    # ключ сохранён: check идёт с ним и сообщает ту же причину по-русски
    c = api.post("/settings/llm/gemini/check").json()
    assert c["ok"] is False and c["key_status"] == "invalid" and "Ключ ИИ не подходит" in c["message"]
    assert "API key not valid" not in c["message"]


def test_put_key_without_network_is_unknown(api):
    api.state["reply"] = httpx.ConnectError("нет сети")
    r = api.put("/settings/llm/gemini/key", json={"key": "AIza-good"})
    assert r.status_code == 200 and r.json()["key_status"] == "unknown"
    assert _gemini(api.get("/settings").json())["key_set"] is True


def test_put_key_timeout_is_unknown(api, monkeypatch):
    import asyncio
    monkeypatch.setattr(providers, "CHECK_TIMEOUT_S", 0.05)

    async def slow(*a, **k):
        await asyncio.sleep(1)
    monkeypatch.setattr(providers._HttpProvider, "check", slow)
    r = api.put("/settings/llm/gemini/key", json={"key": "AIza-good"})
    assert r.status_code == 200 and r.json()["key_status"] == "unknown"


def test_delete_key_resets_status(api):
    api.state["reply"] = jr(GEMINI_BAD, 400)
    api.put("/settings/llm/gemini/key", json={"key": "bad"})
    assert api.delete("/settings/llm/gemini/key").status_code == 200
    g = _gemini(api.get("/settings").json())
    assert g["key_set"] is False and g["key_status"] == "unknown" and g["key_message"] is None


def test_key_status_not_written_to_settings_json(api, tmp_path):
    from dubl_engine.settings import settings_path
    api.put("/settings/llm/gemini/key", json={"key": "AIza-good"})
    s = api.get("/settings").json()
    api.put("/settings", json=s)                           # UI шлёт настройки обратно вместе с key_*
    raw = settings_path().read_text(encoding="utf-8")
    assert "key_set" not in raw and "key_status" not in raw and "AIza-good" not in raw


# ───────────── русские ошибки провайдера и статус ключа при реальных запросах ─────────────
def _prov(kind=K.GEMINI, **kw):
    return ProviderConfig(id="p1", kind=kind, title="Модель", model="m", base_url="http://h" if kind == K.OPENAI_COMPAT else None, **kw)


@pytest.mark.parametrize("status,body,fragment,reason", [
    (400, GEMINI_BAD, "Ключ ИИ не подходит", "auth"),
    (401, {"error": {"message": "Incorrect API key provided"}}, "Ключ ИИ не подходит", "auth"),
    (403, {"error": {"message": "forbidden"}}, "Ключ ИИ не подходит", "auth"),
    (429, {"error": {"message": "slow down"}}, "Лимит запросов исчерпан, попробуй позже", "rate"),
    (503, {"error": {"message": "overloaded"}}, "Сервис ИИ сейчас недоступен", "server"),
    (500, {}, "Сервис ИИ сейчас недоступен", "server"),
])
@pytest.mark.asyncio
async def test_provider_errors_are_russian_with_raw_in_debug(status, body, fragment, reason):
    t = httpx.MockTransport(lambda r: jr(body, status))
    p = make_provider(_prov(K.OPENAI_COMPAT), api_key="sk-secret-123456", transport=t)
    with pytest.raises(EngineError) as ei:
        await p.complete("s", "u")
    e = ei.value
    assert fragment in str(e) and e.reason == reason                       # type: ignore[attr-defined]
    assert not any(w in str(e) for w in ("API key", "Incorrect", "overloaded", "HTTP"))
    assert str(status) in (e.debug or "") and "sk-secret-123456" not in (e.debug or "")


def test_api_error_has_russian_detail_and_debug(api):
    """Через HTTP: detail по-русски, сырое — в debug; ключ после 401/400 «API key not valid» становится invalid."""
    api.put("/settings/llm/gemini/key", json={"key": "AIza-good"})
    assert _gemini(api.get("/settings").json())["key_status"] == "ok"
    api.state["reply"] = jr(GEMINI_BAD, 400)
    from dubl_engine.models import MediaInfo
    store = api.app.state.store
    p = store.create("/x.mp4", MediaInfo(path="x", duration=5, container="mp4", width=2, height=2, fps=25, video_codec="h264"))
    from dubl_engine.models import Scene, SceneLine
    with store.edit(p.id) as pr:
        pr.scenes.append(Scene(id="sc001", title="s", start=0, end=5, stems_ready=True,
                               lines=[SceneLine(line_id="l1", speaker_id=None, start_rel=0, end_rel=2, original="Hi", text="Hi")]))
    s = api.get("/settings").json()
    s["llm"]["routing"] = {"translate": "gemini"}
    api.put("/settings", json=s)
    r = api.post("/llm/translate", json={"project_id": p.id, "scene_id": "sc001"})
    assert r.status_code == 400
    assert "Ключ ИИ не подходит" in r.json()["detail"] and "API key not valid" not in r.json()["detail"]
    assert "API key not valid" in r.json()["debug"]
    g = _gemini(api.get("/settings").json())
    assert g["key_status"] == "invalid" and "Ключ не подходит" in g["key_message"]
    # потом провайдер ответил нормально — ключ снова ok
    api.state["reply"] = jr({"candidates": [{"content": {"parts": [{"text": '{"translations":[{"id":0,"text":"Привет"}]}'}]}}]})
    r = api.post("/llm/translate", json={"project_id": p.id, "scene_id": "sc001"})
    assert r.status_code == 200, r.text
    assert _gemini(api.get("/settings").json())["key_status"] == "ok"


def test_keystatus_persists_between_calls():
    keystatus.set_status("zz", "invalid", "x")
    assert keystatus.get_status("zz") == ("invalid", "x")
    keystatus.clear("zz")
    assert keystatus.get_status("zz") == ("unknown", None)
