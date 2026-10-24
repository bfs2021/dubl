"""Запасная маршрутизация и честный статус: кто реально ответит, пока назначенная модель недоступна или упала."""
from __future__ import annotations

import types

import pytest
from fastapi.testclient import TestClient

from dubl_engine.api.app import create_app
from dubl_engine.jobs import JobManager
from dubl_engine.llm import cli, router, secrets
from dubl_engine.llm.director import command_examples_for, director_feedback, take_report
from dubl_engine.llm.status import fill_status
from dubl_engine.models import (
    LLMSettings,
    LLMTask,
    MediaInfo,
    ProviderConfig,
    ProviderKind,
    Scene,
    SceneLine,
    Speaker,
    Take,
)
from dubl_engine.store import ProjectStore
from dubl_engine.util import EngineError

K = ProviderKind
T = LLMTask


def P(id, kind, model="m", enabled=True):
    return ProviderConfig(id=id, kind=kind, title=id.upper(), model=model, enabled=enabled)


def ALL(provider_ids):
    return {t: provider_ids for t in LLMTask}


@pytest.fixture
def narrow(monkeypatch):
    """Как у приложения из Finder: claude найти нельзя, codex тоже."""
    monkeypatch.setattr(cli, "find_binary", lambda name, refresh=False: None)


def with_claude(monkeypatch, state="ok"):
    monkeypatch.setattr(cli, "find_binary", lambda name, refresh=False: "/x/claude" if name == "claude" else None)
    monkeypatch.setattr(cli, "cli_status", lambda kind, refresh=False: cli.CliStatus(
        state, "", "/x/claude") if kind == K.CLAUDE_CLI else cli.CliStatus("not_installed", ""))


# ───────────── назначенная недоступна → следующая рабочая ─────────────
def test_claude_not_found_falls_to_deepseek(narrow):
    """Исходная проблема: все задачи на claude_cli, а claude в узком PATH не найден."""
    secrets.set_key("deepseek", "k")
    s = LLMSettings(providers=[P("claude_cli", K.CLAUDE_CLI, "sonnet"), P("deepseek", K.DEEPSEEK, "deepseek-flash")],
                    routing=ALL("claude_cli"))
    p = router.provider_for(T.FIT_LENGTH, s)
    assert p is not None and p.id == "deepseek"
    route = router.task_routes(s)[T.FIT_LENGTH]
    assert route.assigned == "claude_cli" and route.answers == "deepseek" and route.fallback
    assert "Claude" in (route.message or "") or "CLAUDE_CLI" in (route.message or "")


def test_no_working_model_at_all_is_the_only_not_configured_case(narrow):
    s = LLMSettings(providers=[P("claude_cli", K.CLAUDE_CLI), P("deepseek", K.DEEPSEEK)], routing=ALL("claude_cli"))
    assert router.provider_for(T.TRANSLATE, s) is None
    text = router.no_model_text(T.TRANSLATE, s)
    assert text.startswith("ИИ для этой задачи не настроен") and "не найден" in text      # причина названа


def test_disabled_and_keyless_models_are_skipped(narrow):
    secrets.set_key("gem", "k")
    s = LLMSettings(providers=[P("a", K.DEEPSEEK, enabled=False), P("b", K.OPENAI_COMPAT), P("gem", K.GEMINI)],
                    routing=ALL("a"))
    assert router.provider_for(T.TRANSLATE, s).id == "gem"


def test_invalid_key_is_skipped_when_another_works(narrow):
    from dubl_engine.llm import keystatus
    secrets.set_key("a", "k")
    secrets.set_key("b", "k")
    keystatus.set_status("a", "invalid", "x")
    s = LLMSettings(providers=[P("a", K.DEEPSEEK), P("b", K.GEMINI)], routing=ALL("a"))
    assert router.provider_for(T.TRANSLATE, s).id == "b"


def test_explicit_off_is_not_overridden(narrow):
    secrets.set_key("b", "k")
    s = LLMSettings(providers=[P("b", K.GEMINI)], routing={T.TRANSLATE: None})
    assert router.provider_for(T.TRANSLATE, s) is None
    assert router.task_routes(s)[T.TRANSLATE].message == "Не используется."


def test_unassigned_task_picks_best_working_model_by_itself(narrow):
    secrets.set_key("flash", "k")
    secrets.set_key("pro", "k")
    s = LLMSettings(providers=[P("pro", K.GEMINI, "gemini-3-pro"), P("flash", K.GEMINI, "gemini-flash")], routing={})
    assert router.provider_for(T.TRANSLATE, s).id == "flash"
    assert router.provider_for(T.DIRECTOR, s).id == "pro"


def test_cheap_tasks_prefer_cheap_fallback_director_prefers_strong(narrow):
    for i in ("x", "cheap", "strong"):
        secrets.set_key(i, "k")
    s = LLMSettings(providers=[P("x", K.GEMINI), P("strong", K.ANTHROPIC, "claude-opus-4"), P("cheap", K.DEEPSEEK, "deepseek-flash")],
                    routing={T.TRANSLATE: "nope", T.DIRECTOR: "nope"})
    assert router.provider_for(T.TRANSLATE, s).id == "cheap"
    assert router.provider_for(T.DIRECTOR, s).id == "strong"


# ───────────── сбой посреди запроса → одна попытка запасной ─────────────
class Prov:
    def __init__(self, id, reply):
        self.id, self.reply, self.calls = id, reply, 0

    async def complete(self, *a, **k):
        self.calls += 1
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply

    async def check(self):
        return True, ""


def routed(task=T.TRANSLATE):
    a, b = Prov("a", EngineError("Лимит запросов исчерпан, попробуй позже.")), Prov("b", "ответ запасной")
    ca, cb = P("a", K.DEEPSEEK, "Модель-А"), P("b", K.GEMINI, "Модель-Б")
    return router.RoutedProvider(task, [(ca, a), (cb, b)], assigned_id="a"), a, b


@pytest.mark.asyncio
async def test_failed_request_retries_on_fallback_once_and_reports_who():
    r, a, b = routed()
    assert await r.complete("s", "u") == "ответ запасной"
    assert (a.calls, b.calls) == (1, 1)
    assert r.answered_by == "B" and r.answered_by_fallback
    assert router.last_used(T.TRANSLATE) == ("B", True)
    assert router.who_answered(r) == ("B", True)
    # следующий вызов того же запроса (пачки перевода) основную, которая только что упала, не дёргает
    await r.complete("s", "u")
    assert (a.calls, b.calls) == (1, 2)


@pytest.mark.asyncio
async def test_both_failed_reports_both_in_russian():
    r, a, b = routed()
    b.reply = EngineError("Сервис ИИ сейчас недоступен.")
    with pytest.raises(EngineError) as e:
        await r.complete("s", "u")
    msg = str(e.value)
    assert "Лимит запросов" in msg and "Запасная модель" in msg and "недоступен" in msg
    assert (a.calls, b.calls) == (1, 1)                    # ровно одна попытка запасной


@pytest.mark.asyncio
async def test_primary_ok_reports_not_fallback():
    r, a, b = routed()
    a.reply = "основная"
    assert await r.complete("s", "u") == "основная"
    assert b.calls == 0 and not r.answered_by_fallback and r.answered_by == "A"


@pytest.mark.asyncio
async def test_single_model_error_propagates():
    a = Prov("a", EngineError("плохо"))
    r = router.RoutedProvider(T.TRANSLATE, [(P("a", K.GEMINI), a)])
    with pytest.raises(EngineError, match="плохо"):
        await r.complete("s", "u")


# ───────────── честный статус ─────────────
def test_status_counts_only_models_that_can_answer(narrow):
    """3 → 2: gemini с «ok» в key_status.json, но без ключа в связке, отвечать не может и в счёт не идёт."""
    from dubl_engine.llm import keystatus
    secrets.set_key("deepseek", "k")
    keystatus.set_status("gemini", "ok")                         # старый след от прошлой сессии
    llm = LLMSettings(providers=[P("claude_cli", K.CLAUDE_CLI), P("deepseek", K.DEEPSEEK), P("gemini", K.GEMINI)],
                      routing=ALL("claude_cli"))
    fill_status(llm)
    by = {p.id: p for p in llm.providers}
    assert [p.id for p in llm.providers if p.can_answer] == ["deepseek"]
    assert by["claude_cli"].state == "not_installed" and by["claude_cli"].login_supported
    assert by["claude_cli"].install_url and "Не найден" in by["claude_cli"].state_message
    assert by["gemini"].state == "no_key" and by["gemini"].key_status == "unknown"       # устаревшее «ok» сброшено
    assert llm.tasks[T.FIT_LENGTH].answers == "deepseek" and llm.tasks[T.FIT_LENGTH].fallback


def test_status_cli_states(monkeypatch):
    with_claude(monkeypatch, "not_logged_in")
    llm = LLMSettings(providers=[P("claude_cli", K.CLAUDE_CLI)], routing=ALL("claude_cli"))
    fill_status(llm)
    p = llm.providers[0]
    assert p.state == "not_logged_in" and not p.can_answer and "нужен вход" in p.state_message
    with_claude(monkeypatch, "ok")
    fill_status(llm)
    assert llm.providers[0].can_answer and "отвечает" in llm.providers[0].state_message
    assert llm.tasks[T.TRANSLATE].answers == "claude_cli" and not llm.tasks[T.TRANSLATE].fallback


def test_status_number_is_stable_between_calls(narrow):
    secrets.set_key("deepseek", "k")
    llm = LLMSettings(providers=[P("deepseek", K.DEEPSEEK), P("gemini", K.GEMINI)], routing=ALL("deepseek"))
    counts = set()
    for _ in range(5):
        fill_status(llm)
        counts.add(sum(1 for p in llm.providers if p.can_answer))
    assert counts == {1}


def test_set_key_fails_loudly_if_keychain_forgets(memory_keyring, monkeypatch):
    monkeypatch.setattr(memory_keyring, "set_password", lambda s, u, p: None)       # «приняла» и не сохранила
    with pytest.raises(EngineError, match="не сохранила"):
        secrets.set_key("gemini", "k")


# ───────────── миграция настроек ─────────────
def test_deepseek_chat_is_migrated_on_load():
    cfg = ProviderConfig(id="deepseek", kind=K.DEEPSEEK, title="D", model="deepseek-chat")
    assert cfg.model == "deepseek-flash"
    assert ProviderConfig(id="o", kind=K.OPENAI_COMPAT, title="O", model="deepseek-chat").model == "deepseek-chat"


# ───────────── API ─────────────
@pytest.fixture
def client(tmp_path, narrow):
    with TestClient(create_app(store=ProjectStore(tmp_path / "projects"), jobs=JobManager())) as c:
        yield c


def _put_llm(client, providers, routing):
    s = client.get("/settings").json()
    s["llm"]["providers"] = [p.model_dump(mode="json") for p in providers]
    s["llm"]["routing"] = routing
    r = client.put("/settings", json=s)
    assert r.status_code == 200, r.text
    return r.json()


def _scene(client):
    store = client.app.state.store
    p = store.create("/x.mp4", MediaInfo(path="x", duration=5, container="mp4", width=2, height=2, fps=25, video_codec="h264"))
    with store.edit(p.id) as pr:
        pr.scenes.append(Scene(id="sc001", title="s", start=0, end=5, stems_ready=True,
                               lines=[SceneLine(line_id="l1", speaker_id=None, start_rel=0, end_rel=2, original="Hi", text="Привет")]))
    return p.id


def test_api_settings_report_who_answers(client):
    secrets.set_key("deepseek", "k")
    s = _put_llm(client, [P("claude_cli", K.CLAUDE_CLI, "sonnet"), P("deepseek", K.DEEPSEEK, "deepseek-chat")], ALL("claude_cli"))
    by = {p["id"]: p for p in s["llm"]["providers"]}
    assert by["claude_cli"]["can_answer"] is False and by["claude_cli"]["state"] == "not_installed"
    assert by["deepseek"]["can_answer"] is True and by["deepseek"]["model"] == "deepseek-flash"
    assert s["llm"]["tasks"]["fit_length"]["answers"] == "deepseek"
    # вычисляемые поля в settings.json не попадают
    import json

    from dubl_engine import settings as st
    raw = json.loads(st.settings_path().read_text())
    assert "tasks" not in raw["llm"] and "can_answer" not in raw["llm"]["providers"][0]
    assert raw["llm"]["providers"][1]["model"] == "deepseek-flash"


def test_api_fit_length_uses_fallback_and_says_so(client, monkeypatch):
    from dubl_engine.llm import providers
    secrets.set_key("deepseek", "k")
    _put_llm(client, [P("claude_cli", K.CLAUDE_CLI), P("deepseek", K.DEEPSEEK, "deepseek-flash")], ALL("claude_cli"))
    pid = _scene(client)

    async def fake_complete(self, system, user, **k):
        return '{"variants": ["Привет", "Здрасьте"]}'
    monkeypatch.setattr(providers.DeepSeekProvider, "complete", fake_complete)
    r = client.post("/llm/fit_length", json={"project_id": pid, "scene_id": "sc001"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ai_model"] == "DEEPSEEK" and body["ai_fallback"] is True


def test_api_no_model_gives_reason_in_russian(client):
    _put_llm(client, [P("claude_cli", K.CLAUDE_CLI)], ALL("claude_cli"))
    pid = _scene(client)
    r = client.post("/llm/translate", json={"project_id": pid, "scene_id": "sc001"})
    assert r.status_code == 400
    d = r.json()["detail"]
    assert d.startswith("ИИ для этой задачи не настроен") and "Claude Code не найден" not in d and "не найден" in d


def test_api_login_unknown_and_keyed_models(client):
    _put_llm(client, [P("deepseek", K.DEEPSEEK)], ALL("deepseek"))
    assert client.post("/settings/llm/nope/login").status_code == 404
    r = client.post("/settings/llm/deepseek/login")
    assert r.status_code == 400 and "ключ" in r.json()["detail"]


def test_api_login_opens_terminal(client, monkeypatch):
    _put_llm(client, [P("claude_cli", K.CLAUDE_CLI)], ALL("claude_cli"))
    seen = []
    monkeypatch.setattr(cli, "open_login_terminal", lambda kind: (seen.append(kind), "Открыл терминал")[1])
    r = client.post("/settings/llm/claude_cli/login")
    assert r.status_code == 200 and r.json() == {"ok": True, "message": "Открыл терминал"} and seen == [K.CLAUDE_CLI]


def test_api_settings_refresh_flag(client, monkeypatch):
    _put_llm(client, [P("claude_cli", K.CLAUDE_CLI)], ALL("claude_cli"))
    flags = []
    real = cli.cli_status
    monkeypatch.setattr(cli, "cli_status", lambda kind, refresh=False: (flags.append(refresh), real(kind, refresh=refresh))[1])
    client.get("/settings")
    assert True not in flags
    client.get("/settings?refresh=true")
    assert flags.count(True) == 1


# ───────────── помощник-режиссёр ─────────────
LINE = SceneLine(line_id="l1", speaker_id="s1", start_rel=0, end_rel=2, original="Who", text="Кто трогал мою колбасу?")


def _take(**kw):
    base = dict(id="t001", line_id="l1", speaker_id="s1", file="t.wav", duration=2.0, preroll=1.0, start_rel=0.0,
                offset_ms=0.0, stretch=1.0, peak_db=-6.0, loudness_lufs=-20.0, recognized_text="Кто трогал мою колбасу")
    base.update(kw)
    return Take(**base)


def _as_file(take: Take):
    """Дубль из файла (Take.source = 'file'). Если поля нет в модели — движок читает его через getattr с дефолтом 'mic'."""
    return take.model_copy(update={"source": "file"})


def test_take_without_source_attribute_counts_as_mic():
    bare = types.SimpleNamespace(**{k: v for k, v in _take(offset_ms=700).model_dump().items() if k != "source"})
    assert "early" in take_report(bare, LINE)["problems"]


def test_file_take_gets_no_timing_or_length_advice():
    t = _as_file(_take(offset_ms=700, stretch=0.8))
    r = take_report(t, LINE)
    assert r["source"] == "file" and r["problems"] == [] and r["ok"]
    mic = take_report(_take(offset_ms=700, stretch=0.8), LINE)
    assert mic["source"] == "mic" and "early" in mic["problems"]


@pytest.mark.asyncio
async def test_file_take_still_warns_about_sound_and_text():
    t = _as_file(_take(offset_ms=700, peak_db=0.0))
    r = take_report(t, LINE)
    assert r["problems"] == ["clipping"]
    assert "перегружен" in await director_feedback(r, None)


def test_no_hurry_phrase_left():
    from dubl_engine.llm import director
    assert not any("спешишь" in v for v in (*director._ADVICE.values(), *director._ADVICE_BIG.values()))


def test_command_examples_use_scene_characters():
    sc = Scene(id="sc1", title="t", start=0, end=5, dubbed_speakers=["s1", "s2"], lines=[
        SceneLine(line_id="l1", speaker_id="s1", start_rel=0, end_rel=1, original="a", text="а"),
        SceneLine(line_id="l2", speaker_id="s2", start_rel=1, end_rel=2, original="b", text="б")])
    sp = [Speaker(id="s1", label="Голос 1", name="Слон", color="#fff"), Speaker(id="s2", label="Голос 2", name="Мила", color="#000")]
    assert command_examples_for(sc, sp) == ["Слон громче", "Мила тише", "фон потише"]
    from dubl_engine.llm import prompts
    ex = prompts.command_examples([("s1", "Слон"), ("s2", "Мила")])
    assert "Слон громче" in ex and "кот" not in ex.lower()
    assert "кот" not in prompts.command_examples([]).lower()
