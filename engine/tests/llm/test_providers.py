from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from dubl_engine.llm import providers, secrets
from dubl_engine.llm.providers import (
    AnthropicProvider,
    ClaudeCliProvider,
    DeepSeekProvider,
    GeminiProvider,
    OpenAICompatProvider,
    extract_json,
    list_models,
    make_provider,
)
from dubl_engine.models import ProviderKind
from dubl_engine.util import EngineError

from .conftest import jr, openai_ok

K = ProviderKind
SCHEMA = {"type": "object", "properties": {"a": {"type": "string"}}, "required": ["a"]}


# ─────────── извлечение JSON ───────────
@pytest.mark.parametrize("raw", [
    '{"a": "x"}',
    '```json\n{"a": "x"}\n```',
    'Вот ответ:\n{"a": "x"}\nНадеюсь, помогло',
    '  ```\n{"a": "x"}\n```  ',
])
def test_extract_json(raw):
    assert extract_json(raw) == {"a": "x"}


def test_extract_json_garbage():
    with pytest.raises(EngineError, match="JSON"):
        extract_json("никакого json тут нет")


def test_make_provider_kinds(cfg):
    assert isinstance(make_provider(cfg(K.OPENAI_COMPAT, base_url="http://x/v1")), OpenAICompatProvider)
    assert isinstance(make_provider(cfg(K.DEEPSEEK)), DeepSeekProvider)
    assert isinstance(make_provider(cfg(K.ANTHROPIC)), AnthropicProvider)
    assert isinstance(make_provider(cfg(K.GEMINI)), GeminiProvider)
    assert isinstance(make_provider(cfg(K.CLAUDE_CLI)), ClaudeCliProvider)


# ─────────── OpenAI-совместимый ───────────
@pytest.mark.asyncio
async def test_openai_success_and_headers(cfg, rec):
    r = rec(openai_ok("привет"))
    p = OpenAICompatProvider(cfg(K.OPENAI_COMPAT, base_url="https://api.example.com/v1/"), api_key="sk-secret", transport=r.transport)
    assert await p.complete("sys", "usr", max_tokens=50, temperature=0.1) == "привет"
    req = r.requests[0]
    assert str(req.url) == "https://api.example.com/v1/chat/completions"
    assert req.headers["authorization"] == "Bearer sk-secret"
    body = r.body()
    assert body["model"] == "m1" and body["max_tokens"] == 50 and body["temperature"] == 0.1
    assert body["messages"][0] == {"role": "system", "content": "sys"}
    assert "response_format" not in body


@pytest.mark.asyncio
async def test_openai_key_from_keyring(cfg, rec):
    secrets.set_key("openai_compat", "kr-key")
    r = rec(openai_ok("ok"))
    p = make_provider(cfg(K.OPENAI_COMPAT, base_url="http://h/v1"), transport=r.transport)
    await p.complete("s", "u")
    assert r.requests[0].headers["authorization"] == "Bearer kr-key"


@pytest.mark.asyncio
async def test_openai_no_key(cfg, rec):
    p = OpenAICompatProvider(cfg(K.OPENAI_COMPAT, base_url="http://h/v1"), transport=rec(openai_ok("x")).transport)
    with pytest.raises(EngineError, match="не задан ключ"):
        await p.complete("s", "u")


@pytest.mark.asyncio
async def test_openai_json_schema_native(cfg, rec):
    r = rec(openai_ok('{"a": "x"}'))
    p = OpenAICompatProvider(cfg(K.OPENAI_COMPAT, base_url="http://h/v1"), api_key="k", transport=r.transport)
    await p.complete("s", "u", json_schema=SCHEMA)
    rf = r.body()["response_format"]
    assert rf["type"] == "json_schema" and rf["json_schema"]["schema"] == SCHEMA


@pytest.mark.asyncio
async def test_openai_json_schema_fallback_to_json_object(cfg, rec):
    bad = jr({"error": {"message": "response_format json_schema is not supported"}}, 400)
    r = rec(bad, openai_ok('{"a": "x"}'))
    p = OpenAICompatProvider(cfg(K.OPENAI_COMPAT, base_url="http://h/v1"), api_key="k", transport=r.transport)
    assert await p.complete("s", "u", json_schema=SCHEMA) == '{"a": "x"}'
    second = r.body(1)
    assert second["response_format"] == {"type": "json_object"}
    assert "JSON" in second["messages"][0]["content"] and '"a"' in second["messages"][0]["content"]


@pytest.mark.asyncio
async def test_deepseek_defaults(cfg, rec):
    r = rec(openai_ok('{"a": "x"}'))
    p = make_provider(cfg(K.DEEPSEEK, model="deepseek-v4-flash"), api_key="k", transport=r.transport)
    await p.complete("s", "u", json_schema=SCHEMA)
    assert str(r.requests[0].url) == "https://api.deepseek.com/chat/completions"
    assert r.body()["model"] == "deepseek-v4-flash"
    assert r.body()["response_format"] == {"type": "json_object"}     # схема уходит в промпт
    assert r.body()["thinking"] == {"type": "disabled"}               # размышления съедают лимит → пустой ответ


@pytest.mark.asyncio
async def test_openai_compat_no_thinking_field(cfg, rec):
    r = rec(openai_ok("ок"))
    p = make_provider(cfg(K.OPENAI_COMPAT, base_url="http://h/v1"), api_key="k", transport=r.transport)
    await p.complete("s", "u")
    assert "thinking" not in r.body()


@pytest.mark.asyncio
async def test_reasoning_ate_whole_limit_is_explained(cfg, rec):
    data = {"choices": [{"message": {"content": "", "reasoning_content": "думаю…"}, "finish_reason": "length"}]}
    r = rec(jr(data))
    p = make_provider(cfg(K.DEEPSEEK, model="deepseek-v4-pro"), api_key="k", transport=r.transport)
    with pytest.raises(EngineError) as ei:
        await p.complete("s", "u")
    assert "размышляет" in str(ei.value)

@pytest.mark.asyncio
async def test_openai_401(cfg, rec):
    r = rec(jr({"error": {"message": "Incorrect API key sk-secret"}}, 401))
    p = OpenAICompatProvider(cfg(K.OPENAI_COMPAT, base_url="http://h/v1", title="Мой"), api_key="sk-secret", transport=r.transport)
    with pytest.raises(EngineError) as ei:
        await p.complete("s", "u")
    assert "Ключ ИИ не подходит" in str(ei.value) and "sk-secret" not in str(ei.value)
    assert len(r.requests) == 1                                         # 401 не повторяем


@pytest.mark.asyncio
async def test_openai_402_money(cfg, rec):
    p = OpenAICompatProvider(cfg(K.DEEPSEEK), api_key="k", transport=rec(jr({"error": {"message": "Insufficient Balance"}}, 402)).transport)
    with pytest.raises(EngineError, match="закончились деньги"):
        await p.complete("s", "u")


@pytest.mark.asyncio
async def test_openai_quota_429_text_not_retried_forever(cfg, rec):
    quota = jr({"error": {"message": "You exceeded your current quota"}}, 429)
    r = rec(quota)
    p = OpenAICompatProvider(cfg(K.OPENAI_COMPAT, base_url="http://h/v1"), api_key="k", transport=r.transport)
    with pytest.raises(EngineError, match="деньги|лимит"):
        await p.complete("s", "u")
    assert len(r.requests) == 3                                        # 1 + 2 ретрая


@pytest.mark.asyncio
async def test_openai_429_retry_then_success(cfg, rec):
    r = rec(jr({"error": {"message": "rate"}}, 429, {"retry-after": "0"}), jr({}, 503), openai_ok("ок"))
    p = OpenAICompatProvider(cfg(K.OPENAI_COMPAT, base_url="http://h/v1"), api_key="k", transport=r.transport)
    assert await p.complete("s", "u") == "ок"
    assert len(r.requests) == 3


@pytest.mark.asyncio
async def test_openai_429_gives_up(cfg, rec):
    r = rec(jr({"error": {"message": "rate"}}, 429))
    p = OpenAICompatProvider(cfg(K.OPENAI_COMPAT, base_url="http://h/v1"), api_key="k", transport=r.transport)
    with pytest.raises(EngineError, match="Лимит запросов исчерпан"):
        await p.complete("s", "u")
    assert len(r.requests) == 3


@pytest.mark.asyncio
async def test_openai_model_not_found(cfg, rec):
    p = OpenAICompatProvider(cfg(K.OPENAI_COMPAT, base_url="http://h/v1"), api_key="k",
                             transport=rec(jr({"error": {"message": "The model `zzz` does not exist"}}, 404)).transport)
    with pytest.raises(EngineError, match="модель не найдена"):
        await p.complete("s", "u")


@pytest.mark.asyncio
async def test_openai_timeout(cfg, rec):
    r = rec(httpx.ReadTimeout("slow"))
    p = OpenAICompatProvider(cfg(K.OPENAI_COMPAT, base_url="http://h/v1"), api_key="k", transport=r.transport)
    with pytest.raises(EngineError, match="не ответила"):
        await p.complete("s", "u")


@pytest.mark.asyncio
async def test_openai_no_network(cfg, rec):
    p = OpenAICompatProvider(cfg(K.OPENAI_COMPAT, base_url="http://h/v1"), api_key="k",
                             transport=rec(httpx.ConnectError("down")).transport)
    with pytest.raises(EngineError, match="Нет связи"):
        await p.complete("s", "u")


@pytest.mark.asyncio
async def test_openai_weird_response(cfg, rec):
    p = OpenAICompatProvider(cfg(K.OPENAI_COMPAT, base_url="http://h/v1"), api_key="k", transport=rec(jr({"oops": 1})).transport)
    with pytest.raises(EngineError, match="неожиданном формате"):
        await p.complete("s", "u")


@pytest.mark.asyncio
async def test_check_ok_and_fail(cfg, rec):
    p = OpenAICompatProvider(cfg(K.OPENAI_COMPAT, base_url="http://h/v1"), api_key="k", transport=rec(openai_ok("ок")).transport)
    ok, msg = await p.check()
    assert ok and "работает" in msg
    p2 = OpenAICompatProvider(cfg(K.OPENAI_COMPAT, base_url="http://h/v1"), api_key="k", transport=rec(jr({}, 401)).transport)
    ok, msg = await p2.check()
    assert not ok and "Ключ ИИ не подходит" in msg


# ─────────── Anthropic ───────────
@pytest.mark.asyncio
async def test_anthropic_success(cfg, rec):
    r = rec(jr({"content": [{"type": "text", "text": "при"}, {"type": "text", "text": "вет"}]}))
    p = AnthropicProvider(cfg(K.ANTHROPIC), api_key="ak", transport=r.transport)
    assert await p.complete("sys", "usr", json_schema=SCHEMA) == "привет"
    req = r.requests[0]
    assert str(req.url) == "https://api.anthropic.com/v1/messages"
    assert req.headers["x-api-key"] == "ak" and req.headers["anthropic-version"] == "2023-06-01"
    body = r.body()
    assert body["system"].startswith("sys") and '"properties"' in body["system"]
    assert body["messages"] == [{"role": "user", "content": "usr"}] and body["max_tokens"] == 1024


@pytest.mark.asyncio
async def test_anthropic_401_and_529_retry(cfg, rec):
    p = AnthropicProvider(cfg(K.ANTHROPIC), api_key="ak", transport=rec(jr({"error": {"message": "invalid x-api-key"}}, 401)).transport)
    with pytest.raises(EngineError, match="Ключ ИИ не подходит"):
        await p.complete("s", "u")
    r = rec(jr({"error": {"type": "overloaded_error"}}, 529), jr({"content": [{"type": "text", "text": "ok"}]}))
    p = AnthropicProvider(cfg(K.ANTHROPIC), api_key="ak", transport=r.transport)
    assert await p.complete("s", "u") == "ok"


# ─────────── Gemini ───────────
@pytest.mark.asyncio
async def test_gemini_success_with_schema(cfg, rec):
    r = rec(jr({"candidates": [{"content": {"parts": [{"text": '{"a": "x"}'}]}, "finishReason": "STOP"}]}))
    p = GeminiProvider(cfg(K.GEMINI, model="gemini-2.5-flash"), api_key="gk", transport=r.transport)
    assert await p.complete("sys", "usr", json_schema=SCHEMA, max_tokens=77) == '{"a": "x"}'
    req = r.requests[0]
    assert str(req.url) == "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
    assert req.headers["x-goog-api-key"] == "gk"
    body = r.body()
    assert body["generationConfig"]["responseMimeType"] == "application/json"
    assert body["generationConfig"]["maxOutputTokens"] == 77
    assert body["contents"][0]["parts"][0]["text"] == "usr"


@pytest.mark.asyncio
async def test_gemini_errors(cfg, rec):
    p = GeminiProvider(cfg(K.GEMINI), api_key="gk", transport=rec(jr({"error": {"message": "API key not valid"}}, 400)).transport)
    # 400 без слова model — общее «запрос отклонён»; ключ не должен просочиться
    with pytest.raises(EngineError) as ei:
        await p.complete("s", "u")
    assert "gk" not in str(ei.value)
    p = GeminiProvider(cfg(K.GEMINI), api_key="gk", transport=rec(jr({"error": {"message": "x"}}, 403)).transport)
    with pytest.raises(EngineError, match="Ключ ИИ не подходит"):
        await p.complete("s", "u")
    p = GeminiProvider(cfg(K.GEMINI), api_key="gk", transport=rec(jr({"promptFeedback": {"blockReason": "SAFETY"}})).transport)
    with pytest.raises(EngineError, match="фильтр"):
        await p.complete("s", "u")


@pytest.mark.asyncio
async def test_gemini_429_retry(cfg, rec):
    r = rec(jr({"error": {"message": "slow down"}}, 429),
            jr({"candidates": [{"content": {"parts": [{"text": "да"}]}}]}))
    p = GeminiProvider(cfg(K.GEMINI), api_key="gk", transport=r.transport)
    assert await p.complete("s", "u") == "да"


# ─────────── list_models ───────────
@pytest.mark.asyncio
async def test_list_models_openai(cfg, rec):
    r = rec(jr({"data": [{"id": "b"}, {"id": "a"}]}))
    out = await list_models(cfg(K.OPENAI_COMPAT, base_url="http://h/v1"), api_key="k", transport=r.transport)
    assert out == ["a", "b"] and str(r.requests[0].url) == "http://h/v1/models"


@pytest.mark.asyncio
async def test_list_models_deepseek_and_gemini(cfg, rec):
    r = rec(jr({"data": [{"id": "deepseek-v4-flash"}, {"id": "deepseek-v4-pro"}]}))
    assert await list_models(cfg(K.DEEPSEEK), api_key="k", transport=r.transport) == ["deepseek-v4-flash", "deepseek-v4-pro"]
    assert str(r.requests[0].url) == "https://api.deepseek.com/models"
    r = rec(jr({"models": [
        {"name": "models/gemini-2.5-pro", "supportedGenerationMethods": ["generateContent"]},
        {"name": "models/embedding-001", "supportedGenerationMethods": ["embedContent"]}]}))
    assert await list_models(cfg(K.GEMINI), api_key="k", transport=r.transport) == ["gemini-2.5-pro"]


@pytest.mark.asyncio
async def test_list_models_claude_cli_no_network(cfg):
    assert "sonnet" in await list_models(cfg(K.CLAUDE_CLI))


# ─────────── Claude CLI (subprocess замокан) ───────────
class FakeProc:
    def __init__(self, stdout=b"", stderr=b"", code=0, hang=False):
        self._out, self._err, self.returncode, self.hang = stdout, stderr, code, hang
        self.stdin_data = None
        self.killed = False

    async def communicate(self, data=None):
        self.stdin_data = data
        if self.hang:
            await asyncio.sleep(3600)
        return self._out, self._err

    def kill(self):
        self.killed = True


@pytest.fixture
def cli(monkeypatch):
    state = {"cmd": None, "proc": FakeProc(), "kwargs": None}

    async def fake_exec(*cmd, **kwargs):
        state["cmd"], state["kwargs"] = list(cmd), kwargs
        return state["proc"]

    monkeypatch.setattr(providers.cli, "find_binary", lambda name, refresh=False: "/usr/bin/claude" if name == "claude" else None)
    monkeypatch.setattr(providers.asyncio, "create_subprocess_exec", fake_exec)
    return state


def result_json(text="привет", **extra):
    return json.dumps({"type": "result", "subtype": "success", "is_error": False, "result": text, **extra}).encode()


@pytest.mark.asyncio
async def test_cli_success(cfg, cli):
    cli["proc"] = FakeProc(stdout=result_json("Готово"))
    p = ClaudeCliProvider(cfg(K.CLAUDE_CLI, model="haiku"))
    assert await p.complete("системный", "вопрос", json_schema=SCHEMA) == "Готово"
    cmd = cli["cmd"]
    assert cmd[0] == "/usr/bin/claude" and "-p" in cmd
    assert cmd[cmd.index("--output-format") + 1] == "json" and cmd[cmd.index("--model") + 1] == "haiku"
    assert "системный" in cmd[cmd.index("--append-system-prompt") + 1] and '"properties"' in cmd[cmd.index("--append-system-prompt") + 1]
    assert cli["proc"].stdin_data == "вопрос".encode()                  # промпт идёт через stdin


@pytest.mark.asyncio
async def test_cli_not_installed(cfg, monkeypatch):
    monkeypatch.setattr(providers.cli, "find_binary", lambda name, refresh=False: None)
    p = ClaudeCliProvider(cfg(K.CLAUDE_CLI))
    with pytest.raises(EngineError, match="Claude Code не найден"):
        await p.complete("s", "u")
    ok, msg = await p.check()
    assert not ok and "не найден" in msg


@pytest.mark.asyncio
async def test_cli_not_logged_in(cfg, cli):
    cli["proc"] = FakeProc(stdout=json.dumps({"is_error": True, "result": "Invalid API key · Please run /login"}).encode(), code=1)
    with pytest.raises(EngineError, match="нужно войти"):
        await ClaudeCliProvider(cfg(K.CLAUDE_CLI)).complete("s", "u")


@pytest.mark.asyncio
async def test_cli_limit_and_garbage(cfg, cli):
    cli["proc"] = FakeProc(stdout=json.dumps({"is_error": True, "result": "Claude AI usage limit reached"}).encode(), code=1)
    with pytest.raises(EngineError, match="лимит"):
        await ClaudeCliProvider(cfg(K.CLAUDE_CLI)).complete("s", "u")
    cli["proc"] = FakeProc(stdout=b"not json at all", code=0)
    with pytest.raises(EngineError, match="ошибкой"):
        await ClaudeCliProvider(cfg(K.CLAUDE_CLI)).complete("s", "u")


@pytest.mark.asyncio
async def test_cli_timeout(cfg, cli, monkeypatch):
    monkeypatch.setattr(providers, "CLI_TIMEOUT_S", 0.05)
    cli["proc"] = FakeProc(hang=True)
    with pytest.raises(EngineError, match="не ответил за"):
        await ClaudeCliProvider(cfg(K.CLAUDE_CLI)).complete("s", "u")
    assert cli["proc"].killed


@pytest.mark.asyncio
async def test_cli_check_ok(cfg, cli):
    cli["proc"] = FakeProc(stdout=result_json("ок"))
    ok, msg = await ClaudeCliProvider(cfg(K.CLAUDE_CLI)).check()
    assert ok and "работает" in msg
