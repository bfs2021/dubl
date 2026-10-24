from __future__ import annotations

import keyring
import pytest
from keyring.backends import fail, null

from dubl_engine.llm import secrets
from dubl_engine.llm.providers import ClaudeCliProvider, DeepSeekProvider
from dubl_engine.llm.router import default_routing, provider_for
from dubl_engine.models import LLMSettings, LLMTask, ProviderConfig, ProviderKind
from dubl_engine.util import EngineError

K = ProviderKind


# ─────────── ключи ───────────
def test_keys_roundtrip(memory_keyring):
    assert secrets.get_key("deepseek") is None
    secrets.set_key("deepseek", "  sk-1  ")
    assert secrets.get_key("deepseek") == "sk-1"
    assert memory_keyring.data == {("dubl", "deepseek"): "sk-1"}      # service="dubl", username=provider_id
    secrets.delete_key("deepseek")
    assert secrets.get_key("deepseek") is None
    secrets.delete_key("deepseek")                                      # повторное удаление не падает


def test_empty_key_rejected():
    with pytest.raises(EngineError):
        secrets.set_key("x", "  ")


@pytest.mark.parametrize("backend", [fail.Keyring, null.Keyring])
def test_no_keyring_gives_clear_error(backend):
    keyring.set_keyring(backend())
    with pytest.raises(EngineError, match="связки ключей"):
        secrets.set_key("x", "k")
    with pytest.raises(EngineError):
        secrets.get_key("x")
    assert secrets.has_key("x") is False


def test_key_not_in_settings_json():
    s = LLMSettings(providers=[ProviderConfig(id="d", kind=K.DEEPSEEK, title="D", model="m")])
    dumped = s.model_dump_json().lower().replace("keyless", "")
    for computed in ("key_set", "key_status", "key_message"):     # вычисляемые поля ответа API; сам ключ — никогда
        dumped = dumped.replace(computed, "")
    assert "key" not in dumped


# ─────────── маршрутизация ───────────
def P(id, kind, model, enabled=True):
    return ProviderConfig(id=id, kind=kind, title=id, model=model, enabled=enabled)


def test_default_routing_cheap_vs_best():
    provs = [P("pro", K.DEEPSEEK, "deepseek-v4-pro"), P("flash", K.DEEPSEEK, "deepseek-v4-flash"),
             P("opus", K.ANTHROPIC, "claude-opus-4")]
    r = default_routing(provs)
    assert r[LLMTask.TRANSLATE] == r[LLMTask.FIT_LENGTH] == r[LLMTask.NAME_SPEAKERS] == "flash"
    assert r[LLMTask.DIRECTOR] in ("pro", "opus")


def test_default_routing_single_and_empty_and_disabled():
    only = [P("a", K.OPENAI_COMPAT, "some-model")]
    assert set(default_routing(only).values()) == {"a"}
    assert set(default_routing([]).values()) == {None}
    assert set(default_routing([P("a", K.DEEPSEEK, "x", enabled=False)]).values()) == {None}
    r = default_routing([P("off", K.DEEPSEEK, "deepseek-v4-flash", enabled=False), P("on", K.GEMINI, "gemini-2.5-pro")])
    assert set(r.values()) == {"on"}


def test_provider_for_none_cases():
    assert provider_for(LLMTask.TRANSLATE, LLMSettings()) is None                       # ничего не настроено
    s = LLMSettings(providers=[P("d", K.DEEPSEEK, "m")], routing={LLMTask.TRANSLATE: None})
    assert provider_for(LLMTask.TRANSLATE, s) is None                                   # None — не использовать
    s = LLMSettings(providers=[P("d", K.DEEPSEEK, "m")], routing={LLMTask.TRANSLATE: "nope"})
    assert provider_for(LLMTask.TRANSLATE, s) is None                                   # нет такого id
    s = LLMSettings(providers=[P("d", K.DEEPSEEK, "m", enabled=False)], routing={LLMTask.TRANSLATE: "d"})
    assert provider_for(LLMTask.TRANSLATE, s) is None                                   # выключен


def test_provider_for_needs_key():
    s = LLMSettings(providers=[P("d", K.DEEPSEEK, "m")], routing={LLMTask.TRANSLATE: "d"})
    assert provider_for(LLMTask.TRANSLATE, s) is None                                   # ключа нет
    secrets.set_key("d", "k")
    p = provider_for(LLMTask.TRANSLATE, s)
    assert isinstance(p.chain[0][1], DeepSeekProvider) and p.id == "d"
    assert provider_for(LLMTask.DIRECTOR, s) is not None                                # не назначена, но рабочая модель есть — берём её
    s = LLMSettings(providers=[P("d", K.DEEPSEEK, "m")], routing={LLMTask.TRANSLATE: "d", LLMTask.DIRECTOR: None})
    assert provider_for(LLMTask.DIRECTOR, s) is None                                    # явное «не использовать»


def test_provider_for_claude_cli(monkeypatch):
    from dubl_engine.llm import cli
    s = LLMSettings(providers=[P("cli", K.CLAUDE_CLI, "sonnet")], routing={LLMTask.DIRECTOR: "cli"})
    assert provider_for(LLMTask.DIRECTOR, s) is None                                    # claude не найден, других моделей нет
    monkeypatch.setattr(cli, "find_binary", lambda n, refresh=False: "/bin/claude")
    monkeypatch.setattr(cli, "cli_status", lambda kind, refresh=False: cli.CliStatus("ok", "ок", "/bin/claude"))
    p = provider_for(LLMTask.DIRECTOR, s)
    assert isinstance(p.chain[0][1], ClaudeCliProvider)                                 # ключ не нужен
