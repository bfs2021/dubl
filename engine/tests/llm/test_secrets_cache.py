"""Связку ключей трогаем как можно реже: каждое чтение на macOS может показать окно с паролем."""
from __future__ import annotations

from dubl_engine.llm import secrets


class CountingKeyring:
    def __init__(self, inner):
        self.inner, self.reads = inner, 0

    def __getattr__(self, name):
        return getattr(self.inner, name)


def test_key_is_read_from_keychain_once(memory_keyring, monkeypatch):
    secrets.set_key("deepseek", "sk-1")
    secrets._reset_cache_for_tests()                     # как после перезапуска движка
    reads = []
    orig = memory_keyring.get_password
    monkeypatch.setattr(memory_keyring, "get_password", lambda s, u: (reads.append(u), orig(s, u))[1])
    for _ in range(20):
        assert secrets.get_key("deepseek") == "sk-1"
        assert secrets.has_key("deepseek")
    assert reads == ["deepseek"]                         # один раз за жизнь процесса


def test_has_key_without_keychain_after_restart(memory_keyring, monkeypatch):
    secrets.set_key("gemini", "g-1")
    secrets._reset_cache_for_tests()
    monkeypatch.setattr(memory_keyring, "get_password", lambda s, u: (_ for _ in ()).throw(AssertionError("связку трогать нельзя")))
    assert secrets.has_key("gemini")                     # экран настроек: ответ из keys_present.json


def test_absent_key_never_touches_keychain(memory_keyring, monkeypatch):
    secrets.set_key("x", "k")
    secrets.delete_key("x")
    secrets._reset_cache_for_tests()
    monkeypatch.setattr(memory_keyring, "get_password", lambda s, u: (_ for _ in ()).throw(AssertionError("связку трогать нельзя")))
    assert secrets.get_key("x") is None
    assert not secrets.has_key("x")


def test_replacing_key_updates_cache(memory_keyring):
    secrets.set_key("deepseek", "old")
    assert secrets.get_key("deepseek") == "old"
    secrets.set_key("deepseek", "new")
    assert secrets.get_key("deepseek") == "new"
