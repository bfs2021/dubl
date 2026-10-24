"""Общие фикстуры тестов слоя ИИ: без сети, без настоящего claude, без настоящей связки ключей."""
from __future__ import annotations

import json

import httpx
import keyring
import pytest
from keyring.backend import KeyringBackend

from dubl_engine.llm import cli as llm_cli
from dubl_engine.llm import providers, router
from dubl_engine.models import ProviderConfig, ProviderKind, SceneLine


class MemoryKeyring(KeyringBackend):
    priority = 1

    def __init__(self) -> None:
        self.data: dict[tuple[str, str], str] = {}

    def set_password(self, service, username, password):
        self.data[(service, username)] = password

    def get_password(self, service, username):
        return self.data.get((service, username))

    def delete_password(self, service, username):
        from keyring.errors import PasswordDeleteError
        if (service, username) not in self.data:
            raise PasswordDeleteError("нет такого")
        del self.data[(service, username)]


@pytest.fixture(autouse=True)
def memory_keyring():
    from dubl_engine.llm import secrets
    old = keyring.get_keyring()
    kr = MemoryKeyring()
    keyring.set_keyring(kr)
    secrets._reset_cache_for_tests()
    yield kr
    keyring.set_keyring(old)
    secrets._reset_cache_for_tests()


@pytest.fixture(autouse=True)
def no_real_cli(monkeypatch):
    """Настоящих claude/codex в тестах нет: поиск ничего не находит, пока тест сам не подставит фейк (fake_cli)."""
    llm_cli.reset_cache_for_tests()
    router.reset_for_tests()
    monkeypatch.setattr(llm_cli, "find_binary", lambda name, refresh=False: None)
    yield
    llm_cli.reset_cache_for_tests()
    router.reset_for_tests()


@pytest.fixture(autouse=True)
def no_backoff(monkeypatch):
    monkeypatch.setattr(providers, "BACKOFF_S", (0.0, 0.0))


@pytest.fixture
def cfg():
    def make(kind: ProviderKind, **kw) -> ProviderConfig:
        base = dict(id=kind.value, kind=kind, title=kind.value, model="m1")
        base.update(kw)
        return ProviderConfig(**base)
    return make


class Recorder:
    """httpx.MockTransport с очередью ответов и журналом запросов."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.requests: list[httpx.Request] = []
        self.transport = httpx.MockTransport(self._handle)

    def _handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        r = self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]
        if isinstance(r, Exception):
            raise r
        if callable(r):
            return r(request)
        return r

    def body(self, i: int = 0) -> dict:
        return json.loads(self.requests[i].content)


@pytest.fixture
def rec():
    return Recorder


def jr(data, status=200, headers=None) -> httpx.Response:
    return httpx.Response(status, json=data, headers=headers)


def openai_ok(text: str) -> httpx.Response:
    return jr({"choices": [{"message": {"role": "assistant", "content": text}}]})


class FakeProvider:
    """Заглушка LLMProvider: отдаёт заготовленные ответы по очереди и запоминает вызовы."""
    id = "fake"

    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls: list[dict] = []

    async def complete(self, system, user, *, json_schema=None, max_tokens=1024, temperature=0.4):
        self.calls.append({"system": system, "user": user, "schema": json_schema})
        r = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        if isinstance(r, Exception):
            raise r
        return r if isinstance(r, str) else json.dumps(r, ensure_ascii=False)

    async def check(self):
        return True, "ok"


@pytest.fixture
def fake():
    return FakeProvider


@pytest.fixture
def scene_line():
    def make(text="Кто трогал мою колбасу?", dur=2.0, speaker="s1", lid="l0001", start=0.0, original="Who touched my sausage?"):
        return SceneLine(line_id=lid, speaker_id=speaker, start_rel=start, end_rel=start + dur, original=original, text=text)
    return make
