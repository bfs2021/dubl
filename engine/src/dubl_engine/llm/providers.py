"""Провайдеры моделей: Claude Code и Codex (по подписке, через CLI), Anthropic API, DeepSeek, Gemini, любой OpenAI-совместимый.

Каждый пользователь подключает свои модели и ключи. Ключ берётся из системной связки ключей
(`secrets.get_key`) в момент запроса и нигде не сохраняется и не печатается.
Все ошибки превращаются в EngineError с понятным русским текстом.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import tempfile
from pathlib import Path
from typing import Any, ClassVar, Protocol, runtime_checkable
from urllib.parse import quote

import httpx

from ..models import ProviderConfig, ProviderKind
from ..util import EngineError, redact
from . import cli, keystatus, prompts, secrets

log = logging.getLogger(__name__)

TIMEOUT_S = 60.0
CLI_TIMEOUT_S = 120.0
RETRIES = 2
BACKOFF_S: tuple[float, ...] = (1.0, 3.0)      # пауза перед 1-м и 2-м повтором; тесты обнуляют
MAX_RETRY_AFTER_S = 10.0

DEFAULT_MODELS: dict[ProviderKind, str] = {
    ProviderKind.CLAUDE_CLI: "sonnet",
    ProviderKind.ANTHROPIC: "claude-sonnet-4-5",
    ProviderKind.DEEPSEEK: "deepseek-flash",     # актуальное имя; deepseek-v4-flash/deepseek-chat — устаревшие
    ProviderKind.GEMINI: "gemini-3.5-flash-lite",  # 2.5 открыты только старым пользователям API
    ProviderKind.OPENAI_COMPAT: "gpt-4o-mini",
    ProviderKind.CODEX_CLI: "default",           # «как настроено в самом Codex»: флаг --model не передаём
}
DEFAULT_BASE_URLS: dict[ProviderKind, str] = {
    ProviderKind.ANTHROPIC: "https://api.anthropic.com",
    ProviderKind.DEEPSEEK: "https://api.deepseek.com",
    ProviderKind.GEMINI: "https://generativelanguage.googleapis.com",
}
KEYLESS_KINDS = {ProviderKind.CLAUDE_CLI, ProviderKind.CODEX_CLI}


def needs_key(kind: ProviderKind) -> bool:
    return kind not in KEYLESS_KINDS


@runtime_checkable
class LLMProvider(Protocol):
    id: str

    async def complete(self, system: str, user: str, *, json_schema: dict | None = None,
                       max_tokens: int = 1024, temperature: float = 0.4) -> str: ...

    async def check(self) -> tuple[bool, str]: ...


# ─────────────────────────── разбор JSON из ответа модели ───────────────────────────
_FENCE = re.compile(r"^\s*```[a-zA-Z]*\s*\n?|\n?\s*```\s*$")


def extract_json(text: str) -> Any:
    """Достать JSON из ответа модели (допускает ```-ограждение и текст вокруг)."""
    s = _FENCE.sub("", text or "").strip()
    try:
        return json.loads(s)
    except ValueError:
        pass
    dec = json.JSONDecoder()
    for i, ch in enumerate(s):
        if ch in "{[":
            try:
                obj, _ = dec.raw_decode(s[i:])
                return obj
            except ValueError:
                continue
    raise EngineError("Модель ответила не в том формате (ожидался JSON). Попробуй ещё раз или выбери другую модель.")


# ─────────────────────────── общая часть HTTP ───────────────────────────
def _scrub(text: str, secret: str | None) -> str:
    if secret:
        text = text.replace(secret, "***")
    return text


def _error_message(body: str) -> str:
    """Короткий текст ошибки из тела ответа API."""
    try:
        data = json.loads(body)
    except ValueError:
        return body.strip()[:200]
    if isinstance(data, list) and data:
        data = data[0]
    if isinstance(data, dict):
        err = data.get("error", data)
        if isinstance(err, dict):
            return str(err.get("message") or err.get("status") or "")[:200]
        if isinstance(err, str):
            return err[:200]
        return str(data.get("message") or "")[:200]
    return ""


KEY_BAD = "Ключ ИИ не подходит — проверь, что скопировал его целиком."
RATE_LIMIT = "Лимит запросов исчерпан, попробуй позже."
SERVICE_DOWN = "Сервис ИИ сейчас недоступен."


def _ai_error(msg: str, reason: str, *, status: int | None = None, raw: str = "", key: str | None = None) -> EngineError:
    """EngineError с русским текстом для человека; сырой ответ провайдера — в debug (без ключа)."""
    debug = redact(_scrub(f"HTTP {status}: {raw}" if status else raw, key)).strip()[:1500] or None
    err = EngineError(msg, debug=debug)
    err.reason = reason                    # type: ignore[attr-defined]  auth | rate | server | network | money | other
    return err


def _map_http_error(title: str, status: int, body: str, key: str | None) -> EngineError:
    low = body.lower()
    detail = _scrub(_error_message(body), key)
    raw = body[:800]
    bad_key_400 = status == 400 and any(w in low for w in ("api key not valid", "invalid api key", "invalid_api_key",
                                                           "api_key_invalid", "incorrect api key"))
    if status == 403 and ("location" in low or "region" in low):
        return _ai_error(f"«{title}»: сервис недоступен из твоей страны или сети.", "server", status=status, raw=raw, key=key)
    if status in (401, 403) or bad_key_400:
        return _ai_error(KEY_BAD, "auth", status=status, raw=raw, key=key)
    if status == 402 or any(w in low for w in ("insufficient", "billing", "quota", "balance", "credit")) and status in (400, 429):
        return _ai_error(f"«{title}»: закончились деньги или лимит на счёте. Пополни баланс у провайдера.",
                         "money", status=status, raw=raw, key=key)
    if status == 429:
        return _ai_error(RATE_LIMIT, "rate", status=status, raw=raw, key=key)
    if status in (404,) or (status == 400 and "model" in low and any(w in low for w in ("not found", "does not exist", "invalid", "unknown", "not supported"))):
        return _ai_error(f"«{title}»: модель не найдена. Проверь название модели в настройках.", "other",
                         status=status, raw=raw, key=key)
    if status >= 500:
        return _ai_error(SERVICE_DOWN, "server", status=status, raw=raw, key=key)
    msg = f"«{title}»: запрос отклонён (ошибка {status})."
    if detail:
        msg += f" {detail}"
    return _ai_error(msg, "other", status=status, raw=raw, key=key)


class _HttpProvider:
    """Базовый класс: ключ из keyring, повторы на 429/5xx, понятные ошибки."""

    def __init__(self, cfg: ProviderConfig, *, api_key: str | None = None,
                 transport: httpx.AsyncBaseTransport | None = None) -> None:
        self.cfg = cfg
        self.id = cfg.id
        self._api_key = api_key
        self._transport = transport
        self.last_reason = ""            # почему check() не прошёл: auth | rate | server | network | other

    # -- ключ
    def _key(self) -> str:
        key = self._api_key or secrets.get_key(self.cfg.id)
        if not key:
            raise EngineError(f"Для «{self.cfg.title}» не задан ключ. Добавь его в настройках ИИ.")
        return key

    @property
    def _base(self) -> str:
        base = self.cfg.base_url or DEFAULT_BASE_URLS.get(self.cfg.kind)
        if not base:
            raise EngineError(f"Для «{self.cfg.title}» не указан адрес сервиса (base_url).")
        return base.rstrip("/")

    def _remember(self, status: str) -> None:
        """Запомнить, подошёл ли сохранённый ключ (ключ, переданный явно при проверке, не считается)."""
        if self._api_key is not None:
            return
        try:
            keystatus.set_status(self.cfg.id, status, keystatus.KEY_INVALID_MSG if status == "invalid" else None)  # type: ignore[arg-type]
        except Exception as e:  # noqa: BLE001 — статус вторичен, запрос из-за него не роняем
            log.debug("статус ключа не записан: %s", e)

    # -- запрос
    async def _request(self, method: str, url: str, *, headers: dict[str, str], json_body: dict | None = None,
                       key: str | None = None) -> Any:
        title = self.cfg.title
        last: EngineError | None = None
        async with httpx.AsyncClient(timeout=TIMEOUT_S, transport=self._transport) as client:
            for attempt in range(RETRIES + 1):
                try:
                    resp = await client.request(method, url, headers=headers, json=json_body)
                except httpx.TimeoutException:
                    raise _ai_error(f"«{title}» не ответила за {int(TIMEOUT_S)} с. Попробуй ещё раз.", "network") from None
                except httpx.ConnectError:
                    raise _ai_error(f"Нет связи с «{title}». Проверь интернет и адрес сервиса.", "network") from None
                except httpx.InvalidURL:
                    raise _ai_error(f"Адрес сервиса «{title}» указан неверно.", "other") from None
                except httpx.HTTPError as e:
                    raise _ai_error(f"«{title}»: ошибка сети ({type(e).__name__}).", "network") from None
                if resp.status_code < 400:
                    try:
                        data = resp.json()
                    except ValueError:
                        raise _ai_error(f"«{title}» вернула не JSON. Проверь адрес сервиса.", "other") from None
                    self._remember("ok")
                    return data
                last = _map_http_error(title, resp.status_code, resp.text, key)
                last.status = resp.status_code          # type: ignore[attr-defined]
                last.body = _scrub(resp.text[:1000], key)  # type: ignore[attr-defined]
                if getattr(last, "reason", "") == "auth":
                    self._remember("invalid")
                if resp.status_code in (429, 500, 502, 503, 504, 529) and attempt < RETRIES:
                    await asyncio.sleep(self._delay(resp, attempt))
                    continue
                raise last
        raise last or EngineError(f"«{title}»: неизвестная ошибка.")

    @staticmethod
    def _delay(resp: httpx.Response, attempt: int) -> float:
        base = BACKOFF_S[min(attempt, len(BACKOFF_S) - 1)] if BACKOFF_S else 0.0
        try:
            ra = float(resp.headers.get("retry-after", ""))
            return min(max(ra, base), MAX_RETRY_AFTER_S) if base else 0.0
        except ValueError:
            return base

    async def check(self) -> tuple[bool, str]:
        self.last_reason = ""
        try:
            # запас по токенам: модели с «размышлениями» (Gemini 3.x и др.) тратят их до ответа
            text = await self.complete("Отвечай одним словом.", "Скажи «ок».", max_tokens=1024, temperature=0.0)
        except EngineError as e:
            self.last_reason = getattr(e, "reason", "other")
            return False, str(e)
        if not text.strip():
            return False, f"«{self.cfg.title}» подключена, но ответила пустотой. Проверь модель."
        return True, f"«{self.cfg.title}» работает (модель {self.cfg.model})."

    async def complete(self, system: str, user: str, *, json_schema: dict | None = None,
                       max_tokens: int = 1024, temperature: float = 0.4) -> str:  # pragma: no cover
        raise NotImplementedError


# ─────────────────────────── OpenAI-совместимые (и DeepSeek) ───────────────────────────
class OpenAICompatProvider(_HttpProvider):
    """/chat/completions. Для json_schema пробует нативный response_format, при отказе — json_object + схема в промпте."""

    supports_json_schema = True
    #: добавляется в каждый запрос (у DeepSeek — выключить «размышления»)
    extra_body: dict[str, Any] = {}

    async def complete(self, system: str, user: str, *, json_schema: dict | None = None,
                       max_tokens: int = 1024, temperature: float = 0.4) -> str:
        key = self._key()
        url = self._base + "/chat/completions"
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        mode = "none"
        if json_schema:
            mode = "schema" if self.supports_json_schema else "object"
        use_temp, tokens_field = True, "max_tokens"
        for _ in range(4):
            sys_text = system
            body: dict[str, Any] = {"model": self.cfg.model, tokens_field: max_tokens, **self.extra_body}
            if use_temp:
                body["temperature"] = temperature
            if mode == "schema":
                body["response_format"] = {"type": "json_schema",
                                           "json_schema": {"name": "result", "schema": json_schema}}
            elif mode == "object":
                body["response_format"] = {"type": "json_object"}
                sys_text = system + "\n\n" + prompts.schema_instruction(json_schema or {})
            body["messages"] = [{"role": "system", "content": sys_text}, {"role": "user", "content": user}]
            try:
                data = await self._request("POST", url, headers=headers, json_body=body, key=key)
            except EngineError as e:
                status, low = getattr(e, "status", 0), getattr(e, "body", "").lower()
                if status == 400:
                    if mode == "schema" and ("response_format" in low or "json_schema" in low):
                        mode = "object"
                        continue
                    if use_temp and "temperature" in low:
                        use_temp = False
                        continue
                    if tokens_field == "max_tokens" and "max_completion_tokens" in low:
                        tokens_field = "max_completion_tokens"
                        continue
                raise
            return self._parse(data)
        raise EngineError(f"«{self.cfg.title}» отклонила запрос. Проверь модель и адрес сервиса.")

    def _parse(self, data: Any) -> str:
        try:
            choice = data["choices"][0]
            msg = choice["message"]
            text = msg.get("content") or ""
        except (KeyError, IndexError, TypeError, AttributeError):
            raise EngineError(f"«{self.cfg.title}» ответила в неожиданном формате. Проверь адрес сервиса.") from None
        if not text.strip() and msg.get("reasoning_content") and choice.get("finish_reason") == "length":
            # модель «размышляла» и потратила весь лимит, до ответа не дошла
            raise EngineError(f"«{self.cfg.title}» не уместила ответ в лимит: модель долго размышляет. "
                              "Выбери модель попроще.")
        return text


class DeepSeekProvider(OpenAICompatProvider):
    """DeepSeek: тот же протокол, base_url https://api.deepseek.com; нативной json_schema нет — только json_object.

    «Размышления» (thinking) у DeepSeek включены по умолчанию: ответ ждёт дольше, стоит дороже, а при малом
    max_tokens весь лимит уходит в reasoning_content и content пустой. Нашим коротким задачам они не нужны.
    """

    supports_json_schema = False
    extra_body = {"thinking": {"type": "disabled"}}


# ─────────────────────────── Anthropic ───────────────────────────
class AnthropicProvider(_HttpProvider):
    async def complete(self, system: str, user: str, *, json_schema: dict | None = None,
                       max_tokens: int = 1024, temperature: float = 0.4) -> str:
        key = self._key()
        headers = {"x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"}
        sys_text = system + ("\n\n" + prompts.schema_instruction(json_schema) if json_schema else "")
        body = {
            "model": self.cfg.model, "max_tokens": max_tokens, "temperature": temperature,
            "system": sys_text, "messages": [{"role": "user", "content": user}],
        }
        data = await self._request("POST", self._base + "/v1/messages", headers=headers, json_body=body, key=key)
        try:
            text = "".join(b.get("text", "") for b in data["content"] if b.get("type") == "text")
        except (KeyError, TypeError, AttributeError):
            raise EngineError(f"«{self.cfg.title}» ответила в неожиданном формате.") from None
        return text


# ─────────────────────────── Gemini ───────────────────────────
class GeminiProvider(_HttpProvider):
    """generateContent. Ключ передаём заголовком x-goog-api-key, а не в адресе — адреса попадают в логи."""

    async def complete(self, system: str, user: str, *, json_schema: dict | None = None,
                       max_tokens: int = 1024, temperature: float = 0.4) -> str:
        key = self._key()
        model = self.cfg.model.removeprefix("models/")
        url = f"{self._base}/v1beta/models/{quote(model, safe='.-_')}:generateContent"
        headers = {"x-goog-api-key": key, "content-type": "application/json"}
        gen: dict[str, Any] = {"maxOutputTokens": max_tokens, "temperature": temperature}
        sys_text = system
        if json_schema:
            gen["responseMimeType"] = "application/json"
            sys_text = system + "\n\n" + prompts.schema_instruction(json_schema)
        body = {
            "systemInstruction": {"parts": [{"text": sys_text}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": gen,
        }
        data = await self._request("POST", url, headers=headers, json_body=body, key=key)
        try:
            cand = data["candidates"][0]
        except (KeyError, IndexError, TypeError):
            reason = (data.get("promptFeedback") or {}).get("blockReason") if isinstance(data, dict) else None
            if reason:
                raise EngineError(f"«{self.cfg.title}» отказалась отвечать на этот текст (фильтр безопасности).") from None
            raise EngineError(f"«{self.cfg.title}» ответила в неожиданном формате.") from None
        parts = (cand.get("content") or {}).get("parts") or []
        text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
        if not text and cand.get("finishReason") in ("SAFETY", "RECITATION", "PROHIBITED_CONTENT"):
            raise EngineError(f"«{self.cfg.title}» отказалась отвечать на этот текст (фильтр безопасности).")
        if not text and cand.get("finishReason") == "MAX_TOKENS":
            raise EngineError(f"«{self.cfg.title}» не уместила ответ в лимит. Выбери модель без долгих размышлений.")
        return text


# ─────────────────────────── CLI-подписки: Claude Code и Codex ───────────────────────────
async def _run_cli(cmd: list[str], stdin: bytes, *, env: dict[str, str], title: str, cwd: str) -> tuple[int | None, str, str]:
    """Запустить CLI, отдать промпт через stdin. Ошибки запуска и таймаут — EngineError с причиной для запасной модели."""
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd, stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE, cwd=cwd, env=env)
    except OSError as e:
        raise _ai_error(f"Не удалось запустить {title}: {e.strerror or type(e).__name__}.", "unavailable") from None
    try:
        out, err = await asyncio.wait_for(proc.communicate(stdin), CLI_TIMEOUT_S)
    except TimeoutError:
        try:
            proc.kill()
        except ProcessLookupError:
            pass
        raise _ai_error(f"{title} не ответил за {int(CLI_TIMEOUT_S)} с. Попробуй ещё раз.", "server") from None
    return proc.returncode, out.decode("utf-8", "replace"), err.decode("utf-8", "replace")


def _cli_text_error(spec: cli.CliSpec, code: int | None, text: str) -> EngineError:
    """Ошибка CLI по тексту, который он напечатал: вход, лимит подписки, модель, сеть."""
    text = (text or "").strip()
    low = text.lower()
    if any(w in low for w in ("not logged in", "please run /login", "/login", "authenticate", "oauth", "invalid api key",
                              "401", "unauthorized", "log in", "sign in", "refresh token")):
        cli.forget_login(spec.kind)
        return _ai_error(f"В {spec.title} нужно войти. Открой Настройки → Модели ИИ и нажми «Войти».", "auth", raw=text)
    if any(w in low for w in ("limit", "quota", "usage", "rate", "429", "overloaded", "capacity")):
        return _ai_error(f"В {spec.title} закончился лимит подписки. Подожди сброса лимита или выбери другую модель.",
                         "rate", raw=text)
    if any(w in low for w in ("network", "connection", "dns", "timed out", "econn", "enotfound", "fetch failed")):
        return _ai_error(f"{spec.title} не достучался до сервиса. Проверь интернет.", "network", raw=text)
    if "model" in low and any(w in low for w in ("not found", "invalid", "unknown", "issue", "not supported", "does not exist")):
        return _ai_error(f"{spec.title}: модель не найдена. Проверь название модели в настройках.", "other", raw=text)
    return _ai_error(f"{spec.title} завершился с ошибкой" + (f": {text[:200]}" if text else f" (код {code})") + ".",
                     "server", raw=text)


class _CliProvider:
    """Общая часть CLI-моделей: поиск файла, проверка, учёт «отвечал в этой сессии»."""

    kind: ProviderKind

    def __init__(self, cfg: ProviderConfig, *, api_key: str | None = None, transport: Any = None) -> None:
        self.cfg = cfg
        self.id = cfg.id
        self.last_reason = ""

    @property
    def spec(self) -> cli.CliSpec:
        return cli.CLI_SPECS[self.kind]

    @classmethod
    def binary(cls) -> str | None:
        return cli.find_binary(cli.CLI_SPECS[cls.kind].binary)

    def _need_binary(self) -> str:
        exe = self.binary()
        if not exe:
            raise _ai_error(cli.not_installed_text(self.spec), "unavailable")
        return exe

    async def check(self) -> tuple[bool, str]:
        self.last_reason = ""
        if not self.binary():
            self.last_reason = "unavailable"
            return False, cli.not_installed_text(self.spec)
        try:
            text = await self.complete("Отвечай одним словом.", "Скажи «ок».")
        except EngineError as e:
            self.last_reason = getattr(e, "reason", "other")
            return False, str(e)
        if not text.strip():
            return False, f"{self.spec.title} ответил пустотой."
        return True, f"{self.spec.title} работает (модель {self.cfg.model or 'по умолчанию'})."

    async def complete(self, system: str, user: str, *, json_schema: dict | None = None,
                       max_tokens: int = 1024, temperature: float = 0.4) -> str:  # pragma: no cover
        raise NotImplementedError


class ClaudeCliProvider(_CliProvider):
    """`claude -p` — использует вход пользователя в Claude Code, ключ не нужен."""

    kind = ProviderKind.CLAUDE_CLI

    async def complete(self, system: str, user: str, *, json_schema: dict | None = None,
                       max_tokens: int = 1024, temperature: float = 0.4) -> str:
        exe = self._need_binary()
        sys_text = system + ("\n\n" + prompts.schema_instruction(json_schema) if json_schema else "")
        cmd = [exe, "-p", "--output-format", "json", "--no-session-persistence", "--tools", ""]
        if self.cfg.model:
            cmd += ["--model", self.cfg.model]
        cmd += ["--append-system-prompt", sys_text]
        with tempfile.TemporaryDirectory(prefix="dubl-claude-") as cwd:   # чтобы не подхватить чужие CLAUDE.md
            code, stdout, stderr = await _run_cli(cmd, user.encode("utf-8"), env=cli.child_env(exe), title=self.spec.title, cwd=cwd)
        data: Any = None
        try:
            data = json.loads(stdout)
        except ValueError:
            pass
        if isinstance(data, list):                       # stream-подобный вывод: берём итоговый элемент
            data = next((d for d in reversed(data) if isinstance(d, dict) and d.get("type") == "result"), None)
        if code != 0 or not isinstance(data, dict) or data.get("is_error"):
            raise self._cli_error(code, data, stderr)
        result = data.get("result")
        if not isinstance(result, str):
            raise _ai_error("Claude Code вернул ответ в неожиданном формате. Обнови Claude Code.", "other")
        cli.mark_verified(self.id)
        return result

    @classmethod
    def _cli_error(cls, code: int | None, data: Any, stderr: str) -> EngineError:
        text = ""
        if isinstance(data, dict):
            text = str(data.get("result") or data.get("error") or "")
        return _cli_text_error(cli.CLI_SPECS[cls.kind], code, text or stderr)


class CodexCliProvider(_CliProvider):
    """`codex exec` — вход в Codex по подписке ChatGPT, ключ не нужен.

    Флаги — по документации OpenAI (developers.openai.com/codex/noninteractive, …/cli/reference): промпт со stdin
    (`-`), песочница read-only (команды модели ничего не пишут), без проверки git-репозитория, последний ответ — в файл
    (`--output-last-message`). Необязательные флаги (`--ephemeral`, `--ignore-user-config`, `--color never`) снимаем,
    если установленная версия их не знает. Живьём на настоящем Codex не проверено (на этом Mac его нет).
    """

    kind = ProviderKind.CODEX_CLI
    OPTIONAL_FLAGS: ClassVar[tuple[str, ...]] = ("--ephemeral", "--ignore-user-config", "--color", "never")

    async def complete(self, system: str, user: str, *, json_schema: dict | None = None,
                       max_tokens: int = 1024, temperature: float = 0.4) -> str:
        exe = self._need_binary()
        sys_text = system + ("\n\n" + prompts.schema_instruction(json_schema) if json_schema else "")
        prompt = f"{sys_text}\n\n---\n\n{user}"          # у codex exec нет отдельного системного промпта
        model = (self.cfg.model or "").strip()
        with tempfile.TemporaryDirectory(prefix="dubl-codex-") as cwd:
            out_file = os.path.join(cwd, "last.txt")
            base = [exe, "exec", "--skip-git-repo-check", "--sandbox", "read-only", "--output-last-message", out_file]
            if model and model != "default":
                base += ["--model", model]
            code, stdout, stderr = 0, "", ""
            for attempt in range(2):
                cmd = base + (list(self.OPTIONAL_FLAGS) if attempt == 0 else []) + ["-"]
                code, stdout, stderr = await _run_cli(cmd, prompt.encode("utf-8"), env=cli.child_env(exe),
                                                      title=self.spec.title, cwd=cwd)
                low = stderr.lower()
                if code != 0 and attempt == 0 and ("unexpected argument" in low or "unrecognized" in low or "unknown option" in low):
                    continue                                 # старая версия без необязательных флагов
                break
            try:
                text = await asyncio.to_thread(Path(out_file).read_text, encoding="utf-8")
            except OSError:
                text = ""
        text = (text or stdout).strip()
        if code != 0 or not text:
            if code == 0:
                raise _ai_error(f"{self.spec.title} ответил пустотой. Попробуй ещё раз.", "other", raw=stderr)
            raise _cli_text_error(self.spec, code, stderr or stdout)
        cli.mark_verified(self.id)
        return text


# ─────────────────────────── фабрика и список моделей ───────────────────────────
def make_provider(cfg: ProviderConfig, *, api_key: str | None = None,
                  transport: httpx.AsyncBaseTransport | None = None) -> LLMProvider:
    """Собрать провайдера по настройке. api_key=None → ключ берётся из keyring при запросе."""
    kind = cfg.kind
    if kind == ProviderKind.CLAUDE_CLI:
        return ClaudeCliProvider(cfg)
    if kind == ProviderKind.CODEX_CLI:
        return CodexCliProvider(cfg)
    if kind == ProviderKind.ANTHROPIC:
        return AnthropicProvider(cfg, api_key=api_key, transport=transport)
    if kind == ProviderKind.GEMINI:
        return GeminiProvider(cfg, api_key=api_key, transport=transport)
    if kind == ProviderKind.DEEPSEEK:
        return DeepSeekProvider(cfg, api_key=api_key, transport=transport)
    if kind == ProviderKind.OPENAI_COMPAT:
        return OpenAICompatProvider(cfg, api_key=api_key, transport=transport)
    raise EngineError(f"Неизвестный тип модели: {kind}")


CLAUDE_CLI_MODELS = ["haiku", "sonnet", "opus"]
CODEX_CLI_MODELS = ["default"]


async def list_models(cfg: ProviderConfig, *, api_key: str | None = None,
                      transport: httpx.AsyncBaseTransport | None = None) -> list[str]:
    """Список моделей, доступных по ключу (для выпадающего списка в настройках)."""
    if cfg.kind == ProviderKind.CLAUDE_CLI:
        return list(CLAUDE_CLI_MODELS)
    if cfg.kind == ProviderKind.CODEX_CLI:
        return list(CODEX_CLI_MODELS)
    p = make_provider(cfg, api_key=api_key, transport=transport)
    assert isinstance(p, _HttpProvider)
    key = p._key()
    if cfg.kind == ProviderKind.GEMINI:
        data = await p._request("GET", p._base + "/v1beta/models?pageSize=200",
                                headers={"x-goog-api-key": key}, key=key)
        out = []
        for m in (data.get("models") or []) if isinstance(data, dict) else []:
            if "generateContent" in (m.get("supportedGenerationMethods") or []):
                out.append(str(m.get("name", "")).removeprefix("models/"))
        return sorted(x for x in out if x)
    if cfg.kind == ProviderKind.ANTHROPIC:
        headers = {"x-api-key": key, "anthropic-version": "2023-06-01"}
        data = await p._request("GET", p._base + "/v1/models?limit=100", headers=headers, key=key)
    else:
        data = await p._request("GET", p._base + "/models", headers={"Authorization": f"Bearer {key}"}, key=key)
    items = data.get("data") if isinstance(data, dict) else data
    ids = [str(m.get("id")) for m in (items or []) if isinstance(m, dict) and m.get("id")]
    return sorted(ids)


# ─────────────────────────── проверка ключа ───────────────────────────
CHECK_TIMEOUT_S = 10.0
CLI_CHECK_TIMEOUT_S = 40.0          # CLI стартует дольше HTTP-запроса


async def verify_key(cfg: ProviderConfig, *, api_key: str | None = None,
                     transport: httpx.AsyncBaseTransport | None = None,
                     timeout: float | None = None) -> tuple[str, str | None, str]:
    """Один короткий запрос к модели: (key_status, key_message, текст для «Проверить»).

    ok — модель ответила; invalid — ключ не подошёл (401/403/«API key not valid»); unknown — не вышло проверить
    (нет сети, таймаут, лимит, сервис лёг, неверный адрес) — ключ при этом не считается плохим.
    Результат при api_key=None сам запоминается в llm.keystatus (в _HttpProvider._remember).
    """
    is_cli = cli.is_cli_kind(cfg.kind)
    timeout = (CLI_CHECK_TIMEOUT_S if is_cli else CHECK_TIMEOUT_S) if timeout is None else timeout
    p = make_provider(cfg, api_key=api_key, transport=transport)
    try:
        ok, msg = await asyncio.wait_for(p.check(), timeout)
    except TimeoutError:
        return "unknown", f"«{cfg.title}» не ответила за {int(timeout)} с — ключ не проверен.", \
            f"«{cfg.title}» не ответила за {int(timeout)} с. Попробуй ещё раз."
    if ok:
        return "ok", None, msg
    if is_cli:                       # у CLI нет ключа: «не вошёл» — это не «ключ не подходит»
        return "unknown", msg, msg
    if getattr(p, "last_reason", "") == "auth":
        return "invalid", keystatus.KEY_INVALID_MSG, msg
    return "unknown", msg, msg
