"""Подписки через CLI: поиск программы при узком PATH, статус, Codex с фейковым бинарником, кнопка «Войти»."""
from __future__ import annotations

import json
import os
import stat
import sys
from pathlib import Path

import pytest

from dubl_engine.llm import cli, providers
from dubl_engine.llm.providers import ClaudeCliProvider, CodexCliProvider, make_provider
from dubl_engine.models import ProviderConfig, ProviderKind
from dubl_engine.util import EngineError

K = ProviderKind
_REAL_FIND = cli.find_binary      # до того, как conftest подменит его заглушкой
pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="фейковые бинарники — shell-скрипты")


@pytest.fixture
def real_find(monkeypatch):
    """Настоящий поиск программы (conftest его глушит) — в тестах поиска."""
    monkeypatch.setattr(cli, "find_binary", _REAL_FIND)
    cli.reset_cache_for_tests()
    return cli


def _script(path, body: str):
    path.write_text("#!/bin/sh\n" + body, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return str(path)


def _cfg(kind: ProviderKind, **kw) -> ProviderConfig:
    return ProviderConfig(**{"id": kind.value, "kind": kind, "title": kind.value, "model": "m", **kw})


# ───────────── поиск программы ─────────────
def test_find_binary_in_local_bin_when_path_is_narrow(tmp_path, monkeypatch, real_find):
    """Главный случай: приложение из Finder → PATH без ~/.local/bin, а claude лежит именно там."""
    home = tmp_path / "home"
    (home / ".local" / "bin").mkdir(parents=True)
    exe = _script(home / ".local" / "bin" / "claude", "echo hi")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("PATH", "/usr/bin:/bin:/usr/sbin:/sbin")
    monkeypatch.setattr(cli, "_from_login_shell", lambda name: None)
    assert cli.find_binary("claude", refresh=True) == exe


def test_find_binary_nvm_versions_newest_first(tmp_path, monkeypatch, real_find):
    home = tmp_path / "home"
    for v in ("v18.0.0", "v20.11.1"):
        (home / ".nvm" / "versions" / "node" / v / "bin").mkdir(parents=True)
    _script(home / ".nvm" / "versions" / "node" / "v18.0.0" / "bin" / "codex", "echo old")
    new = _script(home / ".nvm" / "versions" / "node" / "v20.11.1" / "bin" / "codex", "echo new")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    monkeypatch.setattr(cli, "_from_login_shell", lambda name: None)
    assert cli.find_binary("codex", refresh=True) == new


def test_node_version_key_orders_numerically_not_by_listing_order():
    """Порядок iterdir зависит от ФС (ext4 — хэш, APFS — по алфавиту); сортируем по номеру версии."""
    names = ["v9.11.2", "v20.11.1", "v18.0.0", "v20.9.0"]
    got = sorted((Path("/x/.nvm/versions/node") / n for n in names), key=cli._node_version_key, reverse=True)
    assert [p.name for p in got] == ["v20.11.1", "v20.9.0", "v18.0.0", "v9.11.2"]


def test_find_binary_falls_back_to_login_shell_and_caches(tmp_path, monkeypatch, real_find):
    monkeypatch.setenv("HOME", str(tmp_path / "empty"))
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    exe = _script(tmp_path / "weird_claude", "echo hi")
    calls = []
    monkeypatch.setattr(cli, "_from_login_shell", lambda name: (calls.append(name), exe)[1])
    assert cli.find_binary("claude", refresh=True) == exe
    assert cli.find_binary("claude") == exe
    assert calls == ["claude"]                          # входную оболочку (медленную) спросили один раз


def test_find_binary_missing_is_remembered_briefly(tmp_path, monkeypatch, real_find):
    monkeypatch.setenv("HOME", str(tmp_path / "empty"))
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    calls = []
    monkeypatch.setattr(cli, "_from_login_shell", lambda name: (calls.append(1), None)[1])
    assert cli.find_binary("nope-cli", refresh=True) is None
    assert cli.find_binary("nope-cli") is None
    assert len(calls) == 1
    assert cli.find_binary("nope-cli", refresh=True) is None   # «Проверить заново» — спрашивает снова
    assert len(calls) == 2


def test_login_shell_output_ignores_aliases_and_noise(tmp_path, monkeypatch, real_find):
    exe = _script(tmp_path / "claude", "echo hi")

    class Done:
        stdout = f"claude: aliased to something\ncommand not found: compdef\n{exe}\n"
        stderr = ""
    monkeypatch.setattr(cli.subprocess, "run", lambda *a, **k: Done())
    assert cli._from_login_shell("claude") == exe


def test_child_env_adds_binary_dir_first(tmp_path, monkeypatch):
    exe = tmp_path / "bin" / "claude"
    exe.parent.mkdir()
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    env = cli.child_env(str(exe))
    parts = env["PATH"].split(os.pathsep)
    assert parts[0] == str(exe.parent) and "/usr/bin" in parts and len(parts) == len(set(parts))


# ───────────── статус ─────────────
def _status_with(monkeypatch, tmp_path, kind, body: str):
    exe = _script(tmp_path / ("claude" if kind == K.CLAUDE_CLI else "codex"), body)
    monkeypatch.setattr(cli, "find_binary", lambda name, refresh=False: exe)
    cli.reset_cache_for_tests()
    return cli.cli_status(kind, refresh=True)


def test_status_not_installed():
    st = cli.cli_status(K.CLAUDE_CLI, refresh=True)
    assert st.state == "not_installed" and "Claude Code не найден" in st.message and not st.can_answer


def test_claude_status_logged_in_and_out(tmp_path, monkeypatch):
    ok = _status_with(monkeypatch, tmp_path, K.CLAUDE_CLI, 'echo \'{"loggedIn": true, "authMethod": "claude.ai"}\'')
    assert ok.state == "ok" and ok.can_answer
    out = _status_with(monkeypatch, tmp_path, K.CLAUDE_CLI, 'echo \'{"loggedIn": false}\'; exit 1')
    assert out.state == "not_logged_in" and "Войти" in out.message


def test_codex_status_uses_login_status_exit_code(tmp_path, monkeypatch):
    ok = _status_with(monkeypatch, tmp_path, K.CODEX_CLI, '[ "$1 $2" = "login status" ] && echo "Logged in using ChatGPT" && exit 0; exit 2')
    assert ok.state == "ok"
    out = _status_with(monkeypatch, tmp_path, K.CODEX_CLI, 'echo "Not logged in" >&2; exit 1')
    assert out.state == "not_logged_in"


def test_status_check_failure_does_not_erase_last_good(tmp_path, monkeypatch):
    exe = _script(tmp_path / "claude", 'echo \'{"loggedIn": true}\'')
    monkeypatch.setattr(cli, "find_binary", lambda name, refresh=False: exe)
    cli.reset_cache_for_tests()
    assert cli.cli_status(K.CLAUDE_CLI, refresh=True).state == "ok"
    monkeypatch.setattr(cli, "_run_status", lambda cmd, env: (None, "timeout"))      # разовый сбой проверки
    assert cli.cli_status(K.CLAUDE_CLI, refresh=True).state == "ok"                  # число моделей не прыгает


# ───────────── Claude Code: запрос ─────────────
@pytest.mark.asyncio
async def test_claude_runs_with_binary_dir_in_child_path(tmp_path, monkeypatch):
    """Бинарник найден вне PATH → в дочерний процесс уходит PATH с его каталогом (node рядом)."""
    exe = _script(tmp_path / "claude", 'echo "{\\"type\\":\\"result\\",\\"is_error\\":false,\\"result\\":\\"PATH=$PATH\\"}"')
    monkeypatch.setattr(cli, "find_binary", lambda name, refresh=False: exe)
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    text = await ClaudeCliProvider(_cfg(K.CLAUDE_CLI, model="haiku")).complete("s", "u")
    assert text.startswith(f"PATH={tmp_path}")
    assert cli.is_verified("claude_cli")


# ───────────── Codex CLI: фейковый бинарник ─────────────
CODEX_FAKE = r'''
out=""; prev=""; args="$*"
for a in "$@"; do
  [ "$prev" = "--output-last-message" ] && out="$a"
  prev="$a"
done
echo "$args" > "$FAKE_LOG"
cat > "$FAKE_STDIN"
case "$FAKE_MODE" in
  fail_login) echo "ERROR: not logged in, please run codex login" >&2; exit 1;;
  limit) echo "You've hit your usage limit" >&2; exit 1;;
  old) case "$args" in *--ephemeral*) echo "error: unexpected argument '--ephemeral' found" >&2; exit 2;; esac;;
esac
echo '{"type":"turn.completed"}'
printf 'Привет, мир' > "$out"
'''


@pytest.fixture
def fake_codex(tmp_path, monkeypatch):
    exe = _script(tmp_path / "codex", CODEX_FAKE)
    monkeypatch.setattr(cli, "find_binary", lambda name, refresh=False: exe if name == "codex" else None)
    monkeypatch.setenv("FAKE_LOG", str(tmp_path / "args.txt"))
    monkeypatch.setenv("FAKE_STDIN", str(tmp_path / "stdin.txt"))
    monkeypatch.setenv("FAKE_MODE", "ok")
    return tmp_path


@pytest.mark.asyncio
async def test_codex_exec_flags_and_prompt_via_stdin(fake_codex):
    p = make_provider(_cfg(K.CODEX_CLI, model="gpt-x"))
    assert isinstance(p, CodexCliProvider)
    assert await p.complete("системный", "вопрос", json_schema={"type": "object"}) == "Привет, мир"
    args = (fake_codex / "args.txt").read_text(encoding="utf-8").split()
    assert args[0] == "exec" and args[-1] == "-"                       # промпт со stdin
    assert args[args.index("--sandbox") + 1] == "read-only"            # модель ничего не пишет на диск
    assert "--skip-git-repo-check" in args and "--output-last-message" in args
    assert args[args.index("--model") + 1] == "gpt-x"
    stdin = (fake_codex / "stdin.txt").read_text(encoding="utf-8")
    assert "системный" in stdin and "вопрос" in stdin and "properties" in stdin or "object" in stdin


@pytest.mark.asyncio
async def test_codex_default_model_has_no_model_flag(fake_codex):
    await CodexCliProvider(_cfg(K.CODEX_CLI, model="default")).complete("s", "u")
    assert "--model" not in (fake_codex / "args.txt").read_text(encoding="utf-8").split()


@pytest.mark.asyncio
async def test_codex_old_version_retries_without_optional_flags(fake_codex, monkeypatch):
    monkeypatch.setenv("FAKE_MODE", "old")
    assert await CodexCliProvider(_cfg(K.CODEX_CLI)).complete("s", "u") == "Привет, мир"
    assert "--ephemeral" not in (fake_codex / "args.txt").read_text(encoding="utf-8")


@pytest.mark.asyncio
@pytest.mark.parametrize("mode,fragment,reason", [("fail_login", "нужно войти", "auth"), ("limit", "лимит", "rate")])
async def test_codex_errors_are_russian(fake_codex, monkeypatch, mode, fragment, reason):
    monkeypatch.setenv("FAKE_MODE", mode)
    with pytest.raises(EngineError, match=fragment) as e:
        await CodexCliProvider(_cfg(K.CODEX_CLI)).complete("s", "u")
    assert e.value.reason == reason                                    # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_codex_not_installed_check(monkeypatch):
    ok, msg = await CodexCliProvider(_cfg(K.CODEX_CLI)).check()
    assert not ok and "Codex не найден" in msg


@pytest.mark.asyncio
async def test_codex_check_ok(fake_codex):
    ok, msg = await CodexCliProvider(_cfg(K.CODEX_CLI)).check()
    assert ok and "работает" in msg


@pytest.mark.asyncio
async def test_list_models_codex_and_claude_order(cfg):
    assert await providers.list_models(cfg(K.CODEX_CLI)) == ["default"]
    assert await providers.list_models(cfg(K.CLAUDE_CLI)) == ["haiku", "sonnet", "opus"]


# ───────────── кнопка «Войти» ─────────────
def test_login_command_is_fixed_list(tmp_path, monkeypatch):
    exe = _script(tmp_path / "claude", "")
    monkeypatch.setattr(cli, "find_binary", lambda name, refresh=False: exe)
    assert cli.login_command(K.CLAUDE_CLI) == [exe, "auth", "login"]
    codex = _script(tmp_path / "codex", "")
    monkeypatch.setattr(cli, "find_binary", lambda name, refresh=False: codex)
    assert cli.login_command(K.CODEX_CLI) == [codex, "login"]
    with pytest.raises(EngineError, match="ключ"):
        cli.login_command(K.DEEPSEEK)


def test_login_command_not_installed():
    with pytest.raises(EngineError, match="не найден"):
        cli.login_command(K.CODEX_CLI)


def test_terminal_launcher_macos_uses_command_file_with_quoting(monkeypatch):
    monkeypatch.setattr(cli.sys, "platform", "darwin")
    [launcher] = cli.terminal_launcher(["/Users/me/my dir/claude", "auth", "login"])
    assert launcher[:3] == ["open", "-a", "Terminal"]                  # без Apple Events: окно «управлять Terminal» не нужно
    path = launcher[3]
    body = open(path, encoding="utf-8").read()
    assert path.endswith(".command") and os.access(path, os.X_OK)
    assert "'/Users/me/my dir/claude' auth login" in body and body.startswith("#!/bin/zsh")
    assert "export PATH='/Users/me/my dir':" in body


def test_open_login_terminal_runs_open(tmp_path, monkeypatch):
    exe = _script(tmp_path / "claude", "")
    monkeypatch.setattr(cli, "find_binary", lambda name, refresh=False: exe)
    monkeypatch.setattr(cli.sys, "platform", "darwin")
    monkeypatch.setattr(cli.shutil, "which", lambda n: "/usr/bin/open")
    ran = []

    class R:
        returncode, stderr = 0, ""
    monkeypatch.setattr(cli.subprocess, "run", lambda cmd, **k: (ran.append(cmd), R())[1])
    msg = cli.open_login_terminal(K.CLAUDE_CLI)
    assert "Открыл терминал" in msg and ran[0][:3] == ["open", "-a", "Terminal"]
    assert exe in open(ran[0][3], encoding="utf-8").read()


def test_open_login_terminal_failure_gives_manual_command(tmp_path, monkeypatch):
    exe = _script(tmp_path / "codex", "")
    monkeypatch.setattr(cli, "find_binary", lambda name, refresh=False: exe)
    monkeypatch.setattr(cli.sys, "platform", "darwin")
    monkeypatch.setattr(cli.shutil, "which", lambda n: None)
    with pytest.raises(EngineError, match="codex login"):
        cli.open_login_terminal(K.CODEX_CLI)


def test_json_roundtrip_of_codex_kind():
    cfg = _cfg(K.CODEX_CLI)
    assert ProviderConfig.model_validate_json(cfg.model_dump_json()).kind == K.CODEX_CLI
    assert json.loads(cfg.model_dump_json())["kind"] == "codex_cli"
