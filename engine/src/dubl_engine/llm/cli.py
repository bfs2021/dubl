"""Модели «через подписку»: Claude Code и Codex CLI запускаются как программы, ключ не нужен.

Общая часть для всех таких CLI:
  · поиск исполняемого файла — не только в PATH. Приложение, запущенное из Finder, получает урезанный PATH
    (`/usr/bin:/bin:/usr/sbin:/sbin`), и `claude` из `~/.local/bin` в нём не виден. Ищем в PATH, в типичных каталогах
    установки, в версиях nvm, а в конце спрашиваем у входной оболочки пользователя (`zsh -lic 'command -v X'`);
  · PATH для дочернего процесса: каталог найденного файла + типичные каталоги (CLI на node нужен node рядом);
  · статус: not_installed / not_logged_in / ok / error и причина по-русски;
  · кнопка «Войти»: открыть системный терминал с фиксированной командой входа (произвольные строки не принимаем).

Найденный путь кэшируется до перезапуска движка; «не найден» помним недолго, чтобы установка подхватывалась.
"""
from __future__ import annotations

import json
import logging
import os
import shlex
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from ..models import ProviderKind
from ..util import EngineError

log = logging.getLogger(__name__)

CliState = Literal["not_installed", "not_logged_in", "ok", "error"]

LOGIN_SHELL_TIMEOUT_S = 6.0        # входная оболочка читает .zshrc пользователя — даём запас, но не вечность
STATUS_TIMEOUT_S = 15.0
NEGATIVE_TTL_S = 20.0              # «не найден» помним недолго: человек мог только что установить программу
LOGIN_TTL_S = 30.0                 # результат проверки входа


@dataclass(frozen=True)
class CliSpec:
    kind: ProviderKind
    binary: str                    # имя программы
    title: str                     # как называть в тексте
    install_url: str
    install_hint: str
    login_args: tuple[str, ...]    # аргументы команды входа (фиксированные)
    models: tuple[str, ...]


CLI_SPECS: dict[ProviderKind, CliSpec] = {
    ProviderKind.CLAUDE_CLI: CliSpec(
        kind=ProviderKind.CLAUDE_CLI, binary="claude", title="Claude Code",
        install_url="https://claude.com/product/claude-code",
        install_hint="Установи Claude Code и войди в аккаунт Claude.",
        login_args=("auth", "login"), models=("haiku", "sonnet", "opus")),
    ProviderKind.CODEX_CLI: CliSpec(
        kind=ProviderKind.CODEX_CLI, binary="codex", title="Codex",
        install_url="https://developers.openai.com/codex/cli",
        install_hint="Установи Codex CLI и войди в аккаунт ChatGPT.",
        login_args=("login",), models=("default",)),
}


def spec_for(kind: ProviderKind) -> CliSpec | None:
    return CLI_SPECS.get(kind)


def is_cli_kind(kind: ProviderKind) -> bool:
    return kind in CLI_SPECS


# ─────────────────────────── поиск исполняемого файла ───────────────────────────
_lock = threading.Lock()
_found: dict[str, str] = {}              # имя → путь (до перезапуска движка)
_missing_at: dict[str, float] = {}       # имя → когда не нашли


def _home() -> Path:
    return Path(os.path.expanduser("~"))


def _node_version_key(p: Path) -> tuple:
    parts = p.parent.parent.name.lstrip("v").split(".")        # …/versions/node/v20.11.1/bin
    return tuple(int(x) if x.isdigit() else 0 for x in parts)


def candidate_dirs() -> list[Path]:
    """Каталоги, где обычно лежат CLI, по убыванию вероятности. Несуществующие отбрасываются."""
    home = _home()
    dirs: list[Path] = []
    if sys.platform == "win32":
        appdata = os.environ.get("APPDATA")
        local = os.environ.get("LOCALAPPDATA")
        if appdata:
            dirs.append(Path(appdata) / "npm")
        if local:
            dirs += [Path(local) / "Programs" / "claude", Path(local) / "Microsoft" / "WinGet" / "Links"]
        dirs += [home / ".local" / "bin", home / ".claude" / "local", home / ".bun" / "bin", home / ".volta" / "bin"]
    else:
        dirs += [home / ".local" / "bin", home / ".claude" / "local", Path("/opt/homebrew/bin"),
                 Path("/usr/local/bin"), home / ".npm-global" / "bin", home / ".bun" / "bin", home / ".volta" / "bin"]
        nvm = home / ".nvm" / "versions" / "node"
        if nvm.is_dir():
            dirs += [p / "bin" for p in sorted(nvm.iterdir(), key=lambda p: _node_version_key(p / "bin"), reverse=True)]
        if sys.platform != "darwin":
            dirs.append(Path("/usr/bin"))
    out: list[Path] = []
    for d in dirs:
        try:
            if d.is_dir() and d not in out:
                out.append(d)
        except OSError:
            continue
    return out


def _is_exe(path: str | Path) -> bool:
    try:
        return os.path.isfile(path) and os.access(path, os.X_OK)
    except OSError:
        return False


def _from_login_shell(name: str) -> str | None:
    """`command -v name` во входной оболочке пользователя — подхватит PATH из .zprofile / .zshrc."""
    if sys.platform == "win32":
        return None
    shell = os.environ.get("SHELL") or "/bin/zsh"
    if not os.path.exists(shell):
        shell = "/bin/zsh" if os.path.exists("/bin/zsh") else "/bin/sh"
    try:
        proc = subprocess.run([shell, "-lic", f"command -v {shlex.quote(name)}"], capture_output=True, text=True,
                              encoding="utf-8", errors="replace", timeout=LOGIN_SHELL_TIMEOUT_S, stdin=subprocess.DEVNULL, start_new_session=True, check=False,
                              env={**os.environ, "TERM": "dumb"})
    except (OSError, subprocess.TimeoutExpired) as e:
        log.debug("входная оболочка не помогла найти %s: %s", name, type(e).__name__)
        return None
    for line in reversed(proc.stdout.splitlines()):
        line = line.strip()
        if line.startswith("/") and _is_exe(line):          # alias и функции начинаются не с «/» — их пропускаем
            return line
    return None


def find_binary(name: str, *, refresh: bool = False) -> str | None:
    """Путь к программе: PATH → типичные каталоги → входная оболочка. Результат кэшируется."""
    with _lock:
        if not refresh:
            cached = _found.get(name)
            if cached and _is_exe(cached):
                return cached
            miss = _missing_at.get(name)
            if miss is not None and time.monotonic() - miss < NEGATIVE_TTL_S:
                return None
        _found.pop(name, None)
    path = shutil.which(name)
    if not path:
        for d in candidate_dirs():
            path = shutil.which(name, path=str(d))
            if path:
                break
    if not path:
        path = _from_login_shell(name)
    with _lock:
        if path:
            _found[name] = path
            _missing_at.pop(name, None)
        else:
            _missing_at[name] = time.monotonic()
    return path


def child_env(exe: str) -> dict[str, str]:
    """Окружение для запуска CLI: PATH дополнен каталогом программы и типичными каталогами (node рядом с CLI)."""
    env = os.environ.copy()
    parts = [str(Path(exe).parent)]
    parts += [p for p in env.get("PATH", "").split(os.pathsep) if p]
    parts += [str(d) for d in candidate_dirs()]
    seen: list[str] = []
    for p in parts:
        if p not in seen:
            seen.append(p)
    env["PATH"] = os.pathsep.join(seen)
    env.setdefault("TERM", "dumb")
    return env


def reset_cache_for_tests() -> None:
    with _lock:
        _found.clear()
        _missing_at.clear()
        _login_cache.clear()
        _last_ok.clear()
        _verified.clear()


# ─────────────────────────── статус ───────────────────────────
@dataclass(frozen=True)
class CliStatus:
    state: CliState
    message: str                   # по-русски, для человека
    path: str | None = None

    @property
    def can_answer(self) -> bool:
        return self.state == "ok"


_login_cache: dict[str, tuple[float, CliStatus]] = {}
_last_ok: dict[str, CliStatus] = {}       # последняя удачная проверка: разовый сбой проверки не должен «гасить» модель
_verified: set[str] = set()               # id моделей, которые уже отвечали на настоящий запрос в этой сессии


def mark_verified(provider_id: str) -> None:
    _verified.add(provider_id)


def is_verified(provider_id: str) -> bool:
    return provider_id in _verified


def unverify(provider_id: str) -> None:
    _verified.discard(provider_id)


def not_installed_text(spec: CliSpec) -> str:
    return f"{spec.title} не найден на этом компьютере. {spec.install_hint}"


def not_logged_in_text(spec: CliSpec) -> str:
    return f"В {spec.title} нужно войти. Нажми «Войти» и пройди вход в открывшемся окне."


def _run_status(cmd: list[str], env: dict[str, str]) -> tuple[int | None, str]:
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=STATUS_TIMEOUT_S, stdin=subprocess.DEVNULL, check=False,
                              env=env, cwd=str(_home()) if _home().is_dir() else None)
    except subprocess.TimeoutExpired:
        return None, "timeout"
    except OSError as e:
        return None, e.strerror or type(e).__name__
    return proc.returncode, (proc.stdout or "") + "\n" + (proc.stderr or "")


def _claude_login(exe: str, spec: CliSpec) -> CliStatus:
    code, out = _run_status([exe, "auth", "status"], child_env(exe))
    if code is None:
        return CliStatus("error", f"{spec.title} не ответил на проверку входа. Попробуй ещё раз.", exe)
    data = None
    for chunk in (out.strip(), *[ln for ln in out.splitlines() if ln.strip().startswith("{")]):
        try:
            data = json.loads(chunk)
            break
        except ValueError:
            continue
    if isinstance(data, dict) and "loggedIn" in data:
        if data["loggedIn"]:
            return CliStatus("ok", f"{spec.title} установлен, вход выполнен.", exe)
        return CliStatus("not_logged_in", not_logged_in_text(spec), exe)
    low = out.lower()
    if "not logged in" in low or "/login" in low:
        return CliStatus("not_logged_in", not_logged_in_text(spec), exe)
    # старая версия без `auth status` — по входу судить не можем; настоящий запрос покажет, если что не так
    return CliStatus("ok", f"{spec.title} установлен.", exe)


def _codex_login(exe: str, spec: CliSpec) -> CliStatus:
    code, out = _run_status([exe, "login", "status"], child_env(exe))
    if code is None:
        return CliStatus("error", f"{spec.title} не ответил на проверку входа. Попробуй ещё раз.", exe)
    if code == 0:
        return CliStatus("ok", f"{spec.title} установлен, вход выполнен.", exe)
    low = out.lower()
    if any(w in low for w in ("not logged in", "not signed in", "no credentials", "login")) or code == 1:
        return CliStatus("not_logged_in", not_logged_in_text(spec), exe)
    return CliStatus("error", f"{spec.title} ответил с ошибкой (код {code}).", exe)


_LOGIN_CHECKS = {ProviderKind.CLAUDE_CLI: _claude_login, ProviderKind.CODEX_CLI: _codex_login}


def cli_status(kind: ProviderKind, *, refresh: bool = False) -> CliStatus:
    """Установлен ли CLI, вошёл ли пользователь. Быстрая проверка без запроса к модели (≈0,2 с)."""
    spec = CLI_SPECS[kind]
    exe = find_binary(spec.binary, refresh=refresh)
    if not exe:
        return CliStatus("not_installed", not_installed_text(spec))
    key = f"{kind.value}:{exe}"
    with _lock:
        hit = _login_cache.get(key)
    if hit and not refresh and time.monotonic() - hit[0] < LOGIN_TTL_S:
        return hit[1]
    try:
        st = _LOGIN_CHECKS[kind](exe, spec)
    except Exception as e:  # noqa: BLE001 — статус вторичен: не роняем экран настроек
        log.debug("проверка входа %s: %s", spec.binary, type(e).__name__)
        st = CliStatus("error", f"Не удалось проверить {spec.title}.", exe)
    with _lock:
        if st.state == "ok":
            _last_ok[key] = st
        elif st.state == "error" and key in _last_ok:        # сбой самой проверки (таймаут), а не отказ входа
            st = _last_ok[key]
        _login_cache[key] = (time.monotonic(), st)
    return st


def forget_login(kind: ProviderKind) -> None:
    """Забыть результат проверки входа (после реального отказа авторизации или после кнопки «Войти»)."""
    with _lock:
        for k in [k for k in _login_cache if k.startswith(kind.value + ":")]:
            _login_cache.pop(k, None)


# ─────────────────────────── вход через системный терминал ───────────────────────────
def login_command(kind: ProviderKind) -> list[str]:
    """Команда входа целиком из фиксированного списка: путь к программе, найденный нами, + login_args."""
    spec = CLI_SPECS.get(kind)
    if spec is None:
        raise EngineError("Для этой модели вход не нужен: она работает по ключу.")
    exe = find_binary(spec.binary, refresh=True)
    if not exe:
        raise EngineError(not_installed_text(spec))
    return [exe, *spec.login_args]


def _shell_line(cmd: list[str]) -> str:
    """Одна строка для оболочки: PATH с каталогом программы + сама команда. Всё экранировано."""
    path_dir = shlex.quote(str(Path(cmd[0]).parent))
    return f'export PATH={path_dir}:"$PATH"; ' + " ".join(shlex.quote(c) for c in cmd)


def _command_file(cmd: list[str]) -> str:
    """Временный файл .command: Terminal.app открывает такие файлы сам, без запроса «управлять Terminal» (Apple Events)."""
    import tempfile
    d = tempfile.mkdtemp(prefix="dubl-login-")
    path = os.path.join(d, "Вход.command")
    body = ("#!/bin/zsh\n" + _shell_line(cmd) + "\n"
            'echo; echo "Готово. Это окно можно закрыть и вернуться в Дубль."\n'
            'rm -rf "$(dirname "$0")" 2>/dev/null\n')
    with open(path, "w", encoding="utf-8") as f:
        f.write(body)
    os.chmod(path, 0o700)
    return path


def terminal_launcher(cmd: list[str]) -> list[list[str]]:
    """Что запускать, чтобы открыть системный терминал с командой. Список вариантов: пробуем по очереди."""
    line = _shell_line(cmd)
    if sys.platform == "darwin":
        return [["open", "-a", "Terminal", _command_file(cmd)]]
    if sys.platform == "win32":
        return [["cmd", "/c", "start", "", "cmd", "/k", subprocess.list2cmdline(cmd)]]
    keep = f"{line}; echo; echo 'Готово. Это окно можно закрыть.'; exec \"${{SHELL:-sh}}\""
    return [["x-terminal-emulator", "-e", "sh", "-c", keep], ["gnome-terminal", "--", "sh", "-c", keep],
            ["konsole", "-e", "sh", "-c", keep], ["xterm", "-e", "sh", "-c", keep]]


def open_login_terminal(kind: ProviderKind) -> str:
    """Открыть терминал с командой входа. Возвращает текст для человека; ошибки — EngineError."""
    spec = CLI_SPECS.get(kind)
    cmd = login_command(kind)
    last = ""
    for launcher in terminal_launcher(cmd):
        if shutil.which(launcher[0]) is None:
            continue
        try:
            proc = subprocess.run(launcher, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=15, stdin=subprocess.DEVNULL, check=False,
                                  creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0) if sys.platform == "win32" else 0)
        except (OSError, subprocess.TimeoutExpired) as e:
            last = type(e).__name__
            continue
        if proc.returncode == 0 or sys.platform == "win32":
            forget_login(kind)
            return f"Открыл терминал: войди в {spec.title if spec else 'аккаунт'} и вернись сюда."
        last = (proc.stderr or "").strip()[:200]
    detail = f" ({last})" if last else ""
    raise EngineError("Не получилось открыть терминал" + detail + ". Открой его сам и запусти: "
                      + " ".join(shlex.quote(c) for c in [Path(cmd[0]).name, *cmd[1:]]))
