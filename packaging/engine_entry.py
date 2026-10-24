"""Точка входа sidecar-а `dubl-engine` (то, что собирает PyInstaller, см. dubl-engine.spec).

Режимы:
  dubl-engine [--host --port --token --data-dir]     запустить движок
  dubl-engine install-ml-pack [--data-dir --device]  скачать torch и нейросетевые библиотеки
  dubl-engine ml-pack-status [--data-dir]            JSON: что установлено
  dubl-engine self-check                             JSON: версия, какой ffmpeg найден

Движок в самой сборке — «лёгкий» (без torch). Если в <data_dir>/runtime/ml лежит установленный
нейросетевой пакет той же версии, лаунчер запускает движок из того окружения:
  · Unix — через exec (процесс заменяется, PID тот же: `child.kill()` из Tauri убивает именно движок);
  · Windows — дочерним процессом; от «осиротения» защищает Job Object в Rust-лаунчере.
Строка DUBL_ENGINE_READY доходит до Tauri как есть (stdout общий).

Страховка от осиротевшего процесса: движок следит за родителем (Tauri) и завершается, если тот пропал
(в т.ч. SIGKILL). Отключить: DUBL_PARENT_WATCH=0 (systemd, nohup, запуск «в фоне» из терминала).
Принудительно лёгкий режим: DUBL_ENGINE_LIGHT=1.
"""
from __future__ import annotations

import json
import multiprocessing
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

# Выполняется внутри python из нейросетевого venv: следит за родителем и запускает движок.
_BOOT = r"""
import os, sys, threading, time
def _alive(pid):
    if os.name == "nt":
        import ctypes
        k = ctypes.windll.kernel32
        h = k.OpenProcess(0x00100000, False, pid)
        if not h:
            return False
        try:
            return k.WaitForSingleObject(h, 0) == 0x102
        finally:
            k.CloseHandle(h)
    return os.getppid() == pid
_w = int(os.environ.pop("DUBL_WATCH_PID", "0") or 0)
if _w > 1 and os.environ.get("DUBL_PARENT_WATCH", "1") != "0":
    def _loop():
        while _alive(_w):
            time.sleep(1.0)
        os._exit(0)
    threading.Thread(target=_loop, daemon=True, name="parent-watch").start()
from dubl_engine.main import run
run(sys.argv[1:])
"""


def _arg_value(argv: list[str], name: str) -> str | None:
    for i, a in enumerate(argv):
        if a == name and i + 1 < len(argv):
            return argv[i + 1]
        if a.startswith(name + "="):
            return a.split("=", 1)[1]
    return None


def _data_dir(argv: list[str]) -> Path:
    d = _arg_value(argv, "--data-dir") or os.environ.get("DUBL_DATA_DIR")
    if d:
        return Path(d)
    from platformdirs import user_data_dir
    return Path(user_data_dir("Dubl", "Dubl"))


def _pid_alive(pid: int) -> bool:
    if os.name == "nt":
        import ctypes
        k32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        h = k32.OpenProcess(0x00100000, False, pid)  # SYNCHRONIZE
        if not h:
            return False
        try:
            return k32.WaitForSingleObject(h, 0) == 0x102  # WAIT_TIMEOUT => ещё жив
        finally:
            k32.CloseHandle(h)
    return os.getppid() == pid


def _parent_watch_enabled() -> bool:
    return os.environ.get("DUBL_PARENT_WATCH", "1") != "0"


def _start_watchdog(on_gone) -> None:
    parent = os.getppid()
    if parent <= 1 or not _parent_watch_enabled():
        return

    def loop() -> None:
        while _pid_alive(parent):
            time.sleep(1.0)
        on_gone()

    threading.Thread(target=loop, daemon=True, name="parent-watch").start()


def _engine_version() -> str:
    try:
        from importlib.metadata import version
        return version("dubl-engine")
    except Exception:  # noqa: BLE001
        return "?"


def _ml_python(argv: list[str]) -> str | None:
    if os.environ.get("DUBL_ENGINE_LIGHT") == "1":
        return None
    import ml_pack
    data_dir = _data_dir(argv)
    marker = ml_pack.read_marker(data_dir)
    py = ml_pack.venv_python(data_dir)
    if not marker or not py.exists():
        return None
    if marker.get("engine_version") != _engine_version():
        print(f"Нейросетевой пакет от другой версии движка ({marker.get('engine_version')} != "
              f"{_engine_version()}), запускаю лёгкий движок. Переустановите пакет.", file=sys.stderr, flush=True)
        return None
    ml_pack.sync_engine(data_dir)          # код движка в venv — как в этой сборке (секунды, без сети)
    return str(py)


def _run_ml(py: str, argv: list[str]) -> int:
    import ml_pack
    env = ml_pack.clean_env()
    env["DUBL_ENGINE_FULL"] = "1"
    env["PYTHONUTF8"] = "1"
    ml_pack.gpu_env(_data_dir(argv), env)
    env["DUBL_WATCH_PID"] = str(os.getppid())
    cmd = [py, "-c", _BOOT, *argv]
    if os.name != "nt":
        sys.stdout.flush()
        sys.stderr.flush()
        os.execve(py, cmd, env)  # не возвращается
    child = subprocess.Popen(cmd, env=env, creationflags=0x00000200)  # CREATE_NEW_PROCESS_GROUP
    return child.wait()


def _utf8_stdio() -> None:
    """На Windows stdout в пайпе — cp1251/cp1252; русские сообщения иначе роняют движок (UnicodeEncodeError)."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
        except Exception:  # noqa: BLE001
            pass


def main() -> None:
    multiprocessing.freeze_support()
    _utf8_stdio()
    argv = sys.argv[1:]
    if argv and argv[0] == "install-ml-pack":
        import ml_pack
        sys.exit(ml_pack.main(argv[1:]))
    if argv and argv[0] == "ml-pack-status":
        import ml_pack
        data_dir = _data_dir(argv[1:])
        m = ml_pack.read_marker(data_dir)
        print(json.dumps({"installed": bool(m and ml_pack.venv_python(data_dir).exists()),
                          "engine_version": _engine_version(), "pack": m,
                          "recommended_device": ml_pack.detect_device()}, ensure_ascii=False), flush=True)
        return

    if argv and argv[0] == "self-check":
        # диагностика сборки: какой ffmpeg видит движок и что подключено
        from dubl_engine.util import find_binary
        res: dict = {"engine_version": _engine_version(), "frozen": bool(getattr(sys, "frozen", False)),
                     "ffmpeg_dir_env": os.environ.get("DUBL_FFMPEG_DIR")}
        for name in ("ffmpeg", "ffprobe"):
            try:
                res[name] = find_binary(name)
            except Exception as e:  # noqa: BLE001
                res[name] = None
                res[name + "_error"] = str(e)
        print(json.dumps(res, ensure_ascii=False), flush=True)
        return

    py = _ml_python(argv)
    if py:
        sys.exit(_run_ml(py, argv))

    _start_watchdog(lambda: os._exit(0))
    from dubl_engine.main import run
    run(argv)


if __name__ == "__main__":
    main()
