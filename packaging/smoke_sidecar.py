#!/usr/bin/env python3
"""Дымовая проверка собранного sidecar: запуск, READY, /health, ffmpeg из ресурсов, смерть движка после SIGKILL родителя.

  python packaging/smoke_sidecar.py [--stage DIR] [путь/к/dubl-engine-<triple>[.exe]]
  DIR — папка с binaries/ и resources/ (по умолчанию packaging/build/stage; для релиза — app/src-tauri)
Скрипт сам собирает раскладку «как в установленной программе» во временной папке (с кириллицей в путях). Выход 0 — всё хорошо.

  python packaging/smoke_sidecar.py --installed ПУТЬ/К/dubl-engine[.exe]
Проверка уже установленной программы (распакованный AppImage/deb, установленный NSIS): движок запускается как есть,
сам находит свои ресурсы и ffmpeg; папка данных — с кириллицей и пробелом.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
STAGE = HERE / "build" / "stage"


def start(cmd: list[str], env: dict[str, str]):
    t0 = time.time()
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", env=env)
    assert p.stdout
    for line in p.stdout:
        print("  |", line.rstrip(), flush=True)          # лог движка виден в CI: если упадёт — понятно где
        m = re.search(r"DUBL_ENGINE_READY port=(\d+)", line)
        if m:
            return p, int(m.group(1)), time.time() - t0
    raise SystemExit("движок завершился без DUBL_ENGINE_READY")


def alive(pid: int) -> bool:
    if os.name == "nt":
        out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}"], capture_output=True, text=True, encoding="utf-8", errors="replace").stdout
        return str(pid) in out
    return subprocess.run(["kill", "-0", str(pid)], capture_output=True).returncode == 0


def check_running(launcher: Path, env: dict[str, str], data: Path, *, runs: int = 3) -> None:
    """READY, /health, список проектов, что-то создано в папке данных; затем мягкая остановка. Повторяем для времени старта."""
    times = []
    for i in range(runs):
        p, port, dt = start([str(launcher), "--port", "0", "--data-dir", str(data)], env)
        times.append(dt)
        try:
            if i == 0:
                health = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=10).read())
                assert health.get("ok") is True, health
                print("health:", health)
                projects = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{port}/projects", timeout=10).read())
                assert isinstance(projects, list), projects
                assert data.is_dir() and any(data.iterdir()), f"движок ничего не создал в папке данных {data}"
                print("папка данных с кириллицей:", data, "→", sorted(x.name for x in data.iterdir()))
        finally:
            p.terminate()
            p.wait(15)
    print("READY через, с:", ", ".join(f"{x:.2f}" for x in times), "(первый — холодный)")


def self_check(launcher: Path, env: dict[str, str]) -> dict:
    r = subprocess.run([str(launcher), "self-check"], capture_output=True, text=True, encoding="utf-8", errors="replace",
                       env=env)
    if r.returncode != 0:
        raise SystemExit(f"self-check: код {r.returncode}\n{r.stdout}\n{r.stderr}")
    chk = json.loads(r.stdout)
    print("self-check:", chk)
    assert chk["frozen"] is True
    return chk


def installed(exe: Path) -> int:
    """Программа уже установлена/распакована: ничего не подкладываем, движок должен найти всё сам."""
    exe = exe.resolve()
    env = dict(os.environ)
    for k in ("DUBL_RESOURCE_DIR", "DUBL_ENGINE_DIR", "DUBL_FFMPEG_DIR", "DUBL_DATA_DIR"):
        env.pop(k, None)
    chk = self_check(exe, env)
    assert chk.get("ffmpeg") and chk.get("ffprobe"), "движок не нашёл ffmpeg в своих ресурсах"
    assert chk.get("ffmpeg_dir_env"), "лаунчер не передал движку DUBL_FFMPEG_DIR (ресурсы не найдены)"
    tmp = Path(tempfile.mkdtemp(prefix="dubl-smoke-"))
    try:
        check_running(exe, env, tmp / "Дубль данные", runs=2)
        print("OK")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> int:
    global STAGE
    args = sys.argv[1:]
    if "--installed" in args:
        return installed(Path(args[args.index("--installed") + 1]))
    if "--stage" in args:
        i = args.index("--stage")
        STAGE = Path(args[i + 1]).resolve()
        del args[i:i + 2]
    exe = Path(args[0]).resolve() if args else next((STAGE / "binaries").glob("dubl-engine-*"))
    tmp = Path(tempfile.mkdtemp(prefix="dubl-smoke-"))
    try:
        # Кириллица и пробел в путях — как у пользователя: программа ставится в «…\\Дубль», данные — в папку с русским именем
        app = tmp / "Дубль"
        (app / "resources").mkdir(parents=True)
        launcher = app / exe.name
        shutil.copy2(exe, launcher)
        shutil.copytree(STAGE / "resources" / "engine", app / "resources" / "engine", symlinks=True)
        has_ff = (STAGE / "resources" / "ffmpeg").is_dir()
        if has_ff:
            shutil.copytree(STAGE / "resources" / "ffmpeg", app / "resources" / "ffmpeg")
        env = dict(os.environ, DUBL_RESOURCE_DIR=str(app / "resources"))
        env.pop("DUBL_FFMPEG_DIR", None)
        data = tmp / "Данные Дубль-тест"

        check_running(launcher, env, data)
        chk = self_check(launcher, env)
        if has_ff:
            assert str(app / "resources" / "ffmpeg") in (chk["ffmpeg"] or ""), "движок не увидел ffmpeg из ресурсов"

        # родитель убит SIGKILL-ом → движок должен завершиться сам (Unix: лаунчер делает exec, PID один)
        if os.name != "nt":
            helper = ("import subprocess,sys;"
                      "p=subprocess.Popen([sys.argv[1],'--port','0','--data-dir',sys.argv[2]],stdout=subprocess.PIPE,text=True,encoding='utf-8');"
                      "next(l for l in p.stdout if 'READY' in l);"
                      "print(p.pid,flush=True);import time;time.sleep(120)")
            parent = subprocess.Popen([sys.executable, "-c", helper, str(launcher), str(data)], stdout=subprocess.PIPE, text=True,
                                      encoding="utf-8", errors="replace", env=env)
            assert parent.stdout
            pid = int(parent.stdout.readline())
            assert alive(pid), "движок не запустился"
            parent.kill()
            parent.wait()
            for _ in range(10):
                time.sleep(1)
                if not alive(pid):
                    break
            else:
                os.kill(pid, 9)
                raise SystemExit("движок пережил убитого родителя")
            print("осиротевший движок завершился сам: ок")
        print("OK")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    for _s in (sys.stdout, sys.stderr):   # Windows: консоль/пайп в cp1252 — русский print иначе падает (UnicodeEncodeError)
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
        except Exception:  # noqa: BLE001
            pass
    sys.exit(main())
