#!/usr/bin/env python3
"""Дымовая проверка собранного sidecar: запуск, READY, /health, ffmpeg из ресурсов, смерть движка после SIGKILL родителя.

  python packaging/smoke_sidecar.py [--stage DIR] [путь/к/dubl-engine-<triple>[.exe]]
  DIR — папка с binaries/ и resources/ (по умолчанию packaging/build/stage; для релиза — app/src-tauri)
Скрипт сам собирает раскладку «как в установленной программе» во временной папке. Выход 0 — всё хорошо.
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
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
    assert p.stdout
    for line in p.stdout:
        m = re.search(r"DUBL_ENGINE_READY port=(\d+)", line)
        if m:
            return p, int(m.group(1)), time.time() - t0
    raise SystemExit("движок завершился без DUBL_ENGINE_READY")


def alive(pid: int) -> bool:
    if os.name == "nt":
        out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}"], capture_output=True, text=True).stdout
        return str(pid) in out
    return subprocess.run(["kill", "-0", str(pid)], capture_output=True).returncode == 0


def main() -> int:
    global STAGE
    args = sys.argv[1:]
    if "--stage" in args:
        i = args.index("--stage")
        STAGE = Path(args[i + 1]).resolve()
        del args[i:i + 2]
    exe = Path(args[0]).resolve() if args else next((STAGE / "binaries").glob("dubl-engine-*"))
    tmp = Path(tempfile.mkdtemp(prefix="dubl-smoke-"))
    try:
        app = tmp / "app"
        (app / "resources").mkdir(parents=True)
        launcher = app / exe.name
        shutil.copy2(exe, launcher)
        shutil.copytree(STAGE / "resources" / "engine", app / "resources" / "engine", symlinks=True)
        has_ff = (STAGE / "resources" / "ffmpeg").is_dir()
        if has_ff:
            shutil.copytree(STAGE / "resources" / "ffmpeg", app / "resources" / "ffmpeg")
        env = dict(os.environ, DUBL_RESOURCE_DIR=str(app / "resources"))
        env.pop("DUBL_FFMPEG_DIR", None)
        data = tmp / "data"

        times = []
        for i in range(3):
            p, port, dt = start([str(launcher), "--port", "0", "--data-dir", str(data)], env)
            times.append(dt)
            if i == 0:
                health = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=10).read())
                assert health.get("ok") is True, health
                print("health:", health)
            p.terminate()
            p.wait(15)
        print("READY через, с:", ", ".join(f"{x:.2f}" for x in times), "(первый — холодный)")

        chk = json.loads(subprocess.run([str(launcher), "self-check"], capture_output=True, text=True, env=env, check=True).stdout)
        print("self-check:", chk)
        assert chk["frozen"] is True
        if has_ff:
            assert str(app / "resources" / "ffmpeg") in (chk["ffmpeg"] or ""), "движок не увидел ffmpeg из ресурсов"

        # родитель убит SIGKILL-ом → движок должен завершиться сам (Unix: лаунчер делает exec, PID один)
        if os.name != "nt":
            helper = ("import subprocess,sys;"
                      "p=subprocess.Popen([sys.argv[1],'--port','0','--data-dir',sys.argv[2]],stdout=subprocess.PIPE,text=True);"
                      "next(l for l in p.stdout if 'READY' in l);"
                      "print(p.pid,flush=True);import time;time.sleep(120)")
            parent = subprocess.Popen([sys.executable, "-c", helper, str(launcher), str(data)], stdout=subprocess.PIPE, text=True, env=env)
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
    sys.exit(main())
