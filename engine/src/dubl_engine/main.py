"""Запуск движка: dubl-engine --host 127.0.0.1 --port 8765 [--token T] [--data-dir D].

Порт 0 — свободный; в stdout печатается `DUBL_ENGINE_READY port=<n>` (по ней Tauri понимает, что движок поднялся).
"""
from __future__ import annotations

import argparse
import logging
import os
import re
import socket
import sys

_TOKEN_Q = re.compile(r"([?&]token=)[^&\s\"']+", re.IGNORECASE)


class ScrubTokenFilter(logging.Filter):
    """В адресах `/files/...?token=…` и `/ws/jobs?token=…` токен — секрет; в журнал доступа он попадать не должен."""

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.args, tuple):
            record.args = tuple(_TOKEN_Q.sub(r"\1***", a) if isinstance(a, str) else a for a in record.args)
        if isinstance(record.msg, str):
            record.msg = _TOKEN_Q.sub(r"\1***", record.msg)
        return True


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="dubl-engine", description="Движок «Дубль»")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765, help="0 — свободный порт")
    ap.add_argument("--token", default=os.environ.get("DUBL_TOKEN"), help="токен доступа (для удалённого режима)")
    ap.add_argument("--data-dir", default=None, help="папка данных (проекты, модели, настройки)")
    return ap.parse_args(argv)


def run(argv: list[str] | None = None) -> None:
    import uvicorn

    args = parse_args(argv)
    if args.data_dir:
        os.environ["DUBL_DATA_DIR"] = args.data_dir
    if args.host not in ("127.0.0.1", "localhost", "::1") and not args.token:
        print("Внимание: движок открыт в сеть без токена. Задайте --token.", file=sys.stderr, flush=True)

    from .api import create_app
    app = create_app(token=args.token, recover=True)

    sock = socket.socket(socket.AF_INET6 if ":" in args.host else socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((args.host, args.port))
    port = sock.getsockname()[1]

    config = uvicorn.Config(app, log_level="info")
    for name in ("uvicorn.access", "uvicorn.error", "uvicorn"):
        logging.getLogger(name).addFilter(ScrubTokenFilter())
    server = uvicorn.Server(config)
    orig_startup = server.startup

    async def startup(sockets=None):
        await orig_startup(sockets=sockets)
        if not server.should_exit:
            print(f"DUBL_ENGINE_READY port={port}", flush=True)

    server.startup = startup  # type: ignore[method-assign]
    server.run(sockets=[sock])


if __name__ == "__main__":
    run()
