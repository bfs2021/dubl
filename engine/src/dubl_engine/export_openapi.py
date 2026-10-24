"""Экспорт OpenAPI-схемы движка: python -m dubl_engine.export_openapi <out.json>

Фронтенд генерирует из неё типы (app: pnpm gen:api). Схема — единственный источник правды для UI:
поменял models.py / api — перегенерируй типы, CI проверяет, что они не устарели.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path


def build_schema() -> dict:
    import os

    os.environ.setdefault("DUBL_DATA_DIR", tempfile.mkdtemp(prefix="dubl-openapi-"))
    from dubl_engine.api.app import create_app

    return create_app().openapi()


def main() -> None:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("openapi.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(build_schema(), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
