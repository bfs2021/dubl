"""Страховка для Windows: текст читаем/пишем и разбираем вывод процессов только в UTF-8 явно.

На Windows без PYTHONUTF8 open()/read_text()/write_text() без encoding= работают в cp1251/cp1252,
а subprocess с text=True без encoding= декодирует вывод ffmpeg/yt-dlp так же — кириллица в путях
(папка «Дубль») и в сообщениях превращается в мусор или роняет движок UnicodeEncodeError/DecodeError.
"""
from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FILES = sorted((ROOT / "engine" / "src").rglob("*.py")) + sorted((ROOT / "packaging").glob("*.py"))

_TEXT_METHODS = {"read_text": 0, "write_text": 1}          # индекс позиционного encoding
_SUBPROCESS = {"run", "Popen", "check_output", "check_call", "call"}


def _kw(call: ast.Call, name: str) -> ast.expr | None:
    return next((k.value for k in call.keywords if k.arg == name), None)


def _is_true(node: ast.expr | None) -> bool:
    return isinstance(node, ast.Constant) and node.value is True


def _binary_mode(call: ast.Call, pos: int) -> bool:
    mode = _kw(call, "mode") or (call.args[pos] if len(call.args) > pos else None)
    return isinstance(mode, ast.Constant) and isinstance(mode.value, str) and "b" in mode.value


def problems(path: Path) -> list[str]:
    out: list[str] = []
    tree = ast.parse(path.read_text(encoding="utf-8"), str(path))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        where = f"{path.relative_to(ROOT)}:{node.lineno}"
        has_enc = _kw(node, "encoding") is not None
        if isinstance(f, ast.Name) and f.id == "open" and not has_enc and not _binary_mode(node, 1):
            out.append(f"{where}: open() без encoding=")
        elif isinstance(f, ast.Attribute) and f.attr == "open" and isinstance(f.value, ast.Name) \
                and f.value.id == "io" and not has_enc and not _binary_mode(node, 1):
            out.append(f"{where}: io.open() без encoding=")
        elif isinstance(f, ast.Attribute) and f.attr in _TEXT_METHODS and not has_enc \
                and len(node.args) <= _TEXT_METHODS[f.attr]:
            out.append(f"{where}: .{f.attr}() без encoding=")
        elif isinstance(f, ast.Attribute) and f.attr in _SUBPROCESS and isinstance(f.value, ast.Name) \
                and f.value.id == "subprocess" and (_is_true(_kw(node, "text")) or _is_true(_kw(node, "universal_newlines"))) \
                and not has_enc:
            out.append(f"{where}: subprocess.{f.attr}(text=True) без encoding=")
    return out


def test_explicit_utf8_everywhere():
    assert FILES, "не нашёл исходники"
    found = [p for f in FILES for p in problems(f)]
    assert not found, "Укажи encoding=\"utf-8\" (для вывода процессов ещё errors=\"replace\"):\n" + "\n".join(found)


def test_checker_catches_mistakes(tmp_path):
    bad = tmp_path / "bad.py"
    bad.write_text(
        "import subprocess\nfrom pathlib import Path\n"
        "open('a')\nPath('a').read_text()\nPath('a').write_text('x')\n"
        "subprocess.run(['x'], text=True)\n"
        "open('a', 'rb')\nPath('a').read_text('utf-8')\nsubprocess.run(['x'], text=True, encoding='utf-8')\n",
        encoding="utf-8",
    )
    global ROOT
    old, ROOT = ROOT, tmp_path
    try:
        found = problems(bad)
    finally:
        ROOT = old
    assert len(found) == 4, found
