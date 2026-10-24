#!/bin/bash
# Поставить свежую сборку в /Applications и убрать её из папки сборки.
# Зачем убирать: Spotlight и Launchpad видят каждый .app на диске, и в поиске появляются два «Дубля».
# Запускать после `pnpm tauri build`. Установщик .dmg в bundle/dmg остаётся.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$ROOT/app/src-tauri/target/release/bundle/macos/Дубль.app"
DST="/Applications/Дубль.app"

if [ ! -d "$SRC" ]; then
  echo "Свежей сборки нет: $SRC" >&2
  echo "Сначала собери приложение (docs/BUILD.md)." >&2
  exit 1
fi

# открытую программу закрываем, иначе старая копия останется запущенной
if pgrep -f "$DST/Contents/MacOS/dubl" >/dev/null; then
  osascript -e 'quit app "Дубль"' || true
  for _ in $(seq 1 20); do
    pgrep -f "$DST/Contents/MacOS/dubl" >/dev/null || break
    sleep 0.5
  done
fi

rm -rf "$DST"
mv "$SRC" "$DST"          # тот же диск: перенос мгновенный, вторая копия не остаётся
echo "Установлено: $DST"
