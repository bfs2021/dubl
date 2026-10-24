# Tauri 2 — правила для «Дубль»

Источник: официальная документация v2.tauri.app (sidecar, capabilities, CSP). Сторонний навык tauri отклонён сканером безопасности.

## Устройство
- `app/src-tauri/` — Rust-оболочка. Логики приложения в Rust почти нет: запуск движка, диалоги файлов, права.
- Движок — **sidecar**: `bundle.externalBin: ["binaries/dubl-engine"]` в `tauri.conf.json`.
  Файл должен называться `binaries/dubl-engine-<target-triple>` (`rustc --print host-tuple`, напр. `aarch64-apple-darwin`,
  `x86_64-pc-windows-msvc` + `.exe`, `x86_64-unknown-linux-gnu`).
- Запуск из Rust: `tauri_plugin_shell::ShellExt` → `app.shell().sidecar("dubl-engine")?.args(["--port","0"]).spawn()`,
  читать `CommandEvent::Stdout`, найти строку `DUBL_ENGINE_READY port=<n>`, отдать порт фронту командой `engine_url`
  и/или событием `engine-ready`. На закрытии окна — `child.kill()`.
- В dev-режиме (`pnpm tauri dev`) допускается переменная `DUBL_ENGINE_URL` — тогда sidecar не запускается, используется уже работающий движок.

## Безопасность (обязательно)
- Capabilities — минимальные, отдельный файл `capabilities/main.json` для окна `main`:
  `core:default`, `dialog:allow-open`, `dialog:allow-save`, `shell:allow-spawn` / `shell:allow-execute` **только** с `sidecar: true`
  и `name: "binaries/dubl-engine"` (никаких произвольных команд), `shell:allow-kill`; для кнопок после рендера — плагин `opener`: `opener:allow-reveal-item-in-dir` и `opener:allow-open-path` только для `$HOME/**`, `$VIDEO/**`, `/Volumes/**`. Без `fs:*` на весь диск — файлы читает движок.
- CSP в `tauri.conf.json` → `app.security.csp`: `default-src 'self'; connect-src 'self' http://127.0.0.1:* ws://127.0.0.1:* ipc: http://ipc.localhost;
  media-src 'self' http://127.0.0.1:* blob:; img-src 'self' data: blob: http://127.0.0.1:*; style-src 'self' 'unsafe-inline'; font-src 'self' data:`.
  Удалённый движок (свой сервер) — адрес пользователь вводит в настройках; для него connect-src расширяется в рантайме не нужно:
  ходим через команду Rust-прокси или добавляем `https:` в connect-src (решение — в отчёте).
- Движок слушает только `127.0.0.1`. Для удалённого режима — токен (Bearer).
- Микрофон: macOS требует `NSMicrophoneUsageDescription` (bundle.macOS → `Info.plist`) и entitlement
  `com.apple.security.device.audio-input`. Текст: «Нужен, чтобы записывать твою озвучку».
- Не включать `withGlobalTauri`, не открывать devtools в релизе.

## Установка нейросетей (Rust-команды)
- `install_ml_pack`, `cancel_ml_pack`, `ml_pack_status`, `ml_pack_state` — свои команды окна (`src/ml_pack.rs`). Разрешены поштучно (`allow-…` в `capabilities/main.json`, список в `build.rs` через `AppManifest::commands`); `shell:allow-spawn` для них не нужен — sidecar запускается из Rust с фиксированными аргументами, а JS не может подставить свои.
- События `ml-pack` (stage / log / done / failed / cancelled) и перезапуск движка с новым `engine-ready` — см. `docs/BUILD.md`.
- `DUBL_DATA_DIR` переопределяет папку данных (отладка, проверки в /tmp).

## Проверки
- `cd app/src-tauri && cargo check` — компиляция.
- `pnpm tauri dev` — окно с живым движком.
- `pnpm tauri build --debug` — сборка бандла (нужен sidecar-бинарник; для проверки допускается заглушка).

## Чего в WebView нет (и что делаем вместо)
- `window.confirm / alert / prompt` в WebView Tauri 2 молча возвращают `false`/ничего не показывают (в браузере работали — поэтому корзина проекта «не срабатывала» только в собранном приложении). В `app/src` их быть не должно (проверяет тест «в src нет window.confirm / alert / prompt»). Вместо них — `useStore.getState().askConfirm({...})` и `<ConfirmHost/>` (`components/ConfirmDialog.tsx`): фокус, Esc, Enter, красная кнопка для опасных действий.
- Ссылки наружу — через `openExternal` (`lib/external.ts`, плагин shell/opener), `window.open` там только запасной путь для браузера. `<a download>` не используется: готовые файлы лежат на диске, открываем через `openFile` / `revealFile`.
- Место сохранения ролика — системный диалог `plugin-dialog` `save` (`pickSavePath`), права `dialog:allow-save` уже в `capabilities/main.json`.
- `<details>` управляем сами (`components/Disclosure.tsx`): встроенное раскрытие терялось при перерисовке.
