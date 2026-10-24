//! Оболочка «Дубль»: запускает движок как sidecar, отдаёт окну его адрес и убивает при выходе.
//! Логики приложения здесь нет — вся работа в движке (Python) и интерфейсе (React).

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

use tauri::{AppHandle, Emitter, Manager, RunEvent, State};
use tauri_plugin_shell::{
    process::{CommandChild, CommandEvent},
    ShellExt,
};

mod ml_pack;

const READY_PREFIX: &str = "DUBL_ENGINE_READY";

#[derive(Default)]
struct EngineState {
    /// Адрес движка вида http://127.0.0.1:PORT, когда он готов
    url: Mutex<Option<String>>,
    child: Mutex<Option<CommandChild>>,
    /// Номер запуска: события завершившегося (убитого при перезапуске) движка не должны сбрасывать адрес нового
    generation: AtomicU64,
}

/// Достаёт порт из строки `DUBL_ENGINE_READY port=8765`.
fn parse_ready(line: &str) -> Option<u16> {
    let rest = line.trim().strip_prefix(READY_PREFIX)?;
    rest.split_whitespace()
        .find_map(|part| part.strip_prefix("port="))
        .and_then(|p| p.parse::<u16>().ok())
}

fn publish(app: &AppHandle, url: String) {
    if let Some(state) = app.try_state::<EngineState>() {
        *state.url.lock().unwrap() = Some(url.clone());
    }
    eprintln!("[engine] готов: {url}");
    let _ = app.emit("engine-ready", url);
}

/// Статус нейросетевого пакета: JSON из `dubl-engine ml-pack-status` (installed, engine_version, pack, recommended_device).
#[tauri::command]
async fn ml_pack_status(app: AppHandle) -> Result<serde_json::Value, String> {
    ml_pack::status(&app).await
}

/// Запустить установку нейросетей. Ход — события `ml-pack`; в конце движок перезапускается и шлёт `engine-ready`.
#[tauri::command]
fn install_ml_pack(app: AppHandle) -> Result<(), String> {
    ml_pack::start(&app)
}

/// Отменить идущую установку.
#[tauri::command]
fn cancel_ml_pack(app: AppHandle) -> bool {
    ml_pack::cancel(&app)
}

/// Идёт ли установка и что было последним событием (окно могли перезагрузить посреди установки).
#[tauri::command]
fn ml_pack_state(app: AppHandle) -> ml_pack::MlSnapshot {
    ml_pack::snapshot(&app)
}

/// Адрес движка или None, если он ещё не запущен (тогда фронт ждёт событие `engine-ready`).
#[tauri::command]
fn engine_url(state: State<'_, EngineState>) -> Option<String> {
    state.url.lock().unwrap().clone()
}

/// Папка данных: `DUBL_DATA_DIR` (отладка, проверки в /tmp) или папка данных приложения.
/// И движок, и установка нейросетей берут её отсюда — это должна быть одна и та же папка.
pub(crate) fn data_dir(app: &AppHandle) -> Result<PathBuf, String> {
    if let Some(d) = std::env::var_os("DUBL_DATA_DIR").filter(|v| !v.is_empty()) {
        return Ok(PathBuf::from(d));
    }
    app.path().app_data_dir().map_err(|e| format!("нет папки данных: {e}"))
}

/// Нейросетевой пакет поставлен: останавливаем старый (лёгкий) движок и запускаем заново — лаунчер подхватит `runtime/ml`.
/// `stopped` вызывается между остановкой и запуском (сообщить окну «готово, перезапускаю»).
pub(crate) fn restart_engine(app: &AppHandle, stopped: impl FnOnce()) {
    if let Some(state) = app.try_state::<EngineState>() {
        state.generation.fetch_add(1, Ordering::SeqCst); // события старого движка больше ничего не меняют
    }
    stop_engine(app);
    if let Some(state) = app.try_state::<EngineState>() {
        *state.url.lock().unwrap() = None;
    }
    stopped();
    if let Err(e) = start_engine(app) {
        eprintln!("[engine] {e}");
        let _ = app.emit("engine-failed", e);
    }
}

fn start_engine(app: &AppHandle) -> Result<(), String> {
    // dev: движок уже запущен вручную — sidecar не нужен
    if let Ok(url) = std::env::var("DUBL_ENGINE_URL") {
        if !url.trim().is_empty() {
            publish(app, url.trim().trim_end_matches('/').to_string());
            return Ok(());
        }
    }

    let data_dir = data_dir(app)?;
    let _ = std::fs::create_dir_all(&data_dir);

    let (mut rx, child) = app
        .shell()
        .sidecar("dubl-engine")
        .map_err(|e| format!("нет sidecar: {e}"))?
        .args(["--port", "0", "--data-dir", &data_dir.to_string_lossy()])
        .spawn()
        .map_err(|e| format!("движок не запустился: {e}"))?;

    let state = app.state::<EngineState>();
    let my_gen = state.generation.fetch_add(1, Ordering::SeqCst) + 1;
    *state.child.lock().unwrap() = Some(child);

    let handle = app.clone();
    tauri::async_runtime::spawn(async move {
        while let Some(event) = rx.recv().await {
            match event {
                CommandEvent::Stdout(bytes) => {
                    let text = String::from_utf8_lossy(&bytes);
                    for line in text.lines() {
                        if let Some(port) = parse_ready(line) {
                            if current(&handle, my_gen) {
                                publish(&handle, format!("http://127.0.0.1:{port}"));
                            }
                        }
                    }
                }
                CommandEvent::Stderr(bytes) => {
                    eprintln!("[engine] {}", String::from_utf8_lossy(&bytes).trim_end());
                }
                CommandEvent::Terminated(payload) => {
                    eprintln!("[engine] завершился: {:?}", payload.code);
                    if current(&handle, my_gen) {
                        if let Some(state) = handle.try_state::<EngineState>() {
                            *state.url.lock().unwrap() = None;
                            *state.child.lock().unwrap() = None;
                        }
                        let _ = handle.emit("engine-exit", payload.code);
                    }
                    break;
                }
                _ => {}
            }
        }
    });
    Ok(())
}

/// Это ещё тот запуск движка, а не уже заменённый перезапуском?
fn current(app: &AppHandle, generation: u64) -> bool {
    app.try_state::<EngineState>().is_some_and(|s| s.generation.load(Ordering::SeqCst) == generation)
}

fn stop_engine(app: &AppHandle) {
    if let Some(state) = app.try_state::<EngineState>() {
        if let Some(child) = state.child.lock().unwrap().take() {
            let _ = child.kill();
        }
    }
}

pub fn run() {
    let app = tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_dialog::init())
        // «Открыть ролик» и «Показать в папке» после рендера
        .plugin(tauri_plugin_opener::init())
        .manage(EngineState::default())
        .manage(ml_pack::MlState::default())
        .invoke_handler(tauri::generate_handler![engine_url, ml_pack_status, install_ml_pack, cancel_ml_pack, ml_pack_state])
        .setup(|app| {
            if let Err(e) = start_engine(app.handle()) {
                // UI покажет демо-режим, если движка нет
                eprintln!("[engine] {e}");
                let _ = app.emit("engine-failed", e);
            }
            // Только отладочная сборка: DUBL_DEBUG_INSTALL_ML=1 сама запускает установку нейросетей после старта
            // движка — проверка Rust-части без нажатия кнопки в окне (в релизе этого кода нет).
            #[cfg(debug_assertions)]
            if std::env::var("DUBL_DEBUG_INSTALL_ML").is_ok_and(|v| v == "1") {
                let handle = app.handle().clone();
                std::thread::spawn(move || {
                    std::thread::sleep(std::time::Duration::from_secs(3));
                    eprintln!("[debug] ml_pack_status: {:?}", tauri::async_runtime::block_on(ml_pack::status(&handle)));
                    eprintln!("[debug] install_ml_pack: {:?}", ml_pack::start(&handle));
                    // DUBL_DEBUG_CANCEL_AFTER=<секунды>: отменить установку (проверка «Отмены» без окна)
                    if let Some(secs) = std::env::var("DUBL_DEBUG_CANCEL_AFTER").ok().and_then(|v| v.parse::<u64>().ok()) {
                        std::thread::sleep(std::time::Duration::from_secs(secs));
                        eprintln!("[debug] cancel_ml_pack: {}", ml_pack::cancel(&handle));
                    }
                });
            }
            Ok(())
        })
        .build(tauri::generate_context!())
        .expect("ошибка сборки приложения");

    app.run(|handle, event| {
        if let RunEvent::Exit | RunEvent::ExitRequested { .. } = event {
            stop_engine(handle);
        }
    });
}

#[cfg(test)]
mod tests {
    use super::parse_ready;

    #[test]
    fn parses_ready_line() {
        assert_eq!(parse_ready("DUBL_ENGINE_READY port=8765"), Some(8765));
        assert_eq!(parse_ready("  DUBL_ENGINE_READY port=51234\n"), Some(51234));
        assert_eq!(parse_ready("INFO: started"), None);
        assert_eq!(parse_ready("DUBL_ENGINE_READY port=abc"), None);
    }
}
