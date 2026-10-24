//! «Скачать нейросети»: запуск `dubl-engine install-ml-pack` и передача прогресса окну.
//!
//! Аргументы sidecar-а собираются здесь, в Rust, и из JS не задаются — поэтому в capabilities для этого
//! никаких прав не нужно (область действия `shell:allow-spawn` относится только к вызовам из JS).
//!
//! События окну — одно имя `ml-pack`, полезная нагрузка `MlEvent`:
//!   {kind:"stage", stage:"torch", index:3, total:4}   этап установки (uv · python · torch · engine)
//!   {kind:"log", detail:"Downloading torch (60.9MiB)"} последняя строка вывода uv — для подробностей
//!   {kind:"done"}                                       пакет установлен, движок перезапускается (потом придёт engine-ready)
//!   {kind:"failed", reason:"…"}                         не получилось (сырой текст, человеческий — в интерфейсе)
//!   {kind:"cancelled"}                                  пользователь отменил

use std::process::Command;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;

use serde::Serialize;
use tauri::{AppHandle, Emitter, Manager};
use tauri_plugin_shell::{
    process::{CommandChild, CommandEvent},
    ShellExt,
};

use crate::{data_dir, restart_engine};

pub const EVENT: &str = "ml-pack";

#[derive(Default)]
pub struct MlState {
    running: AtomicBool,
    cancelled: AtomicBool,
    child: Mutex<Option<CommandChild>>,
    last: Mutex<Option<MlEvent>>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct MlEvent {
    pub kind: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stage: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub index: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
}

impl MlEvent {
    fn kind(kind: &str) -> Self {
        MlEvent { kind: kind.into(), ..Default::default() }
    }
    fn failed(reason: String) -> Self {
        MlEvent { reason: Some(reason), ..Self::kind("failed") }
    }
}

/// Разобранная строка вывода установщика.
#[derive(Debug, PartialEq)]
pub enum MlLine {
    Stage { name: String, index: u32, total: u32 },
    Done,
    Failed(String),
    /// Обычный вывод (в основном uv): годится как «деталь» под полосой прогресса.
    Detail(String),
}

fn strip_ansi(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '\u{1b}' && chars.peek() == Some(&'[') {
            chars.next();
            for n in chars.by_ref() {
                if n.is_ascii_alphabetic() {
                    break;
                }
            }
        } else {
            out.push(c);
        }
    }
    out
}

/// Строки, которые человеку ничего не говорят: эхо команды, список поставленных пакетов, подсказки uv.
fn is_noise(line: &str) -> bool {
    line.starts_with("$ ")
        || line.starts_with("+ ")
        || line.starts_with("- ")
        || line.starts_with("Activate with")
        || line.starts_with("Using ")
        || line.starts_with("Creating virtual")
}

pub fn parse_line(raw: &str) -> Option<MlLine> {
    let clean = strip_ansi(raw);
    let line = clean.trim();
    if line.is_empty() {
        return None;
    }
    if let Some(rest) = line.strip_prefix("DUBL_ML_STAGE") {
        // `DUBL_ML_STAGE torch 3/4`
        let mut it = rest.split_whitespace();
        let name = it.next()?.to_string();
        let (i, n) = it.next()?.split_once('/')?;
        return Some(MlLine::Stage { name, index: i.parse().ok()?, total: n.parse().ok()? });
    }
    if line == "DUBL_ML_DONE" {
        return Some(MlLine::Done);
    }
    if let Some(rest) = line.strip_prefix("DUBL_ML_FAILED") {
        return Some(MlLine::Failed(rest.trim().to_string()));
    }
    if is_noise(line) {
        return None;
    }
    Some(MlLine::Detail(line.chars().take(200).collect()))
}

/// Режет поток байтов на строки (uv обновляет строку через `\r`, питон печатает `\n`).
#[derive(Default)]
struct LineBuf(String);

impl LineBuf {
    fn push(&mut self, bytes: &[u8]) -> Vec<String> {
        self.0.push_str(&String::from_utf8_lossy(bytes));
        let mut lines = Vec::new();
        while let Some(pos) = self.0.find(['\n', '\r']) {
            let line: String = self.0.drain(..=pos).collect();
            lines.push(line.trim_end_matches(['\n', '\r']).to_string());
        }
        lines
    }
}

fn emit(app: &AppHandle, ev: MlEvent) {
    if let Some(st) = app.try_state::<MlState>() {
        *st.last.lock().unwrap() = Some(ev.clone());
    }
    eprintln!("[ml-pack] {ev:?}");
    let _ = app.emit(EVENT, ev);
}

/// Запустить установку. Возвращает сразу; ход дела — события `ml-pack`.
pub fn start(app: &AppHandle) -> Result<(), String> {
    let st = app.state::<MlState>();
    if st.running.swap(true, Ordering::SeqCst) {
        return Err("Нейросети уже устанавливаются.".into());
    }
    st.cancelled.store(false, Ordering::SeqCst);
    *st.last.lock().unwrap() = None;

    let spawned = (|| {
        let dir = data_dir(app)?;
        app.shell()
            .sidecar("dubl-engine")
            .map_err(|e| format!("нет sidecar: {e}"))?
            .args(["install-ml-pack", "--data-dir", &dir.to_string_lossy(), "--device", "auto"])
            .spawn()
            .map_err(|e| format!("установщик не запустился: {e}"))
    })();
    let (mut rx, child) = match spawned {
        Ok(v) => v,
        Err(e) => {
            st.running.store(false, Ordering::SeqCst);
            return Err(e);
        }
    };
    *st.child.lock().unwrap() = Some(child);
    emit(app, MlEvent { stage: Some("uv".into()), index: Some(1), total: Some(4), ..MlEvent::kind("stage") });

    let handle = app.clone();
    tauri::async_runtime::spawn(async move {
        let (mut out_buf, mut err_buf) = (LineBuf::default(), LineBuf::default());
        let mut done = false;
        let mut failed: Option<String> = None;
        let mut last_detail = String::new();
        while let Some(event) = rx.recv().await {
            let lines = match &event {
                CommandEvent::Stdout(b) => out_buf.push(b),
                CommandEvent::Stderr(b) => err_buf.push(b),
                CommandEvent::Terminated(payload) => {
                    finish(&handle, payload.code, done, failed.take(), &last_detail);
                    return;
                }
                CommandEvent::Error(e) => {
                    finish(&handle, None, false, Some(e.clone()), &last_detail);
                    return;
                }
                _ => continue,
            };
            for l in lines {
                match parse_line(&l) {
                    Some(MlLine::Stage { name, index, total }) => {
                        emit(&handle, MlEvent { stage: Some(name), index: Some(index), total: Some(total), ..MlEvent::kind("stage") })
                    }
                    Some(MlLine::Done) => done = true,
                    Some(MlLine::Failed(r)) => failed = Some(r),
                    Some(MlLine::Detail(d)) => {
                        if d != last_detail {
                            last_detail = d.clone();
                            emit(&handle, MlEvent { detail: Some(d), ..MlEvent::kind("log") });
                        }
                    }
                    None => {}
                }
            }
        }
        // канал закрылся без Terminated — считаем сбоем
        finish(&handle, None, false, failed, &last_detail);
    });
    Ok(())
}

/// Установщик завершился: отмена, ошибка или успех (тогда перезапускаем движок).
fn finish(app: &AppHandle, code: Option<i32>, done: bool, failed: Option<String>, last_detail: &str) {
    let st = app.state::<MlState>();
    st.child.lock().unwrap().take();
    let cancelled = st.cancelled.load(Ordering::SeqCst);
    if cancelled {
        st.running.store(false, Ordering::SeqCst);
        emit(app, MlEvent::kind("cancelled"));
    } else if done && code == Some(0) {
        // сначала сбрасываем адрес старого движка, чтобы окно не подхватило его вместо нового
        restart_engine(app, || emit(app, MlEvent::kind("done")));
        st.running.store(false, Ordering::SeqCst);
    } else {
        st.running.store(false, Ordering::SeqCst);
        let reason = failed
            .filter(|r| !r.is_empty())
            .or_else(|| (!last_detail.is_empty()).then(|| last_detail.to_string()))
            .unwrap_or_else(|| format!("установщик завершился с кодом {}", code.map_or("?".into(), |c| c.to_string())));
        emit(app, MlEvent::failed(reason));
    }
}

/// Отменить установку: убиваем установщик и его дочерний uv (иначе докачка продолжится в фоне).
pub fn cancel(app: &AppHandle) -> bool {
    let st = app.state::<MlState>();
    st.cancelled.store(true, Ordering::SeqCst);
    let child = st.child.lock().unwrap().take();
    match child {
        Some(child) => {
            kill_descendants(child.pid());
            let _ = child.kill();
            true
        }
        None => false,
    }
}

#[cfg(unix)]
fn kill_descendants(pid: u32) {
    // uv — прямой потомок установщика; без него докачка шла бы дальше
    let _ = Command::new("pkill").args(["-KILL", "-P", &pid.to_string()]).status();
}

#[cfg(windows)]
fn kill_descendants(pid: u32) {
    let _ = Command::new("taskkill").args(["/PID", &pid.to_string(), "/T", "/F"]).status();
}

#[derive(Serialize)]
pub struct MlSnapshot {
    pub running: bool,
    pub last: Option<MlEvent>,
}

pub fn snapshot(app: &AppHandle) -> MlSnapshot {
    let st = app.state::<MlState>();
    let last = st.last.lock().unwrap().clone();
    MlSnapshot { running: st.running.load(Ordering::SeqCst), last }
}

/// `dubl-engine ml-pack-status --data-dir …` → JSON как есть (installed, engine_version, pack, recommended_device).
pub async fn status(app: &AppHandle) -> Result<serde_json::Value, String> {
    let dir = data_dir(app)?;
    let out = app
        .shell()
        .sidecar("dubl-engine")
        .map_err(|e| format!("нет sidecar: {e}"))?
        .args(["ml-pack-status", "--data-dir", &dir.to_string_lossy()])
        .output()
        .await
        .map_err(|e| format!("не удалось спросить статус: {e}"))?;
    let text = String::from_utf8_lossy(&out.stdout);
    let json_line = text.lines().rev().find(|l| l.trim_start().starts_with('{')).ok_or_else(|| {
        format!("пустой ответ статуса (код {:?}): {}", out.status.code(), String::from_utf8_lossy(&out.stderr).trim())
    })?;
    serde_json::from_str(json_line).map_err(|e| format!("не разобрал статус: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_stage() {
        assert_eq!(
            parse_line("DUBL_ML_STAGE torch 3/4"),
            Some(MlLine::Stage { name: "torch".into(), index: 3, total: 4 })
        );
        assert_eq!(parse_line("  DUBL_ML_STAGE uv 1/4\r"), Some(MlLine::Stage { name: "uv".into(), index: 1, total: 4 }));
        // испорченная строка этапа не должна падать и не должна считаться деталью установки
        assert_eq!(parse_line("DUBL_ML_STAGE torch"), None);
        assert_eq!(parse_line("DUBL_ML_STAGE torch x/y"), None);
    }

    #[test]
    fn parses_done_and_failed() {
        assert_eq!(parse_line("DUBL_ML_DONE"), Some(MlLine::Done));
        assert_eq!(
            parse_line("DUBL_ML_FAILED команда завершилась с кодом 2: uv pip"),
            Some(MlLine::Failed("команда завершилась с кодом 2: uv pip".into()))
        );
        assert_eq!(parse_line("DUBL_ML_FAILED"), Some(MlLine::Failed(String::new())));
    }

    #[test]
    fn uv_output_is_detail_and_noise_is_dropped() {
        assert_eq!(parse_line("Downloading torch (60.9MiB)"), Some(MlLine::Detail("Downloading torch (60.9MiB)".into())));
        assert_eq!(parse_line(" Downloaded torch"), Some(MlLine::Detail("Downloaded torch".into())));
        assert_eq!(parse_line("\u{1b}[32mPrepared\u{1b}[0m 10 packages in 7.93s"), Some(MlLine::Detail("Prepared 10 packages in 7.93s".into())));
        assert_eq!(parse_line("$ /x/uv pip install torch==2.5.1"), None);
        assert_eq!(parse_line(" + torch==2.5.1"), None);
        assert_eq!(parse_line("Using Python 3.11.16 environment at: /x"), None);
        assert_eq!(parse_line("   "), None);
    }

    #[test]
    fn line_buffer_splits_on_cr_and_lf_and_keeps_tail() {
        let mut b = LineBuf::default();
        assert_eq!(b.push(b"DUBL_ML_STAGE uv 1/4\nDown"), vec!["DUBL_ML_STAGE uv 1/4"]);
        assert_eq!(b.push(b"loading a\rDownloading b\n"), vec!["Downloading a", "Downloading b"]);
        assert!(b.push(b"tail").is_empty());
    }
}
