//! Tauri ждёт sidecar одним файлом (`binaries/dubl-engine-<triple>`), а движок собран папкой
//! (PyInstaller --onedir: быстрый старт, без распаковки в temp при каждом запуске).
//! Этот лаунчер — и есть тот «один файл»: находит `engine/dubl-engine` в ресурсах приложения
//! и запускает его с теми же аргументами.
//!
//! Unix: `exec` — процесс заменяется движком (тот же PID, `child.kill()` из Tauri убивает именно движок).
//! Windows: дочерний процесс + Job Object с KILL_ON_JOB_CLOSE (убили лаунчер — умер и движок);
//!          stdin/stdout/stderr наследуются, код выхода пробрасывается.
//!
//! Поиск ресурсов (первое совпадение, где есть `engine/dubl-engine[.exe]`):
//!   $DUBL_RESOURCE_DIR · <exe>/../Resources (macOS .app) · <exe> (Windows) · <exe>/../lib/*/ (Linux deb/AppImage)
//! Если задан не пустой $DUBL_ENGINE_DIR — берётся он (папка с самим движком).
//! $DUBL_FFMPEG_DIR, если не задан, указывает на `<resources>/ffmpeg`.

use std::env;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode};

const ENGINE_EXE: &str = if cfg!(windows) { "dubl-engine.exe" } else { "dubl-engine" };

fn has_engine(dir: &Path) -> bool {
    dir.join("engine").join(ENGINE_EXE).is_file()
}

fn find_resources(exe_dir: &Path) -> Option<PathBuf> {
    let mut cands: Vec<PathBuf> = Vec::new();
    if let Some(d) = env::var_os("DUBL_RESOURCE_DIR") {
        cands.push(PathBuf::from(d));
    }
    cands.push(exe_dir.join("../Resources"));
    cands.push(exe_dir.to_path_buf());
    cands.push(exe_dir.join("resources"));
    if let Ok(rd) = std::fs::read_dir(exe_dir.join("../lib")) {
        let mut subs: Vec<PathBuf> = rd.flatten().map(|e| e.path()).filter(|p| p.is_dir()).collect();
        subs.sort();
        cands.extend(subs);
    }
    cands.into_iter().find(|d| has_engine(d))
}

fn main() -> ExitCode {
    let exe = match env::current_exe().and_then(|p| p.canonicalize()) {
        Ok(p) => p,
        Err(e) => return fail(&format!("не удалось определить путь запуска: {e}")),
    };
    let exe_dir = exe.parent().unwrap_or(Path::new(".")).to_path_buf();

    let (engine_dir, resources) = match env::var_os("DUBL_ENGINE_DIR").filter(|v| !v.is_empty()) {
        Some(d) => (PathBuf::from(d), None),
        None => match find_resources(&exe_dir) {
            Some(r) => (r.join("engine"), Some(r)),
            None => return fail(&format!("не найдена папка engine/ рядом с {}", exe.display())),
        },
    };
    let engine = engine_dir.join(ENGINE_EXE);
    if !engine.is_file() {
        return fail(&format!("нет файла движка: {}", engine.display()));
    }

    let mut cmd = Command::new(&engine);
    cmd.args(env::args_os().skip(1));
    if env::var_os("DUBL_FFMPEG_DIR").is_none() {
        if let Some(r) = resources {
            let ff = r.join("ffmpeg");
            if ff.is_dir() {
                cmd.env("DUBL_FFMPEG_DIR", ff);
            }
        }
    }
    run(cmd)
}

fn fail(msg: &str) -> ExitCode {
    eprintln!("dubl-engine (лаунчер): {msg}");
    ExitCode::from(2)
}

#[cfg(unix)]
fn run(mut cmd: Command) -> ExitCode {
    use std::os::unix::process::CommandExt;
    let err = cmd.exec(); // возвращается только при ошибке
    fail(&format!("не удалось запустить движок: {err}"))
}

#[cfg(windows)]
fn run(mut cmd: Command) -> ExitCode {
    use std::ffi::c_void;
    use std::os::windows::io::AsRawHandle;

    #[repr(C)]
    struct BasicLimits {
        per_process_user_time_limit: i64,
        per_job_user_time_limit: i64,
        limit_flags: u32,
        minimum_working_set_size: usize,
        maximum_working_set_size: usize,
        active_process_limit: u32,
        affinity: usize,
        priority_class: u32,
        scheduling_class: u32,
    }
    #[repr(C)]
    struct IoCounters {
        _v: [u64; 6],
    }
    #[repr(C)]
    struct ExtendedLimits {
        basic: BasicLimits,
        io: IoCounters,
        process_memory_limit: usize,
        job_memory_limit: usize,
        peak_process_memory_used: usize,
        peak_job_memory_used: usize,
    }
    extern "system" {
        fn CreateJobObjectW(attrs: *mut c_void, name: *const u16) -> *mut c_void;
        fn SetInformationJobObject(job: *mut c_void, class: i32, info: *mut c_void, len: u32) -> i32;
        fn AssignProcessToJobObject(job: *mut c_void, process: *mut c_void) -> i32;
    }
    const JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE: u32 = 0x2000;
    const JOB_OBJECT_EXTENDED_LIMIT_INFORMATION: i32 = 9;
    const CREATE_NO_WINDOW: u32 = 0x0800_0000;

    use std::os::windows::process::CommandExt;
    cmd.creation_flags(CREATE_NO_WINDOW);
    let mut child = match cmd.spawn() {
        Ok(c) => c,
        Err(e) => return fail(&format!("не удалось запустить движок: {e}")),
    };
    unsafe {
        let job = CreateJobObjectW(std::ptr::null_mut(), std::ptr::null());
        if !job.is_null() {
            let mut info: ExtendedLimits = std::mem::zeroed();
            info.basic.limit_flags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
            SetInformationJobObject(
                job,
                JOB_OBJECT_EXTENDED_LIMIT_INFORMATION,
                &mut info as *mut _ as *mut c_void,
                std::mem::size_of::<ExtendedLimits>() as u32,
            );
            AssignProcessToJobObject(job, child.as_raw_handle() as *mut c_void);
            // handle job намеренно не закрываем: закроется вместе с процессом лаунчера
        }
    }
    match child.wait() {
        Ok(s) => ExitCode::from(s.code().unwrap_or(1).clamp(0, 255) as u8),
        Err(e) => fail(&format!("ожидание движка: {e}")),
    }
}
