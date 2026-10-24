fn main() {
    // Свои команды окна — по одной разрешающей записи на каждую (capabilities/main.json), а не «всё разрешено».
    tauri_build::try_build(tauri_build::Attributes::new().app_manifest(tauri_build::AppManifest::new().commands(&[
        "engine_url",
        "ml_pack_status",
        "install_ml_pack",
        "cancel_ml_pack",
        "ml_pack_state",
    ])))
    .expect("ошибка tauri-build");
}
