// Не открываем консольное окно в релизе на Windows
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

fn main() {
    dubl_lib::run()
}
