//! Headless-прогон диагностики окружения (envdeps::env_check) на текущей машине.
//!
//!   cargo run --example env_check_smoke
//!   cargo run --example env_check_smoke -- <engine_dir>
//!
//! Печатает статус драйвера NVIDIA, CUDA runtime DLL и VC++ runtime DLL — без
//! Tauri и без загрузки движка. По умолчанию engine_dir = resources/engine дерева
//! разработки; можно передать другую папку первым аргументом (например пустую
//! temp-папку, чтобы увидеть «всё отсутствует»).

use std::path::PathBuf;

use higgs_audio_studio_lib::envdeps;

fn main() {
    let engine_dir = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("resources")
                .join("engine")
        });

    println!("engine_dir = {}", engine_dir.display());
    println!();

    let check = envdeps::env_check(&engine_dir);

    println!("=== Драйвер NVIDIA ===");
    println!(
        "  ok = {}, version = {}",
        check.driver.ok,
        check.driver.version.as_deref().unwrap_or("(нет)")
    );

    print_group("CUDA runtime", &check.cuda);
    print_group("VC++ runtime", &check.vcruntime);

    println!();
    let all_ok = check.driver.ok && check.cuda.ok && check.vcruntime.ok;
    println!("ИТОГ: {}", if all_ok { "всё на месте" } else { "есть недостающее" });
}

fn print_group(title: &str, g: &envdeps::DllGroupStatus) {
    println!();
    println!("=== {title} ===");
    println!("  ok = {}, скачать ~{} МБ", g.ok, g.download_mb);
    if !g.found.is_empty() {
        println!("  найдено:");
        for f in &g.found {
            println!("    {} -> {}", f.name, f.path);
        }
    }
    if !g.missing.is_empty() {
        println!("  отсутствует:");
        for m in &g.missing {
            println!("    {m}");
        }
    }
}
