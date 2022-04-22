#![forbid(unsafe_code)]
#![cfg_attr(not(debug_assertions), deny(warnings))] // Forbid warnings in release builds
#![warn(clippy::all, rust_2018_idioms)]
use fracgen_gui::FracgenGui;

// When compiling natively:
#[cfg(not(target_arch = "wasm32"))]
fn main() {
    let app = FracgenGui::default();
    let native_options = eframe::NativeOptions::default();
    eframe::run_native(Box::new(app), native_options);
}
