#![feature(specialization)]
#![feature(adt_const_params)]
#![feature(trivial_bounds)]
use fracgen_gui::FracgenGui;

// When compiling natively:
#[cfg(not(target_arch = "wasm32"))]
fn main() {
    let app = FracgenGui::default();
    let native_options = eframe::NativeOptions::default();
    eframe::run_native(Box::new(app), native_options);
}
