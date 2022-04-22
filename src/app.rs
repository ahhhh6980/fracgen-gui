// use std::f64::consts::PI;

use crate::color::{Color, ColorType};
use crate::renderer::{normalize_coords, Args, Functs, RenderData, Renderer};
use eframe::{egui, epi};
use egui_nodes::NodeConstructor;
use num::Complex;
use std::f64::consts::PI;
type Cf64 = Complex<f64>;

#[allow(dead_code, unused_variables)]
fn default_bail(rend: &Renderer, z: Cf64, c: Cf64, der: Cf64, der_sum: Cf64) -> bool {
    z.norm_sqr() < rend.args.bail

    // (c - z).norm() * (z - der).norm() < rend.args.bail

    // z.norm_sqr() < rend.args.bail
}

#[allow(dead_code, unused_variables)]
fn der_bail(rend: &Renderer, z: Cf64, c: Cf64, der: Cf64, der_sum: Cf64) -> bool {
    (der_sum * der_sum).norm_sqr() < rend.args.derbail
        && z.norm_sqr() * z.norm_sqr() < rend.args.bail
}

#[allow(dead_code, unused_variables)]
fn stripe_coloring(rend: &Renderer, data: RenderData) {}

#[allow(dead_code, unused_variables)]
fn coloring(rend: &Renderer, data: RenderData) -> Color {
    let hue = (((data.s / rend.args.limit).powf(rend.args.cexp)) * 360.0).powf(1.5);
    Color::new([hue, 1.0, 1.0, 1.0], ColorType::HSVA).to_RGBA()
}
#[allow(dead_code, unused_variables)]
fn idk_coloring(rend: &Renderer, data: RenderData) -> Color {
    let s = (((data.s / rend.args.limit) * 2.0 * PI).cos() + 1.0) / 2.0;
    let v = ((((s / rend.args.limit) * 2.0 * PI) - (PI / 2.0)).sin() + 1.0) / 2.0;
    Color::new(
        [(s / rend.args.limit), 1.0 - s, 1.0 - v, 1.0],
        ColorType::HSVA,
    )
    .to_RGBA()
}

#[allow(dead_code, unused_variables)]
fn miles_coloring(rend: &Renderer, data: RenderData) -> Color {
    let iter_count = data.s;
    // let iter_count = s;

    let sat = (4096.0 / 360.0 * PI * iter_count).cos() / 2.0 + 0.5;
    let val = 1.0 - (2048.0 / 360.0 * PI * iter_count).sin() / 2.0 - 0.5;

    // # convert u into rgb of hue cycle
    let mut r = (((1.0 - 2.0 * (iter_count).cos()) / 2.0).max(0.0)).min(1.0);
    let mut g = (((1.0 - 2.0 * (iter_count + PI * 2.0 / 3.0).cos()) / 2.0).max(0.0)).min(1.0);
    let mut b = (((1.0 - 2.0 * (iter_count + PI * 4.0 / 3.0).cos()) / 2.0).max(0.0)).min(1.0);

    // # apply saturation and brightness to the rgb
    r = ((1.0 + r * sat - sat) * val).sqrt();
    g = ((1.0 + g * sat - sat) * val).sqrt();
    b = ((1.0 + b * sat - sat) * val).sqrt();

    let light_deg = rend.angle;
    let norm_height = 1.5;
    let light_vec = Cf64::new(
        ((light_deg * PI) / 180.0).cos(),
        ((light_deg * PI) / 180.0).sin(),
    );
    let normal_vec = data.z / data.der;
    let normal_vec = normal_vec / normal_vec.norm(); // abs norm_vec
    let mut value = ((normal_vec.re * light_vec.re) + (normal_vec.im * light_vec.im) + norm_height)
        / (1.0 + norm_height);
    if value < 0.0 {
        value = 0.0;
    }
    if value > 1.0 {
        value = 1.0;
    }

    // let hue = (((s / limit).powf(cexp)) * 360.0).powf(1.5);
    let mut color = Color::new([r, g, b, 1.0], ColorType::RGBA).to_RGBA();
    color.ch[2] = color.ch[2] * value;
    color
}

#[allow(dead_code, unused_variables)]
fn miles_coloring2(rend: &Renderer, data: RenderData) -> Color {
    let iter_count = data.s.sqrt().powf(rend.args.cexp);
    // let iter_count = s;

    let sat = (4096.0 / 360.0 * PI * iter_count).cos() / 2.0 + 0.5;
    let sat = 1.0;
    let val = 1.0 - (2048.0 / 360.0 * PI * iter_count).sin() / 2.0 - 0.5;
    let val = 1.0;
    // # convert u into rgb of hue cycle
    let mut r = (((1.0 - 2.0 * (iter_count).cos()) / 2.0).max(0.0)).min(1.0);
    let mut g = (((1.0 - 2.0 * (iter_count + PI * 2.0 / 3.0).cos()) / 2.0).max(0.0)).min(1.0);
    let mut b = (((1.0 - 2.0 * (iter_count + PI * 4.0 / 3.0).cos()) / 2.0).max(0.0)).min(1.0);

    // # apply saturation and brightness to the rgb
    r = ((1.0 + r * sat - sat) * val).sqrt();
    g = ((1.0 + g * sat - sat) * val).sqrt();
    b = ((1.0 + b * sat - sat) * val).sqrt();

    let light_deg = rend.angle;
    let norm_height = 1.5;
    let light_vec = Cf64::new(
        ((light_deg * PI) / 180.0).cos(),
        ((light_deg * PI) / 180.0).sin(),
    );
    let normal_vec = data.z / data.der;
    let normal_vec = normal_vec / normal_vec.norm(); // abs norm_vec
    let mut value = ((normal_vec.re * light_vec.re) + (normal_vec.im * light_vec.im) + norm_height)
        / (1.0 + norm_height);
    if value < 0.0 {
        value = 0.0;
    }
    if value > 1.0 {
        value = 1.0;
    }

    // let hue = (((s / limit).powf(cexp)) * 360.0).powf(1.5);
    let mut color = Color::new([r, g, b, 1.0], ColorType::RGBA).to_RGBA();
    color.ch[2] = color.ch[2] * value;
    color
}

#[allow(dead_code, unused_variables)]
fn normal_map(rend: &Renderer, data: RenderData) -> Color {
    let light_deg = rend.angle;
    let norm_height = 1.5;
    let light_vec = Cf64::new(
        ((light_deg * PI) / 180.0).cos(),
        ((light_deg * PI) / 180.0).sin(),
    );
    let normal_vec = data.z / data.der;
    let normal_vec = normal_vec / normal_vec.norm(); // abs norm_vec
    let mut value = ((normal_vec.re * light_vec.re) + (normal_vec.im * light_vec.im) + norm_height)
        / (1.0 + norm_height);
    if value < 0.0 {
        value = 0.0;
    }
    if value > 1.0 {
        value = 1.0;
    }
    let hue = (((data.s / rend.args.limit).powf(rend.args.cexp)) * 360.0).powf(1.5);
    Color::new([hue, 1.0, value, 1.0], ColorType::HSVA).to_RGBA()
}

#[allow(dead_code, unused_variables)]
fn normal_map_dual(rend: &Renderer, data: RenderData) -> Color {
    let light_deg = rend.angle;
    let norm_height = 1.5;
    let light_vec = Cf64::new(
        ((light_deg * PI) / 180.0).cos(),
        ((light_deg * PI) / 180.0).sin(),
    );
    let normal_vec = data.z / data.der;
    let normal_vec = normal_vec / normal_vec.norm(); // abs norm_vec
    let mut value = ((normal_vec.re * light_vec.re) + (normal_vec.im * light_vec.im) + norm_height)
        / (1.0 + norm_height);
    if value < 0.0 {
        value = 0.0;
    }
    if value > 1.0 {
        value = 1.0;
    }

    let light_deg = rend.angle - rend.angle_offset;
    let norm_height = 2.0;
    let light_vec = Cf64::new(
        ((light_deg * PI) / 180.0).cos(),
        ((light_deg * PI) / 180.0).sin(),
    );
    let normal_vec = data.z / data.der;
    let normal_vec = normal_vec / normal_vec.norm(); // abs norm_vec
    let mut value2 =
        ((normal_vec.re * light_vec.re) + (normal_vec.im * light_vec.im) + norm_height)
            / (1.0 + norm_height);
    if value2 < 0.0 {
        value2 = 0.0;
    }
    if value2 > 1.0 {
        value2 = 1.0;
    }

    let hue = (((data.s / rend.args.limit).powf(rend.args.cexp)) * 360.0).powf(1.5);
    let bound_a = value.powf(1.0);
    let bound_b = 1.0 - value2.powf(1.0);

    let scalar = 1.0 - (data.s / rend.args.limit);
    Color::new(
        [hue, 1.0, ((-bound_b) + bound_a) * scalar, 1.0],
        ColorType::HSLA,
    )
    .to_RGBA_HSLA()
}

#[allow(dead_code, unused_variables)]
fn image_mapping(rend: &Renderer, data: RenderData) -> Color {
    let (w, h) = (rend.texture.width(), rend.texture.height());
    let width = ((data.z.im.atan2(data.z.re) + PI) / (PI * 2.0) * w as f64).round() as u32 % w;
    let height = (h as f64 - 1.0f64)
        - ((data.z.norm() / rend.args.bail).log(rend.args.bail) * (h as f64 - 1.0f64)).floor();
    let mut height = ((height as u32) * 2) % h;
    if data.i as u32 % 2 == 1 {
        height = (h - 1) - height;
    }
    let mut color = Color::new(
        rend.texture
            .get_pixel(width, height)
            .0
            .map(|x| x as f64 / u8::MAX as f64),
        ColorType::RGBA,
    )
    .to_HSVA();

    let light_deg = 270f64;
    let norm_height = 1.5;
    let light_vec = Cf64::new(
        ((light_deg * PI) / 180.0).cos(),
        ((light_deg * PI) / 180.0).sin(),
    );
    let normal_vec = data.z / data.der;
    let normal_vec = normal_vec / normal_vec.norm(); // abs norm_vec
    let mut value = ((normal_vec.re * light_vec.re) + (normal_vec.im * light_vec.im) + norm_height)
        / (1.0 + norm_height);
    if value < 0.0 {
        value = 0.0;
    }
    if value > 1.0 {
        value = 1.0;
    }

    color.ch[2] *= value;
    color.to_RGBA()
}

#[allow(dead_code, unused_variables)]
fn map_complex(rend: &Renderer, c: Cf64) -> Cf64 {
    let nc = Cf64::new(c.im, c.re);
    (nc + 1.0) / ((-nc / 1.25) + 1.0)
}

#[allow(dead_code, unused_variables)]
fn map_complex2(rend: &Renderer, c: Cf64) -> Cf64 {
    // let nc = Cf64::new(c.im, c.re);
    (c + rend.args.julia.re) / ((-c / rend.args.julia.im) + 1.0)
}

#[allow(dead_code, unused_variables)]
fn map_complex3(rend: &Renderer, c: Cf64) -> Cf64 {
    // let nc = Cf64::new(c.im, c.re);
    c / 2.0 - c * c / 4.0
}

#[allow(dead_code, unused_variables)]
fn map_circle(c: Cf64) -> Cf64 {
    1.0 / c
}

#[allow(dead_code, unused_variables)]
fn identity(rend: &Renderer, c: Cf64) -> Cf64 {
    c
}

#[allow(dead_code, unused_variables)]
fn mandelbrot(z: Cf64, c: Cf64, j: Cf64) -> Cf64 {
    let mut nz = z * z * z + c;
    // nz = nz.powc(nz) + (nz / (c + Cf64::new(0.0000001, 0.0000001)));
    nz = Cf64::new(nz.re.abs(), nz.im.abs());
    nz = nz * nz + c;
    nz = Cf64::new(nz.re.abs(), nz.im.abs());
    nz + c
}

#[allow(dead_code, unused_variables)]
fn mandelbrot_j(z: Cf64, c: Cf64, j: Cf64) -> Cf64 {
    let mut nz = z * z * z + j;
    // nz = nz.powc(nz) + (nz / (c + Cf64::new(0.0000001, 0.0000001)));
    nz = Cf64::new(nz.re.abs(), nz.im.abs());
    nz = nz * nz + j;
    nz = Cf64::new(nz.re.abs(), nz.im.abs());
    nz + j
}
// w 1/3
// h 2/3
#[allow(dead_code)]
/// We derive Deserialize/Serialize so we can persist app state on shutdown.
#[cfg_attr(feature = "persistence", derive(serde::Deserialize, serde::Serialize))]
#[cfg_attr(feature = "persistence", serde(default))] // if we add new fields, give them default values when deserializing old state
pub struct FracgenGui<'a> {
    #[cfg_attr(feature = "persistence", serde(skip))]
    args: Args,

    #[cfg_attr(feature = "persistence", serde(skip))]
    renderer: Renderer,

    #[cfg_attr(feature = "persistence", serde(skip))]
    hd_renderer: Renderer,

    #[cfg_attr(feature = "persistence", serde(skip))]
    size: egui::Vec2,

    #[cfg_attr(feature = "persistence", serde(skip))]
    nodes: Vec<NodeConstructor<'a>>,

    #[cfg_attr(feature = "persistence", serde(skip))]
    max_samples: usize,

    #[cfg_attr(feature = "persistence", serde(skip))]
    samples_per_pass: usize,

    #[cfg_attr(feature = "persistence", serde(skip))]
    texture: Option<egui::TextureHandle>,

    #[cfg_attr(feature = "persistence", serde(skip))]
    resize: (bool, f64, f64),

    #[cfg_attr(feature = "persistence", serde(skip))]
    outstanding_size: f64,

    #[cfg_attr(feature = "persistence", serde(skip))]
    last_origin: Cf64,

    #[cfg_attr(feature = "persistence", serde(skip))]
    view_size: egui::Vec2,

    #[cfg_attr(feature = "persistence", serde(skip))]
    julia: bool,

    #[cfg_attr(feature = "persistence", serde(skip))]
    current_iter: fn(Cf64, Cf64, Cf64) -> Cf64,

    #[cfg_attr(feature = "persistence", serde(skip))]
    current_iter_j: fn(Cf64, Cf64, Cf64) -> Cf64,

    #[cfg_attr(feature = "persistence", serde(skip))]
    iter_changed: bool,

    #[cfg_attr(feature = "persistence", serde(skip))]
    keyp: [bool; 32],
}

#[allow(unused_variables)]
impl<'a> Default for FracgenGui<'a> {
    fn default() -> Self {
        Self {
            args: Args::new(),
            renderer: Renderer::new(
                Args::new(),
                Functs::new(
                    mandelbrot,
                    move |z, c| z,
                    identity,
                    normal_map_dual,
                    der_bail,
                ),
            ),
            hd_renderer: Renderer::new(
                Args::new(),
                Functs::new(
                    mandelbrot,
                    move |z, c| z,
                    identity,
                    normal_map_dual,
                    der_bail,
                ),
            ),
            size: egui::Vec2::new(960.0 / 2.0, 840.0 / 2.0),
            nodes: Vec::new(),
            max_samples: usize::MAX,
            samples_per_pass: 1,
            texture: None,
            resize: (false, 0.0, 0.0),
            outstanding_size: 1.0,
            last_origin: Cf64::new(0.0, 0.0),
            view_size: egui::Vec2::new(0.0, 0.0),
            julia: false,
            current_iter: move |z, c, j| z * z + c,
            current_iter_j: move |z, c, j| z * z + j,
            iter_changed: false,
            keyp: [true; 32],
        }
    }
}

#[allow(unused_variables)]
impl<'a> epi::App for FracgenGui<'a> {
    fn name(&self) -> &str {
        "Fracgen Gui"
    }

    /// Called once before the first frame.
    fn setup(
        &mut self,
        _ctx: &egui::Context,
        _frame: &epi::Frame,
        _storage: Option<&dyn epi::Storage>,
    ) {
        rayon::ThreadPoolBuilder::new()
            .num_threads(self.renderer.args.threads)
            .build_global()
            .unwrap();
        _frame.set_window_size(self.size);
        self.renderer.texture = image::open("JoeBiden-2.webp").unwrap().to_rgba8();
        self.renderer.args.cexp = 0.8;
        self.renderer.render_samples(self.samples_per_pass, true);
        println!("Hanging until first render");
        while self.renderer.not_rendering == false {}
        self.renderer.process_image();
        println!("Done with first render");
        let texture: &egui::TextureHandle = self.texture.get_or_insert_with(|| {
            // Load the texture only once.
            _ctx.load_texture(
                "my-image",
                epaint::ColorImage {
                    size: [self.args.width as usize, self.args.height as usize],
                    pixels: self
                        .renderer
                        .image
                        .clone()
                        .pixels()
                        .map(|x| -> epaint::Color32 {
                            epaint::color::Color32::from_rgba_premultiplied(
                                x.0[0], x.0[1], x.0[2], x.0[3],
                            )
                        })
                        .collect::<Vec<_>>(),
                },
            )
        });
        // self.texture = egui::TextureHandle.
        // Load previous app state (if any).
        // Note that you must enable the `persistence` feature for this to work.
        #[cfg(feature = "persistence")]
        if let Some(storage) = _storage {
            *self = epi::get_value(storage, epi::APP_KEY).unwrap_or_default()
        }
    }

    /// Called by the frame work to save state before shutdown.
    /// Note that you must enable the `persistence` feature for this to work.
    #[cfg(feature = "persistence")]
    fn save(&mut self, storage: &mut dyn epi::Storage) {
        epi::set_value(storage, epi::APP_KEY, self);
    }

    /// Called each time the UI needs repainting, which may be many times per second.
    /// Put your widgets into a `SidePanel`, `TopPanel`, `CentralPanel`, `Window` or `Area`.
    fn update(&mut self, ctx: &egui::Context, frame: &epi::Frame) {
        // frame.set_window_size(self.size);

        // if self.resize.0 {
        //     self.resize.0 = false;
        //     self.texture = None;
        //     let texture: &egui::TextureHandle = self.texture.get_or_insert_with(|| {
        //         ctx.load_texture("my-image", egui::ColorImage::example())
        //     });
        //     self.renderer.resize(self.resize.1 as usize, self.resize.2 as usize);
        //     self.renderer.render_samples(1);
        //     while self.renderer.not_rendering == false {}
        //     self.renderer.process_image();
        // }

        let input = ctx.input().clone();
        if self.keyp[4] && input.key_pressed(egui::Key::Enter) {
            self.keyp[4] = false;
            println!("Waiting for frame!");
            while self.renderer.not_rendering == false {}
            println!("Will render now!");
            self.hd_renderer.angle = self.renderer.angle;
            self.hd_renderer.angle_offset = self.renderer.angle_offset;
            self.hd_renderer.update_args(self.renderer.args.clone());
            self.hd_renderer.update_functs(self.renderer.functs.clone());
            self.hd_renderer.texture = self.renderer.texture.clone();
            self.hd_renderer.resize(
                3840,
                (self.renderer.args.height as f64 * ((3840.0) / self.renderer.args.width as f64))
                    .round() as usize,
            );
            self.hd_renderer
                .render_samples(self.renderer.rendered_samples, true);
            println!("Starting Rendering!");
            while self.hd_renderer.not_rendering == false {}
            println!("Rendered!");
            self.hd_renderer.process_image();
            self.hd_renderer.image.save("QHD.png").unwrap();
            println!("Saved!");
        }
        if input.key_released(egui::Key::Enter) {
            self.keyp[4] = true;
        }
        let files = input.raw.clone().dropped_files.clone();
        if files.len() > 0 {
            println!(
                "{}",
                String::from(files[0].path.clone().unwrap().as_os_str().to_string_lossy())
            );
            self.renderer.texture = image::open(String::from(
                files[0].path.clone().unwrap().as_os_str().to_string_lossy(),
            ))
            .unwrap()
            .to_rgba8();
            self.renderer.rendered_samples = 0;
            self.renderer.render_samples(self.samples_per_pass, false);
        }
        let mut pressed = false;
        // for event in input.events.clone() {
        //     match event {
        //         egui::Event::Scroll(v) => {
        //             if v.y > 0.0 {
        //                 self.renderer.args.zoom *= 1.5 * (v.y / (v.y.abs())).abs();
        //             } else {
        //                 self.renderer.args.zoom /= 1.5 * (v.y / (v.y.abs())).abs();
        //             }

        //             pressed = true;
        //         },
        //         _ => (),
        //     }
        // }
        let mut scalar_mult = 1.0;
        let mut axis_bool = false;
        let mut alt_code = false;
        let mut ctrl_code = false;
        match input.modifiers {
            egui::Modifiers {
                alt: a,
                ctrl: c,
                shift: s,
                mac_cmd: m,
                command: c2,
            } => {
                if s {
                    axis_bool = true;
                    scalar_mult = 0.5;
                }
                if a {
                    alt_code = true;
                }
                if c {
                    ctrl_code = true;
                }
            }
        }
        if alt_code {
            for event in input.events.clone() {
                match event {
                    egui::Event::Key {
                        key: k,
                        pressed: p,
                        modifiers: m,
                    } => match k {
                        egui::Key::Num1 => {
                            self.current_iter = mandelbrot;
                            self.current_iter_j = |z, c, j| z * z + j;
                            self.iter_changed = true;
                        }
                        egui::Key::Num4 => {
                            self.current_iter = |z, c, j| z * z * z + c;
                            self.current_iter_j = |z, c, j| z * z * z + j;
                            self.iter_changed = true;
                        }
                        egui::Key::Num5 => {
                            self.current_iter = |z, c, j| (z * c).powc(z / c) + (z / c);
                            self.current_iter_j = |z, c, j| (z * c).powc(z / c) + (z / c);
                            self.iter_changed = true;
                        }
                        egui::Key::Num6 => {
                            self.current_iter = |z, c, j| z.powf(c.re) + c;
                            self.current_iter_j = |z, c, j| (z * c).powc(z / c) + (z / c);
                            self.iter_changed = true;
                        }
                        egui::Key::Num7 => {
                            self.current_iter = mandelbrot;
                            self.current_iter_j = mandelbrot_j;
                            self.iter_changed = true;
                        }
                        egui::Key::Num2 => {
                            self.current_iter = |z, c, j| {
                                Cf64::new(z.re.abs(), z.im.abs())
                                    * Cf64::new(z.re.abs(), z.im.abs())
                                    + c
                            };
                            self.current_iter_j = |z, c, j| {
                                Cf64::new(z.re.abs(), z.im.abs())
                                    * Cf64::new(z.re.abs(), z.im.abs())
                                    + j
                            };
                            self.iter_changed = true;
                        }
                        egui::Key::Num3 => {
                            self.current_iter = |z, c, j| {
                                let mut tz = z * z + c;
                                tz = Cf64::new(tz.re.abs(), tz.im.abs())
                                    * Cf64::new(tz.re.abs(), tz.im.abs())
                                    + c;
                                tz = tz * tz + c;
                                tz = tz * tz + c;
                                tz = Cf64::new(tz.re.abs(), tz.im) * Cf64::new(tz.re, tz.im.abs())
                                    + j;
                                tz
                            };
                            self.current_iter_j = |z, c, j| {
                                let mut tz = z * z + j;
                                tz = Cf64::new(tz.re.abs(), tz.im.abs())
                                    * Cf64::new(tz.re.abs(), tz.im.abs())
                                    + j;
                                tz = tz * tz + j;
                                tz = tz * tz + j;
                                tz = Cf64::new(tz.re.abs(), tz.im) * Cf64::new(tz.re, tz.im.abs())
                                    + j;
                                tz
                            };
                            self.iter_changed = true;
                        }
                        _ => (),
                    },
                    _ => (),
                }
            }
        } else if ctrl_code {
            for event in input.events.clone() {
                match event {
                    egui::Event::Key {
                        key: k,
                        pressed: p,
                        modifiers: m,
                    } => match k {
                        egui::Key::Num1 => {
                            self.renderer.functs.color_funct = coloring;
                            pressed = true;
                        }
                        egui::Key::Num2 => {
                            self.renderer.functs.color_funct = miles_coloring2;
                            pressed = true;
                        }
                        egui::Key::Num3 => {
                            self.renderer.functs.color_funct = normal_map;
                            pressed = true;
                        }
                        egui::Key::Num4 => {
                            self.renderer.functs.color_funct = image_mapping;
                            pressed = true;
                        }
                        egui::Key::Num5 => {
                            self.renderer.functs.color_funct = miles_coloring;
                            pressed = true;
                        }
                        egui::Key::Num6 => {
                            self.renderer.functs.color_funct = normal_map_dual;
                            pressed = true;
                        }
                        _ => (),
                    },
                    _ => (),
                }
            }
        } else {
            for event in input.events.clone() {
                match event {
                    egui::Event::Key {
                        key: k,
                        pressed: p,
                        modifiers: m,
                    } => match k {
                        egui::Key::Num1 => {
                            self.renderer.functs.color_funct = coloring;
                            pressed = true;
                        }
                        egui::Key::Num2 => {
                            self.renderer.functs.color_funct = normal_map;
                            pressed = true;
                        }
                        egui::Key::Num3 => {
                            self.renderer.functs.color_funct = image_mapping;
                            pressed = true;
                        }
                        egui::Key::Num4 => {
                            self.renderer.functs.color_funct = miles_coloring;
                            pressed = true;
                        }
                        egui::Key::Num6 => {
                            self.renderer.functs.color_funct = normal_map_dual;
                            pressed = true;
                        }
                        _ => (),
                    },
                    _ => (),
                }
            }
        }
        for event in input.events.clone() {
            match event {
                egui::Event::Scroll(v) => {
                    let mut q = v.y;
                    if axis_bool {
                        q = v.x;
                    }
                    if (q > 0.0 && axis_bool == false) || (q < 0.0 && axis_bool) {
                        self.renderer.args.zoom *= 1.5 * scalar_mult;
                    } else {
                        self.renderer.args.zoom /= 1.5 * scalar_mult;
                    }

                    pressed = true;
                }
                _ => (),
            }
        }
        if input.pointer.primary_down() {
            let mut m: egui::Vec2 = egui::Vec2::new(0.0, 0.0);
            if let Some(o) = input.pointer.press_origin() {
                if let Some(n) = input.pointer.hover_pos() {
                    m = n - o;
                    m = egui::Vec2::new(m.x, m.y);
                }
            }
            let m = (m / self.renderer.args.width.min(self.renderer.args.height) as f32)
                / self.renderer.args.zoom as f32;
            self.renderer.args.origin =
                self.last_origin - (Cf64::new(m.x as f64, m.y as f64) / self.outstanding_size);
            pressed = true;
        } else {
            self.last_origin = self.renderer.args.origin;
        }
        if input.key_pressed(egui::Key::J) {
            self.julia = !self.julia;
            if self.julia {
                self.renderer.functs.init_funct = |z, c| c;
                self.renderer.functs.iter_funct = self.current_iter_j;
            } else {
                self.renderer.functs.init_funct = |z, c| z;
                self.renderer.functs.iter_funct = self.current_iter;
            }
            pressed = true;
        }
        if self.julia {
            if input.pointer.middle_down() {
                self.renderer.functs.init_funct = |z, c| z;
                self.renderer.functs.iter_funct = self.current_iter;
            } else {
                self.renderer.functs.init_funct = |z, c| c;
                self.renderer.functs.iter_funct = self.current_iter_j;
            }
        }

        if input.pointer.secondary_down() && self.julia {
            let mut m: egui::Vec2 = egui::Vec2::new(0.0, 0.0);
            if let Some(n) = input.pointer.hover_pos() {
                m = egui::Vec2::new(n.x, n.y);
            }
            let m = normalize_coords(
                m.x as i32,
                m.y as i32 - 32,
                (self.renderer.args.width as f64 * self.outstanding_size) as i32 + 16,
                (self.renderer.args.height as f64 * self.outstanding_size) as i32 + 16,
                self.renderer.args.zoom,
            ) + self.renderer.args.origin;
            self.renderer.args.julia = m;
            println!("{}", m);
            pressed = true;
        }

        if self.keyp[5] && input.key_pressed(egui::Key::Z) {
            self.renderer.args.derbail /= 1.5;
            self.keyp[5] = false;
            pressed = true;
        }
        if !self.keyp[5] && input.key_released(egui::Key::Z) {
            self.keyp[5] = true;
        }

        if self.keyp[5] && input.key_pressed(egui::Key::X) {
            self.renderer.args.derbail *= 1.5;
            self.keyp[5] = false;
            pressed = true;
        }
        if !self.keyp[5] && input.key_released(egui::Key::X) {
            self.keyp[5] = true;
        }

        if self.keyp[7] && input.key_pressed(egui::Key::O) {
            self.renderer.angle += (360. / 64.) % 360.0;
            println!(
                "a:{}, o:{}",
                self.renderer.angle, self.renderer.angle_offset
            );
            self.keyp[7] = false;
            pressed = true;
        }
        if !self.keyp[7] && input.key_released(egui::Key::O) {
            self.keyp[7] = true;
        }

        if self.keyp[6] && input.key_pressed(egui::Key::I) {
            self.renderer.angle -= (360. / 64.) % 360.0;
            println!(
                "a:{}, o:{}",
                self.renderer.angle, self.renderer.angle_offset
            );
            self.keyp[6] = false;
            pressed = true;
        }
        if !self.keyp[6] && input.key_released(egui::Key::I) {
            self.keyp[6] = true;
        }

        if self.keyp[9] && input.key_pressed(egui::Key::L) {
            self.renderer.angle_offset += (360. / 64.) % 360.0;
            println!(
                "a:{}, o:{}",
                self.renderer.angle, self.renderer.angle_offset
            );
            self.keyp[9] = false;
            pressed = true;
        }
        if !self.keyp[9] && input.key_released(egui::Key::L) {
            self.keyp[9] = true;
        }

        if self.keyp[8] && input.key_pressed(egui::Key::K) {
            self.renderer.angle_offset -= (360. / 64.) % 360.0;
            println!(
                "a:{}, o:{}",
                self.renderer.angle, self.renderer.angle_offset
            );
            self.keyp[8] = false;
            pressed = true;
        }
        if !self.keyp[8] && input.key_released(egui::Key::K) {
            self.keyp[8] = true;
        }

        if alt_code {
            if self.keyp[0] && input.key_pressed(egui::Key::ArrowUp) {
                self.renderer.args.bail *= 1.5;
                self.keyp[0] = false;
                pressed = true;
            }
            if !self.keyp[0] && input.key_released(egui::Key::ArrowUp) {
                self.keyp[0] = true;
            }

            if self.keyp[1] && input.key_pressed(egui::Key::ArrowDown) {
                self.renderer.args.bail /= 1.5;
                self.keyp[1] = false;
                pressed = true;
            }
            if !self.keyp[1] && input.key_released(egui::Key::ArrowDown) {
                self.keyp[1] = true;
            }

            if self.keyp[2] && input.key_pressed(egui::Key::ArrowLeft) {
                self.renderer.args.cexp *= scalar_mult * 1.1;
                self.keyp[2] = false;
                pressed = true;
            }
            if !self.keyp[2] && input.key_released(egui::Key::ArrowLeft) {
                self.keyp[2] = true;
            }
            if self.keyp[3] && input.key_pressed(egui::Key::ArrowRight) {
                self.renderer.args.cexp /= scalar_mult * 1.1;
                self.keyp[3] = false;
                pressed = true;
            }
            if !self.keyp[3] && input.key_released(egui::Key::ArrowRight) {
                self.keyp[3] = true;
            }
        } else {
            if self.keyp[0] && input.key_pressed(egui::Key::ArrowUp) {
                self.renderer.args.sampled *= 1.5;
                self.keyp[0] = false;
                pressed = true;
            }
            if !self.keyp[0] && input.key_released(egui::Key::ArrowUp) {
                self.keyp[0] = true;
            }
            if self.keyp[1] && input.key_pressed(egui::Key::ArrowDown) {
                self.renderer.args.sampled /= 1.5;
                self.keyp[1] = false;
                pressed = true;
            }
            if !self.keyp[1] && input.key_released(egui::Key::ArrowDown) {
                self.keyp[1] = true;
            }
            if self.keyp[2]
                && input.key_pressed(egui::Key::ArrowLeft)
                && self.renderer.args.limit > 16.0
            {
                self.renderer.args.limit /= scalar_mult * 1.1;
                self.keyp[2] = false;
                pressed = true;
            }
            if !self.keyp[2] && input.key_released(egui::Key::ArrowLeft) {
                self.keyp[2] = true;
            }
            if self.keyp[3] && input.key_pressed(egui::Key::ArrowRight) {
                self.renderer.args.limit *= scalar_mult * 1.1;
                self.keyp[3] = false;
                pressed = true;
            }
            if !self.keyp[3] && input.key_released(egui::Key::ArrowRight) {
                self.keyp[3] = true;
            }
        }

        if input.key_pressed(egui::Key::Space) {
            self.renderer.render_samples(8, false);
        }

        if input.key_pressed(egui::Key::E) {
            self.renderer.process_image();
            self.renderer
                .image
                .save(format!("{}_{}.png", self.renderer.args.name, "lol"))
                .unwrap();
        }

        if self.iter_changed {
            if self.julia {
                self.renderer.functs.iter_funct = self.current_iter_j;
            } else {
                self.renderer.functs.iter_funct = self.current_iter;
            }
            self.iter_changed = false;
            pressed = true;
        }

        if pressed {
            self.renderer.rendered_samples = 0;
            self.renderer.render_samples(self.samples_per_pass, false);
        }
        // let screen = egui::Vec2::new(input.screen_rect().width(), input.screen_rect().height());
        // // debug!(ctx.used_size);
        // if screen != self.size {
        //     self.size = screen;
        // }

        egui::TopBottomPanel::top("top_panel").show(ctx, |ui| {
            // The top panel is often a good place for a menu bar:
            egui::menu::bar(ui, |ui| {
                ui.menu_button("File", |ui| {
                    if ui.button("Quit").clicked() {
                        frame.quit();
                    }
                });
            });
        });

        // egui::TopBottomPanel::bottom("bottom_panel")
        //     .default_height(self.size.x / 8.0)
        //     .resizable(true)
        //     .show(ctx, |ui| {
        //         ui.with_layout(
        //             egui::Layout::top_down_justified(egui::Align::Center),
        //             |ui| {

        //                 ui.add(egui::Slider::new(&mut self.args.sampled, 1.0..=16.0).suffix("^-1"));

        //                 // if ui.slider("").dragged() {
        //                 //     self.renderer.rendered_samples = 0;
        //                 //     println!("ok");
        //                 // }
        //             },
        //         );
        //         ui.allocate_space(ui.available_size());
        //     });

        egui::CentralPanel::default().show(ctx, |ui| {
            let mut texture: &egui::TextureHandle = self.texture.get_or_insert_with(|| {
                // Load the texture only once.
                ui.ctx()
                    .load_texture("my-image", egui::ColorImage::example())
            });
            if self.renderer.rendered_samples < self.max_samples && self.renderer.not_rendering {
                self.renderer.process_image();
                self.texture = None;
                texture = self.texture.get_or_insert_with(|| {
                    // Load the texture only once.
                    ctx.load_texture(
                        "my-image",
                        epaint::ColorImage {
                            size: [self.args.width as usize, self.args.height as usize],
                            pixels: self
                                .renderer
                                .image
                                .pixels()
                                .map(|x| -> epaint::Color32 {
                                    epaint::color::Color32::from_rgba_premultiplied(
                                        x.0[0], x.0[1], x.0[2], x.0[3],
                                    )
                                })
                                .collect::<Vec<_>>(),
                        },
                    )
                });

                // println!("Rendering Pass {}", self.renderer.rendered_samples / self.samples_per_pass);
                self.renderer.render_samples(self.samples_per_pass, false);
            }
            ui.with_layout(
                egui::Layout::top_down_justified(egui::Align::Center),
                |ui| {
                    // let w = ui.available_width();
                    // let h = self.renderer.args.height as f64 * (w as f64 / self.renderer.args.width as f64);
                    self.view_size = egui::Vec2::new(ui.available_width(), ui.available_height());
                    let w = ui.available_width();
                    let h = self.renderer.args.height as f64
                        * (w as f64 / self.renderer.args.width as f64);
                    self.outstanding_size = w as f64 / self.renderer.args.width as f64;
                    ui.image(texture, egui::Vec2::new(w as f32, h as f32));
                    if w != self.args.width as f32 || h != self.args.height as f64 {
                        self.resize = (true, w as f64, h as f64);
                    }
                },
            );
            egui::warn_if_debug_build(ui);
        });

        egui::TopBottomPanel::bottom("bottom_panel")
            .default_height(self.size.x / 8.0)
            .resizable(true)
            .show(ctx, |ui| {
                // if ui.ui_contains_pointer() {
                //     self.viewer = true;
                // } else {
                //     self.viewer = false;
                // }
                ui.horizontal(|ui| {
                    ui.label("limit");
                    let mut string = String::from(format!("{}", self.renderer.args.limit));
                    let response = ui.add(egui::TextEdit::singleline(&mut string));
                    if response.lost_focus() {
                        // use my_string
                    }
                    ui.label("bail");
                    let mut string = String::from(format!("{}", self.renderer.args.bail));
                    let response = ui.add(egui::TextEdit::singleline(&mut string));
                    if response.lost_focus() {
                        // use my_string
                    }
                    ui.label("zoom");
                    let mut string = String::from(format!("{}", self.renderer.args.zoom));
                    let response = ui.add(egui::TextEdit::singleline(&mut string));
                    if response.lost_focus() {
                        // use my_string
                    }
                });
                ui.horizontal(|ui| {
                    ui.label("samples");
                    let mut string = String::from(format!("{}", self.renderer.rendered_samples));
                    let response = ui.add(
                        egui::TextEdit::singleline(&mut string).desired_width(self.size.x / 7.5),
                    );
                    if response.lost_focus() {
                        // use my_string
                    }
                    ui.label("max samples");
                    let mut string = String::from(format!("{}", self.max_samples));
                    let response = ui.add(
                        egui::TextEdit::singleline(&mut string).desired_width(self.size.x / 7.5),
                    );
                    if response.lost_focus() {
                        // use my_string
                    }
                    ui.label("samples per pass");
                    let mut string = String::from(format!("{}", self.samples_per_pass));
                    let response = ui.add(
                        egui::TextEdit::singleline(&mut string).desired_width(self.size.x / 7.5),
                    );
                    if response.lost_focus() {
                        // use my_string
                    }
                    ui.label("sample distance (1/n px)");
                    let mut string = String::from(format!("{}", self.renderer.args.sampled));
                    let response = ui.add(
                        egui::TextEdit::singleline(&mut string).desired_width(self.size.x / 7.5),
                    );
                    if response.lost_focus() {
                        // use my_string
                    }
                });
                ui.horizontal(|ui| {
                    ui.label("Origin");
                    ui.add_space(self.size.x / 24.0);
                    ui.label("Re");
                    let mut string = String::from(format!("{}", self.renderer.args.origin.re));
                    let response = ui.add(
                        egui::TextEdit::singleline(&mut string).desired_width(self.size.x / 6.0),
                    );
                    if response.lost_focus() {
                        // use my_string
                    }
                    ui.label("Im");
                    let mut string = String::from(format!("{}", self.renderer.args.origin.im));
                    let response = ui.add(
                        egui::TextEdit::singleline(&mut string).desired_width(self.size.x / 6.0),
                    );
                    if response.lost_focus() {
                        // use my_string
                    }
                    ui.label("Julia");
                    ui.add_space(self.size.x / 24.0);
                    ui.label("Re");
                    let mut string = String::from(format!("{}", self.renderer.args.julia.re));
                    let response = ui.add(
                        egui::TextEdit::singleline(&mut string).desired_width(self.size.x / 6.0),
                    );
                    if response.lost_focus() {
                        // use my_string
                    }
                    ui.label("Im");
                    let mut string = String::from(format!("{}", self.renderer.args.julia.im));
                    let response = ui.add(
                        egui::TextEdit::singleline(&mut string).desired_width(self.size.x / 6.0),
                    );
                    if response.lost_focus() {
                        // use my_string
                    }
                });

                // ui.label("Hello World!");
            });
        // egui::TopBottomPanel::bottom("bottom_panel")
        //     .default_height(self.size.x / 8.0)
        //     .resizable(true)
        //     .show(ctx, |ui| {
        //         ui.with_layout(
        //             egui::Layout::top_down_justified(egui::Align::Center),
        //             |ui| {

        //                 ui.add(egui::Slider::new(&mut self.args.sampled, 1.0..=16.0).suffix("^-1"));

        //                 // if ui.slider("").dragged() {
        //                 //     self.renderer.rendered_samples = 0;
        //                 //     println!("ok");
        //                 // }
        //             },
        //         );
        //         ui.allocate_space(ui.available_size());
        //     });
    }
}
