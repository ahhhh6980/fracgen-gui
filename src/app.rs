use epaint::{Color32, ColorImage, TextureHandle};
use gif::{Encoder, Frame, Repeat};
use std::borrow::Cow;
use std::error::Error;
use std::ffi::OsString;
use std::fs::File;
use std::io::{stdin, stdout, Write};
use std::ops::RangeInclusive;
use std::thread;
use std::time::{Duration, SystemTime};
// use std::f64::consts::PI;

// use crate::color::{Color, ColorType};
use crate::renderer::{normalize_coords, Args, Functs, RenderData, Renderer};
use colortypes::{CIELab, CIELch, Color, FromColorType, Image, Rgb, D65};
use eframe::epi::file_storage::FileStorage;
use eframe::{egui, epi};
use egui_nodes::NodeConstructor;
use num::traits::MulAdd;
use num::Complex;
use std::f64::consts::PI;
type Cf64 = Complex<f64>;

#[allow(dead_code, unused_variables)]
fn default_bail(rend: &Renderer, z: Cf64, c: Cf64, der: Cf64, der_sum: Cf64) -> bool {
    z.norm_sqr() < rend.args.bail
}

#[allow(dead_code, unused_variables)]
fn der_bail(rend: &Renderer, z: Cf64, c: Cf64, der: Cf64, der_sum: Cf64) -> bool {
    (der_sum * der_sum).norm_sqr() < rend.args.derbail
        && z.norm_sqr() * z.norm_sqr() < rend.args.bail
}

#[allow(dead_code, unused_variables)]
fn coloring(rend: &Renderer, data: RenderData) -> Color<CIELab, D65> {
    let s = (data.s / rend.args.limit).powf(rend.args.cexp);
    let val = (PI * s).cos();
    let val = 1.0 - (val * val);

    CIELab::from_color(CIELch::new::<D65>([
        74.0 - (74.0 * val),
        28.0 + (74.0 * val),
        (s * 360.0).powf(1.5) % 360.0,
        1.0,
    ]))
}

#[allow(dead_code, unused_variables)]
fn normal_map(rend: &Renderer, data: RenderData) -> Color<CIELab, D65> {
    //  Normal mapped variant
    let lo = 0.5 * data.z.norm_sqr().ln();
    let u = data.z
        * data.der
        * ((1.0 + lo) * (data.der * data.der).conj() - lo * (data.z * data.der2).conj());
    let u = u / u.norm();

    let v = Cf64::new(
        (rend.angle.to_radians()).cos(),
        (rend.angle.to_radians()).sin(),
    );
    let norm_height = rend.args.h;
    let t = (u.re * v.re + u.im * v.im + norm_height) / (1.0 + norm_height);

    // Normal
    let lo = 0.5 * data.lastz.norm_sqr().ln();
    let u = data.lastz
        * data.lastder
        * ((1.0 + lo) * (data.lastder * data.lastder).conj()
            - lo * (data.lastz * data.lastder2).conj());
    let u = u / u.norm();

    let v = Cf64::new(
        (rend.angle.to_radians()).cos(),
        (rend.angle.to_radians()).sin(),
    );
    let norm_height = rend.args.h;
    let t2 = (u.re * v.re + u.im * v.im + norm_height) / (1.0 + norm_height);

    let f = 1.0 + (((rend.args.bail).ln() / (data.lastz.norm_sqr()).ln()).log2());

    let v = (t * f) + (t2 * (1.0 - f));

    let val = (PI * (1.0 - t)).cos();
    let val = 1.0 - (val * val);

    // Color::new(t.to_arr().0, ColorType::RGBA)
    CIELab::from_color(CIELch::new::<D65>([
        74.0 - (74.0 * val),
        28.0 + (74.0 * val),
        ((data.s / rend.args.limit).powf(rend.args.cexp) * 360.0).powf(1.5) % 360.0,
        1.0,
    ]))
}

#[allow(dead_code, unused_variables)]
fn stripe_normal(rend: &Renderer, data: RenderData) -> Color<CIELab, D65> {
    // Triangle inequality average
    let last_orbit = 0.5 + 0.5 * (3.0 * data.lastz.arg()).sin();
    // let last_orbit = idk(data.lastz);
    let f = 1.0 + (((rend.args.bail).ln() / (data.lastz.norm_sqr()).ln()).log2());
    let value = (f * (data.sum / data.i)) + (1.0 - f) * ((data.sum - last_orbit) / (data.i - 1.0));

    //  Normal mapped variant
    let lo = 0.5 * data.z.norm_sqr().ln();
    let u = data.z
        * data.der
        * ((1.0 + lo) * (data.der * data.der).conj() - lo * (data.z * data.der2).conj());
    let u = u / u.norm();

    let v = Cf64::new(
        (rend.angle.to_radians()).cos(),
        (rend.angle.to_radians()).sin(),
    );
    let norm_height = 1.5;
    let t = (value * (u.re * v.re + u.im * v.im + norm_height)) / (1.0 + norm_height);

    // LCH Computation
    // let t = data.s / data.i;
    let val = (PI * t).cos();
    let val = 1.0 - (val * val);

    let c = 28.0 + (74.0 * val);
    let l = 74.0 - (74.0 * val);
    let h = (((data.s / rend.args.limit).powf(rend.args.cexp)) * 360.0).powf(1.5);

    CIELab::from_color(CIELch::new::<D65>([l, c, h % 360.0, 1.0]))
}

fn fade_original(rend: &Renderer, data: RenderData) -> Color<CIELab, D65> {
    // v = range1.mapExp( out.i, range2, exponent);
    // rgba = hsv_to_rgb( (v * cMult) + cOffset , 1, 1 - Range<double>{limit}.mapExp(valueChange, Range<double>{}, 0.5 ) );
    // valueChange = lightnessScale * pow((double)(out.i), lightnessExponent);
    let value = rend.args.cexp * ((data.s / rend.args.limit).powf(rend.args.h) * 360.0).powf(1.5);
    let t = data.s.powf(rend.args.h2);
    let h = (value * 1.0) % 360.0;
    // let s = 1.0;
    // let v = 1.0 - (t / rend.args.limit).powf(0.5);

    let val = (PI * (1.0 - (t / rend.args.limit).powf(0.5))).cos();
    let val = 1.0 - (val * val);

    let c = 28.0 + (74.0 * val);
    let l = 74.0 - (74.0 * val);

    let color = CIELch::new([l, c, h, 1.0]);
    CIELab::from_color(color)
    // CIELab::from_color(colortypes::Rgb::from_color(Hsv::new([h, s, v, 1.0])))
}

fn fade_mc_bw(rend: &Renderer, data: RenderData) -> Color<CIELab, D65> {
    // v = range1.mapExp( out.i, range2, exponent);
    let value = ((data.s / rend.args.limit).powf(rend.args.h) * 360.0)
        .powf(rend.args.cexp)
        .powf(1.5);
    let t = data.s.powf(rend.args.h2);
    let h = (value * 1.0) % 360.0;
    // let s = 1.0;
    // let v = 1.0 - (t / rend.args.limit).powf(0.5);

    let val = (PI * (1.0 - (t / rend.args.limit).powf(0.5))).cos();
    let val = 1.0 - (val * val);

    let c = 28.0 + (74.0 * val);
    let l = 74.0 - (74.0 * val);

    let color = CIELch::new([l, c, h, 1.0]);
    CIELab::from_color(color)
    // CIELab::from_color(colortypes::Rgb::from_color(Hsv::new([h, s, v, 1.0])))
}

fn paper_cut(rend: &Renderer, data: RenderData) -> Color<CIELab, D65> {
    let f = 1.0 + (((rend.args.bail).ln() / (data.lastz.norm_sqr()).ln()).log2());

    let val = ((data.der2.arg() + (PI / 2.0)) / PI)
        * ((data.der.arg() + (PI / 2.0)) / PI)
        * ((data.z.arg() + (PI / 2.0)) / PI);

    let val2 = ((data.lastder2.arg() + (PI / 2.0)) / PI)
        * ((data.lastder.arg() + (PI / 2.0)) / PI)
        * ((data.lastz.arg() + (PI / 2.0)) / PI);

    let value = (val * f) + (val2 * (1.0 - f));

    CIELab::new([
        100.0 * (value * (data.s / rend.args.limit).powf(rend.args.cexp)),
        10.0 * (value - 0.5),
        10.0 * (value - 0.5),
        1.0,
    ])
}

fn paper_cut_bw(rend: &Renderer, data: RenderData) -> Color<CIELab, D65> {
    let f = 1.0 + (((rend.args.bail).ln() / (data.lastz.norm_sqr()).ln()).log2());

    let val = ((data.der2.arg() + (PI / 2.0)) / PI)
        * ((data.der.arg() + (PI / 2.0)) / PI)
        * ((data.z.arg() + (PI / 2.0)) / PI);

    let val2 = ((data.lastder2.arg() + (PI / 2.0)) / PI)
        * ((data.lastder.arg() + (PI / 2.0)) / PI)
        * ((data.lastz.arg() + (PI / 2.0)) / PI);

    let value = (val * f) + (val2 * (1.0 - f));

    CIELab::new([
        2.0 * value * (data.s / rend.args.limit).powf(rend.args.cexp),
        0.0,
        0.0,
        1.0,
    ])
}

fn stripe(rend: &Renderer, data: RenderData) -> Color<CIELab, D65> {
    // Triangle inequality average
    let last_orbit = 0.5 + 0.5 * (3.0 * data.lastz.arg()).sin();
    // let last_orbit = idk(data.lastz);
    let f = 1.0 + (((rend.args.bail).ln() / (data.lastz.norm_sqr()).ln()).log2());
    let v = 1.0
        - ((f * (data.sum / data.i).powf(1.0 / PI))
            + (1.0 - f) * ((data.sum - last_orbit) / (data.i - 1.0)).powf(1.0 / PI));

    let val = (PI * v).cos();
    let val = 1.0 - (val * val);

    let c = 28.0 + (74.0 * val);
    let l = 74.0 - (74.0 * val);
    let h = (((data.s / rend.args.limit).powf(rend.args.cexp)) * 360.0).powf(1.5);

    CIELab::from_color(CIELch::new::<D65>([l, c, h % 360.0, 1.0]))
}

#[allow(dead_code, unused_variables)]
fn normal_map_dual(rend: &Renderer, data: RenderData) -> Color<CIELab, D65> {
    let lo = 0.5 * data.z.norm_sqr().ln();
    let u = data.z
        * data.der
        * ((1.0 + lo) * (data.der * data.der).conj() - lo * (data.z * data.der2).conj());
    let u = u / u.norm();

    let v = Cf64::new(
        (rend.angle.to_radians()).cos(),
        (rend.angle.to_radians()).sin(),
    );
    let norm_height = rend.args.h;

    let value = (u.re * v.re + u.im * v.im + norm_height) / (1.0 + norm_height);

    let v = Cf64::new(
        (rend.angle_offset.to_radians()).cos(),
        (rend.angle_offset.to_radians()).sin(),
    );
    let norm_height = rend.args.h;

    let value2 = (u.re * v.re + u.im * v.im + norm_height) / (1.0 + norm_height);

    let hue = (((data.s / rend.args.limit).powf(rend.args.cexp)) * 360.0).powf(1.5);
    let bound_a = value.powf(1.0);
    let bound_b = 1.0 - value2.powf(1.0);

    let val = (PI * (data.s / rend.args.limit).powf(rend.args.cexp)).cos();
    let val = 1.0 - (val * val);

    let scalar = 1.0 - (data.s / rend.args.limit);
    let hsv = CIELch::new::<D65>([
        74.0 - (74.0 * val),
        28.0 + (74.0 * (1.0 - val)),
        hue % 360.0,
        1.0,
    ]);
    // Color::new([hsv.0, hsv.1, hsv.2, 1.0], ColorType::HSLA).to_RGBA_HSLA()
    CIELab::from_color(hsv)
}

#[allow(dead_code, unused_variables)]
fn testingf(rend: &Renderer, data: RenderData) -> Color<CIELab, D65> {
    let theta = (data.z.arg() + PI) % (2.0 * PI);
    let r = theta.cos();
    let b = theta.sin();
    let g = (r + b) / 2.0;

    let theta = (data.lastz.arg() + PI) % (2.0 * PI);
    let r2 = theta.cos();
    let b2 = theta.sin();
    let g2 = (r2 + b2) / 2.0;
    let f = 1.0 + (((rend.args.bail).ln() / (data.lastz.norm_sqr()).ln()).log2());

    let ro = (r * f) + (r2 * (1.0 - f));
    let go = (g * f) + (g2 * (1.0 - f));
    let bo = (b * f) + (b2 * (1.0 - f));

    CIELab::from_color(Rgb::new([ro, go, bo, 1.0]))
}

#[allow(dead_code, unused_variables)]
fn image_mapping(rend: &Renderer, data: RenderData) -> Color<CIELab, D65> {
    let g_size = rend.args.media.len();

    let mut color = if rend.args.isgif && g_size > 0 {
        let (w, h) = rend.args.media[1].size;

        let width =
            ((data.z.im.atan2(data.z.re) + PI) / (PI * 2.0) * w as f64).round() as u32 % w as u32;
        let height = (h as f64 - 1.0f64)
            - ((data.z.norm() / rend.args.bail).log(rend.args.bail) * (h as f64 - 1.0f64)).floor();
        let mut height = ((height as u32) * 2) % h as u32;
        if data.i as u32 % 2 == 1 {
            height = (h as u32 - 1) - height;
        }

        let pixel = if let Some(px) =
            rend.args.media[rend.itime % g_size].get_pixel((width as usize, height as usize))
        {
            px.to_arr().0
        } else {
            [0.0, 0.0, 0.0, 1.0]
        };

        CIELch::from_color(colortypes::Rgb::new(pixel))
    } else {
        let (w, h) = (
            rend.texture.width() as usize,
            rend.texture.height() as usize,
        );

        let width =
            ((data.z.im.atan2(data.z.re) + PI) / (PI * 2.0) * w as f64).round() as u32 % w as u32;
        let height = (h as f64 - 1.0f64)
            - ((data.z.norm() / rend.args.bail).log(rend.args.bail) * (h as f64 - 1.0f64)).floor();
        let mut height = ((height as u32) * 2) % h as u32;
        if data.i as u32 % 2 == 1 {
            height = (h as u32 - 1) - height;
        }

        CIELch::from_color(colortypes::Rgb::new(
            rend.texture
                .get_pixel(width as u32, height as u32)
                .0
                .map(|x| x as f64 / u8::MAX as f64),
        ))
    };

    let lo = 0.5 * data.z.norm_sqr().ln();
    let u = data.z
        * data.der
        * ((1.0 + lo) * (data.der * data.der).conj() - lo * (data.z * data.der2).conj());
    let u = u / u.norm();

    let v = Cf64::new(
        (rend.angle.to_radians()).cos(),
        (rend.angle.to_radians()).sin(),
    );
    let norm_height = rend.args.h;

    let value = (1.0 - (data.s / rend.args.limit).powf(rend.args.cexp))
        * (u.re * v.re + u.im * v.im + norm_height)
        / (1.0 + norm_height);

    color.0 *= value.powf(1.0 / rend.args.cexp);
    // color.to_RGBA()
    CIELab::from_color(color)
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
fn mandelbrot(z: Cf64, c: Cf64, _j: Cf64) -> Cf64 {
    z.mul_add(z, c)
}

#[allow(dead_code, unused_variables)]
fn mandelbrot_j(z: Cf64, c: Cf64, j: Cf64) -> Cf64 {
    z.mul_add(z, j)
}

#[allow(dead_code, unused_variables)]
fn testing(z: Cf64, c: Cf64, j: Cf64) -> Cf64 {
    let mut nz = z * z * z * z + c;
    // nz = nz.powc(nz) + (nz / (c + Cf64::new(0.0000001, 0.0000001)));
    nz = Cf64::new(nz.re.abs(), nz.im);
    nz = nz * nz * nz + c;
    nz = Cf64::new(nz.re * nz.im, nz.im.abs());

    nz = nz * nz + c;
    nz = Cf64::new(nz.re.abs(), nz.im);
    nz + c
}

#[allow(dead_code, unused_variables)]
fn testing_j(z: Cf64, c: Cf64, j: Cf64) -> Cf64 {
    let mut nz = z * z * z * z + j;
    // nz = nz.powc(nz) + (nz / (c + Cf64::new(0.0000001, 0.0000001)));
    nz = Cf64::new(nz.re.abs(), nz.im);
    nz = nz * nz * nz + j;
    nz = Cf64::new(nz.re * nz.im, nz.im.abs());
    nz = nz * nz + j;
    nz = Cf64::new(nz.re.abs(), nz.im);
    nz + j
}

#[allow(dead_code, unused_variables)]
fn hybrid_lober(z: Cf64, c: Cf64, j: Cf64) -> Cf64 {
    let mut nz = z * z * z + c;
    // nz = nz.powc(nz) + (nz / (c + Cf64::new(0.0000001, 0.0000001)));
    nz = Cf64::new(nz.re.abs(), nz.im.abs());
    nz = nz * nz + c;
    nz = Cf64::new(nz.re.abs(), nz.im.abs());
    nz + c
}

#[allow(dead_code, unused_variables)]
fn hybrid_lober_j(z: Cf64, c: Cf64, j: Cf64) -> Cf64 {
    let mut nz = z * z * z + j;
    // nz = nz.powc(nz) + (nz / (c + Cf64::new(0.0000001, 0.0000001)));
    nz = Cf64::new(nz.re.abs(), nz.im.abs());
    nz = nz * nz + j;
    nz = Cf64::new(nz.re.abs(), nz.im.abs());
    nz + j
}

fn spade(z: Cf64, c: Cf64, _j: Cf64) -> Cf64 {
    z.powc(z) + (z / (c + (2.0 * f64::EPSILON)))
}

fn spade_j(z: Cf64, c: Cf64, _j: Cf64) -> Cf64 {
    z.powc(z) + (z / (c + (2.0 * f64::EPSILON)))
}

// w 1/3
// h 2/3
#[allow(dead_code)]
/// We derive Deserialize/Serialize so we can persist app state on shutdown.
#[cfg_attr(feature = "persistence", derive(serde::Deserialize, serde::Serialize))]
#[cfg_attr(feature = "persistence", serde(default))] // if we add new fields, give them default values when deserializing old state
pub struct FracgenGui<'a> {
    #[serde(skip)]
    args: Args,
    #[serde(skip)]
    scalar_value: f32,
    #[serde(skip)]
    owidth: f32,
    #[serde(skip)]
    renderer: Renderer,
    #[serde(skip)]
    hd_renderer: Renderer,
    #[serde(skip)]
    size: egui::Vec2,
    #[serde(skip)]
    samples_gif: usize,
    #[serde(skip)]
    nodes: Vec<NodeConstructor<'a>>,
    #[serde(skip)]
    max_samples: usize,
    #[serde(skip)]
    samples_per_pass: usize,
    #[serde(skip)]
    texture: Option<egui::TextureHandle>,
    #[serde(skip)]
    resize: (bool, f64, f64),
    #[serde(skip)]
    outstanding_size: f64,
    #[serde(skip)]
    last_origin: Cf64,
    #[serde(skip)]
    view_size: egui::Vec2,
    #[serde(skip)]
    isgif: bool,
    julia: bool,
    #[serde(skip)]
    current_iter: fn(Cf64, Cf64, Cf64) -> Cf64,
    #[serde(skip)]
    current_iter_j: fn(Cf64, Cf64, Cf64) -> Cf64,
    #[serde(skip)]
    iter_changed: bool,
    #[serde(skip)]
    keyp: [bool; 32],
}
fn reasonable_name(z: Cf64, c: Cf64, _j: Cf64) -> Cf64 {
    // der = der * ((2.0 * z) - (c / (z * z)));
    (c / z) + z * z
}
fn reasonable_name_j(z: Cf64, _c: Cf64, j: Cf64) -> Cf64 {
    // der = der * ((2.0 * z) - (c / (z * z)));
    (j / z) + z * z
}

fn prompt_number(
    bounds: RangeInclusive<u32>,
    message: &str,
    def: i32,
) -> Result<u32, Box<dyn Error>> {
    let stdin = stdin();
    let mut buffer = String::new();

    buffer.clear();
    // Keep prompting until the user passes a value within the bounds
    Ok(loop {
        print!("{}c", 27 as char);
        if message != "" {
            print!("{}c", 27 as char);
            if def >= 0 {
                println!(
                    "{} in the range [{}:{}] (default: {})",
                    message,
                    bounds.start(),
                    bounds.end() - 1,
                    def
                );
            } else {
                println!(
                    "{} in the range [{}:{}]",
                    message,
                    bounds.start(),
                    bounds.end() - 1
                );
            }
        }
        stdin.read_line(&mut buffer)?;
        print!("\r\u{8}");
        stdout().flush().unwrap();
        if let Ok(value) = buffer.trim().parse() {
            if bounds.contains(&value) {
                break value;
            }
        } else if def >= 0 {
            print!("\r\u{8}");
            println!("{}", &def);
            stdout().flush().unwrap();
            break def as u32;
        }
        buffer.clear();
    })
}

const SCALAR: f32 = 6.0;
#[allow(unused_variables)]
impl<'a> Default for FracgenGui<'a> {
    fn default() -> Self {
        Self {
            scalar_value: SCALAR,
            args: Args::new(SCALAR),
            renderer: Renderer::new(
                Args::new(SCALAR),
                Functs::new(
                    mandelbrot,
                    move |z, c| z,
                    identity,
                    normal_map_dual,
                    default_bail,
                ),
            ),
            hd_renderer: Renderer::new(
                Args::new(SCALAR),
                Functs::new(
                    mandelbrot,
                    move |z, c| z,
                    identity,
                    normal_map_dual,
                    default_bail,
                ),
            ),
            isgif: false,
            owidth: (1920.0f32 / SCALAR).floor(),
            size: egui::Vec2::new((1920.0f32 / SCALAR).floor(), (1080.0f32 / SCALAR).floor()),
            nodes: Vec::new(),
            max_samples: usize::MAX,
            samples_per_pass: 1,
            samples_gif: 32,
            texture: None,
            resize: (false, 0.0, 0.0),
            outstanding_size: 1.0,
            last_origin: Cf64::new(0.0, 0.0),
            view_size: egui::Vec2::new(0.0, 0.0),
            julia: false,
            current_iter: mandelbrot,
            current_iter_j: mandelbrot_j,
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
        // _frame.set_window_size(self.size);
        rayon::ThreadPoolBuilder::new()
            .num_threads(self.renderer.args.threads)
            .build_global()
            .unwrap();
        _frame.set_window_size(egui::Vec2 {
            x: 1920.0 / self.scalar_value,
            y: 1388.0 / self.scalar_value,
        });
        _frame.set_decorations(true);

        // self.texture = egui::TextureHandle.
        // Load previous app state (if any).
        // Note that you must enable the `persistence` feature for this to work.
        #[cfg(feature = "persistence")]
        if let Some(storage) = _storage {
            *self = epi::get_value(storage, epi::APP_KEY).unwrap_or_default();
            // *self =
            // self.renderer.args.cexp = epi::get_value(storage, "exp_for_c").unwrap_or_default();
        }
        // #[serde(rename(serialize = "exp_for_c"))]

        self.renderer.args.origin = Cf64::new(self.renderer.args.o_re, self.renderer.args.o_im);
        self.last_origin = self.renderer.args.origin;
        self.renderer.args.julia = Cf64::new(self.renderer.args.j_re, self.renderer.args.j_im);

        self.renderer.texture = image::open("test.jpg").unwrap().to_rgba8();
        // self.renderer.args.cexp = 0.8;
        self.renderer.render_samples(self.samples_per_pass, true);
        println!("Hanging until first render");
        #[allow(clippy::while_immutable_condition)]
        while !self.renderer.not_rendering {}
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
                        .pixels()
                        .map(|x| {
                            let nx = x.to_arr8().0;
                            epaint::color::Color32::from_rgb(nx[0], nx[1], nx[2])
                        })
                        .collect::<Vec<_>>(),
                },
            )
        });
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
        self.scalar_value = self.size.x / self.owidth;
        self.renderer.args.j_re = self.renderer.args.julia.re;
        self.renderer.args.j_im = self.renderer.args.julia.im;

        self.renderer.args.o_re = self.renderer.args.origin.re;
        self.renderer.args.o_im = self.renderer.args.origin.im;

        let input = ctx.input().clone();
        if self.keyp[4] && input.key_pressed(egui::Key::Enter) {
            if let Some(mut s) = FileStorage::from_app_name(self.name()) {
                println!("saving");
                self.save(&mut s);
            }
            // epi::set_value(, epi::APP_KEY, self);
            // self.save(storage)
            self.keyp[4] = false;
            println!("Waiting for frame!");
            #[allow(clippy::while_immutable_condition)]
            while !self.renderer.not_rendering {}
            println!("Will render now!");
            // self.hd_renderer.args
            self.hd_renderer.angle = self.renderer.angle;
            self.hd_renderer.angle_offset = self.renderer.angle_offset;
            self.hd_renderer.update_args(self.renderer.args.clone());
            self.hd_renderer.update_functs(self.renderer.functs.clone());
            self.hd_renderer.texture = self.renderer.texture.clone();
            println!("Pick a width!");
            let w = prompt_number(240..=32768, "Please choose a Width", -1).unwrap();
            let h = prompt_number(
                240..=32768,
                "Please choose a Height\nDefault conforms to viewport ratio!",
                (self.renderer.args.height as f64 * (w as f64 / self.renderer.args.width as f64))
                    as i32,
            )
            .unwrap();

            self.hd_renderer.resize(w as usize, h as usize);

            println!("Starting Rendering!");
            #[allow(clippy::while_immutable_condition)]
            while !self.hd_renderer.not_rendering {}
            if self.hd_renderer.args.isgif {
                self.hd_renderer.save_gif(self.samples_gif);
                println!("Rendered and Saved!");
            } else {
                // self.hd_renderer
                //     .render_samples(self.renderer.rendered_samples, true);
                // println!("Rendered!");
                // self.hd_renderer.process_image();
                // self.hd_rendere  r.image.save("QHD.png").unwrap();
                self.hd_renderer.rendered_samples = 0;
                self.hd_renderer
                    .render_samples(self.renderer.rendered_samples, true);
                self.hd_renderer.save_image();
                println!("Rendered and Saved!");
            }
        }
        if input.key_released(egui::Key::Enter) {
            self.keyp[4] = true;
        }
        // let files = ;
        if !input.raw.dropped_files.is_empty() {
            if let Some(fpath) = input.raw.dropped_files[0].path.clone() {
                if if let Some(ext) = fpath.extension() {
                    ext.eq_ignore_ascii_case(OsString::from("gif"))
                } else {
                    false
                } {
                    self.isgif = true;
                    self.renderer.args.isgif = true;
                    self.hd_renderer.args.isgif = true;

                    // Gif input
                    let mut decoder = gif::DecodeOptions::new();
                    decoder.set_color_output(gif::ColorOutput::RGBA);
                    // Decode gif
                    let mut gif = decoder.read_info(File::open(fpath).unwrap()).unwrap();

                    self.renderer.args.media = Vec::new();
                    self.hd_renderer.args.media = Vec::new();

                    // Move all frames into media
                    let mut i = 0;
                    while let Some(f) = gif.read_next_frame().unwrap() {
                        let (w, h) = (f.width, f.height);
                        self.renderer
                            .args
                            .media
                            .push(Image::new((w as usize, h as usize)));
                        // self.hd_renderer.args.media.push(Image::new((w as usize, h as usize)));
                        for (j, chunk) in f.buffer.chunks_exact(4).enumerate() {
                            let (x, y) = (j % w as usize, j / w as usize);
                            self.renderer.args.media[i].data[j].0 = chunk[0] as f64 / 255.0;
                            self.renderer.args.media[i].data[j].1 = chunk[1] as f64 / 255.0;
                            self.renderer.args.media[i].data[j].2 = chunk[2] as f64 / 255.0;
                            self.renderer.args.media[i].data[j].3 = chunk[3] as f64 / 255.0;
                        }
                        i += 1;
                    }
                    // println!("\nFRAMES: {}\n", self.renderer.args.media.len());
                    println!(
                        "\n{},{} over {}\n",
                        self.renderer.args.media[0].size.0,
                        self.renderer.args.media[0].size.1,
                        self.renderer.args.media.len()
                    );
                    // thread::sleep(Duration::from_secs(100));
                } else {
                    self.renderer.args.isgif = false;
                    self.hd_renderer.args.isgif = false;
                    // Image input
                    println!("{}", fpath.to_string_lossy());
                    self.renderer.texture = image::open(fpath).unwrap().to_rgba8();
                }
                self.isgif = false;
                self.renderer.rendered_samples = 0;
                self.renderer.render_samples(self.samples_per_pass, false);
            }
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
                            self.current_iter_j = mandelbrot_j;
                            self.iter_changed = true;
                        }
                        egui::Key::Num4 => {
                            self.current_iter = |z, c, j| z * z * z + c;
                            self.current_iter_j = |z, c, j| z * z * z + j;
                            self.iter_changed = true;
                        }
                        egui::Key::Num5 => {
                            self.current_iter =
                                |z, c, j| z.powc(z) + (z / (c + (2.0 * f64::EPSILON)));
                            self.current_iter_j =
                                |z, c, j| z.powc(z) + (z / (j + (2.0 * f64::EPSILON)));
                            self.iter_changed = true;
                        }
                        egui::Key::Num6 => {
                            self.current_iter = |z, c, j| z.powf(c.re) + c;
                            self.current_iter_j = |z, c, j| (z * c).powc(z / c) + (z / c);
                            self.iter_changed = true;
                        }
                        egui::Key::Num7 => {
                            self.current_iter = hybrid_lober;
                            self.current_iter_j = hybrid_lober_j;
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
                                tz = Cf64::new(tz.re.abs(), tz.im.abs());
                                tz = tz * tz + c;
                                tz = tz * tz + c;
                                tz = Cf64::new(tz.re.abs(), tz.im.abs());
                                tz = tz * tz + c;

                                tz
                            };
                            self.current_iter_j = |z, c, j| {
                                let mut tz = z * z + j;
                                tz = Cf64::new(tz.re.abs(), tz.im.abs());
                                tz = tz * tz + j;
                                tz = tz * tz + j;
                                tz = Cf64::new(tz.re.abs(), tz.im.abs());
                                tz = tz * tz + j;

                                tz
                            };
                            self.iter_changed = true;
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
                        egui::Key::Num0 => {
                            self.renderer.functs.color_funct = testingf;
                            println!("\nCOLORING\n");
                            pressed = true;
                        }
                        egui::Key::Num1 => {
                            self.renderer.functs.color_funct = coloring;
                            println!("\nCOLORING\n");
                            pressed = true;
                        }
                        egui::Key::Num2 => {
                            self.renderer.functs.color_funct = normal_map;
                            println!("\nNORMAL MAP\n");
                            pressed = true;
                        }
                        egui::Key::Num3 => {
                            self.renderer.functs.color_funct = image_mapping;
                            println!("\nIMAGE MAPPING\n");
                            pressed = true;
                        }
                        egui::Key::Num4 => {
                            self.renderer.functs.color_funct = stripe_normal;
                            println!("\nNORMAL MAPPED STRIPE\n");
                            pressed = true;
                        }
                        egui::Key::Num5 => {
                            self.renderer.functs.color_funct = fade_original;
                            println!("\nFADE\n");
                            pressed = true;
                        }
                        egui::Key::Num6 => {
                            self.renderer.functs.color_funct = fade_mc_bw;
                            println!("\nCYCLE_FADE\n");
                            pressed = true;
                        }
                        egui::Key::Num7 => {
                            self.renderer.functs.color_funct = paper_cut;
                            println!("\nPAPER\n");
                            pressed = true;
                        }
                        egui::Key::Num8 => {
                            self.renderer.functs.color_funct = paper_cut_bw;
                            println!("\nPAPER BW\n");
                            pressed = true;
                        }
                        egui::Key::Num9 => {
                            self.renderer.functs.color_funct = normal_map_dual;
                            println!("\nNORMAL MAP DUAL\n");
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
            let m = self.scalar_value
                * (m / self.renderer.args.width.min(self.renderer.args.height) as f32)
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
                (self.scalar_value * m.x) as i32,
                (self.scalar_value * m.y) as i32 - (16.0 * self.scalar_value) as i32,
                (self.renderer.args.width as f64 * self.outstanding_size) as i32
                    + (16.0 * self.scalar_value) as i32,
                (self.renderer.args.height as f64 * self.outstanding_size) as i32
                    + (32.0 * self.scalar_value) as i32
                    + (8.0 * self.scalar_value) as i32,
                self.renderer.args.zoom,
            ) + self.renderer.args.origin;
            self.renderer.args.julia = m;
            println!("{}", m);
            pressed = true;
        }
        // self.renderer.args.h += 0.1;

        if self.keyp[11] && input.key_pressed(egui::Key::Q) {
            self.renderer.args.h /= 1.5;
            println!("\nH: {}\n", self.renderer.args.h);
            self.keyp[11] = false;
            pressed = true;
        }
        if !self.keyp[11] && input.key_released(egui::Key::Q) {
            self.keyp[11] = true;
        }

        if self.keyp[12] && input.key_pressed(egui::Key::W) {
            self.renderer.args.h *= 1.5;
            println!("\nH2: {}\n", self.renderer.args.h);

            self.keyp[12] = false;
            pressed = true;
        }

        if !self.keyp[12] && input.key_released(egui::Key::W) {
            self.keyp[12] = true;
        }

        if self.keyp[13] && input.key_pressed(egui::Key::A) {
            if self.renderer.args.h2 < 60.0 {
                self.renderer.args.h2 += 6.0;
            }

            println!("\nH2: {}\n", self.renderer.args.h2);
            self.keyp[13] = false;
            pressed = true;
        }
        if !self.keyp[13] && input.key_released(egui::Key::A) {
            self.keyp[13] = true;
        }

        if self.keyp[14] && input.key_pressed(egui::Key::S) {
            if self.renderer.args.h2 > 6.0 {
                self.renderer.args.h2 -= 6.0;
            }
            println!("\nH2: {}\n", self.renderer.args.h2);
            self.keyp[14] = false;
            pressed = true;
        }

        if !self.keyp[14] && input.key_released(egui::Key::S) {
            self.keyp[14] = true;
        }

        if self.keyp[5] && input.key_pressed(egui::Key::Z) {
            self.renderer.args.derbail /= 1.5;
            println!("\nDBAIL: {}\n", self.renderer.args.derbail);
            self.keyp[5] = false;
            pressed = true;
        }
        if !self.keyp[5] && input.key_released(egui::Key::Z) {
            self.keyp[5] = true;
        }

        if self.keyp[10] && input.key_pressed(egui::Key::X) {
            self.renderer.args.derbail *= 1.5;
            println!("\nDBAIL: {}\n", self.renderer.args.derbail);
            self.keyp[10] = false;
            pressed = true;
        }
        if !self.keyp[10] && input.key_released(egui::Key::X) {
            self.keyp[10] = true;
        }

        if self.keyp[7] && input.key_pressed(egui::Key::O) {
            self.renderer.angle += (360. / 64.) % 360.0;
            println!(
                "\nANGLE:{}, OFFSET:{}\n",
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
                "\nANGLE:{}, OFFSET:{}\n",
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
                "\nANGLE:{}, OFFSET:{}\n",
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
                "\nANGLE:{}, OFFSET:{}\n",
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
                println!("\nBAIL: {}\n", self.renderer.args.bail);
                self.keyp[0] = false;
                pressed = true;
            }
            if !self.keyp[0] && input.key_released(egui::Key::ArrowUp) {
                self.keyp[0] = true;
            }

            if self.keyp[1] && input.key_pressed(egui::Key::ArrowDown) {
                self.renderer.args.bail /= 1.5;
                println!("\nBAIL: {}\n", self.renderer.args.bail);
                self.keyp[1] = false;
                pressed = true;
            }
            if !self.keyp[1] && input.key_released(egui::Key::ArrowDown) {
                self.keyp[1] = true;
            }

            if self.keyp[2] && input.key_pressed(egui::Key::ArrowLeft) {
                self.renderer.args.cexp *= scalar_mult * 1.1;
                println!("\nCEXP: {}\n", self.renderer.args.cexp);
                self.keyp[2] = false;
                pressed = true;
            }
            if !self.keyp[2] && input.key_released(egui::Key::ArrowLeft) {
                self.keyp[2] = true;
            }
            if self.keyp[3] && input.key_pressed(egui::Key::ArrowRight) {
                self.renderer.args.cexp /= scalar_mult * 1.1;
                println!("\nCEXP: {}\n", self.renderer.args.cexp);
                self.keyp[3] = false;
                pressed = true;
            }
            if !self.keyp[3] && input.key_released(egui::Key::ArrowRight) {
                self.keyp[3] = true;
            }
        } else {
            if self.keyp[0] && input.key_pressed(egui::Key::ArrowUp) {
                self.renderer.args.sampled *= 1.5;
                println!("\nSAMPLE DISTANCE: {}\n", self.renderer.args.sampled);
                self.keyp[0] = false;
                pressed = true;
            }
            if !self.keyp[0] && input.key_released(egui::Key::ArrowUp) {
                self.keyp[0] = true;
            }
            if self.keyp[1] && input.key_pressed(egui::Key::ArrowDown) {
                self.renderer.args.sampled /= 1.5;
                println!("\nSAMPLE DISTANCE: {}\n", self.renderer.args.sampled);
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
                println!("\nLIMIT: {}\n", self.renderer.args.limit);
                self.keyp[2] = false;
                pressed = true;
            }
            if !self.keyp[2] && input.key_released(egui::Key::ArrowLeft) {
                self.keyp[2] = true;
            }
            if self.keyp[3] && input.key_pressed(egui::Key::ArrowRight) {
                self.renderer.args.limit *= scalar_mult * 1.1;
                println!("\nLIMIT: {}\n", self.renderer.args.limit);
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

        if self.iter_changed {
            if self.julia {
                self.renderer.functs.iter_funct = self.current_iter_j;
            } else {
                self.renderer.functs.iter_funct = self.current_iter;
            }
            self.iter_changed = false;
            pressed = true;
        }

        if pressed || self.isgif {
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
            let (w, h) = (self.renderer.image.width(), self.renderer.image.height());
            if self.renderer.rendered_samples < self.max_samples && self.renderer.not_rendering {
                self.renderer.process_image();
                self.texture = None;
                texture = self.texture.get_or_insert_with(|| {
                    // let (w, h) = (self.renderer.image.width(), self.renderer.image.height());
                    // Load the texture only once.
                    ctx.load_texture(
                        "my-image",
                        epaint::ColorImage {
                            size: [(w).try_into().unwrap(), (h).try_into().unwrap()],
                            pixels: self
                                .renderer
                                .image
                                .pixels()
                                .map(|x| {
                                    let nx = x.to_arr8().0;
                                    epaint::color::Color32::from_rgb(nx[0], nx[1], nx[2])
                                })
                                .collect::<Vec<_>>(),
                        },
                    )
                });

                // println!("Rendering Pass {}", self.renderer.rendered_samples / self.samples_per_pass);
                self.renderer.render_samples(self.samples_per_pass, false);
            }
            ui.with_layout(egui::Layout::top_down_justified(egui::Align::LEFT), |ui| {
                // let w = ui.available_width();
                // let h = self.renderer.args.height as f64 * (w as f64 / self.renderer.args.width as f64);
                self.view_size = egui::Vec2::new(ui.available_width(), ui.available_height());
                let (rw, rh) = (self.renderer.image.width(), self.renderer.image.height());

                let w = ui.available_width() * 1.0;
                let h = rh as f64 * (w as f64 / rw as f64);
                self.outstanding_size = w as f64 / rw as f64;
                ui.image(texture, egui::Vec2::new(w as f32, h as f32));
                if w != self.args.width as f32 || h != self.args.height as f64 {
                    self.resize = (true, w as f64, h as f64);
                }
            });
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
