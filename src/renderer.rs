#![allow(incomplete_features, unused_imports, dead_code)]
use chrono::prelude::*;
use gif::{Encoder, Frame, Repeat};
use std::error::Error;
use std::fs::{self, File};
use std::io::{stdin, stdout, BufWriter, Write};
use std::ops::{Range, RangeInclusive};
use std::path::PathBuf;
use std::process::exit;
use std::time::Duration;
use std::{borrow::Cow, thread};
// fracgen
// Main File
// (C) 2022 by Jacob (ahhhh6980@gmail.com)

// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.
use colortypes::{CIELab, CIELch, Color, FromColorType, Hsv, Image, Rgb, Srgb, Xyz, Yxy, D65};
use image::{DynamicImage, EncodableLayout, ImageBuffer, Rgba};
use linya::{Bar, Progress};
use num::{complex::Complex, traits::MulAdd};
use rand::Rng;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use std::{
    f64::consts::PI,
    path,
    sync::Mutex,
    time::{Instant, SystemTime},
};
type Cf64 = Complex<f64>;
type Img8 = ImageBuffer<Rgba<u8>, Vec<u8>>;
// use crate::color::{Color, ColorType};

use colortypes::impl_conversion;

// impl_conversion!(CIELab, CIELch, |color| )

pub struct RenderData {
    pub i: f64,
    pub s: f64,
    pub z: Cf64,
    pub der: Cf64,
    pub der2: Cf64,
    pub sum: f64,
    pub lastz: Cf64,
    pub lastder: Cf64,
    pub lastder2: Cf64,
}
// type Cf64 = Cf64;

#[cfg_attr(feature = "persistence", derive(serde::Deserialize, serde::Serialize))]
#[cfg_attr(feature = "persistence", serde(default))]
#[derive(Debug, Clone)]
// #[clap(author, version, about, long_about = None)]
pub struct Args {
    #[serde(skip)]
    // #[clap(short, long, default_value = "1920")]
    pub width: i32,
    #[serde(skip)]
    // #[clap(short, long, default_value = "1680")]
    pub height: i32,
    #[serde(skip)]
    pub h: f64,
    #[serde(skip)]
    pub h2: f64,
    #[serde(skip)]
    pub isgif: bool,
    // #[clap(short, long, default_value = "mandel")]
    pub name: String,
    #[serde(skip)]
    // #[clap(short, long, default_value_t=((num_cpus::get() as f64) * 0.75).ceil() as usize)]
    pub threads: usize,
    #[serde(skip)]
    // #[clap(short, long, default_value_t=Complex::<f64>::new(-0.75,0.0))]
    pub origin: Cf64,
    #[serde(skip)]
    pub media: Vec<Image<Rgb, D65>>,
    pub o_re: f64,
    pub o_im: f64,

    #[serde(skip)]
    // #[clap(short, long, default_value_t=Complex::<f64>::new(0.0,0.0))]
    pub z_init: Cf64,
    #[serde(skip)]
    pub julia: Cf64,
    pub j_re: f64,
    pub j_im: f64,

    // #[clap(short, long, default_value = ".7")]
    pub zoom: f64,

    // #[clap(short, long, default_value = "8192")]
    pub samples: usize,

    // #[clap(short, long, default_value = "2.0")]
    pub sampled: f64,

    // #[clap(short, long, default_value = "64.0")]
    pub limit: f64,

    // #[clap(short, long, default_value = "16.0")]
    pub bail: f64,
    #[serde(skip)]
    pub derbail: f64,
    // #[clap(short, long, default_value = "2.0")]
    pub cexp: f64,

    // #[clap(short, long, default_value = "0,0,0,255")]
    #[serde(skip)]
    pub set_color: Color<Srgb, D65>,
}

impl Args {
    pub fn new(scalar: f32) -> Args {
        Args {
            media: Vec::new(),
            width: (1920.0f32 / scalar).floor() as i32,
            height: (1080.0f32 / scalar).floor() as i32,
            h: 5.0,
            h2: 12.0,
            isgif: false,
            name: String::from("mandel"),
            threads: (num_cpus::get() as f64).ceil() as usize,
            origin: Complex::<f64>::new(-0.75, 0.0),
            o_re: -0.75,
            o_im: 0.0,
            z_init: Complex::<f64>::new(0.0, 0.0),
            julia: Complex::<f64>::new(-0.596202123, -0.078088350),
            j_re: -0.596202123,
            j_im: -0.078088350,
            zoom: 0.7,
            samples: 1,
            sampled: 1.0,
            limit: 1024.0,
            bail: 16384.0,
            derbail: 10f64.powi(20),
            cexp: 0.5,
            set_color: Srgb::new([0.0, 0.0, 0.0, 1.0]),
        }
    }
}

fn abs(z: Cf64) -> f64 {
    z.re * z.re + z.im * z.im
}

pub fn normalize_coords(x: i32, y: i32, w: i32, h: i32, z: f64) -> Cf64 {
    let nx = 2.0 * (x as f64 / w as f64) - 1.0;
    let ny = 2.0 * (y as f64 / h as f64) - 1.0;
    Complex::new(nx / z, ny * (h as f64 / w as f64) / z)
}

#[cfg_attr(feature = "persistence", derive(serde::Deserialize, serde::Serialize))]
#[cfg_attr(feature = "persistence", serde(default))]
#[derive(Clone)]
pub struct Functs {
    #[serde(skip)]
    pub iter_funct: fn(Cf64, Cf64, Cf64) -> Cf64,
    #[serde(skip)]
    pub init_funct: fn(Cf64, Cf64) -> Cf64,
    #[serde(skip)]
    pub cmap_funct: fn(&Renderer, Cf64) -> Cf64,
    #[serde(skip)]
    pub color_funct: fn(&Renderer, RenderData) -> Color<CIELab, D65>,
    #[serde(skip)]
    pub conditional: fn(&Renderer, Cf64, Cf64, Cf64, Cf64) -> bool,
}

#[allow(dead_code, unused_variables)]
fn coloring(rend: &Renderer, data: RenderData) -> Color<CIELab, D65> {
    let c = (data.s / rend.args.limit).powf(rend.args.cexp);
    let hue = (c * 360.0).powf(1.5);
    // let new_color = colortypes::Rgb::new([64.0 / 255.0, 128.0 / 255.0, 255.0, 1.0]);
    let mut c = CIELab::from_color(CIELch::new::<D65>([
        74.0 - (74.0 * (1.0 - (data.s / rend.args.limit)).powi(16)),
        28.0 + (74.0 * c),
        hue % 360.0,
        1.0,
    ]));
    // let mut new_color =
    //     crate::colortypes::Rgb::from_color(crate::colortypes::Color::<
    //         CIELch,
    //         { crate::colortypes::RefWhite::D65 },
    //     >::new([50.0, 50.0, hue % 360.0, 1.0]));

    // Color::new(new_color.to_arr().0, ColorType::RGBA)new_color
    c
}
#[allow(dead_code, unused_variables)]
fn mandelbrot(z: Cf64, c: Cf64, _j: Cf64) -> Cf64 {
    z.mul_add(z, c)
    // let nz = Cf64::new(z.re.abs(), z.im.abs());
    // nz.mul_add(nz, c)
}

#[allow(dead_code, unused_variables)]
fn mandelbrot_j(z: Cf64, c: Cf64, j: Cf64) -> Cf64 {
    z.powc(z) + ((j + f64::EPSILON) / (z + f64::EPSILON))
}
#[allow(dead_code, unused_variables)]
fn default_bail(rend: &Renderer, z: Cf64, c: Cf64, der: Cf64, der_sum: Cf64) -> bool {
    z.norm() < rend.args.bail

    // (c - z).norm() * (z - der).norm() < rend.args.bail

    // z.norm_sqr() < rend.args.bail
}

fn idk(c: Cf64) -> f64 {
    // 0.5 + 0.5 * (3.0 * data.lastz.arg()).sin()
    // 0.5 + 0.5 * (3.0 * (1.0 / c).arg()).sin()
    // let nc = Cf64::new(
    //     if c.re == 0.0 { 1.0 } else { c.re.abs() },
    //     if c.im == 0.0 { 1.0 } else { c.im.abs() },
    // );
    // 0.5 + 0.5 * (10.0 * (nc.powc(1.0 / (nc * nc))).arg()).sin()
    0.5 + 0.5 * (3.0 * c.arg()).sin()
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

fn testing(rend: &Renderer, data: RenderData) -> Color<CIELab, D65> {
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
        (value - 0.5),
        (value - 0.5),
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
    let light_deg = rend.angle;
    let norm_height = 1.5;
    let light_vec = Cf64::new(
        ((light_deg * PI) / 180.0).cos(),
        ((light_deg * PI) / 180.0).sin(),
    );
    let ok_bail = 12376.0;
    let normal_vec = data.z / data.der;
    let normal_vec = normal_vec / normal_vec.norm(); // abs norm_vec
    let mut value = normal_vec.re.mul_add(
        light_vec.re,
        normal_vec.im.mul_add(light_vec.im, norm_height),
    ) / (1.0 + norm_height);
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
    let mut value2 = normal_vec.re.mul_add(
        light_vec.re,
        normal_vec.im.mul_add(light_vec.im, norm_height),
    ) / (1.0 + norm_height);
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
    /*
        ((-bound_b) + bound_a)
    ((-bound_b) + bound_a)
        */
    // let v = ((-bound_b) + bound_a);
    // let mut t = colortypes::Rgb::from_color(colortypes::CIELch::new([
    //     scalar * (74.0 + (-25.0 * v)),
    //     scalar * (28.0 + (25.0 * v)),
    //     hue % 360.0,
    //     1.0,
    // ]));

    let last_orbit = 0.5 + 0.5 * (3.0 * data.lastz.arg()).sin();
    let f = 1.0 + (((rend.args.bail).ln() / (data.lastz.norm_sqr()).ln()).log2());
    let v = (f * (data.sum / data.i)) + (1.0 - f) * ((data.sum - last_orbit) / (data.i - 1.0));

    let value = bound_a;
    let c = 28.0 + (74.0 * (0.5 + (0.5 * (2.0 * value * PI + PI))));
    let l = 74.0 - (74.0 * (0.5 + (0.5 * (2.0 * value * PI + PI))));

    // let c = (((((l / 50.0) * PI) + PI).cos() / 2.0) + 0.5) * 100.0;
    // let c = 43.89 - 10.8933 cos(0.125664 x) - 39.9 cos((π x)/50) + 13.6801 sin(0.125664 x) + 15;

    let mut t = CIELab::from_color(CIELch::new::<D65>([l, c, hue % 360.0, 1.0]));

    // t *= ((-bound_b) + bound_a);

    // fn f(c: Cf64) -> f64 {
    //     (0.5 * (3.0 * c.arg()).sin()) + 0.5
    // }

    // let [r, g, b, a] = t.to_arr().0;

    // Color::new(t.to_arr().0, ColorType::RGBA)
    t
}

fn refract(rend: &Renderer, data: RenderData) -> Color<CIELab, D65> {
    // let h = 1.0;
    // let s = data.der.norm();
    // let s = 2.0f64;
    // let angles = [data.der.re.atan2(data.z.re), data.der.im.atan2(data.z.im)];
    // let dist = [
    //     rend.args.h * (angles[0]).tan(),
    //     rend.args.h * (angles[1]).tan(),
    // ];
    // let ratio_x = dist[0] / rend.args.width as f64;
    // let ratio_y = dist[1] / rend.args.width as f64;

    // let c = Yxy::new::<D65>([ratio_x * 1.0, ratio_y, 0.5, 1.0]);

    fn rad_deg(v: f64) -> f64 {
        (v + PI).abs() * (180.0 * PI)
    }

    fn deg_rad(v: f64) -> f64 {
        // (v + PI).abs() * (180.0 * PI)
        (v.abs() * (PI / 180.0)) - PI
    }
    let f = 1.0 + (((rend.args.bail).ln() / (data.lastz.norm_sqr()).ln()).log2());

    let nz = data.z / (1.0 + data.z.norm());
    let nder = data.der / (1.0 + data.der.norm());

    let nzl = data.lastz / (1.0 + data.lastz.norm());
    let nderl = data.lastder / (1.0 + data.lastder.norm());

    let al = [
        (((rad_deg(nzl.re.atan2(nderl.re))).abs() * (PI / 180.0)) / 360.0).powf(1.224744871)
            * 360.0,
        (((rad_deg(nzl.im.atan2(nderl.im))).abs() * (PI / 180.0)) / 360.0).powf(1.224744871)
            * 360.0,
    ];

    let a = [
        (((rad_deg(nz.re.atan2(nder.re))).abs() * (PI / 180.0)) / 360.0).powf(1.224744871) * 360.0,
        (((rad_deg(nz.im.atan2(nder.im))).abs() * (PI / 180.0)) / 360.0).powf(1.224744871) * 360.0,
    ];

    let angle = (a[1] * f) + (al[1] * (1.0 - f));

    let v2 = ((a[0] * f) + (al[0] * (1.0 - f)) % 360.0) / 360.0;

    // let ratio_x = dists[0] / (rend.args.h * (rend.height as f64 / rend.width as f64));
    // let ratio_y = dists[1] / rend.args.h;

    // let angle = ((rad_deg(ratio_y.atan2(ratio_x)) / 360.0).powf(rend.args.cexp) * 360.0).powf(1.0);

    // let mut c = Yxy::new::<D65>([ratio_x * 1.0, ratio_y, 0.5, 1.0]);

    // let angle = (((nz.arg().cos().atan2(nder.arg()).abs() * (180.0 * PI)) / 360.0)
    //     .powf(rend.args.cexp)
    //     * 360.0)
    //     .powf(1.5);

    // let hue = 0.0;

    //  Normal mapped variant
    let lo = 0.5 * data.z.norm_sqr().ln();
    let u = data.z
        * data.der
        * ((1.0 + lo) * (data.der * data.der).conj() - lo * (data.z * data.der2).conj());
    let u = u / u.norm();

    let v = Cf64::new((deg_rad(rend.angle)).cos(), (deg_rad(rend.angle)).sin());
    let norm_height = 3.0;
    let t = (u.re * v.re + u.im * v.im + norm_height) / (1.0 + norm_height);

    // LCH Computation
    let v = (data.s / rend.args.limit).powf(0.4);
    let val = (PI * v).cos();
    let val = 1.0 - (val * val);

    let c = (28.0 + (74.0 * val)) * (1.0 - v2);
    let l = (74.0 - (74.0 * val)) * (1.0 - v2) * t;

    let mut c = CIELab::from_color(CIELch::new::<D65>([l, c, angle.powf(1.5) % 360.0, 1.0]));
    // let mut c = Srgb::from_color(Xyz::from_color(c));
    c
}

#[allow(dead_code, unused_variables)]
fn der_bail(rend: &Renderer, z: Cf64, c: Cf64, der: Cf64, der_sum: Cf64) -> bool {
    (der_sum * der_sum).norm_sqr() < rend.args.derbail
        && z.norm_sqr() * z.norm_sqr() < rend.args.bail
}
fn spade(z: Cf64, c: Cf64, _j: Cf64) -> Cf64 {
    z.powc(z) + (z / (c + (2.0 * f64::EPSILON)))
}

fn spade_j(z: Cf64, c: Cf64, _j: Cf64) -> Cf64 {
    z.powc(z) + (z / (c + (2.0 * f64::EPSILON)))
}

fn jksdhkjsda(z: Cf64, c: Cf64, _j: Cf64) -> Cf64 {
    // der = der * ((2.0 * z) - (c / (z * z)));
    (c / z) + z * z
}

impl Default for Functs {
    fn default() -> Functs {
        Functs::new(
            mandelbrot,
            move |z, _c| z,
            |_r, z| z,
            stripe_normal,
            default_bail,
        )
    }
}

impl Functs {
    pub fn new(
        a: fn(Cf64, Cf64, Cf64) -> Cf64,
        b: fn(Cf64, Cf64) -> Cf64,
        c: fn(&Renderer, Cf64) -> Cf64,
        d: fn(&Renderer, RenderData) -> Color<CIELab, D65>,
        e: fn(&Renderer, Cf64, Cf64, Cf64, Cf64) -> bool,
    ) -> Functs {
        Functs {
            iter_funct: a,
            init_funct: b,
            cmap_funct: c,
            color_funct: d,
            conditional: e,
        }
    }
}

#[cfg_attr(feature = "persistence", derive(serde::Deserialize, serde::Serialize))]
#[cfg_attr(feature = "persistence", serde(default))]
pub struct Renderer {
    #[serde(skip)]
    pub args: Args,
    #[serde(skip)]
    pub time: Instant,
    #[serde(skip)]
    pub itime: usize,
    pub width: i32,
    pub height: i32,
    #[serde(skip)]
    pub functs: Functs,
    #[serde(skip)]
    pub image: Image<Rgb, D65>,
    #[serde(skip)]
    pub raw: Image<CIELab, D65>,
    #[serde(skip)]
    pub raw_gif: Vec<Image<CIELab, D65>>,
    #[serde(skip)]
    pub rendered_samples: usize,
    #[serde(skip)]
    pub not_rendering: bool,
    #[serde(skip)]
    pub texture: Img8,
    pub angle: f64,
    pub angle_offset: f64,
}

fn list_options(input: &[u32]) -> Result<(), Box<dyn Error>> {
    // let inputs = fs::read_dir(&input)?;
    if input.is_empty() {
        println!("There are no files!");
    }
    for (i, thing) in input.iter().enumerate() {
        println!("{}: Render {} samples per frame", i, thing);
    }
    Ok(())
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

impl Renderer {
    pub fn new(args: Args, functs: Functs) -> Renderer {
        Renderer {
            itime: 0,
            time: Instant::now(),
            args: args.clone(),
            width: args.width,
            height: args.height,
            functs,
            image: Image::new((args.width as usize, args.height as usize)),
            raw: Image::new((args.width as usize, args.height as usize)),
            raw_gif: Vec::new(),
            rendered_samples: 0,
            not_rendering: true,
            texture: Img8::new(0, 0),
            angle: 90f64,
            angle_offset: 180f64,
        }
    }

    pub fn resize(&mut self, w: usize, h: usize) {
        self.args.width = w as i32;
        self.args.height = h as i32;
        self.width = w as i32;
        self.height = h as i32;
        self.image = Image::new((w, h));
        self.raw = Image::new((w, h));
        self.rendered_samples = 0;
    }

    pub fn pixel(&self, i: i32, samples: usize) -> Color<CIELab, D65> {
        let mut out = CIELab::new([0.0, 0.0, 0.0, 1.0]);
        let d: Cf64 = normalize_coords(1, 1, self.width, self.height, self.args.zoom)
            - normalize_coords(0, 0, self.width, self.height, self.args.zoom);
        let mut rng = rand::thread_rng();
        if self.rendered_samples > 0
            && self.image.data[i as usize] == Rgb::new([0.0, 0.0, 0.0, 1.0])
        {
            return CIELab::from_color(Xyz::from_color(self.args.set_color));
        }
        for _ in 0..samples {
            let mut c = normalize_coords(
                i % self.width,
                i / self.width,
                self.width,
                self.height,
                self.args.zoom,
            ) + self.args.origin;
            let (wx, wy) = ((rng.gen_range(-0.5..0.5)), (rng.gen_range(-0.5..0.5)));
            c.re += d.re * wx;
            c.im += d.im * wy;
            let weighting = (wx.sin() / wx) * (wy.sin() / wy);
            let wm = ((wx * wx) + (wy * wy)).sqrt().abs();
            let g_weighting = (-((2.0 * wm).powi(2))).exp()
                + ((1.0 / 24.0) * ((-((2.0 * wm).powi(2))).exp() / d.norm()));
            c = (self.functs.cmap_funct)(self, c);
            // let c = c);
            let mut z = (self.functs.init_funct)(self.args.z_init, c);
            let mut i = 0.0f64;
            let mut s = 0.0f64;
            let mut der = Cf64::new(1.0, 0.0);
            let mut der2 = Cf64::new(0.0, 0.0);

            let mut tot_der = Cf64::new(0.0, 0.0);
            let dc = Cf64::new(1.0, 0.0);
            // let mut series: Vec<Cf64> = Vec::new();
            // let mut test = z;
            // let test2 = self.args.limit.sqrt();
            let mut old = z;
            let chk = d.re.min(d.im);

            let mut period = 1;
            let compute = true;

            // let mut orbit: Vec<Cf64> = Vec::new();

            // let checkpoint_b = Cf64::new(0.0, 0.6);
            // let checkpoint_a = Cf64::new(0.0, 0.3);
            // if ((c + 0.0) / ((-c / 0.75) + 1.0)).re
            //     > ((Cf64::new(0.3, 0.0)) / ((-checkpoint_b / 0.75) + 1.0)).re
            // // && ((c + 0.0) / ((-c / 0.75) + 1.0)).re
            // //     < (checkpoint_a / ((-checkpoint_a / 0.75) + 1.0)).re
            // {
            //     compute = false;
            // }
            // if ((c + 0.0) / ((-c / 0.75) + 1.0)).re
            //     < ((Cf64::new(0.0, 1.0)) / ((-Cf64::new(0.0, -0.3) / 0.75) + 1.0)).re
            // {
            //     compute = false;
            // }
            // let a = Cf64::new(0.0, 0.33);
            // let t = 1.0 - (a / 0.75);
            // // let chk = -t * ((a / t) - 0.0);
            // let a = Cf64::new(0.0, 0.3);
            // let chk2 = -t * ((a / t) - 0.0);
            // // let chk2 = a / ((-a / 0.75) + 1.0);
            // let cm = (c + 0.0) / ((-c / 0.75) + 1.0);
            // if cm.im > chk2.im {
            //     compute = false;
            // }
            // let chk1 = (c + 2.5) / 2.0 - (c + 2.5) * (c + 2.5) / 4.0;
            // let chk2 = c / 2.0 - c * c / 4.0;
            // if (chk2 + 0.765).norm() < 0.245 {
            //     compute = false;
            // }
            let mut sum = 0.0;
            let mut lastz = z;
            let mut lastder = der;
            let mut lastder2 = der2;
            // let mut experiments = vec![Cf64::new(0.0, 0.0); 4];

            while compute
                && ((self.functs.conditional)(self, z, c, der2, tot_der) && i < self.args.limit)
            {
                // let t = z.re * z.re - z.im * z.im + c.im;
                // series.push(z);

                // orbit.push(z);

                tot_der += der;
                // let new_der2 =
                der2 = 2.0 * (der2 * z + (der * der));
                der = (der * 2.0 * z) + dc;
                z = (self.functs.iter_funct)(z, c, (self.functs.cmap_funct)(self, self.args.julia));

                // experiments[0] += (der / z).powf(s.ln());
                i += 1.0;
                s += (-(z.norm())).exp();
                if i > 0.0 {
                    sum += 0.5 + 0.5 * (3.0 * z.im.atan2(z.re)).sin();
                    // sum += idk(z);

                    lastz = z;
                    lastder = der;
                    lastder2 = der2;
                }
                // let dif = z - lastz;
                // if dif.re.abs() < chk && dif.im.abs() < chk {
                //     i = self.args.limit;
                //     s = self.args.limit;
                // }

                let dif = z - old;
                if dif.re.abs() < chk && dif.im.abs() < chk {
                    // i = self.args.limit;
                    i = self.args.limit;
                    // s = self.args.limit;
                    s = self.args.limit;
                }

                period += 1;
                if period > 42 {
                    period = 0;
                    old = z;
                }
                // test += z;
            }

            // color = color.to_sRGBA();
            // CIELab::new([weighting, weighting, weighting, 1.0]) *
            let temp = weighting;
            if i < self.args.limit {
                out += CIELab::new([temp, temp, temp, 1.0])
                    * CIELab::from_color(Xyz::from_color((self.functs.color_funct)(
                        self,
                        RenderData {
                            i,
                            s,
                            z,
                            der,
                            der2,
                            sum,
                            lastz,
                            lastder,
                            lastder2,
                        },
                    )));
            } else {
                out += CIELab::new([temp, temp, temp, 1.0])
                    * CIELab::from_color(Xyz::from_color(self.args.set_color));
            }
        }
        out
    }

    pub fn render_samples(&mut self, samples: usize, progress: bool) {
        // let now = SystemTime::now();
        self.itime = (self.time.elapsed().as_secs_f32() * (self.args.h2 as f32)) as usize;

        self.not_rendering = false;
        let out: Vec<Color<CIELab, D65>> = if progress {
            let progress = Mutex::new(Progress::new());
            let bar: Bar = progress
                .lock()
                .unwrap()
                .bar((self.raw.width() * self.raw.height()) as usize, "");
            (0..(self.raw.width() * self.raw.height()))
                .into_par_iter()
                .map(|i| {
                    progress.lock().unwrap().inc_and_draw(&bar, 1);
                    Renderer::pixel(self, i as i32, samples)
                })
                .collect()
        } else {
            (0..(self.raw.width() * self.raw.height()))
                .into_par_iter()
                .map(|i| Renderer::pixel(self, i as i32, samples))
                .collect()
        };
        // println!("{} VS {}", self.width, self.raw.width());
        // thread::sleep(Duration::new(100, 0));
        let w = self.raw.width();

        if self.rendered_samples == 0 || self.args.isgif {
            for (i, e) in out.iter().enumerate() {
                let (x, y) = (i % (self.raw.width()), i / (self.raw.width()));
                self.raw.data[x + (y * w)] = *e;
            }
        } else {
            for (i, e) in out.iter().enumerate() {
                let (x, y) = (i % self.raw.width(), i / self.raw.width());
                self.raw.data[x + (y * w)] += *e;

                // self.raw.data[x + (y * w)] = *e;
            }
        }
        self.rendered_samples += samples;

        self.not_rendering = true;
    }

    pub fn render_samples_at(&mut self, samples: usize, progress: bool, itime: usize) {
        // let now = SystemTime::now();
        self.itime = itime;

        self.not_rendering = false;
        let out: Vec<Color<CIELab, D65>> = if progress {
            let progress = Mutex::new(Progress::new());
            let bar: Bar = progress
                .lock()
                .unwrap()
                .bar((self.raw.width() * self.raw.height()) as usize, "");
            (0..(self.raw.width() * self.raw.height()))
                .into_par_iter()
                .map(|i| {
                    progress.lock().unwrap().inc_and_draw(&bar, 1);
                    Renderer::pixel(self, i as i32, samples)
                })
                .collect()
        } else {
            (0..(self.raw.width() * self.raw.height()))
                .into_par_iter()
                .map(|i| Renderer::pixel(self, i as i32, samples))
                .collect()
        };
        // println!("{} VS {}", self.width, self.raw.width());
        // thread::sleep(Duration::new(100, 0));
        let w = self.raw.width();

        for (i, e) in out.iter().enumerate() {
            let (x, y) = (i % self.raw.width(), i / self.raw.width());
            self.raw.data[x + (y * w)] = *e;
            self.raw.data[x + (y * w)].3 = 1.0;
            // self.raw.data[x + (y * w)] = *e;
        }

        self.rendered_samples += samples;
        self.not_rendering = true;
    }

    pub fn process_image(&mut self) {
        for (i, px) in self.raw.pixels_mut().enumerate() {
            self.image.data[i] =
                colortypes::Rgb::from_color(*px / self.rendered_samples as f64).clamp_to_gamut();
        }
    }

    pub fn save_image(&mut self) {
        let nt = NaiveDateTime::from_timestamp(Utc::now().timestamp(), 0);
        let dt: DateTime<Utc> = DateTime::from_utc(nt, Utc);
        let res = dt.format("%Y-%m-%d %H:%M:%S");

        fs::create_dir(format!("out/{}/", res)).unwrap();
        let txt = File::create(format!("out/{}/params.txt", res)).unwrap();
        // txt.;
        fs::write(
            format!("out/{}/params.txt", res),
            format!(
                "\n
        Origin: {:0<36.32}, {:0<36.32}\n
        Julia: {:0<36.32}, {:0<36.32}\n\n
        Zoom: {:0<36.32}\n
        Bailout: {:0<36.32}\n
        Limit: {:0<36.32}\n
        Cexp:{:0<16.32}
        ",
                self.args.origin.re,
                self.args.origin.im,
                self.args.julia.re,
                self.args.julia.im,
                self.args.zoom,
                self.args.bail,
                self.args.limit,
                self.args.cexp,
            ),
        )
        .unwrap();
        let file = File::create(format!("out/{}/img.png", res)).unwrap();
        let writ = &mut BufWriter::new(file);
        let mut encoder = png::Encoder::new(writ, self.width as u32, self.height as u32);
        encoder.set_color(png::ColorType::Rgba);
        encoder.set_depth(png::BitDepth::Eight);
        encoder.set_trns(vec![0xFFu8, 0xFFu8, 0xFFu8, 0xFFu8]);
        encoder.set_source_gamma(png::ScaledFloat::new(1.0 / 2.2));
        let source_chromaticities = png::SourceChromaticities::new(
            // Using unscaled instantiation here
            (0.31270, 0.32900),
            (0.64000, 0.33000),
            (0.30000, 0.60000),
            (0.15000, 0.06000),
        );
        encoder.set_source_chromaticities(source_chromaticities);
        let mut writer = encoder.write_header().unwrap();
        let now = Instant::now();
        let mut test = writer.stream_writer_with_size(128).unwrap();
        let mut wrote = 0;

        println!("{}", self.not_rendering);
        for px in self.raw.pixels() {
            let c = colortypes::Rgb::from_color(*px / self.rendered_samples as f64)
                .clamp_to_gamut()
                .to_arr8()
                .0;
            wrote += test.write(c.as_slice()).unwrap();
        }
        println!(
            "Wrote {} bytes in {:4.4} seconds",
            wrote,
            now.elapsed().as_secs_f32()
        );
    }

    pub fn save_gif(&mut self, samples: usize) {
        let nt = NaiveDateTime::from_timestamp(Utc::now().timestamp(), 0);
        let dt: DateTime<Utc> = DateTime::from_utc(nt, Utc);
        let res = dt.format("%Y-%m-%d %H:%M:%S");
        fs::create_dir(format!("gif/{}/", res)).unwrap();
        let txt = File::create(format!("gif/{}/params.txt", res)).unwrap();
        // txt.;
        fs::write(
            format!("gif/{}/params.txt", res),
            format!(
                "\n
        Origin: {:0<36.32}, {:0<36.32}\n
        Julia: {:0<36.32}, {:0<36.32}\n\n
        Zoom: {:0<36.32}\n
        Bailout: {:0<36.32}\n
        Limit: {:0<36.32}\n
        Cexp:{:0<16.32}
        ",
                self.args.origin.re,
                self.args.origin.im,
                self.args.julia.re,
                self.args.julia.im,
                self.args.zoom,
                self.args.bail,
                self.args.limit,
                self.args.cexp,
            ),
        )
        .unwrap();
        let samplec = prompt_number(
            1..=8192,
            "Enter how many samples to process per pixel for each frame:",
            32,
        )
        .unwrap() as usize;
        for frame_index in 0..self.args.media.len() {
            let file = File::create(format!("gif/{}/{}.png", res, frame_index)).unwrap();
            let writ = &mut BufWriter::new(file);
            let mut encoder = png::Encoder::new(writ, self.width as u32, self.height as u32);
            encoder.set_color(png::ColorType::Rgba);
            encoder.set_depth(png::BitDepth::Eight);
            encoder.set_trns(vec![0xFFu8, 0xFFu8, 0xFFu8, 0xFFu8]);
            encoder.set_source_gamma(png::ScaledFloat::new(1.0 / 2.2));
            let source_chromaticities = png::SourceChromaticities::new(
                // Using unscaled instantiation here
                (0.31270, 0.32900),
                (0.64000, 0.33000),
                (0.30000, 0.60000),
                (0.15000, 0.06000),
            );
            encoder.set_source_chromaticities(source_chromaticities);
            let mut writer = encoder.write_header().unwrap();
            let now = Instant::now();

            let mut test = writer.stream_writer_with_size(32).unwrap();
            let mut wrote = 0;
            // writer.write_image_data();
            println!("Frame {}", frame_index);
            // self.raw =
            self.raw = Image::new((self.raw.width(), self.raw.height()));
            // self.render_samples_at(1, true, frame_index);
            self.rendered_samples = 0;
            self.render_samples_at(samplec, true, frame_index);
            // self.process_image
            // while self.not_rendering == true {}
            println!("{}", self.not_rendering);
            for px in self.raw.pixels() {
                let mut c = colortypes::Rgb::from_color(*px / self.rendered_samples as f64)
                    .clamp_to_gamut()
                    .to_arr8()
                    .0;
                c[3] = 255;
                wrote += test.write(c.as_slice()).unwrap();
            }
            println!(
                "Wrote {} bytes in {:4.4} seconds",
                wrote,
                now.elapsed().as_secs_f32()
            );
        }
    }

    pub fn update_args(&mut self, args: Args) {
        self.args = args.clone();
        self.width = args.width;
        self.height = args.height;
    }

    pub fn update_functs(&mut self, functs: Functs) {
        self.functs = functs;
    }
}
