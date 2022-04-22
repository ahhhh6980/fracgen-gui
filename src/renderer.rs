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
#![allow(incomplete_features, unused_imports, dead_code)]
use image::{DynamicImage, ImageBuffer, Rgba};
use linya::{Bar, Progress};
use num::complex::Complex;
use rand::Rng;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use std::{
    path,
    sync::Mutex,
    time::{Instant, SystemTime},
};
type Cf64 = Complex<f64>;
type Img8 = ImageBuffer<Rgba<u8>, Vec<u8>>;
use crate::color::{Color, ColorType};

pub struct RenderData {
    pub i: f64,
    pub s: f64,
    pub z: Cf64,
    pub der: Cf64,
    pub orbit: Vec<Cf64>,
}

// type Cf64 = Cf64;

#[derive(Debug, Clone)]
// #[clap(author, version, about, long_about = None)]
pub struct Args {
    // #[clap(short, long, default_value = "1920")]
    pub width: i32,

    // #[clap(short, long, default_value = "1680")]
    pub height: i32,

    // #[clap(short, long, default_value = "mandel")]
    pub name: String,

    // #[clap(short, long, default_value_t=((num_cpus::get() as f64) * 0.75).ceil() as usize)]
    pub threads: usize,

    // #[clap(short, long, default_value_t=Complex::<f64>::new(-0.75,0.0))]
    pub origin: Cf64,

    // #[clap(short, long, default_value_t=Complex::<f64>::new(0.0,0.0))]
    pub z_init: Cf64,
    pub julia: Cf64,

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
    pub derbail: f64,
    // #[clap(short, long, default_value = "2.0")]
    pub cexp: f64,

    // #[clap(short, long, default_value = "0,0,0,255")]
    pub set_color: Color,
}

impl Args {
    pub fn new() -> Args {
        Args {
            width: 960 / 2,
            height: 840 / 2,
            name: String::from("mandel"),
            threads: ((num_cpus::get() as f64) * 0.70).ceil() as usize,
            origin: Complex::<f64>::new(-0.75, 0.0),
            z_init: Complex::<f64>::new(0.0, 0.0),
            julia: Complex::<f64>::new(-0.596202123, -0.078088350),
            zoom: 0.7,
            samples: 1,
            sampled: 1.0,
            limit: 1024.0,
            bail: 64.0,
            derbail: 2.0f64.powf(24.0),
            cexp: 1.0,
            set_color: Color::new([0.0, 0.0, 0.0, 1.0], ColorType::RGBA),
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

#[derive(Clone)]
pub struct Functs {
    pub iter_funct: fn(Cf64, Cf64, Cf64) -> Cf64,
    pub init_funct: fn(Cf64, Cf64) -> Cf64,
    pub cmap_funct: fn(&Renderer, Cf64) -> Cf64,
    pub color_funct: fn(&Renderer, RenderData) -> Color,
    pub conditional: fn(&Renderer, Cf64, Cf64, Cf64, Cf64) -> bool,
}

impl Functs {
    pub fn new(
        a: fn(Cf64, Cf64, Cf64) -> Cf64,
        b: fn(Cf64, Cf64) -> Cf64,
        c: fn(&Renderer, Cf64) -> Cf64,
        d: fn(&Renderer, RenderData) -> Color,
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
pub struct Renderer {
    pub args: Args,
    pub width: i32,
    pub height: i32,
    pub functs: Functs,
    pub image: Img8,
    pub raw: Vec<Vec<Color>>,
    pub rendered_samples: usize,
    pub not_rendering: bool,
    pub texture: Img8,
    pub angle: f64,
    pub angle_offset: f64,
}

impl Renderer {
    pub fn new(args: Args, functs: Functs) -> Renderer {
        Renderer {
            args: args.clone(),
            width: args.width,
            height: args.height,
            functs: functs,
            image: Img8::new(args.width as u32, args.height as u32),
            raw: vec![
                vec![Color::new([0f64; 4], ColorType::SRGBA); args.width as usize];
                args.height as usize
            ],
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
        self.image = Img8::new(w as u32, h as u32);
        self.raw = vec![vec![Color::new([0f64; 4], ColorType::SRGBA); w]; h];
        self.rendered_samples = 0;
    }

    pub fn pixel(&self, i: i32, samples: usize) -> Color {
        let mut out = Color::new([0.0; 4], ColorType::SRGBA);
        let d: Cf64 = normalize_coords(1, 1, self.width, self.height, self.args.zoom)
            - normalize_coords(0, 0, self.width, self.height, self.args.zoom);
        let mut rng = rand::thread_rng();
        for _ in 0..samples {
            let mut c = normalize_coords(
                i / self.height,
                i % self.height,
                self.width,
                self.height,
                self.args.zoom,
            ) + self.args.origin;
            c.re += d.re * (rng.gen_range(-1.0..1.0) / self.args.sampled);
            c.im += d.im * (rng.gen_range(-1.0..1.0) / self.args.sampled);
            c = (self.functs.cmap_funct)(self, c);
            // let c = c);
            let mut z = (self.functs.init_funct)(self.args.z_init, c);
            let mut i = 0.0;
            let mut s = 0.0;
            let mut der = Cf64::new(1.0, 0.0);
            let mut tot_der = Cf64::new(1.0, 0.0);
            let dc = Cf64::new(1.0, 0.0);
            // let mut series: Vec<Cf64> = Vec::new();
            let mut test = z;
            // let test2 = self.args.limit.sqrt();
            let mut old = z;
            let chk = d.re.min(d.im) * 0.5;

            let mut period = 1;
            let compute = true;

            let mut orbit: Vec<Cf64> = Vec::new();

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
            while compute
                && ((self.functs.conditional)(&self, z, c, der, tot_der) && i < self.args.limit)
            {
                // let t = z.re * z.re - z.im * z.im + c.im;
                // series.push(z);
                orbit.push(z);
                tot_der += der;
                der = (der * 2.0 * z) + dc;
                // der = ((self.functs.iter_funct)(
                //     z + Cf64::new(0., 0.000000000001),
                //     c,
                //     (self.functs.cmap_funct)(self, self.args.julia),
                // ) - (self.functs.iter_funct)(
                //     z,
                //     c,
                //     (self.functs.cmap_funct)(self, self.args.julia),
                // )) / Cf64::new(0., 0.000000000001);
                z = (self.functs.iter_funct)(z, c, (self.functs.cmap_funct)(self, self.args.julia));
                i += 1.0;
                s += (-(abs(z + 1.0))).exp();

                // let chk2 = z / 2.0 - z * z / 4.0;
                // if (chk2 + 0.765).norm_sqr() < 0.245 {
                //     s = s + ((s + self.args.limit) / 4.0);
                //     i = i + ((i + self.args.limit) / 4.0);
                // }

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
                if period > 20 {
                    period = 0;
                    old = z;
                }
                test += z;
            }
            let mut color = (self.functs.color_funct)(
                &self,
                RenderData {
                    i,
                    s,
                    z,
                    der,
                    orbit,
                },
            );

            color = color.to_sRGBA();
            if i < self.args.limit {
                out = out + (color * color);
            } else if compute == true {
                out = out + (self.args.set_color * self.args.set_color);
            } else {
                let tcolor = Color::new([0.0, 0.0, 0.0, 1.0], ColorType::RGBA).to_sRGBA();
                out = out + (tcolor * tcolor);
            }
        }
        out
    }

    pub fn render_samples(&mut self, samples: usize, progress: bool) {
        let now = SystemTime::now();
        self.not_rendering = false;
        let out: Vec<Color>;
        if progress {
            let progress = Mutex::new(Progress::new());
            let bar: Bar = progress
                .lock()
                .unwrap()
                .bar((self.width * self.height) as usize, "");
            out = (0..(self.width * self.height))
                .into_par_iter()
                .map(|i| {
                    progress.lock().unwrap().inc_and_draw(&bar, 1);
                    Renderer::pixel(self, i as i32, samples)
                })
                .collect();
        } else {
            out = (0..(self.width * self.height))
                .into_par_iter()
                .map(|i| Renderer::pixel(self, i as i32, samples))
                .collect();
        }

        for (i, e) in out.iter().enumerate() {
            let (x, y) = (
                (i as i32 / (self.height)) as u32,
                (i as i32 % (self.height)) as u32,
            );
            if (y as i32) < self.height {
                if self.rendered_samples > 0 {
                    self.raw[y as usize][x as usize] = self.raw[y as usize][x as usize] + *e;
                } else {
                    self.raw[y as usize][x as usize] = *e;
                }
            }
        }
        println!("{:4.4}", now.elapsed().unwrap().as_secs_f32());
        self.rendered_samples += samples;
        self.not_rendering = true;
    }

    pub fn process_image(&mut self) {
        for i in 0..(self.width * self.height) {
            let (x, y) = (
                (i as i32 / (self.height)) as u32,
                (i as i32 % (self.height)) as u32,
            );
            if (y as i32) < self.height {
                let e = self.raw[y as usize][x as usize] / self.rendered_samples as f64;
                self.image.put_pixel(
                    x,
                    y,
                    Rgba::from(
                        e.to_RGBA()
                            .to_arr()
                            .map(|v| (v.sqrt() * u8::MAX as f64) as u8),
                    ),
                );
            }
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
