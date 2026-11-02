//! Rust side of ranga's differential oracle. See scripts/oracle/README.md.
//!
//! Prints one line per value: `group<TAB>key<TAB>type<TAB>value`, where type is
//! `u8` (hex bytes), `f32` / `f64` (hex bit patterns), `code` (error code as the
//! Cyrius RG_ERR_* integer) or `x` (an f32 bit pattern found by a transition search).
//! scripts/oracle/probe.cyr prints the same lines from the Cyrius port, built from
//! the same inputs. compare.py diffs the two.

use ranga::blend::{blend_pixel, BlendMode};
use ranga::color::*;
use ranga::convert;
use ranga::filter;
use ranga::pixel::{PixelBuffer, PixelFormat};
use ranga::transform::{self, Affine, ScaleFilter};
use ranga::RangaError;
use std::io::{BufWriter, Write};

struct Out(BufWriter<std::io::Stdout>);
impl Out {
    fn line(&mut self, g: &str, k: &str, t: &str, v: &str) {
        writeln!(self.0, "{g}\t{k}\t{t}\t{v}").unwrap();
    }
    fn u8s(&mut self, g: &str, k: &str, d: &[u8]) {
        let s: String = d.iter().map(|b| format!("{b:02x}")).collect();
        self.line(g, k, "u8", &s);
    }
    fn f32(&mut self, g: &str, k: &str, v: f32) {
        self.line(g, k, "f32", &format!("{:08x}", v.to_bits()));
    }
    fn f64(&mut self, g: &str, k: &str, v: f64) {
        self.line(g, k, "f64", &format!("{:016x}", v.to_bits()));
    }
    fn code<T>(&mut self, g: &str, k: &str, r: &Result<T, RangaError>) {
        let c = match r {
            Ok(_) => 0,
            Err(RangaError::InvalidFormat(_)) => -1,
            Err(RangaError::DimensionMismatch { .. }) => -2,
            Err(RangaError::BufferTooSmall { .. }) => -3,
            Err(RangaError::UnsupportedConversion { .. }) => -4,
            Err(RangaError::Other(_)) => -8,
            Err(_) => -99, // RangaError is #[non_exhaustive]
        };
        self.line(g, k, "code", &c.to_string());
    }
}

/// xorshift64. probe.cyr runs the identical generator, so both sides see the same
/// inputs bit for bit.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn byte(&mut self) -> u8 { (self.next() >> 56) as u8 }
    /// Uniform in [0, 1): 24 random bits over 2^24, exact in f32.
    fn f32(&mut self) -> f32 { (self.next() >> 40) as f32 / 16_777_216.0 }
    /// Uniform in [0, 1): 53 random bits over 2^53, exact in f64.
    fn f64(&mut self) -> f64 { (self.next() >> 11) as f64 / 9_007_199_254_740_992.0 }
}

fn fb(u: u32) -> f32 { f32::from_bits(u) }

/// For k in 1..=255, the smallest f32 bit pattern x in [0, 1.0] with f(x) >= k
/// (rising) or f(x) <= 255 - k (falling). For a function monotone in x, these 255
/// boundaries determine its output on EVERY f32 in [0, 1], which is what makes a
/// half-step rounding difference visible without evaluating 10^9 inputs.
fn transitions(o: &mut Out, g: &str, rising: bool, f: &dyn Fn(f32) -> u8) {
    let (lo0, hi0) = (0u32, 0x3F80_0000u32);
    for k in 1u32..=255 {
        let hit = |x: u32| {
            let v = f(fb(x)) as u32;
            if rising { v >= k } else { v <= 255 - k }
        };
        if !hit(hi0) {
            o.line(g, &k.to_string(), "x", "none");
            continue;
        }
        let (mut lo, mut hi) = (lo0, hi0);
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            if hit(mid) { hi = mid; } else { lo = mid + 1; }
        }
        o.line(g, &k.to_string(), "x", &format!("{lo:08x}"));
    }
}

fn rgba(w: u32, h: u32, d: Vec<u8>) -> PixelBuffer {
    PixelBuffer::new(d, w, h, PixelFormat::Rgba8).unwrap()
}

fn rand_img(seed: u64, w: u32, h: u32) -> PixelBuffer {
    let mut r = Rng(seed);
    rgba(w, h, (0..w * h * 4).map(|_| r.byte()).collect())
}

/// A smooth 20x20 gradient: geometry errors show up as moved edges.
fn grad_img() -> PixelBuffer {
    let mut d = Vec::new();
    for y in 0..20u32 {
        for x in 0..20u32 {
            d.extend_from_slice(&[(x * 12) as u8, (y * 12) as u8, ((x + y) * 6) as u8, 255]);
        }
    }
    rgba(20, 20, d)
}

const MODES: [BlendMode; 12] = [
    BlendMode::Normal, BlendMode::Multiply, BlendMode::Screen, BlendMode::Overlay,
    BlendMode::Darken, BlendMode::Lighten, BlendMode::ColorDodge, BlendMode::ColorBurn,
    BlendMode::SoftLight, BlendMode::HardLight, BlendMode::Difference, BlendMode::Exclusion,
];

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let want = |g: &str| args.is_empty() || args.iter().any(|a| a == g);
    let mut o = Out(BufWriter::new(std::io::stdout()));

    if want("srgb_to_linear") {
        for c in 0..=255u8 { o.f32("srgb_to_linear", &c.to_string(), srgb_to_linear(c)); }
    }
    if want("linear_to_srgb") {
        transitions(&mut o, "linear_to_srgb", true, &|x| linear_to_srgb(x));
    }
    if want("lin_to_srgba") {
        transitions(&mut o, "lin_to_srgba.a", true,
            &|x| Srgba::from(LinRgba { r: 0.0, g: 0.0, b: 0.0, a: x }).a);
        transitions(&mut o, "lin_to_srgba.r", true,
            &|x| Srgba::from(LinRgba { r: x, g: 0.0, b: 0.0, a: 1.0 }).r);
    }
    if want("rgbaf32_to_rgba8") {
        transitions(&mut o, "rgbaf32_to_rgba8", true, &|x| {
            let mut d = Vec::new();
            for _ in 0..4 { d.extend_from_slice(&x.to_ne_bytes()); }
            let b = PixelBuffer::new(d, 1, 1, PixelFormat::RgbaF32).unwrap();
            convert::rgbaf32_to_rgba8(&b).unwrap().data()[0]
        });
    }
    if want("hsl") {
        transitions(&mut o, "hsl_to_srgba.grey", true,
            &|x| Srgba::from(Hsl { h: 0.0, s: 0.0, l: x }).r);
        for r in (0..256u32).step_by(5) {
            for g in (0..256u32).step_by(5) {
                for b in (0..256u32).step_by(5) {
                    let k = format!("{r},{g},{b}");
                    let h: Hsl = Srgba { r: r as u8, g: g as u8, b: b as u8, a: 255 }.into();
                    o.f32("srgba_to_hsl.h", &k, h.h);
                    o.f32("srgba_to_hsl.s", &k, h.s);
                    o.f32("srgba_to_hsl.l", &k, h.l);
                    let back: Srgba = h.into();
                    o.u8s("hsl_roundtrip", &k, &[back.r, back.g, back.b]);
                }
            }
        }
        let mut rn = Rng(0x9E37_79B9_7F4A_7C15);
        for i in 0..20000 {
            let h = Hsl { h: rn.f32() * 360.0, s: rn.f32(), l: rn.f32() };
            let s: Srgba = h.into();
            o.u8s("hsl_to_srgba", &i.to_string(), &[s.r, s.g, s.b]);
        }
        // The achromatic threshold: Rust compares `s < 1e-6` in f32.
        for (i, s) in [0x358637bdu32, 0x358637be, 0x358637bc, 0x35800000, 0x36000000].iter().enumerate() {
            let c: Srgba = Hsl { h: 0.0, s: fb(*s), l: 0.5 }.into();
            o.u8s("hsl_to_srgba.eps", &i.to_string(), &[c.r, c.g, c.b]);
        }
    }
    if want("cmyk") {
        transitions(&mut o, "cmyk_to_srgba.k", false,
            &|x| cmyk_to_srgb(&Cmyk { c: 0.0, m: 0.0, y: 0.0, k: x }).r);
        let mut rn = Rng(0xD1B5_4A32_D192_ED03);
        for i in 0..20000 {
            let c = Cmyk { c: rn.f32(), m: rn.f32(), y: rn.f32(), k: rn.f32() };
            let s = cmyk_to_srgb(&c);
            o.u8s("cmyk_to_srgba", &i.to_string(), &[s.r, s.g, s.b]);
        }
        for m in 0..256u32 {
            for x in 0..=m {
                let k = format!("{x},{m}");
                let c = srgb_to_cmyk(&Srgba { r: x as u8, g: m as u8, b: 0, a: 255 });
                o.f32("srgba_to_cmyk.c", &k, c.c);
                o.f32("srgba_to_cmyk.m", &k, c.m);
                o.f32("srgba_to_cmyk.y", &k, c.y);
                o.f32("srgba_to_cmyk.k", &k, c.k);
            }
        }
    }
    if want("oklch") {
        for r in (0..256u32).step_by(5) {
            for g in (0..256u32).step_by(5) {
                for b in (0..256u32).step_by(5) {
                    let k = format!("{r},{g},{b}");
                    let lab: Oklab = Srgba { r: r as u8, g: g as u8, b: b as u8, a: 255 }.into();
                    let lch: Oklch = lab.into();
                    o.f32("oklab_to_oklch.c", &k, lch.c);
                    o.f32("oklab_to_oklch.h", &k, lch.h);
                    let back: Oklab = lch.into();
                    o.f32("oklch_to_oklab.a", &k, back.a);
                    o.f32("oklch_to_oklab.b", &k, back.b);
                }
            }
        }
    }
    if want("color_temperature") {
        for k in (1000..=40000).step_by(7) {
            let t = color_temperature(k as f32);
            for (i, v) in t.iter().enumerate() {
                o.f32("color_temperature", &format!("{k}.{i}"), *v);
            }
        }
    }
    if want("levels") {
        let ramp: Vec<u8> = (0..256u32).flat_map(|v| [v as u8, v as u8, v as u8, 255]).collect();
        for bk in [0u32, 1, 2, 5, 10, 25] {
            for wk in [50u32, 60, 75, 80, 90, 99, 100] {
                for gk in [25u32, 50, 80, 100, 150, 220, 281, 400] {
                    let mut b = rgba(256, 1, ramp.clone());
                    let r = filter::levels(&mut b, bk as f32 / 100.0, wk as f32 / 100.0, gk as f32 / 100.0);
                    let key = format!("{bk},{wk},{gk}");
                    o.code("levels.code", &key, &r);
                    let lut: Vec<u8> = b.data().chunks(4).map(|p| p[0]).collect();
                    o.u8s("levels", &key, &lut);
                }
            }
        }
        // NaN parameters: Rust guards only `gamma <= 0.0`, so each of these proceeds.
        for (key, bk, wk, gk) in [("nan_gamma", 0.0f32, 1.0f32, f32::NAN),
                                  ("nan_black", f32::NAN, 1.0, 1.0),
                                  ("nan_white", 0.0, f32::NAN, 1.0)] {
            let mut b = rgba(256, 1, ramp.clone());
            let r = filter::levels(&mut b, bk, wk, gk);
            o.code("levels.code", key, &r);
            let lut: Vec<u8> = b.data().chunks(4).map(|p| p[0]).collect();
            o.u8s("levels", key, &lut);
        }
    }
    if want("hue_shift") {
        for (i, deg) in [0.0f32, 10.0, 45.0, -90.0, 120.0, 360.0, 725.5, 1.0e9, -1.0e9].iter().enumerate() {
            let mut b = rand_img(0xA5A5_5A5A_1234_5678, 64, 64);
            let r = filter::hue_shift(&mut b, *deg);
            o.code("hue_shift.code", &i.to_string(), &r);
            o.u8s("hue_shift", &i.to_string(), b.data());
        }
    }
    if want("bilateral") {
        let img = rand_img(6, 23, 17);
        for radius in [1u32, 3] {
            for sc in [5.0f32, 10.0, 30.0] {
                let k = format!("{radius},{sc}");
                let r = filter::bilateral(&img, radius, 2.5, sc);
                o.code("bilateral.code", &k, &r);
                if let Ok(out) = r { o.u8s("bilateral", &k, out.data()); }
            }
        }
        for (k, ss) in [("ss0", 0.0f32), ("ssneg", -1.0), ("ssnan", f32::NAN)] {
            let r = filter::bilateral(&img, 2, ss, 30.0);
            o.code("bilateral.code", k, &r);
            if let Ok(out) = r { o.u8s("bilateral", k, out.data()); }
        }
    }
    if want("affine") {
        let mut affs: Vec<Affine> = vec![
            Affine { a: 0.6, b: 0.0, c: 0.0, d: 2.5, tx: -1.9, ty: -1.9 },
            Affine { a: 0.8, b: 0.0, c: 0.0, d: 0.4, tx: -1.9, ty: 0.75 },
            Affine { a: 2.0, b: 0.0, c: 0.0, d: 2.0, tx: 0.25, ty: 0.25 },
            Affine { a: 1.1, b: 0.0, c: 0.0, d: 1.1, tx: 0.7, ty: 0.3 },
            Affine::rotate(f64::from_bits(0x3FF921FB54442D18)),
            Affine::rotate(0.5),
        ];
        let mut rn = Rng(0x0123_4567_89AB_CDEF);
        for _ in 0..200 {
            let a = 0.2 + 3.0 * rn.f64();
            let d = 0.2 + 3.0 * rn.f64();
            let b = rn.f64() - 0.5;
            let c = rn.f64() - 0.5;
            let tx = 20.0 * rn.f64() - 10.0;
            let ty = 20.0 * rn.f64() - 10.0;
            affs.push(Affine { a, b, c, d, tx, ty });
        }
        let img = grad_img();
        for (i, a) in affs.iter().enumerate() {
            let k = i.to_string();
            match a.inverse() {
                Some(inv) => {
                    for (f, v) in [("a", inv.a), ("b", inv.b), ("c", inv.c), ("d", inv.d), ("tx", inv.tx), ("ty", inv.ty)] {
                        o.f64(&format!("affine_inverse.{f}"), &k, v);
                    }
                }
                None => o.line("affine_inverse.a", &k, "x", "singular"),
            }
            if i < 40 {
                for (fi, fl) in [ScaleFilter::Nearest, ScaleFilter::Bilinear, ScaleFilter::Bicubic].iter().enumerate() {
                    let r = transform::affine_transform(&img, a, 30, 30, *fl);
                    let key = format!("{i}.{fi}");
                    o.code("affine_transform.code", &key, &r);
                    if let Ok(out) = r { o.u8s("affine_transform", &key, out.data()); }
                }
            }
        }
        // Non-finite matrices pass the determinant check (NaN is not < eps), so
        // sampling runs on NaN or infinite coordinates. Rust's `as usize` casts
        // saturate there, which keeps every read in bounds.
        let tiny = rgba(1, 1, vec![10, 20, 30, 40]);
        let nonfinite = [
            ("nan_a", Affine { a: f64::NAN, ..Affine::IDENTITY }),
            ("nan_tx", Affine { tx: f64::NAN, ..Affine::IDENTITY }),
            ("inf_d", Affine { d: f64::INFINITY, ..Affine::IDENTITY }),
        ];
        for (name, a) in nonfinite {
            for (fi, fl) in [ScaleFilter::Nearest, ScaleFilter::Bilinear, ScaleFilter::Bicubic].iter().enumerate() {
                for (iname, im) in [("grad", &img), ("1x1", &tiny)] {
                    let r = transform::affine_transform(im, &a, 3, 2, *fl);
                    let key = format!("{name}.{iname}.{fi}");
                    o.code("affine_nonfinite.code", &key, &r);
                    if let Ok(out) = r { o.u8s("affine_nonfinite", &key, out.data()); }
                }
            }
        }
        let id = Affine::IDENTITY;
        let cases = [
            ("identity", id),
            ("nan_a", Affine { a: f64::NAN, ..id }),
            ("nan_tx", Affine { tx: f64::NAN, ..id }),
            ("tiny", Affine { tx: 1e-10, ..id }),
            ("small", Affine { tx: 1e-8, ..id }),
        ];
        for (k, a) in cases {
            o.line("affine_is_identity", k, "u8", &format!("{:02x}", a.is_identity() as u8));
        }
    }
    if want("blend") {
        for (mi, mode) in MODES.iter().enumerate() {
            for s in 0..256u32 {
                for d in 0..256u32 {
                    let src = [s as u8, (255 - s) as u8, (s * 7) as u8, 200];
                    let dst = [d as u8, (d * 3) as u8, (255 - d) as u8, 255];
                    let out = blend_pixel(src, dst, *mode, 200);
                    o.u8s("blend_pixel", &format!("{mi}.{s}.{d}"), &out);
                }
            }
        }
    }
    o.0.flush().unwrap();
}
