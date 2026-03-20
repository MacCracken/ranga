//! Color spaces and conversions.
//!
//! Provides types and conversions between sRGB, linear RGB, HSL, CIE XYZ,
//! CIE L\*a\*b\*, Display P3, CMYK, and YUV color spaces with proper gamma
//! handling. Includes Delta-E color distance metrics and color temperature.

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Core types
// ---------------------------------------------------------------------------

/// A color in linear RGBA float space (0.0–1.0 per channel).
///
/// Linear space is required for physically correct blending and compositing.
/// Use [`From<Srgba>`] to convert from sRGB byte space.
///
/// # Examples
///
/// ```
/// use ranga::color::{LinRgba, Srgba};
///
/// let srgb = Srgba { r: 128, g: 64, b: 200, a: 255 };
/// let linear: LinRgba = srgb.into();
/// assert!(linear.r > 0.0 && linear.r < 1.0);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LinRgba {
    pub r: f32,
    pub g: f32,
    pub b: f32,
    pub a: f32,
}

/// A color in sRGB byte space (0–255 per channel).
///
/// This is the standard color representation for 8-bit displays and image
/// formats. Convert to [`LinRgba`] for blending operations.
///
/// # Examples
///
/// ```
/// use ranga::color::Srgba;
///
/// let red = Srgba { r: 255, g: 0, b: 0, a: 255 };
/// assert_eq!(red.r, 255);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Srgba {
    pub r: u8,
    pub g: u8,
    pub b: u8,
    pub a: u8,
}

/// A color in HSL space.
///
/// # Examples
///
/// ```
/// use ranga::color::{Hsl, Srgba};
///
/// let red = Srgba { r: 255, g: 0, b: 0, a: 255 };
/// let hsl: Hsl = red.into();
/// assert!((hsl.h - 0.0).abs() < 1.0);
/// assert!((hsl.s - 1.0).abs() < 0.01);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Hsl {
    /// Hue in degrees (0.0–360.0).
    pub h: f32,
    /// Saturation (0.0–1.0).
    pub s: f32,
    /// Lightness (0.0–1.0).
    pub l: f32,
}

/// A color in CIE XYZ space (D65 illuminant).
///
/// XYZ is the intermediate space for converting between sRGB, Display P3,
/// and CIE L\*a\*b\*.
///
/// # Examples
///
/// ```
/// use ranga::color::{CieXyz, LinRgba};
///
/// let white = LinRgba { r: 1.0, g: 1.0, b: 1.0, a: 1.0 };
/// let xyz: CieXyz = white.into();
/// assert!((xyz.y - 1.0).abs() < 0.01);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CieXyz {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

/// A color in CIE L\*a\*b\* space (D65 illuminant).
///
/// Perceptually uniform color space used for Delta-E color difference
/// calculations. L\* is lightness (0–100), a\* is green–red, b\* is blue–yellow.
///
/// # Examples
///
/// ```
/// use ranga::color::{CieLab, Srgba};
///
/// let white = Srgba { r: 255, g: 255, b: 255, a: 255 };
/// let lab: CieLab = white.into();
/// assert!((lab.l - 100.0).abs() < 0.1);
/// assert!(lab.a.abs() < 0.5);
/// assert!(lab.b.abs() < 0.5);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CieLab {
    /// Lightness (0.0–100.0).
    pub l: f64,
    /// Green (−) to red (+) axis.
    pub a: f64,
    /// Blue (−) to yellow (+) axis.
    pub b: f64,
}

/// A color in CMYK space (0.0–1.0 per channel).
///
/// Used for print workflows. Convert to/from sRGB with [`cmyk_to_srgb`] and
/// [`srgb_to_cmyk`]. For ICC-accurate conversion, use the [`crate::icc`] module.
///
/// # Examples
///
/// ```
/// use ranga::color::{Cmyk, cmyk_to_srgb};
///
/// let black = Cmyk { c: 0.0, m: 0.0, y: 0.0, k: 1.0 };
/// let srgb = cmyk_to_srgb(&black);
/// assert_eq!(srgb.r, 0);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Cmyk {
    pub c: f32,
    pub m: f32,
    pub y: f32,
    pub k: f32,
}

/// Color space identifiers.
///
/// # Examples
///
/// ```
/// use ranga::color::ColorSpace;
///
/// let cs = ColorSpace::Srgb;
/// assert_eq!(cs, ColorSpace::Srgb);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ColorSpace {
    Srgb,
    LinearRgb,
    DisplayP3,
    Bt601,
    Bt709,
}

// ---------------------------------------------------------------------------
// sRGB ↔ Linear
// ---------------------------------------------------------------------------

/// Convert a single sRGB component (0–255) to linear (0.0–1.0).
///
/// Applies the sRGB transfer function (gamma ~2.2) for physically correct
/// color math.
///
/// # Examples
///
/// ```
/// use ranga::color::srgb_to_linear;
///
/// assert_eq!(srgb_to_linear(0), 0.0);
/// assert!((srgb_to_linear(255) - 1.0).abs() < 0.001);
/// let mid = srgb_to_linear(128);
/// assert!(mid > 0.2 && mid < 0.3);
/// ```
#[inline]
#[must_use]
pub fn srgb_to_linear(c: u8) -> f32 {
    let s = c as f32 / 255.0;
    if s <= 0.04045 {
        s / 12.92
    } else {
        ((s + 0.055) / 1.055).powf(2.4)
    }
}

/// Convert a single linear component (0.0–1.0) to sRGB (0–255).
///
/// Applies the inverse sRGB transfer function.
///
/// # Examples
///
/// ```
/// use ranga::color::linear_to_srgb;
///
/// assert_eq!(linear_to_srgb(0.0), 0);
/// assert_eq!(linear_to_srgb(1.0), 255);
/// ```
#[inline]
#[must_use]
pub fn linear_to_srgb(c: f32) -> u8 {
    let s = if c <= 0.0031308 {
        c * 12.92
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    };
    (s * 255.0 + 0.5).clamp(0.0, 255.0) as u8
}

impl From<Srgba> for LinRgba {
    fn from(c: Srgba) -> Self {
        Self {
            r: srgb_to_linear(c.r),
            g: srgb_to_linear(c.g),
            b: srgb_to_linear(c.b),
            a: c.a as f32 / 255.0,
        }
    }
}

impl From<LinRgba> for Srgba {
    fn from(c: LinRgba) -> Self {
        Self {
            r: linear_to_srgb(c.r),
            g: linear_to_srgb(c.g),
            b: linear_to_srgb(c.b),
            a: (c.a * 255.0 + 0.5).clamp(0.0, 255.0) as u8,
        }
    }
}

// ---------------------------------------------------------------------------
// RGB ↔ HSL (bidirectional — replaces rasa's inline hsl code)
// ---------------------------------------------------------------------------

impl From<Srgba> for Hsl {
    fn from(c: Srgba) -> Self {
        let r = c.r as f32 / 255.0;
        let g = c.g as f32 / 255.0;
        let b = c.b as f32 / 255.0;

        let max = r.max(g).max(b);
        let min = r.min(g).min(b);
        let d = max - min;
        let l = (max + min) / 2.0;

        if d < 1e-6 {
            return Hsl { h: 0.0, s: 0.0, l };
        }

        let s = if l > 0.5 {
            d / (2.0 - max - min)
        } else {
            d / (max + min)
        };

        let h = if (max - r).abs() < 1e-6 {
            (g - b) / d + if g < b { 6.0 } else { 0.0 }
        } else if (max - g).abs() < 1e-6 {
            (b - r) / d + 2.0
        } else {
            (r - g) / d + 4.0
        };

        Hsl { h: h * 60.0, s, l }
    }
}

/// Helper for HSL-to-RGB conversion.
#[inline]
fn hue_to_rgb(p: f32, q: f32, mut t: f32) -> f32 {
    if t < 0.0 {
        t += 1.0;
    }
    if t > 1.0 {
        t -= 1.0;
    }
    if t < 1.0 / 6.0 {
        return p + (q - p) * 6.0 * t;
    }
    if t < 0.5 {
        return q;
    }
    if t < 2.0 / 3.0 {
        return p + (q - p) * (2.0 / 3.0 - t) * 6.0;
    }
    p
}

impl From<Hsl> for Srgba {
    fn from(c: Hsl) -> Self {
        if c.s < 1e-6 {
            let v = (c.l * 255.0 + 0.5).clamp(0.0, 255.0) as u8;
            return Srgba {
                r: v,
                g: v,
                b: v,
                a: 255,
            };
        }
        let q = if c.l < 0.5 {
            c.l * (1.0 + c.s)
        } else {
            c.l + c.s - c.l * c.s
        };
        let p = 2.0 * c.l - q;
        let h = c.h / 360.0;
        let r = hue_to_rgb(p, q, h + 1.0 / 3.0);
        let g = hue_to_rgb(p, q, h);
        let b = hue_to_rgb(p, q, h - 1.0 / 3.0);
        Srgba {
            r: (r * 255.0 + 0.5).clamp(0.0, 255.0) as u8,
            g: (g * 255.0 + 0.5).clamp(0.0, 255.0) as u8,
            b: (b * 255.0 + 0.5).clamp(0.0, 255.0) as u8,
            a: 255,
        }
    }
}

// ---------------------------------------------------------------------------
// sRGB ↔ CIE XYZ (D65)
// ---------------------------------------------------------------------------

// sRGB to XYZ matrix (D65 illuminant, IEC 61966-2-1)
const SRGB_TO_XYZ: [[f64; 3]; 3] = [
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041],
];

// XYZ to sRGB matrix (inverse of above)
const XYZ_TO_SRGB: [[f64; 3]; 3] = [
    [3.2404542, -1.5371385, -0.4985314],
    [-0.9692660, 1.8760108, 0.0415560],
    [0.0556434, -0.2040259, 1.0572252],
];

impl From<LinRgba> for CieXyz {
    fn from(c: LinRgba) -> Self {
        let r = c.r as f64;
        let g = c.g as f64;
        let b = c.b as f64;
        CieXyz {
            x: SRGB_TO_XYZ[0][0] * r + SRGB_TO_XYZ[0][1] * g + SRGB_TO_XYZ[0][2] * b,
            y: SRGB_TO_XYZ[1][0] * r + SRGB_TO_XYZ[1][1] * g + SRGB_TO_XYZ[1][2] * b,
            z: SRGB_TO_XYZ[2][0] * r + SRGB_TO_XYZ[2][1] * g + SRGB_TO_XYZ[2][2] * b,
        }
    }
}

impl From<CieXyz> for LinRgba {
    fn from(c: CieXyz) -> Self {
        LinRgba {
            r: (XYZ_TO_SRGB[0][0] * c.x + XYZ_TO_SRGB[0][1] * c.y + XYZ_TO_SRGB[0][2] * c.z) as f32,
            g: (XYZ_TO_SRGB[1][0] * c.x + XYZ_TO_SRGB[1][1] * c.y + XYZ_TO_SRGB[1][2] * c.z) as f32,
            b: (XYZ_TO_SRGB[2][0] * c.x + XYZ_TO_SRGB[2][1] * c.y + XYZ_TO_SRGB[2][2] * c.z) as f32,
            a: 1.0,
        }
    }
}

// ---------------------------------------------------------------------------
// CIE XYZ ↔ CIE L*a*b* (D65)
// ---------------------------------------------------------------------------

// D65 reference white point
const D65_XN: f64 = 0.95047;
const D65_YN: f64 = 1.00000;
const D65_ZN: f64 = 1.08883;

const LAB_EPSILON: f64 = 0.008856; // (6/29)^3
const LAB_KAPPA: f64 = 903.3; // (29/3)^3

#[inline]
fn lab_f(t: f64) -> f64 {
    if t > LAB_EPSILON {
        t.cbrt()
    } else {
        (LAB_KAPPA * t + 16.0) / 116.0
    }
}

#[inline]
fn lab_f_inv(t: f64) -> f64 {
    if t > 6.0 / 29.0 {
        t * t * t
    } else {
        3.0 * (6.0 / 29.0) * (6.0 / 29.0) * (t - 4.0 / 29.0)
    }
}

impl From<CieXyz> for CieLab {
    fn from(c: CieXyz) -> Self {
        let fx = lab_f(c.x / D65_XN);
        let fy = lab_f(c.y / D65_YN);
        let fz = lab_f(c.z / D65_ZN);
        CieLab {
            l: 116.0 * fy - 16.0,
            a: 500.0 * (fx - fy),
            b: 200.0 * (fy - fz),
        }
    }
}

impl From<CieLab> for CieXyz {
    fn from(c: CieLab) -> Self {
        let fy = (c.l + 16.0) / 116.0;
        let fx = c.a / 500.0 + fy;
        let fz = fy - c.b / 200.0;
        CieXyz {
            x: D65_XN * lab_f_inv(fx),
            y: D65_YN * lab_f_inv(fy),
            z: D65_ZN * lab_f_inv(fz),
        }
    }
}

/// Convenience: sRGB byte → Lab in one step.
impl From<Srgba> for CieLab {
    fn from(c: Srgba) -> Self {
        let lin: LinRgba = c.into();
        let xyz: CieXyz = lin.into();
        xyz.into()
    }
}

// ---------------------------------------------------------------------------
// Display P3 ↔ sRGB (both linear space)
// ---------------------------------------------------------------------------

// P3 linear → sRGB linear (via XYZ)
const P3_TO_SRGB: [[f64; 3]; 3] = [
    [1.2249401, -0.2249402, 0.0000001],
    [-0.0420569, 1.0420571, -0.0000002],
    [-0.0196376, -0.0786361, 1.0982735],
];

// sRGB linear → P3 linear
const SRGB_TO_P3: [[f64; 3]; 3] = [
    [0.8224622, 0.1775380, -0.0000002],
    [0.0331942, 0.9668058, 0.0000000],
    [0.0170608, 0.0723740, 0.9105650],
];

/// Convert a color from Display P3 linear space to sRGB linear space.
///
/// Both input and output are in linear (gamma 1.0) space. P3 and sRGB share
/// the same transfer function, so apply [`srgb_to_linear`]/[`linear_to_srgb`]
/// for encoding.
///
/// # Examples
///
/// ```
/// use ranga::color::p3_to_linear_srgb;
///
/// let (r, g, b) = p3_to_linear_srgb(1.0, 0.0, 0.0);
/// assert!(r > 1.0); // P3 red is outside sRGB gamut
/// ```
#[must_use]
pub fn p3_to_linear_srgb(r: f64, g: f64, b: f64) -> (f64, f64, f64) {
    (
        P3_TO_SRGB[0][0] * r + P3_TO_SRGB[0][1] * g + P3_TO_SRGB[0][2] * b,
        P3_TO_SRGB[1][0] * r + P3_TO_SRGB[1][1] * g + P3_TO_SRGB[1][2] * b,
        P3_TO_SRGB[2][0] * r + P3_TO_SRGB[2][1] * g + P3_TO_SRGB[2][2] * b,
    )
}

/// Convert a color from sRGB linear space to Display P3 linear space.
///
/// # Examples
///
/// ```
/// use ranga::color::linear_srgb_to_p3;
///
/// let (r, g, b) = linear_srgb_to_p3(1.0, 0.0, 0.0);
/// assert!(r < 1.0); // sRGB red fits within P3
/// ```
#[must_use]
pub fn linear_srgb_to_p3(r: f64, g: f64, b: f64) -> (f64, f64, f64) {
    (
        SRGB_TO_P3[0][0] * r + SRGB_TO_P3[0][1] * g + SRGB_TO_P3[0][2] * b,
        SRGB_TO_P3[1][0] * r + SRGB_TO_P3[1][1] * g + SRGB_TO_P3[1][2] * b,
        SRGB_TO_P3[2][0] * r + SRGB_TO_P3[2][1] * g + SRGB_TO_P3[2][2] * b,
    )
}

// ---------------------------------------------------------------------------
// CMYK ↔ sRGB (naive — for ICC-accurate, use crate::icc)
// ---------------------------------------------------------------------------

/// Convert CMYK to sRGB (naive, without ICC profile).
///
/// For print-accurate conversion, use an ICC profile via [`crate::icc`].
///
/// # Examples
///
/// ```
/// use ranga::color::{Cmyk, cmyk_to_srgb};
///
/// let cyan = Cmyk { c: 1.0, m: 0.0, y: 0.0, k: 0.0 };
/// let srgb = cmyk_to_srgb(&cyan);
/// assert_eq!(srgb.r, 0);
/// assert_eq!(srgb.g, 255);
/// assert_eq!(srgb.b, 255);
/// ```
#[must_use]
pub fn cmyk_to_srgb(c: &Cmyk) -> Srgba {
    let r = (255.0 * (1.0 - c.c) * (1.0 - c.k) + 0.5).clamp(0.0, 255.0) as u8;
    let g = (255.0 * (1.0 - c.m) * (1.0 - c.k) + 0.5).clamp(0.0, 255.0) as u8;
    let b = (255.0 * (1.0 - c.y) * (1.0 - c.k) + 0.5).clamp(0.0, 255.0) as u8;
    Srgba { r, g, b, a: 255 }
}

/// Convert sRGB to CMYK (naive, without ICC profile).
///
/// # Examples
///
/// ```
/// use ranga::color::{Srgba, srgb_to_cmyk};
///
/// let white = Srgba { r: 255, g: 255, b: 255, a: 255 };
/// let cmyk = srgb_to_cmyk(&white);
/// assert!((cmyk.k - 0.0).abs() < 0.01);
/// ```
#[must_use]
pub fn srgb_to_cmyk(c: &Srgba) -> Cmyk {
    let r = c.r as f32 / 255.0;
    let g = c.g as f32 / 255.0;
    let b = c.b as f32 / 255.0;
    let k = 1.0 - r.max(g).max(b);
    if k >= 1.0 - 1e-6 {
        return Cmyk {
            c: 0.0,
            m: 0.0,
            y: 0.0,
            k: 1.0,
        };
    }
    let inv_k = 1.0 / (1.0 - k);
    Cmyk {
        c: (1.0 - r - k) * inv_k,
        m: (1.0 - g - k) * inv_k,
        y: (1.0 - b - k) * inv_k,
        k,
    }
}

// ---------------------------------------------------------------------------
// Color temperature (Tanner Helland approximation of Planckian locus)
// ---------------------------------------------------------------------------

/// Compute RGB multipliers for a given color temperature in Kelvin.
///
/// Returns `[R, G, B]` multipliers in 0.0–1.0 range. Multiply with pixel
/// values to apply a white-balance shift. Valid range is 1000–40000 K.
/// 6600 K is approximately daylight (neutral).
///
/// Matches the temperature adjustment used in tazama's color grading pipeline.
///
/// # Examples
///
/// ```
/// use ranga::color::color_temperature;
///
/// let daylight = color_temperature(6600.0);
/// assert!((daylight[0] - 1.0).abs() < 0.01); // neutral at ~6600K
///
/// let warm = color_temperature(3000.0);
/// assert!(warm[2] < warm[0]); // warm = more red, less blue
/// ```
#[must_use]
pub fn color_temperature(kelvin: f32) -> [f32; 3] {
    let temp = kelvin.clamp(1000.0, 40000.0) / 100.0;

    let r = if temp <= 66.0 {
        1.0
    } else {
        (329.699 * (temp - 60.0).powf(-0.133_205) / 255.0).clamp(0.0, 1.0)
    };

    let g = if temp <= 66.0 {
        ((99.4708 * temp.ln() - 161.1196) / 255.0).clamp(0.0, 1.0)
    } else {
        (288.1222 * (temp - 60.0).powf(-0.075_515) / 255.0).clamp(0.0, 1.0)
    };

    let b = if temp >= 66.0 {
        1.0
    } else if temp <= 19.0 {
        0.0
    } else {
        ((138.5177 * (temp - 10.0).ln() - 305.0448) / 255.0).clamp(0.0, 1.0)
    };

    [r, g, b]
}

// ---------------------------------------------------------------------------
// Delta-E color distance (CIE76, CIE94, CIEDE2000)
// ---------------------------------------------------------------------------

/// CIE76 color distance (Euclidean distance in L\*a\*b\* space).
///
/// Simple and fast but not perceptually uniform for large differences.
///
/// # Examples
///
/// ```
/// use ranga::color::{CieLab, delta_e_cie76};
///
/// let a = CieLab { l: 50.0, a: 0.0, b: 0.0 };
/// let b = CieLab { l: 50.0, a: 0.0, b: 0.0 };
/// assert!(delta_e_cie76(&a, &b) < 1e-10);
/// ```
#[must_use]
pub fn delta_e_cie76(a: &CieLab, b: &CieLab) -> f64 {
    let dl = a.l - b.l;
    let da = a.a - b.a;
    let db = a.b - b.b;
    (dl * dl + da * da + db * db).sqrt()
}

/// CIE94 color distance (graphic arts weighting).
///
/// More perceptually uniform than CIE76, especially for chromatic differences.
///
/// # Examples
///
/// ```
/// use ranga::color::{CieLab, delta_e_cie94};
///
/// let a = CieLab { l: 50.0, a: 25.0, b: 0.0 };
/// let b = CieLab { l: 50.0, a: 0.0, b: 0.0 };
/// assert!(delta_e_cie94(&a, &b) > 0.0);
/// ```
#[must_use]
pub fn delta_e_cie94(a: &CieLab, b: &CieLab) -> f64 {
    let dl = a.l - b.l;
    let da = a.a - b.a;
    let db = a.b - b.b;
    let c1 = (a.a * a.a + a.b * a.b).sqrt();
    let c2 = (b.a * b.a + b.b * b.b).sqrt();
    let dc = c1 - c2;
    let dh_sq = da * da + db * db - dc * dc;
    let dh_sq = dh_sq.max(0.0); // guard against floating-point negatives

    // Graphic arts constants
    let sl = 1.0;
    let sc = 1.0 + 0.045 * c1;
    let sh = 1.0 + 0.015 * c1;

    ((dl / sl).powi(2) + (dc / sc).powi(2) + dh_sq / (sh * sh)).sqrt()
}

/// CIEDE2000 color distance — the most perceptually accurate Delta-E formula.
///
/// Implements the full Sharma et al. (2005) specification with lightness,
/// chroma, and hue weighting plus the rotation term.
///
/// # Examples
///
/// ```
/// use ranga::color::{CieLab, delta_e_ciede2000};
///
/// let a = CieLab { l: 50.0, a: 2.6772, b: -79.7751 };
/// let b = CieLab { l: 50.0, a: 0.0, b: -82.7485 };
/// let de = delta_e_ciede2000(&a, &b);
/// assert!(de > 2.0 && de < 3.0);
/// ```
#[must_use]
pub fn delta_e_ciede2000(lab1: &CieLab, lab2: &CieLab) -> f64 {
    use std::f64::consts::PI;

    let c1_star = (lab1.a * lab1.a + lab1.b * lab1.b).sqrt();
    let c2_star = (lab2.a * lab2.a + lab2.b * lab2.b).sqrt();
    let c_bar = (c1_star + c2_star) / 2.0;

    let c_bar_7 = c_bar.powi(7);
    let g = 0.5 * (1.0 - (c_bar_7 / (c_bar_7 + 6103515625.0_f64)).sqrt()); // 25^7

    let a1_prime = lab1.a * (1.0 + g);
    let a2_prime = lab2.a * (1.0 + g);

    let c1_prime = (a1_prime * a1_prime + lab1.b * lab1.b).sqrt();
    let c2_prime = (a2_prime * a2_prime + lab2.b * lab2.b).sqrt();

    let h1_prime = lab1.b.atan2(a1_prime).to_degrees().rem_euclid(360.0);
    let h2_prime = lab2.b.atan2(a2_prime).to_degrees().rem_euclid(360.0);

    // Step 2: ΔL', ΔC', ΔH'
    let dl_prime = lab2.l - lab1.l;
    let dc_prime = c2_prime - c1_prime;

    let dh_prime_deg = if c1_prime * c2_prime < 1e-10 {
        0.0
    } else if (h2_prime - h1_prime).abs() <= 180.0 {
        h2_prime - h1_prime
    } else if h2_prime - h1_prime > 180.0 {
        h2_prime - h1_prime - 360.0
    } else {
        h2_prime - h1_prime + 360.0
    };

    let dh_prime = 2.0 * (c1_prime * c2_prime).sqrt() * (dh_prime_deg / 2.0 * PI / 180.0).sin();

    // Step 3: Weighting functions
    let l_bar_prime = (lab1.l + lab2.l) / 2.0;
    let c_bar_prime = (c1_prime + c2_prime) / 2.0;

    let h_bar_prime = if c1_prime * c2_prime < 1e-10 {
        h1_prime + h2_prime
    } else if (h1_prime - h2_prime).abs() <= 180.0 {
        (h1_prime + h2_prime) / 2.0
    } else if h1_prime + h2_prime < 360.0 {
        (h1_prime + h2_prime + 360.0) / 2.0
    } else {
        (h1_prime + h2_prime - 360.0) / 2.0
    };

    let t = 1.0 - 0.17 * ((h_bar_prime - 30.0) * PI / 180.0).cos()
        + 0.24 * ((2.0 * h_bar_prime) * PI / 180.0).cos()
        + 0.32 * ((3.0 * h_bar_prime + 6.0) * PI / 180.0).cos()
        - 0.20 * ((4.0 * h_bar_prime - 63.0) * PI / 180.0).cos();

    let l_diff = l_bar_prime - 50.0;
    let sl = 1.0 + 0.015 * l_diff * l_diff / (20.0 + l_diff * l_diff).sqrt();
    let sc = 1.0 + 0.045 * c_bar_prime;
    let sh = 1.0 + 0.015 * c_bar_prime * t;

    let d_theta = 30.0 * (-((h_bar_prime - 275.0) / 25.0).powi(2)).exp();
    let c_bar_prime_7 = c_bar_prime.powi(7);
    let rc = 2.0 * (c_bar_prime_7 / (c_bar_prime_7 + 6103515625.0_f64)).sqrt();
    let rt = -(2.0 * d_theta * PI / 180.0).sin() * rc;

    let l_term = dl_prime / sl;
    let c_term = dc_prime / sc;
    let h_term = dh_prime / sh;

    (l_term * l_term + c_term * c_term + h_term * h_term + rt * c_term * h_term).sqrt()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn srgb_linear_roundtrip() {
        for v in [0u8, 1, 50, 128, 200, 255] {
            let lin = srgb_to_linear(v);
            let back = linear_to_srgb(lin);
            assert!(
                (v as i16 - back as i16).unsigned_abs() <= 1,
                "v={v} back={back}"
            );
        }
    }

    #[test]
    fn srgba_to_linrgba_black() {
        let c: LinRgba = Srgba {
            r: 0,
            g: 0,
            b: 0,
            a: 255,
        }
        .into();
        assert_eq!(c.r, 0.0);
        assert_eq!(c.g, 0.0);
        assert_eq!(c.b, 0.0);
        assert!((c.a - 1.0).abs() < 1e-3);
    }

    #[test]
    fn srgba_to_linrgba_white() {
        let c: LinRgba = Srgba {
            r: 255,
            g: 255,
            b: 255,
            a: 255,
        }
        .into();
        assert!((c.r - 1.0).abs() < 1e-3);
    }

    #[test]
    fn hsl_from_red() {
        let hsl: Hsl = Srgba {
            r: 255,
            g: 0,
            b: 0,
            a: 255,
        }
        .into();
        assert!((hsl.h - 0.0).abs() < 1.0);
        assert!((hsl.s - 1.0).abs() < 1e-3);
        assert!((hsl.l - 0.5).abs() < 1e-3);
    }

    #[test]
    fn hsl_from_gray() {
        let hsl: Hsl = Srgba {
            r: 128,
            g: 128,
            b: 128,
            a: 255,
        }
        .into();
        assert!((hsl.s - 0.0).abs() < 1e-3);
    }

    #[test]
    fn hsl_roundtrip() {
        for (r, g, b) in [(255, 0, 0), (0, 255, 0), (0, 0, 255), (128, 64, 200)] {
            let orig = Srgba { r, g, b, a: 255 };
            let hsl: Hsl = orig.into();
            let back: Srgba = hsl.into();
            assert!((orig.r as i16 - back.r as i16).unsigned_abs() <= 1, "r");
            assert!((orig.g as i16 - back.g as i16).unsigned_abs() <= 1, "g");
            assert!((orig.b as i16 - back.b as i16).unsigned_abs() <= 1, "b");
        }
    }

    #[test]
    fn hsl_gray_roundtrip() {
        let orig = Srgba {
            r: 128,
            g: 128,
            b: 128,
            a: 255,
        };
        let hsl: Hsl = orig.into();
        let back: Srgba = hsl.into();
        assert_eq!(back.r, back.g);
        assert_eq!(back.g, back.b);
    }

    #[test]
    fn xyz_white_y_is_one() {
        let white = LinRgba {
            r: 1.0,
            g: 1.0,
            b: 1.0,
            a: 1.0,
        };
        let xyz: CieXyz = white.into();
        assert!((xyz.y - 1.0).abs() < 0.01);
    }

    #[test]
    fn xyz_roundtrip() {
        let orig = LinRgba {
            r: 0.5,
            g: 0.3,
            b: 0.8,
            a: 1.0,
        };
        let xyz: CieXyz = orig.into();
        let back: LinRgba = xyz.into();
        // f64→f32→f64 roundtrip loses precision
        assert!((orig.r - back.r).abs() < 1e-3);
        assert!((orig.g - back.g).abs() < 1e-3);
        assert!((orig.b - back.b).abs() < 1e-3);
    }

    #[test]
    fn lab_white_is_100() {
        let white: CieLab = Srgba {
            r: 255,
            g: 255,
            b: 255,
            a: 255,
        }
        .into();
        assert!((white.l - 100.0).abs() < 0.5);
        assert!(white.a.abs() < 1.0);
        // b* offset is expected: sRGB matrix Z-row sums to 1.0288, not D65 Zn=1.0888
        assert!(white.b.abs() < 5.0);
    }

    #[test]
    fn lab_black_is_zero() {
        let black: CieLab = Srgba {
            r: 0,
            g: 0,
            b: 0,
            a: 255,
        }
        .into();
        assert!(black.l.abs() < 0.5);
    }

    #[test]
    fn lab_roundtrip() {
        let orig = Srgba {
            r: 128,
            g: 64,
            b: 200,
            a: 255,
        };
        let lab: CieLab = orig.into();
        let xyz: CieXyz = lab.into();
        let lin: LinRgba = xyz.into();
        let back: Srgba = lin.into();
        assert!((orig.r as i16 - back.r as i16).unsigned_abs() <= 1);
        assert!((orig.g as i16 - back.g as i16).unsigned_abs() <= 1);
        assert!((orig.b as i16 - back.b as i16).unsigned_abs() <= 1);
    }

    #[test]
    fn p3_srgb_roundtrip() {
        let (r, g, b) = linear_srgb_to_p3(0.5, 0.3, 0.8);
        let (r2, g2, b2) = p3_to_linear_srgb(r, g, b);
        assert!((0.5 - r2).abs() < 1e-4);
        assert!((0.3 - g2).abs() < 1e-4);
        assert!((0.8 - b2).abs() < 1e-4);
    }

    #[test]
    fn p3_red_outside_srgb() {
        let (r, _, _) = p3_to_linear_srgb(1.0, 0.0, 0.0);
        assert!(r > 1.0, "P3 red should exceed sRGB gamut");
    }

    #[test]
    fn cmyk_roundtrip() {
        let orig = Srgba {
            r: 200,
            g: 100,
            b: 50,
            a: 255,
        };
        let cmyk = srgb_to_cmyk(&orig);
        let back = cmyk_to_srgb(&cmyk);
        assert!((orig.r as i16 - back.r as i16).unsigned_abs() <= 1);
        assert!((orig.g as i16 - back.g as i16).unsigned_abs() <= 1);
        assert!((orig.b as i16 - back.b as i16).unsigned_abs() <= 1);
    }

    #[test]
    fn cmyk_black() {
        let cmyk = srgb_to_cmyk(&Srgba {
            r: 0,
            g: 0,
            b: 0,
            a: 255,
        });
        assert!((cmyk.k - 1.0).abs() < 0.01);
    }

    #[test]
    fn temperature_daylight_neutral() {
        let d = color_temperature(6600.0);
        assert!((d[0] - 1.0).abs() < 0.02);
    }

    #[test]
    fn temperature_warm_has_more_red() {
        let w = color_temperature(3000.0);
        assert!(w[0] > w[2], "warm light should have more red than blue");
    }

    #[test]
    fn temperature_cool_has_more_blue() {
        let c = color_temperature(10000.0);
        assert!(c[2] > c[0], "cool light should have more blue than red");
    }

    #[test]
    fn delta_e_76_identical() {
        let a = CieLab {
            l: 50.0,
            a: 25.0,
            b: -10.0,
        };
        assert!(delta_e_cie76(&a, &a) < 1e-10);
    }

    #[test]
    fn delta_e_76_known() {
        let a = CieLab {
            l: 50.0,
            a: 0.0,
            b: 0.0,
        };
        let b = CieLab {
            l: 53.0,
            a: 4.0,
            b: 0.0,
        };
        let de = delta_e_cie76(&a, &b);
        assert!((de - 5.0).abs() < 0.01); // sqrt(9+16) = 5
    }

    #[test]
    fn delta_e_94_positive() {
        let a = CieLab {
            l: 50.0,
            a: 25.0,
            b: 0.0,
        };
        let b = CieLab {
            l: 50.0,
            a: 0.0,
            b: 0.0,
        };
        assert!(delta_e_cie94(&a, &b) > 0.0);
    }

    #[test]
    fn delta_e_2000_identical() {
        let a = CieLab {
            l: 50.0,
            a: 25.0,
            b: -10.0,
        };
        assert!(delta_e_ciede2000(&a, &a) < 1e-10);
    }

    #[test]
    fn delta_e_2000_known_pair() {
        // Sharma (2005) test pair #1
        let a = CieLab {
            l: 50.0,
            a: 2.6772,
            b: -79.7751,
        };
        let b = CieLab {
            l: 50.0,
            a: 0.0,
            b: -82.7485,
        };
        let de = delta_e_ciede2000(&a, &b);
        assert!((de - 2.0425).abs() < 0.01, "got {de}");
    }
}
