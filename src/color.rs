//! Color spaces and conversions.
//!
//! Provides types and conversions between sRGB, linear RGB, HSL,
//! and YUV color spaces with proper gamma handling.

use serde::{Deserialize, Serialize};

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

// --- sRGB ↔ Linear ---

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

// --- RGB ↔ HSL ---

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
}
